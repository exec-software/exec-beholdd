// Real-weight, batch-1 Qwen3.5/3.6 MoE FFN residency experiment for gfx1151.
//
// This standalone harness consumes the strict v1 layer image emitted by
// `pack_qwen35moe_bf16_layer`. It does not link to, modify, or replace the
// Beholdd runtime or any serving deployment.

#include <stddef.h>

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

using Bf16 = rocwmma::bfloat16_t;

constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kFfn = 512;
constexpr std::uint32_t kAllExperts = 256;
constexpr std::uint32_t kTopK = 8;
constexpr std::uint32_t kSharedSlot = kAllExperts;
constexpr std::uint32_t kComputeSlots = kTopK + 1;
constexpr std::uint32_t kPackedSlots = kAllExperts + 1;
constexpr std::uint32_t kPaddedTokens = 16;
constexpr std::uint32_t kTile = 16;
constexpr std::uint32_t kWaveSize = 32;
constexpr std::uint32_t kElementThreads = 256;
constexpr float kRmsEpsilon = 1.0e-6f;
constexpr std::size_t kHeaderBytes = 4096;
constexpr std::size_t kNormBytes = kHidden * sizeof(float);
constexpr std::size_t kRouterBytes =
    static_cast<std::size_t>(kAllExperts) * kHidden * sizeof(float);
constexpr std::size_t kSharedGateBytes = kHidden * sizeof(float);
constexpr std::size_t kExpertMatrixElements =
    static_cast<std::size_t>(kHidden) * kFfn;
constexpr std::size_t kGateUpElements =
    2ULL * kPackedSlots * kExpertMatrixElements;
constexpr std::size_t kDownElements = kPackedSlots * kExpertMatrixElements;
constexpr std::size_t kGateUpBytes = kGateUpElements * sizeof(Bf16);
constexpr std::size_t kDownBytes = kDownElements * sizeof(Bf16);
constexpr std::size_t kNormOffset = kHeaderBytes;
constexpr std::size_t kRouterOffset = kNormOffset + kNormBytes;
constexpr std::size_t kSharedGateOffset = kRouterOffset + kRouterBytes;
constexpr std::size_t kGateUpOffset = kSharedGateOffset + kSharedGateBytes;
constexpr std::size_t kDownOffset = kGateUpOffset + kGateUpBytes;
constexpr std::size_t kPackBytes = kDownOffset + kDownBytes;
constexpr std::size_t kHeadroomBytes = 512ULL * 1024 * 1024;

static_assert(kHidden % kTile == 0);
static_assert(kFfn % kTile == 0);
static_assert(sizeof(Bf16) == 2);
static_assert(kPackBytes == 1'619'021'824ULL);
static_assert(kNormOffset == 4096);
static_assert(kRouterOffset == 12'288);
static_assert(kSharedGateOffset == 2'109'440);
static_assert(kGateUpOffset == 2'117'632);
static_assert(kDownOffset == 1'080'053'760);

[[noreturn]] void throw_hip_error(hipError_t error,
                                  const char* expression,
                                  const char* file,
                                  int line) {
    throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                             ": " + expression + " failed: " +
                             hipGetErrorName(error) + " (" +
                             hipGetErrorString(error) + ")");
}

#define HIP_CHECK(expression)                                                   \
    do {                                                                        \
        const hipError_t hip_check_error = (expression);                        \
        if (hip_check_error != hipSuccess) {                                    \
            throw_hip_error(hip_check_error, #expression, __FILE__, __LINE__);  \
        }                                                                       \
    } while (false)

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ == 0 || count_ > std::numeric_limits<std::size_t>::max() /
                                        sizeof(T)) {
            throw std::runtime_error("invalid device allocation size");
        }
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_),
                            count_ * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) {
            (void)hipFree(data_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)),
          count_(std::exchange(other.count_, 0)) {}

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) {
                (void)hipFree(data_);
            }
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }

    [[nodiscard]] T* get() const { return data_; }
    [[nodiscard]] std::size_t size_bytes() const {
        return count_ * sizeof(T);
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&data_),
                               count_ * sizeof(T), hipHostMallocDefault));
    }

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            (void)hipHostFree(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    [[nodiscard]] T* get() const { return data_; }
    [[nodiscard]] std::size_t size_bytes() const {
        return count_ * sizeof(T);
    }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
  public:
    Stream() {
        HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking));
    }

    ~Stream() {
        if (stream_ != nullptr) {
            (void)hipStreamDestroy(stream_);
        }
    }

    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    [[nodiscard]] hipStream_t get() const { return stream_; }

  private:
    hipStream_t stream_ = nullptr;
};

class MappedPack {
  public:
    explicit MappedPack(const std::string& path) : path_(path) {
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("cannot open layer pack " + path);
        }
        struct stat metadata {};
        if (fstat(fd_, &metadata) != 0 || metadata.st_size < 0) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot stat layer pack " + path);
        }
        bytes_ = static_cast<std::size_t>(metadata.st_size);
        if (bytes_ != kPackBytes) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("layer pack length is " +
                                     std::to_string(bytes_) +
                                     ", expected " +
                                     std::to_string(kPackBytes));
        }
        void* mapped = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (mapped == MAP_FAILED) {
            close(fd_);
            fd_ = -1;
            throw std::runtime_error("cannot mmap layer pack " + path);
        }
        data_ = static_cast<const std::uint8_t*>(mapped);
        validate_header();
    }

    ~MappedPack() {
        if (data_ != nullptr) {
            (void)munmap(const_cast<std::uint8_t*>(data_), bytes_);
        }
        if (fd_ >= 0) {
            (void)close(fd_);
        }
    }

    MappedPack(const MappedPack&) = delete;
    MappedPack& operator=(const MappedPack&) = delete;

    [[nodiscard]] const float* norm() const {
        return reinterpret_cast<const float*>(data_ + kNormOffset);
    }
    [[nodiscard]] const float* router() const {
        return reinterpret_cast<const float*>(data_ + kRouterOffset);
    }
    [[nodiscard]] const float* shared_gate() const {
        return reinterpret_cast<const float*>(data_ + kSharedGateOffset);
    }
    [[nodiscard]] const Bf16* gate_up() const {
        return reinterpret_cast<const Bf16*>(data_ + kGateUpOffset);
    }
    [[nodiscard]] const Bf16* down() const {
        return reinterpret_cast<const Bf16*>(data_ + kDownOffset);
    }
    [[nodiscard]] const std::uint8_t* bytes_at(std::size_t offset) const {
        return data_ + offset;
    }
    [[nodiscard]] const std::string& path() const { return path_; }

  private:
    [[nodiscard]] std::uint32_t u32(std::size_t offset) const {
        const std::uint8_t* p = data_ + offset;
        return static_cast<std::uint32_t>(p[0]) |
               (static_cast<std::uint32_t>(p[1]) << 8) |
               (static_cast<std::uint32_t>(p[2]) << 16) |
               (static_cast<std::uint32_t>(p[3]) << 24);
    }

    [[nodiscard]] std::uint64_t u64(std::size_t offset) const {
        return static_cast<std::uint64_t>(u32(offset)) |
               (static_cast<std::uint64_t>(u32(offset + 4)) << 32);
    }

    void validate_header() const {
        constexpr std::array<std::uint8_t, 16> magic = {
            'B', 'E', 'H', 'O', 'L', 'D', 'D', 'Q',
            '3', '5', 'L', 'A', 'Y', 'E', 'R', 0};
        if (!std::equal(magic.begin(), magic.end(), data_)) {
            throw std::runtime_error("layer pack magic mismatch");
        }
        if (u32(16) != 1 || u32(20) != kHeaderBytes || u32(24) != 3 ||
            u32(28) != kHidden || u32(32) != kFfn ||
            u32(36) != kAllExperts || u32(40) != kTopK ||
            u32(44) != kFfn || u32(48) != 0x358637bdU || u32(52) != 1) {
            throw std::runtime_error(
                "layer pack version, layer, geometry, epsilon, or layout mismatch");
        }
        const std::array<std::array<std::uint64_t, 2>, 5> expected = {{
            {kNormOffset, kNormBytes},
            {kRouterOffset, kRouterBytes},
            {kSharedGateOffset, kSharedGateBytes},
            {kGateUpOffset, kGateUpBytes},
            {kDownOffset, kDownBytes},
        }};
        for (std::size_t index = 0; index < expected.size(); ++index) {
            const std::size_t header_offset = 64 + index * 16;
            if (u64(header_offset) != expected[index][0] ||
                u64(header_offset + 8) != expected[index][1]) {
                throw std::runtime_error("layer pack section table mismatch");
            }
        }
        if (u64(144) != kPackBytes) {
            throw std::runtime_error("layer pack declared length mismatch");
        }
        for (std::size_t index = 56; index < 64; ++index) {
            if (data_[index] != 0) {
                throw std::runtime_error("layer pack reserved header byte is nonzero");
            }
        }
        for (std::size_t index = 152; index < kHeaderBytes; ++index) {
            if (data_[index] != 0) {
                throw std::runtime_error("layer pack reserved header byte is nonzero");
            }
        }
    }

    std::string path_;
    int fd_ = -1;
    std::size_t bytes_ = 0;
    const std::uint8_t* data_ = nullptr;
};

struct LayerOptions {
    std::string pack_path;
    int warmups = 4;
    int samples = 12;
    int burst = 16;
};

int parse_positive(std::string_view value, std::string_view option) {
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size() || parsed <= 0) {
        throw std::runtime_error(std::string(option) +
                                 " requires a positive integer");
    }
    return parsed;
}

LayerOptions parse_options(int argc, char** argv) {
    LayerOptions options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        if (argument == "--quick") {
            options.warmups = 1;
            options.samples = 3;
            options.burst = 2;
        } else if (argument == "--warmups" || argument == "--samples" ||
                   argument == "--burst") {
            if (++index >= argc) {
                throw std::runtime_error(std::string(argument) +
                                         " requires a value");
            }
            const int value = parse_positive(argv[index], argument);
            if (argument == "--warmups") {
                options.warmups = value;
            } else if (argument == "--samples") {
                options.samples = value;
            } else {
                options.burst = value;
            }
        } else if (argument == "--help") {
            std::cout
                << "Usage: rocwmma_qwen35_resident_layer_bench [--quick] "
                   "[--warmups N] [--samples N] [--burst N] <layer.pack>\n";
            std::exit(0);
        } else if (!argument.empty() && argument.front() == '-') {
            throw std::runtime_error("unknown option " + std::string(argument));
        } else if (options.pack_path.empty()) {
            options.pack_path = std::string(argument);
        } else {
            throw std::runtime_error("more than one layer pack was supplied");
        }
    }
    if (options.pack_path.empty()) {
        throw std::runtime_error("a v1 Qwen3.6 layer pack path is required");
    }
    return options;
}

extern "C" __global__ void q35_rmsnorm_f32_to_f32_bf16(
    const float* hidden,
    const float* gamma,
    float* normalized,
    Bf16* padded_input) {
    __shared__ float partial[kElementThreads];
    const std::uint32_t lane = threadIdx.x;
    float sum = 0.0f;
    for (std::uint32_t index = lane; index < kHidden;
         index += kElementThreads) {
        const float value = hidden[index];
        sum += value * value;
    }
    partial[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = kElementThreads / 2; stride != 0;
         stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    const float mean_square =
        partial[0] / static_cast<float>(kHidden) + kRmsEpsilon;
    const float inverse_rms = 1.0f / __builtin_sqrtf(mean_square);
    for (std::uint32_t index = lane; index < kHidden;
         index += kElementThreads) {
        const float value = hidden[index] * inverse_rms * gamma[index];
        normalized[index] = value;
        padded_input[index] = Bf16(value);
    }
    for (std::uint32_t index = kHidden + lane;
         index < kPaddedTokens * kHidden; index += kElementThreads) {
        padded_input[index] = Bf16(0.0f);
    }
}

extern "C" __global__ void q35_router_and_shared_gate_f32(
    const float* normalized,
    const float* router,
    const float* shared_gate,
    float* logits,
    float* shared_logit) {
    __shared__ float partial[kElementThreads];
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t lane = threadIdx.x;
    const float* weights =
        row < kAllExperts ? router + static_cast<std::size_t>(row) * kHidden
                          : shared_gate;
    float sum = 0.0f;
    for (std::uint32_t index = lane; index < kHidden;
         index += kElementThreads) {
        sum += normalized[index] * weights[index];
    }
    partial[lane] = sum;
    __syncthreads();
    for (std::uint32_t stride = kElementThreads / 2; stride != 0;
         stride /= 2) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        if (row < kAllExperts) {
            logits[row] = partial[0];
        } else {
            *shared_logit = partial[0];
        }
    }
}

extern "C" __global__ void q35_stable_softmax_topk8(
    const float* logits,
    const float* shared_logit,
    float* probabilities,
    std::uint32_t* route_ids,
    float* route_weights,
    float* shared_scale,
    float* route_margin,
    std::uint32_t* route_status) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    // Every failure path must leave the later, already-enqueued WMMA launches
    // with bounded indices. The final reduction observes the nonzero status and
    // poisons the user-visible result, but it is too late to protect an expert
    // gather from an uninitialized route ID.
    *route_status = 5;
    *shared_scale = 0.0f;
    *route_margin = 0.0f;
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        route_ids[rank] = 0;
        route_weights[rank] = 0.0f;
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
        if (!__builtin_isfinite(logits[expert])) {
            *route_status = 1;
            return;
        }
        maximum = logits[expert] > maximum ? logits[expert] : maximum;
    }
    float denominator = 0.0f;
    for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
        const float probability = __builtin_expf(logits[expert] - maximum);
        probabilities[expert] = probability;
        denominator += probability;
    }
    if (!__builtin_isfinite(denominator) || !(denominator > 0.0f) ||
        !__builtin_isfinite(*shared_logit)) {
        *route_status = 2;
        return;
    }
    for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
        probabilities[expert] /= denominator;
    }

    float selected_sum = 0.0f;
    float eighth = 0.0f;
    float ninth = 0.0f;
    for (std::uint32_t rank = 0; rank <= kTopK; ++rank) {
        std::uint32_t best_expert = 0;
        float best_probability = -1.0f;
        for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
            const float probability = probabilities[expert];
            bool already_selected = false;
            for (std::uint32_t prior = 0; prior < rank && prior < kTopK;
                 ++prior) {
                already_selected =
                    already_selected || route_ids[prior] == expert;
            }
            if (!already_selected && probability > best_probability) {
                best_probability = probability;
                best_expert = expert;
            }
        }
        if (rank < kTopK) {
            route_ids[rank] = best_expert;
            route_weights[rank] = best_probability;
            selected_sum += best_probability;
            if (rank == kTopK - 1) {
                eighth = best_probability;
            }
        } else {
            ninth = best_probability;
        }
    }
    if (!__builtin_isfinite(selected_sum) || !(selected_sum > 0.0f)) {
        *route_status = 3;
        return;
    }
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        route_weights[rank] /= selected_sum;
    }
    *shared_scale = 1.0f / (1.0f + __builtin_expf(-*shared_logit));
    *route_margin = eighth - ninth;
    *route_status = __builtin_isfinite(*shared_scale) ? 0 : 4;
}

extern "C" __global__ __launch_bounds__(kWaveSize)
void q35_selected_shared_gate_up_wmma(const Bf16* padded_input,
                                      const Bf16* gate_up_weights,
                                      const std::uint32_t* route_ids,
                                      float* gate_up_output) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a,
                                    kTile,
                                    kTile,
                                    kTile,
                                    Bf16,
                                    rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b,
                                    kTile,
                                    kTile,
                                    kTile,
                                    Bf16,
                                    rocwmma::col_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator,
                                      kTile,
                                      kTile,
                                      kTile,
                                      float,
                                      rocwmma::row_major>;

    const std::uint32_t output_column = blockIdx.x * kTile;
    const std::uint32_t slot = blockIdx.y;
    const std::uint32_t projection = blockIdx.z;
    const std::uint32_t routed = slot < kTopK ? route_ids[slot] : 0;
    const std::uint32_t expert =
        slot < kTopK ? (routed < kAllExperts ? routed : 0) : kSharedSlot;
    const std::size_t matrix = kExpertMatrixElements;
    const Bf16* weights =
        gate_up_weights +
        (static_cast<std::size_t>(projection) * kPackedSlots + expert) *
            matrix;

    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);
    for (std::uint32_t inner = 0; inner < kHidden; inner += kTile) {
        FragA input_fragment;
        FragB weight_fragment;
        rocwmma::load_matrix_sync(input_fragment,
                                  padded_input + inner,
                                  kHidden);
        rocwmma::load_matrix_sync(weight_fragment,
                                  weights +
                                      static_cast<std::size_t>(output_column) *
                                          kHidden +
                                      inner,
                                  kHidden);
        rocwmma::mma_sync(accumulator,
                          input_fragment,
                          weight_fragment,
                          accumulator);
    }

    const std::size_t output_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kFfn;
    float* output = gate_up_output +
                    (static_cast<std::size_t>(projection) * kComputeSlots +
                     slot) *
                        output_matrix;
    rocwmma::store_matrix_sync(output + output_column, accumulator, kFfn);
}

extern "C" __global__ void q35_silu_multiply_to_bf16(
    const float* gate_up_output,
    Bf16* activation) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr std::uint32_t plane_elements =
        kComputeSlots * kPaddedTokens * kFfn;
    if (index >= plane_elements) {
        return;
    }
    const float gate = gate_up_output[index];
    const float up = gate_up_output[plane_elements + index];
    const float silu = gate / (1.0f + __builtin_expf(-gate));
    activation[index] = Bf16(silu * up);
}

extern "C" __global__ __launch_bounds__(kWaveSize)
void q35_selected_shared_down_wmma(const Bf16* activation,
                                   const Bf16* down_weights,
                                   const std::uint32_t* route_ids,
                                   float* expert_output) {
    using FragA = rocwmma::fragment<rocwmma::matrix_a,
                                    kTile,
                                    kTile,
                                    kTile,
                                    Bf16,
                                    rocwmma::row_major>;
    using FragB = rocwmma::fragment<rocwmma::matrix_b,
                                    kTile,
                                    kTile,
                                    kTile,
                                    Bf16,
                                    rocwmma::col_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator,
                                      kTile,
                                      kTile,
                                      kTile,
                                      float,
                                      rocwmma::row_major>;

    const std::uint32_t output_column = blockIdx.x * kTile;
    const std::uint32_t slot = blockIdx.y;
    const std::uint32_t routed = slot < kTopK ? route_ids[slot] : 0;
    const std::uint32_t expert =
        slot < kTopK ? (routed < kAllExperts ? routed : 0) : kSharedSlot;
    const std::size_t activation_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kFfn;
    const Bf16* slot_activation = activation + slot * activation_matrix;
    const Bf16* weights =
        down_weights + static_cast<std::size_t>(expert) *
                           kExpertMatrixElements;

    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);
    for (std::uint32_t inner = 0; inner < kFfn; inner += kTile) {
        FragA activation_fragment;
        FragB weight_fragment;
        rocwmma::load_matrix_sync(activation_fragment,
                                  slot_activation + inner,
                                  kFfn);
        rocwmma::load_matrix_sync(weight_fragment,
                                  weights +
                                      static_cast<std::size_t>(output_column) *
                                          kFfn +
                                      inner,
                                  kFfn);
        rocwmma::mma_sync(accumulator,
                          activation_fragment,
                          weight_fragment,
                          accumulator);
    }

    const std::size_t output_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kHidden;
    float* output = expert_output + slot * output_matrix;
    rocwmma::store_matrix_sync(output + output_column, accumulator, kHidden);
}

extern "C" __global__ void q35_route_shared_residual_reduce(
    const float* expert_output,
    const std::uint32_t* route_status,
    const float* route_weights,
    const float* shared_scale,
    const float* residual,
    float* output) {
    const std::uint32_t column = blockIdx.x * blockDim.x + threadIdx.x;
    if (column >= kHidden) {
        return;
    }
    if (*route_status != 0) {
        output[column] = __builtin_nanf("");
        return;
    }
    const std::size_t output_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kHidden;
    float sum = 0.0f;
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        sum += route_weights[rank] *
               expert_output[static_cast<std::size_t>(rank) * output_matrix +
                             column];
    }
    sum += *shared_scale *
           expert_output[static_cast<std::size_t>(kTopK) * output_matrix +
                         column];
    output[column] = residual[column] + sum;
}

struct ResidentLayer {
    explicit ResidentLayer() : norm(kHidden),
                               router(static_cast<std::size_t>(kAllExperts) *
                                      kHidden),
                               shared_gate(kHidden),
                               gate_up(kGateUpElements),
                               down(kDownElements) {}

    DeviceBuffer<float> norm;
    DeviceBuffer<float> router;
    DeviceBuffer<float> shared_gate;
    DeviceBuffer<Bf16> gate_up;
    DeviceBuffer<Bf16> down;
};

struct ResidentState {
    ResidentState()
        : hidden(kHidden),
          normalized(kHidden),
          padded_input(static_cast<std::size_t>(kPaddedTokens) * kHidden),
          logits(kAllExperts),
          shared_logit(1),
          probabilities(kAllExperts),
          route_ids(kTopK),
          route_weights(kTopK),
          shared_scale(1),
          route_margin(1),
          route_status(1),
          control_route_ids(kTopK),
          control_route_weights(kTopK),
          control_shared_scale(1),
          control_route_status(1),
          gate_up_output(2ULL * kComputeSlots * kPaddedTokens * kFfn),
          activation(static_cast<std::size_t>(kComputeSlots) *
                     kPaddedTokens * kFfn),
          expert_output(static_cast<std::size_t>(kComputeSlots) *
                        kPaddedTokens * kHidden),
          output(kHidden) {}

    DeviceBuffer<float> hidden;
    DeviceBuffer<float> normalized;
    DeviceBuffer<Bf16> padded_input;
    DeviceBuffer<float> logits;
    DeviceBuffer<float> shared_logit;
    DeviceBuffer<float> probabilities;
    DeviceBuffer<std::uint32_t> route_ids;
    DeviceBuffer<float> route_weights;
    DeviceBuffer<float> shared_scale;
    DeviceBuffer<float> route_margin;
    DeviceBuffer<std::uint32_t> route_status;
    DeviceBuffer<std::uint32_t> control_route_ids;
    DeviceBuffer<float> control_route_weights;
    DeviceBuffer<float> control_shared_scale;
    DeviceBuffer<std::uint32_t> control_route_status;
    DeviceBuffer<float> gate_up_output;
    DeviceBuffer<Bf16> activation;
    DeviceBuffer<float> expert_output;
    DeviceBuffer<float> output;
};

std::size_t state_bytes() {
    return kHidden * sizeof(float) * 3 +
           static_cast<std::size_t>(kPaddedTokens) * kHidden * sizeof(Bf16) +
           kAllExperts * sizeof(float) * 2 + sizeof(float) +
           2 * kTopK * (sizeof(std::uint32_t) + sizeof(float)) +
           4 * sizeof(float) + 2 * sizeof(std::uint32_t) +
           2ULL * kComputeSlots * kPaddedTokens * kFfn * sizeof(float) +
           static_cast<std::size_t>(kComputeSlots) * kPaddedTokens * kFfn *
               sizeof(Bf16) +
           static_cast<std::size_t>(kComputeSlots) * kPaddedTokens * kHidden *
               sizeof(float) +
           kHidden * sizeof(float);
}

void require_memory_headroom() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t required = kNormBytes + kRouterBytes + kSharedGateBytes +
                                 kGateUpBytes + kDownBytes + state_bytes();
    if (required > std::numeric_limits<std::size_t>::max() -
                       kHeadroomBytes ||
        free_bytes < required + kHeadroomBytes) {
        throw std::runtime_error(
            "insufficient free VRAM for exact layer plus 512 MiB headroom: free=" +
            std::to_string(free_bytes) + " required=" +
            std::to_string(required + kHeadroomBytes));
    }
    std::cout << "vram_total_MiB=" << total_bytes / (1024 * 1024)
              << " vram_free_before_MiB=" << free_bytes / (1024 * 1024)
              << " required_device_MiB=" << required / (1024 * 1024)
              << " reserved_headroom_MiB=" << kHeadroomBytes / (1024 * 1024)
              << '\n';
}

void upload_layer(const MappedPack& pack,
                  ResidentLayer& layer,
                  hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(layer.norm.get(),
                             pack.bytes_at(kNormOffset),
                             kNormBytes,
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(layer.router.get(),
                             pack.bytes_at(kRouterOffset),
                             kRouterBytes,
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(layer.shared_gate.get(),
                             pack.bytes_at(kSharedGateOffset),
                             kSharedGateBytes,
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(layer.gate_up.get(),
                             pack.bytes_at(kGateUpOffset),
                             kGateUpBytes,
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(layer.down.get(),
                             pack.bytes_at(kDownOffset),
                             kDownBytes,
                             hipMemcpyHostToDevice,
                             stream));
}

void launch_prefix(const ResidentLayer& layer,
                   ResidentState& state,
                   hipStream_t stream) {
    hipLaunchKernelGGL(q35_rmsnorm_f32_to_f32_bf16,
                       dim3(1),
                       dim3(kElementThreads),
                       0,
                       stream,
                       state.hidden.get(),
                       layer.norm.get(),
                       state.normalized.get(),
                       state.padded_input.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(q35_router_and_shared_gate_f32,
                       dim3(kAllExperts + 1),
                       dim3(kElementThreads),
                       0,
                       stream,
                       state.normalized.get(),
                       layer.router.get(),
                       layer.shared_gate.get(),
                       state.logits.get(),
                       state.shared_logit.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(q35_stable_softmax_topk8,
                       dim3(1),
                       dim3(1),
                       0,
                       stream,
                       state.logits.get(),
                       state.shared_logit.get(),
                       state.probabilities.get(),
                       state.route_ids.get(),
                       state.route_weights.get(),
                       state.shared_scale.get(),
                       state.route_margin.get(),
                       state.route_status.get());
    HIP_CHECK(hipGetLastError());
}

void launch_suffix(const ResidentLayer& layer,
                   ResidentState& state,
                   const std::uint32_t* route_ids,
                   const float* route_weights,
                   const float* shared_scale,
                   const std::uint32_t* route_status,
                   hipStream_t stream) {
    hipLaunchKernelGGL(q35_selected_shared_gate_up_wmma,
                       dim3(kFfn / kTile, kComputeSlots, 2),
                       dim3(kWaveSize),
                       0,
                       stream,
                       state.padded_input.get(),
                       layer.gate_up.get(),
                       route_ids,
                       state.gate_up_output.get());
    HIP_CHECK(hipGetLastError());

    constexpr std::uint32_t activation_elements =
        kComputeSlots * kPaddedTokens * kFfn;
    hipLaunchKernelGGL(q35_silu_multiply_to_bf16,
                       dim3((activation_elements + kElementThreads - 1) /
                            kElementThreads),
                       dim3(kElementThreads),
                       0,
                       stream,
                       state.gate_up_output.get(),
                       state.activation.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(q35_selected_shared_down_wmma,
                       dim3(kHidden / kTile, kComputeSlots),
                       dim3(kWaveSize),
                       0,
                       stream,
                       state.activation.get(),
                       layer.down.get(),
                       route_ids,
                       state.expert_output.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(q35_route_shared_residual_reduce,
                       dim3((kHidden + kElementThreads - 1) /
                            kElementThreads),
                       dim3(kElementThreads),
                       0,
                       stream,
                       state.expert_output.get(),
                       route_status,
                       route_weights,
                       shared_scale,
                       state.hidden.get(),
                       state.output.get());
    HIP_CHECK(hipGetLastError());
}

void enqueue_resident(const ResidentLayer& layer,
                      ResidentState& state,
                      hipStream_t stream) {
    launch_prefix(layer, state, stream);
    launch_suffix(layer,
                  state,
                  state.route_ids.get(),
                  state.route_weights.get(),
                  state.shared_scale.get(),
                  state.route_status.get(),
                  stream);
}

struct RouteTransfer {
    RouteTransfer()
        : ids(kTopK), weights(kTopK), shared_scale(1), status(1) {}

    PinnedBuffer<std::uint32_t> ids;
    PinnedBuffer<float> weights;
    PinnedBuffer<float> shared_scale;
    PinnedBuffer<std::uint32_t> status;
};

void enqueue_host_route_roundtrip(const ResidentLayer& layer,
                                  ResidentState& state,
                                  RouteTransfer& host,
                                  hipStream_t stream) {
    launch_prefix(layer, state, stream);
    HIP_CHECK(hipMemcpyAsync(host.ids.get(),
                             state.route_ids.get(),
                             host.ids.size_bytes(),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(host.weights.get(),
                             state.route_weights.get(),
                             host.weights.size_bytes(),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(host.shared_scale.get(),
                             state.shared_scale.get(),
                             host.shared_scale.size_bytes(),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(host.status.get(),
                             state.route_status.get(),
                             host.status.size_bytes(),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipMemcpyAsync(state.control_route_ids.get(),
                             host.ids.get(),
                             host.ids.size_bytes(),
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(state.control_route_weights.get(),
                             host.weights.get(),
                             host.weights.size_bytes(),
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(state.control_shared_scale.get(),
                             host.shared_scale.get(),
                             host.shared_scale.size_bytes(),
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(state.control_route_status.get(),
                             host.status.get(),
                             host.status.size_bytes(),
                             hipMemcpyHostToDevice,
                             stream));
    launch_suffix(layer,
                  state,
                  state.control_route_ids.get(),
                  state.control_route_weights.get(),
                  state.control_shared_scale.get(),
                  state.control_route_status.get(),
                  stream);
}

float bf16_round(float value) {
    return static_cast<float>(Bf16(value));
}

float reduction_dot(const float* values, const float* weights) {
    std::array<float, kElementThreads> partial{};
    for (std::uint32_t lane = 0; lane < kElementThreads; ++lane) {
        float sum = 0.0f;
        for (std::uint32_t index = lane; index < kHidden;
             index += kElementThreads) {
            sum += values[index] * weights[index];
        }
        partial[lane] = sum;
    }
    for (std::uint32_t stride = kElementThreads / 2; stride != 0;
         stride /= 2) {
        for (std::uint32_t lane = 0; lane < stride; ++lane) {
            partial[lane] += partial[lane + stride];
        }
    }
    return partial[0];
}

struct CpuReference {
    std::array<std::uint32_t, kTopK> route_ids{};
    std::array<float, kTopK> route_weights{};
    float route_margin = 0.0f;
    float shared_scale = 0.0f;
    std::vector<float> output;
};

CpuReference cpu_reference(const MappedPack& pack,
                           const std::vector<float>& hidden) {
    CpuReference reference;
    reference.output.resize(kHidden);
    std::vector<float> normalized(kHidden);
    std::array<float, kElementThreads> squares{};
    for (std::uint32_t lane = 0; lane < kElementThreads; ++lane) {
        float sum = 0.0f;
        for (std::uint32_t index = lane; index < kHidden;
             index += kElementThreads) {
            sum += hidden[index] * hidden[index];
        }
        squares[lane] = sum;
    }
    for (std::uint32_t stride = kElementThreads / 2; stride != 0;
         stride /= 2) {
        for (std::uint32_t lane = 0; lane < stride; ++lane) {
            squares[lane] += squares[lane + stride];
        }
    }
    const float inverse_rms =
        1.0f / std::sqrt(squares[0] / static_cast<float>(kHidden) +
                         kRmsEpsilon);
    std::vector<float> rounded_input(kHidden);
    for (std::uint32_t index = 0; index < kHidden; ++index) {
        normalized[index] = hidden[index] * inverse_rms * pack.norm()[index];
        rounded_input[index] = bf16_round(normalized[index]);
    }

    std::array<float, kAllExperts> logits{};
    float maximum = -std::numeric_limits<float>::infinity();
    for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
        logits[expert] = reduction_dot(
            normalized.data(),
            pack.router() + static_cast<std::size_t>(expert) * kHidden);
        maximum = std::max(maximum, logits[expert]);
    }
    std::array<float, kAllExperts> probabilities{};
    float denominator = 0.0f;
    for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
        probabilities[expert] = std::exp(logits[expert] - maximum);
        denominator += probabilities[expert];
    }
    for (float& probability : probabilities) {
        probability /= denominator;
    }
    std::array<bool, kAllExperts> selected{};
    float selected_sum = 0.0f;
    float eighth = 0.0f;
    float ninth = 0.0f;
    for (std::uint32_t rank = 0; rank <= kTopK; ++rank) {
        std::uint32_t best = 0;
        float best_probability = -1.0f;
        for (std::uint32_t expert = 0; expert < kAllExperts; ++expert) {
            if (!selected[expert] &&
                probabilities[expert] > best_probability) {
                best = expert;
                best_probability = probabilities[expert];
            }
        }
        selected[best] = true;
        if (rank < kTopK) {
            reference.route_ids[rank] = best;
            reference.route_weights[rank] = best_probability;
            selected_sum += best_probability;
            if (rank == kTopK - 1) {
                eighth = best_probability;
            }
        } else {
            ninth = best_probability;
        }
    }
    for (float& weight : reference.route_weights) {
        weight /= selected_sum;
    }
    reference.route_margin = eighth - ninth;
    const float shared_logit =
        reduction_dot(normalized.data(), pack.shared_gate());
    reference.shared_scale = 1.0f / (1.0f + std::exp(-shared_logit));

    std::vector<float> gate(kFfn);
    std::vector<float> up(kFfn);
    std::vector<float> activation(kFfn);
    std::vector<float> down(kHidden);
    std::fill(reference.output.begin(), reference.output.end(), 0.0f);
    for (std::uint32_t slot = 0; slot < kComputeSlots; ++slot) {
        const std::uint32_t expert =
            slot < kTopK ? reference.route_ids[slot] : kSharedSlot;
        const Bf16* gate_weights =
            pack.gate_up() + static_cast<std::size_t>(expert) *
                                 kExpertMatrixElements;
        const Bf16* up_weights =
            pack.gate_up() +
            (static_cast<std::size_t>(kPackedSlots) + expert) *
                kExpertMatrixElements;
        for (std::uint32_t column = 0; column < kFfn; ++column) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            const std::size_t row =
                static_cast<std::size_t>(column) * kHidden;
            for (std::uint32_t inner = 0; inner < kHidden; ++inner) {
                gate_sum += rounded_input[inner] *
                            static_cast<float>(gate_weights[row + inner]);
                up_sum += rounded_input[inner] *
                          static_cast<float>(up_weights[row + inner]);
            }
            gate[column] = gate_sum;
            up[column] = up_sum;
            const float value =
                (gate_sum / (1.0f + std::exp(-gate_sum))) * up_sum;
            activation[column] = bf16_round(value);
        }
        const Bf16* down_weights =
            pack.down() + static_cast<std::size_t>(expert) *
                              kExpertMatrixElements;
        for (std::uint32_t column = 0; column < kHidden; ++column) {
            float sum = 0.0f;
            const std::size_t row = static_cast<std::size_t>(column) * kFfn;
            for (std::uint32_t inner = 0; inner < kFfn; ++inner) {
                sum += activation[inner] *
                       static_cast<float>(down_weights[row + inner]);
            }
            down[column] = sum;
        }
        const float scale = slot < kTopK
                                ? reference.route_weights[slot]
                                : reference.shared_scale;
        for (std::uint32_t column = 0; column < kHidden; ++column) {
            reference.output[column] += scale * down[column];
        }
    }
    for (std::uint32_t column = 0; column < kHidden; ++column) {
        reference.output[column] += hidden[column];
    }
    return reference;
}

struct ErrorMetrics {
    double relative_l2 = 0.0;
    double max_absolute = 0.0;
    double checksum = 0.0;
    bool all_finite = true;
};

ErrorMetrics compare_output(const std::vector<float>& actual,
                            const std::vector<float>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("output lengths differ");
    }
    double difference_squared = 0.0;
    double expected_squared = 0.0;
    ErrorMetrics result;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const double observed = actual[index];
        const double reference = expected[index];
        result.all_finite = result.all_finite && std::isfinite(observed) &&
                            std::isfinite(reference);
        const double difference = observed - reference;
        difference_squared += difference * difference;
        expected_squared += reference * reference;
        result.max_absolute =
            std::max(result.max_absolute, std::abs(difference));
        result.checksum += observed * static_cast<double>((index % 17) + 1);
    }
    result.relative_l2 =
        std::sqrt(difference_squared /
                  std::max(expected_squared,
                           std::numeric_limits<double>::min()));
    return result;
}

struct TimingStats {
    double minimum_ms = 0.0;
    double median_ms = 0.0;
    double p90_ms = 0.0;
};

TimingStats summarize(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t p90 =
        std::min(values.size() - 1,
                 static_cast<std::size_t>(std::ceil(values.size() * 0.9)) - 1);
    return {values.front(), values[values.size() / 2], values[p90]};
}

template <typename Enqueue>
double timed_burst(int burst, hipStream_t stream, Enqueue&& enqueue) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto begin = std::chrono::steady_clock::now();
    for (int index = 0; index < burst; ++index) {
        enqueue();
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           static_cast<double>(burst);
}

std::vector<float> read_output(ResidentState& state, hipStream_t stream) {
    std::vector<float> output(kHidden);
    HIP_CHECK(hipMemcpyAsync(output.data(),
                             state.output.get(),
                             output.size() * sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return output;
}

struct ObservedRoute {
    std::array<std::uint32_t, kTopK> ids{};
    std::array<float, kTopK> weights{};
    float shared_scale = 0.0f;
    float margin = 0.0f;
    std::uint32_t status = std::numeric_limits<std::uint32_t>::max();
};

ObservedRoute read_route(ResidentState& state, hipStream_t stream) {
    ObservedRoute route;
    HIP_CHECK(hipMemcpyAsync(route.ids.data(),
                             state.route_ids.get(),
                             route.ids.size() * sizeof(std::uint32_t),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(route.weights.data(),
                             state.route_weights.get(),
                             route.weights.size() * sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(&route.shared_scale,
                             state.shared_scale.get(),
                             sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(&route.margin,
                             state.route_margin.get(),
                             sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipMemcpyAsync(&route.status,
                             state.route_status.get(),
                             sizeof(std::uint32_t),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return route;
}

void poison_scratch(ResidentState& state, hipStream_t stream) {
    for (const auto [pointer, bytes] :
         std::array<std::pair<void*, std::size_t>, 9>{{
             {state.normalized.get(), state.normalized.size_bytes()},
             {state.padded_input.get(), state.padded_input.size_bytes()},
             {state.logits.get(), state.logits.size_bytes()},
             {state.probabilities.get(), state.probabilities.size_bytes()},
             {state.route_ids.get(), state.route_ids.size_bytes()},
             {state.route_weights.get(), state.route_weights.size_bytes()},
             {state.gate_up_output.get(), state.gate_up_output.size_bytes()},
             {state.activation.get(), state.activation.size_bytes()},
             {state.expert_output.get(), state.expert_output.size_bytes()},
         }}) {
        HIP_CHECK(hipMemsetAsync(pointer, 0xa5, bytes, stream));
    }
    HIP_CHECK(hipMemsetAsync(state.route_status.get(),
                             0xff,
                             state.route_status.size_bytes(),
                             stream));
    HIP_CHECK(hipMemsetAsync(state.output.get(),
                             0xa5,
                             state.output.size_bytes(),
                             stream));
}

void print_timing(std::string_view label, const TimingStats& timing) {
    std::cout << label << "_min_ms=" << std::fixed << std::setprecision(6)
              << timing.minimum_ms << ' ' << label
              << "_median_ms=" << timing.median_ms << ' ' << label
              << "_p90_ms=" << timing.p90_ms << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const LayerOptions options = parse_options(argc, argv);
        const MappedPack pack(options.pack_path);

        int device_index = 0;
        HIP_CHECK(hipGetDevice(&device_index));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device_index));
        const std::string_view architecture(properties.gcnArchName);
        if (architecture != "gfx1151" &&
            !architecture.starts_with("gfx1151:")) {
            throw std::runtime_error("benchmark requires gfx1151; device reports " +
                                     std::string(architecture));
        }
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error("gfx1151 rocWMMA requires wave32; device reports " +
                                     std::to_string(properties.warpSize));
        }

        std::cout << "device=" << properties.name
                  << " arch=" << properties.gcnArchName
                  << " wave=" << properties.warpSize << '\n';
        std::cout << "pack=" << pack.path()
                  << " bytes=" << kPackBytes
                  << " version=1 layer=3 H=2048 F=512 E=256 K=8 sharedF=512\n";
        std::cout << "weights=native_bf16 checkpoint_output_major "
                     "wmma_B=col_major transpose=false resident=true\n";
        std::cout << "candidate=7_launches one_external_stream_sync "
                     "control=same_gpu_route_plus_D2H_sync_H2D\n";

        require_memory_headroom();
        Stream stream;
        const auto upload_begin = std::chrono::steady_clock::now();
        ResidentLayer layer;
        ResidentState state;
        RouteTransfer transfer;
        upload_layer(pack, layer, stream.get());

        std::vector<float> hidden(kHidden);
        for (std::uint32_t index = 0; index < kHidden; ++index) {
            hidden[index] =
                0.35f * std::sin(static_cast<float>(index) * 0.013f) +
                0.2f * std::cos(static_cast<float>(index) * 0.037f) +
                static_cast<float>(static_cast<int>(index % 11) - 5) *
                    0.007f;
        }
        HIP_CHECK(hipMemcpyAsync(state.hidden.get(),
                                 hidden.data(),
                                 hidden.size() * sizeof(float),
                                 hipMemcpyHostToDevice,
                                 stream.get()));
        HIP_CHECK(hipStreamSynchronize(stream.get()));
        const auto upload_end = std::chrono::steady_clock::now();
        std::cout << "allocation_weight_upload_ms=" << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(upload_end -
                                                               upload_begin)
                         .count()
                  << " excluded_from_inference_timing=true\n";

        const auto oracle_begin = std::chrono::steady_clock::now();
        const CpuReference reference = cpu_reference(pack, hidden);
        const auto oracle_end = std::chrono::steady_clock::now();
        std::cout << "cpu_oracle_ms="
                  << std::chrono::duration<double, std::milli>(oracle_end -
                                                               oracle_begin)
                         .count()
                  << '\n';

        poison_scratch(state, stream.get());
        enqueue_resident(layer, state, stream.get());
        const std::vector<float> first = read_output(state, stream.get());
        const ObservedRoute observed = read_route(state, stream.get());
        enqueue_resident(layer, state, stream.get());
        const std::vector<float> second = read_output(state, stream.get());
        const bool deterministic =
            std::memcmp(first.data(), second.data(),
                        first.size() * sizeof(float)) == 0;

        enqueue_host_route_roundtrip(layer, state, transfer, stream.get());
        const std::vector<float> control = read_output(state, stream.get());
        const bool control_exact =
            std::memcmp(first.data(), control.data(),
                        first.size() * sizeof(float)) == 0;
        const ErrorMetrics error = compare_output(first, reference.output);

        bool route_exact = observed.status == 0;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_exact = route_exact &&
                          observed.ids[rank] == reference.route_ids[rank];
        }
        std::cout << "route_ids=";
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            if (rank != 0) {
                std::cout << ':';
            }
            std::cout << observed.ids[rank];
        }
        std::cout << " route_status=" << observed.status
                  << " route_exact=" << std::boolalpha << route_exact
                  << " rank8_9_margin=" << std::scientific
                  << observed.margin
                  << " shared_scale=" << observed.shared_scale << '\n';
        std::cout << "correctness_relative_l2=" << error.relative_l2
                  << " correctness_max_abs=" << error.max_absolute
                  << " checksum=" << error.checksum
                  << " all_finite=" << error.all_finite
                  << " bit_deterministic=" << deterministic
                  << " route_roundtrip_bit_exact=" << control_exact << '\n';

        for (int index = 0; index < options.warmups; ++index) {
            enqueue_resident(layer, state, stream.get());
            HIP_CHECK(hipStreamSynchronize(stream.get()));
            enqueue_host_route_roundtrip(layer, state, transfer, stream.get());
            HIP_CHECK(hipStreamSynchronize(stream.get()));
        }

        std::vector<double> candidate_samples;
        std::vector<double> control_samples;
        candidate_samples.reserve(options.samples);
        control_samples.reserve(options.samples);
        for (int sample = 0; sample < options.samples; ++sample) {
            auto candidate = [&] {
                candidate_samples.push_back(timed_burst(
                    options.burst,
                    stream.get(),
                    [&] { enqueue_resident(layer, state, stream.get()); }));
            };
            auto host_control = [&] {
                control_samples.push_back(timed_burst(
                    options.burst,
                    stream.get(),
                    [&] {
                        enqueue_host_route_roundtrip(
                            layer, state, transfer, stream.get());
                    }));
            };
            if (sample % 2 == 0) {
                candidate();
                host_control();
            } else {
                host_control();
                candidate();
            }
        }
        const TimingStats candidate = summarize(std::move(candidate_samples));
        const TimingStats host_control = summarize(std::move(control_samples));
        print_timing("resident", candidate);
        print_timing("host_route_roundtrip", host_control);
        const double speedup = host_control.median_ms / candidate.median_ms;
        std::cout << "median_speedup=" << std::fixed << std::setprecision(4)
                  << speedup
                  << " routed_layers_per_second="
                  << 1000.0 / candidate.median_ms
                  << " warmups=" << options.warmups
                  << " samples=" << options.samples
                  << " burst=" << options.burst << '\n';

        constexpr double maximum_relative_l2 = 5.0e-4;
        constexpr double maximum_absolute = 1.0e-3;
        if (!route_exact || !error.all_finite || !deterministic ||
            !control_exact || error.relative_l2 > maximum_relative_l2 ||
            error.max_absolute > maximum_absolute) {
            std::cerr << "FAIL: resident layer correctness gate failed\n";
            return 2;
        }
        std::cout << "PASS: exact routes, finite output, deterministic bits, "
                     "bit-exact route control, and CPU error thresholds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
