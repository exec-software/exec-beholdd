// Standalone BF16 rocWMMA routed-MoE experiment for gfx1151.
//
// This deliberately lives outside the Beholdd runtime. It is a qualification
// harness for a 16-token WMMA execution shape, not a production integration.

#include <stddef.h>

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using Bf16 = rocwmma::bfloat16_t;

constexpr std::uint32_t kHidden = 2048;
constexpr std::uint32_t kFfn = 512;
constexpr std::uint32_t kTopK = 8;
constexpr std::uint32_t kExperts = 8;
constexpr std::uint32_t kPaddedTokens = 16;
constexpr std::uint32_t kTile = 16;
constexpr std::uint32_t kWaveSize = 32;
constexpr std::uint32_t kProjections = 2;

static_assert(kHidden % kTile == 0);
static_assert(kFfn % kTile == 0);
static_assert(kTopK == kExperts,
              "This experiment computes one route through every resident expert");

struct RouteEntry {
    std::uint32_t expert;
    float weight;
};

static_assert(sizeof(RouteEntry) == 8);

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
        HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
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
    [[nodiscard]] std::size_t size_bytes() const { return count_ * sizeof(T); }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

template <typename T>
class PinnedBuffer {
  public:
    explicit PinnedBuffer(std::size_t count) : count_(count) {
        HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&data_),
                               count * sizeof(T), hipHostMallocDefault));
    }

    ~PinnedBuffer() {
        if (data_ != nullptr) {
            (void)hipHostFree(data_);
        }
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;

    [[nodiscard]] T* get() const { return data_; }
    [[nodiscard]] std::size_t size_bytes() const { return count_ * sizeof(T); }

  private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

class Stream {
  public:
    Stream() { HIP_CHECK(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking)); }

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

__global__ void f32_to_bf16_padded(const float* input,
                                   Bf16* padded,
                                   std::uint32_t token_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t element_count = kPaddedTokens * kHidden;
    if (index >= element_count) {
        return;
    }

    const std::uint32_t token = index / kHidden;
    padded[index] = token < token_count ? Bf16(input[index]) : Bf16(0.0f);
}

// gate_up_weights holds two groups of kExperts logical HxF matrices. Matrix B
// is row-major so each WMMA load follows the current public rocWMMA layout that
// was independently validated on gfx1151. This is a one-time transposed
// representation of an ordinary output-major model tensor, not checkpoint-
// native storage.
__global__ __launch_bounds__(kWaveSize)
void gate_up_wmma(const Bf16* padded_input,
                  const Bf16* gate_up_weights,
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
                                    rocwmma::row_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator,
                                      kTile,
                                      kTile,
                                      kTile,
                                      float,
                                      rocwmma::row_major>;

    const std::uint32_t output_tile = blockIdx.x;
    const std::uint32_t expert = blockIdx.y;
    const std::uint32_t projection = blockIdx.z;
    const std::uint32_t output_column = output_tile * kTile;

    const std::size_t matrix_elements =
        static_cast<std::size_t>(kHidden) * kFfn;
    const Bf16* weights = gate_up_weights +
                          (projection * kExperts + expert) * matrix_elements;

    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);

    for (std::uint32_t inner = 0; inner < kHidden; inner += kTile) {
        FragA input_fragment;
        FragB weight_fragment;
        rocwmma::load_matrix_sync(input_fragment,
                                  padded_input + inner,
                                  kHidden);
        rocwmma::load_matrix_sync(weight_fragment,
                                  weights + inner * kFfn + output_column,
                                  kFfn);
        rocwmma::mma_sync(accumulator,
                          input_fragment,
                          weight_fragment,
                          accumulator);
    }

    const std::size_t output_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kFfn;
    float* output = gate_up_output +
                    (projection * kExperts + expert) * output_matrix;
    rocwmma::store_matrix_sync(output + output_column, accumulator, kFfn);
}

__global__ void silu_multiply_to_bf16(const float* gate_up_output,
                                      Bf16* activation) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t plane_elements =
        kExperts * kPaddedTokens * kFfn;
    if (index >= plane_elements) {
        return;
    }

    const float gate = gate_up_output[index];
    const float up = gate_up_output[plane_elements + index];
    const float silu = gate / (1.0f + __builtin_expf(-gate));
    activation[index] = Bf16(silu * up);
}

// down_weights holds kExperts logical FxH row-major matrices. This is likewise
// prepared from the ordinary output-major down-projection tensor once.
__global__ __launch_bounds__(kWaveSize)
void down_wmma(const Bf16* activation,
               const Bf16* down_weights,
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
                                    rocwmma::row_major>;
    using FragAcc = rocwmma::fragment<rocwmma::accumulator,
                                      kTile,
                                      kTile,
                                      kTile,
                                      float,
                                      rocwmma::row_major>;

    const std::uint32_t output_tile = blockIdx.x;
    const std::uint32_t expert = blockIdx.y;
    const std::uint32_t output_column = output_tile * kTile;

    const std::size_t activation_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kFfn;
    const std::size_t weight_matrix =
        static_cast<std::size_t>(kFfn) * kHidden;
    const Bf16* expert_activation = activation + expert * activation_matrix;
    const Bf16* weights = down_weights + expert * weight_matrix;

    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);

    for (std::uint32_t inner = 0; inner < kFfn; inner += kTile) {
        FragA activation_fragment;
        FragB weight_fragment;
        rocwmma::load_matrix_sync(activation_fragment,
                                  expert_activation + inner,
                                  kFfn);
        rocwmma::load_matrix_sync(weight_fragment,
                                  weights + inner * kHidden + output_column,
                                  kHidden);
        rocwmma::mma_sync(accumulator,
                          activation_fragment,
                          weight_fragment,
                          accumulator);
    }

    const std::size_t output_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kHidden;
    float* output = expert_output + expert * output_matrix;
    rocwmma::store_matrix_sync(output + output_column, accumulator, kHidden);
}

__global__ void weighted_route_reduce(const float* expert_output,
                                      const RouteEntry* routes,
                                      float* output,
                                      std::uint32_t token_count) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t element_count = token_count * kHidden;
    if (index >= element_count) {
        return;
    }

    const std::uint32_t token = index / kHidden;
    const std::uint32_t column = index % kHidden;
    const std::size_t expert_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kHidden;

    float sum = 0.0f;
    for (std::uint32_t route_index = 0; route_index < kTopK; ++route_index) {
        const RouteEntry route = routes[token * kTopK + route_index];
        sum += route.weight *
               expert_output[route.expert * expert_matrix +
                             token * kHidden + column];
    }
    output[index] = sum;
}

struct Options {
    int warmups = 4;
    int samples = 9;
    int burst = 8;
    std::vector<std::uint32_t> token_counts = {1, 2, 4, 8, 16};
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

Options parse_options(int argc, char** argv) {
    Options options;
    bool explicit_tokens = false;

    for (int index = 1; index < argc; ++index) {
        const std::string_view argument(argv[index]);
        auto take_value = [&](std::string_view option) -> std::string_view {
            if (++index >= argc) {
                throw std::runtime_error(std::string(option) + " requires a value");
            }
            return argv[index];
        };

        if (argument == "--warmups") {
            options.warmups = parse_positive(take_value(argument), argument);
        } else if (argument == "--samples") {
            options.samples = parse_positive(take_value(argument), argument);
        } else if (argument == "--burst") {
            options.burst = parse_positive(take_value(argument), argument);
        } else if (argument == "--tokens") {
            const int parsed = parse_positive(take_value(argument), argument);
            if (parsed > static_cast<int>(kPaddedTokens) ||
                (parsed != 1 && parsed != 2 && parsed != 4 && parsed != 8 &&
                 parsed != 16)) {
                throw std::runtime_error(
                    "--tokens must be one of 1, 2, 4, 8, or 16");
            }
            if (!explicit_tokens) {
                options.token_counts.clear();
                explicit_tokens = true;
            }
            options.token_counts.push_back(static_cast<std::uint32_t>(parsed));
        } else if (argument == "--quick") {
            options.warmups = 1;
            options.samples = 3;
            options.burst = 2;
        } else if (argument == "--help" || argument == "-h") {
            std::cout
                << "Usage: rocwmma_routed_moe_bench [options]\n"
                << "  --warmups N  untimed operators per envelope (default 4)\n"
                << "  --samples N  wall-clock samples per envelope (default 9)\n"
                << "  --burst N    operators per sample/final sync (default 8)\n"
                << "  --tokens N   select token count; repeat to select several\n"
                << "  --quick      warmups=1, samples=3, burst=2\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + std::string(argument));
        }
    }

    std::sort(options.token_counts.begin(), options.token_counts.end());
    options.token_counts.erase(
        std::unique(options.token_counts.begin(), options.token_counts.end()),
        options.token_counts.end());
    return options;
}

class DeterministicRng {
  public:
    explicit DeterministicRng(std::uint64_t state) : state_(state) {}

    float symmetric(float magnitude) {
        state_ ^= state_ >> 12;
        state_ ^= state_ << 25;
        state_ ^= state_ >> 27;
        const std::uint64_t mixed = state_ * 0x2545F4914F6CDD1DULL;
        const std::uint32_t mantissa = static_cast<std::uint32_t>(mixed >> 40);
        const float unit = static_cast<float>(mantissa) / 16777216.0f;
        return (unit * 2.0f - 1.0f) * magnitude;
    }

  private:
    std::uint64_t state_;
};

struct HostData {
    PinnedBuffer<float> input{kPaddedTokens * kHidden};
    PinnedBuffer<RouteEntry> routes{kPaddedTokens * kTopK};
    PinnedBuffer<float> output{kPaddedTokens * kHidden};
    std::vector<Bf16> gate_up_weights;
    std::vector<Bf16> down_weights;
    std::vector<float> gate_up_weights_f32;
    std::vector<float> down_weights_f32;

    HostData()
        : gate_up_weights(static_cast<std::size_t>(kProjections) * kExperts *
                          kHidden * kFfn),
          down_weights(static_cast<std::size_t>(kExperts) * kFfn * kHidden),
          gate_up_weights_f32(gate_up_weights.size()),
          down_weights_f32(down_weights.size()) {
        DeterministicRng rng(0xBEE011D5A17EULL);

        for (std::size_t index = 0;
             index < static_cast<std::size_t>(kPaddedTokens) * kHidden;
             ++index) {
            input.get()[index] = rng.symmetric(0.25f);
            output.get()[index] = std::numeric_limits<float>::quiet_NaN();
        }

        for (std::uint32_t token = 0; token < kPaddedTokens; ++token) {
            float normalizer = 0.0f;
            std::array<float, kTopK> raw{};
            for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
                raw[rank] = 0.75f + 0.0625f *
                                       static_cast<float>((token + rank * 3) %
                                                          kTopK);
                normalizer += raw[rank];
            }
            for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
                RouteEntry& route = routes.get()[token * kTopK + rank];
                route.expert = (token * 5 + rank * 3) % kExperts;
                route.weight = raw[rank] / normalizer;
            }
        }

        auto fill_bf16 = [&](std::vector<Bf16>& destination,
                             std::vector<float>& expanded) {
            for (std::size_t index = 0; index < destination.size(); ++index) {
                destination[index] = Bf16(rng.symmetric(0.025f));
                expanded[index] = static_cast<float>(destination[index]);
            }
        };
        fill_bf16(gate_up_weights, gate_up_weights_f32);
        fill_bf16(down_weights, down_weights_f32);
    }
};

struct DeviceData {
    DeviceBuffer<float> input{kPaddedTokens * kHidden};
    DeviceBuffer<RouteEntry> routes{kPaddedTokens * kTopK};
    DeviceBuffer<Bf16> padded_input{kPaddedTokens * kHidden};
    DeviceBuffer<Bf16> gate_up_weights{
        static_cast<std::size_t>(kProjections) * kExperts * kHidden * kFfn};
    DeviceBuffer<Bf16> down_weights{
        static_cast<std::size_t>(kExperts) * kFfn * kHidden};
    DeviceBuffer<float> gate_up_output{
        static_cast<std::size_t>(kProjections) * kExperts * kPaddedTokens *
        kFfn};
    DeviceBuffer<Bf16> activation{
        static_cast<std::size_t>(kExperts) * kPaddedTokens * kFfn};
    DeviceBuffer<float> expert_output{
        static_cast<std::size_t>(kExperts) * kPaddedTokens * kHidden};
    DeviceBuffer<float> output{kPaddedTokens * kHidden};
};

void launch_pipeline(DeviceData& device,
                     std::uint32_t token_count,
                     hipStream_t stream) {
    constexpr std::uint32_t element_threads = 256;

    hipLaunchKernelGGL(f32_to_bf16_padded,
                       dim3((kPaddedTokens * kHidden + element_threads - 1) /
                            element_threads),
                       dim3(element_threads),
                       0,
                       stream,
                       device.input.get(),
                       device.padded_input.get(),
                       token_count);
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(gate_up_wmma,
                       dim3(kFfn / kTile, kExperts, kProjections),
                       dim3(kWaveSize),
                       0,
                       stream,
                       device.padded_input.get(),
                       device.gate_up_weights.get(),
                       device.gate_up_output.get());
    HIP_CHECK(hipGetLastError());

    constexpr std::uint32_t activation_elements =
        kExperts * kPaddedTokens * kFfn;
    hipLaunchKernelGGL(silu_multiply_to_bf16,
                       dim3((activation_elements + element_threads - 1) /
                            element_threads),
                       dim3(element_threads),
                       0,
                       stream,
                       device.gate_up_output.get(),
                       device.activation.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(down_wmma,
                       dim3(kHidden / kTile, kExperts),
                       dim3(kWaveSize),
                       0,
                       stream,
                       device.activation.get(),
                       device.down_weights.get(),
                       device.expert_output.get());
    HIP_CHECK(hipGetLastError());

    const std::uint32_t output_elements = token_count * kHidden;
    hipLaunchKernelGGL(weighted_route_reduce,
                       dim3((output_elements + element_threads - 1) /
                            element_threads),
                       dim3(element_threads),
                       0,
                       stream,
                       device.expert_output.get(),
                       device.routes.get(),
                       device.output.get(),
                       token_count);
    HIP_CHECK(hipGetLastError());
}

void enqueue_host_visible_operator(DeviceData& device,
                                   HostData& host,
                                   std::uint32_t token_count,
                                   hipStream_t stream) {
    HIP_CHECK(hipMemcpyAsync(device.input.get(),
                             host.input.get(),
                             static_cast<std::size_t>(token_count) * kHidden *
                                 sizeof(float),
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(device.routes.get(),
                             host.routes.get(),
                             static_cast<std::size_t>(token_count) * kTopK *
                                 sizeof(RouteEntry),
                             hipMemcpyHostToDevice,
                             stream));
    launch_pipeline(device, token_count, stream);
    HIP_CHECK(hipMemcpyAsync(host.output.get(),
                             device.output.get(),
                             static_cast<std::size_t>(token_count) * kHidden *
                                 sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
}

std::vector<float> cpu_reference(const HostData& host,
                                 bool round_wmma_input_boundaries) {
    std::vector<float> output(
        static_cast<std::size_t>(kPaddedTokens) * kHidden, 0.0f);
    std::vector<float> gate(kFfn);
    std::vector<float> up(kFfn);
    std::vector<float> activation(kFfn);
    std::vector<float> rounded_input(
        static_cast<std::size_t>(kPaddedTokens) * kHidden);

    if (round_wmma_input_boundaries) {
        for (std::size_t index = 0; index < rounded_input.size(); ++index) {
            rounded_input[index] =
                static_cast<float>(Bf16(host.input.get()[index]));
        }
    }

    const std::size_t projection_stride =
        static_cast<std::size_t>(kExperts) * kHidden * kFfn;
    const std::size_t expert_gate_stride =
        static_cast<std::size_t>(kHidden) * kFfn;
    const std::size_t expert_down_stride =
        static_cast<std::size_t>(kFfn) * kHidden;

    for (std::uint32_t token = 0; token < kPaddedTokens; ++token) {
        const float* input = round_wmma_input_boundaries
                                 ? rounded_input.data() + token * kHidden
                                 : host.input.get() + token * kHidden;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            const RouteEntry route = host.routes.get()[token * kTopK + rank];
            const std::size_t expert_gate_base =
                static_cast<std::size_t>(route.expert) * expert_gate_stride;
            const float* gate_weights =
                host.gate_up_weights_f32.data() + expert_gate_base;
            const float* up_weights = host.gate_up_weights_f32.data() +
                                      projection_stride + expert_gate_base;

            for (std::uint32_t column = 0; column < kFfn; ++column) {
                float gate_sum = 0.0f;
                float up_sum = 0.0f;
                for (std::uint32_t inner = 0; inner < kHidden; ++inner) {
                    const float input_value = input[inner];
                    gate_sum += input_value * gate_weights[inner * kFfn + column];
                    up_sum += input_value * up_weights[inner * kFfn + column];
                }
                gate[column] = gate_sum;
                up[column] = up_sum;
                const float silu_up =
                    (gate_sum / (1.0f + std::exp(-gate_sum))) * up_sum;
                activation[column] =
                    round_wmma_input_boundaries
                        ? static_cast<float>(Bf16(silu_up))
                        : silu_up;
            }

            const float* down_weights = host.down_weights_f32.data() +
                                        route.expert * expert_down_stride;
            float* token_output = output.data() + token * kHidden;
            for (std::uint32_t column = 0; column < kHidden; ++column) {
                float down_sum = 0.0f;
                for (std::uint32_t inner = 0; inner < kFfn; ++inner) {
                    down_sum += activation[inner] *
                                down_weights[inner * kHidden + column];
                }
                token_output[column] += route.weight * down_sum;
            }
        }
    }
    return output;
}

struct ErrorMetrics {
    double relative_l2;
    double max_absolute;
    double checksum;
    bool all_finite;
};

ErrorMetrics compare_output(const float* actual,
                            const std::vector<float>& reference,
                            std::uint32_t token_count) {
    double difference_squared = 0.0;
    double reference_squared = 0.0;
    double max_absolute = 0.0;
    double checksum = 0.0;
    bool all_finite = true;

    const std::size_t elements =
        static_cast<std::size_t>(token_count) * kHidden;
    for (std::size_t index = 0; index < elements; ++index) {
        const double observed = actual[index];
        const double expected = reference[index];
        all_finite = all_finite && std::isfinite(observed) &&
                     std::isfinite(expected);
        const double difference = observed - expected;
        difference_squared += difference * difference;
        reference_squared += expected * expected;
        max_absolute = std::max(max_absolute, std::abs(difference));
        checksum += observed * static_cast<double>((index % 17) + 1);
    }

    const double denominator =
        std::max(reference_squared, std::numeric_limits<double>::min());
    return {std::sqrt(difference_squared / denominator),
            max_absolute,
            checksum,
            all_finite};
}

struct TimingStats {
    double minimum_ms;
    double median_ms;
    double p90_ms;
};

TimingStats summarize(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t median_index = samples.size() / 2;
    const std::size_t p90_index =
        std::min(samples.size() - 1,
                 static_cast<std::size_t>(std::ceil(samples.size() * 0.9)) - 1);
    return {samples.front(), samples[median_index], samples[p90_index]};
}

template <typename Enqueue>
TimingStats benchmark_envelope(const Options& options,
                               hipStream_t stream,
                               Enqueue&& enqueue) {
    for (int index = 0; index < options.warmups; ++index) {
        enqueue();
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> samples;
    samples.reserve(options.samples);
    for (int sample = 0; sample < options.samples; ++sample) {
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto begin = std::chrono::steady_clock::now();
        for (int operation = 0; operation < options.burst; ++operation) {
            enqueue();
        }
        HIP_CHECK(hipStreamSynchronize(stream));
        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - begin).count();
        samples.push_back(elapsed_ms / static_cast<double>(options.burst));
    }
    return summarize(std::move(samples));
}

void print_result(std::uint32_t token_count,
                  std::string_view envelope,
                  const TimingStats& timing,
                  const ErrorMetrics& mechanical_error,
                  const ErrorMetrics& f32_policy_error,
                  bool bit_deterministic) {
    const double per_token_us = timing.median_ms * 1000.0 / token_count;
    const double routed_evals_per_second =
        1000.0 * static_cast<double>(token_count) / timing.median_ms;
    std::cout << token_count << ',' << envelope << ',' << std::fixed
              << std::setprecision(6) << timing.minimum_ms << ','
              << timing.median_ms << ',' << timing.p90_ms << ','
              << per_token_us << ',' << std::setprecision(2)
              << routed_evals_per_second << ',' << std::scientific
              << std::setprecision(6) << mechanical_error.relative_l2 << ','
              << mechanical_error.max_absolute << ','
              << f32_policy_error.relative_l2 << ','
              << f32_policy_error.max_absolute << ','
              << mechanical_error.checksum << ','
              << (bit_deterministic ? "true" : "false") << '\n';
}

std::size_t mib(std::size_t bytes) { return bytes / (1024 * 1024); }

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);

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
            throw std::runtime_error("gfx11 rocWMMA requires wave32; device reports " +
                                     std::to_string(properties.warpSize));
        }

        std::cout << "device=" << properties.name << " arch=" << properties.gcnArchName
                  << " warp=" << properties.warpSize << '\n';
        std::cout << "shape: H=" << kHidden << " F=" << kFfn
                  << " K=" << kTopK << " experts=" << kExperts
                  << " padded_tokens=" << kPaddedTokens << '\n';
        std::cout << "cohort: ideal high overlap; every token routes exactly once "
                     "through the same 8 resident experts (route order differs)\n";
        std::cout << "pipeline: launches/operator=5 "
                     "[f32_to_bf16,gate_up_wmma,silu_to_bf16,down_wmma,route_reduce]\n";
        std::cout << "resident envelope: one final stream sync/sample; excludes "
                     "allocation, weight upload, input/route H2D, output D2H\n";
        std::cout << "host_visible envelope: pinned H2D input + pinned H2D "
                     "route metadata + 5 kernels + pinned D2H output; one final "
                     "stream sync/sample\n";
        std::cout << "weights: persistent BF16 in one-time transposed row-major "
                     "WMMA-B layout (not checkpoint-native); preparation/upload "
                     "excluded; per-launch fragment loads/register packing included\n";
        std::cout << "timing: chrono wall clock; HIP events intentionally unused; "
                  << "warmups=" << options.warmups
                  << " samples=" << options.samples
                  << " burst=" << options.burst
                  << " envelope_order=alternating_by_N\n";

        HostData host;
        DeviceData device;
        Stream stream;

        HIP_CHECK(hipMemcpyAsync(device.gate_up_weights.get(),
                                 host.gate_up_weights.data(),
                                 device.gate_up_weights.size_bytes(),
                                 hipMemcpyHostToDevice,
                                 stream.get()));
        HIP_CHECK(hipMemcpyAsync(device.down_weights.get(),
                                 host.down_weights.data(),
                                 device.down_weights.size_bytes(),
                                 hipMemcpyHostToDevice,
                                 stream.get()));
        HIP_CHECK(hipMemcpyAsync(device.input.get(),
                                 host.input.get(),
                                 host.input.size_bytes(),
                                 hipMemcpyHostToDevice,
                                 stream.get()));
        HIP_CHECK(hipMemcpyAsync(device.routes.get(),
                                 host.routes.get(),
                                 host.routes.size_bytes(),
                                 hipMemcpyHostToDevice,
                                 stream.get()));
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        std::cout << "persistent_device_MiB="
                  << mib(device.gate_up_weights.size_bytes() +
                         device.down_weights.size_bytes() +
                         device.input.size_bytes() + device.routes.size_bytes() +
                         device.padded_input.size_bytes() +
                         device.gate_up_output.size_bytes() +
                         device.activation.size_bytes() +
                         device.expert_output.size_bytes() +
                         device.output.size_bytes())
                  << " persistent_weight_MiB="
                  << mib(device.gate_up_weights.size_bytes() +
                         device.down_weights.size_bytes())
                  << '\n';

        const auto mechanical_oracle_begin = std::chrono::steady_clock::now();
        const std::vector<float> mechanical_reference =
            cpu_reference(host, true);
        const auto mechanical_oracle_end = std::chrono::steady_clock::now();
        std::cout << "bf16_boundary_cpu_oracle_ms=" << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(
                         mechanical_oracle_end - mechanical_oracle_begin)
                         .count()
                  << " (F32 source input, BF16 WMMA input/activation boundaries, "
                     "BF16 weights, F32 accumulation)\n";
        const auto f32_oracle_begin = std::chrono::steady_clock::now();
        const std::vector<float> f32_policy_reference =
            cpu_reference(host, false);
        const auto f32_oracle_end = std::chrono::steady_clock::now();
        std::cout << "f32_policy_cpu_oracle_ms=" << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(f32_oracle_end -
                                                               f32_oracle_begin)
                         .count()
                  << " (BF16 weights, original F32 input and post-SiLU "
                     "activation, F32 accumulation)\n";

        std::cout << "N,envelope,min_batch_ms,median_batch_ms,p90_batch_ms,"
                     "median_us_per_routed_eval,routed_evals_per_second,"
                     "mechanical_relative_l2,mechanical_max_abs,"
                     "f32_policy_relative_l2,f32_policy_max_abs,checksum,"
                     "bit_deterministic\n";

        bool valid = true;
        std::size_t cohort_index = 0;
        for (const std::uint32_t token_count : options.token_counts) {
            launch_pipeline(device, token_count, stream.get());
            HIP_CHECK(hipMemcpyAsync(host.output.get(),
                                     device.output.get(),
                                     static_cast<std::size_t>(token_count) *
                                         kHidden * sizeof(float),
                                     hipMemcpyDeviceToHost,
                                     stream.get()));
            HIP_CHECK(hipStreamSynchronize(stream.get()));
            const ErrorMetrics mechanical_error = compare_output(
                host.output.get(), mechanical_reference, token_count);
            const ErrorMetrics f32_policy_error = compare_output(
                host.output.get(), f32_policy_reference, token_count);
            const std::size_t output_elements =
                static_cast<std::size_t>(token_count) * kHidden;
            const std::vector<float> first_output(
                host.output.get(), host.output.get() + output_elements);

            launch_pipeline(device, token_count, stream.get());
            HIP_CHECK(hipMemcpyAsync(host.output.get(),
                                     device.output.get(),
                                     output_elements * sizeof(float),
                                     hipMemcpyDeviceToHost,
                                     stream.get()));
            HIP_CHECK(hipStreamSynchronize(stream.get()));
            const bool bit_deterministic =
                std::memcmp(first_output.data(),
                            host.output.get(),
                            output_elements * sizeof(float)) == 0;
            constexpr double maximum_relative_l2 = 5.0e-4;
            constexpr double maximum_absolute = 1.0e-3;

            TimingStats resident{};
            TimingStats host_visible{};
            auto time_resident = [&] {
                resident = benchmark_envelope(
                    options,
                    stream.get(),
                    [&] { launch_pipeline(device, token_count, stream.get()); });
            };
            auto time_host_visible = [&] {
                host_visible = benchmark_envelope(
                    options,
                    stream.get(),
                    [&] {
                        enqueue_host_visible_operator(
                            device, host, token_count, stream.get());
                    });
            };
            if (cohort_index % 2 == 0) {
                time_resident();
                time_host_visible();
            } else {
                time_host_visible();
                time_resident();
            }

            // Exercise and validate the copy-inclusive path separately from the
            // device-resident result used above.
            enqueue_host_visible_operator(
                device, host, token_count, stream.get());
            HIP_CHECK(hipStreamSynchronize(stream.get()));
            const ErrorMetrics host_mechanical_error = compare_output(
                host.output.get(), mechanical_reference, token_count);
            const ErrorMetrics host_f32_policy_error = compare_output(
                host.output.get(), f32_policy_reference, token_count);
            const std::vector<float> first_host_output(
                host.output.get(), host.output.get() + output_elements);
            enqueue_host_visible_operator(
                device, host, token_count, stream.get());
            HIP_CHECK(hipStreamSynchronize(stream.get()));
            const bool host_bit_deterministic =
                std::memcmp(first_host_output.data(),
                            host.output.get(),
                            output_elements * sizeof(float)) == 0;

            const auto meets_error_gate = [&](const ErrorMetrics& metrics) {
                return metrics.all_finite &&
                       metrics.relative_l2 <= maximum_relative_l2 &&
                       metrics.max_absolute <= maximum_absolute;
            };
            const auto valid_policy_metric = [](const ErrorMetrics& metrics) {
                return metrics.all_finite &&
                       std::isfinite(metrics.relative_l2) &&
                       std::isfinite(metrics.max_absolute);
            };
            valid = valid && meets_error_gate(mechanical_error) &&
                    bit_deterministic &&
                    meets_error_gate(host_mechanical_error) &&
                    host_bit_deterministic &&
                    valid_policy_metric(f32_policy_error) &&
                    valid_policy_metric(host_f32_policy_error);

            print_result(token_count,
                         "resident",
                         resident,
                         mechanical_error,
                         f32_policy_error,
                         bit_deterministic);
            print_result(token_count,
                         "host_visible",
                         host_visible,
                         host_mechanical_error,
                         host_f32_policy_error,
                         host_bit_deterministic);
            ++cohort_index;
        }

        if (!valid) {
            std::cerr << "validation failed: both envelopes require finite output, "
                         "relative L2 <= 5e-4, max abs <= 1e-3, and bit-identical "
                         "repeated execution\n";
            return 2;
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
