// Batch-1 vector-dot candidate against the frozen real-weight Qwen3.6 WMMA control.
//
// Including the control source keeps its strict pack parser, route implementation,
// CPU oracle, resident weights, correctness metrics, and timing envelope intact.
// Only its host entry point is renamed; the validated source itself is not modified.

#define main q35_wmma_control_entry_not_called
#include "rocwmma_qwen35_resident_layer_bench.hip.cpp"
#undef main

namespace {

#if defined(__HIP_DEVICE_COMPILE__) && \
    !__has_builtin(__builtin_amdgcn_fdot2_f32_bf16)
#error "gfx1151 candidate requires __builtin_amdgcn_fdot2_f32_bf16"
#endif

using Bf16PairBits = short __attribute__((ext_vector_type(2)));

constexpr std::size_t kVectorGateUpElements =
    2ULL * kComputeSlots * kFfn;
constexpr std::size_t kVectorActivationElements = kComputeSlots * kFfn;
constexpr std::size_t kVectorExpertOutputElements =
    static_cast<std::size_t>(kComputeSlots) * kHidden;

static_assert(kHidden % kWaveSize == 0);
static_assert(kFfn % kWaveSize == 0);
static_assert(kHidden % (2 * kWaveSize) == 0);
static_assert(kFfn % (2 * kWaveSize) == 0);
static_assert(sizeof(Bf16PairBits) == 2 * sizeof(Bf16));

__device__ __forceinline__ Bf16PairBits
q35_load_bf16_pair(const Bf16* values) {
    Bf16PairBits pair;
    __builtin_memcpy(&pair, values, sizeof(pair));
    return pair;
}

__device__ __forceinline__ float q35_vector_wave_sum(float value) {
#pragma unroll
    for (std::uint32_t offset = kWaveSize / 2; offset != 0; offset /= 2) {
        value += __shfl_down(value, offset, kWaveSize);
    }
    return value;
}

// This is arithmetically identical to the control RMSNorm kernel for token zero,
// but it does not materialize the 15 all-zero rows required by a 16x16 WMMA tile.
extern "C" __global__ void q35_vector_rmsnorm_f32_to_f32_bf16(
    const float* hidden,
    const float* gamma,
    float* normalized,
    Bf16* vector_input) {
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
        vector_input[index] = Bf16(value);
    }
}

// One wave owns one output dot. At every load instruction its 32 lanes access a
// contiguous run of checkpoint-output-major BF16 weights and the same contiguous
// input run. The fixed wave32 shuffle tree makes accumulation deterministic.
extern "C" __global__ __launch_bounds__(kWaveSize)
void q35_selected_shared_gate_up_vector_dot(
    const Bf16* vector_input,
    const Bf16* gate_up_weights,
    const std::uint32_t* route_ids,
    float* gate_up_output) {
    const std::uint32_t output = blockIdx.x;
    const std::uint32_t slot = blockIdx.y;
    const std::uint32_t projection = blockIdx.z;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t routed = slot < kTopK ? route_ids[slot] : 0;
    const std::uint32_t expert =
        slot < kTopK ? (routed < kAllExperts ? routed : 0) : kSharedSlot;
    const Bf16* weights =
        gate_up_weights +
        (static_cast<std::size_t>(projection) * kPackedSlots + expert) *
            kExpertMatrixElements +
        static_cast<std::size_t>(output) * kHidden;

    float partial = 0.0f;
    for (std::uint32_t inner = 2 * lane; inner < kHidden;
         inner += 2 * kWaveSize) {
        partial = __builtin_amdgcn_fdot2_f32_bf16(
            q35_load_bf16_pair(vector_input + inner),
            q35_load_bf16_pair(weights + inner),
            partial,
            false);
    }
    partial = q35_vector_wave_sum(partial);
    if (lane == 0) {
        const std::size_t plane = kComputeSlots * kFfn;
        gate_up_output[static_cast<std::size_t>(projection) * plane +
                       static_cast<std::size_t>(slot) * kFfn + output] =
            partial;
    }
}

extern "C" __global__ void q35_vector_silu_multiply_to_bf16(
    const float* gate_up_output,
    Bf16* activation) {
    const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kVectorActivationElements) {
        return;
    }
    const float gate = gate_up_output[index];
    const float up = gate_up_output[kVectorActivationElements + index];
    const float silu = gate / (1.0f + __builtin_expf(-gate));
    activation[index] = Bf16(silu * up);
}

extern "C" __global__ __launch_bounds__(kWaveSize)
void q35_selected_shared_down_vector_dot(
    const Bf16* activation,
    const Bf16* down_weights,
    const std::uint32_t* route_ids,
    float* expert_output) {
    const std::uint32_t output = blockIdx.x;
    const std::uint32_t slot = blockIdx.y;
    const std::uint32_t lane = threadIdx.x;
    const std::uint32_t routed = slot < kTopK ? route_ids[slot] : 0;
    const std::uint32_t expert =
        slot < kTopK ? (routed < kAllExperts ? routed : 0) : kSharedSlot;
    const Bf16* slot_activation =
        activation + static_cast<std::size_t>(slot) * kFfn;
    const Bf16* weights =
        down_weights + static_cast<std::size_t>(expert) *
                           kExpertMatrixElements +
        static_cast<std::size_t>(output) * kFfn;

    float partial = 0.0f;
    for (std::uint32_t inner = 2 * lane; inner < kFfn;
         inner += 2 * kWaveSize) {
        partial = __builtin_amdgcn_fdot2_f32_bf16(
            q35_load_bf16_pair(slot_activation + inner),
            q35_load_bf16_pair(weights + inner),
            partial,
            false);
    }
    partial = q35_vector_wave_sum(partial);
    if (lane == 0) {
        expert_output[static_cast<std::size_t>(slot) * kHidden + output] =
            partial;
    }
}

// Preserve the control's rank-ordered routed sum, then shared expert, then residual.
extern "C" __global__ void q35_vector_route_shared_residual_reduce(
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
    float sum = 0.0f;
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        sum += route_weights[rank] *
               expert_output[static_cast<std::size_t>(rank) * kHidden +
                             column];
    }
    sum += *shared_scale *
           expert_output[static_cast<std::size_t>(kTopK) * kHidden + column];
    output[column] = residual[column] + sum;
}

struct VectorState {
    VectorState()
        : gate_up_output(kVectorGateUpElements),
          activation(kVectorActivationElements),
          expert_output(kVectorExpertOutputElements),
          output(kHidden) {}

    DeviceBuffer<float> gate_up_output;
    DeviceBuffer<Bf16> activation;
    DeviceBuffer<float> expert_output;
    DeviceBuffer<float> output;
};

std::size_t vector_state_bytes() {
    return kVectorGateUpElements * sizeof(float) +
           kVectorActivationElements * sizeof(Bf16) +
           kVectorExpertOutputElements * sizeof(float) +
           kHidden * sizeof(float);
}

void launch_vector_prefix(const ResidentLayer& layer,
                          ResidentState& state,
                          hipStream_t stream) {
    hipLaunchKernelGGL(q35_vector_rmsnorm_f32_to_f32_bf16,
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

void launch_vector_suffix(const ResidentLayer& layer,
                          ResidentState& state,
                          VectorState& vector,
                          hipStream_t stream) {
    hipLaunchKernelGGL(q35_selected_shared_gate_up_vector_dot,
                       dim3(kFfn, kComputeSlots, 2),
                       dim3(kWaveSize),
                       0,
                       stream,
                       state.padded_input.get(),
                       layer.gate_up.get(),
                       state.route_ids.get(),
                       vector.gate_up_output.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(
        q35_vector_silu_multiply_to_bf16,
        dim3((kVectorActivationElements + kElementThreads - 1) /
             kElementThreads),
        dim3(kElementThreads),
        0,
        stream,
        vector.gate_up_output.get(),
        vector.activation.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(q35_selected_shared_down_vector_dot,
                       dim3(kHidden, kComputeSlots),
                       dim3(kWaveSize),
                       0,
                       stream,
                       vector.activation.get(),
                       layer.down.get(),
                       state.route_ids.get(),
                       vector.expert_output.get());
    HIP_CHECK(hipGetLastError());

    hipLaunchKernelGGL(q35_vector_route_shared_residual_reduce,
                       dim3((kHidden + kElementThreads - 1) /
                            kElementThreads),
                       dim3(kElementThreads),
                       0,
                       stream,
                       vector.expert_output.get(),
                       state.route_status.get(),
                       state.route_weights.get(),
                       state.shared_scale.get(),
                       state.hidden.get(),
                       vector.output.get());
    HIP_CHECK(hipGetLastError());
}

void enqueue_vector(const ResidentLayer& layer,
                    ResidentState& state,
                    VectorState& vector,
                    hipStream_t stream) {
    launch_vector_prefix(layer, state, stream);
    launch_vector_suffix(layer, state, vector, stream);
}

std::vector<float> read_vector_output(VectorState& state,
                                      hipStream_t stream) {
    std::vector<float> output(kHidden);
    HIP_CHECK(hipMemcpyAsync(output.data(),
                             state.output.get(),
                             output.size() * sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    return output;
}

void poison_vector_scratch(VectorState& state, hipStream_t stream) {
    for (const auto [pointer, bytes] :
         std::array<std::pair<void*, std::size_t>, 4>{{
             {state.gate_up_output.get(), state.gate_up_output.size_bytes()},
             {state.activation.get(), state.activation.size_bytes()},
             {state.expert_output.get(), state.expert_output.size_bytes()},
             {state.output.get(), state.output.size_bytes()},
         }}) {
        HIP_CHECK(hipMemsetAsync(pointer, 0xa5, bytes, stream));
    }
}

bool route_ids_match(const ObservedRoute& route,
                     const CpuReference& reference) {
    if (route.status != 0) {
        return false;
    }
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        if (route.ids[rank] != reference.route_ids[rank]) {
            return false;
        }
    }
    return true;
}

bool route_metadata_bit_exact(const ObservedRoute& left,
                              const ObservedRoute& right) {
    return left.status == right.status &&
           std::memcmp(left.ids.data(),
                       right.ids.data(),
                       left.ids.size() * sizeof(left.ids[0])) == 0 &&
           std::memcmp(left.weights.data(),
                       right.weights.data(),
                       left.weights.size() * sizeof(left.weights[0])) == 0 &&
           std::memcmp(&left.shared_scale,
                       &right.shared_scale,
                       sizeof(left.shared_scale)) == 0 &&
           std::memcmp(&left.margin, &right.margin, sizeof(left.margin)) == 0;
}

void print_route(std::string_view label, const ObservedRoute& route) {
    std::cout << label << "_route_ids=";
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        if (rank != 0) {
            std::cout << ':';
        }
        std::cout << route.ids[rank];
    }
    std::cout << ' ' << label << "_route_status=" << route.status
              << ' ' << label << "_rank8_9_margin=" << std::scientific
              << route.margin << ' ' << label
              << "_shared_scale=" << route.shared_scale << '\n';
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
            throw std::runtime_error(
                "benchmark requires gfx1151; device reports " +
                std::string(architecture));
        }
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error(
                "gfx1151 vector candidate requires wave32; device reports " +
                std::to_string(properties.warpSize));
        }

        std::cout << "device=" << properties.name
                  << " arch=" << properties.gcnArchName
                  << " wave=" << properties.warpSize << '\n';
        std::cout << "pack=" << pack.path() << " bytes=" << kPackBytes
                  << " version=1 layer=3 H=2048 F=512 E=256 K=8 sharedF=512\n";
        std::cout << "weights=native_bf16 checkpoint_output_major "
                     "transpose=false resident=true\n";
        std::cout << "candidate=wave32_vector_dot batch=1 compact_rows=1 "
                     "control=validated_rocwmma padded_rows=16 "
                     "launches_each=7 one_external_stream_sync\n";
        std::cout << "throughput_unit=routed_layer_evals_per_second "
                     "not_full_model_tokens_per_second=true\n";

        require_memory_headroom();
        Stream stream;
        const auto upload_begin = std::chrono::steady_clock::now();
        ResidentLayer layer;
        ResidentState state;
        VectorState vector;
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
                  << " vector_state_bytes=" << vector_state_bytes()
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
        poison_vector_scratch(vector, stream.get());

        enqueue_resident(layer, state, stream.get());
        const std::vector<float> control_first =
            read_output(state, stream.get());
        const ObservedRoute control_route = read_route(state, stream.get());
        enqueue_resident(layer, state, stream.get());
        const std::vector<float> control_second =
            read_output(state, stream.get());
        const bool control_deterministic =
            std::memcmp(control_first.data(),
                        control_second.data(),
                        control_first.size() * sizeof(float)) == 0;

        enqueue_vector(layer, state, vector, stream.get());
        const std::vector<float> vector_first =
            read_vector_output(vector, stream.get());
        const ObservedRoute vector_route = read_route(state, stream.get());
        enqueue_vector(layer, state, vector, stream.get());
        const std::vector<float> vector_second =
            read_vector_output(vector, stream.get());
        const bool vector_deterministic =
            std::memcmp(vector_first.data(),
                        vector_second.data(),
                        vector_first.size() * sizeof(float)) == 0;

        const bool control_route_exact =
            route_ids_match(control_route, reference);
        const bool vector_route_exact = route_ids_match(vector_route, reference);
        const bool route_metadata_exact =
            route_metadata_bit_exact(control_route, vector_route);
        const ErrorMetrics control_oracle =
            compare_output(control_first, reference.output);
        const ErrorMetrics vector_oracle =
            compare_output(vector_first, reference.output);
        const ErrorMetrics vector_control =
            compare_output(vector_first, control_first);

        print_route("control", control_route);
        print_route("vector", vector_route);
        std::cout << "control_route_exact=" << std::boolalpha
                  << control_route_exact
                  << " vector_route_exact=" << vector_route_exact
                  << " route_metadata_bit_exact=" << route_metadata_exact
                  << '\n';
        std::cout << "control_relative_l2=" << control_oracle.relative_l2
                  << " control_max_abs=" << control_oracle.max_absolute
                  << " control_checksum=" << control_oracle.checksum
                  << " control_finite=" << control_oracle.all_finite
                  << " control_bit_deterministic=" << control_deterministic
                  << '\n';
        std::cout << "vector_relative_l2=" << vector_oracle.relative_l2
                  << " vector_max_abs=" << vector_oracle.max_absolute
                  << " vector_checksum=" << vector_oracle.checksum
                  << " vector_finite=" << vector_oracle.all_finite
                  << " vector_bit_deterministic=" << vector_deterministic
                  << '\n';
        std::cout << "vector_vs_control_relative_l2="
                  << vector_control.relative_l2
                  << " vector_vs_control_max_abs="
                  << vector_control.max_absolute
                  << " vector_vs_control_finite="
                  << vector_control.all_finite << '\n';

        for (int index = 0; index < options.warmups; ++index) {
            auto warm_vector = [&] {
                enqueue_vector(layer, state, vector, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
            };
            auto warm_control = [&] {
                enqueue_resident(layer, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
            };
            if (index % 2 == 0) {
                warm_vector();
                warm_control();
            } else {
                warm_control();
                warm_vector();
            }
        }

        std::vector<double> vector_samples;
        std::vector<double> control_samples;
        vector_samples.reserve(options.samples);
        control_samples.reserve(options.samples);
        for (int sample = 0; sample < options.samples; ++sample) {
            auto time_vector = [&] {
                vector_samples.push_back(timed_burst(
                    options.burst,
                    stream.get(),
                    [&] { enqueue_vector(layer, state, vector, stream.get()); }));
            };
            auto time_control = [&] {
                control_samples.push_back(timed_burst(
                    options.burst,
                    stream.get(),
                    [&] { enqueue_resident(layer, state, stream.get()); }));
            };
            if (sample % 2 == 0) {
                time_vector();
                time_control();
            } else {
                time_control();
                time_vector();
            }
        }
        const TimingStats vector_timing = summarize(std::move(vector_samples));
        const TimingStats control_timing =
            summarize(std::move(control_samples));
        print_timing("vector", vector_timing);
        print_timing("wmma_control", control_timing);
        const double median_speedup =
            control_timing.median_ms / vector_timing.median_ms;
        std::cout << "vector_over_wmma_median_speedup=" << std::fixed
                  << std::setprecision(4) << median_speedup
                  << " vector_routed_layer_evals_per_second="
                  << 1000.0 / vector_timing.median_ms
                  << " wmma_routed_layer_evals_per_second="
                  << 1000.0 / control_timing.median_ms
                  << " warmups=" << options.warmups
                  << " samples=" << options.samples
                  << " burst=" << options.burst << '\n';

        constexpr double maximum_relative_l2 = 5.0e-4;
        constexpr double maximum_absolute = 1.0e-3;
        const bool errors_pass =
            control_oracle.all_finite && vector_oracle.all_finite &&
            vector_control.all_finite &&
            control_oracle.relative_l2 <= maximum_relative_l2 &&
            control_oracle.max_absolute <= maximum_absolute &&
            vector_oracle.relative_l2 <= maximum_relative_l2 &&
            vector_oracle.max_absolute <= maximum_absolute &&
            vector_control.relative_l2 <= maximum_relative_l2 &&
            vector_control.max_absolute <= maximum_absolute;
        if (!control_route_exact || !vector_route_exact ||
            !route_metadata_exact || !control_deterministic ||
            !vector_deterministic || !errors_pass) {
            std::cerr << "FAIL: vector/control resident-layer gate failed\n";
            return 2;
        }
        std::cout << "PASS: exact routes and route metadata, deterministic "
                     "bits, finite output, and vector/control/oracle thresholds\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
