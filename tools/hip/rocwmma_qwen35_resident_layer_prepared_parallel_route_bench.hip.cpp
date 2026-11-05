// Combined real-weight Qwen3.5/3.6 experiment: the already-qualified
// bit-preserving prepared-row-major WMMA path with either the frozen serial
// router (control) or the qualified deterministic 256-thread router
// (candidate). The original harnesses remain byte-for-byte untouched.

#define main q35_frozen_resident_layer_main
#define q35_stable_softmax_topk8 q35_stable_softmax_topk8_prepared_serial
#include "rocwmma_qwen35_resident_layer_bench.hip.cpp"
#undef q35_stable_softmax_topk8
#undef main

namespace {

constexpr std::uint32_t kCombinedRouteThreads = kAllExperts;
static_assert(kCombinedRouteThreads == 256);
static_assert(kCombinedRouteThreads == kElementThreads);

__device__ __forceinline__ bool q35_combined_route_pair_better(
    float candidate_value,
    std::uint32_t candidate_id,
    float incumbent_value,
    std::uint32_t incumbent_id) {
    return candidate_value > incumbent_value ||
           (candidate_value == incumbent_value &&
            candidate_id < incumbent_id);
}

extern "C" __global__ __launch_bounds__(kCombinedRouteThreads)
void q35_prepared_stable_softmax_topk8_parallel(
    const float* logits,
    const float* shared_logit,
    float* probabilities,
    std::uint32_t* route_ids,
    float* route_weights,
    float* shared_scale,
    float* route_margin,
    std::uint32_t* route_status) {
    __shared__ float reduction_values[kCombinedRouteThreads];
    __shared__ std::uint32_t reduction_ids[kCombinedRouteThreads];
    __shared__ float denominator;
    __shared__ std::uint32_t status;

    const std::uint32_t expert = threadIdx.x;
    if (expert == 0) {
        status = 0;
        // Fail closed before any possible early exit: later expert gathers are
        // already enqueued, so every route ID must always remain in bounds.
        *route_status = 1;
        *shared_scale = 0.0f;
        *route_margin = 0.0f;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_ids[rank] = 0;
            route_weights[rank] = 0.0f;
        }
    }
    __syncthreads();

    const float logit = logits[expert];
    if (!__builtin_isfinite(logit)) {
        atomicOr(&status, 1U);
    }
    reduction_values[expert] =
        __builtin_isfinite(logit)
            ? logit
            : -std::numeric_limits<float>::infinity();
    reduction_ids[expert] = expert;
    __syncthreads();
    for (std::uint32_t stride = kCombinedRouteThreads / 2; stride != 0;
         stride /= 2) {
        if (expert < stride) {
            const float candidate_value = reduction_values[expert + stride];
            const std::uint32_t candidate_id =
                reduction_ids[expert + stride];
            if (q35_combined_route_pair_better(candidate_value,
                                               candidate_id,
                                               reduction_values[expert],
                                               reduction_ids[expert])) {
                reduction_values[expert] = candidate_value;
                reduction_ids[expert] = candidate_id;
            }
        }
        __syncthreads();
    }
    if (expert == 0 && status == 0 &&
        !__builtin_isfinite(*shared_logit)) {
        status = 2;
    }
    __syncthreads();
    if (status != 0) {
        if (expert == 0) {
            *route_status = status;
        }
        return;
    }

    const float maximum = reduction_values[0];
    const float unnormalized = __builtin_expf(logit - maximum);
    reduction_values[expert] = unnormalized;
    __syncthreads();
    for (std::uint32_t stride = kCombinedRouteThreads / 2; stride != 0;
         stride /= 2) {
        if (expert < stride) {
            reduction_values[expert] += reduction_values[expert + stride];
        }
        __syncthreads();
    }
    if (expert == 0) {
        denominator = reduction_values[0];
        if (!__builtin_isfinite(denominator) || !(denominator > 0.0f)) {
            status = 2;
        }
    }
    __syncthreads();
    if (status != 0) {
        if (expert == 0) {
            *route_status = status;
        }
        return;
    }

    const float probability = unnormalized / denominator;
    probabilities[expert] = probability;
    bool selected = false;
    float eighth = 0.0f;
    float ninth = 0.0f;
    for (std::uint32_t rank = 0; rank <= kTopK; ++rank) {
        reduction_values[expert] = selected ? -1.0f : probability;
        reduction_ids[expert] = expert;
        __syncthreads();
        for (std::uint32_t stride = kCombinedRouteThreads / 2; stride != 0;
             stride /= 2) {
            if (expert < stride) {
                const float candidate_value =
                    reduction_values[expert + stride];
                const std::uint32_t candidate_id =
                    reduction_ids[expert + stride];
                if (q35_combined_route_pair_better(
                        candidate_value,
                        candidate_id,
                        reduction_values[expert],
                        reduction_ids[expert])) {
                    reduction_values[expert] = candidate_value;
                    reduction_ids[expert] = candidate_id;
                }
            }
            __syncthreads();
        }
        const std::uint32_t winner = reduction_ids[0];
        const float winner_probability = reduction_values[0];
        if (expert == 0) {
            if (rank < kTopK) {
                route_ids[rank] = winner;
                route_weights[rank] = winner_probability;
                if (rank == kTopK - 1) {
                    eighth = winner_probability;
                }
            } else {
                ninth = winner_probability;
            }
        }
        selected = selected || expert == winner;
        __syncthreads();
    }

    if (expert == 0) {
        float selected_sum = 0.0f;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            selected_sum += route_weights[rank];
        }
        if (!__builtin_isfinite(selected_sum) || !(selected_sum > 0.0f)) {
            *route_status = 3;
            return;
        }
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_weights[rank] /= selected_sum;
        }
        const float scale =
            1.0f / (1.0f + __builtin_expf(-*shared_logit));
        *shared_scale = scale;
        *route_margin = eighth - ninth;
        *route_status = __builtin_isfinite(scale) ? 0 : 4;
    }
}

void launch_prepared_parallel_prefix(const ResidentLayer& layer,
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
    hipLaunchKernelGGL(q35_prepared_stable_softmax_topk8_parallel,
                       dim3(1),
                       dim3(kCombinedRouteThreads),
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

// These preparation and row-major Matrix-B kernels intentionally preserve the
// already-qualified prepared-path arithmetic and indexing verbatim.
extern "C" __global__ void q35_combined_prepare_gate_up_row_major(
    const Bf16* checkpoint,
    Bf16* prepared) {
    const std::size_t matrix = blockIdx.y;
    const std::size_t local =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (matrix >= 2ULL * kPackedSlots || local >= kExpertMatrixElements) {
        return;
    }
    const std::size_t output = local / kHidden;
    const std::size_t input = local % kHidden;
    prepared[matrix * kExpertMatrixElements + input * kFfn + output] =
        checkpoint[matrix * kExpertMatrixElements + local];
}

extern "C" __global__ void q35_combined_prepare_down_row_major(
    const Bf16* checkpoint,
    Bf16* prepared) {
    const std::size_t matrix = blockIdx.y;
    const std::size_t local =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (matrix >= kPackedSlots || local >= kExpertMatrixElements) {
        return;
    }
    const std::size_t output = local / kFfn;
    const std::size_t input = local % kFfn;
    prepared[matrix * kExpertMatrixElements + input * kHidden + output] =
        checkpoint[matrix * kExpertMatrixElements + local];
}

extern "C" __global__ __launch_bounds__(kWaveSize)
void q35_combined_prepared_gate_up_wmma(
    const Bf16* padded_input,
    const Bf16* prepared_gate_up,
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
                                    rocwmma::row_major>;
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
    const Bf16* weights =
        prepared_gate_up +
        (static_cast<std::size_t>(projection) * kPackedSlots + expert) *
            kExpertMatrixElements;

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
                                      static_cast<std::size_t>(inner) * kFfn +
                                      output_column,
                                  kFfn);
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

extern "C" __global__ __launch_bounds__(kWaveSize)
void q35_combined_prepared_down_wmma(
    const Bf16* activation,
    const Bf16* prepared_down,
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
                                    rocwmma::row_major>;
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
        prepared_down + static_cast<std::size_t>(expert) *
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
                                      static_cast<std::size_t>(inner) *
                                          kHidden +
                                      output_column,
                                  kHidden);
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

struct CombinedPreparedLayer {
    CombinedPreparedLayer() : gate_up(kGateUpElements), down(kDownElements) {}

    DeviceBuffer<Bf16> gate_up;
    DeviceBuffer<Bf16> down;
};

void require_combined_memory_headroom() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t checkpoint_bytes =
        kNormBytes + kRouterBytes + kSharedGateBytes + kGateUpBytes +
        kDownBytes;
    const std::size_t required =
        checkpoint_bytes + kGateUpBytes + kDownBytes + state_bytes();
    if (required > std::numeric_limits<std::size_t>::max() -
                       kHeadroomBytes ||
        free_bytes < required + kHeadroomBytes) {
        throw std::runtime_error(
            "insufficient VRAM for checkpoint plus prepared weights and "
            "512 MiB headroom");
    }
    std::cout << "vram_total_MiB=" << total_bytes / (1024 * 1024)
              << " vram_free_before_MiB=" << free_bytes / (1024 * 1024)
              << " combined_required_MiB=" << required / (1024 * 1024)
              << " production_prepared_weight_MiB="
              << checkpoint_bytes / (1024 * 1024)
              << " reserved_headroom_MiB=" << kHeadroomBytes / (1024 * 1024)
              << '\n';
}

void prepare_combined_row_major(const ResidentLayer& checkpoint,
                                CombinedPreparedLayer& prepared,
                                hipStream_t stream) {
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks =
        (kExpertMatrixElements + threads - 1) / threads;
    hipLaunchKernelGGL(q35_combined_prepare_gate_up_row_major,
                       dim3(blocks, 2 * kPackedSlots),
                       dim3(threads),
                       0,
                       stream,
                       checkpoint.gate_up.get(),
                       prepared.gate_up.get());
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(q35_combined_prepare_down_row_major,
                       dim3(blocks, kPackedSlots),
                       dim3(threads),
                       0,
                       stream,
                       checkpoint.down.get(),
                       prepared.down.get());
    HIP_CHECK(hipGetLastError());
}

void launch_combined_prepared_suffix(const CombinedPreparedLayer& prepared,
                                     ResidentState& state,
                                     hipStream_t stream) {
    hipLaunchKernelGGL(q35_combined_prepared_gate_up_wmma,
                       dim3(kFfn / kTile, kComputeSlots, 2),
                       dim3(kWaveSize),
                       0,
                       stream,
                       state.padded_input.get(),
                       prepared.gate_up.get(),
                       state.route_ids.get(),
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
    hipLaunchKernelGGL(q35_combined_prepared_down_wmma,
                       dim3(kHidden / kTile, kComputeSlots),
                       dim3(kWaveSize),
                       0,
                       stream,
                       state.activation.get(),
                       prepared.down.get(),
                       state.route_ids.get(),
                       state.expert_output.get());
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(q35_route_shared_residual_reduce,
                       dim3((kHidden + kElementThreads - 1) /
                            kElementThreads),
                       dim3(kElementThreads),
                       0,
                       stream,
                       state.expert_output.get(),
                       state.route_status.get(),
                       state.route_weights.get(),
                       state.shared_scale.get(),
                       state.hidden.get(),
                       state.output.get());
    HIP_CHECK(hipGetLastError());
}

void enqueue_prepared_serial(const ResidentLayer& checkpoint,
                             const CombinedPreparedLayer& prepared,
                             ResidentState& state,
                             hipStream_t stream) {
    launch_prefix(checkpoint, state, stream);
    launch_combined_prepared_suffix(prepared, state, stream);
}

void enqueue_prepared_parallel(const ResidentLayer& checkpoint,
                               const CombinedPreparedLayer& prepared,
                               ResidentState& state,
                               hipStream_t stream) {
    launch_prepared_parallel_prefix(checkpoint, state, stream);
    launch_combined_prepared_suffix(prepared, state, stream);
}

bool validate_combined_selected_bytes(
    const MappedPack& pack,
    const CombinedPreparedLayer& prepared,
    const std::array<std::uint32_t, kTopK>& expected_route_ids,
    hipStream_t stream) {
    std::vector<Bf16> host(kExpertMatrixElements);
    for (std::uint32_t slot = 0; slot < kComputeSlots; ++slot) {
        const std::uint32_t expert =
            slot < kTopK ? expected_route_ids[slot] : kSharedSlot;
        if (expert >= kPackedSlots) {
            return false;
        }
        for (std::uint32_t projection = 0; projection < 2; ++projection) {
            const std::size_t matrix =
                static_cast<std::size_t>(projection) * kPackedSlots + expert;
            HIP_CHECK(hipMemcpyAsync(
                host.data(),
                prepared.gate_up.get() + matrix * kExpertMatrixElements,
                host.size() * sizeof(Bf16),
                hipMemcpyDeviceToHost,
                stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            const Bf16* source =
                pack.gate_up() + matrix * kExpertMatrixElements;
            for (std::uint32_t output = 0; output < kFfn; ++output) {
                for (std::uint32_t input = 0; input < kHidden; ++input) {
                    const Bf16 expected =
                        source[static_cast<std::size_t>(output) * kHidden +
                               input];
                    const Bf16 observed =
                        host[static_cast<std::size_t>(input) * kFfn + output];
                    if (std::memcmp(&expected, &observed, sizeof(Bf16)) != 0) {
                        return false;
                    }
                }
            }
        }

        HIP_CHECK(hipMemcpyAsync(
            host.data(),
            prepared.down.get() +
                static_cast<std::size_t>(expert) * kExpertMatrixElements,
            host.size() * sizeof(Bf16),
            hipMemcpyDeviceToHost,
            stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        const Bf16* source =
            pack.down() + static_cast<std::size_t>(expert) *
                              kExpertMatrixElements;
        for (std::uint32_t output = 0; output < kHidden; ++output) {
            for (std::uint32_t input = 0; input < kFfn; ++input) {
                const Bf16 expected =
                    source[static_cast<std::size_t>(output) * kFfn + input];
                const Bf16 observed =
                    host[static_cast<std::size_t>(input) * kHidden + output];
                if (std::memcmp(&expected, &observed, sizeof(Bf16)) != 0) {
                    return false;
                }
            }
        }
    }
    return true;
}

bool combined_route_bit_exact(const ObservedRoute& left,
                              const ObservedRoute& right) {
    return std::memcmp(left.ids.data(),
                       right.ids.data(),
                       left.ids.size() * sizeof(left.ids[0])) == 0 &&
           std::memcmp(left.weights.data(),
                       right.weights.data(),
                       left.weights.size() * sizeof(left.weights[0])) == 0 &&
           std::memcmp(&left.shared_scale,
                       &right.shared_scale,
                       sizeof(left.shared_scale)) == 0 &&
           std::memcmp(&left.margin, &right.margin, sizeof(left.margin)) == 0 &&
           left.status == right.status;
}

void print_combined_route(std::string_view label,
                          const ObservedRoute& route) {
    std::cout << label << "_route_ids=";
    for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
        if (rank != 0) {
            std::cout << ':';
        }
        std::cout << route.ids[rank];
    }
    std::cout << ' ' << label << "_status=" << route.status << ' ' << label
              << "_margin=" << std::scientific << route.margin << ' ' << label
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
                "prepared parallel-route benchmark requires gfx1151; "
                "device reports " +
                std::string(architecture));
        }
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error(
                "gfx1151 rocWMMA requires wave32; device reports " +
                std::to_string(properties.warpSize));
        }

        std::cout << "device=" << properties.name
                  << " arch=" << properties.gcnArchName
                  << " wave=" << properties.warpSize << '\n';
        std::cout << "pack=" << pack.path()
                  << " bytes=" << kPackBytes
                  << " version=1 layer=3 H=2048 F=512 E=256 K=8 sharedF=512\n";
        std::cout << "control=prepared_row_major_serial_router "
                     "candidate=prepared_row_major_parallel_router "
                     "shared_weight_bytes=true shared_other_six_kernels=true "
                     "resident=true one_external_sync_per_burst=true\n";

        require_combined_memory_headroom();
        Stream stream;
        ResidentLayer checkpoint;
        CombinedPreparedLayer prepared;
        ResidentState state;
        upload_layer(pack, checkpoint, stream.get());
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        const auto prepare_begin = std::chrono::steady_clock::now();
        prepare_combined_row_major(checkpoint, prepared, stream.get());
        HIP_CHECK(hipStreamSynchronize(stream.get()));
        const auto prepare_end = std::chrono::steady_clock::now();
        std::cout << "preparation_ms=" << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(prepare_end -
                                                               prepare_begin)
                         .count()
                  << " excluded_from_inference_timing=true "
                     "bit_rearrangement_only=true\n";

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

        const CpuReference reference = cpu_reference(pack, hidden);
        const bool selected_and_shared_prepared_bytes_exact =
            validate_combined_selected_bytes(
                pack, prepared, reference.route_ids, stream.get());

        poison_scratch(state, stream.get());
        enqueue_resident(checkpoint, state, stream.get());
        const std::vector<float> native_serial =
            read_output(state, stream.get());
        const ObservedRoute native_route = read_route(state, stream.get());

        poison_scratch(state, stream.get());
        enqueue_prepared_serial(checkpoint, prepared, state, stream.get());
        const std::vector<float> serial_first =
            read_output(state, stream.get());
        const ObservedRoute serial_route_first =
            read_route(state, stream.get());
        enqueue_prepared_serial(checkpoint, prepared, state, stream.get());
        const std::vector<float> serial_second =
            read_output(state, stream.get());
        const ObservedRoute serial_route_second =
            read_route(state, stream.get());

        poison_scratch(state, stream.get());
        enqueue_prepared_parallel(checkpoint, prepared, state, stream.get());
        const std::vector<float> parallel_first =
            read_output(state, stream.get());
        const ObservedRoute parallel_route_first =
            read_route(state, stream.get());
        enqueue_prepared_parallel(checkpoint, prepared, state, stream.get());
        const std::vector<float> parallel_second =
            read_output(state, stream.get());
        const ObservedRoute parallel_route_second =
            read_route(state, stream.get());

        const bool native_prepared_serial_bit_exact =
            std::memcmp(native_serial.data(),
                        serial_first.data(),
                        native_serial.size() * sizeof(float)) == 0;
        const bool serial_output_deterministic =
            std::memcmp(serial_first.data(),
                        serial_second.data(),
                        serial_first.size() * sizeof(float)) == 0;
        const bool parallel_output_deterministic =
            std::memcmp(parallel_first.data(),
                        parallel_second.data(),
                        parallel_first.size() * sizeof(float)) == 0;
        const bool serial_route_deterministic =
            combined_route_bit_exact(serial_route_first, serial_route_second);
        const bool parallel_route_deterministic = combined_route_bit_exact(
            parallel_route_first, parallel_route_second);

        bool route_ids_exact = native_route.status == 0 &&
                               serial_route_first.status == 0 &&
                               parallel_route_first.status == 0;
        double route_weight_max_abs = 0.0;
        double serial_cpu_route_weight_max_abs = 0.0;
        double parallel_cpu_route_weight_max_abs = 0.0;
        double serial_route_weight_sum = 0.0;
        double parallel_route_weight_sum = 0.0;
        bool route_values_valid =
            std::isfinite(serial_route_first.shared_scale) &&
            std::isfinite(parallel_route_first.shared_scale) &&
            std::isfinite(serial_route_first.margin) &&
            std::isfinite(parallel_route_first.margin);
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_ids_exact =
                route_ids_exact &&
                native_route.ids[rank] == reference.route_ids[rank] &&
                serial_route_first.ids[rank] == reference.route_ids[rank] &&
                parallel_route_first.ids[rank] == reference.route_ids[rank];
            route_weight_max_abs =
                std::max(route_weight_max_abs,
                         std::abs(static_cast<double>(
                                      parallel_route_first.weights[rank]) -
                                  serial_route_first.weights[rank]));
            serial_cpu_route_weight_max_abs =
                std::max(serial_cpu_route_weight_max_abs,
                         std::abs(static_cast<double>(
                                      serial_route_first.weights[rank]) -
                                  reference.route_weights[rank]));
            parallel_cpu_route_weight_max_abs =
                std::max(parallel_cpu_route_weight_max_abs,
                         std::abs(static_cast<double>(
                                      parallel_route_first.weights[rank]) -
                                  reference.route_weights[rank]));
            serial_route_weight_sum += serial_route_first.weights[rank];
            parallel_route_weight_sum += parallel_route_first.weights[rank];
            route_values_valid =
                route_values_valid &&
                std::isfinite(serial_route_first.weights[rank]) &&
                std::isfinite(parallel_route_first.weights[rank]) &&
                serial_route_first.weights[rank] >= 0.0f &&
                parallel_route_first.weights[rank] >= 0.0f;
            if (rank != 0) {
                route_values_valid =
                    route_values_valid &&
                    serial_route_first.weights[rank - 1] >=
                        serial_route_first.weights[rank] &&
                    parallel_route_first.weights[rank - 1] >=
                        parallel_route_first.weights[rank];
            }
        }
        const double route_margin_abs =
            std::abs(static_cast<double>(parallel_route_first.margin) -
                     serial_route_first.margin);
        const double shared_scale_abs =
            std::abs(static_cast<double>(parallel_route_first.shared_scale) -
                     serial_route_first.shared_scale);
        const double serial_cpu_route_margin_abs =
            std::abs(static_cast<double>(serial_route_first.margin) -
                     reference.route_margin);
        const double parallel_cpu_route_margin_abs =
            std::abs(static_cast<double>(parallel_route_first.margin) -
                     reference.route_margin);
        const double serial_cpu_shared_scale_abs =
            std::abs(static_cast<double>(serial_route_first.shared_scale) -
                     reference.shared_scale);
        const double parallel_cpu_shared_scale_abs =
            std::abs(static_cast<double>(parallel_route_first.shared_scale) -
                     reference.shared_scale);

        const ErrorMetrics native_cpu =
            compare_output(native_serial, reference.output);
        const ErrorMetrics serial_cpu =
            compare_output(serial_first, reference.output);
        const ErrorMetrics parallel_cpu =
            compare_output(parallel_first, reference.output);
        const ErrorMetrics parallel_serial =
            compare_output(parallel_first, serial_first);

        print_combined_route("serial_prepared", serial_route_first);
        print_combined_route("parallel_prepared", parallel_route_first);
        std::cout << "selected_and_shared_prepared_bytes_exact="
                  << std::boolalpha
                  << selected_and_shared_prepared_bytes_exact
                  << " route_ids_exact=" << route_ids_exact
                  << " native_prepared_serial_bit_exact="
                  << native_prepared_serial_bit_exact << '\n';
        std::cout << "route_weight_max_abs=" << std::scientific
                  << route_weight_max_abs
                  << " route_margin_abs=" << route_margin_abs
                  << " shared_scale_abs=" << shared_scale_abs << '\n';
        std::cout << "serial_cpu_route_weight_max_abs="
                  << serial_cpu_route_weight_max_abs
                  << " parallel_cpu_route_weight_max_abs="
                  << parallel_cpu_route_weight_max_abs
                  << " serial_cpu_route_margin_abs="
                  << serial_cpu_route_margin_abs
                  << " parallel_cpu_route_margin_abs="
                  << parallel_cpu_route_margin_abs
                  << " serial_cpu_shared_scale_abs="
                  << serial_cpu_shared_scale_abs
                  << " parallel_cpu_shared_scale_abs="
                  << parallel_cpu_shared_scale_abs
                  << " serial_route_weight_sum=" << serial_route_weight_sum
                  << " parallel_route_weight_sum="
                  << parallel_route_weight_sum
                  << " route_values_finite_normalized_ordered="
                  << route_values_valid << '\n';
        std::cout << "native_cpu_relative_l2=" << native_cpu.relative_l2
                  << " serial_cpu_relative_l2=" << serial_cpu.relative_l2
                  << " parallel_cpu_relative_l2="
                  << parallel_cpu.relative_l2
                  << " parallel_cpu_max_abs=" << parallel_cpu.max_absolute
                  << " parallel_serial_relative_l2="
                  << parallel_serial.relative_l2
                  << " parallel_serial_max_abs="
                  << parallel_serial.max_absolute << '\n';
        std::cout << "serial_output_bit_deterministic="
                  << serial_output_deterministic
                  << " serial_route_bit_deterministic="
                  << serial_route_deterministic
                  << " parallel_output_bit_deterministic="
                  << parallel_output_deterministic
                  << " parallel_route_bit_deterministic="
                  << parallel_route_deterministic
                  << " serial_checksum=" << serial_cpu.checksum
                  << " parallel_checksum=" << parallel_cpu.checksum << '\n';

        for (int warmup = 0; warmup < options.warmups; ++warmup) {
            if (warmup % 2 == 0) {
                enqueue_prepared_serial(
                    checkpoint, prepared, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
                enqueue_prepared_parallel(
                    checkpoint, prepared, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
            } else {
                enqueue_prepared_parallel(
                    checkpoint, prepared, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
                enqueue_prepared_serial(
                    checkpoint, prepared, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
            }
        }

        std::vector<double> serial_samples;
        std::vector<double> parallel_samples;
        std::vector<double> paired_speedups;
        serial_samples.reserve(options.samples);
        parallel_samples.reserve(options.samples);
        paired_speedups.reserve(options.samples);
        for (int sample = 0; sample < options.samples; ++sample) {
            double serial_ms = 0.0;
            double parallel_ms = 0.0;
            auto time_serial = [&] {
                serial_ms = timed_burst(options.burst,
                                        stream.get(),
                                        [&] {
                                            enqueue_prepared_serial(
                                                checkpoint,
                                                prepared,
                                                state,
                                                stream.get());
                                        });
            };
            auto time_parallel = [&] {
                parallel_ms = timed_burst(options.burst,
                                          stream.get(),
                                          [&] {
                                              enqueue_prepared_parallel(
                                                  checkpoint,
                                                  prepared,
                                                  state,
                                                  stream.get());
                                          });
            };
            if (sample % 2 == 0) {
                time_serial();
                time_parallel();
            } else {
                time_parallel();
                time_serial();
            }
            serial_samples.push_back(serial_ms);
            parallel_samples.push_back(parallel_ms);
            paired_speedups.push_back(serial_ms / parallel_ms);
        }

        const TimingStats serial_timing = summarize(serial_samples);
        const TimingStats parallel_timing = summarize(parallel_samples);
        const TimingStats paired = summarize(paired_speedups);
        print_timing("prepared_serial_route_layer", serial_timing);
        print_timing("prepared_parallel_route_layer", parallel_timing);
        std::cout << "paired_speedup_min=" << std::fixed
                  << std::setprecision(6) << paired.minimum_ms
                  << " paired_speedup_median=" << paired.median_ms
                  << " paired_speedup_p90=" << paired.p90_ms
                  << " fixed_route_moe_ffn_sublayers_per_second="
                  << 1000.0 / parallel_timing.median_ms
                  << " warmups=" << options.warmups
                  << " samples=" << options.samples
                  << " burst=" << options.burst << '\n';

        constexpr double kCpuRelativeL2Limit = 5.0e-4;
        constexpr double kCpuMaxAbsoluteLimit = 1.0e-3;
        constexpr double kRouteWeightMaxAbsoluteLimit = 2.0e-6;
        constexpr double kRouteMarginMaxAbsoluteLimit = 2.0e-6;
        constexpr double kSharedScaleMaxAbsoluteLimit = 1.0e-7;
        constexpr double kParallelSerialRelativeL2Limit = 2.0e-6;
        constexpr double kParallelSerialMaxAbsoluteLimit = 2.0e-5;
        constexpr double kRouteWeightSumLimit = 2.0e-6;
        const bool pass =
            selected_and_shared_prepared_bytes_exact && route_ids_exact &&
            native_prepared_serial_bit_exact && native_cpu.all_finite &&
            serial_cpu.all_finite && parallel_cpu.all_finite &&
            parallel_serial.all_finite && serial_output_deterministic &&
            parallel_output_deterministic && serial_route_deterministic &&
            parallel_route_deterministic &&
            native_cpu.relative_l2 <= kCpuRelativeL2Limit &&
            native_cpu.max_absolute <= kCpuMaxAbsoluteLimit &&
            serial_cpu.relative_l2 <= kCpuRelativeL2Limit &&
            serial_cpu.max_absolute <= kCpuMaxAbsoluteLimit &&
            parallel_cpu.relative_l2 <= kCpuRelativeL2Limit &&
            parallel_cpu.max_absolute <= kCpuMaxAbsoluteLimit &&
            route_weight_max_abs <= kRouteWeightMaxAbsoluteLimit &&
            route_margin_abs <= kRouteMarginMaxAbsoluteLimit &&
            shared_scale_abs <= kSharedScaleMaxAbsoluteLimit &&
            serial_cpu_route_weight_max_abs <=
                kRouteWeightMaxAbsoluteLimit &&
            parallel_cpu_route_weight_max_abs <=
                kRouteWeightMaxAbsoluteLimit &&
            serial_cpu_route_margin_abs <=
                kRouteMarginMaxAbsoluteLimit &&
            parallel_cpu_route_margin_abs <=
                kRouteMarginMaxAbsoluteLimit &&
            serial_cpu_shared_scale_abs <=
                kSharedScaleMaxAbsoluteLimit &&
            parallel_cpu_shared_scale_abs <=
                kSharedScaleMaxAbsoluteLimit &&
            route_values_valid &&
            std::abs(serial_route_weight_sum - 1.0) <=
                kRouteWeightSumLimit &&
            std::abs(parallel_route_weight_sum - 1.0) <=
                kRouteWeightSumLimit &&
            parallel_serial.relative_l2 <=
                kParallelSerialRelativeL2Limit &&
            parallel_serial.max_absolute <=
                kParallelSerialMaxAbsoluteLimit;
        if (!pass) {
            std::cerr
                << "FAIL: prepared parallel-route correctness gate failed\n";
            return 2;
        }
        std::cout << "PASS: selected/shared prepared bytes, exact routes, "
                     "native/prepared control, CPU/control tolerances, and "
                     "per-path bit determinism\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
