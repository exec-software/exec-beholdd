// Isolated parallel-router experiment for the real-weight Qwen3.5/3.6 layer.
//
// The validated resident-layer harness is included without modification. Its
// main and serial top-k symbol are renamed so this translation unit can keep
// the exact seven-kernel resident chain as a same-process control while
// replacing only the router selection launch in the candidate.

#define main q35_frozen_resident_layer_main
#define q35_stable_softmax_topk8 q35_stable_softmax_topk8_serial
#include "rocwmma_qwen35_resident_layer_bench.hip.cpp"
#undef q35_stable_softmax_topk8
#undef main

namespace {

constexpr std::uint32_t kParallelRouteThreads = kAllExperts;
static_assert(kParallelRouteThreads == 256);
static_assert(kParallelRouteThreads == kElementThreads);

__device__ __forceinline__ bool q35_route_pair_better(
    float candidate_value,
    std::uint32_t candidate_id,
    float incumbent_value,
    std::uint32_t incumbent_id) {
    return candidate_value > incumbent_value ||
           (candidate_value == incumbent_value &&
            candidate_id < incumbent_id);
}

// One lane owns one expert. The reduction tree is fixed, including all F32
// additions, and route ties always prefer the lower expert ID. The kernel uses
// no per-thread arrays; the build wrapper rejects private memory or scratch.
extern "C" __global__ __launch_bounds__(kParallelRouteThreads)
void q35_stable_softmax_topk8_parallel(
    const float* logits,
    const float* shared_logit,
    float* probabilities,
    std::uint32_t* route_ids,
    float* route_weights,
    float* shared_scale,
    float* route_margin,
    std::uint32_t* route_status) {
    __shared__ float reduction_values[kParallelRouteThreads];
    __shared__ std::uint32_t reduction_ids[kParallelRouteThreads];
    __shared__ float denominator;
    __shared__ std::uint32_t status;

    const std::uint32_t expert = threadIdx.x;
    if (expert == 0) {
        status = 0;
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

    // Stable maximum: higher logit wins and an exact tie selects lower ID.
    for (std::uint32_t stride = kParallelRouteThreads / 2; stride != 0;
         stride /= 2) {
        if (expert < stride) {
            const float candidate_value = reduction_values[expert + stride];
            const std::uint32_t candidate_id =
                reduction_ids[expert + stride];
            if (q35_route_pair_better(candidate_value,
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
    // Fixed F32 tree sum. This intentionally differs from the serial control's
    // left fold; the selected top-8 is renormalized below before expert reduce.
    for (std::uint32_t stride = kParallelRouteThreads / 2; stride != 0;
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

    // Nine deterministic stable maxima produce top-8 plus the rank-9 margin
    // operand. Every iteration uses the same full-block reduction tree.
    for (std::uint32_t rank = 0; rank <= kTopK; ++rank) {
        reduction_values[expert] = selected ? -1.0f : probability;
        reduction_ids[expert] = expert;
        __syncthreads();
        for (std::uint32_t stride = kParallelRouteThreads / 2; stride != 0;
             stride /= 2) {
            if (expert < stride) {
                const float candidate_value =
                    reduction_values[expert + stride];
                const std::uint32_t candidate_id =
                    reduction_ids[expert + stride];
                if (q35_route_pair_better(candidate_value,
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

void launch_prefix_parallel(const ResidentLayer& layer,
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

    hipLaunchKernelGGL(q35_stable_softmax_topk8_parallel,
                       dim3(1),
                       dim3(kParallelRouteThreads),
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

void enqueue_parallel_route(const ResidentLayer& layer,
                            ResidentState& state,
                            hipStream_t stream) {
    launch_prefix_parallel(layer, state, stream);
    launch_suffix(layer,
                  state,
                  state.route_ids.get(),
                  state.route_weights.get(),
                  state.shared_scale.get(),
                  state.route_status.get(),
                  stream);
}

bool route_bit_exact(const ObservedRoute& left, const ObservedRoute& right) {
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

void print_route(std::string_view label, const ObservedRoute& route) {
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
                "parallel-route benchmark requires gfx1151; device reports " +
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
        std::cout << "control=serial_stable_topk candidate=parallel_256_thread_"
                     "stable_topk shared_other_six_kernels=true "
                     "resident=true one_external_sync_per_burst=true\n";

        require_memory_headroom();
        Stream stream;
        ResidentLayer layer;
        ResidentState state;
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

        const CpuReference reference = cpu_reference(pack, hidden);

        poison_scratch(state, stream.get());
        enqueue_resident(layer, state, stream.get());
        const std::vector<float> serial_first =
            read_output(state, stream.get());
        const ObservedRoute serial_route_first =
            read_route(state, stream.get());
        enqueue_resident(layer, state, stream.get());
        const std::vector<float> serial_second =
            read_output(state, stream.get());
        const ObservedRoute serial_route_second =
            read_route(state, stream.get());

        poison_scratch(state, stream.get());
        enqueue_parallel_route(layer, state, stream.get());
        const std::vector<float> parallel_first =
            read_output(state, stream.get());
        const ObservedRoute parallel_route_first =
            read_route(state, stream.get());
        enqueue_parallel_route(layer, state, stream.get());
        const std::vector<float> parallel_second =
            read_output(state, stream.get());
        const ObservedRoute parallel_route_second =
            read_route(state, stream.get());

        const bool serial_output_deterministic =
            std::memcmp(serial_first.data(),
                        serial_second.data(),
                        serial_first.size() * sizeof(float)) == 0;
        const bool parallel_output_deterministic =
            std::memcmp(parallel_first.data(),
                        parallel_second.data(),
                        parallel_first.size() * sizeof(float)) == 0;
        const bool serial_route_deterministic =
            route_bit_exact(serial_route_first, serial_route_second);
        const bool parallel_route_deterministic =
            route_bit_exact(parallel_route_first, parallel_route_second);

        bool route_ids_exact = serial_route_first.status == 0 &&
                               parallel_route_first.status == 0;
        double route_weight_max_abs = 0.0;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_ids_exact =
                route_ids_exact &&
                serial_route_first.ids[rank] == reference.route_ids[rank] &&
                parallel_route_first.ids[rank] == reference.route_ids[rank] &&
                parallel_route_first.ids[rank] == serial_route_first.ids[rank];
            route_weight_max_abs =
                std::max(route_weight_max_abs,
                         std::abs(static_cast<double>(
                                      parallel_route_first.weights[rank]) -
                                  serial_route_first.weights[rank]));
        }
        const double route_margin_abs =
            std::abs(static_cast<double>(parallel_route_first.margin) -
                     serial_route_first.margin);
        const double shared_scale_abs =
            std::abs(static_cast<double>(parallel_route_first.shared_scale) -
                     serial_route_first.shared_scale);

        const ErrorMetrics serial_cpu =
            compare_output(serial_first, reference.output);
        const ErrorMetrics parallel_cpu =
            compare_output(parallel_first, reference.output);
        const ErrorMetrics parallel_serial =
            compare_output(parallel_first, serial_first);

        print_route("serial", serial_route_first);
        print_route("parallel", parallel_route_first);
        std::cout << "route_ids_exact=" << std::boolalpha << route_ids_exact
                  << " route_weight_max_abs=" << std::scientific
                  << route_weight_max_abs
                  << " route_margin_abs=" << route_margin_abs
                  << " shared_scale_abs=" << shared_scale_abs << '\n';
        std::cout << "serial_cpu_relative_l2=" << serial_cpu.relative_l2
                  << " serial_cpu_max_abs=" << serial_cpu.max_absolute
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
                enqueue_resident(layer, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
                enqueue_parallel_route(layer, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
            } else {
                enqueue_parallel_route(layer, state, stream.get());
                HIP_CHECK(hipStreamSynchronize(stream.get()));
                enqueue_resident(layer, state, stream.get());
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
                serial_ms = timed_burst(
                    options.burst,
                    stream.get(),
                    [&] { enqueue_resident(layer, state, stream.get()); });
            };
            auto time_parallel = [&] {
                parallel_ms = timed_burst(
                    options.burst,
                    stream.get(),
                    [&] { enqueue_parallel_route(layer, state, stream.get()); });
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
        print_timing("serial_route_layer", serial_timing);
        print_timing("parallel_route_layer", parallel_timing);
        std::cout << "paired_speedup_min=" << std::fixed
                  << std::setprecision(6) << paired.minimum_ms
                  << " paired_speedup_median=" << paired.median_ms
                  << " paired_speedup_p90=" << paired.p90_ms
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
        const bool pass =
            route_ids_exact && serial_cpu.all_finite &&
            parallel_cpu.all_finite && parallel_serial.all_finite &&
            serial_output_deterministic && parallel_output_deterministic &&
            serial_route_deterministic && parallel_route_deterministic &&
            serial_cpu.relative_l2 <= kCpuRelativeL2Limit &&
            serial_cpu.max_absolute <= kCpuMaxAbsoluteLimit &&
            parallel_cpu.relative_l2 <= kCpuRelativeL2Limit &&
            parallel_cpu.max_absolute <= kCpuMaxAbsoluteLimit &&
            route_weight_max_abs <= kRouteWeightMaxAbsoluteLimit &&
            route_margin_abs <= kRouteMarginMaxAbsoluteLimit &&
            shared_scale_abs <= kSharedScaleMaxAbsoluteLimit &&
            parallel_serial.relative_l2 <=
                kParallelSerialRelativeL2Limit &&
            parallel_serial.max_absolute <=
                kParallelSerialMaxAbsoluteLimit;
        if (!pass) {
            std::cerr << "FAIL: parallel-route correctness gate failed\n";
            return 2;
        }
        std::cout << "PASS: exact routes, route/output tolerances, finite "
                     "dual-oracle outputs, and per-path bit determinism\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
