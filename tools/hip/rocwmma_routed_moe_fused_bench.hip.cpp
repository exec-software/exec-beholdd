// Paired qualification harness for a fused gate+up+SiLU BF16 rocWMMA path.
//
// Keep the previously validated five-launch implementation frozen by including
// it verbatim and renaming only its standalone entry point in this translation
// unit. The candidate and control therefore share buffers, data, kernels, and
// timing process without modifying the baseline source.

#define main beholdd_frozen_control_main
#include "rocwmma_routed_moe_bench.hip.cpp"
#undef main

namespace {

// One wave owns one expert/output-column tile. Each inner iteration loads the
// shared input fragment once, then advances independent gate and up FP32
// accumulators. Two exact 16x16 FP32 LDS tiles expose the completed fragments to
// scalar SiLU/multiply conversion without a global-memory round trip.
extern "C" __global__ __launch_bounds__(kWaveSize)
void fused_gate_up_silu_wmma(const Bf16* padded_input,
                             const Bf16* gate_up_weights,
                             Bf16* activation) {
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

    __shared__ float gate_tile[kTile * kTile];
    __shared__ float up_tile[kTile * kTile];

    const std::uint32_t output_tile = blockIdx.x;
    const std::uint32_t expert = blockIdx.y;
    const std::uint32_t output_column = output_tile * kTile;
    const std::size_t matrix_elements =
        static_cast<std::size_t>(kHidden) * kFfn;
    const std::size_t projection_stride = kExperts * matrix_elements;
    const Bf16* gate_weights = gate_up_weights + expert * matrix_elements;
    const Bf16* up_weights =
        gate_up_weights + projection_stride + expert * matrix_elements;

    FragAcc gate_accumulator;
    FragAcc up_accumulator;
    rocwmma::fill_fragment(gate_accumulator, 0.0f);
    rocwmma::fill_fragment(up_accumulator, 0.0f);

    for (std::uint32_t inner = 0; inner < kHidden; inner += kTile) {
        FragA input_fragment;
        FragB gate_fragment;
        FragB up_fragment;

        rocwmma::load_matrix_sync(input_fragment,
                                  padded_input + inner,
                                  kHidden);
        rocwmma::load_matrix_sync(gate_fragment,
                                  gate_weights + inner * kFfn + output_column,
                                  kFfn);
        rocwmma::load_matrix_sync(up_fragment,
                                  up_weights + inner * kFfn + output_column,
                                  kFfn);
        rocwmma::mma_sync(gate_accumulator,
                          input_fragment,
                          gate_fragment,
                          gate_accumulator);
        rocwmma::mma_sync(up_accumulator,
                          input_fragment,
                          up_fragment,
                          up_accumulator);
    }

    rocwmma::store_matrix_sync(gate_tile, gate_accumulator, kTile);
    rocwmma::store_matrix_sync(up_tile, up_accumulator, kTile);
    __syncthreads();

#pragma unroll 1
    for (std::uint32_t index = threadIdx.x; index < kTile * kTile;
         index += blockDim.x) {
        const std::uint32_t token = index / kTile;
        const std::uint32_t tile_column = index % kTile;
        const float gate = gate_tile[index];
        const float up = up_tile[index];
        const float silu = gate / (1.0f + __builtin_expf(-gate));
        const std::size_t activation_index =
            (static_cast<std::size_t>(expert) * kPaddedTokens + token) * kFfn +
            output_column + tile_column;
        activation[activation_index] = Bf16(silu * up);
    }
}

void launch_fused_pipeline(DeviceData& device,
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

    hipLaunchKernelGGL(fused_gate_up_silu_wmma,
                       dim3(kFfn / kTile, kExperts),
                       dim3(kWaveSize),
                       0,
                       stream,
                       device.padded_input.get(),
                       device.gate_up_weights.get(),
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

void enqueue_fused_host_visible_operator(DeviceData& device,
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
    launch_fused_pipeline(device, token_count, stream);
    HIP_CHECK(hipMemcpyAsync(host.output.get(),
                             device.output.get(),
                             static_cast<std::size_t>(token_count) * kHidden *
                                 sizeof(float),
                             hipMemcpyDeviceToHost,
                             stream));
}

struct ValidationResult {
    ErrorMetrics mechanical_error{};
    ErrorMetrics f32_policy_error{};
    std::vector<float> output;
    bool bit_deterministic = false;
};

template <typename Enqueue>
ValidationResult validate_path(HostData& host,
                               DeviceData& device,
                               const std::vector<float>& mechanical_reference,
                               const std::vector<float>& f32_policy_reference,
                               std::uint32_t token_count,
                               hipStream_t stream,
                               bool enqueue_copies_output,
                               Enqueue&& enqueue) {
    const std::size_t output_elements =
        static_cast<std::size_t>(token_count) * kHidden;
    const std::size_t output_bytes = output_elements * sizeof(float);

    auto enqueue_with_output = [&] {
        enqueue();
        if (!enqueue_copies_output) {
            HIP_CHECK(hipMemcpyAsync(host.output.get(),
                                     device.output.get(),
                                     output_bytes,
                                     hipMemcpyDeviceToHost,
                                     stream));
        }
    };

    enqueue_with_output();
    HIP_CHECK(hipStreamSynchronize(stream));
    ValidationResult result;
    result.output.assign(host.output.get(),
                         host.output.get() + output_elements);
    result.mechanical_error = compare_output(
        result.output.data(), mechanical_reference, token_count);
    result.f32_policy_error = compare_output(
        result.output.data(), f32_policy_reference, token_count);

    enqueue_with_output();
    HIP_CHECK(hipStreamSynchronize(stream));
    result.bit_deterministic =
        std::memcmp(result.output.data(), host.output.get(), output_bytes) == 0;
    return result;
}

bool exact_match(const std::vector<float>& left,
                 const std::vector<float>& right) {
    return left.size() == right.size() &&
           std::memcmp(left.data(),
                       right.data(),
                       left.size() * sizeof(float)) == 0;
}

bool meets_mechanical_gate(const ValidationResult& result) {
    constexpr double maximum_relative_l2 = 5.0e-4;
    constexpr double maximum_absolute = 1.0e-3;
    return result.mechanical_error.all_finite &&
           result.mechanical_error.relative_l2 <= maximum_relative_l2 &&
           result.mechanical_error.max_absolute <= maximum_absolute &&
           result.bit_deterministic && result.f32_policy_error.all_finite &&
           std::isfinite(result.f32_policy_error.relative_l2) &&
           std::isfinite(result.f32_policy_error.max_absolute);
}

void print_validation(std::uint32_t token_count,
                      std::string_view path,
                      std::string_view envelope,
                      const ValidationResult& result,
                      bool control_bit_match) {
    std::cout << token_count << ',' << path << ',' << envelope << ','
              << std::scientific << std::setprecision(6)
              << result.mechanical_error.relative_l2 << ','
              << result.mechanical_error.max_absolute << ','
              << result.f32_policy_error.relative_l2 << ','
              << result.f32_policy_error.max_absolute << ','
              << result.mechanical_error.checksum << ','
              << (result.bit_deterministic ? "true" : "false") << ','
              << (control_bit_match ? "true" : "false") << '\n';
}

struct DistributionStats {
    double minimum = 0.0;
    double p10 = 0.0;
    double median = 0.0;
    double p90 = 0.0;
    double maximum = 0.0;
};

DistributionStats summarize_distribution(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const auto percentile_index = [&](double percentile) {
        return std::min(
            samples.size() - 1,
            static_cast<std::size_t>(std::ceil(samples.size() * percentile)) -
                1);
    };
    return {samples.front(),
            samples[percentile_index(0.1)],
            samples[samples.size() / 2],
            samples[percentile_index(0.9)],
            samples.back()};
}

template <typename Enqueue>
double measure_burst(int burst, hipStream_t stream, Enqueue&& enqueue) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto begin = std::chrono::steady_clock::now();
    for (int operation = 0; operation < burst; ++operation) {
        enqueue();
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - begin).count() /
           static_cast<double>(burst);
}

struct PairedTiming {
    DistributionStats control_ms;
    DistributionStats candidate_ms;
    DistributionStats control_over_candidate;
};

template <typename ControlEnqueue, typename CandidateEnqueue>
PairedTiming benchmark_pair(const Options& options,
                            hipStream_t stream,
                            ControlEnqueue&& enqueue_control,
                            CandidateEnqueue&& enqueue_candidate) {
    for (int warmup = 0; warmup < options.warmups; ++warmup) {
        if (warmup % 2 == 0) {
            enqueue_control();
            enqueue_candidate();
        } else {
            enqueue_candidate();
            enqueue_control();
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> control_samples;
    std::vector<double> candidate_samples;
    std::vector<double> ratios;
    control_samples.reserve(options.samples);
    candidate_samples.reserve(options.samples);
    ratios.reserve(options.samples);

    for (int sample = 0; sample < options.samples; ++sample) {
        double control_ms = 0.0;
        double candidate_ms = 0.0;
        if (sample % 2 == 0) {
            control_ms = measure_burst(
                options.burst, stream, enqueue_control);
            candidate_ms = measure_burst(
                options.burst, stream, enqueue_candidate);
        } else {
            candidate_ms = measure_burst(
                options.burst, stream, enqueue_candidate);
            control_ms = measure_burst(
                options.burst, stream, enqueue_control);
        }
        control_samples.push_back(control_ms);
        candidate_samples.push_back(candidate_ms);
        ratios.push_back(control_ms / candidate_ms);
    }

    return {summarize_distribution(std::move(control_samples)),
            summarize_distribution(std::move(candidate_samples)),
            summarize_distribution(std::move(ratios))};
}

void print_paired_timing(std::uint32_t token_count,
                         std::string_view envelope,
                         const PairedTiming& timing) {
    std::cout << token_count << ',' << envelope << ',' << std::fixed
              << std::setprecision(6) << timing.control_ms.median << ','
              << timing.control_ms.p90 << ',' << timing.candidate_ms.median
              << ',' << timing.candidate_ms.p90 << ','
              << timing.control_over_candidate.p10 << ','
              << timing.control_over_candidate.median << ','
              << timing.control_over_candidate.p90 << '\n';
}

bool meets_qualification_run_gate(const PairedTiming& timing) {
    constexpr double minimum_paired_median_speedup = 1.03;
    constexpr double maximum_candidate_over_control_p90 = 1.05;
    return std::isfinite(timing.control_over_candidate.median) &&
           timing.control_over_candidate.median >=
               minimum_paired_median_speedup &&
           std::isfinite(timing.control_ms.p90) &&
           std::isfinite(timing.candidate_ms.p90) &&
           timing.candidate_ms.p90 <=
               maximum_candidate_over_control_p90 * timing.control_ms.p90;
}

void print_qualification_run_gate(std::uint32_t token_count,
                                  std::string_view envelope,
                                  const PairedTiming& timing,
                                  bool passed) {
    std::cout << "qualification_run_gate: N=" << token_count
              << " envelope=" << envelope
              << " paired_median_speedup=" << std::fixed
              << std::setprecision(6)
              << timing.control_over_candidate.median
              << " minimum=1.030000 candidate_over_control_p90="
              << timing.candidate_ms.p90 / timing.control_ms.p90
              << " maximum=1.050000 status="
              << (passed ? "pass" : "fail") << '\n';
}

std::size_t persistent_bytes(const DeviceData& device) {
    return device.gate_up_weights.size_bytes() +
           device.down_weights.size_bytes() + device.input.size_bytes() +
           device.routes.size_bytes() + device.padded_input.size_bytes() +
           device.gate_up_output.size_bytes() +
           device.activation.size_bytes() +
           device.expert_output.size_bytes() + device.output.size_bytes();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        bool quick_mode = false;
        bool explicit_burst = false;
        bool explicit_samples = false;
        bool explicit_tokens = false;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--help" || argument == "-h") {
                std::cout
                    << "Usage: rocwmma_routed_moe_fused_bench [options]\n"
                    << "  --warmups N  untimed paired operators (default 4; "
                       "minimum 4)\n"
                    << "  --samples N  paired wall-clock samples (default 12; "
                       "minimum 12 and multiple of 4)\n"
                    << "  --burst N    operators per measurement (default 16; "
                       "minimum 16)\n"
                    << "  --tokens N   smoke-only cohort selector; forbidden "
                       "for qualification\n"
                    << "  --quick      smoke settings; not qualification-grade\n";
                return 0;
            }
            quick_mode = quick_mode || argument == "--quick";
            explicit_burst = explicit_burst || argument == "--burst";
            explicit_samples = explicit_samples || argument == "--samples";
            explicit_tokens = explicit_tokens || argument == "--tokens";
        }

        Options options = parse_options(argc, argv);
        if (!quick_mode && !explicit_samples) {
            options.samples = 12;
        }
        if (!quick_mode && !explicit_burst) {
            options.burst = 16;
        }
        if (!quick_mode && options.warmups < 4) {
            throw std::runtime_error(
                "fused qualification requires --warmups >= 4 outside --quick");
        }
        if (!quick_mode &&
            (options.samples < 12 || options.samples % 4 != 0)) {
            throw std::runtime_error(
                "fused qualification requires --samples >= 12 and a "
                "multiple of 4 outside --quick");
        }
        if (!quick_mode && options.burst < 16) {
            throw std::runtime_error(
                "fused qualification requires --burst >= 16 outside --quick");
        }
        if (!quick_mode && explicit_tokens) {
            throw std::runtime_error(
                "fused qualification requires the complete N={1,2,4,8,16} "
                "cohort; --tokens is smoke-only");
        }
        const std::vector<std::uint32_t> qualification_tokens = {
            1, 2, 4, 8, 16};
        if (!quick_mode && options.token_counts != qualification_tokens) {
            throw std::runtime_error(
                "fused qualification requires exactly N={1,2,4,8,16}");
        }

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

        std::cout << "device=" << properties.name << " arch="
                  << properties.gcnArchName << " warp=" << properties.warpSize
                  << '\n';
        std::cout << "shape: H=" << kHidden << " F=" << kFfn
                  << " K=" << kTopK << " experts=" << kExperts
                  << " padded_tokens=" << kPaddedTokens << '\n';
        std::cout << "cohort: ideal high overlap; every token routes exactly once "
                     "through the same 8 resident experts (route order differs)\n";
        std::cout << "control: launches/operator=5 "
                     "[f32_to_bf16,gate_up_wmma,silu_to_bf16,down_wmma,"
                     "route_reduce]\n";
        std::cout << "candidate: launches/operator=4 "
                     "[f32_to_bf16,fused_gate_up_silu_wmma,down_wmma,"
                     "route_reduce]\n";
        std::cout << "candidate fusion: one wave/tile, one input fragment load per "
                     "K tile, two FP32 accumulators, two 16x16 FP32 LDS staging "
                     "tiles, direct BF16 activation write\n";
        std::cout << "timing: paired chrono wall clock, alternating control/"
                     "candidate order per sample; HIP events unused; warmups="
                  << options.warmups << " samples=" << options.samples
                  << " burst=" << options.burst << '\n';

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
                  << mib(persistent_bytes(device))
                  << " persistent_weight_MiB="
                  << mib(device.gate_up_weights.size_bytes() +
                         device.down_weights.size_bytes())
                  << '\n';

        const auto mechanical_oracle_begin = std::chrono::steady_clock::now();
        const std::vector<float> mechanical_reference =
            cpu_reference(host, true);
        const auto mechanical_oracle_end = std::chrono::steady_clock::now();
        const auto f32_oracle_begin = std::chrono::steady_clock::now();
        const std::vector<float> f32_policy_reference =
            cpu_reference(host, false);
        const auto f32_oracle_end = std::chrono::steady_clock::now();
        std::cout << "bf16_boundary_cpu_oracle_ms=" << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(
                         mechanical_oracle_end - mechanical_oracle_begin)
                         .count()
                  << " f32_policy_cpu_oracle_ms="
                  << std::chrono::duration<double, std::milli>(f32_oracle_end -
                                                               f32_oracle_begin)
                         .count()
                  << '\n';

        std::cout << "N,path,envelope,mechanical_relative_l2,mechanical_max_abs,"
                     "f32_policy_relative_l2,f32_policy_max_abs,checksum,"
                     "bit_deterministic,control_bit_match\n";

        bool valid = true;
        for (const std::uint32_t token_count : options.token_counts) {
            const ValidationResult control_resident = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                false,
                [&] { launch_pipeline(device, token_count, stream.get()); });
            const ValidationResult candidate_resident = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                false,
                [&] {
                    launch_fused_pipeline(device, token_count, stream.get());
                });
            const bool resident_match = exact_match(
                control_resident.output, candidate_resident.output);

            const ValidationResult control_host_visible = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                true,
                [&] {
                    enqueue_host_visible_operator(
                        device, host, token_count, stream.get());
                });
            const ValidationResult candidate_host_visible = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                true,
                [&] {
                    enqueue_fused_host_visible_operator(
                        device, host, token_count, stream.get());
                });
            const bool host_visible_match = exact_match(
                control_host_visible.output, candidate_host_visible.output);

            valid = valid && meets_mechanical_gate(control_resident) &&
                    meets_mechanical_gate(candidate_resident) &&
                    resident_match &&
                    meets_mechanical_gate(control_host_visible) &&
                    meets_mechanical_gate(candidate_host_visible) &&
                    host_visible_match;

            print_validation(token_count,
                             "control",
                             "resident",
                             control_resident,
                             true);
            print_validation(token_count,
                             "candidate",
                             "resident",
                             candidate_resident,
                             resident_match);
            print_validation(token_count,
                             "control",
                             "host_visible",
                             control_host_visible,
                             true);
            print_validation(token_count,
                             "candidate",
                             "host_visible",
                             candidate_host_visible,
                             host_visible_match);
        }

        if (!valid) {
            std::cerr << "validation failed: control and candidate require finite "
                         "output, mechanical relative L2 <= 5e-4, max abs <= "
                         "1e-3, repeated bit determinism, and exact candidate/"
                         "control agreement\n";
            return 2;
        }

        std::cout << "N,envelope,control_median_ms,control_p90_ms,"
                     "candidate_median_ms,candidate_p90_ms,"
                     "paired_control_over_candidate_p10,"
                     "paired_control_over_candidate_median,"
                     "paired_control_over_candidate_p90\n";
        bool performance_valid = true;
        std::size_t performance_gate_count = 0;
        for (const std::uint32_t token_count : options.token_counts) {
            if (token_count != 1 && token_count != 16) {
                continue;
            }

            const PairedTiming resident = benchmark_pair(
                options,
                stream.get(),
                [&] { launch_pipeline(device, token_count, stream.get()); },
                [&] {
                    launch_fused_pipeline(device, token_count, stream.get());
                });
            const PairedTiming host_visible = benchmark_pair(
                options,
                stream.get(),
                [&] {
                    enqueue_host_visible_operator(
                        device, host, token_count, stream.get());
                },
                [&] {
                    enqueue_fused_host_visible_operator(
                        device, host, token_count, stream.get());
                });
            print_paired_timing(token_count, "resident", resident);
            print_paired_timing(token_count, "host_visible", host_visible);
            if (!quick_mode) {
                const bool resident_passed =
                    meets_qualification_run_gate(resident);
                const bool host_visible_passed =
                    meets_qualification_run_gate(host_visible);
                print_qualification_run_gate(
                    token_count, "resident", resident, resident_passed);
                print_qualification_run_gate(token_count,
                                             "host_visible",
                                             host_visible,
                                             host_visible_passed);
                performance_valid = performance_valid && resident_passed &&
                                    host_visible_passed;
                performance_gate_count += 2;
            }
        }
        if (!quick_mode &&
            (!performance_valid || performance_gate_count != 4)) {
            std::cerr
                << "qualification run failed: every resident and host-visible "
                   "N=1/N=16 cell requires paired median speedup >= 1.03 and "
                   "candidate p90 <= 1.05x control p90\n";
            return 3;
        }
        if (!quick_mode) {
            std::cout
                << "qualification_run=pass aggregate_gate_pending="
                   "three_guarded_runs each_envelope_median_of_three_N1>=1.10 "
                   "each_envelope_median_of_three_N16>=1.08\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
