// Paired qualification harness for fusing the BF16 down projection with the
// rank-ordered FP32 route reduction.
//
// The validated five-launch implementation remains frozen in its own source
// file. Including it after renaming only its entry point places the control and
// candidate in one process without changing the baseline bytes.

#define main beholdd_frozen_control_main
#include "rocwmma_routed_moe_bench.hip.cpp"
#undef main

namespace {

constexpr std::uint32_t kFusedThreads = kExperts * kWaveSize;
constexpr std::size_t kFusedTileElements =
    static_cast<std::size_t>(kExperts) * kPaddedTokens * kTile;
constexpr std::size_t kFusedLdsBytes =
    kFusedTileElements * sizeof(float);

static_assert(kFusedThreads == 256);
static_assert(kFusedLdsBytes == 8192);
static_assert(kTopK == kExperts);
static_assert(kPaddedTokens == kTile);
static_assert(kFusedThreads == kPaddedTokens * kTile);

// One 256-thread workgroup owns one 16-column output tile. Its eight wave32s
// independently compute that tile for experts 0..7. The raw FP32 expert tiles
// are staged in expert-major LDS, all waves rendezvous, then one thread owns
// each (token, tile-column) output and reproduces the frozen reducer's route-
// rank accumulation order. The token guard must remain after the barrier.
extern "C" __global__ __launch_bounds__(kFusedThreads)
void fused_down_route_wmma(const Bf16* activation,
                           const Bf16* down_weights,
                           const RouteEntry* routes,
                           float* output,
                           std::uint32_t token_count) {
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

    __shared__ float expert_tiles[kFusedTileElements];

    const std::uint32_t local_thread = threadIdx.x;
    const std::uint32_t expert = local_thread / kWaveSize;
    const std::uint32_t output_column = blockIdx.x * kTile;
    const std::size_t activation_matrix =
        static_cast<std::size_t>(kPaddedTokens) * kFfn;
    const std::size_t weight_matrix =
        static_cast<std::size_t>(kFfn) * kHidden;
    const Bf16* expert_activation = activation + expert * activation_matrix;
    const Bf16* weights = down_weights + expert * weight_matrix;

    FragAcc accumulator;
    rocwmma::fill_fragment(accumulator, 0.0f);

#pragma unroll 2
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

    float* expert_tile =
        expert_tiles + static_cast<std::size_t>(expert) * kTile * kTile;
    rocwmma::store_matrix_sync(expert_tile, accumulator, kTile);
    __syncthreads();

    const std::uint32_t token = local_thread / kTile;
    const std::uint32_t tile_column = local_thread % kTile;
    if (token >= token_count) {
        return;
    }

    float sum = 0.0f;
    for (std::uint32_t route_index = 0; route_index < kTopK;
         ++route_index) {
        const RouteEntry route = routes[token * kTopK + route_index];
        sum += route.weight *
               expert_tiles[(static_cast<std::size_t>(route.expert) *
                             kPaddedTokens + token) *
                                kTile +
                            tile_column];
    }
    output[static_cast<std::size_t>(token) * kHidden + output_column +
           tile_column] = sum;
}

void launch_down_fused_pipeline(DeviceData& device,
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

    hipLaunchKernelGGL(fused_down_route_wmma,
                       dim3(kHidden / kTile),
                       dim3(kFusedThreads),
                       0,
                       stream,
                       device.activation.get(),
                       device.down_weights.get(),
                       device.routes.get(),
                       device.output.get(),
                       token_count);
    HIP_CHECK(hipGetLastError());
}

void enqueue_down_fused_host_visible_operator(DeviceData& device,
                                              HostData& host,
                                              std::uint32_t token_count,
                                              hipStream_t stream) {
    const std::size_t input_bytes =
        static_cast<std::size_t>(token_count) * kHidden * sizeof(float);
    const std::size_t route_bytes =
        static_cast<std::size_t>(token_count) * kTopK * sizeof(RouteEntry);
    HIP_CHECK(hipMemcpyAsync(device.input.get(),
                             host.input.get(),
                             input_bytes,
                             hipMemcpyHostToDevice,
                             stream));
    HIP_CHECK(hipMemcpyAsync(device.routes.get(),
                             host.routes.get(),
                             route_bytes,
                             hipMemcpyHostToDevice,
                             stream));
    launch_down_fused_pipeline(device, token_count, stream);
    HIP_CHECK(hipMemcpyAsync(host.output.get(),
                             device.output.get(),
                             input_bytes,
                             hipMemcpyDeviceToHost,
                             stream));
}

enum class RouteVariantKind {
    Permuted,
    Reversed,
    DuplicateMissing,
    OneHot,
};

struct RouteVariant {
    RouteVariantKind kind;
    std::string_view name;
    std::array<RouteEntry, kPaddedTokens * kTopK> routes{};
};

std::vector<RouteVariant> make_route_variants(const HostData& host) {
    RouteVariant permuted{RouteVariantKind::Permuted, "permuted"};
    std::copy_n(host.routes.get(), permuted.routes.size(), permuted.routes.begin());

    RouteVariant reversed = permuted;
    reversed.kind = RouteVariantKind::Reversed;
    reversed.name = "reversed";
    for (std::uint32_t token = 0; token < kPaddedTokens; ++token) {
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            reversed.routes[token * kTopK + rank] =
                permuted.routes[token * kTopK + (kTopK - 1 - rank)];
        }
    }

    RouteVariant duplicate_missing = permuted;
    duplicate_missing.kind = RouteVariantKind::DuplicateMissing;
    duplicate_missing.name = "duplicate_missing";
    for (std::uint32_t token = 0; token < kPaddedTokens; ++token) {
        duplicate_missing.routes[token * kTopK + (kTopK - 1)].expert =
            permuted.routes[token * kTopK].expert;
    }

    RouteVariant one_hot = permuted;
    one_hot.kind = RouteVariantKind::OneHot;
    one_hot.name = "one_hot";
    for (std::uint32_t token = 0; token < kPaddedTokens; ++token) {
        const std::uint32_t hot_rank = (token * 3 + 1) % kTopK;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            one_hot.routes[token * kTopK + rank].weight =
                rank == hot_rank ? 1.0f : 0.0f;
        }
    }

    return {permuted, reversed, duplicate_missing, one_hot};
}

void require_route_variants_valid(const std::vector<RouteVariant>& variants) {
    if (variants.size() != 4) {
        throw std::runtime_error("route-variant construction is incomplete");
    }
    for (const RouteVariant& variant : variants) {
        for (std::uint32_t token = 0; token < kPaddedTokens; ++token) {
            std::array<std::uint32_t, kExperts> expert_counts{};
            std::uint32_t hot_weights = 0;
            for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
                const RouteEntry route =
                    variant.routes[token * kTopK + rank];
                if (route.expert >= kExperts || !std::isfinite(route.weight)) {
                    throw std::runtime_error(
                        std::string(variant.name) +
                        " contains an invalid expert or non-finite weight");
                }
                ++expert_counts[route.expert];
                if (route.weight == 1.0f) {
                    ++hot_weights;
                } else if (variant.kind == RouteVariantKind::OneHot &&
                           route.weight != 0.0f) {
                    throw std::runtime_error(
                        "one_hot contains a weight other than zero or one");
                }
            }

            const std::uint32_t distinct =
                static_cast<std::uint32_t>(std::count_if(
                    expert_counts.begin(),
                    expert_counts.end(),
                    [](std::uint32_t count) { return count != 0; }));
            if ((variant.kind == RouteVariantKind::Permuted ||
                 variant.kind == RouteVariantKind::Reversed ||
                 variant.kind == RouteVariantKind::OneHot) &&
                distinct != kExperts) {
                throw std::runtime_error(std::string(variant.name) +
                                         " is not a full permutation");
            }
            if (variant.kind == RouteVariantKind::DuplicateMissing &&
                (distinct != kExperts - 1 ||
                 std::count(expert_counts.begin(),
                            expert_counts.end(),
                            2U) != 1)) {
                throw std::runtime_error(
                    "duplicate_missing does not contain one duplicate and one omission");
            }
            if (variant.kind == RouteVariantKind::OneHot && hot_weights != 1) {
                throw std::runtime_error(
                    "one_hot does not contain exactly one unit weight");
            }
        }
    }
}

void install_route_variant(HostData& host, const RouteVariant& variant) {
    std::copy(variant.routes.begin(),
              variant.routes.end(),
              host.routes.get());
}

void upload_resident_inputs(HostData& host,
                            DeviceData& device,
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
    HIP_CHECK(hipStreamSynchronize(stream));
}

struct DownValidation {
    ErrorMetrics mechanical_error{};
    ErrorMetrics f32_policy_error{};
    std::vector<float> output;
    bool bit_deterministic = false;
};

void poison_pipeline_state(HostData& host,
                           DeviceData& device,
                           std::uint32_t token_count,
                           hipStream_t stream,
                           bool host_visible) {
    const std::size_t output_bytes =
        static_cast<std::size_t>(token_count) * kHidden * sizeof(float);
    const std::size_t route_bytes =
        static_cast<std::size_t>(token_count) * kTopK * sizeof(RouteEntry);
    HIP_CHECK(hipMemsetAsync(device.padded_input.get(),
                             0xff,
                             device.padded_input.size_bytes(),
                             stream));
    HIP_CHECK(hipMemsetAsync(device.gate_up_output.get(),
                             0xff,
                             device.gate_up_output.size_bytes(),
                             stream));
    HIP_CHECK(hipMemsetAsync(device.activation.get(),
                             0xff,
                             device.activation.size_bytes(),
                             stream));
    HIP_CHECK(hipMemsetAsync(device.expert_output.get(),
                             0xff,
                             device.expert_output.size_bytes(),
                             stream));
    HIP_CHECK(
        hipMemsetAsync(device.output.get(), 0xff, output_bytes, stream));
    if (host_visible) {
        std::fill_n(host.output.get(),
                    static_cast<std::size_t>(token_count) * kHidden,
                    std::numeric_limits<float>::quiet_NaN());
        HIP_CHECK(
            hipMemsetAsync(device.input.get(), 0, output_bytes, stream));
        HIP_CHECK(
            hipMemsetAsync(device.routes.get(), 0, route_bytes, stream));
    }
}

template <typename Enqueue>
DownValidation validate_down_path(
    HostData& host,
    DeviceData& device,
    const std::vector<float>& mechanical_reference,
    const std::vector<float>& f32_policy_reference,
    std::uint32_t token_count,
    hipStream_t stream,
    bool enqueue_includes_output_copy,
    Enqueue&& enqueue) {
    const std::size_t output_elements =
        static_cast<std::size_t>(token_count) * kHidden;
    const std::size_t output_bytes = output_elements * sizeof(float);

    auto enqueue_observable = [&] {
        poison_pipeline_state(host,
                              device,
                              token_count,
                              stream,
                              enqueue_includes_output_copy);
        enqueue();
        if (!enqueue_includes_output_copy) {
            HIP_CHECK(hipMemcpyAsync(host.output.get(),
                                     device.output.get(),
                                     output_bytes,
                                     hipMemcpyDeviceToHost,
                                     stream));
        }
    };

    enqueue_observable();
    HIP_CHECK(hipStreamSynchronize(stream));
    DownValidation result;
    result.output.assign(host.output.get(),
                         host.output.get() + output_elements);
    result.mechanical_error = compare_output(
        result.output.data(), mechanical_reference, token_count);
    result.f32_policy_error = compare_output(
        result.output.data(), f32_policy_reference, token_count);

    enqueue_observable();
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

bool meets_correctness_gate(const DownValidation& result) {
    constexpr double maximum_relative_l2 = 5.0e-4;
    constexpr double maximum_absolute = 1.0e-3;
    constexpr double maximum_f32_policy_relative_l2 = 5.0e-3;
    constexpr double maximum_f32_policy_absolute = 1.0e-4;
    return result.mechanical_error.all_finite &&
           result.mechanical_error.relative_l2 <= maximum_relative_l2 &&
           result.mechanical_error.max_absolute <= maximum_absolute &&
           result.bit_deterministic && result.f32_policy_error.all_finite &&
           result.f32_policy_error.relative_l2 <=
               maximum_f32_policy_relative_l2 &&
           result.f32_policy_error.max_absolute <=
               maximum_f32_policy_absolute;
}

void print_validation(std::string_view variant,
                      std::uint32_t token_count,
                      std::string_view path,
                      std::string_view envelope,
                      const DownValidation& result,
                      bool control_bit_match) {
    std::cout << variant << ',' << token_count << ',' << path << ','
              << envelope << ',' << std::scientific << std::setprecision(6)
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
PairedTiming benchmark_down_pair(const Options& options,
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
            control_ms =
                measure_burst(options.burst, stream, enqueue_control);
            candidate_ms =
                measure_burst(options.burst, stream, enqueue_candidate);
        } else {
            candidate_ms =
                measure_burst(options.burst, stream, enqueue_candidate);
            control_ms =
                measure_burst(options.burst, stream, enqueue_control);
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
    std::cout << token_count << ",direct," << envelope << ',' << std::fixed
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
              << " submission=direct envelope=" << envelope
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
        bool explicit_warmups = false;
        bool explicit_samples = false;
        bool explicit_burst = false;
        bool explicit_tokens = false;
        for (int index = 1; index < argc; ++index) {
            const std::string_view argument(argv[index]);
            if (argument == "--help" || argument == "-h") {
                std::cout
                    << "Usage: rocwmma_routed_moe_down_reduce_fused_bench "
                       "[options]\n"
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
            explicit_warmups = explicit_warmups || argument == "--warmups";
            explicit_samples = explicit_samples || argument == "--samples";
            explicit_burst = explicit_burst || argument == "--burst";
            explicit_tokens = explicit_tokens || argument == "--tokens";
        }

        Options options = parse_options(argc, argv);
        if (quick_mode &&
            (explicit_warmups || explicit_samples || explicit_burst)) {
            throw std::runtime_error(
                "--quick cannot be combined with --warmups, --samples, or "
                "--burst; quick output is smoke-only");
        }
        if (!quick_mode && !explicit_samples) {
            options.samples = 12;
        }
        if (!quick_mode && !explicit_burst) {
            options.burst = 16;
        }
        if (!quick_mode && options.warmups < 4) {
            throw std::runtime_error(
                "down-fusion qualification requires --warmups >= 4 outside "
                "--quick");
        }
        if (!quick_mode &&
            (options.samples < 12 || options.samples % 4 != 0)) {
            throw std::runtime_error(
                "down-fusion qualification requires --samples >= 12 and a "
                "multiple of 4 outside --quick");
        }
        if (!quick_mode && options.burst < 16) {
            throw std::runtime_error(
                "down-fusion qualification requires --burst >= 16 outside "
                "--quick");
        }
        if (!quick_mode && explicit_tokens) {
            throw std::runtime_error(
                "down-fusion qualification requires complete "
                "N={1,2,4,8,16}; --tokens is smoke-only");
        }
        const std::vector<std::uint32_t> qualification_tokens = {
            1, 2, 4, 8, 16};
        if (!quick_mode && options.token_counts != qualification_tokens) {
            throw std::runtime_error(
                "down-fusion qualification requires exactly "
                "N={1,2,4,8,16}");
        }

        int device_index = 0;
        HIP_CHECK(hipGetDevice(&device_index));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device_index));
        const std::string_view architecture(properties.gcnArchName);
        if (architecture != "gfx1151") {
            throw std::runtime_error(
                "benchmark requires exact gfx1151; device reports " +
                std::string(architecture));
        }
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error(
                "gfx1151 rocWMMA requires wave32; device reports " +
                std::to_string(properties.warpSize));
        }

        std::cout << "device=" << properties.name << " arch="
                  << properties.gcnArchName << " warp=" << properties.warpSize
                  << '\n';
        std::cout << "mode=" << (quick_mode ? "smoke" : "qualification")
                  << " qualification_grade="
                  << (quick_mode ? "false" : "true") << '\n';
        std::cout << "shape: H=" << kHidden << " F=" << kFfn
                  << " K=" << kTopK << " experts=" << kExperts
                  << " padded_tokens=" << kPaddedTokens << '\n';
        std::cout << "control: launches/operator=5 "
                     "[f32_to_bf16,gate_up_wmma,silu_to_bf16,down_wmma,"
                     "route_reduce]\n";
        std::cout << "candidate: launches/operator=4 "
                     "[f32_to_bf16,gate_up_wmma,silu_to_bf16,"
                     "fused_down_route_wmma]\n";
        std::cout << "candidate fusion: grid=128 blocks block=256 "
                     "eight_wave32_experts=8 lds_bytes=8192 one_barrier "
                     "route_reduction_order=rank\n";
        std::cout << "correctness_route_variants=permuted,reversed,"
                     "duplicate_missing,one_hot\n";
        std::cout << "timing_route_variant=permuted submission=direct "
                     "paired_chrono_wall_clock=true HIP_events=false warmups="
                  << options.warmups << " samples=" << options.samples
                  << " burst=" << options.burst << '\n';

        HostData host;
        DeviceData device;
        Stream stream;
        const std::vector<RouteVariant> route_variants =
            make_route_variants(host);
        require_route_variants_valid(route_variants);

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
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        std::cout << "persistent_device_MiB=" << mib(persistent_bytes(device))
                  << " persistent_weight_MiB="
                  << mib(device.gate_up_weights.size_bytes() +
                         device.down_weights.size_bytes())
                  << '\n';

        std::cout << "route_variant,N,path,envelope,mechanical_relative_l2,"
                     "mechanical_max_abs,f32_policy_relative_l2,"
                     "f32_policy_max_abs,checksum,bit_deterministic,"
                     "control_bit_match\n";
        bool valid = true;
        for (const RouteVariant& variant : route_variants) {
            install_route_variant(host, variant);
            const std::vector<float> mechanical_reference =
                cpu_reference(host, true);
            const std::vector<float> f32_policy_reference =
                cpu_reference(host, false);

            for (const std::uint32_t token_count : options.token_counts) {
                upload_resident_inputs(
                    host, device, token_count, stream.get());
                const DownValidation control_resident = validate_down_path(
                    host,
                    device,
                    mechanical_reference,
                    f32_policy_reference,
                    token_count,
                    stream.get(),
                    false,
                    [&] {
                        launch_pipeline(device, token_count, stream.get());
                    });
                const DownValidation candidate_resident = validate_down_path(
                    host,
                    device,
                    mechanical_reference,
                    f32_policy_reference,
                    token_count,
                    stream.get(),
                    false,
                    [&] {
                        launch_down_fused_pipeline(
                            device, token_count, stream.get());
                    });
                const bool resident_match = exact_match(
                    control_resident.output, candidate_resident.output);

                const DownValidation control_host_visible =
                    validate_down_path(
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
                const DownValidation candidate_host_visible =
                    validate_down_path(
                        host,
                        device,
                        mechanical_reference,
                        f32_policy_reference,
                        token_count,
                        stream.get(),
                        true,
                        [&] {
                            enqueue_down_fused_host_visible_operator(
                                device, host, token_count, stream.get());
                        });
                const bool host_visible_match = exact_match(
                    control_host_visible.output,
                    candidate_host_visible.output);

                valid = valid && meets_correctness_gate(control_resident) &&
                        meets_correctness_gate(candidate_resident) &&
                        resident_match &&
                        meets_correctness_gate(control_host_visible) &&
                        meets_correctness_gate(candidate_host_visible) &&
                        host_visible_match;

                print_validation(variant.name,
                                 token_count,
                                 "control",
                                 "resident",
                                 control_resident,
                                 true);
                print_validation(variant.name,
                                 token_count,
                                 "candidate",
                                 "resident",
                                 candidate_resident,
                                 resident_match);
                print_validation(variant.name,
                                 token_count,
                                 "control",
                                 "host_visible",
                                 control_host_visible,
                                 true);
                print_validation(variant.name,
                                 token_count,
                                 "candidate",
                                 "host_visible",
                                 candidate_host_visible,
                                 host_visible_match);
            }
        }

        if (!valid) {
            std::cerr
                << "validation failed: every route variant and N requires "
                   "finite output, mechanical relative L2 <= 5e-4, max abs "
                   "<= 1e-3, F32-policy relative L2 <= 5e-3 and max abs <= "
                   "1e-4, repeated bit determinism, and exact candidate/"
                   "control agreement\n";
            return 2;
        }

        install_route_variant(host, route_variants.front());
        std::cout << "N,submission,envelope,control_median_ms,control_p90_ms,"
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
            upload_resident_inputs(host, device, token_count, stream.get());
            const PairedTiming resident = benchmark_down_pair(
                options,
                stream.get(),
                [&] { launch_pipeline(device, token_count, stream.get()); },
                [&] {
                    launch_down_fused_pipeline(
                        device, token_count, stream.get());
                });
            const PairedTiming host_visible = benchmark_down_pair(
                options,
                stream.get(),
                [&] {
                    enqueue_host_visible_operator(
                        device, host, token_count, stream.get());
                },
                [&] {
                    enqueue_down_fused_host_visible_operator(
                        device, host, token_count, stream.get());
                });
            print_paired_timing(token_count, "resident", resident);
            print_paired_timing(
                token_count, "host_visible", host_visible);

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
                << "qualification run failed: every direct resident and "
                   "host-visible N=1/N=16 cell requires paired median "
                   "speedup >= 1.03 and candidate p90 <= 1.05x control p90\n";
            return 3;
        }
        if (!quick_mode) {
            std::cout
                << "single_process_run_gate=pass "
                   "overall_qualification=pending_external_aggregate "
                   "aggregate_gate="
                   "three_guarded_runs each_envelope_median_of_three_N1>=1.10 "
                   "each_envelope_median_of_three_N16>=1.08\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
