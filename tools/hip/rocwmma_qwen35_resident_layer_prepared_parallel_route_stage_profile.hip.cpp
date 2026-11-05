// Wall-clock stage attribution for the qualified prepared-row-major WMMA plus
// parallel-router real-weight layer candidate.
//
// The build wrapper materializes a hash-checked, main-renamed view of the
// frozen combined benchmark as q35_combined_components.generated.hpp. This
// profiler therefore reuses the exact kernels and launch helpers without
// editing the qualified source. HIP event/profiler timestamps are unusable on
// this APU stack, so each observation is chrono around enqueue + stream sync.

#include "q35_combined_components.generated.hpp"

namespace {

template <typename Launch>
double combined_profile_launch(hipStream_t stream, Launch&& launch) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto begin = std::chrono::steady_clock::now();
    launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double combined_profile_idle_sync(hipStream_t stream) {
    const auto begin = std::chrono::steady_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

struct PairedStageSamples {
    std::vector<double> raw;
    std::vector<double> signed_idle_adjusted;
};

struct CombinedStageSamples {
    std::vector<double> idle_sync;
    PairedStageSamples rmsnorm;
    PairedStageSamples router;
    PairedStageSamples serial_topk;
    PairedStageSamples parallel_topk;
    PairedStageSamples prepared_gate_up;
    PairedStageSamples silu;
    PairedStageSamples prepared_down;
    PairedStageSamples reduce;
    PairedStageSamples serial_full_chain;
    PairedStageSamples full_chain;
    std::vector<double> serial_burst_us;
    std::vector<double> parallel_burst_us;
};

double combined_median_us(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename Launch>
void record_paired_stage(hipStream_t stream,
                         Launch&& launch,
                         bool launch_first,
                         PairedStageSamples& samples,
                         std::vector<double>& all_idle_samples) {
    double observed = 0.0;
    double idle = 0.0;
    if (launch_first) {
        observed = combined_profile_launch(stream, launch);
        idle = combined_profile_idle_sync(stream);
    } else {
        idle = combined_profile_idle_sync(stream);
        observed = combined_profile_launch(stream, launch);
    }
    samples.raw.push_back(observed);
    samples.signed_idle_adjusted.push_back(observed - idle);
    all_idle_samples.push_back(idle);
}

struct StageMedians {
    double raw_us = 0.0;
    double signed_adjusted_us = 0.0;
};

StageMedians print_combined_stage(std::string_view name,
                                  const PairedStageSamples& samples) {
    const double observed = combined_median_us(samples.raw);
    const double adjusted =
        combined_median_us(samples.signed_idle_adjusted);
    std::cout << name << "_launch_sync_median_us=" << std::fixed
              << std::setprecision(3) << observed
              << " paired_signed_idle_adjusted_median_us=" << adjusted
              << '\n';
    return {observed, adjusted};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error(
                "usage: rocwmma_qwen35_resident_layer_prepared_parallel_"
                "route_stage_profile <layer.pack>");
        }
        constexpr int kProfileWarmups = 8;
        constexpr int kProfileSamples = 31;

        const MappedPack pack(argv[1]);
        int device_index = 0;
        HIP_CHECK(hipGetDevice(&device_index));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device_index));
        const std::string_view architecture(properties.gcnArchName);
        if (architecture != "gfx1151" &&
            !architecture.starts_with("gfx1151:")) {
            throw std::runtime_error(
                "combined stage profile requires exact gfx1151");
        }
        if (properties.warpSize != static_cast<int>(kWaveSize)) {
            throw std::runtime_error(
                "combined stage profile requires wave32");
        }
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

        // Untimed guard: the profiler must still exercise the exact qualified
        // chain and prepared bytes before reporting stage attribution.
        const CpuReference reference = cpu_reference(pack, hidden);
        const bool prepared_bytes_exact = validate_combined_selected_bytes(
            pack, prepared, reference.route_ids, stream.get());
        poison_scratch(state, stream.get());
        enqueue_prepared_parallel(checkpoint, prepared, state, stream.get());
        const std::vector<float> validation_output =
            read_output(state, stream.get());
        const ObservedRoute validation_route = read_route(state, stream.get());
        const ErrorMetrics validation_error =
            compare_output(validation_output, reference.output);
        bool route_exact = validation_route.status == 0;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_exact = route_exact &&
                          validation_route.ids[rank] ==
                              reference.route_ids[rank];
        }

        poison_scratch(state, stream.get());
        enqueue_resident(checkpoint, state, stream.get());
        const std::vector<float> native_serial_output =
            read_output(state, stream.get());
        poison_scratch(state, stream.get());
        enqueue_prepared_serial(checkpoint, prepared, state, stream.get());
        const std::vector<float> prepared_serial_first =
            read_output(state, stream.get());
        const ObservedRoute prepared_serial_route_first =
            read_route(state, stream.get());
        enqueue_prepared_serial(checkpoint, prepared, state, stream.get());
        const std::vector<float> prepared_serial_second =
            read_output(state, stream.get());
        const ObservedRoute prepared_serial_route_second =
            read_route(state, stream.get());
        poison_scratch(state, stream.get());
        enqueue_prepared_parallel(checkpoint, prepared, state, stream.get());
        const std::vector<float> parallel_repeat =
            read_output(state, stream.get());
        const ObservedRoute parallel_repeat_route =
            read_route(state, stream.get());

        bool serial_route_exact = prepared_serial_route_first.status == 0;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            serial_route_exact =
                serial_route_exact &&
                prepared_serial_route_first.ids[rank] ==
                    reference.route_ids[rank];
        }
        const bool native_prepared_serial_bit_exact =
            std::memcmp(native_serial_output.data(),
                        prepared_serial_first.data(),
                        native_serial_output.size() * sizeof(float)) == 0;
        const bool serial_bit_deterministic =
            std::memcmp(prepared_serial_first.data(),
                        prepared_serial_second.data(),
                        prepared_serial_first.size() * sizeof(float)) == 0 &&
            combined_route_bit_exact(prepared_serial_route_first,
                                     prepared_serial_route_second);
        const bool parallel_bit_deterministic =
            std::memcmp(validation_output.data(),
                        parallel_repeat.data(),
                        validation_output.size() * sizeof(float)) == 0 &&
            combined_route_bit_exact(validation_route,
                                     parallel_repeat_route);
        const ErrorMetrics serial_error =
            compare_output(prepared_serial_first, reference.output);
        const ErrorMetrics parallel_serial_error =
            compare_output(validation_output, prepared_serial_first);
        constexpr double kRelativeL2Limit = 5.0e-4;
        constexpr double kMaxAbsoluteLimit = 1.0e-3;
        constexpr double kParallelSerialRelativeL2Limit = 2.0e-6;
        constexpr double kParallelSerialMaxAbsoluteLimit = 2.0e-5;
        if (!prepared_bytes_exact || !route_exact || !serial_route_exact ||
            !native_prepared_serial_bit_exact ||
            !serial_bit_deterministic || !parallel_bit_deterministic ||
            !validation_error.all_finite || !serial_error.all_finite ||
            !parallel_serial_error.all_finite ||
            validation_error.relative_l2 > kRelativeL2Limit ||
            validation_error.max_absolute > kMaxAbsoluteLimit ||
            serial_error.relative_l2 > kRelativeL2Limit ||
            serial_error.max_absolute > kMaxAbsoluteLimit ||
            parallel_serial_error.relative_l2 >
                kParallelSerialRelativeL2Limit ||
            parallel_serial_error.max_absolute >
                kParallelSerialMaxAbsoluteLimit) {
            throw std::runtime_error(
                "combined candidate failed profiler pre-validation");
        }

        auto rmsnorm = [&] {
            hipLaunchKernelGGL(q35_rmsnorm_f32_to_f32_bf16,
                               dim3(1),
                               dim3(kElementThreads),
                               0,
                               stream.get(),
                               state.hidden.get(),
                               checkpoint.norm.get(),
                               state.normalized.get(),
                               state.padded_input.get());
            HIP_CHECK(hipGetLastError());
        };
        auto router = [&] {
            hipLaunchKernelGGL(q35_router_and_shared_gate_f32,
                               dim3(kAllExperts + 1),
                               dim3(kElementThreads),
                               0,
                               stream.get(),
                               state.normalized.get(),
                               checkpoint.router.get(),
                               checkpoint.shared_gate.get(),
                               state.logits.get(),
                               state.shared_logit.get());
            HIP_CHECK(hipGetLastError());
        };
        auto parallel_topk = [&] {
            hipLaunchKernelGGL(q35_prepared_stable_softmax_topk8_parallel,
                               dim3(1),
                               dim3(kCombinedRouteThreads),
                               0,
                               stream.get(),
                               state.logits.get(),
                               state.shared_logit.get(),
                               state.probabilities.get(),
                               state.route_ids.get(),
                               state.route_weights.get(),
                               state.shared_scale.get(),
                               state.route_margin.get(),
                               state.route_status.get());
            HIP_CHECK(hipGetLastError());
        };
        auto serial_topk = [&] {
            hipLaunchKernelGGL(q35_stable_softmax_topk8_prepared_serial,
                               dim3(1),
                               dim3(1),
                               0,
                               stream.get(),
                               state.logits.get(),
                               state.shared_logit.get(),
                               state.probabilities.get(),
                               state.route_ids.get(),
                               state.route_weights.get(),
                               state.shared_scale.get(),
                               state.route_margin.get(),
                               state.route_status.get());
            HIP_CHECK(hipGetLastError());
        };
        auto prepared_gate_up = [&] {
            hipLaunchKernelGGL(q35_combined_prepared_gate_up_wmma,
                               dim3(kFfn / kTile, kComputeSlots, 2),
                               dim3(kWaveSize),
                               0,
                               stream.get(),
                               state.padded_input.get(),
                               prepared.gate_up.get(),
                               state.route_ids.get(),
                               state.gate_up_output.get());
            HIP_CHECK(hipGetLastError());
        };
        auto silu = [&] {
            constexpr std::uint32_t elements =
                kComputeSlots * kPaddedTokens * kFfn;
            hipLaunchKernelGGL(q35_silu_multiply_to_bf16,
                               dim3((elements + kElementThreads - 1) /
                                    kElementThreads),
                               dim3(kElementThreads),
                               0,
                               stream.get(),
                               state.gate_up_output.get(),
                               state.activation.get());
            HIP_CHECK(hipGetLastError());
        };
        auto prepared_down = [&] {
            hipLaunchKernelGGL(q35_combined_prepared_down_wmma,
                               dim3(kHidden / kTile, kComputeSlots),
                               dim3(kWaveSize),
                               0,
                               stream.get(),
                               state.activation.get(),
                               prepared.down.get(),
                               state.route_ids.get(),
                               state.expert_output.get());
            HIP_CHECK(hipGetLastError());
        };
        auto reduce = [&] {
            hipLaunchKernelGGL(q35_route_shared_residual_reduce,
                               dim3((kHidden + kElementThreads - 1) /
                                    kElementThreads),
                               dim3(kElementThreads),
                               0,
                               stream.get(),
                               state.expert_output.get(),
                               state.route_status.get(),
                               state.route_weights.get(),
                               state.shared_scale.get(),
                               state.hidden.get(),
                               state.output.get());
            HIP_CHECK(hipGetLastError());
        };
        auto full_chain = [&] {
            enqueue_prepared_parallel(
                checkpoint, prepared, state, stream.get());
        };
        auto serial_full_chain = [&] {
            enqueue_prepared_serial(
                checkpoint, prepared, state, stream.get());
        };

        // The manually spelled launches are the subject of this profiler.
        // Prove their wiring is bit-identical to each qualified helper path.
        auto manual_parallel_chain = [&] {
            rmsnorm();
            router();
            parallel_topk();
            prepared_gate_up();
            silu();
            prepared_down();
            reduce();
        };
        auto manual_serial_chain = [&] {
            rmsnorm();
            router();
            serial_topk();
            prepared_gate_up();
            silu();
            prepared_down();
            reduce();
        };
        poison_scratch(state, stream.get());
        manual_parallel_chain();
        const std::vector<float> manual_parallel_output =
            read_output(state, stream.get());
        const ObservedRoute manual_parallel_route =
            read_route(state, stream.get());
        poison_scratch(state, stream.get());
        manual_serial_chain();
        const std::vector<float> manual_serial_output =
            read_output(state, stream.get());
        const ObservedRoute manual_serial_route =
            read_route(state, stream.get());
        const bool manual_parallel_bit_exact =
            std::memcmp(manual_parallel_output.data(),
                        validation_output.data(),
                        manual_parallel_output.size() * sizeof(float)) == 0 &&
            combined_route_bit_exact(manual_parallel_route,
                                     validation_route);
        const bool manual_serial_bit_exact =
            std::memcmp(manual_serial_output.data(),
                        prepared_serial_first.data(),
                        manual_serial_output.size() * sizeof(float)) == 0 &&
            combined_route_bit_exact(manual_serial_route,
                                     prepared_serial_route_first);
        if (!manual_parallel_bit_exact || !manual_serial_bit_exact) {
            throw std::runtime_error(
                "manually staged replay differs from qualified helper path");
        }

        for (int index = 0; index < kProfileWarmups; ++index) {
            if (index % 2 == 0) {
                serial_full_chain();
                full_chain();
            } else {
                full_chain();
                serial_full_chain();
            }
        }
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        CombinedStageSamples timings;
        constexpr int kFullChainBurst = 16;
        for (int index = 0; index < kProfileSamples; ++index) {
            const bool first = index % 2 == 0;
            record_paired_stage(stream.get(),
                                rmsnorm,
                                first,
                                timings.rmsnorm,
                                timings.idle_sync);
            record_paired_stage(stream.get(),
                                router,
                                !first,
                                timings.router,
                                timings.idle_sync);
            if (index % 2 == 0) {
                record_paired_stage(stream.get(),
                                    serial_topk,
                                    first,
                                    timings.serial_topk,
                                    timings.idle_sync);
                record_paired_stage(stream.get(),
                                    parallel_topk,
                                    !first,
                                    timings.parallel_topk,
                                    timings.idle_sync);
            } else {
                record_paired_stage(stream.get(),
                                    parallel_topk,
                                    first,
                                    timings.parallel_topk,
                                    timings.idle_sync);
                record_paired_stage(stream.get(),
                                    serial_topk,
                                    !first,
                                    timings.serial_topk,
                                    timings.idle_sync);
                // Restore candidate route metadata for the suffix stages.
                parallel_topk();
                HIP_CHECK(hipStreamSynchronize(stream.get()));
            }
            record_paired_stage(stream.get(),
                                prepared_gate_up,
                                first,
                                timings.prepared_gate_up,
                                timings.idle_sync);
            record_paired_stage(stream.get(),
                                silu,
                                !first,
                                timings.silu,
                                timings.idle_sync);
            record_paired_stage(stream.get(),
                                prepared_down,
                                first,
                                timings.prepared_down,
                                timings.idle_sync);
            record_paired_stage(stream.get(),
                                reduce,
                                !first,
                                timings.reduce,
                                timings.idle_sync);
            if (index % 2 == 0) {
                record_paired_stage(stream.get(),
                                    serial_full_chain,
                                    first,
                                    timings.serial_full_chain,
                                    timings.idle_sync);
                record_paired_stage(stream.get(),
                                    full_chain,
                                    !first,
                                    timings.full_chain,
                                    timings.idle_sync);
                timings.serial_burst_us.push_back(
                    1000.0 * timed_burst(kFullChainBurst,
                                         stream.get(),
                                         serial_full_chain));
                timings.parallel_burst_us.push_back(
                    1000.0 * timed_burst(
                                 kFullChainBurst, stream.get(), full_chain));
            } else {
                record_paired_stage(stream.get(),
                                    full_chain,
                                    first,
                                    timings.full_chain,
                                    timings.idle_sync);
                record_paired_stage(stream.get(),
                                    serial_full_chain,
                                    !first,
                                    timings.serial_full_chain,
                                    timings.idle_sync);
                timings.parallel_burst_us.push_back(
                    1000.0 * timed_burst(
                                 kFullChainBurst, stream.get(), full_chain));
                timings.serial_burst_us.push_back(
                    1000.0 * timed_burst(kFullChainBurst,
                                         stream.get(),
                                         serial_full_chain));
            }
        }

        full_chain();
        const std::vector<float> post_profile_parallel_output =
            read_output(state, stream.get());
        const ObservedRoute post_profile_parallel_route =
            read_route(state, stream.get());
        serial_full_chain();
        const std::vector<float> post_profile_serial_output =
            read_output(state, stream.get());
        const ObservedRoute post_profile_serial_route =
            read_route(state, stream.get());
        const bool post_profile_parallel_bit_exact =
            std::memcmp(post_profile_parallel_output.data(),
                        validation_output.data(),
                        post_profile_parallel_output.size() * sizeof(float)) ==
                0 &&
            combined_route_bit_exact(post_profile_parallel_route,
                                     validation_route);
        const bool post_profile_serial_bit_exact =
            std::memcmp(post_profile_serial_output.data(),
                        prepared_serial_first.data(),
                        post_profile_serial_output.size() * sizeof(float)) ==
                0 &&
            combined_route_bit_exact(post_profile_serial_route,
                                     prepared_serial_route_first);
        if (!post_profile_parallel_bit_exact ||
            !post_profile_serial_bit_exact) {
            throw std::runtime_error(
                "qualified outputs changed during stage profiling");
        }

        const double idle = combined_median_us(timings.idle_sync);
        std::cout << "method=chrono_enqueue_plus_stream_sync samples="
                  << kProfileSamples
                  << " warmups=" << kProfileWarmups
                  << " profiler_gpu_timestamps_unavailable=true "
                     "pack_mapping=read_only weights_mutated=false "
                     "prepared_copy_derived_once=true "
                     "idle_pairs_counterbalanced=true\n";
        std::cout << "preparation_wall_us=" << std::fixed
                  << std::setprecision(3)
                  << std::chrono::duration<double, std::micro>(prepare_end -
                                                               prepare_begin)
                         .count()
                  << " excluded_from_inference_profile=true\n";
        std::cout << "validation_route_exact=" << std::boolalpha
                  << route_exact
                  << " selected_and_shared_prepared_bytes_exact="
                  << prepared_bytes_exact
                  << " validation_relative_l2=" << std::scientific
                  << validation_error.relative_l2
                  << " validation_max_abs=" << validation_error.max_absolute
                  << " validation_checksum=" << validation_error.checksum
                  << " native_prepared_serial_bit_exact="
                  << native_prepared_serial_bit_exact
                  << " serial_bit_deterministic="
                  << serial_bit_deterministic
                  << " parallel_bit_deterministic="
                  << parallel_bit_deterministic
                  << " manual_serial_bit_exact=" << manual_serial_bit_exact
                  << " manual_parallel_bit_exact="
                  << manual_parallel_bit_exact
                  << " post_profile_serial_bit_exact="
                  << post_profile_serial_bit_exact
                  << " post_profile_parallel_bit_exact="
                  << post_profile_parallel_bit_exact
                  << '\n';
        std::cout << "idle_sync_median_us=" << std::fixed
                  << std::setprecision(3) << idle << '\n';
        const StageMedians rmsnorm_timing =
            print_combined_stage("rmsnorm", timings.rmsnorm);
        const StageMedians router_timing = print_combined_stage(
            "router_shared_gate", timings.router);
        const StageMedians serial_route = print_combined_stage(
            "serial_softmax_topk_control", timings.serial_topk);
        const StageMedians parallel_route = print_combined_stage(
            "parallel_softmax_topk", timings.parallel_topk);
        const StageMedians gate_up_timing = print_combined_stage(
            "prepared_gate_up_wmma", timings.prepared_gate_up);
        const StageMedians silu_timing =
            print_combined_stage("silu_bf16", timings.silu);
        const StageMedians down_timing = print_combined_stage(
            "prepared_down_wmma", timings.prepared_down);
        const StageMedians reduce_timing = print_combined_stage(
            "route_shared_residual", timings.reduce);
        const StageMedians serial_chain = print_combined_stage(
            "full_prepared_serial_chain_control",
            timings.serial_full_chain);
        const StageMedians chain = print_combined_stage(
            "full_prepared_parallel_chain", timings.full_chain);
        const double stage_sum =
            rmsnorm_timing.signed_adjusted_us +
            router_timing.signed_adjusted_us +
            parallel_route.signed_adjusted_us +
            gate_up_timing.signed_adjusted_us +
            silu_timing.signed_adjusted_us +
            down_timing.signed_adjusted_us +
            reduce_timing.signed_adjusted_us;
        const double serial_burst =
            combined_median_us(timings.serial_burst_us);
        const double parallel_burst =
            combined_median_us(timings.parallel_burst_us);
        std::cout << "sum_of_paired_signed_stage_medians_us=" << std::fixed
                  << std::setprecision(3) << stage_sum
                  << " full_chain_paired_signed_us="
                  << chain.signed_adjusted_us
                  << " stage_sum_minus_chain_us="
                  << stage_sum - chain.signed_adjusted_us
                  << " isolated_route_delta_us="
                  << serial_route.signed_adjusted_us -
                         parallel_route.signed_adjusted_us
                  << " full_chain_delta_us="
                  << serial_chain.signed_adjusted_us -
                         chain.signed_adjusted_us
                  << " serial_full_chain_burst16_median_us=" << serial_burst
                  << " parallel_full_chain_burst16_median_us="
                  << parallel_burst
                  << " burst16_serial_over_parallel="
                  << serial_burst / parallel_burst
                  << " subtraction_is_diagnostic_not_additive=true\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
