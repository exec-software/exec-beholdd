// Wall-clock stage attribution for the validated real-weight resident layer.
//
// HIP event timestamps and rocprofiler kernel timestamps are zero on this APU
// stack. Keep the validated implementation frozen, include it verbatim, and
// measure one launch plus a stream synchronization for each stage instead.

#define main beholdd_q35_resident_layer_control_main
#include "rocwmma_qwen35_resident_layer_bench.hip.cpp"
#undef main

namespace {

template <typename Launch>
double profile_launch(hipStream_t stream, Launch&& launch) {
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto begin = std::chrono::steady_clock::now();
    launch();
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

double profile_idle_sync(hipStream_t stream) {
    const auto begin = std::chrono::steady_clock::now();
    HIP_CHECK(hipStreamSynchronize(stream));
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

struct StageSamples {
    std::vector<double> idle_sync;
    std::vector<double> rmsnorm;
    std::vector<double> router;
    std::vector<double> topk;
    std::vector<double> gate_up;
    std::vector<double> silu;
    std::vector<double> down;
    std::vector<double> reduce;
};

double median_us(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void print_stage(std::string_view name,
                 const std::vector<double>& samples,
                 double idle_sync_us) {
    const double observed = median_us(samples);
    std::cout << name << "_launch_sync_median_us=" << std::fixed
              << std::setprecision(3) << observed
              << " idle_sync_adjusted_us=" << std::max(0.0, observed - idle_sync_us)
              << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error(
                "usage: rocwmma_qwen35_resident_layer_stage_profile <layer.pack>");
        }
        constexpr int warmups = 8;
        constexpr int samples = 31;

        const MappedPack pack(argv[1]);
        int device_index = 0;
        HIP_CHECK(hipGetDevice(&device_index));
        hipDeviceProp_t properties{};
        HIP_CHECK(hipGetDeviceProperties(&properties, device_index));
        const std::string_view architecture(properties.gcnArchName);
        if (architecture != "gfx1151" &&
            !architecture.starts_with("gfx1151:")) {
            throw std::runtime_error("stage profile requires gfx1151");
        }
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

        auto rmsnorm = [&] {
            hipLaunchKernelGGL(q35_rmsnorm_f32_to_f32_bf16,
                               dim3(1),
                               dim3(kElementThreads),
                               0,
                               stream.get(),
                               state.hidden.get(),
                               layer.norm.get(),
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
                               layer.router.get(),
                               layer.shared_gate.get(),
                               state.logits.get(),
                               state.shared_logit.get());
            HIP_CHECK(hipGetLastError());
        };
        auto topk = [&] {
            hipLaunchKernelGGL(q35_stable_softmax_topk8,
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
        auto gate_up = [&] {
            hipLaunchKernelGGL(q35_selected_shared_gate_up_wmma,
                               dim3(kFfn / kTile, kComputeSlots, 2),
                               dim3(kWaveSize),
                               0,
                               stream.get(),
                               state.padded_input.get(),
                               layer.gate_up.get(),
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
        auto down = [&] {
            hipLaunchKernelGGL(q35_selected_shared_down_wmma,
                               dim3(kHidden / kTile, kComputeSlots),
                               dim3(kWaveSize),
                               0,
                               stream.get(),
                               state.activation.get(),
                               layer.down.get(),
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

        for (int index = 0; index < warmups; ++index) {
            rmsnorm();
            router();
            topk();
            gate_up();
            silu();
            down();
            reduce();
        }
        HIP_CHECK(hipStreamSynchronize(stream.get()));

        StageSamples timings;
        for (int index = 0; index < samples; ++index) {
            timings.idle_sync.push_back(profile_idle_sync(stream.get()));
            timings.rmsnorm.push_back(profile_launch(stream.get(), rmsnorm));
            timings.router.push_back(profile_launch(stream.get(), router));
            timings.topk.push_back(profile_launch(stream.get(), topk));
            timings.gate_up.push_back(profile_launch(stream.get(), gate_up));
            timings.silu.push_back(profile_launch(stream.get(), silu));
            timings.down.push_back(profile_launch(stream.get(), down));
            timings.reduce.push_back(profile_launch(stream.get(), reduce));
        }

        const double idle = median_us(timings.idle_sync);
        std::cout << "method=chrono_launch_plus_stream_sync samples=" << samples
                  << " profiler_gpu_timestamps_unavailable=true\n";
        std::cout << "idle_sync_median_us=" << std::fixed
                  << std::setprecision(3) << idle << '\n';
        print_stage("rmsnorm", timings.rmsnorm, idle);
        print_stage("router_shared_gate", timings.router, idle);
        print_stage("softmax_topk", timings.topk, idle);
        print_stage("gate_up_wmma", timings.gate_up, idle);
        print_stage("silu_bf16", timings.silu, idle);
        print_stage("down_wmma", timings.down, idle);
        print_stage("route_shared_residual", timings.reduce, idle);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
