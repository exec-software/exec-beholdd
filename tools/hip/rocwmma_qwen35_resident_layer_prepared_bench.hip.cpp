// Paired checkpoint-native-column-major versus bit-preserving prepared-row-major
// real-weight Qwen3.6 resident-layer experiment.

#define main beholdd_q35_checkpoint_native_control_main
#include "rocwmma_qwen35_resident_layer_bench.hip.cpp"
#undef main

namespace {

extern "C" __global__ void q35_prepare_gate_up_row_major(
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

extern "C" __global__ void q35_prepare_down_row_major(
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
void q35_prepared_selected_shared_gate_up_wmma(
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
void q35_prepared_selected_shared_down_wmma(
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

struct PreparedRowMajorLayer {
    PreparedRowMajorLayer() : gate_up(kGateUpElements), down(kDownElements) {}

    DeviceBuffer<Bf16> gate_up;
    DeviceBuffer<Bf16> down;
};

void require_paired_memory_headroom() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
    const std::size_t one_weight_set =
        kNormBytes + kRouterBytes + kSharedGateBytes + kGateUpBytes +
        kDownBytes;
    const std::size_t required =
        one_weight_set + kGateUpBytes + kDownBytes + state_bytes();
    if (required > std::numeric_limits<std::size_t>::max() -
                       kHeadroomBytes ||
        free_bytes < required + kHeadroomBytes) {
        throw std::runtime_error(
            "insufficient VRAM for paired native/prepared weights plus headroom");
    }
    std::cout << "vram_total_MiB=" << total_bytes / (1024 * 1024)
              << " vram_free_before_MiB=" << free_bytes / (1024 * 1024)
              << " paired_required_MiB=" << required / (1024 * 1024)
              << " production_prepared_weight_MiB="
              << one_weight_set / (1024 * 1024)
              << " reserved_headroom_MiB=" << kHeadroomBytes / (1024 * 1024)
              << '\n';
}

void prepare_row_major(const ResidentLayer& checkpoint,
                       PreparedRowMajorLayer& prepared,
                       hipStream_t stream) {
    constexpr std::uint32_t threads = 256;
    constexpr std::uint32_t blocks =
        (kExpertMatrixElements + threads - 1) / threads;
    hipLaunchKernelGGL(q35_prepare_gate_up_row_major,
                       dim3(blocks, 2 * kPackedSlots),
                       dim3(threads),
                       0,
                       stream,
                       checkpoint.gate_up.get(),
                       prepared.gate_up.get());
    HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(q35_prepare_down_row_major,
                       dim3(blocks, kPackedSlots),
                       dim3(threads),
                       0,
                       stream,
                       checkpoint.down.get(),
                       prepared.down.get());
    HIP_CHECK(hipGetLastError());
}

void launch_prepared_suffix(const PreparedRowMajorLayer& prepared,
                            ResidentState& state,
                            hipStream_t stream) {
    hipLaunchKernelGGL(q35_prepared_selected_shared_gate_up_wmma,
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
    hipLaunchKernelGGL(q35_prepared_selected_shared_down_wmma,
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

void enqueue_prepared(const ResidentLayer& checkpoint,
                      const PreparedRowMajorLayer& prepared,
                      ResidentState& state,
                      hipStream_t stream) {
    launch_prefix(checkpoint, state, stream);
    launch_prepared_suffix(prepared, state, stream);
}

bool validate_prepared_selected_bytes(const MappedPack& pack,
                                      const PreparedRowMajorLayer& prepared,
                                      const ObservedRoute& route,
                                      hipStream_t stream) {
    std::vector<Bf16> host(kExpertMatrixElements);
    for (std::uint32_t slot = 0; slot < kComputeSlots; ++slot) {
        const std::uint32_t expert =
            slot < kTopK ? route.ids[slot] : kSharedSlot;
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
            throw std::runtime_error("prepared benchmark requires gfx1151");
        }
        require_paired_memory_headroom();

        Stream stream;
        ResidentLayer checkpoint;
        PreparedRowMajorLayer prepared;
        ResidentState state;
        upload_layer(pack, checkpoint, stream.get());
        HIP_CHECK(hipStreamSynchronize(stream.get()));
        const auto prepare_begin = std::chrono::steady_clock::now();
        prepare_row_major(checkpoint, prepared, stream.get());
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
        std::cout << "preparation_ms=" << std::fixed << std::setprecision(3)
                  << std::chrono::duration<double, std::milli>(prepare_end -
                                                               prepare_begin)
                         .count()
                  << " excluded_from_inference_timing=true bit_rearrangement_only=true\n";

        const CpuReference reference = cpu_reference(pack, hidden);
        enqueue_resident(checkpoint, state, stream.get());
        const std::vector<float> control = read_output(state, stream.get());
        const ObservedRoute route = read_route(state, stream.get());
        enqueue_prepared(checkpoint, prepared, state, stream.get());
        const std::vector<float> candidate = read_output(state, stream.get());
        enqueue_prepared(checkpoint, prepared, state, stream.get());
        const std::vector<float> repeated = read_output(state, stream.get());

        const ErrorMetrics control_error =
            compare_output(control, reference.output);
        const ErrorMetrics candidate_error =
            compare_output(candidate, reference.output);
        const bool control_exact =
            std::memcmp(control.data(), candidate.data(),
                        control.size() * sizeof(float)) == 0;
        const bool deterministic =
            std::memcmp(candidate.data(), repeated.data(),
                        candidate.size() * sizeof(float)) == 0;
        bool route_exact = route.status == 0;
        for (std::uint32_t rank = 0; rank < kTopK; ++rank) {
            route_exact =
                route_exact && route.ids[rank] == reference.route_ids[rank];
        }
        const bool prepared_bytes_exact = validate_prepared_selected_bytes(
            pack, prepared, route, stream.get());
        std::cout << "selected_and_shared_prepared_bytes_exact="
                  << std::boolalpha << prepared_bytes_exact
                  << " route_exact=" << route_exact
                  << " candidate_control_bit_exact=" << control_exact
                  << " candidate_bit_deterministic=" << deterministic << '\n';
        std::cout << "control_relative_l2=" << std::scientific
                  << control_error.relative_l2
                  << " candidate_relative_l2=" << candidate_error.relative_l2
                  << " candidate_max_abs=" << candidate_error.max_absolute
                  << " candidate_all_finite=" << candidate_error.all_finite
                  << '\n';

        for (int index = 0; index < options.warmups; ++index) {
            enqueue_resident(checkpoint, state, stream.get());
            enqueue_prepared(checkpoint, prepared, state, stream.get());
        }
        HIP_CHECK(hipStreamSynchronize(stream.get()));
        std::vector<double> control_samples;
        std::vector<double> candidate_samples;
        control_samples.reserve(options.samples);
        candidate_samples.reserve(options.samples);
        for (int sample = 0; sample < options.samples; ++sample) {
            auto time_control = [&] {
                control_samples.push_back(timed_burst(
                    options.burst,
                    stream.get(),
                    [&] { enqueue_resident(checkpoint, state, stream.get()); }));
            };
            auto time_candidate = [&] {
                candidate_samples.push_back(timed_burst(
                    options.burst,
                    stream.get(),
                    [&] {
                        enqueue_prepared(
                            checkpoint, prepared, state, stream.get());
                    }));
            };
            if (sample % 2 == 0) {
                time_control();
                time_candidate();
            } else {
                time_candidate();
                time_control();
            }
        }
        const TimingStats control_timing = summarize(std::move(control_samples));
        const TimingStats candidate_timing =
            summarize(std::move(candidate_samples));
        print_timing("checkpoint_col_major_control", control_timing);
        print_timing("prepared_row_major_candidate", candidate_timing);
        const double speedup =
            control_timing.median_ms / candidate_timing.median_ms;
        std::cout << "median_speedup=" << std::fixed << std::setprecision(4)
                  << speedup
                  << " prepared_routed_layers_per_second="
                  << 1000.0 / candidate_timing.median_ms
                  << " warmups=" << options.warmups
                  << " samples=" << options.samples
                  << " burst=" << options.burst << '\n';

        constexpr double maximum_relative_l2 = 5.0e-4;
        constexpr double maximum_absolute = 1.0e-3;
        if (!prepared_bytes_exact || !route_exact || !control_exact ||
            !deterministic || !candidate_error.all_finite ||
            candidate_error.relative_l2 > maximum_relative_l2 ||
            candidate_error.max_absolute > maximum_absolute) {
            std::cerr << "FAIL: prepared row-major correctness gate failed\n";
            return 2;
        }
        std::cout << "PASS: bit-preserving preparation and exact paired output\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
