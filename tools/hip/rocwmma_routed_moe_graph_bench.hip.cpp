// Paired HIP Graph replay qualification for the frozen five-launch BF16
// rocWMMA routed-MoE baseline.
//
// The validated baseline remains a separate, immutable source file. Including
// it after renaming only its standalone entry point keeps the direct control
// and graph candidate in one translation unit, with identical kernels,
// arguments, buffers, arithmetic, and stream order.

#define main beholdd_frozen_control_main
#include "rocwmma_routed_moe_bench.hip.cpp"
#undef main

namespace {

struct GraphTopology {
    std::size_t nodes = 0;
    std::size_t kernel_nodes = 0;
    std::size_t memcpy_nodes = 0;
    std::size_t other_nodes = 0;
    std::size_t edges = 0;
    std::size_t roots = 0;
};

struct ExpectedGraphNode {
    hipGraphNodeType type = hipGraphNodeTypeCount;
    std::string_view label;
    void* function = nullptr;
    dim3 grid{};
    dim3 block{};
    unsigned int shared_bytes = 0;
    std::vector<std::vector<unsigned char>> kernel_arguments;
    const void* source = nullptr;
    void* destination = nullptr;
    std::size_t copy_bytes = 0;
    hipMemcpyKind copy_kind = hipMemcpyDefault;
};

template <typename T>
std::vector<unsigned char> argument_bytes(const T& value) {
    std::vector<unsigned char> bytes(sizeof(T));
    std::memcpy(bytes.data(), &value, sizeof(T));
    return bytes;
}

template <typename... Arguments>
ExpectedGraphNode expected_kernel(std::string_view label,
                                  void* function,
                                  dim3 grid,
                                  dim3 block,
                                  const Arguments&... arguments) {
    ExpectedGraphNode node;
    node.type = hipGraphNodeTypeKernel;
    node.label = label;
    node.function = function;
    node.grid = grid;
    node.block = block;
    (node.kernel_arguments.push_back(argument_bytes(arguments)), ...);
    return node;
}

ExpectedGraphNode expected_memcpy(std::string_view label,
                                  const void* source,
                                  void* destination,
                                  std::size_t copy_bytes,
                                  hipMemcpyKind copy_kind) {
    ExpectedGraphNode node;
    node.type = hipGraphNodeTypeMemcpy;
    node.label = label;
    node.source = source;
    node.destination = destination;
    node.copy_bytes = copy_bytes;
    node.copy_kind = copy_kind;
    return node;
}

bool same_dim3(const dim3& left, const dim3& right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

void require_kernel_node(hipGraphNode_t node,
                         const ExpectedGraphNode& expected) {
    hipKernelNodeParams parameters{};
    HIP_CHECK(hipGraphKernelNodeGetParams(node, &parameters));
    if (parameters.func != expected.function ||
        !same_dim3(parameters.gridDim, expected.grid) ||
        !same_dim3(parameters.blockDim, expected.block) ||
        parameters.sharedMemBytes != expected.shared_bytes) {
        throw std::runtime_error(
            std::string(expected.label) +
            " kernel-node parameters do not match the frozen launch");
    }
    if (parameters.kernelParams == nullptr || parameters.extra != nullptr) {
        throw std::runtime_error(
            std::string(expected.label) +
            " kernel-node arguments are not exposed in the expected form");
    }
    for (std::size_t index = 0; index < expected.kernel_arguments.size();
         ++index) {
        if (parameters.kernelParams[index] == nullptr ||
            std::memcmp(parameters.kernelParams[index],
                        expected.kernel_arguments[index].data(),
                        expected.kernel_arguments[index].size()) != 0) {
            throw std::runtime_error(
                std::string(expected.label) + " kernel argument " +
                std::to_string(index) +
                " does not match the frozen launch");
        }
    }
}

void require_memcpy_node(hipGraphNode_t node,
                         const ExpectedGraphNode& expected) {
    hipMemcpy3DParms parameters{};
    HIP_CHECK(hipGraphMemcpyNodeGetParams(node, &parameters));
    const bool zero_positions =
        parameters.srcPos.x == 0 && parameters.srcPos.y == 0 &&
        parameters.srcPos.z == 0 && parameters.dstPos.x == 0 &&
        parameters.dstPos.y == 0 && parameters.dstPos.z == 0;
    if (parameters.srcArray != nullptr || parameters.dstArray != nullptr ||
        parameters.srcPtr.ptr != expected.source ||
        parameters.dstPtr.ptr != expected.destination || !zero_positions ||
        parameters.extent.width != expected.copy_bytes ||
        parameters.extent.height != 1 || parameters.extent.depth != 1 ||
        parameters.kind != expected.copy_kind) {
        throw std::runtime_error(
            std::string(expected.label) +
            " memcpy-node parameters do not match the frozen transfer");
    }
}

GraphTopology inspect_graph(
    hipGraph_t graph,
    const std::vector<ExpectedGraphNode>& expected_nodes) {
    GraphTopology topology;
    HIP_CHECK(hipGraphGetNodes(graph, nullptr, &topology.nodes));
    std::vector<hipGraphNode_t> nodes(topology.nodes);
    std::size_t returned_nodes = topology.nodes;
    HIP_CHECK(hipGraphGetNodes(graph, nodes.data(), &returned_nodes));
    if (returned_nodes != topology.nodes) {
        throw std::runtime_error(
            "HIP graph node count changed while inspecting capture");
    }

    for (const hipGraphNode_t node : nodes) {
        hipGraphNodeType type = hipGraphNodeTypeCount;
        HIP_CHECK(hipGraphNodeGetType(node, &type));
        if (type == hipGraphNodeTypeKernel) {
            ++topology.kernel_nodes;
        } else if (type == hipGraphNodeTypeMemcpy) {
            ++topology.memcpy_nodes;
        } else {
            ++topology.other_nodes;
        }
    }

    HIP_CHECK(hipGraphGetEdges(graph, nullptr, nullptr, &topology.edges));
    HIP_CHECK(hipGraphGetRootNodes(graph, nullptr, &topology.roots));
    if (topology.nodes != expected_nodes.size() ||
        topology.edges + 1 != topology.nodes || topology.roots != 1 ||
        topology.other_nodes != 0) {
        throw std::runtime_error(
            "captured HIP graph does not have the expected linear-chain size");
    }

    std::vector<hipGraphNode_t> edge_from(topology.edges);
    std::vector<hipGraphNode_t> edge_to(topology.edges);
    std::size_t returned_edges = topology.edges;
    HIP_CHECK(hipGraphGetEdges(
        graph, edge_from.data(), edge_to.data(), &returned_edges));
    if (returned_edges != topology.edges) {
        throw std::runtime_error(
            "HIP graph edge count changed while inspecting capture");
    }

    const auto node_index = [&](hipGraphNode_t sought) {
        const auto found = std::find(nodes.begin(), nodes.end(), sought);
        if (found == nodes.end()) {
            throw std::runtime_error(
                "HIP graph edge references a node outside the graph");
        }
        return static_cast<std::size_t>(found - nodes.begin());
    };
    std::vector<std::size_t> indegree(topology.nodes, 0);
    std::vector<std::size_t> outdegree(topology.nodes, 0);
    std::vector<hipGraphNode_t> successor(topology.nodes, nullptr);
    for (std::size_t edge = 0; edge < topology.edges; ++edge) {
        const std::size_t from = node_index(edge_from[edge]);
        const std::size_t to = node_index(edge_to[edge]);
        if (from == to || ++outdegree[from] > 1 || ++indegree[to] > 1) {
            throw std::runtime_error(
                "captured HIP graph is not a simple linear dependency chain");
        }
        successor[from] = edge_to[edge];
    }

    hipGraphNode_t current = nullptr;
    std::size_t computed_roots = 0;
    for (std::size_t index = 0; index < topology.nodes; ++index) {
        if (indegree[index] == 0) {
            current = nodes[index];
            ++computed_roots;
        } else if (indegree[index] != 1) {
            throw std::runtime_error(
                "captured HIP graph has an invalid node indegree");
        }
        const std::size_t expected_outdegree =
            outdegree[index] == 0 ? 0 : 1;
        if (outdegree[index] != expected_outdegree) {
            throw std::runtime_error(
                "captured HIP graph has an invalid node outdegree");
        }
    }
    if (computed_roots != 1 || current == nullptr) {
        throw std::runtime_error(
            "captured HIP graph does not have exactly one computed root");
    }

    std::vector<bool> visited(topology.nodes, false);
    for (std::size_t position = 0; position < expected_nodes.size(); ++position) {
        const std::size_t index = node_index(current);
        if (visited[index]) {
            throw std::runtime_error("captured HIP graph contains a cycle");
        }
        visited[index] = true;
        const ExpectedGraphNode& expected = expected_nodes[position];
        hipGraphNodeType observed_type = hipGraphNodeTypeCount;
        HIP_CHECK(hipGraphNodeGetType(current, &observed_type));
        if (observed_type != expected.type) {
            throw std::runtime_error(
                std::string(expected.label) +
                " node has the wrong type or chain position");
        }
        if (observed_type == hipGraphNodeTypeKernel) {
            require_kernel_node(current, expected);
        } else {
            require_memcpy_node(current, expected);
        }

        const bool is_last = position + 1 == expected_nodes.size();
        if ((is_last && outdegree[index] != 0) ||
            (!is_last && outdegree[index] != 1)) {
            throw std::runtime_error(
                "captured HIP graph chain terminates at the wrong node");
        }
        current = successor[index];
    }
    if (current != nullptr ||
        std::find(visited.begin(), visited.end(), false) != visited.end()) {
        throw std::runtime_error(
            "captured HIP graph chain does not visit every node exactly once");
    }
    return topology;
}

class GraphReplay {
  public:
    GraphReplay(hipGraph_t graph,
                GraphTopology topology,
                double capture_ms,
                std::string label)
        : graph_(graph),
          topology_(topology),
          capture_ms_(capture_ms),
          label_(std::move(label)) {}

    ~GraphReplay() { reset(); }

    GraphReplay(const GraphReplay&) = delete;
    GraphReplay& operator=(const GraphReplay&) = delete;

    GraphReplay(GraphReplay&& other) noexcept
        : graph_(std::exchange(other.graph_, nullptr)),
          executable_(std::exchange(other.executable_, nullptr)),
          topology_(other.topology_),
          capture_ms_(other.capture_ms_),
          instantiate_ms_(other.instantiate_ms_),
          label_(std::move(other.label_)) {}

    GraphReplay& operator=(GraphReplay&& other) noexcept {
        if (this != &other) {
            reset();
            graph_ = std::exchange(other.graph_, nullptr);
            executable_ = std::exchange(other.executable_, nullptr);
            topology_ = other.topology_;
            capture_ms_ = other.capture_ms_;
            instantiate_ms_ = other.instantiate_ms_;
            label_ = std::move(other.label_);
        }
        return *this;
    }

    void instantiate() {
        std::array<char, 4096> log{};
        hipGraphNode_t error_node = nullptr;
        const auto begin = std::chrono::steady_clock::now();
        const hipError_t result = hipGraphInstantiate(
            &executable_, graph_, &error_node, log.data(), log.size());
        const auto end = std::chrono::steady_clock::now();
        instantiate_ms_ =
            std::chrono::duration<double, std::milli>(end - begin).count();
        if (result != hipSuccess) {
            std::string detail;
            if (log[0] != '\0') {
                detail = "; instantiate log: " + std::string(log.data());
            }
            throw std::runtime_error(
                label_ + " hipGraphInstantiate failed: " +
                hipGetErrorName(result) + " (" + hipGetErrorString(result) +
                ")" + detail);
        }
        if (executable_ == nullptr) {
            throw std::runtime_error(label_ +
                                     " returned a null executable graph");
        }
    }

    void launch(hipStream_t stream) const {
        HIP_CHECK(hipGraphLaunch(executable_, stream));
    }

    [[nodiscard]] const GraphTopology& topology() const { return topology_; }
    [[nodiscard]] double capture_ms() const { return capture_ms_; }
    [[nodiscard]] double instantiate_ms() const { return instantiate_ms_; }

  private:
    void reset() noexcept {
        if (executable_ != nullptr) {
            (void)hipGraphExecDestroy(executable_);
            executable_ = nullptr;
        }
        if (graph_ != nullptr) {
            (void)hipGraphDestroy(graph_);
            graph_ = nullptr;
        }
    }

    hipGraph_t graph_ = nullptr;
    hipGraphExec_t executable_ = nullptr;
    GraphTopology topology_{};
    double capture_ms_ = 0.0;
    double instantiate_ms_ = 0.0;
    std::string label_;
};

template <typename Enqueue>
GraphReplay capture_replay(std::string label,
                           hipStream_t stream,
                           const std::vector<ExpectedGraphNode>& expected_nodes,
                           Enqueue&& enqueue) {
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipGetLastError());

    const auto begin = std::chrono::steady_clock::now();
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
    try {
        enqueue();
    } catch (...) {
        hipGraph_t abandoned = nullptr;
        (void)hipStreamEndCapture(stream, &abandoned);
        if (abandoned != nullptr) {
            (void)hipGraphDestroy(abandoned);
        }
        throw;
    }

    hipGraph_t graph = nullptr;
    const hipError_t capture_result = hipStreamEndCapture(stream, &graph);
    const auto end = std::chrono::steady_clock::now();
    if (capture_result != hipSuccess) {
        if (graph != nullptr) {
            (void)hipGraphDestroy(graph);
        }
        throw std::runtime_error(
            label + " hipStreamEndCapture failed: " +
            hipGetErrorName(capture_result) + " (" +
            hipGetErrorString(capture_result) + ")");
    }
    if (graph == nullptr) {
        throw std::runtime_error(label + " capture returned a null graph");
    }

    const double capture_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    GraphTopology topology;
    try {
        topology = inspect_graph(graph, expected_nodes);
    } catch (...) {
        (void)hipGraphDestroy(graph);
        throw;
    }
    GraphReplay replay(
        graph, topology, capture_ms, std::move(label));
    replay.instantiate();
    return replay;
}

std::vector<ExpectedGraphNode> expected_resident_nodes(
    DeviceData& device,
    std::uint32_t token_count) {
    constexpr std::uint32_t element_threads = 256;
    const std::uint32_t output_elements = token_count * kHidden;
    return {
        expected_kernel(
            "f32_to_bf16_padded",
            reinterpret_cast<void*>(f32_to_bf16_padded),
            dim3((kPaddedTokens * kHidden + element_threads - 1) /
                 element_threads),
            dim3(element_threads),
            device.input.get(),
            device.padded_input.get(),
            token_count),
        expected_kernel("gate_up_wmma",
                        reinterpret_cast<void*>(gate_up_wmma),
                        dim3(kFfn / kTile, kExperts, kProjections),
                        dim3(kWaveSize),
                        device.padded_input.get(),
                        device.gate_up_weights.get(),
                        device.gate_up_output.get()),
        expected_kernel(
            "silu_multiply_to_bf16",
            reinterpret_cast<void*>(silu_multiply_to_bf16),
            dim3((kExperts * kPaddedTokens * kFfn + element_threads - 1) /
                 element_threads),
            dim3(element_threads),
            device.gate_up_output.get(),
            device.activation.get()),
        expected_kernel("down_wmma",
                        reinterpret_cast<void*>(down_wmma),
                        dim3(kHidden / kTile, kExperts),
                        dim3(kWaveSize),
                        device.activation.get(),
                        device.down_weights.get(),
                        device.expert_output.get()),
        expected_kernel("weighted_route_reduce",
                        reinterpret_cast<void*>(weighted_route_reduce),
                        dim3((output_elements + element_threads - 1) /
                             element_threads),
                        dim3(element_threads),
                        device.expert_output.get(),
                        device.routes.get(),
                        device.output.get(),
                        token_count),
    };
}

std::vector<ExpectedGraphNode> expected_host_visible_nodes(
    HostData& host,
    DeviceData& device,
    std::uint32_t token_count) {
    const std::size_t input_bytes =
        static_cast<std::size_t>(token_count) * kHidden * sizeof(float);
    const std::size_t route_bytes =
        static_cast<std::size_t>(token_count) * kTopK * sizeof(RouteEntry);
    std::vector<ExpectedGraphNode> nodes;
    nodes.reserve(8);
    nodes.push_back(expected_memcpy("input_h2d",
                                    host.input.get(),
                                    device.input.get(),
                                    input_bytes,
                                    hipMemcpyHostToDevice));
    nodes.push_back(expected_memcpy("routes_h2d",
                                    host.routes.get(),
                                    device.routes.get(),
                                    route_bytes,
                                    hipMemcpyHostToDevice));
    std::vector<ExpectedGraphNode> kernels =
        expected_resident_nodes(device, token_count);
    nodes.insert(nodes.end(), kernels.begin(), kernels.end());
    nodes.push_back(expected_memcpy("output_d2h",
                                    device.output.get(),
                                    host.output.get(),
                                    input_bytes,
                                    hipMemcpyDeviceToHost));
    return nodes;
}

struct CapturedCohort {
    std::uint32_t token_count;
    GraphReplay resident;
    GraphReplay host_visible;

    CapturedCohort(std::uint32_t count,
                   GraphReplay resident_graph,
                   GraphReplay host_visible_graph)
        : token_count(count),
          resident(std::move(resident_graph)),
          host_visible(std::move(host_visible_graph)) {}

    CapturedCohort(const CapturedCohort&) = delete;
    CapturedCohort& operator=(const CapturedCohort&) = delete;
    CapturedCohort(CapturedCohort&&) noexcept = default;
    CapturedCohort& operator=(CapturedCohort&&) noexcept = default;
};

void print_graph(std::uint32_t token_count,
                 std::string_view envelope,
                 const GraphReplay& replay) {
    const GraphTopology& topology = replay.topology();
    std::cout << token_count << ',' << envelope << ',' << topology.nodes << ','
              << topology.kernel_nodes << ',' << topology.memcpy_nodes << ','
              << topology.other_nodes << ',' << topology.edges << ','
              << topology.roots << ',' << std::fixed << std::setprecision(6)
              << replay.capture_ms() << ',' << replay.instantiate_ms() << '\n';
}

struct ReplayValidation {
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
    const std::size_t input_bytes =
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
    HIP_CHECK(hipMemsetAsync(device.output.get(), 0xff, input_bytes, stream));
    if (host_visible) {
        std::fill_n(host.output.get(),
                    static_cast<std::size_t>(token_count) * kHidden,
                    std::numeric_limits<float>::quiet_NaN());
        HIP_CHECK(hipMemsetAsync(device.input.get(), 0, input_bytes, stream));
        HIP_CHECK(hipMemsetAsync(device.routes.get(), 0, route_bytes, stream));
    }
}

template <typename Enqueue>
ReplayValidation validate_path(
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
    ReplayValidation result;
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

bool meets_correctness_gate(const ReplayValidation& result) {
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
                      const ReplayValidation& result,
                      bool direct_graph_bit_match) {
    std::cout << token_count << ',' << path << ',' << envelope << ','
              << std::scientific << std::setprecision(6)
              << result.mechanical_error.relative_l2 << ','
              << result.mechanical_error.max_absolute << ','
              << result.f32_policy_error.relative_l2 << ','
              << result.f32_policy_error.max_absolute << ','
              << result.mechanical_error.checksum << ','
              << (result.bit_deterministic ? "true" : "false") << ','
              << (direct_graph_bit_match ? "true" : "false") << '\n';
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
    DistributionStats direct_ms;
    DistributionStats graph_ms;
    DistributionStats direct_over_graph;
};

template <typename DirectEnqueue, typename GraphEnqueue>
PairedTiming benchmark_pair(const Options& options,
                            hipStream_t stream,
                            DirectEnqueue&& enqueue_direct,
                            GraphEnqueue&& enqueue_graph) {
    for (int warmup = 0; warmup < options.warmups; ++warmup) {
        if (warmup % 2 == 0) {
            enqueue_direct();
            enqueue_graph();
        } else {
            enqueue_graph();
            enqueue_direct();
        }
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<double> direct_samples;
    std::vector<double> graph_samples;
    std::vector<double> ratios;
    direct_samples.reserve(options.samples);
    graph_samples.reserve(options.samples);
    ratios.reserve(options.samples);

    for (int sample = 0; sample < options.samples; ++sample) {
        double direct_ms = 0.0;
        double graph_ms = 0.0;
        if (sample % 2 == 0) {
            direct_ms =
                measure_burst(options.burst, stream, enqueue_direct);
            graph_ms = measure_burst(options.burst, stream, enqueue_graph);
        } else {
            graph_ms = measure_burst(options.burst, stream, enqueue_graph);
            direct_ms =
                measure_burst(options.burst, stream, enqueue_direct);
        }
        direct_samples.push_back(direct_ms);
        graph_samples.push_back(graph_ms);
        ratios.push_back(direct_ms / graph_ms);
    }

    return {summarize_distribution(std::move(direct_samples)),
            summarize_distribution(std::move(graph_samples)),
            summarize_distribution(std::move(ratios))};
}

void print_paired_timing(std::uint32_t token_count,
                         std::string_view envelope,
                         const PairedTiming& timing) {
    std::cout << token_count << ',' << envelope << ',' << std::fixed
              << std::setprecision(6) << timing.direct_ms.median << ','
              << timing.direct_ms.p90 << ',' << timing.graph_ms.median << ','
              << timing.graph_ms.p90 << ',' << timing.direct_over_graph.p10
              << ',' << timing.direct_over_graph.median << ','
              << timing.direct_over_graph.p90 << '\n';
}

bool meets_resident_run_gate(const PairedTiming& timing) {
    constexpr double minimum_paired_median_speedup = 1.02;
    constexpr double maximum_graph_over_direct_p90 = 1.05;
    return std::isfinite(timing.direct_over_graph.median) &&
           timing.direct_over_graph.median >=
               minimum_paired_median_speedup &&
           std::isfinite(timing.direct_ms.p90) &&
           std::isfinite(timing.graph_ms.p90) &&
           timing.graph_ms.p90 <=
               maximum_graph_over_direct_p90 * timing.direct_ms.p90;
}

void print_resident_run_gate(std::uint32_t token_count,
                             const PairedTiming& timing,
                             bool passed) {
    std::cout << "qualification_run_gate: N=" << token_count
              << " envelope=resident paired_median_speedup=" << std::fixed
              << std::setprecision(6) << timing.direct_over_graph.median
              << " minimum=1.020000 graph_over_direct_p90="
              << timing.graph_ms.p90 / timing.direct_ms.p90
              << " maximum=1.050000 status=" << (passed ? "pass" : "fail")
              << '\n';
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
                    << "Usage: rocwmma_routed_moe_graph_bench [options]\n"
                    << "  --warmups N  untimed paired operators (default 4; "
                       "minimum 4)\n"
                    << "  --samples N  paired wall-clock samples (default 12; "
                       "minimum 12 and multiple of 4)\n"
                    << "  --burst N    operators per measurement (default 32; "
                       "minimum 32)\n"
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
            options.burst = 32;
        }
        if (!quick_mode && options.warmups < 4) {
            throw std::runtime_error(
                "graph qualification requires --warmups >= 4 outside --quick");
        }
        if (!quick_mode &&
            (options.samples < 12 || options.samples % 4 != 0)) {
            throw std::runtime_error(
                "graph qualification requires --samples >= 12 and a multiple "
                "of 4 outside --quick");
        }
        if (!quick_mode && options.burst < 32) {
            throw std::runtime_error(
                "graph qualification requires --burst >= 32 outside --quick");
        }
        if (!quick_mode && explicit_tokens) {
            throw std::runtime_error(
                "graph qualification requires the complete N={1,2,4,8,16} "
                "cohort; --tokens is smoke-only");
        }
        const std::vector<std::uint32_t> qualification_tokens = {
            1, 2, 4, 8, 16};
        if (!quick_mode && options.token_counts != qualification_tokens) {
            throw std::runtime_error(
                "graph qualification requires exactly N={1,2,4,8,16}");
        }

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
        std::cout << "control: five direct kernel submissions/operator; frozen "
                     "baseline kernels and arithmetic\n";
        std::cout << "candidate: one hipGraphLaunch/operator replaying the same "
                     "five kernel nodes in the same dependency chain\n";
        std::cout << "resident graph: exactly 5 kernel nodes, 4 edges, 1 root; "
                     "no copies captured\n";
        std::cout << "host_visible graph: exactly 5 kernel + 3 pinned memcpy "
                     "nodes, 7 edges, 1 root\n";
        std::cout << "timing: paired chrono wall clock, alternating direct/graph "
                     "order per sample; capture/instantiate excluded; HIP events "
                     "unused; warmups="
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

        std::cout << "persistent_device_MiB=" << mib(persistent_bytes(device))
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

        std::vector<CapturedCohort> cohorts;
        cohorts.reserve(options.token_counts.size());
        for (const std::uint32_t token_count : options.token_counts) {
            const std::vector<ExpectedGraphNode> resident_nodes =
                expected_resident_nodes(device, token_count);
            const std::vector<ExpectedGraphNode> host_visible_nodes =
                expected_host_visible_nodes(host, device, token_count);
            GraphReplay resident = capture_replay(
                "N=" + std::to_string(token_count) + " resident",
                stream.get(),
                resident_nodes,
                [&] { launch_pipeline(device, token_count, stream.get()); });
            GraphReplay host_visible = capture_replay(
                "N=" + std::to_string(token_count) + " host_visible",
                stream.get(),
                host_visible_nodes,
                [&] {
                    enqueue_host_visible_operator(
                        device, host, token_count, stream.get());
                });
            cohorts.emplace_back(token_count,
                                 std::move(resident),
                                 std::move(host_visible));
        }

        std::cout << "N,envelope,nodes,kernel_nodes,memcpy_nodes,other_nodes,"
                     "edges,roots,capture_ms,instantiate_ms\n";
        for (const CapturedCohort& cohort : cohorts) {
            print_graph(cohort.token_count, "resident", cohort.resident);
            print_graph(
                cohort.token_count, "host_visible", cohort.host_visible);
        }

        std::cout << "N,path,envelope,mechanical_relative_l2,mechanical_max_abs,"
                     "f32_policy_relative_l2,f32_policy_max_abs,checksum,"
                     "bit_deterministic,direct_graph_bit_match\n";
        bool valid = true;
        for (const CapturedCohort& cohort : cohorts) {
            const std::uint32_t token_count = cohort.token_count;
            const ReplayValidation direct_resident = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                false,
                [&] { launch_pipeline(device, token_count, stream.get()); });
            const ReplayValidation graph_resident = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                false,
                [&] { cohort.resident.launch(stream.get()); });
            const bool resident_match =
                exact_match(direct_resident.output, graph_resident.output);

            const ReplayValidation direct_host_visible = validate_path(
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
            const ReplayValidation graph_host_visible = validate_path(
                host,
                device,
                mechanical_reference,
                f32_policy_reference,
                token_count,
                stream.get(),
                true,
                [&] { cohort.host_visible.launch(stream.get()); });
            const bool host_visible_match = exact_match(
                direct_host_visible.output, graph_host_visible.output);

            valid = valid && meets_correctness_gate(direct_resident) &&
                    meets_correctness_gate(graph_resident) && resident_match &&
                    meets_correctness_gate(direct_host_visible) &&
                    meets_correctness_gate(graph_host_visible) &&
                    host_visible_match;

            print_validation(token_count,
                             "direct",
                             "resident",
                             direct_resident,
                             true);
            print_validation(token_count,
                             "graph",
                             "resident",
                             graph_resident,
                             resident_match);
            print_validation(token_count,
                             "direct",
                             "host_visible",
                             direct_host_visible,
                             true);
            print_validation(token_count,
                             "graph",
                             "host_visible",
                             graph_host_visible,
                             host_visible_match);
        }

        if (!valid) {
            std::cerr
                << "validation failed: direct and graph paths require finite "
                   "output, mechanical relative L2 <= 5e-4, max abs <= 1e-3, "
                   "repeated bit determinism, and exact direct/graph agreement "
                   "for resident and host-visible envelopes\n";
            return 2;
        }

        std::cout << "N,envelope,direct_median_ms,direct_p90_ms,graph_median_ms,"
                     "graph_p90_ms,paired_direct_over_graph_p10,"
                     "paired_direct_over_graph_median,"
                     "paired_direct_over_graph_p90\n";
        bool resident_performance_valid = true;
        std::size_t resident_gate_count = 0;
        std::size_t performance_index = 0;
        for (const CapturedCohort& cohort : cohorts) {
            const std::uint32_t token_count = cohort.token_count;
            if (token_count != 1 && token_count != 16) {
                continue;
            }

            PairedTiming resident{};
            PairedTiming host_visible{};
            auto time_resident = [&] {
                resident = benchmark_pair(
                    options,
                    stream.get(),
                    [&] { launch_pipeline(device, token_count, stream.get()); },
                    [&] { cohort.resident.launch(stream.get()); });
            };
            auto time_host_visible = [&] {
                host_visible = benchmark_pair(
                    options,
                    stream.get(),
                    [&] {
                        enqueue_host_visible_operator(
                            device, host, token_count, stream.get());
                    },
                    [&] { cohort.host_visible.launch(stream.get()); });
            };
            if (performance_index % 2 == 0) {
                time_resident();
                time_host_visible();
            } else {
                time_host_visible();
                time_resident();
            }

            print_paired_timing(token_count, "resident", resident);
            print_paired_timing(token_count, "host_visible", host_visible);
            if (!quick_mode) {
                const bool passed = meets_resident_run_gate(resident);
                print_resident_run_gate(token_count, resident, passed);
                resident_performance_valid =
                    resident_performance_valid && passed;
                ++resident_gate_count;
            }
            ++performance_index;
        }

        if (!quick_mode &&
            (!resident_performance_valid || resident_gate_count != 2)) {
            std::cerr
                << "qualification run failed: resident N=1 and N=16 each "
                   "require paired median speedup >= 1.02 and graph p90 <= "
                   "1.05x direct p90; host-visible timing is a separately "
                   "reported secondary decision\n";
            return 3;
        }
        if (!quick_mode) {
            std::cout
                << "qualification_run=pass aggregate_gate_pending="
                   "three_guarded_runs resident_median_of_three_N1>=1.08 "
                   "resident_median_of_three_N16>=1.05; host_visible_optional="
                   "median_of_three_each_N>=1.03_and_no_p90_regression\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
