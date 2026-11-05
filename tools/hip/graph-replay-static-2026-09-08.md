# HIP Graph replay candidate — static qualification

Status: statically and numerically qualified, then rejected for performance.
See `graph-replay-results-2026-09-08.md`.

## Isolation and arithmetic contract

The experiment lives in `rocwmma_routed_moe_graph_bench.hip.cpp`. It includes
the validated standalone baseline after renaming only its `main` token. It does
not define a candidate kernel or alter an operation, launch geometry, argument,
buffer, dependency, or numeric boundary.

The direct control calls the frozen `launch_pipeline`, which submits these five
kernels:

1. F32 input to padded BF16.
2. Gate/up BF16 WMMA with FP32 accumulation.
3. FP32 SiLU/multiply to BF16 activation.
4. Down BF16 WMMA with FP32 accumulation.
5. FP32 weighted route reduction.

The resident candidate captures that exact function on the benchmark's
non-blocking stream, instantiates it once per `N`, and submits one
`hipGraphLaunch` per operator. The copy-inclusive candidate captures the same
three pinned transfers as the direct host-visible envelope: input H2D, route
metadata H2D, and output D2H around the five kernels.

Capture and instantiation happen before validation and timing. The benchmark
requires these exact single-stream chains at runtime:

```text
resident:     5 kernel nodes + 0 memcpy nodes, 4 edges, 1 root
host_visible: 5 kernel nodes + 3 memcpy nodes, 7 edges, 1 root
```

Any other node type or topology is a hard failure. Captured host pointers are
pinned and remain alive until both executable graphs are destroyed. Graphs are
destroyed before their stream and all referenced host/device buffers.
Inspection reconstructs the chain from its edge endpoints and rejects branches,
cycles, disconnected nodes, or early termination. In chain order it also checks
every kernel function, grid, block, and dynamic-shared-memory size, plus every
copy's source/destination address, direction, and byte extent.

## Correctness gate

For every `N={1,2,4,8,16}`, independently in resident and host-visible
envelopes, qualification requires:

- finite output;
- mechanical relative L2 no greater than `5e-4`;
- mechanical max absolute error no greater than `1e-3`;
- finite current-F32-semantics policy-drift metrics;
- bit-identical repeated execution for both direct and graph paths; and
- bit-identical graph output versus the direct frozen control.

Before every direct and graph validation, the harness independently poisons all
five graph-produced regions: padded input, gate/up output, activation, expert
output, and active final output. Host-visible validation additionally poisons
the active device input and routes with zeros and the pinned host output with
NaNs. Each kernel and captured H2D/D2H node must therefore repair its own state
on every replay; correct data left by a preceding direct run cannot mask a bad
argument, omitted write, or truncated copy. Captured kernel argument bytes are
also checked against every expected buffer pointer and `token_count` value.

Because replay uses the exact kernel nodes rather than reimplemented kernels,
the expected policy drift is identical to the existing native-BF16 experiment.
The direct/graph bit comparison is nevertheless mandatory; this catches bad
capture arguments, dependencies, copy sizes, or replay behavior.

## Static build contract

Build command:

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_routed_moe_graph.sh \
  /tmp/beholdd-rocwmma-routed-moe-graph-bench
```

The wrapper pins and hashes the ROCm 7.13 wheel's AMD Clang, HIP runtime/API
header, and OCML bitcode. It also requires the clean public rocWMMA header tree
at commit `bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125`, with header-tree SHA-256
`951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750`.
The included baseline must retain SHA-256
`8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80`.

Static qualification rejects mixed system ROCm, a missing HIP Graph symbol,
the wrong code-object machine, any additional/missing GPU kernel, scratch,
spill, LDS, dynamic stack, non-wave32 metadata, a changed workgroup envelope,
more than 64 VGPRs in either WMMA kernel, or a changed native-WMMA inventory.
The expected device payload is exactly the five frozen kernels and four static
`v_wmma_f32_16x16x16_bf16` instructions: two in gate/up and two in down.

The first successful static build reported:

```text
graph_harness_source_sha256=2f7b092844c14d500d597dfb292a6d12eb194388d280ab97999e90b8273adb89
compiler_sha256=1b2f3f4a10e946b3d2ac133c8f0ba4406b20b58679a3cd6d9054465849d18db0
hip_runtime_sha256=eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05
hip_runtime_api_header_sha256=0c2270e8f18150798e969daec4838ef84d0bc761394ecc933268e8eda149b37e
ocml_sha256=a9d523fa792412ab9980ee76fb43214dc27983a07f0b23354424e9b04e91682f
hip_headers_tree_sha256=ad0a71b702c2dd044dfd7fb3f2e8e8418b8e367682bbb7188816940cb1741043
clang_resource_headers_tree_sha256=413604f0742dee20a585685ad7d52f88869f2434c0914244501619fed532fcb8
device_libraries_tree_sha256=e8b3b2834ddf0982dd1148f532ee0e6ef956d9759d817f4e9e5bc4f61fba9a40
hip_graph_api_exports=12
device_text_sha256=13227a7fab4ce921e01b92a3c7e0afff66a304d290e05091c3524c7fd42c27e6
device_kernel_count=5
native_wmma_instruction_count=4 (gate=2 down=2)
gate_up_wmma: vgpr=45 private=0 lds=0 spills=0 wave32 max_workgroup=32
down_wmma:    vgpr=48 private=0 lds=0 spills=0 wave32 max_workgroup=32
```

The full bundled code-object hash varies across repeat builds because the
disposable HIP-root path contributes non-executable provenance. Two same-source
repeat builds produced different container hashes but the same `.text` hash
shown above. The wrapper therefore pins that deterministic device instruction
payload, every relevant include/resource/device-library tree, and semantic
kernel metadata/ISA properties. It prints the full code-object hash only as a
per-build artifact identifier. It also hashes the reviewed harness both before
and after compilation to reject source drift during a build.

## Predeclared runtime protocol and performance gate

Every command that initializes or executes on the GPU must remain inside the
thermal guard:

```bash
cd "$BEHOLDD_SOURCE_DIR"
python3 tools/thermal_guard.py \
  --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- \
  env LD_LIBRARY_PATH=$ROCM_CORE/lib \
  /tmp/beholdd-rocwmma-routed-moe-graph-bench
```

Use `--quick` only for the first guarded smoke test. It prints
`mode=smoke qualification_grade=false` and rejects explicit warmup, sample, or
burst overrides so a smoke log cannot resemble a qualification run. A
qualification run refuses token subsets and requires all cohorts, at least four
warmups, at least 12 paired samples in a multiple of four, and a burst of at
least 32. Warmups absorb any first-replay materialization. Each sample uses
wall-clock time around a burst plus one final stream synchronization; HIP events
remain excluded. Direct and graph order alternates within every pair. Resident
and host-visible envelope order alternates between `N=1` and `N=16`.

The frozen direct control calls `hipGetLastError` after each of its five kernel
submissions, while the candidate checks one `hipGraphLaunch`. This deliberately
matches the already-measured baseline but is conservative in graph replay's
favor compared with a production direct path that batch-checks launches once.
Any reported speedup must retain that qualification when used to forecast
production gains; the decision is whether graph replay beats this exact frozen
control, not a claim about the theoretical minimum direct-launch overhead.

The primary backend decision is resident replay. A single qualification run is
accepted only when both `N=1` and `N=16` have:

- paired median `direct / graph >= 1.02`; and
- graph p90 no greater than `1.05 * direct p90`.

Run the full qualification three times under the guard. Promote resident graph
replay for backend prototyping only if all correctness checks pass in all three
runs, no individual resident run misses the single-run gate, and the
median-of-three paired-median speedup is at least `1.08x` for `N=1` and `1.05x`
for `N=16`.

Host-visible graph replay is a separate optional decision: require exact
correctness plus median-of-three speedup of at least `1.03x` at both `N=1` and
`N=16`, with no graph-p90 regression. A host-visible miss does not invalidate
resident replay because the runtime can keep copies outside the resident graph.

These are routed-MoE operator timings in an ideal high-overlap cohort. They are
not full-model tokens per second, and graph capture/instantiation latency is
reported separately rather than hidden in replay latency.

The guarded smoke and three full runs completed this protocol. Every topology,
poison, bit-parity, determinism, and oracle check passed, but every resident run
missed its `1.02x` floor. Median-of-three resident ratios were `0.977392x` at
`N=1` and `0.984913x` at `N=16`; graph replay is therefore a recorded no-go,
not a backend option.

## References

- [HIP graphs](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/hipgraph.html)
- [HIP Graph API tutorial](https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/graph_api.html)
