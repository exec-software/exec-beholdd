# HIP Graph replay qualification results — 2026-09-08

Status: **no-go**. HIP Graph replay preserved the frozen five-kernel result
exactly, but it did not clear either the resident or copy-inclusive performance
gate.

## Scope and protocol

This experiment changed submission only. The direct control and graph replay
executed the same five frozen native-BF16 kernels, with the same arguments,
launch geometry, buffers, dependencies, and arithmetic boundaries. The
resident graph contained exactly five kernel nodes, four edges, and one root.
The host-visible graph contained the same five kernel nodes plus three pinned
copies, seven edges, and one root.

Each of three full runs was executed under the thermal guard. A run used four
warmups, twelve counterbalanced paired samples, and a burst of 32 routed-MoE
operators per sample. The reported speedup is the median of the paired
`direct / graph` ratios, so a value below `1.0x` means graph replay was slower.
Timing covered replay plus a final stream synchronization; graph capture and
instantiation remained outside steady-state timing.

The frozen direct control performs five `hipGetLastError` polls per operator,
one after each kernel submission. Graph replay performs one checked graph
launch. This asymmetry favors the graph candidate relative to a future direct
path that batches its launch-error check.

## Correctness and replay integrity

All `N in {1,2,4,8,16}` passed in both resident and host-visible envelopes:

- direct and graph output bits matched exactly;
- repeated output bits were deterministic;
- every output was finite;
- graph topology matched the declared linear chains exactly; and
- poison validation showed that every replay repaired all kernel-produced
  regions and, for the host-visible graph, its input, route, and output copy
  regions.

Against the CPU oracle that models both BF16 activation boundaries, relative
L2 error ranged from `1.097036e-4` to `1.699449e-4`; maximum absolute error was
at most `3.084715e-7`. Against current Beholdd F32-activation semantics, policy
drift ranged from `0.002652090` to `0.002799975` relative L2 and reached at most
`5.584268e-6` absolute. These ranges match the frozen baseline, as required for
a submission-only experiment.

## Performance

The predeclared resident single-run gate required at least `1.02x` at both
`N=1` and `N=16`. Promotion additionally required every run to pass and a
median-of-three of at least `1.08x` at `N=1` and `1.05x` at `N=16`. The
host-visible gate required a median-of-three of at least `1.03x` at both batch
sizes. The candidate missed every median threshold.

| run | `N=1` resident | `N=1` host-visible | `N=16` resident | `N=16` host-visible |
|---:|---:|---:|---:|---:|
| 1 | 0.977392x | 0.974878x | 0.990509x | 1.003240x |
| 2 | 0.979121x | 0.976538x | 0.984913x | 1.002127x |
| 3 | 0.972751x | 0.973575x | 0.981748x | 1.008143x |
| median of three | **0.977392x** | **0.974878x** | **0.984913x** | **1.003240x** |
| required aggregate | 1.08x | 1.03x | 1.05x | 1.03x |
| decision | no-go | no-go | no-go | no-go |

The median-of-three ratios correspond to about a 2.3% resident replay loss at
`N=1` and a 1.5% loss at `N=16`. Copy-inclusive replay lost about 2.6% at
`N=1`; its approximately 0.3% `N=16` gain was far below the `1.03x` gate.
These percentages are interpretations of the paired-ratio medians, not
full-model token-throughput claims.

The guarded smoke test peaked at Tctl `62.125 C`. The three qualification runs
peaked at `63.0 C`, `63.25 C`, and `64.0 C`; none approached the 75-degree
warning or 90-degree abort threshold.

## Frozen artifacts

The successful static qualification recorded:

```text
graph_harness_source_sha256=2f7b092844c14d500d597dfb292a6d12eb194388d280ab97999e90b8273adb89
graph_build_script_sha256=16c7fff40c67b1ec71c454a0e135b43d10a2ea996975a50a403e32a5fd3338f2
baseline_source_sha256=8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80
device_text_sha256=13227a7fab4ce921e01b92a3c7e0afff66a304d290e05091c3524c7fd42c27e6
compiler_sha256=1b2f3f4a10e946b3d2ac133c8f0ba4406b20b58679a3cd6d9054465849d18db0
hip_runtime_sha256=eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05
hip_runtime_api_header_sha256=0c2270e8f18150798e969daec4838ef84d0bc761394ecc933268e8eda149b37e
ocml_sha256=a9d523fa792412ab9980ee76fb43214dc27983a07f0b23354424e9b04e91682f
hip_headers_tree_sha256=ad0a71b702c2dd044dfd7fb3f2e8e8418b8e367682bbb7188816940cb1741043
clang_resource_headers_tree_sha256=413604f0742dee20a585685ad7d52f88869f2434c0914244501619fed532fcb8
device_libraries_tree_sha256=e8b3b2834ddf0982dd1148f532ee0e6ef956d9759d817f4e9e5bc4f61fba9a40
rocwmma_headers_tree_sha256=951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750
```

The code object retained exactly five kernels and four native
`v_wmma_f32_16x16x16_bf16` instructions. See
`graph-replay-static-2026-09-08.md` for the complete static contract and
metadata.

## Decision

Do not add HIP Graph replay as a Beholdd backend option for this operator. It
does not amortize submission overhead even against a direct control that polls
for launch errors five times per operator. Keep the result as an explicit
negative experiment. A subsequent kernel experiment should target material
device work or intermediate traffic—such as a separately gated down-WMMA and
route-reduction fusion—rather than another submission-only optimization.
