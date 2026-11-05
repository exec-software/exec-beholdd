# OpenMythos numerical and serving checkpoint — 2026-10-23

The pinned OpenMythos checkpoint passed a complete native/CPU output comparison
and three bounded Beholdd serving checks on alternate-host. These are diagnostic results;
the OpenMythos HAG profile is **not deployed or qualified for normal Antimatter
requests**. Existing Vulkan rollback and the separately qualified small HAG
profile remain unchanged. Background startup must continue to leave models
unloaded until an actual chat request.

## Fixed numerical comparison

The tested Beholdd sources were `da7956b`, using the separate `hag-openmythos`
feature, on Linux `7.0.0-38-generic` and AMD device `1002:1586`. Checkpoint identity
and architecture are recorded in [hag-openmythos.md](hag-openmythos.md). The full
16,547,399,872-byte file hash was verified inside the guarded CPU scope.

The independent F32 CPU graph uses pinned llama.cpp
`4da6337767f973e2b4d0797e5b323d77d8565e4a`, with row dequantization and ordered F32
products/additions instead of activation quantization. A separate vocabulary-only
proof fixes `Hello` to token `9419`, without added BOS/EOS or special tokens.
Neither input nor tolerance was selected from the native output.

| Gate | Compared values | Maximum absolute error | Relative L2 | Greedy token |
| --- | ---: | ---: | ---: | ---: |
| Large Q4_K/Q6_K tiles | 44,260 | 0 | Exact comparison | — |
| Tiny hybrid, token 7 | 512 | 3.57628e-7 | 2.93364e-7 | 213 |
| Tiny hybrid, token 42 | 512 | 5.36442e-7 | 3.56383e-7 | 339 |
| Actual OpenMythos, token 9419 | 248,320 | 3.67165e-5 | 1.84513e-6 | 11 |

All complete-vector comparisons had zero values outside
`2e-4 + 2e-4 * abs(reference)` and relative L2 below `2e-5`; native and CPU greedy
tokens agreed. The actual CPU graph ran twice with exact reset/replay. The native
probe ran three graphs and passed cancellation recovery, stale/cancelled ticket
refusals, invalid-token/capacity refusals, accounting and reset/replay checks.
Its allocation retired to zero. This establishes the fixed first-position case,
not arbitrary sequences, long-context numerical accuracy or semantic quality.

## Bounded C2048 serving checks

Thinking-on, thinking-off and cancel/retry all passed through a dedicated
loopback Beholdd process. Health, capabilities and catalog discovery left the
model unloaded with no model/device handles. The first chat loaded the model;
streaming included usage and terminal/DONE delivery. Each case unloaded twice,
returned to cold state and exited cleanly.

The cancellation case refused unload while busy, disconnected after the first
fenced prefill position, completed a healthy reset, and reproduced the reference
completion on a retry in the same process. The fixed requests were `Hello`,
greedy, with explicit thinking flags and four output tokens; the cancelled
request allowed eight. Thinking-on used 11 prompt tokens and 14 evaluated
positions; thinking-off used 13 and 16. No editor tools were invoked.

All scopes retained 19 GiB early-stop / 20 GiB hard containment for the actual
model, zero swap, CPU quota/affinity limits, thermal guards and the finite
authenticated crash watcher. CPU-reference peak cgroup charge was
16,190,451,712 bytes. Across the three serving cases, peak cgroup charge was
863,653,888 bytes, maximum CPU/GPU temperatures were 60.75/55°C, and available
memory stayed above 24,623,353,856 bytes. Cgroup charge alone is not total GPU/UMA
usage: admission additionally accounted for the full source file and native
allocation; the C2048 forecast was 18,289,770,166 bytes including 639,152,128
native bytes. There were no memory-pressure/OOM events, swap, or crash captures.
All owned services, cgroups, observed processes and the watcher retired.

## Performance blocks editor deployment

The thinking-on request measured 34.217 seconds for 11 prefill positions and
9.398 seconds for three decode positions. The current token-major implementation
streams the packed weights and computes the full vocabulary head for every
prompt position, even when only the last position's logits are consumed.

Previously measured public Antimatter requests contain 6,210 or 8,279 prompt
tokens. Applying the short-run rate would imply hours of prefill; this is an
extrapolation, not a measured long-prompt result. Increasing context capacity or
memory allowances does not change that execution schedule. The next work is a
checked state-advance operation that omits unused vocabulary work, followed by
bounded tile reuse across prompt positions and fresh numerical/performance
measurement. The first optimization alone does not establish editor readiness.

## Retained evidence

Artifacts are under `work/openmythos-20261022` in the HAG development workspace:

- Actual comparison: `openmythos-v1/runs/compare-1792783937876090204/result.json`,
  SHA-256 `bc0720c46cd353b4bd71c02516774436bda6283bcf7305c18f281dde8a28124e`.
- Serving results: `runtime-thinking-on-v1/runs/native-1792783975399868205`,
  `runtime-thinking-off-v1/runs/native-1792784079825684819`, and
  `runtime-cancel-retry-v1/runs/native-1792784241669443220`.
- `window-20261023-1921-closeout.json` binds those result hashes and records
  complete retirement. `tiny7-v1/watcher/retired.json` closes the shared watcher.
- Independent receipt/source review:
  `work/openmythos-20261023/actual-wholegraph-receipt-review-v1.json`.

Frozen manifests identify the original sources, binaries, token provenance,
checkpoint hash proof, CPU/native output hashes, resource samples, kernel review
and exact comparison policy. Earlier thermal startup refusals remain preserved
as failed attempts; none is relabeled as a passed workload.
