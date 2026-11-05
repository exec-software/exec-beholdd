# gfx1151 BF16 rocWMMA routed-MoE experiment

This directory contains a standalone qualification harness. It does not link
to or modify the Beholdd runtime or any running inference service.

## Explicit local configuration

Run commands from this checkout and provide your own SDK locations. The wrappers
do not search a developer's home directory or select a system SDK implicitly:

```bash
export BEHOLDD_SOURCE_DIR="$PWD"
export ROCM_CORE=/path/to/rocm-sdk-core
export ROCWMMA_INCLUDE=/path/to/rocwmma/library/include
```

Replace both example paths with actual directories. The recorded wrappers require
the documented ROCm 7.13 layout, including Clang 23, HIP headers, device bitcode
and `libamdhip64.so.7`. Several wrappers additionally enforce exact source and
toolchain digests; a different SDK requires its own qualification, not bypassing
those checks. `ROCWMMA_INCLUDE` must contain `rocwmma/rocwmma.hpp`.

Historical result documents normalize host paths and deployment labels for public
use; the [research index](../../docs/research/README.md) explains their scope.
The resident-layer source received a comment-only publication cleanup. Its current
SHA-256 is `dfed8f57bc310e1c8f366a13601ab9dccdd88da82744d5980e8370e858d1f3dd`;
the affected build-wrapper source checks now use that digest. Historical live
results retain their original source/binary digests and are not new live evidence
for a rebuilt artifact. No arithmetic or GPU instruction change was made here.

## Real-weight Qwen3.6 result

The current best batch-1 real-weight path is
`rocwmma_qwen35_resident_layer_prepared_parallel_route_bench.hip.cpp`. It keeps
the checkpoint's expert values in BF16, performs a one-time bit-preserving GPU
transpose into rocWMMA's efficient row-major Matrix-B layout, and uses a
deterministic parallel stable-softmax/top-8 router. Its three-run median of
medians is `0.378158 ms`, or 2,644.40 fixed-route MoE-FFN sublayers/s. The
40-block MoE-FFN-only ceiling is 66.11 tokens/s; this is explicitly not a
full-model token rate.

Build the pinned code object without executing it:

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_qwen35_resident_layer_prepared_parallel_route.sh \
  /tmp/beholdd-q35-prepared-parallel-route-bench
```

Run only through the pack-digest and thermal wrapper:

```bash
tools/hip/run_qwen35_layer_guarded.sh \
  /tmp/beholdd-q35-prepared-parallel-route-bench \
  /tmp/qwen36-layer3-bf16.pack \
  9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
  --warmups 4 --samples 12 --burst 16
```

The full provenance, correctness gates, negative results, and qualification
limits are in
[`qwen35-resident-layer-results-2026-09-08.md`](qwen35-resident-layer-results-2026-09-08.md).

## Route-aware independent-request cohort

The current research decision is to promote request coalescing to traced-
activation replay as the next throughput study while preserving the qualified
batch-1 path and golden runtime unchanged. Production integration remains a
NO-GO.
`rocwmma_qwen35_resident_layer_route_cohort_bench.hip.cpp` is the real-weight
`B=16` proof harness. It independently computes each request's RMSNorm,
router, stable top-8, and shared scale; groups the resulting 128
`(request, route-rank)` pairs by expert; fills useful rows in one 16-row WMMA
tile per active expert; and scatters each result back for a rank-ordered
per-request reduction. Its control is 16 sequential invocations of the exact
qualified prepared-layout and parallel-router path.

Build and statically qualify it without executing the GPU:

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_qwen35_resident_layer_route_cohort.sh \
  /tmp/beholdd-q35-route-cohort-host-corpus-static
```

Run a short smoke only through the digest-bound thermal wrapper:

```bash
cd "$BEHOLDD_SOURCE_DIR"
tools/hip/run_qwen35_layer_guarded.sh \
  /tmp/beholdd-q35-route-cohort-host-corpus-static \
  /tmp/qwen36-layer3-bf16.pack \
  9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
  --quick
```

After the smoke passes, use the same containment for a deeper observation:

```bash
cd "$BEHOLDD_SOURCE_DIR"
tools/hip/run_qwen35_layer_guarded.sh \
  /tmp/beholdd-q35-route-cohort-host-corpus-static \
  /tmp/qwen36-layer3-bf16.pack \
  9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
  --warmups 4 --samples 12 --burst 8
```

Exercise the deterministic hashed-independent corpus by setting `N` to the
explicit corpus seed. The recorded three-seed observation ran this guarded
command for `N=1`, `N=2`, and `N=3`:

```bash
cd "$BEHOLDD_SOURCE_DIR"
N=1
tools/hip/run_qwen35_layer_guarded.sh \
  /tmp/beholdd-q35-route-cohort-host-corpus-static \
  /tmp/qwen36-layer3-bf16.pack \
  9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
  --hashed-independent-corpus --corpus-seed "$N" \
  --warmups 4 --samples 12 --burst 8
```

The median of the three independent-corpus seed medians selected 99 active
routed experts and had a `1.44x` theoretical matrix-tile reduction. Sixteen
sequential controls took `6.321079 ms`; the route-aware cohort took
`4.662903 ms`, with a `1.360078x` paired speedup. The candidate rate was
`3,431.34` independent-request MoE-FFN sublayers/s, or `0.291431 ms` per
request when amortized across the cohort. Maximum observed Tctl was `61.5 C`.
The earlier correlated high-overlap corpus reached `1.892540x`; its greater
route reuse makes that an optimistic special case, not the independent-corpus
expectation.

The frozen host-contract source is
`7169e50ef46b4995a7fd58e7312c1db1d27c08bdda624cf0f6e0114544b988be`;
the build wrapper is
`b338e07241f502a2890c2a09497959a84cddbfe181cfb207952c77f6869ea360`.
The retained artifact
`/tmp/beholdd-q35-route-cohort-host-corpus-static` hashes to
`9de1b8f544f2d48f76441f8d11943b0eb044440972d9fc1269fb87e5a5937357`;
its embedded code object hashes to
`6a7d562b94e78f5cf87951c2c50092d20f3d05b6c0a95a4cccb52169dccd6d82`.
Static qualification found exactly 21 `q35` kernels in an exact `gfx1151`
code object, all wave32 with no private memory, spills, dynamic stack, or
scratch ISA traffic. It contains 16 native BF16 WMMA instructions. The device
text remains frozen at
`1d55ff0ff68f2326db7b3109afcc480fda7c77b017b2c2f56cae3e47b52972fa`,
including after the untimed host diagnostic, contract reporting, and
deterministic corpus options were added.

The cohort has a deliberately explicit numerical boundary. Routes, prepared
weight bytes, planner mappings, gathered BF16 rows, and repeat execution must
remain exact. A strict initial smoke showed that moving an otherwise identical
dot product from the qualified control's row zero to a nonzero gfx1151 WMMA
row can change a small number of FP32 result bits. The untimed stage report
therefore preserves the bit-exact result separately, requires row-zero stage
parity and finite results across all rows, and gates candidate/control output
at relative L2 `5e-6` and maximum absolute error `2e-5`. This is a prospective
numerical-equivalence contract, not permission to relabel the path bit-exact.

Reported cohort throughput is the rate of independent request MoE-FFN
sublayers. Its amortized per-request time is neither single-request latency nor
autoregressive or full-model token throughput. The deterministic synthetic
hidden vectors—including the hashed-independent option—exercise a genuine
mixed expert union, but they are not traced production activations. The
decision is therefore research PROMOTE to traced-activation replay and
production NO-GO. No result here changes the golden runtime; promotion requires
the guarded contract, real activation cohorts, and end-to-end model checks to
pass.

## Synthetic fixed-route operator baseline

The experiment exercises a route-shaped `H=2048`, `F=512`, `K=8` MoE operator
at `N=1,2,4,8,16`:

1. F32 token input to a BF16 matrix padded to 16 rows.
2. BF16 gate and up projections with FP32 rocWMMA accumulation/output.
3. SiLU(gate) times up, rounded to BF16 for the next WMMA input.
4. BF16 down projection with FP32 rocWMMA accumulation/output.
5. FP32 weighted route reduction.

This is deliberately the ideal high-overlap cohort: every token visits each of
the same eight resident experts exactly once; only route order and weight vary.
It is an ideal-overlap candidate measurement for a scheduler-filled 16-row WMMA
tile. It does not characterize arbitrary sparse route mixtures.

The gate/up `H x F` and down `F x H` matrices use rocWMMA's row-major Matrix-B
layout. They are persistent BF16, but this is a one-time transposed/prepared
representation of ordinary output-major model tensors, not checkpoint-native
layout. Weight transformation, allocation, and upload are excluded from both
timing envelopes. Global fragment loads and rocWMMA register packing occur in
the kernels and are included.

## Header and compiler provenance

The validated headers are the public rocWMMA project in the ROCm monorepo at:

```text
repository: https://github.com/ROCm/rocm-libraries.git
commit:     bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125
include:    projects/rocwmma/library/include
```

The local checkout used for the recorded run was:

```text
/tmp/rocwmma-source-audit/projects/rocwmma/library/include
```

Set `ROCWMMA_INCLUDE` to an equivalent checkout and `ROCM_CORE` to the audited
ROCm 7.13 SDK core. Do not mix compiler, runtime and headers from different SDK
installations. A wrapper that discovers system headers through `-idirafter` can
silently mix versions; these scripts select their compiler and headers explicitly.

## Build

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_routed_moe.sh \
  /tmp/beholdd-rocwmma-routed-moe-bench
```

The build script invokes the wheel's Clang directly with `-nogpuinc`, the
wheel's Clang resource root, HIP headers, and AMDGPU bitcode paths explicitly.
Because the wheel has `libamdhip64.so.7` but no unversioned linker name, it also
creates a temporary HIP-root overlay containing only a symlink to that exact
library. The overlay is deleted after linking; neither the venv nor the system
ROCm installation is changed.

After linking, the script prints the compiler and rocWMMA checkout versions and
fails unless the embedded code object is tagged
`EF_AMDGPU_MACH_AMDGCN_GFX1151`, disassembles to
`v_wmma_f32_16x16x16_bf16`, and resolves `libamdhip64.so.7` from the selected
ROCm wheel. It also prints that exact library's SHA-256 hash.

## Run safely

Every execution that reaches the GPU must use Beholdd's thermal guard:

```bash
cd "$BEHOLDD_SOURCE_DIR"
python3 tools/thermal_guard.py \
  --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- \
  env LD_LIBRARY_PATH=$ROCM_CORE/lib \
  /tmp/beholdd-rocwmma-routed-moe-bench
```

Use `--quick` for a short smoke test. `--tokens N` can be repeated to select a
subset. `--warmups`, `--samples`, and `--burst` control measurement depth. Each
sample uses `std::chrono::steady_clock`, queues a burst, performs one final
stream synchronization, and divides wall time by the burst size. HIP events are
intentionally not used because event timing is unreliable on this stack. The
resident/host-visible execution order alternates by `N` to reduce systematic
APU DVFS and thermal-order bias; the CSV remains in a stable display order.

The two printed timing envelopes are distinct:

- `resident`: five kernel launches per operator and one final sync per sample;
  input, route metadata, weights, and output remain device resident.
- `host_visible`: pinned H2D F32 input, pinned H2D route metadata, the same five
  kernels, pinned D2H output, and one final sync per sample.

Throughput is labeled `routed_evals_per_second`. It is the rate of this one MoE
operator, not full-model tokens per second. Correctness requires finite output,
bit-identical repeated GPU execution, relative L2 error no greater than `5e-4`,
and max absolute error no greater than `1e-3` against a CPU oracle that mirrors
both BF16 WMMA input boundaries while retaining FP32 accumulation. The resident
and copy-inclusive host-visible outputs are validated independently. A second
CPU reference preserves current Beholdd precision semantics—BF16 weights with
the original F32 input and F32 post-SiLU activation—and the CSV reports the
candidate's relative-L2 and max-absolute policy drift against it. Policy drift
is reported separately from the mechanical correctness gate.

## Recorded variants and decisions

- `rocwmma_qwen35_resident_layer_bench.hip.cpp` is the real layer-3,
  checkpoint-layout control.
- `rocwmma_qwen35_resident_layer_prepared_bench.hip.cpp` proves that the
  bit-preserving prepared layout is the major batch-1 win.
- `rocwmma_qwen35_resident_layer_parallel_route_bench.hip.cpp` isolates the
  deterministic parallel router.
- `rocwmma_qwen35_resident_layer_prepared_parallel_route_bench.hip.cpp`
  combines both positive results and is the current qualified research path.
- `rocwmma_qwen35_resident_layer_route_cohort_bench.hip.cpp` and
  `build_rocwmma_qwen35_resident_layer_route_cohort.sh` are the current
  real-weight, route-aware `B=16` independent-request research candidate and
  its fail-closed static qualifier. They do not replace the qualified batch-1
  path.
- `rocwmma_qwen35_resident_layer_vector_bench.hip.cpp` is correct but slower
  than prepared WMMA and is a performance NO-GO.

- `rocwmma_routed_moe_bench.hip.cpp` is the frozen five-kernel native-BF16
  control. It qualified as a backend research prototype for `N >= 2`, not as a
  production model path.
- `rocwmma_routed_moe_fused_bench.hip.cpp` fuses gate/up/SiLU and is a runtime
  performance NO-GO.
- `rocwmma_routed_moe_graph_bench.hip.cpp` replays the frozen chain with HIP
  Graphs and is a runtime performance NO-GO.
- `rocwmma_routed_moe_down_reduce_fused_bench.hip.cpp` fuses down WMMA with
  rank-ordered route reduction. It passed exact correctness but reached only
  `1.024x–1.031x` at the median-of-three decision cells and is a runtime
  performance NO-GO.

Do not combine negative variants or infer full-model token throughput from
their operator rates. The final down/reduction evidence is in
`down-reduce-fused-static-2026-09-08.md` and
`down-reduce-fused-results-2026-09-08.md`.
