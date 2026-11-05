# Fused gate+up+SiLU candidate — static qualification

Status: statically qualified, runtime-qualified for correctness, and rejected
for performance. See `fused-gate-up-results-2026-09-08.md`.

## Isolation and frozen control

The candidate lives in
`rocwmma_routed_moe_fused_bench.hip.cpp`. It includes the validated standalone
baseline source after renaming only its `main` token, so the five-launch control
and four-launch candidate execute in one process with the same persistent
weights, buffers, input, routes, CPU references, and HIP stream.

The baseline source and build wrapper were not edited. The candidate build
refuses to proceed unless the included baseline source still has SHA-256:

```text
8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80
```

## Candidate shape

The candidate replaces the separate gate/up WMMA and elementwise activation
kernels with one `fused_gate_up_silu_wmma` launch:

- Grid `(F/16, experts)` with one wave32 block per output tile and expert.
- One shared BF16 input fragment load per 16-wide K tile.
- Independent gate and up BF16 weight fragments.
- Two FP32 WMMA accumulators.
- Two 16x16 FP32 LDS staging tiles, exactly 2,048 bytes total.
- The same `gate / (1 + __builtin_expf(-gate)) * up` sequence as the frozen
  control, followed by the same BF16 conversion.
- Direct write to the persistent BF16 activation buffer.

The remaining input conversion, down WMMA, and weighted route-reduce kernels
are the frozen control kernels. The control therefore has five launches and the
candidate four.

## Static build evidence

Build command:

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_routed_moe_fused.sh \
  /tmp/beholdd-rocwmma-routed-moe-fused-bench
```

The separate build wrapper uses ROCm 7.13 AMD Clang 23 directly with
`-nogpuinc`, the explicit Clang resource/HIP/device-library roots, and strict
warnings. It rejects system ROCm 5.7 mixing and verifies these properties before
returning success:

```text
rocwmma_commit=bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125
rocwmma_header_tree_sha256=951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750
candidate_source_sha256=fd2ed37e4e7a1b7883f61d7e5e429fda4b38acf11e08dbf7d120064d8dffef9d
code_object_target=EF_AMDGPU_MACH_AMDGCN_GFX1151
code_object_sha256=9c07adb5d4e713000c7a6500de397cdfdf9a5c414d4982f9ad535119972c9b78
fused_native_wmma_instruction_count=2
fused_resource: vgpr=54 private_bytes=0 lds_bytes=2048 sgpr_spills=0 vgpr_spills=0 wavefront=32 max_workgroup=32 dynamic_stack=false
libamdhip64_sha256=eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05
```

This passes the predeclared static limits: no spills/private scratch, no more
than 64 VGPRs, exactly 2,048 bytes of LDS, gfx1151 targeting, and native BF16
WMMA instructions.

## Runtime protocol used

No GPU workload was launched while implementing or statically inspecting this
candidate. Independent review cleared the following guarded qualification
invocation:

```bash
cd "$BEHOLDD_SOURCE_DIR"
python3 tools/thermal_guard.py \
  --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- \
  env LD_LIBRARY_PATH=$ROCM_CORE/lib \
  /tmp/beholdd-rocwmma-routed-moe-fused-bench
```

The default run validates control and candidate correctness for
`N=1,2,4,8,16`, independently across resident and pinned-copy envelopes. It
requires finite output, mechanical relative L2 at most `5e-4`, max absolute
error at most `1e-3`, repeat bit determinism, and exact candidate/control bits.

Only `N=1` and `N=16` enter performance timing. Each sample is paired and
alternates control-first/candidate-first order. The report includes control and
candidate median/p90 wall time plus p10/median/p90 of the paired
`control_time / candidate_time` ratio. HIP events are not used.

Non-quick runs mechanically require at least four warmups, at least twelve
samples in multiples of four, burst sixteen or greater, and the complete five-N
correctness cohort. They fail with a nonzero exit unless all four N/envelope
cells reach the individual-run `1.03x` paired-median floor and stay inside the
p90 bound. All three completed runs failed the speed floor; this candidate is a
recorded negative result, not a routing option.
