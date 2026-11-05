# Down-WMMA + route-reduction fusion — static qualification

Date: 2026-09-08. Status: **statically and numerically qualified; runtime performance NO-GO**.

This is an isolated research candidate. It does not modify the validated
five-launch benchmark, the earlier gate/up-fusion experiment, a HIP Graph
experiment, Beholdd's runtime, or application service. No GPU workload was launched while
building or inspecting it. A later thermally guarded runtime qualification is
recorded separately in `down-reduce-fused-results-2026-09-08.md`.

## Isolation and mapping

`rocwmma_routed_moe_down_reduce_fused_bench.hip.cpp` includes the frozen
`rocwmma_routed_moe_bench.hip.cpp` after renaming only its `main` token. The
same process therefore contains:

- the unchanged five-launch control: input conversion, gate/up WMMA, SiLU/up
  conversion, down WMMA, and rank-ordered route reduction; and
- a four-launch candidate that replaces the last two launches with
  `fused_down_route_wmma`.

The fused kernel launches `H/16 = 128` workgroups of 256 threads. Each
workgroup owns one 16-column hidden-output tile. Its eight wave32s map exactly
to experts 0 through 7. Each wave computes the existing `16 x 16` down-WMMA
tile for its expert and stores the FP32 accumulator through rocWMMA's
row-major store into a disjoint expert-major LDS plane.

The mandatory LDS footprint is:

```text
8 experts * 16 token rows * 16 output columns * 4 bytes = 8192 bytes
```

All 256 threads then execute one full-workgroup barrier. Only after that
barrier does thread `tid` map to `token = tid / 16` and
`tile_column = tid % 16`; padded rows with `token >= N` return at this point.
Every active thread reduces route ranks 0 through 7 in the same order as the
frozen reducer. Route entries remain global in this first candidate, so there
is no extra route-staging LDS or second barrier.

This mapping deliberately preserves reordered routes, duplicate expert IDs,
missing experts, zero weights, and one-hot weights. It still computes all
eight resident expert tiles. It is scoped to the ideal high-overlap
`E = K = 8` cohort and does not by itself generalize to a larger expert union.

## Static build result

The build is isolated from Ubuntu's system ROCm 5.7 headers. It uses
`-nogpuinc`, the ROCm 7.13 wheel's Clang resource root/device libraries/HIP
runtime, and a disposable unversioned HIP-library symlink. It pins the compiler,
HIP headers and runtime, Clang resource headers, device-library tree, frozen
baseline, candidate source, public rocWMMA commit/header tree, and deterministic
device `.text` bytes.

Exact build command:

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_routed_moe_down_reduce_fused.sh \
  /tmp/beholdd-rocwmma-routed-moe-down-reduce-fused-bench
```

The strict `-Wall -Wextra -Werror` build passed with:

```text
compiler=AMD clang 23.0.0git 43215c73116c407735c85a180d174f718798c328+PATCHED:2506c552d8428e2cc1778bef048b20f818e06bb3
rocwmma_commit=bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125
rocwmma_header_tree_sha256=951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750
baseline_source_sha256=8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80
candidate_source_sha256=f37105d5ee4ebc3ffac3022b306c16e8096fdb9bf72009bba80f47b738862417
code_object_target=EF_AMDGPU_MACH_AMDGCN_GFX1151
code_object_sha256=6b0b9221a9571c76da0d5e0f206f00fb12688355317755715bc6bfe201899658
device_text_sha256=f514c3a148892e07faa2fde2c6771706f3f0b3d07705c2e0f9dedc9d50e3452d
fused_native_wmma_instruction_count=2
fused_barrier_count=1
fused_lds_instruction_counts=stores:4,loads:8
fused_resource=vgpr:51,private_bytes:0,lds_bytes:8192,sgpr_spills:0,vgpr_spills:0,wavefront:32,max_workgroup:256,dynamic_stack:false
libamdhip64_sha256=eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05
```

The two static WMMA opcodes are intentional: this pinned compiler emits the
32-step `F/16` loop with a two-way unroll, matching the frozen down kernel's
instruction inventory. The build also proves that the candidate contains LDS
stores, LDS loads, and exactly one workgroup barrier.

## Correctness contract

Qualification is required to validate every `N in {1,2,4,8,16}` in both the
device-resident and pinned-copy host-visible envelopes. Before each observation,
the harness poisons every intermediate and output. Host-visible validation also
poisons device input/routes and pinned host output so a missing captured or
direct copy cannot pass from stale state.

For every `N`, both control and candidate execute twice and must be bit
deterministic. Candidate output must exactly match the frozen control and both
paths must independently satisfy:

- finite FP32 output;
- relative L2 at most `5e-4` and max absolute error at most `1e-3` against the
  CPU oracle that mirrors both BF16 activation boundaries; and
- relative L2 at most `5e-3` and max absolute error at most `1e-4` against the
  current Beholdd reference with original F32 input and F32 post-SiLU
  activation. These bounds are deliberately looser than the mechanical oracle
  because they measure the candidate precision policy, not an implementation
  mismatch.

That complete matrix is repeated for four deterministic route tables:

1. the original per-token expert permutation;
2. the same `(expert, weight)` entries in reversed rank order;
3. one duplicated expert replacing the final expert, producing exactly one
   duplicate and one missing expert per token; and
4. a full expert permutation with exactly one unit weight and seven zero
   weights per token.

The host preflight rejects any generated route with an out-of-range expert,
non-finite weight, or a structural mismatch with its declared variant.

## Predeclared performance gate

Only the original permuted route table enters performance timing. Qualification
defaults are four warmups, twelve paired samples, and a burst of sixteen full
operators per wall-clock sample. Non-quick runs reject fewer warmups, fewer than
twelve samples, a sample count not divisible by four, a burst below sixteen,
or a narrowed `--tokens` set. `--quick` output is explicitly smoke-only.

Within every sample, control-first/candidate-first order alternates. Timing is
`std::chrono::steady_clock` around a burst plus final stream synchronization;
HIP events are not used. Three independent guarded runs are required. In each
run, `N=1` and `N=16`, independently for resident and host-visible timing, must
meet both:

- paired median `control_time / candidate_time >= 1.03`; and
- candidate p90 no greater than `1.05 * control p90`.

After three runs, every resident and host-visible cell must have a
median-of-three paired ratio of at least `1.10` for `N=1` and at least `1.08`
for `N=16`. Passing only `N=16` is at most a cohort-only result, not an
interactive decode path. Exit code zero from one process means only that its
four per-run cells passed. The process explicitly reports overall qualification
as pending; an external result aggregation must apply the median-of-three gate
to three independently guarded process runs.

Every future invocation, including a smoke run, must be thermally guarded:

```bash
cd "$BEHOLDD_SOURCE_DIR"
python3 tools/thermal_guard.py \
  --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- \
  env LD_LIBRARY_PATH=$ROCM_CORE/lib \
  /tmp/beholdd-rocwmma-routed-moe-down-reduce-fused-bench
```

The command was subsequently executed as one smoke and three independent full
GPU runs under the stated guard. Correctness passed bit-exactly, but the
candidate failed its performance gate; see
`down-reduce-fused-results-2026-09-08.md`.

## Interpretation boundary

At `N=1`, fusion removes the fixed 1 MiB expert-output write plus 64 KiB of
route-reducer reads. At `N=16`, it removes a 1 MiB write and 1 MiB of reads,
plus one launch in both cases. It does not remove any of the operator's 48 MiB
of persistent BF16 weights.

The risk is scheduling, not LDS capacity: 1,024 one-wave down workgroups become
128 eight-wave workgroups, followed by a workgroup-wide barrier. The gate/up
fusion already showed that eliminating one launch and approximately 2 MiB of
traffic is not sufficient by itself. This candidate later reached only
`1.024441x–1.031432x` at the median-of-three decision cells and failed all four
aggregate thresholds. A graph-vs-graph layer is not warranted unless a future
design first finds a material direct-submission win.
