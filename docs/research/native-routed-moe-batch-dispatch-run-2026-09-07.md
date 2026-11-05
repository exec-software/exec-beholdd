# Native routed-MoE item-dimensional dispatch run — 2026-09-07

## Status before measurement

This run starts from the qualified fixed-`N` `RoutedMoeProgram`: five private
activation allocations, persistent input/output/metadata mappings, immutable
descriptors, two prerecorded overwrite/accumulate command buffers, and one submit plus
one fence per batch. Its control records `N` gate/up dispatches, one barrier, `N` down
dispatches, one barrier, then `N` reductions (`3*N` dispatches total).

The candidate changes only dispatch dimensionality and descriptor span. It binds three
whole-batch descriptor sets and records:

1. gate/up as `(F, K, N)`;
2. down as `(H, K, N)`;
3. route reduction as `(ceil(H/64), 1, N)`.

`gl_WorkGroupID.z` selects the independent item. Buffers remain logically
`input/output=[N,H]`, hidden scratch `[N,K,F]`, route scratch `[N,K,H]`, and metadata
`[N,K,4]`. The candidate does not change BF16 weights, Tree128 pair iteration, dot
expressions, SiLU, shared-memory reduction trees, route order, queue count, host copies,
submission count, or fence behavior.

The reference serving runtime and its live model are outside this experiment and must
not be modified or stopped. All Vulkan workloads on this host run through
`tools/thermal_guard.py --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- ...`.

## Predeclared qualification gate

Qualification uses the exact promoted profile only: Radeon 8060S / RADV GFX1151,
Mesa 25.2.8, BF16 `H=2048`, `F=512`, `K=8`, Tree128, and fixed
`N in {2,4,8,16}`. The existing `3*N` prerecorded program is the paired control; the
ordinary immediate path is reported separately and is not the promotion denominator.

For each `N`, promotion requires all of the following before results are inspected:

- exact `f32::to_bits` equality with independent immediate execution and the `3*N`
  control for overwrite, accumulation, differing inputs, differing route weights, and
  same-block route replacement/reordering;
- deterministic repeatability and item isolation;
- transactional refusal for invalid lengths, non-finite route weights, invalid offsets,
  foreign arena/block selections, descriptor-range overflow, `u32` index overflow, and
  unsupported dispatch geometry;
- three complete thermally guarded paired benchmark runs;
- median-of-paired-ratios at least `1.10x` candidate/control in every run;
- no complete-run paired median below `1.05x`;
- no thermal abort, sensor failure, validation error, or leaked Vulkan allocation.

`N=1` is a correctness/control measurement only and will retain the existing path.
Batch sizes qualify independently; a miss for one `N` does not authorize promotion for
that `N`, and results from this exact hardware/driver/shape do not authorize other
profiles. Report throughput as routed-MoE operator evaluations per second, not tokens
per second.

## Planned follow-up

If item-dimensional dispatch qualifies, the next experimental stage is expert-major
Tile4 reuse: group equal absolute gate/up/down expert tuples across ready items, load
each weight pair once per workgroup, preserve each original `[item,route]` destination,
and retain ascending route reduction. That stage must compare high-, partial-, and
zero-overlap route distributions and fall back to item-dimensional dispatch whenever
the measured reuse threshold is not met.

## Results

Implementation and measurement completed on 2026-09-08. The candidate passed the
live exact-bit overwrite, accumulation, route replacement/reordering, repeatability,
item-isolation, concurrent-call serialization, and transactional-refusal checks. All
three new shaders passed optimized Vulkan 1.2 compilation and `spirv-val`; their
floating-point SPIR-V opcode sequences match the corresponding Tree128 control stages.

The three complete guarded runs produced these paired
`per-item-3N / item-dimensional-3D` medians:

| batch | run 1 | run 2 | run 3 | decision |
|---:|---:|---:|---:|:---|
| 1 | 0.991x | 1.014x | 0.995x | control only |
| 2 | 0.977x | 1.009x | 1.054x | no-go |
| 4 | 1.111x | 1.072x | 1.004x | no-go |
| 8 | 0.852x | 1.197x | 1.023x | no-go |
| 16 | 0.826x | 0.852x | 1.002x | no-go |

Maximum observed Tctl was 65.625 degrees C; no run thermally aborted or lost its
sensor. Every candidate batch remained bit-identical to its independent Tree128 calls.

No batch size met the predeclared rule. `N=4` cleared `1.10x` once but missed it twice;
`N=8` showed the same instability and regressed materially once; `N=2` and `N=16`
never cleared the gate. The production/default program therefore remains the qualified
per-item `3*N` recording. The item-dimensional path stays explicit and experimental so
it remains a useful control for the expert-sharing stage without silently regressing
ordinary execution.

The result also narrows the mechanism: deleting dispatch packets alone is not a stable
speedup on this driver. The next candidate must reduce material work or memory traffic,
not merely host/command-front-end work. Expert-major Tile4 is the next experiment.
