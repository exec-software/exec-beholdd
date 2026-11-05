# Native routed-MoE expert-major Tile4 run — 2026-09-08

## Status before measurement

The production/default fixed-`N` Vulkan program remains the qualified per-item
Tree128 recording. The item-dimensional three-dispatch experiment was exact but did
not produce a stable speedup, so it remains an explicit control only.

This candidate reduces material work instead of only dispatch count. It partitions
the row-major `N*K` route occurrences by the exact absolute
`(gate, up, down)` weight-word triple. In deterministic first-seen expert order, each
group is emitted as contiguous chunks of at most four occurrences. One Tree128
workgroup then loads each BF16 weight pair once and applies it to as many as four
independent input activations and reduction trees. Results scatter back to their
original `item*K + route` slots, and the unchanged reduction shader combines routes
in ascending per-item route order.

The mapped metadata buffer contains the ordinary `[N,K,4]` route records, two
`VkDispatchIndirectCommand` triplets, and worst-case space for `N*K` eight-word tile
records. Inactive tile lanes repeat the first valid occurrence. Gate/up and down use
indirect `(F,T,1)` and `(H,T,1)` dispatches, where
`T = sum_expert ceil(occurrences(expert)/4)`; reduction remains direct
`(ceil(H/64),1,N)`.

The reference serving runtime and its live model are outside this experiment and
must not be modified or stopped. All Vulkan workloads on this host run through
`tools/thermal_guard.py --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- ...`.

## Predeclared qualification gate

The first qualification profile is exact: Radeon 8060S / RADV GFX1151, Mesa 25.2.8,
BF16 `H=2048`, `F=512`, `K=8`, Tree128, and fixed
`N in {2,4,8,16}`. The qualified per-item Tree128 program is the performance and bit
control. `N=1` is reported only as a no-reuse control.

Before results are inspected, promotion requires all of the following:

- exact `f32::to_bits` equality with independent Tree128 execution and the per-item
  program for overwrite and accumulation;
- exactness under different inputs, route weights, repeated routes, and same-block
  route replacement/reordering, plus repeatability and item isolation;
- transactional refusal for invalid lengths, non-finite route weights, invalid
  offsets, foreign arena/block selections, descriptor-range overflow, `u32` index
  overflow, and unsupported dispatch geometry;
- optimized Vulkan 1.2 shader compilation and `spirv-val`, with the Tree128 floating-
  point operation order retained independently for every active occurrence;
- three complete thermally guarded paired high-overlap benchmark runs;
- for each promoted `N`, median-of-paired-ratios at least `1.15x` Tile4/control in
  every complete run and no run below `1.10x`;
- no thermal abort, sensor failure, validation error, or leaked Vulkan allocation.

High overlap uses the same eight physical experts in rotated route order for every
item. It gives `M=N*K` and `T=K*ceil(N/4)`: `2.0x` schedule compression at `N=2`
and `4.0x` at `N=4,8,16`.

Before automatic selection, a representative partial-overlap distribution and a
zero-overlap distribution must also be measured in three guarded runs. The initial
selector floor is exact integer schedule compression `M/T >= 2` with at least eight
saved tiles (`M-T >= 8`). A distribution meeting that floor must clear `1.10x` in
every run to authorize this floor. A distribution below it must select the qualified
per-item fallback; its raw forced-Tile4 timing is diagnostic only. All threshold
comparisons use integer cross-products, never rounded floating-point ratios.

Qualification is scoped to this exact backend/device/driver/shape/precision/kernel
profile. Report throughput as routed-MoE operator evaluations per second, not tokens
per second. Program construction is excluded from steady-state timing but reported
separately; grouping, mapped metadata writes, submission, fence wait, and final host
readback remain inside every sample.

## Results

The candidate passed optimized Vulkan 1.2 shader compilation, `spirv-val`, strict
Rust checks, and live Vulkan validation-layer execution. It was bit-identical to
independent Tree128 calls for overwrite, accumulation, repeat execution, different
inputs and route weights, same-block route replacement, item isolation, and the
count-one, count-two, and count-four tile paths. Pure schedule tests cover exact
absolute-triple grouping, interleaved groups larger than four, stable group/chunk
order, valid inactive-lane padding, and deterministic reconstruction.

Three complete guarded high-overlap runs produced these paired
`per-item Tree128 / expert-major Tile4` medians:

| batch | compression `M/T` | run 1 | run 2 | run 3 | decision |
|---:|---:|---:|---:|---:|:---|
| 1 | 1.0x | 0.869x | 0.961x | 0.952x | control only |
| 2 | 2.0x | 1.053x | 1.069x | 1.118x | no-go |
| 4 | 4.0x | 0.953x | 1.026x | 1.166x | no-go |
| 8 | 4.0x | 0.832x | 1.127x | 0.972x | no-go |
| 16 | 4.0x | 1.049x | 0.831x | 1.244x | no-go |

Maximum Tctl was 83.0 degrees C; no run reached the 90 degree C abort threshold,
lost its sensor, or failed validation. All timed outputs remained bit-exact.

No batch met the predeclared rule. `N=2` never reached the `1.15x` median gate;
every 4x-compression batch had at least one regression below the `1.10x` hard floor.
The largest wins were not repeatable across runs. Because the strongest possible
route-overlap distribution already failed, partial- and zero-overlap qualification
cannot authorize this implementation and was not used to construct an automatic
selector. The default remains the qualified per-item recording; Tile4 remains an
explicit research path.

The result suggests that repeated per-item weight reads are already substantially
served by cache on this UMA device, while four simultaneous accumulator/reduction
trees add register, shared-memory, and barrier pressure. The next experiments target
that cost directly with a 64-lane subgroup-shuffle tail that preserves Tree128's
addition order, and separately investigate `VK_KHR_cooperative_matrix` for native
BF16 matrix execution.

## Shuffle-tail iteration: status before measurement

The second Tile4 iteration changes only how each of the four independent Tree128
reductions is completed. All 128 lanes first publish their local sums once. On the
qualified 64-lane-subgroup profile, subgroup zero loads lanes `i` and `i+64` and adds
them, then `subgroupShuffleDown` performs offsets `32,16,8,4,2,1`. This is the same
pairwise addition tree as the original shared-memory implementation, but uses one
workgroup barrier instead of seven. The host now refuses this candidate unless compute
exposes 64-lane subgroups and `BASIC|SHUFFLE_RELATIVE`.

The qualification profile, high-overlap routes, exact-bit requirements, three-run
protocol, thermal guard, and per-item Tree128 denominator remain unchanged. Promotion
still requires at least `1.15x` in every run and no run below `1.10x`; `N=1` remains a
no-reuse control. This optimization is measured separately from the completed
shared-tree no-go above.

### Shuffle-tail results

The live exact-bit and validation-layer checks passed, but all three guarded
performance runs failed the gate:

| batch | compression `M/T` | run 1 | run 2 | run 3 | decision |
|---:|---:|---:|---:|---:|:---|
| 1 | 1.0x | 0.806x | 0.817x | 0.832x | control only |
| 2 | 2.0x | 0.989x | 0.923x | 0.953x | no-go |
| 4 | 4.0x | 0.948x | 1.007x | 0.886x | no-go |
| 8 | 4.0x | 1.060x | 0.763x | 0.781x | no-go |
| 16 | 4.0x | 0.757x | 0.900x | 0.898x | no-go |

Maximum Tctl was 65.375 degrees C. Replacing shared-memory tail steps with subgroup
shuffles made this workload consistently worse, despite preserving output bits. This
rules out workgroup-barrier count as the primary Tile4 problem on the measured driver;
the remaining evidence points more strongly to register pressure, occupancy, and the
fact that the per-item control already reuses expert weights through cache.
