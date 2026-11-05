# Native acceleration program

Date: 2026-09-07. Host: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S,
RADV GFX1151 (Mesa 25.2.8).

This program is isolated from the working reference serving deployment. That deployment is
the quality and product baseline; no experiment in this document requires changing or
restarting it. Every local sustained benchmark is supervised by `tools/thermal_guard.py`
against the package `Tctl` sensor, with a warning at 75 C and a hard abort at 90 C.

## The target is output tokens, not a misleading kernel number

For batch-1, single-sequence decode, AMD specifies 256 GB/s of memory bandwidth for this
APU. Qwen3.6-35B-A3B activates about 3B parameters per model step, so native BF16 alone
has a lower bound near 6 GB of weight traffic per step. Even an impossible 100%-of-peak
implementation is therefore bounded near 42.7 model steps/s before shared/common
weights, recurrent state, attention, routing and launch overhead. Sources:
[AMD Ryzen AI MAX+ 395 specifications](https://www.amd.com/en/products/processors/desktops/ryzen/ryzen-ai-halo/ryzen-ai-max-plus-395.html)
and [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

Consequently, 50+ *BF16 output tokens/s* needs at least one of:

- more than one accepted output token per expensive target step (MTP/self-speculation);
- fewer bytes per active parameter through an explicitly selected quantized or mixed
  representation;
- a materially different architecture whose active-weight traffic is lower.

Scheduling and kernels are still mandatory: they decide whether the engine approaches
the roofline or leaves most of it unused. They cannot repeal it.

## Completed slice 1: BF16 stays packed through the CPU Device path

Before this work, GGUF BF16 used the generic fallback in `Cpu::run_with_threads`: decode
the complete matrix to an f32 allocation, then read the expanded matrix again for the
dot product. The new path widens BF16 in AVX-512 registers as each row is consumed. It
accepts byte-addressed, potentially unaligned GGUF ranges without casting them to an
invalid `u16` slice, retains f32 accumulation and parallelizes only across independent
rows.

Development before/after wall times from the exact production `Device::matvec` call,
based on the repository starting point `84f8a26` and the first guarded packed-path run:

These are the existing Qwen3-Coder `qwen3moe` bring-up shapes, not the later Qwen3.6
expert/router geometry.

| BF16 shape | old expand-to-f32 | packed in-register | speedup |
|---|---:|---:|---:|
| router 128 x 2048 | 0.121 ms | 0.103 ms | 1.17x |
| expert gate/up 768 x 2048 | 0.462 ms | 0.217 ms | 2.13x |
| expert down 2048 x 768 | 0.494 ms | 0.219 ms | 2.26x |
| attention Wq 4096 x 2048 | 9.140 ms | 0.840 ms | 10.88x |
| attention Wo 2048 x 4096 | 9.167 ms | 0.814 ms | 11.26x |

These historical before/after figures are directional rather than a durable benchmark
artifact; the current command reproduces the packed path but the starting commit did not
contain this example. The large-shape effective packed-weight rate in that run rose from
about 1.7 GiB/s to 19 GiB/s.
Tests compare the accelerated result with the scalar BF16 definition, exercise
overwrite and scaled-accumulate modes, and explicitly cover an unaligned byte range.

Reproduce:

```sh
python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  cargo run -p beholdd-backend --release --example bench_bf16_matvec
```

## Completed slice 2: prepared packed weights on Vulkan

The old `exec-hag` Vulkan code was not carried over. It recreated buffers and command
buffers, uploaded a complete f32 weight, allocated descriptor sets and drained the queue
for each matrix operation. Its source also disagreed with its direct-DRM claims and its
model arithmetic was not a correctness oracle.

The v2 prototype instead establishes the ownership needed by M4:

- Q8_0 and BF16 remain in their checkpoint representation;
- `PreparedMatrix` owns a persistent Vulkan-visible packed allocation;
- pipeline, descriptor set, command buffer, fence, input and output allocations are
  reused;
- every Vulkan object has one RAII destruction path, including partial initialization;
- the shader consumes GGUF row-major output-by-input weights directly;
- CPU and Vulkan overwrite/scaled-accumulate results are compared before timing;
- Q8_0 fixtures vary the scale by block, and every output row has an independent error
  bound in addition to the whole-vector relative-L2 gate;
- dispatch supports matrices with more rows than one Vulkan grid dimension.

The public prototype intentionally does **not** implement `Device` yet. The current trait
accepts a borrowed host byte slice on every call. Implementing that signature literally
would hide a re-upload inside `matvec`, making the API easy to use incorrectly. The
prepared-weight evidence should shape the trait rather than be routed around it.

Representative guarded medians from the final bring-up run, including input copy,
descriptor update, command recording, submit, fence wait and output read but excluding
the separately reported one-time weight upload. Each timing sample averages eight calls.
CPU phases bracket the GPU phase for each shape; the reported comparison uses the faster
CPU median, making the stated GPU advantage conservative within that bracket. These short
operator calls are indicative measurements, not sustained thermal or end-to-end
qualification:

| format / shape | Vulkan | optimized CPU | speedup | CPU-relative L2 |
|---|---:|---:|---:|---:|
| BF16 expert gate/up 512 x 2048 | 0.069981 ms | 0.161215 ms | 2.30x | 4.14e-8 |
| BF16 expert down 2048 x 512 | 0.057699 ms | 0.161556 ms | 2.80x | 3.53e-8 |
| BF16 attention Wq 4096 x 2048 | 0.247161 ms | 0.754160 ms | 3.05x | 4.15e-8 |
| BF16 attention Wo 2048 x 4096 | 0.344784 ms | 0.762219 ms | 2.21x | 3.57e-8 |
| Q8_0 expert gate/up 512 x 2048 | 0.059987 ms | 0.150234 ms | 2.50x | 7.90e-7 |
| Q8_0 expert down 2048 x 512 | 0.069247 ms | 0.151835 ms | 2.19x | 4.07e-7 |
| Q8_0 attention Wq 4096 x 2048 | 0.310304 ms | 0.738038 ms | 2.38x | 8.43e-7 |
| Q8_0 attention Wo 2048 x 4096 | 0.209416 ms | 0.723496 ms | 3.45x | 1.20e-6 |

These shapes match the official Qwen3.6-35B-A3B text configuration: hidden size 2048,
256 router outputs, 512-wide routed experts and 4096-wide full-attention projections.
The best large-shape BF16 call reached 63.2 GiB/s of effective encoded-weight throughput
on a repeatedly reused synthetic allocation; this is not a direct DRAM-bandwidth
measurement. The final runs peaked at 73.625 C for BF16 and 70.625 C for Q8_0; neither
warned or aborted. Repeated back-to-back benchmark runs did enter the 75 C warning band,
so the guard remains mandatory. These are prepared-operator gates, not model-token
measurements and not M4 completion. Exact command output, environment, thermal reports
and source hashes are retained in
[`native-acceleration-run-2026-09-07.md`](native-acceleration-run-2026-09-07.md).

The feature build requires `glslc`; the recorded run used shaderc 2023.8-1build1,
SPIRV-Tools 2023.6~rc1-2 and glslang 14.0.0-2. Set `GLSLC=/absolute/path/to/glslc` to
select a compiler explicitly. Prebuild outside the measured thermal interval, then run
the already-built executables under the guard:

```sh
cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_bf16 --example bench_vulkan_q8 --offline

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  target/release/examples/bench_vulkan_bf16

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  target/release/examples/bench_vulkan_q8

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  cargo test -p beholdd-backend --features vulkan --offline \
  vulkan::tests::live_packed_dispatch_covers_tail_words_odd_bf16_rows_and_a_two_dimensional_grid \
  -- --ignored --exact
```

## Completed slice 3: budgeted arenas and one-submit routed MoE

The prepared-matrix result exposed the real bottleneck in the existing Qwen expert loop:
eight routes meant 24 command recordings, submissions, fence waits and host round trips,
with SwiGLU on the CPU between projections. The next slice makes both ownership and the
dependent GPU graph explicit:

- `ExecutionDevice` is an object-safe owned-resource/recorded-command seam, alongside the
  unchanged immediate `Device` oracle. Opaque weights, tensor views, batches and submissions
  are owner checked; a sequential CPU interpreter defines ordering, alias and failure
  semantics before a second accelerated implementation is admitted.
- `WeightArenaBuilder` requires a fit-authorized byte ceiling. It charges Vulkan's actual
  memory requirement before allocation, retains allocation-slot headroom, maps each block
  once during construction, accepts direct range fills, zeroes shader tail padding, and
  seals into immutable cloneable matrix handles.
- The initial BF16 routed-expert graph fuses gate and up projection with SiLU, computes all
  down projections, then reduces route weights in caller order. Its two intermediate
  activation arrays stay in persistent Vulkan buffers. Explicit compute-to-compute barriers
  separate the three dispatches; there is one command buffer, queue submission, fence wait
  and final host read.
- The bounded prototype deliberately requires the selected projections to share one arena
  block. A fully resident 256-expert layer needs block grouping or descriptor indexing.
  It covers the eight routed experts only, not Qwen3.6's separate sigmoid-gated shared
  expert, so it is a Qwen3.6-shaped operator result rather than a full-layer claim.

At H=2048, F=512 and K=8, 24 BF16 matrices occupy exactly 48 MiB inside one 64 MiB
authorized arena. The sequential CPU oracle used nonsorted conceptual expert IDs
`[231, 3, 128, 17, 255, 0, 72, 9]` and nonuniform route weights. The live hardware gate
matched it at 2.01e-7 relative L2, passed both overwrite and accumulation, and repeated
bit-for-bit.

The first wall-time comparison measured 7.292950 ms for the former 24-call Vulkan plus
CPU-SwiGLU path and 2.136867 ms for the three-dispatch/one-submit path: **3.41x faster**
under the same process conditions. This is strong evidence that submission and host
round-trip elimination is worthwhile, but it is not an idle-host absolute result. A Proton
game process was already active and subsequently observed at 60%, then 100%, GPU busy;
package Tctl reached 87.375 C. The thermal guard did not cross its 90 C abort threshold, and no
second run was attempted while that competing workload remained. The exact command and
qualification are retained in
[`native-routed-moe-run-2026-09-07.md`](native-routed-moe-run-2026-09-07.md).

An idle-host follow-up qualified paired BF16 word loads, direct two-dimensional
dispatch, and host-proven inner-loop bounds. Across three final-binary runs, the guarded
path measured 0.613-0.675 ms and the paired path 0.557-0.569 ms. Fast won every run by
1.08-1.21x; the median-of-run medians was 0.560211 ms versus 0.637241 ms (1.14x), or
about 83.7 versus 73.6 GiB/s of encoded-weight traffic. Peak Tctl was 57.875 C. CPU
relative L2 remained below 3e-7, overwrite/accumulation passed, and repeated results
were bit exact. Eligible even shapes now select paired kernels automatically; an
explicit guarded path and automatic odd/device-limit fallback remain. Exact output,
commands, scope, and hashes are in
[`native-routed-moe-fast-run-2026-09-07.md`](native-routed-moe-fast-run-2026-09-07.md).

Reproduce only on an otherwise idle APU; compile outside the guarded interval:

```sh
cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_moe --offline

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  target/release/examples/bench_vulkan_moe
```

## Completed slice 4: timestamped reduction tournament

Six timestamps now split gate/up, both inter-stage dependencies, down, and route
reduction inside the existing one-submit command buffer. The query pool is opt-in and
lazy; ordinary inference records no query commands. Timestamp packet overhead was too
variable to pass the planned 2% measurement gate consistently, so those values are
directional only and promotion remained based on uninstrumented wall time.

The profile nevertheless made the search boundary unambiguous: gate/up represented
about 74% of the final measured GPU interval, down about 24%, route reduction under 1%,
and both dependencies together about 1.2%. Fixed-tree and subgroup-add gate/down pairs
at 32, 64, and 128 lanes were then compared in a balanced Williams ordering, with the
existing 64-lane reducer held constant. Every candidate passed CPU/guarded tolerance and
within-pipeline bit repeatability.

The fixed 128-lane tree beat the 64-lane control by 1.198x, 1.236x, and 1.205x in the
three qualification runs. Subgroup WG128 was statistically tied, but its floating
reduction order is driver-defined; the explicit deterministic tree was promoted for
this exact H=2048/F=512/K=8 shape on RADV GFX1151 and Mesa 25.2.8 instead. The 64-lane
tree remains the automatic fallback for other shapes, devices, driver versions, or
tighter workgroup limits.

Three final production-path runs measured 0.435-0.464 ms, with a median-of-run-medians
of 0.440122 ms: 1.273x over the prior 0.560211 ms record, a 21.44% elapsed-time
reduction, and 5.61x over the former 24-call path in the final runs. CPU relative L2 was
2.789e-7 and repeated output remained bit exact. Exact commands, all tournament and
thermal results, caveats, and source hashes are in
[`native-routed-moe-tournament-run-2026-09-07.md`](native-routed-moe-tournament-run-2026-09-07.md).

The GPU interval was only about 76% of median profiled wall time. That observation led
to the reusable fixed-N program below. Down/reduce fusion still cannot meet a 10% gate
here: the reducer plus both measured dependencies totals only about 0.0067 ms.

## Completed slice 5: reusable fixed-N routed programs

`RoutedMoeProgram` now pins one arena block, exact H/F/K/N geometry, the selected fast
variant, private persistently mapped input/metadata/output buffers, independent aligned
scratch slices, immutable descriptor sets, two prerecorded overwrite/accumulate command
buffers, and a private fence. A fixed-N execution validates every independent item,
refreshes its input and routing metadata, records no new commands, submits the whole
stage-major batch once, waits once, and reads each independent output back. Descriptors,
buffers, command buffers, and their pinned arena retain explicit synchronized lifetimes.

The benchmark compares that complete package with N ordinary production-fast calls in
12 balanced AB/BA pairs after four warmup pairs. Program construction is excluded and
reported separately; both arms include validation, host copies, execution, synchronous
completion, and output readback. The result therefore includes persistent mapping,
immutable descriptors, private buffers, command reuse, and one submit instead of N. It
does not attribute the improvement to command recording alone.

On the exact promoted RADV GFX1151/Mesa 25.2.8, H=2048/F=512/K=8/Tree128 configuration,
all three clean-binary qualification runs cleared the 1.10x paired gate at every tested
batch size:

| Fixed N | Qualification speedups | Median of run medians |
| ---: | --- | ---: |
| 2 | 1.270x / 1.335x / 1.568x | 1.335x |
| 4 | 1.859x / 1.731x / 1.879x | 1.859x |
| 8 | 2.148x / 2.328x / 2.154x | 2.154x |
| 16 | 2.176x / 2.099x / 2.632x | 2.176x |

A later run of the hardened asserted gate also passed at 1.491x, 1.893x, 2.212x, and
2.126x for N=2/4/8/16. An exact manifested-source release rebuild then passed at 1.367x,
1.873x, 2.323x, and 2.402x, with N=1 at 1.004x. Exact-bit parity held throughout,
including independent item selection, overwrite, accumulation, repeated execution, and
isolation; all four ignored live Vulkan tests passed. Qualification Tctl peaked at
61.500 C. The earlier exploratory run peaked at 72.250 C, the first post-gate
verification at 74.875 C, and final-source verification at 70.125 C; all remained below
the 75 C warning and 90 C abort thresholds.

The explicit N=1 program measured 1.055x, 1.019x, and 1.069x in the clean runs, a
1.055x median below its 1.10x decision line. A later 1.138x observation does not overturn
that three-run NO-GO, and N=1 is not the production default. The fixed-N result applies
only to independent routed-MoE operator evaluations and its rate is routed-MoE operator
evaluations/s, not individual expert evaluations/s or autoregressive tokens/s. No
sequential token dependency was removed. Other devices, drivers, shapes, variants, and
N values remain unqualified. Exact method, results, caveats, and source hashes are in
[`native-routed-moe-program-run-2026-09-07.md`](native-routed-moe-program-run-2026-09-07.md).

## Completed research branch: native BF16 WMMA

A standalone HIP/rocWMMA harness proved exact `gfx1151` native-BF16 matrix
instructions at the same `H=2048/F=512/K=8` routed shape. It preserved BF16
weight bits and FP32 accumulation, while explicitly rounding input and
post-SiLU activations to BF16 for WMMA. Against the frozen Vulkan operator, the
copy-inclusive path reached `2.224x`, `3.076x`, `4.585x`, and `6.957x` at
`N=2/4/8/16`; `N=1` reached `1.409x` and missed its `1.50x` decision line.
These are ideal-overlap routed-operator measurements, not output tokens/s.

Three tightly scoped follow-ups tested whether launch and intermediate traffic
could improve that baseline. Fusing gate/up/SiLU slowed it by roughly 2–3%.
Replaying the unchanged five-kernel chain as a HIP Graph also regressed or
stayed flat. Fusing down WMMA with rank-ordered route reduction remained
bit-identical across original, reversed, duplicate/missing, and one-hot routes,
but its four median-of-three decision cells reached only `1.024441x–1.031432x`
and all missed their `1.08x–1.10x` thresholds. None is a backend option.

The micro-results point away from more launch-only surgery. A routed operator
streams 48 MiB of BF16 weights; saving one launch and at most a few MiB of
intermediate traffic is too small. The next vertical slice is a real-weight,
device-resident routed layer with RMSNorm, GPU router and stable top-k,
shared+routed experts, residual update, and hidden/recurrent state retained on
device. Its control should use the frozen five-kernel WMMA arithmetic and one
external synchronization. Promotion then depends on real-checkpoint logits,
greedy-token parity, TTFT, steady decode output tokens/s, and long-chain coding
quality—not the standalone routed-operator rate.

The full research record is
[`native-bf16-wmma-run-2026-09-08.md`](native-bf16-wmma-run-2026-09-08.md),
with the final fusion decision in
[`down-reduce-fused-results-2026-09-08.md`](../../tools/hip/down-reduce-fused-results-2026-09-08.md).

### Real-weight resident-layer follow-up

The next vertical slice above is now measured. A strict packer staged the
official Qwen3.6 layer-3 expert tensors without quantization, and a seven-stage
resident harness added RMSNorm, the F32 router/shared scalar, stable top-8,
eight routed experts plus the shared expert, and the residual update.

The decisive result was separating numerical precision from physical layout.
A one-time GPU transpose preserves every BF16 bit while putting expert matrices
in rocWMMA's row-major Matrix-B order. That prepared path measured `0.451102 ms`
at the three-run median of medians, a paired `2.9455x` over checkpoint layout.
A deterministic parallel router then reduced the combined path to
`0.378158 ms`, with a `1.236879x` incremental paired median of medians and
2,644.40 fixed-route MoE-FFN sublayers/s. Exact top-8 IDs, normalized finite
route weights, selected/shared prepared bytes, per-path repeatability, and
`1.205e-7` CPU-relative L2 all passed.

This produces a 66.11 tokens/s **MoE-FFN-only ceiling** across 40 blocks. It is
not a full decoder or output-token measurement: one synthetic hidden vector
and one route remain cache favorable, and the oracle mirrors the experiment's
declared mixed-precision boundaries rather than an official-model forward.
The next gate is recorded real activation/route churn, followed by full logits
and token parity. The exact artifact hashes, guarded runs, profiler attribution,
and negative vector/control results are in
[`qwen35-resident-layer-results-2026-09-08.md`](../../tools/hip/qwen35-resident-layer-results-2026-09-08.md).

### B=16 real-weight route-aware cohorts

**Correlated/high-overlap corpus.** The first resident-layer cohort packed 16
distinct phase-shifted sinusoidal hidden states into expert-major rocWMMA rows
after device-side routing. The requests were independent, but their synthetic
inputs produced a correlated, cache-favorable route set. Against 16 sequential
qualified resident-layer controls, the route-aware candidate reached a
`1.892540x` paired speedup and **4,824.28 independent single-layer MoE-FFN
evaluations/s**.

**Independent hashed corpus.** Three deterministic hash seeds then exercised
decorrelated hidden states. At the exact reported median, 99 active experts
covered 128 routed occurrences, giving a theoretical `1.44x` tile reduction.
The sequential control measured `6.321079 ms`, the cohort measured
`4.662903 ms`, and the paired speedup was `1.360078x`: **3,431.34 independent
single-layer MoE-FFN evaluations/s**, or `0.291431 ms/request` amortized across
the 16-item batch. That amortized value is not single-request latency. Dividing
the layer rate by 40 gives an approximately **85.78 requests/s 40-FFN-only
aggregate ceiling**. All three seeds passed the predeclared numerical gate; the
maximum recorded controller temperature was 61.5 C.

Both results are numerical equivalence, **not bit-exact** full output parity.
Routes, prepared weights, gathered inputs, planner mapping, repeated execution,
row-zero stages, and all-stage finiteness retained their stricter gates. These
are independent-request single-layer measurements, not autoregressive tokens/s
or full-model throughput, and no dependency within a request was removed.
Detailed harness evidence remains in the HIP results record rather than being
duplicated here. Real activation and route-trace replay across changing layers
is required before runtime integration, followed by full hidden-state/logit
comparisons and generated-token parity before reporting end-to-end token/s.

## The M4 seam and current progress

The next interface must make the fast path the natural path:

1. **Prepared storage — seam implemented, integration pending.** Binding returns an opaque backend allocation plus checked
   matrix views. CPU can retain an `Arc` to encoded bytes; Vulkan can retain an arena
   offset; HAG can retain a GEM allocation. A backend never infers identity from a host
   pointer.
2. **Arena/suballocation — prototype implemented.** One Vulkan allocation per tensor does not scale to a model.
   Packed tensors are suballocated from large persistent arenas, with explicit alignment,
   storage identity, bounds and lifetime.
3. **Device activations — routed scratch proven.** Hidden state, expert scratch and KV/recurrent state stay in the
   backend. Returning f32 to the host after every operator is a correctness bring-up tool,
   not the serving path.
4. **Recorded batches — CPU seam and fixed-N Vulkan program proven.** The synchronous
   Vulkan program now reuses immutable descriptors and command buffers for N independent
   routed-MoE operator evaluations with one submission and fence. A full token/layer
   graph, activation residency across operators, and timeline synchronization are still
   pending.
5. **Fused routed MoE — paired BF16 selected-route prototype proven.** The backend receives selected expert views and stable route
   weights, computes gate + up + SwiGLU + weighted down expert-major, and reduces in a
   deterministic order. Host-qualified paired loads are the default for eligible even
   shapes; guarded scalar loads remain the correctness/fallback control. Qwen's present
   24 serial expert matvecs per layer are the first design-pressure test.
6. **Observable phases.** Upload, kernel, queue, weight wait and synchronization time are
   reported separately. End-to-end prefill and decode t/s remain the acceptance metrics.

### Minimum safe route-cohort production seam

The benchmark is not itself a serving interface. The smallest safe integration
keeps model arithmetic and per-sequence order unchanged while making only the
resident MoE boundary schedulable:

1. **`DecodeSession` yield.** Move one request's mutable hidden, KV/recurrent,
   sampling, output, and cancellation state into a resumable session. At a
   resident-layer MoE boundary it yields a device-resident work item and resumes
   only when that item's completion event arrives; it never advances two steps
   of the same sequence concurrently.
2. **Device-owned `PreparedExpertBank` and `RouteBatch`.** Bind each immutable,
   prepared per-layer expert bank once, with backend ownership, geometry,
   precision policy, and lifetime checks. A bounded `RouteBatch` may combine
   only independent items for the same model, layer, bank, and numerical policy.
   Routing, expert grouping, gather/scatter, and caller-order reduction stay on
   device; session identity and destination views remain attached to every row.
3. **Reservation ledger.** Before admission, atomically reserve the model bank,
   per-session state, batch scratch, queue slots, and all memory retained until
   device completion. A reservation remains charged while an event can still
   reference it and is released exactly once, preventing optimistic concurrent
   sessions from overcommitting unified memory.
4. **Item-local cancellation.** Cancellation before submission removes and
   refunds only that item. Cancellation after submission marks only its result
   for discard; its reservation and device resources survive until completion.
   It must not reset a shared stream, tear down the batch, publish partial text,
   or fail unrelated sessions.
5. **Bounded event queues.** A fair, age-bounded ready queue feeds admitted
   batches, and a completion queue resumes sessions from backend events without
   a device-wide synchronization. Queue capacity and maximum cohort wait are
   explicit admission inputs; failures are fanned out to the affected items and
   every retained reservation is reconciled.

This seam should first run a one-item `RouteBatch` against the existing
qualified path, then enable multi-item cohorts. Until its ownership, budget,
cancellation, and event invariants are tested, the current serialized runtime
remains the production fallback.

M4 is passed only after the same real checkpoint and prompt produce the same greedy token
sequence as CPU within a stated logit tolerance, Vulkan is faster end-to-end, and the
model/runtime contain no Vulkan-specific branch.

## Qwen3.6 and the route to 50+

Qwen3.6 is not the current `qwen3moe` implementation with another architecture string.
Its official configuration includes 30 Gated DeltaNet layers, 10 full-attention layers,
a shared expert in addition to 8-of-256 routed experts, recurrent state and a trained MTP
head. It should land as a separate model module after the prepared/batched backend seam is
real. Sources: [official config](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json)
and [llama.cpp's Qwen3.6 implementation](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35moe.cpp).

The high-leverage experiment after ordinary MTP parity is **expert-major MTP
verification**:

1. draft several positions with the checkpoint's trained MTP head;
2. collect target routes for the verification positions;
3. group work by expert and evaluate repeated experts as small matrix-matrix operations;
4. scatter outputs back and reduce in original route order;
5. accept the verified prefix exactly as ordinary speculative decoding does.

This can reuse an expert's packed bytes across several candidate positions and amortize
dispatch without changing target-model semantics. It remains a hypothesis until route
traces show enough repeated expert bytes (initial go/no-go threshold: 15%) and an oracle
proves identical accepted tokens.

The fixed-N program is a suitable execution primitive only after those candidate
positions are independently valid work. Its measured 1.27-2.63x batch speedups cannot
be relabeled as autoregressive token throughput or used to bypass target-model
dependencies.

“MoEoE” is not assumed to have a standard meaning. If it is the planned hierarchical
router -> group -> subexpert architecture, it will require explicit outer/inner routing
and two data-dependent prefetch barriers. Its exact definition must be written down before
the engine claims support.
