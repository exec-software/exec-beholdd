# Native fixed-N routed-MoE program qualification — 2026-09-07

This is the durable qualification record for Beholdd v2's reusable fixed-size Vulkan
routed-MoE program. It follows the timestamp and workgroup tournament in
[`native-routed-moe-tournament-run-2026-09-07.md`](native-routed-moe-tournament-run-2026-09-07.md)
and retains that run's promoted deterministic Tree WG128 kernels.

This remains a synthetic H=2048/F=512/K=8 operator benchmark. Its throughput unit is
**independent routed-MoE operator evaluations per second**, not individual expert
evaluations or autoregressive output tokens per second. It does not make sequential
positions in one decode independent, and it is not a full-model or M4 gate. The working
reference deployment deployment, application service, its serving model, and the golden runtime were not changed,
restarted, or exercised.

The tree was based on `main` commit
`84f8a2611e55f50b7e06a950f9cd9aa38afd74a4`; the acceleration work remained an
uncommitted working-tree patch when recorded.

## Program architecture

`VulkanMatvec::prepare_routed_moe_bf16_fast_program` constructs the explicit N=1
program. `prepare_routed_moe_bf16_fast_batch_program` constructs a nonzero, fixed-N
program for independent work. A `RoutedMoeProgram` pins:

- the exact Vulkan context, immutable weight arena block, H/F/K geometry, batch length,
  and production fast-kernel variant selected at construction;
- private, persistently mapped input, route-metadata, and output allocations;
- aligned, nonoverlapping per-item descriptor ranges and private hidden/route scratch;
- immutable descriptor sets for all three stages of every item;
- separate overwrite and accumulation command buffers recorded once; and
- a private fence, with program and queue mutexes serializing reuse.

Each execution validates every item's context, arena, block, dimensions, absolute
weight ranges, lengths, and finite route weights before changing mapped state. It then
copies each independent input and route-weight row, optionally copies the accumulation
base, submits one prerecorded command buffer, waits once, and copies all result rows
back. The command buffer records every gate/up dispatch, one compute dependency, every
down dispatch, a second dependency, and every route reduction. The N items share only
immutable weights and have independent activation, metadata, scratch, and output slices.

The command buffer is not reset or rerecorded between executions. Descriptor sets and
their referenced allocations remain alive and unchanged while it can execute, and fence
completion precedes mapped reads, subsequent writes, and ordinary clean-state
destruction. Device-wide idle is reserved for the conservative poisoned-state fallback
after an uncertain submit or wait.

## Measurement contract

For each N in 2, 4, 8, and 16, the control performs N ordinary production
`routed_moe_bf16_fast` calls. The candidate performs the same N independent routed-MoE
operator evaluations through one fixed-N `execute_batch` call. Inputs and route weights
differ by item, while both arms use the same retained expert arena and the same
automatically selected Tree WG128 arithmetic. Four warmup pairs precede 12 measured
AB/BA pairs, so each arm appears first equally often.

Program construction is outside the timed interval and reported separately. Both timed
arms include public validation, input and route-metadata host copies, Vulkan execution,
synchronous completion, and final result readback. The candidate additionally combines
N evaluations into one queue submission and fence wait.

Consequently, the paired speedup qualifies the complete prepared-program package:
persistent mappings, immutable descriptors, private buffers, prerecorded commands, and
one submit instead of N. It does **not** isolate command recording as the sole cause.
The predeclared go gate is median paired control/candidate speedup of at least 1.10x for
every tested N.

The explicit N=1 program measured 1.055x, 1.019x, and 1.069x in the three clean
qualification runs, a 1.055x median. It did not clear 1.10x, remains NO-GO, and is not
selected by the ordinary production path. There is no claim that persistent recording
alone improves one routed-MoE operator evaluation.

## Exploratory result

The first complete exploratory binary run established the expected scaling and exact-bit
correctness before the three-run qualification:

| N | Paired control/program speedup |
| ---: | ---: |
| 2 | 1.264x |
| 4 | 1.881x |
| 8 | 2.280x |
| 16 | 2.464x |

Maximum Tctl was 72.250 C, below the 75 C warning threshold and well below the
90 C hard abort.

## Three-run binary qualification

Three clean executions of the already-built release binary produced:

| Run | N=2 | N=4 | N=8 | N=16 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.270x | 1.859x | 2.148x | 2.176x |
| 2 | 1.335x | 1.731x | 2.328x | 2.099x |
| 3 | 1.568x | 1.879x | 2.154x | 2.632x |
| Median of run medians | 1.335x | 1.859x | 2.154x | 2.176x |

All 12 measurements cleared 1.10x; the narrowest result was 1.270x at N=2. Every
candidate output matched its independent ordinary fast calls bit for bit, including
overwrite and accumulation. Maximum Tctl across the qualification was 61.500 C.

After those three runs, the benchmark was hardened so a qualified Tree WG128 run exits
with failure unless every N=2/4/8/16 paired median clears 1.10x. A separate post-
qualification execution of that asserted gate passed at 1.491x, 1.893x, 2.212x, and
2.126x respectively. N=1 happened to measure 1.138x in that later run, but its gate is
not asserted and one favorable result does not overturn the three-run NO-GO decision.
The verification reached 74.875 C, still below the 75 C warning threshold, and is
intentionally not folded into the three-run qualification medians.

After the final null-map cleanup and source-manifest refresh, an exact manifested-source
release rebuild passed the guarded asserted gate again: 1.367x at N=2, 1.873x at N=4,
2.323x at N=8, and 2.402x at N=16. N=1 measured 1.004x, consistent with keeping it
explicit and nondefault. This final-source verification reached 70.125 C and is likewise
not folded into the three-run qualification medians.

## Correctness and scope

The live program test compares the reusable path with independent production-fast calls
at exact f32 bit level. It covers repeated overwrite, accumulation, changed inputs and
route weights, different same-block expert selections, per-item isolation, and refusal
of invalid batch shapes, non-finite weights, and out-of-range metadata without changing
the caller's output. All four ignored live Vulkan backend tests passed when explicitly
enabled on the acceleration host. Their final-source-tree rerun passed 4/4 and peaked at
61.625 C.

The performance qualification is intentionally narrow:

- Radeon 8060S / RADV GFX1151;
- Mesa 25.2.8;
- H=2048, F=512, K=8;
- the automatically selected deterministic Tree WG128 fast kernels; and
- fixed batch sizes N=2, 4, 8, and 16.

Every other device, driver version, shape, kernel variant, or batch size remains
unqualified by these numbers. The ordinary selector retains Tree WG64 as its fallback,
and the fixed-N API remains explicit rather than silently changing existing execution.

## Known production-hardening work

These items do not block or invalidate the qualified N<=16 path, but should be closed
before broader or refactor-heavy production use:

- arbitrary N needs fit-authorized activation and scratch budgeting rather than relying
  on the qualified fixed-size envelope;
- mapped-copy length, range, and visibility invariants should become release-checked;
- an explicit in-flight state should precede callbacks, instrumentation, or asynchronous
  execution extensions; and
- the empirical gate should eventually key a named qualification profile rather than
  indirectly using Tree128 selection as its profile proxy.

The result is useful only where the scheduler has genuinely independent routed work,
such as separate requests or independently valid candidate evaluations. Dividing a
batch wall time by N yields routed-MoE operator evaluations/s; it must not be presented
as individual expert evaluations/s or autoregressive tokens/s. A single sequence still
has layer and token dependencies, and full-model token parity and end-to-end decode
throughput remain required for M4.

## Environment and reproduction

- Host: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S.
- Vulkan device: RADV GFX1151, Mesa 25.2.8.
- Thermal guard: package Tctl warning 75 C, hard abort 90 C, 0.25-second polling.
- Build occurs outside the guarded interval; qualification executes the resulting binary
  directly.

```sh
cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_moe --offline

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  target/release/examples/bench_vulkan_moe
```

The ignored hardware suite can be built first and then run under the same guard:

```sh
cargo test -p beholdd-backend --features vulkan --lib --offline --no-run

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  cargo test -p beholdd-backend --features vulkan --lib --offline -- \
  --ignored
```

## Source manifest

```text
29650295b59d3c350518bfb83c79b133e5311bb1fa4f36c5dff2341f42c20440  crates/beholdd-backend/src/vulkan.rs
d7e6b4d82229e648cf0cbc2db35fe90ef0dec8fe892853bc8dee3d9a092cc593  crates/beholdd-backend/examples/bench_vulkan_moe.rs
e42696f4a8f403505f0f469a1a5a6b44390a70b581396c021d728299e036b8ab  crates/beholdd-backend/build.rs
8525498a281c4eea52311a83af17d2bd55874c91cd7d3a98b09accdec7d84e17  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_fast_wg128.comp
1d823721bb2755710554c9682cfd762ffca5482bb41512e8ae2c94a2fb343acc  crates/beholdd-backend/shaders/bf16_routed_moe_down_fast_wg128.comp
a4feb83785a6998da6f3e00a7f565e66caa2a048aaa710a1f533da5deb30f4d5  crates/beholdd-backend/shaders/bf16_routed_moe_reduce_fast.comp
f63408c241b41ce44a6efab84b5a094a4d0f5ee4b449e7b176ad0dff91b3a94b  Cargo.lock
```
