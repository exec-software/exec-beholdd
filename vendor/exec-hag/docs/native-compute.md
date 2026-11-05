<!--
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT
-->

# Experimental reusable native queues

The `native-compute` feature provides `linux::compute::ComputeQueue` on the same
narrowly admitted AMD GC `11.5.1` device as the [fixed probe](native-probe.md).
HAG owns hardware resources and command lifetimes. Trusted kernel providers own
shader semantics; downstream applications own model and request behavior.

## Resource contract

One queue owns a dedicated render connection, normal-priority context, immutable
shader allocation, reusable command buffer, three resident data slots and a BO
list. Its five GTT allocations are charged to a shared `MemoryBudget` before
each individual GEM allocation request. Admission is per allocation, not an
atomic reservation of the whole queue; definite partial-construction failures
release resources already created.

Each logical data slot admits 1 byte through `MAX_SLOT_BYTES`: **4 GiB minus
4 KiB**. Its allocation rounds the payload up to a 4-KiB page and adds a 4-KiB
guard at each end. Shader storage occupies 4–32 KiB, according to the actual
packed library size. The indirect command buffer (IB) occupies 4 KiB. For slot
capacities `sizes`, the full charge is:

```text
shader_library_bytes(programs) + 4096
    + sum(round_up(sizes[i], 4096) + 8192)
```

`shader_library_bytes(&[&[u8]])` checks library geometry and returns the shader
charge without opening a device, auditing ISA or reserving budget.
`ComputeQueue::allocation_bytes()` reports the actual admitted total. Slot
limits do not establish process-memory fit, device availability or qualification
of every workload within them. Host staging, application memory and driver
allocations require separate accounting and limits.

Host reads, writes and dispatch require `&mut self`; GPU addresses, handles,
mappings and borrowed storage do not escape. Initialization uses a fixed 4-KiB
page rather than a slot-sized temporary. Data payload, padding and guards start
with canaries; shader and IB storage start zeroed. Programs are immutable after
upload. Shader and IB mappings are GPU read/execute; data is read/write and
non-executable. No mapping is GPU RWX.

Each dispatch submits once and waits synchronously for one fence. A valid fence,
completed finite wait, healthy context and intact guards are required before
reuse. Uncertain submission, completion or cleanup, and caught submission panic,
permanently poison the queue. Later reads, writes and dispatches refuse.
Uncertain resources retain the complete ownership and budget chain until process
exit; explicit `close()` reports cleanup failure and stops at the first uncertain
release. A timeout or process exit is not a GPU cancellation guarantee.

## Trusted shader boundary

Construction and dispatch are unsafe Rust operations. Programs must be audited
GFX1151 wave64 ISA with bounded indexing, correct instruction-fetch padding and
termination for every submitted argument. Geometry checks cannot validate those
properties. Untrusted model bytes, user programs and arbitrary command packets
must not enter this interface; GPU faults can affect other clients.

The base `GFX1151_SERIAL_ABI` uses three raw byte-buffer descriptors in s0..s11,
two scalar arguments in s12/s13, group X in s14 and packed local ID in v0.
Workgroups contain `[64, 1, 1]` threads. The default ABI allocates eight VGPRs and
admits up to 64 workgroups. Both raw scalar values must be in `1..=65536`.

`KernelAbi::with_vgprs()` admits multiples of four from 8 through 64 and changes
only the VGPR allocation field. `with_max_groups()` admits a library-specific
limit from 1 through 65,536. `open_programs_with_abi()` uses the chosen ABI for
the entire library. These extended bounds do not establish hardware or numerical
qualification for additional registers, groups or workloads.

`open_programs()` packs 1–32 programs with 256-byte-aligned entry points into
one allocation. Each program must be nonempty, consist of four-byte instruction
words and occupy at most 4,096 bytes including audited fetch padding. A program
may cross a page boundary. Alignment gaps and final page rounding count toward
`MAX_SHADER_BYTES`, currently 32 KiB. `open()` retains one program at entry zero;
neither API allows code replacement after construction.

`dispatch_regions()` selects a program and three `BufferRegion` descriptors,
each specifying a data slot, logical offset and nonzero byte length. Disjoint
regions may use the same slot. Out-of-bounds or overlapping descriptors, and
any overlap with the complete shader allocation, are rejected before IB mutation.
Address arithmetic is checked against the descriptor's 48-bit address ceiling.
Provider-specific alignment, shape and numerical constraints remain the caller's
duty. `dispatch()` retains program zero and the three slot prefixes.

Completed outputs can feed later dispatches without host tensor staging.
Ordinary dispatch checks guards and padding rather than numerical payloads.
Payload tracing is an explicit exception; an adapter promising no such readback
must reject `payload_tracing_enabled()`.

Each reused IB requests `AMDGPU_IB_FLAG_EMIT_MEM_SYNC`. The driver performs a
pre-IB GPU cache synchronization for host-rewritten data; completion provides
writeback before host readback. CPU fences alone cannot replace those GPU-side
operations.

## C interface

Build with `cargo build --locked --release --features capi` on Linux and use
[include/exec_hag.h](../include/exec_hag.h). The C interface wraps the same trusted
queue and does not validate arbitrary ISA or relax the Rust execution contract.

The caller owns all input and output memory and must provide valid pointer
ranges for every call. `hag_queue_open` returns one opaque owned handle on
success. Serialize access and destruction: close it exactly once with
`hag_queue_close`, and never use it after close. `hag_queue_dispatch` submits
synchronously; read and write copy checked byte ranges. Negative status codes
report failure. No Rust unwind crosses the boundary; a caught panic returns
`HAG_PANIC` and leaves the handle unusable.

C `budget_bytes = 0` selects unlimited accounting, while Rust
`MemoryBudget::new(0)` means zero capacity. C `max_groups = 0` selects the default
64-group limit. Query the supported VGPR/group bounds through the header's
functions. `hag_queue_allocation_bytes` reports the full charge;
`hag_queue_budget_remaining` returns zero for a null handle. An unlimited budget
reports `UINT64_MAX` minus its live allocation charge.
Closing a poisoned handle does not make its quarantined GPU resources reusable.

## Historical qualification

The following summarizes development results recorded from September 26 through
October 15, 2026. It is not a new test run or a certification of the current source
at every admitted bound. The fixed probe is shipped here; the downstream kernel
libraries, inference harness and application clients are separate software and
are not included in this crate. Original operational archives are not distributed
with these public summaries, so the tables are not a complete reproduction kit.

The selected inference checkpoint was dense **Qwen3-0.6B Q8_0**, with 28 layers.
Earlier operator fixtures used smaller frozen libraries. RoPE/attention used ten
programs in 8 KiB; vocabulary, complete-model and serving fixtures used eleven in
8 KiB. Expanding current library or slot admission did not retroactively qualify
new shader bytes, larger contexts or other models.

### Operator and resident-composition results

| Date | Fixture | Recorded result |
| --- | --- | --- |
| 2026-09-26 | Packed BF16/Q8_0 arithmetic | 16 cases, 32 dispatches, 604 bitwise-checked rows; changed-input queue reuse, guards and zero final budget |
| 2026-09-27 | Resident BF16-to-Q8_0 chain | Three changed-input rounds without intermediate host tensor staging |
| 2026-10-03 | SwiGLU/residual feed-forward composition | Three cases × two rounds, six logical submissions, 30 physical dispatches and 576 stage values within declared tolerances |
| 2026-10-03 | Selected real FFN layer | Three rounds, 12 dispatches and 3,072 checked outputs; 10,027,008 packed-weight bytes and 10,125,312 charged HAG bytes |
| 2026-10-03 | RMSNorm and normalized residual block | 196 diagnostic values, then three real-block rounds with 21 dispatches and 3,072 outputs; maximum absolute error `2.384185791015625e-7`; HAG charge 10,129,408 bytes |
| 2026-10-03 | RoPE | 13 dispatches, 1,562 bitwise-checked values and 26 guard checks; included a program crossing the first shader-page boundary |
| 2026-10-03 | Attention primitives | 183 dispatches, 3,218 values and 14 guard checks; finite tolerance `2e-5 + 2e-5 * abs(reference)` |
| 2026-10-03 | Real layer-0 attention with resident KV | Two three-step requests, 762 dispatches and 6,144 outputs; maximum absolute error `2.384185791015625e-7`; HAG charge 7,098,368 bytes |
| 2026-10-03 | Embedding and tied vocabulary projection | 17,152 synthetic row values and 6,144 real embedding values bitwise exact; three heads checked 455,808 logits with zero observed error and exact CPU greedy agreement; HAG charge 165,974,016 bytes |

Successful resident fixtures reported no submit/wait tensor staging and complete
budget retirement. The vocabulary plan shared 38 packed tiles and bounded
checkpoint payload staging to 4,456,448 bytes. Zero observed head error does not
change its declared acceptance criterion: finite per-value tolerance plus exact
greedy agreement. The real attention and vocabulary fixtures used
`3e-4 + 3e-5 * abs(reference)`.

The first packed-arithmetic attempts failed because the command buffer lacked
GPU execute permission. They triggered recovered compute-ring resets before the
mapping fix; they were not numerical passes. Later successful runs do not erase
that failure history. The original small arithmetic result also does not qualify
all extended BF16/Q8_0 shapes or floating-point edge cases.

### Complete-model and bounded serving results

Here `C16`, `C64` and `C128` name the downstream allocation's evaluated-position
capacity, not a global HAG limit. Complete-model comparisons used the declared
finite tolerance `2e-3 + 2e-4 * abs(reference)` plus exact CPU greedy choices.

| Date / scope | Numerical or lifecycle coverage | HAG charge (bytes) |
| --- | --- | ---: |
| 2026-10-03, complete model | Two three-position requests with reset and cached decode; 911,616 logits, maximum absolute error `5.53131103515625e-5`, all six greedy choices exact; two near-zero sign differences reported | 642,682,880 |
| 2026-10-03, C16 serving | Nine-token prompt plus cached decode; CPU token/text/usage parity, in-process HTTP/SSE cancellation at a completed-token boundary, reuse and unload/reload | 649,277,440 |
| 2026-10-04, original C64 | 911,616 boundary logits, maximum absolute error `9.1552734375e-5`, six exact greedy choices; real TCP gate completed 136 evaluations with cancellation/reuse and clean shutdown | 689,303,552 |
| 2026-10-04, reduced-storage C64 | Fresh boundary and TCP gates retained the same checked-value count, error, greedy agreement and 136 TCP evaluations | 658,952,192 |
| 2026-10-04, C128 | 129 GPU evaluations, 489,297 physical dispatches and 7,482 logical submissions; 911,616 selected full-logit values, maximum absolute error `0.00012683868408203125`, six exact greedy choices and no sign/zero mismatches | 681,910,272 |

These successful gates released their complete HAG charge. The C16 HTTP/SSE
fixture used an in-process router; actual socket behavior was covered by the
later TCP fixtures. C128 compared full logits at positions 0, 31, 63, 64, 127 and
replay 0, totaling 3,646,464 readback bytes. Refusal of a 129th position preserved
state and dispatch counts before reset/replay.

The C128 TCP gate completed 200 evaluations across six successful requests and
one real socket cancellation. Four invalid/overflow requests were refused before
generation. Request reuse, busy-unload HTTP 409, unload/reload and clean SIGTERM
exit passed. The three protocol reference replies contained only two tokens,
`<think>\n`; those checks did not establish complete-answer quality.

Memory forecasts and measured process memory were separate from HAG accounting:

| Historical configuration | HAG charge (bytes) | Combined runtime forecast (bytes) | Included recording/owner allowance |
| --- | ---: | ---: | ---: |
| Original C64 | 689,303,552 | 1,538,288,320 | 67 MiB |
| Reduced-storage C64 | 658,952,192 | 1,485,916,864 | 46 MiB |
| C128 | 681,910,272 | 1,556,061,376 | 91 MiB |

C128's boundary and TCP process-cgroup peaks were 102,744,064 and 305,504,256
bytes, respectively; maximum recorded Tctl was 75.75°C and 81.5°C. Both used a
75% CPU quota, 2-GiB process-memory cap and zero swap. Their recorded intervals
contained no memory/OOM events, matching driver faults or crash captures.
These are bounded observations, not whole-machine or total shared-UMA usage,
controlled performance measurements or a guarantee against future failures.

### Answer and client lifecycle fixtures

On October 4, one integer-list prompt passed exact CPU/native raw text, usage
and EOS parity with **29 prompt tokens, 27 emitted tokens and 56 evaluations**
per request. One native answer and four timing requests reset and unloaded
cleanly. The lazy-load response took 12.439 seconds; three warm SSE requests
reached DONE in 11.588–11.821 seconds. These descriptive client timings included
prefill and protocol overhead under a 75% CPU quota; they were not decode
throughput or controlled cold-cache measurements. Three earlier recall failures,
a terminal-period parser failure and a CPU thermal abort remained failed attempts.
General answer quality was not established.

Later fixtures used a downstream **C2048** allocation without changing the scope
of the earlier C128 results:

| Date | Fixture | Recorded result and boundary |
| --- | --- | --- |
| 2026-10-11 | Seeded conversation history | Complete raw reply, EOS and usage matched CPU: 491 prompt / 6 emitted / 497 evaluated positions; three native requests totaled 1,491 evaluations with healthy resets and unload |
| 2026-10-11 | Generated two-turn history | Each second request included its backend's actual first reply; CPU/native parity at 467/6/473 and 491/6/497, totaling 970 evaluations per backend |
| 2026-10-13 | Bounded headless client Stop and fresh-chat reuse | Stop cancelled native work after 481 evaluations; reset retained the loaded runtime; a fresh conversation matched CPU at 467/6/473; 954 total native evaluations within the 1,021 ceiling |

The standard seeded-history fixture observed a 128.875-second lazy-load response;
a warm stream reached first content at 126.657 seconds and DONE at 128.624 seconds.
These were single client-latency observations under a 75% CPU quota, including
prefill and thinking markup. Maximum recorded native Tctl was 82.875°C for the
standard history fixture, 82.625°C for generated history and 78.75°C for successful
Stop/reuse. The successful recorded windows had no task swap/OOM events, crash
captures or matching faults, with request reset and process retirement checked.
The first Stop/reuse attempt remained unqualified because an empty pooled client
connection prevented reuse; a separate repaired run passed. Earlier thermal
stops likewise did not count as completed parity results.

On October 15, downstream model-free client checks validated discovered capacity
limits and oversized-request refusal. Checkpoint-template inspection measured a
short-user prompt with 15 tool definitions at **4,826 prompt tokens**, requiring
**5,337 evaluated positions** with a 512-token reply allowance. A no-tools control
measured 86/597. These metadata and client checks ran no model or GPU inference;
they did not extend the C2048 qualification to larger contexts or qualify tool
execution. The full configured tool catalog and general coding quality remained
outside those fixtures.

## Remaining qualification

The recorded results establish finite operator, model and client paths on one
narrow device family. They do not cover arbitrary numerical inputs, every model
or context, general answer quality, multi-runtime concurrency, shared-device
isolation, asynchronous retirement, VRAM transfers, device-loss recovery or an
alternate operating-system transport. Current extended admission bounds require
new scoped evidence when used. Follow [backend gates](native-backend.md) and
[exact inference](exact-inference.md) before making broader support or performance
claims.
