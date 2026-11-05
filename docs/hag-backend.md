# Native HAG backend status

HAG is Beholdd's experimental native GPU transport. It is intended to replace
the experimental Vulkan execution path only after the required qualification.
HAG's canonical implementation is the
standalone [`exec-software/exec-hag`](https://github.com/exec-software/exec-hag)
repository, not a historical nested inference prototype. Beholdd
continues to own models, packed weight formats, inference and scheduling.

Measured editor prompt requirements and the next capacity work are tracked in
[HAG context requirements for Antimatter](hag-editor-context.md).

Native OpenMythos/Qwen3.5 development and its separate qualification gates are
tracked in [OpenMythos native HAG development](hag-openmythos.md). The existing
OpenMythos profiles still use llama.cpp/Vulkan until those gates pass.

## What works now

- The optional `beholdd-backend/hag` feature links the standalone HAG library.
- HAG has hardware-free tested shared memory budgets and Linux AMDGPU resource
  discovery, queries, GEM allocation, owned device lifetime and bounded host access.
- Its separate opt-in `native-probe` implements one fixed shader submission with
  finite completion, exact byte/guard verification and uncertain-resource
  quarantine. One live dispatch has passed on the allowlisted GC `11.5.1` device.
  Beholdd's `hag` feature does not enable or invoke that probe;
  see [the probe's scope and live-validation status](../vendor/exec-hag/docs/native-probe.md).
- `beholdd_backend::hag::preflight()` reports native memory support, discovered
  devices and rejected-node diagnostics without allocating buffers or running work.
- `require_compute()` still refuses general native inference selection through the
  discovery API. Explicit experimental serving uses the bounded resident path
  below; discovery never loads a model or selects an implicit CPU/Vulkan fallback.
- Separate `hag-native` opt-in code provides packed BF16/Q8_0 kernels and a
  bounded experimental `ExecutionDevice`. The packed single-operator suite has
  passed on GC 11.5.1, as has one small resident-composition chain. Neither result
  qualifies full model execution or serving.
- The original one-workgroup SwiGLU/scaled-residual extension passed its isolated
  and bounded resident FFN gates on October 3; those records apply to their frozen
  binaries. The extended Q8_0/SwiGLU payloads subsequently passed a 65-row
  two-workgroup diagnostic and three reused submissions of one real-checkpoint
  dense FFN. The extended BF16 payload and the complete admitted geometry
  still require live qualification; see the [model-shaped result](#model-shaped-q8_0swiglu-qualification--2026-10-03).
- Native weighted/unweighted RMSNorm and the normalized residual FFN block passed
  the frozen six-program qualification, including width-1024 norm and residual add
  in that composition. See the
  [exact normalized-block scope](#native-rmsnorm-and-normalized-residual-block--2026-10-03).
- Resident RoPE, incremental causal attention and the model-owned
  `DenseAttentionPlan`/KV lifecycle passed with the frozen ten-program library
  in 8,192 shader bytes, preserving the previous six payloads and entry offsets.
  Isolated RoPE and attention, then two three-position requests through the selected
  real layer-0 attention block, passed bounded live qualification. This includes
  execution across the first code-page boundary and resident KV reset/reuse.
  See the [exact attention scope](#native-rope-attention-and-resident-attention-block--2026-10-03).
- The current eleven-program library adds Q8_0 `embedding_row`, and
  `VocabularyPlan` shares row tiles between embedding lookup and the normalized
  vocabulary head. The packed-weight arena admits up to 1 GiB; public tensors now
  admit 4 GiB minus 4 KiB, while auxiliary storage remains capped at 64 MiB.
  The isolated row decoder and selected
  real-checkpoint vocabulary component passed their bounded live gates, including
  the 165,306,368-byte resident weight arena. See the
  [exact vocabulary scope](#native-row-decode-and-tied-vocabulary-qualification--2026-10-03).
- `DenseModelPlan` now composes the shared vocabulary and every dense
  attention/FFN layer into a private `ResidentDenseModel`. Aggregate limits precede
  preparation I/O, and global position publication waits for the complete head.
  All 91 model tests passed, including seven new complete-model tests. The
  staged native first-token and two three-token requests passed on the selected
  complete 28-layer checkpoint, with **642,682,880 HAG bytes** released after each
  gate. This composition does not change the eleven shader programs or backend
  transport. See the [complete-model result](#complete-resident-dense-model-qualification--2026-10-03-edt).

Native serving defaults to **CPU**. An explicit Linux `hag-native` build with a
configured `execution_backend: hag` now exposes the experimental bounded HAG path;
its descriptor reports `execution_backends: ["cpu", "hag"]`. A CPU-only config
continues to report `["cpu"]`, even when the feature is compiled. This describes
implemented selection. The bounded runtime and in-process HTTP/SSE gates below
passed; CPU remains the default and broader serving coverage is separate. The
Vulkan feature remains separate and is not pulled in by `hag`.
Vulkan/HIP/ROCm shader compilers are not required for a HAG build.

## Explicit experimental HAG serving

`NativeRuntime` now selects each configured model's `ExecutionBackend::Cpu` or
`ExecutionBackend::Hag`; omitted selection preserves CPU behavior. HAG requires
Linux, the `hag-native` Cargo feature, an explicit `/dev/dri/renderD<number>` and
positive HAG allocation budget. It accepts only `fit_policy: exact`,
`weight_mode: streaming` and zero host-cache/prefetch settings. Here streaming
means one-at-a-time preparation reads: all prepared weights remain GPU-resident,
with no `FullyResidentSource` host copy and no fallback to CPU or Vulkan.

The current serving implementation exposes `hag.context_capacity` from **1 through
16384 evaluated positions**, defaulting to **2048**, with 4096 vocabulary rows
per tile, and the qualified dense Qwen3 geometry: 28 layers, width 1024, FFN 3072,
16 query/8 KV heads of width 128, vocabulary 151936, Q8_0 matrices and F32 norms,
epsilon `1e-6` and RoPE base `1e6`. Different shapes/formats are refused explicitly.
Admission requires checked **`prompt_tokens + max_new_tokens - 1 <= context_capacity`**, because
the final selected output is not evaluated. Chat-template tokens count against
this bound, and the checkpoint context and memory budgets must also admit the
configured capacity. The [16k development example](../config.hag-16k.example.yaml)
forecasts **4,629,168,128 native bytes** and **5,417,203,904 combined host/UMA bytes**
with 128 MiB runtime headroom, under separate 4608-MiB native and 6-GiB combined
budgets. Bounded hardware-free validation passed; live C16384 execution remains unqualified.
Existing C2048 results retain their original source identities and scope.

The historical bounded C64 and real
TCP gates passed on October 4 with the prior recording layout. The subsequent
storage reduction below passed fresh C64 live gates; frozen proofs remain separate.
**Both C128 boundary and CLI/TCP gates passed** under the new gate contract below,
with clean final crash/driver closeout.

Loading remains lazy. Before GPU opening or payload reads, the runtime checks
source identity, every layer, logical costs, the page-rounded queue charge and a
conservative combined host/shared-memory forecast. The following figures describe
the historical C128 layout on the 64-bit host:

| Requirement | Bytes |
| --- | ---: |
| Packed weights | 633,233,408 |
| Resident F32 tensors | 33,793,536 |
| Persistent / peak private auxiliary | 14,842,196 / 14,842,200 |
| Page-rounded HAG allocation | 681,910,272 |
| Rounded recording/owner host metadata | 95,420,416 (91 MiB) |
| Combined forecast with this checkpoint and 128-MiB headroom | 1,556,061,376 |

The combined forecast also counts the full
639,447,744-byte checkpoint file for conservative file-cache/metadata pressure,
4,456,448 bytes of sequential staging, a 607,744-byte logits buffer, bounded token
storage and at least 128 MiB of configurable general runtime headroom. Retained
commands, descriptors and ownership storage have a separate checked metadata term.
It must fit both the configured process budget and current host/cgroup available
memory. The separate HAG budget admits actual GEM charges. These forecasts do not
substitute for an external process limit or measured pressure.

The runtime serializes requests and keeps `ResidentDenseModel` behind its execution
mutex. Control callbacks run before/after whole synchronous tokens and before
delivery, including prompt evaluation before any text exists. Each token has
already fenced every dispatch before cancellation can release request admission.
Healthy control/sink cancellation resets the private caches and permits reuse.
Device/resident failures, failed reset, or remaining HAG charge after model/device
cleanup latch **every HAG runtime in the process unhealthy until process restart**.
Explicit unload reports uncertain cleanup as a failure. Eviction cannot restore
HAG health or bypass quarantine; explicit CPU selection can still proceed.
CPU fallback is never substituted for a failed HAG request.

Request serialization/admission is **per `NativeRuntime` instance**. This gate and
the intended server setup use one instance. Concurrent instances/processes and
device sharing are unqualified; the process-wide health latch does not coordinate
admission across instances.

[`config.hag.example.yaml`](../config.hag.example.yaml) retains C2048 with a 2-GiB combined
budget, 128-MiB runtime headroom and separate 1152-MiB HAG budget. The explicit CLI
invocation is:

```sh
HAG_TRACE_SUBMISSION=0 cargo run --release --locked -j1 -p beholdd-cli --features hag-native -- \
  serve --config config.hag.example.yaml
```

Use bounded build and serving supervision; the historical dedicated serving gate uses a
2-GiB process cap, no swap, 1.5-GiB early stop, host-memory checks, unchanged 90°C
cutoff and fresh crash/driver capture. Merely copying the example or compiling
the feature does not launch a service or GPU work. The frozen C16 runtime and
in-process HTTP/SSE stages passed with the scoped evidence below. The C64 and real
TCP gates subsequently passed on October 4. Earlier full-model and serving results
retain their original scopes.

## Integer-list EOS parity and response timing — 2026-10-04

The fresh contained self-check, CPU reference, native answer and timing gates all
passed for **one original integer-list prompt** on the same C128 Qwen3-0.6B
checkpoint and executable. The prompt was
“List the integers from 1 to 8, separated by commas. No explanation. /no_think”,
with temperature zero and an explicit 80-token output limit. CPU and all five
native requests returned exactly
`<think>\n\n</think>\n\n1, 2, 3, 4, 5, 6, 7, 8.`
with EOS `stop` finish. CPU and native nonstream API usage matched exactly:
**29 prompt + 27 emitted tokens**. SSE returns no usage field; its counts come
from the exactly matched nonstream reference, and native audits independently
verify **56 evaluations** for each request. The 27 include thinking markup and
exclude EOS. EOS detection therefore uses `prompt + emitted tokens`, unlike a length-limited request's
`prompt + max_new_tokens - 1` bound. Native answer parity used 56 evaluations;
the separate four-request timing stage used 224. Every native request reported
healthy reset and no poison; final unload and idle CLI exit zero passed.

The freshly preregistered semantic parser permits one optional terminal period;
raw text, final text, usage and finish reason still require exact CPU/HAG parity.
`/no_think` is a soft prompt instruction, not a hard server template option.
Three earlier recall failures, the strict list-parser failure on a terminal
period, and the subsequent 90.375°C CPU thermal abort remain preserved and failed.
None ran native answer or timing work. **Recall and general answer quality remain
unqualified**; this result covers the single list fixture only.

The first nonstream request, including lazy model preparation, took
**12.439175904 seconds**. This excludes CLI startup and is not an OS-cold-cache measurement.
The same loaded model then served three SSE requests, resetting KV each time:

| Warm round | First nonempty content (s) | Request to DONE (s) | API completion tokens / elapsed second |
| --- | ---: | ---: | ---: |
| 1 | 5.943939849 | 11.821288454 | 2.284 |
| 2 | 5.752789912 | 11.587933377 | 2.330 |
| 3 | 5.859566045 | 11.620839078 | 2.323 |

First content includes thinking markup. Rates use the matched nonstream API's
27-token count and include prefill, EOS detection, transport and request overhead.
These are three descriptive client observations under a **75% CPU quota**, without
a speed threshold or controlled cold-cache baseline; they do not qualify decode
or kernel throughput, broader workloads, concurrency or operating-system integration transport.

| Measured stage | Cgroup peak (bytes) | Maximum Tctl |
| --- | ---: | ---: |
| CPU reference | 273,711,104 | 85.625°C |
| Native answer | 146,595,840 | 82.5°C |
| Native timing | 146,345,984 | 88.375°C |

No swap, memory/OOM events or matching driver faults occurred. Cgroup peaks are
not total shared-UMA usage; the **681,910,272-byte HAG charge** and
**1,556,061,376-byte admission forecast** remain unchanged. The 2-GiB process cap,
1.5-GiB early stop, 90°C cutoff, 64-task/no-new-privileges limits, 8/4-GiB host
headroom and 180/195/210-second deadlines also remained unchanged. The harness
passed 60 tests before execution. Minimum available host memory was
17,388,670,976 bytes. Final cleanup verified all four units inactive, all nine
recorded PIDs absent and no residual task processes.

Root watcher **3671362** covered **16:17:50.443940–16:19:29.490478 UTC**, stopped
cleanly with zero dumps and an empty directory. Independent complete CPU and GPU
kernel windows contained **157 / 544 lines**, with no matching faults. Closeout
passed all four gates and reverified **130 compiled sources, 13 gate inputs**,
checkpoint, dependencies and C128 ancestry. Frozen SHA-256 identities are:

- CLI: `b64d5c56027835f72641c799dd9b0ae387ac754de32c60411d7be9ae7d839f9f`.
- Source manifest: `fb7ed6bbbf1e07b5f298eb189b0fadbdbe09e25962960f11e7db204fede0a1a4`.
- This gate manifest: `a49a6d2c2703b66bc36a40550c764c3159028afddff2836c0b4160e4c7bb80e7`.

Evidence is in
`$EVIDENCE_DIR/work/hag-list-cooled-20261004/`.
The archive target is `outputs/hag-list-cooled-qualified-20261004.tar.gz`.
Earlier C128 numerical/lifecycle results retain their own scope. CPU remains the
current default; changing shipped backend defaults requires separate release qualification.

## C128 implementation and qualification — 2026-10-04

Current experimental serving source fixes **128 evaluated positions** for the same
28-layer dense Qwen3-0.6B Q8_0 checkpoint. **Both C128 boundary and CLI/TCP gates passed, with clean final closeout**.
The optimized C64 proof below retains its own frozen source, binaries, numerical
results and resource measurements. The eleven shader payloads and arithmetic,
per-dispatch fences, workspace reuse and exact-bit scalar deduplication are unchanged.
This milestone does not add a C256 candidate or qualify longer contexts by inference.

`execution_limits.max_evaluated_tokens` now reports 128. Admission checks
`prompt_tokens + max_new_tokens - 1 <= 128`, including rendered chat framing.
The final selected output token is emitted without another model evaluation.
Omitted output limits resolve to `min(default_max_new_tokens, 128 - prompt_tokens + 1)`
after tokenization; explicit limits remain intact or return typed HTTP 400 before
JSON or SSE generation. CPU defaults and cancellation/reset semantics are unchanged.

| C128 requirement | Bytes |
| --- | ---: |
| Packed weights | 633,233,408 |
| Resident F32 tensors | 33,793,536 |
| Persistent / peak private auxiliary | 14,842,196 / 14,842,200 |
| Page-rounded HAG charge | 681,910,272 |
| Native recording metadata | 94,711,519 |
| Rounded model/recording host allowance | 95,420,416 (91 MiB) |
| Combined runtime forecast | 1,556,061,376 |
| Conservative boundary-probe forecast | 1,604,527,519 |
| Preparation host writes | 683,966,292 |

These checked costs include 3,614 retained batches, 455,405 dispatch descriptors,
234 packed owners, 452 public tensor owners, 14,451 auxiliary owners and 75,647
retained allocation references. The 3,584 layer-position batches remain below the
model's 4,096 bound. Attention's aligned score/probability workspace needs at most
1,540 bytes here, covered by each batch's 4096-byte workspace. The same 768-MiB
HAG budget and 2-GiB process cap apply; forecasts include conservative file-cache/UMA
pressure and do not substitute for measured cgroup or driver observations.

The preregistered boundary fixture runs **128 CPU reference evaluations before
GPU opening**, followed by **129 GPU evaluations**: fixed token IDs 1..128, refusal
of another position without dispatch/state changes, then reset and replay token 1.
Full logits are compared at positions **0, 31, 63, 64, 127 and replay 0**, totaling
**911,616 values / 3,646,464 readback bytes**. The expected transport counts are
**489,297 physical dispatches / 7,482 logical submissions** and 516 bytes of
per-token recording writes, with no submit-time tensor transfers. Finite tolerance
remains `2e-3 + 2e-4 * abs(reference)` with exact CPU greedy agreement; any mismatch
fails the gate without relaxing that contract. CPU source accounting is 44,544 reads
and 81,087,569,920 cumulative bytes, streamed through a 4,456,448-byte maximum
payload staging buffer. CPU KV is dropped before GPU opening; five retained
reference vectors occupy 3,038,720 bytes.

The separate actual CLI/TCP gate requires that boundary result from the same
frozen inputs and root watcher session. A **127-token rendered prompt** must resolve
an omitted output limit to two; a nine-token Hello prompt with explicit output
limit **121** must return HTTP 400 before headers, both cold and warm. Six successful
requests plus one real socket cancellation retain JSON/SSE parity, healthy reset,
reuse, busy-unload refusal, unload/reload and clean CPU/HAG SIGTERM exit checks.
Expected successful evaluations total 190; cancellation adds 9..23, under a
**214-evaluation ceiling**. The bounded two-token fixtures do not establish
complete-answer quality or throughput.

The new live gates preregister **75% CPU**, reduced from the historical C64 gate's
100% quota to limit heat during the doubled CPU oracle work. Memory remains 2 GiB,
swap is disabled, early stop remains 1.5 GiB, and the 90°C Tctl cutoff is unchanged.
The supervisor also verifies 64 tasks, no new privileges, at least 8 GiB available
host memory at start and a 4-GiB stop threshold, with 180/195/210-second
child/service/supervisor deadlines. New root crash capture and complete driver/OOM
closeout are required; earlier watcher sessions and results are not reused.

Offline validation passed: 43 runtime tests (one ignored), 92 model tests (two
ignored), nine probe tests and 49 supervisor tests, plus strict Clippy and the
harmless containment self-check. The fresh manifest covers 130 source inputs.
Frozen SHA-256 identities are:

- Probe: `cb400fb4ddcfca0c60afbf8843d1c29c8dd84ebd7fb1a2ed268217cc28ec5726`.
- CLI: `b64d5c56027835f72641c799dd9b0ae387ac754de32c60411d7be9ae7d839f9f`.
- Source manifest: `fb7ed6bbbf1e07b5f298eb189b0fadbdbe09e25962960f11e7db204fede0a1a4`.

Fresh evidence is in `work/hag-c128-20261004/`, with archive
`outputs/hag-c128-qualified-20261004.tar.gz`. The measured results and complete
crash/driver closeout follow.

### Boundary result

The C128 boundary gate passed at **12:54:21–12:55:53 UTC**. It made **129 GPU
evaluations**, **489,297 physical dispatches** and **7,482 logical submissions**,
checking **911,616 full-logit values / 3,646,464 readback bytes** at the preregistered
positions and replay. Maximum absolute error was **`0.00012683868408203125`** under
the unchanged finite tolerance; sign and zero/nonzero mismatch counts were both
zero. All six CPU greedy choices matched: **`[14582, 58, 293, 7221, 109, 14582]`**.
Refusal of another position preserved state/counters, and reset/replay passed.
The **681,910,272-byte HAG charge retired to zero**; measured auxiliary peak and
preparation host writes matched **14,842,200 / 683,966,292 bytes**.

The boundary process peaked at **102,744,064 cgroup bytes**, maximum Tctl was
**75.75°C**, and minimum available host memory was **21,982,720,000 bytes**.
Actual containment reported `cpu.max: 75000 100000`, the 2-GiB memory cap, zero swap,
64 tasks and no new privileges. No memory/OOM events, matching driver faults or
thermal abort occurred. These observations do not establish throughput or cold-cache
fit.

### TCP result and final closeout

The actual CLI/TCP gate passed at **12:56:02–12:57:41 UTC**, completing **200
HAG evaluations** across six successful requests and one real socket cancellation.
Audit counts were **128, 10, 22, 10, 10 (cancelled), 10, 10**; every request reported
healthy reset and no poison. Four invalid/overflow requests were refused before
generation. The 127-token prompt resolved its omitted output limit to two;
JSON/SSE parity, busy-unload HTTP 409, cancellation/reuse, unload/reload, final unload
and both idle CLI SIGTERM exits with code zero passed.

All three independent CPU protocol references emitted only two tokens, **`<think>\n`**,
with length finish; prompt lengths were 9, 21 and 127. This qualifies the scoped
numerical and protocol/lifecycle fixtures, not complete-answer quality, throughput,
arbitrary prompts/models, concurrency, shared-device use or a default-backend change.
CPU remains the default.

| C128 measurement | Boundary | CLI/TCP |
| --- | ---: | ---: |
| Process cgroup peak | 102,744,064 bytes | 305,504,256 bytes |
| Maximum Tctl | 75.75°C | 81.5°C |
| Minimum available host memory | 21,982,720,000 bytes | 21,298,528,256 bytes |
| Swap, memory/OOM events, matching driver faults | 0 | 0 |

Both gates verified the preregistered **75% CPU quota** and unchanged memory,
thermal and time limits. Process observations are separate from HAG/GEM charge and
conservative forecasts; they are not controlled performance measurements.
Root watcher **2496224** covered **12:53:48.369770–12:57:50.486262 UTC**, stopped
cleanly with **zero dumps** and an empty directory. The complete **1,327-line kernel
window** had no matching GPU/OOM/fault messages. Final closeout reverified all
**130 source inputs**, **12 gate inputs**, both binaries/dependencies and checkpoint
identity with no failures. Both gate reports and final evidence check are qualified.
Evidence is in `work/hag-c128-20261004/`; the archive is
`outputs/hag-c128-qualified-20261004.tar.gz`.

## Recording and scratch storage reduction — 2026-10-04

The optimized C64 recording implementation shares a private scratch workspace across the serial operations
of one immutable batch, deduplicates scalar parameters by exact F32 bits (including
signed-zero distinctions), and retains each allocation owner once per batch.
Dispatch descriptors no longer each carry duplicate owner references. Every physical
dispatch still fences before the next one; scratch is fully overwritten before
reading, and attention's score/probability regions remain disjoint and aligned.
There is no cross-batch scratch sharing. The eleven shader payloads, arithmetic,
fixtures, tolerances and **C64 serving capacity** were unchanged in that frozen proof.
**Both fresh boundary and real CLI/TCP gates passed at 12:15–12:17 UTC on October 4.**

For this model, one attention or FFN batch uses 4096 workspace bytes and three
four-byte scalar parameters. That workspace covers the `12 * visible + 4`
attention bound through C128. Head norm requires 4096+4 bytes and a transient
lookup adds 4, giving `28 * (capacity + 1) * 4108 + 4104` peak auxiliary bytes.

| C64 cost | Prior frozen layout | Optimized frozen C64 layout |
| --- | ---: | ---: |
| Private auxiliary peak | 37,833,560 | 7,480,664 |
| Page-rounded HAG allocation | 689,303,552 | 658,952,192 |
| Native recording host metadata | 69,778,787 | 47,540,703 |
| Rounded recording/model allowance | 70,254,592 (67 MiB) | 48,234,496 (46 MiB) |
| Combined runtime forecast | 1,538,288,320 | 1,485,916,864 |
| Conservative boundary-probe forecast | 1,572,308,003 | 1,519,718,559 |

Packed weights 633,233,408 and public tensors 18,195,968 bytes remain unchanged.
Persistent auxiliary bytes are 7,480,660. Peak retained counts are 1,822 batches,
227,821 dispatch descriptors, 234 packed owners, 452 tensor owners, 7,283 auxiliary
owners and 38,015 retained allocation references. Host estimates include spare
vector capacity and owner bookkeeping, separately from the 128-MiB runtime margin.
These checked figures do not substitute for measured process/driver pressure.

At this frozen C64 stage, the metadata-only C128 candidate reported 33,793,536 tensor bytes, 14,842,200 auxiliary
bytes, **681,910,272 HAG bytes**, 94,711,519 native metadata bytes and a
**1,604,527,519-byte conservative probe forecast**. It fits the unchanged 768-MiB
HAG and 2-GiB process budgets on paper. Its report explicitly keeps
`execution: not_requested`, `qualified: false` and `serving_capacity: 64`;
that estimate alone did not qualify C128 native execution or serving.

Forty-eight hardware-free supervisor tests and a harmless containment self-check
passed. The new frozen boundary/TCP gates reuse the same C64 numerical and short
protocol fixtures under unchanged process/thermal limits and a fresh root watcher.
Their artifacts are in `work/hag-recording-20261004/`, separate from the historical
C64 result below. New SHA-256 identities:

- Probe: `622feef0ecdc96e2b87fce2965be55de4ffd7e640dd93bbee4dfd0405a94c8b5`.
- CLI: `474d193f69950895dc67004347372263b29711c9136200bfc406ce716e0a72c6`.
- Manifest of 130 source inputs: `dfab6d90d1a4a64a5a4e9adedf7291fd2850e006002cf8eeb74d9027fb76bf9d`.

The checkpoint identity remains unchanged. The earlier C64/C16/C8 numerical and
lifecycle proofs keep their frozen scopes. The changed workspace has its own live
result below; no throughput, answer-quality, concurrency or broader-context
qualification follows from a smaller forecast.

### Fresh optimized-layout result

The boundary evaluated **65 positions / 246,545 physical dispatches / 3,770 logical
submissions**. Six readbacks checked **911,616 logits / 3,646,464 bytes** at the same
selected positions 0, 15, 16, 31, 63 and replay 0. Maximum absolute error remained
**`9.1552734375e-5`**, with one sign mismatch and zero zero/nonzero mismatches
reported under unchanged finite tolerance `2e-3 + 2e-4 * abs(reference)`. All six
CPU greedy IDs matched: `[14582, 17, 18, 58, 293, 14582]`. Refusal of a 65th visible
position left counters/state unchanged, reset/replay passed, auxiliary peak was
**7,480,664 bytes**, preparation writes **660,089,684 bytes**, and the
**658,952,192-byte HAG charge retired to zero**.

TCP again completed six requests and one cancelled request, **136 evaluations**
with audit counts 64, 10, 22, 10, 10 (cancelled), 10, 10. Every audit reported healthy reset
and no poison. Four invalid/overflow requests were refused without generation.
JSON/SSE parity, omitted output planning, real socket cancellation/reuse, busy
unload 409, unload/reload and both CLI exits 0 passed. Each of the three CPU
references emitted two tokens, `<think>\n`, with length finish. Complete-answer
quality and throughput remain unqualified.

| New measurement | Boundary | CLI/TCP |
| --- | ---: | ---: |
| Process cgroup peak | 79,228,928 bytes | 279,310,336 bytes |
| Maximum Tctl | 82.875°C | 57.375°C |
| Minimum available host memory | 22,608,797,696 bytes | 22,081,458,176 bytes |
| Swap, memory/OOM events, matching driver faults | 0 | 0 |

The 90°C cutoff, 2-GiB process cap, zero swap and 1.5-GiB early stop were unchanged.
These process observations are separate from GEM charge and admission forecasts;
comparisons with the earlier run are not controlled performance measurements or
cold-cache guarantees. Root watcher 2224024 covered 12:14:33.521823–12:17:24.117205 UTC,
stopped cleanly with zero dumps and an empty directory, and the complete 777-line
kernel window had no matching GPU/OOM/fault messages. Closeout reverified 130 source
inputs, 12 gate inputs, both binaries/dependencies and checkpoint identity with no
failures. Evidence is in `work/hag-recording-20261004/`, with archive
`outputs/hag-recording-qualified-20261004.tar.gz`. That archive keeps its C128
candidate metadata-only and its frozen serving capacity at C64. The separate C128
gate above records the current implementation status.

## Context-aware C64 and TCP qualification — 2026-10-04

`ExecutionLimits.max_evaluated_tokens` reports `Some(64)` for HAG and `None` for
CPU without opening a checkpoint or GPU; `/v1/models` exposes the field under
`execution_limits` as 64 or `null`. `plan_request` and its controlled variant
tokenize the rendered HAG prompt, then resolve omission to
`min(default_max_new_tokens, 64 - prompt_tokens + 1)`. Explicit output limits are
preserved or refused. CPU retains the caller's fallback (512 for HTTP).

Planning may read tokenizer metadata but does not read weight payloads, publish
residency, allocate GPU resources or reset caches. Both HTTP paths plan before
headers and return typed HTTP 400 for empty/invalid requests or capacity overflow;
overflow uses `code: context_length_exceeded`, `param: max_tokens`. Server admission
stays owned by the blocking worker through planning and generation, including when
the client disconnects. The embedded API exposes the same planning helpers.

The CLI handles SIGINT/SIGTERM with graceful server shutdown and waits for admitted
blocking work to finish. It neither forcibly interrupts an in-flight GPU operation
nor guarantees bounded shutdown for active work. Optional `HAG_SERVING_AUDIT=1`
records request sequence, evaluated positions, outcome, reset and poison status;
it contains no prompts, tokens, logits or tensor payloads.

Both stages passed on October 4 at 06:05–06:07 UTC (02:05–02:07 EDT). The boundary
gate evaluated token IDs 1..64, checked full logits and exact CPU greedy selection
at positions 0, 15, 16, 31 and 63, refused the next position without dispatch/state
change, then reset and checked replayed position 0. Its **65 evaluations** used
**246,545 physical dispatches / 3,770 logical submissions**. Six readbacks checked
**911,616 values / 3,646,464 bytes** with maximum absolute error
**`9.1552734375e-5`**, under unchanged finite tolerance
**`2e-3 + 2e-4 * abs(reference)`**. All six CPU greedy choices matched:
`[14582, 17, 18, 58, 293, 14582]`. One near-zero sign mismatch and zero zero/nonzero
mismatches were reported. The **689,303,552-byte HAG charge returned to zero**.

The separate real CLI/TCP gate completed three CPU references and exited that
process before HAG. Six successful HAG requests and one actual socket cancellation
made **136 evaluations**: success counts 64, 10, 22, 10, 10, 10, cancellation count
10. Four malformed/invalid/overflow requests were refused without generation;
explicit overflow returned HTTP 400 before headers. The 63-token prompt's omitted
output count resolved to two, and the multi-message prompt encoded to 21 tokens.
JSON/SSE text and usage matched CPU, socket cancellation reported healthy reset
before reuse, busy unload returned 409, unload/reload passed, and both CLI processes
exited zero after SIGTERM. All three CPU reference fixtures emitted exactly two
tokens, **`<think>\n`**, with length finish. This qualifies their protocol/lifecycle
behavior, not complete-answer quality or throughput.

| Measurement | Boundary | CLI/TCP |
| --- | ---: | ---: |
| Process cgroup peak | 89,931,776 bytes | 280,555,520 bytes |
| Maximum Tctl | 77.625°C | 59.625°C |
| Minimum available host memory | 21,627,920,384 bytes | 21,909,753,856 bytes |
| Swap, memory/OOM events, matching driver faults | 0 | 0 |

The process peaks are separate from GEM charge and are not cold-cache guarantees.
The boundary oracle forecast was 1,572,308,003 bytes including its CPU references
and KV; runtime admission separately uses 1,538,288,320 bytes. Both stages kept the
2-GiB process cap, zero swap, 1.5-GiB early stop, 90°C cutoff and root crash watcher
PID 152617, armed at 06:01:27.698676 UTC. Capture checks passed. Numerical thresholds,
launch limits, eleven shader payloads and transport remained unchanged between gates.
The watcher stopped cleanly at **06:07:46.876581 UTC**, with zero dumps and an empty
directory. The complete **1,674-line kernel window** had no matching GPU/OOM/fault
messages. Final closeout verified **130 source inputs**, **12 gate inputs**, both
binaries, dependencies and checkpoint identity, with no failures.

Frozen SHA-256 identities:

- Probe: `a9bb5d453fedbdcd465a7f9b068dfddc954071288d964201442e23bec7d9e1db`.
- CLI: `e7e3103b2f569019bf3721b107dfb3287688e5a47a6984f8893ca35c66d29c6b`.
- Manifest of 130 source inputs: `bafd8b404febaa95cb827885254fc7889898c8744345acfa263b1799b9fd89b1`.
- Checkpoint: `e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.

Evidence lives in the development chat's `work/hag-context-20261004/`; the archive
is `outputs/hag-context-qualified-20261004.tar.gz`. Prior frozen qualifications
remain unchanged. Arbitrary prompts/models, longer contexts, concurrent instances/
processes, shared devices, performance and operating-system integration transport remain separate gates.

## Experimental HAG runtime and HTTP/SSE qualification — 2026-10-03 EDT

Both dedicated serving stages passed on the allowlisted GC 11.5.1 host using the
same frozen test binary: runtime at **23:40 EDT on October 3, 2026**
(**03:40 UTC on October 4**), followed by HTTP/SSE at **23:41 EDT**
(**03:41 UTC**). Both reported `qualified=true`. They used one `NativeRuntime`
instance at a time, the fixed sixteen-position scope, the unchanged eleven shader
programs and the existing dense Qwen3-0.6B-Q8_0 checkpoint.

The runtime stage evaluated **nine prompt tokens and one decode token**, exercising
absolute positions **0..9**, and generated **`[151667, 198]`**. CPU token IDs, decoded
text and usage matched exactly. A deliberately insufficient metadata budget refused
before opening a missing device, leaving the model unloaded. Early control
cancellation also left it unloaded. Successful generation then unloaded cleanly.

The HTTP stage called the real Axum router **in process**, including consumption of
the real SSE response body. Four successful one-token requests matched the CPU's
**`[151667]`**, text and usage, including nonstreaming, complete SSE, reuse after
cancellation and reload. The gate observed a first-text SSE disconnect and stopped
that request, then reused the healthy resident model successfully. It also checked
capacity refusal before loading, busy unload returning **409**, idle unload,
idempotent unload, reload and final unloaded state. The diagnostic
`max_evaluated_positions: 46` is an aggregate work bound across those HTTP requests,
not a 46-position context; no request exceeded the fixed capacity.

Each load's checked HAG charge was **649,277,440 bytes (619.19921875 MiB)**.
The prepared resource checks, ordered model/device destruction, process-wide
health latch and unload checks enforced **zero remaining HAG charge after unload**;
reuse and reload completed without bypassing those checks. The 1,428,007,232-byte
combined host/shared-UMA forecast remained within the configured 2-GiB budget.

| Resource evidence | Runtime stage | HTTP/SSE stage |
| --- | ---: | ---: |
| Process/cgroup peak bytes | 245,776,384 | 245,161,984 |
| Minimum host available bytes | 21,536,968,704 | 21,996,560,384 |
| Maximum sampled Tctl | 59.5°C | 58.375°C |
| Swap peak / memory or OOM events | 0 / 0 | 0 / 0 |
| Matching driver fault messages | 0 | 0 |

Process peaks are separate from HAG's charged GPU allocation. Both stages verified
the 2-GiB process cap, no swap, 1.5-GiB early stop, one-CPU/64-task bounds,
`HAG_TRACE_SUBMISSION=0`, host-memory admission and unchanged 90°C cutoff. The
qualification wrappers passed **22 tests** before live execution. Root capture
**3539802** was armed at **2026-10-04 03:40:09.450761 UTC** and stopped cleanly at
**03:41:47.996004 UTC**. It produced zero dumps and an empty dump directory; the
complete **458-line kernel window** had no matching GPU/OOM/fault messages.

Frozen test binary SHA-256:
`87f07bc0d3865859c3f531d2be5251f4f303a1673d20c8410bc1f718b99e2261`.
Frozen CLI SHA-256:
`580c61a788d4ec3789c071d52c564450ebcc53307c8677f30c8b512a7284b856`.
Closeout verified all **128 frozen source inputs** and dependency identities; the
source-manifest SHA-256 is
`72504aebb87f30ccf9b7f3ed572802dda5c8fd329c369de670f5b45ba6c8622d`.
The rehashed 639,447,744-byte checkpoint remained
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.
Results, thermal samples and final capture audit are in
`work/hag-serving-20261003/`; the qualification archive is
`outputs/hag-serving-qualified-20261003.tar.gz`.

Explicit configured HAG serving now passes this bounded token/text, HTTP/SSE body,
cancellation/reuse and unload/reload scope. This gate did **not** open a TCP socket,
measure serving throughput, exercise all sixteen positions, qualify simultaneous
runtime instances/processes or device sharing, or change the CPU default. Broader
contexts, checkpoint families, network-service lifecycle and performance remain
separate gates. Earlier dated component/full-model records retain their scopes.

## Current resident adapter contract

`hag::native_execution::NativeExecutionDevice` supports BF16 and complete-block
GGML Q8_0 linear operations with 1..4096 rows and 1..4096 columns, plus SwiGLU and
scaled residual accumulation over at most 4096 elements and RMSNorm over
1..4096 elements. These operations use 1..64 wave64 workgroups, with
`ceil(rows / 64)` or `ceil(elements / 64)` groups and masked tail lanes. Matvec
retains serial F32 multiply/add within each row. A scaled linear uses two physical submissions
(matvec, then add); overwrite linear and nonempty SwiGLU each use one. Empty
SwiGLU is a retained semantic no-op. Linear input/output overlap and overlapping
SwiGLU gate/up views are refused, as are unsupported operations, nonfinite
scale/clamp parameters and negative clamp limits. No implicit CPU fallback exists.

RMSNorm requires equal input/output lengths, an equally sized optional resident
F32 weight, and finite nonnegative epsilon. The kernel sums F64 squares in input
order, uses approximate F64 reciprocal/reciprocal-square-root, then rounds the
inverse to F32 before input scaling. Unweighted normalization uses one dispatch
and a four-byte immutable epsilon parameter. Weighted normalization additionally
needs `4 * len` bytes of the batch's shared private workspace, then applies the resident weight with a
second F32 multiply dispatch. This preserves the intended arithmetic order while
requiring declared numerical tolerance for the approximate reciprocal operations.
Input and weight values are read at execution time, including preceding-command
writes. Output overlap with either operand is refused before lowering allocates
anything; read-only input/weight overlap is allowed.

RoPE accepts an even coordinate count from 2 through 4096, equally sized resident
cosine/sine vectors of half that length, both interleaved and split-half layouts,
and forward/inverse direction. Empty operations remain retained no-ops. A nonempty
operation uses one dispatch with `ceil(pairs / 64)` groups and separate F32
multiply/add instructions. All three views must be disjoint, including cosine
versus sine. Coefficients are read at execution time; no private scratch, parameter
upload or host trigonometry occurs during submission.

`IncrementalCausalAttention` owns query, output and complete position-major K/V
views plus a copied configuration. Native admission requires 1..256 coordinates
per head, 1..32 query/KV heads with integral GQA grouping, KV width at most 4096,
1..256 query positions and cache positions, positive finite scale, and checked
`query_start + query_count <= cache_positions`. Exact shapes, owners and view
extents are validated before allocation. Output must be disjoint from Q/K/V, and
Q/K overlap is refused; read-only Q/V and K/V overlap is allowed. Every query/head
uses three dispatches: ordered F64 QK scores, stable scaled F32 softmax, and ordered
F64 probability/value accumulation narrowed once to F32. Approximate F32 exp2 and
reciprocal require numerical tolerance. Physical K/V descriptors stop at each
query's last causal position, so future allocated rows are not read.

Attention needs a four-byte immutable scale and a workspace of
`4 + 12 * (query_start + query_count)` bytes: F64 scores with four bytes reserved
for eight-byte alignment and disjoint F32 probabilities. Each query/head reuses
that scratch sequentially. The workspace also serves other serial commands in the
same batch; identical scalar bit patterns share parameter storage. Submit/wait
consumes resident operands, including earlier writes, without tensor transfers.

`CommandList::embedding_row(output, weight, row)` retains one owned prepared weight
and destination, validates `row < weight.rows()` and `output.len() == weight.cols()`,
and leaves the command list unchanged on refusal. Native lowering accepts only
Q8_0 with complete 32-element blocks and width at most 4096. It binds the exact
selected packed-row span, including legal two-byte-aligned row offsets, then
performs one program-10 dispatch. The host resolves the global token to a tile
and local row; a large token ID never enters a bounded GPU scalar argument.
The batch retains one four-byte private dummy descriptor and no decoded matrix.
CPU execution decodes only the selected row. Unsupported formats/owners/extents
are refused explicitly; replay does not reread the checkpoint or transfer tensors.

The immutable program IDs are BF16/Q8_0/SwiGLU/scaled-add 0..3, unweighted
RMSNorm/multiply 4..5, RoPE 6, attention scores/softmax/values 7..9, and Q8_0 row
decode 10. The new 456-byte padded entry starts at **6,912**, ending the eleven
aligned programs at **7,368 bytes**; the shader allocation remains **8,192 bytes**.
The prior ten payloads and entry offsets remain byte-identical. The earlier
six-program qualification used a 4,096-byte shader allocation; current queue
charges are 4,096 bytes greater for otherwise identical arena capacities. HAG's
bounded library admits at most 16 programs and 16 KiB of page-rounded code.
The per-dispatch transport ABI remains unchanged. The isolated RoPE gate exercised
execution across the first code-page boundary; the attention entries also passed
the isolated and real-layer fixtures recorded below. Broader admitted geometry
and arbitrary inputs remain outside this qualification.

`NativeExecutionConfig` independently sizes the packed-weight, public F32 tensor
and private auxiliary arenas. Packed weights accept 1 byte through **1 GiB**,
while public tensors accept up to **4 GiB minus 4 KiB**, matching HAG's
`MAX_SLOT_BYTES`. Auxiliary storage retains a **64-MiB** adapter cap.
Defaults remain 64 KiB of weights, 64 KiB of tensors and
4 KiB of auxiliary storage. Larger allocations require explicit capacity and
budget admission; an admitted upper limit does not establish process-wide fit.
The auxiliary arena retains immutable parameters and one scratch workspace per
batch allocated during recording. Parameters are deduplicated by exact F32 bits;
workspace sizing takes the maximum command requirement, preserving attention's
disjoint aligned score/probability regions. Each batch retains distinct allocation
owners once, separately from its compact physical dispatch descriptors. Four-byte-aligned logical reservations are reclaimed only after
every owning handle, batch and submission releases them.

`NativeExecutionConfig::allocation_bytes()` reports the page-rounded queue charge,
including both guard pages around each arena and the shader/IB pages, without
device access. It is a cost query, not a reservation or a process-wide memory
bound. The shared `MemoryBudget` atomically admits each actual GEM allocation;
logical arena reclamation does not release the queue's preallocated hardware
charge. HAG initialization uses one fixed 4-KiB staging page. Checkpoint staging,
other host allocations and driver-private memory remain outside that GPU charge.

Clones serialize complete synchronous batches; each physical dispatch has its
own fence. Numerical intermediates remain in GTT, and submission performs no
adapter tensor transfers. Payload tracing is refused. Uncertain completion or
cleanup poisons the queue and retains hardware ownership; this does not provide
asynchronous retirement, cancellation isolation or device-loss recovery. Larger
admission bounds do not establish numerical, full-model or serving qualification.
See the [current shader ABI](../crates/beholdd-backend/src/hag/kernels/README.md)
and the [HAG qualification records](../vendor/exec-hag/docs/native-compute.md).

Current live gates use verified process/cgroup limits, zero swap, a 90°C Tctl
software cutoff, read-only crash capture and post-run driver checks. Development
builds/tests run sequentially under a 2-GiB memory cap with zero swap. Earlier
thermal-cutoff exceptions documented below belong to those historical runs.

## Model-owned resident attention

`QwenModel::dense_attention_plan(layer)` derives four exact Q8_0 matrices and
three F32 norm vectors from the strict dense-Qwen3 binder. Dimensions, epsilon,
RoPE base and encoded spans are checked before reading payloads. `memory(capacity)`
checks a positive capacity no greater than 256 or the checkpoint context;
`prepare` checks source identity and all logical packed/tensor/staging limits
before device allocation or source I/O. It stages one matrix or norm at a time,
then owns static operands, shared scratch, fixed K/V storage and coefficient tables.

Consuming `record(device, hidden)` binds one caller hidden vector and prebuilds
one immutable batch per cache position. Each step records input RMSNorm, Q/K/V
projection, per-head weighted Q/K RMSNorm, absolute-position split-half RoPE,
causal GQA and output projection with residual addition. V projection and K
normalization write only the uncommitted cache tail; attention sees its exact
prefix. `step` marks uncertainty before submit and advances `visible_positions`
only after the complete batch, including the residual output, has successfully
waited. A submit/wait error or unwind leaves the cache poisoned without publishing
that position; future steps and reset are refused. A healthy `reset` sets visible count
to zero with no device work. The next request overwrites every cache row before
publishing it. Batches, KV and scratch are private; the cache is not cloneable.

For the selected layer at capacity eight, persistent model tensors total 107,520
bytes and caller hidden adds 4,096, giving a **111,616-byte public tensor arena**.
Four packed matrices total **6,684,672 bytes**; three norms add 5,120 source bytes.
Seven native preparation reads therefore total **6,689,792 bytes**, with at most
2,228,224 bytes of payload staging. Eight retained position batches use **165,168
auxiliary bytes** within an explicit 256-KiB arena. The current cost query predicts
**7,098,368 charged HAG bytes**, checked against a fixed 32-MiB budget before device
access. These logical/HAG counts exclude metadata, CPU oracle staging and driver
memory; verified external process limits remain required.

The attention result still covers one component. The separate tiled vocabulary
implementation below adds qualified lookup and normalized-head component execution.
The complete resident schedule below also passed its selected checkpoint/sequence
fit and logit/token gates. Explicit experimental runtime/HTTP serving subsequently
passed its separate bounded gate above; broader contexts remain unqualified and
CPU is still the default.

## Shared tiled vocabulary contract

`QwenModel::vocabulary_plan(tile_rows)` requires exact dense Qwen3 binding, identical
embedding/output locators, a Q8_0 tied matrix and an F32 final norm. It accepts
1..4096 tile rows and a complete-Q8_0-block hidden width through 4096, with at most
4096 metadata tiles. Shape, encoded spans, finite nonnegative epsilon and checked
ranges are resolved without payload reads. `prepare` checks checkpoint identity,
backend format and packed/tensor/staging limits before reading or allocating.

For Qwen3-0.6B, 38 immutable row tiles cover the tied 151,936-by-1024 matrix:
37 full 4,096-row tiles of 4,456,448 bytes and one 384-row tail of 417,792 bytes.
Each is read and uploaded once, then the same prepared handles serve both semantic
uses. `record_embedding` validates the token before command construction and
retains only its selected tile/output. `record_logits` records final weighted
RMSNorm plus 38 overwrite linears into disjoint logit views: 39 semantic commands
and 40 dispatches. Prepared norm and normalized scratch are shared; callers wait
before changing boundaries or reusing batches. No entire decoded vocabulary or
`FullyResidentSource` checkpoint cache is created.

The `hag_vocabulary_probe` defaults to strict metadata/cost inspection for hidden
width 1024 and vocabulary 151,936. Its explicit native mode requires the isolated
row-decoder gate first, `HAG_TRACE_SUBMISSION=0`, verified process/thermal/crash
supervision and these preregistered checks:

- Six tokens `[0, 4095, 4096, 65536, 151551, 151935]` test tile boundaries, a token
  beyond transport scalar bounds, and the last full/tail rows. All 6,144 embedding
  values must match independent CPU dequantization bitwise, including signed zero.
- Three changed synthetic hidden inputs each produce all 151,936 final-normalized
  logits: 455,808 head values. Every value must be finite and satisfy
  `3e-4 + 3e-5 * abs(reference)`, and greedy selection must match the CPU exactly,
  choosing the first index on ties and refusing nonfinite logits. Sign/zero
  classification differences are reported, not independently rejected; this
  criterion is declared before live results.
- Total requirements are nine submissions, 126 dispatches, 461,952 checked values
  and 18 guards. Only one lookup batch lives at a time, and all lookup dummies
  retire before the head batch. Submit/wait must transfer zero tensor bytes;
  explicit output/guard readback is 1,847,880 bytes, and cleanup must release all
  HAG charge. Recording separately writes 4,124 auxiliary bytes across the six
  lookups and one head batch.

Expected component storage is **165,306,368 packed bytes**, **620,048 public tensor
bytes** including output guards, and **4,100 live auxiliary bytes** in an 8-KiB
arena. The current allocation query predicts **165,974,016 HAG bytes** under an
explicit **192-MiB HAG budget**. Payload staging is capped at 8 MiB; the largest
actual tile is 4,456,448 bytes. Native preparation must make 39 reads totaling
165,310,464 bytes (tiles plus norm), with no rereads/reuploads during the probes.
The independent one-worker CPU oracle runs before GPU access, making 123 bounded
reads and retaining 1,847,808 reference-output bytes, less than 2 MiB. Its decoded
tile scratch, metadata, file-cache and driver memory remain outside the HAG charge;
the external 1-GiB process/no-swap cap is separate.

The selected fixtures passed the separate vocabulary gate below. Synthetic
hidden-to-head parity does not establish full-model logits, generated tokens,
throughput or serving support. The complete-model owner below shares those tiles
with all 28 resident layers and passed a separate bounded whole-model gate.

## Complete resident dense-model contract

`QwenModel::dense_model_plan(capacity, tile_rows)` derives the complete dense Qwen3
graph from metadata, composing every `DenseAttentionPlan`, `DenseFfnBlockPlan` and
one tied `VocabularyPlan`. It requires Q8_0 matrices throughout, F32 norms, complete
Q8_0 hidden/FFN widths at most 4096, 1..128 layers and at most 4096 layer-position
batches. Capacity must be 1..256 within context; component head/tile limits still
apply. All layers are validated before payload reads, including unsupported formats
or malformed shapes in the last layer. Checked aggregate `DenseModelMemory` counts
packed weights, persistent tensors including private hidden/logits, and the largest
one-at-a-time staging payload.

`prepare(device, source, DenseModelLimits)` checks checkpoint identity, all three
aggregate limits and backend Q8_0 support before its first source read or device
allocation. Backend command support and physical capacity can still refuse during
preparation. It stages weights sequentially, owns one private attention cache and
FFN batch per layer (28 each here), the shared vocabulary and one head batch, and never constructs a
full decoded checkpoint or `FullyResidentSource` cache. The resulting
`ResidentDenseModel` is not cloneable and exposes no hidden, logits, cache or batch
handles. Plan getters expose `dim`, `vocab`, `layers`, `capacity` and `memory`.

`step(device, token: u32)` validates token, capacity and every cache state before
recording/submission. It marks the model uncertain before recording the transient
embedding batch, executes and retires that lookup, runs every attention/FFN pair,
then waits for the full normalized head. Global `visible_positions` advances only
after every operation succeeds and all child caches agree. Any execution error or
unwind permanently poisons the whole owner, including when earlier layer caches
already advanced; further steps, reads and reset then refuse. Invalid token/full
capacity refusals leave a healthy owner reusable.

`read_logits(device, output)` requires an exact vocabulary-length buffer and a
completed step. It refuses stale outputs before the first step or after reset;
device read errors/unwinds poison the model, while shape refusals leave it healthy.
`reset()` checks every cache before changing any count, then sets all positions to
zero without allocation or device I/O. The next request overwrites each cache row
before making it visible. `capacity()` and `is_poisoned()` expose lifecycle state.

For the selected 28-layer fixture at capacity eight, the checked requirements are:

| Resource | Bytes |
| --- | ---: |
| Unique packed matrices, including shared embedding/head | 633,233,408 |
| Persistent F32 tensors, including private hidden/logits | 4,548,096 |
| Persistent auxiliary payload | 4,858,516 |
| Auxiliary peak, including one transient lookup dummy | 4,858,520 |
| Largest sequential preparation payload | 4,456,448 |
| Page-rounded HAG allocation, predicted and measured | 642,682,880 |

Preparation reads 347 exact ranges totaling 633,495,552 bytes once. The owner keeps
253 persistent batches (224 attention positions, 28 FFNs and one head); each step
adds one transient lookup. A token executes 1,692 semantic commands in **58 logical
submissions and 3,793 physical dispatches**. Recording the lookup writes four private
bytes; submit/wait must transfer zero tensor bytes. All intermediate activations
and KV stay resident; explicit full-logit readback is 607,744 bytes per token.

`hag_model_probe` defaults to strict metadata/cost inspection without device access.
Its first native stage, `--first-token`, prepares the complete model and checks one
token before the same frozen binary may run `--execute`: two requests of three
serial tokens, using prompts `[1, 2]` and `[4, 5]`, with each third token selected by
the independent CPU greedy result after the second position. The immediate CPU
oracle streams one packed matrix/tile at a time, uses one worker and independent
per-layer KV, and computes every reference before opening the GPU. It retains
607,744 reference bytes for the first stage or 3,646,464 for six steps.

The preregistered complete-model criterion is finite logits with per-value error
at most **`2e-3 + 2e-4 * abs(reference)`** and exact CPU greedy selection, preserving
first-index ties and nonfinite rejection. Sign/zero classification differences are
reported separately. The six-step gate requires 22,758 dispatches, 348 submissions,
911,616 checked logits, zero submit/wait payload transfers, stable residency and
source counts, reset to zero visible positions, and zero HAG charge after cleanup.
Its CPU oracle makes 2,088 bounded reads; native preparation remains at 347.

Live admission requires `HAG_TRACE_SUBMISSION=0`, a **768-MiB HAG budget**, an
independently verified **1-GiB process cap with zero swap**, and an **896-MiB process
early-stop threshold**. Host available memory must start at least 8 GiB; the
supervisor stops below 4 GiB or at the unchanged 90°C Tctl cutoff. Fresh crash
capture and driver-window checks remain required. GPU charge is separate from
host staging, file cache, recorded metadata, CPU references and driver memory.

The model library's **91 tests passed**, including seven small two-layer GGUF tests
for complete immediate-CPU parity, repeated requests, source reuse, aggregate
limits, final-layer metadata refusal, late FFN/head errors and head unwind,
read-failure poisoning, and transactional reset. The native first-token and six-step
gates passed with the unchanged preregistered criteria; the dated record below
qualifies this selected complete graph and bounded sequence scope. Serving integration
was a later gate and passed in the [separate C16 serving scope](#experimental-hag-runtime-and-httpsse-qualification--2026-10-03-edt).
Larger contexts and general numerical/performance coverage remain open.

## Complete resident dense-model qualification — 2026-10-03 EDT

The complete 28-layer Qwen3-0.6B-Q8_0 graph passed both staged gates on the
allowlisted GC 11.5.1 host at **21:19 EDT on October 3, 2026**, corresponding to
**01:19 UTC on October 4, 2026**. The first-token gate ran at 21:19:19–23 EDT;
the repeated-request gate followed at 21:19:29–36 EDT using the same frozen binary.
Both reported `qualified=true`. The eleven native shader programs were unchanged
by the complete-model composition.

The first stage prepared every layer, then processed token `1`: **3,793 physical
dispatches, 58 logical submissions and 151,936 checked logits**. Maximum absolute
error was **0.00001430511474609375**; greedy token **14582** matched the independent
CPU result, with zero sign/zero classification differences.

The full stage completed **22,758 dispatches, 348 submissions and 911,616 checked
logits** across two three-token requests. Inputs were `[1, 2, 1096]` and `[4, 5, 3]`;
each third token came from the CPU greedy result after the second position and was
also required to match the native result before use. All six greedy selections
matched exactly: **`[14582, 1096, 374, 15846, 3, 4]`**. Maximum absolute logit error
was **0.0000553131103515625**. All logits were finite and satisfied the unchanged
preregistered bound **`2e-3 + 2e-4 * abs(reference)`**. There were **two sign
classification differences and zero zero-classification differences**, reported
under the previously declared policy; no criterion was relaxed after execution.

Capacity eight was allocated, while only absolute positions **0..2 per request**
were exercised. Every token traversed embedding, all 28 attention/FFN pairs and
the complete normalized vocabulary head. KV and intermediate activations stayed
resident. Request reset/reuse passed, with the final global visible count zero.

Actual HAG charge matched the cost query at **642,682,880 bytes** in each stage,
within the 768-MiB HAG budget, and returned to **zero** after owner/device cleanup.
Packed payload stayed at 633,233,408 bytes, persistent F32 payload at 4,548,096,
and auxiliary peak at 4,858,520. Native preparation made **347 reads totaling
633,495,552 bytes once**, with no source rereads or weight reuploads during steps.
For the full stage, the independent immediate CPU oracle made **2,088 sequential
reads totaling 3,800,979,840 bytes before GPU opening**, retaining 3,646,464 reference
bytes. Only final logits were read back, also **3,646,464 bytes**; submit/wait
transferred **zero tensor bytes**. Initial host writes totaled 643,016,852 bytes,
including zero-allocation followed by supplied norm/coefficient values. Six dynamic
lookup recordings added 24 private bytes separately from submission traffic.

| Measured resource evidence | First token | Two three-token requests |
| --- | ---: | ---: |
| Process/cgroup peak bytes | 57,454,592 | 59,682,816 |
| Minimum host available bytes | 22,781,796,352 | 23,077,203,968 |
| Maximum sampled Tctl | 57.875°C | 55.875°C |
| Swap peak / memory or OOM events | 0 / 0 | 0 / 0 |
| Matching driver fault messages | 0 | 0 |

Process measurements are separate from the charged GPU allocation. Both launches
verified the 1-GiB process/no-swap limit, 896-MiB early-stop threshold, host-memory
admission and unchanged 90°C cutoff. Root crash watcher **2657219** covered both
gates from **2026-10-04 01:18:15.591430 UTC** through clean shutdown at
**01:19:56.638393 UTC**. Capture produced zero dumps and an empty dump directory;
the complete **508-line kernel window** had no matching GPU/OOM/fault messages.

Frozen probe SHA-256:
`88069c08d91288611d2a27750dee4c4aff859b815ef4da25b4fea976442c9298`.
All **126 frozen source files** and binary/dependency identities were verified at
closeout; the source-manifest SHA-256 is
`ef40bd05b88ce5be6ddd1dc9bd768040f1740a841a071eb72ed40ad0458c8e98`.
The 639,447,744-byte checkpoint was rehashed and remained
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.
Evidence is in `work/hag-full-model-20261003/`, including both result files,
thermal samples and `final-evidence-check.json`, with the qualification archive
at `outputs/hag-full-model-qualified-20261003.tar.gz`.

This clears the selected dense complete-graph gate: bounded full-logit parity,
exact greedy agreement, cached positions, request reset, residency and cleanup.
It does not qualify HTTP/SSE serving, cancellation, unload/reload, a default GPU
backend, throughput, every admitted context/shape or arbitrary models. Beholdd
runtime and serving integration were the next gates at this stage; serving was
then CPU-only. The later serving record above is separate evidence.

## Native row decode and tied vocabulary qualification — 2026-10-03

The frozen eleven-program library passed the isolated Q8_0 row-decoder gate at
**21:02:34 UTC**, followed by the selected Qwen3-0.6B tied-vocabulary component at
**21:02:46 UTC** on the allowlisted GC 11.5.1 host. Both gates reported
`qualified=true`; their numerical and resource criteria were declared before
execution.

The isolated gate completed **16 dispatches/submissions, 17,152 bitwise-exact
values and 32 guard checks**. It replayed selected first/middle/last rows at widths
32 and 96, including a row starting two bytes past four-byte alignment, plus both
rows of a width-4096 fixture. Maximum absolute error was zero. Its 90,112-byte
HAG charge returned to zero, and explicit verification readback was 68,736 bytes.

The real component shared 38 Q8_0 tiles between lookup and head. All six selected
tokens `[0, 4095, 4096, 65536, 151551, 151935]` passed **6,144 exact embedding
checks**, including signed zeros and guards. Three changed synthetic hidden inputs
then produced every one of the 151,936 final-normalized logits: **455,808 head
values**, for **461,952 total values and 18 guards**. Each head reused 39 semantic
commands/40 dispatches; the full probe completed **nine submissions and 126
dispatches**.

Maximum observed head absolute error was **zero**, with zero sign/zero
classification mismatches. Acceptance remains the preregistered finite per-value
bound `3e-4 + 3e-5 * abs(reference)` plus exact CPU greedy selection; this result
does not establish general bitwise head parity. All three greedy choices matched:
`[117270, 121057, 253]`. Those choices belong to independent synthetic head inputs,
not a generated token sequence or full-model activations.

Preparation read the 38 tiles and final norm exactly once: **39 native reads,
165,310,464 bytes**, retaining **165,306,368 packed bytes** with no rereads or
reuploads during lookup/head reuse. The independent one-worker CPU oracle ran
before device access, reading **123 ranges / 495,937,920 bytes** and retaining
1,847,808 reference-output bytes. Logical source staging remained bounded by the largest
4,456,448-byte packed tile. The GPU retained 620,048 public tensor bytes and
4,100 live head auxiliary bytes within the 8-KiB arena. Retired four-byte lookup
dummies did not overlap the head allocation. Recording's 4,124 private-storage
write bytes were accounted separately from boundary writes.

Actual HAG charge matched preflight at **165,974,016 bytes**, below the separate
192-MiB HAG budget, and cleanup released the complete charge. Submit/wait caused
**zero tensor transfers**; explicit embedding/logit/guard readback totaled
**1,847,880 bytes**. Resident counts remained fixed during replay. This charges
GPU resources separately from measured process memory.

| Resource evidence | Row diagnostic | Real vocabulary component |
| --- | ---: | ---: |
| Peak process/cgroup memory, bytes | 22,134,784 | 53,719,040 |
| Maximum sampled Tctl | 46.375°C | 48.5°C |
| Minimum host available memory, bytes | 24,703,868,928 | 24,664,236,032 |
| Swap, memory/OOM events, matching driver faults | 0 | 0 |
| Final charged HAG bytes | 0 | 0 |

Both launches verified `HAG_TRACE_SUBMISSION=0`, actual 1-GiB process memory,
zero swap, 64 tasks, 100%-CPU caps, no new privileges and the unchanged 90°C
thermal cutoff. Root read-only capture PID 1229789 was armed at
21:01:44.277240 UTC and stopped cleanly at 21:03:07.816456 UTC with zero dumps and
an empty capture directory. The full **469-line kernel window** contained no
matching GPU/OOM/fault messages. Final verification rechecked both binaries,
dependencies and all 124 frozen source files.

Frozen row diagnostic SHA-256:
`81a65a75df034931068cb62f0bea4186636919ec9b3f54b5be2a532b1894bdf2`.
Frozen vocabulary probe SHA-256:
`3877e0e016374f1e6bb3623df8b369eaffe2bb006838ed0f7d862a0988461c75`.
The 124-file source manifest SHA-256 is
`f66a1eb1a1295be62f1f59759b9dfd7f63a26ecf1f062282964ebc75902409b2`.
Checkpoint SHA-256 remains
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.
Reports and source identities are in `work/hag-vocabulary-20261003/` in the
development chat; the evidence bundle is
`outputs/hag-vocabulary-qualified-20261003.tar.gz`.

This qualifies the stated finite row fixtures, selected real embedding rows and
normalized vocabulary head with shared packed residency. It does not qualify
arbitrary rows/inputs, the full 1-GiB admission range, a complete 28-layer graph,
whole-model fit/logits/tokens, throughput or serving behavior. Earlier dated
records retain their exact frozen scopes. Native serving was CPU-only at this stage.

## Native RoPE, attention and resident attention block — 2026-10-03

All three preregistered gates passed under fresh capture on the allowlisted
GC 11.5.1 host. This qualifies the stated finite operator fixtures and one real
layer's attention/residual block; it does not qualify whole-model execution.
The older dated records below retain their frozen binary and allocation scopes.

| Gate (UTC start) | Observed checks | Maximum absolute error | Preregistered acceptance |
| --- | --- | --- | --- |
| RoPE, 20:19:42 | 13 dispatches/submissions, 1,562 values, 26 guards | 0 | Bitwise exact |
| Attention, 20:19:51 | 183 dispatches, 7 submissions, 3,218 values, 14 guards | `1.1920928955078125e-7` | `2e-5 + 2e-5 * abs(reference)` |
| Layer-0 attention block, 20:20:10 | 762 dispatches, 6 submissions, 6,144 final hidden values | `2.384185791015625e-7` | `3e-4 + 3e-5 * abs(reference)` |

RoPE first checked an analytic pair, then three changed-input/coefficient rounds
for both layouts and directions over 65 pairs, exercising the second workgroup
and its inactive tail. It also exercised the RoPE entry crossing the first shader
page boundary. The isolated attention gate first checked one analytic position,
then three changed rounds each of three-query prefill (head width 4) and a
two-query cached chunk starting at position 63 (head width 128). Both GQA fixtures
used four query/two KV heads and NaN-filled future K/V rows outside the causal
prefix. This qualifies those chunk shapes and future-key masking; it is separate
from the model block's serial single-position schedule.

The real `Qwen3-0.6B-Q8_0` layer-0 block used hidden width 1024, 16 query heads,
8 KV heads and head width 128. Two requests each evaluated **three consecutive
positions**, with changed synthetic boundary hidden values and real weights.
Cache reset reported zero visible positions before the second request. Capacity
eight describes allocated cache/batches; positions 3..7 were not executed. Each
step retained 54 semantic commands and 127 physical dispatches: input norm 2,
Q/K/V projections 3, 24 weighted head norms 48, 24 RoPE operations 24, 16 attention
heads 48, and output projection/residual 2. Norm width 128 and split-half absolute
positions are covered within this composition. Only final hidden was numerically
checked; intermediate Q/K/V, normalized values and KV remained resident.

Native preparation performed **seven source reads totaling 6,689,792 bytes**,
including 6,684,672 packed matrix bytes; reuse caused no rereads/reuploads. The
independent immediate one-worker CPU oracle performed 42 separate reads before
native construction. All six steps passed finite/sign/zero checks. Submit/wait
performed zero adapter tensor transfers, final verification read 24,576 bytes,
and resident allocation/live-byte counts stayed fixed. Actual HAG charge matched preflight at
**7,098,368 bytes**, including the 111,616-byte public tensor arena and 165,168
live auxiliary bytes within its 256-KiB arena. Cleanup returned the charge to zero.

| Resource evidence | RoPE | Attention | Real attention block |
| --- | ---: | ---: | ---: |
| Actual HAG charge, bytes | 307,200 | 307,200 | 7,098,368 |
| Final HAG charge, bytes | 0 | 0 | 0 |
| Peak process/cgroup memory, bytes | 21,790,720 | 22,134,784 | 50,274,304 |
| Maximum sampled Tctl | 65.375°C | 63.625°C | 55.75°C |
| Swap, memory/OOM events, matching driver faults | 0 | 0 | 0 |

All gates verified `HAG_TRACE_SUBMISSION=0`, actual 1-GiB memory, zero-swap,
64-task and 100%-CPU cgroup caps, no new privileges and the unchanged 90°C thermal
cutoff. Their isolated HAG budget was 1 MiB; the model gate used 32 MiB. One root
read-only crash watcher covered all three gates from 20:19:39.673509 through
20:20:28.695911 UTC, stopping cleanly with zero dumps and an empty capture directory.
The complete 213-line kernel window had no matching GPU/OOM/fault messages.

An earlier same-binary RoPE run had also passed exactly at 89.375°C; the host then
rose to 92.6°C under other workloads. That session held the attention/model gates
at the same 90°C cutoff. The complete results above came from the subsequent
cooler, freshly supervised session; the threshold and numerical bounds were not
relaxed.

Frozen diagnostic binary SHA-256:
`fb35bbd7fab3c433839e21e195cc2e3d35f40527f8f18d29c2f3e6aaee99b329`.
Frozen model binary SHA-256:
`4028dd08a32a75d39b8c14ba4e2185b2ab4c9b5ec95432799f4d72604d831eeb`.
Checkpoint SHA-256:
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.
Source hashes, reports and logs are retained under
`work/hag-rope-attention-resume-20261003-201915-382480/` in the development chat,
with archive `outputs/hag-rope-attention-qualified-20261003.tar.gz`.

These gates do not qualify arbitrary inputs, all admitted widths/positions,
other layers, embeddings/final logits, whole-model budgets or token parity,
throughput, serving integration or the extended BF16 payload. The real block used
synthetic boundary inputs, not full-model activations. Native serving was
CPU-only at this stage.

## Build and verify

```sh
cargo test -p beholdd-backend --features hag --locked
cargo check -p beholdd-backend --all-targets --features hag --locked
python3 tools/test_sync_hag.py
python3 tools/sync-hag.py --source /path/to/exec-hag --check
```

Read-only discovery (deliberately exits nonzero because general compute selection
through this discovery API remains unavailable; the explicit experimental serving
path above is separate):

```sh
cargo run -p beholdd-backend --features hag --example hag_preflight --locked
```

Ordinary tests do not access devices. No existing inference process, model default,
release manifest or Antimatter package is changed by this integration.

The existing Vulkan prototype also now refuses physical devices below its Vulkan
1.2 shader target. Submission/wait failure or a panic poisons the shared queue
before any caller can reuse its buffers or commands. If cleanup cannot establish
idle (success or device loss), it deliberately retains the affected GPU resources
and loader until process exit rather than freeing possibly active memory. This is
fail-closed quarantine, not automatic recovery; a fresh process is required to
reclaim a quarantined context. Hardware-free fault-injection tests cover these
decisions; they do not constitute live GPU qualification.

## Source ownership and reproducibility

`vendor/exec-hag` is an exact, checked source snapshot, not a fork. This keeps a
fresh Beholdd checkout buildable without a sibling directory or an unpublished
Cargo package. `provenance.json` records source revision, working-tree state and
per-file content hashes; a dirty snapshot is explicitly identified, not represented
as a published immutable revision.

Edit the canonical HAG repository, run its tests, then refresh and verify:

```sh
python3 tools/sync-hag.py --source /path/to/exec-hag
python3 tools/sync-hag.py --source /path/to/exec-hag --check
```

The sync tool only accepts its explicit source list, rejects unexpected Rust files
and unsafe destination entries, and preserves unrelated files rather than deleting
them. Before publishing a release, commit HAG and refresh the snapshot so the
provenance records the clean source commit. No source commit/push or package release
is implied by this development change.

## Remaining replacement gates

1. Broaden native memory, repeatability and failure qualification beyond the tested
   GC 11.5.1 device/fixtures; unknown ISA/device combinations remain refused.
2. Broaden composed resident-operation qualification through the bounded
   `ExecutionDevice` adapter beyond the selected dense fixture: capacity-eight
   full-model preparation exercised positions 0..2, and the later capacity-sixteen
   serving preparation/runtime gate exercised positions 0..9. The C64 gate reached
   position 63 with selected full-logit checks; arbitrary inputs, larger contexts and
   broader numerical coverage still require separate qualification.
3. Expand numerical/shape coverage and add model operators/MoE with CPU parity.
4. Extend the owned execution seam beyond synchronous per-operator fences, with
   concurrent-request budgeting and safe in-flight retirement. Avoid hidden expanded
   checkpoints or per-operator tensor transfers through the host.
5. Extend the passing bounded CLI/TCP lifecycle gate with answer-quality and
   performance measurements before changing defaults or removing
   the Vulkan prototype. Broaden context/model coverage separately.

Future operating-system integration requires its own memory/queue/isolation contract.
The current `std`-based Linux DRM transport is not a portable kernel ABI. Keep
model semantics above the kernel boundary and define the native transport with
an explicit ABI design process rather than leaking Linux FDs/ioctls into that contract.

Future architecture-specialized engines stay above HAG. The shared
[exact-inference contract](../vendor/exec-hag/docs/exact-inference.md) separates
bitwise, numerical, distributional and quality parity, with explicit memory/I/O
accounting and end-to-end latency gates. It makes no model-scale speed promises.

## Foundation verification — 2026-09-26

- Standalone HAG: 25 hardware-free tests plus the separately enabled C-header ABI
  check passed; strict Clippy, formatting and standalone Cargo package verification
  passed. The dependency-free build's 10 tests also passed.
- Beholdd workspace with HAG, Vulkan and memory fault injection: **352 passed,
  0 failed, 8 ignored**. The eight opt-in cases require live GPUs/checkpoints and
  were not run. Existing compiler deprecation/unused-dependency warnings remain.
- HAG-only all-target compilation passed with `GLSLC` pointing to a nonexistent
  compiler. Its dependency tree contains no `ash`/Vulkan path.
- Source-snapshot drift verification and all 24 vendoring-tool tests passed.
- Read-only preflight opened `/dev/dri/renderD128` and received AMDGPU device
  `0x1586`, family `150`, 40 compute units. Result: `MemoryOnly`, with an explicit
  native-compute refusal. It created no GPU allocation and submitted no GPU work.

Logs are under `$BEHOLDD_LOG_DIR/hag-foundation-workspace-tests.log`,
`$BEHOLDD_LOG_DIR/hag-only-check.log`,
`$BEHOLDD_LOG_DIR/hag-preflight.log` and
`$HAG_LOG_DIR/native-foundation-tests.log` on the development host.
Builds were isolated under `$BUILD_ROOT`; running inference services and
shipped releases were left untouched. This is foundation validation, not a
GPU-inference performance or numerical-parity claim.

### Fixed-probe extension — 2026-09-26

The source snapshot also includes the closed `native-probe` example and its
hardware-free lifecycle, packet and UAPI tests: 44 HAG tests and both Linux
C-header ABI checks passed, as did strict all-target Clippy and formatting.
The 17-file snapshot drift check and all 24 sync-tool tests passed. The complete
Beholdd workspace suite above was not rerun for this extension.
Read-only discovery matches the probe's narrow identity gate. Initial live
attempts were deferred because the host reached approximately 100°C under
unrelated workloads. No running service,
backend default or inference-capability advertisement was changed.

The follow-up attempt fixed the benchmark thermal supervisor's pre-launch gap:
hot or invalid sensor readings now prevent child creation. All 27 supervisor
tests passed, including process-tree cleanup regressions. The host briefly cooled
to 68°C for a harmless command check, then heated again; the guarded HAG rebuild
refused at 84.625°C before spawning Cargo. Actual execution cgroup limits were
verified with a harmless command. Those attempts did not launch the GPU proof;
see the vendored
[supervision recipe and logs](../vendor/exec-hag/docs/native-probe.md).

The operator subsequently authorized bypassing the software temperature cutoff.
The rebuilt probe then **passed one live native dispatch**: 16,384 output bytes
matched exactly, 8,192 guard bytes stayed intact, fence sequence was 1, and all
32,768 allocated bytes were released (`budget_after=0`). The child exited zero;
the limited unit reported success without timeout/OOM. Whole-probe wall time was
4,876 microseconds, including setup and cleanup, not a kernel benchmark. No GPU
fault/reset messages appeared in the checked kernel-journal window.

Firmware/kernel thermal protections and running services were unchanged. This is
one fixed GTT buffer-fill validation, not an `ExecutionDevice` or GPU inference
backend. Beholdd's advertised serving capabilities and defaults were unchanged
by this historical gate.

### Reusable queue and packed arithmetic prototype

The optional `hag-native` feature now includes a trusted-shader serial HAG queue
and original BF16/Q8_0 matrix-vector shaders. The queue retains all five buffers,
VA mappings, context and BO list across dispatches, uses explicit GPU cache sync,
and permanently poisons/quarantines after uncertain completion or guard damage.
Beholdd owns packed formats, static shader assets and independent CPU references;
HAG contains no model-specific operator semantics. Shader code is immutable.

The first correctness suite is bounded to 64 rows by 256 columns, one wave64
group, serial F32 multiply/add, and complete Q8_0 blocks. It validates bitwise
serial results separately from the optimized CPU oracle, input immutability,
inactive outputs, allocation guards and stable resident memory charge. It is
not a throughput kernel or a `Device`/`ExecutionDevice` implementation.

Hardware-free verification passed: 56 standalone HAG tests, both C-header ABI
checks, strict all-target HAG Clippy, 52 Beholdd backend tests with `hag-native`,
the explicit LLVM shader check, and all 24 source-sync tool tests. The 21-file
snapshot matches the canonical dirty source. Logs are under
`$HAG_LOG_DIR/native-compute-*.log` and
`$BEHOLDD_LOG_DIR/hag-packed-linear-*.log` on the development host.

**The initial live attempt failed before arithmetic qualification.** At 21:00:36 EDT,
AMDGPU reset and recovered `comp_1.1.1`; the outer runner exited 124 and no fixture
passed. The host-tested lifecycle correctly refused reuse and retained resources.
The first dump expired before capture. A subsequent supervised, single-fixture
diagnostic captured the complete dump and identified a CPF execute-permission
fault at the command buffer's exact VA (`0x11000`). The IB had been mapped R-only.
The targeted fix maps it RX, retaining shader RX and data RW. No shader arithmetic
or register values changed. All 58 current hardware-free HAG tests and strict
Clippy pass, including mapping-permission and diagnostic-quarantine regressions.
The corrected path first passed one supervised BF16 1x1 dispatch at 23:05 EDT: exact
scalar/CPU output, intact guards and inputs, successful fence/context checks,
73,728 allocated bytes fully released, and process exit 0. The checked driver-log
window contains no AMDGPU/DRM entries. The 8-ms service runtime includes setup,
tracing and cleanup, not a throughput measurement.

At 23:14 EDT the same binary passed the separately authorized **16-case,
32-dispatch BF16/Q8_0 suite**, including 604 bitwise-checked output rows and
changed-input queue reuse. Optimized-CPU error was zero for the finite fixtures.
All input/weight, inactive-output and allocation-guard checks passed, and memory
accounting returned to zero after both queues closed. Process exit was 0; the
81-ms service runtime includes tracing and CPU verification, not a benchmark.
No AMDGPU/DRM entries appeared in the checked post-run journal window; the
diagnostic watcher expired normally without producing a crash dump.
This is bounded small-kernel correctness on the tested GC 11.5.1 device, not
general numerical, performance, larger-matrix or full-model qualification.

See the vendored [queue contract and validation record](../vendor/exec-hag/docs/native-compute.md)
and the [shader ABI](../crates/beholdd-backend/src/hag/kernels/README.md).
The normal `hag` feature enabled only discovery/preflight, and runtime
capabilities were CPU-only at this stage. No service, model default, package or release changed.

### Resident execution adapter — 2026-09-27

`hag::native_execution::NativeExecutionDevice` implemented the existing owned
execution interface under the explicit `hag-native` feature. At this stage it
supported only BF16 and complete-block GGML Q8_0 `LinearMode::Overwrite`, with
1..64 rows and 1..256 columns. `ScaledAdd`, SwiGLU and overlapping input/output
views were refused at recording, before any output was changed. No CPU fallback
existed in this adapter.

One HAG queue held both immutable programs, a packed-weight arena and an F32 tensor
arena. At this stage `NativeExecutionConfig` set each capacity (1 byte through
1 MiB); the shared `MemoryBudget` included page rounding, guards, shader/IB and
the reserved third slot. Weight preparation aliases the original packed upload
without expansion.
Four-byte-aligned free ranges are reclaimed only when every owning handle/batch/
submission releases them, and never after uncertain completion. Logical arena
reclamation does not release the queue's preallocated hardware charge.

Cloned devices share an owner and serialize queue access. Immutable recorded
batches retain all referenced allocations. Submission executes commands in order,
waiting for one physical fence per operator; the completed token retains its batch
and repeated waits do not resubmit. Device-resident intermediates are rebound as
inputs without host staging. Unsupported/foreign resources are rejected, and
transport failures permanently poison the adapter. Underlying HAG uncertainty
retains the complete hardware ownership chain instead of freeing in-flight memory.

`stats()` separates actual allocation charge, live/peak aligned arena reservations,
packed uploads, explicit host transfers, physical dispatch attempts/completions and
completed logical submissions. Host-transfer counters exclude IB uploads and HAG
guard/padding checks. `HAG_TRACE_SUBMISSION=1` is refused because payload tracing
would read numerical intermediates back to the host.

The explicit `hag_execution_probe` records BF16 32x32 -> Q8_0 4x32 and reuses the
two-command batch for three changed inputs. Its independent exact-integer oracle
is checked against the CPU executor before native access. It retains 2,184 packed
bytes and 272 tensor bytes, checks no adapter host transfers during submission,
reads only four final F32 values per round, and requires budget usage to return to
zero after dropping all owners. This composed path passed its own bounded live
test below; the earlier 32-dispatch single-operator suite alone did not establish it.

Hardware-free checks (these do not launch the example):

```sh
cargo test -p beholdd-backend --features hag-native --locked
cargo test -p beholdd-backend --features hag-native --example hag_execution_probe --locked
```

This stage did not enable serving, change model defaults or restart services.
Parallel reductions, asynchronous batches, additional operators, VRAM placement
and full-model execution remained separate work at this stage.

Offline verification for this extension:

- Beholdd workspace with `hag-native`, Vulkan and memory fault injection:
  **376 passed, 0 failed, 9 ignored**. Live GPU/checkpoint cases remained ignored;
  the separately invoked LLVM source/embedded-byte check passed.
- The 14 resident-adapter mock tests cover all 16 packed fixtures, dependent chains,
  immutable recording, foreign/forged handles, partial overlap, arena exhaustion
  and reclamation, concurrent cloned devices, transfer errors, partial dispatch
  failure, panic, completion-token retention and quarantined budgets.
- The example's two independent CPU-oracle tests passed. The binary builds and
  `--help` is nonexecuting; a tracing-enabled invocation against a nonexistent
  device path was refused before device access, as required.
- HAG passes 65 offline tests, both explicit C-header checks, strict Clippy and
  formatting. Its portable no-feature build passes 10 tests. Both projects' API
  documentation builds pass the strict broken-link check.
- All 24 source-sync tests and the 21-file snapshot check pass. Modified Rust files
  pass formatting and whitespace checks. Beholdd's pre-existing workspace-wide
  formatting drift and neighboring lint warnings remain; no unrelated files were
  reformatted to hide those failures. The new adapter/example have no Clippy warnings.

Logs on the development host are under
`$BEHOLDD_LOG_DIR/hag-resident-*-20260927.log` and
`$HAG_LOG_DIR/native-resident-*-20260927.log`. The built proof is
`$BEHOLDD_BUILD_DIR/hag-foundation/debug/examples/hag_execution_probe`.
The live-tested binary was preserved before rebuilding subsequent extensions at
`$BEHOLDD_BUILD_DIR/hag_execution_probe-resident-20260927-1be60e82`;
its SHA-256 is
`1be60e82a618401af85cca21bf114351b126a95e60018f3f372c1ffea932819a`.

### First resident-chain live pass — 2026-09-27, 12:28 EDT

With a fresh user-armed root crash watcher, one supervised process passed all three
changed-input rounds: **6 dispatches, 12 bitwise-exact final values, 48 verification
readback bytes, and zero budget after cleanup**. Packed uploads stayed at 2,184
bytes; logical tensor reservations stayed at 272 bytes; the fixed hardware charge
was 45,056 bytes. The intermediate stayed GPU-addressable in GTT, not necessarily
VRAM, and neither numerical tensor readback nor host staging occurred during
submission. IB updates and guard/padding reads remain part of HAG's synchronous
transport and are excluded from the adapter's tensor-transfer counters.

`HAG_TRACE_SUBMISSION=0` preserved that condition. Each operator retained its own
physical submission/fence; the test did not fuse commands into one IB. All batch,
clone, repeated-wait and accounting assertions passed. Process exit was zero;
the bounded runner reported success, 14-ms service runtime, 12-ms CPU, 2.0-MiB
peak memory and no swap. These timings include the CPU oracle/setup/cleanup and
are not a throughput measurement. The requested limits remained 256 MiB, no swap,
32 tasks, 50%-CPU, 15-second service and 10-second child with no-new-privileges.

No AMDGPU/DRM messages appeared in the checked 12:28:17–12:30:00 EDT kernel-journal
window, and no GPU devcoredump was present during the post-run check. The root
watcher was still active; its eventual termination status is separate from the
successful proof. No inference service, model default or serving capability changed.
This run checked final results, not a complete readback of intermediate tensors or
unchanged input/weight bytes. It is a bounded composition proof, not full inference.

See the vendored [resident-chain validation record](../vendor/exec-hag/docs/native-compute.md#bounded-resident-chain-validation--1228-edt)
for qualification limits and evidence. The live run log is
`$BEHOLDD_LOG_DIR/hag-resident-live-20260927-1228.log` with SHA-256
`dc1a64d995d06b5be00c2f59a7e1ae1e387854e835384501dbbebb563c2ffbdc`.

### SwiGLU and residual extension — 2026-09-27

The next bounded operator slice adds in-place `SwiGlu` and BF16/Q8_0
`LinearMode::ScaledAdd`. These new programs are not covered by the earlier live
linear proofs. Serving was CPU-only during qualification of this extension.

This initial SwiGLU payload accepted up to 64 elements, required disjoint gate/up
regions, and treated an empty view as a retained no-op. A positive finite limit
clamps gate from above only and up in both directions; zero disables clamping. Residual accumulation
keeps the original matvec kernels intact: a matvec writes a private resident
scratch vector, then a separate multiply/add kernel applies
`output = output + scale * scratch` with no FMA. Scale must be finite. Linear
input/output overlap remains refused, including for scaled accumulation.

At this stage the third HAG data slot became a fixed 4-KiB auxiliary arena for
immutable parameters and private scratch. It occupied the same page-rounded
hardware charge as the previous one-byte reserved slot. Auxiliary live/peak reservations are
reported separately from packed weights and public tensors. Full-list validation
precedes recording-time allocations/uploads; failed capacity admission reclaims
completed auxiliary reservations without changing public tensors. Transport
failure still poisons the queue and follows HAG's uncertain-resource retention.

Recorded batch length counts semantic commands. Physical dispatch counters count
each actual native operation, so one scaled linear uses two dispatches/fences.
Parameters and scratch are allocated while recording, never staged between
operators during submission. Clones serialize complete batches, and reusable
batches retain their private scratch rather than sharing it with another batch.

The explicit `hag_ffn_probe` exercises gate/up BF16 projections, SwiGLU, and a Q8_0
down projection accumulated into a residual. Its f64 mathematical reference and
CPU executor checks are independent of the shader implementation. Unlike the
earlier dyadic linear fixture, nonlinear results require a declared numerical
tolerance; passing this gate must not be reported as bitwise, token-distribution,
model-quality or lossless full-inference parity. See the shared
[exact-inference contract](../vendor/exec-hag/docs/exact-inference.md).

The planned first live gate isolated the new SwiGLU program before the complete
feed-forward suite. Both invocations require explicit CLI execution flags and
bounded supervision with crash capture; ordinary tests never access a GPU.

The full probe records three four-command batches (unclamped positive scale,
clamped negative scale, and clamped zero scale), reusing each for two changed
inputs/residuals. It expects six logical submissions and 30 physical dispatches,
5,184 packed bytes, 512 public tensor bytes and 408 private auxiliary bytes.
It checks all 576 activated-gate/up/final values after completed submissions,
requiring 2,304 explicit verification-readback bytes; submit/wait must not change
the adapter's tensor-transfer counters. The acceptance bounds are
`2e-5 + 2e-5 * abs(f64_reference)` for the activation/up stage and
`3e-4 + 3e-5 * abs(f64_reference)` for final residuals. Nonfinite results,
wrong-sign nonzero values and mismatched signed zeros are rejected separately;
the finite zero-scale fixture must preserve the residual bitwise. These checks
are preregistered, not adjusted after seeing native output.

Offline verification completed before any execution of these new programs:

- Workspace with `hag-native`, Vulkan and memory fault injection:
  **389 passed, 0 failed, 9 ignored**. This includes 21 adapter mock tests for
  allocation rollback, immutable batches, concurrent private-scratch reuse,
  retention and partial-failure quarantine. Ignored live tests were not run.
- All five FFN CPU/oracle tests and both earlier linear-probe CPU tests pass.
  Arithmetic-model edge tests are not physical SFU qualification.
- LLVM 18 independently assembled and matched all four immutable instruction
  payloads byte-for-byte; original BF16/Q8_0 payloads remain unchanged.
- Scoped formatting, whitespace, strict documentation-link checks and targeted
  Clippy pass. Clippy retains explicit allowances for pre-existing deprecation
  and `chunks_exact_to_as_chunks` dependency warnings; unrelated workspace
  formatting drift and manifest warnings were not changed.
- Standalone HAG's 65 offline tests, strict Clippy and formatting pass; its
  transport code and ABI are unchanged. All 24 source-sync tests and the exact
  21-file snapshot check pass.
- Both explicit probe modes refuse tracing-enabled execution before opening a
  nonexistent device; `--help` and test targets never open a GPU.

The probe prepared on September 27 was
`$BEHOLDD_BUILD_DIR/hag-foundation/debug/examples/hag_ffn_probe`, SHA-256
`9ec3ce8b862543b8a3071b81328ca7b5cc8733caeb7cee75a75ef678b366dd52`.
Logs are under `$BEHOLDD_LOG_DIR/hag-ffn-*-20260927.log` and
`$HAG_LOG_DIR/native-ffn-*-20260927.log`; the checksum record is
`$BEHOLDD_LOG_DIR/hag-ffn-probe-20260927.sha256`.
At that point no new GPU dispatch had been performed for this extension. The
replacement-host result below subsequently qualified those one-workgroup
payloads, with fresh crash capture and the isolated gate preceding the full suite.

### Replacement-host SwiGLU/FFN qualification — 2026-10-03

The rebuilt frozen probe passed the isolated eight-value SwiGLU gate at 13:33 EDT,
then all three FFN cases across two changed-input rounds each at 13:34 EDT. The
full suite completed six submissions, 30 physical dispatches and 576 checked
stage values within the preregistered bounds above. Maximum final absolute error
was `1.3606040205615955e-6`; signed-zero/sign/finite checks and bitwise zero-scale
residual preservation passed. Submit/wait performed no adapter tensor transfers,
and final budget usage returned to zero.

Accounting matched 5,184 packed bytes, 512 public tensor bytes, 408 peak auxiliary
bytes and 49,152 charged resident bytes; explicit verification readback was 2,304
bytes. Both launches verified actual cgroup limits, used the thermal guard and
completed with no matching driver errors or crash dumps. The authenticated
read-only watcher was stopped after the post-run diagnostic window.

The binary SHA-256 is
`3a50158c547ad1c17a2e6b7222214713124789cbd8e963d6694d94953f32be07`.
It was built and tested before the concurrent Q6_K decoder edit; that unrelated
change is outside this proof. See the canonical
[HAG qualification record](../vendor/exec-hag/docs/native-compute.md#bounded-swiglu-and-resident-ffn-validation--2026-10-03)
for exact evidence, supervision and scope. This is finite-fixture tolerance
qualification with per-operator fences and GTT intermediates, not full-model
execution, arbitrary numerical coverage or a throughput result.

This proof advances the native execution backend. At this historical stage,
remaining operators and model parity still required separate implementation and
qualification before production defaults could change.

### Model-shaped Q8_0/SwiGLU qualification — 2026-10-03

The extended Q8_0 and SwiGLU payloads passed two bounded gates on the allowlisted
GC 11.5.1 device. At 14:23:42 EDT, `hag_dense_ffn_probe --diagnose-first` completed
one 65-by-32 Q8_0 operation across two workgroups. All 65 outputs matched the
ordered F32 reference bitwise, both adjacent output canaries were unchanged,
and all 45,056 charged bytes were released. This specifically exercises the
second workgroup and its inactive tail lanes.

At 14:23:57 EDT, the same frozen binary passed layer 0 of the selected
`Qwen3-0.6B-Q8_0.gguf` checkpoint: Q8_0 gate/up projections of 3072-by-1024,
SwiGLU over 3072 elements, and a Q8_0 down projection of 1024-by-3072. One recorded
four-command batch was reused for three changed deterministic inputs. Twelve
physical dispatches checked all 3,072 final output values against the immediate
single-worker CPU implementation. Maximum absolute error was
`1.1920928955078125e-7`, within the preregistered
`3e-4 + 3e-5 * abs(CPU_reference)` bound; finite/sign/signed-zero checks passed.
These are real packed weights with synthetic inputs, not captured activations or
a full model pass. Only final FFN outputs were numerically checked in this gate.

Native preparation read and uploaded the three packed matrices once: three source
reads totaling 10,027,008 bytes, versus nine source reads across the three CPU
oracle evaluations. Reuse did not reread or reupload the weights. Submission and
wait performed no adapter tensor transfers; 12,288 bytes of final outputs were
explicitly read back for verification. Intermediate tensors remained in GTT and
each operator retained its own physical fence. The expected and actual HAG charge
were both 10,125,312 bytes, and final budget usage returned to zero.

The launch verified actual cgroup limits of 1 GiB memory, zero swap, 64 tasks,
100% CPU and no new privileges, with a 60-second service limit, a 45-second child
limit and a 90°C Tctl software cutoff. The qualification cgroup peaked at
67,297,280 bytes; maximum sampled Tctl was 44.625°C. Memory-pressure/OOM counters
and swap remained zero. Host available memory stayed at or above 20,712,275,968 bytes,
above the 8-GiB launch and 4-GiB stop thresholds. Both gates exited successfully;
the inspected kernel windows contained no matching GPU/driver/OOM faults. Fresh
root read-only crash capture remained healthy through the post-run window and
stopped with zero dumps.

The frozen binary SHA-256 is
`b4fb5ee0472998fc4ddfe354cc72290b94d04caec384da2f8eac9198bba1f5bf`.
The checkpoint SHA-256 is
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.
Replacement-host logs are under
`$EVIDENCE_DIR/work/hag-model-scale-20261003/live-qualification/`;
the retained evidence archive is
`$EVIDENCE_DIR/outputs/hag-model-sized-qualified-20261003.tar.gz`.

This qualifies the extended Q8_0/SwiGLU payloads only for the stated diagnostic
and layer-0 fixtures. At that gate the extended BF16 and scaled-add payloads had
not been live-qualified, and the admitted 4096-by-4096/4096-element limits were not fully
qualified. Earlier small BF16/residual proofs remain evidence for their archived
bytes. Arbitrary floating-point inputs, additional layers/operators, full-model
logits/tokens, serving behavior and throughput remained separate gates at this stage.

### Native RMSNorm and normalized residual block — 2026-10-03

The current implementation lowers `DenseFfnBlockPlan`'s
`hidden += FFN(RMSNorm(hidden))` graph to seven physical submissions: two for
weighted norm, one each for gate/up/SwiGLU, and two for residual down projection.
All five semantic commands retain resident operands, with no CPU arithmetic
fallback or tensor traffic during submit/wait. Parameters, scratch and completion
tokens obey the adapter's existing ownership and poison contracts.

Offline validation passed 102 backend tests and the explicit LLVM comparison for
all six shaders under bounded development execution. Tests cover changed norm
weights, command dependencies, guarded widths through 4096, alias/owner refusal,
auxiliary exhaustion rollback and failures at either normalization stage.
The mock follows CPU F64 reduction semantics; it does not qualify physical SFU
accuracy. Earlier live proof scopes remain unchanged.

`hag_ffn_block_probe` validates metadata before reading weights or opening a device.
For the selected layer it predicts 10,129,408 charged bytes, with 10,027,008 packed
matrix bytes, 36,864 public tensor bytes and a 32-KiB auxiliary arena containing
8,204 live parameter/scratch bytes. Four native preparation reads total 10,031,104
bytes, including the F32 norm; maximum staging remains 3,342,336 bytes.

After desktop authentication armed fresh capture, both live gates passed at
15:21 EDT (19:21 UTC) with the frozen probe SHA-256
`7e191d5d149a4f8f51075c619c4f846581a5fe0f0da2166a78020ced465f227f` and unchanged
checkpoint SHA-256 `e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.

The diagnostic passed one unweighted element and three changed-input/weight
65-element weighted rounds: 196 values, eight output guards, four logical
submissions and seven dispatches. Maximum absolute error was zero under the
preregistered `2e-6 + 2e-6 * abs(reference)` bound. It charged 45,056 HAG bytes,
peaked at 21,487,616 process bytes and sampled at most 42.75°C Tctl.

The normalized layer-0 block passed three reused submissions and 21 dispatches,
checking 3,072 final hidden-state values under the unchanged
`3e-4 + 3e-5 * abs(reference)` bound. Maximum absolute error was
`2.384185791015625e-7`. Tested shapes were weighted RMSNorm 1024, Q8_0 gate/up
3072x1024, SwiGLU 3072, Q8_0 down 1024x3072 and residual add 1024. Only final block
values were numerically checked; normalization and residual-add coverage at 1024
is within that composition. Four native preparation reads versus 12 independent
CPU-oracle reads were recorded, with no rereads on reuse. Submit/wait caused zero
adapter tensor transfers; final verification read 12,288 bytes. Actual HAG charge
matched 10,129,408 bytes and cleanup returned the budget to zero.

The block process peaked at 66,781,184 bytes (63.6875 MiB), maximum sampled Tctl
was 54.75°C and minimum host available memory was 18,485,620,736 bytes. Both gates
had zero memory events/swap, no matching driver messages and `qualified=true`.
Actual 1-GiB memory, zero-swap, 64-task and 100%-CPU caps, no new privileges and
the 90°C thermal cutoff were verified. The watcher ran from 19:19:34.833450 to
19:21:58.408741 UTC and stopped with zero captures.

Exact evidence is recorded in the
[canonical record](../vendor/exec-hag/docs/native-compute.md#native-rmsnorm-and-normalized-residual-block--2026-10-03)
and the development chat's `outputs/hag-rmsnorm-qualified-20261003.tar.gz`.
These finite fixtures do not qualify all widths/inputs, the extended BF16
payload, per-head-width-128 norm, RoPE, attention/KV, embeddings, full-model
parity or throughput. Native serving was CPU-only at this stage.
