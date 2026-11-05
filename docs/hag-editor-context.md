# HAG context requirements for Antimatter

This is a historical qualification record. Its frozen Antimatter revisions are
absent from the clean public source history. The old client profiles cannot be
reproduced from a public checkout, and their results do not qualify a replacement
client revision. See [public source scope](hag-qualification.md#public-source-scope)
for the standalone profiles and requirements for new client qualification.

This record tracks Beholdd's experimental HAG execution with Antimatter as an
independent editor client. Native fixed-token model execution
qualifies 2048 evaluated positions. The frozen C512 serving path also qualifies
Antimatter's frozen no-tools grounded chat and stream, including reuse and
capacity refusals. This bounded host-provider result does not establish
full-editor/tool readiness. See [hag-qualification.md](hag-qualification.md)
for the repeatable profile and the final evidence below. Separate C2048 attention
fixtures and fixed-token full-model qualification also pass. Current source now
defaults to C2048 serving capacity, with larger explicit development capacities
described below. The additive project-instructions client profile
passed CPU/native chat and streaming qualification on validation-host on 2026-10-08.
The earlier shared-host thermal holds remain preserved below. This qualifies the
exact synthetic no-tools fixture, not arbitrary editor requests or all C2048 positions.
A separate C2048 real-TCP lifecycle profile subsequently passed cancellation,
healthy reuse and same-process reload on validation-host; its short Hello fixture and
direct HTTP transport retain a separate scope from the provider proof.
The subsequent seeded-history request failed its CPU prerequisite because the
EOS response contained an unclosed thinking block. Its native stages were not
started; see the [held attempt](#held-c2048-seeded-project-history-on-validation-host--2026-10-08).

## Bounded native 16k and production catalog admission — 2026-10-18

The [bounded C16384 GPU probes](hag-16k-probes.md#observed-bounded-native-qualification--2026-10-18)
passed: synthetic attention reaches prefixes 8192/16383/16384, and the model
allocates its full 16k capacity while comparing positions 0/1 and reset/replay.
Peak observed Tctl was 55°C, host available memory stayed above 13.48 GiB, and
there were no task OOM/swap events or captured GPU dumps. These diagnostics do
not qualify full-context model numerics or a long editor prompt.

Separately, Antimatter's actual production composer now passes a model-free
39-tool admission smoke: 21 bundled, 15 app and three connected-chat tools.
Its local synthetic peer receives exactly one C16384 request; a fresh C8192
control is refused before HTTP. All schemas, current user text and saved
workload preferences are preserved. Tools are listed through the real MCP SDK
with an in-memory peer; no tool handler, model or GPU inference executes in this
suite. Optional Bot, skills, extensions and long history remain outside scope.

Exact inspection uses the **actual 32,389 transmitted bytes**, not a rewritten
planning fixture. The pinned checkpoint tokenizer measures **8,079 prompt
tokens**, so default M4096 requires **12,174 positions**, leaving 4,210 at C16384.
The C8192 diagnostic retains the same catalog with default M2048 and requires
**10,126 positions**. This closes the production assembly/serialization gap in
the derived October 17 measurements; it does not establish native prefill or
tool-call execution for that request.

The full-catalog and original production smoke modes passed. All editor test
processes/cgroups retired, with zero memory events/swap. Evidence is joined in
`work/hag-production-catalog-20261018/validation.json`. Raw request SHA-256:
`0762094fb7f43a1d306224a51a865e61ec77b51831fae82c6ccc7b3d3bfa4353`;
exact measurement SHA-256:
`b9d41ac03529f8831c79d2812dd6aaf7b621bc0f77eb4a4af15d42d1c5009d1e`.
The [captured-prompt prefill probe](hag-catalog-prefill.md) freezes the exact
token vector and checks a bounded 32-token prefix before scaling to the complete
prompt or running a real full-catalog completion. Its separate CPU reference now
passes in 44.721 seconds with 59.30 MiB observed peak process memory and 45°C Tctl.
After an initial authentication timeout, a fresh authenticated continuation
passed native prefix parity and reset/replay: 455,808 compared logits, maximum
absolute error `3.91e-5`, exact greedy matches, and 11.293 seconds for execution
and checks after 8.516 seconds of preparation. Peak native process charge was
664.07 MiB, separately from 4,629,168,128 HAG allocation bytes; maximum observed
Tctl was 55.875°C. No memory events, swap or GPU dumps were recorded, and all
scopes and the watcher retired. The failed attempt is preserved alongside the
[successful continuation](hag-catalog-prefill.md#observed-native-continuation--2026-10-18).
The remaining 8,047 prompt tokens, full completion and tool execution are still
unqualified. Next is a bounded CPU equivalence check for the layer-major prefill
path before extending the reference to longer prefixes.

## Tool upper bounds and current catalog sizing — 2026-10-17

Beholdd now preserves and enforces JSON Schema `maximum` and `maxItems` in both
Qwen3 JSON and Qwen3-Coder XML tool paths. Bounds apply recursively through
properties, array items and additional properties. Integer/float comparisons
retain exact signed/unsigned boundaries; integral numeric spellings such as
`2.0` and `-0.0` are accepted for array limits. Invalid generated arguments fail
before chat choices or tool-call SSE chunks are delivered.

All **75 server tests** pass. The independent Transformers/Jinja oracle contains
**35 admitted and 5 rejected cases**, with all previous 29 admitted and 5 rejected
fixtures unchanged. Six new oracle fixtures cover nested upper bounds and numeric
spellings; protocol and HTTP tests cover enforcement and failure delivery.

Fresh derived planning fixtures use current production tool composition,
collaboration instructions, prompt budgeting and the pinned checkpoint tokenizer.
The 15 app definitions and empty-memory base remain frozen October 15 inputs.
The 36- and 39-tool requests are byte-identical to the prior fixtures; the optional
43-tool catalog adds four Bot tools and the actual Bot instructions. All schemas
are retained and admitted. These are metadata measurements, not editor dispatches.

| Catalog | Prompt tokens at M4096 | Positions at M4096 | Client position estimate | Positions at M512 |
| --- | ---: | ---: | ---: | ---: |
| Core, 36 tools | 7,690 | 11,785 | 13,685 | 8,199 |
| Interactive, 39 tools | 8,174 | 12,269 | 14,273 | 8,683 |
| Optional Bot, 43 tools | 9,012 | 13,107 | 15,392 | 9,521 |

All three fit **C16384**, including Antimatter's estimates and initial message
budgets. None fits C8192 even with M512. The M512 guidance makes each prompt two
tokens shorter. Skills, third-party extensions, fresh personal/project context
and longer history remain outside these fixtures. At this checkpoint native
C16384 execution and full editor/tool behavior remained unqualified; the later
bounded native checks above retain their narrower scope.

Evidence is in `work/hag-schema-20261017`. Catalog report SHA-256:
`dbaa6cea31a8c837ccce861714073641180cfab1a11684b3384588d2196a15f2`;
measurement SHA-256:
`3b8e7abf2001850d5adfeba60dc7ef2a047f489333a7be68da4e521e14a4c5d6`;
inspector SHA-256:
`f5224b810e8cea3285f06b3c1b3b8465c4f0d40cc61a05c25fda7479db83e9e3`.
All six bounded services, cgroups and recorded children retired. Peak contained
memory was **961,835,008 bytes**, peak Tctl **56.875°C**, and minimum available host
memory **20,317,188,096 bytes**. Recorded memory-pressure/OOM counters and task
swap were zero. No model, GPU, tool action, editor or network request ran.

## Configurable development capacity — 2026-10-15

`hag.context_capacity` now selects **1–16384 evaluated positions**, retaining
**2048** when omitted. The [16k example](../config.hag-16k.example.yaml) provides
a 6-GiB combined/process budget and 4608-MiB native allocation budget. For the
pinned Qwen3-0.6B checkpoint, its forecast with 128 MiB runtime headroom is
**5,417,203,904 combined host/UMA bytes**, including **4,629,168,128 native bytes**.
Both configured budgets and current host/cgroup headroom remain admission checks.

Bounded hardware-free validation passed: 75 standalone HAG tests, 157 Beholdd
backend tests, 101 model tests, 61 runtime tests, 68 server tests, 24 vendoring
checks and one explicit LLVM assembly check. Live C16384 numerical, lifecycle and
editor behavior remains unqualified; the earlier C2048 proofs retain their exact
frozen source and request scopes.

### October 15 planning requests

Fresh metadata-only inspection used the exact pinned checkpoint tokenizer and
current production bundled definitions, wire names, terminal description, system
prompt builder and client budgeting code. The 21 bundled MCP definitions were
combined with the 15 app definitions retained from the earlier production capture.
The extractor first reconstructed that capture's 15-tool system-prompt suffix
exactly, then rebuilt the suffix for the expanded catalog and reply reservation.
These are **derived planning fixtures, not editor-transmitted request bytes**.

| Capacity | Prompt tokens | Reply budget | Minimum evaluated positions | Fits exact limit |
| ---: | ---: | ---: | ---: | --- |
| 8,192 | 7,690 | 2,048 | 9,737 | No |
| 16,384 | 7,690 | 4,096 | 11,785 | Yes |
| 8,192 | 7,688 | 512 | 8,199 | No, exceeds by 7 |
| 16,384 | 7,688 | 512 | 8,199 | Yes |

The first two reply budgets follow Antimatter's quarter-window default; 512 is an
explicit planning override. The output-limit guidance changes the prompt by two
tokens between these variants. For C16384/M4096, Antimatter estimates 9,590 prompt
tokens and **13,685 positions** (`9590 + 4096 - 1`), so its provider check also
admits the request. Its message budget is 4,235 tokens against a 2,008-token
message estimate, with **zero initial fit overflow**. These estimates are separate
from the authoritative checkpoint token counts.

The 39-tool variant uses the production connected-chat wrapper with a metadata
stand-in for a valid active chat. All four variants were refused **before
tokenization** because `parameters.properties.limit` contains unsupported JSON
Schema `maximum`; those constraints were preserved. No exact 39-tool count is
claimed. The fixtures omit `read_skill` and skill-catalog instructions, Public
Bot chat (off by default), third-party MCP extensions and user-specific standing
context. Bot chat additionally requires `maxItems` support.

No model, GPU, tool, terminal or network execution was used for this sizing, and
no editor or chat store was started. Evidence is in `work/hag-capacity-20261015`,
with extraction in `outputs/extract-01/catalog` and exact measurements in
`outputs/measure-01/measurement.json`. The catalog report SHA-256 is
`a9d4e9b49c76b8e18ed72d5361fc7827b9cc6494211806cbece2ed8d6dd8f8f4`;
the inspector SHA-256 is
`0f64f08ea71b88dbcfaa72ac881fc292114064592789cd89c7367a38e29adaf2`.
This refusal motivated the subsequent schema support and fresh catalog sizing
reported above. Native qualification remains a separate gate.

All 13 bounded validation services, cgroups and recorded guard children retired.
Peak contained memory was **1,001,369,600 bytes**, peak Tctl **68.125°C**, and
host available memory stayed above **16,174,071,808 bytes**. Recorded memory
pressure/OOM counters and task swap were zero. The initial thermal hold, task-limit
failure, longer-context normalization check and stale LLVM assertion remain in
the evidence alongside their successful follow-ups. Existing CPU parity tests
explicitly request 1,024 workers, so their supervisor task limit was raised from
128 to 1,152 while retaining the 2-GiB memory cap and 25% CPU quota. Long-context
softmax checks now account for ordered F32 denominator error; the earlier
C2048 mass bound and all per-value CPU comparisons remain unchanged.

## Exact prompt measurements

**Historical renderer scope:** the October 4 counts below describe the renderer
used for those fixtures, not the current exact Qwen3 JSON tool renderer. The
initial October 15 production smoke refused tool prompts before tokenization;
the subsequent tool implementation and current measurements are recorded next.

### Exact Qwen3 tools and current coding prompt — 2026-10-15

Beholdd now renders function definitions and paired tool history for the pinned
4,905-byte Qwen3-0.6B template. Generated JSON calls use the parser selected during
request planning, through both chat and buffered tool SSE; Coder retains XML.
Tool order, null/omitted fields, Unicode, Python JSON separator/float formatting,
historical argument strings, reasoning and grouped results match the reference
contract. Duplicate keys, invalid associations, schemas and generated calls fail
closed. Integer literals outside i64/u64 are rejected rather than rounded.

All **68 server tests** passed, including **29 admitted and 5 rejected independent
Jinja fixtures**, JSON parser edge cases and mock chat/SSE integration. A fresh
metadata-only inspector built from the same source measured the pinned checkpoint:

| Request | Tools | Prompt tokens | Reply budget | Minimum evaluated positions |
| --- | ---: | ---: | ---: | ---: |
| Captured fresh no-tools request | 0 | 86 | 512 | 597 |
| Captured oversized composer diagnostic | 15 | 9,816 | 512 | 10,327 |
| Short-user derivative with production system/tools | 15 | 4,826 | 512 | 5,337 |

The derivative replaces only the oversized final user message with “Inspect the
workspace and summarize the next development step.” It was not transmitted by
the editor. The source capture is Antimatter `9c21d02`'s completed production smoke
with bundled MCP disabled: six preview, four agent, four memory and one PR tool.
This is a 15-tool subset, not the full configured coding catalog. The oversized
diagnostic remains a deliberate client refusal and is not a normal prompt baseline.

At this earlier measurement, native capacity remained **2048**. The short
derivative needs 5337 evaluated positions with the captured explicit reply budget.
The subsequent explicit capacity work and 36-tool sizing are described above.
Actual tool selection quality and execution remain
unqualified; these checks did not load model tensors or execute inference/tools.

Evidence is retained in `work/hag-qwen3-tools-20261015`: final `test-03`, `build-02`
and `measure-01`, with `validation.json` binding retirement and the prior smoke.
All eleven services/cgroups and recorded guard children retired. Peak contained
memory was **920,055,808 bytes**, peak Tctl **84.5°C**, and host available memory
stayed above **19,396,083,712 bytes**. Recorded pressure/OOM counters and swap were
zero; a separate `sock_throttled` counter reached 6. The initial runner path error
and inspector module-path build failure are preserved with their successful fixes.
No GPU workload ran; the October 13 native/UI qualification keeps its original pins.

### Historical October 4 fixtures

On 2026-10-04, static fixtures were extracted from committed Antimatter
`78147b78d8271b993dc3734f469f763df1f0f548`, using its real grounding and tool
declarations. They assume a fresh workspace at `/workspace`, empty memory, ten
seeded agents, a short user message, no conversation history, no external MCP
servers, no skill catalog and no personal profile. The project-instruction
variant adds a synthetic 2-KiB instruction body. No application, tool or model
inference was started to measure these fixtures.

Beholdd's actual message normalization/tool renderer and the Qwen3-0.6B Q8_0
checkpoint's tokenizer metadata produced these counts. Minimum positions are
`prompt + maximum reply - 1`; they do not include Antimatter's planning margin.

| Static fixture | Tools | Exact prompt tokens | Reply budget | Minimum positions |
| --- | ---: | ---: | ---: | ---: |
| Grounding | 0 | 93 | 128 | 220 |
| Grounding plus project instructions | 0 | 467 | 128 | 594 |
| Coding built-ins | 21 | 3,731 | 512 | 4,242 |
| Default coding tools | 36 | 7,977 | 512 | 8,488 |
| Default coding plus project instructions | 36 | 8,350 | 512 | 8,861 |

Antimatter's conservative character estimator and safety margin first admit
these fixtures at C512, C2048, C8192, C16384 and C16384 respectively. These are
planning targets, not qualified capacities. Even C8192 cannot hold the exact
default fixture with its reserved reply. History, real project instructions,
memory, skills and external tools can increase the requirement further.

The historical default tool set initially failed before tokenization because
`read_pull_request` declares numeric `minimum` constraints. Beholdd now admits
finite numeric minimums, renders them without stripping them, and enforces them
when translating generated arguments. Signed/unsigned integer comparisons stay
exact above 2^53; numeric bounds do not constrain string/null union members.
Other unsupported schema keywords continue to fail closed.

To measure a request without opening a runtime or reading model tensor payloads:

```sh
cargo run --offline --locked -j1 -p beholdd-server --example prompt_inspect -- \
  /path/to/Qwen3-0.6B-Q8_0.gguf /path/to/request.json
```

The example accepts multiple request files, caps each at 8 MiB, emits exact
counts or a rejection for each, and exits nonzero if any request is rejected.
Use explicit `max_tokens` for reproducible measurements. If omitted, the report
uses the uncapped server default, not HAG's capacity-dependent resolution.

The measurement evidence retains the original rejection, corrected counts,
five request fixtures, extraction scripts and 22 pinned source hashes under
`work/hag-editor-context-20261004` in the development task workspace. The model
is 639,447,744 bytes with SHA-256
`e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`.

## Capacity work sequence

The existing model records every attention position in advance. At C128 its
native recording metadata estimate is 94,711,519 bytes, and persistent auxiliary
storage is 14,842,196 bytes. Those costs grow with capacity. Increasing a single
constant would also encounter the 4096 layer-position recording guard, 256-token
attention shaders and a 64-MiB tensor arena.

The first change adds opt-in on-demand attention recording. Each layer retains
its prepared weights and cache, records only the current position, waits for its
fence and retires the batch before publishing that position. FFN and vocabulary
batches stay prepared. The existing serving path and all capacity limits remain
unchanged while this strategy is qualified separately.

At commit `4a7bd1aeae0b4e5eb63e29583527a79c579af122`, the C128 on-demand
metadata inspection reports:

| Resource | Bytes |
| --- | ---: |
| Resident tensor payload | 33,793,536 |
| Persistent auxiliary payload | 119,124 |
| Peak auxiliary payload | 123,232 |
| Page-rounded HAG allocation | 667,193,344 |
| Native recording metadata estimate | 396,210 |
| Additional retained attention model metadata | 64,736 |
| Conservative probe combined forecast | 1,495,560,018 |

Recording now writes 115,028 bytes of scratch/scalars per complete token,
including the embedding lookup's four bytes. Submission itself still transfers
no intermediate tensors. The probe checks those writes separately from weight
uploads and logit readback, and requires scratch to return to the persistent
baseline after every step. Failed recording, submission, wait or unwinding
poisons the affected state; a full-capacity refusal leaves healthy state intact.

The additive `--inspect-on-demand`, `--on-demand-first-token` and
`--on-demand-context` modes leave previous probes intact. The context gate uses
the existing independent CPU oracle and fixed token IDs 1 through 128, compares
full logits at positions 0, 31, 63, 64 and 127, refuses a further position, then
resets and replays token 1. Its tolerance remains `2e-3 + 2e-4 * abs(CPU)` with
finite logits and exactly matching greedy token IDs. GPU execution requires the
separate process, memory, thermal and crash-capture supervision.

Before live execution, 39 server tests, 27 recorded-model tests, 12 probe tests
and nine native-runtime regression tests passed. One real 30B tokenizer test
remained ignored because that unrelated checkpoint was not supplied. Clippy
passed using the existing `chunks_exact_to_as_chunks` lint exception. Builds
and tests ran offline with one Cargo job, a 2-GiB process cap, zero swap and a
90°C thermal cutoff. These checks alone do not establish native qualification.

The separate supervisor passed 27 mocked failure-path tests. It verifies frozen
inputs, actual cgroup limits, a live root crash watcher for native stages, current
and peak memory below the 1536-MiB early-stop threshold, service/cgroup retirement
and the exact preregistered report. The process cap remains 2 GiB with no swap,
75% CPU and a 90°C cutoff; initial Tctl must stay at or below 60°C for ten seconds.

The 2026-10-04 live preflight was **refused at 69.0°C**, before even launching the
harmless supervisor self-check child. No CPU model evaluation, native device
open, GPU dispatch or privileged watcher was started for this qualification.
That attempt left native parity and retirement qualification pending. The later
cool-start run below passed; serving still uses the previously qualified
pre-recorded C128 path until the larger-context serving work is qualified.

The frozen release binary SHA-256 is
`3e7e7998941c44a70579d6e7a8ab70c4d24349e4e433b176c8daa9fba67d2a55`.
Its 89 source inputs cover all 75 release depfile inputs plus Cargo metadata;
their manifest SHA-256 is
`7451948758ababc22893bc11fec7e0a85e54a1912a825cf2a1f27f2b08ccf9a1`.
The complete live-input manifest SHA-256 is
`d1802c0356b0940432270dce5a5853d70b302ed96bcc9f17a4d9c84fc116e5aa`.
Evidence and the pending supervisor are under
`work/hag-editor-context-20261004/native`; the refused preflight is retained in
the enclosing `self-check-start.log`. Resume with the same frozen inputs only
after a fresh cool startup check, then the self-check, live crash watcher,
first-token gate and full context gate in order. A failed live stage must retain
its evidence and cannot be silently retried or overwritten.

## C128 on-demand native qualification

The same frozen release passed on the next cool start, at 03:08–03:14 UTC on
2026-10-05 (2026-10-04 local time). The original 69°C refusal remains archived.
Self-check, first token and full context completed in order under the unchanged
memory, swap, CPU, temperature and crash-capture rules.

| Measurement | First token | Full context and replay |
| --- | ---: | ---: |
| Native evaluations | 1 | 129 |
| Logits compared | 151,936 | 911,616 |
| Exact greedy matches | 1 | 6 |
| Maximum absolute logit difference | 0.0000143051 | 0.000126839 |
| Physical dispatches | 3,793 | 489,297 |
| CPU reference time | 0.501 s | 64.845 s |
| Native preparation time | 0.239 s | 0.267 s |
| Native execution and checks | 0.198 s | 27.086 s |
| Maximum sampled Tctl | 50.75°C | 62.5°C |
| Process cgroup peak | 58,277,888 bytes | 86,687,744 bytes |

Process-cgroup memory is not total inference memory: HAG separately reserved
667,193,344 bytes, and admission used the 1,495,560,018-byte combined forecast.
Both runs returned auxiliary storage to 119,124 bytes after every token, peaked
at 123,232 bytes of auxiliary payload, and retired all HAG charges to zero.
All checked logits were finite with zero sign/zero mismatches. Full-capacity
refusal left state unchanged; reset and token-1 replay passed.

An independent closeout passed 175 checks. All memory-pressure/OOM counters and
swap remained zero, all three services/cgroups and ten recorded processes
retired, and the root watcher stopped with zero dumps. The complete kernel-log
interval through watcher shutdown contained no GPU/DRM/reset/OOM diagnostics.
The lowest sampled host headroom was 20,484,575,232 bytes.

The closeout is retained as `work/hag-editor-context-20261004/native-closeout.json`
with SHA-256
`dbe9a8f5a2ffc93714a930ce79d2b7ec5d8567879d9f47a298c72d7fdcc42278`.
This qualifies the on-demand recording strategy for the declared C128 numerical
and lifecycle fixtures. It does not qualify C512, normal editor tool use or a
serving-default change.

The earlier softmax repeated a full reduction for each probability lane; merely
raising its bound would make its work quadratic in prefix length. The additive
C512 implementation below changes that scheduling and its arena accounting. Initialization
staging must also include the largest zero-filled KV allocation. C8192/C16384
would exceed the current single tensor slot and the previous 2-GiB combined
policy even after command metadata is reduced. Full-editor readiness therefore
requires cache/arena design and an explicit measured memory budget, followed by
fresh numerical, lifecycle and client qualification.


## Additive C512 implementation

The separate on-demand planner now admits up to 512 evaluated positions within
checkpoint context. Legacy pre-recorded plans retain their 256-position and
4096 layer-position limits, and an on-demand-only plan refuses pre-recording
before payload reads or allocation. Serving remains on the qualified C128 path.

The attention score/value kernels admit 512 cached positions. Softmax uses one
wave: every active lane performs the same ordered maximum and F32 sum, then
writes positions `lane + 64*k`. This preserves the numerical staging while
avoiding repeated full reductions across additional output waves. Head width,
query-count and cache-stride limits remain unchanged. Shader storage still
rounds to 8192 bytes. Only the tensor arena ceiling increases, to 128 MiB;
auxiliary storage stays capped at 64 MiB.

For the fixed Qwen3-0.6B checkpoint, C512 tensors occupy 127,378,944 bytes,
persistent auxiliary payload is 119,124 bytes, peak auxiliary payload is
125,284 bytes, and the page-rounded HAG allocation is 760,778,752 bytes. Each
step records `4 + 28 * (max(4096, 12 * visible_positions + 4) + 12)` bytes;
512 steps plus reset/replay total 63,950,580 bytes. Submission itself still
transfers no intermediate tensors.

The new `--inspect-on-demand512`, `--on-demand512-first-token` and
`--on-demand512-context` probe modes retain the independent CPU oracle and
unchanged numerical tolerance. Full context checks all logits at positions
0, 31, 63, 64, 127, 254, 255, 256, 340, 341, 510 and 511, then refuses a further
position and resets/replays token 1. Native qualification is pending until the
separate first-token and full-context gates pass.

Before any live C512 work, its supervisor deadlines are fixed at 480 seconds
for the child, 495 for the service and 510 for outer supervision. The measured
C128 CPU plus native time was 91.931 seconds; four times that with a 30% margin
is about 478 seconds. These bounds cannot be extended during a run. Existing
2-GiB process, 1536-MiB early-stop, zero-swap, thermal, host-headroom and root
crash-watcher requirements remain in force. Supervisor failure-path checks
passed 31 mocked tests, including full-length bounded memory receipts.

The completed C128 evidence archive is
`outputs/hag-on-demand-c128-qualified-20261004.tar.gz`, SHA-256
`d9d96faf5412d899844028f4e6b7e1bd31cb2be8f2bd8fe005b66c73ce904d4f`.

The pre-live checks passed 100 backend tests (including LLVM source/embedded-word
parity), 99 model tests, 16 probe tests and nine serving regression tests. Two
unrelated real-checkpoint model tests remained ignored. Changed Rust files pass
format checks; repository-wide formatting still reports pre-existing differences
in unrelated files. Clippy uses the existing `chunks_exact_to_as_chunks`
exception. The development checks use one Cargo job, a 2-GiB cap and zero swap.

### Frozen C512 gate and thermal hold

The implementation and checks are committed at
`5cdbe55a63003b34f6877bb4bd28bffd1e588485`. The successful release build used
one Cargo job, peaked at 185.2 MiB with zero swap, and reached 77.75°C. Metadata
inspection then verified 103 frozen source inputs and a conservative combined
forecast of 1,681,480,242 bytes, including 64,960 bytes of retained attention
metadata. The 760,778,752-byte HAG allocation is separate from cgroup accounting.

The frozen binary SHA-256 is
`266a660171fc2bcfe54d0a2211111454d852120573924f260f1976748b36ade3`;
source-manifest SHA-256 is
`87a816b3081f0d4db1540ebce293655adacdc8b6de2cd2e04a7184a503263e85`;
complete live-input manifest SHA-256 is
`6013651475b22ac8e6ce3174f9d268d15eacef9fc7c6c6fae5df0a44a2a844f2`.

The self-check startup gate refused at **63.125°C**, before creating a workload
service. A subsequent finite 120-second cooldown observation ranged from
60.125°C to 79.25°C while this task's development workloads were stopped. It
did not reach the cooldown target. No second preflight, CPU model evaluation,
native device open, GPU dispatch or root crash watcher was started. All twelve
development services/cgroups had retired; sampled memory-pressure counters and
swap stayed zero. Serving remains pre-recorded C128.

The refusal, development results, exact contract, frozen binary/source and
future closeout audit are retained under `work/hag-context512-20261004` in the
development task workspace. Archive
`outputs/hag-context512-pending-20261004.tar.gz` contains 194 verified files,
SHA-256 `0c9e9e898a609ed70c4a52a325982521b98c12f74d1e61c86a100838eb1b671d`.
The original refusal log must remain intact. Resume on the same frozen inputs
with a successful cool self-check, then a fresh root watcher, first-token and
full-context gates, watcher shutdown and independent closeout. The saved
runtime C512 patch remains unapplied and untested until native qualification;
serving cutover also needs new HTTP/SSE/client proof.

### C512 admission refusal after cooldown

On 2026-10-05 the frozen C512 self-check passed, then first-token execution
returned an application memory-admission error before native device opening.
Its total forecast was 1,681,480,242 bytes, while the second check observed only
1,453,277,184 bytes of remaining cgroup space. The cold checkpoint's pages had
already been charged during the CPU reference. Comparing the whole-job forecast
against remaining space counted that charged portion twice. Both possible
admission-error sites precede the sole native open; no GPU dispatch occurred.

The refusal was safe: maximum Tctl was 43°C, cgroup peak was 694,796,288 bytes,
and all memory-pressure counters and swap remained zero. An independent audit
passed 166 checks, verified all six stage processes and the root watcher had
retired, both services/cgroups were gone, and the watcher stopped with zero
crash dumps. The complete kernel interval had no GPU/DRM/reset/OOM diagnostics.
An initial audit-script schema error was preserved and repaired in a separate
v2 audit; no workload stage was rerun. The successful safety audit remains
`qualified: false`.

The refused attempt is archived as
`outputs/hag-context512-admission-refused-20261005.tar.gz` (227 files), SHA-256
`c01d2e217947ab4c026857933697e56480208766aaa6dbd9b0cf25a652fd9a8b`.
Its independent `failed-attempt-closeout-v2.json` SHA-256 is
`d6f60164329eae19ad9d34bd144a08d82d2d28ca4341bf28c2701b75e422baa7`.
A corrected probe requires a new frozen attempt directory; the old results
and frozen inputs remain immutable.

The corrected C512 probe keeps the initial whole-job forecast and first
remaining-memory admission unchanged. After the CPU oracle retires, a separate
check treats that forecast as the total for the supervisor's private cgroup.
It restores only the smaller of two bracketing leaf-current samples to live
ancestor headroom, retaining all ancestor/host limits and the full checkpoint
allowance. It also requires a conservative 917,299,058-byte native additional
forecast to fit the actual remaining space. No cache warming, eviction or
assumed page residency is used. Missing/invalid samples, unexpected service
membership, absent NoNewPrivileges, or current/peak usage at the unchanged
1536-MiB early threshold refuse before native open. The 2-GiB hard cap, no-swap
rule, CPU/time limits and all thermal rules remain unchanged. This admission
policy is specific to the supervised probe and does not change the general
memory API or serving runtime.

All 19 probe tests pass, including the new cold/warm accounting, ancestor
pressure, remaining-allocation, counter-consistency and supervised-scope cases.
Targeted Clippy and formatting checks pass with the existing lint exception.
The corrected release and live gates use a fresh `work/hag-context512-20261005`
evidence directory and must not reuse the refused first-token result.


## C512 native qualification and serving cutover

The corrected frozen release at `904e3f2947797adc349a6f4f7a567af1ed0474fa`
passed self-check, first token and full context on 2026-10-05. The full run made
513 GPU evaluations against 512 CPU reference steps, compared 1,975,168 logits
at the twelve declared positions plus reset replay, and matched all 13 greedy
choices exactly. Maximum absolute difference was 0.00012683868408203125, with
zero sign/zero mismatches. Full-capacity refusal, reset/replay, per-step scratch
retirement and final zero HAG charge all passed.

The run made 1,945,809 physical dispatches and 29,754 logical submissions. It
recorded exactly 63,950,580 scratch/scalar bytes, transferred no intermediate
tensors during submission, and read 7,900,672 verification bytes. Auxiliary
payload peaked at 125,284 bytes. CPU reference time was 276.652 seconds and
native execution/checks took 127.547 seconds, within the unchanged 480-second
child bound. These are descriptive qualification timings, not a benchmark.

Peak Tctl was 79.25°C, cgroup peak was 181,698,560 bytes, and minimum host
headroom was 20,980,494,336 bytes. The separate HAG allocation remained
760,778,752 bytes and total admission forecast 1,681,480,242 bytes. All memory
pressure/OOM counters and swap were zero. Independent closeout passed 204
checks: all nine stage processes, three services/cgroups and the stopped root
watcher retired; zero dumps; no GPU/DRM/reset/OOM diagnostics in the full window.

Frozen binary SHA-256:
`bb7f9460d3550ef5a090fd012dd6a243541f10af21656ae67057274122a2e6a9`.
Live-input manifest SHA-256:
`42d7982d4a93a519a1d23a6ee259052fef6855ddb366230c3dc32d0916657e9f`.
Independent closeout SHA-256:
`239824f3e2fb996f99a25d8e740aeb3ae4bc4118ef0ef01592c46d172056f273`.
Archive `outputs/hag-context512-qualified-20261005.tar.gz` contains 192 verified
files, SHA-256
`840159cfd3287850aa7f9c31553ac7c1129538339abd52efcf8c380b4581ed91`.

The experimental HAG runtime now plans/prepares C512 on-demand attention,
advertises 512 evaluated positions, and validates position-dependent recording
writes and allocator high-water marks after every token. It keeps serial
requests, cancellation boundaries, the process-wide poison latch and CPU
defaults. The fixed model's inspected retained metadata yields a
1,540,561,088-byte runtime forecast; the standalone conservative metadata bound
used by the client harness yields 1,541,609,664 bytes. The HAG allocation is the
same 760,778,752 bytes in both calculations.

Before live serving qualification, 41 runtime tests and 39 server tests passed,
as did targeted Clippy and changed-file formatting. Real-checkpoint runtime and
serving tests stayed ignored in ordinary test runs. The new additive
`antimatter-grounded-c512` profile uses the measured 93-token no-tools grounding
prompt with 128 output tokens, requiring at most 220 evaluated positions.
It freezes Antimatter `78147b78d8271b993dc3734f469f763df1f0f548`, tests genuine
chat and stream calls plus reuse, and requires cold/warm typed HTTP 400 refusals
and a client-side refusal before fetch or lifecycle hooks. Its Python harness,
CLI and committed-client replay suites passed 175, 11 and 21 tests respectively;
the existing C128 adapter passed all 15 replay tests. Live client qualification
was a separate gate, recorded below. Full editor tools and live cancellation are
outside this declared client fixture.

### Frozen grounded-client gate and thermal hold

The serving changes and additive client harness are committed at
`0c2e11a3eb9be9661ecf42a318f2390bc6fd8b13`. The fresh offline release build
passed under the 2-GiB development cap with one Cargo job, a 421,900,288-byte
sampled memory peak, zero swap and a maximum Tctl of 74.625°C. Preparation froze
133 source inputs, the release executable and committed Antimatter provider.
The containment self-check passed without inference.

A finite 120-second observation then retained 240 temperature samples from
58.875°C to 79.875°C. The longest continuous interval at or below 60°C was
2.001 seconds, short of the required ten seconds. CPU reference, native client
requests and the root crash watcher were not started. This is a pending client
qualification, not a failed model result or a qualified serving result.

Independent audit passed all 299 checks: the observed build receipt, executable,
depfile, committed/frozen source bytes, client contract and safety receipts
agree. This establishes association with the observed build, not reproducible
build attestation. Build, preparation and self-check services/cgroups retired;
memory-pressure counters and swap stayed zero. The complete build-to-hold
kernel interval contained no GPU/DRM/reset/OOM diagnostics. The audit explicitly
retains `qualified: false` because CPU/native prerequisites are unrun.

Frozen binary SHA-256:
`db01fb85aae8d7174013d1026987719c20918f9389627132be10e10b3a504227`.
Source-manifest SHA-256:
`68c9fd0770f8d47c737cbf129f12a85e388d1b9ba39f3594ca3a35bebb54e054`.
Prepared-input audit SHA-256:
`91fdea0c9f9177e05516fdf9ec2800da75275c10c52aebae174889c035069acd`.
Archive `outputs/hag-grounded-serving-pending-20261005.tar.gz` contains 216
verified evidence files plus its manifest, SHA-256
`059c957471ad3ac3d540740a8581d2f1a3e8e619f6bce42f6e708b877b400e61`.
The task's `work/hag-grounded-serving-20261005/RESUME.txt` recorded the exact
frozen inputs and remaining CPU-first sequence at this hold. The cooldown record
remains intact; the resumed attempt and corrected fresh run follow.

### SSE keep-alive verifier failure and bounded initialization

After a successful cooldown, CPU and native chat passed. The first streaming
worker also matched CPU, but its independent supervisor rejected the legitimate
`: keep-alive` SSE comment sent during prefill. The original timing result stays
`qualified: false`. Independent safety audit passed 309 checks, with clean
memory, process retirement, stopped watcher and CPU/GPU kernel windows. The
failed attempt is archived as
`outputs/hag-grounded-serving-framing-refused-20261005.tar.gz`, containing 286
verified evidence files plus its manifest, SHA-256
`9b87ce8cc0db4d17b62bb4d8ec5c07edb7bc8d56f35e4751fb9e50cbcfb0a0f1`.

Commit `9b7b4d1` permits bounded physical SSE comment lines before `[DONE]`.
Comments still count toward byte, event and time bounds; malformed/multiline
entries and nonblank post-DONE entries refuse. They cannot supply content,
EOS or completion. Regression tests cover the observed long-prefill comment
and malformed, oversized, late and incomplete streams. A read-only replay
confirmed the diagnosis without relabeling the failed result.

Commit `02ad1a71977e0d68d8e2f3f2f7ace5c1686f3a14` also bounds logical F32 tensor
zero initialization with one shared immutable 64-KiB buffer. It keeps the full
arena reservation and queue lock until every chunk succeeds. Partial failures
retain the reservation under the existing poison/quarantine policy; no partial
tensor owner escapes. Successful host-write byte accounting is unchanged.
Context capacity, memory budgets, shaders and ordinary F32 upload/read conversion
are unchanged. All 56 native-backend transport tests, 177 qualification-harness
tests and targeted Clippy passed, including new chunk-boundary, reuse, ownership,
partial-failure and panic cases.

### Qualified grounded C512 host client

A fresh release from `02ad1a71977e0d68d8e2f3f2f7ace5c1686f3a14` passed all four
stages and stopped-watcher closeout on 2026-10-05. The preparation audit verified
290 build/source/contract checks; final independent audit passed 404 checks,
and a separate raw-client audit passed 158 checks.
The profile freezes Antimatter `78147b78d8271b993dc3734f469f763df1f0f548` and the
same P93/M128 no-tools request. Both genuine provider chat calls and the warm
provider stream match the CPU raw text exactly, including closed thinking and
the final `ready.`. Each request ends at EOS with 93 prompt and six completion
tokens; all three native requests evaluate 99 positions and reset healthily.

All three client processes refuse an oversized request before fetch or lifecycle
hooks. Cold nonstream and warm streaming server requests with a 421-token reply
allowance return typed HTTP 400 context refusals without changing generation
counts. Discovery advertises C512, the warm stream reuses the loaded model,
callbacks agree with raw content, both native services unload cleanly, and all
eleven stage/client processes, four services/cgroups and the root watcher retire.

CPU cgroup peak was 292,716,544 bytes at 56.5°C; native chat peaked at
121,847,808 bytes/52.625°C and the reuse stage at 124,432,384 bytes/54°C. The
separate HAG allocation remains 760,778,752 bytes; the declared conservative
combined forecast remains 1,541,609,664 bytes. All memory events and swap were
zero. The watcher captured no dumps, and the complete CPU/GPU kernel windows
contained no GPU/DRM/reset/OOM diagnostics. Resource and thermal limits were
unchanged.

The single warm stream received first content at 18.905 seconds and `[DONE]`
at 20.239 seconds, including one correctly handled prefill keep-alive comment.
These are descriptive client timings at 75% CPU, not decode throughput or a
performance qualification. This proof covers only the declared grounded
no-tools fixture; full editor/tools, arbitrary conversations, semantic quality,
live cancellation and concurrency remain outside scope.

Frozen binary SHA-256:
`da277b368bb37dbcde067af22ecdce4dbded6487d93607c045055d1b1a69537c`.
Source-manifest SHA-256:
`6df462ac9e0c615872afc9ea4a6f3d554476307cb5a85f4665282a7825accdcf`.
Standard closeout SHA-256:
`6305ce6f9d86836cf9d5c30d110b264160c293a6a48119676ff943bae3b43bb2`.
Independent final audit SHA-256:
`baedb57307cd2f7c3894f9659572757536e96c3eadff429541c5053a5f62609b`.
Archive `outputs/hag-grounded-serving-qualified-20261005.tar.gz` contains 273
verified evidence files plus its manifest, SHA-256
`1e2dd63ee490c1e7b56eb156d0f46b9c5ccfa698b5dac762950bedb524fe3584`.

### Context planning before C2048 serving

This historical plan preceded the serving qualification recorded later below.
Logical tensor zero initialization was bounded, and the additive C2048 plan,
phase accounting and separate CPU/native fixed-token qualification had passed.
The next step was C2048 serving admission and the larger Antimatter request.
Ordinary F32 upload conversion was unchanged.

C2048 was the next useful target for the measured no-tools prompt with project
instructions. Its 594 exact evaluated positions fit C1024, but Antimatter's
831-token estimate exceeds the C1024 prompt budget of 806. C2048 provides a
1,728-token prompt budget. C1024 can be an intermediate native proof, without
claiming that larger client fixture is supported.

With that F32 layout, C2048 requires 501,720,576 tensor bytes and
1,135,140,864 HAG bytes, exceeding serving's then-current 768-MiB HAG budget. The
fixed-token qualification below exercises these allocations under its own
1,152-MiB HAG budget. Serving admission needed a separate proof before its
capacity or memory budget changed. C8192/C16384 further
exceed the 1-GiB tensor-slot limit and need a bounded pool of tensor slots or
another independently qualified cache design. Full-editor tools remain a
separate milestone.

### C2048 phase accounting and attention boundary implementation

The 2026-10-07 implementation permits on-demand model plans through C2048 while
preserving eager recording's C256 and 4096 layer-position bounds. Native attention
admits at most 2048 cache positions, and an explicitly configured tensor arena may
hold at most 512 MiB. Default arenas, serving C512, its 768-MiB HAG budget, and
HAG's 1-GiB per-slot limit remain unchanged. The three attention shaders change
only four admission words; arithmetic, payload lengths, offsets and the 8-KiB
library remain unchanged. C2049 refuses before native recording mutates state.

The metadata-only `hag_model_probe --inspect-on-demand2048 --model PATH` mode
checks the selected checkpoint and reports costs without reading tensor payloads
or opening a device. At this implementation milestone it did not expose full-model
C2048 execution; the subsequent split-process modes are described below. For the frozen
639,447,744-byte checkpoint, twelve retained reference logits plus one working
vector, 16 MiB of owner allowance and 128 MiB of general headroom, the proposal is:

| Resource or phase | Bytes |
| --- | ---: |
| Packed weights | 633,233,408 |
| Resident tensor payload | 501,720,576 |
| Peak auxiliary payload | 143,716 |
| Page-rounded HAG charge | 1,135,140,864 |
| CPU KV cache | 469,762,048 |
| CPU phase forecast | 1,272,561,856 |
| Native host forecast | 803,260,978 |
| Native phase including HAG | 1,938,401,842 |
| All phases overlapping | 2,408,163,890 |

The overlap exceeds 2 GiB by 260,680,242 bytes. Dropping a CPU owner does not
prove allocator memory was returned, so full-model work must prove CPU
process retirement and resample host and ancestor/leaf cgroup headroom before
native admission. The proposed 1,152-MiB HAG budget was metadata only at this
planning milestone; the subsequent live qualification uses it below. The 2-GiB
process cap, 1,536-MiB early stop and zero-swap policy are unchanged; actual UMA
charging can still trigger that early stop. Coefficients and simultaneous upload
conversion peak at 1,572,864 bytes, below the 4,456,448-byte model staging maximum.

The additive attention-only `--diagnose-attention2048` mode freezes visible
prefixes 511, 512, 513, 1023, 1024, 1025, 2047 and 2048, four query heads/two KV
heads of width 128, and two distinguishable replays per prefix. It computes all
CPU references before opening one device, poisons future K/V with NaNs, checks
eight guards per replay, refuses C2049 transactionally and retires each batch.
Its scope is 16 submissions, 192 physical dispatches, 8,192 output comparisons,
128 guard checks and 33,280 verification-readback bytes. The HAG charge is
8,495,104 bytes under a separate 16-MiB diagnostic budget. It requires explicit
external supervision and makes no full-model or serving qualification claim.

Offline validation passed 153 backend tests, four attention-probe tests,
101 model tests, 21 model-probe tests, LLVM 21 assembly/byte/disassembly checks
for all eleven shaders, and targeted Clippy. Two existing CPU stress tests
explicitly request up to 4096 threads and exceed the retained 128-task development
cap; they were excluded after recording their task-limit failures. GPU execution
and its evidence are separate from these software checks.

### Qualified C2048 attention boundaries

On 2026-10-07, release `9440388da4d4281149c11d4ee280c05b3497c7a4` passed
the containment self-check, original attention regression and additive C2048
diagnostic under a single fresh root crash watcher. The gate froze 71 source
files, 17 inputs and finite 120/135/150-second child/service/outer deadlines.
Twenty hardware-free supervisor tests and a separate metadata inspection of the
selected checkpoint also passed. The earlier metadata-inspection attempt was
refused before process launch at 92.25°C; that refusal and the subsequent cooldown
observations are retained. Neither native stage started before its own
10-second cooldown at or below 60°C.

The baseline checked 3,218 values and 14 guards across seven submissions and
183 physical dispatches. The C2048 diagnostic checked all eight frozen prefixes,
two distinguishable replays per prefix, 8,192 values and 128 guards across
16 submissions and 192 physical dispatches. All eight C2049 requests refused
without changing native statistics. Both stages had maximum absolute error
`1.1920928955078125e-7`, within `2e-5 + 2e-5 * abs(reference)`, with the existing
sign/zero policy. Future K/V NaNs were excluded causally, and each case retired
its recording and tensor owners. No tensor transfers occurred during submission;
C2048 verification readback was exactly 33,280 bytes and its 8,495,104-byte HAG
charge returned to zero.

| Stage | Peak cgroup bytes | Maximum Tctl |
| --- | ---: | ---: |
| Containment self-check | 25,858,048 | 46.75°C |
| Original attention regression | 22,155,264 | 42.125°C |
| C2048 attention diagnostic | 94,679,040 | 42.25°C |

The 2-GiB hard cap, 1,536-MiB early stop, zero swap, 75% CPU quota, 64-task cap,
90°C thermal cutoff and host-headroom rules were unchanged. All workload memory
events and swap were zero. Minimum available host memory in the C2048 stage was
21,507,506,176 bytes. All nine stage processes, three services/cgroups and the root
watcher retired. The watcher captured no dumps and its output directory was empty;
the complete 2,382-line kernel window contained no GPU/DRM/reset/OOM diagnostics.
Standard closeout and 2,790 recorded independent audit checks passed.

This qualifies only the declared single-query GQA fixtures with four query heads,
two KV heads and head dimension 128. It does not qualify a complete C2048 model,
serving, editor integration, arbitrary inputs or throughput. The next step remains
a separately supervised CPU reference process, checked reference handoff and
fresh native admission after CPU process retirement. Serving remains C512.

Frozen binary SHA-256:
`744b272c832415e618b8095a482819a2dc481977f1fe3b408452453f22313007`.
Source-manifest SHA-256:
`a19ee53d8ea384eab241c52cc2d8b295fd8e32822a0cfb41b3965c91c39a9d92`.
Gate-manifest SHA-256:
`60463f7a6a731cab4d389e698e946ef43f6f1a2ead7dc553879dd09f3e19ac1d`.
Standard closeout SHA-256:
`4b4cee59d62d1bc8520a29a8742e798a16867bf75d21266adf562174d079bb79`.
Independent audit SHA-256:
`c5c105512c8264bca0397e2d2a33dbf2bca142fc8e1c23cc5dff95c09b0a2d31`.
Archive `outputs/hag-context2048-attention-qualified-20261007.tar.gz` contains
161 individually verified evidence/source/binary files, with a separate file
manifest and checksum. Archive SHA-256:
`bdfc804ae5d0d228befbd8d3b129002a1b644579f9ad26dd9c7c93f2c762e8da`.

### Split CPU/native C2048 qualification implementation

The probe now provides three qualification-only modes. They do not change serving
capacity or its memory budget:

```text
hag_model_probe --on-demand2048-reference --model PATH --input-identity PATH --reference-out PATH
hag_model_probe --on-demand2048-first-token --model PATH --input-identity PATH --reference PATH /dev/dri/renderD128
hag_model_probe --on-demand2048-context --model PATH --input-identity PATH --reference PATH /dev/dri/renderD128
```

The CPU process evaluates fixed token IDs 1 through 2048 and exports full-vocabulary
F32 logits at positions 0, 127, 340, 341, 510, 511, 512, 1022, 1023, 1024, 2046 and
2047. Its exclusive partial file is published only after generation and the
capacity refusal succeed. The 7,293,376-byte reference preserves raw float bits,
requires finite logits and exact first-maximum greedy IDs, and rejects missing,
reordered, truncated or trailing records. Its 128-byte identity binds the external
supervisor's checkpoint, source-manifest, executable and contract SHA-256 values.
Rust compares those attestations; the supervisor performs cryptographic hashing.

After proving that the CPU process, service and cgroup have retired, the supervisor
authenticates the completed reference and resamples memory. Native modes validate
the reference before device access and use no CPU oracle. The first-token gate
allocates the full C2048 model/cache but evaluates only token 1. The context gate
evaluates 2048 tokens, checks all twelve selected positions, refuses token 2049
without changing state, resets, replays token 1 and resets again. Every comparison
requires finite values within `0.002 + 0.0002 * abs(CPU)` and exact CPU greedy;
sign and zero mismatches remain diagnostics. Expected full-context totals are
2,049 evaluations, 7,771,857 dispatches, 118,842 submissions and 1,975,168 compared
logits. All HAG owners must retire to a zero budget.

Both Rust and the supervisor require the actual private cgroup's 2-GiB hard cap,
1,536-MiB early threshold, zero swap, 75% CPU quota, 64-task cap and no-new-privileges.
Phase admission accounts for the live private cgroup charge and all limiting
ancestors. Start headroom remains 8 GiB, runtime headroom 4 GiB, with ten seconds
at or below 60°C before each stage and a 90°C cutoff. These limits are unchanged.
The native qualification uses an explicit 1,152-MiB HAG budget; serving keeps its
existing 768-MiB budget. Reference I/O uses bounded buffers within the owner allowance.

Finite child/service/outer deadlines are fixed before execution: 120/135/150 seconds
for containment and first token, 1800/1815/1830 for CPU export, and 1080/1095/1110
for native context. The longer bounds use measured C128/C512 timings, a two-point
linear-plus-quadratic estimate and 30% margin; they are estimates, not throughput
claims. One fresh task-local root watcher has a fixed 1500-second duration after
CPU retirement, preserving its four-dump and 64-MiB-per-dump limits. No deadline
is extended after launch.

Offline validation passed 101 model tests, 35 probe tests, targeted Clippy, release
compilation, metadata inspection and 32 hardware-free supervisor tests. These
include fixed watcher-lease admission and bounded refusal when a reference file
is replaced by a FIFO between inspection and open. Independent reviews checked
the reference wire format, accounting, process retirement and supervision.

Release `03f47222a50d22e30ae98ff5315d21ba1be3e5dc` is frozen with 96 compiled-source
inputs and 20 gate inputs. Binary SHA-256:
`de1bf43db6ab688e75f8bec5901ca889d5da577f50bcb134b14b7cc67ea951c4`.
Gate-manifest SHA-256:
`9e1e1e8e36f95263571d18a9bc72dc6c61d93b53593a678b880f7a1d52a81890`.

The first containment preflight refused at 72.25°C before creating a service or
starting any model/device process. Two subsequent 120-second temperature
observations did not establish a continuous 30-second cooldown; both reached
72.75°C while other build work was active. The refusal and observations are
retained under `work/hag-model2048-20261007`. Those attempts started no CPU model,
native stage or root watcher. The frozen inputs and original limits were preserved.

After cooldown, the containment check and separate CPU export passed. The CPU
oracle completed 2,048 evaluations in 1,711.499 seconds, checked its capacity
refusal, and published twelve finite full-vocabulary reference vectors containing
1,823,232 values. Independent reference parsing checked all records, exact greedy
IDs, identity fields and EOF. The file is 7,293,376 bytes with SHA-256
`2c8cf950b1dfe93967ba2a8d1d0e1435a06b6233d23efe4092c1202f59c2d21d`.
Peak cgroup memory was 645,521,408 bytes; maximum Tctl was 74.75°C and minimum
available host memory was 21,137,350,656 bytes. All workload memory events and swap
remained zero, and the CPU process, thermal guard, supervisor, service and cgroup
retired. This proves the reference-export phase only; the following section
records the completed native gates. Serving remains C512.

### Qualified split-process C2048 full model

On 2026-10-07, frozen release `03f47222a50d22e30ae98ff5315d21ba1be3e5dc` passed
the full-allocation first-token gate and the complete C2048 fixed-token gate,
using the authenticated CPU reference above. Both native stages used one fresh
1,500-second root crash watcher after CPU process retirement. The original
deadlines, memory/thermal limits and numerical policy were unchanged.

The context gate completed 2,048 tokens plus one reset/replay evaluation, checking
1,975,168 logits and all thirteen exact CPU greedy results across 7,771,857
physical dispatches and 118,842 submissions. Maximum absolute error was
`0.00012683868408203125`, within the per-value `0.002 + 0.0002 * abs(CPU)` limit.
There was one near-zero sign mismatch and no zero mismatches; the frozen policy
reports these diagnostics without requiring bitwise equality. Capacity overflow
refused without a dispatch or state change, stale logits refused, reset/replay
passed, and all HAG owners retired to a zero budget.

Native preparation took 0.631 seconds and native execution plus checks took
948.882 seconds. These are observed fixture timings, not a throughput or serving
latency qualification. Native preparation read 633,495,552 source bytes; recording
wrote 725,505,780 host bytes, submission transferred no tensors, and verification
read back exactly 7,900,672 bytes. Auxiliary peak was 143,716 bytes, with only one
transient attention batch at a time.

| Stage | Peak cgroup bytes | Maximum Tctl |
| --- | ---: | ---: |
| Containment self-check | 24,535,040 | 44.125°C |
| Separate CPU reference | 645,521,408 | 74.75°C |
| Full-allocation first token | 162,848,768 | 56.875°C |
| Native C2048 context | 153,591,808 | 80°C |

The 1,135,140,864-byte HAG allocation is accounted separately from process-cgroup
memory; the table is not total inference memory. Minimum available host memory
in the context stage was 17,993,093,120 bytes. All workload swap and memory events
were zero. All twelve recorded stage PIDs, four services/cgroups and the root
watcher retired. The watcher stopped explicitly with zero dumps and an empty
private output directory. The complete CPU and watcher kernel windows contained
8,275 and 5,390 lines respectively, with no GPU/DRM/reset/OOM diagnostics.
Standard closeout and 3,735,394 independent audit checks passed.

This qualifies the declared Qwen3-0.6B Q8_0 fixed-token C2048 model fixture, selected
logit comparisons, capacity refusal and reset/replay on this host. It does not
qualify C2048 serving, editor requests, arbitrary inputs, tools or throughput.
Serving remains C512. The next useful profile is Antimatter grounding plus the
measured 2-KiB project-instruction fixture: 467 prompt tokens and 128 maximum reply
tokens, with a 594-position admission bound. That serving slice needs its own
1,152-MiB HAG configuration, CPU reference budget and finite client deadlines.
The existing 768-MiB HAG and 1-GiB CPU budgets refuse the corresponding resource
forecasts; the existing 60-second client deadline is unqualified for the larger
request. Keep the 2-GiB hard cap, 1,536-MiB early stop and current thermal guards.

Source-manifest SHA-256:
`43c6b2e5caec1137bdeb87631589c39b71cff458cde9a0f44135f809b3390023`.
Standard closeout SHA-256:
`a171cf1e4d0c55a552f22317348679e55ac2c03081c05d647d103e056c4b8112`.
Independent audit SHA-256:
`ad238aaa272b0dbac66c282bf889b0d4d2160ea9274ba95ce273982863060d64`.

Archive `outputs/hag-model2048-qualified-20261007.tar.gz` contains 192 individually
verified evidence, reference, source and binary files (51,167,911 uncompressed
bytes), with separate file manifest and checksum. Its documentation snapshot
precedes archive publication; the checkpoint itself remains external.
Archive SHA-256:
`1fc555207f0dff542792a089abd62110f457576232c693555504f793e6593001`.
File-manifest SHA-256:
`fb27fbc764ae4609f84b3da6a909b9c2eb6092b0557035bb9db60cd41401bf31`.

## C2048 serving implementation and startup hold — 2026-10-07

Runtime commit `ddd78bc` advertises 2048 evaluated positions and keeps exact
pre-device admission. A 467-token prompt admits an explicit 1582-token output
limit and refuses 1583. The fixed HAG allocation is 1,135,140,864 bytes; current
private-layout host/UMA forecast is 1,914,935,488 bytes. The explicit example uses
1152 MiB for HAG; CPU remains the default for ordinary configurations.

Dedicated-job reload admission uses `BEHOLDD_HAG_PRIVATE_CGROUP` only after the
launcher verifies its exact unified cgroup. Credit is capped by the checkpoint
allowance and the smaller bracketing leaf charge. Full configured/host bounds and
additional-allocation headroom checks remain separate; unrelated ancestor charges
receive no credit. Unattested serving retains conservative admission.

Profile commit `fe519564674186e70d1a17138259179347d590e9` adds
`antimatter-project-c2048`: the measured 467-token no-tools grounding and synthetic
2-KiB project instructions, 128 output tokens and a 594-position request ceiling.
It freezes provider `78147b78d8271b993dc3734f469f763df1f0f548`. Its CPU budget,
native budget, per-stage deadlines, private containment, genuine chat/stream reuse
and capacity refusals are documented in [hag-qualification.md](hag-qualification.md).

Software validation passed 44 runtime, 39 server, 194 qualifier, 12 CLI, 34 client
replay and 13 crash-watcher tests, plus Clippy and the release build. The client
replays included the committed provider. All 14 development/audit services were
retired; sampled memory/swap events remained zero. Maximum sampled development
memory was 1,203,625,984 bytes and maximum guarded development Tctl was 63.5°C.
Initial test failures and an audit invocation with an invalid commit argument
remain preserved alongside the corrected passing runs.

The independent prepared-input audit passed 277 checks, associating the observed
release build and depfile with the committed/frozen source and client closure.
Release SHA-256 is
`47af2583d9e25950fd05b12ba89ea955757c6b0002fd84ef9119195ae2e3c83b`;
source manifest SHA-256 is
`e8103aba9809a30d043902987d0e61616a848ad3789e95f0dd935edb78d0bad6`.
This is an observed build association, not a reproducible-build attestation.

The first self-check stopped in its 60°C/10-second startup gate. Its 600 samples
over 60 seconds ranged from 59.875°C to 77.5°C while other compilation continued.
No test service, model, GPU submission or crash watcher started. A separate
120-second read-only observation reached 75.625°C and did not sustain cooldown.
The 2-GiB hard cap, 1536-MiB early stop, zero swap and 90°C running cutoff remain
unchanged. CPU and native qualification are unrun; `qualified` remains false.

Evidence is preserved in `work/hag-serving2048-20261007`. Archive
`outputs/hag-serving2048-prepared-20261007.tar.gz` contains 214 individually
verified files, SHA-256
`9032784b8bfb7e7b2e4871a1aa4ef185ef1beb53a8664402ec4ff0434ff99b7e`.
The file-manifest SHA-256 is
`ddac76f6e7801288e06ee904b1877d5f580c6d88e7ce30e46e1e969fd1aec4d5`.

Resume in a quiet host window with a fresh evidence directory, preserving this
held attempt. Run preparation and its independent input audit, then self-check
and CPU reference. Only after CPU retirement may a fresh 1500-second read-only
watcher cover native answers and chat/stream reuse. Stop it explicitly, complete
both closeout audits, and archive the result before claiming serving qualification.

## C2048 project-instructions CPU reference and native hold — 2026-10-08

A fresh `live-cooled` attempt froze clean Beholdd source
`a004a2ae43af6a31f587718d68ec755f19320c00` with the same release binary, compiled
source manifest, profile and Antimatter provider listed above. Its independent
prepared-input audit passed 277 checks. Self-check and CPU reference both passed
their 60°C/10-second startup gates and completed successfully.

The CPU returned `<think>\n\n</think>\n\nready.` with EOS `stop`: 467 prompt
tokens and six completion tokens, or 473 evaluated positions against the
594-position ceiling. This supplies the exact backend reference; the fixture's
passing word/format diagnostics do not qualify general semantic quality.
CPU-reference SHA-256 is
`2c7211e9ae6a034f2f7a43519cf50e1b5232099492a19f65979abf4c952b0e56`.

CPU cgroup peak was 449,527,808 bytes, minimum available host memory was
14,217,285,632 bytes, and maximum Tctl was 60.5°C across 1,793 running samples.
Recorded memory events and swap remained zero. Both completed services, their
cgroups and all four recorded child/thermal processes retired before native work.

A fresh 1500-second read-only crash watcher armed after CPU retirement. The
native answer stage then refused its startup gate: 600 samples over 60 seconds
ranged from 69.75°C to 73.125°C. No native answer service or GPU workload launched;
timing was not attempted. The watcher stopped explicitly with zero dumps, no
surviving process and an empty private output directory. The independent refusal/retirement receipt
passed 618 checks. Overall and native qualification remain false, and all memory,
deadline and thermal bounds remain unchanged.

This attempt is preserved in `work/hag-serving2048-20261007/live-cooled`; its
`native-startup-refusal-cooled.json` receipt is in the parent evidence directory.
Resume with a new prepared directory and fresh per-attempt audits. If using the
dedicated validation-host machine, first record its kernel, driver, device identity and
containment support, then run the complete CPU/native sequence there. Results
from different hosts must retain their own provenance and qualification scope.

Archive `outputs/hag-serving2048-cpu-held-20261008.tar.gz` preserves 428
individually verified files, including earlier held attempts, frozen inputs and
the successful CPU reference. The checkpoint payload remains external. Archive
SHA-256 is `2f095b184510a4a31da5eb5ec007c87fa94b64259e34d720529ebb60711cf04e`;
file-manifest SHA-256 is
`617c1c8dbb93ac12046ebef217c74e57c696bde64121fbbc6abbc274d56dd556`.
This archive is explicitly unqualified for native serving.

## Qualified C2048 project instructions on validation-host — 2026-10-08

The complete `antimatter-project-c2048` profile passed on `validation-host`, an
x86-64 Linux host running `6.17.0-1024-oem` with AMD device `0x1586`/GFX1151 at
`/dev/dri/renderD128`. This is a separate execution record from the shared-host
thermal holds above. CPU reference, native answers and chat/stream reuse all ran
on validation-host under the unchanged profile; no earlier CPU or GPU receipts were
substituted for these stages.

Clean source `11f20bcc824420b29347bca0d5aa43f8e62230ba` and the unchanged release
binary were transferred with the original bounded build records. The independent
prepared-input audit passed 287 checks, linking the transferred binary, all 137
compiled source inputs and 17 frozen gate inputs to their committed identities.
The binary was built on the original host, not rebuilt on validation-host; this remains
an observed build association rather than a reproducible-build attestation.
Antimatter provider `78147b78d8271b993dc3734f469f763df1f0f548` was read from a
separate committed snapshot. A private Node v24.19.0 runtime supported the frozen
provider; the system Node installation and Antimatter working tree were unchanged.

The exact synthetic 2-KiB project-instructions/no-tools request used 467 prompt
tokens and a 128-token output allowance. CPU produced
`<think>\n\n</think>\n\nready.` with six completion tokens and EOS `stop`.
Both genuine `Provider.chat` calls matched CPU raw text, API usage and finish
reason. The warm `Provider.stream` matched raw text/EOS and its native evaluated
position count; its usage comes from the matched non-streaming warmup because
this SSE response carries no usage field. Each native request evaluated 473
positions: 1,419 total across the three requests, below their declared per-stage
ceilings. Every native request reported a successful reset and an unpoisoned
runtime. Final unload and process exits were clean.

Cold non-streaming and warm streaming requests with the explicit 1583-token
overflow allowance returned typed HTTP 400 `context_length_exceeded`, without
generation or additional native audit events. All three provider processes also
refused the oversized request locally before fetch or tracking hooks. These
checks preserve the distinct exact server and conservative client boundaries.

| Stage | Peak cgroup bytes | Minimum host available bytes | Maximum Tctl |
| --- | ---: | ---: | ---: |
| Self-check | 16,879,616 | 21,637,058,560 | 48.125°C |
| CPU reference | 444,768,256 | 20,867,473,408 | 74.125°C |
| Native answers | 115,646,464 | 20,347,826,176 | 74.625°C |
| Chat/stream reuse | 125,222,912 | 20,350,849,024 | 79.125°C |

These are measured process/cgroup charges; the separate planned HAG allocation
remains 1,135,140,864 bytes. The 2-GiB hard cap, 1536-MiB early stop, zero swap,
75% CPU quota, 64-task limit, host-headroom checks, 60°C/10-second startup gate,
90°C/0.1-second running cutoff and finite deadlines were retained. Recorded
memory events and swap remained zero. No retry or deadline extension was needed.

CPU and its cgroup retired before the fresh 1500-second read-only root watcher
armed. The same watcher covered both native stages and stopped explicitly with
zero dumps and an empty private output directory. All eleven recorded stage/client
processes, all four service cgroups and the watcher retired. Complete CPU and
watcher kernel-window queries returned successfully without matching GPU/reset/OOM
diagnostics. Standard closeout and 1,125 independent final audit checks passed.

The first chat after lazy load took 120.594 seconds, excluding CLI startup. The
warm stream produced first raw content, including thinking markup, after 118.623
seconds and completed after 120.544 seconds. These are single client-visible
observations under the 75% CPU quota, not a throughput benchmark or interactive
latency qualification.

This qualifies backend parity, the committed provider's chat/stream reuse and
capacity refusals for this exact fixture on validation-host. It does not qualify arbitrary
conversations, full editor tools, semantic quality, live cancellation, concurrent
instances, device sharing or every position of C2048 on this host. The separate
fixed-token full-model proof retains its original host and source scope.

The subsequent C2048 real-TCP cancellation/reuse and unload/reload slice is
recorded below. The later seeded-history attempt is also retained below; it
failed its CPU prerequisite before native work. Full tool prompts and prefill
latency remain separate work.

Release SHA-256:
`47af2583d9e25950fd05b12ba89ea955757c6b0002fd84ef9119195ae2e3c83b`.
Source-manifest SHA-256:
`e8103aba9809a30d043902987d0e61616a848ad3789e95f0dd935edb78d0bad6`.
Frozen run-manifest SHA-256:
`a20e70e5f675dd5f3996fbd8e7dd1d3d0715298ad6371806bba7b7b59a62841a`.
CPU-reference SHA-256:
`60b3b58cc97197c386a4c2148751cafef449a6f3921a452d8275760e203cb9da`.
Standard closeout SHA-256:
`9a330d6639ff61192e2ccafe13be05c489749d778e67b2073d049a4e15401ef4`.
Independent final audit SHA-256:
`d8c00562e2c11a16d7380418eb952d605f11190c066f29433dad7b1216a628c4`.

Archive `outputs/hag-serving2048-validation-host-qualified-20261008.tar.gz` contains 275
individually verified files (138,215,979 uncompressed bytes), including frozen
source/client inputs, the off-host build records, host/input binding and both
closeouts. The checkpoint payload remains external. The archive was independently
verified again after retrieval to the development host.
Archive SHA-256:
`7f10187677fa86d8f873941aeb3db5594541924495e3f3c878e8f06405d5e3bb`.
File-manifest SHA-256:
`c4db8688a7cf629c05ecfdc35119dbace14ad87300ca672d128f6aa7cd28f340`.

## Qualified C2048 TCP lifecycle on validation-host — 2026-10-08

The committed `lifecycle-c2048` harness at Beholdd
`f2d4c00c8896ec9e49aae63a1b5025cc59499310` passed its first live attempt on
`validation-host`, Linux `6.17.0-1024-oem`, using the same explicit AMD render
device and unchanged release binary as the project-instructions run above.
The fresh 279-check preparation audit bound 137 compiled source inputs, twenty
frozen gate inputs, the checkpoint and original off-host build association.
This profile uses direct real HTTP/TCP; it does not execute an Antimatter provider.

The fixed `Hello` prompt tokenized to nine tokens. A separate CPU process produced
one token, raw text `<think>`, with usage 9/1/10 and finish reason `length`.
Four successful native requests matched that raw text and finish: initial JSON,
complete SSE, reuse after disconnect and lazy reload. Nonstream API usage matched
CPU; complete SSE had the exact text, one length-finish event and `[DONE]`, with
its nine evaluated positions independently checked in the native audit.

The cancellation request used a sixteen-token allowance. After its first content
event, unload on a separate connection returned HTTP409 `runtime_busy`; the driver
then executed `shutdown(SHUT_RDWR)`. The server recorded cancellation after ten
evaluated positions, `reset_ok=true` and `poisoned=false`, before reuse. The ordered
audit counts were **9, 9, 10 cancelled, 9, 9**: 46 total, below the 60-position
ceiling. Natural completion instead of cancellation would have failed the gate.

Unload succeeded twice before reload and twice at the end. Eight snapshots bind
the same native PID, process start ticks, boot and dedicated cgroup through reload.
Both server processes ended unloaded and exited zero on idle SIGTERM. Raw HTTP
bodies, SSE prefixes, complete streams, audit logs and memory snapshots are retained
and replayed independently of the worker's PASS marker.

| Stage | Peak cgroup bytes | Minimum host available bytes | Maximum Tctl |
| --- | ---: | ---: | ---: |
| Self-check | 16,678,912 | 21,586,317,312 | 47.5°C |
| CPU reference | 253,431,808 | 21,306,224,640 | 62.25°C |
| Native lifecycle | 92,327,936 | 20,350,185,472 | 62.25°C |

These measured cgroup charges remain distinct from the planned 1,135,140,864-byte
HAG allocation. Hard memory 2 GiB, early stop 1536 MiB, swap zero, CPU75%, 64 tasks,
no new privileges, host-headroom thresholds, 60°C/10-second startup and
90°C/0.1-second running cutoffs were unchanged. Recorded memory events and swap
were zero. No workload retry or deadline extension occurred.

All three CPU-related PIDs and its cgroup retired before the fresh 1500-second
root watcher armed at 17:47:50 UTC. Native lifecycle ran at 17:48:15–17:48:28 UTC;
the watcher stopped explicitly at 17:48:58 with zero dumps and an empty private
root-owned output directory. All eight recorded stage/server PIDs, all three
cgroups and the watcher retired. Full CPU and watcher kernel windows returned
`-- No entries --`, with no matching GPU/reset/OOM diagnostics.

Standard closeout and **1,095 independent audit checks passed**. The independent
auditor's first invocation stopped at a Python syntax error before replay; its
original source and failed command receipts are preserved. Correcting the single
missing parenthesis allowed read-only replay to finish; no CPU/GPU stage was rerun.
The harness passed 244 distinct model-free qualifier/CLI tests before freezing.

This qualifies the stated direct-TCP lifecycle and exact backend parity for one
short prompt. It does not qualify complete answers, semantic quality, Antimatter
provider cancellation, long conversations, all C2048 positions, concurrency,
device sharing, throughput or termination during active inference. Cancellation
is checked around synchronous token evaluations; this result makes no general
latency or saved-computation guarantee. Reload passed with ample host headroom;
this run does not stress admission near its memory limit.

The subsequent measured seeded-history attempt is recorded below; its CPU
prerequisite failed before native execution. Prompt rendering is the next
correctness slice. Prefill latency for the existing 467-token project fixture
remains a separate optimization target; the observed roughly two-minute response
time has not been resolved by this short lifecycle check.

Frozen run-manifest SHA-256:
`bc2eda5ff67981bd266fad5b9612ee6bf4d5a3c8059579d3212c531f431ee546`.
CPU-reference SHA-256:
`0ea9d231e857dea688ffe364ad645e3ef0d8297a90729c07a47f9843c41d4fc3`.
Standard closeout SHA-256:
`c096b2f406b9276d20be84e2cae7a960652fa9ae8e088a4c20cae53bf0a2accf`.
Independent final audit SHA-256:
`8ad215d0672988ceab0b427827448ac49cac1915b86242cdaeb3d91f3d05bf11`.

Archive `outputs/hag-lifecycle2048-validation-host-qualified-20261008.tar.gz` contains
276 individually verified files (9,333,106 uncompressed bytes), including frozen
inputs, raw lifecycle evidence, original build records, both closeouts and the
preserved audit repair. The checkpoint payload remains external. All members and
their hashes were verified again after download, without extraction or execution.
Six remote and five local development/audit/packaging units also retired.
Archive SHA-256:
`88622588543ea5da545cbe48c9713dac4b76b7c328db55d71cc6659f5e444974`.
File-manifest SHA-256:
`30e967e97b00f9467308f82339595f2790e4dd860b25be54c98885c388ee88ca`.


## Held C2048 seeded project history on validation-host — 2026-10-08

Beholdd `2fe12d3a199be9456a87d06bef5f93d2c17cf044` adds the
`antimatter-history-c2048` profile. It preserves the project fixture's first two
messages, seeds the exact prior raw assistant reply
`<think>\n\n</think>\n\nready.`, and appends
`Reply with that same one word again. /no_think`. This is one seeded second-turn
request, not a generated first turn or full editor conversation. The unchanged
Antimatter `78147b7` provider forwards all four messages verbatim;
prompt rendering happens in Beholdd.

Fresh metadata-only tokenization measured **495 prompt tokens**, 2,842 rendered
bytes and a 622-position maximum with 128 output tokens. The client estimates
862 tokens. Server and client output boundaries are respectively 1554/1555 and
1187/1188. The profile retains one CPU reference, one native answer and two native
reuse requests, with ceilings 622/622/1244 and all existing C2048 deadlines,
private-cgroup, memory, no-swap, CPU and thermal controls.

The frozen inputs passed 319 independent preparation checks on `validation-host`,
Linux `6.17.0-1024-oem`. The unchanged release binary
`47af2583d9e25950fd05b12ba89ea955757c6b0002fd84ef9119195ae2e3c83b`
retains its observed original-host build association with 137 unchanged compiled
source files; no remote or reproducible build is claimed. The self-check passed.
The CPU stage ran from 18:17:25 to 18:20:44 UTC and **failed**: the complete HTTP200
response reached EOS `stop` after six completion tokens (usage 495/6/501), but its
literal content was:

```text
<think>

**ready**
```

No closing thinking tag was present. The response was not truncated: raw and
parsed JSON agree, with no read or parse error. Request time was 195.571 seconds,
within the 600-second CPU HTTP deadline. The gate rejected this structure before
semantic diagnostic assessment or reference/report publication. EOS alone does
not prove a complete final answer. Two additional model-free regressions retain
these raw bytes and verify that this failure cannot publish a CPU reference or
advance to native work. The initial 308 tests and the two added cases passed.

Native history execution, provider live calls, timing and crash capture were
**not started**. This is not a new native qualification, a semantic result or a
latency improvement. The earlier project-instructions and separate TCP lifecycle
proofs remain unchanged. No parser requirement or resource control was relaxed,
and the failed model request was not retried.

| Executed stage | Cgroup peak bytes | Minimum available host bytes | Maximum Tctl | Thermal samples |
| --- | ---: | ---: | ---: | ---: |
| Self-check | 17,092,608 | 21,578,993,664 | 49.625°C | 2 |
| CPU reference, failed output | 465,223,680 | 20,993,556,480 | 76.125°C | 1,949 |

All retained memory-pressure and swap counters are zero. The CPU kernel window
contains no entries, and the thermal supervisor did not abort. The failed CPU
server has no separate unload/exit-zero receipt, so that lifecycle claim is not
made here.

Source inspection found a concrete template mismatch. The checkpoint's actual
GGUF chat template removes thinking content from assistant messages before the
last user query. For the supplied seed it renders only `ready.` in that prior
assistant turn. The current server copies the inline thinking block verbatim.
The template SHA-256 is
`5da44855ab7e0641774d5ed0ff0b4f89671d01df7d316e54d06557a962211cb4`.
This mismatch is established from metadata; whether it caused this malformed
response remains unproven. The checkpoint makes `enable_thinking=false`
conditional; `/no_think` is ordinary prompt text, not a server-enforced mode.
Do not repair the response by stripping markup or inventing a closing tag.

Next, make no-tools history rendering match the supported checkpoint template
while preserving raw provider input, then measure the changed prompt and freeze
a new CPU-first run. Aggregate preparation/prefill/decode/transport timing is a
separate next step; synchronous submission and fence waits need measurement
before choosing a performance change.

Frozen run-manifest SHA-256:
`0b22a1024f00240464facb6cdb4f970817716c9eb815fde5992924adf0bcbdb7`.
Prepared-input audit SHA-256:
`7f396c198f6069806b22c10c459ec4db5056e9ed24826a1f107d227667ba873b`.
Failed CPU result SHA-256:
`c0f5dc9c117d5bae227f31665309504abebe8f72e3a0f86f750f83b1e6e88e7b`.
Raw HTTP response artifact SHA-256:
`02619d3843dca8536ee0cc101ef3d11bd0982f8c986abb7b09312b1023ce1ad6`.

The independent held-run audit passed **930 checks**, while keeping
`qualified:false` and `native_started:false`. Both stage units are inactive,
both cgroups are gone, all four recorded stage/guard PIDs are absent, and the
former CPU listener is closed. Five remote and eight local development,
audit, packaging and verification units also retired. The temporary SSH session
was closed. No driver settings or unrelated service was changed.
Held-audit SHA-256:
`55a58fd50c84126b8de55fbcd0899ead28fc02be9500d97e3e6a39c83a428ebc`.

Archive `outputs/hag-history2048-validation-host-held-20261008.tar.gz` preserves
260 individually verified files (136,171,499 uncompressed bytes), including
frozen source/client inputs, raw failure and resource receipts, original build
association, metadata measurements and the exact inspected chat template.
The checkpoint payload remains external. The archive and every member were
verified again after download without extraction or execution. Its manifest
retains failed CPU qualification and separately verified safe retirement.
Archive SHA-256:
`c4a6aaddedf68c575271b081d491cb867e468535d8f6545a32c8d5e51004a132`.
File-manifest SHA-256:
`1ad5f91648827f092fb0b5b28b94d1967e449dd5ad7683c0327c78b45270a648`.

## Checkpoint-template history correction and native thermal stop — 2026-10-08

**The corrected CPU prerequisite passed; native qualification is held after a
thermal abort.** The earlier P495 CPU-parser failure and its archive remain
unchanged. Source inspection established that the qualified Qwen3-0.6B template
removes completed thinking content from historical assistant messages before
the last user query. The previous renderer retained it. This mismatch is real;
it does not establish the cause of the earlier malformed model output.

Commit `4c09338` selects rendering by exact checkpoint template bytes during
admitted, cancellable planning. The Qwen3-0.6B no-tools path follows its history
rules; the separately matched Coder template retains the existing tool renderer.
Missing or unknown native templates return typed HTTP 400 before generation.
Wire messages remain unchanged. Tools/tool history and extra thinking fields
are refused for the Qwen3-0.6B path; `/no_think` remains ordinary prompt text.
The [template identities and license](../crates/beholdd-server/src/templates/README.md)
are retained with both exact fixtures.

Commit `c9a2e7511303cb471fa040913b64f58de2989352` adds the distinct
`antimatter-history-template-c2048` profile. Its four wire messages match the
held P495 fixture, but exact metadata tokenization through the corrected
renderer measures **P491 / 2,823 rendered bytes**. M128 allows at most 618
evaluated positions; CPU/answers/timing counts remain 1/1/2 with ceilings
618/618/1,236. The server output fit/refusal boundary is 1558/1559. Antimatter's
unchanged E862 estimate retains its conservative 1187/1188 boundary.

Validation passed 106 Rust tests, 260 qualifier tests, 16 CLI tests, 52 Node
tests with the pinned provider, and Clippy. The exact Jinja template also
matched four explicit Rust-test snapshots independently; 53 Jinja goldens were
generated, but no claim is made that all 53 ran against the compiled renderer.
All development commands used finite memory/CPU/time and thermal containment.

The new release was built on `exec-alternate-host` from clean commit `c9a2e75`, with
identical before/after snapshots of **139 compiled inputs**, including both
`.jinja` files. Its depfile and bounded build receipts were retained. validation-host
used a fresh isolated checkout under `$DEVELOPER_DIR/hag-history-rendering-20261008`.
The preparation audit passed 325 checks. This is observed off-host build
association, not a remote or reproducible-build claim.

- Binary SHA-256: `5ef15f6721c78380f66d3ab4edf6b55ddc6ddb64e607eb6e3d4279bcb2cd4697`.
- Compiled-source manifest: `56774a29385146ccd47c4050ac80ac81036434950bbe0e3d7d942f310ad5c130`.
- Build identity: `5d01979dd9cb9953e5afde18c8f40bc8e18b03d562bcf3961fbcf07eea932e92`.
- Prompt measurement: `90a8d4c3c89e11bec127258f8a0c34b037cc22f31f863f9dfeed321d4f626862`.

Execution used validation-host kernel `6.17.0-1024-oem`, boot
`REDACTED-HOST-ID`, the unchanged Qwen3-0.6B Q8_0 checkpoint
and committed Antimatter `78147b78d8271b993dc3734f469f763df1f0f548`.
All prior C2048 limits remained fixed: 2-GiB process cap, 1,536-MiB early stop,
zero swap, 75% CPU, 64 tasks, 8/4-GiB host start/stop headroom, ≤60°C for ten
seconds before launch, and the 90°C running cutoff sampled every 0.1 seconds.

| Stage | Cgroup peak bytes | Minimum available host bytes | Maximum Tctl | Result |
| --- | ---: | ---: | ---: | --- |
| Self-check | 16,801,792 | 21,573,214,208 | 49.125°C | Passed |
| CPU reference | 438,931,456 | 21,108,695,040 | 77.125°C | Passed |
| Native answers | 113,676,288 | 20,245,020,672 | 90.375°C | Thermal stop |

The CPU stage ran at 18:58:33.803957–19:01:50.778275 UTC. Its complete raw reply
was `<think>\n\n</think>\n\nready.`, with EOS stop, usage **491/6/497** and 497
evaluated positions. The HTTP request took 193.286 seconds. It unloaded and the
CPU server exited zero. Its service/cgroup and recorded processes retired before
the fresh 1,500-second root watcher armed at 19:02:26.871114 UTC.

Native answers passed the startup temperature gate. The cold M1559 request
returned HTTP 400 `context_length_exceeded` / `max_tokens`, leaving the model
unloaded and audit count zero. The valid four-message provider request followed.
At **37.810 seconds**, the unchanged thermal guard observed **90.375°C** and
terminated its process group. The native stage ran at
19:02:47.848013–19:03:28.996123 UTC; the guarded process exited −15 and the
supervisor exited 1. There was no accepted native report or runtime generation
audit; the final client log and response artifact were empty. Timing was not
started. Empty final artifacts do not prove that no GPU work ran.

The native trace began at 57.75°C and was near 74°C through 36.707 seconds,
then rose from 74.5°C to 90.375°C in about 1.10 seconds. Earlier project-answer
and timing runs on the same host/kernel/boot peaked at 74.625°C and 79.125°C.
Those runs used different prompts and the prior binary. The current receipts
lack synchronized host activity, GPU power/utilization and runtime-phase data,
so the cause of this sharp rise is unknown.

All recorded workload memory-pressure counters and swap remained zero. The
watcher stopped explicitly at 19:04:05.955076 UTC with zero dumps and an empty
root-owned private directory. The independent held audit passed **1,112 checks**:
three services/cgroups gone, six recorded stage/guard PIDs plus the watcher gone,
no remaining matching server/client processes or listeners, and complete CPU
and watcher kernel windows clear of GPU/reset/OOM diagnostics. This verifies
process retirement, not interrupted native reset, unload, GPU cancellation or
HAG budget release. Five remote and twelve local development services retired;
the SSH control connection was closed.

Held-audit SHA-256:
`d0330b31222ca0c1f11c711d120db63ca9ac09db0051bb54c1418e7cb9eaceea`.
Archive `outputs/hag-history-rendering2048-validation-host-thermal-held-20261008.tar.gz`
retains 294 individually verified files (136,592,889 uncompressed bytes),
including fresh build provenance, frozen templates/client, test and raw stage
receipts, license attribution and thermal closeout. It was verified again after
download without extraction. The checkpoint payload remains external.
Archive SHA-256:
`0c0bf7fd6ede1f2fed586a56a9785290a0bc589c560e277d93deae67670969d4`.
File-manifest SHA-256:
`551a424690bda7e33713c79cb5c87afffedfa2e3e72c4a2d6ca99175353ab4b7`.

Next, collect a finite passive host/thermal baseline and add bounded phase
receipts before a fresh single-answer diagnostic. Preserve this attempt and its
limits; timing still depends on a native parity pass. The CPU success and
model-free transport checks do not qualify a live two-turn conversation,
arbitrary history, semantic quality, tools, concurrency or throughput.

## Bounded thermal diagnostics prepared — 2026-10-11

**Instrumentation is implemented and tested; the fresh validation-host experiment has
not run.** SSH to the validation host returned `No route to
host`; local neighbor discovery also reported failure. The final bounded
connection observation was at 14:11:36–14:11:39 UTC. No remote process, crash
watcher, CPU reference or native model workload was started. The October 8
thermal-held attempt and its archive remain unchanged, and native parity for
the P491 history profile remains unqualified.

Commit `740f1c5` adds opt-in `HAG_SERVING_PHASES=1` runtime receipts and the
passive `tools/hag_host_telemetry.py` collector. Phase lines contain fixed
phase names, PID/request sequence, Unix and elapsed monotonic timestamps and
completed-position counts. They cover generation admission, cold metadata,
preparation, initial/final session reset, prefill and decode; position progress
is recorded after successful synchronous resident steps at the first position
and every 64 positions. Caps are 128 records per generation and 4,096 per
process. No prompt, model path, token ID or payload is logged. Missing ends do
not prove completion, session reset is not device reset, and synchronous
stderr writes can affect diagnostic timing.

Commit `0f4b23e` freezes the optional phase policy for `answers` in the exact
`antimatter-history-template-c2048` profile. Ambient phase settings are removed
from workload/client launch environments. CPU, self-check and timing do not
enable them. Child and outer stage receipts retain the resolved policy
separately; raw phase logs cannot replace the existing qualification evidence.
The P491/M128 workload, 618-position answer ceiling, memory/CPU/headroom limits,
startup cooldown, 90°C cutoff and all stage/client/watcher deadlines are
unchanged. See the [diagnostic procedure](hag-qualification.md#thermal-diagnosis).

Validation on `exec-alternate-host` passed 110 Rust runtime/server tests with HAG
enabled, 39 runtime tests with default features, 272 qualifier tests, 17 CLI
tests and 21 collector tests. Real-checkpoint tests remained ignored. Clippy
passed after replacing a manually expressed divisibility check; the initial
lint failure and its distinct successful rerun are both retained. Tests cover
phase ordering across success/cancellation/failure, record caps, strict frozen
policy and environment isolation, old CPU-reference schema compatibility,
exclusive bounded telemetry, malformed/missing sensors and PID reuse.

A ten-second passive smoke on this host captured ten samples with no skipped
intervals. It read Tctl, GPU activity/clocks/power, CPU/memory pressure and
numeric process activity; the absent fan attribute remained explicitly
unavailable. It did not launch inference. The collector ran in a separate
64-MiB, zero-swap, 5%-CPU, eight-task service with a 20-second service deadline
and 25-second outer bound. Its peak memory was 11,141,120 bytes and peak Tctl
48.5°C. The service/cgroup retired, and memory-pressure/swap counters were zero.
This validates collection on alternate-host; it is not a validation-host baseline or a thermal
cause finding.

The release was built from clean commit
`0f4b23ef20f6c555a90d1f910a288b918fbef20a`, with identical before/after
manifests of 140 compiled inputs, including `native_phase.rs` and the embedded
templates. The depfile and guarded build receipts were retained, along with
an exclusive copy of the 4,781,528-byte executable for the next attempt.

- Binary SHA-256: `e487e22255ab83657523d6e897e23ef9ced52215888e376d6713df988ede6f5b`.
- Source-manifest SHA-256: `6f436fffb5b0989d7c41a9c7e7693f83ca71512aab2f647a36215fdf4b8041ba`.
- Build-identity SHA-256: `c5529bf30e94417c830aaee29d47bf78407ba2dc2886292bd103e1d7d3b0ffab`.

All ten recorded development services and the separate passive collector
retired. Development runs kept the 2-GiB cap, zero swap, finite deadlines and
90°C thermal guard; no recorded memory-pressure events or thermal aborts
occurred. Receipts, the connection failure, plan and retained binary are under
the local task directory `work/hag-thermal-diagnostic-20261011`.

Next, restore validation-host connectivity, inventory its current host/kernel/boot,
freeze a new evidence directory and verify this build association there. Run
self-check and a fresh CPU reference, retire CPU fully, then arm fresh crash
capture. Collect a 60-second passive baseline followed by concurrent telemetry
around one instrumented native answer attempt in the unchanged private workload
cgroup. The collector must use its own bounded sibling cgroup. Retire both
services and stop/audit capture explicitly. Timing still depends on a native
parity pass; no threshold relaxation or automatic retry is authorized by this
diagnostic preparation.

## validation-host prefill thermal diagnostic — 2026-10-11

**Fresh CPU reference passed; native parity remains unqualified.** validation-host came
online and the prepared release ran once in the new isolated directory
`$DEVELOPER_DIR/hag-thermal-diagnostic-20261011`. Host `validation-host` was running
kernel `6.17.0-1024-oem`, boot `REDACTED-HOST-ID`.
The binary, 140 compiled inputs and observed off-host build association above
were verified after transfer; the remote checkout was clean at `0f4b23e`.
The pinned Antimatter client remains `78147b7`; no editor working-tree changes
were included. Preparation passed 363 independent checks. The task-local
telemetry supervisor passed 20 hardware-free tests before its inputs were frozen.

Self-check and a fresh CPU reference passed. The CPU response was the complete
raw `<think>\n\n</think>\n\nready.`, EOS `stop`, with usage 491/6/497 and
497 evaluated positions. CPU Tctl peaked at 72.125°C across 1,905 samples;
its cgroup peak was 439,554,048 bytes. CPU completion, service inactivity,
absent cgroup and absent child/guard PIDs were verified before fresh root crash
capture was armed at 14:44:45.550995 UTC.

The passive collector used its own 64-MiB, zero-swap, 5%-CPU, eight-task sibling
service. Its 62 baseline samples covered 61.2 seconds with Tctl 42.125–43.25°C,
GPU-busy readings of 0% and at least 22,046,576,640 available host bytes.
It retained 127 samples overall with no skipped intervals. The independent
0.1-second thermal guard and all native request/resource/deadline limits stayed
unchanged. Native answers started at 14:46:25.259696 UTC and stopped after the
guard observed **90.125°C at 50.300741 seconds**. The passive sampler later
observed 92.875°C during shutdown; that observation does not change the 90°C
stop threshold or prove device cancellation.

The 19 bounded phase records provide new localization:

- Preparation finished 0.520420 seconds after admission began, including
  metadata, memory admission, device open and resident preparation.
- Initial session reset completed and prefill started at 14:46:26.494843 UTC.
- Successful resident steps were recorded at positions 1, 64, 128 and 192;
  position 192 was recorded at 14:47:05.383673 UTC.
- No prefill-end, decode, request-complete or final-session-reset receipt was
  recorded. These sparse observations establish at least 192 completed
  positions, not an exact final evaluation/dispatch count, complete answer,
  runtime unload, budget retirement or GPU reset.

The rise from 72.25°C to 81.875°C in adjacent passive samples coincided with
host CPU activity rising from about 1.95% to 7.43%; the phase-identified server
remained near 29% of one core. Other numeric PIDs appeared among busy processes.
The retained sampler did not identify their workloads; the two later
high-activity PIDs checked were already absent. Correlation does not attribute the heat to either HAG
or another process. The HAG fence path uses a finite blocking `WAIT_CS` ioctl;
this evidence does not support changing it on a CPU busy-poll hypothesis.

The native cgroup peak was 113,889,280 bytes and minimum available host memory
was 20,135,809,024 bytes. Recorded memory-pressure/OOM counters and swap remained
zero. These process measurements are separate from device memory accounting.
The stopped watcher captured zero dumps. Complete CPU and watcher kernel
windows contained no matching driver faults. All three stage services and the passive
collector retired, and recorded child/guard PIDs were absent. A bounded scan
found no matching task server/client/watcher process or former listener; the
crash directory was empty. Process retirement does not establish successful
native reset, unload, cancellation or release of HAG's runtime budget.

The first independent audit retained two failed reconstruction checks: it omitted
the child's headroom minimum when reconstructing the outer result, and treated
the quoted phase prefix in disabled self-check policy metadata as an emitted
record. A separately named corrected audit passed **2,115 checks**, reverified
the failed audit and its immutable evidence, and left all qualification criteria
unchanged. Neither correction reran a model. Corrected audit SHA-256:
`0b74ff06502f03d3cf347dff0327940defe2e896003ef68ad2036500cafb92e1`.

Archive `outputs/hag-history2048-validation-host-diagnostic-20261011.tar.gz` retains
307 individually verified files (137,421,114 uncompressed bytes), including the
frozen build/client, raw phase/telemetry records and both audit versions. It was
verified again after download without extraction; the checkpoint payload stays
external. Archive SHA-256:
`4c20f7ecba6900e94f67f399ef42b77237f1bd3a4b6d308beacabe5ed3304689`.
File-manifest SHA-256:
`30e32797965dd41050c1d67a998445e37ad321d75398a243c07a36f442d3cfed`.
Local retained evidence is under `work/hag-thermal-diagnostic-20261011`.
The initial packaging wrapper failed before executing the packager because its
output directory was missing; that receipt is preserved, and a separately named
wrapper completed after creating the directory. Final checks retired all nine
remote and thirteen local recorded development services; no model retry ran.

Native answer parity and full-profile qualification remain false. Timing did not
run. The next diagnostic should bind numeric process identity/start time to
cgroups and collect the native leaf's CPU counters, so newly appearing host CPU
activity can be distinguished from HAG. Preserve this attempt and all safety
limits; do not infer a fence-policy fix or automatically repeat inference from
these observations. The system/Beholdd/Antimatter integration scope remains
unchanged.

## Quiet validation-host history parity — 2026-10-11

**The fresh single-answer diagnostic passed native CPU/HAG parity.** Nathan
identified the prior interfering activity as launching Overwatch and compiling
shaders, and reported that work finished. That user report prompted a separate
attempt; the previous thermal-held run and its archive remain unchanged.
The retained data does not reconstruct those earlier numeric PID identities or
provide a controlled causal heat comparison.

The new run used `$DEVELOPER_DIR/hag-quiet-diagnostic-20261011` on the same validation-host
host/kernel/boot, the unchanged verified release from `0f4b23e`, the same 140
compiled inputs, Qwen3-0.6B-Q8_0 checkpoint, pinned Antimatter `78147b7` client and
P491/M128/C2048 fixture. No runtime, shader, fence, admission, resource or thermal
limit changed. The identical supervisor retains its previous 20-test validation;
new task helpers were syntax checked and independently reviewed.

Preparation initially refused the user manager's global `degraded` status.
A separately named audit established that only Ubuntu's unrelated
`update-notifier-release.service` was failed, kernel journal access worked,
and our dedicated service had the actual required cgroup limits and
`NoNewPrivileges`. It also checked retirement of the four prior setup units.
The original refusal remains preserved; no unrelated unit was reset. The
corrected preparation passed 386 checks before self-check or inference began.

Self-check and the fresh CPU reference passed. CPU processes and the private
leaf retired before fresh root crash capture was armed. A separate bounded
passive collector then recorded 62 baseline samples over 61 seconds with Tctl
55.625–58.75°C, followed by concurrent native observations (206 samples total).
The unchanged running guard retained its 90°C cutoff and 0.1-second polling.

The genuine `Antimatter.chat` client made one admitted native request and
returned the same complete raw `<think>\n\n</think>\n\nready.` as CPU,
with EOS `stop` and usage **491 prompt / 6 completion / 497 total tokens**.
The runtime audit confirms **497 evaluations**, successful generation, healthy
session reset and no poisoned state. Cold and warm M1559 requests retained their
typed HTTP 400 refusal, and the model-free provider refusal sent no request.
Final unload and runtime budget retirement were verified.

| Stage | Maximum Tctl | Cgroup memory peak | Minimum available host memory |
| --- | ---: | ---: | ---: |
| CPU reference | 78.625°C | 890,322,944 bytes | 21,045,342,208 bytes |
| Native answers | 79.75°C | 467,345,408 bytes | 20,234,350,592 bytes |

Both stages had zero recorded memory-pressure/OOM events and zero swap.
These process measurements are separate from device memory accounting.
The native stage ran from 15:54:10.923540 to 15:56:23.659065 UTC. Its 29 phase
records show preparation completed at 0.704646 seconds, prefill ended at
126.697397 seconds with 491 positions, and decode and final session reset
completed at 128.667181 seconds with 497 positions. The client's elapsed
128.852307 seconds is descriptive; synchronous phase logging was enabled and
this was not a timing or throughput qualification.

The watcher stopped at 15:56:39.598417 UTC with zero dumps. Full CPU and watcher
kernel windows contained no matching driver faults. All three workload services,
the collector, recorded child/guard/client PIDs and task listeners retired.
All eight remote and two local recorded development services retired as well.
An independent audit passed **2,213 checks**, including unchanged source/input
identities, exact CPU/native response parity, refusal boundaries, reset/unload,
runtime budget retirement and cleanup. Audit SHA-256:
`9927cb8c72a2d84bc12cfe70783a5d1d7409f8db3d8a0302a62aae3f0a9922df`.

Archive `outputs/hag-history2048-validation-host-quiet-diagnostic-20261011.tar.gz`
retains 299 individually verified files (137,733,865 uncompressed bytes). It was
verified again after download without extraction; the checkpoint payload remains
external. Archive SHA-256:
`0c7af6146033cd0e0abe72d87264becfeaf207c480c7129042b2e1ea1d39c0f1`.
File-manifest SHA-256:
`fe9602a34c63184436ba81284546d60b577edcd6b4e4fb3b2525259b21571bae`.
Local evidence is under `work/hag-quiet-diagnostic-20261011`.

**Native answer parity is verified for this exact seeded-history fixture.**
Timing was deliberately excluded, so full-profile and performance qualification
remain false. This is not a live two-turn conversation or general history,
semantic quality, tool use, concurrency or operating-system integration kernel qualification.
Next, run the standard fresh four-stage profile with phase diagnostics disabled,
including timing after CPU/native parity, under the same limits. Preserve both
October 11 diagnostic archives and do not use their instrumented durations as
performance results.

## Standard validation-host history parity and response timing — 2026-10-11

**The fresh standard P491/M128/C2048 profile passed self-check, CPU reference,
native answer parity, timing and final closeout.** Phase diagnostics were
disabled: the opt-in policy key was absent from the frozen contract and stage
receipts, and no phase records were emitted. No passive collector ran.
Both earlier October 11 diagnostic attempts retain their original scope.

The isolated run used `$DEVELOPER_DIR/hag-history-timing-20261011` on
`validation-host`, Linux `6.17.0-1024-oem`, boot
`REDACTED-HOST-ID`. It reused the verified release from
Beholdd `0f4b23ef20f6c555a90d1f910a288b918fbef20a`, Qwen3-0.6B-Q8_0 and
Antimatter `78147b78d8271b993dc3734f469f763df1f0f548`. Binary SHA-256:
`e487e22255ab83657523d6e897e23ef9ced52215888e376d6713df988ede6f5b`.
The unchanged 140-file compiled-source manifest is
`6f436fffb5b0989d7c41a9c7e7693f83ca71512aab2f647a36215fdf4b8041ba`.
This is the retained observed off-host build association; no new remote build
or reproducible-build claim is made. Historical metadata-only prompt measurement
was reused after checking unchanged renderer/tokenizer dependencies; the fresh
CPU response independently confirmed P491.

The fresh preparation audit passed 383 checks. Its capability proof confirmed
the required cgroup controls and journal access despite the unrelated failed
Ubuntu update-notifier unit; no user unit was reset. Each workload kept the
2-GiB memory ceiling, 1.5-GiB early stop, zero task swap, 75% CPU quota,
8-GiB startup/4-GiB running host headroom, ten seconds at or below 60°C before
startup and the 90°C running cutoff. Request counts and deadlines were unchanged.

The complete CPU reply was `<think>\n\n</think>\n\nready.`, EOS `stop`, with
**491 prompt / 6 completion / 497 total tokens**. One native answer request
through the genuine provider, followed by one lazy-load chat and one warm SSE
request in the timing stage, matched those exact bytes, EOS and usage. Native
execution used **1,491 positions total**, including **994 timing positions**.
All three runtime receipts showed successful generation, healthy session reset
and no poisoned state. Final unload and runtime budget retirement passed.
Cold/warm M1559 requests returned the expected generation-free HTTP 400;
the provider's three model-free refusals made no fetch or preparation call.

| Stage | Maximum Tctl | Cgroup memory peak | Minimum available host memory |
| --- | ---: | ---: | ---: |
| CPU reference | 79.25°C | 438,919,168 bytes | 20,854,513,664 bytes |
| Native answers | 78.625°C | 469,819,392 bytes | 20,337,074,176 bytes |
| Timing | 82.875°C | 245,309,440 bytes | 20,347,715,584 bytes |

All workload memory-pressure/OOM counters and task swap readings were zero.
These cgroup measurements are separate from device memory accounting; the
initial host inventory reported 28 KiB of unrelated existing swap use.

The first request after lazy model load completed in **128.875385 seconds**,
excluding CLI startup. The warm stream produced its first nonempty raw content
at **126.656825 seconds**, its last content at **128.298370 seconds**, and
`[DONE]` at **128.623836 seconds**. Six content chunks reproduced the CPU reply;
twelve keep-alive comments arrived before content. First content includes
thinking markup. Streaming usage is associated with the matching nonstream API
response. These are one initial lazy-load and one warm client-latency sample at
75% CPU quota, including prefill; they do not measure decode throughput or
qualify performance.

CPU retirement preceded the fresh crash watcher, which ran from
16:14:10.068529 to 16:22:10.297425 UTC and stopped explicitly with zero dumps.
Native answer retirement was checked before timing. Full retained CPU and
watcher kernel windows contained no matching driver faults. All four workload
units/cgroups, eleven recorded stage/client PIDs and the watcher retired;
standard closeout also verified task listeners and runtime budget retirement.
All seven remote and three local recorded development units retired.

The independent audit passed **1,630 checks** with no corrections or model
retries. Audit SHA-256:
`ec09303527390ff3d225a894cf6b3b84a99cbdcd4f675fdb55f1550e0174a347`.
Archive `outputs/hag-history2048-validation-host-timing-qualified-20261011.tar.gz`
contains 307 individually verified files (138,203,120 uncompressed bytes).
It was verified again after download without extraction; the checkpoint payload
remains external. Archive SHA-256:
`3ce48ca22063e922ca066622003d3fa4894517bff95f3c5ecd941cafeeef7381`.
File-manifest SHA-256:
`507f7f3926cb696d473d9388d5a1e77aecef53e2a235c7d3824d6e13d604d256`.
Local evidence is under `work/hag-history-timing-20261011`.

**Full-profile qualification is true for this exact seeded-history backend
parity gate.** This does not qualify a generated first turn, UI history
persistence, retained KV, general semantic quality, tools, concurrency,
provider cancellation or the portable kernel transport. Next add a bounded
generated two-turn gate: measure and freeze both prompts, establish CPU replies,
then require exact native parity while feeding each actual first reply into its
second provider request. Preserve C2048, the existing safety limits and no-tools
scope before expanding the editor integration.

## Generated two-turn history CPU thermal stop — 2026-10-11

The new `antimatter-generated-history-c2048` profile at helper commit
`c2d6d29b46d43dc839b7d430cc4ce52d0689bc90` makes exactly two ordinary chat requests
per backend. Each second request carries its own backend's complete raw first
reply. CPU measures each concrete prompt before sending it; native execution
must establish full first-turn CPU parity and healthy reset before constructing
its second request. The profile has no timing/SSE stage, tools or phase diagnostics.
It preserves P467/M128 for turn one, P491-or-less/M128 for turn two, and the
1,212-position aggregate ceiling per backend.

All 301 hardware-free Python qualification tests, 19 CLI tests and 80 client
tests passed. The first Python run's fixture-selection failure and its correction
remain in the development receipts. The real provider stays pinned to Antimatter
`78147b78d8271b993dc3734f469f763df1f0f548`. The server executable remains
`e487e22255ab83657523d6e897e23ef9ced52215888e376d6713df988ede6f5b`, from observed
build `0f4b23ef20f6c555a90d1f910a288b918fbef20a`; all 140 compiled source files
match that build. No runtime rebuild, shader change or remote-build claim is
part of this attempt. The explicitly frozen metadata inspector is
`170e8d46f50d029044a93ac23d09800d665e41e408a9479b3331ad5b2551814a`, an observed
25,002,568-byte executable, not a newly attested reproducible build.

Preparation initially refused a short read-only inventory job because its
external cgroup sampler recorded no sample before the job finished. The original
failed audit is preserved as SHA-256
`4304b048e452ac4edff834a6da7e1dd46ad6e4c77d519fc54e34a54559c7ef1c`.
A separate metadata-only inventory captured actual limits and counters
synchronously before and after execution. The supplemental audit passed 187
checks, SHA-256
`f834016601ff5d595a89817444b3954df83704459a7d962d9d9e663a346b54c4`.
It did not fabricate the missing old sample or change any native policy.

Self-check passed. CPU turn one returned the complete raw
`<think>\n\n</think>\n\nready.`, EOS `stop`, and usage **467/6/473**.
That actual reply was appended to the second request; checkpoint metadata
measured **491 prompt tokens, 2,823 rendered bytes**, and a 618-position maximum
with M128. The second CPU request started but did not complete. The unchanged
thermal guard stopped it at **90.5°C**; CPU stage end was
**17:17:26.796962 UTC**. No second response, completed CPU reference, GPU request,
watcher session or successful full-profile closeout exists for this attempt.

The workload kept its 2-GiB hard ceiling, 1.5-GiB early stop, zero task swap,
75% CPU quota, 64-task limit, 8-GiB startup/4-GiB running headroom, ten seconds
at or below 60°C before startup and 90°C running cutoff. CPU task memory peaked
at **863,690,752 bytes**, with minimum available host memory **20,937,093,120
bytes**. Recorded memory-pressure/OOM counters and task swap stayed zero.

The unrelated `antimatter-mistral-native-build-20261011.service` entered active
state at **17:17:22 UTC**. A read-only sample at 17:18:13–14 observed four
`cc1plus` processes in that cgroup using approximately one CPU each, with Tctl
**100.125°C** after this task's CPU process had exited. The build had no CPU
quota. This establishes overlapping load and a plausible thermal contributor;
it does not isolate causality or establish a HAG kernel defect. No unrelated
service was stopped or changed.

The failed CPU and self-check units/cgroups retired. Six recorded child/guard/
inspector PIDs were absent, and an independent process scan found no surviving
frozen server, client or inspector. The CPU server's final process receipt was
interrupted by the thermal kill; retirement is therefore supported by the
inactive cgroup and process scan, not an invented clean server exit. The CPU
journal query exited successfully and returned `-- No entries --`; this is not
native crash-capture coverage, cancellation, clean reset or GPU-budget retirement.

Evidence packaging had two separate, preserved failures. Initial compression
at the development wrapper's 200% CPU quota reached 90.25°C and stopped.
Compression at a narrowed 25% quota stayed below 69.625°C, but the archive
inventory check caught an unexecuted cleanup helper added during its snapshot.
That helper was moved outside the evidence tree; the restored inventory and
every member of the completed archive were then verified before publication,
without recompression or another model request. Both partial archives and all
failed producer receipts remain retained. The original held proof is unchanged:
`dd0d9477575db0de1eef2dd634e60e69e2183249f4f84f8bb5a34aea0bdb014e`.

Archive `outputs/hag-generated-history2048-validation-host-held-v2-20261011.tar.gz`
contains **290 verified files, 161,753,118 uncompressed bytes**; the compressed
archive is 53,095,745 bytes. Archive SHA-256:
`57930411f33e348dfe3a0a9e50190ef76d0dc4beea21607145d1b9cb2fbb61ad`.
File-manifest SHA-256:
`570b3bd9ab9a7e92b302e61be3b8b2c44b9807629aa71c881648d0de24f3ee02`.
The manifest also pins the failed recovery receipts retained alongside the
archive. Local evidence is under `work/hag-generated-history-20261011`, with
remote evidence under `$DEVELOPER_DIR/hag-generated-history-20261011`. The checkpoint
payload remains external. Download verification checked all 290 members without
extraction and retained the unqualified outcome. All nine recorded remote and
nine local development units retired; the SSH control connection was closed.
Retirement does not reclassify their preserved failures as passes.

**Generated two-turn parity remains unqualified.** The earlier successful
seeded-history profile keeps its own scope. Next coordinate a quiet validation-host
window covering both CPU requests and the native stage, then start a fresh
attempt with the same workload and limits. Editor UI persistence, retained KV,
semantic quality, tools, concurrency, throughput and portable kernel transport
remain outside this gate.

## Quiet validation-host generated two-turn parity — 2026-10-11

A fresh `antimatter-generated-history-c2048` attempt passed self-check, CPU
reference, native answers and standard closeout. Antimatter finished its
validation-host benchmark and handed over the quiet window at 17:52:42 UTC; a separate
observation at 17:53:49 found its temporary service inactive with no processes
and Tctl 56.875°C. No unrelated service was stopped by this task. The earlier
thermal-stop attempt above remains preserved and unqualified.

The helper checkout was `b18b294966900356fbb6792c029b982ccb6e0e33`. All 140
compiled inputs still matched the retained `0f4b23e` build and `e487e222…`
server executable documented above. The same pinned Antimatter provider,
checkpoint and metadata inspector were used, with no runtime rebuild or model
retry. Fresh metadata preparation passed 42 checks, including synchronous
before/after observations of its actual private cgroup. Preparation SHA-256:
`4acfdcae5707af7e868baa59271459fe19ca641999657b841f1a52f762e59c3a`.

Both backends returned the complete raw `<think>\n\n</think>\n\nready.` on each
turn, with EOS `stop`. Turn-one usage was **467/6/473** and turn-two usage was
**491/6/497**. Each second request carried its own backend's actual first raw
reply, followed by the fixed follow-up message. CPU metadata measured the
resulting second prompt at 491 tokens and 2,823 rendered bytes. Native turn one
matched CPU text, usage and EOS and recorded a healthy reset before turn two
was constructed. Both native requests used the real pinned `Provider.chat`.
Each backend evaluated **970 positions**, below its 1,212-position ceiling.

CPU ran from 17:54:31.933727 to 18:00:53.606763 UTC. Native ran from
18:01:52.490957 to 18:06:06.965158 UTC. Peak Tctl was **79.75°C CPU / 82.625°C
native**. Recorded cgroup memory peaks were **455,974,912 / 391,901,184 bytes**,
with minimum host headroom **21,003,911,168 / 20,248,432,640 bytes**, respectively.
All recorded memory-pressure/OOM counters and task swap remained zero. The
2-GiB hard ceiling, 1.5-GiB early stop, 75% CPU quota, 64-task limit, 8-GiB
startup/4-GiB running headroom, ten seconds at or below 60°C before each stage
and 90°C running cutoff were unchanged. Phase diagnostics remained disabled;
there was no timing/SSE stage or performance qualification.

CPU retirement preceded the fresh read-only crash watcher. It armed at
18:01:30.898114 and stopped explicitly at 18:06:18.011671 UTC with zero dumps.
The retained full CPU and watcher kernel windows contained no matching faults.
Standard closeout verified healthy resets, final unload and runtime budget
retirement, all three workload units/cgroups, and ten recorded stage/server/
client PIDs. Independent packaging additionally verified both inspector PIDs
were absent and replayed the raw generated-history evidence. Closeout SHA-256:
`e6fce373d9fe4c078afcf5afff1dfe20b93ef2add5a349deef8eb056a0f8ac96`.

Archive `hag-generated-history2048-validation-host-quiet-qualified-20261011.tar.gz`
contains **285 verified files, 163,014,926 uncompressed bytes**, and is
59,917,429 bytes compressed. Archive SHA-256:
`5af9137b5465c521cd68c5c42deaf3864660fe61e7cd282911150ae228d90557`.
File-manifest SHA-256:
`b3e0f14e4dc842073e194021851a6e514ebc4f70752c2e936b440d53fe1cacf4`.
Packaging ran at 25% CPU, peaked at 68.625°C, and kept every producer output
outside the closed evidence tree. No evidence member changed during packaging.
After download, the local archive and sidecars were returned to validation-host for
another complete member/hash verification without extraction or model execution;
the workstation was busy and too hot for guarded local verification. That
verification passed at a maximum 60.375°C. This is round-trip verification of
the downloaded copy, not a verifier executed on the workstation.

Local evidence is under `work/hag-generated-quiet-20261011`, with the downloaded
archive and remote producer receipts in `remote-outputs/`. Remote evidence is
under `$DEVELOPER_DIR/hag-generated-quiet-20261011`. The checkpoint payload remains
external. All three remote development units and the one local development unit
retired, as did the watcher; the SSH control connection was closed and the
coordinated validation-host window released.

**The exact generated two-turn backend-parity profile is qualified.** UI history
assembly and persistence, retained KV, general semantic quality, tools,
concurrency, cancellation, throughput and portable kernel transport remain
outside this proof. The next small editor integration slice is a model-free
regression through Antimatter's actual conversation-history assembly, checking
that the captured assistant reply reaches the next provider request in the
correct order with the expected output budget.

## Antimatter conversation-history replay — 2026-10-13

Antimatter commit `0753e94b72d348ddfed4330c617ac9803caa3e8a`, based on
`c7f9efeac6a2961d61b2f8cd7927f41c5ce15590`, adds a model-free conversation-history
regression. The renderer's existing visible-history snapshot and the host's
initial grounding/profile/budget preparation are extracted into shared helpers
and used by the desktop itself, without changing their behavior. Post-compaction
messages still take precedence in the host's final request.

The committed fixture contains the two native request/response pairs from the
qualified archive above, retaining its SHA-256 and each source member's hash.
Extraction verified the archive and selected members before decoding the replies.
The live provider qualification remains pinned to Antimatter `78147b7…`; this
newer editor revision has replay evidence, not a fresh live qualification.

Five new tests cover warm, disk-reloaded and missing transcript caches, attachment
serialization, and replacement of stale system grounding and caller output limits.
Each two-turn case resolves a real C2048/M128 profile, renders the system prompt,
uses actual transcript storage, and runs the real provider and tool loop with an
empty connected-tool registry. Only HTTP transport is replaced with the captured
ordinary JSON replies. The first raw `<think>\n\n</think>\n\nready.` reply must
reach the next request in system/user/assistant/user order, with one system
message, exact captured content, no compaction/overflow, and `max_tokens: 128`.
The renderer leaves its output limit unset; the host applies the resolved profile.

The desktop request uses `stream: true` and omits temperature, while the live
capture used `stream: false` and temperature zero. The test asserts that complete
expected wire body and exercises the provider's ordinary-JSON streaming fallback.
It does not exercise SSE parsing, GUI submission, IPC event delivery, a full
application restart, connected tools or the engine's default-temperature behavior.

All **135 focused tests passed**: five new history cases and 130 existing
transcript, profile, context, system-prompt, provider and tool-loop tests.
The normal `tsc -b --pretty false` check failed with **22 detached-window
diagnostics**. An isolated copy of the base commit plus the two pre-existing
layout/preload edits produced exactly the same 22 diagnostics using the same
dependencies. No new diagnostics were introduced by this change; this is not a
passing full typecheck. The unrelated layout/preload edits remain uncommitted by
this task. `git diff --check` passed for the committed changes.

Local development receipts are under `work/hag-editor-history-20261013/outputs`.
The first two test runs exposed harness dependency resolution and absent tool
registry setup; both were corrected and their failed receipts retained. The first
baseline comparison lacked the vendor directory and reported one extra missing
module; the corrected comparison and original failure are both retained.
The successful test receipt is `focused-tests-v3.log`; exact baseline comparison
is in `typecheck-comparison-v2.json` and retirement in `development-retirement.json`.

Development jobs kept a 25% CPU quota, 1.5-GiB memory-high/2-GiB hard limit,
zero task swap, 128-task limit, 180-second runtime cap, 8-GiB startup/4-GiB running
host headroom and a 90°C thermal cutoff. Recorded memory-pressure/OOM counters
and swap stayed zero. Maximum sampled task memory was 639,016,960 bytes, minimum
host headroom 17,234,968,576 bytes, and peak Tctl 88.125°C. All seven development
units/cgroups and recorded child processes retired. No model or GPU workload,
server rebuild, remote experiment or change to native safety limits occurred.

This completes the scoped production message-assembly and transcript-restoration
regression for HAG's Antimatter/Beholdd integration. Next add one model-free
two-turn UI/IPC test: submit, deliver content/completion events, verify the stored
assistant content, then submit again and inspect outgoing history. Retained KV,
general quality, tools, concurrency, performance and portable kernel transport
remain separate work.

## Desktop UI/IPC smoke: final verification held — 2026-10-13

Antimatter `5d9182da877e2b0d50fa6d48c137d38b5fc93217` adds the optional
`npm run test:hag-ui` check. It bundles the production App, preload and
conversation store, submits through the real composer DOM, and handles actual
Electron IPC in a small synthetic host. It uses an isolated temporary profile,
blocks renderer network access and disables hardware acceleration. No production
runtime file is changed. The real inference host, provider/SSE parser, model
execution and native HAG backend are outside this check.

The first working-tree run passed in **11.041 seconds**. Two `chat:stream` invokes
carried the expected user then user/assistant/user messages, with raw captured
reply text and response IDs preserved. Three content chunks per reply passed
through the preload and App; an unrelated response ID and separate reasoning
text did not contaminate history. The host held each invoke pending until the
streamed reply reached a real conversation-store checkpoint, then resolved it
and verified completion metadata, released composer controls and final disk
content. The renderer left its output limit unset, as expected for host planning.
This run included the two pre-existing layout/preload edits as hashed inputs.

Review then added explicit visible-transcript and first-user-ID checks, fixture
hashing and timeout escalation. A separate temporary source snapshot from
`0753e94` plus this task's four files excluded the unrelated layout/preload edits.
Its verification stopped at **90°C after 3.310 seconds**, before a result was
produced. The changed assertions and clean-source test therefore remain
**unverified**; the earlier working-tree pass does not qualify them.

Both runs used the retained 25% CPU quota, 1.5-GiB memory-high/2-GiB hard ceiling,
zero task swap, 128-task limit, 180-second runtime cap, 8-GiB startup/4-GiB running
headroom and 90°C thermal cutoff. Sampled task memory peaks were **862,339,072 /
137,076,736 bytes**, respectively. The initial pass peaked at 86°C. All recorded
memory-pressure/OOM counters and task swap stayed zero. Both development units,
cgroups and recorded child PIDs retired; the failed attempt remains a failure.

Later observation found the workstation near 98°C with other active workloads.
Validation-host SSH returned `No route to host`. A local reading fell to
80°C, but a 45-second check could not establish ten seconds at or below 80°C and
ended at 94.5°C. No subsequent test or model run was launched.

Receipts are under `work/hag-editor-ui-20261013/outputs`, including the initial
report/source hashes, both bounded-run logs, `initial-retirement.json` and
`cool-window.json`. The interrupted private source snapshot is retained separately.
Next rerun the committed smoke from a clean source snapshot in a stable quiet
window, preserving the limits and failed receipts. The existing backend parity
and model-free provider/history qualifications retain their original scope.

## Desktop UI/IPC smoke verified — 2026-10-13

A fresh clean-source run of `npm run test:hag-ui` passed on the workstation with
Electron 44.3.0. It used an exact Git archive of Antimatter
`5d9182da877e2b0d50fa6d48c137d38b5fc93217`, with the existing installed dependencies
and vendor tree linked into a private temporary source directory. No working-tree
source overlay was applied. The compiled-input hashes for layout, preload, the
test script and captured JSON fixture matched that commit. This closes the final
verification hold above; the earlier failed attempt is preserved separately.

All strengthened assertions passed: two real composer clicks and `chat:stream`
invocations, visible assistant replies, exact raw captured content in the second
user/assistant/user request, stable first-user and assistant IDs, same conversation
identity, no outgoing empty assistant placeholder, and no renderer output limit.
Each reply arrived through three real preload IPC content events. Unrelated-run
content and separate reasoning text stayed out of history. The composer remained
busy while its invoke was pending, then released after successful completion.
Seven real conversation-store checkpoints were observed, and the final four
messages were read back from the private JSON file.

validation-host remained unreachable, so no remote run was used. The workstation first
held ten seconds at or below 80°C, ending at 70.125°C. The bounded verification
took **12.009 seconds** and **2.688 CPU seconds**, with peak Tctl **75.625°C**,
maximum sampled task memory **863,977,472 bytes**, and minimum host headroom
**17,542,107,136 bytes**. All sampled memory-pressure/OOM counters and task swap
were zero. The existing 25% CPU quota, 1.5-GiB memory-high/2-GiB hard ceiling,
zero swap, 128-task limit, 180-second runtime cap, host headroom limits and 90°C
cutoff were unchanged. The development unit/cgroup retired, its recorded child
PID was absent, and the temporary source snapshot was removed.

Evidence is under `work/hag-editor-ui-closeout-20261013/outputs`, including
`report.json`, `source-hashes.json`, `clean-source.json`, bounded/thermal receipts,
`retirement.json` and the `evidence-files.json` inventory. Report SHA-256:
`dd93c1a9389b0925244f457ca6e75674fe70f7b9f57fa716c8fe62e6b6dffec3`.
Source-hash manifest SHA-256:
`7b932391825eda3e5eca4c66d30bd412e5bd1446edbc5c4b2079967f7a460af8`.
The private Electron fixture output remains at
`/tmp/antimatter-hag-conversation-agDcFx`; the report and source hashes were copied
to the retained evidence directory. No model, GPU inference or new runtime build
was involved. The unrelated working layout/preload edits remain untouched.

This verifies the scoped production UI/preload/conversation-store path. The chat
host is synthetic: production main dispatch, grounding/output planning,
provider/SSE handling and Beholdd/HAG execution are outside this test. Network
blocking applies to the Electron session; persistence is checked by file readback,
without restarting the application. Next reload the renderer after turn one and
submit turn two from the hydrated store, preserving the exact content and IDs.

## Renderer reload history verified — 2026-10-13

Antimatter `4d1eb0f412c23fdc28f403273396f0a3ecd72e31` adds the optional
`npm run test:hag-ui -- --reload-between-turns` variant. It reloads the actual
renderer after the first completed reply, waits for disk hydration and submits
the second turn from the restored conversation. Only the smoke script and README
changed; production runtime files are unchanged.

Both reload and original warm cases passed from separate private Git snapshots
of `5d9182d` with only those two files overlaid. Recorded source hashes match the
new commit, and compiled layout/preload/fixture inputs match the clean base.
The two unrelated layout/preload working edits were excluded and remain untouched.

The reload check proved a new renderer world by removing a marker through a real
`webContents.reload()`. The host clears its conversation cache at each load
boundary, preventing a before-unload save from turning restoration into a cache
hit. Hydration calls increased from two to four (React StrictMode); fresh loads
returned the saved active conversation and exact messages from disk. Typing a
draft only after reload produced an identifiable incoming renderer checkpoint.
That payload preserved message IDs, roles, raw text, creation/completion times,
chunk counts and failure state. Inspecting the incoming payload prevents host
conflict handling from concealing damaged renderer history. Reload and drafting
left the request count at one; the second real composer click sent the expected
user/assistant/user history with stable IDs and no renderer output limit.
Reload recorded ten checkpoints; the default warm case retained seven. Both
finished with exactly two requests and the expected four persisted messages.

The workstation first held ten seconds at or below 80°C, ending at 47.5°C.
Reload/warm runs took **10.868 / 10.188 seconds** and **2.289 / 2.196 CPU seconds**.
Peak Tctl was **53.625 / 54.25°C**, sampled task memory peaked at
**537,309,184 / 501,497,856 bytes**, and minimum host headroom was
**22,881,878,016 / 22,922,153,984 bytes**. All sampled memory-pressure/OOM counters
and task swap stayed zero. The retained 25% CPU quota, 1.5-GiB memory-high/2-GiB
hard ceiling, zero swap, 128-task limit, 180-second runtime cap, host headroom
thresholds and 90°C cutoff were unchanged. Both units/cgroups retired, recorded
child PIDs were absent, and temporary source snapshots were removed.

Evidence is under `work/hag-editor-reload-20261013/outputs`: each case has its
report, source hashes, clean-source identity and console log, alongside bounded
memory/thermal receipts, `retirement.json`, `commit-verification.json` and the
`evidence-files.json` inventory. Reload report SHA-256:
`76e141884004ffc8130e42e41d26dd2f0cdffa2c1c1403f5fd06d3f82d825581`.
Warm report SHA-256:
`8758d2e3ce6eedc708147870f62808da720a2a48d541bf99e2f7a7ff1481565c`.

This remains a model-free production App/preload/conversation-store check with
a synthetic chat host, renderer-session request blocking and hardware acceleration
disabled. The host stays running: full application restart, production main
dispatch, provider/SSE transport and Beholdd/HAG inference are outside its scope.
Next cover restoration across a complete fixture application restart, keeping the
same bounded workload and exact outgoing-history assertions.

## Fixture application restart verified — 2026-10-13

Antimatter `7b166afdbf692f25396ada139c6cbe62cb74d110` adds
`npm run test:hag-ui -- --restart-between-turns`, mutually exclusive with renderer
reload. Only the smoke script and README changed. Restart, reload and original
warm modes passed from separate private snapshots of `4d1eb0f` with those two files
overlaid; recorded hashes match the new commit. The unrelated layout/preload edits
were excluded from compiled inputs and remain untouched.

The restart launcher builds once, then starts two Electron processes sequentially
against the same private profile, sharing one 90-second execution deadline. After
turn one's completed checkpoint, the first process exits; the launcher waits for
successful `close`, checks disk JSON against the first report, and starts phase
two. Distinct PIDs/instance IDs and close/start times prove the process boundary.
The second process's startup file hash matches the file read between launches.
Its new conversation store and renderer restore themselves from disk; the first
report supplies expected assertions only, never hydration data.

Phase two starts with no live requests. Fresh load payloads and an incoming
renderer checkpoint tagged by a newly typed draft preserve active conversation,
message IDs/roles/raw content, creation/completion times, chunk counts and failure
state. Startup and drafting send no request. The second actual composer click
sends the expected user/assistant/user history with the original IDs and no
renderer output limit. Each phase makes exactly one request; together they record
nine checkpoints and four hydrations, ending with four persisted messages.
Reload and warm regressions retain ten/seven checkpoints and four/two hydrations.

| Mode | Runtime | CPU time | Peak Tctl | Sampled memory peak |
| --- | ---: | ---: | ---: | ---: |
| Restart | 12.844 s | 2.723 s | 52.875°C | 867,520,512 bytes |
| Reload | 11.253 s | 2.373 s | 54.375°C | 549,675,008 bytes |
| Warm | 10.459 s | 2.287 s | 57°C | 794,996,736 bytes |

Startup followed ten seconds at or below 80°C, ending at 45.25°C. Minimum host
headroom across the runs was 22,401,609,728 bytes. All sampled memory-pressure/OOM
counters and task swap stayed zero. The retained 25% CPU quota, 1.5-GiB memory-high/
2-GiB ceiling, zero swap, 128-task limit, 180-second outer service cap, host headroom
thresholds and 90°C cutoff were unchanged. All three units/cgroups retired; recorded
supervisor and Electron PIDs were absent, and temporary source snapshots were removed.

Evidence is under `work/hag-editor-restart-20261013/outputs`, with reports/source
hashes for all modes, both restart phase reports, bounded memory/thermal receipts,
`retirement.json`, `commit-verification.json` and an `evidence-files.json` inventory.
Restart report SHA-256:
`d5323a29842beb6e5c5b7a447115556b0f8a000e27faa0b120f239bf41a0c7c9`.

This verifies completed conversation persistence across a full fixture process
restart. It does not exercise production main-process services or graceful shutdown
hooks: the fixture exits after the completed checkpoint. App/preload/store are real,
the chat host is synthetic, renderer-session requests are blocked and hardware
acceleration is disabled. Provider/SSE transport and Beholdd/HAG inference retain
their separate qualification scope; no model or GPU inference ran here.

Next join the actual UI/IPC/store smoke to `prepareInteractiveRequest`,
`runToolLoop` and `OpenAICompatibleProvider` using an isolated host adapter and
controlled SSE responses from the captured reply text. Existing history tests
cover provider JSON fallback, and provider tests cover SSE parsing; the new value
is checking their connection to real composer events and persisted second-turn
history, including the host-resolved 128-token output budget. This can remain
model-free without booting all production main-process services.

## Desktop provider SSE pipeline verified — 2026-10-13

Antimatter `2d675e3a44aabc28b49c1c341927f60ef83ae92f` adds `--provider-sse`,
composable with either lifecycle flag; for example,
`npm run test:hag-ui -- --provider-sse --restart-between-turns`.
A fixture host adapter uses production `prepareInteractiveRequest`, `runToolLoop`,
`OpenAICompatibleProvider`, model profile resolution and transcript persistence.
Only the smoke script, new test fixture and README changed. Runtime sources are
unchanged; the pre-existing layout/preload working edits remain untouched.

The adapter selects private profile/transcript paths and an empty tool runtime.
It replaces fetch only within the isolated fixture host, rejects unexpected URLs
or extra requests, and returns synthetic fragmented SSE derived from the qualified
raw JSON replies. It checks the complete outbound body against the capture with
`stream: true` and temperature omitted, preserving the exact system/history bytes
and `max_tokens: 128` at C2048, without extra fields. This is not a capture of live
Beholdd SSE or a new server qualification.

Each reply uses twenty transport fragments carrying role, separate reasoning,
three content deltas, a terminal stop frame with captured usage, and `[DONE]`.
The real provider/tool-loop callbacks deliver content and reasoning through
preload IPC to the App. The smoke holds terminal SSE until the visible reply and
streaming disk checkpoint exist, asserts the composer remains busy with no
completion timestamp, then lets the provider/loop resolve the IPC invocation.
The completed reply has the expected raw text, three chunks and successful
completion metadata. Separate reasoning and unrelated response IDs never enter
history. Each loop finishes in one round with zero tool calls.

All three SSE lifecycle cases made exactly two provider requests. Turn one reports
no resumed transcript; turn two must resume it. In restart mode, distinct Electron
processes run sequentially against the same private stores, so the second resume
loads the persisted host transcript. Existing fresh-renderer, incoming-checkpoint,
raw history, stable ID and final conversation-file assertions also passed.

| Mode | Runtime | CPU time | Peak Tctl | Sampled memory peak |
| --- | ---: | ---: | ---: | ---: |
| SSE restart | 13.317 s | 2.809 s | 52.5°C | 820,318,208 bytes |
| SSE reload | 11.830 s | 2.501 s | 53.75°C | 678,469,632 bytes |
| SSE warm | 10.836 s | 2.376 s | 55.125°C | 553,148,416 bytes |
| Direct IPC warm | 9.908 s | 2.148 s | 52.375°C | 567,885,824 bytes |

Each run used a separate clean snapshot of `7b166af` plus the three changed files,
whose hashes match the new commit. The snapshots included `mcp/` for transitive
build imports; no MCP tools or services were started. Startup followed ten seconds
at or below 80°C, ending at 45.875°C. Minimum host headroom was 21,829,038,080 bytes.
All sampled memory-pressure/OOM counters and task swap stayed zero. CPU/memory/
swap/task/runtime/headroom limits and the 90°C cutoff were unchanged. All four
units/cgroups and recorded supervisor/Electron PIDs retired; private source
snapshots were removed. No model or GPU inference ran.

Evidence is under `work/hag-editor-provider-sse-20261013/outputs`: per-mode reports,
source hashes, both restart phase reports, copied final profile/transcript/
conversation files for SSE modes, bounded receipts, `retirement.json`,
`commit-verification.json` and the `evidence-files.json` inventory. SSE restart
report SHA-256:
`5694e0ac5b7d9a1ecf029185f69aec29828ccf6e9edf3deb49e1c26809c9b7bc`.

This joins the production UI/preload/store and request/provider/transcript paths
through a small fixture adapter. Production `streamChat`/main-entry services,
graceful shutdown, UI metrics forwarding, live HTTP and Beholdd/HAG execution
remain outside its scope. Next cover cancellation and stream failure through this
connected path, checking partial-history persistence and recovery on the next turn.

## Interrupted desktop conversation recovery verified — 2026-10-13

Antimatter `533ba65c8674b5c8d519050eb60d9dd586b45851` adds
`--interrupt-first=cancel` and `--interrupt-first=error`, requiring `--provider-sse`
and composable with restart/reload. Only the smoke script, host fixture and README
changed. The first reply uses the captured raw text without its final period;
the second reply completes normally. No production runtime code changed.

Cancellation clicks the actual Stop composer button, reaches preload `chat:cancel`,
and aborts the signal passed through the real tool loop/provider to fetch. The
mock transport wakes its pending read with AbortError. The loop checkpoints its
partial transcript; the fixture adapter mirrors `streamChat` by sending the stop
marker and resolving IPC. The renderer persists the prefix plus `\n\n_Stopped._`,
four chunks (three content chunks plus marker), completion time and `failed: false`.
Exactly one matching cancellation is accepted; no terminal SSE frame is released.

The failure variant sends a fragmented SSE error envelope while leaving its body
open. The provider rejects and cancels the body, the loop checkpoints partial
history, and the adapter rethrows. The actual renderer preserves the prefix,
appends its error notice and records three chunks, completion time and
`failed: true`. Exactly one expected provider error and matching renderer console
error are allowed; unrelated errors still fail the smoke. Both paths assert
closed transport, removed abort listeners, settled loop and idle host state.

Warm and fresh-process restart recovery both passed. Each made exactly two actual
composer/provider requests. Turn two resumes the persisted host transcript,
retains original renderer IDs and completes successfully. Visible history contains
the partial prefix plus stop/error notice. Host resume represents the recovered
notice as a separate assistant row, which the provider coalesces with its normal
two-newline separator; assertions distinguish this wire form from visible text
and require the notice exactly once. Restart retains the independent process,
disk hash and incoming restored-checkpoint proofs; both restart cases recorded
nine checkpoints/four hydrations, and both warm cases seven/two.

| Final case | Runtime | Peak Tctl | Sampled memory peak |
| --- | ---: | ---: | ---: |
| Cancel + restart | 12.889 s | 55.5°C | 526,905,344 bytes |
| Error + restart | 13.437 s | 67.875°C | 566,427,648 bytes |
| Cancel + warm | 11.238 s | 73.25°C | 561,668,096 bytes |
| Error + warm | 10.632 s | 65.25°C | 509,419,520 bytes |
| Normal SSE reload | 11.774 s | 60.75°C | 575,332,352 bytes |
| Normal direct IPC warm | 9.829 s | 60.75°C | 531,718,144 bytes |

All six final cases ran from separate clean snapshots of `2d675e3` plus this
commit's three files; recorded hashes match the commit and exclude unrelated
layout/preload edits. An initial cancellation/restart pass remains preserved
separately; it predates stricter option parsing and was rerun above. Five invalid
option combinations now fail before build or Electron launch. Startup followed
ten seconds below 80°C. Minimum host headroom across all seven runs was
22,447,267,840 bytes. All recorded memory-pressure/OOM counters and task swap
stayed zero; CPU/memory/runtime/headroom limits and the 90°C cutoff were unchanged.
All seven units/cgroups and recorded supervisor/Electron PIDs retired, and private
source snapshots were removed. No model or GPU inference ran.

Evidence is under `work/hag-editor-recovery-20261013/outputs`, including per-case
reports/source hashes, restart phase reports, copied final host state, bounded
receipts, `invalid-options.json`, `retirement.json`, `commit-verification.json`
and `evidence-files.json`. Final cancellation/restart report SHA-256:
`5938d2593cc5b379727737285e00cd3e93fff5b54c23d6f7a38519ba469dd333`.
Error/restart report SHA-256:
`a6081bb770d200908df6d0a03888f7a5a7a3a3036e6f0a6be853b415a3cbc1ed`.

The adapter mirrors the main cancellation/error boundary, rather than booting
production `streamChat`. Main services, live HTTP, server cancellation and native
HAG execution remain outside this test. Next add a bounded live two-turn UI/provider
profile, preferably on validation-host after fresh host/kernel/resource and input-identity
checks. Preserve the generated-history C2048/M128 inference limits, but record the
current desktop wire contract separately from the older pinned proof. Use the live
request deadlines, not this model-free fixture's 90-second budget, and record the
actual response transport instead of assuming live SSE support.

## Live editor HTTP client prepared — 2026-10-13

Antimatter `8a0f8dce59fe7e5d71777fd4f709f5d70f67c15e` adds
`--live-base-url=http://127.0.0.1:PORT/v1` and optional `--prepare-live` to the
actual App/preload/conversation-store smoke. The separate live host uses production
preparation, tool loop, OpenAI-compatible provider and transcript APIs. It fixes
C2048/M128 and the current desktop wire contract: `stream: true`, omitted
temperature, and no connected tools. A literal loopback IP and non-default
explicit HTTP port are required. It never starts a server or creates a tunnel.

Admission is limited to two ordered turns and one delegated HTTP attempt per
turn. Exact body, route and method checks precede delegation; redirects and
loading retries cannot generate additional traffic. The first actual raw answer
must pass the retained reply/usage/EOS oracle before another turn is accepted.
Turn two includes that observed answer, with original user/assistant/thread IDs,
and resumes the real host transcript. A failed or cancelled turn closes further
admission. This is client-side admission only: server prompt inspection and
CPU/native reset/parity checks remain external work.

The client records bounded raw JSON or SSE transport (at most 1 MiB), HTTP status,
content type, fragment/byte counts, served model, stop reason and token usage.
Production metrics and content callbacks must agree. The real renderer persists
matching raw history, IDs, callback counts, completion times and non-failed state;
incoming renderer checkpoints are checked as well as the host's saved copy.
There is no synthetic release of live content or completion. Each request's
240-second deadline covers headers and the entire body. A single 355-second
desktop deadline covers both turns, including prepared clients launched directly.
This is a native-stage client budget; it cannot replace the CPU stage's longer
budget (the retained CPU run took about 382 seconds).

Preparation builds outside the eventual live stage and saves a copied launcher,
bundles, esbuild metadata, all bundled source/dependency-input hashes, lockfile
hash, and `prepared-live.json` with artifact hashes/launch settings. Preparation
contacts no endpoint. A prepared client completed both turns after deleting its
temporary source checkout, verifying that runtime execution uses saved artifacts.
An external supervisor must still verify their identities, freeze the Electron
distribution, enforce containment and admission, and close out processes.

The report names candidate profile `antimatter-ui-history-c2048` and explicitly
sets `backendQualified: false` and `modelExecution: external-endpoint-unverified`.
The Beholdd qualification dispatcher does **not** yet register this profile.
Renderer networking remains blocked and hardware acceleration disabled; only the
host's explicit loopback transport is enabled. Live mode rejects replay,
interruption, reload and restart flags rather than mixing evidence contracts.

Fifteen focused tests passed using ephemeral loopback HTTP fixtures: two-turn
JSON/SSE, changed reply/usage/EOS, changed prompt and serialized output budget,
assistant identity, HTTP 503 without retry, HTTP 302/307 without following,
oversized response rejection, endpoint validation, cancellation and a mocked
240-second deadline after headers/content while the body remains open. Both
termination cases close the actual socket and forbid a second request. Sixteen
malformed CLI combinations fail before build or Electron launch.

| Final UI case | Runtime | Peak Tctl | Sampled memory peak |
| --- | ---: | ---: | ---: |
| Live-mode JSON fixture | 9.967 s | 70°C | 756,060,160 bytes |
| Live-mode SSE fixture | 9.501 s | 70.25°C | 483,717,120 bytes |
| Prepared artifact launch | 9.494 s | 72.625°C | 380,624,896 bytes |
| Wrong first reply, rejected after one POST | 8.221 s | 67°C | 495,906,816 bytes |
| Direct IPC warm regression | 9.174 s | 71.125°C | 471,441,408 bytes |
| Provider SSE restart regression | 12.660 s | 70.375°C | 489,623,552 bytes |
| Provider error/restart regression | 12.646 s | 79.5°C | 556,646,400 bytes |

All checks ran from clean snapshots of `533ba65` plus the four owned changes.
Runtime source hashes match the committed version; unrelated working layout and
preload edits were excluded and preserved. One initial unit harness failed module
resolution before testing, and one restart smoke passed but its source-freeze
check detected a concurrent test-only edit. Both corrected runs passed; failed
development receipts remain preserved. Across all 15 bounded runs, minimum host
headroom was 20,466,651,136 bytes and sampled memory peaked at 765,554,688 bytes.
Recorded memory-pressure/OOM counters and swap stayed zero. All units/cgroups
and recorded supervisor/client/Electron PIDs retired, and source snapshots were
removed. Development limits stayed CPU25%, Tasks128, MemoryHigh1536M,
MemoryMax2G, swap0, 180/190-second service/outer deadlines, startup8GiB/running4GiB
headroom and 90°C cutoff. Startup had ten seconds below 80°C. These fixtures used
no model or GPU inference and did not change the retained live inference limits.

Evidence is under `work/hag-editor-live-client-20261013/outputs`, including raw
HTTP request/response records, reports and source hashes, prepared manifests,
copied conversation/transcript state, bounded receipts, `retirement.json`,
`commit-verification.json` and `evidence-files.json`. Final JSON report SHA-256:
`822f11af67eb96897c727f0f84c86e45f54ba848f2106b67b895f6e8939f165d`.
Final SSE report SHA-256:
`1d5e8c062c65953566c8283c5ad30c8c9bb66e74da4794e01c2ecc6dc2d413ce`.
Prepared-launch report SHA-256:
`328eed2cd44ed96b6b6d86ed5fa9d70c24adf05f88fa0064dc2c090eb949af39`.

The validation host returned **No route to host** on both SSH checks; the server
notes describe an older network arrangement and cannot establish its current
identity. No live backend run was attempted. Next implement a separate UI
admission/closeout path, preserving the existing generated-history recipe and
limits: first prompt467, second at most491, M128, per-turn evaluation594/618 and
two-request total1212. Require current-wire CPU reference, native first-reply
parity and healthy reset before turn two, complete transport/raw-history evidence,
fresh host/input/runtime identities, resource/thermal containment, crash/kernel
checks and all process/cgroup retirement. Use local isolated Electron through an
explicit SSH loopback tunnel once validation-host's connectivity is restored.

## Live editor admission and closeout implemented — 2026-10-13

The separate `antimatter-ui-history-c2048` profile now connects the prepared
Antimatter `8a0f8dc` UI to remote CPU/native qualification. It admits exactly two
current-wire requests (`stream: true`, temperature omitted), inspects prompt
bounds, carries each backend's own complete first reply into turn two, and binds
CPU/native text, EOS and usage parity. The native broker forwards incoming JSON
bytes unchanged and releases the actual bounded SSE bytes only after healthy
reset and loaded-residency checks. Failed, malformed or extra attempts terminate
admission. Buffered delivery does not qualify streaming latency or performance.

Source review caught a real transport incompatibility before live execution:
the retained server emitted SSE without usage. Commit `c6b6161` adds usage from
successful generation to the terminal SSE event for text and tool responses;
error streams do not fabricate usage. All 51 server tests passed, including
terminal usage ordering and absence on failure. A new guarded binary build and
source association are required; do not reuse the older retained binary.

The local runner verifies a frozen gate mirror against an explicit remote
manifest hash, a host/boot-qualified editor identity and fresh broker UUID. It
starts a dedicated SSH forward without consuming a broker readiness request,
contains SSH/Electron together, observes actual process identities and memory/
thermal samples, and verifies retirement on the desktop. Remote closeout checks
its own broker, inspector and server processes, then joins imported UI transport,
history, resource and retirement evidence. It cannot infer local process absence
from remote PIDs. A backend-stage pass alone leaves live UI qualification false.

The editor importer verified the real unused prepared bundle at
`/tmp/antimatter-hag-conversation-pXDP2r`: 13 artifacts, 251 committed source
inputs, 2,389 installed dependency inputs, and 72 Electron 44.3.0 files. Identity
SHA-256 is `9c441b0c5e77b0b89991f4db28734d7e2dc838d42a6a88a428cdf3dda55af65d`.
This checks recorded input bytes; it does not claim an independent rebuild or
verification of unpacked npm files from lockfile integrity. No Electron process,
network request or model execution occurs during identity preparation.

All 365 qualification tests pass, covering old-profile contracts, actual loopback
broker admission, strict SSE replay, resource/identity/history mutations, local
runner helpers and refusal of missing UI closeout evidence. Three failed
intermediate development runs remain recorded: stale profile/inventory test
expectations, an escaped committed vendor path in the clean bundle, and a fake
SSE response ID. Each was corrected and its affected checks passed afterward.
The actual SSH/Electron service has not yet run; its coverage is model-free.

All seven development units/cgroups and recorded PIDs retired. Memory-pressure/
OOM counters and swap remained zero; peak Tctl 56.625°C, sampled memory peak
935,407,616 bytes, minimum host headroom 21,915,713,536 bytes. Development guards
remained CPU25%, Tasks128, MemoryHigh1536M/MemoryMax2G, swap0, NNP,
180/190-second deadlines and startup8GiB/running4GiB. Live remote limits are
unchanged. The local UI runner adds its own 355/375/390-second process/service/
outer clocks, 240-second request clock, ten-second 80°C startup and 90°C cutoff.
The remote 60°C startup and two-turn1212-evaluation ceiling remain mandatory.

Evidence is under `work/hag-editor-ui-gate-20261013/outputs`, including bounded
logs, `unit-04/source.json`, real frozen editor metadata, and `retirement.json`.
Unrelated Antimatter layout/preload edits were preserved byte-for-byte.
The validation host still returned **No route to host**; no model/GPU inference ran.
Next obtain fresh host/kernel/resource observations, rebuild the updated server,
freeze fresh remote inputs, run CPU before arming capture, then execute native/UI
admission and combined closeout. See [the procedure](hag-qualification.md#c2048-live-editor-history-profile).

## Same-host editor qualification prepared — 2026-10-13

Nathan confirmed validation-host is offline for the day and authorized the local box.
Beholdd `296ced4` adds an explicit frozen `--ui-client-host same|separate` choice
for the UI profile. Same-host mode binds the prepared editor's loopback port,
requires the current boot identity, verifies broker PID/socket ownership, and
launches Electron without SSH. The default separate-host path still requires a
different boot and its dedicated tunnel. Closeout records the actual topology.
Old profiles and the two-request parity/reset/evaluation contract are unchanged.

Both modes retain separate backend and UI services. On one host their hard
memory caps total 4GiB and early-stop thresholds total 3GiB, with separate HAG
device accounting and independent global headroom checks. CPU retirement must
precede native admission. UI cleanup may remain observed for up to five seconds
after its process receipt, while process355/service375/outer390 deadlines,
sample-gap limits, end coverage and all resource checks remain enforced.

All **377 qualification tests** passed under the development guard. The updated
release server and metadata-only prompt inspector were built offline with one
Cargo job. Build-01 completed the release binary but then failed because the
inspector command named the wrong Cargo package; build-02 used `beholdd-server`
and succeeded. Both observations are preserved. The successful build's 140
compiled inputs match the frozen source snapshot; helper-only changes do not
alter those inputs. This is an observed build association, not a reproducible
build claim. Server SHA-256 is
`788bc89cdfb1c01e8cb814846ff27c5806ae693993f9d54b2f25e89e9cc8e91b`.

Fresh inputs are frozen under
`work/hag-ui-local-20261013/live` on `exec-alternate-host`, kernel `7.0.0-38-generic`,
boot `REDACTED-HOST-ID`. Manifest SHA-256 is
`d95a3ab32a80fb7a2269bbc40a27d622c42e07c44307620c4f9fc0e112fa21c2`.
The unused Antimatter bundle and identity from the preceding entry remain
unchanged, including its direct endpoint `127.0.0.1:42633`. Unrelated Antimatter
layout/preload edits are preserved byte-for-byte.

All four development units/cgroups and recorded PIDs retired. Sampled
memory-pressure/OOM counters and swap stayed zero; peak Tctl was 78.5°C,
sampled memory peaked at 710,299,648 bytes, and minimum host headroom was
21,143,388,160 bytes. Evidence includes build identities, the tested helper
snapshot, prepared-input audit and `outputs/development-retirement.json`.
No model or GPU inference has run in this local attempt yet. The system-integration task
has acknowledged the quiet-window request and is finishing its active matrix.
Next wait for retirement and ten seconds at most 60°C, then self-check, CPU
reference, CPU retirement, read-only authenticated crash capture, native/UI
admission and combined closeout. Never reuse a failed live stage or used UI profile.

## Local UI CPU reference passed; watcher authentication pending — 2026-10-13

The preceding local attempt passed self-check and the complete CPU reference
after unrelated workloads retired. Both current-wire streaming replies were
`<think>\n\n</think>\n\nready.`, ending with EOS and authoritative usage 467/6/473
and 491/6/497. Turn two included the exact raw first reply. The metadata inspector
measured P467 then P491; 970 actual evaluated positions stayed within the 1212
ceiling. Independent inspection verified complete SSE ordering, usage, byte
hashes and lineage. This is CPU reference evidence, not native/UI qualification.

The CPU stage peaked at 77.25°C and 461,295,616 bytes, with minimum host headroom
21,137,342,464 bytes. Memory-pressure/OOM counters and swap stayed zero; the
stage's kernel window was readable and contained no matching driver/OOM faults.
Both stage services/cgroups, their launchers, the CPU server and both prompt
inspectors retired. See `outputs/cpu-checkpoint-retirement.json` in the same
work directory. The prepared editor remains unused.

After CPU retirement the box cooled to 46.75°C. The read-only crash watcher's
Linux authentication prompt did not complete within its two-minute bound. Its
launcher was terminated; capture never armed and native inference/Electron did
not start. The empty capture output is retained separately as
`outputs/capture-auth-attempt-01.log`. No native gate evidence was overwritten or
retried. The quiet window was released back to other workloads.

Next coordinate another quiet native/UI window and complete desktop
authentication for the read-only watcher. With the same boot and unchanged
frozen inputs, recheck the retained CPU retirement and use a new authentication
attempt (`launch-watcher.py 2`, separate attempt receipts), then run `answers`,
launch the direct UI after its ready receipt, import all five UI evidence files,
stop capture and run full closeout. A reboot or changed frozen UI/source inputs
requires a fresh identity and gate instead. The existing CPU stage is complete;
do not rerun or overwrite it merely to reopen authentication.

## Same-host live editor history qualified — 2026-10-13

After Nathan completed authentication, the preceding local run resumed from its
verified CPU checkpoint on the same boot. The binary, all 140 compiled inputs,
33 frozen gate files, checkpoint, UI identity and unused prepared artifacts were
revalidated before native execution. validation-host remained offline. **Both native/UI
turns and full combined closeout passed** on `exec-alternate-host`, using Beholdd's
`296ced4` gate and Antimatter `8a0f8dc` through direct loopback.

The isolated headless editor made two actual composer submissions through the
real preload, conversation store and production provider pipeline. Both native
raw SSE replies were `<think>\n\n</think>\n\nready.`, exactly matching CPU text,
EOS and authoritative usage 467/6/473 then 491/6/497. Turn two carried the exact
raw first native reply. Healthy native reset and loaded residency were verified
before each reply reached the UI. The editor recorded both complete raw replies
in durable history. There was one attempt per turn, P467/P491, M128 and 970
evaluated positions per backend within the unchanged 1212-position ceiling.

Native and UI peak Tctl was **65.625°C**. Their cgroup memory peaks were
102,330,368 and 425,443,328 bytes, respectively; these are process/cgroup figures,
with HAG device accounting checked separately. Minimum observed host headroom
across the native-stage sampler was 20,169,183,232 bytes. All sampled memory-
pressure/OOM counters and swap were zero. Backend CPU75%/Tasks64/MemoryMax2GiB
and UI CPU25%/Tasks128/MemoryHigh1.5GiB/MemoryMax2GiB containment, early stops,
startup/running headroom, thermal limits and deadlines were unchanged. The UI
process completed in 262.129 seconds; its 2611 supervision samples covered the
whole run with a largest observed gap of 103.67 ms. This verifies containment,
not a latency or throughput benchmark.

The read-only watcher armed at 21:55:34.330512 UTC, after CPU retirement, and
stopped explicitly at 22:00:48.470377 UTC with zero captures and an empty dump
directory. Complete CPU and watcher kernel windows contained no matching
GPU/OOM/fault messages. Combined closeout verified both native resets, final
unload, all three backend-stage services/cgroups, the separate UI service/cgroup,
and their recorded processes. Independent review found all 19 recorded stage,
UI and watcher PIDs absent and all 86 closed-evidence hashes unchanged. It also
reconstructed CPU/native/UI SSE equality, callbacks and persisted history.

Final closeout is `work/hag-ui-local-20261013/live/final-evidence-check.json`,
SHA-256 `915bc16c1d4d59e94fd67ab08360578b3f1f255db9dd711e9943c221852eaa7c`.
It sets `generated_history_qualified: true`, `live_ui_qualified: true` and
`ui_client_host: same`. The editor's own report retains
`modelExecution: external-endpoint-unverified` and `backendQualified: false`;
the separate joined closeout establishes backend qualification rather than
rewriting those narrower client receipts.

Archive `outputs/hag-ui-history2048-local-qualified-20261013.tar.gz` contains
364 verified files, 79,668,513 uncompressed bytes and 22,585,805 compressed bytes.
Its SHA-256 is
`6fe29ad587683814450898fa27641fc6206978b32f95efc64b76f0ca6b1264ff`;
the file-manifest SHA-256 is
`4ba273207b0ecf9185fad474af363ee164b87343f6afd1165f9621da276fff35`.
Packaging ran in a separate CPU25% bounded development service, peaked at
54.375°C, verified every member without extraction, and confirmed source files
did not change. The GGUF payload, Electron distribution and installed npm
dependencies remain external and hash-bound. Producer outputs stay outside the
closed evidence tree. Both authentication attempts and the earlier build failure
remain preserved. Unrelated Antimatter layout/preload edits are unchanged.

This qualifies the pinned C2048/M128 no-tools conversation through the actual
headless UI and same-host buffered-SSE broker. It does not qualify semantic
quality, live cancellation, token latency, throughput, restart/reload recovery,
full main-process editor services or the separate-host SSH deployment. The
quiet window is released. Next add a separate small live UI Stop-and-reuse
profile: cancel after actual content, verify downstream disconnect and a healthy
native reset, then require CPU parity for a fresh conversation on the same
server. Keep this completed buffered profile and its limits unchanged.


## Live editor Stop/reuse CPU passed; watcher authentication pending — 2026-10-13

The next slice is implemented and pushed: Antimatter `f8f43c9` (rolling v10)
and Beholdd `cc0775a`. The new `antimatter-ui-stop-reuse-c2048` profile preserves
the completed history profile and its pinned editor. It clicks the actual Stop
button after visible numeric content, requires real downstream disconnect and
upstream socket shutdown, then healthy native cancellation/reset and unchanged
server identity before admitting an actual New chat request. The fresh reply
must match CPU text, usage and EOS. Independent review caught and fixed a nested
stream/control timeout before live execution; two real-alarm regression tests
preserve the original shared 240-second cancellation/reset budget.

All **416 qualification tests**, **23 live transport tests**, and the full
model-free Electron Stop/New chat smoke passed under the development guard.
The smoke observed the actual socket close and two distinct persisted threads.
Antimatter's release-input verification passed. The unrelated layout/preload
edits retain SHA-256
`9ff346b9b7f84b68d93850fb9dc90a84ca6f5d1757ac8b71272eb3e767159844`.
Fresh client artifacts were built from the committed source, excluding those
edits, and checked against the pinned source closure and Electron distribution.

On local `exec-alternate-host`, self-check and the **single CPU reference passed**.
Both requests were metadata-measured: first P476/M64 (ceiling 539), reuse
P467/M16 (ceiling 482). Only reuse executed, returning raw
`<think>\n\n</think>\n\nready.` with EOS and usage **467/6/473**. CPU stage
ceiling is 482; native total ceiling is 1021. CPU peak Tctl was **61.5°C**,
peak cgroup memory 454,324,224 bytes, and minimum available host memory
21,386,379,264 bytes. Memory-pressure/OOM counters and swap were zero; the
stage kernel window had no matching driver/fault messages. Independent audit
confirmed raw SSE, both metadata rows, the absent first CPU response, clean
unload and process retirement.

Linux authentication for the finite read-only watcher did not complete from
22:30:25 through 22:32:25 UTC. Its prompt launcher was terminated cleanly after
the fixed two-minute limit. **No watcher, native inference, or live Electron
run started.** The empty capture log is retained as
`outputs/capture-unarmed-01.log`; no native closeout or qualification is claimed.
All 13 launched guarded services and 10 recorded process IDs are retired.
The shared resource window was released.

The retained work root is `work/hag-ui-stop-reuse-20261013`. Its frozen manifest
SHA-256 is `2a71c4e39633e9e89ae131a077df70d88117108e0a61acb11d27928fa0695d40`;
`live/cpu-reference-result.json` SHA-256 is
`7fb11726bbd72a2bbc0390d2f86c466d5ed33b203c12d1d5af7e54069f0ebac8`.
The prepared UI is still fresh at `/tmp/antimatter-hag-conversation-UW3iNh`,
with identity in that work root's `ui-metadata`. Boot is
`REDACTED-HOST-ID`, kernel `7.0.0-38-generic`.
The existing guarded build's 140 compiled source hashes and both binary hashes
were rechecked; no Rust or HAG kernel source changed in this slice.

Resume on the same boot by revalidating all frozen inputs, the fresh UI bundle,
Electron bytes and CPU retirement, then obtain a quiet window and run
`python3 -B launch-watcher.py 2` from the work root. Complete the desktop
Linux authentication prompt. After an actual ARMED receipt, launch
`launch-stage.py answers`, wait for its ready receipt, then `launch-ui.py` once.
Import the UI evidence, explicitly stop the watcher after workload retirement,
and run full closeout and the prepared archive verifier. Do not repeat the
retained CPU inference, overwrite evidence, or infer cancellation from the
client report alone. A reboot requires fresh UI identity and gate preparation.

## Stop/reuse resume validated; native run deferred for host load — 2026-10-13

Guarded `resume-preflight-02` passed at 00:59:34 UTC on October 14 (October 13
locally). It rechecked the manifest, all 40 gate helpers, compiled source and
binary hashes, checkpoint, 13 prepared UI artifacts and all 72 Electron
distribution inputs. It independently replayed the existing CPU evidence,
confirmed CPU process retirement and verified the UI bundle remained unused.
No CPU model request was repeated. This read-only validation peaked at 61.625°C
and 990,220,288 bytes of sampled cgroup memory, with zero sampled OOM/swap and at least
17,096,654,848 bytes of available host memory.

Watcher attempt 02 authenticated successfully and armed at 00:59:47 UTC with
PID 2885594. Overwatch started during this window; subsequent host Tctl spot
checks reached 79.6°C, above the unchanged 60°C native admission threshold.
**No native inference or live Electron workload started.** The unused watcher
was explicitly stopped at 01:06:19 UTC, exited 0, captured zero dumps, and left
an empty root-owned dump directory. Its original bytes are preserved as
`outputs/capture-stopped-02.log` and `.stop`; the former SHA-256 is
`1d9546d9796537809c29052bb6f0afced89ab5882ecf08bdc342f8b78861dab9`.
`outputs/watcher-retirement-02.json` records these observations.

`outputs/final-retirement-02.json` confirms all 14 guarded services, 11 recorded
process IDs and two cgroups retired. The shared resource window was released.
The manifest and CPU hashes above remain unchanged, and the prepared UI remains
fresh. The latest retained state is `resume-state-02.json`; earlier receipts
are preserved. External orchestration now supports numbered resume/retirement
receipts, collects suffixed watcher records and includes original guarded build
logs in the eventual archive. No frozen gate helper was changed.

On this same boot, first obtain a quiet host window and the mandatory ten
seconds at or below 60°C. From `work/hag-ui-stop-reuse-20261013`, run
`python3 -B run-checks.py resume-preflight-03`, then
`python3 -B launch-watcher.py 3` and complete authentication. Require a current
ARMED heartbeat with sufficient lease before `launch-stage.py answers`; after
its ready receipt, run `launch-ui.py` exactly once. Import all five UI evidence
files unchanged, retire workloads, explicitly stop the watcher, and complete
joined closeout. Use a fresh retirement receipt (attempt 03) and guarded
`archive-01` only after qualification. Preserve prior failures and do not repeat
CPU inference on this boot. A reboot requires fresh UI identity/gate preparation.

After Stop/reuse qualifies, the next integration candidate is production editor
discovery of Beholdd's per-model `execution_limits.max_evaluated_tokens`.
Antimatter currently reads flat context metadata and the qualification client
supplies C2048 explicitly. Model-free coverage of observed cap precedence and
source/model isolation can close that normal-client admission gap before a
larger GPU workload.

## Live Stop observed; empty pooled connection repaired — 2026-10-13

The retained same-host attempt executed native inference and the real editor
after the host cooled and the shared resource window became available. Stop followed
visible numeric text `1`, reached the fetch AbortController, and closed the
transport. Native request 1 evaluated 481 positions and reported `cancelled`,
`reset_ok: true`, `poisoned: false`, with unchanged PID/start ticks and loaded
residency. The editor persisted the partial response plus `_Stopped._`, then
used New chat to create a distinct empty conversation.

**The combined attempt failed and does not qualify Stop/reuse.** Before admitting
the second request, the broker rejected an incomplete HTTP connection and the
fresh client request received ECONNRESET without response bytes. It made only
one attempt, with no retry or timeout. Frozen inputs and all failed evidence
remain in `work/hag-ui-stop-reuse-20261013`; `resume-state-03.json` supersedes
the earlier resume notes. The used UI's failure report and persisted threads
were copied unchanged to `outputs/failed-ui-state`. The final failed closeout
SHA-256 is `5c0911f3229aa01de557ea7dc748b6697751f51780642b0669ce7feaf10ce736`.
Do not rerun or overwrite this gate or reuse its consumed UI bundle.

The root watcher stopped explicitly with exit 0 and zero dumps. Complete CPU
and GPU kernel windows had no matching driver/OOM/fault messages. Native/UI
peak Tctl was 67.875°C; backend and UI memory peaks were 229,822,464 and
411,365,376 bytes, with HAG device accounting separate. Available host memory
stayed above 20,964,843,520 bytes, and sampled memory events and swap were zero.
The failure-path closeout lacks the successful server/UI receipts it requires;
independent retirement verification confirms workload services, cgroups and
recorded processes are gone. The shared resource window was released.

A bounded model-free experiment using the same Electron executable and frozen
broker reproduced the error without injecting a socket. Undici diagnostics and
broker peer ports show an unused replacement connection opened after abort,
closed with zero request bytes, followed by the fresh POST on a different socket.
The old broker rejected that empty connection and reset the queued POST. The
identical experiment passes against the fix: one recorded empty EOF, exactly
two admitted requests, real first fetch abort and a complete fake ready reply.
These experiments validate transport handling, not native inference or UI
qualification. Their outputs are `outputs/empty-connection-repro-01` and `-02`.

The Stop-only broker now permits at most two confirmed zero-byte EOF/reset
closures and records their count, reason, ordering and relation to reset and the
next admission. Partial requests, idle timeouts, extra valid requests and an
exhausted allowance remain fatal. Shared history behavior and the original
request/stage deadlines are unchanged. All **424 qualification tests** passed,
including eight new socket and evidence regressions; independent review found
no actionable defect. Both Electron experiments and the suite ran under the
unchanged development containment, with no model execution or OOM.

A fresh UI bundle is prepared under `work/hag-ui-stop-reuse-v2-20261013`, pinned
to the same Antimatter commit and excluding unrelated layout/preload edits.
Next freeze the committed fixed gate, run a fresh CPU reference for its new
identity, then perform one native/UI attempt and joined closeout in a quiet
window. The failed gate remains immutable and cannot be promoted to success.

## Same-host live editor Stop and reuse qualified — 2026-10-13

The fresh `antimatter-ui-stop-reuse-c2048` run **passed full combined closeout**
on `exec-alternate-host` during October 13 locally (02:21–02:31 UTC October 14).
Its work root is `work/hag-ui-stop-reuse-v2-20261013`, separate from the failed
attempt above. The gate was frozen from pushed Beholdd `fc11e74`; Antimatter
remains pinned to `f8f43c91c00e155121ce99493f123fc3711af547` and Electron
44.3.0. Fresh UI identity and CPU evidence bind the same boot and direct loopback
endpoint. All 140 compiled source hashes, 40 frozen gate/config hashes, both
binaries and checkpoint identity were verified. No Rust or HAG kernel source
changed; the existing guarded build was reused with its source provenance.

The actual editor displayed `1`, clicked Stop, aborted fetch and closed the
transport. The broker observed real EOF and propagated upstream `SHUT_RDWR`.
Native request 1 reported `cancelled` after **481** evaluated positions, with
`reset_ok: true` and `poisoned: false`. The server remained loaded with PID
3465316 and start ticks 3339142. The editor persisted the partial answer and
`_Stopped._`, then used the actual New chat button to create a distinct empty
conversation. One zero-byte pooled EOF occurred after healthy reset and before
the second HTTP admission; the fixed broker recorded it without rejecting the
queued request.

The fresh request returned exact raw `<think>\n\n</think>\n\nready.` with EOS
`stop`, `[DONE]` and usage **467/6/473**, matching the fresh CPU reference.
Native reset was healthy again and the same server stayed loaded until explicit
unload. There were exactly **two admitted and forwarded requests, one cancelled
request, one complete response and zero rejections**, with no retries. Native
evaluations totalled **954**, within the unchanged **1021** ceiling. CPU measured
both prompts (476/467) but executed only the reuse request: 473 evaluations
within its 482 ceiling. Independent review reconstructed both request bodies,
the cancelled client prefix, the complete reuse SSE, persisted conversations,
native audits and CPU parity without finding a material issue.

Native/UI peak Tctl was **78.75°C**; CPU peak was **62.125°C**. Backend, UI and
CPU cgroup memory peaks were **102,793,216**, **262,656,000** and **454,037,504**
bytes respectively; HAG device memory accounting is separate. Available host
memory stayed above **19,960,983,552** bytes during native execution. Recorded
memory-pressure/OOM counters and swap stayed zero. All 2,553 UI supervision
samples were independently checked, with a maximum sample gap of 102 ms.
The read-only watcher covered the native window from 02:25:29.648344 through
02:31:40.807159 UTC, stopped explicitly, captured zero dumps and left its dump
directory empty. Complete CPU/GPU kernel windows contained no matching
driver/OOM/fault messages. After packaging, `outputs/final-retirement-02.json`
confirmed all seven guarded services, 24 recorded PIDs and four cgroups retired.
The shared resource window is released.

Final record `live/final-evidence-check.json` has SHA-256
`b3c3040c2353dfafabc8c0fa081edb62924a2d8ba6b8ef1af96454af46fc1fc9`;
the frozen manifest SHA-256 is
`fa3653a2a594467da2f49c9c26766113e9722bb3794eaec96ce8fdc372170946`.
All **85 closed-evidence hashes** and the frozen helper hashes passed independent
review. The record sets `live_ui_qualified`, `live_cancellation_qualified` and
`same_server_reuse_qualified` to true. The client report retains its narrower
`backendQualified: false`; backend qualification comes from joined closeout.

Verified archive `outputs/hag-ui-stop-reuse2048-local-qualified-20261013.tar.gz`
contains **358 files**, 49,410,704 uncompressed bytes and 13,236,205 compressed
bytes. Its SHA-256 is
`c432c5fb4b0c2e91483159d523825301d5db9f87b04e07a8532f82895609df85`;
the file-manifest SHA-256 is
`1231b41343b1599d6f3646ea55a0a3d52b11bf1c432bc52e994041afda3c5415`.
Packaging verified every member without extraction and confirmed unchanged
source files under CPU25% development containment, peaking at 59.25°C and
44,494,848 bytes with no OOM/swap. GGUF, Electron distribution and installed npm
dependencies remain external and hash-bound. Active packaging outputs and the
post-packaging retirement receipt remain outside the archive.

The original failed run is separately preserved in
`outputs/hag-ui-stop-reuse2048-local-failed-20261013.tar.gz` (534 verified files),
SHA-256 `281fc6b0a29b5ed7668e0a63f453e73c38910da8f9c81677a3bd13e6effa7ebc`.
Neither completed gate should be rerun or overwritten. The broker fix's **424
qualification tests**, including eight new regressions, passed before this
fresh live run. Unrelated Antimatter layout/preload edits remain unchanged.

This qualifies bounded actual headless UI Stop, native cancellation/reset and
fresh-chat reuse through the same-host qualification broker. It does not qualify
cancelled-answer parity, semantic quality, performance, cancelled-thread
continuation, restart recovery, full production editor main-process services,
coding tools or a portable OS transport. Next implement production editor
discovery/admission of Beholdd's per-model
`execution_limits.max_evaluated_tokens`: retain observed caps per endpoint/model,
clamp effective profiles even when curated or user context settings are larger,
and cover absent/null/malformed metadata without treating an absent cap as
unlimited. Keep exact server token admission authoritative. This next slice can
start with model-free tests and needs no additional GPU run for this checkpoint.

## Production client execution-limit integration — 2026-10-15

Antimatter `632036c` is pushed on master. It adapts the retained October 4
`78147b7` branch to the current named workloads, thinking controls, metrics and
source catalog. `/v1/models` discovery now retains each exact source/model's
`execution_limits.max_evaluated_tokens` separately from curated context metadata.
Effective chat, agent, orchestration and recovery profiles apply it after user
preferences, and a narrower local engine allocation still applies. Saved
workload editors remain uncapped so temporary runtime observations cannot
overwrite user settings.

A positive safe integer sets the observed cap, explicit null clears it, omission
retains the prior observation, and malformed advertised values fail discovery.
Null/absence does not establish unlimited context. Startup/manual refresh use
the same current-source and observation-order checks: a late response cannot
overwrite a newer refresh or a source changed while discovery was pending.
Current Library edits survive the merge. Model-free regressions cover source
and model isolation, endpoint/credential/protocol changes, removal/recreation,
overlapping probes and unchanged saved preferences.

The provider checks estimated complete request size before managed preparation,
activity tracking or HTTP transmission. System/current input and tools are
preserved on refusal; explicit output limits are preserved or refused. An
omitted output limit stays omitted for a capped server, allowing Beholdd to
choose it after exact tokenization. Its authoritative `P + M - 1 <= C` check
and HTTP 400 remain visible, without retry or backend switching on that refusal.
At C2048, the default reply reservation is 512 and no-tools prompt budget 1382;
this is planning behavior, not proof that production coding prompts fit.

**210 targeted tests across 19 files** and the full TypeScript check passed.
The external runner initially lacked the Electron module search path for one
existing metrics suite; after correcting the runner, that suite passed. No
assertions failed. Serial CPU25% services enforced MemoryHigh 1.5 GiB, hard
2 GiB, swap zero, task/time limits and thermal supervision. Peak memory was
715,280,384 bytes and peak Tctl 67.875°C, with zero recorded memory events or
swap. All three check services/cgroups retired; no Electron app, model or GPU
workload ran. Logs and validation receipts are in
`work/hag-editor-execution-limits-20261015`.

The successful Stop/reuse gate retains its original client pin and closed
evidence. This change does not promote the newer production client to a native
qualification. Next add a smoke using the production editor entry points for
discovery, effective budget and oversized-request refusal; measure exact prompt
requirements before increasing HAG capacity for coding tools. See
[Antimatter's execution-limit contract](https://github.com/exec-software/exec-antimatter/blob/master/docs/beholdd-execution-limits.md).

## Production entry-point smoke prepared; thermal hold — 2026-10-15

Antimatter `41eeff0` adds `npm run test:hag-production -- --output /absolute/fresh/path`.
It bundles the actual production main, preload and renderer entry points. Source
creation/refresh, saved preferences and no-tools turns use real preload IPC;
an oversized message is also entered and sent through the actual App composer
with its normal tool definitions. Assertions require C2048 discovery to override
larger curated/saved contexts without rewriting them, synthetic positive/null
controls, and a visible refusal with zero completion POSTs. Test-only shims
isolate app/home state, block subprocesses and suppress unrelated bundled-server
and updater startup. Network is restricted to the synthetic loopback peer.

Independent static review found no execution blocker. The first startup guard
held before any build or Electron process: it found no ten-second interval at
or below 80°C within 60 seconds. A separate read during the wait was 94.875°C.
**Runtime validation and production prompt measurements remain pending.** The
earlier execution-limit unit tests and completed native Stop/reuse gate retain
their separate scopes and evidence; neither proves this new smoke passed.

A bounded metadata preflight verified the retained inspector binary, its 140
compiled-source hashes and checkpoint SHA. No model execution occurred. The
inspector remains `e654c6bdbdaabe38c7a599dbe8558dc2c1983242b24ae4ff4b6de227a0d08cef`.
The new measurement harness will join actual accepted wire bodies to synthetic
POSTs and separately label production-serialized refused diagnostics. Current
Qwen3-0.6B template code rejects tools before tokenization; tool support needs
template work before context growth alone can enable the normal coding client.
Historical October 4 tool-token counts above are not supported current requests.

Resume in `work/hag-production-client-20261015` with a fresh guarded smoke attempt,
then metadata-only inspection, exit/service/cgroup retirement and evidence
closeout. The original startup hold is preserved as `outputs/smoke-01-startup.json`.
No additional GPU run is needed for this pending slice.

## Production editor discovery and refusal verified — 2026-10-15

Antimatter `9c21d02` passes the production-entry smoke in `smoke-10`. It runs
the real main/preload/App renderer with a hidden offscreen BrowserWindow and
test-only native tray/menu boundaries. Discovery, profile resolution, grounding,
provider admission and serialization remain real. Source setup/refresh and four
no-tools turns use actual preload IPC; the final oversized submission uses the
actual composer and Send button. Source settings buttons, native desktop shell
behavior, durable error saving/restart and backend inference are outside scope.

Discovery observed C2048 while retaining curated context 65536 against server
context metadata 131072. The effective no-tools context/reply budgets were
2048/512; saved workload context stayed 32768. Synthetic C16384 and explicit-null
controls produced effective contexts 16384 and 32768. There were exactly three
synthetic completion POSTs, whose bodies match the captured production wire
requests. Oversized no-tools and actual composer requests both failed before
HTTP; the composer visibly reported `context_length_exceeded` and carried 15
production tool definitions with bundled MCP disabled. This is not the complete
configured coding tool catalog.

Metadata-only `measure-01` passed with the same pinned Qwen3-0.6B checkpoint,
inspector and 140 verified compiled sources. It joins every request file to the
smoke record and HTTP evidence, hashes inputs before/after and bounds the
inspector to 29 seconds and 1 MiB output per group. No weights were bound and no
runtime, listener or execution device was created by the inspector.

| Captured production request | Prompt tokens | Reply budget | Minimum evaluated positions |
| --- | ---: | ---: | ---: |
| Fresh no-tools C2048 | 86 | 512 | 597 |
| Oversized no-tools, refused diagnostic | 7,583 | 512 | 8,094 |
| Synthetic C16384 control | 86 | 4,096 | 4,181 |
| Null cap, saved C32768 | 86 | 8,192 | 8,277 |

The actual 15-tool composer diagnostic is rejected before tokenization with
`invalid_request: tools are unsupported for this checkpoint chat template`.
That expected inspector exit 1 is recorded separately from successful no-tools
inspection. No tool-prompt count is reported. Synthetic larger-context success
does not qualify HAG at C16384/C32768. The next slice is exact checkpoint-template
tool support and its renderer/parser regressions, followed by fresh metadata
measurements of supported coding requests before increasing runtime capacity.

Initial thermal holds and seven failed harness attempts remain preserved.
Repairs address a long Chromium socket path, ESM ready ordering, native headless
desktop/window startup, a non-cloneable subscription return and a private-field
assertion against the public catalog. The native Electron crashes were fixture
startup failures, not GPU inference runs. All changes are in test scaffolding;
production client code and the closed October 13 native qualification are
unchanged. Independent smoke and measurement reviews found no remaining issue.

All ten bounded services/cgroups and recorded Electron/inspector PIDs retired;
short temporary directories were verified absent. Successful smoke/measurement
peak memory was 750,297,088 bytes and peak Tctl 81.75°C. Across all executed
attempts, peaks were 1,093,853,184 bytes and 84.375°C, with zero memory events
or swap throughout. No model inference or GPU inference workload ran.

Evidence is under `work/hag-production-client-20261015`, joined by
`validation.json`. Smoke report SHA:
`6f75fa2bda36c22a5307ed5d333993ff2cb59c1922e5b0b2cda6afaae86a80fc`.
Measurement SHA:
`c1e0a336120fd1b939e16cc05e14972368873a01c37cea4cd0df8df8c87ba63a`.
