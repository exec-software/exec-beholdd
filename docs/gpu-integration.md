# Whole-model HAG integration

The first complete resident target is the existing dense **Qwen3-0.6B-Q8_0**
fixture. CPU remains the serving default. An explicit experimental HAG serving
path now passes its bounded runtime and in-process HTTP/SSE gates. Passing a packed
operator or feed-forward
component does not establish whole-model GPU execution or serving readiness.
The integration keeps device transport, inference and editor clients independent.
These gates do not define a portable operating-system ABI or change runtime defaults.

The parallel HAG workstream owns native transport, GPU kernels and live device
qualification. Beholdd owns dense FFN and attention plans through `ExecutionDevice`,
including a fixed-capacity resident KV cache. Native RMSNorm and the selected
layer-0 normalized residual FFN block passed bounded live qualification with their
frozen six-program library. Isolated RoPE/attention and the selected real layer-0
attention block then passed with the ten-program library, including code-page
crossing and resident KV reset/reuse. The current eleven-program implementation
adds tiled tied-vocabulary lookup and final-normalized logits; its isolated row
and selected real-vocabulary gates also passed. None of these component gates
registers a GPU serving backend. `DenseModelPlan` and `ResidentDenseModel` now
compose the complete 28-layer graph with a checked capacity-eight memory forecast
and global cache/output lifecycle. All 91 model tests passed, including seven new
full-model tests. Native first-token and two three-token requests subsequently
passed for the selected complete checkpoint, with full-logit tolerance checks and
exact CPU greedy agreement. The backend's eleven shader programs are unchanged
by this composition. The runtime now has an explicit bounded HAG selection;
its bounded runtime/HTTP/SSE lifecycle gate passed separately. Broader network,
sequence and performance qualification remains open.
See [HAG qualification](hag-backend.md).

## Derive the contract from the bound checkpoint

Use the strict `QwenModel` binder before producing execution requirements. Read
actual tensor types, dimensions and encoded byte spans; a model ID or architecture
tag alone is insufficient. Requirements must name the operation and tensor that a
backend refuses, before uploading weights or allocating model-sized arenas.

The migrated fixture has 28 dense layers, hidden width 1,024, FFN width 3,072,
16 query heads, 8 KV heads and head dimension 128. Query width is therefore 2,048,
**not** hidden width. Vocabulary is 151,936; checkpoint context is 40,960.
RMSNorm epsilon is approximately `1e-6`, RoPE base is `1e6`, and RoPE uses split halves.

Matrix dimensions below are **output rows × input columns**, opposite GGUF's
fastest-dimension-first listing. The same shapes occur in every layer.

| Weight / operation | Shape | Encoded bytes |
|---|---:|---:|
| Q projection, Q8_0 | 2,048 × 1,024 | 2,228,224 |
| K or V projection, each Q8_0 | 1,024 × 1,024 | 1,114,112 |
| Attention output, Q8_0 | 1,024 × 2,048 | 2,228,224 |
| FFN gate or up, each Q8_0 | 3,072 × 1,024 | 3,342,336 |
| FFN down, Q8_0 | 1,024 × 3,072 | 3,342,336 |
| Token embedding / tied LM head, Q8_0 | 151,936 × 1,024 | 165,306,368 |
| Attention or FFN norm, each F32 | 1,024 | 4,096 |
| Q or K head norm, each F32 | 128 | 512 |
| Final norm, F32 | 1,024 | 4,096 |

There are 197 Q8_0 tensors (633,233,408 bytes) and 113 F32 norm tensors
(262,144 bytes): **633,495,552 payload bytes** across 310 tensors. One layer's
payload is 16,720,896 bytes. The 639,447,744-byte GGUF file also contains metadata
and alignment; file length is not the resident weight requirement.

This fixture has no separate `output.weight`: the LM head aliases token embeddings.
Deduplicate by checkpoint identity and encoded span, retaining one resident upload
with both semantic uses. F32 norms can be resident F32 tensors; their presence does
not require an F32 matrix-multiplication kernel.

## What the existing execution seam can express

`CommandList` expresses linear overwrite, linear scaled-add, SwiGLU, weighted or
unweighted RMSNorm, RoPE with explicit layout and direction, incremental causal
attention over position-major KV views, and `embedding_row` from an owned prepared
weight. Lookup validates the output width and row bound before appending, retains
the immutable source, and overwrites its destination on replay without source I/O. Norm weights and RoPE sine/cosine tables
are resident operands read at execution time. Attention copies its dimensions,
absolute query position and scale while retaining all four tensor views. Shape,
epsilon, scale, checked causal bounds and ownership checks happen before appending;
failed appends leave the list unchanged. Commands observe earlier writes, and overlapping views have snapshot
semantics. The CPU executor establishes parity with the immediate CPU contract.
Each backend must implement or explicitly refuse the operation; adding a command
does not supply a GPU kernel or authorize host arithmetic fallback.

The four-command `DenseFfnPlan` accepts normalized input and overwrites an output
contribution. `DenseFfnBlockPlan` records five commands for
`hidden += FFN(RMSNorm(hidden))`: norm → gate → up → SwiGLU → down scaled-add.
Prepared weights, norm and intermediate tensors remain owned across recording and
repeated execution. Preparation stages one matrix at a time.

The initial HAG baseline accepted BF16/Q8_0 matrices only up to 64 × 256, SwiGLU
up to 64 elements, 1 MiB packed/tensor arenas and 4 KiB auxiliary storage. A real
1,024-row scaled-add alone needs 4,096 scratch bytes plus a four-byte parameter.
The current HAG adapter admits up to 1 GiB of packed weights, while public tensor
and auxiliary arenas each remain capped at 64 MiB. Auxiliary capacity is explicit,
and dispatch supports multiple workgroups. The selected
real-checkpoint four-command FFN has now passed the native gate below. Its current
source and [qualification record](hag-backend.md) define exact tested shapes;
the larger admitted limits do not qualify every shape. Normalized-block evidence
is recorded separately below.
The current native adapter adds RMSNorm and F32 multiply after the unchanged four
shader payloads. Unweighted normalization uses one dispatch; weighted normalization
uses private scratch plus a second dispatch reading resident F32 norm weights.
The five-command normalized residual block therefore lowers to seven physical
dispatches, including the down projection's separate residual-add pass.

Native RMSNorm accepts 1..4096 elements and finite nonnegative epsilon. It uses
ordered F64 square/add, approximate F64 reciprocal/reciprocal-square-root, then
F32 inverse/input/weight scaling. The adapter refuses output aliases with input
or weight before lowering, while allowing read-only input/weight overlap. Values
remain resident and are read at execution time; recording does not snapshot the
norm weights, and no missing operation runs implicitly on the CPU.

Native RoPE now supports interleaved/split-half layouts, forward/inverse direction
and up to 4096 coordinates with resident coefficients. Causal GQA uses three
passes per query/head: F64 QK reduction, scaled stable F32 softmax and F64 weighted
value reduction. It admits at most 256 head coordinates/positions, 32 query/KV
heads and KV width 4096, with checked absolute query bounds. Its exponential and
reciprocal path needs a declared tolerance. Native Q8_0 `embedding_row` adds one
exact-row decode dispatch with no expanded matrix. The current eleven-program
library places its new entry at byte 6,912 and ends at 7,368, still in an 8,192-byte
shader allocation. All prior ten payloads and offsets remain unchanged. See the
[full native contract](hag-backend.md#current-resident-adapter-contract).

The complete dense graph is implemented; remaining qualification includes:

- Broader numerical/position coverage beyond the qualified isolated RoPE/attention
  fixtures and layer-0 attention block. Its two three-position requests cover
  width-128 Q/K norms, absolute positions, resident cache reuse and request reset;
  the full admitted geometry and longer sequences remain separate work. The
  selected complete-model gate below additionally exercised all 28 layers.
- Broader vocabulary coverage beyond the qualified selected rows and three
  synthetic hidden-to-head inputs, including the shared 38-tile vocabulary in
  complete-model execution.
- Broader whole-model fit and logit/token coverage beyond the selected serial
  two-request, three-token fixture, including longer prefill/decode requests.
  Residual additions use scale one; no MoE router is needed for this fixture.

The initial full-model gate explicitly reads back final logits for CPU greedy
selection: 607,744 bytes per token. Intermediate activations and KV must stay on
device. A later device argmax must preserve nonfinite rejection and first-index tie
selection. Tokenization and HTTP framing remain host operations.

## Account for the actual resident schedule

The first complete-model gate fixes capacity at eight positions and exercises
three per request, refusing steps beyond capacity. Serial one-token prefill and
decode reuse bounded scratch; this is a correctness target, not a
prefill-throughput claim. Larger capacities require their own checked admission.

F32 KV payload is `2 × 28 × C × 8 × 128 × 4 = 229,376 × C` bytes: 58,720,256
at `C=256`, 469,762,048 at `C=2048`, and 9,395,240,960 at full checkpoint context.
These are logical KV bytes, not the complete budget. Add packed weights, F32 norms,
live activations, logits, attention-score scratch, RoPE tables, recorded-command
scratch/parameters, page rounding, guards and queue allocations. Count norm storage
once. Keep host staging, tokenizer/index memory and runtime headroom separate and
include both host and device charges in the process-wide fit decision.

Use checked arithmetic and lifetime-derived peak bytes. The existing CPU forecast
includes dequantized matrix scratch and growing host vectors; it must not be relabeled
as a GPU allocation plan. Preflight should report requirements and refusals without
opening a device, copying full weights or claiming executable compatibility.

### Shared vocabulary implementation

`QwenModel::vocabulary_plan(tile_rows)` derives exact tied embedding/head locator
identity, Q8_0 row spans and an F32 final norm from the strict binder. Source
identity, formats and independent packed/tensor/staging limits are checked before
preparation reads or allocates. For this fixture, `tile_rows=4096` produces 38
shared tiles: 37 full tiles of 4,456,448 bytes and a 384-row tail of 417,792 bytes.
Preparation reads one tile at a time and uploads each exactly once for both uses.

`record_embedding(device, token, hidden)` checks `token < vocab` before building
a command, selects its tile and local row, and binds only the exact 1,088-byte
encoded row to the native decoder. Token IDs above GPU scalar limits are resolved
on the host. `record_logits(device, hidden, logits)` applies resident final RMSNorm
and 38 matvecs into disjoint logit ranges: 39 semantic commands and 40 dispatches.
The same packed handles back lookup and head; only a norm vector and normalized
hidden scratch are additionally owned. Batches retain their resources, and callers
must wait before modifying shared boundaries/scratch or reusing a batch.

The explicit `hag_vocabulary_probe` defaults to metadata-only inspection. It
predicts **165,974,016 charged HAG bytes** under a **192-MiB HAG budget**:
165,306,368 unique packed bytes, 620,048 public tensor bytes including guards,
and an 8-KiB auxiliary arena with 4,100 live head bytes. Native preparation reads
38 tiles plus the norm once, totaling 165,310,464 bytes; the largest payload staged
is 4,456,448 bytes, below the probe's 8-MiB staging cap.

The vocabulary gate first checks tokens `[0, 4095, 4096, 65536, 151551, 151935]`
bitwise against independent CPU row decoding, then three changed synthetic hidden
inputs through the normalized full vocabulary head. It requires 6,144 embedding
and 455,808 head values, 18 output guards, nine submissions and 126 dispatches.
Head acceptance is finite per-value error at most `3e-4 + 3e-5 * abs(reference)`
and exact CPU greedy selection with first-index tie behavior. Head sign/zero
classification mismatches are reported rather than independently rejected; this
policy is preregistered. Embedding values retain exact-bit/signed-zero checks.
The head inputs are independent synthetic vectors, not a full-model generation
sequence or the output of the six preceding lookup checks.

All CPU oracle work precedes device opening: 123 sequential reads and 1,847,808
retained reference-output bytes. Lookup batches/dummies retire individually before
the head batch, which is reused three times. Expected final readback including
guards is 1,847,880 bytes; submit/wait tensor traffic must stay zero. The 4,124 bytes
uploaded while recording private lookup/head storage are accounted separately.
The selected component passed after the isolated row-decoder gate with
`HAG_TRACE_SUBMISSION=0`, verified 1-GiB process/no-swap limits, the unchanged 90°C
cutoff and fresh crash/driver evidence. Those prerequisites remain required for
future live runs. No full-checkpoint host cache is used.

At 21:02 UTC on 2026-10-03, the isolated decoder passed 16 dispatches, 17,152
exact values and 32 guards; its 90,112-byte HAG charge returned to zero. The real
vocabulary gate then passed all six lookups and three head rounds: 126 dispatches,
nine submissions, 461,952 checked values and 18 guards. Embedding values matched
bitwise. Head maximum absolute error was zero with zero sign/zero classification
mismatches, while the acceptance contract remains the declared tolerance plus
exact greedy agreement. Greedy results `[117270, 121057, 253]` matched the CPU;
these are independent synthetic head inputs, not generated model tokens.

Actual HAG charge matched **165,974,016 bytes** and returned to zero. Native
preparation stayed at 39 reads and CPU-oracle reads at 123; submit/wait tensor
traffic was zero and final readback was 1,847,880 bytes. The real component peaked
at **53,719,040 process/cgroup bytes**, separately from HAG's GPU charge, with
maximum sampled Tctl **48.5°C** and minimum host available memory **24,664,236,032
bytes**. Both launches had zero swap/memory/OOM events or matching driver faults.
Fresh capture stopped cleanly with zero dumps; its complete 469-line kernel window
had no matching GPU/OOM/fault messages. See the
[dated vocabulary result](hag-backend.md#native-row-decode-and-tied-vocabulary-qualification--2026-10-03)
for frozen identities, exact scope and `outputs/hag-vocabulary-qualified-20261003.tar.gz`.
No complete checkpoint execution, captured-activation parity or throughput result
is implied by this component pass.

### Complete resident model and staged native gate

`QwenModel::dense_model_plan(capacity, tile_rows)` now composes every attention and
FFN block with one shared vocabulary. It validates exact Q8_0 matrices and F32 norms
in all layers before payload reads, requires complete-block hidden/FFN widths at
most 4096, and bounds metadata to 1..128 layers and at most 4096 layer-position
batches. Capacity is 1..256 within model context, with the existing component
head/tile limits. Checked `DenseModelMemory` includes packed weights, all resident
F32 payloads including private hidden/logits, and the largest sequential staging
payload. Plan getters expose `dim`, `vocab`, `layers`, `capacity` and `memory`.

`prepare(device, source, DenseModelLimits)` checks source identity, all three
aggregate limits and Q8_0 backend support before its first allocation/read.
Unsupported backend command support or physical capacities may still refuse
during preparation. Each packed weight is staged independently; no expanded or
fully resident host checkpoint is created. The capacity-eight plan derives:

| Resource | Logical bytes |
| --- | ---: |
| Unique packed weights, including shared vocabulary | 633,233,408 |
| Public tensors: `28 * (107520 + 32768) + 8192 + 4096 + 607744` | 4,548,096 |
| Persistent auxiliary: `28 * (165168 + 8204) + 4100` | 4,858,516 |
| Auxiliary peak, including one transient lookup dummy | 4,858,520 |
| Largest preparation payload | 4,456,448 |

With exact capacities, guard/page rounding and the 8-KiB library, the resulting
HAG charge is **642,682,880 bytes (612.91 MiB)**, matching the checked cost query
in both staged live gates below. Public tensors include 1,835,008 KV bytes;
hidden/logits here have no diagnostic guards. Preparation makes
347 range reads totaling 633,495,552 bytes. The owner retains 224 attention
position batches, 28 FFN batches and one head batch, plus one transient lookup:
253 persistent batches and 254 at peak. A complete token with logits executes
embedding, then 28 attention/FFN pairs, then the head: 58 logical submissions,
1,692 semantic commands and 3,793 physical dispatches. Recorded-command metadata
also has a host cost; the GPU byte total must not stand in for that forecast.

`ResidentDenseModel` owns private hidden/logits, all layer caches/batches and the
shared vocabulary/head. It cannot be cloned and exposes no resource handles.
`step(device, token: u32)` checks token, remaining capacity and every cache state
before recording/submission. It marks the whole model uncertain before embedding
recording, retires the completed transient lookup, runs all attention/FFN pairs,
then waits for the complete head. Only then does `visible_positions` advance.
Any execution error or unwind permanently poisons the model, even when earlier
layer caches already advanced, and prevents steps, reads and reset. Invalid token
or full-capacity refusals leave a healthy owner reusable.

`read_logits(device, output)` requires the exact vocabulary length and a completed
step, refusing stale output before the first step or after reset. Transport read
errors/unwinds poison the owner; shape refusals do not. `reset()` checks every cache
before changing any position and then resets them all without allocation or device
I/O. Every newly visible row is overwritten before consumption. `capacity()` and
`is_poisoned()` expose the remaining lifecycle state.

Seven small two-layer GGUF tests cover complete-model immediate-CPU parity across
multiple positions and request reset, source/resource retention, checked totals
and overflow, pre-I/O limits and source identity, malformed last-layer formats and
shapes, healthy boundary refusal, actual read-error poisoning, late FFN/head errors
and a head unwind after real CPU execution, and reset prevalidation. The complete
model library passed **91 tests** in the bounded centralized run.

`Gguf` uses positional reads rather than a whole-file mapping. Continue staging
one checked `WeightRange` tile at a time; `FullyResidentSource` would duplicate
633,495,552 host payload bytes alongside GPU storage. File cache, CPU oracle scratch
and driver-private memory still contribute to measured process pressure. The
component probes' memory results are not evidence of complete-model fit. The new
`hag_model_probe` defaults to metadata-only inspection and checks its exact
capacity-eight allocation forecast before device access. Its native qualification
is deliberately staged: `--first-token` prepares every layer and checks one token;
only after that succeeds may the same frozen binary run `--execute` for two
three-token requests. Prompts are `[1, 2]` and `[4, 5]`; the third token of each
request is the CPU greedy selection after its second position, which must also
match the native greedy selection before use. Capacity eight is allocated; only
three positions per request are exercised.

The independent oracle uses immediate CPU operators, one-worker packed matvecs,
separate per-layer KV and a tiled vocabulary head. It streams one matrix/tile at
a time and finishes all reference work before opening the device. The full gate
makes 2,088 CPU reference reads and retains 3,646,464 logit bytes; first-token mode
makes 348 reads and retains 607,744 bytes. Native preparation stays at 347 reads
in either mode, with no source rereads/reuploads during execution.

The complete-model numerical bound is preregistered as finite values with
**`abs(actual - reference) <= 2e-3 + 2e-4 * abs(reference)`**, plus exact CPU greedy
selection with first-index ties and nonfinite rejection. Sign/zero classification
mismatches are reported separately. First-token mode checks 151,936 logits in
3,793 dispatches/58 submissions. The repeated-request gate checks 911,616 logits
in **22,758 dispatches and 348 submissions**. Only final logits are read:
3,646,464 bytes over six steps. Submit/wait must transfer zero tensor payload;
each dynamic lookup separately writes four private bytes during recording.
Residency/source counts must stay fixed, reset must return the global position
to zero, and cleanup must release all HAG charge.

Both native stages require `HAG_TRACE_SUBMISSION=0`, an explicit **768-MiB HAG
budget** and a separately verified **1-GiB process cap with zero swap**. The
supervisor stops at **896 MiB of process/cgroup use**, starts only with at least
**8 GiB host available memory**, and stops below 4 GiB or at 90°C Tctl. Fresh crash
capture and full driver-window checks remain mandatory. Host staging, CPU scratch,
file cache, recorded-command metadata and driver memory are separate from the
HAG charge. Both live stages passed under these unchanged limits and numerical
criteria; their scoped result follows. Serving readiness is a separate gate.

### Complete-model live result — 2026-10-03 EDT

At **21:19 EDT on October 3, 2026** (**01:19 UTC on October 4**), the same frozen
binary passed first-token qualification and then two three-token requests through
every layer of the selected dense Qwen3-0.6B-Q8_0 checkpoint. First-token mode
completed **3,793 dispatches, 58 submissions and 151,936 checked logits**, with
maximum absolute error **0.00001430511474609375** and exact greedy token **14582**.

The full stage completed **22,758 dispatches, 348 submissions and 911,616 checked
logits**, with maximum absolute error **0.0000553131103515625**. All values were
finite and met the preregistered `2e-3 + 2e-4 * abs(reference)` bound. All six CPU
greedy selections matched: **`[14582, 1096, 374, 15846, 3, 4]`**, for request inputs
`[1, 2, 1096]` and `[4, 5, 3]`. The two sign classification differences and zero
zero-classification differences were reported under the policy declared before
the runs; acceptance criteria were unchanged. Capacity eight was allocated, and
only positions **0..2 per request** were exercised. Reset/reuse passed and the
final visible count returned to zero.

Both stages allocated **642,682,880 HAG bytes** and released the entire charge.
Native preparation stayed at **347 reads / 633,495,552 bytes** once; the full CPU
oracle made **2,088 sequential reads / 3,800,979,840 bytes before GPU access**,
retaining 3,646,464 reference bytes. Explicit final-logit readback was also
**3,646,464 bytes**, with **zero submit/wait tensor transfers**. Initial host writes
were 643,016,852 bytes, including initialization and supplied norm/coefficient
values; the six lookup recordings added 24 private bytes. Intermediate activations
and KV remained resident, and live allocations/source counts stayed fixed.

First-token process/cgroup peak was **57,454,592 bytes**; full execution peaked at
**59,682,816 bytes**, separate from the HAG GPU charge. Maximum sampled Tctl was
**57.875°C / 55.875°C** respectively, and minimum host available memory was
**22,781,796,352 / 23,077,203,968 bytes**. Both launches had zero swap, memory/OOM
events or matching driver faults. Root capture **2657219** stopped cleanly after
covering **2026-10-04 01:18:15.591430–01:19:56.638393 UTC**, with zero dumps and no
GPU/OOM/fault matches in the complete **508-line kernel window**. Closeout verified
all 126 frozen source files, binary/dependency identities and the rehashed unchanged
checkpoint. See the [dated complete-model record](hag-backend.md#complete-resident-dense-model-qualification--2026-10-03-edt)
for exact hashes and `outputs/hag-full-model-qualified-20261003.tar.gz`.

This qualifies the selected complete dense graph, bounded cache positions, full
logits, exact greedy agreement and cleanup. HTTP/SSE behavior, cancellation,
unload/reload, general context/model coverage, throughput and changing the serving
default remain separate from this gate. At this stage, `NativeRuntime` advertised
only CPU execution; the later serving result below is separate evidence.

## Completed gates and remaining scope

1. The selected real dense FFN plan passed `CpuExecutionDevice` comparison against
   the existing immediate CPU path with real checkpoint weights.
2. The selected native operator and FFN gates passed their actual shapes, packed
   residency, repeated submissions, intermediate residency and bounded cleanup checks.
3. The selected complete resident schedule passed its bounded memory, full-logit
   and exact greedy gates above. Broaden sequence/context/model coverage without
   treating the three exercised positions as qualification of every admitted shape.
4. The explicit `NativeRuntime` path passed its scoped token/text, in-process
   HTTP/SSE cancellation, unload/reload and fit-refusal gates below. Qualify network
   service lifecycle and performance before promotion beyond this experimental scope.
   The descriptor identifies configured implementations, not broader validation.

This dense target does not qualify larger MoE checkpoints, multimodal models or RPC
distribution. Antimatter's remaining llama.cpp lifecycle and packaging transition
depends on the capabilities actually completed here.

## Experimental serving integration: current C128 implementation

The Linux `beholdd-cli/hag-native` feature reaches the recorded executor through
`NativeRuntime`, using `execution_backend: hag` and an explicit device/allocation
budget. CPU remains the default. A runtime configured with HAG reports
`execution_backends: ["cpu", "hag"]`; a CPU-only configuration reports `["cpu"]`
even in the feature-enabled binary. No backend is selected from an architecture
name or replaced by a fallback after refusal.

The current serving implementation admits the same exact 28-layer dense Qwen3
geometry, formats, epsilon and RoPE base as the complete-model probe, with fixed
capacity **128**, tile rows **4096**, and checked
`prompt_tokens + max_new_tokens - 1 <= 128`. The final selected token is not run
through the model. HTTP chat framing consumes prompt capacity too. This is a
separate scope from the earlier capacity-eight gate that exercised three positions
and the frozen C16 serving gate that exercised positions 0..9. Bounded C64 and real
CLI/TCP qualification subsequently passed on October 4. **The separate frozen C128
boundary and CLI/TCP gates also passed with clean final closeout**.

The subsequent single integer-list fixture passed EOS completion and exact
CPU/HAG raw text/usage/finish parity: 29 prompt + 27 emitted tokens, 56 evaluations
per request. One native answer and four timing requests reset and unloaded cleanly.
The first lazy-load response took 12.439 seconds; three warm SSE responses reached
DONE in 11.588–11.821 seconds. These are descriptive client observations, not
decode or kernel throughput. Recall and general answer quality remain unqualified;
four semantic/format failures and a thermal abort remain preserved. See the
[list fixture record](hag-backend.md#integer-list-eos-parity-and-response-timing--2026-10-04).

At capacity 128 with the current recording layout, the model retains **33,793,536
F32 bytes**, **14,842,196 private auxiliary bytes** and a **14,842,200-byte auxiliary
peak**, alongside 633,233,408 packed bytes. The exact allocation query gives
**681,910,272 HAG bytes**.
Combined admission adds the entire 639,447,744-byte checkpoint file,
4,456,448-byte staging maximum, 607,744-byte host logits buffer, bounded token IDs
and a checked **95,420,416-byte (91-MiB) recording/owner metadata term**, separately
from at least 128 MiB of general runtime headroom: **1,556,061,376 bytes**
with this fixture and headroom on the 64-bit host. Both the configured process budget and current
host/cgroup availability must cover that forecast before device opening/payload
reads. HAG's separate `MemoryBudget` still admits actual hardware allocations.

All preparation is sequential direct-read staging, with no full host checkpoint
cache. `weight_mode: streaming` names that preparation policy; the resulting
weights remain resident on the device. Exact fit is required, all host weight
caches/prefetch are disabled, and unsupported shapes/budgets refuse explicitly.
The example uses a **2-GiB combined budget**, **128-MiB runtime headroom** and
**768-MiB HAG budget**: [config.hag.example.yaml](../config.hag.example.yaml).

The complete owner remains behind the runtime execution mutex. Cancellation is
polled around fully fenced token executions and before delivery, so prompt work
can stop even before text is available. Healthy cancellation or sink errors reset
KV state for the next request. Execution/read/reset failure or nonzero charge
after model-before-device destruction disables HAG process-wide until restart.
Explicit unload reports uncertain cleanup. Eviction cannot restore HAG health or
bypass quarantine, though explicit CPU selection can still proceed. The existing
HTTP streaming and nonstreaming workers keep
their admission until synchronous work actually returns.

Admission is per `NativeRuntime` instance, and this serving gate/setup uses one
instance. Multiple simultaneous instances/processes or device sharing are not
qualified. The process-wide health latch is separate from request admission and
does not serialize separate runtime instances.

Build and launch syntax, without changing any default:

```sh
HAG_TRACE_SUBMISSION=0 cargo run --release --locked -j1 -p beholdd-cli --features hag-native -- \
  serve --config config.hag.example.yaml
```

Run live work under the dedicated process/thermal/crash supervisor. The serving
gate uses a 2-GiB process cap, zero swap and a 1.5-GiB early stop, independently
of HAG accounting. The separate October 3 live serving parity, cancellation and
unload/reload gates passed with their frozen C16 allocation in the bounded scope
below. The C64/TCP gates subsequently passed in their own scope. Historical
records remain unchanged and retain their original scopes.

### Serving lifecycle result — 2026-10-03 EDT

Runtime and HTTP/SSE stages passed at **23:40 and 23:41 EDT on October 3, 2026**
(**03:40 and 03:41 UTC on October 4**). The runtime evaluated nine prompt tokens
and one decode token, positions **0..9**, matching CPU tokens **`[151667, 198]`**,
decoded text and usage exactly. Metadata-budget refusal preceded opening a missing
device; early control cancellation kept the model unloaded, and successful
generation unloaded cleanly.

Four successful one-token HTTP requests matched CPU **`[151667]`**, text and usage.
The gate consumed the real SSE body, observed a first-text disconnect, successfully
reused the resident model, and verified capacity refusal, busy-unload **409**,
idempotent unload, reload and final unloaded state. Calls ran through the real
Axum router **in process**; no TCP socket was opened.

The **649,277,440-byte HAG charge per load** retired completely under explicit
model/device cleanup and health/unload checks. Process peaks were
**245,776,384 / 245,161,984 bytes** for runtime / HTTP, separately from GPU charge.
Maximum sampled Tctl was **59.5°C / 58.375°C** and minimum host available memory
**21,536,968,704 / 21,996,560,384 bytes**. Both stages reported zero swap, memory/OOM
events, matching driver faults and crash dumps. The supervisor retained its
2-GiB process cap, no swap, 1.5-GiB early stop and 90°C cutoff; 22 wrapper tests
passed before launch.

Root capture **3539802** covered **2026-10-04 03:40:09.450761–03:41:47.996004 UTC**
and stopped cleanly; its complete **458-line kernel window** had no matching
GPU/OOM/fault messages. Closeout verified all 128 frozen source inputs, binaries,
dependencies and the unchanged checkpoint. The [dated serving record](hag-backend.md#experimental-hag-runtime-and-httpsse-qualification--2026-10-03-edt)
contains full identities and the archive path
`outputs/hag-serving-qualified-20261003.tar.gz`.

This clears the selected single-runtime token/text and HTTP/SSE lifecycle gate.
It does not qualify TCP/network serving, throughput, every allocated position,
concurrent runtime instances/processes or shared-device use. CPU remains the default;
HAG requires the explicit experimental configuration.

## Context-aware C64 and TCP qualification — 2026-10-04

Context-aware request planning is implemented. `ExecutionLimits` exposes
`max_evaluated_tokens: Some(64)` for HAG and `None` for CPU without checkpoint I/O;
`/v1/models` serializes the field under `execution_limits`. `plan_request` and its
controlled variant encode the rendered HAG prompt, then resolve an omitted limit
to `min(default_max_new_tokens, 64 - prompt_tokens + 1)`. Explicit limits are
preserved or refused. CPU retains its existing HTTP default of 512.

Server admission spans planning and generation, and cancellation keeps the permit
inside any running blocking worker. Both JSON and SSE paths plan before headers;
explicit overflow is HTTP 400 with `code: context_length_exceeded` and
`param: max_tokens`. Planning may bind tokenizer metadata but does not read weight
payloads, open the GPU, publish residency or reset caches. Hardware-free tests cover
cold/loaded refusals, omitted limits, cancelled planning and queued requests.

That frozen C64 runtime counted its retained 1,792 attention batches and other
recording/owner storage before admission; its 67-MiB metadata term was additional
to runtime headroom. The separate boundary probe evaluated fixed token IDs 1..64, checked all
151,936 logits and exact CPU greedy choices at positions 0, 15, 16, 31 and 63, then
checked a refused 65th position and reset/replay of token 1. It retains the finite
`2e-3 + 2e-4 * abs(reference)` tolerance. A tiny fixture verifies the independent
immediate CPU oracle against production forward at those positions through 63.

The real CLI/TCP gate completed CPU references and exited that process before
native HAG. Its fixtures included nine-token `Hello`, a 21-token conversation and
a 63-token boundary prompt whose omitted output count resolved to two. Nonstream/
SSE parity, typed cold/warm refusals, actual socket shutdown, audited cancellation/
reset, reuse, busy unload, unload/reload and idle SIGTERM exit zero all passed. The CLI handles SIGINT/SIGTERM by gracefully stopping HTTP service and
draining admitted blocking workers; it does not forcibly interrupt GPU work or
guarantee bounded shutdown of active requests. Optional `HAG_SERVING_AUDIT=1`
contains only request sequence, evaluated count, outcome, reset and poison status,
without prompts, token IDs or tensor data.

**Both bounded C64 and TCP gates passed at 06:05–06:07 UTC on October 4.**
The boundary gate made 65 evaluations / 246,545 physical dispatches / 3,770 logical
submissions and six full readbacks (911,616 values / 3,646,464 bytes). Maximum error
was `9.1552734375e-5` under the unchanged finite tolerance; all six greedy choices
matched CPU. One near-zero sign mismatch and zero zero/nonzero mismatches were
reported. Reset/replay and capacity refusal passed, and HAG charge returned to zero.

TCP completed six requests and one socket cancellation, **136 evaluations** total;
all request audits reported healthy reset, followed by unload/reload and CPU/HAG
SIGTERM exit zero. Four invalid/overflow requests were refused without generation.
All three CPU references emitted exactly two tokens, **`<think>\n`**, matching HAG
text/usage with length finish. Complete-answer quality and throughput were not tested.
Boundary/TCP process peaks were **89,931,776 / 280,555,520 bytes**, maximum Tctl was
**77.625°C / 59.625°C**, and neither stage recorded swap, OOM or matching driver
faults. The probe's oracle-specific process forecast was 1,572,308,003 bytes;
that frozen runtime admission separately used a 1,538,288,320-byte forecast.

Evidence and supervisors are in the development chat's `work/hag-context-20261004/`,
with archive `outputs/hag-context-qualified-20261004.tar.gz`. The [dated result](hag-backend.md#context-aware-c64-and-tcp-qualification--2026-10-04)
pins all 130 source inputs and binary identities. The October 3 frozen C16 result
retains its original scope. Process, thermal and crash-capture controls stayed
unchanged; arbitrary inputs/models, concurrency and performance remain unqualified.

### C128 implementation and qualification — 2026-10-04

Current source prepares **128 evaluated positions**. **Both scoped live gates passed**
in `work/hag-c128-20261004/`; earlier C64 source identities and measurements remain
historical. Public tensor bytes are 33,793,536, peak auxiliary bytes 14,842,200 and
HAG charge 681,910,272. Native metadata is 94,711,519 bytes, with the rounded
91-MiB model/recording allowance counted in the runtime forecast above. The
independent boundary probe forecasts 1,604,527,519 bytes.

The preregistered boundary fixture makes 128 CPU reference and 129 GPU evaluations,
checks positions 0, 31, 63, 64, 127 and replay 0, and retains finite tolerance plus
exact CPU greedy agreement. The TCP gate uses a 127-token rendered prompt with an
omitted limit resolved to two, typed refusal of Hello plus 121 output tokens,
real cancellation/reuse, unload/reload and clean exit. Its ceiling is 214 evaluations.

The fresh live supervisor uses **75% CPU**, reduced to limit heat during the doubled
CPU oracle work. Memory remains 2 GiB, swap zero, early stop 1.5 GiB and the thermal
cutoff 90°C. It retains 64 tasks, no new privileges, 8-GiB starting / 4-GiB stopping
host headroom, 180/195/210-second child/service/supervisor deadlines and new root
crash capture. The boundary passed 129 evaluations and 911,616 full-logit checks
with maximum error `0.00012683868408203125`, no sign or zero/nonzero mismatches and six exact
CPU greedy choices; HAG charge retired to zero. TCP passed 200 evaluations, including
a real socket cancellation with healthy reset/reuse, unload/reload and clean exits.
Process peaks were 102,744,064 / 305,504,256 bytes and maximum Tctl 75.75°C / 81.5°C,
with no swap, OOM or matching driver faults. Final root capture stopped cleanly with
zero dumps and a clean 1,327-line kernel window, and all 130 source / 12 gate inputs
were verified. The archive is `outputs/hag-c128-qualified-20261004.tar.gz`.
The two-token `<think>\n` protocol fixtures do not qualify complete-answer quality
or throughput; CPU remains the default. No C256 candidate is added.
See the [C128 contract](hag-backend.md#c128-implementation-and-qualification--2026-10-04).

### Recording and scratch storage reduction — 2026-10-04

The subsequent implementation stores compact dispatch descriptors and retains each
allocation owner once per batch. One workspace serves all serial scratch-using
commands; immutable parameters share storage by exact bits. Operation order,
per-dispatch fences and serving capacity 64 were unchanged for that frozen proof. Auxiliary peak falls
from 37,833,560 to 7,480,664 bytes, HAG charge from 689,303,552 to 658,952,192,
native recording metadata from 69,778,787 to 47,540,703, and the rounded host allowance
from 67 to 46 MiB. Runtime forecast is 1,485,916,864 bytes; the boundary probe's separate
conservative forecast is 1,519,718,559. **Both fresh live boundary/TCP gates passed.**

At that frozen C64 stage, C128 became a metadata-only candidate using
14,842,200 auxiliary bytes, 33,793,536 public tensor bytes and 681,910,272 HAG bytes.
Native metadata is 94,711,519 bytes and the conservative probe forecast is
**1,604,527,519 bytes**, under unchanged budgets. That archived report retains
`execution: not_requested`, `qualified: false`, `serving_capacity: 64`. The separate
C128 implementation received its own live boundary and TCP qualification above.

Forty-eight wrapper tests and a harmless containment self-check passed before
fresh live gates in `work/hag-recording-20261004/`. At 12:15–12:17 UTC the boundary
again checked 911,616 values, maximum error `9.1552734375e-5`, one sign mismatch and zero zero/nonzero
mismatches and six exact CPU greedy choices. Its 658,952,192-byte HAG charge returned
to zero. TCP completed 136 evaluations across six successes and one healthy-reset
socket cancellation, followed by reuse, unload/reload and CPU/HAG exits 0. Process
peaks were 79,228,928 / 279,310,336 bytes and maximum Tctl 82.875°C / 57.375°C; swap,
OOM and matching driver faults were zero, with the 90°C cutoff unchanged. These
are scoped correctness/lifecycle results for the same two-token `<think>\n`
references, not throughput or answer quality. The archive is
`outputs/hag-recording-qualified-20261004.tar.gz`. The [recording record](hag-backend.md#recording-and-scratch-storage-reduction--2026-10-04)
separates this frozen identity from the earlier C64 proof above. Measure preparation,
prefill/decode and cancellation latency before changing the current 3,793 separately
fenced dispatches per evaluated token. Smaller storage estimates do not establish
throughput, answer quality, concurrency or broader context coverage.

Operating-system integration requires a separate authority, memory, queue,
completion and failure-retirement contract. Linux handles and process-exit
quarantine are not a portable kernel contract.

## Run the model-side component gate

`QwenModel::dense_ffn_plan(layer)` derives exact ranges and logical payload sizes
from the existing strict binder. `prepare` checks source identity, formats and three
logical limits before reads, then uploads one packed matrix at a time. `record`
consumes preparation into a reusable four-command batch retaining its own scratch.
The caller waits for completion before changing input or reusing the batch.
The execution trait still lacks metadata-only shape/cost queries; backend shape or
physical-capacity refusal may occur during upload or recording. Logical limits do
not establish total hardware/process fit or backend support.

Build and inspect a real checkpoint without reading weight payloads:

```sh
python3 tools/run_dev_guarded.py -- \
  cargo build --release --locked -p beholdd-model --example recorded_dense_ffn
target/release/examples/recorded_dense_ffn \
  --model /absolute/path/to/Qwen3-0.6B-Q8_0.gguf --layer 0
```

The fixture reports 10,027,008 packed bytes, 3,342,336 maximum host staging bytes,
24,576 scratch bytes and 8,192 caller input/output bytes for one FFN. Backend
alignment/program/queue costs and execution temporaries are separate.

Explicit CPU execution compares three changed deterministic inputs against the
existing immediate CPU implementation, checking every output bit, batch reuse and
exactly three preparation reads. These are real weights with synthetic inputs,
not captured activations, a full model pass or a performance benchmark:

```sh
python3 tools/run_dev_guarded.py -- \
  target/release/examples/recorded_dense_ffn \
    --model /absolute/path/to/Qwen3-0.6B-Q8_0.gguf --layer 0 --execute-cpu
```

The example never selects HAG or falls back from a requested GPU. It currently has
no GPU execution option. Live native qualification remains in the HAG workstream.

Add `--block` to inspect or execute the normalized, residual-connected block:

```sh
python3 tools/run_dev_guarded.py -- \
  target/release/examples/recorded_dense_ffn \
    --model /absolute/path/to/Qwen3-0.6B-Q8_0.gguf --layer 0 --block --execute-cpu
```

`QwenModel::dense_ffn_block_plan(layer)` requires an F32 norm and derives
10,027,008 packed matrix bytes, 32,768 resident tensor bytes (norm, normalized
input and gate/up scratch), 3,342,336 maximum staging bytes, and 4,096 caller-owned
hidden bytes for this fixture. All three `DenseFfnBlockLimits` and source identity
are checked before reads or allocation. Four preparation reads total 10,031,104
bytes; the batch retains the norm and updates hidden in place. Replays do not reread
the source. The separate immediate CPU oracle does read its own reference weights;
`preparation_reads` does not count total example I/O.

RMSNorm, RoPE and incremental attention have CPU-qualified command semantics.
Native RMSNorm and the complete residual FFN block passed the frozen gates below.
Native RoPE/attention and model-owned KV lowering also passed their isolated and
real layer-0 component gates. The selected tied-vocabulary component also passed;
the complete resident plan subsequently passed its selected whole-model fit and
logit/token parity gates. Explicit HAG runtime selection is now implemented;
its bounded runtime and in-process HTTP/SSE serving gates have passed separately.

### Native FFN qualification probe

The separate `hag_dense_ffn_probe` example selects the experimental native HAG
executor explicitly. Its default invocation only inspects metadata:

```sh
python3 tools/run_dev_guarded.py -- \
  cargo build --release --locked -p beholdd-model --features hag-native \
    --example hag_dense_ffn_probe
target/release/examples/hag_dense_ffn_probe \
  --model /absolute/path/to/Qwen3-0.6B-Q8_0.gguf --layer 0
```

With the current eleven-program library, this fixture predicts 10,129,408 charged
HAG bytes, including page rounding, guards, shader/IB storage and a 32 KiB
auxiliary arena. The frozen four-command qualification below charged 10,125,312
bytes with its smaller shader allocation. It checks
this against a separate 32 MiB HAG budget before device access. Packed weights
remain Q8_0; preparation reads three ranges and stages one matrix at a time.
Metadata/tokenizer memory, CPU-oracle staging and driver-private allocations
still require host/process headroom outside that HAG budget.

Live modes require `HAG_TRACE_SUBMISSION=0`, fresh privileged crash capture,
fixed process and runtime limits, a 90°C thermal cutoff and host-memory checks.
First `--diagnose-first` checks a 65-row Q8_0 projection across two workgroups.
Only after that succeeds may `--execute` run three changed-input layer-0 FFNs:
gate/up 3,072 × 1,024, 3,072-element SwiGLU, and down 1,024 × 3,072.
Every output is compared with the immediate CPU oracle under the declared
`abs_error <= 3e-4 + 3e-5 * abs(reference)` bound, with finite/sign/zero checks.
Submit/wait must transfer no tensor payloads, replay must not reread weights,
and cleanup must return the HAG charge to zero. Consult the current
[qualification record](hag-backend.md) before live execution. This gate covers
the four-command FFN contribution; normalization and residual-block qualification
are separate.

### Native normalized residual-block probe

The separate `hag_ffn_block_probe` gates native RMSNorm before executing
`hidden += FFN(RMSNorm(hidden))`. Its default invocation checks metadata and cost
without device access or weight-payload reads:

```sh
python3 tools/run_dev_guarded.py -- \
  cargo build --release --locked -p beholdd-model --features hag-native \
    --example hag_ffn_block_probe
target/release/examples/hag_ffn_block_probe \
  --model /absolute/path/to/Qwen3-0.6B-Q8_0.gguf --layer 0
```

With the current eleven-program library, the selected layer predicts **10,133,504
charged HAG bytes** under a 32-MiB HAG budget. The frozen six-program result below
charged 10,129,408 bytes. Its 36,864-byte tensor arena includes the F32 norm and caller hidden
state; 8,204 bytes of retained parameters/scratch fit the 32-KiB auxiliary arena.
Preparation reads three packed matrices plus the norm once, totaling 10,031,104
bytes; it stages at most 3,342,336 bytes of weights at once. The independent CPU
oracle performs its own sequential reads before native device construction.
No complete checkpoint cache is created.

The bounded live diagnostic and selected layer-0 block passed on 2026-10-03,
as recorded below. `--diagnose-first` checks an unweighted one-element case and
three weighted 65-element rounds with changed
inputs and weights, two-workgroup tail masking and guarded output subviews. Its
numerical bound is `2e-6 + 2e-6 * abs(reference)`. Only after that isolated gate
and its driver checks pass may `--execute` run three reused five-command block
submissions, 21 physical dispatches, comparing all 3,072 final values with the
independent single-worker CPU oracle under `3e-4 + 3e-5 * abs(reference)`.

Both gates require finite/sign/signed-zero checks, no submit/wait tensor traffic,
stable residency/source counts and zero final HAG charge. Verified process/cgroup
limits, zero swap, fresh crash capture and a 90°C cutoff remain prerequisites.
The frozen gate's 102 hardware-free backend tests and six-shader LLVM check
passed; those checks alone did not establish hardware reciprocal/rsqrt accuracy
or block GPU parity. Current source has eleven programs; its separately qualified
RoPE/attention fixtures are recorded below.
See the [live status](hag-backend.md#native-rmsnorm-and-normalized-residual-block--2026-10-03).

### Resident attention block and native qualification — 2026-10-03

`QwenModel::dense_attention_plan(layer)` derives exact four-matrix Q8_0 and
three-norm F32 spans without payload reads. It validates dimensions, finite
nonnegative epsilon and positive finite RoPE base. `memory(capacity)` bounds
capacity to 1..256 and the model context; `prepare` validates source identity,
formats and all logical limits before the first source read or device allocation.
Preparation stages one source tensor at a time, creates complete coefficient
tables and owns shared scratch plus fixed position-major K/V storage.

`record(device, hidden)` consumes preparation and binds one hidden view into one
batch per cache position. For this model each batch has 54 semantic commands:
input norm, Q/K/V projections, 24 weighted Q/K head norms, 24 absolute-position
split-half RoPE operations, causal GQA, and O projection added to the residual.
The native lowering requires 127 dispatches. V projection and K head normalization
write the uncommitted tail directly; only the exact visible prefix is attended.

The cache exposes `step`, `reset`, `capacity`, `visible_positions` and `is_poisoned`.
A step marks uncertainty before submit and publishes its next position only after
waiting for the complete batch, including residual output. Submit/wait errors or
unwinds poison the cache without publishing that position; further step/reset calls then
refuse. A healthy reset sets visible count to zero without clearing/uploading
storage. The next request overwrites every newly visible K/V row before use.
Private batches share scratch and cannot be cloned through the cache API.

At capacity eight, this layer requires 6,684,672 packed bytes, 107,520 model-owned
tensor bytes and 4,096 caller-hidden bytes. The public tensor arena is therefore
111,616 bytes. Seven native source reads total 6,689,792 bytes; maximum staging is
2,228,224 bytes. Eight retained batches consume 165,168 auxiliary bytes in a
256-KiB arena. With the ten-program library, `allocation_bytes()` predicts
**7,098,368 HAG bytes**, checked before opening a device against a 32-MiB budget.
This component forecast does not include metadata, CPU oracle or driver memory
and is not a whole-model fit decision.

The new `hag_attention_block_probe` defaults to metadata inspection:

```sh
python3 tools/run_dev_guarded.py -- \
  cargo build --release --locked -p beholdd-model --features hag-native \
    --example hag_attention_block_probe
target/release/examples/hag_attention_block_probe \
  --model /absolute/path/to/Qwen3-0.6B-Q8_0.gguf --layer 0
```

Its explicit `--execute` mode additionally needs a render node,
`HAG_TRACE_SUBMISSION=0` and the external process/thermal/crash supervisor. It
refuses packed payload above 16 MiB, public tensors above 128 KiB or staging above
4 MiB before native access. The standalone `hag_attention_probe` must first pass
13 RoPE dispatches checking 1,562 values and 26 guards exactly, then 183 attention
dispatches checking 3,218 values and 14 guards under
`2e-5 + 2e-5 * abs(reference)`.

Only then may the real-layer gate run two requests of three sequential positions,
resetting and changing hidden inputs between requests. The contract is six logical
submissions, 762 dispatches and 6,144 final hidden checks under the preregistered
`3e-4 + 3e-5 * abs(reference)` bound with finite/sign/zero checks. The independent
one-worker immediate CPU oracle reads seven tensors per step, 42 total, before
native construction. Native preparation remains seven reads across both requests.
Only final hidden is read back: 24,576 bytes; intermediates and KV remain resident.
Submit/wait must cause zero tensor traffic, residency must stay fixed, reset must
report zero visible positions and cleanup must return HAG budget use to zero.

All three gates passed at 20:19–20:20 UTC on 2026-10-03. RoPE matched all
1,562 values and 26 guards bitwise across 13 dispatches, including the code-page
crossing. The attention diagnostic passed 3,218 values and 14 guards across 183
dispatches with maximum absolute error `1.1920928955078125e-7`. Its GQA fixtures
cover three-query prefill and two cached queries beginning at position 63, with
NaN-filled future cache rows and changed-input reuse; those are isolated fixtures.

The real layer-0 block passed both three-position requests: six submissions,
762 dispatches and 6,144 final hidden checks, with maximum absolute error
`2.384185791015625e-7` under the unchanged `3e-4 + 3e-5 * abs(reference)` bound.
**Eight positions were allocated; only three serial positions per request were
executed.** Reset reported zero visible positions, native preparation stayed at
seven reads versus 42 CPU-oracle reads, and submit/wait tensor transfers stayed
zero. Final verification read 24,576 bytes. Actual HAG charge was 7,098,368 bytes,
with 165,168 live auxiliary bytes, and returned to zero after cleanup.

Peak process memory was 21,790,720 bytes for RoPE, 22,134,784 for attention and
50,274,304 for the real block; maximum sampled Tctl was 65.375°C, 63.625°C and
55.75°C respectively. All gates used verified 1-GiB process/no-swap limits and
the unchanged 90°C cutoff, with zero memory/OOM events or matching GPU faults.
Fresh root capture covered all three gates and stopped cleanly with zero dumps;
the complete 213-line kernel window contained no matching GPU/OOM/fault messages.
An earlier thermal hold at 92.6°C remained in force until the cooler rerun; no
threshold or numerical bound was relaxed.

The [qualification record](hag-backend.md#native-rope-attention-and-resident-attention-block--2026-10-03)
contains exact frozen identities and supervision. Evidence is retained in
`outputs/hag-rope-attention-qualified-20261003.tar.gz` in the development chat.
These are real weights with changed synthetic boundary hidden inputs, checking
only final block output. They do not establish captured-activation, full-model,
logit/token or throughput parity. Embedding/head execution, whole-model fit and
serving integration remained open at this stage, when native serving advertised only CPU.
The historical results below retain their original costs and proof scopes.

### Component result — 2026-10-03

The migrated Qwen3-0.6B-Q8_0 fixture's layer 0 passed all three CPU parity rounds:
every output matched bitwise, four commands were reused, and preparation performed
exactly three reads totaling 10,027,008 bytes. The guarded unit exited successfully,
with 59.4 MiB peak process memory, zero swap and 78.5°C maximum sampled Tctl. The
2.050-second service runtime includes supervision and oracle work and is not an
inference latency measurement. The hardware-free workspace suite passed 395 tests,
with nine opt-in cases ignored. Full-model GPU execution was unqualified at this stage;
the later complete-model and serving results above retain separate scopes.

### Normalized block result — 2026-10-03

Layers 0 and 27 each passed three changed-input CPU parity rounds with real
checkpoint weights. All 3,072 checked outputs per layer matched bitwise, five
commands were reused, and each preparation read four ranges totaling 10,031,104
bytes. Layer 0 peaked at 65.4 MiB and 41.875°C; layer 27 peaked at 58.6 MiB and
42.25°C. Both supervised units used zero swap and completed successfully in about
2.03 seconds, including supervisor and oracle overhead. The original four-command
layer-0 gate also retained its bitwise result.

The combined workspace at this checkpoint passed 410 tests (including one doctest)
with HAG native, Vulkan and memory fault injection, and nine opt-in tests ignored.
Its guarded unit peaked at 1.8 GiB with zero swap. The new normalization command and
block tests cover aliasing, replay dependencies, source lifetime, foreign owners,
malformed shapes and refusal one byte below each logical budget. These results
qualify the CPU command/model contract, not HAG normalization or full-model serving.

### Native HAG FFN result — 2026-10-03

The selected layer-0 four-command FFN passed three changed-input GPU rounds at
14:23 EDT on the allowlisted GC 11.5.1 host. All 3,072 checked outputs met the
preregistered tolerance; maximum absolute error was `1.1920928955078125e-7`.
The 12 dispatches reused 10,027,008 packed-weight bytes, with three native
preparation reads and no rereads or tensor transfers during submit/wait. The
independent CPU oracle made nine separate weight reads. Boundary verification
read back 12,288 bytes; the complete 10,125,312-byte HAG charge returned to zero.

The process peaked at 64.1 MiB under an actual 1-GiB cap, with zero swap and zero
memory/OOM events. Maximum sampled Tctl was 44.625°C, and the driver/crash checks
were clean. Before this run, a 65x32 Q8_0 diagnostic qualified two-group indexing
and output guards. An earlier harmless self-check correctly refused to start at
90.625°C; execution resumed only after the host cooled and fresh capture was armed.

This cleared the model-sized Q8_0/SwiGLU FFN contribution gate. The result alone
did not qualify the extended BF16/scaled-add payloads or native RMSNorm. The
subsequent normalized-block result below covers its norm and residual-add shapes.
RoPE, attention/KV, embeddings,
full-model parity and serving integration remained open at this stage. The exact binary, source
snapshot and logs are archived in `hag-model-sized-qualified-20261003.tar.gz`
in the HAG development chat's outputs.

### Native normalized residual-block result — 2026-10-03

At 15:21 EDT (19:21 UTC), the isolated RMSNorm diagnostic passed 196 checked values
and eight output guards across seven dispatches: unweighted width 1 and three
changed-input/weight weighted-width-65 rounds. Maximum absolute error was zero
under `2e-6 + 2e-6 * abs(reference)`. The two-workgroup tail and guard checks passed;
45,056 charged HAG bytes were fully released.

The real Qwen3 layer-0 normalized residual block then passed three reused rounds,
21 dispatches and 3,072 final hidden-state checks under the unchanged FFN tolerance.
Maximum absolute error was `2.384185791015625e-7`. This exercised RMSNorm width
1024, Q8_0 gate/up 3072x1024, SwiGLU width 3072, Q8_0 down 1024x3072 and residual
add width 1024. Numerical checks covered final block outputs, rather than every
intermediate. The observed 10,129,408-byte HAG charge matched preflight and
returned to zero. Four native preparation reads versus 12 independent CPU-oracle
reads were recorded; submission transferred no tensor payloads, and final
verification read 12,288 bytes.

The block process peaked at 66,781,184 bytes (63.6875 MiB), with zero memory
events/swap and maximum sampled Tctl 54.75°C. Minimum host available memory was
18,485,620,736 bytes. Both gates verified their cgroup and 90°C thermal limits,
reported clean driver evidence and `qualified=true`; fresh read-only crash capture
stopped cleanly with zero dumps. The frozen binary SHA-256 was
`7e191d5d149a4f8f51075c619c4f846581a5fe0f0da2166a78020ced465f227f`.
Full identities and supervision are in the
[qualification record](hag-backend.md#native-rmsnorm-and-normalized-residual-block--2026-10-03)
and `outputs/hag-rmsnorm-qualified-20261003.tar.gz` in the HAG development chat.

This qualifies the stated finite diagnostic and selected normalized block only.
The extended BF16 payload, arbitrary widths/inputs, per-head-width-128 RMSNorm,
RoPE, attention/KV, embeddings, full-model logits/tokens and end-to-end performance
remained open at this stage. Native serving was then CPU-only.
