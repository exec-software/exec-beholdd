# Native serving

beholdd v2 now has one path from an HTTP or embedded request to its native kernels. This
M3 slice is implemented. Real-checkpoint serving, SSE, tool translation and execution,
worker-liveness, embedded/HTTP parity, and product-client result replay gates have passed
for the documented local Qwen3-Coder-30B path. This is a scoped integration result, not a
claim of production throughput or a clean service-lifecycle test.

The current native boundary is intentionally narrow:

- GGUF checkpoints with exact `general.architecture` tags `qwen3` or `qwen3moe`;
- CPU execution by default, with explicit experimental HAG selection for the
  bounded dense Qwen3-0.6B graph described below;
- exact resident or exact streamed checkpoint bytes, selected explicitly;
- lazy model loading and process-observed `loaded` state;
- OpenAI-compatible health, capability-bearing model-list, and chat-completion routes;
- deterministic Qwen3-Coder chat and native tool-call translation; and
- the same `NativeRuntime` available in process through `beholdd-embed`.

There is no subprocess or compatibility fallback. An unsupported architecture, unsafe
storage device, impossible memory plan, unsupported request value, or malformed native
tool call is an error.

## Start the server

Place the checkpoint at the path named by `config.example.yaml`, or copy the example and
change its path. Relative checkpoint paths resolve relative to the configuration file,
not the process working directory.

```sh
beholdd serve --config ./config.example.yaml
```

The complete command is:

```text
beholdd serve [--config PATH] [--host HOST] [--port PORT]
```

When `--config` is omitted, beholdd reads `$BEHOLDD_HOME/config.yaml`, or
`$HOME/.beholdd/config.yaml` when `BEHOLDD_HOME` is unset. The default listener is
`127.0.0.1:6768`. Port 6768 is deliberate: the legacy v1 legacy service service remained
untouched on 6767 during the completed v2 gate. Use explicit overrides when another
isolated listener is needed:

```sh
beholdd serve --config ./config.example.yaml --host 127.0.0.1 --port 6770
```

The CLI handles SIGINT and SIGTERM with graceful HTTP shutdown, then waits for
admitted blocking planning/generation workers to retire before returning. A
disconnected request keeps admission until its worker finishes. Shutdown does not
forcibly cancel an in-flight GPU operation or impose a deadline on active work.

Creating the runtime registers configured IDs but does not open or allocate a model. The
first `/v1/chat/completions` request for an ID performs storage checks, binds the GGUF and
tokenizer, makes a fit plan, and materializes the configured weight source. With the
resident example that first request reads the full 32.48 GB checkpoint, so startup
latency belongs to that request. Before a successful bind, `/v1/models` reports
`loaded: false`, `compatibility: "unverified"`, and an empty `capabilities` array; a bad or
unsupported configured path therefore makes no executable-model claim. A published model
reports `compatibility: "compatible"` and `capabilities: ["completion"]`, while `loaded`
continues to come from the live model/weight-source probe.

M3 keeps at most one model bound in a `NativeRuntime`. Before a request loads or uses its
selected ID, the runtime evicts every other bound model (including resident weights,
streaming caches, and prefetch workers). Switching IDs therefore reloads the model on its
next request. This explicit policy prevents independently forecast per-model budgets from
stacking into an unaccounted process-wide allocation; multi-model concurrent residency
requires a future global reservation scheduler.

## Configuration

Configuration is strict YAML. Unknown fields, duplicate model IDs, invalid enum values,
inconsistent prefetch limits, and resident configurations with nonzero streaming caches
are rejected. The root fields are:

| field | required/default | meaning |
|---|---|---|
| `models` | default `[]` | Models exposed by this process. |
| `memory_budget_bytes` | optional | Default hard execution budget for each model. A model entry may override it. If neither is set, the runtime uses half of the currently available memory. |
| `runtime_headroom_bytes` | default `4294967296` (4 GiB) | Default allowance for indexes, stacks, executable mappings, and allocator overhead. |

Each `models` entry accepts exactly these fields:

| field | required/default | meaning |
|---|---|---|
| `id` | required | Stable string used by the API and embedded requests. IDs must be unique, nonempty, and have no surrounding whitespace. |
| `path` | required | Only GGUF file, or first shard of a split GGUF. Relative paths resolve beside the YAML file. |
| `execution_backend` | default `cpu` | `cpu` or explicit experimental `hag`; HAG requires Linux and the `hag-native` build feature. |
| `hag` | absent | Required only for HAG: strict object with absolute `device: /dev/dri/renderD<number>`, positive `memory_budget_bytes` for GPU allocation, and optional `context_capacity` from 1 through 16384 (default 2048 evaluated positions). CPU entries reject this object. |
| `weight_mode` | default `streaming` | `streaming` reads exact weight ranges on demand; `resident` materializes every encoded weight before generation. |
| `fit_policy` | default `exact` | One of `exact`, `degrade`, or `refuse`. The current serving target uses `exact`; if semantic degradation would be required, the runtime still refuses because no degradation executor is present. |
| `memory_budget_bytes` | optional | Per-model hard budget, overriding the root value. CPU refuses a budget larger than current available memory. HAG compares its combined host/UMA forecast with both this budget and current available memory. |
| `runtime_headroom_bytes` | optional | Per-model override of the root runtime allowance. |
| `allow_unsafe_storage` | default `false` | When false, refuse network, rotational, mixed, or unidentified checkpoint storage. Set true only as an explicit acceptance of that I/O risk. |
| `pinned_cache_bytes` | default `0` | Streaming cache for the repeated cyclic weight prefix. |
| `demand_cache_bytes` | default `0` | Streaming LRU cache for data-dependent reads such as routed experts. |
| `prefetch_bytes` | default `0` | Bytes reserved for speculative reads. |
| `prefetch_queue_depth` | default `0` | Maximum outstanding speculative requests. |
| `prefetch_workers` | default `0` | Maximum concurrent speculative reads. |
| `near_miss_prefetch` | default `false` | Enable the measured experimental near-miss expert prefetch policy. |

`prefetch_bytes`, `prefetch_queue_depth`, and `prefetch_workers` must be either all zero or
all nonzero. In `resident` mode every streaming cache and prefetch field must be zero. In
`streaming` mode those byte limits are requested maxima: the exact fit planner can cap
them to the remaining authorized budget.

The checked-in example is deliberately explicit:

```yaml
# Proven inside a separate 48 GiB no-swap service cap.
memory_budget_bytes: 40000000000
runtime_headroom_bytes: 4294967296 # 4 GiB

models:
  - id: qwen3-coder-30b
    path: models/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf
    weight_mode: resident
    fit_policy: exact
    allow_unsafe_storage: false
    pinned_cache_bytes: 0
    demand_cache_bytes: 0
    prefetch_bytes: 0
    prefetch_queue_depth: 0
    prefetch_workers: 0
    near_miss_prefetch: false
```

The model is labelled 30B parameters, just below the preferred 32B–120B testing band,
and its Q8_0 checkpoint occupies 32,483,934,528 bytes. It is the checkpoint used for
the scoped M3 serving, tool and product-client gates above. Dense Qwen3 also has a
strict real-checkpoint bind and stable native decode gate on Qwen3-0.6B-Q8_0;
its independent f64 result agrees with beholdd but does not select
the same first token as llama.cpp's activation-requantized Q8_0 CPU path, so no llama.cpp
token-parity claim is made. See the [dense Qwen3 gate record](research/native-qwen3-dense.md).
Qwen3.6 (`qwen35moe`), Qwen3-Coder-Next (`qwen3next`), dense Qwen3.5 (`qwen35`), and
GPT-OSS (`gpt-oss`) are different architectures and are rejected today. Nemotron 3
Super (`nemotron_h_moe`) is also rejected today and is the next explicit native
architecture target. Its hybrid SSM/attention state and latent/shared MoE require a
separate binder and execution path; recognizing its GGUF header must not publish a model
or capability. The frozen Q4_K_M fixture, staged gates, and Q4/Q8/BF16 provenance are in
the [native Nemotron plan](research/native-nemotron-h-moe.md).

## Experimental HAG selection

The separate [`config.hag.example.yaml`](../config.hag.example.yaml) configures
Qwen3-0.6B Q8_0 explicitly. Build the CLI with `--features hag-native` and set
`HAG_TRACE_SUBMISSION=0` when executing it. Ordinary builds/configurations retain
CPU behavior. HAG does not select a render node automatically or fall back to CPU.

HAG requires `fit_policy: exact`, `weight_mode: streaming`, zero streaming cache and
prefetch limits, and `near_miss_prefetch: false`. Streaming here means sequential
bounded preparation reads; all prepared GPU weights stay resident afterward.
The model's `hag.context_capacity` selects 1–16384 evaluated positions and defaults
to 2048. Admission requires
`prompt_tokens + max_new_tokens - 1 <= context_capacity`, including chat framing. Omitted
`max_tokens` is resolved after tokenization to the smaller of 512 and the remaining
output budget. Explicit limits are preserved or refused with typed HTTP 400 before
either JSON generation or SSE headers. **The scoped C128 boundary and TCP gates passed**.
The bounded C64 gate passed October 4;
the historical October 3 serving proof used a frozen C16 binary.
The selected model geometry, types, tied vocabulary, epsilon and RoPE base are checked
before native allocation.

The example's combined budget is 2 GiB with 128 MiB runtime headroom and a separate
1152-MiB HAG budget. The current on-demand layout's combined forecast is
1,914,935,488 bytes, including 1 MiB of recording/owner metadata separately
from general runtime headroom; HAG allocation is 1,135,140,864 bytes. These fit checks precede device opening
and payload reads. Process/cgroup usage and HAG/GEM allocation measure different
resources. Follow the [bounded development guidance](development.md#explicit-experimental-hag-serving)
for the external process, thermal and crash-capture controls used for this experiment.

[`config.hag-16k.example.yaml`](../config.hag-16k.example.yaml) selects C16384
explicitly. With 128 MiB runtime headroom, its native allocation forecast is
**4,629,168,128 bytes** and combined host/UMA forecast is **5,417,203,904 bytes**.
The example provides a **6-GiB combined budget** and **4608-MiB native budget**;
run it only inside a separately configured finite process cgroup with zero swap
and host/thermal monitoring. YAML budgets do not create that cgroup. The capacity
expansion passed bounded hardware-free validation; no live C16384 run is
qualified. Earlier C2048 results retain their frozen source identities.

Full-model C2048 numerical qualification has passed. The exact synthetic no-tools
Antimatter project-instructions profile also
[qualified on validation-host on October 8, 2026](hag-editor-context.md#qualified-c2048-project-instructions-on-validation-host--2026-10-08),
with Linux `6.17.0-1024-oem`: a 467-token prompt and 128-token reply allowance,
three native requests evaluating 473 positions each, and exact CPU text, EOS and
usage parity. This is bounded experimental serving evidence, not qualification
of every C2048 request, full editor/tools, general quality, concurrency, throughput
or live cancellation. The checkpoint-rendered P491 seeded-history profile also
[passed all four stages on October 11](hag-editor-context.md#standard-validation-host-history-parity-and-response-timing--2026-10-11),
including exact provider chat/SSE parity with phase logging disabled. The separate
[C2048 TCP lifecycle profile](hag-editor-context.md#qualified-c2048-tcp-lifecycle-on-validation-host--2026-10-08)
passed on October 8 with real socket cancellation, healthy reset/reuse, busy-unload409,
idempotent unload, same-process reload and clean idle SIGTERM on validation-host.
It uses a nine-token Hello prompt and one-token successful replies, with 46 native
evaluations total; it does not qualify cancellation through the Antimatter provider
or shutdown during active inference. Historical serving results below retain their original
frozen capacities and source identities. See the [qualification runner](hag-qualification.md).

Explicit HAG configurations report `execution_backends: ["cpu", "hag"]`; this names
implemented paths and does not guarantee a checkpoint/device will pass lazy admission.
Cancellation is checked around complete synchronous token evaluations, including
prefill before text exists. Healthy cancellation resets KV visibility for reuse.
Device or cleanup uncertainty disables HAG process-wide until restart; explicit
unload reports failure if retirement was uncertain. Admission/serialization is per
`NativeRuntime` instance, and this setup uses one instance.

The [bounded serving gate](hag-backend.md#experimental-hag-runtime-and-httpsse-qualification--2026-10-03-edt)
passed exact CPU token/text/usage parity, actual SSE-body disconnection and reuse,
busy-unload refusal, and unload/reload. It used the in-process Axum router and live
positions 0..9. It does not establish TCP-socket behavior, all sixteen positions,
long conversations, tools/vision, throughput, concurrent instances/processes or shared
devices. The earlier 30B CPU/product-client proof retains its separate scope.

The October 4 C64 boundary and actual CLI/TCP gates passed. The boundary probe
checked 911,616 full-logit values at selected positions through 63 and reset/replay;
all six greedy choices matched CPU. The TCP gate completed six requests and one
socket cancellation across 136 evaluations, covering a 21-token conversation,
a 63-token prompt with an omitted limit resolved to two, typed refusals, healthy
reuse, unload/reload and CPU/HAG SIGTERM exit zero. All three reference fixtures
emitted the two-token prefix `<think>\n`; complete-answer quality and throughput
were not qualified. Process peaks were 89,931,776 / 280,555,520 bytes and maximum
Tctl 77.625°C / 59.625°C for boundary / TCP, with no swap, OOM or matching driver
faults.
Optional `HAG_SERVING_AUDIT=1` writes only request sequence, evaluated positions,
outcome, reset status and poison status to stderr. It omits prompts, token IDs,
logits and tensors. See the [C64 qualification record](hag-backend.md#context-aware-c64-and-tcp-qualification--2026-10-04).

### Bounded phase diagnostics

For bounded HAG diagnostics, exact `HAG_SERVING_PHASES=1` additionally emits
`HAG_SERVING_PHASE` JSON lines to stderr. These identify generation admission,
cold metadata and preparation boundaries, session reset, prefill and decode,
plus completed-position progress at the first position and every 64 positions.
Each line has a process ID, independent request sequence, Unix microseconds,
elapsed monotonic microseconds and a completed-position count. No prompt,
model path, token ID, logits or tensor content is recorded. The option is off
by default and limited to 128 records per generation and 4,096 per process.

A position marker means the synchronous resident step returned successfully;
it does not assert subsequent accounting checks, readback or token delivery.
Session reset markers describe the model session, not a GPU device reset.
`request_complete` describes runtime completion, not HTTP delivery.
Missing markers after errors, termination or a record limit prove no completion
or cleanup. These generation receipts do not cover separate HTTP prompt
planning, and are not a replacement for `HAG_SERVING_AUDIT`. Writes are
best-effort but synchronous, so a blocked stderr sink can delay a diagnostic
run. Retain finite external deadlines and the existing thermal/memory guards.

### Recording storage update — 2026-10-04

The new layout reuses one private scratch workspace per recorded batch, shares
identical scalar parameters by exact bits and retains allocation owners once per
batch. Arithmetic, per-dispatch fences, request planning and serving capacity 64
were unchanged for that frozen proof. Auxiliary peak fell to 7,480,664 bytes and
the corresponding C64 runtime forecast was 1,485,916,864 bytes. **Fresh boundary/TCP gates
passed** at 12:15–12:17 UTC, with the same 911,616 checked values, six exact CPU greedy
choices and 136 TCP evaluations. Process peaks were 79,228,928 / 279,310,336 bytes,
maximum Tctl 82.875°C / 57.375°C, and neither stage recorded swap, OOM or matching
driver faults; the 90°C cutoff was unchanged. These two-token fixtures do not
qualify answer quality or throughput. The previous dated result used separate frozen bytes.

That archived C64 report treated C128 as a metadata-only candidate and retained
`qualified: false` and `serving_capacity: 64`. See the
[recording record](hag-backend.md#recording-and-scratch-storage-reduction--2026-10-04).

### C128 implementation and qualification — 2026-10-04

At this milestone, source implemented a 128-position allocation and planning contract;
**both live boundary/TCP gates passed** in `work/hag-c128-20261004/`.
The new gate compares selected full logits through position 127 plus reset/replay,
then checks an omitted output limit on a 127-token chat prompt and explicit overflow
for Hello with 121 requested tokens. Real socket cancellation, healthy reuse,
unload/reload and clean signal exit retain separate assertions.

The fresh supervisor preregisters **75% CPU**, reduced for the doubled CPU oracle
work, with unchanged 2-GiB memory, zero swap, 1.5-GiB early stop and 90°C cutoff.
Its child/service/supervisor deadlines remain 180/195/210 seconds. The conservative
probe forecast is 1,604,527,519 bytes; this is not a measured process peak.
The boundary checked 911,616 values with maximum error `0.00012683868408203125`,
no sign or zero/nonzero mismatches and six exact greedy choices, and retired its HAG charge.
TCP made 200 evaluations across six successes and one real socket cancellation:
128, 10, 22, 10, 10 (cancelled), 10, 10, all with healthy reset and no poison.
Four invalid/overflow requests were refused, busy unload returned HTTP 409, and
reuse and unload/reload passed, and both idle CLI processes exited with code zero
after SIGTERM. Process peaks
were 102,744,064 / 305,504,256 bytes, maximum Tctl 75.75°C / 81.5°C, and swap, OOM
and matching driver faults were zero. Final root capture had zero dumps and a
clean 1,327-line kernel window; all 130 source / 12 gate inputs were verified.
Evidence archive: `outputs/hag-c128-qualified-20261004.tar.gz`. The three CPU
references emitted only two tokens, `<think>\n`, so complete-answer quality and
throughput remain unqualified. CPU remains the default; no C256 candidate is added.
See the [C128 gate contract](hag-backend.md#c128-implementation-and-qualification--2026-10-04).

### Single-list EOS parity and descriptive timing — 2026-10-04

The later original integer-list prompt passed exact CPU/HAG raw text, usage and
EOS finish parity with 29 prompt + 27 emitted tokens. The semantic check permits
one optional terminal period; raw parity is unchanged. One native answer and four
timing requests each made 56 evaluations, reset healthily and unloaded cleanly.
First lazy-load response took 12.439 seconds; three warm SSE responses reached
first nonempty content in 5.753–5.944 seconds and DONE in 11.588–11.821 seconds.
SSE has no usage field; counts come from the exactly matched nonstream reference.
Content includes thinking markup, and the observed 2.284–2.330 API tokens/second
includes prefill, EOS and request overhead; this is not decode throughput.
Four prior semantic/format failures and one thermal abort remain preserved.
Recall and general answer quality remain unqualified; CPU remains the default.
See the [exact fixture, timing and closeout record](hag-backend.md#integer-list-eos-parity-and-response-timing--2026-10-04).

## HTTP API

The server exposes these routes:

| method and path | result |
|---|---|
| `GET /health` | `{"status":"ok"}`; does not load a model. |
| `GET /v1/beholdd/info` | Versioned serving-runtime identity and implemented operations; does not inspect or load checkpoints. |
| `GET /v1/models` | OpenAI-style model list with observed `loaded`, binding-derived `compatibility`, explicit `capabilities`, and `execution_limits` on each entry. |
| `POST /v1/models/{id}/unload` | Release an idle registered model; return `{"id":"...","loaded":false}`. |
| `POST /v1/chat/completions` | Non-streaming JSON or SSE chat completion. The first request lazily loads its model. |

Check health and configured residency:

```sh
curl -s http://127.0.0.1:6768/health
curl -s http://127.0.0.1:6768/v1/beholdd/info
curl -s http://127.0.0.1:6768/v1/models
```

For a CPU-only configuration, the native runtime's identity response is deliberately
narrower than the research code. Explicit HAG configuration adds `"hag"` to
`execution_backends`, as described above:

```json
{
  "schema_version": 1,
  "service": "beholdd",
  "generation": 2,
  "runtime": "native",
  "execution_backends": ["cpu"],
  "architectures": ["qwen3", "qwen3moe"],
  "model_formats": ["gguf"],
  "operations": {"unload": true}
}
```

These are implementation facts, not a guarantee that any configured file is compatible.
Per-model binding and residency remain the responsibility of `/v1/models`. GPU operator
experiments, MiMo's correctness harness, and planned Nemotron support are not serving
capabilities. Alternate runtimes default to `runtime: "unknown"`, empty backend,
architecture, and format lists, and `operations: {"unload": false}`; the HTTP wrapper
never labels an arbitrary runtime as native. `generation: 2` identifies the Beholdd
implementation generation, independently of rolling repository release numbers.

Each HAG model entry includes `execution_limits.max_evaluated_tokens` equal to
its configured `hag.context_capacity` (2048 by default); CPU reports
`{"max_evaluated_tokens": null}`.
This is a metadata-free implementation limit; it does not open a checkpoint,
establish compatibility or publish residency. The runtime API exposes the same
value as `ExecutionLimits { max_evaluated_tokens: Option<usize> }`.

Unload an idle model without deleting its registration, configuration, or checkpoint:

```sh
curl -s -X POST http://127.0.0.1:6768/v1/models/qwen3-coder-30b/unload
```

Encode the complete ID as one URL path segment (`owner/model` becomes `owner%2Fmodel`).
The server decodes it exactly once. An already-unloaded registered model succeeds
idempotently; its next completion loads it again. An unknown ID returns HTTP 404 with
`code: "model_not_found"`; an alternate runtime without this operation returns HTTP 501
with `code: "unsupported_operation"`. Active generation, loading, or cleanup returns
HTTP 409 with `code: "runtime_busy"` instead of queueing an unload to happen later.
Errors use the same OpenAI-style `error` envelope as completions.

Native unload removes the published model under runtime-wide admission, releases the
registry lock, then drops resident weights or streaming caches/workers before releasing
admission. Cleanup runs on a blocking worker, so health, identity, and model-list reads
stay responsive; disconnecting the unload request does not prematurely release admission.
Concurrent completion requests wait until accepted cleanup is finished. These management
routes inherit the server's loopback-local trust model; they do not add authentication.

Make a deterministic non-streaming completion:

```sh
curl -s http://127.0.0.1:6768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64,
    "temperature": 0
  }'
```

`temperature` must be omitted or exactly `0`; sampling is not yet implemented.
Explicit `max_tokens` must be between 1 and 8192. CPU uses 512 when it is omitted.
HAG first tokenizes the rendered prompt, then chooses
`min(512, context_capacity - prompt_tokens + 1)` for an omitted limit. The final selected output
is not evaluated, so a 2047-token prompt may request two output tokens at C2048.
Explicit overflow returns HTTP 400 with `type: "invalid_request_error"`,
`code: "context_length_exceeded"` and `param: "max_tokens"`; it is not silently
clamped. Both streaming and non-streaming requests finish planning before response
headers. Planning may read tokenizer metadata but does not read weight payloads,
open the GPU, publish a model or reset caches. Message content may be a
string, null where the role permits it, or an OpenAI content array made entirely of
`{"type":"text","text":"..."}` parts. Text parts are concatenated in order before
prompt rendering. An `image_url` part is recognized but current native models reject it
with HTTP 400, `type: "invalid_request_error"`, `param: "messages"`, and
`code: "unsupported_content_type"`; it is never silently discarded.

Native HTTP chat rendering now selects from exact `tokenizer.chat_template`
bytes during admitted, cancellable planning. The supported Qwen3-0.6B template
uses its history rules: a prior assistant's completed thinking block
is removed from the rendered prompt before the last user query, while the wire
messages remain unchanged. The separately matched Qwen3-Coder template retains
the tool renderer below. Missing or unknown native templates return HTTP 400
with `code: "unsupported_chat_template"` and `param: "model"` before generation.
See the [exact template identities and provenance](../crates/beholdd-server/src/templates/README.md).
The Qwen3-0.6B path supports function definitions and paired tool history using
its separate JSON tool-call dialect. No
`enable_thinking` or `reasoning_content` field is accepted; `/no_think` remains
ordinary prompt text. This selection applies to HTTP chat rendering; embedded
generation still accepts an already-rendered prompt.

Capabilities come from a successfully published executable runtime model, never from an ID,
configured path, or GGUF architecture name that has not been bound. Consequently a lazy
entry is `unverified` with no capabilities; the current native model becomes `compatible`
with `capabilities: ["completion"]` only after successful binding. Vision is intentionally
not representable yet because the runtime request contains only rendered text. The concrete
native implementation and removal gates are tracked in the [Qwen3.6 vision replacement
plan](research/native-qwen36-vision.md).

For SSE, add `stream: true` and keep curl's output unbuffered:

```sh
curl -N http://127.0.0.1:6768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "Count from one to five."}],
    "max_tokens": 64,
    "stream": true
  }'
```

Text deltas are emitted as `chat.completion.chunk` events. A final chunk carries the
finish reason, followed by `data: [DONE]`; a keepalive comment is sent every ten seconds
while generation is quiet. Tool generations are validated as a complete native tool
block before their translated tool-call delta is emitted.

## Tools

Clients send standard OpenAI function definitions. beholdd validates their supported
JSON Schema subset and renders the selected exact checkpoint template. Qwen3-0.6B
uses complete JSON tool objects with Transformers v4.51.3 `tojson` semantics:
wire key order, explicit null/omitted fields, Unicode and JSON separator spacing
are retained. Assistant historical argument strings are validated and inserted
verbatim. Consecutive tool results share one user block; call IDs and optional
names must match pending calls, and missing/orphan/duplicate results are refused.

Qwen3-0.6B generates calls in this format:

```text
<tool_call>
{"name": "function_name", "arguments": {"argument_name": "value"}}
</tool_call>
```

JSON argument types are validated without XML coercion. Duplicate keys, unknown
functions, extra envelope fields, invalid schemas and trailing prose are errors.
Literal closing tags inside JSON strings remain data. A leading complete thinking
block remains assistant content; examples inside it do not become calls. An
unfinished leading thinking block cannot produce a tool call. Tool-enabled SSE
buffers and validates the whole response before emitting OpenAI tool deltas.
The existing no-tools SSE markup guard remains conservative, including inside
reasoning. Template selection also fixes the response parser for the request.

The Qwen3-Coder real-checkpoint
gate pins the selected GGUF's 6,722-byte
`tokenizer.chat_template` (FNV-1a64 `a72a487d0e223214`) in addition to the checkpoint's
documented SHA-256. A different Qwen-family template is not assumed wire-compatible.

The Qwen3-Coder native call grammar remains:

```text
<tool_call>
<function=function_name>
<parameter=argument_name>
argument value
</parameter>
</function>
</tool_call>
```

Definitions are rendered inside Qwen's `<tools>` block, and subsequent tool results are
rendered as `<tool_response>` blocks. Native output is parsed, checked against the
requested function name and schema, and returned as OpenAI `tool_calls`; malformed or
unknown calls never fall through as ordinary prose.

```sh
curl -s http://127.0.0.1:6768/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen3-coder-30b",
    "messages": [{"role": "user", "content": "What is the weather in Detroit?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a city.",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "City name"}
          },
          "required": ["city"],
          "additionalProperties": false
        }
      }
    }],
    "max_tokens": 128,
    "temperature": 0
  }'
```

The accepted schema keywords are `type`, `description`, `enum`, `properties`, `items`,
`additionalProperties`, `required`, finite numeric `minimum`/`maximum`, and
nonnegative integer `maxItems`. Numeric bounds are inclusive and apply only to
numeric arguments; `maxItems` limits array length, including arrays without an
`items` schema. Integer-valued JSON numbers such as `2.0` are valid count limits,
following [JSON Schema validation semantics](https://json-schema.org/draft/2020-12/json-schema-validation).
The constraints are preserved in both checkpoint renderers and enforced recursively
before generated tool calls reach chat or streaming clients. Tool definitions
and Qwen3 JSON arguments reject duplicate keys and integer literals outside the
exact i64/u64 range rather than silently rounding them. The top-level parameter schema must have
`type: object`. After executing a returned call, append the assistant message containing
its `tool_calls` and then a `role: tool` message with the matching `tool_call_id` before
asking for the final response.

### Recorded product-client gate

On 2026-08-25, pushed beholdd code HEAD `c75a9a8` served the documented resident
`qwen3-coder-30b` checkpoint from local NVMe to disposable application client `a737bfd`.
The tool loop used non-streaming `provider.complete()`; application client exposed the agent's
own event stream over SSE, while beholdd's native SSE route was validated separately.
the agent made exactly one `write_file` call: `path` was `smoke.txt` and `content` was
`"beholdd v2 timeout fixed\n"`. The client wrote 25 bytes, replayed the result, received
`Done.`, and completed one build in 1,622,970 ms. A rerun with port 6768 down reused
that result in 2 ms without a model call or any artifact or ledger change. The full
artifact identity, containment record, and M3.1/M3.2 evidence are in
[the roadmap](../ROADMAP.md#m3--serving).

This narrow two-completion smoke barely fit a 30-minute guard. The inference/build result
completed, but the guarded service then reached `RuntimeMax` and exited with timeout
status 143; do not treat that supervisor lifecycle as a clean pass. At this CPU speed, a
worst-case default 24-round workflow is not viable. The gate covers the loopback-local
server and the schema subset documented above, not a broader deployment surface.

## In-process embedding

`beholdd-embed` wraps the same `NativeRuntime`; it does not start a daemon, bind a port,
or use HTTP. Embedded requests therefore share model loading, storage refusal, fit
planning, serialization, token generation, and observed residency with the server.

```rust
use std::path::Path;

use anyhow::Result;
use beholdd_embed::{Embed, GenerateRequest};

fn run() -> Result<()> {
    let runtime = Embed::from_config_path(Path::new("config.example.yaml"))?;
    let request = GenerateRequest {
        model: "qwen3-coder-30b".into(),
        prompt: concat!(
            "<|im_start|>user\n",
            "Reply with one short sentence.<|im_end|>\n",
            "<|im_start|>assistant\n"
        )
        .into(),
        max_new_tokens: 64,
    };

    let generation = runtime.generate_with(request, |text| {
        print!("{text}");
        Ok(())
    })?;
    eprintln!("\n{} completion tokens", generation.completion_tokens);
    Ok(())
}
```

The embedded boundary accepts an already-rendered prompt. Chat and tool protocol
translation remains owned by `beholdd-server`; embedders that need that protocol should
render it explicitly rather than sending HTTP back into the same process.

Embedded callers may inspect `execution_limits(model)` and use
`plan_request(model, rendered_prompt, max_new_tokens: Option<usize>, default_max_new_tokens)`
to obtain a `GenerateRequest` under the same omission/refusal policy. Existing
explicit `GenerateRequest` construction remains supported and generation rechecks
its limits. `ExecutionLimits` and typed `RequestError` are reexported by
`beholdd-embed`; the runtime also provides `plan_request_controlled` for cancellation
checks around metadata and tokenization work.
