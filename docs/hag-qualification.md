# Repeatable HAG answer qualification

## Public source scope

The clean public history does not contain the earlier Antimatter client commits
referenced by the records below. Profiles that freeze that client, including the
`antimatter-*` profiles and optional Antimatter transport for
`backend-parity-c128`, are historical evidence schemas, not currently runnable
public qualification workflows. Do not replace their commit pins with a current
HEAD and reuse the recorded results. A new public client profile needs its own
source identity, tests and fresh bounded qualification.

Standalone backend profiles such as `list-c128`, `semantic-c128` and
`backend-parity-c128` without the optional Antimatter client remain independent
of that history. They still require their documented model, device and resource
prerequisites; source publication does not rerun or expand hardware qualification.

HAG now accepts an explicit `context_capacity` from 1 through 16384, defaulting to
2048. The capacity expansion passed bounded hardware-free validation and has no live
C16384 qualification. Profiles below retain their frozen capacities, sources and
resource contracts; changing YAML does not extend those results. See the
[16k development budgets](development.md#explicit-experimental-hag-serving).

## C2048 live editor Stop and fresh-chat reuse

`antimatter-ui-stop-reuse-c2048` is a separate opt-in profile pinned to Antimatter
`f8f43c91c00e155121ce99493f123fc3711af547`. The October 13 same-host run on
`exec-alternate-host` with the `fc11e74` gate **passed full combined closeout**: actual
Stop after visible content, real transport cancellation, healthy native reset,
unchanged loaded server identity and actual New chat with CPU text/usage/EOS
parity. Native cancellation evaluated 481 positions; reuse evaluated 473, for
954 within the 1021 ceiling. The broker recorded one zero-byte pooled EOF and
exactly two admissions, with no rejections. Native/UI peak Tctl was 78.75°C,
with zero recorded OOM/swap, crash dumps or matching driver faults. All workloads
retired. The earlier failed attempt remains preserved separately; its bounded
connection fix passed all 424 qualification tests. See the
[qualified run and verified archive](hag-editor-context.md#same-host-live-editor-stop-and-reuse-qualified--2026-10-13)
and [failed attempt and repair record](hag-editor-context.md#live-stop-observed-empty-pooled-connection-repaired--2026-10-13).
Prepare that client's smoke with
`--live-stop-reuse --live-base-url=http://127.0.0.1:PORT/v1 --prepare-live`, then
freeze its fresh bundle using `prepare-ui-client`. Use this profile with the
same explicit inspector, identity, host mode and containment procedure below.

The first request asks for numbers 1 through 100 with C2048/M64 (exact prompt
476, at most 539 evaluated positions). The broker forwards each complete native
SSE event immediately. The actual editor clicks Stop only after visible numeric
content; a separate socket observer propagates the resulting EOF/reset using
`SHUT_RDWR`, including while the upstream reader is blocked. The gate requires
an actual native `cancelled` audit, healthy reset, loaded residency and unchanged
server PID/start time before admitting reuse. Natural completion before Stop
fails admission. No producer delay or synthetic abort qualifies.

The Stop broker tolerates at most two connections that close by EOF/reset before
sending any HTTP bytes, including fetch-pool replacement sockets after an abort.
It records each closure and validates its ordering relative to cancellation/reset
and the next admission. Partial HTTP requests, idle header timeouts and an
exhausted empty-connection allowance still fail. Exactly two complete requests
must be admitted and forwarded; all existing request and stage deadlines apply.

The actual New chat button must create an empty, distinct conversation. Its
ready request is C2048/M16 (prompt 467, ceiling 482), with no cancelled history.
The second complete SSE reply is released only after exact CPU text/usage/EOS
parity and healthy native reset. Both requests omit temperature and tools. CPU
measures both prompts but executes only this reuse request; its evaluation
ceiling is 482, and the two-request native ceiling is 1021. Existing stage,
request, UI, memory and thermal bounds remain unchanged. There are no retries.

Only combined closeout can set `live_cancellation_qualified` and
`same_server_reuse_qualified`. The client report retains `backendQualified:false`.
The cancelled answer has no full-answer or CPU-output parity claim. This profile
does not qualify token latency, throughput, generated history, full editor
main-process services, tools or operating-system integration OS transport. The previous buffered
history profile and its pinned client remain unchanged.

## C2048 live editor history profile

`antimatter-ui-history-c2048` adds a separate, opt-in path for the prepared
Antimatter UI at commit `8a0f8dce59fe7e5d71777fd4f709f5d70f67c15e`.
The October 13 same-host run on `exec-alternate-host` passed self-check, both CPU
references, both native/UI turns and full combined closeout. Both native replies
matched CPU text, EOS and usage; the actual first reply became turn-two history
and both replies were persisted by the editor. Native/UI peak Tctl was 65.625°C,
with zero recorded memory-pressure/OOM events, swap, crash dumps or matching
driver faults. All services, cgroups and recorded processes retired. The
separate-host SSH path remains unqualified. See
[the qualified run and verified archive](hag-editor-context.md#same-host-live-editor-history-qualified--2026-10-13).
The existing generated-history provider profile retains its original contract.

The editor sends two `stream: true` requests with no temperature field. The CPU
stage uses that same wire contract directly. During the native stage, a private
loopback broker admits exactly two requests and forwards their exact serialized
bytes. Before releasing each complete SSE response, it verifies CPU text/usage/
EOS parity, successful native reset and loaded residency. Turn two must include
the actual raw first reply. Failed requests, malformed HTTP and incomplete
connections close admission. Do not probe this listener with curl, a health
request or a TCP readiness connection.

The response is buffered until those checks pass; this profile does not qualify
live token latency, performance, cancellation, tools or full editor services.
Beholdd must include `c6b6161` or later: the server now supplies authoritative
usage in its terminal SSE event. The older retained binary omits that usage and
cannot pass this gate. Rebuild under the development guard and record the fresh
binary/source association before any live stage.

Prepare the editor using its documented `--prepare-live` command. On the desktop,
run the following through the bounded development wrapper, using explicit paths:

```sh
python3 -m tools.hag_qualify prepare-ui-client \
  --prepared /path/to/fresh/prepared-ui \
  --antimatter-repo /path/to/exec-antimatter \
  --electron-dist /path/to/exec-antimatter/node_modules/electron/dist \
  --metadata /path/to/new/ui-metadata
```

This verifies 13 prepared artifacts, their recorded esbuild source closure,
committed source hashes, installed dependency bytes and the Electron distribution.
It does not rebuild the bundles or claim to verify unpacked npm dependencies from
lockfile integrity. Preparation executes neither Electron nor a model. The frozen
identity includes the desktop machine/boot identity; a reboot requires a new
identity. A previously used prepared profile is refused.

Transfer `ui-metadata/identity.json` to the inference host. Use ordinary `prepare`
with `--profile antimatter-ui-history-c2048`, `--ui-client-identity` and the required
`--prompt-inspector`, plus the explicit repository, fresh binary, checkpoint,
device and evidence paths. Legacy provider build options and runtime phase
diagnostics are not accepted for this profile. Run `self-check`, then
`cpu-reference`, and verify CPU retirement before arming the 1500-second watcher.
The native `answers` stage publishes `answers-ui-ready.json` after opening its
finite listener. The default `--ui-client-host separate` requires different boot
identities for the editor and backend. When using one machine, prepare with
`--ui-client-host same`; both identities must match, and the broker binds the
port in the frozen editor URL. This choice is frozen before execution.

Copy the prepared gate files, `expected-manifest.json` and `client/identity.json`
to a local gate mirror, preserving the manifest hash observed on the remote host.
No model or executable needs to be copied into that mirror. Copy the native ready
receipt to the desktop, then launch the mirror's runner:

```sh
python3 /path/to/local-gate/ui-run-local.py \
  --manifest-sha256 REMOTE_EXPECTED_MANIFEST_SHA256 \
  --metadata /path/to/ui-metadata --ready /path/to/answers-ui-ready.json \
  --output /path/to/new/local-ui-evidence --target tester@validation-host_ADDRESS
```

For same-host execution, run the frozen evidence directory's `ui-run-local.py`
directly with its manifest hash and omit `--target`. No gate mirror or SSH forward
is needed. The runner verifies that the recorded broker PID owns the frozen
loopback listener without opening a connection. A target is forbidden in this mode.

For separate hosts, the runner creates its own dedicated SSH forward and observes
its listening socket without opening a broker connection. Electron and SSH share a local
CPU25%, Tasks128, MemoryHigh1536M/MemoryMax2G, swap0, NoNewPrivileges service.
It enforces the 1.5GiB early stop, startup8GiB/running4GiB host headroom, ten
seconds at most 80°C before startup and a running 90°C cutoff. Client/service/outer
deadlines are 355/375/390 seconds; each editor request remains 240 seconds.
Backend CPU75%/Tasks64 containment, 60°C startup, prompt467/at-most491, M128,
594/618 per-turn ceilings and 1212 total evaluations remain unchanged.
The same-host UI retains its own service and these UI limits. Its two services
have a combined 4GiB hard memory cap and 3GiB early-stop threshold, plus separate
HAG device accounting; each independently checks global host headroom. CPU
reference must retire before starting native inference and the UI.

After local completion, import `ui-report.json`, `ui-process.json`,
`ui-retirement.json`, `ui-supervision.json` and `ui-supervision-samples.jsonl`
into the backend evidence directory. Stop crash capture and complete the usual
kernel, watcher and backend retirement checks, then run `closeout`. It joins the
actual SSE bytes, UI history, frozen identity, broker run UUID and both services'
process evidence, and reports the frozen host mode. UI PIDs are verified on their
own host; separate-host PIDs are never looked up in the backend's `/proc`.
A passing backend stage alone leaves `live_ui_qualified: false`;
missing or inconsistent UI/resource/retirement evidence prevents full qualification.

## C2048 generated two-turn history profile

`antimatter-generated-history-c2048` tests a generated conversation through the
pinned Antimatter provider. It has three stages: `self-check`, `cpu-reference`,
and `answers`, followed by explicit watcher stop and `closeout`. Each inference
stage makes exactly two ordinary chat requests; there is no timing stage.

Turn one uses the synthetic project instructions and asks for `ready`. Turn two
appends that backend's actual complete raw first reply as the assistant message,
then asks for the same word again. CPU establishes both references first. HAG
must match the first CPU reply, EOS and API usage before constructing its second
request from its own reply. Both turns must match CPU exactly. The retained
request bodies, raw responses and parent hashes allow independent reconstruction
of this relationship; a seeded or trimmed assistant message does not pass.

Preparation requires the same four committed-client options as the project
profile plus `--prompt-inspector /absolute/path/to/prompt_inspect`. The explicit
metadata-only inspector is copied and hash-bound with the other dependencies;
its build/source association must be established separately, as for the server.
Before each CPU request, it measures the exact rendered prompt using checkpoint
tokenizer metadata. Turn one must be P467; generated turn two must be at most
P491. Both retain M128, with per-turn ceilings of 594 and 618 evaluations and a
1,212-position ceiling per stage. A response outside this scope stops the gate;
it does not increase the registered limits.

All C2048 memory, CPU, headroom, startup-temperature and 90°C running controls
remain unchanged. CPU child/service/outer deadlines are 720/735/750 seconds;
native deadlines are 360/375/390 seconds. Each native client has a 240-second
request deadline and a 245-second process deadline. Arm the fresh 1,500-second
crash watcher only after CPU retirement. Each native client also checks a
model-free over-capacity refusal; no additional model request is made.

A passing run of this profile qualifies only generated history propagation and
backend parity for the declared no-tools exchange. It does not qualify editor UI persistence,
retained KV across requests, semantic quality, SSE, cancellation or throughput.
The previous seeded-history and streaming evidence below retains its own scope.

The implementation passed 301 hardware-free Python qualification tests, 19 CLI
tests and 80 client tests, including replay with the pinned provider bundle.
These tests do not substitute for a fresh live qualification.

The first validation-host attempt on October 11 passed self-check and completed CPU
turn one with EOS and usage 467/6/473. Its actual raw reply produced the bounded
P491 second prompt, but CPU turn two reached the unchanged 90°C thermal cutoff
before completing. Native inference and crash capture did not start; generated
two-turn parity remains unqualified. An overlapping Antimatter native build was
observed. See the [preserved attempt](hag-editor-context.md#generated-two-turn-history-cpu-thermal-stop--2026-10-11).
Next arrange a quiet validation-host window for the complete CPU/native sequence and
use a fresh evidence directory with the same limits.

## C2048 checkpoint-rendered history profile

Use `antimatter-history-template-c2048` with the new checkpoint-aware server
renderer. It preserves the four wire messages and pinned Antimatter client
described below, but the exact Qwen3 template renders the historical assistant
reply as `ready.`. Metadata-only measurement gives **491 prompt tokens and
2,823 rendered bytes**. M128 permits at most 618 evaluated positions per
request, with CPU/native/timing counts 1/1/2 and stage ceilings 618/618/1,236.
The exact server fit/refusal boundary is 1558/1559; the unchanged provider
estimate E862 retains its conservative 1187/1188 boundary.

Prepare with the same explicit inputs and four committed-client options as the
project profile, using a new evidence directory. All C2048 resource, thermal,
timeout and watcher controls remain unchanged. Require the CPU stage to pass
and retire before arming native crash capture; proceed to timing only after
native parity passes. Embedded `.jinja` files are part of the frozen compiled
source inventory. A new renderer requires a new release/source association.

The fresh standard validation-host run on October 11 **passed all four stages and
final closeout**, with phase diagnostics disabled. CPU, native chat and warm
SSE returned identical complete raw responses ending in `ready.`, EOS and usage
491/6/497. The first lazy-load response took 128.875 seconds; warm SSE reached
first content at 126.657 seconds and DONE at 128.624 seconds. These are single
client-latency observations at 75% CPU quota; performance remains unqualified.
Peak Tctl was 82.875°C, with zero task swap/OOM events, crash dumps or matching
driver faults. The independent audit passed 1,630 checks. See the
[standard run and verified archive](hag-editor-context.md#standard-validation-host-history-parity-and-response-timing--2026-10-11).
The generated two-turn profile above now implements the next exchange, while
its first live attempt remains unqualified.
The earlier [quiet diagnostic](hag-editor-context.md#quiet-validation-host-history-parity--2026-10-11),
[October 11 thermal stop](hag-editor-context.md#validation-host-prefill-thermal-diagnostic--2026-10-11)
and [October 8 attempt](hag-editor-context.md#checkpoint-template-history-correction-and-native-thermal-stop--2026-10-08)
remain preserved with their original outcomes, scope and archives.

### Thermal diagnosis

For a fresh P491 diagnostic, add `--runtime-phase-diagnostics` to `prepare`.
This option is accepted only for `antimatter-history-template-c2048`. It freezes
the exact `HAG_SERVING_PHASES=1` setting for the HAG server in `answers` only.
The child and client launch paths remove inherited phase settings; CPU,
self-check and timing do not enable them. The resolved policy is recorded
separately in stage results. Raw phase lines remain diagnostic data and do not
substitute for CPU parity, runtime audits, thermal supervision or closeout.
See [phase semantics and bounds](serving.md#bounded-phase-diagnostics).

`tools/hag_host_telemetry.py` provides a finite passive Linux sampler for the
same diagnostic. For example, under a separate bounded supervisor:

```sh
python3 -B tools/hag_host_telemetry.py --duration-seconds 60 \
  --interval-seconds 1 --top-processes 10 --output /absolute/new-baseline.jsonl
```

It records bracketing UTC/monotonic timestamps, Tctl, exposed AMD GPU activity,
clock/power/fan attributes, aggregate CPU activity, memory/swap, CPU/memory
pressure and optional numeric process activity. Sensor readings are sequential,
not simultaneous. Hardware power attributes retain their kernel meaning;
correlation with a runtime phase does not establish cause. Missing or malformed
readings are explicit unavailable values. Process names, arguments and payloads
are omitted. Output is exclusive, mode 0600 and at most 32 MiB; duration is
at most 900 seconds and the interval is at least 0.1 seconds. The sampler
neither launches a model nor changes power settings and is not a safety guard.

The diagnostic procedure uses a 60-second passive baseline, then concurrent
sampling during one native answer attempt. Keep the collector in a separate
sibling cgroup with a small finite memory/CPU/time allowance; do not add it to
the workload's private leaf. Require healthy telemetry before dispatch and
retire both services independently. A new instrumented binary requires a new
clean-source build association, a fresh evidence directory, self-check and CPU
reference, complete CPU retirement and a fresh crash watcher. Preserve all
P491 request counts, deadlines, memory limits and the 90°C cutoff. A diagnostic
without the timing stage does not complete full-profile qualification.

## C2048 seeded project history profile

This historical P495 profile is retained for its original frozen renderer and
failed evidence. Current checkpoint-aware rendering requires the distinct P491
profile above; do not reclassify or overwrite the earlier attempt.

`antimatter-history-c2048` extends the exact project-instructions request with a
seeded assistant reply and a second user turn. Prepare it with the same four
committed Antimatter client options as the project profile below, then run
`self-check`, `cpu-reference`, a fresh crash watcher, `answers`, `timing`, explicit
watcher stop and `closeout`.

The four messages preserve the existing synthetic 2-KiB project instructions
and first user turn. The assistant seed is the prior qualified raw reply,
`<think>\n\n</think>\n\nready.`, followed by the user message
`Reply with that same one word again. /no_think`. Pinned Antimatter `78147b7`
forwards this content in order without removing inline thinking tags. Exact
checkpoint metadata tokenization measures 495 prompt tokens and 2,842 rendered
bytes; the 128-token output allowance caps each request at 622 evaluated
positions. Counts remain one CPU reference, one native answer and two native
reuse requests, with stage ceilings of 622, 622 and 1,244 positions.

The server output boundary is 1554/1555. The provider's conservative estimate is
862 prompt tokens, giving its distinct 1187/1188 local output boundary. Tests pin
the complete request and reject missing, reordered or altered history before
transmission. Live calls use only 128 output tokens; overflow probes must refuse
without generation. Resource controls and 240-second client, 600-second CPU HTTP,
stage and 1500-second watcher deadlines are identical to the project profile.

This profile tests seeded-history transport and second-turn CPU/native text,
EOS and usage parity. Semantic quality remains diagnostic. It does not generate
the first turn in the same run or qualify UI history persistence, context
compaction, retained KV across requests, semantic recall, full editor/tools or
interactive latency. The [first validation-host history attempt](hag-editor-context.md#held-c2048-seeded-project-history-on-validation-host--2026-10-08)
failed its CPU prerequisite on an unclosed thinking block despite complete HTTP
and EOS. Native stages and crash capture were not started; this profile is not
live-qualified. Preserve that attempt and require a newly measured, frozen run
after any prompt-rendering correction.

## C2048 TCP lifecycle profile

`lifecycle-c2048` is a separate, bounded real-TCP check for cancellation, healthy
reuse and same-process unload/reload. Prepare it with the explicit repository,
binary, checkpoint, device and new evidence directory arguments shown below;
it needs no Antimatter client options. Run only `self-check`, `cpu-reference`
and `answers`, then explicitly stop the fresh crash watcher and run `closeout`.
The profile refuses a `timing` stage before starting any process.

The fixed `Hello` prompt uses nine checkpoint tokens. CPU makes one greedy
one-token reference request. Native makes four successful one-token requests
(JSON, complete SSE, reuse after cancellation and reload), plus one stream with
a sixteen-token limit that is disconnected after its first content event.
The gate requires HTTP409 `runtime_busy` from an unload attempted during that
stream, an actual `SHUT_RDWR`, and a cancelled audit with successful reset and
an unpoisoned runtime. A request that finishes naturally before cancellation
fails the attempt; there is no retry. Native work is capped at 60 evaluated
positions: 36 across the four successes and at most 24 for cancellation.

Raw HTTP/SSE, server audit events and process/cgroup snapshots are retained and
replayed before qualification. Reuse and reload must match CPU text, usage and
the `length` finish reason. The server must keep its PID, process start time and
private cgroup through reload, unload idempotently and exit zero on idle SIGTERM.
This checks lifecycle behavior, not complete-answer quality, Antimatter-provider
cancellation, early cancellation compute savings, active-inference shutdown,
concurrency, throughput or every position in C2048.

Use the unchanged C2048 memory, private-cgroup, headroom and thermal controls.
Child/service/outer deadlines are 180/195/210 seconds for self-check,
720/735/750 for CPU, and 600/615/630 for native lifecycle. CPU HTTP requests have
a 600-second deadline; native HTTP requests have 170 seconds. Arm the fresh
1500-second root watcher only after all CPU processes, its unit and cgroup have
retired. The native stage requires enough remaining watcher time for its full
outer deadline plus twenty seconds. The fresh
[October 8 validation-host run and stopped-watcher closeout passed](hag-editor-context.md#qualified-c2048-tcp-lifecycle-on-validation-host--2026-10-08),
using 46 native positions with zero memory-pressure/swap events or crash dumps.
Another source, binary or host still requires its own fresh frozen run.

## Grounded Antimatter client at C512

The fresh Beholdd `02ad1a71977e0d68d8e2f3f2f7ace5c1686f3a14` run passed this
profile and stopped-watcher closeout on 2026-10-05. Chat, warm streaming/reuse
and capacity refusals passed with exact CPU parity. The full result and archive
identities are recorded in [hag-editor-context.md](hag-editor-context.md).

Use `--profile antimatter-grounded-c512` with the C512 on-demand HAG serving
build. All four client options are required; the provider is frozen from
committed Antimatter `78147b78d8271b993dc3734f469f763df1f0f548`, including its
capacity-aware token estimator. The working editor checkout is never modified.

```sh
python3 -B -m tools.hag_qualify prepare \
  --repo "$BEHOLDD_REPO" --binary "$BEHOLDD_BINARY" --model "$HAG_MODEL" \
  --device /dev/dri/renderD128 --evidence "$HAG_RUN" \
  --profile antimatter-grounded-c512 \
  --antimatter-repo "$ANTIMATTER_REPO" \
  --antimatter-commit 78147b78d8271b993dc3734f469f763df1f0f548 \
  --node /usr/bin/node --esbuild "$ESBUILD_BINARY"
```

This freezes one exact no-tools system/user prompt: 93 checkpoint tokens and an
explicit 128-token output allowance, at most 220 evaluated positions. The CPU
reference runs once. Native answers use a genuine provider chat call; the reuse
stage makes another provider chat call followed by one provider stream call.
Raw output, EOS, usage, callbacks and lifecycle evidence must agree with CPU.
The profile also requires generation-free cold nonstream and warm streaming
HTTP400 context refusals, and the provider's own capacity refusal before any
fetch or managed preparation hooks. Client calls have fixed 60-second deadlines;
child client processes have 65 seconds. Existing cgroup, thermal, host-headroom,
crash-watcher and outer supervision remain unchanged.

Run the existing stage sequence below: self-check, CPU reference, fresh root
watcher, answers, timing/reuse, explicit watcher stop and closeout. The stage
ceilings are 220 CPU, 220 native answers and 440 native reuse positions. Only one
warm stream is measured, so no throughput qualification is claimed. Semantic
quality is diagnostic; full editor/tools, arbitrary conversations and live
cancellation are outside this profile. Native C512 proof and the latest client
status are recorded in [hag-editor-context.md](hag-editor-context.md).

## C2048 project instructions profile

Explicit experimental HAG serving defaults to C2048. The
`antimatter-project-c2048` profile
[qualified on validation-host on October 8, 2026](hag-editor-context.md#qualified-c2048-project-instructions-on-validation-host--2026-10-08),
with Linux `6.17.0-1024-oem`. The exact synthetic no-tools project-instructions
fixture passed CPU/native text, EOS and usage parity through two provider chat
calls and one warm stream. Each native request evaluated 473 positions (467 prompt
tokens and six completion tokens), for 1,419 positions total. This is the scope of
the serving proof; it does not qualify every request up to C2048, full editor/tools,
general semantic quality, concurrency, throughput or live cancellation. CPU remains
the ordinary default. Existing C128/C512 profiles retain their exact historical
contracts and require matching frozen binaries. The completed C2048 fixed-token
model proof remains separate.

Select the new profile in the Antimatter preparation command above. It freezes
the same committed provider and the measured no-tools grounding plus one synthetic
2048-byte `AGENTS.md` body. The exact prompt is 467 tokens, with a 128-token output
allowance and a 594-position maximum per request. CPU reference and native answers
each issue one generation request; timing issues a chat and a warm stream, bounded
by 1188 evaluated positions. Semantic quality remains diagnostic.

The server admits an exact 467-token prompt with an explicit output limit of 1582
and refuses 1583. The provider estimates 831 prompt tokens, so its conservative
local boundary is 1218/1219. Hardware-free tests cover those distinct boundaries;
the live gate requests only 128 output tokens and tests generation-free cold and
warm overflow refusals.

The profile explicitly uses a 1536-MiB logical CPU budget and 1152-MiB HAG budget.
HAG allocation is 1,135,140,864 bytes; the current private-layout runtime forecast
is 1,914,935,488 bytes, below the standalone metadata upper bound of 1,915,984,064
recorded in the contract. The physical process cap remains 2 GiB, early stop
1536 MiB, swap zero, CPU 75% and 64 tasks. Host headroom, 60°C/10-second startup
cooldown and 90°C/0.1-second thermal cutoff remain unchanged.

The private stage cgroup is verified before setting `BEHOLDD_HAG_PRIVATE_CGROUP`
to its exact unified membership path. That attestation permits credit only for
sampled leaf charges, capped by the complete checkpoint allowance. It cannot
credit unrelated ancestor charges. Total forecast and additional allocation
checks both apply; ordinary serving without the attestation keeps conservative
admission. This allows warm reload without counting cached checkpoint pages twice.

Deadlines are registered before any model runs. CPU HTTP requests allow 600 seconds;
the native client allows 240 seconds, with 245 seconds for its process. Stage
child/service/outer deadlines are:

| Stage | Child | Service | Outer |
| --- | ---: | ---: | ---: |
| Self-check | 180 s | 195 s | 210 s |
| CPU reference | 720 s | 735 s | 750 s |
| Native answers | 360 s | 375 s | 390 s |
| Chat/stream reuse | 600 s | 615 s | 630 s |

These finite bounds account for the measured CPU C2048 evaluation time of
1711.499 seconds and prior grounded prefill, with preparation/client overhead.
They are bounds, not promised latency. CPU must finish and retire before starting
a fresh 1500-second read-only crash watcher. Each native stage requires enough
remaining watcher time for its complete outer deadline. Stop capture explicitly
after both stages and audit final kernel, cgroup, process and crash evidence.
A refusal or timeout is preserved, with no automatic retry or deadline extension.

## Legacy C128 profiles


The existing C128 profiles retain their frozen fixture contracts and require a
matching C128 binary. The frozen C512 serving release uses the additive grounded
profile above; current C2048 source uses the project-instructions profile.

`python3 -m tools.hag_qualify` defaults to the bounded `list-c128` profile used by
the October 4 CPU/HAG EOS and timing proof. The opt-in `semantic-c128` profile adds
arithmetic and short conversation recall with strict answer criteria. The separate
`backend-parity-c128` profile compares those same requests across backends while
reporting correctness and format separately. It prepares a new evidence directory,
then executes frozen copies of the runner. It does not build the CLI, discover a
checkpoint, acquire root privileges, retry a failed stage, or change defaults.
Ordinary Python tests never open a checkpoint or GPU.

These profiles are deliberately narrow: the dense Qwen3-0.6B-Q8_0 checkpoint with
SHA-256 `e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c`,
C128, and declared small fixtures. The default profile has one prompt asking
for the integers 1 through 8 separated by commas.
The ordered list may have one terminal period. Full CPU/native output remains
exact, including thinking markup and punctuation. EOS, usage, healthy resets,
unload and clean CLI exit are required. General answer quality remains outside
all profiles.

## Choose a fixed profile

The default `list-c128` retains the original fixture bytes. Select
`--profile semantic-c128` during preparation to freeze all three fixtures below,
in order. The profile cannot change after preparation.

| Profile | Answer requests per backend | Does the declared answer criterion gate success? |
| --- | --- | --- |
| `list-c128` (default) | Original list only | Yes |
| `semantic-c128` | List, arithmetic, recall | Yes |
| `backend-parity-c128` | The same list, arithmetic, recall | No; exact backend parity gates success and quality is diagnostic |

| Fixture | Conversation | Accepted final answer |
| --- | --- | --- |
| `list_1_to_8` | User: `List the integers from 1 to 8, separated by commas. No explanation. /no_think` | Integers 1 through 8 in order, comma separated |
| `arithmetic_7_plus_5` | User: `What is 7 + 5? Reply with just the number. /no_think` | `12` |
| `recall_mango` | User: `The secret word is MANGO.`; assistant: `Understood.`; user: `Reply with just the secret word. /no_think` | `MANGO`, case-insensitive |

For the two strict profiles, boundary whitespace, one terminal period and one correctly
closed leading thinking block are allowed. Extra explanation, Markdown decoration,
wrong values and malformed thinking tags fail. CPU/HAG comparison still requires
the entire raw output to match exactly, including thinking, case and punctuation.
Acceptance rules are fixed before execution and are never broadened to accommodate
an observed response. A semantic failure is a failed qualification even if both
backends return the same answer.

The added checks cover one arithmetic expression and one brief conversation.
They do not qualify longer recall, general reasoning, arbitrary formatting or
production readiness. Their live result is separate from the earlier list proof.

### Separate backend parity and quality

Prepare a fresh run with `--profile backend-parity-c128` to measure backend
equivalence. The profile still requires EOS, bounded token usage, a nonempty final
answer, properly closed thinking markup, exact CPU/HAG raw text and usage, healthy
resets, unload/exit and all containment checks. CPU must complete all three requests
safely before native execution. A wrong but identical answer may pass backend
parity; it cannot establish answer quality.

Each answer in this profile records a non-gating `quality` object with separate
`correctness` (`pass`, `fail`, `unassessed`) and `format` (`pass`, `fail`) fields.
The supervisor recomputes these annotations from the retained answer. Finite,
anchored parsers recognize the entire final text after boundary whitespace and
one optional terminal period:

- Numeric CSV is compared with integers 1 through 8; leading zeros fail format.
- A canonical unsigned integer is compared with 12. The exact equation form
  `7 + 5 = INTEGER`, permitting whitespace around its operators, also receives
  a correctness assessment but fails the standalone-number format.
- One ASCII word is compared case-insensitively with MANGO.

Other forms receive `correctness: unassessed` and `format: fail`; no substring
matching or arbitrary Markdown removal is used. These diagnostics have no effect
on parity eligibility. Reports explicitly identify the `backend_parity` qualification
kind, and closeout records `semantic_quality_qualified: false` even
when all finite diagnostics pass. The original strict profile and its failed
arithmetic result remain unchanged; a fresh parity run never relabels that result.

## Prepare explicit inputs

Use a Linux system with a systemd user manager, readable kernel journal,
working k10temp Tctl sensor, and a device admitted by the experimental native HAG
backend. Prepare a release CLI with the HAG feature before the live stages:

```sh
python3 tools/run_dev_guarded.py -- cargo build --release --offline --locked -j1 \
  -p beholdd-cli --features hag-native
```

Choose explicit absolute paths. The evidence directory must not exist; keep it
outside the checkout's source tree. Paths containing spaces are supported.

```sh
HAG_BINARY=/absolute/path/to/exec-beholdd/target/release/beholdd
HAG_CHECKPOINT=/absolute/path/to/Qwen3-0.6B-Q8_0.gguf
HAG_RUN=/absolute/path/to/new-hag-run

python3 -m tools.hag_qualify prepare \
  --repo "$PWD" --binary "$HAG_BINARY" --model "$HAG_CHECKPOINT" \
  --device /dev/dri/renderD128 --evidence "$HAG_RUN" --profile list-c128
```

Preparation freezes the executable, runner, fixture, generated configurations,
thermal/crash helpers and observed Rust/assembly/Cargo source inputs. It records
Git state and executable/source hashes separately: the runner does not prove
that the supplied executable was built from that checkout. Use a known build
and retain its build record. No qualification is inherited from an earlier run.
The checkpoint stays at its explicit path and its bytes are checked repeatedly.
Changes to the frozen inputs or observed checkout invalidate the run.

### Exercise the Antimatter host streaming client

operating-system integration intends to use Beholdd as its inference engine, but does not yet have
a native inference client or socket transport. The optional host-client gate
exercises Antimatter's real `OpenAICompatibleProvider` as a step toward that
integration. It uses the committed provider at
`0679c442017083d8dfb9fd2b43c3de1a7eb27e99`, preserving any unrelated changes in
the Antimatter working tree.

Add all four explicit client inputs when preparing `backend-parity-c128`:

```sh
python3 -m tools.hag_qualify prepare \
  --repo "$PWD" --binary "$HAG_BINARY" --model "$HAG_CHECKPOINT" \
  --device /dev/dri/renderD128 --evidence "$HAG_RUN" \
  --profile backend-parity-c128 \
  --antimatter-repo /absolute/path/to/exec-antimatter \
  --antimatter-commit 0679c442017083d8dfb9fd2b43c3de1a7eb27e99 \
  --node /absolute/path/to/node \
  --esbuild /absolute/path/to/esbuild
```

The esbuild argument names the executable binary, such as the installed
`node_modules/@esbuild/linux-x64/bin/esbuild`, rather than a JavaScript launcher.
Preparation snapshots the committed transitive client sources, bundles them,
and freezes the Node executable, bundle, build metadata and source hashes. Live
stages import only that snapshot. No package installation is performed.

Use the same four stage commands and stopped-watcher closeout below. CPU and
native nonstream stages retain their three fixtures. Only the three timed list
streams use Antimatter; the nonstream list warmup and all request/evaluation
ceilings stay unchanged. Every client process sends exactly one isolated
loopback request, with explicit `maxTokens: 80`, temperature zero and no tools.
An attempted second request fails before transmission. A 30-second client
deadline and 35-second process deadline fit within the existing stage limits.

Raw SSE is retained and checked independently of the client's parsed result:
model identity, exact CPU text, EOS, `[DONE]`, callbacks, request counts, native
reset audits, process retirement and all existing resource checks must agree.
No client setting, installed service or bundled runtime is replaced. This gate
qualifies the pinned host provider's bounded list stream only; it does not
qualify the full editor, native operating-system integration transport, tools, cancellation or
ordinary application context planning. Antimatter's normal output-token default
is larger, and this pinned client does not consume Beholdd's C128 execution limit.

## Run CPU before native stages

```sh
python3 -m tools.hag_qualify run --evidence "$HAG_RUN" --stage self-check
python3 -m tools.hag_qualify run --evidence "$HAG_RUN" --stage cpu-reference
```

Continue only if each command succeeds. The self-check exercises containment
without inference. Answer profiles require every CPU answer and EOS before
native work is permitted; the lifecycle profile requires its one-token length
reference. Neither stage needs root or opens the GPU.

Each inference stage retains the proven limits: 2 GiB hard memory cap,
1536 MiB early stop, zero swap, 75% CPU, 64 tasks, no new privileges, 8 GiB host
headroom before launch and 4 GiB while running. Tctl is checked every 0.1 seconds
with a 90°C cutoff. Legacy child/service/outer deadlines are 180/195/210 seconds;
the C2048 project and history profiles use the per-stage deadlines above.
Do not replace this supervisor with `run_dev_guarded.py`; that helper has different
limits and is intended for development commands.

## Capture and native checks

Start the frozen crash watcher in a separate terminal. Root is needed only to
read the kernel's GPU crash dumps. It changes no driver settings. Authenticate
through the desktop prompt; never put a password in a command or evidence file.

```sh
pkexec /usr/bin/python3 "$HAG_RUN/dependencies/hag-crash-watch.py" \
  --seconds 600 --stop-file "$HAG_RUN/capture.stop" > "$HAG_RUN/capture.log" 2>&1
```

For C2048 profiles, use `--seconds 1500`; their contracts require that
exact finite lease. The default remains 600 seconds for historical profiles.

Wait for `ARMED` and a fresh `HEARTBEAT` in `capture.log`. The native stages check
that the watcher is a live root process with no recorded dump and a heartbeat
at most 15 seconds old. Keep the same watcher alive across both answer-profile
stages (`lifecycle-c2048` runs only `answers`):

```sh
python3 -m tools.hag_qualify run --evidence "$HAG_RUN" --stage answers
python3 -m tools.hag_qualify run --evidence "$HAG_RUN" --stage timing
```

For the legacy C128 profiles, the answer stage compares each native nonstream
answer with CPU. Requests run
serially: one for `list-c128`, three for either broader profile. Every request retains
the 49-prompt-token ceiling, 80-new-token limit and C128 capacity. Timing performs
one initial nonstream request and three warm SSE requests. Each request resets
KV; warm requests retain model weights. CPU and answer stage evaluation ceilings
are 128 for `list-c128` and 384 for either broader C128 profile; timing remains
512 for all three C128 profiles. The grounded C512 profile uses its separate
93-token prompt, 128-token reply allowance and 220/220/440 stage ceilings
specified above.
The C2048 project profile uses 467 prompt tokens and 594/594/1188 stage ceilings;
the seeded-history profile uses 495 tokens and 622/622/1244 ceilings.
Process, memory and running thermal limits remain unchanged; its finite deadlines
and start cooldown are specified above. Failed CPU or answer evidence
prevents the next stage. No automatic retries or overwrites are allowed.

Whether the checks pass or fail, stop the watcher and wait for its terminal
command to exit before further work:

```sh
touch "$HAG_RUN/capture.stop"
```

After every stage passes and capture reports a clean `STOPPED` event:

```sh
python3 -m tools.hag_qualify closeout --evidence "$HAG_RUN"
```

Only a successful closeout qualifies the complete run. It verifies ordering,
identities, resource telemetry, exact responses, stopped capture and independent
CPU/GPU kernel windows, retaining evidence hashes. Preserve failed runs and use
a new directory after addressing the cause; do not relabel or edit old results.
Snapshots retain the runner version they were prepared with. Old snapshots can
be inspected using their own frozen helpers; the current public dispatcher requires
the current complete helper inventory and rejects mixed runner versions.

## Read the result

`cpu-reference-result.json`, `answers-result.json`, `timing-result.json` and
`final-evidence-check.json` state whether their checks qualified. Raw JSON bodies
and timestamped SSE lines remain available alongside the reports.

Timings are descriptive client measurements under the CPU quota. Initial
nonstream latency includes lazy model preparation and excludes CLI startup; it
is not an OS-cold-cache measurement. First content includes thinking markup.
Warm completion ends at receipt of `[DONE]`. SSE has no usage field: its token
numerator comes from exactly matched nonstream API usage and the native audit
checks evaluated positions. Chunk counts are not token counts. These measures
do not establish decode/kernel throughput or a production performance target.

Run the hardware-free regression suite separately:

```sh
python3 tools/run_dev_guarded.py -- python3 -B -m unittest discover \
  -s tools/hag_qualify -p 'test_*.py'
python3 tools/run_dev_guarded.py -- python3 -B tools/test_hag_qualify_cli.py
python3 tools/run_dev_guarded.py -- python3 -B tools/test_hag_crash_watch.py
```

For the optional client, also replay its frozen provider against synthetic
transport responses before live inference:

```sh
python3 tools/run_dev_guarded.py -- env \
  HAG_ANTIMATTER_PROVIDER_BUNDLE="$HAG_RUN/client/provider.mjs" \
  node --test tools/test_hag_antimatter_client.mjs
```

Without that explicit bundle path the Node tests exercise only the adapter test
double. Neither replay mode contacts an inference server or opens a checkpoint.

## Migration validation — 2026-10-04

The repository runner passed 100 hardware-free checks (82 package, 6 CLI and
12 crash-watcher checks), then all four live stages and closeout through the
public command above. Fresh evidence used a directory containing spaces and
required no ancestor archive. The supplied executable and 130 compiled inputs
matched the preceding C128 proof; executable/source association remains an
explicit caller responsibility in the reusable tool.

CPU and all five native requests matched through the complete list and EOS:
29 prompt tokens, 27 emitted tokens and 56 evaluations per request. All resets,
unload, CLI exits and recorded process/cgroup retirement passed. Maximum Tctl was
65.125°C; no test swap, memory-pressure, OOM or matching driver-fault events were
recorded. CPU/native-answer/native-timing cgroup peaks were
272,867,328 / 145,559,552 / 146,092,032 bytes, separate from shared GPU memory.
The stopped root watcher captured zero dumps; complete CPU/GPU kernel windows
contained 169 / 986 clean lines. The closeout verified 12 frozen gate inputs.

The initial response took 11.881 seconds and three warm responses completed in
11.369–11.431 seconds under the unchanged 75% CPU cap. These retain the descriptive
timing scope above; the migration does not establish a performance improvement.
Evidence is in `work/hag-regression-20261004/live run/` in the local task workspace;
the archive target is `outputs/hag-regression-qualified-20261004.tar.gz`.

## Semantic profile implementation — 2026-10-04

The named-profile extension passed 113 hardware-free checks: 26 preparation and
profile checks, 31 worker checks, 48 supervisor/closeout checks and 8 public CLI
checks. These include synthetic three-fixture CPU/native runs, unchanged list-only
timing, cross-profile evidence rejection, exact raw parity despite semantic case
normalization, and refusal to close out with any response artifact missing.
The original list fixture bytes and all 130 compiled source inputs are unchanged.
The implementation was committed and pushed as `fd7daf6` before its first live
CPU evaluation; these hardware-free checks alone do not qualify the new arithmetic
or recall fixtures.

The first live `semantic-c128` run passed containment self-check, then failed the
CPU prerequisite on output form. The list completed correctly at EOS with 29 prompt
and 27 completion tokens. Arithmetic completed at EOS with 26 prompt and 12
completion tokens, but its final answer was `7 + 5 = 12` instead of the required
standalone `12`. The arithmetic was correct; the declared format was not. Recall
was not attempted. No native answer, timing or crash-watcher session was started,
and this run does not qualify the semantic profile.

The failure was unrelated to memory or heat: peak cgroup memory was 273,031,168
bytes, swap and memory-event counters stayed zero, and maximum Tctl was 54.125°C.
The thermal guard recorded no abort; its nonzero child status came from semantic
validation. The original frozen runner reported that status as a thermal-evidence
failure as well as a child failure, so the raw logs and guard summary are needed
to distinguish the cause. The failed inputs and evidence remain unchanged in
`work/hag-semantics-20261004/live run/` in the local task workspace. Acceptance
criteria were not loosened and the run was not retried.

A follow-up diagnostic fix distinguishes healthy thermal supervision from a
nonzero child exit. Such an exit still fails qualification, even if other records
contradict it. All 51 supervisor/closeout tests passed, including three new
failure-diagnostic regressions; replaying the retained CPU telemetry through the
fix reported `child process exited 1` and still refused success. No model was
rerun for this fix. Together with the unchanged preparation, worker and CLI
suites, the profile tooling now has 116 passing hardware-free checks.

Independent cleanup found both recorded services inactive, their cgroups gone,
and all four recorded child/thermal PIDs absent. The full 199-line CPU kernel
window was clean. The failed CPU evidence and the separate diagnostic replay
are retained in the text/source archive
`outputs/hag-semantic-cpu-failed-20261004.tar.gz`; model and executable payloads
are excluded. A later diagnostic commit changes the current supervisor and its
tests, while the frozen run continues to match the original `fd7daf6` inputs.

## Backend parity profile implementation — 2026-10-04

The separate profile passed 138 hardware-free checks: 30 preparation/profile,
42 worker/quality, 58 supervisor/closeout and 8 public CLI checks. Coverage includes
identical wrong answers with failed quality diagnostics, correct equations with
failed standalone format, forged annotations, exact raw output differences,
profile/strategy confusion, malformed thinking, missing EOS and unchanged strict
semantic refusal. Long numeric diagnostics avoid unbounded integer conversion.
The implementation was committed and pushed as `6358c30` before this profile's
first live run. Runtime sources and the serving defaults are unchanged.

The fresh `backend-parity-c128` run passed all four stages and final closeout
through the public CLI. All three CPU/native answers matched exactly, including
raw thinking markup, final text, EOS and token usage:

| Fixture | Final answer | Prompt / completion tokens | Correctness / format diagnostics |
| --- | --- | --- | --- |
| List | `1, 2, 3, 4, 5, 6, 7, 8.` | 29 / 27 | pass / pass |
| Arithmetic | `7 + 5 = 12` | 26 / 12 | pass / fail |
| Three-message recall | `MANGO` | 39 / 6 | pass / pass |

Each answer stage used 139 evaluations; the unchanged list timing stage used
four requests and 224 evaluations. All seven native request audits reported
successful resets, followed by clean unload and CLI exit. Closeout verified
14 frozen gate inputs, 130 compiled sources, the retained executable and
dependency identities, all four retired services/cgroups and eight absent
recorded child/thermal PIDs. The stopped root watcher captured zero dumps;
the complete CPU/GPU kernel windows contained 315 / 1,466 clean lines.

No test swap, memory-pressure, OOM or matching driver-fault events were recorded.
Maximum Tctl was 72.25°C, with no thermal abort. CPU/native-answer/native-timing
cgroup peaks were 273,633,280 / 146,735,104 / 146,112,512 bytes; these counters are
separate from shared GPU allocation. The lowest sampled host headroom across
model stages was 20,788,031,488 bytes. The fixed memory, thermal, token and
execution limits were unchanged.

The initial list response took 11.833 seconds; three warm responses completed in
11.200–11.384 seconds under the 75% CPU cap. These are descriptive client timings
with the scope above, not a production throughput or performance-improvement
claim. The result qualifies this C128 backend-parity profile only:
`semantic_quality_qualified` remains `false`, the arithmetic format diagnostic
still fails, and the earlier strict semantic failure remains intact. CPU remains
the configured serving default.

Evidence is in `work/hag-parity-20261004/live run/` in the local task workspace;
the text/source archive target is
`outputs/hag-backend-parity-qualified-20261004.tar.gz`. The next integration
milestone is exercising the existing Antimatter host client against Beholdd's
HAG backend with the same bounded context and explicit backend selection, as
a step toward operating-system integration.

## Antimatter host-client implementation — 2026-10-04

Source inspection corrected the next integration target: operating-system integration has no native
inference client or socket transport yet. Beholdd is its intended official
inference engine; the existing Antimatter host provider is the executable client
reference for this checkpoint.

The optional client runner passed 181 checks: 47 worker, 64 supervisor/closeout,
33 preparation/profile, 12 committed-client preparation, 10 public CLI and
15 Node replay checks. Five replay cases exercised the real committed provider
against synthetic transport, covering fragmented UTF-8, exact callback delivery,
complete EOF capture after `[DONE]`, malformed/truncated responses and byte
limits. Adapter tests also rejected changed destinations/requests, repeated
POST attempts and deadline violations. No model or GPU was used by these checks.

Actual model-free preparation froze five committed TypeScript source files and
bundled four runtime inputs using Node v22.22.1 and esbuild 0.28.2. The bundle
SHA-256 is `62ed0aa8b920c5f3af213c4712209f4cf4f4e0b234559c7818ae6b95267dbde6`.
The reviewed implementation was committed and pushed as `10b4da4` before live
execution. The prepared run retains 16 frozen gate inputs and the unchanged
130 compiled source inputs from the preceding executable qualification.

The first launch was held while other work kept host Tctl between 62.5°C and
100.125°C across 76 observations. No inference or crash watcher was started
during that hold. Its original records remain intact in
`work/hag-antimatter-20261004/thermal-hold.json` in the local task workspace.
After the user reported cooling, fresh observations established three consecutive
readings at or below 60°C, and the unchanged prepared inputs were verified again.

The resumed run passed all four stages and final closeout. CPU and native HAG
again produced the exact three raw answers and token counts in the preceding
backend-parity table, using 139 evaluations per answer stage. The timing stage
used one raw nonstream list warmup and three streams through Antimatter's real
`OpenAICompatibleProvider`, pinned at
`0679c442017083d8dfb9fd2b43c3de1a7eb27e99`. Each isolated client made exactly one
completion POST, delivered 27 content callbacks, matched the raw CPU/HAG reference
and returned the expected model and EOS result. Independent raw SSE reconstruction
verified `[DONE]`, complete EOF capture and callback/result agreement. All three
Node processes exited successfully without timeout; the timing stage used
224 evaluations across its four requests.

The first list request completed in 12.414 seconds; the three provider streams
reached `[DONE]` in 11.487–11.772 seconds, with first raw content in
5.793–5.945 seconds. These descriptive client timings include thinking markup
and retain the 75% CPU cap; they do not establish production performance.

All seven native requests reported successful resets, and both native stages
unloaded cleanly. No test swap, memory-pressure, OOM or matching driver-fault
events were recorded. Maximum Tctl was 71.75°C for CPU, 62.75°C for native answers
and 66.75°C for native timing,
with no thermal abort. Their cgroup peaks were 274,292,736 / 146,919,424 /
183,021,568 bytes; these counters remain separate from shared GPU allocation.
The minimum sampled host headroom was 18,740,580,352 bytes. The fixed memory,
thermal, token and execution limits were unchanged.

Closeout verified 75 closed evidence files, 16 frozen gate inputs, 130 compiled
sources and the executable, dependency and client identities. All four services
were inactive, their cgroups gone and all 11 recorded child/thermal/client PIDs
absent. The root watcher stopped with zero captured dumps and an empty dump
directory. The complete CPU/GPU kernel windows contained 253 / 1,000 lines and
no matching faults. Live evidence is retained at
`work/hag-antimatter-20261004/live run/`; the text/source archive target is
`outputs/hag-antimatter-client-qualified-20261004.tar.gz`.

This qualifies the bounded host-provider backend-parity path only.
`semantic_quality_qualified` remains `false`, the arithmetic standalone-number
format diagnostic still fails, and CPU remains the configured serving default.
Native operating-system integration transport, the full editor, tools and cancellation are outside
this checkpoint. Every client request explicitly used `maxTokens: 80`. The
provider falls back to 4096 when this field is omitted; ordinary editor requests
use Antimatter's existing model-profile budgets. Discovery and profile planning
do not yet consume Beholdd's `execution_limits.max_evaluated_tokens`. Teaching
them to respect the advertised C128 execution capacity is the next integration
step.
