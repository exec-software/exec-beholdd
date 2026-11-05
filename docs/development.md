# Development setup

Beholdd builds independently of an installed inference server. Rust/Cargo, a native
C linker, Python 3.11+ and Git are sufficient for CPU development and the HAG source checks.
The optional Vulkan prototype additionally needs `glslc`. HAG's checked-in native
shader bytes do not require Vulkan, HIP or ROCm to build.

Run commands from the repository root. No command below downloads a checkpoint.

The historical Antimatter client profiles require source revisions outside the
clean public history. They are retained as evidence schemas, not supported public
run instructions; see [public source scope](hag-qualification.md#public-source-scope).
Normal builds and standalone backend profiles do not depend on those revisions.

The reusable [C128 answer qualification runner](hag-qualification.md) freezes
explicit inputs and preserves the CPU-first, memory, thermal and crash-capture
requirements of the successful EOS/timing proof. It runs from the repository
without any prior local evidence directory. The default `list-c128` profile retains
the proven list fixture; opt-in `semantic-c128` adds fixed arithmetic and short
conversation recall checks under the same process limits.
Use `backend-parity-c128` to measure exact backend equivalence across those requests
with correctness and formatting recorded separately from parity qualification.

## Optional Linux resource supervisor

The portable CPU build and test commands are in the [repository README](../README.md).
The supervisor below is an additional, hardware-specific development aid. It is not
a prerequisite for installing Beholdd on a machine without an AMD temperature sensor.
Once a supervised run is selected, honor any admission refusal rather than retrying
the workload outside its limits.

Use `python3 tools/run_dev_guarded.py -- COMMAND ...` for builds, tests and short
checkpoint gates. It refuses to launch with less than 11 GiB of Linux
`MemAvailable` (3 GiB command allowance plus 8 GiB host reserve), then uses a
systemd user service with a 2 GiB soft threshold, **3 GiB hard RAM cap, no swap**,
150% CPU quota, five-minute timeout and the thermal guard. Cargo uses one build
job and tests run serially. The 1,152-task cap accommodates existing CPU tests
that deliberately request 1,024 workers. A killed or refused command is never
restarted or retried outside the guard.

The available-memory check is a snapshot, not a reservation against other sessions.
The hard limits cover this command tree; unrelated services and GPU allocations
still require their own limits. The wrapper requires Linux, a systemd user manager
and a `k10temp` thermal sensor, and fails closed when they are unavailable.
Pass command-specific environment values after `--` with `env NAME=value COMMAND`.
This short development supervisor is not a persistent serving service.

## Build and test

```sh
python3 tools/run_dev_guarded.py -- cargo build --release -p beholdd-cli --locked
python3 tools/run_dev_guarded.py -- cargo test --workspace --locked
python3 tools/run_dev_guarded.py -- python3 tools/test_sync_hag.py
python3 tools/run_dev_guarded.py -- python3 tools/test_run_dev_guarded.py
```

To check the native HAG adapter, Vulkan prototype and memory failure paths without
running any live GPU or checkpoint tests:

```sh
python3 tools/run_dev_guarded.py -- cargo test --workspace --locked \
  --features beholdd-backend/hag-native,beholdd-backend/vulkan,beholdd-memory/test-fault-injection
```

If the canonical HAG checkout is alongside Beholdd, verify its source snapshot:

```sh
python3 tools/sync-hag.py --source ../exec-hag --check
```

The check compares source bytes and Git provenance, including working-tree status;
concurrent uncommitted work in HAG can therefore make it fail even if vendored code
has not changed. The snapshot makes a sibling checkout optional. Refresh it with the same command
without `--check` only when intentionally importing that source revision. See
[HAG source ownership](hag-backend.md#source-ownership-and-reproducibility).

## Configure a small native model

The existing `config.example.yaml` is a historical 30B resident configuration with
a 40 GB execution budget. For a smaller development host, start with
`config.dev.example.yaml`, using a local **Qwen3-0.6B-Q8_0** checkpoint:

```sh
mkdir -p .beholdd/models
# Copy once; preserve an existing local configuration.
cp -n config.dev.example.yaml .beholdd/config.yaml
ln -s /absolute/path/to/Qwen3-0.6B-Q8_0.gguf .beholdd/models/Qwen3-0.6B-Q8_0.gguf
target/release/beholdd serve --config .beholdd/config.yaml
```

Alternatively set `path` in the local YAML to the checkpoint's absolute path.
`.beholdd/` is ignored by Git. The server uses loopback port 6768 and loads weights
only on the first completion. This small fixture verifies native execution; it does
not qualify the model as a daily coding assistant.

Check the service's implementation identity before directing clients to it:

```sh
curl --fail http://127.0.0.1:6768/health
curl --fail http://127.0.0.1:6768/v1/beholdd/info
curl --fail http://127.0.0.1:6768/v1/models
```

The identity must report `service: "beholdd"`, `generation: 2`, `runtime: "native"`
and `execution_backends: ["cpu"]`. An unloaded model is initially `unverified`;
only successful binding publishes `compatible` and the `completion` capability.
Use the requests in [native serving](serving.md#http-api) with model `qwen3-0.6b`.

## Real checkpoint gates after moving machines

Ignored tests never run as part of the ordinary suite. Select the exact fixture
explicitly; missing variables or files fail the selected test instead of reporting
a successful gate that never ran.

Prebuild the dense model/runtime test binaries:

```sh
python3 tools/run_dev_guarded.py -- cargo test --release --locked -p beholdd-model -p beholdd-runtime \
  --lib real_dense_qwen3 --no-run
```

The following runs both dense gates under the development supervisor:

```sh
python3 tools/run_dev_guarded.py -- \
  env BEHOLDD_QWEN3_DENSE_MODEL=/absolute/path/to/Qwen3-0.6B-Q8_0.gguf \
  cargo test --release --locked -p beholdd-model -p beholdd-runtime \
    --lib real_dense_qwen3 -- --ignored --nocapture --test-threads=1
```

The thermal guard uses this AMD host's `k10temp` sensor and refuses to start if the
sensor is unavailable or the host is too hot. Other machines require an appropriate
supervisor. The larger Coder tokenizer fixture uses `BEHOLDD_QWEN3_CODER_MODEL`;
the separate full Coder HTTP/embedded parity gate still uses `BEHOLDD_REAL_CONFIG`
and its exact checkpoint/template requirements. A small dense pass does not replace
that larger integration gate.

## Native backend boundary

The serving path is CLI/HTTP or embedded API → `NativeRuntime` → Beholdd model,
memory and the explicitly configured CPU or experimental HAG backend. It has no llama.cpp dependency or subprocess
fallback. GGUF type names, `tokenizer.ggml.*` metadata and frozen reference results
remain checkpoint compatibility contracts, not an external inference engine.

The Q6_K reader now uses Beholdd's contiguous-group plane decoder in place of the
previous GGML loop transcription. Tests exercise every six-bit value at every
block position, signed scales and signed zero while preserving the existing
multiplication order and checkpoint bytes.

HAG is our native GPU transport and bounded execution adapter. Read-only discovery:

```sh
python3 tools/run_dev_guarded.py -- \
  cargo run --locked -p beholdd-backend --features hag --example hag_preflight
```

This deliberately exits nonzero with `MemoryOnly` when memory discovery works but
general compute is not ready. It does not dispatch GPU work. Enabling `hag-native`
alone does not switch serving to a GPU; the explicit model configuration below
selects that experimental path. The selected complete dense graph passed its
bounded gate, followed by the scoped runtime/in-process HTTP/SSE serving gates
below. See [remaining HAG gates](hag-backend.md#remaining-replacement-gates)
and the concrete [whole-model integration target](gpu-integration.md).

## Explicit experimental HAG serving

CPU remains the default for all existing configurations. The
[`config.hag.example.yaml`](../config.hag.example.yaml) explicitly selects the
qualified dense Qwen3-0.6B geometry on `/dev/dri/renderD128`. It requires Linux and
the `hag-native` feature; unsupported devices, architectures, formats or budgets
are refused without fallback. Use a local copy if changing the checkpoint path:
the checked-in example uses a repository-relative model placeholder, which you
must set to the checkpoint location on your own machine.

The model's `hag.context_capacity` accepts **1–16384 evaluated positions** and
defaults to **2048**, including prompt/chat-template tokens. Admission requires
`prompt_tokens + max_new_tokens - 1 <= context_capacity`; the final selected token
is not evaluated. `weight_mode: streaming` means
one-at-a-time preparation reads; prepared GPU weights stay resident, and host
weight caches/prefetch remain disabled. No full-checkpoint host copy is created.
Request planning resolves an omitted `max_tokens` after tokenization to the smaller
of 512 and the remaining output budget. Explicit limits are kept intact or refused
with typed HTTP 400 before SSE headers. `/v1/models` exposes
`execution_limits.max_evaluated_tokens` equal to the configured capacity; CPU reports `null` and retains its
512-token omitted default. **Both scoped C128 gates passed with clean closeout**. The earlier
[C64/TCP qualification](gpu-integration.md#context-aware-c64-and-tcp-qualification--2026-10-04)
passed on October 4; the October 3 result below used a separate frozen C16 binary.

Build under the development supervisor:

```sh
python3 tools/run_dev_guarded.py -- cargo build --release --locked -p beholdd-cli --features hag-native
```

The explicit server invocation is:

```sh
HAG_TRACE_SUBMISSION=0 target/release/beholdd serve --config config.hag.example.yaml
```

Equivalent Cargo invocation:

```sh
HAG_TRACE_SUBMISSION=0 cargo run --release --locked -j1 -p beholdd-cli --features hag-native -- \
  serve --config config.hag.example.yaml
```

Run native serving under its separate process/thermal/crash supervisor, not as an
unbounded replacement for a refused gate. The historical C2048 serving gate uses a **2-GiB process
cap, no swap and 1.5-GiB early stop**, host-memory checks, the unchanged 90°C cutoff
and fresh driver/crash evidence. The short build wrapper above has different
limits and does not establish serving qualification. These commands are explicit
operator actions; configuration parsing and metadata listing never open a GPU.

The example's combined 2-GiB budget and 128-MiB runtime allowance admit a
**1,914,935,488-byte conservative forecast**, including the full checkpoint file,
sequential staging, logits/token buffers, a separate **1-MiB**
recording/owner metadata term and shared-UMA GPU charge. The exact HAG allocation
is **1,135,140,864 bytes**,
admitted against a separate **1152-MiB HAG budget**. Both configured budget and
current host/cgroup headroom are checked before device opening or payload reads.
The process cap and actual pressure measurements remain separate from HAG charges.

For larger development requests,
[`config.hag-16k.example.yaml`](../config.hag-16k.example.yaml) explicitly sets
C16384, a **6-GiB combined budget** and **4608-MiB native allocation budget**.
Its forecast with 128 MiB runtime headroom is **5,417,203,904 combined bytes**,
including **4,629,168,128 native bytes**. Supply a dedicated finite process cgroup
with the larger limit, zero swap, and host/thermal monitoring; the C2048 gate's
2-GiB cap cannot accommodate this forecast. Bounded hardware-free validation
passed; live C16384 numerical, lifecycle and editor behavior remains unqualified. No GPU
run accompanies this configuration change. Historical C2048 evidence does not
qualify the changed source or larger capacity.

C2048 full-model numerical qualification passed. The exact synthetic no-tools
project-instructions serving/client profile also
[qualified on validation-host on October 8, 2026](hag-editor-context.md#qualified-c2048-project-instructions-on-validation-host--2026-10-08),
with Linux `6.17.0-1024-oem`. Its 467-token prompt and 128-token reply allowance
produced three native requests of 473 evaluated positions each, with exact CPU
text, EOS and usage parity. Full editor/tools, general quality, concurrency,
throughput and live cancellation remain outside this proof. A separate
[C2048 TCP lifecycle gate](hag-editor-context.md#qualified-c2048-tcp-lifecycle-on-validation-host--2026-10-08)
passed on the same host: one nine-token CPU reference, four successful native
one-token replies and one healthy socket cancellation, totaling 46 native
evaluations. Busy unload, reuse, idempotent unload, same-process reload and clean
idle SIGTERM all passed under unchanged limits. The frozen C512
host-client evidence remains separate. The [qualification runner](hag-qualification.md) sets
`BEHOLDD_HAG_PRIVATE_CGROUP` only after verifying its dedicated stage cgroup.
This explicit attestation names the exact unified membership path and permits
bounded credit for already charged checkpoint pages during reload. Ordinary
launches without it retain conservative admission. Do not set it for a shared
cgroup: its other work is outside the serving forecast. Total configured/host
bounds and additional-allocation headroom still apply.

`/v1/beholdd/info` reports `["cpu", "hag"]` only when this HAG implementation is
compiled and at least one model explicitly selects it. CPU-only configurations
still report `["cpu"]`. Model registration stays metadata-free; first successful
preparation publishes observed compatibility/residency. Requests and unload retain
runtime admission through synchronous completion/cleanup.

Cancellation is checked around fully fenced tokens, including prompt evaluation
before text exists. Healthy control/sink cancellation resets caches for reuse.
Execution/read/reset uncertainty or retained hardware charges after unload latch
HAG unhealthy process-wide until restart. Explicit unload reports uncertain
cleanup; eviction cannot restore HAG health or bypass quarantine. Explicit CPU
selection can still proceed, but no failed HAG request silently falls back.

Serialization/admission belongs to one `NativeRuntime` instance; this server setup
and gate use one instance. Multiple simultaneous instances/processes and device
sharing are unqualified. The process-wide health latch does not provide admission
across instances.

The CLI accepts SIGINT/SIGTERM for graceful shutdown and waits for admitted blocking
workers to retire before returning. This does not force-cancel GPU work or promise
a fixed deadline for active requests. `HAG_SERVING_AUDIT=1` optionally logs request
sequence, evaluated-position count, outcome, reset and poison status, without
prompts, token IDs or tensor payloads. The real CLI/TCP gate requires a cancelled,
healthy-reset audit before testing reuse and idle unload/clean exit.

### Integer-list EOS and timing result — 2026-10-04

The fresh cooled-machine run passed CPU reference, native answer parity and timing
with clean final closeout. The single list fixture returned identical raw text,
usage and EOS finish: 29 prompt + 27 emitted tokens, 56 evaluations per request.
The native answer used 56 evaluations and four timing requests used 224. First
lazy-load response was 12.439 seconds; three warm SSE responses reached DONE in
11.588–11.821 seconds under the unchanged limits. These are descriptive client
timings; recall, general answer quality and decode throughput remain unqualified.
Four semantic/format failures and the thermal attempt below remain preserved.
See the [scoped result and evidence](hag-backend.md#integer-list-eos-parity-and-response-timing--2026-10-04)
in `$EVIDENCE_DIR/work/hag-list-cooled-20261004/`.
CPU remains the default; the earlier C128 record is unchanged.

### C128 implementation and qualification — 2026-10-04

At this milestone, source prepared **128 evaluated positions**. **Both scoped live gates passed**
in `work/hag-c128-20261004/`; earlier C64 source identities and measurements remain
historical. Public tensor bytes are 33,793,536, peak auxiliary bytes 14,842,200 and
HAG charge 681,910,272. Native metadata is 94,711,519 bytes, with the rounded
91-MiB model/recording allowance counted in its 1,556,061,376-byte runtime forecast. The
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

### Complete-answer attempt and resume — 2026-10-04

The fresh list-answer harness passed 60 tests and its contained self-check, then
the CPU reference hit the unchanged thermal guard at **90.375°C**, before any HTTP
response. Native answer parity and timing did not run. The cgroup peaked at
**269,860,864 bytes**, with no memory pressure, OOM, swap or matching driver faults;
process and service cleanup was verified. Existing C128 qualification is unchanged.

This separately frozen harness accepts one optional terminal period in the ordered
1–8 list; raw CPU/HAG parity, EOS and safety requirements are unchanged. Four prior
CPU semantic/format failures remain preserved. At that point complete-answer and
timing qualification remained open; recall was outside this fixture's scope. See the
attempt evidence under `$EVIDENCE_DIR/work/hag-list-sentence-20261004/STATUS.md`;
that host-local archive is not distributed with the repository.
The subsequent cooled-machine run above used fresh evidence, passed CPU reference
first, and retained the 90°C cutoff and fresh native crash capture.

### Recording and scratch storage reduction — 2026-10-04

The frozen optimized C64 batch representation keeps compact dispatch descriptors and one retained
owner reference per distinct allocation, with exact-bit immutable parameter sharing
and a scratch workspace sized to the largest serial operation. Operation order and
per-dispatch fences were unchanged; that frozen proof kept **serving at C64**. Auxiliary peak is
7,480,664 bytes, native recording metadata 47,540,703 bytes and the rounded
recording/model allowance 46 MiB. The boundary probe forecasts 1,519,718,559 bytes,
with a 1,485,916,864-byte runtime forecast at that frozen C64 stage.

That release's C128 candidate was metadata-only: 14,842,200 auxiliary bytes,
681,910,272 HAG bytes, 94,711,519 native metadata bytes and a 1,604,527,519-byte
conservative probe forecast. Its archived report keeps `qualified: false` and
`serving_capacity: 64`; the separate C128 qualification is recorded above.
Forty-eight wrapper tests and the harmless containment self-check passed. Fresh
frozen inputs are in `work/hag-recording-20261004/`; **both fresh live gates passed**
at 12:15–12:17 UTC. The boundary checked 911,616 values with maximum error
`9.1552734375e-5`, one sign mismatch and zero zero/nonzero mismatches and six matching CPU greedy choices,
then retired the 658,952,192-byte HAG charge. TCP made 136 evaluations with healthy
socket cancellation/reset, reuse, unload/reload and CPU/HAG exit 0. Process peaks
were 79,228,928 / 279,310,336 bytes and maximum Tctl 82.875°C / 57.375°C, with no swap,
OOM or matching driver faults and unchanged 90°C cutoff. Its two-token references
remain only `<think>\n`; no answer-quality or throughput claim follows. The archive
is `outputs/hag-recording-qualified-20261004.tar.gz`. No prior results or stopped
watcher were reused.
The C64 numerical/TCP proof below remains a historical baseline with separate
source identity and costs. See the [recording record](hag-backend.md#recording-and-scratch-storage-reduction--2026-10-04).

### C64/TCP qualification — 2026-10-04

Current hardware-free checks cover context planning, pre-header typed refusals,
queued and blocking cancellation/admission, graceful shutdown and gate validation.
The live boundary probe prepared C64, compared full logits/greedy choices at
positions 0, 15, 16, 31 and 63, refused a 65th visible position, and verified
reset/replay. Across 65 evaluations and six readbacks, all 911,616 values met the
unchanged tolerance; maximum error was `9.1552734375e-5`, all six greedy choices
matched CPU, one near-zero sign mismatch was reported and HAG charge returned to
zero. The independent CPU oracle also passed a tiny production-forward parity
test through position 63.

**The real CLI/TCP gate also passed**: six successful requests and one cancelled
request, 136 evaluations, cold/warm typed refusals, actual socket disconnect,
healthy reset/reuse, busy unload, unload/reload and CPU/HAG SIGTERM exit zero.
The three CPU reference prompts (9, 21 and 63 tokens) each emitted two tokens,
`<think>\n`, matching HAG text/usage. This does not establish complete-answer quality.
Boundary/TCP process peaks were 89,931,776 / 280,555,520 bytes and maximum Tctl was
77.625°C / 59.625°C; no swap, OOM or matching driver fault was recorded.
Evidence is in the development chat's `work/hag-context-20261004/`, with archive
`outputs/hag-context-qualified-20261004.tar.gz`. No persistent service, default or
throughput claim changed.

The October 3 dedicated runtime and in-process HTTP/SSE gates passed CPU token/text parity,
healthy cancellation/reuse and unload/reload in the scope below. The earlier
capacity-eight complete-model result retains its separate two three-token requests;
neither historical result qualifies C64, all sixteen positions, concurrent device sharing or
general production serving.

Before the serving release freeze, the bounded HAG runtime suite passed **36 tests**,
the full workspace passed **466 with five ignored**, and the final targeted
runtime/server/CLI/embed suite passed **74 with three ignored**. Strict Clippy also
passed. The ignored cases include the new explicit live gates; ordinary test
success does not imply they ran. The release build freezes 128 source inputs.

### HAG serving result — 2026-10-03 EDT

The frozen runtime stage passed at **23:40 EDT on October 3, 2026**
(**03:40 UTC on October 4**), followed by HTTP/SSE at **23:41 EDT / 03:41 UTC**.
Runtime generation evaluated nine prompt tokens and one decode token, positions
**0..9**, and matched CPU tokens **`[151667, 198]`**, text and usage exactly.
Insufficient-budget metadata admission refused before opening a missing device;
early cancellation left the model unloaded, and successful execution unloaded.

Four successful one-token HTTP requests matched CPU **`[151667]`**, text and usage
through the real in-process Axum router, including complete SSE, reuse and reload.
A first-text SSE disconnect was observed, followed by healthy reuse. Capacity
refusal, busy-unload **409**, idle/idempotent unload, reload and final unloaded state
all passed. This was a real response-body test without a TCP listener.

HAG's **649,277,440-byte** charge per load retired completely under the enforced
cleanup/health checks. Process/cgroup peaks were **245,776,384 / 245,161,984 bytes**
for runtime / HTTP, separately from GPU charges. Maximum sampled Tctl was
**59.5°C / 58.375°C**, and host available memory stayed above
**21,536,968,704 / 21,996,560,384 bytes**. Both stages had zero swap, memory/OOM
events, driver fault matches or crash dumps. The 2-GiB/no-swap, 1.5-GiB early-stop
and 90°C controls stayed unchanged; the wrappers passed 22 tests beforehand.

Root capture **3539802** covered **2026-10-04 03:40:09.450761–03:41:47.996004 UTC**,
stopped cleanly and had no matching faults in its complete **458-line kernel
window**. Closeout verified 128 source inputs, frozen test/CLI/dependency hashes
and the unchanged checkpoint. See the [complete dated serving record](hag-backend.md#experimental-hag-runtime-and-httpsse-qualification--2026-10-03-edt)
and `outputs/hag-serving-qualified-20261003.tar.gz` for identities and evidence.
These results qualify this single-runtime scope, not TCP serving, throughput,
all sixteen positions, multiple concurrent runtimes/processes or device sharing.
CPU remains the default; no persistent service was enabled by these tests.

## Historical migration qualification — 2026-10-03

The recorded development environment used Rust/Cargo 1.98.1, LLVM 21 and `glslc`.
Linux reported about 30.6 GiB RAM and AMD device `0x1586` with 40 compute
units; HAG preflight discovers `/dev/dri/renderD128` and reports `MemoryOnly`.

The local configuration in `.beholdd/config.yaml` uses the 639,447,744-byte
`$BEHOLDD_MODEL_DIR/qwen3-0.6b-draft/Qwen3-0.6B-Q8_0.gguf` on NVMe,
with a 3 GiB execution budget and 1 GiB runtime allowance. The migration smoke
used a temporary 4 GiB/no-swap service under the thermal guard on port 6768.

The native HTTP smoke passed health/identity checks, eight-token generation,
identical SSE text, publication of observed residency/capabilities and explicit
unload. The temporary service was then stopped. Both dense checkpoint gates passed
under the documented supervisor: 4.062 seconds service runtime, 691.1 MiB peak
memory, zero swap and 71.625°C maximum sampled Tctl. These are short correctness
checks, not throughput measurements or complete Coder tool/HTTP parity gates.

The workspace suite with HAG native, Vulkan and memory fault injection passed
391 tests (including one doctest), with nine opt-in cases ignored; the two dense
cases above were then run separately. All 24 HAG sync-tool tests passed. The initial
setup snapshot was verified against clean canonical revision
`a7cf191497e9a25f3192adeb8b880053000e8882` without changing HAG source bytes;
subsequent snapshot refreshes follow the parallel HAG qualification work.
Standalone HAG passed 65 hardware-free tests; Beholdd's explicit LLVM shader check
also matched all embedded instruction bytes using the installed LLVM 21 tools.

External inference services and their deployment settings are outside this
repository's qualification scope. A small dense checkpoint does not establish
compatibility with larger models, multimodal input or RPC execution. Operators
must budget concurrent services independently and avoid uncontrolled restart loops;
the development wrapper bounds only its own command tree.

## Native RoPE, attention and resident KV qualification — 2026-10-03

The frozen ten-program adapter passed the resumed live gates at 20:19–20:20 UTC
on GC 11.5.1. The earlier heat hold ended without relaxing any limit. RoPE checked
1,562 values bitwise exactly and 26 guards across 13 dispatches. Causal GQA checked
3,218 values and 14 guards across 183 dispatches, including cached suffixes,
65-position tails and poisoned future rows excluded by causal descriptors.
Maximum attention error was `1.1920928955078125e-7` under the predeclared
`2e-5 + 2e-5 * abs(reference)` bound.

The real Qwen3 layer-0 attention/residual block passed two requests of three
positions using one eight-position cache allocation. Its six submissions ran 762
physical dispatches and checked 6,144 final hidden values; maximum error was
`2.384185791015625e-7` under `3e-4 + 3e-5 * abs(reference)`. Reset changed only
cache visibility. Seven native preparation reads loaded four Q8_0 matrices and
three F32 norms once; the independent CPU oracle made 42 separate reads before
GPU open. Submission performed no tensor transfers; verification read 24,576 bytes.
The 7,098,368-byte HAG allocation matched preflight and returned to zero.

The block process peaked at 50,274,304 bytes (47.9453125 MiB), with maximum sampled
Tctl 55.75°C. All gates verified the same 1-GiB, zero-swap, one-CPU, 64-task and
90°C limits and recorded zero memory/OOM events or driver faults. Fresh read-only
capture ran from 20:19:39.673509 to 20:20:28.695911 UTC, stopped cleanly with zero
dumps, and the complete 213-line kernel window contained no GPU/OOM/fault matches.

The frozen binaries, 119-file source manifest, earlier preparation and complete
resumed evidence are preserved in `outputs/hag-rope-attention-qualified-20261003.tar.gz`
in the development chat; `outputs/hag-rope-attention-status.txt` records the current
handoff. This qualifies the stated finite diagnostics and selected real-weight
block with synthetic boundary inputs, including its per-head width-128 norms and
split-half RoPE. Full-model embedding/head residency, logits/token parity, GPU
serving, broader inputs and performance were separate gates at this stage. Serving
was then CPU-only; the later complete-model and serving records retain separate scopes.
