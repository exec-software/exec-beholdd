# OpenMythos native HAG development

Native OpenMythos support is in development. This chronological record begins
with CPU references and bounded native operators, then tracks resident-state and
runtime integration below. Those component results do not independently qualify
full-model execution, broad hardware support, or a production backend default.

The experimental profile requires separate numerical, resource and chat integration
qualification. Model loading remains lazy: starting a service must not load a
checkpoint before the first request. Keep a qualified deployment available until
the replacement passes its own release gates.

## Checkpoint and independent references

The selected checkpoint is `OpenMythos-27B-Q4_K.gguf`, 16,547,399,872 bytes,
SHA-256 `d3c3398ba8b512aa59074efcaa34d7dcabed7381b68a05ae6bcce77e14d7db45`.
Its architecture is dense `qwen35`, with 64 blocks, hidden width 5120, FFN width
17408, and an untied vocabulary projection. Every fourth block has full attention
(24 query heads, 4 KV heads, head width 256, leading rotary width 64). The other
48 blocks use Gated DeltaNet (16 key heads, 48 value heads, head width 128,
four-tap causal convolution). Its 851 tensors contain 433 Q4_K matrices,
65 Q6_K matrices and 353 F32 tensors.

The architecture and GGUF layout reference is llama.cpp commit
`4da6337767f973e2b4d0797e5b323d77d8565e4a`:
`src/models/qwen35.cpp`, `src/models/delta-net-base.cpp`, `src/models/models.h`,
`ggml/src/ggml-cpu/ops.cpp`, `ggml/src/ggml-quants.c`, and `src/llama-vocab.cpp`.
The model/conversion/tokenizer reference files are unchanged in the local pinned
checkout. The Python oracle implements those equations independently of Rust.

Several conventions matter for correctness:

- Stored `ssm_a` is already `-exp(A_log)`; converted RMS weights already contain
  their offset, except the SSM norm. Do not apply those conversions twice.
- DeltaNet value head `h` uses query/key head `h % 16` in this GGUF. Convolution
  history stores raw projected QKV, with taps ordered oldest through current.
- Q/K normalization uses `x / sqrt(sum(x*x) + epsilon)`. The recurrent readout
  uses the updated, decayed state and a `1/sqrt(head_dim)` query scale.
- Full-attention query/gate projection interleaves both vectors within each head.
  Query/key RMS normalization precedes partial RoPE, then grouped attention and
  sigmoid gating. Text positions use identical MRoPE coordinates; image/video
  grids are outside this initial implementation.
- The `qwen35` tokenizer keeps Unicode marks within word bodies. It does not
  normalize Unicode. Qwen2 tokenization retains its existing behavior.

## CPU reference gate

`beholdd_model::qwen35` provides a strict header-only tensor binder, transactional
DeltaNet and full-attention state, and exact row-tiled Q4_K/Q6_K weight reads.
The reference runner forecasts state and scratch before allocation. A failed
partially evaluated block/token requires an explicit sequence reset. It never
silently retries with another backend.

The development example `qwen35_block_reference` and independent
`tools/qwen35_block_ref.py` accept the same versioned hidden-vector JSON and
evaluate one selected block from empty state. Both require explicit inputs;
neither runs as a service or loads a checkpoint automatically. Real-weight
comparisons must run serially under the host's CPU/memory/thermal supervisor.

The first guarded compile and arithmetic run passed 24 tests. The actual
checkpoint's header binding and independent tokenizer token-ID goldens also
passed. The two scopes peaked at 685,133,824 and 110,850,048 bytes, respectively,
with zero OOM events and zero swap. Maximum observed Tctl was 48.75 degrees C.
Evidence: `work/openmythos-20261020/qwen35-first.{log,json}` and
`qwen35-header.{log,json}` in the development chat workspace. These checks do
not qualify complete blocks, full-model inference, GPU execution, or deployment.

## Actual block comparison — 2026-10-20

The CPU reference implementation is committed at `c41a6fa`. Its serial guarded
validation passed 131 model tests, five CLI boundary tests and 12 independent
Python tests; the release example built successfully. The validation scope peaked
at 439,373,824 bytes and Tctl 58 degrees C, with no OOM events or swap.

The release example and independent f64 oracle then evaluated five changing,
exactly F32-representable hidden vectors through actual checkpoint blocks 0 and 3.
This exercises the complete causal convolution history and repeated attention.
All 51,200 output values passed the predetermined element tolerance
`2e-4 + 2e-4 * abs(reference)` and relative L2 limit `2e-5`:

| Block | Path | Maximum absolute error | Relative L2 error |
| --- | --- | ---: | ---: |
| 0 | DeltaNet | 0.0002123713 | 0.0000018475 |
| 3 | Full attention | 0.0001106270 | 0.0000016652 |

Both Rust runs also refused an extra token without reading weights, changing the
position count or poisoning the valid sequence. Reset followed by replay of the
first input exactly reproduced the original output. Each individual weight read
was at most 228,480 bytes. The admitted block-plus-JSON payloads were 22,000,897
bytes (DeltaNet) and 15,321,365 bytes (full attention), against explicit 128 MiB
budgets. Source reads include replay: 1,513,579,776 and 1,402,564,608 bytes.

The complete four-process comparison scope peaked at 548,036,608 bytes, including
charged file cache, with Tctl at most 58 degrees C, at least 18,903,498,752 host
available bytes, zero OOM events and zero swap. All children and the scope retired.
This qualifies these selected CPU blocks; it does not qualify full-model outputs
or native HAG execution.

Evidence in `work/openmythos-20261020`: `qwen35-validation.json`,
`qwen35-real-blocks.json`, `blocks-run.json`, `block-comparison.json`,
`rust-layer{0,3}.json` and `oracle-layer{0,3}.json`. The input SHA-256 is
`892e103524729fd047e29e647f9571a5dfe217a23d27a16b8f8a72f048ee5b84`;
the release probe SHA-256 is
`bd05fec1fb393d8a2423b46ac8b265d99a34a8be7a85a7dd4ede957136121222`.

## Native operator development

The explicit `Qwen35Probe` execution profile has a separate immutable program
library. The default dense Qwen3 profile retains its existing programs, program
IDs, formats and shape bounds. The new profile admits Q4_K/Q6_K matrix rows up to
17408 columns (whole 256-element blocks), at most 4096 output rows per tile, and
embedding-row decode up to 5120 columns. Its RMS program reduces all 5120 input
elements; weighted normalization then applies pointwise weights in bounded slices.

The five new GFX1151 wave64 sources assembled and disassembled successfully with
LLVM 21. Their padded sizes are 856/788 bytes for Q4_K/Q6_K matvec, 788/796 bytes
for row decode, and 544 bytes for wide RMS. The generated words contain no unknown
instructions or fused multiply-add operations. The existing default library is
unchanged. The synthetic operator gate below qualifies these selected fixtures;
it does not qualify a complete native Qwen3.5 model or change serving admission.

The assembler-only guarded scope used 32,972,800 peak bytes, no OOM events or
swap, and at most 41.75 degrees C. Evidence is `qwen35-assembly.{log,json}` and
`isa/assembled.json` plus `.bin`/`.disassembly` files in the same workspace.

### Synthetic native operator qualification

The operator source checkpoint is `18cbbdd`. All 172 backend library tests, both
opt-in LLVM source/embedded-byte tests, and three probe tests passed. The guarded
release build used 392,691,712 peak bytes, at most 62.125 degrees C, no memory
events and no swap. The release `hag_kquant_probe` SHA-256 is
`27cf4051aafa5864f62e0735666c12ec7e8dd48de5d17d8bb8ad0bbe1fefeac8`.

The separately retired CPU reference checked 3,119,920 values without device
access. The native run then passed 44 physical dispatches / 40 logical submissions:
Q4_K and Q6_K matvec at 3/65 rows and 256/5120/17408 columns, changed-input replay,
selected-row decoding including two-byte-aligned Q6_K rows, and weighted/unweighted
5120-wide RMS with ordinary and badly scaled inputs. All 53,552 compared values
had zero observed absolute error; all 80 output guard checks passed. Quantized
row/matvec comparison required exact bits. RMS used preregistered absolute and
relative tolerances of `2e-6` each, with no threshold adjustment after execution.

The queue charged exactly 1,253,376 bytes under an 8 MiB budget and retired to
zero. Weight arena reuse was required: 2,142,408 bytes uploaded through a 1 MiB
arena. Submission performed no host payload transfers. Total explicit readback
was 214,528 bytes; host writes were 3,679,720 bytes including initialization.
Peak live arenas were 928,200 weight / 69,900 tensor / 20,484 auxiliary bytes.
The native scope peaked at 33,574,912 bytes, CPU Tctl 41.875 degrees C and GPU edge
42 degrees C, with zero memory events, swap, boot OOM changes or kernel faults.
The owned service, cgroup, observed processes and read-only root crash watcher
all retired; the watcher captured no dumps.

Evidence is under `work/openmythos-20261020/gpu-v2`: frozen source/binary/policy
inventory, `runs/cpu-1792541351486510083/result.json`,
`runs/native-1792541520674042466/result.json`, their child/log/thermal/cgroup/kernel
records, and `watcher/retired.json`. Earlier attempts refused before probe startup:
the first supervisor expected a cpuset controller absent from user cgroups, and
one native attempt warmed beyond the startup threshold. The corrected supervisor
uses independently attested parent/child process affinity on CPUs 0 and 1; the
same thermal and memory limits remained in force for the successful run.

The subsequent standalone HAG checkpoint `cd23d42` raises the software library
ceilings to 32 programs / 32 KiB for the upcoming DeltaNet and separate attention
gate operators. Each program still fits 4 KiB and starts at a 256-byte boundary;
existing program offsets, payloads and actual charges remain unchanged. All 81
HAG library tests passed with `capi` enabled, including eight-page charging,
last-entry dispatch, retirement and pre-allocation rejection. This capacity
change does not qualify any additional shader or full model. Beholdd vendors the
exact clean source snapshot through `tools/sync-hag.py`.

### Resident state and streaming host gate

The bounded `qwen35::recorded_weights` helper reads, uploads, records and fences
one exact packed tile at a time. Every tile handle is retired before the next
read; caller-owned activations remain resident. Cancellation can leave a partial
destination, so the model owner must reset or abandon that incomplete token.

The explicit probe library now adds four Delta programs (12 through 15) and a
separate sigmoid gate (16). Their padded sizes are 1748, 2348, 1752, 1556 and 564
bytes. The existing dense Qwen3 library and its admission remain unchanged. These
programs have passed assembly/byte agreement, not GPU numerical qualification.

`NativeDeltaSequence` owns immutable parameters, bounded projections, two state
banks and six guard words in one allocation. For OpenMythos this allocation is
6,899,548 bytes. A normal step submits four ordered programs, then reads only
192 status bytes outside submission. It publishes the next bank only when every
head succeeds and cancellation remains clear. Reset actually zeros both banks
and scratch; it never recovers a poisoned native queue. The exact layout and F32
rounding contract are in [qwen35-delta-packet.md](qwen35-delta-packet.md).

The guarded host gate passed 187 backend tests, 135 model tests, three opt-in LLVM
source/byte tests and six independent Python packet/oracle tests. The successful
scope peaked at 472,444,928 bytes and Tctl 60.5 degrees C, with no memory events
or swap. Its service, PID and cgroup retired. The initial scope hit its 64-task
limit in two existing CPU tests that deliberately request 1024 threads; rerunning
with 2048 task slots passed under the same 3 GiB memory ceiling and CPU quota.
No model or GPU was opened by either scope. Evidence is
`work/openmythos-20261020/qwen35-delta-{assembly,tests}.{log,json}` and
`delta-{backend,llvm,model,independent}-tests.log` in the development workspace.

### Resident layer composition and first state GPU gate

The explicit development components `recorded_attention`, `recorded_ffn` and
`recorded_delta` now compose these operators with bounded packed-row streaming.
They admit source identity, exact shapes and payload limits before reading
weights, retain activations privately, and require reset after a partially
mutated step. Full attention retains per-sequence KV and partial-RoPE tables;
Delta retains both recurrent banks. The FFN uses five un-clipped SwiGLU slices
for the checkpoint's 17408-wide intermediate vector. Complete finite checks of
operands and results remain necessary before serving admission; checking only
final hidden values can miss an infinity masked by a nonlinear operation.

Their guarded host validation passed 187 backend tests, 148 model tests, three
opt-in LLVM checks and nine probe tests. This includes two-token CPU comparisons
of resident attention/FFN against the independent component equations, partial
residual cancellation/reset checks, and metadata-only native Delta admission.
The scope peaked at 509,325,312 bytes and Tctl 65.5 degrees C, with no memory
events or swap. Its release probes and source inventory were frozen afterward.
An initial compile caught an ambiguous float type in one new test fixture; the
explicit F32 annotation fixed it. Both attempt records are preserved.

The retired CPU prerequisite passed all three probe contracts: 3,119,920 packed
operator comparisons, 8,324 sigmoid comparisons, and 18 independent Delta golden
checks. Delta also generated 5,911,115 reference values for GPU comparison; these
generated values are not counted as CPU comparisons.

The first native attempt passed packed operators and sigmoid (including exact
`-103`/`-105` F32 tail anchors), then **failed** the Delta tail contract. It had
completed 184 Delta dispatches and 24,161,551 comparisons when convolution SiLU
produced `-7.374658e-38` against reference `-7.374608e-38`, beyond the predetermined
64-ULP product tolerance. The remaining Delta checks did not complete and the
combined gate is not qualified. Tolerances were not adjusted. The diagnostic
retired all allocations, processes and its cgroup; no OOM, swap, kernel fault or
crash dump occurred, and the root watcher stopped cleanly.

Evidence: `work/openmythos-20261020/qwen35-state-build.{json,log}`,
`state-*-tests.log`, and `gpu-state-v1` with frozen policies/binaries,
`runs/cpu-1792544613525206331/result.json`,
`runs/native-1792544642283683514/result.json`, and `watcher/retired.json`.

### Corrected state GPU qualification

Checkpoint `f2542a9` reduces exponential arguments with a split `ln(2)` before
the GPU SFU, preserving the explicit F32 rounding boundaries. The corrected
shader sizes are 1796, 2492, 1752, 1604 and 564 bytes; the library charge is
unchanged. Guarded validation passed 189 backend tests, three LLVM checks and
nine probe tests before a fresh CPU prerequisite and native run.

The corrected native gate passed all three synthetic probes without changing
their numerical tolerances: 290 physical dispatches, 106 logical submissions,
24,232,941 output comparisons, 16,648 input checks and 548 guard checks. Delta
completed all 240 dispatches and 24,171,065 comparisons, including 378 tail
checks, 108 decay-factor checks, 12 arithmetic refusals, 16 cancellations and
four resets. Every probe retired its allocation budget to zero.

This small synthetic gate used a separately reviewed 256 MiB memory-high /
512 MiB hard ceiling, zero swap, and a 9 GiB startup / 8 GiB active host reserve.
Those limits apply only to these fixed probes, not to model or build workloads.
The scope peaked at 60,538,880 bytes, with no memory-limit, OOM or swap events.
No kernel fault or crash dump occurred. All processes, the cgroup and the
authenticated crash watcher retired. This qualifies these operators; complete
model execution and serving admission remain pending.

Evidence: `work/openmythos-20261020/state-build-v2/state-build-receipt.json`,
`gpu-state-v3/memory-policy-review.json`,
`gpu-state-v3/runs/cpu-1792548008946440769/result.json`,
`gpu-state-v3/runs/native-1792550835299688405/result.json`, and
`gpu-state-v3/watcher/retired.json` in the development workspace. Earlier failed
or unarmed attempts remain preserved separately.

### Integrated runtime, protocol and fullgraph host gate

Checkpoint `39f94b4` adds the explicit lazy Qwen3.5 runtime owner in
`beholdd-runtime`, the OpenMythos output dialect, protocol module and chat
template in `beholdd-server`, and the `hag_qwen35_model_probe` fullgraph
development example. These landed from the staged sources after the block
qualification above. No GPU, checkpoint or deployment step ran for them.

That commit's test targets did not build. The new `OutputDialect::OpenMythos`
variant left a `match dialect` in the bounded-schema fixture non-exhaustive, so
`beholdd-server`'s lib test target failed to compile while `cargo check
--workspace` still passed. The fixture now excludes the dialect explicitly:
OpenMythos spells tool-call arguments through its own function-tag format with
separate number and argument spelling, not Qwen3's JSON wrapper, so this fixture
cannot encode it. Bounded-schema validation for the OpenMythos dialect is
therefore not covered and requires its own fixture.

The guarded host gate then passed 524 tests, zero failed, seven ignored:
64 backend, 157 model, 90 server, 60 memory, 49 format, 35 kernel, 43 runtime,
13 core, eight CLI, three embed, one real-native-parity and one documentation
test. The scope peaked at 1,503,238,553 bytes with maximum Tctl 80.75 degrees C,
zero memory events and zero swap; its unit retired. `tools/sync-hag.py --check`
verified 22 vendored HAG files at `cd23d42` against a clean source tree.

This qualifies compilation and the host test suite only. Complete native model
outputs, the chat template's serving behavior, thinking and tool-call parsing,
context admission, cancellation and restart remain unqualified. The native HAG
serving gate still admits only the qualified Qwen3-0.6B Q8_0 checkpoint.

## Remaining qualification

1. Qualify bounded native finite checks and checked resident layer composition,
   then compare actual-weight blocks with the independent reference.
2. Stream exact packed tiles through one native queue while keeping activations,
   KV and recurrent state resident. Retire submissions before reusing tile space.
   The current 1 GiB adapter weight arena and four-GiB buffer-addressing limit
   cannot be removed by relabeling a larger allocation.
3. Compare complete native model outputs with an independent reference, then
   qualify the checkpoint's chat template, thinking/tool-call parsing, context
   admission, cancellation and restart behavior.
4. Deploy qualified binaries/profiles to alternate-host and validation-host; verify cold start
   leaves models unloaded and first-prompt loading, streaming and rollback work.

At context 2048, the model's F32 KV payload alone is 256 MiB. Recurrent state is
144 MiB plus 5.625 MiB of convolution history. Encoded weights, transient buffers,
driver allocations and host memory require separate accounting. Real GPU probes
retain crash watching, resource admission and thermal supervision.
