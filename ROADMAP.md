# Roadmap

Each milestone ends in a **gate**: something that either passes or does not, checkable by
running it. No milestone is "done" because the code exists — that is the failure mode this
project was created to avoid.

Milestones separate hardware-free checks from gates that require an explicitly selected
checkpoint, device, or toolchain. Obtain those prerequisites independently before running
an opt-in gate; they are not included in this repository.

---

## M0 — Foundations, verifiable with no model and no GPU

The bottom of the stack: what a number is, how it is stored, and how it is multiplied.

- `beholdd-core` — dtype and quant-format vocabulary: E2M1/MXFP4, E4M3, E8M0, bf16, f16,
  f32. Decode semantics for each, including the parts that are easy to get subtly wrong
  (E8M0 255 is NaN and means zero; E2M1 has no inf and no NaN; bf16 widening is a pure
  shift with no rebias).
- `beholdd-kernel` — a scalar reference for every operator, then vectorised paths that are
  diffed against it. Decode in the register file from day one: `vpermps` is a 16-entry
  float table and E2M1 has exactly 16 values.
- `beholdd-format` — Safetensors and GGUF readers: header parse, tensor index, byte
  ranges and contiguity detection.

**Gate M0.1** — PASSED. Every quant format is checked against an independent
transcription of its spec, not against its own decoder. Caught three bugs on the first
run, all of the silent-wrong kind.
**Gate M0.2** — PASSED. The vpermps path agrees with the scalar definition to 7.4e-07
relative, against a stated tolerance of 1e-5 justified by E2M1 carrying two mantissa bits.
**Gate M0.3** — PASSED. 46 shards, 69,187 tensors, 159.61 GB on the real DeepSeek-V4
checkpoint, cross-checked three ways. Reproduces the two-pread expert geometry.
**Gate M0.4** — PASSED. 13.57× over the scalar definition, single-threaded.

> Status: M0 kernel surface complete — MXFP4, FP8-E4M3 and bf16 matvecs, plus rmsnorm,
> softmax, SwiGLU and rope. Each has a scalar definition, a vectorised path, and a test
> that they agree within a tolerance justified by input precision.
>
> The earlier GGUF-versus-Safetensors choice is resolved. Qwen3-MoE became the first
> architecture, and the GGUF reader plus the required K-quant kernels landed for its real
> checkpoints. Safetensors remains a first-class format reader; both formats now feed the
> checked memory-source contract proven by Gate M2.4.

---

## M1 — One forward pass, on real weights, provably correct

- `beholdd-backend` — the `Device` trait and the `cpu` implementation.
- `beholdd-model` — **Qwen3-MoE** as the first concrete architecture. Chosen deliberately:
  it is the plainest thing we run — GQA, RMSNorm, RoPE, SwiGLU, top-k router — so any
  difficulty is our seam being wrong rather than the architecture being hard. A common
  model interface and declarative spec loader remain planned; the passed M1 gates do not
  claim that they exist.
- `beholdd-memory` — resident loading only. No streaming yet.

**Gate M1.1 — PASSED.** Four positions through block 0 match an independent f64 NumPy
transcription on the real Qwen3-Coder-30B-A3B Q8_0 checkpoint. All 32 routed-expert
selections are exact; routing weights differ by at most 6.85e-7; the final hidden state
differs by 8.56e-7 relative L2 and 9.54e-7 maximum absolute. The checked path includes
per-head QK-norm, split-half/NEOX RoPE, causal GQA, full-softmax → stable-top-k →
renormalise routing, and demand-loading each selected expert as one positional read.
Checkpoint: `Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf`, SHA-256
`a4a0207f4653bfece73d9818c83acf714f5593525fe3aab7026347fd73090fcc`.

```sh
python3 tools/qwen3moe_ref.py <model.gguf> /tmp/qwen-ref.bin 4
cargo run -p beholdd-model --release --example block -- <model.gguf> /tmp/qwen-rust.bin 4
python3 tools/compare_block.py /tmp/qwen-ref.bin /tmp/qwen-rust.bin
```

**Gate M1.2 — PASSED.** The fixed completion prompt `The capital of France is` tokenizes
to `[785, 6722, 315, 9625, 374]` and greedily generates
`[12095, 13, 576, 6722]` — ` Paris. The capital` — on the same real Q8_0 checkpoint.
The checked path executes all 48 blocks, final RMSNorm and the language-model head,
then retains post-RoPE per-layer K/V caches for incremental decoding. Qwen2 byte-BPE
tokenization is loaded from GGUF metadata and checked against local llama.cpp vectors.

```sh
cargo run -p beholdd-model --release --example decode -- \
  <model.gguf> 4 "The capital of France is"
```

**Gate M1.3 — PASSED.** Two fresh release processes on commit `bf04468` evaluated the
same M1.2 prompt against the checkpoint above and emitted all four `[4, 151936]` `f32`
logit rows as versioned little-endian artifacts. Both 2,431,092-byte files have SHA-256
`efe7862a0891c286db3686fb18221fe2187b1821652d849caa4401a981290475`. The comparator
validates the prompt and generated IDs, finite logits, each lower-ID-stable argmax, exact
shape and envelope before requiring whole-file byte equality. The checkpoint's SHA-256
is the M1.1 value above.

```sh
cargo run -p beholdd-model --release --example decode -- \
  <model.gguf> 4 "The capital of France is" --logits-out /tmp/run-a.bin
cargo run -p beholdd-model --release --example decode -- \
  <model.gguf> 4 "The capital of France is" --logits-out /tmp/run-b.bin
python3 tools/compare_logits.py /tmp/run-a.bin /tmp/run-b.bin
```

**Dense Qwen3 extension — PASSED for native execution, not llama.cpp token parity.** The
explicit `QwenModel` dispatcher now keeps `qwen3` and `qwen3moe` binders separate. A
synthetic dense model executes forward and cached decode, while the local real
Qwen3-0.6B-Q8_0 checkpoint binds all 310 tensors and produces a stable coherent
eight-token decode. An independent f64 transcription agrees with beholdd's first-step
top three logits/order. llama.cpp's activation-requantized Q8_0 CPU path selects a
different first token; this is recorded as numerically different quantized execution,
not an architecture failure, and this extension makes no llama-identical token claim.
`qwen3next`, `qwen35`, and `qwen35moe` remain rejected. See
[`docs/research/native-qwen3-dense.md`](docs/research/native-qwen3-dense.md).

### Next native architecture — `nemotron_h_moe` — OPEN

NVIDIA Nemotron 3 Super 120B A12B is the next native architecture target. It is not a
Qwen alias and is not supported today: its 88-block GGUF interleaves 40 recurrent SSM
blocks, 8 full-attention blocks, and 40 latent-MoE blocks with 512 experts, top-22 routing,
and a separate shared expert. The strict dispatcher must continue to reject
`nemotron_h_moe` until the staged gates pass.

The parity fixture is the three-shard Q4_K_M conversion from
`lmstudio-community/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF`, pinned at revision
`16410500b9bdc62081c34cc2dff52471e001f970` and totaling `86,051,079,584` bytes. Its
current copy is on a read-only CIFS weights share; that is a source to stage and hash, not
a reason to bypass beholdd's unsafe-storage refusal. The implementation order is:

1. verify all three pinned shard hashes after staging to local solid-state storage;
2. add a strict, separate metadata/tensor binder with adversarial refusal tests;
3. gate SSM, sparse full-attention, and latent/shared MoE sublayers independently;
4. compose mixed-block prefill/decode with explicit SSM and KV-state ownership;
5. compare captured states, complete logits, and greedy tokens against a pinned oracle;
6. qualify resident/streaming execution and the existing embedded/HTTP/SSE surface.

No implemented-architecture list or model capability changes before those gates pass.
Exact Q4_K_M, historical Q8_0, upstream BF16, architecture, and license provenance is in
[`docs/research/native-nemotron-h-moe.md`](docs/research/native-nemotron-h-moe.md).

---

## M2 — Streaming: models larger than memory

- `beholdd-memory` — bounded weight-range caches, pinned-prefix for cyclic scans, LRU for
  demand, generic prefetch and direct-I/O experiments, and the `FitPolicy` contract from
  ADR-0001.
- Layer-major prefill: a layer's working set is fetched once, however many tokens want it.

The streaming research is model- and checkpoint-format-neutral. Qwen3-MoE is the
foundation because its conventional structure makes byte demand, routing and exactness
easy to inspect from the 30B resident oracle through the 235B run and the 400B-class
Qwen3-Coder-480B target. Kimi K3 is a later stress target for the same substrate, not a
second streaming engine. `beholdd-format` may name GGUF and Safetensors, and
`beholdd-model` may name architecture concepts; `beholdd-memory` must receive only checked
locators, byte ranges and reuse information.

**Gate M2.1 — PASSED.** A model several times larger than RAM generates coherent text.

**Gate M2.2 — PASSED.** Streamed output is **byte-identical** to resident output on a
model that fits both ways. `Exact` means exact; if it is not identical it is `Degrade`
and must say so.

**Gate M2.3 — PASSED.** The engine reports device, achieved GB/s, cache hit rate and
measured resident bytes, and refuses to start on a rotational or network-backed checkpoint
without an explicit override.

**Gate M2.4 — PASSED.** GGUF and Safetensors locators feed the same direct, fully
resident and streaming range-source contract. A shared test matrix proves identical bytes
through every source mode and refuses a locator from another checkpoint, a nonexistent
shard, reversed or overflowing ranges, out-of-bounds slices and truncated payloads before
returning bytes. The public `beholdd-memory` range API names no concrete checkpoint format.

**Gate M2.5 — PASSED.** A model can expose ordered future weight ranges and a barrier
after which additional data-dependent ranges become known. An instrumented fixture blocks
backend computation after one weight is delivered and proves that I/O for a known future
range begins before the foreground load asks for it. Execution remains byte-identical to
the synchronous path, never exceeds its cache or in-flight byte limits and never reads an
unselected demand range. The scheduler sees no layer, attention, router or expert
semantics. Adjacent ranges can be combined only through a checked, checkpoint- and
shard-bound physical span; coalescing never accepts an unchecked offset tuple or crosses
a shard boundary.

**Gate M2.6 — PASSED.** On a fixed, SHA-256-recorded Qwen 400B-class checkpoint, at
least three paired prefetch-off/prefetch-on runs use the same prompt, token count, device,
storage-cache state and hard memory limit. Tokens and complete logits are byte-identical;
prefetch-on has lower median decode time per token and stays within its declared cache and
in-flight budgets. The report includes demanded, useful and physically read bytes, read
amplification, byte-weighted cache hits, exposed weight-wait time, overlap, queue depth,
prefetch useful/late/wasted counts, achieved GB/s and peak process/cgroup residency.

> Status: M2's exact-byte source seam and preflight boundary are active. Cyclic weight
> reads fill a strictly bounded pinned prefix; embedding rows and routed experts use a
> separately bounded LRU; oversized reads bypass without evicting useful entries. Shared
> immutable handles make cache hits zero-copy, and a synthetic multi-step decode is
> bit-identical through direct and cached sources.
>
> `Exact` is the default fit policy. Checked resident/streaming forecasts include the
> requested workload's decode buffers, the CPU backend's dequantized-matrix scratch, an
> explicit runtime reserve and bounded cache capacity. The executable plan is typed, so
> a refusal or an unresolved degradation cannot fall through to a run. Linux availability
> is capped by every finite cgroup-v2 ancestor; cgroup-v1/hybrid memory control is refused
> rather than guessed. Decode reports device, achieved weight-read GB/s, cache hit rate and
> measured peak/current RSS. A rotational, known network-filesystem-backed or unknown
> supplied path is refused before GGUF parsing unless the caller passes the explicit
> unsafe-storage override; every shard discovered afterward is rechecked before tensor
> payload reads, including mixed-storage layouts.
>
> **Gate M2.6: PASSED; M2.1: PASSED** on `aa1a84b` (2026-08-18). Three alternating
> prefetch-off/prefetch-on pairs ran the real Qwen3-Coder-480B-A35B-Instruct UD-Q4_K_XL
> checkpoint (six GGUF shards, 275,547,594,752 encoded bytes; first shard SHA-256
> `ae0d41c4baa6871a3702e389be00dac564fd28ad407504ec8e2f46b7c4ee0a47`), staged on local
> ext4/NVMe. That is 2.73 times Linux's reported 101,002,686,464-byte `MemTotal`, closing
> M2.1's literal several-times-larger-than-physical-RAM wording left open since the M2.1/M2.3
> record below. `tools/analyze_streaming_benchmark.py` now requires decode's
> `--checkpoint-sha256` line to agree across every paired log before it will compare them.
>
> Each run used the fixed prompt `"The capital of France is"` (`[785, 6722, 315, 9625,
> 374]`) for 8 greedy steps under `Exact`, a 4 GiB pinned / 8 GiB demand / 4 GiB prefetch
> (4 queue depth) weight-cache trunk when prefetch was enabled, a 25,769,803,776-byte
> (24 GiB) planner budget, and a page cache dropped and mincore-verified cold
> (`tools/advise_cold_cache.py`) before every run. Execution was confined to 8 physical
> cores (`AllowedCPUs=0-15` on this SMT2 chip) inside a `systemd-run --user --scope` cgroup
> (`MemoryMax=32G`, `MemoryHigh=28G`, `MemorySwapMax=0`), CPU boost disabled and frequency
> capped to 50% (1.5 GHz). An initial uncapped, full-boost attempt was aborted after the
> die's `Tctl` sensor went from a ~68C idle baseline to 97.5C in about two seconds -- this
> APU's NPU stays engaged on Linux regardless of workload and shares one thermal envelope
> with the CPU and GPU, so an all-CPU decode alone can drive it near its 100C max. The new
> `tools/thermal_guard.py` now supervises every run independently of any box-wide policy:
> it polls `Tctl` only (the discrete GPU-edge sensor reads low and is misleading on this
> die), hard-aborts the supervised process at 90C, and treats an unreadable sensor as a
> critical breach rather than assuming a safe temperature. All six runs stayed well under
> that line (49.6-73.4C peak).
>
> All six runs generated identical tokens `[12095, 13, 3555, 374, 279, 6722, 315, 9856]`
> (`" Paris. What is the capital of Germany"`) and wrote byte-identical `[8, 151936]` `f32`
> logit artifacts -- one shared SHA-256, `5c2949744c27f7e028393bfb64c51f20fe78215a1986ab561bbc6ada9cfe5693`,
> independently checked by `tools/compare_logits.py`. Median decode time was 175.484
> s/generated-token prefetch-off versus 160.393 s/generated-token prefetch-on, an 8.60%
> reduction (15.091 s/token); byte-weighted cache hit rate rose from 23.94% to 30.33% and
> request hit rate from 23.33% to 31.44%, driven by 2,090 useful prefetches (971 late, 0
> wasted) hiding 9.545s behind compute against a 4,070,346,300-byte authorized prefetch
> budget. Read amplification held at 0.761x via range coalescing on both sides. Peak cgroup
> memory was ~30.07 GiB against the 32 GiB hard cap on every run, and
> `tools/analyze_streaming_benchmark.py` independently validated shared checkpoint identity,
> matching prompt and generated IDs, and cgroup-limit compliance before reporting the
> medians above.
>
> The generic streaming substrate this measured is model-agnostic. Kimi K3 (2.8T
> parameters, 1.56 TB official checkpoint) remains the adversarial portability target in
> `docs/research/kimi-k3-streaming.md`, not a separate implementation -- and this 480B
> result is the baseline that program now pushes from.
>
> **Gate M2.4: PASSED** on `5c9fc4e` (2026-08-14). `beholdd-format` now mints opaque,
> checkpoint-bound `WeightLocator` and checked `WeightRange` values for both GGUF and
> Safetensors behind one concurrent positional `RangeReader`. `beholdd-memory` consumes
> only that format-neutral contract for direct, fully resident and streaming sources.
> The shared matrix checks both formats through all three sources, full and partial
> ranges, and both reuse classes. It also proves foreign, reversed, overflowing,
> out-of-bounds, missing-shard and post-open-truncation refusals without cache
> publication or weight-sized allocation before validation. Safetensors additionally
> rejects overlapping, gapped or trailing payload layouts while preserving valid empty
> tensor aliases. The gate passed 36 format tests, 44 memory tests, the complete
> 169-test workspace run (one real-checkpoint tokenizer probe ignored), doctests and
> warning-denied Clippy.
>
> **Gate M2.5: PASSED** on `cd8f577` (2026-08-14), building on the checked
> `CheckpointSpan` primitive from `e71f82e`. `beholdd-memory` runs one bounded,
> format-neutral prefetch worker: reading plus ready bytes never exceed the configured
> prefetch budget, attached jobs never exceed queue depth, and an uninterruptible
> cancelled read remains byte-charged and explicitly counted as at most one detached
> record. Foreground requests share active reads without duplicate I/O; phase rejection,
> cancellation and failures publish no partial bytes. Qwen exposes embedding, fixed-layer,
> routed-expert and output phases, revealing routed ranges only after the router barrier.
> A synchronized integration fixture holds CPU computation after the first embedding is
> delivered and proves the next embedding's physical read completes before its foreground
> request, then matches every output float bit-for-bit with prefetch disabled. The gate
> passed 42 format tests, 52 memory tests and the full 186-test workspace run (one
> real-checkpoint tokenizer probe ignored), doctests and warning-denied Clippy.
>
> **Gate M2.3: PASSED; M2.1: cgroup-qualified evidence, literal gate still open** on
> `3dfb07f` (2026-08-14). The four SHA-256-verified Qwen3-235B-A22B Q5_K_M shards total
> 166,814,157,632 bytes and were staged on local NVMe/ext4. A fresh metadata-only bind
> checked all 94 layers and exactly 1,131 supported F32/Q5_K/Q6_K tensors in 0.13 s at
> 28,292 KiB peak RSS, without reading tensor payloads.
>
> The real decode ran inside a separate hard 12 GiB cgroup (`MemoryHigh=10G`, swap disabled,
> four-CPU quota, low scheduler priority) with an explicit 8 GiB exact-planner budget and
> zero engine weight-cache. The 166,808,152,064 encoded weight bytes were 12.9 times the
> hard 12 GiB cgroup cap and 1.65 times Linux's reported 101,002,678,272-byte `MemTotal`.
> This proves streaming under the `Exact` policy with an enforced execution limit several
> times smaller than the weights, but does not satisfy M2.1's literal
> several-times-larger-than-physical-RAM wording. For prompt tokens `[785, 12884, 374]`
> (`"The sky is"`), the planner forecast 7,300,967,532 bytes and greedy decode generated
> `[2797, 11]`, or `" clear,"`: a coherent two-token completion fragment streamed under the
> `Exact` policy.
>
> The two-token run completed in 159.85 s on CPU. It read 50,925,403,136 bytes in 10,724
> misses over 24.166 s (2.107 GB/s), with the requested zero engine weight-cache reporting
> 0/10,724 hits, 0 retained bytes and 0 evictions. Process RSS was 90,419,200 bytes at exit and
> 3,088,556,032 bytes measured peak; the service cgroup peaked at about 10.0 GiB including
> charged file cache, stayed below its 12 GiB hard cap and used zero swap. The concurrent
> unrelated workload remained alive, and the post-run kernel log contained no OOM event. The same
> preflight had previously refused the CIFS checkpoint path without an unsafe-storage
> override; this run accepted the local solid-state copy without an override.
>
> **Gate M2.2: PASSED** on `a16544a` (2026-08-14). The SHA-256-verified
> Qwen3-Coder-30B-A3B-Instruct Q8_0 checkpoint contains 32,477,962,240 encoded weight
> bytes and has SHA-256
> `a4a0207f4653bfece73d9818c83acf714f5593525fe3aab7026347fd73090fcc`. Two fresh
> release processes on local solid-state storage evaluated prompt token `[64]` (`"a"`)
> for one greedy step under the default `Exact` policy. Both generated token `[25]`.
>
> The streaming process used an 8,000,000,000-byte planner budget inside a hard 12 GiB
> no-swap cgroup. Its exact forecast was 5,872,605,392 bytes with zero weight-cache; it
> performed 1,587 tensor-range reads totaling 3,269,568,640 bytes and measured
> 1,639,964,672 bytes peak process RSS. The resident process used a 40,000,000,000-byte
> planner budget inside a hard 48 GiB no-swap cgroup. Its forecast was 38,019,954,896
> bytes; it preloaded all 32,477,962,240 encoded bytes once into the packed resident
> source, then served every inference request from zero-copy views. It measured
> 33,783,353,344 bytes peak process RSS; systemd reported a 33.4 GiB service peak and
> zero swap.
>
> Each process wrote one complete `[1, 151936]` `f32` logit row plus its exact prompt and
> generated IDs. Both 607,832-byte artifacts have SHA-256
> `3eb01ada9ca55d20376c42bc4cad8381fa5fd412a1c497e4df4af173363b3f16`.
> `tools/compare_logits.py` independently validated their envelopes, finite values and
> stable argmax before requiring whole-file byte equality.
>
> Core commands are below; the recorded run wrapped the first two in the hard cgroups
> described above.
>
> ```sh
> target/release/examples/decode <model.gguf> 1 a \
>   --weight-mode streaming --fit-policy exact \
>   --memory-budget-bytes 8000000000 --runtime-headroom-bytes 4294967296 \
>   --logits-out /tmp/stream.bin
> target/release/examples/decode <model.gguf> 1 a \
>   --weight-mode resident --fit-policy exact \
>   --memory-budget-bytes 40000000000 --runtime-headroom-bytes 4294967296 \
>   --logits-out /tmp/resident.bin
> python3 tools/compare_logits.py /tmp/stream.bin /tmp/resident.bin
> ```
>
> The later 480B Gate M2.6 record above supersedes the older open note that used to be in
> this position: M2.1 and the bounded phase-aware prefetch gate are complete. Further I/O
> policies remain experiments, not missing M2 gates.

---

## M3 — Serving

- `beholdd-runtime` — `.beholdd` layout, YAML config, model registry, request scheduling,
  and the threading policy that consumes the kernel's granularity reports.
- `beholdd-server` — OpenAI-compatible HTTP, SSE with keepalive, tool-call translation.
- `beholdd-embed` — the in-process API, built at the same time as the server and from the
  same runtime, so it cannot drift into a second engine.

**Gate M3.1 — PASSED.** beholdd streams a real answer over its native SSE route, and the
product client talks to beholdd through its own end-to-end path.
**Gate M3.2 — PASSED.** A tool call round-trips: schema in, native format out, OpenAI
`tool_calls` back, client execution, result replay, and a final assistant response. The
previous engine got this wrong in a way that produced fluent prose instead of a tool call,
with no error anywhere.
**Gate M3.3 — PASSED.** `loaded` is answered by probing. Kill a worker and the registry
says so within a second.
**Gate M3.4 — PASSED.** `beholdd-embed` runs the same prompt in-process with no port
open, and gets the same tokens as the server.

> **M3 validation status (2026-08-25): PASSED for the documented local integration.**
>
> - **M3.1: PASSED.** The native SSE gate separately streamed a real Qwen3-Coder response
>   with an assistant-role delta, UTF-8 content, terminal finish reason, keepalives, and
>   `[DONE]`. Against pushed beholdd code HEAD `c75a9a8`, a disposable application client at
>   pushed commit `a737bfd` then started the agent at `2026-08-25T20:24:02.687Z` and
>   completed the live tool turn and final replay using non-streaming
>   `provider.complete()`; application client exposed the agent's own event stream over SSE. The
>   beholdd request crossed the former 300-second Undici response-header ceiling because
>   of the application client timeout fix.
> - **M3.2: PASSED.** The resident `qwen3-coder-30b` Q8_0 model requested exactly one
>   `write_file` call: `path` was `smoke.txt` and `content` was
>   `"beholdd v2 timeout fixed\n"`. application client wrote 25 bytes, replayed the result,
>   received the terminal assistant response `Done.`, and reported `smoke.txt` in
>   `node_done` artifacts. The build finished with `built=1`, `reused=0`, `failed=0`,
>   `blocked=0`, and `ms=1622970` (about 27m03s). Its ledger artifact has SHA-256
>   `8426eb07296ed1606455a7c7fad9a1f3bd0e6500249641c067a705f546f0b6ea` and records
>   `model=qwen3-coder-30b`, `agent=general`, no pins, and `stale=false`.
>   With port 6768 confirmed down, an immediate rerun finished in 2 ms with `built=0`,
>   `reused=1`, and `failed=0`; it made no model call and left the artifact and ledger
>   hashes, inode, and mtime unchanged.
> - **M3.3: PASSED.** A feature-gated test publishes a real streaming source through
>   `NativeRuntime`. The literal feature-only death test terminates its actual prefetch
>   worker without setting normal source shutdown and observes `loaded: false` through
>   the public registry within one second. It performs zero weight reads and exposes no
>   production fault endpoint.
> - **M3.4: PASSED.** A guarded release test ran the same one-token prompt through one
>   shared `NativeRuntime` in process and through the HTTP router. Both returned token ID
>   `9707` (`Hello`) with identical usage and finish metadata. The run stayed below its
>   48 GiB no-swap cgroup and thermal limits.
>
> The client gate used the SHA-256-documented, 32,483,934,528-byte checkpoint resident on
> local NVMe. Across 1,800 one-second thermal samples, temperature was 55.875–67.750C:
> zero samples reached the 75C warning threshold and no sensor read failed. Cgroup memory
> peaked at 40,010,301,440 bytes (37.26 GiB), below `MemoryHigh=44GiB` and
> `MemoryMax=48GiB`; swap and zswap use and the high, max, and OOM counters all
> remained zero. The legacy service on port 6767 was untouched.
>
> Immediately after the successful inference/build result, the guarded service reached
> its configured 30-minute `RuntimeMax` and exited with timeout status 143. This is not a
> clean service-lifecycle pass. The bounded two-completion smoke barely fits 30 minutes
> at the measured CPU speed, and a worst-case default 24-round workflow is not viable.
> The gate covers a loopback-local server, the narrow documented schema subset, and the
> 30B checkpoint as the only currently validated native-serving checkpoint; it does not
> claim broader checkpoint validation or production throughput.
>
> Supporting verification belongs to the previously verified code commit `c75a9a8`, not
> this documentation edit: the no-model suite passed 255 tests with 2 ignored, the
> warning-denied Clippy, rustfmt, and rustdoc checks passed, and all 45 Python tests
> passed.

### Native multimodal follow-up — OPEN

The model-list contract now reports explicit, binding-derived per-model compatibility and
capabilities. A lazy configured path is `unverified` with no capabilities; only a model
successfully bound and published by the runtime reports `"completion"`. The current
capability type is deliberately completion-only. OpenAI text content parts are normalized
before prompt rendering; `image_url` is recognized and rejected with a stable unsupported-
content error. This is protocol groundwork, not native vision support.

Removing Antimatter's working llama.cpp vision path requires native `qwen35moe` text
parity, its hybrid Gated DeltaNet/full-attention execution, the separate 27-layer vision
encoder and Qwen3-VL merger, matching preprocessing and multimodal RoPE/injection,
structured media requests, two-file fit accounting, bounded HTTP image inputs, and
end-to-end parity/product gates. The exact prerequisites and decision line are in the
[native Qwen3.6 vision replacement plan](docs/research/native-qwen36-vision.md). Until
those gates pass, beholdd must not advertise `"vision"` and the compatibility path must
remain available.

---

## M4 — Vulkan

The first accelerated backend, and the second implementation of `Device` — which is what
proves the trait is a seam rather than a guess.

**Gate M4.1** — Vulkan produces the same tokens as the CPU backend on the same prompt,
within the stated tolerance.
**Gate M4.2** — it is faster than the CPU path on this box, measured, with the device
named.
**Gate M4.3** — no model or runtime code changed to add it. If any did, the seam moves.

> **M4 bring-up status (2026-09-07): in progress, no gate claimed.** The first native
> packed-weight prototype now retains prepared Q8_0 and BF16 matrices in Vulkan-visible
> memory, reuses its pipeline/descriptor/command/fence resources, and matches the CPU
> oracle within 1.20e-6 relative L2 or better on the tested Qwen3.6-shaped matrices. On
> the local Radeon 8060S, batched-sample synchronous large BF16 matvecs were about
> 2.2-3.1x faster than the faster of the already-fused CPU measurements bracketing the
> GPU phase in a guarded run. It is intentionally not wired into `Device`
> or model execution until the interface represents persistent weights, activation
> residency and recorded batches instead of hiding a host upload per borrowed slice.
> That interface now exists alongside `Device` as the object-safe `ExecutionDevice`, with
> an owner-checked sequential CPU interpreter as its oracle. Vulkan now has a mandatory-
> budget immutable weight arena (24 selected BF16 expert projections share one allocation),
> device-resident routed-expert scratch, and a deterministic gate/up/SwiGLU/down reduction
> recorded as three dispatches in one submission. At the Qwen3.6 routed shape
> H=2048/F=512/K=8, the first guarded comparison measured 2.137 ms versus 7.293 ms for
> the former 24-submit path (3.41x), with 2.01e-7 CPU-relative L2 and bit-exact repeated
> GPU output. A game began using the same APU before that run and Tctl reached 87.375 C,
> so the ratio is useful design evidence but the absolute time is explicitly not an idle-
> host qualification. A subsequent idle-host qualification added host-validated paired
> BF16 loads and direct two-dimensional dispatch. The paired path passed CPU, guarded,
> overwrite, accumulation, fallback, and repeatability gates, then won three final-binary
> runs by 1.08-1.21x over guarded (0.557-0.569 ms versus 0.613-0.675 ms). The median-of-
> run medians was 0.560 ms, 1.14x faster than guarded and 4.56x faster than the old
> 24-submit path; peak Tctl was 57.875 C. A subsequent six-way fixed-tree/subgroup
> tournament found that 128-lane reductions cut the paired path again. The explicit
> fixed tree beat the 64-lane control by 1.198-1.236x in three qualification runs and
> was promoted for this exact shape on RADV GFX1151/Mesa 25.2.8; three final production
> runs measured 0.435-0.464 ms
> (0.440 ms median), 1.47x over guarded and 5.61x over the old path by median-of-run
> medians. CPU relative L2 remained 2.789e-7 and output remained bit-repeatable. The
> 64-lane implementation stays the device-limit/platform fallback. An explicit fixed-N
> `RoutedMoeProgram` now pins private mapped activations/metadata, immutable per-item
> descriptors and scratch, and reusable overwrite/accumulate command buffers. For
> independent work it records N routed-MoE operator evaluations into one submission and
> fence. On the same qualified RADV GFX1151/Mesa 25.2.8 H=2048/F=512/K=8/Tree128 path,
> all three clean binary runs cleared the 1.10x gate for N=2/4/8/16; median paired
> speedups were 1.335x, 1.859x, 2.154x, and 2.176x, and both a later post-qualification
> run and the exact manifested-source release run passed the asserted gate. N=1's
> three-run median was only 1.055x, so it remains NO-GO and is not the default. All
> program results matched independent production calls bit for bit, all four ignored
> live Vulkan tests passed, and qualification peaked at 61.500 C under the 75/90 C
> guard. These are routed-MoE operator evaluations/s for parallel independent items,
> not individual expert evaluations/s or autoregressive tokens/s.
> Model-scoped fit accounting, direct checkpoint identity/loading,
> complete command coverage, real-checkpoint token parity, and end-to-end throughput are
> still required before any M4 gate can pass.
> Design and commands: `docs/research/native-acceleration.md`; fixed-N qualification:
> `docs/research/native-routed-moe-program-run-2026-09-07.md`.
>
> **Native-BF16 research update (2026-09-08): promoted to trace replay, not runtime.** A
> standalone HIP/rocWMMA route produced native gfx1151 BF16 WMMA and beat the frozen Vulkan
> routed-operator control by `2.224x–6.957x` for `N=2–16`; `N=1` reached `1.409x`
> and missed its `1.50x` gate. It uses ideal eight-expert overlap and two BF16
> activation boundaries, so it is not an end-to-end token result. Gate/up/SiLU
> fusion and HIP Graph replay were negative. A final down-WMMA/reduction fusion
> remained bit-identical over four adversarial route families but delivered only
> `1.024x–1.031x` at the four median-of-three decision cells and failed every
> aggregate threshold. The real-weight resident-layer target is now complete as a
> research gate: layer-3 RMSNorm, device router/stable top-k, shared+routed experts,
> residual, and state remain resident. A one-time GPU transpose preserves every
> expert BF16 value while preparing rocWMMA's efficient Matrix-B layout; it measured
> `2.9455x` over checkpoint layout. A deterministic parallel router added a
> `1.236879x` median-of-three incremental gain. The combined path reached
> `0.378158 ms` per fixed-route MoE-FFN sublayer with exact routes,
> bit-deterministic repeats, and `1.205e-7` CPU-relative L2. Forty FFNs imply a
> 66.11 tokens/s FFN-only ceiling, not full-model throughput. A subsequent
> route-aware `B=16` real-weight cohort retained exact routes and caller-order
> reduction while grouping requests by expert. It measured `1.892540x` on a
> favorable correlated corpus and a `1.360078x` median paired speedup across
> three decorrelated hashed corpora (99 active experts at the median), or
> 3,431.34 independent single-layer FFN evaluations/s. All three independent
> runs passed the prospective numerical-equivalence gate, although complete
> outputs are not bit-exact. This promotes the mechanism only to replay of
> captured real activations across prompts, positions, and layers; production
> remains NO-GO until full logits/token parity, bounded cohort delay, and the
> runtime ownership/admission seam pass. Records:
> `docs/research/native-bf16-wmma-run-2026-09-08.md`,
> `tools/hip/qwen35-resident-layer-results-2026-09-08.md`, and
> `tools/hip/qwen35-route-cohort-results-2026-09-08.md`.

---

## M5 — HAG

Our own AMDGPU DRM path, developed alongside: explicit residency, our own allocator, no
driver heuristics deciding what lives on the device. Sequenced after Vulkan so it arrives
into a proven seam.

**Gate M5.1** — same tokens as CPU and Vulkan.
**Gate M5.2** — a residency decision the Vulkan path cannot express, demonstrated and
measured.

---

## M6 — The instrument

Where the engine becomes research equipment: a novel architecture defined at level 1 or 2
running at full speed, with the same profiler and the same fit policy as anything shipped
in-tree.

**Gate M6.1** — a non-transformer defined in `.beholdd/archs/` runs, and is
indistinguishable in tooling from a built-in.
**Gate M6.2** — an independently defined experimental model runs on beholdd rather than on a fork.

---

## Deliberately not scheduled

**Training.** Inference only.
**Out-of-tree dynamic architecture loading.** A plausible level 4; large surface
(versioning, ABI, unsafe boundaries); levels 1–3 cover what we need.
**Distributed inference.** The RPC work exists elsewhere and is a different problem.
Single node until single node is excellent.
