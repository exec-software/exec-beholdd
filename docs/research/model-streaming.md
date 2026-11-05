# Model-agnostic streaming research program

Research contract: 2026-08-14. This is the canonical plan for making exact weight
streaming fast. It applies to every architecture and checkpoint format beholdd can run.
The [Kimi K3 profile](kimi-k3-streaming.md) supplies one demanding target, not a separate
implementation.

## Contract and boundaries

Qwen3-MoE is the development foundation. Its conventional GQA, RMSNorm, RoPE, SwiGLU and
top-k routing make the arithmetic and the requested weight bytes easy to inspect:

- Qwen3-Coder-30B fits resident and streaming execution, so it is the byte-exact oracle.
- Qwen3-235B already proves exact decoding under a hard memory limit much smaller than its
  encoded weights.
- Qwen3-Coder-480B is the 400B-class performance target for future prefetch, retention,
  coalescing and I/O tuning.

Kimi K3 is an adversarial portability target. Its official Safetensors layout, hybrid
recurrent and latent request state, compressed-tensors MXFP4 encoding and different model
operators must use the same storage, scheduling, fit and telemetry contracts. An
optimization learned on Qwen is not complete if adding K3 requires a Kimi-only streaming
source or scheduler.

The ownership boundary is strict:

- `beholdd-format` must parse a file format and mint checked, checkpoint-bound byte
  locators. It may know GGUF or Safetensors, but not what an attention layer or expert
  means.
- `beholdd-model` must bind locators to architecture semantics and expose what ranges will
  be needed. It may know layers, routers and experts, but not how storage retains or
  fetches bytes.
- `beholdd-memory` must accept opaque locators, ranges, reuse classes and ordering
  barriers. It chooses residency, caching, coalescing and prefetch without naming a
  checkpoint format or architecture.
- `beholdd-backend` and `beholdd-kernel` must execute arithmetic without deciding where
  the model's next bytes come from.

Dense models are the zero-barrier case, conventional MoEs reveal selected ranges after a
router barrier, and hybrid architectures may expose other phases. The range scheduler
does not acquire those meanings.

## What “near full speed” means

Streaming speed must be compared with a stated lower bound, not a model name. For one
decode step, the useful-byte I/O floor is the bytes the exact execution actually demands
divided by sustained bandwidth on the named storage device. The arithmetic floor is the
same work with weight stalls removed. Perfect overlap cannot beat the larger of those two
floors.

The research loop therefore measures both floors, the observed wall time and where the
difference went. It does not pick a universal efficiency percentage before the baseline
exists. Each optimization must improve a preregistered metric on Qwen while preserving
the exact artifact; K3 later measures the same quantities.

Candidate mechanisms remain experiments until they win that comparison:

- bounded lookahead and concurrent positional reads;
- single-flight misses so overlapping consumers do not fetch the same bytes twice;
- containment-aware range hits and safe coalescing of adjacent requested ranges;
- retention selected from measured reuse rather than an architecture name;
- buffered I/O versus direct I/O under the same cache state;
- layer-major prefill and overlap between I/O and backend arithmetic.

## Implementation gates

These gates are mirrored in `ROADMAP.md`. None is passed merely because its types or code
exist.

### M2.4 — Format-neutral exact ranges — passed

GGUF and Safetensors locators must feed the same direct, fully resident and streaming
range-source contract. One shared fixture matrix must prove identical returned bytes and
refuse all of the following before returning data:

- a locator minted by another opened checkpoint;
- a nonexistent shard;
- reversed, overflowing or tensor-out-of-bounds ranges;
- a range outside the actual shard after truncation.

The public `beholdd-memory` range API must contain no `Gguf`, `GgufTensor`, `SafeTensors`
or Safetensors tensor-info type. Format-specific tensor descriptors may retain dtype and
shape while exposing a common opaque locator to memory.

This gate passed on `5c9fc4e` (2026-08-14). A shared fixture matrix exercises both
formats through direct, fully resident and streaming sources, both reuse classes and
complete/partial ranges. Common checked construction and format-reader tests cover every
listed refusal, while post-open truncation tests prove that streaming publishes no
partial cache entry and fully resident loading remains transactional. Validation occurs
before a weight-sized allocation or cache mutation.

### M2.5 — Bounded phase-aware prefetch — passed

An architecture must be able to expose an ordered set of future range requests, then a
barrier after which more data-dependent requests become known. The contract describes
ordering and reuse, not MoE routing.

Before coalescing is enabled, the format contract must be able to combine adjacent
`WeightRange` values into an opaque physical span only when they belong to the same opened
checkpoint and shard. A span cannot cross a gap, overlap, overflow or cross a shard
boundary. The scheduler must never fall back to a raw `(shard, offset, length)` tuple.

An instrumented delayed reader must prove that the implementation:

- begins I/O for a known future range while backend computation using an earlier weight is
  deliberately blocked, before the foreground load asks for that future range;
- serves the later foreground load from the completed prefetch without a second physical
  read;
- produces the same bytes and final artifact as synchronous execution;
- never exceeds declared cache or in-flight byte limits;
- fetches each single-flight range at most once;
- never fetches a demand range that was not revealed after its barrier;
- propagates read failure or cancellation without publishing partial bytes.

The overlap proof uses synchronization events around backend computation, prefetch I/O and
the foreground request rather than a timing-only assertion, so it is deterministic and
does not require concurrent reads.

This gate passed on `cd8f577` (2026-08-14), building on the checked cross-locator
`CheckpointSpan` from `e71f82e`. The format-neutral source owns one bounded worker and
single-flights foreground callers with queued or active prefetches. Reading plus ready
bytes are capped exactly; queue depth caps attached current-generation jobs; cancellation
may leave at most one uninterruptible detached read, whose bytes remain charged and whose
record is exposed in telemetry. Same-key generation replacement, transactional phase
validation, failed reads, cancellation and concurrent foreground attachment are covered
by synchronized fixtures.

Qwen emits ordered embedding, fixed-layer, selected-expert and output phases. The expert
phase is submitted only after stable routing and contains no unselected range. In the
integration proof, the real CPU backend blocks after receiving the first embedding while
the real streaming worker completes the second embedding read; the foreground has made
only one request at that point. Releasing the backend produces a single physical read for
the future range and a block output whose floating-point bits exactly match prefetch-off.

### M2.6 — Qwen performance foundation

A fixed Qwen 400B-class checkpoint and prompt become a paired benchmark. At least three
alternating prefetch-off/prefetch-on trials must hold the checkpoint hash, generated-token
count, backend, CPU policy, storage-cache state and hard memory limit constant.

The gate passes only when complete logits and token IDs remain byte-identical, every
declared memory bound holds and prefetch-on has lower median decode time per token. The
result records enough data to explain the change rather than merely quote it:

- demanded, useful and physically read weight bytes per token;
- read amplification and byte-weighted cache hit rate;
- achieved storage GB/s and exposed weight-wait time;
- arithmetic/I/O overlap and in-flight queue depth and bytes;
- useful, late and wasted prefetches;
- process RSS plus cgroup current and peak memory;
- wall time and time per generated token.

Gate M2.6 passed on 2026-08-18 with the 275.55 GB Qwen3-Coder-480B checkpoint. Three
paired, cold-cache trials reduced median decode time from 175.484 to 160.393 seconds per
generated token (8.60%) while complete logits and tokens remained byte-identical. The
full checkpoint hash, commands, cache/I/O telemetry, cgroup bounds and thermal record are
preserved in `ROADMAP.md`; that result supersedes this document's pre-run wording.

## Work after the generic streaming substrate

Some K3 requirements are model enablement rather than streaming policy. They should still
produce reusable interfaces where the requirement is genuinely shared:

1. Decode state needs checked snapshot, restore and rollback semantics for append-only,
   fixed-size and mixed state. Qwen KV state plus a synthetic fixed-and-append-only fixture
   prove the interface before K3 uses it.
2. Storage-encoding dispatch must represent packed values and scale tensors without a
   Kimi-specific numeric shortcut. Existing independent MXFP4 scalar/vector checks remain
   the arithmetic oracle.
3. K3 then adds KDA, Gated MLA, Block AttnRes, Stable LatentMoE/SiTU and later MoonViT-V2
   inside the model and kernel ownership boundaries.

Adding those semantics must not require edits to the format-neutral range source,
scheduler, fit policy or telemetry established by M2.4 through M2.6.
