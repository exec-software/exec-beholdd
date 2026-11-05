# Qwen3.6 real-activation trace v1

Status: design record. The extractor and replay reader do not exist yet.

This freezes the next experiment before real activations are observed. It
defines an independent, offline Transformers oracle for the existing layer-3
route-cohort benchmark. It does not authorize any change to golden reference deployment or
application service, and it does not promote the cohort path into the serving runtime.

## First contained slice

The first trace is one text-only `B=16` cohort at zero-based decoder layer
`L=3`. It uses one left-padded batch of 16 fixed, public coding prompts and
selects the last non-padding prefill row from each prompt. Prompt/token
coordinates are frozen before routes are inspected; samples must not be chosen
for high expert overlap.

Layer 3 is a full-attention layer in this checkpoint. Its pre-FFN state has
already passed through the embedding, three Gated DeltaNet layers, and layer-3
attention. These are real upstream activations, but they are prefill endpoints,
not cached single-token decode. The distinction is embedded in the trace and
must appear in every result.

The replay target is the native-BF16 layer pack with SHA-256
`9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116`.
Any measured rate is independent single-layer MoE-FFN evaluations/s, never
single-request latency, full-model throughput, or output tokens/s.

## Frozen oracle identity

The immutable local checkpoint is:

```text
$HF_HUB_CACHE/
  models--Qwen--Qwen3.6-35B-A3B/snapshots/
  995ad96eacd98c81ed38be0c5b274b04031597b0
```

- Repository/revision: `Qwen/Qwen3.6-35B-A3B` at
  `995ad96eacd98c81ed38be0c5b274b04031597b0`.
- `config.json` SHA-256:
  `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.
- `model.safetensors.index.json` SHA-256:
  `41b9356101ebf8e7519e150dc811f80c4226e727301fbb032b890f006ed0be83`.
- Indexed tensor payload: `71,903,645,408` bytes in 26 shards (about
  66.97 GiB).
- Transformers implementation:
  `$TRANSFORMERS_SOURCE/models/qwen3_5_moe/modeling_qwen3_5_moe.py`,
  SHA-256
  `efc5f0085cc7a7b0f334d864f2ca1f0a257e6393e257393e329f160f6cc3f97a`.
- Executed software: Transformers `5.8.0`, PyTorch
  `2.10.0+rocm7.12.0`, Accelerate `1.13.0`, and Safetensors `0.7.0`.

The checkpoint config's recorded Transformers version (`4.57.1`) remains
checkpoint metadata; it does not replace the actual capture version above.

## `BEHOLDDQ35TRACE` v1 container

There is one immutable file per layer. `row_count` is a positive multiple of
16, and each consecutive group of 16 rows is a cohort. All integers and tensor
elements are little-endian. Every section begins at a 4096-byte boundary; all
padding and reserved bytes are zero. Compression is not permitted in v1.

### Header

The header is exactly 4096 bytes:

| Offset | Bytes | Field |
|---:|---:|---|
| 0 | 16 | ASCII `BEHOLDDQ35TRACE` plus one NUL |
| 16 | 4 | format version, `1` |
| 20 | 4 | header size, `4096` |
| 24 | 4 | endian marker, `0x01020304` |
| 28 | 4 | flags: bit 0 text-only, bit 1 prefill, bit 2 exact-BF16-widen |
| 32 | 4 | zero-based layer index |
| 36 | 4 | hidden width, `2048` |
| 40 | 4 | routed experts, `256` |
| 44 | 4 | experts per row, `8` |
| 48 | 4 | row count |
| 52 | 4 | cohort width, `16` |
| 56 | 4 | capture-point enum, `1` = post-attention residual before RMSNorm |
| 60 | 4 | scalar enum, `1` = little-endian F32 exact BF16 widening |
| 64 | 4 | route enum, `1` = F32 softmax, top-k, renormalize, BF16 cast |
| 68 | 4 | section count, `9` |
| 72 | 4 | section-descriptor size, `128` |
| 76 | 4 | reserved; zero |
| 80 | 8 | exact file size |
| 88 | 8 | section-table offset, `512` |
| 96 | 32 | layer-pack SHA-256 |
| 128 | 32 | full checkpoint content-root SHA-256 |
| 160 | 32 | config SHA-256 |
| 192 | 32 | Safetensors index SHA-256 |
| 224 | 32 | extractor source SHA-256 |
| 256 | 32 | environment-manifest SHA-256 |
| 288 | 32 | prompt-token-manifest SHA-256 |
| 320 | 40 | ASCII snapshot revision, no terminator |
| 360 | 152 | reserved; zero |
| 512 | 1152 | nine section descriptors |
| 1664 | 2432 | reserved; zero |

Each 128-byte descriptor is: `name[32]`, `u32 kind`, `u32 dtype`, `u32 rank`,
`u32 flags`, `u64 shape[4]`, `u64 offset`, `u64 length`, and the binary
SHA-256 of the section payload. Names are NUL-padded ASCII and unique. Rank is
at most four and unused dimensions are zero. Shapes are row-major C order.
Section flag bit 0 means required; all nine v1 sections set it.

Kinds are `0=manifest`, `1=record metadata`, `2=activation`, `3=routing`, and
`4=oracle output`. Dtypes are `0=bytes`, `1=record32`, `2=u32`, and `3=f32`.
An unknown enum, header/section/record flag bit, required section, or nonzero
reserved value invalidates the file, as does a duplicate name, misalignment,
overlap, out-of-bounds range, length/shape disagreement, or digest mismatch.

### Sections

| Name | kind/dtype and shape | Meaning |
|---|---|---|
| `manifest.json` | manifest/bytes `[N]` | canonical provenance and token manifest |
| `records.bin` | record metadata/record32 `[rows]` | prompt and selected-token coordinates |
| `pre_norm.f32` | activation/f32 `[rows,2048]` | post-attention residual entering RMSNorm |
| `normalized.f32` | activation/f32 `[rows,2048]` | RMSNorm output and MoE input |
| `router_logits.f32` | routing/f32 `[rows,256]` | all raw router logits |
| `route_ids.u32` | routing/u32 `[rows,8]` | top-8 IDs in Transformers rank order |
| `route_weights.f32` | routing/f32 `[rows,8]` | normalized returned top-8 weights |
| `shared_logits.f32` | routing/f32 `[rows]` | shared gate output before sigmoid |
| `post_ffn.f32` | oracle output/f32 `[rows,2048]` | decoder output after MoE and residual |

Every F32 tensor value is an exact widening of an observed BF16 value. Its low
16 storage bits must therefore be zero. The writer rejects non-finite values
and never recomputes, rounds, or otherwise changes a captured value while
widening it. This representation feeds the current F32-input harness without
losing the source BF16 bits.

Each 32-byte record is `u64 prompt_id`, `u64 invocation_step`, `u32 token_id`,
`u32 absolute_position`, `u32 flags`, and a zero `u32 reserved`. Record flag
bit 0 means prefill, bit 1 means last non-padding token, and bit 2 means
text-only. `invocation_step` is zero for the first prefill invocation. The
canonical JSON manifest maps `prompt_id` to the full prompt digest and token
sequence; token IDs can reconstruct text, so the artifact is sensitive even
when raw strings are omitted.

The manifest follows RFC 8785 JSON canonicalization and records the repository,
revision, source paths/digests, shard identities, software and optional-kernel
versions, device/dtype/attention policy, deterministic settings, capture-tool
digest, prompt tokens/masks/positions, hook contract, and all gates in this
record. Header and manifest geometry must agree exactly.

## Digest and provenance rules

Hash each of the 26 shard files before capture. The checkpoint content root is
SHA-256 of this byte stream:

1. ASCII `BEHOLDDQ35CHECKPOINT` plus NUL;
2. the 32-byte config digest and 32-byte index digest; then
3. for each shard in bytewise filename order: `u64 name_length`, filename
   bytes, `u64 file_size`, and its 32-byte content digest.

All integer lengths are little-endian. Keep read-only descriptors for all
shards open throughout capture. Record device/inode, size, nanosecond mtime,
and nanosecond ctime from those descriptors before and after, then rehash every
shard after capture and reproduce the same content root. Any identity or digest
change aborts publication. This root, rather than a cache path, timestamp, or
revision name alone, is the weight identity.

The environment manifest is canonical JSON containing executable/library
paths, versions and hashes, attention implementation, optional
causal-convolution/FLA choices, thread counts, device map, and deterministic
flags. The prompt-token manifest digest covers the canonical JSON subtree with
token IDs, attention masks, prompt IDs, and selected positions. The extractor
digest covers its exact source bytes.

Every section has the digest in its descriptor. The finalized trace also gets
an adjacent `.sha256` in standard `sha256sum` form. Writers use create-new
temporary files, flush and validate bytes, and publish with
`renameat2(RENAME_NOREPLACE)` or the packer's create-new hard-link pattern.
Plain overwriting `rename` is forbidden. A future replay runner must require
`--expected-trace-sha256`, verify trace digest and stat identity before and
after use, and independently retain the existing pack digest/identity checks.

## Exact offline Transformers hooks

Load `Qwen3_5MoeForConditionalGeneration` from the pinned local snapshot and
resolve, without replacing any module:

```python
layer = model.model.language_model.layers[3]
```

Install observation-only hooks that detach, select the frozen rows, make
contiguous CPU copies, and return `None`:

1. `layer.post_attention_layernorm` forward pre-hook: its first input becomes
   `pre_norm.f32`.
2. The same norm's forward hook: its output becomes `normalized.f32`.
3. `layer.mlp.gate` forward hook: capture its returned
   `(router_logits, route_weights, route_ids)` tuple. IDs alone are
   insufficient.
4. `layer.mlp.shared_expert_gate` forward hook: capture the raw linear output,
   before sigmoid, as `shared_logits.f32`.
5. `layer` forward hook: capture the final post-FFN residual output as
   `post_ffn.f32`.

The hook first sees `[B,S,...]` and selects the predeclared last valid row for
each batch item. Capture runs under `model.eval()`, `torch.inference_mode()`,
`use_cache=False`, `logits_to_keep=1`, and no autocast or sampling. The complete
official forward is allowed to finish; no exception, layer-count mutation,
source patch, or server instrumentation is used to stop after layer 3. The
manifest records every executed checkpoint shard even though only layer-3
values are stored.

The existing Beholdd `route_trace` output is not a source for this artifact: it
emits IDs only and cannot establish hidden inputs, normalized values, route
weights, shared gating, or the post-FFN oracle.

## Capture isolation and 31-GiB limit

Capture is a separate local process, never a serving request. It must:

- set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
  `PYTHONDONTWRITEBYTECODE=1`, and pass `local_files_only=True`;
- make no application service connection and never stop, restart, reconfigure, preload,
  monkey-patch, or instrument golden reference deployment or its `llama-server`;
- mount/treat the snapshot and audited reference deployment package tree as read-only, with
  offload, temporary, and trace files outside both trees; and
- run from a separately named, locked capture environment; the golden
  interpreter is not invoked. The isolated environment must contain a
  byte-identical copy of the audited model source, and the manifest records
  both audited and executed paths/digests. Installing into or updating the
  golden environment is forbidden.

The host has 31 GiB RAM, while checkpoint payloads occupy about 66.97 GiB. A
normal resident load is forbidden. The first smoke must use Accelerate
meta-device initialization and explicit CPU/disk dispatch. Immediately before
launch, set the process/cgroup memory ceiling no higher than the smaller of
16 GiB and live `MemAvailable - 8 GiB`; abort if that leaves less than 12 GiB.
Set Accelerate's CPU-weight budget at least 4 GiB below that hard ceiling, and
disable swap for the capture cgroup. This derives headroom from current
availability rather than total installed RAM. The local, non-golden offload
volume must have at least 80 GiB free; path and free space are recorded before
and after.

The single `B=16` forward streams weights once and uses bounded prompt lengths
and `use_cache=False`. Hooks retain only the 16 selected rows. No captured
section may be silently promoted except for the exact BF16-to-F32 widening
defined above; official internal F32 operations remain unchanged. Failure to
construct a complete offload map, OOM, swapping, unexpected device transfer,
or a write under a golden tree aborts without publishing a trace. Any later
GPU-assisted capture is a distinct provenance and must use the existing
thermal guard.

## Gates frozen before implementation

A failed gate remains a failed result. Relaxing a threshold requires a policy
revision and a fresh, unseen prompt corpus.

Capture semantic gates:

- structure, alignment, zero-reserved, exact-widen, finiteness, section,
  checkpoint, environment, and prompt digests all pass;
- each hook fires exactly once, selected records are non-padding rows, and an
  otherwise identical hooked and unhooked forward produces bit-identical final
  logits;
- two clean capture processes produce bit-identical semantic section payloads;
- recomputing RMSNorm from `pre_norm.f32` and the pinned packed gamma reproduces
  `normalized.f32` under the declared mixed-precision policy; and
- fresh F32 softmax/top-k/renormalization from stored logits reproduces unique,
  in-range, ordered IDs and stored BF16-widened weights exactly; shared sigmoid
  is recomputed and recorded as a diagnostic.

Replay semantic gates:

- trace, checkpoint root, config/index, layer pack, and prepared layer-3
  identities match; prepared matrices, gathered rows, and planner bijection
  retain their exact-byte checks;
- candidate and 16-sequential control are deterministic across independently
  poisoned runs, match one another's routes/status/scales exactly, and keep all
  values finite; and
- row-zero rounded input, gate/up, BF16 activation, and down stages retain the
  existing bit-exact checks. Full output bit parity is always calculated and
  reported separately, even when the numerical gate passes.

Numerical and route gates:

- candidate versus the qualified sequential control remains fixed at maximum
  per-row relative L2 `<= 5e-6` and maximum absolute error `<= 2e-5`;
- harness-versus-Transformers route robustness is decided from the eighth/ninth
  router-logit gap. A row is robust only when that gap is greater than eight
  times the maximum harness/HF router-logit error measured on a disjoint
  calibration corpus. Exact HF IDs are required for every robust row; fragile
  mismatches are retained and reported separately, never silently dropped or
  replaced; and
- before HF post-FFN comparison, round harness outputs to BF16 and widen them
  exactly to F32. Gate per-row relative L2, maximum sign-ordered BF16 ULP
  distance, and `max_i(|candidate_i - HF_i| / max(1, |HF_i|))`; report raw
  maximum absolute error as a diagnostic. Exact limits are frozen in the
  manifest from a disjoint calibration before the timed corpus is opened. The
  internal CPU-oracle ceilings of `5e-4` and `1e-3` are not cross-framework HF
  tolerances.

The first B=16 artifact is therefore a semantic/calibration slice unless HF
tolerances were already frozen on a different artifact. Correctness is decided
before timing. A qualifying timed replay uses the existing digest-bound,
thermally guarded, counterbalanced protocol and reports layer, route-margin
bucket, active experts, overlap, occupancy, control/candidate latency, paired
speedup, amortized request cost, and temperature.

## Explicit blockers

1. No extractor or v1 reader exists; this record deliberately adds neither.
2. The exact Transformers `5.8.0` full Qwen3.6 forward under CPU/disk offload,
   a live-availability-derived hard limit, and required offload space are
   unproven on this host.
3. Optional causal-convolution and FLA availability can change the executed
   DeltaNet path. The capture environment must freeze and record the choice;
   an implicit fast/fallback change creates a new oracle identity.
4. The current `route_trace` emits IDs only. Golden `llama-server` has no hidden
   activation endpoint, and adding one would require a custom binary/restart
   that violates capture isolation.
5. Beholdd's current dispatcher supports exact `qwen3` and `qwen3moe` checkpoints;
   its routed binder rejects shared experts, and it lacks `qwen3_5_moe` plus its 30
   Gated DeltaNet layers. Only the contained layer-3 pack can be replayed; a full
   Qwen3.6 forward cannot.
6. The HIP cohort harness currently accepts only synthetic sinusoidal/hashed
   inputs and is fixed to layer 3. It still needs a strict trace reader,
   expected-trace digest, and cohort selector.
7. The layer-pack header carries geometry but no checkpoint provenance. Its
   external SHA-256 and pre/post identity checks remain mandatory.
8. Layer-3 prefill parity does not establish cached-decode, other-layer,
   full-hidden-state, logit, generated-token, scheduling, or end-to-end
   throughput parity. Production remains NO-GO and golden serving remains the
   fallback.
