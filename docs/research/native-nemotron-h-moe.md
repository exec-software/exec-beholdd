# Native `nemotron_h_moe` plan and fixture provenance

Status: **next architecture, not implemented**. Native beholdd still accepts only the
exact GGUF architecture tags `qwen3` and `qwen3moe`. The strict dispatcher must continue
to reject `nemotron_h_moe` until every gate below passes; recognizing the metadata or
opening the checkpoint is not model support.

This record freezes the checkpoint identities and the order of work before implementation.
The immediate parity target is NVIDIA Nemotron 3 Super 120B A12B Q4_K_M. It exercises a
hybrid SSM/attention model and a latent sparse MoE rather than another spelling of Qwen.
That makes it the next useful architecture boundary for beholdd, while keeping the current
Qwen serving path as the golden control.

## Frozen checkpoint identities

All byte counts below are decimal bytes. A repository revision and every LFS object hash
are part of the fixture; a mutable branch name or model label is not enough.

### Operational parity fixture: LM Studio Q4_K_M

- Repository: `lmstudio-community/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF`
- Revision: `16410500b9bdc62081c34cc2dff52471e001f970`
- Variant total: `86,051,079,584` bytes
- Pinned model card:
  [README at `16410500`](https://huggingface.co/lmstudio-community/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/blob/16410500b9bdc62081c34cc2dff52471e001f970/README.md)
- Pinned file/LFS manifest:
  [model API at `16410500`](https://huggingface.co/api/models/lmstudio-community/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/revision/16410500b9bdc62081c34cc2dff52471e001f970?blobs=true)

| shard | bytes | SHA-256 |
|---|---:|---|
| `NVIDIA-Nemotron-3-Super-120B-A12B-Q4_K_M-00001-of-00003.gguf` | 39,205,633,408 | `2cf8e99b9b71babc5932eb594b1fe1f9b9e407a460a5b49b0625f0c25e0ba799` |
| `NVIDIA-Nemotron-3-Super-120B-A12B-Q4_K_M-00002-of-00003.gguf` | 39,720,681,760 | `770b505e6fe3b19b422b5b0dd06201b07218d0af99e72e69086208983fe54219` |
| `NVIDIA-Nemotron-3-Super-120B-A12B-Q4_K_M-00003-of-00003.gguf` | 7,124,764,416 | `7420107ccfc77230deaf57971ae84a58a18c4e6e035aa99cf0e25fbbc4c7614a` |

Before any native parity run, obtain exactly those three shards on local solid-state
storage, verify all three SHA-256 values, and pass the first shard path to the split-GGUF
reader. File names and sizes alone are not integrity checks. Partial downloads and
unrelated temporary files must not enter discovery, byte totals, or parity checks.
Network shares remain unsafe storage under normal preflight policy; do not enable
`allow_unsafe_storage` merely to bypass that refusal.

### Alternative published fixture: Unsloth Q8_0

This published variant is recorded for comparison, not as evidence of native Beholdd
support. The native dispatcher still refuses `nemotron_h_moe`.

- Repository: `unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF`
- Revision: `036038fb30334a2d56a146c6f0d4871ab5edccbb`
- Variant total: `128,472,991,680` bytes
- Pinned model card:
  [README at `036038fb`](https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/blob/036038fb30334a2d56a146c6f0d4871ab5edccbb/README.md)
- Pinned file/LFS manifest:
  [model API at `036038fb`](https://huggingface.co/api/models/unsloth/NVIDIA-Nemotron-3-Super-120B-A12B-GGUF/revision/036038fb30334a2d56a146c6f0d4871ab5edccbb?blobs=true)

| shard | bytes | SHA-256 |
|---|---:|---|
| `Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00001-of-00004.gguf` | 7,872,576 | `7efc3a9c77a65929897e2fed1d1d670d514297508485f80996e77004f5e81332` |
| `Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00002-of-00004.gguf` | 49,004,160,864 | `1dd0a70006703d67ab63a07b49c5ef97cf3081b94549daef6eb792ce225a48b9` |
| `Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00003-of-00004.gguf` | 49,284,659,200 | `31dea2f2172a5ce5456ee5d55f4d4cc2bf18e5ddc6c3e6d512281fe65163deba` |
| `Q8_0/NVIDIA-Nemotron-3-Super-120B-A12B-Q8_0-00004-of-00004.gguf` | 30,176,299,040 | `4962476f3703a5f9e613b431b866d0344ac361ccae61a0e7e21ed8f8b47b5d3b` |

This is published artifact provenance, not the operational fixture or a claim that a
local copy is available. Download and verify the selected variant before testing it.

### Upstream BF16 oracle variant

The same pinned Unsloth revision publishes BF16 GGUF shards. It publishes no variant named
F16 or FP16, so this record must not relabel BF16 as FP16.

| shard | bytes | SHA-256 |
|---|---:|---|
| `BF16/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-00001-of-00005.gguf` | 49,202,224,160 | `779f10f641a71f6e6b75b24444e4f08835d8f8e0deffa5def34843d8488baa29` |
| `BF16/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-00002-of-00005.gguf` | 47,901,185,376 | `52d26fc6b5dac39e02e362a9f67b10c3de26fbca1316b5e76e5ab0401d3763bb` |
| `BF16/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-00003-of-00005.gguf` | 47,829,865,504 | `1e081132b7c548b8b25b64e57f710674d1c25222b1279456f1801f148ae6e928` |
| `BF16/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-00004-of-00005.gguf` | 47,901,185,376 | `e41fe65c67184e7d82096691de718b2c8c5f3e1c3b383d0e48e22ae6fd32c9d0` |
| `BF16/NVIDIA-Nemotron-3-Super-120B-A12B-BF16-00005-of-00005.gguf` | 48,684,199,008 | `a7d93452654ca274571703d2b7b7b6296bacda394e27e078a3a2d0e0d3786fdd` |

The BF16 total is `241,518,659,424` bytes. It is upstream evidence and a possible
high-precision oracle source, not a local checkpoint and not a prerequisite for the first
Q4_K_M native gate. A smaller independent implementation may provide the numerical oracle
as long as the reference, inputs, intermediates, and tolerances are frozen before comparing
beholdd.

## License evidence

The two conversion repositories expose different license labels. Preserve that distinction
rather than normalizing it away:

- The pinned LM Studio Q4 model card declares `license: other`,
  `license_name: nvidia-open-model-license`, and links the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
- The pinned Unsloth Q8_0/BF16 model card declares `license: other`,
  `license_name: nvidia-nemotron-open-model-license`, and links the
  [NVIDIA Nemotron Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/).
- Both cards identify `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16` as the base model.

These are provenance facts, not a legal conclusion. A release that redistributes weights
must review the license named by the exact chosen artifact revision; a downloader should
show and preserve that source/license identity.

## Observed Q4_K_M GGUF contract

The current beholdd GGUF reader opens the split Q4_K_M fixture as metadata, without
claiming execution support. It observes 3 shards, 763 tensors, 52 metadata keys, and
`general.file_type = 15` with quantization version 2. Encoded tensor payloads total about
86.04 decimal GB:

| GGML type | tensors | payload bytes, approximately |
|---|---:|---:|
| F32 | 409 | 0.35 GB |
| Q4_K | 289 | 35.57 GB |
| Q5_0 | 20 | 19.38 GB |
| Q6_K | 25 | 0.81 GB |
| Q8_0 | 20 | 29.95 GB |

The architecture metadata says:

- `general.architecture = "nemotron_h_moe"`, name `Nemotron 3 Super 120B A12B`;
- 88 numbered blocks, hidden width 4,096, vocabulary 131,072, context 1,048,576;
- 8 full-attention blocks at indices `7, 16, 25, 36, 47, 58, 69, 78`, with 32 query
  heads, 2 KV heads where attention is present, and 128-wide keys/values;
- 40 SSM blocks, with convolution width 4, state size 128, 8 groups, inner size 8,192,
  and time-step rank 128;
- 40 latent-MoE blocks, with 512 routed experts, 22 selected experts, expert width 2,688,
  latent width 1,024, one shared expert of width 5,376, normalized expert weights, and
  routing scale 5.0.

The repository API reports 120,668,707,840 parameter elements. That number is neither an
artifact byte count nor the amount of memory a native execution plan may assume.

## Staged native gates

The stages are intentionally ordered so a metadata parser, one correct operator, or one
coherent completion cannot silently become a support claim.

### N0 — Provenance and safe staging

1. Stage the exact three Q4_K_M shards on local solid-state storage.
2. Match all three byte counts and SHA-256 values above before GGUF binding.
3. Freeze the checkpoint revision, tokenizer/chat-template bytes, reference runtime and
   commit, prompts, token IDs, state/intermediate captures, logits, and tolerances.
4. Prove that a missing, reordered, truncated, foreign, or extra shard is refused. The
   `.goutputstream-*` file is a specific negative fixture.

**Gate N0:** a checked manifest identifies one local checkpoint byte-for-byte, and the
existing storage policy accepts it without an unsafe override. This gate says nothing
about architecture execution.

### N1 — Strict binder and tensor manifest

Add a separate `nemotron_h_moe` binder. It must validate the architecture's complete
metadata whitelist, per-block kind arrays, 763 tensor names, shapes, types, and split
placement before returning a model. It must not route through `QwenModel`, alias an
architecture string, or accept a nearby `nemotron_h` checkpoint by accident.

**Gate N1:** the pinned Q4_K_M fixture binds metadata-only; adversarial changes to every
architecture-defining dimension and each of the three block kinds fail before tensor
payload execution. The public serving registry remains `unverified` and capability-free.

### N2 — Independent sublayer oracles

Implement and compare the distinct semantics independently:

1. recurrent SSM state, convolution, time-step transform, normalization, and residual;
2. sparse full-attention positions, RoPE/GQA, causal masking, and incremental KV state;
3. latent router projections, expert probability bias, stable top-22 selection,
   normalization/scale, shared expert, routed experts, and caller-order reduction.

Reuse quantization decoders and generic operators where their existing contracts match,
but do not infer Nemotron semantics from Qwen tensor names. Each sublayer gate requires an
independent implementation or frozen reference trace, finite-value checks, exact selected
expert IDs, declared numerical tolerances, and repeatability.

**Gate N2:** every distinct sublayer matches its frozen oracle on synthetic edge cases and
real Q4_K_M weights. No full-model or serving claim follows yet.

### N3 — Mixed-block execution and state ownership

Compose the 40 SSM, 8 attention, and 40 MoE blocks in the exact metadata order. The model
state must distinguish recurrent SSM state from attention KV state, support prefill and
one-token decode, and make reset/sequence ownership explicit. Streaming phases must expose
only checked ranges and may reveal routed experts only after the router barrier.

**Gate N3:** selected block-boundary states from a real prompt match the frozen reference;
fresh and incremental execution agree at predeclared boundaries; reset, cancellation, and
two independent sequences cannot leak SSM or KV state.

### N4 — Full logits and token parity

Bind the exact tokenizer and template captured in N0, then compare full-model prefill and
incremental decode against the pinned reference. Compare complete selected logit rows
before using greedy token identity as the final visible check. Q4_K_M quantized execution
needs tolerances justified against the reference arithmetic; a coherent sentence alone is
not evidence.

**Gate N4:** at least two fresh beholdd processes produce repeatable logits and greedy
tokens for the fixed prompt set, meet the predeclared reference tolerance at every captured
row, and stay inside an explicit memory/thermal envelope.

### N5 — Resident/streaming and serving qualification

Run the same checked Q4_K_M bytes through resident and streaming sources where the host
budgets permit, with complete state/logit comparison. Then exercise the existing embedded,
HTTP, SSE, cancellation, model-switch, and liveness gates. The 1,048,576-token metadata
limit is not a promise that the first implementation allocates or serves that context;
advertised capacity must come from the tested runtime plan.

Only after N0-N5 pass may `nemotron_h_moe` enter the public implemented-architecture set
or advertise `completion`. Vulkan/HIP acceleration can follow through the same execution
seam; it is not allowed to replace the CPU/reference correctness gate.

## Decision line

`nemotron_h_moe` is the next native architecture target. It is **unsupported today**.
Until all gates pass, Antimatter should continue using its compatible llama.cpp path for
Nemotron Super, and beholdd must continue returning its explicit unsupported-architecture
error rather than falling back or publishing a capability.
