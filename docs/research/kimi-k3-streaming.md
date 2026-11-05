# Kimi K3 stress-target profile

Research snapshot: 2026-08-14. This note separates facts published by Moonshot from
engineering estimates derived from the published checkpoint. K3 is one adversarial target
for the [model-agnostic streaming program](model-streaming.md), not a separate streaming
implementation. This profile is not a claim that beholdd can execute K3 yet.

## Verified target

Moonshot's open-weight Kimi K3 has 2.8 trillion total parameters, 104 billion activated
parameters, 93 decoder layers, and a 1,048,576-token context. Its text stack combines 69
Kimi Delta Attention (KDA) layers with 24 Gated MLA layers. The MoE has 896 routed
experts, selects 16 per token, and also uses two shared experts. Native weights use
MXFP4 and inference activations use MXFP8. K3 also includes a MoonViT-V2 vision tower.

The official checkpoint is not GGUF. It consists of 96 Safetensors weight shards totaling
1,560,936,091,448 bytes (about 1.561 TB / 1.420 TiB). Shards 1 through 93 correspond to
decoder layers 0 through 92; shard 94 holds embeddings, final normalization and the
language-model head, while shards 95 and 96 hold the projector and vision tower. There
is no official smaller K3, alternate lower-bit release, or GGUF release.

Primary sources:

- [Moonshot Kimi K3 repository and technical report](https://github.com/MoonshotAI/Kimi-K3)
- [Official Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3)
- [Official configuration](https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json)
- [Official model implementation](https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py)
- [Official Safetensors index](https://huggingface.co/moonshotai/Kimi-K3/blob/main/model.safetensors.index.json)
- [Official repository metadata and blob sizes](https://huggingface.co/api/models/moonshotai/Kimi-K3?blobs=true)
- [Official Kimi K3 collection](https://huggingface.co/collections/moonshotai/kimi-k3)

## What the layout means for streaming

The official Safetensors headers place each routed expert's six packed-weight and scale
tensors in one contiguous 17,547,264-byte span. That makes expert-selective range reads
possible. It also makes loading an entire layer or shard the wrong primitive.

For text-only decode, a header-derived estimate with perfect top-16 selection is about
136,990,260,224 bytes of uncached weight traffic per token. Selected expert spans alone
account for 25,829,572,608 bytes. The estimate includes common weights for 68 KDA-MoE
layers and 24 MLA-MoE layers, the dense first layer, and the language-model head; it
excludes small embedding-row and normalization reads. Naively rereading whole shards is
about 1.56 TB per token.

At the 2.107 GB/s cold-read rate measured during the Qwen3-235B M2 run, those byte counts
imply approximate I/O floors of 65 seconds/token for selective reads and 741 seconds/token
for whole-shard reads. These are estimates, not performance results: caching, overlap,
route locality and storage parallelism can change the achieved rate. They do establish
that expert-range reads are mandatory and that useful speed also needs prefetch and a
real retention policy.

K3's request state is hybrid rather than an ordinary transformer KV cache. KDA retains a
large fixed recurrent state, while MLA retains token-linear state. From the published
dimensions, KDA's matrix state alone is 69 * 96 * 128 * 128 = 108,527,616 elements,
about 207 MiB in BF16 or 414 MiB in FP32 per request, plus convolution state. A rank-512
MLA cache over 24 layers and the full one-million-token context is roughly 12 GiB at one
byte per element or 24 GiB at two bytes per element, before implementation-specific
metadata and auxiliary state.

## Relationship to the generic program

K3's layout establishes demanding inputs for the shared program: multiple tensors may be
one useful physical range, selected ranges become known only after routing, and request
state is not always an append-only KV cache. Those facts shape generic contracts without
putting KDA, MLA, routing or expert concepts into `beholdd-memory`.

The generic prerequisites and their executable gates live in
[`model-streaming.md`](model-streaming.md): format-neutral checked locators, ordered range
phases with data-dependent barriers, bounded prefetch and paired Qwen performance runs.
Qwen3-MoE remains the implementation and tuning foundation. K3 tests whether the result
actually transfers.

## Additional K3 enablement work

After the shared streaming gates, K3 still requires model and encoding support:

1. The official Safetensors checkpoint must bind through the format-neutral source proven
   by M2.4; it does not get a Kimi-only reader or cache.
2. Quantization dispatch must represent compressed-tensors MXFP4 packed values and scale
   tensors, reusing the independent MXFP4 scalar/vector kernel checks.
3. Decode state must support checked snapshot, restore and rollback for KDA's fixed
   recurrent state together with MLA's token-linear state. Prefix reuse restores both at
   one validated boundary.
4. K3 semantics then add KDA, Gated MLA, Block AttnRes and Stable LatentMoE/SiTU, each with
   a synthetic fixture before real weights. MoonViT-V2 follows after text-only execution.

None of these items authorizes changes to the generic range scheduler, fit policy or
telemetry solely for K3.

The current internal NVMe has about 145 GB free, far below the 1.56 TB checkpoint. Until
at least roughly 1.6 TB of additional fast storage is available, K3 work must use parser,
range-layout and architecture fixtures rather than a local full-checkpoint run.
