# ADR-0002 — `mimo2` is a level-2 architecture of its own

**Status:** accepted 2026-09-23. Implementation in progress; no gate claimed.

## The question

MiMo-V2.6-Flash-RL is the first checkpoint we intend to serve first-rate that is not a
Qwen. It is 308.8B parameters, 14.2B of them active per token, and its experts are
**natively MXFP4** — the published weights are 4-bit with block scales, not a quantization
of an 8-bit release, so there is no higher-fidelity build to hold out for.

The cheap answer would be to widen `qwen3moe`'s shared core until MiMo fits through it.
This records why that is the wrong answer, and what the right one requires.

## What the checkpoint actually is

Read from `MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf`, not from a model card:

| | |
|---|---|
| blocks | 48 — **block 0 dense** (FFN width 16384), the other 47 routed |
| embedding / expert FFN | 4096 / 2048 |
| heads | 64 query, **per-layer** KV counts `[4, 8, 8, 8, 8, 4, …]` |
| head widths | key **192**, value **128** |
| experts | 256 routed, 8 used, gating function 2, group routing 1×1 |
| | per-expert `exp_probs_b.bias` added to the scores **before** the top-k choice |
| attention | full on blocks 0, 5, 11, 17, …, 47; sliding window 128 on the other 39 |
| | per-head sink bias on exactly those 39 windowed blocks |
| RoPE | the leading 64 of each 192-wide head, split-half (NEOX); base 10⁷ full, 10⁴ windowed |
| other | fused QKV projection — 12288 query + 768 key + 512 value = 13568 wide on block 0 |
| | values scaled by 0.707 **after** projection; logits scaled by `key_head_dim^-0.5` |
| context | 1,048,576 |

Two of these are not in the metadata at all. Whether a block routes, and whether it has a
sink, the checkpoint states only by which tensors it carries — so the binder reads the tensor
table for them rather than assuming the model is uniform. The first version of this ADR did
assume it, and was wrong about block 0.

Absent from the checkpoint, and therefore refused if a later one declares them:
`expert_weights_scale` (the routed scaling factor) and `expert_weights_norm` (top-k
renormalisation). Both change the weight a chosen expert contributes, so binding a checkpoint
that has one while ignoring it would answer differently while every shape still agreed.

The arithmetic above is read from `modeling_mimo_v2.py` in the published repository, not
inferred: `rotate_half` over `query_states.split([rope_dim, head_dim - rope_dim])`, sinks
concatenated as an extra logit column before the softmax and dropped after it, and
`value_states * v_scale`.

The one thing the reference could not settle — whether the fused projection really runs
query, key, value in that order, since taking its word for it and checking against an oracle
that took the same word proves nothing — the checkpoint settles itself. The row-norm profile
of `attn_qkv` steps at `q + k` in 25 of 48 blocks and at `q + v` in none;
`docs/research/native-mimo2-block.md` has the method.

## Why it cannot extend the Qwen core

`qwen3moe::Config` is not merely missing fields; four of its invariants are false here.

1. **`value_head_dim == head_dim`.** The Qwen core requires it and says why: "the attention
   output is head-shaped". MiMo's values are 128 wide against 192-wide keys, so the output
   projection no longer has the shape the core assumes.
2. **One `n_kv_heads` for the model.** MiMo varies it per layer, because its full-attention
   and sliding-window layers are shaped differently.
3. **One RoPE base over the full head.** MiMo rotates a third of each head and uses a
   different base depending on the layer's attention type.
4. **Softmax top-k routing.** MiMo declares `expert_gating_func = 2` with a routed scaling
   factor and group routing.

The Qwen core already names `attention.sliding_window` and `expert_gating_func` in
`supports_arch_metadata` as **explicit refusals until implemented**. That list is the
architecture telling us, correctly, that this checkpoint is not one of its own. Widening it
would replace a refusal that is true with a binding that is approximately right, which is
the failure this codebase exists to avoid.

## Decision

`mimo2` becomes a level-2 architecture in its own module, sharing published operators with
the Qwen core where the mathematics is genuinely the same (RMSNorm, SwiGLU inside an expert,
the tokenizer) and owning what is not (attention shape, RoPE schedule, routing).

The `mtp-*.gguf` sidecar is bound as MTP heads for speculative decoding. This is the reason
the work belongs here rather than upstream: llama.cpp's `--spec-type draft-mtp` reads MTP
heads out of the main model and its `--spec-draft-model` expects a whole model, so the
sidecar ggml-org publishes cannot be attached at all — it fails on the missing
`blk.0.attn_q.weight` and takes the target down with it. We write the loader, so the heads
bind beside the weights the way the projector does.

## Gates

There is no small `mimo2` checkpoint to develop against. Every V2-family release — Flash,
Flash-Base, 2.5, 2.5-Pro, 2.5-DFlash — is ~300 GiB with the same 48×256 shape, and the one
small sibling (MiMo-V2.5-ASR, 29.89 GiB) is `qwen2`. So:

- **G1 — bind.** Config reads every hyperparameter above out of the real checkpoint header,
  and refuses, by name, anything it cannot honour. Costs a header read, not a load.
  *Passed, then corrected:* the first pass bound the metadata and assumed the tensor layout,
  which read block 0 as routed and knew nothing of the selection bias.
- **G2 — one block against an f64 oracle.** On a synthetic `mimo2` fixture small enough to
  transcribe independently (2 blocks, 4 experts, 128 wide), covering both attention types,
  the sink bias, the split RoPE schedule and sigmoid routing.
  *Passed:* worst disagreement 3.4e-7 against a 1e-5 gate, and nine deliberate mistakes —
  wrong RoPE base, trailing dimensions rotated, interleaved pairs, value scale dropped,
  attention scale from the value width, heads grouped by remainder, window off by one, sink
  summed rather than dropped, bias folded into the weight — each miss it by orders of
  magnitude. `docs/research/native-mimo2-block.md`.
- **G3 — one real block.** The same comparison against one block read out of the 156 GiB
  checkpoint, so the fixture cannot be quietly wrong about the real tensor layout.
  *Passed on block 0* (dense, full attention: 1.16 GB dequantised, `13568 = 12288 + 768 + 512`
  confirmed against the file, relative disagreement 2.4e-6) *and on block 1* (windowed,
  sinked, routed over MXFP4 experts: `14848 = 12288 + 1536 + 1024`, eight experts read
  selectively, relative disagreement 1.2e-6). Running a routed block first required a reader
  for ggml's MXFP4 layout, which did not exist — `mxfp4::matvec` had one caller in the
  repository and it was a line keeping the symbol alive.
- **G4 — decode.** Coherent greedy decode of the real checkpoint through the streaming path,
  byte-identical between resident and streamed execution.
  *Passed on coherence, and stronger than written:* the top eight tokens and their order match
  llama.cpp's on the same file, worst log-probability difference 0.042 nats. It failed first,
  and found a bug no arithmetic gate could — `norm_topk_prob`, which the GGUF does not carry
  and this loader read as off. The identity half is **not** claimed: 155.87 GiB against 31 GiB
  of RAM means there is no resident path here to compare the streamed one against.
  `docs/research/native-mimo2-decode.md`.
- **G5 — MTP.** Draft acceptance measured against the same prompt decoded without drafting;
  identical tokens, fewer forward passes.

Vision and audio (`mmproj`) stay out of scope here and keep their own roadmap entry: they
are a second checkpoint with a second set of operators, and `mimo2` must decode text before
being asked to see.
