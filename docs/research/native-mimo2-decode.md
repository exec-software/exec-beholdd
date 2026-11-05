# `mimo2` — G4 decode gate record

**Date:** 2026-09-23. **Result:** passed, after failing and finding a real bug.

## What was run

```
cargo run -p beholdd-model --release --example real_mimo2_decode -- \
  "$BEHOLDD_MODEL_DIR"/mimo-v26-flash/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf \
  "The capital of France is" 0
```

All 48 blocks, nothing resident. 155.87 GiB does not fit on this machine, so a block's
weights are read, used and dropped, and only the 8 experts of 256 a token routes to are ever
expanded — 3.1% of the model. Layers run outermost and positions innermost, so each block is
read once per pass rather than once per token. 237 seconds for a five-token prefill and one
prediction, single-threaded and scalar.

## The first run was wrong, and looked fine

```
  step 0 — top 8:
    " the"  logprob -0.9798      " to"       -4.1617
    " a"            -2.1449      " ("        -4.3119
    " not"          -3.2080      " another"  -4.4024
                                 " in"       -4.4664
```

Finite, ordered, grammatical — and `" Paris"` is not in it. Every gate up to here had passed:
the binder read every hyperparameter, one block agreed with an f64 oracle to 3.4e-7, and two
real blocks agreed to about 1e-6. None of them could see this, because **the oracle and the
implementation shared the mistake**. That is the whole reason a decode is its own gate.

llama.cpp, serving the same file on the same machine, says:

```
  " a" -1.8050   " Paris" -1.8789   " the" -2.3815   " one" -2.4260
```

## The bug: `norm_topk_prob`

`config.json` for every published V2 checkpoint says `"norm_topk_prob": true`. The chosen
experts' weights are divided by their sum. llama.cpp's converter does not emit
`expert_weights_norm` for this architecture, so the key is simply absent from the GGUF — and
this loader read absence as "off", which was a decision made by not noticing there was one.

Eight sigmoid scores sum to roughly four rather than one. Every routed block therefore spoke
about four times too loudly, and forty-seven of them compounded it. The result was a model
that wrote fluent English and had forgotten the capital of France.

The size of it shows up in G3 too: the same real block's residual reached **35.3** before the
fix and **4.78** after.

`Config::norm_topk` is now true unless the checkpoint says otherwise, and a checkpoint that
says `expert_weights_norm = 0` is refused rather than approximated. That is the one default
in this binder, and it is the opposite of the previous one, which was also a default and was
merely invisible.

## After the fix

```
  step 0 in 236.9s — top 8:
    " a"       id  264  logprob -1.8019      " known"   id 3881  logprob -3.2685
    " Paris"   id 12095 logprob -1.9207      " famous"  id 11245 logprob -3.3816
    " the"     id  279  logprob -2.3514      " home"    id 2114  logprob -3.4188
    " one"     id  825  logprob -2.4285      " not"     id  537  logprob -3.9491
```

| rank | llama.cpp | beholdd | difference |
|---|---|---|---|
| 1 | `" a"` −1.8050 | `" a"` −1.8019 | 0.0031 |
| 2 | `" Paris"` −1.8789 | `" Paris"` −1.9207 | 0.0418 |
| 3 | `" the"` −2.3815 | `" the"` −2.3514 | 0.0301 |
| 4 | `" one"` −2.4260 | `" one"` −2.4285 | 0.0025 |
| 5 | `" known"` −3.2412 | `" known"` −3.2685 | 0.0273 |
| 6 | `" famous"` −3.3736 | `" famous"` −3.3816 | 0.0080 |
| 7 | `" home"` −3.4356 | `" home"` −3.4188 | 0.0168 |
| 8 | `" not"` −3.9393 | `" not"` −3.9491 | 0.0098 |

Same eight tokens, same order, worst disagreement 0.042 nats. One is llama.cpp's Vulkan graph
striped across two machines in mixed precision; the other is scalar `f32` Rust reading the
same file. They are not the same program and do not share a line of code, which is what makes
the agreement worth anything.

## What this does and does not claim

It claims the architecture is right: the fused projection's order, the two RoPE bases, the
partial rotation, the attention sinks, the asymmetric key and value widths, the per-layer head
counts, the MXFP4 expert layout, the selection bias and now the routing normalisation all
produce, together, the same distribution as an independent implementation.

It does not claim the original G4 wording, "byte-identical between resident and streamed
execution". There is no resident path for this checkpoint on this machine — 155.87 GiB against
31 GiB of RAM — so there is nothing to compare streamed execution against. Everything measured
here is the streamed path. The identity claim needs a machine that can hold the model, or a
smaller `mimo2` checkpoint, and neither exists yet.

Nor is it fast. 237 seconds for six tokens is a correctness harness, not an engine: every
matvec is a scalar loop, every block is re-read from disk, and each routed position runs
attention twice so the router can be asked which experts to fetch before fetching them.

Remaining: G5 (MTP drafting).
