# Dense Qwen3 native execution gate

Dense `qwen3` is the next bounded architecture implemented after `qwen3moe`. It is useful
Hub breadth and a proof that the architecture boundary can share attention/tokenization
without pretending that later Qwen families are aliases. It does **not** close
Antimatter's current `qwen3next` text-model gap or its `qwen35moe` vision-model gap.
Those architecture tags remain rejected.

## Executable contract

`QwenModel::load` dispatches only the exact GGUF tags `qwen3` and `qwen3moe`. The dense
branch reuses Qwen3 GQA, per-head Q/K RMSNorm, split-half RoPE, residuals, final RMSNorm,
the tied/untied language-model head, and the Qwen2 tokenizer. It separately binds a dense
gated SwiGLU FFN:

- `blk.N.ffn_gate.weight`: `dim -> feed_forward_length`;
- `blk.N.ffn_up.weight`: `dim -> feed_forward_length`; and
- `blk.N.ffn_down.weight`: `feed_forward_length -> dim`.

The binder requires the dense Qwen3 metadata used by those operations and accepts exactly
11 tensors per block plus token embeddings and output norm, with one optional untied
`output.weight`. Routed tensors, bias tensors, unknown `qwen3.*` semantic switches, wrong
shapes, missing tensors, and extra tensors fail before payload execution. The
`qwen3moe` binder remains separately strict and will not accept a dense checkpoint.

Weight-fit accounting includes every encoded tensor in the checkpoint total and the
checked sum of dense gate/up/down bytes as one concurrent live cohort. Each matrix is
loaded once and retained across a multi-position prefill instead of being reread for
every prompt token. CPU matvec calls remain sequential, so dequantization scratch is the
largest individual matrix, not the sum of all three. Decode-buffer accounting includes
the dense gate and up scratch vectors. This is deliberately different from the routed
branch, whose largest live request is one expert slab rather than its aggregate expert
tensor.

The ordinary synthetic regression runs a two-token prefill through two dense layers,
asserts that each gate/up/down range is loaded exactly once per layer, and compares a
bounded `StreamingSource` decode bit-for-bit with `FullyResidentSource`. This covers the
runtime's default streaming mode independently of the resident real-checkpoint smoke.

## Real checkpoint result

The local real gate uses:

```text
Qwen3-0.6B-Q8_0.gguf
file bytes: 639,447,744
encoded tensor bytes: 633,495,552
SHA-256: e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c
```

The strict binder reports 28 layers, hidden width 1024, 16 query / 8 KV heads, head width
128, dense FFN width 3072, vocabulary 151,936, context 40,960, and exactly 310 tensors.
The Qwen2 tokenizer maps raw `Hello` to `[9707]`. Two resident eight-token decodes were
identical and produced:

```text
[21806, 11, 358, 2776, 14589, 369, 279, 3681]
" Answer, I'm sorry for the previous"
```

Run the bind, executable smoke, and ignored real-checkpoint regression with:

```sh
cargo run -p beholdd-model --release --example bind -- <Qwen3-0.6B-Q8_0.gguf>
cargo run -p beholdd-model --release --example decode -- \
  <Qwen3-0.6B-Q8_0.gguf> 8 Hello --weight-mode resident --runtime-headroom-bytes 0
cargo test -p beholdd-model \
  real_dense_qwen3_binds_and_has_a_stable_coherent_decode -- --ignored
```

## Numerical reference and non-parity result

`tools/qwen3_dense_ref.py` is a separate f64 NumPy transcription over the GGUF bytes. For
the first `Hello` step it reports:

| rank | token | decoded piece | f64 logit |
|---:|---:|---|---:|
| 1 | 21806 | ` Answer` | 8.0050891847 |
| 2 | 14582 | `Question` | 7.7874742604 |
| 3 | 15846 | ` Question` | 7.6408847611 |

The top-to-runner-up margin is 0.2176149243. Beholdd reports the same ordering with logits
8.0051002502 and 7.7874903679 (margin 0.2176098824), within 1.7e-5 of the independent f64
values.

This is **not llama.cpp token parity**. `llama-completion` b10375 (`ba360efe1`), CPU-only, raw prompt,
`-no-cnv -n 1 --temp 0`, selects token 15846 (` Question`). The differing result is
consistent with different quantized arithmetic: beholdd computes direct decoded
Q8_0-weight × f32 dot products, while llama.cpp's CPU Q8_0 matrix path requantizes
activations to Q8_0 for Q8_0 × Q8_0 dot products. The independent f64 agreement is
evidence that the dense architecture assembly is correct, but it is not evidence of
llama.cpp output identity. Any compatibility requirement for llama-identical quantized
tokens needs its own explicit backend arithmetic policy and gate.

Reproduce the f64 side with:

```sh
python3 tools/qwen3_dense_ref.py <Qwen3-0.6B-Q8_0.gguf> 9707 3
```
