#!/usr/bin/env python3
"""Independent f64 one-position reference for dense GGUF qwen3.

This deliberately shares no Rust execution code. It reuses the small GGUF reader and
Q8_0 decoder from qwen3moe_ref.py, then spells out dense Qwen3 attention/residual/SwiGLU
assembly directly in NumPy. One position is sufficient for the checked `Hello` smoke;
causal attention reduces to grouped repetition of V, while every real layer and the tied
language-model head still execute.

usage: qwen3_dense_ref.py <model.gguf> <token-id> [top-k]
"""

import sys

import numpy as np

from qwen3moe_ref import deq, mat, read_gguf, rmsnorm, vec


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: qwen3_dense_ref.py <model.gguf> <token-id> [top-k]")
    path = sys.argv[1]
    token = int(sys.argv[2])
    top_k = int(sys.argv[3]) if len(sys.argv) == 4 else 10
    if token < 0 or top_k <= 0:
        raise SystemExit("token-id must be non-negative and top-k must be positive")

    source, metadata, tensors = read_gguf(path)
    arch = metadata["general.architecture"]
    assert arch == "qwen3", f"expected qwen3, checkpoint says {arch!r}"
    dim = int(metadata[f"{arch}.embedding_length"])
    n_heads = int(metadata[f"{arch}.attention.head_count"])
    n_kv_heads = int(metadata[f"{arch}.attention.head_count_kv"])
    head_dim = int(metadata[f"{arch}.attention.key_length"])
    assert int(metadata[f"{arch}.attention.value_length"]) == head_dim
    assert n_heads % n_kv_heads == 0
    epsilon = float(metadata[f"{arch}.attention.layer_norm_rms_epsilon"])
    n_layers = int(metadata[f"{arch}.block_count"])

    embed_kind, embed_dims, embed_offset = tensors["token_embd.weight"]
    assert int(embed_dims[0]) == dim
    vocab = int(embed_dims[1])
    assert token < vocab
    if embed_kind == 8:
        row_bytes = dim // 32 * 34
    elif embed_kind == 0:
        row_bytes = dim * 4
    else:
        raise ValueError(f"reference supports only F32/Q8_0 embeddings, got {embed_kind}")
    hidden = deq(
        source,
        embed_kind,
        embed_dims,
        embed_offset + token * row_bytes,
        dim,
    )

    group = n_heads // n_kv_heads
    for layer in range(n_layers):
        prefix = f"blk.{layer}."
        attention_input = rmsnorm(
            hidden, vec(source, tensors, prefix + "attn_norm.weight"), epsilon
        )
        values = mat(source, tensors, prefix + "attn_v.weight") @ attention_input
        attended = np.repeat(values.reshape(n_kv_heads, head_dim), group, axis=0)
        hidden = hidden + mat(
            source, tensors, prefix + "attn_output.weight"
        ) @ attended.reshape(-1)

        ffn_input = rmsnorm(
            hidden, vec(source, tensors, prefix + "ffn_norm.weight"), epsilon
        )
        gate = mat(source, tensors, prefix + "ffn_gate.weight") @ ffn_input
        up = mat(source, tensors, prefix + "ffn_up.weight") @ ffn_input
        activated = gate / (1.0 + np.exp(-gate)) * up
        hidden = hidden + mat(
            source, tensors, prefix + "ffn_down.weight"
        ) @ activated

    hidden = rmsnorm(hidden, vec(source, tensors, "output_norm.weight"), epsilon)
    output_name = "output.weight" if "output.weight" in tensors else "token_embd.weight"
    logits = mat(source, tensors, output_name) @ hidden
    ranked = np.argsort(-logits, kind="stable")[:top_k]
    print(
        f"{arch}: token={token}, layers={n_layers}, dim={dim}, "
        f"heads={n_heads}/{n_kv_heads}, vocab={vocab}"
    )
    for rank, token_id in enumerate(ranked, 1):
        print(f"  {rank:>2}: token {int(token_id):>6}, logit {logits[token_id]:.10f}")
    source.close()


if __name__ == "__main__":
    main()
