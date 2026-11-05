#!/usr/bin/env python3
"""
qwen3moe_ref.py - an f64 reference for one qwen3moe block, on real checkpoint bytes.

WHY THIS EXISTS
    Gate M1.1. Every layer under the block has been checked, and none of those checks
    can catch a block assembled wrongly: a transposed projection, rope applied with the
    wrong convention, a router that softmaxes after top-k instead of before. Each of
    those produces fluent, confident, wrong output and no error anywhere.

    So this is a SECOND implementation, written from the architecture rather than from
    the Rust, with its own GGUF reader and its own dequantiser. Two implementations that
    agree to 1e-6 on real weights are evidence; one implementation checked against
    itself is not.

THE THINGS MOST WORTH PINNING, because they are conventions rather than arithmetic
    - GGUF dims are FASTEST-FIRST: [cols, rows]. The transpose of a square-ish matrix is
      still a valid shape and still runs.
    - Qwen3 rope is NEOX / split-half (rotate_half), NOT interleaved pairs. DeepSeek-V4
      is interleaved, so this is exactly the sort of thing that gets carried across
      wrongly.
    - QK-norm: q and k are RMSNormed PER HEAD with a head_dim-wide weight, BEFORE rope.
    - The router softmaxes over ALL experts, then takes top-k, then RENORMALISES
      (norm_topk_prob). Softmaxing after the top-k gives different weights that still
      sum to one.

usage: qwen3moe_ref.py <model.gguf> <out.bin> [n_positions]
"""
import struct
import sys

import numpy as np

# ---------------------------------------------------------------- gguf ----

TYPES = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def read_gguf(path):
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    ver, = struct.unpack("<I", f.read(4))
    assert ver in (2, 3), f"unsupported GGUF version {ver}"
    nt, = struct.unpack("<Q", f.read(8))
    nkv, = struct.unpack("<Q", f.read(8))

    def s():
        n, = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def val(t):
        if t == 8:
            return s()
        if t == 9:
            et, = struct.unpack("<I", f.read(4))
            n, = struct.unpack("<Q", f.read(8))
            return [val(et) for _ in range(n)]
        fmt = TYPES[t]
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    meta = {}
    for _ in range(nkv):
        k = s()
        t, = struct.unpack("<I", f.read(4))
        meta[k] = val(t)

    infos = []
    for _ in range(nt):
        name = s()
        nd, = struct.unpack("<I", f.read(4))
        dims = [struct.unpack("<Q", f.read(8))[0] for _ in range(nd)]
        kind, = struct.unpack("<I", f.read(4))
        off, = struct.unpack("<Q", f.read(8))
        infos.append((name, kind, dims, off))

    align = meta.get("general.alignment", 32)
    pos = f.tell()
    data = (pos + align - 1) // align * align
    return f, meta, {n: (k, d, data + o) for n, k, d, o in infos}


def deq(f, kind, dims, off, count):
    """Dequantise `count` elements starting at `off`. F32 and Q8_0 only, which is what
    this checkpoint contains."""
    if kind == 0:  # F32
        f.seek(off)
        raw = f.read(count * 4)
        assert len(raw) == count * 4, "short F32 tensor read"
        return np.frombuffer(raw, dtype="<f4").astype(np.float64)
    if kind == 8:  # Q8_0: f16 scale + 32 int8, 34 bytes per block
        nb = count // 32
        assert count % 32 == 0
        f.seek(off)
        encoded = f.read(nb * 34)
        assert len(encoded) == nb * 34, "short Q8_0 tensor read"
        raw = np.frombuffer(encoded, dtype=np.uint8).reshape(nb, 34)
        d = raw[:, :2].copy().view("<f2").astype(np.float64)
        q = raw[:, 2:].copy().view(np.int8).astype(np.float64)
        return (d * q).reshape(-1)
    raise ValueError(f"unsupported ggml type {kind}")


def mat(f, tensors, name):
    """A 2-D weight as (rows, cols). GGUF stores dims fastest-first, so dims[0] is the
    COLUMN count."""
    kind, dims, off = tensors[name]
    cols, rows = int(dims[0]), int(dims[1])
    return deq(f, kind, dims, off, rows * cols).reshape(rows, cols)


def vec(f, tensors, name):
    kind, dims, off = tensors[name]
    return deq(f, kind, dims, off, int(dims[0]))


def expert(f, tensors, name, idx):
    """One expert out of a 3-D [cols, rows, n_experts] tensor. Experts are contiguous
    slabs, so this is an offset computation rather than a lookup."""
    kind, dims, off = tensors[name]
    cols, rows = int(dims[0]), int(dims[1])
    per = rows * cols
    assert kind == 8, "expert tensors are expected to be Q8_0 here"
    stride = per // 32 * 34
    return deq(f, kind, dims, off + stride * idx, per).reshape(rows, cols)


# ------------------------------------------------------------- ops ----

def rmsnorm(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope_neox(x, cos, sin):
    """Split-half rotation, HF's rotate_half. The first half of the head is rotated
    against the second half -- NOT adjacent interleaved pairs."""
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def softmax(v):
    e = np.exp(v - v.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit("usage: qwen3moe_ref.py <model.gguf> <out.bin> [n_positions]")
    path, out = sys.argv[1], sys.argv[2]
    npos = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    if npos <= 0:
        raise SystemExit("n_positions must be positive")
    layer = 0

    f, meta, tensors = read_gguf(path)
    A = meta["general.architecture"]
    assert A == "qwen3moe", A
    unsupported = [
        key for key in meta
        if key.startswith(f"{A}.rope.scaling.")
        or key in {
            f"{A}.rope.scale_linear",
            f"{A}.expert_weights_scale",
            f"{A}.expert_weights_norm",
            f"{A}.expert_gating_func",
            f"{A}.expert_group_scale",
            f"{A}.swiglu_clamp_exp",
            f"{A}.swiglu_clamp_shexp",
        }
    ]
    assert not unsupported, f"unsupported architecture metadata: {unsupported}"
    dim = meta[f"{A}.embedding_length"]
    n_heads = meta[f"{A}.attention.head_count"]
    n_kv = meta[f"{A}.attention.head_count_kv"]
    hd = meta[f"{A}.attention.key_length"]
    assert meta[f"{A}.attention.value_length"] == hd
    n_exp = meta[f"{A}.expert_count"]
    topk = meta[f"{A}.expert_used_count"]
    assert meta[f"{A}.expert_shared_feed_forward_length"] == 0
    eps = meta[f"{A}.attention.layer_norm_rms_epsilon"]
    base = meta[f"{A}.rope.freq_base"]
    group = n_heads // n_kv

    print(f"  {A}: dim {dim}, {n_heads}q/{n_kv}kv heads, hd {hd}, {n_exp} experts top-{topk}")

    P = f"blk.{layer}."
    attn_norm = vec(f, tensors, P + "attn_norm.weight")
    wq = mat(f, tensors, P + "attn_q.weight")
    wk = mat(f, tensors, P + "attn_k.weight")
    wv = mat(f, tensors, P + "attn_v.weight")
    q_norm = vec(f, tensors, P + "attn_q_norm.weight")
    k_norm = vec(f, tensors, P + "attn_k_norm.weight")
    wo = mat(f, tensors, P + "attn_output.weight")
    ffn_norm = vec(f, tensors, P + "ffn_norm.weight")
    router = mat(f, tensors, P + "ffn_gate_inp.weight")

    # Inputs: hc copies of real embedding rows, so the test runs on real activations
    # rather than on noise. Deterministic token ids.
    toks = [(i * 7919 + 13) % 100000 for i in range(npos)]
    ekind, edims, eoff = tensors["token_embd.weight"]
    assert ekind == 8, "reference expects Q8_0 token embeddings"
    assert edims[0] == dim
    xs = np.zeros((npos, dim))
    for i, t in enumerate(toks):
        stride = dim // 32 * 34
        xs[i] = deq(f, ekind, edims, eoff + stride * t, dim)

    # rope table
    inv = 1.0 / (base ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    pos = np.arange(npos, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)

    # ---- attention ----
    hn = rmsnorm(xs, attn_norm, eps)                       # [npos, dim]
    q = (hn @ wq.T).reshape(npos, n_heads, hd)
    k = (hn @ wk.T).reshape(npos, n_kv, hd)
    v = (hn @ wv.T).reshape(npos, n_kv, hd)
    # QK-norm, per head, BEFORE rope
    q = rmsnorm(q, q_norm, eps)
    k = rmsnorm(k, k_norm, eps)
    q = rope_neox(q, cos[:, None, :], sin[:, None, :])
    k = rope_neox(k, cos[:, None, :], sin[:, None, :])

    scale = hd ** -0.5
    o = np.zeros((npos, n_heads, hd))
    for t in range(npos):
        for h in range(n_heads):
            kv = h // group                                # GQA: 8 queries share a kv head
            s = (k[: t + 1, kv, :] @ q[t, h, :]) * scale   # causal: only up to t
            p = softmax(s)
            o[t, h] = p @ v[: t + 1, kv, :]
    xs = xs + (o.reshape(npos, n_heads * hd) @ wo.T)

    # ---- moe ----
    h2 = rmsnorm(xs, ffn_norm, eps)
    logits = h2 @ router.T                                 # [npos, n_exp]
    probs = softmax(logits)                                # softmax over ALL experts...
    sel = np.argsort(-probs, axis=-1, kind="stable")[:, :topk]   # ...then top-k...
    w = np.take_along_axis(probs, sel, axis=-1)
    w = w / w.sum(-1, keepdims=True)                       # ...then renormalise

    y = np.zeros_like(xs)
    for t in range(npos):
        for j in range(topk):
            e = int(sel[t, j])
            g = expert(f, tensors, P + "ffn_gate_exps.weight", e) @ h2[t]
            u = expert(f, tensors, P + "ffn_up_exps.weight", e) @ h2[t]
            act = g / (1.0 + np.exp(-g)) * u                # silu(gate) * up
            y[t] += w[t, j] * (expert(f, tensors, P + "ffn_down_exps.weight", e) @ act)
    xs = xs + y

    for t in range(npos):
        print(f"    pos {t} tok {toks[t]:>6}  experts {list(map(int, sel[t][:4]))}...  "
              f"out rms {np.sqrt((xs[t]**2).mean()):.6f}")

    with open(out, "wb") as fh:
        fh.write(struct.pack("<3i", npos, int(dim), int(topk)))
        fh.write(np.ascontiguousarray(toks, dtype="<i4").tobytes())
        fh.write(np.ascontiguousarray(sel, dtype="<i4").tobytes())
        fh.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
        fh.write(np.ascontiguousarray(xs, dtype="<f4").tobytes())
    f.close()
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
