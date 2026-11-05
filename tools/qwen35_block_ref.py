#!/usr/bin/env python3
"""Independent, f64, text-only oracle for ONE dense Qwen3.5 GGUF block.

This is a validation tool, not an inference backend. It starts with empty KV,
convolution and recurrent state, at text position zero, and consumes supplied
hidden vectors. It neither runs preceding blocks nor reads embedding/output
weights. Q4_K/Q6_K matrices are decoded a few rows at a time; never a whole model.

Architecture/packing references (not the production Rust implementation):
  llama.cpp 4da6337767f973e2b4d0797e5b323d77d8565e4a:
    src/models/qwen35.cpp, src/models/delta-net-base.cpp, src/models/models.h
    ggml/src/ggml-quants.c: get_scale_min_k4, dequantize_row_q4_K/q6_K
For text, all MRoPE axes have the same position, reducing to partial split-half
RoPE. GGUF norms are already converted weights; ssm_a is already -exp(A_log).
Supplied hidden inputs are rounded once to f32, matching the backend input
boundary, then widened to f64. All decoded weights and subsequent operations
here are f64, intentionally independent of F32 backend rounding. This is a
numerical oracle, not a bit-identical GGML run.

Input JSON: {"schema_version": 1, "hidden": [[...], ...]}
Output has schema_version, layer, block_kind, positions, hidden, and optional
stages. Every stage is [position][flattened head-major components]. State is not
serialized. Run under an external CPU/memory/time guard for real checkpoints:

  python3 tools/qwen35_block_ref.py --model MODEL.gguf --layer 0 \
      --input hidden.json --output block.json --max-positions 3 --stages

There are no default checkpoint paths or automatic real-weight executions.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import sys

# Bound BLAS before importing NumPy, including when invoked without a wrapper.
for _thread_env in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_thread_env] = "1"

import numpy as np


MAX_POSITIONS = 8
MAX_HEADER_BYTES = 128 << 20
MAX_INPUT_BYTES = 4 << 20
MAX_STATE_BYTES = 32 << 20
MAX_TILE_BYTES = 16 << 20
MAX_LAYER_BYTES = 1 << 30
SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
           7: "?", 10: "Q", 11: "q", 12: "d"}
GEOMETRY = {0: (1, 4), 12: (256, 144), 14: (256, 210)}
METADATA = {
    "block_count", "context_length", "embedding_length", "feed_forward_length",
    "attention.head_count", "attention.head_count_kv", "attention.key_length",
    "attention.value_length", "attention.layer_norm_rms_epsilon",
    "full_attention_interval", "rope.dimension_count", "rope.dimension_sections",
    "rope.freq_base", "ssm.conv_kernel", "ssm.group_count", "ssm.inner_size",
    "ssm.state_size", "ssm.time_step_rank",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value, label):
    require(np.isfinite(value).all(), f"nonfinite {label}")
    return value


@dataclass(frozen=True)
class Tensor:
    kind: int
    dims: tuple
    offset: int

    def nbytes(self):
        require(self.kind in GEOMETRY, f"unsupported tensor type {self.kind}")
        block, size = GEOMETRY[self.kind]
        require(self.dims[0] % block == 0, "tensor row splits a quantization block")
        return math.prod(self.dims) // block * size


class GGUF:
    """Header-only index, skipping tokenizer arrays instead of retaining them."""

    def __init__(self, path):
        self.file = open(path, "rb")
        try:
            self.size = os.fstat(self.file.fileno()).st_size
            self.meta, self.tensors = self._header()
        except BaseException:
            self.file.close()
            raise

    def close(self):
        self.file.close()

    def _read(self, size):
        require(0 <= size <= MAX_HEADER_BYTES, "oversized GGUF header field")
        require(self.file.tell() + size <= min(self.size, MAX_HEADER_BYTES),
                "GGUF header is truncated or exceeds 128 MiB")
        raw = self.file.read(size)
        require(len(raw) == size, "short GGUF header read")
        return raw

    def _number(self, fmt):
        return struct.unpack("<" + fmt, self._read(struct.calcsize(fmt)))[0]

    def _string(self, keep=True, limit=16 << 20):
        size = self._number("Q")
        require(size <= limit, "oversized GGUF string")
        if keep:
            return self._read(size).decode("utf-8", "strict")
        require(self.file.tell() + size <= min(self.size, MAX_HEADER_BYTES),
                "GGUF string exceeds header bounds")
        self.file.seek(size, 1)
        return None

    def _value(self, kind, keep):
        if kind == 8:
            return self._string(keep)
        if kind == 9:
            element, count = self._number("I"), self._number("Q")
            require(element in SCALARS or element == 8, "invalid GGUF array type")
            require(count <= 1048576, "oversized GGUF array")
            if not keep and element in SCALARS:
                size = count * struct.calcsize(SCALARS[element])
                require(self.file.tell() + size <= min(self.size, MAX_HEADER_BYTES),
                        "GGUF array exceeds header bounds")
                self.file.seek(size, 1)
                return None
            if keep:
                require(count <= 1024, "oversized architecture metadata array")
                return [self._value(element, True) for _ in range(count)]
            for _ in range(count):
                self._value(element, False)
            return None
        require(kind in SCALARS, f"unsupported GGUF metadata type {kind}")
        result = self._number(SCALARS[kind])
        return result if keep else None

    def _header(self):
        require(self._read(4) == b"GGUF", "not a GGUF file")
        require(self._number("I") in (2, 3), "unsupported GGUF version")
        tensors, keys = self._number("Q"), self._number("Q")
        require(0 < tensors <= 65536 and keys <= 4096, "GGUF index exceeds bounds")
        meta, seen = {}, set()
        for _ in range(keys):
            key = self._string(limit=4096)
            require(key not in seen, f"duplicate GGUF key {key}")
            seen.add(key)
            keep = key.startswith("qwen35.") or key in {
                "general.architecture", "general.alignment", "general.type"}
            value = self._value(self._number("I"), keep)
            if keep:
                meta[key] = value
        infos = {}
        for _ in range(tensors):
            name = self._string(limit=4096)
            require(name not in infos, f"duplicate tensor {name}")
            rank = self._number("I")
            require(1 <= rank <= 4, f"unsupported rank for {name}")
            dims = tuple(self._number("Q") for _ in range(rank))
            require(all(0 < d <= (1 << 32) for d in dims), "invalid tensor dimension")
            kind, offset = self._number("I"), self._number("Q")
            infos[name] = Tensor(kind, dims, offset)
        alignment = meta.get("general.alignment", 32)
        require(type(alignment) is int and 0 < alignment <= 4096 and
                alignment & (alignment - 1) == 0, "invalid GGUF alignment")
        start = (self.file.tell() + alignment - 1) // alignment * alignment
        require(start <= self.size, "missing GGUF data section")
        return meta, {name: Tensor(t.kind, t.dims, start + t.offset)
                      for name, t in infos.items()}

    def check(self, name, dims, kinds):
        require(name in self.tensors, f"missing {name}")
        tensor = self.tensors[name]
        require(tensor.dims == tuple(dims),
                f"{name} dimensions {tensor.dims} != {tuple(dims)}")
        require(tensor.kind in kinds, f"{name} has unsupported type {tensor.kind}")
        require(tensor.offset + tensor.nbytes() <= self.size, f"truncated {name}")
        return tensor

    def read(self, offset, size):
        require(0 <= size <= MAX_TILE_BYTES, "payload read exceeds tile limit")
        require(0 <= offset <= self.size - size, "tensor read exceeds file bounds")
        self.file.seek(offset)
        raw = self.file.read(size)
        require(len(raw) == size, "short tensor payload read")
        return raw


def decode(raw, kind):
    """Decode complete packed blocks, in GGUF element order, directly to f64."""
    require(kind in GEOMETRY, f"unsupported tensor type {kind}")
    require(len(raw) % GEOMETRY[kind][1] == 0, "partial quantization block")
    if kind == 0:
        return finite(np.frombuffer(raw, dtype="<f4").astype(np.float64), "F32 weights")
    stride = GEOMETRY[kind][1]
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, stride)
    out = np.empty((len(b), 256), dtype=np.float64)
    if kind == 12:
        d = b[:, :2].copy().view("<f2").astype(np.float64)
        m = b[:, 2:4].copy().view("<f2").astype(np.float64)
        packed = b[:, 4:16]
        for sub in range(8):
            if sub < 4:
                scale, minimum = packed[:, sub] & 63, packed[:, sub + 4] & 63
            else:
                scale = (packed[:, sub + 4] & 15) | ((packed[:, sub - 4] >> 6) << 4)
                minimum = (packed[:, sub + 4] >> 4) | ((packed[:, sub] >> 6) << 4)
            quants = b[:, 16 + (sub // 2) * 32:48 + (sub // 2) * 32]
            quants = (quants >> (4 * (sub % 2))) & 15
            out[:, sub * 32:(sub + 1) * 32] = d * scale[:, None] * quants - m * minimum[:, None]
    else:
        d = b[:, 208:210].copy().view("<f2").astype(np.float64)
        scales = b[:, 192:208].copy().view(np.int8).astype(np.float64)
        for half in range(2):
            high = b[:, 128 + half * 32:160 + half * 32]
            for group in range(4):
                low_start = half * 64 + (group % 2) * 32
                low = (b[:, low_start:low_start + 32] >> (4 * (group // 2))) & 15
                quant = (low | (((high >> (2 * group)) & 3) << 4)).astype(np.int16) - 32
                scale = np.repeat(scales[:, half * 8 + group * 2:half * 8 + group * 2 + 2], 16, axis=1)
                start = half * 128 + group * 32
                out[:, start:start + 32] = d * scale * quant
    return finite(out.reshape(-1), "dequantized weights")


@dataclass(frozen=True)
class Config:
    dim: int
    ffn: int
    heads: int
    kv_heads: int
    head_dim: int
    rope_dim: int
    rope_base: float
    eps: float
    key_heads: int
    value_heads: int
    state_dim: int
    inner: int
    full_attention: bool

    @classmethod
    def from_metadata(cls, meta, layer):
        require(meta.get("general.architecture") == "qwen35", "requires dense qwen35 GGUF")
        require(meta.get("general.type", "model") == "model", "requires a model GGUF")
        unknown = {k[7:] for k in meta if k.startswith("qwen35.")} - METADATA
        require(not unknown, f"unsupported qwen35 metadata: {sorted(unknown)}")

        def integer(key, lo, hi):
            value = meta.get("qwen35." + key)
            require(type(value) is int and lo <= value <= hi, f"invalid qwen35.{key}")
            return value

        def number(key):
            value = meta.get("qwen35." + key)
            require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                    f"invalid qwen35.{key}")
            return float(value)

        layers = integer("block_count", 2, 256)
        require(0 <= layer < layers, "layer is outside checkpoint")
        interval = integer("full_attention_interval", 2, layers)
        require(layers % interval == 0, "partial hybrid group is unsupported")
        integer("context_length", 1, 1 << 24)
        dim, ffn = integer("embedding_length", 256, 8192), integer("feed_forward_length", 256, 32768)
        require(dim % 256 == 0 and ffn % 256 == 0, "matrix width splits a K quant block")
        heads, kv = integer("attention.head_count", 1, 128), integer("attention.head_count_kv", 1, 128)
        hd = integer("attention.key_length", 1, 256)
        require(heads % kv == 0 and integer("attention.value_length", 1, 256) == hd,
                "requires grouped attention with equal key/value dimensions")
        rd = integer("rope.dimension_count", 2, hd)
        sections = meta.get("qwen35.rope.dimension_sections")
        require(isinstance(sections, list) and len(sections) == 4 and
                all(type(n) is int and n >= 0 for n in sections) and sections[3] == 0 and
                all(n > 0 for n in sections[:3]) and sum(sections) * 2 == rd,
                "unsupported text MRoPE sections")
        require(integer("ssm.conv_kernel", 4, 4) == 4, "requires four-tap convolution")
        kh, vh = integer("ssm.group_count", 1, 128), integer("ssm.time_step_rank", 1, 128)
        sd, inner = integer("ssm.state_size", 1, 256), integer("ssm.inner_size", 256, 32768)
        require(vh % kh == 0 and inner == vh * sd, "unsupported DeltaNet head grouping")
        require(inner % 256 == 0 and heads * hd % 256 == 0, "output matrix width splits a K quant block")
        require(vh * sd * sd * 8 <= MAX_STATE_BYTES, "recurrent state exceeds 32 MiB cap")
        require(2 * kh * sd + inner <= 65536 and 2 * heads * hd <= 65536,
                "projected width exceeds oracle bounds")
        return cls(dim, ffn, heads, kv, hd, rd, number("rope.freq_base"),
                   number("attention.layer_norm_rms_epsilon"), kh, vh, sd, inner,
                   (layer + 1) % interval == 0)


class Weights:
    def __init__(self, gguf, config, layer, tile_rows=16):
        require(type(tile_rows) is int and 1 <= tile_rows <= 64, "tile_rows must be 1..64")
        self.gguf, self.prefix, self.tile_rows = gguf, f"blk.{layer}.", tile_rows
        c = config
        matrices = {"ffn_gate.weight": (c.dim, c.ffn), "ffn_up.weight": (c.dim, c.ffn),
                    "ffn_down.weight": (c.ffn, c.dim)}
        vectors = {"attn_norm.weight": (c.dim,), "post_attention_norm.weight": (c.dim,)}
        if c.full_attention:
            matrices.update({"attn_q.weight": (c.dim, c.heads * c.head_dim * 2),
                             "attn_k.weight": (c.dim, c.kv_heads * c.head_dim),
                             "attn_v.weight": (c.dim, c.kv_heads * c.head_dim),
                             "attn_output.weight": (c.heads * c.head_dim, c.dim)})
            vectors.update({"attn_q_norm.weight": (c.head_dim,), "attn_k_norm.weight": (c.head_dim,)})
        else:
            width = 2 * c.key_heads * c.state_dim + c.inner
            matrices.update({"attn_qkv.weight": (c.dim, width), "attn_gate.weight": (c.dim, c.inner),
                             "ssm_alpha.weight": (c.dim, c.value_heads),
                             "ssm_beta.weight": (c.dim, c.value_heads), "ssm_out.weight": (c.inner, c.dim)})
            vectors.update({"ssm_conv1d.weight": (4, width), "ssm_a": (c.value_heads,),
                            "ssm_dt.bias": (c.value_heads,), "ssm_norm.weight": (c.state_dim,)})
        # Validate every selected tensor before reading ANY selected weight payload.
        self.selected = {name: gguf.check(self.prefix + name, dims, (12, 14))
                         for name, dims in matrices.items()}
        self.selected.update({name: gguf.check(self.prefix + name, dims, (0,))
                              for name, dims in vectors.items()})
        extras = {name[len(self.prefix):] for name in gguf.tensors if name.startswith(self.prefix)} - set(self.selected)
        require(not extras, f"unsupported selected-block tensors: {sorted(extras)}")
        require(sum(t.nbytes() for t in self.selected.values()) <= MAX_LAYER_BYTES,
                "selected block exceeds 1 GiB encoded-weight cap")
        intervals = sorted((t.offset, t.offset + t.nbytes()) for t in self.selected.values())
        require(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
                "selected tensors overlap")
        for name in matrices:
            require(tile_rows * self.selected[name].dims[0] * 8 <= MAX_TILE_BYTES,
                    "dequantized row tile exceeds 16 MiB cap")

    def vector(self, name):
        t = self.selected[name]
        require(t.kind == 0, "small tensor must be F32")
        return decode(self.gguf.read(t.offset, t.nbytes()), 0).reshape(tuple(reversed(t.dims)))

    def project(self, name, x):
        t = self.selected[name]
        cols, rows = t.dims
        require(x.ndim == 2 and x.shape[1] == cols and 1 <= len(x) <= MAX_POSITIONS,
                f"invalid input shape for {name}")
        finite(x, name + " input")
        block, size = GEOMETRY[t.kind]
        stride = cols // block * size
        output = np.empty((len(x), rows), dtype=np.float64)
        for start in range(0, rows, self.tile_rows):
            count = min(self.tile_rows, rows - start)
            tile = decode(self.gguf.read(t.offset + start * stride, count * stride), t.kind)
            tile = tile.reshape(count, cols)
            output[:, start:start + count] = x @ tile.T
        return finite(output, name + " projection")


def sigmoid(x):
    # Stable at both tails without evaluating exp(+large).
    e = np.exp(-np.abs(x))
    return np.where(x >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def silu(x):
    return x * sigmoid(x)


def rmsnorm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def text_rope(x, rope_dim, base):
    """Rotate a prefix's two halves, leaving the remainder of every head intact."""
    half = rope_dim // 2
    theta = np.arange(len(x), dtype=np.float64)[:, None, None] * (
        base ** (-2.0 * np.arange(half, dtype=np.float64) / rope_dim))[None, None, :]
    result = x.copy()
    first, second = x[..., :half], x[..., half:rope_dim]
    result[..., :half] = first * np.cos(theta) - second * np.sin(theta)
    result[..., half:rope_dim] = first * np.sin(theta) + second * np.cos(theta)
    return result


def full_attention(w, c, x, emit):
    n = len(x)
    qg = w.project("attn_q.weight", x).reshape(n, c.heads, 2 * c.head_dim)
    k = w.project("attn_k.weight", x).reshape(n, c.kv_heads, c.head_dim)
    v = w.project("attn_v.weight", x).reshape(n, c.kv_heads, c.head_dim)
    emit("q_gate", qg)
    emit("key", k)
    emit("value", v)
    q = rmsnorm(qg[..., :c.head_dim], w.vector("attn_q_norm.weight"), c.eps)
    k = rmsnorm(k, w.vector("attn_k_norm.weight"), c.eps)
    emit("query_norm", q)
    emit("key_norm", k)
    q, k = text_rope(q, c.rope_dim, c.rope_base), text_rope(k, c.rope_dim, c.rope_base)
    emit("query_rope", q)
    emit("key_rope", k)
    attention = np.zeros((n, c.heads, c.head_dim), dtype=np.float64)
    for t in range(n):
        for head in range(c.heads):
            kv = head // (c.heads // c.kv_heads)
            score = k[:t + 1, kv] @ q[t, head] / math.sqrt(c.head_dim)
            probability = np.exp(score - np.max(score))
            probability /= probability.sum()
            attention[t, head] = probability @ v[:t + 1, kv]
    emit("attn_pre_gate", attention)
    gated = attention * sigmoid(qg[..., c.head_dim:])
    emit("attn_gated", gated)
    return w.project("attn_output.weight", gated.reshape(n, -1))


def delta_attention(w, c, x, emit):
    n, d, kh, vh = len(x), c.state_dim, c.key_heads, c.value_heads
    qkv = w.project("attn_qkv.weight", x)
    z = w.project("attn_gate.weight", x).reshape(n, vh, d)
    alpha, beta_raw = w.project("ssm_alpha.weight", x), w.project("ssm_beta.weight", x)
    emit("qkv", qkv)
    emit("gate", z)
    emit("alpha", alpha)
    emit("beta_raw", beta_raw)
    a = w.vector("ssm_a")
    # -exp(A_log) can underflow to signed zero in the stored F32 tensor; that
    # produces unit decay and is valid. A positive stored value is not valid.
    require(np.all(a <= 0), "ssm_a must already contain non-positive -exp(A_log)")
    decay_log = a * np.logaddexp(0.0, alpha + w.vector("ssm_dt.bias"))
    beta = sigmoid(beta_raw)
    emit("decay_log", decay_log)
    emit("beta", beta)
    kernel = w.vector("ssm_conv1d.weight")
    # Causal cross-correlation: GGUF taps are oldest .. current, not reversed.
    conv = np.zeros_like(qkv)
    for t in range(n):
        for lag in range(min(3, t) + 1):
            conv[t] += qkv[t - lag] * kernel[:, 3 - lag]
    emit("conv_raw", conv)
    conv = silu(conv)
    emit("conv_silu", conv)
    q = conv[:, :kh * d].reshape(n, kh, d)
    k = conv[:, kh * d:2 * kh * d].reshape(n, kh, d)
    v = conv[:, 2 * kh * d:].reshape(n, vh, d)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + c.eps)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + c.eps)
    emit("query_norm", q)
    emit("key_norm", k)
    emit("value", v)
    state = np.zeros((vh, d, d), dtype=np.float64)  # [value head, value dim, key dim]
    recurrent = np.zeros((n, vh, d), dtype=np.float64)
    for t in range(n):
        for head in range(vh):
            key = k[t, head % kh]  # GGUF uses tiled, not contiguous, head repetition.
            state[head] *= math.exp(decay_log[t, head])
            correction = beta[t, head] * (v[t, head] - state[head] @ key)
            state[head] += np.outer(correction, key)
            recurrent[t, head] = state[head] @ q[t, head % kh] / math.sqrt(d)
    emit("attn_pre_gate", recurrent)
    gated = rmsnorm(recurrent, w.vector("ssm_norm.weight"), c.eps) * silu(z)
    emit("attn_gated", gated)
    return w.project("ssm_out.weight", gated.reshape(n, -1))


def block(w, c, hidden, include_stages=False):
    require(hidden.ndim == 2 and hidden.shape[1] == c.dim and
            1 <= len(hidden) <= MAX_POSITIONS, "invalid block input dimensions")
    stages = {}

    def emit(name, value):
        finite(value, name)
        if include_stages:
            stages[name] = value.reshape(len(hidden), -1).copy()

    finite(hidden, "hidden input")
    norm = rmsnorm(hidden, w.vector("attn_norm.weight"), c.eps)
    emit("attn_norm", norm)
    attention = (full_attention if c.full_attention else delta_attention)(w, c, norm, emit)
    emit("attn_output", attention)
    residual = hidden + attention
    emit("post_attention", residual)
    norm = rmsnorm(residual, w.vector("post_attention_norm.weight"), c.eps)
    emit("ffn_norm", norm)
    gate, up = w.project("ffn_gate.weight", norm), w.project("ffn_up.weight", norm)
    emit("ffn_gate", gate)
    emit("ffn_up", up)
    activated = silu(gate) * up
    emit("ffn_activated", activated)
    output = w.project("ffn_down.weight", activated)
    emit("ffn_output", output)
    return finite(residual + output, "block hidden output"), stages


def load_input(path, dim, max_positions):
    with open(path, "rb") as source:
        raw = source.read(MAX_INPUT_BYTES + 1)
    require(len(raw) <= MAX_INPUT_BYTES, "input JSON exceeds 4 MiB cap")
    obj = json.loads(raw)
    require(isinstance(obj, dict) and set(obj) == {"schema_version", "hidden"} and
            type(obj["schema_version"]) is int and obj["schema_version"] == 1,
            "input must contain only schema_version:1 and hidden")
    rows = obj["hidden"]
    require(isinstance(rows, list) and 1 <= len(rows) <= max_positions,
            "input positions exceed explicit limit")
    require(all(isinstance(row, list) and len(row) == dim and
                all(type(v) in (int, float) for v in row) for row in rows),
            "hidden must be a rectangular numeric [positions][embedding_length] array")
    # Match the backend's F32 input boundary exactly once, then keep independent
    # F64 arithmetic for every weight operation and block equation after it.
    result = finite(np.array(rows, dtype=np.float64), "input JSON")
    require(np.all(np.abs(result) <= np.finfo(np.float32).max), "input is outside F32 range")
    with np.errstate(under="ignore"):
        return result.astype(np.float32).astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-positions", required=True, type=int)
    parser.add_argument("--tile-rows", default=16, type=int)
    parser.add_argument("--stages", action="store_true")
    args = parser.parse_args()
    require(1 <= args.max_positions <= MAX_POSITIONS, "max-positions must be 1..8")
    require(1 <= args.tile_rows <= 64, "tile-rows must be 1..64")
    require(args.output.resolve() not in {args.model.resolve(), args.input.resolve()},
            "output must differ from model and input")
    # Exclusive create at the end also refuses existing outputs and hardlink aliases.
    require(not args.output.exists(), "output exists; choose a fresh path")
    gguf = GGUF(args.model)
    try:
        config = Config.from_metadata(gguf.meta, args.layer)
        hidden = load_input(args.input, config.dim, args.max_positions)
        require(len(hidden) <= gguf.meta["qwen35.context_length"], "input exceeds checkpoint context")
        weights = Weights(gguf, config, args.layer, args.tile_rows)
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            output, stages = block(weights, config, hidden, args.stages)
        result = {"schema_version": 1, "layer": args.layer,
                  "block_kind": "full_attention" if config.full_attention else "delta_net",
                  "positions": len(hidden), "hidden": output.tolist()}
        if args.stages:
            result["stages"] = {key: value.tolist() for key, value in stages.items()}
        with open(args.output, "x", encoding="utf-8") as dest:
            json.dump(result, dest, allow_nan=False, separators=(",", ":"))
            dest.write("\n")
        print(f"wrote layer {args.layer} {result['block_kind']}: {len(hidden)} positions to {args.output}")
    finally:
        gguf.close()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, OverflowError, FloatingPointError) as error:
        print(f"qwen35 block reference: {error}", file=sys.stderr)
        sys.exit(1)
