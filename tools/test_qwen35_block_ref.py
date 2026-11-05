#!/usr/bin/env python3
"""Tiny synthetic checks only. No checkpoint, network, subprocess or GPU access.

Run: python3 tools/test_qwen35_block_ref.py
The scalar packing/recurrence formulas below intentionally use different indexing
from the vectorized oracle. They check conventions, not production Rust results.
"""

from io import BytesIO
import math
import struct
import tempfile
from pathlib import Path
import unittest

import qwen35_block_ref as ref
import numpy as np


def pack_q4(scale, minimum, codes, d=0.125, dmin=0.0625):
    packed = bytearray(144)
    struct.pack_into("<ee", packed, 0, d, dmin)
    for sub in range(8):
        if sub < 4:
            packed[4 + sub] |= scale[sub]
            packed[8 + sub] |= minimum[sub]
        else:
            packed[sub] |= (scale[sub] >> 4) << 6
            packed[4 + sub] |= (minimum[sub] >> 4) << 6
            packed[8 + sub] = (scale[sub] & 15) | ((minimum[sub] & 15) << 4)
    for index, code in enumerate(codes):
        group, lane = divmod(index, 32)
        packed[16 + (group // 2) * 32 + lane] |= code << (4 * (group % 2))
    expected = [d * scale[i // 32] * code - dmin * minimum[i // 32]
                for i, code in enumerate(codes)]
    return bytes(packed), expected


def pack_q6(scale, codes, d=0.125):
    packed = bytearray(210)
    for index, code in enumerate(codes):
        half, within = divmod(index, 128)
        group, lane = divmod(within, 32)
        low_index = half * 64 + (group % 2) * 32 + lane
        high_index = 128 + half * 32 + lane
        packed[low_index] |= (code & 15) << (4 * (group // 2))
        packed[high_index] |= (code >> 4) << (2 * group)
    struct.pack_into("<16b", packed, 192, *scale)
    struct.pack_into("<e", packed, 208, d)
    expected = [d * scale[i // 16] * (code - 32) for i, code in enumerate(codes)]
    return bytes(packed), expected


class PackedWeightsTests(unittest.TestCase):
    def test_q4_all_scale_min_high_bits_and_nibbles(self):
        raw, expected = pack_q4([1, 17, 32, 63, 61, 33, 18, 7],
                               [62, 31, 16, 4, 3, 19, 34, 63],
                               [(i * 7 + i // 32) % 16 for i in range(256)])
        np.testing.assert_array_equal(ref.decode(raw, 12), expected)
        # Multiple blocks and the smallest positive F16 must retain boundaries.
        tiny, exp_tiny = pack_q4(list(range(8)), list(range(8)), [15] * 256,
                                d=2 ** -24, dmin=2 ** -24)
        np.testing.assert_array_equal(ref.decode(raw + tiny, 12), expected + exp_tiny)

    def test_q6_signed_scale_high_bits_and_subblocks(self):
        scale = [-128, -97, -63, -32, -1, 0, 1, 17, 31, 63, 95, 127, -2, 3, -4, 5]
        raw, expected = pack_q6(scale, [(i * 13 + i // 16) % 64 for i in range(256)])
        np.testing.assert_array_equal(ref.decode(raw, 14), expected)
        tiny, exp_tiny = pack_q6(scale[::-1], list(range(64)) * 4, d=2 ** -24)
        np.testing.assert_array_equal(ref.decode(tiny + raw, 14), exp_tiny + expected)

    def test_bad_block_and_nonfinite_scales_refused(self):
        for kind, length in ((0, 3), (12, 143), (14, 209)):
            with self.assertRaisesRegex(ValueError, "partial"):
                ref.decode(bytes(length), kind)
        raw, _ = pack_q4([1] * 8, [1] * 8, [1] * 256)
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            ref.decode(struct.pack("<e", math.inf) + raw[2:], 12)

    def test_row_tiling_preserves_multiple_blocks_rows_and_positions(self):
        for kind, pack in ((12, lambda j: pack_q4([j + 1] * 8, [2] * 8,
                                                 [(i + j) % 16 for i in range(256)])),
                           (14, lambda j: pack_q6([j - 3] * 16,
                                                 [(i * 3 + j) % 64 for i in range(256)]))):
            blocks = [pack(i) for i in range(6)]  # 3 rows, 512 columns.
            raw = b"".join(b[0] for b in blocks)
            matrix = np.array([n for b in blocks for n in b[1]], dtype=np.float64).reshape(3, 512)
            gguf = ref.GGUF.__new__(ref.GGUF)
            gguf.file, gguf.size = BytesIO(raw), len(raw)
            weights = ref.Weights.__new__(ref.Weights)
            weights.gguf = gguf
            weights.selected = {"matrix": ref.Tensor(kind, (512, 3), 0)}
            x = np.array([[(i % 7 - 3) / 8 for i in range(512)],
                          [(i % 5 - 2) / 4 for i in range(512)]])
            for tile_rows in (1, 2, 3):
                weights.tile_rows = tile_rows
                np.testing.assert_array_equal(weights.project("matrix", x), x @ matrix.T)
            gguf.file = BytesIO(raw[:-1])  # Stat-size snapshot cannot hide short reads.
            with self.assertRaisesRegex(ValueError, "short"):
                weights.project("matrix", x)


class FakeWeights:
    def __init__(self, projections, vectors):
        self.projections, self.vectors = projections, vectors

    def project(self, name, x):
        if name in ("attn_output.weight", "ssm_out.weight"):
            return x.copy()
        return self.projections[name].copy()

    def vector(self, name):
        return self.vectors[name].copy()


def scalar_sigmoid(x):
    return 1 / (1 + math.exp(-x))


def scalar_rms(x, weights, eps):
    den = math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v / den * w for v, w in zip(x, weights)]


class ArchitectureTests(unittest.TestCase):
    def test_complete_block_residual_and_post_attention_ffn_order(self):
        c = ref.Config(4, 4, 2, 1, 2, 2, 100.0, 0.125, 1, 2, 2, 4, True)
        hidden = np.array([[1.0, -0.5, 0.75, 2.0]])
        first_norm, second_norm = np.array([1.0, -0.5, 1.5, 0.75]), np.array([0.5, 1.5, -1.0, 2.0])

        class BlockWeights:
            def vector(self, name):
                return {"attn_norm.weight": first_norm,
                        "post_attention_norm.weight": second_norm,
                        "attn_q_norm.weight": np.ones(2),
                        "attn_k_norm.weight": np.ones(2)}[name]

            def project(self, name, x):
                if name == "attn_q.weight":
                    return np.zeros((1, 8))  # Zero gates give sigmoid = 1/2.
                if name == "attn_k.weight":
                    return np.zeros((1, 2))
                if name == "attn_v.weight":
                    return x[:, :2].copy()
                if name == "ffn_gate.weight":
                    return x * np.array([1.5, -0.75, 2.0, 0.5])
                if name == "ffn_up.weight":
                    return x[:, ::-1].copy()
                return x.copy()  # Attention and FFN output matrices are identities.

        normalized = scalar_rms(hidden[0], first_norm, c.eps)
        residual = [hidden[0, i] + normalized[i % 2] * 0.5 for i in range(4)]
        ff = scalar_rms(residual, second_norm, c.eps)
        gate = [ff[i] * [1.5, -0.75, 2.0, 0.5][i] for i in range(4)]
        expected = [residual[i] + gate[i] * scalar_sigmoid(gate[i]) * ff[3 - i] for i in range(4)]
        output, stages = ref.block(BlockWeights(), c, hidden, True)
        np.testing.assert_allclose(output[0], expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(stages["post_attention"][0], residual, rtol=1e-13)
        np.testing.assert_allclose(stages["ffn_norm"][0], ff, rtol=1e-13)

    def test_full_attention_partial_rope_per_head_gate_and_gqa(self):
        c = ref.Config(24, 12, 4, 2, 6, 4, 100.0, 0.03125, 2, 4, 2, 8, True)
        qg = np.array([[[((t * 19 + h * 7 + j * 3) % 23 - 11) / 8
                         for j in range(12)] for h in range(4)] for t in range(3)])
        keys = np.array([[[((t * 11 + h * 5 + j * 2) % 17 - 8) / 4
                           for j in range(6)] for h in range(2)] for t in range(3)])
        values = np.array([[[((t * 3 + h * 13 + j * 7) % 19 - 9) / 8
                             for j in range(6)] for h in range(2)] for t in range(3)])
        qw, kw = [1, -0.5, 1.5, 0.75, 2, -1], [-1, 2, 0.5, 1.25, 0.75, 1.5]
        weights = FakeWeights({"attn_q.weight": qg, "attn_k.weight": keys,
                               "attn_v.weight": values},
                              {"attn_q_norm.weight": np.array(qw), "attn_k_norm.weight": np.array(kw)})

        def rotated(vec, norm, t):
            y = scalar_rms(vec, norm, c.eps)
            # Explicit independent frequency pairs: coords 0/2 and 1/3;
            # unrotated 4/5 must remain normalized but otherwise unchanged.
            for a, b, frequency in ((0, 2, 1.0), (1, 3, 0.1)):
                olda, oldb = y[a], y[b]
                y[a] = olda * math.cos(t * frequency) - oldb * math.sin(t * frequency)
                y[b] = olda * math.sin(t * frequency) + oldb * math.cos(t * frequency)
            return y

        expected = []
        for t in range(3):
            row = []
            for h in range(4):
                q = rotated(qg[t, h, :6], qw, t)
                scores = [sum(a * b for a, b in zip(q, rotated(keys[s, h // 2], kw, s))) / math.sqrt(6)
                          for s in range(t + 1)]
                probs = [math.exp(s - max(scores)) for s in scores]
                total = sum(probs)
                for j in range(6):
                    attn = sum(probs[s] / total * values[s, h // 2, j] for s in range(t + 1))
                    row.append(attn * scalar_sigmoid(qg[t, h, 6 + j]))
            expected.append(row)
        output = ref.full_attention(weights, c, np.zeros((3, 24)), lambda *_: None)
        np.testing.assert_allclose(output, expected, rtol=1e-13, atol=1e-14)

    def test_delta_four_taps_tiled_heads_decay_and_gate_order(self):
        c = ref.Config(8, 12, 4, 2, 2, 2, 100.0, 0.0625, 2, 4, 2, 8, False)
        n, channels = 5, 16
        qkv = np.array([[((t * 7 + j * 3) % 19 - 9) / 8 for j in range(channels)] for t in range(n)])
        gate = np.array([[((t + j * 3) % 13 - 6) / 8 for j in range(8)] for t in range(n)])
        alpha = np.array([[(t * 3 - h * 2) / 8 for h in range(4)] for t in range(n)])
        beta = np.array([[(h * 3 - t * 2) / 8 for h in range(4)] for t in range(n)])
        kernel = np.array([[((j * 3 + tap * 2) % 11 - 5) / 8 for tap in range(4)] for j in range(channels)])
        a, bias, norm = [-0.125, -0.5, -1.5, -2.0], [0.25, -0.5, 0.75, -1.0], [1.5, -0.75]
        weights = FakeWeights({"attn_qkv.weight": qkv, "attn_gate.weight": gate,
                               "ssm_alpha.weight": alpha, "ssm_beta.weight": beta},
                              {"ssm_a": np.array(a), "ssm_dt.bias": np.array(bias),
                               "ssm_conv1d.weight": kernel, "ssm_norm.weight": np.array(norm)})
        states = [[[0.0, 0.0], [0.0, 0.0]] for _ in range(4)]
        expected = []
        for t in range(n):
            conv = []
            for channel in range(channels):
                mixed = sum(float(qkv[s, channel]) * float(kernel[channel, s - t + 3])
                            for s in range(max(0, t - 3), t + 1))
                conv.append(mixed * scalar_sigmoid(mixed))
            q = [conv[0:2], conv[2:4]]
            k = [conv[4:6], conv[6:8]]
            for vec in q + k:
                scale = math.sqrt(sum(v * v for v in vec) + c.eps)
                vec[:] = [v / scale for v in vec]
            row = []
            for h in range(4):
                factor = math.exp(a[h] * math.log1p(math.exp(alpha[t, h] + bias[h])))
                b = scalar_sigmoid(beta[t, h])
                # Scalar state equation computes each coordinate from the previous
                # state, independently of the oracle's vectorized outer product.
                previous = [[v * factor for v in coord] for coord in states[h]]
                for value in range(2):
                    prediction = sum(previous[value][key] * k[h % 2][key] for key in range(2))
                    correction = b * (conv[8 + h * 2 + value] - prediction)
                    for key in range(2):
                        states[h][value][key] = previous[value][key] + correction * k[h % 2][key]
                output = [sum(states[h][value][key] * q[h % 2][key] for key in range(2)) / math.sqrt(2)
                          for value in range(2)]
                output = scalar_rms(output, norm, c.eps)
                for j in range(2):
                    z = gate[t, h * 2 + j]
                    row.append(output[j] * z * scalar_sigmoid(z))
            expected.append(row)
        output = ref.delta_attention(weights, c, np.zeros((n, 8)), lambda *_: None)
        np.testing.assert_allclose(output, expected, rtol=1e-12, atol=1e-14)

    def test_extreme_stable_gates(self):
        x = np.array([-np.finfo(np.float32).max, -1000, 0, 1000, np.finfo(np.float32).max])
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            np.testing.assert_array_equal(ref.sigmoid(x), [0, 0, 0.5, 1, 1])
            np.testing.assert_array_equal(ref.silu(x), [0, 0, 0, 1000, np.finfo(np.float32).max])

    def test_delta_zero_a_has_unit_decay_and_positive_a_is_refused(self):
        c = ref.Config(1, 1, 1, 1, 2, 2, 100.0, 0.125, 1, 1, 1, 1, False)
        weights = FakeWeights({"attn_qkv.weight": np.ones((2, 3)),
                               "attn_gate.weight": np.ones((2, 1)),
                               "ssm_alpha.weight": np.full((2, 1), 2.0),
                               "ssm_beta.weight": np.zeros((2, 1))},
                              {"ssm_a": np.array([0.0]), "ssm_dt.bias": np.zeros(1),
                               "ssm_conv1d.weight": np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
                               "ssm_norm.weight": np.ones(1)})
        value = scalar_sigmoid(1.0)  # SiLU(1).
        key = value / math.sqrt(value * value + c.eps)
        state = 0.0
        expected = []
        for _ in range(2):
            # Unit decay retains the previous state in full before correction.
            state += key * 0.5 * (value - state * key)
            core = state * key
            expected.append([core / math.sqrt(core * core + c.eps) * value])
        for stored_a in (0.0, -0.0):
            weights.vectors["ssm_a"] = np.array([stored_a])
            actual = ref.delta_attention(weights, c, np.zeros((2, 1)), lambda *_: None)
            np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-14)
        weights.vectors["ssm_a"] = np.array([1e-30])
        with self.assertRaisesRegex(ValueError, "non-positive"):
            ref.delta_attention(weights, c, np.zeros((2, 1)), lambda *_: None)


class InputBoundsTests(unittest.TestCase):
    def test_decimal_inputs_round_once_to_f32_before_f64_math(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            path.write_text('{"schema_version":1,"hidden":[[0.1,-0.2,1.00000006,16777217]]}')
            actual = ref.load_input(path, 4, 1)
            # Independent IEEE-754 bit patterns, not another NumPy conversion.
            expected = [[struct.unpack("<f", struct.pack("<I", bits))[0]
                         for bits in (0x3DCCCCCD, 0xBE4CCCCD, 0x3F800001, 0x4B800000)]]
            self.assertEqual(actual.dtype, np.dtype(np.float64))
            np.testing.assert_array_equal(actual, expected)

    def test_truncated_header_and_oversized_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "synthetic.gguf"
            for raw in (b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 65537, 0)):
                path.write_bytes(raw)
                with self.assertRaises(ValueError):
                    ref.GGUF(path)

    def test_strict_input_shapes_limits_and_finite_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.json"
            for text in ('{"schema_version":1,"hidden":[[NaN,0]]}',
                         '{"schema_version":1,"hidden":[[1,0],[1,0]]}',
                         '{"schema_version":1,"hidden":[[true,0]]}',
                         '{"schema_version":1,"hidden":[[1]]}',
                         '{"schema_version":1,"hidden":[[1,0]],"state":[]}'):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    ref.load_input(path, 2, 1)


if __name__ == "__main__":
    unittest.main()
