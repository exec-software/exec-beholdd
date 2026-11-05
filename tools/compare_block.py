#!/usr/bin/env python3
"""Compare a Rust Qwen3-MoE block result with the independent f64 interchange.

usage: compare_block.py <reference.bin> <candidate.bin>
"""

import struct
import sys

import numpy as np


MAX_ROUTING_ABS = 1e-6
# Relative L2 is the normative gate. Keep a small absolute guard too, with enough room
# that a one-ULP libm difference on another host does not make the gate platform brittle.
MAX_HIDDEN_ABS = 2e-6
MAX_HIDDEN_REL_L2 = 1e-6


def read(path):
    data = open(path, "rb").read()
    if len(data) < 12:
        raise ValueError(f"{path}: truncated header")
    npos, dim, topk = struct.unpack_from("<3i", data)
    if npos <= 0 or dim <= 0 or topk <= 0:
        raise ValueError(f"{path}: invalid shape {(npos, dim, topk)}")

    offset = 12

    def take(dtype, count, label):
        nonlocal offset
        nbytes = np.dtype(dtype).itemsize * count
        end = offset + nbytes
        if end > len(data):
            raise ValueError(f"{path}: truncated {label}")
        values = np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy()
        offset = end
        return values

    tokens = take("<i4", npos, "tokens")
    experts = take("<i4", npos * topk, "experts").reshape(npos, topk)
    weights = take("<f4", npos * topk, "routing weights").reshape(npos, topk)
    hidden = take("<f4", npos * dim, "hidden states").reshape(npos, dim)
    if offset != len(data):
        raise ValueError(f"{path}: {len(data) - offset} trailing bytes")
    return (npos, dim, topk), tokens, experts, weights, hidden


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: compare_block.py <reference.bin> <candidate.bin>")
    reference = read(sys.argv[1])
    candidate = read(sys.argv[2])

    failures = []
    if reference[0] != candidate[0]:
        raise SystemExit(f"FAIL: shape differs: {reference[0]} vs {candidate[0]}")
    if not np.array_equal(reference[1], candidate[1]):
        failures.append("token IDs differ")
    if not np.array_equal(reference[2], candidate[2]):
        failures.append(
            f"selected experts differ:\nreference {reference[2]}\ncandidate {candidate[2]}"
        )

    routing_diff = candidate[3].astype(np.float64) - reference[3].astype(np.float64)
    hidden_diff = candidate[4].astype(np.float64) - reference[4].astype(np.float64)
    if not np.all(np.isfinite(candidate[3])):
        failures.append("candidate routing weights contain non-finite values")
    if not np.all(np.isfinite(candidate[4])):
        failures.append("candidate hidden states contain non-finite values")
    routing_max_abs = float(np.max(np.abs(routing_diff)))
    hidden_max_abs = float(np.max(np.abs(hidden_diff)))
    reference_norm = float(np.linalg.norm(reference[4].astype(np.float64)))
    difference_norm = float(np.linalg.norm(hidden_diff))
    hidden_rel_l2 = difference_norm / reference_norm if reference_norm else (
        0.0 if difference_norm == 0.0 else float("inf")
    )
    hidden_rms_diff = float(np.sqrt(np.mean(hidden_diff * hidden_diff)))

    print(f"  exact token IDs:       {np.array_equal(reference[1], candidate[1])}")
    print(f"  exact selected experts: {np.array_equal(reference[2], candidate[2])}")
    print(f"  routing max abs:       {routing_max_abs:.9e}  (limit {MAX_ROUTING_ABS:.1e})")
    print(f"  hidden max abs:        {hidden_max_abs:.9e}  (limit {MAX_HIDDEN_ABS:.1e})")
    print(f"  hidden relative L2:    {hidden_rel_l2:.9e}  (limit {MAX_HIDDEN_REL_L2:.1e})")
    print(f"  hidden RMS difference: {hidden_rms_diff:.9e}")

    if not np.isfinite(routing_max_abs) or routing_max_abs > MAX_ROUTING_ABS:
        failures.append(f"routing max abs {routing_max_abs:.9e} exceeds {MAX_ROUTING_ABS:.1e}")
    if not np.isfinite(hidden_max_abs) or hidden_max_abs > MAX_HIDDEN_ABS:
        failures.append(f"hidden max abs {hidden_max_abs:.9e} exceeds {MAX_HIDDEN_ABS:.1e}")
    if not np.isfinite(hidden_rel_l2) or hidden_rel_l2 > MAX_HIDDEN_REL_L2:
        failures.append(
            f"hidden relative L2 {hidden_rel_l2:.9e} exceeds {MAX_HIDDEN_REL_L2:.1e}"
        )
    if failures:
        raise SystemExit("FAIL:\n  " + "\n  ".join(failures))
    print("  Gate M1.1: PASS")


if __name__ == "__main__":
    main()
