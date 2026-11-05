#!/usr/bin/env python3
"""Validate and byte-compare two exact logit-capture files.

usage: compare_logits.py <reference.bin> <candidate.bin>
"""

import hashlib
import math
import struct
import sys
from dataclasses import dataclass


MAGIC = b"BHDLOGIT"
FOOTER_MAGIC = b"BHDLEND!"
VERSION = 1
HEADER_LEN = 64
SCALAR_ENCODING_F32 = 1
FOOTER_LEN = 16
MAX_U64 = (1 << 64) - 1
MAX_TOKEN_ID_SPACE = 1 << 32

HEADER = struct.Struct("<8sHHIQQQQQQ")
FOOTER = struct.Struct("<Q8s")


@dataclass(frozen=True)
class Capture:
    path: str
    data: bytes
    prompt_len: int
    step_count: int
    vocab: int
    logit_count: int
    body_len: int
    prompt_ids: tuple
    generated_ids: tuple

    @property
    def descriptor(self):
        return (
            self.prompt_len,
            self.step_count,
            self.vocab,
            self.logit_count,
            self.body_len,
            self.prompt_ids,
            self.generated_ids,
        )

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def checked_add(label, *values):
    result = sum(values)
    if result > MAX_U64:
        raise ValueError(f"{label} overflows u64")
    return result


def checked_mul(label, left, right):
    if left and right > MAX_U64 // left:
        raise ValueError(f"{label} overflows u64")
    return left * right


def unpack_u32s(data, start, count):
    end = start + count * 4
    return tuple(value for (value,) in struct.iter_unpack("<I", data[start:end]))


def read_capture(path):
    try:
        with open(path, "rb") as source:
            data = source.read()
    except OSError as error:
        raise ValueError(f"{path}: cannot read file: {error}") from error

    if len(data) < HEADER_LEN:
        raise ValueError(
            f"{path}: truncated header: {len(data)} bytes, need {HEADER_LEN}"
        )

    (
        magic,
        version,
        header_len,
        scalar_encoding,
        prompt_len,
        step_count,
        vocab,
        logit_count,
        body_len,
        reserved,
    ) = HEADER.unpack_from(data)

    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")
    if version != VERSION:
        raise ValueError(f"{path}: unsupported version {version}, expected {VERSION}")
    if header_len != HEADER_LEN:
        raise ValueError(
            f"{path}: invalid header length {header_len}, expected {HEADER_LEN}"
        )
    if scalar_encoding != SCALAR_ENCODING_F32:
        raise ValueError(
            f"{path}: unsupported scalar encoding {scalar_encoding}, "
            f"expected {SCALAR_ENCODING_F32} (f32)"
        )
    if reserved != 0:
        raise ValueError(f"{path}: reserved header field is nonzero ({reserved})")
    if prompt_len == 0:
        raise ValueError(f"{path}: prompt length must be nonzero")
    if step_count == 0:
        raise ValueError(f"{path}: step count must be nonzero")
    if vocab == 0:
        raise ValueError(f"{path}: vocabulary size must be nonzero")
    if vocab > MAX_TOKEN_ID_SPACE:
        raise ValueError(
            f"{path}: implausible vocabulary size {vocab}; u32 token IDs can "
            f"address at most {MAX_TOKEN_ID_SPACE} entries"
        )

    expected_logits = checked_mul(f"{path}: step_count * vocab", step_count, vocab)
    if logit_count != expected_logits:
        raise ValueError(
            f"{path}: logit count {logit_count} does not equal "
            f"steps * vocab ({step_count} * {vocab} = {expected_logits})"
        )

    body_words = checked_add(
        f"{path}: body element count", prompt_len, step_count, logit_count
    )
    expected_body_len = checked_mul(f"{path}: body byte length", body_words, 4)
    if body_len != expected_body_len:
        raise ValueError(
            f"{path}: body length {body_len} does not equal "
            f"4 * (prompt + steps + logits) ({expected_body_len})"
        )

    expected_total = checked_add(
        f"{path}: total byte length", HEADER_LEN, body_len, FOOTER_LEN
    )
    if expected_total > sys.maxsize:
        raise ValueError(
            f"{path}: declared total length {expected_total} exceeds platform limits"
        )
    if len(data) != expected_total:
        relation = "truncated" if len(data) < expected_total else "has trailing bytes"
        raise ValueError(
            f"{path}: {relation}: file is {len(data)} bytes, "
            f"header requires exactly {expected_total}"
        )

    footer_offset = HEADER_LEN + body_len
    footer_total, footer_magic = FOOTER.unpack_from(data, footer_offset)
    if footer_total != expected_total:
        raise ValueError(
            f"{path}: footer total length {footer_total} does not equal "
            f"exact file length {expected_total}"
        )
    if footer_magic != FOOTER_MAGIC:
        raise ValueError(
            f"{path}: bad footer magic {footer_magic!r}, expected {FOOTER_MAGIC!r}"
        )

    prompt_offset = HEADER_LEN
    generated_offset = prompt_offset + prompt_len * 4
    logits_offset = generated_offset + step_count * 4
    prompt_ids = unpack_u32s(data, prompt_offset, prompt_len)
    generated_ids = unpack_u32s(data, generated_offset, step_count)

    for label, token_ids in (
        ("prompt", prompt_ids),
        ("generated", generated_ids),
    ):
        for index, token_id in enumerate(token_ids):
            if token_id >= vocab:
                raise ValueError(
                    f"{path}: {label} token {index} has ID {token_id}, "
                    f"outside vocabulary [0, {vocab})"
                )

    logits_end = logits_offset + logit_count * 4
    for index, (value,) in enumerate(
        struct.iter_unpack("<f", data[logits_offset:logits_end])
    ):
        if not math.isfinite(value):
            bits = struct.unpack_from("<I", data, logits_offset + index * 4)[0]
            step, token = divmod(index, vocab)
            raise ValueError(
                f"{path}: non-finite logit at step {step}, token {token} "
                f"(bits 0x{bits:08x})"
            )

    for step, generated_id in enumerate(generated_ids):
        row_offset = logits_offset + step * vocab * 4
        best_token = 0
        best_logit = struct.unpack_from("<f", data, row_offset)[0]
        for token in range(1, vocab):
            value = struct.unpack_from("<f", data, row_offset + token * 4)[0]
            if value > best_logit:
                best_token = token
                best_logit = value
        if generated_id != best_token:
            raise ValueError(
                f"{path}: generated token at step {step} is {generated_id}, "
                f"but the logits' deterministic argmax is {best_token}"
            )

    return Capture(
        path=path,
        data=data,
        prompt_len=prompt_len,
        step_count=step_count,
        vocab=vocab,
        logit_count=logit_count,
        body_len=body_len,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
    )


def first_sequence_difference(reference, candidate):
    for index, (left, right) in enumerate(zip(reference, candidate)):
        if left != right:
            return index, left, right
    return None


def compare_descriptors(reference, candidate):
    scalar_fields = (
        ("prompt length", reference.prompt_len, candidate.prompt_len),
        ("step count", reference.step_count, candidate.step_count),
        ("vocabulary size", reference.vocab, candidate.vocab),
        ("logit count", reference.logit_count, candidate.logit_count),
        ("body length", reference.body_len, candidate.body_len),
    )
    for label, left, right in scalar_fields:
        if left != right:
            raise ValueError(f"descriptor mismatch: {label} differs: {left} vs {right}")

    for label, left, right in (
        ("prompt IDs", reference.prompt_ids, candidate.prompt_ids),
        ("generated IDs", reference.generated_ids, candidate.generated_ids),
    ):
        difference = first_sequence_difference(left, right)
        if difference is not None:
            index, left_value, right_value = difference
            raise ValueError(
                f"descriptor mismatch: {label} differ at index {index}: "
                f"{left_value} vs {right_value}"
            )


def first_byte_difference(reference, candidate):
    chunk_size = 1024 * 1024
    for start in range(0, min(len(reference), len(candidate)), chunk_size):
        left_chunk = reference[start : start + chunk_size]
        right_chunk = candidate[start : start + chunk_size]
        if left_chunk != right_chunk:
            for relative, (left, right) in enumerate(zip(left_chunk, right_chunk)):
                if left != right:
                    return start + relative, left, right
    return None


def compare_bytes(reference, candidate):
    if reference.data == candidate.data:
        return

    difference = first_byte_difference(reference.data, candidate.data)
    hashes = (
        f"reference SHA-256 {reference.sha256}, "
        f"candidate SHA-256 {candidate.sha256}"
    )
    if difference is None:
        raise ValueError(
            f"file bytes differ in length: {len(reference.data)} vs "
            f"{len(candidate.data)}; {hashes}"
        )
    offset, left, right = difference
    raise ValueError(
        f"file bytes differ at offset {offset}: reference 0x{left:02x}, "
        f"candidate 0x{right:02x}; {hashes}"
    )


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit("usage: compare_logits.py <reference.bin> <candidate.bin>")

    try:
        reference = read_capture(argv[0])
        candidate = read_capture(argv[1])
        compare_descriptors(reference, candidate)
        compare_bytes(reference, candidate)
    except ValueError as error:
        raise SystemExit(f"FAIL: {error}") from None

    print(
        f"  shape: prompt={reference.prompt_len}, steps={reference.step_count}, "
        f"vocab={reference.vocab}, logits=({reference.step_count}, {reference.vocab})"
    )
    print(f"  prompt IDs: {list(reference.prompt_ids)}")
    print(f"  generated IDs: {list(reference.generated_ids)}")
    print(f"  SHA-256: {reference.sha256}")
    print("  Exact logit comparison: PASS")


if __name__ == "__main__":
    main()
