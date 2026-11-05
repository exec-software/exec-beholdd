#!/usr/bin/env python3
"""Freeze a tiny independent regex/BPE oracle from GGUF metadata, never weights.

Requires Python's `regex` package. The expanded ASCII-contraction expressions are
transcribed from llama.cpp b11223 / 4da6337767f973e2b4d0797e5b323d77d8565e4a,
src/llama-vocab.cpp, QWEN2 and QWEN35 branches. Its BPE path does not apply NFC.
Usage: qwen35_oracle.py /absolute/OpenMythos-27B-Q4_K.gguf output.json
"""
import hashlib
import json
import struct
import sys

import regex

QWEN2 = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
QWEN35 = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
CASES = [
    "", "Hello, world!", " leading and  multiple spaces\tline\n",
    "I'm can't won't we're I'd you've shouldn't've I'M 'RE 'ſ",
    "012345 १२३ Ⅳ²", "line1\r\nline2\n\n", " \t\r\n \n  X\t  ",
    "café naïve — 中文🙂", "cafe\u0301 nai\u0308ve", "é", "e\u0301",
    "a\u0301b काि a\u20ddz", "\u0301\u0308abc !\u0301x ?\u0301",
    " \u0301 \u0308!", "!!!\r\n?., — +_test", "✈️ 👩‍💻 𐍈", "a\u00a0b\u2003c",
    "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
    "<think>abc</think><|im_end|>",
]


def header(path):
    # No tensor headers or payload are needed. Read typed metadata then stop.
    with open(path, "rb") as file:
        def unpack(fmt):
            return struct.unpack("<" + fmt, file.read(struct.calcsize("<" + fmt)))[0]

        def string():
            return file.read(unpack("Q")).decode("utf-8")

        def value(kind):
            formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
            if kind in formats:
                return unpack(formats[kind])
            if kind == 8:
                return string()
            if kind == 9:
                item_kind, size = unpack("I"), unpack("Q")
                assert size <= 2_000_000
                return [value(item_kind) for _ in range(size)]
            raise ValueError(f"unsupported GGUF value tag {kind}")

        assert file.read(4) == b"GGUF" and unpack("I") == 3
        unpack("Q")
        metadata_count = unpack("Q")
        assert metadata_count <= 4096
        metadata = {}
        for _ in range(metadata_count):
            key = string()
            metadata[key] = value(unpack("I"))
        position = file.tell()
        file.seek(0)
        digest = hashlib.sha256(file.read(position)).hexdigest()
    return metadata, position, digest


def byte_map():
    visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    extra = [b for b in range(256) if b not in visible]
    mapping = {b: chr(b) for b in visible}
    mapping.update({b: chr(256 + i) for i, b in enumerate(extra)})
    return mapping


def main():
    metadata, size, digest = header(sys.argv[1])
    assert metadata["tokenizer.ggml.model"] == "gpt2"
    assert metadata["tokenizer.ggml.pre"] == "qwen35"
    tokens = metadata["tokenizer.ggml.tokens"]
    kinds = metadata["tokenizer.ggml.token_type"]
    vocabulary = {text: index for index, text in enumerate(tokens)}
    assert len(tokens) == len(kinds) == len(vocabulary)
    ranks = {tuple(pair.split(" ")): rank for rank, pair in enumerate(metadata["tokenizer.ggml.merges"])}
    assert all(len(pair) == 2 for pair in ranks)
    mapping = byte_map()

    def plain(text):
        output = []
        for word in regex.findall(QWEN35, text):
            pieces = [mapping[b] for b in word.encode("utf-8")]
            while len(pieces) > 1:
                candidates = [(ranks[pair], i) for i, pair in enumerate(zip(pieces, pieces[1:])) if pair in ranks]
                if not candidates:
                    break
                _, index = min(candidates)
                pieces[index:index + 2] = [pieces[index] + pieces[index + 1]]
            output.extend(vocabulary[piece] for piece in pieces)
        return output

    def encode(text, parse_special):
        specials = sorted([(token, index) for index, (token, kind) in enumerate(zip(tokens, kinds))
                           if kind == 4 or (parse_special and kind in (2, 3))], key=lambda pair: (-len(pair[0]), pair[1]))
        output = []
        if metadata.get("tokenizer.ggml.add_bos_token", False):
            output.append(metadata["tokenizer.ggml.bos_token_id"])
        while text:
            matches = [(text.index(token), -len(token), index, token) for token, index in specials if token in text]
            if not matches:
                output.extend(plain(text))
                break
            start, _, index, token = min(matches)
            output.extend(plain(text[:start]))
            output.append(index)
            text = text[start + len(token):]
        if metadata.get("tokenizer.ggml.add_eos_token", False):
            output.append(metadata["tokenizer.ggml.eos_token_id"])
        return output

    result = {
        "provenance": {
            "oracle": "independent Python regex + byte-level BPE over GGUF metadata; no NFC",
            "llamaRevision": "4da6337767f973e2b4d0797e5b323d77d8565e4a",
            "ggufMetadataBytes": size, "ggufMetadataSha256": digest,
            "vocabularySize": len(tokens), "mergeCount": len(ranks),
            "regexVersion": regex.__version__, "qwen2Pattern": QWEN2, "qwen35Pattern": QWEN35,
            "tokenPolicy": {k: v for k, v in metadata.items() if k.startswith("tokenizer.ggml.") and not isinstance(v, list)},
        },
        "cases": [{"text": text, "qwen2Words": regex.findall(QWEN2, text),
                   "qwen35Words": regex.findall(QWEN35, text),
                   "specialIds": encode(text, True), "literalIds": encode(text, False)} for text in CASES],
    }
    with open(sys.argv[2], "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
