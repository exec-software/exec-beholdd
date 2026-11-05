# Supported checkpoint chat templates

These files are exact `tokenizer.chat_template` metadata bytes, including their
original final-newline state. Runtime selection compares complete strings; a
model name or architecture tag is not a template identity. The two original Qwen
template files are from Qwen models distributed under Apache-2.0. See the included
[license text](LICENSE-APACHE-2.0), copied from the local system's standard
Apache-2.0 license, and the official upstream licenses for
[Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B/blob/main/LICENSE) and
[Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/blob/main/LICENSE).
Those upstream license files were checked on 2026-10-08. The template bytes below
remain unchanged; this attribution does not claim that today's upstream tokenizer
configuration equals the templates embedded in the qualified GGUF artifacts.
Both upstream license files retain the notice: Copyright 2024 Alibaba Cloud.

- `qwen3-0.6b.jinja`: the locally qualified `Qwen3-0.6B-Q8_0.gguf`, read from
  `$BEHOLDD_MODEL_DIR/qwen3-0.6b-draft/` on 2026-10-08 using only GGUF
  metadata. Length 4,905 bytes; SHA-256
  `5da44855ab7e0641774d5ed0ff0b4f89671d01df7d316e54d06557a962211cb4`.
  Text and function-tool messages are implemented, with unspecified thinking mode.
  Tool definitions use the Transformers v4.51.3 `tojson` filter semantics
  (insertion order, Unicode, and spaces after commas/colons); historical argument
  strings are inserted verbatim after validation. Generated calls use JSON,
  separately from the Coder XML dialect. The independent Jinja
  [oracle fixtures](../../tests/fixtures/qwen3-tools/README.md) pin that contract.
- `qwen3-coder-30b.jinja`: [the qualified Qwen3-Coder-30B-A3B-Instruct-Q8_0 artifact](https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/1f4ceb1041258b3fbfe59e1175d1321c6b41863b/Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf).
  An 8-MiB metadata-prefix request recovered the exact 6,722-byte template;
  SHA-256 `672e747c77e990320152343b0a4951222e40de5645297905d89afba05586d827`.
  It matches the existing native-parity fixture fingerprint
  `a72a487d0e223214` (FNV-1a64). The repository's LFS metadata associates this
  artifact with checkpoint SHA-256
  `a4a0207f4653bfece73d9818c83acf714f5593525fe3aab7026347fd73090fcc`;
  this extraction did not download or rehash the full checkpoint.
- `openmythos-27b.jinja`: exact local OpenMythos-27B-Q4_K.gguf metadata,
  8,057 bytes; SHA-256
  `55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c`.
  The unchanged bytes also match the public
  [upstream template](https://huggingface.co/build-small-hackathon/OpenMythos/blob/2d43d02d7772c916d48bf12b5bbc261211ee60a3/chat_template.jinja).
  Its pinned [model card](https://huggingface.co/build-small-hackathon/OpenMythos/blob/2d43d02d7772c916d48bf12b5bbc261211ee60a3/README.md)
  declares Apache-2.0 and identifies Qwen3.6-27B as its base; that base's
  [license](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/LICENSE)
  retains Copyright 2026 Alibaba Cloud. The included Apache-2.0 license applies.
  The [quantized checkpoint's public metadata](https://huggingface.co/jabbatheduck/OpenMythos-GGUF/blob/4a1a1f97fbfc621d6f33cebc9f3ecc5c12e8c7d7/OpenMythos-27B-Q4_K.gguf)
  matches the recorded 16,547,399,872-byte artifact and SHA-256
  `d3c3398ba8b512aa59074efcaa34d7dcabed7381b68a05ae6bcce77e14d7db45`.
  These metadata and template checks were repeated on 2026-11-05 UTC without
  downloading or rehashing the complete checkpoint.
  Its text-only protocol retains separate `reasoning_content`, default thinking
  and history preservation from the pinned alternate-host llama deployment. It uses XML
  tool calls, ordered arguments, ASCII whitespace trimming and the pinned C++
  JSON numeric spelling. Explicit `enable_thinking` and `preserve_thinking` request
  options, developer instructions and assistant `reasoning_content` are scoped
  to this exact template. Image/audio/video input remains unsupported. Recognition
  does not qualify a checkpoint for native execution or establish its vocabulary
  stop IDs; those are separate runtime admission requirements.

The Rust renderer is intentionally explicit rather than a general Jinja engine.
Unknown or missing native templates are refused. The existing Coder rendering
is preserved; this does not extend its supported tool-schema subset.

Tool definitions and Qwen3 JSON arguments reject duplicate keys and integer
literals outside the exact i64/u64 range. Floating-point numbers must be finite;
integer `-0` follows Python JSON normalization to `0`, while `-0.0` stays a float.
Explicit `reasoning_content` and thinking request options remain unsupported by
the original Qwen3 and Coder protocols. OpenMythos admits signed-i64 integers and
finite floats in `[-2^63,2^63)` for its pinned renderer's numeric conversion domain.
Its six-significant-digit prompt formatting does not modify the request JSON or
the actual values returned in generated tool calls.
