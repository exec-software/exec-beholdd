# Native Qwen3.6 vision replacement plan

**Status: OPEN.** The HTTP compatibility work described here does not implement vision and
does not justify removing a working llama.cpp path. The current runtime capability type is
intentionally completion-only; an unbound model reports no capabilities, and only a
successfully bound model may report `"completion"`. An `image_url` content part receives a
checked `unsupported_content_type` error before prompt rendering or native inference. A
model name, unverified GGUF architecture, or the presence of a projector file must never
turn that claim into `"vision"`.

This plan targets the deployed Qwen3.6-35B-A3B pair: a `qwen35moe` text GGUF and a
separate vision projector whose GGUF metadata identifies a `clip` vision encoder and a
`qwen3vl_merger`. It is a native beholdd implementation plan. It must not proxy, spawn, or
silently fall back to llama.cpp.

## Why the wire change is not the implementation

Antimatter sends an OpenAI chat message whose `content` is an ordered array containing a
text part followed by an `image_url` data URL. Parsing that array is necessary, but the
current runtime accepts only an already-rendered `prompt: String`. Flattening the text and
discarding the image would be a false success. Advertising vision before image bytes,
placement, and position metadata reach an executable model would be a false capability
claim.

The replacement therefore has three independently gated layers: Qwen3.6 text execution,
the vision projector and multimodal position/injection path, and bounded HTTP/runtime
transport. They may land in that order; `"vision"` may be advertised only after all three
are connected for the listed model.

## 1. Qwen3.6 text prerequisite

`qwen35moe` is not an alias for the implemented `qwen3moe` model. It needs a separate,
checked model module and binder. For the deployed 35B-A3B checkpoint that includes:

- the hybrid 40-layer schedule: 30 Gated DeltaNet layers and 10 full-attention layers;
- Qwen3.6 normalization, projection, causal-cache, and multimodal RoPE semantics;
- routed experts plus the shared expert, with routing and reduction order validated;
- the checkpoint's tokenizer/chat-template and special visual token IDs; and
- explicit handling of the trained MTP tensors: either execute them or validate and skip
  them without allowing them to alter ordinary greedy-decoding semantics.

The first gate is text-only. A metadata bind is not enough: fixed prompts must match an
independent implementation at selected intermediate activations, complete logits, greedy
token IDs, and decoded bytes. Resident and streamed execution must remain exact under the
same fit policy. Vision work does not begin by weakening that oracle.

## 2. Vision encoder, merger, and positions

The observed projector is about 861 MiB and contains BF16/F32 tensors for a 27-layer
vision transformer, patch embeddings, learned position embeddings, attention/FFN blocks,
normalization, and the `qwen3vl_merger`. Its relevant metadata includes a 768-pixel image
size, 16-pixel patches, spatial merge size 2, and a 2,048-wide projection into the text
model. Loader validation must check those facts against the text checkpoint rather than
trusting a filename.

The native path needs:

- deterministic image decode, orientation, color conversion, resize, normalization, and
  patchification matching the chosen reference implementation;
- batched matrix multiplication, LayerNorm (including bias where specified), GELU,
  non-causal vision attention, patch embedding, and merger operators with independent CPU
  oracles before optimized backends are used;
- the complete 27-layer encoder and Qwen3-VL merger, with projector output width checked
  against the text model's hidden width;
- replacement of visual placeholder tokens with projected embeddings at the exact message
  position; and
- Qwen3.6 multimodal RoPE indices across text and the spatial/temporal image axes. Ordinary
  text positions and cache continuation must remain unchanged when no image is present.

The projector cannot be treated as a preprocessing convenience outside the fit contract.
Its weights, decoded pixels, patch activations, attention scratch, merged embeddings, and
multimodal KV/recurrent state all count toward admission and peak-memory accounting.

## 3. Runtime, configuration, and HTTP boundary

The runtime request must become structured, for example as ordered prompt parts carrying
text or validated image bytes. Image placement cannot be reconstructed from a flattened
string. The native text-only runtime must reject an image-bearing request even when called
in process; HTTP validation is not a substitute for that invariant.

A per-model configuration field such as `vision_projector_path` should resolve beside the
runtime YAML just like the text checkpoint. Binding must reject a missing projector, a
non-`clip` architecture, a projector without a vision encoder, a merger other than the
implemented `qwen3vl_merger`, incompatible projection/hidden dimensions, unsupported
tensor types, and a storage or memory plan that accounts for only one of the two files.
There should be no process-global `--mmproj` flag in a multi-model registry.

Initially accept only `data:` image URLs. Network fetching would violate the current local,
no-hidden-I/O boundary and introduces a separate security policy. Validate MIME type,
base64 form, decoded byte count, image dimensions, total pixels, and image count before
allocation-heavy decode. Axum's default JSON-body limit is too small for many base64
images, but simply removing it is unsafe: set an explicit encoded-body cap together with
smaller decoded-byte and pixel caps. Base64 adds roughly one third to the payload, and
Antimatter currently scales the long edge to 1,920 pixels while accepting source files up
to 64 MiB, so the two sides need one documented wire limit (or Antimatter must re-encode
the transmitted image to a bounded format).

Only after a model/projector pair passes binding and the structured request reaches that
executable path may its `ModelCapabilities` represent vision and `/v1/models` emit
`["completion", "vision"]`. A configured path, a recognized architecture, or successful
HTTP parsing alone is insufficient.

## Removal gates

The Antimatter compatibility path may be removed only when all of these pass on the exact
deployed artifacts:

1. **Text gate:** Qwen3.6 text-only intermediate values, selected full logits, greedy
   tokens, and decoded bytes match a pinned independent reference; resident and streamed
   modes agree exactly.
2. **Projector gate:** preprocessing fixtures and selected outputs from all 27 vision
   layers plus the merger match a pinned independent reference within stated tolerances.
3. **Multimodal gate:** image-token placement, multimodal RoPE indices, first-token logits,
   and a multi-token greedy completion match the reference for fixed real images and
   prompts; the text-only result remains unchanged.
4. **Resource gate:** admission includes both files and all image/prefill scratch; boundary
   cases for body bytes, decoded bytes, pixels, and image count fail before inference and
   stay under the declared memory limit.
5. **Protocol gate:** the exact Antimatter request shape succeeds through non-streaming
   `/v1/chat/completions`; malformed, remote, unsupported, and oversized image inputs use
   stable OpenAI error envelopes; cancellation is observed during decode, projector, and
   text prefill.
6. **Product gate:** Antimatter discovers the exact model through per-model capabilities,
   performs its real image-description flow against beholdd, and no request reaches
   llama.cpp. Only then may the legacy `/props` probe and direct llama endpoint be removed.

This closes only the vision-specific compatibility path. Replacing the legacy service's
arbitrary GGUF path loading still requires native architecture/quantization coverage and a
separate model-registration/lifecycle decision; this document makes no such claim.
