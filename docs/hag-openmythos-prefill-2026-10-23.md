# Qwen3.5 state-only prefill checkpoint — 2026-10-23

Source revision `eca282e` skips the unused final normalization and vocabulary
projection for non-final prompt tokens. Embedding, every attention/DeltaNet core,
every FFN, and their consumed-activation checks still execute. Final prompt and
decode positions retain the full checked head and vocabulary readback.

State advancement invalidates earlier logits tickets and publishes only a completed
state position. Interrupted mutation remains poisoned until a physical reset;
transport health and the process health latch remain authoritative. This change
does not increase context or memory limits.

## Validation

A serial guarded host run passed 70 model tests, 80 runtime tests and 94 server/doc
tests. Six tests requiring explicit fixtures or hardware profiles remained ignored. The fixed native
probe compiled, all three release binaries built, and metadata-only preflight
passed without creating a device owner or output directory. Peak host-build memory
was 1,957,097,472 bytes; maximum Tctl was 63.125°C. Memory events, swap and host OOM
counters were zero, and the build scope retired completely.

The new `hag_qwen35_advance_probe` then passed on alternate-host's AMD 1002:1586,
`/dev/dri/renderD128`, kernel `7.0.0-38-generic`. It uses the existing nonzero
four-layer hybrid fixture, vocabulary 512, capacity four and prompt `[7, 42, 7]`.
The decode input is the baseline's selected prompt-end token.

- Optimized, mixed and cancellation-recovered sequences matched the full-head
  baseline bitwise across all 512 prefill logits and all 512 decode logits.
- Every sequence selected token 213 at prompt end and token 478 after decode.
- Stale-ticket and poisoned-retry refusal, cancellation after a fenced embedding
  tile, five physical resets and clean native-resource retirement passed.
- Actual accounting matched eleven full steps, five state-only advances, one
  cancelled embedding tile and nine complete vocabulary readbacks.
- Each state-only step used 131 packed tiles / 1,367,376 bytes / 252 dispatches,
  versus 139 tiles / 1,474,896 bytes / 264 dispatches for a full step.

The native probe completed in 0.256 seconds. Its cgroup peaked at 17,694,720 bytes;
the separately admitted native device charge was 1,007,616 bytes and the complete
process forecast was 138,310,298 bytes. Maximum sampled CPU/GPU temperatures were
49.75°C / 45°C. No memory events, swap, host OOM, GPU fault or crash capture occurred.
Both scopes retired, and the authenticated read-only watcher stopped with zero
captures and its process gone.

## Evidence and limits

Local evidence is under `work/openmythos-20261022/prefill-advance-v1` in the HAG task
workspace. The frozen manifest SHA256 is
`df267efeff0b16c8b4f12317a4b1159b002deb11f6beb5cbffadff0dcd05cec8`.
The guarded preflight result SHA256 is
`520adab2ea1d6505fdd15d1e36c2019a3752d5e5d5e1e60fbf35a509472a67e5`;
the native result SHA256 is
`7b8738bc6cc64ec182d4f5fe74f4b9dd9eca81aef90d76e2943ec1fd76ed47cf`.
`closeout.json` binds complete scope and watcher retirement.

This establishes equivalence to the existing native sequence on the tiny fixture.
It is not independent CPU sequence accuracy, full-model performance, serving or
normal Antimatter prompt qualification. No OpenMythos deployment changed. The
earlier full-model numerical and bounded serving results are recorded separately
in `hag-openmythos-2026-10-23.md` and apply to their tested revision.

The next prefill work is bounded FFN weight-tile reuse across multiple prompt
tokens, followed by complete model accounting and native validation. That seam is
staged only; it is not active runtime behavior. Long-prompt performance and context
qualification remain prerequisites for the OpenMythos editor rollout.
