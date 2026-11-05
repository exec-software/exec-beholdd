# Research and qualification records

This directory contains Beholdd's implementation notes, experiment designs, and
bounded validation records. It is the standalone public research index; no
external notebook or unrelated project checkout is needed to follow it.

## Start here

- [Engineering notebook](../../notebook/Index.md): storage, scheduling and
  numerical evidence required for useful comparisons.
- [Model streaming](model-streaming.md): exact-byte residency and I/O experiments.
- [Native acceleration](native-acceleration.md): the experimental GPU execution
  program and its current limits.
- [Dense Qwen3](native-qwen3-dense.md): native binding and reference checks.
- [MiMo binding](native-mimo2-bind.md), [block execution](native-mimo2-block.md),
  and [decode](native-mimo2-decode.md): correctness harnesses, not serving support.
- [Nemotron architecture plan](native-nemotron-h-moe.md): pinned published
  artifacts and uncompleted implementation gates.
- [HAG integration](../hag-backend.md): native submission and bounded resident
  execution proofs.
- [HIP/rocWMMA experiments](../../tools/hip/README.md): isolated benchmarks with
  pinned source/toolchain checks.

## Reading historical results

Historical measurements retain their dates, fixture revisions, numerical bounds,
successes and failures. Host-specific directory names and unrelated deployment
labels have been normalized for publication. Paths in old command transcripts are
illustrative, not downloadable artifacts or claims that a file exists locally.
Raw host logs and crash dumps are not distributed in this repository.

Where a record uses a symbolic path, set it explicitly for your own environment:

| Variable | Meaning |
| --- | --- |
| `BEHOLDD_SOURCE_DIR` | Root of this checkout |
| `BEHOLDD_MODEL_DIR` | Local model directory, containing separately acquired weights |
| `HF_HUB_CACHE` | Local Hugging Face snapshot cache |
| `TRANSFORMERS_SOURCE` | Installed or audited `transformers` package directory |
| `ROCM_CORE` | Exact ROCm SDK core used by a HIP qualification wrapper |
| `BEHOLDD_BUILD_DIR`, `BUILD_ROOT` | External build directories chosen by the developer |
| `BEHOLDD_LOG_DIR`, `HAG_LOG_DIR` | External log directories chosen by the developer |
| `EVIDENCE_DIR`, `DEVELOPER_DIR` | Symbolic roots replacing private historical evidence paths |
| `WORKSPACE_ROOT`, `ANTIMATTER_SOURCE_DIR` | Optional local workspace and independent client checkout |

Names such as `validation-host` and `alternate-host` are generic labels, not DNS
targets. `REDACTED-HOST-ID` marks an intentionally omitted host identifier; it is
not an input to a qualification tool. Original host records must be reviewed
separately before any evidence archive is distributed.

Public model IDs and immutable digests identify the fixture; a local cache path
does not. Verify model licenses and bytes before use. Resource limits and
hardware allowlists in a record are part of its scope, not permission to run an
experiment on an arbitrary machine.

Operator parity is not full-model parity. A fixed-layer benchmark is not token
throughput. Compilation and mocked transport tests are not live-GPU evidence.
The [exact-inference contract](../../vendor/exec-hag/docs/exact-inference.md)
defines these distinctions.
