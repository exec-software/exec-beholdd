# Beholdd

Beholdd is a modular inference engine and HTTP server written in Rust. It provides
native model execution, explicit memory budgeting, exact-byte weight streaming,
and an in-process embedding API. Unsupported models and execution modes are
reported as errors; requests do not fall back to an external inference server.

The project is under active development. Linux is the implemented platform.
CPU serving is available for GGUF checkpoints with the exact architecture tags
`qwen3` and `qwen3moe`. HAG is an explicitly selected experimental backend, including scoped dense-Qwen3
qualification and ongoing OpenMythos development. CPU remains the default.
Vulkan and HIP operator benchmarks are separate from serving qualification.

The public source history starts at rolling version **v1**. Cargo's `0.0.1`
component version is independent. Earlier measurements remain historical records;
this source import does not requalify devices or publish a new binary.

## Build and run

A Rust/Cargo toolchain compatible with the locked dependencies and a native C
build toolchain are required. Python 3.11 or newer is needed for the repository's
development scripts. The default CPU build does not require a GPU SDK, HAG
checkout, private package registry, or another application.

From the repository root:

```sh
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$HOME/.cache/beholdd/target}"
cargo build -p beholdd-cli --release --locked
"$CARGO_TARGET_DIR/release/beholdd" serve --config ./config.example.yaml
```

Before requesting inference, obtain a supported checkpoint under its own license
and edit [config.example.yaml](config.example.yaml) to name its location and an
appropriate memory budget. For a smaller bring-up fixture, use
[config.dev.example.yaml](config.dev.example.yaml) with Qwen3-0.6B Q8_0.
Relative checkpoint paths resolve beside the
configuration file. Weights are not included or downloaded automatically.

The example describes a Qwen3-Coder-30B Q8_0 checkpoint of approximately 32.48 GB;
its 40 GB execution budget is not suitable for every machine. Review
[serving configuration](docs/serving.md#configuration) before using it. Model IDs
are registered at startup, but checkpoint loading is lazy and begins with the
first completion request.

The server listens on `127.0.0.1:6768` by default. Keep it on loopback unless you
provide appropriate network access controls; the native serving interface is not
an authenticated public gateway.

```sh
curl http://127.0.0.1:6768/health
curl http://127.0.0.1:6768/v1/beholdd/info
curl http://127.0.0.1:6768/v1/models
```

See [Native serving](docs/serving.md) for chat completions, SSE, tool-call
translation, model unloading, and the `beholdd-embed` API.

## Current capabilities and limits

| Area | Status |
| --- | --- |
| Native CPU serving | Dense `qwen3` and routed `qwen3moe` GGUF checkpoints |
| Storage formats | Checked GGUF and Safetensors readers; serving currently requires GGUF |
| Memory policy | Explicit budgets, resident or streamed weights, cgroup-aware admission, storage checks |
| HTTP API | Health, capability discovery, model list/unload, chat completions and SSE |
| Embedded API | The same native runtime without an HTTP listener |
| Vulkan | Experimental packed operators and bounded routed-MoE qualification |
| HAG | Explicit experimental Linux AMDGPU serving; model/device/context-specific qualification |
| HIP/rocWMMA | Isolated research benchmarks, not a serving backend |
| Other architectures | OpenMythos/Qwen3.5 experimental HAG profile; MiMo correctness work and Nemotron planning |
| Other operating systems | Not currently implemented and validated |

Backend compilation does not establish model support. HAG qualification is tied
to named model fixtures, devices, context positions and request profiles. The
16k admission/attention checks do not qualify every full-context model request,
and operator timing is not generated-token throughput. Current qualification limits are documented in
[HAG backend status](docs/hag-backend.md),
[native acceleration](docs/research/native-acceleration.md), and
[the roadmap](ROADMAP.md).

Historical Antimatter client profiles require commits absent from the clean public
history and are not runnable public workflows. See the
[qualification scope](docs/hag-qualification.md#public-source-scope) before using
those records. The ordinary CPU build and standalone backend profiles do not
require those client commits.

The recorded CPU client integration test completed its two-completion tool
workflow in about 27 minutes. Its guarded service later reached the configured
runtime timeout, so the service lifecycle was not a clean pass. This is evidence
of a narrow functional integration, not production latency or reliability.

## Design principles

- **Explicit fit policy.** `Exact` preserves checkpoint precision and streams when
  supported; `Refuse` reports a shortfall. `Degrade` is an explicit policy value,
  but serving refuses if semantic degradation would require an unimplemented
  executor. It never silently requantizes a model.
- **Accounted memory.** Weight storage, caches, generation buffers, backend
  scratch, and headroom contribute to admission. Observed memory and I/O accompany
  forecasts.
- **Replaceable layers.** Storage, arithmetic, execution, model definitions, and
  serving have separate interfaces and tests.
- **Measured claims.** Numerical equivalence, bitwise equivalence, model quality,
  and throughput are different claims. Research records state their fixtures,
  devices, bounds, failures, and remaining gates.
- **No implicit fallback.** A request either uses an implemented native execution
  path or returns a diagnostic error.

The accepted [fit contract](docs/adr/0001-fit-contract.md) and
[project notebook](notebook/Index.md) describe these decisions in detail.

## Workspace structure

| Crate | Responsibility |
| --- | --- |
| `beholdd-core` | Tensor and quantization types |
| `beholdd-format` | Checkpoint parsing and validated weight ranges |
| `beholdd-kernel` | Numerical operators |
| `beholdd-backend` | CPU and experimental device execution |
| `beholdd-model` | Model binding, tokenization and forward execution |
| `beholdd-memory` | Fit planning, residency, caches and storage inspection |
| `beholdd-runtime` | Configuration, model registry and generation scheduling |
| `beholdd-server` | HTTP protocol and native tool-call translation |
| `beholdd-embed` | In-process inference |
| `beholdd-cli` | Server executable |

## Development and verification

The manual [public Linux build](docs/public-release-build.md) produces separate
CPU/HAG artifacts from an exact clean commit, with bounded hardware-free tests,
an older glibc toolchain and complete dependency notices. It does not publish or
qualify live inference automatically.

[Development setup](docs/development.md) describes build prerequisites and the
optional Linux resource supervisor. The ordinary suite uses synthetic fixtures; it does not require downloading a
checkpoint or submitting work to a GPU:

```sh
cargo test --workspace --locked
python3 -m unittest discover -s tools -p 'test_*.py'
```

Use an external Cargo target directory as shown above. Tests marked ignored may
require large checkpoints, a supported GPU, or a separate toolchain. Do not run
all ignored tests indiscriminately. The opt-in model tests use
`BEHOLDD_QWEN3_DENSE_MODEL` and `BEHOLDD_QWEN3_CODER_MODEL`; supply the exact
documented fixture and apply resource limits before running them.

HAG is a checked-in source snapshot under `vendor/exec-hag`. A normal checkout
builds without a sibling repository. Maintainers can refresh it from an explicit
standalone HAG checkout using [the documented sync procedure](docs/hag-backend.md#source-ownership-and-reproducibility).

The HIP research wrappers require explicit `ROCM_CORE` and `ROCWMMA_INCLUDE`
paths. See [HIP benchmark documentation](tools/hip/README.md); their frozen
toolchain and source checks are intentional.

## Documentation

- [Native serving and configuration](docs/serving.md)
- [Architecture decisions](docs/adr/0001-fit-contract.md)
- [Research and qualification records](docs/research/README.md)
- [HAG integration](docs/hag-backend.md)
- [Roadmap and acceptance gates](ROADMAP.md)

All essential setup and design documentation is maintained in this repository.
A private notebook, developer-specific filesystem layout, or unrelated project
checkout is not required.

## Support

Beholdd is built by Execute Software and its community. See
[CONTRIBUTORS.md](CONTRIBUTORS.md) for the core team and contributor credits.

Contributions and reproducible bug reports are welcome. You can support ongoing
development through [Execute Software donations](https://www.buymeacoffee.com/execsoftware).

For enterprise licensing and support, contact
[admin@executesoftware.com](mailto:admin@executesoftware.com).
