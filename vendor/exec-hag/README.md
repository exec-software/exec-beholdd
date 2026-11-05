<!--
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT
-->

# HAG

HAG (`exec-hag`) is a Rust library for budgeted native hardware resources and
experimental GPU execution. It provides allocation ownership, checked memory
accounting and bounded command submission. Model loading, numerical operators
and inference policy belong to applications above HAG.

The current transport uses Linux DRM render nodes and AMDGPU directly. It requires
no Vulkan, SPIR-V, Mesa, ROCm or libdrm userspace runtime. The portable accounting
API has no dependencies when default features are used.

## Status

Native Linux discovery and memory management are implemented. An opt-in fixed
shader probe and a reusable synchronous compute queue have historical hardware
qualification on AMD device `0x1586`, physical GC `11.5.1`. Selected downstream
inference fixtures also passed, including bounded Qwen3-0.6B Q8_0 execution.
Those results cover the recorded programs, checkpoints and requests; they do
not establish general GPU, model, concurrency or performance support. See the
[qualification summary](docs/native-compute.md#historical-qualification).

`implementation_status()` reports compiled functionality, without probing the
machine. Its `native_compute` field remains false, and
`require_native_compute()` returns a typed error. Experimental queue access is
separate from general backend readiness. HAG never silently selects CPU or
another GPU backend after a refusal.

## Features

| Cargo feature | API |
| --- | --- |
| Default: none | Shared memory budgets and implementation status |
| `linux-drm` | Linux AMDGPU discovery, queries, budgeted GEM allocation and checked host access |
| `native-probe` | Explicit fixed-shader submission/completion proof; includes `linux-drm` |
| `native-compute` | Trusted-program library and reusable serial queue; includes `native-probe` |
| `capi` | C ABI over the experimental queue; includes `native-compute` |

The queue owns three guarded data slots, immutable shader storage and a reusable
command buffer. Its admission limits are 32 programs, 32 KiB of shader storage
and **4 GiB minus 4 KiB per logical data slot**. These are checked API bounds,
not guarantees of memory availability or hardware qualification at every size.

## Rust use

Use Rust 1.85 or newer. From a sibling application, a local dependency can be
declared as:

```toml
[dependencies]
exec-hag = { path = "../exec-hag" }
```

Portable accounting requires no GPU:

```rust
use exec_hag::budget::MemoryBudget;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let budget = MemoryBudget::new(1024);
    let reservation = budget.reserve(256)?;
    assert_eq!(budget.current_bytes(), 256);
    drop(reservation);
    assert_eq!(budget.current_bytes(), 0);
    Ok(())
}
```

A reservation accounts for bytes; it does not allocate physical memory.
`MemoryBudget::new(0)` has zero capacity. Use `MemoryBudget::unlimited()` only
when unlimited accounting is intended. Enable `linux-drm` for Linux device APIs;
`AmdGpuDevice::discover_report()` reports discovery and rejection diagnostics
without allocating GPU buffers. See [development setup](docs/development.md)
and the [compute contract](docs/native-compute.md) before enabling execution.

## C ABI

On Linux, build the shared and static libraries with:

```sh
cargo build --locked --release --features capi
```

Include [include/exec_hag.h](include/exec_hag.h) and link `libexec_hag.so` or
`libexec_hag.a` from Cargo's release output. The API exposes an opaque queue,
checked byte copies, synchronous dispatch and explicit close. The caller must
supply audited native shader code and serialize access to each handle. C
`budget_bytes = 0` means unlimited, unlike Rust's `MemoryBudget::new(0)`.
The [C ownership and error contract](docs/native-compute.md#c-interface) applies
in addition to the trusted shader requirements.

## Development and safety

Ordinary tests are hardware-free and model-free:

```sh
cargo test --locked --all-features
```

Read [development setup](docs/development.md) for the full check matrix and
[native probe instructions](docs/native-probe.md) for separately supervised
hardware work. GPU execution is never part of ordinary tests or discovery.

Allocations reserve their page-rounded charge before kernel allocation. Handles
and mappings remain owned; host accesses are bounds checked. An uncertain
submission or cleanup retains ownership and budget charges rather than allowing
unsafe reuse. A timeout or process exit does not guarantee cancellation of GPU
work. Experimental shader APIs accept trusted, audited code only.

The [backend roadmap](docs/native-backend.md) describes remaining qualification
gates. [Exact inference](docs/exact-inference.md) defines the downstream numerical
and measurement contract. The archived C implementation in `legacy/` is retained
for reference and is not built by Cargo.

## Contributing and license

See [CONTRIBUTING.md](CONTRIBUTING.md), [CONTRIBUTORS.md](CONTRIBUTORS.md) and
[SECURITY.md](SECURITY.md). Cargo's `0.1.0` tracks API compatibility; `VERSION`
tracks the repository's rolling revision independently.

Use the [publication checklist](docs/publication.md) before sharing a source
snapshot, research evidence, or binary package.

HAG is licensed under [MIT](LICENSE). Third-party notices and exceptions are
documented in [thirdparty.md](thirdparty.md).

## Support

Contributions, reproducible bug reports, and documentation improvements are
welcome. To support development financially, visit
[Buy Me a Coffee](https://www.buymeacoffee.com/execsoftware).
