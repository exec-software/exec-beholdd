<!--
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT
-->

# Development setup

The active library is the Rust crate at the repository root. Cargo does not
build the archived C implementation in `legacy/`. Development commands below
run from the repository root and use Cargo's normal target directory.

## Prerequisites

Use Rust 1.85 or newer with Cargo, rustfmt and Clippy. Linux APIs require a Linux
target and a normal C/link toolchain. The optional installed-header checks also
need a C compiler and Linux DRM development headers, commonly packaged as
`libdrm-dev`. HAG uses the kernel UAPI directly and does not link against libdrm.

Default features are empty. The portable budget API works without Linux or GPU
dependencies. Linux-only features are cumulative:

```text
linux-drm <- native-probe <- native-compute <- capi
```

No model download, GPU, privileged service or authentication prompt is required
for ordinary builds and tests. Live execution has a separate supervision and
qualification process described in [native-probe.md](native-probe.md).

## Routine checks

```sh
cargo fmt --check
cargo test --locked --no-default-features
cargo test --locked --features linux-drm
cargo test --locked --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
```

CI also checks the declared minimum Rust version. With that toolchain installed,
run `cargo +1.85.0 test --locked --all-features` before adding APIs that may
require a newer compiler. Set `CARGO_TARGET_DIR` to place build artifacts outside
the checkout when needed.

Ordinary tests inject driver behavior and never open live GPU nodes, allocate
GPU memory or dispatch work. They cover budget races and overflow, UAPI layout,
resource lifetimes, bounds, submission failures, poisoning and cleanup. A passing
test suite establishes those software contracts; it does not qualify a device,
program or model.

With the optional Linux development headers installed, check the UAPI against
the installed headers explicitly:

```sh
cargo test --locked --all-features linux::uapi::tests::matches_installed_linux_headers -- --ignored --exact
cargo test --locked --all-features linux::submission_uapi::tests::submission_matches_installed_linux_headers -- --ignored --exact
```

These checks compile layout helpers and do not access a GPU. Shader assembly
comparison, when used by a kernel provider, is likewise an encoding check rather
than an execution test. Normal HAG builds do not require a shader compiler.

Use a constrained build job count or an external process-memory limit when
developing on a shared or memory-limited machine. Build limits and GPU allocation
budgets cover different resources; neither replaces the other.

## Working on the implementation

Keep portable accounting separate from Linux UAPI and ioctls. Device discovery
must remain distinct from allocation and execution. Kernel-reported properties
and explicit supported combinations determine admission; unknown combinations
must refuse rather than guess an ISA or substitute a backend.

An allocation owner retains its device connection and budget reservation until
cleanup is definite. Exercise partial construction, interrupted syscalls,
timeouts, driver failures and unwinding through injected behavior. Do not credit
an uncertain release or make a poisoned queue reusable without a separately
designed recovery contract.

The trusted-program API is intentionally unsafe. Changes to register layout,
instruction payloads, descriptor geometry or dispatch limits require their own
review and qualification scope. The existing hardware result does not transfer
automatically to a larger admitted shape or a different program.

Applications own model formats, numerical operators, tokenization, request
admission and inference scheduling. Keep those policies out of HAG. For C callers,
maintain the ownership, pointer validity, status and panic boundary in
[the public header](../include/exec_hag.h).

## Hardware qualification workflow

1. Freeze source, binary, dependency and shader identities plus the intended
   device, driver and workload. Record any dirty source state.
2. Complete the relevant hardware-free checks and independent numerical
   reference before opening a GPU queue.
3. Establish and verify process memory, swap, task, runtime and thermal limits.
   Account separately for HAG allocations, host staging and driver overhead.
4. Run only the explicitly selected bounded fixture. Preserve failed and aborted
   attempts separately from later successful attempts.
5. Check complete outputs, guards, completion status, budget retirement and
   driver/OOM observations over a recorded interval. Confirm process cleanup;
   process disappearance alone does not prove GPU cancellation.
6. Report the exact tested scope and remaining limitations. Do not turn a finite
   fixture or an admission-limit change into general backend readiness.

The [fixed probe](native-probe.md) is the smallest execution check. The
[reusable queue contract and historical summary](native-compute.md) describe
later results. Those historical records were condensed for public documentation;
the documentation update did not rerun GPU or model workloads. Detailed local
operational logs and downstream application harnesses are not shipped by this
crate, so the summaries are not a complete reproduction package.

## Further reading

- [Native backend gates](native-backend.md): responsibilities and promotion criteria.
- [Exact inference](exact-inference.md): parity categories and end-to-end measurement.
- [Contributing](../CONTRIBUTING.md): review and contribution conventions.
- [Security](../SECURITY.md): reporting and trust boundaries.
- [Third-party notices](../thirdparty.md): upstream code and attribution.
