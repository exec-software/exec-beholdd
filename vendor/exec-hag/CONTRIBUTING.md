<!-- Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT -->

# Contributing to HAG

The active library is the Rust crate at the repository root. Keep hardware
resources and command lifetimes here; model formats, tokenizers, inference
policy, and application behavior belong in callers. The C prototype under
`legacy/` is an unsupported archive, not an alternative production backend.

## Local workflow

1. Use Rust 1.85 or newer. Follow [Development setup](docs/development.md) for
   feature boundaries and portable build commands.
2. Keep changes focused and add a reproducer for behavioral or security fixes.
   Test allocation ownership, bounds, cancellation, failure, and cleanup through
   private syscall injection. Explain every unsafe block's preconditions.
3. Keep generated artifacts under ignored build directories. Do not commit
   credentials, local machine configuration, model files, crash dumps, private
   integration notes, or workstation-specific paths.
4. Update public documentation when behavior changes. Distinguish API
   implementation, hardware-free testing, and actual device qualification.
5. Run the affected checks and `git diff --check` before submitting a change.

The ordinary checks require no GPU or model:

```sh
cargo fmt --check
cargo test --locked --no-default-features
cargo test --locked --all-features
cargo clippy --locked --all-targets --all-features -- -D warnings
python3 scripts/check_public_tree.py
git diff --check
```

With a C compiler and Linux DRM development headers installed, run the explicit
ABI contract checks. They compile host helpers and do not access a GPU:

```sh
cargo test --locked --all-features linux::uapi::tests::matches_installed_linux_headers -- --ignored --exact
cargo test --locked --all-features linux::submission_uapi::tests::submission_matches_installed_linux_headers -- --ignored --exact
```

## Safety and review

Linux CI also compiles the standalone archived allocator and thread regressions
with AddressSanitizer and UndefinedBehaviorSanitizer. These exercise only host
helpers; they do not build or qualify the incomplete legacy runtime. Their source
comments describe direct compilation outside the archived CMake project.

Normal tests must not open a GPU, submit commands, or load a model. A live
qualification run is a separate, explicitly selected operation with finite
memory, CPU, thermal, and time limits and verified cleanup. Follow
[the native probe guide](docs/native-probe.md) and [security policy](SECURITY.md).

Keep portable builds independent of Linux dependencies. Linux UAPI belongs in
`src/linux/`; do not add Vulkan, a shader compiler, a model loader, implicit CPU
fallback, or a public arbitrary-command-packet interface to the current
transport. Discovery and allocation are not proof of compute support.

Keep third-party notices intact. New dependencies or copied code must have
provenance and compatible licensing documented in [thirdparty.md](thirdparty.md).
Before release, inspect package contents and scan both the current tree and Git
history for secrets. The public-tree check is a regression check for known
publication hazards, not a comprehensive security or secret scanner.

## Issues and pull requests

Use the [issue tracker](https://github.com/exec-software/exec-hag/issues) for
non-sensitive bug reports and proposals. Include the commit/version, enabled
features, platform, a minimal reproducer, expected behavior, and actual behavior.
Report vulnerabilities through [SECURITY.md](SECURITY.md).

Describe the problem, resulting behavior, relevant validation, and remaining
limitations in each pull request. Preserve unrelated changes. Source
contributions do not change `VERSION`, publish a release, or qualify new hardware.
Release maintainers manage repository versioning and distribution separately.

## Licensing and attribution

First-party contributions are made under the repository's [MIT License](LICENSE).
Project-owned files carry:

```text
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
```

Retain existing third-party and contributor notices; do not replace their
attribution with the project's notice. Generated `Cargo.lock` and the
machine-readable `VERSION` file are covered by the repository license without
inserting comments that their generators or consumers do not preserve.

Original and community contributors are recognized in
[CONTRIBUTORS.md](CONTRIBUTORS.md). You may add your preferred name to the
community section when submitting a contribution.
