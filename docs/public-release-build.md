# Public Linux build

The manual **Public Linux build** GitHub Actions workflow builds the exact selected
commit into separate CPU and experimental HAG executables. It runs on a fresh
public Ubuntu x86_64 runner and does not publish a release automatically. Repository
maintainers review and promote its artifact separately.

The builder uses digest-pinned official Rust 1.98.0, Python 3.12.10 and Node.js
22.20.0 images with a Debian bullseye base. The resulting compiler environment uses
glibc 2.31. Every binary is inspected for required glibc symbols; versions above
2.34 and unremapped `/home/` paths are rejected. The exact digests are in
[`tools/release/Dockerfile`](../tools/release/Dockerfile), and upstream image sources
are maintained by [Rust](https://github.com/rust-lang/docker-rust),
[Python](https://github.com/docker-library/python) and
[Node.js](https://github.com/nodejs/docker-node).

## Resource and execution boundary

Every Cargo/test/smoke stage requires at least 11 GiB of host `MemAvailable` before
launch. Its container has a 3 GiB memory limit, zero swap, a 1.5-CPU quota, 1,152
processes/threads, and a 900-second deadline. Limits are verified inside the cgroup.
A refused, killed or failed stage ends the build; it is never retried outside its
limits. The workflow itself has a 60-minute deadline.

Source is exported from a clean Git commit into a read-only mount. Containers run
without added capabilities, with no-new-privileges and a read-only root filesystem.
Only Cargo dependency fetching has network access; compilation and tests are
offline. No GPU device, model file, private checkout or credential directory is
mounted. This is a hosted-build supervisor; it does not replace or relax the local
[development supervisor](development.md#optional-linux-resource-supervisor).
Executable temporary compiler/test files use the bounded writable work mount;
the separate `/tmp` mount remains non-executable.

The build runs the default Rust workspace suite and an expanded suite with HAG,
OpenMythos and memory-failure test features. The separate Vulkan prototype needs
`glslc` and is not part of this CPU/HAG build or its test matrix. Ignored live tests stay ignored.
Python and JavaScript tool suites run with the pinned interpreter versions. Each
actual executable then serves an empty model registry on isolated loopback;
health, native identity, empty model listing and graceful SIGTERM are checked.
These checks do not qualify model output, GPU execution, performance or hardware.

## Artifact contract

The workflow artifact is `beholdd-linux-x86_64-<full-commit>`. It contains
`beholdd-v<N>-linux-x86_64-build.tar.gz` and `SHA256SUMS`. The tarball has no enclosing
directory and contains:

- `cpu/bin/beholdd` and `hag/bin/beholdd`, with separate receipts and feature lists.
- `manifest.json`, binding the source commit/tree, rolling version, Cargo lock,
  toolchain, immutable images, build limits, stage outcomes and binary hashes.
- `source-files.json` and `files.json`, recording source and packaged file hashes.
- Project documentation, example configurations, Cargo lock and license.
- Per-component `licenses/` text and inventories for the exact normal/build
  dependency graph, including the linked HAG MIT notice.
- Each component also retains the pinned Rust distribution's standard-library
  copyright notice and license files with hashes. If that image provides only a
  broader distribution copyright notice, its receipt records that broader scope.
  CPU/HAG server artifacts include chat-template provenance and Apache-2.0 text.
- `hag-library/`, containing the standalone HAG shared/static libraries, C header,
  dependency notices and its own unchanged vendored source revision and lockfile.
  Both libraries pass a C link/execution smoke using constant-query functions;
  HAG's hardware-free feature tests also run. No device is opened by these checks.

Archive entries are regular files/directories with normalized ownership and modes.
`files.json` inventories every packaged file except itself. SHA-256 checks detect
corruption; a distributor must also authenticate the workflow run, repository and
source commit before promotion. Never select the most recent artifact by name
alone or relabel a binary with another source revision.

Antimatter can import the reviewed CPU and HAG binaries into separate vendor
directories and regenerate its own sealed license manifests at the same commit.
Keep the build receipt with that import. The artifact contains no bundled model,
Vulkan compatibility server or GPU driver; those are separate release inputs.

On an explicitly provisioned build host with Docker and adequate resources, the
same entry point is available without GitHub:

```sh
python3 tools/build_public_release.py --work-dir /path/to/new/external/build-directory
```

The directory must be new, outside the checkout, and on a filesystem with enough
space for pinned container layers, Cargo dependencies, test objects and outputs.
