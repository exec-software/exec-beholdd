<!-- Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT -->

# Third-party dependencies

The standalone Rust library has one optional dependency. Exact versions and
registry checksums are recorded in `Cargo.lock`.

| Dependency | Source | License | Purpose |
|---|---|---|---|
| libc 0.2 | https://github.com/rust-lang/libc | MIT OR Apache-2.0 | Linux system calls and architecture-aware ioctl encoding |
| IGT GPU Tools positive GFX11 memset shader/packets | https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/tree/691cc6fd50d5a3f6f2b5805c9698af1df28f6830 | MIT | Closed native proof and reusable compute-register initialization, adapted in src/linux/probe_packets.rs and src/linux/compute/packets.rs; upstream notices retained there |

The IGT input is pinned source, not a linked runtime dependency. Only its positive
buffer-fill recipe and compute-register initialization are used: no deliberate
hangs, scheduler changes, resets or user-mode queues. The 316-byte shader SHA-256 is
`735ea6ea8faa4e398ee231d55e9f80ec5c577346c333533ab2417f9beb2b795c`;
offline LLVM18 assembly for gfx1151 reproduced it exactly. LLVM is a verification
tool, not a build or runtime dependency of this crate.

Cargo distributes the upstream license files with its dependency source. The
crate is unmodified. No libdrm, Mesa, Vulkan, SPIR-V compiler, model-serving
library or tokenizer is linked by HAG.

The Linux AMDGPU UAPI definitions are transcribed from the kernel's userspace
interface headers (`include/uapi/drm/amdgpu_drm.h` and `drm.h`), under their
permissive MIT-style notices retained in `src/linux/uapi.rs`. Keep field layouts and constants
verified against the headers rather than inventing structures or command
numbers. See https://docs.kernel.org/userspace-api/ and
https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/include/uapi/drm/amdgpu_drm.h.

Historical dependencies of the archived C implementation remain under
`legacy/`; they are not part of the root Cargo build.

The files named `legacy/libs/unity/unity.c` and `unity.h` are small
Execute Software-authored compatibility placeholders, not a vendored release
of the upstream Unity test framework. The archived CMake files do not download
Unity. If those placeholders are ever replaced with upstream source, preserve
that release's copyright and license notices and record its exact version.

Execute Software's header in an adapted file covers the project's contributions;
it does not replace the upstream authors' notices. The complete upstream
permission and warranty notices in the four Linux packet/UAPI source files must
remain with redistributed copies.
