<!--
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT
-->

# Closed native submission proof

The opt-in `native-probe` feature adds one fixed AMD/IGT GFX11 buffer-fill
shader. It does not accept arbitrary kernels or change the general readiness
reported by `implementation_status()` and `require_native_compute()`.

The recipe adapts only IGT's positive memset dispatch, pinned at commit
`691cc6fd50d5a3f6f2b5805c9698af1df28f6830`. The full upstream dispatch suite also
contains deliberate hangs, resets and scheduler-mask changes; it is not a
reproduction command for this proof. See [third-party notices](../thirdparty.md).

## What it tests

A dedicated render-node connection owns one normal-priority compute context,
three GTT buffers, GPU virtual-address mappings, a BO list and one indirect
command buffer. The fixed shader fills 16,384 bytes with `0x22`. Readback checks
every output byte and both 4,096-byte canary guards. Shader and command storage
add 8,192 bytes, for a total allocation charge of **32,768 bytes**. Storage is
initialized and mapped before submission.

Admission requires AMD device `0x1586`, family `150`, physical GC `11.5.1`
(`0x0b0501`), the compatible kernel command interface and ring zero. On the
qualified device the kernel interface reports `11.0`, with ring mask `0xf`.
Unknown identity, IP version, VA range, alignment or host page size is refused;
the current recipe requires 4-KiB pages. Another GFX11 device is not automatically
qualified by its family name.

The CS ioctl is issued once and never replayed on EINTR. A fence must be a real
sequence rather than zero or the latest-sequence sentinel. The two-second
completion deadline is an absolute `CLOCK_MONOTONIC` value below `i64::MAX`;
larger values are an infinite-wait sentinel in the driver. Readback requires
completed wait status and a healthy context-state query.

An uncertain submission, wait, mapping operation or cleanup quarantines its
ownership chain and budget until process exit. This prevents reuse of uncertain
resources; it does not cancel submitted GPU work. The fixed shader has no loops.

## Build and inspect

Complete the hardware-free checks in [development setup](development.md), then
build the example:

```sh
cargo build --locked --features native-probe --example native_probe
```

Use the render node selected for the intended development device. This invocation
only inspects it and does not allocate GPU buffers or submit work:

```sh
target/debug/examples/native_probe /dev/dri/renderD128
```

The explicit execution form is:

```text
target/debug/examples/native_probe --execute /dev/dri/renderD128
```

Run that form only through an established external supervisor with verified
memory, swap, task, runtime and thermal limits. Build before applying a small
execution-only memory cap. Check host headroom and capture driver faults over
the run's recorded interval. This crate does not bundle a portable thermal or
crash-capture service; refusing an unsupervised run is preferable to treating a
missing supervisor as permission to launch unrestricted work.

The fence deadline does not bound every driver ioctl. Process termination and
thermal supervision cannot guarantee GPU cancellation or prevent a reset from
affecting other clients sharing the GPU. Do not change clocks, reset the GPU or
alter scheduler policy merely to obtain a passing result.

A successful run must verify all 16,384 output bytes, 8,192 guard bytes and
`budget_after=0`. Wall time includes discovery, allocation and cleanup; it is
not a kernel benchmark or an inference throughput measurement.

## Historical qualification: 2026-09-26

The historical development snapshot passed 44 hardware-free unit tests, both
installed Linux-header ABI checks, strict all-target Clippy and formatting.
An offline `gfx1151` assembly comparison reproduced the pinned shader bytes.
Those software checks did not themselves execute a GPU workload.

One supervised live dispatch on the narrowly admitted device returned:

```text
PASS native shader: 16384 output bytes exact; 8192 guard bytes intact; fence=1; allocation=32768 bytes; budget_after=0; wall_time_us=4876
```

The child exited zero after completed fence status, healthy context, exact
readback and successful teardown. The observed kernel-journal interval contained
no matching reset, ring-timeout, VM-fault or OOM messages. That observation covers
the recorded interval, not the entire boot or every possible driver failure.

The run used Linux `6.17.0-1024-oem` and Rust
`1.100.0-nightly (809936eac 2026-09-12)`. It came from a dirty development tree,
not a published release. Execution limits were 256 MiB of process memory,
zero swap, 32 tasks, 50% of one CPU, a 10-second child timeout and a 15-second
service limit. The historical run did not enforce a software temperature cutoff;
that fact is not a recommended policy for future qualification.

This is one positive GTT-backed fill result, not a stress test, general device
support claim or model qualification. The public documentation cleanup did not
rerun it, and its original operational evidence is not distributed in this
repository. Later queue and inference results are scoped separately in
[the compute summary](native-compute.md#historical-qualification).

See [backend gates](native-backend.md) for remaining execution work and
[exact inference](exact-inference.md) for downstream numerical evaluation.
