<!--
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT
-->

# Native backend gates

HAG supplies hardware mechanisms: discovery, allocation ownership, accounting,
command submission and completion. Applications supply numerical kernels, model
formats, scheduling, request state and user-facing policy. A downstream inference
engine can use HAG without turning HAG into another inference runtime.

## Current foundation

The Linux transport uses DRM render nodes and AMDGPU UAPI. Discovery reads
kernel-reported capabilities and rejection diagnostics without allocating GPU
buffers. Memory allocation and host access are explicit, budgeted operations.
There is no implicit CPU, Vulkan or alternate-heap fallback.

The [fixed native probe](native-probe.md) covers one bounded positive dispatch on
AMD device `0x1586`, physical GC `11.5.1`. The experimental
[serial queue](native-compute.md) adds immutable program selection, resident
subregions and reuse between synchronous dispatches. Historical downstream
fixtures progressed from packed arithmetic to selected complete-model and
client lifecycle checks; each retains its own numerical and resource scope.

General readiness remains deliberately unavailable:
`implementation_status().native_compute` is false and `require_native_compute()`
returns a typed error. Experimental feature availability and selected successful
fixtures do not imply a supported general execution backend.

## Responsibility boundaries

| Layer | Responsibilities |
| --- | --- |
| HAG portable API | Checked budgets, ownership and explicit capability/refusal reporting |
| Linux transport | Render-node connection, UAPI validation, mappings, submission and completion |
| Trusted kernel provider | ISA audit, ABI compatibility, indexing, termination and numerical reference |
| Inference engine | Checkpoint encoding, operators, state, sampling, cancellation and fit forecasts |
| Application client | Capability discovery, request limits, transport lifecycle and user-visible results |

HAG currently uses Rust `std`. It is not a kernel or `no_std` library and defines
no alternate operating-system kernel ABI. A future transport must expose an
explicit memory, queue, isolation and completion contract without making Linux
file descriptors and ioctls the portable interface.

## Promotion criteria

New execution support should establish these properties in order:

1. **Device and ABI identity.** Validate discovery, command-interface versions,
   page size, alignment and address ranges. Refuse unknown combinations.
2. **Memory and ownership.** Bound every allocation and host access; retain all
   resources through in-flight work and uncertain cleanup. Demonstrate failures
   without double release, stale reuse or false budget credit.
3. **Command lifecycle.** Require valid submission identity, finite completion,
   healthy context and guard integrity. Define poisoning and retirement before
   adding asynchronous work, sharing or recovery.
4. **Numerical operators.** Verify packed representation, complete output and
   preregistered error criteria for identified programs and shapes. Include reuse
   with changed inputs and independent reference implementations.
5. **Model and state composition.** Check selected real checkpoints, complete
   logits, committed tokens, capacity refusal, reset and cached state. Account
   for staging, workspace, retained work and host memory as well as GPU buffers.
6. **Serving and client lifecycle.** Exercise request admission, cancellation,
   reuse, unload, transport failures and shutdown. Then evaluate broader quality,
   concurrency, shared-device behavior and measured performance separately.

Historical progress through later criteria does not qualify every device or
program admitted by the API. In particular, the expanded 32-program/32-KiB
library, larger slot limit and derived register/workgroup limits need scoped
hardware evidence for the workloads that use them.

## Remaining work

The public queue remains serial and synchronous, with one physical submission
and fence per dispatch. Asynchronous retirement, cross-queue sharing, explicit
VRAM transfers, device-loss recovery and general isolation are separate design
and qualification work. Larger models, arbitrary numerical inputs, broader
contexts and production throughput cannot be inferred from the finite fixtures.

Follow [development setup](development.md) for ordinary checks and
[exact inference](exact-inference.md) for preservation and measurement rules.
A change to a product's default backend requires its complete workload and
lifecycle gates, beyond HAG's hardware-free checks.
