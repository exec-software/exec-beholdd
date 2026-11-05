<!-- Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT -->

# Security

## Reporting a vulnerability

Use the repository's [private vulnerability reporting form](https://github.com/exec-software/exec-hag/security/advisories/new)
when it is available. If private reporting is unavailable, open an issue asking
for a private reporting channel without including exploit details, credentials,
private logs, or affected users' information. A maintainer can then arrange a
private report. Do not publish a working GPU fault or denial-of-service program
in a public issue.

Include the affected commit or version, enabled Cargo features, operating
system and driver versions, affected API, expected behavior, and the smallest
safe reproducer. Remove usernames, device identifiers unrelated to the bug,
machine paths, tokens, and model data from logs before sharing them.

The active Rust crate on `main` is the maintained implementation. The archived
C implementation under `legacy/` is unsupported and must not be deployed as a
security boundary. There is no production-support or response-time guarantee
for the experimental compute interfaces.

## Trust boundaries

- HAG is an in-process native library, not a sandbox or a network service.
  A caller with arbitrary native-code execution already controls its process.
- Linux operations use DRM render nodes and the caller's existing device
  permissions. HAG does not install a service, require root, or elevate access.
  Driver and firmware vulnerabilities are outside the library's control.
- Compute programs must be trusted, audited machine code for the explicitly
  supported device and ABI. Size, alignment, register, descriptor, and guard
  checks do **not** validate shader semantics or make hostile shaders safe.
  A faulty program can reset the GPU and affect other applications.
- Model contents, downloaded artifacts, user input, and network payloads must
  never be treated as shader code. HAG does not parse models or compile shaders.
- C callers must satisfy the documented pointer, alignment, lifetime, and
  exclusive-access requirements. Returning an error cannot make invalid C
  pointers or concurrent use of the same handle safe.

## Resource containment

Use an explicit finite memory budget, finite dispatch deadlines, and a separate
process with host-enforced memory and CPU limits for live GPU qualification.
Library accounting is not a substitute for operating-system limits: a fatal
allocator failure or a driver fault need not be recoverable with Rust panic
handling. Account for host staging, page cache, and shared system/GPU memory.

On uncertain completion or cleanup failure, HAG conservatively retains affected
resources and their accounting instead of making memory available for reuse.
Terminate the containing process after such a failure; do not attempt to recover
by bypassing a poisoned handle or silently switching backends.

Payload tracing can expose buffer contents. Leave `HAG_TRACE_SUBMISSION` disabled
for ordinary use; explicitly enable it only with non-sensitive test inputs and
private logs. Normal automated tests must not open a GPU or load a model.

## Release hygiene

Keep secrets and workstation configuration outside the repository. Review
tracked files, Git history, release archives, and package contents separately;
removing a value from the current tree does not remove earlier commits. Rotate
any credential that has been committed, even if the repository was private.
Retain third-party notices and review dependency advisories when publishing.
