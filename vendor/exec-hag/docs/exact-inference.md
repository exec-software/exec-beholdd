<!--
Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
SPDX-License-Identifier: MIT
-->

# Exact inference: design and evaluation contract

This is a research contract, not a throughput forecast or a claim of model support.
The objective is lower end-to-end latency and memory use without silently changing
the selected model. Large checkpoints and real-time interaction are targets to
measure, not capabilities implied by a backend name or parameter count.

## Freeze the reference before optimizing

Every experiment records checkpoint and shard hashes, original storage encoding,
scale tensors, tokenizer, prompt/template bytes, architecture configuration,
context length, and request-state representation. Record arithmetic/accumulation
precision, backend/compiler versions, sampling transforms and their order,
temperature, truncation settings, seed/RNG algorithm, stop rules, and tie-breaking.
An already-quantized checkpoint can be the reference; preserving that artifact
does not prove parity with its unquantized ancestor.

Report these separately:

- **Bitwise parity:** identical complete logits and request state at declared
  boundaries under a reproducible execution policy, plus identical committed
  token IDs for deterministic/seed-coupled tests. Matching one final answer is
  insufficient. Storage-only changes should clear this stronger gate.
- **Numerical parity:** preregistered absolute/relative error bounds, finite-value
  checks and outlier reporting against the reference. Changed reduction order,
  fusion or accumulation precision may require this category; a tolerance does
  not imply identical sampling decisions or generated tokens.
- **Distributional parity:** the sampling procedure preserves the target
  conditional distribution. This does not mean the same seed consumes the same
  random stream or produces the same individual sequence.
- **Quality parity:** paired task evaluations, fixed datasets, uncertainty and
  regression thresholds. Passing a finite evaluation is evidence, not a proof
  of universally unchanged quality. Do not label it bitwise or distributional
  equivalence.

## Keep specialized engines above HAG

An architecture-specialized engine owns model operators and token-generation
semantics above the hardware layer. Its implementation language alone supplies
neither acceleration nor model support. Reuse checked checkpoint ranges, fit
budgets, streaming policy and telemetry; avoid duplicate model caches and
implicit backend fallback.

An execution adapter should expose owned packed weights, owner-checked tensors,
reusable commands and explicit submission/completion. In-flight work must retain
resources even after an unwaited submission is dropped; conflicting host access
must wait or refuse. HAG supplies hardware mechanisms, not tokenization, routing
policy or architecture semantics. A future operating-system transport does not
move inference semantics into the kernel.

The existing [C ABI](../include/exec_hag.h) has explicit ownership, buffer bounds,
status and synchronous completion rules. Rust object layouts are not a C ABI.
Any future asynchronous interface also needs explicit cancellation and retirement
rules; the current queue does not provide asynchronous resource sharing.

## Prioritize preservation of the reference

1. Remove redundant allocation, copying, host readbacks and submissions. Retain
   packed checkpoint bytes and reuse prepared resources. Account for alignment,
   staging, scratch, KV/recurrent state and retired in-flight allocations.
2. Reuse exact weights and prefix/request state only under complete identity keys:
   checkpoint, configuration, token history, positional state and relevant
   precision/backend policy. Snapshot/rollback must cover every state component,
   not only an append-only KV buffer.
3. Explore reversible layouts and lossless compression with byte-for-byte decode
   tests. Measure decompression, indexes and temporary buffers; compressibility
   and net speedup are observations, not assumptions.
4. Overlap bounded I/O and compute; single-flight duplicate misses and coalesce
   only validated ranges. Prefetch must not change routing or omit demanded
   weights. Predictions may waste bandwidth but must never substitute guessed
   experts. Preserve data-dependent barriers and cancellation accounting.
5. Explore verified speculative decoding separately. Greedy verification must
   match the target's decisions; stochastic verification needs the correct
   acceptance/residual sampling algorithm using the actual proposal and target
   distributions after all sampling transforms. Reject/rollback tentative state
   correctly, handle stop tokens, and publish only committed tokens. Target
   verification is mandatory; accepting a plausible draft is not exact decoding.
   Distribution preservation is an algorithmic property to validate, not a
   same-seed token-identity claim. See the primary
   [speculative decoding paper](https://proceedings.mlr.press/v202/leviathan23a.html).

New quantization, lossy cache compression, pruning, skipped experts, approximate
attention and reduced context are separate, explicit opt-in modes. Give them
distinct configurations and quality gates; never use them to satisfy an exact
mode's fit or speed claim silently.

## Measure the complete path and its limits

Define TTFT from request arrival to the first committed output token, including
queueing, lazy load, prompt processing and delivery. Report server timing and
client-observed timing separately. Record steady decode TPOT and inter-token
latency distributions, not only average throughput; specify handling of first
tokens, short requests and speculative token bursts. Record full request latency,
committed output count, concurrency, prompt/context lengths and sampling policy.

Run paired alternating baseline/candidate trials with both declared cold and hot
cache states. Distinguish application cache, OS page cache, storage cache and
prefix reuse; process restart does not prove cold storage. Cache-control methods
must be isolated and authorized, never host-wide eviction during unrelated work.
Hold memory/cgroup, CPU/GPU policy and thermal limits constant; report deviations.

Measure demanded packed bytes, unique active ranges, cache-miss bytes, physical
read amplification, link traffic, wasted/late prefetch, exposed stalls and actual
overlap. Include RSS/cgroup peaks, device allocations and staging/retirement peaks.
Total checkpoint size and activated parameter counts do not replace these bytes.

For a measured interval with `N` committed output tokens, use conditional floors
such as `T_io >= physical_bytes / bandwidth_ceiling` and
`T_arithmetic >= required_operations / operation_rate_ceiling`. Qualify those
ceilings for the named access pattern, precision and hardware; a measured rate
is an empirical comparison, not proof of an absolute limit. Include memory-bus
traffic as well as arithmetic in the compute bound. With independent resources
and perfect overlap:

```text
T_interval >= max(T_storage, T_host_device_link, T_compute)
TPOT_floor = T_interval_floor / N
```

Shared channels require combined traffic accounting; dependency barriers and
non-overlappable stages make the real critical path longer. For illustration
only, 12 GiB of uncached reads at 4 GiB/s already costs at least 3 seconds before
other work. Prefetch cannot remove required bytes or exceed the supplying
bandwidth; residency/reuse can change those required reads.

## Promotion gates

Start with checked byte/state fixtures and numerical operator references; then
test recorded execution, concurrency, failure/cancellation and memory bounds.
Only then run identified real-checkpoint parity and end-to-end latency trials.
Publish commands, hashes, hardware, cache state, raw measurements, uncertainties
and failures alongside improvements. Speculation includes draft/verification
cost and acceptance rate per committed token.

A resident kernel benchmark is not model tokens/second. A native HAG dispatch
proof is not a complete execution adapter, supported model architecture or serving
release. Keep these gates separate from the
[native backend qualification gates](native-backend.md) and do not change shipped
defaults until the complete workload clears them.
