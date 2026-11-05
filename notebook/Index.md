# Beholdd engineering notebook

Beholdd separates checkpoint storage, arithmetic, device execution, model
definitions and serving. Its design aims to make memory policy and numerical
behavior explicit, with no automatic fallback to an external inference engine.

## Design notes

- [The fit contract](<ADR-0001 — The Fit Contract.md>): preserve precision,
  explicitly degrade only where implemented, or refuse admission.
- [Storage-aware measurement](<The Device Is Part Of The Measurement.md>):
  identify the device and reproduce the workload's access pattern.
- [Scheduling granularity](<Region Granularity Beats Thread Count.md>):
  distinguish synchronization cost from arithmetic cost.
- [Packed arithmetic](<Most Of A Quantised Matmul Is Not The Matmul.md>):
  measure decoding, reduction and staging separately.

These are methodology and design notes, not claims that a particular model or
optimization is production-ready. Current implementation and acceptance evidence
live in the [research index](../docs/research/README.md),
[serving guide](../docs/serving.md), and [HAG integration record](../docs/hag-backend.md).

## Evidence requirements

Each performance claim should identify source and artifact revisions, input
shape, checkpoint format, hardware/software environment, numerical comparison,
resource limits and the measured scope. Report failed gates as well as passing
ones. Operator timing, full-model latency and generated tokens per second are
different measurements.
