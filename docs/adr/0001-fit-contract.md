# ADR-0001 — The fit contract

**Status:** accepted for Beholdd. This decision defines memory-fit behavior for
the native runtime and is independent of any external inference server.

## The question

Every inference server hits the case where the model does not fit. There are exactly
three things you can give up, and you must give up one:

| Tradeoff | Required behavior |
| --- | --- |
| Fidelity | Explicitly identify any supported semantic degradation |
| Latency | Stream exact checkpoint bytes without silently changing precision |
| Execution | Refuse admission and explain the resource shortfall |

The decision is not which is best. It is that **the choice must be named, and be the same
one every time** — a user who cannot tell whether today's worse answer is the model or the
fit policy cannot debug their own product.

## Decision

```rust
pub enum FitPolicy {
    Exact,      // never change what the model computes; stream if it does not fit
    Degrade,    // reduce ctx/offload to fit — but say so, in the plan AND the response
    Refuse,     // do not start; name the shortfall and the ways out
}
```

**`Exact` is the default.** The native CPU runtime supports both resident and streamed
checkpoint bytes. This is not a promise that every model, backend, or memory budget
can execute: unsupported configurations still fail admission. See the
[streaming qualification record](../research/model-streaming.md) for measured scope.

A `Degrade` run must be **identifiable from its own output**. The plan's rationale reaches
the caller, not the log.

## Why latency is the right thing to trade

Memory capacity, storage throughput, and compute latency must be measured together.
Streaming can preserve stored model precision when resident execution does not fit,
but the resulting latency may be unsuitable for a workload. The caller should receive
that tradeoff explicitly rather than an undocumented precision change.

## Consequences

- The fit planner must produce a plan a human can read, including what it would cost to
  make the model fit.
- `Exact` obliges beholdd to have a streaming path for every backend, including GPU ones.
  That is a real constraint on `beholdd-backend` and it is intentional.
- Refusing is a supported outcome, not an error path bolted on. It gets the same care as
  a successful load.

## Related

- Residency is observed, not remembered — `loaded` comes from probing, never bookkeeping.
- Forecast, refuse, measure, quote the measurement — report actual resident bytes next to
  the forecast so the forecast becomes calibratable.
