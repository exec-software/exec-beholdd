# ADR-0001 — The fit contract

The accepted decision is maintained in
[docs/adr/0001-fit-contract.md](../docs/adr/0001-fit-contract.md) to avoid
duplicating policy in the notebook.

Beholdd exposes named fit policies:

- `Exact`: preserve checkpoint precision and use supported streaming where
  resident execution does not fit.
- `Degrade`: require an explicit, implemented semantic change and disclose it.
  The current serving path refuses when the requested degradation has no executor.
- `Refuse`: do not begin execution when admission requirements cannot be met.

A policy value does not establish backend or architecture support. See the
[serving configuration](../docs/serving.md#configuration) and
[streaming research](../docs/research/model-streaming.md) for implemented scope.
