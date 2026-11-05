# The device is part of the measurement

Streaming inference depends on the storage access pattern as well as arithmetic.
A checkpoint on a rotational disk or network filesystem can behave differently
from one on local solid-state storage, even when the model and executable match.

## Reproducible comparisons

Record the checkpoint digest, storage class, memory budget, cache state, request
sizes, read concurrency and observed I/O alongside elapsed time. Compare equivalent
inputs and separate checkpoint loading, prefill and decode. Sequential throughput
alone does not characterize data-dependent expert reads.

The `beholdd-memory` storage inspector and fit planner report admission evidence.
The repository's `tools/probe_read_concurrency.py` accepts explicit local inputs;
inspect its help and apply resource limits before benchmarking.

## Interpretation

Do not infer a compute regression from elapsed time alone. A moved checkpoint,
filesystem change, cache difference or concurrent workload can dominate a result.
Report those conditions and avoid claiming a general device ranking from one
fixture.

Host-specific paths and private infrastructure are not needed in a public report:
identify the storage type, relevant specifications and tested access pattern.
See [streaming qualification](../docs/research/model-streaming.md) for this
repository's measured results.
