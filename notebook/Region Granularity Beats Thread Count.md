# Region granularity and thread count

Thread count is only one input to parallel execution. The amount of work between
barriers, scheduling overhead, memory traffic and affinity can determine whether
additional workers improve a workload.

## Controlled experiment

Keep input values, operation count, checkpoint bytes and numerical contract fixed.
Vary thread count and region structure separately. Measure phase timings rather
than attributing every end-to-end change to the matrix kernel. Record physical
cores, SMT configuration and other active workloads.

Compare many small regions with fewer larger regions before choosing a global
thread default. A result for one shape or architecture does not establish a
universal recommendation to enable or disable SMT.

## Beholdd tooling

`crates/beholdd-backend/examples/bench_matvec_threads.rs` provides an explicit
thread-scaling benchmark. The [native acceleration program](../docs/research/native-acceleration.md)
tracks device execution separately from CPU scheduling.

Kernel ratios are not full-model throughput. Preserve numerical comparisons and
account for storage and synchronization when translating an operator improvement
into a runtime hypothesis.
