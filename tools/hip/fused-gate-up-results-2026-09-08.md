# Fused gate+up+SiLU qualification — 2026-09-08

## Decision

**No-go.** The four-launch fused candidate is mechanically correct and exactly
matches the frozen five-launch control, but it is consistently slower at both
interactive and full-tile batch sizes. It remains an isolated research harness
and must not replace the frozen control or any Beholdd default.

The predeclared aggregate gate required median-of-three paired speedup of at
least `1.10x` at `N=1` and `1.08x` at `N=16`, independently for resident and
host-visible timing. Every individual run also had to reach `1.03x`, with
candidate p90 no more than `1.05x` the paired control p90. All four aggregate
cells missed their speed threshold, and every individual paired median was
below `1.03x`. All p90 bounds passed.

## Frozen build evidence

```text
compiler=AMD clang 23.0.0git 43215c73116c407735c85a180d174f718798c328+PATCHED:2506c552d8428e2cc1778bef048b20f818e06bb3
rocwmma_commit=bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125
rocwmma_header_tree_sha256=951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750
baseline_source_sha256=8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80
candidate_source_sha256=fd2ed37e4e7a1b7883f61d7e5e429fda4b38acf11e08dbf7d120064d8dffef9d
code_object_target=EF_AMDGPU_MACH_AMDGCN_GFX1151
code_object_sha256=9c07adb5d4e713000c7a6500de397cdfdf9a5c414d4982f9ad535119972c9b78
fused_native_wmma_instruction_count=2
fused_resource=vgpr=54 private_bytes=0 lds_bytes=2048 sgpr_spills=0 vgpr_spills=0 wavefront=32 max_workgroup=32 dynamic_stack=false
libamdhip64_sha256=eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05
```

The build wrapper rejects a different or dirty rocWMMA header tree, a changed
frozen control, a non-gfx1151 code object, anything other than exactly two
native BF16 WMMA instructions in the fused symbol, resource-envelope drift, or
resolution to a HIP runtime other than the selected ROCm 7.13 wheel.

## Correctness

A guarded smoke and all three qualification runs validated every
`N in {1,2,4,8,16}` in both resident and pinned-copy host-visible envelopes.
The candidate and control were bit-identical in every cell and each path was
bit-deterministic over repeated execution.

Against the CPU oracle that mirrors both BF16 activation boundaries, relative
L2 error was `1.097036e-4–1.699449e-4` and maximum absolute error was at most
`3.084715e-7`. Against current Beholdd F32-activation semantics, the unchanged
policy drift was `0.002652090–0.002799975` relative L2 with maximum absolute
error at most `5.584268e-6`.

## Counterbalanced performance

Each qualification process used four warmups, twelve paired samples, and a
burst of sixteen complete routed-MoE operators per timing sample. Within every
sample, control-first/candidate-first order alternated. Timing used synchronized
`steady_clock` wall time; HIP events were not used.

`paired` is the median of the twelve per-sample `control_ms / candidate_ms`
ratios. A value below one means the fused candidate is slower.

| run | `N` | envelope | control median ms | control p90 ms | candidate median ms | candidate p90 ms | paired p10 | paired median | paired p90 | candidate/control p90 | run gate |
|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 1 | resident | 0.306948 | 0.312537 | 0.314645 | 0.319874 | 0.928307 | 0.990532 | 1.017059 | 1.023477 | fail |
| 1 | 1 | host-visible | 0.304088 | 0.314159 | 0.315040 | 0.321056 | 0.930460 | 0.980816 | 1.002245 | 1.021955 | fail |
| 1 | 16 | resident | 0.300475 | 0.304636 | 0.307598 | 0.312234 | 0.949927 | 0.975637 | 0.991823 | 1.024939 | fail |
| 1 | 16 | host-visible | 0.312887 | 0.319991 | 0.322105 | 0.329187 | 0.953062 | 0.981904 | 0.994097 | 1.028738 | fail |
| 2 | 1 | resident | 0.286015 | 0.294011 | 0.298189 | 0.301439 | 0.927004 | 0.970111 | 1.019242 | 1.025265 | fail |
| 2 | 1 | host-visible | 0.293886 | 0.300438 | 0.303067 | 0.309424 | 0.935270 | 0.972397 | 1.015921 | 1.029908 | fail |
| 2 | 16 | resident | 0.293375 | 0.300351 | 0.299804 | 0.307463 | 0.942828 | 0.977858 | 1.008514 | 1.023679 | fail |
| 2 | 16 | host-visible | 0.311845 | 0.313825 | 0.320115 | 0.325430 | 0.942690 | 0.972665 | 0.993478 | 1.036979 | fail |
| 3 | 1 | resident | 0.295397 | 0.308528 | 0.302172 | 0.308540 | 0.936860 | 0.981638 | 1.041454 | 1.000038 | fail |
| 3 | 1 | host-visible | 0.294202 | 0.302138 | 0.305140 | 0.310181 | 0.936189 | 0.967230 | 1.010481 | 1.026621 | fail |
| 3 | 16 | resident | 0.294305 | 0.299403 | 0.301919 | 0.306604 | 0.942433 | 0.973589 | 1.007486 | 1.024051 | fail |
| 3 | 16 | host-visible | 0.311089 | 0.312861 | 0.321035 | 0.324172 | 0.957682 | 0.968764 | 0.993161 | 1.036152 | fail |

The aggregate result is the median of the three paired medians:

| `N` | envelope | median-of-three paired ratio | required | decision |
|---:|:---|---:|---:|:---|
| 1 | resident | 0.981638x | 1.10x | no-go |
| 1 | host-visible | 0.972397x | 1.10x | no-go |
| 16 | resident | 0.975637x | 1.08x | no-go |
| 16 | host-visible | 0.972665x | 1.08x | no-go |

Thus the candidate regressed by roughly `1.9–2.8%`, depending on cell, despite
removing one launch and the global gate/up scratch round trip. The likely cause
must be established from control/candidate ISA and occupancy before another
fusion variant is attempted.

## Thermal record

Every GPU process ran under `tools/thermal_guard.py` with warning/abort
thresholds of `75/90 C` and a `0.25 s` poll period. Peak Tctl was `63.25 C` for
the smoke and `62.75 C`, `62.25 C`, and `61.875 C` for the three qualification
runs. No run was thermally aborted. The qualification processes deliberately
returned exit code 3 because their enforced performance gate failed; the
thermal supervisor itself reported `ok=true` and `aborted_for_heat=false`.

