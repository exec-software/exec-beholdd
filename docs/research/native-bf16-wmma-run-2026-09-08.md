# Native BF16 WMMA routed-MoE run — 2026-09-08

## Status before route-level measurement

The production/default accelerated path remains the qualified Vulkan per-item
Tree128 program. The reference serving runtime and its live model are outside this
experiment and must not be modified or stopped.

The measured RADV path does not expose a BF16 cooperative-matrix tuple. Shader ISA
inspection for the current paired-load Vulkan kernel shows BF16 pairs unpacked with
integer shifts/masks and accumulated with scalar `v_fmac_f32`; it does not contain a
BF16 dot or WMMA instruction.

A newer ROCm stack already installed in `the isolated ROCm environment` identifies the
Radeon 8060S as `gfx1151`. A minimal wave32 kernel compiled by its Clang 23 toolchain,
loaded through its matching HIP runtime, and dispatched on this GPU produced the
expected result while containing `v_wmma_f32_16x16x16_bf16`. This proves the local
compile/load/dispatch seam and one simple numerical case, but not fragment mapping,
route-level correctness, or a useful performance win.

A second public-API probe used rocWMMA `load_matrix_sync`, `mma_sync`, and
`store_matrix_sync` on asymmetric signed 16-by-16 tiles. All 256 result positions
matched the CPU product within `3.81469727e-6`; 107 positions were not bit-identical.
That validates upstream fragment orientation, replication, and lane scatter on this
device while confirming that WMMA needs tolerance-based qualification rather than an
exact-bit promise.

The next candidate is therefore a standalone HIP/rocWMMA research harness for the
qualified routed shape `H=2048`, `F=512`, `K=8` and fixed batch sizes
`N in {1,2,4,8,16}`. It keeps model weights in BF16 and accumulators in F32, but must
round activations to BF16 at WMMA boundaries. That activation conversion is an
explicit precision-policy change from the current Vulkan path, which uses BF16
weights with F32 activations and F32 accumulation.

All GPU work on this host must run through:

```text
python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- ...
```

## Predeclared research gate

This stage answers two questions separately: whether the WMMA kernel is mechanically
correct for its declared BF16 arithmetic, and whether the hardware path is fast
enough to justify a backend integration. It cannot authorize a production default or
a model precision change by itself.

The harness must:

- use the matching ROCm 7.13 compiler, device libraries, and `libamdhip64.so.7` from
  the existing Python environment rather than mixing them with the system ROCm 5.7
  headers/runtime;
- use persistent device allocations and keep weight preparation outside steady-state
  timing;
- compute gate and up projections, SiLU-times-up, down projection, and deterministic
  ascending-route weighted reduction for eight routes;
- report device-resident execution separately from any host-to-device input/metadata
  and device-to-host output transfers;
- use synchronized wall-clock timing because HIP event timing is not trustworthy in
  the current wheel/runtime combination;
- validate multiple nontrivial deterministic inputs and route weights against a CPU
  oracle that explicitly models both BF16 activation boundaries;
- report relative L2 error, maximum absolute error, repeatability, launch count, warmup,
  sample count, and maximum observed Tctl;
- refuse invalid shapes or launch geometry before dispatch; and
- leave the existing Vulkan default and all external runtimes untouched.

## Verified toolchain boundary

The host has two incompatible ROCm generations. Ubuntu's ROCm 5.7 runtime exposes
only the CPU agent and has no native `gfx1151` support. Its Clang 18 can emit the WMMA
instruction and a valid code object, but that does not make the system runtime usable.

The isolated development environment contains a coherent development/runtime set under:

```text
$ROCM_CORE
```

The relevant snapshot is ROCm `7.13.0a20260513`, HIP
`7.13.26183-83e9908b71`, and AMD Clang 23. Its `rocminfo` exposes both the exact
`gfx1151` GPU and a later `aie2p` DSP/NPU agent. A backend must choose device type GPU
and exact ISA `gfx1151`; enumerating the final non-CPU agent would select the wrong
processor and is unsafe on this machine.

The wheel ships versioned runtime libraries without ordinary linker-name symlinks.
Using `-lamdhip64` or `-lhsa-runtime64` can therefore fall through to Ubuntu's ROCm
5.7. Reproducible builds must use explicit include, Clang resource, and device-bitcode
paths, and host launchers must link the exact versioned library
`libamdhip64.so.7`/`libhsa-runtime64.so.1` with a matching rpath or use an absolute
`dlopen`. Unqualified `hipcc` is also rejected because it has been observed to mix the
wheel compiler with `/usr/include/hip` 5.7 and to plan the wrong GPU architecture.

Mechanical correctness requires finite outputs, deterministic repeated output bits on
this device, relative L2 error no greater than `5e-4`, and maximum absolute error no
greater than `1e-3` versus the declared BF16 CPU oracle. The absolute bound prevents a
single badly mapped output from disappearing inside a global norm. A result outside
either bound is a no-go regardless of speed.

Performance is compared with the already-qualified Vulkan per-item Tree128 program at
the same `H/F/K/N`. The primary metric is complete routed-MoE operator evaluations per
second, not tokens per second. For a backend prototype to proceed, the synchronized
device-resident WMMA chain must meet both conditions in three complete guarded runs:

- at least `1.50x` the Vulkan per-item median at one batch size; and
- no regression at the smallest batch size proposed for automatic routing.

The `1.50x` threshold is intentionally above ordinary run-to-run noise and integration
cost. Results that win only at larger `N` may justify a future cohort-batching path,
but they do not authorize routing interactive `N=1` decode through it.

## Production boundary

Even a successful primitive remains research-only until an end-to-end model gate
compares logits, greedy tokens, long-context stability, and real decode throughput
against the native-weight reference. Any future backend must also make runtime and
code-object compatibility explicit, fail closed when `gfx1151` or the matching HIP
runtime is absent, and preserve Vulkan/CPU fallback behavior without silently changing
precision.

## Results

### Native path and correctness

The standalone harness, strict build wrapper, run guide, and full raw results live in
`tools/hip`. The build used AMD Clang 23 from the isolated ROCm 7.13 wheel and public
rocWMMA headers from `ROCm/rocm-libraries` commit
`bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125`. It refused the build unless the
extracted device image was tagged `EF_AMDGPU_MACH_AMDGCN_GFX1151`, contained
`v_wmma_f32_16x16x16_bf16`, and the host binary resolved the wheel's exact
`libamdhip64.so.7` rather than Ubuntu ROCm 5.7.

All `N in {1,2,4,8,16}` passed both the resident and copy-inclusive paths. Outputs
were finite and bit-identical over repeated execution. Against the CPU oracle that
mirrors both BF16 WMMA activation boundaries, relative L2 error was
`1.097036e-4–1.699449e-4` and maximum absolute error was at most
`3.084715e-7`, comfortably inside the predeclared mechanical gate.

The second oracle quantified the intentional precision change. Against current
Beholdd semantics—BF16 weights, original F32 token input, F32 post-SiLU activation,
and F32 accumulation—the WMMA candidate differed by `0.002652090–0.002799975`
relative L2 (`0.265–0.280%`) and at most `5.584268e-6` absolute. This is not a
kernel-correctness failure, but it is large enough to require model/logit/token quality
gates before any automatic use.

### Performance

Three guarded WMMA runs used 4 warmups, 9 samples, and bursts of 8 operators per
sample. The fair comparison below uses its `host_visible` envelope, which includes
pinned H2D input and route metadata, five launches, D2H output, and a final stream
synchronization. Weight transformation/allocation/upload remain outside timing.

A fresh release build of the hardened Vulkan benchmark then completed three guarded
runs. Its per-item Tree128 program likewise includes input upload and final readback,
keeps weights prepared/resident, and uses balanced bursts of at least 16 routed
evaluations per timing cell. No HIP and Vulkan GPU workloads overlapped.

| batch `N` | Vulkan per-item median ms | WMMA host-visible median ms | ratio of medians | individual-run ratio range | decision |
|---:|---:|---:|---:|---:|:---|
| 1 | 0.437562 | 0.310579 | 1.409x | 1.308–1.448x | win, below 1.50x gate |
| 2 | 0.665986 | 0.299393 | 2.224x | 2.077–2.299x | prototype pass |
| 4 | 0.937472 | 0.304728 | 3.076x | 2.767–3.262x | prototype pass |
| 8 | 1.391090 | 0.303418 | 4.585x | 3.885–4.713x | prototype pass |
| 16 | 2.329220 | 0.334783 | 6.957x | 6.899–8.294x | prototype pass |

The median WMMA resident time stayed nearly flat at `0.289872–0.313171 ms` across
the five useful-row counts. At `N=16` it delivered a median
`52,195.45 routed evaluations/s`; the copy-inclusive median was `47,792.10/s`.
These are rates for one routed-MoE operator, not full-model tokens per second.

The three deep WMMA runs peaked at Tctl `67.375`, `66.5`, and `65.75` degrees C.
The fresh Vulkan controls peaked at `64.875`, `65.125`, and `66.375` degrees C. A
later dual-oracle validation entered the warning range at `81.875` degrees C because
two approximately 1.46-second CPU oracles ran on the same APU; it remained below the
90-degree abort threshold and no further GPU work followed.

### Decision

The native WMMA route clears the predeclared performance threshold for backend
prototyping at every `N >= 2`, and even `N=1` was consistently faster than the Vulkan
control. It does **not** become a production default. The result is scoped to an ideal
high-overlap cohort in which every item visits the same eight resident experts, its
weights use a one-time transposed Matrix-B representation, and its BF16 activation
boundaries deliberately differ from the current engine's arithmetic.

The next kernel experiment should fuse gate, up, SiLU, and the second BF16 conversion.
One wave can reuse each input fragment across gate and up, hold two F32 accumulators,
stage the two 16-by-16 results through 2 KiB of LDS, and write the BF16 activation
directly. This removes the 1 MiB gate/up scratch allocation and one launch without
adding another precision boundary. It must be benchmarked against this frozen
five-launch control with identical inputs, weight layout, synchronization, and
numerical gates. Scheduler integration remains later work: partial expert overlap and
real model-level quality/throughput must qualify before cohort routing is automatic.

Before implementing that iteration, its promotion gate is fixed as follows:

- retain the five-launch path as a same-process control and counterbalance A/B order;
- require bit-identical candidate/control output for every `N in {1,2,4,8,16}`, both
  mechanical oracle bounds, and repeated-run bit determinism;
- reject the build if the fused kernel spills or uses more than 64 VGPRs per wave;
- run three guarded measurements with at least 9 samples and bursts of at least 16;
- require median-of-three paired speedup of at least `1.10x` at `N=1` and `1.08x` at
  `N=16`, no individual-run ratio below `1.03x`, and no candidate p90 more than 5%
  slower than its paired control; and
- classify a result that qualifies only at `N=16` as cohort-only rather than an
  interactive decode path.

### Fused gate/up/SiLU iteration

The fused candidate passed its static and numerical gates but failed the
predeclared performance gate. The final build contained exactly two native BF16
WMMA instructions, used 54 VGPRs and 2,048 bytes of LDS, used wave32 with a
32-thread maximum workgroup, and allocated no private scratch or spills. All
candidate outputs were bit-identical to the frozen five-launch control for every
tested N and both timing envelopes.

Three guarded runs tightened the protocol to four warmups, twelve paired samples
in balanced multiples of four, and bursts of sixteen. Median-of-three paired
`control/candidate` ratios were:

| batch `N` | resident | host-visible | required | decision |
|---:|---:|---:|---:|:---|
| 1 | 0.981638x | 0.972397x | 1.10x | no-go |
| 16 | 0.975637x | 0.972665x | 1.08x | no-go |

Every individual run also missed the `1.03x` floor, although all p90 bounds
passed. Peak Tctl was `62.75 C`, `62.25 C`, and `61.875 C`; no run approached
the thermal abort threshold. The candidate therefore remains an explicit
negative experiment and is not a backend option.

Static control/candidate comparison explains why removing a launch did not pay.
The control exposes 512 gate/up waves followed by 2,048 lightweight SiLU waves;
the fused grid exposes only 256 waves. Each fused lane then executes eight
serial exp/div/BF16 epilogues after its WMMA work and a 2 KiB LDS handoff. The
compiler still reports occupancy 16, so the limiting change is finite-grid
latency hiding rather than an occupancy-tier drop. The saved input is a small,
cache-hot 64 KiB matrix and the gate/up scratch traffic is minor beside 48 MiB
of streamed weights.

The complete static and runtime records are
`tools/hip/fused-gate-up-static-2026-09-08.md` and
`tools/hip/fused-gate-up-results-2026-09-08.md`. The next experiment should
preserve the frozen five-kernel geometry and test HIP Graph replay first. That
isolates host dispatch overhead without changing arithmetic or sacrificing
wave-level parallelism. Down-WMMA/route-reduction fusion remains a second,
higher-risk candidate because it can remove an intermediate and a launch while
retaining one expert wave per route.

### HIP Graph replay iteration

HIP Graph replay passed every static, topology, poison, and numerical gate, but
failed its predeclared performance gate. It replayed the frozen five-kernel
arithmetic unchanged: the resident graph had exactly five kernels, four edges,
and one root; the host-visible graph had five kernels plus three pinned copies,
seven edges, and one root. All `N in {1,2,4,8,16}` produced finite,
deterministic output that was bit-identical between direct submission and graph
replay in both envelopes. Mechanical error and current-semantics policy drift
were unchanged from the baseline.

Three guarded qualification runs used four warmups, twelve counterbalanced
paired samples, and bursts of 32 operators. Paired speedup is `direct / graph`:

| run | `N=1` resident | `N=1` host-visible | `N=16` resident | `N=16` host-visible |
|---:|---:|---:|---:|---:|
| 1 | 0.977392x | 0.974878x | 0.990509x | 1.003240x |
| 2 | 0.979121x | 0.976538x | 0.984913x | 1.002127x |
| 3 | 0.972751x | 0.973575x | 0.981748x | 1.008143x |
| median of three | **0.977392x** | **0.974878x** | **0.984913x** | **1.003240x** |

Every resident run missed the `1.02x` single-run floor, and the aggregate
resident results missed the `1.08x` (`N=1`) and `1.05x` (`N=16`) promotion
thresholds. Host-visible replay also missed its `1.03x` threshold at both batch
sizes. This is a particularly clear no-go because the frozen direct path makes
five `hipGetLastError` polls per operator while graph replay makes one checked
launch, an asymmetry favorable to the graph candidate.

The guarded smoke test peaked at Tctl `62.125 C`; the three full runs peaked at
`63.0 C`, `63.25 C`, and `64.0 C`. The complete decision record is
`tools/hip/graph-replay-results-2026-09-08.md`; static provenance and topology
checks remain in `tools/hip/graph-replay-static-2026-09-08.md`. Graph replay
therefore remains a documented negative experiment rather than a backend
option. The next acceleration attempt should target material device work or
intermediate traffic, with down-WMMA/route-reduction fusion retained as the
next separately gated candidate.

### Down-WMMA/route-reduction fusion iteration

The final micro-kernel candidate fused the native-BF16 down projection with
the rank-ordered FP32 route reduction. Each of 128 workgroups used eight wave32
expert waves, staged one `8 x 16 x 16` FP32 tile in 8 KiB of LDS, crossed one
barrier, and wrote the reduced hidden tile directly. The exact gfx1151 image
used 51 VGPRs, two native BF16 WMMA instructions, no scratch or spills, and a
256-thread workgroup.

Correctness passed the full adversarial matrix: every `N in {1,2,4,8,16}` in
both envelopes, with original, reversed-rank, duplicate/missing-expert, and
one-hot route tables. Candidate output was deterministic and bit-identical to
the frozen control in every case. Both the mechanical BF16 oracle and the
separate current-F32-policy bounds passed.

Three guarded runs used four warmups, twelve counterbalanced samples, and
bursts of sixteen. Paired speedup is `frozen control / fused candidate`:

| run | `N=1` resident | `N=1` host-visible | `N=16` resident | `N=16` host-visible |
|---:|---:|---:|---:|---:|
| 1 | 1.028339x | 1.025724x | 1.024441x | 1.027226x |
| 2 | 1.041357x | 1.039298x | 1.022543x | 1.017043x |
| 3 | 1.031432x | 1.014909x | 1.025584x | 1.034550x |
| median of three | **1.031432x** | **1.025724x** | **1.024441x** | **1.027226x** |
| required | **1.10x** | **1.10x** | **1.08x** | **1.08x** |

All p90 bounds passed, but only four of twelve individual cells cleared the
`1.03x` per-run floor and every aggregate cell missed its decision line. The
three full runs reported peak Tctl of `58.25 C`, `58.25 C`, and `58.75 C`.
The candidate is therefore a NO-GO and is not a backend option.

The saved fixed 1 MiB expert-output write, 64 KiB–1 MiB reducer read, and one
launch are small beside 48 MiB of BF16 weights. Along with the failed gate/up
fusion and Graph replay experiments, this closes the current launch-level
micro-optimization branch. The next useful test is a real-weight,
device-resident routed layer that removes host routing and synchronization
across RMSNorm, router/top-k, shared+routed experts, residual, and persistent
state. Complete static and runtime records are in
`tools/hip/down-reduce-fused-static-2026-09-08.md` and
`tools/hip/down-reduce-fused-results-2026-09-08.md`.
