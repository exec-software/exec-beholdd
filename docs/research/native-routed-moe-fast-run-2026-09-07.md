# Native paired-load routed-MoE qualification — 2026-09-07

This is the durable qualification record for Beholdd v2's paired-load BF16 routed-MoE
kernels. It supersedes the performance conclusion, but not the historical record, in
`native-routed-moe-run-2026-09-07.md`: that first run was contaminated by a game using
the same APU. This run was made after the competing process exited. It remains a
synthetic operator benchmark, not full-model inference and not an M4 gate.

The tree was based on `main` commit
`84f8a2611e55f50b7e06a950f9cd9aa38afd74a4`; all acceleration work remained an
uncommitted working-tree patch when recorded. The golden reference deployment runtime and the model
served by application service were not changed or exercised.

## Change qualified

The fast path keeps packed native BF16 weights and the same one-submit, three-stage
graph, but changes the hot shader contract:

- gate/up and down consume two BF16 values from each 32-bit weight word;
- f32 sources are exposed as aligned `vec2` pairs;
- gate/up dispatches `(F, K, 1)` and down dispatches `(H, K, 1)`, avoiding flattened
  workgroup division and modulo;
- route metadata contains word offsets rather than byte offsets for fast dispatches;
- checks already proven by the host are removed from the inner shader loops;
- the deterministic fixed reduction order, route order, overwrite/accumulate behavior,
  device-resident intermediates, barriers, command buffer, submission, fence, and final
  readback remain explicit.

The host validates even nonzero H/F, 4-byte relative weight alignment, every matrix word
extent, nonwrapping H/F/K products, scratch/descriptor bounds, and direct-dispatch
limits before recording. `routed_moe_bf16_fast` refuses an ineligible call.
`routed_moe_bf16` now selects the qualified fast path for eligible shapes and otherwise
falls back to the retained `routed_moe_bf16_guarded` implementation.

Pipeline initialization also destroys any partial handles returned with a Vulkan
pipeline-creation failure; all three new handles participate in both ordinary and
partial-initialization cleanup.

## Scope

- Qwen3.6-shaped routed dimensions: H=2048, F=512, K=8.
- Gate, up, and down native BF16 matrices for eight routes: 24 matrices / exactly
  48 MiB encoded in one budgeted 64 MiB arena block.
- Conceptual expert order: `[231, 3, 128, 17, 255, 0, 72, 9]`.
- Route weights: `[0.31, 0.19, 0.14, 0.11, 0.09, 0.07, 0.05, 0.04]`.
- Weight preparation and CPU-oracle construction are outside timed samples. All timed
  paths include the input host write, final host read, and fence wait.
- Each reported median contains 12 samples after two warmups. The benchmark cycles all
  six permutations of legacy, guarded, and fast order twice.
- This excludes the shared expert, router, attention/DeltaNet, recurrent state, norms,
  residuals, logits, sampling, checkpoint I/O, and MTP verification.

## Environment and thermal policy

- Host: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S.
- Vulkan device: RADV GFX1151, Mesa 25.2.8, device API 1.4.318.
- Rust: `rustc 1.97.0-nightly (f964de49b 2026-05-07)`.
- Cargo shader compiler: shaderc 2023.8-1build1 / glslang 14.0.0-2.
- All eight generated modules passed `spirv-val --target-env vulkan1.2`; the fast
  modules declare only `OpCapability Shader`.
- Package Tctl warning: 75 C; hard abort: 90 C; polling: 0.25 seconds.
- Before qualification the GPU reported 5-7% desktop background activity and Tctl was
  approximately 55-57 C. No NPU path was used.

## Commands

The release binary was built under the guard but outside the measured runs:

```sh
python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_moe --offline

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  ./target/release/examples/bench_vulkan_moe
```

The second command was executed three times against the same final binary and source.

## Results

| Run | Legacy ms | Guarded ms | Fast ms | Legacy/safe | Safe/fast | Legacy/fast | Max Tctl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.402044 | 0.675132 | 0.560211 | 3.56x | 1.21x | 4.29x | 56.875 C |
| 2 | 2.561007 | 0.637241 | 0.557011 | 4.02x | 1.14x | 4.60x | 57.750 C |
| 3 | 2.553092 | 0.612955 | 0.569469 | 4.17x | 1.08x | 4.48x | 57.875 C |
| Median of run medians | 2.553092 | 0.637241 | 0.560211 | 4.01x | 1.14x | 4.56x | — |

The paired-load path won all three runs. Its median-of-run-median 0.560211 ms represents
about 83.7 GiB/s of encoded-weight traffic before counting activation and metadata
traffic, versus 73.6 GiB/s for the guarded path. This is a repeatable **7-17% reduction
in elapsed time per run** (1.08-1.21x speedup), not a full-token throughput claim. The
benchmark requires fast to beat guarded by at least 1.05x as well as guarded to beat the
legacy path by at least 1.25x.

Representative exact output (run 3):

```text
Vulkan device: Radeon 8060S Graphics (RADV GFX1151)
Qwen3.6-shaped routed experts: H=2048, F=512, K=8; conceptual IDs=[231, 3, 128, 17, 255, 0, 72, 9]
Arena: 24 matrices, 1 block, 48.0 MiB encoded / 64.0 MiB reserved
Timing excludes weight preparation; all paths include input upload and final readback. Medians: 12 samples after 2 warmups.
legacy 24 calls + CPU SwiGLU:   2.553092 ms ( 391.68 routed eval/s)
guarded/safe 3 dispatches / 1 submit:   0.612955 ms (1631.44 routed eval/s)
fast paired-load, 3 dispatches / 1 submit:   0.569469 ms (1756.02 routed eval/s)
ratios: legacy/safe=4.17x  safe/fast=1.08x  legacy/fast=4.48x
legacy rel-L2=1.983e-7 max-abs=2.794e-9  guarded rel-L2=2.013e-7 max-abs=2.794e-9  fast rel-L2=2.936e-7 max-abs=3.754e-9
guarded accumulate rel-L2=2.101e-8 max-abs=1.490e-8  fast accumulate rel-L2=2.511e-8 max-abs=1.490e-8  repeatability=bit-exact
{"abort_reason": null, "aborted_for_heat": false, "exit_code": 0, "max_tctl_c": 57.875, "ok": true, "pid": 1856650, "residual_group_cleaned": false, "samples": 1}
```

## Correctness and fallback qualification

All four ignored live Vulkan tests passed together under the thermal guard at a maximum
Tctl of 62.875 C. The new even-shape test uses H=130, F=66, K=3 so paired loops cross a
64-lane iteration and the final reduction workgroup has a tail. It proves CPU parity,
guarded/fast parity, fast overwrite and accumulation, automatic fast selection, and
bit-exact repeated fast output. The existing H=5, F=3 test proves the guarded odd-width
path still works, the explicit fast call refuses it, and refusal leaves the output
sentinel unchanged.

The Qwen-shaped benchmark passed the same CPU-relative numerical gate on every run:

- guarded: relative L2 `2.013e-7`, max absolute `2.794e-9`;
- fast: relative L2 `2.936e-7`, max absolute `3.754e-9`;
- fast accumulate: relative L2 `2.511e-8`, max absolute `1.490e-8`;
- repeated output: bit exact within each implementation.

## Interpretation

This qualifies the paired-load kernels and automatic selection for the routed BF16
operator. It does not establish 50 generated tokens/s. Native BF16 must read roughly two
bytes per active weight, so 50 single-token evaluations/s cannot be inferred from one
48 MiB layer slice. Full-model work must next eliminate the remaining per-layer host
boundary, integrate the shared expert and attention/DeltaNet graph, and measure real
checkpoint token parity and decode throughput. MTP verification and expert-major reuse
remain the paths that can reduce target-weight traffic per accepted token without
silently lowering weight precision.

## Source manifest

```text
f2f1ac1635c23a75c54b62edace5f88f6b6c244625a0a291fd717ac201b42077  crates/beholdd-backend/src/vulkan.rs
f1b0d1899f749459521102e043b6c1452253bb91025a3d70d37883b6d112b778  crates/beholdd-backend/src/execution.rs
84bf8d99af45ea37dc1b727a169f94df9135ca4fea809c1b3b7bf65cd2fd349a  crates/beholdd-backend/src/cpu.rs
996208098c70376043bae91e71b488ffe89236c4010d3dba42bc0ac34ef0a14c  crates/beholdd-kernel/src/bf16.rs
3b7ea5094d5e93b6a50f7d8dd5013f6fdccb5c45be52c5224463106deb76f643  crates/beholdd-backend/shaders/q8_0_matvec.comp
b272d07c51b4f4181316b96f3f4a2c293c177e63a9c753bf9fafb752c0615dea  crates/beholdd-backend/shaders/bf16_matvec.comp
a1448e565fada6e9639b7fff34ae55570ebc70ec1b7f8a65b8432564373b155a  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up.comp
721be69d080afb0af9818a189aa552c439f6807ddefbfaf201381d02b9b3027e  crates/beholdd-backend/shaders/bf16_routed_moe_down.comp
34145187ceb20729104b5778304accfe79ec9be5aa7cd0e1b5a84efb58c88d29  crates/beholdd-backend/shaders/bf16_routed_moe_reduce.comp
670e6f7491a3d528733a2a66c0b8aaa3fe22112bdbc0f4bd4eca15514388d637  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_fast.comp
3d615ae3e7dfa2ade5ad818c5bc23fcd81ef05759666306ffc9fc5c364d14352  crates/beholdd-backend/shaders/bf16_routed_moe_down_fast.comp
a4feb83785a6998da6f3e00a7f565e66caa2a048aaa710a1f533da5deb30f4d5  crates/beholdd-backend/shaders/bf16_routed_moe_reduce_fast.comp
e4692c8e720bdbce75bbc4d684af083f38cd4d3fd4eb54cabca8643645872c1e  crates/beholdd-backend/examples/bench_vulkan_moe.rs
3a2dbb5ddb76386407f8d04ed769851110a27c796768619492a23f1d2d7327ba  crates/beholdd-backend/build.rs
ddde546b7618969083c71b023869a119da87c8aa95e6af947da710e8d494f409  crates/beholdd-backend/Cargo.toml
f63408c241b41ce44a6efab84b5a094a4d0f5ee4b449e7b176ad0dff91b3a94b  Cargo.lock
```
