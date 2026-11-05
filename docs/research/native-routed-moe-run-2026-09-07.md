# Native routed-MoE bring-up run — 2026-09-07

This is the durable command/output record for the first arena-backed, device-resident
BF16 routed-expert graph. It is a synthetic operator benchmark, not model inference and
not an M4 gate. The tree was based on `main` commit
`84f8a2611e55f50b7e06a950f9cd9aa38afd74a4`; the acceleration work remained an
uncommitted working-tree patch when recorded.

## Scope

- Qwen3.6-shaped routed dimensions: hidden 2048, expert width 512, top 8.
- The selected working set contains gate, up and down BF16 matrices for eight routes:
  exactly 24 matrices / 48 MiB encoded.
- Conceptual expert order: `[231, 3, 128, 17, 255, 0, 72, 9]`.
- Route weights: `[0.31, 0.19, 0.14, 0.11, 0.09, 0.07, 0.05, 0.04]`.
- The fused path records gate+up+SiLU, down, and deterministic route reduction as three
  dispatches in one command buffer/submission/fence. Intermediate f32 tensors remain in
  Vulkan buffers.
- This does not include Qwen3.6's separate shared expert, router, attention, recurrent
  state, norms, residuals, logits, sampling, or checkpoint I/O.

## Environment and qualification

- Host: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S.
- Vulkan: RADV GFX1151, Mesa 25.2.8, Vulkan device API 1.4.318.
- Rust: `rustc 1.97.0-nightly (f964de49b 2026-05-07)`.
- Shader compiler used by Cargo: shaderc 2023.8-1build1, SPIRV-Tools
  2023.6~rc1-2, glslang 14.0.0-2.
- Thermal guard: package Tctl warning 75 C, hard abort 90 C, 0.25-second polling.
- Preparation and CPU-oracle construction are outside the timed samples. Both timed
  paths include their input host write and final output host read.
- Two warmups precede nine alternating-order samples; medians are reported.

This was not an uncontended host. `Overwatch.exe` had started at 18:57:10 local time,
before this benchmark, and was later observed consuming roughly 270% CPU while the GPU
reported 60% and then 100% busy. Tctl reached 87.375 C. The guard did not abort, but no repeat was
attempted while the competing workload remained. The relative same-process comparison
is useful design evidence; the absolute milliseconds and routed evaluations per second
are not an idle-host qualification.

## Command

The release binary was built before the guarded interval:

```sh
cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_moe --offline

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
  ./target/release/examples/bench_vulkan_moe
```

## Exact output

```text
Vulkan device: Radeon 8060S Graphics (RADV GFX1151)
Qwen3.6-shaped routed experts: H=2048, F=512, K=8; conceptual IDs=[231, 3, 128, 17, 255, 0, 72, 9]
Arena: 24 matrices, 1 block, 48.0 MiB encoded / 64.0 MiB reserved
Timing excludes weight preparation; both paths include input upload and final readback. Medians: 9 samples after 2 warmups.
legacy 24 calls + CPU SwiGLU:   7.292950 ms ( 137.12 routed eval/s)
fused 3 dispatches / 1 submit:   2.136867 ms ( 467.97 routed eval/s)
speedup=3.41x  legacy rel-L2=1.983e-7 max-abs=2.794e-9  fused rel-L2=2.013e-7 max-abs=2.794e-9  accumulate rel-L2=2.101e-8 max-abs=1.490e-8  repeatability=bit-exact
{"abort_reason": null, "aborted_for_heat": false, "exit_code": 0, "max_tctl_c": 87.375, "ok": true, "samples": 2}
```

The benchmark's go gate requires at least 1.25x over the former 24-submit path; this run
measured 3.41x. The fused output also passed overwrite and accumulation comparisons and
was bit-identical across repeated Vulkan calls.

## Hardware correctness gates

Before the competing game load was identified, two short ignored tests were run under
the same thermal guard:

```text
test vulkan::tests::live_weight_arena_suballocates_and_keeps_handles_alive ... ok
max_tctl_c=57.0, aborted_for_heat=false

test vulkan::tests::live_fused_routed_moe_matches_sequential_cpu_and_is_deterministic ... ok
max_tctl_c=59.0, aborted_for_heat=false
```

They cover BF16 and Q8_0 matrices sharing a nonzero-offset arena block, arena-handle
lifetime after dropping the top-level arena value, small odd dimensions, overwrite,
accumulation, deterministic route order, and repeated bitwise-stable GPU output.

## Unselected fast-shader candidates

Three `*_fast.comp` variants were compiled and SPIR-V validated but were not connected
to the host or executed. They replace byte offsets with word offsets, pair BF16 loads,
remove shader-side checks already proven by the host, and use direct two-dimensional
dispatch. Their optimized SPIR-V function bodies contain 38.2–53.7% fewer operations
than the active safe shaders. They remain candidates—not measured improvements—until an
idle, thermally guarded parity and performance run validates their distinct ABI.

## Source manifest

```text
96d613c4f35f57e2a3423785c46b262bc9a1cb43891d30a949a1e5732f734ddf  crates/beholdd-backend/src/vulkan.rs
f1b0d1899f749459521102e043b6c1452253bb91025a3d70d37883b6d112b778  crates/beholdd-backend/src/execution.rs
3b7ea5094d5e93b6a50f7d8dd5013f6fdccb5c45be52c5224463106deb76f643  crates/beholdd-backend/shaders/q8_0_matvec.comp
b272d07c51b4f4181316b96f3f4a2c293c177e63a9c753bf9fafb752c0615dea  crates/beholdd-backend/shaders/bf16_matvec.comp
a1448e565fada6e9639b7fff34ae55570ebc70ec1b7f8a65b8432564373b155a  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up.comp
721be69d080afb0af9818a189aa552c439f6807ddefbfaf201381d02b9b3027e  crates/beholdd-backend/shaders/bf16_routed_moe_down.comp
34145187ceb20729104b5778304accfe79ec9be5aa7cd0e1b5a84efb58c88d29  crates/beholdd-backend/shaders/bf16_routed_moe_reduce.comp
670e6f7491a3d528733a2a66c0b8aaa3fe22112bdbc0f4bd4eca15514388d637  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_fast.comp
3d615ae3e7dfa2ade5ad818c5bc23fcd81ef05759666306ffc9fc5c364d14352  crates/beholdd-backend/shaders/bf16_routed_moe_down_fast.comp
a4feb83785a6998da6f3e00a7f565e66caa2a048aaa710a1f533da5deb30f4d5  crates/beholdd-backend/shaders/bf16_routed_moe_reduce_fast.comp
19f1ffcc2701026b978d654fb2f14877ac5481563b9d88165e0123f026d24204  crates/beholdd-backend/examples/bench_vulkan_moe.rs
a21a196b619be57698d9520ebb7277a48ec949057f7b8e3c91eec8d9cfd70f58  crates/beholdd-backend/build.rs
ddde546b7618969083c71b023869a119da87c8aa95e6af947da710e8d494f409  crates/beholdd-backend/Cargo.toml
```
