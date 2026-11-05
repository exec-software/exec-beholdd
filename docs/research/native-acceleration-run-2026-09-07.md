# Native acceleration bring-up run — 2026-09-07

This is the durable command/output record for the first prepared-weight Vulkan slice.
It is an operator benchmark, not model inference or an M4 gate. The tree was based on
`main` commit `84f8a2611e55f50b7e06a950f9cd9aa38afd74a4`; the acceleration work was still an
uncommitted working-tree patch when recorded.

## Environment

- Host: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S
- Vulkan: RADV GFX1151, Mesa 25.2.8
- Rust: `rustc 1.97.0-nightly (f964de49b 2026-05-07)`
- Shader compiler: shaderc 2023.8-1build1, SPIRV-Tools 2023.6~rc1-2,
  glslang 14.0.0-2
- Process CPU affinity: logical CPUs 0-31
- CPU frequency policy at launch: `performance`, boost enabled
- Thermal guard: Tctl warning 75 C, hard abort 90 C, 0.25-second polling
- Each timing sample averages eight synchronous calls. For each shape, a CPU sample
  phase runs before and after the GPU phase; `cpu_best` and `conservative_speedup` use
  the faster of those two CPU medians.

The examples were compiled before starting either measured interval:

```sh
cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_bf16 --example bench_vulkan_q8 --offline
```

## BF16

```text
Vulkan device: Radeon 8060S Graphics (RADV GFX1151)
Native BF16 storage with f32 accumulation; one-time upload is reported separately.
Medians contain 8 calls per timed sample; CPU phases bracket GPU.
Shapes follow the Qwen3.6-35B-A3B text configuration (batch-1 decode).
router           [   256 x 2048 ] upload=  0.165381 ms  steady_gpu=  0.051076 ms  cpu_best=  0.126520 ms  cpu_pre/post=  0.126520/  0.128554 ms  conservative_speedup= 2.48x  gpu=  19.1 GiB/s  rel_l2=4.358e-8 max_abs=3.815e-6
expert gate/up   [   512 x 2048 ] upload=  0.243518 ms  steady_gpu=  0.069981 ms  cpu_best=  0.161215 ms  cpu_pre/post=  0.166353/  0.161215 ms  conservative_speedup= 2.30x  gpu=  27.9 GiB/s  rel_l2=4.136e-8 max_abs=3.815e-6
expert down      [  2048 x 512  ] upload=  0.225784 ms  steady_gpu=  0.057699 ms  cpu_best=  0.161556 ms  cpu_pre/post=  0.165278/  0.161556 ms  conservative_speedup= 2.80x  gpu=  33.9 GiB/s  rel_l2=3.533e-8 max_abs=3.815e-6
attention wq     [  4096 x 2048 ] upload=  1.412344 ms  steady_gpu=  0.247161 ms  cpu_best=  0.754160 ms  cpu_pre/post=  0.754160/  0.760859 ms  conservative_speedup= 3.05x  gpu=  63.2 GiB/s  rel_l2=4.152e-8 max_abs=7.629e-6
attention wo     [  2048 x 4096 ] upload=  1.442540 ms  steady_gpu=  0.344784 ms  cpu_best=  0.762219 ms  cpu_pre/post=  0.762219/  0.804944 ms  conservative_speedup= 2.21x  gpu=  45.3 GiB/s  rel_l2=3.573e-8 max_abs=7.629e-6
```

Thermal report:

```json
{"abort_reason":null,"aborted_for_heat":false,"exit_code":0,"max_tctl_c":73.625,"ok":true,"samples":4}
```

## Q8_0

The synthetic fixture changes scale on every adjacent block; a uniform scale cannot
hide incorrect block addressing.

```text
Vulkan device: Radeon 8060S Graphics (RADV GFX1151)
Times include input copy, dispatch, fence wait, and output read; upload is separate.
Medians contain 8 calls per timed sample; CPU phases bracket GPU.
Shapes follow the Qwen3.6-35B-A3B text configuration (batch-1 decode).
router           [   256 x 2048 ] upload=  0.123211 ms  steady_gpu=  0.057290 ms  cpu_best=  0.116552 ms  cpu_pre/post=  0.121523/  0.116552 ms  conservative_speedup= 2.03x  rel_l2=8.351e-7 max_abs=1.526e-4
expert gate/up   [   512 x 2048 ] upload=  0.175870 ms  steady_gpu=  0.059987 ms  cpu_best=  0.150234 ms  cpu_pre/post=  0.150234/  0.156729 ms  conservative_speedup= 2.50x  rel_l2=7.899e-7 max_abs=1.526e-4
expert down      [  2048 x 512  ] upload=  0.174277 ms  steady_gpu=  0.069247 ms  cpu_best=  0.151835 ms  cpu_pre/post=  0.151835/  0.158103 ms  conservative_speedup= 2.19x  rel_l2=4.070e-7 max_abs=5.722e-5
attention wq     [  4096 x 2048 ] upload=  1.622939 ms  steady_gpu=  0.310304 ms  cpu_best=  0.738038 ms  cpu_pre/post=  0.738633/  0.738038 ms  conservative_speedup= 2.38x  rel_l2=8.425e-7 max_abs=2.441e-4
attention wo     [  2048 x 4096 ] upload=  0.860947 ms  steady_gpu=  0.209416 ms  cpu_best=  0.723496 ms  cpu_pre/post=  0.723496/  0.742267 ms  conservative_speedup= 3.45x  rel_l2=1.197e-6 max_abs=4.730e-4
```

Thermal report:

```json
{"abort_reason":null,"aborted_for_heat":false,"exit_code":0,"max_tctl_c":70.625,"ok":true,"samples":3}
```

## Hardware-only correctness gate

The explicitly ignored test was run under the same thermal guard. It passed native BF16
rows that cross packed uint boundaries, the Q8_0 padded final word, and a forced real
64x3 dispatch grid against the CPU oracle.

```text
test vulkan::tests::live_packed_dispatch_covers_tail_words_odd_bf16_rows_and_a_two_dimensional_grid ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 28 filtered out
max_tctl_c=71.25, aborted_for_heat=false
```

## Source manifest

```text
faf0b78db5c46d7a55110787aa9e4049ef1e762bfb57f344490c90451b2e560d  crates/beholdd-backend/src/vulkan.rs
3b7ea5094d5e93b6a50f7d8dd5013f6fdccb5c45be52c5224463106deb76f643  crates/beholdd-backend/shaders/q8_0_matvec.comp
b272d07c51b4f4181316b96f3f4a2c293c177e63a9c753bf9fafb752c0615dea  crates/beholdd-backend/shaders/bf16_matvec.comp
52786312bbde394c28ae134f07707bc9b750339cc91131abe0a4ea6830e88715  crates/beholdd-backend/examples/bench_vulkan_q8.rs
4dd1130d7734d62eb32b7f0fa3fa60d86e22bd91311c13b008510652d421e520  crates/beholdd-backend/examples/bench_vulkan_bf16.rs
2f6394cefcc6004c131ace9cb588008ece6649ce16879badea7a21dffb4a4f4a  crates/beholdd-backend/build.rs
ddde546b7618969083c71b023869a119da87c8aa95e6af947da710e8d494f409  crates/beholdd-backend/Cargo.toml
84bf8d99af45ea37dc1b727a169f94df9135ca4fea809c1b3b7bf65cd2fd349a  crates/beholdd-backend/src/cpu.rs
996208098c70376043bae91e71b488ffe89236c4010d3dba42bc0ac34ef0a14c  crates/beholdd-kernel/src/bf16.rs
```
