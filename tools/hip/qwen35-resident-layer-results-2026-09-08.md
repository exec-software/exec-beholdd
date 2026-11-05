# Qwen3.6 native-BF16 resident MoE-FFN results — 2026-09-08

## Outcome

The real-weight layer-3 experiment found a qualified performance direction for
the Radeon 8060S (`gfx1151`): keep the checkpoint's expert values in BF16, but
rearrange those bits once on the GPU into the row-major Matrix-B layout that
rocWMMA executes efficiently. Combining that prepared layout with a
deterministic 256-thread router reduced the median fixed-route MoE-FFN sublayer
time to **0.378158 ms**.

Across three thermally guarded deep runs, the final candidate produced:

| Run | prepared + serial router | prepared + parallel router | paired speedup | candidate sublayers/s | max Tctl |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.463705 ms | 0.378158 ms | 1.228711x | 2,644.40 | 66.125 C |
| 2 | 0.467382 ms | 0.380739 ms | 1.236879x | 2,626.47 | 67.500 C |
| 3 | 0.463915 ms | 0.375004 ms | 1.237448x | 2,666.64 | 69.500 C |
| median of run medians | **0.463915 ms** | **0.378158 ms** | **1.236879x** | **2,644.40** | — |

The clean paired layout experiment had already reduced the checkpoint-native
column-major path from a 1.328719 ms median of medians to 0.451102 ms, a
2.9455x paired speedup. The parallel router then supplied the independently
measured incremental improvement above. The final two optimizations should not
be multiplied into a synthetic exact speedup because the experiments ran in
separate, DVFS-sensitive processes.

At 40 MoE blocks, `40 * 0.378158 ms = 15.12632 ms`. That is a **66.11
tokens/s MoE-FFN-only ceiling** for this one fixed route. It excludes attention,
recurrent/state-space work, embeddings, the LM head, sampling, scheduling, KV
traffic, and every other full-model cost. It is not an end-to-end token rate.
It does show that the MoE FFNs alone no longer rule out the rumored 50 tokens/s
target on this hardware.

## What remains native precision

Preparation is a bit-preserving transpose, not quantization or conversion.
Every expert BF16 value is copied unchanged from checkpoint-output-major order
to the rocWMMA-friendly Matrix-B order. Once prepared, the checkpoint-order GPU
copy can be released, so production storage is not doubled.

The exact per-layer expert/scalar allocation is 1,619,017,728 bytes
(1,544.016 MiB). Forty such layers require 60.313 GiB. The complete local
Safetensors checkpoint occupies 66.966 GiB, so a one-copy prepared model is
plausible in the 96 GiB unified-memory device, subject to a checked whole-model
plan for KV cache and runtime scratch.

Precision scope is exact and intentionally narrower than “the whole model is
BF16”:

- Expert gate/up/down matrices and the post-SiLU boundary are BF16.
- RMSNorm, router weights, shared-expert scalar gate, logits, routing,
  accumulation, reduction, and residual are F32.
- The CPU oracle mirrors those declared boundaries. It proves mechanical
  agreement with this policy, not parity with an official Transformers forward
  pass.

## Real artifact and provenance

- Checkpoint snapshot:
  `$HF_HUB_CACHE/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0`
- Geometry: layer 3, `H=2048`, `F=512`, 256 routed experts, top 8, one
  `F=512` shared expert, 40 model layers, RMS epsilon `1e-6`.
- Pack: `/tmp/qwen36-layer3-bf16.pack`, 1,619,021,824 bytes.
- Pack SHA-256:
  `9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116`.
- Source shard SHA-256:
  `8f7d72178d3f4431864978e5bcfa4c6cb1c204bc00590644d90bb19d6d522eeb`.
- Source config SHA-256:
  `93a4693fa9d8392fbfccd4b3c9873f4bfdcb14fdede978b123d07d19675efe99`.
- Combined source SHA-256:
  `4a9b834332c76accd0d86cd0acace19d448ba1aecdaf8ab2b90310cdad958c60`.
- Combined build-wrapper SHA-256:
  `bed5483771113aceffe04fe8aa4a4b9434e08b36a4e732d1685facb42d5449fb`.
- Combined device-text SHA-256:
  `d902de80d06ab66842c18fd74d03d06ce210d8ac6f44eba298bc9be814b70ef2`.
- Hardened stage-profiler source SHA-256:
  `b1e60c52579da6e172404680a1f80f7f6c7df1fbfd160fd791f233906591fd4c`.
- Stage-profiler build-wrapper SHA-256:
  `04854107da7569583515e40d3e31239ff58c728e7828edefe777b48e08cc4d94`.
- Public rocWMMA checkout: `ROCm/rocm-libraries` commit
  `bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125`.
- Compiler/runtime: ROCm 7.13 wheel, AMD Clang 23, pinned
  `libamdhip64.so.7` SHA-256
  `eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05`.

An independent mmap audit compared every expert/shared BF16 byte and every
scalar transform in the pack against the Safetensors source. It also checked
the complete header, section boundaries, reserved bytes, and output digest.
The guarded runner independently checked the pack digest and file identity
before and after every final run.

## Final correctness gate

All smoke and deep runs passed. The fixed hidden vector selected experts
`38:20:31:226:200:153:167:50`.

- Prepared gate/up/down matrices for all eight selected experts and the shared
  expert matched the source BF16 bytes exactly.
- Serial prepared output was bit-identical to the checkpoint-layout native
  control.
- Serial and parallel routes selected the exact CPU top-8 IDs.
- Both route-weight sums were 1.0; values were finite, normalized, and ordered.
- Maximum route-weight error against the CPU oracle: `5.215e-8`.
- Route-margin error against the CPU oracle: `2.794e-9`.
- Shared-scale error against the CPU oracle: zero.
- Parallel CPU-relative L2: `1.205e-7`; max absolute error: `1.490e-7`.
- Parallel-versus-serial relative L2: `1.388e-8`; max absolute error:
  `5.960e-8`.
- Repeated outputs and route metadata were bit deterministic within each path.

Static qualification found an exact `gfx1151`, wave32 code object with 12
native BF16 WMMA instructions. The parallel router uses 2,056 bytes of LDS and
30 VGPRs; prepared gate/up uses 45 VGPRs and prepared down uses 48. All checked
kernels have zero private memory, zero spills, no dynamic stack, and no scratch
ISA traffic.

## Experiments and decisions

| Experiment | Representative deep result | Decision |
|---|---:|---|
| Eliminate the route host round trip while retaining checkpoint layout | 1.0331x median-of-three | Correct, but performance NO-GO |
| GPU bit-preserving row-major preparation + rocWMMA | 0.451102 ms, 2.9455x paired | GO |
| Direct checkpoint-layout wave32 BF16 dot products | 0.557929 ms | Correct, slower than prepared WMMA; NO-GO |
| Parallel stable-softmax/top-8 on checkpoint layout | 1.085432x median-of-three | Correct; useful component |
| Prepared WMMA + parallel stable-softmax/top-8 | 0.378158 ms, 1.236879x incremental | GO |

The checkpoint-layout stage profile explains the largest win:

| Stage | Wall time |
|---|---:|
| RMSNorm + BF16 input | 19.186 us |
| Router/shared scalar | 31.409 us |
| Serial softmax/top-8 | 114.104 us |
| Checkpoint-layout gate/up WMMA | 870.233 us |
| SiLU/up to BF16 | 25.438 us |
| Checkpoint-layout down WMMA | 301.676 us |
| Route/shared/residual reduction | 22.722 us |

The expensive part was not BF16 precision; it was making rocWMMA consume the
checkpoint's unfavorable physical orientation. Prepared Matrix-B weights cut
register pressure from 75 VGPRs in each checkpoint-layout WMMA kernel to 45
and 48 VGPRs while retaining the exact values.

A guarded 31-sample wall-clock profile of the final combined chain then
reported:

| Final stage | Idle-sync-adjusted median |
|---|---:|
| RMSNorm + BF16 input | 29.034 us |
| Router/shared scalar | 34.464 us |
| Serial softmax/top-8 control | 104.536 us |
| Parallel softmax/top-8 | 29.045 us |
| Prepared gate/up WMMA | 200.157 us |
| SiLU/up to BF16 | 29.154 us |
| Prepared down WMMA | 132.068 us |
| Route/shared/residual reduction | 27.130 us |
| Complete prepared + parallel chain | 381.788 us |

Individually synchronized stages do not add to the asynchronously queued full
chain, so the table is attribution rather than an additive timing model. It
does make two results clear. Parallel routing removes 75.491 us in isolation
and 80.222 us from the full chain in the same profile. Prepared expert-matrix
execution remains the next constraint. Previously rejected launch/fusion
variants should not be revived without a new mechanism. The next large lever
is useful row occupancy—packing independent agent requests or exact
speculative/MTP verification tokens into the 16-row WMMA tile—followed by
route-aware expert cohorts.

## Reproduce

Create the strict layer pack from the official snapshot (the packer refuses to
overwrite an existing output):

```bash
cd "$BEHOLDD_SOURCE_DIR"
cargo run -p beholdd-model --release \
  --example pack_qwen35moe_bf16_layer -- \
  $HF_HUB_CACHE/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0 \
  3 /tmp/qwen36-layer3-bf16.pack
sha256sum /tmp/qwen36-layer3-bf16.pack
```

Build without executing the GPU:

```bash
cd "$BEHOLDD_SOURCE_DIR"
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_qwen35_resident_layer_prepared_parallel_route.sh \
  /tmp/beholdd-q35-prepared-parallel-route-bench
```

Run with both thermal and pack-identity containment:

```bash
cd "$BEHOLDD_SOURCE_DIR"
tools/hip/run_qwen35_layer_guarded.sh \
  /tmp/beholdd-q35-prepared-parallel-route-bench \
  /tmp/qwen36-layer3-bf16.pack \
  9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
  --warmups 4 --samples 12 --burst 16
```

## Qualification limits and next gate

This remains a fixed-route, single-hidden-vector microbenchmark. Reusing one
expert cohort makes it cache favorable. It does not measure route churn, a
distribution of real activations, attention, or full-model token generation.
Pack v1 itself carries geometry but no payload digest; use the digest-bound
runner above until a future pack format embeds one. Direct execution could use
loader override variables instead of the pinned RUNPATH; the guarded runner
clears `LD_LIBRARY_PATH`, `LD_PRELOAD`, and `LD_AUDIT` before launch.

Production promotion requires:

1. Capture several real layer-3 activation/route cohorts and add rotating-route
   correctness and performance runs.
2. Prepare all model layers during load, release checkpoint-order device
   buffers, and enforce a whole-model VRAM/KV/scratch budget.
3. Compare full hidden states, logits, and generated tokens to an independent
   official-model oracle.
4. Integrate attention and the remaining decoder path, then measure actual
   output tokens/s.

Reference serving were not modified.
