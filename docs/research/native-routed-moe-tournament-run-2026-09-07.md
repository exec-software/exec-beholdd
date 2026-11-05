# Native routed-MoE timestamp and workgroup tournament — 2026-09-07

This is the durable qualification record for Beholdd v2's second native BF16
routed-MoE optimization. It follows
`native-routed-moe-fast-run-2026-09-07.md` and keeps that run's 64-lane paired-load
path as the matched control. This remains a synthetic H=2048/F=512/K=8 operator
benchmark, not a full-model token-throughput or M4 claim.

The tree was based on `main` commit
`84f8a2611e55f50b7e06a950f9cd9aa38afd74a4`; all acceleration changes remained an
uncommitted working-tree patch. The working reference deployment runtime, application service, its serving model,
and the NPU were not changed or exercised.

## Measurement seam

An explicit profiling call lazily creates one six-entry Vulkan timestamp query pool.
It resets and writes the queries inside the existing command buffer, uses the existing
single submit and fence, and reads results only after that fence. Ordinary inference
records no query commands. The six boundaries separately measure gate/up, the
gate-to-down dependency, down, the down-to-reduce dependency, and route reduction.

Support comes from the selected queue family's `timestampValidBits`; the counter is
decoded modulo that width and multiplied by the selected physical device's
`timestampPeriod`. This queue exposes 64 valid bits and 10 ns/tick. Queues without a
conformant counter refuse only the explicit profile call; normal inference remains
available. The query pool and every lazily created candidate pipeline have explicit
destruction paths.

The timestamp perturbation was not stable enough to qualify as performance evidence.
Command-adjusted wall overhead ranged from negative noise to +7.85% across short runs;
host query retrieval itself was only about 0.00021 ms. Timestamp results are therefore
used to choose the search direction, while all promotion decisions use uninstrumented
wall time. This is deliberately reported rather than hidden by relaxing the planned 2%
instrumentation gate.

## Tournament

Gate/up consumed about three quarters of profiled GPU time, down about one quarter, and
the route reducer less than 1%. The two dependencies together were about 1.2%. That made
dot-product reduction geometry the first candidate and put down/reduce fusion below the
10% promotion threshold even if its reducer and dependency cost vanished completely.

The retained 64-lane route reducer was paired with six gate/up and down configurations:

- fixed shared-memory trees at 32, 64, and 128 lanes;
- subgroup-add reductions at 32, 64, and 128 lanes.

The existing 64-lane tree shaders were not edited. Ten new stage modules share their
descriptor and push-constant ABI and are created lazily only when selected. WG128 is
refused if either device workgroup limit is below 128. Subgroup candidates additionally
require compute-stage `BASIC | ARITHMETIC` support. Every configuration passed the CPU
and guarded numerical gates and repeated bit-for-bit within its own pipeline. Cross-
variant equality is tolerance based because every workgroup size changes f32 reduction
order and `subgroupAdd` has implementation-defined intra-subgroup order.

Two warmups precede 12 samples per configuration. A six-sequence Williams ordering is
repeated twice so every candidate occupies every timing position equally and first-order
carryover is balanced. A separate identically ordered profiling pass cannot contaminate
the uninstrumented selection samples.

The first three binary-only qualification runs produced:

| Run | WG64 control ms | Tree WG128 ms | Control/tree | Subgroup WG128 ms | Max Tctl |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.510705 | 0.426366 | 1.198x | 0.426061 | 64.750 C |
| 2 | 0.537976 | 0.435303 | 1.236x | 0.415100 | 66.000 C |
| 3 | 0.474582 | 0.393840 | 1.205x | 0.394436 | 65.500 C |

Both WG128 variants cleared 10% in every run and exchanged the absolute lead within
run noise. The fixed tree was promoted because its reduction order is explicit rather
than driver-defined. For exactly H=2048/F=512/K=8 on RADV GFX1151, Mesa 25.2.8, with
at least 128 invocations in X, `routed_moe_bf16` and `routed_moe_bf16_fast` now select
Tree WG128. The qualified WG64 tree remains the automatic fallback for every other
shape, device, driver version, or tighter workgroup limit, and every tournament variant
remains explicitly callable for future qualification.

## Final production-path qualification

The final binary was executed three more times under the thermal guard after promotion:

| Run | Legacy ms | Guarded ms | Production fast ms | Legacy/safe | Safe/fast | Legacy/fast | Max Tctl |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.628932 | 0.646415 | 0.440122 | 4.07x | 1.47x | 5.97x | 67.250 C |
| 2 | 2.469547 | 0.619069 | 0.435183 | 3.99x | 1.42x | 5.67x | 65.375 C |
| 3 | 2.428916 | 0.648750 | 0.464147 | 3.74x | 1.40x | 5.23x | 64.625 C |
| Median of run medians | 2.469547 | 0.646415 | 0.440122 | 3.82x | 1.47x | 5.61x | — |

The matched tournament embedded in those same runs measured Tree WG128 against WG64 as
1.232x, 1.274x, and 1.261x faster. Median-of-run medians was 0.409534 ms versus
0.516385 ms, or 1.261x. The ordinary production path's 0.440122 ms median is 1.273x
faster than the prior qualified 0.560211 ms record: a 21.44% elapsed-time reduction and
about 106.5 GiB/s of encoded 48 MiB expert-weight traffic before activation/metadata
traffic. It evaluates this isolated routed operator about 2,272 times/s; that is not an
output-token rate.

After the final selector-policy review, the exact source manifest below was rebuilt and
run once more. Production Tree WG128 measured 0.441389 ms, its matched WG64 control
measured 0.519612 ms (1.167x), safe/fast was 1.48x, legacy/fast was 5.44x, adjusted
timestamp perturbation was +0.42%, and maximum Tctl was 68.500 C. This is a current-tree
consistency run and is not folded into the three-run qualification median above.

Tree WG128 produced CPU relative L2 `2.789e-7` and maximum absolute error `4.657e-9`.
Its guarded-relative L2 was `2.722e-7`; accumulate relative L2 was `2.470e-8` with
`1.490e-8` maximum absolute error. Overwrite, accumulation, automatic selection, odd-
shape guarded fallback, explicit refusal, two consecutive query-pool resets, and within-
pipeline bit repeatability all passed.

Median exploratory production timestamps were approximately:

| Interval | Time | Share of GPU interval |
| --- | ---: | ---: |
| Gate/up | 0.244500 ms | 74.2% |
| Gate-to-down dependency | 0.001920 ms | 0.6% |
| Down | 0.078620 ms | 23.9% |
| Down-to-reduce dependency | 0.001920 ms | 0.6% |
| Route reduction | 0.002860 ms | 0.9% |
| Total | 0.329460 ms | 100% |

The total was about 76% of median profiled wall time, below the predeclared 85% boundary.
The next experiment is therefore reusable routed command recording and a lower-frequency
host boundary, not down/reduce fusion or another arithmetic rewrite. A perfectly free
route reducer plus both dependencies is only about 0.0067 ms here.

## Environment, command, and thermal policy

- Host: AMD Ryzen AI MAX+ PRO 395 / Radeon 8060S.
- Vulkan device: RADV GFX1151, Mesa 25.2.8.
- Package Tctl warning: 75 C; hard abort: 90 C; polling: 0.25 seconds.
- The final three qualification runs stayed below 67.250 C; the post-review consistency
  run stayed below 68.500 C. None aborted.

```sh
cargo build -p beholdd-backend --release --features vulkan \
  --example bench_vulkan_moe --offline

python3 tools/thermal_guard.py --warn-c 75 --critical-c 90 \
  --poll-seconds 0.25 -- \
target/release/examples/bench_vulkan_moe
```

## Verification

The exact recorded tree passed:

- `cargo fmt --all -- --check` and `git diff --check`;
- `cargo test --workspace --all-targets --all-features`: 281 passed, 6 intentionally
  ignored environment-gated tests, 0 failed;
- workspace Clippy across all targets/features with warnings denied;
- workspace rustdoc with warnings denied, plus all doctests (1 passed);
- 45 Python tool tests;
- all four ignored live Vulkan backend tests when explicitly enabled under the same
  thermal guard (4 passed, maximum Tctl 60.000 C);
- all 18 release SPIR-V modules under `spirv-val --target-env vulkan1.2`; and
- an independent final code review with no remaining P0, P1, or P2 findings.

## Source manifest

```text
983e15c69675a3e04e5a89ad3e08f3a70174d0608c1f4400990d217fc246104b  crates/beholdd-backend/src/vulkan.rs
3e4eb17902daedfd224cb7dbd0b1220d2f86550fdbdf1970db7bb89105e142f0  crates/beholdd-backend/examples/bench_vulkan_moe.rs
e42696f4a8f403505f0f469a1a5a6b44390a70b581396c021d728299e036b8ab  crates/beholdd-backend/build.rs
670e6f7491a3d528733a2a66c0b8aaa3fe22112bdbc0f4bd4eca15514388d637  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_fast.comp
3d615ae3e7dfa2ade5ad818c5bc23fcd81ef05759666306ffc9fc5c364d14352  crates/beholdd-backend/shaders/bf16_routed_moe_down_fast.comp
a4feb83785a6998da6f3e00a7f565e66caa2a048aaa710a1f533da5deb30f4d5  crates/beholdd-backend/shaders/bf16_routed_moe_reduce_fast.comp
d8e34353888f83bf053a418090d23920642d9b154757a31529324639aa038b90  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_fast_wg32.comp
5e07a530116ff157af19d4cd4de3852a9ebd852b50ed3dfcbfa1beb0c59f14ef  crates/beholdd-backend/shaders/bf16_routed_moe_down_fast_wg32.comp
8525498a281c4eea52311a83af17d2bd55874c91cd7d3a98b09accdec7d84e17  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_fast_wg128.comp
1d823721bb2755710554c9682cfd762ffca5482bb41512e8ae2c94a2fb343acc  crates/beholdd-backend/shaders/bf16_routed_moe_down_fast_wg128.comp
47174ae969e7f4ee9532991bfcd366dce71a663acdc084549ad191d8ca30fe32  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_subgroup_wg32.comp
6042f5dd1643fb6b265c05fde875f1b6e0754b83bb20246ec88e9de1b581d11d  crates/beholdd-backend/shaders/bf16_routed_moe_down_subgroup_wg32.comp
0edd3bf6870862c296f448d36437730699d6b4d086a0c04f9c4654d7a8093951  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_subgroup_wg64.comp
85bc8895ba9541e1337b6de47834999e2c7de3523ceda5f94c829f0b688ed623  crates/beholdd-backend/shaders/bf16_routed_moe_down_subgroup_wg64.comp
534e1cdc5f4806c17945599aaf0a1f5fd49ab849e8b98660fdca774d55d582ed  crates/beholdd-backend/shaders/bf16_routed_moe_gate_up_subgroup_wg128.comp
4d05a48e5a0d76aea9d42671d253ec7da2d9d1a8942994043b74f2efe9cc3022  crates/beholdd-backend/shaders/bf16_routed_moe_down_subgroup_wg128.comp
f63408c241b41ce44a6efab84b5a094a4d0f5ee4b449e7b176ad0dff91b3a94b  Cargo.lock
```
