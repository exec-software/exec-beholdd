# Qwen3.6 route-aware B=16 cohort results — 2026-09-08

## Decision

**Research: PROMOTE. Production: NO-GO.**

The real-weight layer-3 experiment shows that grouping 16 independent requests
by routed expert can nearly double MoE-FFN sublayer throughput on the Radeon
8060S (`gfx1151`). Across three thermally guarded deep runs, the route-aware
cohort took a 3.316558 ms median of run medians versus 6.286991 ms for 16
sequential executions of the qualified prepared-weight path. The paired
speedup median of run medians was 1.892540x.

This promotes the mechanism to the next research phase. It does not promote
the implementation into Beholdd's production runtime. The measurement covers
one real-weight MoE-FFN sublayer, a fixed synthetic corpus with unusually
correlated routes, and 16 concurrent independent requests. It is neither
single-request latency nor autoregressive output-token throughput.

The later hashed-independent robustness follow-up establishes **1.360078x**,
not 1.892540x, as the working synthetic baseline across three independent
seeds. The high-overlap result remains valid for its frozen corpus but is not a
universal speedup. The updated decision remains research PROMOTE and production
NO-GO, now with real activation-trace replay as the next evidence gate.

## Deep-run results

Each run used exactly `--warmups 4 --samples 12 --burst 8`. The control and
candidate were measured in counterbalanced A/B then B/A order.

| Run | 16 sequential controls | Route-aware B=16 cohort | Paired speedup | Candidate independent-request MoE-FFN sublayers/s | Max Tctl |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.286570 ms | 3.316558 ms | 1.892540x | 4,824.279392 | 60.625 C |
| 2 | 6.286991 ms | 3.368297 ms | 1.886039x | 4,750.174587 | 61.500 C |
| 3 | 6.308530 ms | 3.299103 ms | 1.905721x | 4,849.802562 | 61.500 C |
| median of run medians | **6.286991 ms** | **3.316558 ms** | **1.892540x** | **4,824.279392** | **61.500 C** |

The candidate median amortizes to 0.207285 ms per independent request. That is
an arithmetic throughput figure (`3.316558 / 16`), not the latency experienced
by an individual request. Similarly, 4,824.279392 is a rate of independent
request *sublayers*, not generated tokens/s and not a full-model result.

## Exact experiment contract

The benchmark consumes the strict native-BF16 Qwen3.6 layer-3 pack:

- Hidden width `H=2048`.
- Routed-expert FFN width `F=512`.
- 256 routed experts, top 8 per request.
- One shared expert with `F=512`.
- Cohort size `B=16`, equal to the rocWMMA tile row count.
- Expert gate/up/down weights and the post-SiLU activation boundary are BF16.
- RMSNorm, routing, WMMA accumulation, expert mixing, and the residual are
  F32.

The 16 hidden vectors are deterministic and were frozen before GPU execution:

```text
h[t,i] = 0.35f * sin(i * 0.013f + t * 0.731f)
       + 0.20f * cos(i * 0.037f + t * 1.117f)
       + (((i * 3 + t * 7) % 11) - 5) * 0.007f
```

The control owns 16 independent `ResidentState` objects. For each measured
cohort it enqueues the exact qualified `enqueue_prepared_parallel` chain 16
times on one stream, then synchronizes only at the outer burst boundary.

The candidate performs the following work entirely on the same GPU stream:

1. Run independent RMSNorm, router/shared-gate, stable softmax, and top-8 for
   all 16 inputs.
2. Deterministically scan the 128 routed `(request, rank)` pairs, list active
   experts in ascending expert-ID order, and assign one 16-row tile to each
   active expert.
3. Gather each request's rounded BF16 normalized input into the appropriate
   expert-tile row. Padding rows are zero. A fixed final tile holds all 16 rows
   for the shared expert.
4. Execute prepared row-major BF16 rocWMMA gate/up, BF16 SiLU/up, and down for
   every active expert tile and the shared tile.
5. Scatter by the validated pair-to-tile/row map, accumulate routed outputs in
   original rank order, add the shared expert, and add the residual.

Route planning, gathering, all matrix work, activation, scatter/reduction, and
residual addition are inside candidate timing. One-time weight preparation,
CPU references, correctness copies, and stage diagnostics are outside timing.
There is no host synchronization inside a candidate cohort. `timed_burst`
performs one initial synchronization to establish an empty-stream boundary and
one final synchronization after all eight cohorts in the burst.

All GPU runs used the digest-bound runner:

```bash
tools/hip/run_qwen35_layer_guarded.sh \
  /tmp/beholdd-q35-route-cohort-bench \
  /tmp/qwen36-layer3-bf16.pack \
  9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
  --warmups 4 --samples 12 --burst 8
```

The runner checked the pack's digest and file identity before and after each
run, cleared `LD_LIBRARY_PATH`, `LD_PRELOAD`, and `LD_AUDIT`, warned at 75 C,
terminated at 90 C, and polled at 0.25-second intervals. All reported runs
stayed below 62 C Tctl. Reference serving were not modified.

## Route occupancy and its limitation

The frozen corpus selected 67 unique routed experts among 128 routed pairs.
Occupancy was:

| Requests using an expert | Expert count |
|---:|---:|
| 1 | 28 |
| 2 | 22 |
| 3 | 13 |
| 4 | 3 |
| 5 | 1 |

Thus, routed-row utilization was `128 / (67 * 16) = 11.9403%`. Sequential
execution presents 144 logical expert/shared tiles (`16 * (8 + 1)`), while the
cohort issues useful matrix work for 68 expert/shared tiles (`67 + 1`), a
theoretical tile-count reduction of 2.117647x before planning, gathering, and
fixed-grid overhead. The measured paired speedup was lower, as expected, at
1.892540x.

This corpus is correlated. Read-only previews of ten deterministic independent
Gaussian corpora produced 89–100 active experts rather than 67. Those previews
are not GPU benchmark results, but they show that the present cohort has more
route reuse than a plausibly independent workload. The 1.892540x result must
therefore be treated as the frozen correlated-corpus result, not a general
parallel-agent speedup. A hashed-independent corpus and real captured
activations are required before the mechanism receives a broader performance
claim.

## Correctness and the failed bit-exact contract

The original quick smoke used `--quick` and required the complete candidate
output to be bit-identical to all 16 qualified controls. It exited with status
2 because full output bit identity failed. A second quick run with untimed
stage instrumentation and the same exact-output gate also exited with status
2. The failure was preserved and diagnosed; it was not silently relabeled as
an exact result.

All structural and routing invariants passed:

- All 16 route-ID lists matched their CPU references.
- Candidate and qualified-control route IDs, route weights, shared scales,
  margins, and status values were bit-identical.
- Route weights were finite, nonnegative, ordered, and normalized.
- The planner reported 67 ascending, unique, in-range experts.
- Every one of the 128 routed pairs had a unique, in-range tile/row mapping;
  tile token ownership and padding were exact.
- Routed and fixed shared input rows matched the rounded BF16 inputs exactly.
- Every selected routed/shared prepared gate/up/down matrix matched the source
  BF16 bytes, including the reported expected/observed digest.
- Control and candidate outputs, route metadata, and planner metadata were bit
  deterministic across separately poisoned executions.

The maximum error across the 16 CPU references remained comfortably within
the pre-existing CPU-oracle gate:

| Path | Maximum per-request relative L2 | Maximum absolute error |
|---|---:|---:|
| Qualified control vs CPU | 2.086448e-5 | 2.160389e-5 |
| Route-aware candidate vs CPU | 2.086449e-5 | 2.160482e-5 |
| Candidate vs qualified control | 2.047526e-6 | 2.324581e-6 |

The CPU gate remains `relative L2 <= 5e-4` and `max absolute <= 1e-3`.

## Stage diagnosis

The host-only diagnostic compares only logical rows. It bins all 144 logical
expert/shared occurrences by row zero/nonzero, routed/shared, and the four
crossed categories. Adding it did not change device text.

All 294,912 compared rounded BF16 input elements were bit-identical. More
importantly, all four stages were bit-identical for the 68 row-zero
occurrences. Those 68 occurrences are the first row for each of 67 routed
experts plus request zero's shared-expert row.

| Candidate row group | Stage | Bit mismatches / elements | Relative L2 | Max absolute |
|---|---|---:|---:|---:|
| Row zero, 68 occurrences | Rounded BF16 input | 0 / 139,264 | 0 | 0 |
| Row zero, 68 occurrences | Gate/up F32 | 0 / 69,632 | 0 | 0 |
| Row zero, 68 occurrences | Activation BF16 | 0 / 34,816 | 0 | 0 |
| Row zero, 68 occurrences | Down F32 | 0 / 139,264 | 0 | 0 |
| Nonzero rows, 76 occurrences | Rounded BF16 input | 0 / 155,648 | 0 | 0 |
| Nonzero rows, 76 occurrences | Gate/up F32 | 44,233 / 77,824 | 2.322e-7 | 1.550e-6 |
| Nonzero rows, 76 occurrences | Activation BF16 | 9 / 38,912 | 8.435e-6 | 2.441e-4 |
| Nonzero rows, 76 occurrences | Down F32 | 86,577 / 155,648 | 7.957e-6 | 9.436e-6 |

This isolates the first numerical difference to executing an otherwise exact
input/weight matrix on nonzero rocWMMA rows. It rules out routing, prepared
weights, planning, gathering, and row-zero matrix arithmetic as the source.
The values remain finite, and the final candidate/control discrepancy is much
smaller than the intermediate BF16 activation's largest absolute difference.

Host reconstruction used each path's captured down outputs, its own exact
route weights/shared scale, original rank order, and F32 `fma` operations:

- Candidate host reconstruction versus candidate device output: 0 / 32,768
  bit mismatches.
- Control host reconstruction versus control device output: 0 / 32,768 bit
  mismatches.
- Candidate host reconstruction versus control host reconstruction: 7,287 /
  32,768 bit mismatches, relative L2 `5.241e-7`, max absolute `2.325e-6`.

The first two observations confirm the scatter map and reduction order for
this toolchain/run. They remain supporting diagnostics rather than a portable
host/device exactness promise because host and GPU FP modes can differ.

## Prospective numerical-equivalence gate

After the exact smoke failed and the stage diagnostic isolated a row-position
effect, the final frozen harness added a prospective numerical-equivalence
gate. It does not erase the exact result: the full-output bit-exact contract is
still computed and printed separately, and it remains false for this corpus.

The prospective gate requires:

- Every rounded BF16 input to be bit-identical.
- All compared gate/up, activation, and down values to be finite.
- Row-zero rounded input, gate/up, activation, and down values to be
  bit-identical.
- Candidate/control maximum per-request relative L2 at or below `5e-6`.
- Candidate/control maximum absolute error at or below `2e-5`.
- All earlier CPU, route, prepared-byte, gathered-row, planner, and
  determinism gates to pass unchanged.

All three deep runs passed this prospective gate. The observed
candidate/control limits, `2.047526e-6` relative L2 and `2.324581e-6` maximum
absolute, retain 2.44x and 8.60x headroom respectively. This is evidence for a
research numerical contract, not proof of acceptable full-model behavior. The
thresholds were established after observing this synthetic corpus and must be
tested prospectively on unseen independent and real activation cohorts.

## Frozen correlated-corpus provenance and static qualification

- Cohort source used for the correlated-corpus deep runs:
  `tools/hip/rocwmma_qwen35_resident_layer_route_cohort_bench.hip.cpp`.
- Frozen correlated-corpus source SHA-256:
  `363559f727eb27261fd6595a6ba1943826bdab6114e43b246ec153abceaed9c0`.
- Correlated-corpus build/static-qualification wrapper:
  `tools/hip/build_rocwmma_qwen35_resident_layer_route_cohort.sh`.
- Frozen correlated-corpus build-wrapper SHA-256:
  `2c30ce4d2f38b051357d4060c4ff5f9ba3bbdd1cdcf1479e9adad16ffe957dc1`.
- Qualified correlated-corpus benchmark executable SHA-256:
  `8a61267f4ddb2e7bec48264da3a1bc9b4126b4b2d39c2c84b05ccc93ea52aefe`.
- Exact `gfx1151` device-text SHA-256:
  `1d55ff0ff68f2326db7b3109afcc480fda7c77b017b2c2f56cae3e47b52972fa`.
- Real-weight layer pack: `/tmp/qwen36-layer3-bf16.pack`, 1,619,021,824
  bytes.
- Pack SHA-256:
  `9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116`.
- Inherited qualified combined source SHA-256:
  `4a9b834332c76accd0d86cd0acace19d448ba1aecdaf8ab2b90310cdad958c60`.
- Inherited qualified build-wrapper SHA-256:
  `bed5483771113aceffe04fe8aa4a4b9434e08b36a4e732d1685facb42d5449fb`.
- Generated main-renamed combined view SHA-256:
  `3bc820e5574842299c21cdf0fc68349ef32fdfefd46e9ba45ef9bbc30fce61d7`.

The wrapper requalified the inherited prepared/parallel path before compiling
the cohort. The final translation unit contains exactly 21 `q35` kernels. All
are wave32 with zero private memory, zero SGPR/VGPR spills, no dynamic stack,
and no scratch-memory ISA. It contains exactly 16 native
`v_wmma_f32_16x16x16_bf16` instructions; each cohort WMMA kernel contains two.
The cohort router uses 30 VGPRs, planner 21, gate/up 45, and down 48. The
parallel route and planner contain 51 and 5 synchronization barriers,
respectively.

The host-only diagnostic and prospective contract changed host-side source but
left the device-text digest unchanged, providing a direct check that the
measured GPU kernels did not change between diagnosis and the correlated-
corpus deep runs. Later follow-up work may move the working-tree source and
wrapper hashes; the two frozen hashes above identify the artifacts that
produced the measurements in this document.

## Reproduce

Build and statically qualify without running the GPU:

```bash
cd "$BEHOLDD_SOURCE_DIR"
sha256sum \
  tools/hip/rocwmma_qwen35_resident_layer_route_cohort_bench.hip.cpp \
  tools/hip/build_rocwmma_qwen35_resident_layer_route_cohort.sh
ROCWMMA_INCLUDE=/tmp/rocwmma-source-audit/projects/rocwmma/library/include \
  bash tools/hip/build_rocwmma_qwen35_resident_layer_route_cohort.sh \
  /tmp/beholdd-q35-route-cohort-bench
```

For an exact reproduction of the correlated-corpus result, the two pre-build
digests must match the frozen source and wrapper digests above, and the built
executable must match `8a61267f...`. A later working-tree follow-up is not the
same host artifact even if its GPU device-text digest remains identical. Then
use the digest-bound thermal runner shown in the experiment contract. Direct
execution does not provide the pack-identity, loader-environment, or thermal
containment used for the reported measurements.

## Correlated-corpus promotion boundary at that checkpoint

At the correlated-corpus checkpoint, research promotion meant retaining this
implementation as the baseline for:

1. A frozen, hashed-independent B=16 corpus measured with the same control,
   gate, runner, and three-run protocol.
2. Captured real hidden states from multiple prompts, layers, and decode
   positions, including low- and high-route-overlap cohorts.
3. A scheduler experiment that bounds queueing delay while forming cohorts;
   throughput must be reported alongside per-request latency percentiles.
4. Whole-model prepared-weight residency, attention/KV integration, and actual
   generation quality/eval comparisons.

Production remains NO-GO until those gates pass. In particular, no current
result establishes 4,824 generated tokens/s, a 1.89x single-request speedup,
full-model numerical equivalence, unchanged logits, or unchanged generated
sequences. The reference serving runtime remains the production baseline.

---

## Independent hashed-corpus robustness follow-up

### Updated outcome

The hashed-independent corpus confirms that route-aware B=16 execution still
provides a material throughput gain when expert overlap is substantially lower.
Its median paired speedup is **1.360078x**, versus 1.892540x for the correlated
phase-shifted corpus. The former is the working robust-synthetic baseline; the
latter is a valid high-overlap case.

**Research: PROMOTE to captured real-trace replay. Production: NO-GO.**

This follow-up directly tests the immediate synthetic-correlation concern, but
it does not establish a production distribution. Uniform hash-generated hidden
vectors are designed to decorrelate the test inputs; they are not statistically
independent samples from Qwen's real hidden-state distribution.

### Corpus and measurement contract

The follow-up adds a host-only corpus selector. With
`--hashed-independent-corpus`, each `(request, hidden-index, seed)` tuple is
mixed separately by a fixed 32-bit integer hash. The upper 24 hash bits are
mapped deterministically to an F32 value in `[-1, 1)`. Seeds 1, 2, and 3 define
three frozen B=16 corpora. No GPU kernel, pack byte, weight layout, route
planner, timing boundary, control, or correctness threshold changed.

Every seed used:

- `--warmups 4 --samples 12 --burst 8`.
- The same digest-bound thermal runner and layer pack as the correlated run.
- The same 16-sequential-request qualified control.
- The same route-aware candidate, one stream, A/B then B/A counterbalancing,
  and one outer synchronization per measured burst.
- The prospective numerical-equivalence gate documented above.

### Three-seed results

| Corpus seed | Active routed experts | Theoretical tile reduction | 16-control median | Candidate median | Paired speedup | Candidate independent-request MoE-FFN sublayers/s | Max Tctl |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98 | 1.454545x | 6.314967 ms | 4.592443 ms | 1.378822x | 3,483.984260 | 60.375 C |
| 2 | 102 | 1.398058x | 6.458366 ms | 4.865521 ms | 1.345880x | 3,288.445116 | 61.375 C |
| 3 | 99 | 1.440000x | 6.321079 ms | 4.662903 ms | 1.360078x | 3,431.338803 | 61.500 C |
| median across seeds | **99** | **1.440000x** | **6.321079 ms** | **4.662903 ms** | **1.360078x** | **3,431.338803** | **61.500 C** |

Paired speedup is the median of within-sample control/candidate ratios, not the
quotient of the two displayed marginal medians. The rate is calculated from
the harness's full-precision timing value; the table displays time to six
decimal places. Small quotient discrepancies are therefore expected.

The median candidate time amortizes to **0.291431 ms per independent
request** (`4.662903 / 16`). This is an aggregate-throughput calculation, not
single-request latency.

If the same median rate were sustained through 40 otherwise cost-free MoE-FFN
layers, its arithmetic aggregate ceiling would be
`3,431.338803 / 40 = 85.783470` independent requests/s. That is not a measured
full-model rate, not an autoregressive token rate, and not a claim of 85.78
generated tokens/s. Attention, state-space/recurrent work, embeddings, the LM
head, sampling, KV traffic, scheduling, queue formation, and inter-layer
effects are all excluded.

### Occupancy interpretation

The three independent seeds activate 98, 102, and 99 routed experts, compared
with 67 in the correlated corpus. Their theoretical matrix-tile reductions are
computed as `144 / (active_experts + 1)`, where 144 is the 16-request
sequential total and the additional cohort tile is the shared expert.

| Corpus | Active routed experts | Theoretical tile reduction | Measured paired speedup |
|---|---:|---:|---:|
| Correlated phase-shifted | 67 | 2.117647x | 1.892540x |
| Hashed independent, median | 99 | 1.440000x | 1.360078x |

The corresponding routed-row utilizations for independent seeds 1–3 are
8.1633%, 7.8431%, and 8.0808%. Lower row utilization here does not mean less
work per request; it reflects less expert sharing, hence more partially filled
expert tiles. The measured performance tracks the available tile reuse without
equalling the ideal tile-count ratio because routing, planning, gathering,
fixed-grid launches, activation, and reduction remain in timing.

The high-overlap 1.892540x result is therefore not universal. For synthetic
planning and capacity estimates, **1.360078x is the defensible baseline from
this experiment**, while 1.892540x characterizes a favorable overlap case.

### Correctness status

All three independent-seed runs passed the prospective numerical-equivalence
contract. That includes the unchanged CPU-oracle bounds, exact candidate/control
route metadata, prepared-byte and gathered-row checks, planner bijection,
finite/order/normalization checks, row-zero bit parity, and repeated-run
determinism. The candidate/control relative-L2 and maximum-absolute bounds
remain `5e-6` and `2e-5` respectively.

Passing that contract does not assert full-output bit identity. The harness
continues to calculate and print the stricter bit-exact result separately. It
also does not establish full-model logit, generated-sequence, or quality parity.

### Independent-corpus artifact lineage

This follow-up is a separately frozen host artifact. It preserves the
correlated-corpus source, wrapper, and executable hashes recorded earlier.

- Hashed-independent-capable source SHA-256:
  `7169e50ef46b4995a7fd58e7312c1db1d27c08bdda624cf0f6e0114544b988be`.
- Build/static-qualification wrapper SHA-256:
  `b338e07241f502a2890c2a09497959a84cddbfe181cfb207952c77f6869ea360`.
- Qualified executable SHA-256:
  `9de1b8f544f2d48f76441f8d11943b0eb044440972d9fc1269fb87e5a5937357`.
- Exact `gfx1151` code-object SHA-256:
  `6a7d562b94e78f5cf87951c2c50092d20f3d05b6c0a95a4cccb52169dccd6d82`.
- Device-text SHA-256:
  `1d55ff0ff68f2326db7b3109afcc480fda7c77b017b2c2f56cae3e47b52972fa`.
- Layer-pack SHA-256:
  `9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116`.

The device-text digest is identical to the correlated-corpus artifact. The
host artifact and code-object container have distinct rebuild hashes, while
the identical device text proves the measured GPU kernel instruction bytes did
not change.

### Reproduce the independent seeds

First verify that the source, wrapper, executable, and pack match the hashes
above. Then run each seed under the same containment:

```bash
cd "$BEHOLDD_SOURCE_DIR"
for corpus_seed in 1 2 3; do
  tools/hip/run_qwen35_layer_guarded.sh \
    /tmp/beholdd-q35-route-cohort-host-corpus-static \
    /tmp/qwen36-layer3-bf16.pack \
    9157ce6955d5c0b386622e982f7ae2bfc0856723f05698f1ca04b8833a418116 \
    --hashed-independent-corpus \
    --corpus-seed "$corpus_seed" \
    --warmups 4 --samples 12 --burst 8
done
```

The loop runs three separate guarded processes. The median row above is the
median of the three reported per-process medians for each metric; it is not a
pooled-sample median.

### Remaining limitations and next gate

The independent follow-up is still limited to three deterministic uniform
synthetic corpora, one real-weight layer, fixed B=16 admission with all requests
ready, and one APU. It does not model prompt/decode activation structure,
expert popularity across production traffic, per-layer route distributions,
arrival times, batching delay, cancellations, fairness, memory pressure from a
whole model and KV cache, or concurrent attention. It also does not measure
tail latency or end-to-end generation.

The next promoted experiment is replay of frozen real activation/route traces
across multiple prompts, decode positions, and layers. It must report route
overlap, cohort fill delay, throughput, and per-request latency percentiles,
then validate full hidden states and logits against an independent model
oracle. Until that work passes, production remains NO-GO and the golden
reference serving runtime remains untouched.
