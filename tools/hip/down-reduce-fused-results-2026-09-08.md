# Down-WMMA + route-reduction fusion — runtime qualification

Date: 2026-09-08. Decision: **NO-GO; retain the frozen five-launch control.**

The isolated candidate fused the native-BF16 down projection and the
rank-ordered FP32 route reduction. It removed one launch and the global
`expert_output` handoff, but it did not meet any of the four predeclared
median-of-three promotion thresholds. It is not a backend option and was not
wired into Beholdd, reference deployment, or application service.

## Qualified artifact

An independent strict rebuild used the pinned ROCm 7.13 wheel and public
rocWMMA checkout, targeted exact `gfx1151`, and reproduced the frozen device
`.text` payload:

```text
baseline_source_sha256=8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80
candidate_source_sha256=f37105d5ee4ebc3ffac3022b306c16e8096fdb9bf72009bba80f47b738862417
build_wrapper_sha256=a6ae793e014770f9ade75d1a858968cb408b0ae9becdb1d518395d1235058836
device_text_sha256=f514c3a148892e07faa2fde2c6771706f3f0b3d07705c2e0f9dedc9d50e3452d
independent_build_code_object_sha256=66eb9b8ab6cf9c5396f34a3e3744f0f44d145f892701113f1e03cae0297a5440
```

The complete code-object container includes the disposable HIP overlay path,
so its whole-file hash is an artifact identifier rather than the reproducible
payload identity. The extracted device `.text` hash above is deterministic.

Static inspection found exactly six kernels: the five frozen controls and the
candidate. The fused kernel used 51 VGPRs, 8,192 bytes of LDS, wave32 and a
256-thread maximum workgroup, with no private scratch, spills, or dynamic
stack. Its ISA contained exactly two native BF16 WMMA instructions, one
workgroup barrier, four LDS-store instructions, and eight LDS-load
instructions.

## Correctness result

The guarded smoke and all three full runs validated every combination of:

- `N in {1,2,4,8,16}`;
- resident and pinned-copy host-visible envelopes;
- the frozen control and fused candidate; and
- permuted, reversed-rank, duplicate/missing-expert, and one-hot route tables.

Every observation was finite and repeat-deterministic. Every candidate output
matched its frozen control bit for bit, including reordered and duplicate
routes. Across the complete matrix, the largest observed mechanical-oracle
errors were relative L2 `3.495511e-4` and max absolute `1.527893e-6`, within
the `5e-4` and `1e-3` gates. The largest current-F32-policy drift was relative
L2 `2.838885e-3` and max absolute `1.788326e-5`, within the separately declared
`5e-3` and `1e-4` bounds.

## Paired performance result

Each full process used four warmup pairs, twelve counterbalanced paired
wall-clock samples, and bursts of sixteen complete operators. The table reports
the paired median `control / candidate` speedup:

| run | `N=1` resident | `N=1` host-visible | `N=16` resident | `N=16` host-visible |
|---:|---:|---:|---:|---:|
| 1 | 1.028339x | 1.025724x | 1.024441x | 1.027226x |
| 2 | 1.041357x | 1.039298x | 1.022543x | 1.017043x |
| 3 | 1.031432x | 1.014909x | 1.025584x | 1.034550x |
| median of three | **1.031432x** | **1.025724x** | **1.024441x** | **1.027226x** |
| required | **1.10x** | **1.10x** | **1.08x** | **1.08x** |

All twelve candidate/control p90 comparisons passed the `candidate <= 1.05 *
control` tail bound; the worst ratio was `1.007959`. Only four of twelve cells
met the separate per-run `1.03x` paired-median floor. Run 1 passed zero cells,
run 2 passed both `N=1` cells, and run 3 passed `N=1` resident plus `N=16`
host-visible. Every full process therefore returned the intentional
performance-gate failure status.

The median improvement is real but only about 2.4–3.1%, far below the 8–10%
minimum chosen before measurement. It does not justify a second execution path.

## Thermal containment

Every GPU invocation ran through `tools/thermal_guard.py` with a 75 C warning,
90 C hard abort, and 0.25-second polling. The three full runs reported peak
Tctl of `58.25 C`, `58.25 C`, and `58.75 C`. No thermal warning or abort
occurred.

## Interpretation and next step

At `N=1`, fusion avoids a fixed 1 MiB global expert-output write and 64 KiB of
reducer reads. At `N=16`, it avoids the same write and 1 MiB of reducer reads.
Those savings remain small beside 48 MiB of persistent BF16 weights per routed
operator. The candidate also changes 1,024 one-wave down workgroups into 128
eight-wave workgroups with an 8 KiB LDS rendezvous, so launch and intermediate
traffic removal does not translate into a material wall-time win.

This result joins gate/up/SiLU fusion and HIP Graph replay as a documented
negative micro-optimization. The next acceleration slice should preserve the
frozen five-kernel arithmetic while removing host boundaries across material
model work: a real-weight, device-resident routed layer containing RMSNorm,
device-side router/stable top-k, shared and routed experts, residual update, and
resident hidden/state buffers. After that baseline exists, route-aware cohorts
across concurrent requests and trained MTP verification can be measured in true
output tokens per second.
