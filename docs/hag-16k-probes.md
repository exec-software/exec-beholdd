# Bounded HAG C16384 probes

These additive diagnostics separate late-position attention coverage from full
model allocation. Hardware-free validation passed: five attention-probe tests,
45 model-probe tests and 15 supervisor tests. The frozen release build, separate
CPU reference and both bounded GPU gates passed. Counts below were checked by
the native runs; the qualification scope remains limited to these diagnostics.

The [16k development configuration](../config.hag-16k.example.yaml) selects
`context_capacity: 16384`; the serving default remains 2048. See
[capacity and catalog sizing](hag-editor-context.md#configurable-development-capacity--2026-10-15)
and [native runtime admission](hag-backend.md#explicit-experimental-hag-serving).
Existing C2048 qualifications retain their original inputs and scope.

## What each probe covers

`hag_attention_probe --diagnose-attention16384` uses synthetic full-capacity K/V
buffers with visible prefixes **8192, 16383 and 16384**. Each prefix evaluates one
query with two query heads, one KV head and head dimension 128, then repeats with
changed inputs. Six CPU references finish before native device opening. Future
K/V positions contain NaNs; input/output guards, finite per-value CPU agreement,
changed replay and C16385 recording refusal are checked. Work grows linearly with
capacity; it does not evaluate a square attention matrix or load a checkpoint.
The numerical bound is `2e-5 + 2e-5 * abs(CPU)`.

`hag_model_probe --capacity16384-reference` exports full-vocabulary CPU logits for
fixed token IDs **1, 2** using a **two-position CPU cache**, then exits. Its separate
`--capacity16384-smoke` process prepares the complete dense Qwen3-0.6B Q8_0 model
with **16384 native positions**, evaluates positions **0 and 1**, resets, refuses
stale logits, and replays token 1. Each of the three native evaluations compares
all 151,936 logits against the completed references using
`2e-3 + 2e-4 * abs(CPU)` and exact greedy-token equality. Sign/zero differences are
reported. Native preparation must perform zero dispatches and zero readbacks;
final reset and resource retirement must leave zero HAG budget charged.

The two-position CPU cache changes reservation and its own refusal boundary,
not the first two positions' RoPE or visible attention values. The model probe
does **not** exercise its native capacity boundary or late-position model logits.
Neither probe establishes long-prompt answers, editor/tool execution, serving
lifecycle, throughput or complete 16k numerical qualification.

| Required result | Attention diagnostic | Model native smoke |
| --- | ---: | ---: |
| Native allocation, bytes | 25,468,928 | 4,629,168,128 |
| Native allocation budget | 32 MiB | 4608 MiB |
| Physical dispatches | 36 | 11,379 |
| Logical submissions | 6 | 174 |
| Compared output values | 1,536, plus 48 guard checks | 455,808 |
| Verification readback, bytes | 6,336 | 1,823,232 |
| Retained HAG budget after drop | 0 | 0 |

The model allocates **3,995,575,808 tensor bytes**, reserves **315,748 auxiliary
bytes**, and expects only **123,232 auxiliary peak bytes** at these early
positions. Preparation writes are **4,864,071,508 bytes**; the three steps add
**345,084 recording bytes**. Packed-source staging is capped at **4,456,448
bytes**. The full-capacity coefficient pair needs **8,388,608 bytes**, increasing
the preparation staging forecast to **12,582,912 bytes** during upload conversion.
The probe forecast also includes reference buffers and owner allowances, so it is
separate from the serving configuration's 5,417,203,904-byte combined forecast.

## Supervised invocation contract

These are child argument forms for a frozen external supervisor, **not standalone
launch instructions**. Require Linux, the supported native device, `hag-native`
builds, a clean pinned source/build relationship, and the pinned checkpoint for
the model modes. Disable payload tracing with `HAG_TRACE_SUBMISSION=0`.

```text
hag_attention_probe --diagnose-attention16384 /dev/dri/renderD128
hag_model_probe --capacity16384-reference --model MODEL --reference-out REF --input-identity ID
hag_model_probe --capacity16384-smoke --model MODEL --reference REF --input-identity ID /dev/dri/renderD128
```

The attention mode requires `HAG_ATTENTION16384_GATE=1`. Both model modes require
`HAG_CAPACITY16384_GATE=1` and `HAG_CAPACITY16384_UNIT` naming their actual owned
service, `hag-capacity16384-cpu-DIGITS` or `hag-capacity16384-native-DIGITS`.
Setting these variables alone does not provide supervision. The model verifies
effective cgroup controls, `NoNewPrivs=1`, and live host/ancestor headroom; external
supervision supplies thermal, deadline, crash-capture and retirement evidence.

| Phase | Process memory maximum | Memory high / early stop |
| --- | ---: | ---: |
| CPU model reference | 2 GiB | 1536 MiB |
| Attention diagnostic, including its CPU references | 2 GiB | 1536 MiB |
| Native model smoke | 6 GiB | 5632 MiB |

Run phases serially, proving CPU-reference process/cgroup retirement before the
native model starts. Each phase requires zero swap, 64 tasks, 25% CPU quota
(`cpu.max=25000 100000`), affinity to CPUs 0 and 1, and bounded thread pools.
Require at least **12 GiB available host memory** at startup and retain **4 GiB**
while running. Tctl must stay at or below **60°C for ten seconds** before launch;
the runtime cutoff is **90°C**. Planned child/service/outer deadlines are
**360/375/390 seconds**. Stop on memory pressure, OOM/swap evidence, thermal or
deadline failure; preserve the failed attempt without automatic retry.

Native phases additionally require a live read-only privileged GPU crash watcher
with a fresh heartbeat and sufficient lease, plus clean driver/kernel closeout.
The task-local supervisor under `work/hag-native16k-v3-20261018` freezes these
controls and retains per-phase evidence. The ordinary development
build/test wrapper is not this native probe supervisor.

## Reference and completion evidence

The model's canonical 128-byte identity file concatenates the checkpoint,
source-manifest, model-binary and contract SHA-256 values. The supervisor
authenticates those hashes and the completed reference artifact; the probe checks
the exact attested bytes. The **1,215,776-byte** reference uses distinct
`HAGREF16K2F32` framing, records native capacity 16384 and CPU capacity/evaluations
2 separately, and contains exactly two finite vocabulary vectors. Publication is
exclusive and occurs only after successful CPU checks; incomplete output is not
accepted by native execution.

Model success emits one `HAG_CAPACITY16384_REPORT` JSON record and
`PASS hag-capacity16384 stage=cpu|native bounded_probe=true`. Attention success
requires all six position/replay PASS lines and its final JSON counters. A child
PASS is only one part of acceptance: frozen input identities, clean resource and
crash evidence, and process/cgroup/watcher retirement must also pass. Reports keep
full-model late-context and long-prompt qualification false.

## Observed preparation and CPU reference — 2026-10-18

Source commit `e378829` passed the 50 example tests above and a guarded release
build. The supervisor passed 15 hardware-free tests, including explicit cleanup
retry after an interrupted retirement. Its source inventory covers the 23
embedded `.comp` shaders as well as Rust, assembly and Cargo inputs. An earlier
preparation attempt omitted those shader dependencies and refused to freeze;
that receipt was preserved before preparing the fresh `v2` directory.

The separate CPU process exported both full-vocabulary vectors, with greedy
tokens **14582 and 1096**, and refused a third CPU position without source reads.
The reference is **1,215,776 bytes**, SHA-256
`893487ea74ba910cb24bfa667dff770474545d2d0d9aa89b98bfbca601a27582`.
Its observed cgroup peak was **57,663,488 bytes**, with **42.25°C** peak Tctl,
zero memory-pressure/OOM events and zero task swap. This warm-checkpoint
observation does not replace the conservative admission forecast. The CPU
process and cgroup retired, and its kernel interval contained no new entries.

Linux authentication for the read-only crash watcher then exceeded its
120-second deadline. The watcher never armed, neither GPU stage launched, and
all recorded CPU, authentication and supervisor processes were confirmed absent.
The quiet window was released. Evidence is retained in the task workspace under
`work/hag-native16k-v2-20261018`, including `development-validation.json`,
`runs/cpu-1792336002362338193/result.json`, and
`authentication-closeout.json`. The frozen manifest SHA-256 is
`5831e996f3e9a07a8c884cc54e0df0500a7df8b30ae18eedb407d05d411e813c`.

That attempt was closed without GPU qualification; the fresh attempt below
completed the native gates without reusing the closed watcher session.

## Observed bounded native qualification — 2026-10-18

A fresh clean build at `50c04994` and a new two-position CPU reference preceded
the GPU gates. All six attention cases passed at prefixes **8192, 16383 and
16384**, including changed inputs, 48 guards and three C16385 recording refusals.
The maximum absolute attention error was **5.960464477539063e-8** across 1,536
values, within the unchanged per-value tolerance. Allocation and all transfer,
dispatch and submission counts matched the table above; HAG budget returned to
zero after retirement.

The model prepared all **4,629,168,128 native allocation bytes** at C16384 and
compared **455,808 logits** across positions 0, 1 and reset/replay. All three
greedy tokens matched (**14582, 1096, 14582**), with no sign/zero mismatches and
maximum absolute error **3.62396240234375e-5** under the unchanged tolerance.
Stale logits were refused after reset; unload left zero weight, tensor,
auxiliary and budget counters. Preparation issued no dispatches/readbacks.

| Observed resource | CPU reference | Attention | Native model |
| --- | ---: | ---: | ---: |
| Peak process-cgroup bytes | 57,139,200 | 141,193,216 | 695,468,032 |
| Maximum observed Tctl | 48.625°C | 48.625°C | 55°C |
| Minimum running host-available bytes | 20,270,424,064 | 20,204,654,592 | 14,478,114,816 |
| Memory-pressure/OOM events and task swap | 0 | 0 | 0 |

Process-cgroup figures are separate from the HAG native allocation and must not
be read as total host-plus-GPU use. The native run retained at least **13.48 GiB**
of available host memory. Every recorded process/cgroup and the privileged
read-only watcher retired. The watcher exited successfully with no captured
dumps. Kernel intervals contained unrelated split-lock/AppArmor messages and no
matching GPU/OOM/fault/reset events. Earlier display errors remain recorded in
the boot baseline; this is not a whole-boot clean-kernel claim.

Evidence is under `work/hag-native16k-v3-20261018`, including `validation.json`
and `completion.json`. Frozen manifest SHA-256:
`2822550ae4fb3b5aba5e0047e19f9926c53672d03933e755e9f754b97a0ee351`.
Completion SHA-256:
`3b9007e2eeff490b2c5ae1a02417c90ee371315b9af139e7eb7620519cbafb5f`.

This establishes late-position synthetic attention and full model allocation
with early-position numerical/reset checks. It does not establish late-position
model logits, full-catalog native prefill, long-prompt answers or complete 16k
numerical qualification. The next editor integration slice captures an actual
39-tool composer request and measures its exact transmitted bytes before any
longer native inference run.
