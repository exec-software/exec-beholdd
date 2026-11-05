# Captured catalog prefill qualification

The production Antimatter composer emitted a 32,389-byte request containing
39 tools and two messages. Its raw SHA-256 is
`0762094fb7f43a1d306224a51a865e61ec77b51831fae82c6ccc7b3d3bfa4353`.
The pinned Qwen3-0.6B Q8_0 tokenizer measures 8,079 prompt tokens, M4096 and
12,174 evaluated positions. See [production catalog admission](hag-editor-context.md#bounded-native-16k-and-production-catalog-admission--2026-10-18).

## Exact token handoff

`prompt_inspect MODEL.gguf REQUEST.json --tokens-out TOKENS.json` exports the
same token vector used for inspection. The request bytes are parsed through the
server's normalization and chat-template path; tokenization enables special
tokens. Export requires exactly one request and refuses an existing destination.
Ordinary inspection still accepts multiple request paths without an export.

The versioned JSON contains the complete token vector, request/checkpoint paths,
prompt length, tool/message counts, output budget and required context positions.
It is metadata-only: no checkpoint weight source, model execution or GPU opens.
Paths identify inputs; they are not cryptographic proof. The external supervisor
must bind raw request, token artifact, export receipt, source, binary and checkpoint
hashes before generating a CPU reference or opening the native device.

## Bounded first 32 tokens

The additive model-probe modes consume all 8,079 exported token IDs for validation,
then evaluate only the first 32. The independent CPU Oracle exports full-vocabulary
references at positions 0 and 31 and exits before native execution. The native job
allocates C16384, evaluates the same prefix, checks those two references, resets,
refuses stale logits and replays the first prompt token. It uses the existing
`2e-3 + 2e-4 * abs(CPU)` tolerance and requires exact greedy equality.

These are child command forms for the frozen supervisor, not standalone launch
instructions:

```text
hag_model_probe --catalog-prefill32-reference --model MODEL --prompt-tokens TOKENS --reference-out REF --input-identity ID
hag_model_probe --catalog-prefill32-smoke --model MODEL --prompt-tokens TOKENS --reference REF --input-identity ID /dev/dri/renderD128
```

Both phases retain the [C16384 process, memory and thermal controls](hag-16k-probes.md#supervised-invocation-contract),
including the mandatory owned cgroup and separate CPU-process retirement.
The two selected reference vectors occupy 1,215,488 payload bytes. The reference
format distinguishes this real-prompt prefix from the older fixed-token gate.

| Verified accounting | First-32-token pilot |
| --- | ---: |
| CPU KV payload | 7,340,032 bytes |
| CPU source reads | 11,136 |
| CPU logical source bytes | 20,271,892,480 |
| Native evaluations, including replay | 33 |
| Native dispatches / submissions | 125,169 / 1,914 |
| Native recording writes | 3,795,924 bytes |
| Native HAG allocation | 4,629,168,128 bytes |
| Compared logit values / readback bytes | 455,808 / 1,823,232 |

The existing CPU 2 GiB and native 6 GiB limits cover this pilot's forecasts.
The 360-second child deadline remains a bound, not a runtime prediction.
Hardware-free validation passed: 49 model-probe tests, 45 inspector/protocol
tests (including seven token-export tests), and 17 original supervisor tests.
The separate CPU phase and the subsequent native continuation both passed;
the continuation adds 19 passing supervisor tests. The observed results below
retain the pilot's bounded scope.

## Observed CPU result and authentication hold — 2026-10-18

The clean release build at `ea7ff9f8c9f96777525060e5f25ef4e688eabdc0`
exported the exact 8,079-token artifact. Its SHA-256 is
`56f520bb50ec480cc385bfddb00a137f264b7e5654a0df35c52d0ceeb7f89a82`;
the raw request hash above is unchanged. The frozen supervisor manifest is
`37456d16af73482b77bb6453a04a9b9fdd3f2eab0d92b473d04a33a1fc68bd71`.

The CPU evaluated all 32 prefix tokens in **44.721 seconds** at 25% CPU quota.
It exported 303,872 finite logits in a 1,215,776-byte reference artifact with SHA-256
`786e9ef679dc512157d74236627e9ba2dbafc27198cd302bddcbb15f7883e510`.
Position 0 consumes token 151644 and selects greedy token 14582; position 31
consumes token 12667 and selects greedy token 11. These are prefix-checkpoint
outputs, not a completion of the editor request. Source reads and bytes match
the planned CPU counts above. The CPU capacity refusal occurs before further
reads or cache mutation, and the reference process/cgroup fully retired.

Peak observed CPU cgroup memory was **62,185,472 bytes (59.30 MiB)**, Tctl reached
**45°C**, and host available memory remained at least **18,589,876,224 bytes
(17.31 GiB)**. All CPU memory event counters and swap usage were zero. The kernel interval
contained unrelated split-lock/AppArmor messages, with no matched GPU/OOM faults.
The cgroup peak is an observed process charge, not a replacement for the
804,062,400-byte conservative CPU phase forecast.

Linux authentication did not complete within the crash watcher's 120-second
deadline. The watcher never armed; neither attention nor native model execution
started in this attempt. The failed authentication and incomplete qualification
closeout are preserved. All seven recorded CPU/authentication/supervisor PIDs
and the CPU cgroup are absent; the host's boot-wide `oom_kill` counter remained 0.
The resource window was returned to operating-system integration. At that checkpoint, a new native
attempt required fresh frozen inputs and completed watcher authentication before
device access; the successful continuation below preserves this failed attempt.

Evidence is under `work/hag-catalog-prefill32-20261018`: `development.json`,
`runs/cpu-1792339147940113197/result.json`, `authentication-closeout.json`, and
`validation.json`. The very short formatting job has no cgroup samples; its
individual event counters are unobserved. The other five development jobs have
sampled zero memory/task events and swap, and all six jobs retired.

## Observed native continuation — 2026-10-18

Fresh watcher authentication succeeded, and the native continuation passed after
operating-system integration released the shared resource window. The compiled source remains
`ea7ff9f8c9f96777525060e5f25ef4e688eabdc0`; the observed repository HEAD was
`b5b817ab8481d059356db7700e93b57d9f305b94`, whose intervening changes were
documentation only. The continuation verified the complete compiled source
inventory and reused the original CPU receipt/reference without rerunning or
relabeling that phase. The exact request, token artifact, model binary, contract
and 128-byte reference identity remained unchanged.

The prerequisite attention diagnostic again passed at prefixes 8192/16383/16384:
1,536 values and 48 guards across six changed-query runs, with maximum absolute
error `5.960464477539063e-8`. The native model then evaluated the first 32 real
prompt tokens and reset/replayed the first token. All **455,808** compared logits
passed the unchanged tolerance, all three greedy outputs matched
`14582 / 11 / 14582`, and sign/zero mismatch counts were zero. Maximum absolute
error was **`3.910064697265625e-5`**. Stale logits were refused after reset;
unload returned all recorded native charges and the HAG budget to zero.

Preparation took **8.516 seconds**. Native execution and checks, including
reset/replay, took **11.293 seconds**. The first 32 step/check timings sum to
10.921386 seconds; that excludes replay and preparation. These timings are for
this bounded probe under a 25% CPU quota, not serving throughput. Dispatch,
submission, allocation, write and readback counts matched the table above.

Peak observed native cgroup memory was **696,324,096 bytes (664.07 MiB)**.
The separate HAG allocation was **4,629,168,128 bytes**; the process charge does
not represent total host/UMA consumption. Across the native outer supervisor and
probe, Tctl reached **55.875°C** and available host memory stayed above
**14,560,894,976 bytes (13.56 GiB)**. Sampled and final memory event counters and
swap usage were zero in both GPU stages. Both continuation development jobs also
passed with sampled zero memory/task events and swap.

The privileged read-only watcher captured no dumps and was explicitly stopped
and reaped after both GPU scopes retired. All recorded stage/watcher PIDs and
cgroups are absent; both development units are inactive. The host's boot-wide
`oom_kill` counter remained zero. Reviewed kernel intervals run from continuation
preparation through closeout, including each GPU stage: they contain unrelated
split-lock/AppArmor messages, with no matched GPU/OOM faults. This does not claim
an empty journal or kernel coverage back to watcher authentication. The shared
resource window was returned to operating-system integration after retirement.

Evidence is under `work/hag-catalog-prefill32-continuation-20261018`, with the
external watcher under `work/hag-catalog-prefill32-watch-v2-20261018` and the two
development jobs under `work/hag-catalog-prefill32-continuation-dev-20261018`.
The continuation manifest SHA-256 is
`2aef988516e9bbe9313e2e44619665cd2a61b7c434fa2383517d7d144cc17166`;
joined validation SHA-256 is
`f558f0c04f1bd3af2a7d2de33e86c82775b0cf9b7b36b2e1baf08bb0ea60341b`.
This qualifies only the first 32 tokens and reset/replay. The remaining 8,047
prompt tokens, generated completion, tool execution and full-context model
numerics remain unqualified.

## Scaling to the complete prompt

The current native serving path evaluates a complete model step, including its
vocabulary head, for each prompt token. Reading logits only at selected checkpoints
reduces readback, not computation. All 8,079 tokens would require 30,643,647
dispatches, 468,582 submissions and 10,989,887,388 recording-write bytes before
any generated continuation. Prefix timing is not a full-prompt throughput result:
attention work increases with the visible prefix.

Extending the token-at-a-time CPU Oracle directly would retain 1,853,128,704 bytes
of KV and make about 5.12 TB of logical source reads. For a later full reference,
`QwenModel::forward` can process the prompt one layer at a time and project only
the final-position logits. That path needs comparison against the independent
Oracle first and a separate memory forecast: its tied vocabulary head read is
165,306,368 bytes, exceeding the pilot's packed-source staging bound.

The next bounded step is a CPU-only equivalence gate for `QwenModel::forward`
at prefix lengths 1 and 32 against the saved Oracle outputs at positions 0 and
31. Preserve exact greedy equality and the existing logit tolerance, introduce
an explicit full-forward source-read bound and memory forecast, and measure
resource use before increasing the prefix length.

Use the pilot's observed metrics to choose a new finite full-prompt deadline and
crash-watcher lease. Complete prompt inference, generated answers, tool calls and
full-context model numerics remain outside this prefix qualification.
