# `mimo2` — G2 oracle gate record

**Date:** 2026-09-23. **Result:** passed. One block, synthetic weights, no real tensor read.

## What was run

```
cargo test -p beholdd-model mimo2
```

Two blocks, six positions, every step compared against an f64 transcription of the same
block. Worst disagreement across the whole residual stream:

```
worst disagreement over six positions: 3.39398e-7
```

The gate is set at `1e-5`, thirty times that.

## What the fixture is

`dim 128`, 4 query heads, key 24 / value 16 per head, 8 of each head rotated, window 3,
4 experts of which 2 run, expert width 32, dense width 64.

Two blocks is the fewest that covers what makes this architecture its own:

| | block 0 | block 1 |
|---|---|---|
| attention | full | sliding window 3 |
| kv heads | 2 (group 2) | 4 (group 1) |
| RoPE base | 1e7 | 1e4 |
| sink | no | yes |
| feed-forward | dense | routed, biased selection |

That is the real model's shape in miniature: block 0 of MiMo-V2.6-Flash really is the dense,
full-attention one, and block 1 really is windowed, sinked and routed. Six positions against
a window of three means the mask drops positions on the later tokens rather than merely being
present.

## Why an oracle rather than a golden file

There is no small `mimo2` checkpoint, and a golden vector taken from this implementation
would only prove the implementation had not changed — not that it was ever right. The oracle
is written from `modeling_mimo_v2.py` in the published repository rather than from the module
it checks, in f64, with no kernel calls and nothing fused. Where the reference is a one-line
tensor expression, the oracle's loop order is the order that expression implies.

Both sides are handed the **same f32 weights**. Giving the oracle wider weights would have
made the comparison a test of the cast.

## Whether the gate bites

A gate that cannot fail is decoration. Nine deliberate mistakes, each a thing a person could
plausibly write, were applied to the block and the suite re-run:

| mistake | caught |
|---|---|
| RoPE base always the full-attention one | yes |
| rotate the trailing dimensions, not the leading ones | yes |
| interleaved rope pairs instead of split-half | yes |
| value scale dropped | yes |
| attention scale from the value width, not the key width | yes |
| query heads grouped by remainder, not by quotient | yes |
| window off by one | yes |
| sink summed into the output instead of dropped after the softmax | yes |
| selection bias folded into the expert's weight | yes |

Every one missed the tolerance by orders of magnitude. A tenth attempt —
`rope_dim.min(key_head_dim)` — was caught by nothing, because `8.min(24)` is 8 and the
mutation was a no-op. Worth recording: a mutation that does not mutate proves nothing about
the test, and it looked like a gap until it was read.

## What this does not claim

No real tensor has been read. The weights here are a deterministic spread of small numbers,
not MXFP4 experts, and nothing has been quantised, streamed or cached across a re-entry.

The two facts most likely to be wrong are the two the oracle cannot check, because both
sides were written from the same reading of the reference: the **fused QKV order** (query,
then key, then value) and the **expert-major layout** of the stacked expert tensors. Reading
a real block, as G3 does below, checks the *widths* against the file but not the order within
them — an oracle handed the same slices agrees about them whichever way round they are. Only
a decode compared against another implementation settles that.

## G3 — the same block, read out of the real checkpoint

```
cargo run -p beholdd-model --release --example real_mimo2_block -- \
  "$BEHOLDD_MODEL_DIR"/mimo-v26-flash/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf

  block 0: Full, 4 kv heads
  read and dequantised 1.16 GB in 0.3s
  qkv 13568 = 12288 query + 768 key + 512 value, over 4096 in
  block 0.121s, oracle 0.157s
  worst disagreement 9.508e-7 against a residual reaching 3.949e-1
  relative 2.407e-6
```

Block 0 is the one to read first because it is the model's only dense block: every tensor it
needs is `Q8_0` or `F32`, so a wrong answer is the block or the dequantiser and never the
expert format, which has its own gate. The gate is relative rather than absolute — this
residual is not order one — and set at `3e-5`, twelve times the measurement rather than four,
because a bound that close to its own measurement trips on a machine that sums in a different
order and teaches everyone to ignore it.

The `13568 = 12288 + 768 + 512` line is the part that could not be checked against invented
weights. `attn_qkv` is one tensor in the file, and 64 query heads of 192, 4 key heads of 192
and 4 value heads of 128 adding to exactly its width is the checkpoint agreeing with the
binder about the head counts, the two head widths and the fused projection.

### The routed, windowed, sinked case

Block 0 is the model's exception. The other 47 are where the architecture lives, and they
could not be run at all until v2 could read an MXFP4 tensor — which it could not:
`mxfp4::matvec` had exactly one caller in the whole repository, a line in `cpu.rs` that
exists to keep the symbol alive. The kernel could multiply by experts nobody could load.

```
  block 1: SlidingWindow, 8 kv heads, routed (biased), sink
  qkv 14848 = 12288 query + 1536 key + 1024 value, over 4096 in
  router chose [137, 91, 84, 231, 156, 252, 104, 87]
  without the bias it would have chosen [104, 137, 84, 1, 231, 118, 171, 123] — 4 of 8 differ
  8 experts (0.81 GB of MXFP4 expanded) in 1.1s
  worst disagreement 4.151e-5 against a residual reaching 3.526e1
  relative 1.177e-6
```

**Half the experts change.** That line is what the selection bias is worth: ignoring
`exp_probs_b.bias`, as the first binder did, sends this token to four different experts out
of eight, and every shape still agrees, and the model still writes fluent English. It is the
most expensive thing found in this whole exercise and it cost a tensor census to see.

A routed block cannot be materialised whole — 256 experts of 4096×2048 over three tensors is
25.8 GB of `f32` — so only the eight the router selects are read, which is what the shipping
path will do anyway. The selection is made by the example, from the real router over all 256
experts with the real bias, before any expert is read; `forward` then re-derives the same
weights from the same scores. Its own top-k is what G2 covers.

## The MXFP4 layout, which is two layouts

ggml stores 32 values in 17 bytes: one E8M0 scale byte, then 16 packed bytes whose **low
nibbles are elements 0–15 and high nibbles are elements 16–31**. Split in half, exactly as
`Q4_0` is. beholdd's kernel wants something else — codes interleaved, low nibble the even
element, with the scales lifted into an array of their own so a row is contiguous.

`beholdd_core::mxfp4_unpack` decodes the second of those. Pointed at file bytes it does not
fail: it permutes the weights within every block, and a permuted block has the same values,
the same distribution and the same norm as the right one. Its own doc comment already said
this is "the first thing to suspect if a model is fluent and wrong". So the new reader
de-interleaves, and a test asserts the naive reading is a permutation rather than an error,
because that is the only way to state in code why the module exists.

The scales needed no adjustment, which is worth writing down because it looks like it should:
ggml folds a factor of two into its code table and takes it back out with a halved scale,
while beholdd holds the true E2M1 values against the true `2^(b-127)`. A test multiplies both
out at five different exponents and requires them equal.

## Asking the checkpoint about the fused QKV order

The order inside `attn_qkv` — query, then key, then value — was the one thing every gate had
taken on the reference's word and handed identically to both sides of the oracle, which
cannot disagree with itself. It turns out the file will answer.

A trained projection's three parts sit at different scales. So: find the split of the tail
that best separates it into two regimes, told nothing about where to look, and see which
order that split agrees with. `query | key | value` puts the boundary at `q + k`;
`query | value | key` puts it at `q + v`, 256 rows earlier on a 4-kv-head block.

```
  row-norm profile over the last 1280 rows of block 0, in chunks of 64:
 0.0831 0.0929 0.0918 0.0722 0.0863 0.0874 0.0928 0.0999
 0.1023 0.0496 0.0788 0.0741 0.0070 0.0071 0.0071 0.0071
 0.0100 0.0100 0.0065 0.0064

  the tail splits best at row 13056 (separation 25.2)
    query | key | value would put it at 13056
    query | value | key would put it at 12800
```

Twelve chunks of 64 at ~0.08, then eight at ~0.008: 768 rows and 512 rows, and the change-
point search lands on 13056 without being told it exists. Over all 48 blocks:

```
  25 blocks say query|key|value, 0 say query|value|key, 23 have no boundary to find
```

The 23 silent blocks are not evidence against. Where a block's key and value rows happen to
sit at the same scale — block 1's are 0.0101 and 0.0100 — there is no boundary to find, and
the separation score says one rather than picking a row at random. What matters is that no
block ever splits where the other order predicts.

The expert-major stacking never needed this treatment: the file's own dims say it out loud,
`[4096, 2048, 256]` for the gate and `[2048, 4096, 256]` for the down.

Remaining: G4 (coherent decode, resident and streamed identical), G5 (MTP drafting).
