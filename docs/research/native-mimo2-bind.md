# `mimo2` — G1 bind gate record

**Date:** 2026-09-23. **Result:** passed. Binding only; no execution claim is made here.

## What was run

```
cargo run -p beholdd-model --release --example bind_mimo2 -- \
  "$BEHOLDD_MODEL_DIR"/mimo-v26-flash/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf
cargo test -p beholdd-model mimo2
```

## What it reported

```
  bound in 0.027s -- header only

  48 blocks, dim 4096, vocab 152576
  attention: 64 q heads, key 192 / value 128 per head, value scale 0.707
  hybrid: 39 sliding-window blocks (window 128) and 9 full blocks
  rope: 64 of 192 dimensions rotated, base 1e7 full / 1e4 windowed
  experts: 256 routed, 8 used per token (3.12% active), inner width 2048
  context: 1048576

  first blocks:
     0: full           4 kv heads, group 16, rope base 1e7
     1: sliding-window 8 kv heads, group 8, rope base 1e4
     ...
     5: full           4 kv heads, group 16, rope base 1e7
```

Eight unit tests cover the refusals: a per-layer array whose length disagrees with
`block_count`, softmax gating declared where sigmoid is required, query heads that do not
group evenly over a block's key/value heads, a rotation wider than the key head, each
unimplemented semantic by name, and a missing hyperparameter that must not silently
default.

## Why 27 milliseconds matters

The checkpoint is 155.87 GiB and does not fit on one machine here; loading it under v1
takes about seven minutes across two. A binder that only fails at load makes every
architecture mistake cost seven minutes to see. Reading the header alone turns the
question "does beholdd understand this checkpoint" into something answerable before any
memory is committed, which is what makes the remaining gates affordable to iterate on.

## What this does not claim

No block has been executed. The `3.12%` activation, the window pattern and the split RoPE
bases are what the checkpoint *declares*; whether beholdd computes them correctly is G2
(one block against an f64 oracle on a synthetic fixture), G3 (the same against one real
block) and G4 (coherent decode). The MXFP4 expert tensors have not been read.

One detail worth carrying into G2: the real pattern is 39 windowed to 9 full, which is not
the uniform "every fifth block" the synthetic fixture uses. The fixture exercises both
attention types and both RoPE bases, which is what it is for, but a block index taken from
it is not a block index in the real model.

## Correction, same day

The paragraph above was right that the fixture's pattern was invented, and wrong about how
much that mattered. The fixture now transcribes the real one — block 0, then every sixth
block from block 5 — because two further facts turned out to be per-block and to be readable
only from the tensor table, which the first pass did not read at all:

```
48 attn_qkv / attn_output / attn_norm / ffn_norm
47 ffn_gate_exps + ffn_gate_inp + exp_probs_b.bias
39 attn_sinks.weight
 1 ffn_gate / ffn_up / ffn_down
```

**Block 0 is dense.** Every other block routes. A binder that assumed a uniform MoE, as this
one did, would look for experts block 0 does not have — and `feed_forward_length = 16384` was
sitting in the metadata the whole time as the width of the one dense block, bound but unused.

**`exp_probs_b.bias` is a routing input.** It is added to the sigmoid scores before the top-k
choice and does not change the weight a chosen expert gets. Ignoring it changes *which*
experts run while every shape still agrees, which is the class of mistake that produces
plausible text and no error.

The binder now reads both from the tensor table, refuses `expert_weights_scale` alongside
`expert_weights_norm`, and refuses group routing with more than one group. The real
checkpoint still binds, in 24 ms:

```
  feed-forward: 47 routed, 1 dense (width 16384)
  sinks: 39 blocks carry one
     0: full           4 kv heads, group 16, rope base 1e7, dense ffn
     1: sliding-window 8 kv heads, group  8, rope base 1e4, routed (biased selection), sink
     5: full           4 kv heads, group 16, rope base 1e7, routed (biased selection)
```

Four more tests cover it: a block with no feed-forward, a block with both, a router whose
bias is absent, and group routing declared with eight groups.
