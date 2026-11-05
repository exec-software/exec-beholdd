//! One `mimo2` block, executed.
//!
//! Gate G2 of ADR-0002. The arithmetic here is checked against [`super::oracle`], an f64
//! transcription of the same block written from `modeling_mimo_v2.py` rather than from this
//! file, so the two disagree wherever one of them has the architecture wrong.
//!
//! Weights are plain `f32` slices. The shipping path will hand the backend quantised rows —
//! the experts are natively MXFP4 and nothing else in the checkpoint is wider than Q8_0 —
//! but quantisation is a separate claim from "the block computes MiMo", and mixing them
//! would make a failure ambiguous between the two. G3 reads real tensors; this gate settles
//! the shape of the computation.
//!
//! **Layout.** A weight matrix is row-major over its output rows: `w[o * n_in + i]`, which is
//! GGUF's `[in, out]` dims read in order. `attn_qkv.weight` is `[4096, 13568]` in the real
//! checkpoint, so it is 13568 rows of 4096.

use anyhow::{bail, ensure, Result};
use beholdd_kernel::vecops;

use super::{Attention, Config, Ffn};

/// Which feed-forward weights a block carries, matching its bound [`Ffn`].
#[derive(Debug, Clone, Copy)]
pub enum FfnWeights<'a> {
    /// `blk.N.ffn_{gate,up,down}.weight`, evaluated for every token.
    Dense {
        /// `[dim, feed_forward]`.
        gate: &'a [f32],
        /// `[dim, feed_forward]`.
        up: &'a [f32],
        /// `[feed_forward, dim]`.
        down: &'a [f32],
    },
    /// `blk.N.ffn_gate_inp.weight` plus the stacked expert tensors.
    Routed {
        /// `[dim, n_experts]` — the router.
        router: &'a [f32],
        /// `blk.N.exp_probs_b.bias`, `n_experts` wide. Steers selection only.
        bias: Option<&'a [f32]>,
        /// `n_experts` × `[dim, expert_ff]`, expert-major.
        gate: &'a [f32],
        /// `n_experts` × `[dim, expert_ff]`, expert-major.
        up: &'a [f32],
        /// `n_experts` × `[expert_ff, dim]`, expert-major.
        down: &'a [f32],
    },
}

/// Everything one block needs, in the checkpoint's own layout.
#[derive(Debug, Clone, Copy)]
pub struct Weights<'a> {
    /// `blk.N.attn_norm.weight`, `dim` wide.
    pub attn_norm: &'a [f32],
    /// `blk.N.attn_qkv.weight` — query, then key, then value rows, in that order.
    pub qkv: &'a [f32],
    /// `blk.N.attn_output.weight`, `[n_heads * value_head_dim, dim]`.
    pub attn_out: &'a [f32],
    /// `blk.N.attn_sinks.weight`, one logit per query head, on the blocks that carry it.
    pub sinks: Option<&'a [f32]>,
    /// `blk.N.ffn_norm.weight`, `dim` wide.
    pub ffn_norm: &'a [f32],
    /// The block's feed-forward.
    pub ffn: FfnWeights<'a>,
}

/// Keys and values already computed for earlier positions of this block.
///
/// Flat and per-block: one of these belongs to one layer, because MiMo's key/value head count
/// varies by layer and a cache shaped for one block is the wrong shape for its neighbour.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    /// `positions` × `n_kv_heads * key_head_dim`, rotated.
    pub keys: Vec<f32>,
    /// `positions` × `n_kv_heads * value_head_dim`, already value-scaled.
    pub values: Vec<f32>,
    /// How many positions are stored.
    pub positions: usize,
}

impl Cache {
    /// An empty cache for a block.
    pub fn new() -> Cache {
        Cache::default()
    }
}

/// Run one token through block `layer`, appending its key and value to `cache`.
///
/// `x` is the residual stream, `dim` wide, and is updated in place — both residual additions
/// happen here, so the caller stacks blocks by calling this in order.
pub fn forward(
    cfg: &Config,
    layer: usize,
    w: &Weights<'_>,
    x: &mut [f32],
    cache: &mut Cache,
) -> Result<()> {
    let block = *cfg
        .blocks
        .get(layer)
        .ok_or_else(|| anyhow::anyhow!("layer {layer} is past the {} in this model", cfg.n_layers()))?;
    ensure!(x.len() == cfg.dim, "residual is {} wide, model is {}", x.len(), cfg.dim);
    check_shapes(cfg, layer, w, block.ffn)?;

    let q_size = cfg.n_heads * cfg.key_head_dim;
    let k_size = block.n_kv_heads * cfg.key_head_dim;
    let v_size = block.n_kv_heads * cfg.value_head_dim;

    // --- attention ---------------------------------------------------------------------
    let mut h = vec![0.0f32; cfg.dim];
    vecops::rmsnorm(&mut h, x, Some(w.attn_norm), cfg.eps);

    let mut qkv = vec![0.0f32; q_size + k_size + v_size];
    matvec(&mut qkv, &h, w.qkv, cfg.dim);
    let (q, kv) = qkv.split_at_mut(q_size);
    let (k, v) = kv.split_at_mut(k_size);

    // The value scale is applied once, here, so everything downstream — the cache included —
    // holds scaled values. Applying it to the attention output instead would give the same
    // answer for one token and a different one for every cached position after a re-entry.
    for value in v.iter_mut() {
        *value *= cfg.value_scale;
    }

    let pos = cache.positions;
    let freqs = vecops::rope_freqs(cfg.rope_base_for(layer)?, cfg.rope_dim, 0, 1.0, 0.0, 0.0);
    let mut cos = vec![0.0f32; cfg.rope_dim / 2];
    let mut sin = vec![0.0f32; cfg.rope_dim / 2];
    vecops::rope_at(&freqs, pos, &mut cos, &mut sin);
    // Only the leading `rope_dim` of each head rotates; the rest of the head carries no
    // position at all. Rotating the whole head is the mistake this loop exists to not make.
    for head in 0..cfg.n_heads {
        let at = head * cfg.key_head_dim;
        vecops::rope_apply_split_half(&mut q[at..at + cfg.rope_dim], &cos, &sin, false);
    }
    for head in 0..block.n_kv_heads {
        let at = head * cfg.key_head_dim;
        vecops::rope_apply_split_half(&mut k[at..at + cfg.rope_dim], &cos, &sin, false);
    }

    cache.keys.extend_from_slice(k);
    cache.values.extend_from_slice(v);
    cache.positions += 1;
    let seen = cache.positions;

    // A windowed block attends to the last `window` positions, this one included; a full
    // block attends to everything before it.
    let first = match block.attention {
        Attention::Full => 0,
        Attention::SlidingWindow => seen.saturating_sub(cfg.window),
    };
    let scale = 1.0 / (cfg.key_head_dim as f32).sqrt();
    let group = cfg.group(layer)?;

    let mut attended = vec![0.0f32; cfg.n_heads * cfg.value_head_dim];
    // One row longer than the positions attended to: the sink occupies the last entry when
    // the block has one, takes part in the softmax, and is then dropped rather than summed
    // into the output. That is what lets a position decline to attend.
    let mut row = vec![0.0f32; seen - first + 1];
    for head in 0..cfg.n_heads {
        let kv_head = head / group;
        let q_at = head * cfg.key_head_dim;
        let width = seen - first;
        for (slot, position) in (first..seen).enumerate() {
            let k_at = position * k_size + kv_head * cfg.key_head_dim;
            row[slot] = dot(&q[q_at..q_at + cfg.key_head_dim], &cache.keys[k_at..k_at + cfg.key_head_dim])
                * scale;
        }
        let used = match w.sinks {
            Some(sinks) => {
                row[width] = sinks[head];
                width + 1
            }
            None => width,
        };
        vecops::softmax(&mut row[..used]);

        let out_at = head * cfg.value_head_dim;
        for (slot, position) in (first..seen).enumerate() {
            let weight = row[slot];
            let v_at = position * v_size + kv_head * cfg.value_head_dim;
            for i in 0..cfg.value_head_dim {
                attended[out_at + i] += weight * cache.values[v_at + i];
            }
        }
    }

    let mut projected = vec![0.0f32; cfg.dim];
    matvec(&mut projected, &attended, w.attn_out, cfg.n_heads * cfg.value_head_dim);
    for (residual, delta) in x.iter_mut().zip(&projected) {
        *residual += delta;
    }

    // --- feed-forward ------------------------------------------------------------------
    vecops::rmsnorm(&mut h, x, Some(w.ffn_norm), cfg.eps);
    let mut out = vec![0.0f32; cfg.dim];
    match w.ffn {
        FfnWeights::Dense { gate, up, down } => {
            feed_forward(&mut out, &h, Mlp { gate, up, down }, cfg.dim, cfg.feed_forward, 1.0);
        }
        FfnWeights::Routed { router, bias, gate, up, down } => {
            let mut scores = vec![0.0f32; cfg.n_experts];
            matvec(&mut scores, &h, router, cfg.dim);
            for score in scores.iter_mut() {
                *score = sigmoid(*score);
            }
            for (expert, weight) in choose(&scores, bias, cfg.n_experts_used, cfg.norm_topk) {
                let gate_at = expert * cfg.dim * cfg.expert_ff;
                let down_at = expert * cfg.expert_ff * cfg.dim;
                feed_forward(
                    &mut out,
                    &h,
                    Mlp {
                        gate: &gate[gate_at..gate_at + cfg.dim * cfg.expert_ff],
                        up: &up[gate_at..gate_at + cfg.dim * cfg.expert_ff],
                        down: &down[down_at..down_at + cfg.expert_ff * cfg.dim],
                    },
                    cfg.dim,
                    cfg.expert_ff,
                    weight,
                );
            }
        }
    }
    for (residual, delta) in x.iter_mut().zip(&out) {
        *residual += delta;
    }
    Ok(())
}

/// One feed-forward's three matrices, whichever expert they were sliced out of.
struct Mlp<'a> {
    gate: &'a [f32],
    up: &'a [f32],
    down: &'a [f32],
}

/// `down(silu(gate(h)) * up(h))`, scaled by `weight` and accumulated into `out`.
///
/// Accumulating rather than assigning is what lets the routed case sum its experts without a
/// scratch buffer per expert.
fn feed_forward(out: &mut [f32], h: &[f32], mlp: Mlp<'_>, dim: usize, inner: usize, weight: f32) {
    let Mlp { gate, up, down } = mlp;
    let mut g = vec![0.0f32; inner];
    let mut u = vec![0.0f32; inner];
    matvec(&mut g, h, gate, dim);
    matvec(&mut u, h, up, dim);
    // No clamp: MiMo declares no activation limit, and `swiglu`'s clamps are asymmetric, so
    // passing a non-zero limit here would change the activation rather than bound it.
    vecops::swiglu(&mut g, &u, 0.0);
    for (o, row) in out.iter_mut().enumerate() {
        *row += weight * dot(&g, &down[o * inner..(o + 1) * inner]);
    }
}

/// The experts this token routes to, and the weight each contributes.
///
/// Selection reads `scores + bias`; the weight is the score **without** the bias. The bias
/// exists to move which experts are chosen without moving what a chosen one is worth, and
/// folding it into the weight would quietly change the second.
///
/// With `norm`, the chosen weights are then divided by their sum, so the feed-forward's
/// contribution does not grow with how confident the router happened to be. Without it eight
/// sigmoid scores sum to roughly four and every routed block shouts; see `Config::norm_topk`.
fn choose(scores: &[f32], bias: Option<&[f32]>, k: usize, norm: bool) -> Vec<(usize, f32)> {
    let mut ranked: Vec<(usize, f32)> = scores
        .iter()
        .enumerate()
        .map(|(expert, &score)| (expert, score + bias.map_or(0.0, |b| b[expert])))
        .collect();
    // Ties break towards the lower expert index, which is what `topk` does and what makes
    // this reproducible rather than dependent on the sort's stability.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    ranked.truncate(k);
    let mut chosen: Vec<(usize, f32)> =
        ranked.into_iter().map(|(expert, _)| (expert, scores[expert])).collect();
    if norm && chosen.len() > 1 {
        // The reference adds 1e-20 before dividing. Keep it: the denominator is a sum of
        // sigmoids and cannot be zero in practice, but a guard that costs nothing is cheaper
        // than a NaN that reaches the residual stream.
        let total: f32 = chosen.iter().map(|(_, weight)| *weight).sum::<f32>() + 1e-20;
        for (_, weight) in chosen.iter_mut() {
            *weight /= total;
        }
    }
    chosen
}

/// `y = W x`, with `w` row-major over output rows of `n_in` each.
fn matvec(y: &mut [f32], x: &[f32], w: &[f32], n_in: usize) {
    for (o, out) in y.iter_mut().enumerate() {
        *out = dot(x, &w[o * n_in..(o + 1) * n_in]);
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Every weight this block will index, checked once before any of it is read.
///
/// A slice that is short by one row is a panic deep inside a loop; a slice that is *long* is
/// silently the wrong tensor. Both are worth catching where the shapes are still named.
fn check_shapes(cfg: &Config, layer: usize, w: &Weights<'_>, ffn: Ffn) -> Result<()> {
    let block = cfg.blocks[layer];
    let q_size = cfg.n_heads * cfg.key_head_dim;
    let k_size = block.n_kv_heads * cfg.key_head_dim;
    let v_size = block.n_kv_heads * cfg.value_head_dim;
    let want = [
        ("attn_norm", w.attn_norm.len(), cfg.dim),
        ("attn_qkv", w.qkv.len(), (q_size + k_size + v_size) * cfg.dim),
        ("attn_output", w.attn_out.len(), cfg.dim * cfg.n_heads * cfg.value_head_dim),
        ("ffn_norm", w.ffn_norm.len(), cfg.dim),
    ];
    for (name, have, expected) in want {
        ensure!(have == expected, "blk.{layer}.{name} holds {have} weights, block needs {expected}");
    }
    match (ffn, w.ffn) {
        (Ffn::Dense, FfnWeights::Dense { gate, up, down }) => {
            let plane = cfg.dim * cfg.feed_forward;
            for (name, have) in [("ffn_gate", gate.len()), ("ffn_up", up.len()), ("ffn_down", down.len())] {
                ensure!(have == plane, "blk.{layer}.{name} holds {have} weights, block needs {plane}");
            }
        }
        (Ffn::Routed { selection_bias }, FfnWeights::Routed { router, bias, gate, up, down }) => {
            let plane = cfg.n_experts * cfg.dim * cfg.expert_ff;
            ensure!(
                router.len() == cfg.n_experts * cfg.dim,
                "blk.{layer}.ffn_gate_inp holds {} weights, block needs {}",
                router.len(),
                cfg.n_experts * cfg.dim
            );
            for (name, have) in
                [("ffn_gate_exps", gate.len()), ("ffn_up_exps", up.len()), ("ffn_down_exps", down.len())]
            {
                ensure!(have == plane, "blk.{layer}.{name} holds {have} weights, block needs {plane}");
            }
            // The binder read whether this block has a bias out of the tensor table. Weights
            // that disagree with it mean one of the two is describing a different checkpoint.
            ensure!(
                selection_bias == bias.is_some(),
                "blk.{layer} was bound with selection_bias={selection_bias} but was given {} bias",
                if bias.is_some() { "a" } else { "no" }
            );
            if let Some(bias) = bias {
                ensure!(
                    bias.len() == cfg.n_experts,
                    "blk.{layer}.exp_probs_b holds {} entries for {} experts",
                    bias.len(),
                    cfg.n_experts
                );
            }
        }
        (bound, _) => bail!("blk.{layer} was bound as {bound:?} and given the other kind of weights"),
    }
    match (block.sink, w.sinks) {
        (true, Some(sinks)) => ensure!(
            sinks.len() == cfg.n_heads,
            "blk.{layer}.attn_sinks holds {} entries for {} query heads",
            sinks.len(),
            cfg.n_heads
        ),
        (true, None) => bail!("blk.{layer} was bound with a sink and given none"),
        (false, Some(_)) => bail!("blk.{layer} was bound without a sink and given one"),
        (false, None) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mimo2::{oracle, Attention, Block};

    /// A `mimo2` small enough to hold in the head, and hybrid enough to be MiMo.
    ///
    /// Block 0 is full attention with a dense feed-forward — the shape of the real block 0.
    /// Block 1 is windowed, sinked, and routed over four experts with a selection bias. Two
    /// blocks is the fewest that covers both attention types, both RoPE bases and both kinds
    /// of feed-forward, which is what this fixture is for.
    fn config() -> Config {
        Config {
            blocks: vec![
                Block {
                    attention: Attention::Full,
                    n_kv_heads: 2,
                    ffn: Ffn::Dense,
                    sink: false,
                },
                Block {
                    attention: Attention::SlidingWindow,
                    n_kv_heads: 4,
                    ffn: Ffn::Routed { selection_bias: true },
                    sink: true,
                },
            ],
            dim: 128,
            n_heads: 4,
            key_head_dim: 24,
            value_head_dim: 16,
            rope_dim: 8,
            rope_base: 1e7,
            rope_base_swa: 1e4,
            window: 3,
            value_scale: 0.707,
            n_experts: 4,
            n_experts_used: 2,
            expert_ff: 32,
            feed_forward: 64,
            vocab: 32,
            eps: 1e-6,
            context: 64,
            norm_topk: true,
        }
    }

    /// A deterministic spread of small weights. Not random: the same fixture must produce the
    /// same disagreement on every machine, or a failure here is not reproducible.
    fn weights(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                ((unit - 0.5) * 0.2) as f32
            })
            .collect()
    }

    struct Held {
        attn_norm: Vec<f32>,
        qkv: Vec<f32>,
        attn_out: Vec<f32>,
        sinks: Vec<f32>,
        ffn_norm: Vec<f32>,
        a: Vec<f32>,
        b: Vec<f32>,
        c: Vec<f32>,
        router: Vec<f32>,
        bias: Vec<f32>,
    }

    fn held(cfg: &Config, layer: usize) -> Held {
        let block = cfg.blocks[layer];
        let q = cfg.n_heads * cfg.key_head_dim;
        let k = block.n_kv_heads * cfg.key_head_dim;
        let v = block.n_kv_heads * cfg.value_head_dim;
        let seed = 1 + layer as u64 * 17;
        let (inner, experts) = match block.ffn {
            Ffn::Dense => (cfg.feed_forward, 1),
            Ffn::Routed { .. } => (cfg.expert_ff, cfg.n_experts),
        };
        Held {
            // Norm gains sit around one; a norm weight drawn around zero would scale the
            // whole block to nothing and hide a mistake under the noise floor.
            attn_norm: weights(cfg.dim, seed).iter().map(|w| 1.0 + w).collect(),
            qkv: weights((q + k + v) * cfg.dim, seed + 1),
            attn_out: weights(cfg.dim * cfg.n_heads * cfg.value_head_dim, seed + 2),
            sinks: weights(cfg.n_heads, seed + 3).iter().map(|w| w * 10.0).collect(),
            ffn_norm: weights(cfg.dim, seed + 4).iter().map(|w| 1.0 + w).collect(),
            a: weights(experts * cfg.dim * inner, seed + 5),
            b: weights(experts * cfg.dim * inner, seed + 6),
            c: weights(experts * inner * cfg.dim, seed + 7),
            router: weights(cfg.n_experts * cfg.dim, seed + 8),
            bias: weights(cfg.n_experts, seed + 9),
        }
    }

    fn view<'a>(cfg: &Config, layer: usize, h: &'a Held, bias: bool) -> Weights<'a> {
        Weights {
            attn_norm: &h.attn_norm,
            qkv: &h.qkv,
            attn_out: &h.attn_out,
            sinks: cfg.blocks[layer].sink.then_some(h.sinks.as_slice()),
            ffn_norm: &h.ffn_norm,
            ffn: match cfg.blocks[layer].ffn {
                Ffn::Dense => FfnWeights::Dense { gate: &h.a, up: &h.b, down: &h.c },
                Ffn::Routed { .. } => FfnWeights::Routed {
                    router: &h.router,
                    bias: bias.then_some(h.bias.as_slice()),
                    gate: &h.a,
                    up: &h.b,
                    down: &h.c,
                },
            },
        }
    }

    fn token(cfg: &Config, position: usize) -> Vec<f32> {
        weights(cfg.dim, 900 + position as u64).iter().map(|w| w * 5.0).collect()
    }

    /// How far f32 accumulation may drift from the f64 oracle before it means something.
    ///
    /// Measured at 3.4e-7 over six positions, so this leaves thirty times the observed
    /// margin. It is not arbitrary in the other direction either: every mistake tried
    /// against it — the wrong RoPE base, the trailing dimensions rotated instead of the
    /// leading ones, interleaved pairs instead of split-half, the value scale dropped, the
    /// attention scale taken from the value width, query heads grouped by remainder, the
    /// window off by one, the sink summed into the output, the selection bias folded into
    /// the weight — misses it by orders of magnitude rather than squeaking past.
    const TOLERANCE: f64 = 1e-5;

    /// G2: both blocks, six positions, against the oracle.
    ///
    /// Six positions with a window of three means block 1 drops positions on the later
    /// tokens, so the mask is exercised rather than merely present.
    #[test]
    fn a_block_agrees_with_the_f64_oracle() {
        let cfg = config();
        let holders: Vec<Held> = (0..cfg.n_layers()).map(|layer| held(&cfg, layer)).collect();

        let mut caches: Vec<Cache> = (0..cfg.n_layers()).map(|_| Cache::new()).collect();
        let mut references: Vec<oracle::Cache> =
            (0..cfg.n_layers()).map(|_| oracle::Cache::new()).collect();
        let mut worst = 0.0f64;

        for position in 0..6 {
            let start = token(&cfg, position);
            let mut x: Vec<f32> = start.clone();
            let mut reference: Vec<f64> = start.iter().map(|v| *v as f64).collect();

            for layer in 0..cfg.n_layers() {
                let w = view(&cfg, layer, &holders[layer], true);
                forward(&cfg, layer, &w, &mut x, &mut caches[layer]).unwrap();
                oracle::forward(&cfg, layer, &w, &mut reference, &mut references[layer]).unwrap();
            }

            for (got, want) in x.iter().zip(&reference) {
                worst = worst.max((*got as f64 - want).abs());
            }
            assert!(
                worst < TOLERANCE,
                "position {position} diverged by {worst:e}; f32 accumulation over {} should not",
                cfg.dim
            );
        }
        assert!(worst < TOLERANCE, "worst disagreement {worst:e}");
    }

    /// The sink is the reason the attention weights do not sum to one.
    #[test]
    fn a_sink_takes_weight_away_from_the_positions() {
        let cfg = config();
        let holders: Vec<Held> = (0..cfg.n_layers()).map(|layer| held(&cfg, layer)).collect();

        let run = |sinks: Option<&[f32]>| {
            let mut w = view(&cfg, 1, &holders[1], true);
            w.sinks = sinks;
            let mut x = token(&cfg, 0);
            let mut cache = Cache::new();
            // The shape check is the binder's agreement with the weights; here the point is
            // the arithmetic, so run the block that was bound without a sink for the control.
            let mut cfg = cfg.clone();
            cfg.blocks[1].sink = sinks.is_some();
            forward(&cfg, 1, &w, &mut x, &mut cache).unwrap();
            x
        };

        let with = run(Some(holders[1].sinks.as_slice()));
        let without = run(None);
        let moved: f32 = with.iter().zip(&without).map(|(a, b)| (a - b).abs()).sum();
        assert!(moved > 1e-3, "a sink that changes nothing is not being applied: {moved:e}");
    }

    /// The window is not decoration: at position 5 with a window of 3, the first two
    /// positions must not reach the answer.
    #[test]
    fn the_window_actually_masks() {
        let cfg = config();
        let holders: Vec<Held> = (0..cfg.n_layers()).map(|layer| held(&cfg, layer)).collect();
        let w = view(&cfg, 1, &holders[1], true);

        // Same last three tokens, different first three. A windowed block must answer the
        // same; the difference, if any, arrived through positions it should not have seen.
        let run = |first: u64| {
            let mut cache = Cache::new();
            let mut last = Vec::new();
            for position in 0..6 {
                let seed = if position < 3 { first + position as u64 } else { 500 + position as u64 };
                let mut x: Vec<f32> = weights(cfg.dim, seed).iter().map(|v| v * 5.0).collect();
                forward(&cfg, 1, &w, &mut x, &mut cache).unwrap();
                last = x;
            }
            last
        };

        let a = run(10);
        let b = run(70);
        let drift: f32 = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).sum();
        assert!(drift < 1e-4, "positions outside the window reached the answer: {drift:e}");
    }

    /// The bias moves which experts run. If it did not, it would not be worth binding.
    #[test]
    fn the_selection_bias_changes_which_experts_run() {
        let scores = [0.51f32, 0.49, 0.30, 0.10];
        let plain = choose(&scores, None, 2, false);
        assert_eq!(plain.iter().map(|(e, _)| *e).collect::<Vec<_>>(), vec![0, 1]);

        // Enough bias on expert 3 to overtake expert 1, which must not change what either
        // contributes once chosen.
        let bias = [0.0f32, 0.0, 0.0, 0.5];
        let biased = choose(&scores, Some(&bias), 2, false);
        assert_eq!(biased.iter().map(|(e, _)| *e).collect::<Vec<_>>(), vec![3, 0]);
        assert!(
            biased.iter().any(|(e, weight)| *e == 3 && (*weight - 0.10).abs() < 1e-6),
            "the weight must be the score without the bias, got {biased:?}"
        );
    }

    /// The weights the experts are summed with add to one, whatever the router's confidence.
    #[test]
    fn normalised_routing_weights_sum_to_one() {
        for scores in [[0.9f32, 0.8, 0.7, 0.6], [0.05, 0.04, 0.03, 0.02]] {
            let chosen = choose(&scores, None, 2, true);
            let total: f32 = chosen.iter().map(|(_, weight)| *weight).sum();
            assert!((total - 1.0).abs() < 1e-6, "weights summed to {total} for {scores:?}");
            // Which experts run is not affected by normalising what they are worth.
            assert_eq!(chosen.iter().map(|(e, _)| *e).collect::<Vec<_>>(), vec![0, 1]);
        }
        // Unnormalised, the same two weights add to 1.7 — the bug G4 caught, stated as a
        // number: a routed block speaking 1.7 times too loudly, every layer, compounding.
        let loose: f32 = choose(&[0.9, 0.8, 0.7, 0.6], None, 2, false).iter().map(|(_, w)| *w).sum();
        assert!((loose - 1.7).abs() < 1e-6, "{loose}");
    }

    /// Weights that disagree with what the binder read are refused before any of them is used.
    #[test]
    fn weights_that_contradict_the_binding_are_refused() {
        let cfg = config();
        let holders: Vec<Held> = (0..cfg.n_layers()).map(|layer| held(&cfg, layer)).collect();

        let mut w = view(&cfg, 1, &holders[1], false);
        let mut x = token(&cfg, 0);
        let error = forward(&cfg, 1, &w, &mut x, &mut Cache::new())
            .expect_err("a block bound with a bias must not run without one");
        assert!(format!("{error:#}").contains("selection_bias=true"), "{error:#}");

        w = view(&cfg, 1, &holders[1], true);
        w.sinks = None;
        let error = forward(&cfg, 1, &w, &mut x, &mut Cache::new())
            .expect_err("a block bound with a sink must not run without one");
        assert!(format!("{error:#}").contains("given none"), "{error:#}");

        let short = vec![0.0f32; cfg.dim - 1];
        w = view(&cfg, 1, &holders[1], true);
        w.attn_norm = &short;
        let error = forward(&cfg, 1, &w, &mut x, &mut Cache::new())
            .expect_err("a short norm must be named, not panicked on");
        assert!(format!("{error:#}").contains("attn_norm"), "{error:#}");
    }
}
