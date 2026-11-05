//! An f64 transcription of one `mimo2` block.
//!
//! This is the reference [`super::block`] is diffed against, and it is deliberately the
//! stupidest correct thing: every step is a loop over `f64`, nothing is fused, nothing is
//! reused, no kernel is called. It is slow on purpose — an oracle that shares an
//! optimisation with the code it checks cannot catch a mistake in that optimisation.
//!
//! It is written from `modeling_mimo_v2.py` in the published repository rather than from the
//! module it checks, so the two agree only if both have the architecture right. Where the
//! reference is a one-line tensor expression, the order here is the order it implies:
//!
//! - `q, k` rotate on their leading `rope_dim` with `rotate_half`, the rest untouched;
//! - `v` is multiplied by `v_scale` at projection;
//! - logits are scaled by `key_head_dim^-0.5`, then the per-head sink is **appended as one
//!   more column**, then softmax, then the sink column is dropped;
//! - routing is `sigmoid`, top-k over `scores + bias`, weighted by the score without bias,
//!   and those weights are divided by their sum (`norm_topk_prob`, true for every published
//!   V2 checkpoint and absent from the GGUF, which is how it came to be missed).
//!
//! Weights arrive as the same `f32` the block gets and are promoted here. Handing the oracle
//! wider weights would make the comparison a test of the cast rather than of the arithmetic.

use anyhow::Result;

use super::block::{FfnWeights, Weights};
use super::{Attention, Config};

/// Keys and values for earlier positions, in f64.
#[derive(Debug, Clone, Default)]
pub struct Cache {
    /// `positions` × `n_kv_heads * key_head_dim`, rotated.
    pub keys: Vec<f64>,
    /// `positions` × `n_kv_heads * value_head_dim`, already value-scaled.
    pub values: Vec<f64>,
    /// How many positions are stored.
    pub positions: usize,
}

impl Cache {
    /// An empty cache.
    pub fn new() -> Cache {
        Cache::default()
    }
}

/// Run one token through block `layer` of `cfg`, in f64.
pub fn forward(
    cfg: &Config,
    layer: usize,
    w: &Weights<'_>,
    x: &mut [f64],
    cache: &mut Cache,
) -> Result<()> {
    let block = cfg.blocks[layer];
    let q_size = cfg.n_heads * cfg.key_head_dim;
    let k_size = block.n_kv_heads * cfg.key_head_dim;
    let v_size = block.n_kv_heads * cfg.value_head_dim;

    let h = rmsnorm(x, w.attn_norm, cfg.eps as f64);
    let qkv = matvec(&h, w.qkv, cfg.dim, q_size + k_size + v_size);
    let mut q = qkv[..q_size].to_vec();
    let mut k = qkv[q_size..q_size + k_size].to_vec();
    let mut v = qkv[q_size + k_size..].to_vec();
    for value in v.iter_mut() {
        *value *= cfg.value_scale as f64;
    }

    let pos = cache.positions;
    let theta = cfg.rope_base_for(layer)?;
    for head in 0..cfg.n_heads {
        rotate(&mut q[head * cfg.key_head_dim..][..cfg.rope_dim], pos, theta);
    }
    for head in 0..block.n_kv_heads {
        rotate(&mut k[head * cfg.key_head_dim..][..cfg.rope_dim], pos, theta);
    }
    cache.keys.extend_from_slice(&k);
    cache.values.extend_from_slice(&v);
    cache.positions += 1;
    let seen = cache.positions;

    let first = match block.attention {
        Attention::Full => 0,
        Attention::SlidingWindow => seen.saturating_sub(cfg.window),
    };
    let scale = 1.0 / (cfg.key_head_dim as f64).sqrt();
    let group = cfg.n_heads / block.n_kv_heads;

    let mut attended = vec![0.0f64; cfg.n_heads * cfg.value_head_dim];
    for head in 0..cfg.n_heads {
        let kv_head = head / group;
        let mut logits: Vec<f64> = (first..seen)
            .map(|position| {
                let mut sum = 0.0;
                for i in 0..cfg.key_head_dim {
                    sum += q[head * cfg.key_head_dim + i]
                        * cache.keys[position * k_size + kv_head * cfg.key_head_dim + i];
                }
                sum * scale
            })
            .collect();
        let attended_to = logits.len();
        if let Some(sinks) = w.sinks {
            logits.push(sinks[head] as f64);
        }
        let weights = softmax(&logits);
        for (slot, position) in (first..seen).enumerate() {
            for i in 0..cfg.value_head_dim {
                attended[head * cfg.value_head_dim + i] +=
                    weights[slot] * cache.values[position * v_size + kv_head * cfg.value_head_dim + i];
            }
        }
        debug_assert_eq!(attended_to, seen - first);
    }

    let projected = matvec(&attended, w.attn_out, cfg.n_heads * cfg.value_head_dim, cfg.dim);
    for (residual, delta) in x.iter_mut().zip(&projected) {
        *residual += delta;
    }

    let h = rmsnorm(x, w.ffn_norm, cfg.eps as f64);
    let mut out = vec![0.0f64; cfg.dim];
    match w.ffn {
        FfnWeights::Dense { gate, up, down } => {
            let y = expert(&h, gate, up, down, cfg.dim, cfg.feed_forward);
            for (o, value) in y.into_iter().enumerate() {
                out[o] += value;
            }
        }
        FfnWeights::Routed { router, bias, gate, up, down } => {
            let logits = matvec(&h, router, cfg.dim, cfg.n_experts);
            let scores: Vec<f64> = logits.iter().map(|z| 1.0 / (1.0 + (-z).exp())).collect();
            let mut ranked: Vec<(usize, f64)> = scores
                .iter()
                .enumerate()
                .map(|(e, &s)| (e, s + bias.map_or(0.0, |b| b[e] as f64)))
                .collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            let denominator = if cfg.norm_topk && cfg.n_experts_used > 1 {
                ranked.iter().take(cfg.n_experts_used).map(|(e, _)| scores[*e]).sum::<f64>() + 1e-20
            } else {
                1.0
            };
            for &(e, _) in ranked.iter().take(cfg.n_experts_used) {
                let gate_at = e * cfg.dim * cfg.expert_ff;
                let down_at = e * cfg.expert_ff * cfg.dim;
                let y = expert(
                    &h,
                    &gate[gate_at..gate_at + cfg.dim * cfg.expert_ff],
                    &up[gate_at..gate_at + cfg.dim * cfg.expert_ff],
                    &down[down_at..down_at + cfg.expert_ff * cfg.dim],
                    cfg.dim,
                    cfg.expert_ff,
                );
                for (o, value) in y.into_iter().enumerate() {
                    out[o] += (scores[e] / denominator) * value;
                }
            }
        }
    }
    for (residual, delta) in x.iter_mut().zip(&out) {
        *residual += delta;
    }
    Ok(())
}

fn rmsnorm(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    let mean: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv = 1.0 / (mean + eps).sqrt();
    x.iter().zip(w).map(|(v, g)| v * inv * *g as f64).collect()
}

fn matvec(x: &[f64], w: &[f32], n_in: usize, n_out: usize) -> Vec<f64> {
    (0..n_out)
        .map(|o| {
            let mut sum = 0.0;
            for i in 0..n_in {
                sum += x[i] * w[o * n_in + i] as f64;
            }
            sum
        })
        .collect()
}

/// `rotate_half` over the first `x.len()` dimensions: the first half supplies the real parts
/// and the second half the imaginary parts, which is Hugging Face's convention and not the
/// adjacent-pairs one.
fn rotate(x: &mut [f64], pos: usize, theta: f64) {
    let dim = x.len();
    let half = dim / 2;
    let original = x.to_vec();
    for i in 0..half {
        let freq = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
        let (cos, sin) = ((pos as f64 * freq).cos(), (pos as f64 * freq).sin());
        x[i] = original[i] * cos - original[i + half] * sin;
        x[i + half] = original[i + half] * cos + original[i] * sin;
    }
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = v.iter().map(|z| (z - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    exp.into_iter().map(|e| e / sum).collect()
}

fn expert(h: &[f64], gate: &[f32], up: &[f32], down: &[f32], dim: usize, inner: usize) -> Vec<f64> {
    let g = matvec(h, gate, dim, inner);
    let u = matvec(h, up, dim, inner);
    let activated: Vec<f64> = g
        .iter()
        .zip(&u)
        .map(|(gv, uv)| (gv / (1.0 + (-gv).exp())) * uv)
        .collect();
    matvec(&activated, down, inner, dim)
}
