//! Run one block of a real `mimo2` checkpoint, and check it against the f64 oracle.
//!
//! Gate G3 of ADR-0002. G2 proved the block's arithmetic against a reference on weights this
//! repository invented; the fixture could still have been wrong about the real tensors in
//! exactly the way the implementation is. This reads them.
//!
//! Two blocks are worth running, and they test different things:
//!
//! - **Block 0** is the model's only dense, full-attention block, so every tensor is `Q8_0`
//!   or `F32`. A wrong answer there is the block or the dequantiser, never the expert format.
//! - **Any other block** is windowed, sinked and routed over 256 MXFP4 experts. That is the
//!   real shape of 47 of the 48 blocks, and the only place the MXFP4 reader meets the file.
//!
//! A routed block cannot be materialised whole: 256 experts of 4096x2048, three tensors
//! each, is 25.8 GB of `f32`. Only the eight the router actually selects are read, which is
//! also what the shipping path will do — the difference between 3.1% of the weights and all
//! of them is the reason this architecture is worth streaming.
//!
//! usage: cargo run -p beholdd-model --release --example real_mimo2_block -- <file.gguf> [block]

use anyhow::{bail, Context, Result};
use beholdd_format::{mxfp4, Gguf};
use beholdd_kernel::kquant::q8_0;
use beholdd_model::mimo2::block::{self, Cache, FfnWeights, Weights};
use beholdd_model::mimo2::{oracle, Attention, Config, Ffn};

/// Read one whole tensor as `f32`, whatever it is stored as.
fn read(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.get(name).with_context(|| format!("{name} is not in this checkpoint"))?;
    let count = usize::try_from(t.numel().with_context(|| format!("{name} has no element count"))?)?;
    let mut raw = vec![0u8; usize::try_from(t.nbytes())?];
    g.read_into(t, &mut raw)?;

    let mut out = vec![0.0f32; count];
    match t.kind().name().as_str() {
        "F32" => {
            for (value, chunk) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        "Q8_0" => q8_0::dequant(&raw, &mut out),
        other => bail!("{name} is {other}; this gate reads F32 and Q8_0 whole"),
    }
    Ok(out)
}

/// Read one expert's plane out of a stacked MXFP4 tensor, without touching the other 255.
///
/// The stack is expert-major — `[n_in, rows, experts]` in GGUF's order — so an expert is a
/// contiguous run of `rows` rows, which is what makes selective reading possible at all.
fn read_expert(g: &Gguf, name: &str, expert: usize, n_in: usize, rows: usize) -> Result<Vec<f32>> {
    let t = g.get(name).with_context(|| format!("{name} is not in this checkpoint"))?;
    if t.kind().name() != "MXFP4" {
        bail!("{name} is {}; the experts of this checkpoint are MXFP4", t.kind().name());
    }
    let per_expert = rows * (n_in / mxfp4::BLOCK) * mxfp4::BLOCK_BYTES;
    let at = (expert * per_expert) as u64;
    let mut raw = vec![0u8; per_expert];
    g.read_tensor_slice_into(t, at..at + per_expert as u64, &mut raw)?;
    Ok(mxfp4::read(&raw, n_in, rows)?.dequant(n_in))
}

/// The same deterministic spread the synthetic fixture uses.
fn token(dim: usize) -> Vec<f32> {
    let mut state = 900u64.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    (0..dim)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
            ((unit - 0.5) * 0.2) as f32
        })
        .collect()
}

/// `rmsnorm(x, w)`. The one piece of the block the example recomputes, so it can ask the
/// router which experts to read before it reads any of them.
fn normed(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = (1.0 / (ss / x.len() as f64 + eps as f64).sqrt()) as f32;
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

struct Attn {
    norm: Vec<f32>,
    qkv: Vec<f32>,
    out: Vec<f32>,
    sinks: Vec<f32>,
    ffn_norm: Vec<f32>,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap_or_else(|| {
        eprintln!("usage: real_mimo2_block <file.gguf> [block]");
        std::process::exit(2)
    });
    let layer: usize = args.next().map_or(Ok(0), |a| a.parse())?;

    let g = Gguf::open(&path)?;
    let cfg = Config::from_header(&g)?;
    let block = *cfg
        .blocks
        .get(layer)
        .with_context(|| format!("block {layer} is past the {} in this model", cfg.n_layers()))?;
    println!("{path}");
    println!(
        "  block {layer}: {:?}, {} kv heads, {}, {}",
        block.attention,
        block.n_kv_heads,
        match block.ffn {
            Ffn::Dense => "dense".to_string(),
            Ffn::Routed { selection_bias } =>
                format!("routed{}", if selection_bias { " (biased)" } else { "" }),
        },
        if block.sink { "sink" } else { "no sink" }
    );

    let t0 = std::time::Instant::now();
    let attn = Attn {
        norm: read(&g, &format!("blk.{layer}.attn_norm.weight"))?,
        qkv: read(&g, &format!("blk.{layer}.attn_qkv.weight"))?,
        out: read(&g, &format!("blk.{layer}.attn_output.weight"))?,
        sinks: if block.sink {
            read(&g, &format!("blk.{layer}.attn_sinks.weight"))?
        } else {
            Vec::new()
        },
        ffn_norm: read(&g, &format!("blk.{layer}.ffn_norm.weight"))?,
    };
    println!("  attention read in {:.1}s", t0.elapsed().as_secs_f64());

    // The widths are the first thing invented weights could not check: `attn_qkv` is one
    // tensor in the file, and the head counts have to add up to it.
    let q = cfg.n_heads * cfg.key_head_dim;
    let k = block.n_kv_heads * cfg.key_head_dim;
    let v = block.n_kv_heads * cfg.value_head_dim;
    println!("  qkv {} = {q} query + {k} key + {v} value, over {} in", attn.qkv.len() / cfg.dim, cfg.dim);
    if attn.qkv.len() / cfg.dim != q + k + v {
        bail!("the fused projection is {} wide, the head counts say {}", attn.qkv.len() / cfg.dim, q + k + v);
    }

    let start = token(cfg.dim);
    // What the block is run against, and what it is checked with.
    let (cfg, router_rows, held) = match block.ffn {
        Ffn::Dense => {
            let gate = read(&g, &format!("blk.{layer}.ffn_gate.weight"))?;
            let up = read(&g, &format!("blk.{layer}.ffn_up.weight"))?;
            let down = read(&g, &format!("blk.{layer}.ffn_down.weight"))?;
            (cfg, None, (gate, up, down))
        }
        Ffn::Routed { selection_bias } => {
            // Which experts this token reaches has to be settled before they can be read, so
            // the attention runs once with the feed-forward switched off. Its output is the
            // real post-attention residual; nothing about the routing is approximated.
            let mut probe = cfg.clone();
            probe.blocks[layer].ffn = Ffn::Dense;
            probe.feed_forward = 1;
            let zero = vec![0.0f32; cfg.dim];
            let probe_weights = Weights {
                attn_norm: &attn.norm,
                qkv: &attn.qkv,
                attn_out: &attn.out,
                sinks: block.sink.then_some(attn.sinks.as_slice()),
                ffn_norm: &attn.ffn_norm,
                ffn: FfnWeights::Dense { gate: &zero, up: &zero, down: &zero },
            };
            let mut x = start.clone();
            block::forward(&probe, layer, &probe_weights, &mut x, &mut Cache::new())?;
            let h = normed(&x, &attn.ffn_norm, cfg.eps);

            let router = read(&g, &format!("blk.{layer}.ffn_gate_inp.weight"))?;
            let bias = if selection_bias {
                read(&g, &format!("blk.{layer}.exp_probs_b.bias"))?
            } else {
                vec![0.0; cfg.n_experts]
            };
            let mut ranked: Vec<(usize, f32, f32)> = (0..cfg.n_experts)
                .map(|e| {
                    let logit: f32 =
                        h.iter().zip(&router[e * cfg.dim..(e + 1) * cfg.dim]).map(|(a, b)| a * b).sum();
                    let score = 1.0 / (1.0 + (-logit).exp());
                    (e, score, score + bias[e])
                })
                .collect();
            let unbiased: Vec<usize> = {
                let mut by_score = ranked.clone();
                by_score.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                by_score.iter().take(cfg.n_experts_used).map(|(e, _, _)| *e).collect()
            };
            ranked.sort_by(|a, b| b.2.total_cmp(&a.2).then(a.0.cmp(&b.0)));
            let chosen: Vec<usize> = ranked.iter().take(cfg.n_experts_used).map(|(e, _, _)| *e).collect();
            println!("  router chose {chosen:?}");
            // How much the bias moved the choice. Nothing else in this run would notice if it
            // were being ignored, which is precisely why it is worth printing.
            println!(
                "  without the bias it would have chosen {unbiased:?} — {} of {} differ",
                chosen.iter().filter(|e| !unbiased.contains(e)).count(),
                cfg.n_experts_used
            );

            let t1 = std::time::Instant::now();
            let mut gate = Vec::new();
            let mut up = Vec::new();
            let mut down = Vec::new();
            for &e in &chosen {
                gate.extend(read_expert(&g, &format!("blk.{layer}.ffn_gate_exps.weight"), e, cfg.dim, cfg.expert_ff)?);
                up.extend(read_expert(&g, &format!("blk.{layer}.ffn_up_exps.weight"), e, cfg.dim, cfg.expert_ff)?);
                down.extend(read_expert(&g, &format!("blk.{layer}.ffn_down_exps.weight"), e, cfg.expert_ff, cfg.dim)?);
            }
            println!(
                "  {} experts ({:.2} GB of MXFP4 expanded) in {:.1}s",
                chosen.len(),
                (gate.len() + up.len() + down.len()) as f64 * 4.0 / 1e9,
                t1.elapsed().as_secs_f64()
            );

            // The chosen experts, renumbered, with the router rows that chose them. Taking
            // all eight of eight with their own sigmoid scores is the same arithmetic as
            // taking eight of 256: the scores are the same numbers, and the bias only ever
            // decided which eight, which has already happened.
            let mut compact = cfg.clone();
            compact.n_experts = chosen.len();
            compact.n_experts_used = chosen.len();
            compact.blocks[layer].ffn = Ffn::Routed { selection_bias: false };
            let rows: Vec<f32> =
                chosen.iter().flat_map(|&e| router[e * cfg.dim..(e + 1) * cfg.dim].to_vec()).collect();
            (compact, Some(rows), (gate, up, down))
        }
    };

    let (gate, up, down) = (&held.0, &held.1, &held.2);
    let w = Weights {
        attn_norm: &attn.norm,
        qkv: &attn.qkv,
        attn_out: &attn.out,
        sinks: block.sink.then_some(attn.sinks.as_slice()),
        ffn_norm: &attn.ffn_norm,
        ffn: match &router_rows {
            None => FfnWeights::Dense { gate, up, down },
            Some(router) => FfnWeights::Routed { router, bias: None, gate, up, down },
        },
    };

    let mut x = start.clone();
    let mut reference: Vec<f64> = start.iter().map(|value| *value as f64).collect();
    let t2 = std::time::Instant::now();
    block::forward(&cfg, layer, &w, &mut x, &mut Cache::new())?;
    let fast = t2.elapsed();
    let t3 = std::time::Instant::now();
    oracle::forward(&cfg, layer, &w, &mut reference, &mut oracle::Cache::new())?;
    println!("  block {:.3}s, oracle {:.3}s", fast.as_secs_f64(), t3.elapsed().as_secs_f64());

    let worst = x
        .iter()
        .zip(&reference)
        .map(|(got, want)| (*got as f64 - want).abs())
        .fold(0.0f64, f64::max);
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    println!("  worst disagreement {worst:e} against a residual reaching {scale:e}");

    if !x.iter().all(|value| value.is_finite()) {
        bail!("the block produced a non-finite residual");
    }
    // Relative, unlike G2's absolute gate: this residual is not order one, and a bound that
    // ignored its scale would either pass anything or fail on arithmetic that is fine.
    //
    // Measured at 2.4e-6 on block 0, which is what naive f32 accumulation over 4096 terms
    // should cost. The gate is twelve times that rather than four, because a bound this close
    // to the measurement would trip on a machine that sums in a different order and teach
    // everyone to ignore it. A real mistake here is orders of magnitude, not a factor of three.
    let relative = worst / scale.max(f64::MIN_POSITIVE);
    println!("  relative {relative:e}");
    if relative > 3e-5 {
        bail!("block and oracle disagree by {relative:e} of the residual's own scale");
    }
    let shape = match (block.attention, block.ffn) {
        (Attention::Full, Ffn::Dense) => "dense, full-attention",
        (Attention::Full, _) => "routed, full-attention",
        (_, Ffn::Dense) => "dense, windowed",
        _ => "routed, windowed",
    };
    println!("\n  G3 passed: the real {shape} block {layer} computes what the oracle says it should.");
    Ok(())
}
