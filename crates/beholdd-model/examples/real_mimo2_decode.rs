//! Decode text with a real `mimo2` checkpoint, one block at a time.
//!
//! Gate G4 of ADR-0002. G3 showed that two blocks compute what the oracle says they should.
//! This stacks all 48 of them on a real prompt and reads what comes out, which is the first
//! thing in this series that can be wrong in a way the arithmetic gates cannot see: a
//! transposed projection or a misread expert produces perfectly finite numbers and gibberish
//! text, and only text says which.
//!
//! **Nothing is resident.** 155.87 GiB does not fit on this machine, so a block's weights are
//! read, used and dropped, and only the 8 experts of 256 that a token actually routes to are
//! ever expanded — 3.1% of the model. Layers run outermost and positions innermost so a
//! block is read once per pass rather than once per token.
//!
//! usage: cargo run -p beholdd-model --release --example real_mimo2_decode -- \
//!          <file.gguf> ["prompt"] [tokens]

use std::collections::HashMap;

use anyhow::{bail, Context, Result};
use beholdd_format::{mxfp4, Gguf};
use beholdd_kernel::kquant::q8_0;
use beholdd_model::mimo2::block::{self, Cache, FfnWeights, Weights};
use beholdd_model::mimo2::{oracle, Config, Ffn};
use beholdd_model::Qwen2Tokenizer;

/// Read a whole tensor as `f32`.
fn read(g: &Gguf, name: &str) -> Result<Vec<f32>> {
    let t = g.get(name).with_context(|| format!("{name} is not in this checkpoint"))?;
    let count = usize::try_from(t.numel().context("no element count")?)?;
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
        other => bail!("{name} is {other}; this reads F32 and Q8_0"),
    }
    Ok(out)
}

/// Read one row of a `Q8_0` matrix without touching the rest of it.
///
/// The vocabulary is 152576 rows of 4096. Materialising it to look up six tokens would cost
/// 2.5 GB to read 100 KB.
fn read_q8_row(g: &Gguf, name: &str, row: usize, n_in: usize) -> Result<Vec<f32>> {
    let t = g.get(name).with_context(|| format!("{name} is not in this checkpoint"))?;
    let bytes = n_in / q8_0::ELEMS * q8_0::BYTES;
    let at = (row * bytes) as u64;
    let mut raw = vec![0u8; bytes];
    g.read_tensor_slice_into(t, at..at + bytes as u64, &mut raw)?;
    let mut out = vec![0.0f32; n_in];
    q8_0::dequant(&raw, &mut out);
    Ok(out)
}

fn read_expert(g: &Gguf, name: &str, expert: usize, n_in: usize, rows: usize) -> Result<Vec<f32>> {
    let t = g.get(name).with_context(|| format!("{name} is not in this checkpoint"))?;
    let per = rows * (n_in / mxfp4::BLOCK) * mxfp4::BLOCK_BYTES;
    let at = (expert * per) as u64;
    let mut raw = vec![0u8; per];
    g.read_tensor_slice_into(t, at..at + per as u64, &mut raw)?;
    Ok(mxfp4::read(&raw, n_in, rows)?.dequant(n_in))
}

fn normed(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = (1.0 / (ss / x.len() as f64 + eps as f64).sqrt()) as f32;
    x.iter().zip(w).map(|(v, g)| v * inv * g).collect()
}

/// One block's attention weights, read and dropped with the block.
struct Attn {
    norm: Vec<f32>,
    qkv: Vec<f32>,
    out: Vec<f32>,
    sinks: Vec<f32>,
    ffn_norm: Vec<f32>,
}

fn attention_of(g: &Gguf, layer: usize, sink: bool) -> Result<Attn> {
    Ok(Attn {
        norm: read(g, &format!("blk.{layer}.attn_norm.weight"))?,
        qkv: read(g, &format!("blk.{layer}.attn_qkv.weight"))?,
        out: read(g, &format!("blk.{layer}.attn_output.weight"))?,
        sinks: if sink { read(g, &format!("blk.{layer}.attn_sinks.weight"))? } else { Vec::new() },
        ffn_norm: read(g, &format!("blk.{layer}.ffn_norm.weight"))?,
    })
}

/// One expert's three matrices, expanded.
type Expert = (Vec<f32>, Vec<f32>, Vec<f32>);

/// The experts of one layer, expanded on demand and kept only while that layer is open.
struct Experts<'a> {
    g: &'a Gguf,
    layer: usize,
    cfg: &'a Config,
    held: HashMap<usize, Expert>,
}

impl Experts<'_> {
    fn get(&mut self, expert: usize) -> Result<&Expert> {
        if !self.held.contains_key(&expert) {
            let (layer, dim, ff) = (self.layer, self.cfg.dim, self.cfg.expert_ff);
            let gate = read_expert(self.g, &format!("blk.{layer}.ffn_gate_exps.weight"), expert, dim, ff)?;
            let up = read_expert(self.g, &format!("blk.{layer}.ffn_up_exps.weight"), expert, dim, ff)?;
            let down = read_expert(self.g, &format!("blk.{layer}.ffn_down_exps.weight"), expert, ff, dim)?;
            self.held.insert(expert, (gate, up, down));
        }
        Ok(&self.held[&expert])
    }
}

/// Run every position of `xs` through one block, reading it once.
#[allow(clippy::too_many_arguments)]
fn run_layer(
    g: &Gguf,
    cfg: &Config,
    layer: usize,
    xs: &mut [Vec<f32>],
    caches: &mut [Cache],
    from: usize,
) -> Result<()> {
    let block = cfg.blocks[layer];
    let attn = attention_of(g, layer, block.sink)?;

    match block.ffn {
        Ffn::Dense => {
            let gate = read(g, &format!("blk.{layer}.ffn_gate.weight"))?;
            let up = read(g, &format!("blk.{layer}.ffn_up.weight"))?;
            let down = read(g, &format!("blk.{layer}.ffn_down.weight"))?;
            let w = Weights {
                attn_norm: &attn.norm,
                qkv: &attn.qkv,
                attn_out: &attn.out,
                sinks: block.sink.then_some(attn.sinks.as_slice()),
                ffn_norm: &attn.ffn_norm,
                ffn: FfnWeights::Dense { gate: &gate, up: &up, down: &down },
            };
            for x in xs.iter_mut().skip(from) {
                block::forward(cfg, layer, &w, x, &mut caches[layer])?;
            }
        }
        Ffn::Routed { selection_bias } => {
            let router = read(g, &format!("blk.{layer}.ffn_gate_inp.weight"))?;
            let bias = if selection_bias {
                read(g, &format!("blk.{layer}.exp_probs_b.bias"))?
            } else {
                vec![0.0; cfg.n_experts]
            };
            let mut experts = Experts { g, layer, cfg, held: HashMap::new() };

            #[allow(clippy::needless_range_loop)]
            for index in from..xs.len() {
                // Attention first with the feed-forward switched off, so the router can be
                // asked which experts this position needs before any of them is read. The
                // residual this leaves is the real post-attention one.
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
                let mut probed = xs[index].clone();
                let mut probe_cache = caches[layer].clone();
                block::forward(&probe, layer, &probe_weights, &mut probed, &mut probe_cache)?;
                let h = normed(&probed, &attn.ffn_norm, cfg.eps);

                let mut ranked: Vec<(usize, f32)> = (0..cfg.n_experts)
                    .map(|e| {
                        let logit: f32 =
                            h.iter().zip(&router[e * cfg.dim..(e + 1) * cfg.dim]).map(|(a, b)| a * b).sum();
                        (e, 1.0 / (1.0 + (-logit).exp()) + bias[e])
                    })
                    .collect();
                ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
                let chosen: Vec<usize> =
                    ranked.iter().take(cfg.n_experts_used).map(|(e, _)| *e).collect();

                let mut gate = Vec::new();
                let mut up = Vec::new();
                let mut down = Vec::new();
                for &e in &chosen {
                    let held = experts.get(e)?;
                    gate.extend_from_slice(&held.0);
                    up.extend_from_slice(&held.1);
                    down.extend_from_slice(&held.2);
                }
                let mut compact = cfg.clone();
                compact.n_experts = chosen.len();
                compact.n_experts_used = chosen.len();
                compact.blocks[layer].ffn = Ffn::Routed { selection_bias: false };
                let rows: Vec<f32> =
                    chosen.iter().flat_map(|&e| router[e * cfg.dim..(e + 1) * cfg.dim].to_vec()).collect();
                let w = Weights {
                    attn_norm: &attn.norm,
                    qkv: &attn.qkv,
                    attn_out: &attn.out,
                    sinks: block.sink.then_some(attn.sinks.as_slice()),
                    ffn_norm: &attn.ffn_norm,
                    ffn: FfnWeights::Routed { router: &rows, bias: None, gate: &gate, up: &up, down: &down },
                };
                block::forward(&compact, layer, &w, &mut xs[index], &mut caches[layer])?;
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: real_mimo2_decode <file.gguf> [prompt] [tokens]")?;
    let prompt = args.next().unwrap_or_else(|| "The capital of France is".to_string());
    let want: usize = args.next().map_or(Ok(4), |a| a.parse())?;

    let g = Gguf::open(&path)?;
    let cfg = Config::from_header(&g)?;
    let tokenizer = Qwen2Tokenizer::from_gguf(&g)?;
    let mut ids = tokenizer.encode(&prompt, false)?;
    println!("{path}");
    println!("  {} blocks, {} experts, {} active per token", cfg.n_layers(), cfg.n_experts, cfg.n_experts_used);
    println!("  prompt {prompt:?} -> {} tokens {ids:?}\n", ids.len());

    let output_norm = read(&g, "output_norm.weight")?;
    let output = g.get("output.weight").context("no output projection")?;
    let row_bytes = cfg.dim / q8_0::ELEMS * q8_0::BYTES;

    let mut caches: Vec<Cache> = (0..cfg.n_layers()).map(|_| Cache::new()).collect();
    let mut xs: Vec<Vec<f32>> = Vec::new();
    let mut done = 0usize;
    let started = std::time::Instant::now();

    for step in 0..=want {
        // Embed whatever is not embedded yet: the whole prompt on the first pass, then one
        // token per pass.
        for &id in &ids[xs.len()..] {
            xs.push(read_q8_row(&g, "token_embd.weight", id as usize, cfg.dim)?);
        }
        let pass = std::time::Instant::now();
        for layer in 0..cfg.n_layers() {
            run_layer(&g, &cfg, layer, &mut xs, &mut caches, done)?;
        }
        done = xs.len();

        let last = normed(xs.last().expect("at least one position"), &output_norm, cfg.eps);
        let mut raw = vec![0u8; row_bytes];
        let mut logits = vec![0.0f32; cfg.vocab];
        for (token, logit) in logits.iter_mut().enumerate() {
            let at = (token * row_bytes) as u64;
            g.read_tensor_slice_into(output, at..at + row_bytes as u64, &mut raw)?;
            *logit = q8_0::dot(&raw, &last);
        }
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if !top.is_finite() {
            bail!("the logits are not finite; the stack has diverged");
        }
        // Reported as log-probabilities, because that is what another implementation reports
        // and a raw logit is not comparable across two of them.
        let total: f64 = logits.iter().map(|z| ((z - top) as f64).exp()).sum();
        let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        println!("  step {step} in {:.1}s — top 8:", pass.elapsed().as_secs_f64());
        for &(token, logit) in ranked.iter().take(8) {
            let piece = tokenizer.decode(&[token as u32], false).unwrap_or_default();
            let logprob = (logit - top) as f64 - total.ln();
            println!("    {piece:>12?}  id {token:>6}  logprob {logprob:>8.4}  logit {logit:>8.3}");
        }
        let best = ranked[0].0 as u32;
        ids.push(best);
    }

    let text = tokenizer.decode(&ids, false)?;
    println!("\n  {:.0}s total\n\n  {text:?}", started.elapsed().as_secs_f64());
    let _ = oracle::Cache::new();
    Ok(())
}
