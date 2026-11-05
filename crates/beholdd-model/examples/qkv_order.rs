//! Does the fused projection really run query, then key, then value?
//!
//! Every gate so far has taken that order from `modeling_mimo_v2.py` and handed the same
//! reading to both sides of the oracle comparison, which cannot disagree with itself. This
//! asks the checkpoint instead.
//!
//! A trained projection's three parts are not scaled alike, and the row norms show it. So:
//! find the split of the tail that best separates it into two regimes, told nothing about
//! where to look, and see which order that split agrees with. `query | key | value` puts the
//! boundary at `q + k`; `query | value | key` puts it 256 rows earlier, at `q + v`.
//!
//! Not every block answers. Where a block's key and value rows happen to sit at the same
//! scale there is no boundary to find, and the separation score says so rather than picking
//! one at random.
//!
//! usage: cargo run -p beholdd-model --release --example qkv_order -- <file.gguf> [block]
//!        (with no block, every block is scanned and the verdicts are tallied)

use anyhow::{Context, Result};
use beholdd_format::Gguf;
use beholdd_kernel::kquant::q8_0;
use beholdd_model::mimo2::Config;

/// What one block's row-norm profile says about the order.
enum Verdict {
    /// The tail splits where `query | key | value` predicts.
    KeyFirst,
    /// The tail splits where `query | value | key` predicts.
    ValueFirst,
    /// It splits somewhere else, or nowhere: this block has nothing to say.
    Silent,
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().context("usage: qkv_order <file.gguf> [block]")?;
    let asked: Option<usize> = args.next().map(|a| a.parse()).transpose()?;

    let g = Gguf::open(&path)?;
    let cfg = Config::from_header(&g)?;
    let Some(layer) = asked else {
        let (mut key_first, mut value_first, mut silent) = (0, 0, 0);
        for layer in 0..cfg.n_layers() {
            match examine(&g, &cfg, layer, false)? {
                Verdict::KeyFirst => key_first += 1,
                Verdict::ValueFirst => value_first += 1,
                Verdict::Silent => silent += 1,
            }
        }
        println!(
            "\n  {} blocks say query|key|value, {value_first} say query|value|key, {silent} have no \
             boundary to find",
            key_first
        );
        return Ok(());
    };
    examine(&g, &cfg, layer, true)?;
    Ok(())
}

fn examine(g: &Gguf, cfg: &Config, layer: usize, verbose: bool) -> Result<Verdict> {
    let block = cfg.blocks[layer];
    let name = format!("blk.{layer}.attn_qkv.weight");
    let t = g.get(&name).context("no fused projection in this block")?;

    let count = usize::try_from(t.numel().context("no element count")?)?;
    let mut raw = vec![0u8; usize::try_from(t.nbytes())?];
    g.read_into(t, &mut raw)?;
    let mut w = vec![0.0f32; count];
    q8_0::dequant(&raw, &mut w);

    let rows = count / cfg.dim;
    let norms: Vec<f64> = (0..rows)
        .map(|r| {
            (w[r * cfg.dim..(r + 1) * cfg.dim].iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>()
                / cfg.dim as f64)
                .sqrt()
        })
        .collect();

    let q = cfg.n_heads * cfg.key_head_dim;
    let k = block.n_kv_heads * cfg.key_head_dim;
    let v = block.n_kv_heads * cfg.value_head_dim;
    if verbose {
        println!("{name}: {rows} rows over {} in", cfg.dim);
        println!("  head counts predict query {q} | key {k} | value {v}\n");
    }

    let mean = |range: std::ops::Range<usize>| -> f64 {
        norms[range.clone()].iter().sum::<f64>() / range.len() as f64
    };
    if verbose {
        for (label, range) in [("query", 0..q), ("key", q..q + k), ("value", q + k..q + k + v)] {
            let (start, end) = (range.start, range.end);
            println!("  {label:<6} rows {start:>6}..{end:<6} mean row norm {:.6}", mean(range));
        }
    }

    // Norms alone cannot order key against value in every block, but they can in some: the
    // three parts of a trained fused projection sit at different scales, and where they
    // differ enough the boundary is visible in the profile itself. Print it, then find the
    // split that separates the tail into two regions best, and see which order it agrees with.
    if verbose {
        println!("\n  row-norm profile over the last {} rows, in chunks of 64:", k + v);
        for chunk in ((rows - k - v)..rows).step_by(64) {
            let end = (chunk + 64).min(rows);
            print!(" {:.4}", mean(chunk..end));
            if (chunk - (rows - k - v)) % (64 * 8) == 64 * 7 {
                println!();
            }
        }
        println!();
    }

    // The best two-way split of the tail, by how much it separates the two halves' means
    // relative to the spread inside them. No boundary is assumed; the file is asked.
    let tail = rows - k - v;
    let spread = |range: std::ops::Range<usize>| -> f64 {
        let m = mean(range.clone());
        norms[range.clone()].iter().map(|value| (value - m).powi(2)).sum::<f64>() / range.len() as f64
    };
    let mut best = (0usize, f64::NEG_INFINITY);
    for at in tail + 64..rows - 64 {
        let within = spread(tail..at) * (at - tail) as f64 + spread(at..rows) * (rows - at) as f64;
        let score = (mean(tail..at) - mean(at..rows)).powi(2) / (within / (rows - tail) as f64).max(f64::MIN_POSITIVE);
        if score > best.1 {
            best = (at, score);
        }
    }
    // A split found in noise scores near one. Below this there is no regime change in the
    // tail at all, which happens when a block's key and value rows sit at the same scale.
    const AUDIBLE: f64 = 3.0;
    let verdict = match best {
        (at, score) if score >= AUDIBLE && at == q + k => Verdict::KeyFirst,
        (at, score) if score >= AUDIBLE && at == q + v => Verdict::ValueFirst,
        _ => Verdict::Silent,
    };
    let said = match verdict {
        Verdict::KeyFirst => "query, then key, then value",
        Verdict::ValueFirst => "query, then VALUE, then key -- every gate so far has it backwards",
        Verdict::Silent => "nothing; no boundary where either order predicts one",
    };
    if verbose {
        println!("\n  the tail splits best at row {} (separation {:.1})", best.0, best.1);
        println!("    query | key | value would put it at {}", q + k);
        println!("    query | value | key would put it at {}", q + v);
        println!("\n  block {layer} says: {said}");
    } else {
        println!("  block {layer:>2}: splits at {:>5} (separation {:>5.1}) — {said}", best.0, best.1);
    }
    Ok(verdict)
}
