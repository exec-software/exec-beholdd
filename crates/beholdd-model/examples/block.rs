//! Run one Qwen3-MoE block on real checkpoint bytes and emit the Gate M1.1 interchange.
//!
//! The output layout intentionally matches `tools/qwen3moe_ref.py`: three little-endian
//! i32 dimensions, token IDs, selected experts, routing weights, then hidden states.
//!
//! usage: cargo run -p beholdd-model --release --example block -- <model.gguf> <out.bin> [n_positions]

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use beholdd_backend::cpu::Cpu;
use beholdd_format::Gguf;
use beholdd_memory::ResidentSource;
use beholdd_model::{BlockOutput, Qwen3Moe};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let model_path = args
        .next()
        .context("usage: block <model.gguf> <out.bin> [n_positions]")?;
    let output_path = args
        .next()
        .context("usage: block <model.gguf> <out.bin> [n_positions]")?;
    let n_positions: usize = args
        .next()
        .as_deref()
        .unwrap_or("4")
        .parse()
        .context("n_positions must be a positive integer")?;
    if n_positions == 0 {
        bail!("n_positions must be positive");
    }
    if args.next().is_some() {
        bail!("usage: block <model.gguf> <out.bin> [n_positions]");
    }

    let started = Instant::now();
    let checkpoint = Gguf::open(&model_path)?;
    let model = Qwen3Moe::load(&checkpoint)?;
    let weights = ResidentSource::new(&checkpoint);
    let tokens: Vec<u32> = (0..n_positions)
        .map(|position| ((position as u64 * 7919 + 13) % 100_000) as u32)
        .collect();
    let result = model.forward_block(&Cpu::new(), &weights, 0, &tokens)?;
    write_interchange(&output_path, &result)?;

    println!("{model_path}");
    println!(
        "  qwen3moe block 0 on CPU in {:.2}s",
        started.elapsed().as_secs_f64()
    );
    for position in 0..result.n_positions {
        let routes = position * result.top_k..(position + 1) * result.top_k;
        let hidden = &result.hidden[position * result.dim..(position + 1) * result.dim];
        let rms = (hidden
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            / result.dim as f64)
            .sqrt();
        println!(
            "    pos {position} tok {:>6}  experts {:?}...  out rms {rms:.6}",
            result.tokens[position],
            &result.experts[routes][..4.min(result.top_k)]
        );
    }
    println!("  wrote {output_path}");
    Ok(())
}

fn write_interchange(path: &str, result: &BlockOutput) -> Result<()> {
    let mut out = BufWriter::new(File::create(path).with_context(|| format!("creating {path}"))?);
    for (label, value) in [
        ("n_positions", result.n_positions),
        ("dim", result.dim),
        ("top_k", result.top_k),
    ] {
        out.write_all(&as_i32(value, label)?.to_le_bytes())?;
    }
    for &token in &result.tokens {
        let token = i32::try_from(token).context("token ID does not fit i32 interchange")?;
        out.write_all(&token.to_le_bytes())?;
    }
    for &expert in &result.experts {
        out.write_all(&as_i32(expert, "expert index")?.to_le_bytes())?;
    }
    for &weight in &result.expert_weights {
        out.write_all(&weight.to_le_bytes())?;
    }
    for &value in &result.hidden {
        out.write_all(&value.to_le_bytes())?;
    }
    out.flush()?;
    Ok(())
}

fn as_i32(value: usize, label: &str) -> Result<i32> {
    i32::try_from(value).with_context(|| format!("{label} {value} does not fit i32 interchange"))
}
