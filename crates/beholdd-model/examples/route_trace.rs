//! Dump per-layer routed-expert selections across a real greedy decode.
//!
//! Diagnostic tool, not a gate: exists to answer one empirical question before any
//! route-prediction prefetch or popularity-weighted retention work starts --  does
//! Qwen3-MoE's router send nearby tokens to overlapping/predictable experts, or is
//! selection close to uniform-random? See docs/research/model-streaming.md.
//!
//! Always loads fully resident. The question here is routing behaviour, not streaming
//! performance, and resident execution removes I/O wait as a confound entirely.
//!
//! usage: cargo run -p beholdd-model --release --example route_trace -- <model.gguf> <new_tokens> <prompt text> <trace-out.tsv>

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{bail, Context, Result};
use beholdd_backend::cpu::Cpu;
use beholdd_format::Gguf;
use beholdd_memory::FullyResidentSource;
use beholdd_model::{Qwen2Tokenizer, Qwen3Moe};

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: route_trace <model.gguf> <new_tokens> <prompt text> <trace-out.tsv>";
    let model_path = args.next().context(usage)?;
    let max_new_tokens: usize = args
        .next()
        .context(usage)?
        .parse()
        .context("new_tokens must be a non-negative integer")?;
    let prompt_text = args.next().context(usage)?;
    let trace_path = args.next().context(usage)?;
    if args.next().is_some() {
        bail!("too many arguments; {usage}");
    }
    if Path::new(&trace_path).exists() {
        bail!("refusing to overwrite existing trace file {trace_path}");
    }
    if max_new_tokens == 0 {
        bail!("new_tokens must be at least 1; there is nothing to trace otherwise");
    }

    let checkpoint = Gguf::open(&model_path)?;
    let tokenizer = Qwen2Tokenizer::from_gguf(&checkpoint)?;
    let prompt = tokenizer.encode(&prompt_text, false)?;
    if prompt.is_empty() {
        bail!("prompt tokenized to an empty sequence");
    }
    let model = Qwen3Moe::load(&checkpoint)?;
    let device = Cpu::new();
    model.validate_device(&device)?;

    eprintln!("loading {model_path} fully resident...");
    let load_started = std::time::Instant::now();
    let weights = FullyResidentSource::load(&checkpoint)?;
    eprintln!(
        "loaded {} bytes in {:.1}s; decoding {max_new_tokens} tokens...",
        weights.resident_bytes(),
        load_started.elapsed().as_secs_f64()
    );

    let decode_started = std::time::Instant::now();
    let (output, route_trace) =
        model.greedy_decode_with_trace(&device, &weights, &prompt, max_new_tokens, false)?;
    let decode_elapsed = decode_started.elapsed().as_secs_f64();
    eprintln!(
        "decoded in {decode_elapsed:.1}s ({:.3}s/token)",
        decode_elapsed / max_new_tokens as f64
    );
    eprintln!("generated ids: {:?}", output.generated);
    eprintln!(
        "generated text: {:?}",
        tokenizer.decode(&output.generated, false)?
    );

    let file =
        File::create(&trace_path).with_context(|| format!("creating trace output {trace_path}"))?;
    let mut out = BufWriter::new(file);
    writeln!(out, "step\ttoken_id\tlayer\texperts\tnear_miss")?;
    for (step, step_routes) in route_trace.iter().enumerate() {
        // route_trace[0] is the last prompt position; route_trace[i>0] is generated[i-1] --
        // see Qwen3Moe::greedy_decode_with_trace.
        let token_id = if step == 0 {
            *prompt.last().expect("prompt was checked non-empty above")
        } else {
            output.generated[step - 1]
        };
        for route in step_routes {
            let experts: Vec<String> = route.experts.iter().map(usize::to_string).collect();
            let near_miss: Vec<String> = route
                .near_miss_experts
                .iter()
                .map(usize::to_string)
                .collect();
            writeln!(
                out,
                "{step}\t{token_id}\t{}\t{}\t{}",
                route.layer,
                experts.join(","),
                near_miss.join(",")
            )?;
        }
    }
    out.flush()?;
    eprintln!(
        "wrote {} steps of routing to {trace_path}",
        route_trace.len()
    );
    Ok(())
}
