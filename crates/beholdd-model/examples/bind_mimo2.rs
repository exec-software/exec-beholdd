//! Bind a real `mimo2` checkpoint's header and report what it declares.
//!
//! Gate G1 of ADR-0002. Header only: this reads what the checkpoint says it is and either
//! binds every hyperparameter or refuses one by name. Nothing is loaded, so the 156 GiB
//! MiMo-V2.6-Flash checkpoint costs a few milliseconds to check rather than seven minutes
//! to stripe across two machines — which is the whole reason binding is its own gate.
//!
//! usage: cargo run -p beholdd-model --release --example bind_mimo2 -- <file.gguf>
use anyhow::Result;
use beholdd_format::Gguf;
use beholdd_model::mimo2::{Attention, Config, Ffn};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: bind_mimo2 <file.gguf>");
        std::process::exit(2)
    });
    let t0 = std::time::Instant::now();
    let g = Gguf::open(&path)?;
    let c = Config::from_header(&g)?;

    println!("{path}");
    println!("  bound in {:.3}s -- header only\n", t0.elapsed().as_secs_f64());
    println!("  {} blocks, dim {}, vocab {}", c.n_layers(), c.dim, c.vocab);
    println!(
        "  attention: {} q heads, key {} / value {} per head, value scale {}",
        c.n_heads, c.key_head_dim, c.value_head_dim, c.value_scale
    );

    let windowed = c
        .blocks
        .iter()
        .filter(|b| b.attention == Attention::SlidingWindow)
        .count();
    println!(
        "  hybrid: {} sliding-window blocks (window {}) and {} full blocks",
        windowed,
        c.window,
        c.n_layers() - windowed
    );
    println!(
        "  rope: {} of {} dimensions rotated, base {:e} full / {:e} windowed",
        c.rope_dim, c.key_head_dim, c.rope_base, c.rope_base_swa
    );
    println!(
        "  experts: {} routed, {} used per token ({:.2}% active), inner width {}",
        c.n_experts,
        c.n_experts_used,
        c.activation() * 100.0,
        c.expert_ff
    );
    println!("  context: {}", c.context);

    // Two facts the metadata does not carry, and that a loader would otherwise assume:
    // which blocks route, and which carry a sink.
    let routed = c.blocks.iter().filter(|b| matches!(b.ffn, Ffn::Routed { .. })).count();
    println!(
        "  feed-forward: {} routed, {} dense (width {})",
        routed,
        c.n_layers() - routed,
        c.feed_forward
    );
    println!("  sinks: {} blocks carry one", c.blocks.iter().filter(|b| b.sink).count());

    // The per-layer shapes are the reason this is not a Qwen config with extra fields, so
    // print the first blocks rather than only the summary: a pattern read the wrong way
    // round would otherwise look identical in every aggregate above.
    println!("\n  first blocks:");
    for (index, block) in c.blocks.iter().take(6).enumerate() {
        println!(
            "    {index:>2}: {:<14} {} kv heads, group {:>2}, rope base {:e}, {}{}",
            match block.attention {
                Attention::Full => "full",
                Attention::SlidingWindow => "sliding-window",
            },
            block.n_kv_heads,
            c.group(index)?,
            c.rope_base_for(index)?,
            match block.ffn {
                Ffn::Dense => "dense ffn".to_string(),
                Ffn::Routed { selection_bias } => format!(
                    "routed{}",
                    if selection_bias { " (biased selection)" } else { "" }
                ),
            },
            if block.sink { ", sink" } else { "" }
        );
    }
    Ok(())
}
