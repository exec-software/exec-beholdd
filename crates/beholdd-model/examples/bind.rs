//! Bind a real dense Qwen3 or Qwen3-MoE checkpoint and report what was located.
//!
//! Every tensor is checked against the shape the metadata implies. A tensor that is
//! present but wrongly shaped is caught here rather than at the first matvec; a tensor
//! that is absent would otherwise read as zeros and still generate text.
//!
//! usage: cargo run -p beholdd-model --release --example bind -- <file.gguf>
use anyhow::Result;
use beholdd_format::Gguf;
use beholdd_model::{QwenArchitecture, QwenModel};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: bind <file.gguf>");
        std::process::exit(2)
    });
    let t0 = std::time::Instant::now();
    let g = Gguf::open(&path)?;
    let m = QwenModel::load(&g)?;
    let c = m.config();

    println!("{path}");
    println!(
        "  bound in {:.2}s -- every tensor shape-checked\n",
        t0.elapsed().as_secs_f64()
    );
    println!(
        "  {} layers, dim {}, vocab {}",
        c.n_layers(),
        c.dim(),
        c.vocab()
    );
    println!(
        "  attention: {} q heads / {} kv heads (group {}), head_dim {}",
        c.n_heads(),
        c.n_kv_heads(),
        c.group(),
        c.head_dim()
    );
    match m.architecture() {
        QwenArchitecture::Qwen3 => {
            println!("  dense SwiGLU FFN: {}", c.feed_forward_length());
        }
        QwenArchitecture::Qwen3Moe => println!(
            "  moe: {} experts, top-{}, expert ff {}",
            c.n_experts(),
            c.n_experts_used(),
            c.expert_ff()
        ),
    }
    println!(
        "  rope base {:.0}, rms eps {:e}, context {}",
        c.rope_base(),
        c.eps(),
        c.context()
    );

    let total: u64 = g.names().map(|n| g.get(n).unwrap().nbytes()).sum();
    println!("\n  weights {:.2} GB", total as f64 / 1e9);
    if m.architecture() == QwenArchitecture::Qwen3Moe {
        let exp = m.expert_bytes()?;
        println!(
            "  routed experts {:.2} GB ({:.1}%)",
            exp as f64 / 1e9,
            100.0 * exp as f64 / total as f64
        );
        let per_token = (exp as f64 / c.n_experts() as f64) * c.n_experts_used() as f64;
        println!(
            "  routed-expert bytes per token: {:.2} GB (top-{} of {})",
            per_token / 1e9,
            c.n_experts_used(),
            c.n_experts()
        );

        let (offset_a, stride) = m.gate_expert_slice(0, 0)?;
        let (offset_b, _) = m.gate_expert_slice(0, 1)?;
        assert_eq!(offset_b - offset_a, stride, "experts must be contiguous");
        println!("  expert stride {stride} bytes; experts are contiguous");
    }
    Ok(())
}
