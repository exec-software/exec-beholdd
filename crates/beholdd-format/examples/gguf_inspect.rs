//! Open a real GGUF and report architecture, shapes and tensor types.
//!
//! usage: cargo run -p beholdd-format --example gguf_inspect -- <file.gguf>
use std::collections::BTreeMap;

use anyhow::Result;
use beholdd_format::Gguf;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: gguf_inspect <file.gguf>");
        std::process::exit(2)
    });
    let t0 = std::time::Instant::now();
    let g = Gguf::open(&path)?;
    println!(
        "{}\n  arch {:?}  {} shards  {} tensors  {} metadata keys  ({:.2}s)",
        path,
        g.arch().unwrap_or("?"),
        g.shards().len(),
        g.len(),
        g.metadata().len(),
        t0.elapsed().as_secs_f64()
    );

    // Architecture-namespaced fields, resolved through general.architecture.
    print!("  ");
    for k in [
        "block_count",
        "embedding_length",
        "attention.head_count",
        "attention.head_count_kv",
        "expert_count",
        "expert_used_count",
        "context_length",
    ] {
        if let Some(v) = g.kv(k).and_then(|v| v.as_u64()) {
            print!("{k}={v}  ");
        }
    }
    println!();

    let mut by_type: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for n in g.names() {
        let t = g.get(n).unwrap();
        let e = by_type.entry(t.kind().name()).or_default();
        e.0 += 1;
        e.1 += t.nbytes();
    }
    println!("\n  {:<10} {:>8} {:>12}", "ggml type", "tensors", "GB");
    for (k, (n, b)) in &by_type {
        println!("  {:<10} {:>8} {:>12.2}", k, n, *b as f64 / 1e9);
    }
    let tot: u64 = by_type.values().map(|v| v.1).sum();
    println!(
        "  {:<10} {:>8} {:>12.2}",
        "TOTAL",
        g.len(),
        tot as f64 / 1e9
    );

    // Spot-check that a real tensor reads back.
    if let Some(name) = g.names().find(|n| n.contains("blk.0.")).map(str::to_string) {
        let t = g.get(&name).unwrap();
        let mut probe = vec![0u8; t.nbytes() as usize];
        g.read_into(t, &mut probe)?;
        let buf = &probe[..probe.len().min(64)];
        println!(
            "\n  {name}\n    {} {:?}  {} bytes @ {}\n    first 16: {:02x?}",
            t.kind().name(),
            t.dims(),
            t.nbytes(),
            t.offset(),
            &buf[..buf.len().min(16)]
        );
    }
    Ok(())
}
