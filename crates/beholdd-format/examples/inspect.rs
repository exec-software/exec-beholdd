//! Gate M0.3 -- open a real checkpoint and report what is in it.
//!
//! Fixtures prove the parser handles what we thought to write down. This proves it
//! handles what a checkpoint actually contains, which is a different claim.
//!
//! usage: cargo run -p beholdd-format --example inspect -- <dir> [tensor-name]
use std::collections::BTreeMap;

use anyhow::Result;
use beholdd_format::SafeTensors;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args.next().unwrap_or_else(|| {
        eprintln!("usage: inspect <checkpoint-dir> [tensor]");
        std::process::exit(2)
    });

    let t0 = std::time::Instant::now();
    let st = SafeTensors::open(&dir)?;
    println!(
        "{}\n  {} shards, {} tensors, indexed in {:.2}s",
        dir,
        st.shards().len(),
        st.len(),
        t0.elapsed().as_secs_f64()
    );

    let mut by_dtype: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for n in st.names() {
        let t = st.get(n).unwrap();
        let e = by_dtype.entry(format!("{:?}", t.dtype())).or_default();
        e.0 += 1;
        e.1 += t.nbytes();
    }
    println!("\n  {:<10} {:>8}  {:>12}", "dtype", "tensors", "GB");
    for (d, (n, bytes)) in &by_dtype {
        println!("  {:<10} {:>8}  {:>12.2}", d, n, *bytes as f64 / 1e9);
    }
    let total: u64 = by_dtype.values().map(|v| v.1).sum();
    println!(
        "  {:<10} {:>8}  {:>12.2}",
        "TOTAL",
        st.len(),
        total as f64 / 1e9
    );

    // Reproduce the expert-geometry finding through this reader: on DeepSeek-V4 an
    // expert's three weight blobs are one contiguous run and its three scale blobs are
    // another, somewhere else entirely -- so a fetch is TWO preads, not six. Checked per
    // expert rather than assumed, because a revision that interleaved them differently
    // would read the wrong bytes at full speed.
    for (layer, expert) in [(0, 147), (2, 30), (20, 5), (42, 200)] {
        let w: Vec<String> = (1..=3)
            .map(|i| format!("layers.{layer}.ffn.experts.{expert}.w{i}.weight"))
            .collect();
        let s: Vec<String> = (1..=3)
            .map(|i| format!("layers.{layer}.ffn.experts.{expert}.w{i}.scale"))
            .collect();
        let wr: Vec<&str> = w.iter().map(String::as_str).collect();
        let sr: Vec<&str> = s.iter().map(String::as_str).collect();
        if st.get(wr[0]).is_none() {
            break;
        }
        match (st.contiguous_run(&wr), st.contiguous_run(&sr)) {
            (Some(a), Some(b)) => println!(
                "  L{layer:<3} E{expert:<4} weights {:>11} B contiguous @{:<12} | scales {:>9} B contiguous  -> 2 preads",
                a.2, a.1, b.2
            ),
            (aw, sc) => println!(
                "  L{layer:<3} E{expert:<4} NOT contiguous (weights {:?}, scales {:?}) -> 6 preads",
                aw.is_some(), sc.is_some()
            ),
        }
    }

    if let Some(name) = args.next() {
        match st.get(&name) {
            Some(t) => {
                println!("\n  {name}\n    {:?} {:?}", t.dtype(), t.shape());
                println!(
                    "    shard {} offset {} nbytes {}",
                    t.shard(),
                    t.offset(),
                    t.nbytes()
                );
                let bytes = st.read(t)?;
                println!("    first 16 bytes: {:02x?}", &bytes[..bytes.len().min(16)]);
            }
            None => println!("\n  {name}: NOT FOUND"),
        }
    }
    Ok(())
}
