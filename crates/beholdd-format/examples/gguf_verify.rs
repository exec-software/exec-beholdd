//! Byte-accounting check: does our block geometry agree with a real checkpoint?
//!
//! For every tensor we claim to decode, `numel / ELEMS * BYTES` must equal the byte
//! length the file actually reserves. If a block-size constant is wrong that identity
//! fails, for every tensor, in every file -- and there is no way to satisfy it across
//! thousands of tensors of differing shapes by accident.
//!
//! This is the closest thing to an oracle available without running llama.cpp: the
//! checkpoint itself is asserting how many bytes a tensor of that shape and type takes.
//!
//! usage: cargo run -p beholdd-format --example gguf_verify -- <file.gguf>
use std::collections::BTreeMap;

use anyhow::Result;
use beholdd_core::ggml_block_geometry;
use beholdd_format::Gguf;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: gguf_verify <file.gguf>");
        std::process::exit(2)
    });
    let g = Gguf::open(&path)?;
    println!("{}  arch {:?}", path, g.arch().unwrap_or("?"));

    let mut ok: BTreeMap<String, usize> = BTreeMap::new();
    let mut bad: Vec<String> = Vec::new();
    let mut unsupported: BTreeMap<String, usize> = BTreeMap::new();

    for name in g.names() {
        let t = g.get(name).unwrap();
        let ty = t.kind().name();
        match ggml_block_geometry(t.kind().0) {
            None => *unsupported.entry(ty).or_default() += 1,
            Some((elems, bytes)) => {
                let Some(n) = t.numel() else {
                    bad.push(format!("{name}: element count overflows"));
                    continue;
                };
                if n % elems as u64 != 0 {
                    bad.push(format!(
                        "{name}: {n} elements is not whole blocks of {elems}"
                    ));
                    continue;
                }
                let want = n / elems as u64 * bytes as u64;
                if t.nbytes() == want {
                    *ok.entry(ty).or_default() += 1;
                } else {
                    bad.push(format!(
                        "{name}: {ty} {:?} -> want {want} bytes, file says {}",
                        t.dims(),
                        t.nbytes()
                    ));
                }
            }
        }
    }

    println!("\n  verified:");
    for (ty, n) in &ok {
        println!("    {ty:<22} {n:>5} tensors");
    }
    if !unsupported.is_empty() {
        println!("  not yet decodable:");
        for (ty, n) in &unsupported {
            println!("    {ty:<22} {n:>5} tensors");
        }
    }
    if bad.is_empty() {
        println!("\n  BYTE ACCOUNTING AGREES on every decodable tensor");
    } else {
        println!("\n  {} MISMATCHES:", bad.len());
        for b in bad.iter().take(8) {
            println!("    {b}");
        }
        std::process::exit(1);
    }
    Ok(())
}
