//! Print a checkpoint's metadata, or the tensors whose names contain a filter.
//!
//! `gguf_inspect` summarises a file and `names` shows a representative block; this answers
//! questions about a *particular* block, which is how the mimo2 binder learned that block 0
//! of MiMo-V2.6-Flash is dense while the other 47 route. Both modes read the header only.
//!
//! usage: gguf_grep <file.gguf> [name-filter]
use anyhow::Result;
use beholdd_format::Gguf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: gguf_grep <file.gguf> [filter]");
    let filter = args.next();
    let g = Gguf::open(&path)?;
    match filter {
        None => {
            for (k, v) in g.metadata() {
                if k.starts_with("tokenizer.") {
                    continue;
                }
                let shown = format!("{v:?}");
                println!("{k} = {}", if shown.len() > 120 { format!("{}…", &shown[..120]) } else { shown });
            }
        }
        Some(f) => {
            let names: Vec<String> = g.names().map(str::to_string).collect();
            for name in names {
                if !name.contains(&f) {
                    continue;
                }
                if let Some(t) = g.get(&name) {
                    println!("{:<40} {:<8} {:?}", name, t.kind().name(), t.dims());
                }
            }
        }
    }
    Ok(())
}
