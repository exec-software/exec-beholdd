use anyhow::Result;
use beholdd_format::Gguf;
fn main() -> Result<()> {
    let g = Gguf::open(std::env::args().nth(1).unwrap())?;
    let mut blk0: Vec<&str> = g.names().filter(|n| n.starts_with("blk.0.")).collect();
    blk0.sort();
    println!("-- blk.0 --");
    for n in &blk0 {
        let t = g.get(n).unwrap();
        println!("  {:<34} {:<6} {:?}", n, t.kind().name(), t.dims());
    }
    println!("-- non-blk --");
    let mut top: Vec<&str> = g.names().filter(|n| !n.starts_with("blk.")).collect();
    top.sort();
    for n in &top {
        let t = g.get(n).unwrap();
        println!("  {:<34} {:<6} {:?}", n, t.kind().name(), t.dims());
    }
    println!("-- selected metadata --");
    for (k, v) in g.metadata() {
        let s = format!("{v:?}");
        if s.len() < 120 && !k.contains("tokenizer.ggml.tok") && !k.contains("merges") {
            println!("  {k} = {s}");
        }
    }
    Ok(())
}
