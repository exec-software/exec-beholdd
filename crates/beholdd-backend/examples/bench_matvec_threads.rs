//! Thread-count vs. wall time for the CPU Q4_K matvec path, on this codebase's own
//! kernel and shapes -- not the 9.5x/24.8x/22% numbers in README.md's "Parallel-region
//! granularity beats thread count" section, which describe a prior, non-Rust engine and
//! predate any kernel code in this repository (see `cpu.rs`'s "# Threading" module doc).
//!
//! Q4_K is the format the real, already-gated Qwen3-Coder-480B-A35B checkpoint actually
//! uses (ROADMAP.md Gate M2.6), so that is what this measures rather than Q8_0 or MXFP4.
//!
//! usage: cargo run -p beholdd-backend --release --example bench_matvec_threads
use std::time::Instant;

use beholdd_backend::cpu::Cpu;
use beholdd_backend::{Device, Weight};
use beholdd_format::GgmlType;

const Q4_K_ELEMS: usize = 256;
const Q4_K_BYTES: usize = 144;

fn fill(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u8
        })
        .collect()
}

fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let man = ((bits >> 13) & 0x3FF) as u16;
    sign | ((exp as u16) << 10) | man
}

/// One deterministic Q4_K weight matrix: fixed, finite `d`/`dmin` per super-block (so no
/// random byte pattern can decode to NaN/Inf -- `scale_min_k4` only ever extracts bounded
/// 6-bit fields from `scales`/`qs` via masking, never reinterprets them as floats),
/// random scales/quant nibbles otherwise. Layout confirmed directly against
/// `beholdd_kernel::kquant::q4_k`: d(2) + dmin(2) + scales(12) + qs(128) = 144 bytes per
/// 256-element super-block.
fn q4_k_weight(rows: usize, cols: usize, seed: u64) -> Vec<u8> {
    assert_eq!(
        cols % Q4_K_ELEMS,
        0,
        "cols must be a whole number of Q4_K super-blocks"
    );
    let n_blocks = rows * (cols / Q4_K_ELEMS);
    let d = f16_bits(0.02).to_le_bytes();
    let dmin = f16_bits(0.01).to_le_bytes();
    let body = fill(n_blocks * 12, seed ^ 0x5CA1);
    let quants = fill(n_blocks * 128, seed ^ 0xBEE5);
    let mut bytes = Vec::with_capacity(n_blocks * Q4_K_BYTES);
    for b in 0..n_blocks {
        bytes.extend_from_slice(&d);
        bytes.extend_from_slice(&dmin);
        bytes.extend_from_slice(&body[b * 12..(b + 1) * 12]);
        bytes.extend_from_slice(&quants[b * 128..(b + 1) * 128]);
    }
    bytes
}

fn bench_shape(cpu: &Cpu, label: &str, rows: usize, cols: usize, reps: usize) {
    let bytes = q4_k_weight(rows, cols, 0xC0FFEE);
    let w = Weight {
        bytes: &bytes,
        kind: GgmlType(12),
        rows,
        cols,
    };
    let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();

    let mut tmp = vec![0.0f32; rows * cols];
    let mut best_dequant = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        cpu.dequantize(&mut tmp, &w).unwrap();
        best_dequant = best_dequant.min(t.elapsed().as_secs_f64());
    }

    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut threads: Vec<usize> = [1, 2, 4, 8].into_iter().filter(|&t| t <= rows).collect();
    threads.push(available.min(rows).max(1));
    threads.sort_unstable();
    threads.dedup();

    let flop = 2.0 * rows as f64 * cols as f64;
    let mut reference: Option<Vec<f32>> = None;
    let mut best_at_one = f64::INFINITY;

    println!("\n{label} Q4_K matvec [{rows} x {cols}], {available} logical CPUs, best of {reps}");
    println!(
        "  {:<8} {:>9} {:>10} {:>14}",
        "threads", "ms", "GFLOP/s", "dot-only ms*"
    );
    for &t in &threads {
        let mut y = vec![0.0f32; rows];
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let start = Instant::now();
            cpu.matvec_with_threads(&mut y, &x, &w, t).unwrap();
            best = best.min(start.elapsed().as_secs_f64());
        }
        if t == 1 {
            best_at_one = best;
        }
        println!(
            "  {:<8} {:>9.2} {:>10.1} {:>14.2}",
            t,
            best * 1e3,
            flop / best / 1e9,
            (best - best_dequant).max(0.0) * 1e3
        );
        match &reference {
            None => reference = Some(y),
            Some(r) => {
                for (row, (a, b)) in r.iter().zip(&y).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{label}: row {row} differs at {t} threads vs 1 -- NOT bit-identical"
                    );
                }
            }
        }
    }
    println!(
        "  dequant alone: {:.2} ms ({:.1}% of the 1-thread full call)",
        best_dequant * 1e3,
        100.0 * best_dequant / best_at_one
    );
    println!("  * dot-only ms is derived (full - dequant), not independently timed");
    println!("  every thread count above reproduced the 1-thread output bit-for-bit.");
}

fn main() {
    let cpu = Cpu::new();
    bench_shape(&cpu, "router-scale", 128, 2048, 7);
    bench_shape(&cpu, "wq/wo-scale", 4096, 2048, 7);
}
