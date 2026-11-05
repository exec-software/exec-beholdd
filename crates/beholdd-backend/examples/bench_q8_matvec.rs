//! Wall-time benchmark for the GGUF Q8_0 CPU matvec path at Qwen3-MoE shapes.
//!
//! Usage: `cargo run -p beholdd-backend --release --example bench_q8_matvec`.

use std::time::Instant;

use beholdd_backend::cpu::Cpu;
use beholdd_backend::{Device, Weight};
use beholdd_format::GgmlType;

const Q8_0_ELEMS: usize = 32;
const Q8_0_BYTES: usize = 34;

fn f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let man = ((bits >> 13) & 0x3ff) as u16;
    sign | ((exp as u16) << 10) | man
}

fn q8_0_weight(rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(cols % Q8_0_ELEMS, 0);
    let blocks = rows * cols / Q8_0_ELEMS;
    let mut bytes = Vec::with_capacity(blocks * Q8_0_BYTES);
    let scale = f16_bits(0.015625).to_le_bytes();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for _ in 0..blocks {
        bytes.extend_from_slice(&scale);
        for _ in 0..Q8_0_ELEMS {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            bytes.push(((state >> 56) as i8).clamp(-120, 120) as u8);
        }
    }
    bytes
}

fn bench_shape(cpu: &Cpu, label: &str, rows: usize, cols: usize, reps: usize) {
    let bytes = q8_0_weight(rows, cols);
    let weight = Weight {
        bytes: &bytes,
        kind: GgmlType(8),
        rows,
        cols,
    };
    let x: Vec<f32> = (0..cols)
        .map(|i| ((i as f32) * 0.017).sin() * 0.5)
        .collect();

    let mut dequantized = vec![0.0f32; rows * cols];
    let mut best_dequant = f64::INFINITY;
    for _ in 0..reps {
        let start = Instant::now();
        cpu.dequantize(&mut dequantized, &weight).unwrap();
        best_dequant = best_dequant.min(start.elapsed().as_secs_f64());
    }

    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mut threads: Vec<usize> = [1, 2, 4, 8]
        .into_iter()
        .filter(|&count| count <= rows)
        .collect();
    threads.push(available.min(rows).max(1));
    threads.sort_unstable();
    threads.dedup();

    println!("\n{label} Q8_0 [{rows} x {cols}], {available} logical CPUs, best of {reps}");
    println!("  {:<8} {:>10}", "threads", "full ms");
    let mut reference: Option<Vec<f32>> = None;
    for threads in threads {
        let mut y = vec![0.0f32; rows];
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let start = Instant::now();
            cpu.matvec_with_threads(&mut y, &x, &weight, threads)
                .unwrap();
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!("  {threads:<8} {:>10.3}", best * 1e3);
        match &reference {
            None => reference = Some(y),
            Some(reference) => {
                for (row, (actual, expected)) in y.iter().zip(reference).enumerate() {
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "{label}: row {row} differs at {threads} threads"
                    );
                }
            }
        }
    }
    println!("  dequant only: {:.3} ms", best_dequant * 1e3);
}

fn main() {
    let cpu = Cpu::new();
    bench_shape(&cpu, "router", 128, 2048, 7);
    bench_shape(&cpu, "expert gate/up", 768, 2048, 7);
    bench_shape(&cpu, "expert down", 2048, 768, 7);
    bench_shape(&cpu, "attention wq", 4096, 2048, 5);
    bench_shape(&cpu, "attention wo", 2048, 4096, 5);
}
