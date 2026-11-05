//! Wall-time benchmark for the GGUF BF16 CPU matvec path at Qwen3-MoE shapes.
//!
//! This exists to keep the native-weight claim honest. A BF16 checkpoint should be read
//! and multiplied in BF16 storage; expanding an entire matrix to f32 before every
//! matvec adds a four-byte write and four-byte read per parameter, plus an equally sized
//! scratch allocation.
//!
//! Usage: `cargo run -p beholdd-backend --release --example bench_bf16_matvec`.

use std::time::Instant;

use beholdd_backend::cpu::Cpu;
use beholdd_backend::{Device, Weight};
use beholdd_format::GgmlType;

fn bf16_weight(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    for index in 0..rows * cols {
        let value = ((index as f32) * 0.013).sin() * 0.4;
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    bytes
}

fn bench_shape(cpu: &Cpu, label: &str, rows: usize, cols: usize, reps: usize) {
    let bytes = bf16_weight(rows, cols);
    let weight = Weight {
        bytes: &bytes,
        kind: GgmlType(30),
        rows,
        cols,
    };
    let x: Vec<f32> = (0..cols)
        .map(|index| ((index as f32) * 0.017).cos() * 0.5)
        .collect();
    let mut y = vec![0.0f32; rows];

    // Warm the exact production path once before timing it.
    cpu.matvec(&mut y, &x, &weight).unwrap();
    let mut best = f64::INFINITY;
    for _ in 0..reps {
        let started = Instant::now();
        cpu.matvec(&mut y, &x, &weight).unwrap();
        best = best.min(started.elapsed().as_secs_f64());
    }

    let encoded_mib = bytes.len() as f64 / (1024.0 * 1024.0);
    let gib_per_second = bytes.len() as f64 / best / (1024.0 * 1024.0 * 1024.0);
    let checksum: f64 = y.iter().map(|value| *value as f64).sum();
    println!(
        "{label:<16} [{rows:>6} x {cols:<5}] {encoded_mib:>7.2} MiB  \
         {ms:>9.3} ms  {gib_per_second:>7.2} GiB/s  checksum={checksum:.9e}",
        ms = best * 1e3,
    );
}

fn main() {
    let cpu = Cpu::new();
    let available = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    println!("BF16 production matvec path, {available} logical CPUs, best wall time");
    bench_shape(&cpu, "router", 128, 2048, 9);
    bench_shape(&cpu, "expert gate/up", 768, 2048, 9);
    bench_shape(&cpu, "expert down", 2048, 768, 9);
    bench_shape(&cpu, "attention wq", 4096, 2048, 7);
    bench_shape(&cpu, "attention wo", 2048, 4096, 7);
}
