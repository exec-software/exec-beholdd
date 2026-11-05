//! Correctness and wall-time gate for native BF16 on the prepared Vulkan prototype.
//!
//! Usage:
//! `cargo run -p beholdd-backend --release --features vulkan --example bench_vulkan_bf16`

#[cfg(feature = "vulkan")]
use std::time::{Duration, Instant};

#[cfg(feature = "vulkan")]
use beholdd_backend::cpu::Cpu;
#[cfg(feature = "vulkan")]
use beholdd_backend::vulkan::VulkanMatvec;
#[cfg(feature = "vulkan")]
use beholdd_backend::{Device, Weight};
#[cfg(feature = "vulkan")]
use beholdd_format::GgmlType;

#[cfg(feature = "vulkan")]
const CALLS_PER_SAMPLE: u32 = 8;

#[cfg(feature = "vulkan")]
fn bf16_weight(rows: usize, cols: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(rows * cols * 2);
    for index in 0..rows * cols {
        let value = ((index as f32) * 0.013).sin() * 0.4;
        bytes.extend_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
    bytes
}

#[cfg(feature = "vulkan")]
fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

#[cfg(feature = "vulkan")]
fn error_metrics(label: &str, expected: &[f32], actual: &[f32]) -> (f64, f32) {
    let difference_l2 = expected
        .iter()
        .zip(actual)
        .map(|(cpu, gpu)| (*cpu as f64 - *gpu as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let reference_l2 = expected
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let mut worst_absolute = 0.0f32;
    for (row, (&cpu, &gpu)) in expected.iter().zip(actual).enumerate() {
        let difference = (cpu - gpu).abs();
        worst_absolute = worst_absolute.max(difference);
        let allowed = 1e-5 + 1e-5 * cpu.abs();
        assert!(
            difference <= allowed,
            "{label}: row {row} differs by {difference:.3e}, exceeding {allowed:.3e} (CPU={cpu:.7e}, Vulkan={gpu:.7e})"
        );
    }
    (
        difference_l2 / reference_l2.max(f64::MIN_POSITIVE),
        worst_absolute,
    )
}

#[cfg(feature = "vulkan")]
fn benchmark_shape(
    cpu: &Cpu,
    vulkan: &VulkanMatvec,
    label: &str,
    rows: usize,
    cols: usize,
    repetitions: usize,
) {
    let bytes = bf16_weight(rows, cols);
    let weight = Weight {
        bytes: &bytes,
        kind: GgmlType(30),
        rows,
        cols,
    };
    let input: Vec<f32> = (0..cols)
        .map(|index| ((index as f32) * 0.017).cos() * 0.5)
        .collect();
    let mut expected = vec![0.0f32; rows];
    cpu.matvec(&mut expected, &input, &weight).unwrap();

    // Bracket this shape's GPU phase with CPU samples. Alternating every individual call
    // creates immediate APU power-state interference; a CPU phase on only one side lets
    // slower thermal/frequency drift choose the denominator.
    let mut cpu_before_samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let started = Instant::now();
        for _ in 0..CALLS_PER_SAMPLE {
            cpu.matvec(&mut expected, &input, &weight).unwrap();
        }
        cpu_before_samples.push(started.elapsed() / CALLS_PER_SAMPLE);
    }

    let upload_started = Instant::now();
    let prepared = vulkan.prepare(&weight).unwrap();
    let upload = upload_started.elapsed();
    let mut actual = vec![0.0f32; rows];
    vulkan.matvec(&mut actual, &input, &prepared).unwrap();
    let (relative_l2, worst_absolute) = error_metrics(label, &expected, &actual);
    assert!(
        relative_l2 < 1e-5,
        "{label}: Vulkan relative L2 {relative_l2:.3e} exceeds 1e-5"
    );

    // Exercise the overwrite-vs-accumulate contract separately from the steady-state
    // timing. The initial output is nonzero so a shader that overwrites by accident is
    // caught immediately.
    let mut cpu_added = vec![0.25f32; rows];
    let mut gpu_added = cpu_added.clone();
    cpu.matvec_add(&mut cpu_added, &input, &weight, 0.375)
        .unwrap();
    vulkan
        .matvec_add(&mut gpu_added, &input, &prepared, 0.375)
        .unwrap();
    let (add_difference, _) = error_metrics(label, &cpu_added, &gpu_added);
    assert!(add_difference < 1e-5);

    let mut gpu_samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let started = Instant::now();
        for _ in 0..CALLS_PER_SAMPLE {
            vulkan.matvec(&mut actual, &input, &prepared).unwrap();
        }
        gpu_samples.push(started.elapsed() / CALLS_PER_SAMPLE);
    }
    let mut cpu_after_samples = Vec::with_capacity(repetitions);
    for _ in 0..repetitions {
        let started = Instant::now();
        for _ in 0..CALLS_PER_SAMPLE {
            cpu.matvec(&mut expected, &input, &weight).unwrap();
        }
        cpu_after_samples.push(started.elapsed() / CALLS_PER_SAMPLE);
    }
    let gpu = median(gpu_samples);
    let cpu_before = median(cpu_before_samples);
    let cpu_after = median(cpu_after_samples);
    let cpu_best = cpu_before.min(cpu_after);
    let bandwidth = bytes.len() as f64 / gpu.as_secs_f64() / (1024.0 * 1024.0 * 1024.0);
    println!(
        "{label:<16} [{rows:>6} x {cols:<5}] upload={:>10.6} ms  \
         steady_gpu={:>10.6} ms  cpu_best={:>10.6} ms  cpu_pre/post={:>10.6}/{:>10.6} ms  conservative_speedup={:>5.2}x  gpu={bandwidth:>6.1} GiB/s  rel_l2={relative_l2:.3e} max_abs={worst_absolute:.3e}",
        upload.as_secs_f64() * 1e3,
        gpu.as_secs_f64() * 1e3,
        cpu_best.as_secs_f64() * 1e3,
        cpu_before.as_secs_f64() * 1e3,
        cpu_after.as_secs_f64() * 1e3,
        cpu_best.as_secs_f64() / gpu.as_secs_f64(),
    );
}

#[cfg(feature = "vulkan")]
fn main() {
    let cpu = Cpu::new();
    let vulkan = VulkanMatvec::new().expect("initialize the Vulkan matvec prototype");
    println!("Vulkan device: {}", vulkan.physical_device_name());
    println!("Native BF16 storage with f32 accumulation; one-time upload is reported separately.");
    println!("Medians contain {CALLS_PER_SAMPLE} calls per timed sample; CPU phases bracket GPU.");
    println!("Shapes follow the Qwen3.6-35B-A3B text configuration (batch-1 decode).");
    benchmark_shape(&cpu, &vulkan, "router", 256, 2048, 21);
    benchmark_shape(&cpu, &vulkan, "expert gate/up", 512, 2048, 21);
    benchmark_shape(&cpu, &vulkan, "expert down", 2048, 512, 21);
    benchmark_shape(&cpu, &vulkan, "attention wq", 4096, 2048, 15);
    benchmark_shape(&cpu, &vulkan, "attention wo", 2048, 4096, 15);
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("re-run with `--features vulkan`");
    std::process::exit(2);
}
