//! Correctness and wall-time gate for the arena-backed fused BF16 routed-MoE prototype.
//!
//! Usage:
//! `cargo run -p beholdd-backend --release --features vulkan --example bench_vulkan_moe`

#[cfg(feature = "vulkan")]
use std::num::NonZeroUsize;
#[cfg(feature = "vulkan")]
use std::time::{Duration, Instant};

#[cfg(feature = "vulkan")]
use beholdd_backend::cpu::Cpu;
#[cfg(feature = "vulkan")]
use beholdd_backend::vulkan::{
    Bf16Expert, MatrixId, MatrixSpec, PreparedMatrix, PreparedRoutedExperts,
    RoutedMoeBatchDispatch, RoutedMoeFastVariant, RoutedMoeProgramConfig, VulkanMatvec,
    WeightArenaBuilder, WeightArenaConfig,
};
#[cfg(feature = "vulkan")]
use beholdd_backend::{Device, DeviceError, Weight};
#[cfg(feature = "vulkan")]
use beholdd_format::GgmlType;

#[cfg(feature = "vulkan")]
const HIDDEN: usize = 2048;
#[cfg(feature = "vulkan")]
const EXPERT_WIDTH: usize = 512;
#[cfg(feature = "vulkan")]
const ROUTES: usize = 8;
#[cfg(feature = "vulkan")]
const BF16_KIND: GgmlType = GgmlType(30);
#[cfg(feature = "vulkan")]
const MIB: usize = 1024 * 1024;
#[cfg(feature = "vulkan")]
const ROUTED_PROGRAM_BUDGET_BYTES: u64 = (4 * MIB) as u64;
#[cfg(feature = "vulkan")]
const WARMUPS: usize = 2;
#[cfg(feature = "vulkan")]
const SAMPLES: usize = 12;
#[cfg(feature = "vulkan")]
const PROFILE_WARMUPS: usize = 2;
#[cfg(feature = "vulkan")]
const PROGRAM_WARMUPS: usize = 4;
#[cfg(feature = "vulkan")]
const PROGRAM_SAMPLES: usize = 24;
#[cfg(feature = "vulkan")]
const BATCH_WARMUPS: usize = 4;
#[cfg(feature = "vulkan")]
const BATCH_SAMPLES: usize = 12;
#[cfg(feature = "vulkan")]
const BATCH_BURST_EVALS: usize = 16;
#[cfg(feature = "vulkan")]
const PROGRAM_BATCH_SIZES: [usize; 5] = [1, 2, 4, 8, 16];
#[cfg(feature = "vulkan")]
const BATCH_GO_GATE: f64 = 1.10;
#[cfg(feature = "vulkan")]
const ITEM_DIMENSION_GO_GATE: f64 = 1.10;
#[cfg(feature = "vulkan")]
const TILE4_GO_GATE: f64 = 1.15;
#[cfg(feature = "vulkan")]
const FAST_VARIANTS: [RoutedMoeFastVariant; 6] = [
    RoutedMoeFastVariant::Tree32,
    RoutedMoeFastVariant::QualifiedTree64,
    RoutedMoeFastVariant::Tree128,
    RoutedMoeFastVariant::Subgroup32,
    RoutedMoeFastVariant::Subgroup64,
    RoutedMoeFastVariant::Subgroup128,
];
#[cfg(feature = "vulkan")]
const WILLIAMS_BASE_ORDER: [usize; 6] = [0, 1, 5, 2, 4, 3];
#[cfg(feature = "vulkan")]
const EXPERT_IDS: [u32; ROUTES] = [231, 3, 128, 17, 255, 0, 72, 9];
#[cfg(feature = "vulkan")]
const ROUTE_WEIGHTS: [f32; ROUTES] = [0.31, 0.19, 0.14, 0.11, 0.09, 0.07, 0.05, 0.04];

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy)]
enum Projection {
    Gate,
    Up,
    Down,
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy)]
enum TimingPath {
    Legacy,
    Guarded,
    Fast,
}

#[cfg(feature = "vulkan")]
const TIMING_ORDERS: [[TimingPath; 3]; 6] = [
    [TimingPath::Legacy, TimingPath::Guarded, TimingPath::Fast],
    [TimingPath::Legacy, TimingPath::Fast, TimingPath::Guarded],
    [TimingPath::Guarded, TimingPath::Legacy, TimingPath::Fast],
    [TimingPath::Guarded, TimingPath::Fast, TimingPath::Legacy],
    [TimingPath::Fast, TimingPath::Legacy, TimingPath::Guarded],
    [TimingPath::Fast, TimingPath::Guarded, TimingPath::Legacy],
];

#[cfg(feature = "vulkan")]
impl Projection {
    const fn tag(self) -> u64 {
        match self {
            Self::Gate => 0x6761_7465,
            Self::Up => 0x7570,
            Self::Down => 0x646f_776e,
        }
    }
}

#[cfg(feature = "vulkan")]
#[derive(Debug, Clone, Copy)]
struct ErrorMetrics {
    relative_l2: f64,
    max_absolute: f32,
}

#[cfg(feature = "vulkan")]
struct BatchProgramTiming {
    batch_len: usize,
    per_item_prepare: Duration,
    item_dimensional_prepare: Duration,
    tile4_prepare: Duration,
    immediate: Duration,
    per_item: Duration,
    item_dimensional: Duration,
    tile4: Duration,
    per_item_over_immediate: f64,
    item_dimensional_over_per_item: f64,
    item_dimensional_saved_us: f64,
    tile4_over_per_item: f64,
    tile4_saved_us: f64,
    tile4_tiles: usize,
}

#[cfg(feature = "vulkan")]
#[derive(Clone, Copy)]
enum BatchTimingPath {
    Immediate,
    PerItem,
    ItemDimension,
    Tile4,
}

#[cfg(feature = "vulkan")]
const BATCH_TIMING_ORDERS: [[BatchTimingPath; 4]; 4] = [
    [
        BatchTimingPath::Immediate,
        BatchTimingPath::PerItem,
        BatchTimingPath::Tile4,
        BatchTimingPath::ItemDimension,
    ],
    [
        BatchTimingPath::PerItem,
        BatchTimingPath::ItemDimension,
        BatchTimingPath::Immediate,
        BatchTimingPath::Tile4,
    ],
    [
        BatchTimingPath::ItemDimension,
        BatchTimingPath::Tile4,
        BatchTimingPath::PerItem,
        BatchTimingPath::Immediate,
    ],
    [
        BatchTimingPath::Tile4,
        BatchTimingPath::Immediate,
        BatchTimingPath::ItemDimension,
        BatchTimingPath::PerItem,
    ],
];

#[cfg(feature = "vulkan")]
fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[cfg(feature = "vulkan")]
fn matrix_seed(expert_id: u32, projection: Projection) -> u64 {
    splitmix64(u64::from(expert_id) ^ projection.tag().rotate_left(29))
}

#[cfg(feature = "vulkan")]
fn fill_bf16(destination: &mut [u8], seed: u64) {
    assert!(destination.len().is_multiple_of(2));
    for (index, encoded) in destination.chunks_exact_mut(2).enumerate() {
        let mixed = splitmix64(seed ^ (index as u64).wrapping_mul(0xd6e8_feb8_6659_fd93));
        let signed = (mixed >> 48) as u16 as i16;
        let value = f32::from(signed) * (0.025 / 32768.0);
        encoded.copy_from_slice(&((value.to_bits() >> 16) as u16).to_le_bytes());
    }
}

#[cfg(feature = "vulkan")]
fn spec(projection: Projection) -> MatrixSpec {
    let (rows, cols) = match projection {
        Projection::Gate | Projection::Up => (EXPERT_WIDTH, HIDDEN),
        Projection::Down => (HIDDEN, EXPERT_WIDTH),
    };
    MatrixSpec {
        kind: BF16_KIND,
        rows,
        cols,
    }
}

#[cfg(feature = "vulkan")]
fn push_generated(
    builder: &mut WeightArenaBuilder,
    expert_id: u32,
    projection: Projection,
) -> Result<MatrixId, DeviceError> {
    let seed = matrix_seed(expert_id, projection);
    builder.push_with(spec(projection), |destination| {
        fill_bf16(destination, seed);
        Ok(())
    })
}

#[cfg(feature = "vulkan")]
fn generated_cpu_matvec(
    cpu: &Cpu,
    scratch_bytes: &mut [u8],
    y: &mut [f32],
    x: &[f32],
    expert_id: u32,
    projection: Projection,
    add_scale: Option<f32>,
) -> Result<(), DeviceError> {
    let matrix = spec(projection);
    let encoded_bytes = matrix.rows * matrix.cols * 2;
    let bytes = &mut scratch_bytes[..encoded_bytes];
    fill_bf16(bytes, matrix_seed(expert_id, projection));
    let weight = Weight {
        bytes,
        kind: matrix.kind,
        rows: matrix.rows,
        cols: matrix.cols,
    };
    if let Some(scale) = add_scale {
        cpu.matvec_add(y, x, &weight, scale)
    } else {
        cpu.matvec(y, x, &weight)
    }
}

#[cfg(feature = "vulkan")]
fn cpu_oracle(cpu: &Cpu, input: &[f32]) -> Result<Vec<f32>, DeviceError> {
    // Every projection has 1,048,576 BF16 values. Reuse one 2 MiB byte buffer rather
    // than materialising either 48 MiB of host weights or any f32 weight matrix.
    let mut encoded = vec![0u8; HIDDEN * EXPERT_WIDTH * 2];
    let mut gate = vec![0.0f32; EXPERT_WIDTH];
    let mut up = vec![0.0f32; EXPERT_WIDTH];
    let mut output = vec![0.0f32; HIDDEN];
    for ((&expert_id, &route_weight), route) in EXPERT_IDS.iter().zip(&ROUTE_WEIGHTS).zip(0..ROUTES)
    {
        generated_cpu_matvec(
            cpu,
            &mut encoded,
            &mut gate,
            input,
            expert_id,
            Projection::Gate,
            None,
        )?;
        generated_cpu_matvec(
            cpu,
            &mut encoded,
            &mut up,
            input,
            expert_id,
            Projection::Up,
            None,
        )?;
        cpu.swiglu(&mut gate, &up, 0.0)?;
        generated_cpu_matvec(
            cpu,
            &mut encoded,
            &mut output,
            &gate,
            expert_id,
            Projection::Down,
            Some(route_weight),
        )?;
        debug_assert_eq!(EXPERT_IDS[route], expert_id);
    }
    Ok(output)
}

#[cfg(feature = "vulkan")]
fn legacy_routed_moe(
    cpu: &Cpu,
    vulkan: &VulkanMatvec,
    output: &mut [f32],
    input: &[f32],
    matrices: &[[PreparedMatrix; 3]],
    gate: &mut [f32],
    up: &mut [f32],
) -> Result<(), DeviceError> {
    output.fill(0.0);
    for (route, prepared) in matrices.iter().enumerate() {
        vulkan.matvec(gate, input, &prepared[0])?;
        vulkan.matvec(up, input, &prepared[1])?;
        cpu.swiglu(gate, up, 0.0)?;
        vulkan.matvec_add(output, gate, &prepared[2], ROUTE_WEIGHTS[route])?;
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn error_metrics(expected: &[f32], actual: &[f32]) -> ErrorMetrics {
    assert_eq!(expected.len(), actual.len());
    let mut difference_squared = 0.0f64;
    let mut reference_squared = 0.0f64;
    let mut max_absolute = 0.0f32;
    for (&reference, &observed) in expected.iter().zip(actual) {
        let difference = reference - observed;
        difference_squared += f64::from(difference).powi(2);
        reference_squared += f64::from(reference).powi(2);
        max_absolute = max_absolute.max(difference.abs());
    }
    ErrorMetrics {
        relative_l2: difference_squared.sqrt() / reference_squared.sqrt().max(f64::MIN_POSITIVE),
        max_absolute,
    }
}

#[cfg(feature = "vulkan")]
fn assert_correct(label: &str, expected: &[f32], actual: &[f32]) -> ErrorMetrics {
    let metrics = error_metrics(expected, actual);
    assert!(
        metrics.relative_l2 < 2e-4 && metrics.max_absolute < 2e-4,
        "{label}: rel-L2 {:.3e}, max-abs {:.3e} exceeds the BF16 gate",
        metrics.relative_l2,
        metrics.max_absolute
    );
    metrics
}

#[cfg(feature = "vulkan")]
fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) / 2
    } else {
        samples[middle]
    }
}

#[cfg(feature = "vulkan")]
fn median_f64(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    let middle = samples.len() / 2;
    if samples.len().is_multiple_of(2) {
        (samples[middle - 1] + samples[middle]) * 0.5
    } else {
        samples[middle]
    }
}

#[cfg(feature = "vulkan")]
fn batch_burst_iterations(batch_len: usize) -> usize {
    BATCH_BURST_EVALS.div_ceil(batch_len)
}

#[cfg(feature = "vulkan")]
fn normalized_burst_duration(started: Instant, iterations: usize) -> Duration {
    Duration::from_secs_f64(started.elapsed().as_secs_f64() / iterations as f64)
}

#[cfg(feature = "vulkan")]
fn run_immediate_batch(
    vulkan: &VulkanMatvec,
    ys: &mut [f32],
    xs: &[f32],
    experts: &[PreparedRoutedExperts],
    route_weights: &[f32],
    accumulate: bool,
) -> Result<(), DeviceError> {
    debug_assert_eq!(ys.len(), experts.len() * HIDDEN);
    debug_assert_eq!(xs.len(), experts.len() * HIDDEN);
    debug_assert_eq!(route_weights.len(), experts.len() * ROUTES);
    for (((y, x), expert), weights) in ys
        .chunks_exact_mut(HIDDEN)
        .zip(xs.chunks_exact(HIDDEN))
        .zip(experts)
        .zip(route_weights.chunks_exact(ROUTES))
    {
        vulkan.routed_moe_bf16_fast(y, x, expert, weights, accumulate)?;
    }
    Ok(())
}

#[cfg(feature = "vulkan")]
fn benchmark_program_batch(
    vulkan: &VulkanMatvec,
    input: &[f32],
    experts: &[PreparedRoutedExperts],
) -> Result<BatchProgramTiming, DeviceError> {
    let batch_len = experts.len();
    let batch_len_nonzero = NonZeroUsize::new(batch_len).expect("batch sizes are non-zero");
    let template = &experts[0];
    let mut inputs = Vec::with_capacity(batch_len * HIDDEN);
    let mut weights = Vec::with_capacity(batch_len * ROUTES);
    for item in 0..batch_len {
        inputs.extend(input.iter().enumerate().map(|(index, value)| {
            let wobble = (((index + item * 19) as f32) * 0.011).sin() * 0.0005;
            value * (1.0 - item as f32 * 0.002) + wobble
        }));
        weights.extend((0..ROUTES).map(|route| ROUTE_WEIGHTS[(route + item) % ROUTES]));
    }

    let per_item_prepare_started = Instant::now();
    let per_item = vulkan.prepare_routed_moe_bf16_fast_batch_program(
        template,
        batch_len_nonzero,
        RoutedMoeProgramConfig::new(ROUTED_PROGRAM_BUDGET_BYTES),
    )?;
    let per_item_prepare = per_item_prepare_started.elapsed();
    let item_dimensional_prepare_started = Instant::now();
    let item_dimensional = vulkan.prepare_routed_moe_bf16_fast_batch_program_with_dispatch(
        template,
        batch_len_nonzero,
        RoutedMoeProgramConfig::new(ROUTED_PROGRAM_BUDGET_BYTES),
        RoutedMoeBatchDispatch::ItemDimension,
    )?;
    let item_dimensional_prepare = item_dimensional_prepare_started.elapsed();
    let tile4_prepare_started = Instant::now();
    let tile4 = vulkan.prepare_routed_moe_bf16_fast_batch_program_with_dispatch(
        template,
        batch_len_nonzero,
        RoutedMoeProgramConfig::new(ROUTED_PROGRAM_BUDGET_BYTES),
        RoutedMoeBatchDispatch::ExpertTile4,
    )?;
    let tile4_prepare = tile4_prepare_started.elapsed();
    assert_eq!(per_item.batch_len(), batch_len);
    assert_eq!(item_dimensional.batch_len(), batch_len);
    assert_eq!(tile4.batch_len(), batch_len);
    assert_eq!(
        per_item.fast_variant(),
        vulkan.selected_routed_moe_fast_variant(template)
    );
    assert_eq!(per_item.batch_dispatch(), RoutedMoeBatchDispatch::PerItem);
    assert_eq!(
        item_dimensional.batch_dispatch(),
        RoutedMoeBatchDispatch::ItemDimension
    );
    assert_eq!(
        per_item.fast_variant(),
        RoutedMoeFastVariant::Tree128,
        "the item-dimensional qualification control must use the same Tree128 arithmetic"
    );
    assert_eq!(
        item_dimensional.fast_variant(),
        RoutedMoeFastVariant::Tree128
    );
    assert_eq!(tile4.batch_dispatch(), RoutedMoeBatchDispatch::ExpertTile4);
    assert_eq!(tile4.fast_variant(), RoutedMoeFastVariant::Tree128);

    let mut immediate_output = vec![0.0f32; batch_len * HIDDEN];
    run_immediate_batch(
        vulkan,
        &mut immediate_output,
        &inputs,
        experts,
        &weights,
        false,
    )?;
    let expected_bits: Vec<_> = immediate_output
        .iter()
        .map(|value| value.to_bits())
        .collect();
    let mut per_item_output = vec![83.0f32; batch_len * HIDDEN];
    per_item.execute_batch(&mut per_item_output, &inputs, experts, &weights, false)?;
    assert_eq!(
        expected_bits,
        per_item_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} per-item program changed independent overwrite results"
    );
    let mut item_dimensional_output = vec![-83.0f32; batch_len * HIDDEN];
    item_dimensional.execute_batch(
        &mut item_dimensional_output,
        &inputs,
        experts,
        &weights,
        false,
    )?;
    assert_eq!(
        expected_bits,
        item_dimensional_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} item-dimensional program changed independent overwrite results"
    );
    item_dimensional.execute_batch(
        &mut item_dimensional_output,
        &inputs,
        experts,
        &weights,
        false,
    )?;
    assert_eq!(
        expected_bits,
        item_dimensional_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} item-dimensional program was not bit-repeatable"
    );
    let mut tile4_output = vec![127.0f32; batch_len * HIDDEN];
    tile4.execute_batch(&mut tile4_output, &inputs, experts, &weights, false)?;
    assert_eq!(
        expected_bits,
        tile4_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} expert-major Tile4 program changed independent overwrite results"
    );
    tile4.execute_batch(&mut tile4_output, &inputs, experts, &weights, false)?;
    assert_eq!(
        expected_bits,
        tile4_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} expert-major Tile4 program was not bit-repeatable"
    );

    let initial: Vec<f32> = (0..batch_len * HIDDEN)
        .map(|index| 0.0625 + (index % 29) as f32 * 0.000_125)
        .collect();
    let mut immediate_accumulated = initial.clone();
    run_immediate_batch(
        vulkan,
        &mut immediate_accumulated,
        &inputs,
        experts,
        &weights,
        true,
    )?;
    let mut per_item_accumulated = initial.clone();
    per_item.execute_batch(&mut per_item_accumulated, &inputs, experts, &weights, true)?;
    assert_eq!(
        immediate_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        per_item_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} per-item program changed independent accumulation results"
    );
    let mut item_dimensional_accumulated = initial;
    item_dimensional.execute_batch(
        &mut item_dimensional_accumulated,
        &inputs,
        experts,
        &weights,
        true,
    )?;
    assert_eq!(
        immediate_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        item_dimensional_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} item-dimensional program changed independent accumulation results"
    );
    let mut tile4_accumulated = (0..batch_len * HIDDEN)
        .map(|index| 0.0625 + (index % 29) as f32 * 0.000_125)
        .collect::<Vec<_>>();
    tile4.execute_batch(&mut tile4_accumulated, &inputs, experts, &weights, true)?;
    assert_eq!(
        immediate_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        tile4_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} expert-major Tile4 program changed independent accumulation results"
    );

    for warmup in 0..BATCH_WARMUPS {
        for path in BATCH_TIMING_ORDERS[warmup % BATCH_TIMING_ORDERS.len()] {
            match path {
                BatchTimingPath::Immediate => run_immediate_batch(
                    vulkan,
                    &mut immediate_output,
                    &inputs,
                    experts,
                    &weights,
                    false,
                )?,
                BatchTimingPath::PerItem => per_item.execute_batch(
                    &mut per_item_output,
                    &inputs,
                    experts,
                    &weights,
                    false,
                )?,
                BatchTimingPath::ItemDimension => item_dimensional.execute_batch(
                    &mut item_dimensional_output,
                    &inputs,
                    experts,
                    &weights,
                    false,
                )?,
                BatchTimingPath::Tile4 => {
                    tile4.execute_batch(&mut tile4_output, &inputs, experts, &weights, false)?
                }
            }
        }
    }
    let mut immediate_samples = Vec::with_capacity(BATCH_SAMPLES);
    let mut per_item_samples = Vec::with_capacity(BATCH_SAMPLES);
    let mut item_dimensional_samples = Vec::with_capacity(BATCH_SAMPLES);
    let mut tile4_samples = Vec::with_capacity(BATCH_SAMPLES);
    let burst_iterations = batch_burst_iterations(batch_len);
    for sample in 0..BATCH_SAMPLES {
        for path in BATCH_TIMING_ORDERS[sample % BATCH_TIMING_ORDERS.len()] {
            let started = Instant::now();
            match path {
                BatchTimingPath::Immediate => {
                    for _ in 0..burst_iterations {
                        run_immediate_batch(
                            vulkan,
                            &mut immediate_output,
                            &inputs,
                            experts,
                            &weights,
                            false,
                        )?;
                    }
                    immediate_samples.push(normalized_burst_duration(started, burst_iterations));
                }
                BatchTimingPath::PerItem => {
                    for _ in 0..burst_iterations {
                        per_item.execute_batch(
                            &mut per_item_output,
                            &inputs,
                            experts,
                            &weights,
                            false,
                        )?;
                    }
                    per_item_samples.push(normalized_burst_duration(started, burst_iterations));
                }
                BatchTimingPath::ItemDimension => {
                    for _ in 0..burst_iterations {
                        item_dimensional.execute_batch(
                            &mut item_dimensional_output,
                            &inputs,
                            experts,
                            &weights,
                            false,
                        )?;
                    }
                    item_dimensional_samples
                        .push(normalized_burst_duration(started, burst_iterations));
                }
                BatchTimingPath::Tile4 => {
                    for _ in 0..burst_iterations {
                        tile4.execute_batch(
                            &mut tile4_output,
                            &inputs,
                            experts,
                            &weights,
                            false,
                        )?;
                    }
                    tile4_samples.push(normalized_burst_duration(started, burst_iterations));
                }
            }
        }
    }
    assert_eq!(
        immediate_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        per_item_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} per-item timing changed independent results"
    );
    assert_eq!(
        immediate_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        item_dimensional_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} item-dimensional timing changed independent results"
    );
    assert_eq!(
        immediate_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        tile4_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "batch-{batch_len} expert-major Tile4 timing changed independent results"
    );
    let per_item_over_immediate = median_f64(
        immediate_samples
            .iter()
            .zip(&per_item_samples)
            .map(|(control, candidate)| control.as_secs_f64() / candidate.as_secs_f64())
            .collect(),
    );
    let item_dimensional_over_per_item = median_f64(
        per_item_samples
            .iter()
            .zip(&item_dimensional_samples)
            .map(|(control, candidate)| control.as_secs_f64() / candidate.as_secs_f64())
            .collect(),
    );
    let item_dimensional_saved_us = median_f64(
        per_item_samples
            .iter()
            .zip(&item_dimensional_samples)
            .map(|(control, candidate)| (control.as_secs_f64() - candidate.as_secs_f64()) * 1e6)
            .collect(),
    );
    let tile4_over_per_item = median_f64(
        per_item_samples
            .iter()
            .zip(&tile4_samples)
            .map(|(control, candidate)| control.as_secs_f64() / candidate.as_secs_f64())
            .collect(),
    );
    let tile4_saved_us = median_f64(
        per_item_samples
            .iter()
            .zip(&tile4_samples)
            .map(|(control, candidate)| (control.as_secs_f64() - candidate.as_secs_f64()) * 1e6)
            .collect(),
    );
    let tile4_tiles = ROUTES * batch_len.div_ceil(4);
    Ok(BatchProgramTiming {
        batch_len,
        per_item_prepare,
        item_dimensional_prepare,
        tile4_prepare,
        immediate: median(immediate_samples),
        per_item: median(per_item_samples),
        item_dimensional: median(item_dimensional_samples),
        tile4: median(tile4_samples),
        per_item_over_immediate,
        item_dimensional_over_per_item,
        item_dimensional_saved_us,
        tile4_over_per_item,
        tile4_saved_us,
        tile4_tiles,
    })
}

#[cfg(feature = "vulkan")]
fn main() -> anyhow::Result<()> {
    let cpu = Cpu::new();
    let vulkan = VulkanMatvec::new()?;
    let input: Vec<f32> = (0..HIDDEN)
        .map(|index| ((index as f32) * 0.017).cos() * 0.5)
        .collect();

    // The selected top-8 working set is exactly 48 MiB. Stream deterministic BF16
    // directly into one bounded 64 MiB arena block, with no full host-side copy.
    let mut builder = vulkan.arena_builder(WeightArenaConfig {
        block_bytes: 64 * MIB,
        budget_bytes: (64 * MIB) as u64,
        allocation_headroom: 64,
    })?;
    let mut matrix_ids = Vec::with_capacity(ROUTES);
    for &expert_id in &EXPERT_IDS {
        matrix_ids.push([
            push_generated(&mut builder, expert_id, Projection::Gate)?,
            push_generated(&mut builder, expert_id, Projection::Up)?,
            push_generated(&mut builder, expert_id, Projection::Down)?,
        ]);
    }
    let arena = builder.finish()?;
    let stats = arena.stats();
    assert_eq!(
        stats.blocks, 1,
        "selected experts must share one arena block"
    );
    assert_eq!(stats.matrices, ROUTES * 3);
    assert_eq!(stats.encoded_bytes, 48 * MIB as u64);

    let matrices = matrix_ids
        .iter()
        .map(|ids| {
            Ok([
                arena.matrix(ids[0])?,
                arena.matrix(ids[1])?,
                arena.matrix(ids[2])?,
            ])
        })
        .collect::<Result<Vec<_>, DeviceError>>()?;
    let expert_views: Vec<_> = matrices
        .iter()
        .map(|prepared| Bf16Expert {
            gate: &prepared[0],
            up: &prepared[1],
            down: &prepared[2],
        })
        .collect();
    let fused_experts = vulkan.prepare_bf16_routed_experts(&expert_views)?;
    let qualification_experts = (0..*PROGRAM_BATCH_SIZES.last().unwrap())
        .map(|item| {
            let rotated = (0..ROUTES)
                .map(|route| expert_views[(route + item) % ROUTES])
                .collect::<Vec<_>>();
            vulkan.prepare_bf16_routed_experts(&rotated)
        })
        .collect::<Result<Vec<_>, DeviceError>>()?;

    let expected = cpu_oracle(&cpu, &input)?;
    let mut legacy_output = vec![0.0f32; HIDDEN];
    let mut gate = vec![0.0f32; EXPERT_WIDTH];
    let mut up = vec![0.0f32; EXPERT_WIDTH];
    legacy_routed_moe(
        &cpu,
        &vulkan,
        &mut legacy_output,
        &input,
        &matrices,
        &mut gate,
        &mut up,
    )?;
    let legacy_error = assert_correct("legacy 24-submit path", &expected, &legacy_output);

    // Start from sentinels to prove overwrite semantics, then repeat bit-for-bit.
    let mut guarded_output = vec![123.0f32; HIDDEN];
    vulkan.routed_moe_bf16_guarded(
        &mut guarded_output,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        false,
    )?;
    let guarded_error = assert_correct("guarded overwrite path", &expected, &guarded_output);
    let guarded_first_bits: Vec<_> = guarded_output.iter().map(|value| value.to_bits()).collect();
    vulkan.routed_moe_bf16_guarded(
        &mut guarded_output,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        false,
    )?;
    assert_eq!(
        guarded_first_bits,
        guarded_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "guarded routed-MoE output was not bit-repeatable"
    );

    let mut fast_output = vec![-123.0f32; HIDDEN];
    vulkan.routed_moe_bf16_fast(
        &mut fast_output,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        false,
    )?;
    let fast_error = assert_correct("fast overwrite path", &expected, &fast_output);
    assert_correct("guarded/fast parity", &guarded_output, &fast_output);
    let fast_first_bits: Vec<_> = fast_output.iter().map(|value| value.to_bits()).collect();
    vulkan.routed_moe_bf16_fast(
        &mut fast_output,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        false,
    )?;
    assert_eq!(
        fast_first_bits,
        fast_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "fast routed-MoE output was not bit-repeatable"
    );

    let initial: Vec<f32> = (0..HIDDEN)
        .map(|index| 0.125 + (index % 17) as f32 * 0.000_125)
        .collect();
    let expected_accumulated: Vec<_> = initial
        .iter()
        .zip(&expected)
        .map(|(&base, &delta)| base + delta)
        .collect();
    let mut guarded_accumulated = initial.clone();
    vulkan.routed_moe_bf16_guarded(
        &mut guarded_accumulated,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        true,
    )?;
    let guarded_accumulate_error = assert_correct(
        "guarded accumulate path",
        &expected_accumulated,
        &guarded_accumulated,
    );
    let mut fast_accumulated = initial.clone();
    vulkan.routed_moe_bf16_fast(
        &mut fast_accumulated,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        true,
    )?;
    let fast_accumulate_error = assert_correct(
        "fast accumulate path",
        &expected_accumulated,
        &fast_accumulated,
    );

    // Prepare a private program once. Its descriptor sets, activation buffers,
    // persistent mappings, and overwrite/accumulate command buffers are excluded from
    // steady-state timing, while every execute still uploads x/route weights, submits,
    // waits, and reads y back.
    let program_started = Instant::now();
    let program = vulkan.prepare_routed_moe_bf16_fast_program(
        &fused_experts,
        RoutedMoeProgramConfig::new(ROUTED_PROGRAM_BUDGET_BYTES),
    )?;
    let program_prepare = program_started.elapsed();
    assert_eq!(
        program.fast_variant(),
        vulkan.selected_routed_moe_fast_variant(&fused_experts),
        "the reusable program must pin the ordinary production variant"
    );
    let mut program_output = vec![73.0f32; HIDDEN];
    program.execute(
        &mut program_output,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        false,
    )?;
    assert_eq!(
        fast_first_bits,
        program_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the reusable program changed production fast output"
    );
    let mut program_accumulated = initial;
    program.execute(
        &mut program_accumulated,
        &input,
        &fused_experts,
        &ROUTE_WEIGHTS,
        true,
    )?;
    let program_accumulate_error = assert_correct(
        "reusable-program accumulate path",
        &expected_accumulated,
        &program_accumulated,
    );
    assert_eq!(
        fast_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        program_accumulated
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "the reusable program changed production accumulate bits"
    );

    for _ in 0..WARMUPS {
        legacy_routed_moe(
            &cpu,
            &vulkan,
            &mut legacy_output,
            &input,
            &matrices,
            &mut gate,
            &mut up,
        )?;
        vulkan.routed_moe_bf16_guarded(
            &mut guarded_output,
            &input,
            &fused_experts,
            &ROUTE_WEIGHTS,
            false,
        )?;
        vulkan.routed_moe_bf16_fast(
            &mut fast_output,
            &input,
            &fused_experts,
            &ROUTE_WEIGHTS,
            false,
        )?;
    }

    let mut legacy_samples = Vec::with_capacity(SAMPLES);
    let mut guarded_samples = Vec::with_capacity(SAMPLES);
    let mut fast_samples = Vec::with_capacity(SAMPLES);
    for sample in 0..SAMPLES {
        // Cycle through every path ordering twice so each path occupies each timing
        // position equally and follows each other path equally often.
        for path in TIMING_ORDERS[sample % TIMING_ORDERS.len()] {
            match path {
                TimingPath::Legacy => {
                    let started = Instant::now();
                    legacy_routed_moe(
                        &cpu,
                        &vulkan,
                        &mut legacy_output,
                        &input,
                        &matrices,
                        &mut gate,
                        &mut up,
                    )?;
                    legacy_samples.push(started.elapsed());
                }
                TimingPath::Guarded => {
                    let started = Instant::now();
                    vulkan.routed_moe_bf16_guarded(
                        &mut guarded_output,
                        &input,
                        &fused_experts,
                        &ROUTE_WEIGHTS,
                        false,
                    )?;
                    guarded_samples.push(started.elapsed());
                }
                TimingPath::Fast => {
                    let started = Instant::now();
                    vulkan.routed_moe_bf16_fast(
                        &mut fast_output,
                        &input,
                        &fused_experts,
                        &ROUTE_WEIGHTS,
                        false,
                    )?;
                    fast_samples.push(started.elapsed());
                }
            }
        }
    }

    // Measure timestamp perturbation separately so the qualified three-path control
    // above remains byte-for-byte uninstrumented. Alternating pair order removes the
    // simplest hot-cache/order bias while keeping an equal sample count for each path.
    let mut profiled_output = vec![0.0f32; HIDDEN];
    for warmup in 0..PROFILE_WARMUPS {
        if warmup.is_multiple_of(2) {
            vulkan.routed_moe_bf16_fast(
                &mut fast_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
            vulkan.routed_moe_bf16_fast_profiled(
                &mut profiled_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
        } else {
            vulkan.routed_moe_bf16_fast_profiled(
                &mut profiled_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
            vulkan.routed_moe_bf16_fast(
                &mut fast_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
        }
    }
    let mut profile_control_samples = Vec::with_capacity(SAMPLES);
    let mut profile_wall_samples = Vec::with_capacity(SAMPLES);
    let mut profile_adjusted_wall_samples = Vec::with_capacity(SAMPLES);
    let mut query_readback_samples = Vec::with_capacity(SAMPLES);
    let mut gate_up_samples = Vec::with_capacity(SAMPLES);
    let mut gate_to_down_samples = Vec::with_capacity(SAMPLES);
    let mut down_samples = Vec::with_capacity(SAMPLES);
    let mut down_to_reduce_samples = Vec::with_capacity(SAMPLES);
    let mut reduce_samples = Vec::with_capacity(SAMPLES);
    let mut gpu_total_samples = Vec::with_capacity(SAMPLES);
    let mut timestamp_valid_bits = 0;
    let mut timestamp_period_ns = 0.0;
    for sample in 0..SAMPLES {
        let order = if sample.is_multiple_of(2) {
            [false, true]
        } else {
            [true, false]
        };
        for profiled in order {
            let started = Instant::now();
            if profiled {
                let profile = vulkan.routed_moe_bf16_fast_profiled(
                    &mut profiled_output,
                    &input,
                    &fused_experts,
                    &ROUTE_WEIGHTS,
                    false,
                )?;
                let wall = started.elapsed();
                profile_wall_samples.push(wall);
                profile_adjusted_wall_samples
                    .push(wall.saturating_sub(profile.host_query_readback));
                query_readback_samples.push(profile.host_query_readback);
                gate_up_samples.push(profile.gate_up);
                gate_to_down_samples.push(profile.gate_to_down_dependency);
                down_samples.push(profile.down);
                down_to_reduce_samples.push(profile.down_to_reduce_dependency);
                reduce_samples.push(profile.reduce);
                gpu_total_samples.push(profile.total);
                timestamp_valid_bits = profile.counter_valid_bits;
                timestamp_period_ns = profile.timestamp_period_ns;
            } else {
                vulkan.routed_moe_bf16_fast(
                    &mut fast_output,
                    &input,
                    &fused_experts,
                    &ROUTE_WEIGHTS,
                    false,
                )?;
                profile_control_samples.push(started.elapsed());
            }
        }
    }
    assert_eq!(
        fast_first_bits,
        profiled_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "timestamp instrumentation changed fast routed-MoE output"
    );

    // The qualified shape on GFX1151 selects Tree128; QualifiedTree64 remains the
    // portable matched control. Every candidate proves CPU/guarded parity and within-
    // pipeline repeatability before timing. Lazy pipeline creation happens in these
    // calls and is therefore excluded from every sample.
    let mut tournament_output = vec![0.0f32; HIDDEN];
    let mut tournament_errors = Vec::with_capacity(FAST_VARIANTS.len());
    for variant in FAST_VARIANTS {
        tournament_output.fill(41.0);
        vulkan.routed_moe_bf16_fast_variant(
            &mut tournament_output,
            &input,
            &fused_experts,
            &ROUTE_WEIGHTS,
            false,
            variant,
        )?;
        let cpu_error = assert_correct(variant.label(), &expected, &tournament_output);
        let guarded_error = assert_correct(
            &format!("guarded/{} parity", variant.label()),
            &guarded_output,
            &tournament_output,
        );
        tournament_errors.push((cpu_error, guarded_error));
        let first_bits: Vec<_> = tournament_output
            .iter()
            .map(|value| value.to_bits())
            .collect();
        if variant == vulkan.selected_routed_moe_fast_variant(&fused_experts) {
            assert_eq!(
                fast_first_bits, first_bits,
                "the automatic path did not use its reported paired-load variant"
            );
        }
        vulkan.routed_moe_bf16_fast_variant(
            &mut tournament_output,
            &input,
            &fused_experts,
            &ROUTE_WEIGHTS,
            false,
            variant,
        )?;
        assert_eq!(
            first_bits,
            tournament_output
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "{} was not bit-repeatable",
            variant.label()
        );
    }
    for warmup in 0..WARMUPS {
        for &base_index in &WILLIAMS_BASE_ORDER {
            let variant = FAST_VARIANTS[(base_index + warmup) % FAST_VARIANTS.len()];
            vulkan.routed_moe_bf16_fast_variant(
                &mut tournament_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
                variant,
            )?;
        }
    }
    let mut tournament_wall_samples: [Vec<Duration>; 6] =
        std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
    for sample in 0..SAMPLES {
        let shift = sample % FAST_VARIANTS.len();
        for &base_index in &WILLIAMS_BASE_ORDER {
            let index = (base_index + shift) % FAST_VARIANTS.len();
            let started = Instant::now();
            vulkan.routed_moe_bf16_fast_variant(
                &mut tournament_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
                FAST_VARIANTS[index],
            )?;
            tournament_wall_samples[index].push(started.elapsed());
        }
    }

    // Profile in a separate balanced pass: timestamp packets are diagnostic and never
    // contaminate the wall-time ordering used to choose or promote a candidate.
    let mut tournament_gate_samples: [Vec<Duration>; 6] =
        std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
    let mut tournament_down_samples: [Vec<Duration>; 6] =
        std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
    let mut tournament_reduce_samples: [Vec<Duration>; 6] =
        std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
    let mut tournament_gpu_samples: [Vec<Duration>; 6] =
        std::array::from_fn(|_| Vec::with_capacity(SAMPLES));
    for sample in 0..SAMPLES {
        let shift = sample % FAST_VARIANTS.len();
        for &base_index in &WILLIAMS_BASE_ORDER {
            let index = (base_index + shift) % FAST_VARIANTS.len();
            let profile = vulkan.routed_moe_bf16_fast_variant_profiled(
                &mut tournament_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
                FAST_VARIANTS[index],
            )?;
            tournament_gate_samples[index].push(profile.gate_up);
            tournament_down_samples[index].push(profile.down);
            tournament_reduce_samples[index].push(profile.reduce);
            tournament_gpu_samples[index].push(profile.total);
        }
    }

    // A longer AB/BA comparison isolates host-command reuse from the six-way shader
    // tournament. Both paths run the same selected pipeline and retain the same one-
    // submit/one-fence host boundary. The candidate measures the complete prepared-
    // program package: persistent mappings, immutable descriptors, private buffers,
    // and command reuse, rather than command recording in isolation.
    for warmup in 0..PROGRAM_WARMUPS {
        if warmup.is_multiple_of(2) {
            vulkan.routed_moe_bf16_fast(
                &mut fast_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
            program.execute(
                &mut program_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
        } else {
            program.execute(
                &mut program_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
            vulkan.routed_moe_bf16_fast(
                &mut fast_output,
                &input,
                &fused_experts,
                &ROUTE_WEIGHTS,
                false,
            )?;
        }
    }
    let mut program_control_samples = Vec::with_capacity(PROGRAM_SAMPLES);
    let mut program_wall_samples = Vec::with_capacity(PROGRAM_SAMPLES);
    for sample in 0..PROGRAM_SAMPLES {
        let program_first = !sample.is_multiple_of(2);
        for use_program in [program_first, !program_first] {
            let started = Instant::now();
            if use_program {
                program.execute(
                    &mut program_output,
                    &input,
                    &fused_experts,
                    &ROUTE_WEIGHTS,
                    false,
                )?;
                program_wall_samples.push(started.elapsed());
            } else {
                vulkan.routed_moe_bf16_fast(
                    &mut fast_output,
                    &input,
                    &fused_experts,
                    &ROUTE_WEIGHTS,
                    false,
                )?;
                program_control_samples.push(started.elapsed());
            }
        }
    }
    assert_eq!(
        fast_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        program_output
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "interleaving ordinary calls changed reusable-program output"
    );
    let program_paired_speedup = median_f64(
        program_control_samples
            .iter()
            .zip(&program_wall_samples)
            .map(|(control, candidate)| control.as_secs_f64() / candidate.as_secs_f64())
            .collect(),
    );
    let program_paired_saved_us = median_f64(
        program_control_samples
            .iter()
            .zip(&program_wall_samples)
            .map(|(control, candidate)| (control.as_secs_f64() - candidate.as_secs_f64()) * 1e6)
            .collect(),
    );
    let program_control = median(program_control_samples);
    let program_wall = median(program_wall_samples);
    let batch_timings = PROGRAM_BATCH_SIZES
        .into_iter()
        .map(|batch_len| {
            benchmark_program_batch(&vulkan, &input, &qualification_experts[..batch_len])
        })
        .collect::<Result<Vec<_>, DeviceError>>()?;

    let legacy = median(legacy_samples);
    let guarded = median(guarded_samples);
    let fast = median(fast_samples);
    let profile_control = median(profile_control_samples);
    let profile_wall = median(profile_wall_samples);
    let profile_adjusted_wall = median(profile_adjusted_wall_samples);
    let query_readback = median(query_readback_samples);
    let gate_up = median(gate_up_samples);
    let gate_to_down = median(gate_to_down_samples);
    let down = median(down_samples);
    let down_to_reduce = median(down_to_reduce_samples);
    let reduce = median(reduce_samples);
    let gpu_total = median(gpu_total_samples);
    let tournament_wall = tournament_wall_samples.map(median);
    let tournament_gate = tournament_gate_samples.map(median);
    let tournament_down = tournament_down_samples.map(median);
    let tournament_reduce = tournament_reduce_samples.map(median);
    let tournament_gpu = tournament_gpu_samples.map(median);
    let legacy_over_guarded = legacy.as_secs_f64() / guarded.as_secs_f64();
    let guarded_over_fast = guarded.as_secs_f64() / fast.as_secs_f64();
    let legacy_over_fast = legacy.as_secs_f64() / fast.as_secs_f64();
    let program_ratio_of_medians = program_control.as_secs_f64() / program_wall.as_secs_f64();
    let profile_overhead = profile_wall.as_secs_f64() / profile_control.as_secs_f64() - 1.0;
    let profile_command_overhead =
        profile_adjusted_wall.as_secs_f64() / profile_control.as_secs_f64() - 1.0;
    let gpu_wall_share = gpu_total.as_secs_f64() / profile_wall.as_secs_f64();
    let percent_of_gpu = |duration: Duration| {
        100.0 * duration.as_secs_f64() / gpu_total.as_secs_f64().max(f64::MIN_POSITIVE)
    };
    println!("Vulkan device: {}", vulkan.physical_device_name());
    println!(
        "Qwen3.6-shaped routed experts: H={HIDDEN}, F={EXPERT_WIDTH}, K={ROUTES}; conceptual IDs={EXPERT_IDS:?}"
    );
    println!(
        "Arena: {} matrices, {} block, {:.1} MiB encoded / {:.1} MiB reserved",
        stats.matrices,
        stats.blocks,
        stats.encoded_bytes as f64 / MIB as f64,
        stats.buffer_bytes as f64 / MIB as f64
    );
    println!(
        "Timing excludes weight preparation; all paths include input upload and final readback. Medians: {SAMPLES} samples after {WARMUPS} warmups."
    );
    println!(
        "legacy 24 calls + CPU SwiGLU: {:>10.6} ms ({:>7.2} routed eval/s)",
        legacy.as_secs_f64() * 1e3,
        1.0 / legacy.as_secs_f64()
    );
    println!(
        "guarded/safe 3 dispatches / 1 submit: {:>10.6} ms ({:>7.2} routed eval/s)",
        guarded.as_secs_f64() * 1e3,
        1.0 / guarded.as_secs_f64()
    );
    println!(
        "fast paired-load, 3 dispatches / 1 submit: {:>10.6} ms ({:>7.2} routed eval/s)",
        fast.as_secs_f64() * 1e3,
        1.0 / fast.as_secs_f64()
    );
    println!(
        "reusable program, persistent maps + recorded commands: {:>10.6} ms ({:>7.2} routed eval/s); immediate production control={:.6} ms",
        program_wall.as_secs_f64() * 1e3,
        1.0 / program_wall.as_secs_f64(),
        program_control.as_secs_f64() * 1e3,
    );
    println!(
        "reusable-program package: ratio-of-medians={program_ratio_of_medians:.3}x  median paired={program_paired_speedup:.3}x  paired saving={program_paired_saved_us:+.3} us  prepare={:.3} ms  gate (>=1.10x): {} ({PROGRAM_SAMPLES} alternating pairs)",
        program_prepare.as_secs_f64() * 1e3,
        if program_paired_speedup >= 1.10 {
            "PASS"
        } else {
            "NO-GO"
        },
    );
    println!(
        "independent high-overlap batch programs (distinct x/routes/weights/y; {BATCH_SAMPLES} balanced burst quartets, at least {BATCH_BURST_EVALS} evals/path/sample):"
    );
    for timing in &batch_timings {
        println!(
            "  N={:<2}: immediate={:.6} ms  per-item-3N={:.6} ms ({:.1} eval/s, immediate/control={:.3}x)  item-3D={:.6} ms ({:.6} ms/eval, {:.1} eval/s)  control/candidate={:.3}x  saved={:+.3} us/batch  prepare={:.3}/{:.3} ms  gate={}",
            timing.batch_len,
            timing.immediate.as_secs_f64() * 1e3,
            timing.per_item.as_secs_f64() * 1e3,
            timing.batch_len as f64 / timing.per_item.as_secs_f64(),
            timing.per_item_over_immediate,
            timing.item_dimensional.as_secs_f64() * 1e3,
            timing.item_dimensional.as_secs_f64() * 1e3 / timing.batch_len as f64,
            timing.batch_len as f64 / timing.item_dimensional.as_secs_f64(),
            timing.item_dimensional_over_per_item,
            timing.item_dimensional_saved_us,
            timing.per_item_prepare.as_secs_f64() * 1e3,
            timing.item_dimensional_prepare.as_secs_f64() * 1e3,
            if timing.batch_len == 1 {
                "CONTROL"
            } else if timing.item_dimensional_over_per_item >= ITEM_DIMENSION_GO_GATE {
                "PASS"
            } else {
                "NO-GO"
            },
        );
        let occurrences = timing.batch_len * ROUTES;
        println!(
            "        Tile4={:.6} ms ({:.6} ms/eval, {:.1} eval/s)  control/candidate={:.3}x  saved={:+.3} us/batch  schedule M/T={}/{}={:.3}x  prepare={:.3} ms  gate={}",
            timing.tile4.as_secs_f64() * 1e3,
            timing.tile4.as_secs_f64() * 1e3 / timing.batch_len as f64,
            timing.batch_len as f64 / timing.tile4.as_secs_f64(),
            timing.tile4_over_per_item,
            timing.tile4_saved_us,
            occurrences,
            timing.tile4_tiles,
            occurrences as f64 / timing.tile4_tiles as f64,
            timing.tile4_prepare.as_secs_f64() * 1e3,
            if timing.batch_len == 1 {
                "CONTROL"
            } else if timing.tile4_over_per_item >= TILE4_GO_GATE {
                "PASS"
            } else {
                "NO-GO"
            },
        );
    }
    println!(
        "ratios: legacy/safe={legacy_over_guarded:.2}x  safe/fast={guarded_over_fast:.2}x  legacy/fast={legacy_over_fast:.2}x"
    );
    println!(
        "timestamp probe wall: control={:.6} ms  profiled={:.6} ms  overhead={:+.2}%  query-read={:.6} ms  command-adjusted={:+.2}% ({SAMPLES} alternating pairs)",
        profile_control.as_secs_f64() * 1e3,
        profile_wall.as_secs_f64() * 1e3,
        profile_overhead * 100.0,
        query_readback.as_secs_f64() * 1e3,
        profile_command_overhead * 100.0,
    );
    println!(
        "GPU timestamps ({} valid bits, {:.3} ns/tick): gate/up={:.6} ms ({:.1}%)  dependency={:.6} ms ({:.1}%)  down={:.6} ms ({:.1}%)  dependency={:.6} ms ({:.1}%)  reduce={:.6} ms ({:.1}%)  total={:.6} ms ({:.1}% of profiled wall)",
        timestamp_valid_bits,
        timestamp_period_ns,
        gate_up.as_secs_f64() * 1e3,
        percent_of_gpu(gate_up),
        gate_to_down.as_secs_f64() * 1e3,
        percent_of_gpu(gate_to_down),
        down.as_secs_f64() * 1e3,
        percent_of_gpu(down),
        down_to_reduce.as_secs_f64() * 1e3,
        percent_of_gpu(down_to_reduce),
        reduce.as_secs_f64() * 1e3,
        percent_of_gpu(reduce),
        gpu_total.as_secs_f64() * 1e3,
        gpu_wall_share * 100.0,
    );
    let qualified_index = FAST_VARIANTS
        .iter()
        .position(|variant| *variant == RoutedMoeFastVariant::QualifiedTree64)
        .expect("qualified control is in FAST_VARIANTS");
    let winner_index = (0..FAST_VARIANTS.len())
        .min_by_key(|&index| tournament_wall[index])
        .expect("the tournament is non-empty");
    let best_gate_index = (0..FAST_VARIANTS.len())
        .min_by_key(|&index| tournament_gate[index])
        .expect("the tournament is non-empty");
    let best_down_index = (0..FAST_VARIANTS.len())
        .min_by_key(|&index| tournament_down[index])
        .expect("the tournament is non-empty");
    println!("paired-load reduction tournament (uninstrumented wall; profiled GPU stages):");
    for (index, variant) in FAST_VARIANTS.iter().enumerate() {
        let speedup =
            tournament_wall[qualified_index].as_secs_f64() / tournament_wall[index].as_secs_f64();
        println!(
            "  {:>23}: wall={:.6} ms  control/candidate={:.3}x  gate/up={:.6} ms  down={:.6} ms  reduce={:.6} ms  GPU-total={:.6} ms  CPU rel-L2={:.3e}  guarded rel-L2={:.3e}",
            variant.label(),
            tournament_wall[index].as_secs_f64() * 1e3,
            speedup,
            tournament_gate[index].as_secs_f64() * 1e3,
            tournament_down[index].as_secs_f64() * 1e3,
            tournament_reduce[index].as_secs_f64() * 1e3,
            tournament_gpu[index].as_secs_f64() * 1e3,
            tournament_errors[index].0.relative_l2,
            tournament_errors[index].1.relative_l2,
        );
    }
    let winner_speedup = tournament_wall[qualified_index].as_secs_f64()
        / tournament_wall[winner_index].as_secs_f64();
    let promoted_index = FAST_VARIANTS
        .iter()
        .position(|variant| *variant == RoutedMoeFastVariant::Tree128)
        .expect("the promoted GFX1151 variant is in FAST_VARIANTS");
    let promoted_speedup = tournament_wall[qualified_index].as_secs_f64()
        / tournament_wall[promoted_index].as_secs_f64();
    println!(
        "tournament winner: {} at {:.6} ms ({winner_speedup:.3}x over qualified control); fastest gate/up={} fastest down={}; promotion gate (>=1.10x): {}",
        FAST_VARIANTS[winner_index].label(),
        tournament_wall[winner_index].as_secs_f64() * 1e3,
        FAST_VARIANTS[best_gate_index].label(),
        FAST_VARIANTS[best_down_index].label(),
        if promoted_speedup >= 1.10 { "PASS" } else { "NO-GO" },
    );
    println!(
        "legacy rel-L2={:.3e} max-abs={:.3e}  guarded rel-L2={:.3e} max-abs={:.3e}  fast rel-L2={:.3e} max-abs={:.3e}",
        legacy_error.relative_l2,
        legacy_error.max_absolute,
        guarded_error.relative_l2,
        guarded_error.max_absolute,
        fast_error.relative_l2,
        fast_error.max_absolute
    );
    println!(
        "guarded accumulate rel-L2={:.3e} max-abs={:.3e}  fast accumulate rel-L2={:.3e} max-abs={:.3e}  repeatability=bit-exact",
        guarded_accumulate_error.relative_l2,
        guarded_accumulate_error.max_absolute,
        fast_accumulate_error.relative_l2,
        fast_accumulate_error.max_absolute
    );
    println!(
        "reusable program accumulate rel-L2={:.3e} max-abs={:.3e}; overwrite/accumulate match production bits",
        program_accumulate_error.relative_l2,
        program_accumulate_error.max_absolute,
    );
    assert!(
        legacy_over_guarded >= 1.25,
        "guarded routed-MoE go gate requires >=1.25x, measured {legacy_over_guarded:.2}x"
    );
    assert!(
        guarded_over_fast >= 1.05,
        "paired-load routed-MoE go gate requires >=1.05x over guarded, measured {guarded_over_fast:.2}x"
    );
    if vulkan.selected_routed_moe_fast_variant(&fused_experts) == RoutedMoeFastVariant::Tree128 {
        assert!(
            promoted_speedup >= 1.10,
            "promoted GFX1151 Tree128 gate requires >=1.10x over Tree64, measured {promoted_speedup:.3}x"
        );
        for timing in &batch_timings {
            if timing.batch_len > 1 {
                assert!(
                    timing.per_item_over_immediate >= BATCH_GO_GATE,
                    "qualified GFX1151 batch-{} control gate requires >={BATCH_GO_GATE:.2}x over independent immediate calls, measured {:.3}x",
                    timing.batch_len,
                    timing.per_item_over_immediate,
                );
            }
        }
    }
    println!(
        "timestamp perturbation gate (<=2% command-adjusted): {}",
        if profile_command_overhead <= 0.02 {
            "PASS"
        } else {
            "EXPLORATORY ONLY"
        }
    );
    Ok(())
}

#[cfg(not(feature = "vulkan"))]
fn main() {
    eprintln!("re-run with `--features vulkan`");
    std::process::exit(2);
}
