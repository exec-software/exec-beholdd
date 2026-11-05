//! Opt-in resident FFN qualification; CPU oracle checks run before native access.
//!
//! Preregistered finite-fixture tolerances: activation/up stage and isolated
//! SwiGLU absolute error <= 2e-5 + 2e-5 * |f64 reference|; final residual output
//! <= 3e-4 + 3e-5 * |reference|. Nonzero signs and signed exact zeros are checked
//! separately. This is numerical parity for fixed fixtures, not a bitwise,
//! full-model quality, lossless-inference, or throughput guarantee.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{ensure, Context, Result};
    use beholdd_backend::execution::{
        CommandList, CpuExecutionDevice, ExecutionDevice, F32Tensor, LinearMode, RecordedBatch,
    };
    use beholdd_backend::hag::native_execution::{
        NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::Weight;
    use beholdd_format::GgmlType;
    use exec_hag::budget::MemoryBudget;
    use std::path::Path;

    const WIDTH: usize = 32;
    const ROUNDS: usize = 2;
    const PACKED_BYTES: u64 = 5184;
    const TENSOR_BYTES: u64 = 512;
    const AUXILIARY_PER_BATCH: u64 = 136;
    const DISPATCHES_PER_BATCH: u64 = 5;
    const BUDGET_BYTES: u64 = 256 * 1024;
    const DIAGNOSTIC_GATE: [f32; 8] = [-8.0, -2.0, -0.75, -0.0, 0.75, 2.0, 8.0, -0.001];
    const DIAGNOSTIC_UP: [f32; 8] = [-3.0, -2.0, 0.0, 3.0, 2.0, -4.0, 2.0, 1.0];

    #[derive(Clone, Copy)]
    struct Case {
        name: &'static str,
        limit: f32,
        scale: f32,
    }

    const CASES: [Case; 3] = [
        Case {
            name: "unclamped-positive",
            limit: 0.0,
            scale: 0.5,
        },
        Case {
            name: "clamped-negative",
            limit: 0.75,
            scale: -0.75,
        },
        Case {
            name: "clamped-zero",
            limit: 0.75,
            scale: 0.0,
        },
    ];

    #[derive(Clone, Copy)]
    struct Tolerance {
        absolute: f64,
        relative: f64,
    }

    // Explicit bounds for these small finite fixtures, not universal error claims.
    const ACTIVATION_TOLERANCE: Tolerance = Tolerance {
        absolute: 2e-5,
        relative: 2e-5,
    };
    const FINAL_TOLERANCE: Tolerance = Tolerance {
        absolute: 3e-4,
        relative: 3e-5,
    };

    struct Fixture {
        gate: Vec<u8>,
        up: Vec<u8>,
        down: Vec<u8>,
    }

    struct Chain {
        input: F32Tensor,
        gate: F32Tensor,
        up: F32Tensor,
        output: F32Tensor,
        batches: Vec<RecordedBatch>,
    }

    fn gate_weight(row: usize, column: usize) -> f64 {
        let diagonal = if row == column { 1.0 } else { 0.0 };
        diagonal + (((row * 3 + column * 5) % 7) as f64 - 3.0) / 128.0
    }

    fn up_weight(row: usize, column: usize) -> f64 {
        let diagonal = if row == column { -0.75 } else { 0.0 };
        let neighbor = if (row + 1) % WIDTH == column {
            0.5
        } else {
            0.0
        };
        diagonal + neighbor + (((row * 7 + column * 3) % 7) as f64 - 3.0) / 256.0
    }

    fn down_scale(row: usize) -> (u16, f64) {
        match row % 3 {
            0 => (0x2800, 1.0 / 32.0),
            1 => (0xa400, -1.0 / 64.0),
            _ => (0x2c00, 1.0 / 16.0),
        }
    }

    fn down_integer(row: usize, column: usize) -> i8 {
        ((row * 11 + column * 5) % 15) as i8 - 7
    }

    fn inputs(round: usize) -> Vec<f32> {
        (0..WIDTH)
            .map(|column| (((column * 7 + round * 3) % 19) as f32 - 9.0) / 2.0)
            .collect()
    }

    fn residuals(round: usize) -> Vec<f32> {
        (0..WIDTH)
            .map(|row| (((row * 3 + round * 5) % 23) as f32 - 11.0) / 16.0)
            .collect()
    }

    fn pack_bf16(weight: fn(usize, usize) -> f64) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(WIDTH * WIDTH * 2);
        for row in 0..WIDTH {
            for column in 0..WIDTH {
                let bits = (weight(row, column) as f32).to_bits();
                ensure!(bits & 0xffff == 0, "fixture coefficient must be exact BF16");
                bytes.extend_from_slice(&((bits >> 16) as u16).to_le_bytes());
            }
        }
        Ok(bytes)
    }

    fn fixture() -> Result<Fixture> {
        let mut down = Vec::with_capacity(WIDTH * 34);
        for row in 0..WIDTH {
            down.extend_from_slice(&down_scale(row).0.to_le_bytes());
            for column in 0..WIDTH {
                down.push(down_integer(row, column) as u8);
            }
        }
        let data = Fixture {
            gate: pack_bf16(gate_weight)?,
            up: pack_bf16(up_weight)?,
            down,
        };
        ensure!((data.gate.len() + data.up.len() + data.down.len()) as u64 == PACKED_BYTES);
        Ok(data)
    }

    fn activation(gate: f64, up: f64, limit: f64) -> f64 {
        let bounded_gate = if limit > 0.0 { gate.min(limit) } else { gate };
        let bounded_up = if limit > 0.0 {
            up.clamp(-limit, limit)
        } else {
            up
        };
        bounded_gate / (1.0 + (-bounded_gate).exp()) * bounded_up
    }

    fn projected(weight: fn(usize, usize) -> f64, input: &[f32]) -> Vec<f64> {
        (0..WIDTH)
            .map(|row| {
                input
                    .iter()
                    .enumerate()
                    .map(|(column, value)| weight(row, column) * f64::from(*value))
                    .sum()
            })
            .collect()
    }

    fn reference(case: Case, round: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let input = inputs(round);
        let gate = projected(gate_weight, &input);
        let up = projected(up_weight, &input);
        ensure!(gate
            .iter()
            .chain(&up)
            .all(|value| value.is_finite() && value.abs() <= 20.0));
        if case.limit > 0.0 {
            let limit = f64::from(case.limit);
            ensure!(gate.iter().any(|value| *value < -limit));
            ensure!(gate.iter().any(|value| *value > limit));
            ensure!(up.iter().any(|value| *value < -limit));
            ensure!(up.iter().any(|value| *value > limit));
        }
        let activated: Vec<_> = gate
            .iter()
            .zip(&up)
            .map(|(gate_value, up_value)| activation(*gate_value, *up_value, f64::from(case.limit)))
            .collect();
        let residual = residuals(round);
        let output = (0..WIDTH)
            .map(|row| {
                let down: f64 = activated
                    .iter()
                    .enumerate()
                    .map(|(column, value)| {
                        down_scale(row).1 * f64::from(down_integer(row, column)) * value
                    })
                    .sum();
                f64::from(residual[row]) + f64::from(case.scale) * down
            })
            .collect();
        Ok((activated, up, output))
    }

    fn compare(label: &str, actual: &[f32], expected: &[f64], tolerance: Tolerance) -> Result<f64> {
        ensure!(actual.len() == expected.len());
        let mut maximum = 0.0_f64;
        for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(observed) - reference).abs();
            let allowed = tolerance.absolute + tolerance.relative * reference.abs();
            ensure!(observed.is_finite() && reference.is_finite() && error <= allowed,
                "{label}[{index}]: got {observed:?}, expected {reference:?}, error={error}, allowed={allowed}");
            let sign_matches = if reference == 0.0 {
                observed == 0.0 && observed.is_sign_negative() == reference.is_sign_negative()
            } else {
                observed != 0.0 && observed.is_sign_negative() == reference.is_sign_negative()
            };
            ensure!(
                sign_matches,
                "{label}[{index}]: zero/sign mismatch: got {observed:?}, expected {reference:?}"
            );
            maximum = maximum.max(error);
        }
        Ok(maximum)
    }

    fn record_chain(device: &dyn ExecutionDevice, data: &Fixture) -> Result<Chain> {
        let mut prepared = Vec::new();
        for (bytes, kind) in [
            (&data.gate, GgmlType(30)),
            (&data.up, GgmlType(30)),
            (&data.down, GgmlType(8)),
        ] {
            let encoded = device.upload_weight(&Weight {
                bytes,
                kind,
                rows: WIDTH,
                cols: WIDTH,
            })?;
            let weight = device.prepare_weight(&encoded)?;
            ensure!(weight.byte_len() == bytes.len());
            prepared.push(weight);
        }
        let input = device.upload_f32(&inputs(0))?;
        let gate = device.allocate_f32(WIDTH)?;
        let up = device.allocate_f32(WIDTH)?;
        let output = device.upload_f32(&residuals(0))?;
        let mut batches = Vec::new();
        for case in CASES {
            let mut commands = CommandList::new();
            commands.linear(
                &gate.full_view(),
                &input.full_view(),
                &prepared[0],
                LinearMode::Overwrite,
            )?;
            commands.linear(
                &up.full_view(),
                &input.full_view(),
                &prepared[1],
                LinearMode::Overwrite,
            )?;
            commands.swiglu(&gate.full_view(), &up.full_view(), case.limit)?;
            commands.linear(
                &output.full_view(),
                &gate.full_view(),
                &prepared[2],
                LinearMode::ScaledAdd { scale: case.scale },
            )?;
            let batch = device.record(&commands)?;
            ensure!(batch.len() == 4);
            batches.push(batch);
        }
        // Batch snapshots retain their packed weights after all original weight
        // handles and source command lists leave this function.
        Ok(Chain {
            input: input.clone(),
            gate,
            up,
            output,
            batches,
        })
    }

    fn verify_stages(
        device: &dyn ExecutionDevice,
        chain: &Chain,
        case: Case,
        round: usize,
    ) -> Result<f64> {
        let (expected_gate, expected_up, expected_output) = reference(case, round)?;
        let mut gate = [f32::NAN; WIDTH];
        let mut up = [f32::NAN; WIDTH];
        let mut output = [f32::NAN; WIDTH];
        device.read_f32(&chain.gate.full_view(), &mut gate)?;
        device.read_f32(&chain.up.full_view(), &mut up)?;
        device.read_f32(&chain.output.full_view(), &mut output)?;
        compare(
            "activated gate",
            &gate,
            &expected_gate,
            ACTIVATION_TOLERANCE,
        )?;
        compare("up projection", &up, &expected_up, ACTIVATION_TOLERANCE)?;
        let maximum = compare(
            "residual output",
            &output,
            &expected_output,
            FINAL_TOLERANCE,
        )?;
        if case.scale == 0.0 {
            ensure!(
                output
                    .iter()
                    .zip(residuals(round))
                    .all(|(observed, residual)| observed.to_bits() == residual.to_bits()),
                "zero scale changed the finite residual fixture"
            );
        }
        Ok(maximum)
    }

    fn verify_cpu(data: &Fixture) -> Result<()> {
        let cpu = CpuExecutionDevice::new();
        let chain = record_chain(&cpu, data)?;
        for (index, case) in CASES.into_iter().enumerate() {
            for round in 0..ROUNDS {
                cpu.write_f32(&chain.input.full_view(), &inputs(round))?;
                cpu.write_f32(&chain.output.full_view(), &residuals(round))?;
                let submitted = cpu.submit(&chain.batches[index])?;
                cpu.wait(&submitted)?;
                verify_stages(&cpu, &chain, case, round)
                    .context("CPU FFN/reference disagreement")?;
            }
        }
        verify_cpu_diagnostic()
    }

    fn verify_cpu_diagnostic() -> Result<()> {
        let cpu = CpuExecutionDevice::new();
        let gate = cpu.upload_f32(&DIAGNOSTIC_GATE)?;
        let up = cpu.upload_f32(&DIAGNOSTIC_UP)?;
        let mut commands = CommandList::new();
        commands.swiglu(&gate.full_view(), &up.full_view(), 0.75)?;
        let submitted = cpu.submit(&cpu.record(&commands)?)?;
        cpu.wait(&submitted)?;
        let mut actual = [f32::NAN; DIAGNOSTIC_GATE.len()];
        cpu.read_f32(&gate.full_view(), &mut actual)?;
        compare(
            "CPU diagnostic",
            &actual,
            &diagnostic_reference(),
            ACTIVATION_TOLERANCE,
        )?;
        Ok(())
    }

    fn diagnostic_reference() -> Vec<f64> {
        DIAGNOSTIC_GATE
            .iter()
            .zip(DIAGNOSTIC_UP)
            .map(|(gate, up)| activation(f64::from(*gate), f64::from(up), 0.75))
            .collect()
    }

    fn verify_submission(
        before: NativeExecutionStats,
        after: NativeExecutionStats,
        dispatches: u64,
    ) -> Result<()> {
        ensure!(after.dispatch_attempts == before.dispatch_attempts + dispatches);
        ensure!(after.dispatch_count == before.dispatch_count + dispatches);
        ensure!(after.logical_submissions == before.logical_submissions + 1);
        ensure!(
            after.host_write_bytes == before.host_write_bytes
                && after.host_read_bytes == before.host_read_bytes
        );
        ensure!(after.uploaded_weight_bytes == before.uploaded_weight_bytes);
        ensure!(after.allocation_bytes == before.allocation_bytes);
        ensure!(
            after.weight_live_bytes == before.weight_live_bytes
                && after.tensor_live_bytes == before.tensor_live_bytes
        );
        ensure!(after.auxiliary_live_bytes == before.auxiliary_live_bytes);
        ensure!(!after.poisoned);
        Ok(())
    }

    fn open(path: &Path, budget: &MemoryBudget) -> Result<NativeExecutionDevice> {
        Ok(NativeExecutionDevice::open(
            path,
            budget,
            NativeExecutionConfig {
                weight_capacity_bytes: 8192,
                tensor_capacity_bytes: 4096,
                ..Default::default()
            },
        )?)
    }

    fn announce_tolerances() {
        println!("RUN numerical contract: |error| <= absolute + relative * |f64 reference|; stages absolute={} relative={}; final absolute={} relative={}; finite results and zero/sign preservation required", ACTIVATION_TOLERANCE.absolute, ACTIVATION_TOLERANCE.relative, FINAL_TOLERANCE.absolute, FINAL_TOLERANCE.relative);
    }

    pub(super) fn run(path: &Path) -> Result<()> {
        announce_tolerances();
        let data = fixture()?;
        verify_cpu(&data)?;
        let budget = MemoryBudget::new(BUDGET_BYTES);
        let original = open(path, &budget)?;
        let chain = record_chain(&original, &data)?;
        let device = original.clone();
        drop(original);
        let initial = device.stats()?;
        ensure!(
            initial.allocation_bytes > 0
                && initial.allocation_bytes == budget.current_bytes()
                && initial.allocation_bytes <= BUDGET_BYTES
        );
        ensure!(
            initial.weight_live_bytes == PACKED_BYTES
                && initial.uploaded_weight_bytes == PACKED_BYTES
        );
        ensure!(initial.tensor_live_bytes == TENSOR_BYTES);
        ensure!(initial.auxiliary_live_bytes == AUXILIARY_PER_BATCH * CASES.len() as u64);
        ensure!(initial.weight_peak_bytes == initial.weight_live_bytes);
        ensure!(initial.tensor_peak_bytes == initial.tensor_live_bytes);
        ensure!(initial.auxiliary_peak_bytes == initial.auxiliary_live_bytes);
        ensure!(
            initial.dispatch_attempts == 0
                && initial.dispatch_count == 0
                && initial.logical_submissions == 0
                && initial.host_read_bytes == 0
                && !initial.poisoned
        );
        let mut maximum_error = 0.0_f64;
        for (index, case) in CASES.into_iter().enumerate() {
            for round in 0..ROUNDS {
                device.write_f32(&chain.input.full_view(), &inputs(round))?;
                device.write_f32(&chain.output.full_view(), &residuals(round))?;
                let before = device.stats()?;
                println!("RUN FFN case={} round={round} limit={} scale={} semantic_commands=4 physical_dispatches=5", case.name, case.limit, case.scale);
                let batch_alias = chain.batches[index].clone();
                let submitted = device.submit(&batch_alias)?;
                device.wait(&submitted)?;
                let submission_alias = submitted.clone();
                device.wait(&submission_alias)?;
                let after = device.stats()?;
                verify_submission(before, after, DISPATCHES_PER_BATCH)?;
                ensure!(budget.current_bytes() == initial.allocation_bytes);
                maximum_error = maximum_error.max(verify_stages(&device, &chain, case, round)?);
                ensure!(
                    device.stats()?.host_read_bytes
                        == after.host_read_bytes + (3 * WIDTH * 4) as u64
                );
                println!("PASS FFN case={} round={round} checked_stage_values={} submission_tensor_transfer_bytes=0", case.name, 3 * WIDTH);
            }
        }
        let final_stats = device.stats()?;
        let batches = (CASES.len() * ROUNDS) as u64;
        ensure!(
            final_stats.dispatch_count == batches * DISPATCHES_PER_BATCH
                && final_stats.dispatch_attempts == final_stats.dispatch_count
        );
        ensure!(final_stats.logical_submissions == batches);
        ensure!(final_stats.host_read_bytes == batches * (3 * WIDTH * 4) as u64);
        ensure!(
            final_stats.weight_peak_bytes == initial.weight_peak_bytes
                && final_stats.tensor_peak_bytes == initial.tensor_peak_bytes
        );
        ensure!(final_stats.auxiliary_peak_bytes == initial.auxiliary_peak_bytes);
        drop(chain);
        let retired = device.stats()?;
        ensure!(
            retired.weight_live_bytes == 0
                && retired.tensor_live_bytes == 0
                && retired.auxiliary_live_bytes == 0
        );
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "FFN proof retained HAG allocations"
        );
        println!("PASS resident FFN: cases={} rounds_per_case={ROUNDS} semantic_commands={} dispatches={} checked_stage_values={} packed_bytes={PACKED_BYTES} public_tensor_bytes={TENSOR_BYTES} auxiliary_peak_bytes={} resident_bytes={} verification_readback_bytes={} max_final_absolute_error={maximum_error} budget_after=0", CASES.len(), batches * 4, final_stats.dispatch_count, batches * (3 * WIDTH) as u64, final_stats.auxiliary_peak_bytes, initial.allocation_bytes, final_stats.host_read_bytes);
        println!("Finite-fixture tolerance qualification only; per-operator fences, not a single IB, full model, or throughput claim.");
        Ok(())
    }

    pub(super) fn diagnose_first(path: &Path) -> Result<()> {
        announce_tolerances();
        verify_cpu_diagnostic()?;
        let budget = MemoryBudget::new(BUDGET_BYTES);
        let device = open(path, &budget)?;
        let gate = device.upload_f32(&DIAGNOSTIC_GATE)?;
        let up = device.upload_f32(&DIAGNOSTIC_UP)?;
        let mut commands = CommandList::new();
        commands.swiglu(&gate.full_view(), &up.full_view(), 0.75)?;
        let batch = device.record(&commands)?;
        drop(commands);
        let before = device.stats()?;
        ensure!(before.auxiliary_live_bytes == 4 && before.uploaded_weight_bytes == 0);
        ensure!(before.weight_live_bytes == 0);
        ensure!(before.tensor_live_bytes == (DIAGNOSTIC_GATE.len() * 2 * 4) as u64);
        ensure!(
            before.allocation_bytes > 0
                && before.allocation_bytes == budget.current_bytes()
                && before.allocation_bytes <= BUDGET_BYTES
        );
        ensure!(
            before.dispatch_attempts == 0
                && before.dispatch_count == 0
                && before.logical_submissions == 0
                && before.host_read_bytes == 0
                && !before.poisoned
        );
        println!(
            "RUN isolated SwiGLU elements={} limit=0.75 dispatches=1",
            DIAGNOSTIC_GATE.len()
        );
        let submitted = device.submit(&batch)?;
        device.wait(&submitted)?;
        let after = device.stats()?;
        verify_submission(before, after, 1)?;
        ensure!(budget.current_bytes() == before.allocation_bytes);
        let mut actual = [f32::NAN; DIAGNOSTIC_GATE.len()];
        device.read_f32(&gate.full_view(), &mut actual)?;
        let error = compare(
            "native diagnostic",
            &actual,
            &diagnostic_reference(),
            ACTIVATION_TOLERANCE,
        )?;
        let final_stats = device.stats()?;
        ensure!(final_stats.host_read_bytes == (DIAGNOSTIC_GATE.len() * 4) as u64);
        drop((submitted, batch, gate, up));
        let retired = device.stats()?;
        ensure!(retired.tensor_live_bytes == 0 && retired.auxiliary_live_bytes == 0);
        drop(device);
        ensure!(budget.current_bytes() == 0);
        println!("PASS isolated SwiGLU: dispatches=1 checked_values={} max_absolute_error={error} verification_readback_bytes={} budget_after=0", DIAGNOSTIC_GATE.len(), final_stats.host_read_bytes);
        println!("Diagnostic only: no scaled-add, composed FFN, queue-reuse, or general numerical qualification.");
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn finite_ffn_fixtures_match_cpu_before_native_access() {
            verify_cpu(&fixture().unwrap()).unwrap();
        }

        #[test]
        fn clamp_oracle_is_asymmetric_and_preserves_zero_gate() {
            assert!((activation(-8.0, -3.0, 0.75) - activation(-8.0, -0.75, 0.0)).abs() < 1e-15);
            assert!((activation(-8.0, -3.0, 0.75) - activation(-0.75, -0.75, 0.0)).abs() > 0.01);
            assert_eq!(activation(8.0, 3.0, 0.75), activation(0.75, 0.75, 0.0));
            assert_eq!(activation(0.0, 4.0, 0.75), 0.0);
            assert_eq!(activation(-0.0, 4.0, 0.75).to_bits(), (-0.0_f64).to_bits());
        }

        #[test]
        fn changed_inputs_and_residuals_change_expected_results() {
            assert_ne!(inputs(0), inputs(1));
            assert_ne!(residuals(0), residuals(1));
            for case in CASES {
                assert_ne!(reference(case, 0).unwrap().2, reference(case, 1).unwrap().2);
            }
        }

        #[test]
        fn numerical_checker_rejects_nonfinite_wrong_zero_and_wrong_sign() {
            for observed in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, 1e-8] {
                assert!(compare("bad", &[observed], &[-1e-8], ACTIVATION_TOLERANCE).is_err());
            }
            assert!(compare("bad zero", &[1e-8], &[0.0], ACTIVATION_TOLERANCE).is_err());
            assert!(compare("bad zero sign", &[0.0], &[-0.0], ACTIVATION_TOLERANCE).is_err());
            assert!(compare("bad zero sign", &[-0.0], &[0.0], ACTIVATION_TOLERANCE).is_err());
            assert!(compare("bad oracle", &[1.0], &[f64::NAN], ACTIVATION_TOLERANCE).is_err());
        }

        #[test]
        fn cpu_tiny_gate_activation_times_huge_up_does_not_collapse_to_zero() {
            // CPU-only edge: exp(-gate) remains finite f32, but its reciprocal
            // is subnormal. A GPU SFU reciprocal that flushes early would turn
            // the final ordinary-sized product into a false zero. These values
            // are deliberately NOT included in the native qualification scope.
            let gates = [-87.0_f32, -88.0, -88.5];
            let ups = [1e30_f32, -1e30, 1e30];
            let expected: Vec<_> = gates
                .iter()
                .zip(ups)
                .map(|(gate, up)| activation(f64::from(*gate), f64::from(up), 0.0))
                .collect();
            assert!(expected.iter().all(|value| value.abs() > 1e-8));
            let cpu = CpuExecutionDevice::new();
            let gate = cpu.upload_f32(&gates).unwrap();
            let up = cpu.upload_f32(&ups).unwrap();
            let mut commands = CommandList::new();
            commands
                .swiglu(&gate.full_view(), &up.full_view(), 0.0)
                .unwrap();
            let submitted = cpu.submit(&cpu.record(&commands).unwrap()).unwrap();
            cpu.wait(&submitted).unwrap();
            let mut actual = [f32::NAN; 3];
            cpu.read_f32(&gate.full_view(), &mut actual).unwrap();
            compare(
                "CPU subnormal-reciprocal edge",
                &actual,
                &expected,
                Tolerance {
                    absolute: 1e-12,
                    relative: 5e-6,
                },
            )
            .unwrap();
            assert!(compare("false FTZ zero", &[0.0; 3], &expected, ACTIVATION_TOLERANCE).is_err());
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();
    if arguments.len() == 1 && (arguments[0] == "--help" || arguments[0] == "-h") {
        println!("Usage: hag_ffn_probe --execute|--diagnose-first /dev/dri/renderD128");
        println!("--execute: 30 native dispatches across 6 submissions of 3 reused FFN batches; 256 KiB HAG cap.");
        println!("--diagnose-first: one 8-element SwiGLU dispatch; no linear/FFN dispatches.");
        println!(
            "Use an external runtime/process limit. No native access without either explicit flag."
        );
        return Ok(());
    }
    let [flag, path] = arguments.as_slice() else {
        anyhow::bail!("usage: hag_ffn_probe --execute|--diagnose-first /dev/dri/renderD128");
    };
    match flag.to_str() {
        Some("--execute") => proof::run(std::path::Path::new(path)),
        Some("--diagnose-first") => proof::diagnose_first(std::path::Path::new(path)),
        _ => anyhow::bail!("native execution requires explicit --execute or --diagnose-first"),
    }
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native FFN qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
