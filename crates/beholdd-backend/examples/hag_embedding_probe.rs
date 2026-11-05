//! Explicit, bounded Q8_0 row-decode qualification; never selected by serving.
//! All independent references precede device access. External process, thermal
//! and crash-capture supervision is required. Every checked value must match bits.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{ensure, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::{CommandList, ExecutionDevice, RecordedBatch};
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice,
    };
    use beholdd_backend::{Device, Weight};
    use beholdd_format::GgmlType;
    use std::path::Path;

    const GUARD: f32 = -12345.5;
    const BUDGET: u64 = 1024 * 1024;
    const REPLAYS: usize = 2;
    const DISPATCHES: u64 = 16;
    const CHECKED: usize = 17_152;
    const GUARDS: usize = 32;
    const READBACK: u64 = 68_736;
    const ALLOCATION: u64 = 90_112;

    fn config() -> NativeExecutionConfig {
        NativeExecutionConfig {
            weight_capacity_bytes: 16 * 1024,
            tensor_capacity_bytes: 32 * 1024,
            auxiliary_capacity_bytes: 4096,
        }
    }

    struct Fixture {
        rows: usize,
        cols: usize,
        packed: Vec<u8>,
        selected: Vec<usize>,
        expected: Vec<Vec<f32>>,
    }

    fn finite_half(bits: u16) -> f32 {
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = (bits >> 10) & 31;
        let fraction = bits & 1023;
        assert!(exponent < 31);
        if exponent == 0 {
            if fraction == 0 {
                return f32::from_bits(sign);
            }
            let value = f32::from(fraction) * (1.0 / 16_777_216.0);
            if sign == 0 {
                value
            } else {
                -value
            }
        } else {
            f32::from_bits(sign | ((u32::from(exponent) + 112) << 23) | (u32::from(fraction) << 13))
        }
    }

    fn fixtures() -> Result<Vec<Fixture>> {
        let cpu = Cpu::new();
        let mut fixtures = Vec::new();
        for (rows, cols, selected) in [
            (3, 32, vec![0, 1, 2]),
            (4, 96, vec![0, 1, 3]),
            (2, 4096, vec![0, 1]),
        ] {
            let scales = [
                0x3c00u16, 0xbc00, 0x0000, 0x8000, 0x0001, 0x8001, 0x0400, 0x3555, 0x7bff,
            ];
            let quants = [-128i8, -127, -1, 0, 1, 127];
            let mut packed = Vec::new();
            for row in 0..rows {
                for block in 0..cols / 32 {
                    packed
                        .extend_from_slice(&scales[(row * 3 + block) % scales.len()].to_le_bytes());
                    for column in 0..32 {
                        packed.push(quants[(row + block + column) % quants.len()] as u8);
                    }
                }
            }
            let stride = cols / 32 * 34;
            let mut expected = Vec::new();
            for &row in &selected {
                let encoded = &packed[row * stride..(row + 1) * stride];
                let reference: Vec<f32> = encoded
                    .chunks_exact(34)
                    .flat_map(|block| {
                        let scale = finite_half(u16::from_le_bytes([block[0], block[1]]));
                        block[2..].iter().map(move |q| scale * f32::from(*q as i8))
                    })
                    .collect();
                let mut actual = vec![f32::NAN; cols];
                cpu.dequantize(
                    &mut actual,
                    &Weight {
                        bytes: encoded,
                        kind: GgmlType(8),
                        rows: 1,
                        cols,
                    },
                )?;
                ensure!(
                    actual
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "independent row reference differs from permanent CPU"
                );
                ensure!(reference.iter().all(|v| v.is_finite()), "nonfinite fixture");
                expected.push(reference);
            }
            fixtures.push(Fixture {
                rows,
                cols,
                packed,
                selected,
                expected,
            });
        }
        Ok(fixtures)
    }

    fn compare(actual: &[f32], expected: &[f32]) -> Result<()> {
        ensure!(actual.len() == expected.len() + 2, "guarded extent differs");
        ensure!(
            actual[0].to_bits() == GUARD.to_bits()
                && actual[actual.len() - 1].to_bits() == GUARD.to_bits(),
            "output guard changed"
        );
        for (index, (a, b)) in actual[1..actual.len() - 1].iter().zip(expected).enumerate() {
            ensure!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "row output {index}: {a:?} != {b:?}"
            );
        }
        Ok(())
    }

    fn submit(device: &NativeExecutionDevice, batch: &RecordedBatch) -> Result<()> {
        let before = device.stats()?;
        let completion = device.submit(batch)?;
        device.wait(&completion)?;
        let after = device.stats()?;
        ensure!(
            after.dispatch_count == before.dispatch_count + 1
                && after.dispatch_attempts == before.dispatch_attempts + 1
                && after.logical_submissions == before.logical_submissions + 1,
            "dispatch accounting differs"
        );
        ensure!(
            after.host_read_bytes == before.host_read_bytes
                && after.host_write_bytes == before.host_write_bytes
                && after.uploaded_weight_bytes == before.uploaded_weight_bytes,
            "submission transferred payloads"
        );
        ensure!(
            after.allocation_bytes == before.allocation_bytes
                && after.weight_live_bytes == before.weight_live_bytes
                && after.tensor_live_bytes == before.tensor_live_bytes
                && after.auxiliary_live_bytes == before.auxiliary_live_bytes
                && after.weight_peak_bytes == before.weight_peak_bytes
                && after.tensor_peak_bytes == before.tensor_peak_bytes
                && after.auxiliary_peak_bytes == before.auxiliary_peak_bytes
                && !after.poisoned,
            "submission changed residency or poisoned queue"
        );
        Ok(())
    }

    fn run(path: &Path) -> Result<()> {
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(std::ffi::OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly 0 before device access"
        );
        let fixtures = fixtures()?;
        let config = config();
        ensure!(
            config.allocation_bytes()? == ALLOCATION,
            "queue preflight differs"
        );
        let budget = MemoryBudget::new(BUDGET);
        let device = NativeExecutionDevice::open(path, &budget, config)?;
        let initial = device.stats()?;
        ensure!(
            initial.allocation_bytes == ALLOCATION
                && budget.current_bytes() == ALLOCATION
                && initial.dispatch_attempts == 0
                && initial.host_read_bytes == 0
                && !initial.poisoned,
            "initial queue differs"
        );
        let mut checked = 0usize;
        let mut guards = 0usize;
        for (case, fixture) in fixtures.iter().enumerate() {
            let output = device.allocate_f32(fixture.cols + 2)?;
            let view = output.view(1, fixture.cols)?;
            let encoded = device.upload_weight(&Weight {
                bytes: &fixture.packed,
                kind: GgmlType(8),
                rows: fixture.rows,
                cols: fixture.cols,
            })?;
            let prepared = device.prepare_weight(&encoded)?;
            let batches = fixture
                .selected
                .iter()
                .map(|&row| {
                    let mut commands = CommandList::new();
                    commands.embedding_row(&view, &prepared, row)?;
                    device.record(&commands)
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            drop(prepared);
            drop(encoded);
            for ((&row, expected), batch) in
                fixture.selected.iter().zip(&fixture.expected).zip(&batches)
            {
                for replay in 0..REPLAYS {
                    let mut sentinel = vec![f32::NAN; fixture.cols + 2];
                    sentinel[0] = GUARD;
                    sentinel[fixture.cols + 1] = GUARD;
                    device.write_f32(&output.full_view(), &sentinel)?;
                    submit(&device, batch)?;
                    let mut actual = vec![f32::NAN; fixture.cols + 2];
                    device.read_f32(&output.full_view(), &mut actual)?;
                    compare(&actual, expected)?;
                    checked += expected.len();
                    guards += 2;
                    println!("PASS embedding case={case} row={row} replay={replay} checked_values={} guards=2", fixture.cols);
                }
            }
            drop(batches);
            drop(view);
            drop(output);
            let retired = device.stats()?;
            ensure!(
                retired.weight_live_bytes == 0
                    && retired.tensor_live_bytes == 0
                    && retired.auxiliary_live_bytes == 0
                    && !retired.poisoned,
                "fixture retained allocations"
            );
        }
        let final_stats = device.stats()?;
        ensure!(
            checked == CHECKED
                && guards == GUARDS
                && final_stats.dispatch_count == DISPATCHES
                && final_stats.dispatch_attempts == DISPATCHES
                && final_stats.logical_submissions == DISPATCHES
                && final_stats.host_read_bytes == READBACK
                && budget.current_bytes() == ALLOCATION,
            "final diagnostic accounting differs"
        );
        drop(device);
        ensure!(budget.current_bytes() == 0, "queue charge did not retire");
        println!("{{\"execution\":\"native-hag-embedding-diagnostic\",\"physical_dispatches\":{DISPATCHES},\"logical_submissions\":{DISPATCHES},\"checked_values\":{checked},\"guard_checks\":{guards},\"exact\":true,\"max_absolute_error\":0,\"absolute_tolerance\":0,\"relative_tolerance\":0,\"actual_hag_allocation_bytes\":{ALLOCATION},\"expected_hag_allocation_bytes\":{ALLOCATION},\"hag_budget_bytes\":{BUDGET},\"verification_readback_bytes\":{READBACK},\"submission_tensor_transfer_bytes\":0,\"budget_after\":0,\"required_hag_trace_submission\":\"0\",\"hag_trace_requirement_verified\":true}}");
        Ok(())
    }

    pub fn main() -> Result<()> {
        let arguments: Vec<_> = std::env::args_os().skip(1).collect();
        if arguments.is_empty() || arguments == [std::ffi::OsString::from("--help")] {
            println!("hag_embedding_probe --diagnose /dev/dri/renderD128\nExplicit native gate, HAG_TRACE_SUBMISSION=0 and external process/thermal/crash supervision required.");
            return Ok(());
        }
        ensure!(
            arguments.len() == 2 && arguments[0] == "--diagnose",
            "expected --diagnose RENDER_NODE"
        );
        run(Path::new(&arguments[1]))
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn preregistered_rows_cover_alignment_tails_range_and_signed_zero() {
            let cases = fixtures().unwrap();
            assert_eq!(
                cases
                    .iter()
                    .map(|c| c.expected.len() * REPLAYS)
                    .sum::<usize>(),
                DISPATCHES as usize
            );
            assert_eq!(
                cases
                    .iter()
                    .map(|c| c.expected.len() * c.cols * REPLAYS)
                    .sum::<usize>(),
                CHECKED
            );
            assert_eq!((CHECKED + GUARDS) as u64 * 4, READBACK);
            assert_eq!(config().allocation_bytes().unwrap(), ALLOCATION);
            assert_eq!(cases[1].cols / 32 * 34 % 4, 2);
            assert!(cases
                .iter()
                .flat_map(|c| &c.expected)
                .flatten()
                .any(|v| v.to_bits() == 0x8000_0000));
            assert!(cases
                .iter()
                .flat_map(|c| &c.expected)
                .flatten()
                .any(|v| v.abs() == 1.0 / 16_777_216.0));
            for case in cases {
                assert!(case.expected.windows(2).all(|p| p[0]
                    .iter()
                    .zip(&p[1])
                    .any(|(a, b)| a.to_bits() != b.to_bits())));
            }
        }
        #[test]
        fn exact_comparison_rejects_stale_values_guards_signs_and_nans() {
            let expected = [1.0, -0.0, -128.0];
            let correct = [GUARD, 1.0, -0.0, -128.0, GUARD];
            compare(&correct, &expected).unwrap();
            for index in 0..correct.len() {
                let mut wrong = correct;
                wrong[index] = 0.0;
                assert!(compare(&wrong, &expected).is_err());
            }
            let mut nan = correct;
            nan[1] = f32::NAN;
            assert!(compare(&nan, &expected).is_err());
            assert!(compare(&correct[..4], &expected).is_err());
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::main()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("native HAG qualification requires Linux");
    std::process::exit(2);
}
