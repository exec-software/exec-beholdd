//! SOURCE STAGE: fixed large packed tiles, no model and no runtime admission.
//! Generate, independently check, and drop one fixture at a time. Native submits
//! only after the complete CPU reference stage passes. External supervision is
//! mandatory; cooperative180s does not interrupt an in-flight wait.
#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{ensure, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::{
        CommandList, ExecutionDevice, LinearMode, PreparedWeight, RecordedBatch,
    };
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionProfile,
        NativeExecutionStats,
    };
    use beholdd_backend::{Device, Weight};
    use beholdd_format::GgmlType;
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};
    const MIB: u64 = 1024 * 1024;
    const GPU_BUDGET: u64 = 32 * MIB;
    const CPU_PAYLOAD_BOUND: u64 = 160 * MIB;
    const EXPECTED_ALLOCATION: u64 = 16_961_536;
    const SHAPES: [(usize, usize); 4] = [(4096, 256), (3276, 5120), (2730, 6144), (963, 17408)];
    const GUARD: f32 = -12345.5;
    const REPLAYS: usize = 2;
    const EXPECTED_DISPATCHES: u64 = 16;
    const EXPECTED_LOGICAL: u64 = 16;
    const EXPECTED_VALUES: u64 = 44_260;
    const EXPECTED_CPU_VALUES: u64 = 102_761_700;
    const EXPECTED_GUARDS: u64 = 32;
    const EXPECTED_READBACK: u64 = 177_168;
    const EXPECTED_UPLOAD: u64 = 71_019_480;
    const EXPECTED_WRITES: u64 = 71_979_504;
    const WEIGHT_PEAK: u64 = 13_759_200;
    const TENSOR_PEAK: u64 = 73_492;
    const AUXILIARY_PEAK: u64 = 0;
    fn config() -> NativeExecutionConfig {
        NativeExecutionConfig {
            weight_capacity_bytes: 16 * MIB as u32,
            tensor_capacity_bytes: 128 * 1024,
            auxiliary_capacity_bytes: 4,
        }
    }
    enum Mode {
        Cpu,
        Native(PathBuf),
        Help,
    }

    fn mode(arguments: &[OsString]) -> Result<Mode> {
        match arguments {
            [flag] if flag == "--cpu-reference" => Ok(Mode::Cpu),
            [flag] if flag == "--help" || flag == "-h" => Ok(Mode::Help),
            [flag, path] if flag == "--native" => {
                let path = PathBuf::from(path);
                let name = path.file_name().and_then(OsStr::to_str).unwrap_or("");
                let suffix = name.strip_prefix("renderD").unwrap_or("");
                ensure!(
                    path.parent() == Some(Path::new("/dev/dri"))
                        && !suffix.is_empty()
                        && suffix.bytes().all(|b| b.is_ascii_digit()),
                    "--native requires an explicit /dev/dri/renderD... node"
                );
                Ok(Mode::Native(path))
            }
            _ => anyhow::bail!(
                "expected --cpu-reference or --native /dev/dri/renderD...; use --help"
            ),
        }
    }

    fn check_time(deadline: Instant) -> Result<()> {
        ensure!(
            Instant::now() < deadline,
            "180-second cooperative probe deadline exceeded"
        );
        Ok(())
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
            let magnitude = f32::from(fraction) * (1.0 / 16_777_216.0);
            if sign == 0 {
                magnitude
            } else {
                -magnitude
            }
        } else {
            f32::from_bits(sign | ((u32::from(exponent) + 112) << 23) | (u32::from(fraction) << 13))
        }
    }

    struct Fixture {
        kind: GgmlType,
        rows: usize,
        cols: usize,
        packed: Vec<u8>,
        semantic: Vec<f32>,
        inputs: [Vec<f32>; REPLAYS],
        expected: [Vec<f32>; REPLAYS],
    }

    impl Fixture {
        fn weight(&self) -> Weight<'_> {
            Weight {
                bytes: &self.packed,
                kind: self.kind,
                rows: self.rows,
                cols: self.cols,
            }
        }
        fn payload_bytes(&self) -> usize {
            self.packed.len()
                + 4 * (self.semantic.len()
                    + self.inputs.iter().map(Vec::len).sum::<usize>()
                    + self.expected.iter().map(Vec::len).sum::<usize>())
        }
    }

    // Encode semantic scales/codes, then compute expected weights from those same
    // original values. Expected values never decode the generated packed bytes.
    fn fixture(kind: u32, rows: usize, cols: usize, deadline: Instant) -> Result<Fixture> {
        ensure!(
            matches!(kind, 12 | 14) && SHAPES.contains(&(rows, cols)),
            "fixture shape/format outside fixed plan"
        );
        check_time(deadline)?;
        let packed_bytes = rows * (cols / 256) * if kind == 12 { 144 } else { 210 };
        // Retained fixture + one full CPU decode/matvec scratch + inputs/results
        // plus16MiB transfer/allocator/library payload headroom, before allocation.
        let forecast = packed_bytes + 8 * rows * cols + 8 * (cols + rows) + 16 * MIB as usize;
        ensure!(
            forecast as u64 <= CPU_PAYLOAD_BOUND,
            "fixture payload exceeds160MiB"
        );
        const HALVES: [u16; 9] = [
            0x3c00, 0xbc00, 0x0001, 0x8001, 0x0400, 0x3555, 0x3800, 0x8000, 0x7bff,
        ];
        const Q4_SCALES: [u8; 8] = [0, 1, 15, 16, 31, 32, 47, 63];
        const Q6_SCALES: [i8; 16] = [
            -128, -97, -63, -32, -1, 0, 1, 17, 31, 63, 95, 127, -2, 3, -4, 5,
        ];
        let stride = if kind == 12 { 144 } else { 210 };
        let mut packed = Vec::with_capacity(rows * (cols / 256) * stride);
        let mut semantic = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            check_time(deadline)?;
            for block in 0..cols / 256 {
                let d_bits = HALVES[(row + block) % HALVES.len()];
                let d = finite_half(d_bits);
                if kind == 12 {
                    let m_bits = HALVES[(row * 3 + block + 5) % HALVES.len()];
                    let minimum = finite_half(m_bits);
                    let scales: [u8; 8] = std::array::from_fn(|j| Q4_SCALES[(j + row + block) % 8]);
                    let minima: [u8; 8] =
                        std::array::from_fn(|j| Q4_SCALES[(7 - j + row + block * 3) % 8]);
                    let mut bytes = [0u8; 144];
                    bytes[..2].copy_from_slice(&d_bits.to_le_bytes());
                    bytes[2..4].copy_from_slice(&m_bits.to_le_bytes());
                    for group in 0..8 {
                        if group < 4 {
                            bytes[4 + group] |= scales[group];
                            bytes[8 + group] |= minima[group];
                        } else {
                            bytes[group] |= (scales[group] >> 4) << 6;
                            bytes[4 + group] |= (minima[group] >> 4) << 6;
                            bytes[8 + group] = (scales[group] & 15) | ((minima[group] & 15) << 4);
                        }
                    }
                    for index in 0..256 {
                        let group = index / 32;
                        let code = ((index * 7 + row * 3 + block * 5) % 16) as u8;
                        bytes[16 + (group / 2) * 32 + index % 32] |= code << (4 * (group % 2));
                        semantic.push(
                            (d * f32::from(scales[group])) * f32::from(code)
                                - minimum * f32::from(minima[group]),
                        );
                    }
                    packed.extend_from_slice(&bytes);
                } else {
                    let scales: [i8; 16] =
                        std::array::from_fn(|j| Q6_SCALES[(j + row + block) % 16]);
                    let mut bytes = [0u8; 210];
                    for (j, scale) in scales.iter().enumerate() {
                        bytes[192 + j] = *scale as u8;
                    }
                    bytes[208..].copy_from_slice(&d_bits.to_le_bytes());
                    for index in 0..256 {
                        let half = index / 128;
                        let group = (index % 128) / 32;
                        let lane = index % 32;
                        let code = ((index * 13 + row * 3 + block * 7) % 64) as u8;
                        bytes[half * 64 + (group % 2) * 32 + lane] |=
                            (code & 15) << (4 * (group / 2));
                        bytes[128 + half * 32 + lane] |= (code >> 4) << (2 * group);
                        semantic.push(
                            (d * f32::from(scales[index / 16])) * f32::from(i16::from(code) - 32),
                        );
                    }
                    packed.extend_from_slice(&bytes);
                }
            }
        }
        let inputs = [
            (0..cols)
                .map(|i| ((i % 17) as i32 - 8) as f32 / 16.0)
                .collect::<Vec<_>>(),
            (0..cols)
                .map(|i| (((i * 3 + 5) % 23) as i32 - 11) as f32 / 32.0 + 0.125)
                .collect::<Vec<_>>(),
        ];
        let mut expected = [Vec::with_capacity(rows), Vec::with_capacity(rows)];
        for replay in 0..REPLAYS {
            for row in semantic.chunks_exact(cols) {
                check_time(deadline)?;
                let mut sum = 0.0f32;
                for (&weight, &input) in row.iter().zip(&inputs[replay]) {
                    let product = weight * input;
                    sum += product;
                }
                expected[replay].push(sum);
            }
        }
        ensure!(
            semantic
                .iter()
                .chain(expected.iter().flatten())
                .all(|v| v.is_finite()),
            "nonfinite synthetic reference"
        );
        ensure!(
            expected[0]
                .iter()
                .zip(&expected[1])
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "changed inputs produced stale-reference fixture"
        );
        Ok(Fixture {
            kind: GgmlType(kind),
            rows,
            cols,
            packed,
            semantic,
            inputs,
            expected,
        })
    }

    fn exact(actual: &[f32], expected: &[f32]) -> Result<()> {
        ensure!(actual.len() == expected.len(), "reference extent differs");
        for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            ensure!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "exact reference index {index}: {a:?} != {b:?}"
            );
        }
        Ok(())
    }

    fn references(deadline: Instant) -> Result<u64> {
        let mut compared = 0u64;
        for kind in [12, 14] {
            for (rows, cols) in SHAPES {
                check_time(deadline)?;
                let case = fixture(kind, rows, cols, deadline)?;
                ensure!(
                    (case.payload_bytes() + rows * cols * 4 + 16 * MIB as usize) as u64
                        <= CPU_PAYLOAD_BOUND,
                    "actual fixture payload exceeds preflight"
                );
                let mut decoded = vec![f32::NAN; rows * cols];
                Cpu.dequantize(&mut decoded, &case.weight())?;
                exact(&decoded, &case.semantic)?;
                compared += decoded.len() as u64;
                drop(decoded);
                for replay in 0..REPLAYS {
                    check_time(deadline)?;
                    let mut output = vec![f32::NAN; rows];
                    Cpu.matvec(&mut output, &case.inputs[replay], &case.weight())?;
                    exact(&output, &case.expected[replay])?;
                    compared += rows as u64;
                }
                drop(case);
            }
        }
        ensure!(
            compared == EXPECTED_CPU_VALUES,
            "CPU fixture comparison count differs"
        );
        check_time(deadline)?;
        Ok(compared)
    }

    #[derive(Default)]
    struct Receipt {
        physical: u64,
        logical: u64,
        compared: u64,
        cpu_compared: u64,
        guards: u64,
        readback: u64,
        max_error: f64,
        allocation: u64,
        expected_allocation: u64,
        budget_current: u64,
        weight_peak: u64,
        tensor_peak: u64,
        auxiliary_peak: u64,
        uploaded: u64,
        host_writes: u64,
    }

    impl Receipt {
        fn print(&self, execution: &str, passed: bool, elapsed: f64) {
            println!("{{\"execution\":\"{execution}\",\"passed\":{passed},\"physical_dispatches\":{},\"logical_submissions\":{},\"compared_values\":{},\"cpu_reference_values\":{},\"guard_checks\":{},\"readback_bytes\":{},\"max_absolute_error\":{},\"allocation_bytes\":{},\"expected_allocation_bytes\":{},\"hag_budget_bytes\":{GPU_BUDGET},\"budget_current_bytes\":{},\"budget_after\":{},\"elapsed_seconds\":{elapsed},\"cpu_payload_bound_bytes\":{CPU_PAYLOAD_BOUND},\"weight_peak_bytes\":{},\"tensor_peak_bytes\":{},\"auxiliary_peak_bytes\":{},\"uploaded_weight_bytes\":{},\"host_write_bytes\":{},\"row_matvec_exact\":true,\"fixture_count\":8,\"large_runtime_tiles\":true}}",
                self.physical,self.logical,self.compared,self.cpu_compared,self.guards,self.readback,self.max_error,
                self.allocation,self.expected_allocation,self.budget_current,self.budget_current,self.weight_peak,
                self.tensor_peak,self.auxiliary_peak,self.uploaded,self.host_writes);
        }
    }

    fn residency(
        device: &NativeExecutionDevice,
        weight: u64,
        tensor: u64,
        auxiliary: u64,
    ) -> Result<()> {
        let stats = device.stats()?;
        ensure!(
            stats.weight_live_bytes == weight
                && stats.tensor_live_bytes == tensor
                && stats.auxiliary_live_bytes == auxiliary
                && !stats.poisoned,
            "resident live bytes differ: {stats:?}"
        );
        Ok(())
    }

    fn submit(
        device: &NativeExecutionDevice,
        batch: &RecordedBatch,
        dispatches: u64,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        check_time(deadline)?;
        let before = device.stats()?;
        let completion = device.submit(batch)?;
        device.wait(&completion)?;
        let after = device.stats()?;
        ensure!(
            after.dispatch_count == before.dispatch_count + dispatches
                && after.dispatch_attempts == before.dispatch_attempts + dispatches
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
        receipt.physical += dispatches;
        receipt.logical += 1;
        check_time(deadline)
    }

    fn guarded(expected: usize) -> Vec<f32> {
        let mut values = vec![f32::NAN; expected + 2];
        values[0] = GUARD;
        values[expected + 1] = GUARD;
        values
    }

    fn compare(actual: &[f32], expected: &[f32], exact_bits: bool) -> Result<f64> {
        ensure!(actual.len() == expected.len() + 2, "guarded extent differs");
        ensure!(
            actual[0].to_bits() == GUARD.to_bits()
                && actual[actual.len() - 1].to_bits() == GUARD.to_bits(),
            "output guard changed"
        );
        let mut maximum = 0.0f64;
        for (index, (&a, &b)) in actual[1..actual.len() - 1].iter().zip(expected).enumerate() {
            let difference = (f64::from(a) - f64::from(b)).abs();
            ensure!(
                a.is_finite()
                    && b.is_finite()
                    && if exact_bits {
                        a.to_bits() == b.to_bits()
                    } else {
                        false // This probe has no approximate comparison mode.
                    },
                "output {index}: {a:?} != {b:?}, error {difference}, exact={exact_bits}"
            );
            maximum = maximum.max(difference);
        }
        Ok(maximum)
    }

    fn read_compare(
        device: &NativeExecutionDevice,
        output: &beholdd_backend::execution::F32Tensor,
        expected: &[f32],
        exact_bits: bool,
        receipt: &mut Receipt,
    ) -> Result<()> {
        let before = device.stats()?;
        let mut actual = vec![f32::NAN; expected.len() + 2];
        device.read_f32(&output.full_view(), &mut actual)?;
        let after = device.stats()?;
        ensure!(
            after.host_read_bytes == before.host_read_bytes + (actual.len() * 4) as u64
                && after.host_write_bytes == before.host_write_bytes,
            "readback accounting differs"
        );
        receipt.readback += (actual.len() * 4) as u64;
        receipt.max_error = receipt
            .max_error
            .max(compare(&actual, expected, exact_bits)?);
        receipt.compared += expected.len() as u64;
        receipt.guards += 2;
        Ok(())
    }

    fn matvec(
        device: &NativeExecutionDevice,
        case: &Fixture,
        weight: &PreparedWeight,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        let input = device.allocate_f32(case.cols)?;
        let output = device.allocate_f32(case.rows + 2)?;
        let view = output.view(1, case.rows)?;
        let batch = {
            let mut commands = CommandList::new();
            commands.linear(&view, &input.full_view(), weight, LinearMode::Overwrite)?;
            device.record(&commands)?
        };
        residency(
            device,
            ((case.packed.len() + 3) & !3) as u64,
            ((case.cols + case.rows + 2) * 4) as u64,
            0,
        )?;
        for replay in 0..REPLAYS {
            check_time(deadline)?;
            device.write_f32(&input.full_view(), &case.inputs[replay])?;
            device.write_f32(&output.full_view(), &guarded(case.rows))?;
            submit(device, &batch, 1, deadline, receipt)?;
            read_compare(device, &output, &case.expected[replay], true, receipt)?;
        }
        Ok(())
    }

    fn native(path: &Path, deadline: Instant, receipt: &mut Receipt) -> Result<()> {
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly 0 before device access"
        );
        let configuration = config();
        let allocation =
            configuration.allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?;
        ensure!(
            allocation == EXPECTED_ALLOCATION
                && allocation < GPU_BUDGET
                && CPU_PAYLOAD_BOUND == 160 * MIB,
            "fixed allocation plan exceeds bounds"
        );
        receipt.expected_allocation = allocation;
        check_time(deadline)?;
        let budget = MemoryBudget::new(GPU_BUDGET);
        let result = (|| -> Result<()> {
            let device = NativeExecutionDevice::open_with_profile(
                path,
                &budget,
                configuration,
                NativeExecutionProfile::Qwen35Probe,
            )?;
            receipt.allocation = device.stats()?.allocation_bytes;
            receipt.budget_current = budget.current_bytes();
            ensure!(
                device.stats()?
                    == NativeExecutionStats {
                        allocation_bytes: allocation,
                        ..NativeExecutionStats::default()
                    }
                    && budget.current_bytes() == allocation,
                "initial queue differs"
            );
            for kind in [12, 14] {
                for (rows, cols) in SHAPES {
                    check_time(deadline)?;
                    let case = fixture(kind, rows, cols, deadline)?;
                    let before = device.stats()?;
                    let encoded = device.upload_weight(&case.weight())?;
                    let after_upload = device.stats()?;
                    ensure!(
                        after_upload.uploaded_weight_bytes
                            == before.uploaded_weight_bytes + case.packed.len() as u64
                            && after_upload.host_write_bytes
                                == before.host_write_bytes + case.packed.len() as u64,
                        "upload accounting differs"
                    );
                    let prepared = device.prepare_weight(&encoded)?;
                    ensure!(
                        device.stats()? == after_upload,
                        "preparation transferred or allocated packed data"
                    );
                    drop(encoded);
                    matvec(&device, &case, &prepared, deadline, receipt)?;
                    residency(&device, ((case.packed.len() + 3) & !3) as u64, 0, 0)?;
                    drop(prepared);
                    residency(&device, 0, 0, 0)?;
                    drop(case);
                }
            }
            residency(&device, 0, 0, 0)?;
            let final_stats = device.stats()?;
            receipt.weight_peak = final_stats.weight_peak_bytes;
            receipt.tensor_peak = final_stats.tensor_peak_bytes;
            receipt.auxiliary_peak = final_stats.auxiliary_peak_bytes;
            receipt.uploaded = final_stats.uploaded_weight_bytes;
            receipt.host_writes = final_stats.host_write_bytes;
            ensure!(
                receipt.physical == EXPECTED_DISPATCHES
                    && receipt.logical == EXPECTED_LOGICAL
                    && receipt.compared == EXPECTED_VALUES
                    && receipt.guards == EXPECTED_GUARDS
                    && receipt.readback == EXPECTED_READBACK,
                "comparison totals differ from fixed plan"
            );
            ensure!(
                final_stats.dispatch_count == EXPECTED_DISPATCHES
                    && final_stats.dispatch_attempts == EXPECTED_DISPATCHES
                    && final_stats.logical_submissions == EXPECTED_LOGICAL
                    && final_stats.host_read_bytes == EXPECTED_READBACK
                    && final_stats.uploaded_weight_bytes == EXPECTED_UPLOAD
                    && final_stats.host_write_bytes == EXPECTED_WRITES
                    && final_stats.weight_peak_bytes == WEIGHT_PEAK
                    && final_stats.tensor_peak_bytes == TENSOR_PEAK
                    && final_stats.auxiliary_peak_bytes == AUXILIARY_PEAK
                    && final_stats.allocation_bytes == allocation
                    && budget.current_bytes() == allocation
                    && !final_stats.poisoned,
                "final queue accounting differs: {final_stats:?}"
            );
            ensure!(
                EXPECTED_UPLOAD > u64::from(configuration.weight_capacity_bytes),
                "plan must require weight-arena reuse"
            );
            drop(device);
            Ok(())
        })();
        receipt.budget_current = budget.current_bytes();
        result?;
        ensure!(
            budget.current_bytes() == 0,
            "queue budget did not fully retire"
        );
        check_time(deadline)
    }

    pub fn main() -> Result<()> {
        let mode = mode(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
        if matches!(mode, Mode::Help) {
            println!("hag_kquant_large_tile_probe --cpu-reference | --native /dev/dri/renderD...\nFixed synthetic fixtures only. Native requires HAG_TRACE_SUBMISSION=0 and external process/thermal/crash supervision. Cooperative deadline180s; GPU budget32MiB, CPU payload160MiB. No serving registration or model discovery.");
            return Ok(());
        }
        let started = Instant::now();
        let deadline = started + Duration::from_secs(180);
        let mut receipt = Receipt::default();
        let execution = if matches!(mode, Mode::Cpu) {
            "cpu-kquant-large-tile-reference"
        } else {
            "native-hag-kquant-large-tile-diagnostic"
        };
        let result = (|| -> Result<()> {
            let compared = references(deadline)?;
            receipt.cpu_compared = compared;
            match mode {
                Mode::Cpu => {
                    receipt.compared = compared;
                    Ok(())
                }
                Mode::Native(path) => native(&path, deadline, &mut receipt),
                Mode::Help => unreachable!(),
            }
        })();
        receipt.print(execution, result.is_ok(), started.elapsed().as_secs_f64());
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        #[test]
        fn large_shapes_cover_runtime_tiles_and_full_row_boundary() {
            assert_eq!(SHAPES[0], (4096, 256));
            for &(rows, cols) in &SHAPES[1..] {
                assert_eq!(rows, (64 * 1024 * 1024) / (cols * 4));
                assert!(rows <= 4096 && cols <= 17408 && cols % 256 == 0);
            }
        }
        #[test]
        fn fixed_payload_forecast_precedes_allocation() {
            let mut cpu = 0;
            let mut upload = 0;
            let mut writes = 0;
            let mut reads = 0;
            let mut max_weight = 0;
            let mut max_tensor = 0;
            for block_bytes in [144, 210] {
                for (rows, cols) in SHAPES {
                    let packed = rows * (cols / 256) * block_bytes;
                    assert!(
                        (packed + 8 * rows * cols + 8 * (cols + rows) + 16 * MIB as usize) as u64
                            <= CPU_PAYLOAD_BOUND
                    );
                    cpu += rows * cols + 2 * rows;
                    upload += packed;
                    writes += packed + 3 * (rows + cols + 2) * 4;
                    reads += 2 * (rows + 2) * 4;
                    max_weight = max_weight.max((packed + 3) & !3);
                    max_tensor = max_tensor.max((rows + cols + 2) * 4);
                }
            }
            assert_eq!(cpu as u64, EXPECTED_CPU_VALUES);
            assert_eq!(upload as u64, EXPECTED_UPLOAD);
            assert_eq!(writes as u64, EXPECTED_WRITES);
            assert_eq!(reads as u64, EXPECTED_READBACK);
            assert_eq!(max_weight as u64, WEIGHT_PEAK);
            assert_eq!(max_tensor as u64, TENSOR_PEAK);
            assert_eq!(
                config()
                    .allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)
                    .unwrap(),
                EXPECTED_ALLOCATION
            );
        }
        #[test]
        fn unregistered_shape_refuses_before_fixture_allocation() {
            assert!(fixture(12, 4097, 256, Instant::now() + Duration::from_secs(1)).is_err());
            assert!(fixture(14, 4096, 17408, Instant::now() + Duration::from_secs(1)).is_err());
            assert!(matches!(
                mode(&[OsString::from("--cpu-reference")]).unwrap(),
                Mode::Cpu
            ));
            assert!(mode(&[]).is_err());
        }
    }
}
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::main()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("large native packed diagnostic requires Linux");
    std::process::exit(1);
}
