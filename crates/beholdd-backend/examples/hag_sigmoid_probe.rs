//! Fixed slot16 qualification; no model loading or serving registration.
//! Native mode requires external process, thermal and GPU-crash supervision.
//! Its cooperative180s deadline cannot interrupt a blocked native fence.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{ensure, Result};
    use beholdd_backend::execution::{
        CommandList, CpuExecutionDevice, ExecutionDevice, F32Tensor, RecordedBatch, Submission,
    };
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionProfile,
        NativeExecutionStats,
    };
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const LENGTHS: [usize; 3] = [1, 65, 4096];
    const REPLAYS: usize = 2;
    const GUARD: f32 = -12345.5;
    const ABS_TOL: f64 = 2e-7;
    const REL_TOL: f64 = 3e-5;
    const SMALL_THRESHOLD: f64 = 1e-4;
    const SMALL_REL_TOL: f64 = 3e-5;
    const GPU_BUDGET: u64 = 256 * 1024;
    const CPU_PAYLOAD_BOUND: u64 = 2 * 1024 * 1024;
    // Current17-program library:20KiB + command4KiB + guarded arenas12/72/12KiB.
    const EXPECTED_ALLOCATION: u64 = 122_880;
    const EXPECTED_DISPATCHES: u64 = 6;
    const EXPECTED_VALUES: u64 = 8_324;
    const EXPECTED_INPUT_CHECKS: u64 = 16_648;
    const EXPECTED_GUARDS: u64 = 36;
    const EXPECTED_READBACK: u64 = 100_032;
    const EXPECTED_WRITES: u64 = 150_048;
    const EXPECTED_TENSOR_PEAK: u64 = 49_176;
    const EXPECTED_GROUPS: u64 = 134;

    fn config() -> NativeExecutionConfig {
        NativeExecutionConfig {
            weight_capacity_bytes: 4,
            tensor_capacity_bytes: 64 * 1024,
            auxiliary_capacity_bytes: 4,
        }
    }

    #[derive(Debug)]
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
                        && suffix.bytes().all(|byte| byte.is_ascii_digit()),
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
            "180-second cooperative deadline exceeded"
        );
        Ok(())
    }

    // Independent literal CPU contract. Each named operation is F32; there is
    // no F64 exponential, algebraically fused tail, SFU approximation, or FMA.
    fn literal(input: f32, gate: f32) -> f32 {
        let sigmoid = if gate >= 0.0 {
            let exponential: f32 = (-gate).exp();
            let denominator: f32 = 1.0 + exponential;
            1.0 / denominator
        } else {
            let exponential: f32 = gate.exp();
            let denominator: f32 = 1.0 + exponential;
            exponential / denominator
        };
        input * sigmoid
    }

    fn exact_case(input: f32, gate: f32) -> bool {
        input == 0.0 || gate == 0.0 || gate.abs() == f32::MAX || gate == -103.0 || gate == -105.0
    }

    // The two large fixtures visit all24 cases. The single-element fixture uses
    // amplified exp(-103) then negative exp(-105), changing both operands.
    fn operand(index: usize) -> (f32, f32) {
        match index {
            0 => (0.0, 0.0),
            1 => (-0.0, -0.0),
            2 => (f32::MAX, f32::MAX),
            3 => (-f32::MAX, f32::MAX),
            4 => (f32::MAX, -f32::MAX),
            5 => (-f32::MAX, -f32::MAX),
            6 => (f32::MAX, -90.0),
            7 => (-f32::MAX, -90.0),
            8 => (f32::MAX, -103.0),
            9 => (-f32::MAX, -103.0),
            10 => (f32::MAX, -105.0),
            11 => (-f32::MAX, -105.0),
            12 => (1.0, -48.0),
            13 => (-2.0, 48.0),
            14 => (0.125, -1.0),
            15 => (-8.0, 1.0),
            16 => (3.5, 0.5),
            17 => (-4.25, -0.5),
            18 => (f32::MIN_POSITIVE, 0.0),
            19 => (f32::from_bits(1), 0.0),
            20 => (f32::from_bits(3), 0.0),
            21 => (1e30, -80.0),
            22 => (-1e30, -87.0),
            23 => (1.0, 0.0),
            _ => unreachable!(),
        }
    }

    struct Fixture {
        len: usize,
        input: [Vec<f32>; REPLAYS],
        gate: [Vec<f32>; REPLAYS],
        expected: [Vec<f32>; REPLAYS],
    }

    fn fixture(len: usize) -> Result<Fixture> {
        ensure!(
            LENGTHS.contains(&len),
            "length is outside fixed qualification plan"
        );
        let pairs: [Vec<(f32, f32)>; REPLAYS] = std::array::from_fn(|replay| {
            (0..len)
                .map(|coordinate| {
                    operand(if len == 1 {
                        [8, 11][replay]
                    } else {
                        (coordinate * 7 + replay * 11) % 24
                    })
                })
                .collect()
        });
        let input = std::array::from_fn(|replay| pairs[replay].iter().map(|pair| pair.0).collect());
        let gate = std::array::from_fn(|replay| pairs[replay].iter().map(|pair| pair.1).collect());
        let expected: [Vec<f32>; REPLAYS] = std::array::from_fn(|replay| {
            pairs[replay]
                .iter()
                .map(|&(input, gate)| literal(input, gate))
                .collect()
        });
        ensure!(
            expected.iter().flatten().all(|value| value.is_finite()),
            "nonfinite fixture output"
        );
        for selector in [0, 1] {
            ensure!(
                pairs[0].iter().zip(&pairs[1]).any(|(a, b)| {
                    let (a, b) = if selector == 0 {
                        (a.0, b.0)
                    } else {
                        (a.1, b.1)
                    };
                    a.to_bits() != b.to_bits()
                }),
                "replay must change each input operand"
            );
        }
        ensure!(
            expected[0]
                .iter()
                .zip(&expected[1])
                .any(|(a, b)| a.to_bits() != b.to_bits()),
            "stale fixture outputs"
        );
        Ok(Fixture {
            len,
            input,
            gate,
            expected,
        })
    }

    fn guarded(payload: &[f32]) -> Vec<f32> {
        let mut values = Vec::with_capacity(payload.len() + 2);
        values.push(GUARD);
        values.extend_from_slice(payload);
        values.push(GUARD);
        values
    }

    fn guards(values: &[f32], len: usize) -> Result<()> {
        ensure!(values.len() == len + 2, "guarded extent differs");
        ensure!(
            values[0].to_bits() == GUARD.to_bits() && values[len + 1].to_bits() == GUARD.to_bits(),
            "guard changed"
        );
        Ok(())
    }

    fn compare_value(actual: f32, expected: f32, exact: bool) -> Result<f64> {
        ensure!(
            actual.is_finite() && expected.is_finite(),
            "nonfinite sigmoid result"
        );
        let error = (f64::from(actual) - f64::from(expected)).abs();
        if exact || expected == 0.0 {
            ensure!(
                actual.to_bits() == expected.to_bits(),
                "exact bits differ: {actual:?} != {expected:?}"
            );
        } else {
            ensure!(actual != 0.0, "nonzero expected value was flushed to zero");
            ensure!(
                actual.is_sign_negative() == expected.is_sign_negative(),
                "sigmoid output sign differs"
            );
            let magnitude = f64::from(expected).abs();
            ensure!(
                error <= ABS_TOL + REL_TOL * magnitude,
                "sigmoid tolerance failed: {actual:?} != {expected:?}, error={error}"
            );
            if magnitude < SMALL_THRESHOLD {
                ensure!(
                    error <= SMALL_REL_TOL * magnitude,
                    "small sigmoid relative tolerance failed: {actual:?} != {expected:?}"
                );
            }
        }
        Ok(error)
    }

    fn compare_case(
        actual: &[Vec<f32>; 3],
        case: &Fixture,
        replay: usize,
        cpu_exact: bool,
    ) -> Result<f64> {
        for values in actual {
            guards(values, case.len)?;
        }
        let mut maximum = 0.0_f64;
        for coordinate in 0..case.len {
            ensure!(
                actual[0][coordinate + 1].to_bits() == case.input[replay][coordinate].to_bits(),
                "read-only input changed at{coordinate}"
            );
            ensure!(
                actual[1][coordinate + 1].to_bits() == case.gate[replay][coordinate].to_bits(),
                "read-only gate changed at{coordinate}"
            );
            maximum = maximum.max(compare_value(
                actual[2][coordinate + 1],
                case.expected[replay][coordinate],
                cpu_exact
                    || exact_case(
                        case.input[replay][coordinate],
                        case.gate[replay][coordinate],
                    ),
            )?);
        }
        Ok(maximum)
    }

    fn write_case(
        device: &dyn ExecutionDevice,
        tensors: &[F32Tensor; 3],
        case: &Fixture,
        replay: usize,
        deadline: Instant,
    ) -> Result<()> {
        let payloads = [
            guarded(&case.input[replay]),
            guarded(&case.gate[replay]),
            guarded(&vec![f32::NAN; case.len]),
        ];
        for (tensor, payload) in tensors.iter().zip(&payloads) {
            check_time(deadline)?;
            device.write_f32(&tensor.full_view(), payload)?;
        }
        Ok(())
    }

    fn read_case(
        device: &dyn ExecutionDevice,
        tensors: &[F32Tensor; 3],
        case: &Fixture,
        deadline: Instant,
    ) -> Result<[Vec<f32>; 3]> {
        let mut actual: [Vec<f32>; 3] = std::array::from_fn(|_| vec![f32::NAN; case.len + 2]);
        for (tensor, destination) in tensors.iter().zip(&mut actual) {
            check_time(deadline)?;
            device.read_f32(&tensor.full_view(), destination)?;
        }
        Ok(actual)
    }

    fn record(
        device: &dyn ExecutionDevice,
        tensors: &[F32Tensor; 3],
        len: usize,
    ) -> Result<RecordedBatch> {
        let mut commands = CommandList::new();
        commands.sigmoid_gate(
            &tensors[2].view(1, len)?,
            &tensors[0].view(1, len)?,
            &tensors[1].view(1, len)?,
        )?;
        Ok(device.record(&commands)?)
    }

    fn references(fixtures: &[Fixture], deadline: Instant) -> Result<()> {
        // Explicit anchors prevent a changed oracle or host FTZ mode from silently
        // weakening the two underflow-boundary checks before any native open.
        ensure!(
            (-103.0_f32).exp().to_bits() == 1 && (-105.0_f32).exp().to_bits() == 0,
            "host F32 exponential tail anchors differ"
        );
        ensure!(
            literal(f32::MAX, -103.0).to_bits() == 0x34ff_ffff,
            "amplified bit1 reference differs"
        );
        ensure!(
            literal(-f32::MAX, -105.0).to_bits() == 0x8000_0000,
            "negative zero reference differs"
        );
        ensure!(
            literal(f32::MAX, -90.0) != 0.0,
            "subnormal exponential reference flushed"
        );
        let device = CpuExecutionDevice::new();
        let mut compared = 0_u64;
        for case in fixtures {
            check_time(deadline)?;
            let tensors = [
                device.allocate_f32(case.len + 2)?,
                device.allocate_f32(case.len + 2)?,
                device.allocate_f32(case.len + 2)?,
            ];
            let batch = record(&device, &tensors, case.len)?;
            for replay in 0..REPLAYS {
                write_case(&device, &tensors, case, replay, deadline)?;
                check_time(deadline)?;
                let completion = device.submit(&batch)?;
                device.wait(&completion)?;
                drop(completion);
                let actual = read_case(&device, &tensors, case, deadline)?;
                compare_case(&actual, case, replay, true)?;
                compared += case.len as u64;
            }
        }
        ensure!(compared == EXPECTED_VALUES, "CPU comparison count differs");
        check_time(deadline)
    }

    #[derive(Default)]
    struct Receipt {
        physical: u64,
        logical: u64,
        compared: u64,
        input_checks: u64,
        cpu_compared: u64,
        guards: u64,
        readback: u64,
        host_writes: u64,
        allocation: u64,
        expected_allocation: u64,
        budget_after: u64,
        tensor_peak: u64,
        max_error: f64,
        retired: bool,
    }

    impl Receipt {
        fn print(&self, execution: &str, qualified: bool, elapsed: f64) {
            println!("{{\"execution\":\"{execution}\",\"passed\":{qualified},\"qualified\":{qualified},\"compared_values\":{},\"cpu_reference_values\":{},\"input_checks\":{},\"guard_checks\":{},\"physical_dispatches\":{},\"logical_submissions\":{},\"expected_allocation_bytes\":{},\"allocation_bytes\":{},\"hag_budget_bytes\":{GPU_BUDGET},\"cpu_payload_bound_bytes\":{CPU_PAYLOAD_BOUND},\"readback_bytes\":{},\"host_write_bytes\":{},\"uploaded_weight_bytes\":0,\"weight_peak_bytes\":0,\"tensor_peak_bytes\":{},\"auxiliary_peak_bytes\":0,\"budget_after\":{},\"budget_current_bytes\":{},\"retired\":{},\"max_absolute_error\":{},\"absolute_tolerance\":{ABS_TOL},\"relative_tolerance\":{REL_TOL},\"small_relative_tolerance\":{SMALL_REL_TOL},\"nonzero_required\":true,\"tail_103_and_105_require_exact\":true,\"elapsed_seconds\":{elapsed}}}", self.compared,self.cpu_compared,self.input_checks,self.guards,self.physical,self.logical,self.expected_allocation,self.allocation,self.readback,self.host_writes,self.tensor_peak,self.budget_after,self.budget_after,self.retired,self.max_error);
        }
    }

    fn residency(device: &NativeExecutionDevice, tensor_bytes: u64) -> Result<()> {
        let state = device.stats()?;
        ensure!(
            state.weight_live_bytes == 0
                && state.tensor_live_bytes == tensor_bytes
                && state.auxiliary_live_bytes == 0
                && !state.poisoned,
            "native residency differs: {state:?}"
        );
        Ok(())
    }

    fn submit(
        device: &NativeExecutionDevice,
        batch: &RecordedBatch,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<Submission> {
        check_time(deadline)?;
        let before = device.stats()?;
        let completion = device.submit(batch)?;
        device.wait(&completion)?;
        device.wait(&completion)?;
        let after = device.stats()?;
        ensure!(
            after.dispatch_attempts == before.dispatch_attempts + 1
                && after.dispatch_count == before.dispatch_count + 1
                && after.logical_submissions == before.logical_submissions + 1,
            "submission counts differ"
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
            "submission changed residency"
        );
        receipt.physical += 1;
        receipt.logical += 1;
        check_time(deadline)?;
        Ok(completion)
    }

    fn native_case(
        device: &NativeExecutionDevice,
        case: &Fixture,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        residency(device, 0)?;
        let before = device.stats()?;
        let tensors = [
            device.allocate_f32(case.len + 2)?,
            device.allocate_f32(case.len + 2)?,
            device.allocate_f32(case.len + 2)?,
        ];
        let live = (3 * (case.len + 2) * 4) as u64;
        residency(device, live)?;
        ensure!(
            device.stats()?.host_write_bytes == before.host_write_bytes + live,
            "zero-initialization accounting differs"
        );
        let before_record = device.stats()?;
        let batch = record(device, &tensors, case.len)?;
        ensure!(
            device.stats()? == before_record,
            "recording performed a transfer/allocation"
        );
        let mut retained_completion = None;
        for replay in 0..REPLAYS {
            let before_write = device.stats()?;
            write_case(device, &tensors, case, replay, deadline)?;
            ensure!(
                device.stats()?.host_write_bytes == before_write.host_write_bytes + live,
                "fixture write count differs"
            );
            retained_completion = Some(submit(device, &batch, deadline, receipt)?);
            let before_read = device.stats()?;
            let actual = read_case(device, &tensors, case, deadline)?;
            let after_read = device.stats()?;
            ensure!(
                after_read.host_read_bytes == before_read.host_read_bytes + live
                    && after_read.host_write_bytes == before_read.host_write_bytes,
                "explicit readback accounting differs"
            );
            receipt.readback += live;
            receipt.max_error = receipt
                .max_error
                .max(compare_case(&actual, case, replay, false)?);
            receipt.compared += case.len as u64;
            receipt.input_checks += 2 * case.len as u64;
            receipt.guards += 6;
        }
        // Public tensor handles can retire while the recording/completion retain
        // all three operands. Last completion owns the batch after batch drops.
        drop(tensors);
        residency(device, live)?;
        drop(batch);
        residency(device, live)?;
        drop(retained_completion);
        residency(device, 0)
    }

    fn native(
        path: &Path,
        fixtures: &[Fixture],
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly0 before native access"
        );
        let configuration = config();
        let allocation =
            configuration.allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?;
        ensure!(
            allocation == EXPECTED_ALLOCATION && allocation < GPU_BUDGET,
            "immutable library/allocation changed from preregistered plan"
        );
        receipt.expected_allocation = allocation;
        let budget = MemoryBudget::new(GPU_BUDGET);
        receipt.retired = false;
        let result = (|| -> Result<()> {
            check_time(deadline)?;
            let device = NativeExecutionDevice::open_with_profile(
                path,
                &budget,
                configuration,
                NativeExecutionProfile::Qwen35Probe,
            )?;
            receipt.allocation = device.stats()?.allocation_bytes;
            ensure!(
                device.stats()?
                    == NativeExecutionStats {
                        allocation_bytes: allocation,
                        ..NativeExecutionStats::default()
                    }
                    && budget.current_bytes() == allocation,
                "initial queue differs"
            );
            for case in fixtures {
                check_time(deadline)?;
                let outcome = native_case(&device, case, deadline, receipt);
                // Preserve observed traffic/dispatch counters on a refusal too;
                // case-local handles have already retired or been quarantined.
                let observed = device.stats()?;
                receipt.host_writes = observed.host_write_bytes;
                receipt.readback = observed.host_read_bytes;
                receipt.tensor_peak = observed.tensor_peak_bytes;
                receipt.physical = observed.dispatch_count;
                receipt.logical = observed.logical_submissions;
                outcome?;
            }
            residency(&device, 0)?;
            let state = device.stats()?;
            receipt.host_writes = state.host_write_bytes;
            receipt.tensor_peak = state.tensor_peak_bytes;
            ensure!(
                receipt.physical == EXPECTED_DISPATCHES
                    && receipt.logical == EXPECTED_DISPATCHES
                    && receipt.compared == EXPECTED_VALUES
                    && receipt.input_checks == EXPECTED_INPUT_CHECKS
                    && receipt.guards == EXPECTED_GUARDS
                    && receipt.readback == EXPECTED_READBACK,
                "comparison receipt differs from fixed plan"
            );
            ensure!(
                state.dispatch_count == EXPECTED_DISPATCHES
                    && state.dispatch_attempts == EXPECTED_DISPATCHES
                    && state.logical_submissions == EXPECTED_DISPATCHES
                    && state.host_read_bytes == EXPECTED_READBACK
                    && state.host_write_bytes == EXPECTED_WRITES
                    && state.uploaded_weight_bytes == 0
                    && state.weight_peak_bytes == 0
                    && state.tensor_peak_bytes == EXPECTED_TENSOR_PEAK
                    && state.auxiliary_peak_bytes == 0
                    && state.allocation_bytes == allocation
                    && budget.current_bytes() == allocation
                    && !state.poisoned,
                "final native accounting differs: {state:?}"
            );
            drop(device);
            Ok(())
        })();
        receipt.budget_after = budget.current_bytes();
        receipt.retired = receipt.budget_after == 0;
        result?;
        ensure!(receipt.retired, "native queue did not fully retire");
        check_time(deadline)
    }

    pub fn main() -> Result<()> {
        let mode = mode(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
        if matches!(mode, Mode::Help) {
            println!("native_sigmoid_probe --cpu-reference | --native /dev/dri/renderD...\nFixed synthetic slot16 qualification only. Native mode requires HAG_TRACE_SUBMISSION=0 and an external process/thermal/GPU-crash supervisor; cooperative deadline180s does not interrupt a blocked wait. GPU budget256KiB, host payload2MiB. No model loading or serving registration.");
            return Ok(());
        }
        let started = Instant::now();
        let deadline = started + Duration::from_secs(180);
        let execution = if matches!(mode, Mode::Cpu) {
            "cpu-sigmoid-reference"
        } else {
            "native-hag-sigmoid-diagnostic"
        };
        let mut receipt = Receipt {
            retired: true,
            ..Receipt::default()
        };
        let result = (|| -> Result<()> {
            let fixtures = LENGTHS
                .into_iter()
                .map(fixture)
                .collect::<Result<Vec<_>>>()?;
            // Six vectors/fixture plus conservative concurrent CPU/backend/guard
            // payloads and512KiB library/staging headroom. Process RSS is external.
            let retained = fixtures.iter().map(|case| 6 * case.len * 4).sum::<usize>() as u64;
            ensure!(
                retained + 10 * EXPECTED_TENSOR_PEAK + 512 * 1024 <= CPU_PAYLOAD_BOUND,
                "host payload forecast exceeds bound"
            );
            references(&fixtures, deadline)?;
            receipt.cpu_compared = EXPECTED_VALUES;
            match mode {
                Mode::Cpu => {
                    receipt.compared = EXPECTED_VALUES;
                    receipt.input_checks = EXPECTED_INPUT_CHECKS;
                    receipt.guards = EXPECTED_GUARDS;
                    Ok(())
                }
                Mode::Native(path) => native(&path, &fixtures, deadline, &mut receipt),
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
        fn cli_requires_one_explicit_mode_and_render_node() {
            assert!(matches!(
                mode(&["--cpu-reference".into()]).unwrap(),
                Mode::Cpu
            ));
            assert!(matches!(
                mode(&["--native".into(), "/dev/dri/renderD128".into()]).unwrap(),
                Mode::Native(_)
            ));
            assert!(matches!(mode(&["--help".into()]).unwrap(), Mode::Help));
            for arguments in [
                vec![],
                vec!["--native".into()],
                vec!["--native".into(), "/tmp/renderD128".into()],
                vec!["--native".into(), "/dev/dri/renderD".into()],
                vec!["--native".into(), "/dev/dri/card0".into()],
                vec!["--native".into(), "/dev/dri/renderD128/extra".into()],
                vec!["--cpu-reference".into(), "extra".into()],
            ] {
                assert!(mode(&arguments).is_err());
            }
        }

        #[test]
        fn fixed_fixtures_cover_f32_tail_and_rounding_boundaries() -> Result<()> {
            assert_eq!((-103.0_f32).exp().to_bits(), 1);
            assert_eq!((-105.0_f32).exp().to_bits(), 0);
            assert_eq!(literal(f32::MAX, -103.0).to_bits(), 0x34ff_ffff);
            assert_eq!(literal(-f32::MAX, -103.0).to_bits(), 0xb4ff_ffff);
            assert_eq!(literal(-f32::MAX, -105.0).to_bits(), 0x8000_0000);
            assert_eq!(literal(f32::from_bits(3), 0.0).to_bits(), 2);
            for len in [0, 2, 64, 4097] {
                assert!(fixture(len).is_err());
            }
            for len in LENGTHS {
                let case = fixture(len)?;
                if len > 1 {
                    for tag in 0..24 {
                        let pair = operand(tag);
                        assert!(case.input[0]
                            .iter()
                            .zip(&case.gate[0])
                            .any(|(&a, &b)| a.to_bits() == pair.0.to_bits()
                                && b.to_bits() == pair.1.to_bits()));
                    }
                }
            }
            Ok(())
        }

        #[test]
        fn zero_flush_wrong_tail_stale_output_and_guard_writes_fail() -> Result<()> {
            let expected = literal(f32::MAX, -90.0);
            assert!(expected > 0.0);
            assert!(compare_value(0.0, expected, false).is_err());
            assert!(compare_value(-expected, expected, false).is_err());
            assert!(compare_value(1e-8, literal(1.0, -48.0), false).is_err());
            assert!(compare_value(f32::NAN, expected, false).is_err());
            assert!(compare_value(-0.0, 0.0, true).is_err());
            assert!(
                compare_value(f32::from_bits(0x34ff_fffe), literal(f32::MAX, -103.0), true)
                    .is_err()
            );
            let case = fixture(65)?;
            let mut actual = [
                guarded(&case.input[1]),
                guarded(&case.gate[1]),
                guarded(&case.expected[1]),
            ];
            assert_eq!(compare_case(&actual, &case, 1, false)?, 0.0);
            actual[2] = guarded(&case.expected[0]);
            assert!(compare_case(&actual, &case, 1, false).is_err());
            actual[2] = guarded(&case.expected[1]);
            for tensor in 0..3 {
                for index in [0, case.len + 1] {
                    actual[tensor][index] = 0.0;
                    assert!(compare_case(&actual, &case, 1, false).is_err());
                    actual[tensor][index] = GUARD;
                }
            }
            actual[0][1] = -actual[0][1];
            assert!(compare_case(&actual, &case, 1, false).is_err());
            Ok(())
        }

        #[test]
        fn counters_allocation_and_payload_forecasts_match_fixed_plan() -> Result<()> {
            let sum = LENGTHS.iter().sum::<usize>() as u64;
            let guarded_sum = LENGTHS.iter().map(|len| len + 2).sum::<usize>() as u64;
            assert_eq!(EXPECTED_DISPATCHES, (LENGTHS.len() * REPLAYS) as u64);
            assert_eq!(EXPECTED_VALUES, sum * REPLAYS as u64);
            assert_eq!(EXPECTED_INPUT_CHECKS, EXPECTED_VALUES * 2);
            assert_eq!(EXPECTED_GUARDS, EXPECTED_DISPATCHES * 6);
            assert_eq!(EXPECTED_READBACK, guarded_sum * 3 * 4 * REPLAYS as u64);
            assert_eq!(EXPECTED_WRITES, guarded_sum * 3 * 4 * (1 + REPLAYS) as u64);
            assert_eq!(EXPECTED_TENSOR_PEAK, (4096 + 2) * 3 * 4);
            assert_eq!(
                EXPECTED_GROUPS,
                LENGTHS
                    .iter()
                    .map(|len| len.div_ceil(64) as u64)
                    .sum::<u64>()
                    * REPLAYS as u64
            );
            assert_eq!(
                config().allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?,
                EXPECTED_ALLOCATION
            );
            assert!(EXPECTED_ALLOCATION < GPU_BUDGET);
            assert!(sum * 6 * 4 + 10 * EXPECTED_TENSOR_PEAK + 512 * 1024 <= CPU_PAYLOAD_BOUND);
            Ok(())
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::main()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("HAG sigmoid qualification requires Linux; no work submitted.");
    std::process::exit(2);
}
