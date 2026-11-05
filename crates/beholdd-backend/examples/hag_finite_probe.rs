//! SOURCE-ONLY future slot17 diagnostic. This is not yet a repository example.
//! Requires the staged native_execution::finite module and its real ISA library.
//! No model paths, discovery, serving registration, or implicit device selection.
//! Native execution requires external thermal/OOM/crash supervision and retirement.
//! The cooperative180s deadline cannot interrupt a blocked native dispatch/fence.
//!
//! --cpu-reference is explicitly golden classification ONLY: it compares fixed
//! bit-pattern expectations with Rust f32::is_finite, not an execution adapter.
//! A future generic CPU checker must be separately tested against these fixtures.

#[cfg(all(target_os = "linux", feature = "hag-native"))]
mod proof {
    use anyhow::{ensure, Context, Result};
    use beholdd_backend::execution::{
        CpuExecutionDevice, ExecutionDevice, ExecutionError, F32TensorView,
    };
    use beholdd_backend::hag::native_execution::finite::{
        NativeFiniteCheck, NativeFiniteStats, NATIVE_FINITE_GUARD, NATIVE_FINITE_MAX_ELEMENTS,
        NATIVE_FINITE_PACKET_BYTES,
    };
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionProfile,
        NativeExecutionStats,
    };
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::time::{Duration, Instant};

    const LENGTHS: [usize; 3] = [1, 65, 65_536];
    const FINITE_BITS: [u32; 12] = [
        0x0000_0000,
        0x8000_0000, // signed zero
        0x0000_0001,
        0x8000_0001, // signed least subnormal
        0x007f_ffff,
        0x807f_ffff, // signed largest subnormal
        0x0080_0000,
        0x8080_0000, // signed least normal
        0x7f7f_ffff,
        0xff7f_ffff, // signed largest finite
        0x3f80_0000,
        0xbf80_0000,
    ];
    const BAD_BITS: [u32; 3] = [0x7fc0_0001, 0x7f80_0000, 0xff80_0000];
    const LARGE_EXTRAS: [usize; 8] = [64, 127, 128, 32_767, 32_768, 65_471, 65_472, 65_535];
    const CANCEL_MARKER: &str = "finite-probe intentional cancellation";
    const GPU_BUDGET: u64 = 2 * 1024 * 1024;
    const CPU_PAYLOAD_BOUND: u64 = 8 * 1024 * 1024;
    // Fixed queue: weight4B, tensor512KiB, auxiliary4B; guarded/page-rounded.
    // This preregisters20KiB for the future18-program library, the same page
    // extent as the current17-program library. If adding slot17 crosses a page,
    // preflight MUST refuse; update this only after independent source review.
    const EXPECTED_ALLOCATION: u64 = 581_632;
    const EXPECTED_BAD_CASES: u64 = 414; // (1 +65 +72) *3
    const EXPECTED_GOLDEN_CASES: u64 = 831; // 3 baselines +414 bad/recovery pairs
    const EXPECTED_GOLDEN_VALUES: u64 = 28_402_510;
    const EXPECTED_ANCHORS: u64 = 15;
    const EXPECTED_ACCEPTED: u64 = 422; // 3 +414 +5 cancellation recoveries
    const EXPECTED_CANCELLED: u64 = 5;
    const EXPECTED_CHECK_ATTEMPTS: u64 = 841; // 831 +5 cancellations +5 recoveries
    const EXPECTED_PREFLIGHT_REFUSALS: u64 = 3;
    const EXPECTED_PENDING: u64 = 840; // cancellation1 writes nothing
    const EXPECTED_DISPATCHES: u64 = 839; // cancellation1/2 submit nothing
    const EXPECTED_STATUS_READS: u64 = 837; // cancellation1..4 read nothing
    const EXPECTED_GUARDS: u64 = 5_052; // 841*6 +3*2 whole-input guard rechecks
    const EXPECTED_INPUT_CHECKS: u64 = 66_430; // sum(lengths) +2*414 changed scalars
    const EXPECTED_READBACK: u64 = 500_200;
    const EXPECTED_WRITES: u64 = 743_764;
    const EXPECTED_TENSOR_PEAK: u64 = 262_428; // (65536+2)*4 +276
    const EXPECTED_RETENTION_CHECKS: u64 = 11; // 3 per input,2 checker/queue

    fn config() -> NativeExecutionConfig {
        NativeExecutionConfig {
            weight_capacity_bytes: 4,
            tensor_capacity_bytes: 512 * 1024,
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
            _ => anyhow::bail!("expected --cpu-reference or --native /dev/dri/renderD..."),
        }
    }

    fn check_time(deadline: Instant) -> Result<()> {
        ensure!(
            Instant::now() < deadline,
            "180-second cooperative deadline exceeded"
        );
        Ok(())
    }

    fn control_time(deadline: Instant) -> Result<(), ExecutionError> {
        if Instant::now() < deadline {
            Ok(())
        } else {
            Err(ExecutionError::Shape(
                "180-second cooperative deadline exceeded".into(),
            ))
        }
    }

    fn finite_word(index: usize) -> f32 {
        f32::from_bits(FINITE_BITS[index % FINITE_BITS.len()])
    }

    fn input_payload(len: usize) -> Result<Vec<f32>> {
        ensure!(LENGTHS.contains(&len), "unregistered finite fixture length");
        let mut values = Vec::new();
        values
            .try_reserve_exact(len + 2)
            .context("allocating finite fixture")?;
        values.push(NATIVE_FINITE_GUARD);
        values.extend((0..len).map(finite_word));
        values.push(NATIVE_FINITE_GUARD);
        Ok(values)
    }

    fn positions(len: usize) -> Result<Vec<usize>> {
        let mut result = Vec::new();
        let count = match len {
            1 => 1,
            65 => 65,
            65_536 => 72,
            _ => anyhow::bail!("unregistered finite fixture length"),
        };
        result
            .try_reserve_exact(count)
            .context("allocating position metadata")?;
        match len {
            1 => result.push(0),
            65 => result.extend(0..65),
            65_536 => {
                result.extend(0..64);
                result.extend(LARGE_EXTRAS);
            }
            _ => unreachable!(),
        }
        ensure!(
            result.len() == count && result.iter().all(|&index| index < len),
            "position plan differs"
        );
        for (index, value) in result.iter().enumerate() {
            ensure!(
                !result[..index].contains(value),
                "duplicate diagnostic position"
            );
        }
        Ok(result)
    }

    #[derive(Default)]
    struct Golden {
        cases: u64,
        classified_values: u64,
        anchors: u64,
        refusals: u64,
    }

    // The expected answer is a fixture property, not calculated by the shader's
    // exponent-mask implementation. Visit EVERY element, including after a bad
    // value, so classified_values is an actual performed count, not an estimate.
    fn classify(values: &[f32], expected: bool, golden: &mut Golden) -> Result<()> {
        let mut finite = true;
        for value in values {
            finite &= value.is_finite();
            golden.classified_values += 1;
        }
        ensure!(
            finite == expected,
            "independent golden classification differs"
        );
        golden.cases += 1;
        golden.refusals += u64::from(!finite);
        Ok(())
    }

    fn references(deadline: Instant) -> Result<Golden> {
        let mut golden = Golden::default();
        for bits in FINITE_BITS {
            ensure!(
                f32::from_bits(bits).is_finite(),
                "finite anchor was misclassified"
            );
            golden.anchors += 1;
        }
        for bits in BAD_BITS {
            ensure!(
                !f32::from_bits(bits).is_finite(),
                "nonfinite anchor was misclassified"
            );
            golden.anchors += 1;
        }
        for len in LENGTHS {
            check_time(deadline)?;
            let mut values = input_payload(len)?;
            classify(&values[1..len + 1], true, &mut golden)?;
            for index in positions(len)? {
                for bits in BAD_BITS {
                    check_time(deadline)?;
                    values[index + 1] = f32::from_bits(bits);
                    classify(&values[1..len + 1], false, &mut golden)?;
                    values[index + 1] = finite_word(index);
                    classify(&values[1..len + 1], true, &mut golden)?;
                }
            }
        }
        ensure!(
            golden.cases == EXPECTED_GOLDEN_CASES
                && golden.classified_values == EXPECTED_GOLDEN_VALUES
                && golden.anchors == EXPECTED_ANCHORS
                && golden.refusals == EXPECTED_BAD_CASES,
            "golden counts differ from preregistered fixtures"
        );
        check_time(deadline)?;
        Ok(golden)
    }

    #[derive(Clone, Copy)]
    enum Expected {
        Accept,
        Refuse,
        Cancel(usize),
    }

    impl Expected {
        fn traffic(self) -> (u64, u64, u64, u64, usize) {
            match self {
                Self::Accept => (1, 1, 1, 1, 5),
                Self::Refuse => (1, 1, 1, 0, 4),
                Self::Cancel(stage) => (
                    u64::from(stage > 1),
                    u64::from(stage > 2),
                    u64::from(stage > 4),
                    0,
                    stage,
                ),
            }
        }
    }

    #[derive(Default)]
    struct Receipt {
        golden: Golden,
        native_executed: bool,
        attempts: u64,
        accepted: u64,
        refused: u64,
        cancelled: u64,
        preflight_refusals: u64,
        pending: u64,
        status_reads: u64,
        physical: u64,
        dispatch_attempts: u64,
        logical: u64,
        input_checks: u64,
        guards: u64,
        retention_checks: u64,
        readback: u64,
        writes: u64,
        allocation: u64,
        tensor_peak: u64,
        budget_after: u64,
        retired: bool,
    }

    impl Receipt {
        fn print(&self, cpu: bool, passed: bool, elapsed: f64) {
            let execution = if cpu {
                "cpu-finite-golden-classification"
            } else {
                "native-hag-finite-diagnostic"
            };
            let qualified = !cpu && passed;
            println!(
                "{{\"execution\":\"{execution}\",\"passed\":{passed},\"qualified\":{qualified},\"cpu_adapter_executed\":false,\"cpu_reference_kind\":\"golden-classification-only\",\"cpu_reference_cases\":{},\"cpu_reference_values\":{},\"cpu_anchor_checks\":{},\"native_executed\":{},\"check_attempts\":{},\"accepted_checks\":{},\"finite_refusals\":{},\"cancellation_refusals\":{},\"preflight_refusals\":{},\"pending_writes\":{},\"pending_write_bytes\":{},\"status_reads\":{},\"status_read_bytes\":{},\"physical_dispatches\":{},\"dispatch_attempts\":{},\"logical_submissions\":{},\"input_checks\":{},\"guard_checks\":{},\"retention_checks\":{},\"readback_bytes\":{},\"host_write_bytes\":{},\"expected_allocation_bytes\":{EXPECTED_ALLOCATION},\"allocation_bytes\":{},\"hag_budget_bytes\":{GPU_BUDGET},\"cpu_payload_bound_bytes\":{CPU_PAYLOAD_BOUND},\"tensor_peak_bytes\":{},\"budget_after\":{},\"budget_current_bytes\":{},\"retired\":{},\"elapsed_seconds\":{elapsed}}}",
                self.golden.cases, self.golden.classified_values, self.golden.anchors,
                self.native_executed, self.attempts, self.accepted, self.refused,
                self.cancelled, self.preflight_refusals, self.pending, self.pending * 256,
                self.status_reads, self.status_reads * 256, self.physical,
                self.dispatch_attempts, self.logical, self.input_checks, self.guards,
                self.retention_checks, self.readback, self.writes, self.allocation,
                self.tensor_peak, self.budget_after, self.budget_after, self.retired,
            );
        }
    }

    fn stat_words(value: NativeFiniteStats) -> [u64; 6] {
        [
            value.initialization_write_bytes,
            value.pending_writes,
            value.pending_write_bytes,
            value.status_reads,
            value.status_read_bytes,
            value.accepted_checks,
        ]
    }

    fn residency(device: &NativeExecutionDevice, tensor_bytes: u64) -> Result<()> {
        let state = device.stats()?;
        ensure!(
            !state.poisoned
                && state.tensor_live_bytes == tensor_bytes
                && state.weight_live_bytes == 0
                && state.auxiliary_live_bytes == 0,
            "finite diagnostic residency differs: {state:?}"
        );
        Ok(())
    }

    fn inspect_guards(
        device: &NativeExecutionDevice,
        checker: &NativeFiniteCheck,
        full: &F32TensorView,
        len: usize,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        check_time(deadline)?;
        let before = device.stats()?;
        ensure!(
            checker.diagnostic_read_guards()?.map(f32::to_bits)
                == [NATIVE_FINITE_GUARD.to_bits(); 4],
            "checker guard changed"
        );
        let mut actual = [0.0; 2];
        for (slot, index) in [0, len + 1].into_iter().enumerate() {
            check_time(deadline)?;
            device.read_f32(&full.view(index, 1)?, &mut actual[slot..slot + 1])?;
        }
        ensure!(
            actual.map(f32::to_bits) == [NATIVE_FINITE_GUARD.to_bits(); 2],
            "input guard changed"
        );
        let mut expected = before;
        expected.host_read_bytes += 24;
        ensure!(
            device.stats()? == expected,
            "guard diagnostics changed unrelated counters"
        );
        receipt.guards += 6;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn checked(
        device: &NativeExecutionDevice,
        checker: &mut NativeFiniteCheck,
        input: &F32TensorView,
        full: &F32TensorView,
        expected: Expected,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        check_time(deadline)?;
        let before = device.stats()?;
        let checker_before = checker.stats();
        let mut calls = 0;
        let result = checker.check_controlled(input, &mut || {
            control_time(deadline)?;
            calls += 1;
            if matches!(expected, Expected::Cancel(stage) if calls == stage) {
                return Err(ExecutionError::Shape(CANCEL_MARKER.into()));
            }
            Ok(())
        });
        let (pending, dispatches, reads, accepted, controls) = expected.traffic();
        ensure!(calls == controls, "checker control boundary count differs");
        match expected {
            Expected::Accept => result.context("finite fixture unexpectedly refused")?,
            Expected::Refuse => {
                let error = result.err().context("nonfinite fixture was accepted")?;
                let message = error.to_string();
                ensure!(
                    message.contains("native finite check lane")
                        && message.contains("refused with status"),
                    "nonfinite refusal was not a shader status: {message}"
                );
                receipt.refused += 1;
            }
            Expected::Cancel(_) => {
                let error = result.err().context("cancellation was ignored")?;
                ensure!(
                    error.to_string().contains(CANCEL_MARKER),
                    "unexpected cancellation error: {error}"
                );
                receipt.cancelled += 1;
            }
        }
        let mut after_expected = before;
        after_expected.host_write_bytes += pending * 256;
        after_expected.host_read_bytes += reads * 256;
        after_expected.dispatch_attempts += dispatches;
        after_expected.dispatch_count += dispatches;
        after_expected.logical_submissions += dispatches;
        ensure!(
            device.stats()? == after_expected,
            "check transferred activations or changed resource/counter contract"
        );
        let mut checker_expected = stat_words(checker_before);
        checker_expected[1] += pending;
        checker_expected[2] += pending * 256;
        checker_expected[3] += reads;
        checker_expected[4] += reads * 256;
        checker_expected[5] += accepted;
        ensure!(
            stat_words(checker.stats()) == checker_expected,
            "checker accounting differs"
        );
        receipt.attempts += 1;
        receipt.accepted += accepted;
        receipt.pending += pending;
        receipt.status_reads += reads;
        // These reads are explicit qualification diagnostics after the checker,
        // never part of its status-only execution path.
        inspect_guards(device, checker, full, input.len(), deadline, receipt)
    }

    fn scalar_read(
        device: &NativeExecutionDevice,
        input: &F32TensorView,
        index: usize,
        bits: u32,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        check_time(deadline)?;
        let before = device.stats()?;
        let mut value = [0.0];
        device.read_f32(&input.view(index, 1)?, &mut value)?;
        ensure!(
            value[0].to_bits() == bits,
            "checker changed its read-only input"
        );
        let mut expected = before;
        expected.host_read_bytes += 4;
        ensure!(
            device.stats()? == expected,
            "scalar diagnostic accounting differs"
        );
        receipt.input_checks += 1;
        Ok(())
    }

    fn preflight_refusal(
        device: &NativeExecutionDevice,
        checker: &mut NativeFiniteCheck,
        view: &F32TensorView,
        receipt: &mut Receipt,
    ) -> Result<()> {
        let before = device.stats()?;
        let checker_before = stat_words(checker.stats());
        ensure!(
            checker.check(view).is_err(),
            "malformed/foreign input was admitted"
        );
        ensure!(
            device.stats()? == before && stat_words(checker.stats()) == checker_before,
            "admission refusal performed native work"
        );
        receipt.preflight_refusals += 1;
        Ok(())
    }

    fn native_case(
        device: &NativeExecutionDevice,
        checker: &mut NativeFiniteCheck,
        len: usize,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<()> {
        check_time(deadline)?;
        residency(device, u64::from(NATIVE_FINITE_PACKET_BYTES))?;
        let bytes = ((len + 2) * 4) as u64;
        let payload = input_payload(len)?;
        let before = device.stats()?;
        let tensor = device.allocate_f32(len + 2)?;
        let full = tensor.full_view();
        let input = tensor.view(1, len)?;
        device.write_f32(&full, &payload)?;
        ensure!(
            device.stats()?.host_write_bytes == before.host_write_bytes + 2 * bytes,
            "input initialization/write accounting differs"
        );
        // Only the two checked views retain the input from here onward. The
        // checker's internal recording/completion retention is additionally
        // covered by its hardware-free private-API tests, not forged here.
        drop(tensor);
        residency(device, bytes + u64::from(NATIVE_FINITE_PACKET_BYTES))?;
        receipt.retention_checks += 1;
        if len == 1 {
            preflight_refusal(device, checker, &full.view(1, 0)?, receipt)?;
        }
        if len == 65_536 {
            preflight_refusal(device, checker, &full.view(0, 65_537)?, receipt)?;
        }
        checked(
            device,
            checker,
            &input,
            &full,
            Expected::Accept,
            deadline,
            receipt,
        )?;
        for index in positions(len)? {
            for bits in BAD_BITS {
                check_time(deadline)?;
                device.write_f32(&input.view(index, 1)?, &[f32::from_bits(bits)])?;
                checked(
                    device,
                    checker,
                    &input,
                    &full,
                    Expected::Refuse,
                    deadline,
                    receipt,
                )?;
                scalar_read(device, &input, index, bits, deadline, receipt)?;
                check_time(deadline)?;
                device.write_f32(&input.view(index, 1)?, &[finite_word(index)])?;
                checked(
                    device,
                    checker,
                    &input,
                    &full,
                    Expected::Accept,
                    deadline,
                    receipt,
                )?;
                scalar_read(
                    device,
                    &input,
                    index,
                    finite_word(index).to_bits(),
                    deadline,
                    receipt,
                )?;
            }
        }
        if len == 65_536 {
            for boundary in 1..=5 {
                checked(
                    device,
                    checker,
                    &input,
                    &full,
                    Expected::Cancel(boundary),
                    deadline,
                    receipt,
                )?;
                // Each cancellation is followed by a healthy same-owner reuse,
                // including the post-status/pre-acceptance cancellation.
                checked(
                    device,
                    checker,
                    &input,
                    &full,
                    Expected::Accept,
                    deadline,
                    receipt,
                )?;
            }
        }
        check_time(deadline)?;
        let mut actual = Vec::new();
        actual
            .try_reserve_exact(len + 2)
            .context("allocating bounded input diagnostic")?;
        actual.resize(len + 2, f32::NAN);
        let before = device.stats()?;
        device.read_f32(&full, &mut actual)?;
        let mut expected = before;
        expected.host_read_bytes += bytes;
        ensure!(
            device.stats()? == expected,
            "whole-input diagnostic accounting differs"
        );
        for (got, want) in actual.iter().zip(&payload) {
            ensure!(
                got.to_bits() == want.to_bits(),
                "read-only input or guard changed"
            );
        }
        receipt.input_checks += len as u64;
        receipt.guards += 2;
        drop(actual);
        drop(payload);
        drop(full);
        residency(device, bytes + u64::from(NATIVE_FINITE_PACKET_BYTES))?;
        receipt.retention_checks += 1;
        drop(input);
        residency(device, u64::from(NATIVE_FINITE_PACKET_BYTES))?;
        receipt.retention_checks += 1;
        check_time(deadline)
    }

    fn suite(
        device: &NativeExecutionDevice,
        deadline: Instant,
        receipt: &mut Receipt,
    ) -> Result<NativeFiniteCheck> {
        let mut checker = NativeFiniteCheck::new(device)?;
        ensure!(
            checker.stats().initialization_write_bytes == 548
                && device.stats()?.host_write_bytes == 548,
            "checker preparation accounting differs"
        );
        // Different CPU resource ownership is a metadata refusal; this is NOT
        // a CPU finite-check adapter qualification and opens no second GPU.
        let foreign = CpuExecutionDevice::new();
        let foreign_input = foreign.allocate_f32(1)?;
        preflight_refusal(device, &mut checker, &foreign_input.full_view(), receipt)?;
        drop(foreign_input);
        drop(foreign);
        for len in LENGTHS {
            native_case(device, &mut checker, len, deadline, receipt)?;
        }
        ensure!(
            receipt.attempts == EXPECTED_CHECK_ATTEMPTS
                && receipt.accepted == EXPECTED_ACCEPTED
                && receipt.refused == EXPECTED_BAD_CASES
                && receipt.cancelled == EXPECTED_CANCELLED
                && receipt.preflight_refusals == EXPECTED_PREFLIGHT_REFUSALS
                && receipt.pending == EXPECTED_PENDING
                && receipt.status_reads == EXPECTED_STATUS_READS
                && receipt.guards == EXPECTED_GUARDS
                && receipt.input_checks == EXPECTED_INPUT_CHECKS
                && receipt.retention_checks == EXPECTED_RETENTION_CHECKS - 2,
            "native suite coverage differs from fixed plan"
        );
        ensure!(
            stat_words(checker.stats())
                == [
                    548,
                    EXPECTED_PENDING,
                    EXPECTED_PENDING * 256,
                    EXPECTED_STATUS_READS,
                    EXPECTED_STATUS_READS * 256,
                    EXPECTED_ACCEPTED
                ],
            "final checker statistics differ"
        );
        Ok(checker)
    }

    fn native(path: &Path, deadline: Instant, receipt: &mut Receipt) -> Result<()> {
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly0"
        );
        ensure!(
            NATIVE_FINITE_MAX_ELEMENTS == 65_536 && NATIVE_FINITE_PACKET_BYTES == 276,
            "future checker ABI differs from preregistered plan"
        );
        let configuration = config();
        let expected_allocation =
            configuration.allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?;
        ensure!(
            expected_allocation == EXPECTED_ALLOCATION && expected_allocation < GPU_BUDGET,
            "immutable program-library/arena charge differs; review before launch"
        );
        let budget = MemoryBudget::new(GPU_BUDGET);
        receipt.retired = false;
        let outcome = (|| -> Result<()> {
            check_time(deadline)?;
            let device = NativeExecutionDevice::open_with_profile(
                path,
                &budget,
                configuration,
                NativeExecutionProfile::Qwen35Probe,
            )?;
            receipt.native_executed = true;
            receipt.allocation = device.stats()?.allocation_bytes;
            ensure!(
                device.stats()?
                    == NativeExecutionStats {
                        allocation_bytes: expected_allocation,
                        ..NativeExecutionStats::default()
                    }
                    && budget.current_bytes() == expected_allocation,
                "initial queue differs"
            );
            let result = suite(&device, deadline, receipt);
            // On a refusal, suite-local handles have already dropped. Preserve
            // observed transfers and resource peaks without declaring success.
            let observed = device.stats()?;
            receipt.physical = observed.dispatch_count;
            receipt.dispatch_attempts = observed.dispatch_attempts;
            receipt.logical = observed.logical_submissions;
            receipt.readback = observed.host_read_bytes;
            receipt.writes = observed.host_write_bytes;
            receipt.tensor_peak = observed.tensor_peak_bytes;
            let checker = result?;
            ensure!(
                observed.dispatch_count == EXPECTED_DISPATCHES
                    && observed.dispatch_attempts == EXPECTED_DISPATCHES
                    && observed.logical_submissions == EXPECTED_DISPATCHES
                    && observed.host_read_bytes == EXPECTED_READBACK
                    && observed.host_write_bytes == EXPECTED_WRITES
                    && observed.tensor_peak_bytes == EXPECTED_TENSOR_PEAK
                    && observed.tensor_live_bytes == u64::from(NATIVE_FINITE_PACKET_BYTES)
                    && observed.weight_peak_bytes == 0
                    && observed.weight_live_bytes == 0
                    && observed.auxiliary_peak_bytes == 0
                    && observed.auxiliary_live_bytes == 0
                    && observed.uploaded_weight_bytes == 0
                    && observed.allocation_bytes == EXPECTED_ALLOCATION
                    && budget.current_bytes() == EXPECTED_ALLOCATION
                    && !observed.poisoned,
                "final native accounting differs: {observed:?}"
            );
            // Checker and its private allocation retain the queue even after its
            // primary device handle drops. Final checker drop must release both.
            drop(device);
            ensure!(
                budget.current_bytes() == EXPECTED_ALLOCATION,
                "checker failed to retain its queue"
            );
            receipt.retention_checks += 1;
            drop(checker);
            ensure!(
                budget.current_bytes() == 0,
                "checker/queue allocation did not retire"
            );
            receipt.retention_checks += 1;
            Ok(())
        })();
        receipt.budget_after = budget.current_bytes();
        receipt.retired = receipt.budget_after == 0;
        outcome?;
        ensure!(
            receipt.retired && receipt.retention_checks == EXPECTED_RETENTION_CHECKS,
            "final native retirement evidence differs"
        );
        check_time(deadline)
    }

    fn preflight() -> Result<()> {
        let guarded_bytes = LENGTHS.iter().map(|n| (n + 2) as u64 * 4).sum::<u64>();
        let positions_count = LENGTHS
            .iter()
            .map(|&n| positions(n).map(|p| p.len() as u64))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        ensure!(
            positions_count * 3 == EXPECTED_BAD_CASES,
            "bad-case forecast differs"
        );
        ensure!(
            3 + 2 * EXPECTED_BAD_CASES == EXPECTED_GOLDEN_CASES,
            "golden-case forecast differs"
        );
        let classified = LENGTHS
            .iter()
            .map(|&n| positions(n).map(|p| n as u64 * (1 + 6 * p.len() as u64)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .sum::<u64>();
        ensure!(
            classified == EXPECTED_GOLDEN_VALUES,
            "classified-value forecast differs"
        );
        ensure!(
            2 * EXPECTED_BAD_CASES + LENGTHS.iter().sum::<usize>() as u64 == EXPECTED_INPUT_CHECKS,
            "input-read coverage differs"
        );
        ensure!(
            EXPECTED_CHECK_ATTEMPTS * 6 + 2 * LENGTHS.len() as u64 == EXPECTED_GUARDS,
            "guard coverage differs"
        );
        ensure!(
            EXPECTED_STATUS_READS * 256
                + EXPECTED_CHECK_ATTEMPTS * 24
                + 2 * EXPECTED_BAD_CASES * 4
                + guarded_bytes
                == EXPECTED_READBACK,
            "read traffic forecast differs"
        );
        ensure!(
            548 + guarded_bytes * 2 + 2 * EXPECTED_BAD_CASES * 4 + EXPECTED_PENDING * 256
                == EXPECTED_WRITES,
            "write traffic forecast differs"
        );
        ensure!(
            (65_536 + 2) * 4 + 276 == EXPECTED_TENSOR_PEAK,
            "tensor payload forecast differs"
        );
        // Max input payload + diagnostic output + native conversion, a CPU foreign
        // token, tiny position metadata/status stacks, and512KiB source/library
        // staging allowance. Actual process/driver RSS is externally supervised.
        ensure!(
            3 * ((65_536 + 2) * 4) + 2 * 65_536 + 512 * 1024 <= CPU_PAYLOAD_BOUND,
            "host payload forecast exceeds fixed bound"
        );
        Ok(())
    }

    pub fn main() -> Result<()> {
        let selected = mode(&std::env::args_os().skip(1).collect::<Vec<_>>())?;
        if matches!(selected, Mode::Help) {
            println!("native_finite_probe --cpu-reference | --native /dev/dri/renderD...\nFuture slot17 synthetic diagnostic only. CPU mode is golden classification, NOT adapter qualification. Native mode requires HAG_TRACE_SUBMISSION=0 and external process/thermal/OOM/GPU-crash supervision. Cooperative deadline180s cannot interrupt a blocked fence. GPU budget2MiB, host payload8MiB; no model or serving capability.");
            return Ok(());
        }
        let started = Instant::now();
        let deadline = started + Duration::from_secs(180);
        let cpu = matches!(selected, Mode::Cpu);
        let mut receipt = Receipt {
            retired: true,
            ..Receipt::default()
        };
        let result = (|| -> Result<()> {
            preflight()?;
            receipt.golden = references(deadline)?;
            match selected {
                Mode::Cpu => Ok(()),
                Mode::Native(path) => native(&path, deadline, &mut receipt),
                Mode::Help => unreachable!(),
            }
        })();
        receipt.print(cpu, result.is_ok(), started.elapsed().as_secs_f64());
        result
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn explicit_cli_refuses_ambiguous_or_non_render_paths() {
            assert!(matches!(
                mode(&["--cpu-reference".into()]).unwrap(),
                Mode::Cpu
            ));
            assert!(matches!(
                mode(&["--native".into(), "/dev/dri/renderD128".into()]).unwrap(),
                Mode::Native(_)
            ));
            for args in [
                vec![],
                vec!["--native".into()],
                vec!["--native".into(), "/tmp/renderD128".into()],
                vec!["--native".into(), "/dev/dri/renderD".into()],
                vec!["--native".into(), "/dev/dri/card0".into()],
                vec!["--native".into(), "/dev/dri/renderD128/extra".into()],
                vec!["--cpu-reference".into(), "extra".into()],
            ] {
                assert!(mode(&args).is_err());
            }
        }

        #[test]
        fn fixtures_cover_every_lane_and_registered_stride_tail() -> Result<()> {
            assert_eq!(positions(1)?, [0]);
            assert_eq!(positions(65)?, (0..65).collect::<Vec<_>>());
            let large = positions(65_536)?;
            assert_eq!(&large[..64], &(0..64).collect::<Vec<_>>());
            assert_eq!(&large[64..], &LARGE_EXTRAS);
            let payload = input_payload(65)?;
            for bits in FINITE_BITS {
                assert!(payload[1..66].iter().any(|value| value.to_bits() == bits));
            }
            for len in [0, 2, 64, 66, 65_537] {
                assert!(positions(len).is_err());
                assert!(input_payload(len).is_err());
            }
            Ok(())
        }

        #[test]
        fn golden_classification_counts_performed_work_and_rejects_wrong_expectations() -> Result<()>
        {
            let mut count = Golden::default();
            classify(&[0.0, -0.0, f32::from_bits(1), f32::MAX], true, &mut count)?;
            classify(&[f32::INFINITY, 1.0, 2.0], false, &mut count)?;
            assert_eq!(
                (count.cases, count.classified_values, count.refusals),
                (2, 7, 1)
            );
            assert!(classify(&[f32::NAN], true, &mut Golden::default()).is_err());
            assert!(classify(&[1.0], false, &mut Golden::default()).is_err());
            Ok(())
        }

        #[test]
        fn fixed_traffic_and_payload_bounds_are_independent_of_native_observations() -> Result<()> {
            preflight()?;
            assert_eq!(Expected::Cancel(1).traffic(), (0, 0, 0, 0, 1));
            assert_eq!(Expected::Cancel(2).traffic(), (1, 0, 0, 0, 2));
            assert_eq!(Expected::Cancel(3).traffic(), (1, 1, 0, 0, 3));
            assert_eq!(Expected::Cancel(4).traffic(), (1, 1, 0, 0, 4));
            assert_eq!(Expected::Cancel(5).traffic(), (1, 1, 1, 0, 5));
            assert_eq!(
                EXPECTED_ALLOCATION,
                20 * 1024 + 4096 + 12 * 1024 + 520 * 1024 + 12 * 1024
            );
            assert_eq!(
                config().allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?,
                EXPECTED_ALLOCATION,
                "assembled library charge differs from the preregistered diagnostic"
            );
            assert!(EXPECTED_ALLOCATION < GPU_BUDGET);
            assert!(EXPECTED_TENSOR_PEAK < u64::from(config().tensor_capacity_bytes));
            Ok(())
        }
    }
}

#[cfg(all(target_os = "linux", feature = "hag-native"))]
fn main() -> anyhow::Result<()> {
    proof::main()
}

#[cfg(not(all(target_os = "linux", feature = "hag-native")))]
fn main() {
    eprintln!("Future HAG finite diagnostic requires Linux and hag-native; no work submitted.");
    std::process::exit(2);
}
