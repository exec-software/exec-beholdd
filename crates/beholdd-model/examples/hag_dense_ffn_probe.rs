//! Opt-in native HAG gate for one packed Q8_0 dense-Qwen3 FFN layer.
//!
//! Default inspection binds metadata only. Native execution needs an explicit mode,
//! HAG_TRACE_SUBMISSION=0, and an external process/thermal supervisor. The diagnostic
//! first qualifies a 65x32 projection across two workgroups under a 512 KiB HAG cap.
//! Model execution compares three changed synthetic inputs with the immediate CPU
//! reference: |error| <= 3e-4 + 3e-5 * |reference|, finite values, nonzero signs and
//! signed exact zeros. These preregistered finite-fixture bounds do not establish
//! full-model quality, lossless inference, throughput, or serving readiness.

#[cfg(target_os = "linux")]
mod proof {
    use std::ffi::{OsStr, OsString};
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicU64, Ordering};

    use anyhow::{bail, ensure, Context, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::{CommandList, ExecutionDevice, LinearMode};
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::{Device, Weight};
    use beholdd_format::{CheckpointId, GgmlType, Gguf, WeightRange};
    use beholdd_memory::{
        check_checkpoint_storage, ResidentSource, ResidentWeight, StorageOverride, WeightAccess,
        WeightSource,
    };
    use beholdd_model::qwen3moe::{DenseFfnLimits, DenseFfnMemory};
    use beholdd_model::QwenModel;
    use serde_json::json;

    const USAGE: &str = "hag_dense_ffn_probe [--model PATH] [--layer N]\n\
        hag_dense_ffn_probe --diagnose-first /dev/dri/renderD128\n\
        hag_dense_ffn_probe --execute --model PATH [--layer N] /dev/dri/renderD128\n\
        Default: metadata inspection only; no native device access. Layer defaults to 0.\n\
        Native modes require HAG_TRACE_SUBMISSION=0 and external process/thermal limits.\n\
        Diagnostic: one 65x32 Q8_0 dispatch, two workgroups, 512 KiB HAG cap.\n\
        Model: three reused four-command FFN submissions, 12 dispatches, 32 MiB HAG cap.";
    const MAX_DIM: usize = 4096;
    const MAX_PACKED: u64 = 16 * 1024 * 1024;
    const MAX_TENSOR: u64 = 128 * 1024;
    const MAX_STAGING: u64 = 4 * 1024 * 1024;
    const AUXILIARY_CAPACITY: u32 = 32 * 1024;
    const MODEL_BUDGET: u64 = 32 * 1024 * 1024;
    const DIAGNOSTIC_BUDGET: u64 = 512 * 1024;
    const ABSOLUTE_TOLERANCE: f64 = 3e-4;
    const RELATIVE_TOLERANCE: f64 = 3e-5;
    const ROUNDS: usize = 3;
    const DIAGNOSTIC_ROWS: usize = 65;
    const DIAGNOSTIC_COLS: usize = 32;
    const GUARD: f32 = -12345.5;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Mode {
        Inspect,
        Diagnose,
        Execute,
    }

    #[derive(Debug)]
    struct Options {
        mode: Mode,
        model: Option<PathBuf>,
        layer: usize,
        device: Option<PathBuf>,
        help: bool,
    }

    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Options> {
        let mut args = arguments.into_iter();
        let mut options = Options {
            mode: Mode::Inspect,
            model: None,
            layer: 0,
            device: None,
            help: false,
        };
        let mut layer_seen = false;
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--help" | "-h") => options.help = true,
                Some("--model") if options.model.is_none() => {
                    options.model = Some(PathBuf::from(args.next().context("--model needs PATH")?));
                }
                Some("--layer") if !layer_seen => {
                    options.layer = args
                        .next()
                        .context("--layer needs a nonnegative index")?
                        .to_str()
                        .context("layer index is not Unicode")?
                        .parse()
                        .context("--layer needs a nonnegative index")?;
                    layer_seen = true;
                }
                Some("--execute") if options.mode == Mode::Inspect => options.mode = Mode::Execute,
                Some("--diagnose-first") if options.mode == Mode::Inspect => {
                    options.mode = Mode::Diagnose;
                }
                Some(value) if !value.starts_with('-') && options.device.is_none() => {
                    options.device = Some(PathBuf::from(arg));
                }
                _ => bail!("unknown or repeated argument {arg:?}\n{USAGE}"),
            }
        }
        if options.help {
            return Ok(options);
        }
        match options.mode {
            Mode::Inspect => {
                ensure!(
                    options.device.is_none(),
                    "device path requires an explicit native mode"
                );
                ensure!(
                    options.model.is_some() || !layer_seen,
                    "--layer requires --model"
                );
            }
            Mode::Diagnose => {
                ensure!(
                    options.device.is_some(),
                    "diagnostic requires an explicit render node"
                );
                ensure!(
                    options.model.is_none() && !layer_seen,
                    "diagnostic takes no model or layer"
                );
            }
            Mode::Execute => {
                ensure!(
                    options.device.is_some(),
                    "execution requires an explicit render node"
                );
                ensure!(options.model.is_some(), "execution requires --model PATH");
            }
        }
        Ok(options)
    }

    // Exact opt-in checked before metadata loading, source reads or device opening.
    fn check_trace(value: Option<&OsStr>) -> Result<()> {
        ensure!(
            value == Some(OsStr::new("0")),
            "native qualification requires HAG_TRACE_SUBMISSION exactly 0; payload tracing is refused"
        );
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    struct Matrix {
        kind: GgmlType,
        rows: usize,
        cols: usize,
        bytes: u64,
    }

    #[derive(Clone, Copy, Debug)]
    struct Capacity {
        weights: u32,
        tensors: u32,
        memory: DenseFfnMemory,
    }

    fn packed_bytes(rows: usize, cols: usize) -> Result<u64> {
        ensure!(
            rows > 0 && cols > 0 && cols.is_multiple_of(32),
            "invalid Q8_0 shape"
        );
        let bytes = rows
            .checked_mul(cols / 32)
            .and_then(|n| n.checked_mul(34))
            .context("Q8_0 byte count overflows")?;
        u64::try_from(bytes).context("Q8_0 bytes do not fit u64")
    }

    fn inspect(
        dim: usize,
        width: usize,
        matrices: [Matrix; 3],
        memory: DenseFfnMemory,
    ) -> Result<Capacity> {
        ensure!(
            (1..=MAX_DIM).contains(&dim) && (1..=MAX_DIM).contains(&width),
            "native FFN dimensions must be 1..={MAX_DIM}"
        );
        ensure!(
            dim.is_multiple_of(32) && width.is_multiple_of(32),
            "native Q8_0 FFN dimensions require complete 32-element blocks"
        );
        let mut packed = 0u64;
        let mut aligned = 0u64;
        let mut staging = 0u64;
        for (matrix, (rows, cols)) in
            matrices
                .iter()
                .zip([(width, dim), (width, dim), (dim, width)])
        {
            ensure!(
                matrix.kind == GgmlType(8),
                "this qualification accepts only packed Q8_0 FFN weights"
            );
            ensure!(
                (matrix.rows, matrix.cols) == (rows, cols),
                "FFN descriptor shape differs from plan"
            );
            let bytes = packed_bytes(rows, cols)?;
            ensure!(
                matrix.bytes == bytes,
                "FFN descriptor byte length differs from Q8_0 geometry"
            );
            packed = packed
                .checked_add(bytes)
                .context("packed total overflows")?;
            aligned = aligned
                .checked_add(bytes.checked_add(3).context("alignment overflows")? & !3)
                .context("aligned packed total overflows")?;
            staging = staging.max(bytes);
        }
        let scratch = (width as u64)
            .checked_mul(8)
            .context("scratch bytes overflow")?;
        let tensors = (dim as u64)
            .checked_mul(8)
            .and_then(|n| n.checked_add(scratch))
            .context("tensor bytes overflow")?;
        ensure!(
            memory
                == DenseFfnMemory {
                    packed_weight_bytes: packed,
                    largest_staging_bytes: staging,
                    scratch_bytes: scratch,
                },
            "logical plan costs differ from independently checked Q8_0 costs"
        );
        ensure!(
            packed <= MAX_PACKED,
            "packed weights exceed 16 MiB qualification limit"
        );
        ensure!(
            staging <= MAX_STAGING,
            "weight staging exceeds 4 MiB qualification limit"
        );
        ensure!(
            tensors <= MAX_TENSOR,
            "tensor payloads exceed 128 KiB qualification limit"
        );
        Ok(Capacity {
            weights: u32::try_from(aligned).context("weight capacity does not fit u32")?,
            tensors: u32::try_from(tensors).context("tensor capacity does not fit u32")?,
            memory,
        })
    }

    fn matrices(checkpoint: &Gguf, layer: usize) -> Result<[Matrix; 3]> {
        let descriptor = |name: &str| -> Result<Matrix> {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.{name}.weight"))
                .with_context(|| format!("missing {name} weight"))?;
            let [cols, rows] = tensor.dims() else {
                bail!("{name} must be a rank-two matrix");
            };
            Ok(Matrix {
                kind: tensor.kind(),
                rows: usize::try_from(*rows).context("rows do not fit usize")?,
                cols: usize::try_from(*cols).context("columns do not fit usize")?,
                bytes: tensor.nbytes(),
            })
        };
        Ok([
            descriptor("ffn_gate")?,
            descriptor("ffn_up")?,
            descriptor("ffn_down")?,
        ])
    }

    struct CountingSource<'a> {
        inner: ResidentSource<'a>,
        reads: AtomicU64,
        bytes: AtomicU64,
    }

    impl WeightSource for CountingSource<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.inner.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            let weight = self.inner.load_range(range, access)?;
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes
                .fetch_add(weight.bytes().len() as u64, Ordering::Relaxed);
            Ok(weight)
        }
    }

    fn inputs(dim: usize, round: usize) -> Vec<f32> {
        (0..dim)
            .map(|index| (((index % 31) * 7 + round * 3) % 31) as f32 / 32.0 - 0.5)
            .collect()
    }

    // Same immediate CPU path as recorded_dense_ffn, one worker and one staged
    // packed matrix at a time. These oracle reads are separate from the three
    // instrumented native-preparation source reads, and finish before device open.
    fn immediate_reference(
        checkpoint: &Gguf,
        layer: usize,
        input: &[f32],
        width: usize,
    ) -> Result<Vec<f32>> {
        let cpu = Cpu::new();
        let project = |name: &str, input: &[f32], output: &mut [f32]| -> Result<()> {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.{name}.weight"))
                .with_context(|| format!("missing reference weight {name}"))?;
            let storage = ResidentWeight::load(checkpoint, tensor.locator())?;
            cpu.matvec_with_threads(
                output,
                input,
                &Weight {
                    bytes: storage.bytes(),
                    kind: tensor.kind(),
                    rows: output.len(),
                    cols: input.len(),
                },
                1,
            )?;
            Ok(())
        };
        let mut gate = vec![0.0; width];
        let mut up = vec![0.0; width];
        project("ffn_gate", input, &mut gate)?;
        project("ffn_up", input, &mut up)?;
        cpu.swiglu(&mut gate, &up, 0.0)?;
        let mut output = vec![0.0; input.len()];
        project("ffn_down", &gate, &mut output)?;
        ensure!(
            output.iter().all(|value| value.is_finite()),
            "CPU reference is nonfinite"
        );
        Ok(output)
    }

    fn compare(actual: &[f32], expected: &[f32]) -> Result<f64> {
        ensure!(actual.len() == expected.len(), "comparison length differs");
        let mut maximum = 0.0_f64;
        for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(observed) - f64::from(reference)).abs();
            let allowed = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * f64::from(reference).abs();
            ensure!(observed.is_finite() && reference.is_finite() && error <= allowed,
                "output {index}: got {observed:?}, reference {reference:?}, error {error}, allowed {allowed}");
            ensure!(
                observed.is_sign_negative() == reference.is_sign_negative()
                    && (observed == 0.0) == (reference == 0.0),
                "output {index}: zero/sign mismatch: got {observed:?}, reference {reference:?}"
            );
            maximum = maximum.max(error);
        }
        Ok(maximum)
    }

    fn verify_submission(
        before: NativeExecutionStats,
        after: NativeExecutionStats,
        dispatches: u64,
    ) -> Result<()> {
        ensure!(
            after.dispatch_attempts == before.dispatch_attempts + dispatches
                && after.dispatch_count == before.dispatch_count + dispatches
                && after.logical_submissions == before.logical_submissions + 1,
            "submission/dispatch accounting differs"
        );
        ensure!(
            after.host_read_bytes == before.host_read_bytes
                && after.host_write_bytes == before.host_write_bytes,
            "submit/wait transferred payload bytes"
        );
        ensure!(
            after.uploaded_weight_bytes == before.uploaded_weight_bytes
                && after.allocation_bytes == before.allocation_bytes
                && after.weight_live_bytes == before.weight_live_bytes
                && after.tensor_live_bytes == before.tensor_live_bytes
                && after.auxiliary_live_bytes == before.auxiliary_live_bytes
                && !after.poisoned,
            "resident allocations changed or queue poisoned during submission"
        );
        Ok(())
    }

    fn verify_initial(
        device: &NativeExecutionDevice,
        budget: &MemoryBudget,
        cap: u64,
    ) -> Result<NativeExecutionStats> {
        let stats = device.stats()?;
        ensure!(
            stats.allocation_bytes > 0
                && stats.allocation_bytes == budget.current_bytes()
                && stats.allocation_bytes <= cap,
            "HAG budget accounting differs"
        );
        ensure!(
            stats.dispatch_attempts == 0
                && stats.dispatch_count == 0
                && stats.logical_submissions == 0
                && stats.host_read_bytes == 0
                && !stats.poisoned,
            "unexpected initial queue activity"
        );
        Ok(stats)
    }

    fn verify_retired(device: &NativeExecutionDevice) -> Result<()> {
        let retired = device.stats()?;
        ensure!(
            retired.weight_live_bytes == 0
                && retired.tensor_live_bytes == 0
                && retired.auxiliary_live_bytes == 0
                && !retired.poisoned,
            "resources did not retire cleanly"
        );
        Ok(())
    }

    fn checked_allocation_bytes(config: NativeExecutionConfig, budget: u64) -> Result<u64> {
        let bytes = config.allocation_bytes()?;
        ensure!(
            bytes <= budget,
            "native queue requires {bytes} bytes, HAG budget is {budget}"
        );
        Ok(bytes)
    }

    fn run_model(options: &Options) -> Result<()> {
        let path = options.model.as_ref().context("missing model")?;
        check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
        let checkpoint = Gguf::open(path)?;
        check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
        let model = QwenModel::load(&checkpoint)?;
        let plan = model.dense_ffn_plan(options.layer)?;
        // Every geometry, format and logical limit is checked before source reads
        // or native device construction; plan.prepare repeats its logical limits.
        let capacity = inspect(
            plan.dim(),
            plan.feed_forward_length(),
            matrices(&checkpoint, options.layer)?,
            plan.memory(),
        )?;
        let config = NativeExecutionConfig {
            weight_capacity_bytes: capacity.weights,
            tensor_capacity_bytes: capacity.tensors,
            auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
        };
        let expected_allocation_bytes = checked_allocation_bytes(config, MODEL_BUDGET)?;
        let mut report = json!({
            "execution": "not_requested", "architecture": "qwen3", "layer": options.layer,
            "input_width": plan.dim(), "ffn_width": plan.feed_forward_length(), "weight_format": "Q8_0",
            "packed_weight_bytes": capacity.memory.packed_weight_bytes,
            "largest_staging_bytes": capacity.memory.largest_staging_bytes,
            "scratch_bytes": capacity.memory.scratch_bytes,
            "weight_arena_capacity_bytes": capacity.weights, "tensor_arena_capacity_bytes": capacity.tensors,
            "auxiliary_arena_capacity_bytes": AUXILIARY_CAPACITY, "hag_budget_bytes": MODEL_BUDGET,
            "expected_hag_allocation_bytes": expected_allocation_bytes,
            "required_hag_trace_submission": "0",
            "hag_trace_requirement_verified": options.mode == Mode::Execute,
            "external_process_and_thermal_limits_required": true,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
            "oracle": "immediate CPU, one worker, finite/sign/signed-zero checks"
        });
        if options.mode == Mode::Inspect {
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        println!("RUN preregistered numerical bound: |error| <= {ABSOLUTE_TOLERANCE} + {RELATIVE_TOLERANCE} * |CPU reference|; finite/sign/signed-zero checks");
        let expected: Vec<Vec<f32>> = (0..ROUNDS)
            .map(|round| {
                immediate_reference(
                    &checkpoint,
                    options.layer,
                    &inputs(plan.dim(), round),
                    plan.feed_forward_length(),
                )
            })
            .collect::<Result<_>>()?;
        let budget = MemoryBudget::new(MODEL_BUDGET);
        let device = NativeExecutionDevice::open(
            options.device.as_ref().context("missing device")?,
            &budget,
            config,
        )?;
        let source = CountingSource {
            inner: ResidentSource::new(&checkpoint),
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        };
        let prepared = plan.prepare(
            &device,
            &source,
            DenseFfnLimits {
                packed_weight_bytes: capacity.memory.packed_weight_bytes,
                staging_bytes: capacity.memory.largest_staging_bytes,
                scratch_bytes: capacity.memory.scratch_bytes,
            },
        )?;
        ensure!(
            source.reads.load(Ordering::Relaxed) == 3
                && source.bytes.load(Ordering::Relaxed) == capacity.memory.packed_weight_bytes,
            "native preparation must read exactly three unchanged packed weights"
        );
        let input = device.allocate_f32(plan.dim())?;
        let output = device.allocate_f32(plan.dim())?;
        let batch = prepared.record(&device, &input.full_view(), &output.full_view())?;
        ensure!(
            batch.len() == 4,
            "expected gate/up/SwiGLU/down semantic commands"
        );
        let initial = verify_initial(&device, &budget, MODEL_BUDGET)?;
        ensure!(
            initial.allocation_bytes == expected_allocation_bytes,
            "actual model allocation differs from preflight cost"
        );
        ensure!(
            initial.weight_live_bytes == u64::from(capacity.weights)
                && initial.uploaded_weight_bytes == capacity.memory.packed_weight_bytes
                && initial.tensor_live_bytes == u64::from(capacity.tensors)
                && initial.auxiliary_live_bytes == 4,
            "prepared resource accounting differs"
        );
        let mut maximum_error = 0.0_f64;
        let mut checksum = 0u64;
        for (round, expected) in expected.iter().enumerate() {
            device.write_f32(&input.full_view(), &inputs(plan.dim(), round))?;
            let before = device.stats()?;
            println!("RUN dense FFN round={round} semantic_commands=4 physical_dispatches=4");
            let submission = device.submit(&batch)?;
            device.wait(&submission)?;
            let after = device.stats()?;
            verify_submission(before, after, 4)?;
            ensure!(
                budget.current_bytes() == initial.allocation_bytes,
                "GPU budget changed on reuse"
            );
            let mut actual = vec![f32::NAN; plan.dim()];
            device.read_f32(&output.full_view(), &mut actual)?;
            maximum_error = maximum_error
                .max(compare(&actual, expected).with_context(|| format!("FFN round {round}"))?);
            checksum = actual.iter().fold(checksum, |sum, value| {
                sum.wrapping_add(u64::from(value.to_bits()))
            });
            println!(
                "PASS dense FFN round={round} checked_values={} submission_tensor_transfer_bytes=0",
                actual.len()
            );
        }
        let final_stats = device.stats()?;
        ensure!(
            final_stats.dispatch_count == 12
                && final_stats.dispatch_attempts == 12
                && final_stats.logical_submissions == 3,
            "expected 12 physical dispatches over three reused submissions"
        );
        ensure!(
            final_stats.uploaded_weight_bytes == capacity.memory.packed_weight_bytes
                && source.reads.load(Ordering::Relaxed) == 3
                && source.bytes.load(Ordering::Relaxed) == capacity.memory.packed_weight_bytes,
            "reused model batch reread or reuploaded weights"
        );
        let transfer_bytes = (plan.dim() * 4 * ROUNDS) as u64;
        ensure!(
            final_stats.host_read_bytes == transfer_bytes
                && final_stats.host_write_bytes == initial.host_write_bytes + transfer_bytes,
            "boundary tensor traffic differs"
        );
        ensure!(
            final_stats.weight_peak_bytes == initial.weight_peak_bytes
                && final_stats.tensor_peak_bytes == initial.tensor_peak_bytes
                && final_stats.auxiliary_peak_bytes == initial.auxiliary_peak_bytes,
            "reuse grew resident resources"
        );
        drop((batch, input, output));
        verify_retired(&device)?;
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "HAG retained allocations after model qualification"
        );
        report["execution"] = json!("native-hag-dense-ffn");
        report["rounds"] = json!(ROUNDS);
        report["physical_dispatches"] = json!(final_stats.dispatch_count);
        report["native_preparation_source_reads"] = json!(3);
        report["cpu_oracle_source_reads"] = json!(3 * ROUNDS);
        report["submission_tensor_transfer_bytes"] = json!(0);
        report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
        report["actual_hag_allocation_bytes"] = json!(initial.allocation_bytes);
        report["max_absolute_error"] = json!(maximum_error);
        report["output_checksum"] = json!(checksum);
        report["budget_after"] = json!(0);
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    fn diagnostic_scale(row: usize) -> (u16, f32) {
        match row % 3 {
            0 => (0x2800, 1.0 / 32.0),
            1 => (0xa400, -1.0 / 64.0),
            _ => (0x2c00, 1.0 / 16.0),
        }
    }

    fn diagnostic_quant(row: usize, column: usize) -> i8 {
        ((row * 11 + column * 5) % 15) as i8 - 7
    }

    fn diagnostic_fixture() -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let input: Vec<f32> = (0..DIAGNOSTIC_COLS)
            .map(|column| ((column * 7 % 19) as f32 - 9.0) / 8.0)
            .collect();
        let mut packed = Vec::with_capacity(DIAGNOSTIC_ROWS * 34);
        let mut expected = Vec::with_capacity(DIAGNOSTIC_ROWS);
        for row in 0..DIAGNOSTIC_ROWS {
            let (bits, scale) = diagnostic_scale(row);
            packed.extend_from_slice(&bits.to_le_bytes());
            let mut sum = 0.0_f32;
            for (column, value) in input.iter().enumerate() {
                let quant = diagnostic_quant(row, column);
                packed.push(quant as u8);
                // Separate ordered f32 multiply/add, matching the audited serial shader.
                // All operands and partial sums are exact dyadic values in this fixture.
                let weight = scale * f32::from(quant);
                let product = weight * value;
                sum += product;
            }
            expected.push(sum);
        }
        (packed, input, expected)
    }

    fn diagnose(path: &Path) -> Result<()> {
        let (packed, values, expected) = diagnostic_fixture();
        let mut cpu = vec![f32::NAN; DIAGNOSTIC_ROWS];
        Cpu::new().matvec_with_threads(
            &mut cpu,
            &values,
            &Weight {
                bytes: &packed,
                kind: GgmlType(8),
                rows: DIAGNOSTIC_ROWS,
                cols: DIAGNOSTIC_COLS,
            },
            1,
        )?;
        ensure!(
            cpu.iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "diagnostic CPU/ordered oracle disagree"
        );
        let config = NativeExecutionConfig {
            weight_capacity_bytes: (packed.len() as u32 + 3) & !3,
            tensor_capacity_bytes: ((DIAGNOSTIC_COLS + DIAGNOSTIC_ROWS + 2) * 4) as u32,
            auxiliary_capacity_bytes: 4096,
        };
        let expected_allocation_bytes = checked_allocation_bytes(config, DIAGNOSTIC_BUDGET)?;
        let budget = MemoryBudget::new(DIAGNOSTIC_BUDGET);
        let device = NativeExecutionDevice::open(path, &budget, config)?;
        let encoded = device.upload_weight(&Weight {
            bytes: &packed,
            kind: GgmlType(8),
            rows: DIAGNOSTIC_ROWS,
            cols: DIAGNOSTIC_COLS,
        })?;
        let weight = device.prepare_weight(&encoded)?;
        let input = device.upload_f32(&values)?;
        let output = device.upload_f32(&[GUARD; DIAGNOSTIC_ROWS + 2])?;
        let mut commands = CommandList::new();
        commands.linear(
            &output.view(1, DIAGNOSTIC_ROWS)?,
            &input.full_view(),
            &weight,
            LinearMode::Overwrite,
        )?;
        let batch = device.record(&commands)?;
        drop((commands, weight, encoded));
        let initial = verify_initial(&device, &budget, DIAGNOSTIC_BUDGET)?;
        ensure!(
            initial.allocation_bytes == expected_allocation_bytes,
            "actual diagnostic allocation differs from preflight cost"
        );
        ensure!(
            batch.len() == 1
                && initial.uploaded_weight_bytes == packed.len() as u64
                && initial.weight_live_bytes == ((packed.len() as u64 + 3) & !3)
                && initial.tensor_live_bytes
                    == ((DIAGNOSTIC_COLS + DIAGNOSTIC_ROWS + 2) * 4) as u64
                && initial.auxiliary_live_bytes == 0,
            "diagnostic preparation accounting differs"
        );
        println!("RUN Q8_0 diagnostic rows=65 cols=32 workgroups=2 physical_dispatches=1");
        let submission = device.submit(&batch)?;
        device.wait(&submission)?;
        verify_submission(initial, device.stats()?, 1)?;
        ensure!(
            budget.current_bytes() == initial.allocation_bytes,
            "diagnostic budget changed"
        );
        let mut actual = [f32::NAN; DIAGNOSTIC_ROWS + 2];
        device.read_f32(&output.full_view(), &mut actual)?;
        ensure!(
            actual[0].to_bits() == GUARD.to_bits()
                && actual[DIAGNOSTIC_ROWS + 1].to_bits() == GUARD.to_bits(),
            "diagnostic output canary changed"
        );
        for (row, (actual, expected)) in
            actual[1..=DIAGNOSTIC_ROWS].iter().zip(expected).enumerate()
        {
            ensure!(
                actual.is_finite() && actual.to_bits() == expected.to_bits(),
                "diagnostic row {row}: got {actual:?}, ordered f32 reference {expected:?}"
            );
        }
        let final_stats = device.stats()?;
        ensure!(
            final_stats.host_read_bytes == ((DIAGNOSTIC_ROWS + 2) * 4) as u64,
            "diagnostic readback accounting differs"
        );
        drop((submission, batch, input, output));
        verify_retired(&device)?;
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "diagnostic retained HAG allocations"
        );
        println!("PASS Q8_0 two-workgroup diagnostic checked_values=65 guard_values=2 physical_dispatches=1 submission_tensor_transfer_bytes=0 actual_hag_allocation_bytes={} HAG_TRACE_SUBMISSION=0 budget_after=0", initial.allocation_bytes);
        Ok(())
    }

    pub(super) fn run() -> Result<()> {
        let options = parse(std::env::args_os().skip(1))?;
        if options.help || (options.mode == Mode::Inspect && options.model.is_none()) {
            println!("{USAGE}");
            return Ok(());
        }
        if options.mode != Mode::Inspect {
            check_trace(std::env::var_os("HAG_TRACE_SUBMISSION").as_deref())?;
        }
        if options.mode == Mode::Diagnose {
            diagnose(options.device.as_ref().context("missing device")?)
        } else {
            run_model(&options)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn arguments(values: &[&str]) -> Result<Options> {
            parse(values.iter().map(OsString::from))
        }

        fn metadata(dim: usize, width: usize) -> ([Matrix; 3], DenseFfnMemory) {
            let bytes = packed_bytes(dim, width).unwrap();
            (
                [
                    Matrix {
                        kind: GgmlType(8),
                        rows: width,
                        cols: dim,
                        bytes,
                    },
                    Matrix {
                        kind: GgmlType(8),
                        rows: width,
                        cols: dim,
                        bytes,
                    },
                    Matrix {
                        kind: GgmlType(8),
                        rows: dim,
                        cols: width,
                        bytes,
                    },
                ],
                DenseFfnMemory {
                    packed_weight_bytes: bytes * 3,
                    largest_staging_bytes: bytes,
                    scratch_bytes: width as u64 * 8,
                },
            )
        }

        #[test]
        fn native_access_requires_explicit_mode_device_and_exact_trace_opt_out() {
            assert_eq!(arguments(&[]).unwrap().mode, Mode::Inspect);
            assert_eq!(
                arguments(&["--model", "fixture.gguf"]).unwrap().mode,
                Mode::Inspect
            );
            assert!(arguments(&["--help"]).unwrap().help);
            for args in [
                vec!["/dev/dri/renderD128"],
                vec!["--execute", "/dev/dri/renderD128"],
                vec!["--execute", "--model", "fixture.gguf"],
                vec!["--diagnose-first"],
                vec!["--diagnose-first", "--execute", "/dev/dri/renderD128"],
                vec![
                    "--diagnose-first",
                    "--model",
                    "fixture.gguf",
                    "/dev/dri/renderD128",
                ],
            ] {
                assert!(arguments(&args).is_err(), "accepted {args:?}");
            }
            for value in [
                None,
                Some(OsStr::new("1")),
                Some(OsStr::new("")),
                Some(OsStr::new("false")),
                Some(OsStr::new("00")),
            ] {
                assert!(check_trace(value).is_err());
            }
            check_trace(Some(OsStr::new("0"))).unwrap();
        }

        #[test]
        fn selected_model_cost_is_checked_without_loading_weights() {
            let (matrices, memory) = metadata(1024, 3072);
            let capacity = inspect(1024, 3072, matrices, memory).unwrap();
            assert_eq!(capacity.weights, 10_027_008);
            assert_eq!(capacity.tensors, 32_768);
            assert_eq!(memory.largest_staging_bytes, 3_342_336);
            assert_eq!(memory.scratch_bytes, 24_576);
        }

        #[test]
        fn preflight_checks_page_rounded_charge_before_device_open() {
            let config = NativeExecutionConfig {
                weight_capacity_bytes: 10_027_008,
                tensor_capacity_bytes: 32_768,
                auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
            };
            let bytes = checked_allocation_bytes(config, MODEL_BUDGET).unwrap();
            assert_eq!(bytes, 10_129_408);
            assert!(checked_allocation_bytes(config, bytes - 1).is_err());
            assert_eq!(checked_allocation_bytes(config, bytes).unwrap(), bytes);
        }

        #[test]
        fn inspector_refuses_format_shape_span_memory_and_cost_mismatch() {
            let (matrices, memory) = metadata(1024, 3072);
            for index in 0..3 {
                let mut unsupported = matrices;
                unsupported[index].kind = GgmlType(30);
                assert!(inspect(1024, 3072, unsupported, memory).is_err());
                let mut shape = matrices;
                shape[index].cols += 32;
                assert!(inspect(1024, 3072, shape, memory).is_err());
                let mut short = matrices;
                short[index].bytes -= 1;
                assert!(inspect(1024, 3072, short, memory).is_err());
            }
            for (dim, width) in [
                (0, 3072),
                (1024, 0),
                (1023, 3072),
                (1024, 4097),
                (usize::MAX, 3072),
            ] {
                assert!(inspect(dim, width, matrices, memory).is_err());
            }
            for wrong in [
                DenseFfnMemory {
                    packed_weight_bytes: memory.packed_weight_bytes + 1,
                    ..memory
                },
                DenseFfnMemory {
                    largest_staging_bytes: 0,
                    ..memory
                },
                DenseFfnMemory {
                    scratch_bytes: 0,
                    ..memory
                },
            ] {
                assert!(inspect(1024, 3072, matrices, wrong).is_err());
            }
            let (large, costs) = metadata(4096, 4096);
            assert!(inspect(4096, 4096, large, costs).is_err());
            let (staging, costs) = metadata(2048, 2048);
            assert!(
                costs.packed_weight_bytes <= MAX_PACKED
                    && costs.largest_staging_bytes > MAX_STAGING
            );
            assert!(inspect(2048, 2048, staging, costs).is_err());
            assert!(packed_bytes(usize::MAX, 4096).is_err());
        }

        #[test]
        fn diagnostic_q8_fixture_matches_cpu_exactly_across_workgroup_boundary() {
            let (packed, input, expected) = diagnostic_fixture();
            assert_eq!(packed.len(), 65 * 34);
            assert_ne!(expected[0], expected[64]);
            assert!(expected.iter().any(|value| *value < 0.0));
            assert!(expected.iter().any(|value| *value > 0.0));
            let mut actual = vec![f32::NAN; 65];
            Cpu::new()
                .matvec_with_threads(
                    &mut actual,
                    &input,
                    &Weight {
                        bytes: &packed,
                        kind: GgmlType(8),
                        rows: 65,
                        cols: 32,
                    },
                    1,
                )
                .unwrap();
            assert!(actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits()));
        }

        #[test]
        fn finite_tolerance_keeps_sign_and_signed_zero_checks() {
            compare(&[1.0001, -1.0001, 0.0, -0.0], &[1.0, -1.0, 0.0, -0.0]).unwrap();
            for (actual, expected) in [
                (f32::NAN, 1.0),
                (f32::INFINITY, 1.0),
                (1.0, f32::NAN),
                (0.0, -0.0),
                (-0.0, 0.0),
                (0.0, 1e-8),
                (-1e-8, 1e-8),
                (1e-8, 0.0),
                (1.001, 1.0),
            ] {
                assert!(compare(&[actual], &[expected]).is_err());
            }
            assert!(compare(&[], &[1.0]).is_err());
        }

        #[test]
        fn three_deterministic_model_inputs_are_distinct_and_bounded() {
            let values: Vec<_> = (0..ROUNDS).map(|round| inputs(1024, round)).collect();
            for (index, input) in values.iter().enumerate() {
                assert_eq!(input.len(), 1024);
                assert!(input
                    .iter()
                    .all(|x| x.is_finite() && (-0.5..0.5).contains(x)));
                assert!(values.iter().skip(index + 1).all(|other| input != other));
            }
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native HAG dense FFN qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
