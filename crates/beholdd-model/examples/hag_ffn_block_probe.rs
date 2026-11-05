//! Opt-in native HAG gate for one weighted-RMSNorm, residual-connected Qwen3 FFN block.
//!
//! Default inspection binds metadata only. Native execution needs an explicit mode,
//! HAG_TRACE_SUBMISSION=0, and an external process/thermal supervisor. The diagnostic
//! qualifies unweighted one-element RMSNorm first, then weighted 65-element reuse.
//! RMSNorm uses 2e-6 + 2e-6 * |reference|. The block uses the existing FFN bound.
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
    use beholdd_backend::execution::{CommandList, ExecutionDevice};
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::{Device, Weight};
    use beholdd_format::{CheckpointId, GgmlType, Gguf, WeightRange};
    use beholdd_memory::{
        check_checkpoint_storage, ResidentSource, ResidentWeight, StorageOverride, WeightAccess,
        WeightSource,
    };
    use beholdd_model::qwen3moe::{DenseFfnBlockLimits, DenseFfnBlockMemory};
    use beholdd_model::QwenModel;
    use serde_json::json;

    const USAGE: &str = "hag_ffn_block_probe [--model PATH] [--layer N]\n\
        hag_ffn_block_probe --diagnose-first /dev/dri/renderD128\n\
        hag_ffn_block_probe --execute --model PATH [--layer N] /dev/dri/renderD128\n\
        Default: metadata inspection only; no native device access. Layer defaults to 0.\n\
        Native modes require HAG_TRACE_SUBMISSION=0 and external process/thermal limits.\n\
        Diagnostic: one unweighted1 then three weighted65 RMSNorm submissions, seven dispatches, 512 KiB HAG cap.\n\
        Model: three reused five-command FFN block submissions, 21 dispatches, 32 MiB HAG cap.";
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
    const RMS_ABSOLUTE_TOLERANCE: f64 = 2e-6;
    const RMS_RELATIVE_TOLERANCE: f64 = 2e-6;
    const RMS_EPS: f32 = 1e-5;
    const WEIGHTED_LEN: usize = 65;
    const UNWEIGHTED_LEN: usize = 1;
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
    struct Norm {
        kind: GgmlType,
        len: usize,
        bytes: u64,
    }

    fn norm_descriptor(checkpoint: &Gguf, layer: usize) -> Result<Norm> {
        let tensor = checkpoint
            .get(&format!("blk.{layer}.ffn_norm.weight"))
            .context("missing block normalization weight")?;
        let [len] = tensor.dims() else {
            bail!("block normalization weight must be rank one");
        };
        Ok(Norm {
            kind: tensor.kind(),
            len: usize::try_from(*len).context("norm width exceeds usize")?,
            bytes: tensor.nbytes(),
        })
    }

    #[derive(Clone, Copy, Debug)]
    struct Capacity {
        weights: u32,
        tensors: u32,
        memory: DenseFfnBlockMemory,
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

    fn block_auxiliary_bytes(norm_bytes: u64, eps: f32) -> Result<u64> {
        // RMSNorm and the residual projection reuse one batch-private workspace.
        // Epsilon, the SwiGLU limit and residual scale are interned by exact bits.
        let mut scalars = [eps.to_bits(), 0.0_f32.to_bits(), 1.0_f32.to_bits()];
        scalars.sort_unstable();
        let words = 1 + scalars.windows(2).filter(|pair| pair[0] != pair[1]).count();
        norm_bytes
            .checked_add(words as u64 * 4)
            .context("block auxiliary bytes overflow")
    }

    fn inspect(
        dim: usize,
        width: usize,
        matrices: [Matrix; 3],
        norm: Norm,
        eps: f32,
        memory: DenseFfnBlockMemory,
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
        let norm_bytes = (dim as u64).checked_mul(4).context("norm bytes overflow")?;
        ensure!(
            norm.kind == GgmlType(0) && norm.len == dim && norm.bytes == norm_bytes,
            "qualification requires an exact F32 norm vector matching hidden width"
        );
        ensure!(
            eps.is_finite() && eps >= 0.0,
            "RMSNorm epsilon must be finite and nonnegative"
        );
        let resident = (width as u64)
            .checked_mul(8)
            .and_then(|n| n.checked_add(norm_bytes * 2))
            .context("resident tensor bytes overflow")?;
        let tensors = resident
            .checked_add(norm_bytes)
            .context("hidden tensor bytes overflow")?;
        staging = staging.max(
            norm_bytes
                .checked_mul(2)
                .context("norm staging bytes overflow")?,
        );
        ensure!(
            memory
                == DenseFfnBlockMemory {
                    packed_weight_bytes: packed,
                    largest_staging_bytes: staging,
                    resident_tensor_bytes: resident,
                },
            "block plan costs differ from independently checked matrix/norm costs"
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

    // Independent f64 reduction with the CPU reference's f32 scale/application
    // rounding. The immediate matrix path remains one-worker packed Q8_0; each
    // source handle drops before the next read and all oracle work precedes GPU open.
    fn norm_reference(input: &[f32], weights: Option<&[f32]>, eps: f32) -> Result<Vec<f32>> {
        ensure!(
            !input.is_empty() && input.iter().all(|x| x.is_finite()),
            "invalid RMSNorm input"
        );
        ensure!(eps.is_finite() && eps >= 0.0, "invalid RMSNorm epsilon");
        if let Some(weights) = weights {
            ensure!(
                weights.len() == input.len() && weights.iter().all(|x| x.is_finite()),
                "invalid RMSNorm weights"
            );
        }
        let squares: f64 = input.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        let inverse = (1.0 / (squares / input.len() as f64 + f64::from(eps)).sqrt()) as f32;
        ensure!(
            inverse.is_finite(),
            "RMSNorm reference has nonfinite inverse"
        );
        let output: Vec<_> = input
            .iter()
            .enumerate()
            .map(|(i, x)| *x * inverse * weights.map_or(1.0, |weights| weights[i]))
            .collect();
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "RMSNorm reference is nonfinite"
        );
        Ok(output)
    }

    fn immediate_reference(
        checkpoint: &Gguf,
        layer: usize,
        hidden: &[f32],
        width: usize,
        eps: f32,
    ) -> Result<Vec<f32>> {
        let cpu = Cpu::new();
        let norm = checkpoint
            .get(&format!("blk.{layer}.ffn_norm.weight"))
            .context("missing norm reference")?;
        let normalized = {
            let storage = ResidentWeight::load(checkpoint, norm.locator())?;
            let weights: Vec<_> = storage
                .bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect();
            norm_reference(hidden, Some(&weights), eps)?
        };
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
        project("ffn_gate", &normalized, &mut gate)?;
        project("ffn_up", &normalized, &mut up)?;
        cpu.swiglu(&mut gate, &up, 0.0)?;
        let mut contribution = vec![0.0; hidden.len()];
        project("ffn_down", &gate, &mut contribution)?;
        let output: Vec<_> = hidden
            .iter()
            .zip(contribution)
            .map(|(residual, projected)| *residual + projected)
            .collect();
        ensure!(
            output.iter().all(|value| value.is_finite()),
            "CPU block reference is nonfinite"
        );
        Ok(output)
    }

    fn compare(actual: &[f32], expected: &[f32], absolute: f64, relative: f64) -> Result<f64> {
        ensure!(actual.len() == expected.len(), "comparison length differs");
        let mut maximum = 0.0_f64;
        for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(observed) - f64::from(reference)).abs();
            let allowed = absolute + relative * f64::from(reference).abs();
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
        let plan = model.dense_ffn_block_plan(options.layer)?;
        // Every geometry, format and logical limit is checked before source reads
        // or native device construction; plan.prepare repeats its logical limits.
        let eps = model.config().eps();
        let capacity = inspect(
            plan.dim(),
            plan.feed_forward_length(),
            matrices(&checkpoint, options.layer)?,
            norm_descriptor(&checkpoint, options.layer)?,
            eps,
            plan.memory(),
        )?;
        let config = NativeExecutionConfig {
            weight_capacity_bytes: capacity.weights,
            tensor_capacity_bytes: capacity.tensors,
            auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
        };
        let norm_bytes = (plan.dim() as u64) * 4;
        let source_bytes = capacity
            .memory
            .packed_weight_bytes
            .checked_add(norm_bytes)
            .context("source bytes overflow")?;
        let auxiliary_live_bytes = block_auxiliary_bytes(norm_bytes, eps)?;
        ensure!(
            auxiliary_live_bytes <= u64::from(AUXILIARY_CAPACITY),
            "block auxiliary payload exceeds fixed arena"
        );
        let expected_allocation_bytes = checked_allocation_bytes(config, MODEL_BUDGET)?;
        let mut report = json!({
            "execution": "not_requested", "architecture": "qwen3", "layer": options.layer,
            "input_width": plan.dim(), "ffn_width": plan.feed_forward_length(), "weight_format": "Q8_0",
            "packed_weight_bytes": capacity.memory.packed_weight_bytes,
            "largest_staging_bytes": capacity.memory.largest_staging_bytes,
            "resident_tensor_bytes": capacity.memory.resident_tensor_bytes,
            "source_weight_bytes": source_bytes, "expected_auxiliary_live_bytes": auxiliary_live_bytes,
            "norm_format": "F32", "rmsnorm_epsilon": eps,
            "weight_arena_capacity_bytes": capacity.weights, "tensor_arena_capacity_bytes": capacity.tensors,
            "auxiliary_arena_capacity_bytes": AUXILIARY_CAPACITY, "hag_budget_bytes": MODEL_BUDGET,
            "expected_hag_allocation_bytes": expected_allocation_bytes,
            "required_hag_trace_submission": "0",
            "hag_trace_requirement_verified": options.mode == Mode::Execute,
            "external_process_and_thermal_limits_required": true,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
            "oracle": "independent f64 RMS reduction, f32 scaling, immediate one-worker Q8 CPU, residual; finite/sign/signed-zero checks"
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
                    eps,
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
            DenseFfnBlockLimits {
                packed_weight_bytes: capacity.memory.packed_weight_bytes,
                largest_staging_bytes: capacity.memory.largest_staging_bytes,
                resident_tensor_bytes: capacity.memory.resident_tensor_bytes,
            },
        )?;
        ensure!(
            source.reads.load(Ordering::Relaxed) == 4
                && source.bytes.load(Ordering::Relaxed) == source_bytes,
            "native preparation must read exactly three packed matrices and one F32 norm"
        );
        let hidden = device.allocate_f32(plan.dim())?;
        let batch = prepared.record(&device, &hidden.full_view())?;
        ensure!(
            batch.len() == 5,
            "expected RMSNorm/gate/up/SwiGLU/residual-down semantic commands"
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
                && initial.auxiliary_live_bytes == auxiliary_live_bytes,
            "prepared resource accounting differs"
        );
        let mut maximum_error = 0.0_f64;
        let mut checksum = 0u64;
        for (round, expected) in expected.iter().enumerate() {
            device.write_f32(&hidden.full_view(), &inputs(plan.dim(), round))?;
            let before = device.stats()?;
            println!("RUN dense FFN block round={round} semantic_commands=5 physical_dispatches=7");
            let submission = device.submit(&batch)?;
            device.wait(&submission)?;
            let after = device.stats()?;
            verify_submission(before, after, 7)?;
            ensure!(
                budget.current_bytes() == initial.allocation_bytes,
                "GPU budget changed on reuse"
            );
            let mut actual = vec![f32::NAN; plan.dim()];
            device.read_f32(&hidden.full_view(), &mut actual)?;
            maximum_error = maximum_error.max(
                compare(&actual, expected, ABSOLUTE_TOLERANCE, RELATIVE_TOLERANCE)
                    .with_context(|| format!("FFN round {round}"))?,
            );
            checksum = actual.iter().fold(checksum, |sum, value| {
                sum.wrapping_add(u64::from(value.to_bits()))
            });
            println!(
                "PASS dense FFN block round={round} checked_values={} submission_tensor_transfer_bytes=0",
                actual.len()
            );
        }
        let final_stats = device.stats()?;
        ensure!(
            final_stats.dispatch_count == 21
                && final_stats.dispatch_attempts == 21
                && final_stats.logical_submissions == 3,
            "expected 21 physical dispatches over three reused submissions"
        );
        ensure!(
            final_stats.uploaded_weight_bytes == capacity.memory.packed_weight_bytes
                && source.reads.load(Ordering::Relaxed) == 4
                && source.bytes.load(Ordering::Relaxed) == source_bytes,
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
        drop((batch, hidden));
        verify_retired(&device)?;
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "HAG retained allocations after model qualification"
        );
        report["execution"] = json!("native-hag-ffn-block");
        report["rounds"] = json!(ROUNDS);
        report["semantic_commands_per_batch"] = json!(5);
        report["physical_dispatches"] = json!(final_stats.dispatch_count);
        report["logical_submissions"] = json!(final_stats.logical_submissions);
        report["auxiliary_live_bytes"] = json!(initial.auxiliary_live_bytes);
        report["native_preparation_source_reads"] = json!(4);
        report["cpu_oracle_source_reads"] = json!(4 * ROUNDS);
        report["submission_tensor_transfer_bytes"] = json!(0);
        report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
        report["actual_hag_allocation_bytes"] = json!(initial.allocation_bytes);
        report["max_absolute_error"] = json!(maximum_error);
        report["output_checksum"] = json!(checksum);
        report["budget_after"] = json!(0);
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    fn diagnostic_inputs(len: usize, round: usize) -> Vec<f32> {
        (0..len)
            .map(|i| {
                if i == 0 {
                    -0.0
                } else {
                    (((i * 7 + round * 3) % 23) as f32 - 11.0) / 8.0
                }
            })
            .collect()
    }

    fn diagnostic_weights(round: usize) -> Vec<f32> {
        (0..WEIGHTED_LEN)
            .map(|i| (((i * 5 + round * 7) % 17) as f32 - 8.0) / 8.0)
            .collect()
    }

    fn verify_diagnostic_output(actual: &[f32], expected: &[f32]) -> Result<f64> {
        ensure!(
            actual.len() == expected.len() + 2,
            "diagnostic length differs"
        );
        ensure!(
            actual[0].to_bits() == GUARD.to_bits()
                && actual[actual.len() - 1].to_bits() == GUARD.to_bits(),
            "RMSNorm output canary changed"
        );
        compare(
            &actual[1..actual.len() - 1],
            expected,
            RMS_ABSOLUTE_TOLERANCE,
            RMS_RELATIVE_TOLERANCE,
        )
    }

    fn diagnose(path: &Path) -> Result<()> {
        println!("RUN RMSNorm preregistered bound: |error| <= {RMS_ABSOLUTE_TOLERANCE} + {RMS_RELATIVE_TOLERANCE} * |reference|; finite/sign/signed-zero checks");
        let expected: Vec<_> = (0..ROUNDS)
            .map(|round| {
                norm_reference(
                    &diagnostic_inputs(WEIGHTED_LEN, round),
                    Some(&diagnostic_weights(round)),
                    RMS_EPS,
                )
            })
            .collect::<Result<_>>()?;
        let unweighted_values = [2.0_f32];
        let unweighted_expected = norm_reference(&unweighted_values, None, 0.0)?;
        ensure!(
            unweighted_expected[0].to_bits() == 1.0_f32.to_bits(),
            "known unweighted CPU fixture differs"
        );
        let config = NativeExecutionConfig {
            weight_capacity_bytes: 4,
            tensor_capacity_bytes: ((WEIGHTED_LEN * 3 + UNWEIGHTED_LEN * 2 + 4) * 4) as u32,
            auxiliary_capacity_bytes: 4096,
        };
        let expected_allocation_bytes = checked_allocation_bytes(config, DIAGNOSTIC_BUDGET)?;
        let budget = MemoryBudget::new(DIAGNOSTIC_BUDGET);
        let device = NativeExecutionDevice::open(path, &budget, config)?;
        let input = device.allocate_f32(WEIGHTED_LEN)?;
        let weight = device.allocate_f32(WEIGHTED_LEN)?;
        let output = device.upload_f32(&[GUARD; WEIGHTED_LEN + 2])?;
        let unweighted_input = device.allocate_f32(UNWEIGHTED_LEN)?;
        let unweighted_output = device.upload_f32(&[GUARD; UNWEIGHTED_LEN + 2])?;
        let mut weighted_commands = CommandList::new();
        weighted_commands.rmsnorm(
            &output.view(1, WEIGHTED_LEN)?,
            &input.full_view(),
            Some(&weight.full_view()),
            RMS_EPS,
        )?;
        let weighted_batch = device.record(&weighted_commands)?;
        let mut unweighted_commands = CommandList::new();
        unweighted_commands.rmsnorm(
            &unweighted_output.view(1, UNWEIGHTED_LEN)?,
            &unweighted_input.full_view(),
            None,
            0.0,
        )?;
        let unweighted_batch = device.record(&unweighted_commands)?;
        drop((weighted_commands, unweighted_commands));
        let initial = verify_initial(&device, &budget, DIAGNOSTIC_BUDGET)?;
        ensure!(
            initial.allocation_bytes == expected_allocation_bytes
                && initial.tensor_live_bytes == u64::from(config.tensor_capacity_bytes)
                && initial.auxiliary_live_bytes == (WEIGHTED_LEN * 4 + 8) as u64
                && initial.uploaded_weight_bytes == 0
                && initial.weight_live_bytes == 0,
            "RMSNorm diagnostic preparation accounting differs"
        );
        ensure!(
            weighted_batch.len() == 1 && unweighted_batch.len() == 1,
            "expected one RMSNorm command per diagnostic batch"
        );
        let mut maximum_error = 0.0_f64;
        // The first native use of the new ISA is one active lane with exact x=2,
        // eps=0 -> y=1; only a successful readback permits multi-workgroup reuse.
        device.write_f32(&unweighted_input.full_view(), &unweighted_values)?;
        let before = device.stats()?;
        println!("RUN unweighted RMSNorm elements=1 workgroups=1 physical_dispatches=1");
        let submission = device.submit(&unweighted_batch)?;
        device.wait(&submission)?;
        verify_submission(before, device.stats()?, 1)?;
        let mut actual = vec![f32::NAN; UNWEIGHTED_LEN + 2];
        device.read_f32(&unweighted_output.full_view(), &mut actual)?;
        maximum_error = maximum_error.max(verify_diagnostic_output(&actual, &unweighted_expected)?);
        ensure!(
            actual[1].to_bits() == 1.0_f32.to_bits(),
            "known unweighted native fixture must equal 1.0 exactly"
        );
        drop(submission);
        println!("PASS rmsnorm unweighted width=1 checked_values=1 guards=2");
        for (round, expected) in expected.iter().enumerate() {
            device.write_f32(&input.full_view(), &diagnostic_inputs(WEIGHTED_LEN, round))?;
            device.write_f32(&weight.full_view(), &diagnostic_weights(round))?;
            let before = device.stats()?;
            println!(
                "RUN weighted RMSNorm round={round} elements=65 workgroups=2 physical_dispatches=2"
            );
            let submission = device.submit(&weighted_batch)?;
            device.wait(&submission)?;
            verify_submission(before, device.stats()?, 2)?;
            let mut actual = vec![f32::NAN; WEIGHTED_LEN + 2];
            device.read_f32(&output.full_view(), &mut actual)?;
            maximum_error = maximum_error.max(verify_diagnostic_output(&actual, expected)?);
            println!("PASS rmsnorm weighted round={round} checked_values=65 guards=2");
        }
        let final_stats = device.stats()?;
        let expected_writes = ((WEIGHTED_LEN * 2 * ROUNDS + UNWEIGHTED_LEN) * 4) as u64;
        let expected_reads = (((WEIGHTED_LEN + 2) * ROUNDS + UNWEIGHTED_LEN + 2) * 4) as u64;
        ensure!(
            final_stats.dispatch_count == 7
                && final_stats.dispatch_attempts == 7
                && final_stats.logical_submissions == 4
                && final_stats.host_read_bytes == expected_reads
                && final_stats.host_write_bytes == initial.host_write_bytes + expected_writes
                && budget.current_bytes() == initial.allocation_bytes,
            "RMSNorm diagnostic reuse accounting differs"
        );
        ensure!(
            final_stats.weight_peak_bytes == initial.weight_peak_bytes
                && final_stats.tensor_peak_bytes == initial.tensor_peak_bytes
                && final_stats.auxiliary_peak_bytes == initial.auxiliary_peak_bytes,
            "RMSNorm reuse grew resources"
        );
        drop((
            weighted_batch,
            unweighted_batch,
            input,
            weight,
            output,
            unweighted_input,
            unweighted_output,
        ));
        verify_retired(&device)?;
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "RMSNorm diagnostic retained HAG allocations"
        );
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "execution": "native-hag-rmsnorm-diagnostic", "weighted_rounds": ROUNDS,
                "weighted_width": WEIGHTED_LEN, "unweighted_width": UNWEIGHTED_LEN,
                "physical_dispatches": 7, "logical_submissions": 4,
                "checked_values": WEIGHTED_LEN * ROUNDS + UNWEIGHTED_LEN,
                "guard_checks": (ROUNDS + 1) * 2,
                "absolute_tolerance": RMS_ABSOLUTE_TOLERANCE, "relative_tolerance": RMS_RELATIVE_TOLERANCE,
                "max_absolute_error": maximum_error, "verification_readback_bytes": final_stats.host_read_bytes,
                "expected_hag_allocation_bytes": expected_allocation_bytes, "actual_hag_allocation_bytes": initial.allocation_bytes,
                "hag_budget_bytes": DIAGNOSTIC_BUDGET, "required_hag_trace_submission": "0", "hag_trace_requirement_verified": true,
                "submission_tensor_transfer_bytes": 0, "budget_after": 0,
            }))?
        );
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

        fn metadata(dim: usize, width: usize) -> ([Matrix; 3], Norm, DenseFfnBlockMemory) {
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
                Norm {
                    kind: GgmlType(0),
                    len: dim,
                    bytes: dim as u64 * 4,
                },
                DenseFfnBlockMemory {
                    packed_weight_bytes: bytes * 3,
                    resident_tensor_bytes: (dim + width) as u64 * 8,
                    largest_staging_bytes: bytes.max(dim as u64 * 8),
                },
            )
        }

        #[test]
        fn native_execution_requires_explicit_mode_node_and_exact_trace_setting() {
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
                Some(OsStr::new("false")),
                Some(OsStr::new("00")),
                Some(OsStr::new("")),
            ] {
                assert!(check_trace(value).is_err());
            }
            check_trace(Some(OsStr::new("0"))).unwrap();
        }

        #[test]
        fn block_cost_includes_norm_hidden_scratch_and_physical_overhead() {
            let (matrices, norm, memory) = metadata(1024, 3072);
            let capacity = inspect(1024, 3072, matrices, norm, RMS_EPS, memory).unwrap();
            assert_eq!(capacity.weights, 10_027_008);
            assert_eq!(capacity.tensors, 36_864);
            assert_eq!(memory.resident_tensor_bytes, 32_768);
            assert_eq!(memory.largest_staging_bytes, 3_342_336);
            assert_eq!(memory.packed_weight_bytes + norm.bytes, 10_031_104);
            assert_eq!(block_auxiliary_bytes(norm.bytes, RMS_EPS).unwrap(), 4_108);
            let config = NativeExecutionConfig {
                weight_capacity_bytes: capacity.weights,
                tensor_capacity_bytes: capacity.tensors,
                auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
            };
            let bytes = checked_allocation_bytes(config, MODEL_BUDGET).unwrap();
            assert_eq!(bytes, 10_133_504);
            assert!(checked_allocation_bytes(config, bytes - 1).is_err());
        }

        #[test]
        fn block_workspace_counts_distinct_scalar_bits_and_checked_extent() {
            assert_eq!(block_auxiliary_bytes(4096, 0.0).unwrap(), 4_104);
            assert_eq!(block_auxiliary_bytes(4096, 1.0).unwrap(), 4_104);
            assert_eq!(block_auxiliary_bytes(4096, -0.0).unwrap(), 4_108);
            assert!(block_auxiliary_bytes(u64::MAX, RMS_EPS).is_err());
        }

        #[test]
        fn block_inspector_refuses_bad_norm_matrix_shape_staging_and_epsilon() {
            let (matrices, norm, memory) = metadata(1024, 3072);
            for invalid in [
                Norm {
                    kind: GgmlType(30),
                    ..norm
                },
                Norm { len: 1023, ..norm },
                Norm {
                    bytes: 4095,
                    ..norm
                },
            ] {
                assert!(inspect(1024, 3072, matrices, invalid, RMS_EPS, memory).is_err());
            }
            for index in 0..3 {
                let mut wrong = matrices;
                wrong[index].kind = GgmlType(30);
                assert!(inspect(1024, 3072, wrong, norm, RMS_EPS, memory).is_err());
                wrong = matrices;
                wrong[index].cols += 32;
                assert!(inspect(1024, 3072, wrong, norm, RMS_EPS, memory).is_err());
                wrong = matrices;
                wrong[index].bytes -= 1;
                assert!(inspect(1024, 3072, wrong, norm, RMS_EPS, memory).is_err());
            }
            for eps in [f32::NAN, f32::INFINITY, -1.0] {
                assert!(inspect(1024, 3072, matrices, norm, eps, memory).is_err());
            }
            for (dim, width) in [(0, 3072), (1023, 3072), (1024, 4097), (usize::MAX, 3072)] {
                assert!(inspect(dim, width, matrices, norm, RMS_EPS, memory).is_err());
            }
            assert!(inspect(
                1024,
                3072,
                matrices,
                norm,
                RMS_EPS,
                DenseFfnBlockMemory {
                    resident_tensor_bytes: 0,
                    ..memory
                }
            )
            .is_err());
            let (large, norm, memory) = metadata(2048, 2048);
            assert!(
                memory.packed_weight_bytes <= MAX_PACKED
                    && memory.largest_staging_bytes > MAX_STAGING
            );
            assert!(inspect(2048, 2048, large, norm, RMS_EPS, memory).is_err());
            assert!(packed_bytes(usize::MAX, 4096).is_err());
        }

        #[test]
        fn norm_reference_matches_cpu_for_changed_weighted_and_unweighted_fixtures() {
            let cpu = Cpu::new();
            assert_eq!(
                norm_reference(&[2.0], None, 0.0).unwrap()[0].to_bits(),
                1.0_f32.to_bits()
            );
            for round in 0..ROUNDS {
                let values = diagnostic_inputs(WEIGHTED_LEN, round);
                let weights = diagnostic_weights(round);
                let expected = norm_reference(&values, Some(&weights), RMS_EPS).unwrap();
                let mut actual = vec![f32::NAN; WEIGHTED_LEN];
                cpu.rmsnorm(&mut actual, &values, Some(&weights), RMS_EPS)
                    .unwrap();
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            for len in [1, 128] {
                let values = diagnostic_inputs(len, ROUNDS);
                let expected = norm_reference(&values, None, RMS_EPS).unwrap();
                let mut actual = vec![f32::NAN; len];
                cpu.rmsnorm(&mut actual, &values, None, RMS_EPS).unwrap();
                assert!(actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            assert_eq!(
                norm_reference(&[3.0, 4.0], None, 0.0).unwrap()[1],
                4.0 * (1.0f64 / 12.5f64.sqrt()) as f32
            );
            assert!(norm_reference(&[], None, RMS_EPS).is_err());
            assert!(norm_reference(&[f32::NAN], None, RMS_EPS).is_err());
            assert!(norm_reference(&[1.0], Some(&[]), RMS_EPS).is_err());
            assert!(norm_reference(&[0.0], None, 0.0).is_err());
        }

        #[test]
        fn diagnostic_replay_changes_inputs_weights_and_reference_outputs() {
            for round in 1..ROUNDS {
                assert_ne!(
                    diagnostic_inputs(WEIGHTED_LEN, 0),
                    diagnostic_inputs(WEIGHTED_LEN, round)
                );
                assert_ne!(diagnostic_weights(0), diagnostic_weights(round));
                assert_ne!(
                    norm_reference(
                        &diagnostic_inputs(WEIGHTED_LEN, 0),
                        Some(&diagnostic_weights(0)),
                        RMS_EPS
                    )
                    .unwrap(),
                    norm_reference(
                        &diagnostic_inputs(WEIGHTED_LEN, round),
                        Some(&diagnostic_weights(round)),
                        RMS_EPS
                    )
                    .unwrap()
                );
                assert_ne!(inputs(1024, 0), inputs(1024, round));
            }
        }

        #[test]
        fn tolerance_and_canaries_reject_nonfinite_sign_zero_and_extent_errors() {
            compare(
                &[1.0001, -1.0001, 0.0, -0.0],
                &[1.0, -1.0, 0.0, -0.0],
                ABSOLUTE_TOLERANCE,
                RELATIVE_TOLERANCE,
            )
            .unwrap();
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
                assert!(compare(
                    &[actual],
                    &[expected],
                    ABSOLUTE_TOLERANCE,
                    RELATIVE_TOLERANCE
                )
                .is_err());
            }
            verify_diagnostic_output(&[GUARD, 1.0, GUARD], &[1.0]).unwrap();
            assert!(verify_diagnostic_output(&[0.0, 1.0, GUARD], &[1.0]).is_err());
            assert!(verify_diagnostic_output(&[GUARD, 1.0], &[1.0]).is_err());
            assert!(verify_diagnostic_output(&[GUARD, 1.00001, GUARD], &[1.0]).is_err());
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native HAG FFN block qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
