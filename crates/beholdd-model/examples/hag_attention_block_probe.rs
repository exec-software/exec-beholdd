//! Explicit native qualification of one complete Qwen3 attention/residual block.
//! Metadata inspection is the default. Execution requires HAG_TRACE_SUBMISSION=0
//! and external process/thermal supervision. Only layer weights and an eight-position
//! resident KV cache are allocated. Two three-position requests compare final hidden
//! with an independent immediate CPU path; no intermediate GPU tensors are read.
//! Preregistered finite-fixture bound: 3e-4 + 3e-5 * |reference|, plus sign/zero checks.
//! This gate does not claim whole-model quality, throughput, or serving readiness.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{bail, ensure, Context, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::ExecutionDevice;
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::{Device, IncrementalAttentionConfig, Weight};
    use beholdd_format::{CheckpointId, GgmlType, Gguf, WeightRange};
    use beholdd_memory::{
        check_checkpoint_storage, ResidentSource, ResidentWeight, StorageOverride, WeightAccess,
        WeightSource,
    };
    use beholdd_model::qwen3moe::{DenseAttentionLimits, DenseAttentionMemory};
    use beholdd_model::QwenModel;
    use serde_json::json;
    use std::ffi::{OsStr, OsString};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const USAGE: &str = "hag_attention_block_probe [--model PATH] [--layer N]\n\
        hag_attention_block_probe --execute --model PATH [--layer N] /dev/dri/renderD128\n\
        Default: metadata inspection only. Native execution requires HAG_TRACE_SUBMISSION=0,\n\
        external process/thermal supervision, and previous isolated RoPE/attention qualification.\n\
        Fixed cache capacity8; two requests of three positions; 32 MiB HAG cap.";
    const CACHE_CAPACITY: usize = 8;
    const REQUESTS: usize = 2;
    const STEPS: usize = 3;
    const MAX_PACKED: u64 = 16 * 1024 * 1024;
    const MAX_TENSOR: u64 = 128 * 1024;
    const MAX_STAGING: u64 = 4 * 1024 * 1024;
    const AUXILIARY_CAPACITY: u32 = 256 * 1024;
    const MODEL_BUDGET: u64 = 32 * 1024 * 1024;
    const ABSOLUTE_TOLERANCE: f64 = 3e-4;
    const RELATIVE_TOLERANCE: f64 = 3e-5;

    #[derive(Debug)]
    struct Options {
        execute: bool,
        model: Option<PathBuf>,
        layer: usize,
        device: Option<PathBuf>,
        help: bool,
    }
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Options> {
        let mut args = arguments.into_iter();
        let mut options = Options {
            execute: false,
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
                    options.model = Some(PathBuf::from(args.next().context("--model needs PATH")?))
                }
                Some("--layer") if !layer_seen => {
                    options.layer = args
                        .next()
                        .context("--layer needs an index")?
                        .to_str()
                        .context("layer must be Unicode")?
                        .parse()
                        .context("layer must be nonnegative")?;
                    layer_seen = true;
                }
                Some("--execute") if !options.execute => options.execute = true,
                Some(value) if !value.starts_with('-') && options.device.is_none() => {
                    options.device = Some(PathBuf::from(arg))
                }
                _ => bail!("unknown or repeated argument {arg:?}\n{USAGE}"),
            }
        }
        if !options.help {
            ensure!(
                options.execute == options.device.is_some(),
                "an explicit execution flag and render node are required together"
            );
            ensure!(
                !options.execute || options.model.is_some(),
                "execution requires --model PATH"
            );
            ensure!(
                !layer_seen || options.model.is_some(),
                "--layer requires --model"
            );
        }
        Ok(options)
    }
    fn check_trace(value: Option<&OsStr>) -> Result<()> {
        ensure!(value == Some(OsStr::new("0")), "native qualification requires HAG_TRACE_SUBMISSION exactly 0; payload tracing is refused");
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    struct Shape {
        dim: usize,
        heads: usize,
        kv_heads: usize,
        head: usize,
        eps: f32,
        theta: f64,
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
    #[derive(Clone, Copy, Debug)]
    struct Capacity {
        weights: u32,
        tensors: u32,
        auxiliary_live: u64,
        source_bytes: u64,
        memory: DenseAttentionMemory,
    }

    fn auxiliary_for_position(shape: Shape, visible: usize) -> Result<u64> {
        // Norm/residual scratch is reused across this batch. Scores and
        // probabilities coexist; the extra word permits eight-byte score alignment.
        let norm_scratch = shape
            .dim
            .max(shape.head)
            .checked_mul(4)
            .context("attention norm scratch overflows")?;
        let attention_scratch = visible
            .checked_mul(12)
            .and_then(|bytes| bytes.checked_add(4))
            .context("attention score/probability scratch overflows")?;
        let mut scalars = [
            shape.eps.to_bits(),
            (shape.head as f32).sqrt().recip().to_bits(),
            1.0_f32.to_bits(),
        ];
        scalars.sort_unstable();
        let words = 1 + scalars.windows(2).filter(|pair| pair[0] != pair[1]).count();
        u64::try_from(norm_scratch.max(attention_scratch))?
            .checked_add(words as u64 * 4)
            .context("attention batch auxiliary bytes overflow")
    }

    fn descriptors(checkpoint: &Gguf, layer: usize) -> Result<([Matrix; 4], [Norm; 3])> {
        let matrix = |name: &str| -> Result<Matrix> {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.{name}.weight"))
                .context("missing attention matrix")?;
            let [cols, rows] = tensor.dims() else {
                bail!("attention matrix must have rank two");
            };
            Ok(Matrix {
                kind: tensor.kind(),
                rows: usize::try_from(*rows)?,
                cols: usize::try_from(*cols)?,
                bytes: tensor.nbytes(),
            })
        };
        let norm = |name: &str| -> Result<Norm> {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.{name}.weight"))
                .context("missing attention norm")?;
            let [len] = tensor.dims() else {
                bail!("attention norm must have rank one");
            };
            Ok(Norm {
                kind: tensor.kind(),
                len: usize::try_from(*len)?,
                bytes: tensor.nbytes(),
            })
        };
        Ok((
            [
                matrix("attn_q")?,
                matrix("attn_k")?,
                matrix("attn_v")?,
                matrix("attn_output")?,
            ],
            [
                norm("attn_norm")?,
                norm("attn_q_norm")?,
                norm("attn_k_norm")?,
            ],
        ))
    }

    fn inspect(
        shape: Shape,
        matrices: [Matrix; 4],
        norms: [Norm; 3],
        memory: DenseAttentionMemory,
    ) -> Result<Capacity> {
        let Shape {
            dim,
            heads,
            kv_heads,
            head,
            eps,
            theta,
        } = shape;
        ensure!(
            (1..=4096).contains(&dim)
                && (1..=32).contains(&heads)
                && (1..=32).contains(&kv_heads)
                && heads.is_multiple_of(kv_heads),
            "unsupported dimensions/GQA"
        );
        ensure!(
            (2..=256).contains(&head) && head.is_multiple_of(2),
            "unsupported head width"
        );
        ensure!(
            eps.is_finite() && eps >= 0.0 && theta.is_finite() && theta > 0.0,
            "invalid epsilon/RoPE base"
        );
        let q = heads * head;
        let kv = kv_heads * head;
        ensure!(q <= 4096 && kv <= 4096, "projection width exceeds 4096");
        let mut packed = 0u64;
        let mut aligned = 0u64;
        let mut staging = 0u64;
        for (matrix, (rows, cols)) in
            matrices
                .into_iter()
                .zip([(q, dim), (kv, dim), (kv, dim), (dim, q)])
        {
            ensure!(
                matrix.kind.0 == 8
                    && matrix.rows == rows
                    && matrix.cols == cols
                    && cols.is_multiple_of(32),
                "attention requires exact Q8_0 shapes"
            );
            let bytes = (rows * (cols / 32) * 34) as u64;
            ensure!(matrix.bytes == bytes, "attention packed span differs");
            packed += bytes;
            aligned += bytes.next_multiple_of(4);
            staging = staging.max(bytes);
        }
        for (norm, len) in norms.into_iter().zip([dim, head, head]) {
            ensure!(
                norm.kind.0 == 0 && norm.len == len && norm.bytes == (len * 4) as u64,
                "attention requires exact F32 norm vectors"
            );
            staging = staging.max((len * 8) as u64);
        }
        staging = staging.max((CACHE_CAPACITY * head * 4) as u64);
        let resident =
            ((2 * dim + 2 * head + 3 * q + kv + 2 * CACHE_CAPACITY * kv + CACHE_CAPACITY * head)
                * 4) as u64;
        let tensors = resident + (dim * 4) as u64;
        ensure!(
            memory
                == DenseAttentionMemory {
                    packed_weight_bytes: packed,
                    resident_tensor_bytes: resident,
                    largest_staging_bytes: staging
                },
            "attention plan and independent costs differ"
        );
        ensure!(
            packed <= MAX_PACKED && tensors <= MAX_TENSOR && staging <= MAX_STAGING,
            "attention payload exceeds qualification bounds"
        );
        let auxiliary_live = (1..=CACHE_CAPACITY).try_fold(0u64, |total, visible| {
            total
                .checked_add(auxiliary_for_position(shape, visible)?)
                .context("attention auxiliary total overflows")
        })?;
        ensure!(
            auxiliary_live <= u64::from(AUXILIARY_CAPACITY),
            "attention batches exceed auxiliary arena"
        );
        Ok(Capacity {
            weights: u32::try_from(aligned)?,
            tensors: u32::try_from(tensors)?,
            auxiliary_live,
            source_bytes: packed + ((dim + 2 * head) * 4) as u64,
            memory,
        })
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

    // Independent immediate oracle: one-worker Q8 projections, f64 RMS reduction,
    // explicit split-half rotation, and existing CPU incremental attention. No recorded
    // command, native shader, native cache or intermediate readback contributes to it.
    fn reference_step(
        checkpoint: &Gguf,
        source: &CountingSource<'_>,
        layer: usize,
        shape: Shape,
        hidden: &[f32],
        k: &mut Vec<f32>,
        v: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let cpu = Cpu::new();
        let project = |name: &str, input: &[f32], rows: usize| -> Result<Vec<f32>> {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.{name}.weight"))
                .context("missing CPU matrix")?;
            let storage = source.load_range(
                &tensor.locator().range(0..tensor.nbytes())?,
                WeightAccess::Cyclic,
            )?;
            let mut output = vec![0.0; rows];
            cpu.matvec_with_threads(
                &mut output,
                input,
                &Weight {
                    bytes: storage.bytes(),
                    kind: tensor.kind(),
                    rows,
                    cols: input.len(),
                },
                1,
            )?;
            Ok(output)
        };
        let norm = |name: &str| -> Result<Vec<f32>> {
            let tensor = checkpoint
                .get(&format!("blk.{layer}.{name}.weight"))
                .context("missing CPU norm")?;
            let storage = source.load_range(
                &tensor.locator().range(0..tensor.nbytes())?,
                WeightAccess::Cyclic,
            )?;
            Ok(storage
                .bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect())
        };
        let normalized = norm_reference(hidden, Some(&norm("attn_norm")?), shape.eps)?;
        let q_width = shape.heads * shape.head;
        let kv_width = shape.kv_heads * shape.head;
        let raw_q = project("attn_q", &normalized, q_width)?;
        let raw_k = project("attn_k", &normalized, kv_width)?;
        let raw_v = project("attn_v", &normalized, kv_width)?;
        ensure!(
            k.len().is_multiple_of(kv_width) && k.len() == v.len(),
            "invalid CPU KV cache"
        );
        let position = k.len() / kv_width;
        let norm_rotate = |raw: &[f32], weights: &[f32]| -> Result<Vec<f32>> {
            let mut output = Vec::with_capacity(raw.len());
            for head in raw.chunks_exact(shape.head) {
                let mut head = norm_reference(head, Some(weights), shape.eps)?;
                let half = shape.head / 2;
                for index in 0..half {
                    let frequency = 1.0 / shape.theta.powf((2 * index) as f64 / shape.head as f64);
                    let angle = position as f64 * frequency;
                    let cos = angle.cos() as f32;
                    let sin = angle.sin() as f32;
                    let a = head[index];
                    let b = head[index + half];
                    head[index] = a * cos - b * sin;
                    head[index + half] = a * sin + b * cos;
                }
                output.extend(head);
            }
            Ok(output)
        };
        let q = norm_rotate(&raw_q, &norm("attn_q_norm")?)?;
        k.extend(norm_rotate(&raw_k, &norm("attn_k_norm")?)?);
        v.extend(raw_v);
        let mut attended = vec![0.0; q_width];
        cpu.incremental_causal_attention(
            &mut attended,
            &q,
            k,
            v,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: position,
                n_kv_positions: position + 1,
                n_heads: shape.heads,
                n_kv_heads: shape.kv_heads,
                head_dim: shape.head,
                scale: (shape.head as f32).sqrt().recip(),
            },
        )?;
        let contribution = project("attn_output", &attended, shape.dim)?;
        let output = hidden
            .iter()
            .zip(contribution)
            .map(|(residual, value)| *residual + value)
            .collect::<Vec<_>>();
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "CPU attention reference is nonfinite"
        );
        Ok(output)
    }

    fn run_model(options: &Options) -> Result<()> {
        let path = options.model.as_ref().context("missing model")?;
        check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
        let checkpoint = Gguf::open(path)?;
        check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
        let model = QwenModel::load(&checkpoint)?;
        let plan = model.dense_attention_plan(options.layer)?;
        let shape = Shape {
            dim: plan.dim(),
            heads: plan.n_heads(),
            kv_heads: plan.n_kv_heads(),
            head: plan.head_dim(),
            eps: plan.eps(),
            theta: plan.rope_base(),
        };
        let (matrices, norms) = descriptors(&checkpoint, options.layer)?;
        let capacity = inspect(shape, matrices, norms, plan.memory(CACHE_CAPACITY)?)?;
        let config = NativeExecutionConfig {
            weight_capacity_bytes: capacity.weights,
            tensor_capacity_bytes: capacity.tensors,
            auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
        };
        let expected_allocation_bytes = checked_allocation_bytes(config, MODEL_BUDGET)?;
        let physical_per_step =
            (2 + 3 + 3 * (shape.heads + shape.kv_heads) + 3 * shape.heads + 2) as u64;
        let semantic_per_step = 6 + 2 * (shape.heads + shape.kv_heads);
        ensure!(
            plan.semantic_commands_per_step() == semantic_per_step,
            "semantic accounting differs"
        );
        let mut report = json!({
            "execution": "not_requested", "architecture": "qwen3", "layer": options.layer,
            "input_width": shape.dim, "query_heads": shape.heads, "kv_heads": shape.kv_heads, "head_dim": shape.head,
            "cache_capacity": CACHE_CAPACITY, "requests": REQUESTS, "steps_per_request": STEPS,
            "semantic_commands_per_step": semantic_per_step, "expected_physical_dispatches_per_step": physical_per_step,
            "packed_weight_bytes": capacity.memory.packed_weight_bytes, "source_weight_bytes": capacity.source_bytes,
            "resident_tensor_bytes": capacity.memory.resident_tensor_bytes, "largest_staging_bytes": capacity.memory.largest_staging_bytes,
            "weight_arena_capacity_bytes": capacity.weights, "tensor_arena_capacity_bytes": capacity.tensors,
            "auxiliary_arena_capacity_bytes": AUXILIARY_CAPACITY, "expected_auxiliary_live_bytes": capacity.auxiliary_live,
            "hag_budget_bytes": MODEL_BUDGET, "expected_hag_allocation_bytes": expected_allocation_bytes,
            "required_hag_trace_submission": "0", "hag_trace_requirement_verified": options.execute,
            "external_process_and_thermal_limits_required": true,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
            "oracle": "independent f64 RMS, split-half RoPE, immediate one-worker Q8/CPU attention/residual; finite/sign/zero checks",
            "rmsnorm_epsilon": shape.eps, "rope_base": shape.theta, "weight_format": "Q8_0", "norm_format": "F32"
        });
        if !options.execute {
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        println!("RUN preregistered bound: |error| <= {ABSOLUTE_TOLERANCE} + {RELATIVE_TOLERANCE} * |reference|; finite/sign/zero checks");
        let oracle_source = CountingSource {
            inner: ResidentSource::new(&checkpoint),
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        };
        let mut expected = Vec::new();
        for request in 0..REQUESTS {
            let mut k = Vec::new();
            let mut v = Vec::new();
            for step in 0..STEPS {
                expected.push(reference_step(
                    &checkpoint,
                    &oracle_source,
                    options.layer,
                    shape,
                    &inputs(shape.dim, request * STEPS + step),
                    &mut k,
                    &mut v,
                )?);
            }
        }
        let oracle_reads = oracle_source.reads.load(Ordering::Relaxed);
        ensure!(
            oracle_reads == (7 * REQUESTS * STEPS) as u64
                && oracle_source.bytes.load(Ordering::Relaxed)
                    == capacity.source_bytes * (REQUESTS * STEPS) as u64,
            "CPU reference source accounting differs"
        );
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
            CACHE_CAPACITY,
            DenseAttentionLimits {
                packed_weight_bytes: capacity.memory.packed_weight_bytes,
                resident_tensor_bytes: capacity.memory.resident_tensor_bytes,
                largest_staging_bytes: capacity.memory.largest_staging_bytes,
            },
        )?;
        ensure!(
            source.reads.load(Ordering::Relaxed) == 7
                && source.bytes.load(Ordering::Relaxed) == capacity.source_bytes,
            "native preparation must read exactly four matrices and three norms"
        );
        let hidden = device.allocate_f32(shape.dim)?;
        let mut cache = prepared.record(&device, &hidden.full_view())?;
        ensure!(
            cache.capacity() == CACHE_CAPACITY
                && cache.visible_positions() == 0
                && !cache.is_poisoned(),
            "initial cache state differs"
        );
        let initial = verify_initial(&device, &budget, MODEL_BUDGET)?;
        ensure!(
            initial.allocation_bytes == expected_allocation_bytes
                && initial.weight_live_bytes == u64::from(capacity.weights)
                && initial.tensor_live_bytes == u64::from(capacity.tensors)
                && initial.auxiliary_live_bytes == capacity.auxiliary_live
                && initial.uploaded_weight_bytes == capacity.memory.packed_weight_bytes,
            "attention preparation resource accounting differs"
        );
        let mut maximum_error = 0.0f64;
        let mut checksum = 0u64;
        let mut reset_visible = usize::MAX;
        for request in 0..REQUESTS {
            if request > 0 {
                let before = device.stats()?;
                cache.reset()?;
                reset_visible = cache.visible_positions();
                ensure!(
                    reset_visible == 0 && !cache.is_poisoned(),
                    "cache reset failed"
                );
                let after = device.stats()?;
                ensure!(
                    after.dispatch_count == before.dispatch_count
                        && after.host_read_bytes == before.host_read_bytes
                        && after.host_write_bytes == before.host_write_bytes
                        && after.allocation_bytes == before.allocation_bytes,
                    "reset performed device work"
                );
            }
            for step in 0..STEPS {
                let round = request * STEPS + step;
                device.write_f32(&hidden.full_view(), &inputs(shape.dim, round))?;
                let before = device.stats()?;
                println!("RUN attention block request={request} step={step} semantic_commands={semantic_per_step} physical_dispatches={physical_per_step}");
                cache.step(&device)?;
                verify_submission(before, device.stats()?, physical_per_step)?;
                ensure!(
                    cache.visible_positions() == step + 1 && !cache.is_poisoned(),
                    "completed attention position was not committed"
                );
                ensure!(
                    budget.current_bytes() == initial.allocation_bytes,
                    "GPU allocation changed during reuse"
                );
                let mut actual = vec![f32::NAN; shape.dim];
                device.read_f32(&hidden.full_view(), &mut actual)?;
                maximum_error = maximum_error.max(
                    compare(
                        &actual,
                        &expected[round],
                        ABSOLUTE_TOLERANCE,
                        RELATIVE_TOLERANCE,
                    )
                    .with_context(|| format!("attention request {request} step {step}"))?,
                );
                checksum = actual.iter().fold(checksum, |sum, value| {
                    sum.wrapping_add(u64::from(value.to_bits()))
                });
                println!("PASS attention block request={request} step={step} checked_values={} submission_tensor_transfer_bytes=0", actual.len());
            }
        }
        let final_stats = device.stats()?;
        let submissions = (REQUESTS * STEPS) as u64;
        let transfers = shape.dim as u64 * 4 * submissions;
        ensure!(
            final_stats.dispatch_count == physical_per_step * submissions
                && final_stats.dispatch_attempts == final_stats.dispatch_count
                && final_stats.logical_submissions == submissions,
            "final dispatch/submission accounting differs"
        );
        ensure!(
            final_stats.uploaded_weight_bytes == capacity.memory.packed_weight_bytes
                && source.reads.load(Ordering::Relaxed) == 7
                && source.bytes.load(Ordering::Relaxed) == capacity.source_bytes,
            "attention reuse reread/reuploaded source weights"
        );
        ensure!(
            final_stats.host_read_bytes == transfers
                && final_stats.host_write_bytes == initial.host_write_bytes + transfers,
            "attention boundary traffic differs"
        );
        ensure!(
            final_stats.weight_peak_bytes == initial.weight_peak_bytes
                && final_stats.tensor_peak_bytes == initial.tensor_peak_bytes
                && final_stats.auxiliary_peak_bytes == initial.auxiliary_peak_bytes,
            "cache reuse grew resources"
        );
        drop((cache, hidden));
        verify_retired(&device)?;
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "HAG retained attention resources after cleanup"
        );
        report["execution"] = json!("native-hag-attention-block");
        report["physical_dispatches"] = json!(final_stats.dispatch_count);
        report["logical_submissions"] = json!(final_stats.logical_submissions);
        report["checked_values"] = json!(shape.dim * REQUESTS * STEPS);
        report["native_preparation_source_reads"] = json!(7);
        report["cpu_oracle_source_reads"] = json!(oracle_reads);
        report["actual_hag_allocation_bytes"] = json!(initial.allocation_bytes);
        report["auxiliary_live_bytes"] = json!(initial.auxiliary_live_bytes);
        report["budget_after"] = json!(0);
        report["submission_tensor_transfer_bytes"] = json!(0);
        report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
        report["max_absolute_error"] = json!(maximum_error);
        report["cache_visible_positions_after_reset"] = json!(reset_visible);
        report["output_checksum"] = json!(checksum);
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    pub fn run() -> Result<()> {
        let options = parse(std::env::args_os().skip(1))?;
        if options.help {
            println!("{USAGE}");
            return Ok(());
        }
        if options.execute {
            check_trace(std::env::var_os("HAG_TRACE_SUBMISSION").as_deref())?;
        }
        if options.model.is_none() {
            println!("{USAGE}");
            return Ok(());
        }
        run_model(&options)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn args(values: &[&str]) -> Vec<OsString> {
            values.iter().map(OsString::from).collect()
        }
        fn fixture() -> (Shape, [Matrix; 4], [Norm; 3], DenseAttentionMemory) {
            let shape = Shape {
                dim: 1024,
                heads: 16,
                kv_heads: 8,
                head: 128,
                eps: 1e-6,
                theta: 1_000_000.0,
            };
            let matrices =
                [(2048, 1024), (1024, 1024), (1024, 1024), (1024, 2048)].map(|(rows, cols)| {
                    Matrix {
                        kind: GgmlType(8),
                        rows,
                        cols,
                        bytes: (rows * cols / 32 * 34) as u64,
                    }
                });
            let norms = [1024, 128, 128].map(|len| Norm {
                kind: GgmlType(0),
                len,
                bytes: (len * 4) as u64,
            });
            (
                shape,
                matrices,
                norms,
                DenseAttentionMemory {
                    packed_weight_bytes: 6684672,
                    resident_tensor_bytes: 107520,
                    largest_staging_bytes: 2228224,
                },
            )
        }
        #[test]
        fn cli_defaults_safe_and_rejects_ambiguous_device_access() {
            assert!(!parse(args(&[])).unwrap().execute);
            assert!(parse(args(&["--help"])).unwrap().help);
            assert!(!parse(args(&["--model", "fixture.gguf"])).unwrap().execute);
            assert!(
                parse(args(&[
                    "--execute",
                    "--model",
                    "fixture.gguf",
                    "/dev/dri/renderD128"
                ]))
                .unwrap()
                .execute
            );
            for values in [
                vec!["/dev/dri/renderD128"],
                vec!["--execute"],
                vec!["--layer", "1"],
                vec!["--execute", "--execute"],
                vec!["--diagnose-first"],
                vec!["--model", "a", "--model", "b"],
            ] {
                assert!(parse(args(&values)).is_err());
            }
            for value in [
                None,
                Some(OsStr::new("")),
                Some(OsStr::new("1")),
                Some(OsStr::new("false")),
                Some(OsStr::new("00")),
            ] {
                assert!(check_trace(value).is_err());
            }
            check_trace(Some(OsStr::new("0"))).unwrap();
        }
        #[test]
        fn exact_model_cost_includes_all_position_batches_and_hidden() {
            let (shape, matrices, norms, memory) = fixture();
            let cap = inspect(shape, matrices, norms, memory).unwrap();
            assert_eq!(cap.weights, 6684672);
            assert_eq!(cap.tensors, 111616);
            assert_eq!(cap.auxiliary_live, 32_864);
            assert_eq!(cap.source_bytes, 6689792);
            let config = NativeExecutionConfig {
                weight_capacity_bytes: cap.weights,
                tensor_capacity_bytes: cap.tensors,
                auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
            };
            let bytes = checked_allocation_bytes(config, MODEL_BUDGET).unwrap();
            assert!(bytes < MODEL_BUDGET);
            assert!(checked_allocation_bytes(config, bytes - 1).is_err());
        }

        #[test]
        fn batch_workspace_covers_head_and_attention_scratch_with_scalar_interning() {
            let (shape, _, _, _) = fixture();
            assert_eq!(auxiliary_for_position(shape, 8).unwrap(), 4_108);
            let shared_eps = Shape {
                eps: (shape.head as f32).sqrt().recip(),
                ..shape
            };
            assert_eq!(auxiliary_for_position(shared_eps, 8).unwrap(), 4_104);
            let narrow = Shape { dim: 32, ..shape };
            assert_eq!(auxiliary_for_position(narrow, 8).unwrap(), 524);
            assert_eq!(auxiliary_for_position(narrow, 64).unwrap(), 784);
            assert!(auxiliary_for_position(shape, usize::MAX).is_err());
        }

        #[test]
        fn preflight_refuses_shape_format_extent_cost_and_nonfinite_metadata() {
            let (shape, matrices, norms, memory) = fixture();
            for eps in [f32::NAN, f32::INFINITY, -1.0] {
                assert!(inspect(Shape { eps, ..shape }, matrices, norms, memory).is_err());
            }
            for theta in [f64::NAN, 0.0, -1.0, f64::INFINITY] {
                assert!(inspect(Shape { theta, ..shape }, matrices, norms, memory).is_err());
            }
            for dim in [0, 4097, usize::MAX] {
                assert!(inspect(Shape { dim, ..shape }, matrices, norms, memory).is_err());
            }
            for index in 0..4 {
                let mut bad = matrices;
                bad[index].kind = GgmlType(0);
                assert!(inspect(shape, bad, norms, memory).is_err());
                let mut bad = matrices;
                bad[index].bytes -= 1;
                assert!(inspect(shape, bad, norms, memory).is_err());
                let mut bad = matrices;
                bad[index].rows += 1;
                assert!(inspect(shape, bad, norms, memory).is_err());
            }
            for index in 0..3 {
                let mut bad = norms;
                bad[index].len += 1;
                assert!(inspect(shape, matrices, bad, memory).is_err());
            }
            assert!(inspect(
                shape,
                matrices,
                norms,
                DenseAttentionMemory {
                    resident_tensor_bytes: 1,
                    ..memory
                }
            )
            .is_err());
        }
        #[test]
        fn inputs_change_at_every_cached_step_and_after_reset() {
            let inputs = (0..REQUESTS * STEPS)
                .map(|round| inputs(1024, round))
                .collect::<Vec<_>>();
            for i in 0..inputs.len() {
                for j in i + 1..inputs.len() {
                    assert_ne!(inputs[i], inputs[j]);
                }
            }
            assert!(inputs.iter().flatten().all(|value| value.is_finite()));
        }
        #[test]
        fn preregistered_tolerance_rejects_bad_values_and_checks_norm_oracle() {
            assert_eq!(norm_reference(&[2.0], None, 0.0).unwrap(), vec![1.0]);
            assert_eq!(
                norm_reference(&[2.0, -2.0], Some(&[0.5, 2.0]), 0.0).unwrap(),
                vec![0.5, -2.0]
            );
            compare(
                &[1.0001, -2.0001],
                &[1.0, -2.0],
                ABSOLUTE_TOLERANCE,
                RELATIVE_TOLERANCE,
            )
            .unwrap();
            for (actual, expected) in [
                (f32::NAN, 1.0),
                (1.0, f32::INFINITY),
                (0.0, -0.0),
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
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::run()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native HAG attention qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
