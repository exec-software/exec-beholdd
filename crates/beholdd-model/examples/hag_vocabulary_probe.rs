//! Explicit native gate for a tiled tied Qwen3 embedding/final-normalized head.
//! Default inspection reads metadata only. Six embedding rows are compared bitwise;
//! three changed synthetic hidden vectors produce complete vocabulary logits.
//! Head acceptance is finite values, 3e-4 + 3e-5*abs(reference), and identical CPU
//! greedy selection. Sign/zero differences are reported, not independently rejected.
//! This policy is preregistered before live execution. All CPU oracle work precedes
//! native access; it stages one packed tile at a time and retains under 2 MiB of
//! reference outputs. This is a vocabulary component gate, not full-model inference.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{bail, ensure, Context, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::ExecutionDevice;
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::{Device, Weight};
    use beholdd_format::{CheckpointId, GgmlType, Gguf, WeightRange};
    use beholdd_memory::{
        check_checkpoint_storage, ResidentSource, ResidentWeight, StorageOverride, WeightAccess,
        WeightSource,
    };
    use beholdd_model::qwen3moe::{VocabularyLimits, VocabularyMemory, VocabularyPlan};
    use beholdd_model::QwenModel;
    use serde_json::json;
    use std::ffi::{OsStr, OsString};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const USAGE: &str = "hag_vocabulary_probe [--model PATH]\n\
        hag_vocabulary_probe --execute --model PATH /dev/dri/renderD128\n\
        Default: metadata inspection only. Requires exact tied Qwen3 Q8_0 1024x151936.\n\
        Execution requires HAG_TRACE_SUBMISSION=0, prior isolated embedding-row qualification,\n\
        external process/thermal supervision, and a 192 MiB HAG budget.";
    const DIM: usize = 1024;
    const VOCAB: usize = 151936;
    const TILE_ROWS: usize = 4096;
    const TILE_COUNT: usize = 38;
    const TOKENS: [usize; 6] = [0, 4095, 4096, 65536, 151551, 151935];
    const HEAD_ROUNDS: usize = 3;
    const MAX_PACKED: u64 = 176 * 1024 * 1024;
    const MAX_TENSOR: u64 = 1024 * 1024;
    const MAX_STAGING: u64 = 8 * 1024 * 1024;
    const AUXILIARY_CAPACITY: u32 = 8192;
    const MODEL_BUDGET: u64 = 192 * 1024 * 1024;
    const ABSOLUTE_TOLERANCE: f64 = 3e-4;
    const RELATIVE_TOLERANCE: f64 = 3e-5;
    const GUARD: f32 = -12345.5;

    #[derive(Debug)]
    struct Options {
        execute: bool,
        model: Option<PathBuf>,
        device: Option<PathBuf>,
        help: bool,
    }
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Options> {
        let mut args = arguments.into_iter();
        let mut options = Options {
            execute: false,
            model: None,
            device: None,
            help: false,
        };
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--help" | "-h") => options.help = true,
                Some("--model") if options.model.is_none() => {
                    options.model = Some(PathBuf::from(args.next().context("--model needs PATH")?))
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
                "explicit execution flag and render node are required together"
            );
            ensure!(
                !options.execute || options.model.is_some(),
                "execution requires --model PATH"
            );
        }
        Ok(options)
    }
    fn check_trace(value: Option<&OsStr>) -> Result<()> {
        ensure!(value == Some(OsStr::new("0")), "native qualification requires HAG_TRACE_SUBMISSION exactly 0; payload tracing is refused");
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    struct Descriptor {
        kind: GgmlType,
        dim: usize,
        vocab: usize,
        bytes: u64,
        norm_kind: GgmlType,
        norm_len: usize,
        norm_bytes: u64,
        tied: bool,
        eps: f32,
    }
    #[derive(Clone, Copy, Debug)]
    struct Capacity {
        weights: u32,
        tensors: u32,
        source_bytes: u64,
        memory: VocabularyMemory,
    }
    fn descriptor(checkpoint: &Gguf, eps: f32) -> Result<Descriptor> {
        let embedding = checkpoint
            .get("token_embd.weight")
            .context("missing embedding")?;
        let [dim, vocab] = embedding.dims() else {
            bail!("embedding must have rank two");
        };
        let norm = checkpoint
            .get("output_norm.weight")
            .context("missing final norm")?;
        let [norm_len] = norm.dims() else {
            bail!("final norm must have rank one");
        };
        Ok(Descriptor {
            kind: embedding.kind(),
            dim: usize::try_from(*dim)?,
            vocab: usize::try_from(*vocab)?,
            bytes: embedding.nbytes(),
            norm_kind: norm.kind(),
            norm_len: usize::try_from(*norm_len)?,
            norm_bytes: norm.nbytes(),
            tied: checkpoint.get("output.weight").is_none(),
            eps,
        })
    }
    fn inspect(
        descriptor: Descriptor,
        tile_rows: usize,
        tiles: &[(usize, usize, u64)],
        memory: VocabularyMemory,
    ) -> Result<Capacity> {
        let d = descriptor;
        ensure!(
            d.dim == DIM && d.vocab == VOCAB && d.kind.0 == 8 && d.tied,
            "qualification requires exact tied Qwen3 Q8_0 dimensions1024x151936"
        );
        ensure!(
            d.eps.is_finite()
                && d.eps >= 0.0
                && d.norm_kind.0 == 0
                && d.norm_len == DIM
                && d.norm_bytes == (DIM * 4) as u64,
            "qualification requires exact F32 final norm and finite nonnegative epsilon"
        );
        let row_bytes = (DIM / 32 * 34) as u64;
        let packed = VOCAB as u64 * row_bytes;
        ensure!(
            d.bytes == packed && tile_rows == TILE_ROWS && tiles.len() == TILE_COUNT,
            "packed vocabulary/tile geometry differs"
        );
        let mut first = 0usize;
        let mut aligned = 0u64;
        let mut largest = 0u64;
        for &(actual_first, rows, bytes) in tiles {
            let expected_rows = TILE_ROWS.min(VOCAB - first);
            ensure!(
                actual_first == first && rows == expected_rows && bytes == rows as u64 * row_bytes,
                "vocabulary tiles must exactly partition the tied matrix"
            );
            first += rows;
            aligned += bytes.next_multiple_of(4);
            largest = largest.max(bytes);
        }
        ensure!(first == VOCAB, "vocabulary tile coverage differs");
        let resident = (2 * DIM * 4) as u64;
        let expected = VocabularyMemory {
            packed_weight_bytes: packed,
            resident_tensor_bytes: resident,
            largest_staging_bytes: largest.max((DIM * 8) as u64),
        };
        ensure!(
            memory == expected,
            "vocabulary plan/independent costs differ"
        );
        let tensors = resident + ((DIM + 2 + VOCAB + 2) * 4) as u64;
        ensure!(
            packed <= MAX_PACKED
                && tensors <= MAX_TENSOR
                && memory.largest_staging_bytes <= MAX_STAGING,
            "vocabulary payload exceeds qualification limits"
        );
        ensure!(
            TOKENS.iter().all(|token| *token < VOCAB),
            "preregistered token is out of bounds"
        );
        Ok(Capacity {
            weights: u32::try_from(aligned)?,
            tensors: u32::try_from(tensors)?,
            source_bytes: packed + d.norm_bytes,
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

    fn check_guards(actual: &[f32], elements: usize) -> Result<&[f32]> {
        ensure!(
            actual.len() == elements + 2,
            "guarded output length differs"
        );
        ensure!(
            actual[0].to_bits() == GUARD.to_bits()
                && actual[elements + 1].to_bits() == GUARD.to_bits(),
            "vocabulary output guard changed"
        );
        Ok(&actual[1..elements + 1])
    }

    fn compare_embedding(actual: &[f32], expected: &[f32]) -> Result<()> {
        ensure!(actual.len() == expected.len(), "embedding length differs");
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            ensure!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "embedding value {index} differs: {a:?} versus {b:?}"
            );
        }
        Ok(())
    }

    #[derive(Default, Debug)]
    struct HeadComparison {
        maximum_error: f64,
        sign_mismatches: u64,
        zero_mismatches: u64,
    }
    fn compare_head(actual: &[f32], expected: &[f32]) -> Result<HeadComparison> {
        ensure!(
            actual.len() == expected.len(),
            "head comparison length differs"
        );
        let mut comparison = HeadComparison::default();
        for (index, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            let allowed = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * f64::from(b).abs();
            ensure!(
                a.is_finite() && b.is_finite() && error <= allowed,
                "logit {index}: got {a:?}, reference {b:?}, error {error}, allowed {allowed}"
            );
            comparison.maximum_error = comparison.maximum_error.max(error);
            comparison.sign_mismatches += u64::from(a.is_sign_negative() != b.is_sign_negative());
            comparison.zero_mismatches += u64::from((a == 0.0) != (b == 0.0));
        }
        Ok(comparison)
    }

    // Same greedy boundary contract as ForwardOutput: reject every nonfinite
    // value, choose the first index on ties, and require a representable token ID.
    fn greedy(logits: &[f32]) -> Result<u32> {
        let (&first, rest) = logits.split_first().context("empty logits")?;
        ensure!(first.is_finite(), "nonfinite logit0");
        let mut best = 0usize;
        let mut value = first;
        for (offset, &candidate) in rest.iter().enumerate() {
            ensure!(candidate.is_finite(), "nonfinite logit {}", offset + 1);
            if candidate > value {
                best = offset + 1;
                value = candidate;
            }
        }
        u32::try_from(best).context("greedy token does not fit u32")
    }

    fn embedding_reference(
        checkpoint: &Gguf,
        source: &CountingSource<'_>,
        token: usize,
    ) -> Result<Vec<f32>> {
        ensure!(token < VOCAB, "reference token exceeds vocabulary");
        let tensor = checkpoint
            .get("token_embd.weight")
            .context("missing CPU embedding")?;
        let row_bytes = (DIM / 32 * 34) as u64;
        let first = token as u64 * row_bytes;
        let resident = source.load_range(
            &tensor.locator().range(first..first + row_bytes)?,
            WeightAccess::Cyclic,
        )?;
        let mut output = vec![0.0; DIM];
        Cpu::new().dequantize(
            &mut output,
            &Weight {
                bytes: resident.bytes(),
                kind: tensor.kind(),
                rows: 1,
                cols: DIM,
            },
        )?;
        ensure!(
            output.iter().all(|v| v.is_finite()),
            "embedding CPU reference is nonfinite"
        );
        Ok(output)
    }

    // Derive row ranges directly from GGUF and the preregistered geometry, separate
    // from recorded plan ranges. Never decode or cache a complete vocabulary matrix.
    fn head_reference(
        checkpoint: &Gguf,
        source: &CountingSource<'_>,
        hidden: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>> {
        let norm = checkpoint
            .get("output_norm.weight")
            .context("missing CPU final norm")?;
        let normalized = {
            let resident =
                source.load_range(&norm.locator().whole_range(), WeightAccess::Cyclic)?;
            let values = resident
                .bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_le_bytes(*bytes))
                .collect::<Vec<_>>();
            norm_reference(hidden, Some(&values), eps)?
        };
        let tensor = checkpoint
            .get("token_embd.weight")
            .context("missing tied CPU head")?;
        let row_bytes = (DIM / 32 * 34) as u64;
        let cpu = Cpu::new();
        let mut logits = vec![0.0; VOCAB];
        for first in (0..VOCAB).step_by(TILE_ROWS) {
            let rows = TILE_ROWS.min(VOCAB - first);
            let start = first as u64 * row_bytes;
            let end = (first + rows) as u64 * row_bytes;
            let resident =
                source.load_range(&tensor.locator().range(start..end)?, WeightAccess::Cyclic)?;
            cpu.matvec_with_threads(
                &mut logits[first..first + rows],
                &normalized,
                &Weight {
                    bytes: resident.bytes(),
                    kind: tensor.kind(),
                    rows,
                    cols: DIM,
                },
                1,
            )?;
        }
        ensure!(
            logits.iter().all(|v| v.is_finite()),
            "CPU head reference is nonfinite"
        );
        Ok(logits)
    }

    // Recording uploads private parameters/scratch after the initial snapshot.
    // Keep those bytes distinct from caller writes and from zero-traffic submission.
    fn expected_host_writes(initial: u64, recording: u64, boundary: u64) -> Result<u64> {
        initial
            .checked_add(recording)
            .and_then(|total| total.checked_add(boundary))
            .context("vocabulary host-write accounting overflows")
    }

    fn run_model(options: &Options) -> Result<()> {
        let path = options.model.as_ref().context("missing model")?;
        check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
        let checkpoint = Gguf::open(path)?;
        check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
        let model = QwenModel::load(&checkpoint)?;
        let plan: VocabularyPlan = model.vocabulary_plan(TILE_ROWS)?;
        let tiles = plan
            .tiles()
            .iter()
            .map(|t| (t.first_token(), t.rows(), t.range().nbytes()))
            .collect::<Vec<_>>();
        let cap = inspect(
            descriptor(&checkpoint, plan.eps())?,
            plan.tile_rows(),
            &tiles,
            plan.memory(),
        )?;
        ensure!(
            plan.dim() == DIM && plan.vocab() == VOCAB,
            "bound plan geometry differs"
        );
        let config = NativeExecutionConfig {
            weight_capacity_bytes: cap.weights,
            tensor_capacity_bytes: cap.tensors,
            auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
        };
        let expected_allocation_bytes = checked_allocation_bytes(config, MODEL_BUDGET)?;
        let auxiliary_live = (DIM * 4 + 4) as u64;
        ensure!(
            auxiliary_live <= u64::from(AUXILIARY_CAPACITY),
            "head auxiliary bound differs"
        );
        let mut report = json!({
            "execution": "not_requested", "architecture": "qwen3", "input_width": DIM, "vocabulary_size": VOCAB,
            "tile_count": TILE_COUNT, "tile_rows": TILE_ROWS, "last_tile_rows": VOCAB % TILE_ROWS,
            "tied_embedding_head": true, "embedding_token_ids": TOKENS,
            "packed_weight_bytes": cap.memory.packed_weight_bytes, "source_weight_bytes": cap.source_bytes,
            "resident_tensor_bytes": cap.memory.resident_tensor_bytes, "largest_staging_bytes": cap.memory.largest_staging_bytes,
            "weight_arena_capacity_bytes": cap.weights, "tensor_arena_capacity_bytes": cap.tensors,
            "auxiliary_arena_capacity_bytes": AUXILIARY_CAPACITY, "expected_auxiliary_live_bytes": auxiliary_live,
            "hag_budget_bytes": MODEL_BUDGET, "expected_hag_allocation_bytes": expected_allocation_bytes,
            "required_hag_trace_submission": "0", "hag_trace_requirement_verified": options.execute,
            "external_process_and_thermal_limits_required": true,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
            "head_comparison_policy": "finite per-value tolerance and exact CPU greedy; sign/zero mismatches reported",
            "embedding_comparison_policy": "bitwise exact finite values including signed zero",
            "oracle": "independent CPU row decode; f64 RMS and one-worker row-tiled Q8 head before native open",
            "rmsnorm_epsilon": plan.eps(), "weight_format": "Q8_0", "norm_format": "F32"
        });
        if !options.execute {
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        println!("RUN preregistered head bound: |error| <= {ABSOLUTE_TOLERANCE} + {RELATIVE_TOLERANCE} * |reference|; finite and exact greedy; sign/zero mismatch counts reported");
        let oracle_source = CountingSource {
            inner: ResidentSource::new(&checkpoint),
            reads: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        };
        let embeddings = TOKENS
            .iter()
            .map(|token| embedding_reference(&checkpoint, &oracle_source, *token))
            .collect::<Result<Vec<_>>>()?;
        let heads = (0..HEAD_ROUNDS)
            .map(|round| {
                head_reference(&checkpoint, &oracle_source, &inputs(DIM, round), plan.eps())
            })
            .collect::<Result<Vec<_>>>()?;
        let reference_greedy = heads
            .iter()
            .map(|logits| greedy(logits))
            .collect::<Result<Vec<_>>>()?;
        let oracle_reads = oracle_source.reads.load(Ordering::Relaxed);
        let oracle_bytes = oracle_source.bytes.load(Ordering::Relaxed);
        ensure!(
            oracle_reads == (TOKENS.len() + HEAD_ROUNDS * (TILE_COUNT + 1)) as u64
                && oracle_bytes
                    == TOKENS.len() as u64 * (DIM / 32 * 34) as u64
                        + HEAD_ROUNDS as u64 * cap.source_bytes,
            "CPU oracle source accounting differs"
        );
        let reference_bytes = ((TOKENS.len() * DIM + HEAD_ROUNDS * VOCAB) * 4) as u64;
        ensure!(
            reference_bytes < 2 * 1024 * 1024,
            "retained reference outputs exceed2MiB"
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
            VocabularyLimits {
                packed_weight_bytes: cap.memory.packed_weight_bytes,
                resident_tensor_bytes: cap.memory.resident_tensor_bytes,
                largest_staging_bytes: cap.memory.largest_staging_bytes,
            },
        )?;
        ensure!(
            source.reads.load(Ordering::Relaxed) == (TILE_COUNT + 1) as u64
                && source.bytes.load(Ordering::Relaxed) == cap.source_bytes,
            "native vocabulary preparation must read38tiles+norm exactly once"
        );
        let hidden = device.upload_f32(&vec![GUARD; DIM + 2])?;
        let logits = device.upload_f32(&vec![GUARD; VOCAB + 2])?;
        let hidden_view = hidden.view(1, DIM)?;
        let logits_view = logits.view(1, VOCAB)?;
        let initial = verify_initial(&device, &budget, MODEL_BUDGET)?;
        ensure!(
            initial.allocation_bytes == expected_allocation_bytes
                && initial.weight_live_bytes == u64::from(cap.weights)
                && initial.tensor_live_bytes == u64::from(cap.tensors)
                && initial.uploaded_weight_bytes == cap.memory.packed_weight_bytes
                && initial.auxiliary_live_bytes == 0,
            "vocabulary initial resource accounting differs"
        );
        for (lookup, (&token, expected)) in TOKENS.iter().zip(&embeddings).enumerate() {
            let batch = prepared.record_embedding(&device, token, &hidden_view)?;
            ensure!(batch.len() == 1, "lookup must have one semantic command");
            let before = device.stats()?;
            ensure!(
                before.auxiliary_live_bytes == 4,
                "lookup must retain exactly one private dummy word"
            );
            let submission = device.submit(&batch)?;
            device.wait(&submission)?;
            verify_submission(before, device.stats()?, 1)?;
            let mut actual = vec![f32::NAN; DIM + 2];
            device.read_f32(&hidden.full_view(), &mut actual)?;
            compare_embedding(check_guards(&actual, DIM)?, expected)
                .with_context(|| format!("lookup token{token}"))?;
            drop((submission, batch));
            ensure!(
                device.stats()?.auxiliary_live_bytes == 0,
                "lookup dummy was retained after completion/batch drop"
            );
            println!("PASS embedding lookup={lookup} token={token} checked_values={DIM} guards=2 submission_tensor_transfer_bytes=0");
        }
        let head_batch = prepared.record_logits(&device, &hidden_view, &logits_view)?;
        ensure!(
            head_batch.len() == TILE_COUNT + 1,
            "head must record one weighted norm plus38matvecs"
        );
        let head_initial = device.stats()?;
        // The six retired lookup dummies and this head's norm parameter/scratch
        // were uploaded while recording, after the initial public-resource snapshot.
        let recording_write_bytes = (TOKENS.len() * 4) as u64 + auxiliary_live;
        ensure!(
            head_initial.host_write_bytes
                == expected_host_writes(initial.host_write_bytes, recording_write_bytes, 0)?,
            "recording parameter/scratch writes differ"
        );
        ensure!(
            head_initial.auxiliary_live_bytes == auxiliary_live
                && head_initial.auxiliary_peak_bytes == auxiliary_live,
            "head scratch accounting differs"
        );
        let mut maximum_error = 0.0f64;
        let mut sign_mismatches = 0u64;
        let mut zero_mismatches = 0u64;
        let mut greedy_tokens = Vec::new();
        for (round, expected) in heads.iter().enumerate() {
            device.write_f32(&hidden_view, &inputs(DIM, round))?;
            let before = device.stats()?;
            let submission = device.submit(&head_batch)?;
            device.wait(&submission)?;
            verify_submission(before, device.stats()?, (TILE_COUNT + 2) as u64)?;
            let mut actual = vec![f32::NAN; VOCAB + 2];
            device.read_f32(&logits.full_view(), &mut actual)?;
            let actual = check_guards(&actual, VOCAB)?;
            let comparison =
                compare_head(actual, expected).with_context(|| format!("head round{round}"))?;
            maximum_error = maximum_error.max(comparison.maximum_error);
            sign_mismatches += comparison.sign_mismatches;
            zero_mismatches += comparison.zero_mismatches;
            let selected = greedy(actual)?;
            ensure!(
                selected == reference_greedy[round],
                "head greedy differs: native{selected},CPU{}",
                reference_greedy[round]
            );
            greedy_tokens.push(selected);
            ensure!(
                budget.current_bytes() == expected_allocation_bytes,
                "vocabulary reuse changed HAG charge"
            );
            println!("PASS vocabulary round={round} checked_values={VOCAB} guards=2 greedy={selected} submission_tensor_transfer_bytes=0");
        }
        let final_stats = device.stats()?;
        let physical = (TOKENS.len() + HEAD_ROUNDS * (TILE_COUNT + 2)) as u64;
        let submissions = (TOKENS.len() + HEAD_ROUNDS) as u64;
        let readback = ((TOKENS.len() * (DIM + 2) + HEAD_ROUNDS * (VOCAB + 2)) * 4) as u64;
        ensure!(
            final_stats.dispatch_count == physical
                && final_stats.dispatch_attempts == physical
                && final_stats.logical_submissions == submissions,
            "vocabulary dispatch/submission accounting differs"
        );
        ensure!(
            final_stats.host_read_bytes == readback
                && final_stats.host_write_bytes
                    == expected_host_writes(
                        initial.host_write_bytes,
                        recording_write_bytes,
                        (HEAD_ROUNDS * DIM * 4) as u64
                    )?,
            "vocabulary boundary transfers differ"
        );
        ensure!(
            final_stats.uploaded_weight_bytes == cap.memory.packed_weight_bytes
                && source.reads.load(Ordering::Relaxed) == (TILE_COUNT + 1) as u64
                && source.bytes.load(Ordering::Relaxed) == cap.source_bytes,
            "shared vocabulary reread/reuploaded weights"
        );
        ensure!(
            final_stats.weight_peak_bytes == initial.weight_peak_bytes
                && final_stats.tensor_peak_bytes == initial.tensor_peak_bytes
                && final_stats.auxiliary_live_bytes == auxiliary_live
                && final_stats.auxiliary_peak_bytes == auxiliary_live,
            "vocabulary reuse grew resident resources"
        );
        drop((
            head_batch,
            prepared,
            hidden_view,
            logits_view,
            hidden,
            logits,
        ));
        verify_retired(&device)?;
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "HAG retained vocabulary resources after cleanup"
        );
        report["execution"] = json!("native-hag-vocabulary");
        report["embedding_lookups"] = json!(TOKENS.len());
        report["head_rounds"] = json!(HEAD_ROUNDS);
        report["embedding_checked_values"] = json!(TOKENS.len() * DIM);
        report["head_checked_values"] = json!(HEAD_ROUNDS * VOCAB);
        report["checked_values"] = json!(TOKENS.len() * DIM + HEAD_ROUNDS * VOCAB);
        report["guard_checks"] = json!(2 * (TOKENS.len() + HEAD_ROUNDS));
        report["embedding_exact"] = json!(true);
        report["greedy_matches"] = json!(greedy_tokens.len());
        report["greedy_tokens"] = json!(greedy_tokens);
        report["head_sign_mismatches"] = json!(sign_mismatches);
        report["head_zero_mismatches"] = json!(zero_mismatches);
        report["head_semantic_commands"] = json!(TILE_COUNT + 1);
        report["head_physical_dispatches_per_round"] = json!(TILE_COUNT + 2);
        report["physical_dispatches"] = json!(final_stats.dispatch_count);
        report["logical_submissions"] = json!(final_stats.logical_submissions);
        report["native_preparation_source_reads"] = json!(TILE_COUNT + 1);
        report["cpu_oracle_source_reads"] = json!(oracle_reads);
        report["cpu_oracle_source_bytes"] = json!(oracle_bytes);
        report["retained_reference_bytes"] = json!(reference_bytes);
        report["actual_hag_allocation_bytes"] = json!(initial.allocation_bytes);
        report["auxiliary_live_bytes"] = json!(auxiliary_live);
        report["budget_after"] = json!(0);
        report["submission_tensor_transfer_bytes"] = json!(0);
        report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
        report["recording_host_write_bytes"] = json!(recording_write_bytes);
        report["max_absolute_error"] = json!(maximum_error);
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
        fn fixture() -> (Descriptor, Vec<(usize, usize, u64)>, VocabularyMemory) {
            let descriptor = Descriptor {
                kind: GgmlType(8),
                dim: DIM,
                vocab: VOCAB,
                bytes: 165306368,
                norm_kind: GgmlType(0),
                norm_len: DIM,
                norm_bytes: 4096,
                tied: true,
                eps: 1e-6,
            };
            let tiles = (0..VOCAB)
                .step_by(TILE_ROWS)
                .map(|first| {
                    let rows = TILE_ROWS.min(VOCAB - first);
                    (first, rows, (rows * DIM / 32 * 34) as u64)
                })
                .collect();
            (
                descriptor,
                tiles,
                VocabularyMemory {
                    packed_weight_bytes: 165306368,
                    resident_tensor_bytes: 8192,
                    largest_staging_bytes: 4456448,
                },
            )
        }
        #[test]
        fn safe_cli_and_exact_trace_opt_in() {
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
            for bad in [
                vec!["/dev/dri/renderD128"],
                vec!["--execute"],
                vec!["--execute", "--execute"],
                vec!["--layer", "0"],
                vec!["--model", "a", "--model", "b"],
                vec!["--diagnose-first"],
            ] {
                assert!(parse(args(&bad)).is_err());
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
        fn geometry_shared_costs_guard_traffic_and_ranges_are_preregistered() {
            let (d, tiles, memory) = fixture();
            let cap = inspect(d, TILE_ROWS, &tiles, memory).unwrap();
            assert_eq!(tiles.len(), 38);
            assert_eq!(tiles[36], (147456, 4096, 4456448));
            assert_eq!(tiles[37], (151552, 384, 417792));
            assert_eq!(cap.weights, 165306368);
            assert_eq!(cap.tensors, 620048);
            assert_eq!(cap.source_bytes, 165310464);
            assert_eq!(
                TOKENS.map(|t| (t / TILE_ROWS, t % TILE_ROWS)),
                [(0, 0), (0, 4095), (1, 0), (16, 0), (36, 4095), (37, 383)]
            );
            assert_eq!(
                (TOKENS.len() * (DIM + 2) + HEAD_ROUNDS * (VOCAB + 2)) * 4,
                1847880
            );
            assert_eq!(TOKENS.len() + HEAD_ROUNDS * (TILE_COUNT + 1), 123);
            assert_eq!(TOKENS.len() * 4 + DIM * 4 + 4, 4124);
            let config = NativeExecutionConfig {
                weight_capacity_bytes: cap.weights,
                tensor_capacity_bytes: cap.tensors,
                auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
            };
            let cost = checked_allocation_bytes(config, MODEL_BUDGET).unwrap();
            assert!(cost < MODEL_BUDGET);
            assert!(checked_allocation_bytes(config, cost - 1).is_err());
        }
        #[test]
        fn post_initial_recording_uploads_are_included_before_boundary_writes() {
            let initial = 100;
            let recording = (TOKENS.len() * 4 + DIM * 4 + 4) as u64;
            let boundary = (HEAD_ROUNDS * DIM * 4) as u64;
            assert_eq!(recording, 4124);
            assert_eq!(expected_host_writes(initial, recording, 0).unwrap(), 4224);
            let complete = expected_host_writes(initial, recording, boundary).unwrap();
            assert_eq!(complete, 16512);
            assert_ne!(complete, initial + boundary);
            assert!(expected_host_writes(u64::MAX, 1, 0).is_err());
            assert!(expected_host_writes(0, u64::MAX, 1).is_err());
        }

        #[test]
        fn inspector_refuses_untied_format_shape_span_tile_and_cost_changes() {
            let (d, tiles, memory) = fixture();
            for bad in [
                Descriptor { tied: false, ..d },
                Descriptor {
                    kind: GgmlType(0),
                    ..d
                },
                Descriptor {
                    dim: usize::MAX,
                    ..d
                },
                Descriptor { vocab: 0, ..d },
                Descriptor { bytes: 1, ..d },
                Descriptor {
                    norm_kind: GgmlType(8),
                    ..d
                },
                Descriptor { norm_len: 1, ..d },
                Descriptor { norm_bytes: 1, ..d },
                Descriptor { eps: f32::NAN, ..d },
                Descriptor { eps: -1.0, ..d },
            ] {
                assert!(inspect(bad, TILE_ROWS, &tiles, memory).is_err());
            }
            assert!(inspect(d, 4095, &tiles, memory).is_err());
            assert!(inspect(d, TILE_ROWS, &tiles[..37], memory).is_err());
            let mut bad = tiles.clone();
            bad[37].1 += 1;
            assert!(inspect(d, TILE_ROWS, &bad, memory).is_err());
            let mut bad = tiles.clone();
            bad[1].0 += 1;
            assert!(inspect(d, TILE_ROWS, &bad, memory).is_err());
            assert!(inspect(
                d,
                TILE_ROWS,
                &tiles,
                VocabularyMemory {
                    resident_tensor_bytes: 0,
                    ..memory
                }
            )
            .is_err());
        }
        #[test]
        fn embedding_exactness_head_policy_and_guards_are_distinct() {
            compare_embedding(&[1.0, -0.0], &[1.0, -0.0]).unwrap();
            assert!(compare_embedding(&[0.0], &[-0.0]).is_err());
            assert!(compare_embedding(&[1.0001], &[1.0]).is_err());
            let result = compare_head(&[-1e-8, 0.0, -0.0], &[1e-8, 1e-8, 0.0]).unwrap();
            assert_eq!(result.sign_mismatches, 2);
            assert_eq!(result.zero_mismatches, 1);
            for (a, b) in [(f32::NAN, 1.0), (1.0, f32::INFINITY), (1.001, 1.0)] {
                assert!(compare_head(&[a], &[b]).is_err());
            }
            assert_eq!(check_guards(&[GUARD, 1.0, GUARD], 1).unwrap(), &[1.0]);
            assert!(check_guards(&[0.0, 1.0, GUARD], 1).is_err());
            assert!(check_guards(&[GUARD, 1.0], 1).is_err());
        }
        #[test]
        fn greedy_matches_existing_first_index_and_nonfinite_contract() {
            use beholdd_model::qwen3moe::ForwardOutput;
            for values in [
                vec![-1.0, 3.0, 3.0, 2.0],
                vec![-0.0, 0.0],
                vec![f32::NAN, 1.0],
                vec![1.0, f32::INFINITY],
            ] {
                let prior = ForwardOutput {
                    n_positions: 1,
                    dim: 1,
                    vocab: values.len(),
                    tokens: vec![0],
                    hidden: vec![],
                    logits: values.clone(),
                }
                .greedy_token();
                let current = greedy(&values);
                assert_eq!(current.is_ok(), prior.is_ok());
                if let Ok(token) = prior {
                    assert_eq!(current.unwrap(), token);
                }
            }
            assert!(greedy(&[]).is_err());
            assert_eq!(greedy(&[2.0, 2.0, 1.0]).unwrap(), 0);
        }
        #[test]
        fn cpu_norm_and_changed_inputs_are_finite_and_bounded() {
            assert_eq!(
                norm_reference(&[2.0, -2.0], Some(&[0.5, 2.0]), 0.0).unwrap(),
                vec![0.5, -2.0]
            );
            let input = (0..HEAD_ROUNDS)
                .map(|round| inputs(DIM, round))
                .collect::<Vec<_>>();
            for i in 0..HEAD_ROUNDS {
                assert!(input[i].iter().all(|v| v.is_finite()));
                for j in i + 1..HEAD_ROUNDS {
                    assert_ne!(input[i], input[j]);
                }
            }
            assert!((TOKENS.len() * DIM + HEAD_ROUNDS * VOCAB) * 4 < 2 * 1024 * 1024);
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::run()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native HAG vocabulary qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
