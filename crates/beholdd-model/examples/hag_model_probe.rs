//! Supervised full dense Qwen3 checkpoint qualification, never a serving default.
//! Metadata-only by default. CPU references finish before native device access,
//! staging one packed matrix/tile at a time. Native intermediates and KV stay resident.
//! Preregistered logits policy: finite, 2e-3 + 2e-4*abs(CPU), exact CPU greedy;
//! sign/zero differences are reported. No automatic tolerance relaxation or retry.

#[cfg(target_os = "linux")]
mod proof {
    use anyhow::{bail, ensure, Context, Result};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::hag::native_execution::{
        MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    };
    use beholdd_backend::{Device, IncrementalAttentionConfig, RopeLayout, Weight};
    use beholdd_format::{CheckpointId, GgmlType, Gguf, WeightRange};
    use beholdd_memory::{
        check_checkpoint_storage, ResidentSource, ResidentWeight, StorageOverride, WeightAccess,
        WeightSource,
    };
    use beholdd_model::qwen3moe::{DenseModelLimits, DenseModelMemory, QwenArchitecture};
    use beholdd_model::QwenModel;
    use serde_json::json;
    use std::ffi::{OsStr, OsString};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    const USAGE: &str = "hag_model_probe [--model PATH]\n\
        hag_model_probe --first-token --model PATH /dev/dri/renderD128\n\
        hag_model_probe --execute --model PATH /dev/dri/renderD128\n\
        hag_model_probe --inspect-context --model PATH\n\
        hag_model_probe --context-boundary --model PATH /dev/dri/renderD128\n\
        hag_model_probe --inspect-on-demand --model PATH\n\
        hag_model_probe --on-demand-first-token --model PATH /dev/dri/renderD128\n\
        hag_model_probe --on-demand-context --model PATH /dev/dri/renderD128\n\
        hag_model_probe --inspect-on-demand512 --model PATH\n\
        hag_model_probe --on-demand512-first-token --model PATH /dev/dri/renderD128\n\
        hag_model_probe --on-demand512-context --model PATH /dev/dri/renderD128\n\
        hag_model_probe --inspect-on-demand2048 --model PATH\n\
        hag_model_probe --on-demand2048-reference --model PATH --reference-out PATH --input-identity PATH\n\
        hag_model_probe --on-demand2048-first-token|--on-demand2048-context --model PATH --reference PATH --input-identity PATH /dev/dri/renderD128\n\
        hag_model_probe --capacity16384-reference --model PATH --reference-out PATH --input-identity PATH\n\
        hag_model_probe --capacity16384-smoke --model PATH --reference PATH --input-identity PATH /dev/dri/renderD128\n\
        hag_model_probe --catalog-prefill32-reference --model PATH --prompt-tokens PATH --reference-out PATH --input-identity PATH\n\
        hag_model_probe --catalog-prefill32-smoke --model PATH --prompt-tokens PATH --reference PATH --input-identity PATH /dev/dri/renderD128\n\
        Default: metadata only. Fixed Qwen3-0.6B Q8_0, C8, 768MiB HAG budget.\n\
        Context modes: separate C128 gate, 128 positions plus reset/replay.\n\
        On-demand modes: separate C128 recording-lifetime gate, unchanged numerical fixtures.\n\
        On-demand512 modes: additive C512 gate with position-dependent scratch accounting.\n\
        On-demand2048 modes: historical C2048 boundary gate with separate supervised CPU and native jobs.\n\
        Capacity16384 modes: full C16384 allocation, two token positions and reset/replay only.\n\
        Catalog-prefill32 modes: first32 of the frozen8079 prompt IDs, checkpoints0/31 and reset/replay.\n\
        Execution requires HAG_TRACE_SUBMISSION=0 and external process/thermal/crash supervision.\n\
        Full execution follows a successful first-token gate on the same frozen binary.";
    const CAPACITY: usize = 8;
    const DIM: usize = 1024;
    const VOCAB: usize = 151_936;
    const LAYERS: usize = 28;
    const TILE_ROWS: usize = 4096;
    const REQUESTS: usize = 2;
    const STEPS_PER_REQUEST: usize = 3;
    const PROMPTS: [[u32; 2]; REQUESTS] = [[1, 2], [4, 5]];
    const PACKED: u64 = 633_233_408;
    const TENSORS: u64 = 4_548_096;
    const STAGING: u64 = 4_456_448;
    const SOURCE_BYTES: u64 = 633_495_552;
    const PREPARATION_READS: u64 = 347;
    const PERSISTENT_AUXILIARY: u64 = 1_039_316;
    // upload_f32 allocates zero-filled storage, then writes the supplied norm/RoPE values.
    const UPLOADED_TENSOR_VALUE_BYTES: u64 = 376_832;
    const INITIAL_HOST_WRITE_BYTES: u64 =
        PACKED + TENSORS + PERSISTENT_AUXILIARY + UPLOADED_TENSOR_VALUE_BYTES;
    const AUXILIARY_CAPACITY: u32 = 1_039_320;
    const EXPECTED_ALLOCATION: u64 = 638_861_312;
    const MODEL_BUDGET: u64 = 768 * 1024 * 1024;
    const PHYSICAL_PER_STEP: u64 = 3793;
    const SUBMISSIONS_PER_STEP: u64 = 58;
    const ABSOLUTE_TOLERANCE: f64 = 2e-3;
    const RELATIVE_TOLERANCE: f64 = 2e-4;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Mode {
        Inspect,
        FirstToken,
        Execute,
        InspectContext,
        ContextBoundary,
        InspectOnDemand,
        OnDemandFirstToken,
        OnDemandContext,
        InspectOnDemand512,
        InspectOnDemand2048,
        OnDemand512FirstToken,
        OnDemand512Context,
        OnDemand2048Reference,
        OnDemand2048FirstToken,
        OnDemand2048Context,
        Capacity16384Reference,
        Capacity16384Smoke,
        CatalogPrefill32Reference,
        CatalogPrefill32Smoke,
    }
    impl Mode {
        fn executes(self) -> bool {
            matches!(
                self,
                Self::FirstToken
                    | Self::Execute
                    | Self::ContextBoundary
                    | Self::OnDemandFirstToken
                    | Self::OnDemandContext
                    | Self::OnDemand512FirstToken
                    | Self::OnDemand512Context
                    | Self::OnDemand2048FirstToken
                    | Self::OnDemand2048Context
                    | Self::Capacity16384Smoke
                    | Self::CatalogPrefill32Smoke
            )
        }
        fn steps(self) -> usize {
            match self {
                Self::Inspect
                | Self::InspectContext
                | Self::InspectOnDemand
                | Self::InspectOnDemand512
                | Self::InspectOnDemand2048 => 0,
                Self::FirstToken
                | Self::OnDemandFirstToken
                | Self::OnDemand512FirstToken
                | Self::OnDemand2048FirstToken => 1,
                Self::Execute => 6,
                Self::ContextBoundary => context::CONTEXT + 1,
                Self::OnDemandContext => demand::CONTEXT + 1,
                Self::OnDemand512Context => demand512::CONTEXT + 1,
                Self::OnDemand2048Reference => demand2048::CONTEXT,
                Self::OnDemand2048Context => demand2048::CONTEXT + 1,
                Self::Capacity16384Reference => capacity16384::CPU_STEPS,
                Self::Capacity16384Smoke => capacity16384::CPU_STEPS + 1,
                Self::CatalogPrefill32Reference => catalog_prefill32::CPU_STEPS,
                Self::CatalogPrefill32Smoke => catalog_prefill32::CPU_STEPS + 1,
            }
        }
        fn requests(self) -> usize {
            if matches!(
                self,
                Self::Execute
                    | Self::ContextBoundary
                    | Self::OnDemandContext
                    | Self::OnDemand512Context
                    | Self::OnDemand2048Context
                    | Self::Capacity16384Smoke
                    | Self::CatalogPrefill32Smoke
            ) {
                2
            } else {
                1
            }
        }
        fn execution(self) -> &'static str {
            match self {
                Self::Inspect
                | Self::InspectContext
                | Self::InspectOnDemand
                | Self::InspectOnDemand512
                | Self::InspectOnDemand2048 => "not_requested",
                Self::FirstToken => "native-hag-model-first-token",
                Self::Execute => "native-hag-model",
                Self::ContextBoundary => "native-hag-context-boundary",
                Self::OnDemandFirstToken => "native-hag-on-demand-first-token",
                Self::OnDemandContext => "native-hag-on-demand-context",
                Self::OnDemand512FirstToken => "native-hag-on-demand512-first-token",
                Self::OnDemand512Context => "native-hag-on-demand512-context",
                Self::OnDemand2048Reference => "cpu-hag-on-demand2048-reference",
                Self::OnDemand2048FirstToken => "native-hag-on-demand2048-first-token",
                Self::OnDemand2048Context => "native-hag-on-demand2048-context",
                Self::Capacity16384Reference => "cpu-hag-capacity16384-reference",
                Self::Capacity16384Smoke => "native-hag-capacity16384-smoke",
                Self::CatalogPrefill32Reference => "cpu-hag-catalog-prefill32-reference",
                Self::CatalogPrefill32Smoke => "native-hag-catalog-prefill32-smoke",
            }
        }
    }
    #[derive(Debug)]
    struct Options {
        mode: Mode,
        model: Option<PathBuf>,
        device: Option<PathBuf>,
        reference: Option<PathBuf>,
        reference_out: Option<PathBuf>,
        input_identity: Option<PathBuf>,
        prompt_tokens: Option<PathBuf>,
        help: bool,
    }
    fn parse(arguments: impl IntoIterator<Item = OsString>) -> Result<Options> {
        let mut args = arguments.into_iter();
        let mut options = Options {
            mode: Mode::Inspect,
            model: None,
            device: None,
            reference: None,
            reference_out: None,
            input_identity: None,
            prompt_tokens: None,
            help: false,
        };
        while let Some(arg) = args.next() {
            match arg.to_str() {
                Some("--help" | "-h") => options.help = true,
                Some("--model") if options.model.is_none() => {
                    options.model = Some(PathBuf::from(args.next().context("--model needs PATH")?))
                }
                Some("--reference") if options.reference.is_none() => {
                    options.reference = Some(PathBuf::from(
                        args.next().context("--reference needs PATH")?,
                    ))
                }
                Some("--reference-out") if options.reference_out.is_none() => {
                    options.reference_out = Some(PathBuf::from(
                        args.next().context("--reference-out needs PATH")?,
                    ))
                }
                Some("--input-identity") if options.input_identity.is_none() => {
                    options.input_identity = Some(PathBuf::from(
                        args.next().context("--input-identity needs PATH")?,
                    ))
                }
                Some("--prompt-tokens") if options.prompt_tokens.is_none() => {
                    options.prompt_tokens = Some(PathBuf::from(
                        args.next().context("--prompt-tokens needs PATH")?,
                    ))
                }
                Some("--first-token") if options.mode == Mode::Inspect => {
                    options.mode = Mode::FirstToken
                }
                Some("--execute") if options.mode == Mode::Inspect => options.mode = Mode::Execute,
                Some("--inspect-context") if options.mode == Mode::Inspect => {
                    options.mode = Mode::InspectContext
                }
                Some("--context-boundary") if options.mode == Mode::Inspect => {
                    options.mode = Mode::ContextBoundary
                }
                Some("--inspect-on-demand") if options.mode == Mode::Inspect => {
                    options.mode = Mode::InspectOnDemand
                }
                Some("--on-demand-first-token") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemandFirstToken
                }
                Some("--on-demand-context") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemandContext
                }
                Some("--inspect-on-demand512") if options.mode == Mode::Inspect => {
                    options.mode = Mode::InspectOnDemand512
                }
                Some("--inspect-on-demand2048") if options.mode == Mode::Inspect => {
                    options.mode = Mode::InspectOnDemand2048
                }
                Some("--on-demand512-first-token") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemand512FirstToken
                }
                Some("--on-demand512-context") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemand512Context
                }
                Some("--on-demand2048-reference") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemand2048Reference
                }
                Some("--on-demand2048-first-token") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemand2048FirstToken
                }
                Some("--on-demand2048-context") if options.mode == Mode::Inspect => {
                    options.mode = Mode::OnDemand2048Context
                }
                Some("--capacity16384-reference") if options.mode == Mode::Inspect => {
                    options.mode = Mode::Capacity16384Reference
                }
                Some("--capacity16384-smoke") if options.mode == Mode::Inspect => {
                    options.mode = Mode::Capacity16384Smoke
                }
                Some("--catalog-prefill32-reference") if options.mode == Mode::Inspect => {
                    options.mode = Mode::CatalogPrefill32Reference
                }
                Some("--catalog-prefill32-smoke") if options.mode == Mode::Inspect => {
                    options.mode = Mode::CatalogPrefill32Smoke
                }
                Some(value) if !value.starts_with('-') && options.device.is_none() => {
                    options.device = Some(PathBuf::from(arg))
                }
                _ => bail!("unknown or repeated argument {arg:?}\n{USAGE}"),
            }
        }
        if !options.help {
            ensure!(
                options.mode.executes() == options.device.is_some(),
                "explicit execution mode and render node are required together"
            );
            ensure!(
                !options.mode.executes() || options.model.is_some(),
                "execution requires --model PATH"
            );
            ensure!(
                matches!(
                    options.mode,
                    Mode::CatalogPrefill32Reference | Mode::CatalogPrefill32Smoke
                ) == options.prompt_tokens.is_some(),
                "catalog prefix modes require --prompt-tokens, which other modes refuse"
            );
            match options.mode {
                Mode::OnDemand2048Reference
                | Mode::Capacity16384Reference
                | Mode::CatalogPrefill32Reference => ensure!(
                    options.model.is_some()
                        && options.input_identity.is_some()
                        && options.reference_out.is_some()
                        && options.reference.is_none(),
                    "CPU export requires model, input identity and exclusive reference output"
                ),
                Mode::OnDemand2048FirstToken
                | Mode::OnDemand2048Context
                | Mode::Capacity16384Smoke
                | Mode::CatalogPrefill32Smoke => ensure!(
                    options.input_identity.is_some()
                        && options.reference.is_some()
                        && options.reference_out.is_none(),
                    "native execution requires input identity and completed CPU reference"
                ),
                _ => ensure!(
                    options.reference.is_none()
                        && options.reference_out.is_none()
                        && options.input_identity.is_none(),
                    "reference arguments require an explicit separate-process reference mode"
                ),
            }
        }
        Ok(options)
    }
    fn check_trace(value: Option<&OsStr>) -> Result<()> {
        ensure!(
            value == Some(OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly0; payload tracing refused"
        );
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    struct Shape {
        layers: usize,
        dim: usize,
        heads: usize,
        kv_heads: usize,
        head: usize,
        ff: usize,
        vocab: usize,
        context: usize,
        experts: usize,
        eps: f32,
        theta: f64,
    }
    impl Shape {
        fn from_model(model: &QwenModel) -> Self {
            let c = model.config();
            Self {
                layers: c.n_layers(),
                dim: c.dim(),
                heads: c.n_heads(),
                kv_heads: c.n_kv_heads(),
                head: c.head_dim(),
                ff: c.feed_forward_length(),
                vocab: c.vocab(),
                context: c.context(),
                experts: c.n_experts(),
                eps: c.eps(),
                theta: c.rope_base(),
            }
        }
    }
    fn inspect_geometry(shape: Shape, capacity: usize) -> Result<()> {
        ensure!(
            (
                shape.layers,
                shape.dim,
                shape.heads,
                shape.kv_heads,
                shape.head,
                shape.ff,
                shape.vocab,
                shape.experts
            ) == (28, 1024, 16, 8, 128, 3072, 151936, 0),
            "qualification requires exact dense Qwen3-0.6B dimensions"
        );
        ensure!(
            shape.context >= capacity
                && shape.eps.to_bits() == 1e-6_f32.to_bits()
                && shape.theta == 1_000_000.0,
            "qualification context/epsilon/RoPE base differs"
        );
        Ok(())
    }
    fn inspect_shape(shape: Shape, memory: DenseModelMemory) -> Result<NativeExecutionConfig> {
        inspect_geometry(shape, CAPACITY)?;
        ensure!(
            memory.packed_weight_bytes == PACKED
                && memory.resident_tensor_bytes == TENSORS
                && memory.largest_staging_bytes == STAGING,
            "full-model logical plan costs differ"
        );
        let config = NativeExecutionConfig {
            weight_capacity_bytes: u32::try_from(PACKED)?,
            tensor_capacity_bytes: u32::try_from(TENSORS)?,
            auxiliary_capacity_bytes: AUXILIARY_CAPACITY,
        };
        ensure!(
            config.allocation_bytes()? == EXPECTED_ALLOCATION
                && EXPECTED_ALLOCATION <= MODEL_BUDGET,
            "full-model page-rounded charge differs"
        );
        Ok(config)
    }
    // Independent exact descriptor checks precede any payload read or native open.
    fn inspect_checkpoint(checkpoint: &Gguf, shape: Shape) -> Result<()> {
        ensure!(
            checkpoint.arch() == Some("qwen3") && checkpoint.get("output.weight").is_none(),
            "qualification requires tied dense qwen3"
        );
        let mut source_bytes = 0u64;
        let mut check = |name: &str, kind: u32, dims: &[u64]| -> Result<()> {
            let tensor = checkpoint
                .get(name)
                .with_context(|| format!("missing {name}"))?;
            ensure!(
                tensor.kind() == GgmlType(kind) && tensor.dims() == dims,
                "descriptor differs for {name}"
            );
            let elements: u64 = dims.iter().product();
            let bytes = if kind == 8 {
                elements / 32 * 34
            } else {
                elements * 4
            };
            ensure!(tensor.nbytes() == bytes, "packed extent differs for {name}");
            source_bytes += bytes;
            Ok(())
        };
        check("token_embd.weight", 8, &[DIM as u64, VOCAB as u64])?;
        check("output_norm.weight", 0, &[DIM as u64])?;
        for layer in 0..shape.layers {
            for (name, rows, cols) in [
                ("attn_q", 2048, 1024),
                ("attn_k", 1024, 1024),
                ("attn_v", 1024, 1024),
                ("attn_output", 1024, 2048),
                ("ffn_gate", 3072, 1024),
                ("ffn_up", 3072, 1024),
                ("ffn_down", 1024, 3072),
            ] {
                check(&format!("blk.{layer}.{name}.weight"), 8, &[cols, rows])?;
            }
            for (name, width) in [
                ("attn_norm", 1024),
                ("attn_q_norm", 128),
                ("attn_k_norm", 128),
                ("ffn_norm", 1024),
            ] {
                check(&format!("blk.{layer}.{name}.weight"), 0, &[width])?;
            }
        }
        ensure!(
            source_bytes == SOURCE_BYTES,
            "unique checkpoint weight bytes differ"
        );
        Ok(())
    }

    struct CountingSource<'a> {
        inner: ResidentSource<'a>,
        reads: AtomicU64,
        bytes: AtomicU64,
        largest: AtomicU64,
    }
    impl<'a> CountingSource<'a> {
        fn new(checkpoint: &'a Gguf) -> Self {
            Self {
                inner: ResidentSource::new(checkpoint),
                reads: AtomicU64::new(0),
                bytes: AtomicU64::new(0),
                largest: AtomicU64::new(0),
            }
        }
        fn counts(&self) -> (u64, u64, u64) {
            (
                self.reads.load(Ordering::Relaxed),
                self.bytes.load(Ordering::Relaxed),
                self.largest.load(Ordering::Relaxed),
            )
        }
    }
    impl WeightSource for CountingSource<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.inner.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            ensure!(
                range.nbytes() <= STAGING,
                "source range exceeds one-matrix/tile staging cap"
            );
            let resident = self.inner.load_range(range, access)?;
            ensure!(
                resident.bytes().len() as u64 == range.nbytes(),
                "source extent differs"
            );
            self.reads.fetch_add(1, Ordering::Relaxed);
            self.bytes.fetch_add(range.nbytes(), Ordering::Relaxed);
            self.largest.fetch_max(range.nbytes(), Ordering::Relaxed);
            Ok(resident)
        }
    }

    fn greedy(logits: &[f32]) -> Result<u32> {
        let (&first, rest) = logits.split_first().context("empty logits")?;
        ensure!(first.is_finite(), "nonfinite logit0");
        let mut best = 0;
        let mut value = first;
        for (index, &candidate) in rest.iter().enumerate() {
            ensure!(candidate.is_finite(), "nonfinite logit{}", index + 1);
            if candidate > value {
                best = index + 1;
                value = candidate;
            }
        }
        u32::try_from(best).context("greedy token exceeds u32")
    }
    #[derive(Default, Debug)]
    struct Comparison {
        maximum: f64,
        signs: u64,
        zeros: u64,
    }
    fn compare(actual: &[f32], expected: &[f32]) -> Result<Comparison> {
        ensure!(
            !actual.is_empty() && actual.len() == expected.len(),
            "logit comparison shape differs"
        );
        let mut result = Comparison::default();
        for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
            let error = (f64::from(a) - f64::from(b)).abs();
            let bound = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * f64::from(b).abs();
            ensure!(
                a.is_finite() && b.is_finite() && error <= bound,
                "logit{i}: {a:?} versus {b:?}; error{error}, allowed{bound}"
            );
            result.maximum = result.maximum.max(error);
            result.signs += u64::from(a.is_sign_negative() != b.is_sign_negative());
            result.zeros += u64::from((a == 0.0) != (b == 0.0));
        }
        Ok(result)
    }

    fn norm_reference(input: &[f32], weights: &[f32], eps: f32) -> Result<Vec<f32>> {
        ensure!(
            !input.is_empty()
                && input.len() == weights.len()
                && eps.is_finite()
                && eps >= 0.0
                && input.iter().chain(weights).all(|x| x.is_finite()),
            "invalid oracle norm input"
        );
        let square_sum: f64 = input.iter().map(|x| f64::from(*x) * f64::from(*x)).sum();
        let inverse = (1.0 / (square_sum / input.len() as f64 + f64::from(eps)).sqrt()) as f32;
        let output: Vec<_> = input
            .iter()
            .zip(weights)
            .map(|(x, w)| (*x * inverse) * *w)
            .collect();
        ensure!(
            output.iter().all(|x| x.is_finite()),
            "nonfinite oracle norm output"
        );
        Ok(output)
    }

    // Independent immediate CPU composition, not the recorded plan or a native
    // readback. Every packed source owner is scoped to one matrix/tile operation.
    struct Oracle<'a, 'g> {
        checkpoint: &'g Gguf,
        source: &'a CountingSource<'g>,
        shape: Shape,
        caches: Vec<(Vec<f32>, Vec<f32>)>,
        position: usize,
        capacity: usize,
    }
    impl<'a, 'g> Oracle<'a, 'g> {
        fn new(checkpoint: &'g Gguf, source: &'a CountingSource<'g>, shape: Shape) -> Self {
            Self::with_capacity(checkpoint, source, shape, STEPS_PER_REQUEST)
        }
        fn with_capacity(
            checkpoint: &'g Gguf,
            source: &'a CountingSource<'g>,
            shape: Shape,
            capacity: usize,
        ) -> Self {
            let n = capacity * shape.kv_heads * shape.head;
            Self {
                checkpoint,
                source,
                shape,
                position: 0,
                capacity,
                caches: (0..shape.layers)
                    .map(|_| (Vec::with_capacity(n), Vec::with_capacity(n)))
                    .collect(),
            }
        }
        fn norm(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
            let tensor = self.checkpoint.get(name).context("missing oracle norm")?;
            let raw = self
                .source
                .load_range(&tensor.locator().whole_range(), WeightAccess::Cyclic)?;
            let weights: Vec<_> = raw
                .bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(
                weights.len() == input.len() && weights.iter().all(|v| v.is_finite()),
                "invalid oracle norm"
            );
            norm_reference(input, &weights, self.shape.eps)
        }
        fn project(&self, name: &str, input: &[f32], rows: usize) -> Result<Vec<f32>> {
            let tensor = self.checkpoint.get(name).context("missing oracle matrix")?;
            let raw = self
                .source
                .load_range(&tensor.locator().whole_range(), WeightAccess::Cyclic)?;
            let mut output = vec![0.0; rows];
            Cpu::new().matvec_with_threads(
                &mut output,
                input,
                &Weight {
                    bytes: raw.bytes(),
                    kind: tensor.kind(),
                    rows,
                    cols: input.len(),
                },
                1,
            )?;
            Ok(output)
        }
        fn norm_heads(&self, name: &str, input: &[f32]) -> Result<Vec<f32>> {
            let tensor = self
                .checkpoint
                .get(name)
                .context("missing oracle head norm")?;
            let raw = self
                .source
                .load_range(&tensor.locator().whole_range(), WeightAccess::Cyclic)?;
            let weights: Vec<_> = raw
                .bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            ensure!(
                weights.len() == self.shape.head,
                "oracle head norm shape differs"
            );
            let mut output = vec![0.0; input.len()];
            let cpu = Cpu::new();
            let mut cos = Vec::with_capacity(self.shape.head / 2);
            let mut sin = Vec::with_capacity(self.shape.head / 2);
            for i in 0..self.shape.head / 2 {
                let frequency = 1.0
                    / self
                        .shape
                        .theta
                        .powf((2 * i) as f64 / self.shape.head as f64);
                let angle = self.position as f64 * frequency;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
            for (source, destination) in input
                .chunks_exact(self.shape.head)
                .zip(output.chunks_exact_mut(self.shape.head))
            {
                destination.copy_from_slice(&norm_reference(source, &weights, self.shape.eps)?);
                cpu.rope(destination, &cos, &sin, RopeLayout::SplitHalf, false)?;
            }
            Ok(output)
        }
        fn step(&mut self, token: u32) -> Result<Vec<f32>> {
            let s = self.shape;
            ensure!(
                (token as usize) < s.vocab && self.position < self.capacity,
                "oracle token/position exceeds fixture"
            );
            let cpu = Cpu::new();
            let mut hidden = {
                let tensor = self
                    .checkpoint
                    .get("token_embd.weight")
                    .context("missing oracle embedding")?;
                let stride = (s.dim / 32 * 34) as u64;
                let start = u64::from(token) * stride;
                let raw = self.source.load_range(
                    &tensor.locator().range(start..start + stride)?,
                    WeightAccess::Cyclic,
                )?;
                let mut output = vec![0.0; s.dim];
                cpu.dequantize(
                    &mut output,
                    &Weight {
                        bytes: raw.bytes(),
                        kind: tensor.kind(),
                        rows: 1,
                        cols: s.dim,
                    },
                )?;
                output
            };
            for layer in 0..s.layers {
                let name = |suffix: &str| format!("blk.{layer}.{suffix}.weight");
                let normal = self.norm(&name("attn_norm"), &hidden)?;
                let q = self.project(&name("attn_q"), &normal, s.heads * s.head)?;
                let k = self.project(&name("attn_k"), &normal, s.kv_heads * s.head)?;
                let v = self.project(&name("attn_v"), &normal, s.kv_heads * s.head)?;
                let q = self.norm_heads(&name("attn_q_norm"), &q)?;
                let k = self.norm_heads(&name("attn_k_norm"), &k)?;
                let attended = {
                    let (keys, values) = &mut self.caches[layer];
                    ensure!(
                        keys.len() == self.position * s.kv_heads * s.head
                            && values.len() == keys.len(),
                        "oracle KV position differs"
                    );
                    keys.extend(k);
                    values.extend(v);
                    let mut output = vec![0.0; q.len()];
                    cpu.incremental_causal_attention(
                        &mut output,
                        &q,
                        keys,
                        values,
                        IncrementalAttentionConfig {
                            n_query_positions: 1,
                            query_start: self.position,
                            n_kv_positions: self.position + 1,
                            n_heads: s.heads,
                            n_kv_heads: s.kv_heads,
                            head_dim: s.head,
                            scale: (s.head as f32).sqrt().recip(),
                        },
                    )?;
                    output
                };
                cpu.add_in_place(
                    &mut hidden,
                    &self.project(&name("attn_output"), &attended, s.dim)?,
                )?;
                let normal = self.norm(&name("ffn_norm"), &hidden)?;
                let mut gate = self.project(&name("ffn_gate"), &normal, s.ff)?;
                let up = self.project(&name("ffn_up"), &normal, s.ff)?;
                cpu.swiglu(&mut gate, &up, 0.0)?;
                cpu.add_in_place(&mut hidden, &self.project(&name("ffn_down"), &gate, s.dim)?)?;
                ensure!(
                    hidden.iter().all(|v| v.is_finite()),
                    "nonfinite CPU hidden at layer{layer}"
                );
            }
            let normal = self.norm("output_norm.weight", &hidden)?;
            let tensor = self
                .checkpoint
                .get("token_embd.weight")
                .context("missing tied oracle head")?;
            let stride = (s.dim / 32 * 34) as u64;
            let mut logits = vec![0.0; s.vocab];
            for first in (0..s.vocab).step_by(TILE_ROWS) {
                let rows = TILE_ROWS.min(s.vocab - first);
                let range = tensor
                    .locator()
                    .range(first as u64 * stride..(first + rows) as u64 * stride)?;
                let raw = self.source.load_range(&range, WeightAccess::Cyclic)?;
                cpu.matvec_with_threads(
                    &mut logits[first..first + rows],
                    &normal,
                    &Weight {
                        bytes: raw.bytes(),
                        kind: tensor.kind(),
                        rows,
                        cols: s.dim,
                    },
                    1,
                )?;
            }
            ensure!(
                logits.iter().all(|v| v.is_finite()),
                "CPU logits are nonfinite"
            );
            self.position += 1;
            Ok(logits)
        }
    }

    fn verify_step(before: NativeExecutionStats, after: NativeExecutionStats) -> Result<()> {
        ensure!(
            after.dispatch_count == before.dispatch_count + PHYSICAL_PER_STEP
                && after.dispatch_attempts == before.dispatch_attempts + PHYSICAL_PER_STEP
                && after.logical_submissions == before.logical_submissions + SUBMISSIONS_PER_STEP,
            "full-model step dispatch accounting differs"
        );
        // One transient lookup dummy is written while recording, before its submit.
        ensure!(
            after.host_write_bytes == before.host_write_bytes + 4
                && after.host_read_bytes == before.host_read_bytes,
            "step traffic differs from the one four-byte recording parameter"
        );
        ensure!(
            after.uploaded_weight_bytes == before.uploaded_weight_bytes
                && after.allocation_bytes == before.allocation_bytes
                && after.weight_live_bytes == before.weight_live_bytes
                && after.tensor_live_bytes == before.tensor_live_bytes
                && after.auxiliary_live_bytes == PERSISTENT_AUXILIARY
                && after.auxiliary_peak_bytes == u64::from(AUXILIARY_CAPACITY)
                && after.weight_peak_bytes == before.weight_peak_bytes
                && after.tensor_peak_bytes == before.tensor_peak_bytes
                && !after.poisoned,
            "step changed resident resources or poisoned queue"
        );
        Ok(())
    }

    fn run_model(options: &Options) -> Result<()> {
        let path = options.model.as_ref().context("missing model")?;
        check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
        let checkpoint = Gguf::open(path)?;
        check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
        let model = QwenModel::load(&checkpoint)?;
        ensure!(
            model.architecture() == QwenArchitecture::Qwen3,
            "dense qwen3 required"
        );
        let shape = Shape::from_model(&model);
        let plan = model.dense_model_plan(CAPACITY, TILE_ROWS)?;
        let config = inspect_shape(shape, plan.memory())?;
        inspect_checkpoint(&checkpoint, shape)?;
        ensure!(
            (plan.layers(), plan.dim(), plan.vocab(), plan.capacity())
                == (LAYERS, DIM, VOCAB, CAPACITY),
            "plan geometry differs"
        );
        let mut report = json!({
            "execution": "not_requested", "architecture": "qwen3", "model_layers": LAYERS,
            "input_width": DIM, "vocabulary_size": VOCAB, "capacity": CAPACITY,
            "requests": options.mode.requests(), "steps": options.mode.steps(), "steps_per_request": STEPS_PER_REQUEST,
            "packed_weight_bytes": PACKED, "source_weight_bytes": SOURCE_BYTES,
            "resident_tensor_bytes": TENSORS, "largest_staging_bytes": STAGING,
            "weight_arena_capacity_bytes": config.weight_capacity_bytes, "tensor_arena_capacity_bytes": config.tensor_capacity_bytes,
            "auxiliary_arena_capacity_bytes": AUXILIARY_CAPACITY, "persistent_auxiliary_bytes": PERSISTENT_AUXILIARY,
            "expected_hag_allocation_bytes": EXPECTED_ALLOCATION, "hag_budget_bytes": MODEL_BUDGET,
            "persistent_batches": 253, "expected_physical_dispatches_per_step": PHYSICAL_PER_STEP,
            "expected_logical_submissions_per_step": SUBMISSIONS_PER_STEP, "recording_host_write_bytes_per_step": 4,
            "required_hag_trace_submission": "0", "hag_trace_requirement_verified": options.mode != Mode::Inspect,
            "external_process_and_thermal_limits_required": true,
            "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
            "comparison_policy": "finite per-value tolerance and exact CPU greedy; sign/zero mismatches reported",
            "oracle": "independent immediate CPU operators, explicit per-layer KV, one-worker packed matrices and tiled head; all references before native open",
            "prompt_token_ids": PROMPTS, "third_token_policy": "CPU greedy after second position, required to match native greedy before use",
            "rmsnorm_epsilon": shape.eps, "rope_base": shape.theta, "weight_format": "Q8_0", "norm_format": "F32"
        });
        if options.mode == Mode::Inspect {
            println!("{}", serde_json::to_string_pretty(&report)?);
            return Ok(());
        }
        println!("RUN preregistered full-model bound: |error| <= {ABSOLUTE_TOLERANCE} + {RELATIVE_TOLERANCE} * |CPU|; finite and exact greedy; no automatic relaxation");
        let oracle_source = CountingSource::new(&checkpoint);
        let mut expected = Vec::new();
        let mut tokens = Vec::new();
        let mut oracle_greedy = Vec::new();
        for prompt in PROMPTS.iter().take(options.mode.requests()) {
            let mut oracle = Oracle::new(&checkpoint, &oracle_source, shape);
            let steps = if options.mode == Mode::FirstToken {
                1
            } else {
                STEPS_PER_REQUEST
            };
            for position in 0..steps {
                let token = if let Some(&token) = prompt.get(position) {
                    token
                } else {
                    *oracle_greedy.last().context("missing prior CPU greedy")?
                };
                let logits = oracle.step(token)?;
                tokens.push(token);
                oracle_greedy.push(greedy(&logits)?);
                expected.push(logits);
            }
        }
        let steps = options.mode.steps();
        ensure!(
            expected.len() == steps && tokens.len() == steps,
            "CPU fixture count differs"
        );
        let (oracle_reads, oracle_bytes, oracle_largest) = oracle_source.counts();
        ensure!(
            oracle_reads == steps as u64 * (PREPARATION_READS + 1)
                && oracle_bytes == steps as u64 * (SOURCE_BYTES + (DIM / 32 * 34) as u64)
                && oracle_largest == STAGING,
            "CPU source accounting differs"
        );
        let reference_bytes = steps * VOCAB * 4;
        ensure!(
            reference_bytes < 4 * 1024 * 1024,
            "reference logits exceed4MiB"
        );
        let budget = MemoryBudget::new(MODEL_BUDGET);
        let device = NativeExecutionDevice::open(
            options.device.as_ref().context("missing device")?,
            &budget,
            config,
        )?;
        let source = CountingSource::new(&checkpoint);
        let mut resident = plan.prepare(
            &device,
            &source,
            DenseModelLimits {
                packed_weight_bytes: PACKED,
                resident_tensor_bytes: TENSORS,
                largest_staging_bytes: STAGING,
            },
        )?;
        ensure!(
            source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING),
            "native preparation source accounting differs"
        );
        let initial = device.stats()?;
        ensure!(
            initial.allocation_bytes == EXPECTED_ALLOCATION
                && budget.current_bytes() == EXPECTED_ALLOCATION
                && initial.weight_live_bytes == PACKED
                && initial.tensor_live_bytes == TENSORS
                && initial.auxiliary_live_bytes == PERSISTENT_AUXILIARY
                && initial.auxiliary_peak_bytes == PERSISTENT_AUXILIARY
                && initial.uploaded_weight_bytes == PACKED
                && initial.host_write_bytes == INITIAL_HOST_WRITE_BYTES
                && initial.host_read_bytes == 0
                && initial.dispatch_attempts == 0
                && initial.dispatch_count == 0
                && initial.logical_submissions == 0
                && !initial.poisoned,
            "native initial resources/counters differ"
        );
        ensure!(
            resident.capacity() == CAPACITY
                && resident.visible_positions() == 0
                && !resident.is_poisoned(),
            "initial model state differs"
        );
        let mut maximum = 0.0_f64;
        let mut signs = 0u64;
        let mut zeros = 0u64;
        let mut selected = Vec::new();
        for step in 0..steps {
            let request = step / STEPS_PER_REQUEST;
            let position = step % STEPS_PER_REQUEST;
            if step > 0 && position == 0 {
                let before = device.stats()?;
                resident.reset()?;
                ensure!(
                    resident.visible_positions() == 0
                        && !resident.is_poisoned()
                        && device.stats()? == before,
                    "request reset changed counters or left visible state"
                );
            }
            if position == 2 {
                ensure!(
                    selected.last() == Some(&tokens[step]),
                    "autoregressive native choice differs before third input"
                );
            }
            ensure!(
                resident.visible_positions() == position,
                "model cursor differs before step"
            );
            let before = device.stats()?;
            resident.step(&device, tokens[step])?;
            verify_step(before, device.stats()?)?;
            ensure!(
                resident.visible_positions() == position + 1 && !resident.is_poisoned(),
                "model cursor did not commit exactly one step"
            );
            let mut actual = vec![f32::NAN; VOCAB];
            resident.read_logits(&device, &mut actual)?;
            let comparison = compare(&actual, &expected[step])
                .with_context(|| format!("request{request} position{position}"))?;
            maximum = maximum.max(comparison.maximum);
            signs += comparison.signs;
            zeros += comparison.zeros;
            let choice = greedy(&actual)?;
            ensure!(
                choice == oracle_greedy[step],
                "model greedy differs: native{choice}, CPU{}",
                oracle_greedy[step]
            );
            selected.push(choice);
            ensure!(
                budget.current_bytes() == EXPECTED_ALLOCATION,
                "model reuse changed HAG charge"
            );
            println!("PASS model request={request} position={position} token={} checked_values={VOCAB} greedy={choice} submission_tensor_transfer_bytes=0", tokens[step]);
        }
        let final_stats = device.stats()?;
        ensure!(
            final_stats.dispatch_count == steps as u64 * PHYSICAL_PER_STEP
                && final_stats.dispatch_attempts == steps as u64 * PHYSICAL_PER_STEP
                && final_stats.logical_submissions == steps as u64 * SUBMISSIONS_PER_STEP
                && final_stats.host_read_bytes == reference_bytes as u64
                && final_stats.host_write_bytes == initial.host_write_bytes + steps as u64 * 4
                && final_stats.uploaded_weight_bytes == PACKED
                && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING),
            "final execution/traffic/source counters differ"
        );
        resident.reset()?;
        ensure!(
            resident.visible_positions() == 0
                && !resident.is_poisoned()
                && device.stats()? == final_stats,
            "final reset differs"
        );
        drop(resident);
        let retired = device.stats()?;
        ensure!(
            retired.weight_live_bytes == 0
                && retired.tensor_live_bytes == 0
                && retired.auxiliary_live_bytes == 0
                && !retired.poisoned,
            "full-model resources did not retire"
        );
        drop(device);
        ensure!(
            budget.current_bytes() == 0,
            "HAG budget retained after model cleanup"
        );
        report["execution"] = json!(options.mode.execution());
        report["input_tokens"] = json!(tokens);
        report["greedy_tokens"] = json!(selected);
        report["greedy_matches"] = json!(steps);
        report["checked_values"] = json!(steps * VOCAB);
        report["physical_dispatches"] = json!(final_stats.dispatch_count);
        report["logical_submissions"] = json!(final_stats.logical_submissions);
        report["max_absolute_error"] = json!(maximum);
        report["logit_sign_mismatches"] = json!(signs);
        report["logit_zero_mismatches"] = json!(zeros);
        report["native_preparation_source_reads"] = json!(PREPARATION_READS);
        report["cpu_oracle_source_reads"] = json!(oracle_reads);
        report["cpu_oracle_source_bytes"] = json!(oracle_bytes);
        report["retained_reference_bytes"] = json!(reference_bytes);
        report["actual_hag_allocation_bytes"] = json!(EXPECTED_ALLOCATION);
        report["auxiliary_peak_bytes"] = json!(final_stats.auxiliary_peak_bytes);
        report["initial_host_write_bytes"] = json!(initial.host_write_bytes);
        report["uploaded_tensor_value_bytes"] = json!(UPLOADED_TENSOR_VALUE_BYTES);
        report["recording_host_write_bytes"] = json!(steps * 4);
        report["submission_tensor_transfer_bytes"] = json!(0);
        report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
        report["cache_visible_positions_after_reset"] = json!(0);
        report["budget_after"] = json!(0);
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }
    pub fn run() -> Result<()> {
        let options = parse(std::env::args_os().skip(1))?;
        if options.help || options.model.is_none() {
            println!("{USAGE}");
            return Ok(());
        }
        if options.mode.executes()
            || matches!(
                options.mode,
                Mode::OnDemand2048Reference
                    | Mode::Capacity16384Reference
                    | Mode::CatalogPrefill32Reference
            )
        {
            check_trace(std::env::var_os("HAG_TRACE_SUBMISSION").as_deref())?;
        }
        if matches!(options.mode, Mode::InspectContext | Mode::ContextBoundary) {
            context::run(&options)
        } else if matches!(
            options.mode,
            Mode::InspectOnDemand | Mode::OnDemandFirstToken | Mode::OnDemandContext
        ) {
            demand::run(&options)
        } else if matches!(
            options.mode,
            Mode::InspectOnDemand512 | Mode::OnDemand512FirstToken | Mode::OnDemand512Context
        ) {
            demand512::run(&options)
        } else if options.mode == Mode::InspectOnDemand2048 {
            demand2048::inspect(&options)
        } else if matches!(
            options.mode,
            Mode::OnDemand2048Reference | Mode::OnDemand2048FirstToken | Mode::OnDemand2048Context
        ) {
            demand2048::run(&options)
        } else if matches!(
            options.mode,
            Mode::Capacity16384Reference
                | Mode::Capacity16384Smoke
                | Mode::CatalogPrefill32Reference
                | Mode::CatalogPrefill32Smoke
        ) {
            capacity16384::run(&options)
        } else {
            run_model(&options)
        }
    }

    mod context {
        include!("hag_model_probe/context.rs");
    }

    mod demand {
        include!("hag_model_probe/demand.rs");
    }

    mod demand512 {
        include!("hag_model_probe/demand512.rs");
    }

    mod demand2048 {
        include!("hag_model_probe/demand2048.rs");
    }

    mod catalog_prefill32 {
        include!("hag_model_probe/catalog_prefill32.rs");
    }

    mod capacity16384 {
        include!("hag_model_probe/capacity16384.rs");
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        fn args(values: &[&str]) -> Vec<OsString> {
            values.iter().map(OsString::from).collect()
        }
        pub(super) fn shape() -> Shape {
            Shape {
                layers: 28,
                dim: 1024,
                heads: 16,
                kv_heads: 8,
                head: 128,
                ff: 3072,
                vocab: 151936,
                context: 40960,
                experts: 0,
                eps: 1e-6,
                theta: 1_000_000.0,
            }
        }
        fn memory() -> DenseModelMemory {
            DenseModelMemory {
                packed_weight_bytes: PACKED,
                resident_tensor_bytes: TENSORS,
                largest_staging_bytes: STAGING,
            }
        }
        #[test]
        fn metadata_and_exclusive_live_modes_require_device_model_and_trace() {
            assert_eq!(parse(args(&[])).unwrap().mode, Mode::Inspect);
            assert_eq!(parse(args(&["--model", "m"])).unwrap().mode, Mode::Inspect);
            assert_eq!(
                parse(args(&["--inspect-context", "--model", "m"]))
                    .unwrap()
                    .mode,
                Mode::InspectContext
            );
            assert_eq!(
                parse(args(&["--inspect-on-demand", "--model", "m"]))
                    .unwrap()
                    .mode,
                Mode::InspectOnDemand
            );
            assert_eq!(
                parse(args(&["--inspect-on-demand512", "--model", "m"]))
                    .unwrap()
                    .mode,
                Mode::InspectOnDemand512
            );
            for (flag, mode) in [
                ("--first-token", Mode::FirstToken),
                ("--execute", Mode::Execute),
                ("--context-boundary", Mode::ContextBoundary),
                ("--on-demand-first-token", Mode::OnDemandFirstToken),
                ("--on-demand-context", Mode::OnDemandContext),
                ("--on-demand512-first-token", Mode::OnDemand512FirstToken),
                ("--on-demand512-context", Mode::OnDemand512Context),
            ] {
                assert_eq!(
                    parse(args(&[flag, "--model", "m", "/dev/dri/renderD128"]))
                        .unwrap()
                        .mode,
                    mode
                );
                assert!(parse(args(&[flag])).is_err());
                assert!(parse(args(&[flag, "/dev/dri/renderD128"])).is_err());
                assert!(parse(args(&[flag, flag, "--model", "m", "/dev/dri/renderD128"])).is_err());
            }
            for bad in [
                vec!["/dev/dri/renderD128"],
                vec!["--execute", "--first-token", "--model", "m", "node"],
                vec![
                    "--inspect-context",
                    "--context-boundary",
                    "--model",
                    "m",
                    "node",
                ],
                vec!["--inspect-context", "--model", "m", "node"],
                vec!["--inspect-on-demand", "--model", "m", "node"],
                vec!["--inspect-on-demand512", "--model", "m", "node"],
                vec![
                    "--on-demand512-context",
                    "--on-demand-context",
                    "--model",
                    "m",
                    "node",
                ],
                vec![
                    "--on-demand512-first-token",
                    "--execute",
                    "--model",
                    "m",
                    "node",
                ],
                vec![
                    "--on-demand-context",
                    "--context-boundary",
                    "--model",
                    "m",
                    "node",
                ],
                vec![
                    "--on-demand-first-token",
                    "--execute",
                    "--model",
                    "m",
                    "node",
                ],
                vec!["--model", "m", "--model", "m"],
                vec!["--model"],
                vec!["--layer", "0"],
            ] {
                assert!(parse(args(&bad)).is_err());
            }
            for value in [
                None,
                Some(OsStr::new("1")),
                Some(OsStr::new("00")),
                Some(OsStr::new("")),
            ] {
                assert!(check_trace(value).is_err());
            }
            check_trace(Some(OsStr::new("0"))).unwrap();
        }
        #[test]
        fn catalog_prefix_requires_exclusive_tokens_and_separate_phase_inputs() {
            let cpu = [
                "--catalog-prefill32-reference",
                "--model",
                "m",
                "--prompt-tokens",
                "t",
                "--reference-out",
                "r",
                "--input-identity",
                "i",
            ];
            let native = [
                "--catalog-prefill32-smoke",
                "--model",
                "m",
                "--prompt-tokens",
                "t",
                "--reference",
                "r",
                "--input-identity",
                "i",
                "/dev/dri/renderD128",
            ];
            assert_eq!(
                parse(args(&cpu)).unwrap().mode,
                Mode::CatalogPrefill32Reference
            );
            assert_eq!(
                parse(args(&native)).unwrap().mode,
                Mode::CatalogPrefill32Smoke
            );
            assert_eq!(Mode::CatalogPrefill32Reference.steps(), 32);
            assert_eq!(Mode::CatalogPrefill32Smoke.steps(), 33);
            for mut bad in [cpu.to_vec(), native.to_vec()] {
                bad.drain(3..5);
                assert!(parse(args(&bad)).is_err());
            }
            for bad in [
                vec!["--model", "m", "--prompt-tokens", "t"],
                vec![
                    "--capacity16384-reference",
                    "--model",
                    "m",
                    "--prompt-tokens",
                    "t",
                    "--reference-out",
                    "r",
                    "--input-identity",
                    "i",
                ],
                vec![
                    "--catalog-prefill32-reference",
                    "--model",
                    "m",
                    "--prompt-tokens",
                    "t",
                    "--reference",
                    "r",
                    "--input-identity",
                    "i",
                ],
                vec![
                    "--catalog-prefill32-smoke",
                    "--model",
                    "m",
                    "--prompt-tokens",
                    "t",
                    "--reference-out",
                    "r",
                    "--input-identity",
                    "i",
                    "/dev/dri/renderD128",
                ],
                vec![
                    "--catalog-prefill32-reference",
                    "--model",
                    "m",
                    "--prompt-tokens",
                    "t",
                    "--prompt-tokens",
                    "t",
                    "--reference-out",
                    "r",
                    "--input-identity",
                    "i",
                ],
            ] {
                assert!(parse(args(&bad)).is_err());
            }
        }

        #[test]
        fn exact_shape_and_logical_or_physical_cost_drift_refuse_without_allocations() {
            let config = inspect_shape(shape(), memory()).unwrap();
            assert_eq!(config.allocation_bytes().unwrap(), 638_861_312);
            for bad in [
                Shape {
                    layers: 27,
                    ..shape()
                },
                Shape {
                    dim: 2048,
                    ..shape()
                },
                Shape {
                    heads: 8,
                    ..shape()
                },
                Shape {
                    kv_heads: 16,
                    ..shape()
                },
                Shape {
                    head: 64,
                    ..shape()
                },
                Shape {
                    ff: 4096,
                    ..shape()
                },
                Shape {
                    vocab: 151937,
                    ..shape()
                },
                Shape {
                    experts: 1,
                    ..shape()
                },
                Shape {
                    context: 7,
                    ..shape()
                },
                Shape {
                    eps: f32::NAN,
                    ..shape()
                },
                Shape {
                    eps: -1.0,
                    ..shape()
                },
                Shape {
                    theta: 10000.0,
                    ..shape()
                },
            ] {
                assert!(inspect_shape(bad, memory()).is_err());
            }
            for bad in [
                DenseModelMemory {
                    packed_weight_bytes: PACKED + 1,
                    ..memory()
                },
                DenseModelMemory {
                    resident_tensor_bytes: TENSORS - 4,
                    ..memory()
                },
                DenseModelMemory {
                    largest_staging_bytes: STAGING + 1,
                    ..memory()
                },
            ] {
                assert!(inspect_shape(shape(), bad).is_err());
            }
        }
        #[test]
        fn independent_resource_dispatch_and_source_formulas_pin_the_whole_gate() {
            let q = 16 * 128_u64;
            let kv = 8 * 128_u64;
            let d = 1024_u64;
            let f = 3072_u64;
            let c = 8_u64;
            let attention = (q * d + 2 * kv * d + d * q) / 32 * 34;
            let ffn = 3 * d * f / 32 * 34;
            let vocab = 151936 * d / 32 * 34;
            assert_eq!(28 * (attention + ffn) + vocab, PACKED);
            assert_eq!(PACKED + 28 * (2 * d + 2 * 128) * 4 + d * 4, SOURCE_BYTES);
            let attention_tensors = (2 * d + 2 * 128 + 3 * q + kv + 2 * c * kv + c * 128) * 4;
            let ffn_tensors = (2 * d + 2 * f) * 4;
            assert_eq!(
                28 * (attention_tensors + ffn_tensors) + 3 * d * 4 + 151936 * 4,
                TENSORS
            );
            let attention_aux = (1..=c).map(|p| (d * 4).max(12 * p + 4) + 12).sum::<u64>();
            assert_eq!(
                28 * (attention_aux + (d * 4 + 12)) + (d * 4 + 4),
                PERSISTENT_AUXILIARY
            );
            assert_eq!(PERSISTENT_AUXILIARY + 4, u64::from(AUXILIARY_CAPACITY));
            assert_eq!(1 + 28 * (127 + 7) + 40, PHYSICAL_PER_STEP);
            assert_eq!(1 + 28 * 2 + 1, SUBMISSIONS_PER_STEP);
            assert_eq!(28 * 11 + 39, PREPARATION_READS);
            assert_eq!(6 * (PREPARATION_READS + 1), 2088);
            assert_eq!(6 * (SOURCE_BYTES + 1088), 3_800_979_840);
            assert_eq!(6 * VOCAB * 4, 3_646_464);
            assert_eq!(PACKED + TENSORS + PERSISTENT_AUXILIARY, 638_820_820);
            let norms = 28 * (2 * d + 2 * 128) * 4 + d * 4;
            let rope = 28 * c * 128 * 4;
            assert_eq!(norms + rope, UPLOADED_TENSOR_VALUE_BYTES);
            assert_eq!(INITIAL_HOST_WRITE_BYTES, 639_197_652);
        }
        #[test]
        fn logit_policy_refuses_nonfinite_and_excess_error_without_hiding_classification() {
            let c = compare(&[-1e-8, 0.0, -0.0], &[1e-8, 1e-8, 0.0]).unwrap();
            assert_eq!((c.signs, c.zeros), (2, 1));
            for (a, b) in [(f32::NAN, 0.0), (1.0, f32::INFINITY), (1.01, 1.0)] {
                assert!(compare(&[a], &[b]).is_err());
            }
            assert!(compare(&[], &[]).is_err());
            assert!(compare(&[1.0], &[]).is_err());
            assert_eq!(greedy(&[-1.0, 3.0, 3.0]).unwrap(), 1);
            assert_eq!(greedy(&[-0.0, 0.0]).unwrap(), 0);
            assert!(greedy(&[]).is_err());
            assert!(greedy(&[1.0, f32::NAN]).is_err());
        }
        #[test]
        fn per_step_admission_detects_extra_traffic_missing_dispatch_and_residency_growth() {
            let before = NativeExecutionStats {
                allocation_bytes: EXPECTED_ALLOCATION,
                weight_live_bytes: PACKED,
                tensor_live_bytes: TENSORS,
                auxiliary_live_bytes: PERSISTENT_AUXILIARY,
                uploaded_weight_bytes: PACKED,
                auxiliary_peak_bytes: PERSISTENT_AUXILIARY,
                ..Default::default()
            };
            let after = NativeExecutionStats {
                dispatch_count: PHYSICAL_PER_STEP,
                dispatch_attempts: PHYSICAL_PER_STEP,
                logical_submissions: SUBMISSIONS_PER_STEP,
                host_write_bytes: 4,
                auxiliary_peak_bytes: u64::from(AUXILIARY_CAPACITY),
                ..before
            };
            verify_step(before, after).unwrap();
            for bad in [
                NativeExecutionStats {
                    host_read_bytes: 4,
                    ..after
                },
                NativeExecutionStats {
                    host_write_bytes: 8,
                    ..after
                },
                NativeExecutionStats {
                    dispatch_count: PHYSICAL_PER_STEP - 1,
                    ..after
                },
                NativeExecutionStats {
                    logical_submissions: 59,
                    ..after
                },
                NativeExecutionStats {
                    uploaded_weight_bytes: PACKED + 4,
                    ..after
                },
                NativeExecutionStats {
                    tensor_live_bytes: TENSORS + 4,
                    ..after
                },
                NativeExecutionStats {
                    auxiliary_live_bytes: u64::from(AUXILIARY_CAPACITY),
                    ..after
                },
                NativeExecutionStats {
                    poisoned: true,
                    ..after
                },
            ] {
                assert!(verify_step(before, bad).is_err());
            }
        }

        // Two dense layers and GQA on a tiny real GGUF fixture make the independent
        // immediate oracle comparable to the public production full-prefix path.
        // Only small payloads are constructed; no model download or native access.
        struct Fixture(PathBuf);
        impl Drop for Fixture {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
        fn tiny_checkpoint() -> Fixture {
            tiny_checkpoint_with_context(8)
        }
        fn tiny_checkpoint_with_context(context: u32) -> Fixture {
            fn string(bytes: &mut Vec<u8>, s: &str) {
                bytes.extend((s.len() as u64).to_le_bytes());
                bytes.extend(s.as_bytes());
            }
            let mut tensors: Vec<(String, Vec<u64>, u32)> = vec![
                ("token_embd.weight".into(), vec![32, 9], 8),
                ("output_norm.weight".into(), vec![32], 0),
            ];
            for layer in 0..2 {
                for name in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"] {
                    tensors.push((format!("blk.{layer}.{name}.weight"), vec![32], 0));
                }
                for (name, cols, rows) in [
                    ("attn_q", 32, 64),
                    ("attn_k", 32, 32),
                    ("attn_v", 32, 32),
                    ("attn_output", 64, 32),
                    ("ffn_gate", 32, 64),
                    ("ffn_up", 32, 64),
                    ("ffn_down", 64, 32),
                ] {
                    tensors.push((format!("blk.{layer}.{name}.weight"), vec![cols, rows], 8));
                }
            }
            let metadata = [
                ("block_count", 2_u32),
                ("embedding_length", 32),
                ("attention.head_count", 2),
                ("attention.head_count_kv", 1),
                ("attention.key_length", 32),
                ("attention.value_length", 32),
                ("feed_forward_length", 64),
                ("context_length", context),
            ];
            let mut header = b"GGUF".to_vec();
            header.extend(3_u32.to_le_bytes());
            header.extend((tensors.len() as u64).to_le_bytes());
            header.extend((metadata.len() as u64 + 3).to_le_bytes());
            string(&mut header, "general.architecture");
            header.extend(8_u32.to_le_bytes());
            string(&mut header, "qwen3");
            for (key, value) in metadata {
                string(&mut header, &format!("qwen3.{key}"));
                header.extend(4_u32.to_le_bytes());
                header.extend(value.to_le_bytes());
            }
            for (key, value) in [
                ("attention.layer_norm_rms_epsilon", 1e-6_f32),
                ("rope.freq_base", 10000.0),
            ] {
                string(&mut header, &format!("qwen3.{key}"));
                header.extend(6_u32.to_le_bytes());
                header.extend(value.to_le_bytes());
            }
            let mut payload = Vec::new();
            for (tensor_index, (name, dims, kind)) in tensors.iter().enumerate() {
                string(&mut header, name);
                header.extend((dims.len() as u32).to_le_bytes());
                for dim in dims {
                    header.extend(dim.to_le_bytes());
                }
                header.extend(kind.to_le_bytes());
                header.extend((payload.len() as u64).to_le_bytes());
                let elements = dims.iter().product::<u64>() as usize;
                if *kind == 0 {
                    for i in 0..elements {
                        payload.extend(
                            (0.75_f32 + ((i + tensor_index) % 5) as f32 / 16.0).to_le_bytes(),
                        );
                    }
                } else {
                    for block in 0..elements / 32 {
                        payload.extend(0x2800_u16.to_le_bytes()); // F16 scale1/32
                        payload.extend(
                            (0..32).map(|i| {
                                (((i * 3 + block * 5 + tensor_index) % 11) as i8 - 5) as u8
                            }),
                        );
                    }
                }
                payload.resize(payload.len().next_multiple_of(32), 0);
            }
            header.resize(header.len().next_multiple_of(32), 0);
            header.extend(payload);
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let file = std::env::temp_dir().join(format!(
                "beholdd-model-oracle-{}-{}.gguf",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::write(&file, header).unwrap();
            Fixture(file)
        }
        #[test]
        fn independent_cached_oracle_matches_production_full_prefixes_and_fresh_requests() {
            let fixture = tiny_checkpoint();
            let checkpoint = Gguf::open(&fixture.0).unwrap();
            let model = QwenModel::load(&checkpoint).unwrap();
            let shape = Shape::from_model(&model);
            let source = CountingSource::new(&checkpoint);
            let production = ResidentSource::new(&checkpoint);
            for prompt in PROMPTS {
                let mut oracle = Oracle::new(&checkpoint, &source, shape);
                let mut tokens = Vec::new();
                let mut previous = 0;
                for position in 0..3 {
                    let token = if position < 2 {
                        prompt[position]
                    } else {
                        previous
                    };
                    tokens.push(token);
                    let actual = oracle.step(token).unwrap();
                    let expected = model.forward(&Cpu::new(), &production, &tokens).unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected
                            .logits
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        "prefix={tokens:?}"
                    );
                    previous = greedy(&actual).unwrap();
                    assert_eq!(previous, expected.greedy_token().unwrap());
                }
            }
            assert_eq!(source.counts().0, 6 * (1 + 2 * 11 + 2));
            assert!(source.counts().2 < 8192);
        }

        #[test]
        fn context_oracle_reaches_c128_and_refuses_one_more_before_reads() {
            let fixture = tiny_checkpoint_with_context(128);
            let checkpoint = Gguf::open(&fixture.0).unwrap();
            let model = QwenModel::load(&checkpoint).unwrap();
            let shape = Shape::from_model(&model);
            let source = CountingSource::new(&checkpoint);
            let production = ResidentSource::new(&checkpoint);
            let mut oracle = Oracle::with_capacity(&checkpoint, &source, shape, 128);
            let mut tokens = Vec::new();
            for position in 0..128 {
                let token = (position % shape.vocab) as u32;
                tokens.push(token);
                let actual = oracle.step(token).unwrap();
                if [0, 31, 63, 64, 127].contains(&position) {
                    let expected = model.forward(&Cpu::new(), &production, &tokens).unwrap();
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected
                            .logits
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
            }
            let before = source.counts();
            assert_eq!(oracle.position, 128);
            assert!(oracle.step(1).is_err());
            assert_eq!(source.counts(), before);
        }

        #[test]
        fn on_demand512_oracle_crosses_legacy_limit_and_refuses_before_reads() {
            let fixture = tiny_checkpoint_with_context(512);
            let checkpoint = Gguf::open(&fixture.0).unwrap();
            let model = QwenModel::load(&checkpoint).unwrap();
            let shape = Shape::from_model(&model);
            let source = CountingSource::new(&checkpoint);
            let production = ResidentSource::new(&checkpoint);
            let mut oracle = Oracle::with_capacity(&checkpoint, &source, shape, 512);
            let mut tokens = Vec::new();
            for position in 0..512 {
                let token = (position % shape.vocab) as u32;
                tokens.push(token);
                let actual = oracle.step(token).unwrap();
                if [0, 255, 256, 511].contains(&position) {
                    let expected = model.forward(&Cpu::new(), &production, &tokens).unwrap();
                    assert_eq!(
                        actual
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        expected
                            .logits
                            .iter()
                            .map(|value| value.to_bits())
                            .collect::<Vec<_>>(),
                        "position {position}",
                    );
                    assert_eq!(greedy(&actual).unwrap(), expected.greedy_token().unwrap());
                }
            }
            let before = source.counts();
            assert_eq!(oracle.position, 512);
            assert!(oracle.step(1).is_err());
            assert_eq!(source.counts(), before);
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    proof::run()
}
#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native HAG model qualification requires Linux; no work submitted.");
    std::process::exit(1);
}
