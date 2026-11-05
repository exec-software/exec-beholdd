//! Bounded actual OpenMythos scalar-state/B4 comparison and observed timing.
//! External cgroup, thermal, crash-watch, deadline and frozen provenance required.
//! No serving, independent accuracy or sustained performance qualification.
#[cfg(target_os = "linux")]
mod probe {
    use anyhow::{ensure, Context, Result};
    use beholdd_backend::hag::native_execution::{MemoryBudget, NativeExecutionProfile};
    use beholdd_format::{Gguf, RangeReader};
    use beholdd_model::qwen35::config::{Attention, Binding};
    use beholdd_model::qwen35::recorded_model::{
        CheckedChunkRequirements, CheckedRequirements, ModelLimits, ModelPhase, ModelStateSnapshot,
        RecordedQwen35Plan, ResidentQwen35CheckedModel,
    };
    use beholdd_model::qwen35::recorded_weights::PackedRowPlan;
    use beholdd_model::qwen35::weights::ReadLimits;
    use serde_json::{json, Value};
    use std::{
        collections::BTreeMap,
        env,
        fs::{self, OpenOptions},
        io::Write,
        path::{Path, PathBuf},
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    const MODEL_BYTES: u64 = 16_547_399_872;
    const MODEL_SHA256: &str = "d3c3398ba8b512aa59074efcaa34d7dcabed7381b68a05ae6bcce77e14d7db45";
    const VOCAB: usize = 248_320;
    const PROMPT: [u32; 5] = [7, 42, 7, 42, 7];
    const CAPACITY: usize = 8;
    const CHUNK: usize = 4;
    const WORK_SECONDS: u64 = 180;
    const CLEANUP_SECONDS: u64 = 60;
    const PROCESS_BYTES: u64 = 20 * GIB;
    const HEADROOM_BYTES: u64 = GIB;
    const DEVICE_BYTES: u64 = 512 * MIB;
    const HARNESS_BYTES: u64 = 16 * MIB;
    const MAX_RECEIPT: usize = 128 << 10;

    struct Args {
        native: bool,
        model: PathBuf,
        render: PathBuf,
        output: PathBuf,
    }
    fn args() -> Result<Args> {
        let mut values = BTreeMap::new();
        let mut it = env::args().skip(1);
        while let Some(key) = it.next() {
            ensure!(
                ["--mode", "--model", "--render-node", "--output-dir"].contains(&key.as_str()),
                "unknown option {key}"
            );
            let value = it.next().context("missing option value")?;
            ensure!(values.insert(key, value).is_none(), "duplicate option");
        }
        let mut take = |key: &str| {
            values
                .remove(key)
                .with_context(|| format!("required {key}"))
        };
        let mode = take("--mode")?;
        ensure!(
            mode == "preflight" || mode == "native",
            "mode must be preflight/native"
        );
        let model = PathBuf::from(take("--model")?);
        let render = PathBuf::from(take("--render-node")?);
        let output = PathBuf::from(take("--output-dir")?);
        ensure!(
            model.is_absolute() && output.is_absolute(),
            "absolute model/output paths required"
        );
        let suffix = render
            .file_name()
            .and_then(|v| v.to_str())
            .and_then(|v| v.strip_prefix("renderD"))
            .unwrap_or("");
        ensure!(
            render.parent() == Some(Path::new("/dev/dri"))
                && !suffix.is_empty()
                && suffix.bytes().all(|v| v.is_ascii_digit()),
            "explicit render node required"
        );
        Ok(Args {
            native: mode == "native",
            model,
            render,
            output,
        })
    }
    fn deadline(end: Instant) -> Result<()> {
        ensure!(Instant::now() < end, "cooperative deadline exceeded");
        Ok(())
    }
    fn vector() -> Result<Vec<f32>> {
        let mut values = Vec::new();
        values.try_reserve_exact(VOCAB)?;
        values.resize(VOCAB, 0.0);
        Ok(values)
    }
    fn greedy(values: &[f32]) -> Result<u32> {
        ensure!(
            values.len() == VOCAB && values.iter().all(|v| v.is_finite()),
            "invalid full logits"
        );
        let mut best = 0;
        for index in 1..values.len() {
            if values[index] > values[best] {
                best = index;
            }
        }
        Ok(best as u32)
    }
    fn exact(actual: &[f32], expected: &[f32]) -> Result<()> {
        ensure!(
            actual.len() == VOCAB && expected.len() == VOCAB,
            "complete vocabulary required"
        );
        for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
            ensure!(
                a.is_finite() && b.is_finite() && a.to_bits() == b.to_bits(),
                "sequence logits differ at coordinate {index}: {a} != {b}"
            );
        }
        ensure!(greedy(actual)? == greedy(expected)?, "greedy token differs");
        Ok(())
    }
    fn output(path: &Path, bytes: &[u8]) -> Result<()> {
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(())
    }
    fn logits_file(path: &Path, values: &[f32]) -> Result<()> {
        greedy(values)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(VOCAB * 4)?;
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        output(path, &bytes)
    }
    fn state(s: ModelStateSnapshot) -> Value {
        json!({"phase":format!("{:?}",s.phase),"position":s.committed_positions,
            "capacity":s.capacity,"poisoned":s.poisoned,"process_healthy":s.process_healthy,
            "source_attempts":s.source.attempts,"source_bytes":s.source.returned_bytes,
            "packed_reads":s.packed.read_attempts,"packed_bytes":s.packed.returned_read_bytes,
            "packed_upload_bytes":s.packed.uploaded_bytes,"packed_submissions":s.packed.submission_attempts,
            "completed_tiles":s.packed.completed_tiles,"reset_bytes":s.reset_write_bytes,
            "logits_read_bytes":s.logits_read_bytes,"budget_bytes":s.budget_current_bytes,
            "expected_budget_bytes":s.budget_expected_bytes,"native":{
                "dispatches":s.native.dispatch_count,"dispatch_attempts":s.native.dispatch_attempts,
                "logical_submissions":s.native.logical_submissions,"host_read_bytes":s.native.host_read_bytes,
                "host_write_bytes":s.native.host_write_bytes,"uploaded_weight_bytes":s.native.uploaded_weight_bytes,
                "weight_live_bytes":s.native.weight_live_bytes,"auxiliary_live_bytes":s.native.auxiliary_live_bytes,
                "tensor_live_bytes":s.native.tensor_live_bytes,"poisoned":s.native.poisoned}})
    }
    fn live(s: ModelStateSnapshot, r: CheckedRequirements, position: usize) -> Result<()> {
        ensure!(
            s.phase == ModelPhase::Ready
                && s.committed_positions == position
                && s.capacity == CAPACITY
                && !s.poisoned
                && s.process_healthy
                && !s.native.poisoned
                && s.native.weight_live_bytes == 0
                && s.native.auxiliary_live_bytes == 0
                && s.native.tensor_live_bytes == r.resident_tensor_bytes
                && s.budget_current_bytes == r.device_charge_bytes
                && s.budget_expected_bytes == r.device_charge_bytes,
            "state/lifetime invariant differs"
        );
        Ok(())
    }
    fn delta(after: u64, before: u64) -> Result<u64> {
        after.checked_sub(before).context("counter regressed")
    }

    #[derive(Clone, Copy, Default)]
    struct Traffic {
        tiles: u64,
        packed_submissions: u64,
        chunk_applications: u64,
        bytes: u64,
        dispatches: u64,
        submissions: u64,
        status_bytes: u64,
        host_writes: u64,
    }
    impl Traffic {
        fn times(self, n: u64) -> Self {
            // Fixed geometry/extent and exactly25 positions bound these products.
            Self {
                tiles: self.tiles * n,
                packed_submissions: self.packed_submissions * n,
                chunk_applications: self.chunk_applications * n,
                bytes: self.bytes * n,
                dispatches: self.dispatches * n,
                submissions: self.submissions * n,
                status_bytes: self.status_bytes * n,
                host_writes: self.host_writes * n,
            }
        }
        fn plus(self, other: Self) -> Self {
            Self {
                tiles: self.tiles + other.tiles,
                packed_submissions: self.packed_submissions + other.packed_submissions,
                chunk_applications: self.chunk_applications + other.chunk_applications,
                bytes: self.bytes + other.bytes,
                dispatches: self.dispatches + other.dispatches,
                submissions: self.submissions + other.submissions,
                status_bytes: self.status_bytes + other.status_bytes,
                host_writes: self.host_writes + other.host_writes,
            }
        }
        fn json(self) -> Value {
            json!({"source_tiles":self.tiles,"packed_submissions":self.packed_submissions,
                "ffn_chunk_applications":self.chunk_applications,"packed_bytes":self.bytes,
                "dispatches":self.dispatches,"logical_submissions":self.submissions,
                "status_read_bytes":self.status_bytes,"host_write_bytes":self.host_writes})
        }
    }
    #[derive(Clone, Copy)]
    struct Forecast {
        full: Traffic,
        advance: Traffic,
        chunk: Traffic,
        full_attention_layers: u64,
    }
    impl Forecast {
        fn step(self, position: usize, full: bool) -> Traffic {
            let mut value = if full { self.full } else { self.advance };
            // Each full-attention record allocates 12*(position+1)+4 bytes
            // of scratch and one 4-byte scale. Base traffic already counts p=0.
            value.host_writes += 12 * self.full_attention_layers * position as u64;
            value
        }
        fn chunk(self, position: usize) -> Traffic {
            let mut value = self.chunk;
            value.host_writes += 12 * self.full_attention_layers * CHUNK as u64 * position as u64;
            value
        }
        fn prefix(self, mode: Mode) -> Traffic {
            match mode {
                Mode::Scalar => {
                    (0..CHUNK).fold(Traffic::default(), |sum, p| sum.plus(self.step(p, false)))
                }
                Mode::Chunk => self.chunk(0),
            }
        }
        fn sequence(self, mode: Mode) -> Traffic {
            self.prefix(mode)
                .plus(self.step(4, true))
                .plus(self.step(5, true))
        }
    }
    fn accounting(
        before: ModelStateSnapshot,
        after: ModelStateSnapshot,
        before_apps: u64,
        after_apps: u64,
        expected: Traffic,
        read_bytes: u64,
        reset_bytes: u64,
    ) -> Result<()> {
        ensure!(
            delta(after.packed.read_attempts, before.packed.read_attempts)? == expected.tiles
                && delta(after.packed.completed_tiles, before.packed.completed_tiles)?
                    == expected.tiles
                && delta(
                    after.packed.submission_attempts,
                    before.packed.submission_attempts
                )? == expected.packed_submissions
                && delta(
                    after.packed.returned_read_bytes,
                    before.packed.returned_read_bytes
                )? == expected.bytes
                && delta(after.packed.uploaded_bytes, before.packed.uploaded_bytes)?
                    == expected.bytes
                && delta(after.source.attempts, before.source.attempts)? == expected.tiles
                && delta(after.source.returned_bytes, before.source.returned_bytes)?
                    == expected.bytes
                && delta(
                    after.native.uploaded_weight_bytes,
                    before.native.uploaded_weight_bytes
                )? == expected.bytes
                && delta(after.native.dispatch_count, before.native.dispatch_count)?
                    == expected.dispatches
                && delta(
                    after.native.dispatch_attempts,
                    before.native.dispatch_attempts
                )? == expected.dispatches
                && delta(
                    after.native.logical_submissions,
                    before.native.logical_submissions
                )? == expected.submissions
                && delta(after.native.host_read_bytes, before.native.host_read_bytes)?
                    == expected.status_bytes + read_bytes
                && delta(after.logits_read_bytes, before.logits_read_bytes)? == read_bytes
                && delta(
                    after.native.host_write_bytes,
                    before.native.host_write_bytes
                )? == expected.host_writes + reset_bytes
                && delta(after.reset_write_bytes, before.reset_write_bytes)? == reset_bytes
                && delta(after_apps, before_apps)? == expected.chunk_applications,
            "source-derived exact chunk/source/native/host traffic differs"
        );
        Ok(())
    }
    fn elapsed_ns(start: Instant) -> Result<u64> {
        let value = u64::try_from(start.elapsed().as_nanos()).context("duration overflow")?;
        ensure!(value > 0, "zero measured duration");
        Ok(value)
    }
    fn phase(
        before: ModelStateSnapshot,
        after: ModelStateSnapshot,
        before_apps: u64,
        after_apps: u64,
        expected: Traffic,
        read_bytes: u64,
        reset_bytes: u64,
        name: &str,
        duration_ns: u64,
    ) -> Result<Value> {
        accounting(
            before,
            after,
            before_apps,
            after_apps,
            expected,
            read_bytes,
            reset_bytes,
        )?;
        let observed = Traffic {
            tiles: delta(after.packed.completed_tiles, before.packed.completed_tiles)?,
            packed_submissions: delta(
                after.packed.submission_attempts,
                before.packed.submission_attempts,
            )?,
            chunk_applications: delta(after_apps, before_apps)?,
            bytes: delta(after.packed.uploaded_bytes, before.packed.uploaded_bytes)?,
            dispatches: delta(after.native.dispatch_count, before.native.dispatch_count)?,
            submissions: delta(
                after.native.logical_submissions,
                before.native.logical_submissions,
            )?,
            status_bytes: delta(
                delta(after.native.host_read_bytes, before.native.host_read_bytes)?,
                read_bytes,
            )?,
            host_writes: delta(
                delta(
                    after.native.host_write_bytes,
                    before.native.host_write_bytes,
                )?,
                reset_bytes,
            )?,
        };
        Ok(
            json!({"name":name,"duration_ns":duration_ns,"expected":expected.json(),"observed":observed.json(),
            "logits_read_bytes":delta(after.logits_read_bytes,before.logits_read_bytes)?,
            "reset_write_bytes":delta(after.reset_write_bytes,before.reset_write_bytes)?,
            "host_read_bytes":delta(after.native.host_read_bytes,before.native.host_read_bytes)?,
            "host_write_bytes":delta(after.native.host_write_bytes,before.native.host_write_bytes)?}),
        )
    }
    fn reset(
        owner: &mut ResidentQwen35CheckedModel,
        r: CheckedRequirements,
        end: Instant,
        label: &str,
        evidence: &mut Vec<Value>,
    ) -> Result<()> {
        deadline(end)?;
        let before = owner.state()?;
        let apps = owner.chunk_packed_stats().completed_applications;
        let cleanup = Instant::now() + Duration::from_secs(CLEANUP_SECONDS);
        let start = Instant::now();
        owner.reset_checked(&mut || deadline(cleanup))?;
        let duration = elapsed_ns(start)?;
        let after = owner.state()?;
        live(after, r, 0)?;
        evidence.push(json!({"name":label,"phase":phase(before,after,apps,
            owner.chunk_packed_stats().completed_applications,Traffic::default(),0,r.reset_write_bytes,
            "reset",duration)?,"state":state(after)}));
        deadline(end)
    }
    #[derive(Clone, Copy)]
    enum Mode {
        Scalar,
        Chunk,
    }
    impl Mode {
        fn label(self) -> &'static str {
            match self {
                Self::Scalar => "scalar-state-advance",
                Self::Chunk => "b4",
            }
        }
    }
    struct Sequence {
        prefill: Vec<f32>,
        decode: Vec<f32>,
        selected: u32,
        next: u32,
        prefix_ns: u64,
    }
    fn sequence(
        owner: &mut ResidentQwen35CheckedModel,
        mode: Mode,
        label: &str,
        reference: Option<&Sequence>,
        r: CheckedRequirements,
        forecast: Forecast,
        end: Instant,
        evidence: &mut Vec<Value>,
    ) -> Result<Sequence> {
        deadline(end)?;
        let before = owner.state()?;
        let before_apps = owner.chunk_packed_stats().completed_applications;
        live(before, r, 0)?;
        let mut prefill = vector()?;
        let mut decode = vector()?;
        let mut phases = Vec::new();
        let start = Instant::now();
        match mode {
            Mode::Scalar => {
                for &token in &PROMPT[..CHUNK] {
                    owner.advance_checked(token, &mut || deadline(end))?;
                }
            }
            Mode::Chunk => {
                owner.advance_chunk_checked(&PROMPT[..CHUNK], &mut || deadline(end))?;
            }
        }
        let prefix_ns = elapsed_ns(start)?;
        let prefix_state = owner.state()?;
        let apps = owner.chunk_packed_stats().completed_applications;
        live(prefix_state, r, 4)?;
        phases.push(phase(
            before,
            prefix_state,
            before_apps,
            apps,
            forecast.prefix(mode),
            0,
            0,
            "prefix4",
            prefix_ns,
        )?);

        let start = Instant::now();
        let ticket = owner.step_checked(PROMPT[4], &mut || deadline(end))?;
        let head_ns = elapsed_ns(start)?;
        let head_state = owner.state()?;
        live(head_state, r, 5)?;
        phases.push(phase(
            prefix_state,
            head_state,
            apps,
            owner.chunk_packed_stats().completed_applications,
            forecast.step(4, true),
            0,
            0,
            "final-prompt-step",
            head_ns,
        )?);
        let start = Instant::now();
        owner.read_logits(&ticket, &mut prefill, &mut || deadline(end))?;
        let prefill_read_ns = elapsed_ns(start)?;
        let prefill_state = owner.state()?;
        live(prefill_state, r, 5)?;
        phases.push(phase(
            head_state,
            prefill_state,
            apps,
            owner.chunk_packed_stats().completed_applications,
            Traffic::default(),
            (VOCAB * 4) as u64,
            0,
            "prompt-logit-read",
            prefill_read_ns,
        )?);
        let selected = greedy(&prefill)?;
        if let Some(reference) = reference {
            exact(&prefill, &reference.prefill)?;
            ensure!(selected == reference.selected, "prefill token differs");
        } else {
            ensure!(
                prefill.iter().any(|v| *v != 0.0)
                    && prefill.windows(2).any(|v| v[0].to_bits() != v[1].to_bits()),
                "degenerate actual output"
            );
        }
        let chosen = reference.map_or(selected, |v| v.selected);
        let start = Instant::now();
        let ticket = owner.step_checked(chosen, &mut || deadline(end))?;
        let decode_ns = elapsed_ns(start)?;
        let decode_state = owner.state()?;
        live(decode_state, r, 6)?;
        phases.push(phase(
            prefill_state,
            decode_state,
            apps,
            owner.chunk_packed_stats().completed_applications,
            forecast.step(5, true),
            0,
            0,
            "decode-step",
            decode_ns,
        )?);
        let start = Instant::now();
        owner.read_logits(&ticket, &mut decode, &mut || deadline(end))?;
        let decode_read_ns = elapsed_ns(start)?;
        let next = greedy(&decode)?;
        if let Some(reference) = reference {
            exact(&decode, &reference.decode)?;
        }
        let after = owner.state()?;
        live(after, r, 6)?;
        phases.push(phase(
            decode_state,
            after,
            apps,
            owner.chunk_packed_stats().completed_applications,
            Traffic::default(),
            (VOCAB * 4) as u64,
            0,
            "decode-logit-read",
            decode_read_ns,
        )?);
        let expected = forecast.sequence(mode);
        accounting(
            before,
            after,
            before_apps,
            owner.chunk_packed_stats().completed_applications,
            expected,
            (2 * VOCAB * 4) as u64,
            0,
        )?;
        evidence.push(json!({"name":label,"mode":mode.label(),"selected_token":selected,
            "decode_input_token":chosen,"next_token":next,"traffic":expected.json(),
            "phases":phases,"prefix_ns":prefix_ns,"final_prompt_step_ns":head_ns,
            "prompt_logits_read_ns":prefill_read_ns,"decode_step_ns":decode_ns,
            "decode_logits_read_ns":decode_read_ns,
            "prompt_compute_ns":prefix_ns+head_ns,"prompt_with_readback_ns":prefix_ns+head_ns+prefill_read_ns,
            "chunk_applications":owner.chunk_packed_stats().completed_applications,"state":state(after)}));
        Ok(Sequence {
            prefill,
            decode,
            selected,
            next,
            prefix_ns,
        })
    }
    fn chunk_json(c: CheckedChunkRequirements) -> Value {
        json!({"tokens":c.tokens,"extra_tensor_allocations":c.extra_tensor_allocations,
            "extra_resident_tensor_bytes":c.extra_resident_tensor_bytes,
            "extra_device_charge_bytes":c.extra_device_charge_bytes,
            "extra_host_payload_bytes":c.extra_host_payload_bytes,
            "extra_recording_metadata_bytes":c.extra_recording_metadata_bytes,
            "extra_model_metadata_bytes":c.extra_model_metadata_bytes,
            "initialization_write_bytes":c.initialization_write_bytes,"reset_write_bytes":c.reset_write_bytes,
            "packed_source_tiles":c.packed_source_tiles,"packed_source_bytes":c.packed_source_bytes,
            "packed_submissions":c.packed_submissions,"ffn_tile_applications":c.ffn_tile_applications,
            "physical_dispatches":c.physical_dispatches,"logical_submissions":c.logical_submissions,
            "finite_checks":c.finite_checks,"finite_pending_bytes":c.finite_pending_bytes,
            "status_read_bytes":c.status_read_bytes,
            "extra_ffn_recording_write_bytes":c.extra_ffn_recording_write_bytes,
            "avoided_packed_write_bytes":c.avoided_packed_write_bytes})
    }

    pub fn run() -> Result<()> {
        let a = args()?;
        let started = Instant::now();
        let end = started + Duration::from_secs(WORK_SECONDS);
        ensure!(
            env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(std::ffi::OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly 0"
        );
        let metadata = fs::metadata(&a.model)?;
        ensure!(
            metadata.is_file() && metadata.len() == MODEL_BYTES,
            "fixed actual extent differs"
        );
        let gguf = Arc::new(Gguf::open(&a.model)?); // Header/index only before admission.
        ensure!(
            gguf.shard_count() == 1 && gguf.shard_len(0) == Some(MODEL_BYTES),
            "single fixed checkpoint required"
        );
        let binding = Binding::load(&gguf)?;
        let c = binding.config();
        ensure!(
            [
                c.n_layers,
                c.dim,
                c.feed_forward,
                c.vocab,
                c.n_heads,
                c.n_kv_heads,
                c.key_head_dim,
                c.rope_dim,
                c.ssm_group_count,
                c.ssm_state_size,
                c.ssm_time_step_rank,
                c.delta_key_width(),
                c.delta_qkv_width()
            ] == [64, 5120, 17408, VOCAB, 24, 4, 256, 64, 16, 128, 48, 2048, 10240],
            "fixed actual geometry differs"
        );
        let reads = ReadLimits {
            tile_rows: 4096,
            encoded_bytes: 64 * MIB as usize,
            decoded_bytes: 64 * MIB as usize,
        };
        let checker = NativeExecutionProfile::Qwen35Probe
            .finite_check_spec()
            .context("finite API missing")?;
        let scalar = RecordedQwen35Plan::new(&binding, CAPACITY, reads)?;
        ensure!(
            scalar.prefill_chunk_size().is_none()
                && scalar.checked_chunk_requirements(Some(checker))?.is_none(),
            "scalar default unexpectedly opted in"
        );
        let scalar_r = scalar.checked_requirements(Some(checker))?;
        let plan = scalar.with_prefill_chunk(CHUNK)?;
        let r = plan.checked_requirements(Some(checker))?;
        let chunk_r = plan
            .checked_chunk_requirements(Some(checker))?
            .context("missing B4 forecast")?;
        ensure!(
            plan.prefill_chunk_size() == Some(CHUNK)
                && chunk_r.tokens == CHUNK
                && chunk_r.extra_tensor_allocations == 1
                && chunk_r.extra_resident_tensor_bytes
                    == (CHUNK * (2 * c.dim + 2 * c.feed_forward) * 4) as u64
                && delta(r.resident_tensor_bytes, scalar_r.resident_tensor_bytes)?
                    == chunk_r.extra_resident_tensor_bytes
                && delta(r.device_charge_bytes, scalar_r.device_charge_bytes)?
                    == chunk_r.extra_device_charge_bytes
                && delta(r.reset_write_bytes, scalar_r.reset_write_bytes)?
                    == chunk_r.reset_write_bytes
                && chunk_r.initialization_write_bytes == chunk_r.extra_resident_tensor_bytes
                && chunk_r.reset_write_bytes == chunk_r.extra_resident_tensor_bytes
                && chunk_r.extra_resident_tensor_bytes == 720896
                && chunk_r.extra_device_charge_bytes == 720896
                && chunk_r.extra_host_payload_bytes == 0
                && r.auxiliary_bytes == scalar_r.auxiliary_bytes
                && r.packed_tile_bytes == scalar_r.packed_tile_bytes,
            "chunk resources differ from one shared workspace"
        );

        // Independently count unchanged scalar graph traffic from bound tensors.
        // Every constant is bounded by the fixed metadata geometry above.
        let norm_dispatches = 1 + c.dim.div_ceil(4096) as u64;
        let norm_writes = (4 * c.dim + 4) as u64;
        let mut arithmetic_dispatches = norm_dispatches;
        let mut arithmetic_submissions = 1;
        let mut record_writes = 4 + norm_writes; // embedding dummy and final RMS.
        let mut scaled_tiles = 0;
        let mut full_layers = 0;
        let mut ffn_tiles = 0;
        let mut ffn_bytes = 0;
        for block in binding.blocks() {
            let (core_output, core_dispatches, core_submissions) = match &block.attention {
                Attention::Full(v) => {
                    full_layers += 1;
                    // Separate Q/K RMS batch shares one head-sized workspace and
                    // epsilon; attention has 12*visible+4 scratch and scale4.
                    record_writes += (4 * c.key_head_dim + 4) as u64 + 12 + 8;
                    (
                        &v.output,
                        norm_dispatches
                            + 3 * (c.n_heads + c.n_kv_heads) as u64
                            + 4 * c.n_heads as u64,
                        4,
                    )
                }
                Attention::DeltaNet(v) => (&v.output, norm_dispatches + 4, 2),
            };
            arithmetic_dispatches +=
                core_dispatches + norm_dispatches + c.feed_forward.div_ceil(4096) as u64;
            arithmetic_submissions += core_submissions + 2;
            record_writes += norm_writes + norm_writes + 4; // core RMS, FFN RMS and activation.
            for matrix in [core_output, &block.ffn.down] {
                let p = PackedRowPlan::new(matrix, reads)?;
                let tiles = p.rows().div_ceil(p.tile_rows()) as u64;
                scaled_tiles += tiles;
                record_writes += p.rows() as u64 * 4 + tiles * 4; // scratch and residual scale.
            }
            for matrix in [&block.ffn.gate, &block.ffn.up, &block.ffn.down] {
                let p = PackedRowPlan::new(matrix, reads)?;
                ffn_tiles += p.rows().div_ceil(p.tile_rows()) as u64;
                ffn_bytes += matrix.nbytes();
            }
        }
        ensure!(full_layers == 16, "actual Full/Delta layer split differs");
        let full = Traffic {
            tiles: r.packed_tiles_per_step,
            packed_submissions: r.packed_tiles_per_step,
            chunk_applications: 0,
            bytes: r.packed_bytes_per_step,
            dispatches: r.packed_tiles_per_step
                + scaled_tiles
                + arithmetic_dispatches
                + r.finite_dispatches_per_step,
            submissions: r.packed_tiles_per_step
                + arithmetic_submissions
                + r.finite_submissions_per_step,
            status_bytes: r.finite_status_bytes_per_step + r.delta_status_bytes_per_step,
            host_writes: r.packed_bytes_per_step + r.finite_pending_bytes_per_step + record_writes,
        };
        let head = PackedRowPlan::new(binding.output(), reads)?;
        let head_tiles = head.rows().div_ceil(head.tile_rows()) as u64;
        let head_checks = 1 + c.vocab.div_ceil(checker.max_elements) as u64;
        let advance = Traffic {
            tiles: delta(full.tiles, head_tiles)?,
            packed_submissions: delta(full.tiles, head_tiles)?,
            chunk_applications: 0,
            bytes: delta(full.bytes, binding.output().nbytes())?,
            dispatches: delta(
                full.dispatches,
                head_tiles + norm_dispatches + head_checks * checker.physical_dispatches,
            )?,
            submissions: delta(
                full.submissions,
                head_tiles + 1 + head_checks * checker.logical_submissions,
            )?,
            status_bytes: delta(full.status_bytes, head_checks * checker.status_read_bytes)?,
            host_writes: delta(
                full.host_writes,
                binding.output().nbytes() + norm_writes + head_checks * checker.pending_write_bytes,
            )?,
        };
        let b = CHUNK as u64;
        ensure!(
            chunk_r.packed_source_tiles == b * (advance.tiles - ffn_tiles) + ffn_tiles
                && chunk_r.packed_source_bytes == b * (advance.bytes - ffn_bytes) + ffn_bytes
                && chunk_r.packed_submissions == b * advance.packed_submissions
                && chunk_r.ffn_tile_applications == b * ffn_tiles
                && chunk_r.physical_dispatches == b * advance.dispatches
                && chunk_r.logical_submissions == b * advance.submissions
                && chunk_r.status_read_bytes == b * advance.status_bytes
                && chunk_r.finite_checks == b * (r.finite_checks_per_step - head_checks)
                && chunk_r.finite_pending_bytes
                    == chunk_r.finite_checks * checker.pending_write_bytes
                && chunk_r.extra_ffn_recording_write_bytes
                    == b * c.n_layers as u64 * ((4 * c.dim + 8) as u64)
                && chunk_r.avoided_packed_write_bytes == (b - 1) * ffn_bytes,
            "model chunk forecast differs from independent scalar/FFN derivation"
        );
        let chunk = Traffic {
            tiles: chunk_r.packed_source_tiles,
            packed_submissions: chunk_r.packed_submissions,
            chunk_applications: chunk_r.ffn_tile_applications,
            bytes: chunk_r.packed_source_bytes,
            dispatches: chunk_r.physical_dispatches,
            submissions: chunk_r.logical_submissions,
            status_bytes: chunk_r.status_read_bytes,
            host_writes: delta(
                b * advance.host_writes
                    + 12 * full_layers * (b * (b - 1) / 2)
                    + chunk_r.extra_ffn_recording_write_bytes,
                chunk_r.avoided_packed_write_bytes,
            )?,
        };
        let traffic = Forecast {
            full,
            advance,
            chunk,
            full_attention_layers: full_layers,
        };
        ensure!(
            chunk.bytes < traffic.prefix(Mode::Scalar).bytes
                && chunk.tiles < traffic.prefix(Mode::Scalar).tiles
                && chunk.host_writes < traffic.prefix(Mode::Scalar).host_writes,
            "B4 forecast does not reduce packed/host traffic"
        );
        let aggregate = traffic
            .step(0, true)
            .plus(traffic.sequence(Mode::Scalar).times(2))
            .plus(traffic.sequence(Mode::Chunk).times(2));
        let budget = MemoryBudget::new(DEVICE_BYTES);
        let limits = ModelLimits {
            device_bytes: DEVICE_BYTES,
            total_process_bytes: PROCESS_BYTES,
            process_headroom_bytes: HEADROOM_BYTES,
        };
        let admission = plan.admit(&gguf, r, limits, &budget)?;
        // Eight retained output vectors, one temporary encoder, bounded JSON and
        // bookkeeping fit16MiB; the owner separately charges one caller vector.
        let extras = HARNESS_BYTES;
        let total = admission
            .total_process_bytes
            .checked_add(extras)
            .context("harness forecast overflow")?;
        ensure!(
            total <= PROCESS_BYTES,
            "owner and harness exceed process ceiling"
        );
        let forecast = json!({"source_file_bytes":MODEL_BYTES,"device_charge_bytes":r.device_charge_bytes,
            "resident_tensor_bytes":r.resident_tensor_bytes,"packed_tile_bytes":r.packed_tile_bytes,
            "auxiliary_bytes":r.auxiliary_bytes,"host_payload_bytes":r.host_payload_bytes,
            "recording_metadata_bytes":r.recording_metadata_bytes,"model_metadata_bytes":r.model_metadata_bytes,
            "process_headroom_bytes":HEADROOM_BYTES,"harness_extra_bytes":extras,
            "total_process_bytes":total,"process_ceiling_bytes":PROCESS_BYTES,"device_ceiling_bytes":DEVICE_BYTES,
            "reset_write_bytes":r.reset_write_bytes,"chunk_requirements":chunk_json(chunk_r),
            "scalar_full_at_position0":full.json(),"scalar_advance_at_position0":advance.json(),
            "chunk_at_position0":chunk.json(),"attention_write_increment_per_position":12*full_layers,
            "ffn_packed_bytes":ffn_bytes,"ffn_source_tiles":ffn_tiles,
            "packed_write_saving_per_prefix":chunk_r.avoided_packed_write_bytes,
            "host_write_saving_per_prefix":delta(traffic.prefix(Mode::Scalar).host_writes,chunk.host_writes)?,
            "scalar_prefix":traffic.prefix(Mode::Scalar).json(),"chunk_prefix":traffic.prefix(Mode::Chunk).json(),
            "scalar_sequence":traffic.sequence(Mode::Scalar).json(),"chunk_sequence":traffic.sequence(Mode::Chunk).json(),
            "aggregate":aggregate.json(),"logits_read_bytes":8*VOCAB*4,"reset_count":5,
            "tile_rows":reads.tile_rows,"encoded_bytes":reads.encoded_bytes,"decoded_bytes":reads.decoded_bytes});
        if !a.native {
            println!(
                "{}",
                json!({"schema_version":1,"execution":"metadata-only","qualified":false,
                "capacity":CAPACITY,"chunk_size":CHUNK,"prompt":PROMPT,"work_seconds":WORK_SECONDS,
                "cleanup_seconds":CLEANUP_SECONDS,"model_sha256_expected":MODEL_SHA256,
                "model_sha256_verified_by_probe":false,"vocab":VOCAB,"source_file_bytes":MODEL_BYTES,
                "model_payload_read":false,"device_opened":false,"forecast":forecast,"budget_after":budget.current_bytes()})
            );
            return Ok(());
        }
        deadline(end)?;
        fs::create_dir(&a.output)?;
        let healthy = Arc::new(AtomicBool::new(true));
        let mut evidence = Vec::new();
        let mut retirement = json!(null);
        let mut saved: Vec<(&str, Sequence)> = Vec::new();
        let mut flags = json!({"fixed_chunk_admission":false,"warmup_checked":false,
            "exact_prefill_and_decode":false,"lower_packed_and_host_write_traffic":false,
            "all_phase_accounting":false,"final_reset":false,"aggregate_accounting":false});
        let mut initialization_ns = None;
        let mut retirement_ns = None;
        let result = (|| -> Result<()> {
            let opening = Instant::now();
            let mut owner = plan.open_checked(&a.render,gguf,checker,limits,&budget,healthy.clone(),
                &mut || deadline(end))?;
            // Every fallible operation after open remains inside body, so ordered
            // retirement happens even if timing/accounting/serialization fails.
            let body = (|| -> Result<()> {
                initialization_ns = Some(elapsed_ns(opening)?);
                ensure!(owner.prefill_chunk_size() == Some(CHUNK)
                    && owner.chunk_requirements()? == Some(chunk_r),"opened chunk admission differs");
                flags["fixed_chunk_admission"] = json!(true);
                let prepared = owner.state()?;
                live(prepared,r,0)?;
                ensure!(prepared.packed.completed_tiles == 0 && prepared.native.dispatch_count == 0
                    && prepared.logits_read_bytes == 0 && prepared.reset_write_bytes == 0
                    && owner.chunk_packed_stats().completed_applications == 0,"eager token/reset activity");
                evidence.push(json!({"name":"prepared","state":state(prepared)}));
                let warm = Instant::now();
                let _ticket = owner.step_checked(PROMPT[0],&mut || deadline(end))?;
                let warm_ns = elapsed_ns(warm)?;
                let warmed = owner.state()?;
                live(warmed,r,1)?;
                evidence.push(json!({"name":"warmup","phase":phase(prepared,warmed,0,
                    owner.chunk_packed_stats().completed_applications,traffic.step(0,true),0,0,
                    "full-step-no-readback",warm_ns)?,"state":state(warmed)}));
                flags["warmup_checked"] = json!(true);
                reset(&mut owner,r,end,"after-warmup",&mut evidence)?;
                let a1 = sequence(&mut owner,Mode::Scalar,"a1",None,r,traffic,end,&mut evidence)?;
                reset(&mut owner,r,end,"after-a1",&mut evidence)?;
                let b1 = sequence(&mut owner,Mode::Chunk,"b1",Some(&a1),r,traffic,end,&mut evidence)?;
                reset(&mut owner,r,end,"after-b1",&mut evidence)?;
                let b2 = sequence(&mut owner,Mode::Chunk,"b2",Some(&a1),r,traffic,end,&mut evidence)?;
                reset(&mut owner,r,end,"after-b2",&mut evidence)?;
                let a2 = sequence(&mut owner,Mode::Scalar,"a2",Some(&a1),r,traffic,end,&mut evidence)?;
                flags["exact_prefill_and_decode"] = json!(true);
                flags["lower_packed_and_host_write_traffic"] = json!(true);
                flags["all_phase_accounting"] = json!(true);
                reset(&mut owner,r,end,"after-a2-final",&mut evidence)?;
                let final_state = owner.state()?;
                live(final_state,r,0)?;
                ensure!(final_state.reset_write_bytes == 5*r.reset_write_bytes,"final reset total differs");
                flags["final_reset"] = json!(true);
                accounting(prepared,final_state,0,owner.chunk_packed_stats().completed_applications,
                    aggregate,(8*VOCAB*4) as u64,5*r.reset_write_bytes)?;
                flags["aggregate_accounting"] = json!(true);
                evidence.push(json!({"name":"aggregate","traffic":aggregate.json(),"state":state(final_state),
                    "chunk_applications":owner.chunk_packed_stats().completed_applications}));
                saved.extend([("a1",a1),("b1",b1),("b2",b2),("a2",a2)]);
                Ok(())
            })();
            if body.is_err() {
                if let Ok(snapshot) = owner.state() {
                    evidence.push(json!({"name":"failed","state":state(snapshot),
                        "chunk_applications":owner.chunk_packed_stats().completed_applications}));
                }
            }
            let retiring = Instant::now();
            let retired = owner.retire();
            retirement_ns = Some(elapsed_ns(retiring)?);
            retirement = json!({"clean":retired.clean,"expected_device_charge_bytes":retired.expected_device_charge_bytes,
                "budget_before_bytes":retired.budget_before_bytes,"budget_after_bytes":retired.budget_after_bytes,
                "native_evidence_present":retired.native.is_some()});
            body?;
            ensure!(retired.clean && retired.native.is_some() && retired.budget_after_bytes == 0,
                "native owner did not retire cleanly");
            Ok(())
        })().and_then(|()| {
            ensure!(budget.current_bytes() == 0 && healthy.load(Ordering::Acquire),"retirement health/budget failed");
            deadline(end)
        });
        let result = result.and_then(|()| {
            for (label, value) in &saved {
                logits_file(
                    &a.output.join(format!("{label}-prefill.f32le")),
                    &value.prefill,
                )?;
                logits_file(
                    &a.output.join(format!("{label}-decode.f32le")),
                    &value.decode,
                )?;
            }
            deadline(end)
        });
        let selections: Vec<Value> = saved
            .iter()
            .map(|(label, value)| {
                json!({"name":label,
            "selected_token":value.selected,"next_token":value.next})
            })
            .collect();
        let timing = if saved.len() == 4 {
            let a1 = saved[0].1.prefix_ns as f64;
            let b1 = saved[1].1.prefix_ns as f64;
            let b2 = saved[2].1.prefix_ns as f64;
            let a2 = saved[3].1.prefix_ns as f64;
            let a_mean = (a1 + a2) / 2.0;
            let b_mean = (b1 + b2) / 2.0;
            let a_spread = (a1 - a2).abs() / a_mean;
            let b_spread = (b1 - b2).abs() / b_mean;
            json!({"order":["a1","b1","b2","a2"],"prefix_tokens":4,
                "scalar_prefix_mean_ns":a_mean,"chunk_prefix_mean_ns":b_mean,
                "scalar_prefix_seconds_per_token":a_mean/(4.0*1e9),
                "chunk_prefix_seconds_per_token":b_mean/(4.0*1e9),
                "chunk_to_scalar_ratio":b_mean/a_mean,"prefix_reduction_fraction":1.0-b_mean/a_mean,
                "scalar_relative_spread":a_spread,"chunk_relative_spread":b_spread,
                "each_chunk_faster_than_each_scalar":b1.max(b2)<a1.min(a2),
                "observed_benefit_heuristic":b_mean<=0.9*a_mean && b1.max(b2)<a1.min(a2)
                    && a_spread<=0.1 && b_spread<=0.1,
                "heuristic_affects_passed":false,"sustained_performance_claim":false})
        } else {
            json!(null)
        };
        let receipt = json!({"schema_version":1,"execution":"native-openmythos-b4-abba-diagnostic",
            "passed":result.is_ok(),"qualified":false,"runtime_qualified":false,
            "independent_sequence_accuracy_qualified":false,"normal_antimatter_qualified":false,
            "performance_qualified":false,
            "comparison":"complete finite logits bitwise against same-owner scalar state-advance sequence",
            "model_sha256_expected":MODEL_SHA256,"model_sha256_verified_by_probe":false,
            "model_path":a.model,"source_file_bytes":MODEL_BYTES,"capacity":CAPACITY,"chunk_size":CHUNK,
            "prompt":PROMPT,"vocab":VOCAB,"completed_positions_expected":25,"scalar_full_head_steps_expected":9,
            "scalar_state_advances_expected":8,"successful_chunks_expected":2,"warmup_steps_expected":1,
            "logits_read_vectors_expected":8,"reset_count_expected":5,"work_seconds":WORK_SECONDS,
            "cleanup_seconds":CLEANUP_SECONDS,"forecast":forecast,"flags":flags,"selections":selections,
            "initialization_ns":initialization_ns,"retirement_ns":retirement_ns,"observed_timing":timing,
            "evidence":evidence,"retirement":retirement,"budget_after":budget.current_bytes(),
            "process_healthy":healthy.load(Ordering::Acquire),"elapsed_seconds":started.elapsed().as_secs_f64(),
            "error":result.as_ref().err().map(|error| format!("{error:#}"))});
        let mut encoded = serde_json::to_vec_pretty(&receipt)?;
        encoded.push(b'\n');
        ensure!(encoded.len() <= MAX_RECEIPT, "receipt bound exceeded");
        output(&a.output.join("receipt.json"), &encoded)?;
        println!("{}", receipt);
        result
    }
}
#[cfg(target_os = "linux")]
fn main() -> anyhow::Result<()> {
    probe::run()
}
#[cfg(not(target_os = "linux"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("Linux native diagnostic only")
}
