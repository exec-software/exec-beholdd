//! Explicit bounded tiny-fixture diagnostic. External cgroup, thermal, crash-watch,
//! deadline and frozen source/binary/fixture provenance are mandatory.
//! Establishes only native scalar/B4 FFN-chunk sequence equivalence.
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
    const FIXTURE_BYTES: u64 = 1_601_120;
    const FIXTURE_SHA256: &str = "67e8dd3346a6d3363fed4f8a98cc2bd290b719a306a54e127259400a8d1d0077";
    const PROMPT: [u32; 6] = [7, 42, 7, 42, 7, 42];
    const CAPACITY: usize = 8;
    const CHUNK: usize = 4;
    const WORK_SECONDS: u64 = 180;
    const CLEANUP_SECONDS: u64 = 60;
    const PROCESS_BYTES: u64 = 512 * MIB;
    const HEADROOM_BYTES: u64 = 128 * MIB;
    const DEVICE_BYTES: u64 = 8 * MIB;
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
        values.try_reserve_exact(512)?;
        values.resize(512, 0.0);
        Ok(values)
    }
    fn greedy(values: &[f32]) -> Result<u32> {
        ensure!(
            values.len() == 512 && values.iter().all(|v| v.is_finite()),
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
            actual.len() == 512 && expected.len() == 512,
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
        bytes.try_reserve_exact(512 * 4)?;
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
    fn no_effect(a: ModelStateSnapshot, b: ModelStateSnapshot) -> bool {
        a.phase == b.phase
            && a.committed_positions == b.committed_positions
            && a.capacity == b.capacity
            && a.poisoned == b.poisoned
            && a.process_healthy == b.process_healthy
            && a.source == b.source
            && a.packed == b.packed
            && a.native == b.native
            && a.reset_write_bytes == b.reset_write_bytes
            && a.logits_read_bytes == b.logits_read_bytes
            && a.budget_current_bytes == b.budget_current_bytes
            && a.budget_expected_bytes == b.budget_expected_bytes
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
            // The complete fixture is fixed to four layers, at most seven token
            // positions per sequence and four sequences. No caller count enters.
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
        fn baseline(self) -> Traffic {
            (0..7).fold(Traffic::default(), |total, p| {
                total.plus(self.step(p, true))
            })
        }
        fn sequence(self, mode: Mode) -> Traffic {
            match mode {
                Mode::Baseline => self.baseline(),
                Mode::Chunk => self
                    .chunk(0)
                    .plus(self.step(4, false))
                    .plus(self.step(5, true))
                    .plus(self.step(6, true)),
                Mode::StaleTicket => self
                    .step(0, true)
                    .plus(self.chunk(1))
                    .plus(self.step(5, true))
                    .plus(self.step(6, true)),
            }
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
    fn reset(owner: &mut ResidentQwen35CheckedModel, r: CheckedRequirements) -> Result<()> {
        let before = owner.state()?;
        let apps = owner.chunk_packed_stats().completed_applications;
        let cleanup = Instant::now() + Duration::from_secs(CLEANUP_SECONDS);
        owner.reset_checked(&mut || deadline(cleanup))?;
        let after = owner.state()?;
        live(after, r, 0)?;
        accounting(
            before,
            after,
            apps,
            owner.chunk_packed_stats().completed_applications,
            Traffic::default(),
            0,
            r.reset_write_bytes,
        )?;
        Ok(())
    }
    fn refused(owner: &mut ResidentQwen35CheckedModel, tokens: &[u32]) -> Result<()> {
        let before = owner.state()?;
        let apps = owner.chunk_packed_stats();
        let mut callbacks = 0usize;
        let result = owner.advance_chunk_checked(tokens, &mut || {
            callbacks += 1;
            anyhow::bail!("invalid chunk reached control")
        });
        ensure!(
            result.is_err()
                && callbacks == 0
                && no_effect(before, owner.state()?)
                && apps == owner.chunk_packed_stats(),
            "invalid chunk changed state or called control"
        );
        Ok(())
    }
    fn invalid_requests(owner: &mut ResidentQwen35CheckedModel) -> Result<()> {
        for tokens in [
            &[][..],
            &[7][..],
            &[7, 42][..],
            &[7, 42, 7][..],
            &[7, 42, 7, 42, 7][..],
            &[7, 42, 512, 42][..],
        ] {
            refused(owner, tokens)?;
        }
        Ok(())
    }
    #[derive(Clone, Copy)]
    enum Mode {
        Baseline,
        Chunk,
        StaleTicket,
    }
    impl Mode {
        fn label(self) -> &'static str {
            match self {
                Self::Baseline => "baseline",
                Self::Chunk => "chunk-tail",
                Self::StaleTicket => "mixed-stale-ticket",
            }
        }
    }
    struct Sequence {
        prefill: Vec<f32>,
        decode: Vec<f32>,
        selected: u32,
        next: u32,
    }
    fn sequence(
        owner: &mut ResidentQwen35CheckedModel,
        mode: Mode,
        reference: Option<&Sequence>,
        r: CheckedRequirements,
        forecast: Forecast,
        end: Instant,
        evidence: &mut Vec<Value>,
    ) -> Result<Sequence> {
        let before = owner.state()?;
        let before_apps = owner.chunk_packed_stats().completed_applications;
        live(before, r, 0)?;
        let mut prefill = vector()?;
        let mut decode = vector()?;
        match mode {
            Mode::Baseline => {
                for (index, &token) in PROMPT[..5].iter().enumerate() {
                    let _ticket = owner.step_checked(token, &mut || deadline(end))?;
                    live(owner.state()?, r, index + 1)?;
                }
            }
            Mode::Chunk => {
                owner.advance_chunk_checked(&PROMPT[..CHUNK], &mut || deadline(end))?;
                live(owner.state()?, r, 4)?;
                owner.advance_checked(PROMPT[4], &mut || deadline(end))?;
                live(owner.state()?, r, 5)?;
            }
            Mode::StaleTicket => {
                let old = owner.step_checked(PROMPT[0], &mut || deadline(end))?;
                owner.read_logits(&old, &mut prefill, &mut || deadline(end))?;
                greedy(&prefill)?;
                live(owner.state()?, r, 1)?;
                owner.advance_chunk_checked(&PROMPT[1..5], &mut || deadline(end))?;
                live(owner.state()?, r, 5)?;
                let snapshot = owner.state()?;
                let apps = owner.chunk_packed_stats();
                prefill.fill(-123.25);
                ensure!(
                    owner
                        .read_logits(&old, &mut prefill, &mut || deadline(end))
                        .is_err()
                        && no_effect(snapshot, owner.state()?)
                        && apps == owner.chunk_packed_stats()
                        && prefill
                            .iter()
                            .all(|v| v.to_bits() == (-123.25f32).to_bits()),
                    "chunk exposed stale logits or modified refusal output"
                );
            }
        }
        let ticket = owner.step_checked(PROMPT[5], &mut || deadline(end))?;
        owner.read_logits(&ticket, &mut prefill, &mut || deadline(end))?;
        live(owner.state()?, r, 6)?;
        let selected = greedy(&prefill)?;
        if let Some(reference) = reference {
            exact(&prefill, &reference.prefill)?;
            ensure!(selected == reference.selected, "prefill token differs");
        } else {
            ensure!(
                prefill.iter().any(|v| *v != 0.0)
                    && prefill.windows(2).any(|v| v[0].to_bits() != v[1].to_bits()),
                "degenerate fixture output"
            );
        }
        let chosen = reference.map_or(selected, |v| v.selected);
        let ticket = owner.step_checked(chosen, &mut || deadline(end))?;
        owner.read_logits(&ticket, &mut decode, &mut || deadline(end))?;
        let next = greedy(&decode)?;
        if let Some(reference) = reference {
            exact(&decode, &reference.decode)?;
        }
        let after = owner.state()?;
        live(after, r, 7)?;
        let expected = forecast.sequence(mode);
        let read_bytes = (2 + u64::from(matches!(mode, Mode::StaleTicket))) * 512 * 4;
        accounting(
            before,
            after,
            before_apps,
            owner.chunk_packed_stats().completed_applications,
            expected,
            read_bytes,
            0,
        )?;
        if !matches!(mode, Mode::Baseline) {
            let baseline = forecast.baseline();
            ensure!(
                expected.tiles < baseline.tiles
                    && expected.bytes < baseline.bytes
                    && expected.host_writes < baseline.host_writes,
                "chunk path did not reduce source/upload/host-write traffic"
            );
        }
        // Four rows cannot fit at p=7 in C8, and refusal must happen before control.
        refused(owner, &PROMPT[..CHUNK])?;
        evidence.push(json!({"name":mode.label(),"selected_token":selected,
            "decode_input_token":chosen,"next_token":next,"traffic":expected.json(),
            "chunk_applications":owner.chunk_packed_stats().completed_applications,"state":state(after)}));
        Ok(Sequence {
            prefill,
            decode,
            selected,
            next,
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
            metadata.is_file() && metadata.len() == FIXTURE_BYTES,
            "fixed tiny extent differs"
        );
        let gguf = Arc::new(Gguf::open(&a.model)?); // Header/index only before admission.
        ensure!(
            gguf.shard_count() == 1 && gguf.shard_len(0) == Some(FIXTURE_BYTES),
            "single fixed fixture required"
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
                c.value_head_dim,
                c.rope_dim,
                c.full_attention_interval,
                c.ssm_conv_kernel,
                c.ssm_group_count,
                c.ssm_inner_size,
                c.ssm_state_size,
                c.ssm_time_step_rank
            ] == [4, 256, 256, 512, 2, 1, 256, 256, 64, 4, 4, 1, 256, 128, 2],
            "fixed tiny geometry differs"
        );
        let reads = ReadLimits {
            tile_rows: 64,
            encoded_bytes: MIB as usize,
            decoded_bytes: 4 * MIB as usize,
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
        let embedding_bytes = PackedRowPlan::new(binding.embedding(), reads)?.row_bytes() as u64;
        let cancelled_traffic = Traffic {
            tiles: 1,
            packed_submissions: 1,
            chunk_applications: 0,
            bytes: embedding_bytes,
            dispatches: 1,
            submissions: 1,
            status_bytes: 0,
            host_writes: embedding_bytes + 4,
        };
        let budget = MemoryBudget::new(DEVICE_BYTES);
        let limits = ModelLimits {
            device_bytes: DEVICE_BYTES,
            total_process_bytes: PROCESS_BYTES,
            process_headroom_bytes: HEADROOM_BYTES,
        };
        let admission = plan.admit(&gguf, r, limits, &budget)?;
        let extras = (16 * 512 * 4 + MAX_RECEIPT + (1 << 20)) as u64;
        let total = admission
            .total_process_bytes
            .checked_add(extras)
            .context("harness forecast overflow")?;
        ensure!(
            total <= PROCESS_BYTES,
            "owner and harness exceed process ceiling"
        );
        let forecast = json!({"source_file_bytes":FIXTURE_BYTES,"device_charge_bytes":r.device_charge_bytes,
            "resident_tensor_bytes":r.resident_tensor_bytes,"packed_tile_bytes":r.packed_tile_bytes,
            "auxiliary_bytes":r.auxiliary_bytes,"host_payload_bytes":r.host_payload_bytes,
            "recording_metadata_bytes":r.recording_metadata_bytes,"model_metadata_bytes":r.model_metadata_bytes,
            "process_headroom_bytes":HEADROOM_BYTES,"harness_extra_bytes":extras,
            "total_process_bytes":total,"process_ceiling_bytes":PROCESS_BYTES,"device_ceiling_bytes":DEVICE_BYTES,
            "reset_write_bytes":r.reset_write_bytes,"chunk_requirements":chunk_json(chunk_r),
            "scalar_full_at_position0":full.json(),"scalar_advance_at_position0":advance.json(),
            "chunk_at_position0":chunk.json(),"attention_write_increment_per_position":12*full_layers,
            "baseline_sequence":traffic.sequence(Mode::Baseline).json(),
            "chunk_tail_sequence":traffic.sequence(Mode::Chunk).json(),
            "mixed_sequence":traffic.sequence(Mode::StaleTicket).json(),"cancelled_embedding":cancelled_traffic.json()});
        if !a.native {
            println!(
                "{}",
                json!({"schema_version":1,"execution":"metadata-only","qualified":false,
                "capacity":CAPACITY,"chunk_size":CHUNK,"prompt":PROMPT,"work_seconds":WORK_SECONDS,
                "cleanup_seconds":CLEANUP_SECONDS,"fixture_sha256_expected":FIXTURE_SHA256,
                "fixture_sha256_verified_by_probe":false,"forecast":forecast,"budget_after":budget.current_bytes()})
            );
            return Ok(());
        }
        deadline(end)?;
        fs::create_dir(&a.output)?;
        let healthy = Arc::new(AtomicBool::new(true));
        let mut evidence = Vec::new();
        let mut retirement = json!(null);
        let mut saved: Vec<(&str, Sequence)> = Vec::new();
        let mut flags = json!({"fixed_chunk_admission":false,"invalid_requests_no_effect":false,
            "exact_prefill_and_decode":false,"lower_packed_and_host_write_traffic":false,
            "stale_ticket_refusal":false,"cancel_after_embedding_fence":false,
            "poisoned_retry_refusal":false,"cancel_reset_replay":false,"final_reset":false,
            "aggregate_accounting":false});
        let result = (|| -> Result<()> {
            let mut owner = plan.open_checked(&a.render,gguf,checker,limits,&budget,healthy.clone(),
                &mut || deadline(end))?;
            let body = (|| -> Result<()> {
                ensure!(owner.prefill_chunk_size() == Some(CHUNK)
                    && owner.chunk_requirements()? == Some(chunk_r),"opened chunk admission differs");
                flags["fixed_chunk_admission"] = json!(true);
                let prepared = owner.state()?;
                live(prepared,r,0)?;
                ensure!(prepared.packed.completed_tiles == 0 && prepared.native.dispatch_count == 0
                    && prepared.logits_read_bytes == 0 && prepared.reset_write_bytes == 0
                    && owner.chunk_packed_stats().completed_applications == 0,"eager token/reset activity");
                invalid_requests(&mut owner)?;
                flags["invalid_requests_no_effect"] = json!(true);
                evidence.push(json!({"name":"prepared","state":state(prepared)}));
                let baseline = sequence(&mut owner,Mode::Baseline,None,r,traffic,end,&mut evidence)?;
                reset(&mut owner,r)?;
                let optimized = sequence(&mut owner,Mode::Chunk,Some(&baseline),r,traffic,end,&mut evidence)?;
                flags["exact_prefill_and_decode"] = json!(true);
                flags["lower_packed_and_host_write_traffic"] = json!(true);
                reset(&mut owner,r)?;
                let mixed = sequence(&mut owner,Mode::StaleTicket,Some(&baseline),r,traffic,end,&mut evidence)?;
                flags["stale_ticket_refusal"] = json!(true);
                reset(&mut owner,r)?;
                let before = owner.state()?;
                let before_apps = owner.chunk_packed_stats().completed_applications;
                let mut callbacks = 0;
                let cancelled = owner.advance_chunk_checked(&PROMPT[..CHUNK],&mut || {
                    deadline(end)?;
                    callbacks += 1;
                    // Model admission then tile read/upload/submit/post-fence.
                    ensure!(callbacks != 5,"intentional after-embedding-fence chunk cancellation");
                    Ok(())
                });
                let after = owner.state()?;
                ensure!(cancelled.is_err() && callbacks == 5 && after.phase == ModelPhase::Embedding
                    && after.committed_positions == 0 && after.poisoned && after.process_healthy
                    && !after.native.poisoned && after.native.weight_live_bytes == 0
                    && after.native.auxiliary_live_bytes == 0
                    && after.native.tensor_live_bytes == r.resident_tensor_bytes
                    && after.budget_current_bytes == r.device_charge_bytes,
                    "chunk cancellation did not preserve poisoned unpublished fenced prefix");
                accounting(before,after,before_apps,owner.chunk_packed_stats().completed_applications,
                    cancelled_traffic,0,0)?;
                flags["cancel_after_embedding_fence"] = json!(true);
                ensure!(owner.advance_chunk_checked(&PROMPT[..CHUNK],&mut || deadline(end)).is_err()
                    && no_effect(after,owner.state()?)
                    && owner.advance_checked(PROMPT[0],&mut || deadline(end)).is_err()
                    && no_effect(after,owner.state()?)
                    && owner.step_checked(PROMPT[0],&mut || deadline(end)).is_err()
                    && no_effect(after,owner.state()?)
                    && owner.chunk_packed_stats().completed_applications == before_apps,
                    "poisoned retry changed chunk/scalar owner");
                flags["poisoned_retry_refusal"] = json!(true);
                evidence.push(json!({"name":"cancelled-embedding","state":state(after)}));
                reset(&mut owner,r)?;
                let recovered = sequence(&mut owner,Mode::Chunk,Some(&baseline),r,traffic,end,&mut evidence)?;
                flags["cancel_reset_replay"] = json!(true);
                reset(&mut owner,r)?;
                let final_state = owner.state()?;
                live(final_state,r,0)?;
                ensure!(final_state.reset_write_bytes == 5*r.reset_write_bytes,"final reset total differs");
                flags["final_reset"] = json!(true);
                let total = traffic.sequence(Mode::Baseline).plus(traffic.sequence(Mode::Chunk).times(2))
                    .plus(traffic.sequence(Mode::StaleTicket)).plus(cancelled_traffic);
                accounting(prepared,final_state,0,owner.chunk_packed_stats().completed_applications,
                    total,9*512*4,5*r.reset_write_bytes)?;
                flags["aggregate_accounting"] = json!(true);
                evidence.push(json!({"name":"final-reset","traffic":total.json(),"state":state(final_state),
                    "chunk_applications":owner.chunk_packed_stats().completed_applications}));
                saved.extend([("baseline",baseline),("chunk",optimized),("mixed",mixed),("recovered",recovered)]);
                Ok(())
            })();
            if body.is_err() {
                if let Ok(snapshot) = owner.state() {
                    evidence.push(json!({"name":"failed","state":state(snapshot),
                        "chunk_applications":owner.chunk_packed_stats().completed_applications}));
                }
            }
            let retired = owner.retire();
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
        let receipt = json!({"schema_version":1,"execution":"native-qwen35-b4-ffn-chunk-equivalence",
            "passed":result.is_ok(),"qualified":false,"runtime_qualified":false,
            "independent_sequence_accuracy_qualified":false,"normal_antimatter_qualified":false,
            "performance_qualified":false,
            "comparison":"complete finite logits bitwise against native scalar full-head sequence only",
            "fixture_sha256_expected":FIXTURE_SHA256,"fixture_sha256_verified_by_probe":false,
            "model_path":a.model,"source_file_bytes":FIXTURE_BYTES,"capacity":CAPACITY,"chunk_size":CHUNK,
            "prompt":PROMPT,"vocab":512,"completed_positions_expected":28,"scalar_full_head_steps_expected":14,
            "scalar_state_advances_expected":2,"successful_chunks_expected":3,"cancelled_embedding_tiles_expected":1,
            "logits_read_vectors_expected":9,"reset_count_expected":5,"work_seconds":WORK_SECONDS,
            "cleanup_seconds":CLEANUP_SECONDS,"forecast":forecast,"flags":flags,"selections":selections,
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
