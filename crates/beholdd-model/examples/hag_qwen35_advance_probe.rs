//! Explicit bounded tiny-fixture diagnostic. External cgroup, thermal, crash-watch,
//! deadline and frozen source/binary/fixture provenance are mandatory.
//! Establishes only native full-head/state-advance sequence equivalence.
#[cfg(target_os = "linux")]
mod probe {
    use anyhow::{ensure, Context, Result};
    use beholdd_backend::hag::native_execution::{MemoryBudget, NativeExecutionProfile};
    use beholdd_format::{Gguf, RangeReader};
    use beholdd_model::qwen35::config::{Attention, Binding};
    use beholdd_model::qwen35::recorded_model::{
        CheckedRequirements, ModelLimits, ModelPhase, ModelStateSnapshot, RecordedQwen35Plan,
        ResidentQwen35CheckedModel,
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
    const PROMPT: [u32; 3] = [7, 42, 7];
    const CAPACITY: usize = 4;
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

    #[derive(Clone, Copy)]
    struct Traffic {
        tiles: u64,
        bytes: u64,
        dispatches: u64,
        submissions: u64,
        status_bytes: u64,
    }
    impl Traffic {
        fn times(self, n: u64) -> Self {
            // Only fixed counts 0..=16 and the admitted four-layer fixture enter here.
            Self {
                tiles: self.tiles * n,
                bytes: self.bytes * n,
                dispatches: self.dispatches * n,
                submissions: self.submissions * n,
                status_bytes: self.status_bytes * n,
            }
        }
        fn plus(self, other: Self) -> Self {
            Self {
                tiles: self.tiles + other.tiles,
                bytes: self.bytes + other.bytes,
                dispatches: self.dispatches + other.dispatches,
                submissions: self.submissions + other.submissions,
                status_bytes: self.status_bytes + other.status_bytes,
            }
        }
        fn json(self) -> Value {
            json!({"tiles":self.tiles,"packed_bytes":self.bytes,"dispatches":self.dispatches,
                "logical_submissions":self.submissions,"status_read_bytes":self.status_bytes})
        }
    }
    fn accounting(
        before: ModelStateSnapshot,
        after: ModelStateSnapshot,
        t: Traffic,
        reads: u64,
    ) -> Result<()> {
        ensure!(
            delta(after.packed.read_attempts, before.packed.read_attempts)? == t.tiles
                && delta(
                    after.packed.submission_attempts,
                    before.packed.submission_attempts
                )? == t.tiles
                && delta(after.packed.completed_tiles, before.packed.completed_tiles)? == t.tiles
                && delta(
                    after.packed.returned_read_bytes,
                    before.packed.returned_read_bytes
                )? == t.bytes
                && delta(after.packed.uploaded_bytes, before.packed.uploaded_bytes)? == t.bytes
                && delta(after.source.attempts, before.source.attempts)? == t.tiles
                && delta(after.source.returned_bytes, before.source.returned_bytes)? == t.bytes
                && delta(
                    after.native.uploaded_weight_bytes,
                    before.native.uploaded_weight_bytes
                )? == t.bytes
                && delta(after.native.dispatch_count, before.native.dispatch_count)?
                    == t.dispatches
                && delta(
                    after.native.dispatch_attempts,
                    before.native.dispatch_attempts
                )? == t.dispatches
                && delta(
                    after.native.logical_submissions,
                    before.native.logical_submissions
                )? == t.submissions
                && delta(after.native.host_read_bytes, before.native.host_read_bytes)?
                    == t.status_bytes + reads
                && delta(after.logits_read_bytes, before.logits_read_bytes)? == reads,
            "source-derived exact inference traffic differs"
        );
        Ok(())
    }
    fn reset(owner: &mut ResidentQwen35CheckedModel, r: CheckedRequirements) -> Result<()> {
        let before = owner.state()?;
        let cleanup = Instant::now() + Duration::from_secs(CLEANUP_SECONDS);
        owner.reset_checked(&mut || deadline(cleanup))?;
        let after = owner.state()?;
        live(after, r, 0)?;
        ensure!(
            delta(after.reset_write_bytes, before.reset_write_bytes)? == r.reset_write_bytes
                && after.source == before.source
                && after.packed == before.packed
                && after.logits_read_bytes == before.logits_read_bytes
                && after.native.dispatch_count == before.native.dispatch_count,
            "physical reset accounting differs"
        );
        Ok(())
    }

    #[derive(Clone, Copy)]
    enum Mode {
        Baseline,
        Advance,
        StaleTicket,
    }
    impl Mode {
        fn skipped(self) -> u64 {
            match self {
                Self::Baseline => 0,
                Self::Advance => 2,
                Self::StaleTicket => 1,
            }
        }
        fn label(self) -> &'static str {
            match self {
                Self::Baseline => "baseline",
                Self::Advance => "advance",
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
    #[allow(clippy::too_many_arguments)]
    fn sequence(
        owner: &mut ResidentQwen35CheckedModel,
        mode: Mode,
        reference: Option<&Sequence>,
        r: CheckedRequirements,
        full: Traffic,
        advance: Traffic,
        end: Instant,
        evidence: &mut Vec<Value>,
    ) -> Result<Sequence> {
        let before = owner.state()?;
        live(before, r, 0)?;
        let mut prefill = vector()?;
        let mut decode = vector()?;
        let mut stale = None;
        for (index, token) in PROMPT.iter().copied().enumerate() {
            let state_only = index < 2
                && match mode {
                    Mode::Baseline => false,
                    Mode::Advance => true,
                    Mode::StaleTicket => index == 1,
                };
            if state_only {
                owner.advance_checked(token, &mut || deadline(end))?;
                if let Some(ticket) = stale.take() {
                    let snapshot = owner.state()?;
                    prefill.fill(-123.25);
                    ensure!(
                        owner
                            .read_logits(&ticket, &mut prefill, &mut || deadline(end))
                            .is_err()
                            && no_effect(snapshot, owner.state()?)
                            && prefill
                                .iter()
                                .all(|v| v.to_bits() == (-123.25f32).to_bits()),
                        "state advance exposed stale logits or modified refusal output"
                    );
                }
            } else {
                let ticket = owner.step_checked(token, &mut || deadline(end))?;
                if index == 2 {
                    owner.read_logits(&ticket, &mut prefill, &mut || deadline(end))?;
                } else if matches!(mode, Mode::StaleTicket) && index == 0 {
                    owner.read_logits(&ticket, &mut prefill, &mut || deadline(end))?;
                    greedy(&prefill)?; // Prove the ticket was valid before advancement.
                    stale = Some(ticket);
                }
            }
            live(owner.state()?, r, index + 1)?;
        }
        ensure!(stale.is_none(), "stale ticket refusal was not exercised");
        let selected = greedy(&prefill)?;
        if let Some(reference) = reference {
            exact(&prefill, &reference.prefill)?;
            ensure!(
                selected == reference.selected,
                "prefill selected token differs"
            );
        } else {
            ensure!(
                prefill.iter().any(|v| *v != 0.0)
                    && prefill.windows(2).any(|v| v[0].to_bits() != v[1].to_bits()),
                "degenerate fixture output cannot establish nonzero sequence equivalence"
            );
        }
        // Every comparison evaluates the baseline-selected token after proving
        // full prompt logits and selection equal, rather than diverging inputs.
        let decode_token = reference.map_or(selected, |value| value.selected);
        let ticket = owner.step_checked(decode_token, &mut || deadline(end))?;
        owner.read_logits(&ticket, &mut decode, &mut || deadline(end))?;
        let next = greedy(&decode)?;
        if let Some(reference) = reference {
            exact(&decode, &reference.decode)?;
        }
        let after = owner.state()?;
        live(after, r, CAPACITY)?;
        let skipped = mode.skipped();
        let expected = full.times(4 - skipped).plus(advance.times(skipped));
        let reads = (2 + u64::from(matches!(mode, Mode::StaleTicket))) * 512 * 4;
        accounting(before, after, expected, reads)?;
        if skipped > 0 {
            ensure!(
                expected.tiles < full.times(4).tiles
                    && expected.bytes < full.times(4).bytes
                    && expected.dispatches < full.times(4).dispatches,
                "state advance did not reduce packed work and dispatches"
            );
        }
        let refusal = owner.state()?;
        ensure!(
            owner
                .advance_checked(PROMPT[0], &mut || deadline(end))
                .is_err()
                && no_effect(refusal, owner.state()?),
            "capacity refusal changed state"
        );
        evidence.push(json!({"name":mode.label(),"skipped_heads":skipped,
            "selected_token":selected,"decode_input_token":decode_token,"next_token":next,
            "traffic":expected.json(),"state":state(after)}));
        Ok(Sequence {
            prefill,
            decode,
            selected,
            next,
        })
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
            "fixed tiny fixture extent differs"
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
        let plan = RecordedQwen35Plan::new(&binding, CAPACITY, reads)?;
        let checker = NativeExecutionProfile::Qwen35Probe
            .finite_check_spec()
            .context("finite API missing")?;
        let r = plan.checked_requirements(Some(checker))?;
        let norm_dispatches = 1 + c.dim.div_ceil(4096) as u64;
        let mut arithmetic_dispatches = norm_dispatches;
        let mut arithmetic_submissions = 1;
        let mut scaled_tiles = 0;
        for block in binding.blocks() {
            let (core_output, core_dispatches, core_submissions) = match &block.attention {
                Attention::Full(value) => (
                    &value.output,
                    norm_dispatches + 3 * (c.n_heads + c.n_kv_heads) as u64 + 4 * c.n_heads as u64,
                    4,
                ),
                Attention::DeltaNet(value) => (&value.output, norm_dispatches + 4, 2),
            };
            arithmetic_dispatches +=
                core_dispatches + norm_dispatches + c.feed_forward.div_ceil(4096) as u64;
            arithmetic_submissions += core_submissions + 2;
            for matrix in [core_output, &block.ffn.down] {
                let p = PackedRowPlan::new(matrix, reads)?;
                scaled_tiles += p.rows().div_ceil(p.tile_rows()) as u64;
            }
        }
        let full = Traffic {
            tiles: r.packed_tiles_per_step,
            bytes: r.packed_bytes_per_step,
            dispatches: r.packed_tiles_per_step
                + scaled_tiles
                + arithmetic_dispatches
                + r.finite_dispatches_per_step,
            submissions: r.packed_tiles_per_step
                + arithmetic_submissions
                + r.finite_submissions_per_step,
            status_bytes: r.finite_status_bytes_per_step + r.delta_status_bytes_per_step,
        };
        let head = PackedRowPlan::new(binding.output(), reads)?;
        let head_tiles = head.rows().div_ceil(head.tile_rows()) as u64;
        let head_checks = 1 + c.vocab.div_ceil(checker.max_elements) as u64;
        let advance = Traffic {
            tiles: delta(full.tiles, head_tiles)?,
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
        };
        let embedding_bytes = PackedRowPlan::new(binding.embedding(), reads)?.row_bytes() as u64;
        let budget = MemoryBudget::new(DEVICE_BYTES);
        let limits = ModelLimits {
            device_bytes: DEVICE_BYTES,
            total_process_bytes: PROCESS_BYTES,
            process_headroom_bytes: HEADROOM_BYTES,
        };
        let admission = plan.admit(&gguf, r, limits, &budget)?;
        // Eight saved full vectors, one transient encoded output, bounded JSON
        // and bookkeeping are covered conservatively by sixteen vectors + 1 MiB.
        let extras = (16 * 512 * 4 + MAX_RECEIPT + (1 << 20)) as u64;
        let total = admission
            .total_process_bytes
            .checked_add(extras)
            .context("harness forecast overflow")?;
        ensure!(
            total <= PROCESS_BYTES,
            "owner plus harness exceeds process limit"
        );
        let forecast = json!({"source_file_bytes":FIXTURE_BYTES,"device_charge_bytes":r.device_charge_bytes,
            "resident_tensor_bytes":r.resident_tensor_bytes,"packed_tile_bytes":r.packed_tile_bytes,
            "auxiliary_bytes":r.auxiliary_bytes,"host_payload_bytes":r.host_payload_bytes,
            "recording_metadata_bytes":r.recording_metadata_bytes,"model_metadata_bytes":r.model_metadata_bytes,
            "process_headroom_bytes":HEADROOM_BYTES,"harness_extra_bytes":extras,
            "total_process_bytes":total,"process_ceiling_bytes":PROCESS_BYTES,
            "device_ceiling_bytes":DEVICE_BYTES,"reset_write_bytes":r.reset_write_bytes,
            "full_step":full.json(),"state_advance":advance.json(),"cancelled_embedding_bytes":embedding_bytes});
        if !a.native {
            println!(
                "{}",
                json!({"schema_version":1,"execution":"metadata-only","qualified":false,
                "capacity":CAPACITY,"prompt":PROMPT,"work_seconds":WORK_SECONDS,
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
        let mut flags = json!({"exact_prefill_and_decode":false,"lower_packed_traffic":false,
            "stale_ticket_refusal":false,"cancel_after_embedding_fence":false,
            "poisoned_retry_refusal":false,"cancel_reset_replay":false,
            "final_reset":false,"aggregate_accounting":false});
        let result = (|| -> Result<()> {
            let mut owner = plan.open_checked(&a.render,gguf,checker,limits,&budget,healthy.clone(),
                &mut || deadline(end))?;
            let body = (|| -> Result<()> {
                let prepared = owner.state()?;
                live(prepared,r,0)?;
                ensure!(prepared.packed.completed_tiles == 0 && prepared.native.dispatch_count == 0
                    && prepared.logits_read_bytes == 0 && prepared.reset_write_bytes == 0,
                    "eager token or reset activity");
                evidence.push(json!({"name":"prepared","state":state(prepared)}));
                let baseline = sequence(&mut owner,Mode::Baseline,None,r,full,advance,end,&mut evidence)?;
                reset(&mut owner,r)?;
                let optimized = sequence(&mut owner,Mode::Advance,Some(&baseline),r,full,advance,end,&mut evidence)?;
                flags["exact_prefill_and_decode"] = json!(true);
                flags["lower_packed_traffic"] = json!(true);
                reset(&mut owner,r)?;
                let mixed = sequence(&mut owner,Mode::StaleTicket,Some(&baseline),r,full,advance,end,&mut evidence)?;
                flags["stale_ticket_refusal"] = json!(true);
                reset(&mut owner,r)?;
                let before = owner.state()?;
                let mut callbacks = 0;
                let cancelled = owner.advance_checked(PROMPT[0], &mut || {
                    deadline(end)?;
                    callbacks += 1;
                    // Pre-invalidation, before read/upload/submit, after fence.
                    ensure!(callbacks != 5, "intentional after-embedding-fence cancellation");
                    Ok(())
                });
                let after = owner.state()?;
                ensure!(cancelled.is_err() && callbacks == 5 && after.phase == ModelPhase::Embedding
                    && after.committed_positions == 0 && after.poisoned && after.process_healthy
                    && !after.native.poisoned && after.native.weight_live_bytes == 0
                    && after.native.auxiliary_live_bytes == 0
                    && after.native.tensor_live_bytes == r.resident_tensor_bytes
                    && after.budget_current_bytes == r.device_charge_bytes,
                    "cancellation did not preserve poisoned unpublished fenced prefix");
                let cancelled_traffic = Traffic { tiles:1,bytes:embedding_bytes,dispatches:1,submissions:1,status_bytes:0 };
                accounting(before,after,cancelled_traffic,0)?;
                flags["cancel_after_embedding_fence"] = json!(true);
                ensure!(owner.advance_checked(PROMPT[0],&mut || deadline(end)).is_err()
                    && no_effect(after,owner.state()?)
                    && owner.step_checked(PROMPT[0],&mut || deadline(end)).is_err()
                    && no_effect(after,owner.state()?), "poisoned retry changed owner");
                flags["poisoned_retry_refusal"] = json!(true);
                evidence.push(json!({"name":"cancelled-embedding","state":state(after)}));
                reset(&mut owner,r)?;
                let recovered = sequence(&mut owner,Mode::Advance,Some(&baseline),r,full,advance,end,&mut evidence)?;
                flags["cancel_reset_replay"] = json!(true);
                reset(&mut owner,r)?;
                let final_state = owner.state()?;
                live(final_state,r,0)?;
                ensure!(final_state.reset_write_bytes == 5*r.reset_write_bytes, "final reset total differs");
                flags["final_reset"] = json!(true);
                accounting(prepared,final_state,full.times(11).plus(advance.times(5)).plus(cancelled_traffic),
                    9*512*4)?;
                flags["aggregate_accounting"] = json!(true);
                evidence.push(json!({"name":"final-reset","state":state(final_state)}));
                saved.extend([("baseline",baseline),("advance",optimized),("mixed",mixed),("recovered",recovered)]);
                Ok(())
            })();
            if body.is_err() {
                if let Ok(snapshot) = owner.state() { evidence.push(json!({"name":"failed","state":state(snapshot)})); }
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
            ensure!(budget.current_bytes() == 0 && healthy.load(Ordering::Acquire), "post-retirement health/budget failed");
            deadline(end)
        });
        let result = result.and_then(|()| {
            for (label, values) in &saved {
                logits_file(
                    &a.output.join(format!("{label}-prefill.f32le")),
                    &values.prefill,
                )?;
                logits_file(
                    &a.output.join(format!("{label}-decode.f32le")),
                    &values.decode,
                )?;
            }
            Ok(())
        });
        let selections: Vec<Value> = saved
            .iter()
            .map(|(label, value)| {
                json!({"name":label,
            "selected_token":value.selected,"next_token":value.next})
            })
            .collect();
        let receipt = json!({"schema_version":1,"execution":"native-qwen35-state-advance-equivalence",
            "passed":result.is_ok(),"qualified":false,"runtime_qualified":false,
            "independent_sequence_accuracy_qualified":false,"normal_antimatter_qualified":false,
            "comparison":"complete finite logits bitwise against native full-head sequence only",
            "fixture_sha256_expected":FIXTURE_SHA256,"fixture_sha256_verified_by_probe":false,
            "model_path":a.model,"source_file_bytes":FIXTURE_BYTES,"capacity":CAPACITY,"prompt":PROMPT,
            "vocab":512,"completed_positions_expected":16,"full_head_steps_expected":11,
            "state_advances_expected":5,"cancelled_embedding_tiles_expected":1,
            "work_seconds":WORK_SECONDS,"cleanup_seconds":CLEANUP_SECONDS,
            "forecast":forecast,"flags":flags,"selections":selections,"evidence":evidence,
            "retirement":retirement,"budget_after":budget.current_bytes(),
            "process_healthy":healthy.load(Ordering::Acquire),"elapsed_seconds":started.elapsed().as_secs_f64(),
            "error":result.as_ref().err().map(|error|format!("{error:#}"))});
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
