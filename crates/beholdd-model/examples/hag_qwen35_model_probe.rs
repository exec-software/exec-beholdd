//! STAGED ONLY. Explicit whole-graph diagnostic; no runtime/serving admission.
//! Required feature: hag-native. External cgroup/thermal/OOM/hard deadline guards
//! and frozen source/model/binary provenance are mandatory. No model discovery.
#[cfg(target_os = "linux")]
mod probe {
    use anyhow::{ensure, Context, Result};
    use beholdd_backend::hag::native_execution::{MemoryBudget, NativeExecutionProfile};
    use beholdd_format::{Gguf, RangeReader};
    use beholdd_model::qwen35::config::{Attention, Binding};
    use beholdd_model::qwen35::recorded_model::{
        ModelLimits, ModelStateSnapshot, RecordedQwen35Plan,
    };
    use beholdd_model::qwen35::recorded_weights::PackedRowPlan;
    use beholdd_model::qwen35::weights::ReadLimits;
    use serde_json::{json, Value};
    use std::{
        collections::BTreeMap,
        env,
        fs::{self, OpenOptions},
        io::Write,
        path::PathBuf,
        sync::{
            atomic::{AtomicBool, Ordering},
            Arc,
        },
        time::{Duration, Instant},
    };

    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    const ACTUAL_BYTES: u64 = 16_547_399_872;
    const MAX_RECEIPT: usize = 128 << 10;
    struct Args {
        native: bool,
        case: String,
        model: PathBuf,
        render: PathBuf,
        output: PathBuf,
        token: u32,
        seconds: u64,
        cleanup_seconds: u64,
        process_bytes: u64,
        headroom_bytes: u64,
        device_bytes: u64,
    }
    fn args() -> Result<Args> {
        let mut m = BTreeMap::new();
        let mut it = env::args().skip(1);
        while let Some(k) = it.next() {
            ensure!(
                [
                    "--mode",
                    "--case",
                    "--model",
                    "--render-node",
                    "--output-dir",
                    "--token",
                    "--deadline-secs",
                    "--cleanup-secs",
                    "--process-mib",
                    "--headroom-mib",
                    "--device-mib"
                ]
                .contains(&k.as_str()),
                "unknown option {k}"
            );
            let v = it.next().context("missing option value")?;
            ensure!(m.insert(k, v).is_none(), "duplicate option");
        }
        let mut take = |k: &str| m.remove(k).with_context(|| format!("required {k}"));
        let mode = take("--mode")?;
        ensure!(
            mode == "preflight" || mode == "native",
            "mode must be preflight/native"
        );
        let case = take("--case")?;
        ensure!(
            case == "tiny" || case == "openmythos",
            "case must be tiny/openmythos"
        );
        let model = PathBuf::from(take("--model")?);
        let render = PathBuf::from(take("--render-node")?);
        let output = PathBuf::from(take("--output-dir")?);
        let token = take("--token")?.parse()?;
        let seconds = take("--deadline-secs")?.parse()?;
        let cleanup_seconds = take("--cleanup-secs")?.parse()?;
        let process_mib: u64 = take("--process-mib")?.parse()?;
        let headroom_mib: u64 = take("--headroom-mib")?.parse()?;
        let device_mib: u64 = take("--device-mib")?.parse()?;
        ensure!(
            (1..=600).contains(&seconds) && (1..=120).contains(&cleanup_seconds),
            "deadline bounds"
        );
        ensure!(
            (1..=20480).contains(&process_mib)
                && (128..=1024).contains(&headroom_mib)
                && (1..=512).contains(&device_mib),
            "process/headroom/device ceilings exceeded"
        );
        let suffix = render
            .file_name()
            .and_then(|v| v.to_str())
            .and_then(|v| v.strip_prefix("renderD"))
            .unwrap_or("");
        ensure!(
            render.parent() == Some(std::path::Path::new("/dev/dri"))
                && !suffix.is_empty()
                && suffix.bytes().all(|b| b.is_ascii_digit()),
            "explicit render node required"
        );
        ensure!(
            model.is_absolute() && output.is_absolute(),
            "absolute model/output paths required"
        );
        Ok(Args {
            native: mode == "native",
            case,
            model,
            render,
            output,
            token,
            seconds,
            cleanup_seconds,
            process_bytes: process_mib * MIB,
            headroom_bytes: headroom_mib * MIB,
            device_bytes: device_mib * MIB,
        })
    }
    fn deadline(t: Instant) -> Result<()> {
        ensure!(Instant::now() < t, "cooperative deadline exceeded");
        Ok(())
    }
    fn state(s: ModelStateSnapshot) -> Value {
        json!({"phase":format!("{:?}",s.phase),"position":s.committed_positions,
        "capacity":s.capacity,"poisoned":s.poisoned,"process_healthy":s.process_healthy,
        "source_attempts":s.source.attempts,"source_bytes":s.source.returned_bytes,
        "source_peak_bytes":s.source.peak_bytes,"packed_read_attempts":s.packed.read_attempts,
        "packed_read_bytes":s.packed.returned_read_bytes,"packed_upload_bytes":s.packed.uploaded_bytes,
        "completed_tiles":s.packed.completed_tiles,"reset_write_bytes":s.reset_write_bytes,
        "logits_read_bytes":s.logits_read_bytes,"budget_current_bytes":s.budget_current_bytes,
        "expected_budget_bytes":s.budget_expected_bytes,"native":{
            "dispatches":s.native.dispatch_count,"dispatch_attempts":s.native.dispatch_attempts,
            "logical_submissions":s.native.logical_submissions,"host_write_bytes":s.native.host_write_bytes,
            "host_read_bytes":s.native.host_read_bytes,"uploaded_weight_bytes":s.native.uploaded_weight_bytes,
            "weight_peak_bytes":s.native.weight_peak_bytes,"tensor_peak_bytes":s.native.tensor_peak_bytes,
            "auxiliary_peak_bytes":s.native.auxiliary_peak_bytes,"poisoned":s.native.poisoned}})
    }
    fn no_effect(a: ModelStateSnapshot, b: ModelStateSnapshot) -> bool {
        a.phase == b.phase
            && a.committed_positions == b.committed_positions
            && a.poisoned == b.poisoned
            && a.process_healthy == b.process_healthy
            && a.source == b.source
            && a.packed == b.packed
            && a.native == b.native
            && a.reset_write_bytes == b.reset_write_bytes
            && a.logits_read_bytes == b.logits_read_bytes
            && a.budget_current_bytes == b.budget_current_bytes
    }
    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }
    fn vector(n: usize) -> Result<Vec<f32>> {
        let mut v = Vec::new();
        v.try_reserve_exact(n)?;
        v.resize(n, 0.0);
        Ok(v)
    }
    fn greedy(values: &[f32]) -> Result<usize> {
        ensure!(
            !values.is_empty() && values.iter().all(|v| v.is_finite()),
            "invalid complete logits"
        );
        let mut best = 0;
        for i in 1..values.len() {
            if values[i] > values[best] {
                best = i;
            }
        }
        Ok(best) // Strict > gives the lowest token ID on an exact tie.
    }
    fn output(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
        let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        Ok(())
    }
    pub fn run() -> Result<()> {
        let a = args()?;
        let started = Instant::now();
        let request_deadline = started + Duration::from_secs(a.seconds);
        ensure!(
            env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(std::ffi::OsStr::new("0")),
            "HAG_TRACE_SUBMISSION must be exactly0"
        );
        ensure!(
            fs::metadata(&a.model)?.is_file(),
            "model must be a regular file"
        );
        let gguf = Arc::new(Gguf::open(&a.model)?); // Header/index only before admit.
        let binding = Binding::load(&gguf)?;
        let c = binding.config();
        let vocab = c.vocab;
        ensure!((a.token as usize) < vocab, "input token outside vocabulary");
        let file_bytes = (0..gguf.shard_count()).try_fold(0u64, |n, i| {
            n.checked_add(gguf.shard_len(i).context("missing shard extent")?)
                .context("file extent overflow")
        })?;
        let reads = if a.case == "openmythos" {
            ensure!(
                file_bytes == ACTUAL_BYTES
                    && c.n_layers == 64
                    && c.dim == 5120
                    && c.feed_forward == 17408
                    && vocab == 248320
                    && c.n_heads == 24
                    && c.n_kv_heads == 4
                    && c.key_head_dim == 256
                    && c.rope_dim == 64
                    && c.ssm_group_count == 16
                    && c.ssm_time_step_rank == 48
                    && c.ssm_state_size == 128,
                "actual checkpoint geometry/extent differs"
            );
            ensure!(
                a.headroom_bytes == GIB
                    && a.process_bytes == 20 * GIB
                    && a.device_bytes == 512 * MIB,
                "actual gate requires explicit 20GiB total/1GiB headroom/512MiB device policy"
            );
            ReadLimits {
                tile_rows: 4096,
                encoded_bytes: 64 * MIB as usize,
                decoded_bytes: 64 * MIB as usize,
            }
        } else {
            ensure!(
                file_bytes <= 4 * MIB
                    && c.n_layers == 4
                    && c.dim == 256
                    && c.feed_forward == 256
                    && vocab == 512
                    && c.n_heads == 2
                    && c.n_kv_heads == 1
                    && c.key_head_dim == 256
                    && c.ssm_group_count == 1
                    && c.ssm_time_step_rank == 2
                    && c.ssm_state_size == 128,
                "tiny fixture geometry differs"
            );
            ensure!(
                a.process_bytes == 512 * MIB
                    && a.headroom_bytes == 128 * MIB
                    && a.device_bytes == 8 * MIB,
                "tiny gate requires explicit 512MiB total/128MiB headroom/8MiB device policy"
            );
            ReadLimits {
                tile_rows: 64,
                encoded_bytes: MIB as usize,
                decoded_bytes: 4 * MIB as usize,
            }
        };
        let plan = RecordedQwen35Plan::new(&binding, 1, reads)?;
        let checker = NativeExecutionProfile::Qwen35Probe
            .finite_check_spec()
            .context("missing qualified-stage finite API")?;
        let r = plan.checked_requirements(Some(checker))?;
        let norm_dispatches = 1 + c.dim.div_ceil(4096) as u64;
        let mut arithmetic_dispatches = norm_dispatches; // Final norm.
        let mut arithmetic_submissions = 1u64;
        let mut scaled_tiles = 0u64;
        for block in binding.blocks() {
            let (core_output, core_dispatches, core_submissions) = match &block.attention {
                Attention::Full(v) => (
                    &v.output,
                    norm_dispatches + 3 * (c.n_heads + c.n_kv_heads) as u64 + 4 * c.n_heads as u64,
                    4,
                ),
                Attention::DeltaNet(v) => (&v.output, norm_dispatches + 4, 2),
            };
            arithmetic_dispatches +=
                core_dispatches + norm_dispatches + c.feed_forward.div_ceil(4096) as u64;
            arithmetic_submissions += core_submissions + 2;
            for matrix in [core_output, &block.ffn.down] {
                let rows = PackedRowPlan::new(matrix, reads)?;
                scaled_tiles += rows.rows().div_ceil(rows.tile_rows()) as u64;
            }
        }
        // All dimensions are fixed by the two admitted cases above. These are
        // source-derived exact successful per-graph counts, never learned from a run.
        let dispatches_per_graph = r.packed_tiles_per_step
            + scaled_tiles
            + arithmetic_dispatches
            + r.finite_dispatches_per_step;
        let submissions_per_graph =
            r.packed_tiles_per_step + arithmetic_submissions + r.finite_submissions_per_step;
        let budget = MemoryBudget::new(a.device_bytes);
        let limits = ModelLimits {
            device_bytes: a.device_bytes,
            total_process_bytes: a.process_bytes,
            process_headroom_bytes: a.headroom_bytes,
        };
        let admission = plan.admit(&gguf, r, limits, &budget)?;
        // Owner counts one caller logits vector; this harness retains three plus an
        // encoded output, bounded receipt and local bookkeeping. Admit extras too.
        let extras = (3 * vocab * 4 + MAX_RECEIPT + (1 << 20)) as u64;
        let total = admission
            .total_process_bytes
            .checked_add(extras)
            .context("harness forecast overflow")?;
        ensure!(
            total <= a.process_bytes,
            "whole-file+owner+harness exceeds process ceiling"
        );
        let forecast = json!({"source_file_bytes":file_bytes,"source_cache_bytes":0,
        "device_charge_bytes":r.device_charge_bytes,"resident_tensor_bytes":r.resident_tensor_bytes,
        "packed_tile_bytes":r.packed_tile_bytes,"auxiliary_bytes":r.auxiliary_bytes,
        "host_payload_bytes":r.host_payload_bytes,"recording_metadata_bytes":r.recording_metadata_bytes,
        "model_metadata_bytes":r.model_metadata_bytes,"process_headroom_bytes":a.headroom_bytes,
        "harness_extra_bytes":extras,"total_process_bytes":total,"process_ceiling_bytes":a.process_bytes,
        "packed_tiles_per_step":r.packed_tiles_per_step,"packed_bytes_per_step":r.packed_bytes_per_step,
        "finite_checks_per_step":r.finite_checks_per_step,"finite_status_bytes_per_step":r.finite_status_bytes_per_step,
        "delta_status_bytes_per_step":r.delta_status_bytes_per_step,"reset_write_bytes":r.reset_write_bytes,
        "physical_dispatches_per_graph":dispatches_per_graph,"logical_submissions_per_graph":submissions_per_graph,
        "tile_rows":reads.tile_rows,"encoded_bytes":reads.encoded_bytes,"decoded_bytes":reads.decoded_bytes});
        if !a.native {
            println!(
                "{}",
                json!({"schema_version":1,"execution":"metadata-only", "case":a.case,
            "token":a.token,"vocab":vocab,"forecast":forecast,"budget_after":budget.current_bytes()})
            );
            return Ok(());
        }
        deadline(request_deadline)?;
        fs::create_dir(&a.output)?; // Exclusive task-owned directory; never overwrites.
        let healthy = Arc::new(AtomicBool::new(true));
        let mut first = vector(vocab)?;
        let mut scratch = vector(vocab)?;
        let mut evidence = Vec::<Value>::new();
        let mut retired = json!(null);
        let mut flags = json!({"capacity_refusal":false,"invalid_token_refusal":false,"stale_ticket_refusal":false,
        "reset_replay_exact":false,"read_cancel_poison":false,"poisoned_retry_refusal":false,
        "cancelled_ticket_refusal":false,"cancel_recovery_exact":false,"accounting_verified":false});
        let result = (|| -> Result<()> {
            let mut owner = plan.open_checked(
                &a.render,
                gguf,
                checker,
                limits,
                &budget,
                healthy.clone(),
                &mut || deadline(request_deadline),
            )?;
            let body = (|| -> Result<()> {
                let prepared = owner.state()?;
                evidence.push(json!({"name":"prepared","state":state(prepared)}));
                ensure!(
                    prepared.committed_positions == 0
                        && prepared.packed.completed_tiles == 0
                        && prepared.native.dispatch_count == 0
                        && prepared.logits_read_bytes == 0,
                    "eager token execution"
                );
                let first_ticket =
                    owner.step_checked(a.token, &mut || deadline(request_deadline))?;
                owner.read_logits(&first_ticket, &mut first, &mut || {
                    deadline(request_deadline)
                })?;
                greedy(&first)?;
                let before = owner.state()?;
                ensure!(
                    owner
                        .step_checked(a.token, &mut || deadline(request_deadline))
                        .is_err()
                        && no_effect(before, owner.state()?),
                    "capacity refusal changed owner"
                );
                flags["capacity_refusal"] = json!(true);
                ensure!(
                    owner
                        .step_checked(vocab as u32, &mut || deadline(request_deadline))
                        .is_err()
                        && no_effect(before, owner.state()?),
                    "invalid token refusal changed owner"
                );
                flags["invalid_token_refusal"] = json!(true);
                evidence.push(json!({"name":"first","state":state(before)}));
                let cleanup = Instant::now() + Duration::from_secs(a.cleanup_seconds);
                owner.reset_checked(&mut || deadline(cleanup))?;
                let reset = owner.state()?;
                ensure!(
                    reset.committed_positions == 0
                        && !reset.poisoned
                        && reset.reset_write_bytes == r.reset_write_bytes,
                    "first reset incomplete"
                );
                ensure!(
                    owner
                        .read_logits(&first_ticket, &mut scratch, &mut || deadline(
                            request_deadline
                        ))
                        .is_err()
                        && no_effect(reset, owner.state()?),
                    "stale ticket changed owner"
                );
                flags["stale_ticket_refusal"] = json!(true);
                let replay_ticket =
                    owner.step_checked(a.token, &mut || deadline(request_deadline))?;
                owner.read_logits(&replay_ticket, &mut scratch, &mut || {
                    deadline(request_deadline)
                })?;
                ensure!(same_bits(&first, &scratch), "physical reset replay differs");
                flags["reset_replay_exact"] = json!(true);
                let before_cancel = owner.state()?;
                let mut callbacks = 0usize;
                let refusal = owner.read_logits(&replay_ticket, &mut scratch, &mut || {
                    deadline(request_deadline)?;
                    callbacks += 1;
                    ensure!(
                        callbacks != 2,
                        "intentional after-first-logit-chunk cancellation"
                    );
                    Ok(())
                });
                let cancelled = owner.state()?;
                ensure!(
                    refusal.is_err()
                        && callbacks == 2
                        && cancelled.poisoned
                        && cancelled.process_healthy
                        && cancelled.committed_positions == 1
                        && !cancelled.native.poisoned
                        && cancelled.logits_read_bytes - before_cancel.logits_read_bytes
                            == (vocab.min(16384) * 4) as u64
                        && cancelled.source == before_cancel.source
                        && cancelled.packed == before_cancel.packed
                        && cancelled.native.dispatch_count == before_cancel.native.dispatch_count,
                    "read cancellation did not preserve truthful unpublished partial transfer"
                );
                flags["read_cancel_poison"] = json!(true);
                ensure!(
                    owner
                        .step_checked(a.token, &mut || deadline(request_deadline))
                        .is_err()
                        && no_effect(cancelled, owner.state()?),
                    "poisoned retry changed owner"
                );
                flags["poisoned_retry_refusal"] = json!(true);
                ensure!(
                    owner
                        .read_logits(&replay_ticket, &mut scratch, &mut || deadline(
                            request_deadline
                        ))
                        .is_err()
                        && no_effect(cancelled, owner.state()?),
                    "cancelled ticket was reused"
                );
                flags["cancelled_ticket_refusal"] = json!(true);
                evidence.push(json!({"name":"cancelled_read","state":state(cancelled)}));
                let cleanup = Instant::now() + Duration::from_secs(a.cleanup_seconds);
                owner.reset_checked(&mut || deadline(cleanup))?;
                let ticket = owner.step_checked(a.token, &mut || deadline(request_deadline))?;
                owner.read_logits(&ticket, &mut scratch, &mut || deadline(request_deadline))?;
                ensure!(same_bits(&first, &scratch), "cancellation recovery differs");
                flags["cancel_recovery_exact"] = json!(true);
                let final_state = owner.state()?;
                let expected_reads = 3 * (vocab * 4) as u64 + (vocab.min(16384) * 4) as u64;
                ensure!(
                    final_state.committed_positions == 1
                        && !final_state.poisoned
                        && final_state.reset_write_bytes == 2 * r.reset_write_bytes
                        && final_state.packed.completed_tiles == 3 * r.packed_tiles_per_step
                        && final_state.packed.read_attempts == 3 * r.packed_tiles_per_step
                        && final_state.packed.returned_read_bytes == 3 * r.packed_bytes_per_step
                        && final_state.packed.uploaded_bytes == 3 * r.packed_bytes_per_step
                        && final_state.native.uploaded_weight_bytes == 3 * r.packed_bytes_per_step
                        && final_state.logits_read_bytes == expected_reads
                        && final_state.native.host_read_bytes
                            == expected_reads
                                + 3 * (r.finite_status_bytes_per_step
                                    + r.delta_status_bytes_per_step)
                        && final_state.source.attempts
                            == prepared.source.attempts + 3 * r.packed_tiles_per_step
                        && final_state.source.returned_bytes
                            == prepared.source.returned_bytes + 3 * r.packed_bytes_per_step
                        && final_state.native.dispatch_count == 3 * dispatches_per_graph
                        && final_state.native.logical_submissions == 3 * submissions_per_graph
                        && final_state.native.dispatch_count
                            == final_state.native.dispatch_attempts
                        && final_state.native.weight_live_bytes == 0
                        && final_state.native.auxiliary_live_bytes == 0
                        && final_state.native.tensor_live_bytes == r.resident_tensor_bytes,
                    "exact successful traffic/lifetime accounting differs"
                );
                flags["accounting_verified"] = json!(true);
                evidence.push(json!({"name":"final","state":state(final_state)}));
                Ok(())
            })();
            if body.is_err() {
                if let Ok(s) = owner.state() {
                    evidence.push(json!({"name":"failed","state":state(s)}));
                }
            }
            let retirement = owner.retire(); // Runs on every post-construction result.
            retired = json!({"clean":retirement.clean,"expected_device_charge_bytes":retirement.expected_device_charge_bytes,
            "budget_before_bytes":retirement.budget_before_bytes,"budget_after_bytes":retirement.budget_after_bytes,
            "native_evidence_present":retirement.native.is_some()});
            body?;
            ensure!(
                retirement.clean
                    && retirement.native.is_some()
                    && retirement.budget_after_bytes == 0,
                "native owner did not retire cleanly"
            );
            Ok(())
        })();
        let result = result.and_then(|()| {
            ensure!(
                budget.current_bytes() == 0 && healthy.load(Ordering::Acquire),
                "budget or one-way process health failed after retirement"
            );
            deadline(request_deadline)
        });
        let mut receipt = json!({"schema_version":1,"execution":"native-qwen35-complete-graph",
        "case":a.case,"token":a.token,"vocab":vocab,"positions":1,"evaluated_graphs":3,
        "source_file_bytes":file_bytes,"model_path":a.model,"passed":result.is_ok(),
        "numerical_parity_compared":false,"qualified":false,"forecast":forecast,
        "flags":flags,"evidence":evidence,"retirement":retired,"budget_after":budget.current_bytes(),
        "process_healthy":healthy.load(Ordering::Acquire),"elapsed_seconds":started.elapsed().as_secs_f64(),
        "error":result.as_ref().err().map(|e|format!("{e:#}"))});
        if result.is_ok() {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(vocab * 4)?;
            for value in &first {
                bytes.extend_from_slice(&value.to_le_bytes());
            }
            output(&a.output.join("logits.f32le"), &bytes)?;
            receipt["greedy_token"] = json!(greedy(&first)?);
            receipt["logits_bytes"] = json!(bytes.len());
        }
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
