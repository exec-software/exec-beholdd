// Additive C512 qualification of attention batches recorded immediately before
// execution and retired after their fence. Existing context fixtures stay frozen.
use super::*;
use beholdd_backend::hag::native_execution::NativeRecordingFootprint;
use beholdd_memory::{available_memory, cgroup_memory};
use std::time::Instant;

pub(super) const CONTEXT: usize = 512;
const CHECK_POSITIONS: [usize; 12] = [0, 31, 63, 64, 127, 254, 255, 256, 340, 341, 510, 511];
const DEMAND_TENSORS: u64 = 127_378_944;
const PERSISTENT_AUX: u64 = 119_124;
const PEAK_AUX: u32 = 125_284;
const ALLOCATION: u64 = 760_778_752;
const PROCESS_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
const EARLY_STOP_BYTES: u64 = 1536 * 1024 * 1024;
const POST_ORACLE_POLICY: &str = "private-job-total-plus-native-additional-v1";
const PREPARATION_WRITES: u64 = PACKED
    + DEMAND_TENSORS
    + PERSISTENT_AUX
    + (SOURCE_BYTES - PACKED)
    + (LAYERS * CONTEXT * 128 * 4) as u64;

fn footprint() -> NativeRecordingFootprint {
    // Persist 28 FFN batches and the vocabulary head. Conservatively count one
    // transient attention AND embedding batch together, although their scopes do
    // not overlap. All packed/public allocations remain resident in either mode.
    NativeRecordingFootprint {
        batches: 31,
        dispatches: 364,
        weight_allocations: 234,
        tensor_allocations: 452,
        auxiliary_allocations: 119,
        retained_allocation_references: 404,
        noop_operands: 0,
        max_batch_commands: 54,
        max_batch_dispatches: 127,
    }
}

fn demand_config(shape: Shape, memory: DenseModelMemory) -> Result<NativeExecutionConfig> {
    inspect_geometry(shape, CONTEXT)?;
    ensure!(
        memory.packed_weight_bytes == PACKED
            && memory.resident_tensor_bytes == DEMAND_TENSORS
            && memory.largest_staging_bytes == STAGING,
        "on-demand logical plan costs differ"
    );
    let config = NativeExecutionConfig {
        weight_capacity_bytes: u32::try_from(PACKED)?,
        tensor_capacity_bytes: u32::try_from(DEMAND_TENSORS)?,
        auxiliary_capacity_bytes: PEAK_AUX,
    };
    ensure!(
        config.allocation_bytes()? == ALLOCATION && ALLOCATION <= MODEL_BUDGET,
        "on-demand hardware charge differs"
    );
    Ok(config)
}

fn checked_forecast(checkpoint_bytes: u64, prepared_metadata: u64) -> Result<u64> {
    // CPU KV is retired before GPU open. Counting both here remains conservative.
    // The model API counts retained on-demand preparations using actual layouts;
    // a separate 16-MiB allowance covers the remaining model/plan owner tables.
    [
        ALLOCATION,
        checkpoint_bytes,
        footprint().host_bytes()?,
        prepared_metadata,
        STAGING,
        (CHECK_POSITIONS.len() + 1) as u64 * VOCAB as u64 * 4,
        (LAYERS * 2 * CONTEXT * 8 * 128 * 4) as u64,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
    ]
    .into_iter()
    .try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .context("on-demand combined forecast overflows")
    })
}

fn admit_cost(required: u64, available: u64) -> Result<()> {
    ensure!(
        required <= PROCESS_BUDGET && required <= available,
        "on-demand forecast {required} exceeds process budget {PROCESS_BUDGET} or available {available}"
    );
    Ok(())
}

fn native_additional_forecast(prepared_metadata: u64) -> Result<u64> {
    // Existing checkpoint pages and retained CPU references are already part of
    // the full job forecast. This independent bound reserves future native work.
    [
        ALLOCATION,
        footprint().host_bytes()?,
        prepared_metadata,
        STAGING,
        VOCAB as u64 * 4,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
    ]
    .into_iter()
    .try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .context("on-demand native additional forecast overflows")
    })
}

fn validate_supervised_scope(
    unit: &str,
    stage: &str,
    membership: &str,
    status: &str,
) -> Result<()> {
    ensure!(
        matches!(stage, "first-token" | "context"),
        "invalid supervised stage"
    );
    let prefix = format!("hag-demand512-{stage}-");
    let suffix = unit.strip_prefix(&prefix).unwrap_or("");
    ensure!(
        !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()),
        "HAG_DEMAND_UNIT does not identify this supervised stage"
    );
    let mut memberships = membership.lines();
    let path = memberships
        .next()
        .and_then(|line| line.strip_prefix("0::"))
        .context("supervised job requires unified cgroup membership")?;
    ensure!(
        memberships.next().is_none()
            && path.starts_with('/')
            && path.ends_with(&format!("/{unit}.service"))
            && path
                .split('/')
                .skip(1)
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "unified cgroup membership does not match the supervised service"
    );
    let mut no_new_privileges = status.lines().filter_map(|line| {
        let (key, value) = line.split_once(':')?;
        (key == "NoNewPrivs").then_some(value.trim())
    });
    ensure!(
        no_new_privileges.next() == Some("1") && no_new_privileges.next().is_none(),
        "supervised job requires NoNewPrivs=1"
    );
    Ok(())
}

fn require_supervised_scope(stage: &str) -> Result<()> {
    validate_supervised_scope(
        &std::env::var("HAG_DEMAND_UNIT").context("missing HAG_DEMAND_UNIT")?,
        stage,
        &std::fs::read_to_string("/proc/self/cgroup")?,
        &std::fs::read_to_string("/proc/self/status")?,
    )
}

#[derive(Clone, Copy)]
struct ChargeSample {
    current: u64,
    peak: u64,
}

#[derive(Clone, Copy)]
struct PostOracleMemory {
    host_available: u64,
    ancestor_remaining: Option<u64>,
    before: Option<ChargeSample>,
    after: Option<ChargeSample>,
}

fn admit_post_oracle(required: u64, additional: u64, memory: PostOracleMemory) -> Result<()> {
    let before = memory
        .before
        .context("missing pre-sample private cgroup charge")?;
    let after = memory
        .after
        .context("missing post-sample private cgroup charge")?;
    let remaining = memory
        .ancestor_remaining
        .context("supervised job requires finite cgroup headroom")?;
    for sample in [before, after] {
        ensure!(
            sample.current <= sample.peak,
            "inconsistent private cgroup charge"
        );
        ensure!(
            sample.current < EARLY_STOP_BYTES && sample.peak < EARLY_STOP_BYTES,
            "private cgroup reached the unchanged early-stop limit"
        );
    }
    ensure!(
        after.peak >= before.peak,
        "private cgroup peak decreased while sampling"
    );
    // F is the total private job, including its full checkpoint allowance. Add
    // only this leaf's already charged bytes back to ancestor remaining capacity;
    // other jobs in ancestors never receive credit. The smaller bracketing sample
    // avoids crediting a concurrent increase. No page-cache residency is assumed.
    let total_allowance = remaining
        .checked_add(before.current.min(after.current))
        .context("private job total allowance overflows")?;
    ensure!(
        required <= PROCESS_BUDGET
            && required <= memory.host_available
            && required <= total_allowance,
        "post-oracle full forecast {required} exceeds process, host or private-job allowance"
    );
    ensure!(
        additional <= required && additional <= memory.host_available.min(remaining),
        "post-oracle native additional forecast {additional} exceeds live remaining headroom"
    );
    Ok(())
}

fn post_oracle_admission(stage: &str, required: u64, additional: u64) -> Result<()> {
    require_supervised_scope(stage)?;
    let sample = || -> Result<Option<ChargeSample>> {
        Ok(cgroup_memory()?.map(|memory| ChargeSample {
            current: memory.current_bytes(),
            peak: memory.peak_bytes(),
        }))
    };
    let before = sample()?;
    let available = available_memory()?;
    let after = sample()?;
    // Recheck scope after sampling so moving out of the named service refuses.
    require_supervised_scope(stage)?;
    admit_post_oracle(
        required,
        additional,
        PostOracleMemory {
            host_available: available.host_available_bytes(),
            ancestor_remaining: available.cgroup_available_bytes(),
            before,
            after,
        },
    )
}

fn attention_scratch_bytes(visible: usize) -> Result<u64> {
    ensure!(
        (1..=CONTEXT).contains(&visible),
        "on-demand position outside capacity"
    );
    // Scores/probabilities are disjoint inside shared scratch, with four bytes
    // reserved for alignment. Norm scratch dominates through visible position341.
    Ok((DIM as u64 * 4).max(visible as u64 * 12 + 4) + 12)
}

fn recording_bytes(visible: usize) -> Result<u64> {
    Ok(4 + LAYERS as u64 * attention_scratch_bytes(visible)?)
}

fn total_recording_bytes(cpu_steps: usize, replay: bool) -> Result<u64> {
    ensure!(
        (1..=CONTEXT).contains(&cpu_steps),
        "invalid evaluated position count"
    );
    let mut total = 0_u64;
    for visible in 1..=cpu_steps {
        total = total
            .checked_add(recording_bytes(visible)?)
            .context("recording total overflows")?;
    }
    if replay {
        total = total
            .checked_add(recording_bytes(1)?)
            .context("replay recording overflows")?;
    }
    Ok(total)
}

fn verify_step(
    before: NativeExecutionStats,
    after: NativeExecutionStats,
    visible: usize,
) -> Result<()> {
    ensure!(
        !after.poisoned
            && after.dispatch_count == before.dispatch_count + PHYSICAL_PER_STEP
            && after.dispatch_attempts == before.dispatch_attempts + PHYSICAL_PER_STEP
            && after.logical_submissions == before.logical_submissions + SUBMISSIONS_PER_STEP
            && after.host_write_bytes == before.host_write_bytes + recording_bytes(visible)?
            && after.host_read_bytes == before.host_read_bytes
            && after.uploaded_weight_bytes == before.uploaded_weight_bytes
            && after.allocation_bytes == ALLOCATION
            && after.weight_live_bytes == PACKED
            && after.tensor_live_bytes == DEMAND_TENSORS
            && after.auxiliary_live_bytes == PERSISTENT_AUX
            && after.auxiliary_peak_bytes
                == before
                    .auxiliary_peak_bytes
                    .max(PERSISTENT_AUX + attention_scratch_bytes(visible)?)
            && after.weight_peak_bytes == before.weight_peak_bytes
            && after.tensor_peak_bytes == before.tensor_peak_bytes,
        "on-demand step counters/residency differ"
    );
    Ok(())
}

fn check_logits(
    actual: &[f32],
    expected: &[f32],
    maximum: &mut f64,
    signs: &mut u64,
    zeros: &mut u64,
    selected: &mut Vec<u32>,
) -> Result<()> {
    let comparison = compare(actual, expected)?;
    *maximum = maximum.max(comparison.maximum);
    *signs += comparison.signs;
    *zeros += comparison.zeros;
    let choice = greedy(actual)?;
    ensure!(
        choice == greedy(expected)?,
        "on-demand greedy token differs"
    );
    selected.push(choice);
    Ok(())
}

pub(super) fn run(options: &Options) -> Result<()> {
    let path = options
        .model
        .as_ref()
        .context("on-demand requires checkpoint")?;
    check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
    let checkpoint = Gguf::open(path)?;
    check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
    let model = QwenModel::load(&checkpoint)?;
    ensure!(
        model.architecture() == QwenArchitecture::Qwen3,
        "dense qwen3 required"
    );
    let shape = Shape::from_model(&model);
    let plan = model.dense_model_plan_on_demand(CONTEXT, TILE_ROWS)?;
    let config = demand_config(shape, plan.memory())?;
    inspect_checkpoint(&checkpoint, shape)?;
    ensure!(
        plan.requires_on_demand_attention()
            && (plan.layers(), plan.dim(), plan.vocab(), plan.capacity())
                == (LAYERS, DIM, VOCAB, CONTEXT),
        "on-demand plan geometry differs"
    );
    let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |sum, path| {
        sum.checked_add(std::fs::metadata(path)?.len())
            .context("checkpoint size overflows")
    })?;
    let prepared_metadata = plan.on_demand_attention_host_metadata_bytes()?;
    let required = checked_forecast(checkpoint_bytes, prepared_metadata)?;
    let additional = native_additional_forecast(prepared_metadata)?;
    admit_cost(required, available_memory()?.effective_available_bytes())
        .context("initial remaining-memory admission")?;
    let first = options.mode == Mode::OnDemand512FirstToken;
    let checked_positions: &[usize] = if first { &[0] } else { &CHECK_POSITIONS };
    let cpu_steps = if first { 1 } else { CONTEXT };
    let evaluations = if first { 1 } else { CONTEXT + 1 };
    let planned_recording_bytes = total_recording_bytes(cpu_steps, !first)?;
    let planned_auxiliary_peak = PERSISTENT_AUX + attention_scratch_bytes(cpu_steps)?;
    let stage = if first { "first-token" } else { "context" };
    let mut report = json!({
        "execution": "not_requested", "recording_strategy": "on-demand-attention",
        "capacity": CONTEXT, "model_layers": LAYERS,
        "compared_positions": checked_positions, "planned_evaluations": evaluations,
        "planned_cpu_evaluations": cpu_steps,
        "packed_weight_bytes": PACKED, "source_weight_bytes": SOURCE_BYTES,
        "largest_staging_bytes": STAGING, "resident_tensor_bytes": DEMAND_TENSORS,
        "expected_hag_allocation_bytes": ALLOCATION, "hag_budget_bytes": MODEL_BUDGET,
        "persistent_auxiliary_bytes": PERSISTENT_AUX, "auxiliary_peak_bytes": PEAK_AUX,
        "persistent_batches": LAYERS + 1, "maximum_transient_attention_batches": 1,
        "recording_host_write_bytes_first_step": recording_bytes(1)?,
        "maximum_recording_host_write_bytes_per_step": recording_bytes(CONTEXT)?,
        "recording_host_write_bytes_formula": "4 + 28 * (max(4096, 12 * visible_positions + 4) + 12)",
        "planned_recording_host_write_bytes": planned_recording_bytes,
        "planned_auxiliary_peak_bytes": planned_auxiliary_peak,
        "preparation_host_write_bytes": PREPARATION_WRITES,
        "native_recording_metadata_bytes": footprint().host_bytes()?,
        "retained_attention_metadata_bytes": prepared_metadata,
        "combined_forecast_bytes": required, "process_budget_bytes": PROCESS_BUDGET,
        "post_oracle_admission_policy": POST_ORACLE_POLICY,
        "native_additional_forecast_bytes": additional,
        "retained_reference_bytes": checked_positions.len() * VOCAB * 4,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "fixture": "fixed token IDs 1..512; first-token mode checks token1 only; context mode checks full logits and exact CPU greedy at 0,31,63,64,127,254,255,256,340,341,510,511, refuses position512, resets and replays token1",
        "comparison_policy": "finite per-value tolerance and exact CPU greedy; sign/zero mismatches reported",
        "required_hag_trace_submission": "0", "external_process_and_thermal_limits_required": true,
    });
    if options.mode == Mode::InspectOnDemand512 {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    ensure!(
        matches!(
            options.mode,
            Mode::OnDemand512FirstToken | Mode::OnDemand512Context
        ),
        "invalid on-demand execution mode"
    );
    require_supervised_scope(stage).context("pre-oracle supervised-scope admission")?;
    let cpu_started = Instant::now();
    let oracle_source = CountingSource::new(&checkpoint);
    let mut expected = Vec::with_capacity(checked_positions.len());
    {
        let mut oracle = Oracle::with_capacity(&checkpoint, &oracle_source, shape, CONTEXT);
        for position in 0..cpu_steps {
            let logits = oracle.step((position + 1) as u32)?;
            if checked_positions.contains(&position) {
                ensure!(
                    logits.len() == VOCAB && logits.iter().all(|v| v.is_finite()),
                    "nonfinite CPU on-demand reference"
                );
                expected.push(logits);
            }
            if (position + 1) % 64 == 0 {
                println!("PROGRESS demand512 cpu_evaluations={}", position + 1);
            }
        }
    }
    let cpu_ms = cpu_started.elapsed().as_millis();
    let oracle_counts = oracle_source.counts();
    ensure!(
        expected.len() == checked_positions.len()
            && oracle_counts
                == (
                    cpu_steps as u64 * (PREPARATION_READS + 1),
                    cpu_steps as u64 * (SOURCE_BYTES + (DIM / 32 * 34) as u64),
                    STAGING,
                ),
        "on-demand CPU source/reference accounting differs"
    );
    // CPU KV is gone. Keep the conservative full forecast, but distinguish its
    // already charged private-job bytes from the native work still to allocate.
    post_oracle_admission(stage, required, additional)
        .context("post-oracle private-job admission before native open")?;
    let budget = MemoryBudget::new(MODEL_BUDGET);
    let prepare_started = Instant::now();
    let device = NativeExecutionDevice::open(
        options.device.as_ref().context("missing render node")?,
        &budget,
        config,
    )?;
    let source = CountingSource::new(&checkpoint);
    let mut resident = plan.prepare_on_demand_attention(
        &device,
        &source,
        DenseModelLimits {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: DEMAND_TENSORS,
            largest_staging_bytes: STAGING,
        },
    )?;
    let preparation_ms = prepare_started.elapsed().as_millis();
    let initial = device.stats()?;
    ensure!(
        initial.allocation_bytes == ALLOCATION
            && budget.current_bytes() == ALLOCATION
            && initial.weight_live_bytes == PACKED
            && initial.tensor_live_bytes == DEMAND_TENSORS
            && initial.auxiliary_live_bytes == PERSISTENT_AUX
            && initial.auxiliary_peak_bytes == PERSISTENT_AUX
            && initial.uploaded_weight_bytes == PACKED
            && initial.host_write_bytes == PREPARATION_WRITES
            && initial.host_read_bytes == 0
            && initial.dispatch_count == 0
            && initial.dispatch_attempts == 0
            && initial.logical_submissions == 0
            && !initial.poisoned
            && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING)
            && resident.records_attention_on_demand()
            && resident.capacity() == CONTEXT
            && resident.visible_positions() == 0
            && !resident.is_poisoned(),
        "on-demand preparation accounting/state differs"
    );
    let native_started = Instant::now();
    let mut actual = vec![f32::NAN; VOCAB];
    let mut maximum = 0.0_f64;
    let mut signs = 0u64;
    let mut zeros = 0u64;
    let mut selected = Vec::new();
    for position in 0..cpu_steps {
        let before = device.stats()?;
        resident.step(&device, (position + 1) as u32)?;
        verify_step(before, device.stats()?, position + 1)?;
        ensure!(
            resident.visible_positions() == position + 1
                && !resident.is_poisoned()
                && budget.current_bytes() == ALLOCATION,
            "on-demand cursor/charge differs"
        );
        if let Some(index) = checked_positions.iter().position(|&p| p == position) {
            resident.read_logits(&device, &mut actual)?;
            check_logits(
                &actual,
                &expected[index],
                &mut maximum,
                &mut signs,
                &mut zeros,
                &mut selected,
            )?;
            println!("PASS demand512 position={position} checked_values={VOCAB}");
        }
    }
    if !first {
        let full = device.stats()?;
        ensure!(
            resident.step(&device, (CONTEXT + 1) as u32).is_err()
                && device.stats()? == full
                && resident.visible_positions() == CONTEXT
                && !resident.is_poisoned(),
            "on-demand capacity refusal changed recording/transport/state"
        );
        resident.reset()?;
        ensure!(
            resident.visible_positions() == 0 && !resident.is_poisoned() && device.stats()? == full,
            "on-demand reset changed transport/state"
        );
        resident.step(&device, 1)?;
        verify_step(full, device.stats()?, 1)?;
        ensure!(
            resident.visible_positions() == 1 && !resident.is_poisoned(),
            "replay cursor differs"
        );
        resident.read_logits(&device, &mut actual)?;
        check_logits(
            &actual,
            &expected[0],
            &mut maximum,
            &mut signs,
            &mut zeros,
            &mut selected,
        )?;
    }
    let native_ms = native_started.elapsed().as_millis();
    let final_stats = device.stats()?;
    let comparisons = checked_positions.len() + usize::from(!first);
    ensure!(
        final_stats.dispatch_count == evaluations as u64 * PHYSICAL_PER_STEP
            && final_stats.dispatch_attempts == evaluations as u64 * PHYSICAL_PER_STEP
            && final_stats.logical_submissions == evaluations as u64 * SUBMISSIONS_PER_STEP
            && final_stats.host_read_bytes == comparisons as u64 * VOCAB as u64 * 4
            && final_stats.host_write_bytes == initial.host_write_bytes + planned_recording_bytes
            && final_stats.auxiliary_peak_bytes == planned_auxiliary_peak
            && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING)
            && selected.len() == comparisons,
        "on-demand final execution/traffic/source counters differ"
    );
    resident.reset()?;
    ensure!(
        device.stats()? == final_stats
            && resident.visible_positions() == 0
            && !resident.is_poisoned(),
        "on-demand final reset differs"
    );
    drop(resident);
    let retired = device.stats()?;
    ensure!(
        retired.weight_live_bytes == 0
            && retired.tensor_live_bytes == 0
            && retired.auxiliary_live_bytes == 0
            && !retired.poisoned,
        "on-demand resources did not retire"
    );
    drop(device);
    ensure!(budget.current_bytes() == 0, "on-demand HAG charge retained");
    report["execution"] = json!(options.mode.execution());
    report["evaluations"] = json!(evaluations);
    report["cpu_evaluations"] = json!(cpu_steps);
    report["checked_values"] = json!(comparisons * VOCAB);
    report["greedy_tokens"] = json!(selected);
    report["greedy_matches"] = json!(comparisons);
    report["max_absolute_error"] = json!(maximum);
    report["logit_sign_mismatches"] = json!(signs);
    report["logit_zero_mismatches"] = json!(zeros);
    report["capacity_refusal_checked"] = json!(!first);
    report["capacity_refused_without_dispatch"] = json!(!first);
    report["reset_replay_passed"] = json!(!first);
    report["physical_dispatches"] = json!(final_stats.dispatch_count);
    report["logical_submissions"] = json!(final_stats.logical_submissions);
    report["native_preparation_source_reads"] = json!(PREPARATION_READS);
    report["native_preparation_source_bytes"] = json!(SOURCE_BYTES);
    report["cpu_oracle_source_reads"] = json!(oracle_counts.0);
    report["cpu_oracle_source_bytes"] = json!(oracle_counts.1);
    report["actual_hag_allocation_bytes"] = json!(ALLOCATION);
    report["initial_host_write_bytes"] = json!(initial.host_write_bytes);
    report["recording_host_write_bytes"] = json!(planned_recording_bytes);
    report["actual_auxiliary_peak_bytes"] = json!(final_stats.auxiliary_peak_bytes);
    report["submission_tensor_transfer_bytes"] = json!(0);
    report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
    report["auxiliary_live_bytes_after_steps"] = json!(final_stats.auxiliary_live_bytes);
    report["cache_visible_positions_after_reset"] = json!(0);
    report["budget_after"] = json!(0);
    report["cpu_reference_ms"] = json!(cpu_ms);
    report["native_preparation_ms"] = json!(preparation_ms);
    report["native_execution_and_checks_ms"] = json!(native_ms);
    println!("HAG_DEMAND512_REPORT {}", serde_json::to_string(&report)?);
    println!("PASS hag-demand512 stage={stage} qualified=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn post_oracle_memory(current: u64) -> PostOracleMemory {
        let sample = ChargeSample {
            current,
            peak: current,
        };
        PostOracleMemory {
            host_available: 8 * 1024 * 1024 * 1024,
            ancestor_remaining: Some(PROCESS_BUDGET - current),
            before: Some(sample),
            after: Some(sample),
        }
    }

    #[test]
    fn post_oracle_total_and_additional_bounds_admit_cold_and_warm_checkpoint_charges() {
        let prepared = 64_960;
        let required = checked_forecast(639_447_744, prepared).unwrap();
        let additional = native_additional_forecast(prepared).unwrap();
        assert_eq!(
            required - additional,
            639_447_744
                + CHECK_POSITIONS.len() as u64 * VOCAB as u64 * 4
                + (LAYERS * 2 * CONTEXT * 8 * 128 * 4) as u64
        );
        assert!(additional >= ALLOCATION + STAGING + 144 * 1024 * 1024);
        // A cold oracle charges checkpoint pages to this job; a warm file may
        // remain charged elsewhere. Both retain the same complete file allowance.
        for current in [48 * 1024 * 1024, 694_206_464] {
            admit_post_oracle(required, additional, post_oracle_memory(current)).unwrap();
        }
        let cold = post_oracle_memory(694_206_464);
        assert!(admit_cost(required, cold.ancestor_remaining.unwrap()).is_err());
        assert!(native_additional_forecast(u64::MAX).is_err());
    }

    #[test]
    fn post_oracle_admission_refuses_pressure_missing_samples_and_invalid_bounds() {
        let required = checked_forecast(639_447_744, 64_960).unwrap();
        let additional = native_additional_forecast(64_960).unwrap();
        let baseline = post_oracle_memory(694_206_464);
        let sample = baseline.before.unwrap();
        for bad in [
            PostOracleMemory {
                host_available: required - 1,
                ..baseline
            },
            PostOracleMemory {
                ancestor_remaining: None,
                ..baseline
            },
            PostOracleMemory {
                before: None,
                ..baseline
            },
            PostOracleMemory {
                after: None,
                ..baseline
            },
            // Other jobs consuming a parent limit must never receive leaf credit.
            PostOracleMemory {
                ancestor_remaining: Some(required - sample.current - 1),
                ..baseline
            },
            // The static total fits, but unexpectedly high current use leaves too
            // little room for future native allocations.
            post_oracle_memory(EARLY_STOP_BYTES - 1),
            PostOracleMemory {
                ancestor_remaining: Some(u64::MAX),
                ..baseline
            },
            PostOracleMemory {
                before: Some(ChargeSample {
                    current: EARLY_STOP_BYTES,
                    peak: EARLY_STOP_BYTES,
                }),
                ..baseline
            },
            PostOracleMemory {
                after: Some(ChargeSample {
                    current: sample.current,
                    peak: EARLY_STOP_BYTES,
                }),
                ..baseline
            },
            PostOracleMemory {
                after: Some(ChargeSample {
                    current: sample.peak + 1,
                    peak: sample.peak,
                }),
                ..baseline
            },
            PostOracleMemory {
                after: Some(ChargeSample {
                    current: sample.current - 1,
                    peak: sample.peak - 1,
                }),
                ..baseline
            },
            // Credit the smaller bracketing current, not the later increase.
            PostOracleMemory {
                before: Some(ChargeSample {
                    current: 1,
                    peak: sample.peak,
                }),
                ..baseline
            },
        ] {
            assert!(admit_post_oracle(required, additional, bad).is_err());
        }
        assert!(admit_post_oracle(PROCESS_BUDGET + 1, additional, baseline).is_err());
        assert!(admit_post_oracle(required, required + 1, baseline).is_err());
        let ample_total = PostOracleMemory {
            ancestor_remaining: Some(additional - 1),
            before: Some(ChargeSample {
                current: required - additional + 1,
                peak: required - additional + 1,
            }),
            after: Some(ChargeSample {
                current: required - additional + 1,
                peak: required - additional + 1,
            }),
            ..baseline
        };
        assert!(admit_post_oracle(required, additional, ample_total).is_err());
    }

    #[test]
    fn post_oracle_total_credit_requires_exact_supervised_service_and_no_new_privileges() {
        for stage in ["first-token", "context"] {
            let unit = format!("hag-demand512-{stage}-123456");
            let membership = format!("0::/user.slice/user-1000.slice/{unit}.service\n");
            let status = "Name:\thag_model_probe\nNoNewPrivs:\t1\n";
            validate_supervised_scope(&unit, stage, &membership, status).unwrap();
            for bad_unit in [
                "",
                "hag-demand512-context-",
                "hag-demand512-context-1a",
                "hag-demand-first-token-1",
            ] {
                assert!(validate_supervised_scope(bad_unit, stage, &membership, status).is_err());
            }
            for bad_membership in [
                String::new(),
                format!("0::/{unit}.service/child\n"),
                "0::/other.service\n".to_owned(),
                format!("0::/../{unit}.service\n"),
                format!("0:://{unit}.service\n"),
                format!("0::/{unit}.service\n1:memory:/other\n"),
            ] {
                assert!(validate_supervised_scope(&unit, stage, &bad_membership, status).is_err());
            }
            for bad_status in [
                "",
                "NoNewPrivs:\t0\n",
                "NoNewPrivs:\t01\n",
                "NoNewPrivs: 1\nNoNewPrivs: 1\n",
            ] {
                assert!(validate_supervised_scope(&unit, stage, &membership, bad_status).is_err());
            }
            assert!(validate_supervised_scope(&unit, "other", &membership, status).is_err());
        }
    }

    #[test]
    fn demand_forecast_admits_only_exact_geometry_and_bounded_cost() {
        let mut shape = super::super::tests::shape();
        let memory = DenseModelMemory {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: DEMAND_TENSORS,
            largest_staging_bytes: STAGING,
        };
        assert_eq!(
            demand_config(shape, memory)
                .unwrap()
                .allocation_bytes()
                .unwrap(),
            ALLOCATION
        );
        shape.context = CONTEXT - 1;
        assert!(demand_config(shape, memory).is_err());
        assert!(demand_config(
            super::super::tests::shape(),
            DenseModelMemory {
                resident_tensor_bytes: DEMAND_TENSORS + 4,
                ..memory
            }
        )
        .is_err());
        assert!(checked_forecast(u64::MAX, 0).is_err());
        assert!(checked_forecast(639_447_744, u64::MAX).is_err());
        let cost = checked_forecast(639_447_744, 1024 * 1024).unwrap();
        assert!(cost < PROCESS_BUDGET);
        admit_cost(cost, cost).unwrap();
        assert!(admit_cost(cost, cost - 1).is_err());
        assert!(admit_cost(PROCESS_BUDGET + 1, u64::MAX).is_err());
    }

    #[test]
    fn demand_accounting_counts_one_transient_attention_and_new_recording_writes() {
        let layer_aux = DIM as u64 * 4 + 12;
        let head_aux = DIM as u64 * 4 + 4;
        assert_eq!(PERSISTENT_AUX, LAYERS as u64 * layer_aux + head_aux);
        assert_eq!(
            u64::from(PEAK_AUX),
            PERSISTENT_AUX + attention_scratch_bytes(CONTEXT).unwrap()
        );
        for visible in [1, 255, 256, 257, 341] {
            assert_eq!(attention_scratch_bytes(visible).unwrap(), 4108);
            assert_eq!(recording_bytes(visible).unwrap(), 115_028);
        }
        assert_eq!(attention_scratch_bytes(342).unwrap(), 4120);
        assert_eq!(recording_bytes(342).unwrap(), 115_364);
        assert_eq!(attention_scratch_bytes(512).unwrap(), 6160);
        assert_eq!(recording_bytes(512).unwrap(), 172_484);
        for invalid in [0, 513, usize::MAX] {
            assert!(attention_scratch_bytes(invalid).is_err());
            assert!(recording_bytes(invalid).is_err());
            assert!(total_recording_bytes(invalid, false).is_err());
        }
        assert_eq!(PREPARATION_WRITES, 768_333_652);
        assert_eq!(footprint().batches, LAYERS as u64 + 3);
        assert_eq!(footprint().dispatches, LAYERS as u64 * 7 + 40 + 127 + 1);
        assert_eq!(
            footprint().auxiliary_allocations,
            LAYERS as u64 * 4 + 2 + 4 + 1
        );
        assert_eq!(
            footprint().retained_allocation_references,
            LAYERS as u64 * 12 + 44 + 21 + 3
        );
        assert!(footprint().host_bytes().unwrap() < 1024 * 1024);
        assert_eq!((CONTEXT as u64 + 1) * PHYSICAL_PER_STEP, 1_945_809);
        assert_eq!((CONTEXT as u64 + 1) * SUBMISSIONS_PER_STEP, 29_754);
        assert_eq!(total_recording_bytes(CONTEXT, true).unwrap(), 63_950_580);
        assert_eq!(total_recording_bytes(1, false).unwrap(), 115_028);
        assert_eq!((CHECK_POSITIONS.len() + 1) * VOCAB, 1_975_168);
        assert_eq!((CHECK_POSITIONS.len() + 1) * VOCAB * 4, 7_900_672);
    }

    #[test]
    fn demand_step_rejects_retained_scratch_extra_traffic_and_missing_work() {
        let before = NativeExecutionStats {
            allocation_bytes: ALLOCATION,
            weight_live_bytes: PACKED,
            tensor_live_bytes: DEMAND_TENSORS,
            auxiliary_live_bytes: PERSISTENT_AUX,
            auxiliary_peak_bytes: PERSISTENT_AUX,
            uploaded_weight_bytes: PACKED,
            ..Default::default()
        };
        let after = NativeExecutionStats {
            dispatch_count: PHYSICAL_PER_STEP,
            dispatch_attempts: PHYSICAL_PER_STEP,
            logical_submissions: SUBMISSIONS_PER_STEP,
            host_write_bytes: recording_bytes(1).unwrap(),
            auxiliary_peak_bytes: PERSISTENT_AUX + attention_scratch_bytes(1).unwrap(),
            ..before
        };
        verify_step(before, after, 1).unwrap();
        for bad in [
            NativeExecutionStats {
                auxiliary_live_bytes: PERSISTENT_AUX + 4,
                ..after
            },
            NativeExecutionStats {
                auxiliary_peak_bytes: u64::from(PEAK_AUX),
                ..after
            },
            NativeExecutionStats {
                host_write_bytes: 4,
                ..after
            },
            NativeExecutionStats {
                host_read_bytes: 4,
                ..after
            },
            NativeExecutionStats {
                dispatch_count: PHYSICAL_PER_STEP - 1,
                ..after
            },
            NativeExecutionStats {
                dispatch_attempts: PHYSICAL_PER_STEP + 1,
                ..after
            },
            NativeExecutionStats {
                logical_submissions: SUBMISSIONS_PER_STEP - 1,
                ..after
            },
            NativeExecutionStats {
                uploaded_weight_bytes: PACKED + 4,
                ..after
            },
            NativeExecutionStats {
                poisoned: true,
                ..after
            },
        ] {
            assert!(verify_step(before, bad, 1).is_err());
        }
        assert!(verify_step(before, after, 0).is_err());
        assert!(verify_step(before, after, CONTEXT + 1).is_err());
        // Reset does not lower the allocator high-water mark. A replay writes
        // only first-position scratch and must retain the earlier maximum.
        let full = NativeExecutionStats {
            auxiliary_peak_bytes: u64::from(PEAK_AUX),
            ..before
        };
        let replay = NativeExecutionStats {
            auxiliary_peak_bytes: u64::from(PEAK_AUX),
            ..after
        };
        verify_step(full, replay, 1).unwrap();
        assert!(verify_step(full, after, 1).is_err());
        let tail = NativeExecutionStats {
            host_write_bytes: recording_bytes(CONTEXT).unwrap(),
            auxiliary_peak_bytes: u64::from(PEAK_AUX),
            ..after
        };
        verify_step(before, tail, CONTEXT).unwrap();
        assert!(verify_step(before, tail, 341).is_err());
    }
}
