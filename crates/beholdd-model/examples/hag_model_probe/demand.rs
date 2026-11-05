// Additive C128 qualification of attention batches recorded immediately before
// execution and retired after their fence. Existing context fixtures stay frozen.
use super::*;
use beholdd_backend::hag::native_execution::NativeRecordingFootprint;
use beholdd_memory::available_memory;
use std::time::Instant;

pub(super) const CONTEXT: usize = 128;
const CHECK_POSITIONS: [usize; 5] = [0, 31, 63, 64, 127];
const DEMAND_TENSORS: u64 = 33_793_536;
const PERSISTENT_AUX: u64 = 119_124;
const PEAK_AUX: u32 = 123_232;
const ALLOCATION: u64 = 667_193_344;
const PROCESS_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
const RECORDING_BYTES_PER_STEP: u64 = 115_028;
const PREPARATION_WRITES: u64 = PACKED + DEMAND_TENSORS + PERSISTENT_AUX
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

fn verify_step(before: NativeExecutionStats, after: NativeExecutionStats) -> Result<()> {
    ensure!(
        !after.poisoned
            && after.dispatch_count == before.dispatch_count + PHYSICAL_PER_STEP
            && after.dispatch_attempts == before.dispatch_attempts + PHYSICAL_PER_STEP
            && after.logical_submissions == before.logical_submissions + SUBMISSIONS_PER_STEP
            && after.host_write_bytes == before.host_write_bytes + RECORDING_BYTES_PER_STEP
            && after.host_read_bytes == before.host_read_bytes
            && after.uploaded_weight_bytes == before.uploaded_weight_bytes
            && after.allocation_bytes == ALLOCATION
            && after.weight_live_bytes == PACKED
            && after.tensor_live_bytes == DEMAND_TENSORS
            && after.auxiliary_live_bytes == PERSISTENT_AUX
            && after.auxiliary_peak_bytes == u64::from(PEAK_AUX)
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
    ensure!(choice == greedy(expected)?, "on-demand greedy token differs");
    selected.push(choice);
    Ok(())
}

pub(super) fn run(options: &Options) -> Result<()> {
    let path = options.model.as_ref().context("on-demand requires checkpoint")?;
    check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
    let checkpoint = Gguf::open(path)?;
    check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
    let model = QwenModel::load(&checkpoint)?;
    ensure!(model.architecture() == QwenArchitecture::Qwen3, "dense qwen3 required");
    let shape = Shape::from_model(&model);
    let plan = model.dense_model_plan(CONTEXT, TILE_ROWS)?;
    let config = demand_config(shape, plan.memory())?;
    inspect_checkpoint(&checkpoint, shape)?;
    ensure!(
        (plan.layers(), plan.dim(), plan.vocab(), plan.capacity())
            == (LAYERS, DIM, VOCAB, CONTEXT),
        "on-demand plan geometry differs"
    );
    let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |sum, path| {
        sum.checked_add(std::fs::metadata(path)?.len())
            .context("checkpoint size overflows")
    })?;
    let prepared_metadata = plan.on_demand_attention_host_metadata_bytes()?;
    let required = checked_forecast(checkpoint_bytes, prepared_metadata)?;
    admit_cost(required, available_memory()?.effective_available_bytes())?;
    let first = options.mode == Mode::OnDemandFirstToken;
    let checked_positions: &[usize] = if first { &[0] } else { &CHECK_POSITIONS };
    let cpu_steps = if first { 1 } else { CONTEXT };
    let evaluations = if first { 1 } else { CONTEXT + 1 };
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
        "recording_host_write_bytes_per_step": RECORDING_BYTES_PER_STEP,
        "preparation_host_write_bytes": PREPARATION_WRITES,
        "native_recording_metadata_bytes": footprint().host_bytes()?,
        "retained_attention_metadata_bytes": prepared_metadata,
        "combined_forecast_bytes": required, "process_budget_bytes": PROCESS_BUDGET,
        "retained_reference_bytes": checked_positions.len() * VOCAB * 4,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "fixture": "fixed token IDs 1..128; first-token mode checks token1 only; context mode checks full logits and exact CPU greedy at 0,31,63,64,127, refuses position128, resets and replays token1",
        "comparison_policy": "finite per-value tolerance and exact CPU greedy; sign/zero mismatches reported",
        "required_hag_trace_submission": "0", "external_process_and_thermal_limits_required": true,
    });
    if options.mode == Mode::InspectOnDemand {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    ensure!(
        matches!(options.mode, Mode::OnDemandFirstToken | Mode::OnDemandContext),
        "invalid on-demand execution mode"
    );
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
        }
    }
    let cpu_ms = cpu_started.elapsed().as_millis();
    let oracle_counts = oracle_source.counts();
    ensure!(
        expected.len() == checked_positions.len()
            && oracle_counts == (
                cpu_steps as u64 * (PREPARATION_READS + 1),
                cpu_steps as u64 * (SOURCE_BYTES + (DIM / 32 * 34) as u64),
                STAGING,
            ),
        "on-demand CPU source/reference accounting differs"
    );
    // CPU KV is gone; check host/cgroup headroom again before any native open.
    admit_cost(required, available_memory()?.effective_available_bytes())?;
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
        verify_step(before, device.stats()?)?;
        ensure!(
            resident.visible_positions() == position + 1
                && !resident.is_poisoned()
                && budget.current_bytes() == ALLOCATION,
            "on-demand cursor/charge differs"
        );
        if let Some(index) = checked_positions.iter().position(|&p| p == position) {
            resident.read_logits(&device, &mut actual)?;
            check_logits(&actual, &expected[index], &mut maximum, &mut signs, &mut zeros, &mut selected)?;
            println!("PASS demand position={position} checked_values={VOCAB}");
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
        verify_step(full, device.stats()?)?;
        ensure!(resident.visible_positions() == 1 && !resident.is_poisoned(), "replay cursor differs");
        resident.read_logits(&device, &mut actual)?;
        check_logits(&actual, &expected[0], &mut maximum, &mut signs, &mut zeros, &mut selected)?;
    }
    let native_ms = native_started.elapsed().as_millis();
    let final_stats = device.stats()?;
    let comparisons = checked_positions.len() + usize::from(!first);
    ensure!(
        final_stats.dispatch_count == evaluations as u64 * PHYSICAL_PER_STEP
            && final_stats.dispatch_attempts == evaluations as u64 * PHYSICAL_PER_STEP
            && final_stats.logical_submissions == evaluations as u64 * SUBMISSIONS_PER_STEP
            && final_stats.host_read_bytes == comparisons as u64 * VOCAB as u64 * 4
            && final_stats.host_write_bytes == initial.host_write_bytes + evaluations as u64 * RECORDING_BYTES_PER_STEP
            && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING)
            && selected.len() == comparisons,
        "on-demand final execution/traffic/source counters differ"
    );
    resident.reset()?;
    ensure!(
        device.stats()? == final_stats && resident.visible_positions() == 0 && !resident.is_poisoned(),
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
    report["recording_host_write_bytes"] = json!(evaluations as u64 * RECORDING_BYTES_PER_STEP);
    report["submission_tensor_transfer_bytes"] = json!(0);
    report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
    report["auxiliary_live_bytes_after_steps"] = json!(final_stats.auxiliary_live_bytes);
    report["cache_visible_positions_after_reset"] = json!(0);
    report["budget_after"] = json!(0);
    report["cpu_reference_ms"] = json!(cpu_ms);
    report["native_preparation_ms"] = json!(preparation_ms);
    report["native_execution_and_checks_ms"] = json!(native_ms);
    println!("HAG_DEMAND_REPORT {}", serde_json::to_string(&report)?);
    println!("PASS hag-demand stage={stage} qualified=true");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demand_forecast_admits_only_exact_geometry_and_bounded_cost() {
        let mut shape = super::super::tests::shape();
        let memory = DenseModelMemory {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: DEMAND_TENSORS,
            largest_staging_bytes: STAGING,
        };
        assert_eq!(demand_config(shape, memory).unwrap().allocation_bytes().unwrap(), ALLOCATION);
        shape.context = CONTEXT - 1;
        assert!(demand_config(shape, memory).is_err());
        assert!(demand_config(super::super::tests::shape(), DenseModelMemory {
            resident_tensor_bytes: DEMAND_TENSORS + 4, ..memory
        }).is_err());
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
        assert_eq!(u64::from(PEAK_AUX), PERSISTENT_AUX + layer_aux);
        assert_eq!(RECORDING_BYTES_PER_STEP, LAYERS as u64 * layer_aux + 4);
        assert_eq!(PREPARATION_WRITES, 669_243_220);
        assert_eq!(footprint().batches, LAYERS as u64 + 3);
        assert_eq!(footprint().dispatches, LAYERS as u64 * 7 + 40 + 127 + 1);
        assert_eq!(footprint().auxiliary_allocations, LAYERS as u64 * 4 + 2 + 4 + 1);
        assert_eq!(footprint().retained_allocation_references, LAYERS as u64 * 12 + 44 + 21 + 3);
        assert!(footprint().host_bytes().unwrap() < 1024 * 1024);
        assert_eq!((CONTEXT as u64 + 1) * PHYSICAL_PER_STEP, 489_297);
        assert_eq!((CONTEXT as u64 + 1) * SUBMISSIONS_PER_STEP, 7482);
        assert_eq!((CONTEXT as u64 + 1) * RECORDING_BYTES_PER_STEP, 14_838_612);
        assert_eq!((CHECK_POSITIONS.len() + 1) * VOCAB, 911_616);
        assert!((CHECK_POSITIONS.len() + 1) * VOCAB * 4 < 4 * 1024 * 1024);
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
            host_write_bytes: RECORDING_BYTES_PER_STEP,
            auxiliary_peak_bytes: u64::from(PEAK_AUX),
            ..before
        };
        verify_step(before, after).unwrap();
        for bad in [
            NativeExecutionStats { auxiliary_live_bytes: PERSISTENT_AUX + 4, ..after },
            NativeExecutionStats { auxiliary_peak_bytes: u64::from(PEAK_AUX) + 4, ..after },
            NativeExecutionStats { host_write_bytes: 4, ..after },
            NativeExecutionStats { host_read_bytes: 4, ..after },
            NativeExecutionStats { dispatch_count: PHYSICAL_PER_STEP - 1, ..after },
            NativeExecutionStats { dispatch_attempts: PHYSICAL_PER_STEP + 1, ..after },
            NativeExecutionStats { logical_submissions: SUBMISSIONS_PER_STEP - 1, ..after },
            NativeExecutionStats { uploaded_weight_bytes: PACKED + 4, ..after },
            NativeExecutionStats { poisoned: true, ..after },
        ] {
            assert!(verify_step(before, bad).is_err());
        }
    }
}
