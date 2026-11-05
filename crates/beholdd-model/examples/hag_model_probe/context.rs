// Separate C128 qualification: reuse the independently checked immediate CPU oracle,
// while keeping the numerical fixtures separate from archived allocation identities.
use super::*;
use beholdd_backend::hag::native_execution::NativeRecordingFootprint;
use beholdd_memory::available_memory;
use std::time::Instant;

pub(super) const CONTEXT: usize = 128;
const CHECK_POSITIONS: [usize; 5] = [0, 31, 63, 64, 127];
const CONTEXT_TENSORS: u64 = 33_793_536;
const CONTEXT_AUX: u32 = 14_842_200;
const CONTEXT_ALLOCATION: u64 = 681_910_272;
const PROCESS_BUDGET: u64 = 2 * 1024 * 1024 * 1024;
const CONTEXT_PREPARATION_WRITES: u64 = PACKED + CONTEXT_TENSORS + CONTEXT_AUX as u64 - 4
    + (SOURCE_BYTES - PACKED)
    + (LAYERS * CONTEXT * 128 * 4) as u64;

fn footprint() -> NativeRecordingFootprint {
    NativeRecordingFootprint {
        batches: 3614,
        dispatches: 455_405,
        weight_allocations: 234,
        tensor_allocations: 452,
        auxiliary_allocations: 14_451,
        retained_allocation_references: 75_647,
        noop_operands: 0,
        max_batch_commands: 54,
        max_batch_dispatches: 127,
    }
}

fn context_config(shape: Shape, memory: DenseModelMemory) -> Result<NativeExecutionConfig> {
    inspect_geometry(shape, CONTEXT)?;
    ensure!(
        memory.packed_weight_bytes == PACKED
            && memory.resident_tensor_bytes == CONTEXT_TENSORS
            && memory.largest_staging_bytes == STAGING,
        "C128 plan costs differ"
    );
    let config = NativeExecutionConfig {
        weight_capacity_bytes: u32::try_from(PACKED)?,
        tensor_capacity_bytes: u32::try_from(CONTEXT_TENSORS)?,
        auxiliary_capacity_bytes: CONTEXT_AUX,
    };
    ensure!(
        config.allocation_bytes()? == CONTEXT_ALLOCATION && CONTEXT_ALLOCATION <= MODEL_BUDGET,
        "C128 hardware charge differs"
    );
    Ok(config)
}

fn checked_forecast(checkpoint_bytes: u64) -> Result<u64> {
    // Count file-cache/UMA pressure, actual-layout native metadata, all retained
    // references, the oracle's full KV and a separate 16 MiB model-table allowance.
    // The 128 MiB general margin remains additional. CPU and GPU execute sequentially.
    [
        CONTEXT_ALLOCATION,
        checkpoint_bytes,
        footprint().host_bytes()?,
        STAGING,
        (CHECK_POSITIONS.len() + 1) as u64 * VOCAB as u64 * 4,
        (LAYERS * 2 * CONTEXT * 8 * 128 * 4) as u64,
        16 * 1024 * 1024,
        128 * 1024 * 1024,
    ]
    .into_iter()
    .try_fold(0u64, |sum, value| {
        sum.checked_add(value)
            .context("C128 process forecast overflows")
    })
}

fn admit_process(required: u64) -> Result<()> {
    let available = available_memory()?.effective_available_bytes();
    ensure!(
        required <= PROCESS_BUDGET && required <= available,
        "C128 forecast {required} exceeds process budget {PROCESS_BUDGET} or available {available}"
    );
    Ok(())
}

fn verify_context_step(before: NativeExecutionStats, after: NativeExecutionStats) -> Result<()> {
    ensure!(
        !after.poisoned
            && after.dispatch_count == before.dispatch_count + PHYSICAL_PER_STEP
            && after.dispatch_attempts == before.dispatch_attempts + PHYSICAL_PER_STEP
            && after.logical_submissions == before.logical_submissions + SUBMISSIONS_PER_STEP
            && after.host_write_bytes == before.host_write_bytes + 4
            && after.host_read_bytes == before.host_read_bytes
            && after.uploaded_weight_bytes == before.uploaded_weight_bytes
            && after.allocation_bytes == CONTEXT_ALLOCATION
            && after.weight_live_bytes == PACKED
            && after.tensor_live_bytes == CONTEXT_TENSORS
            && after.auxiliary_live_bytes == u64::from(CONTEXT_AUX) - 4
            && after.auxiliary_peak_bytes == u64::from(CONTEXT_AUX),
        "C128 step counters/residency differ"
    );
    Ok(())
}

pub(super) fn run(options: &Options) -> Result<()> {
    let path = options
        .model
        .as_ref()
        .context("C128 requires explicit checkpoint")?;
    check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
    let checkpoint = Gguf::open(path)?;
    check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
    let model = QwenModel::load(&checkpoint)?;
    ensure!(
        model.architecture() == QwenArchitecture::Qwen3,
        "dense qwen3 required"
    );
    let shape = Shape::from_model(&model);
    let plan = model.dense_model_plan(CONTEXT, TILE_ROWS)?;
    let config = context_config(shape, plan.memory())?;
    inspect_checkpoint(&checkpoint, shape)?;
    let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |sum, path| {
        sum.checked_add(std::fs::metadata(path)?.len())
            .context("checkpoint size overflows")
    })?;
    let required = checked_forecast(checkpoint_bytes)?;
    admit_process(required)?;
    let mut report = json!({
        "execution": "not_requested", "capacity": CONTEXT, "model_layers": LAYERS,
        "compared_positions": CHECK_POSITIONS, "planned_evaluations": CONTEXT + 1,
        "expected_hag_allocation_bytes": CONTEXT_ALLOCATION, "hag_budget_bytes": MODEL_BUDGET,
        "resident_tensor_bytes": CONTEXT_TENSORS, "auxiliary_peak_bytes": CONTEXT_AUX,
        "persistent_auxiliary_bytes": CONTEXT_AUX - 4,
        "preparation_host_write_bytes": CONTEXT_PREPARATION_WRITES,
        "native_recording_metadata_bytes": footprint().host_bytes()?,
        "combined_forecast_bytes": required, "process_budget_bytes": PROCESS_BUDGET,
        "retained_reference_bytes": CHECK_POSITIONS.len() * VOCAB * 4,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "fixture": "fixed token IDs 1..128, then reset and replay token 1; finite full logits and exact CPU greedy at 0,31,63,64,127 and replay 0",
        "required_hag_trace_submission": "0", "external_process_and_thermal_limits_required": true,
    });
    if options.mode == Mode::InspectContext {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    ensure!(
        options.mode == Mode::ContextBoundary,
        "invalid C128 execution mode"
    );
    let cpu_started = Instant::now();
    let oracle_source = CountingSource::new(&checkpoint);
    let mut expected = Vec::with_capacity(CHECK_POSITIONS.len());
    {
        let mut oracle = Oracle::with_capacity(&checkpoint, &oracle_source, shape, CONTEXT);
        for position in 0..CONTEXT {
            let logits = oracle.step((position + 1) as u32)?;
            if CHECK_POSITIONS.contains(&position) {
                ensure!(
                    logits.len() == VOCAB && logits.iter().all(|v| v.is_finite()),
                    "nonfinite CPU C128 reference"
                );
                expected.push(logits);
            }
        }
    }
    let cpu_ms = cpu_started.elapsed().as_millis();
    ensure!(
        expected.len() == CHECK_POSITIONS.len(),
        "C128 reference count differs"
    );
    ensure!(
        oracle_source.counts()
            == (
                CONTEXT as u64 * (PREPARATION_READS + 1),
                CONTEXT as u64 * (SOURCE_BYTES + (DIM / 32 * 34) as u64),
                STAGING
            ),
        "C128 CPU source accounting differs"
    );
    // The CPU KV owner has been destroyed. Recheck headroom before native allocation.
    admit_process(required)?;
    let budget = MemoryBudget::new(MODEL_BUDGET);
    let prepare_started = Instant::now();
    let device = NativeExecutionDevice::open(
        options.device.as_ref().context("missing render node")?,
        &budget,
        config,
    )?;
    let source = CountingSource::new(&checkpoint);
    let mut resident = plan.prepare(
        &device,
        &source,
        DenseModelLimits {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: CONTEXT_TENSORS,
            largest_staging_bytes: STAGING,
        },
    )?;
    let preparation_ms = prepare_started.elapsed().as_millis();
    let initial = device.stats()?;
    ensure!(
        initial.allocation_bytes == CONTEXT_ALLOCATION
            && budget.current_bytes() == CONTEXT_ALLOCATION
            && initial.weight_live_bytes == PACKED
            && initial.tensor_live_bytes == CONTEXT_TENSORS
            && initial.auxiliary_live_bytes == u64::from(CONTEXT_AUX) - 4
            && initial.uploaded_weight_bytes == PACKED
            && initial.host_write_bytes == CONTEXT_PREPARATION_WRITES
            && initial.host_read_bytes == 0
            && initial.dispatch_count == 0
            && !initial.poisoned
            && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING),
        "C128 preparation accounting differs"
    );
    let native_started = Instant::now();
    let mut actual = vec![f32::NAN; VOCAB];
    let mut maximum = 0.0_f64;
    let mut signs = 0u64;
    let mut zeros = 0u64;
    let mut selected = Vec::new();
    for position in 0..CONTEXT {
        let before = device.stats()?;
        resident.step(&device, (position + 1) as u32)?;
        verify_context_step(before, device.stats()?)?;
        ensure!(
            resident.visible_positions() == position + 1 && !resident.is_poisoned(),
            "C128 cursor differs"
        );
        if let Some(index) = CHECK_POSITIONS.iter().position(|&p| p == position) {
            resident.read_logits(&device, &mut actual)?;
            check_logits(
                &actual,
                &expected[index],
                &mut maximum,
                &mut signs,
                &mut zeros,
                &mut selected,
            )?;
            println!("PASS context position={position} checked_values={VOCAB}");
        }
    }
    let full = device.stats()?;
    ensure!(
        resident.step(&device, (CONTEXT + 1) as u32).is_err()
            && device.stats()? == full
            && resident.visible_positions() == CONTEXT
            && !resident.is_poisoned(),
        "C129 refusal changed native state"
    );
    resident.reset()?;
    ensure!(
        resident.visible_positions() == 0 && device.stats()? == full,
        "C128 reset changed transport"
    );
    resident.step(&device, 1)?;
    verify_context_step(full, device.stats()?)?;
    resident.read_logits(&device, &mut actual)?;
    check_logits(
        &actual,
        &expected[0],
        &mut maximum,
        &mut signs,
        &mut zeros,
        &mut selected,
    )?;
    let native_ms = native_started.elapsed().as_millis();
    let final_stats = device.stats()?;
    let evaluations = CONTEXT as u64 + 1;
    ensure!(
        final_stats.dispatch_count == evaluations * PHYSICAL_PER_STEP
            && final_stats.logical_submissions == evaluations * SUBMISSIONS_PER_STEP
            && final_stats.host_read_bytes == (CHECK_POSITIONS.len() as u64 + 1) * VOCAB as u64 * 4
            && final_stats.host_write_bytes == initial.host_write_bytes + evaluations * 4
            && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING),
        "C128 final counters differ"
    );
    resident.reset()?;
    ensure!(
        device.stats()? == final_stats && resident.visible_positions() == 0,
        "final reset differs"
    );
    drop(resident);
    let retired = device.stats()?;
    ensure!(
        retired.weight_live_bytes == 0
            && retired.tensor_live_bytes == 0
            && retired.auxiliary_live_bytes == 0
            && !retired.poisoned,
        "C128 resources did not retire"
    );
    drop(device);
    ensure!(budget.current_bytes() == 0, "C128 HAG charge retained");
    report["execution"] = json!("native-hag-context-boundary");
    report["evaluations"] = json!(evaluations);
    report["checked_values"] = json!((CHECK_POSITIONS.len() + 1) * VOCAB);
    report["greedy_tokens"] = json!(selected);
    report["max_absolute_error"] = json!(maximum);
    report["logit_sign_mismatches"] = json!(signs);
    report["logit_zero_mismatches"] = json!(zeros);
    report["capacity_refused_without_dispatch"] = json!(true);
    report["reset_replay_passed"] = json!(true);
    report["physical_dispatches"] = json!(final_stats.dispatch_count);
    report["logical_submissions"] = json!(final_stats.logical_submissions);
    report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
    report["budget_after"] = json!(0);
    report["cpu_reference_ms"] = json!(cpu_ms);
    report["native_preparation_ms"] = json!(preparation_ms);
    report["native_execution_and_checks_ms"] = json!(native_ms);
    println!("HAG_CONTEXT_REPORT {}", serde_json::to_string(&report)?);
    println!("PASS hag-context stage=boundary qualified=true");
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
    ensure!(choice == greedy(expected)?, "C128 greedy token differs");
    selected.push(choice);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn context_cost_and_reference_storage_are_bounded_before_payload_reads() {
        let mut shape = super::super::tests::shape();
        let memory = DenseModelMemory {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: CONTEXT_TENSORS,
            largest_staging_bytes: STAGING,
        };
        assert_eq!(
            context_config(shape, memory)
                .unwrap()
                .allocation_bytes()
                .unwrap(),
            CONTEXT_ALLOCATION
        );
        shape.context = 127;
        assert!(context_config(shape, memory).is_err());
        assert!(checked_forecast(u64::MAX).is_err());
        assert!(checked_forecast(639_447_744).unwrap() < PROCESS_BUDGET);
        assert!((CHECK_POSITIONS.len() + 1) * VOCAB * 4 < 4 * 1024 * 1024);
    }

    #[test]
    fn c128_counts_include_the_old_boundary_final_position_and_reset_replay() {
        assert_eq!(CONTEXT, 128);
        assert_eq!(CHECK_POSITIONS, [0, 31, 63, 64, 127]);
        assert_eq!((CONTEXT as u64 + 1) * PHYSICAL_PER_STEP, 489_297);
        assert_eq!((CONTEXT as u64 + 1) * SUBMISSIONS_PER_STEP, 7482);
        assert_eq!((CHECK_POSITIONS.len() + 1) * VOCAB, 911_616);
        let positions = LAYERS as u64 * CONTEXT as u64;
        assert_eq!(footprint().batches, positions + LAYERS as u64 + 2);
        assert_eq!(
            footprint().dispatches,
            positions * 127 + LAYERS as u64 * 7 + 41
        );
        assert_eq!(
            footprint().auxiliary_allocations,
            positions * 4 + LAYERS as u64 * 4 + 3
        );
        assert_eq!(
            footprint().retained_allocation_references,
            positions * 21 + LAYERS as u64 * 12 + 47
        );
        let workspace = (DIM as u64 * 4).max(CONTEXT as u64 * 12 + 4);
        assert_eq!(
            u64::from(CONTEXT_AUX),
            LAYERS as u64 * (CONTEXT as u64 + 1) * (workspace + 12) + DIM as u64 * 4 + 8
        );
        assert_eq!(CONTEXT_PREPARATION_WRITES, 683_966_292);
        #[cfg(target_pointer_width = "64")]
        {
            // Current ComputeQueue retains its 32-byte KernelAbi. host_bytes()
            // counts that owner once through NativeTransport, independent of
            // batch count; archived C128 forecasts used the earlier owner layout.
            assert_eq!(footprint().host_bytes().unwrap(), 94_711_551);
            assert_eq!(checked_forecast(639_447_744).unwrap(), 1_604_527_551);
        }
    }
}
