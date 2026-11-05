// Separate-process C2048 reference export and native verification. The native
// branch never constructs an Oracle or retains CPU KV. The external supervisor
// authenticates the reference and proves the CPU process has retired first.
use super::reference::{read_identity, ReferenceSnapshot, ReferenceWriter};
use super::*;
use std::time::Instant;

const PREPARATION_WRITES: u64 = PACKED
    + TENSOR_BYTES
    + PERSISTENT_AUX
    + (SOURCE_BYTES - PACKED)
    + (LAYERS * CONTEXT * 128 * 4) as u64;

fn execution_config(shape: Shape, memory: DenseModelMemory) -> Result<NativeExecutionConfig> {
    inspect_geometry(shape, CONTEXT)?;
    ensure!(
        memory.packed_weight_bytes == PACKED
            && memory.resident_tensor_bytes == TENSOR_BYTES
            && memory.largest_staging_bytes == STAGING,
        "C2048 logical plan costs differ"
    );
    let config = NativeExecutionConfig {
        weight_capacity_bytes: u32::try_from(PACKED)?,
        tensor_capacity_bytes: u32::try_from(TENSOR_BYTES)?,
        auxiliary_capacity_bytes: PEAK_AUX,
    };
    ensure!(
        config.allocation_bytes()? == HAG_BYTES && HAG_BYTES <= PROPOSED_HAG_BUDGET,
        "C2048 physical charge differs"
    );
    Ok(config)
}

fn oracle_counts(evaluations: usize) -> Result<(u64, u64, u64)> {
    ensure!(
        (1..=CONTEXT).contains(&evaluations),
        "invalid CPU evaluation count"
    );
    let steps = u64::try_from(evaluations)?;
    Ok((
        steps
            .checked_mul(PREPARATION_READS + 1)
            .context("CPU read count overflows")?,
        steps
            .checked_mul(SOURCE_BYTES + (DIM / 32 * 34) as u64)
            .context("CPU source byte count overflows")?,
        STAGING,
    ))
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
            && after.allocation_bytes == HAG_BYTES
            && after.weight_live_bytes == PACKED
            && after.tensor_live_bytes == TENSOR_BYTES
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

fn emit_report(stage: &str, report: &serde_json::Value) -> Result<()> {
    println!("HAG_DEMAND2048_REPORT {}", serde_json::to_string(report)?);
    println!("PASS hag-demand2048 stage={stage} qualified=true");
    Ok(())
}

pub(super) fn run(options: &Options) -> Result<()> {
    let stage = match options.mode {
        Mode::OnDemand2048Reference => "cpu",
        Mode::OnDemand2048FirstToken => "first-token",
        Mode::OnDemand2048Context => "context",
        _ => bail!("invalid C2048 execution mode"),
    };
    let cpu = stage == "cpu";
    let first = stage == "first-token";
    ensure!(
        if cpu {
            options.device.is_none()
                && options.reference.is_none()
                && options.reference_out.is_some()
        } else {
            options.device.is_some()
                && options.reference.is_some()
                && options.reference_out.is_none()
        },
        "C2048 phase inputs differ"
    );
    let identity = read_identity(
        options
            .input_identity
            .as_ref()
            .context("missing input identity")?,
    )?;
    let path = options
        .model
        .as_ref()
        .context("C2048 requires checkpoint")?;
    check_checkpoint_storage([path], StorageOverride::RefuseUnsafe)?;
    let checkpoint = Gguf::open(path)?;
    check_checkpoint_storage(checkpoint.shards(), StorageOverride::RefuseUnsafe)?;
    let model = QwenModel::load(&checkpoint)?;
    ensure!(
        model.architecture() == QwenArchitecture::Qwen3,
        "dense qwen3 required"
    );
    let shape = Shape::from_model(&model);
    inspect_checkpoint(&checkpoint, shape)?;
    let plan = model.dense_model_plan_on_demand(CONTEXT, TILE_ROWS)?;
    let config = execution_config(shape, plan.memory())?;
    ensure!(
        plan.requires_on_demand_attention()
            && (plan.layers(), plan.dim(), plan.vocab(), plan.capacity())
                == (LAYERS, DIM, VOCAB, CONTEXT),
        "C2048 plan geometry differs"
    );
    let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |total, path| {
        sum(&[total, std::fs::metadata(path)?.len()])
    })?;
    let prepared_metadata = plan.on_demand_attention_host_metadata_bytes()?;
    // CPU admission precedes KV allocation. Native admission follows reference
    // parsing: its additional-allocation bound credits those retained vectors.
    let costs = if cpu {
        admit_phase(stage, checkpoint_bytes, prepared_metadata)?
    } else {
        let costs = forecast(checkpoint_bytes, prepared_metadata)?;
        validate_phase_bounds(&costs, PROCESS_LIMIT)?;
        costs
    };
    let checked_positions: &[usize] = if first { &[0] } else { &CHECK_POSITIONS };
    let mut report = json!({
        "stage": stage, "capacity": CONTEXT, "model_layers": LAYERS,
        "recording_strategy": "on-demand-attention", "compared_positions": checked_positions,
        "input_identity_attestation_loaded": true,
        "input_identity_policy": "exact supervisor-attested 128 bytes; cryptographic authentication external",
        "memory_policy": MEMORY_POLICY,
        "cpu_process_retirement_required": true, "allocator_release_from_drop_assumed": false,
        "checkpoint_allowance_per_phase_bytes": checkpoint_bytes,
        "packed_weight_bytes": PACKED, "source_weight_bytes": SOURCE_BYTES,
        "largest_staging_bytes": STAGING, "resident_tensor_bytes": TENSOR_BYTES,
        "expected_hag_allocation_bytes": HAG_BYTES, "hag_budget_bytes": PROPOSED_HAG_BUDGET,
        "persistent_auxiliary_bytes": PERSISTENT_AUX, "auxiliary_peak_bytes": PEAK_AUX,
        "persistent_batches": LAYERS + 1, "maximum_transient_attention_batches": 1,
        "recording_host_write_bytes_first_step": recording_bytes(1)?,
        "maximum_recording_host_write_bytes_per_step": recording_bytes(CONTEXT)?,
        "recording_host_write_bytes_formula": "4 + 28 * (max(4096, 12 * visible_positions + 4) + 12)",
        "preparation_host_write_bytes": PREPARATION_WRITES,
        "native_recording_metadata_bytes": footprint().host_bytes()?,
        "retained_attention_metadata_bytes": prepared_metadata,
        "cpu_phase_forecast_bytes": costs.cpu_total,
        "native_host_forecast_bytes": costs.native_host,
        "native_phase_combined_forecast_bytes": costs.native_total,
        "all_overlap_forecast_bytes": costs.all_overlap,
        "process_budget_bytes": PROCESS_LIMIT, "early_stop_bytes": EARLY_STOP,
        "reference_record_count": CHECK_POSITIONS.len(),
        "reference_logits_bytes": CHECK_POSITIONS.len() * VOCAB * 4,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "comparison_policy": "finite per-value tolerance and exact CPU greedy; sign/zero mismatches reported",
        "fixture": "fixed token IDs 1..2048; full context checks selected logits, refuses position2048, resets and replays token1",
        "required_hag_trace_submission": "0", "external_process_and_thermal_limits_required": true,
    });
    if cpu {
        let cpu_started = Instant::now();
        let source = CountingSource::new(&checkpoint);
        let mut writer = ReferenceWriter::create(
            options
                .reference_out
                .as_ref()
                .context("missing reference output")?,
            &identity,
        )?;
        let mut selected = Vec::with_capacity(CHECK_POSITIONS.len());
        {
            let mut oracle = Oracle::with_capacity(&checkpoint, &source, shape, CONTEXT);
            for position in 0..CONTEXT {
                let logits = oracle.step(u32::try_from(position + 1)?)?;
                ensure!(
                    logits.len() == VOCAB && logits.iter().all(|v| v.is_finite()),
                    "nonfinite or wrong-size C2048 CPU logits"
                );
                if CHECK_POSITIONS.contains(&position) {
                    selected.push(greedy(&logits)?);
                    writer.append(position, &logits)?;
                }
                // The full vector is dropped every iteration. Only the bounded
                // writer buffer and the oracle's growing KV remain in this job.
                if (position + 1) % 64 == 0 {
                    println!("PROGRESS demand2048 cpu_evaluations={}", position + 1);
                }
            }
            let counts = source.counts();
            let lengths: Vec<_> = oracle
                .caches
                .iter()
                .map(|(k, v)| (k.len(), v.len()))
                .collect();
            let expected_cache_values = CONTEXT * shape.kv_heads * shape.head;
            ensure!(
                oracle.position == CONTEXT
                    && lengths.len() == LAYERS
                    && lengths
                        .iter()
                        .all(|&pair| pair == (expected_cache_values, expected_cache_values))
                    && counts == oracle_counts(CONTEXT)?
                    && selected.len() == CHECK_POSITIONS.len(),
                "C2048 CPU source/cache/reference accounting differs"
            );
            // Oracle rejects at entry, before a load or cache mutation. Check the
            // cursor, all cache extents and source counters without cloning KV.
            ensure!(
                oracle.step(u32::try_from(CONTEXT + 1)?).is_err()
                    && oracle.position == CONTEXT
                    && oracle
                        .caches
                        .iter()
                        .map(|(k, v)| (k.len(), v.len()))
                        .eq(lengths)
                    && source.counts() == counts,
                "C2048 CPU capacity refusal changed source/cache state"
            );
        }
        ensure!(
            source.counts() == oracle_counts(CONTEXT)?,
            "C2048 CPU source total differs"
        );
        // Publication is last: incomplete, failed or refused work leaves no final
        // reference path. CPU process retirement is still an external gate.
        writer.finish()?;
        report["execution"] = json!(options.mode.execution());
        report["cpu_evaluations"] = json!(CONTEXT);
        report["evaluations"] = json!(0);
        report["exported_positions"] = json!(CHECK_POSITIONS.len());
        report["exported_values"] = json!(CHECK_POSITIONS.len() * VOCAB);
        report["greedy_tokens"] = json!(selected);
        report["cpu_oracle_source_reads"] = json!(source.counts().0);
        report["cpu_oracle_source_bytes"] = json!(source.counts().1);
        report["capacity_refusal_checked"] = json!(true);
        report["capacity_refused_without_source_reads"] = json!(true);
        report["reference_published"] = json!(true);
        report["reference_identity_attestation_written"] = json!(true);
        report["device_opened"] = json!(false);
        report["cpu_reference_ms"] = json!(cpu_started.elapsed().as_millis());
        return emit_report(stage, &report);
    }

    // Parse every finite vector and identity before opening a native device.
    // This branch deliberately has no Oracle and cannot regenerate references.
    let expected = ReferenceSnapshot::read(
        options
            .reference
            .as_ref()
            .context("missing native reference")?,
        &identity,
    )?;
    admit_phase(stage, checkpoint_bytes, prepared_metadata)?;
    let native_steps = if first { 1 } else { CONTEXT };
    let evaluations = native_steps + usize::from(!first);
    let comparisons = checked_positions.len() + usize::from(!first);
    let planned_recording = total_recording_bytes(native_steps, !first)?;
    let planned_auxiliary_peak = PERSISTENT_AUX + attention_scratch_bytes(native_steps)?;
    let budget = MemoryBudget::new(PROPOSED_HAG_BUDGET);
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
            resident_tensor_bytes: TENSOR_BYTES,
            largest_staging_bytes: STAGING,
        },
    )?;
    let preparation_ms = prepare_started.elapsed().as_millis();
    let initial = device.stats()?;
    ensure!(
        initial.allocation_bytes == HAG_BYTES
            && budget.current_bytes() == HAG_BYTES
            && initial.weight_live_bytes == PACKED
            && initial.tensor_live_bytes == TENSOR_BYTES
            && initial.weight_peak_bytes == PACKED
            && initial.tensor_peak_bytes == TENSOR_BYTES
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
        "C2048 preparation accounting/state differs"
    );
    let native_started = Instant::now();
    let mut actual = vec![f32::NAN; VOCAB];
    ensure!(
        resident.read_logits(&device, &mut actual).is_err()
            && device.stats()? == initial
            && actual.iter().all(|v| v.is_nan())
            && !resident.is_poisoned(),
        "C2048 stale initial logits were readable or mutated state"
    );
    let mut maximum = 0.0_f64;
    let mut signs = 0u64;
    let mut zeros = 0u64;
    let mut selected = Vec::with_capacity(comparisons);
    for position in 0..native_steps {
        let before = device.stats()?;
        resident.step(&device, u32::try_from(position + 1)?)?;
        verify_step(before, device.stats()?, position + 1)?;
        ensure!(
            resident.visible_positions() == position + 1
                && !resident.is_poisoned()
                && budget.current_bytes() == HAG_BYTES,
            "C2048 native cursor/charge differs"
        );
        if checked_positions.contains(&position) {
            resident.read_logits(&device, &mut actual)?;
            check_logits(
                &actual,
                expected.logits(position)?,
                &mut maximum,
                &mut signs,
                &mut zeros,
                &mut selected,
            )?;
            println!("PASS demand2048 position={position} checked_values={VOCAB}");
        }
        if (position + 1) % 64 == 0 {
            println!("PROGRESS demand2048 native_evaluations={}", position + 1);
        }
    }
    if !first {
        let full = device.stats()?;
        ensure!(
            resident.step(&device, u32::try_from(CONTEXT + 1)?).is_err()
                && device.stats()? == full
                && resident.visible_positions() == CONTEXT
                && !resident.is_poisoned(),
            "C2048 capacity refusal changed recording/transport/state"
        );
        resident.reset()?;
        actual.fill(f32::NAN);
        ensure!(
            resident.visible_positions() == 0
                && !resident.is_poisoned()
                && resident.read_logits(&device, &mut actual).is_err()
                && actual.iter().all(|v| v.is_nan())
                && device.stats()? == full,
            "C2048 reset changed transport or exposed stale logits"
        );
        resident.step(&device, 1)?;
        verify_step(full, device.stats()?, 1)?;
        ensure!(
            resident.visible_positions() == 1 && !resident.is_poisoned(),
            "C2048 replay cursor differs"
        );
        resident.read_logits(&device, &mut actual)?;
        check_logits(
            &actual,
            expected.logits(0)?,
            &mut maximum,
            &mut signs,
            &mut zeros,
            &mut selected,
        )?;
    }
    let native_ms = native_started.elapsed().as_millis();
    let final_stats = device.stats()?;
    ensure!(
        final_stats.dispatch_count == evaluations as u64 * PHYSICAL_PER_STEP
            && final_stats.dispatch_attempts == evaluations as u64 * PHYSICAL_PER_STEP
            && final_stats.logical_submissions == evaluations as u64 * SUBMISSIONS_PER_STEP
            && final_stats.host_read_bytes == comparisons as u64 * VOCAB as u64 * 4
            && final_stats.host_write_bytes == initial.host_write_bytes + planned_recording
            && final_stats.auxiliary_peak_bytes == planned_auxiliary_peak
            && final_stats.auxiliary_live_bytes == PERSISTENT_AUX
            && source.counts() == (PREPARATION_READS, SOURCE_BYTES, STAGING)
            && selected.len() == comparisons
            && !final_stats.poisoned,
        "C2048 final execution/traffic/source counters differ"
    );
    resident.reset()?;
    actual.fill(f32::NAN);
    ensure!(
        resident.visible_positions() == 0
            && !resident.is_poisoned()
            && resident.read_logits(&device, &mut actual).is_err()
            && actual.iter().all(|v| v.is_nan())
            && device.stats()? == final_stats,
        "C2048 final reset or stale-logit refusal differs"
    );
    drop(resident);
    let retired = device.stats()?;
    ensure!(
        retired.weight_live_bytes == 0
            && retired.tensor_live_bytes == 0
            && retired.auxiliary_live_bytes == 0
            && !retired.poisoned,
        "C2048 native resources did not retire"
    );
    drop(device);
    ensure!(budget.current_bytes() == 0, "C2048 HAG charge retained");
    report["execution"] = json!(options.mode.execution());
    report["evaluations"] = json!(evaluations);
    report["cpu_evaluations"] = json!(0);
    report["reference_cpu_evaluations"] = json!(CONTEXT);
    report["reference_validated_before_device_open"] = json!(true);
    report["reference_identity_matched"] = json!(true);
    report["device_opened"] = json!(true);
    report["checked_values"] = json!(comparisons * VOCAB);
    report["greedy_tokens"] = json!(selected);
    report["greedy_matches"] = json!(comparisons);
    report["max_absolute_error"] = json!(maximum);
    report["logit_sign_mismatches"] = json!(signs);
    report["logit_zero_mismatches"] = json!(zeros);
    report["capacity_refusal_checked"] = json!(!first);
    report["capacity_refused_without_dispatch"] = json!(!first);
    report["reset_replay_passed"] = json!(!first);
    report["stale_logits_refused"] = json!(true);
    report["physical_dispatches"] = json!(final_stats.dispatch_count);
    report["logical_submissions"] = json!(final_stats.logical_submissions);
    report["native_preparation_source_reads"] = json!(PREPARATION_READS);
    report["native_preparation_source_bytes"] = json!(SOURCE_BYTES);
    report["cpu_oracle_source_reads"] = json!(0);
    report["cpu_oracle_source_bytes"] = json!(0);
    report["actual_hag_allocation_bytes"] = json!(HAG_BYTES);
    report["initial_host_write_bytes"] = json!(initial.host_write_bytes);
    report["recording_host_write_bytes"] = json!(planned_recording);
    report["actual_auxiliary_peak_bytes"] = json!(final_stats.auxiliary_peak_bytes);
    report["submission_tensor_transfer_bytes"] = json!(0);
    report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
    report["auxiliary_live_bytes_after_steps"] = json!(final_stats.auxiliary_live_bytes);
    report["cache_visible_positions_after_reset"] = json!(0);
    report["budget_after"] = json!(0);
    report["native_preparation_ms"] = json!(preparation_ms);
    report["native_execution_and_checks_ms"] = json!(native_ms);
    emit_report(stage, &report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c2048_scratch_bounds_and_streamed_recording_totals_are_exact() {
        for (visible, scratch) in [
            (1, 4108),
            (341, 4108),
            (342, 4120),
            (512, 6160),
            (1024, 12304),
            (2047, 24580),
            (2048, 24592),
        ] {
            assert_eq!(attention_scratch_bytes(visible).unwrap(), scratch);
            assert_eq!(
                recording_bytes(visible).unwrap(),
                4 + LAYERS as u64 * scratch
            );
        }
        for invalid in [0, CONTEXT + 1, usize::MAX] {
            assert!(attention_scratch_bytes(invalid).is_err());
            assert!(recording_bytes(invalid).is_err());
            assert!(total_recording_bytes(invalid, false).is_err());
            assert!(oracle_counts(invalid).is_err());
        }
        assert_eq!(PERSISTENT_AUX, LAYERS as u64 * 4108 + 4100);
        assert_eq!(
            u64::from(PEAK_AUX),
            PERSISTENT_AUX + attention_scratch_bytes(CONTEXT).unwrap()
        );
        assert_eq!(recording_bytes(CONTEXT).unwrap(), 688_580);
        assert_eq!(PREPARATION_WRITES, 1_164_695_380);
        assert_eq!(total_recording_bytes(1, false).unwrap(), 115_028);
        assert_eq!(total_recording_bytes(CONTEXT, true).unwrap(), 725_505_780);
        assert_eq!(
            total_recording_bytes(CONTEXT, true).unwrap()
                - total_recording_bytes(CONTEXT, false).unwrap(),
            recording_bytes(1).unwrap()
        );
        assert_eq!((CONTEXT as u64 + 1) * PHYSICAL_PER_STEP, 7_771_857);
        assert_eq!((CONTEXT as u64 + 1) * SUBMISSIONS_PER_STEP, 118_842);
        assert_eq!(
            oracle_counts(CONTEXT).unwrap(),
            (712_704, 1_297_401_118_720, STAGING)
        );
        assert_eq!((CHECK_POSITIONS.len() + 1) * VOCAB, 1_975_168);
        assert_eq!((CHECK_POSITIONS.len() + 1) * VOCAB * 4, 7_900_672);
        assert_eq!(
            CHECK_POSITIONS,
            [0, 127, 340, 341, 510, 511, 512, 1022, 1023, 1024, 2046, 2047]
        );
    }

    #[test]
    fn full_c2048_preparation_is_required_even_for_first_token() {
        let mut shape = super::super::super::tests::shape();
        let memory = DenseModelMemory {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: TENSOR_BYTES,
            largest_staging_bytes: STAGING,
        };
        assert_eq!(
            execution_config(shape, memory)
                .unwrap()
                .allocation_bytes()
                .unwrap(),
            HAG_BYTES
        );
        assert!(HAG_BYTES <= PROPOSED_HAG_BUDGET);
        assert!(HAG_BYTES > MODEL_BUDGET);
        shape.context = CONTEXT - 1;
        assert!(execution_config(shape, memory).is_err());
        let shape = super::super::super::tests::shape();
        for invalid in [
            DenseModelMemory {
                resident_tensor_bytes: TENSOR_BYTES - 4,
                ..memory
            },
            DenseModelMemory {
                packed_weight_bytes: PACKED - 4,
                ..memory
            },
            DenseModelMemory {
                largest_staging_bytes: STAGING + 4,
                ..memory
            },
        ] {
            assert!(execution_config(shape, invalid).is_err());
        }
    }

    #[test]
    fn logit_checks_preserve_finite_tolerance_and_exact_greedy_policy() {
        let mut maximum = 0.0;
        let mut signs = 0;
        let mut zeros = 0;
        let mut selected = Vec::new();
        check_logits(
            &[0.0, 1.0, 2.0001],
            &[0.0, 1.0, 2.0],
            &mut maximum,
            &mut signs,
            &mut zeros,
            &mut selected,
        )
        .unwrap();
        assert_eq!(selected, [2]);
        assert!(maximum > 0.0);
        for (actual, expected) in [
            ([0.0, f32::NAN], [0.0, 1.0]),
            ([0.0, f32::INFINITY], [0.0, 1.0]),
            ([0.0, 2.0], [0.0, 1.0]),
            // Numerically within tolerance, but the stable-first greedy differs.
            ([1.0, 1.0001], [1.0, 1.0]),
        ] {
            assert!(check_logits(
                &actual,
                &expected,
                &mut maximum,
                &mut signs,
                &mut zeros,
                &mut selected
            )
            .is_err());
        }
        assert_eq!(selected, [2]);
    }

    #[test]
    fn demand_step_rejects_retained_scratch_extra_traffic_and_missing_work() {
        let before = NativeExecutionStats {
            allocation_bytes: HAG_BYTES,
            weight_live_bytes: PACKED,
            tensor_live_bytes: TENSOR_BYTES,
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
