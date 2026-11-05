// Separate-process C16384 allocation/two-position reference and native probe. The native
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
            && memory.largest_staging_bytes == MODEL_STAGING,
        "C16384 logical plan costs differ"
    );
    let config = NativeExecutionConfig {
        weight_capacity_bytes: u32::try_from(PACKED)?,
        tensor_capacity_bytes: u32::try_from(TENSOR_BYTES)?,
        auxiliary_capacity_bytes: PEAK_AUX,
    };
    ensure!(
        config.allocation_bytes()? == HAG_BYTES && HAG_BYTES <= PROPOSED_HAG_BUDGET,
        "C16384 physical charge differs"
    );
    Ok(config)
}

fn oracle_counts(evaluations: usize) -> Result<(u64, u64, u64)> {
    ensure!(
        (1..=catalog_prefill32::CPU_STEPS).contains(&evaluations),
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

fn total_recording_bytes(steps: usize) -> Result<u64> {
    let mut total = recording_bytes(1)?; // Reset/replay.
    for visible in 1..=steps {
        total = sum(&[total, recording_bytes(visible)?])?;
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

fn emit_report(stage: &str, report: &serde_json::Value, catalog: bool) -> Result<()> {
    let prefix = if catalog {
        "HAG_CATALOG_PREFILL32_REPORT"
    } else {
        "HAG_CAPACITY16384_REPORT"
    };
    let label = if catalog {
        "catalog-prefill32"
    } else {
        "capacity16384"
    };
    println!("{prefix} {}", serde_json::to_string(report)?);
    println!("PASS hag-{label} stage={stage} bounded_probe=true");
    Ok(())
}

pub(super) fn run(options: &Options) -> Result<()> {
    let stage = match options.mode {
        Mode::Capacity16384Reference | Mode::CatalogPrefill32Reference => "cpu",
        Mode::Capacity16384Smoke | Mode::CatalogPrefill32Smoke => "native",
        _ => bail!("invalid C16384 execution mode"),
    };
    let cpu = stage == "cpu";
    let catalog = matches!(
        options.mode,
        Mode::CatalogPrefill32Reference | Mode::CatalogPrefill32Smoke
    );
    let label = if catalog {
        "catalog-prefill32"
    } else {
        "capacity16384"
    };
    let cpu_steps = if catalog {
        catalog_prefill32::CPU_STEPS
    } else {
        CPU_STEPS
    };
    let checked_positions: &[usize] = if catalog {
        &catalog_prefill32::CHECK_POSITIONS
    } else {
        &CHECK_POSITIONS
    };
    require_scope(stage)?;
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
        "C16384 phase inputs differ"
    );
    ensure!(
        catalog == options.prompt_tokens.is_some(),
        "catalog token artifact mode differs"
    );
    let tokens = if catalog {
        catalog_prefill32::read(
            options
                .prompt_tokens
                .as_ref()
                .context("missing prompt tokens")?,
        )?
        .tokens
    } else {
        vec![1, 2]
    };
    let checked_tokens = [tokens[checked_positions[0]], tokens[checked_positions[1]]];
    let identity = read_identity(
        options
            .input_identity
            .as_ref()
            .context("missing input identity")?,
    )?;
    let path = options
        .model
        .as_ref()
        .context("C16384 requires checkpoint")?;
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
        "C16384 plan geometry differs"
    );
    let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |total, path| {
        sum(&[total, std::fs::metadata(path)?.len()])
    })?;
    let prepared_metadata = plan.on_demand_attention_host_metadata_bytes()?;
    // CPU admission precedes KV allocation. Native admission follows reference
    // parsing: its additional-allocation bound credits those retained vectors.
    let costs = if cpu {
        admit_phase(stage, checkpoint_bytes, prepared_metadata, cpu_steps)?
    } else {
        let costs = forecast_steps(checkpoint_bytes, prepared_metadata, cpu_steps)?;
        validate_phase_bounds(&costs)?;
        costs
    };
    let (process_limit, early_stop) = phase_bounds(stage)?;
    let mut report = json!({
        "stage": stage, "capacity": CONTEXT, "model_layers": LAYERS,
        "recording_strategy": "on-demand-attention", "compared_positions": checked_positions,
        "input_identity_attestation_loaded": true,
        "input_identity_policy": "exact supervisor-attested 128 bytes; cryptographic authentication external",
        "memory_policy": if catalog { "separate-cpu-native-catalog-prefill32-v1" } else { MEMORY_POLICY },
        "cpu_process_retirement_required": true, "allocator_release_from_drop_assumed": false,
        "checkpoint_allowance_per_phase_bytes": checkpoint_bytes,
        "packed_weight_bytes": PACKED, "source_weight_bytes": SOURCE_BYTES,
        "largest_staging_bytes": MODEL_STAGING,
        "resident_tensor_bytes": TENSOR_BYTES,
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
        "process_budget_bytes": process_limit, "early_stop_bytes": early_stop,
        "reference_record_count": checked_positions.len(),
        "reference_logits_bytes": checked_positions.len() * VOCAB * 4,
        "absolute_tolerance": ABSOLUTE_TOLERANCE, "relative_tolerance": RELATIVE_TOLERANCE,
        "comparison_policy": "finite per-value tolerance and exact CPU greedy; sign/zero mismatches reported",
        "fixture": if catalog { "first32 of8079 exact production39-tool prompt IDs; checkpoints0,31; full C16384 allocation and first-token reset/replay; remaining8047 prompt positions not evaluated" } else { "fixed token IDs 1,2 with full C16384 native allocation; reset and replay token1; native capacity boundary not exercised" },
        "required_hag_trace_submission": "0", "external_process_and_thermal_limits_required": true,
    });
    report["native_allocation_requested_capacity"] = json!(CONTEXT);
    report["native_allocated_capacity"] = json!(if cpu { 0 } else { CONTEXT });
    report["cpu_reference_capacity"] = json!(cpu_steps);
    report["maximum_evaluated_position"] = json!(cpu_steps - 1);
    report["full_context_numerical_qualification"] = json!(false);
    report["long_prompt_qualification"] = json!(false);
    report["largest_packed_source_staging_bytes"] = json!(STAGING);
    report["preparation_staging_peak_bytes"] = json!(PREPARATION_STAGING);
    report["startup_host_headroom_bytes"] = json!(STARTUP_HEADROOM);
    report["running_host_headroom_bytes"] = json!(RETAINED_HOST_HEADROOM);
    if catalog {
        report["prompt_token_artifact_validated_before_device_open"] = json!(true);
        report["prompt_tokens"] = json!(tokens.len());
        report["prompt_tool_count"] = json!(39);
        report["prompt_message_count"] = json!(2);
        report["requested_max_tokens"] = json!(4096);
        report["minimum_context_positions"] = json!(12174);
        report["rendered_prompt_bytes"] = json!(33541);
        report["prefill_prefix_tokens"] = json!(cpu_steps);
        report["cpu_reference_kv_bytes"] = json!(LAYERS * 2 * cpu_steps * 8 * 128 * 4);
        report["token_artifact_maximum_bytes"] = json!(256 * 1024);
        report["remaining_prompt_tokens"] = json!(tokens.len() - cpu_steps);
        report["checked_input_tokens"] = json!(checked_tokens);
        report["prompt_token_identity_policy"] = json!("whole token artifact and captured raw request cryptographically bound by frozen supervisor contract");
        report["full_prompt_prefill_qualification"] = json!(false);
        report["serving_throughput_qualification"] = json!(false);
    }
    if cpu {
        let cpu_started = Instant::now();
        let source = CountingSource::new(&checkpoint);
        let reference_path = options
            .reference_out
            .as_ref()
            .context("missing reference output")?;
        let mut writer = if catalog {
            ReferenceWriter::create_catalog(reference_path, &identity, checked_tokens)?
        } else {
            ReferenceWriter::create(reference_path, &identity)?
        };
        let mut step_micros = Vec::with_capacity(cpu_steps);
        let mut selected = Vec::with_capacity(checked_positions.len());
        {
            // The bounded CPU prefix sees the same tokens as native.
            // Its capacity only limits cache growth; it does not change RoPE or
            // visible attention values. Do not reserve C16384 CPU KV here.
            let mut oracle = Oracle::with_capacity(&checkpoint, &source, shape, cpu_steps);
            for position in 0..cpu_steps {
                let step_started = Instant::now();
                let logits = oracle.step(tokens[position])?;
                step_micros.push(step_started.elapsed().as_micros());
                ensure!(
                    logits.len() == VOCAB && logits.iter().all(|v| v.is_finite()),
                    "nonfinite or wrong-size C16384 CPU logits"
                );
                if checked_positions.contains(&position) {
                    selected.push(greedy(&logits)?);
                    writer.append(position, &logits)?;
                }
                // The full vector is dropped every iteration. Only the bounded
                // writer buffer and the oracle's growing KV remain in this job.
                println!("PROGRESS {label} cpu_evaluations={}", position + 1);
            }
            let counts = source.counts();
            let lengths: Vec<_> = oracle
                .caches
                .iter()
                .map(|(k, v)| (k.len(), v.len()))
                .collect();
            let expected_cache_values = cpu_steps * shape.kv_heads * shape.head;
            ensure!(
                oracle.position == cpu_steps
                    && lengths.len() == LAYERS
                    && lengths
                        .iter()
                        .all(|&pair| pair == (expected_cache_values, expected_cache_values))
                    && counts == oracle_counts(cpu_steps)?
                    && selected.len() == checked_positions.len(),
                "C16384 CPU source/cache/reference accounting differs"
            );
            // Oracle rejects at entry, before a load or cache mutation. Check the
            // cursor, all cache extents and source counters without cloning KV.
            ensure!(
                oracle
                    .step(if catalog {
                        tokens[0]
                    } else {
                        u32::try_from(CPU_STEPS + 1)?
                    })
                    .is_err()
                    && oracle.position == cpu_steps
                    && oracle
                        .caches
                        .iter()
                        .map(|(k, v)| (k.len(), v.len()))
                        .eq(lengths)
                    && source.counts() == counts,
                "C16384 CPU capacity refusal changed source/cache state"
            );
        }
        ensure!(
            source.counts() == oracle_counts(cpu_steps)?,
            "C16384 CPU source total differs"
        );
        // Publication is last: incomplete, failed or refused work leaves no final
        // reference path. CPU process retirement is still an external gate.
        require_scope(stage)?;
        writer.finish()?;
        report["execution"] = json!(options.mode.execution());
        report["cpu_evaluations"] = json!(cpu_steps);
        report["evaluations"] = json!(0);
        report["exported_positions"] = json!(checked_positions.len());
        report["exported_values"] = json!(checked_positions.len() * VOCAB);
        report["greedy_tokens"] = json!(selected);
        report["cpu_oracle_source_reads"] = json!(source.counts().0);
        report["cpu_oracle_source_bytes"] = json!(source.counts().1);
        report["capacity_refusal_checked"] = json!(false);
        report["cpu_reference_capacity_refusal_checked"] = json!(true);
        report["cpu_reference_capacity_refused_without_source_reads"] = json!(true);
        report["reference_published"] = json!(true);
        report["reference_identity_attestation_written"] = json!(true);
        report["device_opened"] = json!(false);
        report["physical_dispatches"] = json!(0);
        report["logical_submissions"] = json!(0);
        report["verification_readback_bytes"] = json!(0);
        report["cpu_reference_ms"] = json!(cpu_started.elapsed().as_millis());
        if catalog {
            report["cpu_step_microseconds"] = json!(step_micros);
        }
        return emit_report(stage, &report, catalog);
    }

    // Parse every finite vector and identity before opening a native device.
    // This branch deliberately has no Oracle and cannot regenerate references.
    let reference_path = options
        .reference
        .as_ref()
        .context("missing native reference")?;
    let expected = if catalog {
        ReferenceSnapshot::read_catalog(reference_path, &identity, checked_tokens)?
    } else {
        ReferenceSnapshot::read(reference_path, &identity)?
    };
    admit_phase(stage, checkpoint_bytes, prepared_metadata, cpu_steps)?;
    let native_steps = cpu_steps;
    let evaluations = native_steps + 1;
    let comparisons = checked_positions.len() + 1;
    let planned_recording = total_recording_bytes(native_steps)?;
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
            largest_staging_bytes: MODEL_STAGING,
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
        "C16384 preparation accounting/state differs"
    );
    let native_started = Instant::now();
    let mut actual = vec![f32::NAN; VOCAB];
    ensure!(
        resident.read_logits(&device, &mut actual).is_err()
            && device.stats()? == initial
            && actual.iter().all(|v| v.is_nan())
            && !resident.is_poisoned(),
        "C16384 stale initial logits were readable or mutated state"
    );
    let mut maximum = 0.0_f64;
    let mut signs = 0u64;
    let mut zeros = 0u64;
    let mut selected = Vec::with_capacity(comparisons);
    let mut step_micros = Vec::with_capacity(native_steps);
    for position in 0..native_steps {
        let step_started = Instant::now();
        let before = device.stats()?;
        resident.step(&device, tokens[position])?;
        verify_step(before, device.stats()?, position + 1)?;
        ensure!(
            resident.visible_positions() == position + 1
                && !resident.is_poisoned()
                && budget.current_bytes() == HAG_BYTES,
            "C16384 native cursor/charge differs"
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
            println!("PASS {label} position={position} checked_values={VOCAB}");
        }
        step_micros.push(step_started.elapsed().as_micros());
        println!("PROGRESS {label} native_evaluations={}", position + 1);
    }
    {
        let full = device.stats()?;
        resident.reset()?;
        actual.fill(f32::NAN);
        ensure!(
            resident.visible_positions() == 0
                && !resident.is_poisoned()
                && resident.read_logits(&device, &mut actual).is_err()
                && actual.iter().all(|v| v.is_nan())
                && device.stats()? == full,
            "C16384 reset changed transport or exposed stale logits"
        );
        resident.step(&device, tokens[0])?;
        verify_step(full, device.stats()?, 1)?;
        ensure!(
            resident.visible_positions() == 1 && !resident.is_poisoned(),
            "C16384 replay cursor differs"
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
        "C16384 final execution/traffic/source counters differ"
    );
    resident.reset()?;
    actual.fill(f32::NAN);
    ensure!(
        resident.visible_positions() == 0
            && !resident.is_poisoned()
            && resident.read_logits(&device, &mut actual).is_err()
            && actual.iter().all(|v| v.is_nan())
            && device.stats()? == final_stats,
        "C16384 final reset or stale-logit refusal differs"
    );
    drop(resident);
    let retired = device.stats()?;
    ensure!(
        retired.weight_live_bytes == 0
            && retired.tensor_live_bytes == 0
            && retired.auxiliary_live_bytes == 0
            && !retired.poisoned,
        "C16384 native resources did not retire"
    );
    drop(device);
    ensure!(budget.current_bytes() == 0, "C16384 HAG charge retained");
    require_scope(stage)?;
    report["execution"] = json!(options.mode.execution());
    report["evaluations"] = json!(evaluations);
    report["cpu_evaluations"] = json!(0);
    report["reference_cpu_evaluations"] = json!(cpu_steps);
    report["reference_validated_before_device_open"] = json!(true);
    report["reference_identity_matched"] = json!(true);
    report["device_opened"] = json!(true);
    report["checked_values"] = json!(comparisons * VOCAB);
    report["greedy_tokens"] = json!(selected);
    report["greedy_matches"] = json!(comparisons);
    report["max_absolute_error"] = json!(maximum);
    report["logit_sign_mismatches"] = json!(signs);
    report["logit_zero_mismatches"] = json!(zeros);
    report["capacity_refusal_checked"] = json!(false);
    report["capacity_refused_without_dispatch"] = json!(false);
    report["reset_replay_passed"] = json!(true);
    report["stale_logits_refused"] = json!(true);
    report["physical_dispatches"] = json!(final_stats.dispatch_count);
    report["logical_submissions"] = json!(final_stats.logical_submissions);
    report["native_preparation_source_reads"] = json!(PREPARATION_READS);
    report["native_preparation_source_bytes"] = json!(SOURCE_BYTES);
    report["cpu_oracle_source_reads"] = json!(0);
    report["cpu_oracle_source_bytes"] = json!(0);
    report["actual_hag_allocation_bytes"] = json!(HAG_BYTES);
    report["initial_host_write_bytes"] = json!(initial.host_write_bytes);
    report["preparation_physical_dispatches"] = json!(initial.dispatch_count);
    report["preparation_dispatch_attempts"] = json!(initial.dispatch_attempts);
    report["preparation_logical_submissions"] = json!(initial.logical_submissions);
    report["preparation_host_read_bytes"] = json!(initial.host_read_bytes);
    report["recording_host_write_bytes"] = json!(planned_recording);
    report["actual_auxiliary_peak_bytes"] = json!(final_stats.auxiliary_peak_bytes);
    report["submission_tensor_transfer_bytes"] = json!(0);
    report["verification_readback_bytes"] = json!(final_stats.host_read_bytes);
    report["auxiliary_live_bytes_after_steps"] = json!(final_stats.auxiliary_live_bytes);
    report["cache_visible_positions_after_reset"] = json!(0);
    report["budget_after"] = json!(0);
    report["retired_weight_live_bytes"] = json!(retired.weight_live_bytes);
    report["retired_tensor_live_bytes"] = json!(retired.tensor_live_bytes);
    report["retired_auxiliary_live_bytes"] = json!(retired.auxiliary_live_bytes);
    report["native_preparation_ms"] = json!(preparation_ms);
    report["native_execution_and_checks_ms"] = json!(native_ms);
    if catalog {
        report["native_step_and_checks_microseconds"] = json!(step_micros);
    }
    emit_report(stage, &report, catalog)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_capacity_allocation_and_two_position_accounting_are_exact() {
        let shape = super::super::super::tests::shape();
        let memory = DenseModelMemory {
            packed_weight_bytes: PACKED,
            resident_tensor_bytes: TENSOR_BYTES,
            largest_staging_bytes: MODEL_STAGING,
        };
        assert_eq!(
            execution_config(shape, memory)
                .unwrap()
                .allocation_bytes()
                .unwrap(),
            4_629_168_128
        );
        assert_eq!(PREPARATION_WRITES, 4_864_071_508);
        assert_eq!(total_recording_bytes(CPU_STEPS).unwrap(), 345_084);
        assert_eq!(total_recording_bytes(32).unwrap(), 3_795_924);
        assert_eq!(
            oracle_counts(32).unwrap(),
            (11_136, 20_271_892_480, STAGING)
        );
        assert_eq!(33 * PHYSICAL_PER_STEP, 125_169);
        assert_eq!(33 * SUBMISSIONS_PER_STEP, 1_914);
        assert_eq!(
            PERSISTENT_AUX + attention_scratch_bytes(CPU_STEPS).unwrap(),
            123_232
        );
        assert_eq!(
            PERSISTENT_AUX + attention_scratch_bytes(CONTEXT).unwrap(),
            u64::from(PEAK_AUX)
        );
        assert_eq!(
            oracle_counts(CPU_STEPS).unwrap(),
            (696, 1_266_993_280, STAGING)
        );
        assert_eq!((CPU_STEPS + 1) as u64 * PHYSICAL_PER_STEP, 11_379);
        assert_eq!((CPU_STEPS + 1) as u64 * SUBMISSIONS_PER_STEP, 174);
        assert_eq!((CPU_STEPS + 1) * VOCAB * 4, 1_823_232);
        assert!(oracle_counts(CONTEXT).is_err());
        assert!(execution_config(
            Shape {
                context: CONTEXT - 1,
                ..shape
            },
            memory
        )
        .is_err());
        assert!(execution_config(
            shape,
            DenseModelMemory {
                resident_tensor_bytes: TENSOR_BYTES - 4,
                ..memory
            }
        )
        .is_err());
        assert!(execution_config(
            shape,
            DenseModelMemory {
                largest_staging_bytes: MODEL_STAGING + 4,
                ..memory
            }
        )
        .is_err());
        assert!(execution_config(
            shape,
            DenseModelMemory {
                largest_staging_bytes: STAGING,
                ..memory
            }
        )
        .is_err());
    }

    #[test]
    fn counters_reject_wrong_allocation_retained_scratch_and_extra_transfers() {
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
                allocation_bytes: HAG_BYTES - 4096,
                ..after
            },
            NativeExecutionStats {
                tensor_live_bytes: TENSOR_BYTES - 4,
                ..after
            },
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
    }

    #[test]
    fn comparison_requires_full_finite_tolerance_and_exact_greedy() {
        let mut maximum = 0.0;
        let mut signs = 0;
        let mut zeros = 0;
        let mut selected = Vec::new();
        check_logits(
            &[0.0, 1.0001],
            &[0.0, 1.0],
            &mut maximum,
            &mut signs,
            &mut zeros,
            &mut selected,
        )
        .unwrap();
        for (actual, expected) in [
            ([0.0, f32::NAN], [0.0, 1.0]),
            ([0.0, f32::INFINITY], [0.0, 1.0]),
            ([0.0, 2.0], [0.0, 1.0]),
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
        assert_eq!(selected, [1]);
    }

    #[test]
    fn separate_process_cli_requires_reference_direction_and_identity() {
        let parse_args = |args: &[&str]| parse(args.iter().map(OsString::from));
        let cpu = [
            "--capacity16384-reference",
            "--model",
            "/model",
            "--reference-out",
            "/out",
            "--input-identity",
            "/identity",
        ];
        let native = [
            "--capacity16384-smoke",
            "--model",
            "/model",
            "--reference",
            "/in",
            "--input-identity",
            "/identity",
            "/dev/dri/renderD128",
        ];
        let options = parse_args(&cpu).unwrap();
        assert_eq!(options.mode, Mode::Capacity16384Reference);
        assert!(!options.mode.executes());
        assert_eq!(options.mode.steps(), 2);
        let options = parse_args(&native).unwrap();
        assert_eq!(options.mode, Mode::Capacity16384Smoke);
        assert!(options.mode.executes());
        assert_eq!(options.mode.steps(), 3);
        assert_eq!(options.mode.requests(), 2);
        assert!(parse_args(&cpu[..5]).is_err());
        assert!(parse_args(&native[..7]).is_err());
        let mut invalid = cpu.to_vec();
        invalid.push("/dev/dri/renderD128");
        assert!(parse_args(&invalid).is_err());
        let mut invalid = native.to_vec();
        invalid.extend(["--reference-out", "/out"]);
        assert!(parse_args(&invalid).is_err());
        let mut invalid = cpu.to_vec();
        invalid.push("--on-demand2048-reference");
        assert!(parse_args(&invalid).is_err());
    }
}
