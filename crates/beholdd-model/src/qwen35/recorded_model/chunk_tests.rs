//! Hardware-free metadata/publication/order tests. These do not open a native
//! queue, execute arithmetic or establish device retirement/numerical parity.

use super::*;
use crate::qwen35::config::tests::Fixture;

fn checker() -> FiniteCheckSpec {
    NativeExecutionProfile::Qwen35Probe
        .finite_check_spec()
        .unwrap()
}

#[test]
fn chunk_is_explicit_fixed_size_and_small_scalar_capacities_remain_valid() {
    let (_file, gguf) = Fixture::hybrid().open();
    let binding = Binding::load(&gguf).unwrap();
    for capacity in 1..=8 {
        let plan = RecordedQwen35Plan::new(&binding, capacity, ReadLimits::default()).unwrap();
        let original = plan.checked_requirements(Some(checker())).unwrap();
        assert_eq!(plan.prefill_chunk_size(), None);
        assert_eq!(
            plan.checked_chunk_requirements(Some(checker())).unwrap(),
            None
        );
        assert!(plan.chunk_next(&[7], 0).is_err());
        for count in [0, 3, 5, 8, usize::MAX] {
            assert!(plan.clone().with_prefill_chunk(count).is_err());
        }
        for count in [1, 2, 4] {
            let configured = plan.clone().with_prefill_chunk(count);
            assert_eq!(configured.is_ok(), count <= capacity);
            if let Ok(configured) = configured {
                assert_eq!(configured.prefill_chunk_size(), Some(count));
                assert!(configured.clone().with_prefill_chunk(count).is_err());
                assert!(configured.checked_requirements(None).is_err());
                assert!(configured.checked_chunk_requirements(None).is_err());
                // The scalar baseline remains an independently available
                // requirement; opting in did not mutate the original plan.
                assert_eq!(
                    configured
                        .requirements_with_chunk(Some(checker()), None)
                        .unwrap(),
                    original
                );
            }
        }
        assert_eq!(
            plan.checked_requirements(Some(checker())).unwrap(),
            original
        );
    }
}

#[test]
fn chunk_charges_one_workspace_and_only_the_tensor_arena_page_delta() {
    let (_file, gguf) = Fixture::hybrid().open();
    let binding = Binding::load(&gguf).unwrap();
    let plan = RecordedQwen35Plan::new(&binding, 8, ReadLimits::default()).unwrap();
    let scalar = plan.checked_requirements(Some(checker())).unwrap();
    for count in [1, 2, 4] {
        let chunk = plan.clone().with_prefill_chunk(count).unwrap();
        let r = chunk.checked_requirements(Some(checker())).unwrap();
        let extra = chunk
            .checked_chunk_requirements(Some(checker()))
            .unwrap()
            .unwrap();
        let payload =
            (count * (2 * binding.config().dim + 2 * binding.config().feed_forward) * 4) as u64;
        assert_eq!(extra.extra_tensor_allocations, 1);
        assert_eq!(extra.extra_resident_tensor_bytes, payload);
        assert_eq!(extra.initialization_write_bytes, payload);
        assert_eq!(extra.reset_write_bytes, payload);
        assert_eq!(
            r.resident_tensor_bytes,
            scalar.resident_tensor_bytes + payload
        );
        assert_eq!(r.reset_write_bytes, scalar.reset_write_bytes + payload);
        assert_eq!(
            r.config.weight_capacity_bytes,
            scalar.config.weight_capacity_bytes
        );
        assert_eq!(
            r.config.auxiliary_capacity_bytes,
            scalar.config.auxiliary_capacity_bytes
        );
        let page = |n: u32| u64::from(n).div_ceil(4096) * 4096;
        assert_eq!(
            extra.extra_device_charge_bytes,
            page(r.config.tensor_capacity_bytes) - page(scalar.config.tensor_capacity_bytes)
        );
        assert_eq!(
            r.device_charge_bytes,
            scalar.device_charge_bytes + extra.extra_device_charge_bytes
        );
        assert_eq!(extra.extra_host_payload_bytes, 0);
        assert_eq!(r.host_payload_bytes, scalar.host_payload_bytes);
        assert!(extra.extra_recording_metadata_bytes > 0);
        assert_eq!(
            r.recording_metadata_bytes,
            scalar.recording_metadata_bytes + extra.extra_recording_metadata_bytes
        );
        assert_eq!(
            extra.extra_model_metadata_bytes,
            chunk.chunk.unwrap().memory().host_metadata_bytes
        );
        // Full-head scalar traffic and checker requirements remain unchanged.
        assert_eq!(r.packed_tiles_per_step, scalar.packed_tiles_per_step);
        assert_eq!(r.packed_bytes_per_step, scalar.packed_bytes_per_step);
        assert_eq!(r.finite_checks_per_step, scalar.finite_checks_per_step);
        assert_eq!(r.checker, scalar.checker);
    }
}

#[test]
fn chunk_admission_refuses_scalar_or_short_device_and_process_limits_without_charge() {
    let (_file, gguf) = Fixture::hybrid().open();
    let binding = Binding::load(&gguf).unwrap();
    let plan = RecordedQwen35Plan::new(&binding, 8, ReadLimits::default()).unwrap();
    let scalar = plan.checked_requirements(Some(checker())).unwrap();
    let plan = plan.with_prefill_chunk(4).unwrap();
    let r = plan.checked_requirements(Some(checker())).unwrap();
    let budget = MemoryBudget::new(r.device_charge_bytes);
    let limits = ModelLimits {
        device_bytes: r.device_charge_bytes,
        total_process_bytes: u64::MAX,
        process_headroom_bytes: 1234,
    };
    assert!(plan.admit(&gguf, scalar, limits, &budget).is_err());
    assert!(plan
        .admit(
            &gguf,
            r,
            ModelLimits {
                device_bytes: r.device_charge_bytes - 1,
                ..limits
            },
            &budget
        )
        .is_err());
    let admission = plan.admit(&gguf, r, limits, &budget).unwrap();
    assert!(plan
        .admit(
            &gguf,
            r,
            ModelLimits {
                total_process_bytes: admission.total_process_bytes - 1,
                ..limits
            },
            &budget
        )
        .is_err());
    let short = MemoryBudget::new(r.device_charge_bytes - 1);
    assert!(plan.admit(&gguf, r, limits, &short).is_err());
    assert_eq!(
        admission.source_cache_bytes + admission.source_resident_bytes,
        0
    );
    assert_eq!(budget.current_bytes(), 0);
    assert_eq!(short.current_bytes(), 0);
}

#[test]
fn chunk_traffic_reads_ffn_tiles_once_but_preserves_every_scalar_application() {
    let (_file, gguf) = Fixture::hybrid().open();
    let binding = Binding::load(&gguf).unwrap();
    let reads = ReadLimits::default();
    let scalar = RecordedQwen35Plan::new(&binding, 8, reads).unwrap();
    let r = scalar.checked_requirements(Some(checker())).unwrap();
    let head = PackedRowPlan::new(binding.output(), reads).unwrap();
    let head_tiles = head.rows().div_ceil(head.tile_rows()) as u64;
    let advance_tiles = r.packed_tiles_per_step - head_tiles;
    let advance_bytes = r.packed_bytes_per_step - binding.output().nbytes();
    let mut ffn_tiles = 0;
    let mut ffn_bytes = 0;
    let mut scaled_tiles = 0;
    let mut arithmetic_dispatches = 0;
    let mut arithmetic_submissions = 0;
    let c = binding.config();
    let norm = 1 + c.dim.div_ceil(4096) as u64;
    for block in binding.blocks() {
        for tensor in [&block.ffn.gate, &block.ffn.up, &block.ffn.down] {
            let rows = PackedRowPlan::new(tensor, reads).unwrap();
            ffn_tiles += rows.rows().div_ceil(rows.tile_rows()) as u64;
            ffn_bytes += tensor.nbytes();
        }
        let (output, core_dispatches, core_submissions) = match &block.attention {
            Attention::Full(core) => (
                &core.output,
                norm + 3 * (c.n_heads + c.n_kv_heads) as u64 + 4 * c.n_heads as u64,
                4,
            ),
            Attention::DeltaNet(core) => (&core.output, norm + 4, 2),
        };
        for tensor in [output, &block.ffn.down] {
            let rows = PackedRowPlan::new(tensor, reads).unwrap();
            scaled_tiles += rows.rows().div_ceil(rows.tile_rows()) as u64;
        }
        arithmetic_dispatches += core_dispatches + norm + c.feed_forward.div_ceil(4096) as u64;
        arithmetic_submissions += core_submissions + 2;
    }
    let checks = r.finite_checks_per_step - 1 - c.vocab.div_ceil(checker().max_elements) as u64;
    for count in [1, 2, 4] {
        let q = scalar
            .clone()
            .with_prefill_chunk(count)
            .unwrap()
            .checked_chunk_requirements(Some(checker()))
            .unwrap()
            .unwrap();
        let b = count as u64;
        assert_eq!(
            q.packed_source_tiles,
            b * advance_tiles - (b - 1) * ffn_tiles
        );
        assert_eq!(
            q.packed_source_bytes,
            b * advance_bytes - (b - 1) * ffn_bytes
        );
        assert_eq!(q.packed_submissions, b * advance_tiles);
        assert_eq!(q.ffn_tile_applications, b * ffn_tiles);
        assert_eq!(
            q.physical_dispatches,
            b * (advance_tiles
                + scaled_tiles
                + arithmetic_dispatches
                + checks * checker().physical_dispatches)
        );
        assert_eq!(
            q.logical_submissions,
            b * (advance_tiles + arithmetic_submissions + checks * checker().logical_submissions)
        );
        assert_eq!(q.finite_checks, b * checks);
        assert_eq!(
            q.finite_pending_bytes,
            b * checks * checker().pending_write_bytes
        );
        assert_eq!(
            q.status_read_bytes,
            b * (checks * checker().status_read_bytes + r.delta_status_bytes_per_step)
        );
        assert_eq!(
            q.extra_ffn_recording_write_bytes,
            b * c.n_layers as u64 * (4 * c.dim as u64 + 8)
        );
        assert_eq!(q.avoided_packed_write_bytes, (b - 1) * ffn_bytes);
        if count > 1 {
            assert!(q.packed_source_bytes < b * advance_bytes);
            assert!(q.packed_source_tiles < q.packed_submissions);
        }
    }
}

#[test]
fn all_chunk_token_and_capacity_refusals_precede_publication_changes() {
    let (_file, gguf) = Fixture::hybrid().open();
    let binding = Binding::load(&gguf).unwrap();
    let plan = RecordedQwen35Plan::new(&binding, 8, ReadLimits::default())
        .unwrap()
        .with_prefill_chunk(4)
        .unwrap();
    let mut publication = Publication::new();
    let old = publication.commit(1);
    let identity = (
        publication.epoch,
        publication.generation,
        publication.position,
    );
    for tokens in [vec![], vec![7], vec![7, 42], vec![7; 3], vec![7; 5]] {
        assert!(plan.chunk_next(&tokens, publication.position).is_err());
    }
    for index in 0..4 {
        let mut tokens = [7; 4];
        tokens[index] = binding.config().vocab as u32;
        assert!(plan.chunk_next(&tokens, publication.position).is_err());
        tokens[index] = u32::MAX;
        assert!(plan.chunk_next(&tokens, publication.position).is_err());
    }
    assert_eq!(plan.chunk_next(&[7, 42, 7, 42], 4).unwrap(), 8);
    assert!(plan.chunk_next(&[7, 42, 7, 42], 5).is_err());
    assert!(plan.chunk_next(&[7, 42, 7, 42], usize::MAX).is_err());
    assert_eq!(
        (
            publication.epoch,
            publication.generation,
            publication.position
        ),
        identity
    );
    publication.accept(&old).unwrap();
    assert!(!publication.poisoned);
}

#[test]
fn production_chunk_schedule_has_chronological_cores_and_layer_ffn_barriers() {
    for count in [1, 2, 4] {
        let mut embedded = [false; 4];
        let mut cores = [3usize; 4];
        let mut ffns = [3usize; 4];
        let mut trace = Vec::new();
        execute_chunk_stages(PrefillChunkSize::new(count).unwrap(), 4, |stage| {
            trace.push(stage);
            match stage {
                ChunkStage::Embedding(token) => {
                    assert!(!embedded[token]);
                    embedded[token] = true;
                    assert_eq!(cores, [3; 4]);
                }
                ChunkStage::Core { layer, token } => {
                    assert!(embedded[..count].iter().all(|v| *v));
                    assert_eq!(cores[layer], 3 + token);
                    assert_eq!(ffns[layer], 3);
                    if layer > 0 {
                        assert_eq!(ffns[layer - 1], 3 + count);
                    }
                    cores[layer] += 1;
                }
                ChunkStage::Ffn(layer) => {
                    assert_eq!(cores[layer], 3 + count);
                    assert_eq!(ffns[layer], 3);
                    ffns[layer] += count;
                }
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(trace.len(), count + 4 * (count + 1));
        assert_eq!(cores, [3 + count; 4]);
        assert_eq!(ffns, [3 + count; 4]);
    }
}

#[test]
fn every_partial_chunk_stage_and_final_barrier_keeps_old_position_poisoned() {
    for count in [1, 2, 4] {
        let stages = count + 4 * (count + 1);
        // Includes refusal immediately before final whole-chunk publication,
        // after all mock components have already reached the end position.
        for fail_at in 0..=stages {
            let mut p = Publication::new();
            let old = p.commit(1);
            p.invalidate().unwrap();
            let mut calls = 0;
            let result = execute_chunk_stages(PrefillChunkSize::new(count).unwrap(), 4, |_| {
                ensure!(calls != fail_at, "injected stage failure");
                calls += 1;
                assert_eq!(p.position, 1);
                assert!(p.poisoned && !p.published);
                Ok(())
            })
            .and_then(|()| {
                ensure!(calls != fail_at, "injected final barrier failure");
                p.commit_state(1 + count);
                Ok(())
            });
            assert!(result.is_err());
            assert_eq!(calls, fail_at);
            assert_eq!(p.position, 1);
            assert!(p.poisoned && !p.published);
            assert!(p.accept(&old).is_err());
            p.invalidate().unwrap(); // Incomplete physical reset cannot commit.
            assert_eq!(p.position, 1);
            assert!(p.poisoned);
            p.reset_complete(1); // Represents success of every physical clearing.
            assert_eq!(p.position, 0);
            assert!(!p.poisoned && !p.published);
            assert!(p.accept(&old).is_err());
        }
    }
}

#[test]
fn successful_chunk_then_scalar_tail_and_head_never_revives_stale_tickets() {
    for count in [1, 2, 4] {
        let mut p = Publication::new();
        let old = p.commit(1);
        p.invalidate().unwrap();
        execute_chunk_stages(PrefillChunkSize::new(count).unwrap(), 4, |_| {
            assert_eq!(p.position, 1);
            assert!(p.poisoned && !p.published);
            Ok(())
        })
        .unwrap();
        p.commit_state(1 + count);
        assert!(p.accept(&old).is_err());
        let fake = LogitsTicket {
            owner: p.owner.clone(),
            epoch: p.epoch,
            generation: p.generation,
            position: p.position,
        };
        assert!(p.accept(&fake).is_err());
        p.invalidate().unwrap();
        p.commit_state(2 + count); // Ordinary scalar no-head tail.
        p.invalidate().unwrap();
        let final_head = p.commit(3 + count);
        p.accept(&final_head).unwrap();
        assert!(p.accept(&old).is_err());
        assert!(p.accept(&fake).is_err());
        assert_eq!(p.position, 3 + count);
    }
}
