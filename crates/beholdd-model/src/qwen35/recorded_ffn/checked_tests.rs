// Included in recorded_ffn::tests so these checks reuse the independent GGUF
// fixture and immediate-Device oracle rather than a second FFN implementation.

fn chunk_limits(plan: FfnChunkPlan) -> FfnChunkLimits {
    let m = plan.memory();
    FfnChunkLimits {
        resident_tensor_bytes: m.resident_tensor_bytes,
        host_staging_bytes: m.host_staging_bytes,
        host_metadata_bytes: m.host_metadata_bytes,
    }
}

fn chunk_inputs(
    device: &dyn ExecutionDevice,
    workspace: &FfnChunkWorkspace,
) -> Result<Vec<Vec<f32>>> {
    let mut inputs = Vec::new();
    for token in 0..workspace.plan.tokens() {
        let input: Vec<f32> = (0..workspace.plan.dim())
            .map(|i| (i % 11) as f32 * 0.1 - 0.3 + token as f32 * 0.2)
            .collect();
        device.write_f32(&workspace.hidden(token)?, &input)?;
        inputs.push(input);
    }
    Ok(inputs)
}

#[test]
fn chunk_workspace_bounds_accounting_and_admission_are_metadata_only() -> Result<()> {
    for (count, bytes) in [(1, 180_224), (2, 360_448), (4, 720_896)] {
        let p = FfnChunkPlan::new(5120, 17408, count)?;
        let m = p.memory();
        assert_eq!(m.resident_tensor_bytes, bytes);
        assert_eq!(
            (m.initialization_write_bytes, m.reset_write_bytes),
            (bytes, bytes)
        );
        assert_eq!(m.hidden_bytes, (count * 5120 * 4) as u64);
        assert_eq!(
            (
                m.tensor_allocations,
                m.max_live_recordings,
                m.max_batch_commands
            ),
            (1, 1, 5)
        );
        assert_eq!(m.host_staging_bytes, 65_536);
        assert_eq!(
            m.host_metadata_bytes,
            (std::mem::size_of::<FfnChunkWorkspace>() + 29 * std::mem::size_of::<F32TensorView>())
                as u64
        );
    }
    for count in [0, 3, 5, usize::MAX] {
        assert!(FfnChunkPlan::new(256, 256, count).is_err());
    }
    for (dim, width) in [
        (0, 256),
        (255, 256),
        (5376, 256),
        (256, 257),
        (256, 17664),
        (usize::MAX, usize::MAX),
    ] {
        assert!(FfnChunkPlan::new(dim, width, 2).is_err());
    }
    let device = CpuExecutionDevice::new();
    let p = FfnChunkPlan::new(256, 256, 2)?;
    let good = chunk_limits(p);
    for limits in [
        FfnChunkLimits {
            resident_tensor_bytes: good.resident_tensor_bytes - 1,
            ..good
        },
        FfnChunkLimits {
            host_staging_bytes: good.host_staging_bytes - 1,
            ..good
        },
        FfnChunkLimits {
            host_metadata_bytes: good.host_metadata_bytes - 1,
            ..good
        },
    ] {
        assert!(p
            .prepare(&device, limits, &mut || panic!(
                "inadmissible chunk reached effects"
            ))
            .is_err());
    }
    let w = p.prepare(&device, good, &mut || Ok(()))?;
    assert_eq!(w.packet.len(), 2048);
    assert!(!w.hidden(0)?.overlaps(&w.hidden(1)?));
    assert!(w.hidden(2).is_err());
    Ok(())
}

#[test]
fn checked_chunk_b1_b2_b4_match_scalar_and_independent_ffn_oracle_with_one_read_pass() -> Result<()>
{
    for count in [1, 2, 4] {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        let source = Source::new(&gguf);
        let oracle_source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let mut scalar =
            plan.prepare_checked(&device, &oracle_source, admitted(&plan), &mut || Ok(()))?;
        let p = FfnChunkPlan::new(plan.dim(), plan.width(), count)?;
        let mut w = p.prepare(&device, chunk_limits(p), &mut || Ok(()))?;
        let mut stats = StreamStats::default();
        let mut chunks = ChunkStreamStats::default();
        let tiles = checked_tiles(&plan, 3) as u64;
        for iteration in 1..=2 {
            let inputs = chunk_inputs(&device, &w)?;
            resident.step_chunk_checked(
                &device,
                &source,
                &mut w,
                &mut checker,
                &mut stats,
                &mut chunks,
                &mut || Ok(()),
            )?;
            for (token, input) in inputs.iter().enumerate() {
                let actual = read(&device, &w.hidden(token)?)?;
                close(&actual, &reference(&binding, &oracle_source, input)?);
                let scalar_hidden = device.upload_f32(input)?;
                scalar.step_checked(
                    &device,
                    &oracle_source,
                    &scalar_hidden.full_view(),
                    &mut checker,
                    &mut StreamStats::default(),
                    &mut || Ok(()),
                )?;
                let scalar_values = read(&device, &scalar_hidden.full_view())?;
                assert!(actual
                    .iter()
                    .zip(scalar_values)
                    .all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            assert_eq!(resident.completed_steps(), iteration * count as u64);
            assert!(!resident.is_poisoned() && !w.is_poisoned());
            assert_eq!(stats.completed_tiles, iteration * tiles);
            assert_eq!(stats.submission_attempts, iteration * tiles * count as u64);
            assert_eq!(chunks.completed_applications, stats.submission_attempts);
            assert_eq!(source.reads.load(Ordering::Relaxed), 1 + iteration * tiles);
            assert_eq!(stats.returned_read_bytes, stats.uploaded_bytes);
        }
    }
    Ok(())
}

#[test]
fn chunk_second_entry_nonfinite_refuses_before_any_arithmetic_or_source_read() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let mut checker = device.prepare_finite_check()?;
    let mut resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    let p = FfnChunkPlan::new(plan.dim(), plan.width(), 2)?;
    let mut w = p.prepare(&device, chunk_limits(p), &mut || Ok(()))?;
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        chunk_inputs(&device, &w)?;
        device.write_f32(&w.hidden(1)?.view(255, 1)?, &[bad])?;
        let before: Vec<u32> = read(&device, &w.packet.full_view())?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let mut stats = StreamStats::default();
        let mut chunks = ChunkStreamStats::default();
        let error = resident
            .step_chunk_checked(
                &device,
                &source,
                &mut w,
                &mut checker,
                &mut stats,
                &mut chunks,
                &mut || Ok(()),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("entry hidden"));
        assert_eq!(resident.completed_steps(), 0);
        assert!(!resident.is_poisoned() && !w.is_poisoned());
        assert_eq!(stats, StreamStats::default());
        assert_eq!(chunks, ChunkStreamStats::default());
        assert_eq!(source.reads.load(Ordering::Relaxed), 1);
        assert_eq!(
            read(&device, &w.packet.full_view())?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            before
        );
    }
    Ok(())
}

#[test]
fn chunk_checks_second_raw_pair_before_activating_even_the_first_token() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let mut checker = device.prepare_finite_check()?;
    let mut resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    let p = FfnChunkPlan::new(plan.dim(), plan.width(), 2)?;
    let mut w = p.prepare(&device, chunk_limits(p), &mut || Ok(()))?;
    let original = chunk_inputs(&device, &w)?;
    let first_gate = w.views(0)?.gate;
    let second_raw = w.views(1)?.raw;
    let last_projection_read = 1 + checked_tiles(&plan, 2) as u64;
    let mut boundaries_after_read = 0;
    let mut first_raw_before = None;
    let mut stats = StreamStats::default();
    let mut chunks = ChunkStreamStats::default();
    let error = resident
        .step_chunk_checked(
            &device,
            &source,
            &mut w,
            &mut checker,
            &mut stats,
            &mut chunks,
            &mut || {
                if source.reads.load(Ordering::Relaxed) == last_projection_read {
                    boundaries_after_read += 1;
                    // After upload, three boundaries/application and full tile retirement.
                    // The fake corruption occurs only after both final up applications
                    // fenced. It isolates the second token's pre-activation finite gate.
                    if boundaries_after_read == 2 + 3 * p.tokens() {
                        first_raw_before = Some(read(&device, &first_gate)?);
                        device.write_f32(&second_raw.view(0, 1)?, &[f32::NAN])?;
                    }
                }
                Ok(())
            },
        )
        .unwrap_err();
    assert!(format!("{error:#}").contains("raw gate/up"));
    assert_eq!(
        read(&device, &first_gate)?,
        first_raw_before.context("raw injection boundary was not reached")?
    );
    assert_eq!(stats.completed_tiles, checked_tiles(&plan, 2) as u64);
    assert_eq!(chunks.completed_applications, 2 * stats.completed_tiles);
    for (token, input) in original.iter().enumerate() {
        assert_eq!(read(&device, &w.hidden(token)?)?, *input);
    }
    assert!(resident.is_poisoned() && w.is_poisoned());
    assert_eq!(resident.completed_steps(), 0);
    Ok(())
}

#[test]
fn chunk_geometry_foreign_owners_and_step_overflow_refuse_without_arithmetic() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let other = CpuExecutionDevice::new();
    let mut checker = device.prepare_finite_check()?;
    let mut resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    let p = FfnChunkPlan::new(plan.dim(), plan.width(), 2)?;
    let wrong = FfnChunkPlan::new(plan.dim(), plan.width() * 2, 2)?;
    let mut bad_shape = wrong.prepare(&device, chunk_limits(wrong), &mut || Ok(()))?;
    let mut foreign = p.prepare(&other, chunk_limits(p), &mut || Ok(()))?;
    for w in [&mut bad_shape, &mut foreign] {
        assert!(resident
            .step_chunk_checked(
                &device,
                &source,
                w,
                &mut checker,
                &mut StreamStats::default(),
                &mut ChunkStreamStats::default(),
                &mut || panic!("bad chunk reached control")
            )
            .is_err());
        assert!(!resident.is_poisoned() && !w.is_poisoned());
    }
    let mut w = p.prepare(&device, chunk_limits(p), &mut || Ok(()))?;
    chunk_inputs(&device, &w)?;
    let before = read(&device, &w.packet.full_view())?;
    let mut foreign_checker = other.prepare_finite_check()?;
    assert!(resident
        .step_chunk_checked(
            &device,
            &source,
            &mut w,
            &mut foreign_checker,
            &mut StreamStats::default(),
            &mut ChunkStreamStats::default(),
            &mut || Ok(())
        )
        .is_err());
    assert!(resident
        .step_chunk_checked(
            &other,
            &source,
            &mut w,
            &mut checker,
            &mut StreamStats::default(),
            &mut ChunkStreamStats::default(),
            &mut || Ok(())
        )
        .is_err());
    resident.completed_steps = u64::MAX - 1;
    assert!(resident
        .step_chunk_checked(
            &device,
            &source,
            &mut w,
            &mut checker,
            &mut StreamStats::default(),
            &mut ChunkStreamStats::default(),
            &mut || panic!("counter overflow reached control")
        )
        .is_err());
    assert_eq!(resident.completed_steps(), u64::MAX - 1);
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    assert_eq!(read(&device, &w.packet.full_view())?, before);
    assert!(!resident.is_poisoned() && !w.is_poisoned());
    Ok(())
}

#[test]
fn chunk_cancelled_mutation_and_final_boundary_require_both_resets_before_replay() -> Result<()> {
    for selected in [0, 1, 2] {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let p = FfnChunkPlan::new(plan.dim(), plan.width(), 2)?;
        let mut w = p.prepare(&device, chunk_limits(p), &mut || Ok(()))?;
        chunk_inputs(&device, &w)?;
        let mut total = 0;
        resident.step_chunk_checked(
            &device,
            &source,
            &mut w,
            &mut checker,
            &mut StreamStats::default(),
            &mut ChunkStreamStats::default(),
            &mut || {
                total += 1;
                Ok(())
            },
        )?;
        assert_eq!(resident.completed_steps(), 2);
        chunk_inputs(&device, &w)?;
        // The last entry finite callback, first normalization post-fence, and
        // last residual publication boundary respectively.
        let stop = [5 * p.tokens(), 5 * p.tokens() + 3, total][selected];
        let mut calls = 0;
        assert!(resident
            .step_chunk_checked(
                &device,
                &source,
                &mut w,
                &mut checker,
                &mut StreamStats::default(),
                &mut ChunkStreamStats::default(),
                &mut || {
                    calls += 1;
                    ensure!(calls != stop, "cancelled");
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(calls, stop);
        assert_eq!(resident.completed_steps(), 2);
        assert_eq!(resident.is_poisoned(), selected != 0);
        assert_eq!(w.is_poisoned(), selected != 0);
        if selected == 0 {
            continue;
        }
        resident.reset(&device, &mut || Ok(()))?;
        assert!(resident
            .step_chunk_checked(
                &device,
                &source,
                &mut w,
                &mut checker,
                &mut StreamStats::default(),
                &mut ChunkStreamStats::default(),
                &mut || panic!("poisoned shared workspace continued")
            )
            .is_err());
        assert!(w
            .reset(&device, &mut || anyhow::bail!("cancel reset"))
            .is_err());
        assert!(w.is_poisoned());
        w.reset(&device, &mut || Ok(()))?;
        assert_eq!(w.reset_write_bytes(), p.memory().reset_write_bytes);
        assert!(read(&device, &w.packet.full_view())?
            .iter()
            .all(|v| v.to_bits() == 0));
        chunk_inputs(&device, &w)?;
        resident.step_chunk_checked(
            &device,
            &source,
            &mut w,
            &mut checker,
            &mut StreamStats::default(),
            &mut ChunkStreamStats::default(),
            &mut || Ok(()),
        )?;
        assert_eq!(resident.completed_steps(), 2);
        assert!(!resident.is_poisoned() && !w.is_poisoned());
    }
    Ok(())
}

fn checked_tiles(plan: &RecordedFfnPlan, count: usize) -> usize {
    plan.matrices[..count]
        .iter()
        .map(|matrix| matrix.rows().div_ceil(matrix.tile_rows()))
        .sum()
}

fn checked_patch_scale(gguf: &Gguf, role: &str, bits: u16) -> Result<()> {
    let tensor = gguf.get(&format!("blk.0.ffn_{role}.weight")).unwrap();
    let offset = match tensor.kind().0 {
        12 => 0,
        14 => 208,
        other => anyhow::bail!("unexpected fixture quantization {other}"),
    };
    let mut file = OpenOptions::new().write(true).open(&gguf.shards()[0])?;
    file.seek(SeekFrom::Start(tensor.offset() + offset))?;
    file.write_all(&bits.to_le_bytes())?;
    Ok(())
}

fn checked_patch_norm(gguf: &Gguf, value: f32) -> Result<()> {
    assert!(value.is_finite());
    let tensor = gguf.get("blk.0.post_attention_norm.weight").unwrap();
    let mut file = OpenOptions::new().write(true).open(&gguf.shards()[0])?;
    file.seek(SeekFrom::Start(tensor.offset()))?;
    for _ in 0..256 {
        file.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

#[test]
fn checked_requirements_bound_raw_pair_and_keep_shared_costs_separate() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let memory = plan.memory();
    let cpu = device.finite_check_spec().unwrap();
    assert_eq!(plan.finite_checks_per_step(), 5);
    assert!(plan.finite_check_requirements(None).is_err());
    assert!(plan
        .finite_check_requirements(Some(FiniteCheckSpec {
            max_elements: 511,
            ..cpu
        }))
        .is_err());
    let requirements = plan.finite_check_requirements(Some(cpu))?;
    assert_eq!(requirements.max_view_elements, 512);
    assert_eq!(requirements.checks_per_step, 5);
    assert_eq!(requirements.shared_checker.host_scratch_bytes, 262_144);
    assert_eq!(requirements.physical_dispatches_per_step, 0);
    assert_eq!(requirements.logical_submissions_per_step, 0);
    assert_eq!(requirements.pending_write_bytes_per_step, 0);
    assert_eq!(requirements.status_read_bytes_per_step, 0);
    let native = FiniteCheckSpec {
        max_elements: 65_536,
        resident_tensor_bytes: 276,
        initialization_write_bytes: 548,
        host_scratch_bytes: 512,
        physical_dispatches: 1,
        logical_submissions: 1,
        pending_write_bytes: 256,
        status_read_bytes: 256,
    };
    let requirements = plan.finite_check_requirements(Some(native))?;
    assert_eq!(requirements.shared_checker, native); // one allocation/init, not five
    assert_eq!(requirements.physical_dispatches_per_step, 5);
    assert_eq!(requirements.logical_submissions_per_step, 5);
    assert_eq!(requirements.pending_write_bytes_per_step, 1280);
    assert_eq!(requirements.status_read_bytes_per_step, 1280);
    for overflowing in [
        FiniteCheckSpec {
            physical_dispatches: u64::MAX,
            ..native
        },
        FiniteCheckSpec {
            logical_submissions: u64::MAX,
            ..native
        },
        FiniteCheckSpec {
            pending_write_bytes: u64::MAX,
            ..native
        },
        FiniteCheckSpec {
            status_read_bytes: u64::MAX,
            ..native
        },
    ] {
        assert!(plan.finite_check_requirements(Some(overflowing)).is_err());
    }
    // Metadata-only real-model geometry; never prepare these substituted shapes.
    let mut actual = plan.clone();
    actual.layout = Layout::new(5120, 17408)?;
    assert_eq!(
        actual
            .finite_check_requirements(Some(native))?
            .max_view_elements,
        34_816
    );
    assert!(actual
        .finite_check_requirements(Some(FiniteCheckSpec {
            max_elements: 34_815,
            ..native
        }))
        .is_err());
    assert_eq!(plan.memory(), memory);
    assert_eq!(source.reads.load(Ordering::Relaxed), 0);
    let _resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    Ok(())
}

#[test]
fn checked_replays_match_independent_reference_with_five_checks_and_hidden_guards() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let oracle_source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let mut checker = device.prepare_finite_check()?;
    let mut resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    let hidden = device.upload_f32(&[-123.; 258])?;
    let view = hidden.view(1, 256)?;
    let mut stats = StreamStats::default();
    let tiles = checked_tiles(&plan, 3);
    for step in 0..2 {
        let input: Vec<f32> = (0..256)
            .map(|i| (i % 11) as f32 * 0.1 + if step == 0 { 0.25 } else { -0.75 })
            .collect();
        let expected = reference(&binding, &oracle_source, &input)?;
        device.write_f32(&view, &input)?;
        let mut calls = 0;
        resident.step_checked(
            &device,
            &source,
            &view,
            &mut checker,
            &mut stats,
            &mut || {
                calls += 1;
                Ok(())
            },
        )?;
        // CPU validator has three controls/check. Existing operations have seven
        // controls plus four per streamed tile, independent of checker storage.
        assert_eq!(calls, 7 + 4 * tiles + 3 * 5);
        close(&read(&device, &view)?, &expected);
        assert_eq!(resident.completed_steps(), step + 1);
        assert!(!resident.is_poisoned());
    }
    let all = read(&device, &hidden.full_view())?;
    assert_eq!((all[0], all[257]), (-123., -123.));
    assert_eq!(stats.completed_tiles, (2 * tiles) as u64);
    assert_eq!(source.reads.load(Ordering::Relaxed), 1 + 2 * tiles as u64);
    assert_eq!(stats.returned_read_bytes, stats.uploaded_bytes);
    Ok(())
}

#[test]
fn checked_entry_nonfinite_refusal_is_nonmutating_and_checker_remains_reusable() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let mut checker = device.prepare_finite_check()?;
    let mut resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    let before = read(&device, &resident.packet.full_view())?;
    let hidden = device.upload_f32(&[1.; 256])?;
    let mut stats = StreamStats::default();
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut input = [1.; 256];
        input[255] = invalid;
        device.write_f32(&hidden.full_view(), &input)?;
        let error = resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || Ok(()),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("entry hidden"));
        assert_eq!(resident.completed_steps(), 0);
        assert!(!resident.is_poisoned());
        assert_eq!(stats, StreamStats::default());
        assert_eq!(source.reads.load(Ordering::Relaxed), 1);
        assert_eq!(read(&device, &resident.packet.full_view())?, before);
        assert_eq!(
            read(&device, &hidden.full_view())?[255].to_bits(),
            invalid.to_bits()
        );
    }
    device.write_f32(&hidden.full_view(), &[1.; 256])?;
    resident.step_checked(
        &device,
        &source,
        &hidden.full_view(),
        &mut checker,
        &mut stats,
        &mut || Ok(()),
    )?;
    assert_eq!(resident.completed_steps(), 1);
    Ok(())
}

#[test]
fn checked_foreign_resources_alias_extent_and_counter_refuse_before_controls() -> Result<()> {
    let (_temporary, gguf) = fixture()?;
    let binding = Binding::load(&gguf)?;
    let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
    let source = Source::new(&gguf);
    let device = CpuExecutionDevice::new();
    let foreign_device = CpuExecutionDevice::new();
    let mut checker = device.prepare_finite_check()?;
    let mut foreign_checker = foreign_device.prepare_finite_check()?;
    let mut resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
    let hidden = device.upload_f32(&[1.; 256])?;
    let wrong_hidden = foreign_device.upload_f32(&[1.; 256])?;
    let before = read(&device, &resident.packet.full_view())?;
    let mut stats = StreamStats::default();
    let mut no_control = || -> Result<()> { panic!("preflight refusal reached control") };
    assert!(resident
        .step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut foreign_checker,
            &mut stats,
            &mut no_control
        )
        .is_err());
    assert!(resident
        .step_checked(
            &foreign_device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut no_control
        )
        .is_err());
    for wrong in [
        wrong_hidden.full_view(),
        hidden.view(0, 255)?,
        resident.packet.view(plan.layout.normalized, 256)?,
    ] {
        assert!(resident
            .step_checked(
                &device,
                &source,
                &wrong,
                &mut checker,
                &mut stats,
                &mut no_control
            )
            .is_err());
    }
    let (_other_temporary, other) = Fixture::hybrid().open();
    let foreign_source = Source::new(&other);
    assert!(plan
        .prepare_checked(&device, &foreign_source, admitted(&plan), &mut no_control)
        .is_err());
    assert!(resident
        .step_checked(
            &device,
            &foreign_source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut no_control
        )
        .is_err());
    resident.completed_steps = u64::MAX;
    assert!(resident
        .step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut no_control
        )
        .is_err());
    assert_eq!(resident.completed_steps(), u64::MAX);
    assert!(!resident.is_poisoned());
    assert_eq!(source.reads.load(Ordering::Relaxed), 1);
    assert_eq!(foreign_source.reads.load(Ordering::Relaxed), 0);
    assert_eq!(stats, StreamStats::default());
    assert_eq!(read(&device, &resident.packet.full_view())?, before);
    assert_eq!(read(&device, &hidden.full_view())?, vec![1.; 256]);
    Ok(())
}

#[test]
fn checked_raw_gate_and_up_nonfinite_are_refused_before_swiglu_then_reset_retries() -> Result<()> {
    // Corrupt actual encoded scale bytes, not a fake execution result. Literal
    // unclipped CPU SwiGLU does not mask these into finite values; this test proves
    // refusal at the raw boundary before it could transform the bad projections.
    for role in ["gate", "up"] {
        for bits in [0x7c00, 0xfc00, 0x7e01] {
            let (_temporary, gguf) = fixture()?;
            checked_patch_scale(&gguf, role, bits)?;
            let binding = Binding::load(&gguf)?;
            let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
            let source = Source::new(&gguf);
            let device = CpuExecutionDevice::new();
            let mut checker = device.prepare_finite_check()?;
            let mut resident =
                plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
            let hidden = device.upload_f32(&[1.; 256])?;
            let mut stats = StreamStats::default();
            let error = resident
                .step_checked(
                    &device,
                    &source,
                    &hidden.full_view(),
                    &mut checker,
                    &mut stats,
                    &mut || Ok(()),
                )
                .unwrap_err();
            assert!(
                format!("{error:#}").contains("raw gate/up"),
                "{role} {bits:x}: {error:#}"
            );
            let offset = if role == "gate" {
                plan.layout.gate
            } else {
                plan.layout.up
            };
            let bad = read(&device, &resident.packet.view(offset, 1)?)?[0];
            if bits == 0x7e01 {
                assert!(bad.is_nan());
            } else {
                assert!(bad.is_infinite());
                assert_eq!(bad.is_sign_negative(), bits == 0xfc00);
            }
            assert_eq!(stats.completed_tiles, checked_tiles(&plan, 2) as u64);
            assert_eq!(
                source.reads.load(Ordering::Relaxed),
                1 + checked_tiles(&plan, 2) as u64
            );
            assert_eq!(resident.completed_steps(), 0);
            assert!(resident.is_poisoned());
            assert_eq!(read(&device, &hidden.full_view())?, vec![1.; 256]);
            assert!(resident
                .step_checked(
                    &device,
                    &source,
                    &hidden.full_view(),
                    &mut checker,
                    &mut stats,
                    &mut || panic!("poisoned component continued")
                )
                .is_err());
            // Arithmetic refusal poisons the component, not a healthy checker.
            device.check_finite(&mut checker, &hidden.full_view(), &mut || Ok(()))?;
            checked_patch_scale(&gguf, role, 0x1400)?;
            resident.reset(&device, &mut || Ok(()))?;
            resident.step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || Ok(()),
            )?;
            assert_eq!(resident.completed_steps(), 1);
            assert!(!resident.is_poisoned());
        }
    }
    Ok(())
}

#[test]
fn checked_real_overflow_refuses_normalized_activated_and_final_residual_phases() -> Result<()> {
    for phase in ["normalized hidden", "activated gate", "residual hidden"] {
        let (_temporary, gguf) = fixture()?;
        let mut input = [1.; 256];
        match phase {
            "normalized hidden" => {
                checked_patch_norm(&gguf, f32::MAX)?;
                input.fill(0.);
                input[0] = 1.; // RMS normalization scales it by about sixteen
            }
            "activated gate" => checked_patch_norm(&gguf, 1e20)?,
            "residual hidden" => {
                checked_patch_norm(&gguf, 1e18)?;
                // Finite activated values, followed by the largest finite half
                // scale in the first down row, overflow during the residual path.
                checked_patch_scale(&gguf, "down", 0x7bff)?;
            }
            _ => unreachable!(),
        }
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&input)?;
        let mut stats = StreamStats::default();
        let error = resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || Ok(()),
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains(phase), "{phase}: {error:#}");
        let completed = match phase {
            "normalized hidden" => 0,
            "activated gate" => checked_tiles(&plan, 2),
            "residual hidden" => checked_tiles(&plan, 3),
            _ => unreachable!(),
        };
        assert_eq!(stats.completed_tiles, completed as u64);
        assert!(resident.is_poisoned());
        assert_eq!(resident.completed_steps(), 0);
        if phase == "residual hidden" {
            assert!(read(&device, &hidden.full_view())?
                .iter()
                .any(|v| !v.is_finite()));
        } else {
            assert_eq!(read(&device, &hidden.full_view())?, input);
        }
    }
    Ok(())
}

#[test]
fn checked_cancellation_at_every_phase_and_final_commit_preserves_poison_and_counter() -> Result<()>
{
    // Five CPU-check completion controls plus final publication control. Later
    // values deliberately occur after real private/residual mutation has happened.
    for cancel_at in [3, 10, 77, 82, 117, 118] {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        assert_eq!(checked_tiles(&plan, 3), 24);
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[1.; 256])?;
        let mut stats = StreamStats::default();
        resident.step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(resident.completed_steps(), 1);
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        let norm_before = read(&device, &resident.packet.view(0, plan.dim())?)?;
        let mut calls = 0;
        assert!(resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || {
                    calls += 1;
                    ensure!(calls != cancel_at, "cancel checked phase {cancel_at}");
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(calls, cancel_at);
        assert_eq!(resident.completed_steps(), 1);
        assert_eq!(resident.is_poisoned(), cancel_at != 3);
        if cancel_at == 3 {
            assert_eq!(read(&device, &hidden.full_view())?, vec![1.; 256]);
            resident.step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || Ok(()),
            )?;
            assert_eq!(resident.completed_steps(), 2);
            continue;
        }
        let changed_hidden = read(&device, &hidden.full_view())?;
        assert!(resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || panic!("poisoned step reached control")
            )
            .is_err());
        assert!(resident
            .reset(&device, &mut || anyhow::bail!("cancel reset"))
            .is_err());
        assert!(resident.is_poisoned());
        assert_eq!(resident.completed_steps(), 1);
        resident.reset(&device, &mut || Ok(()))?;
        assert!(!resident.is_poisoned());
        assert_eq!(resident.completed_steps(), 0);
        assert_eq!(
            read(&device, &resident.packet.view(0, plan.dim())?)?,
            norm_before
        );
        assert!(read(
            &device,
            &resident.packet.view(
                plan.layout.normalized,
                plan.layout.len - plan.layout.normalized
            )?
        )?
        .iter()
        .all(|v| v.to_bits() == 0));
        assert_eq!(read(&device, &hidden.full_view())?, changed_hidden);
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        resident.step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(resident.completed_steps(), 1);
    }
    Ok(())
}
