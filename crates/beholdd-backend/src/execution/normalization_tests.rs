use super::*;

fn read(device: &dyn ExecutionDevice, view: &F32TensorView) -> Vec<f32> {
    let mut values = vec![0.0; view.len()];
    device.read_f32(view, &mut values).unwrap();
    values
}

fn execute(device: &dyn ExecutionDevice, batch: &RecordedBatch) {
    let submission = device.submit(batch).unwrap();
    device.wait(&submission).unwrap();
}

fn assert_bits(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
    }
}

#[test]
fn recorded_rmsnorm_matches_cpu_with_weight_and_activation_aliases() {
    let initial = [9.0, 1.5, -2.0, -0.0, 0.25, 2.0, -0.5, 7.0, 8.0, 11.0];
    // (output start, input start, optional weight start). Both exact and partial
    // aliases are included; comparing the entire arena checks untouched surrounds.
    for (output_start, input_start, weight_start) in [
        (5, 1, None),
        (1, 1, None),
        (2, 1, None),
        (5, 1, Some(1)),
        (1, 4, Some(2)),
        (2, 1, Some(3)),
        (1, 1, Some(1)),
    ] {
        for eps in [0.0, -0.0, 1e-5] {
            let device = CpuExecutionDevice::new();
            let arena = device.upload_f32(&initial).unwrap();
            let output = arena.view(output_start, 4).unwrap();
            let input = arena.view(input_start, 4).unwrap();
            let weight = weight_start.map(|start| arena.view(start, 4).unwrap());
            let mut normalized = [0.0; 4];
            Cpu::new()
                .rmsnorm(
                    &mut normalized,
                    &initial[input_start..input_start + 4],
                    weight_start.map(|start| &initial[start..start + 4]),
                    eps,
                )
                .unwrap();
            let mut expected = initial;
            expected[output_start..output_start + 4].copy_from_slice(&normalized);
            let mut commands = CommandList::new();
            commands
                .rmsnorm(&output, &input, weight.as_ref(), eps)
                .unwrap();
            execute(&device, &device.record(&commands).unwrap());
            assert_bits(&read(&device, &arena.full_view()), &expected);
        }
    }
}

#[test]
fn recorded_rope_matches_cpu_for_both_layouts_directions_and_coefficient_aliases() {
    let initial = [0.25, -0.5, 0.75, -1.0, -0.0, 1.0, 0.5, -0.75, 2.0, 3.0];
    // Coefficients deliberately need not be a unit-circle table: they are exact
    // resident operands. Some views overlap x, each other, or both.
    for (x_start, cos_start, sin_start, pairs) in [
        (0, 4, 6, 2),
        (0, 0, 2, 2),
        (2, 0, 2, 2),
        (2, 3, 4, 2),
        (0, 4, 4, 2),
        (0, 4, 7, 3),
    ] {
        for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
            for inverse in [false, true] {
                let device = CpuExecutionDevice::new();
                let arena = device.upload_f32(&initial).unwrap();
                let x = arena.view(x_start, pairs * 2).unwrap();
                let cos = arena.view(cos_start, pairs).unwrap();
                let sin = arena.view(sin_start, pairs).unwrap();
                let mut rotated = initial[x_start..x_start + pairs * 2].to_vec();
                Cpu::new()
                    .rope(
                        &mut rotated,
                        &initial[cos_start..cos_start + pairs],
                        &initial[sin_start..sin_start + pairs],
                        layout,
                        inverse,
                    )
                    .unwrap();
                let mut expected = initial;
                expected[x_start..x_start + pairs * 2].copy_from_slice(&rotated);
                let mut commands = CommandList::new();
                commands.rope(&x, &cos, &sin, layout, inverse).unwrap();
                execute(&device, &device.record(&commands).unwrap());
                assert_bits(&read(&device, &arena.full_view()), &expected);
            }
        }
    }
}

#[test]
fn normalization_command_validation_is_eager_and_transactional() {
    let device = CpuExecutionDevice::new();
    let four = device.upload_f32(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let two = four.view(0, 2).unwrap();
    let one = four.view(0, 1).unwrap();
    let empty = four.view(0, 0).unwrap();
    let mut commands = CommandList::new();
    // Check an already-owned, nonempty list as well as a fresh list. Refusal must
    // preserve owner metadata and every previously accepted command.
    for populated in [false, true] {
        if populated {
            commands.rmsnorm(&two, &two, None, 1e-5).unwrap();
        }
        let owner = commands.owner;
        let count = commands.len();
        for (output, input, weight) in [
            (&empty, &empty, None),
            (&one, &two, None),
            (&two, &one, None),
            (&two, &two, Some(&one)),
        ] {
            assert!(matches!(
                commands.rmsnorm(output, input, weight, 1e-5),
                Err(ExecutionError::Shape(_))
            ));
            assert_eq!(commands.owner, owner);
            assert_eq!(commands.len(), count);
        }
        for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
            assert!(matches!(
                commands.rmsnorm(&two, &two, None, eps),
                Err(ExecutionError::Shape(_))
            ));
            assert_eq!(commands.owner, owner);
            assert_eq!(commands.len(), count);
        }
        for (x, cos, sin) in [(&two, &two, &two), (&two, &one, &two), (&one, &one, &one)] {
            assert!(matches!(
                commands.rope(x, cos, sin, RopeLayout::SplitHalf, false),
                Err(ExecutionError::Shape(_))
            ));
            assert_eq!(commands.owner, owner);
            assert_eq!(commands.len(), count);
        }
        // Forge only metadata inside this private test: exercise the multiplication
        // overflow refusal without allocating a gigantic tensor or accessing it.
        let mut huge = one.clone();
        huge.len = usize::MAX;
        assert!(matches!(
            commands.rope(&two, &huge, &huge, RopeLayout::Interleaved, false),
            Err(ExecutionError::Shape(_))
        ));
        assert_eq!(commands.owner, owner);
        assert_eq!(commands.len(), count);
    }
    assert_bits(&read(&device, &four.full_view()), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn every_normalization_operand_is_owner_checked_without_claiming_a_failed_list() {
    let first = CpuExecutionDevice::new();
    let second = CpuExecutionDevice::new();
    let a = first.upload_f32(&[1.0; 4]).unwrap();
    let b = second.upload_f32(&[2.0; 4]).unwrap();
    for foreign_slot in 0..3 {
        let norm = std::array::from_fn::<_, 3, _>(|slot| {
            if slot == foreign_slot {
                b.full_view()
            } else {
                a.full_view()
            }
        });
        let mut commands = CommandList::new();
        assert!(matches!(
            commands.rmsnorm(&norm[0], &norm[1], Some(&norm[2]), 1e-5),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert!(commands.is_empty());
        assert_eq!(commands.owner, None);
        // A subsequent valid command from either resource domain may own the list.
        commands
            .rmsnorm(&b.full_view(), &b.full_view(), None, 1e-5)
            .unwrap();
        let owner = commands.owner;
        assert!(matches!(
            commands.rmsnorm(&a.full_view(), &a.full_view(), None, 1e-5),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert_eq!(commands.len(), 1);
        assert_eq!(commands.owner, owner);

        let rope = std::array::from_fn::<_, 3, _>(|slot| {
            let tensor = if slot == foreign_slot { &b } else { &a };
            tensor.view(0, if slot == 0 { 4 } else { 2 }).unwrap()
        });
        let mut commands = CommandList::new();
        assert!(matches!(
            commands.rope(&rope[0], &rope[1], &rope[2], RopeLayout::Interleaved, false),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert!(commands.is_empty());
        assert_eq!(commands.owner, None);
        commands
            .rope(
                &b.full_view(),
                &b.view(0, 2).unwrap(),
                &b.view(2, 2).unwrap(),
                RopeLayout::Interleaved,
                false,
            )
            .unwrap();
        let owner = commands.owner;
        assert!(matches!(
            commands.rope(
                &a.full_view(),
                &a.view(0, 2).unwrap(),
                &a.view(2, 2).unwrap(),
                RopeLayout::Interleaved,
                false
            ),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert_eq!(commands.len(), 1);
        assert_eq!(commands.owner, owner);
    }
}

#[test]
fn composed_normalization_reads_updated_resident_operands_on_each_replay() {
    let device = CpuExecutionDevice::new();
    let input = device.allocate_f32(4).unwrap();
    let weight = device.allocate_f32(4).unwrap();
    let output = device.allocate_f32(4).unwrap();
    let raw_cos = device.allocate_f32(2).unwrap();
    let cos = device.allocate_f32(2).unwrap();
    let sin = device.allocate_f32(2).unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(
            &output.full_view(),
            &input.full_view(),
            Some(&weight.full_view()),
            1e-5,
        )
        .unwrap();
    commands
        .rmsnorm(&cos.full_view(), &raw_cos.full_view(), None, 0.25)
        .unwrap();
    commands
        .rope(
            &output.full_view(),
            &cos.full_view(),
            &sin.full_view(),
            RopeLayout::SplitHalf,
            true,
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    assert_eq!(batch.len(), 3);
    drop(commands);
    let cpu = Cpu::new();
    for (input_values, weight_values, cos_values, sin_values) in [
        (
            [1.0, -2.0, 0.5, 0.0],
            [0.5, 1.0, 2.0, -1.0],
            [0.25, 0.75],
            [-0.5, 0.5],
        ),
        (
            [-3.0, 0.25, 0.0, 2.0],
            [2.0, -0.5, 1.0, 0.0],
            [-0.5, 0.125],
            [1.0, -0.25],
        ),
    ] {
        device.write_f32(&input.full_view(), &input_values).unwrap();
        device
            .write_f32(&weight.full_view(), &weight_values)
            .unwrap();
        device.write_f32(&raw_cos.full_view(), &cos_values).unwrap();
        device.write_f32(&sin.full_view(), &sin_values).unwrap();
        let mut expected = [0.0; 4];
        let mut normalized_cos = [0.0; 2];
        cpu.rmsnorm(&mut expected, &input_values, Some(&weight_values), 1e-5)
            .unwrap();
        cpu.rmsnorm(&mut normalized_cos, &cos_values, None, 0.25)
            .unwrap();
        cpu.rope(
            &mut expected,
            &normalized_cos,
            &sin_values,
            RopeLayout::SplitHalf,
            true,
        )
        .unwrap();
        execute(&device, &batch);
        assert_bits(&read(&device, &output.full_view()), &expected);
        assert_bits(&read(&device, &cos.full_view()), &normalized_cos);
        assert_bits(&read(&device, &input.full_view()), &input_values);
        assert_bits(&read(&device, &weight.full_view()), &weight_values);
        assert_bits(&read(&device, &sin.full_view()), &sin_values);
    }
}

#[test]
fn empty_rope_is_a_valid_recorded_noop() {
    let device = CpuExecutionDevice::new();
    let empty = device.allocate_f32(0).unwrap();
    for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
        for inverse in [false, true] {
            let mut commands = CommandList::new();
            commands
                .rope(
                    &empty.full_view(),
                    &empty.full_view(),
                    &empty.full_view(),
                    layout,
                    inverse,
                )
                .unwrap();
            execute(&device, &device.record(&commands).unwrap());
            assert!(read(&device, &empty.full_view()).is_empty());
        }
    }
}
