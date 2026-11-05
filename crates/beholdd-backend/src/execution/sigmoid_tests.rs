use super::*;

fn read(device: &CpuExecutionDevice, view: &F32TensorView) -> Vec<f32> {
    let mut values = vec![0.0; view.len()];
    device.read_f32(view, &mut values).unwrap();
    values
}

fn run(device: &CpuExecutionDevice, batch: &RecordedBatch) {
    device.wait(&device.submit(batch).unwrap()).unwrap();
}

#[test]
fn sigmoid_snapshots_overlapping_views_and_replays_current_inputs() {
    let device = CpuExecutionDevice::new();
    let tensor = device.upload_f32(&[7.0, 0.0, 2.0, -2.0, -9.0]).unwrap();
    let mut commands = CommandList::new();
    commands
        .sigmoid_gate(
            &tensor.view(2, 3).unwrap(),
            &tensor.view(1, 3).unwrap(),
            &tensor.view(0, 3).unwrap(),
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    run(&device, &batch);
    let first = read(&device, &tensor.full_view());
    assert_eq!(&first[..4], &[7.0, 0.0, 0.0, 1.0]);
    assert!((first[4] + 1.7615942).abs() < 2e-7);
    device
        .write_f32(&tensor.full_view(), &[8.0, 0.0, 4.0, -4.0, -9.0])
        .unwrap();
    run(&device, &batch);
    let second = read(&device, &tensor.full_view());
    assert_eq!(&second[..4], &[8.0, 0.0, 0.0, 2.0]);
    assert!((second[4] + 3.928055).abs() < 5e-7);
}

#[test]
fn sigmoid_f32_exp_boundary_survives_amplification() {
    let device = CpuExecutionDevice::new();
    let large = f32::from_bits((127 + 100) << 23);
    let input = device.upload_f32(&[large; 4]).unwrap();
    let gate = device
        .upload_f32(&[-103.0, -105.0, f32::MAX, -f32::MAX])
        .unwrap();
    let result = device.upload_f32(&[-91.0; 6]).unwrap();
    let mut commands = CommandList::new();
    commands
        .sigmoid_gate(
            &result.view(1, 4).unwrap(),
            &input.full_view(),
            &gate.full_view(),
        )
        .unwrap();
    run(&device, &device.record(&commands).unwrap());
    assert_eq!(
        read(&device, &result.full_view()),
        vec![
            -91.0,
            f32::from_bits((127 - 49) << 23),
            0.0,
            large,
            0.0,
            -91.0
        ]
    );
}

#[test]
fn sigmoid_refusal_does_not_bind_command_owner_and_empty_is_retained() {
    let first = CpuExecutionDevice::new();
    let second = CpuExecutionDevice::new();
    let a = first.allocate_f32(2).unwrap();
    let b = second.allocate_f32(2).unwrap();
    let mut commands = CommandList::new();
    assert!(matches!(
        commands.sigmoid_gate(&a.full_view(), &a.view(0, 1).unwrap(), &a.full_view()),
        Err(ExecutionError::Shape(_))
    ));
    assert!(matches!(
        commands.sigmoid_gate(&a.full_view(), &b.full_view(), &a.full_view()),
        Err(ExecutionError::ForeignResource { .. })
    ));
    assert!(commands.is_empty());
    let empty = b.view(1, 0).unwrap();
    commands.sigmoid_gate(&empty, &empty, &empty).unwrap();
    let batch = second.record(&commands).unwrap();
    assert_eq!(batch.len(), 1);
    run(&second, &batch);
    assert_eq!(read(&second, &b.full_view()), vec![0.0; 2]);
}
