use super::*;

fn weight(
    device: &CpuExecutionDevice,
    kind: GgmlType,
    rows: usize,
    cols: usize,
    bytes: &[u8],
) -> PreparedWeight {
    let encoded = device
        .upload_weight(&Weight {
            bytes,
            kind,
            rows,
            cols,
        })
        .unwrap();
    device.prepare_weight(&encoded).unwrap()
}

fn read(device: &CpuExecutionDevice, tensor: &F32Tensor) -> Vec<f32> {
    let mut values = vec![f32::NAN; tensor.len()];
    device.read_f32(&tensor.full_view(), &mut values).unwrap();
    values
}

#[test]
fn selected_owned_rows_preserve_guards_and_replay_after_sources_drop() {
    for (kind, cols, bytes, expected) in [
        (
            GgmlType(0),
            3,
            (0..9)
                .flat_map(|n| (n as f32 - 4.0).to_le_bytes())
                .collect::<Vec<_>>(),
            vec![-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        ),
        (
            GgmlType(30),
            3,
            [
                0x3f80u16, 0xc000, 0x4040, 0x4080, 0xc0a0, 0x40c0, 0x40e0, 0xc100, 0x4110,
            ]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect(),
            vec![1.0, -2.0, 3.0, 4.0, -5.0, 6.0, 7.0, -8.0, 9.0],
        ),
        (
            GgmlType(8),
            32,
            (0..3)
                .flat_map(|row| {
                    let mut block = 0x3800u16.to_le_bytes().to_vec();
                    block.extend((0..32).map(|column| (row * 7 + column - 20) as i8 as u8));
                    block
                })
                .collect(),
            (0..3)
                .flat_map(|row| (0..32).map(move |column| (row * 7 + column - 20) as f32 * 0.5))
                .collect(),
        ),
    ] {
        let device = CpuExecutionDevice::new();
        let prepared = weight(&device, kind, 3, cols, &bytes);
        let weak = Arc::downgrade(&prepared.resource);
        let output = device.allocate_f32(cols + 2).unwrap();
        let mut batches = Vec::new();
        for row in [0, 2] {
            let mut commands = CommandList::new();
            commands
                .embedding_row(&output.view(1, cols).unwrap(), &prepared, row)
                .unwrap();
            batches.push(device.record(&commands).unwrap());
            // Later list changes must not alter the recorded selected row.
            commands
                .embedding_row(&output.view(1, cols).unwrap(), &prepared, 1)
                .unwrap();
        }
        drop((prepared, bytes));
        assert!(weak.upgrade().is_some());
        for repeat in 0..2 {
            for (batch, row) in batches.iter().zip([0, 2]) {
                let guard = 123.0 + repeat as f32;
                device
                    .write_f32(&output.full_view(), &vec![guard; cols + 2])
                    .unwrap();
                let submission = device.submit(batch).unwrap();
                device.wait(&submission).unwrap();
                let actual = read(&device, &output);
                assert_eq!(actual[0], guard);
                assert_eq!(actual[cols + 1], guard);
                assert_eq!(
                    &actual[1..cols + 1],
                    &expected[row * cols..(row + 1) * cols]
                );
            }
        }
        drop(batches);
        assert!(weak.upgrade().is_none());
    }
}

#[test]
fn embedding_output_is_observed_by_later_commands() {
    let device = CpuExecutionDevice::new();
    let prepared = weight(
        &device,
        GgmlType(0),
        2,
        2,
        &[1.0f32, 2.0, 3.0, 4.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let output = device.allocate_f32(2).unwrap();
    let normalized = device.allocate_f32(2).unwrap();
    let mut commands = CommandList::new();
    commands
        .embedding_row(&output.full_view(), &prepared, 0)
        .unwrap();
    commands
        .embedding_row(&output.full_view(), &prepared, 1)
        .unwrap();
    commands
        .rmsnorm(&normalized.full_view(), &output.full_view(), None, 1e-6)
        .unwrap();
    let batch = device.record(&commands).unwrap();
    let submission = device.submit(&batch).unwrap();
    device.wait(&submission).unwrap();
    let mut expected = [0.0; 2];
    Cpu::new()
        .rmsnorm(&mut expected, &[3.0, 4.0], None, 1e-6)
        .unwrap();
    assert_eq!(read(&device, &normalized), expected);
}

#[test]
fn invalid_embedding_appends_preserve_owner_and_record_refuses_forged_storage() {
    let device = CpuExecutionDevice::new();
    let foreign = CpuExecutionDevice::new();
    let bytes = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let prepared = weight(&device, GgmlType(0), 2, 2, &bytes);
    let output = device.upload_f32(&[99.0; 2]).unwrap();
    let mut commands = CommandList::new();
    for row in [2, usize::MAX] {
        assert!(commands
            .embedding_row(&output.full_view(), &prepared, row)
            .is_err());
    }
    assert!(commands
        .embedding_row(&output.view(0, 1).unwrap(), &prepared, 0)
        .is_err());
    assert!(commands
        .embedding_row(&foreign.allocate_f32(2).unwrap().full_view(), &prepared, 0)
        .is_err());
    assert!(commands.is_empty());
    assert!(commands.owner().is_none());
    commands
        .embedding_row(&output.full_view(), &prepared, 0)
        .unwrap();
    let foreign_weight = weight(&foreign, GgmlType(0), 2, 2, &bytes);
    assert!(commands
        .embedding_row(&output.full_view(), &foreign_weight, 0)
        .is_err());
    assert_eq!(commands.len(), 1);
    assert_eq!(commands.owner(), Some(device.owner));

    let forged_weights = [
        PreparedWeight::new(device.owner, GgmlType(0), 2, 2, 16, Arc::new(())),
        PreparedWeight::new(
            device.owner,
            GgmlType(0),
            2,
            2,
            12,
            Arc::clone(&prepared.resource),
        ),
        PreparedWeight::new(
            device.owner,
            GgmlType(30),
            2,
            2,
            16,
            Arc::clone(&prepared.resource),
        ),
        PreparedWeight::new(
            device.owner,
            GgmlType(0),
            3,
            2,
            16,
            Arc::clone(&prepared.resource),
        ),
        PreparedWeight::new(
            device.owner,
            GgmlType(0),
            2,
            2,
            16,
            Arc::new(CpuPreparedWeight {
                bytes: Arc::new(vec![0; 12]),
                kind: GgmlType(0),
                rows: 2,
                cols: 2,
            }),
        ),
    ];
    for forged in forged_weights {
        let mut list = commands.clone();
        list.embedding_row(&output.full_view(), &forged, 1).unwrap();
        assert!(device.record(&list).is_err());
        assert_eq!(read(&device, &output), [99.0; 2]);
    }
    for tensor in [
        F32Tensor::new(device.owner, 2, Arc::new(())),
        F32Tensor::new(device.owner, 4, Arc::clone(&output.resource)),
    ] {
        let mut list = commands.clone();
        list.embedding_row(&tensor.view(0, 2).unwrap(), &prepared, 0)
            .unwrap();
        assert!(device.record(&list).is_err());
    }
    let invalid_row = Command::EmbeddingRow {
        output: output.full_view(),
        weight: prepared,
        row: usize::MAX,
    };
    assert!(device.validate_command(&invalid_row).is_err());
}
