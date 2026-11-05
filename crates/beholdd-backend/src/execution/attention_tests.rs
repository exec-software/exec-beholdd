use super::*;
use crate::AttentionConfig;

fn read(device: &dyn ExecutionDevice, view: &F32TensorView) -> Vec<f32> {
    let mut values = vec![f32::NAN; view.len()];
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

fn config() -> IncrementalAttentionConfig {
    IncrementalAttentionConfig {
        n_query_positions: 1,
        query_start: 1,
        n_kv_positions: 2,
        n_heads: 1,
        n_kv_heads: 1,
        head_dim: 2,
        scale: 1.0,
    }
}

fn reference(q: &[f32], k: &[f32], v: &[f32], config: IncrementalAttentionConfig) -> Vec<f32> {
    let mut output = vec![f32::NAN; q.len()];
    Cpu::new()
        .incremental_causal_attention(&mut output, q, k, v, config)
        .unwrap();
    output
}

#[test]
fn recorded_prefill_and_cached_suffix_match_full_cpu_and_retain_inputs() {
    let device = CpuExecutionDevice::new();
    let cfg = IncrementalAttentionConfig {
        n_query_positions: 3,
        query_start: 0,
        n_kv_positions: 3,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 2,
        scale: 0.5,
    };
    let queries: Vec<_> = (0..24).map(|i| (i as f32 - 11.0) / 8.0).collect();
    let keys: Vec<_> = (0..12).map(|i| ((i * 3 % 11) as f32 - 5.0) / 4.0).collect();
    let values: Vec<_> = (0..12).map(|i| ((i * 7 % 13) as f32 - 6.0) / 2.0).collect();
    let mut expected = vec![f32::NAN; queries.len()];
    Cpu::new()
        .causal_attention(
            &mut expected,
            &queries,
            &keys,
            &values,
            AttentionConfig {
                n_positions: 3,
                n_heads: 4,
                n_kv_heads: 2,
                head_dim: 2,
                scale: 0.5,
            },
        )
        .unwrap();
    let q = device.upload_f32(&queries).unwrap();
    let k = device.upload_f32(&keys).unwrap();
    let v = device.upload_f32(&values).unwrap();
    let full_output = device.allocate_f32(24).unwrap();
    let split_output = device.allocate_f32(24).unwrap();
    let weak = [
        Arc::downgrade(&q.resource),
        Arc::downgrade(&k.resource),
        Arc::downgrade(&v.resource),
    ];
    let mut commands = CommandList::new();
    commands
        .incremental_causal_attention(
            &full_output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            cfg,
        )
        .unwrap();
    commands
        .incremental_causal_attention(
            &split_output.view(0, 16).unwrap(),
            &q.view(0, 16).unwrap(),
            &k.view(0, 8).unwrap(),
            &v.view(0, 8).unwrap(),
            IncrementalAttentionConfig {
                n_query_positions: 2,
                n_kv_positions: 2,
                ..cfg
            },
        )
        .unwrap();
    commands
        .incremental_causal_attention(
            &split_output.view(16, 8).unwrap(),
            &q.view(16, 8).unwrap(),
            &k.full_view(),
            &v.full_view(),
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: 2,
                ..cfg
            },
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    assert_eq!(batch.len(), 3);
    drop((commands, q, k, v));
    assert!(weak.iter().all(|resource| resource.upgrade().is_some()));
    execute(&device, &batch);
    assert_bits(&read(&device, &full_output.full_view()), &expected);
    assert_bits(&read(&device, &split_output.full_view()), &expected);
    drop(batch);
    assert!(weak.iter().all(|resource| resource.upgrade().is_none()));
}

#[test]
fn causal_mask_ignores_future_nans_and_maps_contiguous_gqa_heads() {
    let device = CpuExecutionDevice::new();
    let cfg = IncrementalAttentionConfig {
        n_query_positions: 2,
        query_start: 0,
        n_kv_positions: 3,
        n_heads: 4,
        n_kv_heads: 2,
        head_dim: 2,
        scale: 1.0,
    };
    let q = device.upload_f32(&[0.0; 16]).unwrap();
    let k = device
        .upload_f32(&[
            1.0,
            2.0,
            3.0,
            4.0,
            -1.0,
            -2.0,
            -3.0,
            -4.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ])
        .unwrap();
    let v = device
        .upload_f32(&[
            1.0,
            2.0,
            10.0,
            20.0,
            3.0,
            4.0,
            30.0,
            40.0,
            f32::NAN,
            f32::NAN,
            f32::NAN,
            f32::NAN,
        ])
        .unwrap();
    let output = device.allocate_f32(16).unwrap();
    let mut commands = CommandList::new();
    commands
        .incremental_causal_attention(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            cfg,
        )
        .unwrap();
    execute(&device, &device.record(&commands).unwrap());
    assert_bits(
        &read(&device, &output.full_view()),
        &[
            1.0, 2.0, 1.0, 2.0, 10.0, 20.0, 10.0, 20.0, 2.0, 3.0, 2.0, 3.0, 20.0, 30.0, 20.0, 30.0,
        ],
    );
}

fn identity(device: &dyn ExecutionDevice, width: usize) -> PreparedWeight {
    let bytes: Vec<_> = (0..width * width)
        .flat_map(|index| {
            let value = if index / width == index % width {
                1.0_f32
            } else {
                0.0
            };
            value.to_le_bytes()
        })
        .collect();
    device
        .prepare_weight(
            &device
                .upload_weight(&Weight {
                    bytes: &bytes,
                    kind: GgmlType(0),
                    rows: width,
                    cols: width,
                })
                .unwrap(),
        )
        .unwrap()
}

#[test]
fn reused_attention_observes_kv_written_by_earlier_commands_and_new_host_inputs() {
    let device = CpuExecutionDevice::new();
    let q = device.upload_f32(&[1.0, 0.0]).unwrap();
    let k_source = device.allocate_f32(4).unwrap();
    let v_source = device.allocate_f32(4).unwrap();
    let k = device.allocate_f32(4).unwrap();
    let v = device.allocate_f32(4).unwrap();
    let output = device.allocate_f32(2).unwrap();
    let copy = identity(&device, 4);
    let mut commands = CommandList::new();
    commands
        .linear(
            &k.full_view(),
            &k_source.full_view(),
            &copy,
            LinearMode::Overwrite,
        )
        .unwrap();
    commands
        .linear(
            &v.full_view(),
            &v_source.full_view(),
            &copy,
            LinearMode::Overwrite,
        )
        .unwrap();
    commands
        .incremental_causal_attention(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            config(),
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    drop((commands, copy, k, v));
    let mut previous = None;
    for (keys, values) in [
        ([1.0, 0.0, 0.0, 1.0], [10.0, 20.0, 30.0, 40.0]),
        ([-1.0, 0.0, 0.0, 1.0], [10.0, -10.0, 2.0, 4.0]),
    ] {
        device.write_f32(&k_source.full_view(), &keys).unwrap();
        device.write_f32(&v_source.full_view(), &values).unwrap();
        let expected = reference(&[1.0, 0.0], &keys, &values, config());
        execute(&device, &batch);
        let actual = read(&device, &output.full_view());
        assert_bits(&actual, &expected);
        if let Some(previous) = previous {
            assert_ne!(actual, previous);
        }
        previous = Some(actual);
    }
}

#[test]
fn attention_output_aliases_snapshot_all_three_inputs_before_writing() {
    let device = CpuExecutionDevice::new();
    let original = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let expected = reference(&original[..2], &original[1..5], &original[2..], config());
    for offset in [0, 1, 2, 4] {
        let arena = device.upload_f32(&original).unwrap();
        let mut commands = CommandList::new();
        commands
            .incremental_causal_attention(
                &arena.view(offset, 2).unwrap(),
                &arena.view(0, 2).unwrap(),
                &arena.view(1, 4).unwrap(),
                &arena.view(2, 4).unwrap(),
                config(),
            )
            .unwrap();
        execute(&device, &device.record(&commands).unwrap());
        let mut complete = original;
        complete[offset..offset + 2].copy_from_slice(&expected);
        assert_bits(&read(&device, &arena.full_view()), &complete);
    }
}

#[test]
fn invalid_attention_append_preserves_owner_commands_and_storage() {
    let device = CpuExecutionDevice::new();
    let q = device.upload_f32(&[1.0, 0.0]).unwrap();
    let cache = device.upload_f32(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    let output = device.upload_f32(&[91.0, 92.0]).unwrap();
    let valid = config();
    let invalid = [
        IncrementalAttentionConfig {
            n_query_positions: 0,
            ..valid
        },
        IncrementalAttentionConfig {
            n_kv_positions: 0,
            ..valid
        },
        IncrementalAttentionConfig {
            n_heads: 0,
            ..valid
        },
        IncrementalAttentionConfig {
            n_kv_heads: 0,
            ..valid
        },
        IncrementalAttentionConfig {
            head_dim: 0,
            ..valid
        },
        IncrementalAttentionConfig {
            n_heads: 3,
            n_kv_heads: 2,
            ..valid
        },
        IncrementalAttentionConfig {
            scale: 0.0,
            ..valid
        },
        IncrementalAttentionConfig {
            scale: -0.0,
            ..valid
        },
        IncrementalAttentionConfig {
            scale: -1.0,
            ..valid
        },
        IncrementalAttentionConfig {
            scale: f32::NAN,
            ..valid
        },
        IncrementalAttentionConfig {
            scale: f32::INFINITY,
            ..valid
        },
        IncrementalAttentionConfig {
            query_start: 2,
            ..valid
        },
        IncrementalAttentionConfig {
            query_start: usize::MAX,
            ..valid
        },
        IncrementalAttentionConfig {
            n_heads: usize::MAX,
            ..valid
        },
        IncrementalAttentionConfig {
            n_kv_positions: usize::MAX,
            ..valid
        },
        IncrementalAttentionConfig {
            n_query_positions: usize::MAX,
            query_start: 0,
            n_kv_positions: usize::MAX,
            head_dim: 1,
            ..valid
        },
    ];
    for already_owned in [false, true] {
        let mut commands = CommandList::new();
        if already_owned {
            commands
                .incremental_causal_attention(
                    &output.full_view(),
                    &q.full_view(),
                    &cache.full_view(),
                    &cache.full_view(),
                    valid,
                )
                .unwrap();
        }
        let state = (commands.owner, commands.len(), commands.commands.capacity());
        for cfg in invalid {
            assert!(commands
                .incremental_causal_attention(
                    &output.full_view(),
                    &q.full_view(),
                    &cache.full_view(),
                    &cache.full_view(),
                    cfg
                )
                .is_err());
            assert_eq!(
                (commands.owner, commands.len(), commands.commands.capacity()),
                state
            );
        }
        let views = [
            output.full_view(),
            q.full_view(),
            cache.full_view(),
            cache.full_view(),
        ];
        for index in 0..4 {
            let mut bad = views.clone();
            bad[index] = bad[index].view(0, bad[index].len() - 1).unwrap();
            assert!(commands
                .incremental_causal_attention(&bad[0], &bad[1], &bad[2], &bad[3], valid)
                .is_err());
            assert_eq!(
                (commands.owner, commands.len(), commands.commands.capacity()),
                state
            );
        }
    }
    assert_bits(&read(&device, &output.full_view()), &[91.0, 92.0]);
    assert_bits(&read(&device, &cache.full_view()), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn attention_checks_each_owner_and_rejects_forged_metadata_before_recording() {
    let device = CpuExecutionDevice::new();
    let other = CpuExecutionDevice::new();
    let tensor = device.upload_f32(&[91.0, 92.0, 93.0, 94.0]).unwrap();
    let foreign = other.allocate_f32(4).unwrap();
    let good = [
        tensor.view(0, 2).unwrap(),
        tensor.view(0, 2).unwrap(),
        tensor.full_view(),
        tensor.full_view(),
    ];
    for index in 0..4 {
        let mut views = good.clone();
        views[index] = foreign.view(0, good[index].len()).unwrap();
        let mut commands = CommandList::new();
        assert!(matches!(
            commands.incremental_causal_attention(
                &views[0],
                &views[1],
                &views[2],
                &views[3],
                config()
            ),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert!(commands.is_empty() && commands.owner.is_none());
    }
    // Internal constructors permit synthetic erased payloads/metadata for backend
    // tests. Public callers cannot forge these, but recording must still refuse them
    // before copying a batch or touching output, including after a valid command.
    for index in 0..4 {
        for corruption in 0..6 {
            let mut views = good.clone();
            let len = views[index].len();
            views[index] = match corruption {
                0 => F32Tensor::new(device.owner, len, Arc::new(())).full_view(),
                1 => F32TensorView {
                    tensor: F32Tensor::new(device.owner, 5, Arc::clone(&tensor.resource)),
                    offset: 0,
                    len,
                },
                2 => F32TensorView {
                    tensor: tensor.clone(),
                    offset: 4,
                    len,
                },
                3 => F32TensorView {
                    tensor: tensor.clone(),
                    offset: usize::MAX,
                    len,
                },
                4 => foreign.view(0, len).unwrap(),
                _ => F32TensorView {
                    tensor: F32Tensor::new(device.owner, 3, Arc::clone(&tensor.resource)),
                    offset: 0,
                    len,
                },
            };
            let mut commands = CommandList::new();
            commands
                .incremental_causal_attention(&good[0], &good[1], &good[2], &good[3], config())
                .unwrap();
            commands.commands.push(Command::IncrementalCausalAttention {
                output: views[0].clone(),
                q: views[1].clone(),
                k_cache: views[2].clone(),
                v_cache: views[3].clone(),
                config: config(),
            });
            assert!(
                device.record(&commands).is_err(),
                "accepted operand {index}, corruption {corruption}"
            );
            assert_bits(
                &read(&device, &tensor.full_view()),
                &[91.0, 92.0, 93.0, 94.0],
            );
        }
    }
    let mut commands = CommandList::new();
    commands
        .incremental_causal_attention(&good[0], &good[1], &good[2], &good[3], config())
        .unwrap();
    if let Command::IncrementalCausalAttention { config, .. } = &mut commands.commands[0] {
        config.scale = f32::NAN;
    }
    assert!(device.record(&commands).is_err());
    assert!(other.record(&commands).is_err());
}
