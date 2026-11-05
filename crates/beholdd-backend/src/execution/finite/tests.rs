use super::*;

fn cancelled() -> ExecutionError {
    DeviceError::Backend {
        device: "finite-test-control",
        message: "cancelled".into(),
    }
    .into()
}

fn bits(device: &CpuExecutionDevice, view: &F32TensorView) -> Vec<u32> {
    let mut values = vec![0.0; view.len()];
    device.read_f32(view, &mut values).unwrap();
    values.into_iter().map(f32::to_bits).collect()
}

struct Unsupported;
impl sealed::Sealed for Unsupported {}
impl ExecutionDevice for Unsupported {
    fn name(&self) -> &'static str {
        "unsupported-finite-test"
    }
    fn supports(&self, _: GgmlType) -> bool {
        false
    }
    fn upload_weight(&self, _: &Weight<'_>) -> Result<EncodedWeight, ExecutionError> {
        panic!("unsupported finite validation must not upload weights")
    }
    fn prepare_weight(&self, _: &EncodedWeight) -> Result<PreparedWeight, ExecutionError> {
        panic!("unsupported finite validation must not prepare weights")
    }
    fn allocate_f32(&self, _: usize) -> Result<F32Tensor, ExecutionError> {
        panic!("unsupported finite validation must not allocate tensors")
    }
    fn write_f32(&self, _: &F32TensorView, _: &[f32]) -> Result<(), ExecutionError> {
        panic!("unsupported finite validation must not write tensors")
    }
    fn read_f32(&self, _: &F32TensorView, _: &mut [f32]) -> Result<(), ExecutionError> {
        panic!("unsupported finite validation must not read activations")
    }
    fn record(&self, _: &CommandList) -> Result<RecordedBatch, ExecutionError> {
        panic!("unsupported finite validation must not record")
    }
    fn submit(&self, _: &RecordedBatch) -> Result<Submission, ExecutionError> {
        panic!("unsupported finite validation must not submit")
    }
    fn wait(&self, _: &Submission) -> Result<(), ExecutionError> {
        panic!("unsupported finite validation must not wait")
    }
}

#[test]
fn explicit_capability_is_object_safe_and_defaults_do_not_fallback() {
    fn require_send_sync<T: Send + Sync>() {}
    require_send_sync::<FiniteCheck>();
    let cpu = CpuExecutionDevice::new();
    let device: &dyn ExecutionDevice = &cpu;
    assert_eq!(
        device.finite_check_spec(),
        Some(FiniteCheckSpec {
            max_elements: 65_536,
            resident_tensor_bytes: 0,
            initialization_write_bytes: 0,
            host_scratch_bytes: 262_144,
            physical_dispatches: 0,
            logical_submissions: 0,
            pending_write_bytes: 0,
            status_read_bytes: 0,
        })
    );
    let mut check = device.prepare_finite_check().unwrap();
    let input = device.upload_f32(&[1.0]).unwrap();
    let unsupported: &dyn ExecutionDevice = &Unsupported;
    assert_eq!(unsupported.finite_check_spec(), None);
    assert!(unsupported
        .prepare_finite_check()
        .unwrap_err()
        .to_string()
        .contains("unsupported"));
    let error = unsupported
        .check_finite(&mut check, &input.full_view(), &mut || {
            panic!("unsupported must not call control")
        })
        .unwrap_err();
    assert!(error.to_string().contains("no activation-read fallback"));
    device
        .check_finite(&mut check, &input.full_view(), &mut || Ok(()))
        .unwrap();
}

#[test]
fn finite_extremes_subnormals_and_exact_maximum_are_preserved() {
    let device = CpuExecutionDevice::new();
    let mut check = device.prepare_finite_check().unwrap();
    let patterns = [
        0x00000000, 0x80000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff, 0x00800000,
        0x80800000, 0x7f7fffff, 0xff7fffff, 0x3f800000, 0xbf800000,
    ];
    for len in [1, 63, 64, 65, 65_535, 65_536] {
        let values: Vec<f32> = (0..len)
            .map(|i| f32::from_bits(patterns[i % patterns.len()]))
            .collect();
        let tensor = device.upload_f32(&values).unwrap();
        let mut controls = 0;
        device
            .check_finite(&mut check, &tensor.full_view(), &mut || {
                controls += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(controls, 3);
        assert_eq!(
            bits(&device, &tensor.full_view()),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
}

#[test]
fn every_nonfinite_class_refuses_at_subview_indices_and_retries_current_data() {
    let device = CpuExecutionDevice::new();
    let mut check = device.prepare_finite_check().unwrap();
    let tensor = device.upload_f32(&[1.0; 67]).unwrap();
    let view = tensor.view(1, 65).unwrap();
    device
        .check_finite(&mut check, &view, &mut || Ok(()))
        .unwrap();
    for bad_bits in [
        0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000, 0x7f800001, 0xff800001,
    ] {
        for index in [0, 63, 64] {
            let bad = f32::from_bits(bad_bits);
            device
                .write_f32(&view.view(index, 1).unwrap(), &[bad])
                .unwrap();
            let before = bits(&device, &tensor.full_view());
            let mut controls = 0;
            let error = device
                .check_finite(&mut check, &view, &mut || {
                    controls += 1;
                    Ok(())
                })
                .unwrap_err();
            assert_eq!(
                controls, 2,
                "refusal must not reach successful acceptance callback"
            );
            assert!(error.to_string().contains(&format!("view index {index}")));
            assert_eq!(bits(&device, &tensor.full_view()), before);
            device
                .write_f32(&view.view(index, 1).unwrap(), &[1.0])
                .unwrap();
            device
                .check_finite(&mut check, &view, &mut || Ok(()))
                .unwrap();
        }
    }
    // Only the selected view is certified; surrounding allocations are not scanned.
    device
        .write_f32(&tensor.view(0, 1).unwrap(), &[f32::INFINITY])
        .unwrap();
    device
        .write_f32(&tensor.view(66, 1).unwrap(), &[f32::NAN])
        .unwrap();
    device
        .check_finite(&mut check, &view, &mut || Ok(()))
        .unwrap();
}

#[test]
fn foreign_checker_input_and_incompatible_payload_refuse_before_control() {
    let device = CpuExecutionDevice::new();
    let other = CpuExecutionDevice::new();
    let input = device.upload_f32(&[1.0]).unwrap();
    let foreign_input = other.upload_f32(&[1.0]).unwrap();
    let mut check = device.prepare_finite_check().unwrap();
    let mut foreign_check = other.prepare_finite_check().unwrap();
    let mut never = || -> Result<(), ExecutionError> {
        panic!("foreign/incompatible resource must refuse before control")
    };
    assert!(matches!(
        device.check_finite(&mut foreign_check, &input.full_view(), &mut never),
        Err(ExecutionError::ForeignResource {
            resource: "finite checker"
        })
    ));
    assert!(matches!(
        device.check_finite(&mut check, &foreign_input.full_view(), &mut never),
        Err(ExecutionError::ForeignResource {
            resource: "finite-check input"
        })
    ));
    let mut incompatible = FiniteCheck::new(device.owner, ());
    assert!(matches!(
        device.check_finite(&mut incompatible, &input.full_view(), &mut never),
        Err(ExecutionError::IncompatibleResource {
            resource: "finite checker",
            ..
        })
    ));
    let incompatible_input = F32Tensor::new(device.owner, 1, Arc::new(()));
    assert!(matches!(
        device.check_finite(&mut check, &incompatible_input.full_view(), &mut never),
        Err(ExecutionError::IncompatibleResource {
            resource: "finite-check input",
            ..
        })
    ));
    let clone = device.clone();
    drop(device);
    clone
        .check_finite(&mut check, &input.full_view(), &mut || Ok(()))
        .unwrap();
}

#[test]
fn empty_oversized_overflowed_and_inconsistent_views_refuse_before_snapshot() {
    let device = CpuExecutionDevice::new();
    let mut check = device.prepare_finite_check().unwrap();
    let tensor = device.upload_f32(&[1.0; 8]).unwrap();
    let oversized = F32Tensor::new(device.owner, 65_537, Arc::new(()));
    let mismatched = F32Tensor::new(
        device.owner,
        8,
        Arc::new(CpuTensor {
            values: Mutex::new(vec![1.0; 7]),
        }),
    );
    let views = [
        tensor.view(0, 0).unwrap(),
        oversized.full_view(),
        F32TensorView {
            tensor: tensor.clone(),
            offset: usize::MAX,
            len: 1,
        },
        F32TensorView {
            tensor: tensor.clone(),
            offset: 7,
            len: 2,
        },
        mismatched.full_view(),
    ];
    for view in views {
        let error = device
            .check_finite(&mut check, &view, &mut || {
                panic!("malformed input must refuse before callback/snapshot")
            })
            .unwrap_err();
        assert!(matches!(error, ExecutionError::Shape(_)), "{error}");
    }
    device
        .check_finite(&mut check, &tensor.full_view(), &mut || Ok(()))
        .unwrap();
}

#[test]
fn all_three_cancellation_boundaries_abort_without_mutation_or_checker_poison() {
    let device = CpuExecutionDevice::new();
    let tensor = device.upload_f32(&[0.0, -0.0, f32::MAX]).unwrap();
    let expected = bits(&device, &tensor.full_view());
    let mut check = device.prepare_finite_check().unwrap();
    for cancel_at in 1..=3 {
        let mut calls = 0;
        let error = device
            .check_finite(&mut check, &tensor.full_view(), &mut || {
                calls += 1;
                if calls == cancel_at {
                    Err(cancelled())
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert!(error.to_string().contains("cancelled"));
        assert_eq!(calls, cancel_at);
        assert_eq!(bits(&device, &tensor.full_view()), expected);
        device
            .check_finite(&mut check, &tensor.full_view(), &mut || Ok(()))
            .unwrap();
    }
}

#[test]
fn callbacks_hold_neither_queue_nor_input_locks() {
    let device = CpuExecutionDevice::new();
    let tensor = device.upload_f32(&[1.0]).unwrap();
    let raw = tensor.payload::<CpuTensor>().unwrap();
    let mut check = device.prepare_finite_check().unwrap();
    let mut calls = 0;
    device
        .check_finite(&mut check, &tensor.full_view(), &mut || {
            assert!(device.queue.try_lock().is_ok());
            assert!(raw.values.try_lock().is_ok());
            // Reentrant read of the small input is safe at every boundary.
            let mut value = [0.0];
            device.read_f32(&tensor.full_view(), &mut value)?;
            assert_eq!(value, [1.0]);
            calls += 1;
            Ok(())
        })
        .unwrap();
    assert_eq!(calls, 3);
}

#[test]
fn storage_is_revalidated_after_control_before_snapshot_allocation() {
    let device = CpuExecutionDevice::new();
    let tensor = device.upload_f32(&[1.0, 2.0]).unwrap();
    let raw = tensor.payload::<CpuTensor>().unwrap();
    let mut check = device.prepare_finite_check().unwrap();
    let mut calls = 0;
    let error = device
        .check_finite(&mut check, &tensor.full_view(), &mut || {
            calls += 1;
            assert_eq!(calls, 1);
            raw.values.lock().unwrap().pop();
            Ok(())
        })
        .unwrap_err();
    assert!(matches!(error, ExecutionError::Shape(_)));
    assert_eq!(calls, 1);
    raw.values.lock().unwrap().push(2.0);
    device
        .check_finite(&mut check, &tensor.full_view(), &mut || Ok(()))
        .unwrap();
}

#[test]
fn poisoned_queue_refuses_preparation_and_check_without_control() {
    let device = CpuExecutionDevice::new();
    let tensor = device.upload_f32(&[1.0]).unwrap();
    let mut check = device.prepare_finite_check().unwrap();
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = device.queue.lock().unwrap();
        panic!("injected queue poison");
    }));
    assert!(panic.is_err());
    assert_eq!(device.finite_check_spec(), Some(CPU_FINITE_SPEC));
    assert!(matches!(
        device.prepare_finite_check(),
        Err(ExecutionError::Synchronization { .. })
    ));
    assert!(matches!(
        device.check_finite(&mut check, &tensor.full_view(), &mut || {
            panic!("poisoned queue must refuse before callback")
        }),
        Err(ExecutionError::Synchronization { .. })
    ));
}
