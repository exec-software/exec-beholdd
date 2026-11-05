//! Hardware-free transport tests. The scalar emulator exists only in this module;
//! production never substitutes CPU work for a refused or failed native submission.

use std::panic::{catch_unwind, AssertUnwindSafe};

use exec_hag::budget::Reservation;

use super::*;
use crate::execution::CpuExecutionDevice;
use crate::hag::fixtures::linear_fixtures;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Event {
    Write(usize, u64, usize),
    Read(usize, u64, usize),
    Dispatch(usize),
}

#[derive(Default)]
struct Control {
    events: Vec<Event>,
    writes: u64,
    dispatches: u64,
    groups: Vec<u32>,
    fail_read: bool,
    fail_write: bool,
    fail_write_at: Option<u64>,
    panic_write_at: Option<u64>,
    fail_dispatch: Option<u64>,
    panic_dispatch: Option<u64>,
    drops: usize,
    // Models the native transport's uncertain-cleanup ownership. Tests release
    // these only after declaring their fake device work stopped, never on failure.
    quarantine: Vec<Reservation>,
}

struct MockTransport {
    slots: [Vec<u8>; 3],
    allocation_bytes: u64,
    charge: Option<Reservation>,
    control: Arc<Mutex<Control>>,
    poisoned: bool,
}

impl MockTransport {
    fn value(&self, region: BufferRegion, index: usize) -> f32 {
        let start = region.offset as usize + index * 4;
        f32::from_le_bytes(
            self.slots[region.slot][start..start + 4]
                .try_into()
                .unwrap(),
        )
    }

    fn store(&mut self, region: BufferRegion, index: usize, value: f32) {
        let start = region.offset as usize + index * 4;
        self.slots[region.slot][start..start + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn value_f64(&self, region: BufferRegion, index: usize) -> f64 {
        let start = region.offset as usize + index * 8;
        f64::from_le_bytes(
            self.slots[region.slot][start..start + 8]
                .try_into()
                .unwrap(),
        )
    }

    fn store_f64(&mut self, region: BufferRegion, index: usize, value: f64) {
        let start = region.offset as usize + index * 8;
        self.slots[region.slot][start..start + 8].copy_from_slice(&value.to_le_bytes());
    }
}

impl Drop for MockTransport {
    fn drop(&mut self) {
        let mut control = self.control.lock().unwrap();
        control.drops += 1;
        if self.poisoned {
            control.quarantine.push(self.charge.take().unwrap());
        }
    }
}

impl Transport for MockTransport {
    fn allocation_bytes(&self) -> u64 {
        self.allocation_bytes
    }

    fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        let (fail, panic) = {
            let mut control = self.control.lock().unwrap();
            control.events.push(Event::Write(slot, offset, bytes.len()));
            control.writes += 1;
            (
                std::mem::take(&mut control.fail_write)
                    || control.fail_write_at == Some(control.writes),
                control.panic_write_at == Some(control.writes),
            )
        };
        if fail || panic {
            self.poisoned = true;
            assert!(!panic, "injected host write panic with uncertain transfer");
            return Err(backend("injected host write failure"));
        }
        let start = offset as usize;
        self.slots[slot][start..start + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn read(&mut self, slot: usize, offset: u64, bytes: &mut [u8]) -> Result<(), ExecutionError> {
        let mut control = self.control.lock().unwrap();
        control.events.push(Event::Read(slot, offset, bytes.len()));
        if std::mem::take(&mut control.fail_read) {
            bytes.fill(0xff); // A failed transfer must not alter the caller's output.
            self.poisoned = true;
            return Err(backend("injected host read failure"));
        }
        let start = offset as usize;
        bytes.copy_from_slice(&self.slots[slot][start..start + bytes.len()]);
        Ok(())
    }

    fn dispatch(
        &mut self,
        program: usize,
        bindings: [BufferRegion; 3],
        scalars: [u32; 2],
        groups: u32,
    ) -> Result<u64, ExecutionError> {
        let (sequence, fail, panic) = {
            let mut control = self.control.lock().unwrap();
            control.events.push(Event::Dispatch(program));
            control.dispatches += 1;
            control.groups.push(groups);
            (
                control.dispatches,
                control.fail_dispatch == Some(control.dispatches),
                control.panic_dispatch == Some(control.dispatches),
            )
        };
        if fail || panic {
            self.poisoned = true;
            assert!(!panic, "injected dispatch panic with uncertain work");
            return Err(backend("injected dispatch failure with uncertain work"));
        }
        assert!(program <= 10);
        assert!(!overlaps(bindings[0], bindings[1]));
        assert!(!overlaps(bindings[0], bindings[2]));
        assert!(!overlaps(bindings[1], bindings[2]));
        if program == 10 {
            let cols = scalars[0] as usize;
            assert!((32..=4096).contains(&cols) && cols.is_multiple_of(32));
            assert_eq!(scalars[1], 1);
            assert_eq!(groups, cols.div_ceil(64) as u32);
            assert_eq!(
                bindings.map(|region| region.slot),
                [WEIGHTS, AUXILIARY, TENSORS]
            );
            assert_eq!(
                bindings.map(|region| region.length),
                [(cols / 32 * 34) as u32, 4, (cols * 4) as u32]
            );
            assert_eq!(bindings[0].offset % 2, 0);
            for column in 0..cols {
                let block = bindings[0].offset as usize + column / 32 * 34;
                let scale = beholdd_core::f16_to_f32(u16::from_le_bytes(
                    self.slots[WEIGHTS][block..block + 2].try_into().unwrap(),
                ));
                let value = scale * f32::from(self.slots[WEIGHTS][block + 2 + column % 32] as i8);
                self.store(bindings[2], column, value);
            }
            return Ok(sequence);
        }
        if program == 7 {
            let dim = scalars[0] as usize;
            let stride = scalars[1] as usize;
            let visible = bindings[2].length as usize / 8;
            assert!((1..=256).contains(&dim));
            assert!((dim..=4096).contains(&stride));
            assert!((1..=kernels::MAX_ATTENTION_POSITIONS).contains(&visible));
            assert_eq!(groups, visible.div_ceil(64) as u32);
            assert_eq!(
                bindings.map(|region| region.slot),
                [TENSORS, TENSORS, AUXILIARY]
            );
            assert_eq!(bindings[0].length as usize, dim * 4);
            assert_eq!(
                bindings[1].length as usize,
                ((visible - 1) * stride + dim) * 4
            );
            assert_eq!(bindings[2].length as usize, visible * 8);
            assert_eq!(bindings[2].offset % 8, 0);
            for position in 0..visible {
                let mut sum = 0.0_f64;
                for coordinate in 0..dim {
                    sum += f64::from(self.value(bindings[0], coordinate))
                        * f64::from(self.value(bindings[1], position * stride + coordinate));
                }
                self.store_f64(bindings[2], position, sum);
            }
            return Ok(sequence);
        }
        if program == 8 {
            let visible = scalars[0] as usize;
            assert!((1..=kernels::MAX_ATTENTION_POSITIONS).contains(&visible));
            assert_eq!(scalars[1], 1);
            assert_eq!(groups, 1);
            assert_eq!(bindings.map(|region| region.slot), [AUXILIARY; 3]);
            assert_eq!(
                bindings.map(|region| region.length),
                [4, visible as u32 * 8, visible as u32 * 4]
            );
            assert_eq!(bindings[1].offset % 8, 0);
            let scale = self.value(bindings[0], 0);
            assert!(scale.is_finite() && scale > 0.0);
            let mut probabilities: Vec<f32> = (0..visible)
                .map(|position| (self.value_f64(bindings[1], position) * f64::from(scale)) as f32)
                .collect();
            // Match the immediate CPU arithmetic for adapter tests; hardware
            // exponential/reciprocal qualification is separate.
            let maximum = probabilities
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            if maximum.is_finite() {
                let mut sum = 0.0_f32;
                for probability in &mut probabilities {
                    *probability = if probability.is_finite() {
                        (*probability - maximum).exp()
                    } else {
                        0.0
                    };
                    sum += *probability;
                }
                let inverse = 1.0_f32 / sum;
                for probability in &mut probabilities {
                    *probability *= inverse;
                }
            } else {
                probabilities.fill(0.0);
            }
            for (position, probability) in probabilities.into_iter().enumerate() {
                self.store(bindings[2], position, probability);
            }
            return Ok(sequence);
        }
        if program == 9 {
            let visible = scalars[0] as usize;
            let stride = scalars[1] as usize;
            let dim = bindings[2].length as usize / 4;
            assert!((1..=256).contains(&dim));
            assert!((dim..=4096).contains(&stride));
            assert!((1..=kernels::MAX_ATTENTION_POSITIONS).contains(&visible));
            assert_eq!(groups, dim.div_ceil(64) as u32);
            assert_eq!(
                bindings.map(|region| region.slot),
                [AUXILIARY, TENSORS, TENSORS]
            );
            assert_eq!(bindings[0].length as usize, visible * 4);
            assert_eq!(
                bindings[1].length as usize,
                ((visible - 1) * stride + dim) * 4
            );
            assert_eq!(bindings[2].length as usize, dim * 4);
            for coordinate in 0..dim {
                let mut sum = 0.0_f64;
                for position in 0..visible {
                    sum += f64::from(self.value(bindings[0], position))
                        * f64::from(self.value(bindings[1], position * stride + coordinate));
                }
                self.store(bindings[2], coordinate, sum as f32);
            }
            return Ok(sequence);
        }
        if program == 6 {
            let pairs = scalars[0] as usize;
            assert!((1..=kernels::MAX_ELEMENTS / 2).contains(&pairs));
            assert!((1..=4).contains(&scalars[1]));
            assert_eq!(groups, pairs.div_ceil(64) as u32);
            assert!(bindings.iter().all(|region| region.slot == TENSORS));
            assert_eq!(bindings[0].length as usize, pairs * 4);
            assert_eq!(bindings[1].length as usize, pairs * 4);
            assert_eq!(bindings[2].length as usize, pairs * 8);
            let flags = scalars[1] - 1;
            let split_half = flags & 1 != 0;
            let sign = if flags & 2 != 0 { -1.0 } else { 1.0 };
            for pair in 0..pairs {
                let first = if split_half { pair } else { 2 * pair };
                let second = if split_half { pair + pairs } else { first + 1 };
                let a = self.value(bindings[2], first);
                let b = self.value(bindings[2], second);
                let cosine = self.value(bindings[0], pair);
                let sine = sign * self.value(bindings[1], pair);
                self.store(bindings[2], first, a * cosine - b * sine);
                self.store(bindings[2], second, a * sine + b * cosine);
            }
            return Ok(sequence);
        }
        if program == 4 || program == 5 {
            let len = scalars[0] as usize;
            assert_eq!(scalars[1], 1);
            assert!((1..=kernels::MAX_ELEMENTS).contains(&len));
            assert_eq!(groups, len.div_ceil(64) as u32);
            assert_eq!(bindings[1].length as usize, len * 4);
            assert_eq!(bindings[2].length as usize, len * 4);
            if program == 4 {
                assert_eq!(bindings[0].slot, AUXILIARY);
                assert_eq!(bindings[0].length, 4);
                assert_eq!(bindings[1].slot, TENSORS);
                assert!([TENSORS, AUXILIARY].contains(&bindings[2].slot));
                let epsilon = self.value(bindings[0], 0);
                assert!(epsilon.is_finite() && epsilon >= 0.0);
                // Model CPU semantics for adapter lifetime tests, not SFU/ISA accuracy.
                let squares: f64 = (0..len)
                    .map(|index| {
                        let value = f64::from(self.value(bindings[1], index));
                        value * value
                    })
                    .sum();
                let inverse = (1.0 / (squares / len as f64 + f64::from(epsilon)).sqrt()) as f32;
                for index in 0..len {
                    let value = self.value(bindings[1], index) * inverse;
                    self.store(bindings[2], index, value);
                }
            } else {
                assert_eq!(bindings[0].slot, TENSORS);
                assert_eq!(bindings[0].length as usize, len * 4);
                assert_eq!(bindings[1].slot, AUXILIARY);
                assert_eq!(bindings[2].slot, TENSORS);
                for index in 0..len {
                    let value = self.value(bindings[1], index) * self.value(bindings[0], index);
                    self.store(bindings[2], index, value);
                }
            }
            return Ok(sequence);
        }
        if program >= 2 {
            assert_eq!(scalars[1], 1);
            let len = scalars[0] as usize;
            assert!((1..=kernels::MAX_ELEMENTS).contains(&len));
            assert_eq!(groups, len.div_ceil(64) as u32);
            assert_eq!(bindings[0].slot, AUXILIARY);
            assert_eq!(bindings[0].length, 4);
            assert_eq!(bindings[1].length as usize, len * 4);
            assert_eq!(bindings[2].length as usize, len * 4);
            let offset = bindings[0].offset as usize;
            let parameter = f32::from_le_bytes(
                self.slots[AUXILIARY][offset..offset + 4]
                    .try_into()
                    .unwrap(),
            );
            for index in 0..len {
                let input_offset = bindings[1].offset as usize + index * 4;
                let output_offset = bindings[2].offset as usize + index * 4;
                let input = f32::from_le_bytes(
                    self.slots[bindings[1].slot][input_offset..input_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let output = f32::from_le_bytes(
                    self.slots[bindings[2].slot][output_offset..output_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let result = if program == 2 {
                    let (gate, up) = if parameter > 0.0 {
                        (output.min(parameter), input.clamp(-parameter, parameter))
                    } else {
                        (output, input)
                    };
                    (gate / (1.0 + (-gate).exp())) * up
                } else {
                    let product = parameter * input;
                    output + product
                };
                self.slots[bindings[2].slot][output_offset..output_offset + 4]
                    .copy_from_slice(&result.to_le_bytes());
            }
            return Ok(sequence);
        }
        assert_eq!(bindings[0].slot, WEIGHTS);
        assert_eq!(bindings[1].slot, TENSORS);
        assert!([TENSORS, AUXILIARY].contains(&bindings[2].slot));
        let [cols, rows] = scalars.map(|value| value as usize);
        assert_eq!(groups, rows.div_ceil(64) as u32);
        let expected = if program == 0 {
            rows * cols * 2
        } else {
            rows * (cols / 32) * 34
        };
        assert_eq!(bindings[0].length as usize, expected);
        assert_eq!(bindings[1].length as usize, cols * 4);
        assert_eq!(bindings[2].length as usize, rows * 4);
        let weight_offset = bindings[0].offset as usize;
        let input_offset = bindings[1].offset as usize;
        let output_offset = bindings[2].offset as usize;
        // Direct memory accesses represent fake device loads/stores, not transport
        // host transfers. No production implementation calls this scalar oracle.
        for row in 0..rows {
            let mut sum = 0.0_f32;
            for column in 0..cols {
                let value_offset = input_offset + column * 4;
                let input = f32::from_le_bytes(
                    self.slots[TENSORS][value_offset..value_offset + 4]
                        .try_into()
                        .unwrap(),
                );
                let weight = if program == 0 {
                    let offset = weight_offset + (row * cols + column) * 2;
                    let bits = u16::from_le_bytes(
                        self.slots[WEIGHTS][offset..offset + 2].try_into().unwrap(),
                    );
                    f32::from_bits(u32::from(bits) << 16)
                } else {
                    let block = weight_offset + (row * (cols / 32) + column / 32) * 34;
                    let scale = beholdd_core::f16_to_f32(u16::from_le_bytes(
                        self.slots[WEIGHTS][block..block + 2].try_into().unwrap(),
                    ));
                    scale * f32::from(self.slots[WEIGHTS][block + 2 + column % 32] as i8)
                };
                sum += weight * input;
            }
            let offset = output_offset + row * 4;
            self.slots[bindings[2].slot][offset..offset + 4].copy_from_slice(&sum.to_le_bytes());
        }
        Ok(sequence)
    }
}

fn mock(weights: u32, tensors: u32) -> (NativeExecutionDevice, MemoryBudget, Arc<Mutex<Control>>) {
    mock_config(NativeExecutionConfig {
        weight_capacity_bytes: weights,
        tensor_capacity_bytes: tensors,
        ..Default::default()
    })
}

fn mock_config(
    config: NativeExecutionConfig,
) -> (NativeExecutionDevice, MemoryBudget, Arc<Mutex<Control>>) {
    let NativeExecutionConfig {
        weight_capacity_bytes: weights,
        tensor_capacity_bytes: tensors,
        auxiliary_capacity_bytes: auxiliary,
    } = config;
    config.validate().unwrap();
    let allocation_bytes = config.allocation_bytes().unwrap();
    let budget = MemoryBudget::new(allocation_bytes);
    let charge = budget.reserve(allocation_bytes).unwrap();
    let control = Arc::new(Mutex::new(Control::default()));
    let transport = MockTransport {
        slots: [
            vec![0xa5; weights as usize],
            vec![0xa5; tensors as usize],
            vec![0xa5; auxiliary as usize],
        ],
        allocation_bytes,
        charge: Some(charge),
        control: Arc::clone(&control),
        poisoned: false,
    };
    let device = NativeExecutionDevice::from_transport(config, Box::new(transport)).unwrap();
    (device, budget, control)
}

fn prepared_bf16(device: &NativeExecutionDevice, rows: usize, cols: usize) -> PreparedWeight {
    let bytes: Vec<_> = (0..rows * cols)
        .flat_map(|_| 0x3f80_u16.to_le_bytes())
        .collect();
    let encoded = device
        .upload_weight(&Weight {
            bytes: &bytes,
            kind: GgmlType(30),
            rows,
            cols,
        })
        .unwrap();
    device.prepare_weight(&encoded).unwrap()
}

fn one_command(device: &NativeExecutionDevice) -> (CommandList, F32Tensor, F32Tensor) {
    let prepared = prepared_bf16(device, 1, 1);
    let input = device.allocate_f32(1).unwrap();
    let output = device.allocate_f32(1).unwrap();
    device.write_f32(&input.full_view(), &[7.0]).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(
            &output.full_view(),
            &input.full_view(),
            &prepared,
            LinearMode::Overwrite,
        )
        .unwrap();
    (commands, input, output)
}

fn event_count(control: &Mutex<Control>) -> usize {
    control.lock().unwrap().events.len()
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn validates_capacities_formats_shapes_and_exact_packed_extent_before_transport() {
    for capacity in [0, MAX_TENSOR_ARENA_BYTES + 1, u32::MAX] {
        assert!(NativeExecutionConfig {
            tensor_capacity_bytes: capacity,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
    for capacity in [0, MAX_ARENA_BYTES + 1, u32::MAX] {
        assert!(NativeExecutionConfig {
            auxiliary_capacity_bytes: capacity,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
    for capacity in [0, MAX_WEIGHT_ARENA_BYTES + 1, u32::MAX] {
        assert!(NativeExecutionConfig {
            weight_capacity_bytes: capacity,
            ..Default::default()
        }
        .validate()
        .is_err());
    }
    NativeExecutionConfig {
        weight_capacity_bytes: 1,
        tensor_capacity_bytes: MAX_TENSOR_ARENA_BYTES,
        ..Default::default()
    }
    .validate()
    .unwrap();
    let (device, _, control) = mock(4096, 4096);
    assert!(device.supports(GgmlType(30)));
    assert!(device.supports(GgmlType(8)));
    assert!(!device.supports(GgmlType(0)));
    for (kind, rows, cols, bytes) in [
        (30, 0, 1, 0),
        (30, 1, 0, 0),
        (30, 4097, 1, 8194),
        (30, 1, 4097, 8194),
        (30, usize::MAX, usize::MAX, 0),
        (30, 1, 1, 1),
        (30, 1, 1, 3),
        (8, 1, 31, 34),
        (8, 1, 33, 68),
        (8, 1, 32, 33),
        (8, 1, 32, 35),
        (0, 1, 1, 4),
    ] {
        assert!(device
            .upload_weight(&Weight {
                bytes: &vec![0; bytes],
                kind: GgmlType(kind),
                rows,
                cols
            })
            .is_err());
    }
    assert!(device.allocate_f32(1025).is_err());
    assert!(device.allocate_f32(usize::MAX).is_err());
    assert_eq!(event_count(&control), 0);
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
    assert!(!device.stats().unwrap().poisoned);
}

#[test]
fn arenas_zero_initialize_reuse_coalesce_and_retain_budget_until_last_owner() {
    let (device, budget, control) = mock(8, 32);
    let charge = budget.current_bytes();
    let encoded = device
        .upload_weight(&Weight {
            bytes: &[0x80, 0x3f],
            kind: GgmlType(30),
            rows: 1,
            cols: 1,
        })
        .unwrap();
    let prepared = device.prepare_weight(&encoded).unwrap();
    let stats = device.stats().unwrap();
    assert_eq!(stats.weight_live_bytes, 4);
    assert_eq!(stats.uploaded_weight_bytes, 2);
    drop(encoded);
    assert_eq!(device.stats().unwrap().weight_live_bytes, 4);
    drop(prepared);
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
    let first = device.allocate_f32(2).unwrap();
    let middle = device.allocate_f32(2).unwrap();
    let last = device.allocate_f32(4).unwrap();
    device.write_f32(&middle.full_view(), &[3.0, -7.0]).unwrap();
    assert!(device.allocate_f32(1).is_err());
    assert!(!device.stats().unwrap().poisoned);
    drop(middle);
    let replacement = device.allocate_f32(2).unwrap();
    let mut values = [f32::NAN; 2];
    device
        .read_f32(&replacement.full_view(), &mut values)
        .unwrap();
    assert_eq!(bits(&values), vec![0, 0]);
    drop(first);
    drop(replacement);
    let coalesced = device.allocate_f32(4).unwrap();
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 32);
    drop(last);
    drop(coalesced);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    let all = device.allocate_f32(8).unwrap();
    drop(all);
    let empty = device.allocate_f32(0).unwrap();
    let before = event_count(&control);
    device.write_f32(&empty.full_view(), &[]).unwrap();
    device.read_f32(&empty.full_view(), &mut []).unwrap();
    assert_eq!(event_count(&control), before);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), charge);
    assert_eq!(control.lock().unwrap().drops, 0);
    drop(empty);
    assert_eq!(budget.current_bytes(), 0);
    assert_eq!(control.lock().unwrap().drops, 1);
}

#[test]
fn tensor_zero_initialization_bounds_every_write_and_preserves_adjacent_values() {
    for byte_len in [
        0,
        4,
        F32_INIT_CHUNK_BYTES - 4,
        F32_INIT_CHUNK_BYTES,
        F32_INIT_CHUNK_BYTES + 4,
        2 * F32_INIT_CHUNK_BYTES,
        2 * F32_INIT_CHUNK_BYTES + 12,
    ] {
        let (device, budget, control) = mock(4, (byte_len + 8) as u32);
        let prefix = device.allocate_f32(1).unwrap();
        device.write_f32(&prefix.full_view(), &[-19.0]).unwrap();
        let before = device.stats().unwrap();
        let start = event_count(&control);
        let tensor = device.allocate_f32(byte_len / 4).unwrap();
        let after = device.stats().unwrap();
        assert_eq!(
            after.host_write_bytes,
            before.host_write_bytes + byte_len as u64
        );
        assert_eq!(
            after.tensor_live_bytes,
            before.tensor_live_bytes + byte_len as u64
        );
        assert_eq!(after.tensor_peak_bytes, after.tensor_live_bytes);
        assert_eq!(after.allocation_bytes, before.allocation_bytes);
        assert_eq!(after.uploaded_weight_bytes, before.uploaded_weight_bytes);
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        assert_eq!(after.dispatch_attempts, 0);
        assert_eq!(after.logical_submissions, 0);
        assert!(!after.poisoned);
        let expected = (0..byte_len)
            .step_by(F32_INIT_CHUNK_BYTES)
            .map(|offset| {
                Event::Write(
                    TENSORS,
                    4 + offset as u64,
                    (byte_len - offset).min(F32_INIT_CHUNK_BYTES),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            &control.lock().unwrap().events[start..],
            expected.as_slice()
        );
        let suffix = device.allocate_f32(1).unwrap();
        device.write_f32(&suffix.full_view(), &[23.0]).unwrap();
        let mut actual = vec![f32::NAN; byte_len / 4];
        device.read_f32(&tensor.full_view(), &mut actual).unwrap();
        assert!(actual.into_iter().all(|value| value.to_bits() == 0));
        let mut guard = [0.0];
        device.read_f32(&prefix.full_view(), &mut guard).unwrap();
        assert_eq!(guard, [-19.0]);
        device.read_f32(&suffix.full_view(), &mut guard).unwrap();
        assert_eq!(guard, [23.0]);
        drop((prefix, tensor, suffix));
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
        drop(device);
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn chunked_tensor_reuse_clears_old_contents_and_views_retain_the_queue() {
    let byte_len = 2 * F32_INIT_CHUNK_BYTES + 4;
    let (device, budget, control) = mock(4, byte_len as u32);
    let tensor = device.allocate_f32(byte_len / 4).unwrap();
    device
        .write_f32(&tensor.full_view(), &vec![-7.0; byte_len / 4])
        .unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    assert!(device.allocate_f32(1).is_err());
    assert!(device.allocate_f32(byte_len / 4 + 1).is_err());
    assert!(device.allocate_f32(usize::MAX).is_err());
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
    drop(tensor);
    let replacement = device.allocate_f32(byte_len / 4).unwrap();
    assert_eq!(
        device.stats().unwrap().host_write_bytes,
        before.host_write_bytes + byte_len as u64
    );
    assert_eq!(
        &control.lock().unwrap().events[events..],
        &[
            Event::Write(TENSORS, 0, F32_INIT_CHUNK_BYTES),
            Event::Write(TENSORS, F32_INIT_CHUNK_BYTES as u64, F32_INIT_CHUNK_BYTES),
            Event::Write(TENSORS, (2 * F32_INIT_CHUNK_BYTES) as u64, 4),
        ]
    );
    let view = replacement.full_view();
    let peer = device.clone();
    drop((replacement, device));
    let mut actual = vec![f32::NAN; byte_len / 4];
    peer.read_f32(&view, &mut actual).unwrap();
    assert!(actual.into_iter().all(|value| value.to_bits() == 0));
    drop(peer);
    assert!(budget.current_bytes() > 0);
    assert_eq!(control.lock().unwrap().drops, 0);
    drop(view);
    assert_eq!(budget.current_bytes(), 0);
    assert_eq!(control.lock().unwrap().drops, 1);
}

#[test]
fn first_middle_or_last_zero_write_failure_retains_full_reservation_and_successful_byte_count() {
    let byte_len = 2 * F32_INIT_CHUNK_BYTES + 12;
    for failed_chunk in 1..=3 {
        let (device, budget, control) = mock(4, (byte_len + 4) as u32);
        let prefix = device.allocate_f32(1).unwrap();
        let empty_batch = device.record(&CommandList::new()).unwrap();
        let before = device.stats().unwrap();
        let events = event_count(&control);
        {
            let mut control = control.lock().unwrap();
            control.fail_write_at = Some(control.writes + failed_chunk);
        }
        assert!(device.allocate_f32(byte_len / 4).is_err());
        let after = device.stats().unwrap();
        assert!(after.poisoned);
        assert_eq!(
            after.tensor_live_bytes,
            before.tensor_live_bytes + byte_len as u64
        );
        assert_eq!(after.tensor_peak_bytes, after.tensor_live_bytes);
        assert_eq!(
            after.host_write_bytes,
            before.host_write_bytes + (failed_chunk - 1) * F32_INIT_CHUNK_BYTES as u64
        );
        assert_eq!(after.uploaded_weight_bytes, before.uploaded_weight_bytes);
        assert_eq!(after.allocation_bytes, before.allocation_bytes);
        assert_eq!(event_count(&control) - events, failed_chunk as usize);
        assert_eq!(
            control.lock().unwrap().events.last(),
            Some(&Event::Write(
                TENSORS,
                4 + (failed_chunk - 1) * F32_INIT_CHUNK_BYTES as u64,
                if failed_chunk == 3 {
                    12
                } else {
                    F32_INIT_CHUNK_BYTES
                },
            ))
        );
        let peer = device.clone();
        let stopped_events = event_count(&control);
        assert!(peer.allocate_f32(1).is_err());
        assert!(peer.write_f32(&prefix.full_view(), &[1.0]).is_err());
        assert!(peer.read_f32(&prefix.full_view(), &mut [0.0]).is_err());
        assert!(peer.record(&CommandList::new()).is_err());
        assert!(peer.submit(&empty_batch).is_err());
        assert_eq!(event_count(&control), stopped_events);
        drop((prefix, empty_batch));
        assert_eq!(
            device.stats().unwrap().tensor_live_bytes,
            after.tensor_live_bytes
        );
        drop((device, peer));
        assert!(budget.current_bytes() > 0);
        assert_eq!(control.lock().unwrap().quarantine.len(), 1);
        control.lock().unwrap().quarantine.clear(); // Only the fake transport is declared safe now.
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn zero_write_unwind_preserves_partial_accounting_and_never_reuses_uncertain_storage() {
    let byte_len = 2 * F32_INIT_CHUNK_BYTES + 4;
    let (device, budget, control) = mock(4, byte_len as u32);
    control.lock().unwrap().panic_write_at = Some(2);
    assert!(catch_unwind(AssertUnwindSafe(|| device.allocate_f32(byte_len / 4))).is_err());
    let after = device.stats().unwrap();
    assert!(after.poisoned);
    assert_eq!(after.host_write_bytes, F32_INIT_CHUNK_BYTES as u64);
    assert_eq!(after.tensor_live_bytes, byte_len as u64);
    assert_eq!(after.tensor_peak_bytes, byte_len as u64);
    assert_eq!(after.dispatch_attempts, 0);
    let events = event_count(&control);
    assert!(device.clone().allocate_f32(0).is_err());
    assert_eq!(event_count(&control), events);
    drop(device);
    assert_eq!(control.lock().unwrap().quarantine.len(), 1);
    assert!(budget.current_bytes() > 0);
    control.lock().unwrap().quarantine.clear();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn all_packed_fixtures_execute_without_submit_payload_transfers() {
    let (device, budget, _) = mock(64 * 1024, 4096);
    for fixture in linear_fixtures() {
        let encoded = device.upload_weight(&fixture.weight()).unwrap();
        let prepared = device.prepare_weight(&encoded).unwrap();
        let input = device.allocate_f32(fixture.cols).unwrap();
        let output = device.allocate_f32(fixture.rows).unwrap();
        device
            .write_f32(&input.full_view(), &fixture.input)
            .unwrap();
        device
            .write_f32(&output.full_view(), &fixture.unwritten_output().unwrap())
            .unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &prepared,
                LinearMode::Overwrite,
            )
            .unwrap();
        let batch = device.record(&commands).unwrap();
        let before = device.stats().unwrap();
        let submission = device.submit(&batch).unwrap();
        device.wait(&submission).unwrap();
        device.wait(&submission).unwrap();
        let after = device.stats().unwrap();
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        assert_eq!(after.host_write_bytes, before.host_write_bytes);
        assert_eq!(after.dispatch_count, before.dispatch_count + 1);
        assert_eq!(after.logical_submissions, before.logical_submissions + 1);
        let mut actual = vec![f32::NAN; fixture.rows];
        device.read_f32(&output.full_view(), &mut actual).unwrap();
        fixture.verify_serial_output(&actual).unwrap();
    }
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn record_refuses_later_invalid_commands_and_aliases_without_any_mutation() {
    let (device, _, control) = mock(4096, 64 * 1024);
    let (commands, input, output) = one_command(&device);
    let prepared = prepared_bf16(&device, 1, 1);
    let large_gate = device.allocate_f32(kernels::MAX_ELEMENTS + 1).unwrap();
    let large_up = device.allocate_f32(kernels::MAX_ELEMENTS + 1).unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    let mut unsupported = commands.clone();
    unsupported
        .linear(
            &output.full_view(),
            &input.full_view(),
            &prepared,
            LinearMode::ScaledAdd { scale: 1.0 },
        )
        .unwrap();
    unsupported
        .swiglu(&output.full_view(), &output.full_view(), 0.0)
        .unwrap();
    assert!(device.record(&unsupported).is_err());
    let mut unsupported = commands.clone();
    unsupported
        .swiglu(&large_gate.full_view(), &large_up.full_view(), 0.0)
        .unwrap();
    assert!(device.record(&unsupported).is_err());
    let mut alias = commands.clone();
    alias
        .linear(
            &input.full_view(),
            &input.full_view(),
            &prepared,
            LinearMode::Overwrite,
        )
        .unwrap();
    assert!(device.record(&alias).is_err());
    assert_eq!(event_count(&control), events);
    assert_eq!(device.stats().unwrap(), before);
    let mut actual = [f32::NAN];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [0.0]);
}

#[test]
fn disjoint_subviews_use_actual_offsets_and_immutable_recorded_snapshot() {
    let (device, _, _) = mock(4096, 4096);
    let _prefix = device.allocate_f32(3).unwrap();
    let tensor = device.allocate_f32(6).unwrap();
    device
        .write_f32(&tensor.full_view(), &[91.0, 2.0, 5.0, -99.0, -99.0, 92.0])
        .unwrap();
    let input = tensor.view(1, 2).unwrap();
    let output = tensor.view(3, 2).unwrap();
    let prepared = prepared_bf16(&device, 2, 2);
    let mut commands = CommandList::new();
    commands
        .linear(&output, &input, &prepared, LinearMode::Overwrite)
        .unwrap();
    let batch = device.record(&commands).unwrap();
    commands.swiglu(&output, &input, 0.0).unwrap();
    assert_eq!(batch.len(), 1);
    let submission = device.submit(&batch).unwrap();
    device.wait(&submission).unwrap();
    let mut actual = [f32::NAN; 6];
    device.read_f32(&tensor.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [91.0, 2.0, 5.0, 7.0, 7.0, 92.0]);
}

#[test]
fn partial_and_payload_aliased_views_are_refused_using_physical_ranges() {
    let (device, _, control) = mock(4096, 4096);
    let prepared = prepared_bf16(&device, 2, 2);
    let tensor = device.allocate_f32(3).unwrap();
    let input = tensor.view(0, 2).unwrap();
    let overlapping = tensor.view(1, 2).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(&overlapping, &input, &prepared, LinearMode::Overwrite)
        .unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    assert!(device.record(&commands).is_err());
    // Separate erased payload Arcs can still refer to the same arena reservation.
    // Public pointer identity cannot detect this crate-internal backend mistake.
    let allocation = &tensor.payload::<NativeTensor>().unwrap().allocation;
    let alias = F32Tensor::new(
        device.inner.owner,
        3,
        Arc::new(NativeTensor {
            allocation: Arc::clone(allocation),
            len: 3,
        }),
    );
    let output = alias.view(1, 2).unwrap();
    assert!(!input.overlaps(&output));
    let mut commands = CommandList::new();
    commands
        .linear(&output, &input, &prepared, LinearMode::Overwrite)
        .unwrap();
    assert!(device.record(&commands).is_err());
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
}

#[test]
fn weight_and_recorded_payload_metadata_cannot_override_owned_extents() {
    let (device, _, control) = mock(4096, 4096);
    let real = prepared_bf16(&device, 1, 1);
    let packed = real.payload::<NativeWeight>().unwrap();
    let owner = device.inner.owner;
    let forged = EncodedWeight::new(owner, GgmlType(30), 1, 2, 4, Arc::new(packed.clone()));
    let events = event_count(&control);
    assert!(device.prepare_weight(&forged).is_err());
    assert_eq!(event_count(&control), events);
    let input = device.allocate_f32(2).unwrap();
    let output = device.allocate_f32(1).unwrap();
    for payload in [
        Arc::new(()) as Arc<dyn std::any::Any + Send + Sync>,
        Arc::new(packed.clone()),
    ] {
        let forged = PreparedWeight::new(owner, GgmlType(30), 1, 2, 4, payload);
        let mut commands = CommandList::new();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &forged,
                LinearMode::Overwrite,
            )
            .unwrap();
        let events = event_count(&control);
        assert!(device.record(&commands).is_err());
        assert_eq!(event_count(&control), events);
    }
    let forged_batch = RecordedBatch::new(
        owner,
        1,
        Arc::new(NativeBatch {
            inner: Arc::clone(&device.inner),
            semantic_count: 0,
            commands: Vec::new(),
            _retained: Vec::new(),
        }),
    );
    let events = event_count(&control);
    assert!(device.submit(&forged_batch).is_err());
    assert_eq!(event_count(&control), events);
    assert!(!device.stats().unwrap().poisoned);
}

// Two different kernels in a resident chain. Its deliberately simple, exact
// dyadic oracle is 32 copies of sum(input), followed by 0.5 * sum(intermediate).
fn chain(device: &NativeExecutionDevice) -> (RecordedBatch, F32Tensor, F32Tensor) {
    let first = prepared_bf16(device, 32, 32);
    let mut bytes = vec![0x00, 0x38]; // f16 0.5
    bytes.extend_from_slice(&[1; 32]);
    let encoded = device
        .upload_weight(&Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 1,
            cols: 32,
        })
        .unwrap();
    let second = device.prepare_weight(&encoded).unwrap();
    let input = device.allocate_f32(32).unwrap();
    let intermediate = device.allocate_f32(32).unwrap();
    let output = device.allocate_f32(1).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(
            &intermediate.full_view(),
            &input.full_view(),
            &first,
            LinearMode::Overwrite,
        )
        .unwrap();
    commands
        .linear(
            &output.full_view(),
            &intermediate.full_view(),
            &second,
            LinearMode::Overwrite,
        )
        .unwrap();
    (device.record(&commands).unwrap(), input, output)
}

#[test]
fn resident_chain_reuses_recording_and_submission_retains_resources_and_budget() {
    let (device, budget, control) = mock(4096, 4096);
    let (batch, input, output) = chain(&device);
    let clone = device.clone();
    drop(device); // The source commands, weights and intermediate already dropped.
    let charge = budget.current_bytes();
    let mut last_submission = None;
    for round in 1..=3 {
        clone
            .write_f32(&input.full_view(), &[round as f32; 32])
            .unwrap();
        clone.write_f32(&output.full_view(), &[f32::NAN]).unwrap();
        let before = clone.stats().unwrap();
        let submission = clone.submit(&batch).unwrap();
        clone.wait(&submission).unwrap();
        let after = clone.stats().unwrap();
        assert_eq!(after.host_write_bytes, before.host_write_bytes);
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        assert_eq!(after.uploaded_weight_bytes, 2082);
        assert_eq!(after.weight_live_bytes, 2084);
        assert_eq!(after.tensor_live_bytes, 260);
        assert_eq!(after.allocation_bytes, charge);
        assert_eq!(after.dispatch_count, round * 2);
        assert_eq!(after.logical_submissions, round);
        let mut actual = [f32::NAN];
        clone.read_f32(&output.full_view(), &mut actual).unwrap();
        assert_eq!(actual, [512.0 * round as f32]);
        last_submission = Some(submission);
    }
    drop(input);
    drop(output);
    drop(batch);
    assert_eq!(clone.stats().unwrap().tensor_live_bytes, 260);
    drop(last_submission);
    assert_eq!(clone.stats().unwrap().weight_live_bytes, 0);
    assert_eq!(clone.stats().unwrap().tensor_live_bytes, 0);
    assert_eq!(budget.current_bytes(), charge);
    drop(clone);
    assert_eq!(budget.current_bytes(), 0);
    assert_eq!(control.lock().unwrap().drops, 1);
}

#[test]
fn all_interfaces_reject_foreign_owners_and_same_owner_invalid_payloads() {
    let (device, _, control) = mock(4096, 4096);
    let (foreign, _, _) = mock(4096, 4096);
    let (commands, input, _) = one_command(&foreign);
    let batch = foreign.record(&commands).unwrap();
    let submission = foreign.submit(&batch).unwrap();
    let encoded = foreign
        .upload_weight(&Weight {
            bytes: &[0, 0],
            kind: GgmlType(30),
            rows: 1,
            cols: 1,
        })
        .unwrap();
    assert!(device.prepare_weight(&encoded).is_err());
    assert!(device.write_f32(&input.full_view(), &[0.0]).is_err());
    assert!(device.read_f32(&input.full_view(), &mut [0.0]).is_err());
    assert!(device.record(&commands).is_err());
    assert!(device.submit(&batch).is_err());
    assert!(device.wait(&submission).is_err());
    let cpu = CpuExecutionDevice::new();
    assert!(device
        .read_f32(&cpu.allocate_f32(1).unwrap().full_view(), &mut [0.0])
        .is_err());
    let owner = device.inner.owner;
    let tensor = F32Tensor::new(owner, 1, Arc::new(()));
    assert!(device.write_f32(&tensor.full_view(), &[0.0]).is_err());
    assert!(device.read_f32(&tensor.full_view(), &mut [0.0]).is_err());
    let encoded = EncodedWeight::new(owner, GgmlType(30), 1, 1, 2, Arc::new(()));
    assert!(device.prepare_weight(&encoded).is_err());
    assert!(device
        .submit(&RecordedBatch::new(owner, 0, Arc::new(())))
        .is_err());
    assert!(device.wait(&Submission::new(owner, Arc::new(()))).is_err());
    // An owner identifier alone is insufficient: the payload's retained queue must
    // also agree. This is only forgeable by crate-internal implementation mistakes.
    let foreign_tensor = input.payload::<NativeTensor>().unwrap();
    let spoof = F32Tensor::new(
        owner,
        1,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&foreign_tensor.allocation),
            len: 1,
        }),
    );
    assert!(device.write_f32(&spoof.full_view(), &[0.0]).is_err());
    assert_eq!(event_count(&control), 0);
    let real = device.allocate_f32(1).unwrap();
    let real_payload = real.payload::<NativeTensor>().unwrap();
    let spoof = F32Tensor::new(
        owner,
        2,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&real_payload.allocation),
            len: 2,
        }),
    );
    let before = event_count(&control);
    assert!(device.write_f32(&spoof.full_view(), &[0.0; 2]).is_err());
    assert_eq!(event_count(&control), before);
}

#[test]
fn read_failure_preserves_caller_output_and_poison_blocks_reuse_and_transfers() {
    let (device, budget, control) = mock(4096, 4096);
    let tensor = device.allocate_f32(2).unwrap();
    control.lock().unwrap().fail_read = true;
    let mut output = [1.0, 2.0];
    assert!(device.read_f32(&tensor.full_view(), &mut output).is_err());
    assert_eq!(output, [1.0, 2.0]);
    assert_eq!(device.stats().unwrap().host_read_bytes, 0);
    let before = event_count(&control);
    assert!(device.write_f32(&tensor.full_view(), &[0.0; 2]).is_err());
    assert!(device.read_f32(&tensor.full_view(), &mut output).is_err());
    assert!(device.allocate_f32(1).is_err());
    assert!(device.record(&CommandList::new()).is_err());
    drop(tensor);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 8);
    assert!(device.stats().unwrap().poisoned);
    assert_eq!(event_count(&control), before);
    let charge = budget.current_bytes();
    drop(device);
    assert_eq!(budget.current_bytes(), charge);
    assert_eq!(control.lock().unwrap().quarantine.len(), 1);
    control.lock().unwrap().quarantine.clear(); // Fake work is now declared stopped.
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn failed_initial_upload_keeps_reserved_range_and_charge_until_safe_cleanup() {
    let (device, budget, control) = mock(4096, 4096);
    control.lock().unwrap().fail_write = true;
    assert!(device.allocate_f32(3).is_err());
    let stats = device.stats().unwrap();
    assert!(stats.poisoned);
    assert_eq!(stats.tensor_live_bytes, 12);
    assert_eq!(stats.host_write_bytes, 0);
    let before = event_count(&control);
    assert!(device.allocate_f32(1).is_err());
    assert_eq!(event_count(&control), before);
    drop(device);
    assert!(budget.current_bytes() > 0);
    control.lock().unwrap().quarantine.clear();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn failed_second_dispatch_poison_is_shared_and_previous_completion_stays_known() {
    let (device, budget, control) = mock(4096, 4096);
    let (batch, input, output) = chain(&device);
    let completed = device.submit(&batch).unwrap();
    let clone = device.clone();
    control.lock().unwrap().fail_dispatch = Some(4);
    let before = device.stats().unwrap();
    assert!(clone.submit(&batch).is_err());
    let after = device.stats().unwrap();
    assert_eq!(after.dispatch_attempts, 4);
    assert_eq!(after.dispatch_count, 3);
    assert_eq!(after.logical_submissions, 1);
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(after.host_write_bytes, before.host_write_bytes);
    assert!(after.poisoned);
    let events = event_count(&control);
    assert!(device.submit(&batch).is_err());
    assert!(device.write_f32(&input.full_view(), &[0.0; 32]).is_err());
    device.wait(&completed).unwrap(); // Cached completion predates the failed batch.
    assert_eq!(event_count(&control), events);
    drop((batch, input, output, completed));
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 260);
    drop((device, clone));
    assert!(budget.current_bytes() > 0);
    assert_eq!(control.lock().unwrap().quarantine.len(), 1);
    control.lock().unwrap().quarantine.clear();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn unwinding_transport_poisons_mutex_and_never_recycles_uncertain_ranges() {
    let (device, budget, control) = mock(4096, 4096);
    let (commands, input, output) = one_command(&device);
    let batch = device.record(&commands).unwrap();
    control.lock().unwrap().panic_dispatch = Some(1);
    assert!(catch_unwind(AssertUnwindSafe(|| device.submit(&batch))).is_err());
    assert!(device.stats().unwrap().poisoned);
    let before = event_count(&control);
    assert!(device.submit(&batch).is_err());
    assert!(device.allocate_f32(1).is_err());
    drop((commands, batch, input, output));
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 8);
    assert_eq!(event_count(&control), before);
    drop(device);
    assert!(budget.current_bytes() > 0);
    control.lock().unwrap().quarantine.clear();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn cloned_devices_serialize_whole_batches_and_empty_completion_retains_owner() {
    let (device, budget, control) = mock(4096, 4096);
    let (batch, input, output) = chain(&device);
    std::thread::scope(|scope| {
        for _ in 0..8 {
            let clone = device.clone();
            let batch = batch.clone();
            scope.spawn(move || {
                for _ in 0..3 {
                    let submission = clone.submit(&batch).unwrap();
                    clone.wait(&submission).unwrap();
                }
            });
        }
    });
    let programs: Vec<_> = control
        .lock()
        .unwrap()
        .events
        .iter()
        .filter_map(|event| {
            if let Event::Dispatch(program) = event {
                Some(*program)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(programs, [0, 1].repeat(24));
    assert_eq!(device.stats().unwrap().logical_submissions, 24);
    assert_eq!(device.stats().unwrap().dispatch_count, 48);
    drop((batch, input, output));
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    let empty_batch = device.record(&CommandList::new()).unwrap();
    let submission = device.submit(&empty_batch).unwrap();
    assert_eq!(device.stats().unwrap().logical_submissions, 25);
    assert_eq!(device.stats().unwrap().dispatch_count, 48);
    drop((device, empty_batch));
    assert!(budget.current_bytes() > 0);
    drop(submission);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn swiglu_clamps_asymmetrically_reuses_parameters_and_skips_empty_dispatches() {
    let (device, budget, _) = mock(4096, 64 * 1024);
    for len in [0, 1, 31, 32, 33, 64, 65, 3072, 4096] {
        for limit in [0.0, 1.0, 10.0] {
            let gate = device.allocate_f32(len).unwrap();
            let up = device.allocate_f32(len).unwrap();
            let mut commands = CommandList::new();
            commands
                .swiglu(&gate.full_view(), &up.full_view(), limit)
                .unwrap();
            let batch = device.record(&commands).unwrap();
            assert_eq!(batch.len(), 1);
            assert_eq!(
                device.stats().unwrap().auxiliary_live_bytes,
                if len == 0 { 0 } else { 4 }
            );
            drop(commands);
            for round in 0..2 {
                let mut expected: Vec<f32> = (0..len)
                    .map(|index| [-20.0, -2.0, 0.0, 1.0, 4.0, 20.0][(index + round) % 6])
                    .collect();
                let up_values: Vec<f32> = (0..len)
                    .map(|index| [-15.0, 0.0, 0.5, 2.0, 9.0, 15.0][(index + round) % 6])
                    .collect();
                device.write_f32(&gate.full_view(), &expected).unwrap();
                device.write_f32(&up.full_view(), &up_values).unwrap();
                beholdd_kernel::vecops::swiglu(&mut expected, &up_values, limit);
                let before = device.stats().unwrap();
                let submission = device.submit(&batch).unwrap();
                device.wait(&submission).unwrap();
                device.wait(&submission).unwrap();
                let after = device.stats().unwrap();
                assert_eq!(after.logical_submissions, before.logical_submissions + 1);
                assert_eq!(
                    after.dispatch_count,
                    before.dispatch_count + u64::from(len != 0)
                );
                assert_eq!(after.host_write_bytes, before.host_write_bytes);
                assert_eq!(after.host_read_bytes, before.host_read_bytes);
                let mut actual = vec![f32::NAN; len];
                device.read_f32(&gate.full_view(), &mut actual).unwrap();
                assert_eq!(bits(&actual), bits(&expected));
            }
        }
    }
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn scaled_add_is_one_semantic_command_and_two_resident_dispatches() {
    let (device, _, control) = mock(4096, 4096);
    for rows in [1, 31, 64] {
        for scale in [0.0, -0.0, 0.5, -0.75] {
            let prepared = prepared_bf16(&device, rows, 32);
            let input = device.allocate_f32(32).unwrap();
            let output = device.allocate_f32(rows).unwrap();
            let mut commands = CommandList::new();
            commands
                .linear(
                    &output.full_view(),
                    &input.full_view(),
                    &prepared,
                    LinearMode::ScaledAdd { scale },
                )
                .unwrap();
            let batch = device.record(&commands).unwrap();
            assert_eq!(batch.len(), 1);
            assert_eq!(
                device.stats().unwrap().auxiliary_live_bytes,
                (rows * 4 + 4) as u64
            );
            drop((prepared, commands));
            let mut expected = vec![-2.0; rows];
            device.write_f32(&output.full_view(), &expected).unwrap();
            for round in 1..=3 {
                device
                    .write_f32(&input.full_view(), &[round as f32; 32])
                    .unwrap();
                for value in &mut expected {
                    *value += scale * (round as f32 * 32.0);
                }
                let before = device.stats().unwrap();
                let events = event_count(&control);
                let submission = device.submit(&batch).unwrap();
                device.wait(&submission).unwrap();
                let after = device.stats().unwrap();
                assert_eq!(after.logical_submissions, before.logical_submissions + 1);
                assert_eq!(after.dispatch_count, before.dispatch_count + 2);
                assert_eq!(after.host_write_bytes, before.host_write_bytes);
                assert_eq!(after.host_read_bytes, before.host_read_bytes);
                assert_eq!(
                    &control.lock().unwrap().events[events..],
                    &[Event::Dispatch(0), Event::Dispatch(3)]
                );
                let mut actual = vec![f32::NAN; rows];
                device.read_f32(&output.full_view(), &mut actual).unwrap();
                assert_eq!(bits(&actual), bits(&expected));
            }
        }
    }
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
}

#[test]
fn auxiliary_exhaustion_on_parameter_rolls_back_workspace_without_poison_or_deadlock() {
    let (device, _, control) = mock(4096, 4096);
    let (mut commands, input, output) = one_command(&device);
    let prepared = prepared_bf16(&device, 1, 1);
    commands
        .linear(
            &output.full_view(),
            &input.full_view(),
            &prepared,
            LinearMode::ScaledAdd { scale: 2.0 },
        )
        .unwrap();
    // Leave one word: workspace succeeds; the parameter reservation must fail.
    let blocker = device
        .allocate(AUXILIARY, &vec![0; DEFAULT_AUXILIARY_BYTES as usize - 4])
        .unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    assert!(device.record(&commands).is_err());
    let after = device.stats().unwrap();
    assert_eq!(after.auxiliary_live_bytes, before.auxiliary_live_bytes);
    assert_eq!(
        after.auxiliary_peak_bytes,
        u64::from(DEFAULT_AUXILIARY_BYTES)
    );
    assert!(!after.poisoned);
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(after.host_write_bytes, before.host_write_bytes + 4);
    assert_eq!(
        &control.lock().unwrap().events[events..],
        &[Event::Write(
            AUXILIARY,
            u64::from(DEFAULT_AUXILIARY_BYTES - 4),
            4
        )]
    );
    let mut actual = [f32::NAN];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [0.0]);
    drop(blocker);
    let batch = device.record(&commands).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 8);
    let submission = device.submit(&batch).unwrap();
    device.wait(&submission).unwrap();
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [21.0]);
    drop((submission, batch));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
}

#[test]
fn elementwise_parameter_validation_and_alias_rejection_never_touch_transport() {
    let (device, _, control) = mock(4096, 4096);
    let gate = device.allocate_f32(4).unwrap();
    let up = device.allocate_f32(4).unwrap();
    let prepared = prepared_bf16(&device, 2, 2);
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for limit in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert!(device
            .plan(&Command::SwiGlu {
                gate: gate.full_view(),
                up: up.full_view(),
                limit
            })
            .is_err());
    }
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(device
            .plan(&Command::Linear {
                output: gate.view(0, 2).unwrap(),
                input: up.view(0, 2).unwrap(),
                weight: prepared.clone(),
                mode: LinearMode::ScaledAdd { scale }
            })
            .is_err());
    }
    for offset in [0, 1] {
        let mut commands = CommandList::new();
        commands
            .swiglu(
                &gate.view(0, 2).unwrap(),
                &gate.view(offset, 2).unwrap(),
                1.0,
            )
            .unwrap();
        assert!(device.record(&commands).is_err());
        let mut commands = CommandList::new();
        commands
            .linear(
                &gate.view(offset, 2).unwrap(),
                &gate.view(0, 2).unwrap(),
                &prepared,
                LinearMode::ScaledAdd { scale: 1.0 },
            )
            .unwrap();
        assert!(device.record(&commands).is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
}

fn ffn_commands(
    device: &impl ExecutionDevice,
    limit: f32,
    scale: f32,
) -> (CommandList, F32Tensor, F32Tensor) {
    let gate_bytes: Vec<_> = (0..32 * 32)
        .flat_map(|index| {
            let value = ((index % 5) as f32 - 2.0) * 0.25;
            ((value.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect();
    let mut up_bytes = Vec::new();
    for row in 0..32 {
        up_bytes.extend_from_slice(&0x3400_u16.to_le_bytes()); // 0.25
        up_bytes.extend((0..32).map(|col| (((row + col) % 5) as i8 - 2) as u8));
    }
    let mut down_bytes = Vec::new();
    for row in 0..4 {
        down_bytes.extend_from_slice(&0x3800_u16.to_le_bytes()); // 0.5
        down_bytes.extend((0..32).map(|col| (((row * 3 + col) % 7) as i8 - 3) as u8));
    }
    let mut weights = Vec::new();
    for (bytes, kind, rows, cols) in [
        (&gate_bytes, GgmlType(30), 32, 32),
        (&up_bytes, GgmlType(8), 32, 32),
        (&down_bytes, GgmlType(8), 4, 32),
    ] {
        let encoded = device
            .upload_weight(&Weight {
                bytes,
                kind,
                rows,
                cols,
            })
            .unwrap();
        weights.push(device.prepare_weight(&encoded).unwrap());
    }
    let input = device.allocate_f32(32).unwrap();
    let gate = device.allocate_f32(32).unwrap();
    let up = device.allocate_f32(32).unwrap();
    let output = device.allocate_f32(4).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(
            &gate.full_view(),
            &input.full_view(),
            &weights[0],
            LinearMode::Overwrite,
        )
        .unwrap();
    commands
        .linear(
            &up.full_view(),
            &input.full_view(),
            &weights[1],
            LinearMode::Overwrite,
        )
        .unwrap();
    commands
        .swiglu(&gate.full_view(), &up.full_view(), limit)
        .unwrap();
    commands
        .linear(
            &output.full_view(),
            &gate.full_view(),
            &weights[2],
            LinearMode::ScaledAdd { scale },
        )
        .unwrap();
    (commands, input, output)
}

#[test]
fn bounded_ffn_matches_cpu_reference_across_repeated_resident_batches() {
    let (device, budget, control) = mock(4096, 4096);
    let clone = device.clone();
    drop(device);
    let cpu = CpuExecutionDevice::new();
    for (limit, scale) in [(0.0, 0.5), (1.0, -0.75), (10.0, 0.0)] {
        let (commands, input, output) = ffn_commands(&clone, limit, scale);
        let (cpu_commands, cpu_input, cpu_output) = ffn_commands(&cpu, limit, scale);
        let batch = clone.record(&commands).unwrap();
        let cpu_batch = cpu.record(&cpu_commands).unwrap();
        drop((commands, cpu_commands));
        assert_eq!(batch.len(), 4);
        assert_eq!(clone.stats().unwrap().auxiliary_live_bytes, 24);
        let packed_bytes = clone.stats().unwrap().uploaded_weight_bytes;
        for round in 0..3 {
            let input_values: Vec<f32> = (0..32)
                .map(|col| ((col * 3 + round * 5) % 17) as f32 / 4.0 - 2.0)
                .collect();
            let residual = [round as f32, -3.0, 0.125, 8.0];
            clone.write_f32(&input.full_view(), &input_values).unwrap();
            cpu.write_f32(&cpu_input.full_view(), &input_values)
                .unwrap();
            clone.write_f32(&output.full_view(), &residual).unwrap();
            cpu.write_f32(&cpu_output.full_view(), &residual).unwrap();
            let before = clone.stats().unwrap();
            let events = event_count(&control);
            let submission = clone.submit(&batch).unwrap();
            clone.wait(&submission).unwrap();
            clone.wait(&submission).unwrap();
            let after = clone.stats().unwrap();
            assert_eq!(after.dispatch_count, before.dispatch_count + 5);
            assert_eq!(after.logical_submissions, before.logical_submissions + 1);
            assert_eq!(after.host_write_bytes, before.host_write_bytes);
            assert_eq!(after.host_read_bytes, before.host_read_bytes);
            assert_eq!(after.uploaded_weight_bytes, packed_bytes);
            assert_eq!(after.allocation_bytes, budget.current_bytes());
            assert_eq!(
                &control.lock().unwrap().events[events..],
                &[
                    Event::Dispatch(0),
                    Event::Dispatch(1),
                    Event::Dispatch(2),
                    Event::Dispatch(1),
                    Event::Dispatch(3),
                ]
            );
            cpu.wait(&cpu.submit(&cpu_batch).unwrap()).unwrap();
            let mut actual = [f32::NAN; 4];
            let mut expected = [f32::NAN; 4];
            clone.read_f32(&output.full_view(), &mut actual).unwrap();
            cpu.read_f32(&cpu_output.full_view(), &mut expected)
                .unwrap();
            // This checks adapter lowering against the CPU executor, not native
            // transcendental ISA accuracy. Modest tolerance admits CPU reductions.
            for (value, reference) in actual.into_iter().zip(expected) {
                assert!(value.is_finite() && reference.is_finite());
                assert!((value - reference).abs() <= 1.0e-5 + 2.0e-5 * reference.abs());
            }
        }
    }
    assert_eq!(clone.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(clone.stats().unwrap().tensor_live_bytes, 0);
    assert_eq!(clone.stats().unwrap().weight_live_bytes, 0);
    drop(clone);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn scaled_add_second_stage_failure_preserves_old_completion_and_quarantines_auxiliary() {
    let (device, budget, control) = mock(4096, 4096);
    let prepared = prepared_bf16(&device, 1, 1);
    let input = device.allocate_f32(1).unwrap();
    let output = device.allocate_f32(1).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(
            &output.full_view(),
            &input.full_view(),
            &prepared,
            LinearMode::ScaledAdd { scale: -2.0 },
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    assert_eq!(batch.len(), 1);
    let completed = device.submit(&batch).unwrap();
    control.lock().unwrap().fail_dispatch = Some(4);
    let before = device.stats().unwrap();
    assert!(device.submit(&batch).is_err());
    let after = device.stats().unwrap();
    assert_eq!(after.dispatch_attempts, 4);
    assert_eq!(after.dispatch_count, 3);
    assert_eq!(after.logical_submissions, 1);
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(after.host_write_bytes, before.host_write_bytes);
    assert_eq!(after.auxiliary_live_bytes, 8);
    device.wait(&completed).unwrap();
    let events = event_count(&control);
    assert!(device.submit(&batch).is_err());
    assert!(device.record(&commands).is_err());
    assert!(device.write_f32(&output.full_view(), &[9.0]).is_err());
    assert!(device.read_f32(&output.full_view(), &mut [0.0]).is_err());
    assert_eq!(event_count(&control), events);
    drop((commands, batch, completed, prepared, input, output));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 8);
    drop(device);
    assert!(budget.current_bytes() > 0);
    assert_eq!(control.lock().unwrap().quarantine.len(), 1);
    control.lock().unwrap().quarantine.clear();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn cloned_scaled_batches_share_scratch_safely_and_completion_retains_auxiliary() {
    let (device, budget, control) = mock(4096, 4096);
    let prepared = prepared_bf16(&device, 1, 1);
    let input = device.allocate_f32(1).unwrap();
    let output = device.allocate_f32(1).unwrap();
    device.write_f32(&input.full_view(), &[2.0]).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(
            &output.full_view(),
            &input.full_view(),
            &prepared,
            LinearMode::ScaledAdd { scale: 0.5 },
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    drop((prepared, commands));
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let clone = device.clone();
            let batch = batch.clone();
            scope.spawn(move || {
                for _ in 0..5 {
                    clone.wait(&clone.submit(&batch).unwrap()).unwrap();
                }
            });
        }
    });
    let mut actual = [f32::NAN];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [20.0]);
    let programs: Vec<_> = control
        .lock()
        .unwrap()
        .events
        .iter()
        .filter_map(|event| {
            if let Event::Dispatch(program) = event {
                Some(*program)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(programs, [0, 3].repeat(20));
    let completed = device.submit(&batch).unwrap();
    drop((batch, input, output));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 8);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 8);
    device.wait(&completed).unwrap();
    device.wait(&completed).unwrap();
    drop(completed);
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
    assert!(budget.current_bytes() > 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn capacity_cost_includes_rounding_guards_and_auxiliary_before_device_access() {
    let config = NativeExecutionConfig {
        weight_capacity_bytes: 10_027_008,
        tensor_capacity_bytes: 32_768,
        auxiliary_capacity_bytes: 32_768,
    };
    // The real dense FFN payload is already page-aligned. All three guard pairs
    // plus the two-page shader library and command page add exactly 36 KiB.
    assert_eq!(config.allocation_bytes().unwrap(), 10_129_408);
    let partial = NativeExecutionConfig {
        weight_capacity_bytes: 4097,
        tensor_capacity_bytes: 1,
        auxiliary_capacity_bytes: 1,
    };
    assert_eq!(partial.allocation_bytes().unwrap(), 53_248);
    let (device, budget, _) = mock_config(partial);
    assert_eq!(device.stats().unwrap().allocation_bytes, 53_248);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
    for auxiliary in [0, MAX_ARENA_BYTES + 1, u32::MAX] {
        assert!(NativeExecutionConfig {
            auxiliary_capacity_bytes: auxiliary,
            ..config
        }
        .allocation_bytes()
        .is_err());
    }
}

#[test]
fn multiple_groups_and_large_private_scratch_preserve_views_and_reuse() {
    for rows in [65, 1024, 4096] {
        let config = NativeExecutionConfig {
            weight_capacity_bytes: 512 * 1024,
            tensor_capacity_bytes: 64 * 1024,
            auxiliary_capacity_bytes: 32 * 1024,
        };
        let (device, budget, control) = mock_config(config);
        let weight = prepared_bf16(&device, rows, 32);
        let input = device.allocate_f32(32).unwrap();
        let output = device.allocate_f32(rows + 2).unwrap();
        let active = output.view(1, rows).unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(
                &active,
                &input.full_view(),
                &weight,
                LinearMode::ScaledAdd { scale: 0.5 },
            )
            .unwrap();
        let batch = device.record(&commands).unwrap();
        assert_eq!(
            device.stats().unwrap().auxiliary_live_bytes,
            (rows * 4 + 4) as u64
        );
        drop((commands, weight));
        for value in [2.0, -4.0] {
            device.write_f32(&input.full_view(), &[value; 32]).unwrap();
            let mut before = vec![-3.0; rows + 2];
            before[0] = 91.0;
            before[rows + 1] = -92.0;
            device.write_f32(&output.full_view(), &before).unwrap();
            let stats = device.stats().unwrap();
            let submission = device.submit(&batch).unwrap();
            device.wait(&submission).unwrap();
            device.wait(&submission).unwrap();
            let after = device.stats().unwrap();
            assert_eq!(after.host_read_bytes, stats.host_read_bytes);
            assert_eq!(after.host_write_bytes, stats.host_write_bytes);
            assert_eq!(after.dispatch_count, stats.dispatch_count + 2);
            let mut actual = vec![0.0; rows + 2];
            device.read_f32(&output.full_view(), &mut actual).unwrap();
            assert_eq!(actual[0], 91.0);
            assert_eq!(actual[rows + 1], -92.0);
            let expected = -3.0 + 0.5 * 32.0 * value;
            assert!(actual[1..rows + 1]
                .iter()
                .all(|v| v.to_bits() == expected.to_bits()));
        }
        assert_eq!(
            control.lock().unwrap().groups,
            vec![rows.div_ceil(64) as u32; 4]
        );
        drop((active, input, output, batch));
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
        drop(device);
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn model_width_scaled_add_refuses_undersized_auxiliary_without_poisoning() {
    let (device, budget, control) = mock(4096, 8192);
    let weight = prepared_bf16(&device, 1024, 1);
    let input = device.allocate_f32(1).unwrap();
    let output = device.allocate_f32(1024).unwrap();
    let mut commands = CommandList::new();
    commands
        .linear(
            &output.full_view(),
            &input.full_view(),
            &weight,
            LinearMode::ScaledAdd { scale: 1.0 },
        )
        .unwrap();
    assert!(device.record(&commands).is_err());
    let stats = device.stats().unwrap();
    assert_eq!(stats.auxiliary_live_bytes, 0);
    assert!(!stats.poisoned);
    assert_eq!(control.lock().unwrap().dispatches, 0);
    drop((commands, input, output, weight, device));
    assert_eq!(budget.current_bytes(), 0);
}

fn rmsnorm_oracle(input: &[f32], weight: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut output = vec![f32::NAN; input.len()];
    crate::Device::rmsnorm(&crate::cpu::Cpu::new(), &mut output, input, weight, eps).unwrap();
    output
}

#[test]
fn rmsnorm_replays_resident_values_and_preserves_guarded_subviews_at_supported_widths() {
    for len in [1, 65, 128, 1024, 4096] {
        for weighted in [false, true] {
            let (device, budget, control) = mock_config(NativeExecutionConfig {
                weight_capacity_bytes: 4,
                tensor_capacity_bytes: 64 * 1024,
                auxiliary_capacity_bytes: 32 * 1024,
            });
            let input = device.allocate_f32(len + 2).unwrap();
            let weight = device.allocate_f32(len + 2).unwrap();
            let output = device.allocate_f32(len + 2).unwrap();
            let input_view = input.view(1, len).unwrap();
            let weight_view = weight.view(1, len).unwrap();
            let output_view = output.view(1, len).unwrap();
            let mut commands = CommandList::new();
            let eps = if len == 1 { 0.0 } else { 1e-5 };
            commands
                .rmsnorm(
                    &output_view,
                    &input_view,
                    weighted.then_some(&weight_view),
                    eps,
                )
                .unwrap();
            let before = device.stats().unwrap();
            let events = event_count(&control);
            let batch = device.record(&commands).unwrap();
            drop(commands);
            let auxiliary = 4 + if weighted { len as u64 * 4 } else { 0 };
            let recorded = device.stats().unwrap();
            assert_eq!(batch.len(), 1);
            assert_eq!(recorded.auxiliary_live_bytes, auxiliary);
            assert_eq!(
                recorded.host_write_bytes,
                before.host_write_bytes + auxiliary
            );
            assert_eq!(recorded.host_read_bytes, before.host_read_bytes);
            assert!(control.lock().unwrap().events[events..]
                .iter()
                .all(|event| matches!(event, Event::Write(AUXILIARY, _, _))));

            let mut completed = None;
            for round in 0..2 {
                let mut input_values = vec![91.0; len + 2];
                let mut weight_values = vec![-92.0; len + 2];
                for index in 0..len {
                    input_values[index + 1] =
                        (index as i32 % 19 - 9) as f32 * 0.25 + round as f32 * 0.0625;
                    weight_values[index + 1] =
                        (index as i32 % 7 - 3) as f32 * 0.5 - round as f32 * 0.125;
                }
                let mut initial_output = vec![123.0; len + 2];
                initial_output[len + 1] = -456.0;
                device.write_f32(&input.full_view(), &input_values).unwrap();
                device
                    .write_f32(&weight.full_view(), &weight_values)
                    .unwrap();
                device
                    .write_f32(&output.full_view(), &initial_output)
                    .unwrap();
                let expected = rmsnorm_oracle(
                    &input_values[1..len + 1],
                    weighted.then_some(&weight_values[1..len + 1]),
                    eps,
                );
                let before = device.stats().unwrap();
                let events = event_count(&control);
                let submission = device.submit(&batch).unwrap();
                device.wait(&submission).unwrap();
                device.wait(&submission).unwrap();
                let after = device.stats().unwrap();
                let programs: &[Event] = if weighted {
                    &[Event::Dispatch(4), Event::Dispatch(5)]
                } else {
                    &[Event::Dispatch(4)]
                };
                assert_eq!(&control.lock().unwrap().events[events..], programs);
                assert_eq!(
                    after.dispatch_count,
                    before.dispatch_count + programs.len() as u64
                );
                assert_eq!(after.logical_submissions, before.logical_submissions + 1);
                assert_eq!(after.host_read_bytes, before.host_read_bytes);
                assert_eq!(after.host_write_bytes, before.host_write_bytes);
                let mut actual = vec![f32::NAN; len + 2];
                device.read_f32(&output.full_view(), &mut actual).unwrap();
                assert_eq!(actual[0], initial_output[0]);
                assert_eq!(actual[len + 1], initial_output[len + 1]);
                assert_eq!(bits(&actual[1..len + 1]), bits(&expected));
                device.read_f32(&input.full_view(), &mut actual).unwrap();
                assert_eq!(bits(&actual), bits(&input_values));
                device.read_f32(&weight.full_view(), &mut actual).unwrap();
                assert_eq!(bits(&actual), bits(&weight_values));
                completed = Some(submission);
            }
            assert_eq!(
                control.lock().unwrap().groups,
                vec![len.div_ceil(64) as u32; if weighted { 4 } else { 2 }]
            );
            // The final completion alone keeps the recorded parameter, scratch,
            // input, optional weight and output allocations alive.
            drop((
                batch,
                input_view,
                weight_view,
                output_view,
                input,
                weight,
                output,
            ));
            let retained = device.stats().unwrap();
            assert_eq!(retained.auxiliary_live_bytes, auxiliary);
            assert_eq!(
                retained.tensor_live_bytes,
                ((len + 2) * 4 * if weighted { 3 } else { 2 }) as u64
            );
            device.wait(completed.as_ref().unwrap()).unwrap();
            drop(completed);
            assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
            assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
            drop(device);
            assert_eq!(budget.current_bytes(), 0);
        }
    }
}

#[test]
fn rmsnorm_read_only_input_weight_overlap_is_allowed_and_uses_submission_values() {
    let (device, budget, control) = mock(4, 4096);
    let tensor = device.allocate_f32(9).unwrap();
    let input = tensor.view(1, 3).unwrap();
    let weight = tensor.view(2, 3).unwrap();
    let output = tensor.view(5, 3).unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(&output, &input, Some(&weight), 0.25)
        .unwrap();
    let batch = device.record(&commands).unwrap();
    for values in [
        [91.0, 1.0, -2.0, 3.0, -4.0, 99.0, 99.0, 99.0, 92.0],
        [91.0, -4.0, 0.0, 2.0, -3.0, -99.0, -99.0, -99.0, 92.0],
    ] {
        device.write_f32(&tensor.full_view(), &values).unwrap();
        let expected = rmsnorm_oracle(&values[1..4], Some(&values[2..5]), 0.25);
        let events = event_count(&control);
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        assert_eq!(
            &control.lock().unwrap().events[events..],
            &[Event::Dispatch(4), Event::Dispatch(5)]
        );
        let mut actual = [f32::NAN; 9];
        device.read_f32(&tensor.full_view(), &mut actual).unwrap();
        assert_eq!(bits(&actual[..5]), bits(&values[..5]));
        assert_eq!(bits(&actual[5..8]), bits(&expected));
        assert_eq!(actual[8], values[8]);
    }
    drop((commands, batch, input, weight, output, tensor, device));
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn rmsnorm_validates_every_operand_epsilon_and_extent_without_transport_mutation() {
    let (device, _, control) = mock(4, 4096);
    let (foreign, _, _) = mock(4, 4096);
    let input = device.allocate_f32(3).unwrap();
    let output = device.allocate_f32(3).unwrap();
    let weight = device.allocate_f32(3).unwrap();
    let foreign_tensor = foreign.allocate_f32(3).unwrap();
    let owner = device.inner.owner;
    let wrong_payload = F32Tensor::new(owner, 3, Arc::new(()));
    let wrong_queue = F32Tensor::new(
        owner,
        3,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&foreign_tensor.payload::<NativeTensor>().unwrap().allocation),
            len: 3,
        }),
    );
    let wrong_extent = F32Tensor::new(
        owner,
        4,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&input.payload::<NativeTensor>().unwrap().allocation),
            len: 4,
        }),
    );
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for invalid in [&foreign_tensor, &wrong_payload, &wrong_queue, &wrong_extent] {
        for role in 0..3 {
            let command = Command::RmsNorm {
                output: if role == 0 {
                    invalid.full_view()
                } else {
                    output.full_view()
                },
                input: if role == 1 {
                    invalid.full_view()
                } else {
                    input.full_view()
                },
                weight: Some(if role == 2 {
                    invalid.full_view()
                } else {
                    weight.full_view()
                }),
                eps: 1e-5,
            };
            assert!(device.plan(&command).is_err());
        }
    }
    for eps in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1.0] {
        assert!(device
            .plan(&Command::RmsNorm {
                output: output.full_view(),
                input: input.full_view(),
                weight: None,
                eps,
            })
            .is_err());
    }
    for (input_len, output_len, weight_len) in [(0, 0, 0), (3, 2, 3), (3, 3, 2)] {
        assert!(device
            .plan(&Command::RmsNorm {
                output: output.view(0, output_len).unwrap(),
                input: input.view(0, input_len).unwrap(),
                weight: Some(weight.view(0, weight_len).unwrap()),
                eps: 1e-5,
            })
            .is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
}

#[test]
fn rmsnorm_refuses_late_oversized_and_physical_output_aliases_before_lowering() {
    let (device, _, control) = mock(4, 64 * 1024);
    let input = device.allocate_f32(3).unwrap();
    let weight = device.allocate_f32(3).unwrap();
    let output = device.allocate_f32(2).unwrap();
    let large_input = device.allocate_f32(kernels::MAX_ELEMENTS + 1).unwrap();
    let large_output = device.allocate_f32(kernels::MAX_ELEMENTS + 1).unwrap();
    let hidden_alias = F32Tensor::new(
        device.inner.owner,
        3,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&input.payload::<NativeTensor>().unwrap().allocation),
            len: 3,
        }),
    );
    assert!(!input
        .view(0, 2)
        .unwrap()
        .overlaps(&hidden_alias.view(1, 2).unwrap()));
    let cases = [
        (
            input.view(1, 2).unwrap(),
            input.view(0, 2).unwrap(),
            weight.view(0, 2).unwrap(),
        ),
        (
            hidden_alias.view(1, 2).unwrap(),
            input.view(0, 2).unwrap(),
            weight.view(0, 2).unwrap(),
        ),
        (
            weight.view(1, 2).unwrap(),
            input.view(0, 2).unwrap(),
            weight.view(0, 2).unwrap(),
        ),
        (
            large_output.full_view(),
            large_input.full_view(),
            large_input.full_view(),
        ),
    ];
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for (invalid_output, invalid_input, invalid_weight) in cases {
        let mut commands = CommandList::new();
        commands
            .rmsnorm(&output.full_view(), &input.view(0, 2).unwrap(), None, 0.0)
            .unwrap();
        commands
            .rmsnorm(&invalid_output, &invalid_input, Some(&invalid_weight), 1e-5)
            .unwrap();
        assert!(device.record(&commands).is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
}

#[test]
fn rmsnorm_later_scalar_exhaustion_rolls_back_shared_workspace_and_parameters() {
    let len = 1024;
    let auxiliary = (len * 4 + 8) as u32;
    let (device, budget, control) = mock_config(NativeExecutionConfig {
        weight_capacity_bytes: 4,
        tensor_capacity_bytes: 16 * 1024,
        auxiliary_capacity_bytes: auxiliary,
    });
    let input = device.allocate_f32(len).unwrap();
    let weight = device.allocate_f32(len).unwrap();
    let output = device.allocate_f32(len).unwrap();
    device
        .write_f32(&input.full_view(), &vec![2.0; len])
        .unwrap();
    device
        .write_f32(&weight.full_view(), &vec![3.0; len])
        .unwrap();
    device
        .write_f32(&output.full_view(), &vec![-17.0; len])
        .unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(
            &output.full_view(),
            &input.full_view(),
            Some(&weight.full_view()),
            0.0,
        )
        .unwrap();
    let single = commands.clone();
    commands
        .rmsnorm(
            &output.full_view(),
            &input.full_view(),
            Some(&weight.full_view()),
            1.0,
        )
        .unwrap();
    commands
        .rmsnorm(
            &output.full_view(),
            &input.full_view(),
            Some(&weight.full_view()),
            2.0,
        )
        .unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    assert!(device.record(&commands).is_err());
    let after = device.stats().unwrap();
    assert_eq!(after.auxiliary_live_bytes, 0);
    assert_eq!(after.auxiliary_peak_bytes, u64::from(auxiliary));
    assert_eq!(
        after.host_write_bytes,
        before.host_write_bytes + u64::from(auxiliary)
    );
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(after.dispatch_attempts, 0);
    assert!(!after.poisoned);
    assert_eq!(
        &control.lock().unwrap().events[events..],
        &[
            Event::Write(AUXILIARY, 0, len * 4),
            Event::Write(AUXILIARY, (len * 4) as u64, 4),
            Event::Write(AUXILIARY, (len * 4 + 4) as u64, 4)
        ]
    );
    let mut actual = vec![f32::NAN; len];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, vec![-17.0; len]);
    let batch = device.record(&single).unwrap();
    device.wait(&device.submit(&batch).unwrap()).unwrap();
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, vec![3.0; len]);
    drop((commands, single, batch, input, weight, output));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn rmsnorm_physical_failure_retains_resources_and_prior_completion_at_either_stage() {
    for stage in [1, 2] {
        let (device, budget, control) = mock(4, 4096);
        let input = device.allocate_f32(3).unwrap();
        let weight = device.allocate_f32(3).unwrap();
        let output = device.allocate_f32(3).unwrap();
        device
            .write_f32(&input.full_view(), &[1.0, -2.0, 3.0])
            .unwrap();
        device
            .write_f32(&weight.full_view(), &[2.0, -0.5, 1.0])
            .unwrap();
        let mut commands = CommandList::new();
        commands
            .rmsnorm(
                &output.full_view(),
                &input.full_view(),
                Some(&weight.full_view()),
                1e-5,
            )
            .unwrap();
        let batch = device.record(&commands).unwrap();
        let completed = device.submit(&batch).unwrap();
        control.lock().unwrap().fail_dispatch = Some(2 + stage);
        let before = device.stats().unwrap();
        assert!(device.submit(&batch).is_err());
        let after = device.stats().unwrap();
        assert!(after.poisoned);
        assert_eq!(after.dispatch_attempts, 2 + stage);
        assert_eq!(after.dispatch_count, 1 + stage);
        assert_eq!(after.logical_submissions, 1);
        assert_eq!(after.auxiliary_live_bytes, 16);
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        assert_eq!(after.host_write_bytes, before.host_write_bytes);
        device.wait(&completed).unwrap();
        let events = event_count(&control);
        assert!(device.submit(&batch).is_err());
        assert!(device.record(&commands).is_err());
        assert!(device.write_f32(&weight.full_view(), &[1.0; 3]).is_err());
        assert!(device.read_f32(&output.full_view(), &mut [0.0; 3]).is_err());
        assert_eq!(event_count(&control), events);
        drop((commands, batch, completed, input, weight, output));
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 16);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 36);
        drop(device);
        assert!(budget.current_bytes() > 0);
        assert_eq!(control.lock().unwrap().quarantine.len(), 1);
        control.lock().unwrap().quarantine.clear(); // Fake device work is now stopped.
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn rmsnorm_observes_weight_written_by_an_earlier_command_in_the_same_batch() {
    let (device, budget, control) = mock(4, 4096);
    let input = device.allocate_f32(3).unwrap();
    let weight = device.allocate_f32(3).unwrap();
    let output = device.allocate_f32(3).unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(&weight.full_view(), &input.full_view(), None, 0.25)
        .unwrap();
    commands
        .rmsnorm(
            &output.full_view(),
            &input.full_view(),
            Some(&weight.full_view()),
            1e-5,
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    for values in [[1.0, -2.0, 3.0], [-4.0, 0.0, 2.0]] {
        device.write_f32(&input.full_view(), &values).unwrap();
        let expected_weight = rmsnorm_oracle(&values, None, 0.25);
        let expected = rmsnorm_oracle(&values, Some(&expected_weight), 1e-5);
        let events = event_count(&control);
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        assert_eq!(
            &control.lock().unwrap().events[events..],
            &[Event::Dispatch(4), Event::Dispatch(4), Event::Dispatch(5)]
        );
        let mut actual = [f32::NAN; 3];
        device.read_f32(&output.full_view(), &mut actual).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
    }
    drop((commands, batch, input, weight, output, device));
    assert_eq!(budget.current_bytes(), 0);
}

fn rope_oracle(x: &[f32], cos: &[f32], sin: &[f32], layout: RopeLayout, inverse: bool) -> Vec<f32> {
    let mut output = x.to_vec();
    crate::Device::rope(
        &crate::cpu::Cpu::new(),
        &mut output,
        cos,
        sin,
        layout,
        inverse,
    )
    .unwrap();
    output
}

#[test]
fn rope_replays_changed_coordinates_and_coefficients_with_guarded_layouts_and_tails() {
    for pairs in [1, 64, 65, 2048] {
        for layout in [RopeLayout::Interleaved, RopeLayout::SplitHalf] {
            for inverse in [false, true] {
                let (device, budget, control) = mock(4, 64 * 1024);
                let x = device.allocate_f32(pairs * 2 + 2).unwrap();
                let cos = device.allocate_f32(pairs + 2).unwrap();
                let sin = device.allocate_f32(pairs + 2).unwrap();
                let x_view = x.view(1, pairs * 2).unwrap();
                let cos_view = cos.view(1, pairs).unwrap();
                let sin_view = sin.view(1, pairs).unwrap();
                let mut commands = CommandList::new();
                commands
                    .rope(&x_view, &cos_view, &sin_view, layout, inverse)
                    .unwrap();
                let before = device.stats().unwrap();
                let events = event_count(&control);
                let batch = device.record(&commands).unwrap();
                drop(commands);
                assert_eq!(device.stats().unwrap(), before);
                assert_eq!(event_count(&control), events);
                let mut completed = None;
                for round in 0..2 {
                    let mut values = vec![91.0; pairs * 2 + 2];
                    let mut cosines = vec![-92.0; pairs + 2];
                    let mut sines = vec![93.0; pairs + 2];
                    for (index, value) in values[1..pairs * 2 + 1].iter_mut().enumerate() {
                        *value = (index as i32 % 17 - 8) as f32 * 0.25 + round as f32 * 0.125;
                    }
                    for pair in 0..pairs {
                        cosines[pair + 1] =
                            (pair as i32 % 7 - 3) as f32 * 0.25 + round as f32 * 0.5;
                        sines[pair + 1] =
                            (pair as i32 % 11 - 5) as f32 * 0.125 - round as f32 * 0.25;
                    }
                    device.write_f32(&x.full_view(), &values).unwrap();
                    device.write_f32(&cos.full_view(), &cosines).unwrap();
                    device.write_f32(&sin.full_view(), &sines).unwrap();
                    let expected = rope_oracle(
                        &values[1..pairs * 2 + 1],
                        &cosines[1..pairs + 1],
                        &sines[1..pairs + 1],
                        layout,
                        inverse,
                    );
                    let before = device.stats().unwrap();
                    let events = event_count(&control);
                    let submission = device.submit(&batch).unwrap();
                    device.wait(&submission).unwrap();
                    device.wait(&submission).unwrap();
                    assert_eq!(
                        &control.lock().unwrap().events[events..],
                        &[Event::Dispatch(6)]
                    );
                    let after = device.stats().unwrap();
                    assert_eq!(after.host_write_bytes, before.host_write_bytes);
                    assert_eq!(after.host_read_bytes, before.host_read_bytes);
                    assert_eq!(after.auxiliary_live_bytes, 0);
                    assert_eq!(after.dispatch_count, before.dispatch_count + 1);
                    let mut actual = vec![f32::NAN; values.len()];
                    device.read_f32(&x.full_view(), &mut actual).unwrap();
                    assert_eq!(bits(&actual[1..pairs * 2 + 1]), bits(&expected));
                    assert_eq!(actual[0], values[0]);
                    assert_eq!(actual[pairs * 2 + 1], values[pairs * 2 + 1]);
                    actual.resize(pairs + 2, f32::NAN);
                    device.read_f32(&cos.full_view(), &mut actual).unwrap();
                    assert_eq!(bits(&actual), bits(&cosines));
                    device.read_f32(&sin.full_view(), &mut actual).unwrap();
                    assert_eq!(bits(&actual), bits(&sines));
                    completed = Some(submission);
                }
                assert_eq!(
                    control.lock().unwrap().groups,
                    vec![pairs.div_ceil(64) as u32; 2]
                );
                drop((batch, x_view, cos_view, sin_view, x, cos, sin));
                assert_eq!(
                    device.stats().unwrap().tensor_live_bytes,
                    ((pairs * 4 + 6) * 4) as u64
                );
                device.wait(completed.as_ref().unwrap()).unwrap();
                drop(completed);
                assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
                drop(device);
                assert_eq!(budget.current_bytes(), 0);
            }
        }
    }
}

#[test]
fn empty_rope_retains_validated_backing_owners_without_allocating_or_dispatching() {
    let (device, budget, control) = mock(4, 4096);
    let x = device.allocate_f32(2).unwrap();
    let cos = device.allocate_f32(2).unwrap();
    let sin = device.allocate_f32(2).unwrap();
    let mut commands = CommandList::new();
    commands
        .rope(
            &x.view(2, 0).unwrap(),
            &cos.view(1, 0).unwrap(),
            &sin.view(0, 0).unwrap(),
            RopeLayout::SplitHalf,
            true,
        )
        .unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    let batch = device.record(&commands).unwrap();
    assert_eq!(batch.len(), 1);
    assert_eq!(device.stats().unwrap(), before);
    let completed = device.submit(&batch).unwrap();
    drop((commands, batch, x, cos, sin));
    let after = device.stats().unwrap();
    assert_eq!(after.logical_submissions, 1);
    assert_eq!(after.dispatch_attempts, 0);
    assert_eq!(after.auxiliary_live_bytes, 0);
    assert_eq!(after.tensor_live_bytes, 24);
    assert_eq!(event_count(&control), events);
    device.wait(&completed).unwrap();
    drop(completed);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn rope_refuses_invalid_owners_payloads_and_shapes_without_transport_access() {
    let (device, _, control) = mock(4, 4096);
    let (foreign, _, _) = mock(4, 4096);
    let x = device.allocate_f32(4).unwrap();
    let cos = device.allocate_f32(2).unwrap();
    let sin = device.allocate_f32(2).unwrap();
    let foreign_tensor = foreign.allocate_f32(4).unwrap();
    let owner = device.inner.owner;
    let invalid_payload = F32Tensor::new(owner, 4, Arc::new(()));
    let invalid_queue = F32Tensor::new(
        owner,
        4,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&foreign_tensor.payload::<NativeTensor>().unwrap().allocation),
            len: 4,
        }),
    );
    let invalid_extent = F32Tensor::new(
        owner,
        6,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&x.payload::<NativeTensor>().unwrap().allocation),
            len: 6,
        }),
    );
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for invalid in [
        &foreign_tensor,
        &invalid_payload,
        &invalid_queue,
        &invalid_extent,
    ] {
        for role in 0..3 {
            assert!(device
                .plan(&Command::Rope {
                    x: if role == 0 {
                        invalid.full_view()
                    } else {
                        x.full_view()
                    },
                    cos: if role == 1 {
                        invalid.full_view()
                    } else {
                        cos.full_view()
                    },
                    sin: if role == 2 {
                        invalid.full_view()
                    } else {
                        sin.full_view()
                    },
                    layout: RopeLayout::Interleaved,
                    inverse: false,
                })
                .is_err());
        }
    }
    // Empty views must still reject foreign or malformed owners.
    for invalid in [
        &foreign_tensor,
        &invalid_payload,
        &invalid_queue,
        &invalid_extent,
    ] {
        assert!(device
            .plan(&Command::Rope {
                x: invalid.view(0, 0).unwrap(),
                cos: cos.view(0, 0).unwrap(),
                sin: sin.view(0, 0).unwrap(),
                layout: RopeLayout::SplitHalf,
                inverse: true,
            })
            .is_err());
    }
    for (coordinates, cosines, sines) in [(3, 2, 2), (4, 1, 2), (4, 2, 1), (0, 1, 1)] {
        assert!(device
            .plan(&Command::Rope {
                x: x.view(0, coordinates).unwrap(),
                cos: cos.view(0, cosines).unwrap(),
                sin: sin.view(0, sines).unwrap(),
                layout: RopeLayout::Interleaved,
                inverse: false,
            })
            .is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
}

#[test]
fn rope_late_aliases_and_oversized_pairs_refuse_before_lowering_other_commands() {
    let (device, _, control) = mock(4, 64 * 1024);
    let storage = device.allocate_f32(12).unwrap();
    let normalized = device.allocate_f32(4).unwrap();
    let x = storage.view(0, 4).unwrap();
    let cos = storage.view(4, 2).unwrap();
    let sin = storage.view(6, 2).unwrap();
    let hidden_alias = F32Tensor::new(
        device.inner.owner,
        12,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&storage.payload::<NativeTensor>().unwrap().allocation),
            len: 12,
        }),
    );
    let large_x = device.allocate_f32(4098).unwrap();
    let large_cos = device.allocate_f32(2049).unwrap();
    let large_sin = device.allocate_f32(2049).unwrap();
    let cases = [
        (x.clone(), storage.view(2, 2).unwrap(), sin.clone()),
        (x.clone(), cos.clone(), storage.view(3, 2).unwrap()),
        (x.clone(), cos.clone(), storage.view(5, 2).unwrap()),
        (x.clone(), cos.clone(), cos.clone()),
        (x.clone(), cos.clone(), hidden_alias.view(4, 2).unwrap()),
        (
            large_x.full_view(),
            large_cos.full_view(),
            large_sin.full_view(),
        ),
    ];
    assert!(!cos.overlaps(&hidden_alias.view(4, 2).unwrap()));
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for (coordinates, cosines, sines) in cases {
        let mut commands = CommandList::new();
        commands
            .rmsnorm(&normalized.full_view(), &x, None, 1e-5)
            .unwrap();
        commands
            .rope(&coordinates, &cosines, &sines, RopeLayout::SplitHalf, false)
            .unwrap();
        assert!(device.record(&commands).is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
    // Adjacent, nonoverlapping subviews of the same allocation remain legal.
    let mut valid = CommandList::new();
    valid
        .rope(&x, &cos, &sin, RopeLayout::SplitHalf, false)
        .unwrap();
    device.record(&valid).unwrap();
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
}

#[test]
fn rope_observes_coefficient_writes_from_earlier_recorded_commands() {
    let (device, budget, control) = mock(4, 4096);
    let seed = device.allocate_f32(4).unwrap();
    let x = device.allocate_f32(4).unwrap();
    let cos = device.allocate_f32(2).unwrap();
    let sin = device.allocate_f32(2).unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(&cos.full_view(), &seed.view(0, 2).unwrap(), None, 0.25)
        .unwrap();
    commands
        .rmsnorm(&sin.full_view(), &seed.view(2, 2).unwrap(), None, 0.5)
        .unwrap();
    commands
        .rope(
            &x.full_view(),
            &cos.full_view(),
            &sin.full_view(),
            RopeLayout::SplitHalf,
            true,
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    for values in [[1.0, -2.0, 3.0, -4.0], [-2.0, 1.0, 0.0, 5.0]] {
        device.write_f32(&seed.full_view(), &values).unwrap();
        device.write_f32(&x.full_view(), &values).unwrap();
        let expected_cos = rmsnorm_oracle(&values[..2], None, 0.25);
        let expected_sin = rmsnorm_oracle(&values[2..], None, 0.5);
        let expected = rope_oracle(
            &values,
            &expected_cos,
            &expected_sin,
            RopeLayout::SplitHalf,
            true,
        );
        let events = event_count(&control);
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        assert_eq!(
            &control.lock().unwrap().events[events..],
            &[Event::Dispatch(4), Event::Dispatch(4), Event::Dispatch(6)]
        );
        let mut actual = [f32::NAN; 4];
        device.read_f32(&x.full_view(), &mut actual).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
    }
    drop((commands, batch, seed, x, cos, sin));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn rope_partial_batch_failure_poisons_and_retains_all_coefficient_owners() {
    let (device, budget, control) = mock(4, 4096);
    let x = device.upload_f32(&[1.0, 2.0, -3.0, -4.0]).unwrap();
    let cos = device.upload_f32(&[0.0, 0.5]).unwrap();
    let sin = device.upload_f32(&[1.0, -0.5]).unwrap();
    let mut commands = CommandList::new();
    for inverse in [false, true] {
        commands
            .rope(
                &x.full_view(),
                &cos.full_view(),
                &sin.full_view(),
                RopeLayout::Interleaved,
                inverse,
            )
            .unwrap();
    }
    let batch = device.record(&commands).unwrap();
    let completed = device.submit(&batch).unwrap();
    control.lock().unwrap().fail_dispatch = Some(4);
    let before = device.stats().unwrap();
    assert!(device.submit(&batch).is_err());
    let after = device.stats().unwrap();
    assert!(after.poisoned);
    assert_eq!(after.dispatch_attempts, 4);
    assert_eq!(after.dispatch_count, 3);
    assert_eq!(after.logical_submissions, 1);
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(after.host_write_bytes, before.host_write_bytes);
    assert_eq!(after.auxiliary_live_bytes, 0);
    device.wait(&completed).unwrap();
    let events = event_count(&control);
    assert!(device.submit(&batch).is_err());
    assert!(device.record(&commands).is_err());
    assert!(device.write_f32(&cos.full_view(), &[1.0; 2]).is_err());
    assert!(device.read_f32(&x.full_view(), &mut [0.0; 4]).is_err());
    assert_eq!(event_count(&control), events);
    drop((commands, batch, completed, x, cos, sin));
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 32);
    drop(device);
    assert!(budget.current_bytes() > 0);
    assert_eq!(control.lock().unwrap().quarantine.len(), 1);
    control.lock().unwrap().quarantine.clear(); // Fake device work is now stopped.
    assert_eq!(budget.current_bytes(), 0);
}

fn attention_config(
    queries: usize,
    start: usize,
    cached: usize,
    heads: usize,
    kv_heads: usize,
    dim: usize,
) -> IncrementalAttentionConfig {
    IncrementalAttentionConfig {
        n_query_positions: queries,
        query_start: start,
        n_kv_positions: cached,
        n_heads: heads,
        n_kv_heads: kv_heads,
        head_dim: dim,
        scale: (dim as f32).sqrt().recip(),
    }
}

fn attention_list(
    output: &F32TensorView,
    q: &F32TensorView,
    k: &F32TensorView,
    v: &F32TensorView,
    config: IncrementalAttentionConfig,
) -> CommandList {
    let mut commands = CommandList::new();
    commands
        .incremental_causal_attention(output, q, k, v, config)
        .unwrap();
    commands
}

fn attention_oracle(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: IncrementalAttentionConfig,
) -> Vec<f32> {
    let mut output = vec![f32::NAN; q.len()];
    crate::Device::incremental_causal_attention(
        &crate::cpu::Cpu::new(),
        &mut output,
        q,
        k,
        v,
        config,
    )
    .unwrap();
    output
}

#[test]
fn attention_prefill_suffix_gqa_and_boundary_shapes_replay_without_future_reads() {
    for config in [
        attention_config(3, 0, 3, 2, 1, 4),
        attention_config(1, 2, 4, 2, 1, 4),
        attention_config(1, 64, 66, 2, 1, 65),
        attention_config(1, 7, 8, 16, 8, 128),
        attention_config(1, 255, 256, 1, 1, 256),
        attention_config(1, 254, 255, 2, 1, 17),
        attention_config(1, 256, 257, 2, 1, 17),
        attention_config(1, 340, 341, 2, 1, 17),
        attention_config(1, 341, 342, 2, 1, 17),
        attention_config(1, 510, 511, 2, 1, 17),
        attention_config(2, 510, 512, 2, 1, 17),
        attention_config(1, 511, 512, 1, 1, 256),
        attention_config(1, 511, 2048, 2, 1, 17),
        attention_config(1, 512, 2048, 2, 1, 17),
        attention_config(1, 1022, 2048, 2, 1, 17),
        attention_config(1, 1023, 2048, 2, 1, 17),
        attention_config(1, 1024, 2048, 2, 1, 17),
        attention_config(1, 2046, 2048, 2, 1, 17),
        attention_config(1, 2047, 2048, 2, 1, 17),
        // Single-query suffixes exercise larger contexts with O(C) work and
        // small head widths; full-context prefill would make this quadratic.
        attention_config(1, 8190, 8192, 2, 1, 3),
        attention_config(1, 8191, 8192, 2, 1, 3),
        attention_config(1, 16382, 16384, 2, 1, 3),
        attention_config(1, 16383, 16384, 2, 1, 3),
    ] {
        let (device, budget, control) = mock_config(NativeExecutionConfig {
            weight_capacity_bytes: 4,
            tensor_capacity_bytes: 2 * 1024 * 1024,
            auxiliary_capacity_bytes: ((config.query_start + config.n_query_positions) * 12 + 8)
                .max(32 * 1024) as u32,
        });
        let query_len = config.n_query_positions * config.n_heads * config.head_dim;
        let stride = config.n_kv_heads * config.head_dim;
        let cache_len = config.n_kv_positions * stride;
        let q = device.allocate_f32(query_len + 2).unwrap();
        let k = device.allocate_f32(cache_len + 2).unwrap();
        let v = device.allocate_f32(cache_len + 2).unwrap();
        let output = device.allocate_f32(query_len + 2).unwrap();
        let q_view = q.view(1, query_len).unwrap();
        let k_view = k.view(1, cache_len).unwrap();
        let v_view = v.view(1, cache_len).unwrap();
        let output_view = output.view(1, query_len).unwrap();
        let batch = device
            .record(&attention_list(
                &output_view,
                &q_view,
                &k_view,
                &v_view,
                config,
            ))
            .unwrap();
        let query_end = config.query_start + config.n_query_positions;
        assert_eq!(
            device.stats().unwrap().auxiliary_live_bytes,
            (8 + 12 * query_end) as u64
        );
        let native = batch.payload::<NativeBatch>().unwrap();
        let (_, key_region) = device.tensor_region(&k_view, "key fixture").unwrap();
        let (_, value_region) = device.tensor_region(&v_view, "value fixture").unwrap();
        assert_eq!(
            native.commands.len(),
            config.n_query_positions * config.n_heads * 3
        );
        for query in 0..config.n_query_positions {
            let visible = config.query_start + query + 1;
            let cache_bytes = ((visible - 1) * stride + config.head_dim) * 4;
            for head in 0..config.n_heads {
                let first = (query * config.n_heads + head) * 3;
                let score = &native.commands[first];
                let softmax = &native.commands[first + 1];
                let value = &native.commands[first + 2];
                let head_offset =
                    (head / (config.n_heads / config.n_kv_heads)) * config.head_dim * 4;
                assert_eq!(score.groups, visible.div_ceil(64) as u32);
                assert_eq!(softmax.groups, 1);
                assert_eq!(value.groups, config.head_dim.div_ceil(64) as u32);
                assert_eq!(
                    score.bindings[1].offset,
                    key_region.offset + head_offset as u64
                );
                assert_eq!(
                    value.bindings[1].offset,
                    value_region.offset + head_offset as u64
                );
                assert_eq!(score.bindings[1].length as usize, cache_bytes);
                assert_eq!(value.bindings[1].length as usize, cache_bytes);
                assert_eq!(score.bindings[2], softmax.bindings[1]);
                assert_eq!(softmax.bindings[2], value.bindings[0]);
                assert_eq!(score.bindings[2].length as usize, visible * 8);
                assert_eq!(softmax.bindings[2].length as usize, visible * 4);
                assert_eq!(score.bindings[2].offset % 8, 0);
                assert!(!overlaps(score.bindings[2], softmax.bindings[2]));
            }
        }
        let mut completed = None;
        for round in 0..2 {
            let mut queries = vec![91.0; query_len + 2];
            let mut keys = vec![-92.0; cache_len + 2];
            let mut values = vec![93.0; cache_len + 2];
            for index in 0..query_len {
                queries[index + 1] = (index as i32 % 13 - 6) as f32 * 0.125 + round as f32 * 0.0625;
            }
            for index in 0..cache_len {
                keys[index + 1] = (index as i32 % 17 - 8) as f32 * 0.0625 - round as f32 * 0.125;
                values[index + 1] = (index as i32 % 23 - 11) as f32 * 0.25 + round as f32 * 0.5;
                if index / stride >= query_end {
                    keys[index + 1] = f32::NAN;
                    values[index + 1] = f32::NAN;
                }
            }
            device.write_f32(&q.full_view(), &queries).unwrap();
            device.write_f32(&k.full_view(), &keys).unwrap();
            device.write_f32(&v.full_view(), &values).unwrap();
            device
                .write_f32(&output.full_view(), &vec![94.0; query_len + 2])
                .unwrap();
            let expected = attention_oracle(
                &queries[1..query_len + 1],
                &keys[1..cache_len + 1],
                &values[1..cache_len + 1],
                config,
            );
            let before = device.stats().unwrap();
            let events = event_count(&control);
            let submission = device.submit(&batch).unwrap();
            device.wait(&submission).unwrap();
            device.wait(&submission).unwrap();
            let operations = config.n_query_positions * config.n_heads;
            assert_eq!(
                &control.lock().unwrap().events[events..],
                &[Event::Dispatch(7), Event::Dispatch(8), Event::Dispatch(9)].repeat(operations)
            );
            let after = device.stats().unwrap();
            assert_eq!(
                after.dispatch_count,
                before.dispatch_count + (operations * 3) as u64
            );
            assert_eq!(after.host_read_bytes, before.host_read_bytes);
            assert_eq!(after.host_write_bytes, before.host_write_bytes);
            assert_eq!(after.auxiliary_peak_bytes, before.auxiliary_peak_bytes);
            let mut actual = vec![f32::NAN; query_len + 2];
            device.read_f32(&output.full_view(), &mut actual).unwrap();
            assert_eq!(actual[0], 94.0);
            assert_eq!(actual[query_len + 1], 94.0);
            assert_eq!(bits(&actual[1..query_len + 1]), bits(&expected));
            device.read_f32(&q.full_view(), &mut actual).unwrap();
            assert_eq!(bits(&actual), bits(&queries));
            actual.resize(cache_len + 2, f32::NAN);
            device.read_f32(&k.full_view(), &mut actual).unwrap();
            assert_eq!(bits(&actual), bits(&keys));
            device.read_f32(&v.full_view(), &mut actual).unwrap();
            assert_eq!(bits(&actual), bits(&values));
            completed = Some(submission);
        }
        drop((batch, q_view, k_view, v_view, output_view, q, k, v, output));
        assert_eq!(
            device.stats().unwrap().tensor_live_bytes,
            ((query_len * 2 + cache_len * 2 + 8) * 4) as u64
        );
        drop(completed);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
        drop(device);
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn attention_observes_prior_cache_tail_writes_without_mutating_the_prefix() {
    let (device, budget, control) = mock(4, 4096);
    let config = attention_config(1, 2, 3, 2, 1, 2);
    let q = device.upload_f32(&[1.0, -2.0, 0.5, 3.0]).unwrap();
    let k = device
        .upload_f32(&[0.5, 1.0, -1.0, 0.25, 0.0, 0.0])
        .unwrap();
    let v = device
        .upload_f32(&[-2.0, 1.0, 3.0, -0.5, 0.0, 0.0])
        .unwrap();
    let output = device.allocate_f32(4).unwrap();
    let seed = device.allocate_f32(4).unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(
            &k.view(4, 2).unwrap(),
            &seed.view(0, 2).unwrap(),
            None,
            0.25,
        )
        .unwrap();
    commands
        .rmsnorm(&v.view(4, 2).unwrap(), &seed.view(2, 2).unwrap(), None, 0.5)
        .unwrap();
    commands
        .incremental_causal_attention(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            config,
        )
        .unwrap();
    let batch = device.record(&commands).unwrap();
    for values in [[1.0, -2.0, 3.0, 4.0], [-3.0, 0.5, 0.0, 2.0]] {
        device.write_f32(&seed.full_view(), &values).unwrap();
        let mut keys = vec![0.5, 1.0, -1.0, 0.25];
        keys.extend(rmsnorm_oracle(&values[..2], None, 0.25));
        let mut cache_values = vec![-2.0, 1.0, 3.0, -0.5];
        cache_values.extend(rmsnorm_oracle(&values[2..], None, 0.5));
        let expected = attention_oracle(&[1.0, -2.0, 0.5, 3.0], &keys, &cache_values, config);
        let events = event_count(&control);
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        let mut expected_events = vec![Event::Dispatch(4), Event::Dispatch(4)];
        expected_events
            .extend([Event::Dispatch(7), Event::Dispatch(8), Event::Dispatch(9)].repeat(2));
        assert_eq!(&control.lock().unwrap().events[events..], &expected_events);
        let mut actual = [f32::NAN; 4];
        device.read_f32(&output.full_view(), &mut actual).unwrap();
        assert_eq!(bits(&actual), bits(&expected));
        device
            .read_f32(&k.view(0, 4).unwrap(), &mut actual)
            .unwrap();
        assert_eq!(actual, [0.5, 1.0, -1.0, 0.25]);
        device
            .read_f32(&v.view(0, 4).unwrap(), &mut actual)
            .unwrap();
        assert_eq!(actual, [-2.0, 1.0, 3.0, -0.5]);
    }
    drop((commands, batch, q, k, v, output, seed, device));
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn attention_late_physical_alias_refusal_is_transactional_but_separate_read_aliases_work() {
    let (device, budget, control) = mock(4, 4096);
    let config = attention_config(1, 0, 1, 1, 1, 2);
    let output = device.allocate_f32(2).unwrap();
    let q = device.upload_f32(&[2.0, -1.0]).unwrap();
    let k = device.upload_f32(&[0.5, 3.0]).unwrap();
    let v = device.upload_f32(&[-3.0, 4.0]).unwrap();
    let prefix = device.allocate_f32(2).unwrap();
    let alias = F32Tensor::new(
        device.inner.owner,
        2,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&q.payload::<NativeTensor>().unwrap().allocation),
            len: 2,
        }),
    );
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for (out, query, key, value) in [
        (q.full_view(), q.full_view(), k.full_view(), v.full_view()),
        (k.full_view(), q.full_view(), k.full_view(), v.full_view()),
        (v.full_view(), q.full_view(), k.full_view(), v.full_view()),
        (
            output.full_view(),
            q.full_view(),
            alias.full_view(),
            v.full_view(),
        ),
    ] {
        let mut commands = CommandList::new();
        commands
            .rmsnorm(&prefix.full_view(), &q.full_view(), None, 0.25)
            .unwrap();
        commands
            .incremental_causal_attention(&out, &query, &key, &value, config)
            .unwrap();
        assert!(device.record(&commands).is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
    for (value, expected) in [(&k, [0.5, 3.0]), (&q, [2.0, -1.0])] {
        let batch = device
            .record(&attention_list(
                &output.full_view(),
                &q.full_view(),
                &k.full_view(),
                &value.full_view(),
                config,
            ))
            .unwrap();
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        let mut actual = [f32::NAN; 2];
        device.read_f32(&output.full_view(), &mut actual).unwrap();
        assert_eq!(actual, expected);
    }
    drop((alias, output, q, k, v, prefix, device));
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn attention_validates_all_owners_payloads_and_native_limits_before_lowering() {
    let (device, _, control) = mock(4, 4096);
    let (foreign, _, _) = mock(4, 4096);
    let output = device.allocate_f32(2).unwrap();
    let q = device.allocate_f32(2).unwrap();
    let k = device.allocate_f32(2).unwrap();
    let v = device.allocate_f32(2).unwrap();
    let other = foreign.allocate_f32(2).unwrap();
    let wrong_payload = F32Tensor::new(device.inner.owner, 2, Arc::new(()));
    let wrong_queue = F32Tensor::new(
        device.inner.owner,
        2,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&other.payload::<NativeTensor>().unwrap().allocation),
            len: 2,
        }),
    );
    let wrong_extent = F32Tensor::new(
        device.inner.owner,
        3,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&q.payload::<NativeTensor>().unwrap().allocation),
            len: 3,
        }),
    );
    let before = device.stats().unwrap();
    let events = event_count(&control);
    for invalid in [&other, &wrong_payload, &wrong_queue, &wrong_extent] {
        for role in 0..4 {
            assert!(device
                .plan(&Command::IncrementalCausalAttention {
                    output: if role == 0 {
                        invalid.full_view()
                    } else {
                        output.full_view()
                    },
                    q: if role == 1 {
                        invalid.full_view()
                    } else {
                        q.full_view()
                    },
                    k_cache: if role == 2 {
                        invalid.full_view()
                    } else {
                        k.full_view()
                    },
                    v_cache: if role == 3 {
                        invalid.full_view()
                    } else {
                        v.full_view()
                    },
                    config: attention_config(1, 0, 1, 1, 1, 2),
                })
                .is_err());
        }
    }
    for config in [
        IncrementalAttentionConfig {
            scale: f32::NAN,
            ..attention_config(1, 0, 1, 1, 1, 2)
        },
        IncrementalAttentionConfig {
            scale: 0.0,
            ..attention_config(1, 0, 1, 1, 1, 2)
        },
        attention_config(1, usize::MAX, 1, 1, 1, 2),
        attention_config(1, 1, 1, 1, 1, 2),
        attention_config(1, 0, 1, 2, 3, 2),
    ] {
        assert!(device
            .plan(&Command::IncrementalCausalAttention {
                output: output.full_view(),
                q: q.full_view(),
                k_cache: k.full_view(),
                v_cache: v.full_view(),
                config
            })
            .is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&control), events);
    for config in [
        attention_config(1, 0, 1, 1, 1, 257),
        attention_config(257, 0, 257, 1, 1, 1),
        attention_config(1, 0, 16385, 1, 1, 1),
        attention_config(1, 0, 1, 33, 1, 1),
        attention_config(1, 0, 1, 33, 33, 1),
        attention_config(1, 0, 1, 17, 17, 256),
    ] {
        let (device, _, control) = mock(4, 256 * 1024);
        let query_len = config.n_query_positions * config.n_heads * config.head_dim;
        let cache_len = config.n_kv_positions * config.n_kv_heads * config.head_dim;
        let output = device.allocate_f32(query_len).unwrap();
        let q = device.allocate_f32(query_len).unwrap();
        let k = device.allocate_f32(cache_len).unwrap();
        let v = device.allocate_f32(cache_len).unwrap();
        let commands = attention_list(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            config,
        );
        let before = device.stats().unwrap();
        let events = event_count(&control);
        assert!(device.record(&commands).is_err());
        assert_eq!(device.stats().unwrap(), before);
        assert_eq!(event_count(&control), events);
    }
}

#[test]
fn attention_later_scalar_refusal_reclaims_shared_workspace_and_parameter() {
    let (device, budget, control) = mock_config(NativeExecutionConfig {
        weight_capacity_bytes: 4,
        tensor_capacity_bytes: 4096,
        auxiliary_capacity_bytes: 44,
    });
    let config = attention_config(1, 2, 3, 1, 1, 2);
    let q = device.upload_f32(&[1.0, -2.0]).unwrap();
    let k = device
        .upload_f32(&[0.5, 1.0, 2.0, -1.0, -0.5, 0.0])
        .unwrap();
    let v = device
        .upload_f32(&[1.0, -1.0, 3.0, 4.0, -2.0, 0.5])
        .unwrap();
    let output = device.upload_f32(&[91.0, 92.0]).unwrap();
    let single = attention_list(
        &output.full_view(),
        &q.full_view(),
        &k.full_view(),
        &v.full_view(),
        config,
    );
    let mut twice = single.clone();
    twice
        .incremental_causal_attention(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            IncrementalAttentionConfig {
                scale: 0.5,
                ..config
            },
        )
        .unwrap();
    let before = device.stats().unwrap();
    let events = event_count(&control);
    assert!(device.record(&twice).is_err());
    let after = device.stats().unwrap();
    assert!(!after.poisoned);
    assert_eq!(after.auxiliary_live_bytes, 0);
    assert_eq!(after.auxiliary_peak_bytes, 44);
    assert_eq!(after.host_write_bytes, before.host_write_bytes + 44);
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(after.dispatch_attempts, 0);
    assert_eq!(
        &control.lock().unwrap().events[events..],
        &[
            Event::Write(AUXILIARY, 0, 40),
            Event::Write(AUXILIARY, 40, 4),
        ]
    );
    let mut actual = [f32::NAN; 2];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [91.0, 92.0]);
    let batch = device.record(&single).unwrap();
    device.wait(&device.submit(&batch).unwrap()).unwrap();
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(
        bits(&actual),
        bits(&attention_oracle(
            &[1.0, -2.0],
            &[0.5, 1.0, 2.0, -1.0, -0.5, 0.0],
            &[1.0, -1.0, 3.0, 4.0, -2.0, 0.5],
            config
        ))
    );
    drop((single, twice, batch, output, q, k, v));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn attention_failure_at_each_physical_stage_retains_scratch_and_old_completion() {
    for stage in 1..=3 {
        let (device, budget, control) = mock(4, 4096);
        let q = device.upload_f32(&[1.0, -2.0]).unwrap();
        let k = device.upload_f32(&[0.5, 1.0]).unwrap();
        let v = device.upload_f32(&[-1.0, 3.0]).unwrap();
        let output = device.allocate_f32(2).unwrap();
        let commands = attention_list(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            attention_config(1, 0, 1, 1, 1, 2),
        );
        let batch = device.record(&commands).unwrap();
        let completed = device.submit(&batch).unwrap();
        control.lock().unwrap().fail_dispatch = Some(3 + stage);
        let before = device.stats().unwrap();
        assert!(device.submit(&batch).is_err());
        let after = device.stats().unwrap();
        assert!(after.poisoned);
        assert_eq!(after.dispatch_attempts, 3 + stage);
        assert_eq!(after.dispatch_count, 2 + stage);
        assert_eq!(after.logical_submissions, 1);
        assert_eq!(after.host_write_bytes, before.host_write_bytes);
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        device.wait(&completed).unwrap();
        let events = event_count(&control);
        assert!(device.submit(&batch).is_err());
        assert!(device.record(&commands).is_err());
        assert!(device.write_f32(&k.full_view(), &[0.0; 2]).is_err());
        assert!(device.read_f32(&output.full_view(), &mut [0.0; 2]).is_err());
        assert_eq!(event_count(&control), events);
        drop((commands, batch, completed, output, q, k, v));
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 20);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 32);
        drop(device);
        assert!(budget.current_bytes() > 0);
        assert_eq!(control.lock().unwrap().quarantine.len(), 1);
        control.lock().unwrap().quarantine.clear(); // Fake device work is now stopped.
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn attention_f64_descriptors_stay_aligned_after_an_odd_auxiliary_word() {
    let (device, budget, _) = mock(4, 4096);
    let prefix = device.allocate(AUXILIARY, &[0; 4]).unwrap();
    let q = device.upload_f32(&[1.0]).unwrap();
    let k = device.upload_f32(&[2.0]).unwrap();
    let v = device.upload_f32(&[3.0]).unwrap();
    let output = device.allocate_f32(1).unwrap();
    let batch = device
        .record(&attention_list(
            &output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            attention_config(1, 0, 1, 1, 1, 1),
        ))
        .unwrap();
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 24);
    let native = batch.payload::<NativeBatch>().unwrap();
    assert_eq!(native.commands[0].bindings[2].offset, 8);
    assert_eq!(native.commands[1].bindings[1].offset, 8);
    device.wait(&device.submit(&batch).unwrap()).unwrap();
    let mut actual = [f32::NAN];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [3.0]);
    drop((prefix, batch, q, k, v, output));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

fn embedding_weight(
    device: &NativeExecutionDevice,
    rows: usize,
    cols: usize,
) -> (PreparedWeight, Vec<f32>) {
    let mut bytes = Vec::new();
    let mut expected = Vec::new();
    for row in 0..rows {
        for block in 0..cols / 32 {
            bytes.extend(0x3800u16.to_le_bytes());
            for lane in 0..32 {
                let value = ((row * 17 + block * 3 + lane) % 255) as i16 - 127;
                bytes.push(value as i8 as u8);
                expected.push(value as f32 * 0.5);
            }
        }
    }
    let encoded = device
        .upload_weight(&Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows,
            cols,
        })
        .unwrap();
    (device.prepare_weight(&encoded).unwrap(), expected)
}

#[test]
fn weight_and_tensor_capacity_limits_expand_without_allocating_them() {
    assert_eq!(MAX_WEIGHT_ARENA_BYTES, 1024 * 1024 * 1024);
    assert_eq!(MAX_TENSOR_ARENA_BYTES, u32::MAX - 4095);
    assert_eq!(MAX_ARENA_BYTES, 64 * 1024 * 1024);
    assert_eq!(
        NativeExecutionConfig::default().tensor_capacity_bytes,
        64 * 1024
    );
    assert_eq!(
        NativeExecutionConfig::default().weight_capacity_bytes,
        64 * 1024
    );
    assert_eq!(
        NativeExecutionConfig::default().auxiliary_capacity_bytes,
        4096
    );
    // Cost/admission only: never allocate these large mock arenas.
    for tensors in [
        128 * 1024 * 1024,
        128 * 1024 * 1024 + 1,
        256 * 1024 * 1024,
        256 * 1024 * 1024 + 1,
        512 * 1024 * 1024 + 1,
        1 << 31,
        MAX_TENSOR_ARENA_BYTES - 1,
        MAX_TENSOR_ARENA_BYTES,
    ] {
        let config = NativeExecutionConfig {
            weight_capacity_bytes: 1,
            tensor_capacity_bytes: tensors,
            auxiliary_capacity_bytes: 1,
        };
        assert_eq!(
            config.allocation_bytes().unwrap(),
            u64::from(tensors).div_ceil(4096) * 4096 + 45_056
        );
        assert!(config.tensor_capacity_bytes <= MAX_SLOT_BYTES);
    }
    for weights in [
        MAX_ARENA_BYTES + 1,
        165_306_368,
        633_233_408,
        MAX_WEIGHT_ARENA_BYTES,
    ] {
        let config = NativeExecutionConfig {
            weight_capacity_bytes: weights,
            tensor_capacity_bytes: MAX_ARENA_BYTES,
            auxiliary_capacity_bytes: MAX_ARENA_BYTES,
        };
        config.validate().unwrap();
        let bytes = config.allocation_bytes().unwrap();
        assert_eq!(
            bytes,
            u64::from(weights).div_ceil(4096) * 4096 + 128 * 1024 * 1024 + 36_864
        );
        assert!(bytes > u64::from(weights) + 128 * 1024 * 1024);
    }
    for config in [
        NativeExecutionConfig {
            weight_capacity_bytes: MAX_WEIGHT_ARENA_BYTES + 1,
            ..Default::default()
        },
        NativeExecutionConfig {
            tensor_capacity_bytes: MAX_TENSOR_ARENA_BYTES + 1,
            ..Default::default()
        },
        NativeExecutionConfig {
            auxiliary_capacity_bytes: MAX_ARENA_BYTES + 1,
            ..Default::default()
        },
    ] {
        assert!(config.allocation_bytes().is_err());
    }
}

#[test]
fn maximum_tensor_arena_metadata_reserves_and_coalesces_without_payload_allocation() {
    let mut arena = Arena::new(MAX_TENSOR_ARENA_BYTES);
    let first = arena.reserve(1_usize << 31).unwrap();
    let tail = arena
        .reserve((MAX_TENSOR_ARENA_BYTES - first.bytes) as usize)
        .unwrap();
    assert_eq!(
        first,
        Span {
            offset: 0,
            bytes: 1 << 31
        }
    );
    assert_eq!(tail.offset, 1 << 31);
    assert_eq!(
        u64::from(tail.offset) + u64::from(tail.bytes),
        u64::from(MAX_TENSOR_ARENA_BYTES)
    );
    assert_eq!(arena.live, u64::from(MAX_TENSOR_ARENA_BYTES));
    assert!(arena.reserve(1).is_none());
    assert!(arena.reserve(u32::MAX as usize).is_none());
    arena.release(first);
    arena.release(tail);
    assert_eq!(arena.live, 0);
    assert_eq!(arena.peak, u64::from(MAX_TENSOR_ARENA_BYTES));
    assert_eq!(
        arena.free,
        [Span {
            offset: 0,
            bytes: MAX_TENSOR_ARENA_BYTES
        }]
    );
    // Four-byte rounding can reach the ceiling, but cannot wrap at u32::MAX.
    assert_eq!(
        arena
            .reserve(MAX_TENSOR_ARENA_BYTES as usize - 1)
            .unwrap()
            .bytes,
        MAX_TENSOR_ARENA_BYTES
    );
    assert!(arena.reserve(4).is_none());
}

#[test]
fn attention_library_uses_derived_group_abi_without_widening_vector_kernels() {
    assert_eq!(NATIVE_KERNEL_ABI.max_groups(), 256);
    assert_eq!(NATIVE_KERNEL_ABI.vgprs(), GFX1151_SERIAL_ABI.vgprs());
    assert_eq!(
        NATIVE_KERNEL_ABI.program_resources,
        GFX1151_SERIAL_ABI.program_resources
    );
    assert_eq!(
        NATIVE_KERNEL_ABI.raw_descriptor,
        GFX1151_SERIAL_ABI.raw_descriptor
    );
    assert_eq!(NATIVE_KERNEL_ABI.threads, GFX1151_SERIAL_ABI.threads);
    assert_eq!(GFX1151_SERIAL_ABI.max_groups(), 64);
    assert_eq!(kernels::attention_dispatch_groups(8192), Ok(128));
    assert_eq!(kernels::attention_dispatch_groups(16384), Ok(256));
    for count in [4097, 8192, 16384] {
        assert!(kernels::dispatch_groups(count).is_err());
    }
}

#[test]
fn embedding_rows_use_exact_offsets_and_guards_and_retain_owned_weights_on_replay() {
    for cols in [32, 96, 1024, 4096] {
        let (device, budget, control) = mock(32 * 1024, 32 * 1024);
        let prefix = prepared_bf16(&device, 1, 1); // Keep a nonzero packed allocation base.
        let (weight, expected) = embedding_weight(&device, 3, cols);
        let output = device.allocate_f32(cols + 2).unwrap();
        let mut batches = Vec::new();
        for row in 0..3 {
            let mut commands = CommandList::new();
            commands
                .embedding_row(&output.view(1, cols).unwrap(), &weight, row)
                .unwrap();
            let before = device.stats().unwrap();
            let batch = device.record(&commands).unwrap();
            let after = device.stats().unwrap();
            assert_eq!(after.auxiliary_live_bytes, before.auxiliary_live_bytes + 4);
            assert_eq!(after.host_write_bytes, before.host_write_bytes + 4);
            let native = batch.payload::<NativeBatch>().unwrap();
            assert_eq!(native.commands.len(), 1);
            assert_eq!(
                native.commands[0].bindings[0],
                BufferRegion {
                    slot: WEIGHTS,
                    offset: 4 + (row * cols / 32 * 34) as u64,
                    length: (cols / 32 * 34) as u32
                }
            );
            if cols == 96 && row == 1 {
                assert_eq!(native.commands[0].bindings[0].offset % 4, 2);
            }
            batches.push(batch);
        }
        drop((prefix, weight));
        assert_eq!(
            device.stats().unwrap().weight_live_bytes,
            (3 * cols / 32 * 34).next_multiple_of(4) as u64
        );
        let mut completion = None;
        for repeat in 0..2 {
            for (row, batch) in batches.iter().enumerate() {
                let guard = 97.0 + repeat as f32;
                device
                    .write_f32(&output.full_view(), &vec![guard; cols + 2])
                    .unwrap();
                let events = event_count(&control);
                let before = device.stats().unwrap();
                let submitted = device.submit(batch).unwrap();
                device.wait(&submitted).unwrap();
                assert_eq!(
                    &control.lock().unwrap().events[events..],
                    &[Event::Dispatch(10)]
                );
                let after = device.stats().unwrap();
                assert_eq!(after.host_write_bytes, before.host_write_bytes);
                assert_eq!(after.host_read_bytes, before.host_read_bytes);
                let mut actual = vec![f32::NAN; cols + 2];
                device.read_f32(&output.full_view(), &mut actual).unwrap();
                assert_eq!(actual[0], guard);
                assert_eq!(actual[cols + 1], guard);
                assert_eq!(
                    bits(&actual[1..cols + 1]),
                    bits(&expected[row * cols..(row + 1) * cols])
                );
                completion = Some(submitted);
            }
        }
        drop((output, batches));
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 4);
        assert_eq!(
            device.stats().unwrap().tensor_live_bytes,
            ((cols + 2) * 4) as u64
        );
        drop(completion);
        assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
        drop(device);
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn embedding_record_refuses_late_bad_rows_formats_and_forged_handles_before_allocation() {
    let (device, budget, control) = mock(4096, 4096);
    let (weight, _) = embedding_weight(&device, 3, 96);
    let output = device.upload_f32(&[93.0; 96]).unwrap();
    let bf16 = prepared_bf16(&device, 3, 96);
    let bad_payload =
        PreparedWeight::new(device.inner.owner, GgmlType(8), 3, 96, 306, Arc::new(()));
    let packed = weight.payload::<NativeWeight>().unwrap();
    let bad_extent = PreparedWeight::new(
        device.inner.owner,
        GgmlType(8),
        3,
        96,
        305,
        Arc::new(packed.clone()),
    );
    let forged_output = F32Tensor::new(device.inner.owner, 96, Arc::new(()));
    let wrong_extent = F32Tensor::new(
        device.inner.owner,
        97,
        Arc::new(NativeTensor {
            allocation: Arc::clone(&output.payload::<NativeTensor>().unwrap().allocation),
            len: 96,
        }),
    );
    let (foreign, foreign_budget, _) = mock(4096, 4096);
    let (foreign_weight, _) = embedding_weight(&foreign, 3, 96);
    let foreign_output = foreign.allocate_f32(96).unwrap();
    for (view, prepared, row) in [
        (output.full_view(), weight.clone(), 3),
        (output.full_view(), weight.clone(), usize::MAX),
        (output.view(0, 95).unwrap(), weight.clone(), 0),
        (output.full_view(), bf16, 0),
        (output.full_view(), bad_payload, 0),
        (output.full_view(), bad_extent, 0),
        (forged_output.full_view(), weight.clone(), 0),
        (wrong_extent.view(0, 96).unwrap(), weight.clone(), 0),
        (foreign_output.full_view(), weight.clone(), 0),
        (output.full_view(), foreign_weight, 0),
    ] {
        let mut commands = CommandList::new();
        commands
            .embedding_row(&output.full_view(), &weight, 0)
            .unwrap();
        let before = device.stats().unwrap();
        let events = event_count(&control);
        if commands.embedding_row(&view, &prepared, row).is_ok() {
            assert!(device.record(&commands).is_err());
        } else {
            // Internal malformed commands must independently refuse during planning.
            assert!(device
                .plan(&Command::EmbeddingRow {
                    output: view,
                    weight: prepared,
                    row
                })
                .is_err());
        }
        assert_eq!(device.stats().unwrap(), before);
        assert_eq!(event_count(&control), events);
    }
    let mut actual = [0.0; 96];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [93.0; 96]);
    drop((foreign_output, foreign));
    assert_eq!(foreign_budget.current_bytes(), 0);
    drop((forged_output, wrong_extent, weight, output, device));
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn embedding_auxiliary_exhaustion_rolls_back_recording_and_later_lookup_succeeds() {
    let (device, budget, control) = mock_config(NativeExecutionConfig {
        weight_capacity_bytes: 4096,
        tensor_capacity_bytes: 4096,
        auxiliary_capacity_bytes: 4,
    });
    let (weight, expected) = embedding_weight(&device, 2, 32);
    let output = device.upload_f32(&[91.0; 32]).unwrap();
    let mut commands = CommandList::new();
    commands
        .embedding_row(&output.full_view(), &weight, 1)
        .unwrap();
    let mut twice = commands.clone();
    twice
        .embedding_row(&output.full_view(), &weight, 0)
        .unwrap();
    // Two lookups in one batch share the dummy. A separate batch must retain
    // its own immutable parameter and cannot fit until this batch retires.
    let occupying_batch = device.record(&twice).unwrap();
    assert_eq!(
        occupying_batch
            .payload::<NativeBatch>()
            .unwrap()
            ._retained
            .len(),
        3
    );
    let events = event_count(&control);
    assert!(device.record(&commands).is_err());
    assert_eq!(&control.lock().unwrap().events[events..], &[]);
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 4);
    assert!(!device.stats().unwrap().poisoned);
    let mut actual = [0.0; 32];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, [91.0; 32]);
    drop(occupying_batch);
    let batch = device.record(&commands).unwrap();
    device.wait(&device.submit(&batch).unwrap()).unwrap();
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(actual, expected[32..]);
    drop((batch, commands, twice, weight, output, device));
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn embedding_partial_failure_or_panic_poisons_and_retains_weight_output_and_dummy() {
    for panic in [false, true] {
        let (device, budget, control) = mock(4096, 4096);
        let (weight, _) = embedding_weight(&device, 2, 32);
        let output = device.allocate_f32(32).unwrap();
        let mut commands = CommandList::new();
        for row in 0..2 {
            commands
                .embedding_row(&output.full_view(), &weight, row)
                .unwrap();
        }
        let batch = device.record(&commands).unwrap();
        let completed = device.submit(&batch).unwrap();
        if panic {
            control.lock().unwrap().panic_dispatch = Some(4);
        } else {
            control.lock().unwrap().fail_dispatch = Some(4);
        }
        let before = device.stats().unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| device.submit(&batch)));
        if panic {
            assert!(result.is_err());
        } else {
            assert!(result.unwrap().is_err());
        }
        let after = device.stats().unwrap();
        assert!(after.poisoned);
        assert_eq!(after.dispatch_attempts, 4);
        assert_eq!(after.dispatch_count, 3);
        assert_eq!(after.logical_submissions, 1);
        assert_eq!(after.host_write_bytes, before.host_write_bytes);
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        device.wait(&completed).unwrap();
        assert!(device.submit(&batch).is_err());
        assert!(device.record(&commands).is_err());
        drop((commands, batch, completed, weight, output));
        let stats = device.stats().unwrap();
        assert_eq!(stats.weight_live_bytes, 68);
        assert_eq!(stats.tensor_live_bytes, 128);
        assert_eq!(stats.auxiliary_live_bytes, 4);
        drop(device);
        assert!(budget.current_bytes() > 0);
        assert_eq!(control.lock().unwrap().quarantine.len(), 1);
        control.lock().unwrap().quarantine.clear(); // Fake work declared stopped.
        assert_eq!(budget.current_bytes(), 0);
    }
}

struct WorkspaceFixture {
    commands: CommandList,
    input: F32Tensor,
    norm: F32Tensor,
    k: F32Tensor,
    v: F32Tensor,
    attention_output: F32Tensor,
    guarded_output: F32Tensor,
}

fn workspace_fixture(device: &dyn ExecutionDevice, positions: usize) -> WorkspaceFixture {
    let packed = [0x80_u8, 0x3f].repeat(32 * 32); // BF16 1.0.
    let encoded = device
        .upload_weight(&Weight {
            bytes: &packed,
            kind: GgmlType(30),
            rows: 32,
            cols: 32,
        })
        .unwrap();
    let weight = device.prepare_weight(&encoded).unwrap();
    let input = device.allocate_f32(32).unwrap();
    let norm = device.allocate_f32(32).unwrap();
    let q = device.allocate_f32(32).unwrap();
    let k = device.allocate_f32(positions * 32).unwrap();
    let v = device.allocate_f32(positions * 32).unwrap();
    let attention_output = device.allocate_f32(32).unwrap();
    let guarded_output = device.allocate_f32(34).unwrap();
    let mut commands = CommandList::new();
    commands
        .rmsnorm(
            &q.full_view(),
            &input.full_view(),
            Some(&norm.full_view()),
            0.125,
        )
        .unwrap();
    commands
        .linear(
            &k.view((positions - 1) * 32, 32).unwrap(),
            &q.full_view(),
            &weight,
            LinearMode::ScaledAdd { scale: 0.5 },
        )
        .unwrap();
    commands
        .incremental_causal_attention(
            &attention_output.full_view(),
            &q.full_view(),
            &k.full_view(),
            &v.full_view(),
            IncrementalAttentionConfig {
                scale: 0.5,
                ..attention_config(1, positions - 1, positions, 1, 1, 32)
            },
        )
        .unwrap();
    commands
        .rmsnorm(
            &q.full_view(),
            &attention_output.full_view(),
            Some(&norm.full_view()),
            0.125,
        )
        .unwrap();
    commands
        .linear(
            &guarded_output.view(1, 32).unwrap(),
            &q.full_view(),
            &weight,
            LinearMode::ScaledAdd { scale: 0.5 },
        )
        .unwrap();
    WorkspaceFixture {
        commands,
        input,
        norm,
        k,
        v,
        attention_output,
        guarded_output,
    }
}

fn write_workspace_fixture(
    device: &dyn ExecutionDevice,
    fixture: &WorkspaceFixture,
    round: usize,
    positions: usize,
) {
    for (tensor, modulus, factor) in [
        (&fixture.input, 11, 0.125),
        (&fixture.norm, 7, 0.25),
        (&fixture.k, 13, 0.0625),
        (&fixture.v, 17, 0.125),
    ] {
        let values: Vec<_> = (0..tensor.len())
            .map(|index| ((index + round * 3) % modulus) as f32 * factor - 0.5)
            .collect();
        device.write_f32(&tensor.full_view(), &values).unwrap();
    }
    assert_eq!(fixture.k.len(), positions * 32);
    device
        .write_f32(&fixture.guarded_output.full_view(), &[19.0; 34])
        .unwrap();
}

#[test]
fn shared_workspace_replays_mixed_chains_and_separate_batches_with_attention_boundaries() {
    for positions in [
        1, 256, 257, 341, 342, 511, 512, 513, 1023, 1024, 1025, 2047, 2048,
    ] {
        let (device, budget, _) = mock_config(NativeExecutionConfig {
            weight_capacity_bytes: 4096,
            tensor_capacity_bytes: 1024 * 1024,
            auxiliary_capacity_bytes: 64 * 1024,
        });
        // Exercise F64 alignment when the shared workspace begins at offset4.
        let prefix = device.allocate(AUXILIARY, &[0; 4]).unwrap();
        let cpu = CpuExecutionDevice::new();
        let fixture = workspace_fixture(&device, positions);
        let cpu_fixture = workspace_fixture(&cpu, positions);
        let batch = device.record(&fixture.commands).unwrap();
        let second_batch = device.record(&fixture.commands).unwrap();
        let cpu_batch = cpu.record(&cpu_fixture.commands).unwrap();
        let workspace_bytes = 128.max(positions * 12 + 4);
        let native = batch.payload::<NativeBatch>().unwrap();
        let second = second_batch.payload::<NativeBatch>().unwrap();
        assert_eq!(native.commands.len(), 11);
        assert_eq!(native._retained.len(), 11); // Eight public owners + workspace/two scalars.
        assert_eq!(
            device.stats().unwrap().auxiliary_live_bytes,
            (4 + 2 * (workspace_bytes + 8)) as u64
        );
        let scratch = native.commands[0].bindings[2];
        assert_eq!(scratch.offset, native.commands[2].bindings[2].offset);
        assert_eq!(scratch.offset, native.commands[7].bindings[2].offset);
        assert_eq!(scratch.offset, native.commands[9].bindings[2].offset);
        assert_ne!(scratch.offset, second.commands[0].bindings[2].offset);
        let scores = native.commands[4].bindings[2];
        let probabilities = native.commands[5].bindings[2];
        assert_eq!(scores.offset % 8, 0);
        assert!(!overlaps(scores, probabilities));
        assert!(
            probabilities.offset + u64::from(probabilities.length)
                <= scratch.offset + workspace_bytes as u64
        );
        assert_eq!(
            native.commands[3].bindings[0],
            native.commands[5].bindings[0]
        ); // Same scalar bits across operations.
        let mut last_completion = None;
        for (round, selected) in [&batch, &second_batch, &batch].into_iter().enumerate() {
            write_workspace_fixture(&device, &fixture, round, positions);
            write_workspace_fixture(&cpu, &cpu_fixture, round, positions);
            let before = device.stats().unwrap();
            let clone = device.clone();
            let completed = clone.submit(selected).unwrap();
            clone.wait(&completed).unwrap();
            cpu.wait(&cpu.submit(&cpu_batch).unwrap()).unwrap();
            let after = device.stats().unwrap();
            assert_eq!(after.dispatch_count, before.dispatch_count + 11);
            assert_eq!(after.host_read_bytes, before.host_read_bytes);
            assert_eq!(after.host_write_bytes, before.host_write_bytes);
            assert_eq!(after.auxiliary_live_bytes, before.auxiliary_live_bytes);
            for (actual_tensor, expected_tensor) in [
                (&fixture.guarded_output, &cpu_fixture.guarded_output),
                (&fixture.attention_output, &cpu_fixture.attention_output),
                (&fixture.k, &cpu_fixture.k),
            ] {
                let mut actual = vec![f32::NAN; actual_tensor.len()];
                let mut expected = vec![f32::NAN; expected_tensor.len()];
                device
                    .read_f32(&actual_tensor.full_view(), &mut actual)
                    .unwrap();
                cpu.read_f32(&expected_tensor.full_view(), &mut expected)
                    .unwrap();
                for (actual, expected) in actual.iter().zip(&expected) {
                    assert!(
                        actual.is_finite()
                            && (actual - expected).abs() <= 1e-5 + 1e-6 * expected.abs(),
                        "{actual} != {expected}"
                    );
                }
            }
            last_completion = Some(completed);
        }
        drop((fixture, batch, second_batch, prefix));
        assert_eq!(
            device.stats().unwrap().auxiliary_live_bytes,
            (workspace_bytes + 8) as u64
        );
        assert_eq!(device.stats().unwrap().weight_live_bytes, 2048);
        drop(last_completion);
        assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
        assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
        drop(device);
        assert_eq!(budget.current_bytes(), 0);
    }
}

#[test]
fn scalar_interning_preserves_signed_zero_bits_and_deduplicates_across_operator_kinds() {
    let (device, budget, _) = mock(4096, 4096);
    let input = device.upload_f32(&[2.0; 32]).unwrap();
    let norm = device.upload_f32(&[1.0; 32]).unwrap();
    let output = device.allocate_f32(32).unwrap();
    let linear = prepared_bf16(&device, 32, 32);
    let (embedding, expected) = embedding_weight(&device, 1, 32);
    let mut commands = CommandList::new();
    for zero in [0.0, -0.0, 0.0] {
        commands
            .rmsnorm(
                &output.full_view(),
                &input.full_view(),
                Some(&norm.full_view()),
                zero,
            )
            .unwrap();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &linear,
                LinearMode::ScaledAdd { scale: zero },
            )
            .unwrap();
        commands
            .swiglu(&output.full_view(), &input.full_view(), zero)
            .unwrap();
    }
    commands
        .embedding_row(&output.full_view(), &embedding, 0)
        .unwrap();
    let batch = device.record(&commands).unwrap();
    let native = batch.payload::<NativeBatch>().unwrap();
    assert_eq!(native._retained.len(), 8); // Five public owners and three private owners.
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 128 + 8);
    let positive = native.commands[0].bindings[0];
    let negative = native.commands[5].bindings[0];
    assert_ne!(positive, negative);
    for (start, parameter) in [(0, positive), (5, negative), (10, positive)] {
        assert_eq!(native.commands[start].bindings[0], parameter);
        assert_eq!(native.commands[start + 3].bindings[0], parameter);
        assert_eq!(native.commands[start + 4].bindings[0], parameter);
    }
    assert_eq!(native.commands[15].bindings[1], positive); // Embedding dummy is +0.
    {
        let mut state = device.state().unwrap();
        for (region, bits) in [(positive, 0_u32), (negative, 0x8000_0000)] {
            let mut bytes = [0; 4];
            state
                .transport
                .read(region.slot, region.offset, &mut bytes)
                .unwrap();
            assert_eq!(u32::from_le_bytes(bytes), bits);
        }
    }
    drop((commands, linear, embedding, norm, input));
    let submission = device.submit(&batch).unwrap();
    device.wait(&submission).unwrap();
    let mut actual = [f32::NAN; 32];
    device.read_f32(&output.full_view(), &mut actual).unwrap();
    assert_eq!(bits(&actual), bits(&expected));
    drop((batch, output));
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 136);
    drop(submission);
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    assert_eq!(device.stats().unwrap().weight_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn compact_batch_ownership_retains_distinct_noop_operands_once() {
    let (device, budget, _) = mock(4, 4096);
    let first = device.allocate_f32(1).unwrap();
    let second = device.allocate_f32(1).unwrap();
    let third = device.allocate_f32(1).unwrap();
    let mut commands = CommandList::new();
    for _ in 0..3 {
        commands
            .swiglu(&first.view(0, 0).unwrap(), &second.view(0, 0).unwrap(), 0.0)
            .unwrap();
        commands
            .rope(
                &third.view(0, 0).unwrap(),
                &first.view(0, 0).unwrap(),
                &second.view(0, 0).unwrap(),
                RopeLayout::SplitHalf,
                false,
            )
            .unwrap();
    }
    let batch = device.record(&commands).unwrap();
    let native = batch.payload::<NativeBatch>().unwrap();
    assert_eq!(native._retained.len(), 3);
    assert!(native.commands.is_empty());
    drop((commands, first, second, third));
    let completion = device.submit(&batch).unwrap();
    drop(batch);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 12);
    assert_eq!(device.stats().unwrap().auxiliary_live_bytes, 0);
    drop(completion);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    drop(device);
    assert_eq!(budget.current_bytes(), 0);
}
