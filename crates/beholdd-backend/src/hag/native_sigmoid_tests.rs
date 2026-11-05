//! Adapter contract tests for the explicit Qwen35 sigmoid-gate program.
//!
//! The transport records current operand bytes at dispatch; it does not compute
//! sigmoid, mutate output, or stand in for an independent ISA arithmetic oracle.
//! Integrate as a child test module of `native_execution` after slot16 is embedded.

use super::*;

#[derive(Clone, Debug, PartialEq, Eq)]
enum Event {
    Write(usize, u64, usize),
    Read(usize, u64, usize),
    Dispatch(usize),
}

#[derive(Debug)]
struct DispatchEvidence {
    program: usize,
    bindings: [BufferRegion; 3],
    scalars: [u32; 2],
    groups: u32,
    input_bits: Vec<u32>,
    gate_bits: Vec<u32>,
}

#[derive(Default)]
struct Evidence {
    events: Vec<Event>,
    dispatches: Vec<DispatchEvidence>,
    transport_drops: usize,
}

struct ProbeTransport {
    tensors: Vec<u8>,
    evidence: Arc<Mutex<Evidence>>,
}

impl ProbeTransport {
    fn bits(&self, region: BufferRegion) -> Vec<u32> {
        assert_eq!(region.slot, TENSORS);
        assert_eq!(region.length % 4, 0);
        let first = region.offset as usize;
        self.tensors[first..first + region.length as usize]
            .chunks_exact(4)
            .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
            .collect()
    }
}

impl Drop for ProbeTransport {
    fn drop(&mut self) {
        self.evidence.lock().unwrap().transport_drops += 1;
    }
}

impl Transport for ProbeTransport {
    fn allocation_bytes(&self) -> u64 {
        self.tensors.len() as u64
    }

    fn is_poisoned(&self) -> bool {
        false
    }

    fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        assert_eq!(slot, TENSORS, "sigmoid recording needs no auxiliary upload");
        self.evidence
            .lock()
            .unwrap()
            .events
            .push(Event::Write(slot, offset, bytes.len()));
        let first = offset as usize;
        self.tensors[first..first + bytes.len()].copy_from_slice(bytes);
        Ok(())
    }

    fn read(&mut self, slot: usize, offset: u64, bytes: &mut [u8]) -> Result<(), ExecutionError> {
        assert_eq!(slot, TENSORS);
        self.evidence
            .lock()
            .unwrap()
            .events
            .push(Event::Read(slot, offset, bytes.len()));
        let first = offset as usize;
        bytes.copy_from_slice(&self.tensors[first..first + bytes.len()]);
        Ok(())
    }

    fn dispatch(
        &mut self,
        program: usize,
        bindings: [BufferRegion; 3],
        scalars: [u32; 2],
        groups: u32,
    ) -> Result<u64, ExecutionError> {
        assert_eq!(program, 16);
        assert!((1..=4096).contains(&scalars[0]));
        assert_eq!(scalars[1], 1);
        assert_eq!(groups, scalars[0].div_ceil(64));
        assert_eq!(bindings.map(|region| region.slot), [TENSORS; 3]);
        assert_eq!(bindings.map(|region| region.length), [scalars[0] * 4; 3]);
        for pair in [(0, 1), (0, 2), (1, 2)] {
            assert!(!overlaps(bindings[pair.0], bindings[pair.1]));
        }
        // This fixture observes late-bound input bytes only. No arithmetic or
        // host transfer is performed on behalf of the native implementation.
        let observed = DispatchEvidence {
            program,
            bindings,
            scalars,
            groups,
            input_bits: self.bits(bindings[1]),
            gate_bits: self.bits(bindings[0]),
        };
        let mut evidence = self.evidence.lock().unwrap();
        evidence.events.push(Event::Dispatch(program));
        evidence.dispatches.push(observed);
        Ok(evidence.dispatches.len() as u64)
    }
}

fn device(profile: NativeExecutionProfile) -> (NativeExecutionDevice, Arc<Mutex<Evidence>>) {
    let config = NativeExecutionConfig {
        weight_capacity_bytes: 4,
        tensor_capacity_bytes: 256 * 1024,
        auxiliary_capacity_bytes: 4,
    };
    let evidence = Arc::new(Mutex::new(Evidence::default()));
    let device = NativeExecutionDevice::from_transport_with_profile(
        config,
        profile,
        Box::new(ProbeTransport {
            tensors: vec![0; config.tensor_capacity_bytes as usize],
            evidence: Arc::clone(&evidence),
        }),
    )
    .unwrap();
    (device, evidence)
}

fn event_count(evidence: &Arc<Mutex<Evidence>>) -> usize {
    evidence.lock().unwrap().events.len()
}

fn list(output: &F32TensorView, input: &F32TensorView, gate: &F32TensorView) -> CommandList {
    let mut list = CommandList::new();
    list.sigmoid_gate(output, input, gate).unwrap();
    list
}

#[test]
fn dense_profile_refuses_sigmoid_without_allocations_or_transfers() {
    let (device, evidence) = device(NativeExecutionProfile::DenseQwen3);
    let output = device.allocate_f32(1).unwrap();
    let input = device.allocate_f32(1).unwrap();
    let gate = device.allocate_f32(1).unwrap();
    for len in [0, 1] {
        let commands = list(
            &output.view(0, len).unwrap(),
            &input.view(0, len).unwrap(),
            &gate.view(0, len).unwrap(),
        );
        let before = device.stats().unwrap();
        let events = event_count(&evidence);
        assert!(device.record(&commands).is_err());
        assert_eq!(device.stats().unwrap(), before);
        assert_eq!(event_count(&evidence), events);
    }
}

#[test]
fn probe_slot16_has_exact_subview_bindings_groups_and_retained_operands() {
    for len in [1_usize, 63, 64, 65, 4096] {
        let (device, evidence) = device(NativeExecutionProfile::Qwen35Probe);
        let prefix = device.allocate_f32(3).unwrap();
        let output = device.allocate_f32(len + 2).unwrap();
        let input = device.allocate_f32(len + 4).unwrap();
        let gate = device.allocate_f32(len + 6).unwrap();
        let commands = list(
            &output.view(1, len).unwrap(),
            &input.view(2, len).unwrap(),
            &gate.view(3, len).unwrap(),
        );
        // Four-byte arena packing, deliberately nonzero tensor and view offsets.
        let expected = [
            BufferRegion {
                slot: TENSORS,
                offset: ((3 + len + 2 + len + 4 + 3) * 4) as u64,
                length: (len * 4) as u32,
            },
            BufferRegion {
                slot: TENSORS,
                offset: ((3 + len + 2 + 2) * 4) as u64,
                length: (len * 4) as u32,
            },
            BufferRegion {
                slot: TENSORS,
                offset: 16,
                length: (len * 4) as u32,
            },
        ];
        let before = device.stats().unwrap();
        let events = event_count(&evidence);
        let batch = device.record(&commands).unwrap();
        assert_eq!(device.stats().unwrap(), before);
        assert_eq!(event_count(&evidence), events);
        assert_eq!(batch.len(), 1);
        let native = batch.payload::<NativeBatch>().unwrap();
        assert_eq!(native.semantic_count, 1);
        assert_eq!(native.commands.len(), 1);
        assert_eq!(native._retained.len(), 3);
        let command = &native.commands[0];
        assert_eq!(command.program, 16);
        assert_eq!(command.bindings, expected);
        assert_eq!(command.scalars, [len as u32, 1]);
        assert_eq!(command.groups, len.div_ceil(64) as u32);

        drop(commands);
        drop(output);
        drop(input);
        drop(gate);
        drop(prefix);
        let retained_bytes = ((3 * len + 12) * 4) as u64;
        assert_eq!(device.stats().unwrap().tensor_live_bytes, retained_bytes);
        let submission = device.submit(&batch).unwrap();
        drop(batch);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, retained_bytes);
        device.wait(&submission).unwrap();
        let proof = evidence.lock().unwrap();
        assert_eq!(proof.dispatches.len(), 1);
        assert_eq!(proof.dispatches[0].program, 16);
        assert_eq!(proof.dispatches[0].bindings, expected);
        assert_eq!(proof.dispatches[0].scalars, [len as u32, 1]);
        assert_eq!(proof.dispatches[0].groups, len.div_ceil(64) as u32);
        drop(proof);
        drop(submission);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
        assert_eq!(evidence.lock().unwrap().transport_drops, 0);
        drop(device);
        assert_eq!(evidence.lock().unwrap().transport_drops, 1);
    }
}

#[test]
fn overlap_each_pair_and_oversize_refuse_the_whole_recording_before_side_effects() {
    let (device, evidence) = device(NativeExecutionProfile::Qwen35Probe);
    let shared = device.allocate_f32(16).unwrap();
    let separate = device.allocate_f32(8).unwrap();
    let first = shared.view(1, 8).unwrap();
    let partial = shared.view(4, 8).unwrap();
    let disjoint = separate.full_view();
    for (output, input, gate) in [
        (&first, &partial, &disjoint),
        (&first, &disjoint, &partial),
        (&disjoint, &first, &partial),
        (&first, &first, &disjoint),
    ] {
        let mut commands = CommandList::new();
        // This earlier valid command would upload an epsilon parameter if the
        // recorder performed lowering before validating the later sigmoid.
        commands.rmsnorm(&disjoint, &first, None, 1e-6).unwrap();
        commands.sigmoid_gate(output, input, gate).unwrap();
        let before = device.stats().unwrap();
        let events = event_count(&evidence);
        assert!(device.record(&commands).is_err());
        assert_eq!(device.stats().unwrap(), before);
        assert_eq!(event_count(&evidence), events);
    }
    let output = device.allocate_f32(4097).unwrap();
    let input = device.allocate_f32(4097).unwrap();
    let gate = device.allocate_f32(4097).unwrap();
    let commands = list(&output.full_view(), &input.full_view(), &gate.full_view());
    let before = device.stats().unwrap();
    let events = event_count(&evidence);
    assert!(device.record(&commands).is_err());
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&evidence), events);
}

#[test]
fn mismatched_lengths_and_every_foreign_operand_refuse_without_state_change() {
    let (device, evidence) = device(NativeExecutionProfile::Qwen35Probe);
    let (foreign_device, foreign_evidence) = self::device(NativeExecutionProfile::Qwen35Probe);
    let output = device.allocate_f32(4).unwrap();
    let input = device.allocate_f32(4).unwrap();
    let gate = device.allocate_f32(4).unwrap();
    let foreign = foreign_device.allocate_f32(4).unwrap();
    let local = [output.full_view(), input.full_view(), gate.full_view()];
    let foreign = foreign.full_view();
    let before = device.stats().unwrap();
    let foreign_before = foreign_device.stats().unwrap();
    let events = event_count(&evidence);
    let foreign_events = event_count(&foreign_evidence);
    let mut commands = list(&local[0], &local[1], &local[2]);
    for role in 0..3 {
        let mut operands = local.clone();
        operands[role] = local[role].view(0, 3).unwrap();
        assert!(commands
            .sigmoid_gate(&operands[0], &operands[1], &operands[2])
            .is_err());
        assert_eq!(commands.len(), 1);
        // Independently exercise native planning, which cannot trust wrappers.
        assert!(device
            .plan(&Command::SigmoidGate {
                output: operands[0].clone(),
                input: operands[1].clone(),
                gate: operands[2].clone(),
            })
            .is_err());
        operands = local.clone();
        operands[role] = foreign.clone();
        assert!(commands
            .sigmoid_gate(&operands[0], &operands[1], &operands[2])
            .is_err());
        assert_eq!(commands.len(), 1);
        assert!(device
            .plan(&Command::SigmoidGate {
                output: operands[0].clone(),
                input: operands[1].clone(),
                gate: operands[2].clone(),
            })
            .is_err());
    }
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(foreign_device.stats().unwrap(), foreign_before);
    assert_eq!(event_count(&evidence), events);
    assert_eq!(event_count(&foreign_evidence), foreign_events);
    // A completely foreign list must also fail at the queue-owner boundary.
    let foreign_list = list(&foreign, &foreign, &foreign);
    assert!(device.record(&foreign_list).is_err());
    assert_eq!(device.stats().unwrap(), before);
    assert_eq!(event_count(&evidence), events);
}

#[test]
fn empty_sigmoid_is_a_retained_semantic_noop_without_dispatch_or_transfers() {
    let (device, evidence) = device(NativeExecutionProfile::Qwen35Probe);
    let output = device.allocate_f32(3).unwrap();
    let input = device.allocate_f32(5).unwrap();
    let gate = device.allocate_f32(7).unwrap();
    let commands = list(
        &output.view(1, 0).unwrap(),
        &input.view(2, 0).unwrap(),
        &gate.view(3, 0).unwrap(),
    );
    let before = device.stats().unwrap();
    let events = event_count(&evidence);
    let batch = device.record(&commands).unwrap();
    assert_eq!(batch.len(), 1);
    let native = batch.payload::<NativeBatch>().unwrap();
    assert!(native.commands.is_empty());
    assert_eq!(native._retained.len(), 3);
    assert_eq!(device.stats().unwrap(), before);
    drop(commands);
    drop(output);
    drop(input);
    drop(gate);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 60);
    let submission = device.submit(&batch).unwrap();
    device.wait(&submission).unwrap();
    let after = device.stats().unwrap();
    assert_eq!(after.logical_submissions, before.logical_submissions + 1);
    assert_eq!(after.dispatch_attempts, before.dispatch_attempts);
    assert_eq!(after.dispatch_count, before.dispatch_count);
    assert_eq!(after.host_write_bytes, before.host_write_bytes);
    assert_eq!(after.host_read_bytes, before.host_read_bytes);
    assert_eq!(event_count(&evidence), events);
    drop(batch);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 60);
    drop(submission);
    assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
}

#[test]
fn repeated_submit_observes_current_operands_without_adapter_transfers() {
    let (device, evidence) = device(NativeExecutionProfile::Qwen35Probe);
    let output = device.allocate_f32(3).unwrap();
    let input = device.allocate_f32(3).unwrap();
    let gate = device.allocate_f32(3).unwrap();
    let commands = list(&output.full_view(), &input.full_view(), &gate.full_view());
    let batch = device.record(&commands).unwrap();
    drop(commands);
    for (iteration, (input_values, gate_values)) in [
        ([1.0_f32, -2.0, 3.0], [4.0_f32, 5.0, 6.0]),
        ([-7.0_f32, 8.0, 9.0], [-90.0_f32, -105.0, 0.0]),
    ]
    .into_iter()
    .enumerate()
    {
        device.write_f32(&input.full_view(), &input_values).unwrap();
        device.write_f32(&gate.full_view(), &gate_values).unwrap();
        let before = device.stats().unwrap();
        let events = event_count(&evidence);
        let submission = device.submit(&batch).unwrap();
        device.wait(&submission).unwrap();
        device.wait(&submission).unwrap();
        let after = device.stats().unwrap();
        assert_eq!(after.host_write_bytes, before.host_write_bytes);
        assert_eq!(after.host_read_bytes, before.host_read_bytes);
        assert_eq!(after.uploaded_weight_bytes, before.uploaded_weight_bytes);
        assert_eq!(after.auxiliary_live_bytes, before.auxiliary_live_bytes);
        assert_eq!(after.tensor_live_bytes, before.tensor_live_bytes);
        assert_eq!(after.logical_submissions, before.logical_submissions + 1);
        assert_eq!(after.dispatch_attempts, before.dispatch_attempts + 1);
        assert_eq!(after.dispatch_count, before.dispatch_count + 1);
        let proof = evidence.lock().unwrap();
        assert_eq!(&proof.events[events..], &[Event::Dispatch(16)]);
        assert_eq!(proof.dispatches.len(), iteration + 1);
        assert_eq!(
            proof.dispatches[iteration].input_bits,
            input_values.map(f32::to_bits)
        );
        assert_eq!(
            proof.dispatches[iteration].gate_bits,
            gate_values.map(f32::to_bits)
        );
    }
}
