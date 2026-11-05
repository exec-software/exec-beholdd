//! SOURCE-ONLY future child module of `native_execution`; not integrated.
//! Native finite refusal for an explicitly supplied immutable program17 library.
//! Each check reads only64 raw status words after the native fence. No activation
//! readback, arithmetic emulation, fallback, or serving capability is implied.
//! Each attempt also writes256B of pending status before submission, so a skipped
//! or early-returned dispatch cannot reuse success from an earlier check.
//! The caller serializes its input/queue for the whole check. This owner exposes
//! no private packet views, and cannot recover or bypass a poisoned device.

use super::*;

/// Diagnostic sentinel surrounding the private dummy and status extents.
pub const NATIVE_FINITE_GUARD: f32 = -12345.5;
/// Largest input view scanned by one native dispatch.
pub const NATIVE_FINITE_MAX_ELEMENTS: usize = 65_536;
/// One private allocation, including dummy/status payloads and four sentinels.
pub const NATIVE_FINITE_PACKET_BYTES: u32 = 276;
const DUMMY_OFFSET: u32 = 4;
const STATUS_OFFSET: u32 = 16;
const STATUS_WORDS: usize = 64;
const STATUS_BYTES: u32 = 256;
const GUARDS: [u32; 4] = [0, 8, 12, 272];
static PENDING_STATUSES: [f32; STATUS_WORDS] = [f32::from_bits(0xffff_ffff); STATUS_WORDS];

fn require(condition: bool, message: &str) -> Result<(), ExecutionError> {
    if condition {
        Ok(())
    } else {
        Err(ExecutionError::Shape(message.into()))
    }
}

/// Checker-local successful control-transfer and acceptance counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeFiniteStats {
    /// Packet zeroing276B, four guard writes16B, initial pending status256B.
    pub initialization_write_bytes: u64,
    /// Successful256B pending-status writes before dispatch, excluding initialization.
    pub pending_writes: u64,
    /// Bytes in successful per-check pending writes, including cancelled attempts.
    pub pending_write_bytes: u64,
    /// Successful explicit status reads, including refusals/post-read cancellation.
    pub status_reads: u64,
    /// Bytes successfully read from the private status extent.
    pub status_read_bytes: u64,
    /// Successful checks after all statuses and the final control callback pass.
    pub accepted_checks: u64,
}

/// One queue-bound reusable status owner; checks are serialized by `&mut self`.
/// Larger inputs must be split explicitly by the caller into bounded views.
pub struct NativeFiniteCheck {
    device: NativeExecutionDevice,
    packet: F32Tensor,
    stats: NativeFiniteStats,
}

impl NativeFiniteCheck {
    /// Reject missing future program17, incompatible profiles or arena shortage
    /// before allocation/upload. The real library count is never caller-supplied.
    pub fn new(device: &NativeExecutionDevice) -> Result<Self, ExecutionError> {
        // The constant is the return-array length of qwen35_program_bytes().
        // create checks the profile before evaluating it; no library vectors
        // need to be materialized for this metadata-only admission.
        Self::create(device, || kernels::QWEN35_PROGRAM_COUNT)
    }

    fn create(
        device: &NativeExecutionDevice,
        program_count: impl FnOnce() -> usize,
    ) -> Result<Self, ExecutionError> {
        require(
            device.profile() == NativeExecutionProfile::Qwen35Probe,
            "native finite check requires the explicit Qwen35Probe profile",
        )?;
        require(
            program_count() > 17,
            "native finite check requires immutable program17",
        )?;
        require(
            NATIVE_FINITE_PACKET_BYTES <= device.inner.config.tensor_capacity_bytes,
            "native finite packet exceeds tensor arena capacity",
        )?;
        {
            let state = device.state()?;
            require(
                state.arenas[TENSORS]
                    .free
                    .iter()
                    .any(|span| span.bytes >= NATIVE_FINITE_PACKET_BYTES),
                "native finite packet requires one contiguous tensor span",
            )?;
        }
        let packet = device.allocate_f32(NATIVE_FINITE_PACKET_BYTES as usize / 4)?;
        for offset in GUARDS {
            device.write_f32(
                &packet.view(offset as usize / 4, 1)?,
                &[NATIVE_FINITE_GUARD],
            )?;
        }
        // Deliberately not zero initially: accidental checking without a complete
        // valid dispatch must refuse. Every valid shader run rewrites all64 words.
        device.write_f32(
            &packet.view(STATUS_OFFSET as usize / 4, STATUS_WORDS)?,
            &PENDING_STATUSES,
        )?;
        Ok(Self {
            device: device.clone(),
            packet,
            stats: NativeFiniteStats {
                initialization_write_bytes: u64::from(
                    NATIVE_FINITE_PACKET_BYTES + 16 + STATUS_BYTES,
                ),
                ..NativeFiniteStats::default()
            },
        })
    }

    /// Snapshot counters without accessing the device or clearing refusal state.
    pub fn stats(&self) -> NativeFiniteStats {
        self.stats
    }

    /// No input or packet ownership escapes this method; the recording retains
    /// both allocations until the fence and status read have completed.
    fn record(&self, input: &F32TensorView) -> Result<RecordedBatch, ExecutionError> {
        require(
            (1..=NATIVE_FINITE_MAX_ELEMENTS).contains(&input.len()),
            "native finite input must contain1..65536 elements",
        )?;
        let (source, source_region) = self.device.tensor_region(input, "finite-check input")?;
        let packet_view = self.packet.full_view();
        let (packet, packet_region) = self
            .device
            .tensor_region(&packet_view, "finite-check private packet")?;
        let dummy = BufferRegion {
            slot: TENSORS,
            offset: packet_region.offset + u64::from(DUMMY_OFFSET),
            length: 4,
        };
        let status = BufferRegion {
            slot: TENSORS,
            offset: packet_region.offset + u64::from(STATUS_OFFSET),
            length: STATUS_BYTES,
        };
        // The private packet has no public view, but keep the descriptor contract
        // explicit even for internally constructed/forged same-owner wrappers.
        require(
            !overlaps(source_region, packet_region),
            "finite-check input must not overlap the private packet",
        )?;
        let mut commands = Vec::new();
        commands.try_reserve_exact(1).map_err(backend)?;
        commands.push(NativeDispatch {
            program: 17,
            bindings: [source_region, dummy, status],
            scalars: [input.len() as u32, 1],
            groups: 1,
        });
        let mut retained = Vec::new();
        retained.try_reserve_exact(2).map_err(backend)?;
        retained.push(Arc::clone(&source.allocation));
        retained.push(Arc::clone(&packet.allocation));
        Ok(RecordedBatch::new(
            self.device.inner.owner,
            1,
            Arc::new(NativeBatch {
                inner: Arc::clone(&self.device.inner),
                semantic_count: 1,
                commands,
                _retained: retained,
            }),
        ))
    }

    /// Checks only the supplied view. Success does not describe any other tensor
    /// or later mutation. Finite refusal is retryable; transport failures poison.
    pub fn check(&mut self, input: &F32TensorView) -> Result<(), ExecutionError> {
        self.check_controlled(input, &mut || Ok(()))
    }

    /// Cancellation runs before the pending write, before submission, after its
    /// fence, before status read, and after valid statuses before acceptance.
    /// A normal check writes256B and reads256B of control data, never activations.
    pub fn check_controlled(
        &mut self,
        input: &F32TensorView,
        control: &mut dyn FnMut() -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        drop(self.device.state()?);
        let batch = self.record(input)?;
        let pending = self
            .stats
            .pending_writes
            .checked_add(1)
            .ok_or_else(|| backend("native finite pending counter overflows"))?;
        let pending_bytes = self
            .stats
            .pending_write_bytes
            .checked_add(u64::from(STATUS_BYTES))
            .ok_or_else(|| backend("native finite pending byte counter overflows"))?;
        let reads = self
            .stats
            .status_reads
            .checked_add(1)
            .ok_or_else(|| backend("native finite status counter overflows"))?;
        let bytes = self
            .stats
            .status_read_bytes
            .checked_add(u64::from(STATUS_BYTES))
            .ok_or_else(|| backend("native finite status byte counter overflows"))?;
        let accepted = self
            .stats
            .accepted_checks
            .checked_add(1)
            .ok_or_else(|| backend("native finite acceptance counter overflows"))?;
        let status = self.packet.view(STATUS_OFFSET as usize / 4, STATUS_WORDS)?;
        control()?;
        self.device.write_f32(&status, &PENDING_STATUSES)?;
        self.stats.pending_writes = pending;
        self.stats.pending_write_bytes = pending_bytes;
        control()?;
        let completion = self.device.submit(&batch)?;
        self.device.wait(&completion)?;
        control()?;
        let mut words = [0.0_f32; STATUS_WORDS];
        control()?;
        self.device.read_f32(&status, &mut words)?;
        self.stats.status_reads = reads;
        self.stats.status_read_bytes = bytes;
        for (lane, word) in words.into_iter().enumerate() {
            if word.to_bits() != 0 {
                return Err(backend(format!(
                    "native finite check lane {lane} refused with status 0x{:08x}",
                    word.to_bits()
                )));
            }
        }
        control()?;
        self.stats.accepted_checks = accepted;
        // Explicit retirement here documents that wait alone does not release
        // the retained input/private allocation. Error paths use the same RAII.
        drop(completion);
        drop(batch);
        Ok(())
    }

    /// Qualification-only bounded sentinel copy. Normal checking never reads
    /// guards, dummy contents, or activations, and no mutable packet view escapes.
    pub fn diagnostic_read_guards(&self) -> Result<[f32; 4], ExecutionError> {
        let mut guards = [0.0; 4];
        for (index, offset) in GUARDS.into_iter().enumerate() {
            self.device.read_f32(
                &self.packet.view(offset as usize / 4, 1)?,
                &mut guards[index..index + 1],
            )?;
        }
        Ok(guards)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Event {
        Write(u64, usize),
        Read(u64, usize),
        Dispatch,
    }
    struct Evidence {
        bytes: Vec<u8>,
        statuses: [u32; STATUS_WORDS],
        events: Vec<Event>,
        calls: Vec<(usize, [BufferRegion; 3], [u32; 2], u32)>,
        fail_dispatch: bool,
        fail_read: bool,
        fail_write: bool,
        skip_dispatch_at: Option<usize>,
        drops: usize,
    }
    struct Mock(Arc<Mutex<Evidence>>);
    impl Drop for Mock {
        fn drop(&mut self) {
            self.0.lock().unwrap().drops += 1;
        }
    }
    impl Transport for Mock {
        fn allocation_bytes(&self) -> u64 {
            self.0.lock().unwrap().bytes.len() as u64
        }
        fn is_poisoned(&self) -> bool {
            false
        }
        fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
            assert_eq!(slot, TENSORS);
            let mut state = self.0.lock().unwrap();
            state.events.push(Event::Write(offset, bytes.len()));
            if state.fail_write {
                return Err(backend("injected write failure"));
            }
            let first = offset as usize;
            state.bytes[first..first + bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
        fn read(
            &mut self,
            slot: usize,
            offset: u64,
            bytes: &mut [u8],
        ) -> Result<(), ExecutionError> {
            assert_eq!(slot, TENSORS);
            let mut state = self.0.lock().unwrap();
            state.events.push(Event::Read(offset, bytes.len()));
            if state.fail_read {
                return Err(backend("injected read failure"));
            }
            let first = offset as usize;
            bytes.copy_from_slice(&state.bytes[first..first + bytes.len()]);
            Ok(())
        }
        fn dispatch(
            &mut self,
            program: usize,
            bindings: [BufferRegion; 3],
            scalars: [u32; 2],
            groups: u32,
        ) -> Result<u64, ExecutionError> {
            let mut state = self.0.lock().unwrap();
            state.events.push(Event::Dispatch);
            state.calls.push((program, bindings, scalars, groups));
            if state.fail_dispatch {
                return Err(backend("injected dispatch failure"));
            }
            assert_eq!(program, 17);
            assert_eq!(groups, 1);
            assert_eq!(scalars[1], 1);
            assert_eq!(bindings.map(|region| region.slot), [TENSORS; 3]);
            assert_eq!(
                bindings.map(|region| region.length),
                [scalars[0] * 4, 4, 256]
            );
            if state.skip_dispatch_at == Some(state.calls.len()) {
                return Ok(state.calls.len() as u64);
            }
            // Explicit completion fixture only: never inspect/classify inputs or
            // emulate the shader's scan. GPU finite classification is separate.
            for lane in 0..STATUS_WORDS {
                let first = bindings[2].offset as usize + lane * 4;
                let word = state.statuses[lane].to_le_bytes();
                state.bytes[first..first + 4].copy_from_slice(&word);
            }
            Ok(state.calls.len() as u64)
        }
    }
    fn mock(
        profile: NativeExecutionProfile,
        capacity: u32,
    ) -> (NativeExecutionDevice, Arc<Mutex<Evidence>>) {
        let evidence = Arc::new(Mutex::new(Evidence {
            bytes: vec![0; capacity as usize],
            statuses: [0; STATUS_WORDS],
            events: Vec::new(),
            calls: Vec::new(),
            fail_dispatch: false,
            fail_read: false,
            fail_write: false,
            skip_dispatch_at: None,
            drops: 0,
        }));
        let device = NativeExecutionDevice::from_transport_with_profile(
            NativeExecutionConfig {
                weight_capacity_bytes: 4,
                tensor_capacity_bytes: capacity,
                auxiliary_capacity_bytes: 4,
            },
            profile,
            Box::new(Mock(evidence.clone())),
        )
        .unwrap();
        (device, evidence)
    }
    fn checker(device: &NativeExecutionDevice) -> NativeFiniteCheck {
        NativeFiniteCheck::create(device, || 18).unwrap()
    }

    #[test]
    fn admission_and_initial_packet_are_bounded_before_io() {
        let (dense, dense_events) = mock(NativeExecutionProfile::DenseQwen3, 512);
        assert!(NativeFiniteCheck::create(&dense, || panic!("profile must refuse first")).is_err());
        assert!(dense_events.lock().unwrap().events.is_empty());
        let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 512);
        for count in [0, 17] {
            assert!(NativeFiniteCheck::create(&device, || count).is_err());
        }
        assert!(evidence.lock().unwrap().events.is_empty());
        let (small, evidence) = mock(
            NativeExecutionProfile::Qwen35Probe,
            NATIVE_FINITE_PACKET_BYTES - 4,
        );
        assert!(NativeFiniteCheck::create(&small, || 18).is_err());
        assert!(evidence.lock().unwrap().events.is_empty());
        let owner = checker(&device);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 276);
        assert_eq!(owner.stats().initialization_write_bytes, 548);
        assert_eq!(device.stats().unwrap().host_write_bytes, 548);
        assert_eq!(
            owner.diagnostic_read_guards().unwrap().map(f32::to_bits),
            [NATIVE_FINITE_GUARD.to_bits(); 4]
        );
        drop(owner);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
    }

    #[test]
    fn exact_layout_status_only_io_and_retained_lifetime() {
        let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 512 * 1024);
        let prefix = device.allocate_f32(3).unwrap();
        let input = device.allocate_f32(NATIVE_FINITE_MAX_ELEMENTS + 2).unwrap();
        let mut owner = checker(&device);
        let packet_base = (3 + NATIVE_FINITE_MAX_ELEMENTS + 2) * 4;
        for len in [1, 63, 64, 65, 65536] {
            let before = device.stats().unwrap();
            let first_event = evidence.lock().unwrap().events.len();
            owner.check(&input.view(1, len).unwrap()).unwrap();
            let after = device.stats().unwrap();
            assert_eq!(after.dispatch_count, before.dispatch_count + 1);
            assert_eq!(after.dispatch_attempts, before.dispatch_attempts + 1);
            assert_eq!(after.logical_submissions, before.logical_submissions + 1);
            assert_eq!(after.host_read_bytes, before.host_read_bytes + 256);
            assert_eq!(after.host_write_bytes, before.host_write_bytes + 256);
            assert_eq!(after.tensor_live_bytes, before.tensor_live_bytes);
            let state = evidence.lock().unwrap();
            assert_eq!(
                &state.events[first_event..],
                &[
                    Event::Write((packet_base + 16) as u64, 256),
                    Event::Dispatch,
                    Event::Read((packet_base + 16) as u64, 256)
                ]
            );
            let (_, bindings, scalars, groups) = state.calls.last().unwrap();
            assert_eq!(
                bindings.map(|region| region.offset),
                [16, (packet_base + 4) as u64, (packet_base + 16) as u64]
            );
            assert_eq!(
                bindings.map(|region| region.length),
                [(len * 4) as u32, 4, 256]
            );
            assert_eq!(*scalars, [len as u32, 1]);
            assert_eq!(*groups, 1);
        }
        assert_eq!(owner.stats().accepted_checks, 5);
        assert_eq!(owner.stats().status_reads, 5);
        assert_eq!(owner.stats().status_read_bytes, 1280);
        assert_eq!(owner.stats().pending_writes, 5);
        assert_eq!(owner.stats().pending_write_bytes, 1280);
        let batch = owner.record(&input.view(1, 65).unwrap()).unwrap();
        assert_eq!(batch.payload::<NativeBatch>().unwrap()._retained.len(), 2);
        drop(input);
        drop(owner);
        drop(prefix);
        let retained = ((NATIVE_FINITE_MAX_ELEMENTS + 2) * 4) as u64 + 276;
        assert_eq!(device.stats().unwrap().tensor_live_bytes, retained);
        let completion = device.submit(&batch).unwrap();
        drop(batch);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, retained);
        device.wait(&completion).unwrap();
        drop(completion);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 0);
        drop(device);
        assert_eq!(evidence.lock().unwrap().drops, 1);
    }

    #[test]
    fn malformed_foreign_and_private_packet_inputs_refuse_before_dispatch() {
        let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 512 * 1024);
        let (foreign, _) = mock(NativeExecutionProfile::Qwen35Probe, 512);
        let input = device.allocate_f32(65537).unwrap();
        let foreign_input = foreign.allocate_f32(1).unwrap();
        let mut owner = checker(&device);
        let private_input = owner.packet.view(1, 1).unwrap();
        for view in [
            input.view(0, 0).unwrap(),
            input.full_view(),
            foreign_input.full_view(),
            private_input,
        ] {
            let before = device.stats().unwrap();
            let events = evidence.lock().unwrap().events.len();
            assert!(owner.check(&view).is_err());
            assert_eq!(device.stats().unwrap(), before);
            assert_eq!(evidence.lock().unwrap().events.len(), events);
        }
        assert_eq!(owner.stats().status_reads, 0);
    }

    #[test]
    fn all_nonzero_status_bits_refuse_and_retry_without_poisoning() {
        let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 1024);
        let input = device.allocate_f32(65).unwrap();
        let mut owner = checker(&device);
        for (lane, status) in [
            (0, 1),
            (63, 1),
            (31, 0xffff_ffff),
            (5, 0x8000_0000),
            (42, 2),
        ] {
            evidence.lock().unwrap().statuses[lane] = status;
            assert!(owner.check(&input.full_view()).is_err());
            assert!(!device.stats().unwrap().poisoned);
            assert_eq!(owner.stats().accepted_checks, 0);
            evidence.lock().unwrap().statuses[lane] = 0;
        }
        owner.check(&input.full_view()).unwrap();
        assert_eq!(owner.stats().accepted_checks, 1);
        assert_eq!(owner.stats().status_reads, 6);
        assert_eq!(owner.stats().status_read_bytes, 1536);
    }

    #[test]
    fn every_cancellation_boundary_preserves_acceptance_and_has_exact_traffic() {
        let (device, _) = mock(NativeExecutionProfile::Qwen35Probe, 1024);
        let input = device.allocate_f32(1).unwrap();
        let mut owner = checker(&device);
        for stop in 1..=5 {
            let before = device.stats().unwrap();
            let prior = owner.stats();
            let mut calls = 0;
            assert!(owner
                .check_controlled(&input.full_view(), &mut || {
                    calls += 1;
                    if calls == stop {
                        Err(backend("cancelled"))
                    } else {
                        Ok(())
                    }
                })
                .is_err());
            let after = device.stats().unwrap();
            assert_eq!(calls, stop);
            assert_eq!(
                after.dispatch_count,
                before.dispatch_count + u64::from(stop > 2)
            );
            assert_eq!(
                after.host_read_bytes,
                before.host_read_bytes + if stop == 5 { 256 } else { 0 }
            );
            assert_eq!(
                after.host_write_bytes,
                before.host_write_bytes + if stop > 1 { 256 } else { 0 }
            );
            assert_eq!(
                owner.stats().pending_writes,
                prior.pending_writes + u64::from(stop > 1)
            );
            assert_eq!(
                owner.stats().pending_write_bytes,
                prior.pending_write_bytes + if stop > 1 { 256 } else { 0 }
            );
            assert_eq!(
                owner.stats().status_reads,
                prior.status_reads + u64::from(stop == 5)
            );
            assert_eq!(owner.stats().accepted_checks, 0);
        }
        owner.check(&input.full_view()).unwrap();
        assert_eq!(owner.stats().accepted_checks, 1);
    }

    #[test]
    fn transport_failures_permanently_poison_and_retain_uncertain_ranges() {
        for failure in 0..3 {
            let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 1024);
            let input = device.allocate_f32(1).unwrap();
            let mut owner = checker(&device);
            match failure {
                0 => evidence.lock().unwrap().fail_dispatch = true,
                1 => evidence.lock().unwrap().fail_read = true,
                _ => evidence.lock().unwrap().fail_write = true,
            }
            assert!(owner.check(&input.full_view()).is_err());
            assert!(device.stats().unwrap().poisoned);
            let before = device.stats().unwrap();
            assert!(owner.check(&input.full_view()).is_err());
            assert!(owner.diagnostic_read_guards().is_err());
            assert_eq!(device.stats().unwrap(), before);
            drop(owner);
            drop(input);
            assert_eq!(device.stats().unwrap().tensor_live_bytes, 280);
        }
        let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 1024);
        evidence.lock().unwrap().fail_write = true;
        assert!(NativeFiniteCheck::create(&device, || 18).is_err());
        assert!(device.stats().unwrap().poisoned);
        assert_eq!(device.stats().unwrap().tensor_live_bytes, 276);
    }

    #[test]
    fn skipped_second_dispatch_cannot_reuse_the_first_success() {
        let (device, evidence) = mock(NativeExecutionProfile::Qwen35Probe, 1024);
        let input = device.allocate_f32(1).unwrap();
        let mut owner = checker(&device);
        owner.check(&input.full_view()).unwrap();
        evidence.lock().unwrap().skip_dispatch_at = Some(2);
        let error = owner.check(&input.full_view()).unwrap_err();
        assert!(error.to_string().contains("status 0xffffffff"));
        assert_eq!(owner.stats().accepted_checks, 1);
        assert_eq!(owner.stats().pending_writes, 2);
        assert_eq!(owner.stats().status_reads, 2);
        assert!(!device.stats().unwrap().poisoned);
        owner.check(&input.full_view()).unwrap();
        assert_eq!(owner.stats().accepted_checks, 2);
    }
}
