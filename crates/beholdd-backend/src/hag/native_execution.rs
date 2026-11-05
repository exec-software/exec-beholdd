//! Experimental resident HAG execution: bounded linear, activation and attention.
//!
//! One native queue holds an immutable program library, a packed-weight arena, a tensor
//! arena and a bounded auxiliary arena for private scratch and immutable parameters.
//! A logical batch executes in list order with one synchronous physical submission
//! per overwrite linear/SwiGLU and two per scaled-add linear (matvec then add).
//! Unweighted RMSNorm uses one submission; weighted RMSNorm uses private resident
//! scratch and a following pointwise multiply stage. Parameters are immutable, while input
//! and weight tensor values are read when the recorded command executes.
//! RoPE reads resident coefficients and updates disjoint coordinate pairs in place.
//! Incremental causal attention reads resident caches without appending to them;
//! each query/head uses ordered QK, stable softmax and PV dispatches with private scratch.
//! Q8_0 embedding lookup decodes one selected packed row in one submission, retaining
//! a four-byte dummy for the unchanged three-descriptor ABI. Each batch shares one
//! private workspace across its ordered operators and interns immutable scalar bits.
//! The explicit Qwen35 probe profile replaces only its private library/admission:
//! Q4_K/Q6_K linear and row decode, plus full-width RMS up to 5120 elements whose
//! pointwise weight stage is sliced to the existing 4096-element multiply limit.
//! Dispatches contain descriptors only; a deduplicated batch table retains ownership.
//! Empty SwiGLU/RoPE are retained no-ops with no physical submission. Intermediate tensors stay
//! resident: submit performs no adapter host data transfers. This is not an optimized
//! serving backend, asynchronous executor, or complete model-operator implementation.
//!
//! Clones serialize all queue access. Resources, batches and completed submissions
//! retain the queue and its budget. Arena reuse does not release the preallocated
//! hardware budget; only final successful queue cleanup does. Failed/uncertain work
//! permanently disables reuse; HAG retains uncertain hardware allocations on cleanup.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};

use beholdd_format::GgmlType;
pub use exec_hag::budget::MemoryBudget;
use exec_hag::linux::compute::{
    shader_library_bytes, BufferRegion, ComputeQueue, KernelAbi, GFX1151_SERIAL_ABI, MAX_SLOT_BYTES,
};

use super::kernels::kquant::KQuantKind;
use super::kernels::{self, KernelKind};
use crate::execution::{
    sealed, Command, CommandList, EncodedWeight, ExecutionDevice, ExecutionError, ExecutionOwner,
    F32Tensor, F32TensorView, FiniteCheck, FiniteCheckSpec, LinearMode, PreparedWeight,
    RecordedBatch, Submission,
};
use crate::{DeviceError, IncrementalAttentionConfig, RopeLayout, Weight};

#[path = "native_delta.rs"]
pub mod delta;

#[path = "native_finite.rs"]
pub mod finite;

#[cfg(test)]
#[path = "native_sigmoid_tests.rs"]
mod native_sigmoid_tests;

const BACKEND: &str = "hag-native-experimental";
const MAX_ARENA_BYTES: u32 = 64 * 1024 * 1024;
const MAX_TENSOR_ARENA_BYTES: u32 = MAX_SLOT_BYTES;
const MAX_WEIGHT_ARENA_BYTES: u32 = 1024 * 1024 * 1024;
// Attention scores need up to 256 groups. Other programs retain their own
// narrower host and ISA admission limits inside this shared library.
const NATIVE_KERNEL_ABI: KernelAbi = match GFX1151_SERIAL_ABI.with_max_groups(256) {
    Some(abi) => abi,
    None => panic!("native attention workgroup limit exceeds HAG ABI"),
};
const WEIGHTS: usize = 0;
const TENSORS: usize = 1;
const AUXILIARY: usize = 2;
const DEFAULT_AUXILIARY_BYTES: u32 = 4096;
// Shared immutable storage bounds tensor initialization independently of cache
// capacity. This is not an additional per-tensor host allocation.
const F32_INIT_CHUNK_BYTES: usize = 64 * 1024;
static F32_INIT_ZEROES: [u8; F32_INIT_CHUNK_BYTES] = [0; F32_INIT_CHUNK_BYTES];

/// Explicit immutable kernel library and its matching admission rules.
///
/// The probe profile is not selected automatically by model metadata, and does
/// not advertise a complete Qwen35 runtime or a production serving capability.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NativeExecutionProfile {
    /// Existing BF16/Q8_0 dense runtime, with its unchanged 4096-element limits.
    #[default]
    DenseQwen3,
    /// Isolated Q4_K/Q6_K operator probes, including 5120-wide RMS/row lookup.
    Qwen35Probe,
}

impl NativeExecutionProfile {
    /// Inspect finite-validation support before opening a queue or allocating its
    /// arenas. This reads immutable profile/library metadata only: no program
    /// vectors, queue lock, allocation, device access or other I/O is required.
    ///
    /// Availability describes the real embedded program slot, not numerical or
    /// serving qualification. Preparation still validates the actual device and
    /// checker resources; this query does not certify a queue's current health.
    pub fn finite_check_spec(self) -> Option<FiniteCheckSpec> {
        // The typed count is the array length of qwen35_program_bytes(), never a
        // caller-supplied toggle. Do not materialize programs() just for metadata.
        if self != Self::Qwen35Probe || kernels::QWEN35_PROGRAM_COUNT <= 17 {
            return None;
        }
        Some(FiniteCheckSpec {
            max_elements: finite::NATIVE_FINITE_MAX_ELEMENTS,
            resident_tensor_bytes: u64::from(finite::NATIVE_FINITE_PACKET_BYTES),
            initialization_write_bytes: u64::from(finite::NATIVE_FINITE_PACKET_BYTES) + 16 + 256,
            // Status words plus the transport's simultaneous encoded staging.
            // Static pending template and recorder metadata are separate costs.
            host_scratch_bytes: 512,
            physical_dispatches: 1,
            logical_submissions: 1,
            pending_write_bytes: 256,
            status_read_bytes: 256,
        })
    }

    fn programs(self) -> Vec<Vec<u8>> {
        match self {
            Self::DenseQwen3 => kernels::program_bytes().into(),
            Self::Qwen35Probe => kernels::qwen35_program_bytes().into(),
        }
    }

    fn supports(self, kind: GgmlType) -> bool {
        match self {
            Self::DenseQwen3 => packed_kind(kind).is_some(),
            Self::Qwen35Probe => kquant_kind(kind).is_some(),
        }
    }

    fn packed_size(
        self,
        kind: GgmlType,
        rows: usize,
        cols: usize,
    ) -> Result<usize, ExecutionError> {
        let unsupported = || DeviceError::UnsupportedFormat {
            device: BACKEND,
            kind: kind.name(),
            rows,
            cols,
        };
        let bytes = match self {
            Self::DenseQwen3 => packed_kind(kind)
                .ok_or_else(unsupported)?
                .packed_bytes(rows, cols),
            Self::Qwen35Probe => kquant_kind(kind)
                .ok_or_else(unsupported)?
                .packed_bytes(rows, cols),
        };
        bytes.map_err(|message| ExecutionError::Shape(message.into()))
    }

    fn linear_program(self, kind: GgmlType) -> Result<usize, ExecutionError> {
        match (self, kind.0) {
            (Self::DenseQwen3, 30) | (Self::Qwen35Probe, 12) => Ok(0),
            (Self::DenseQwen3, 8) | (Self::Qwen35Probe, 14) => Ok(1),
            _ => Err(backend(
                "packed format does not belong to native execution profile",
            )),
        }
    }

    fn embedding_layout(
        self,
        kind: GgmlType,
        cols: usize,
    ) -> Result<(usize, usize, u32), ExecutionError> {
        match self {
            Self::DenseQwen3 if kind.0 == 8 => Ok((
                10,
                self.packed_size(kind, 1, cols)?,
                kernels::dispatch_groups(cols)
                    .map_err(|message| ExecutionError::Shape(message.into()))?,
            )),
            Self::Qwen35Probe => {
                let kernel = kquant_kind(kind).ok_or_else(|| {
                    backend("native Qwen35 probe embedding requires Q4_K or Q6_K; no CPU fallback")
                })?;
                let bytes = kernel
                    .row_bytes(cols)
                    .map_err(|message| ExecutionError::Shape(message.into()))?;
                Ok((
                    if kind.0 == 12 { 10 } else { 11 },
                    bytes,
                    self.wide_groups(cols)?,
                ))
            }
            _ => Err(backend(
                "native embedding rows require Q8_0; no CPU fallback",
            )),
        }
    }

    fn rms_limit(self) -> usize {
        match self {
            Self::DenseQwen3 => kernels::MAX_ELEMENTS,
            Self::Qwen35Probe => 5120,
        }
    }

    fn wide_groups(self, count: usize) -> Result<u32, ExecutionError> {
        if !(1..=self.rms_limit()).contains(&count) {
            return Err(ExecutionError::Shape(format!(
                "native {self:?} dispatch requires 1..={} elements",
                self.rms_limit()
            )));
        }
        Ok(count.div_ceil(kernels::THREADS[0] as usize) as u32)
    }
}

/// Fixed logical capacities; the queue separately charges page rounding and guards.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeExecutionConfig {
    /// Packed-weight bytes, 1..=1 GiB. Individual reservations are four-byte aligned.
    pub weight_capacity_bytes: u32,
    /// Tensor bytes, 1..=4 GiB minus 4 KiB. Each f32 element consumes four bytes.
    pub tensor_capacity_bytes: u32,
    /// Private scratch/parameter bytes, 1..=64 MiB; charged even when unused.
    pub auxiliary_capacity_bytes: u32,
}

impl Default for NativeExecutionConfig {
    fn default() -> Self {
        Self {
            weight_capacity_bytes: 64 * 1024,
            tensor_capacity_bytes: 64 * 1024,
            auxiliary_capacity_bytes: DEFAULT_AUXILIARY_BYTES,
        }
    }
}

impl NativeExecutionConfig {
    fn validate(self) -> Result<(), ExecutionError> {
        if !(1..=MAX_WEIGHT_ARENA_BYTES).contains(&self.weight_capacity_bytes)
            || !(1..=MAX_TENSOR_ARENA_BYTES).contains(&self.tensor_capacity_bytes)
            || !(1..=MAX_ARENA_BYTES).contains(&self.auxiliary_capacity_bytes)
        {
            return Err(ExecutionError::Shape(
                "native HAG weight capacity must be 1 byte..=1 GiB; tensor capacity must be 1 byte..=4 GiB minus 4 KiB and auxiliary capacity 1 byte..=64 MiB".into(),
            ));
        }
        Ok(())
    }

    /// Page-rounded GPU charge for this fixed three-arena queue, without device I/O.
    ///
    /// Includes the current page-rounded shader library, one command page and both
    /// guard pages around every arena.
    /// This is a cost query, not a reservation or a process-wide memory forecast:
    /// host staging and driver-private allocations are separate. The shared HAG
    /// budget still admits each actual allocation atomically when the queue opens.
    pub fn allocation_bytes(self) -> Result<u64, ExecutionError> {
        self.allocation_bytes_for_profile(NativeExecutionProfile::DenseQwen3)
    }

    /// Page-rounded cost of the explicitly selected immutable program library.
    /// The default cost query continues to use only the dense Qwen3 library.
    pub fn allocation_bytes_for_profile(
        self,
        profile: NativeExecutionProfile,
    ) -> Result<u64, ExecutionError> {
        self.validate()?;
        let programs = profile.programs();
        let references: Vec<_> = programs.iter().map(Vec::as_slice).collect();
        let shader_bytes = shader_library_bytes(&references).map_err(backend)?;
        [
            self.weight_capacity_bytes,
            self.tensor_capacity_bytes,
            self.auxiliary_capacity_bytes,
        ]
        .into_iter()
        .try_fold(shader_bytes + 4096, |total, size| {
            total
                .checked_add(u64::from(size).div_ceil(4096) * 4096 + 8192)
                .ok_or_else(|| ExecutionError::Shape("native allocation charge overflows".into()))
        })
    }
}

/// Peak counts for a sequential application's retained native recordings.
///
/// Count concurrently retained resources, including temporary lookup batches.
/// This describes host ownership metadata, not tensor payloads or GPU storage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeRecordingFootprint {
    /// Recorded batches simultaneously retained by the application.
    pub batches: u64,
    /// Physical dispatches across those batches, after semantic lowering.
    pub dispatches: u64,
    /// Distinct packed-weight allocations, including shared tiles only once.
    pub weight_allocations: u64,
    /// Distinct public F32 allocations, including norms, scratch and caches.
    pub tensor_allocations: u64,
    /// Distinct private scratch and parameter allocations retained by batches.
    pub auxiliary_allocations: u64,
    /// Sum of distinct allocation references in each batch's ownership table.
    /// A shared public allocation counts once for every batch that retains it.
    pub retained_allocation_references: u64,
    /// Extra allocation references retained by no-op semantic commands.
    pub noop_operands: u64,
    /// Largest semantic command list constructed at one time.
    pub max_batch_commands: u64,
    /// Largest physical dispatch list constructed at one time.
    pub max_batch_dispatches: u64,
}

impl NativeRecordingFootprint {
    /// Conservative checked host-metadata estimate without device access.
    ///
    /// Uses actual private structure sizes, Arc counters/alignment, and a 64-byte
    /// per-allocation bookkeeping allowance. Vec storage allows twice the element
    /// count plus four slots per batch, with an additional largest-batch workspace
    /// for simultaneous semantic/planned commands and dispatch-vector reallocation.
    /// Arena fragmentation during retirement is included. These are conservative
    /// estimates for the supported Rust/system allocator, not reservations or a
    /// portable allocator overhead guarantee. Payload staging, model-side metadata,
    /// tokenizer/HTTP state and a separate general process margin remain required.
    pub fn host_bytes(self) -> Result<u64, ExecutionError> {
        fn overflow() -> ExecutionError {
            ExecutionError::Shape("native recording host metadata overflows".into())
        }
        fn add(a: u64, b: u64) -> Result<u64, ExecutionError> {
            a.checked_add(b).ok_or_else(overflow)
        }
        fn mul(a: u64, b: u64) -> Result<u64, ExecutionError> {
            a.checked_mul(b).ok_or_else(overflow)
        }
        fn array<T>(slots: u64) -> Result<u64, ExecutionError> {
            add(mul(slots, std::mem::size_of::<T>() as u64)?, 64)
        }
        fn arc<T>() -> u64 {
            (std::mem::size_of::<T>()
                + 2 * std::mem::size_of::<usize>()
                + std::mem::align_of::<T>()
                - 1) as u64
                + 64
        }
        let allocations = add(
            add(self.weight_allocations, self.tensor_allocations)?,
            self.auxiliary_allocations,
        )?;
        let dispatch_slots = add(mul(self.dispatches, 2)?, mul(self.batches, 4)?)?;
        let retained_slots = add(
            mul(
                add(self.retained_allocation_references, self.noop_operands)?,
                2,
            )?,
            mul(self.batches, 4)?,
        )?;
        let command_slots = add(mul(self.max_batch_commands, 2)?, 4)?;
        let workspace_dispatches = add(mul(self.max_batch_dispatches, 2)?, 4)?;
        let free_spans = add(mul(allocations, 2)?, 12)?;
        [
            arc::<Inner>(),
            array::<NativeTransport>(1)?,
            mul(self.batches, arc::<NativeBatch>() + 2 * 64)?,
            array::<NativeDispatch>(dispatch_slots)?,
            array::<Arc<Allocation>>(retained_slots)?,
            mul(allocations, arc::<Allocation>())?,
            // Encoded and prepared wrappers coexist during preparation. Counting
            // both for every weight overestimates that one-at-a-time transition.
            mul(self.weight_allocations, 2 * arc::<NativeWeight>())?,
            mul(self.tensor_allocations, arc::<NativeTensor>())?,
            array::<Span>(free_spans)?,
            array::<Command>(command_slots)?,
            array::<PlannedCommand>(command_slots)?,
            array::<NativeDispatch>(workspace_dispatches)?,
            // The owner Vec can reallocate while a batch is recorded. Counting
            // all retained slots again bounds its largest old backing store.
            array::<Arc<Allocation>>(retained_slots)?,
            // One scalar key/descriptor per semantic command at most; the shared
            // workspace's zero staging is already part of payload staging.
            array::<(u32, BufferRegion)>(command_slots)?,
            arc::<CompletedSubmission>(),
        ]
        .into_iter()
        .try_fold(0, add)
    }
}

#[cfg(test)]
mod recording_footprint_tests {
    use super::*;

    #[test]
    fn vector_spare_capacity_allowance_covers_native_recording_element_layouts() {
        fn check<const WIDTH: usize>() {
            let mut values = Vec::<[u8; WIDTH]>::new();
            for len in 1..=512 {
                values.push([0; WIDTH]);
                assert!(values.capacity() <= 2 * len + 4);
            }
        }
        check::<{ std::mem::size_of::<NativeDispatch>() }>();
        check::<{ std::mem::size_of::<Command>() }>();
        check::<{ std::mem::size_of::<PlannedCommand>() }>();
        check::<{ std::mem::size_of::<Arc<Allocation>>() }>();
        check::<{ std::mem::size_of::<Span>() }>();
    }

    #[test]
    fn footprint_includes_layout_growth_and_rejects_every_count_overflow() {
        let baseline = NativeRecordingFootprint::default().host_bytes().unwrap();
        for field in 0..9 {
            let change = |n| {
                let mut counts = NativeRecordingFootprint::default();
                match field {
                    0 => counts.batches = n,
                    1 => counts.dispatches = n,
                    2 => counts.weight_allocations = n,
                    3 => counts.tensor_allocations = n,
                    4 => counts.auxiliary_allocations = n,
                    5 => counts.noop_operands = n,
                    6 => counts.retained_allocation_references = n,
                    7 => counts.max_batch_commands = n,
                    _ => counts.max_batch_dispatches = n,
                }
                counts
            };
            assert!(change(1).host_bytes().unwrap() > baseline);
            assert!(change(u64::MAX).host_bytes().is_err());
        }
        let one_dispatch = NativeRecordingFootprint {
            dispatches: 1,
            ..Default::default()
        }
        .host_bytes()
        .unwrap();
        assert_eq!(
            one_dispatch - baseline,
            2 * std::mem::size_of::<NativeDispatch>() as u64
        );
    }
}

/// Queue accounting. Transfer counters describe adapter data traffic only, not IB
/// uploads or HAG's guard checks. All cumulative counters saturate at u64::MAX.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NativeExecutionStats {
    /// Actual HAG page-rounded charge including programs, IB, guards and auxiliary arena.
    pub allocation_bytes: u64,
    /// Bytes currently reserved in the packed arena, including four-byte tail alignment.
    pub weight_live_bytes: u64,
    /// Bytes currently reserved in the tensor arena.
    pub tensor_live_bytes: u64,
    /// Private scratch and immutable parameter bytes retained by recorded batches.
    pub auxiliary_live_bytes: u64,
    /// Peak reserved bytes in the packed arena.
    pub weight_peak_bytes: u64,
    /// Peak reserved bytes in the tensor arena.
    pub tensor_peak_bytes: u64,
    /// Peak private scratch and immutable parameter bytes.
    pub auxiliary_peak_bytes: u64,
    /// Successfully uploaded packed checkpoint bytes; preparing weights adds nothing.
    pub uploaded_weight_bytes: u64,
    /// Successful host-to-arena bytes, including tensor/scratch zeros and parameters.
    pub host_write_bytes: u64,
    /// Successful explicit arena-to-host bytes.
    pub host_read_bytes: u64,
    /// Physical dispatch calls attempted, including failures.
    pub dispatch_attempts: u64,
    /// Successfully fenced and validated physical dispatches.
    pub dispatch_count: u64,
    /// Fully completed logical batches, including empty batches.
    pub logical_submissions: u64,
    /// Further allocation, recording, transfers and submission are refused.
    pub poisoned: bool,
}

/// An explicit, experimental `ExecutionDevice`; never selected as a CPU fallback.
#[derive(Clone)]
pub struct NativeExecutionDevice {
    inner: Arc<Inner>,
}

impl fmt::Debug for NativeExecutionDevice {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("NativeExecutionDevice")
            .field("config", &self.inner.config)
            .field("profile", &self.inner.profile)
            .finish_non_exhaustive()
    }
}

struct Inner {
    owner: ExecutionOwner,
    config: NativeExecutionConfig,
    profile: NativeExecutionProfile,
    state: Mutex<State>,
}

struct State {
    transport: Box<dyn Transport>,
    arenas: [Arena; 3],
    counters: NativeExecutionStats,
    poisoned: bool,
}

impl State {
    fn ready(&self) -> Result<(), ExecutionError> {
        if self.poisoned || self.transport.is_poisoned() {
            return Err(backend(
                "native HAG queue is poisoned; no automatic recovery or fallback",
            ));
        }
        Ok(())
    }

    fn observe<T>(&mut self, result: Result<T, ExecutionError>) -> Result<T, ExecutionError> {
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
}

/// Private transport seam: mock implementations exist only in hardware-free tests.
trait Transport: Send {
    fn allocation_bytes(&self) -> u64;
    fn is_poisoned(&self) -> bool;
    fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError>;
    fn read(&mut self, slot: usize, offset: u64, bytes: &mut [u8]) -> Result<(), ExecutionError>;
    fn dispatch(
        &mut self,
        program: usize,
        bindings: [BufferRegion; 3],
        scalars: [u32; 2],
        groups: u32,
    ) -> Result<u64, ExecutionError>;
}

struct NativeTransport(ComputeQueue);

impl Transport for NativeTransport {
    fn allocation_bytes(&self) -> u64 {
        self.0.allocation_bytes()
    }

    fn is_poisoned(&self) -> bool {
        self.0.is_poisoned()
    }

    fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
        self.0.write(slot, offset, bytes).map_err(backend)
    }

    fn read(&mut self, slot: usize, offset: u64, bytes: &mut [u8]) -> Result<(), ExecutionError> {
        self.0.read(slot, offset, bytes).map_err(backend)
    }

    fn dispatch(
        &mut self,
        program: usize,
        bindings: [BufferRegion; 3],
        scalars: [u32; 2],
        groups: u32,
    ) -> Result<u64, ExecutionError> {
        // SAFETY: only the immutable embedded library programs are reachable. Recording
        // validates their exact packed byte lengths, bounded shapes, f32 alignment
        // and disjoint descriptor extents. One to 256 wave64 groups cover the
        // output exactly; the immutable program masks the final partial group.
        unsafe { self.0.dispatch_regions(program, bindings, scalars, groups) }.map_err(backend)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    offset: u32,
    bytes: u32,
}

struct Arena {
    free: Vec<Span>,
    live: u64,
    peak: u64,
}

impl Arena {
    fn new(capacity: u32) -> Self {
        Self {
            free: vec![Span {
                offset: 0,
                bytes: capacity,
            }],
            live: 0,
            peak: 0,
        }
    }

    fn reserve(&mut self, logical_bytes: usize) -> Option<Span> {
        let bytes = u32::try_from(logical_bytes).ok()?.checked_add(3)? & !3;
        if bytes == 0 {
            return Some(Span {
                offset: 0,
                bytes: 0,
            });
        }
        let index = self.free.iter().position(|span| span.bytes >= bytes)?;
        let span = Span {
            offset: self.free[index].offset,
            bytes,
        };
        self.free[index].offset += bytes;
        self.free[index].bytes -= bytes;
        if self.free[index].bytes == 0 {
            self.free.remove(index);
        }
        self.live += u64::from(bytes);
        self.peak = self.peak.max(self.live);
        Some(span)
    }

    fn release(&mut self, span: Span) {
        if span.bytes == 0 {
            return;
        }
        let index = self.free.partition_point(|free| free.offset < span.offset);
        self.free.insert(index, span);
        let mut index = index.saturating_sub(1);
        while index + 1 < self.free.len() {
            if self.free[index].offset + self.free[index].bytes == self.free[index + 1].offset {
                let next = self.free.remove(index + 1);
                self.free[index].bytes += next.bytes;
            } else {
                index += 1;
            }
        }
        self.live -= u64::from(span.bytes);
    }
}

/// No state-to-allocation edges: allocations retain the queue without an Arc cycle.
struct Allocation {
    inner: Arc<Inner>,
    slot: usize,
    span: Span,
    logical_bytes: usize,
}

impl Drop for Allocation {
    fn drop(&mut self) {
        let mut state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                state
            }
        };
        // Never recycle a possibly in-flight range. The native queue retains all
        // arena backing allocations and quarantines uncertain cleanup itself.
        if state.ready().is_ok() {
            state.arenas[self.slot].release(self.span);
        }
    }
}

#[derive(Clone)]
struct NativeWeight {
    allocation: Arc<Allocation>,
    kind: GgmlType,
    rows: usize,
    cols: usize,
}

struct NativeTensor {
    allocation: Arc<Allocation>,
    len: usize,
}

struct NativeDispatch {
    program: usize,
    bindings: [BufferRegion; 3],
    scalars: [u32; 2],
    groups: u32,
}

struct NativeBatch {
    inner: Arc<Inner>,
    semantic_count: usize,
    commands: Vec<NativeDispatch>,
    // Every public/private allocation is retained once, including no-op operands.
    _retained: Vec<Arc<Allocation>>,
}

enum PlannedCommand {
    EmbeddingRow {
        program: usize,
        weight: BufferRegion,
        output: BufferRegion,
        allocations: [Arc<Allocation>; 2],
        cols: u32,
        groups: u32,
    },
    Linear {
        command: NativeDispatch,
        allocations: [Arc<Allocation>; 3],
        scale: Option<f32>,
    },
    SwiGlu {
        gate: BufferRegion,
        up: BufferRegion,
        allocations: [Arc<Allocation>; 2],
        limit: f32,
    },
    SigmoidGate {
        output: BufferRegion,
        input: BufferRegion,
        gate: BufferRegion,
        allocations: [Arc<Allocation>; 3],
    },
    RmsNorm {
        output: BufferRegion,
        input: BufferRegion,
        allocations: [Arc<Allocation>; 2],
        weight: Option<(BufferRegion, Arc<Allocation>)>,
        eps: f32,
    },
    Rope {
        bindings: [BufferRegion; 3],
        allocations: [Arc<Allocation>; 3],
        pairs: u32,
        flags: u32,
    },
    Attention {
        output: BufferRegion,
        q: BufferRegion,
        k: BufferRegion,
        v: BufferRegion,
        allocations: [Arc<Allocation>; 4],
        config: IncrementalAttentionConfig,
    },
}

impl PlannedCommand {
    fn workspace_bytes(&self) -> Result<usize, ExecutionError> {
        Ok(match self {
            Self::Linear {
                command,
                scale: Some(_),
                ..
            } => command.bindings[2].length as usize,
            Self::RmsNorm {
                input,
                weight: Some(_),
                ..
            } => input.length as usize,
            Self::Attention { config, .. } => config
                .query_start
                .checked_add(config.n_query_positions)
                .and_then(|end| end.checked_mul(12))
                .and_then(|bytes| bytes.checked_add(4))
                .ok_or_else(|| {
                    ExecutionError::Shape("native attention workspace overflows".into())
                })?,
            _ => 0,
        })
    }

    fn retain_operands(&self, resources: &mut RecordingResources) {
        let allocations: &[Arc<Allocation>] = match self {
            Self::EmbeddingRow { allocations, .. }
            | Self::SwiGlu { allocations, .. }
            | Self::RmsNorm { allocations, .. } => allocations,
            Self::Linear { allocations, .. }
            | Self::Rope { allocations, .. }
            | Self::SigmoidGate { allocations, .. } => allocations,
            Self::Attention { allocations, .. } => allocations,
        };
        for allocation in allocations {
            resources.retain(Arc::clone(allocation));
        }
        if let Self::RmsNorm {
            weight: Some((_, allocation)),
            ..
        } = self
        {
            resources.retain(Arc::clone(allocation));
        }
    }
}

/// Recording-local state only: no queue-to-allocation edge or mutable shared cache.
struct RecordingResources {
    workspace: Option<BufferRegion>,
    parameters: Vec<(u32, BufferRegion)>,
    retained: Vec<Arc<Allocation>>,
}

impl RecordingResources {
    fn new(
        device: &NativeExecutionDevice,
        plans: &[PlannedCommand],
    ) -> Result<Self, ExecutionError> {
        // All commands and workspace arithmetic are validated before allocation.
        let workspace_bytes = plans.iter().try_fold(0_usize, |bytes, plan| {
            plan.workspace_bytes().map(|required| bytes.max(required))
        })?;
        let mut resources = Self {
            workspace: None,
            parameters: Vec::new(),
            retained: Vec::new(),
        };
        for plan in plans {
            plan.retain_operands(&mut resources);
        }
        if workspace_bytes != 0 {
            let workspace = device.allocate(
                AUXILIARY,
                &zero_bytes(workspace_bytes, "native batch workspace")?,
            )?;
            resources.workspace = Some(NativeExecutionDevice::auxiliary_region(&workspace));
            resources.retain(workspace);
        }
        Ok(resources)
    }

    fn retain(&mut self, allocation: Arc<Allocation>) {
        if !self
            .retained
            .iter()
            .any(|existing| Arc::ptr_eq(existing, &allocation))
        {
            self.retained.push(allocation);
        }
    }

    fn parameter(
        &mut self,
        device: &NativeExecutionDevice,
        value: f32,
    ) -> Result<BufferRegion, ExecutionError> {
        let bits = value.to_bits();
        if let Some((_, region)) = self.parameters.iter().find(|(key, _)| *key == bits) {
            return Ok(*region);
        }
        let allocation = device.allocate(AUXILIARY, &bits.to_le_bytes())?;
        let region = NativeExecutionDevice::auxiliary_region(&allocation);
        self.retain(allocation);
        self.parameters.push((bits, region));
        Ok(region)
    }

    fn workspace_region(&self, bytes: usize) -> Result<BufferRegion, ExecutionError> {
        let mut region = self
            .workspace
            .ok_or_else(|| backend("missing planned batch workspace"))?;
        if bytes > region.length as usize {
            return Err(backend("planned batch workspace is too small"));
        }
        region.length = bytes as u32;
        Ok(region)
    }
}

struct CompletedSubmission {
    inner: Arc<Inner>,
    _batch: RecordedBatch,
}

fn backend(error: impl fmt::Display) -> ExecutionError {
    DeviceError::Backend {
        device: BACKEND,
        message: error.to_string(),
    }
    .into()
}

fn incompatible(resource: &'static str) -> ExecutionError {
    ExecutionError::IncompatibleResource {
        device: BACKEND,
        resource,
    }
}

fn packed_kind(kind: GgmlType) -> Option<KernelKind> {
    match kind.0 {
        30 => Some(KernelKind::Bf16),
        8 => Some(KernelKind::Q8_0),
        _ => None,
    }
}

fn kquant_kind(kind: GgmlType) -> Option<KQuantKind> {
    match kind.0 {
        12 => Some(KQuantKind::Q4K),
        14 => Some(KQuantKind::Q6K),
        _ => None,
    }
}

fn zero_bytes(len: usize, purpose: &'static str) -> Result<Vec<u8>, ExecutionError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| DeviceError::Allocation {
            device: BACKEND,
            bytes: len,
            purpose,
        })?;
    bytes.resize(len, 0);
    Ok(bytes)
}

fn overlaps(first: BufferRegion, second: BufferRegion) -> bool {
    first.slot == second.slot
        && first.offset < second.offset + u64::from(second.length)
        && second.offset < first.offset + u64::from(first.length)
}

impl NativeExecutionDevice {
    /// Open an explicit allowlisted native queue with resident fixed-size arenas.
    ///
    /// Only the embedded audited programs are accepted; arbitrary shader bytes are
    /// never exposed. Payload tracing is refused because it reads intermediate data.
    /// HAG enforces the caller's shared budget including guards and page rounding.
    pub fn open(
        path: impl AsRef<Path>,
        budget: &MemoryBudget,
        config: NativeExecutionConfig,
    ) -> Result<Self, ExecutionError> {
        Self::open_with_profile(path, budget, config, NativeExecutionProfile::DenseQwen3)
    }

    /// Open a queue with an explicit operator-probe profile. This does not widen
    /// default serving admission, and resources cannot cross profile/queue owners.
    pub fn open_with_profile(
        path: impl AsRef<Path>,
        budget: &MemoryBudget,
        config: NativeExecutionConfig,
        profile: NativeExecutionProfile,
    ) -> Result<Self, ExecutionError> {
        config.validate()?;
        if std::env::var("HAG_TRACE_SUBMISSION").as_deref() == Ok("1") {
            return Err(backend(
                "disable HAG_TRACE_SUBMISSION=1 before opening resident execution; payload tracing reads intermediate tensors",
            ));
        }
        if NATIVE_KERNEL_ABI
            != (KernelAbi {
                program_resources: [kernels::RSRC1, kernels::RSRC2, kernels::RSRC3],
                raw_descriptor: kernels::RAW_SRD_WORD3,
                threads: kernels::THREADS,
                max_groups: 256,
            })
        {
            return Err(backend("native kernel/queue ABI mismatch"));
        }
        let programs = profile.programs();
        let references: Vec<_> = programs.iter().map(Vec::as_slice).collect();
        // SAFETY: all immutable trusted programs obey the matched GFX1151 ABI;
        // upload is once per queue and every later dispatch is validated below.
        let queue = unsafe {
            ComputeQueue::open_programs_with_abi(
                path,
                budget,
                &references,
                [
                    config.weight_capacity_bytes,
                    config.tensor_capacity_bytes,
                    config.auxiliary_capacity_bytes,
                ],
                NATIVE_KERNEL_ABI,
            )
        }
        .map_err(backend)?;
        if queue.payload_tracing_enabled() {
            return Err(backend(
                "native queue captured payload tracing; disable HAG_TRACE_SUBMISSION and reopen",
            ));
        }
        Self::from_transport_with_profile(config, profile, Box::new(NativeTransport(queue)))
    }

    #[cfg(test)]
    fn from_transport(
        config: NativeExecutionConfig,
        transport: Box<dyn Transport>,
    ) -> Result<Self, ExecutionError> {
        Self::from_transport_with_profile(config, NativeExecutionProfile::DenseQwen3, transport)
    }

    fn from_transport_with_profile(
        config: NativeExecutionConfig,
        profile: NativeExecutionProfile,
        transport: Box<dyn Transport>,
    ) -> Result<Self, ExecutionError> {
        config.validate()?;
        let counters = NativeExecutionStats {
            allocation_bytes: transport.allocation_bytes(),
            ..Default::default()
        };
        Ok(Self {
            inner: Arc::new(Inner {
                owner: ExecutionOwner::fresh(),
                config,
                profile,
                state: Mutex::new(State {
                    transport,
                    arenas: [
                        Arena::new(config.weight_capacity_bytes),
                        Arena::new(config.tensor_capacity_bytes),
                        Arena::new(config.auxiliary_capacity_bytes),
                    ],
                    counters,
                    poisoned: false,
                }),
            }),
        })
    }

    /// The fixed library/admission profile selected when this queue was opened.
    pub fn profile(&self) -> NativeExecutionProfile {
        self.inner.profile
    }

    fn state(&self) -> Result<MutexGuard<'_, State>, ExecutionError> {
        let state = self
            .inner
            .state
            .lock()
            .map_err(|_| ExecutionError::Synchronization {
                device: BACKEND,
                resource: "native queue",
            })?;
        state.ready()?;
        Ok(state)
    }

    /// Snapshot counters without touching GPU memory; available even after poisoning.
    pub fn stats(&self) -> Result<NativeExecutionStats, ExecutionError> {
        let state = match self.inner.state.lock() {
            Ok(state) => state,
            Err(poisoned) => {
                let mut state = poisoned.into_inner();
                state.poisoned = true;
                state
            }
        };
        Ok(NativeExecutionStats {
            weight_live_bytes: state.arenas[WEIGHTS].live,
            tensor_live_bytes: state.arenas[TENSORS].live,
            auxiliary_live_bytes: state.arenas[AUXILIARY].live,
            weight_peak_bytes: state.arenas[WEIGHTS].peak,
            tensor_peak_bytes: state.arenas[TENSORS].peak,
            auxiliary_peak_bytes: state.arenas[AUXILIARY].peak,
            poisoned: state.poisoned || state.transport.is_poisoned(),
            ..state.counters
        })
    }

    fn allocate(&self, slot: usize, bytes: &[u8]) -> Result<Arc<Allocation>, ExecutionError> {
        let span = {
            let mut state = self.state()?;
            let span = state.arenas[slot]
                .reserve(bytes.len())
                .ok_or(DeviceError::Allocation {
                    device: BACKEND,
                    bytes: bytes.len(),
                    purpose: "native HAG arena range",
                })?;
            if !bytes.is_empty() {
                let result = state.transport.write(slot, u64::from(span.offset), bytes);
                state.observe(result)?;
                state.counters.host_write_bytes = state
                    .counters
                    .host_write_bytes
                    .saturating_add(bytes.len() as u64);
                if slot == WEIGHTS {
                    state.counters.uploaded_weight_bytes = state
                        .counters
                        .uploaded_weight_bytes
                        .saturating_add(bytes.len() as u64);
                }
            }
            span
        };
        Ok(Arc::new(Allocation {
            inner: Arc::clone(&self.inner),
            slot,
            span,
            logical_bytes: bytes.len(),
        }))
    }

    fn allocate_zeroed_tensor(&self, byte_len: usize) -> Result<Arc<Allocation>, ExecutionError> {
        let span = {
            let mut state = self.state()?;
            let span = state.arenas[TENSORS]
                .reserve(byte_len)
                .ok_or(DeviceError::Allocation {
                    device: BACKEND,
                    bytes: byte_len,
                    purpose: "native HAG arena range",
                })?;
            // Keep one reservation and the queue lock for the complete operation.
            // A failed write retains the entire range under the existing poison
            // policy; no partially initialized tensor owner can escape.
            for offset in (0..byte_len).step_by(F32_INIT_CHUNK_BYTES) {
                let count = (byte_len - offset).min(F32_INIT_CHUNK_BYTES);
                let result = state.transport.write(
                    TENSORS,
                    u64::from(span.offset) + offset as u64,
                    &F32_INIT_ZEROES[..count],
                );
                state.observe(result)?;
                state.counters.host_write_bytes =
                    state.counters.host_write_bytes.saturating_add(count as u64);
            }
            span
        };
        Ok(Arc::new(Allocation {
            inner: Arc::clone(&self.inner),
            slot: TENSORS,
            span,
            logical_bytes: byte_len,
        }))
    }

    fn validate_weight<'a>(
        &self,
        owner: ExecutionOwner,
        payload: Option<&'a NativeWeight>,
        geometry: (GgmlType, usize, usize, usize),
        resource: &'static str,
    ) -> Result<&'a NativeWeight, ExecutionError> {
        let (kind, rows, cols, byte_len) = geometry;
        self.inner.owner.require(owner, resource)?;
        let weight = payload.ok_or_else(|| incompatible(resource))?;
        if !Arc::ptr_eq(&self.inner, &weight.allocation.inner)
            || weight.allocation.slot != WEIGHTS
            || (kind, rows, cols, byte_len)
                != (
                    weight.kind,
                    weight.rows,
                    weight.cols,
                    weight.allocation.logical_bytes,
                )
            || self.inner.profile.packed_size(kind, rows, cols)? != byte_len
        {
            return Err(incompatible(resource));
        }
        Ok(weight)
    }

    fn tensor_region<'a>(
        &self,
        view: &'a F32TensorView,
        resource: &'static str,
    ) -> Result<(&'a NativeTensor, BufferRegion), ExecutionError> {
        self.inner.owner.require(view.owner(), resource)?;
        let tensor = view
            .tensor()
            .payload::<NativeTensor>()
            .ok_or_else(|| incompatible(resource))?;
        if !Arc::ptr_eq(&self.inner, &tensor.allocation.inner)
            || tensor.allocation.slot != TENSORS
            || tensor.len != view.tensor().len()
            || tensor.len.checked_mul(4) != Some(tensor.allocation.logical_bytes)
            || view
                .offset()
                .checked_add(view.len())
                .filter(|end| *end <= tensor.len)
                .is_none()
        {
            return Err(incompatible(resource));
        }
        Ok((
            tensor,
            BufferRegion {
                slot: TENSORS,
                offset: u64::from(tensor.allocation.span.offset) + (view.offset() as u64) * 4,
                length: u32::try_from(view.len() * 4).map_err(|_| incompatible(resource))?,
            },
        ))
    }

    // Validation does not allocate arena ranges or touch transport memory. The
    // complete command list is planned before lowering its first command.
    fn plan(&self, command: &Command) -> Result<PlannedCommand, ExecutionError> {
        if let Command::EmbeddingRow {
            output,
            weight,
            row,
        } = command
        {
            let packed = self.validate_weight(
                weight.owner(),
                weight.payload(),
                (
                    weight.kind(),
                    weight.rows(),
                    weight.cols(),
                    weight.byte_len(),
                ),
                "embedding prepared weight",
            )?;
            let (tensor, output_region) = self.tensor_region(output, "embedding output tensor")?;
            let (program, stride, groups) = self
                .inner
                .profile
                .embedding_layout(weight.kind(), packed.cols)?;
            if output.len() != packed.cols || *row >= packed.rows {
                return Err(ExecutionError::Shape(
                    "native embedding row/output shape differs from weight".into(),
                ));
            }
            let offset = row.checked_mul(stride).ok_or_else(|| {
                ExecutionError::Shape("native embedding row offset overflows".into())
            })?;
            if offset
                .checked_add(stride)
                .is_none_or(|end| end > packed.allocation.logical_bytes)
            {
                return Err(incompatible("embedding prepared weight extent"));
            }
            return Ok(PlannedCommand::EmbeddingRow {
                program,
                weight: BufferRegion {
                    slot: WEIGHTS,
                    offset: u64::from(packed.allocation.span.offset) + offset as u64,
                    length: stride as u32,
                },
                output: output_region,
                allocations: [
                    Arc::clone(&packed.allocation),
                    Arc::clone(&tensor.allocation),
                ],
                cols: packed.cols as u32,
                groups,
            });
        }
        if let Command::IncrementalCausalAttention {
            output,
            q,
            k_cache,
            v_cache,
            config,
        } = command
        {
            let (out_tensor, out_region) = self.tensor_region(output, "attention output tensor")?;
            let (q_tensor, q_region) = self.tensor_region(q, "attention query tensor")?;
            let (k_tensor, k_region) = self.tensor_region(k_cache, "attention key cache")?;
            let (v_tensor, v_region) = self.tensor_region(v_cache, "attention value cache")?;
            crate::execution::validate_incremental_attention_shape(
                output.len(),
                q.len(),
                k_cache.len(),
                v_cache.len(),
                *config,
            )?;
            if !(1..=256).contains(&config.head_dim)
                || !(1..=32).contains(&config.n_heads)
                || !(1..=32).contains(&config.n_kv_heads)
                || !(1..=256).contains(&config.n_query_positions)
                || !(1..=kernels::MAX_ATTENTION_POSITIONS).contains(&config.n_kv_positions)
            {
                return Err(ExecutionError::Shape(
                    "native attention requires 1..=256 head coordinates/query positions, 1..=16384 cache positions, 1..=32 heads and positive finite scale".into(),
                ));
            }
            let kv_width = config.n_kv_heads * config.head_dim;
            if kv_width > 4096 {
                return Err(ExecutionError::Shape(
                    "native attention requires KV width <=4096".into(),
                ));
            }
            if [q_region, k_region, v_region]
                .into_iter()
                .any(|region| overlaps(out_region, region))
                || overlaps(q_region, k_region)
            {
                return Err(backend(
                    "native HAG refuses attention output/input overlap and query/key descriptor overlap at record time",
                ));
            }
            // K and V are read-only and are never bound in the same physical
            // dispatch. The same is true of Q and V, so those aliases are safe.
            return Ok(PlannedCommand::Attention {
                output: out_region,
                q: q_region,
                k: k_region,
                v: v_region,
                allocations: [
                    Arc::clone(&out_tensor.allocation),
                    Arc::clone(&q_tensor.allocation),
                    Arc::clone(&k_tensor.allocation),
                    Arc::clone(&v_tensor.allocation),
                ],
                config: *config,
            });
        }
        if let Command::Rope {
            x,
            cos,
            sin,
            layout,
            inverse,
        } = command
        {
            let (x_tensor, x_region) = self.tensor_region(x, "RoPE coordinate tensor")?;
            let (cos_tensor, cos_region) = self.tensor_region(cos, "RoPE cosine tensor")?;
            let (sin_tensor, sin_region) = self.tensor_region(sin, "RoPE sine tensor")?;
            if cos.len().checked_mul(2) != Some(x.len())
                || sin.len() != cos.len()
                || cos.len() > kernels::MAX_ROPE_PAIRS
            {
                return Err(ExecutionError::Shape(
                    "native RoPE requires 0..=4096 coordinates and equal half-length coefficients"
                        .into(),
                ));
            }
            if overlaps(cos_region, sin_region)
                || overlaps(cos_region, x_region)
                || overlaps(sin_region, x_region)
            {
                return Err(backend(
                    "native HAG refuses overlapping RoPE coordinate/coefficient views at record time",
                ));
            }
            return Ok(PlannedCommand::Rope {
                bindings: [cos_region, sin_region, x_region],
                allocations: [
                    Arc::clone(&cos_tensor.allocation),
                    Arc::clone(&sin_tensor.allocation),
                    Arc::clone(&x_tensor.allocation),
                ],
                pairs: cos.len() as u32,
                flags: u32::from(*layout == RopeLayout::SplitHalf) | (u32::from(*inverse) << 1),
            });
        }
        if let Command::RmsNorm {
            output,
            input,
            weight,
            eps,
        } = command
        {
            if !eps.is_finite() || *eps < 0.0 {
                return Err(ExecutionError::Shape(
                    "native RMSNorm epsilon must be finite and non-negative".into(),
                ));
            }
            let (input_tensor, input_region) = self.tensor_region(input, "RMSNorm input tensor")?;
            let (output_tensor, output_region) =
                self.tensor_region(output, "RMSNorm output tensor")?;
            if !(1..=self.inner.profile.rms_limit()).contains(&input.len())
                || output.len() != input.len()
            {
                return Err(ExecutionError::Shape(format!(
                    "native RMSNorm requires equal input/output lengths of 1..={} elements",
                    self.inner.profile.rms_limit()
                )));
            }
            if overlaps(input_region, output_region) {
                return Err(backend(
                    "native HAG refuses overlapping RMSNorm input/output views at record time",
                ));
            }
            let weight = if let Some(weight) = weight {
                let (tensor, region) = self.tensor_region(weight, "RMSNorm weight tensor")?;
                if weight.len() != input.len() {
                    return Err(ExecutionError::Shape(
                        "native RMSNorm weight length must match input".into(),
                    ));
                }
                if overlaps(region, output_region) {
                    return Err(backend(
                        "native HAG refuses overlapping RMSNorm weight/output views at record time",
                    ));
                }
                // Input and weight may overlap: neither dispatch writes either.
                Some((region, Arc::clone(&tensor.allocation)))
            } else {
                None
            };
            return Ok(PlannedCommand::RmsNorm {
                output: output_region,
                input: input_region,
                allocations: [
                    Arc::clone(&input_tensor.allocation),
                    Arc::clone(&output_tensor.allocation),
                ],
                weight,
                eps: *eps,
            });
        }
        if let Command::SigmoidGate {
            output,
            input,
            gate,
        } = command
        {
            if self.profile() != NativeExecutionProfile::Qwen35Probe
                || kernels::QWEN35_PROGRAM_COUNT <= 16
            {
                return Err(backend(
                    "sigmoid gate is unavailable in this native program library",
                ));
            }
            let (out, out_region) = self.tensor_region(output, "sigmoid output tensor")?;
            let (src, in_region) = self.tensor_region(input, "sigmoid input tensor")?;
            let (g, gate_region) = self.tensor_region(gate, "sigmoid gate tensor")?;
            if output.len() != input.len()
                || input.len() != gate.len()
                || input.len() > kernels::MAX_ELEMENTS
                || overlaps(out_region, in_region)
                || overlaps(out_region, gate_region)
                || overlaps(in_region, gate_region)
            {
                return Err(ExecutionError::Shape(
                    "native sigmoid gate requires equal disjoint views of at most 4096 elements"
                        .into(),
                ));
            }
            return Ok(PlannedCommand::SigmoidGate {
                output: out_region,
                input: in_region,
                gate: gate_region,
                allocations: [
                    Arc::clone(&out.allocation),
                    Arc::clone(&src.allocation),
                    Arc::clone(&g.allocation),
                ],
            });
        }
        if let Command::SwiGlu { gate, up, limit } = command {
            if !limit.is_finite() || *limit < 0.0 {
                return Err(ExecutionError::Shape(
                    "native SwiGLU limit must be finite and non-negative".into(),
                ));
            }
            let (gate_tensor, gate_region) = self.tensor_region(gate, "SwiGLU gate tensor")?;
            let (up_tensor, up_region) = self.tensor_region(up, "SwiGLU up tensor")?;
            if gate.len() != up.len() || gate.len() > kernels::MAX_ELEMENTS {
                return Err(ExecutionError::Shape(
                    "native SwiGLU requires equal view lengths of 0..=4096 elements".into(),
                ));
            }
            if overlaps(gate_region, up_region) {
                return Err(backend(
                    "native HAG refuses overlapping SwiGLU gate/up views at record time",
                ));
            }
            return Ok(PlannedCommand::SwiGlu {
                gate: gate_region,
                up: up_region,
                allocations: [
                    Arc::clone(&gate_tensor.allocation),
                    Arc::clone(&up_tensor.allocation),
                ],
                limit: *limit,
            });
        }
        let Command::Linear {
            output,
            input,
            weight,
            mode,
        } = command
        else {
            return Err(backend(
                "native HAG does not implement this operation; no CPU fallback",
            ));
        };
        let scale = match *mode {
            LinearMode::Overwrite => None,
            LinearMode::ScaledAdd { scale } if scale.is_finite() => Some(scale),
            LinearMode::ScaledAdd { .. } => {
                return Err(ExecutionError::Shape(
                    "native scaled-add multiplier must be finite".into(),
                ));
            }
        };
        let packed = self.validate_weight(
            weight.owner(),
            weight.payload(),
            (
                weight.kind(),
                weight.rows(),
                weight.cols(),
                weight.byte_len(),
            ),
            "prepared weight",
        )?;
        let (input_tensor, input_region) = self.tensor_region(input, "linear input tensor")?;
        let (output_tensor, output_region) = self.tensor_region(output, "linear output tensor")?;
        if input.len() != packed.cols || output.len() != packed.rows {
            return Err(ExecutionError::Shape(
                "native linear input/output lengths do not match packed weight".into(),
            ));
        }
        if overlaps(input_region, output_region) {
            return Err(backend(
                "native HAG refuses overlapping linear input/output views at record time",
            ));
        }
        let weight_region = BufferRegion {
            slot: WEIGHTS,
            offset: u64::from(packed.allocation.span.offset),
            length: weight.byte_len() as u32,
        };
        Ok(PlannedCommand::Linear {
            command: NativeDispatch {
                program: self.inner.profile.linear_program(weight.kind())?,
                bindings: [weight_region, input_region, output_region],
                scalars: [packed.cols as u32, packed.rows as u32],
                groups: kernels::dispatch_groups(packed.rows)
                    .map_err(|message| ExecutionError::Shape(message.into()))?,
            },
            allocations: [
                Arc::clone(&packed.allocation),
                Arc::clone(&input_tensor.allocation),
                Arc::clone(&output_tensor.allocation),
            ],
            scale,
        })
    }

    fn auxiliary_region(allocation: &Allocation) -> BufferRegion {
        BufferRegion {
            slot: AUXILIARY,
            offset: u64::from(allocation.span.offset),
            length: allocation.logical_bytes as u32,
        }
    }

    // No queue guard is held here: a failed allocation may drop newly created
    // Allocation owners, whose Drop must reacquire the mutex to reclaim spans.
    fn lower(
        &self,
        plan: PlannedCommand,
        commands: &mut Vec<NativeDispatch>,
        resources: &mut RecordingResources,
    ) -> Result<(), ExecutionError> {
        // Operands were retained before lowering. Every operator fully overwrites
        // its workspace before reading it, and submit fences each dispatch before
        // the next operator may reuse that workspace. Parameters never overlap it.
        match plan {
            PlannedCommand::EmbeddingRow {
                program,
                weight,
                output,
                cols,
                groups,
                ..
            } => {
                let dummy = resources.parameter(self, 0.0)?;
                commands.push(NativeDispatch {
                    program,
                    bindings: [weight, dummy, output],
                    scalars: [cols, 1],
                    groups,
                });
            }
            PlannedCommand::Attention {
                output,
                q,
                k,
                v,
                config,
                ..
            } => {
                let end = config.query_start + config.n_query_positions;
                let scale = resources.parameter(self, config.scale)?;
                let workspace = resources.workspace_region(end * 12 + 4)?;
                let stride = config.n_kv_heads * config.head_dim;
                let group = config.n_heads / config.n_kv_heads;
                let region = |base: BufferRegion, offset: usize, length: usize| BufferRegion {
                    slot: base.slot,
                    offset: base.offset + offset as u64,
                    length: length as u32,
                };
                // Arena offsets are four-byte aligned. Padding keeps scores
                // aligned to eight bytes; probabilities follow the maximum score
                // extent, so they remain disjoint for every query in this command.
                let scores_padding = ((workspace.offset + 7) & !7) - workspace.offset;
                let probabilities_offset = scores_padding as usize + end * 8;
                for query in 0..config.n_query_positions {
                    let visible = config.query_start + query + 1;
                    let scores_region = region(workspace, scores_padding as usize, visible * 8);
                    let probabilities_region = region(workspace, probabilities_offset, visible * 4);
                    let visible_groups = kernels::attention_dispatch_groups(visible)
                        .map_err(|message| ExecutionError::Shape(message.into()))?;
                    for head in 0..config.n_heads {
                        let query_offset = (query * config.n_heads + head) * config.head_dim * 4;
                        let cache_offset = (head / group) * config.head_dim * 4;
                        // Stop at this head's final visible key, excluding future
                        // cache positions even if their public allocation is larger.
                        let cache_bytes = ((visible - 1) * stride + config.head_dim) * 4;
                        commands.push(NativeDispatch {
                            program: 7,
                            bindings: [
                                region(q, query_offset, config.head_dim * 4),
                                region(k, cache_offset, cache_bytes),
                                scores_region,
                            ],
                            scalars: [config.head_dim as u32, stride as u32],
                            groups: visible_groups,
                        });
                        commands.push(NativeDispatch {
                            program: 8,
                            bindings: [scale, scores_region, probabilities_region],
                            scalars: [visible as u32, 1],
                            // One wave preserves the ordered max/sum passes,
                            // then writes probabilities at lane + 64*k.
                            groups: 1,
                        });
                        commands.push(NativeDispatch {
                            program: 9,
                            bindings: [
                                probabilities_region,
                                region(v, cache_offset, cache_bytes),
                                region(output, query_offset, config.head_dim * 4),
                            ],
                            scalars: [visible as u32, stride as u32],
                            groups: kernels::dispatch_groups(config.head_dim)
                                .map_err(|message| ExecutionError::Shape(message.into()))?,
                        });
                    }
                }
            }
            PlannedCommand::Rope {
                bindings,
                pairs,
                flags,
                ..
            } => {
                if pairs != 0 {
                    commands.push(NativeDispatch {
                        program: 6,
                        bindings,
                        scalars: [pairs, 1 + flags],
                        groups: kernels::dispatch_groups(pairs as usize)
                            .map_err(|message| ExecutionError::Shape(message.into()))?,
                    });
                }
            }
            PlannedCommand::Linear {
                mut command, scale, ..
            } => {
                if let Some(scale) = scale {
                    let parameter = resources.parameter(self, scale)?;
                    let output_region = command.bindings[2];
                    let scratch = resources.workspace_region(output_region.length as usize)?;
                    command.bindings[2] = scratch;
                    let len = command.scalars[1];
                    let groups = command.groups;
                    commands.push(command);
                    commands.push(NativeDispatch {
                        program: 3,
                        bindings: [parameter, scratch, output_region],
                        scalars: [len, 1],
                        groups,
                    });
                } else {
                    commands.push(command);
                }
            }
            PlannedCommand::SwiGlu {
                gate, up, limit, ..
            } => {
                if gate.length != 0 {
                    let parameter = resources.parameter(self, limit)?;
                    commands.push(NativeDispatch {
                        program: 2,
                        bindings: [parameter, up, gate],
                        scalars: [gate.length / 4, 1],
                        groups: kernels::dispatch_groups(gate.length as usize / 4)
                            .map_err(|message| ExecutionError::Shape(message.into()))?,
                    });
                }
            }
            PlannedCommand::SigmoidGate {
                output,
                input,
                gate,
                ..
            } => {
                if input.length != 0 {
                    commands.push(NativeDispatch {
                        program: 16,
                        bindings: [gate, input, output],
                        scalars: [input.length / 4, 1],
                        groups: kernels::dispatch_groups(input.length as usize / 4)
                            .map_err(|message| ExecutionError::Shape(message.into()))?,
                    });
                }
            }
            PlannedCommand::RmsNorm {
                output,
                input,
                weight,
                eps,
                ..
            } => {
                let parameter = resources.parameter(self, eps)?;
                let scalars = [input.length / 4, 1];
                let groups = self.inner.profile.wide_groups(input.length as usize / 4)?;
                if let Some((weight_region, _)) = weight {
                    let scratch = resources.workspace_region(input.length as usize)?;
                    commands.push(NativeDispatch {
                        program: 4,
                        bindings: [parameter, input, scratch],
                        scalars,
                        groups,
                    });
                    // Normalization is one full-width reduction. Only the
                    // following pointwise multiply is sliced to the unchanged
                    // 4096-element shader limit; each slice has exact extents.
                    let len = input.length as usize / 4;
                    for first in (0..len).step_by(kernels::MAX_ELEMENTS) {
                        let count = (len - first).min(kernels::MAX_ELEMENTS);
                        let slice = |region: BufferRegion| BufferRegion {
                            slot: region.slot,
                            offset: region.offset + (first * 4) as u64,
                            length: (count * 4) as u32,
                        };
                        commands.push(NativeDispatch {
                            program: 5,
                            bindings: [slice(weight_region), slice(scratch), slice(output)],
                            scalars: [count as u32, 1],
                            groups: kernels::dispatch_groups(count)
                                .map_err(|message| ExecutionError::Shape(message.into()))?,
                        });
                    }
                } else {
                    commands.push(NativeDispatch {
                        program: 4,
                        bindings: [parameter, input, output],
                        scalars,
                        groups,
                    });
                }
            }
        }
        Ok(())
    }
}

impl sealed::Sealed for NativeExecutionDevice {}

impl ExecutionDevice for NativeExecutionDevice {
    fn name(&self) -> &'static str {
        BACKEND
    }

    fn supports(&self, kind: GgmlType) -> bool {
        self.inner.profile.supports(kind)
    }

    fn finite_check_spec(&self) -> Option<FiniteCheckSpec> {
        self.profile().finite_check_spec()
    }

    fn prepare_finite_check(&self) -> Result<FiniteCheck, ExecutionError> {
        if self.finite_check_spec().is_none() {
            return Err(backend(
                "native finite validation requires Qwen35Probe with immutable program17; no fallback",
            ));
        }
        let check = finite::NativeFiniteCheck::new(self)?;
        Ok(FiniteCheck::new(self.inner.owner, check))
    }

    fn check_finite(
        &self,
        check: &mut FiniteCheck,
        input: &F32TensorView,
        control: &mut dyn FnMut() -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        self.inner.owner.require(check.owner(), "finite checker")?;
        self.inner
            .owner
            .require(input.owner(), "finite-check input")?;
        if self.finite_check_spec().is_none() {
            return Err(backend(
                "native finite validation requires Qwen35Probe with immutable program17; no fallback",
            ));
        }
        let native = check
            .payload_mut::<finite::NativeFiniteCheck>()
            .ok_or_else(|| incompatible("finite checker"))?;
        // The private owner revalidates actual input storage/extent and rejects
        // private packet aliases before pending writes or dispatch. It retains
        // both allocations through fence/status read and preserves queue poison.
        native.check_controlled(input, control)
    }

    fn upload_weight(&self, weight: &Weight<'_>) -> Result<EncodedWeight, ExecutionError> {
        let expected = self
            .inner
            .profile
            .packed_size(weight.kind, weight.rows, weight.cols)?;
        if weight.bytes.len() != expected {
            return Err(ExecutionError::Shape(format!(
                "native packed weight needs {expected} bytes, got {}",
                weight.bytes.len()
            )));
        }
        let allocation = self.allocate(WEIGHTS, weight.bytes)?;
        Ok(EncodedWeight::new(
            self.inner.owner,
            weight.kind,
            weight.rows,
            weight.cols,
            expected,
            Arc::new(NativeWeight {
                allocation,
                kind: weight.kind,
                rows: weight.rows,
                cols: weight.cols,
            }),
        ))
    }

    fn prepare_weight(&self, weight: &EncodedWeight) -> Result<PreparedWeight, ExecutionError> {
        let packed = self.validate_weight(
            weight.owner(),
            weight.payload(),
            (
                weight.kind(),
                weight.rows(),
                weight.cols(),
                weight.byte_len(),
            ),
            "encoded weight",
        )?;
        let _state = self.state()?;
        Ok(PreparedWeight::new(
            self.inner.owner,
            weight.kind(),
            weight.rows(),
            weight.cols(),
            weight.byte_len(),
            Arc::new(packed.clone()),
        ))
    }

    fn allocate_f32(&self, len: usize) -> Result<F32Tensor, ExecutionError> {
        let byte_len = len
            .checked_mul(4)
            .filter(|bytes| *bytes <= self.inner.config.tensor_capacity_bytes as usize)
            .ok_or_else(|| {
                ExecutionError::Shape(
                    "native f32 allocation exceeds tensor arena capacity or overflows".into(),
                )
            })?;
        let allocation = self.allocate_zeroed_tensor(byte_len)?;
        Ok(F32Tensor::new(
            self.inner.owner,
            len,
            Arc::new(NativeTensor { allocation, len }),
        ))
    }

    fn write_f32(&self, destination: &F32TensorView, values: &[f32]) -> Result<(), ExecutionError> {
        let (_, region) = self.tensor_region(destination, "host-write tensor view")?;
        if values.len() != destination.len() {
            return Err(ExecutionError::Shape(
                "native host write length does not match tensor view".into(),
            ));
        }
        let mut bytes = zero_bytes(region.length as usize, "native host write staging")?;
        for (chunk, value) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *chunk = value.to_le_bytes();
        }
        let mut state = self.state()?;
        if !bytes.is_empty() {
            let result = state.transport.write(region.slot, region.offset, &bytes);
            state.observe(result)?;
            state.counters.host_write_bytes = state
                .counters
                .host_write_bytes
                .saturating_add(bytes.len() as u64);
        }
        Ok(())
    }

    fn read_f32(&self, source: &F32TensorView, values: &mut [f32]) -> Result<(), ExecutionError> {
        let (_, region) = self.tensor_region(source, "host-read tensor view")?;
        if values.len() != source.len() {
            return Err(ExecutionError::Shape(
                "native host read length does not match tensor view".into(),
            ));
        }
        let mut bytes = zero_bytes(region.length as usize, "native host read staging")?;
        let mut state = self.state()?;
        if !bytes.is_empty() {
            let result = state.transport.read(region.slot, region.offset, &mut bytes);
            state.observe(result)?;
            state.counters.host_read_bytes = state
                .counters
                .host_read_bytes
                .saturating_add(bytes.len() as u64);
        }
        for (value, chunk) in values.iter_mut().zip(bytes.as_chunks::<4>().0) {
            *value = f32::from_le_bytes(*chunk);
        }
        Ok(())
    }

    fn record(&self, commands: &CommandList) -> Result<RecordedBatch, ExecutionError> {
        if let Some(owner) = commands.owner() {
            self.inner.owner.require(owner, "command list")?;
        }
        let plans = {
            let _state = self.state()?;
            commands
                .commands()
                .iter()
                .map(|command| self.plan(command))
                .collect::<Result<Vec<_>, _>>()?
        };
        let mut resources = RecordingResources::new(self, &plans)?;
        let mut native = Vec::new();
        for plan in plans {
            self.lower(plan, &mut native, &mut resources)?;
        }
        Ok(RecordedBatch::new(
            self.inner.owner,
            commands.len(),
            Arc::new(NativeBatch {
                inner: Arc::clone(&self.inner),
                semantic_count: commands.len(),
                commands: native,
                _retained: resources.retained,
            }),
        ))
    }

    fn submit(&self, batch: &RecordedBatch) -> Result<Submission, ExecutionError> {
        self.inner.owner.require(batch.owner(), "recorded batch")?;
        let native = batch
            .payload::<NativeBatch>()
            .ok_or_else(|| incompatible("recorded batch"))?;
        if !Arc::ptr_eq(&self.inner, &native.inner) || native.semantic_count != batch.len() {
            return Err(incompatible("recorded batch"));
        }
        {
            let mut state = self.state()?;
            for command in &native.commands {
                state.counters.dispatch_attempts =
                    state.counters.dispatch_attempts.saturating_add(1);
                let result = state.transport.dispatch(
                    command.program,
                    command.bindings,
                    command.scalars,
                    command.groups,
                );
                state.observe(result)?;
                state.counters.dispatch_count = state.counters.dispatch_count.saturating_add(1);
            }
            state.counters.logical_submissions =
                state.counters.logical_submissions.saturating_add(1);
        }
        Ok(Submission::new(
            self.inner.owner,
            Arc::new(CompletedSubmission {
                inner: Arc::clone(&self.inner),
                _batch: batch.clone(),
            }),
        ))
    }

    fn wait(&self, submission: &Submission) -> Result<(), ExecutionError> {
        self.inner.owner.require(submission.owner(), "submission")?;
        let completed = submission
            .payload::<CompletedSubmission>()
            .ok_or_else(|| incompatible("submission"))?;
        if !Arc::ptr_eq(&self.inner, &completed.inner) {
            return Err(incompatible("submission"));
        }
        // submit already waited every native fence. Repeated waits replay this known
        // successful completion even if a later, separate submission poisons the queue.
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod profile_tests {
    use super::*;

    #[test]
    fn finite_profile_metadata_uses_real_library_without_opening_a_device() {
        assert_eq!(NativeExecutionProfile::DenseQwen3.finite_check_spec(), None);
        let actual = NativeExecutionProfile::Qwen35Probe.finite_check_spec();
        assert_eq!(actual.is_some(), kernels::QWEN35_PROGRAM_COUNT > 17);
        if kernels::QWEN35_PROGRAM_COUNT > 17 {
            assert_eq!(
                actual,
                Some(FiniteCheckSpec {
                    max_elements: 65_536,
                    resident_tensor_bytes: 276,
                    initialization_write_bytes: 548,
                    host_scratch_bytes: 512,
                    physical_dispatches: 1,
                    logical_submissions: 1,
                    pending_write_bytes: 256,
                    status_read_bytes: 256,
                })
            );
        } else {
            assert_eq!(actual, None);
        }
    }

    #[test]
    fn finite_instance_metadata_delegates_without_resources_transfers_or_health_claim() {
        for profile in [
            NativeExecutionProfile::DenseQwen3,
            NativeExecutionProfile::Qwen35Probe,
        ] {
            // Only the existing hardware-free recorder is constructed. No fake
            // program count/support override is supplied to either query.
            let device = recorder(profile);
            let before = device.stats().unwrap();
            assert_eq!(
                ExecutionDevice::finite_check_spec(&device),
                profile.finite_check_spec()
            );
            assert_eq!(device.stats().unwrap(), before);
            device.inner.state.lock().unwrap().poisoned = true;
            let poisoned = device.stats().unwrap();
            assert!(poisoned.poisoned);
            assert_eq!(
                ExecutionDevice::finite_check_spec(&device),
                profile.finite_check_spec()
            );
            assert_eq!(device.stats().unwrap(), poisoned);
        }
    }

    // Exercises admission and recording only. There is deliberately no CPU
    // imitation of the new ISA, and reaching submit/read is a test failure.
    struct RecordingTransport {
        capacities: [u32; 3],
        charge: u64,
    }
    impl Transport for RecordingTransport {
        fn allocation_bytes(&self) -> u64 {
            self.charge
        }
        fn is_poisoned(&self) -> bool {
            false
        }
        fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ExecutionError> {
            assert!(offset + bytes.len() as u64 <= u64::from(self.capacities[slot]));
            Ok(())
        }
        fn read(
            &mut self,
            _slot: usize,
            _offset: u64,
            _bytes: &mut [u8],
        ) -> Result<(), ExecutionError> {
            panic!("profile recording tests must not read device data")
        }
        fn dispatch(
            &mut self,
            _program: usize,
            _bindings: [BufferRegion; 3],
            _scalars: [u32; 2],
            _groups: u32,
        ) -> Result<u64, ExecutionError> {
            panic!("profile recording tests must not execute ISA")
        }
    }

    fn recorder(profile: NativeExecutionProfile) -> NativeExecutionDevice {
        let config = NativeExecutionConfig {
            weight_capacity_bytes: 256 * 1024,
            tensor_capacity_bytes: 256 * 1024,
            auxiliary_capacity_bytes: 64 * 1024,
        };
        NativeExecutionDevice::from_transport_with_profile(
            config,
            profile,
            Box::new(RecordingTransport {
                capacities: [
                    config.weight_capacity_bytes,
                    config.tensor_capacity_bytes,
                    config.auxiliary_capacity_bytes,
                ],
                charge: config.allocation_bytes_for_profile(profile).unwrap(),
            }),
        )
        .unwrap()
    }

    fn prepared(
        device: &NativeExecutionDevice,
        kind: u32,
        rows: usize,
        cols: usize,
    ) -> PreparedWeight {
        let kind = GgmlType(kind);
        let bytes = vec![0; device.profile().packed_size(kind, rows, cols).unwrap()];
        let encoded = device
            .upload_weight(&Weight {
                bytes: &bytes,
                kind,
                rows,
                cols,
            })
            .unwrap();
        device.prepare_weight(&encoded).unwrap()
    }

    #[test]
    fn explicit_profiles_keep_libraries_costs_and_format_admission_separate() {
        use NativeExecutionProfile::*;
        assert_eq!(NativeExecutionProfile::default(), DenseQwen3);
        assert_eq!(DenseQwen3.programs(), kernels::program_bytes());
        assert_eq!(DenseQwen3.programs().len(), 11);
        assert_eq!(Qwen35Probe.programs(), kernels::qwen35_program_bytes());
        assert_eq!(Qwen35Probe.programs().len(), 18);
        let config = NativeExecutionConfig {
            weight_capacity_bytes: 4097,
            tensor_capacity_bytes: 1,
            auxiliary_capacity_bytes: 1,
        };
        assert_eq!(config.allocation_bytes().unwrap(), 53_248);
        assert_eq!(
            config.allocation_bytes().unwrap(),
            config.allocation_bytes_for_profile(DenseQwen3).unwrap()
        );
        let library = Qwen35Probe.programs();
        let references: Vec<_> = library.iter().map(Vec::as_slice).collect();
        let charge = shader_library_bytes(&references).unwrap() + 4096 + 8192 + 4096 * 2 + 3 * 8192;
        assert_eq!(
            config.allocation_bytes_for_profile(Qwen35Probe).unwrap(),
            charge
        );
        for profile in [DenseQwen3, Qwen35Probe] {
            let device = recorder(profile);
            assert_eq!(device.profile(), profile);
            for kind in [0, 8, 12, 14, 30] {
                assert_eq!(
                    device.supports(GgmlType(kind)),
                    match profile {
                        DenseQwen3 => matches!(kind, 8 | 30),
                        Qwen35Probe => matches!(kind, 12 | 14),
                    }
                );
            }
        }
    }

    #[test]
    fn k_quant_matrix_and_row_geometries_are_independently_bounded() {
        use NativeExecutionProfile::*;
        for (kind, block_bytes, matvec, row) in [(12, 144, 0, 10), (14, 210, 1, 11)] {
            let kind = GgmlType(kind);
            assert_eq!(
                Qwen35Probe.packed_size(kind, 4096, 17408).unwrap(),
                4096 * 68 * block_bytes
            );
            assert_eq!(Qwen35Probe.linear_program(kind).unwrap(), matvec);
            assert_eq!(
                Qwen35Probe.embedding_layout(kind, 5120).unwrap(),
                (row, 20 * block_bytes, 80)
            );
            for (rows, cols) in [
                (0, 256),
                (4097, 256),
                (1, 0),
                (1, 255),
                (1, 257),
                (1, 17664),
            ] {
                assert!(Qwen35Probe.packed_size(kind, rows, cols).is_err());
            }
            for cols in [0, 255, 257, 5376, 17408] {
                assert!(Qwen35Probe.embedding_layout(kind, cols).is_err());
            }
            assert!(DenseQwen3.packed_size(kind, 1, 256).is_err());
            assert!(DenseQwen3.embedding_layout(kind, 256).is_err());
        }
        for kind in [GgmlType(8), GgmlType(30)] {
            assert!(Qwen35Probe.packed_size(kind, 1, 256).is_err());
            assert!(Qwen35Probe.linear_program(kind).is_err());
            assert!(DenseQwen3.packed_size(kind, 1, 5120).is_err());
        }
        assert_eq!(
            DenseQwen3.embedding_layout(GgmlType(8), 4096).unwrap(),
            (10, 128 * 34, 64)
        );
    }

    #[test]
    fn upload_prepare_and_foreign_resources_obey_the_selected_profile_before_side_effects() {
        use NativeExecutionProfile::*;
        let probe = recorder(Qwen35Probe);
        let dense = recorder(DenseQwen3);
        for (device, kind, bytes) in [(&probe, 8, 272), (&dense, 12, 144)] {
            let before = device.stats().unwrap();
            assert!(device
                .upload_weight(&Weight {
                    bytes: &vec![0; bytes],
                    kind: GgmlType(kind),
                    rows: 1,
                    cols: 256
                })
                .is_err());
            assert_eq!(device.stats().unwrap(), before);
        }
        let before = probe.stats().unwrap();
        assert!(probe
            .upload_weight(&Weight {
                bytes: &[0; 143],
                kind: GgmlType(12),
                rows: 1,
                cols: 256
            })
            .is_err());
        assert_eq!(probe.stats().unwrap(), before);
        let encoded = probe
            .upload_weight(&Weight {
                bytes: &[0; 144],
                kind: GgmlType(12),
                rows: 1,
                cols: 256,
            })
            .unwrap();
        let after_upload = probe.stats().unwrap();
        let weight = probe.prepare_weight(&encoded).unwrap();
        assert_eq!(probe.stats().unwrap(), after_upload);
        assert!(dense.prepare_weight(&encoded).is_err());
        assert!(recorder(Qwen35Probe).prepare_weight(&encoded).is_err());
        // A forged wrapper cannot relabel packed bytes into the other program.
        let packed = encoded.payload::<NativeWeight>().unwrap();
        let forged = EncodedWeight::new(
            encoded.owner(),
            GgmlType(14),
            1,
            256,
            144,
            Arc::new(packed.clone()),
        );
        assert!(probe.prepare_weight(&forged).is_err());
        assert_eq!(probe.stats().unwrap(), after_upload);
        let input = dense.allocate_f32(256).unwrap();
        let output = dense.allocate_f32(1).unwrap();
        let mut commands = CommandList::new();
        assert!(commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &weight,
                LinearMode::Overwrite
            )
            .is_err());
    }

    #[test]
    fn wide_k_quant_linear_uses_profile_program_and_bounded_scaled_add() {
        let device = recorder(NativeExecutionProfile::Qwen35Probe);
        for (kind, program) in [(12, 0), (14, 1)] {
            let weight = prepared(&device, kind, 3, 17408);
            let input = device.allocate_f32(17408).unwrap();
            let output = device.allocate_f32(3).unwrap();
            let mut list = CommandList::new();
            list.linear(
                &output.full_view(),
                &input.full_view(),
                &weight,
                LinearMode::ScaledAdd { scale: 0.5 },
            )
            .unwrap();
            let recorded = device.record(&list).unwrap();
            let native = recorded.payload::<NativeBatch>().unwrap();
            assert_eq!(native.commands.len(), 2);
            assert_eq!(native.commands[0].program, program);
            assert_eq!(native.commands[0].scalars, [17408, 3]);
            assert_eq!(native.commands[0].groups, 1);
            assert_eq!(
                native.commands[0].bindings[0].length as usize,
                weight.byte_len()
            );
            assert_eq!(native.commands[0].bindings[1].length, 17408 * 4);
            assert_eq!(native.commands[0].bindings[2].length, 3 * 4);
            assert_eq!(native.commands[1].program, 3);
            assert_eq!(native.commands[1].scalars, [3, 1]);
            assert_eq!(
                native.commands[1].bindings[1],
                native.commands[0].bindings[2]
            );
        }
    }

    #[test]
    fn k_quant_embedding_decodes_only_the_selected_exact_row_with_eighty_groups() {
        let device = recorder(NativeExecutionProfile::Qwen35Probe);
        for (kind, program, stride) in [(12, 10, 2880), (14, 11, 4200)] {
            let weight = prepared(&device, kind, 3, 5120);
            let output = device.allocate_f32(5120).unwrap();
            let allocation = &weight.payload::<NativeWeight>().unwrap().allocation;
            let mut list = CommandList::new();
            list.embedding_row(&output.full_view(), &weight, 2).unwrap();
            let recorded = device.record(&list).unwrap();
            let native = recorded.payload::<NativeBatch>().unwrap();
            assert_eq!(native.commands.len(), 1);
            let command = &native.commands[0];
            assert_eq!(command.program, program);
            assert_eq!(command.scalars, [5120, 1]);
            assert_eq!(command.groups, 80);
            assert_eq!(
                command.bindings[0],
                BufferRegion {
                    slot: WEIGHTS,
                    offset: u64::from(allocation.span.offset) + 2 * stride,
                    length: stride as u32
                }
            );
            assert_eq!(command.bindings[1].length, 4);
            assert_eq!(command.bindings[2].length, 5120 * 4);
        }
        let weight = prepared(&device, 12, 1, 5376);
        let output = device.allocate_f32(5376).unwrap();
        let mut list = CommandList::new();
        list.embedding_row(&output.full_view(), &weight, 0).unwrap();
        let before = device.stats().unwrap();
        assert!(device.record(&list).is_err());
        assert_eq!(device.stats().unwrap(), before);
    }

    #[test]
    fn wide_weighted_rms_reduces_once_and_only_slices_the_pointwise_multiply() {
        let device = recorder(NativeExecutionProfile::Qwen35Probe);
        for len in [4096, 4097, 5120] {
            let input = device.allocate_f32(len).unwrap();
            let output = device.allocate_f32(len).unwrap();
            let weight = device.allocate_f32(len).unwrap();
            for weighted in [false, true] {
                let input_view = input.full_view();
                let output_view = output.full_view();
                let weight_view = weight.full_view();
                let mut list = CommandList::new();
                list.rmsnorm(
                    &output_view,
                    &input_view,
                    weighted.then_some(&weight_view),
                    1e-6,
                )
                .unwrap();
                let recorded = device.record(&list).unwrap();
                let native = recorded.payload::<NativeBatch>().unwrap();
                assert_eq!(
                    native.commands.len(),
                    1 + if weighted { len.div_ceil(4096) } else { 0 }
                );
                let norm = &native.commands[0];
                assert_eq!(norm.program, 4);
                assert_eq!(norm.scalars, [len as u32, 1]);
                assert_eq!(norm.groups, len.div_ceil(64) as u32);
                assert_eq!(norm.bindings[1].length, len as u32 * 4);
                assert_eq!(norm.bindings[2].length, len as u32 * 4);
                for (slice, command) in native.commands[1..].iter().enumerate() {
                    let first = slice * 4096;
                    let count = (len - first).min(4096);
                    assert_eq!(command.program, 5);
                    assert_eq!(command.scalars, [count as u32, 1]);
                    assert_eq!(command.groups, count.div_ceil(64) as u32);
                    assert!(command
                        .bindings
                        .iter()
                        .all(|r| r.length == count as u32 * 4));
                    assert_eq!(
                        command.bindings[1].offset,
                        norm.bindings[2].offset + (first * 4) as u64
                    );
                    let (_, weight_region) = device.tensor_region(&weight_view, "test").unwrap();
                    let (_, output_region) = device.tensor_region(&output_view, "test").unwrap();
                    assert_eq!(
                        command.bindings[0].offset,
                        weight_region.offset + (first * 4) as u64
                    );
                    assert_eq!(
                        command.bindings[2].offset,
                        output_region.offset + (first * 4) as u64
                    );
                }
            }
        }
    }

    #[test]
    fn wide_probe_bounds_do_not_widen_swiglu_or_default_rms_and_refusal_is_transactional() {
        for (profile, len) in [
            (NativeExecutionProfile::DenseQwen3, 4097),
            (NativeExecutionProfile::Qwen35Probe, 5121),
        ] {
            let device = recorder(profile);
            let input = device.allocate_f32(len).unwrap();
            let output = device.allocate_f32(len).unwrap();
            let mut list = CommandList::new();
            list.rmsnorm(&output.full_view(), &input.full_view(), None, 1e-6)
                .unwrap();
            let before = device.stats().unwrap();
            assert!(device.record(&list).is_err());
            assert_eq!(device.stats().unwrap(), before);
        }
        let device = recorder(NativeExecutionProfile::Qwen35Probe);
        let input = device.allocate_f32(5120).unwrap();
        let output = device.allocate_f32(5120).unwrap();
        let mut list = CommandList::new();
        list.rmsnorm(
            &output.full_view(),
            &input.full_view(),
            Some(&input.full_view()),
            1e-6,
        )
        .unwrap();
        list.swiglu(&output.full_view(), &input.full_view(), 0.0)
            .unwrap();
        let before = device.stats().unwrap();
        assert!(device.record(&list).is_err());
        assert_eq!(
            device.stats().unwrap(),
            before,
            "late invalid command must not allocate RMS workspace/parameters"
        );
    }
}
