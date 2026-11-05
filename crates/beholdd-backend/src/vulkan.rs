//! Vulkan matvec bring-up for M4.
//!
//! This module is deliberately a prepared-matrix vertical slice rather than a premature
//! [`Device`](crate::Device) implementation. The current trait receives a borrowed host
//! byte slice for every matvec; implementing it literally would upload the same weight
//! and wait for the queue on every call. [`VulkanMatvec::prepare`] makes device residency an
//! explicit, owned fact and lets the benchmark measure both the one-time upload and the
//! steady-state kernel. The result is evidence for the tensor/residency seam M4 needs,
//! not a hidden CPU fallback wearing a Vulkan name.
//!
//! A failed submission/completion poisons its shared context; callers must create a
//! new context instead of retrying on possibly in-flight resources. During teardown,
//! an idle result other than success or device loss quarantines device resources and
//! loader ownership until process exit, with a diagnostic. This intentionally leaks
//! rather than destroying potentially in-use GPU objects; there is no automatic recovery.

use std::cell::Cell;
use std::ffi::CStr;
use std::io::{Cursor, Write};
use std::mem::ManuallyDrop;
use std::num::NonZeroUsize;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::{vk, Device, Entry, Instance};

use crate::{DeviceError, Weight};

const BACKEND: &str = "vulkan";
// Keep in sync with build.rs's `--target-env=vulkan1.2` shader compilation.
const REQUIRED_VULKAN_API_VERSION: u32 = vk::API_VERSION_1_2;
const Q8_0_KIND: u32 = 8;
const BF16_KIND: u32 = 30;
const Q8_0_ELEMENTS: usize = 32;
const Q8_0_BYTES: usize = 34;
const DESCRIPTOR_WORD_BYTES: usize = 4;
const DEFAULT_ARENA_BLOCK_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_ALLOCATION_HEADROOM: u32 = 64;
const ROUTED_MOE_PROGRAM_ALLOCATION_SLOTS: u32 = 5;
const ROUTED_MOE_TIMESTAMP_COUNT: u32 = 6;
const ROUTED_MOE_TILE4_INDIRECT_WORDS: usize = 6;
const ROUTED_MOE_TILE4_RECORD_WORDS: usize = 8;
const ROUTED_MOE_TILE4_WIDTH: usize = 4;

static NEXT_ARENA_ID: AtomicU64 = AtomicU64::new(1);

/// A Vulkan context with a persistent pipeline, command buffer, fence, and activation
/// buffers for packed matrix-vector products.
#[derive(Clone)]
pub struct VulkanMatvec {
    inner: Arc<Context>,
}

impl std::fmt::Debug for VulkanMatvec {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VulkanMatvec")
            .field("physical_device", &self.inner.device_name)
            .field("queue_family", &self.inner.queue_family)
            .finish_non_exhaustive()
    }
}

/// Configuration for a bounded, immutable Vulkan weight arena.
///
/// `budget_bytes` is deliberately mandatory. On a unified-memory APU the Vulkan heap
/// budget can be much larger than the process's real RAM/cgroup allowance, so the
/// model fit plan remains the authority for how many bytes this arena may reserve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightArenaConfig {
    /// Preferred size of each shared Vulkan buffer. Oversized matrices get a dedicated
    /// block instead.
    pub block_bytes: usize,
    /// Maximum sum of Vulkan memory-requirement sizes for all arena blocks.
    pub budget_bytes: u64,
    /// Vulkan allocation slots reserved for activations, KV cache, staging, and
    /// replacement buffers.
    pub allocation_headroom: u32,
}

impl WeightArenaConfig {
    /// Create a configuration with 512 MiB blocks and 64 allocation slots of headroom.
    pub const fn new(budget_bytes: u64) -> Self {
        Self {
            block_bytes: DEFAULT_ARENA_BLOCK_BYTES,
            budget_bytes,
            allocation_headroom: DEFAULT_ALLOCATION_HEADROOM,
        }
    }
}

/// Fit-authorized storage limits for one reusable routed-MoE program.
///
/// `budget_bytes` is a disjoint slice of the model's hard memory budget: callers must
/// not pass the complete model budget after separately authorizing weights, KV state, or
/// other working memory. The program charges Vulkan's actual memory-requirement sizes,
/// not only its logical activation lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutedMoeProgramConfig {
    /// Maximum sum of Vulkan memory-requirement sizes for all program-owned buffers.
    pub budget_bytes: u64,
    /// Vulkan allocation slots that must remain available after program construction.
    pub allocation_headroom: u32,
}

impl RoutedMoeProgramConfig {
    /// Create a configuration that preserves 64 Vulkan allocation slots for other work.
    pub const fn new(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            allocation_headroom: DEFAULT_ALLOCATION_HEADROOM,
        }
    }
}

/// Exact storage accounting for a reusable routed-MoE program.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RoutedMoeProgramStats {
    /// Logical input, output, metadata, and scratch bytes requested by all batch items.
    pub requested_bytes: u64,
    /// Vulkan buffer capacity requested after per-item descriptor alignment.
    pub buffer_bytes: u64,
    /// Bytes actually charged by Vulkan memory requirements.
    pub allocation_bytes: u64,
    /// Number of Vulkan device-memory allocation slots consumed by the program.
    pub allocation_slots: u32,
    /// Fit-authorized ceiling applied to `allocation_bytes` at construction.
    pub budget_bytes: u64,
}

/// Geometry used when checkpoint bytes are streamed directly into an arena slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatrixSpec {
    /// GGML storage kind.
    pub kind: beholdd_format::GgmlType,
    /// Matrix row count.
    pub rows: usize,
    /// Matrix column count.
    pub cols: usize,
}

impl MatrixSpec {
    /// Copy geometry from a borrowed weight without retaining its byte slice.
    pub fn from_weight(weight: &Weight<'_>) -> Self {
        Self {
            kind: weight.kind,
            rows: weight.rows,
            cols: weight.cols,
        }
    }
}

/// Opaque identifier returned while building an arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatrixId {
    arena_id: u64,
    index: u32,
}

/// Exact storage accounting for a sealed weight arena.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// Number of prepared matrices.
    pub matrices: usize,
    /// Number of Vulkan buffer/memory allocations.
    pub blocks: usize,
    /// Checkpoint bytes containing actual encoded weights.
    pub encoded_bytes: u64,
    /// Encoded bytes plus shader-readable four-byte tail padding.
    pub descriptor_bytes: u64,
    /// Gaps inserted so descriptor offsets satisfy the device alignment.
    pub alignment_bytes: u64,
    /// Unused block tails after sealing.
    pub tail_bytes: u64,
    /// Nominal Vulkan buffer capacity reserved by the arena.
    pub buffer_bytes: u64,
    /// Bytes actually charged by Vulkan memory requirements.
    pub allocation_bytes: u64,
}

/// GPU-side stage timings for one profiled paired-load routed-MoE execution.
///
/// The three kernel durations exclude both inter-stage dependency barriers. `total`
/// spans from immediately before gate/up to immediately after reduction, so it also
/// contains the two separately reported dependencies and timestamp-command overhead.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoutedMoeGpuProfile {
    /// Gate/up matrix products and SwiGLU activation.
    pub gate_up: Duration,
    /// Compute write-to-read dependency between gate/up and down.
    pub gate_to_down_dependency: Duration,
    /// Down-projection matrix products for every route.
    pub down: Duration,
    /// Compute write-to-read dependency between down and route reduction.
    pub down_to_reduce_dependency: Duration,
    /// Weighted route reduction into the output vector.
    pub reduce: Duration,
    /// Complete GPU interval covered by the six timestamps.
    pub total: Duration,
    /// Host time spent retrieving already-complete query results after the fence.
    pub host_query_readback: Duration,
    /// Number of meaningful low bits in this queue family's timestamp counter.
    pub counter_valid_bits: u32,
    /// Nanoseconds represented by one timestamp tick on the selected device.
    pub timestamp_period_ns: f32,
}

/// Paired-load dot-product implementation used by an explicit routed-MoE experiment.
///
/// [`Self::QualifiedTree64`] is the portable control; [`Self::Tree128`] is promoted for
/// the qualified H=2048/F=512/K=8 shape on the measured GFX1151 driver. The variants
/// stay explicit so benchmarks can qualify a replacement without silently changing
/// ordinary model execution elsewhere.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoutedMoeFastVariant {
    /// Qualified 64-lane shared-memory reduction tree.
    QualifiedTree64,
    /// 32-lane shared-memory reduction tree.
    Tree32,
    /// 128-lane shared-memory tree, promoted for the qualified GFX1151 routed shape.
    Tree128,
    /// 32-lane subgroup-arithmetic reduction.
    Subgroup32,
    /// 64-lane subgroup-arithmetic reduction.
    Subgroup64,
    /// 128-lane subgroup-arithmetic reduction.
    Subgroup128,
}

/// Command-recording strategy for a fixed-size routed-MoE batch.
///
/// [`Self::PerItem`] is the qualified control: it records one dispatch per item for
/// each of the three stages. [`Self::ItemDimension`] is an explicit Tree128 candidate
/// that binds whole, tightly packed batch buffers and maps the independent item to
/// `gl_WorkGroupID.z`, reducing `3*N` dispatches to three without changing floating-
/// point arithmetic or route order. [`Self::ExpertTile4`] is a separate explicit
/// candidate that groups as many as four occurrences of one expert so a workgroup can
/// reuse its weight loads. Neither candidate is selected by the default constructor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RoutedMoeBatchDispatch {
    /// Record `N` gate/up, `N` down, and `N` route-reduction dispatches.
    PerItem,
    /// Record one three-dimensional dispatch for each of the three stages.
    ItemDimension,
    /// Group up to four equal expert occurrences and dispatch only the occupied tiles.
    ///
    /// This remains a benchmark-only experiment after failing its repeatability gate;
    /// callers must opt in through the explicit-dispatch constructor.
    ExpertTile4,
}

impl RoutedMoeBatchDispatch {
    /// Stable human-readable label used in qualification records.
    pub const fn label(self) -> &'static str {
        match self {
            Self::PerItem => "per-item 3*N",
            Self::ItemDimension => "item-dimensional 3-dispatch",
            Self::ExpertTile4 => "expert-major Tile4 indirect",
        }
    }
}

impl RoutedMoeFastVariant {
    /// Stable human-readable label used in benchmark records.
    pub const fn label(self) -> &'static str {
        match self {
            Self::QualifiedTree64 => "tree-wg64 (qualified)",
            Self::Tree32 => "tree-wg32",
            Self::Tree128 => "tree-wg128",
            Self::Subgroup32 => "subgroup-wg32",
            Self::Subgroup64 => "subgroup-wg64",
            Self::Subgroup128 => "subgroup-wg128",
        }
    }

    const fn workgroup_size(self) -> u32 {
        match self {
            Self::Tree32 | Self::Subgroup32 => 32,
            Self::QualifiedTree64 | Self::Subgroup64 => 64,
            Self::Tree128 | Self::Subgroup128 => 128,
        }
    }

    const fn uses_subgroup_arithmetic(self) -> bool {
        matches!(
            self,
            Self::Subgroup32 | Self::Subgroup64 | Self::Subgroup128
        )
    }

    const fn experimental_index(self) -> Option<usize> {
        match self {
            Self::QualifiedTree64 => None,
            Self::Tree32 => Some(0),
            Self::Tree128 => Some(1),
            Self::Subgroup32 => Some(2),
            Self::Subgroup64 => Some(3),
            Self::Subgroup128 => Some(4),
        }
    }
}

/// Mutable upload phase for an immutable, shared Vulkan weight arena.
pub struct WeightArenaBuilder {
    context: Arc<Context>,
    config: WeightArenaConfig,
    arena_id: u64,
    blocks: Vec<ArenaBlockBuilder>,
    records: Vec<MatrixRecord>,
    stats: ArenaStats,
}

impl std::fmt::Debug for WeightArenaBuilder {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("WeightArenaBuilder")
            .field("physical_device", &self.context.device_name)
            .field("config", &self.config)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

/// A sealed collection of packed matrices sharing a small number of Vulkan
/// allocations.
#[derive(Clone)]
pub struct PreparedWeightArena {
    inner: Arc<ArenaStorage>,
}

impl std::fmt::Debug for PreparedWeightArena {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedWeightArena")
            .field("physical_device", &self.inner.context.device_name)
            .field("stats", &self.inner.stats)
            .finish_non_exhaustive()
    }
}

/// One packed matrix uploaded once and retained in Vulkan-visible memory.
#[derive(Clone)]
pub struct PreparedMatrix {
    arena: Arc<ArenaStorage>,
    id: MatrixId,
}

/// Three BF16 projections forming one routed SwiGLU expert.
#[derive(Debug, Clone, Copy)]
pub struct Bf16Expert<'a> {
    /// `[expert_width, hidden]` gate projection.
    pub gate: &'a PreparedMatrix,
    /// `[expert_width, hidden]` up projection.
    pub up: &'a PreparedMatrix,
    /// `[hidden, expert_width]` down projection.
    pub down: &'a PreparedMatrix,
}

/// Prepared expert views for one deterministic routed-MoE batch.
///
/// The first prototype requires all selected projections to share one arena block. A
/// fully resident 256-expert layer can later group routes by block or use descriptor
/// indexing; exact streaming can already pack its selected working set together.
#[derive(Clone)]
pub struct PreparedRoutedExperts {
    arena: Arc<ArenaStorage>,
    block: u32,
    weight_offset: usize,
    weight_range: usize,
    hidden: usize,
    expert_width: usize,
    offsets_bytes: Box<[[u32; 3]]>,
}

/// A reusable, shape-specialized BF16 routed-MoE dispatch program.
///
/// The program owns private activation buffers, descriptors, command buffers, and a
/// fence. Its two command buffers are recorded once for overwrite and accumulation
/// respectively; each execution only copies input and route metadata, submits the
/// selected buffer, waits, and copies the result back. The weight arena and exact
/// paired-load variant selected at construction remain pinned for the program's full
/// lifetime.
pub struct RoutedMoeProgram {
    context: Arc<Context>,
    arena: Arc<ArenaStorage>,
    block: u32,
    batch_len: usize,
    hidden: usize,
    expert_width: usize,
    routes: usize,
    fast_variant: RoutedMoeFastVariant,
    batch_dispatch: RoutedMoeBatchDispatch,
    weight_capacity: usize,
    geometry: RoutedMoeProgramGeometry,
    stats: RoutedMoeProgramStats,
    state: Mutex<RoutedMoeProgramState>,
}

impl std::fmt::Debug for RoutedMoeProgram {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RoutedMoeProgram")
            .field("physical_device", &self.context.device_name)
            .field("arena_id", &self.arena.arena_id)
            .field("block", &self.block)
            .field("batch_len", &self.batch_len)
            .field("hidden", &self.hidden)
            .field("expert_width", &self.expert_width)
            .field("routes", &self.routes)
            .field("fast_variant", &self.fast_variant)
            .field("batch_dispatch", &self.batch_dispatch)
            .field("stats", &self.stats)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for PreparedRoutedExperts {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PreparedRoutedExperts")
            .field("physical_device", &self.arena.context.device_name)
            .field("routes", &self.offsets_bytes.len())
            .field("hidden", &self.hidden)
            .field("expert_width", &self.expert_width)
            .field("weight_range", &self.weight_range)
            .finish_non_exhaustive()
    }
}

impl PreparedRoutedExperts {
    /// Number of routed experts, preserving caller order.
    pub fn routes(&self) -> usize {
        self.offsets_bytes.len()
    }

    /// Input and output hidden width.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Intermediate width of each routed expert.
    pub fn expert_width(&self) -> usize {
        self.expert_width
    }
}

impl std::fmt::Debug for PreparedMatrix {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let record = self.record();
        formatter
            .debug_struct("PreparedMatrix")
            .field("physical_device", &self.arena.context.device_name)
            .field("rows", &record.rows)
            .field("cols", &record.cols)
            .field("format", &record.format)
            .field("encoded_bytes", &self.encoded_bytes())
            .finish()
    }
}

impl PreparedMatrix {
    fn record(&self) -> &MatrixRecord {
        &self.arena.records[self.id.index as usize]
    }

    /// Matrix row count.
    pub fn rows(&self) -> usize {
        self.record().rows
    }

    /// Matrix column count.
    pub fn cols(&self) -> usize {
        self.record().cols
    }

    /// Encoded bytes retained by this allocation.
    pub fn encoded_bytes(&self) -> usize {
        self.record().encoded_len
    }

    /// GGML storage-format name for this matrix.
    pub fn format_name(&self) -> &'static str {
        self.record().format.name()
    }
}

#[derive(Debug, Clone, Copy)]
struct MatrixRecord {
    block: u32,
    offset: usize,
    descriptor_len: usize,
    encoded_len: usize,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    format: PreparedFormat,
}

struct ArenaStorage {
    context: Arc<Context>,
    arena_id: u64,
    blocks: Box<[BufferAllocation]>,
    records: Box<[MatrixRecord]>,
    stats: ArenaStats,
}

impl Drop for ArenaStorage {
    fn drop(&mut self) {
        let state = lock_unpoisoned(&self.context.state);
        // Synchronous calls normally make this redundant, but it also covers the
        // device-lost/error path where queue submission succeeded and fence waiting did
        // not. Future asynchronous submissions must retain this Arc until completion.
        if !context_idle_for_teardown(&self.context, &state, "weight arena") {
            return;
        }
        unsafe {
            for block in std::mem::take(&mut self.blocks).into_vec() {
                block.destroy(&self.context.device, &self.context.live_allocations);
            }
        }
    }
}

struct ArenaBlockBuilder {
    context: Arc<Context>,
    allocation: Option<BufferAllocation>,
    mapped: Option<NonNull<u8>>,
    used: usize,
}

impl Drop for ArenaBlockBuilder {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            unsafe {
                if self.mapped.take().is_some() {
                    self.context.device.unmap_memory(allocation.memory);
                }
                allocation.destroy(&self.context.device, &self.context.live_allocations);
            }
        }
    }
}

struct Context {
    // An uncertain teardown intentionally retains the loader as well as device
    // objects. Unloading it while leaked device work remains pending is not safe.
    _entry: ManuallyDrop<Entry>,
    instance: Instance,
    device: Device,
    device_name: String,
    driver_version: u32,
    queue_family: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    max_group_count: [u32; 3],
    max_work_group_invocations: u32,
    max_work_group_size_x: u32,
    subgroup_arithmetic: bool,
    max_storage_buffer_range: usize,
    storage_buffer_alignment: usize,
    max_memory_allocations: u32,
    timestamp_properties: Option<TimestampProperties>,
    live_allocations: AtomicU32,
    state: Mutex<QueueState>,
}

struct QueueState {
    health: QueueHealth,
    queue: vk::Queue,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets: [vk::DescriptorSet; 4],
    pipeline_layout: vk::PipelineLayout,
    q8_pipeline: vk::Pipeline,
    bf16_pipeline: vk::Pipeline,
    moe_gate_up_pipeline: vk::Pipeline,
    moe_down_pipeline: vk::Pipeline,
    moe_reduce_pipeline: vk::Pipeline,
    moe_gate_up_fast_pipeline: vk::Pipeline,
    moe_down_fast_pipeline: vk::Pipeline,
    moe_reduce_fast_pipeline: vk::Pipeline,
    experimental_fast_pipelines: [Option<RoutedMoePipelinePair>; 5],
    item_dimensional_fast_pipelines: Option<RoutedMoePipelineTriple>,
    tile4_fast_pipelines: Option<RoutedMoePipelinePair>,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
    timestamp_query_pool: Option<vk::QueryPool>,
    input: Option<BufferAllocation>,
    output: Option<BufferAllocation>,
    moe_metadata: Option<BufferAllocation>,
    moe_hidden: Option<BufferAllocation>,
    moe_routes: Option<BufferAllocation>,
}

/// Shared by every submission on a context, including prerecorded programs.
/// Access is serialized by `Context::state`; interior mutability lets a submission
/// borrow its buffer descriptors while changing only the queue's health state.
#[derive(Default)]
struct QueueHealth {
    uncertain: Cell<bool>,
}

impl QueueHealth {
    fn require_ready(&self) -> Result<(), DeviceError> {
        if self.uncertain.get() {
            return Err(backend_error(
                "the Vulkan context is poisoned after an uncertain submission; create a new context before continuing",
            ));
        }
        Ok(())
    }

    fn submit_and_wait(
        &self,
        submit: impl FnOnce() -> Result<(), DeviceError>,
        wait: impl FnOnce() -> Result<(), DeviceError>,
    ) -> Result<(), DeviceError> {
        self.require_ready()?;
        // Do not reuse mapped buffers, descriptors, fences, or command buffers unless
        // submission AND completion were both observed. This also fails closed when
        // either callback panics. Cleanup still waits for the device before freeing
        // resources that may be in flight; it never restores this context to service.
        self.uncertain.set(true);
        submit()?;
        wait()?;
        self.uncertain.set(false);
        Ok(())
    }
}

/// DEVICE_LOST is equivalent to success for resource-in-use/pending-state decisions
/// (Vulkan specification, "Lost Device"). OOM and other errors offer no such promise.
fn teardown_is_safe(
    completion: Result<(), vk::Result>,
    quarantine: impl FnOnce(vk::Result),
) -> bool {
    match completion {
        Ok(()) | Err(vk::Result::ERROR_DEVICE_LOST) => true,
        Err(error) => {
            quarantine(error);
            false
        }
    }
}

fn context_idle_for_teardown(context: &Arc<Context>, state: &QueueState, owner: &str) -> bool {
    let completion = unsafe { context.device.device_wait_idle() };
    if completion.is_err() {
        state.health.uncertain.set(true);
    }
    teardown_is_safe(completion, |error| {
        // Raw Vulkan handles have no automatic Drop. Retain the owner chain as well,
        // so its device/instance/loader cannot disappear after these handles leak.
        // This is process-lifetime quarantine, not automatic recovery or a retry.
        std::mem::forget(Arc::clone(context));
        report_teardown_quarantine(owner, error);
    })
}

fn report_teardown_quarantine(owner: &str, error: vk::Result) {
    let _ = writeln!(
        std::io::stderr().lock(),
        "beholdd Vulkan: quarantining {owner} and its device owner until process exit: device idle failed with {error:?}; completion is unknown, resources will not be destroyed and this context cannot be reused"
    );
}

struct RoutedMoeProgramState {
    resources: RoutedMoeProgramResources,
    poisoned: bool,
}

#[derive(Default)]
struct RoutedMoeProgramResources {
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Box<[vk::DescriptorSet]>,
    command_pool: vk::CommandPool,
    command_buffers: [vk::CommandBuffer; 2],
    fence: vk::Fence,
    input: Option<PersistentlyMappedBuffer>,
    output: Option<PersistentlyMappedBuffer>,
    metadata: Option<PersistentlyMappedBuffer>,
    hidden: Option<BufferAllocation>,
    routes: Option<BufferAllocation>,
}

struct PersistentlyMappedBuffer {
    allocation: BufferAllocation,
    mapped: NonNull<u8>,
}

// SAFETY: the pointer denotes a stable Vulkan mapping owned by the allocation. All
// access is through `RoutedMoeProgramState`, which is serialized by its mutex, and a
// fence completes device access before host reads or subsequent host writes.
unsafe impl Send for PersistentlyMappedBuffer {}

struct RoutedMoeProgramInit {
    context: Arc<Context>,
    _arena: Arc<ArenaStorage>,
    resources: RoutedMoeProgramResources,
    armed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeProgramGeometry {
    hidden: u32,
    expert_width: u32,
    routes: u32,
    batch_len: usize,
    input: RoutedMoeBufferGeometry,
    output: RoutedMoeBufferGeometry,
    metadata: RoutedMoeBufferGeometry,
    hidden_scratch: RoutedMoeBufferGeometry,
    route_scratch: RoutedMoeBufferGeometry,
    metadata_extra_bytes: usize,
    matrix_words: u32,
    dispatch: RoutedMoeDispatch,
    tile4: Option<RoutedMoeTile4Geometry>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeTile4Geometry {
    gate_indirect_offset: u64,
    down_indirect_offset: u64,
    tile_word_offset: u32,
    max_tiles: u32,
    schedule_words: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeProgramRecording {
    geometry: RoutedMoeProgramGeometry,
    batch_dispatch: RoutedMoeBatchDispatch,
    indirect_buffer: vk::Buffer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeBufferGeometry {
    descriptor_bytes: usize,
    item_stride: usize,
    total_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeProgramByteCounts {
    requested_bytes: u64,
    buffer_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutedMoeKernel {
    Guarded,
    Fast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoutedMoeTiming {
    Disabled,
    Profile,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeRun {
    kernel: RoutedMoeKernel,
    fast_variant: RoutedMoeFastVariant,
    timing: RoutedMoeTiming,
}

#[derive(Debug, Clone, Copy)]
struct RoutedMoePipelinePair {
    gate_up: vk::Pipeline,
    down: vk::Pipeline,
}

#[derive(Debug, Clone, Copy)]
struct RoutedMoePipelineTriple {
    gate_up: vk::Pipeline,
    down: vk::Pipeline,
    reduce: vk::Pipeline,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TimestampProperties {
    valid_bits: u32,
    period_ns: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RoutedMoeDispatch {
    gate_up: (u32, u32),
    down: (u32, u32),
    reduce: (u32, u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreparedFormat {
    Q8_0,
    Bf16,
}

impl PreparedFormat {
    fn name(self) -> &'static str {
        match self {
            Self::Q8_0 => "Q8_0",
            Self::Bf16 => "BF16",
        }
    }
}

#[derive(Debug)]
struct BufferAllocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    capacity: usize,
    allocation_bytes: u64,
}

impl BufferAllocation {
    unsafe fn destroy(self, device: &Device, live_allocations: &AtomicU32) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
        let previous = live_allocations.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "Vulkan allocation ledger underflow");
    }
}

impl WeightArenaBuilder {
    /// Append one complete borrowed weight and return its stable arena identifier.
    pub fn push(&mut self, weight: &Weight<'_>) -> Result<MatrixId, DeviceError> {
        validate_weight(weight)?;
        let bytes = weight.bytes;
        self.push_with(MatrixSpec::from_weight(weight), |destination| {
            destination.copy_from_slice(bytes);
            Ok(())
        })
    }

    /// Reserve a matrix slice and fill its encoded bytes directly.
    ///
    /// The callback sees exactly the unpadded checkpoint extent. This lets a range
    /// reader stream into mapped Vulkan memory without first creating a second full
    /// host copy. Shader-visible tail padding is zeroed by the arena.
    pub fn push_with(
        &mut self,
        spec: MatrixSpec,
        fill: impl FnOnce(&mut [u8]) -> Result<(), DeviceError>,
    ) -> Result<MatrixId, DeviceError> {
        let geometry = validate_matrix_spec(spec)?;
        let descriptor_len = align_up(
            geometry.encoded_len.max(DESCRIPTOR_WORD_BYTES),
            DESCRIPTOR_WORD_BYTES,
            "Vulkan matrix descriptor range",
        )?;
        if descriptor_len > self.context.max_storage_buffer_range {
            return Err(DeviceError::Shape(format!(
                "{} matrix needs a {descriptor_len}-byte storage-buffer range, exceeding {} on {}",
                geometry.format.name(),
                self.context.max_storage_buffer_range,
                self.context.device_name
            )));
        }
        let alignment = self
            .context
            .storage_buffer_alignment
            .max(DESCRIPTOR_WORD_BYTES);
        let mut selected = select_block_tail(&self.blocks, descriptor_len, alignment)?;
        let mut created_block = false;
        if selected.is_none() {
            let block_bytes = self.config.block_bytes.max(descriptor_len);
            if block_bytes > self.context.max_storage_buffer_range {
                return Err(DeviceError::Shape(format!(
                    "Vulkan arena block needs {block_bytes} bytes, exceeding the {}-byte storage-buffer limit on {}",
                    self.context.max_storage_buffer_range, self.context.device_name
                )));
            }
            let remaining_budget = self
                .config
                .budget_bytes
                .checked_sub(self.stats.allocation_bytes)
                .ok_or_else(|| backend_error("Vulkan arena accounting exceeded its byte budget"))?;
            let allocation_limit = self
                .context
                .max_memory_allocations
                .saturating_sub(self.config.allocation_headroom);
            let allocation = allocate_buffer_limited(
                &self.context,
                block_bytes,
                "Vulkan weight-arena block",
                allocation_limit,
                remaining_budget,
            )?;
            let mapped = match unsafe {
                self.context.device.map_memory(
                    allocation.memory,
                    0,
                    allocation.capacity as u64,
                    vk::MemoryMapFlags::empty(),
                )
            } {
                Ok(pointer) => NonNull::new(pointer.cast::<u8>()).ok_or_else(|| {
                    backend_error("mapping a Vulkan weight-arena block returned null")
                }),
                Err(error) => Err(vk_error("mapping a Vulkan weight-arena block", error)),
            };
            let mapped = match mapped {
                Ok(mapped) => mapped,
                Err(error) => {
                    unsafe {
                        allocation.destroy(&self.context.device, &self.context.live_allocations)
                    };
                    return Err(error);
                }
            };
            self.stats.blocks += 1;
            self.stats.buffer_bytes = self
                .stats
                .buffer_bytes
                .checked_add(allocation.capacity as u64)
                .ok_or_else(|| backend_error("Vulkan arena buffer accounting overflowed"))?;
            self.stats.allocation_bytes = self
                .stats
                .allocation_bytes
                .checked_add(allocation.allocation_bytes)
                .ok_or_else(|| backend_error("Vulkan arena allocation accounting overflowed"))?;
            self.blocks.push(ArenaBlockBuilder {
                context: Arc::clone(&self.context),
                allocation: Some(allocation),
                mapped: Some(mapped),
                used: 0,
            });
            selected = Some((self.blocks.len() - 1, 0));
            created_block = true;
        }
        let (block_index, offset) = selected.expect("a block was selected or created");
        let block = &mut self.blocks[block_index];
        let previous_used = block.used;
        let mapped = block.mapped.expect("building blocks remain mapped");
        let descriptor_end = offset
            .checked_add(descriptor_len)
            .ok_or_else(|| DeviceError::Shape("Vulkan arena slice end overflows".into()))?;
        debug_assert!(
            descriptor_end
                <= block
                    .allocation
                    .as_ref()
                    .expect("building block owns its allocation")
                    .capacity
        );
        // SAFETY: the selected range is checked against this uniquely borrowed block;
        // it remains mapped for the builder's lifetime and no committed record overlaps
        // the bump-allocated tail.
        let destination =
            unsafe { std::slice::from_raw_parts_mut(mapped.as_ptr().add(offset), descriptor_len) };
        destination.fill(0);
        if let Err(error) = fill(&mut destination[..geometry.encoded_len]) {
            destination.fill(0);
            if created_block {
                let mut unused = self.blocks.pop().expect("newly created block is last");
                let allocation = unused
                    .allocation
                    .take()
                    .expect("newly created block owns an allocation");
                let capacity = allocation.capacity as u64;
                let allocation_bytes = allocation.allocation_bytes;
                unsafe {
                    if unused.mapped.take().is_some() {
                        self.context.device.unmap_memory(allocation.memory);
                    }
                    allocation.destroy(&self.context.device, &self.context.live_allocations);
                }
                self.stats.blocks -= 1;
                self.stats.buffer_bytes -= capacity;
                self.stats.allocation_bytes -= allocation_bytes;
            }
            return Err(error);
        }
        block.used = descriptor_end;
        let index = u32::try_from(self.records.len())
            .map_err(|_| backend_error("a Vulkan arena cannot hold more than u32::MAX matrices"))?;
        let id = MatrixId {
            arena_id: self.arena_id,
            index,
        };
        self.records.push(MatrixRecord {
            block: u32::try_from(block_index).map_err(|_| {
                backend_error("a Vulkan arena cannot hold more than u32::MAX blocks")
            })?,
            offset,
            descriptor_len,
            encoded_len: geometry.encoded_len,
            rows: spec.rows,
            cols: spec.cols,
            row_bytes: geometry.row_bytes,
            format: geometry.format,
        });
        self.stats.matrices += 1;
        self.stats.encoded_bytes = self
            .stats
            .encoded_bytes
            .checked_add(geometry.encoded_len as u64)
            .ok_or_else(|| backend_error("Vulkan arena encoded-byte accounting overflowed"))?;
        self.stats.descriptor_bytes = self
            .stats
            .descriptor_bytes
            .checked_add(descriptor_len as u64)
            .ok_or_else(|| backend_error("Vulkan arena descriptor accounting overflowed"))?;
        self.stats.alignment_bytes = self
            .stats
            .alignment_bytes
            .checked_add((offset - previous_used) as u64)
            .ok_or_else(|| backend_error("Vulkan arena alignment accounting overflowed"))?;
        Ok(id)
    }

    /// Unmap all blocks and seal the arena for concurrent read-only dispatches.
    pub fn finish(mut self) -> Result<PreparedWeightArena, DeviceError> {
        let mut blocks = Vec::with_capacity(self.blocks.len());
        for builder in &mut self.blocks {
            let allocation = builder
                .allocation
                .take()
                .expect("an unfinished arena block owns its allocation");
            unsafe {
                if builder.mapped.take().is_some() {
                    self.context.device.unmap_memory(allocation.memory);
                }
            }
            self.stats.tail_bytes = self
                .stats
                .tail_bytes
                .checked_add((allocation.capacity - builder.used) as u64)
                .ok_or_else(|| backend_error("Vulkan arena tail accounting overflowed"))?;
            blocks.push(allocation);
        }
        debug_assert_eq!(
            self.stats.buffer_bytes,
            self.stats.descriptor_bytes + self.stats.alignment_bytes + self.stats.tail_bytes
        );
        Ok(PreparedWeightArena {
            inner: Arc::new(ArenaStorage {
                context: Arc::clone(&self.context),
                arena_id: self.arena_id,
                blocks: blocks.into_boxed_slice(),
                records: std::mem::take(&mut self.records).into_boxed_slice(),
                stats: self.stats,
            }),
        })
    }
}

impl PreparedWeightArena {
    /// Create a lightweight matrix handle from an identifier returned by this arena's
    /// builder.
    pub fn matrix(&self, id: MatrixId) -> Result<PreparedMatrix, DeviceError> {
        if id.arena_id != self.inner.arena_id || id.index as usize >= self.inner.records.len() {
            return Err(backend_error(
                "a matrix identifier does not belong to this Vulkan weight arena",
            ));
        }
        Ok(PreparedMatrix {
            arena: Arc::clone(&self.inner),
            id,
        })
    }

    /// Return exact storage accounting captured when the arena was sealed.
    pub fn stats(&self) -> ArenaStats {
        self.inner.stats
    }
}

impl VulkanMatvec {
    /// Initialize the highest-ranked non-CPU Vulkan compute device.
    pub fn new() -> Result<Self, DeviceError> {
        Ok(Self {
            inner: Arc::new(Context::new()?),
        })
    }

    /// The physical device named by the Vulkan driver.
    pub fn physical_device_name(&self) -> &str {
        &self.inner.device_name
    }

    /// Start a bounded persistent weight arena.
    pub fn arena_builder(
        &self,
        config: WeightArenaConfig,
    ) -> Result<WeightArenaBuilder, DeviceError> {
        if config.block_bytes == 0 {
            return Err(DeviceError::Shape(
                "Vulkan weight-arena block size must be non-zero".into(),
            ));
        }
        if config.budget_bytes == 0 {
            return Err(DeviceError::Shape(
                "Vulkan weight-arena budget must be non-zero".into(),
            ));
        }
        let arena_id = NEXT_ARENA_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| backend_error("exhausted Vulkan weight-arena identities"))?;
        Ok(WeightArenaBuilder {
            context: Arc::clone(&self.inner),
            config,
            arena_id,
            blocks: Vec::new(),
            records: Vec::new(),
            stats: ArenaStats::default(),
        })
    }

    /// Upload a complete Q8_0 or BF16 matrix once for repeated steady-state matvecs.
    pub fn prepare(&self, weight: &Weight<'_>) -> Result<PreparedMatrix, DeviceError> {
        let geometry = validate_weight(weight)?;
        let descriptor_bytes = align_up(
            geometry.encoded_len.max(DESCRIPTOR_WORD_BYTES),
            DESCRIPTOR_WORD_BYTES,
            "Vulkan matrix descriptor range",
        )?;
        // Compatibility convenience for benchmarks and isolated matrices. Full-model
        // callers must use `arena_builder` with a fit-authorized budget.
        let config = WeightArenaConfig {
            block_bytes: descriptor_bytes,
            budget_bytes: u64::MAX,
            allocation_headroom: 0,
        };
        let mut builder = self.arena_builder(config)?;
        let id = builder.push(weight)?;
        builder.finish()?.matrix(id)
    }

    /// Validate and retain an ordered set of BF16 routed experts for one fused batch.
    pub fn prepare_bf16_routed_experts(
        &self,
        experts: &[Bf16Expert<'_>],
    ) -> Result<PreparedRoutedExperts, DeviceError> {
        let first = experts.first().ok_or_else(|| {
            DeviceError::Shape("a routed-MoE batch needs at least one expert".into())
        })?;
        if !Arc::ptr_eq(&self.inner, &first.gate.arena.context) {
            return Err(backend_error(
                "routed expert weights belong to a different Vulkan context",
            ));
        }
        let arena = Arc::clone(&first.gate.arena);
        let first_gate = first.gate.record();
        let hidden = first_gate.cols;
        let expert_width = first_gate.rows;
        let block = first_gate.block;
        let mut minimum_offset = usize::MAX;
        let mut maximum_end = 0usize;

        for (route, expert) in experts.iter().enumerate() {
            for (name, matrix) in [
                ("gate", expert.gate),
                ("up", expert.up),
                ("down", expert.down),
            ] {
                if !Arc::ptr_eq(&arena, &matrix.arena) {
                    return Err(backend_error(format!(
                        "route {route} {name} projection belongs to a different Vulkan arena"
                    )));
                }
                let record = matrix.record();
                if record.format != PreparedFormat::Bf16 {
                    return Err(DeviceError::Shape(format!(
                        "route {route} {name} projection is {}, but fused routed-MoE currently requires BF16",
                        record.format.name()
                    )));
                }
                if record.block != block {
                    return Err(backend_error(format!(
                        "route {route} {name} projection is in arena block {}, while the fused batch currently requires block {block}",
                        record.block
                    )));
                }
                minimum_offset = minimum_offset.min(record.offset);
                maximum_end = maximum_end.max(
                    record
                        .offset
                        .checked_add(record.descriptor_len)
                        .ok_or_else(|| {
                            DeviceError::Shape("routed weight range overflows".into())
                        })?,
                );
            }
            let gate = expert.gate.record();
            let up = expert.up.record();
            let down = expert.down.record();
            if gate.rows != expert_width
                || gate.cols != hidden
                || up.rows != expert_width
                || up.cols != hidden
                || down.rows != hidden
                || down.cols != expert_width
            {
                return Err(DeviceError::Shape(format!(
                    "route {route} projections must be gate/up={expert_width}x{hidden} and down={hidden}x{expert_width}; got gate={}x{}, up={}x{}, down={}x{}",
                    gate.rows, gate.cols, up.rows, up.cols, down.rows, down.cols
                )));
            }
        }
        debug_assert_ne!(minimum_offset, usize::MAX);
        let weight_range = maximum_end - minimum_offset;
        if weight_range > self.inner.max_storage_buffer_range {
            return Err(DeviceError::Shape(format!(
                "routed weight span needs {weight_range} descriptor bytes, exceeding {} on {}",
                self.inner.max_storage_buffer_range, self.inner.device_name
            )));
        }
        let offsets_bytes = experts
            .iter()
            .map(|expert| {
                let convert = |matrix: &PreparedMatrix| {
                    let relative = matrix.record().offset - minimum_offset;
                    debug_assert!(relative.is_multiple_of(2));
                    u32::try_from(relative).map_err(|_| {
                        DeviceError::Shape("routed BF16 byte offset exceeds u32".into())
                    })
                };
                Ok([
                    convert(expert.gate)?,
                    convert(expert.up)?,
                    convert(expert.down)?,
                ])
            })
            .collect::<Result<Box<[[_; 3]]>, _>>()?;
        Ok(PreparedRoutedExperts {
            arena,
            block,
            weight_offset: minimum_offset,
            weight_range,
            hidden,
            expert_width,
            offsets_bytes,
        })
    }

    /// Prepare a reusable paired-load routed-MoE program for one arena block and shape.
    ///
    /// Construction pins the currently selected fast variant and records separate
    /// overwrite and accumulation command buffers. Later executions may use any expert
    /// ordering from the same retained arena block with exactly the same `H`, `F`, and
    /// `K`; absolute weight offsets are rebuilt in mapped metadata for each call.
    pub fn prepare_routed_moe_bf16_fast_program(
        &self,
        template: &PreparedRoutedExperts,
        config: RoutedMoeProgramConfig,
    ) -> Result<RoutedMoeProgram, DeviceError> {
        self.prepare_routed_moe_bf16_fast_batch_program(
            template,
            NonZeroUsize::new(1).expect("one is non-zero"),
            config,
        )
    }

    /// Prepare a fixed-size independent batch of paired-load routed-MoE executions.
    ///
    /// The returned program records all `N` gate/up dispatches, one dependency barrier,
    /// all `N` down dispatches, one dependency barrier, and all `N` reductions into each
    /// of its overwrite and accumulation command buffers. Every item retains independent
    /// activation and metadata slices, while all descriptors bind the complete stable
    /// arena block selected by `template` for weights.
    pub fn prepare_routed_moe_bf16_fast_batch_program(
        &self,
        template: &PreparedRoutedExperts,
        batch_len: NonZeroUsize,
        config: RoutedMoeProgramConfig,
    ) -> Result<RoutedMoeProgram, DeviceError> {
        self.prepare_routed_moe_bf16_fast_batch_program_with_dispatch(
            template,
            batch_len,
            config,
            RoutedMoeBatchDispatch::PerItem,
        )
    }

    /// Prepare a fixed-size routed-MoE batch with an explicit dispatch strategy.
    ///
    /// This entry point exists so candidates can be qualified against the unchanged
    /// [`RoutedMoeBatchDispatch::PerItem`] control. `ItemDimension` explicitly uses the
    /// Tree128 arithmetic and requires 128-lane workgroup support plus tightly packed
    /// whole-batch descriptor ranges. `ExpertTile4` additionally builds an indirect,
    /// expert-major schedule that shares weight loads across as many as four route
    /// occurrences. Unsupported configurations are refused before any program-owned
    /// Vulkan allocation is created. Both alternatives remain explicit experiments;
    /// the ordinary batch constructor always chooses the qualified per-item recording.
    pub fn prepare_routed_moe_bf16_fast_batch_program_with_dispatch(
        &self,
        template: &PreparedRoutedExperts,
        batch_len: NonZeroUsize,
        config: RoutedMoeProgramConfig,
        batch_dispatch: RoutedMoeBatchDispatch,
    ) -> Result<RoutedMoeProgram, DeviceError> {
        if config.budget_bytes == 0 {
            return Err(DeviceError::Shape(
                "Vulkan routed-program budget must be non-zero".into(),
            ));
        }
        if !Arc::ptr_eq(&self.inner, &template.arena.context) {
            return Err(backend_error(
                "prepared routed experts belong to a different Vulkan context",
            ));
        }
        validate_fast_routed_moe_weights(template)?;
        let geometry =
            routed_moe_program_geometry(&self.inner, template, batch_len, batch_dispatch)?;
        let fast_variant = match batch_dispatch {
            RoutedMoeBatchDispatch::PerItem => self.selected_routed_moe_fast_variant(template),
            RoutedMoeBatchDispatch::ItemDimension | RoutedMoeBatchDispatch::ExpertTile4 => {
                RoutedMoeFastVariant::Tree128
            }
        };
        match batch_dispatch {
            RoutedMoeBatchDispatch::PerItem => {}
            RoutedMoeBatchDispatch::ItemDimension => {
                validate_item_dimensional_routed_moe_program(
                    geometry,
                    fast_variant,
                    self.inner.max_group_count,
                    self.inner.max_storage_buffer_range,
                    &self.inner.device_name,
                )?;
            }
            RoutedMoeBatchDispatch::ExpertTile4 => {
                validate_tile4_routed_moe_program(
                    geometry,
                    fast_variant,
                    self.inner.max_group_count,
                    self.inner.max_storage_buffer_range,
                    &self.inner.device_name,
                )?;
            }
        }
        let byte_counts = routed_moe_program_byte_counts(geometry)?;
        if byte_counts.buffer_bytes > config.budget_bytes {
            return Err(backend_error(format!(
                "Vulkan routed program requests {} buffer bytes but its authorized budget is {} bytes (short by {} bytes)",
                byte_counts.buffer_bytes,
                config.budget_bytes,
                byte_counts.buffer_bytes - config.budget_bytes
            )));
        }
        let allocation_limit = self
            .inner
            .max_memory_allocations
            .saturating_sub(config.allocation_headroom);
        let mut stats = RoutedMoeProgramStats {
            requested_bytes: byte_counts.requested_bytes,
            buffer_bytes: 0,
            allocation_bytes: 0,
            allocation_slots: 0,
            budget_bytes: config.budget_bytes,
        };
        let (descriptor_set_count, descriptor_count) =
            routed_moe_program_descriptor_counts(batch_len, batch_dispatch)?;
        let block_index = usize::try_from(template.block)
            .map_err(|_| backend_error("routed-MoE arena block index exceeds usize"))?;
        let weight = template.arena.blocks.get(block_index).ok_or_else(|| {
            backend_error("prepared routed experts reference a missing arena block")
        })?;
        let weight_capacity = weight.capacity;
        validate_absolute_fast_routed_moe_offsets(
            template.weight_offset,
            &template.offsets_bytes,
            geometry.matrix_words,
            weight_capacity,
        )?;
        let weight_range = u64::try_from(weight_capacity)
            .map_err(|_| DeviceError::Shape("routed arena block range exceeds u64".into()))?;
        if weight_capacity > self.inner.max_storage_buffer_range {
            return Err(DeviceError::Shape(format!(
                "routed arena block needs {weight_capacity} descriptor bytes, exceeding {} on {}",
                self.inner.max_storage_buffer_range, self.inner.device_name
            )));
        }

        let (pipelines, pipeline_layout, descriptor_set_layout) = {
            let mut queue_state = self.state()?;
            let pipelines = match batch_dispatch {
                RoutedMoeBatchDispatch::PerItem => routed_moe_pipelines(
                    &self.inner,
                    &mut queue_state,
                    RoutedMoeRun {
                        kernel: RoutedMoeKernel::Fast,
                        fast_variant,
                        timing: RoutedMoeTiming::Disabled,
                    },
                )?,
                RoutedMoeBatchDispatch::ItemDimension => {
                    let pipelines =
                        item_dimensional_routed_moe_pipelines(&self.inner, &mut queue_state)?;
                    (pipelines.gate_up, pipelines.down, pipelines.reduce)
                }
                RoutedMoeBatchDispatch::ExpertTile4 => {
                    let pipelines = tile4_routed_moe_pipelines(&self.inner, &mut queue_state)?;
                    let reduce =
                        item_dimensional_routed_moe_pipelines(&self.inner, &mut queue_state)?
                            .reduce;
                    (pipelines.gate_up, pipelines.down, reduce)
                }
            };
            (
                pipelines,
                queue_state.pipeline_layout,
                queue_state.descriptor_set_layout,
            )
        };

        let mut init = RoutedMoeProgramInit {
            context: Arc::clone(&self.inner),
            _arena: Arc::clone(&template.arena),
            resources: RoutedMoeProgramResources::default(),
            armed: true,
        };
        let remaining_budget = routed_moe_program_remaining_budget(stats)?;
        init.resources.input = Some(allocate_persistently_mapped_buffer(
            &self.inner,
            geometry.input.total_bytes,
            "Vulkan routed-program input",
            allocation_limit,
            remaining_budget,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?);
        charge_routed_moe_program_allocation(
            &mut stats,
            &init
                .resources
                .input
                .as_ref()
                .expect("routed-program input was allocated")
                .allocation,
        )?;
        let remaining_budget = routed_moe_program_remaining_budget(stats)?;
        init.resources.output = Some(allocate_persistently_mapped_buffer(
            &self.inner,
            geometry.output.total_bytes,
            "Vulkan routed-program output",
            allocation_limit,
            remaining_budget,
            vk::BufferUsageFlags::STORAGE_BUFFER,
        )?);
        charge_routed_moe_program_allocation(
            &mut stats,
            &init
                .resources
                .output
                .as_ref()
                .expect("routed-program output was allocated")
                .allocation,
        )?;
        let remaining_budget = routed_moe_program_remaining_budget(stats)?;
        init.resources.metadata = Some(allocate_persistently_mapped_buffer(
            &self.inner,
            geometry.metadata.total_bytes,
            "Vulkan routed-program metadata",
            allocation_limit,
            remaining_budget,
            if batch_dispatch == RoutedMoeBatchDispatch::ExpertTile4 {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER
            } else {
                vk::BufferUsageFlags::STORAGE_BUFFER
            },
        )?);
        charge_routed_moe_program_allocation(
            &mut stats,
            &init
                .resources
                .metadata
                .as_ref()
                .expect("routed-program metadata was allocated")
                .allocation,
        )?;
        let remaining_budget = routed_moe_program_remaining_budget(stats)?;
        init.resources.hidden = Some(allocate_buffer_limited(
            &self.inner,
            geometry.hidden_scratch.total_bytes,
            "Vulkan routed-program hidden scratch",
            allocation_limit,
            remaining_budget,
        )?);
        charge_routed_moe_program_allocation(
            &mut stats,
            init.resources
                .hidden
                .as_ref()
                .expect("routed-program hidden scratch was allocated"),
        )?;
        let remaining_budget = routed_moe_program_remaining_budget(stats)?;
        init.resources.routes = Some(allocate_buffer_limited(
            &self.inner,
            geometry.route_scratch.total_bytes,
            "Vulkan routed-program output scratch",
            allocation_limit,
            remaining_budget,
        )?);
        charge_routed_moe_program_allocation(
            &mut stats,
            init.resources
                .routes
                .as_ref()
                .expect("routed-program output scratch was allocated"),
        )?;
        if stats.buffer_bytes != byte_counts.buffer_bytes
            || stats.allocation_slots != ROUTED_MOE_PROGRAM_ALLOCATION_SLOTS
        {
            return Err(backend_error(format!(
                "Vulkan routed-program allocation accounting mismatch: planned {} buffer bytes in {} slots, allocated {} bytes in {} slots",
                byte_counts.buffer_bytes,
                ROUTED_MOE_PROGRAM_ALLOCATION_SLOTS,
                stats.buffer_bytes,
                stats.allocation_slots
            )));
        }

        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(descriptor_count)];
        init.resources.descriptor_pool = unsafe {
            self.inner.device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(u32::try_from(descriptor_set_count).map_err(|_| {
                        DeviceError::Shape("routed-program descriptor-set count exceeds u32".into())
                    })?)
                    .pool_sizes(&pool_sizes),
                None,
            )
        }
        .map_err(|error| vk_error("creating the routed-program descriptor pool", error))?;
        let mut set_layouts = Vec::new();
        set_layouts
            .try_reserve_exact(descriptor_set_count)
            .map_err(|_| DeviceError::Allocation {
                device: BACKEND,
                bytes: descriptor_set_count
                    .saturating_mul(std::mem::size_of::<vk::DescriptorSetLayout>()),
                purpose: "routed-program descriptor layouts",
            })?;
        set_layouts.resize(descriptor_set_count, descriptor_set_layout);
        let descriptor_sets = unsafe {
            self.inner.device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(init.resources.descriptor_pool)
                    .set_layouts(&set_layouts),
            )
        }
        .map_err(|error| vk_error("allocating routed-program descriptor sets", error))?;
        if descriptor_sets.len() != descriptor_set_count {
            return Err(backend_error(format!(
                "Vulkan returned {} routed-program descriptor sets after {descriptor_set_count} were requested",
                descriptor_sets.len()
            )));
        }
        init.resources.descriptor_sets = descriptor_sets.into_boxed_slice();

        let input = &init
            .resources
            .input
            .as_ref()
            .expect("routed-program input was allocated")
            .allocation;
        let output = &init
            .resources
            .output
            .as_ref()
            .expect("routed-program output was allocated")
            .allocation;
        let metadata = &init
            .resources
            .metadata
            .as_ref()
            .expect("routed-program metadata was allocated")
            .allocation;
        let hidden = init
            .resources
            .hidden
            .as_ref()
            .expect("routed-program hidden scratch was allocated");
        let routes = init
            .resources
            .routes
            .as_ref()
            .expect("routed-program output scratch was allocated");
        let buffer_info = |allocation: &BufferAllocation,
                           geometry: RoutedMoeBufferGeometry,
                           item: usize,
                           purpose: &str|
         -> Result<[vk::DescriptorBufferInfo; 1], DeviceError> {
            let offset = geometry.item_stride.checked_mul(item).ok_or_else(|| {
                DeviceError::Shape(format!("routed-program {purpose} item offset overflows"))
            })?;
            let end = offset
                .checked_add(geometry.descriptor_bytes)
                .ok_or_else(|| {
                    DeviceError::Shape(format!("routed-program {purpose} descriptor end overflows"))
                })?;
            if end > allocation.capacity {
                return Err(DeviceError::Shape(format!(
                    "routed-program {purpose} descriptor {}..{end} exceeds buffer capacity {}",
                    offset, allocation.capacity
                )));
            }
            let required_alignment = self
                .inner
                .storage_buffer_alignment
                .max(DESCRIPTOR_WORD_BYTES);
            if !offset.is_multiple_of(required_alignment) {
                return Err(DeviceError::Shape(format!(
                    "routed-program {purpose} descriptor offset {offset} is not aligned to {required_alignment}"
                )));
            }
            Ok([vk::DescriptorBufferInfo::default()
                .buffer(allocation.buffer)
                .offset(u64::try_from(offset).map_err(|_| {
                    DeviceError::Shape(format!(
                        "routed-program {purpose} descriptor offset exceeds u64"
                    ))
                })?)
                .range(u64::try_from(geometry.descriptor_bytes).map_err(|_| {
                    DeviceError::Shape(format!(
                        "routed-program {purpose} descriptor range exceeds u64"
                    ))
                })?)])
        };
        let whole_buffer_info = |allocation: &BufferAllocation,
                                 geometry: RoutedMoeBufferGeometry,
                                 purpose: &str|
         -> Result<[vk::DescriptorBufferInfo; 1], DeviceError> {
            if geometry.total_bytes > allocation.capacity {
                return Err(DeviceError::Shape(format!(
                    "routed-program whole-batch {purpose} range {} exceeds buffer capacity {}",
                    geometry.total_bytes, allocation.capacity
                )));
            }
            if geometry.total_bytes > self.inner.max_storage_buffer_range {
                return Err(DeviceError::Shape(format!(
                    "routed-program whole-batch {purpose} range {} exceeds {} on {}",
                    geometry.total_bytes,
                    self.inner.max_storage_buffer_range,
                    self.inner.device_name
                )));
            }
            Ok([vk::DescriptorBufferInfo::default()
                .buffer(allocation.buffer)
                .offset(0)
                .range(u64::try_from(geometry.total_bytes).map_err(|_| {
                    DeviceError::Shape(format!(
                        "routed-program whole-batch {purpose} range exceeds u64"
                    ))
                })?)])
        };
        let weight_info = [vk::DescriptorBufferInfo::default()
            .buffer(weight.buffer)
            .offset(0)
            .range(weight_range)];
        match batch_dispatch {
            RoutedMoeBatchDispatch::PerItem => {
                for item in 0..batch_len.get() {
                    let input_info = buffer_info(input, geometry.input, item, "input")?;
                    let output_info = buffer_info(output, geometry.output, item, "output")?;
                    let metadata_info = buffer_info(metadata, geometry.metadata, item, "metadata")?;
                    let hidden_info =
                        buffer_info(hidden, geometry.hidden_scratch, item, "hidden scratch")?;
                    let routes_info =
                        buffer_info(routes, geometry.route_scratch, item, "route scratch")?;
                    let set_base = item.checked_mul(3).ok_or_else(|| {
                        DeviceError::Shape("descriptor-set index overflows".into())
                    })?;
                    let sets = &init.resources.descriptor_sets[set_base..set_base + 3];
                    let writes = [
                        descriptor_write(sets[0], 0, &input_info),
                        descriptor_write(sets[0], 1, &weight_info),
                        descriptor_write(sets[0], 2, &hidden_info),
                        descriptor_write(sets[0], 3, &metadata_info),
                        descriptor_write(sets[1], 0, &hidden_info),
                        descriptor_write(sets[1], 1, &weight_info),
                        descriptor_write(sets[1], 2, &routes_info),
                        descriptor_write(sets[1], 3, &metadata_info),
                        descriptor_write(sets[2], 0, &routes_info),
                        descriptor_write(sets[2], 1, &weight_info),
                        descriptor_write(sets[2], 2, &output_info),
                        descriptor_write(sets[2], 3, &metadata_info),
                    ];
                    unsafe { self.inner.device.update_descriptor_sets(&writes, &[]) };
                }
            }
            RoutedMoeBatchDispatch::ItemDimension | RoutedMoeBatchDispatch::ExpertTile4 => {
                let input_info = whole_buffer_info(input, geometry.input, "input")?;
                let output_info = whole_buffer_info(output, geometry.output, "output")?;
                let metadata_info = whole_buffer_info(metadata, geometry.metadata, "metadata")?;
                let hidden_info =
                    whole_buffer_info(hidden, geometry.hidden_scratch, "hidden scratch")?;
                let routes_info =
                    whole_buffer_info(routes, geometry.route_scratch, "route scratch")?;
                let sets = &init.resources.descriptor_sets;
                let writes = [
                    descriptor_write(sets[0], 0, &input_info),
                    descriptor_write(sets[0], 1, &weight_info),
                    descriptor_write(sets[0], 2, &hidden_info),
                    descriptor_write(sets[0], 3, &metadata_info),
                    descriptor_write(sets[1], 0, &hidden_info),
                    descriptor_write(sets[1], 1, &weight_info),
                    descriptor_write(sets[1], 2, &routes_info),
                    descriptor_write(sets[1], 3, &metadata_info),
                    descriptor_write(sets[2], 0, &routes_info),
                    descriptor_write(sets[2], 1, &weight_info),
                    descriptor_write(sets[2], 2, &output_info),
                    descriptor_write(sets[2], 3, &metadata_info),
                ];
                unsafe { self.inner.device.update_descriptor_sets(&writes, &[]) };
            }
        }

        init.resources.command_pool = unsafe {
            self.inner.device.create_command_pool(
                &vk::CommandPoolCreateInfo::default().queue_family_index(self.inner.queue_family),
                None,
            )
        }
        .map_err(|error| vk_error("creating the routed-program command pool", error))?;
        let command_buffers = unsafe {
            self.inner.device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(init.resources.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(2),
            )
        }
        .map_err(|error| vk_error("allocating routed-program command buffers", error))?;
        init.resources
            .command_buffers
            .copy_from_slice(&command_buffers);
        init.resources.fence = unsafe {
            self.inner.device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .map_err(|error| vk_error("creating the routed-program fence", error))?;

        for (index, accumulate) in [false, true].into_iter().enumerate() {
            record_routed_moe_program_command_buffer(
                &self.inner.device,
                init.resources.command_buffers[index],
                pipeline_layout,
                &init.resources.descriptor_sets,
                pipelines,
                RoutedMoeProgramRecording {
                    geometry,
                    batch_dispatch,
                    indirect_buffer: metadata.buffer,
                },
                accumulate,
            )?;
        }

        let resources = std::mem::take(&mut init.resources);
        init.armed = false;
        Ok(RoutedMoeProgram {
            context: Arc::clone(&self.inner),
            arena: Arc::clone(&template.arena),
            block: template.block,
            batch_len: batch_len.get(),
            hidden: template.hidden,
            expert_width: template.expert_width,
            routes: template.routes(),
            fast_variant,
            batch_dispatch,
            weight_capacity,
            geometry,
            stats,
            state: Mutex::new(RoutedMoeProgramState {
                resources,
                poisoned: false,
            }),
        })
    }

    /// Execute gate/up/SwiGLU/down for all prepared routes with one queue submission.
    ///
    /// Only the input, route weights, and final output cross the host boundary. The
    /// `[route, expert_width]` and `[route, hidden]` intermediates stay resident in
    /// Vulkan buffers, with explicit compute-to-compute barriers between three stages.
    /// Eligible even-width shapes use the qualified paired-load kernels; other valid
    /// shapes automatically retain the guarded scalar-load implementation.
    pub fn routed_moe_bf16(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        let kernel = self.preferred_routed_moe_kernel(experts);
        let fast_variant = self.selected_routed_moe_fast_variant(experts);
        self.run_routed_moe_bf16(
            y,
            x,
            experts,
            route_weights,
            accumulate,
            RoutedMoeRun {
                kernel,
                fast_variant,
                timing: RoutedMoeTiming::Disabled,
            },
        )
        .map(|_| ())
    }

    /// Execute the guarded scalar-load BF16 routed-MoE implementation explicitly.
    ///
    /// This entry point is retained as a correctness and performance control for the
    /// automatically selected paired-load path.
    pub fn routed_moe_bf16_guarded(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        self.run_routed_moe_bf16(
            y,
            x,
            experts,
            route_weights,
            accumulate,
            RoutedMoeRun {
                kernel: RoutedMoeKernel::Guarded,
                fast_variant: RoutedMoeFastVariant::QualifiedTree64,
                timing: RoutedMoeTiming::Disabled,
            },
        )
        .map(|_| ())
    }

    /// Execute the paired-load BF16 routed-MoE implementation explicitly.
    ///
    /// This path uses direct two-dimensional dispatches and consumes two BF16 values
    /// per weight word. The ordinary routed method selects it automatically when
    /// eligible; this entry point lets tests and callers require it. Odd hidden or
    /// expert widths, unsupported dispatch geometry, and any violated word-alignment
    /// invariant are rejected before the command buffer is recorded.
    pub fn routed_moe_bf16_fast(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        let fast_variant = self.selected_routed_moe_fast_variant(experts);
        self.run_routed_moe_bf16(
            y,
            x,
            experts,
            route_weights,
            accumulate,
            RoutedMoeRun {
                kernel: RoutedMoeKernel::Fast,
                fast_variant,
                timing: RoutedMoeTiming::Disabled,
            },
        )
        .map(|_| ())
    }

    /// Execute and profile the paired-load BF16 routed-MoE implementation.
    ///
    /// Six GPU timestamps are recorded inside the ordinary command buffer. This adds
    /// no submission or fence, separates each kernel from both compute dependencies,
    /// and leaves [`Self::routed_moe_bf16_fast`] completely uninstrumented. Queues
    /// without timestamp support refuse this explicit profiling request before any
    /// input is uploaded or `y` is modified.
    pub fn routed_moe_bf16_fast_profiled(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
    ) -> Result<RoutedMoeGpuProfile, DeviceError> {
        let fast_variant = self.selected_routed_moe_fast_variant(experts);
        self.run_routed_moe_bf16(
            y,
            x,
            experts,
            route_weights,
            accumulate,
            RoutedMoeRun {
                kernel: RoutedMoeKernel::Fast,
                fast_variant,
                timing: RoutedMoeTiming::Profile,
            },
        )?
        .ok_or_else(|| backend_error("profiled routed-MoE execution returned no timestamps"))
    }

    /// Execute an explicit paired-load reduction variant for qualification.
    ///
    /// Candidate pipelines are created lazily and do not alter the automatic choice in
    /// [`Self::routed_moe_bf16`]. Unsupported workgroup sizes or subgroup arithmetic
    /// are refused before uploading input or modifying `y`.
    pub fn routed_moe_bf16_fast_variant(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
        variant: RoutedMoeFastVariant,
    ) -> Result<(), DeviceError> {
        self.run_routed_moe_bf16(
            y,
            x,
            experts,
            route_weights,
            accumulate,
            RoutedMoeRun {
                kernel: RoutedMoeKernel::Fast,
                fast_variant: variant,
                timing: RoutedMoeTiming::Disabled,
            },
        )
        .map(|_| ())
    }

    /// Execute and GPU-profile an explicit paired-load reduction variant.
    ///
    /// The same six timestamp boundaries as
    /// [`Self::routed_moe_bf16_fast_profiled`] are used.
    pub fn routed_moe_bf16_fast_variant_profiled(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
        variant: RoutedMoeFastVariant,
    ) -> Result<RoutedMoeGpuProfile, DeviceError> {
        self.run_routed_moe_bf16(
            y,
            x,
            experts,
            route_weights,
            accumulate,
            RoutedMoeRun {
                kernel: RoutedMoeKernel::Fast,
                fast_variant: variant,
                timing: RoutedMoeTiming::Profile,
            },
        )?
        .ok_or_else(|| backend_error("profiled routed-MoE execution returned no timestamps"))
    }

    fn preferred_routed_moe_kernel(&self, experts: &PreparedRoutedExperts) -> RoutedMoeKernel {
        let dimensions = (
            u32::try_from(experts.hidden),
            u32::try_from(experts.expert_width),
            u32::try_from(experts.routes()),
        );
        let (Ok(hidden), Ok(expert_width), Ok(routes)) = dimensions else {
            return RoutedMoeKernel::Guarded;
        };
        if routed_moe_dispatch(
            RoutedMoeKernel::Fast,
            hidden,
            expert_width,
            routes,
            self.inner.max_group_count,
            &self.inner.device_name,
        )
        .is_ok()
            && validate_fast_routed_moe_weights(experts).is_ok()
        {
            RoutedMoeKernel::Fast
        } else {
            RoutedMoeKernel::Guarded
        }
    }

    /// Return the paired-load reduction variant selected for prepared experts.
    ///
    /// This exposes the same device, driver, workgroup-limit, and routed-shape
    /// qualification used by the automatic execution path. Callers can therefore
    /// distinguish an exact promoted configuration from the portable Tree64 fallback
    /// without duplicating the qualification policy.
    pub fn selected_routed_moe_fast_variant(
        &self,
        experts: &PreparedRoutedExperts,
    ) -> RoutedMoeFastVariant {
        preferred_fast_variant_for_device(
            &self.inner.device_name,
            self.inner.max_work_group_invocations,
            self.inner.max_work_group_size_x,
            self.inner.driver_version,
            experts.hidden,
            experts.expert_width,
            experts.routes(),
        )
    }

    fn run_routed_moe_bf16(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
        run: RoutedMoeRun,
    ) -> Result<Option<RoutedMoeGpuProfile>, DeviceError> {
        if !Arc::ptr_eq(&self.inner, &experts.arena.context) {
            return Err(backend_error(
                "prepared routed experts belong to a different Vulkan context",
            ));
        }
        let timestamp_properties = match run.timing {
            RoutedMoeTiming::Disabled => None,
            RoutedMoeTiming::Profile => Some(self.inner.timestamp_properties.ok_or_else(|| {
                backend_error(format!(
                    "{} compute queue does not expose usable GPU timestamps",
                    self.inner.device_name
                ))
            })?),
        };
        if x.len() != experts.hidden || y.len() != experts.hidden {
            return Err(DeviceError::Shape(format!(
                "routed-MoE hidden width is {}, got x={} y={}",
                experts.hidden,
                x.len(),
                y.len()
            )));
        }
        if route_weights.len() != experts.routes() {
            return Err(DeviceError::Shape(format!(
                "routed-MoE has {} routes but received {} route weights",
                experts.routes(),
                route_weights.len()
            )));
        }
        if let Some((route, weight)) = route_weights
            .iter()
            .copied()
            .enumerate()
            .find(|(_, weight)| !weight.is_finite())
        {
            return Err(DeviceError::Shape(format!(
                "routed-MoE route {route} has non-finite weight {weight}"
            )));
        }

        let hidden = u32::try_from(experts.hidden)
            .map_err(|_| DeviceError::Shape("routed-MoE hidden width exceeds u32".into()))?;
        let expert_width = u32::try_from(experts.expert_width)
            .map_err(|_| DeviceError::Shape("routed-MoE expert width exceeds u32".into()))?;
        let routes = u32::try_from(experts.routes())
            .map_err(|_| DeviceError::Shape("routed-MoE route count exceeds u32".into()))?;
        let hidden_elements = experts
            .routes()
            .checked_mul(experts.expert_width)
            .ok_or_else(|| DeviceError::Shape("routed hidden scratch size overflows".into()))?;
        let route_elements = experts
            .routes()
            .checked_mul(experts.hidden)
            .ok_or_else(|| DeviceError::Shape("routed output scratch size overflows".into()))?;
        let metadata_words = experts
            .routes()
            .checked_mul(4)
            .ok_or_else(|| DeviceError::Shape("routed metadata size overflows".into()))?;
        u32::try_from(metadata_words)
            .map_err(|_| DeviceError::Shape("routed metadata word count exceeds u32".into()))?;
        let byte_len = |elements: usize, purpose: &str| {
            elements
                .checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| DeviceError::Shape(format!("{purpose} byte count overflows")))
        };
        let input_bytes = byte_len(x.len(), "routed input")?;
        let output_bytes = byte_len(y.len(), "routed output")?;
        let hidden_bytes = byte_len(hidden_elements, "routed hidden scratch")?;
        let route_bytes = byte_len(route_elements, "routed output scratch")?;
        let metadata_bytes = metadata_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| DeviceError::Shape("routed metadata byte count overflows".into()))?;
        for (purpose, bytes) in [
            ("input", input_bytes),
            ("output", output_bytes),
            ("metadata", metadata_bytes),
            ("hidden scratch", hidden_bytes),
            ("route scratch", route_bytes),
        ] {
            if bytes > self.inner.max_storage_buffer_range {
                return Err(DeviceError::Shape(format!(
                    "routed-MoE {purpose} needs {bytes} descriptor bytes, exceeding {} on {}",
                    self.inner.max_storage_buffer_range, self.inner.device_name
                )));
            }
        }
        let dispatch = routed_moe_dispatch(
            run.kernel,
            hidden,
            expert_width,
            routes,
            self.inner.max_group_count,
            &self.inner.device_name,
        )?;
        if run.kernel == RoutedMoeKernel::Fast {
            validate_fast_routed_moe_weights(experts)?;
        }

        let mut metadata = Vec::with_capacity(metadata_words);
        for (offsets, route_weight) in experts.offsets_bytes.iter().zip(route_weights) {
            metadata.extend_from_slice(&[
                routed_moe_metadata_offset(run.kernel, offsets[0])?,
                routed_moe_metadata_offset(run.kernel, offsets[1])?,
                routed_moe_metadata_offset(run.kernel, offsets[2])?,
                route_weight.to_bits(),
            ]);
        }

        let mut state = self.state()?;
        let timestamp_query_pool = if timestamp_properties.is_some() {
            Some(ensure_timestamp_query_pool(&self.inner.device, &mut state)?)
        } else {
            None
        };
        let pipelines = routed_moe_pipelines(&self.inner, &mut state, run)?;
        ensure_buffer(
            &self.inner,
            &mut state.input,
            input_bytes,
            "Vulkan routed input",
        )?;
        ensure_buffer(
            &self.inner,
            &mut state.output,
            output_bytes,
            "Vulkan routed output",
        )?;
        ensure_buffer(
            &self.inner,
            &mut state.moe_metadata,
            metadata_bytes,
            "Vulkan routed metadata",
        )?;
        ensure_buffer(
            &self.inner,
            &mut state.moe_hidden,
            hidden_bytes,
            "Vulkan routed hidden scratch",
        )?;
        ensure_buffer(
            &self.inner,
            &mut state.moe_routes,
            route_bytes,
            "Vulkan routed output scratch",
        )?;
        let input = state.input.as_ref().expect("routed input was ensured");
        let output = state.output.as_ref().expect("routed output was ensured");
        let metadata_buffer = state
            .moe_metadata
            .as_ref()
            .expect("routed metadata was ensured");
        let hidden_buffer = state
            .moe_hidden
            .as_ref()
            .expect("routed hidden scratch was ensured");
        let routes_buffer = state
            .moe_routes
            .as_ref()
            .expect("routed output scratch was ensured");
        let weight_buffer = &experts.arena.blocks[experts.block as usize];
        write_f32(&self.inner.device, input, x)?;
        write_u32(&self.inner.device, metadata_buffer, &metadata)?;
        if accumulate {
            write_f32(&self.inner.device, output, y)?;
        }

        let buffer_info = |allocation: &BufferAllocation, range: usize| {
            [vk::DescriptorBufferInfo::default()
                .buffer(allocation.buffer)
                .offset(0)
                .range(range as u64)]
        };
        let input_info = buffer_info(input, input_bytes);
        let weight_info = [vk::DescriptorBufferInfo::default()
            .buffer(weight_buffer.buffer)
            .offset(experts.weight_offset as u64)
            .range(experts.weight_range as u64)];
        let metadata_info = buffer_info(metadata_buffer, metadata_bytes);
        let hidden_info = buffer_info(hidden_buffer, hidden_bytes);
        let routes_info = buffer_info(routes_buffer, route_bytes);
        let output_info = buffer_info(output, output_bytes);
        let writes = [
            descriptor_write(state.descriptor_sets[1], 0, &input_info),
            descriptor_write(state.descriptor_sets[1], 1, &weight_info),
            descriptor_write(state.descriptor_sets[1], 2, &hidden_info),
            descriptor_write(state.descriptor_sets[1], 3, &metadata_info),
            descriptor_write(state.descriptor_sets[2], 0, &hidden_info),
            descriptor_write(state.descriptor_sets[2], 1, &weight_info),
            descriptor_write(state.descriptor_sets[2], 2, &routes_info),
            descriptor_write(state.descriptor_sets[2], 3, &metadata_info),
            descriptor_write(state.descriptor_sets[3], 0, &routes_info),
            descriptor_write(state.descriptor_sets[3], 1, &weight_info),
            descriptor_write(state.descriptor_sets[3], 2, &output_info),
            descriptor_write(state.descriptor_sets[3], 3, &metadata_info),
        ];
        unsafe { self.inner.device.update_descriptor_sets(&writes, &[]) };

        let parameters = [hidden, expert_width, routes, u32::from(accumulate)];
        let parameter_bytes = unsafe {
            std::slice::from_raw_parts(
                parameters.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&parameters),
            )
        };
        unsafe {
            self.inner
                .device
                .reset_fences(&[state.fence])
                .map_err(|error| vk_error("resetting the routed-MoE fence", error))?;
            self.inner
                .device
                .reset_command_buffer(
                    state.command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .map_err(|error| vk_error("resetting the routed-MoE command buffer", error))?;
            self.inner
                .device
                .begin_command_buffer(
                    state.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| vk_error("beginning the routed-MoE command buffer", error))?;
            if let Some(query_pool) = timestamp_query_pool {
                self.inner.device.cmd_reset_query_pool(
                    state.command_buffer,
                    query_pool,
                    0,
                    ROUTED_MOE_TIMESTAMP_COUNT,
                );
            }
            let host_write = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)];
            self.inner.device.cmd_pipeline_barrier(
                state.command_buffer,
                vk::PipelineStageFlags::HOST,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &host_write,
                &[],
                &[],
            );
            let write_timestamp = |query: u32| {
                if let Some(query_pool) = timestamp_query_pool {
                    self.inner.device.cmd_write_timestamp(
                        state.command_buffer,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        query_pool,
                        query,
                    );
                }
            };
            let record_dispatch = |pipeline, descriptor_set, groups: (u32, u32)| {
                self.inner.device.cmd_bind_pipeline(
                    state.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline,
                );
                self.inner.device.cmd_bind_descriptor_sets(
                    state.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    state.pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                self.inner.device.cmd_push_constants(
                    state.command_buffer,
                    state.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    parameter_bytes,
                );
                self.inner
                    .device
                    .cmd_dispatch(state.command_buffer, groups.0, groups.1, 1);
            };
            write_timestamp(0);
            record_dispatch(pipelines.0, state.descriptor_sets[1], dispatch.gate_up);
            write_timestamp(1);
            compute_read_after_write(&self.inner.device, state.command_buffer);
            write_timestamp(2);
            record_dispatch(pipelines.1, state.descriptor_sets[2], dispatch.down);
            write_timestamp(3);
            compute_read_after_write(&self.inner.device, state.command_buffer);
            write_timestamp(4);
            record_dispatch(pipelines.2, state.descriptor_sets[3], dispatch.reduce);
            write_timestamp(5);
            let host_read = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            self.inner.device.cmd_pipeline_barrier(
                state.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &host_read,
                &[],
                &[],
            );
            self.inner
                .device
                .end_command_buffer(state.command_buffer)
                .map_err(|error| vk_error("ending the routed-MoE command buffer", error))?;
            let command_buffers = [state.command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            state.health.submit_and_wait(
                || {
                    self.inner
                        .device
                        .queue_submit(state.queue, &submits, state.fence)
                        .map_err(|error| vk_error("submitting routed-MoE batch", error))
                },
                || {
                    self.inner
                        .device
                        .wait_for_fences(&[state.fence], true, u64::MAX)
                        .map_err(|error| vk_error("waiting for routed-MoE batch", error))
                },
            )?;
        }
        let profile = match (timestamp_query_pool, timestamp_properties) {
            (Some(query_pool), Some(properties)) => {
                let mut timestamps = [0u64; ROUTED_MOE_TIMESTAMP_COUNT as usize];
                let query_started = Instant::now();
                unsafe {
                    self.inner
                        .device
                        .get_query_pool_results(
                            query_pool,
                            0,
                            &mut timestamps,
                            vk::QueryResultFlags::TYPE_64,
                        )
                        .map_err(|error| {
                            vk_error("reading routed-MoE timestamp query results", error)
                        })?;
                }
                Some(routed_moe_gpu_profile(
                    timestamps,
                    properties,
                    query_started.elapsed(),
                )?)
            }
            (None, None) => None,
            _ => {
                return Err(backend_error(
                    "routed-MoE timestamp state became internally inconsistent",
                ));
            }
        };
        read_f32(&self.inner.device, output, y)?;
        Ok(profile)
    }

    /// Overwrite `y` with a prepared matrix-vector product.
    pub fn matvec(
        &self,
        y: &mut [f32],
        x: &[f32],
        weight: &PreparedMatrix,
    ) -> Result<(), DeviceError> {
        self.run(y, x, weight, 1.0, false)
    }

    /// Accumulate a scaled prepared matrix-vector product into `y`.
    pub fn matvec_add(
        &self,
        y: &mut [f32],
        x: &[f32],
        weight: &PreparedMatrix,
        scale: f32,
    ) -> Result<(), DeviceError> {
        self.run(y, x, weight, scale, true)
    }

    fn state(&self) -> Result<MutexGuard<'_, QueueState>, DeviceError> {
        let state = self.inner.state.lock().map_err(|_| {
            backend_error("the Vulkan queue mutex was poisoned by an earlier failed dispatch")
        })?;
        // Check before callers replace allocations, map/write activations, update
        // descriptors, or reset a command buffer from an uncertain earlier dispatch.
        state.health.require_ready()?;
        Ok(state)
    }

    fn run(
        &self,
        y: &mut [f32],
        x: &[f32],
        weight: &PreparedMatrix,
        scale: f32,
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        self.run_with_group_limits(y, x, weight, scale, accumulate, self.inner.max_group_count)
    }

    fn run_with_group_limits(
        &self,
        y: &mut [f32],
        x: &[f32],
        weight: &PreparedMatrix,
        scale: f32,
        accumulate: bool,
        requested_group_limits: [u32; 3],
    ) -> Result<(), DeviceError> {
        if !Arc::ptr_eq(&self.inner, &weight.arena.context) {
            return Err(backend_error(
                "a prepared weight belongs to a different Vulkan context",
            ));
        }
        let record = weight.record();
        if x.len() != record.cols || y.len() != record.rows {
            return Err(DeviceError::Shape(format!(
                "prepared weight is {}x{}, got x={} y={}",
                record.rows,
                record.cols,
                x.len(),
                y.len()
            )));
        }
        if !scale.is_finite() {
            return Err(DeviceError::Shape(format!(
                "Vulkan {} output scale must be finite, got {scale}",
                record.format.name()
            )));
        }

        let rows = u32::try_from(record.rows)
            .map_err(|_| DeviceError::Shape("Vulkan row count exceeds u32".into()))?;
        let cols = u32::try_from(record.cols)
            .map_err(|_| DeviceError::Shape("Vulkan column count exceeds u32".into()))?;
        let row_bytes = u32::try_from(record.row_bytes)
            .map_err(|_| DeviceError::Shape("Vulkan encoded row size exceeds u32".into()))?;
        let group_limits = [
            requested_group_limits[0].min(self.inner.max_group_count[0]),
            requested_group_limits[1].min(self.inner.max_group_count[1]),
            requested_group_limits[2].min(self.inner.max_group_count[2]),
        ];
        let (groups_x, groups_y) = dispatch_grid(rows, group_limits)?;
        let input_bytes = x
            .len()
            .checked_mul(4)
            .ok_or_else(|| DeviceError::Shape("Vulkan input-vector byte count overflows".into()))?;
        let output_bytes = y.len().checked_mul(4).ok_or_else(|| {
            DeviceError::Shape("Vulkan output-vector byte count overflows".into())
        })?;
        for (purpose, bytes) in [("input", input_bytes), ("output", output_bytes)] {
            if bytes > self.inner.max_storage_buffer_range {
                return Err(DeviceError::Shape(format!(
                    "Vulkan {purpose} needs a {bytes}-byte storage-buffer range, exceeding {} on {}",
                    self.inner.max_storage_buffer_range, self.inner.device_name
                )));
            }
        }

        let mut state = self.state()?;
        ensure_buffer(
            &self.inner,
            &mut state.input,
            input_bytes,
            "Vulkan input vector",
        )?;
        ensure_buffer(
            &self.inner,
            &mut state.output,
            output_bytes,
            "Vulkan output vector",
        )?;
        let input = state.input.as_ref().expect("input buffer was ensured");
        let output = state.output.as_ref().expect("output buffer was ensured");
        let prepared = &weight.arena.blocks[record.block as usize];
        write_f32(&self.inner.device, input, x)?;
        if accumulate {
            write_f32(&self.inner.device, output, y)?;
        }

        let input_info = [vk::DescriptorBufferInfo::default()
            .buffer(input.buffer)
            .offset(0)
            .range(input_bytes as u64)];
        let weight_info = [vk::DescriptorBufferInfo::default()
            .buffer(prepared.buffer)
            .offset(record.offset as u64)
            .range(record.descriptor_len as u64)];
        let output_info = [vk::DescriptorBufferInfo::default()
            .buffer(output.buffer)
            .offset(0)
            .range(output_bytes as u64)];
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(state.descriptor_sets[0])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&input_info),
            vk::WriteDescriptorSet::default()
                .dst_set(state.descriptor_sets[0])
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&weight_info),
            vk::WriteDescriptorSet::default()
                .dst_set(state.descriptor_sets[0])
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&output_info),
        ];
        unsafe { self.inner.device.update_descriptor_sets(&writes, &[]) };

        unsafe {
            self.inner
                .device
                .reset_fences(&[state.fence])
                .map_err(|error| vk_error("resetting the dispatch fence", error))?;
            self.inner
                .device
                .reset_command_buffer(
                    state.command_buffer,
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .map_err(|error| vk_error("resetting the command buffer", error))?;
            self.inner
                .device
                .begin_command_buffer(
                    state.command_buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .map_err(|error| vk_error("beginning the command buffer", error))?;
            let pipeline = match record.format {
                PreparedFormat::Q8_0 => state.q8_pipeline,
                PreparedFormat::Bf16 => state.bf16_pipeline,
            };
            self.inner.device.cmd_bind_pipeline(
                state.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline,
            );
            self.inner.device.cmd_bind_descriptor_sets(
                state.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                state.pipeline_layout,
                0,
                &[state.descriptor_sets[0]],
                &[],
            );
            let parameters = [
                rows,
                cols,
                row_bytes,
                scale.to_bits(),
                u32::from(accumulate),
                0,
            ];
            let parameter_bytes = std::slice::from_raw_parts(
                parameters.as_ptr().cast::<u8>(),
                std::mem::size_of_val(&parameters),
            );
            self.inner.device.cmd_push_constants(
                state.command_buffer,
                state.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                parameter_bytes,
            );
            self.inner
                .device
                .cmd_dispatch(state.command_buffer, groups_x, groups_y, 1);
            let host_read = [vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)];
            self.inner.device.cmd_pipeline_barrier(
                state.command_buffer,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &host_read,
                &[],
                &[],
            );
            self.inner
                .device
                .end_command_buffer(state.command_buffer)
                .map_err(|error| vk_error("ending the command buffer", error))?;
            let command_buffers = [state.command_buffer];
            let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
            state.health.submit_and_wait(
                || {
                    self.inner
                        .device
                        .queue_submit(state.queue, &submits, state.fence)
                        .map_err(|error| vk_error("submitting Vulkan matvec", error))
                },
                || {
                    self.inner
                        .device
                        .wait_for_fences(&[state.fence], true, u64::MAX)
                        .map_err(|error| vk_error("waiting for Vulkan matvec", error))
                },
            )?;
        }
        read_f32(&self.inner.device, output, y)
    }
}

impl RoutedMoeProgram {
    /// Return the paired-load kernel variant pinned when this program was prepared.
    pub fn fast_variant(&self) -> RoutedMoeFastVariant {
        self.fast_variant
    }

    /// Return the command-recording strategy pinned when this program was prepared.
    pub fn batch_dispatch(&self) -> RoutedMoeBatchDispatch {
        self.batch_dispatch
    }

    /// Return the input and output hidden width accepted by this program.
    pub fn hidden(&self) -> usize {
        self.hidden
    }

    /// Return the intermediate expert width accepted by this program.
    pub fn expert_width(&self) -> usize {
        self.expert_width
    }

    /// Return the exact routed-expert count accepted by this program.
    pub fn routes(&self) -> usize {
        self.routes
    }

    /// Return the exact number of independent items recorded into this program.
    pub fn batch_len(&self) -> usize {
        self.batch_len
    }

    /// Return the exact logical, buffer, Vulkan-allocation, and slot accounting captured
    /// when this program was prepared.
    pub fn stats(&self) -> RoutedMoeProgramStats {
        self.stats
    }

    /// Execute the prerecorded routed-MoE program synchronously.
    ///
    /// `experts` may reorder or replace routes within the retained arena block, but it
    /// must preserve the program's context, arena identity, block, and `H`/`F`/`K`
    /// geometry. All fallible validation completes before mapped state is updated, and
    /// `y` is changed only after the submitted command buffer completes successfully.
    pub fn execute(
        &self,
        y: &mut [f32],
        x: &[f32],
        experts: &PreparedRoutedExperts,
        route_weights: &[f32],
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        if self.batch_len != 1 {
            return Err(DeviceError::Shape(format!(
                "singular routed-program execute requires batch length 1, but this program records {} items",
                self.batch_len
            )));
        }
        self.execute_batch(
            y,
            x,
            std::slice::from_ref(experts),
            route_weights,
            accumulate,
        )
    }

    /// Execute the fixed-size prerecorded routed-MoE batch synchronously.
    ///
    /// `xs` and `ys` are row-major `[N,H]`; `route_weights` is row-major `[N,K]`.
    /// Each item may supply a different route ordering or route replacement, provided
    /// every [`PreparedRoutedExperts`] value retains the program's exact context, arena,
    /// block, `H`, `F`, and `K`. All lengths, finite route weights, and absolute weight
    /// ranges for every item are validated before any persistently mapped write. The
    /// caller's `ys` is changed only after the complete batch finishes successfully.
    pub fn execute_batch(
        &self,
        ys: &mut [f32],
        xs: &[f32],
        experts: &[PreparedRoutedExperts],
        route_weights: &[f32],
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        if experts.len() != self.batch_len {
            return Err(DeviceError::Shape(format!(
                "routed-program batch length is {}, got {} expert selections",
                self.batch_len,
                experts.len()
            )));
        }
        let hidden_elements = self.batch_len.checked_mul(self.hidden).ok_or_else(|| {
            DeviceError::Shape("routed-program batched hidden length overflows".into())
        })?;
        if xs.len() != hidden_elements || ys.len() != hidden_elements {
            return Err(DeviceError::Shape(format!(
                "routed-program batch needs x={} y={} for N={} H={}, got x={} y={}",
                hidden_elements,
                hidden_elements,
                self.batch_len,
                self.hidden,
                xs.len(),
                ys.len()
            )));
        }
        let route_elements = self.batch_len.checked_mul(self.routes).ok_or_else(|| {
            DeviceError::Shape("routed-program batched route-weight length overflows".into())
        })?;
        if route_weights.len() != route_elements {
            return Err(DeviceError::Shape(format!(
                "routed-program batch needs {route_elements} route weights for N={} K={}, got {}",
                self.batch_len,
                self.routes,
                route_weights.len()
            )));
        }

        for (item, (experts, item_weights)) in experts
            .iter()
            .zip(route_weights.chunks_exact(self.routes))
            .enumerate()
        {
            if !Arc::ptr_eq(&self.context, &experts.arena.context) {
                return Err(backend_error(format!(
                    "routed-program item {item} experts belong to a different Vulkan context"
                )));
            }
            if !Arc::ptr_eq(&self.arena, &experts.arena) {
                return Err(backend_error(format!(
                    "routed-program item {item} experts belong to a different Vulkan weight arena"
                )));
            }
            if experts.block != self.block {
                return Err(backend_error(format!(
                    "routed-program item {item} uses arena block {}, but the program pins block {}",
                    experts.block, self.block
                )));
            }
            if (experts.hidden, experts.expert_width, experts.routes())
                != (self.hidden, self.expert_width, self.routes)
            {
                return Err(DeviceError::Shape(format!(
                    "routed-program item {item} requires H={} F={} K={}, got H={} F={} K={}",
                    self.hidden,
                    self.expert_width,
                    self.routes,
                    experts.hidden,
                    experts.expert_width,
                    experts.routes()
                )));
            }
            if let Some((route, weight)) = item_weights
                .iter()
                .copied()
                .enumerate()
                .find(|(_, weight)| !weight.is_finite())
            {
                return Err(DeviceError::Shape(format!(
                    "routed-program item {item} route {route} has non-finite weight {weight}"
                )));
            }
            validate_fast_routed_moe_weights(experts)?;
            validate_absolute_fast_routed_moe_offsets(
                experts.weight_offset,
                &experts.offsets_bytes,
                self.geometry.matrix_words,
                self.weight_capacity,
            )?;
        }

        let mut program_state = self.state.lock().map_err(|_| {
            backend_error("the routed-program mutex was poisoned by an earlier failed execution")
        })?;
        if program_state.poisoned {
            return Err(backend_error(
                "the routed program is poisoned after an uncertain Vulkan submission",
            ));
        }
        let queue_state = self.context.state.lock().map_err(|_| {
            backend_error("the Vulkan queue mutex was poisoned by an earlier failed dispatch")
        })?;
        queue_state.health.require_ready()?;
        let resources = &program_state.resources;
        let fence = resources.fence;
        if let Err(error) = unsafe { self.context.device.reset_fences(&[fence]) } {
            program_state.poisoned = true;
            return Err(vk_error("resetting the routed-program fence", error));
        }

        for (item, ((x, y), (experts, item_weights))) in xs
            .chunks_exact(self.hidden)
            .zip(ys.chunks_exact(self.hidden))
            .zip(experts.iter().zip(route_weights.chunks_exact(self.routes)))
            .enumerate()
        {
            resources
                .input
                .as_ref()
                .expect("routed-program input remains allocated")
                .write_f32_at(self.geometry.input.item_stride * item, x);
            resources
                .metadata
                .as_ref()
                .expect("routed-program metadata remains allocated")
                .write_routed_metadata_at(
                    self.geometry.metadata.item_stride * item,
                    experts.weight_offset,
                    &experts.offsets_bytes,
                    item_weights,
                );
            if accumulate {
                resources
                    .output
                    .as_ref()
                    .expect("routed-program output remains allocated")
                    .write_f32_at(self.geometry.output.item_stride * item, y);
            }
        }
        if self.batch_dispatch == RoutedMoeBatchDispatch::ExpertTile4 {
            resources
                .metadata
                .as_ref()
                .expect("routed-program metadata remains allocated")
                .write_tile4_schedule(self.geometry);
        }

        let command_buffer = resources.command_buffers[usize::from(accumulate)];
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        if let Err(error) = queue_state.health.submit_and_wait(
            || unsafe {
                self.context
                    .device
                    .queue_submit(queue_state.queue, &submits, fence)
                    .map_err(|error| vk_error("submitting the routed program", error))
            },
            || unsafe {
                self.context
                    .device
                    .wait_for_fences(&[fence], true, u64::MAX)
                    .map_err(|error| vk_error("waiting for the routed program", error))
            },
        ) {
            program_state.poisoned = true;
            return Err(error);
        }
        drop(queue_state);
        let output = program_state
            .resources
            .output
            .as_ref()
            .expect("routed-program output remains allocated");
        for (item, y) in ys.chunks_exact_mut(self.hidden).enumerate() {
            output.read_f32_at(self.geometry.output.item_stride * item, y);
        }
        Ok(())
    }
}

impl Drop for RoutedMoeProgram {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let queue_state = lock_unpoisoned(&self.context.state);
        // Every successful execution waits its private fence before returning.
        // Include shared health so even a panic before the private poison flag
        // is set requires device-wide idle before destroying private resources.
        if (state.poisoned || queue_state.health.uncertain.get())
            && !context_idle_for_teardown(&self.context, &queue_state, "routed program")
        {
            return;
        }
        destroy_routed_moe_program_resources(&self.context, &mut state.resources);
    }
}

impl PersistentlyMappedBuffer {
    fn write_f32_at(&self, byte_offset: usize, values: &[f32]) {
        debug_assert!(byte_offset.is_multiple_of(std::mem::align_of::<f32>()));
        debug_assert!(
            byte_offset.saturating_add(values.len().saturating_mul(4)) <= self.allocation.capacity
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                values.as_ptr(),
                self.mapped.as_ptr().add(byte_offset).cast::<f32>(),
                values.len(),
            );
        }
    }

    fn write_routed_metadata_at(
        &self,
        byte_offset: usize,
        weight_offset: usize,
        offsets_bytes: &[[u32; 3]],
        route_weights: &[f32],
    ) {
        debug_assert_eq!(offsets_bytes.len(), route_weights.len());
        debug_assert!(byte_offset.is_multiple_of(std::mem::align_of::<u32>()));
        debug_assert!(
            byte_offset.saturating_add(offsets_bytes.len().saturating_mul(4).saturating_mul(4))
                <= self.allocation.capacity
        );
        unsafe {
            let words = std::slice::from_raw_parts_mut(
                self.mapped.as_ptr().add(byte_offset).cast::<u32>(),
                offsets_bytes.len() * 4,
            );
            for (route, (offsets, route_weight)) in
                offsets_bytes.iter().zip(route_weights).enumerate()
            {
                let base = route * 4;
                for (destination, relative) in words[base..base + 3].iter_mut().zip(offsets) {
                    let absolute = weight_offset
                        .checked_add(*relative as usize)
                        .expect("routed-program offsets were validated before mapping");
                    *destination = u32::try_from(absolute / DESCRIPTOR_WORD_BYTES)
                        .expect("routed-program word offsets were validated before mapping");
                }
                words[base + 3] = route_weight.to_bits();
            }
        }
    }

    fn write_tile4_schedule(&self, geometry: RoutedMoeProgramGeometry) -> u32 {
        let tile = geometry
            .tile4
            .expect("Tile4 program geometry was validated before allocation");
        debug_assert!(self.allocation.capacity >= geometry.metadata.total_bytes);
        debug_assert!(geometry
            .metadata
            .total_bytes
            .is_multiple_of(DESCRIPTOR_WORD_BYTES));
        let total_words = geometry.metadata.total_bytes / DESCRIPTOR_WORD_BYTES;
        let schedule_start = usize::try_from(tile.tile_word_offset)
            .expect("validated Tile4 schedule offset fits usize");
        let occurrence_count = geometry
            .batch_len
            .checked_mul(geometry.routes as usize)
            .expect("validated Tile4 occurrence count fits usize");
        unsafe {
            let words =
                std::slice::from_raw_parts_mut(self.mapped.as_ptr().cast::<u32>(), total_words);
            let route_word_count = occurrence_count
                .checked_mul(4)
                .expect("validated Tile4 route metadata fits usize");
            debug_assert_eq!(
                schedule_start,
                route_word_count + ROUTED_MOE_TILE4_INDIRECT_WORDS
            );
            debug_assert!(schedule_start + tile.schedule_words <= words.len());
            let tile_count = {
                let (metadata_and_commands, schedule) = words.split_at_mut(schedule_start);
                write_tile4_records(
                    &metadata_and_commands[..route_word_count],
                    &mut schedule[..tile.schedule_words],
                )
            };
            debug_assert!(tile_count <= tile.max_tiles);
            let gate_command = usize::try_from(tile.gate_indirect_offset / 4)
                .expect("validated gate indirect offset fits usize");
            let down_command = usize::try_from(tile.down_indirect_offset / 4)
                .expect("validated down indirect offset fits usize");
            words[gate_command..gate_command + 3].copy_from_slice(&[
                geometry.expert_width,
                tile_count,
                1,
            ]);
            words[down_command..down_command + 3].copy_from_slice(&[
                geometry.hidden,
                tile_count,
                1,
            ]);
            tile_count
        }
    }

    fn read_f32_at(&self, byte_offset: usize, output: &mut [f32]) {
        debug_assert!(byte_offset.is_multiple_of(std::mem::align_of::<f32>()));
        debug_assert!(
            byte_offset.saturating_add(output.len().saturating_mul(4)) <= self.allocation.capacity
        );
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.mapped.as_ptr().add(byte_offset).cast::<f32>(),
                output.as_mut_ptr(),
                output.len(),
            );
        }
    }
}

/// Build stable expert-major Tile4 records without allocating.
///
/// Expert groups follow the first occurrence of each exact absolute gate/up/down
/// tuple. Occurrences within a group remain in ascending flat `[item, route]` order,
/// and every group is emitted as contiguous chunks of at most four. Inactive lanes
/// repeat the first occurrence in their record so even eagerly calculated shader
/// indices remain valid.
fn write_tile4_records(route_metadata: &[u32], records: &mut [u32]) -> u32 {
    debug_assert!(route_metadata.len().is_multiple_of(4));
    let occurrence_count = route_metadata.len() / 4;
    debug_assert!(
        records.len()
            >= occurrence_count
                .checked_mul(ROUTED_MOE_TILE4_RECORD_WORDS)
                .expect("validated Tile4 record capacity fits usize")
    );

    let mut tile_count = 0usize;
    for leader in 0..occurrence_count {
        let leader_base = leader * 4;
        let key = [
            route_metadata[leader_base],
            route_metadata[leader_base + 1],
            route_metadata[leader_base + 2],
        ];
        if (0..leader).any(|prior| {
            let prior_base = prior * 4;
            route_metadata[prior_base..prior_base + 3] == key
        }) {
            continue;
        }

        let mut chunk_count = 0usize;
        for occurrence in leader..occurrence_count {
            let metadata_base = occurrence * 4;
            if route_metadata[metadata_base..metadata_base + 3] != key {
                continue;
            }
            let occurrence =
                u32::try_from(occurrence).expect("validated Tile4 occurrence count fits GLSL uint");
            let record_base = tile_count * ROUTED_MOE_TILE4_RECORD_WORDS;
            if chunk_count == 0 {
                records[record_base..record_base + 3].copy_from_slice(&key);
                records[record_base + 3] = 0;
                records[record_base + 4..record_base + 8].fill(occurrence);
            }
            records[record_base + 4 + chunk_count] = occurrence;
            chunk_count += 1;
            records[record_base + 3] = u32::try_from(chunk_count).expect("Tile4 width fits u32");
            if chunk_count == ROUTED_MOE_TILE4_WIDTH {
                tile_count += 1;
                chunk_count = 0;
            }
        }
        if chunk_count != 0 {
            tile_count += 1;
        }
    }
    u32::try_from(tile_count).expect("validated Tile4 tile capacity fits u32")
}

impl Drop for RoutedMoeProgramInit {
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        // Construction never submits either command buffer, so partially initialized
        // private resources cannot be in flight and need no device-wide idle.
        destroy_routed_moe_program_resources(&self.context, &mut self.resources);
    }
}

impl Context {
    fn new() -> Result<Self, DeviceError> {
        let mut init = InitGuard::default();
        let entry = unsafe { Entry::load() }
            .map_err(|error| backend_error(format!("loading libvulkan: {error}")))?;
        init.entry = Some(entry);

        let application = vk::ApplicationInfo::default()
            .application_name(c"beholdd")
            .application_version(1)
            .engine_name(c"beholdd")
            .engine_version(1)
            .api_version(REQUIRED_VULKAN_API_VERSION);
        let instance = unsafe {
            init.entry
                .as_ref()
                .expect("entry was initialized")
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(&application),
                    None,
                )
        }
        .map_err(|error| vk_error("creating the Vulkan instance", error))?;
        init.instance = Some(instance);
        let instance = init.instance.as_ref().expect("instance was initialized");

        let mut candidates = Vec::new();
        let mut incompatible_devices = Vec::new();
        for physical in unsafe { instance.enumerate_physical_devices() }
            .map_err(|error| vk_error("enumerating physical devices", error))?
        {
            let properties = unsafe { instance.get_physical_device_properties(physical) };
            if properties.device_type == vk::PhysicalDeviceType::CPU {
                continue;
            }
            // Instance support does not imply physical-device support. Use only
            // core-1.0 queries until this check passes, before properties2, logical
            // device creation, memory-requirements2, or Vulkan-1.2 shader modules.
            if !supports_required_vulkan_api(properties.api_version) {
                let name =
                    unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }.to_string_lossy();
                incompatible_devices.push(format!(
                    "{name} (API {}.{}.{}, variant {})",
                    vk::api_version_major(properties.api_version),
                    vk::api_version_minor(properties.api_version),
                    vk::api_version_patch(properties.api_version),
                    vk::api_version_variant(properties.api_version),
                ));
                continue;
            }
            let queues = unsafe { instance.get_physical_device_queue_family_properties(physical) };
            for (index, queue) in queues.iter().enumerate() {
                if !queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                    continue;
                }
                let device_rank = match properties.device_type {
                    // Integrated first on the unified-memory machines beholdd targets;
                    // discrete remains ahead of virtual/other devices.
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 4,
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    _ => 1,
                };
                let dedicated = u32::from(!queue.queue_flags.contains(vk::QueueFlags::GRAPHICS));
                candidates.push((
                    device_rank,
                    dedicated,
                    physical,
                    index as u32,
                    properties,
                    queue.timestamp_valid_bits,
                ));
            }
        }
        let (_, _, physical, queue_family, properties, timestamp_valid_bits) = candidates
            .into_iter()
            .max_by_key(|(device, dedicated, _, _, _, _)| (*device, *dedicated))
            .ok_or_else(|| {
                let skipped = if incompatible_devices.is_empty() {
                    String::new()
                } else {
                    format!("; incompatible devices: {}", incompatible_devices.join(", "))
                };
                backend_error(format!(
                    "no non-CPU Vulkan 1.2+ compute queue is available{skipped}; update the GPU driver or select the CPU backend explicitly"
                ))
            })?;
        let device_name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let mut subgroup_properties = vk::PhysicalDeviceSubgroupProperties::default();
        let mut extended_properties =
            vk::PhysicalDeviceProperties2::default().push_next(&mut subgroup_properties);
        unsafe {
            instance.get_physical_device_properties2(physical, &mut extended_properties);
        }
        let subgroup_arithmetic = subgroup_properties
            .supported_stages
            .contains(vk::ShaderStageFlags::COMPUTE)
            && subgroup_properties
                .supported_operations
                .contains(vk::SubgroupFeatureFlags::BASIC | vk::SubgroupFeatureFlags::ARITHMETIC);
        if properties.limits.max_compute_work_group_invocations < 64
            || properties.limits.max_compute_work_group_size[0] < 64
        {
            return Err(backend_error(format!(
                "{device_name} cannot run the 64-lane matvec workgroup"
            )));
        }
        if properties.limits.max_push_constants_size < 24 {
            return Err(backend_error(format!(
                "{device_name} exposes only {} push-constant bytes; matvec needs 24",
                properties.limits.max_push_constants_size
            )));
        }
        let priorities = [1.0f32];
        let queue_info = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(&priorities)];
        let device = unsafe {
            instance.create_device(
                physical,
                &vk::DeviceCreateInfo::default().queue_create_infos(&queue_info),
                None,
            )
        }
        .map_err(|error| vk_error("creating the logical device", error))?;
        init.device = Some(device);
        let device = init
            .device
            .as_ref()
            .expect("logical device was initialized");
        let queue = unsafe { device.get_device_queue(queue_family, 0) };

        let bindings = [0, 1, 2, 3].map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
        });
        init.descriptor_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .map_err(|error| vk_error("creating the descriptor layout", error))?;
        let set_layouts = [init.descriptor_set_layout];
        let push_range = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(24)];
        init.pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&set_layouts)
                    .push_constant_ranges(&push_range),
                None,
            )
        }
        .map_err(|error| vk_error("creating the pipeline layout", error))?;

        init.q8_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/q8_0_matvec.spv")),
            "Q8_0",
        )?;
        init.bf16_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bf16_matvec.spv")),
            "BF16",
        )?;
        init.moe_gate_up_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bf16_routed_moe_gate_up.spv")),
            "BF16 routed-MoE gate/up",
        )?;
        init.moe_down_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bf16_routed_moe_down.spv")),
            "BF16 routed-MoE down",
        )?;
        init.moe_reduce_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bf16_routed_moe_reduce.spv")),
            "BF16 routed-MoE reduction",
        )?;
        init.moe_gate_up_fast_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_gate_up_fast.spv"
            )),
            "fast BF16 routed-MoE gate/up",
        )?;
        init.moe_down_fast_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bf16_routed_moe_down_fast.spv")),
            "fast BF16 routed-MoE down",
        )?;
        init.moe_reduce_fast_pipeline = create_compute_pipeline(
            device,
            init.pipeline_layout,
            include_bytes!(concat!(env!("OUT_DIR"), "/bf16_routed_moe_reduce_fast.spv")),
            "fast BF16 routed-MoE reduction",
        )?;

        let pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(16)];
        init.descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(4)
                    .pool_sizes(&pool_size),
                None,
            )
        }
        .map_err(|error| vk_error("creating the descriptor pool", error))?;
        let descriptor_layouts = [init.descriptor_set_layout; 4];
        let descriptor_sets = unsafe {
            device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(init.descriptor_pool)
                    .set_layouts(&descriptor_layouts),
            )
        }
        .map_err(|error| vk_error("allocating descriptor sets", error))?;
        init.descriptor_sets.copy_from_slice(&descriptor_sets);
        init.command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )
        }
        .map_err(|error| vk_error("creating the command pool", error))?;
        init.command_buffer = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(init.command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(|error| vk_error("allocating the command buffer", error))?[0];
        init.fence = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                None,
            )
        }
        .map_err(|error| vk_error("creating the dispatch fence", error))?;
        let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical) };
        let timestamp_properties =
            usable_timestamp_properties(timestamp_valid_bits, properties.limits.timestamp_period);
        // Complete fallible conversions before disarming the initialization owner.
        let storage_buffer_alignment =
            usize::try_from(properties.limits.min_storage_buffer_offset_alignment)
                .map_err(|_| backend_error("Vulkan storage-buffer alignment exceeds usize"))?;

        Ok(Self {
            _entry: ManuallyDrop::new(init.entry.take().expect("entry is owned")),
            instance: init.instance.take().expect("instance is owned"),
            device: init.device.take().expect("device is owned"),
            device_name,
            driver_version: properties.driver_version,
            queue_family,
            memory_properties,
            max_group_count: properties.limits.max_compute_work_group_count,
            max_work_group_invocations: properties.limits.max_compute_work_group_invocations,
            max_work_group_size_x: properties.limits.max_compute_work_group_size[0],
            subgroup_arithmetic,
            max_storage_buffer_range: properties.limits.max_storage_buffer_range as usize,
            storage_buffer_alignment,
            max_memory_allocations: properties.limits.max_memory_allocation_count,
            timestamp_properties,
            live_allocations: AtomicU32::new(0),
            state: Mutex::new(QueueState {
                health: QueueHealth::default(),
                queue,
                descriptor_pool: std::mem::replace(
                    &mut init.descriptor_pool,
                    vk::DescriptorPool::null(),
                ),
                descriptor_set_layout: std::mem::replace(
                    &mut init.descriptor_set_layout,
                    vk::DescriptorSetLayout::null(),
                ),
                descriptor_sets: init.descriptor_sets,
                pipeline_layout: std::mem::replace(
                    &mut init.pipeline_layout,
                    vk::PipelineLayout::null(),
                ),
                q8_pipeline: std::mem::replace(&mut init.q8_pipeline, vk::Pipeline::null()),
                bf16_pipeline: std::mem::replace(&mut init.bf16_pipeline, vk::Pipeline::null()),
                moe_gate_up_pipeline: std::mem::replace(
                    &mut init.moe_gate_up_pipeline,
                    vk::Pipeline::null(),
                ),
                moe_down_pipeline: std::mem::replace(
                    &mut init.moe_down_pipeline,
                    vk::Pipeline::null(),
                ),
                moe_reduce_pipeline: std::mem::replace(
                    &mut init.moe_reduce_pipeline,
                    vk::Pipeline::null(),
                ),
                moe_gate_up_fast_pipeline: std::mem::replace(
                    &mut init.moe_gate_up_fast_pipeline,
                    vk::Pipeline::null(),
                ),
                moe_down_fast_pipeline: std::mem::replace(
                    &mut init.moe_down_fast_pipeline,
                    vk::Pipeline::null(),
                ),
                moe_reduce_fast_pipeline: std::mem::replace(
                    &mut init.moe_reduce_fast_pipeline,
                    vk::Pipeline::null(),
                ),
                experimental_fast_pipelines: [None; 5],
                item_dimensional_fast_pipelines: None,
                tile4_fast_pipelines: None,
                command_pool: std::mem::replace(&mut init.command_pool, vk::CommandPool::null()),
                command_buffer: init.command_buffer,
                fence: std::mem::replace(&mut init.fence, vk::Fence::null()),
                timestamp_query_pool: None,
                input: None,
                output: None,
                moe_metadata: None,
                moe_hidden: None,
                moe_routes: None,
            }),
        })
    }
}

fn supports_required_vulkan_api(version: u32) -> bool {
    // A nonzero variant is a different Vulkan API (for example Vulkan SC), not a
    // newer version compatible with the standard Vulkan calls and SPIR-V here.
    vk::api_version_variant(version) == 0 && version >= REQUIRED_VULKAN_API_VERSION
}

impl Drop for Context {
    fn drop(&mut self) {
        let state = match self.state.get_mut() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        let completion = unsafe { self.device.device_wait_idle() };
        if !teardown_is_safe(completion, |error| {
            report_teardown_quarantine("context", error);
        }) {
            // All Vulkan handles and the ManuallyDrop loader deliberately remain
            // alive. Ordinary Rust metadata may be dropped, but it cannot free GPU
            // resources or unload code still used by potentially pending work.
            return;
        }
        unsafe {
            if let Some(input) = state.input.take() {
                input.destroy(&self.device, &self.live_allocations);
            }
            if let Some(output) = state.output.take() {
                output.destroy(&self.device, &self.live_allocations);
            }
            if let Some(metadata) = state.moe_metadata.take() {
                metadata.destroy(&self.device, &self.live_allocations);
            }
            if let Some(hidden) = state.moe_hidden.take() {
                hidden.destroy(&self.device, &self.live_allocations);
            }
            if let Some(routes) = state.moe_routes.take() {
                routes.destroy(&self.device, &self.live_allocations);
            }
            if let Some(query_pool) = state.timestamp_query_pool.take() {
                self.device.destroy_query_pool(query_pool, None);
            }
            self.device.destroy_fence(state.fence, None);
            self.device.destroy_command_pool(state.command_pool, None);
            self.device.destroy_pipeline(state.q8_pipeline, None);
            self.device.destroy_pipeline(state.bf16_pipeline, None);
            self.device
                .destroy_pipeline(state.moe_gate_up_pipeline, None);
            self.device.destroy_pipeline(state.moe_down_pipeline, None);
            self.device
                .destroy_pipeline(state.moe_reduce_pipeline, None);
            self.device
                .destroy_pipeline(state.moe_gate_up_fast_pipeline, None);
            self.device
                .destroy_pipeline(state.moe_down_fast_pipeline, None);
            self.device
                .destroy_pipeline(state.moe_reduce_fast_pipeline, None);
            for pipelines in state.experimental_fast_pipelines.iter_mut() {
                if let Some(pipelines) = pipelines.take() {
                    self.device.destroy_pipeline(pipelines.gate_up, None);
                    self.device.destroy_pipeline(pipelines.down, None);
                }
            }
            if let Some(pipelines) = state.item_dimensional_fast_pipelines.take() {
                self.device.destroy_pipeline(pipelines.gate_up, None);
                self.device.destroy_pipeline(pipelines.down, None);
                self.device.destroy_pipeline(pipelines.reduce, None);
            }
            if let Some(pipelines) = state.tile4_fast_pipelines.take() {
                self.device.destroy_pipeline(pipelines.gate_up, None);
                self.device.destroy_pipeline(pipelines.down, None);
            }
            self.device
                .destroy_pipeline_layout(state.pipeline_layout, None);
            self.device
                .destroy_descriptor_pool(state.descriptor_pool, None);
            self.device
                .destroy_descriptor_set_layout(state.descriptor_set_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
            ManuallyDrop::drop(&mut self._entry);
        }
    }
}

#[derive(Default)]
struct InitGuard {
    entry: Option<Entry>,
    instance: Option<Instance>,
    device: Option<Device>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_sets: [vk::DescriptorSet; 4],
    pipeline_layout: vk::PipelineLayout,
    q8_pipeline: vk::Pipeline,
    bf16_pipeline: vk::Pipeline,
    moe_gate_up_pipeline: vk::Pipeline,
    moe_down_pipeline: vk::Pipeline,
    moe_reduce_pipeline: vk::Pipeline,
    moe_gate_up_fast_pipeline: vk::Pipeline,
    moe_down_fast_pipeline: vk::Pipeline,
    moe_reduce_fast_pipeline: vk::Pipeline,
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
    fence: vk::Fence,
}

impl Drop for InitGuard {
    fn drop(&mut self) {
        if let Some(device) = self.device.take() {
            unsafe {
                // Context initialization never submits work. Its partial resources
                // can be destroyed without asking an idle API that may itself fail.
                if self.fence != vk::Fence::null() {
                    device.destroy_fence(self.fence, None);
                }
                if self.command_pool != vk::CommandPool::null() {
                    device.destroy_command_pool(self.command_pool, None);
                }
                if self.q8_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.q8_pipeline, None);
                }
                if self.bf16_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.bf16_pipeline, None);
                }
                if self.moe_gate_up_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.moe_gate_up_pipeline, None);
                }
                if self.moe_down_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.moe_down_pipeline, None);
                }
                if self.moe_reduce_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.moe_reduce_pipeline, None);
                }
                if self.moe_gate_up_fast_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.moe_gate_up_fast_pipeline, None);
                }
                if self.moe_down_fast_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.moe_down_fast_pipeline, None);
                }
                if self.moe_reduce_fast_pipeline != vk::Pipeline::null() {
                    device.destroy_pipeline(self.moe_reduce_fast_pipeline, None);
                }
                if self.pipeline_layout != vk::PipelineLayout::null() {
                    device.destroy_pipeline_layout(self.pipeline_layout, None);
                }
                if self.descriptor_pool != vk::DescriptorPool::null() {
                    device.destroy_descriptor_pool(self.descriptor_pool, None);
                }
                if self.descriptor_set_layout != vk::DescriptorSetLayout::null() {
                    device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
                }
                device.destroy_device(None);
            }
        }
        if let Some(instance) = self.instance.take() {
            unsafe { instance.destroy_instance(None) };
        }
    }
}

fn create_compute_pipeline(
    device: &Device,
    layout: vk::PipelineLayout,
    shader_bytes: &[u8],
    format: &str,
) -> Result<vk::Pipeline, DeviceError> {
    let shader_words = ash::util::read_spv(&mut Cursor::new(shader_bytes))
        .map_err(|error| backend_error(format!("reading embedded {format} SPIR-V: {error}")))?;
    let shader = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&shader_words),
            None,
        )
    }
    .map_err(|error| vk_error(format!("creating the {format} shader module"), error))?;
    let pipeline_info = [vk::ComputePipelineCreateInfo::default()
        .stage(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(shader)
                .name(c"main"),
        )
        .layout(layout)];
    let created =
        unsafe { device.create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None) };
    unsafe { device.destroy_shader_module(shader, None) };
    match created {
        Ok(pipelines) => Ok(pipelines[0]),
        Err((pipelines, error)) => {
            for pipeline in pipelines {
                if pipeline != vk::Pipeline::null() {
                    unsafe { device.destroy_pipeline(pipeline, None) };
                }
            }
            Err(vk_error(
                format!("creating the {format} compute pipeline"),
                error,
            ))
        }
    }
}

fn routed_moe_pipelines(
    context: &Context,
    state: &mut QueueState,
    run: RoutedMoeRun,
) -> Result<(vk::Pipeline, vk::Pipeline, vk::Pipeline), DeviceError> {
    if run.kernel == RoutedMoeKernel::Guarded {
        return Ok((
            state.moe_gate_up_pipeline,
            state.moe_down_pipeline,
            state.moe_reduce_pipeline,
        ));
    }
    if run.fast_variant == RoutedMoeFastVariant::QualifiedTree64 {
        return Ok((
            state.moe_gate_up_fast_pipeline,
            state.moe_down_fast_pipeline,
            state.moe_reduce_fast_pipeline,
        ));
    }

    let workgroup_size = run.fast_variant.workgroup_size();
    if workgroup_size > context.max_work_group_invocations
        || workgroup_size > context.max_work_group_size_x
    {
        return Err(backend_error(format!(
            "{} cannot run {}: requested {workgroup_size} lanes, limits are {} invocations and {} in X",
            context.device_name,
            run.fast_variant.label(),
            context.max_work_group_invocations,
            context.max_work_group_size_x,
        )));
    }
    if run.fast_variant.uses_subgroup_arithmetic() && !context.subgroup_arithmetic {
        return Err(backend_error(format!(
            "{} does not expose compute BASIC|ARITHMETIC subgroup operations required by {}",
            context.device_name,
            run.fast_variant.label(),
        )));
    }

    let index = run
        .fast_variant
        .experimental_index()
        .expect("the qualified variant returned above");
    if let Some(pipelines) = state.experimental_fast_pipelines[index] {
        return Ok((
            pipelines.gate_up,
            pipelines.down,
            state.moe_reduce_fast_pipeline,
        ));
    }
    let (gate_up_shader, down_shader): (&[u8], &[u8]) = match run.fast_variant {
        RoutedMoeFastVariant::Tree32 => (
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_gate_up_fast_wg32.spv"
            )),
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_down_fast_wg32.spv"
            )),
        ),
        RoutedMoeFastVariant::Tree128 => (
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_gate_up_fast_wg128.spv"
            )),
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_down_fast_wg128.spv"
            )),
        ),
        RoutedMoeFastVariant::Subgroup32 => (
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_gate_up_subgroup_wg32.spv"
            )),
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_down_subgroup_wg32.spv"
            )),
        ),
        RoutedMoeFastVariant::Subgroup64 => (
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_gate_up_subgroup_wg64.spv"
            )),
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_down_subgroup_wg64.spv"
            )),
        ),
        RoutedMoeFastVariant::Subgroup128 => (
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_gate_up_subgroup_wg128.spv"
            )),
            include_bytes!(concat!(
                env!("OUT_DIR"),
                "/bf16_routed_moe_down_subgroup_wg128.spv"
            )),
        ),
        RoutedMoeFastVariant::QualifiedTree64 => unreachable!("returned above"),
    };
    let gate_up = create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        gate_up_shader,
        &format!("{} routed-MoE gate/up", run.fast_variant.label()),
    )?;
    let down = match create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        down_shader,
        &format!("{} routed-MoE down", run.fast_variant.label()),
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe { context.device.destroy_pipeline(gate_up, None) };
            return Err(error);
        }
    };
    let pipelines = RoutedMoePipelinePair { gate_up, down };
    state.experimental_fast_pipelines[index] = Some(pipelines);
    Ok((gate_up, down, state.moe_reduce_fast_pipeline))
}

fn item_dimensional_routed_moe_pipelines(
    context: &Context,
    state: &mut QueueState,
) -> Result<RoutedMoePipelineTriple, DeviceError> {
    const WORKGROUP_SIZE: u32 = 128;
    if WORKGROUP_SIZE > context.max_work_group_invocations
        || WORKGROUP_SIZE > context.max_work_group_size_x
    {
        return Err(backend_error(format!(
            "{} cannot run item-dimensional Tree128 routed-MoE: requested {WORKGROUP_SIZE} lanes, limits are {} invocations and {} in X",
            context.device_name,
            context.max_work_group_invocations,
            context.max_work_group_size_x,
        )));
    }
    if let Some(pipelines) = state.item_dimensional_fast_pipelines {
        return Ok(pipelines);
    }

    let gate_up = create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/bf16_routed_moe_gate_up_fast_batch_wg128.spv"
        )),
        "item-dimensional Tree128 routed-MoE gate/up",
    )?;
    let down = match create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/bf16_routed_moe_down_fast_batch_wg128.spv"
        )),
        "item-dimensional Tree128 routed-MoE down",
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe { context.device.destroy_pipeline(gate_up, None) };
            return Err(error);
        }
    };
    let reduce = match create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/bf16_routed_moe_reduce_fast_batch.spv"
        )),
        "item-dimensional routed-MoE reduction",
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe {
                context.device.destroy_pipeline(gate_up, None);
                context.device.destroy_pipeline(down, None);
            }
            return Err(error);
        }
    };
    let pipelines = RoutedMoePipelineTriple {
        gate_up,
        down,
        reduce,
    };
    state.item_dimensional_fast_pipelines = Some(pipelines);
    Ok(pipelines)
}

fn tile4_routed_moe_pipelines(
    context: &Context,
    state: &mut QueueState,
) -> Result<RoutedMoePipelinePair, DeviceError> {
    const WORKGROUP_SIZE: u32 = 128;
    if WORKGROUP_SIZE > context.max_work_group_invocations
        || WORKGROUP_SIZE > context.max_work_group_size_x
    {
        return Err(backend_error(format!(
            "{} cannot run expert-major Tile4 routed-MoE: requested {WORKGROUP_SIZE} lanes, limits are {} invocations and {} in X",
            context.device_name,
            context.max_work_group_invocations,
            context.max_work_group_size_x,
        )));
    }
    if let Some(pipelines) = state.tile4_fast_pipelines {
        return Ok(pipelines);
    }
    let gate_up = create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/bf16_routed_moe_gate_up_fast_tile4_wg128.spv"
        )),
        "expert-major Tile4 Tree128 routed-MoE gate/up",
    )?;
    let down = match create_compute_pipeline(
        &context.device,
        state.pipeline_layout,
        include_bytes!(concat!(
            env!("OUT_DIR"),
            "/bf16_routed_moe_down_fast_tile4_wg128.spv"
        )),
        "expert-major Tile4 Tree128 routed-MoE down",
    ) {
        Ok(pipeline) => pipeline,
        Err(error) => {
            unsafe { context.device.destroy_pipeline(gate_up, None) };
            return Err(error);
        }
    };
    let pipelines = RoutedMoePipelinePair { gate_up, down };
    state.tile4_fast_pipelines = Some(pipelines);
    Ok(pipelines)
}

fn routed_moe_program_geometry(
    context: &Context,
    experts: &PreparedRoutedExperts,
    batch_len: NonZeroUsize,
    batch_dispatch: RoutedMoeBatchDispatch,
) -> Result<RoutedMoeProgramGeometry, DeviceError> {
    let hidden = u32::try_from(experts.hidden)
        .map_err(|_| DeviceError::Shape("routed-program hidden width exceeds u32".into()))?;
    let expert_width = u32::try_from(experts.expert_width)
        .map_err(|_| DeviceError::Shape("routed-program expert width exceeds u32".into()))?;
    let routes = u32::try_from(experts.routes())
        .map_err(|_| DeviceError::Shape("routed-program route count exceeds u32".into()))?;
    let hidden_elements = experts
        .routes()
        .checked_mul(experts.expert_width)
        .ok_or_else(|| DeviceError::Shape("routed-program hidden scratch size overflows".into()))?;
    let route_elements = experts
        .routes()
        .checked_mul(experts.hidden)
        .ok_or_else(|| DeviceError::Shape("routed-program output scratch size overflows".into()))?;
    let metadata_words = experts
        .routes()
        .checked_mul(4)
        .ok_or_else(|| DeviceError::Shape("routed-program metadata size overflows".into()))?;
    u32::try_from(metadata_words)
        .map_err(|_| DeviceError::Shape("routed-program metadata word count exceeds u32".into()))?;
    let byte_len = |elements: usize, purpose: &str| {
        elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| DeviceError::Shape(format!("{purpose} byte count overflows")))
    };
    let input_bytes = byte_len(experts.hidden, "routed-program input")?;
    let output_bytes = byte_len(experts.hidden, "routed-program output")?;
    let metadata_bytes = metadata_words
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| DeviceError::Shape("routed-program metadata byte count overflows".into()))?;
    let hidden_bytes = byte_len(hidden_elements, "routed-program hidden scratch")?;
    let route_bytes = byte_len(route_elements, "routed-program output scratch")?;
    // The fast gate/up and down shaders expose f32 buffers as vec2 arrays, so their
    // descriptor bases require eight-byte alignment even when Vulkan reports a smaller
    // storage-buffer-offset minimum. Using the same conservative stride for all five
    // buffers keeps every per-item descriptor independently bindable.
    let item_alignment = context
        .storage_buffer_alignment
        .max(2 * std::mem::size_of::<f32>())
        .max(DESCRIPTOR_WORD_BYTES);
    let buffer_geometry = |bytes, purpose| {
        routed_moe_buffer_geometry(
            bytes,
            batch_len,
            item_alignment,
            context.max_storage_buffer_range,
            &context.device_name,
            purpose,
        )
    };
    let input = buffer_geometry(input_bytes, "input")?;
    let output = buffer_geometry(output_bytes, "output")?;
    let mut metadata = buffer_geometry(metadata_bytes, "metadata")?;
    let hidden_scratch = buffer_geometry(hidden_bytes, "hidden scratch")?;
    let route_scratch = buffer_geometry(route_bytes, "route scratch")?;
    let matrix_elements = experts
        .hidden
        .checked_mul(experts.expert_width)
        .ok_or_else(|| {
            DeviceError::Shape("routed-program matrix element count overflows".into())
        })?;
    if !matrix_elements.is_multiple_of(2) {
        return Err(DeviceError::Shape(
            "routed-program matrix does not contain whole BF16 word pairs".into(),
        ));
    }
    let matrix_words = u32::try_from(matrix_elements / 2)
        .map_err(|_| DeviceError::Shape("routed-program matrix word count exceeds u32".into()))?;
    let dispatch = routed_moe_dispatch(
        RoutedMoeKernel::Fast,
        hidden,
        expert_width,
        routes,
        context.max_group_count,
        &context.device_name,
    )?;
    let (metadata_extra_bytes, tile4) = if batch_dispatch == RoutedMoeBatchDispatch::ExpertTile4 {
        let occurrence_count = batch_len
            .get()
            .checked_mul(experts.routes())
            .ok_or_else(|| {
                DeviceError::Shape("Tile4 routed-program occurrence count overflows".into())
            })?;
        let max_tiles = u32::try_from(occurrence_count).map_err(|_| {
            DeviceError::Shape("Tile4 routed-program tile count exceeds u32".into())
        })?;
        let route_words = metadata_words.checked_mul(batch_len.get()).ok_or_else(|| {
            DeviceError::Shape("Tile4 routed-program route metadata overflows".into())
        })?;
        let tile_word_offset = route_words
            .checked_add(ROUTED_MOE_TILE4_INDIRECT_WORDS)
            .ok_or_else(|| {
                DeviceError::Shape("Tile4 routed-program indirect metadata overflows".into())
            })?;
        let schedule_words = occurrence_count
            .checked_mul(ROUTED_MOE_TILE4_RECORD_WORDS)
            .ok_or_else(|| {
                DeviceError::Shape("Tile4 routed-program schedule size overflows".into())
            })?;
        let total_words = tile_word_offset
            .checked_add(schedule_words)
            .ok_or_else(|| {
                DeviceError::Shape("Tile4 routed-program metadata size overflows".into())
            })?;
        u32::try_from(total_words).map_err(|_| {
            DeviceError::Shape("Tile4 routed-program metadata exceeds GLSL uint indexing".into())
        })?;
        let total_bytes = total_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                DeviceError::Shape("Tile4 routed-program metadata bytes overflow".into())
            })?;
        let logical_route_bytes = route_words
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| {
                DeviceError::Shape("Tile4 routed-program route metadata bytes overflow".into())
            })?;
        let metadata_extra_bytes = total_bytes
            .checked_sub(logical_route_bytes)
            .ok_or_else(|| backend_error("Tile4 routed-program metadata accounting underflowed"))?;
        metadata.total_bytes = total_bytes;
        let gate_indirect_offset = u64::try_from(logical_route_bytes)
            .map_err(|_| DeviceError::Shape("Tile4 gate indirect offset exceeds u64".into()))?;
        let down_indirect_offset = gate_indirect_offset
            .checked_add(3 * std::mem::size_of::<u32>() as u64)
            .ok_or_else(|| DeviceError::Shape("Tile4 down indirect offset overflows".into()))?;
        let tile_word_offset = u32::try_from(tile_word_offset)
            .map_err(|_| DeviceError::Shape("Tile4 schedule word offset exceeds u32".into()))?;
        (
            metadata_extra_bytes,
            Some(RoutedMoeTile4Geometry {
                gate_indirect_offset,
                down_indirect_offset,
                tile_word_offset,
                max_tiles,
                schedule_words,
            }),
        )
    } else {
        (0, None)
    };
    Ok(RoutedMoeProgramGeometry {
        hidden,
        expert_width,
        routes,
        batch_len: batch_len.get(),
        input,
        output,
        metadata,
        hidden_scratch,
        route_scratch,
        metadata_extra_bytes,
        matrix_words,
        dispatch,
        tile4,
    })
}

fn validate_item_dimensional_routed_moe_program(
    geometry: RoutedMoeProgramGeometry,
    fast_variant: RoutedMoeFastVariant,
    max_group_count: [u32; 3],
    max_storage_buffer_range: usize,
    device_name: &str,
) -> Result<(), DeviceError> {
    if fast_variant != RoutedMoeFastVariant::Tree128 {
        return Err(backend_error(format!(
            "item-dimensional routed-MoE requires Tree128 arithmetic, but {} selected {}",
            device_name,
            fast_variant.label()
        )));
    }
    let batch_len = u32::try_from(geometry.batch_len).map_err(|_| {
        DeviceError::Shape("item-dimensional routed-program batch length exceeds u32".into())
    })?;
    if batch_len == 0 {
        return Err(DeviceError::Shape(
            "item-dimensional routed-program batch length must be non-zero".into(),
        ));
    }
    if geometry.expert_width > max_group_count[0]
        || geometry.hidden > max_group_count[0]
        || geometry.routes > max_group_count[1]
        || batch_len > max_group_count[2]
    {
        return Err(DeviceError::Shape(format!(
            "item-dimensional routed-program dispatch H={} F={} K={} N={} exceeds Vulkan workgroup-count limits {}x{}x{} on {}",
            geometry.hidden,
            geometry.expert_width,
            geometry.routes,
            batch_len,
            max_group_count[0],
            max_group_count[1],
            max_group_count[2],
            device_name
        )));
    }
    if geometry.dispatch.gate_up != (geometry.expert_width, geometry.routes)
        || geometry.dispatch.down != (geometry.hidden, geometry.routes)
        || geometry.dispatch.reduce != (geometry.hidden.div_ceil(64), 1)
    {
        return Err(backend_error(
            "item-dimensional routed-program dispatch geometry does not match the fast shader ABI",
        ));
    }

    for (name, buffer) in [
        ("input", geometry.input),
        ("output", geometry.output),
        ("metadata", geometry.metadata),
        ("hidden scratch", geometry.hidden_scratch),
        ("route scratch", geometry.route_scratch),
    ] {
        let expected_total = buffer
            .descriptor_bytes
            .checked_mul(geometry.batch_len)
            .ok_or_else(|| {
                DeviceError::Shape(format!(
                    "item-dimensional routed-program {name} whole-batch range overflows"
                ))
            })?;
        if buffer.item_stride != buffer.descriptor_bytes {
            return Err(DeviceError::Shape(format!(
                "item-dimensional routed-program {name} must be tightly packed, but its item range is {} bytes and stride is {} bytes",
                buffer.descriptor_bytes, buffer.item_stride
            )));
        }
        if buffer.total_bytes != expected_total {
            return Err(DeviceError::Shape(format!(
                "item-dimensional routed-program {name} whole-batch range is {}, expected tightly packed {}",
                buffer.total_bytes, expected_total
            )));
        }
        if buffer.total_bytes > max_storage_buffer_range {
            return Err(DeviceError::Shape(format!(
                "item-dimensional routed-program whole-batch {name} needs {} descriptor bytes, exceeding {} on {}",
                buffer.total_bytes, max_storage_buffer_range, device_name
            )));
        }
    }
    for (name, buffer) in [
        ("input", geometry.input),
        ("hidden scratch", geometry.hidden_scratch),
    ] {
        if !buffer
            .descriptor_bytes
            .is_multiple_of(2 * std::mem::size_of::<f32>())
        {
            return Err(DeviceError::Shape(format!(
                "item-dimensional routed-program {name} item range {} is not aligned for vec2 access",
                buffer.descriptor_bytes
            )));
        }
    }

    validate_glsl_u32_product(&[batch_len, geometry.hidden], "N*H input/output")?;
    validate_glsl_u32_product(
        &[batch_len, geometry.routes, geometry.expert_width],
        "N*K*F hidden scratch",
    )?;
    validate_glsl_u32_product(
        &[batch_len, geometry.routes, geometry.hidden],
        "N*K*H route scratch",
    )?;
    validate_glsl_u32_product(&[batch_len, geometry.routes, 4], "N*K*4 route metadata")?;
    Ok(())
}

fn validate_tile4_routed_moe_program(
    geometry: RoutedMoeProgramGeometry,
    fast_variant: RoutedMoeFastVariant,
    max_group_count: [u32; 3],
    max_storage_buffer_range: usize,
    device_name: &str,
) -> Result<(), DeviceError> {
    let tile = geometry.tile4.ok_or_else(|| {
        backend_error("Tile4 routed-program geometry is missing its indirect schedule")
    })?;
    let logical_metadata_bytes = geometry
        .metadata
        .descriptor_bytes
        .checked_mul(geometry.batch_len)
        .ok_or_else(|| DeviceError::Shape("Tile4 route metadata size overflows".into()))?;
    let mut base_geometry = geometry;
    base_geometry.metadata.total_bytes = logical_metadata_bytes;
    base_geometry.metadata_extra_bytes = 0;
    base_geometry.tile4 = None;
    validate_item_dimensional_routed_moe_program(
        base_geometry,
        fast_variant,
        max_group_count,
        max_storage_buffer_range,
        device_name,
    )?;

    let batch_len = u32::try_from(geometry.batch_len)
        .map_err(|_| DeviceError::Shape("Tile4 batch length exceeds u32".into()))?;
    let max_tiles = batch_len.checked_mul(geometry.routes).ok_or_else(|| {
        DeviceError::Shape("Tile4 maximum occupied-tile count overflows u32".into())
    })?;
    if max_tiles > max_group_count[1] {
        return Err(DeviceError::Shape(format!(
            "Tile4 routed-program worst-case tile count {max_tiles} exceeds Vulkan Y workgroup-count limit {} on {device_name}",
            max_group_count[1]
        )));
    }
    if tile.max_tiles != max_tiles {
        return Err(backend_error(format!(
            "Tile4 geometry records {} maximum tiles, expected {max_tiles}",
            tile.max_tiles
        )));
    }
    let expected_schedule_words = usize::try_from(max_tiles)
        .ok()
        .and_then(|tiles| tiles.checked_mul(ROUTED_MOE_TILE4_RECORD_WORDS))
        .ok_or_else(|| DeviceError::Shape("Tile4 schedule word count overflows".into()))?;
    if tile.schedule_words != expected_schedule_words {
        return Err(backend_error(format!(
            "Tile4 geometry records {} schedule words, expected {expected_schedule_words}",
            tile.schedule_words
        )));
    }
    let logical_metadata_words = logical_metadata_bytes
        .checked_div(std::mem::size_of::<u32>())
        .ok_or_else(|| backend_error("Tile4 metadata word conversion failed"))?;
    let expected_tile_word_offset = logical_metadata_words
        .checked_add(ROUTED_MOE_TILE4_INDIRECT_WORDS)
        .ok_or_else(|| DeviceError::Shape("Tile4 schedule offset overflows".into()))?;
    if usize::try_from(tile.tile_word_offset).ok() != Some(expected_tile_word_offset) {
        return Err(backend_error(format!(
            "Tile4 schedule starts at word {}, expected {expected_tile_word_offset}",
            tile.tile_word_offset
        )));
    }
    let expected_total_bytes = expected_tile_word_offset
        .checked_add(expected_schedule_words)
        .and_then(|words| words.checked_mul(std::mem::size_of::<u32>()))
        .ok_or_else(|| DeviceError::Shape("Tile4 complete metadata range overflows".into()))?;
    if geometry.metadata.total_bytes != expected_total_bytes
        || geometry.metadata_extra_bytes
            != expected_total_bytes
                .checked_sub(logical_metadata_bytes)
                .ok_or_else(|| backend_error("Tile4 metadata accounting underflowed"))?
    {
        return Err(backend_error(format!(
            "Tile4 metadata geometry is inconsistent: total={} extra={}, expected total={expected_total_bytes}",
            geometry.metadata.total_bytes, geometry.metadata_extra_bytes
        )));
    }
    if expected_total_bytes > max_storage_buffer_range {
        return Err(DeviceError::Shape(format!(
            "Tile4 routed-program metadata needs {expected_total_bytes} descriptor bytes, exceeding {max_storage_buffer_range} on {device_name}"
        )));
    }
    let expected_gate_offset = u64::try_from(logical_metadata_bytes)
        .map_err(|_| DeviceError::Shape("Tile4 gate indirect offset exceeds u64".into()))?;
    let expected_down_offset = expected_gate_offset
        .checked_add(3 * std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| DeviceError::Shape("Tile4 down indirect offset overflows".into()))?;
    if tile.gate_indirect_offset != expected_gate_offset
        || tile.down_indirect_offset != expected_down_offset
        || !tile.gate_indirect_offset.is_multiple_of(4)
        || !tile.down_indirect_offset.is_multiple_of(4)
    {
        return Err(backend_error(
            "Tile4 indirect-command offsets do not match the metadata ABI",
        ));
    }
    let last_command_end = tile
        .down_indirect_offset
        .checked_add(3 * std::mem::size_of::<u32>() as u64)
        .ok_or_else(|| DeviceError::Shape("Tile4 indirect command range overflows".into()))?;
    if last_command_end
        > u64::try_from(geometry.metadata.total_bytes)
            .map_err(|_| DeviceError::Shape("Tile4 metadata range exceeds u64".into()))?
    {
        return Err(backend_error(
            "Tile4 indirect commands exceed the metadata buffer",
        ));
    }
    Ok(())
}

fn validate_glsl_u32_product(factors: &[u32], purpose: &str) -> Result<(), DeviceError> {
    let product = factors
        .iter()
        .try_fold(1u128, |product, &factor| {
            product.checked_mul(u128::from(factor))
        })
        .ok_or_else(|| DeviceError::Shape(format!("{purpose} index product overflows")))?;
    if product > u128::from(u32::MAX) {
        return Err(DeviceError::Shape(format!(
            "item-dimensional routed-program {purpose} element count {product} exceeds GLSL uint indexing"
        )));
    }
    Ok(())
}

fn routed_moe_program_byte_counts(
    geometry: RoutedMoeProgramGeometry,
) -> Result<RoutedMoeProgramByteCounts, DeviceError> {
    let batch_len = u64::try_from(geometry.batch_len)
        .map_err(|_| DeviceError::Shape("routed-program batch length exceeds u64".into()))?;
    let mut requested_bytes = 0u64;
    let mut buffer_bytes = 0u64;
    for buffer in [
        geometry.input,
        geometry.output,
        geometry.metadata,
        geometry.hidden_scratch,
        geometry.route_scratch,
    ] {
        let descriptor_bytes = u64::try_from(buffer.descriptor_bytes).map_err(|_| {
            DeviceError::Shape("routed-program descriptor byte count exceeds u64".into())
        })?;
        let requested = descriptor_bytes.checked_mul(batch_len).ok_or_else(|| {
            DeviceError::Shape("routed-program requested-byte accounting overflows".into())
        })?;
        requested_bytes = requested_bytes.checked_add(requested).ok_or_else(|| {
            DeviceError::Shape("routed-program requested-byte accounting overflows".into())
        })?;
        buffer_bytes = buffer_bytes
            .checked_add(u64::try_from(buffer.total_bytes).map_err(|_| {
                DeviceError::Shape("routed-program buffer-byte accounting exceeds u64".into())
            })?)
            .ok_or_else(|| {
                DeviceError::Shape("routed-program buffer-byte accounting overflows".into())
            })?;
    }
    requested_bytes = requested_bytes
        .checked_add(u64::try_from(geometry.metadata_extra_bytes).map_err(|_| {
            DeviceError::Shape("routed-program extra metadata bytes exceed u64".into())
        })?)
        .ok_or_else(|| {
            DeviceError::Shape("routed-program requested-byte accounting overflows".into())
        })?;
    Ok(RoutedMoeProgramByteCounts {
        requested_bytes,
        buffer_bytes,
    })
}

fn routed_moe_program_remaining_budget(stats: RoutedMoeProgramStats) -> Result<u64, DeviceError> {
    stats
        .budget_bytes
        .checked_sub(stats.allocation_bytes)
        .ok_or_else(|| {
            backend_error("Vulkan routed-program allocation accounting exceeded its budget")
        })
}

fn charge_routed_moe_program_allocation(
    stats: &mut RoutedMoeProgramStats,
    allocation: &BufferAllocation,
) -> Result<(), DeviceError> {
    let buffer_bytes = stats
        .buffer_bytes
        .checked_add(
            u64::try_from(allocation.capacity)
                .map_err(|_| backend_error("Vulkan routed-program buffer capacity exceeds u64"))?,
        )
        .ok_or_else(|| backend_error("Vulkan routed-program buffer accounting overflowed"))?;
    let allocation_bytes = stats
        .allocation_bytes
        .checked_add(allocation.allocation_bytes)
        .ok_or_else(|| backend_error("Vulkan routed-program allocation accounting overflowed"))?;
    if allocation_bytes > stats.budget_bytes {
        return Err(backend_error(format!(
            "Vulkan routed program charged {allocation_bytes} allocation bytes beyond its {}-byte authorized budget",
            stats.budget_bytes
        )));
    }
    let allocation_slots = stats.allocation_slots.checked_add(1).ok_or_else(|| {
        backend_error("Vulkan routed-program allocation-slot accounting overflowed")
    })?;
    stats.buffer_bytes = buffer_bytes;
    stats.allocation_bytes = allocation_bytes;
    stats.allocation_slots = allocation_slots;
    Ok(())
}

fn routed_moe_buffer_geometry(
    descriptor_bytes: usize,
    batch_len: NonZeroUsize,
    item_alignment: usize,
    max_storage_buffer_range: usize,
    device_name: &str,
    purpose: &str,
) -> Result<RoutedMoeBufferGeometry, DeviceError> {
    if descriptor_bytes == 0 {
        return Err(DeviceError::Shape(format!(
            "routed-program {purpose} descriptor range must be non-zero"
        )));
    }
    if descriptor_bytes > max_storage_buffer_range {
        return Err(DeviceError::Shape(format!(
            "routed-program {purpose} needs {descriptor_bytes} descriptor bytes, exceeding {max_storage_buffer_range} on {device_name}"
        )));
    }
    let item_stride = align_up(
        descriptor_bytes,
        item_alignment,
        &format!("routed-program {purpose} item stride"),
    )?;
    let total_bytes = item_stride.checked_mul(batch_len.get()).ok_or_else(|| {
        DeviceError::Shape(format!(
            "routed-program {purpose} batch allocation size overflows"
        ))
    })?;
    u64::try_from(total_bytes).map_err(|_| {
        DeviceError::Shape(format!(
            "routed-program {purpose} batch allocation exceeds Vulkan device-size range"
        ))
    })?;
    Ok(RoutedMoeBufferGeometry {
        descriptor_bytes,
        item_stride,
        total_bytes,
    })
}

fn routed_moe_program_descriptor_counts(
    batch_len: NonZeroUsize,
    batch_dispatch: RoutedMoeBatchDispatch,
) -> Result<(usize, u32), DeviceError> {
    let descriptor_sets = match batch_dispatch {
        RoutedMoeBatchDispatch::PerItem => batch_len.get().checked_mul(3).ok_or_else(|| {
            DeviceError::Shape("routed-program descriptor-set count overflows".into())
        })?,
        RoutedMoeBatchDispatch::ItemDimension | RoutedMoeBatchDispatch::ExpertTile4 => 3,
    };
    u32::try_from(descriptor_sets).map_err(|_| {
        DeviceError::Shape("routed-program descriptor-set count exceeds u32".into())
    })?;
    let descriptors = descriptor_sets.checked_mul(4).ok_or_else(|| {
        DeviceError::Shape("routed-program storage-descriptor count overflows".into())
    })?;
    let descriptors = u32::try_from(descriptors).map_err(|_| {
        DeviceError::Shape("routed-program storage-descriptor count exceeds u32".into())
    })?;
    Ok((descriptor_sets, descriptors))
}

fn validate_absolute_fast_routed_moe_offsets(
    weight_offset: usize,
    offsets_bytes: &[[u32; 3]],
    matrix_words: u32,
    weight_capacity: usize,
) -> Result<(), DeviceError> {
    if !weight_capacity.is_multiple_of(DESCRIPTOR_WORD_BYTES) {
        return Err(DeviceError::Shape(format!(
            "routed-program arena block capacity {weight_capacity} is not four-byte aligned"
        )));
    }
    let weight_words = u32::try_from(weight_capacity / DESCRIPTOR_WORD_BYTES).map_err(|_| {
        DeviceError::Shape("routed-program arena block word count exceeds u32".into())
    })?;
    for (route, offsets) in offsets_bytes.iter().enumerate() {
        for (projection, relative) in ["gate", "up", "down"]
            .into_iter()
            .zip(offsets.iter().copied())
        {
            let byte_offset = weight_offset
                .checked_add(relative as usize)
                .ok_or_else(|| {
                    DeviceError::Shape(format!(
                        "routed-program route {route} {projection} block offset overflows"
                    ))
                })?;
            if !byte_offset.is_multiple_of(DESCRIPTOR_WORD_BYTES) {
                return Err(DeviceError::Shape(format!(
                        "routed-program route {route} {projection} block offset {byte_offset} is not four-byte aligned"
                    )));
            }
            let word_offset = u32::try_from(byte_offset / DESCRIPTOR_WORD_BYTES).map_err(|_| {
                DeviceError::Shape(format!(
                    "routed-program route {route} {projection} word offset exceeds u32"
                ))
            })?;
            let end = word_offset.checked_add(matrix_words).ok_or_else(|| {
                DeviceError::Shape(format!(
                    "routed-program route {route} {projection} word range overflows"
                ))
            })?;
            if end > weight_words {
                return Err(DeviceError::Shape(format!(
                        "routed-program route {route} {projection} ends at word {end}, beyond arena block word {weight_words}"
                    )));
            }
        }
    }
    Ok(())
}

fn allocate_persistently_mapped_buffer(
    context: &Context,
    required: usize,
    purpose: &'static str,
    allocation_limit: u32,
    budget_remaining: u64,
    usage: vk::BufferUsageFlags,
) -> Result<PersistentlyMappedBuffer, DeviceError> {
    let allocation = allocate_buffer_limited_with_usage(
        context,
        required,
        purpose,
        allocation_limit,
        budget_remaining,
        usage,
    )?;
    let mapped = unsafe {
        context.device.map_memory(
            allocation.memory,
            0,
            allocation.capacity as u64,
            vk::MemoryMapFlags::empty(),
        )
    };
    match mapped {
        Ok(pointer) => match NonNull::new(pointer.cast::<u8>()) {
            Some(mapped) => Ok(PersistentlyMappedBuffer { allocation, mapped }),
            None => {
                // A successful Vulkan mapping must return a valid host pointer. Keep
                // cleanup well-formed even if a broken implementation violates that
                // contract after establishing the mapping.
                unsafe {
                    context.device.unmap_memory(allocation.memory);
                    allocation.destroy(&context.device, &context.live_allocations);
                }
                Err(backend_error(format!("mapping {purpose} returned null")))
            }
        },
        Err(error) => {
            unsafe { allocation.destroy(&context.device, &context.live_allocations) };
            Err(vk_error(format!("mapping {purpose}"), error))
        }
    }
}

fn record_routed_moe_program_command_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    descriptor_sets: &[vk::DescriptorSet],
    pipelines: (vk::Pipeline, vk::Pipeline, vk::Pipeline),
    recording: RoutedMoeProgramRecording,
    accumulate: bool,
) -> Result<(), DeviceError> {
    let RoutedMoeProgramRecording {
        geometry,
        batch_dispatch,
        indirect_buffer,
    } = recording;
    let expected_sets = match batch_dispatch {
        RoutedMoeBatchDispatch::PerItem => geometry.batch_len.checked_mul(3).ok_or_else(|| {
            DeviceError::Shape("routed-program descriptor count overflows".into())
        })?,
        RoutedMoeBatchDispatch::ItemDimension | RoutedMoeBatchDispatch::ExpertTile4 => 3,
    };
    if descriptor_sets.len() != expected_sets {
        return Err(DeviceError::Shape(format!(
            "routed-program batch of {} needs {expected_sets} descriptor sets, got {}",
            geometry.batch_len,
            descriptor_sets.len()
        )));
    }
    let batch_len = u32::try_from(geometry.batch_len)
        .map_err(|_| DeviceError::Shape("routed-program batch length exceeds u32".into()))?;
    let parameters = [
        geometry.hidden,
        geometry.expert_width,
        geometry.routes,
        u32::from(accumulate),
        batch_len,
        geometry.tile4.map_or(0, |tile| tile.tile_word_offset),
    ];
    let parameter_bytes = unsafe {
        std::slice::from_raw_parts(
            parameters.as_ptr().cast::<u8>(),
            std::mem::size_of_val(&parameters),
        )
    };
    unsafe {
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::empty()),
            )
            .map_err(|error| vk_error("beginning a routed-program command buffer", error))?;
        let tile4 = batch_dispatch == RoutedMoeBatchDispatch::ExpertTile4;
        let host_write = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::HOST_WRITE)
            .dst_access_mask(if tile4 {
                vk::AccessFlags::SHADER_READ | vk::AccessFlags::INDIRECT_COMMAND_READ
            } else {
                vk::AccessFlags::SHADER_READ
            })];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::HOST,
            if tile4 {
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT
            } else {
                vk::PipelineStageFlags::COMPUTE_SHADER
            },
            vk::DependencyFlags::empty(),
            &host_write,
            &[],
            &[],
        );
        let bind = |pipeline, descriptor_set| {
            device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
            device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_layout,
                0,
                &[descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                command_buffer,
                pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                parameter_bytes,
            );
        };
        let record_dispatch = |pipeline, descriptor_set, groups: (u32, u32), groups_z| {
            bind(pipeline, descriptor_set);
            device.cmd_dispatch(command_buffer, groups.0, groups.1, groups_z);
        };
        match batch_dispatch {
            RoutedMoeBatchDispatch::PerItem => {
                for item in 0..geometry.batch_len {
                    record_dispatch(
                        pipelines.0,
                        descriptor_sets[item * 3],
                        geometry.dispatch.gate_up,
                        1,
                    );
                }
                compute_read_after_write(device, command_buffer);
                for item in 0..geometry.batch_len {
                    record_dispatch(
                        pipelines.1,
                        descriptor_sets[item * 3 + 1],
                        geometry.dispatch.down,
                        1,
                    );
                }
                compute_read_after_write(device, command_buffer);
                for item in 0..geometry.batch_len {
                    record_dispatch(
                        pipelines.2,
                        descriptor_sets[item * 3 + 2],
                        geometry.dispatch.reduce,
                        1,
                    );
                }
            }
            RoutedMoeBatchDispatch::ItemDimension => {
                record_dispatch(
                    pipelines.0,
                    descriptor_sets[0],
                    geometry.dispatch.gate_up,
                    batch_len,
                );
                compute_read_after_write(device, command_buffer);
                record_dispatch(
                    pipelines.1,
                    descriptor_sets[1],
                    geometry.dispatch.down,
                    batch_len,
                );
                compute_read_after_write(device, command_buffer);
                record_dispatch(
                    pipelines.2,
                    descriptor_sets[2],
                    geometry.dispatch.reduce,
                    batch_len,
                );
            }
            RoutedMoeBatchDispatch::ExpertTile4 => {
                let tile = geometry.tile4.ok_or_else(|| {
                    backend_error("Tile4 command recording is missing schedule geometry")
                })?;
                if indirect_buffer == vk::Buffer::null() {
                    return Err(backend_error(
                        "Tile4 command recording is missing its indirect buffer",
                    ));
                }
                bind(pipelines.0, descriptor_sets[0]);
                device.cmd_dispatch_indirect(
                    command_buffer,
                    indirect_buffer,
                    tile.gate_indirect_offset,
                );
                compute_read_after_write(device, command_buffer);
                bind(pipelines.1, descriptor_sets[1]);
                device.cmd_dispatch_indirect(
                    command_buffer,
                    indirect_buffer,
                    tile.down_indirect_offset,
                );
                compute_read_after_write(device, command_buffer);
                record_dispatch(
                    pipelines.2,
                    descriptor_sets[2],
                    geometry.dispatch.reduce,
                    batch_len,
                );
            }
        }
        let host_read = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::SHADER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &host_read,
            &[],
            &[],
        );
        device
            .end_command_buffer(command_buffer)
            .map_err(|error| vk_error("ending a routed-program command buffer", error))?;
    }
    Ok(())
}

fn destroy_routed_moe_program_resources(
    context: &Context,
    resources: &mut RoutedMoeProgramResources,
) {
    unsafe {
        if resources.fence != vk::Fence::null() {
            context.device.destroy_fence(resources.fence, None);
            resources.fence = vk::Fence::null();
        }
        if resources.command_pool != vk::CommandPool::null() {
            context
                .device
                .destroy_command_pool(resources.command_pool, None);
            resources.command_pool = vk::CommandPool::null();
            resources.command_buffers = [vk::CommandBuffer::null(); 2];
        }
        if resources.descriptor_pool != vk::DescriptorPool::null() {
            context
                .device
                .destroy_descriptor_pool(resources.descriptor_pool, None);
            resources.descriptor_pool = vk::DescriptorPool::null();
            resources.descriptor_sets = Box::default();
        }
        for mapped in [
            &mut resources.input,
            &mut resources.output,
            &mut resources.metadata,
        ] {
            if let Some(mapped) = mapped.take() {
                context.device.unmap_memory(mapped.allocation.memory);
                mapped
                    .allocation
                    .destroy(&context.device, &context.live_allocations);
            }
        }
        if let Some(hidden) = resources.hidden.take() {
            hidden.destroy(&context.device, &context.live_allocations);
        }
        if let Some(routes) = resources.routes.take() {
            routes.destroy(&context.device, &context.live_allocations);
        }
    }
}

fn descriptor_write<'a>(
    set: vk::DescriptorSet,
    binding: u32,
    info: &'a [vk::DescriptorBufferInfo],
) -> vk::WriteDescriptorSet<'a> {
    vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(info)
}

unsafe fn compute_read_after_write(device: &Device, command_buffer: vk::CommandBuffer) {
    let dependency = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)];
    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::PipelineStageFlags::COMPUTE_SHADER,
            vk::DependencyFlags::empty(),
            &dependency,
            &[],
            &[],
        );
    }
}

fn usable_timestamp_properties(valid_bits: u32, period_ns: f32) -> Option<TimestampProperties> {
    ((36..=64).contains(&valid_bits) && period_ns.is_finite() && period_ns > 0.0).then_some(
        TimestampProperties {
            valid_bits,
            period_ns,
        },
    )
}

fn ensure_timestamp_query_pool(
    device: &Device,
    state: &mut QueueState,
) -> Result<vk::QueryPool, DeviceError> {
    if let Some(query_pool) = state.timestamp_query_pool {
        return Ok(query_pool);
    }
    let query_pool = unsafe {
        device.create_query_pool(
            &vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::TIMESTAMP)
                .query_count(ROUTED_MOE_TIMESTAMP_COUNT),
            None,
        )
    }
    .map_err(|error| vk_error("creating the routed-MoE timestamp query pool", error))?;
    state.timestamp_query_pool = Some(query_pool);
    Ok(query_pool)
}

fn timestamp_delta(start: u64, end: u64, valid_bits: u32) -> Result<u64, DeviceError> {
    let mask = match valid_bits {
        36..=63 => (1u64 << valid_bits) - 1,
        64 => u64::MAX,
        _ => {
            return Err(backend_error(format!(
                "timestamp counter exposes invalid width {valid_bits}"
            )));
        }
    };
    Ok(end.wrapping_sub(start) & mask)
}

fn timestamp_duration(
    start: u64,
    end: u64,
    properties: TimestampProperties,
) -> Result<Duration, DeviceError> {
    let seconds = timestamp_delta(start, end, properties.valid_bits)? as f64
        * f64::from(properties.period_ns)
        * 1e-9;
    Duration::try_from_secs_f64(seconds).map_err(|_| {
        backend_error(format!(
            "GPU timestamp interval {seconds:.6e}s is not representable"
        ))
    })
}

fn routed_moe_gpu_profile(
    timestamps: [u64; ROUTED_MOE_TIMESTAMP_COUNT as usize],
    properties: TimestampProperties,
    host_query_readback: Duration,
) -> Result<RoutedMoeGpuProfile, DeviceError> {
    let interval = |start, end| timestamp_duration(start, end, properties);
    Ok(RoutedMoeGpuProfile {
        gate_up: interval(timestamps[0], timestamps[1])?,
        gate_to_down_dependency: interval(timestamps[1], timestamps[2])?,
        down: interval(timestamps[2], timestamps[3])?,
        down_to_reduce_dependency: interval(timestamps[3], timestamps[4])?,
        reduce: interval(timestamps[4], timestamps[5])?,
        total: interval(timestamps[0], timestamps[5])?,
        host_query_readback,
        counter_valid_bits: properties.valid_bits,
        timestamp_period_ns: properties.period_ns,
    })
}

struct MatrixGeometry {
    row_bytes: usize,
    encoded_len: usize,
    format: PreparedFormat,
}

fn validate_weight(weight: &Weight<'_>) -> Result<MatrixGeometry, DeviceError> {
    let geometry = validate_matrix_spec(MatrixSpec::from_weight(weight))?;
    if weight.bytes.len() != geometry.encoded_len {
        return Err(DeviceError::Shape(format!(
            "{}x{} {} weight needs {} bytes, got {}",
            weight.rows,
            weight.cols,
            geometry.format.name(),
            geometry.encoded_len,
            weight.bytes.len()
        )));
    }
    Ok(geometry)
}

fn validate_matrix_spec(spec: MatrixSpec) -> Result<MatrixGeometry, DeviceError> {
    if spec.rows == 0 || spec.cols == 0 {
        return Err(DeviceError::Shape(
            "Vulkan weights must have non-zero rows and columns".into(),
        ));
    }
    let (format, row_bytes) = match spec.kind.0 {
        Q8_0_KIND => {
            if !spec.cols.is_multiple_of(Q8_0_ELEMENTS) {
                return Err(DeviceError::Shape(format!(
                    "{}x{} Q8_0 weight has {} columns, not a whole number of {Q8_0_ELEMENTS}-element blocks",
                    spec.rows, spec.cols, spec.cols
                )));
            }
            let bytes = (spec.cols / Q8_0_ELEMENTS)
                .checked_mul(Q8_0_BYTES)
                .ok_or_else(|| DeviceError::Shape("Q8_0 row byte count overflows".into()))?;
            (PreparedFormat::Q8_0, bytes)
        }
        BF16_KIND => {
            let bytes = spec
                .cols
                .checked_mul(2)
                .ok_or_else(|| DeviceError::Shape("BF16 row byte count overflows".into()))?;
            (PreparedFormat::Bf16, bytes)
        }
        _ => {
            return Err(DeviceError::UnsupportedFormat {
                device: BACKEND,
                kind: spec.kind.name(),
                rows: spec.rows,
                cols: spec.cols,
            });
        }
    };
    let encoded_len = spec
        .rows
        .checked_mul(row_bytes)
        .ok_or_else(|| DeviceError::Shape("Vulkan matrix byte count overflows".into()))?;
    if encoded_len > u32::MAX as usize {
        return Err(DeviceError::Shape(format!(
            "{} matrix has {encoded_len} encoded bytes; the current Vulkan shader uses 32-bit byte offsets",
            format.name()
        )));
    }
    Ok(MatrixGeometry {
        row_bytes,
        encoded_len,
        format,
    })
}

fn align_up(value: usize, alignment: usize, purpose: &str) -> Result<usize, DeviceError> {
    if alignment == 0 {
        return Err(DeviceError::Shape(format!(
            "{purpose} alignment must be non-zero"
        )));
    }
    value
        .checked_add(alignment - 1)
        .map(|rounded| rounded / alignment * alignment)
        .ok_or_else(|| DeviceError::Shape(format!("{purpose} alignment overflows")))
}

fn select_block_tail(
    blocks: &[ArenaBlockBuilder],
    descriptor_len: usize,
    alignment: usize,
) -> Result<Option<(usize, usize)>, DeviceError> {
    let tails: Vec<_> = blocks
        .iter()
        .map(|block| {
            (
                block.used,
                block
                    .allocation
                    .as_ref()
                    .expect("building block owns its allocation")
                    .capacity,
            )
        })
        .collect();
    select_tail(&tails, descriptor_len, alignment)
}

fn select_tail(
    tails: &[(usize, usize)],
    descriptor_len: usize,
    alignment: usize,
) -> Result<Option<(usize, usize)>, DeviceError> {
    let mut best: Option<(usize, usize, usize)> = None;
    for (index, &(used, capacity)) in tails.iter().enumerate() {
        let offset = align_up(used, alignment, "Vulkan arena slice offset")?;
        let Some(end) = offset.checked_add(descriptor_len) else {
            continue;
        };
        if end > capacity {
            continue;
        }
        let remaining = capacity - end;
        if best.is_none_or(|(_, _, best_remaining)| remaining < best_remaining) {
            best = Some((index, offset, remaining));
        }
    }
    Ok(best.map(|(index, offset, _)| (index, offset)))
}

fn preferred_fast_variant_for_device(
    device_name: &str,
    max_work_group_invocations: u32,
    max_work_group_size_x: u32,
    driver_version: u32,
    hidden: usize,
    expert_width: usize,
    routes: usize,
) -> RoutedMoeFastVariant {
    if device_name.contains("RADV GFX1151")
        && max_work_group_invocations >= 128
        && max_work_group_size_x >= 128
        && driver_version == vk::make_api_version(0, 25, 2, 8)
        && (hidden, expert_width, routes) == (2048, 512, 8)
    {
        RoutedMoeFastVariant::Tree128
    } else {
        RoutedMoeFastVariant::QualifiedTree64
    }
}

fn validate_fast_routed_moe_weights(experts: &PreparedRoutedExperts) -> Result<(), DeviceError> {
    if !experts.weight_offset.is_multiple_of(DESCRIPTOR_WORD_BYTES)
        || !experts.weight_range.is_multiple_of(DESCRIPTOR_WORD_BYTES)
    {
        return Err(DeviceError::Shape(format!(
            "fast routed-MoE weight span offset={} range={} is not four-byte aligned",
            experts.weight_offset, experts.weight_range
        )));
    }
    let matrix_elements = experts
        .hidden
        .checked_mul(experts.expert_width)
        .ok_or_else(|| DeviceError::Shape("fast routed-MoE matrix size overflows".into()))?;
    if !matrix_elements.is_multiple_of(2) {
        return Err(DeviceError::Shape(
            "fast routed-MoE matrix does not contain whole BF16 word pairs".into(),
        ));
    }
    let matrix_words = u32::try_from(matrix_elements / 2)
        .map_err(|_| DeviceError::Shape("fast routed-MoE matrix word count exceeds u32".into()))?;
    let weight_words =
        u32::try_from(experts.weight_range / DESCRIPTOR_WORD_BYTES).map_err(|_| {
            DeviceError::Shape("fast routed-MoE descriptor word count exceeds u32".into())
        })?;
    for (route, offsets) in experts.offsets_bytes.iter().enumerate() {
        for (projection, byte_offset) in ["gate", "up", "down"].into_iter().zip(offsets) {
            let word_offset = routed_moe_metadata_offset(RoutedMoeKernel::Fast, *byte_offset)?;
            let end = word_offset.checked_add(matrix_words).ok_or_else(|| {
                DeviceError::Shape(format!(
                    "fast routed-MoE route {route} {projection} word range overflows"
                ))
            })?;
            if end > weight_words {
                return Err(DeviceError::Shape(format!(
                    "fast routed-MoE route {route} {projection} ends at word {end}, beyond descriptor word {weight_words}"
                )));
            }
        }
    }
    Ok(())
}

fn routed_moe_metadata_offset(
    kernel: RoutedMoeKernel,
    byte_offset: u32,
) -> Result<u32, DeviceError> {
    match kernel {
        RoutedMoeKernel::Guarded => Ok(byte_offset),
        RoutedMoeKernel::Fast if byte_offset.is_multiple_of(DESCRIPTOR_WORD_BYTES as u32) => {
            Ok(byte_offset / DESCRIPTOR_WORD_BYTES as u32)
        }
        RoutedMoeKernel::Fast => Err(DeviceError::Shape(format!(
            "fast routed-MoE weight offset {byte_offset} is not four-byte aligned"
        ))),
    }
}

fn routed_moe_dispatch(
    kernel: RoutedMoeKernel,
    hidden: u32,
    expert_width: u32,
    routes: u32,
    limits: [u32; 3],
    device_name: &str,
) -> Result<RoutedMoeDispatch, DeviceError> {
    if hidden == 0 || expert_width == 0 || routes == 0 {
        return Err(DeviceError::Shape(
            "routed-MoE dimensions must be non-zero".into(),
        ));
    }
    if limits[0] == 0 || limits[1] == 0 || limits[2] == 0 {
        return Err(DeviceError::Shape(format!(
            "{device_name} exposes a zero-sized Vulkan dispatch dimension"
        )));
    }
    let gate_up_rows = routes
        .checked_mul(expert_width)
        .ok_or_else(|| DeviceError::Shape("routed gate/up dispatch exceeds u32 rows".into()))?;
    let down_rows = routes
        .checked_mul(hidden)
        .ok_or_else(|| DeviceError::Shape("routed down dispatch exceeds u32 rows".into()))?;
    routes
        .checked_mul(4)
        .ok_or_else(|| DeviceError::Shape("routed metadata word count exceeds u32".into()))?;
    let reduce_groups = hidden.div_ceil(64);
    if reduce_groups > limits[0] {
        return Err(DeviceError::Shape(format!(
            "routed reduction needs {reduce_groups} workgroups, exceeding {} on {device_name}",
            limits[0]
        )));
    }

    match kernel {
        RoutedMoeKernel::Guarded => Ok(RoutedMoeDispatch {
            gate_up: dispatch_grid(gate_up_rows, limits)?,
            down: dispatch_grid(down_rows, limits)?,
            reduce: (reduce_groups, 1),
        }),
        RoutedMoeKernel::Fast => {
            if !hidden.is_multiple_of(2) || !expert_width.is_multiple_of(2) {
                return Err(DeviceError::Shape(format!(
                    "fast routed-MoE requires even hidden and expert widths, got H={hidden} F={expert_width}"
                )));
            }
            if expert_width > limits[0] || hidden > limits[0] || routes > limits[1] {
                return Err(DeviceError::Shape(format!(
                    "fast routed-MoE dispatch H={hidden} F={expert_width} K={routes} exceeds Vulkan workgroup-count limits {}x{} on {device_name}",
                    limits[0], limits[1]
                )));
            }
            Ok(RoutedMoeDispatch {
                gate_up: (expert_width, routes),
                down: (hidden, routes),
                reduce: (reduce_groups, 1),
            })
        }
    }
}

fn dispatch_grid(rows: u32, limits: [u32; 3]) -> Result<(u32, u32), DeviceError> {
    if rows == 0 || limits[0] == 0 || limits[1] == 0 {
        return Err(DeviceError::Shape(
            "Vulkan dispatch dimensions and limits must be non-zero".into(),
        ));
    }
    let x = rows.min(limits[0]);
    let y = rows.div_ceil(x);
    if y > limits[1] {
        return Err(DeviceError::Shape(format!(
            "{rows} matrix rows exceed Vulkan dispatch grid {}x{}",
            limits[0], limits[1]
        )));
    }
    if x.checked_mul(y).is_none() {
        return Err(DeviceError::Shape(format!(
            "{rows} matrix rows require a padded Vulkan grid whose flattened index exceeds u32"
        )));
    }
    Ok((x, y))
}

fn ensure_buffer(
    context: &Context,
    slot: &mut Option<BufferAllocation>,
    required: usize,
    purpose: &'static str,
) -> Result<(), DeviceError> {
    if slot
        .as_ref()
        .is_some_and(|allocation| allocation.capacity >= required)
    {
        return Ok(());
    }
    let replacement = allocate_buffer_limited(
        context,
        required,
        purpose,
        context.max_memory_allocations,
        u64::MAX,
    )?;
    if let Some(previous) = slot.replace(replacement) {
        unsafe { previous.destroy(&context.device, &context.live_allocations) };
    }
    Ok(())
}

fn allocate_buffer_limited(
    context: &Context,
    required: usize,
    purpose: &'static str,
    allocation_limit: u32,
    budget_remaining: u64,
) -> Result<BufferAllocation, DeviceError> {
    allocate_buffer_limited_with_usage(
        context,
        required,
        purpose,
        allocation_limit,
        budget_remaining,
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )
}

fn allocate_buffer_limited_with_usage(
    context: &Context,
    required: usize,
    purpose: &'static str,
    allocation_limit: u32,
    budget_remaining: u64,
    usage: vk::BufferUsageFlags,
) -> Result<BufferAllocation, DeviceError> {
    let capacity = required
        .max(4)
        .checked_add(3)
        .map(|bytes| bytes & !3)
        .ok_or_else(|| DeviceError::Shape(format!("{purpose} allocation size overflows")))?;
    let buffer = unsafe {
        context.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(capacity as u64)
                .usage(usage)
                .sharing_mode(vk::SharingMode::EXCLUSIVE),
            None,
        )
    }
    .map_err(|error| vk_error(format!("creating {purpose} buffer"), error))?;
    let requirements_info = vk::BufferMemoryRequirementsInfo2::default().buffer(buffer);
    let mut dedicated_requirements = vk::MemoryDedicatedRequirements::default();
    let mut requirements =
        vk::MemoryRequirements2::default().push_next(&mut dedicated_requirements);
    unsafe {
        context
            .device
            .get_buffer_memory_requirements2(&requirements_info, &mut requirements);
    }
    let requirements = requirements.memory_requirements;
    if requirements.size > budget_remaining {
        unsafe { context.device.destroy_buffer(buffer, None) };
        return Err(backend_error(format!(
            "{purpose} requires {} Vulkan bytes but only {budget_remaining} bytes remain in its authorized buffer budget",
            requirements.size
        )));
    }
    let memory_type = choose_memory_type(requirements.memory_type_bits, &context.memory_properties)
        .ok_or_else(|| {
            unsafe { context.device.destroy_buffer(buffer, None) };
            backend_error(format!(
                "no HOST_VISIBLE | HOST_COHERENT memory type can back {purpose}"
            ))
        })?;
    if context
        .live_allocations
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |live| {
            (live < allocation_limit).then_some(live + 1)
        })
        .is_err()
    {
        unsafe { context.device.destroy_buffer(buffer, None) };
        return Err(backend_error(format!(
            "{purpose} would consume Vulkan allocation slot {allocation_limit}; reserved headroom must remain available"
        )));
    }
    let base_allocation = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(memory_type);
    let mut dedicated_allocation = vk::MemoryDedicatedAllocateInfo::default().buffer(buffer);
    let memory = match unsafe {
        if dedicated_requirements.requires_dedicated_allocation == vk::TRUE {
            context
                .device
                .allocate_memory(&base_allocation.push_next(&mut dedicated_allocation), None)
        } else {
            context.device.allocate_memory(&base_allocation, None)
        }
    } {
        Ok(memory) => memory,
        Err(_) => {
            context.live_allocations.fetch_sub(1, Ordering::AcqRel);
            unsafe { context.device.destroy_buffer(buffer, None) };
            return Err(DeviceError::Allocation {
                device: BACKEND,
                bytes: capacity,
                purpose,
            });
        }
    };
    if let Err(error) = unsafe { context.device.bind_buffer_memory(buffer, memory, 0) } {
        unsafe {
            context.device.free_memory(memory, None);
            context.device.destroy_buffer(buffer, None);
        }
        context.live_allocations.fetch_sub(1, Ordering::AcqRel);
        return Err(vk_error(format!("binding {purpose} memory"), error));
    }
    Ok(BufferAllocation {
        buffer,
        memory,
        capacity,
        allocation_bytes: requirements.size,
    })
}

fn choose_memory_type(
    compatible: u32,
    properties: &vk::PhysicalDeviceMemoryProperties,
) -> Option<u32> {
    let required = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
    (0..properties.memory_type_count)
        .filter(|index| compatible & (1u32 << index) != 0)
        .filter_map(|index| {
            let flags = properties.memory_types[index as usize].property_flags;
            flags.contains(required).then_some((
                u32::from(flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)) * 2
                    + u32::from(flags.contains(vk::MemoryPropertyFlags::HOST_CACHED)),
                index,
            ))
        })
        .max_by_key(|(rank, _)| *rank)
        .map(|(_, index)| index)
}

fn write_f32(
    device: &Device,
    allocation: &BufferAllocation,
    values: &[f32],
) -> Result<(), DeviceError> {
    let byte_len = values
        .len()
        .checked_mul(4)
        .ok_or_else(|| DeviceError::Shape("f32 input byte count overflows".into()))?;
    if byte_len > allocation.capacity {
        return Err(DeviceError::Shape(format!(
            "buffer capacity {} is smaller than {byte_len} f32 input bytes",
            allocation.capacity
        )));
    }
    unsafe {
        let mapped = device
            .map_memory(
                allocation.memory,
                0,
                byte_len as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|error| vk_error("mapping Vulkan f32 input memory", error))?;
        std::ptr::copy_nonoverlapping(values.as_ptr(), mapped.cast::<f32>(), values.len());
        device.unmap_memory(allocation.memory);
    }
    Ok(())
}

fn write_u32(
    device: &Device,
    allocation: &BufferAllocation,
    values: &[u32],
) -> Result<(), DeviceError> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| DeviceError::Shape("u32 input byte count overflows".into()))?;
    if byte_len > allocation.capacity {
        return Err(DeviceError::Shape(format!(
            "buffer capacity {} is smaller than {byte_len} u32 input bytes",
            allocation.capacity
        )));
    }
    unsafe {
        let mapped = device
            .map_memory(
                allocation.memory,
                0,
                byte_len as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|error| vk_error("mapping Vulkan u32 input memory", error))?;
        std::ptr::copy_nonoverlapping(values.as_ptr(), mapped.cast::<u32>(), values.len());
        device.unmap_memory(allocation.memory);
    }
    Ok(())
}

fn read_f32(
    device: &Device,
    allocation: &BufferAllocation,
    output: &mut [f32],
) -> Result<(), DeviceError> {
    let byte_len = output
        .len()
        .checked_mul(4)
        .ok_or_else(|| DeviceError::Shape("f32 output byte count overflows".into()))?;
    if byte_len > allocation.capacity {
        return Err(DeviceError::Shape(format!(
            "buffer capacity {} is smaller than {byte_len} f32 output bytes",
            allocation.capacity
        )));
    }
    unsafe {
        let mapped = device
            .map_memory(
                allocation.memory,
                0,
                byte_len as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(|error| vk_error("mapping Vulkan f32 output memory", error))?;
        std::ptr::copy_nonoverlapping(mapped.cast::<f32>(), output.as_mut_ptr(), output.len());
        device.unmap_memory(allocation.memory);
    }
    Ok(())
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn backend_error(message: impl Into<String>) -> DeviceError {
    DeviceError::Backend {
        device: BACKEND,
        message: message.into(),
    }
}

fn vk_error(action: impl std::fmt::Display, error: vk::Result) -> DeviceError {
    backend_error(format!("{action}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cpu::Cpu, Device};

    #[test]
    fn physical_device_api_must_support_the_shader_target() {
        assert!(!supports_required_vulkan_api(0));
        assert!(!supports_required_vulkan_api(vk::API_VERSION_1_0));
        assert!(!supports_required_vulkan_api(vk::make_api_version(
            0, 1, 1, 4095,
        )));
        assert!(supports_required_vulkan_api(vk::API_VERSION_1_2));
        assert!(supports_required_vulkan_api(vk::make_api_version(
            0, 1, 2, 1,
        )));
        assert!(supports_required_vulkan_api(vk::API_VERSION_1_3));
        assert!(!supports_required_vulkan_api(vk::make_api_version(
            1, 1, 2, 0,
        )));
    }

    #[test]
    fn teardown_only_destroys_resources_after_completion_or_device_loss() {
        for (result, may_destroy) in [
            (Ok(()), true),
            (Err(vk::Result::ERROR_DEVICE_LOST), true),
            (Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY), false),
            (Err(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY), false),
            (Err(vk::Result::ERROR_UNKNOWN), false),
            (Err(vk::Result::TIMEOUT), false),
        ] {
            let cleaned = Cell::new(0);
            let quarantined = Cell::new(0);
            let destroy = || cleaned.set(cleaned.get() + 1);
            let quarantine = |error| {
                assert_eq!(Err(error), result);
                quarantined.set(quarantined.get() + 1);
            };
            if teardown_is_safe(result, quarantine) {
                destroy();
            }
            assert_eq!(cleaned.get(), usize::from(may_destroy));
            assert_eq!(quarantined.get(), usize::from(!may_destroy));
        }
    }

    #[test]
    fn uncertain_teardown_retains_owner_and_skips_cleanup() {
        use std::cell::RefCell;

        let owner = Arc::new(());
        let observer = Arc::downgrade(&owner);
        // The fake sink can be released at test end; production deliberately forgets
        // this retained Arc so driver and loader ownership survive until exit.
        let quarantine_sink = RefCell::new(None);
        let cleanup = || panic!("uncertain completion must not destroy raw device handles");
        if teardown_is_safe(Err(vk::Result::ERROR_OUT_OF_HOST_MEMORY), |_| {
            quarantine_sink.replace(Some(Arc::clone(&owner)));
        }) {
            cleanup();
        }
        drop(owner);
        assert!(observer.upgrade().is_some());
        drop(quarantine_sink);
        assert!(observer.upgrade().is_none());
    }

    #[test]
    fn successful_submissions_leave_shared_queue_reusable() {
        let health = QueueHealth::default();
        let submits = Cell::new(0);
        let waits = Cell::new(0);
        for _ in 0..3 {
            health.require_ready().unwrap();
            health
                .submit_and_wait(
                    || {
                        submits.set(submits.get() + 1);
                        Ok(())
                    },
                    || {
                        waits.set(waits.get() + 1);
                        Ok(())
                    },
                )
                .unwrap();
            health.require_ready().unwrap();
        }
        assert_eq!(submits.get(), 3);
        assert_eq!(waits.get(), 3);
    }

    #[test]
    fn submit_failure_poison_is_shared_and_refuses_later_mutation() {
        let health = QueueHealth::default();
        let result = health.submit_and_wait(
            || Err(vk_error("injected submit", vk::Result::ERROR_DEVICE_LOST)),
            || panic!("a failed submission must not be waited on"),
        );
        assert!(result.unwrap_err().to_string().contains("injected submit"));
        // Immediate matvec, routed dispatch, and a prepared program all check the
        // same context health before mapping/writing or resetting private resources.
        for _consumer in ["matvec", "routed dispatch", "prepared program"] {
            let writes = Cell::new(0);
            let attempt = || -> Result<(), DeviceError> {
                health.require_ready()?;
                writes.set(writes.get() + 1);
                Ok(())
            };
            assert!(attempt().unwrap_err().to_string().contains("poisoned"));
            assert_eq!(writes.get(), 0);
        }
        assert!(health
            .submit_and_wait(
                || panic!("poisoned contexts must not submit again"),
                || panic!("poisoned contexts must not wait again"),
            )
            .is_err());
    }

    #[test]
    fn wait_failure_keeps_shared_queue_poisoned() {
        let health = QueueHealth::default();
        let submits = Cell::new(0);
        let result = health.submit_and_wait(
            || {
                submits.set(submits.get() + 1);
                Ok(())
            },
            || {
                Err(vk_error(
                    "injected wait",
                    vk::Result::ERROR_OUT_OF_HOST_MEMORY,
                ))
            },
        );
        assert!(result.unwrap_err().to_string().contains("injected wait"));
        assert_eq!(submits.get(), 1);
        assert!(health.require_ready().is_err());
        assert!(health
            .submit_and_wait(
                || panic!("an unobserved completion must not permit another submission"),
                || panic!("an unobserved completion must not permit another wait"),
            )
            .is_err());
    }

    #[test]
    fn submission_panic_keeps_shared_queue_poisoned() {
        let health = QueueHealth::default();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            health.submit_and_wait(|| Ok(()), || panic!("injected wait panic"))
        }));
        assert!(result.is_err());
        assert!(health.require_ready().is_err());
    }

    fn assert_close(expected: &[f32], actual: &[f32], tolerance: f32) {
        assert_eq!(expected.len(), actual.len());
        for (index, (&expected, &actual)) in expected.iter().zip(actual).enumerate() {
            let difference = (expected - actual).abs();
            let allowed = tolerance * expected.abs().max(1.0);
            assert!(
                difference <= allowed,
                "output {index} differs by {difference:.3e}, exceeding {allowed:.3e}: CPU={expected:.7e}, Vulkan={actual:.7e}"
            );
        }
    }

    #[test]
    fn dispatch_grid_uses_two_dimensions_before_refusing() {
        assert_eq!(dispatch_grid(17, [64, 64, 1]).unwrap(), (17, 1));
        assert_eq!(dispatch_grid(129, [64, 64, 1]).unwrap(), (64, 3));
        assert!(dispatch_grid(65, [1, 64, 1]).is_err());
        assert!(dispatch_grid(u32::MAX, [65_536, 65_536, 1]).is_err());
    }

    #[test]
    fn tile4_records_are_stable_expert_major_chunks_with_valid_padding() {
        const A: [u32; 3] = [10, 20, 30];
        const B: [u32; 3] = [40, 50, 60];
        let keys = [A, B, A, A, A, A, B, B, B, B, B];
        let mut route_metadata = Vec::with_capacity(keys.len() * 4);
        for (occurrence, key) in keys.into_iter().enumerate() {
            route_metadata.extend_from_slice(&key);
            route_metadata.push(0x3f00_0000 + occurrence as u32);
        }
        let mut records = vec![0xdead_beef; keys.len() * ROUTED_MOE_TILE4_RECORD_WORDS];

        assert_eq!(write_tile4_records(&route_metadata, &mut records), 4);
        assert_eq!(
            &records[..4 * ROUTED_MOE_TILE4_RECORD_WORDS],
            &[
                10, 20, 30, 4, 0, 2, 3, 4, // A, first full chunk.
                10, 20, 30, 1, 5, 5, 5, 5, // A, residual padded with occurrence 5.
                40, 50, 60, 4, 1, 6, 7, 8, // B, first full chunk.
                40, 50, 60, 2, 9, 10, 9, 9, // B, residual padded with occurrence 9.
            ]
        );
        assert!(records[4 * ROUTED_MOE_TILE4_RECORD_WORDS..]
            .iter()
            .all(|&word| word == 0xdead_beef));
    }

    #[test]
    fn tile4_records_group_only_exact_absolute_projection_triples() {
        let keys = [[7, 8, 9], [7, 8, 10], [7, 8, 9], [7, 11, 9], [7, 8, 10]];
        let mut route_metadata = Vec::with_capacity(keys.len() * 4);
        for (occurrence, key) in keys.into_iter().enumerate() {
            route_metadata.extend_from_slice(&key);
            route_metadata.push(occurrence as u32);
        }
        let mut first = vec![0; keys.len() * ROUTED_MOE_TILE4_RECORD_WORDS];
        let mut second = first.clone();

        assert_eq!(write_tile4_records(&route_metadata, &mut first), 3);
        assert_eq!(write_tile4_records(&route_metadata, &mut second), 3);
        assert_eq!(first, second);
        assert_eq!(
            &first[..3 * ROUTED_MOE_TILE4_RECORD_WORDS],
            &[
                7, 8, 9, 2, 0, 2, 0, 0, // Exact triple 0, first-seen order.
                7, 8, 10, 2, 1, 4, 1, 1, // Different down projection.
                7, 11, 9, 1, 3, 3, 3, 3, // Different up projection.
            ]
        );
    }

    #[test]
    fn timestamp_math_handles_counter_wrap_and_preserves_stage_boundaries() {
        assert_eq!(timestamp_delta(100, 145, 36).unwrap(), 45);
        let mask_36 = (1u64 << 36) - 1;
        assert_eq!(timestamp_delta(mask_36 - 3, 2, 36).unwrap(), 6);
        assert_eq!(timestamp_delta(u64::MAX - 2, 1, 64).unwrap(), 4);
        assert!(timestamp_delta(0, 1, 35).is_err());

        let properties = TimestampProperties {
            valid_bits: 64,
            period_ns: 2.0,
        };
        let profile = routed_moe_gpu_profile(
            [10, 20, 22, 50, 55, 65],
            properties,
            Duration::from_nanos(7),
        )
        .unwrap();
        assert_eq!(profile.gate_up, Duration::from_nanos(20));
        assert_eq!(profile.gate_to_down_dependency, Duration::from_nanos(4));
        assert_eq!(profile.down, Duration::from_nanos(56));
        assert_eq!(profile.down_to_reduce_dependency, Duration::from_nanos(10));
        assert_eq!(profile.reduce, Duration::from_nanos(20));
        assert_eq!(profile.total, Duration::from_nanos(110));
        assert_eq!(profile.host_query_readback, Duration::from_nanos(7));
    }

    #[test]
    fn timestamp_support_requires_a_conformant_counter_and_period() {
        assert!(usable_timestamp_properties(0, 1.0).is_none());
        assert!(usable_timestamp_properties(35, 1.0).is_none());
        assert!(usable_timestamp_properties(65, 1.0).is_none());
        assert!(usable_timestamp_properties(64, 0.0).is_none());
        assert!(usable_timestamp_properties(64, f32::NAN).is_none());
        assert_eq!(
            usable_timestamp_properties(36, 2.5),
            Some(TimestampProperties {
                valid_bits: 36,
                period_ns: 2.5,
            })
        );
    }

    #[test]
    fn fast_routed_moe_planning_enforces_the_paired_shader_contract() {
        let grid = routed_moe_dispatch(
            RoutedMoeKernel::Fast,
            2048,
            512,
            8,
            [65_535, 65_535, 65_535],
            "test GPU",
        )
        .unwrap();
        assert_eq!(grid.gate_up, (512, 8));
        assert_eq!(grid.down, (2048, 8));
        assert_eq!(grid.reduce, (32, 1));

        assert!(
            routed_moe_dispatch(RoutedMoeKernel::Fast, 2049, 512, 8, [65_535; 3], "test GPU")
                .is_err()
        );
        assert!(
            routed_moe_dispatch(RoutedMoeKernel::Fast, 2048, 511, 8, [65_535; 3], "test GPU")
                .is_err()
        );
        assert!(routed_moe_dispatch(
            RoutedMoeKernel::Fast,
            2048,
            512,
            8,
            [1024, 65_535, 1],
            "test GPU"
        )
        .is_err());
        assert!(routed_moe_dispatch(
            RoutedMoeKernel::Fast,
            2048,
            512,
            8,
            [65_535, 4, 1],
            "test GPU"
        )
        .is_err());

        assert_eq!(
            routed_moe_metadata_offset(RoutedMoeKernel::Guarded, 12).unwrap(),
            12
        );
        assert_eq!(
            routed_moe_metadata_offset(RoutedMoeKernel::Fast, 12).unwrap(),
            3
        );
        assert!(routed_moe_metadata_offset(RoutedMoeKernel::Fast, 10).is_err());
    }

    #[test]
    fn promoted_workgroup_is_scoped_to_its_qualified_device_and_shape() {
        let qualified_driver = vk::make_api_version(0, 25, 2, 8);
        let select = |device, invocations, size_x, hidden, expert_width, routes| {
            preferred_fast_variant_for_device(
                device,
                invocations,
                size_x,
                qualified_driver,
                hidden,
                expert_width,
                routes,
            )
        };
        assert_eq!(
            select(
                "Radeon 8060S Graphics (RADV GFX1151)",
                1024,
                1024,
                2048,
                512,
                8
            ),
            RoutedMoeFastVariant::Tree128
        );
        assert_eq!(
            select(
                "Radeon 8060S Graphics (RADV GFX1151)",
                64,
                1024,
                2048,
                512,
                8
            ),
            RoutedMoeFastVariant::QualifiedTree64
        );
        assert_eq!(
            preferred_fast_variant_for_device(
                "Radeon 8060S Graphics (RADV GFX1151)",
                1024,
                1024,
                vk::make_api_version(0, 25, 2, 9),
                2048,
                512,
                8,
            ),
            RoutedMoeFastVariant::QualifiedTree64
        );
        assert_eq!(
            select(
                "Radeon 8060S Graphics (RADV GFX1151)",
                1024,
                1024,
                2048,
                768,
                8
            ),
            RoutedMoeFastVariant::QualifiedTree64
        );
        assert_eq!(
            select("another capable GPU", 1024, 1024, 2048, 512, 8),
            RoutedMoeFastVariant::QualifiedTree64
        );
    }

    #[test]
    fn routed_program_offsets_are_absolute_aligned_and_bounded() {
        let offsets = [[0, 16, 32], [40, 48, 56]];
        assert!(validate_absolute_fast_routed_moe_offsets(8, &offsets, 2, 72).is_ok());
        assert!(validate_absolute_fast_routed_moe_offsets(1, &offsets, 2, 72).is_err());
        assert!(validate_absolute_fast_routed_moe_offsets(8, &offsets, 2, 66).is_err());
        assert!(validate_absolute_fast_routed_moe_offsets(8, &offsets, 3, 72).is_err());
        assert!(
            validate_absolute_fast_routed_moe_offsets(usize::MAX, &[[4, 8, 12]], 1, 72).is_err()
        );
    }

    #[test]
    fn routed_program_batch_geometry_aligns_slices_and_refuses_overflow() {
        let four = NonZeroUsize::new(4).unwrap();
        let geometry =
            routed_moe_buffer_geometry(520, four, 256, 4096, "test GPU", "input").unwrap();
        assert_eq!(geometry.descriptor_bytes, 520);
        assert_eq!(geometry.item_stride, 768);
        assert_eq!(geometry.total_bytes, 3072);
        assert!(geometry.item_stride.is_multiple_of(256));
        for item in 0..four.get() {
            let offset = item * geometry.item_stride;
            assert!(offset.is_multiple_of(256));
            assert!(offset + geometry.descriptor_bytes <= geometry.total_bytes);
        }

        assert!(routed_moe_buffer_geometry(0, four, 8, 4096, "test GPU", "input").is_err());
        assert!(routed_moe_buffer_geometry(4097, four, 8, 4096, "test GPU", "input").is_err());
        assert!(routed_moe_buffer_geometry(
            1,
            NonZeroUsize::new(usize::MAX).unwrap(),
            8,
            usize::MAX,
            "test GPU",
            "input",
        )
        .is_err());
    }

    #[test]
    fn item_dimensional_program_geometry_is_tight_bounded_and_u32_indexable() {
        let batch_len = 4usize;
        let tight = |descriptor_bytes: usize| RoutedMoeBufferGeometry {
            descriptor_bytes,
            item_stride: descriptor_bytes,
            total_bytes: descriptor_bytes * batch_len,
        };
        let geometry = RoutedMoeProgramGeometry {
            hidden: 2048,
            expert_width: 512,
            routes: 8,
            batch_len,
            metadata_extra_bytes: 0,
            tile4: None,
            input: tight(2048 * 4),
            output: tight(2048 * 4),
            metadata: tight(8 * 4 * 4),
            hidden_scratch: tight(8 * 512 * 4),
            route_scratch: tight(8 * 2048 * 4),
            matrix_words: 2048 * 512 / 2,
            dispatch: RoutedMoeDispatch {
                gate_up: (512, 8),
                down: (2048, 8),
                reduce: (32, 1),
            },
        };
        let limits = [65_535, 65_535, 65_535];
        assert!(validate_item_dimensional_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_ok());
        assert!(validate_item_dimensional_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::QualifiedTree64,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_err());
        assert!(validate_item_dimensional_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            [65_535, 65_535, 3],
            usize::MAX,
            "test GPU",
        )
        .is_err());

        let mut padded = geometry;
        padded.input.item_stride += 8;
        padded.input.total_bytes = padded.input.item_stride * batch_len;
        assert!(validate_item_dimensional_routed_moe_program(
            padded,
            RoutedMoeFastVariant::Tree128,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_err());

        assert!(validate_item_dimensional_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            limits,
            geometry.route_scratch.total_bytes - 1,
            "test GPU",
        )
        .is_err());
        assert!(validate_glsl_u32_product(&[u32::MAX, 1], "limit").is_ok());
        assert!(validate_glsl_u32_product(&[u32::MAX, 2], "overflow").is_err());
    }

    #[test]
    fn tile4_program_geometry_accounts_for_the_complete_indirect_schedule() {
        let batch_len = 16usize;
        let tight = |descriptor_bytes: usize| RoutedMoeBufferGeometry {
            descriptor_bytes,
            item_stride: descriptor_bytes,
            total_bytes: descriptor_bytes * batch_len,
        };
        let mut geometry = RoutedMoeProgramGeometry {
            hidden: 2048,
            expert_width: 512,
            routes: 8,
            batch_len,
            input: tight(2048 * 4),
            output: tight(2048 * 4),
            metadata: RoutedMoeBufferGeometry {
                descriptor_bytes: 8 * 4 * 4,
                item_stride: 8 * 4 * 4,
                total_bytes: 6_168,
            },
            hidden_scratch: tight(8 * 512 * 4),
            route_scratch: tight(8 * 2048 * 4),
            metadata_extra_bytes: 4_120,
            matrix_words: 2048 * 512 / 2,
            dispatch: RoutedMoeDispatch {
                gate_up: (512, 8),
                down: (2048, 8),
                reduce: (32, 1),
            },
            tile4: Some(RoutedMoeTile4Geometry {
                gate_indirect_offset: 2_048,
                down_indirect_offset: 2_060,
                tile_word_offset: 518,
                max_tiles: 128,
                schedule_words: 1_024,
            }),
        };
        let limits = [65_535, 65_535, 65_535];
        assert!(validate_tile4_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_ok());
        let counts = routed_moe_program_byte_counts(geometry).unwrap();
        assert_eq!(counts.requested_bytes, 1_579_032);
        assert_eq!(counts.buffer_bytes, 1_579_032);

        assert!(validate_tile4_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            [65_535, 127, 65_535],
            usize::MAX,
            "test GPU",
        )
        .is_err());
        geometry.tile4.as_mut().unwrap().max_tiles -= 1;
        assert!(validate_tile4_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_err());
        geometry.tile4.as_mut().unwrap().max_tiles += 1;
        geometry.tile4.as_mut().unwrap().tile_word_offset += 1;
        assert!(validate_tile4_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_err());
        geometry.tile4.as_mut().unwrap().tile_word_offset -= 1;
        geometry.metadata.total_bytes -= 4;
        assert!(validate_tile4_routed_moe_program(
            geometry,
            RoutedMoeFastVariant::Tree128,
            limits,
            usize::MAX,
            "test GPU",
        )
        .is_err());
    }

    #[test]
    fn routed_program_memory_accounting_separates_payload_buffer_and_allocation_bytes() {
        let four = NonZeroUsize::new(4).unwrap();
        let buffer = |bytes, purpose| {
            routed_moe_buffer_geometry(bytes, four, 256, usize::MAX, "test GPU", purpose).unwrap()
        };
        let geometry = RoutedMoeProgramGeometry {
            hidden: 2048,
            expert_width: 512,
            routes: 8,
            batch_len: four.get(),
            metadata_extra_bytes: 0,
            tile4: None,
            input: buffer(2048 * 4, "input"),
            output: buffer(2048 * 4, "output"),
            metadata: buffer(8 * 4 * 4, "metadata"),
            hidden_scratch: buffer(8 * 512 * 4, "hidden scratch"),
            route_scratch: buffer(8 * 2048 * 4, "route scratch"),
            matrix_words: 2048 * 512 / 2,
            dispatch: RoutedMoeDispatch {
                gate_up: (512, 8),
                down: (2048, 8),
                reduce: (32, 1),
            },
        };
        let counts = routed_moe_program_byte_counts(geometry).unwrap();
        assert_eq!(counts.requested_bytes, 393_728);
        assert_eq!(counts.buffer_bytes, 394_240);

        let mut stats = RoutedMoeProgramStats {
            requested_bytes: counts.requested_bytes,
            buffer_bytes: 0,
            allocation_bytes: 0,
            allocation_slots: 0,
            budget_bytes: 131_072,
        };
        let first = BufferAllocation {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity: 60_000,
            allocation_bytes: 65_536,
        };
        let second = BufferAllocation {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity: 60_000,
            allocation_bytes: 65_536,
        };
        charge_routed_moe_program_allocation(&mut stats, &first).unwrap();
        charge_routed_moe_program_allocation(&mut stats, &second).unwrap();
        assert_eq!(stats.buffer_bytes, 120_000);
        assert_eq!(stats.allocation_bytes, 131_072);
        assert_eq!(stats.allocation_slots, 2);
        assert_eq!(routed_moe_program_remaining_budget(stats).unwrap(), 0);

        let before_refusal = stats;
        let excess = BufferAllocation {
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            capacity: 4,
            allocation_bytes: 1,
        };
        assert!(charge_routed_moe_program_allocation(&mut stats, &excess).is_err());
        assert_eq!(stats, before_refusal);
    }

    #[test]
    fn routed_program_memory_accounting_refuses_integer_overflow() {
        let buffer = RoutedMoeBufferGeometry {
            descriptor_bytes: usize::MAX,
            item_stride: usize::MAX,
            total_bytes: usize::MAX,
        };
        let geometry = RoutedMoeProgramGeometry {
            hidden: 2,
            expert_width: 2,
            routes: 1,
            batch_len: 2,
            metadata_extra_bytes: 0,
            tile4: None,
            input: buffer,
            output: buffer,
            metadata: buffer,
            hidden_scratch: buffer,
            route_scratch: buffer,
            matrix_words: 2,
            dispatch: RoutedMoeDispatch {
                gate_up: (1, 1),
                down: (1, 1),
                reduce: (1, 1),
            },
        };
        assert!(routed_moe_program_byte_counts(geometry).is_err());

        let stats = RoutedMoeProgramStats {
            allocation_bytes: 11,
            budget_bytes: 10,
            ..RoutedMoeProgramStats::default()
        };
        assert!(routed_moe_program_remaining_budget(stats).is_err());
    }

    #[test]
    fn routed_program_batch_descriptor_counts_are_exact_and_bounded() {
        assert_eq!(
            routed_moe_program_descriptor_counts(
                NonZeroUsize::new(1).unwrap(),
                RoutedMoeBatchDispatch::PerItem,
            )
            .unwrap(),
            (3, 12)
        );
        assert_eq!(
            routed_moe_program_descriptor_counts(
                NonZeroUsize::new(4).unwrap(),
                RoutedMoeBatchDispatch::PerItem,
            )
            .unwrap(),
            (12, 48)
        );
        assert_eq!(
            routed_moe_program_descriptor_counts(
                NonZeroUsize::new(usize::MAX).unwrap(),
                RoutedMoeBatchDispatch::ItemDimension,
            )
            .unwrap(),
            (3, 12)
        );
        assert_eq!(
            routed_moe_program_descriptor_counts(
                NonZeroUsize::new(usize::MAX).unwrap(),
                RoutedMoeBatchDispatch::ExpertTile4,
            )
            .unwrap(),
            (3, 12)
        );
        assert!(routed_moe_program_descriptor_counts(
            NonZeroUsize::new(usize::MAX).unwrap(),
            RoutedMoeBatchDispatch::PerItem,
        )
        .is_err());
        let too_many_sets = usize::try_from(u32::MAX).unwrap() / 3 + 1;
        assert!(routed_moe_program_descriptor_counts(
            NonZeroUsize::new(too_many_sets).unwrap(),
            RoutedMoeBatchDispatch::PerItem,
        )
        .is_err());
    }

    #[test]
    fn routed_program_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RoutedMoeProgram>();
    }

    #[test]
    fn supported_weight_validation_is_exact() {
        let bytes = vec![0u8; 2 * 34];
        let weight = Weight {
            bytes: &bytes,
            kind: beholdd_format::GgmlType(Q8_0_KIND),
            rows: 2,
            cols: 32,
        };
        let geometry = validate_weight(&weight).unwrap();
        assert_eq!(geometry.row_bytes, 34);
        assert_eq!(geometry.format, PreparedFormat::Q8_0);

        let truncated = Weight {
            bytes: &bytes[..67],
            ..weight
        };
        assert!(validate_weight(&truncated).is_err());

        let bf16_bytes = vec![0u8; 2 * 5 * 2];
        let bf16 = Weight {
            bytes: &bf16_bytes,
            kind: beholdd_format::GgmlType(BF16_KIND),
            rows: 2,
            cols: 5,
        };
        let geometry = validate_weight(&bf16).unwrap();
        assert_eq!(geometry.row_bytes, 10);
        assert_eq!(geometry.format, PreparedFormat::Bf16);
    }

    #[test]
    fn memory_selection_prefers_device_local_coherent_host_memory() {
        let mut properties = vk::PhysicalDeviceMemoryProperties {
            memory_type_count: 3,
            ..Default::default()
        };
        properties.memory_types[0].property_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        properties.memory_types[1].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::HOST_CACHED;
        properties.memory_types[2].property_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
            | vk::MemoryPropertyFlags::HOST_COHERENT
            | vk::MemoryPropertyFlags::DEVICE_LOCAL;
        assert_eq!(choose_memory_type(0b111, &properties), Some(2));
    }

    #[test]
    fn arena_tail_selection_aligns_and_uses_best_fit() {
        assert_eq!(align_up(34, 4, "test").unwrap(), 36);
        assert_eq!(align_up(34, 16, "test").unwrap(), 48);
        assert!(align_up(usize::MAX, 4, "test").is_err());

        let tails = [(34, 256), (96, 160), (0, 512)];
        // The middle block has the tightest viable tail after 16-byte alignment.
        assert_eq!(select_tail(&tails, 36, 16).unwrap(), Some((1, 96)));
        assert_eq!(select_tail(&[(224, 256)], 32, 4).unwrap(), Some((0, 224)));
        assert_eq!(select_tail(&[(225, 256)], 32, 4).unwrap(), None);
    }

    #[test]
    fn arena_geometry_accounts_for_shader_tail_words() {
        let q8 = validate_matrix_spec(MatrixSpec {
            kind: beholdd_format::GgmlType(Q8_0_KIND),
            rows: 1,
            cols: 32,
        })
        .unwrap();
        assert_eq!(q8.encoded_len, 34);
        assert_eq!(align_up(q8.encoded_len, 4, "test").unwrap(), 36);

        let bf16 = validate_matrix_spec(MatrixSpec {
            kind: beholdd_format::GgmlType(BF16_KIND),
            rows: 3,
            cols: 5,
        })
        .unwrap();
        assert_eq!(bf16.encoded_len, 30);
        assert_eq!(align_up(bf16.encoded_len, 4, "test").unwrap(), 32);
    }

    #[test]
    #[ignore = "requires a non-CPU Vulkan device; run explicitly on an acceleration host"]
    fn live_weight_arena_suballocates_and_keeps_handles_alive() {
        let cpu = Cpu::new();
        let vulkan = VulkanMatvec::new().unwrap();
        let bf16_values: Vec<f32> = (0..15).map(|index| index as f32 * 0.125 - 0.75).collect();
        let bf16_bytes: Vec<u8> = bf16_values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let bf16 = Weight {
            bytes: &bf16_bytes,
            kind: beholdd_format::GgmlType(BF16_KIND),
            rows: 3,
            cols: 5,
        };
        let mut q8_bytes = Vec::with_capacity(Q8_0_BYTES);
        q8_bytes.extend_from_slice(&0x3800u16.to_le_bytes());
        q8_bytes.extend((-16i8..16).map(|value| value as u8));
        let q8 = Weight {
            bytes: &q8_bytes,
            kind: beholdd_format::GgmlType(Q8_0_KIND),
            rows: 1,
            cols: 32,
        };

        let mut builder = vulkan
            .arena_builder(WeightArenaConfig {
                block_bytes: 4096,
                budget_bytes: 1024 * 1024,
                allocation_headroom: 64,
            })
            .unwrap();
        let bf16_id = builder.push(&bf16).unwrap();
        let q8_id = builder.push(&q8).unwrap();
        let arena = builder.finish().unwrap();
        assert_eq!(arena.stats().matrices, 2);
        assert_eq!(arena.stats().blocks, 1);
        assert_eq!(arena.stats().encoded_bytes, 64);
        assert_eq!(arena.stats().descriptor_bytes, 68);

        let prepared_bf16 = arena.matrix(bf16_id).unwrap();
        let prepared_q8 = arena.matrix(q8_id).unwrap();
        assert_ne!(prepared_q8.record().offset, 0);
        drop(arena);

        let bf16_input = [0.25, -0.5, 0.75, 1.0, -1.25];
        let mut expected = vec![0.0; 3];
        let mut actual = vec![0.0; 3];
        cpu.matvec(&mut expected, &bf16_input, &bf16).unwrap();
        vulkan
            .matvec(&mut actual, &bf16_input, &prepared_bf16)
            .unwrap();
        assert_close(&expected, &actual, 1e-6);

        let q8_input: Vec<f32> = (0..32).map(|index| index as f32 * 0.03125 - 0.5).collect();
        let mut expected = [0.0];
        let mut actual = [0.0];
        cpu.matvec(&mut expected, &q8_input, &q8).unwrap();
        vulkan.matvec(&mut actual, &q8_input, &prepared_q8).unwrap();
        assert_close(&expected, &actual, 1e-6);
    }

    #[test]
    #[ignore = "requires a non-CPU Vulkan device; run explicitly on an acceleration host"]
    fn live_fused_routed_moe_matches_sequential_cpu_and_is_deterministic() {
        let cpu = Cpu::new();
        let vulkan = VulkanMatvec::new().unwrap();
        let (hidden, expert_width, routes) = (5usize, 3usize, 3usize);
        let encode = |rows: usize, cols: usize, seed: usize| {
            (0..rows * cols)
                .flat_map(|index| {
                    let value = (((index + seed * 7) as f32) * 0.173).sin() * 0.35;
                    ((value.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect::<Vec<_>>()
        };
        let host_weights: Vec<_> = (0..routes)
            .map(|route| {
                (
                    encode(expert_width, hidden, route * 3),
                    encode(expert_width, hidden, route * 3 + 1),
                    encode(hidden, expert_width, route * 3 + 2),
                )
            })
            .collect();
        let mut builder = vulkan
            .arena_builder(WeightArenaConfig {
                block_bytes: 4096,
                budget_bytes: 1024 * 1024,
                allocation_headroom: 64,
            })
            .unwrap();
        let mut ids = Vec::new();
        for (gate, up, down) in &host_weights {
            let gate = Weight {
                bytes: gate,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: expert_width,
                cols: hidden,
            };
            let up = Weight { bytes: up, ..gate };
            let down = Weight {
                bytes: down,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: hidden,
                cols: expert_width,
            };
            ids.push([
                builder.push(&gate).unwrap(),
                builder.push(&up).unwrap(),
                builder.push(&down).unwrap(),
            ]);
        }
        let arena = builder.finish().unwrap();
        let matrices: Vec<_> = ids
            .iter()
            .map(|ids| {
                [
                    arena.matrix(ids[0]).unwrap(),
                    arena.matrix(ids[1]).unwrap(),
                    arena.matrix(ids[2]).unwrap(),
                ]
            })
            .collect();
        let views: Vec<_> = matrices
            .iter()
            .map(|matrices| Bf16Expert {
                gate: &matrices[0],
                up: &matrices[1],
                down: &matrices[2],
            })
            .collect();
        let prepared = vulkan.prepare_bf16_routed_experts(&views).unwrap();
        assert_eq!(
            vulkan.preferred_routed_moe_kernel(&prepared),
            RoutedMoeKernel::Guarded
        );
        let input = [0.25f32, -0.5, 0.75, 1.0, -1.25];
        let route_weights = [0.51f32, 0.31, 0.18];
        let mut expected = vec![0.0f32; hidden];
        for (route, (gate_bytes, up_bytes, down_bytes)) in host_weights.iter().enumerate() {
            let gate_weight = Weight {
                bytes: gate_bytes,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: expert_width,
                cols: hidden,
            };
            let up_weight = Weight {
                bytes: up_bytes,
                ..gate_weight
            };
            let down_weight = Weight {
                bytes: down_bytes,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: hidden,
                cols: expert_width,
            };
            let mut gate = vec![0.0; expert_width];
            let mut up = vec![0.0; expert_width];
            cpu.matvec(&mut gate, &input, &gate_weight).unwrap();
            cpu.matvec(&mut up, &input, &up_weight).unwrap();
            cpu.swiglu(&mut gate, &up, 0.0).unwrap();
            cpu.matvec_add(&mut expected, &gate, &down_weight, route_weights[route])
                .unwrap();
        }

        let mut actual = vec![0.0f32; hidden];
        vulkan
            .routed_moe_bf16(&mut actual, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_close(&expected, &actual, 2e-5);
        let first_bits: Vec<_> = actual.iter().map(|value| value.to_bits()).collect();
        vulkan
            .routed_moe_bf16(&mut actual, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            actual
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let initial = vec![0.125f32; hidden];
        let expected_added: Vec<_> = expected.iter().map(|value| value + 0.125).collect();
        let mut actual_added = initial;
        vulkan
            .routed_moe_bf16(&mut actual_added, &input, &prepared, &route_weights, true)
            .unwrap();
        assert_close(&expected_added, &actual_added, 2e-5);

        let sentinel = vec![19.75f32; hidden];
        let mut refused = sentinel.clone();
        let error = vulkan
            .routed_moe_bf16_fast(&mut refused, &input, &prepared, &route_weights, false)
            .unwrap_err();
        assert!(matches!(error, DeviceError::Shape(_)));
        assert_eq!(refused, sentinel, "a refused fast call must not modify y");
    }

    #[test]
    #[ignore = "requires a non-CPU Vulkan device; run explicitly on an acceleration host"]
    fn live_fast_routed_moe_matches_cpu_safe_and_accumulate_semantics() {
        let cpu = Cpu::new();
        let vulkan = VulkanMatvec::new().unwrap();
        let (hidden, expert_width, routes) = (130usize, 66usize, 3usize);
        let encode = |rows: usize, cols: usize, seed: usize| {
            (0..rows * cols)
                .flat_map(|index| {
                    let value = (((index + seed * 17) as f32) * 0.071).sin() * 0.04;
                    ((value.to_bits() >> 16) as u16).to_le_bytes()
                })
                .collect::<Vec<_>>()
        };
        // Expert zero is packed first but excluded from the initial K=3 selection.
        // That forces a nonzero absolute base offset and leaves a same-shape fourth
        // expert available for a true route replacement later in the test.
        let host_weights: Vec<_> = (0..routes + 1)
            .map(|route| {
                (
                    encode(expert_width, hidden, route * 3),
                    encode(expert_width, hidden, route * 3 + 1),
                    encode(hidden, expert_width, route * 3 + 2),
                )
            })
            .collect();
        let mut builder = vulkan
            .arena_builder(WeightArenaConfig {
                block_bytes: 1024 * 1024,
                budget_bytes: 1024 * 1024,
                allocation_headroom: 64,
            })
            .unwrap();
        let mut ids = Vec::new();
        for (gate, up, down) in &host_weights {
            let gate = Weight {
                bytes: gate,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: expert_width,
                cols: hidden,
            };
            let up = Weight { bytes: up, ..gate };
            let down = Weight {
                bytes: down,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: hidden,
                cols: expert_width,
            };
            ids.push([
                builder.push(&gate).unwrap(),
                builder.push(&up).unwrap(),
                builder.push(&down).unwrap(),
            ]);
        }
        let arena = builder.finish().unwrap();
        let matrices: Vec<_> = ids
            .iter()
            .map(|ids| {
                [
                    arena.matrix(ids[0]).unwrap(),
                    arena.matrix(ids[1]).unwrap(),
                    arena.matrix(ids[2]).unwrap(),
                ]
            })
            .collect();
        let views: Vec<_> = matrices
            .iter()
            .map(|matrices| Bf16Expert {
                gate: &matrices[0],
                up: &matrices[1],
                down: &matrices[2],
            })
            .collect();
        let selected_views = [views[1], views[2], views[3]];
        let prepared = vulkan.prepare_bf16_routed_experts(&selected_views).unwrap();
        assert_ne!(prepared.weight_offset, 0);
        assert_eq!(
            vulkan.preferred_routed_moe_kernel(&prepared),
            RoutedMoeKernel::Fast
        );
        let input: Vec<f32> = (0..hidden)
            .map(|index| ((index as f32) * 0.037).cos() * 0.3)
            .collect();
        let route_weights = [0.53f32, 0.29, 0.18];

        let mut expected = vec![0.0f32; hidden];
        for (route, (gate_bytes, up_bytes, down_bytes)) in host_weights[1..].iter().enumerate() {
            let gate_weight = Weight {
                bytes: gate_bytes,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: expert_width,
                cols: hidden,
            };
            let up_weight = Weight {
                bytes: up_bytes,
                ..gate_weight
            };
            let down_weight = Weight {
                bytes: down_bytes,
                kind: beholdd_format::GgmlType(BF16_KIND),
                rows: hidden,
                cols: expert_width,
            };
            let mut gate = vec![0.0; expert_width];
            let mut up = vec![0.0; expert_width];
            cpu.matvec(&mut gate, &input, &gate_weight).unwrap();
            cpu.matvec(&mut up, &input, &up_weight).unwrap();
            cpu.swiglu(&mut gate, &up, 0.0).unwrap();
            cpu.matvec_add(&mut expected, &gate, &down_weight, route_weights[route])
                .unwrap();
        }

        let mut safe = vec![17.0f32; hidden];
        vulkan
            .routed_moe_bf16_guarded(&mut safe, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_close(&expected, &safe, 8e-5);

        let mut fast = vec![-23.0f32; hidden];
        vulkan
            .routed_moe_bf16_fast(&mut fast, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_close(&expected, &fast, 8e-5);
        assert_close(&safe, &fast, 8e-5);
        let first_bits: Vec<_> = fast.iter().map(|value| value.to_bits()).collect();
        vulkan
            .routed_moe_bf16_fast(&mut fast, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            fast.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
        );

        let program_config = RoutedMoeProgramConfig {
            budget_bytes: 1024 * 1024,
            allocation_headroom: 64,
        };
        let allocations_before_program = vulkan.inner.live_allocations.load(Ordering::Acquire);
        let program = vulkan
            .prepare_routed_moe_bf16_fast_program(&prepared, program_config)
            .unwrap();
        let program_stats = program.stats();
        assert!(program_stats.requested_bytes <= program_stats.buffer_bytes);
        assert!(program_stats.buffer_bytes <= program_stats.allocation_bytes);
        assert!(program_stats.allocation_bytes <= program_stats.budget_bytes);
        assert_eq!(program_stats.budget_bytes, program_config.budget_bytes);
        assert_eq!(
            program_stats.allocation_slots,
            ROUTED_MOE_PROGRAM_ALLOCATION_SLOTS
        );
        assert_eq!(
            vulkan.inner.live_allocations.load(Ordering::Acquire),
            allocations_before_program + ROUTED_MOE_PROGRAM_ALLOCATION_SLOTS
        );

        let allocations_before_exact = vulkan.inner.live_allocations.load(Ordering::Acquire);
        let exact_budget_program = vulkan
            .prepare_routed_moe_bf16_fast_program(
                &prepared,
                RoutedMoeProgramConfig {
                    budget_bytes: program_stats.allocation_bytes,
                    ..program_config
                },
            )
            .unwrap();
        assert_eq!(
            exact_budget_program.stats().allocation_bytes,
            program_stats.allocation_bytes
        );
        drop(exact_budget_program);
        assert_eq!(
            vulkan.inner.live_allocations.load(Ordering::Acquire),
            allocations_before_exact,
            "dropping an exact-budget program must release all allocation slots"
        );

        let allocations_before_tight = vulkan.inner.live_allocations.load(Ordering::Acquire);
        let tight_budget_error = vulkan
            .prepare_routed_moe_bf16_fast_program(
                &prepared,
                RoutedMoeProgramConfig {
                    budget_bytes: program_stats.allocation_bytes - 1,
                    ..program_config
                },
            )
            .unwrap_err();
        let tight_budget_message = tight_budget_error.to_string();
        assert!(tight_budget_message.contains("authorized"));
        assert!(tight_budget_message.contains("budget"));
        assert_eq!(
            vulkan.inner.live_allocations.load(Ordering::Acquire),
            allocations_before_tight,
            "a budget refusal must release every partial allocation"
        );

        let allocation_limit = allocations_before_tight + ROUTED_MOE_PROGRAM_ALLOCATION_SLOTS - 1;
        let slot_error = vulkan
            .prepare_routed_moe_bf16_fast_program(
                &prepared,
                RoutedMoeProgramConfig {
                    budget_bytes: program_config.budget_bytes,
                    allocation_headroom: vulkan.inner.max_memory_allocations - allocation_limit,
                },
            )
            .unwrap_err();
        assert!(slot_error.to_string().contains("allocation slot"));
        assert_eq!(
            vulkan.inner.live_allocations.load(Ordering::Acquire),
            allocations_before_tight,
            "an allocation-slot refusal must release every partial allocation"
        );
        assert_eq!(program.hidden(), hidden);
        assert_eq!(program.expert_width(), expert_width);
        assert_eq!(program.routes(), routes);
        assert_eq!(
            program.fast_variant(),
            vulkan.selected_routed_moe_fast_variant(&prepared)
        );
        let mut programmed = vec![101.0f32; hidden];
        program
            .execute(&mut programmed, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            programmed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "the prerecorded overwrite path changed fast-kernel bits"
        );
        program
            .execute(&mut programmed, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            programmed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reusing the prerecorded overwrite command buffer changed output"
        );

        let changed_input: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(index, value)| value * 0.73 + (index % 7) as f32 * 0.002)
            .collect();
        let changed_route_weights = [0.17f32, 0.61, 0.22];
        let mut ordinary_changed = vec![0.0f32; hidden];
        vulkan
            .routed_moe_bf16_fast(
                &mut ordinary_changed,
                &changed_input,
                &prepared,
                &changed_route_weights,
                false,
            )
            .unwrap();
        let mut programmed_changed = vec![0.0f32; hidden];
        program
            .execute(
                &mut programmed_changed,
                &changed_input,
                &prepared,
                &changed_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            ordinary_changed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            programmed_changed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "mapped input or route metadata did not update between executions"
        );

        let reordered_views = [views[3], views[1], views[2]];
        let reordered = vulkan
            .prepare_bf16_routed_experts(&reordered_views)
            .unwrap();
        let reordered_route_weights = [0.41f32, 0.37, 0.22];
        let mut ordinary_reordered = vec![0.0f32; hidden];
        vulkan
            .routed_moe_bf16_fast(
                &mut ordinary_reordered,
                &input,
                &reordered,
                &reordered_route_weights,
                false,
            )
            .unwrap();
        let mut programmed_reordered = vec![0.0f32; hidden];
        program
            .execute(
                &mut programmed_reordered,
                &input,
                &reordered,
                &reordered_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            ordinary_reordered
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            programmed_reordered
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "same-block expert reordering changed prerecorded execution"
        );

        let replacement_views = [views[0], views[2], views[3]];
        let replacement = vulkan
            .prepare_bf16_routed_experts(&replacement_views)
            .unwrap();
        let replacement_route_weights = [0.23f32, 0.32, 0.45];
        let mut ordinary_replacement = vec![0.0f32; hidden];
        vulkan
            .routed_moe_bf16_fast(
                &mut ordinary_replacement,
                &changed_input,
                &replacement,
                &replacement_route_weights,
                false,
            )
            .unwrap();
        let mut programmed_replacement = vec![0.0f32; hidden];
        program
            .execute(
                &mut programmed_replacement,
                &changed_input,
                &replacement,
                &replacement_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            ordinary_replacement
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            programmed_replacement
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "same-block expert replacement changed prerecorded execution"
        );

        program
            .execute(&mut programmed, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            programmed
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "interleaved ordinary calls mutated private program state"
        );
        let mut profiled = vec![37.0f32; hidden];
        let first_profile = vulkan
            .routed_moe_bf16_fast_profiled(&mut profiled, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            profiled
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "profiling changed paired-load output"
        );
        let second_profile = vulkan
            .routed_moe_bf16_fast_profiled(&mut profiled, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            profiled
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reusing the timestamp query pool changed paired-load output"
        );
        for profile in [first_profile, second_profile] {
            assert!((36..=64).contains(&profile.counter_valid_bits));
            assert!(profile.timestamp_period_ns.is_finite());
            assert!(profile.timestamp_period_ns > 0.0);
            for stage in [
                profile.gate_up,
                profile.gate_to_down_dependency,
                profile.down,
                profile.down_to_reduce_dependency,
                profile.reduce,
            ] {
                assert!(profile.total >= stage);
            }
        }
        let mut automatic = vec![91.0f32; hidden];
        vulkan
            .routed_moe_bf16(&mut automatic, &input, &prepared, &route_weights, false)
            .unwrap();
        assert_eq!(
            first_bits,
            automatic
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "eligible automatic dispatch did not select the fast kernel"
        );

        let initial: Vec<f32> = (0..hidden)
            .map(|index| 0.125 + (index % 11) as f32 * 0.001)
            .collect();
        let expected_added: Vec<_> = initial
            .iter()
            .zip(&fast)
            .map(|(&base, &delta)| base + delta)
            .collect();
        let mut actual_added = initial;
        vulkan
            .routed_moe_bf16_fast(&mut actual_added, &input, &prepared, &route_weights, true)
            .unwrap();
        assert_close(&expected_added, &actual_added, 2e-5);
        let mut programmed_added: Vec<f32> = (0..hidden)
            .map(|index| 0.125 + (index % 11) as f32 * 0.001)
            .collect();
        program
            .execute(
                &mut programmed_added,
                &input,
                &prepared,
                &route_weights,
                true,
            )
            .unwrap();
        assert_eq!(
            actual_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            programmed_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "the prerecorded accumulation path changed fast-kernel bits"
        );

        let batch_program = vulkan
            .prepare_routed_moe_bf16_fast_batch_program(
                &prepared,
                NonZeroUsize::new(2).unwrap(),
                RoutedMoeProgramConfig {
                    budget_bytes: 1024 * 1024,
                    allocation_headroom: 64,
                },
            )
            .unwrap();
        let item_dimensional_batch_program = vulkan
            .prepare_routed_moe_bf16_fast_batch_program_with_dispatch(
                &prepared,
                NonZeroUsize::new(2).unwrap(),
                RoutedMoeProgramConfig {
                    budget_bytes: 1024 * 1024,
                    allocation_headroom: 64,
                },
                RoutedMoeBatchDispatch::ItemDimension,
            )
            .unwrap();
        let tile4_batch_program = vulkan
            .prepare_routed_moe_bf16_fast_batch_program_with_dispatch(
                &prepared,
                NonZeroUsize::new(2).unwrap(),
                RoutedMoeProgramConfig {
                    budget_bytes: 1024 * 1024,
                    allocation_headroom: 64,
                },
                RoutedMoeBatchDispatch::ExpertTile4,
            )
            .unwrap();
        assert_eq!(batch_program.batch_len(), 2);
        assert_eq!(
            batch_program.batch_dispatch(),
            RoutedMoeBatchDispatch::PerItem
        );
        assert_eq!(item_dimensional_batch_program.batch_len(), 2);
        assert_eq!(
            item_dimensional_batch_program.batch_dispatch(),
            RoutedMoeBatchDispatch::ItemDimension
        );
        assert_eq!(
            item_dimensional_batch_program.fast_variant(),
            RoutedMoeFastVariant::Tree128
        );
        assert_eq!(tile4_batch_program.batch_len(), 2);
        assert_eq!(
            tile4_batch_program.batch_dispatch(),
            RoutedMoeBatchDispatch::ExpertTile4
        );
        assert_eq!(
            tile4_batch_program.fast_variant(),
            RoutedMoeFastVariant::Tree128
        );
        assert_eq!(batch_program.hidden(), hidden);
        assert_eq!(batch_program.expert_width(), expert_width);
        assert_eq!(batch_program.routes(), routes);
        assert_eq!(
            batch_program.stats().requested_bytes,
            2 * program_stats.requested_bytes
        );
        assert_eq!(
            batch_program.stats().buffer_bytes,
            2 * program_stats.buffer_bytes
        );
        let batch_experts = [prepared.clone(), replacement.clone()];
        let mut batch_inputs = Vec::with_capacity(2 * hidden);
        batch_inputs.extend_from_slice(&input);
        batch_inputs.extend_from_slice(&changed_input);
        let mut batch_route_weights = Vec::with_capacity(2 * routes);
        batch_route_weights.extend_from_slice(&route_weights);
        batch_route_weights.extend_from_slice(&replacement_route_weights);

        let mut separate = vec![0.0f32; 2 * hidden];
        vulkan
            .routed_moe_bf16_fast(
                &mut separate[..hidden],
                &batch_inputs[..hidden],
                &batch_experts[0],
                &batch_route_weights[..routes],
                false,
            )
            .unwrap();
        vulkan
            .routed_moe_bf16_fast(
                &mut separate[hidden..],
                &batch_inputs[hidden..],
                &batch_experts[1],
                &batch_route_weights[routes..],
                false,
            )
            .unwrap();
        let separate_bits: Vec<_> = separate.iter().map(|value| value.to_bits()).collect();
        let mut tree128_separate = vec![0.0f32; 2 * hidden];
        vulkan
            .routed_moe_bf16_fast_variant(
                &mut tree128_separate[..hidden],
                &batch_inputs[..hidden],
                &batch_experts[0],
                &batch_route_weights[..routes],
                false,
                RoutedMoeFastVariant::Tree128,
            )
            .unwrap();
        vulkan
            .routed_moe_bf16_fast_variant(
                &mut tree128_separate[hidden..],
                &batch_inputs[hidden..],
                &batch_experts[1],
                &batch_route_weights[routes..],
                false,
                RoutedMoeFastVariant::Tree128,
            )
            .unwrap();
        let tree128_separate_bits: Vec<_> = tree128_separate
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let mut batched = (0..2 * hidden)
            .map(|index| -71.0 + index as f32 * 0.001)
            .collect::<Vec<_>>();
        batch_program
            .execute_batch(
                &mut batched,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            separate_bits,
            batched
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "fixed-N overwrite changed bits relative to independent fast calls"
        );
        let mut item_dimensional = vec![97.0f32; 2 * hidden];
        item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            tree128_separate_bits,
            item_dimensional
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "item-dimensional overwrite changed Tree128 arithmetic bits"
        );
        let mut tile4 = vec![-131.0f32; 2 * hidden];
        tile4_batch_program
            .execute_batch(
                &mut tile4,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            tree128_separate_bits,
            tile4
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "expert-major Tile4 overwrite changed Tree128 arithmetic bits"
        );
        tile4.fill(149.0);
        tile4_batch_program
            .execute_batch(
                &mut tile4,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            tree128_separate_bits,
            tile4
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reusing the expert-major Tile4 command buffer changed output"
        );
        item_dimensional.fill(-101.0);
        item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            tree128_separate_bits,
            item_dimensional
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reusing the item-dimensional command buffer changed output"
        );
        batched.fill(83.0);
        batch_program
            .execute_batch(
                &mut batched,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            separate_bits,
            batched
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "reusing the fixed-N overwrite command buffer changed output"
        );
        let (concurrent_first, concurrent_second) = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                let mut output = vec![0.0; 2 * hidden];
                batch_program
                    .execute_batch(
                        &mut output,
                        &batch_inputs,
                        &batch_experts,
                        &batch_route_weights,
                        false,
                    )
                    .unwrap();
                output
            });
            let second = scope.spawn(|| {
                let mut output = vec![0.0; 2 * hidden];
                batch_program
                    .execute_batch(
                        &mut output,
                        &batch_inputs,
                        &batch_experts,
                        &batch_route_weights,
                        false,
                    )
                    .unwrap();
                output
            });
            (first.join().unwrap(), second.join().unwrap())
        });
        for concurrent in [concurrent_first, concurrent_second] {
            assert_eq!(
                separate_bits,
                concurrent
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "concurrent calls were not serialized against private program state"
            );
        }

        let batch_initial = (0..2 * hidden)
            .map(|index| 0.25 + (index % 17) as f32 * 0.003)
            .collect::<Vec<_>>();
        let mut separate_added = batch_initial.clone();
        vulkan
            .routed_moe_bf16_fast(
                &mut separate_added[..hidden],
                &batch_inputs[..hidden],
                &batch_experts[0],
                &batch_route_weights[..routes],
                true,
            )
            .unwrap();
        vulkan
            .routed_moe_bf16_fast(
                &mut separate_added[hidden..],
                &batch_inputs[hidden..],
                &batch_experts[1],
                &batch_route_weights[routes..],
                true,
            )
            .unwrap();
        let mut batched_added = batch_initial;
        batch_program
            .execute_batch(
                &mut batched_added,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                true,
            )
            .unwrap();
        assert_eq!(
            separate_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            batched_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "fixed-N accumulation changed bits relative to independent fast calls"
        );
        let item_dimensional_initial = (0..2 * hidden)
            .map(|index| 0.25 + (index % 17) as f32 * 0.003)
            .collect::<Vec<_>>();
        let mut tree128_separate_added = item_dimensional_initial.clone();
        vulkan
            .routed_moe_bf16_fast_variant(
                &mut tree128_separate_added[..hidden],
                &batch_inputs[..hidden],
                &batch_experts[0],
                &batch_route_weights[..routes],
                true,
                RoutedMoeFastVariant::Tree128,
            )
            .unwrap();
        vulkan
            .routed_moe_bf16_fast_variant(
                &mut tree128_separate_added[hidden..],
                &batch_inputs[hidden..],
                &batch_experts[1],
                &batch_route_weights[routes..],
                true,
                RoutedMoeFastVariant::Tree128,
            )
            .unwrap();
        let mut item_dimensional_added = item_dimensional_initial;
        item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional_added,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                true,
            )
            .unwrap();
        assert_eq!(
            tree128_separate_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            item_dimensional_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "item-dimensional accumulation changed Tree128 arithmetic bits"
        );
        let tile4_initial = (0..2 * hidden)
            .map(|index| 0.25 + (index % 17) as f32 * 0.003)
            .collect::<Vec<_>>();
        let mut tile4_added = tile4_initial;
        tile4_batch_program
            .execute_batch(
                &mut tile4_added,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                true,
            )
            .unwrap();
        assert_eq!(
            tree128_separate_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            tile4_added
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "expert-major Tile4 accumulation changed Tree128 arithmetic bits"
        );

        let mut isolated_inputs = batch_inputs.clone();
        for value in &mut isolated_inputs[..hidden] {
            *value = *value * -0.37 + 0.019;
        }
        let mut isolated = vec![0.0; 2 * hidden];
        batch_program
            .execute_batch(
                &mut isolated,
                &isolated_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        let mut separate_isolated_first = vec![0.0; hidden];
        vulkan
            .routed_moe_bf16_fast(
                &mut separate_isolated_first,
                &isolated_inputs[..hidden],
                &batch_experts[0],
                &batch_route_weights[..routes],
                false,
            )
            .unwrap();
        assert_eq!(
            separate_isolated_first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            isolated[..hidden]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &separate_bits[hidden..],
            &isolated[hidden..]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "changing item zero leaked into item one"
        );
        let mut item_dimensional_isolated = vec![0.0; 2 * hidden];
        item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional_isolated,
                &isolated_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        let mut tree128_isolated_first = vec![0.0; hidden];
        vulkan
            .routed_moe_bf16_fast_variant(
                &mut tree128_isolated_first,
                &isolated_inputs[..hidden],
                &batch_experts[0],
                &batch_route_weights[..routes],
                false,
                RoutedMoeFastVariant::Tree128,
            )
            .unwrap();
        assert_eq!(
            tree128_isolated_first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            item_dimensional_isolated[..hidden]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &tree128_separate_bits[hidden..],
            &item_dimensional_isolated[hidden..]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "changing item zero leaked into item one in item-dimensional dispatch"
        );
        let mut tile4_isolated = vec![0.0; 2 * hidden];
        tile4_batch_program
            .execute_batch(
                &mut tile4_isolated,
                &isolated_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .unwrap();
        assert_eq!(
            tree128_isolated_first
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            tile4_isolated[..hidden]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            &tree128_separate_bits[hidden..],
            &tile4_isolated[hidden..]
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "changing item zero leaked into item one in expert-major Tile4 dispatch"
        );

        let batch_sentinel = vec![19.75f32; 2 * hidden];
        let mut batch_refused = batch_sentinel.clone();
        assert!(batch_program
            .execute(
                &mut batch_refused[..hidden],
                &input,
                &prepared,
                &route_weights,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        assert!(batch_program
            .execute_batch(
                &mut batch_refused[..2 * hidden - 1],
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        assert!(batch_program
            .execute_batch(
                &mut batch_refused,
                &batch_inputs[..2 * hidden - 1],
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        assert!(batch_program
            .execute_batch(
                &mut batch_refused,
                &batch_inputs,
                &batch_experts[..1],
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        assert!(batch_program
            .execute_batch(
                &mut batch_refused,
                &batch_inputs,
                &batch_experts,
                &batch_route_weights[..2 * routes - 1],
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        let mut batch_non_finite = batch_route_weights.clone();
        batch_non_finite[routes + 1] = f32::INFINITY;
        assert!(batch_program
            .execute_batch(
                &mut batch_refused,
                &batch_inputs,
                &batch_experts,
                &batch_non_finite,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        let mut invalid_batch_experts = batch_experts.clone();
        invalid_batch_experts[1].offsets_bytes[0][0] = u32::MAX;
        assert!(batch_program
            .execute_batch(
                &mut batch_refused,
                &batch_inputs,
                &invalid_batch_experts,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);
        let mut invalid_batch_block = batch_experts.clone();
        invalid_batch_block[1].block = invalid_batch_block[1].block.saturating_add(1);
        assert!(batch_program
            .execute_batch(
                &mut batch_refused,
                &batch_inputs,
                &invalid_batch_block,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(batch_refused, batch_sentinel);

        let item_dimensional_sentinel = vec![-29.5f32; 2 * hidden];
        let mut item_dimensional_refused = item_dimensional_sentinel.clone();
        assert!(item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional_refused[..2 * hidden - 1],
                &batch_inputs,
                &batch_experts,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(item_dimensional_refused, item_dimensional_sentinel);

        let tile4_sentinel = vec![43.25f32; 2 * hidden];
        let mut tile4_refused = tile4_sentinel.clone();
        assert!(tile4_batch_program
            .execute_batch(
                &mut tile4_refused,
                &batch_inputs,
                &batch_experts,
                &batch_non_finite,
                false,
            )
            .is_err());
        assert_eq!(tile4_refused, tile4_sentinel);
        assert!(tile4_batch_program
            .execute_batch(
                &mut tile4_refused,
                &batch_inputs,
                &invalid_batch_experts,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(tile4_refused, tile4_sentinel);
        assert!(tile4_batch_program
            .execute_batch(
                &mut tile4_refused,
                &batch_inputs,
                &invalid_batch_block,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(tile4_refused, tile4_sentinel);
        assert!(item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional_refused,
                &batch_inputs,
                &batch_experts,
                &batch_non_finite,
                false,
            )
            .is_err());
        assert_eq!(item_dimensional_refused, item_dimensional_sentinel);
        assert!(item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional_refused,
                &batch_inputs,
                &invalid_batch_experts,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(item_dimensional_refused, item_dimensional_sentinel);
        assert!(item_dimensional_batch_program
            .execute_batch(
                &mut item_dimensional_refused,
                &batch_inputs,
                &invalid_batch_block,
                &batch_route_weights,
                false,
            )
            .is_err());
        assert_eq!(item_dimensional_refused, item_dimensional_sentinel);

        let sentinel = vec![19.75f32; hidden];
        let mut refused = sentinel.clone();
        assert!(program
            .execute(
                &mut refused,
                &input[..hidden - 1],
                &prepared,
                &route_weights,
                false,
            )
            .is_err());
        assert_eq!(refused, sentinel);
        let mut non_finite_weights = route_weights;
        non_finite_weights[1] = f32::NAN;
        assert!(program
            .execute(&mut refused, &input, &prepared, &non_finite_weights, false,)
            .is_err());
        assert_eq!(refused, sentinel);
        let mut invalid_offsets = prepared.clone();
        invalid_offsets.offsets_bytes[0][0] = u32::MAX;
        assert!(program
            .execute(
                &mut refused,
                &input,
                &invalid_offsets,
                &route_weights,
                false,
            )
            .is_err());
        assert_eq!(refused, sentinel);
        let mut invalid_block = prepared.clone();
        invalid_block.block = invalid_block.block.saturating_add(1);
        assert!(program
            .execute(&mut refused, &input, &invalid_block, &route_weights, false,)
            .is_err());
        assert_eq!(refused, sentinel);
    }

    #[test]
    #[ignore = "requires a non-CPU Vulkan device; run explicitly on an acceleration host"]
    fn live_packed_dispatch_covers_tail_words_odd_bf16_rows_and_a_two_dimensional_grid() {
        let cpu = Cpu::new();
        let vulkan = VulkanMatvec::new().unwrap();

        // Three odd-width rows cross uint-word boundaries, and the allocation ends with
        // two padding bytes. This catches a shader that assumes rows or storage end on a
        // four-byte boundary.
        let (rows, cols) = (3usize, 5usize);
        let bf16_values: Vec<f32> = (0..rows * cols)
            .map(|index| (index as f32 - 7.0) * 0.125)
            .collect();
        let bf16_bytes: Vec<u8> = bf16_values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        let bf16_weight = Weight {
            bytes: &bf16_bytes,
            kind: beholdd_format::GgmlType(BF16_KIND),
            rows,
            cols,
        };
        let input = [0.25f32, -0.5, 0.75, 1.0, -1.25];
        let mut expected = vec![0.0f32; rows];
        let mut actual = vec![0.0f32; rows];
        cpu.matvec(&mut expected, &input, &bf16_weight).unwrap();
        let prepared = vulkan.prepare(&bf16_weight).unwrap();
        vulkan.matvec(&mut actual, &input, &prepared).unwrap();
        assert_close(&expected, &actual, 1e-6);

        // One 34-byte Q8_0 block forces the final uint load to span the two bytes of
        // rounded allocation padding.
        let mut q8_bytes = Vec::with_capacity(Q8_0_BYTES);
        q8_bytes.extend_from_slice(&0x3800u16.to_le_bytes()); // f16 0.5
        q8_bytes.extend((-16i8..16).map(|value| value as u8));
        let q8_weight = Weight {
            bytes: &q8_bytes,
            kind: beholdd_format::GgmlType(Q8_0_KIND),
            rows: 1,
            cols: Q8_0_ELEMENTS,
        };
        let q8_input: Vec<f32> = (0..Q8_0_ELEMENTS)
            .map(|index| (index as f32 - 15.0) * 0.0625)
            .collect();
        let mut expected = [0.0f32];
        let mut actual = [0.0f32];
        cpu.matvec(&mut expected, &q8_input, &q8_weight).unwrap();
        let prepared = vulkan.prepare(&q8_weight).unwrap();
        vulkan.matvec(&mut actual, &q8_input, &prepared).unwrap();
        assert_close(&expected, &actual, 1e-6);

        // Force a smaller valid X limit so the shader must reconstruct row ids from a
        // genuine 64x3 dispatch. RADV exposes u32::MAX groups in X, which no practical
        // model matrix can naturally exceed.
        let grid_rows = 129usize;
        let grid_bytes: Vec<u8> = (0..grid_rows)
            .flat_map(|row| {
                let value = (row % 17) as f32 * 0.125 - 1.0;
                ((value.to_bits() >> 16) as u16).to_le_bytes()
            })
            .collect();
        let grid_weight = Weight {
            bytes: &grid_bytes,
            kind: beholdd_format::GgmlType(BF16_KIND),
            rows: grid_rows,
            cols: 1,
        };
        let mut expected = vec![0.0f32; grid_rows];
        let mut actual = vec![0.0f32; grid_rows];
        cpu.matvec(&mut expected, &[0.75], &grid_weight).unwrap();
        let prepared = vulkan.prepare(&grid_weight).unwrap();
        vulkan
            .run_with_group_limits(&mut actual, &[0.75], &prepared, 1.0, false, [64, 64, 1])
            .unwrap();
        assert_close(&expected, &actual, 1e-6);
    }
}
