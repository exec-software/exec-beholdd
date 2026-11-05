//! Residency, streaming and the fit contract.
//!
//! Owns: residency, bounded weight-range caches, pinning, single-worker prefetch, and
//! `FitPolicy` (Exact/Degrade/Refuse -- see docs/adr/0001-fit-contract.md). Prefetch uses
//! the same format-neutral checked-range seam as foreground streaming.
//!
//! Two measured rules govern the caches: LRU returns EXACTLY ZERO on a cyclic layer scan
//! and must be pinned-prefix there; model residency is demand-driven and LRU is correct
//! there. Getting them the wrong way round is its own bug.
//!
//! Does NOT own: model semantics.
//!
//! [`ResidentWeight`] keeps exact checkpoint bytes alive for an operation.
//! [`ResidentSource`] is the compatibility direct-read source and performs a positional
//! read for every request. [`FullyResidentSource`] eagerly packs every tensor payload
//! into one shared allocation, while [`StreamingSource`] adds the two deliberately
//! different M2 policies: a pinned prefix for cyclic scans and LRU for demand-driven
//! reads.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::ops::Range;
use std::str::FromStr;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use beholdd_format::{CheckpointId, RangeReader, WeightLocator, WeightRange};

mod fit;
mod process;
mod storage;

pub use fit::{
    ExactStreamingPlan, FitDecision, FitDegradation, FitEstimate, FitEstimateError, FitRefusal,
    FullyResidentPlan, RefusalReason, StreamingFitDecision,
};
pub use process::{
    available_memory, cgroup_memory, process_memory, AvailableMemory, AvailableMemoryError,
    CgroupMemory, CgroupMemoryError, ProcessMemory, ProcessMemoryError,
};
pub use storage::{
    check_checkpoint_storage, classify_checkpoint_storage, require_safe_checkpoint_storage,
    ShardStorage, StorageAssessment, StorageClass, StorageOverride, StorageRefusal,
    StorageSafetyError,
};

/// What beholdd may give up when the requested model does not fit.
///
/// This is the public form of the contract accepted in ADR-0001. In particular,
/// [`FitPolicy::Exact`] may trade latency for fit, but never changes model arithmetic.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FitPolicy {
    /// Never change what the model computes; stream exact bytes when it does not fit.
    #[default]
    Exact,
    /// Reduce fidelity-affecting resources, while identifying the change in the plan
    /// and the response.
    Degrade,
    /// Do not start; report the memory shortfall and the available alternatives.
    Refuse,
}

impl FitPolicy {
    /// Whether this policy permits an exact streaming plan when resident weights do not
    /// fit. Streaming changes latency, not arithmetic.
    pub const fn allows_streaming(self) -> bool {
        matches!(self, Self::Exact | Self::Degrade)
    }

    /// Whether this policy permits an explicitly reported semantic degradation.
    pub const fn allows_degradation(self) -> bool {
        matches!(self, Self::Degrade)
    }
}

impl fmt::Display for FitPolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Exact => "exact",
            Self::Degrade => "degrade",
            Self::Refuse => "refuse",
        })
    }
}

/// Error returned when a fit policy is not one of its three stable spellings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseFitPolicyError {
    value: String,
}

impl fmt::Display for ParseFitPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unknown fit policy {:?}; expected exact, degrade, or refuse",
            self.value
        )
    }
}

impl std::error::Error for ParseFitPolicyError {}

impl FromStr for FitPolicy {
    type Err = ParseFitPolicyError;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "exact" => Ok(Self::Exact),
            "degrade" => Ok(Self::Degrade),
            "refuse" => Ok(Self::Refuse),
            _ => Err(ParseFitPolicyError {
                value: value.to_owned(),
            }),
        }
    }
}

/// Expected reuse pattern for one weight request.
///
/// The distinction is semantic, not a performance hint that a source may reinterpret:
/// cyclic scans use a pinned prefix because LRU provably thrashes on them, while demand
/// reads use LRU because their reuse is data-dependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightAccess {
    /// Part of a stable scan repeated in the same order for every decode step.
    Cyclic,
    /// Selected from data at runtime, such as an embedding row or routed expert.
    Demand,
}

/// Supplies exact encoded weight ranges to a model.
///
/// This is the seam between model semantics and residency policy. A model names the
/// tensor bytes needed for an operation; a source decides whether those bytes come from
/// a resident allocation, a streaming cache, a prefetched buffer, or another strategy.
pub trait WeightSource: Send + Sync {
    /// Identity of the checkpoint whose tensor locators this source accepts.
    fn checkpoint_id(&self) -> CheckpointId;

    /// Materialise one checked half-open checkpoint byte range.
    ///
    /// `access` selects a retention policy only. It must never change the bytes returned.
    fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight>;

    /// Submit one ordered phase of speculative reads.
    ///
    /// Sources without a prefetch implementation deliberately accept this as a no-op.
    /// Callers may therefore expose one model-level schedule without branching on the
    /// concrete residency policy.
    fn prefetch_phase(&self, requests: &[WeightRequest]) -> Result<()> {
        for request in requests {
            if request.range.checkpoint_id() != self.checkpoint_id() {
                return Err(anyhow!(
                    "prefetch range belongs to checkpoint {:?}, not source checkpoint {:?}",
                    request.range.checkpoint_id(),
                    self.checkpoint_id()
                ));
            }
        }
        Ok(())
    }

    /// Cancel speculative work that has not become a foreground demand.
    ///
    /// Sources without a prefetch implementation deliberately accept this as a no-op.
    fn cancel_prefetch(&self) -> Result<()> {
        Ok(())
    }
}

/// One format-neutral request in an ordered prefetch phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightRequest {
    range: WeightRange,
    access: WeightAccess,
}

impl WeightRequest {
    /// Pair a checked checkpoint byte range with its eventual retention policy.
    pub const fn new(range: WeightRange, access: WeightAccess) -> Self {
        Self { range, access }
    }

    /// Checked byte range to fetch.
    pub const fn range(self) -> WeightRange {
        self.range
    }

    /// Retention policy to apply when the foreground consumes this request.
    pub const fn access(self) -> WeightAccess {
        self.access
    }
}

/// Correctness-first direct-read source that reads every requested checkpoint range.
///
/// Despite its compatibility name, this does not keep the full checkpoint resident: it
/// has no cache and performs a positional read for every request. Use
/// [`FullyResidentSource`] when every tensor payload must be loaded before execution.
#[derive(Clone, Copy)]
pub struct ResidentSource<'a> {
    checkpoint: &'a dyn RangeReader,
}

impl fmt::Debug for ResidentSource<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidentSource")
            .field("checkpoint_id", &self.checkpoint.checkpoint_id())
            .finish()
    }
}

impl<'a> ResidentSource<'a> {
    /// Borrow a checkpoint as a direct-read source.
    pub fn new(checkpoint: &'a dyn RangeReader) -> Self {
        Self { checkpoint }
    }
}

impl WeightSource for ResidentSource<'_> {
    fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint.checkpoint_id()
    }

    fn load_range(&self, range: &WeightRange, _access: WeightAccess) -> Result<ResidentWeight> {
        ResidentWeight::load_range(self.checkpoint, range)
    }
}

/// Every encoded tensor payload packed into one shared resident allocation.
///
/// Construction reads the complete checkpoint transactionally before returning. Tensor
/// payloads are ordered by shard and disk offset while they are read, then packed without
/// checkpoint headers or alignment padding. Consequently [`Self::resident_bytes`] is exactly
/// the sum of encoded tensor bytes used by the fit planner. Requests return immutable
/// zero-copy views into the packed allocation and perform no further checkpoint I/O.
pub struct FullyResidentSource {
    checkpoint_id: CheckpointId,
    bytes: Arc<Vec<u8>>,
    tensors: HashMap<WeightLocator, Range<usize>>,
    resident_bytes: u64,
}

impl fmt::Debug for FullyResidentSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FullyResidentSource")
            .field("checkpoint_id", &self.checkpoint_id)
            .field("resident_bytes", &self.resident_bytes)
            .field("tensors", &self.tensors.len())
            .finish()
    }
}

impl FullyResidentSource {
    /// Load every tensor payload from `checkpoint` into one packed allocation.
    pub fn load(checkpoint: &dyn RangeReader) -> Result<Self> {
        let mut tensors = Vec::new();
        for tensor in checkpoint.locators() {
            tensors.try_reserve(1).map_err(|error| {
                anyhow!("cannot allocate fully resident weight-locator list: {error}")
            })?;
            tensors.push(tensor);
        }
        tensors.sort_unstable_by_key(|tensor| (tensor.shard(), tensor.start(), tensor.end()));

        // Non-empty encoded weights must not alias or overlap. Besides making the
        // packed-byte forecast untrustworthy, accepting overlaps lets a tiny malformed
        // file amplify into an arbitrarily large resident allocation. Empty tensors are
        // valid and may share the same [N,N] extent, so coalesce those harmless aliases.
        for pair in tensors.windows(2) {
            let previous = pair[0];
            let current = pair[1];
            if previous == current && current.nbytes() != 0 {
                return Err(anyhow!(
                    "duplicate non-empty weight locator in shard {} at {}..{}",
                    current.shard(),
                    current.start(),
                    current.end()
                ));
            }
        }
        tensors.dedup();
        let mut previous_nonempty: Option<WeightLocator> = None;
        for tensor in &tensors {
            if tensor.nbytes() == 0 {
                continue;
            }
            if let Some(previous) = previous_nonempty {
                if previous.shard() == tensor.shard() && tensor.start() < previous.end() {
                    return Err(anyhow!(
                        "overlapping weight locators in shard {}: {}..{} and {}..{}",
                        tensor.shard(),
                        previous.start(),
                        previous.end(),
                        tensor.start(),
                        tensor.end()
                    ));
                }
            }
            previous_nonempty = Some(*tensor);
        }
        let tensor_count = tensors.len();

        // Validate the entire index before computing or allocating the packed model.
        // A malformed reader must not turn an eventual refusal into a model-sized
        // speculative allocation.
        for tensor in &tensors {
            checkpoint
                .validate_range(&tensor.whole_range())
                .with_context(|| {
                    format!(
                        "validating weight in shard {} at {}..{} before resident allocation",
                        tensor.shard(),
                        tensor.start(),
                        tensor.end()
                    )
                })?;
        }

        let resident_bytes = tensors.iter().try_fold(0u64, |total, tensor| {
            total
                .checked_add(tensor.nbytes())
                .context("fully resident checkpoint byte count overflows")
        })?;
        let resident_len = usize::try_from(resident_bytes).with_context(|| {
            format!(
                "fully resident checkpoint has {resident_bytes} encoded bytes, which does not fit in this address space"
            )
        })?;

        // Reserving explicitly turns an impossible model allocation into an ordinary
        // error rather than the abort or panic that a direct `vec![0; len]` may cause.
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(resident_len).map_err(|error| {
            anyhow!(
                "cannot allocate {resident_bytes} bytes for a fully resident checkpoint: {error}"
            )
        })?;
        bytes.resize(resident_len, 0);

        let mut index = HashMap::new();
        index
            .try_reserve(tensor_count)
            .map_err(|error| anyhow!("cannot allocate fully resident tensor index: {error}"))?;
        let mut packed_start = 0usize;
        for tensor in tensors {
            let tensor_len = usize::try_from(tensor.nbytes()).with_context(|| {
                format!(
                    "weight in shard {} at {} has {} encoded bytes, which does not fit in this address space",
                    tensor.shard(),
                    tensor.start(),
                    tensor.nbytes()
                )
            })?;
            let packed_end = packed_start
                .checked_add(tensor_len)
                .context("fully resident packed tensor range overflows")?;
            checkpoint
                .read_range_into(&tensor.whole_range(), &mut bytes[packed_start..packed_end])
                .with_context(|| {
                    format!(
                        "preloading weight from shard {} at {}..{} into resident memory",
                        tensor.shard(),
                        tensor.start(),
                        tensor.end()
                    )
                })?;

            if index.insert(tensor, packed_start..packed_end).is_some() {
                return Err(anyhow!(
                    "duplicate weight locator in shard {} at {}..{}",
                    tensor.shard(),
                    tensor.start(),
                    tensor.end()
                ));
            }
            packed_start = packed_end;
        }
        debug_assert_eq!(packed_start, resident_len);

        Ok(Self {
            checkpoint_id: checkpoint.checkpoint_id(),
            bytes: Arc::new(bytes),
            tensors: index,
            resident_bytes,
        })
    }

    /// Exact number of encoded checkpoint bytes retained by this source.
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }
}

impl WeightSource for FullyResidentSource {
    fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint_id
    }

    fn load_range(&self, range: &WeightRange, _access: WeightAccess) -> Result<ResidentWeight> {
        if range.checkpoint_id() != self.checkpoint_id {
            return Err(anyhow!(
                "weight range belongs to checkpoint {:?}, not source checkpoint {:?}",
                range.checkpoint_id(),
                self.checkpoint_id
            ));
        }
        let locator = range.locator();
        let packed = self.tensors.get(&locator).with_context(|| {
            format!(
                "weight locator in shard {} at {}..{} was not preloaded",
                locator.shard(),
                locator.start(),
                locator.end()
            )
        })?;
        let relative_start = usize::try_from(range.start() - locator.start())
            .context("fully resident tensor slice start does not fit in this address space")?;
        let relative_end = usize::try_from(range.end() - locator.start())
            .context("fully resident tensor slice end does not fit in this address space")?;
        let start = packed
            .start
            .checked_add(relative_start)
            .context("fully resident packed slice start overflows")?;
        let end = packed
            .start
            .checked_add(relative_end)
            .context("fully resident packed slice end overflows")?;
        if start < packed.start || end > packed.end {
            return Err(anyhow!(
                "fully resident packed slice {start}..{end} is outside tensor range {}..{}",
                packed.start,
                packed.end
            ));
        }
        ResidentWeight::shared_view(self.bytes.clone(), start..end)
    }
}

/// Independent byte limits for retained weights and speculative reads.
///
/// The sum is checked whenever the configuration is constructed or extended so every
/// reported total is exact. No cache allocates its budget up front. Queue depth bounds
/// attached/current-generation speculative-request metadata. Because an active system
/// read cannot be interrupted, cancellation may leave at most one detached worker-owned
/// record until that read returns; [`CacheStats::prefetch_detached`] exposes it.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheConfig {
    pinned_bytes: u64,
    demand_bytes: u64,
    prefetch_bytes: u64,
    prefetch_queue_depth: usize,
    prefetch_workers: usize,
}

impl CacheConfig {
    /// Configure the maximum retained bytes for cyclic and demand-driven reads.
    pub fn new(pinned_bytes: u64, demand_bytes: u64) -> Result<Self> {
        pinned_bytes
            .checked_add(demand_bytes)
            .context("combined weight-cache budget overflows u64")?;
        Ok(Self {
            pinned_bytes,
            demand_bytes,
            prefetch_bytes: 0,
            prefetch_queue_depth: 0,
            prefetch_workers: 0,
        })
    }

    /// Add bounded speculative-read storage, queue metadata and worker concurrency to
    /// this configuration.
    ///
    /// All three limits must be zero (disabled) or all three nonzero (enabled); mismatched
    /// limits are rejected so [`Self::total_bytes`] remains an exact actionable memory
    /// budget. `queue_depth` and `workers` are deliberately separate dials: queue depth
    /// bounds how many speculative requests may be outstanding (queued, reading or ready)
    /// at once, while `workers` bounds how many of those are physically read from storage
    /// at the same instant. A device that benefits from deep queuing (most NVMe: measure
    /// with `tools/probe_read_concurrency.py`) wants `workers` set from that measurement,
    /// not assumed -- this box's own NVMe delivered 2.2x the single-reader throughput at
    /// 8 concurrent readers and gained nothing past that.
    pub fn with_prefetch(
        mut self,
        prefetch_bytes: u64,
        queue_depth: usize,
        workers: usize,
    ) -> Result<Self> {
        let all_zero = prefetch_bytes == 0 && queue_depth == 0 && workers == 0;
        let all_nonzero = prefetch_bytes != 0 && queue_depth != 0 && workers != 0;
        if !all_zero && !all_nonzero {
            return Err(anyhow!(
                "prefetch byte, queue-depth and worker-count limits must all be zero or all be nonzero"
            ));
        }
        self.pinned_bytes
            .checked_add(self.demand_bytes)
            .and_then(|total| total.checked_add(prefetch_bytes))
            .context("combined weight-cache and prefetch budget overflows u64")?;
        self.prefetch_bytes = prefetch_bytes;
        self.prefetch_queue_depth = queue_depth;
        self.prefetch_workers = workers;
        Ok(self)
    }

    /// Maximum bytes retained by the pinned cyclic prefix.
    pub fn pinned_bytes(self) -> u64 {
        self.pinned_bytes
    }

    /// Maximum bytes retained by the demand LRU.
    pub fn demand_bytes(self) -> u64 {
        self.demand_bytes
    }

    /// Maximum bytes reserved by speculative reads and ready results.
    pub fn prefetch_bytes(self) -> u64 {
        self.prefetch_bytes
    }

    /// Maximum number of attached/current-generation speculative request records.
    pub fn prefetch_queue_depth(self) -> usize {
        self.prefetch_queue_depth
    }

    /// Number of background threads concurrently draining the prefetch queue.
    pub fn prefetch_workers(self) -> usize {
        self.prefetch_workers
    }

    /// Whether a worker should be started for this configuration.
    pub fn prefetch_enabled(self) -> bool {
        self.prefetch_bytes != 0 && self.prefetch_queue_depth != 0 && self.prefetch_workers != 0
    }

    /// Combined maximum retained bytes.
    pub fn total_bytes(self) -> u64 {
        // Construction established that this cannot overflow.
        self.pinned_bytes + self.demand_bytes + self.prefetch_bytes
    }
}

/// Snapshot of observable streaming-cache behaviour.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    requests: u64,
    hits: u64,
    misses: u64,
    foreground_demanded_bytes: u64,
    hit_bytes: u64,
    physical_reads: u64,
    read_bytes: u64,
    read_nanos: u64,
    physical_read_failures: u64,
    foreground_wait_nanos: u64,
    evictions: u64,
    pinned_bytes: u64,
    demand_bytes: u64,
    pinned_entries: usize,
    demand_entries: usize,
    prefetch_submitted: u64,
    prefetch_admitted: u64,
    prefetch_skipped: u64,
    prefetch_cancelled: u64,
    prefetch_useful: u64,
    prefetch_late: u64,
    prefetch_wasted: u64,
    prefetch_useful_bytes: u64,
    prefetch_wasted_bytes: u64,
    prefetch_physical_reads: u64,
    prefetch_physical_read_bytes: u64,
    prefetch_physical_read_nanos: u64,
    prefetch_physical_read_failures: u64,
    prefetch_hidden_nanos: u64,
    prefetch_outstanding: usize,
    prefetch_outstanding_peak: usize,
    prefetch_detached: usize,
    prefetch_detached_peak: usize,
    prefetch_queued: usize,
    prefetch_queued_peak: usize,
    prefetch_reading: usize,
    prefetch_reading_peak: usize,
    prefetch_reading_bytes: u64,
    prefetch_reading_bytes_peak: u64,
    prefetch_ready: usize,
    prefetch_ready_peak: usize,
    prefetch_ready_bytes: u64,
    prefetch_ready_bytes_peak: u64,
    prefetch_reserved_bytes_peak: u64,
}

impl CacheStats {
    /// Successful and failed lookup attempts, saturating at `u64::MAX`.
    pub fn requests(self) -> u64 {
        self.requests
    }

    /// Requests returned immediately from retained or ready-prefetched bytes.
    pub fn hits(self) -> u64 {
        self.hits
    }

    /// Validated foreground requests not returned immediately from retained or ready
    /// bytes. This includes callers that wait for an active read, so together with
    /// [`Self::hits`] it partitions [`Self::requests`].
    pub fn misses(self) -> u64 {
        self.misses
    }

    /// Bytes named by foreground requests, whether those requests succeed or fail.
    pub fn foreground_demanded_bytes(self) -> u64 {
        self.foreground_demanded_bytes
    }

    /// Foreground bytes returned immediately from a retained cache or ready prefetch.
    pub fn hit_bytes(self) -> u64 {
        self.hit_bytes
    }

    /// Completed positional checkpoint-read attempts from foreground and worker paths.
    pub fn physical_reads(self) -> u64 {
        self.physical_reads
    }

    /// Bytes successfully fetched from the checkpoint, including uncached reads.
    pub fn read_bytes(self) -> u64 {
        self.read_bytes
    }

    /// Positional checkpoint reads that returned an error.
    pub fn physical_read_failures(self) -> u64 {
        self.physical_read_failures
    }

    /// Wall time spent in all foreground and speculative physical reads.
    pub fn read_duration(self) -> Duration {
        Duration::from_nanos(self.read_nanos)
    }

    /// Bytes fetched per second during measured physical reads, or zero before any timed
    /// read completes.
    pub fn read_bytes_per_second(self) -> f64 {
        if self.read_nanos == 0 {
            0.0
        } else {
            self.read_bytes as f64 * 1_000_000_000.0 / self.read_nanos as f64
        }
    }

    /// Wall time foreground callers spent waiting for an already-active physical read.
    pub fn foreground_wait_duration(self) -> Duration {
        Duration::from_nanos(self.foreground_wait_nanos)
    }

    /// Demand entries removed by LRU replacement.
    pub fn evictions(self) -> u64 {
        self.evictions
    }

    /// Bytes retained in the pinned cyclic prefix.
    pub fn pinned_bytes(self) -> u64 {
        self.pinned_bytes
    }

    /// Bytes retained in the demand LRU.
    pub fn demand_bytes(self) -> u64 {
        self.demand_bytes
    }

    /// Bytes retained by both caches.
    pub fn cached_bytes(self) -> u64 {
        self.pinned_bytes + self.demand_bytes
    }

    /// Number of pinned cyclic entries.
    pub fn pinned_entries(self) -> usize {
        self.pinned_entries
    }

    /// Number of demand-LRU entries.
    pub fn demand_entries(self) -> usize {
        self.demand_entries
    }

    /// Validated requests offered to an enabled prefetch scheduler.
    pub fn prefetch_submitted(self) -> u64 {
        self.prefetch_submitted
    }

    /// Requests admitted to the bounded speculative queue.
    pub fn prefetch_admitted(self) -> u64 {
        self.prefetch_admitted
    }

    /// Submitted requests skipped because they were empty, oversized, duplicate,
    /// already retained, or beyond the outstanding-request limit.
    pub fn prefetch_skipped(self) -> u64 {
        self.prefetch_skipped
    }

    /// Speculative requests discarded by an explicit cancellation or source shutdown.
    pub fn prefetch_cancelled(self) -> u64 {
        self.prefetch_cancelled
    }

    /// Prefetch reads whose bytes were consumed by a foreground request.
    pub fn prefetch_useful(self) -> u64 {
        self.prefetch_useful
    }

    /// Speculative jobs first demanded while still queued or reading.
    pub fn prefetch_late(self) -> u64 {
        self.prefetch_late
    }

    /// Successful prefetch reads discarded without foreground consumption.
    pub fn prefetch_wasted(self) -> u64 {
        self.prefetch_wasted
    }

    /// Bytes delivered by useful prefetch reads.
    pub fn prefetch_useful_bytes(self) -> u64 {
        self.prefetch_useful_bytes
    }

    /// Successfully read speculative bytes discarded unused.
    pub fn prefetch_wasted_bytes(self) -> u64 {
        self.prefetch_wasted_bytes
    }

    /// Completed physical read attempts started by the speculative worker.
    pub fn prefetch_physical_reads(self) -> u64 {
        self.prefetch_physical_reads
    }

    /// Bytes successfully returned by speculative-worker physical reads.
    pub fn prefetch_physical_read_bytes(self) -> u64 {
        self.prefetch_physical_read_bytes
    }

    /// Wall time spent in speculative-worker physical reads.
    pub fn prefetch_physical_read_duration(self) -> Duration {
        Duration::from_nanos(self.prefetch_physical_read_nanos)
    }

    /// Speculative-worker physical reads that returned an error.
    pub fn prefetch_physical_read_failures(self) -> u64 {
        self.prefetch_physical_read_failures
    }

    /// Useful speculative-read time completed before the first foreground waiter.
    pub fn prefetch_hidden_duration(self) -> Duration {
        Duration::from_nanos(self.prefetch_hidden_nanos)
    }

    /// Attached/current-generation speculative request records in all states.
    pub fn prefetch_outstanding(self) -> usize {
        self.prefetch_outstanding
    }

    /// Largest observed outstanding speculative-request count.
    pub fn prefetch_outstanding_peak(self) -> usize {
        self.prefetch_outstanding_peak
    }

    /// Cancelled active reads detached from the current generation.
    ///
    /// The single-worker design makes this value at most one. Its bytes remain included
    /// in the reading and reserved-byte counters until the system read returns.
    pub fn prefetch_detached(self) -> usize {
        self.prefetch_detached
    }

    /// Largest observed detached active-read count; at most one.
    pub fn prefetch_detached_peak(self) -> usize {
        self.prefetch_detached_peak
    }

    /// Speculative requests currently waiting for the single worker.
    pub fn prefetch_queued(self) -> usize {
        self.prefetch_queued
    }

    /// Largest observed speculative queue length.
    pub fn prefetch_queued_peak(self) -> usize {
        self.prefetch_queued_peak
    }

    /// Speculative reads currently executing; zero or one for this source.
    pub fn prefetch_reading(self) -> usize {
        self.prefetch_reading
    }

    /// Largest observed count of executing speculative reads.
    pub fn prefetch_reading_peak(self) -> usize {
        self.prefetch_reading_peak
    }

    /// Bytes reserved by the currently executing speculative read.
    pub fn prefetch_reading_bytes(self) -> u64 {
        self.prefetch_reading_bytes
    }

    /// Largest observed executing speculative byte reservation.
    pub fn prefetch_reading_bytes_peak(self) -> u64 {
        self.prefetch_reading_bytes_peak
    }

    /// Completed speculative buffers waiting for foreground consumption.
    pub fn prefetch_ready(self) -> usize {
        self.prefetch_ready
    }

    /// Largest observed count of ready speculative buffers.
    pub fn prefetch_ready_peak(self) -> usize {
        self.prefetch_ready_peak
    }

    /// Bytes retained in completed speculative buffers.
    pub fn prefetch_ready_bytes(self) -> u64 {
        self.prefetch_ready_bytes
    }

    /// Largest observed ready speculative byte retention.
    pub fn prefetch_ready_bytes_peak(self) -> u64 {
        self.prefetch_ready_bytes_peak
    }

    /// Bytes reserved by executing and ready speculative reads.
    pub fn prefetch_reserved_bytes(self) -> u64 {
        self.prefetch_reading_bytes + self.prefetch_ready_bytes
    }

    /// Largest simultaneous reading-plus-ready speculative byte reservation.
    pub fn prefetch_reserved_bytes_peak(self) -> u64 {
        self.prefetch_reserved_bytes_peak
    }

    /// Fraction of requests returned immediately, or zero before the first request.
    pub fn hit_rate(self) -> f64 {
        if self.requests == 0 {
            0.0
        } else {
            self.hits as f64 / self.requests as f64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct CacheKey {
    shard: usize,
    start: u64,
    end: u64,
}

#[derive(Debug)]
struct DemandEntry {
    weight: ResidentWeight,
    last_used: u64,
}

#[derive(Debug, Default)]
struct CacheState {
    pinned: HashMap<CacheKey, ResidentWeight>,
    demand: HashMap<CacheKey, DemandEntry>,
    flights: HashMap<CacheKey, Arc<Flight>>,
    prefetch_queue: VecDeque<Arc<Flight>>,
    pinning_closed: bool,
    shutdown: bool,
    #[cfg(feature = "test-fault-injection")]
    terminate_workers_for_test: bool,
    clock: u64,
    next_generation: u64,
    requests: u64,
    hits: u64,
    misses: u64,
    foreground_demanded_bytes: u64,
    hit_bytes: u64,
    physical_reads: u64,
    read_bytes: u64,
    read_nanos: u64,
    physical_read_failures: u64,
    foreground_wait_nanos: u64,
    evictions: u64,
    pinned_bytes: u64,
    demand_bytes: u64,
    prefetch_submitted: u64,
    prefetch_admitted: u64,
    prefetch_skipped: u64,
    prefetch_cancelled: u64,
    prefetch_useful: u64,
    prefetch_late: u64,
    prefetch_wasted: u64,
    prefetch_useful_bytes: u64,
    prefetch_wasted_bytes: u64,
    prefetch_physical_reads: u64,
    prefetch_physical_read_bytes: u64,
    prefetch_physical_read_nanos: u64,
    prefetch_physical_read_failures: u64,
    prefetch_hidden_nanos: u64,
    prefetch_outstanding: usize,
    prefetch_outstanding_peak: usize,
    prefetch_detached: usize,
    prefetch_detached_peak: usize,
    prefetch_queued: usize,
    prefetch_queued_peak: usize,
    prefetch_reading: usize,
    prefetch_reading_peak: usize,
    prefetch_reading_bytes: u64,
    prefetch_reading_bytes_peak: u64,
    prefetch_ready: usize,
    prefetch_ready_peak: usize,
    prefetch_ready_bytes: u64,
    prefetch_ready_bytes_peak: u64,
    prefetch_reserved_bytes_peak: u64,
}

impl CacheState {
    fn stats(&self) -> CacheStats {
        CacheStats {
            requests: self.requests,
            hits: self.hits,
            misses: self.misses,
            foreground_demanded_bytes: self.foreground_demanded_bytes,
            hit_bytes: self.hit_bytes,
            physical_reads: self.physical_reads,
            read_bytes: self.read_bytes,
            read_nanos: self.read_nanos,
            physical_read_failures: self.physical_read_failures,
            foreground_wait_nanos: self.foreground_wait_nanos,
            evictions: self.evictions,
            pinned_bytes: self.pinned_bytes,
            demand_bytes: self.demand_bytes,
            pinned_entries: self.pinned.len(),
            demand_entries: self.demand.len(),
            prefetch_submitted: self.prefetch_submitted,
            prefetch_admitted: self.prefetch_admitted,
            prefetch_skipped: self.prefetch_skipped,
            prefetch_cancelled: self.prefetch_cancelled,
            prefetch_useful: self.prefetch_useful,
            prefetch_late: self.prefetch_late,
            prefetch_wasted: self.prefetch_wasted,
            prefetch_useful_bytes: self.prefetch_useful_bytes,
            prefetch_wasted_bytes: self.prefetch_wasted_bytes,
            prefetch_physical_reads: self.prefetch_physical_reads,
            prefetch_physical_read_bytes: self.prefetch_physical_read_bytes,
            prefetch_physical_read_nanos: self.prefetch_physical_read_nanos,
            prefetch_physical_read_failures: self.prefetch_physical_read_failures,
            prefetch_hidden_nanos: self.prefetch_hidden_nanos,
            prefetch_outstanding: self.prefetch_outstanding,
            prefetch_outstanding_peak: self.prefetch_outstanding_peak,
            prefetch_detached: self.prefetch_detached,
            prefetch_detached_peak: self.prefetch_detached_peak,
            prefetch_queued: self.prefetch_queued,
            prefetch_queued_peak: self.prefetch_queued_peak,
            prefetch_reading: self.prefetch_reading,
            prefetch_reading_peak: self.prefetch_reading_peak,
            prefetch_reading_bytes: self.prefetch_reading_bytes,
            prefetch_reading_bytes_peak: self.prefetch_reading_bytes_peak,
            prefetch_ready: self.prefetch_ready,
            prefetch_ready_peak: self.prefetch_ready_peak,
            prefetch_ready_bytes: self.prefetch_ready_bytes,
            prefetch_ready_bytes_peak: self.prefetch_ready_bytes_peak,
            prefetch_reserved_bytes_peak: self.prefetch_reserved_bytes_peak,
        }
    }

    fn request(&mut self, nbytes: u64) {
        self.requests = self.requests.saturating_add(1);
        self.foreground_demanded_bytes = self.foreground_demanded_bytes.saturating_add(nbytes);
    }

    fn hit(&mut self, nbytes: u64) {
        self.hits = self.hits.saturating_add(1);
        self.hit_bytes = self.hit_bytes.saturating_add(nbytes);
    }

    fn miss(&mut self) {
        self.misses = self.misses.saturating_add(1);
    }

    fn next_stamp(&mut self) -> u64 {
        self.clock = self.clock.saturating_add(1);
        self.clock
    }

    fn record_physical_read(&mut self, result_bytes: Option<u64>, nanos: u64, prefetch: bool) {
        self.physical_reads = self.physical_reads.saturating_add(1);
        self.read_nanos = self.read_nanos.saturating_add(nanos);
        if let Some(bytes) = result_bytes {
            self.read_bytes = self.read_bytes.saturating_add(bytes);
        } else {
            self.physical_read_failures = self.physical_read_failures.saturating_add(1);
        }
        if prefetch {
            self.prefetch_physical_reads = self.prefetch_physical_reads.saturating_add(1);
            self.prefetch_physical_read_nanos =
                self.prefetch_physical_read_nanos.saturating_add(nanos);
            if let Some(bytes) = result_bytes {
                self.prefetch_physical_read_bytes =
                    self.prefetch_physical_read_bytes.saturating_add(bytes);
            } else {
                self.prefetch_physical_read_failures =
                    self.prefetch_physical_read_failures.saturating_add(1);
            }
        }
    }

    fn reserve_prefetch_read(&mut self, nbytes: u64) {
        self.prefetch_queued -= 1;
        self.prefetch_reading += 1;
        self.prefetch_reading_bytes += nbytes;
        self.prefetch_reading_peak = self.prefetch_reading_peak.max(self.prefetch_reading);
        self.prefetch_reading_bytes_peak = self
            .prefetch_reading_bytes_peak
            .max(self.prefetch_reading_bytes);
        self.prefetch_reserved_bytes_peak = self
            .prefetch_reserved_bytes_peak
            .max(self.prefetch_reading_bytes + self.prefetch_ready_bytes);
    }

    fn finish_prefetch_reading(&mut self, nbytes: u64) {
        self.prefetch_reading -= 1;
        self.prefetch_reading_bytes -= nbytes;
    }

    fn forget_prefetch(&mut self) {
        self.prefetch_outstanding -= 1;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReadOrigin {
    Foreground,
    Prefetch,
}

#[derive(Debug)]
enum FlightState {
    Queued,
    Reading {
        origin: ReadOrigin,
        demand_access: Option<WeightAccess>,
        demanded_at: Option<Instant>,
        cancel_requested: bool,
    },
    Ready {
        weight: ResidentWeight,
        prefetched: bool,
        consumed: bool,
        read_nanos: u64,
    },
    Failed {
        message: Arc<str>,
        prefetched: bool,
    },
    Cancelled,
}

#[derive(Debug)]
struct Flight {
    key: CacheKey,
    generation: u64,
    range: WeightRange,
    state: Mutex<FlightState>,
    ready: Condvar,
}

impl Flight {
    fn queued(key: CacheKey, generation: u64, range: WeightRange) -> Self {
        Self {
            key,
            generation,
            range,
            state: Mutex::new(FlightState::Queued),
            ready: Condvar::new(),
        }
    }

    fn foreground(
        key: CacheKey,
        generation: u64,
        range: WeightRange,
        access: WeightAccess,
    ) -> Self {
        Self {
            key,
            generation,
            range,
            state: Mutex::new(FlightState::Reading {
                origin: ReadOrigin::Foreground,
                demand_access: Some(access),
                demanded_at: None,
                cancel_requested: false,
            }),
            ready: Condvar::new(),
        }
    }
}

#[derive(Debug, Default)]
struct SharedState {
    cache: Mutex<CacheState>,
    worker_wake: Condvar,
}

struct PhysicalRead {
    result: Result<ResidentWeight>,
    started: Instant,
    nanos: u64,
}

fn physical_read(checkpoint: &dyn RangeReader, range: &WeightRange) -> PhysicalRead {
    let started = Instant::now();
    let result = ResidentWeight::load_range(checkpoint, range);
    PhysicalRead {
        result,
        started,
        nanos: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
    }
}

fn hidden_prefetch_nanos(
    read_started: Instant,
    read_nanos: u64,
    demanded_at: Option<Instant>,
) -> u64 {
    demanded_at
        .map(|demanded_at| {
            u64::try_from(
                demanded_at
                    .saturating_duration_since(read_started)
                    .as_nanos(),
            )
            .unwrap_or(u64::MAX)
        })
        .unwrap_or(read_nanos)
        .min(read_nanos)
}

fn same_flight(state: &CacheState, flight: &Arc<Flight>) -> bool {
    state.flights.get(&flight.key).is_some_and(|current| {
        Arc::ptr_eq(current, flight) && current.generation == flight.generation
    })
}

fn cached_weight(
    state: &mut CacheState,
    key: CacheKey,
    access: WeightAccess,
) -> Option<ResidentWeight> {
    match access {
        WeightAccess::Cyclic => state.pinned.get(&key).cloned(),
        WeightAccess::Demand => {
            if !state.demand.contains_key(&key) {
                return None;
            }
            let stamp = state.next_stamp();
            let entry = state
                .demand
                .get_mut(&key)
                .expect("demand key was checked while the cache lock is held");
            entry.last_used = stamp;
            Some(entry.weight.clone())
        }
    }
}

fn retain_weight(
    config: CacheConfig,
    state: &mut CacheState,
    key: CacheKey,
    access: WeightAccess,
    nbytes: u64,
    weight: &ResidentWeight,
) {
    if nbytes == 0 {
        return;
    }
    match access {
        WeightAccess::Cyclic => {
            if state.pinning_closed || state.pinned.contains_key(&key) {
                return;
            }
            let Some(projected) = state.pinned_bytes.checked_add(nbytes) else {
                state.pinning_closed = true;
                return;
            };
            if projected > config.pinned_bytes {
                state.pinning_closed = true;
                return;
            }
            state.pinned.insert(key, weight.clone());
            state.pinned_bytes = projected;
        }
        WeightAccess::Demand => {
            if nbytes > config.demand_bytes {
                return;
            }
            while state.demand_bytes > config.demand_bytes - nbytes {
                let victim = state
                    .demand
                    .iter()
                    .min_by_key(|(candidate, entry)| (entry.last_used, **candidate))
                    .map(|(candidate, _)| *candidate)
                    .expect("a non-empty demand cache has an LRU entry");
                let removed = state
                    .demand
                    .remove(&victim)
                    .expect("the selected demand-cache victim still exists");
                state.demand_bytes -= removed.weight.len() as u64;
                state.evictions = state.evictions.saturating_add(1);
            }
            let stamp = state.next_stamp();
            if let Some(previous) = state.demand.insert(
                key,
                DemandEntry {
                    weight: weight.clone(),
                    last_used: stamp,
                },
            ) {
                state.demand_bytes -= previous.weight.len() as u64;
            }
            state.demand_bytes += nbytes;
        }
    }
}

fn prefetch_worker(
    checkpoint: Arc<dyn RangeReader>,
    config: CacheConfig,
    shared: Arc<SharedState>,
) {
    loop {
        let flight = {
            let mut state = match shared.cache.lock() {
                Ok(state) => state,
                Err(_) => return,
            };
            loop {
                #[cfg(feature = "test-fault-injection")]
                if state.terminate_workers_for_test {
                    return;
                }
                if state.shutdown {
                    return;
                }
                let Some(candidate) = state.prefetch_queue.front().cloned() else {
                    state = match shared.worker_wake.wait(state) {
                        Ok(state) => state,
                        Err(_) => return,
                    };
                    continue;
                };
                let reserved = state
                    .prefetch_reading_bytes
                    .saturating_add(state.prefetch_ready_bytes);
                if candidate.range.nbytes() > config.prefetch_bytes.saturating_sub(reserved) {
                    state = match shared.worker_wake.wait(state) {
                        Ok(state) => state,
                        Err(_) => return,
                    };
                    continue;
                }

                state.prefetch_queue.pop_front();
                let mut flight_state = match candidate.state.lock() {
                    Ok(state) => state,
                    Err(_) => return,
                };
                if !matches!(*flight_state, FlightState::Queued) {
                    state.prefetch_queued -= 1;
                    continue;
                }
                *flight_state = FlightState::Reading {
                    origin: ReadOrigin::Prefetch,
                    demand_access: None,
                    demanded_at: None,
                    cancel_requested: false,
                };
                state.reserve_prefetch_read(candidate.range.nbytes());
                drop(flight_state);
                break candidate;
            }
        };

        let physical = physical_read(checkpoint.as_ref(), &flight.range);
        let mut state = match shared.cache.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
        let mut flight_state = match flight.state.lock() {
            Ok(state) => state,
            Err(_) => return,
        };
        let (demand_access, demanded_at, cancel_requested) = match *flight_state {
            FlightState::Reading {
                origin: ReadOrigin::Prefetch,
                demand_access,
                demanded_at,
                cancel_requested,
                ..
            } => (demand_access, demanded_at, cancel_requested),
            _ => return,
        };
        let result_bytes = physical
            .result
            .as_ref()
            .ok()
            .map(|weight| weight.len() as u64);
        state.record_physical_read(result_bytes, physical.nanos, true);
        state.finish_prefetch_reading(flight.range.nbytes());

        if (cancel_requested || state.shutdown) && demand_access.is_none() {
            if cancel_requested {
                state.prefetch_detached -= 1;
            }
            state.prefetch_cancelled = state.prefetch_cancelled.saturating_add(1);
            if let Some(bytes) = result_bytes {
                state.prefetch_wasted = state.prefetch_wasted.saturating_add(1);
                state.prefetch_wasted_bytes = state.prefetch_wasted_bytes.saturating_add(bytes);
            }
            if same_flight(&state, &flight) {
                state.flights.remove(&flight.key);
                state.forget_prefetch();
            }
            *flight_state = FlightState::Cancelled;
        } else {
            match physical.result {
                Ok(weight) => {
                    if let Some(access) = demand_access {
                        if same_flight(&state, &flight) {
                            state.flights.remove(&flight.key);
                            state.forget_prefetch();
                        }
                        state.prefetch_useful = state.prefetch_useful.saturating_add(1);
                        state.prefetch_useful_bytes = state
                            .prefetch_useful_bytes
                            .saturating_add(weight.len() as u64);
                        state.prefetch_hidden_nanos =
                            state
                                .prefetch_hidden_nanos
                                .saturating_add(hidden_prefetch_nanos(
                                    physical.started,
                                    physical.nanos,
                                    demanded_at,
                                ));
                        retain_weight(
                            config,
                            &mut state,
                            flight.key,
                            access,
                            flight.range.nbytes(),
                            &weight,
                        );
                        *flight_state = FlightState::Ready {
                            weight,
                            prefetched: true,
                            consumed: true,
                            read_nanos: physical.nanos,
                        };
                    } else {
                        state.prefetch_ready += 1;
                        state.prefetch_ready_bytes += flight.range.nbytes();
                        state.prefetch_ready_peak =
                            state.prefetch_ready_peak.max(state.prefetch_ready);
                        state.prefetch_ready_bytes_peak = state
                            .prefetch_ready_bytes_peak
                            .max(state.prefetch_ready_bytes);
                        state.prefetch_reserved_bytes_peak = state
                            .prefetch_reserved_bytes_peak
                            .max(state.prefetch_reading_bytes + state.prefetch_ready_bytes);
                        *flight_state = FlightState::Ready {
                            weight,
                            prefetched: true,
                            consumed: false,
                            read_nanos: physical.nanos,
                        };
                    }
                }
                Err(error) => {
                    let message: Arc<str> = format!("{error:#}").into();
                    if demand_access.is_some() && same_flight(&state, &flight) {
                        state.flights.remove(&flight.key);
                        state.forget_prefetch();
                    }
                    *flight_state = FlightState::Failed {
                        message,
                        prefetched: true,
                    };
                }
            }
        }
        flight.ready.notify_all();
        shared.worker_wake.notify_all();
    }
}

/// Exact positional-read source with a bounded two-policy cache and optional prefetch.
///
/// A cyclic request fills the pinned cache only while it extends the initial request
/// prefix. Once the next unique request does not fit, pinning closes permanently; later
/// small reads cannot turn the cache into a misleading non-prefix sample. Demand reads
/// use exact LRU replacement. Entries are shared with returned [`ResidentWeight`] values,
/// so a cache hit performs no checkpoint read and no byte copy.
pub struct StreamingSource {
    checkpoint: Arc<dyn RangeReader>,
    config: CacheConfig,
    shared: Arc<SharedState>,
    workers: Vec<JoinHandle<()>>,
}

impl fmt::Debug for StreamingSource {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamingSource")
            .field("checkpoint_id", &self.checkpoint.checkpoint_id())
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

impl StreamingSource {
    /// Create a streaming source with explicit pinned and demand-cache limits.
    pub fn new(checkpoint: Arc<dyn RangeReader>, config: CacheConfig) -> Result<Self> {
        let shared = Arc::new(SharedState::default());
        let mut workers = Vec::new();
        if config.prefetch_enabled() {
            workers
                .try_reserve_exact(config.prefetch_workers())
                .map_err(|error| {
                    anyhow!("cannot allocate prefetch worker handle table: {error}")
                })?;
            for index in 0..config.prefetch_workers() {
                let worker_checkpoint = checkpoint.clone();
                let worker_shared = shared.clone();
                workers.push(
                    thread::Builder::new()
                        .name(format!("beholdd-prefetch-{index}"))
                        .spawn(move || prefetch_worker(worker_checkpoint, config, worker_shared))
                        .context("spawning bounded checkpoint-prefetch worker")?,
                );
            }
        }
        Ok(Self {
            checkpoint,
            config,
            shared,
            workers,
        })
    }

    /// Configured cache limits.
    pub fn config(&self) -> CacheConfig {
        self.config
    }

    /// Snapshot counters and currently retained bytes.
    pub fn stats(&self) -> Result<CacheStats> {
        let state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        Ok(state.stats())
    }

    /// Probe whether this source can still serve requests.
    ///
    /// This checks the live cache state and every configured prefetch worker rather than
    /// trusting the fact that construction once succeeded. A source with no prefetch
    /// workers is healthy when its cache lock is usable; a configured worker that has
    /// exited is a failed source even if foreground reads could otherwise continue.
    pub fn probe(&self) -> Result<()> {
        let state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        if state.shutdown {
            return Err(anyhow!("streaming weight source is shutting down"));
        }
        drop(state);

        let stopped = self
            .workers
            .iter()
            .filter(|worker| worker.is_finished())
            .count();
        if stopped != 0 {
            return Err(anyhow!(
                "{stopped} of {} checkpoint-prefetch worker(s) exited",
                self.workers.len()
            ));
        }
        Ok(())
    }

    /// Stop every real prefetch worker while leaving this source published for probing.
    ///
    /// This exists only under the non-default `test-fault-injection` feature. It is the
    /// narrow fault seam for the M3.3 registry gate: production builds have no callable
    /// worker-stop control, and HTTP never enables or exposes this feature. The method
    /// uses a separate test-only flag rather than the source's normal `shutdown` state,
    /// then waits until every worker handle reports completion. A successful return
    /// therefore proves that a subsequent failed liveness probe detects the exited real
    /// worker rather than merely detecting a requested source shutdown.
    #[cfg(feature = "test-fault-injection")]
    #[doc(hidden)]
    pub fn terminate_prefetch_workers_for_test(&self, timeout: Duration) -> Result<usize> {
        if self.workers.is_empty() {
            return Err(anyhow!(
                "test fault injection requires at least one checkpoint-prefetch worker"
            ));
        }
        let mut state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        state.terminate_workers_for_test = true;
        self.shared.worker_wake.notify_all();
        drop(state);

        let deadline = Instant::now()
            .checked_add(timeout)
            .context("test worker-termination deadline overflows")?;
        while self.workers.iter().any(|worker| !worker.is_finished()) {
            let now = Instant::now();
            if now >= deadline {
                let running = self
                    .workers
                    .iter()
                    .filter(|worker| !worker.is_finished())
                    .count();
                return Err(anyhow!(
                    "{running} of {} checkpoint-prefetch worker(s) did not exit within {timeout:?}",
                    self.workers.len()
                ));
            }
            thread::sleep(Duration::from_millis(1).min(deadline - now));
        }
        Ok(self.workers.len())
    }

    /// Cancel speculative work, wait for the worker's current positional read, and
    /// return a final telemetry snapshot.
    ///
    /// A positional read already in the kernel cannot be interrupted. This consuming
    /// operation therefore may block until that one read returns, but guarantees that
    /// its physical-I/O, cancellation, wasted-byte and reservation counters are present
    /// in the returned snapshot. Callers measuring model execution should stop their
    /// decode timer before invoking this drain.
    pub fn finish(mut self) -> Result<CacheStats> {
        let cancellation = self
            .cancel_prefetch()
            .context("cancelling speculative reads before final telemetry");
        let shutdown = self.stop_workers();
        cancellation?;
        shutdown?;
        let stats = self
            .stats()
            .context("sampling final streaming telemetry after worker shutdown")?;
        if stats.prefetch_outstanding() != 0
            || stats.prefetch_detached() != 0
            || stats.prefetch_queued() != 0
            || stats.prefetch_reading() != 0
            || stats.prefetch_ready() != 0
            || stats.prefetch_reserved_bytes() != 0
        {
            return Err(anyhow!(
                "prefetch worker stopped with unsettled telemetry: {} attached, {} detached, {} queued, {} reading, {} ready, {} reserved bytes",
                stats.prefetch_outstanding(),
                stats.prefetch_detached(),
                stats.prefetch_queued(),
                stats.prefetch_reading(),
                stats.prefetch_ready(),
                stats.prefetch_reserved_bytes()
            ));
        }
        Ok(stats)
    }

    fn stop_workers(&mut self) -> Result<()> {
        let (mut state, lock_poisoned) = match self.shared.cache.lock() {
            Ok(state) => (state, false),
            Err(poisoned) => (poisoned.into_inner(), true),
        };
        state.shutdown = true;
        self.shared.worker_wake.notify_all();
        drop(state);

        let panicked = self
            .workers
            .drain(..)
            .map(|worker| worker.join().is_err())
            .filter(|&panicked| panicked)
            .count();
        match (lock_poisoned, panicked) {
            (false, 0) => Ok(()),
            (true, 0) => Err(anyhow!("streaming weight-cache lock is poisoned")),
            (false, n) => Err(anyhow!("{n} checkpoint-prefetch worker(s) panicked")),
            (true, n) => Err(anyhow!(
                "streaming weight-cache lock is poisoned and {n} checkpoint-prefetch worker(s) panicked"
            )),
        }
    }

    #[cfg(test)]
    fn wait_for_stats_for_test(
        &self,
        predicate: impl Fn(CacheStats) -> bool,
    ) -> Result<CacheStats> {
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        loop {
            let stats = state.stats();
            if predicate(stats) {
                return Ok(stats);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            let (next, timeout) = self
                .shared
                .worker_wake
                .wait_timeout(state, remaining)
                .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
            state = next;
            if timeout.timed_out() && !predicate(state.stats()) {
                return Err(anyhow!("timed out waiting for streaming test state"));
            }
        }
    }

    fn finish_foreground_read(
        &self,
        flight: &Arc<Flight>,
        access: WeightAccess,
        physical: &PhysicalRead,
    ) -> Result<()> {
        let mut state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        let mut flight_state = flight
            .state
            .lock()
            .map_err(|_| anyhow!("streaming weight-flight lock is poisoned"))?;
        state.record_physical_read(
            physical
                .result
                .as_ref()
                .ok()
                .map(|weight| weight.len() as u64),
            physical.nanos,
            false,
        );
        if same_flight(&state, flight) {
            state.flights.remove(&flight.key);
        }
        match &physical.result {
            Ok(weight) => {
                retain_weight(
                    self.config,
                    &mut state,
                    flight.key,
                    access,
                    flight.range.nbytes(),
                    weight,
                );
                *flight_state = FlightState::Ready {
                    weight: weight.clone(),
                    prefetched: false,
                    consumed: true,
                    read_nanos: physical.nanos,
                };
            }
            Err(error) => {
                *flight_state = FlightState::Failed {
                    message: format!("{error:#}").into(),
                    prefetched: false,
                };
            }
        }
        flight.ready.notify_all();
        self.shared.worker_wake.notify_all();
        Ok(())
    }

    fn cancel_prefetch_locked(&self, state: &mut CacheState) -> Result<()> {
        let mut flights = Vec::new();
        flights
            .try_reserve(state.prefetch_outstanding)
            .map_err(|error| anyhow!("cannot allocate prefetch-cancellation list: {error}"))?;
        for flight in state.flights.values() {
            let flight_state = flight
                .state
                .lock()
                .map_err(|_| anyhow!("streaming weight-flight lock is poisoned"))?;
            let speculative = matches!(
                *flight_state,
                FlightState::Queued
                    | FlightState::Reading {
                        origin: ReadOrigin::Prefetch,
                        ..
                    }
                    | FlightState::Ready {
                        prefetched: true,
                        ..
                    }
                    | FlightState::Failed {
                        prefetched: true,
                        ..
                    }
            );
            drop(flight_state);
            if speculative {
                flights.push(flight.clone());
            }
        }

        for flight in flights {
            let mut flight_state = flight
                .state
                .lock()
                .map_err(|_| anyhow!("streaming weight-flight lock is poisoned"))?;
            match &mut *flight_state {
                FlightState::Queued => {
                    state
                        .prefetch_queue
                        .retain(|queued| !Arc::ptr_eq(queued, &flight));
                    state.prefetch_queued -= 1;
                    if same_flight(state, &flight) {
                        state.flights.remove(&flight.key);
                        state.forget_prefetch();
                    }
                    state.prefetch_cancelled = state.prefetch_cancelled.saturating_add(1);
                    *flight_state = FlightState::Cancelled;
                    flight.ready.notify_all();
                }
                FlightState::Reading {
                    origin: ReadOrigin::Prefetch,
                    demand_access,
                    cancel_requested,
                    ..
                } if demand_access.is_none() => {
                    *cancel_requested = true;
                    if same_flight(state, &flight) {
                        state.flights.remove(&flight.key);
                        state.forget_prefetch();
                        state.prefetch_detached += 1;
                        state.prefetch_detached_peak =
                            state.prefetch_detached_peak.max(state.prefetch_detached);
                    }
                }
                FlightState::Ready {
                    weight,
                    prefetched: true,
                    consumed: false,
                    ..
                } => {
                    let nbytes = weight.len() as u64;
                    state.prefetch_ready -= 1;
                    state.prefetch_ready_bytes -= nbytes;
                    if same_flight(state, &flight) {
                        state.flights.remove(&flight.key);
                        state.forget_prefetch();
                    }
                    state.prefetch_cancelled = state.prefetch_cancelled.saturating_add(1);
                    state.prefetch_wasted = state.prefetch_wasted.saturating_add(1);
                    state.prefetch_wasted_bytes =
                        state.prefetch_wasted_bytes.saturating_add(nbytes);
                    *flight_state = FlightState::Cancelled;
                    flight.ready.notify_all();
                }
                FlightState::Failed {
                    prefetched: true, ..
                } => {
                    if same_flight(state, &flight) {
                        state.flights.remove(&flight.key);
                        state.forget_prefetch();
                    }
                    state.prefetch_cancelled = state.prefetch_cancelled.saturating_add(1);
                    *flight_state = FlightState::Cancelled;
                    flight.ready.notify_all();
                }
                _ => {}
            }
        }
        self.shared.worker_wake.notify_all();
        Ok(())
    }
}

impl WeightSource for StreamingSource {
    fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint.checkpoint_id()
    }

    fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
        self.checkpoint
            .validate_range(range)
            .context("validating streaming weight range before cache lookup")?;
        let nbytes = range.nbytes();
        let key = CacheKey {
            shard: range.shard(),
            start: range.start(),
            end: range.end(),
        };
        {
            let mut state = self
                .shared
                .cache
                .lock()
                .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
            state.request(nbytes);
            self.shared.worker_wake.notify_all();
        }

        enum Action {
            Read(Arc<Flight>),
            Wait(Arc<Flight>),
        }

        loop {
            let action = {
                let mut state = self
                    .shared
                    .cache
                    .lock()
                    .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
                if let Some(weight) = cached_weight(&mut state, key, access) {
                    state.hit(nbytes);
                    return Ok(weight);
                }

                if let Some(flight) = state.flights.get(&key).cloned() {
                    let mut flight_state = flight
                        .state
                        .lock()
                        .map_err(|_| anyhow!("streaming weight-flight lock is poisoned"))?;
                    match &mut *flight_state {
                        FlightState::Queued => {
                            state.miss();
                            state.prefetch_late = state.prefetch_late.saturating_add(1);
                            state
                                .prefetch_queue
                                .retain(|queued| !Arc::ptr_eq(queued, &flight));
                            state.prefetch_queued -= 1;
                            state.forget_prefetch();
                            *flight_state = FlightState::Reading {
                                origin: ReadOrigin::Foreground,
                                demand_access: Some(access),
                                demanded_at: None,
                                cancel_requested: false,
                            };
                            self.shared.worker_wake.notify_all();
                            Action::Read(flight.clone())
                        }
                        FlightState::Reading {
                            origin,
                            demand_access,
                            demanded_at,
                            ..
                        } => {
                            state.miss();
                            if *origin == ReadOrigin::Prefetch && demand_access.is_none() {
                                state.prefetch_late = state.prefetch_late.saturating_add(1);
                                *demand_access = Some(access);
                                *demanded_at = Some(Instant::now());
                                self.shared.worker_wake.notify_all();
                            }
                            Action::Wait(flight.clone())
                        }
                        FlightState::Ready {
                            weight,
                            prefetched: true,
                            consumed,
                            read_nanos,
                        } if !*consumed => {
                            let weight = weight.clone();
                            *consumed = true;
                            state.prefetch_ready -= 1;
                            state.prefetch_ready_bytes -= nbytes;
                            state.forget_prefetch();
                            state.flights.remove(&key);
                            state.prefetch_useful = state.prefetch_useful.saturating_add(1);
                            state.prefetch_useful_bytes =
                                state.prefetch_useful_bytes.saturating_add(nbytes);
                            state.prefetch_hidden_nanos =
                                state.prefetch_hidden_nanos.saturating_add(*read_nanos);
                            state.hit(nbytes);
                            retain_weight(self.config, &mut state, key, access, nbytes, &weight);
                            self.shared.worker_wake.notify_all();
                            return Ok(weight);
                        }
                        FlightState::Failed {
                            message,
                            prefetched,
                        } => {
                            let message = message.clone();
                            if *prefetched {
                                state.forget_prefetch();
                            }
                            state.flights.remove(&key);
                            state.miss();
                            return Err(anyhow!("checkpoint prefetch failed: {message}"));
                        }
                        FlightState::Cancelled => {
                            state.flights.remove(&key);
                            continue;
                        }
                        FlightState::Ready { weight, .. } => {
                            let weight = weight.clone();
                            state.hit(nbytes);
                            return Ok(weight);
                        }
                    }
                } else {
                    state.miss();
                    let generation = state
                        .next_generation
                        .checked_add(1)
                        .context("streaming weight-flight generation overflows")?;
                    state.next_generation = generation;
                    state
                        .flights
                        .try_reserve(1)
                        .map_err(|error| anyhow!("cannot allocate weight-flight index: {error}"))?;
                    let flight = Arc::new(Flight::foreground(key, generation, *range, access));
                    state.flights.insert(key, flight.clone());
                    Action::Read(flight)
                }
            };

            match action {
                Action::Read(flight) => {
                    let physical = physical_read(self.checkpoint.as_ref(), range);
                    self.finish_foreground_read(&flight, access, &physical)?;
                    return physical.result;
                }
                Action::Wait(flight) => {
                    let started = Instant::now();
                    let mut flight_state = flight
                        .state
                        .lock()
                        .map_err(|_| anyhow!("streaming weight-flight lock is poisoned"))?;
                    while matches!(
                        *flight_state,
                        FlightState::Queued | FlightState::Reading { .. }
                    ) {
                        flight_state = flight
                            .ready
                            .wait(flight_state)
                            .map_err(|_| anyhow!("streaming weight-flight lock is poisoned"))?;
                    }
                    let waited = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
                    let result = match &*flight_state {
                        FlightState::Ready { weight, .. } => Ok(weight.clone()),
                        FlightState::Failed { message, .. } => {
                            Err(anyhow!("checkpoint read failed: {message}"))
                        }
                        FlightState::Cancelled => Err(anyhow!(
                            "checkpoint read was cancelled before satisfying foreground demand"
                        )),
                        FlightState::Queued | FlightState::Reading { .. } => unreachable!(),
                    };
                    drop(flight_state);
                    let mut state = self
                        .shared
                        .cache
                        .lock()
                        .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
                    state.foreground_wait_nanos =
                        state.foreground_wait_nanos.saturating_add(waited);
                    return result;
                }
            }
        }
    }

    fn prefetch_phase(&self, requests: &[WeightRequest]) -> Result<()> {
        for request in requests {
            self.checkpoint
                .validate_range(&request.range)
                .context("validating weight range before prefetch submission")?;
        }
        if !self.config.prefetch_enabled() {
            return Ok(());
        }

        let mut state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        let available_slots = self
            .config
            .prefetch_queue_depth
            .saturating_sub(state.prefetch_outstanding);
        let mut planned: Vec<(CacheKey, u64, WeightRange)> = Vec::new();
        planned
            .try_reserve(requests.len().min(available_slots))
            .map_err(|error| anyhow!("cannot allocate prefetch admission plan: {error}"))?;
        let mut generation = state.next_generation;
        for request in requests {
            let range = request.range;
            let key = CacheKey {
                shard: range.shard(),
                start: range.start(),
                end: range.end(),
            };
            let retained = match request.access {
                WeightAccess::Cyclic => state.pinned.contains_key(&key),
                WeightAccess::Demand => state.demand.contains_key(&key),
            };
            if range.nbytes() == 0
                || range.nbytes() > self.config.prefetch_bytes
                || retained
                || state.flights.contains_key(&key)
                || planned.iter().any(|(candidate, _, _)| *candidate == key)
                || planned.len() >= available_slots
            {
                continue;
            }
            generation = generation
                .checked_add(1)
                .context("streaming weight-flight generation overflows")?;
            planned.push((key, generation, range));
        }

        state
            .prefetch_queue
            .try_reserve(planned.len())
            .map_err(|error| anyhow!("cannot allocate bounded prefetch queue metadata: {error}"))?;
        state
            .flights
            .try_reserve(planned.len())
            .map_err(|error| anyhow!("cannot allocate bounded prefetch index metadata: {error}"))?;

        let submitted = u64::try_from(requests.len()).unwrap_or(u64::MAX);
        let admitted = u64::try_from(planned.len()).unwrap_or(u64::MAX);
        let skipped =
            u64::try_from(requests.len().saturating_sub(planned.len())).unwrap_or(u64::MAX);
        state.prefetch_submitted = state.prefetch_submitted.saturating_add(submitted);
        state.prefetch_admitted = state.prefetch_admitted.saturating_add(admitted);
        state.prefetch_skipped = state.prefetch_skipped.saturating_add(skipped);
        state.next_generation = generation;
        for (key, generation, range) in planned {
            let flight = Arc::new(Flight::queued(key, generation, range));
            state.prefetch_queue.push_back(flight.clone());
            state.flights.insert(key, flight);
            state.prefetch_outstanding += 1;
            state.prefetch_queued += 1;
            state.prefetch_outstanding_peak = state
                .prefetch_outstanding_peak
                .max(state.prefetch_outstanding);
            state.prefetch_queued_peak = state.prefetch_queued_peak.max(state.prefetch_queued);
        }
        self.shared.worker_wake.notify_all();
        Ok(())
    }

    fn cancel_prefetch(&self) -> Result<()> {
        let mut state = self
            .shared
            .cache
            .lock()
            .map_err(|_| anyhow!("streaming weight-cache lock is poisoned"))?;
        self.cancel_prefetch_locked(&mut state)
    }
}

impl Drop for StreamingSource {
    fn drop(&mut self) {
        let _ = self.cancel_prefetch();
        let _ = self.stop_workers();
    }
}

/// Raw checkpoint bytes held resident in host memory.
///
/// This type deliberately carries no row, column, layer or expert meaning. The format
/// layer owns where bytes come from, this layer owns their lifetime in memory, and the
/// model layer decides how the bytes are interpreted.
#[derive(Clone)]
pub struct ResidentWeight {
    bytes: Arc<Vec<u8>>,
    range: Range<usize>,
}

impl fmt::Debug for ResidentWeight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResidentWeight")
            .field("allocation_bytes", &self.bytes.len())
            .field("range", &self.range)
            .finish()
    }
}

impl ResidentWeight {
    /// Load the complete encoded bytes of a format-neutral weight locator.
    pub fn load(checkpoint: &dyn RangeReader, locator: WeightLocator) -> Result<Self> {
        Self::load_range(checkpoint, &locator.whole_range()).with_context(|| {
            format!(
                "loading {}-byte weight from shard {} at offset {}",
                locator.nbytes(),
                locator.shard(),
                locator.start()
            )
        })
    }

    /// Resolve and load a byte subrange of a format-neutral weight locator.
    ///
    /// `range` is half-open and relative to the tensor's start. Invalid, overflowing or
    /// out-of-bounds ranges are rejected by the checkpoint reader before any I/O.
    pub fn load_slice(
        checkpoint: &dyn RangeReader,
        locator: WeightLocator,
        range: Range<u64>,
    ) -> Result<Self> {
        let range = locator
            .range(range)
            .context("validating resident weight slice before allocation")?;
        Self::load_range(checkpoint, &range)
    }

    /// Load one checked format-neutral checkpoint range into resident memory.
    pub fn load_range(checkpoint: &dyn RangeReader, range: &WeightRange) -> Result<Self> {
        checkpoint
            .validate_range(range)
            .context("validating resident weight range before allocation")?;
        let nbytes = range.nbytes();
        let len = usize::try_from(nbytes).with_context(|| {
            format!(
                "resident weight range {}..{} ({nbytes} bytes) does not fit in address space",
                range.start(),
                range.end()
            )
        })?;

        // `vec![0; len]` aborts or panics on an impossible allocation. A checkpoint is
        // untrusted input, so make allocation failure an ordinary, actionable error.
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|err| {
            anyhow!(
                "cannot allocate {nbytes} resident bytes for weight range {}..{} in shard {}: {err}",
                range.start(),
                range.end(),
                range.shard()
            )
        })?;
        bytes.resize(len, 0);

        checkpoint.read_range_into(range, &mut bytes)?;
        let range = 0..bytes.len();
        Ok(Self {
            bytes: Arc::new(bytes),
            range,
        })
    }

    fn shared_view(bytes: Arc<Vec<u8>>, range: Range<usize>) -> Result<Self> {
        if range.end < range.start {
            return Err(anyhow!(
                "invalid resident byte view {}..{}",
                range.start,
                range.end
            ));
        }
        if range.end > bytes.len() {
            return Err(anyhow!(
                "resident byte view {}..{} is outside a {}-byte allocation",
                range.start,
                range.end,
                bytes.len()
            ));
        }
        Ok(Self { bytes, range })
    }

    /// Borrow the resident bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.bytes[self.range.clone()]
    }

    /// Number of resident bytes.
    pub fn len(&self) -> usize {
        self.range.end - self.range.start
    }

    /// Whether this buffer owns no bytes.
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }

    /// Release the wrapper and return owned bytes.
    ///
    /// This reuses the original allocation when the handle is a unique view of all its
    /// bytes. A cache hit or a subrange of a larger resident allocation is copied.
    pub fn into_bytes(self) -> Vec<u8> {
        let Self { bytes, range } = self;
        if range.start == 0 && range.end == bytes.len() {
            match Arc::try_unwrap(bytes) {
                Ok(bytes) => bytes,
                Err(bytes) => bytes.as_ref().clone(),
            }
        } else {
            bytes[range].to_vec()
        }
    }
}

impl PartialEq for ResidentWeight {
    fn eq(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
}

impl Eq for ResidentWeight {}

impl AsRef<[u8]> for ResidentWeight {
    fn as_ref(&self) -> &[u8] {
        self.bytes()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs::{File, OpenOptions};
    use std::io::Write;
    use std::os::unix::fs::FileExt;
    use std::path::{Path, PathBuf};
    use std::sync::{mpsc, Barrier};

    use beholdd_format::{CheckpointSpan, Gguf, GgufTensor, SafeTensors};

    use super::*;

    fn gguf_range(tensor: &GgufTensor, range: Range<u64>) -> WeightRange {
        tensor.locator().range(range).unwrap()
    }

    #[test]
    fn hidden_prefetch_time_uses_the_physical_read_origin() {
        let demand_before_read = Instant::now();
        let read_started = demand_before_read + Duration::from_nanos(10);

        assert_eq!(
            hidden_prefetch_nanos(read_started, 20, Some(demand_before_read)),
            0,
            "demand before physical service starts must hide no read time"
        );
        assert_eq!(
            hidden_prefetch_nanos(
                read_started,
                20,
                Some(read_started + Duration::from_nanos(7))
            ),
            7
        );
        assert_eq!(
            hidden_prefetch_nanos(
                read_started,
                20,
                Some(read_started + Duration::from_nanos(30))
            ),
            20,
            "demand after completion cannot hide more than the measured read"
        );
        assert_eq!(hidden_prefetch_nanos(read_started, 20, None), 20);
    }

    fn write_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata

        let name = "weight";
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(8u64.to_le_bytes());
        bytes.extend(24u32.to_le_bytes()); // ggml I8
        bytes.extend(0u64.to_le_bytes());

        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend(10u8..18);
        File::create(path).unwrap().write_all(&bytes).unwrap();
    }

    fn write_two_tensor_fixture(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(2u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata

        for (name, len, offset) in [("first", 5u64, 0u64), ("second", 3, 32)] {
            bytes.extend((name.len() as u64).to_le_bytes());
            bytes.extend(name.as_bytes());
            bytes.extend(1u32.to_le_bytes());
            bytes.extend(len.to_le_bytes());
            bytes.extend(24u32.to_le_bytes()); // ggml I8
            bytes.extend(offset.to_le_bytes());
        }

        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        let data_start = bytes.len();
        bytes.extend([10, 11, 12, 13, 14]);
        bytes.resize(data_start + 32, 0);
        bytes.extend([20, 21, 22]);
        File::create(path).unwrap().write_all(&bytes).unwrap();
    }

    fn write_two_tensor_safetensors_fixture(path: &Path) {
        let header = br#"{"first":{"dtype":"I8","shape":[5],"data_offsets":[0,5]},"second":{"dtype":"I8","shape":[3],"data_offsets":[5,8]}}"#;
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[10, 11, 12, 13, 14, 20, 21, 22]).unwrap();
    }

    fn write_empty_alias_safetensors_fixture(path: &Path) {
        let header = br#"{"empty_a":{"dtype":"I8","shape":[0],"data_offsets":[0,0]},"empty_b":{"dtype":"I8","shape":[0],"data_offsets":[0,0]},"value":{"dtype":"I8","shape":[1],"data_offsets":[0,1]}}"#;
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header).unwrap();
        file.write_all(&[42]).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("beholdd-memory-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[derive(Debug, Default)]
    struct GateState {
        calls: Vec<(u64, u64)>,
        completed: usize,
        permits: usize,
        release_all: bool,
        fail_starts: HashSet<u64>,
    }

    #[derive(Debug)]
    struct DelayedReader {
        inner: Arc<Gguf>,
        state: Mutex<GateState>,
        changed: Condvar,
    }

    impl DelayedReader {
        fn new(inner: Arc<Gguf>) -> Self {
            Self {
                inner,
                state: Mutex::new(GateState::default()),
                changed: Condvar::new(),
            }
        }

        fn wait_for_calls(&self, count: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut state = self.state.lock().unwrap();
            while state.calls.len() < count {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let (next, timeout) = self.changed.wait_timeout(state, remaining).unwrap();
                state = next;
                assert!(
                    !timeout.timed_out() || state.calls.len() >= count,
                    "timed out waiting for {count} physical calls; saw {:?}",
                    state.calls
                );
            }
        }

        fn wait_for_completed(&self, count: usize) {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut state = self.state.lock().unwrap();
            while state.completed < count {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let (next, timeout) = self.changed.wait_timeout(state, remaining).unwrap();
                state = next;
                assert!(
                    !timeout.timed_out() || state.completed >= count,
                    "timed out waiting for {count} completed calls; saw {}",
                    state.completed
                );
            }
        }

        fn release_one(&self) {
            let mut state = self.state.lock().unwrap();
            state.permits += 1;
            self.changed.notify_all();
        }

        fn release_everything(&self) {
            let mut state = self.state.lock().unwrap();
            state.release_all = true;
            self.changed.notify_all();
        }

        fn fail_at(&self, start: u64) {
            self.state.lock().unwrap().fail_starts.insert(start);
        }

        fn calls(&self) -> Vec<(u64, u64)> {
            self.state.lock().unwrap().calls.clone()
        }
    }

    impl RangeReader for DelayedReader {
        fn checkpoint_id(&self) -> CheckpointId {
            self.inner.checkpoint_id()
        }

        fn locators(&self) -> Box<dyn Iterator<Item = WeightLocator> + '_> {
            self.inner.locators()
        }

        fn shard_count(&self) -> usize {
            self.inner.shard_count()
        }

        fn shard_len(&self, shard: usize) -> Option<u64> {
            self.inner.shard_len(shard)
        }

        fn read_span_into(&self, span: &CheckpointSpan, buf: &mut [u8]) -> Result<()> {
            self.inner.validate_span(span)?;
            let fail = {
                let mut state = self.state.lock().unwrap();
                state.calls.push((span.start(), span.end()));
                self.changed.notify_all();
                while state.permits == 0 && !state.release_all {
                    state = self.changed.wait(state).unwrap();
                }
                if !state.release_all {
                    state.permits -= 1;
                }
                state.fail_starts.contains(&span.start())
            };

            let result = if fail {
                Err(anyhow!("injected read failure at {}", span.start()))
            } else {
                self.inner.read_span_into(span, buf)
            };
            let mut state = self.state.lock().unwrap();
            state.completed += 1;
            self.changed.notify_all();
            result
        }
    }

    struct ReleaseReadsOnDrop(Arc<DelayedReader>);

    impl Drop for ReleaseReadsOnDrop {
        fn drop(&mut self) {
            self.0.release_everything();
        }
    }

    fn delayed_fixture(tag: &str) -> (PathBuf, Arc<DelayedReader>, WeightLocator) {
        let dir = tmpdir(tag);
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let locator = checkpoint.get("weight").unwrap().locator();
        (dir, Arc::new(DelayedReader::new(checkpoint)), locator)
    }

    fn assert_source_matrix(
        checkpoint: Arc<dyn RangeReader>,
        first: WeightLocator,
        second: WeightLocator,
    ) {
        let first_full = first.whole_range();
        let first_slice = first.range(1..4).unwrap();
        let second_full = second.whole_range();
        let second_slice = second.range(1..3).unwrap();
        let requests: [(WeightRange, &[u8]); 4] = [
            (first_full, &[10, 11, 12, 13, 14]),
            (first_slice, &[11, 12, 13]),
            (second_full, &[20, 21, 22]),
            (second_slice, &[21, 22]),
        ];

        let assert_all = |source: &dyn WeightSource| {
            for access in [WeightAccess::Cyclic, WeightAccess::Demand] {
                for (range, expected) in requests {
                    assert_eq!(
                        source.load_range(&range, access).unwrap().bytes(),
                        expected,
                        "wrong bytes for {range:?} with {access:?}"
                    );
                }
            }
        };

        let direct = ResidentSource::new(checkpoint.as_ref());
        assert_all(&direct);

        let resident = FullyResidentSource::load(checkpoint.as_ref()).unwrap();
        assert_eq!(resident.resident_bytes(), 8);
        assert_all(&resident);

        let streamed = StreamingSource::new(checkpoint, CacheConfig::new(16, 16).unwrap()).unwrap();
        assert_all(&streamed);
        let initial = streamed
            .load_range(&first_full, WeightAccess::Cyclic)
            .unwrap();
        let hit = streamed
            .load_range(&first_full, WeightAccess::Cyclic)
            .unwrap();
        assert_eq!(initial.bytes(), &[10, 11, 12, 13, 14]);
        assert!(Arc::ptr_eq(&initial.bytes, &hit.bytes));
        assert!(streamed.stats().unwrap().hits() >= 1);
        assert!(streamed.stats().unwrap().cached_bytes() <= streamed.config().total_bytes());
    }

    fn assert_failure_matrix(
        checkpoint: Arc<dyn RangeReader>,
        local: WeightLocator,
        foreign: WeightLocator,
        path: &Path,
    ) {
        let direct = ResidentSource::new(checkpoint.as_ref());
        let resident = FullyResidentSource::load(checkpoint.as_ref()).unwrap();
        let streamed =
            StreamingSource::new(checkpoint.clone(), CacheConfig::new(16, 16).unwrap()).unwrap();
        for source in [
            &direct as &dyn WeightSource,
            &resident as &dyn WeightSource,
            &streamed as &dyn WeightSource,
        ] {
            let foreign_error = source
                .load_range(&foreign.whole_range(), WeightAccess::Demand)
                .unwrap_err();
            let foreign_error = format!("{foreign_error:#}");
            assert!(
                foreign_error.contains("belongs to checkpoint"),
                "got: {foreign_error}"
            );
        }
        assert_eq!(streamed.stats().unwrap(), CacheStats::default());

        let reversed = local
            .range(Range { start: 2, end: 1 })
            .unwrap_err()
            .to_string();
        assert!(reversed.contains("invalid weight-relative byte range"));
        let oob = local
            .range(local.nbytes()..local.nbytes() + 1)
            .unwrap_err()
            .to_string();
        assert!(oob.contains("outside the"), "got: {oob}");
        assert_eq!(streamed.stats().unwrap(), CacheStats::default());

        File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_len(local.end() - 1)
            .unwrap();
        let truncated = direct
            .load_range(&local.whole_range(), WeightAccess::Demand)
            .unwrap_err();
        let truncated = format!("{truncated:#}");
        assert!(
            truncated.contains("failed to fill whole buffer"),
            "got: {truncated}"
        );

        let failed_stream =
            StreamingSource::new(checkpoint.clone(), CacheConfig::new(16, 16).unwrap()).unwrap();
        let truncated = failed_stream
            .load_range(&local.whole_range(), WeightAccess::Demand)
            .unwrap_err();
        assert!(
            format!("{truncated:#}").contains("failed to fill whole buffer"),
            "got: {truncated:#}"
        );
        let stats = failed_stream.stats().unwrap();
        assert_eq!(stats.requests(), 1);
        assert_eq!(stats.misses(), 1);
        assert_eq!(stats.hits(), 0);
        assert_eq!(stats.read_bytes(), 0);
        assert_eq!(stats.cached_bytes(), 0);

        let truncated = FullyResidentSource::load(checkpoint.as_ref()).unwrap_err();
        assert!(
            format!("{truncated:#}").contains("failed to fill whole buffer"),
            "got: {truncated:#}"
        );
    }

    #[test]
    fn gguf_and_safetensors_share_the_same_direct_resident_and_streaming_contract() {
        let gguf_dir = tmpdir("format-matrix-gguf");
        let gguf_path = gguf_dir.join("m.gguf");
        write_two_tensor_fixture(&gguf_path);
        let gguf = Arc::new(Gguf::open(&gguf_path).unwrap());
        assert_source_matrix(
            gguf.clone(),
            gguf.get("first").unwrap().locator(),
            gguf.get("second").unwrap().locator(),
        );

        let safetensors_dir = tmpdir("format-matrix-safetensors");
        let safetensors_path = safetensors_dir.join("m.safetensors");
        write_two_tensor_safetensors_fixture(&safetensors_path);
        let safetensors = Arc::new(SafeTensors::open(&safetensors_path).unwrap());
        assert_source_matrix(
            safetensors.clone(),
            safetensors.get("first").unwrap().locator(),
            safetensors.get("second").unwrap().locator(),
        );

        std::fs::remove_dir_all(gguf_dir).ok();
        std::fs::remove_dir_all(safetensors_dir).ok();
    }

    #[test]
    fn gguf_and_safetensors_share_the_same_range_refusal_contract() {
        let gguf_dir = tmpdir("format-refusal-gguf");
        let gguf_path = gguf_dir.join("m.gguf");
        write_two_tensor_fixture(&gguf_path);
        let gguf = Arc::new(Gguf::open(&gguf_path).unwrap());
        let gguf_foreign = Gguf::open(&gguf_path).unwrap();
        assert_failure_matrix(
            gguf.clone(),
            gguf.get("second").unwrap().locator(),
            gguf_foreign.get("second").unwrap().locator(),
            &gguf_path,
        );

        let safetensors_dir = tmpdir("format-refusal-safetensors");
        let safetensors_path = safetensors_dir.join("m.safetensors");
        write_two_tensor_safetensors_fixture(&safetensors_path);
        let safetensors = Arc::new(SafeTensors::open(&safetensors_path).unwrap());
        let safetensors_foreign = SafeTensors::open(&safetensors_path).unwrap();
        assert_failure_matrix(
            safetensors.clone(),
            safetensors.get("second").unwrap().locator(),
            safetensors_foreign.get("second").unwrap().locator(),
            &safetensors_path,
        );

        std::fs::remove_dir_all(gguf_dir).ok();
        std::fs::remove_dir_all(safetensors_dir).ok();
    }

    #[test]
    fn fully_resident_source_coalesces_valid_empty_safetensors_aliases() {
        let dir = tmpdir("empty-safetensors-aliases");
        let path = dir.join("m.safetensors");
        write_empty_alias_safetensors_fixture(&path);
        let checkpoint = SafeTensors::open(&path).unwrap();
        let first = checkpoint.get("empty_a").unwrap().locator();
        let second = checkpoint.get("empty_b").unwrap().locator();
        assert_eq!(first, second);

        let source = FullyResidentSource::load(&checkpoint).unwrap();
        assert_eq!(source.resident_bytes(), 1);
        assert!(source
            .load_range(&first.whole_range(), WeightAccess::Demand)
            .unwrap()
            .is_empty());
        assert!(source
            .load_range(&second.whole_range(), WeightAccess::Cyclic)
            .unwrap()
            .is_empty());
        assert_eq!(
            source
                .load_range(
                    &checkpoint.get("value").unwrap().locator().whole_range(),
                    WeightAccess::Demand,
                )
                .unwrap()
                .bytes(),
            &[42]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn loads_a_full_tensor_and_a_relative_slice() {
        let dir = tmpdir("resident");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Gguf::open(&path).unwrap();
        let tensor = checkpoint.get("weight").unwrap();

        let full = ResidentWeight::load(&checkpoint, tensor.locator()).unwrap();
        assert_eq!(full.bytes(), &[10, 11, 12, 13, 14, 15, 16, 17]);
        assert_eq!(full.len(), 8);
        assert!(!full.is_empty());
        assert_eq!(full.as_ref(), full.bytes());

        let slice = ResidentWeight::load_slice(&checkpoint, tensor.locator(), 2..6).unwrap();
        assert_eq!(slice.into_bytes(), vec![12, 13, 14, 15]);

        let empty = ResidentWeight::load_slice(&checkpoint, tensor.locator(), 8..8).unwrap();
        assert!(empty.is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn invalid_slice_is_an_error_not_an_allocation_or_read() {
        let dir = tmpdir("invalid");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Gguf::open(&path).unwrap();
        let tensor = checkpoint.get("weight").unwrap();

        let err = format!(
            "{:#}",
            ResidentWeight::load_slice(&checkpoint, tensor.locator(), 7..9).unwrap_err()
        );
        assert!(err.contains("outside the 8-byte weight"), "got: {err}");

        let reversed = Range { start: 9, end: 7 };
        let err = format!(
            "{:#}",
            ResidentWeight::load_slice(&checkpoint, tensor.locator(), reversed).unwrap_err()
        );
        assert!(
            err.contains("invalid weight-relative byte range 9..7"),
            "got: {err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn fully_resident_source_packs_only_payloads_and_returns_shared_views() {
        let dir = tmpdir("fully-resident-packed");
        let path = dir.join("m.gguf");
        write_two_tensor_fixture(&path);
        let checkpoint = Gguf::open(&path).unwrap();
        let source = FullyResidentSource::load(&checkpoint).unwrap();

        assert_eq!(source.resident_bytes(), 8);
        assert_eq!(source.bytes.len(), 8);
        assert!(path.metadata().unwrap().len() > source.resident_bytes());
        assert_eq!(source.bytes.as_slice(), &[10, 11, 12, 13, 14, 20, 21, 22]);

        let first = checkpoint.get("first").unwrap();
        let second = checkpoint.get("second").unwrap();
        let first_full = gguf_range(first, 0..5);
        let first_slice = gguf_range(first, 1..4);
        let second_full = gguf_range(second, 0..3);
        let second_empty = gguf_range(second, 3..3);
        let full = source
            .load_range(&first_full, WeightAccess::Cyclic)
            .unwrap();
        let slice = source
            .load_range(&first_slice, WeightAccess::Demand)
            .unwrap();
        let other = source
            .load_range(&second_full, WeightAccess::Cyclic)
            .unwrap();
        let empty = source
            .load_range(&second_empty, WeightAccess::Demand)
            .unwrap();

        assert_eq!(full.bytes(), &[10, 11, 12, 13, 14]);
        assert_eq!(slice.bytes(), &[11, 12, 13]);
        assert_eq!(other.bytes(), &[20, 21, 22]);
        assert!(empty.is_empty());
        assert!(Arc::ptr_eq(&full.bytes, &slice.bytes));
        assert!(Arc::ptr_eq(&full.bytes, &other.bytes));
        assert!(Arc::ptr_eq(&full.bytes, &source.bytes));
        assert_ne!(full, slice);
        assert_eq!(
            slice,
            ResidentWeight::load_slice(&checkpoint, first.locator(), 1..4).unwrap()
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fully_resident_source_performs_no_reads_after_preload() {
        let dir = tmpdir("fully-resident-no-reread");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Gguf::open(&path).unwrap();
        let tensor = checkpoint.get("weight").unwrap();
        let source = FullyResidentSource::load(&checkpoint).unwrap();

        let replacement = [90, 91, 92, 93, 94, 95, 96, 97];
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.write_all_at(&replacement, tensor.offset()).unwrap();
        file.sync_all().unwrap();

        let resident = source
            .load_range(&gguf_range(tensor, 0..8), WeightAccess::Demand)
            .unwrap();
        assert_eq!(resident.bytes(), &[10, 11, 12, 13, 14, 15, 16, 17]);

        let direct = ResidentSource::new(&checkpoint)
            .load_range(&gguf_range(tensor, 0..8), WeightAccess::Demand)
            .unwrap();
        assert_eq!(direct.bytes(), replacement);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn fully_resident_source_refuses_foreign_ranges_and_locators_refuse_invalid_slices() {
        let first_dir = tmpdir("fully-resident-source-first");
        let second_dir = tmpdir("fully-resident-source-second");
        let first_path = first_dir.join("m.gguf");
        let second_path = second_dir.join("m.gguf");
        write_fixture(&first_path);
        write_fixture(&second_path);
        let first = Gguf::open(&first_path).unwrap();
        let second = Gguf::open(&second_path).unwrap();
        let source = FullyResidentSource::load(&first).unwrap();

        let foreign = format!(
            "{:#}",
            source
                .load_range(
                    &gguf_range(second.get("weight").unwrap(), 0..2),
                    WeightAccess::Demand,
                )
                .unwrap_err()
        );
        assert!(
            foreign.contains("weight range belongs to checkpoint"),
            "got: {foreign}"
        );

        let tensor = first.get("weight").unwrap();
        let reversed = tensor
            .locator()
            .range(Range { start: 3, end: 2 })
            .unwrap_err()
            .to_string();
        assert!(reversed.contains("invalid weight-relative byte range 3..2"));

        let oob = tensor.locator().range(7..9).unwrap_err().to_string();
        assert!(oob.contains("outside the 8-byte weight"), "got: {oob}");
        std::fs::remove_dir_all(first_dir).ok();
        std::fs::remove_dir_all(second_dir).ok();
    }

    #[test]
    fn fully_resident_view_outlives_source_checkpoint_and_file() {
        let dir = tmpdir("fully-resident-view-lifetime");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Gguf::open(&path).unwrap();
        let source = FullyResidentSource::load(&checkpoint).unwrap();
        let retained_range = gguf_range(checkpoint.get("weight").unwrap(), 2..6);
        let retained = source
            .load_range(&retained_range, WeightAccess::Demand)
            .unwrap();

        drop(source);
        drop(checkpoint);
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(retained.bytes(), &[12, 13, 14, 15]);
        assert_eq!(retained.into_bytes(), vec![12, 13, 14, 15]);
    }

    #[test]
    fn fit_policy_has_stable_spelling_default_and_permissions() {
        assert_eq!(FitPolicy::default(), FitPolicy::Exact);
        for (policy, spelling, streaming, degradation) in [
            (FitPolicy::Exact, "exact", true, false),
            (FitPolicy::Degrade, "degrade", true, true),
            (FitPolicy::Refuse, "refuse", false, false),
        ] {
            assert_eq!(policy.to_string(), spelling);
            assert_eq!(spelling.parse::<FitPolicy>().unwrap(), policy);
            assert_eq!(policy.allows_streaming(), streaming);
            assert_eq!(policy.allows_degradation(), degradation);
        }
        let error = "auto".parse::<FitPolicy>().unwrap_err().to_string();
        assert_eq!(
            error,
            "unknown fit policy \"auto\"; expected exact, degrade, or refuse"
        );
        assert!("Exact".parse::<FitPolicy>().is_err());
    }

    #[test]
    fn cache_config_checks_and_reports_its_exact_budget() {
        let config = CacheConfig::new(7, 11).unwrap();
        assert_eq!(config.pinned_bytes(), 7);
        assert_eq!(config.demand_bytes(), 11);
        assert_eq!(config.total_bytes(), 18);
        assert_eq!(CacheConfig::default().total_bytes(), 0);
        assert!(CacheConfig::new(u64::MAX, 1).is_err());

        let prefetched = config.with_prefetch(13, 3, 5).unwrap();
        assert_eq!(prefetched.prefetch_bytes(), 13);
        assert_eq!(prefetched.prefetch_queue_depth(), 3);
        assert_eq!(prefetched.prefetch_workers(), 5);
        assert!(prefetched.prefetch_enabled());
        assert_eq!(prefetched.total_bytes(), 31);
        assert!(!config.prefetch_enabled());
        // Every pairwise and single-field mismatch among the three prefetch limits is
        // rejected: only all-zero (disabled) or all-nonzero (enabled) is a valid state.
        assert!(config.with_prefetch(1, 0, 0).is_err());
        assert!(config.with_prefetch(0, 1, 0).is_err());
        assert!(config.with_prefetch(0, 0, 1).is_err());
        assert!(config.with_prefetch(1, 1, 0).is_err());
        assert!(config.with_prefetch(1, 0, 1).is_err());
        assert!(config.with_prefetch(0, 1, 1).is_err());
        assert!(CacheConfig::new(u64::MAX - 1, 1)
            .unwrap()
            .with_prefetch(1, 1, 1)
            .is_err());
    }

    #[test]
    fn streaming_probe_observes_live_and_stopped_workers() {
        let dir = tmpdir("streaming-probe");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(8, 1, 1)
            .unwrap();
        let mut source = StreamingSource::new(checkpoint, config).unwrap();

        source.probe().unwrap();
        source.stop_workers().unwrap();
        assert!(source.probe().is_err());

        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cyclic_cache_pins_only_the_initial_request_prefix() {
        let dir = tmpdir("pinned-prefix");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let tensor = checkpoint.get("weight").unwrap();
        let source =
            StreamingSource::new(checkpoint.clone(), CacheConfig::new(4, 0).unwrap()).unwrap();

        assert_eq!(
            source
                .load_range(&gguf_range(tensor, 0..2), WeightAccess::Cyclic)
                .unwrap()
                .bytes(),
            &[10, 11]
        );
        source
            .load_range(&gguf_range(tensor, 0..2), WeightAccess::Cyclic)
            .unwrap();
        // This does not fit the two remaining bytes, so it closes the pinned prefix.
        source
            .load_range(&gguf_range(tensor, 2..5), WeightAccess::Cyclic)
            .unwrap();
        // A later small request would fit, but caching it would no longer be a prefix.
        source
            .load_range(&gguf_range(tensor, 5..6), WeightAccess::Cyclic)
            .unwrap();
        source
            .load_range(&gguf_range(tensor, 5..6), WeightAccess::Cyclic)
            .unwrap();

        let stats = source.stats().unwrap();
        assert_eq!(stats.requests(), 5);
        assert_eq!(stats.hits(), 1);
        assert_eq!(stats.misses(), 4);
        assert_eq!(stats.read_bytes(), 7);
        assert_eq!(stats.pinned_bytes(), 2);
        assert_eq!(stats.demand_bytes(), 0);
        assert_eq!(stats.cached_bytes(), 2);
        assert_eq!(stats.pinned_entries(), 1);
        assert_eq!(stats.demand_entries(), 0);
        assert_eq!(stats.hit_rate(), 0.2);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn demand_cache_is_exact_lru_with_a_strict_byte_limit() {
        let dir = tmpdir("demand-lru");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let tensor = checkpoint.get("weight").unwrap();
        let source =
            StreamingSource::new(checkpoint.clone(), CacheConfig::new(0, 4).unwrap()).unwrap();

        let a = source
            .load_range(&gguf_range(tensor, 0..2), WeightAccess::Demand)
            .unwrap();
        let b = source
            .load_range(&gguf_range(tensor, 2..4), WeightAccess::Demand)
            .unwrap();
        assert_eq!(a.bytes(), &[10, 11]);
        assert_eq!(b.bytes(), &[12, 13]);
        let a_hit = source
            .load_range(&gguf_range(tensor, 0..2), WeightAccess::Demand)
            .unwrap(); // A is newest.
        assert!(Arc::ptr_eq(&a.bytes, &a_hit.bytes));
        source
            .load_range(&gguf_range(tensor, 4..6), WeightAccess::Demand)
            .unwrap(); // evicts B.
        source
            .load_range(&gguf_range(tensor, 0..2), WeightAccess::Demand)
            .unwrap(); // A survived.
        source
            .load_range(&gguf_range(tensor, 2..4), WeightAccess::Demand)
            .unwrap(); // B was evicted.
        assert_eq!(a.bytes(), &[10, 11], "an evicted handle remains valid");

        let stats = source.stats().unwrap();
        assert_eq!(stats.requests(), 6);
        assert_eq!(stats.hits(), 2);
        assert_eq!(stats.misses(), 4);
        assert_eq!(stats.read_bytes(), 8);
        assert_eq!(stats.evictions(), 2);
        assert_eq!(stats.demand_bytes(), 4);
        assert_eq!(stats.demand_entries(), 2);
        assert!(stats.cached_bytes() <= source.config().total_bytes());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn oversized_demand_read_bypasses_without_flushing_useful_entries() {
        let dir = tmpdir("demand-bypass");
        let path = dir.join("m.gguf");
        write_fixture(&path);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let tensor = checkpoint.get("weight").unwrap();
        let source =
            StreamingSource::new(checkpoint.clone(), CacheConfig::new(0, 2).unwrap()).unwrap();

        source
            .load_range(&gguf_range(tensor, 0..2), WeightAccess::Demand)
            .unwrap();
        source
            .load_range(&gguf_range(tensor, 2..5), WeightAccess::Demand)
            .unwrap();
        source
            .load_range(&gguf_range(tensor, 0..2), WeightAccess::Demand)
            .unwrap();

        let stats = source.stats().unwrap();
        assert_eq!(stats.hits(), 1);
        assert_eq!(stats.misses(), 2);
        assert_eq!(stats.evictions(), 0);
        assert_eq!(stats.demand_bytes(), 2);
        assert_eq!(stats.demand_entries(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn streaming_source_refuses_foreign_locators_before_a_cache_lookup() {
        let first_dir = tmpdir("source-first");
        let second_dir = tmpdir("source-second");
        let first_path = first_dir.join("m.gguf");
        let second_path = second_dir.join("m.gguf");
        write_fixture(&first_path);
        write_fixture(&second_path);
        let first = Arc::new(Gguf::open(&first_path).unwrap());
        let second = Gguf::open(&second_path).unwrap();
        let source = StreamingSource::new(first.clone(), CacheConfig::new(8, 8).unwrap()).unwrap();

        let error = source
            .load_range(
                &gguf_range(second.get("weight").unwrap(), 0..2),
                WeightAccess::Demand,
            )
            .unwrap_err();
        let error = format!("{error:#}");
        assert!(error.contains("weight range belongs to checkpoint"));
        assert_eq!(source.stats().unwrap(), CacheStats::default());

        let reversed = Range { start: 3, end: 2 };
        let error = first
            .get("weight")
            .unwrap()
            .locator()
            .range(reversed)
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid weight-relative byte range"));
        assert_eq!(source.stats().unwrap(), CacheStats::default());
        std::fs::remove_dir_all(&first_dir).ok();
        std::fs::remove_dir_all(&second_dir).ok();
    }

    #[test]
    fn ready_prefetch_overlaps_prior_work_and_is_consumed_without_another_read() {
        let (dir, reader, locator) = delayed_fixture("prefetch-ready");
        let config = CacheConfig::new(0, 8)
            .unwrap()
            .with_prefetch(4, 2, 1)
            .unwrap();
        let source = Arc::new(StreamingSource::new(reader.clone(), config).unwrap());
        let _release = ReleaseReadsOnDrop(reader.clone());
        let range = locator.range(0..4).unwrap();

        source
            .prefetch_phase(&[WeightRequest::new(range, WeightAccess::Demand)])
            .unwrap();
        reader.wait_for_calls(1);
        let reading = source.stats().unwrap();
        assert_eq!(reading.requests(), 0, "I/O began before foreground demand");
        assert_eq!(reading.prefetch_reading(), 1);
        assert_eq!(reading.prefetch_reserved_bytes(), 4);

        reader.release_one();
        reader.wait_for_completed(1);
        source
            .wait_for_stats_for_test(|stats| stats.prefetch_ready() == 1)
            .unwrap();
        let weight = source.load_range(&range, WeightAccess::Demand).unwrap();
        assert_eq!(weight.bytes(), &[10, 11, 12, 13]);
        assert_eq!(reader.calls(), vec![(range.start(), range.end())]);

        let stats = source.stats().unwrap();
        assert_eq!(stats.requests(), 1);
        assert_eq!(stats.hits(), 1);
        assert_eq!(stats.misses(), 0);
        assert_eq!(stats.hit_bytes(), 4);
        assert_eq!(stats.physical_reads(), 1);
        assert_eq!(stats.prefetch_physical_reads(), 1);
        assert_eq!(stats.prefetch_useful(), 1);
        assert_eq!(stats.prefetch_useful_bytes(), 4);
        assert_eq!(stats.prefetch_outstanding(), 0);
        assert_eq!(stats.prefetch_reserved_bytes(), 0);
        assert_eq!(stats.requests(), stats.hits() + stats.misses());
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn late_prefetch_waits_for_one_physical_read_and_remains_cancellable_to_query() {
        let (dir, reader, locator) = delayed_fixture("prefetch-late");
        let config = CacheConfig::new(0, 8)
            .unwrap()
            .with_prefetch(4, 1, 1)
            .unwrap();
        let source = Arc::new(StreamingSource::new(reader.clone(), config).unwrap());
        let _release = ReleaseReadsOnDrop(reader.clone());
        let range = locator.range(0..4).unwrap();
        source
            .prefetch_phase(&[WeightRequest::new(range, WeightAccess::Demand)])
            .unwrap();
        reader.wait_for_calls(1);

        let barrier = Arc::new(Barrier::new(3));
        let mut callers = Vec::new();
        for _ in 0..2 {
            let caller_source = source.clone();
            let caller_barrier = barrier.clone();
            callers.push(std::thread::spawn(move || {
                caller_barrier.wait();
                caller_source.load_range(&range, WeightAccess::Demand)
            }));
        }
        barrier.wait();
        let blocked = source
            .wait_for_stats_for_test(|stats| stats.prefetch_late() == 1 && stats.requests() == 2)
            .unwrap();
        assert_eq!(
            blocked.physical_reads(),
            0,
            "the blocked read has not returned"
        );
        assert_eq!(blocked.prefetch_reading(), 1);
        assert_eq!(reader.calls().len(), 1);
        source.cancel_prefetch().unwrap();
        assert_eq!(source.stats().unwrap().prefetch_cancelled(), 0);

        reader.release_one();
        let first = callers.remove(0).join().unwrap().unwrap();
        let second = callers.remove(0).join().unwrap().unwrap();
        assert_eq!(first.bytes(), &[10, 11, 12, 13]);
        assert_eq!(second.bytes(), &[10, 11, 12, 13]);
        assert!(Arc::ptr_eq(&first.bytes, &second.bytes));
        let stats = source.stats().unwrap();
        assert_eq!(reader.calls().len(), 1);
        assert_eq!(stats.requests(), 2);
        assert_eq!(stats.hits(), 0);
        assert_eq!(stats.misses(), 2);
        assert_eq!(stats.physical_reads(), 1);
        assert_eq!(stats.prefetch_useful(), 1);
        assert_eq!(stats.prefetch_late(), 1);
        assert_eq!(stats.prefetch_cancelled(), 0);
        assert!(stats.foreground_wait_duration() > Duration::ZERO);
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn prefetch_pool_performs_concurrent_physical_reads_up_to_its_worker_count() {
        let (dir, reader, locator) = delayed_fixture("prefetch-pool-concurrency");
        let ranges = [
            locator.range(0..2).unwrap(),
            locator.range(2..4).unwrap(),
            locator.range(4..6).unwrap(),
            locator.range(6..8).unwrap(),
        ];
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(8, 4, 4)
            .unwrap();
        let source = StreamingSource::new(reader.clone(), config).unwrap();
        let _release = ReleaseReadsOnDrop(reader.clone());

        source
            .prefetch_phase(&ranges.map(|range| WeightRequest::new(range, WeightAccess::Demand)))
            .unwrap();

        // A single worker could never register four calls before any of them is
        // released: each blocks on a permit inside read_span_into, so one drainer can
        // only ever have one call outstanding at a time. Four simultaneous calls within
        // the timeout is only reachable with four workers each blocked in their own
        // physical read at once -- this is the pool's concurrency, proven rather than
        // timed.
        reader.wait_for_calls(4);
        let mut calls = reader.calls();
        calls.sort_unstable();
        let mut expected: Vec<(u64, u64)> = ranges.iter().map(|r| (r.start(), r.end())).collect();
        expected.sort_unstable();
        assert_eq!(
            calls, expected,
            "all four ranges must be concurrently in flight"
        );

        let mid = source.stats().unwrap();
        assert_eq!(mid.prefetch_reading(), 4);
        assert_eq!(mid.prefetch_reading_bytes(), 8);
        assert!(mid.prefetch_reserved_bytes() <= config.prefetch_bytes());
        assert!(mid.prefetch_outstanding() <= config.prefetch_queue_depth());

        reader.release_everything();
        for range in ranges {
            let weight = source.load_range(&range, WeightAccess::Demand).unwrap();
            assert_eq!(weight.bytes().len(), 2);
        }
        let stats = source.stats().unwrap();
        assert_eq!(stats.prefetch_physical_reads(), 4);
        assert_eq!(stats.prefetch_useful(), 4);
        assert_eq!(stats.prefetch_outstanding(), 0);
        assert_eq!(stats.prefetch_reserved_bytes(), 0);

        // finish() joins every pool thread, not just one -- exercise that path directly
        // rather than only through Drop.
        let final_stats = source.finish().unwrap();
        assert_eq!(final_stats.prefetch_physical_reads(), 4);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn prefetch_byte_and_metadata_limits_hold_while_ready_bytes_block_the_worker() {
        let (dir, reader, locator) = delayed_fixture("prefetch-bounds");
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(2, 2, 1)
            .unwrap();
        let source = StreamingSource::new(reader.clone(), config).unwrap();
        let _release = ReleaseReadsOnDrop(reader.clone());
        let ranges = [
            locator.range(0..2).unwrap(),
            locator.range(2..4).unwrap(),
            locator.range(4..6).unwrap(),
        ];
        source
            .prefetch_phase(&ranges.map(|range| WeightRequest::new(range, WeightAccess::Cyclic)))
            .unwrap();
        reader.wait_for_calls(1);
        let first = source.stats().unwrap();
        assert_eq!(first.prefetch_submitted(), 3);
        assert_eq!(first.prefetch_admitted(), 2);
        assert_eq!(first.prefetch_skipped(), 1);
        assert_eq!(first.prefetch_outstanding(), 2);
        assert_eq!(first.prefetch_reading_bytes(), 2);
        assert!(first.prefetch_reserved_bytes() <= config.prefetch_bytes());

        reader.release_one();
        reader.wait_for_completed(1);
        let ready = source
            .wait_for_stats_for_test(|stats| stats.prefetch_ready() == 1)
            .unwrap();
        assert_eq!(reader.calls().len(), 1, "ready bytes hold the full budget");
        assert_eq!(ready.prefetch_queued(), 1);
        assert_eq!(ready.prefetch_reserved_bytes(), 2);

        assert_eq!(
            source
                .load_range(&ranges[0], WeightAccess::Cyclic)
                .unwrap()
                .bytes(),
            &[10, 11]
        );
        reader.wait_for_calls(2);
        let second = source.stats().unwrap();
        assert!(second.prefetch_reserved_bytes() <= config.prefetch_bytes());
        assert!(second.prefetch_outstanding() <= config.prefetch_queue_depth());
        assert!(second.prefetch_reserved_bytes_peak() <= config.prefetch_bytes());
        assert!(second.prefetch_outstanding_peak() <= config.prefetch_queue_depth());

        source.cancel_prefetch().unwrap();
        reader.release_one();
        reader.wait_for_completed(2);
        source
            .wait_for_stats_for_test(|stats| stats.prefetch_reading() == 0)
            .unwrap();
        let final_stats = source.stats().unwrap();
        assert_eq!(final_stats.prefetch_outstanding(), 0);
        assert_eq!(final_stats.prefetch_reserved_bytes(), 0);
        assert_eq!(final_stats.prefetch_cancelled(), 1);
        assert_eq!(final_stats.prefetch_wasted(), 1);
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cancellation_detaches_a_blocked_generation_and_discards_its_result() {
        let (dir, reader, locator) = delayed_fixture("prefetch-cancel");
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(4, 1, 1)
            .unwrap();
        let source = StreamingSource::new(reader.clone(), config).unwrap();
        let _release = ReleaseReadsOnDrop(reader.clone());
        let first = locator.range(0..4).unwrap();
        let replacement = first;
        source
            .prefetch_phase(&[WeightRequest::new(first, WeightAccess::Demand)])
            .unwrap();
        reader.wait_for_calls(1);
        source.cancel_prefetch().unwrap();
        let detached = source.stats().unwrap();
        assert_eq!(detached.prefetch_outstanding(), 0);
        assert_eq!(detached.prefetch_detached(), 1);
        assert_eq!(detached.prefetch_detached_peak(), 1);
        assert_eq!(detached.prefetch_reading_bytes(), 4);
        assert_eq!(detached.prefetch_cancelled(), 0);

        source
            .prefetch_phase(&[WeightRequest::new(replacement, WeightAccess::Demand)])
            .unwrap();
        assert_eq!(source.stats().unwrap().prefetch_outstanding(), 1);
        reader.release_one();
        reader.wait_for_completed(1);
        reader.wait_for_calls(2);
        assert_eq!(
            reader.calls(),
            vec![
                (first.start(), first.end()),
                (replacement.start(), replacement.end())
            ]
        );
        assert_eq!(source.stats().unwrap().prefetch_outstanding(), 1);
        source
            .wait_for_stats_for_test(|stats| stats.prefetch_cancelled() == 1)
            .unwrap();
        reader.release_one();
        reader.wait_for_completed(2);
        source
            .wait_for_stats_for_test(|stats| stats.prefetch_ready() == 1)
            .unwrap();
        let stats = source.stats().unwrap();
        assert_eq!(stats.prefetch_detached(), 0);
        assert_eq!(stats.prefetch_wasted(), 1);
        assert_eq!(stats.prefetch_wasted_bytes(), 4);
        assert_eq!(stats.prefetch_admitted(), 2);
        source.cancel_prefetch().unwrap();
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn finish_waits_for_an_active_prefetch_and_returns_settled_counters() {
        let (dir, reader, locator) = delayed_fixture("prefetch-finish");
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(4, 1, 1)
            .unwrap();
        let source = StreamingSource::new(reader.clone(), config).unwrap();
        let _release = ReleaseReadsOnDrop(reader.clone());
        let range = locator.range(0..4).unwrap();
        source
            .prefetch_phase(&[WeightRequest::new(range, WeightAccess::Demand)])
            .unwrap();
        reader.wait_for_calls(1);

        let (finished_tx, finished_rx) = mpsc::sync_channel(1);
        let finisher = std::thread::spawn(move || {
            finished_tx.send(source.finish()).unwrap();
        });
        assert!(matches!(
            finished_rx.try_recv(),
            Err(mpsc::TryRecvError::Empty)
        ));

        reader.release_one();
        let stats = finished_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("timed out waiting for final streaming telemetry")
            .unwrap();
        finisher.join().unwrap();
        assert_eq!(stats.physical_reads(), 1);
        assert_eq!(stats.read_bytes(), 4);
        assert_eq!(stats.prefetch_physical_reads(), 1);
        assert_eq!(stats.prefetch_cancelled(), 1);
        assert_eq!(stats.prefetch_wasted(), 1);
        assert_eq!(stats.prefetch_wasted_bytes(), 4);
        assert_eq!(stats.prefetch_outstanding(), 0);
        assert_eq!(stats.prefetch_detached(), 0);
        assert_eq!(stats.prefetch_reading(), 0);
        assert_eq!(stats.prefetch_reserved_bytes(), 0);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn prefetched_failure_is_propagated_without_a_duplicate_foreground_read() {
        let (dir, reader, locator) = delayed_fixture("prefetch-failure");
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(4, 1, 1)
            .unwrap();
        let source = StreamingSource::new(reader.clone(), config).unwrap();
        let _release = ReleaseReadsOnDrop(reader.clone());
        let range = locator.range(0..4).unwrap();
        reader.fail_at(range.start());
        source
            .prefetch_phase(&[WeightRequest::new(range, WeightAccess::Demand)])
            .unwrap();
        reader.wait_for_calls(1);
        reader.release_one();
        reader.wait_for_completed(1);
        source
            .wait_for_stats_for_test(|stats| stats.prefetch_physical_read_failures() == 1)
            .unwrap();

        let error = source.load_range(&range, WeightAccess::Demand).unwrap_err();
        assert!(format!("{error:#}").contains("injected read failure"));
        let stats = source.stats().unwrap();
        assert_eq!(reader.calls().len(), 1);
        assert_eq!(stats.requests(), 1);
        assert_eq!(stats.misses(), 1);
        assert_eq!(stats.physical_reads(), 1);
        assert_eq!(stats.physical_read_failures(), 1);
        assert_eq!(stats.prefetch_outstanding(), 0);
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn concurrent_foreground_callers_registered_before_completion_share_one_read() {
        let (dir, reader, locator) = delayed_fixture("foreground-single-flight");
        let source = Arc::new(
            StreamingSource::new(reader.clone(), CacheConfig::new(0, 0).unwrap()).unwrap(),
        );
        let _release = ReleaseReadsOnDrop(reader.clone());
        let range = locator.range(0..4).unwrap();
        let barrier = Arc::new(Barrier::new(3));
        let mut callers = Vec::new();
        for _ in 0..2 {
            let caller_source = source.clone();
            let caller_barrier = barrier.clone();
            callers.push(std::thread::spawn(move || {
                caller_barrier.wait();
                caller_source.load_range(&range, WeightAccess::Demand)
            }));
        }
        barrier.wait();
        reader.wait_for_calls(1);
        source
            .wait_for_stats_for_test(|stats| stats.requests() == 2)
            .unwrap();
        assert_eq!(reader.calls().len(), 1);
        reader.release_one();
        for caller in callers {
            assert_eq!(caller.join().unwrap().unwrap().bytes(), &[10, 11, 12, 13]);
        }
        let stats = source.stats().unwrap();
        assert_eq!(stats.requests(), 2);
        assert_eq!(stats.hits(), 0);
        assert_eq!(stats.misses(), 2);
        assert_eq!(stats.physical_reads(), 1);
        assert_eq!(reader.calls().len(), 1);
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn oversized_empty_duplicate_and_unsubmitted_ranges_have_exact_io_accounting() {
        let (dir, reader, locator) = delayed_fixture("prefetch-accounting");
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(2, 2, 1)
            .unwrap();
        let source = Arc::new(StreamingSource::new(reader.clone(), config).unwrap());
        let _release = ReleaseReadsOnDrop(reader.clone());
        let admitted = locator.range(0..2).unwrap();
        let oversized = locator.range(2..5).unwrap();
        let empty = locator.range(5..5).unwrap();
        source
            .prefetch_phase(&[
                WeightRequest::new(admitted, WeightAccess::Demand),
                WeightRequest::new(admitted, WeightAccess::Demand),
                WeightRequest::new(oversized, WeightAccess::Demand),
                WeightRequest::new(empty, WeightAccess::Demand),
            ])
            .unwrap();
        reader.wait_for_calls(1);
        reader.release_one();
        reader.wait_for_completed(1);
        source
            .wait_for_stats_for_test(|stats| stats.prefetch_ready() == 1)
            .unwrap();
        let stats = source.stats().unwrap();
        assert_eq!(stats.prefetch_submitted(), 4);
        assert_eq!(stats.prefetch_admitted(), 1);
        assert_eq!(stats.prefetch_skipped(), 3);

        let unsubmitted = locator.range(6..8).unwrap();
        let caller_source = source.clone();
        let worker_source = caller_source.clone();
        let caller = std::thread::spawn(move || {
            worker_source.load_range(&unsubmitted, WeightAccess::Demand)
        });
        reader.wait_for_calls(2);
        assert_eq!(reader.calls()[1], (unsubmitted.start(), unsubmitted.end()));
        reader.release_one();
        assert_eq!(caller.join().unwrap().unwrap().bytes(), &[16, 17]);
        assert_eq!(caller_source.stats().unwrap().physical_reads(), 2);
        caller_source.cancel_prefetch().unwrap();
        drop(caller_source);
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn prefetch_batch_validation_is_transactional_before_stats_queue_or_io() {
        let (dir, reader, locator) = delayed_fixture("prefetch-foreign-batch");
        let foreign_dir = tmpdir("prefetch-foreign-batch-other");
        let foreign_path = foreign_dir.join("m.gguf");
        write_fixture(&foreign_path);
        let foreign = Gguf::open(&foreign_path).unwrap();
        let foreign_range = foreign
            .get("weight")
            .unwrap()
            .locator()
            .range(0..2)
            .unwrap();
        let source = StreamingSource::new(
            reader.clone(),
            CacheConfig::new(0, 0)
                .unwrap()
                .with_prefetch(4, 2, 1)
                .unwrap(),
        )
        .unwrap();
        let _release = ReleaseReadsOnDrop(reader.clone());
        let local = locator.range(0..2).unwrap();

        let error = source
            .prefetch_phase(&[
                WeightRequest::new(local, WeightAccess::Demand),
                WeightRequest::new(foreign_range, WeightAccess::Demand),
            ])
            .unwrap_err();
        assert!(format!("{error:#}").contains("belongs to checkpoint"));
        assert_eq!(source.stats().unwrap(), CacheStats::default());
        assert!(reader.calls().is_empty());
        drop(source);
        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(foreign_dir).ok();
    }
}
