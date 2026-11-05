//! Explicit, bounded resident HAG serving for the qualified dense Qwen3 geometry.
//!
//! Serving uses an explicit cache capacity (default 2048), on-demand attention and serial requests. Token
//! execution is synchronous; control/sink cancellation is observed only outside a
//! complete token and resets healthy state. Device uncertainty fails closed across
//! every runtime in this process, including after unload, until process restart.

use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc, Mutex, OnceLock,
};

use anyhow::{bail, ensure, Context, Result};
use beholdd_backend::execution::{PreparedWeight, RecordedBatch};
use beholdd_backend::hag::native_execution::{
    MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionStats,
    NativeRecordingFootprint,
};
use beholdd_format::{CheckpointId, Gguf, WeightRange};
use beholdd_memory::{
    available_memory, cgroup_memory, ResidentSource, ResidentWeight, WeightAccess, WeightSource,
};
use beholdd_model::qwen3moe::{
    DenseAttentionPlan, DenseFfnBlockPlan, DenseModelLimits, DenseModelMemory, DenseModelPlan,
    GenerateOutput, GenerationFinishReason, PreparedDenseAttention, PreparedDenseFfnBlock,
    PreparedVocabulary, QwenArchitecture, ResidentDenseAttention, ResidentDenseModel,
    VocabularyPlan, VocabularyTile,
};
use beholdd_model::QwenModel;

use super::native_phase::{session_reset, Phase, Recorder};
use crate::{ExecutionLimits, ModelConfig};

#[cfg(test)]
pub(crate) const CAPACITY: usize = 2048;
const MAX_CAPACITY: usize = beholdd_model::qwen3moe::MAX_ON_DEMAND_CAPACITY;
const TILE_ROWS: usize = 4096;
const DIM: usize = 1024;
const VOCAB: usize = 151_936;
const LAYERS: u64 = 28;
const PACKED_BYTES: u64 = 633_233_408;
const SOURCE_BYTES: u64 = 633_495_552;
const STAGING_BYTES: u64 = 4_456_448;
const PREPARATION_READS: u64 = 347;
const LOGIT_BYTES: u64 = VOCAB as u64 * 4;
const MINIMUM_HEADROOM: u64 = 128 * 1024 * 1024;
const PERSISTENT_AUXILIARY_BYTES: u64 = LAYERS * (DIM as u64 * 4 + 12) + DIM as u64 * 4 + 4;
// Fixed-geometry ceiling used by standalone cost tests. Live admission instead
// counts the actual retained on-demand preparations through the model plan API.
const RETAINED_ATTENTION_METADATA_BOUND: u64 = 1024 * 1024;
static PROCESS_HEALTH: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static AUDIT_SEQUENCE: AtomicU64 = AtomicU64::new(1);

pub(super) fn process_health() -> Arc<AtomicBool> {
    Arc::clone(PROCESS_HEALTH.get_or_init(|| Arc::new(AtomicBool::new(true))))
}

/// Observe the process-wide uncertainty latch without touching the device.
/// Call after unload/eviction to surface quarantined retirement as a failure.
pub(crate) fn require_process_healthy() -> Result<()> {
    ensure!(
        process_health().load(Ordering::Acquire),
        "HAG cleanup/execution is uncertain; restart the process"
    );
    Ok(())
}

/// Reject over-capacity requests before loading or resetting any resident state.
pub(crate) fn validate_request(
    capacity: usize,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<()> {
    ensure!(
        (1..=MAX_CAPACITY).contains(&capacity),
        "unsupported HAG context capacity"
    );
    ExecutionLimits {
        max_evaluated_tokens: Some(capacity),
    }
    .resolve(prompt_len, Some(max_new_tokens), max_new_tokens)
    .map(|_| ())
    .map_err(Into::into)
}

fn tensor_bytes(capacity: usize) -> Result<u64> {
    let capacity = u64::try_from(capacity)?;
    let attention = 37_888u64
        .checked_add(
            capacity
                .checked_mul((2 * 1024 + 128) * 4)
                .context("HAG KV/coefficient bytes overflow")?,
        )
        .context("HAG attention tensor bytes overflow")?;
    attention
        .checked_add(32_768)
        .and_then(|n| n.checked_mul(LAYERS))
        .and_then(|n| n.checked_add(8192 + DIM as u64 * 4 + LOGIT_BYTES))
        .context("HAG resident tensor bytes overflow")
}

fn model_staging_bytes(capacity: usize) -> Result<u64> {
    let coefficients = u64::try_from(capacity)?
        .checked_mul(128 * 4)
        .context("HAG coefficient staging overflows")?;
    Ok(STAGING_BYTES.max(coefficients))
}

fn preparation_staging_bytes(capacity: usize) -> Result<u64> {
    // Both F32 cosine/sine tables remain alive while upload_f32 converts one
    // half-table to bytes. Include that temporary in addition to model staging.
    let coefficients = u64::try_from(capacity)?
        .checked_mul(128 * 6)
        .context("HAG coefficient upload staging overflows")?;
    Ok(model_staging_bytes(capacity)?.max(coefficients))
}

fn attention_recording_bytes(visible: usize) -> Result<u64> {
    ensure!(
        (1..=MAX_CAPACITY).contains(&visible),
        "HAG recording position is outside serving capacity"
    );
    let attention = u64::try_from(visible)?
        .checked_mul(12)
        .and_then(|bytes| bytes.checked_add(4))
        .context("HAG attention scratch bytes overflow")?;
    // One workspace covers weighted norm/scaled-add and disjoint scores/probs.
    // Three immutable scalar patterns require twelve additional bytes.
    attention
        .max(DIM as u64 * 4)
        .checked_add(12)
        .context("HAG attention recording bytes overflow")
}

fn auxiliary_bytes(capacity: usize) -> Result<u64> {
    // The four-byte embedding dummy retires before attention recording. Exactly
    // one attention batch is live beside persistent FFN/head recordings.
    PERSISTENT_AUXILIARY_BYTES
        .checked_add(attention_recording_bytes(capacity)?)
        .context("HAG private auxiliary bytes overflow")
}

fn recording_write_bytes(visible: usize) -> Result<u64> {
    attention_recording_bytes(visible)?
        .checked_mul(LAYERS)
        .and_then(|bytes| bytes.checked_add(4))
        .context("HAG token recording writes overflow")
}

fn preparation_write_bytes(capacity: usize) -> Result<u64> {
    ensure!(
        (1..=MAX_CAPACITY).contains(&capacity),
        "HAG preparation capacity is outside scope"
    );
    let coefficients = u64::try_from(capacity)?
        .checked_mul(LAYERS * 128 * 4)
        .context("HAG coefficient upload bytes overflow")?;
    // upload_f32 first initializes storage, then writes supplied norms/tables.
    [
        PACKED_BYTES,
        tensor_bytes(capacity)?,
        PERSISTENT_AUXILIARY_BYTES,
        SOURCE_BYTES - PACKED_BYTES,
        coefficients,
    ]
    .into_iter()
    .try_fold(0u64, |sum, bytes| {
        sum.checked_add(bytes)
            .context("HAG preparation writes overflow")
    })
}

fn verify_token_accounting(
    capacity: usize,
    before: NativeExecutionStats,
    after: NativeExecutionStats,
    visible: usize,
) -> Result<()> {
    ensure!(
        visible <= capacity,
        "HAG visible positions exceed resident capacity"
    );
    let peak = before.auxiliary_peak_bytes.max(auxiliary_bytes(visible)?);
    ensure!(
        !before.poisoned
            && !after.poisoned
            && before.dispatch_count.checked_add(3793) == Some(after.dispatch_count)
            && before.dispatch_attempts.checked_add(3793) == Some(after.dispatch_attempts)
            && before.logical_submissions.checked_add(58) == Some(after.logical_submissions)
            && before
                .host_write_bytes
                .checked_add(recording_write_bytes(visible)?)
                == Some(after.host_write_bytes)
            && after.host_read_bytes == before.host_read_bytes
            && after.allocation_bytes == before.allocation_bytes
            && after.weight_live_bytes == PACKED_BYTES
            && after.tensor_live_bytes == tensor_bytes(capacity)?
            && before.auxiliary_live_bytes == PERSISTENT_AUXILIARY_BYTES
            && after.auxiliary_live_bytes == PERSISTENT_AUXILIARY_BYTES
            && after.auxiliary_peak_bytes == peak
            && peak <= auxiliary_bytes(capacity)?
            && after.weight_peak_bytes == before.weight_peak_bytes
            && after.tensor_peak_bytes == before.tensor_peak_bytes
            && after.uploaded_weight_bytes == before.uploaded_weight_bytes,
        "HAG serving token execution/traffic accounting differs"
    );
    Ok(())
}

fn resource_config(capacity: usize, memory: DenseModelMemory) -> Result<NativeExecutionConfig> {
    ensure!(
        (1..=MAX_CAPACITY).contains(&capacity),
        "HAG serving cost scope requires 1..={MAX_CAPACITY} positions"
    );
    ensure!(
        memory.packed_weight_bytes == PACKED_BYTES
            && memory.resident_tensor_bytes == tensor_bytes(capacity)?
            && memory.largest_staging_bytes == model_staging_bytes(capacity)?,
        "HAG dense model payload requirements differ from the admitted graph"
    );
    let config = NativeExecutionConfig {
        weight_capacity_bytes: u32::try_from(memory.packed_weight_bytes)?,
        tensor_capacity_bytes: u32::try_from(memory.resident_tensor_bytes)?,
        auxiliary_capacity_bytes: u32::try_from(auxiliary_bytes(capacity)?)?,
    };
    config.allocation_bytes()?;
    Ok(config)
}

fn recording_footprint(capacity: usize) -> Result<NativeRecordingFootprint> {
    ensure!(
        (1..=MAX_CAPACITY).contains(&capacity),
        "HAG metadata capacity is outside scope"
    );
    Ok(NativeRecordingFootprint {
        // Persist FFN/head. Conservatively count transient attention and lookup
        // together even though their execution scopes never overlap.
        batches: LAYERS + 3,
        dispatches: LAYERS * 7 + 40 + 127 + 1,
        weight_allocations: LAYERS * 7 + 38,
        tensor_allocations: LAYERS * (12 + 4) + 4,
        auxiliary_allocations: LAYERS * 4 + 2 + 4 + 1,
        retained_allocation_references: LAYERS * 12 + 44 + 21 + 3,
        noop_operands: 0,
        max_batch_commands: 54,
        max_batch_dispatches: 127,
    })
}

fn host_metadata_bytes(capacity: usize, retained_attention_metadata: u64) -> Result<u64> {
    ensure!(
        retained_attention_metadata <= RETAINED_ATTENTION_METADATA_BOUND,
        "HAG retained attention metadata exceeds the fixed-geometry bound"
    );
    let footprint = recording_footprint(capacity)?;
    // Native ownership uses the backend's real private layouts. Model-side
    // tables use public owner/plan layouts, twofold Vec spare capacity, and
    // 64 bytes per heap allocation. Count simultaneous preparation plans and
    // resident handles rather than only the smaller post-preparation owner.
    let layer_entry = (std::mem::size_of::<DenseAttentionPlan>()
        + std::mem::size_of::<DenseFfnBlockPlan>()
        + std::mem::size_of::<ResidentDenseAttention>()
        + std::mem::size_of::<RecordedBatch>()
        + 32) as u64;
    let model_entries = [
        (
            1,
            (std::mem::size_of::<DenseModelPlan>() + std::mem::size_of::<ResidentDenseModel>())
                as u64,
        ),
        (2 * LAYERS, layer_entry),
        (
            2 * footprint.batches,
            std::mem::size_of::<RecordedBatch>() as u64,
        ),
        (
            2 * 38,
            (std::mem::size_of::<VocabularyTile>() + std::mem::size_of::<PreparedWeight>()) as u64,
        ),
        (
            2,
            (std::mem::size_of::<PreparedDenseAttention>()
                + std::mem::size_of::<PreparedDenseFfnBlock>()
                + std::mem::size_of::<PreparedVocabulary>()
                + std::mem::size_of::<VocabularyPlan>()) as u64,
        ),
        // Bound tensor clones own at most two dimension words in this strict
        // graph. Four copies of every source span conservatively cover model
        // binding, full plan and temporary attention/FFN preparation clones.
        (4 * PREPARATION_READS, 2 * 8 + 64),
        (2 * LAYERS + 12, 64),
    ];
    let bytes = model_entries.into_iter().try_fold(
        footprint
            .host_bytes()?
            .checked_add(retained_attention_metadata)
            .context("HAG retained model metadata overflows")?,
        |sum, (count, size)| {
            count
                .checked_mul(size)
                .and_then(|n| sum.checked_add(n))
                .context("HAG model host metadata overflows")
        },
    )?;
    // Page-independent policy rounding keeps a visible margin above the counted
    // layouts. This estimate is additional to general runtime/allocator headroom.
    let mib = 1024 * 1024;
    bytes
        .checked_add(mib - 1)
        .map(|n| n / mib * mib)
        .context("HAG rounded host metadata overflows")
}

fn host_and_device_forecast(
    capacity: usize,
    checkpoint_bytes: u64,
    allocation: u64,
    headroom: u64,
    retained_attention_metadata: u64,
) -> Result<u64> {
    // Count the complete file as a conservative page-cache/metadata allowance in
    // addition to counted recording metadata and runtime headroom. This never retains that file
    // as a Rust allocation. GPU bytes are also charged here for shared UMA pressure.
    [
        allocation,
        checkpoint_bytes,
        preparation_staging_bytes(capacity)?,
        LOGIT_BYTES,
        u64::try_from(capacity)?
            .checked_mul(8)
            .context("HAG token storage overflows")?,
        host_metadata_bytes(capacity, retained_attention_metadata)?,
        headroom.max(MINIMUM_HEADROOM),
    ]
    .into_iter()
    .try_fold(0u64, |sum, n| {
        sum.checked_add(n)
            .context("HAG combined memory forecast overflows")
    })
}

fn admit_cost(required: u64, budget: u64, available: u64) -> Result<()> {
    ensure!(
        required <= budget,
        "HAG full resident forecast requires {required} bytes, exceeding the configured process budget {budget}; streaming/degradation is unsupported"
    );
    ensure!(
        required <= available,
        "HAG full resident forecast requires {required} bytes, but host/cgroup available memory is {available}"
    );
    Ok(())
}

/// Explicit attestation that the current unified cgroup is an owned, dedicated
/// serving job. The launcher must verify that scope before setting this variable;
/// matching a path alone does not establish that unrelated processes are absent.
/// Unattested callers retain conservative admission against remaining memory.
const PRIVATE_CGROUP_ENV: &str = "BEHOLDD_HAG_PRIVATE_CGROUP";

#[derive(Clone, Copy)]
struct ChargeSample {
    current: u64,
    peak: u64,
}

#[derive(Clone, Copy)]
struct PrivateJobMemory {
    host_available: u64,
    ancestor_remaining: Option<u64>,
    before: Option<ChargeSample>,
    after: Option<ChargeSample>,
}

fn validate_private_membership(attested: &str, membership: &str) -> Result<()> {
    ensure!(
        attested.starts_with('/')
            && attested.len() > 1
            && attested
                .split('/')
                .skip(1)
                .all(|part| !part.is_empty() && part != "." && part != ".."),
        "HAG private cgroup attestation must be a normal, non-root unified membership path"
    );
    let mut lines = membership.lines();
    ensure!(
        lines.next().and_then(|line| line.strip_prefix("0::")) == Some(attested)
            && lines.next().is_none(),
        "HAG private cgroup attestation differs from the current unified membership"
    );
    Ok(())
}

fn validate_private_scope(
    attested: &str,
    before_membership: &str,
    before_mountinfo: &str,
    after_membership: &str,
    after_mountinfo: &str,
) -> Result<()> {
    validate_private_membership(attested, before_membership)?;
    validate_private_membership(attested, after_membership)?;
    ensure!(
        !before_mountinfo.is_empty()
            && before_membership == after_membership
            && before_mountinfo == after_mountinfo,
        "HAG private cgroup membership or mount identity changed during admission"
    );
    Ok(())
}

fn admit_private_cost(
    required: u64,
    budget: u64,
    checkpoint_bytes: u64,
    memory: PrivateJobMemory,
) -> Result<()> {
    // Host availability and the configured full-job budget receive no credit.
    admit_cost(required, budget, memory.host_available)?;
    let remaining = memory
        .ancestor_remaining
        .context("HAG private job requires finite cgroup headroom")?;
    let before = memory
        .before
        .context("missing HAG private cgroup pre-sample")?;
    let after = memory
        .after
        .context("missing HAG private cgroup post-sample")?;
    for sample in [before, after] {
        ensure!(
            sample.current <= sample.peak && sample.peak <= budget,
            "HAG private cgroup charge is invalid or exceeds the configured process budget"
        );
    }
    ensure!(
        after.peak >= before.peak,
        "HAG private cgroup peak decreased while sampling"
    );
    // Only this attested leaf's smaller bracketing current charge is credited,
    // capped by the complete checkpoint allowance already in the full forecast.
    // Ancestor current use and historical peaks never contribute credit.
    let credit = before.current.min(after.current).min(checkpoint_bytes);
    let total_allowance = remaining
        .checked_add(credit)
        .context("HAG private total allowance overflows")?;
    ensure!(
        required <= total_allowance,
        "HAG full resident forecast exceeds private-job total allowance"
    );
    // This is a separate remaining-memory reservation. Uncharged checkpoint
    // portions still have to fit through the full-forecast check above; other
    // native allocations cannot spend the credited bytes a second time.
    let additional = required
        .checked_sub(checkpoint_bytes)
        .context("HAG additional forecast underflows")?;
    ensure!(
        additional <= memory.host_available.min(remaining),
        "HAG additional allocations exceed live remaining host/cgroup headroom"
    );
    Ok(())
}

pub(super) fn admit_memory(required: u64, budget: u64, checkpoint_bytes: u64) -> Result<()> {
    let Some(attested) = std::env::var_os(PRIVATE_CGROUP_ENV) else {
        let available =
            available_memory().context("discovering HAG host/cgroup memory headroom")?;
        return admit_cost(required, budget, available.effective_available_bytes());
    };
    let attested = attested
        .to_str()
        .context("HAG private cgroup attestation is not UTF-8")?;
    let before_membership = std::fs::read_to_string("/proc/self/cgroup")?;
    let before_mountinfo = std::fs::read_to_string("/proc/self/mountinfo")?;
    validate_private_membership(attested, &before_membership)?;
    let sample = || -> Result<Option<ChargeSample>> {
        Ok(cgroup_memory()?.map(|value| ChargeSample {
            current: value.current_bytes(),
            peak: value.peak_bytes(),
        }))
    };
    let before = sample()?;
    let available = available_memory().context("discovering HAG private-job memory headroom")?;
    let after = sample()?;
    validate_private_scope(
        attested,
        &before_membership,
        &before_mountinfo,
        &std::fs::read_to_string("/proc/self/cgroup")?,
        &std::fs::read_to_string("/proc/self/mountinfo")?,
    )?;
    admit_private_cost(
        required,
        budget,
        checkpoint_bytes,
        PrivateJobMemory {
            host_available: available.host_available_bytes(),
            ancestor_remaining: available.cgroup_available_bytes(),
            before,
            after,
        },
    )
}

/// A private direct-read source polls serving control between sequential reads.
/// Neither successful reads nor cancellation create a checkpoint payload cache.
struct ControlledSource<'a, 'b> {
    source: ResidentSource<'a>,
    control: Mutex<&'b mut (dyn FnMut() -> Result<()> + Send)>,
    reads: AtomicU64,
    bytes: AtomicU64,
}
impl WeightSource for ControlledSource<'_, '_> {
    fn checkpoint_id(&self) -> CheckpointId {
        self.source.checkpoint_id()
    }
    fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
        let mut control = self
            .control
            .lock()
            .map_err(|_| anyhow::anyhow!("HAG load control is poisoned"))?;
        control().context("checking HAG control before reading a weight")?;
        let loaded = self.source.load_range(range, access)?;
        control().context("checking HAG control after reading a weight")?;
        self.reads.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(range.nbytes(), Ordering::Relaxed);
        Ok(loaded)
    }
}

// Created outside the preparation closure, so its locals retire before this
// guard observes retained/quarantined charges on an error or unwind.
struct PreparationCleanup<'a> {
    budget: &'a MemoryBudget,
    active: bool,
}
impl Drop for PreparationCleanup<'_> {
    fn drop(&mut self) {
        if self.active && self.budget.current_bytes() != 0 {
            process_health().store(false, Ordering::Release);
        }
    }
}

/// One loaded HAG model. ResidentDenseModel is Send but intentionally not Sync;
/// the parent runtime holds this complete owner behind its execution mutex.
pub(crate) struct HagExecution {
    resident: Option<ResidentDenseModel>,
    device: Option<NativeExecutionDevice>,
    budget: MemoryBudget,
    healthy: Arc<AtomicBool>,
    logits: Vec<f32>,
    capacity: usize,
}

impl HagExecution {
    pub(crate) fn prepare(
        model: &QwenModel,
        checkpoint: &Arc<Gguf>,
        config: &ModelConfig,
        memory_budget_bytes: u64,
        runtime_headroom_bytes: u64,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        phases: &Recorder,
    ) -> Result<Self> {
        phases.emit(Phase::PreparationStart);
        let healthy = process_health();
        ensure!(
            healthy.load(Ordering::Acquire),
            "HAG execution is disabled after device uncertainty; restart the process"
        );
        let hag = config
            .hag
            .as_ref()
            .context("HAG backend requires explicit device/budget configuration")?;
        let capacity = hag.context_capacity;
        ensure!(
            (1..=MAX_CAPACITY).contains(&capacity),
            "unsupported HAG context capacity"
        );
        ensure!(
            std::env::var_os("HAG_TRACE_SUBMISSION").as_deref() == Some(std::ffi::OsStr::new("0")),
            "HAG serving requires HAG_TRACE_SUBMISSION exactly 0"
        );
        control().context("checking HAG control before metadata planning")?;
        let shape = model.config();
        ensure!(
            model.architecture() == QwenArchitecture::Qwen3
                && (
                    shape.n_layers(),
                    shape.dim(),
                    shape.n_heads(),
                    shape.n_kv_heads(),
                    shape.head_dim(),
                    shape.feed_forward_length(),
                    shape.vocab(),
                    shape.n_experts()
                ) == (28, 1024, 16, 8, 128, 3072, 151936, 0)
                && shape.context() >= capacity
                && shape.eps().to_bits() == 1e-6f32.to_bits()
                && shape.rope_base() == 1_000_000.0,
            "HAG serving currently requires the qualified dense Qwen3-0.6B Q8_0 geometry, epsilon and RoPE base"
        );
        let plan = model.dense_model_plan_on_demand(capacity, TILE_ROWS)?;
        ensure!(
            plan.requires_on_demand_attention(),
            "HAG requires on-demand model planning"
        );
        let retained_attention_metadata = plan.on_demand_attention_host_metadata_bytes()?;
        let model_checkpoint = model.vocabulary_plan(TILE_ROWS)?.tiles()[0]
            .range()
            .checkpoint_id();
        ensure!(
            model_checkpoint == checkpoint.checkpoint_id(),
            "HAG model and checkpoint source belong to different checkpoints"
        );
        ensure!(
            model.weight_memory()?.checkpoint_bytes() == SOURCE_BYTES,
            "HAG unique checkpoint payload differs"
        );
        let native_config = resource_config(capacity, plan.memory())?;
        let allocation = native_config.allocation_bytes()?;
        ensure!(
            allocation <= hag.memory_budget_bytes,
            "HAG queue requires {allocation} bytes, exceeding the configured HAG budget {}",
            hag.memory_budget_bytes
        );
        let checkpoint_bytes = checkpoint.shards().iter().try_fold(0u64, |sum, path| {
            let bytes = std::fs::metadata(path)
                .with_context(|| format!("checking HAG checkpoint file size {}", path.display()))?
                .len();
            sum.checked_add(bytes)
                .context("HAG checkpoint file total overflows")
        })?;
        let required = host_and_device_forecast(
            capacity,
            checkpoint_bytes,
            allocation,
            runtime_headroom_bytes,
            retained_attention_metadata,
        )?;
        phases.emit(Phase::MemoryAdmissionStart);
        admit_memory(required, memory_budget_bytes, checkpoint_bytes)?;
        phases.emit(Phase::MemoryAdmissionEnd);
        control().context("checking HAG control before device opening")?;
        let budget = MemoryBudget::new(hag.memory_budget_bytes);
        let mut cleanup = PreparationCleanup {
            budget: &budget,
            active: true,
        };
        let prepared = (|| -> Result<(NativeExecutionDevice, ResidentDenseModel, Vec<f32>)> {
            phases.emit(Phase::DeviceOpenStart);
            let device = NativeExecutionDevice::open(&hag.device, &budget, native_config)?;
            phases.emit(Phase::DeviceOpenEnd);
            let result: Result<(ResidentDenseModel, Vec<f32>)> = (|| {
                control().context("checking HAG control before preparation")?;
                let source = ControlledSource {
                    source: ResidentSource::new(checkpoint.as_ref()),
                    control: Mutex::new(&mut *control),
                    reads: AtomicU64::new(0),
                    bytes: AtomicU64::new(0),
                };
                let memory = plan.memory();
                phases.emit(Phase::ResidentPreparationStart);
                let resident = plan.prepare_on_demand_attention(
                    &device,
                    &source,
                    DenseModelLimits {
                        packed_weight_bytes: memory.packed_weight_bytes,
                        resident_tensor_bytes: memory.resident_tensor_bytes,
                        largest_staging_bytes: memory.largest_staging_bytes,
                    },
                )?;
                phases.emit(Phase::ResidentPreparationEnd);
                ensure!(
                    source.reads.load(Ordering::Relaxed) == PREPARATION_READS
                        && source.bytes.load(Ordering::Relaxed) == SOURCE_BYTES,
                    "HAG preparation source accounting differs"
                );
                let stats = device.stats()?;
                ensure!(
                    !stats.poisoned
                        && stats.allocation_bytes == allocation
                        && stats.weight_live_bytes == PACKED_BYTES
                        && stats.uploaded_weight_bytes == PACKED_BYTES
                        && stats.tensor_live_bytes == memory.resident_tensor_bytes
                        && stats.auxiliary_live_bytes == PERSISTENT_AUXILIARY_BYTES
                        && stats.auxiliary_peak_bytes == PERSISTENT_AUXILIARY_BYTES
                        && stats.host_write_bytes == preparation_write_bytes(capacity)?
                        && stats.host_read_bytes == 0
                        && stats.dispatch_count == 0
                        && stats.dispatch_attempts == 0
                        && stats.logical_submissions == 0
                        && resident.records_attention_on_demand()
                        && resident.capacity() == capacity
                        && resident.visible_positions() == 0
                        && !resident.is_poisoned(),
                    "HAG prepared resource accounting differs"
                );
                let mut logits = Vec::new();
                logits
                    .try_reserve_exact(VOCAB)
                    .context("allocating HAG logits readback")?;
                logits.resize(VOCAB, 0.0);
                Ok((resident, logits))
            })();
            if device.stats().map(|stats| stats.poisoned).unwrap_or(true) {
                healthy.store(false, Ordering::Release);
            }
            let (resident, logits) = result?;
            Ok((device, resident, logits))
        })();
        let (device, resident, logits) = prepared?;
        cleanup.active = false;
        drop(cleanup);
        let owner = Self {
            resident: Some(resident),
            device: Some(device),
            budget,
            healthy,
            logits,
            capacity,
        };
        // Cancellation here drops the complete unpublished owner in the explicit
        // model/device order, with the same retained-budget check as normal unload.
        control().context("checking HAG control after preparation")?;
        phases.emit(Phase::PreparationEnd);
        Ok(owner)
    }

    pub(crate) fn health(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.healthy)
    }

    pub(crate) fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        eos: u32,
        control: &mut (dyn FnMut() -> Result<()> + Send),
        phases: &Recorder,
        mut on_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenerateOutput> {
        let capacity = self.capacity;
        validate_request(capacity, prompt.len(), max_new_tokens)?;
        ensure!(
            prompt.iter().all(|&token| (token as usize) < VOCAB) && (eos as usize) < VOCAB,
            "HAG prompt/stop token is outside vocabulary"
        );
        ensure!(
            self.healthy.load(Ordering::Acquire),
            "HAG execution is disabled after device uncertainty; restart the process"
        );
        let device = self.device.as_ref().context("HAG device is unavailable")?;
        let resident = self.resident.as_mut().context("HAG model is unavailable")?;
        let stats = device.stats().inspect_err(|_| {
            self.healthy.store(false, Ordering::Release);
        })?;
        if resident.is_poisoned() || stats.poisoned {
            self.healthy.store(false, Ordering::Release);
            bail!("HAG model/device is poisoned; restart the process");
        }
        let mut session = RequestSession::new(resident, &self.healthy, phases)?;
        let audit_sequence = (std::env::var_os("HAG_SERVING_AUDIT").as_deref()
            == Some(std::ffi::OsStr::new("1")))
        .then(|| AUDIT_SEQUENCE.fetch_add(1, Ordering::Relaxed));
        let callback_failed = std::cell::Cell::new(false);
        let mut observed_control = || {
            let result = control();
            callback_failed.set(callback_failed.get() || result.is_err());
            result
        };
        let observed_sink = |token| {
            let result = on_token(token);
            callback_failed.set(callback_failed.get() || result.is_err());
            result
        };
        let mut evaluated_positions = 0usize;
        let logits = &mut self.logits;
        let result = generate_tokens(
            capacity,
            prompt,
            max_new_tokens,
            eos,
            &mut observed_control,
            observed_sink,
            |token, select| {
                let result = (|| {
                    let before = device.stats()?;
                    session.resident.step(device, token)?;
                    evaluated_positions += 1;
                    phases.position_fenced(evaluated_positions);
                    let after = device.stats()?;
                    verify_token_accounting(
                        capacity,
                        before,
                        after,
                        session.resident.visible_positions(),
                    )?;
                    if select {
                        session.resident.read_logits(device, logits)?;
                        Ok(Some(greedy_token(logits)?))
                    } else {
                        Ok(None)
                    }
                })();
                if result.is_err() {
                    session.healthy.store(false, Ordering::Release);
                }
                result
            },
            |phase| phases.emit(phase),
        );
        let cleanup = session.finish();
        if let Some(request_sequence) = audit_sequence {
            let poisoned =
                session.resident.is_poisoned() || !session.healthy.load(Ordering::Acquire);
            let event = RequestAudit::new(
                request_sequence,
                evaluated_positions,
                result.is_ok(),
                callback_failed.get(),
                cleanup.is_ok(),
                poisoned,
            );
            // Explicit qualification telemetry contains only lifecycle counters;
            // it never changes cancellation boundaries or emits prompt/token data.
            eprintln!("HAG_SERVING_AUDIT {event}");
        }
        match (result, cleanup) {
            (Ok(output), Ok(())) => Ok(output),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Err(cleanup)) => {
                Err(error.context(format!("HAG request cleanup also failed: {cleanup:#}")))
            }
        }
    }
}

#[derive(Debug)]
struct RequestAudit {
    request_sequence: u64,
    evaluated_positions: usize,
    outcome: &'static str,
    reset_ok: bool,
    poisoned: bool,
}

impl std::fmt::Display for RequestAudit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // All fields are integers, booleans or a fixed internal enum spelling.
        write!(f, "{{\"request_sequence\":{},\"evaluated_positions\":{},\"outcome\":\"{}\",\"reset_ok\":{},\"poisoned\":{}}}",
            self.request_sequence, self.evaluated_positions, self.outcome, self.reset_ok, self.poisoned)
    }
}

impl RequestAudit {
    fn new(
        request_sequence: u64,
        evaluated_positions: usize,
        succeeded: bool,
        callback_failed: bool,
        reset_ok: bool,
        poisoned: bool,
    ) -> Self {
        let outcome = if succeeded && reset_ok && !poisoned {
            "success"
        } else if callback_failed && reset_ok && !poisoned {
            "cancelled"
        } else {
            "failed"
        };
        Self {
            request_sequence,
            evaluated_positions,
            outcome,
            reset_ok,
            poisoned,
        }
    }
}

impl Drop for HagExecution {
    fn drop(&mut self) {
        if self
            .resident
            .as_ref()
            .is_some_and(ResidentDenseModel::is_poisoned)
            || self
                .device
                .as_ref()
                .is_some_and(|device| device.stats().map(|stats| stats.poisoned).unwrap_or(true))
        {
            self.healthy.store(false, Ordering::Release);
        }
        drop(self.resident.take());
        drop(self.device.take());
        if self.budget.current_bytes() != 0 {
            self.healthy.store(false, Ordering::Release);
        }
    }
}

struct RequestSession<'a> {
    resident: &'a mut ResidentDenseModel,
    healthy: &'a AtomicBool,
    active: bool,
    phases: &'a Recorder,
}
impl<'a> RequestSession<'a> {
    fn new(
        resident: &'a mut ResidentDenseModel,
        healthy: &'a AtomicBool,
        phases: &'a Recorder,
    ) -> Result<Self> {
        if let Err(error) = session_reset(true, || resident.reset(), |phase| phases.emit(phase)) {
            healthy.store(false, Ordering::Release);
            return Err(error);
        }
        Ok(Self {
            resident,
            healthy,
            active: true,
            phases,
        })
    }
    fn finish(&mut self) -> Result<()> {
        self.active = false;
        let phases = self.phases;
        let result = session_reset(false, || self.resident.reset(), |phase| phases.emit(phase));
        if result.is_err() {
            self.healthy.store(false, Ordering::Release);
        }
        result
    }
}
impl Drop for RequestSession<'_> {
    fn drop(&mut self) {
        if self.active {
            let _ = self.finish();
        }
    }
}

// Pure serving schedule: callbacks run only outside whole synchronous evaluations.
// The last emitted token is not evaluated, matching the permanent CPU API.
fn generate_tokens(
    capacity: usize,
    prompt: &[u32],
    max_new_tokens: usize,
    eos: u32,
    control: &mut dyn FnMut() -> Result<()>,
    mut emit: impl FnMut(u32) -> Result<()>,
    mut evaluate: impl FnMut(u32, bool) -> Result<Option<u32>>,
    mut phase: impl FnMut(Phase),
) -> Result<GenerateOutput> {
    validate_request(capacity, prompt.len(), max_new_tokens)?;
    let mut generated = Vec::new();
    generated
        .try_reserve_exact(max_new_tokens)
        .context("allocating HAG generated token IDs")?;
    let mut selected = None;
    phase(Phase::PrefillStart);
    for (index, &token) in prompt.iter().enumerate() {
        control().context("checking HAG control before a prompt token")?;
        selected = evaluate(token, index + 1 == prompt.len())?;
        control().context("checking HAG control after a prompt token")?;
    }
    let mut token = selected.context("HAG prefill did not return logits")?;
    phase(Phase::PrefillEnd);
    let mut decode_started = false;
    for step in 0..max_new_tokens {
        control().context("checking HAG control before token delivery")?;
        if token == eos {
            if decode_started {
                phase(Phase::DecodeEnd);
            }
            return Ok(GenerateOutput {
                generated,
                stop_token: Some(token),
                finish_reason: GenerationFinishReason::StopToken,
            });
        }
        emit(token).context("delivering HAG generated token")?;
        generated.push(token);
        if step + 1 < max_new_tokens {
            if !decode_started {
                phase(Phase::DecodeStart);
                decode_started = true;
            }
            control().context("checking HAG control before a decode token")?;
            token = evaluate(token, true)?.context("HAG decode did not return logits")?;
            control().context("checking HAG control after a decode token")?;
        }
    }
    if decode_started {
        phase(Phase::DecodeEnd);
    }
    Ok(GenerateOutput {
        generated,
        stop_token: None,
        finish_reason: GenerationFinishReason::MaxNewTokens,
    })
}

fn greedy_token(logits: &[f32]) -> Result<u32> {
    let (&first, rest) = logits
        .split_first()
        .context("cannot select from empty HAG logits")?;
    ensure!(first.is_finite(), "HAG logit0 is not finite");
    let mut best = (0usize, first);
    for (offset, &value) in rest.iter().enumerate() {
        ensure!(value.is_finite(), "HAG logit{} is not finite", offset + 1);
        if value > best.1 {
            best = (offset + 1, value);
        }
    }
    u32::try_from(best.0).context("HAG selected token does not fit u32")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configurable_capacity_schedule_preserves_the_last_token_boundary() {
        for capacity in [1, 2048, 8192, 16384] {
            validate_request(capacity, capacity, 1).unwrap();
            assert!(validate_request(capacity, capacity, 2).is_err());
            assert!(validate_request(capacity, capacity + 1, 1).is_err());
            let prompt = vec![1; capacity];
            let mut evaluated = 0;
            let output = generate_tokens(
                capacity,
                &prompt,
                1,
                99,
                &mut || Ok(()),
                |_| Ok(()),
                |_, select| {
                    evaluated += 1;
                    Ok(select.then_some(2))
                },
                |_| {},
            )
            .unwrap();
            assert_eq!(evaluated, capacity);
            assert_eq!(output.generated, [2]);
        }
        for capacity in [0, MAX_CAPACITY + 1, usize::MAX] {
            assert!(validate_request(capacity, 1, 1).is_err());
        }
    }

    #[test]
    fn larger_context_forecasts_include_coefficient_upload_and_uma_allocation() {
        for (capacity, tensors, staging, allocation, forecast) in [
            (8192, 1_999_087_104, 6_291_456, 2_632_581_120, 3_414_259_904),
            (
                16384,
                3_995_575_808,
                12_582_912,
                4_629_168_128,
                5_417_203_904,
            ),
        ] {
            assert_eq!(tensor_bytes(capacity).unwrap(), tensors);
            assert_eq!(preparation_staging_bytes(capacity).unwrap(), staging);
            let config = resource_config(
                capacity,
                DenseModelMemory {
                    packed_weight_bytes: PACKED_BYTES,
                    resident_tensor_bytes: tensors,
                    largest_staging_bytes: model_staging_bytes(capacity).unwrap(),
                },
            )
            .unwrap();
            assert_eq!(config.allocation_bytes().unwrap(), allocation);
            let required =
                host_and_device_forecast(capacity, 639_447_744, allocation, 0, 64_960).unwrap();
            assert_eq!(required, forecast);
            assert!(admit_cost(required, 2 << 30, u64::MAX).is_err());
            admit_cost(required, 6 << 30, 8 << 30).unwrap();
            assert!(admit_cost(required, 6 << 30, required - 1).is_err());
            let default_headroom = host_and_device_forecast(
                capacity,
                639_447_744,
                allocation,
                crate::config::DEFAULT_RUNTIME_HEADROOM_BYTES,
                64_960,
            )
            .unwrap();
            assert!(default_headroom > required);
            println!("HAG_CAPACITY_FORECAST {{\"capacity\":{capacity},\"tensor_bytes\":{tensors},\"staging_peak_bytes\":{staging},\"native_allocation_bytes\":{allocation},\"combined_minimum_headroom_bytes\":{required},\"combined_default_headroom_bytes\":{default_headroom}}}");
        }
    }
    use std::cell::Cell;

    #[test]
    fn request_capacity_accounts_for_unevaluated_final_output_and_overflow() {
        for (prompt, generated) in [
            (2048, 1),
            (1, 2048),
            (9, 2040),
            (2047, 2),
            (93, 128),
            (1024, 512),
            (467, 128),
            (467, 1582),
        ] {
            validate_request(CAPACITY, prompt, generated).unwrap();
        }
        assert_eq!(
            validate_request(CAPACITY, 0, 1)
                .unwrap_err()
                .downcast_ref::<crate::RequestError>(),
            Some(&crate::RequestError::EmptyPrompt)
        );
        assert_eq!(
            validate_request(CAPACITY, 1, 0)
                .unwrap_err()
                .downcast_ref::<crate::RequestError>(),
            Some(&crate::RequestError::InvalidOutputLimit)
        );
        for (prompt, generated) in [
            (2048, 2),
            (1, 2049),
            (9, 2041),
            (2049, 1),
            (467, 1583),
            (usize::MAX, 2),
        ] {
            assert_eq!(
                validate_request(CAPACITY, prompt, generated)
                    .unwrap_err()
                    .downcast_ref::<crate::RequestError>(),
                Some(&crate::RequestError::ContextCapacity {
                    prompt_tokens: prompt,
                    max_new_tokens: generated,
                    capacity: CAPACITY
                })
            );
        }
    }

    #[test]
    fn on_demand_costs_are_bounded_before_payloads() {
        for (capacity, tensors, auxiliary, allocation) in [
            (8, 4_548_096, 123_232, 637_947_904),
            (16, 6_497_792, 123_232, 639_897_600),
            (64, 18_195_968, 123_232, 651_595_776),
            (128, 33_793_536, 123_232, 667_193_344),
            (341, 85_704_192, 123_232, 719_101_952),
            (342, 85_947_904, 123_244, 719_347_712),
            (512, 127_378_944, 125_284, 760_778_752),
            (2048, 501_720_576, 143_716, 1_135_140_864),
        ] {
            let config = resource_config(
                capacity,
                DenseModelMemory {
                    packed_weight_bytes: PACKED_BYTES,
                    resident_tensor_bytes: tensors,
                    largest_staging_bytes: STAGING_BYTES,
                },
            )
            .unwrap();
            assert_eq!(u64::from(config.auxiliary_capacity_bytes), auxiliary);
            assert_eq!(config.allocation_bytes().unwrap(), allocation);
            assert!(allocation < 1152 * 1024 * 1024);
            if capacity <= 512 {
                assert!(allocation < 768 * 1024 * 1024);
            }
        }
        assert_eq!(PERSISTENT_AUXILIARY_BYTES, 119_124);
        assert_eq!(preparation_write_bytes(CAPACITY).unwrap(), 1_164_695_380);
        // The independently frozen C2048 model inspection measured 64,960 bytes
        // for retained attention owners on this 64-bit target. Keep the serving
        // formula consistent with that evidence without opening a checkpoint.
        #[cfg(target_pointer_width = "64")]
        {
            assert_eq!(host_metadata_bytes(CAPACITY, 64_960).unwrap(), 1_048_576);
            assert_eq!(
                host_and_device_forecast(CAPACITY, 639_447_744, 1_135_140_864, 0, 64_960).unwrap(),
                1_914_935_488
            );
        }
        // No checkpoint is opened: use a one-MiB upper bound for retained
        // attention preparations. Real preparation supplies its actual API cost.
        let required = host_and_device_forecast(
            CAPACITY,
            639_447_744,
            1_135_140_864,
            0,
            RETAINED_ATTENTION_METADATA_BOUND,
        )
        .unwrap();
        println!(
            "HAG_CONTEXT2048_COST auxiliary_peak_bytes={} hag_charge_bytes={} native_metadata_bytes={} retained_attention_metadata_bound={} standalone_metadata_allowance_bytes={} standalone_forecast_bytes={}",
            auxiliary_bytes(CAPACITY).unwrap(),
            1_135_140_864u64,
            recording_footprint(CAPACITY).unwrap().host_bytes().unwrap(),
            RETAINED_ATTENTION_METADATA_BOUND,
            host_metadata_bytes(CAPACITY, RETAINED_ATTENTION_METADATA_BOUND).unwrap(),
            required
        );
        assert_eq!(
            host_and_device_forecast(
                CAPACITY,
                639_447_744,
                1_135_140_864,
                MINIMUM_HEADROOM,
                RETAINED_ATTENTION_METADATA_BOUND
            )
            .unwrap(),
            required
        );
        assert_eq!(
            required,
            1_913_886_912
                + host_metadata_bytes(CAPACITY, RETAINED_ATTENTION_METADATA_BOUND).unwrap()
        );
        assert!(required < 2 * 1024 * 1024 * 1024);
        assert!(host_and_device_forecast(CAPACITY, u64::MAX, 1, 0, 0).is_err());
        assert!(host_and_device_forecast(CAPACITY, 1, 1, u64::MAX, 0).is_err());
        assert!(host_and_device_forecast(CAPACITY, 1, 1, 0, u64::MAX).is_err());
        admit_cost(required, required, required).unwrap();
        assert!(admit_cost(required, required - 1, u64::MAX).is_err());
        assert!(admit_cost(required, u64::MAX, required - 1).is_err());
        assert!(resource_config(
            512,
            DenseModelMemory {
                packed_weight_bytes: PACKED_BYTES + 1,
                resident_tensor_bytes: 127_378_944,
                largest_staging_bytes: STAGING_BYTES
            }
        )
        .is_err());
        for capacity in [0, MAX_CAPACITY + 1, usize::MAX] {
            assert!(recording_footprint(capacity).is_err());
            assert!(attention_recording_bytes(capacity).is_err());
            assert!(recording_write_bytes(capacity).is_err());
            assert!(preparation_write_bytes(capacity).is_err());
            assert!(host_metadata_bytes(capacity, 0).is_err());
            assert!(resource_config(
                capacity,
                DenseModelMemory {
                    packed_weight_bytes: PACKED_BYTES,
                    resident_tensor_bytes: 127_378_944,
                    largest_staging_bytes: STAGING_BYTES,
                }
            )
            .is_err());
        }
        #[cfg(target_pointer_width = "64")]
        assert!(tensor_bytes(usize::MAX).is_err());
    }

    fn private_memory(current: u64) -> PrivateJobMemory {
        let sample = ChargeSample {
            current,
            peak: current,
        };
        PrivateJobMemory {
            host_available: 8 * 1024 * 1024 * 1024,
            ancestor_remaining: Some(2 * 1024 * 1024 * 1024 - current),
            before: Some(sample),
            after: Some(sample),
        }
    }

    fn serving_forecast() -> u64 {
        host_and_device_forecast(
            CAPACITY,
            639_447_744,
            1_135_140_864,
            MINIMUM_HEADROOM,
            RETAINED_ATTENTION_METADATA_BOUND,
        )
        .unwrap()
    }

    #[test]
    fn private_job_admission_handles_cold_and_warm_reload_without_ancestor_credit() {
        let required = serving_forecast();
        let budget = 2 * 1024 * 1024 * 1024;
        let checkpoint = 639_447_744;
        for current in [32 * 1024 * 1024, checkpoint + 96 * 1024 * 1024] {
            admit_private_cost(required, budget, checkpoint, private_memory(current)).unwrap();
        }
        let warm = private_memory(checkpoint + 96 * 1024 * 1024);
        // The old remaining-only check double counts the already charged file.
        assert!(admit_cost(required, budget, warm.ancestor_remaining.unwrap()).is_err());
        // A busy ancestor is not this private job. Its tight remaining allowance
        // may receive only the sampled leaf's smaller current charge.
        let small_leaf = private_memory(32 * 1024 * 1024);
        assert!(admit_private_cost(
            required,
            budget,
            checkpoint,
            PrivateJobMemory {
                ancestor_remaining: Some(required - 32 * 1024 * 1024 - 1),
                ..small_leaf
            }
        )
        .is_err());
        // Neither a larger historical peak nor unrelated bytes beyond the file
        // allowance can be credited to make a tight ancestor appear to fit.
        assert!(admit_private_cost(
            required,
            budget,
            checkpoint,
            PrivateJobMemory {
                before: Some(ChargeSample {
                    current: 32 * 1024 * 1024,
                    peak: checkpoint
                }),
                after: Some(ChargeSample {
                    current: 32 * 1024 * 1024,
                    peak: checkpoint
                }),
                ancestor_remaining: Some(required - checkpoint),
                ..small_leaf
            }
        )
        .is_err());
        assert!(admit_private_cost(
            required,
            budget,
            checkpoint,
            PrivateJobMemory {
                ancestor_remaining: Some(required - checkpoint - 1),
                ..private_memory(checkpoint + 256 * 1024 * 1024)
            }
        )
        .is_err());
    }

    #[test]
    fn private_job_admission_uses_smaller_bracket_and_refuses_missing_or_invalid_samples() {
        let required = serving_forecast();
        let budget = 2 * 1024 * 1024 * 1024;
        let checkpoint = 639_447_744;
        let baseline = private_memory(checkpoint);
        let before = baseline.before.unwrap();
        for bad in [
            PrivateJobMemory {
                host_available: required - 1,
                ..baseline
            },
            PrivateJobMemory {
                ancestor_remaining: None,
                ..baseline
            },
            PrivateJobMemory {
                before: None,
                ..baseline
            },
            PrivateJobMemory {
                after: None,
                ..baseline
            },
            PrivateJobMemory {
                ancestor_remaining: Some(u64::MAX),
                ..baseline
            },
            PrivateJobMemory {
                before: Some(ChargeSample {
                    current: checkpoint + 1,
                    peak: checkpoint,
                }),
                ..baseline
            },
            PrivateJobMemory {
                after: Some(ChargeSample {
                    current: checkpoint + 1,
                    peak: checkpoint,
                }),
                ..baseline
            },
            PrivateJobMemory {
                after: Some(ChargeSample {
                    current: checkpoint - 1,
                    peak: checkpoint - 1,
                }),
                ..baseline
            },
            PrivateJobMemory {
                before: Some(ChargeSample {
                    current: 0,
                    peak: budget + 1,
                }),
                after: Some(ChargeSample {
                    current: 0,
                    peak: budget + 1,
                }),
                ..baseline
            },
            // Crediting the larger later sample would incorrectly admit this.
            PrivateJobMemory {
                before: Some(ChargeSample {
                    current: 1,
                    ..before
                }),
                ancestor_remaining: Some(required - 2),
                ..baseline
            },
            // Likewise when the current charge shrinks after headroom sampling.
            PrivateJobMemory {
                after: Some(ChargeSample {
                    current: 1,
                    ..before
                }),
                ancestor_remaining: Some(required - 2),
                ..baseline
            },
        ] {
            assert!(admit_private_cost(required, budget, checkpoint, bad).is_err());
        }
        assert!(admit_private_cost(required, required - 1, checkpoint, baseline).is_err());
        assert!(admit_private_cost(required, budget, required + 1, baseline).is_err());
        let tight = PrivateJobMemory {
            before: Some(ChargeSample {
                current: 10,
                peak: 20,
            }),
            after: Some(ChargeSample {
                current: 15,
                peak: 20,
            }),
            ancestor_remaining: Some(required - 10),
            ..baseline
        };
        admit_private_cost(required, budget, checkpoint, tight).unwrap();
        assert!(admit_private_cost(
            required,
            budget,
            checkpoint,
            PrivateJobMemory {
                ancestor_remaining: Some(required - 11),
                ..tight
            }
        )
        .is_err());
    }

    #[test]
    fn private_job_credit_requires_exact_stable_unified_membership_and_mount_identity() {
        let path = "/user.slice/user-1000.slice/hag-answers-timing-123.service";
        let membership = format!("0::{path}\n");
        let mount = "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n";
        validate_private_scope(path, &membership, mount, &membership, mount).unwrap();
        for attested in [
            "",
            "/",
            "relative",
            "/a//b",
            "/a/../b",
            "/a/./b",
            "/a/",
            "/other.service",
        ] {
            assert!(
                validate_private_scope(attested, &membership, mount, &membership, mount).is_err()
            );
        }
        for bad in [
            String::new(),
            "0::/other.service\n".into(),
            format!("0::{path}\n0::{path}\n"),
            format!("0::{path}\n2:memory:/old\n"),
            format!("1::{path}\n"),
        ] {
            assert!(validate_private_scope(path, &bad, mount, &membership, mount).is_err());
            assert!(validate_private_scope(path, &membership, mount, &bad, mount).is_err());
        }
        assert!(validate_private_scope(path, &membership, "", &membership, "").is_err());
        assert!(
            validate_private_scope(path, &membership, mount, &membership, "changed mount").is_err()
        );
    }

    #[test]
    fn recording_metadata_is_capacity_independent_and_includes_retained_preparations() {
        let expected = NativeRecordingFootprint {
            batches: 31,
            dispatches: 364,
            weight_allocations: 234,
            tensor_allocations: 452,
            auxiliary_allocations: 119,
            retained_allocation_references: 404,
            noop_operands: 0,
            max_batch_commands: 54,
            max_batch_dispatches: 127,
        };
        for capacity in [1, 8, 128, 341, 342, 512, 1024, 2048] {
            assert_eq!(recording_footprint(capacity).unwrap(), expected);
            let native = expected.host_bytes().unwrap();
            let baseline = host_metadata_bytes(capacity, 0).unwrap();
            let bound = host_metadata_bytes(capacity, RETAINED_ATTENTION_METADATA_BOUND).unwrap();
            assert!(baseline >= native);
            assert_eq!(bound, baseline + RETAINED_ATTENTION_METADATA_BOUND);
            assert_eq!(bound % (1024 * 1024), 0);
            assert_eq!(
                bound,
                host_metadata_bytes(512, RETAINED_ATTENTION_METADATA_BOUND).unwrap()
            );
            assert!(bound < 4 * 1024 * 1024);
        }
        assert!(host_metadata_bytes(512, RETAINED_ATTENTION_METADATA_BOUND + 1).is_err());
    }

    #[test]
    fn larger_context_token_accounting_preserves_reset_high_water() {
        // Synthetic counters only: no model, cache, device or large allocation.
        for (capacity, tensors, allocation, preparation_writes, token_writes, peak) in [
            (
                8192,
                1_999_087_104,
                2_632_581_120,
                2_750_142_292,
                2_752_964,
                217_444,
            ),
            (
                16384,
                3_995_575_808,
                4_629_168_128,
                4_864_071_508,
                5_505_476,
                315_748,
            ),
        ] {
            assert_eq!(
                preparation_write_bytes(capacity).unwrap(),
                preparation_writes
            );
            assert_eq!(recording_write_bytes(capacity).unwrap(), token_writes);
            let before = NativeExecutionStats {
                allocation_bytes: allocation,
                weight_live_bytes: PACKED_BYTES,
                weight_peak_bytes: PACKED_BYTES,
                tensor_live_bytes: tensors,
                tensor_peak_bytes: tensors,
                auxiliary_live_bytes: PERSISTENT_AUXILIARY_BYTES,
                auxiliary_peak_bytes: PERSISTENT_AUXILIARY_BYTES,
                uploaded_weight_bytes: PACKED_BYTES,
                host_write_bytes: preparation_writes,
                ..Default::default()
            };
            let after = NativeExecutionStats {
                dispatch_count: 3793,
                dispatch_attempts: 3793,
                logical_submissions: 58,
                host_write_bytes: preparation_writes + token_writes,
                auxiliary_peak_bytes: peak,
                ..before
            };
            verify_token_accounting(capacity, before, after, capacity).unwrap();
            assert!(verify_token_accounting(CAPACITY, before, after, capacity).is_err());

            // Reset changes logical visibility, but counters and allocation peaks
            // persist. The next first-position recording writes only its scratch.
            let replay = NativeExecutionStats {
                dispatch_count: after.dispatch_count + 3793,
                dispatch_attempts: after.dispatch_attempts + 3793,
                logical_submissions: after.logical_submissions + 58,
                host_write_bytes: after.host_write_bytes + 115_028,
                ..after
            };
            verify_token_accounting(capacity, after, replay, 1).unwrap();
            assert!(verify_token_accounting(
                capacity,
                after,
                NativeExecutionStats {
                    auxiliary_peak_bytes: 123_232,
                    ..replay
                },
                1,
            )
            .is_err());
        }
    }

    #[test]
    fn token_accounting_tracks_scratch_growth_and_preserves_reset_high_water() {
        let before = NativeExecutionStats {
            allocation_bytes: 1_135_140_864,
            weight_live_bytes: PACKED_BYTES,
            weight_peak_bytes: PACKED_BYTES,
            tensor_live_bytes: 501_720_576,
            tensor_peak_bytes: 501_720_576,
            auxiliary_live_bytes: PERSISTENT_AUXILIARY_BYTES,
            auxiliary_peak_bytes: PERSISTENT_AUXILIARY_BYTES,
            uploaded_weight_bytes: PACKED_BYTES,
            host_write_bytes: 1_164_695_380,
            ..Default::default()
        };
        let after_for = |visible: usize, peak: u64| NativeExecutionStats {
            dispatch_count: 3793,
            dispatch_attempts: 3793,
            logical_submissions: 58,
            host_write_bytes: before.host_write_bytes + recording_write_bytes(visible).unwrap(),
            auxiliary_peak_bytes: peak,
            ..before
        };
        for (visible, writes, peak) in [
            (1, 115_028, 123_232),
            (341, 115_028, 123_232),
            (342, 115_364, 123_244),
            (512, 172_484, 125_284),
            (1023, 344_180, 131_416),
            (1024, 344_516, 131_428),
            (1025, 344_852, 131_440),
            (2047, 688_244, 143_704),
            (2048, 688_580, 143_716),
        ] {
            assert_eq!(recording_write_bytes(visible).unwrap(), writes);
            let after = after_for(visible, peak);
            verify_token_accounting(CAPACITY, before, after, visible).unwrap();
            for bad in [
                NativeExecutionStats {
                    host_write_bytes: after.host_write_bytes + 4,
                    ..after
                },
                NativeExecutionStats {
                    host_read_bytes: 4,
                    ..after
                },
                NativeExecutionStats {
                    auxiliary_live_bytes: PERSISTENT_AUXILIARY_BYTES + 4,
                    ..after
                },
                NativeExecutionStats {
                    auxiliary_peak_bytes: peak + 4,
                    ..after
                },
                NativeExecutionStats {
                    dispatch_count: 3792,
                    ..after
                },
                NativeExecutionStats {
                    dispatch_attempts: 3794,
                    ..after
                },
                NativeExecutionStats {
                    logical_submissions: 57,
                    ..after
                },
                NativeExecutionStats {
                    allocation_bytes: after.allocation_bytes + 4096,
                    ..after
                },
                NativeExecutionStats {
                    uploaded_weight_bytes: PACKED_BYTES + 4,
                    ..after
                },
                NativeExecutionStats {
                    poisoned: true,
                    ..after
                },
            ] {
                assert!(verify_token_accounting(CAPACITY, before, bad, visible).is_err());
            }
        }
        let prior_request = NativeExecutionStats {
            auxiliary_peak_bytes: 143_716,
            ..before
        };
        let replay = after_for(1, 143_716);
        verify_token_accounting(CAPACITY, prior_request, replay, 1).unwrap();
        assert!(
            verify_token_accounting(CAPACITY, prior_request, after_for(1, 123_232), 1).is_err()
        );
        assert!(verify_token_accounting(CAPACITY, before, replay, 1).is_err());
        assert!(verify_token_accounting(CAPACITY, before, replay, 0).is_err());
        assert!(verify_token_accounting(CAPACITY, before, replay, 2049).is_err());
        assert!(verify_token_accounting(
            CAPACITY,
            NativeExecutionStats {
                dispatch_count: u64::MAX,
                ..before
            },
            replay,
            1,
        )
        .is_err());
    }

    #[test]
    fn request_audit_distinguishes_callback_cancel_from_native_failure() {
        let success = RequestAudit::new(1, 64, true, false, true, false);
        assert_eq!(success.outcome, "success");
        let cancelled = RequestAudit::new(2, 9, false, true, true, false);
        assert_eq!(cancelled.to_string(),
            "{\"request_sequence\":2,\"evaluated_positions\":9,\"outcome\":\"cancelled\",\"reset_ok\":true,\"poisoned\":false}");
        for (callback, reset, poisoned) in [
            (false, true, false),
            (true, false, false),
            (true, true, true),
        ] {
            assert_eq!(
                RequestAudit::new(3, 9, false, callback, reset, poisoned).outcome,
                "failed"
            );
        }
    }

    #[test]
    fn serial_schedule_selects_only_final_prompt_and_does_not_evaluate_last_output() {
        let mut evaluated = Vec::new();
        let mut emitted = Vec::new();
        let mut phases = Vec::new();
        let result = generate_tokens(
            CAPACITY,
            &[1, 2],
            3,
            99,
            &mut || Ok(()),
            |token| {
                emitted.push(token);
                Ok(())
            },
            |token, select| {
                evaluated.push((token, select));
                Ok(select.then_some(token + 10))
            },
            |phase| phases.push(phase),
        )
        .unwrap();
        assert_eq!(evaluated, [(1, false), (2, true), (12, true), (22, true)]);
        assert_eq!(result.generated, [12, 22, 32]);
        assert_eq!(emitted, result.generated);
        assert_eq!(result.finish_reason, GenerationFinishReason::MaxNewTokens);
        assert_eq!(
            phases,
            [
                Phase::PrefillStart,
                Phase::PrefillEnd,
                Phase::DecodeStart,
                Phase::DecodeEnd
            ]
        );
    }

    #[test]
    fn eos_is_selected_but_never_emitted_or_evaluated() {
        let mut evaluated = 0;
        let mut phases = Vec::new();
        let result = generate_tokens(
            CAPACITY,
            &[1],
            4,
            7,
            &mut || Ok(()),
            |_| panic!("EOS must not be emitted"),
            |_, select| {
                evaluated += 1;
                assert!(select);
                Ok(Some(7))
            },
            |phase| phases.push(phase),
        )
        .unwrap();
        assert_eq!(evaluated, 1);
        assert!(result.generated.is_empty());
        assert_eq!(result.stop_token, Some(7));
        assert_eq!(result.finish_reason, GenerationFinishReason::StopToken);
        assert_eq!(phases, [Phase::PrefillStart, Phase::PrefillEnd]);
    }

    #[test]
    fn cancellation_after_prompt_work_stops_before_delivery_or_next_evaluation() {
        let completed = Cell::new(0);
        let mut phases = Vec::new();
        let result = generate_tokens(
            CAPACITY,
            &[1, 2],
            3,
            99,
            &mut || {
                ensure!(completed.get() < 1, "cancelled");
                Ok(())
            },
            |_| panic!("cancelled request emitted"),
            |_, _| {
                completed.set(completed.get() + 1);
                Ok(None)
            },
            |phase| phases.push((phase, completed.get())),
        );
        assert!(result.is_err());
        assert_eq!(completed.get(), 1);
        assert_eq!(phases, [(Phase::PrefillStart, 0)]);
        let evaluated = Cell::new(0);
        phases.clear();
        let result = generate_tokens(
            CAPACITY,
            &[1],
            3,
            99,
            &mut || Ok(()),
            |_| bail!("sink disconnected"),
            |_, _| {
                evaluated.set(evaluated.get() + 1);
                Ok(Some(2))
            },
            |phase| phases.push((phase, evaluated.get())),
        );
        assert!(result.is_err());
        assert_eq!(evaluated.get(), 1);
        assert_eq!(phases, [(Phase::PrefillStart, 0), (Phase::PrefillEnd, 1)]);
    }

    #[test]
    fn phase_schedule_refusal_and_failed_decode_do_not_emit_completion() {
        let mut phases = Vec::new();
        assert!(generate_tokens(
            CAPACITY,
            &[],
            1,
            7,
            &mut || Ok(()),
            |_| panic!("refused emit"),
            |_, _| panic!("refused evaluation"),
            |phase| phases.push((phase, 0))
        )
        .is_err());
        assert!(phases.is_empty());
        let completed = Cell::new(0);
        let emitted = Cell::new(0);
        let result = generate_tokens(
            CAPACITY,
            &[1],
            3,
            7,
            &mut || Ok(()),
            |_| {
                emitted.set(emitted.get() + 1);
                Ok(())
            },
            |_, _| {
                if completed.get() == 1 {
                    bail!("decode failed");
                }
                completed.set(completed.get() + 1);
                Ok(Some(2))
            },
            |phase| phases.push((phase, completed.get())),
        );
        assert!(result.is_err());
        assert_eq!((completed.get(), emitted.get()), (1, 1));
        assert_eq!(
            phases,
            [
                (Phase::PrefillStart, 0),
                (Phase::PrefillEnd, 1),
                (Phase::DecodeStart, 1)
            ]
        );
    }

    #[test]
    fn greedy_is_first_index_on_ties_and_refuses_all_nonfinite_values() {
        assert_eq!(greedy_token(&[-0.0, 0.0, -1.0]).unwrap(), 0);
        assert_eq!(greedy_token(&[2.0, 3.0, 3.0]).unwrap(), 1);
        for logits in [
            vec![],
            vec![f32::NAN],
            vec![0.0, f32::INFINITY],
            vec![0.0, f32::NEG_INFINITY],
        ] {
            assert!(greedy_token(&logits).is_err());
        }
    }

    #[test]
    fn resident_owner_is_send_and_becomes_sync_only_behind_runtime_mutex() {
        fn send<T: Send>() {}
        fn send_sync<T: Send + Sync>() {}
        send::<HagExecution>();
        send::<ResidentDenseModel>();
        send_sync::<Mutex<HagExecution>>();
        send_sync::<NativeExecutionDevice>();
    }
}
