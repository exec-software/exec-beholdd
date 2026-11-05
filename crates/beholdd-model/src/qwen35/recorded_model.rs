//! Staged, unqualified whole-graph native owner. No runtime or serving admission.
//!
//! An explicitly requested construction opens its own Qwen35Probe queue against
//! the caller's finite, empty, exclusively assigned budget. Every device clone,
//! component, tensor and finite checker stays private. The only source is a direct
//! reader of the supplied opened GGUF; there is no cache, matrix preload or CPU
//! arithmetic fallback. The whole GGUF file extent is nevertheless charged as a
//! conservative possible page-cache allowance, separately from bounded reads.
//!
//! Only checked component methods are reachable here. Failure after the beginning
//! of a token leaves the complete owner poisoned, including when earlier layers
//! have already committed. Reset clears actual state; it cannot heal a poisoned
//! native queue. State-only advancement validates every consumed activation but
//! skips the unused final head and publishes no logits ticket. A logits ticket
//! exposes only a completely validated final vocabulary.
//! The type is deliberately !Sync: a future runtime must hold its request mutex
//! over a complete sequence and supply its existing one-way process health latch.

#![cfg(all(feature = "hag-native", target_os = "linux"))]

use std::cell::Cell;
use std::marker::PhantomData;
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, ExecutionError, F32Tensor, F32TensorView, FiniteCheck,
    FiniteCheckSpec, LinearMode,
};
use beholdd_backend::hag::native_execution::{
    MemoryBudget, NativeExecutionConfig, NativeExecutionDevice, NativeExecutionProfile,
    NativeExecutionStats, NativeRecordingFootprint,
};
use beholdd_format::{CheckpointId, Gguf, GgufTensor, RangeReader, WeightRange};
use beholdd_memory::{ResidentSource, ResidentWeight, WeightAccess, WeightSource};

use super::config::{Attention, Binding};
use super::recorded_attention::{AttentionLimits, RecordedAttentionPlan, ResidentAttention};
use super::recorded_delta::{DeltaLimits, RecordedDeltaPlan, ResidentDelta};
use super::recorded_ffn::{
    FfnChunkLimits, FfnChunkPlan, FfnChunkWorkspace, FfnLimits, RecordedFfnPlan, ResidentFfn,
};
use super::recorded_weights::{ChunkStreamStats, PackedRowPlan, PrefillChunkSize, StreamStats};
use super::weights::{self, ReadLimits};

const IO_ELEMENTS: usize = 16_384;
const MAX_LAYERS: usize = 64;
const MAX_VOCAB: usize = 262_144;

fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .context("checked-model byte/counter sum overflows")
}
fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .context("checked-model byte/counter product overflows")
}
fn bytes(elements: usize) -> Result<u64> {
    mul(u64::try_from(elements)?, 4)
}
fn aligned4(value: u64) -> Result<u32> {
    Ok(u32::try_from(add(value, 3)? & !3)?)
}

#[derive(Debug, Clone)]
enum CorePlan {
    Full(RecordedAttentionPlan),
    Delta(RecordedDeltaPlan),
}

#[derive(Debug, Clone)]
struct LayerPlan {
    core: CorePlan,
    ffn: RecordedFfnPlan,
}

/// Metadata-only graph plan. This is not an advertised execution capability.
#[derive(Debug, Clone)]
pub struct RecordedQwen35Plan {
    checkpoint: CheckpointId,
    dim: usize,
    vocab: usize,
    capacity: usize,
    eps: f32,
    layers: Vec<LayerPlan>,
    embedding: PackedRowPlan,
    output: PackedRowPlan,
    norm: GgufTensor,
    reads: ReadLimits,
    tensors: u64,
    packed: u64,
    auxiliary: u64,
    host_staging: u64,
    recording_metadata: u64,
    model_metadata: u64,
    reset_bytes: u64,
    matrix_tiles: u64,
    matrix_bytes: u64,
    delta_status_bytes: u64,
    recording_footprint: NativeRecordingFootprint,
    ffn_matrix_tiles: u64,
    ffn_matrix_bytes: u64,
    scaled_matrix_tiles: u64,
    advance_arithmetic_dispatches: u64,
    advance_arithmetic_submissions: u64,
    chunk: Option<FfnChunkPlan>,
}

/// Separate checked costs, without opening a device or reading weight payload.
/// Per-step traffic describes a full logits-producing step; state-only advances
/// use no more than these bounds. Both paths retain the same admitted resources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedRequirements {
    /// Exact private tensor payload including the one shared checker.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time aligned packed tile.
    pub packed_tile_bytes: u64,
    /// Conservative simultaneous private workspace/parameter upper bound.
    pub auxiliary_bytes: u64,
    /// Exact arena capacities used by this owner's private queue.
    pub config: NativeExecutionConfig,
    /// Page-rounded programs/IB/arenas/guards charged to the supplied budget.
    pub device_charge_bytes: u64,
    /// Host payload upper bound, including backend conversion and caller logits.
    pub host_payload_bytes: u64,
    /// Conservative native ownership/recording metadata upper bound.
    pub recording_metadata_bytes: u64,
    /// Model plan/resident metadata allowance, excluding the supplied GGUF index.
    pub model_metadata_bytes: u64,
    /// Successful full reset writes, excluding unchanged norms/RoPE/checker.
    pub reset_write_bytes: u64,
    /// Successful scalar complete-token packed submissions, including embedding.
    pub packed_tiles_per_step: u64,
    /// Successful complete-token packed source/upload bytes.
    pub packed_bytes_per_step: u64,
    /// Generic finite validation calls across the full graph.
    pub finite_checks_per_step: u64,
    /// Additional physical finite-validation dispatches.
    pub finite_dispatches_per_step: u64,
    /// Additional logical finite-validation submissions.
    pub finite_submissions_per_step: u64,
    /// Pending writes across all generic checks; checker init is separate.
    pub finite_pending_bytes_per_step: u64,
    /// Status-only reads across all generic checks.
    pub finite_status_bytes_per_step: u64,
    /// Additional Delta core status reads, separate from generic validation.
    pub delta_status_bytes_per_step: u64,
    /// Exact admitted checker costs; verified against the actual opened device.
    pub checker: FiniteCheckSpec,
}

/// Opt-in fixed-B costs relative to the same plan with scalar resources only.
/// Traffic describes one successful no-head chunk, not a full-head token. All
/// arithmetic and finite checks still execute separately for every token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckedChunkRequirements {
    /// Exact token count required by every chunk call: 1, 2 or 4.
    pub tokens: usize,
    /// One additional logical tensor allocation inside the existing tensor arena.
    pub extra_tensor_allocations: u64,
    /// B * (2D + 2W) * 4, with the scalar layer packets still resident.
    pub extra_resident_tensor_bytes: u64,
    /// Difference between the two page-rounded queue charges; no new guard pair.
    pub extra_device_charge_bytes: u64,
    /// Extra peak host payload; zero while the scalar 64KiB bound covers reset.
    pub extra_host_payload_bytes: u64,
    /// Additional conservative native allocation/reference metadata allowance.
    pub extra_recording_metadata_bytes: u64,
    /// Workspace and its bounded temporary view metadata allowance.
    pub extra_model_metadata_bytes: u64,
    /// Exact extra native zero initialization writes for the one workspace.
    pub initialization_write_bytes: u64,
    /// Exact extra writes for each successful whole-model physical reset.
    pub reset_write_bytes: u64,
    /// Read/upload/completed SOURCE tiles, with every FFN source tile counted once.
    pub packed_source_tiles: u64,
    /// Packed source/upload bytes, with no decoded weight copy.
    pub packed_source_bytes: u64,
    /// Packed logical submissions: core/embedding plus B uses of every FFN tile.
    pub packed_submissions: u64,
    /// Successful FFN tile uses added to the separate ChunkStreamStats counter.
    pub ffn_tile_applications: u64,
    /// All actual arithmetic and finite physical dispatches; prerecords submit none.
    pub physical_dispatches: u64,
    /// All actual arithmetic and finite logical submissions.
    pub logical_submissions: u64,
    /// Generic embedding/core/FFN checks; no final norm or vocabulary checks.
    pub finite_checks: u64,
    /// Generic finite-check pending writes.
    pub finite_pending_bytes: u64,
    /// Generic finite-check status reads plus Delta status reads.
    pub status_read_bytes: u64,
    /// Gross extra record-time writes versus B scalar advances: B*layers*(4D+8).
    /// One extra weighted-norm workspace/epsilon and one SwiGLU scalar per FFN.
    pub extra_ffn_recording_write_bytes: u64,
    /// Gross packed host writes avoided versus B scalar advances: (B-1)*FFN bytes.
    /// The signed total host-write difference is extra_recording minus this value.
    pub avoided_packed_write_bytes: u64,
}

/// Explicit process and device ceilings. Neither is inferred from available RAM.
#[derive(Debug, Clone, Copy)]
pub struct ModelLimits {
    /// Maximum fixed native queue charge, also enforced by the supplied budget.
    pub device_bytes: u64,
    /// Total forecast: device + complete file allowance + host + process headroom.
    /// Device bytes count here too because this target shares system DRAM.
    pub total_process_bytes: u64,
    /// Caller-verified allowance for existing process/GGUF metadata, allocator,
    /// driver-private memory and runtime state, outside the enumerated payloads.
    pub process_headroom_bytes: u64,
}

/// Verified source and admitted process extents. The direct source has no cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAdmission {
    /// Sum of recorded opened shard lengths, including headers and alignment.
    pub source_file_bytes: u64,
    /// Always zero for the sole supported direct source.
    pub source_cache_bytes: u64,
    /// Always zero; no full-weight resident allocation is created.
    pub source_resident_bytes: u64,
    /// Explicit external process allowance supplied by the caller.
    pub process_headroom_bytes: u64,
    /// Checked total compared with `ModelLimits::total_process_bytes`.
    pub total_process_bytes: u64,
    /// Expected outstanding charge until ordered native retirement.
    pub expected_device_charge_bytes: u64,
}

impl RecordedQwen35Plan {
    /// Bind bounded typed layers and full-vocabulary head using only metadata.
    pub fn new(binding: &Binding, capacity: usize, reads: ReadLimits) -> Result<Self> {
        let c = binding.config();
        ensure!(
            (1..=MAX_LAYERS).contains(&c.n_layers),
            "checked model layer bound exceeded"
        );
        ensure!(
            (1..=MAX_VOCAB).contains(&c.vocab),
            "checked model vocabulary bound exceeded"
        );
        ensure!(
            (1..=16_384).contains(&capacity) && capacity <= c.context,
            "checked model capacity outside evaluated native bound"
        );
        let embedding = PackedRowPlan::new(binding.embedding(), reads)?;
        let output = PackedRowPlan::new(binding.output(), reads)?;
        ensure!(
            embedding.cols() == c.dim
                && output.cols() == c.dim
                && embedding.rows() == c.vocab
                && output.rows() == c.vocab,
            "checked model embedding/head dimensions differ"
        );
        ensure!(
            bytes(c.vocab)? <= reads.decoded_bytes as u64,
            "full vocabulary exceeds streamed result limit"
        );
        ensure!(
            c.dim <= 5120 && c.eps.is_finite() && c.eps > 0.0,
            "checked model final normalization unsupported"
        );
        let head = bytes(
            c.dim
                .checked_mul(3)
                .and_then(|v| v.checked_add(c.vocab))
                .context("checked model head extent overflows")?,
        )?;
        let mut tensors = head;
        let mut reset_bytes = bytes(2 * c.dim + c.vocab)?;
        let mut packed = output.peak_read_bytes().max(embedding.row_bytes()) as u64;
        let mut host_staging = packed.max(bytes(2 * c.dim)?).max(65_536);
        // FFN norm+activation scalar blocks remain retained across the temporary
        // ScaledAdd down projection: epsilon, SwiGLU limit, residual scale.
        let mut auxiliary = add(bytes(c.dim)?, 12)?;
        let mut delta_count = 0u64;
        let mut delta_status_bytes = 0u64;
        let mut max_commands = 1u64;
        let mut max_tile_rows = output.tile_rows();
        let mut matrix_tiles = add(1, output.rows().div_ceil(output.tile_rows()) as u64)?;
        let mut matrix_bytes = add(embedding.row_bytes() as u64, binding.output().nbytes())?;
        let mut ffn_matrix_tiles = 0;
        let mut ffn_matrix_bytes = 0;
        let mut scaled_matrix_tiles = 0;
        let mut advance_arithmetic_dispatches = 0;
        let mut advance_arithmetic_submissions = 0;
        let norm_dispatches = add(1, c.dim.div_ceil(4096) as u64)?;
        let mut layers = Vec::new();
        layers.try_reserve_exact(c.n_layers)?;
        for (index, block) in binding.blocks().iter().enumerate() {
            let core = match &block.attention {
                Attention::Full(full) => {
                    let p = RecordedAttentionPlan::new(binding, index, capacity, reads)?;
                    let m = p.memory();
                    tensors = add(tensors, m.resident_tensor_bytes)?;
                    packed = packed.max(m.packed_tile_bytes);
                    host_staging = host_staging.max(m.host_staging_bytes);
                    let s = p.shape();
                    let immutable = bytes(c.dim + 2 * s.head_dim + capacity * s.rope_dim)?;
                    reset_bytes = add(
                        reset_bytes,
                        m.resident_tensor_bytes
                            .checked_sub(immutable)
                            .context("attention immutable forecast exceeds packet")?,
                    )?;
                    // Four prerecordings: hidden RMS, Q/K RMS+RoPE, attention,
                    // gate; plus one temporary linear batch. Scratch is reused
                    // within each batch. Include three scalar blocks and scale.
                    auxiliary = auxiliary.max(add(
                        add(bytes(c.dim + s.head_dim)?, mul(capacity as u64, 12)?)?,
                        20,
                    )?);
                    max_commands = max_commands.max(p.retained_commands() as u64);
                    for tensor in [&full.query_gate, &full.key, &full.value, &full.output] {
                        let rows = PackedRowPlan::new(tensor, reads)?;
                        max_tile_rows = max_tile_rows.max(rows.tile_rows().min(rows.rows()));
                        matrix_tiles =
                            add(matrix_tiles, rows.rows().div_ceil(rows.tile_rows()) as u64)?;
                        matrix_bytes = add(matrix_bytes, tensor.nbytes())?;
                    }
                    let output = PackedRowPlan::new(&full.output, reads)?;
                    scaled_matrix_tiles = add(
                        scaled_matrix_tiles,
                        output.rows().div_ceil(output.tile_rows()) as u64,
                    )?;
                    advance_arithmetic_dispatches = add(
                        advance_arithmetic_dispatches,
                        add(
                            norm_dispatches,
                            add(
                                mul(3, (s.query_heads + s.kv_heads) as u64)?,
                                mul(4, s.query_heads as u64)?,
                            )?,
                        )?,
                    )?;
                    advance_arithmetic_submissions = add(advance_arithmetic_submissions, 4)?;
                    CorePlan::Full(p)
                }
                Attention::DeltaNet(delta) => {
                    let p = RecordedDeltaPlan::new(binding, index, capacity, reads)?;
                    let m = p.memory();
                    tensors = add(tensors, m.resident_tensor_bytes)?;
                    packed = packed.max(m.packed_tile_bytes);
                    host_staging = host_staging.max(m.host_staging_bytes);
                    reset_bytes = add(
                        reset_bytes,
                        add(p.layout().frame_bytes as u64, bytes(c.dim)?)?,
                    )?;
                    delta_count = add(delta_count, 1)?;
                    delta_status_bytes = add(delta_status_bytes, bytes(p.shape().value_heads)?)?;
                    for tensor in [
                        &delta.qkv,
                        &delta.gate,
                        &delta.alpha,
                        &delta.beta,
                        &delta.output,
                    ] {
                        let rows = PackedRowPlan::new(tensor, reads)?;
                        max_tile_rows = max_tile_rows.max(rows.tile_rows().min(rows.rows()));
                        matrix_tiles =
                            add(matrix_tiles, rows.rows().div_ceil(rows.tile_rows()) as u64)?;
                        matrix_bytes = add(matrix_bytes, tensor.nbytes())?;
                    }
                    let output = PackedRowPlan::new(&delta.output, reads)?;
                    scaled_matrix_tiles = add(
                        scaled_matrix_tiles,
                        output.rows().div_ceil(output.tile_rows()) as u64,
                    )?;
                    advance_arithmetic_dispatches =
                        add(advance_arithmetic_dispatches, add(norm_dispatches, 4)?)?;
                    advance_arithmetic_submissions = add(advance_arithmetic_submissions, 2)?;
                    CorePlan::Delta(p)
                }
            };
            let ffn = RecordedFfnPlan::new(binding, index, reads)?;
            let m = ffn.memory();
            tensors = add(tensors, m.resident_tensor_bytes)?;
            reset_bytes = add(reset_bytes, m.resident_tensor_bytes - bytes(c.dim)?)?;
            packed = packed.max(m.packed_tile_bytes);
            host_staging = host_staging.max(m.host_staging_bytes);
            max_commands = max_commands.max(ffn.activation_commands() as u64);
            for tensor in [&block.ffn.gate, &block.ffn.up, &block.ffn.down] {
                let rows = PackedRowPlan::new(tensor, reads)?;
                max_tile_rows = max_tile_rows.max(rows.tile_rows().min(rows.rows()));
                matrix_tiles = add(matrix_tiles, rows.rows().div_ceil(rows.tile_rows()) as u64)?;
                matrix_bytes = add(matrix_bytes, tensor.nbytes())?;
                ffn_matrix_tiles = add(
                    ffn_matrix_tiles,
                    rows.rows().div_ceil(rows.tile_rows()) as u64,
                )?;
                ffn_matrix_bytes = add(ffn_matrix_bytes, tensor.nbytes())?;
            }
            let down = PackedRowPlan::new(&block.ffn.down, reads)?;
            scaled_matrix_tiles = add(
                scaled_matrix_tiles,
                down.rows().div_ceil(down.tile_rows()) as u64,
            )?;
            advance_arithmetic_dispatches = add(
                advance_arithmetic_dispatches,
                add(norm_dispatches, ffn.activation_commands() as u64)?,
            )?;
            advance_arithmetic_submissions = add(advance_arithmetic_submissions, 2)?;
            layers.push(LayerPlan { core, ffn });
        }
        ensure!(
            layers.len() == c.n_layers,
            "checked model layer count differs"
        );
        // Worst simultaneous prerecordings plus an output ScaledAdd scratch.
        auxiliary = add(auxiliary, bytes(max_tile_rows)?)?;
        let parity_batches = mul(delta_count, 2)?;
        let batches = add(parity_batches, 5)?;
        // Each semantic operation lowers to <=3 dispatches except wide weighted
        // RMS (<=3 here too); a temporary checker/packed batch adds <=2.
        let peak_dispatches = add(mul(max_commands, 3)?, 2)?;
        let footprint = NativeRecordingFootprint {
            batches,
            dispatches: add(mul(parity_batches, 4)?, peak_dispatches)?,
            weight_allocations: 1,
            tensor_allocations: add(add(mul(c.n_layers as u64, 2)?, delta_count)?, 2)?,
            auxiliary_allocations: 12,
            retained_allocation_references: add(mul(parity_batches, 1)?, 32)?,
            noop_operands: 0,
            max_batch_commands: max_commands.max(4),
            max_batch_dispatches: peak_dispatches.max(4),
        };
        // Layer structures retain cloned plans and <=16 tiny dimension Vecs per
        // layer. Doubling covers plan+resident Vec capacity; extra4096/layer
        // covers dimension backing and allocator bookkeeping. GGUF index belongs
        // to the explicit external process headroom, never inferred here.
        let model_metadata = add(
            65_536,
            mul(
                c.n_layers as u64,
                add(mul(std::mem::size_of::<LayerPlan>() as u64, 2)?, 4096)?,
            )?,
        )?;
        Ok(Self {
            checkpoint: binding.checkpoint_id(),
            dim: c.dim,
            vocab: c.vocab,
            capacity,
            eps: c.eps,
            layers,
            embedding,
            output,
            norm: binding.output_norm().clone(),
            reads,
            tensors,
            packed,
            auxiliary,
            host_staging,
            recording_metadata: footprint.host_bytes()?,
            model_metadata,
            reset_bytes,
            matrix_tiles,
            matrix_bytes,
            delta_status_bytes,
            recording_footprint: footprint,
            ffn_matrix_tiles,
            ffn_matrix_bytes,
            scaled_matrix_tiles,
            advance_arithmetic_dispatches,
            advance_arithmetic_submissions,
            chunk: None,
        })
    }

    /// Explicitly admit one fixed B1/B2/B4 FFN workspace at plan construction.
    /// Scalar APIs remain available and keep their arithmetic/finite boundaries.
    /// The ordinary constructor does not allocate or charge this workspace.
    pub fn with_prefill_chunk(mut self, tokens: usize) -> Result<Self> {
        ensure!(self.chunk.is_none(), "prefill chunk already configured");
        let size = PrefillChunkSize::new(tokens)?;
        ensure!(
            size.tokens() <= self.capacity,
            "prefill chunk exceeds model capacity"
        );
        let first = self
            .layers
            .first()
            .context("prefill chunk has no model layers")?;
        let chunk = FfnChunkPlan::new(self.dim, first.ffn.width(), size.tokens())?;
        ensure!(
            self.layers
                .iter()
                .all(|layer| layer.ffn.dim() == chunk.dim() && layer.ffn.width() == chunk.width()),
            "prefill chunk requires uniform FFN geometry"
        );
        self.chunk = Some(chunk);
        Ok(self)
    }

    /// Exact optional chunk size. Scalar tails and full-head steps remain valid.
    pub fn prefill_chunk_size(&self) -> Option<usize> {
        self.chunk.map(FfnChunkPlan::tokens)
    }

    fn chunk_next(&self, tokens: &[u32], position: usize) -> Result<usize> {
        let count = self
            .prefill_chunk_size()
            .context("model has no admitted prefill chunk")?;
        ensure!(
            tokens.len() == count,
            "prefill chunk token count differs from admitted size"
        );
        ensure!(
            tokens.iter().all(|&token| (token as usize) < self.vocab),
            "prefill chunk token outside vocabulary"
        );
        let next = position
            .checked_add(count)
            .context("prefill chunk position overflows")?;
        ensure!(
            next <= self.capacity,
            "prefill chunk exceeds remaining model capacity"
        );
        Ok(next)
    }

    /// Admit every checked boundary and sum exact payload/traffic plus documented
    /// conservative metadata/scratch upper bounds. No device effects occur.
    pub fn checked_requirements(
        &self,
        capability: Option<FiniteCheckSpec>,
    ) -> Result<CheckedRequirements> {
        self.requirements_with_chunk(capability, self.chunk)
    }

    fn requirements_with_chunk(
        &self,
        capability: Option<FiniteCheckSpec>,
        chunk: Option<FfnChunkPlan>,
    ) -> Result<CheckedRequirements> {
        let checker =
            capability.context("whole native model requires explicit finite capability")?;
        ensure!(
            checker.max_elements > 0 && self.dim <= checker.max_elements,
            "whole native model finite-check extent too small"
        );
        let mut checks = add(2, self.vocab.div_ceil(checker.max_elements) as u64)?;
        for layer in &self.layers {
            let count = match &layer.core {
                CorePlan::Full(p) => p.finite_check_requirements(Some(checker))?.checks_per_step,
                CorePlan::Delta(p) => p.finite_check_requirements(Some(checker))?.checks_per_step,
            };
            checks = add(
                checks,
                add(
                    count,
                    layer
                        .ffn
                        .finite_check_requirements(Some(checker))?
                        .checks_per_step,
                )?,
            )?;
        }
        let chunk_memory = chunk.map(FfnChunkPlan::memory);
        let tensor = add(
            add(self.tensors, checker.resident_tensor_bytes)?,
            chunk_memory.map_or(0, |m| m.resident_tensor_bytes),
        )?;
        let config = NativeExecutionConfig {
            tensor_capacity_bytes: aligned4(tensor)?,
            weight_capacity_bytes: aligned4(self.packed)?,
            auxiliary_capacity_bytes: aligned4(self.auxiliary)?,
        };
        // At most one backend conversion copy overlaps model staging. Bounding
        // it by the entire model staging covers both full RoPE tables and large
        // Delta initialization; reset/readback use <=64KiB. Caller logits are
        // counted separately so the read destination cannot evade admission.
        let host_payload_bytes = add(
            add(
                mul(
                    self.host_staging
                        .max(self.auxiliary)
                        .max(65_536)
                        .max(chunk_memory.map_or(0, |m| m.host_staging_bytes)),
                    2,
                )?,
                checker.host_scratch_bytes,
            )?,
            bytes(self.vocab)?,
        )?;
        let recording_metadata = if let Some(m) = chunk_memory {
            let mut footprint = self.recording_footprint;
            footprint.tensor_allocations = add(footprint.tensor_allocations, m.tensor_allocations)?;
            // The shared workspace replaces the scalar hidden/FFN packet in an
            // active recording. Add one reference conservatively; no extra batch
            // is retained across a scalar core or a chunk FFN recording.
            footprint.retained_allocation_references =
                add(footprint.retained_allocation_references, 1)?;
            footprint.host_bytes()?
        } else {
            self.recording_metadata
        };
        Ok(CheckedRequirements {
            resident_tensor_bytes: tensor,
            packed_tile_bytes: u64::from(config.weight_capacity_bytes),
            auxiliary_bytes: u64::from(config.auxiliary_capacity_bytes),
            config,
            device_charge_bytes: config
                .allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)?,
            host_payload_bytes,
            recording_metadata_bytes: recording_metadata,
            model_metadata_bytes: add(
                self.model_metadata,
                chunk_memory.map_or(0, |m| m.host_metadata_bytes),
            )?,
            reset_write_bytes: add(
                self.reset_bytes,
                chunk_memory.map_or(0, |m| m.reset_write_bytes),
            )?,
            packed_tiles_per_step: self.matrix_tiles,
            packed_bytes_per_step: self.matrix_bytes,
            finite_checks_per_step: checks,
            finite_dispatches_per_step: mul(checks, checker.physical_dispatches)?,
            finite_submissions_per_step: mul(checks, checker.logical_submissions)?,
            finite_pending_bytes_per_step: mul(checks, checker.pending_write_bytes)?,
            finite_status_bytes_per_step: mul(checks, checker.status_read_bytes)?,
            delta_status_bytes_per_step: self.delta_status_bytes,
            checker,
        })
    }

    /// Optional exact successful chunk traffic and resource deltas. This also
    /// checks the full model finite capability; no payload/device access occurs.
    pub fn checked_chunk_requirements(
        &self,
        capability: Option<FiniteCheckSpec>,
    ) -> Result<Option<CheckedChunkRequirements>> {
        let Some(chunk) = self.chunk else {
            return Ok(None);
        };
        let scalar = self.requirements_with_chunk(capability, None)?;
        let admitted = self.checked_requirements(capability)?;
        let m = chunk.memory();
        let count = chunk.tokens() as u64;
        let sub = |a: u64, b: u64| {
            a.checked_sub(b)
                .context("chunk forecast subtraction underflows")
        };
        let head_tiles = self.output.rows().div_ceil(self.output.tile_rows()) as u64;
        let advance_tiles = sub(self.matrix_tiles, head_tiles)?;
        let head_bytes = mul(self.output.rows() as u64, self.output.row_bytes() as u64)?;
        let advance_bytes = sub(self.matrix_bytes, head_bytes)?;
        let head_checks = add(1, self.vocab.div_ceil(scalar.checker.max_elements) as u64)?;
        let checks = mul(count, sub(scalar.finite_checks_per_step, head_checks)?)?;
        let applications = mul(count, advance_tiles)?;
        Ok(Some(CheckedChunkRequirements {
            tokens: chunk.tokens(),
            extra_tensor_allocations: m.tensor_allocations,
            extra_resident_tensor_bytes: m.resident_tensor_bytes,
            extra_device_charge_bytes: sub(
                admitted.device_charge_bytes,
                scalar.device_charge_bytes,
            )?,
            extra_host_payload_bytes: sub(admitted.host_payload_bytes, scalar.host_payload_bytes)?,
            extra_recording_metadata_bytes: sub(
                admitted.recording_metadata_bytes,
                scalar.recording_metadata_bytes,
            )?,
            extra_model_metadata_bytes: sub(
                admitted.model_metadata_bytes,
                scalar.model_metadata_bytes,
            )?,
            initialization_write_bytes: m.initialization_write_bytes,
            reset_write_bytes: m.reset_write_bytes,
            packed_source_tiles: add(
                mul(count, sub(advance_tiles, self.ffn_matrix_tiles)?)?,
                self.ffn_matrix_tiles,
            )?,
            packed_source_bytes: add(
                mul(count, sub(advance_bytes, self.ffn_matrix_bytes)?)?,
                self.ffn_matrix_bytes,
            )?,
            packed_submissions: applications,
            ffn_tile_applications: mul(count, self.ffn_matrix_tiles)?,
            physical_dispatches: add(
                add(
                    applications,
                    mul(
                        count,
                        add(self.scaled_matrix_tiles, self.advance_arithmetic_dispatches)?,
                    )?,
                )?,
                mul(checks, scalar.checker.physical_dispatches)?,
            )?,
            logical_submissions: add(
                add(
                    applications,
                    mul(count, self.advance_arithmetic_submissions)?,
                )?,
                mul(checks, scalar.checker.logical_submissions)?,
            )?,
            finite_checks: checks,
            finite_pending_bytes: mul(checks, scalar.checker.pending_write_bytes)?,
            status_read_bytes: add(
                mul(checks, scalar.checker.status_read_bytes)?,
                mul(count, self.delta_status_bytes)?,
            )?,
            extra_ffn_recording_write_bytes: mul(
                mul(count, self.layers.len() as u64)?,
                add(bytes(self.dim)?, 8)?,
            )?,
            avoided_packed_write_bytes: mul(count - 1, self.ffn_matrix_bytes)?,
        }))
    }

    /// Verify opened shard extents and explicit ceilings without payload/device
    /// access. Requirements must match this plan's complete admitted checker.
    pub fn admit(
        &self,
        checkpoint: &Gguf,
        requirements: CheckedRequirements,
        limits: ModelLimits,
        budget: &MemoryBudget,
    ) -> Result<ModelAdmission> {
        ensure!(
            requirements == self.checked_requirements(Some(requirements.checker))?,
            "model requirements differ from metadata plan"
        );
        ensure!(
            checkpoint.checkpoint_id() == self.checkpoint,
            "checked model source identity differs"
        );
        ensure!(
            budget.limit_bytes().is_some() && budget.current_bytes() == 0,
            "checked model requires a finite empty exclusively assigned shared budget"
        );
        ensure!(
            requirements.device_charge_bytes <= limits.device_bytes
                && requirements.device_charge_bytes <= budget.remaining(),
            "checked model device allowance too small"
        );
        ensure!(
            limits.process_headroom_bytes > 0,
            "checked model needs explicit external process headroom"
        );
        let file = (0..checkpoint.shard_count()).try_fold(0u64, |sum, shard| {
            add(
                sum,
                checkpoint
                    .shard_len(shard)
                    .context("missing opened shard extent")?,
            )
        })?;
        ensure!(file > 0, "checked model source extent is empty");
        let mut total = file;
        for value in [
            requirements.device_charge_bytes,
            requirements.host_payload_bytes,
            requirements.recording_metadata_bytes,
            requirements.model_metadata_bytes,
            limits.process_headroom_bytes,
        ] {
            total = add(total, value)?;
        }
        ensure!(
            total <= limits.total_process_bytes,
            "checked model process allowance too small"
        );
        Ok(ModelAdmission {
            source_file_bytes: file,
            source_cache_bytes: 0,
            source_resident_bytes: 0,
            process_headroom_bytes: limits.process_headroom_bytes,
            total_process_bytes: total,
            expected_device_charge_bytes: requirements.device_charge_bytes,
        })
    }

    /// Explicit first-request construction only. The caller must exclusively assign
    /// the supplied shared budget until retirement and pass its existing one-way
    /// process health latch. No selection/startup/service hook calls this API.
    /// The actual typed-library profile capability is required before opening
    /// the large fixed queue, then verified against the real device before any
    /// checkpoint payload reads or per-layer tensor allocations.
    #[allow(clippy::too_many_arguments)]
    pub fn open_checked(
        self,
        render_node: impl AsRef<Path>,
        checkpoint: Arc<Gguf>,
        admitted_checker: FiniteCheckSpec,
        limits: ModelLimits,
        budget: &MemoryBudget,
        healthy: Arc<AtomicBool>,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentQwen35CheckedModel> {
        ensure!(
            NativeExecutionProfile::Qwen35Probe.finite_check_spec() == Some(admitted_checker),
            "typed native library does not provide the admitted finite capability"
        );
        let requirements = self.checked_requirements(Some(admitted_checker))?;
        let admission = self.admit(&checkpoint, requirements, limits, budget)?;
        ensure!(
            healthy.load(Ordering::Acquire),
            "process native HAG health is latched failed"
        );
        control()?;
        // Declared before the closure so any failed partial construction retires
        // all its queue handles before outstanding charge is inspected.
        let mut cleanup = PreparationCleanup {
            budget,
            healthy: &healthy,
            armed: true,
        };
        let result = (|| -> Result<ResidentQwen35CheckedModel> {
            let device = NativeExecutionDevice::open_with_profile(
                render_node,
                budget,
                requirements.config,
                NativeExecutionProfile::Qwen35Probe,
            )?;
            ensure!(
                device.finite_check_spec() == Some(requirements.checker),
                "actual native finite capability differs from admitted spec"
            );
            let initial = device.stats()?;
            ensure!(
                initial
                    == NativeExecutionStats {
                        allocation_bytes: requirements.device_charge_bytes,
                        ..Default::default()
                    },
                "checked model requires a fresh private queue"
            );
            ensure!(
                budget.current_bytes() == requirements.device_charge_bytes,
                "shared budget charge differs after queue construction"
            );
            control()?;
            let mut owner = ResidentQwen35CheckedModel {
                plan: self,
                requirements,
                admission,
                source: DirectSource::new(checkpoint),
                layers: Vec::new(),
                checker: None,
                head: None,
                chunk_workspace: None,
                device: Some(device),
                budget: budget.clone(),
                healthy: healthy.clone(),
                publication: Publication::new(),
                phase: ModelPhase::Preparing,
                stream: StreamStats::default(),
                chunk_stream: ChunkStreamStats::default(),
                head_reset_bytes: 0,
                logits_read_bytes: 0,
                retired: false,
                _not_sync: PhantomData,
            };
            let latch = healthy.clone();
            owner.initialize(&mut || {
                ensure!(
                    latch.load(Ordering::Acquire),
                    "process native health failed during preparation"
                );
                control()
            })?;
            Ok(owner)
        })();
        if result.is_ok() {
            cleanup.armed = false;
        }
        result
    }
}

/// Every direct source read, including preparation norms and failed attempts.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SourceReadStats {
    /// Calls attempted on the direct positional source.
    pub attempts: u64,
    /// Actual bytes returned, independent of upload success.
    pub returned_bytes: u64,
    /// Maximum one returned slice; no source retention or cache is present.
    pub peak_bytes: u64,
}

struct DirectSource {
    checkpoint: Arc<Gguf>,
    stats: Mutex<SourceReadStats>,
}
impl DirectSource {
    fn new(checkpoint: Arc<Gguf>) -> Self {
        Self {
            checkpoint,
            stats: Mutex::new(SourceReadStats::default()),
        }
    }
    fn stats(&self) -> Result<SourceReadStats> {
        Ok(*self
            .stats
            .lock()
            .map_err(|_| anyhow::anyhow!("source stats lock poisoned"))?)
    }
}
impl WeightSource for DirectSource {
    fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint.checkpoint_id()
    }
    fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
        {
            let mut stats = self
                .stats
                .lock()
                .map_err(|_| anyhow::anyhow!("source stats lock poisoned"))?;
            stats.attempts = add(stats.attempts, 1)?;
        }
        let weight = ResidentSource::new(self.checkpoint.as_ref()).load_range(range, access)?;
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| anyhow::anyhow!("source stats lock poisoned"))?;
        stats.returned_bytes = add(stats.returned_bytes, weight.bytes().len() as u64)?;
        stats.peak_bytes = stats.peak_bytes.max(weight.bytes().len() as u64);
        Ok(weight)
    }
}

enum CheckedCore {
    Full(ResidentAttention),
    Delta(ResidentDelta),
}
struct CheckedLayer {
    core: CheckedCore,
    ffn: ResidentFfn,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkStage {
    Embedding(usize),
    Core { layer: usize, token: usize },
    Ffn(usize),
}

// Allocation-free ordering seam shared by native execution and hardware-free
// schedule/failure tests. Each core consumes p, p+1, ... in chronological order;
// a layer's FFN completes all rows before the next layer can observe any row.
fn execute_chunk_stages(
    size: PrefillChunkSize,
    layers: usize,
    mut execute: impl FnMut(ChunkStage) -> Result<()>,
) -> Result<()> {
    for token in 0..size.tokens() {
        execute(ChunkStage::Embedding(token))?;
    }
    for layer in 0..layers {
        for token in 0..size.tokens() {
            execute(ChunkStage::Core { layer, token })?;
        }
        execute(ChunkStage::Ffn(layer))?;
    }
    Ok(())
}
impl CheckedCore {
    fn position(&self) -> usize {
        match self {
            Self::Full(p) => p.positions(),
            Self::Delta(p) => p.positions(),
        }
    }
    fn poisoned(&self) -> bool {
        match self {
            Self::Full(p) => p.is_poisoned(),
            Self::Delta(p) => p.is_poisoned(),
        }
    }
    fn reset_bytes(&self) -> Result<u64> {
        match self {
            Self::Full(p) => Ok(p.reset_write_bytes()),
            Self::Delta(p) => add(
                p.outside_reset_write_bytes(),
                p.sequence_stats().reset_write_bytes,
            ),
        }
    }
    fn step_checked(
        &mut self,
        device: &NativeExecutionDevice,
        source: &dyn WeightSource,
        hidden: &F32TensorView,
        checker: &mut FiniteCheck,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Full(p) => p.step_checked(device, source, hidden, checker, stats, control),
            Self::Delta(p) => p.step_checked(source, hidden, checker, stats, control),
        }
    }
    fn reset(
        &mut self,
        device: &NativeExecutionDevice,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        match self {
            Self::Full(p) => p.reset(device, control),
            Self::Delta(p) => p.reset(control),
        }
    }
}

/// Last reached phase. Layer progress never substitutes for model publication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelPhase {
    /// Parameters/state are being initialized.
    Preparing,
    /// A token may start if the owner remains healthy.
    Ready,
    /// Selected packed embedding row decode/validation.
    Embedding,
    /// Checked core at this zero-based layer.
    Core(usize),
    /// Checked common FFN at this zero-based layer.
    Ffn(usize),
    /// Final weighted RMS and validation.
    FinalNorm,
    /// Complete untied vocabulary projection.
    Vocabulary,
    /// Every logit chunk is being validated.
    LogitsValidation,
    /// Bounded final-logit readback.
    LogitsRead,
    /// Physical state clearing is incomplete.
    Resetting,
    /// Native resources were explicitly retired.
    Retired,
}

/// Opaque publication proof. It retains no device/tensor resources.
#[derive(Debug, Clone)]
pub struct LogitsTicket {
    owner: Arc<()>,
    epoch: u64,
    generation: u64,
    position: usize,
}

struct Publication {
    owner: Arc<()>,
    epoch: u64,
    generation: u64,
    position: usize,
    poisoned: bool,
    published: bool,
}
impl Publication {
    fn new() -> Self {
        Self {
            owner: Arc::new(()),
            epoch: 0,
            generation: 0,
            position: 0,
            poisoned: true,
            published: false,
        }
    }
    fn invalidate(&mut self) -> Result<()> {
        self.published = false;
        self.poisoned = true;
        self.generation = self
            .generation
            .checked_add(1)
            .context("publication generation exhausted")?;
        Ok(())
    }
    fn commit(&mut self, next: usize) -> LogitsTicket {
        self.commit_state(next);
        self.published = true;
        LogitsTicket {
            owner: self.owner.clone(),
            epoch: self.epoch,
            generation: self.generation,
            position: next,
        }
    }
    fn commit_state(&mut self, next: usize) {
        self.position = next;
        self.poisoned = false;
        self.published = false;
    }
    fn accept(&self, ticket: &LogitsTicket) -> Result<()> {
        ensure!(
            !self.poisoned
                && self.published
                && Arc::ptr_eq(&self.owner, &ticket.owner)
                && ticket.epoch == self.epoch
                && ticket.generation == self.generation
                && ticket.position == self.position,
            "stale, foreign or unpublished logits ticket"
        );
        Ok(())
    }
    fn reset_complete(&mut self, epoch: u64) {
        self.epoch = epoch;
        self.position = 0;
        self.poisoned = false;
        self.published = false;
    }
}

/// Observable counters, including successful prefixes of refused operations.
#[derive(Debug, Clone, Copy)]
pub struct ModelStateSnapshot {
    /// Last phase reached, retained on error.
    pub phase: ModelPhase,
    /// Fully checked state positions, including advances without a final head.
    pub committed_positions: usize,
    /// Fixed admitted evaluated-position bound.
    pub capacity: usize,
    /// Complete owner requires physical reset or abandonment.
    pub poisoned: bool,
    /// External process health latch, never reset by this model.
    pub process_healthy: bool,
    /// Model-created source activity, including preparation reads.
    pub source: SourceReadStats,
    /// Packed matrix progress, including partial tokens.
    pub packed: StreamStats,
    /// Actual complete native queue counters.
    pub native: NativeExecutionStats,
    /// Successful reset writes across all components and head, including prefixes.
    pub reset_write_bytes: u64,
    /// Successful final-logit transfer bytes, including cancelled read prefixes.
    pub logits_read_bytes: u64,
    /// Actual outstanding supplied-budget charge.
    pub budget_current_bytes: u64,
    /// Fixed expected outstanding charge while active.
    pub budget_expected_bytes: u64,
}

/// Distinct checked native owner; no public device, checker or activation views.
pub struct ResidentQwen35CheckedModel {
    plan: RecordedQwen35Plan,
    requirements: CheckedRequirements,
    admission: ModelAdmission,
    source: DirectSource,
    layers: Vec<CheckedLayer>,
    checker: Option<FiniteCheck>,
    head: Option<F32Tensor>,
    chunk_workspace: Option<FfnChunkWorkspace>,
    device: Option<NativeExecutionDevice>,
    budget: MemoryBudget,
    healthy: Arc<AtomicBool>,
    publication: Publication,
    phase: ModelPhase,
    stream: StreamStats,
    chunk_stream: ChunkStreamStats,
    head_reset_bytes: u64,
    logits_read_bytes: u64,
    retired: bool,
    _not_sync: PhantomData<Cell<()>>,
}

fn finite(
    device: &NativeExecutionDevice,
    checker: &mut FiniteCheck,
    view: &F32TensorView,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    device.check_finite(checker, view, &mut || {
        control().map_err(|e| ExecutionError::Shape(format!("checked-model finite control: {e:#}")))
    })?;
    Ok(())
}

impl ResidentQwen35CheckedModel {
    fn initialize(&mut self, control: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        let device = self.device.as_ref().context("missing owned queue")?;
        self.checker = Some(device.prepare_finite_check()?);
        self.layers.try_reserve_exact(self.plan.layers.len())?;
        for layer in &self.plan.layers {
            control()?;
            ensure!(
                self.healthy.load(Ordering::Acquire),
                "process health failed during preparation"
            );
            let core = match &layer.core {
                CorePlan::Full(p) => {
                    let m = p.memory();
                    CheckedCore::Full(p.prepare_checked(
                        device,
                        &self.source,
                        AttentionLimits {
                            resident_tensor_bytes: m.resident_tensor_bytes,
                            packed_tile_bytes: m.packed_tile_bytes,
                            host_staging_bytes: m.host_staging_bytes,
                        },
                        control,
                    )?)
                }
                CorePlan::Delta(p) => {
                    let m = p.memory();
                    CheckedCore::Delta(p.prepare_checked(
                        device,
                        &self.source,
                        DeltaLimits {
                            resident_tensor_bytes: m.resident_tensor_bytes,
                            packed_tile_bytes: m.packed_tile_bytes,
                            host_staging_bytes: m.host_staging_bytes,
                        },
                        control,
                    )?)
                }
            };
            let m = layer.ffn.memory();
            let ffn = layer.ffn.prepare_checked(
                device,
                &self.source,
                FfnLimits {
                    resident_tensor_bytes: m.resident_tensor_bytes,
                    packed_tile_bytes: m.packed_tile_bytes,
                    host_staging_bytes: m.host_staging_bytes,
                },
                control,
            )?;
            self.layers.push(CheckedLayer { core, ffn });
        }
        if let Some(plan) = self.plan.chunk {
            let m = plan.memory();
            self.chunk_workspace = Some(plan.prepare(
                device,
                FfnChunkLimits {
                    resident_tensor_bytes: m.resident_tensor_bytes,
                    host_staging_bytes: m.host_staging_bytes,
                    host_metadata_bytes: m.host_metadata_bytes,
                },
                control,
            )?);
        }
        control()?;
        let packet = device.allocate_f32(3 * self.plan.dim + self.plan.vocab)?;
        self.head = Some(packet);
        control()?;
        let norm = weights::f32_tensor(&self.plan.norm, &self.source, self.plan.reads)?;
        control()?;
        device.write_f32(&self.head.as_ref().unwrap().view(0, self.plan.dim)?, &norm)?;
        drop(norm);
        control()?;
        self.check_queue()?;
        ensure!(
            device.stats()?.tensor_live_bytes == self.requirements.resident_tensor_bytes,
            "prepared tensor charge differs from checked model forecast"
        );
        self.uniform(0)?;
        self.publication.poisoned = false;
        self.phase = ModelPhase::Ready;
        Ok(())
    }

    fn check_queue(&self) -> Result<()> {
        ensure!(
            self.healthy.load(Ordering::Acquire),
            "process native health failed"
        );
        let result = (|| -> Result<()> {
            let device = self.device.as_ref().context("native owner retired")?;
            ensure!(!device.stats()?.poisoned, "native queue poisoned");
            ensure!(
                self.budget.current_bytes() == self.requirements.device_charge_bytes,
                "exclusive shared budget charge changed"
            );
            Ok(())
        })();
        if result.is_err() {
            self.healthy.store(false, Ordering::Release);
        }
        result
    }

    fn uniform(&self, expected: usize) -> Result<()> {
        ensure!(
            self.layers.len() == self.plan.layers.len(),
            "model preparation incomplete"
        );
        ensure!(
            self.layers.iter().all(|l| !l.core.poisoned()
                && !l.ffn.is_poisoned()
                && l.core.position() == expected
                && l.ffn.completed_steps() == expected as u64),
            "model component counters/health differ from whole-token position"
        );
        ensure!(
            match (&self.plan.chunk, &self.chunk_workspace) {
                (None, None) => true,
                (Some(plan), Some(workspace)) =>
                    !workspace.is_poisoned()
                        && plan.tokens() == workspace.plan().tokens()
                        && plan.dim() == workspace.plan().dim()
                        && plan.width() == workspace.plan().width(),
                _ => false,
            },
            "model chunk workspace is missing, poisoned or differs from admission"
        );
        Ok(())
    }

    /// Admitted costs, including independently verified full-file allowance.
    pub fn admission(&self) -> ModelAdmission {
        self.admission
    }
    /// Checked static tensor/traffic/config forecasts.
    pub fn requirements(&self) -> CheckedRequirements {
        self.requirements
    }

    /// Exact configured chunk size, or None for the unchanged scalar owner.
    pub fn prefill_chunk_size(&self) -> Option<usize> {
        self.plan.prefill_chunk_size()
    }

    /// FFN tile applications across chunk calls, including successful prefixes of
    /// failed chunks. Scalar calls do not add to this independent counter.
    pub fn chunk_packed_stats(&self) -> ChunkStreamStats {
        self.chunk_stream
    }

    /// Optional admitted chunk costs/traffic, matching this owner's checker.
    pub fn chunk_requirements(&self) -> Result<Option<CheckedChunkRequirements>> {
        self.plan
            .checked_chunk_requirements(Some(self.requirements.checker))
    }

    /// One full token. Every phase uses the mandatory shared finite checker.
    /// Error after invalidation leaves poison and the last committed position.
    pub fn step_checked(
        &mut self,
        token: u32,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<LogitsTicket> {
        self.evaluate_checked(token, true, control)?
            .context("full checked token omitted its logits ticket")
    }

    /// Advance recurrent/KV state for a prompt token whose logits are unused.
    /// Embedding and every core/FFN retain their mandatory finite checks. Final
    /// normalization, vocabulary projection and logits validation are not run:
    /// none feeds later state, and this operation publishes no logits ticket.
    /// Any previous ticket is invalidated before mutation. Cancellation/failure
    /// after invalidation leaves poison and the last committed position, exactly
    /// as for a full checked token; a successful physical reset is required.
    pub fn advance_checked(
        &mut self,
        token: u32,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.evaluate_checked(token, false, control).map(|_| ())
    }

    /// Advance exactly the opted-in B1/B2/B4 prompt rows without a final head.
    /// Every token and the complete end position are checked before invalidation
    /// or callbacks. Each layer advances its cores chronologically, then applies
    /// its checked FFN to all rows. Only uniform whole-chunk state is committed.
    /// Any error after invalidation requires complete physical model reset;
    /// neither a prefix position nor the stale vocabulary buffer is published.
    pub fn advance_chunk_checked(
        &mut self,
        tokens: &[u32],
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let next = self.plan.chunk_next(tokens, self.publication.position)?;
        self.check_queue()?;
        ensure!(
            !self.publication.poisoned,
            "model is poisoned; reset or abandon it"
        );
        self.uniform(self.publication.position)?;
        control()?;
        self.publication.invalidate()?;
        let latch = self.healthy.clone();
        let result = self.execute_chunk(tokens, next, &mut || {
            ensure!(
                latch.load(Ordering::Acquire),
                "process native health failed during chunk"
            );
            control()
        });
        if result.is_err() {
            let _ = self.check_queue();
        }
        result
    }

    fn execute_chunk(
        &mut self,
        tokens: &[u32],
        next: usize,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let device = self.device.as_ref().context("owner retired")?;
        let checker = self.checker.as_mut().context("mandatory checker absent")?;
        let workspace = self
            .chunk_workspace
            .as_mut()
            .context("chunk workspace absent")?;
        let size = PrefillChunkSize::new(workspace.plan().tokens())?;
        execute_chunk_stages(size, self.layers.len(), |stage| {
            match stage {
                ChunkStage::Embedding(token) => {
                    self.phase = ModelPhase::Embedding;
                    let hidden = workspace.hidden(token)?;
                    self.plan.embedding.embedding_row(
                        device,
                        &self.source,
                        tokens[token] as usize,
                        &hidden,
                        &mut self.stream,
                        control,
                    )?;
                    finite(device, checker, &hidden, control)?;
                }
                ChunkStage::Core { layer, token } => {
                    self.phase = ModelPhase::Core(layer);
                    let hidden = workspace.hidden(token)?;
                    self.layers[layer].core.step_checked(
                        device,
                        &self.source,
                        &hidden,
                        checker,
                        &mut self.stream,
                        control,
                    )?;
                }
                ChunkStage::Ffn(layer) => {
                    self.phase = ModelPhase::Ffn(layer);
                    self.layers[layer].ffn.step_chunk_checked(
                        device,
                        &self.source,
                        workspace,
                        checker,
                        &mut self.stream,
                        &mut self.chunk_stream,
                        control,
                    )?;
                }
            }
            Ok(())
        })?;
        self.uniform(next)?;
        control()?;
        self.check_queue()?;
        self.publication.commit_state(next);
        self.phase = ModelPhase::Ready;
        Ok(())
    }

    fn evaluate_checked(
        &mut self,
        token: u32,
        with_logits: bool,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<LogitsTicket>> {
        self.check_queue()?;
        ensure!(
            !self.publication.poisoned,
            "model is poisoned; reset or abandon it"
        );
        ensure!(
            (token as usize) < self.plan.vocab,
            "token outside vocabulary"
        );
        ensure!(
            self.publication.position < self.plan.capacity,
            "model capacity exhausted"
        );
        self.uniform(self.publication.position)?;
        control()?;
        let next = self.publication.position + 1;
        self.publication.invalidate()?;
        let latch = self.healthy.clone();
        let result = self.execute_token(token, next, with_logits, &mut || {
            ensure!(
                latch.load(Ordering::Acquire),
                "process native health failed during token"
            );
            control()
        });
        if result.is_err() {
            let _ = self.check_queue();
        }
        result
    }

    fn execute_token(
        &mut self,
        token: u32,
        next: usize,
        with_logits: bool,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Option<LogitsTicket>> {
        let device = self.device.as_ref().context("owner retired")?;
        let head = self.head.as_ref().context("head absent")?;
        let hidden = head.view(self.plan.dim, self.plan.dim)?;
        let checker = self.checker.as_mut().context("mandatory checker absent")?;
        self.phase = ModelPhase::Embedding;
        self.plan.embedding.embedding_row(
            device,
            &self.source,
            token as usize,
            &hidden,
            &mut self.stream,
            control,
        )?;
        finite(device, checker, &hidden, control)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            self.phase = ModelPhase::Core(index);
            layer.core.step_checked(
                device,
                &self.source,
                &hidden,
                checker,
                &mut self.stream,
                control,
            )?;
            self.phase = ModelPhase::Ffn(index);
            layer.ffn.step_checked(
                device,
                &self.source,
                &hidden,
                checker,
                &mut self.stream,
                control,
            )?;
        }
        if with_logits {
            let normalized = head.view(2 * self.plan.dim, self.plan.dim)?;
            let logits = head.view(3 * self.plan.dim, self.plan.vocab)?;
            self.phase = ModelPhase::FinalNorm;
            let mut commands = CommandList::new();
            commands.rmsnorm(
                &normalized,
                &hidden,
                Some(&head.view(0, self.plan.dim)?),
                self.plan.eps,
            )?;
            let batch = device.record(&commands)?;
            drop(commands);
            control()?;
            let submitted = device.submit(&batch)?;
            device.wait(&submitted)?;
            drop(submitted);
            drop(batch);
            control()?;
            finite(device, checker, &normalized, control)?;
            self.phase = ModelPhase::Vocabulary;
            self.plan.output.execute_rows(
                device,
                &self.source,
                0..self.plan.vocab,
                &normalized,
                &logits,
                LinearMode::Overwrite,
                &mut self.stream,
                control,
            )?;
            self.phase = ModelPhase::LogitsValidation;
            for start in (0..self.plan.vocab).step_by(self.requirements.checker.max_elements) {
                finite(
                    device,
                    checker,
                    &logits.view(
                        start,
                        (self.plan.vocab - start).min(self.requirements.checker.max_elements),
                    )?,
                    control,
                )?;
            }
        }
        self.uniform(next)?;
        control()?;
        self.check_queue()?;
        let ticket = if with_logits {
            Some(self.publication.commit(next))
        } else {
            self.publication.commit_state(next);
            None
        };
        self.phase = ModelPhase::Ready;
        Ok(ticket)
    }

    /// Read only validated final logits, in <=64KiB transfers. Stale/foreign
    /// tickets and wrong destination lengths refuse before transfer. Any failed
    /// read invalidates publication; partially filled output is not a result.
    pub fn read_logits(
        &mut self,
        ticket: &LogitsTicket,
        output: &mut [f32],
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.publication.accept(ticket)?;
        ensure!(
            output.len() == self.plan.vocab,
            "logits destination width differs"
        );
        self.check_queue()?;
        self.phase = ModelPhase::LogitsRead;
        let result = (|| -> Result<()> {
            let device = self.device.as_ref().context("owner retired")?;
            let head = self.head.as_ref().context("head absent")?;
            for (chunk, values) in output.chunks_mut(IO_ELEMENTS).enumerate() {
                ensure!(
                    self.healthy.load(Ordering::Acquire),
                    "process native health failed during logits read"
                );
                control()?;
                let view = head.view(3 * self.plan.dim + chunk * IO_ELEMENTS, values.len())?;
                device.read_f32(&view, values)?;
                self.logits_read_bytes = add(self.logits_read_bytes, bytes(values.len())?)?;
                control()?;
            }
            self.check_queue()?;
            Ok(())
        })();
        if result.is_err() {
            // No need to advance generation: poisoned/unpublished already makes
            // every ticket unusable; preserve the original transfer error.
            self.publication.poisoned = true;
            self.publication.published = false;
            let _ = self.check_queue();
        } else {
            self.phase = ModelPhase::Ready;
        }
        result
    }

    /// Physical reset of all component state and model-owned mutable head. The
    /// caller supplies a distinct bounded cleanup control after cancellation.
    /// Current Delta internals poll between layers, not within their <=6.7MiB
    /// bounded clearing loop; this explicit cancellation-latency gap is retained.
    pub fn reset_checked(&mut self, control: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        self.check_queue()?;
        let epoch = self
            .publication
            .epoch
            .checked_add(1)
            .context("reset epoch exhausted")?;
        self.publication.invalidate()?;
        self.phase = ModelPhase::Resetting;
        let latch = self.healthy.clone();
        let mut guarded_control = || -> Result<()> {
            ensure!(
                latch.load(Ordering::Acquire),
                "process native health failed during reset"
            );
            control()
        };
        let result = (|| -> Result<()> {
            let device = self.device.as_ref().context("owner retired")?;
            for layer in &mut self.layers {
                guarded_control()?;
                layer.core.reset(device, &mut guarded_control)?;
                guarded_control()?;
                layer.ffn.reset(device, &mut guarded_control)?;
            }
            if let Some(workspace) = &mut self.chunk_workspace {
                guarded_control()?;
                workspace.reset(device, &mut guarded_control)?;
            }
            let zeros = vec![0.0f32; IO_ELEMENTS];
            let head = self.head.as_ref().context("head absent")?;
            let mutable = 2 * self.plan.dim + self.plan.vocab;
            for start in (0..mutable).step_by(IO_ELEMENTS) {
                guarded_control()?;
                let count = (mutable - start).min(IO_ELEMENTS);
                device.write_f32(&head.view(self.plan.dim + start, count)?, &zeros[..count])?;
                self.head_reset_bytes = add(self.head_reset_bytes, bytes(count)?)?;
                guarded_control()?;
            }
            self.uniform(0)?;
            guarded_control()?;
            self.check_queue()?;
            Ok(())
        })();
        if result.is_ok() {
            self.publication.reset_complete(epoch);
            self.phase = ModelPhase::Ready;
        } else {
            let _ = self.check_queue();
        }
        result
    }

    /// Read-only bounded metadata; failed arithmetic never invents completed work.
    pub fn state(&self) -> Result<ModelStateSnapshot> {
        let mut reset = self.head_reset_bytes;
        if let Some(workspace) = &self.chunk_workspace {
            reset = add(reset, workspace.reset_write_bytes())?;
        }
        for layer in &self.layers {
            reset = add(
                reset,
                add(layer.core.reset_bytes()?, layer.ffn.reset_write_bytes())?,
            )?;
        }
        let native = self.device.as_ref().context("owner retired")?.stats()?;
        let charge = self.budget.current_bytes();
        if native.poisoned || charge != self.requirements.device_charge_bytes {
            self.healthy.store(false, Ordering::Release);
        }
        let process_healthy = self.healthy.load(Ordering::Acquire);
        Ok(ModelStateSnapshot {
            phase: self.phase,
            committed_positions: self.publication.position,
            capacity: self.plan.capacity,
            poisoned: self.publication.poisoned || !process_healthy,
            process_healthy,
            source: self.source.stats()?,
            packed: self.stream,
            native,
            reset_write_bytes: reset,
            logits_read_bytes: self.logits_read_bytes,
            budget_current_bytes: charge,
            budget_expected_bytes: self.requirements.device_charge_bytes,
        })
    }

    fn retire_inner(&mut self) -> Retirement {
        let before = self.budget.current_bytes();
        let native = self.device.as_ref().and_then(|d| d.stats().ok());
        self.publication.poisoned = true;
        self.publication.published = false;
        // Drop every component parity recording/device clone and tensor view,
        // then checker/head, then the last queue owner. Tickets own no resources.
        self.layers.clear();
        drop(self.chunk_workspace.take());
        drop(self.checker.take());
        drop(self.head.take());
        drop(self.device.take());
        let after = self.budget.current_bytes();
        let clean = before == self.requirements.device_charge_bytes
            && after == 0
            && native.as_ref().is_some_and(|s| !s.poisoned);
        if !clean {
            self.healthy.store(false, Ordering::Release);
        }
        self.retired = true;
        self.phase = ModelPhase::Retired;
        Retirement {
            clean,
            expected_device_charge_bytes: self.requirements.device_charge_bytes,
            budget_before_bytes: before,
            budget_after_bytes: after,
            native,
        }
    }

    /// Ordered retirement with explicit budget-zero evidence. `clean=false`
    /// permanently lowers the supplied process health latch; no retry bypass.
    pub fn retire(mut self) -> Retirement {
        self.retire_inner()
    }
}

/// Final native evidence after all model-owned GPU handles have been dropped.
#[derive(Debug, Clone, Copy)]
pub struct Retirement {
    /// Healthy queue, exact prior charge and zero charge after ordered destruction.
    pub clean: bool,
    /// Exact fixed queue charge expected while active.
    pub expected_device_charge_bytes: u64,
    /// Observed charge immediately before resource destruction.
    pub budget_before_bytes: u64,
    /// Observed charge after the final queue owner was dropped.
    pub budget_after_bytes: u64,
    /// Actual last queue counters; absent if metadata retrieval itself failed.
    pub native: Option<NativeExecutionStats>,
}

impl Drop for ResidentQwen35CheckedModel {
    fn drop(&mut self) {
        if !self.retired {
            let _ = self.retire_inner();
        }
    }
}
struct PreparationCleanup<'a> {
    budget: &'a MemoryBudget,
    healthy: &'a AtomicBool,
    armed: bool,
}
impl Drop for PreparationCleanup<'_> {
    fn drop(&mut self) {
        if self.armed && self.budget.current_bytes() != 0 {
            self.healthy.store(false, Ordering::Release);
        }
    }
}

#[cfg(test)]
#[path = "recorded_model/chunk_tests.rs"]
mod chunk_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::tests::Fixture;

    fn spec() -> FiniteCheckSpec {
        FiniteCheckSpec {
            max_elements: 65_536,
            resident_tensor_bytes: 276,
            initialization_write_bytes: 548,
            host_scratch_bytes: 512,
            physical_dispatches: 1,
            logical_submissions: 1,
            pending_write_bytes: 256,
            status_read_bytes: 256,
        }
    }

    #[test]
    fn metadata_requires_capability_and_counts_one_shared_checker() {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf).unwrap();
        let plan = RecordedQwen35Plan::new(&binding, 5, ReadLimits::default()).unwrap();
        assert!(plan.checked_requirements(None).is_err());
        let r = plan.checked_requirements(Some(spec())).unwrap();
        // 1 full*10 +3 Delta*3 +4 FFN*5 +embedding +final +one logit check.
        assert_eq!(r.finite_checks_per_step, 42);
        assert_eq!(r.finite_status_bytes_per_step, 42 * 256);
        assert_eq!(r.resident_tensor_bytes, plan.tensors + 276);
        assert_eq!(r.delta_status_bytes_per_step, 3 * 2 * 4);
        assert_eq!(
            r.device_charge_bytes,
            r.config
                .allocation_bytes_for_profile(NativeExecutionProfile::Qwen35Probe)
                .unwrap()
        );
        let mut too_small = spec();
        too_small.max_elements = 256;
        assert!(plan.checked_requirements(Some(too_small)).is_err());
        let mut overflow = spec();
        overflow.pending_write_bytes = u64::MAX;
        assert!(plan.checked_requirements(Some(overflow)).is_err());
    }

    #[test]
    fn full_file_allowance_and_external_headroom_are_not_tensor_payloads() {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf).unwrap();
        let plan = RecordedQwen35Plan::new(&binding, 5, ReadLimits::default()).unwrap();
        let r = plan.checked_requirements(Some(spec())).unwrap();
        let budget = MemoryBudget::new(r.device_charge_bytes);
        let limits = ModelLimits {
            device_bytes: r.device_charge_bytes,
            total_process_bytes: u64::MAX,
            process_headroom_bytes: 1234,
        };
        let a = plan.admit(&gguf, r, limits, &budget).unwrap();
        assert_eq!(
            a.source_file_bytes,
            (0..gguf.shard_count())
                .map(|i| gguf.shard_len(i).unwrap())
                .sum::<u64>()
        );
        assert!(a.source_file_bytes > binding.checkpoint_bytes());
        assert_eq!(a.source_cache_bytes + a.source_resident_bytes, 0);
        assert!(plan
            .admit(
                &gguf,
                r,
                ModelLimits {
                    total_process_bytes: a.total_process_bytes - 1,
                    ..limits
                },
                &budget
            )
            .is_err());
        assert!(plan
            .admit(
                &gguf,
                r,
                ModelLimits {
                    process_headroom_bytes: 0,
                    ..limits
                },
                &budget
            )
            .is_err());
        assert_eq!(budget.current_bytes(), 0); // metadata test performs no device work.
    }

    #[test]
    fn publication_never_advances_on_partial_token_or_partial_reset() {
        let mut p = Publication::new();
        p.poisoned = false;
        p.invalidate().unwrap();
        let first = p.commit(1);
        p.accept(&first).unwrap();
        p.invalidate().unwrap(); // represents embedding or a later failed phase.
        assert_eq!(p.position, 1);
        assert!(p.poisoned && p.accept(&first).is_err());
        p.invalidate().unwrap(); // partial reset remains poisoned and unpublished.
        assert_eq!(p.position, 1);
        p.reset_complete(1);
        assert_eq!(p.position, 0);
        assert!(p.accept(&first).is_err());
        p.invalidate().unwrap();
        let replay = p.commit(1);
        p.accept(&replay).unwrap();
        assert!(p.accept(&first).is_err());
    }

    #[test]
    fn foreign_ticket_and_generation_exhaustion_cannot_publish() {
        let mut p = Publication::new();
        let mut foreign = Publication::new();
        let ticket = foreign.commit(1);
        p.commit(1);
        assert!(p.accept(&ticket).is_err());
        p.generation = u64::MAX;
        assert!(p.invalidate().is_err());
        assert!(p.poisoned && !p.published);
        assert_eq!(p.position, 1);
    }

    #[test]
    fn state_advance_invalidates_prior_logits_without_publishing_stale_head() {
        let mut p = Publication::new();
        p.invalidate().unwrap();
        let first = p.commit(1);
        p.accept(&first).unwrap();

        p.invalidate().unwrap();
        p.commit_state(2);
        assert_eq!(p.position, 2);
        assert!(!p.poisoned && !p.published);
        assert!(p.accept(&first).is_err());
        // Even a test-only ticket carrying current identity cannot turn the
        // unused, stale vocabulary buffer into a validated result.
        let unpublished = LogitsTicket {
            owner: p.owner.clone(),
            epoch: p.epoch,
            generation: p.generation,
            position: p.position,
        };
        assert!(p.accept(&unpublished).is_err());

        p.invalidate().unwrap();
        let next = p.commit(3);
        p.accept(&next).unwrap();
        assert_eq!(p.position, 3);
        assert!(p.accept(&first).is_err());
        assert!(p.accept(&unpublished).is_err());
    }

    #[test]
    fn cancelled_advance_retains_last_committed_position_until_physical_reset() {
        let mut p = Publication::new();
        p.invalidate().unwrap();
        p.commit_state(1);
        p.invalidate().unwrap(); // A later state advance fails after mutation.
        assert_eq!(p.position, 1);
        assert!(p.poisoned && !p.published);

        p.invalidate().unwrap(); // An incomplete reset cannot publish state.
        assert_eq!(p.position, 1);
        assert!(p.poisoned && !p.published);
        p.reset_complete(1);
        assert_eq!(p.position, 0);
        assert!(!p.poisoned && !p.published);

        p.invalidate().unwrap();
        p.commit_state(1);
        p.invalidate().unwrap();
        let replay = p.commit(2);
        p.accept(&replay).unwrap();
        assert_eq!(replay.epoch, 1);
        assert_eq!(p.position, 2);
    }

    #[test]
    fn generation_exhaustion_after_state_advance_never_publishes() {
        let mut p = Publication::new();
        p.invalidate().unwrap();
        p.commit_state(1);
        p.generation = u64::MAX;
        assert!(p.invalidate().is_err());
        assert_eq!(p.position, 1);
        assert!(p.poisoned && !p.published);
    }

    #[test]
    fn outstanding_preparation_charge_latches_health_without_recovery() {
        let budget = MemoryBudget::new(64);
        let healthy = AtomicBool::new(true);
        let reservation = budget.reserve(4).unwrap();
        drop(PreparationCleanup {
            budget: &budget,
            healthy: &healthy,
            armed: true,
        });
        assert!(!healthy.load(Ordering::Acquire));
        drop(reservation);
        assert_eq!(budget.current_bytes(), 0);
        drop(PreparationCleanup {
            budget: &budget,
            healthy: &healthy,
            armed: true,
        });
        assert!(!healthy.load(Ordering::Acquire));
        // This checks the accounting seam only, not native fence retirement.
    }
}
