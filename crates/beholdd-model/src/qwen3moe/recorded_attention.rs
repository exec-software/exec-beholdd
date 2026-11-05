//! Bounded resident Qwen3 attention with a model-owned, position-major KV cache.
//!
//! Preparation reads only this layer's four Q8_0 matrices and three F32 norms.
//! Recording binds hidden and either prebuilds position batches or records on demand. A step
//! publishes its position only after the complete residual output has finished. An
//! uncertain submission poisons this cache; reset cannot make uncertain storage safe.

use anyhow::{bail, ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, F32Tensor, F32TensorView, LinearMode, PreparedWeight,
    RecordedBatch,
};
use beholdd_backend::{IncrementalAttentionConfig, RopeLayout, Weight};
use beholdd_format::WeightRange;
use beholdd_memory::{WeightAccess, WeightSource};

use super::{Bound, CheckpointId, QwenArchitecture, QwenModel};

/// Logical payloads excluding caller hidden, backend private scratch/alignment,
/// programs, guards, prepared-weight expansion and independently retained source caches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseAttentionMemory {
    /// Sum of the four unchanged packed matrix payloads.
    pub packed_weight_bytes: u64,
    /// Norms, scratch, fixed KV cache and complete cosine/sine table payloads.
    pub resident_tensor_bytes: u64,
    /// Largest model-owned matrix read, raw-plus-decoded norm, or coefficient
    /// staging pair. Backend upload conversion buffers require a separate budget.
    pub largest_staging_bytes: u64,
}

/// All logical limits are checked before the first source read or device allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseAttentionLimits {
    /// Maximum sum of the four packed matrix payloads.
    pub packed_weight_bytes: u64,
    /// Maximum persistent norm, scratch, cache and coefficient payloads.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time host staging payload.
    pub largest_staging_bytes: u64,
}

#[derive(Debug, Clone)]
struct PlannedWeight {
    bound: Bound,
    range: WeightRange,
}

/// Metadata-only plan for `hidden += O(attention(RMSNorm(hidden)))` in one Qwen3 layer.
#[derive(Debug, Clone)]
pub struct DenseAttentionPlan {
    checkpoint_id: CheckpointId,
    dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    context: usize,
    eps: f32,
    theta: f64,
    weights: [PlannedWeight; 4],
    norms: [PlannedWeight; 3],
}

impl QwenModel {
    /// Derive a strict Q8_0/F32 attention plan without loading any tensor payload.
    /// This bounded first recorded component accepts at most 32 heads of width 256.
    pub fn dense_attention_plan(&self, layer_index: usize) -> Result<DenseAttentionPlan> {
        ensure!(
            self.architecture == QwenArchitecture::Qwen3,
            "recorded dense attention requires exact qwen3 architecture"
        );
        let cfg = &self.core.cfg;
        let layer = self
            .core
            .layers
            .get(layer_index)
            .context("attention layer is outside checkpoint")?;
        ensure!(
            (1..=4096).contains(&cfg.dim)
                && (1..=32).contains(&cfg.n_heads)
                && (1..=32).contains(&cfg.n_kv_heads)
                && cfg.n_heads.is_multiple_of(cfg.n_kv_heads),
            "unsupported recorded attention dimensions or GQA"
        );
        ensure!(
            (2..=256).contains(&cfg.head_dim) && cfg.head_dim.is_multiple_of(2),
            "recorded attention requires an even head width in 2..=256"
        );
        ensure!(
            cfg.eps.is_finite() && cfg.eps >= 0.0,
            "attention epsilon must be finite and nonnegative"
        );
        let theta = cfg.rope_base;
        ensure!(
            theta.is_finite() && theta > 0.0,
            "RoPE base must be finite and positive"
        );
        let q_width = cfg
            .n_heads
            .checked_mul(cfg.head_dim)
            .context("Q width overflows")?;
        let kv_width = cfg
            .n_kv_heads
            .checked_mul(cfg.head_dim)
            .context("KV width overflows")?;
        ensure!(
            q_width <= 4096 && kv_width <= 4096,
            "recorded attention projection width exceeds 4096"
        );
        let matrix = |bound: &Bound, rows: usize, cols: usize| -> Result<PlannedWeight> {
            ensure!(
                bound.tensor.kind().0 == 8,
                "recorded attention matrices must be Q8_0"
            );
            ensure!(
                bound.rows == rows && bound.cols == cols && cols.is_multiple_of(32),
                "recorded attention matrix shape differs"
            );
            let bytes = rows
                .checked_mul(cols / 32)
                .and_then(|n| n.checked_mul(34))
                .context("attention packed size overflows")?;
            let bytes = u64::try_from(bytes)?;
            ensure!(
                bound.tensor.nbytes() == bytes,
                "attention packed matrix span differs"
            );
            Ok(PlannedWeight {
                bound: bound.clone(),
                range: bound.tensor.locator().range(0..bytes)?,
            })
        };
        let norm = |bound: &Bound, len: usize| -> Result<PlannedWeight> {
            let bytes = u64::try_from(len)?
                .checked_mul(4)
                .context("attention norm size overflows")?;
            ensure!(
                bound.tensor.kind().0 == 0
                    && bound.rows == 1
                    && bound.cols == len
                    && bound.tensor.nbytes() == bytes,
                "attention norms must be exact F32 vectors"
            );
            Ok(PlannedWeight {
                bound: bound.clone(),
                range: bound.tensor.locator().range(0..bytes)?,
            })
        };
        Ok(DenseAttentionPlan {
            checkpoint_id: self.core.checkpoint_id,
            dim: cfg.dim,
            n_heads: cfg.n_heads,
            n_kv_heads: cfg.n_kv_heads,
            head_dim: cfg.head_dim,
            context: cfg.context,
            eps: cfg.eps,
            theta,
            weights: [
                matrix(&layer.wq, q_width, cfg.dim)?,
                matrix(&layer.wk, kv_width, cfg.dim)?,
                matrix(&layer.wv, kv_width, cfg.dim)?,
                matrix(&layer.wo, cfg.dim, q_width)?,
            ],
            norms: [
                norm(&layer.attn_norm, cfg.dim)?,
                norm(&layer.q_norm, cfg.head_dim)?,
                norm(&layer.k_norm, cfg.head_dim)?,
            ],
        })
    }
}

/// Maximum cache capacity admitted by on-demand metadata planning.
/// Backend support, memory admission and numerical qualification are separate.
pub const MAX_ON_DEMAND_CAPACITY: usize = 16_384;

impl DenseAttentionPlan {
    /// Width of the caller-owned hidden vector, in f32 elements.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of query heads.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    /// Number of key/value heads shared by groups of query heads.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    /// Number of coordinates per query, key and value head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    /// Finite, nonnegative RMSNorm epsilon bound from checkpoint metadata.
    pub fn eps(&self) -> f32 {
        self.eps
    }
    /// Positive, finite base frequency used for the split-half RoPE tables.
    pub fn rope_base(&self) -> f64 {
        self.theta
    }
    /// Total query/attention-output width across all query heads.
    pub fn query_width(&self) -> usize {
        self.n_heads * self.head_dim
    }
    /// Width of one position in each key/value cache.
    pub fn kv_width(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }
    /// Public commands per complete block, independent of backend dispatch lowering.
    pub fn semantic_commands_per_step(&self) -> usize {
        6 + 2 * (self.n_heads + self.n_kv_heads)
    }

    /// Additional retained host storage for on-demand binding, excluding the
    /// resident owner itself and backend-owned resource metadata/payloads.
    /// Counts the boxed prepared state and its seven cloned dimension vectors,
    /// with twofold vector storage and 64 bytes per heap allocation. This is a
    /// conservative layout estimate, not an allocator reservation.
    pub fn on_demand_host_metadata_bytes(&self) -> Result<u64> {
        self.weights.iter().chain(self.norms.iter()).try_fold(
            std::mem::size_of::<BoundDenseAttention>() as u64 + 64,
            |sum, weight| {
                u64::try_from(weight.bound.tensor.dims().len())?
                    .checked_mul(2 * std::mem::size_of::<u64>() as u64)
                    .and_then(|bytes| bytes.checked_add(64))
                    .and_then(|bytes| sum.checked_add(bytes))
                    .context("on-demand attention host metadata overflows")
            },
        )
    }

    /// Exact logical storage at a fixed capacity, constrained to 1..=256 and context.
    pub fn memory(&self, capacity: usize) -> Result<DenseAttentionMemory> {
        self.memory_with_capacity_limit(capacity, 256)
    }

    /// Exact logical storage for on-demand recording, constrained to 1..=16384 and
    /// checkpoint context. This does not imply backend support or qualification.
    pub fn memory_on_demand(&self, capacity: usize) -> Result<DenseAttentionMemory> {
        self.memory_with_capacity_limit(capacity, MAX_ON_DEMAND_CAPACITY)
    }

    fn memory_with_capacity_limit(
        &self,
        capacity: usize,
        maximum: usize,
    ) -> Result<DenseAttentionMemory> {
        ensure!(
            (1..=maximum).contains(&capacity) && capacity <= self.context,
            "attention cache capacity must be positive, at most {maximum} and within model context"
        );
        let packed_weight_bytes = self.weights.iter().try_fold(0u64, |sum, w| {
            sum.checked_add(w.range.nbytes())
                .context("attention packed total overflows")
        })?;
        let q = self.query_width();
        let kv = self.kv_width();
        let parts = [
            self.dim,
            self.dim,
            self.head_dim,
            self.head_dim,
            q,
            q,
            q,
            kv,
            capacity.checked_mul(kv).context("K cache overflows")?,
            capacity.checked_mul(kv).context("V cache overflows")?,
            capacity
                .checked_mul(self.head_dim)
                .context("coefficient table overflows")?,
        ];
        let elements = parts.into_iter().try_fold(0usize, |sum, n| {
            sum.checked_add(n)
                .context("attention resident total overflows")
        })?;
        let resident_tensor_bytes = u64::try_from(elements)?
            .checked_mul(4)
            .context("attention tensor bytes overflow")?;
        let matrix_staging = self.weights.iter().map(|w| w.range.nbytes()).max().unwrap();
        let norm_staging = self
            .norms
            .iter()
            .map(|w| w.range.nbytes())
            .max()
            .unwrap()
            .checked_mul(2)
            .context("norm staging overflows")?;
        let coefficient_staging = u64::try_from(capacity)?
            .checked_mul(self.head_dim as u64)
            .and_then(|n| n.checked_mul(4))
            .context("coefficient staging overflows")?;
        Ok(DenseAttentionMemory {
            packed_weight_bytes,
            resident_tensor_bytes,
            largest_staging_bytes: matrix_staging.max(norm_staging).max(coefficient_staging),
        })
    }

    /// Upload seven source tensors once, generate resident coefficients, and allocate
    /// fixed cache and scratch. Payload staging is sequential; no whole-model load.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        capacity: usize,
        limits: DenseAttentionLimits,
    ) -> Result<PreparedDenseAttention> {
        self.prepare_with_recording_mode(device, source, capacity, limits, false)
    }

    /// Prepare capacity 1..=16384 for on-demand recording only. All bounds and
    /// logical limits precede payload reads and allocations. The returned owner
    /// refuses `record`; bind it with `record_on_demand` instead. Backend command
    /// support is checked when each position is recorded, with no fallback.
    pub fn prepare_on_demand(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        capacity: usize,
        limits: DenseAttentionLimits,
    ) -> Result<PreparedDenseAttention> {
        self.prepare_with_recording_mode(device, source, capacity, limits, true)
    }

    fn prepare_with_recording_mode(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        capacity: usize,
        limits: DenseAttentionLimits,
        on_demand_only: bool,
    ) -> Result<PreparedDenseAttention> {
        ensure!(
            source.checkpoint_id() == self.checkpoint_id,
            "attention plan and source belong to different checkpoints"
        );
        let memory = if on_demand_only {
            self.memory_on_demand(capacity)?
        } else {
            self.memory(capacity)?
        };
        for (label, required, limit) in [
            (
                "packed",
                memory.packed_weight_bytes,
                limits.packed_weight_bytes,
            ),
            (
                "tensors",
                memory.resident_tensor_bytes,
                limits.resident_tensor_bytes,
            ),
            (
                "staging",
                memory.largest_staging_bytes,
                limits.largest_staging_bytes,
            ),
        ] {
            ensure!(
                required <= limit,
                "attention {label} needs {required} logical bytes, limit is {limit}"
            );
        }
        for weight in &self.weights {
            ensure!(
                device.supports(weight.bound.tensor.kind()),
                "backend refuses attention weight format"
            );
        }
        let upload = |weight: &PlannedWeight| -> Result<PreparedWeight> {
            let resident = source.load_range(&weight.range, WeightAccess::Cyclic)?;
            ensure!(
                resident.bytes().len() as u64 == weight.range.nbytes(),
                "source returned an incorrect matrix span"
            );
            let encoded = device.upload_weight(&Weight {
                bytes: resident.bytes(),
                kind: weight.bound.tensor.kind(),
                rows: weight.bound.rows,
                cols: weight.bound.cols,
            })?;
            Ok(device.prepare_weight(&encoded)?)
        };
        let weights = [
            upload(&self.weights[0])?,
            upload(&self.weights[1])?,
            upload(&self.weights[2])?,
            upload(&self.weights[3])?,
        ];
        let upload_norm = |weight: &PlannedWeight| -> Result<F32Tensor> {
            let resident = source.load_range(&weight.range, WeightAccess::Cyclic)?;
            ensure!(
                resident.bytes().len() as u64 == weight.range.nbytes(),
                "source returned an incorrect norm span"
            );
            let mut values = Vec::new();
            values
                .try_reserve_exact(weight.bound.cols)
                .context("allocating norm staging")?;
            values.extend(
                resident
                    .bytes()
                    .chunks_exact(4)
                    .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
            );
            Ok(device.upload_f32(&values)?)
        };
        let norms = [
            upload_norm(&self.norms[0])?,
            upload_norm(&self.norms[1])?,
            upload_norm(&self.norms[2])?,
        ];
        let (cos_values, sin_values) = coefficients(capacity, self.head_dim, self.theta)?;
        let cos = device.upload_f32(&cos_values)?;
        let sin = device.upload_f32(&sin_values)?;
        drop((cos_values, sin_values));
        Ok(PreparedDenseAttention {
            plan: self.clone(),
            capacity,
            on_demand_only,
            weights,
            norms,
            cos,
            sin,
            normalized: device.allocate_f32(self.dim)?,
            raw_q: device.allocate_f32(self.query_width())?,
            raw_k: device.allocate_f32(self.kv_width())?,
            q: device.allocate_f32(self.query_width())?,
            attended: device.allocate_f32(self.query_width())?,
            k_cache: device.allocate_f32(capacity * self.kv_width())?,
            v_cache: device.allocate_f32(capacity * self.kv_width())?,
        })
    }
}

fn coefficients(capacity: usize, head_dim: usize, theta: f64) -> Result<(Vec<f32>, Vec<f32>)> {
    let len = capacity
        .checked_mul(head_dim / 2)
        .context("coefficient length overflows")?;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    cos.try_reserve_exact(len)
        .context("allocating cosine table")?;
    sin.try_reserve_exact(len)
        .context("allocating sine table")?;
    for position in 0..capacity {
        for index in 0..head_dim / 2 {
            let frequency = 1.0 / theta.powf((2 * index) as f64 / head_dim as f64);
            let angle = position as f64 * frequency;
            cos.push(angle.cos() as f32);
            sin.push(angle.sin() as f32);
        }
    }
    Ok((cos, sin))
}

/// Owned static operands, scratch and fixed cache before binding caller hidden.
#[derive(Debug)]
pub struct PreparedDenseAttention {
    plan: DenseAttentionPlan,
    capacity: usize,
    on_demand_only: bool,
    weights: [PreparedWeight; 4],
    norms: [F32Tensor; 3],
    normalized: F32Tensor,
    raw_q: F32Tensor,
    raw_k: F32Tensor,
    q: F32Tensor,
    attended: F32Tensor,
    k_cache: F32Tensor,
    v_cache: F32Tensor,
    cos: F32Tensor,
    sin: F32Tensor,
}

impl PreparedDenseAttention {
    /// Consume preparation to bind hidden and retain one batch for each position.
    /// The returned cache is deliberately not Clone: its batches share scratch and
    /// must execute sequentially. Caller hidden writes must finish before `step`.
    pub fn record(
        self,
        device: &dyn ExecutionDevice,
        hidden: &F32TensorView,
    ) -> Result<ResidentDenseAttention> {
        ensure!(
            !self.on_demand_only,
            "on-demand attention preparation cannot retain batches for every position"
        );
        ensure!(
            hidden.len() == self.plan.dim,
            "attention hidden width differs"
        );
        let mut batches = Vec::new();
        batches
            .try_reserve_exact(self.capacity)
            .context("allocating attention batch table")?;
        for position in 0..self.capacity {
            batches.push(device.record(&self.position_commands(hidden, position)?)?);
        }
        // Batches retain all resources and hidden. Dropping preparation releases
        // duplicate handles, not their allocations. There are no escaped KV views.
        Ok(ResidentDenseAttention {
            recording: AttentionRecording::PreRecorded(batches),
            visible: 0,
            poisoned: false,
        })
    }

    /// Bind hidden while retaining prepared operands instead of all position batches.
    /// Binding validates semantic shape/ownership without backend recording or writes.
    /// Each `step` records one immutable batch and retires it after successful wait.
    /// Backend-specific command support is checked at that step, with no fallback.
    pub fn record_on_demand(self, hidden: &F32TensorView) -> Result<ResidentDenseAttention> {
        self.position_commands(hidden, 0)?;
        Ok(ResidentDenseAttention {
            recording: AttentionRecording::OnDemand(Box::new(BoundDenseAttention {
                prepared: self,
                hidden: hidden.clone(),
            })),
            visible: 0,
            poisoned: false,
        })
    }

    fn position_commands(&self, hidden: &F32TensorView, position: usize) -> Result<CommandList> {
        ensure!(
            hidden.len() == self.plan.dim,
            "attention hidden width differs"
        );
        ensure!(position < self.capacity, "attention cache is full");
        let p = &self.plan;
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &self.normalized.full_view(),
            hidden,
            Some(&self.norms[0].full_view()),
            p.eps,
        )?;
        commands.linear(
            &self.raw_q.full_view(),
            &self.normalized.full_view(),
            &self.weights[0],
            LinearMode::Overwrite,
        )?;
        commands.linear(
            &self.raw_k.full_view(),
            &self.normalized.full_view(),
            &self.weights[1],
            LinearMode::Overwrite,
        )?;
        commands.linear(
            &self.v_cache.view(position * p.kv_width(), p.kv_width())?,
            &self.normalized.full_view(),
            &self.weights[2],
            LinearMode::Overwrite,
        )?;
        let cos = self.cos.view(position * (p.head_dim / 2), p.head_dim / 2)?;
        let sin = self.sin.view(position * (p.head_dim / 2), p.head_dim / 2)?;
        for head in 0..p.n_heads {
            let target = self.q.view(head * p.head_dim, p.head_dim)?;
            commands.rmsnorm(
                &target,
                &self.raw_q.view(head * p.head_dim, p.head_dim)?,
                Some(&self.norms[1].full_view()),
                p.eps,
            )?;
            commands.rope(&target, &cos, &sin, RopeLayout::SplitHalf, false)?;
        }
        for head in 0..p.n_kv_heads {
            let target = self
                .k_cache
                .view(position * p.kv_width() + head * p.head_dim, p.head_dim)?;
            commands.rmsnorm(
                &target,
                &self.raw_k.view(head * p.head_dim, p.head_dim)?,
                Some(&self.norms[2].full_view()),
                p.eps,
            )?;
            commands.rope(&target, &cos, &sin, RopeLayout::SplitHalf, false)?;
        }
        commands.incremental_causal_attention(
            &self.attended.full_view(),
            &self.q.full_view(),
            &self.k_cache.view(0, (position + 1) * p.kv_width())?,
            &self.v_cache.view(0, (position + 1) * p.kv_width())?,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: position,
                n_kv_positions: position + 1,
                n_heads: p.n_heads,
                n_kv_heads: p.n_kv_heads,
                head_dim: p.head_dim,
                scale: (p.head_dim as f32).sqrt().recip(),
            },
        )?;
        commands.linear(
            hidden,
            &self.attended.full_view(),
            &self.weights[3],
            LinearMode::ScaledAdd { scale: 1.0 },
        )?;
        debug_assert_eq!(commands.len(), p.semantic_commands_per_step());
        Ok(commands)
    }
}

#[derive(Debug)]
struct BoundDenseAttention {
    prepared: PreparedDenseAttention,
    hidden: F32TensorView,
}

impl BoundDenseAttention {
    fn execute_position<B>(
        &self,
        position: usize,
        record: impl FnOnce(&CommandList) -> Result<B>,
        execute_and_wait: impl FnOnce(&B) -> Result<()>,
    ) -> Result<()> {
        let commands = self.prepared.position_commands(&self.hidden, position)?;
        let batch = record(&commands)?;
        drop(commands);
        execute_and_wait(&batch)?;
        // No batch/submission escapes this position. The backend must retain any
        // uncertain in-flight resources itself; healthy resources can now retire.
        drop(batch);
        Ok(())
    }
}

#[derive(Debug)]
enum AttentionRecording {
    PreRecorded(Vec<RecordedBatch>),
    OnDemand(Box<BoundDenseAttention>),
}

/// Sequential cache lifecycle, owning all batches or prepared operands and KV state.
#[derive(Debug)]
pub struct ResidentDenseAttention {
    recording: AttentionRecording,
    visible: usize,
    poisoned: bool,
}
impl ResidentDenseAttention {
    /// Fixed maximum number of positions in one request.
    pub fn capacity(&self) -> usize {
        match &self.recording {
            AttentionRecording::PreRecorded(batches) => batches.len(),
            AttentionRecording::OnDemand(bound) => bound.prepared.capacity,
        }
    }
    /// Whether attention records and retires one batch per step.
    pub fn records_on_demand(&self) -> bool {
        matches!(&self.recording, AttentionRecording::OnDemand(_))
    }
    /// Number of positions whose complete attention/residual batch has finished.
    pub fn visible_positions(&self) -> usize {
        self.visible
    }
    /// Whether an uncertain submission has made further use and reset invalid.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Execute the whole block and publish exactly one position after successful wait.
    /// Any recording/submit/wait error or unwind leaves it poisoned and unpublished.
    pub fn step(&mut self, device: &dyn ExecutionDevice) -> Result<()> {
        self.step_with(|recording, position| {
            let execute = |batch: &RecordedBatch| {
                let submission = device.submit(batch)?;
                device.wait(&submission)?;
                drop(submission);
                Ok(())
            };
            match recording {
                AttentionRecording::PreRecorded(batches) => execute(&batches[position]),
                AttentionRecording::OnDemand(bound) => bound.execute_position(
                    position,
                    |commands| Ok(device.record(commands)?),
                    execute,
                ),
            }
        })
    }

    // One shared transition keeps recording/execution errors and unwinds poisoned,
    // including a wait failure after hidden and the KV tail may have changed.
    fn step_with(
        &mut self,
        execute_and_wait: impl FnOnce(&AttentionRecording, usize) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "attention cache is poisoned; create a new preparation"
        );
        ensure!(self.visible < self.capacity(), "attention cache is full");
        self.poisoned = true;
        execute_and_wait(&self.recording, self.visible)?;
        self.visible += 1;
        self.poisoned = false;
        Ok(())
    }
    /// Start another request without reallocating. Only the visible count changes;
    /// every newly visible K/V row is overwritten before attention can consume it.
    pub fn reset(&mut self) -> Result<()> {
        if self.poisoned {
            bail!("cannot reset a poisoned attention cache");
        }
        self.visible = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::CpuExecutionDevice;
    use beholdd_backend::Device;
    use beholdd_format::Gguf;
    use beholdd_memory::{ResidentSource, ResidentWeight};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;
    struct Fixture {
        checkpoint: Gguf,
        path: PathBuf,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.path);
        }
    }

    fn string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }

    // Real GGUF metadata/locators and the strict production binder are used, with
    // small deterministic matrices. FFN kinds deliberately differ from each other.
    fn make_fixture(dense: bool, down_kind: u32) -> Fixture {
        make_fixture_with_norm(dense, down_kind, 0)
    }

    fn make_fixture_with_norm(dense: bool, down_kind: u32, norm_kind: u32) -> Fixture {
        let arch = if dense { "qwen3" } else { "qwen3moe" };
        let mut tensors: Vec<(String, Vec<u64>, u32)> = vec![
            ("token_embd.weight".into(), vec![32, 2], 0),
            ("output_norm.weight".into(), vec![32], 0),
        ];
        for name in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"] {
            let kind = if name == "ffn_norm" { norm_kind } else { 0 };
            tensors.push((format!("blk.0.{name}.weight"), vec![32], kind));
        }
        for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            tensors.push((format!("blk.0.{name}.weight"), vec![32, 32], 8));
        }
        if dense {
            tensors.extend([
                ("blk.0.ffn_gate.weight".into(), vec![32, 64], 30),
                ("blk.0.ffn_up.weight".into(), vec![32, 64], 8),
                ("blk.0.ffn_down.weight".into(), vec![64, 32], down_kind),
            ]);
        } else {
            tensors.extend([
                ("blk.0.ffn_gate_inp.weight".into(), vec![32, 2], 0),
                ("blk.0.ffn_gate_exps.weight".into(), vec![32, 64, 2], 30),
                ("blk.0.ffn_up_exps.weight".into(), vec![32, 64, 2], 8),
                ("blk.0.ffn_down_exps.weight".into(), vec![64, 32, 2], 8),
            ]);
        }
        let mut metadata = vec![
            ("block_count", 1u32),
            ("embedding_length", 32),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 32),
            ("attention.value_length", 32),
            ("feed_forward_length", 64),
            ("context_length", 16),
        ];
        if !dense {
            metadata.extend([
                ("expert_count", 2),
                ("expert_used_count", 1),
                ("expert_feed_forward_length", 64),
            ]);
        }
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend((tensors.len() as u64).to_le_bytes());
        header.extend((metadata.len() as u64 + 3).to_le_bytes());
        string(&mut header, "general.architecture");
        header.extend(8u32.to_le_bytes());
        string(&mut header, arch);
        for (key, value) in metadata {
            string(&mut header, &format!("{arch}.{key}"));
            header.extend(4u32.to_le_bytes());
            header.extend(value.to_le_bytes());
        }
        for (key, value) in [
            ("attention.layer_norm_rms_epsilon", 1e-6f32),
            ("rope.freq_base", 10_000.0),
        ] {
            string(&mut header, &format!("{arch}.{key}"));
            header.extend(6u32.to_le_bytes());
            header.extend(value.to_le_bytes());
        }
        let mut payload = Vec::new();
        for (tensor_index, (name, dims, kind)) in tensors.iter().enumerate() {
            string(&mut header, name);
            header.extend((dims.len() as u32).to_le_bytes());
            for dim in dims {
                header.extend(dim.to_le_bytes());
            }
            header.extend(kind.to_le_bytes());
            header.extend((payload.len() as u64).to_le_bytes());
            let elements = dims.iter().product::<u64>() as usize;
            match *kind {
                0 | 30 => {
                    for index in 0..elements {
                        let value =
                            (((index * 3 + index / 32 + tensor_index) % 11) as f32 - 5.0) * 0.125;
                        if *kind == 0 {
                            payload.extend(value.to_le_bytes());
                        } else {
                            payload.extend(((value.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                }
                8 => {
                    for block in 0..elements / 32 {
                        payload.extend(0x3400u16.to_le_bytes()); // scale = 0.25
                        payload.extend(
                            (0..32).map(|lane| {
                                (((block * 3 + lane + tensor_index) % 7) as i8 - 3) as u8
                            }),
                        );
                    }
                }
                2 => payload.resize(payload.len() + elements / 32 * 18, 0),
                _ => panic!("unimplemented fixture format"),
            }
            payload.resize(payload.len().next_multiple_of(32), 0);
        }
        header.resize(header.len().next_multiple_of(32), 0);
        header.extend(payload);
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "beholdd-recorded-attention-{}-{}.gguf",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::write(&path, header).unwrap();
        Fixture {
            checkpoint: Gguf::open(&path).unwrap(),
            path,
        }
    }

    struct CountingSource<'a> {
        source: ResidentSource<'a>,
        reads: Mutex<Vec<WeightRange>>,
    }

    impl<'a> CountingSource<'a> {
        fn new(checkpoint: &'a Gguf) -> Self {
            Self {
                source: ResidentSource::new(checkpoint),
                reads: Mutex::new(Vec::new()),
            }
        }
    }

    impl WeightSource for CountingSource<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.source.checkpoint_id()
        }

        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            assert_eq!(access, WeightAccess::Cyclic);
            self.reads.lock().unwrap().push(*range);
            self.source.load_range(range, access)
        }
    }

    fn limits(plan: &DenseAttentionPlan, capacity: usize) -> DenseAttentionLimits {
        let m = plan.memory(capacity).unwrap();
        DenseAttentionLimits {
            packed_weight_bytes: m.packed_weight_bytes,
            resident_tensor_bytes: m.resident_tensor_bytes,
            largest_staging_bytes: m.largest_staging_bytes,
        }
    }

    fn reference(
        plan: &DenseAttentionPlan,
        checkpoint: &Gguf,
        hidden: &[f32],
        k: &mut Vec<f32>,
        v: &mut Vec<f32>,
    ) -> Vec<f32> {
        let cpu = Cpu::new();
        let load_norm = |index: usize| {
            let resident =
                ResidentWeight::load_range(checkpoint, &plan.norms[index].range).unwrap();
            resident
                .bytes()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect::<Vec<_>>()
        };
        let project = |index: usize, input: &[f32]| {
            let w = &plan.weights[index];
            let resident = ResidentWeight::load_range(checkpoint, &w.range).unwrap();
            let mut output = vec![0.0; w.bound.rows];
            cpu.matvec_with_threads(
                &mut output,
                input,
                &Weight {
                    bytes: resident.bytes(),
                    kind: w.bound.tensor.kind(),
                    rows: w.bound.rows,
                    cols: w.bound.cols,
                },
                1,
            )
            .unwrap();
            output
        };
        let mut normalized = vec![0.0; plan.dim];
        cpu.rmsnorm(&mut normalized, hidden, Some(&load_norm(0)), plan.eps)
            .unwrap();
        let raw_q = project(0, &normalized);
        let raw_k = project(1, &normalized);
        let raw_v = project(2, &normalized);
        let position = k.len() / plan.kv_width();
        let (cos, sin) = coefficients(position + 1, plan.head_dim, plan.theta).unwrap();
        let offset = position * plan.head_dim / 2;
        let norm_rope = |raw: &[f32], weights: &[f32]| {
            let mut normalized = vec![0.0; raw.len()];
            for (input, output) in raw
                .chunks_exact(plan.head_dim)
                .zip(normalized.chunks_exact_mut(plan.head_dim))
            {
                cpu.rmsnorm(output, input, Some(weights), plan.eps).unwrap();
                cpu.rope(
                    output,
                    &cos[offset..],
                    &sin[offset..],
                    RopeLayout::SplitHalf,
                    false,
                )
                .unwrap();
            }
            normalized
        };
        let q = norm_rope(&raw_q, &load_norm(1));
        k.extend(norm_rope(&raw_k, &load_norm(2)));
        v.extend(raw_v);
        let mut attended = vec![0.0; plan.query_width()];
        cpu.incremental_causal_attention(
            &mut attended,
            &q,
            k,
            v,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: position,
                n_kv_positions: position + 1,
                n_heads: plan.n_heads,
                n_kv_heads: plan.n_kv_heads,
                head_dim: plan.head_dim,
                scale: (plan.head_dim as f32).sqrt().recip(),
            },
        )
        .unwrap();
        hidden
            .iter()
            .zip(project(3, &attended))
            .map(|(a, b)| a + b)
            .collect()
    }

    #[test]
    fn resident_cache_matches_immediate_attention_across_positions_and_reset() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_attention_plan(0).unwrap();
        for on_demand in [false, true] {
            let source = CountingSource::new(&fixture.checkpoint);
            let device = CpuExecutionDevice::new();
            let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
            assert_eq!(source.reads.lock().unwrap().len(), 7);
            let hidden = device.allocate_f32(plan.dim()).unwrap();
            let mut cache = if on_demand {
                prepared.record_on_demand(&hidden.full_view()).unwrap()
            } else {
                prepared.record(&device, &hidden.full_view()).unwrap()
            };
            assert_eq!(cache.records_on_demand(), on_demand);
            assert_eq!(cache.capacity(), 3);
            if let AttentionRecording::PreRecorded(batches) = &cache.recording {
                assert_eq!(batches[0].len(), plan.semantic_commands_per_step());
            }
            for request in 0..2 {
                let mut k = Vec::new();
                let mut v = Vec::new();
                assert_eq!(cache.visible_positions(), 0);
                for position in 0..3 {
                    let input = (0..plan.dim())
                        .map(|i| (((i * 7 + position * 3 + request * 5) % 23) as f32 - 11.0) / 16.0)
                        .collect::<Vec<_>>();
                    let expected = reference(&plan, &fixture.checkpoint, &input, &mut k, &mut v);
                    device.write_f32(&hidden.full_view(), &input).unwrap();
                    cache.step(&device).unwrap();
                    assert_eq!(cache.visible_positions(), position + 1);
                    let mut actual = vec![f32::NAN; plan.dim()];
                    device.read_f32(&hidden.full_view(), &mut actual).unwrap();
                    assert_eq!(actual, expected);
                }
                assert!(cache.step(&device).is_err());
                assert!(!cache.is_poisoned());
                assert_eq!(cache.visible_positions(), 3);
                cache.reset().unwrap();
            }
            assert_eq!(source.reads.lock().unwrap().len(), 7);
        }
    }

    #[test]
    fn on_demand_capacity_512_matches_immediate_attention_and_refuses_legacy_recording() {
        let fixture = make_fixture(true, 8);
        let mut plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        plan.context = 512;
        let memory = plan.memory_on_demand(512).unwrap();
        assert_eq!(
            memory,
            DenseAttentionMemory {
                packed_weight_bytes: 4352,
                resident_tensor_bytes: 197632,
                largest_staging_bytes: 65536,
            }
        );
        let limits = DenseAttentionLimits {
            packed_weight_bytes: memory.packed_weight_bytes,
            resident_tensor_bytes: memory.resident_tensor_bytes,
            largest_staging_bytes: memory.largest_staging_bytes,
        };
        let device = CpuExecutionDevice::new();
        let source = CountingSource::new(&fixture.checkpoint);
        for capacity in [0, 513, usize::MAX] {
            assert!(plan.memory_on_demand(capacity).is_err());
            assert!(plan
                .prepare_on_demand(&device, &source, capacity, limits)
                .is_err());
        }
        assert!(plan.memory(257).is_err());
        assert!(plan.prepare(&device, &source, 512, limits).is_err());
        for field in 0..3 {
            let mut insufficient = limits;
            match field {
                0 => insufficient.packed_weight_bytes -= 1,
                1 => insufficient.resident_tensor_bytes -= 1,
                _ => insufficient.largest_staging_bytes -= 1,
            }
            assert!(plan
                .prepare_on_demand(&device, &source, 512, insufficient)
                .is_err());
        }
        plan.context = 511;
        assert!(plan
            .prepare_on_demand(&device, &source, 512, limits)
            .is_err());
        plan.context = 512;
        assert!(source.reads.lock().unwrap().is_empty());
        let hidden = device.allocate_f32(plan.dim()).unwrap();
        // The mode boundary applies even below the legacy capacity ceiling.
        for capacity in [3, 512] {
            let prepared = plan
                .prepare_on_demand(&device, &source, capacity, limits)
                .unwrap();
            let error = prepared.record(&device, &hidden.full_view()).unwrap_err();
            assert!(error
                .to_string()
                .contains("on-demand attention preparation"));
        }
        let prepared = plan
            .prepare_on_demand(&device, &source, 512, limits)
            .unwrap();
        let mut cache = prepared.record_on_demand(&hidden.full_view()).unwrap();
        let reads = source.reads.lock().unwrap().len();
        let mut k = Vec::new();
        let mut v = Vec::new();
        let mut first = Vec::new();
        for position in 0..512 {
            let input = (0..plan.dim())
                .map(|i| (((i * 7 + position * 3) % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>();
            let expected = reference(&plan, &fixture.checkpoint, &input, &mut k, &mut v);
            device.write_f32(&hidden.full_view(), &input).unwrap();
            cache.step(&device).unwrap();
            let mut actual = vec![f32::NAN; plan.dim()];
            device.read_f32(&hidden.full_view(), &mut actual).unwrap();
            assert_eq!(actual, expected, "position {position}");
            assert_eq!(cache.visible_positions(), position + 1);
            if position == 0 {
                first = actual;
            }
        }
        assert!(cache.step(&device).is_err());
        assert!(!cache.is_poisoned());
        assert_eq!(cache.visible_positions(), 512);
        cache.reset().unwrap();
        let input = (0..plan.dim())
            .map(|i| (((i * 7) % 23) as f32 - 11.0) / 16.0)
            .collect::<Vec<_>>();
        device.write_f32(&hidden.full_view(), &input).unwrap();
        cache.step(&device).unwrap();
        let mut replay = vec![f32::NAN; plan.dim()];
        device.read_f32(&hidden.full_view(), &mut replay).unwrap();
        assert_eq!(replay, first);
        assert_eq!(source.reads.lock().unwrap().len(), reads);
    }

    #[test]
    fn on_demand_capacity_16384_limits_refuse_before_reads() {
        let fixture = make_fixture(true, 8);
        let mut plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        plan.context = 32768;
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        // These are metadata and pre-I/O refusal checks: no cache is allocated.
        for capacity in [2048, 2049, 4096, 8192, 16384] {
            let memory = plan.memory_on_demand(capacity).unwrap();
            assert_eq!(
                memory,
                DenseAttentionMemory {
                    packed_weight_bytes: 4352,
                    resident_tensor_bytes: 1024 + 384 * capacity as u64,
                    largest_staging_bytes: 128 * capacity as u64,
                }
            );
            let limits = DenseAttentionLimits {
                packed_weight_bytes: memory.packed_weight_bytes,
                resident_tensor_bytes: memory.resident_tensor_bytes,
                largest_staging_bytes: memory.largest_staging_bytes,
            };
            for field in 0..3 {
                let mut insufficient = limits;
                match field {
                    0 => insufficient.packed_weight_bytes -= 1,
                    1 => insufficient.resident_tensor_bytes -= 1,
                    _ => insufficient.largest_staging_bytes -= 1,
                }
                assert!(plan
                    .prepare_on_demand(&device, &source, capacity, insufficient)
                    .is_err());
            }
            assert!(plan.prepare(&device, &source, capacity, limits).is_err());
            plan.context = capacity - 1;
            assert!(plan
                .prepare_on_demand(&device, &source, capacity, limits)
                .is_err());
            plan.context = 32768;
        }
        let unlimited = DenseAttentionLimits {
            packed_weight_bytes: u64::MAX,
            resident_tensor_bytes: u64::MAX,
            largest_staging_bytes: u64::MAX,
        };
        for capacity in [0, 16385, usize::MAX] {
            assert!(plan.memory_on_demand(capacity).is_err());
            assert!(plan
                .prepare_on_demand(&device, &source, capacity, unlimited)
                .is_err());
        }
        for capacity in [257, 512, 2048] {
            assert!(plan.prepare(&device, &source, capacity, unlimited).is_err());
        }
        assert!(source.reads.lock().unwrap().is_empty());
    }

    #[test]
    fn on_demand_extended_positions_match_seeded_prefix_and_reset() {
        let fixture = make_fixture(true, 8);
        let mut plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        plan.context = 2048;
        let memory = plan.memory_on_demand(2048).unwrap();
        let limits = DenseAttentionLimits {
            packed_weight_bytes: memory.packed_weight_bytes,
            resident_tensor_bytes: memory.resident_tensor_bytes,
            largest_staging_bytes: memory.largest_staging_bytes,
        };
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let hidden = device.allocate_f32(plan.dim()).unwrap();
        let prepared = plan
            .prepare_on_demand(&device, &source, 2048, limits)
            .unwrap();
        assert!(prepared
            .position_commands(&hidden.full_view(), 2048)
            .is_err());
        let mut cache = prepared.record_on_demand(&hidden.full_view()).unwrap();
        // Seed the same finite prior KV into both implementations. This exercises
        // extended absolute RoPE positions and visible-prefix binding without an
        // O(capacity^2) full prefill or a claim about a sequential GPU run.
        for position in [512, 1024, 2047] {
            let mut k = (0..position * plan.kv_width())
                .map(|i| ((i % 17) as f32 - 8.0) / 32.0)
                .collect::<Vec<_>>();
            let mut v = (0..k.len())
                .map(|i| ((i % 13) as f32 - 6.0) / 16.0)
                .collect::<Vec<_>>();
            let AttentionRecording::OnDemand(bound) = &cache.recording else {
                panic!("expected on-demand attention")
            };
            for (tensor, prefix) in [(&bound.prepared.k_cache, &k), (&bound.prepared.v_cache, &v)] {
                device
                    .write_f32(&tensor.full_view(), &vec![f32::NAN; tensor.len()])
                    .unwrap();
                device
                    .write_f32(&tensor.view(0, prefix.len()).unwrap(), prefix)
                    .unwrap();
            }
            cache.visible = position;
            let input = (0..plan.dim())
                .map(|i| (((i * 7 + position * 3) % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>();
            let expected = reference(&plan, &fixture.checkpoint, &input, &mut k, &mut v);
            device.write_f32(&hidden.full_view(), &input).unwrap();
            cache.step(&device).unwrap();
            let mut actual = vec![f32::NAN; plan.dim()];
            device.read_f32(&hidden.full_view(), &mut actual).unwrap();
            assert_eq!(actual, expected, "absolute position {position}");
            assert_eq!(cache.visible_positions(), position + 1);
            assert!(!cache.is_poisoned());
        }
        assert!(cache.step(&device).is_err());
        assert_eq!(cache.visible_positions(), 2048);
        assert!(!cache.is_poisoned());
        cache.reset().unwrap();
        assert_eq!(cache.visible_positions(), 0);
        let input = vec![0.25; plan.dim()];
        let expected = reference(
            &plan,
            &fixture.checkpoint,
            &input,
            &mut Vec::new(),
            &mut Vec::new(),
        );
        device.write_f32(&hidden.full_view(), &input).unwrap();
        cache.step(&device).unwrap();
        let mut actual = vec![f32::NAN; plan.dim()];
        device.read_f32(&hidden.full_view(), &mut actual).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(cache.visible_positions(), 1);
        assert_eq!(source.reads.lock().unwrap().len(), 7);
    }

    #[test]
    fn source_and_limits_refuse_before_reads_and_failed_steps_never_publish() {
        let fixture = make_fixture(true, 8);
        let foreign = make_fixture(true, 8);
        let plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        let device = CpuExecutionDevice::new();
        let source = CountingSource::new(&fixture.checkpoint);
        let other = CountingSource::new(&foreign.checkpoint);
        assert!(plan.prepare(&device, &other, 3, limits(&plan, 3)).is_err());
        for field in 0..3 {
            let mut limit = limits(&plan, 3);
            match field {
                0 => limit.packed_weight_bytes -= 1,
                1 => limit.resident_tensor_bytes -= 1,
                _ => limit.largest_staging_bytes -= 1,
            }
            assert!(plan.prepare(&device, &source, 3, limit).is_err());
        }
        assert!(plan.memory(0).is_err());
        assert!(plan.memory(257).is_err());
        assert!(plan.memory(17).is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(other.reads.lock().unwrap().is_empty());
        let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
        let hidden = device.allocate_f32(plan.dim()).unwrap();
        let mut cache = prepared.record(&device, &hidden.full_view()).unwrap();
        assert!(cache.step(&CpuExecutionDevice::new()).is_err());
        assert_eq!(cache.visible_positions(), 0);
        assert!(cache.is_poisoned());
        assert!(cache.step(&device).is_err());
        assert!(cache.reset().is_err());
    }

    #[test]
    fn post_submit_error_and_unwind_poison_without_publishing() {
        let fixture = make_fixture(true, 8);
        let plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        for unwind in [false, true] {
            let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
            let hidden = device.upload_f32(&vec![0.25; plan.dim()]).unwrap();
            let mut cache = prepared.record(&device, &hidden.full_view()).unwrap();
            // Start from an already-published prefix, then fail after the next
            // real CPU submission. The old prefix remains the published count.
            cache.step(&device).unwrap();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cache.step_with(|recording, position| {
                    let AttentionRecording::PreRecorded(batches) = recording else {
                        panic!("expected pre-recorded attention")
                    };
                    let submission = device.submit(&batches[position])?;
                    device.wait(&submission)?;
                    if unwind {
                        panic!("injected post-submit wait unwind");
                    }
                    bail!("injected post-submit wait error")
                })
            }));
            if unwind {
                assert!(outcome.is_err());
            } else {
                assert!(outcome.unwrap().is_err());
            }
            assert_eq!(cache.visible_positions(), 1);
            assert!(cache.is_poisoned());
            assert!(cache.reset().is_err());
            assert!(cache.step(&device).is_err());
        }
    }

    #[test]
    fn on_demand_binding_checks_hidden_and_retires_each_completed_batch() {
        use std::cell::Cell;

        struct ObservedBatch<'a> {
            batch: RecordedBatch,
            live: &'a Cell<usize>,
        }
        impl Drop for ObservedBatch<'_> {
            fn drop(&mut self) {
                assert_eq!(self.live.replace(0), 1);
            }
        }

        let fixture = make_fixture(true, 8);
        let plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        for hidden in [
            device.allocate_f32(plan.dim() - 1).unwrap(),
            CpuExecutionDevice::new().allocate_f32(plan.dim()).unwrap(),
        ] {
            let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
            assert!(prepared.record_on_demand(&hidden.full_view()).is_err());
        }
        let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
        let hidden = device.upload_f32(&vec![0.25; plan.dim()]).unwrap();
        let mut cache = prepared.record_on_demand(&hidden.full_view()).unwrap();
        let reads = source.reads.lock().unwrap().len();
        let live = Cell::new(0);
        let recordings = Cell::new(0);
        for _request in 0..2 {
            for position in 0..3 {
                cache
                    .step_with(|recording, visible| {
                        assert_eq!(visible, position);
                        let AttentionRecording::OnDemand(bound) = recording else {
                            panic!("expected on-demand attention")
                        };
                        bound.execute_position(
                            visible,
                            |commands| {
                                assert_eq!(live.get(), 0, "the previous batch must have retired");
                                let batch = device.record(commands)?;
                                assert_eq!(batch.len(), plan.semantic_commands_per_step());
                                live.set(1);
                                recordings.set(recordings.get() + 1);
                                Ok(ObservedBatch { batch, live: &live })
                            },
                            |batch| {
                                assert_eq!(live.get(), 1);
                                let submission = device.submit(&batch.batch)?;
                                device.wait(&submission)?;
                                drop(submission);
                                Ok(())
                            },
                        )?;
                        assert_eq!(live.get(), 0, "retire before publishing the position");
                        Ok(())
                    })
                    .unwrap();
                assert_eq!(cache.visible_positions(), position + 1);
                assert_eq!(live.get(), 0);
            }
            assert!(cache.step(&device).is_err());
            assert!(!cache.is_poisoned());
            cache.reset().unwrap();
        }
        assert_eq!(recordings.get(), 6);
        assert_eq!(source.reads.lock().unwrap().len(), reads);
    }

    #[test]
    fn on_demand_recording_and_execution_errors_or_unwinds_poison_the_old_prefix() {
        let fixture = make_fixture(true, 8);
        let plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        for (during_recording, unwind) in
            [(true, false), (true, true), (false, false), (false, true)]
        {
            let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
            let hidden = device.upload_f32(&vec![0.25; plan.dim()]).unwrap();
            let mut cache = prepared.record_on_demand(&hidden.full_view()).unwrap();
            cache.step(&device).unwrap();
            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                cache.step_with(|recording, position| {
                    let AttentionRecording::OnDemand(bound) = recording else {
                        panic!("expected on-demand attention")
                    };
                    bound.execute_position(
                        position,
                        |commands| {
                            let batch = device.record(commands)?;
                            if during_recording {
                                if unwind {
                                    panic!("injected recording unwind");
                                }
                                bail!("injected recording error");
                            }
                            Ok(batch)
                        },
                        |batch| {
                            let submission = device.submit(batch)?;
                            device.wait(&submission)?;
                            if unwind {
                                panic!("injected post-wait unwind");
                            }
                            bail!("injected post-wait error")
                        },
                    )
                })
            }));
            if unwind {
                assert!(outcome.is_err());
            } else {
                assert!(outcome.unwrap().is_err());
            }
            assert_eq!(cache.visible_positions(), 1);
            assert!(cache.is_poisoned());
            assert!(cache.reset().is_err());
            assert!(cache.step(&device).is_err());
        }
        let prepared = plan.prepare(&device, &source, 3, limits(&plan, 3)).unwrap();
        let hidden = device.allocate_f32(plan.dim()).unwrap();
        let mut cache = prepared.record_on_demand(&hidden.full_view()).unwrap();
        assert!(
            cache.step(&CpuExecutionDevice::new()).is_err(),
            "recording on a foreign device must fail"
        );
        assert_eq!(cache.visible_positions(), 0);
        assert!(cache.is_poisoned());
        assert!(cache.reset().is_err());
    }

    #[test]
    fn metadata_cost_and_coefficients_cover_full_fixed_capacity() {
        let fixture = make_fixture(true, 8);
        let mut plan = QwenModel::load(&fixture.checkpoint)
            .unwrap()
            .dense_attention_plan(0)
            .unwrap();
        assert_eq!(
            plan.memory(3).unwrap(),
            DenseAttentionMemory {
                packed_weight_bytes: 4 * 1088,
                resident_tensor_bytes: 2176,
                largest_staging_bytes: 1088
            }
        );
        let (cos, sin) = coefficients(3, 32, 10000.0).unwrap();
        assert_eq!(cos.len(), 48);
        assert_eq!(sin.len(), 48);
        assert_eq!(&cos[..16], &[1.0; 16]);
        assert_eq!(&sin[..16], &[0.0; 16]);
        assert_eq!(cos[16], 1.0f64.cos() as f32);
        assert_eq!(sin[32], 2.0f64.sin() as f32);
        plan.dim = 1024;
        plan.n_heads = 16;
        plan.n_kv_heads = 8;
        plan.head_dim = 128;
        assert_eq!(plan.memory(8).unwrap().resident_tensor_bytes, 107520);
        plan.context = 2048;
        for (capacity, tensors, coefficient_bytes) in [
            (512, 4494336, 262144),
            (1024, 8950784, 524288),
            (2048, 17863680, 1048576),
        ] {
            let memory = plan.memory_on_demand(capacity).unwrap();
            assert_eq!(memory.resident_tensor_bytes, tensors);
            assert_eq!(memory.largest_staging_bytes, coefficient_bytes);
            let (cos, sin) = coefficients(capacity, plan.head_dim, plan.theta).unwrap();
            assert_eq!(((cos.len() + sin.len()) * 4) as u64, coefficient_bytes);
            assert_eq!(
                cos[(capacity - 1) * plan.head_dim / 2],
                ((capacity - 1) as f64).cos() as f32
            );
        }
        // Extended capacities are cost queries only; retain the bounded table
        // generation above without materializing the larger coefficient pairs.
        plan.context = MAX_ON_DEMAND_CAPACITY;
        for (capacity, tensors, coefficient_bytes) in [
            (4096, 35_689_472, 2_097_152),
            (8192, 71_341_056, 4_194_304),
            (16384, 142_644_224, 8_388_608),
        ] {
            let memory = plan.memory_on_demand(capacity).unwrap();
            assert_eq!(memory.resident_tensor_bytes, tensors);
            assert_eq!(memory.largest_staging_bytes, coefficient_bytes);
        }
        assert_eq!(
            plan.on_demand_host_metadata_bytes().unwrap(),
            std::mem::size_of::<BoundDenseAttention>() as u64 + 64 + 7 * 64 + 11 * 2 * 8,
        );
    }
}
