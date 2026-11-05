//! Bounded resident gated full attention for one strict `qwen35` block.
//!
//! This is an unqualified graph component, not a serving capability. It streams
//! unchanged packed projection rows and keeps all activations, coefficients and KV
//! on the explicitly supplied execution device. In particular it does not read
//! activations back or substitute CPU arithmetic. `step_checked` requires an
//! explicit backend finite-check capability and validates ten phase boundaries.
//! The older `step` remains an unchecked diagnostic path. Neither path alone
//! establishes native numerical qualification or serving admission.
//!
//! Q/G stays `[head][query, gate]`. RMS spans the entire head, while RoPE touches
//! only its leading rotated prefix. Stored GGUF norm weights already include the
//! converter's transformation and are used unchanged. Text positions share the
//! same position on every IMRoPE section; image/video positions are unsupported.
//!
//! The caller must serialize the complete step, including streamed tiles, with all
//! other accesses to its hidden vector/device. A failed mutating step may have
//! changed hidden and the unpublished KV tail. It poisons this component until an
//! explicit successful reset; the model owner must also reset/abandon the whole
//! token and restore its own hidden vector. Reset cannot repair a poisoned native
//! transport, whose writes continue to fail.

use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, ExecutionError, F32Tensor, F32TensorView, FiniteCheck,
    FiniteCheckSpec, LinearMode, RecordedBatch,
};
use beholdd_backend::{IncrementalAttentionConfig, RopeLayout};
use beholdd_format::{CheckpointId, GgmlType, GgufTensor};
use beholdd_memory::WeightSource;

use super::attention::AttentionShape;
use super::config::{Attention, Binding};
use super::recorded_weights::{PackedRowPlan, StreamStats};
use super::weights::{self, ReadLimits};

const MAX_CAPACITY: usize = 16_384;
const RESET_ELEMENTS: usize = 16_384;

/// Successful finite-validation boundaries per checked attention/residual step.
/// Checker packet initialization is shared separately by the enclosing model.
pub const ATTENTION_FINITE_CHECKS_PER_STEP: usize = 10;

/// Logical payload forecast. Backend programs, private attention/linear scratch,
/// resource/recording metadata, alignment, caller hidden and independently retained
/// source caches are separate costs. The externally shared finite checker and its
/// per-check host scratch are additional costs; query `finite_check_requirements`.
/// CPU oracle backends also own decoded scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionMemory {
    /// One private resident F32 packet, including norms, coefficients and KV.
    pub resident_tensor_bytes: u64,
    /// Largest simultaneously retained packed projection tile on the device.
    pub packed_tile_bytes: u64,
    /// Maximum model-owned host payload: returned packed tile, raw+decoded norm,
    /// both full coefficient staging tables, or the bounded reset zero buffer.
    /// These stages do not overlap. Device upload/conversion copies are separate.
    pub host_staging_bytes: u64,
}

/// Admission ceilings with the same scope as [`AttentionMemory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionLimits {
    /// Maximum private resident F32 packet.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time packed projection tile.
    pub packed_tile_bytes: u64,
    /// Maximum model-owned host staging payload.
    pub host_staging_bytes: u64,
}

/// Additional finite-check costs for one successful checked attention step.
/// Per-step traffic excludes one-time initialization of the checker shared with
/// other model components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AttentionFiniteRequirements {
    /// Largest view checked at one boundary; no internal chunking is hidden.
    pub max_view_elements: usize,
    /// Explicit validation boundaries per successful step.
    pub checks_per_step: u64,
    /// Additional physical dispatches for all validation boundaries.
    pub physical_dispatches_per_step: u64,
    /// Additional logical submissions for all validation boundaries.
    pub logical_submissions_per_step: u64,
    /// Additional pending-status writes, excluding one-time initialization.
    pub pending_write_bytes_per_step: u64,
    /// Additional bounded status readback, never activation readback.
    pub status_read_bytes_per_step: u64,
    /// Shared checker payload/init/host scratch costs; admit once per model.
    pub shared_checker: FiniteCheckSpec,
}

#[derive(Debug, Clone, Copy)]
struct Span {
    start: usize,
    len: usize,
}

impl Span {
    fn view(self, packet: &F32Tensor) -> Result<F32TensorView> {
        Ok(packet.view(self.start, self.len)?)
    }
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    norms: [Span; 3],
    cos: Span,
    sin: Span,
    mutable_start: usize,
    normalized: Span,
    query_gate: Span,
    raw_key: Span,
    query: Span,
    attended: Span,
    gated: Span,
    key_cache: Span,
    value_cache: Span,
    len: usize,
}

impl Layout {
    fn new(dim: usize, shape: AttentionShape) -> Result<Self> {
        let q = shape.query_heads * shape.head_dim;
        let kv = shape.kv_heads * shape.head_dim;
        let table = shape
            .capacity
            .checked_mul(shape.rope_dim / 2)
            .context("resident attention coefficient size overflows")?;
        let cache = shape
            .capacity
            .checked_mul(kv)
            .context("resident attention cache size overflows")?;
        let mut end = 0usize;
        let mut take = |len| -> Result<Span> {
            let start = end;
            end = end
                .checked_add(len)
                .context("resident attention packet size overflows")?;
            Ok(Span { start, len })
        };
        let norms = [take(dim)?, take(shape.head_dim)?, take(shape.head_dim)?];
        let cos = take(table)?;
        let sin = take(table)?;
        let normalized = take(dim)?;
        let mutable_start = normalized.start;
        let query_gate = take(2 * q)?;
        let raw_key = take(kv)?;
        let query = take(q)?;
        let attended = take(q)?;
        let gated = take(q)?;
        let key_cache = take(cache)?;
        let value_cache = take(cache)?;
        Ok(Self {
            norms,
            cos,
            sin,
            mutable_start,
            normalized,
            query_gate,
            raw_key,
            query,
            attended,
            gated,
            key_cache,
            value_cache,
            len: end,
        })
    }
}

/// Metadata-only plan for `hidden += O(sigmoid(G) * attention(Q,K,V))`.
/// It includes pre-attention normalization; FFN/post-attention normalization belong
/// to the enclosing block. No per-position batches or per-row metadata are retained.
#[derive(Debug, Clone)]
pub struct RecordedAttentionPlan {
    checkpoint: CheckpointId,
    dim: usize,
    shape: AttentionShape,
    matrices: [PackedRowPlan; 4],
    kinds: [GgmlType; 4],
    norms: [GgufTensor; 3],
    read_limits: ReadLimits,
    layout: Layout,
    memory: AttentionMemory,
}

impl RecordedAttentionPlan {
    /// Bind only a full-attention block, with explicit cache and read limits.
    /// Native probe geometry is checked here without any source/device effects;
    /// successful planning alone does not establish numerical/backend support.
    pub fn new(
        binding: &Binding,
        layer: usize,
        capacity: usize,
        read_limits: ReadLimits,
    ) -> Result<Self> {
        let config = binding.config();
        let block = binding
            .blocks()
            .get(layer)
            .context("attention layer is outside checkpoint")?;
        let Attention::Full(full) = &block.attention else {
            anyhow::bail!("resident full attention refuses a DeltaNet block");
        };
        ensure!(
            (1..=MAX_CAPACITY).contains(&capacity) && capacity <= config.context,
            "resident attention capacity must be 1..=16384 and within checkpoint context"
        );
        ensure!(
            config.dim <= 5120
                && (1..=32).contains(&config.n_heads)
                && (1..=32).contains(&config.n_kv_heads)
                && config.key_head_dim <= 256
                && config.n_kv_heads * config.key_head_dim <= 4096,
            "resident attention dimensions exceed the bounded native probe profile"
        );
        let shape = AttentionShape {
            query_heads: config.n_heads,
            kv_heads: config.n_kv_heads,
            head_dim: config.key_head_dim,
            rope_dim: config.rope_dim,
            rope_base: config.rope_base,
            epsilon: config.eps,
            capacity,
        };
        // Reuse the CPU shape validation in addition to strict binder invariants.
        shape.cache_bytes()?;
        let matrices = [
            PackedRowPlan::new(&full.query_gate, read_limits)?,
            PackedRowPlan::new(&full.key, read_limits)?,
            PackedRowPlan::new(&full.value, read_limits)?,
            PackedRowPlan::new(&full.output, read_limits)?,
        ];
        let q = shape.query_heads * shape.head_dim;
        let kv = shape.kv_heads * shape.head_dim;
        for (matrix, rows, cols) in [
            (&matrices[0], 2 * q, config.dim),
            (&matrices[1], kv, config.dim),
            (&matrices[2], kv, config.dim),
            (&matrices[3], config.dim, q),
        ] {
            ensure!(
                matrix.rows() == rows && matrix.cols() == cols,
                "resident projection shape differs"
            );
            ensure!(
                rows.checked_mul(4).context("projection bytes overflow")?
                    <= read_limits.decoded_bytes,
                "resident projection output exceeds the configured decoded/result limit"
            );
        }
        let norms = [
            block.attention_norm.clone(),
            full.query_norm.clone(),
            full.key_norm.clone(),
        ];
        for (norm, len) in norms
            .iter()
            .zip([config.dim, shape.head_dim, shape.head_dim])
        {
            let bytes = len.checked_mul(4).context("norm bytes overflow")?;
            ensure!(
                norm.kind().0 == 0 && norm.dims() == [len as u64] && norm.nbytes() == bytes as u64,
                "resident attention requires exact F32 normalization vectors"
            );
            ensure!(
                bytes <= read_limits.encoded_bytes && bytes <= read_limits.decoded_bytes,
                "resident attention norm exceeds the configured read/scratch limits"
            );
        }
        let layout = Layout::new(config.dim, shape)?;
        let packed = matrices
            .iter()
            .map(PackedRowPlan::peak_read_bytes)
            .max()
            .unwrap();
        let norm_staging = norms
            .iter()
            .map(GgufTensor::nbytes)
            .max()
            .unwrap()
            .checked_mul(2)
            .context("normalization staging bytes overflow")?;
        let coefficients = (layout.cos.len as u64)
            .checked_mul(8)
            .context("coefficient staging bytes overflow")?;
        let memory = AttentionMemory {
            resident_tensor_bytes: (layout.len as u64)
                .checked_mul(4)
                .context("resident bytes overflow")?,
            packed_tile_bytes: packed as u64,
            host_staging_bytes: (packed as u64)
                .max(norm_staging)
                .max(coefficients)
                .max((RESET_ELEMENTS * 4) as u64),
        };
        Ok(Self {
            checkpoint: binding.checkpoint_id(),
            dim: config.dim,
            shape,
            matrices,
            kinds: [
                full.query_gate.kind(),
                full.key.kind(),
                full.value.kind(),
                full.output.kind(),
            ],
            norms,
            read_limits,
            layout,
            memory,
        })
    }

    /// Caller-owned hidden width; its allocation is not in [`Self::memory`].
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Fixed full-attention geometry and evaluated-position limit.
    pub fn shape(&self) -> AttentionShape {
        self.shape
    }
    /// Exact retained logical payload and bounded host staging forecast.
    pub fn memory(&self) -> AttentionMemory {
        self.memory
    }
    /// Maximum simultaneously retained non-linear semantic commands: one pre-RMS
    /// plus per-head Q/K RMS+RoPE, attention, and per-query-head sigmoid gating.
    /// A streamed tile adds one linear command while these two batches are held.
    pub fn retained_commands(&self) -> usize {
        2 + 3 * self.shape.query_heads + 2 * self.shape.kv_heads
    }

    /// Required finite-check capability and its separate payload/traffic costs.
    /// This metadata-only admission precedes parameter reads or packet mutation.
    /// Every named phase fits one check; no hidden unbounded chunking is performed.
    pub fn finite_check_requirements(
        &self,
        spec: Option<FiniteCheckSpec>,
    ) -> Result<AttentionFiniteRequirements> {
        let spec = spec.context("checked attention requires explicit finite validation")?;
        let largest = [
            self.dim,
            self.layout.query_gate.len,
            self.layout.raw_key.len,
            self.layout.query.len,
            self.layout.attended.len,
            self.layout.gated.len,
        ]
        .into_iter()
        .max()
        .unwrap();
        ensure!(
            largest > 0 && largest <= spec.max_elements,
            "checked attention phase exceeds finite-check capability"
        );
        let checks_per_step = ATTENTION_FINITE_CHECKS_PER_STEP as u64;
        let traffic = |per_check: u64| {
            per_check
                .checked_mul(checks_per_step)
                .context("checked attention finite traffic overflows")
        };
        Ok(AttentionFiniteRequirements {
            max_view_elements: largest,
            checks_per_step,
            physical_dispatches_per_step: traffic(spec.physical_dispatches)?,
            logical_submissions_per_step: traffic(spec.logical_submissions)?,
            pending_write_bytes_per_step: traffic(spec.pending_write_bytes)?,
            status_read_bytes_per_step: traffic(spec.status_read_bytes)?,
            shared_checker: spec,
        })
    }

    /// Prepare a checked-path component only when every finite boundary fits the
    /// backend's explicit capability. The caller admits/prepares one shared
    /// checker separately and passes it to `step_checked`; its storage is not
    /// charged once per layer. Checker ownership is checked before arithmetic.
    pub fn prepare_checked(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: AttentionLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentAttention> {
        self.finite_check_requirements(device.finite_check_spec())?;
        self.prepare(device, source, limits, control)
    }

    fn validate_source(&self, source: &dyn WeightSource) -> Result<()> {
        ensure!(
            source.checkpoint_id() == self.checkpoint,
            "resident attention source belongs to a different checkpoint"
        );
        Ok(())
    }

    /// Allocate one private packet and initialize only immutable norms/coefficient
    /// tables. No checkpoint projection matrix is loaded until `step`.
    /// All forecast/identity refusals precede reads and device effects. Small norm
    /// reads are sequential; both coefficient vectors are counted while uploading.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: AttentionLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentAttention> {
        self.validate_source(source)?;
        for (name, required, admitted) in [
            (
                "resident tensors",
                self.memory.resident_tensor_bytes,
                limits.resident_tensor_bytes,
            ),
            (
                "packed tile",
                self.memory.packed_tile_bytes,
                limits.packed_tile_bytes,
            ),
            (
                "host staging",
                self.memory.host_staging_bytes,
                limits.host_staging_bytes,
            ),
        ] {
            ensure!(
                required <= admitted,
                "resident attention {name} needs {required} bytes, limit is {admitted}"
            );
        }
        for kind in self.kinds {
            ensure!(
                device.supports(kind),
                "backend refuses resident attention projection format"
            );
        }
        control()?;
        let packet = device.allocate_f32(self.layout.len)?;
        for (tensor, span) in self.norms.iter().zip(self.layout.norms) {
            control()?;
            let values = weights::f32_tensor(tensor, source, self.read_limits)?;
            control()?;
            device.write_f32(&span.view(&packet)?, &values)?;
            drop(values);
        }
        let (cos, sin) = coefficients(self.shape, control)?;
        control()?;
        device.write_f32(&self.layout.cos.view(&packet)?, &cos)?;
        control()?;
        device.write_f32(&self.layout.sin.view(&packet)?, &sin)?;
        drop((cos, sin));
        control()?;
        Ok(ResidentAttention {
            plan: self.clone(),
            packet,
            positions: 0,
            poisoned: false,
            reset_write_bytes: 0,
        })
    }
}

fn coefficients(
    shape: AttentionShape,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let len = shape
        .capacity
        .checked_mul(shape.rope_dim / 2)
        .context("coefficient count overflows")?;
    let mut cos = Vec::new();
    let mut sin = Vec::new();
    cos.try_reserve_exact(len)
        .context("allocating cosine staging")?;
    sin.try_reserve_exact(len)
        .context("allocating sine staging")?;
    for position in 0..shape.capacity {
        control()?;
        for i in 0..shape.rope_dim / 2 {
            // Deliberately identical operation ordering to attention::AttentionState.
            let theta = position as f64
                * shape
                    .rope_base
                    .powf(-2.0 * i as f64 / shape.rope_dim as f64);
            let (c, s) = (theta.cos() as f32, theta.sin() as f32);
            ensure!(
                c.is_finite() && s.is_finite(),
                "nonfinite resident RoPE coefficient"
            );
            cos.push(c);
            sin.push(s);
        }
    }
    Ok((cos, sin))
}

/// Sequential owner of one layer's private packet and committed KV prefix.
/// No cache/state/scratch view escapes this type. Step recording is on demand and
/// bounded by the head count, independent of the admitted context capacity.
#[derive(Debug)]
pub struct ResidentAttention {
    plan: RecordedAttentionPlan,
    packet: F32Tensor,
    positions: usize,
    poisoned: bool,
    reset_write_bytes: u64,
}

impl ResidentAttention {
    /// Successfully completed attention/residual positions.
    pub fn positions(&self) -> usize {
        self.positions
    }
    /// Fixed cache capacity.
    pub fn capacity(&self) -> usize {
        self.plan.shape.capacity
    }
    /// Any failure after mutation starts keeps this set until a complete reset.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Successfully written reset bytes, including a partial failed reset.
    pub fn reset_write_bytes(&self) -> u64 {
        self.reset_write_bytes
    }

    fn normalization_commands(&self, hidden: &F32TensorView) -> Result<CommandList> {
        let l = self.plan.layout;
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &l.normalized.view(&self.packet)?,
            hidden,
            Some(&l.norms[0].view(&self.packet)?),
            self.plan.shape.epsilon,
        )?;
        Ok(commands)
    }

    fn append_qk_commands(&self, commands: &mut CommandList) -> Result<()> {
        let s = self.plan.shape;
        let l = self.plan.layout;
        let cos = self.packet.view(
            l.cos.start + self.positions * (s.rope_dim / 2),
            s.rope_dim / 2,
        )?;
        let sin = self.packet.view(
            l.sin.start + self.positions * (s.rope_dim / 2),
            s.rope_dim / 2,
        )?;
        let query_norm = l.norms[1].view(&self.packet)?;
        let key_norm = l.norms[2].view(&self.packet)?;
        for head in 0..s.query_heads {
            let output = self
                .packet
                .view(l.query.start + head * s.head_dim, s.head_dim)?;
            let input = self
                .packet
                .view(l.query_gate.start + 2 * head * s.head_dim, s.head_dim)?;
            commands.rmsnorm(&output, &input, Some(&query_norm), s.epsilon)?;
            commands.rope(
                &output.view(0, s.rope_dim)?,
                &cos,
                &sin,
                RopeLayout::SplitHalf,
                false,
            )?;
        }
        let kv_width = s.kv_heads * s.head_dim;
        for head in 0..s.kv_heads {
            let output = self.packet.view(
                l.key_cache.start + self.positions * kv_width + head * s.head_dim,
                s.head_dim,
            )?;
            let input = self
                .packet
                .view(l.raw_key.start + head * s.head_dim, s.head_dim)?;
            commands.rmsnorm(&output, &input, Some(&key_norm), s.epsilon)?;
            commands.rope(
                &output.view(0, s.rope_dim)?,
                &cos,
                &sin,
                RopeLayout::SplitHalf,
                false,
            )?;
        }
        Ok(())
    }

    fn append_attended_commands(&self, commands: &mut CommandList) -> Result<()> {
        let s = self.plan.shape;
        let l = self.plan.layout;
        let kv_width = s.kv_heads * s.head_dim;
        commands.incremental_causal_attention(
            &l.attended.view(&self.packet)?,
            &l.query.view(&self.packet)?,
            &self
                .packet
                .view(l.key_cache.start, (self.positions + 1) * kv_width)?,
            &self
                .packet
                .view(l.value_cache.start, (self.positions + 1) * kv_width)?,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: self.positions,
                n_kv_positions: self.positions + 1,
                n_heads: s.query_heads,
                n_kv_heads: s.kv_heads,
                head_dim: s.head_dim,
                scale: 1.0 / (s.head_dim as f32).sqrt(),
            },
        )?;
        Ok(())
    }

    fn append_gating_commands(&self, commands: &mut CommandList) -> Result<()> {
        let s = self.plan.shape;
        let l = self.plan.layout;
        for head in 0..s.query_heads {
            commands.sigmoid_gate(
                &self
                    .packet
                    .view(l.gated.start + head * s.head_dim, s.head_dim)?,
                &self
                    .packet
                    .view(l.attended.start + head * s.head_dim, s.head_dim)?,
                &self
                    .packet
                    .view(l.query_gate.start + (2 * head + 1) * s.head_dim, s.head_dim)?,
            )?;
        }
        Ok(())
    }

    fn attention_commands(&self) -> Result<CommandList> {
        let mut commands = CommandList::new();
        self.append_qk_commands(&mut commands)?;
        self.append_attended_commands(&mut commands)?;
        self.append_gating_commands(&mut commands)?;
        debug_assert_eq!(commands.len() + 1, self.plan.retained_commands());
        Ok(commands)
    }

    fn record_phase(
        &self,
        device: &dyn ExecutionDevice,
        append: fn(&Self, &mut CommandList) -> Result<()>,
    ) -> Result<RecordedBatch> {
        let mut commands = CommandList::new();
        append(self, &mut commands)?;
        Ok(device.record(&commands)?)
    }

    /// Unchecked diagnostic path: stream four projections and publish one position
    /// only after the complete output projection/residual finishes. `StreamStats` accumulates
    /// actual matrix reads/uploads/tiles even on failure; preparation norm reads and
    /// the two non-linear batches are outside its matrix-only scope.
    ///
    /// Identity, capacity, hidden extent/aliasing, and non-linear recording support
    /// are checked before mutation. Every streamed tile performs its own explicit
    /// wait and drops all weight handles before reading the next tile.
    pub fn step(
        &mut self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        hidden: &F32TensorView,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "resident attention is poisoned; reset or replace it"
        );
        ensure!(
            self.positions < self.capacity(),
            "resident attention cache is full"
        );
        self.plan.validate_source(source)?;
        ensure!(
            hidden.len() == self.plan.dim && !hidden.overlaps(&self.packet.full_view()),
            "resident attention hidden shape or private-packet alias differs"
        );
        control()?;
        let normalization = device.record(&self.normalization_commands(hidden)?)?;
        let attention = device.record(&self.attention_commands()?)?;
        control()?;
        // Record/identity refusals above cannot modify state. Any later error or
        // unwind is conservative: hidden or the unpublished cache tail may differ.
        self.poisoned = true;
        execute(device, normalization, control)?;
        let l = self.plan.layout;
        let normalized = l.normalized.view(&self.packet)?;
        let kv_width = self.plan.shape.kv_heads * self.plan.shape.head_dim;
        let destinations = [
            l.query_gate.view(&self.packet)?,
            l.raw_key.view(&self.packet)?,
            self.packet
                .view(l.value_cache.start + self.positions * kv_width, kv_width)?,
        ];
        for (matrix, output) in self.plan.matrices[..3].iter().zip(destinations) {
            matrix.execute_rows(
                device,
                source,
                0..matrix.rows(),
                &normalized,
                &output,
                LinearMode::Overwrite,
                stats,
                control,
            )?;
        }
        execute(device, attention, control)?;
        let output = &self.plan.matrices[3];
        output.execute_rows(
            device,
            source,
            0..output.rows(),
            &l.gated.view(&self.packet)?,
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
            stats,
            control,
        )?;
        control()?;
        self.positions += 1;
        self.poisoned = false;
        Ok(())
    }

    /// Evaluate the gated attention/residual with ten explicit finite boundaries.
    /// The externally shared checker must belong to `device`. Entry validation
    /// precedes arithmetic; every refusal/cancellation after arithmetic is armed
    /// keeps this component poisoned and leaves the new KV position unpublished.
    ///
    /// Host-only command construction checks hidden/private packet ownership, then
    /// entry validation checks the actual device/checker owners before recording
    /// can allocate or initialize native auxiliary storage. The four arithmetic
    /// batches subsequently validate operation support before activation mutation.
    /// They contain the same semantic commands as the diagnostic path. At most
    /// five batches coexist, including one temporary finite-check or packed tile
    /// batch. The caller admits checker payload/scratch separately and serializes
    /// every check with immediate consumption of the checked views.
    pub fn step_checked(
        &mut self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        hidden: &F32TensorView,
        checker: &mut FiniteCheck,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "resident attention is poisoned; reset or replace it"
        );
        ensure!(
            self.positions < self.capacity(),
            "resident attention cache is full"
        );
        self.plan.validate_source(source)?;
        ensure!(
            hidden.len() == self.plan.dim && !hidden.overlaps(&self.packet.full_view()),
            "resident attention hidden shape or private-packet alias differs"
        );
        self.plan
            .finite_check_requirements(device.finite_check_spec())?;
        let normalization_commands = self.normalization_commands(hidden)?;
        finite(device, checker, hidden, "entry hidden", control)?; // 1
        control()?;
        let normalization = device.record(&normalization_commands)?;
        drop(normalization_commands);
        let qk = self.record_phase(device, Self::append_qk_commands)?;
        let attention = self.record_phase(device, Self::append_attended_commands)?;
        let gating = self.record_phase(device, Self::append_gating_commands)?;
        control()?;

        // Entry/pre-recording rejection changes no component activations. Native
        // entry validation may update checker status; device uncertainty remains
        // permanent even before this component is armed. Every error after arming
        // (including a finite checker refusal) requires reset.
        self.poisoned = true;
        execute(device, normalization, control)?;
        let l = self.plan.layout;
        let normalized = l.normalized.view(&self.packet)?;
        finite(device, checker, &normalized, "normalized hidden", control)?; // 2
        let kv_width = self.plan.shape.kv_heads * self.plan.shape.head_dim;
        let raw_qg = l.query_gate.view(&self.packet)?;
        let raw_k = l.raw_key.view(&self.packet)?;
        let new_v = self
            .packet
            .view(l.value_cache.start + self.positions * kv_width, kv_width)?;
        for (matrix, destination) in self.plan.matrices[..3]
            .iter()
            .zip([&raw_qg, &raw_k, &new_v])
        {
            matrix.execute_rows(
                device,
                source,
                0..matrix.rows(),
                &normalized,
                destination,
                LinearMode::Overwrite,
                stats,
                control,
            )?;
        }
        // Validate raw projections BEFORE RMS or sigmoid can mask infinities.
        finite(device, checker, &raw_qg, "raw query/gate", control)?; // 3
        finite(device, checker, &raw_k, "raw key", control)?; // 4
        finite(device, checker, &new_v, "new value cache", control)?; // 5
        execute(device, qk, control)?;
        finite(
            device,
            checker,
            &l.query.view(&self.packet)?,
            "rotated query",
            control,
        )?; // 6
        let new_k = self
            .packet
            .view(l.key_cache.start + self.positions * kv_width, kv_width)?;
        finite(device, checker, &new_k, "rotated key cache", control)?; // 7
        execute(device, attention, control)?;
        // Checking attended values is a conservative extra phase boundary; the
        // CPU reference does not claim this exact placement is mathematically required.
        finite(
            device,
            checker,
            &l.attended.view(&self.packet)?,
            "attended values",
            control,
        )?; // 8
        execute(device, gating, control)?;
        let gated = l.gated.view(&self.packet)?;
        finite(device, checker, &gated, "gated attention", control)?; // 9
        let output = &self.plan.matrices[3];
        output.execute_rows(
            device,
            source,
            0..output.rows(),
            &gated,
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
            stats,
            control,
        )?;
        finite(device, checker, hidden, "residual hidden", control)?; // 10
        control()?;
        self.positions += 1;
        self.poisoned = false;
        Ok(())
    }

    /// Explicitly clear every mutable activation and both complete caches in
    /// <=64 KiB writes. Only complete success resets positions and poison state.
    /// Norms/coefficient tables remain intact. This does not restore caller hidden,
    /// reset other layers, or recover an uncertain native transport.
    pub fn reset(
        &mut self,
        device: &dyn ExecutionDevice,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.poisoned = true;
        let l = self.plan.layout;
        let mut zeroes = Vec::new();
        let chunk = (l.len - l.mutable_start).min(RESET_ELEMENTS);
        zeroes
            .try_reserve_exact(chunk)
            .context("allocating bounded attention reset buffer")?;
        zeroes.resize(chunk, 0.0);
        for start in (l.mutable_start..l.len).step_by(RESET_ELEMENTS) {
            control()?;
            let len = (l.len - start).min(RESET_ELEMENTS);
            let next = self
                .reset_write_bytes
                .checked_add((len * 4) as u64)
                .context("reset byte counter overflows")?;
            device.write_f32(&self.packet.view(start, len)?, &zeroes[..len])?;
            self.reset_write_bytes = next;
        }
        control()?;
        self.positions = 0;
        self.poisoned = false;
        Ok(())
    }
}

fn finite(
    device: &dyn ExecutionDevice,
    checker: &mut FiniteCheck,
    view: &F32TensorView,
    phase: &'static str,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    device
        .check_finite(checker, view, &mut || {
            control().map_err(|error| {
                ExecutionError::Shape(format!("checked attention control interrupted: {error:#}"))
            })
        })
        .with_context(|| format!("checked attention {phase}"))
}

fn execute(
    device: &dyn ExecutionDevice,
    batch: RecordedBatch,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    control()?;
    let submission = device.submit(&batch)?;
    device.wait(&submission)?;
    drop(submission);
    drop(batch);
    control()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::attention::AttentionState;
    use crate::qwen35::config::tests::{Fixture, Temporary};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::CpuExecutionDevice;
    use beholdd_backend::Device;
    use beholdd_format::{Gguf, WeightRange};
    use beholdd_memory::{ResidentWeight, WeightAccess};
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    struct Source<'a> {
        gguf: &'a Gguf,
        reads: AtomicU64,
    }
    impl<'a> Source<'a> {
        fn new(gguf: &'a Gguf) -> Self {
            Self {
                gguf,
                reads: AtomicU64::new(0),
            }
        }
    }
    impl WeightSource for Source<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.gguf.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, _access: WeightAccess) -> Result<ResidentWeight> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            ResidentWeight::load_range(self.gguf, range)
        }
    }

    fn read_limits() -> ReadLimits {
        ReadLimits {
            encoded_bytes: 32 * 1024,
            decoded_bytes: 64 * 1024,
            tile_rows: 32,
        }
    }
    fn admitted(plan: &RecordedAttentionPlan) -> AttentionLimits {
        let m = plan.memory();
        AttentionLimits {
            resident_tensor_bytes: m.resident_tensor_bytes,
            packed_tile_bytes: m.packed_tile_bytes,
            host_staging_bytes: m.host_staging_bytes,
        }
    }

    // Real locators and strict hybrid binding, with nonzero deterministic F32
    // norms and independent constant-row packed fixtures. The reference below
    // uses Device + AttentionState, not this recorded command construction.
    fn fixture() -> Result<(Temporary, Gguf)> {
        let (temporary, gguf) = Fixture::hybrid().open();
        let mut file = OpenOptions::new().write(true).open(&gguf.shards()[0])?;
        for (name, period, scale) in [
            ("attn_norm.weight", 9, 0.01f32),
            ("attn_q_norm.weight", 7, 0.03),
            ("attn_k_norm.weight", 5, 0.02),
        ] {
            let t = gguf.get(&format!("blk.3.{name}")).unwrap();
            let mut bytes = Vec::new();
            for i in 0..t.dims()[0] as usize {
                bytes.extend((1.0 + (i % period) as f32 * scale).to_le_bytes());
            }
            file.seek(SeekFrom::Start(t.offset()))?;
            file.write_all(&bytes)?;
        }
        for name in [
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
        ] {
            let t = gguf.get(&format!("blk.3.{name}")).unwrap();
            let mut bytes = Vec::new();
            for row in 0..t.dims()[1] as usize {
                let value = if name == "attn_q.weight" && row % 512 >= 256 {
                    if row < 512 {
                        1
                    } else {
                        7
                    }
                } else {
                    (row % 13 + 1) as u8
                };
                for _ in 0..t.dims()[0] / 256 {
                    let mut block = vec![0u8; if t.kind().0 == 12 { 144 } else { 210 }];
                    if t.kind().0 == 12 {
                        block[..2].copy_from_slice(&0x1400u16.to_le_bytes()); // 2^-10
                        block[4..8].fill(1);
                        block[12..16].fill(1);
                        block[16..].fill(value | (value << 4));
                    } else {
                        block[..128].fill(value | (value << 4));
                        block[128..192].fill(0xaa); // unsigned code 32+value
                        block[192..208].fill(1);
                        block[208..].copy_from_slice(&0x1400u16.to_le_bytes());
                    }
                    bytes.extend(block);
                }
            }
            assert_eq!(bytes.len() as u64, t.nbytes());
            file.seek(SeekFrom::Start(t.offset()))?;
            file.write_all(&bytes)?;
        }
        Ok((temporary, gguf))
    }

    fn reference(
        binding: &Binding,
        source: &Source<'_>,
        state: &mut AttentionState,
        input: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let device = Cpu;
        let block = &binding.blocks()[3];
        let Attention::Full(full) = &block.attention else {
            unreachable!()
        };
        let norm = weights::f32_tensor(&block.attention_norm, source, read_limits())?;
        let mut normalized = vec![0.; input.len()];
        device.rmsnorm(&mut normalized, input, Some(&norm), binding.config().eps)?;
        let mat = |tensor| {
            weights::matvec(
                tensor,
                source,
                &device,
                &normalized,
                read_limits(),
                &mut || Ok(()),
            )
        };
        let qg = mat(&full.query_gate)?;
        let k = mat(&full.key)?;
        let v = mat(&full.value)?;
        let qn = weights::f32_tensor(&full.query_norm, source, read_limits())?;
        let kn = weights::f32_tensor(&full.key_norm, source, read_limits())?;
        let gated = state.step(&device, &qg, &k, &v, &qn, &kn)?;
        let projected = weights::matvec(
            &full.output,
            source,
            &device,
            &gated,
            read_limits(),
            &mut || Ok(()),
        )?;
        let result = input.iter().zip(projected).map(|(a, b)| a + b).collect();
        Ok((result, qg, gated))
    }

    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.is_finite() && (a - b).abs() <= 2e-5 * b.abs().max(1.0),
                "coordinate {i}: {a} != {b}"
            );
        }
    }

    fn read(device: &dyn ExecutionDevice, view: &F32TensorView) -> Result<Vec<f32>> {
        let mut values = vec![0.; view.len()];
        device.read_f32(view, &mut values)?;
        Ok(values)
    }

    #[test]
    fn checked_capability_requires_every_phase_before_preparation() -> Result<()> {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let requirements = plan.finite_check_requirements(device.finite_check_spec())?;
        let spec = requirements.shared_checker;
        assert_eq!(ATTENTION_FINITE_CHECKS_PER_STEP, 10);
        assert_eq!(requirements.checks_per_step, 10);
        assert_eq!(requirements.max_view_elements, plan.layout.query_gate.len);
        assert_eq!(requirements.physical_dispatches_per_step, 0);
        assert_eq!(spec, device.finite_check_spec().unwrap());
        assert!(plan.finite_check_requirements(None).is_err());
        assert!(plan
            .finite_check_requirements(Some(FiniteCheckSpec {
                max_elements: plan.layout.query_gate.len - 1,
                ..spec
            }))
            .is_err());
        assert!(plan
            .finite_check_requirements(Some(FiniteCheckSpec {
                max_elements: plan.layout.query_gate.len,
                ..spec
            }))
            .is_ok());
        assert!(plan
            .finite_check_requirements(Some(FiniteCheckSpec {
                pending_write_bytes: u64::MAX,
                ..spec
            }))
            .is_err());
        let native = plan.finite_check_requirements(Some(FiniteCheckSpec {
            max_elements: 65_536,
            resident_tensor_bytes: 276,
            initialization_write_bytes: 548,
            host_scratch_bytes: 512,
            physical_dispatches: 1,
            logical_submissions: 1,
            pending_write_bytes: 256,
            status_read_bytes: 256,
        }))?;
        assert_eq!(
            (
                native.physical_dispatches_per_step,
                native.logical_submissions_per_step
            ),
            (10, 10)
        );
        assert_eq!(
            (
                native.pending_write_bytes_per_step,
                native.status_read_bytes_per_step
            ),
            (2560, 2560)
        );
        assert_eq!(
            (
                native.shared_checker.resident_tensor_bytes,
                native.shared_checker.initialization_write_bytes
            ),
            (276, 548)
        );
        assert_eq!(source.reads.load(Ordering::Relaxed), 0);
        let _resident = plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        assert_eq!(source.reads.load(Ordering::Relaxed), 3);
        Ok(())
    }

    #[test]
    fn checked_two_positions_match_reference_and_reset_replay_exactly() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[-123.; 258])?;
        let view = hidden.view(1, 256)?;
        let mut oracle = AttentionState::new(plan.shape(), plan.shape().cache_bytes()?)?;
        let mut stats = StreamStats::default();
        let tiles: usize = plan
            .matrices
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        let mut first_input = Vec::new();
        let mut first_output = Vec::new();
        for position in 0..2 {
            let input: Vec<_> = (0..256)
                .map(|i| (i % 11) as f32 * 0.1 + if position == 0 { 0.25 } else { -0.75 })
                .collect();
            let (expected, _, _) = reference(&binding, &source, &mut oracle, &input)?;
            device.write_f32(&view, &input)?;
            let mut controls = 0;
            resident.step_checked(
                &device,
                &source,
                &view,
                &mut checker,
                &mut stats,
                &mut || {
                    controls += 1;
                    Ok(())
                },
            )?;
            // Two preflight, ten CPU checks*3, four arithmetic phases*2,
            // four boundaries per tile, and one final publication control.
            assert_eq!(controls, 41 + 4 * tiles);
            let actual = read(&device, &view)?;
            close(&actual, &expected);
            assert_eq!(resident.positions(), position + 1);
            assert!(!resident.is_poisoned());
            if position == 0 {
                first_input = input;
                first_output = actual;
            }
        }
        let before = stats;
        assert!(resident
            .step_checked(
                &device,
                &source,
                &view,
                &mut checker,
                &mut stats,
                &mut || panic!("capacity refusal must precede control")
            )
            .is_err());
        assert_eq!(stats, before);
        assert!(!resident.is_poisoned());
        resident.reset(&device, &mut || Ok(()))?;
        device.write_f32(&view, &first_input)?;
        resident.step_checked(
            &device,
            &source,
            &view,
            &mut checker,
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(
            read(&device, &view)?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            first_output.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let guarded = read(&device, &hidden.full_view())?;
        assert_eq!((guarded[0], guarded[257]), (-123., -123.));
        Ok(())
    }

    #[test]
    fn checked_raw_projection_faults_refuse_before_qk_or_sigmoid_masking() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let prior_tiles: usize = plan.matrices[..3]
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        for (offset, value, phase) in [
            (
                plan.layout.query_gate.start + plan.shape.head_dim,
                f32::INFINITY,
                "raw query/gate",
            ),
            (plan.layout.raw_key.start, f32::NAN, "raw key"),
            (
                plan.layout.value_cache.start,
                f32::NEG_INFINITY,
                "new value cache",
            ),
        ] {
            let source = Source::new(&gguf);
            let device = CpuExecutionDevice::new();
            let mut checker = device.prepare_finite_check()?;
            let mut resident =
                plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
            let hidden = device.upload_f32(&[1.; 256])?;
            let fault = resident.packet.view(offset, 1)?;
            let mut stats = StreamStats::default();
            let mut controls = 0;
            let error = resident
                .step_checked(
                    &device,
                    &source,
                    &hidden.full_view(),
                    &mut checker,
                    &mut stats,
                    &mut || {
                        controls += 1;
                        // Explicit diagnostic corruption after the last raw projection
                        // fence and before any of the three raw finite boundaries.
                        if controls == 10 + 4 * prior_tiles {
                            device.write_f32(&fault, &[value])?;
                        }
                        Ok(())
                    },
                )
                .unwrap_err();
            assert!(format!("{error:#}").contains(phase), "{error:#}");
            assert_eq!(stats.completed_tiles, prior_tiles as u64);
            assert_eq!(source.reads.load(Ordering::Relaxed), 3 + prior_tiles as u64);
            assert_eq!(resident.positions(), 0);
            assert!(resident.is_poisoned());
            assert!(read(&device, &plan.layout.query.view(&resident.packet)?)?
                .iter()
                .all(|v| v.to_bits() == 0));
            assert!(
                read(&device, &plan.layout.key_cache.view(&resident.packet)?)?
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
            assert!(read(&device, &plan.layout.gated.view(&resident.packet)?)?
                .iter()
                .all(|v| v.to_bits() == 0));
            close(&read(&device, &hidden.full_view())?, &[1.; 256]);
            let before = stats;
            assert!(resident
                .step_checked(
                    &device,
                    &source,
                    &hidden.full_view(),
                    &mut checker,
                    &mut stats,
                    &mut || panic!("poisoned component must refuse")
                )
                .is_err());
            assert_eq!(stats, before);
            resident.reset(&device, &mut || Ok(()))?;
            device.write_f32(&hidden.full_view(), &[1.; 256])?;
            resident.step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || Ok(()),
            )?;
            assert_eq!(resident.positions(), 1);
            assert!(!resident.is_poisoned());
        }
        Ok(())
    }

    #[test]
    fn checked_late_cancellation_never_publishes_the_written_cache_position() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[1.; 256])?;
        let mut stats = StreamStats::default();
        resident.step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut || Ok(()),
        )?;
        let expected: Vec<_> = read(&device, &hidden.full_view())?
            .iter()
            .map(|v| v.to_bits())
            .collect();
        resident.reset(&device, &mut || Ok(()))?;
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        let tiles: usize = plan
            .matrices
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        let mut controls = 0;
        let before_tiles = stats.completed_tiles;
        let error = resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || {
                    controls += 1;
                    ensure!(
                        controls != 41 + 4 * tiles,
                        "cancel after residual validation"
                    );
                    Ok(())
                },
            )
            .unwrap_err();
        assert!(format!("{error:#}").contains("cancel after residual validation"));
        assert_eq!(stats.completed_tiles - before_tiles, tiles as u64);
        assert_eq!(resident.positions(), 0);
        assert!(resident.is_poisoned());
        assert_eq!(
            read(&device, &hidden.full_view())?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        assert!(
            read(&device, &plan.layout.key_cache.view(&resident.packet)?)?
                .iter()
                .any(|v| v.to_bits() != 0)
        );
        resident.reset(&device, &mut || Ok(()))?;
        assert!(
            read(&device, &plan.layout.key_cache.view(&resident.packet)?)?
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        resident.step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(resident.positions(), 1);
        assert!(!resident.is_poisoned());
        assert_eq!(
            read(&device, &hidden.full_view())?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            expected
        );
        Ok(())
    }

    #[test]
    fn checked_foreign_owners_refuse_before_controls_and_bad_entry_preserves_state() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let other = CpuExecutionDevice::new();
        let mut foreign = other.prepare_finite_check()?;
        let mut checker = device.prepare_finite_check()?;
        let mut resident =
            plan.prepare_checked(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[1.; 256])?;
        let foreign_hidden = other.upload_f32(&[1.; 256])?;
        let before = read(&device, &resident.packet.full_view())?;
        let before_reads = source.reads.load(Ordering::Relaxed);
        let mut stats = StreamStats::default();
        let mut controls = 0;
        assert!(resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut foreign,
                &mut stats,
                &mut || {
                    controls += 1;
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(controls, 0);
        assert!(resident
            .step_checked(
                &other,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || {
                    controls += 1;
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(controls, 0);
        // Matching checker/device still cannot admit a foreign input, and matching
        // input/checker cannot bypass the component packet's different owner.
        assert!(resident
            .step_checked(
                &other,
                &source,
                &hidden.full_view(),
                &mut foreign,
                &mut stats,
                &mut || {
                    controls += 1;
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(controls, 0);
        assert!(resident
            .step_checked(
                &other,
                &source,
                &foreign_hidden.full_view(),
                &mut foreign,
                &mut stats,
                &mut || {
                    controls += 1;
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(controls, 0);
        assert_eq!(read(&device, &resident.packet.full_view())?, before);
        assert_eq!(read(&device, &hidden.full_view())?, vec![1.; 256]);
        assert_eq!(read(&other, &foreign_hidden.full_view())?, vec![1.; 256]);
        assert_eq!(source.reads.load(Ordering::Relaxed), before_reads);
        assert_eq!(stats, StreamStats::default());
        assert_eq!(resident.positions(), 0);
        assert!(!resident.is_poisoned());
        device.write_f32(&hidden.view(0, 1)?, &[f32::INFINITY])?;
        let error = resident
            .step_checked(
                &device,
                &source,
                &hidden.full_view(),
                &mut checker,
                &mut stats,
                &mut || {
                    controls += 1;
                    Ok(())
                },
            )
            .unwrap_err();
        // Only the CPU entry check's pre/post-snapshot controls ran. No arithmetic
        // recording/preflight callback, projection read or component write occurred.
        assert_eq!(controls, 2);
        assert!(format!("{error:#}").contains("entry hidden"));
        assert_eq!(read(&device, &resident.packet.full_view())?, before);
        assert_eq!(source.reads.load(Ordering::Relaxed), before_reads);
        assert_eq!(stats, StreamStats::default());
        assert_eq!(resident.positions(), 0);
        assert!(!resident.is_poisoned());
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        resident.step_checked(
            &device,
            &source,
            &hidden.full_view(),
            &mut checker,
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(resident.positions(), 1);
        Ok(())
    }

    #[test]
    fn forecast_and_identity_admission_precede_payload_or_device_effects() -> Result<()> {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 3, read_limits())?;
        assert_eq!(
            plan.memory().resident_tensor_bytes,
            (2 * 256 + 2 * 256 + 5 * 512 + 256 + 2 * 3 * 256 + 3 * 64) * 4
        );
        assert_eq!(plan.memory().packed_tile_bytes, 32 * 288);
        assert_eq!(plan.memory().host_staging_bytes, 65_536);
        assert_eq!(plan.retained_commands(), 10);
        for (layer, capacity) in [(0, 3), (4, 3), (3, 0), (3, 17)] {
            assert!(RecordedAttentionPlan::new(&binding, layer, capacity, read_limits()).is_err());
        }
        assert!(RecordedAttentionPlan::new(
            &binding,
            3,
            3,
            ReadLimits {
                encoded_bytes: 1023,
                ..read_limits()
            }
        )
        .is_err());
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        for limits in [
            AttentionLimits {
                resident_tensor_bytes: plan.memory().resident_tensor_bytes - 1,
                ..admitted(&plan)
            },
            AttentionLimits {
                packed_tile_bytes: plan.memory().packed_tile_bytes - 1,
                ..admitted(&plan)
            },
            AttentionLimits {
                host_staging_bytes: plan.memory().host_staging_bytes - 1,
                ..admitted(&plan)
            },
        ] {
            assert!(plan
                .prepare(&device, &source, limits, &mut || panic!(
                    "preflight refusal must precede effects"
                ))
                .is_err());
        }
        let (_other_temporary, other) = Fixture::hybrid().open();
        assert!(plan
            .prepare(
                &device,
                &Source::new(&other),
                admitted(&plan),
                &mut || panic!("foreign source reached effects")
            )
            .is_err());
        assert_eq!(source.reads.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[test]
    fn openmythos_layout_keeps_full_width_heads_and_counts_every_resident_payload() -> Result<()> {
        let shape = AttentionShape {
            query_heads: 24,
            kv_heads: 4,
            head_dim: 256,
            rope_dim: 64,
            rope_base: 10_000_000.0,
            epsilon: 1e-6,
            capacity: 16_384,
        };
        let l = Layout::new(5120, shape)?;
        assert_eq!(l.len * 4, 138_582_016);
        assert_eq!(l.query_gate.len, 12_288);
        assert_eq!(
            (l.query.len, l.attended.len, l.gated.len),
            (6144, 6144, 6144)
        );
        assert_eq!(l.raw_key.len, 1024);
        assert_eq!(l.cos.len + l.sin.len, shape.capacity * 64);
        assert_eq!(
            (l.key_cache.len + l.value_cache.len) * 4,
            shape.cache_bytes()?
        );
        let spans = [
            l.norms[0],
            l.norms[1],
            l.norms[2],
            l.cos,
            l.sin,
            l.normalized,
            l.query_gate,
            l.raw_key,
            l.query,
            l.attended,
            l.gated,
            l.key_cache,
            l.value_cache,
        ];
        let mut end = 0;
        for span in spans {
            assert_eq!(span.start, end);
            end += span.len;
        }
        assert_eq!(end, l.len);
        Ok(())
    }

    #[test]
    fn resident_two_positions_match_reference_interleaved_gates_and_partial_rope() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut resident = plan.prepare(&device, &source, admitted(&plan), &mut || Ok(()))?;
        assert_eq!(source.reads.load(Ordering::Relaxed), 3); // only F32 norms at preparation
        let mut oracle = AttentionState::new(plan.shape(), plan.shape().cache_bytes()?)?;
        let hidden = device.upload_f32(&[-123.; 258])?;
        let view = hidden.view(1, 256)?;
        let mut stats = StreamStats::default();
        for position in 0..2 {
            let input: Vec<_> = (0..256)
                .map(|i| (i % 11) as f32 * 0.1 + if position == 0 { 0.25 } else { -0.75 })
                .collect();
            let (expected, qg, gated) = reference(&binding, &source, &mut oracle, &input)?;
            device.write_f32(&view, &input)?;
            resident.step(&device, &source, &view, &mut stats, &mut || Ok(()))?;
            assert_eq!(resident.positions(), position + 1);
            close(&read(&device, &view)?, &expected);
            close(
                &read(&device, &plan.layout.query_gate.view(&resident.packet)?)?,
                &qg,
            );
            close(
                &read(&device, &plan.layout.gated.view(&resident.packet)?)?,
                &gated,
            );
            // Coordinates outside the rotated prefix remain normalized Q exactly.
            let qn = weights::f32_tensor(&plan.norms[1], &source, read_limits())?;
            let mut expected_q = [0.; 256];
            Cpu.rmsnorm(
                &mut expected_q,
                &qg[512..768],
                Some(&qn),
                plan.shape().epsilon,
            )?;
            let q = read(
                &device,
                &resident.packet.view(plan.layout.query.start + 256, 256)?,
            )?;
            close(&q[64..], &expected_q[64..]);
            if position == 1 {
                assert!(q[..64]
                    .iter()
                    .zip(&expected_q[..64])
                    .any(|(a, b)| (a - b).abs() > 0.01));
            }
        }
        let full = read(&device, &hidden.full_view())?;
        assert_eq!((full[0], full[257]), (-123., -123.));
        let expected_tiles: usize = plan
            .matrices
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        assert_eq!(stats.completed_tiles, (2 * expected_tiles) as u64);
        assert_eq!(stats.peak_read_bytes, plan.memory().packed_tile_bytes);
        let before = stats;
        assert!(resident
            .step(&device, &source, &view, &mut stats, &mut || panic!(
                "full cache must refuse before effects"
            ))
            .is_err());
        assert_eq!(stats, before);
        assert!(!resident.is_poisoned());
        resident.reset(&device, &mut || Ok(()))?;
        assert_eq!(resident.positions(), 0);
        assert_eq!(
            resident.reset_write_bytes(),
            ((plan.layout.len - plan.layout.mutable_start) * 4) as u64
        );
        assert!(read(
            &device,
            &resident.packet.view(
                plan.layout.mutable_start,
                plan.layout.len - plan.layout.mutable_start
            )?
        )?
        .iter()
        .all(|v| v.to_bits() == 0));
        assert!(
            read(&device, &plan.layout.norms[0].view(&resident.packet)?)?
                .iter()
                .all(|v| *v >= 1.0)
        );
        Ok(())
    }

    #[test]
    fn cancellation_after_partial_output_requires_complete_reset_and_caller_hidden_restore(
    ) -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut resident = plan.prepare(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[1.; 256])?;
        let mut stats = StreamStats::default();
        // Each successful matrix tile invokes four control boundaries. Cancel at
        // the after-fence boundary of the first output tile: hidden is partially
        // changed, but the position must never be published.
        let prior_tiles: usize = plan.matrices[..3]
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        let cancel_at = 4 + prior_tiles * 4 + 2 + 4;
        let mut calls = 0;
        assert!(resident
            .step(
                &device,
                &source,
                &hidden.full_view(),
                &mut stats,
                &mut || {
                    calls += 1;
                    ensure!(
                        calls != cancel_at,
                        "injected cancellation after first residual tile"
                    );
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(stats.completed_tiles as usize, prior_tiles + 1);
        assert_eq!(resident.positions(), 0);
        assert!(resident.is_poisoned());
        let partial = read(&device, &hidden.full_view())?;
        let tile = plan.matrices[3].tile_rows();
        assert!(partial[..tile].iter().all(|v| *v > 1.0));
        assert_eq!(&partial[tile..], &vec![1.; 256 - tile]);
        let before = stats;
        assert!(resident
            .step(
                &device,
                &source,
                &hidden.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert_eq!(stats, before);
        // Even a cancellation after the reset writes retains poison until an
        // entire reset, including its final control boundary, succeeds.
        let mut reset_calls = 0;
        assert!(resident
            .reset(&device, &mut || {
                reset_calls += 1;
                ensure!(reset_calls != 2, "reset interrupted");
                Ok(())
            })
            .is_err());
        assert!(resident.is_poisoned());
        assert!(resident.reset_write_bytes() > 0);
        resident.reset(&device, &mut || Ok(()))?;
        assert!(!resident.is_poisoned());
        close(&read(&device, &hidden.full_view())?, &partial); // reset never claims to restore caller hidden
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        resident.step(
            &device,
            &source,
            &hidden.full_view(),
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(resident.positions(), 1);
        Ok(())
    }

    #[test]
    fn foreign_source_owner_and_prestart_cancellation_leave_positions_unpoisoned() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedAttentionPlan::new(&binding, 3, 2, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let other_device = CpuExecutionDevice::new();
        let mut resident = plan.prepare(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[1.; 256])?;
        let wrong_hidden = other_device.upload_f32(&[1.; 256])?;
        let mut stats = StreamStats::default();
        let (_other_temporary, other) = Fixture::hybrid().open();
        assert!(resident
            .step(
                &device,
                &Source::new(&other),
                &hidden.full_view(),
                &mut stats,
                &mut || panic!("foreign source reached control")
            )
            .is_err());
        assert!(resident
            .step(
                &device,
                &source,
                &wrong_hidden.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert!(resident
            .step(
                &other_device,
                &source,
                &hidden.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert!(resident
            .step(
                &device,
                &source,
                &hidden.full_view(),
                &mut stats,
                &mut || anyhow::bail!("cancel before effects")
            )
            .is_err());
        assert_eq!(stats, StreamStats::default());
        assert_eq!(resident.positions(), 0);
        assert!(!resident.is_poisoned());
        close(&read(&device, &hidden.full_view())?, &[1.; 256]);
        Ok(())
    }
}
