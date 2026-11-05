//! Bounded resident FFN for a strict dense `qwen35` block.
//!
//! The CPU reference evaluates post-attention RMSNorm, gate/up projections,
//! unclipped SwiGLU (`limit=0`), and a down projection added to the residual stream.
//! This component records the same graph on an explicitly supplied device. It
//! streams unchanged Q4_K/Q6_K rows; activations never leave the device and there
//! is no CPU fallback. It does not establish complete-model or serving support.
//! The diagnostic `step` remains unchecked. `step_checked` requires an explicit
//! prepared finite validator and checks all five phase boundaries; native adapters
//! must provide status-only validation without an activation-read fallback. These
//! source-level checks do not establish native operator or whole-model qualification.
//!
//! The caller serializes the entire operation, including every tile and control
//! boundary. A partial failure may change caller hidden; the component stays
//! poisoned until an explicit successful reset. The enclosing model must restore
//! its hidden vector and reset/abandon its complete token as appropriate. Reset
//! clears owned mutable storage only; it cannot repair a poisoned native queue.

use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, ExecutionError, F32Tensor, F32TensorView, FiniteCheck,
    FiniteCheckSpec, LinearMode, RecordedBatch,
};
use beholdd_format::{CheckpointId, GgmlType, GgufTensor};
use beholdd_memory::WeightSource;

use super::config::Binding;
use super::recorded_weights::{ChunkStreamStats, PackedRowPlan, PrefillChunkSize, StreamStats};
use super::weights::{self, ReadLimits};

const ACTIVATION_ELEMENTS: usize = 4096;
const RESET_ELEMENTS: usize = 16_384;
const FINITE_CHECKS_PER_STEP: usize = 5;

/// Logical payload forecast, excluding caller hidden, source caches, allocator and
/// recording metadata, alignment and backend-private programs/scratch/conversion
/// buffers. A CPU oracle backend additionally owns its decoded matrix scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnMemory {
    /// One private F32 packet: norm + normalized hidden + gate + up.
    pub resident_tensor_bytes: u64,
    /// Largest simultaneously retained packed tile in device weight storage.
    pub packed_tile_bytes: u64,
    /// Largest model-owned host payload: packed read, raw+decoded norm, or bounded
    /// reset zero buffer. These stages have disjoint lifetimes.
    pub host_staging_bytes: u64,
}

/// Explicit admission ceilings with the same scope as [`FfnMemory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnLimits {
    /// Maximum resident F32 packet payload.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time packed device tile.
    pub packed_tile_bytes: u64,
    /// Maximum model-owned host staging payload.
    pub host_staging_bytes: u64,
}

/// Added finite-validation work, separate from layer payload and matrix traffic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnFiniteRequirements {
    /// Largest phase view: max(hidden width, twice the FFN width).
    pub max_view_elements: usize,
    /// Successful checks required by one complete FFN step.
    pub checks_per_step: u64,
    /// Added physical dispatches for all checks in one successful step.
    pub physical_dispatches_per_step: u64,
    /// Added logical submissions for all checks in one successful step.
    pub logical_submissions_per_step: u64,
    /// Added pending-status writes for all checks in one successful step.
    pub pending_write_bytes_per_step: u64,
    /// Added status reads for all checks in one successful step.
    pub status_read_bytes_per_step: u64,
    /// Backend's per-check costs and checker storage; the model budgets retained
    /// storage/initialization once for its shared checker, outside each layer.
    pub shared_checker: FiniteCheckSpec,
}

#[derive(Debug, Clone, Copy)]
struct Layout {
    dim: usize,
    width: usize,
    normalized: usize,
    gate: usize,
    up: usize,
    len: usize,
}

impl Layout {
    fn new(dim: usize, width: usize) -> Result<Self> {
        ensure!(
            (1..=5120).contains(&dim) && (1..=17408).contains(&width),
            "resident FFN dimensions exceed the bounded native probe profile"
        );
        ensure!(
            dim.is_multiple_of(256) && width.is_multiple_of(256),
            "resident FFN dimensions must contain whole 256-element blocks"
        );
        let normalized = dim;
        let gate = dim
            .checked_mul(2)
            .context("FFN norm/hidden size overflows")?;
        let up = gate.checked_add(width).context("FFN gate size overflows")?;
        let len = up.checked_add(width).context("FFN packet size overflows")?;
        Ok(Self {
            dim,
            width,
            normalized,
            gate,
            up,
            len,
        })
    }

    fn activation_commands(self, packet: &F32Tensor) -> Result<CommandList> {
        let mut commands = CommandList::new();
        for start in (0..self.width).step_by(ACTIVATION_ELEMENTS) {
            let count = (self.width - start).min(ACTIVATION_ELEMENTS);
            commands.swiglu(
                &packet.view(self.gate + start, count)?,
                &packet.view(self.up + start, count)?,
                0.0,
            )?;
        }
        Ok(commands)
    }
}

/// Additional model-owned chunk payload. Existing layer packets, packed tiles,
/// checker storage and backend recording/scratch costs remain separately charged.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnChunkMemory {
    /// B * (hidden D + normalized D + gate W + up W) F32 values.
    pub resident_tensor_bytes: u64,
    /// Hidden rows included in `resident_tensor_bytes`, not an additional charge.
    pub hidden_bytes: u64,
    /// Zero initialization performed by `allocate_f32`.
    pub initialization_write_bytes: u64,
    /// Complete explicit reset, including the hidden rows.
    pub reset_write_bytes: u64,
    /// One bounded zero buffer; no activation readback or decoded weights.
    pub host_staging_bytes: u64,
    /// Workspace struct plus 29 fixed temporary view slots. Logical metadata
    /// only; allocator/CommandList/backend metadata belongs to the model forecast.
    pub host_metadata_bytes: u64,
    /// One additional tensor allocation; never one allocation per layer.
    pub tensor_allocations: u64,
    /// Chunk code retains at most one nonlinear/packed recording at a time.
    pub max_live_recordings: u64,
    /// Maximum semantic commands in that recording (ceil(W/4096), at least one).
    pub max_batch_commands: u64,
}

/// Ceilings for the additional workspace, separate from existing layer limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FfnChunkLimits {
    /// Maximum payload of the single additional F32 arena.
    pub resident_tensor_bytes: u64,
    /// Maximum bounded host reset buffer.
    pub host_staging_bytes: u64,
    /// Maximum workspace/fixed-view metadata payload.
    pub host_metadata_bytes: u64,
}

/// Fixed, metadata-only layout shared across all same-width FFNs in one model.
#[derive(Debug, Clone, Copy)]
pub struct FfnChunkPlan {
    layout: Layout,
    size: PrefillChunkSize,
    elements: usize,
}

impl FfnChunkPlan {
    /// B1/B2/B4 only, using the same bounded D/W geometry as scalar FFN plans.
    pub fn new(dim: usize, width: usize, tokens: usize) -> Result<Self> {
        let size = PrefillChunkSize::new(tokens)?;
        let layout = Layout::new(dim, width)?;
        let elements = layout
            .len
            .checked_mul(size.tokens())
            .context("FFN chunk workspace overflows")?;
        elements
            .checked_mul(4)
            .context("FFN chunk byte extent overflows")?;
        Ok(Self {
            layout,
            size,
            elements,
        })
    }

    /// Fixed number of token rows in this workspace.
    pub fn tokens(self) -> usize {
        self.size.tokens()
    }
    /// Hidden width of each row.
    pub fn dim(self) -> usize {
        self.layout.dim
    }
    /// Gate/up width of each row.
    pub fn width(self) -> usize {
        self.layout.width
    }

    /// Additional bounded payload and peak recording counts, without allocation.
    pub fn memory(self) -> FfnChunkMemory {
        let bytes = self.elements as u64 * 4;
        FfnChunkMemory {
            resident_tensor_bytes: bytes,
            hidden_bytes: (self.tokens() * self.layout.dim) as u64 * 4,
            initialization_write_bytes: bytes,
            reset_write_bytes: bytes,
            host_staging_bytes: (RESET_ELEMENTS * 4) as u64,
            host_metadata_bytes: (std::mem::size_of::<FfnChunkWorkspace>()
                + 29 * std::mem::size_of::<F32TensorView>())
                as u64,
            tensor_allocations: 1,
            max_live_recordings: 1,
            max_batch_commands: self.layout.width.div_ceil(ACTIVATION_ELEMENTS) as u64,
        }
    }

    /// Admit before any control/device call, then allocate one zeroed workspace.
    /// The model's device/process admission must include this payload beforehand.
    pub fn prepare(
        self,
        device: &dyn ExecutionDevice,
        limits: FfnChunkLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<FfnChunkWorkspace> {
        let m = self.memory();
        for (name, required, admitted) in [
            (
                "tensor",
                m.resident_tensor_bytes,
                limits.resident_tensor_bytes,
            ),
            (
                "host staging",
                m.host_staging_bytes,
                limits.host_staging_bytes,
            ),
            (
                "metadata",
                m.host_metadata_bytes,
                limits.host_metadata_bytes,
            ),
        ] {
            ensure!(
                required <= admitted,
                "FFN chunk {name} needs {required} bytes, limit is {admitted}"
            );
        }
        control()?;
        let packet = device.allocate_f32(self.elements)?;
        control()?;
        Ok(FfnChunkWorkspace {
            plan: self,
            packet,
            poisoned: false,
            reset_write_bytes: 0,
        })
    }
}

/// One model-owned arena, reused across layers; private FFN temporaries never
/// escape. Hidden views are working storage, not published model output tickets.
/// Caller serialization covers embedding, serial cores, this FFN and reset.
#[derive(Debug)]
pub struct FfnChunkWorkspace {
    plan: FfnChunkPlan,
    packet: F32Tensor,
    poisoned: bool,
    reset_write_bytes: u64,
}

#[derive(Clone)]
struct ChunkViews {
    hidden: F32TensorView,
    normalized: F32TensorView,
    raw: F32TensorView,
    gate: F32TensorView,
    up: F32TensorView,
}

impl FfnChunkWorkspace {
    /// Immutable geometry and accounting for this arena.
    pub fn plan(&self) -> FfnChunkPlan {
        self.plan
    }
    /// Whether a failed mutation/reset blocks further checked chunk execution.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Successful reset writes, including writes preceding an interrupted reset.
    pub fn reset_write_bytes(&self) -> u64 {
        self.reset_write_bytes
    }

    /// Exact hidden row for embedding/core execution. Retaining this view keeps
    /// storage alive across reset but does not preserve its old semantic contents.
    pub fn hidden(&self, token: usize) -> Result<F32TensorView> {
        ensure!(
            token < self.plan.tokens(),
            "FFN chunk token index is outside workspace"
        );
        Ok(self
            .packet
            .view(token * self.plan.layout.len, self.plan.layout.dim)?)
    }

    fn views(&self, token: usize) -> Result<ChunkViews> {
        let hidden = self.hidden(token)?;
        let l = self.plan.layout;
        let base = token * l.len;
        Ok(ChunkViews {
            hidden,
            normalized: self.packet.view(base + l.normalized, l.dim)?,
            raw: self.packet.view(base + l.gate, 2 * l.width)?,
            gate: self.packet.view(base + l.gate, l.width)?,
            up: self.packet.view(base + l.up, l.width)?,
        })
    }

    /// Clear the whole shared arena in <=64KiB writes. The enclosing model must
    /// also reset all stateful cores/FFNs; clearing this arena alone is no rollback.
    pub fn reset(
        &mut self,
        device: &dyn ExecutionDevice,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        const ZEROES: [f32; RESET_ELEMENTS] = [0.0; RESET_ELEMENTS];
        self.poisoned = true;
        for start in (0..self.plan.elements).step_by(RESET_ELEMENTS) {
            control()?;
            let count = RESET_ELEMENTS.min(self.plan.elements - start);
            let next = self
                .reset_write_bytes
                .checked_add((count * 4) as u64)
                .context("FFN chunk reset counter overflows")?;
            device.write_f32(&self.packet.view(start, count)?, &ZEROES[..count])?;
            self.reset_write_bytes = next;
        }
        control()?;
        self.poisoned = false;
        Ok(())
    }
}

fn chunk_norm_commands(views: &ChunkViews, norm: &F32TensorView, eps: f32) -> Result<CommandList> {
    let mut commands = CommandList::new();
    commands.rmsnorm(&views.normalized, &views.hidden, Some(norm), eps)?;
    Ok(commands)
}

fn chunk_activation_commands(views: &ChunkViews) -> Result<CommandList> {
    let mut commands = CommandList::new();
    for first in (0..views.gate.len()).step_by(ACTIVATION_ELEMENTS) {
        let count = ACTIVATION_ELEMENTS.min(views.gate.len() - first);
        commands.swiglu(
            &views.gate.view(first, count)?,
            &views.up.view(first, count)?,
            0.0,
        )?;
    }
    Ok(commands)
}

/// Metadata-only plan for `hidden += down(silu(gate(RMS(hidden))) * up(RMS(hidden)))`.
/// Norm weights are stored GGUF values; no extra converter transformation is applied.
#[derive(Debug, Clone)]
pub struct RecordedFfnPlan {
    checkpoint: CheckpointId,
    eps: f32,
    norm: GgufTensor,
    matrices: [PackedRowPlan; 3],
    kinds: [GgmlType; 3],
    read_limits: ReadLimits,
    layout: Layout,
    memory: FfnMemory,
}

impl RecordedFfnPlan {
    /// Plan a bound hybrid layer's common dense FFN without payload reads/device
    /// effects. Full-attention and DeltaNet blocks use the same FFN equations.
    pub fn new(binding: &Binding, layer: usize, read_limits: ReadLimits) -> Result<Self> {
        let config = binding.config();
        let block = binding
            .blocks()
            .get(layer)
            .context("FFN layer is outside checkpoint")?;
        let layout = Layout::new(config.dim, config.feed_forward)?;
        ensure!(
            config.eps.is_finite() && config.eps > 0.0,
            "resident FFN epsilon must be finite and positive"
        );
        let matrices = [
            PackedRowPlan::new(&block.ffn.gate, read_limits)?,
            PackedRowPlan::new(&block.ffn.up, read_limits)?,
            PackedRowPlan::new(&block.ffn.down, read_limits)?,
        ];
        for (matrix, rows, cols) in [
            (&matrices[0], layout.width, layout.dim),
            (&matrices[1], layout.width, layout.dim),
            (&matrices[2], layout.dim, layout.width),
        ] {
            ensure!(
                matrix.rows() == rows && matrix.cols() == cols,
                "resident FFN projection shape differs"
            );
            ensure!(
                rows.checked_mul(4).context("FFN output bytes overflow")?
                    <= read_limits.decoded_bytes,
                "resident FFN projection output exceeds the configured decoded/result limit"
            );
        }
        let norm = block.post_attention_norm.clone();
        let norm_bytes = layout
            .dim
            .checked_mul(4)
            .context("FFN norm bytes overflow")?;
        ensure!(
            norm.kind().0 == 0
                && norm.dims() == [layout.dim as u64]
                && norm.nbytes() == norm_bytes as u64,
            "resident FFN requires an exact F32 post-attention normalization vector"
        );
        ensure!(
            norm_bytes <= read_limits.encoded_bytes && norm_bytes <= read_limits.decoded_bytes,
            "resident FFN norm exceeds the configured read/scratch limits"
        );
        let packed = matrices
            .iter()
            .map(PackedRowPlan::peak_read_bytes)
            .max()
            .unwrap();
        let norm_staging = norm_bytes
            .checked_mul(2)
            .context("FFN norm staging overflows")?;
        let reset_staging = (layout.len - layout.normalized).min(RESET_ELEMENTS) * 4;
        let memory = FfnMemory {
            resident_tensor_bytes: (layout.len as u64)
                .checked_mul(4)
                .context("FFN resident bytes overflow")?,
            packed_tile_bytes: packed as u64,
            host_staging_bytes: packed.max(norm_staging).max(reset_staging) as u64,
        };
        Ok(Self {
            checkpoint: binding.checkpoint_id(),
            eps: config.eps,
            norm,
            matrices,
            kinds: [
                block.ffn.gate.kind(),
                block.ffn.up.kind(),
                block.ffn.down.kind(),
            ],
            read_limits,
            layout,
            memory,
        })
    }

    /// Caller hidden width, excluded from the owned payload forecast.
    pub fn dim(&self) -> usize {
        self.layout.dim
    }
    /// Gate/up intermediate width.
    pub fn width(&self) -> usize {
        self.layout.width
    }
    /// Payload and staging forecast under the selected read/tile limits.
    pub fn memory(&self) -> FfnMemory {
        self.memory
    }
    /// Number of <=4096-coordinate SwiGLU commands per step.
    pub fn activation_commands(&self) -> usize {
        self.layout.width.div_ceil(ACTIVATION_ELEMENTS)
    }

    /// Fixed finite-validation phase count, separate from matrix stream counters.
    pub fn finite_checks_per_step(&self) -> usize {
        FINITE_CHECKS_PER_STEP
    }

    /// Admit an explicit finite capability and forecast its added per-step work.
    /// This is metadata only. Preparation/step call it with the actual device's
    /// capability, so callers cannot supply an override to enable an unsupported
    /// backend. Shared checker storage is excluded from [`FfnMemory`].
    pub fn finite_check_requirements(
        &self,
        spec: Option<FiniteCheckSpec>,
    ) -> Result<FfnFiniteRequirements> {
        let spec = spec.context("checked FFN requires explicit finite validation; no fallback")?;
        let raw_width = self
            .layout
            .width
            .checked_mul(2)
            .context("FFN raw gate/up size overflows")?;
        let max_view_elements = self.layout.dim.max(raw_width);
        ensure!(
            spec.max_elements >= max_view_elements,
            "finite checker cannot admit the FFN hidden and contiguous raw gate/up views"
        );
        let checks_per_step = FINITE_CHECKS_PER_STEP as u64;
        let total = |value: u64| {
            value
                .checked_mul(checks_per_step)
                .context("FFN finite-validation traffic forecast overflows")
        };
        Ok(FfnFiniteRequirements {
            max_view_elements,
            checks_per_step,
            physical_dispatches_per_step: total(spec.physical_dispatches)?,
            logical_submissions_per_step: total(spec.logical_submissions)?,
            pending_write_bytes_per_step: total(spec.pending_write_bytes)?,
            status_read_bytes_per_step: total(spec.status_read_bytes)?,
            shared_checker: spec,
        })
    }

    /// Prepare a checked component after finite-capability admission, before any
    /// packet allocation or norm read. The model prepares/budgets its shared
    /// checker separately and must supply it explicitly to every `step_checked`.
    pub fn prepare_checked(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: FfnLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentFfn> {
        self.finite_check_requirements(device.finite_check_spec())?;
        self.prepare(device, source, limits, control)
    }

    fn validate_source(&self, source: &dyn WeightSource) -> Result<()> {
        ensure!(
            source.checkpoint_id() == self.checkpoint,
            "resident FFN source belongs to a different checkpoint"
        );
        Ok(())
    }

    /// Allocate the private packet and read/upload only the small norm vector.
    /// The caller invokes this during its explicit first-request model load; this
    /// component has no service/discovery/selection hooks or background loading.
    /// Every identity and memory refusal precedes reads or device effects.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: FfnLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentFfn> {
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
                "resident FFN {name} needs {required} bytes, limit is {admitted}"
            );
        }
        for kind in self.kinds {
            ensure!(
                device.supports(kind),
                "backend refuses resident FFN projection format"
            );
        }
        control()?;
        let packet = device.allocate_f32(self.layout.len)?;
        control()?;
        let values = weights::f32_tensor(&self.norm, source, self.read_limits)?;
        control()?;
        device.write_f32(&packet.view(0, self.layout.dim)?, &values)?;
        drop(values);
        control()?;
        Ok(ResidentFfn {
            plan: self.clone(),
            packet,
            completed_steps: 0,
            poisoned: false,
            reset_write_bytes: 0,
        })
    }
}

/// One layer's private norm and scratch packet. No mutable scratch view escapes.
/// Each call accepts caller hidden explicitly, preserving its extent/ownership
/// checks even when the caller uses a subview of a larger activation allocation.
#[derive(Debug)]
pub struct ResidentFfn {
    plan: RecordedFfnPlan,
    packet: F32Tensor,
    completed_steps: u64,
    poisoned: bool,
    reset_write_bytes: u64,
}

impl ResidentFfn {
    /// Number of successful full residual updates since explicit reset.
    pub fn completed_steps(&self) -> u64 {
        self.completed_steps
    }
    /// Whether a partial operation/reset blocks further steps.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Successful mutable-packet reset writes, including partial failed resets.
    pub fn reset_write_bytes(&self) -> u64 {
        self.reset_write_bytes
    }

    fn normalization_commands(&self, hidden: &F32TensorView) -> Result<CommandList> {
        let l = self.plan.layout;
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &self.packet.view(l.normalized, l.dim)?,
            hidden,
            Some(&self.packet.view(0, l.dim)?),
            self.plan.eps,
        )?;
        Ok(commands)
    }

    /// Diagnostic unchecked post-attention FFN/residual update, publishing its
    /// count only after the final fence and control boundary. Matrix-only `StreamStats`
    /// retains actual read/upload/tile progress on failure; preparation norm reads
    /// and non-linear batches are intentionally outside those matrix counters.
    ///
    /// Ownership, identity, geometry and non-linear capability are checked before
    /// mutation. Streamed helpers retire each tile's batch/submission/weight handles
    /// before its next read. No intermediate activation readback occurs.
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
            "resident FFN is poisoned; reset or replace it"
        );
        self.plan.validate_source(source)?;
        let next = self
            .completed_steps
            .checked_add(1)
            .context("FFN completed-step counter overflows")?;
        let l = self.plan.layout;
        ensure!(
            hidden.len() == l.dim && !hidden.overlaps(&self.packet.full_view()),
            "resident FFN hidden shape or private-packet alias differs"
        );
        control()?;
        let normalization = device.record(&self.normalization_commands(hidden)?)?;
        let activation = device.record(&l.activation_commands(&self.packet)?)?;
        control()?;
        self.poisoned = true;
        execute(device, normalization, control)?;
        let normalized = self.packet.view(l.normalized, l.dim)?;
        for (matrix, destination) in self.plan.matrices[..2].iter().zip([l.gate, l.up]) {
            matrix.execute_rows(
                device,
                source,
                0..matrix.rows(),
                &normalized,
                &self.packet.view(destination, l.width)?,
                LinearMode::Overwrite,
                stats,
                control,
            )?;
        }
        execute(device, activation, control)?;
        let down = &self.plan.matrices[2];
        down.execute_rows(
            device,
            source,
            0..down.rows(),
            &self.packet.view(l.gate, l.width)?,
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
            stats,
            control,
        )?;
        control()?;
        self.completed_steps = next;
        self.poisoned = false;
        Ok(())
    }

    /// Complete a checked FFN update using a mandatory model-owned validator.
    ///
    /// Checks entry hidden, normalized hidden, contiguous raw gate/up BEFORE
    /// SwiGLU, activated gate, and residual hidden. There is no optional/unchecked
    /// fallback. Private intermediate views remain inside this component. Costs of
    /// the shared checker and its five checks are separate from matrix StreamStats.
    ///
    /// Identity, view ownership/geometry and finite capability refuse before any
    /// arithmetic mutation. Entry refusal/cancellation leaves this component usable.
    /// After mutation is armed, every error leaves poison set, including a final
    /// checker/control refusal after all residual tiles have completed. Only full
    /// success publishes the completed-step count. Reset never restores hidden.
    #[allow(clippy::too_many_arguments)]
    pub fn step_checked(
        &mut self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        hidden: &F32TensorView,
        check: &mut FiniteCheck,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "resident FFN is poisoned; reset or replace it"
        );
        self.plan.validate_source(source)?;
        self.plan
            .finite_check_requirements(device.finite_check_spec())?;
        let next = self
            .completed_steps
            .checked_add(1)
            .context("FFN completed-step counter overflows")?;
        let l = self.plan.layout;
        ensure!(
            hidden.len() == l.dim && !hidden.overlaps(&self.packet.full_view()),
            "resident FFN hidden shape or private-packet alias differs"
        );
        // Command construction checks owners across hidden and every private
        // operand without touching a device. The entry checker then proves that
        // those operands and the checker belong to the invoking execution owner.
        let normalization_commands = self.normalization_commands(hidden)?;
        let activation_commands = l.activation_commands(&self.packet)?;
        let normalized = self.packet.view(l.normalized, l.dim)?;
        let raw = self.packet.view(l.gate, 2 * l.width)?;
        let activated = self.packet.view(l.gate, l.width)?;
        finite_phase(device, check, hidden, "entry hidden", control)?;
        control()?;
        let normalization = device.record(&normalization_commands)?;
        let activation = device.record(&activation_commands)?;
        drop(normalization_commands);
        drop(activation_commands);
        control()?;
        self.poisoned = true;
        execute(device, normalization, control)?;
        finite_phase(device, check, &normalized, "normalized hidden", control)?;
        for (matrix, destination) in self.plan.matrices[..2].iter().zip([l.gate, l.up]) {
            matrix.execute_rows(
                device,
                source,
                0..matrix.rows(),
                &normalized,
                &self.packet.view(destination, l.width)?,
                LinearMode::Overwrite,
                stats,
                control,
            )?;
        }
        finite_phase(device, check, &raw, "raw gate/up", control)?;
        execute(device, activation, control)?;
        finite_phase(device, check, &activated, "activated gate", control)?;
        let down = &self.plan.matrices[2];
        down.execute_rows(
            device,
            source,
            0..down.rows(),
            &activated,
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
            stats,
            control,
        )?;
        finite_phase(device, check, hidden, "residual hidden", control)?;
        control()?;
        self.completed_steps = next;
        self.poisoned = false;
        Ok(())
    }

    /// Checked B1/B2/B4 FFN update, preserving each token's arithmetic graph and
    /// all five finite boundaries. Packed gate/up/down tiles are read once each,
    /// then used by serial unchanged scalar matvecs for every token.
    ///
    /// Every entry/owner/shape/capability refusal precedes arithmetic mutation.
    /// After arming, any error poisons both this component and the shared arena;
    /// no completed-step count advances until all B residual checks pass. The
    /// model must reset both owners and its cores/hidden before reuse. This method
    /// never publishes a model position or logits, and never reads activations.
    #[allow(clippy::too_many_arguments)]
    pub fn step_chunk_checked(
        &mut self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        workspace: &mut FfnChunkWorkspace,
        check: &mut FiniteCheck,
        stats: &mut StreamStats,
        chunk_stats: &mut ChunkStreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned && !workspace.poisoned,
            "FFN component/chunk workspace is poisoned; reset both owners"
        );
        self.plan.validate_source(source)?;
        self.plan
            .finite_check_requirements(device.finite_check_spec())?;
        let l = self.plan.layout;
        ensure!(
            workspace.plan.dim() == l.dim && workspace.plan.width() == l.width,
            "FFN chunk workspace geometry differs from the resident layer"
        );
        ensure!(
            !workspace
                .packet
                .full_view()
                .overlaps(&self.packet.full_view()),
            "FFN chunk aliases private layer storage"
        );
        let count = workspace.plan.tokens();
        let next = self
            .completed_steps
            .checked_add(count as u64)
            .context("FFN chunk completed-step counter overflows")?;
        for kind in self.plan.kinds {
            ensure!(
                device.supports(kind),
                "backend refuses resident FFN projection format"
            );
        }
        let first = workspace.views(0)?;
        let mut views: [ChunkViews; 4] = std::array::from_fn(|_| first.clone());
        drop(first);
        for (index, view) in views[..count].iter_mut().enumerate() {
            *view = workspace.views(index)?;
        }
        let norm = self.packet.view(0, l.dim)?;
        // Validate each nonlinear recording and retire it immediately. This
        // avoids B retained scratch allocations while retaining pre-mutation
        // capability/ownership refusal. Actual executions record again below.
        for view in &views[..count] {
            let commands = chunk_norm_commands(view, &norm, self.plan.eps)?;
            control()?;
            drop(device.record(&commands)?);
            drop(commands);
            let commands = chunk_activation_commands(view)?;
            control()?;
            drop(device.record(&commands)?);
            drop(commands);
        }
        for view in &views[..count] {
            finite_phase(device, check, &view.hidden, "entry hidden", control)?;
        }
        control()?;
        self.poisoned = true;
        workspace.poisoned = true;
        for view in &views[..count] {
            let commands = chunk_norm_commands(view, &norm, self.plan.eps)?;
            let batch = device.record(&commands)?;
            drop(commands);
            execute(device, batch, control)?;
            finite_phase(
                device,
                check,
                &view.normalized,
                "normalized hidden",
                control,
            )?;
        }
        {
            let inputs: [F32TensorView; 4] = std::array::from_fn(|i| views[i].normalized.clone());
            for (index, matrix) in self.plan.matrices[..2].iter().enumerate() {
                let outputs: [F32TensorView; 4] = std::array::from_fn(|i| {
                    if index == 0 {
                        views[i].gate.clone()
                    } else {
                        views[i].up.clone()
                    }
                });
                matrix.execute_rows_chunk(
                    device,
                    source,
                    0..matrix.rows(),
                    &inputs[..count],
                    &outputs[..count],
                    LinearMode::Overwrite,
                    stats,
                    chunk_stats,
                    control,
                )?;
            }
        }
        // Validate ALL raw pairs before any SwiGLU can mask a bad projection.
        for view in &views[..count] {
            finite_phase(device, check, &view.raw, "raw gate/up", control)?;
        }
        for view in &views[..count] {
            let commands = chunk_activation_commands(view)?;
            let batch = device.record(&commands)?;
            drop(commands);
            execute(device, batch, control)?;
            finite_phase(device, check, &view.gate, "activated gate", control)?;
        }
        {
            let inputs: [F32TensorView; 4] = std::array::from_fn(|i| views[i].gate.clone());
            let outputs: [F32TensorView; 4] = std::array::from_fn(|i| views[i].hidden.clone());
            let down = &self.plan.matrices[2];
            down.execute_rows_chunk(
                device,
                source,
                0..down.rows(),
                &inputs[..count],
                &outputs[..count],
                LinearMode::ScaledAdd { scale: 1.0 },
                stats,
                chunk_stats,
                control,
            )?;
        }
        for view in &views[..count] {
            finite_phase(device, check, &view.hidden, "residual hidden", control)?;
        }
        control()?;
        self.completed_steps = next;
        self.poisoned = false;
        workspace.poisoned = false;
        Ok(())
    }

    /// Clear normalized/gate/up storage in <=64 KiB writes, preserving the norm.
    /// Only a complete reset clears poison and completed-step metadata. The caller
    /// must restore hidden separately; native transport poison still refuses writes.
    pub fn reset(
        &mut self,
        device: &dyn ExecutionDevice,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.poisoned = true;
        let l = self.plan.layout;
        let count = (l.len - l.normalized).min(RESET_ELEMENTS);
        let mut zeroes = Vec::new();
        zeroes
            .try_reserve_exact(count)
            .context("allocating bounded FFN reset buffer")?;
        zeroes.resize(count, 0.0);
        for start in (l.normalized..l.len).step_by(RESET_ELEMENTS) {
            control()?;
            let len = (l.len - start).min(RESET_ELEMENTS);
            let next = self
                .reset_write_bytes
                .checked_add((len * 4) as u64)
                .context("FFN reset counter overflows")?;
            device.write_f32(&self.packet.view(start, len)?, &zeroes[..len])?;
            self.reset_write_bytes = next;
        }
        control()?;
        self.completed_steps = 0;
        self.poisoned = false;
        Ok(())
    }
}

fn finite_phase(
    device: &dyn ExecutionDevice,
    check: &mut FiniteCheck,
    input: &F32TensorView,
    phase: &'static str,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    device
        .check_finite(check, input, &mut || {
            control().map_err(|error| {
                ExecutionError::Shape(format!("FFN control interrupted: {error:#}"))
            })
        })
        .with_context(|| format!("resident FFN {phase} finite validation failed"))
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
    use crate::qwen35::config::tests::{Fixture, Temporary};
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::execution::CpuExecutionDevice;
    use beholdd_backend::Device;
    use beholdd_format::{Gguf, WeightRange};
    use beholdd_memory::{ResidentWeight, WeightAccess};
    use std::fs::OpenOptions;
    use std::io::{Seek, SeekFrom, Write};
    use std::sync::atomic::{AtomicU64, Ordering};

    include!("recorded_ffn/checked_tests.rs");

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
    fn admitted(plan: &RecordedFfnPlan) -> FfnLimits {
        let m = plan.memory();
        FfnLimits {
            resident_tensor_bytes: m.resident_tensor_bytes,
            packed_tile_bytes: m.packed_tile_bytes,
            host_staging_bytes: m.host_staging_bytes,
        }
    }
    fn read(device: &dyn ExecutionDevice, view: &F32TensorView) -> Result<Vec<f32>> {
        let mut values = vec![0.; view.len()];
        device.read_f32(view, &mut values)?;
        Ok(values)
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

    // Keep the production binder and exact locators. Only small synthetic payloads
    // are replaced: F32 norm, signed constant-row Q4 gate, and Q6 up/down.
    fn fixture() -> Result<(Temporary, Gguf)> {
        let mut f = Fixture::hybrid();
        f.tensor("blk.0.ffn_up.weight", 14, &[256, 256]);
        let (temporary, gguf) = f.open();
        let mut file = OpenOptions::new().write(true).open(&gguf.shards()[0])?;
        let norm = gguf.get("blk.0.post_attention_norm.weight").unwrap();
        let mut values = Vec::new();
        for i in 0..256 {
            values.extend((1.0 + (i % 7) as f32 * 0.02).to_le_bytes());
        }
        file.seek(SeekFrom::Start(norm.offset()))?;
        file.write_all(&values)?;
        for name in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
            let t = gguf.get(&format!("blk.0.{name}")).unwrap();
            let mut bytes = Vec::new();
            for row in 0..t.dims()[1] as usize {
                let value = (row % 13 + 1) as u8;
                for _ in 0..t.dims()[0] / 256 {
                    let mut block = vec![0; if t.kind().0 == 12 { 144 } else { 210 }];
                    if t.kind().0 == 12 {
                        let scale: u16 = if row % 2 == 0 { 0x1400 } else { 0x9400 };
                        block[..2].copy_from_slice(&scale.to_le_bytes()); // +/-2^-10
                        block[4..8].fill(1);
                        block[12..16].fill(1);
                        block[16..].fill(value | (value << 4));
                    } else {
                        block[..128].fill(value | (value << 4));
                        block[128..192].fill(0xaa);
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

    // This oracle follows reference.rs through the immediate Device interface,
    // independently of packet layout, recorded slicing and residual tile planning.
    fn reference(binding: &Binding, source: &Source<'_>, input: &[f32]) -> Result<Vec<f32>> {
        let block = &binding.blocks()[0];
        let norm = weights::f32_tensor(&block.post_attention_norm, source, read_limits())?;
        let mut normalized = vec![0.; input.len()];
        Cpu.rmsnorm(&mut normalized, input, Some(&norm), binding.config().eps)?;
        let mut gate = weights::matvec(
            &block.ffn.gate,
            source,
            &Cpu,
            &normalized,
            read_limits(),
            &mut || Ok(()),
        )?;
        let up = weights::matvec(
            &block.ffn.up,
            source,
            &Cpu,
            &normalized,
            read_limits(),
            &mut || Ok(()),
        )?;
        Cpu.swiglu(&mut gate, &up, 0.0)?;
        let down = weights::matvec(
            &block.ffn.down,
            source,
            &Cpu,
            &gate,
            read_limits(),
            &mut || Ok(()),
        )?;
        Ok(input.iter().zip(down).map(|(a, b)| a + b).collect())
    }

    #[test]
    fn strict_geometry_limits_and_checkpoint_admission_precede_effects() -> Result<()> {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        assert!(RecordedFfnPlan::new(&binding, 3, read_limits()).is_ok());
        assert!(RecordedFfnPlan::new(&binding, 4, read_limits()).is_err());
        assert_eq!(
            plan.memory(),
            FfnMemory {
                resident_tensor_bytes: 4096,
                packed_tile_bytes: 6720,
                host_staging_bytes: 6720
            }
        );
        assert_eq!(plan.activation_commands(), 1);
        for limits in [
            ReadLimits {
                encoded_bytes: 1023,
                ..read_limits()
            },
            ReadLimits {
                decoded_bytes: 1023,
                ..read_limits()
            },
            ReadLimits {
                tile_rows: 0,
                ..read_limits()
            },
        ] {
            assert!(RecordedFfnPlan::new(&binding, 0, limits).is_err());
        }
        for (dim, width) in [(0, 256), (5376, 256), (256, 17664), (255, 256), (256, 257)] {
            assert!(Layout::new(dim, width).is_err());
        }
        let actual = Layout::new(5120, 17408)?;
        assert_eq!(
            (actual.normalized, actual.gate, actual.up, actual.len),
            (5120, 10240, 27648, 45056)
        );
        assert_eq!(actual.len * 4, 180_224);
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        for limit in [
            FfnLimits {
                resident_tensor_bytes: plan.memory().resident_tensor_bytes - 1,
                ..admitted(&plan)
            },
            FfnLimits {
                packed_tile_bytes: plan.memory().packed_tile_bytes - 1,
                ..admitted(&plan)
            },
            FfnLimits {
                host_staging_bytes: plan.memory().host_staging_bytes - 1,
                ..admitted(&plan)
            },
        ] {
            assert!(plan
                .prepare(&device, &source, limit, &mut || panic!(
                    "admission failure reached effects"
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
    fn streamed_mixed_k_quants_match_cpu_reference_and_preserve_hidden_subview_guards() -> Result<()>
    {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut resident = plan.prepare(&device, &source, admitted(&plan), &mut || Ok(()))?;
        assert_eq!(source.reads.load(Ordering::Relaxed), 1); // no projection preload
        let hidden = device.upload_f32(&[-123.; 258])?;
        let view = hidden.view(1, 256)?;
        let mut stats = StreamStats::default();
        for step in 0..2 {
            let input: Vec<_> = (0..256)
                .map(|i| (i % 11) as f32 * 0.1 + if step == 0 { 0.25 } else { -0.75 })
                .collect();
            let expected = reference(&binding, &source, &input)?;
            device.write_f32(&view, &input)?;
            resident.step(&device, &source, &view, &mut stats, &mut || Ok(()))?;
            close(&read(&device, &view)?, &expected);
            assert_eq!(resident.completed_steps(), step + 1);
        }
        let all = read(&device, &hidden.full_view())?;
        assert_eq!((all[0], all[257]), (-123., -123.));
        let tiles: usize = plan
            .matrices
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        assert_eq!(stats.completed_tiles, (2 * tiles) as u64);
        assert_eq!(stats.peak_read_bytes, plan.memory().packed_tile_bytes);
        assert_eq!(stats.returned_read_bytes, stats.uploaded_bytes);
        Ok(())
    }

    #[test]
    fn five_swiglu_slices_cover_openmythos_width_without_clipping_or_up_writes() -> Result<()> {
        let device = CpuExecutionDevice::new();
        let layout = Layout::new(5120, 17408)?;
        let packet = device.upload_f32(&vec![-123.; layout.len])?;
        let gates: Vec<f32> = (0..layout.width)
            .map(|i| [-100., -10., -1., 0., 1., 10., 100.][i % 7])
            .collect();
        let up: Vec<_> = (0..layout.width).map(|i| (i % 7) as f32 - 3.).collect();
        let expected: Vec<_> = gates
            .iter()
            .zip(&up)
            .map(|(g, u)| g / (1.0f32 + (-g).exp()) * u)
            .collect();
        device.write_f32(&packet.view(layout.gate, layout.width)?, &gates)?;
        device.write_f32(&packet.view(layout.up, layout.width)?, &up)?;
        let commands = layout.activation_commands(&packet)?;
        assert_eq!(commands.len(), 5);
        let batch = device.record(&commands)?;
        drop(commands);
        execute(&device, batch, &mut || Ok(()))?;
        close(
            &read(&device, &packet.view(layout.gate, layout.width)?)?,
            &expected,
        );
        assert_eq!(read(&device, &packet.view(layout.up, layout.width)?)?, up);
        assert_eq!(
            read(&device, &packet.view(0, layout.gate)?)?,
            vec![-123.; layout.gate]
        );
        assert_eq!(expected[6], 300.); // limit=0 must not clamp the positive gate
        Ok(())
    }

    #[test]
    fn partial_down_projection_cancellation_poison_and_explicit_reset_are_honest() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
        let source = Source::new(&gguf);
        let device = CpuExecutionDevice::new();
        let mut resident = plan.prepare(&device, &source, admitted(&plan), &mut || Ok(()))?;
        let hidden = device.upload_f32(&[1.; 256])?;
        let mut stats = StreamStats::default();
        let prior_tiles: usize = plan.matrices[..2]
            .iter()
            .map(|m| m.rows().div_ceil(m.tile_rows()))
            .sum();
        let cancel_at = 4 + 4 * prior_tiles + 2 + 4;
        let mut calls = 0;
        assert!(resident
            .step(
                &device,
                &source,
                &hidden.full_view(),
                &mut stats,
                &mut || {
                    calls += 1;
                    ensure!(calls != cancel_at, "cancel after first residual tile fence");
                    Ok(())
                }
            )
            .is_err());
        assert_eq!(stats.completed_tiles as usize, prior_tiles + 1);
        assert!(resident.is_poisoned());
        assert_eq!(resident.completed_steps(), 0);
        let partial = read(&device, &hidden.full_view())?;
        let tile = plan.matrices[2].tile_rows();
        assert!(partial[..tile].iter().all(|v| *v > 1.));
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
        let mut reset_calls = 0;
        assert!(resident
            .reset(&device, &mut || {
                reset_calls += 1;
                ensure!(reset_calls != 2, "cancel after reset write");
                Ok(())
            })
            .is_err());
        assert!(resident.is_poisoned());
        assert_eq!(resident.reset_write_bytes(), 3072);
        resident.reset(&device, &mut || Ok(()))?;
        assert_eq!(resident.reset_write_bytes(), 6144);
        assert!(!resident.is_poisoned());
        assert!(read(
            &device,
            &resident.packet.view(
                plan.layout.normalized,
                plan.layout.len - plan.layout.normalized
            )?
        )?
        .iter()
        .all(|v| v.to_bits() == 0));
        assert!(read(&device, &resident.packet.view(0, plan.dim())?)?
            .iter()
            .all(|v| *v >= 1.));
        close(&read(&device, &hidden.full_view())?, &partial);
        device.write_f32(&hidden.full_view(), &[1.; 256])?;
        resident.step(
            &device,
            &source,
            &hidden.full_view(),
            &mut stats,
            &mut || Ok(()),
        )?;
        assert_eq!(resident.completed_steps(), 1);
        Ok(())
    }

    #[test]
    fn foreign_source_owner_counter_and_prestart_control_fail_without_mutation() -> Result<()> {
        let (_temporary, gguf) = fixture()?;
        let binding = Binding::load(&gguf)?;
        let plan = RecordedFfnPlan::new(&binding, 0, read_limits())?;
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
        resident.completed_steps = u64::MAX;
        assert!(resident
            .step(
                &device,
                &source,
                &hidden.full_view(),
                &mut stats,
                &mut || panic!("counter overflow reached effects")
            )
            .is_err());
        assert_eq!(stats, StreamStats::default());
        assert!(!resident.is_poisoned());
        close(&read(&device, &hidden.full_view())?, &[1.; 256]);
        Ok(())
    }
}
