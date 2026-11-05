//! Explicit native-only composition of one strict `qwen35` Delta attention block.
//!
//! Pre-attention normalization and five packed projections surround the resident
//! four-stage Delta sequence. Projection tiles write directly into its checked
//! packet views; no host repacking, activation/state readback or CPU fallback is
//! used. The sequence performs its own bounded status-only control read. FFN is a
//! separate component applied after this block's residual update.
//!
//! This is unqualified graph composition, not a serving capability. The diagnostic
//! `prepare`/`step` APIs retain their original behavior. `prepare_checked` requires
//! explicit finite capability, and `step_checked` requires a shared opaque checker
//! for hidden entry, normalized hidden and final residual. The core's own status
//! protocol covers its projected inputs and arithmetic. Source-level composition
//! does not establish native numerical qualification or authorize serving.
//! The caller must serialize complete steps with all uses of its hidden vector and
//! device. Any failure after mutation begins poisons this component even if the
//! inner sequence already committed. Explicit reset restores owned state only;
//! the model owner must restore hidden and reset/abandon the complete token.

#![cfg(all(feature = "hag-native", target_os = "linux"))]

use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, ExecutionError, F32Tensor, F32TensorView, FiniteCheck,
    FiniteCheckSpec, LinearMode,
};
use beholdd_backend::hag::native_execution::delta::{
    NativeDeltaLayout, NativeDeltaParameters, NativeDeltaSequence, NativeDeltaShape,
    NativeDeltaStats,
};
use beholdd_backend::hag::native_execution::{NativeExecutionDevice, NativeExecutionProfile};
use beholdd_format::{CheckpointId, GgmlType, GgufTensor};
use beholdd_memory::WeightSource;

use super::config::{Attention, Binding};
use super::recorded_weights::{PackedRowPlan, StreamStats};
use super::weights::{self, ReadLimits};

const MAX_CAPACITY: usize = 16_384;
const RESET_CHUNK_BYTES: u64 = 64 * 1024;

/// Logical payload forecast. Caller hidden, source caches, allocator/recording
/// metadata, arena alignment, program storage and backend-private staging/scratch
/// are separate costs. Resident parameters include the epsilon trailer and guards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaMemory {
    /// Sequence packet plus a separate [attention_norm, normalized_hidden] packet.
    pub resident_tensor_bytes: u64,
    /// Largest simultaneously retained device weight tile.
    pub packed_tile_bytes: u64,
    /// Maximum of retained parameter vectors plus largest temporary encoded read,
    /// raw+decoded attention norm, packed tile, and bounded reset staging.
    pub host_staging_bytes: u64,
}

/// Explicit admission limits with the same scope as [`DeltaMemory`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaLimits {
    /// Maximum aggregate resident F32 packet payloads.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time packed weight tile.
    pub packed_tile_bytes: u64,
    /// Maximum model-owned host staging payload.
    pub host_staging_bytes: u64,
}

/// Additional finite-validation costs, separate from [`DeltaMemory`].
///
/// The model owns one shared checker, so its retained tensor/initialization costs
/// must be counted once across the model, not once for every Delta layer. This
/// wrapper adds three checks; the core's existing per-head status read remains a
/// separate cost. Refused/cancelled attempts may incur a prefix of these transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeltaFiniteRequirements {
    /// Largest contiguous view this wrapper passes to the checker.
    pub max_view_elements: usize,
    /// Exact capability supplied by the execution device's metadata query.
    pub shared_checker: FiniteCheckSpec,
    /// Hidden entry, normalized hidden, and final residual.
    pub checks_per_step: u64,
    /// Additional successful-step physical dispatches.
    pub physical_dispatches_per_step: u64,
    /// Additional successful-step logical submissions.
    pub logical_submissions_per_step: u64,
    /// Additional pending-status writes per complete step.
    pub pending_write_bytes_per_step: u64,
    /// Additional status-only reads per complete step.
    pub status_read_bytes_per_step: u64,
}

fn finite_requirements(
    dim: usize,
    capability: Option<FiniteCheckSpec>,
) -> Result<DeltaFiniteRequirements> {
    let spec = capability
        .context("checked resident Delta requires explicit finite validation; no fallback")?;
    ensure!(
        dim != 0 && dim <= spec.max_elements,
        "checked resident Delta hidden width exceeds one finite-check view"
    );
    let times_three = |value: u64| {
        value
            .checked_mul(3)
            .context("Delta finite-check traffic overflows")
    };
    Ok(DeltaFiniteRequirements {
        max_view_elements: dim,
        shared_checker: spec,
        checks_per_step: 3,
        physical_dispatches_per_step: times_three(spec.physical_dispatches)?,
        logical_submissions_per_step: times_three(spec.logical_submissions)?,
        pending_write_bytes_per_step: times_three(spec.pending_write_bytes)?,
        status_read_bytes_per_step: times_three(spec.status_read_bytes)?,
    })
}

/// Invoke the explicitly implemented backend capability. The helper does not read
/// activations, prepare a second checker, or substitute a host arithmetic path.
fn check_resident_finite(
    device: &dyn ExecutionDevice,
    checker: &mut FiniteCheck,
    input: &F32TensorView,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    device.check_finite(checker, input, &mut || {
        control().map_err(|error| {
            ExecutionError::Shape(format!(
                "resident Delta finite control interrupted: {error:#}"
            ))
        })
    })?;
    Ok(())
}

fn forecast(
    dim: usize,
    layout: NativeDeltaLayout,
    parameter_bytes: [u64; 4],
    packed: u64,
) -> Result<DeltaMemory> {
    let outside = (dim as u64)
        .checked_mul(8)
        .context("Delta norm/hidden bytes overflow")?;
    let parameters = parameter_bytes.iter().try_fold(0u64, |sum, bytes| {
        sum.checked_add(*bytes)
            .context("Delta parameter total overflows")
    })?;
    ensure!(
        parameters.checked_add(4) == Some(u64::from(layout.parameter_bytes)),
        "Delta stored parameter payload differs from sequence layout"
    );
    let staging = parameters
        .checked_add(*parameter_bytes.iter().max().unwrap())
        .context("Delta parameter staging overflows")?;
    Ok(DeltaMemory {
        resident_tensor_bytes: u64::from(layout.packet_bytes)
            .checked_add(outside)
            .context("Delta resident total overflows")?,
        packed_tile_bytes: packed,
        host_staging_bytes: staging.max(outside).max(packed).max(RESET_CHUNK_BYTES),
    })
}

/// Metadata-only native Delta composition plan, distinct from full attention.
#[derive(Debug, Clone)]
pub struct RecordedDeltaPlan {
    checkpoint: CheckpointId,
    dim: usize,
    capacity: usize,
    shape: NativeDeltaShape,
    layout: NativeDeltaLayout,
    attention_norm: GgufTensor,
    parameters: [GgufTensor; 4],
    matrices: [PackedRowPlan; 5],
    kinds: [GgmlType; 5],
    read_limits: ReadLimits,
    memory: DeltaMemory,
}

impl RecordedDeltaPlan {
    /// Metadata-only checked-path admission and additional traffic forecast.
    /// The whole-model owner queries every plan before any preparation effects.
    /// A supplied spec describes capability, not checker ownership or numerical
    /// qualification; `step_checked` validates the actual checker and input.
    pub fn finite_check_requirements(
        &self,
        capability: Option<FiniteCheckSpec>,
    ) -> Result<DeltaFiniteRequirements> {
        finite_requirements(self.dim, capability)
    }

    /// Validate strict Delta geometry, exact tensor manifests and all read limits
    /// without loading any checkpoint payload or opening a device.
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
            .context("Delta layer is outside checkpoint")?;
        let Attention::DeltaNet(delta) = &block.attention else {
            anyhow::bail!("resident Delta refuses a full-attention block");
        };
        ensure!(
            (1..=MAX_CAPACITY).contains(&capacity) && capacity <= config.context,
            "resident Delta capacity must be 1..=16384 and within checkpoint context"
        );
        ensure!(
            (1..=5120).contains(&config.dim) && config.ssm_conv_kernel == 4,
            "resident Delta requires hidden width <=5120 and four convolution taps"
        );
        let shape = NativeDeltaShape {
            key_heads: config.ssm_group_count,
            value_heads: config.ssm_time_step_rank,
            head_dim: config.ssm_state_size,
            epsilon: config.eps,
        };
        let layout = shape.layout()?;
        ensure!(
            layout.channels as usize == config.delta_qkv_width()
                && layout.value_width as usize == config.ssm_inner_size
                && shape.head_dim == config.delta_value_head_dim(),
            "Delta sequence geometry differs from binding"
        );
        let matrices = [
            PackedRowPlan::new(&delta.qkv, read_limits)?,
            PackedRowPlan::new(&delta.gate, read_limits)?,
            PackedRowPlan::new(&delta.alpha, read_limits)?,
            PackedRowPlan::new(&delta.beta, read_limits)?,
            PackedRowPlan::new(&delta.output, read_limits)?,
        ];
        for (matrix, rows, cols) in [
            (&matrices[0], layout.channels as usize, config.dim),
            (&matrices[1], layout.value_width as usize, config.dim),
            (&matrices[2], shape.value_heads, config.dim),
            (&matrices[3], shape.value_heads, config.dim),
            (&matrices[4], config.dim, layout.value_width as usize),
        ] {
            ensure!(
                matrix.rows() == rows && matrix.cols() == cols,
                "resident Delta projection shape differs"
            );
            ensure!(
                rows.checked_mul(4)
                    .context("Delta projection output bytes overflow")?
                    <= read_limits.decoded_bytes,
                "Delta projection output exceeds the configured decoded/result limit"
            );
        }
        let attention_norm = block.attention_norm.clone();
        let parameters = [
            delta.conv.clone(),
            delta.a.clone(),
            delta.dt_bias.clone(),
            delta.norm.clone(),
        ];
        let checked_f32 = |tensor: &GgufTensor, dims: &[u64]| -> Result<u64> {
            ensure!(
                tensor.kind().0 == 0 && tensor.dims() == dims,
                "resident Delta requires exact F32 parameter geometry"
            );
            let bytes = dims.iter().try_fold(4u64, |bytes, dim| {
                bytes
                    .checked_mul(*dim)
                    .context("Delta F32 parameter size overflows")
            })?;
            ensure!(
                bytes == tensor.nbytes(),
                "resident Delta F32 parameter extent differs"
            );
            ensure!(
                bytes <= read_limits.encoded_bytes as u64
                    && bytes <= read_limits.decoded_bytes as u64,
                "resident Delta F32 parameter exceeds configured read/scratch limits"
            );
            Ok(bytes)
        };
        checked_f32(&attention_norm, &[config.dim as u64])?;
        let parameter_bytes = [
            checked_f32(&parameters[0], &[4, u64::from(layout.channels)])?,
            checked_f32(&parameters[1], &[shape.value_heads as u64])?,
            checked_f32(&parameters[2], &[shape.value_heads as u64])?,
            checked_f32(&parameters[3], &[shape.head_dim as u64])?,
        ];
        let packed = matrices
            .iter()
            .map(|matrix| matrix.peak_read_bytes() as u64)
            .max()
            .unwrap();
        let memory = forecast(config.dim, layout, parameter_bytes, packed)?;
        Ok(Self {
            checkpoint: binding.checkpoint_id(),
            dim: config.dim,
            capacity,
            shape,
            layout,
            attention_norm,
            parameters,
            matrices,
            kinds: [
                delta.qkv.kind(),
                delta.gate.kind(),
                delta.alpha.kind(),
                delta.beta.kind(),
                delta.output.kind(),
            ],
            read_limits,
            memory,
        })
    }

    /// Caller hidden width, excluded from this plan's resident payload.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Maximum successfully evaluated positions before explicit reset.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Checked recurrent geometry.
    pub fn shape(&self) -> NativeDeltaShape {
        self.shape
    }
    /// Sequence packet geometry for independent admission/proof.
    pub fn layout(&self) -> NativeDeltaLayout {
        self.layout
    }
    /// Exact resident payload and conservative staging forecast.
    pub fn memory(&self) -> DeltaMemory {
        self.memory
    }

    fn validate_source(&self, source: &dyn WeightSource) -> Result<()> {
        ensure!(
            source.checkpoint_id() == self.checkpoint,
            "resident Delta source belongs to a different checkpoint"
        );
        Ok(())
    }

    fn admit(
        &self,
        source: &dyn WeightSource,
        limits: DeltaLimits,
        profile: NativeExecutionProfile,
    ) -> Result<()> {
        self.validate_source(source)?;
        ensure!(
            profile == NativeExecutionProfile::Qwen35Probe,
            "resident Delta requires explicit Qwen35Probe profile"
        );
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
                "resident Delta {name} needs {required} bytes, limit is {admitted}"
            );
        }
        Ok(())
    }

    /// Prepare a checked-path component during an explicitly requested model load.
    /// Capability/width/traffic admission precedes all packet allocations and
    /// parameter reads. The caller separately prepares/budgets one shared checker;
    /// this method does not allocate another checker or certify a particular handle.
    pub fn prepare_checked(
        &self,
        device: &NativeExecutionDevice,
        source: &dyn WeightSource,
        limits: DeltaLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentDelta> {
        self.admit(source, limits, device.profile())?;
        self.finite_check_requirements(device.finite_check_spec())?;
        self.prepare(device, source, limits, control)
    }

    /// Diagnostic preparation during an explicitly requested model load.
    /// It does not require the finite capability; checked owners must call
    /// `prepare_checked` instead. Identity/profile/format
    /// and payload limits precede every allocation/read. First upload/drop the
    /// attention norm, then retain only the four small Delta parameter vectors for
    /// sequence construction, dropping all four immediately afterward. No matrix
    /// is preloaded. Backend arena availability is enforced by its allocators.
    pub fn prepare(
        &self,
        device: &NativeExecutionDevice,
        source: &dyn WeightSource,
        limits: DeltaLimits,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<ResidentDelta> {
        self.admit(source, limits, device.profile())?;
        ensure!(
            !device.stats()?.poisoned,
            "resident Delta cannot prepare on a poisoned queue"
        );
        for kind in self.kinds {
            ensure!(
                device.supports(kind),
                "native device refuses Delta projection format"
            );
        }
        control()?;
        let outside = device.allocate_f32(2 * self.dim)?;
        control()?;
        let norm = weights::f32_tensor(&self.attention_norm, source, self.read_limits)?;
        control()?;
        device.write_f32(&outside.view(0, self.dim)?, &norm)?;
        drop(norm);
        let mut load = |index| -> Result<Vec<f32>> {
            control()?;
            weights::f32_tensor(&self.parameters[index], source, self.read_limits)
        };
        let convolution = load(0)?;
        let a = load(1)?;
        let dt_bias = load(2)?;
        let norm = load(3)?;
        control()?;
        let sequence = NativeDeltaSequence::new(
            device,
            self.shape,
            NativeDeltaParameters {
                convolution: &convolution,
                a: &a,
                dt_bias: &dt_bias,
                norm: &norm,
            },
        )?;
        drop((convolution, a, dt_bias, norm));
        control()?;
        Ok(ResidentDelta {
            plan: self.clone(),
            device: device.clone(),
            outside,
            sequence,
            positions: 0,
            poisoned: false,
            outside_reset_write_bytes: 0,
        })
    }
}

/// Private native Delta layer state. No sequence projection/state/output aliases
/// escape this owner; only bounded accounting and immutable geometry are exposed.
pub struct ResidentDelta {
    plan: RecordedDeltaPlan,
    device: NativeExecutionDevice,
    outside: F32Tensor,
    sequence: NativeDeltaSequence,
    positions: usize,
    poisoned: bool,
    outside_reset_write_bytes: u64,
}

impl ResidentDelta {
    /// Successfully completed whole-layer residual updates, not inner commits.
    pub fn positions(&self) -> usize {
        self.positions
    }
    /// Explicit evaluated-position capacity.
    pub fn capacity(&self) -> usize {
        self.plan.capacity
    }
    /// Whether a partial step/reset blocks further use until explicit reset.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Status/reset/initialization accounting from the private sequence.
    pub fn sequence_stats(&self) -> NativeDeltaStats {
        self.sequence.stats()
    }
    /// Successful normalized-hidden reset writes, additional to sequence stats.
    pub fn outside_reset_write_bytes(&self) -> u64 {
        self.outside_reset_write_bytes
    }

    /// Evaluate one layer through the original unchecked diagnostic path.
    /// Qualified composition must require `step_checked`; this API is not its
    /// fallback. Uses only resident arithmetic and status-only control
    /// reads. Source/capacity/owner refusals precede mutation. Every error after
    /// normalization starts poisons the whole component, including cancellation
    /// after the inner sequence has committed but before residual projection ends.
    pub fn step(
        &mut self,
        source: &dyn WeightSource,
        hidden: &F32TensorView,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "resident Delta is poisoned; reset or replace it"
        );
        ensure!(
            self.positions < self.capacity(),
            "resident Delta capacity exhausted"
        );
        self.plan.validate_source(source)?;
        ensure!(
            hidden.len() == self.plan.dim && !hidden.overlaps(&self.outside.full_view()),
            "resident Delta hidden shape or private-packet alias differs"
        );
        control()?;
        let normalized = self.outside.view(self.plan.dim, self.plan.dim)?;
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &normalized,
            hidden,
            Some(&self.outside.view(0, self.plan.dim)?),
            self.plan.shape.epsilon,
        )?;
        let batch = self.device.record(&commands)?;
        drop(commands);
        control()?;
        self.poisoned = true;
        let submission = self.device.submit(&batch)?;
        self.device.wait(&submission)?;
        drop(submission);
        drop(batch);
        control()?;
        let views = self.sequence.projection_views()?;
        for (matrix, output) in
            self.plan.matrices[..4]
                .iter()
                .zip([views.qkv, views.z, views.alpha, views.beta])
        {
            matrix.execute_rows(
                &self.device,
                source,
                0..matrix.rows(),
                &normalized,
                &output,
                LinearMode::Overwrite,
                stats,
                control,
            )?;
        }
        self.sequence.step_controlled(&mut || {
            control().map_err(|error| {
                ExecutionError::Shape(format!("resident Delta control interrupted: {error:#}"))
            })
        })?;
        let output = self.sequence.output_view()?;
        let matrix = &self.plan.matrices[4];
        matrix.execute_rows(
            &self.device,
            source,
            0..matrix.rows(),
            &output,
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
            stats,
            control,
        )?;
        drop(output);
        control()?;
        self.positions += 1;
        self.poisoned = false;
        Ok(())
    }

    /// Evaluate a layer with mandatory shared finite validation at three boundaries.
    ///
    /// Source/capacity/shape/capability and the entry checker/input ownership check
    /// precede any normalization or projection mutation. An entry finite refusal
    /// leaves this component's positions/state unchanged. Native uncertainty still
    /// permanently poisons the device, including during the entry status check.
    /// Every error after normalization begins leaves this wrapper poisoned, even
    /// if the inner Delta sequence has already committed. Only a final successful
    /// residual check and control callback publish the wrapper's next position.
    ///
    /// The core validates projected qkv/z/alpha/beta and its internal arithmetic
    /// before status publication; these checks are additional to that protocol.
    /// Whole-model serialization must span checks and immediate consumption.
    pub fn step_checked(
        &mut self,
        source: &dyn WeightSource,
        hidden: &F32TensorView,
        checker: &mut FiniteCheck,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            !self.poisoned,
            "resident Delta is poisoned; reset or replace it"
        );
        ensure!(
            self.positions < self.capacity(),
            "resident Delta capacity exhausted"
        );
        self.plan.validate_source(source)?;
        ensure!(
            hidden.len() == self.plan.dim && !hidden.overlaps(&self.outside.full_view()),
            "resident Delta hidden shape or private-packet alias differs"
        );
        self.plan
            .finite_check_requirements(self.device.finite_check_spec())?;
        // This call validates the real checker/input owners before its own control
        // transfers. No RMS recording or checkpoint payload read has occurred.
        check_resident_finite(&self.device, checker, hidden, control)
            .context("checking Delta entry hidden")?;
        control()?;
        let normalized = self.outside.view(self.plan.dim, self.plan.dim)?;
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &normalized,
            hidden,
            Some(&self.outside.view(0, self.plan.dim)?),
            self.plan.shape.epsilon,
        )?;
        let batch = self.device.record(&commands)?;
        drop(commands);
        control()?;
        self.poisoned = true;
        let submission = self.device.submit(&batch)?;
        self.device.wait(&submission)?;
        drop(submission);
        drop(batch);
        control()?;
        check_resident_finite(&self.device, checker, &normalized, control)
            .context("checking Delta normalized hidden")?;
        let views = self.sequence.projection_views()?;
        for (matrix, output) in
            self.plan.matrices[..4]
                .iter()
                .zip([views.qkv, views.z, views.alpha, views.beta])
        {
            matrix.execute_rows(
                &self.device,
                source,
                0..matrix.rows(),
                &normalized,
                &output,
                LinearMode::Overwrite,
                stats,
                control,
            )?;
        }
        self.sequence.step_controlled(&mut || {
            control().map_err(|error| {
                ExecutionError::Shape(format!("resident Delta control interrupted: {error:#}"))
            })
        })?;
        let output = self.sequence.output_view()?;
        let matrix = &self.plan.matrices[4];
        matrix.execute_rows(
            &self.device,
            source,
            0..matrix.rows(),
            &output,
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
            stats,
            control,
        )?;
        drop(output);
        check_resident_finite(&self.device, checker, hidden, control)
            .context("checking Delta residual hidden")?;
        control()?;
        self.positions += 1;
        self.poisoned = false;
        Ok(())
    }

    /// Zero normalized hidden and both complete sequence banks/scratch. Norm and
    /// sequence parameters remain immutable. Partial failure keeps this component
    /// poisoned even when the inner sequence reset has already completed. Native
    /// transport poison cannot be recovered; caller hidden is never restored here.
    pub fn reset(&mut self, control: &mut dyn FnMut() -> Result<()>) -> Result<()> {
        self.poisoned = true;
        ensure!(
            !self.device.stats()?.poisoned,
            "cannot reset resident Delta on a poisoned queue"
        );
        control()?;
        let mut zeroes = Vec::new();
        zeroes
            .try_reserve_exact(self.plan.dim)
            .context("allocating bounded Delta normalization reset buffer")?;
        zeroes.resize(self.plan.dim, 0.0);
        let next = self
            .outside_reset_write_bytes
            .checked_add((self.plan.dim * 4) as u64)
            .context("Delta reset counter overflows")?;
        control()?;
        self.device
            .write_f32(&self.outside.view(self.plan.dim, self.plan.dim)?, &zeroes)?;
        self.outside_reset_write_bytes = next;
        drop(zeroes);
        control()?;
        self.sequence.reset()?;
        control()?;
        self.positions = 0;
        self.poisoned = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35::config::tests::Fixture;
    use beholdd_backend::execution::CpuExecutionDevice;
    use beholdd_format::{Gguf, WeightRange};
    use beholdd_memory::{ResidentWeight, WeightAccess};

    struct NoRead<'a>(&'a Gguf);
    impl WeightSource for NoRead<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.0.checkpoint_id()
        }
        fn load_range(
            &self,
            _range: &WeightRange,
            _access: WeightAccess,
        ) -> Result<ResidentWeight> {
            panic!("metadata admission must not read payloads")
        }
    }
    fn limits(plan: &RecordedDeltaPlan) -> DeltaLimits {
        let m = plan.memory();
        DeltaLimits {
            resident_tensor_bytes: m.resident_tensor_bytes,
            packed_tile_bytes: m.packed_tile_bytes,
            host_staging_bytes: m.host_staging_bytes,
        }
    }

    fn native_finite_spec() -> FiniteCheckSpec {
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
    fn checked_metadata_refuses_absent_or_small_capability_without_a_device() -> Result<()> {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let plan = RecordedDeltaPlan::new(&binding, 0, 1, ReadLimits::default())?;
        let original_memory = plan.memory();
        assert!(plan.finite_check_requirements(None).is_err());
        for max_elements in [0, plan.dim() - 1] {
            assert!(plan
                .finite_check_requirements(Some(FiniteCheckSpec {
                    max_elements,
                    ..native_finite_spec()
                }))
                .is_err());
        }
        let requirements = plan.finite_check_requirements(Some(native_finite_spec()))?;
        assert_eq!(
            requirements,
            DeltaFiniteRequirements {
                max_view_elements: plan.dim(),
                shared_checker: native_finite_spec(),
                checks_per_step: 3,
                physical_dispatches_per_step: 3,
                logical_submissions_per_step: 3,
                pending_write_bytes_per_step: 768,
                status_read_bytes_per_step: 768,
            }
        );
        assert_eq!(plan.memory(), original_memory);
        // The three-call traffic is per layer, while276B/548B belong to the one
        // model-shared checker and are deliberately not folded into DeltaMemory.
        assert_eq!(requirements.shared_checker.resident_tensor_bytes, 276);
        assert_eq!(requirements.shared_checker.initialization_write_bytes, 548);
        Ok(())
    }

    #[test]
    fn checked_traffic_overflow_and_zero_width_refuse_in_metadata() {
        assert!(finite_requirements(0, Some(native_finite_spec())).is_err());
        for spec in [
            FiniteCheckSpec {
                physical_dispatches: u64::MAX,
                ..native_finite_spec()
            },
            FiniteCheckSpec {
                logical_submissions: u64::MAX,
                ..native_finite_spec()
            },
            FiniteCheckSpec {
                pending_write_bytes: u64::MAX,
                ..native_finite_spec()
            },
            FiniteCheckSpec {
                status_read_bytes: u64::MAX,
                ..native_finite_spec()
            },
        ] {
            assert!(finite_requirements(5120, Some(spec)).is_err());
        }
    }

    #[test]
    fn checked_boundary_helper_refuses_foreign_owners_before_control() -> Result<()> {
        let cpu = CpuExecutionDevice::new();
        let other = CpuExecutionDevice::new();
        let input = cpu.upload_f32(&[1.0, -0.0])?;
        let foreign_input = other.upload_f32(&[1.0, -0.0])?;
        let mut checker = cpu.prepare_finite_check()?;
        let mut foreign_checker = other.prepare_finite_check()?;
        let mut calls = 0;
        assert!(
            check_resident_finite(&cpu, &mut foreign_checker, &input.full_view(), &mut || {
                calls += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, 0);
        assert!(
            check_resident_finite(&cpu, &mut checker, &foreign_input.full_view(), &mut || {
                calls += 1;
                Ok(())
            })
            .is_err()
        );
        assert_eq!(calls, 0);
        check_resident_finite(&cpu, &mut checker, &input.full_view(), &mut || Ok(()))?;
        Ok(())
    }

    #[test]
    fn checked_boundary_helper_preserves_refused_values_and_control_errors() -> Result<()> {
        // These tests exercise the public finite-capability helper on its explicit
        // CPU implementation. They neither emulate native Delta nor claim coverage
        // of the native wrapper's state transitions without a real/future gate.
        let cpu = CpuExecutionDevice::new();
        let mut checker = cpu.prepare_finite_check()?;
        let bits = [1.0f32.to_bits(), f32::NAN.to_bits(), (-0.0f32).to_bits()];
        let input = cpu.upload_f32(&bits.map(f32::from_bits))?;
        assert!(
            check_resident_finite(&cpu, &mut checker, &input.full_view(), &mut || Ok(())).is_err()
        );
        let mut actual = [0.0; 3];
        cpu.read_f32(&input.full_view(), &mut actual)?;
        assert_eq!(actual.map(f32::to_bits), bits);
        cpu.write_f32(&input.view(1, 1)?, &[2.0])?;
        for fail_at in 1..=3 {
            let mut calls = 0;
            let error = check_resident_finite(&cpu, &mut checker, &input.full_view(), &mut || {
                calls += 1;
                ensure!(
                    calls != fail_at,
                    "intentional checked-boundary cancellation"
                );
                Ok(())
            })
            .unwrap_err();
            assert_eq!(calls, fail_at);
            assert!(format!("{error:#}").contains("intentional checked-boundary cancellation"));
        }
        check_resident_finite(&cpu, &mut checker, &input.full_view(), &mut || Ok(()))?;
        Ok(())
    }

    #[test]
    fn strict_delta_header_and_capacity_admission_have_no_payload_reads() -> Result<()> {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let plan = RecordedDeltaPlan::new(&binding, 0, 16, ReadLimits::default())?;
        assert_eq!(plan.dim(), 256);
        assert_eq!(plan.capacity(), 16);
        assert_eq!(
            plan.shape(),
            NativeDeltaShape {
                key_heads: 1,
                value_heads: 2,
                head_dim: 128,
                epsilon: 1e-6
            }
        );
        assert_eq!(plan.layout().packet_bytes, 292_436);
        assert_eq!(plan.memory().resident_tensor_bytes, 294_484);
        assert_eq!(plan.memory().packed_tile_bytes, 13_440);
        assert_eq!(plan.memory().host_staging_bytes, 65_536);
        plan.admit(
            &NoRead(&gguf),
            limits(&plan),
            NativeExecutionProfile::Qwen35Probe,
        )?;
        for (layer, capacity) in [(3, 16), (4, 16), (0, 0), (0, 17)] {
            assert!(
                RecordedDeltaPlan::new(&binding, layer, capacity, ReadLimits::default()).is_err()
            );
        }
        assert!(RecordedDeltaPlan::new(
            &binding,
            0,
            1,
            ReadLimits {
                encoded_bytes: 8191,
                ..ReadLimits::default()
            }
        )
        .is_err());
        assert!(RecordedDeltaPlan::new(
            &binding,
            0,
            1,
            ReadLimits {
                decoded_bytes: 8191,
                ..ReadLimits::default()
            }
        )
        .is_err());
        assert!(RecordedDeltaPlan::new(
            &binding,
            0,
            1,
            ReadLimits {
                tile_rows: 0,
                ..ReadLimits::default()
            }
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn source_profile_and_each_payload_limit_refuse_without_a_device() -> Result<()> {
        let (_temporary, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let plan = RecordedDeltaPlan::new(&binding, 0, 1, ReadLimits::default())?;
        let source = NoRead(&gguf);
        let profile = NativeExecutionProfile::Qwen35Probe;
        assert!(plan
            .admit(&source, limits(&plan), NativeExecutionProfile::DenseQwen3)
            .is_err());
        let (_other_temporary, other) = Fixture::hybrid().open();
        assert!(plan.admit(&NoRead(&other), limits(&plan), profile).is_err());
        for low in [
            DeltaLimits {
                resident_tensor_bytes: plan.memory().resident_tensor_bytes - 1,
                ..limits(&plan)
            },
            DeltaLimits {
                packed_tile_bytes: plan.memory().packed_tile_bytes - 1,
                ..limits(&plan)
            },
            DeltaLimits {
                host_staging_bytes: plan.memory().host_staging_bytes - 1,
                ..limits(&plan)
            },
        ] {
            assert!(plan.admit(&source, low, profile).is_err());
        }
        Ok(())
    }

    #[test]
    fn openmythos_forecast_counts_guarded_packet_outside_norms_and_concurrent_parameters(
    ) -> Result<()> {
        let shape = NativeDeltaShape {
            key_heads: 16,
            value_heads: 48,
            head_dim: 128,
            epsilon: 1e-6,
        };
        let layout = shape.layout()?;
        let parameters = [163_840, 192, 192, 512];
        let memory = forecast(5120, layout, parameters, 268_800)?;
        assert_eq!(layout.packet_bytes, 6_899_548);
        assert_eq!(
            memory,
            DeltaMemory {
                resident_tensor_bytes: 6_940_508,
                packed_tile_bytes: 268_800,
                host_staging_bytes: 328_576
            }
        );
        assert!(forecast(5120, layout, [163_836, 192, 192, 512], 268_800).is_err());
        let larger_tile = forecast(5120, layout, parameters, 1_048_576)?;
        assert_eq!(larger_tile.host_staging_bytes, 1_048_576);
        assert_eq!(
            larger_tile.resident_tensor_bytes,
            memory.resident_tensor_bytes
        );
        Ok(())
    }
}
