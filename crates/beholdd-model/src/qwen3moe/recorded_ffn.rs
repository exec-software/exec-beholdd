//! A single dense Qwen3 FFN through the backend-neutral recorded execution seam.
//!
//! The component accepts normalized input; the block also records weighted RMSNorm
//! and residual addition. Neither includes attention, routing or serving registration.
//! Backend format/shape refusal propagates; no CPU arithmetic fallback occurs.

use anyhow::{bail, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, F32Tensor, F32TensorView, LinearMode, PreparedWeight,
    RecordedBatch,
};
use beholdd_backend::Weight;
use beholdd_format::WeightRange;
use beholdd_memory::{WeightAccess, WeightSource};

use super::{ggml_block_geometry, Bound, CheckpointId, FeedForward, QwenArchitecture, QwenModel};

/// Logical storage owned or temporarily staged by one prepared FFN.
///
/// These are separate payload counts, not a process or hardware memory forecast.
/// Backend allocation alignment, guards, programs, preparation expansion and execution
/// temporaries are excluded. So are the source's own caches and caller input/output
/// tensors. Callers must budget those independently before creating the device/source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnMemory {
    /// Sum of the three unchanged packed matrix payloads uploaded once.
    pub packed_weight_bytes: u64,
    /// Largest individual packed matrix read; preparation stages one matrix at a time.
    pub largest_staging_bytes: u64,
    /// Persistent gate and up activation payloads, each one FFN width of f32 values.
    pub scratch_bytes: u64,
}

/// Independent upper bounds on the logical payloads described by [`DenseFfnMemory`].
///
/// All three bounds are checked before the first source read or device allocation.
/// Passing them does not establish backend physical capacity or operator support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnLimits {
    /// Maximum sum of packed matrix payloads.
    pub packed_weight_bytes: u64,
    /// Maximum one-at-a-time host staging payload.
    pub staging_bytes: u64,
    /// Maximum persistent gate/up activation payload.
    pub scratch_bytes: u64,
}

#[derive(Debug, Clone)]
struct PlannedWeight {
    bound: Bound,
    range: WeightRange,
}

/// Metadata-only plan derived from the strict Qwen3 binder for one dense FFN layer.
#[derive(Debug, Clone)]
pub struct DenseFfnPlan {
    checkpoint_id: CheckpointId,
    dim: usize,
    feed_forward_length: usize,
    weights: [PlannedWeight; 3],
    memory: DenseFfnMemory,
}

/// Logical payloads for the normalized, residual-connected dense FFN block.
///
/// These counts exclude backend alignment, programs, guards, preparation expansion,
/// execution temporaries, source caches and caller-owned hidden storage. They are
/// component requirements, not a complete physical or process memory forecast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnBlockMemory {
    /// Three unchanged packed FFN matrices; the F32 norm is a resident tensor.
    pub packed_weight_bytes: u64,
    /// F32 norm, normalized input and persistent gate/up activations.
    pub resident_tensor_bytes: u64,
    /// Maximum matrix staging or simultaneous raw and decoded F32 norm payloads.
    pub largest_staging_bytes: u64,
}

/// Independent logical bounds checked before block preparation reads or allocates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseFfnBlockLimits {
    /// Maximum sum of the three encoded matrix payloads.
    pub packed_weight_bytes: u64,
    /// Maximum norm, normalized input and gate/up activation payloads.
    pub resident_tensor_bytes: u64,
    /// Maximum one-matrix or raw-plus-decoded norm staging payload.
    pub largest_staging_bytes: u64,
}

/// Metadata-only plan for `hidden += FFN(RMSNorm(hidden))` in one dense layer.
#[derive(Debug, Clone)]
pub struct DenseFfnBlockPlan {
    ffn: DenseFfnPlan,
    norm: PlannedWeight,
    eps: f32,
    memory: DenseFfnBlockMemory,
}

impl QwenModel {
    /// Plan gate/up/SwiGLU/down for one dense layer without reading tensor payloads.
    ///
    /// Only exact `qwen3` binding is accepted. The input boundary is the normalized
    /// FFN activation; output is the FFN contribution before residual addition.
    pub fn dense_ffn_plan(&self, layer_index: usize) -> Result<DenseFfnPlan> {
        if self.architecture != QwenArchitecture::Qwen3 {
            bail!(
                "recorded dense FFN requires qwen3, got {:?}",
                self.architecture
            );
        }
        let layer =
            self.core.layers.get(layer_index).with_context(|| {
                format!("dense FFN layer {layer_index} is outside the checkpoint")
            })?;
        let FeedForward::Dense { gate, up, down } = &layer.ffn else {
            bail!("layer {layer_index} is not a dense FFN");
        };
        let locate = |bound: &Bound| -> Result<PlannedWeight> {
            let (elements, bytes) = ggml_block_geometry(bound.tensor.kind().0)
                .context("dense FFN weight has unknown storage geometry")?;
            if !bound.cols.is_multiple_of(elements) {
                bail!("dense FFN weight row is not a whole number of storage blocks");
            }
            let encoded = bound
                .rows
                .checked_mul(bound.cols / elements)
                .and_then(|blocks| blocks.checked_mul(bytes))
                .context("dense FFN encoded matrix size overflows")?;
            let encoded = u64::try_from(encoded).context("dense FFN size does not fit u64")?;
            if encoded != bound.tensor.nbytes() {
                bail!("dense FFN tensor span does not match its encoded matrix size");
            }
            Ok(PlannedWeight {
                bound: bound.clone(),
                range: bound.tensor.locator().range(0..encoded)?,
            })
        };
        let weights = [locate(gate)?, locate(up)?, locate(down)?];
        let packed_weight_bytes = weights.iter().try_fold(0u64, |sum, weight| {
            sum.checked_add(weight.bound.tensor.nbytes())
                .context("dense FFN packed-weight total overflows")
        })?;
        let largest_staging_bytes = weights
            .iter()
            .map(|weight| weight.bound.tensor.nbytes())
            .max()
            .expect("dense FFN has three weights");
        let scratch_bytes = u64::try_from(self.core.cfg.expert_ff)
            .context("dense FFN width does not fit u64")?
            .checked_mul(2 * std::mem::size_of::<f32>() as u64)
            .context("dense FFN scratch size overflows")?;
        Ok(DenseFfnPlan {
            checkpoint_id: self.core.checkpoint_id,
            dim: self.core.cfg.dim,
            feed_forward_length: self.core.cfg.expert_ff,
            weights,
            memory: DenseFfnMemory {
                packed_weight_bytes,
                largest_staging_bytes,
                scratch_bytes,
            },
        })
    }

    /// Plan weighted RMSNorm, dense FFN and residual addition without payload reads.
    ///
    /// This first block target requires an F32 norm. Other norm formats are refused
    /// explicitly rather than converted or executed on the host during submission.
    pub fn dense_ffn_block_plan(&self, layer_index: usize) -> Result<DenseFfnBlockPlan> {
        let ffn = self.dense_ffn_plan(layer_index)?;
        let norm = &self.core.layers[layer_index].ffn_norm;
        if norm.tensor.kind().0 != 0 {
            bail!(
                "recorded dense FFN block requires F32 ffn_norm.weight, got {}",
                norm.tensor.kind().name()
            );
        }
        let norm_bytes = u64::try_from(ffn.dim)
            .context("dense FFN block width does not fit u64")?
            .checked_mul(std::mem::size_of::<f32>() as u64)
            .context("dense FFN block norm size overflows")?;
        if norm.rows != 1 || norm.cols != ffn.dim || norm.tensor.nbytes() != norm_bytes {
            bail!("dense FFN block norm span does not match its F32 vector shape");
        }
        let norm_pair_bytes = norm_bytes
            .checked_mul(2)
            .context("dense FFN block norm staging size overflows")?;
        let memory = DenseFfnBlockMemory {
            packed_weight_bytes: ffn.memory.packed_weight_bytes,
            resident_tensor_bytes: norm_pair_bytes
                .checked_add(ffn.memory.scratch_bytes)
                .context("dense FFN block resident tensor size overflows")?,
            largest_staging_bytes: ffn.memory.largest_staging_bytes.max(norm_pair_bytes),
        };
        Ok(DenseFfnBlockPlan {
            ffn,
            norm: PlannedWeight {
                bound: norm.clone(),
                range: norm.tensor.locator().range(0..norm_bytes)?,
            },
            eps: self.core.cfg.eps,
            memory,
        })
    }
}

impl DenseFfnPlan {
    /// Input and output width, each in f32 elements.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Width of each intermediate gate/up activation.
    pub fn feed_forward_length(&self) -> usize {
        self.feed_forward_length
    }

    /// Logical payload accounting; see [`DenseFfnMemory`] for excluded allocations.
    pub fn memory(&self) -> DenseFfnMemory {
        self.memory
    }

    /// Upload three exact packed weights and allocate persistent gate/up scratch.
    ///
    /// Source identity, every logical bound and every weight format are checked
    /// before reading payloads. Backend shape/operator limits can still refuse during
    /// upload or recording: the current execution trait has no metadata-only query
    /// for them. Each source staging handle is dropped before reading the next matrix;
    /// independently retained source caches are outside this plan's accounting.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: DenseFfnLimits,
    ) -> Result<PreparedDenseFfn> {
        self.check_preparation(device, source, limits)?;
        let upload = |weight: &PlannedWeight| -> Result<PreparedWeight> {
            let resident = source.load_range(&weight.range, WeightAccess::Cyclic)?;
            let encoded = device.upload_weight(&Weight {
                bytes: resident.bytes(),
                kind: weight.bound.tensor.kind(),
                rows: weight.bound.rows,
                cols: weight.bound.cols,
            })?;
            Ok(device.prepare_weight(&encoded)?)
        };
        let gate_weight = upload(&self.weights[0]).context("preparing dense FFN gate")?;
        let up_weight = upload(&self.weights[1]).context("preparing dense FFN up")?;
        let down_weight = upload(&self.weights[2]).context("preparing dense FFN down")?;
        Ok(PreparedDenseFfn {
            dim: self.dim,
            gate_weight,
            up_weight,
            down_weight,
            gate: device.allocate_f32(self.feed_forward_length)?,
            up: device.allocate_f32(self.feed_forward_length)?,
        })
    }

    fn check_preparation(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: DenseFfnLimits,
    ) -> Result<()> {
        if source.checkpoint_id() != self.checkpoint_id {
            bail!("dense FFN plan and weight source belong to different checkpoints");
        }
        for (label, required, limit) in [
            (
                "packed weights",
                self.memory.packed_weight_bytes,
                limits.packed_weight_bytes,
            ),
            (
                "staging",
                self.memory.largest_staging_bytes,
                limits.staging_bytes,
            ),
            ("scratch", self.memory.scratch_bytes, limits.scratch_bytes),
        ] {
            if required > limit {
                bail!("dense FFN {label} requires {required} logical bytes, limit is {limit}");
            }
        }
        for weight in &self.weights {
            if !device.supports(weight.bound.tensor.kind()) {
                bail!(
                    "{} cannot prepare dense FFN {} weight",
                    device.name(),
                    weight.bound.tensor.kind().name()
                );
            }
        }
        Ok(())
    }
}

impl DenseFfnBlockPlan {
    /// Width of caller hidden and the normalized input, in f32 elements.
    pub fn dim(&self) -> usize {
        self.ffn.dim()
    }

    /// Width of each persistent FFN gate/up activation.
    pub fn feed_forward_length(&self) -> usize {
        self.ffn.feed_forward_length()
    }

    /// Logical payload counts; see [`DenseFfnBlockMemory`] for exclusions.
    pub fn memory(&self) -> DenseFfnBlockMemory {
        self.memory
    }

    /// Prepare packed weights, the F32 norm and resident block scratch.
    ///
    /// Checkpoint identity, all logical bounds and all matrix formats are checked
    /// before I/O or device allocation. The norm's F32 format is checked when planning.
    /// Backend shape, RMSNorm support and physical capacity may still refuse at
    /// upload/recording. Norm preparation only decodes little-endian F32 storage once;
    /// normalization itself is a recorded device operation, never a CPU fallback.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: DenseFfnBlockLimits,
    ) -> Result<PreparedDenseFfnBlock> {
        if source.checkpoint_id() != self.ffn.checkpoint_id {
            bail!("dense FFN block plan and weight source belong to different checkpoints");
        }
        for (label, required, limit) in [
            (
                "packed weights",
                self.memory.packed_weight_bytes,
                limits.packed_weight_bytes,
            ),
            (
                "resident tensors",
                self.memory.resident_tensor_bytes,
                limits.resident_tensor_bytes,
            ),
            (
                "staging",
                self.memory.largest_staging_bytes,
                limits.largest_staging_bytes,
            ),
        ] {
            if required > limit {
                bail!(
                    "dense FFN block {label} requires {required} logical bytes, limit is {limit}"
                );
            }
        }
        let ffn_limits = DenseFfnLimits {
            packed_weight_bytes: limits.packed_weight_bytes,
            staging_bytes: limits.largest_staging_bytes,
            scratch_bytes: limits.resident_tensor_bytes,
        };
        let ffn = self.ffn.prepare(device, source, ffn_limits)?;
        let norm = {
            let resident = source.load_range(&self.norm.range, WeightAccess::Cyclic)?;
            if u64::try_from(resident.bytes().len()).ok() != Some(self.norm.range.nbytes()) {
                bail!("dense FFN block source returned an incorrect F32 norm length");
            }
            let mut values = Vec::new();
            values
                .try_reserve_exact(self.dim())
                .context("allocating dense FFN block F32 norm staging")?;
            values.extend(resident.bytes().chunks_exact(4).map(|bytes| {
                f32::from_le_bytes(bytes.try_into().expect("four-byte F32 storage chunk"))
            }));
            device.upload_f32(&values)?
        };
        Ok(PreparedDenseFfnBlock {
            ffn,
            norm,
            normalized: device.allocate_f32(self.dim())?,
            eps: self.eps,
        })
    }
}

/// Owned packed matrices and persistent scratch, ready to bind to caller tensors.
#[derive(Debug)]
pub struct PreparedDenseFfn {
    dim: usize,
    gate_weight: PreparedWeight,
    up_weight: PreparedWeight,
    down_weight: PreparedWeight,
    gate: F32Tensor,
    up: F32Tensor,
}

impl PreparedDenseFfn {
    /// Consume preparation and record one reusable four-command FFN batch.
    ///
    /// Input/output views must each contain exactly `plan.dim()` f32 elements and
    /// belong to the preparation device. The final linear operation overwrites output;
    /// it does not add a residual. The batch retains weights, scratch and boundary
    /// views, so neither source nor plan must stay alive after recording.
    ///
    /// Reuse one batch by writing input, submitting, waiting, then reading output.
    /// Wait before changing boundary tensors or submitting this batch again; cloned
    /// batches share scratch and must not execute concurrently. Consuming preparation
    /// prevents separately recorded batches from accidentally sharing its scratch.
    pub fn record(
        self,
        device: &dyn ExecutionDevice,
        input: &F32TensorView,
        output: &F32TensorView,
    ) -> Result<RecordedBatch> {
        let mut commands = CommandList::new();
        self.append(&mut commands, input, output, LinearMode::Overwrite)?;
        Ok(device.record(&commands)?)
    }

    fn append(
        &self,
        commands: &mut CommandList,
        input: &F32TensorView,
        output: &F32TensorView,
        mode: LinearMode,
    ) -> Result<()> {
        if input.len() != self.dim || output.len() != self.dim {
            bail!(
                "dense FFN input/output must each contain {} values",
                self.dim
            );
        }
        commands.linear(
            &self.gate.full_view(),
            input,
            &self.gate_weight,
            LinearMode::Overwrite,
        )?;
        commands.linear(
            &self.up.full_view(),
            input,
            &self.up_weight,
            LinearMode::Overwrite,
        )?;
        commands.swiglu(&self.gate.full_view(), &self.up.full_view(), 0.0)?;
        commands.linear(output, &self.gate.full_view(), &self.down_weight, mode)?;
        Ok(())
    }
}

/// Prepared norm, packed matrices and scratch for one residual-connected FFN block.
#[derive(Debug)]
pub struct PreparedDenseFfnBlock {
    ffn: PreparedDenseFfn,
    norm: F32Tensor,
    normalized: F32Tensor,
    eps: f32,
}

impl PreparedDenseFfnBlock {
    /// Consume preparation into five commands updating caller-owned hidden in place.
    ///
    /// Hidden must have exactly `plan.dim()` elements and share the preparation
    /// device. The batch retains every allocation after the plan/source are dropped.
    /// Wait before writing hidden or reusing the batch; clones share the same scratch.
    pub fn record(
        self,
        device: &dyn ExecutionDevice,
        hidden: &F32TensorView,
    ) -> Result<RecordedBatch> {
        if hidden.len() != self.ffn.dim {
            bail!(
                "dense FFN block hidden must contain {} values",
                self.ffn.dim
            );
        }
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &self.normalized.full_view(),
            hidden,
            Some(&self.norm.full_view()),
            self.eps,
        )?;
        self.ffn.append(
            &mut commands,
            &self.normalized.full_view(),
            hidden,
            LinearMode::ScaledAdd { scale: 1.0 },
        )?;
        Ok(device.record(&commands)?)
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
            tensors.push((format!("blk.0.{name}.weight"), vec![32, 32], 0));
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
            "beholdd-recorded-ffn-{}-{}.gguf",
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

    fn limits(plan: &DenseFfnPlan) -> DenseFfnLimits {
        let memory = plan.memory();
        DenseFfnLimits {
            packed_weight_bytes: memory.packed_weight_bytes,
            staging_bytes: memory.largest_staging_bytes,
            scratch_bytes: memory.scratch_bytes,
        }
    }

    fn block_limits(plan: &DenseFfnBlockPlan) -> DenseFfnBlockLimits {
        let memory = plan.memory();
        DenseFfnBlockLimits {
            packed_weight_bytes: memory.packed_weight_bytes,
            resident_tensor_bytes: memory.resident_tensor_bytes,
            largest_staging_bytes: memory.largest_staging_bytes,
        }
    }

    fn reference(plan: &DenseFfnPlan, checkpoint: &Gguf, input: &[f32]) -> Vec<f32> {
        let cpu = Cpu::new();
        let source = ResidentSource::new(checkpoint);
        let load = |index: usize| {
            ResidentWeight::load_range(checkpoint, &plan.weights[index].range).unwrap()
        };
        let resident = [load(0), load(1), load(2)];
        let weight = |index: usize| Weight {
            bytes: resident[index].bytes(),
            kind: plan.weights[index].bound.tensor.kind(),
            rows: plan.weights[index].bound.rows,
            cols: plan.weights[index].bound.cols,
        };
        assert_eq!(source.checkpoint_id(), plan.checkpoint_id);
        let mut gate = vec![0.0; plan.feed_forward_length()];
        let mut up = gate.clone();
        let mut output = vec![0.0; plan.dim()];
        cpu.matvec(&mut gate, input, &weight(0)).unwrap();
        cpu.matvec(&mut up, input, &weight(1)).unwrap();
        cpu.swiglu(&mut gate, &up, 0.0).unwrap();
        cpu.matvec(&mut output, &gate, &weight(2)).unwrap();
        output
    }

    fn block_reference(plan: &DenseFfnBlockPlan, checkpoint: &Gguf, hidden: &[f32]) -> Vec<f32> {
        let cpu = Cpu::new();
        let storage = ResidentWeight::load_range(checkpoint, &plan.norm.range).unwrap();
        let norm: Vec<f32> = storage
            .bytes()
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect();
        let mut normalized = vec![0.0; hidden.len()];
        cpu.rmsnorm(&mut normalized, hidden, Some(&norm), plan.eps)
            .unwrap();
        let contribution = reference(&plan.ffn, checkpoint, &normalized);
        let mut output = hidden.to_vec();
        cpu.add_in_place(&mut output, &contribution).unwrap();
        output
    }

    #[test]
    fn real_binder_recorded_ffn_matches_cpu_and_reuses_exact_weights() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_ffn_plan(0).unwrap();
        assert_eq!(plan.dim(), 32);
        assert_eq!(plan.feed_forward_length(), 64);
        assert_eq!(
            plan.memory(),
            DenseFfnMemory {
                packed_weight_bytes: 4096 + 2176 + 2176,
                largest_staging_bytes: 4096,
                scratch_bytes: 512,
            }
        );
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let prepared = plan.prepare(&device, &source, limits(&plan)).unwrap();
        assert_eq!(
            source.reads.lock().unwrap().as_slice(),
            &plan.weights.each_ref().map(|weight| weight.range)
        );
        let inputs: Vec<Vec<f32>> = (0..2)
            .map(|step| {
                (0..32)
                    .map(|index| ((index * 7 + step * 3) % 13) as f32 * 0.125 - 0.75)
                    .collect()
            })
            .collect();
        let expected: Vec<_> = inputs
            .iter()
            .map(|input| reference(&plan, &fixture.checkpoint, input))
            .collect();
        assert_ne!(expected[0], expected[1]);
        let input = device.allocate_f32(32).unwrap();
        let output = device.allocate_f32(32).unwrap();
        let batch = prepared
            .record(&device, &input.full_view(), &output.full_view())
            .unwrap();
        drop(plan);
        drop(model);
        for step in [0, 1, 0] {
            device.write_f32(&input.full_view(), &inputs[step]).unwrap();
            device.write_f32(&output.full_view(), &[123.0; 32]).unwrap();
            let submission = device.submit(&batch).unwrap();
            device.wait(&submission).unwrap();
            let mut actual = vec![0.0; 32];
            device.read_f32(&output.full_view(), &mut actual).unwrap();
            assert_eq!(actual, expected[step]);
            assert_eq!(source.reads.lock().unwrap().len(), 3);
        }
        drop(source);
        drop(fixture);
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        let mut actual = vec![0.0; 32];
        device.read_f32(&output.full_view(), &mut actual).unwrap();
        assert_eq!(
            actual, expected[0],
            "batch owns weights after source/checkpoint are dropped"
        );
    }

    #[test]
    fn logical_budget_and_source_identity_refuse_before_reads() {
        let fixture = make_fixture(true, 8);
        let other = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_ffn_plan(0).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let foreign_source = CountingSource::new(&other.checkpoint);
        let device = CpuExecutionDevice::new();
        let valid = limits(&plan);
        for refused in [
            DenseFfnLimits {
                packed_weight_bytes: valid.packed_weight_bytes - 1,
                ..valid
            },
            DenseFfnLimits {
                staging_bytes: valid.staging_bytes - 1,
                ..valid
            },
            DenseFfnLimits {
                scratch_bytes: valid.scratch_bytes - 1,
                ..valid
            },
        ] {
            assert!(plan.prepare(&device, &source, refused).is_err());
        }
        assert!(plan.prepare(&device, &foreign_source, valid).is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(foreign_source.reads.lock().unwrap().is_empty());
        assert!(model.dense_ffn_plan(1).is_err());
    }

    #[test]
    fn routed_architecture_and_unsupported_format_are_refused() {
        let routed = make_fixture(false, 8);
        let model = QwenModel::load(&routed.checkpoint).unwrap();
        assert!(model.dense_ffn_plan(0).is_err());
        let unsupported = make_fixture(true, 2);
        let model = QwenModel::load(&unsupported.checkpoint).unwrap();
        let plan = model.dense_ffn_plan(0).unwrap();
        let source = CountingSource::new(&unsupported.checkpoint);
        assert!(plan
            .prepare(&CpuExecutionDevice::new(), &source, limits(&plan))
            .is_err());
        assert!(source.reads.lock().unwrap().is_empty());
    }

    #[test]
    fn recording_refuses_wrong_shapes_and_foreign_device_resources() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_ffn_plan(0).unwrap();
        let source = ResidentSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let other = CpuExecutionDevice::new();
        let input = device.allocate_f32(32).unwrap();
        let output = device.allocate_f32(32).unwrap();
        for (input, output) in [
            (input.view(0, 31).unwrap(), output.full_view()),
            (input.full_view(), output.view(0, 31).unwrap()),
            (
                other.allocate_f32(32).unwrap().full_view(),
                output.full_view(),
            ),
            (
                input.full_view(),
                other.allocate_f32(32).unwrap().full_view(),
            ),
        ] {
            let prepared = plan.prepare(&device, &source, limits(&plan)).unwrap();
            assert!(prepared.record(&device, &input, &output).is_err());
        }
        let prepared = plan.prepare(&device, &source, limits(&plan)).unwrap();
        assert!(prepared
            .record(&other, &input.full_view(), &output.full_view())
            .is_err());
    }

    #[test]
    fn dense_ffn_block_matches_normalized_cpu_residual_and_replays_owned_resources() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_ffn_block_plan(0).unwrap();
        assert_eq!(plan.dim(), 32);
        assert_eq!(plan.feed_forward_length(), 64);
        assert_eq!(
            plan.memory(),
            DenseFfnBlockMemory {
                packed_weight_bytes: 4096 + 2176 + 2176,
                resident_tensor_bytes: 32 * 4 * 2 + 64 * 4 * 2,
                largest_staging_bytes: 4096,
            }
        );
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let prepared = plan.prepare(&device, &source, block_limits(&plan)).unwrap();
        let expected_ranges = [
            plan.ffn.weights[0].range,
            plan.ffn.weights[1].range,
            plan.ffn.weights[2].range,
            plan.norm.range,
        ];
        assert_eq!(source.reads.lock().unwrap().as_slice(), &expected_ranges);
        let inputs: Vec<Vec<f32>> = (0..2)
            .map(|step| {
                (0..32)
                    .map(|index| ((index * 7 + step * 3) % 13) as f32 * 0.125 - 0.75)
                    .collect()
            })
            .collect();
        let expected: Vec<_> = inputs
            .iter()
            .map(|hidden| block_reference(&plan, &fixture.checkpoint, hidden))
            .collect();
        assert_ne!(expected[0], expected[1]);
        assert_ne!(
            expected[0], inputs[0],
            "fixture must exercise the FFN contribution"
        );
        let storage = device.upload_f32(&[123.0; 40]).unwrap();
        let hidden = storage.view(4, 32).unwrap();
        let batch = prepared.record(&device, &hidden).unwrap();
        assert_eq!(batch.len(), 5);
        drop(plan);
        drop(model);
        for step in [0, 1, 0] {
            device.write_f32(&hidden, &inputs[step]).unwrap();
            device.wait(&device.submit(&batch).unwrap()).unwrap();
            let mut actual = vec![0.0; 40];
            device.read_f32(&storage.full_view(), &mut actual).unwrap();
            assert_eq!(&actual[..4], &[123.0; 4]);
            assert_eq!(&actual[36..], &[123.0; 4]);
            for (actual, expected) in actual[4..36].iter().zip(&expected[step]) {
                assert!(actual.is_finite());
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
            assert_eq!(source.reads.lock().unwrap().len(), 4);
        }
        drop(source);
        drop(fixture);
        device.write_f32(&hidden, &inputs[1]).unwrap();
        device.wait(&device.submit(&batch).unwrap()).unwrap();
        let mut actual = vec![0.0; 32];
        device.read_f32(&hidden, &mut actual).unwrap();
        assert_eq!(
            actual, expected[1],
            "batch retains norm, weights and scratch"
        );
    }

    #[test]
    fn dense_ffn_block_limits_and_source_identity_refuse_before_reads() {
        let fixture = make_fixture(true, 8);
        let other_fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_ffn_block_plan(0).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let foreign = CountingSource::new(&other_fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let valid = block_limits(&plan);
        for refused in [
            DenseFfnBlockLimits {
                packed_weight_bytes: valid.packed_weight_bytes - 1,
                ..valid
            },
            DenseFfnBlockLimits {
                resident_tensor_bytes: valid.resident_tensor_bytes - 1,
                ..valid
            },
            DenseFfnBlockLimits {
                largest_staging_bytes: valid.largest_staging_bytes - 1,
                ..valid
            },
        ] {
            assert!(plan.prepare(&device, &source, refused).is_err());
        }
        assert!(plan.prepare(&device, &foreign, valid).is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(foreign.reads.lock().unwrap().is_empty());
        assert!(model.dense_ffn_block_plan(1).is_err());
    }

    #[test]
    fn dense_ffn_block_refuses_non_f32_norm_routed_model_and_unsupported_matrix() {
        let device = CpuExecutionDevice::new();
        for norm_kind in [30, 8] {
            let fixture = make_fixture_with_norm(true, 8, norm_kind);
            let source = CountingSource::new(&fixture.checkpoint);
            let model = QwenModel::load(&fixture.checkpoint).unwrap();
            let error = model.dense_ffn_block_plan(0).unwrap_err();
            assert!(error.to_string().contains("requires F32 ffn_norm.weight"));
            assert!(source.reads.lock().unwrap().is_empty());
        }
        let routed = make_fixture(false, 8);
        let model = QwenModel::load(&routed.checkpoint).unwrap();
        assert!(model.dense_ffn_block_plan(0).is_err());
        let unsupported = make_fixture(true, 2);
        let source = CountingSource::new(&unsupported.checkpoint);
        let model = QwenModel::load(&unsupported.checkpoint).unwrap();
        let plan = model.dense_ffn_block_plan(0).unwrap();
        assert!(plan.prepare(&device, &source, block_limits(&plan)).is_err());
        assert!(source.reads.lock().unwrap().is_empty());
    }

    #[test]
    fn dense_ffn_block_recording_refuses_wrong_shape_and_foreign_owners() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_ffn_block_plan(0).unwrap();
        let source = ResidentSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let other = CpuExecutionDevice::new();
        let hidden = device.allocate_f32(32).unwrap();
        let foreign = other.allocate_f32(32).unwrap();
        for view in [hidden.view(0, 31).unwrap(), foreign.full_view()] {
            let prepared = plan.prepare(&device, &source, block_limits(&plan)).unwrap();
            assert!(prepared.record(&device, &view).is_err());
        }
        let prepared = plan.prepare(&device, &source, block_limits(&plan)).unwrap();
        assert!(prepared.record(&other, &hidden.full_view()).is_err());
    }
}
