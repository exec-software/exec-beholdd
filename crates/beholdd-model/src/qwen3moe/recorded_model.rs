//! Complete resident dense Qwen3 execution with private hidden, logits and KV state.
//!
//! Metadata and aggregate logical limits are checked before payload I/O. Preparation
//! stages one component weight at a time; all source tensors are then retained by
//! device resources. A position becomes visible only after embedding, every layer
//! and the final head have completed. Any execution error or unwind poisons the
//! entire preparation, even if an individual layer already advanced its private KV.

use super::{
    CheckpointId, DenseAttentionLimits, DenseAttentionPlan, DenseFfnBlockLimits, DenseFfnBlockPlan,
    FeedForward, PreparedVocabulary, QwenArchitecture, QwenModel, ResidentDenseAttention,
    VocabularyLimits, VocabularyPlan, MAX_ON_DEMAND_CAPACITY,
};
use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{ExecutionDevice, F32Tensor, RecordedBatch};
use beholdd_format::GgmlType;
use beholdd_memory::WeightSource;
use std::cell::Cell;

/// Checked logical payload totals for the complete model, including private outputs.
/// Backend expansion, private command scratch, alignment, guards, programs and
/// independently retained source caches are excluded and require separate budgets.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseModelMemory {
    /// All packed layer matrices and the single shared embedding/head vocabulary.
    pub packed_weight_bytes: u64,
    /// All norms, activations, fixed caches, coefficients, private hidden and logits.
    pub resident_tensor_bytes: u64,
    /// Largest model-owned source, norm-decode or coefficient staging payload.
    /// Backend upload conversion buffers require a separate budget.
    pub largest_staging_bytes: u64,
}

/// Independent aggregate limits enforced before source I/O or device allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DenseModelLimits {
    /// Maximum total packed matrix payload.
    pub packed_weight_bytes: u64,
    /// Maximum persistent F32 payload including private hidden and logits.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time host staging payload.
    pub largest_staging_bytes: u64,
}

#[derive(Clone, Debug)]
struct LayerPlan {
    attention: DenseAttentionPlan,
    ffn: DenseFfnBlockPlan,
}

/// Metadata-only composition of every dense layer and one tied vocabulary.
#[derive(Clone, Debug)]
pub struct DenseModelPlan {
    checkpoint_id: CheckpointId,
    capacity: usize,
    on_demand_only: bool,
    layers: Vec<LayerPlan>,
    vocabulary: VocabularyPlan,
    memory: DenseModelMemory,
}

impl QwenModel {
    /// Plan complete Q8_0/F32 dense Qwen3 execution without payload reads.
    ///
    /// The first resident target permits 1..=128 layers, at most 4096 total
    /// layer-position batches, cache capacity 1..=256 within context, and hidden/FFN
    /// widths at most 4096. Attention and vocabulary retain their stricter checked
    /// geometry and tied-storage requirements. All matrix formats are fixed here
    /// before preparation can perform I/O for an earlier layer.
    pub fn dense_model_plan(&self, capacity: usize, tile_rows: usize) -> Result<DenseModelPlan> {
        self.dense_model_plan_with_recording_mode(capacity, tile_rows, false)
    }

    /// Plan complete Q8_0/F32 dense execution with on-demand attention only.
    /// Capacity may be 1..=16384 within checkpoint context; the layer-position table
    /// bound does not apply because at most one attention batch exists at a time.
    /// The layer count and all matrix/geometry checks remain unchanged. This is a
    /// metadata plan, not a backend-support or numerical-qualification guarantee.
    pub fn dense_model_plan_on_demand(
        &self,
        capacity: usize,
        tile_rows: usize,
    ) -> Result<DenseModelPlan> {
        self.dense_model_plan_with_recording_mode(capacity, tile_rows, true)
    }

    fn dense_model_plan_with_recording_mode(
        &self,
        capacity: usize,
        tile_rows: usize,
        on_demand_only: bool,
    ) -> Result<DenseModelPlan> {
        ensure!(
            self.architecture == QwenArchitecture::Qwen3,
            "resident dense model requires exact qwen3 architecture"
        );
        let cfg = &self.core.cfg;
        ensure!(
            (1..=128).contains(&cfg.n_layers) && self.core.layers.len() == cfg.n_layers,
            "resident model requires 1..=128 complete layers"
        );
        let maximum = if on_demand_only {
            MAX_ON_DEMAND_CAPACITY
        } else {
            256
        };
        ensure!(
            (1..=maximum).contains(&capacity) && capacity <= cfg.context,
            "resident model capacity must be in 1..={maximum} and within context"
        );
        if !on_demand_only {
            ensure!(
                cfg.n_layers
                    .checked_mul(capacity)
                    .is_some_and(|n| n <= 4096),
                "resident model metadata exceeds 4096 layer-position batches"
            );
        }
        ensure!(
            (1..=4096).contains(&cfg.dim)
                && cfg.dim.is_multiple_of(32)
                && (1..=4096).contains(&cfg.expert_ff)
                && cfg.expert_ff.is_multiple_of(32),
            "resident model requires complete Q8_0 hidden/FFN widths at most 4096"
        );
        let vocabulary = self.vocabulary_plan(tile_rows)?;
        let mut memory = DenseModelMemory {
            packed_weight_bytes: vocabulary.memory().packed_weight_bytes,
            resident_tensor_bytes: vocabulary.memory().resident_tensor_bytes,
            largest_staging_bytes: vocabulary.memory().largest_staging_bytes,
        };
        let private_bytes = u64::try_from(
            cfg.dim
                .checked_add(cfg.vocab)
                .context("private model elements overflow")?,
        )?
        .checked_mul(4)
        .context("private model tensor bytes overflow")?;
        memory.add(0, private_bytes, 0)?;
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(cfg.n_layers)
            .context("allocating model plan metadata")?;
        for (index, layer) in self.core.layers.iter().enumerate() {
            let FeedForward::Dense { gate, up, down } = &layer.ffn else {
                anyhow::bail!("resident model layer {index} is not dense");
            };
            for (weight, rows, cols) in [
                (gate, cfg.expert_ff, cfg.dim),
                (up, cfg.expert_ff, cfg.dim),
                (down, cfg.dim, cfg.expert_ff),
            ] {
                ensure!(
                    weight.tensor.kind().0 == 8 && weight.rows == rows && weight.cols == cols,
                    "resident model layer {index} requires exact Q8_0 FFN matrix shapes"
                );
            }
            let attention = self.dense_attention_plan(index)?;
            let ffn = self.dense_ffn_block_plan(index)?;
            let a = if on_demand_only {
                attention.memory_on_demand(capacity)?
            } else {
                attention.memory(capacity)?
            };
            let f = ffn.memory();
            memory.add(
                a.packed_weight_bytes,
                a.resident_tensor_bytes,
                a.largest_staging_bytes,
            )?;
            memory.add(
                f.packed_weight_bytes,
                f.resident_tensor_bytes,
                f.largest_staging_bytes,
            )?;
            layers.push(LayerPlan { attention, ffn });
        }
        Ok(DenseModelPlan {
            checkpoint_id: self.core.checkpoint_id,
            capacity,
            on_demand_only,
            layers,
            vocabulary,
            memory,
        })
    }
}

impl DenseModelMemory {
    fn add(&mut self, packed: u64, tensors: u64, staging: u64) -> Result<()> {
        let packed = self
            .packed_weight_bytes
            .checked_add(packed)
            .context("model packed total overflows")?;
        let tensors = self
            .resident_tensor_bytes
            .checked_add(tensors)
            .context("model tensor total overflows")?;
        self.packed_weight_bytes = packed;
        self.resident_tensor_bytes = tensors;
        self.largest_staging_bytes = self.largest_staging_bytes.max(staging);
        Ok(())
    }
}

impl DenseModelPlan {
    /// Width of the private residual stream in F32 elements.
    pub fn dim(&self) -> usize {
        self.vocabulary.dim()
    }
    /// Number of complete logits produced by every step.
    pub fn vocab(&self) -> usize {
        self.vocabulary.vocab()
    }
    /// Number of ordered attention/FFN layer pairs.
    pub fn layers(&self) -> usize {
        self.layers.len()
    }
    /// Fixed maximum positions in one request.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Whether this plan must use on-demand attention preparation, even when its
    /// capacity would fit the legacy pre-recorded path.
    pub fn requires_on_demand_attention(&self) -> bool {
        self.on_demand_only
    }
    /// Aggregate logical payload requirements, before backend-specific overhead.
    pub fn memory(&self) -> DenseModelMemory {
        self.memory
    }

    /// Prepare all matrices once and record all persistent layer/head batches.
    ///
    /// Source identity, every aggregate limit and Q8_0 backend support are checked
    /// before the first allocation/read. Physical backend capacity or command
    /// support can still refuse during preparation; there is no execution fallback.
    /// Only one packed source weight is staged at a time. Hidden, logits and caches
    /// remain private, and the returned owner deliberately cannot be cloned.
    /// Plans created by `dense_model_plan_on_demand` refuse before allocation/I/O.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: DenseModelLimits,
    ) -> Result<ResidentDenseModel> {
        self.prepare_with_attention_recording(device, source, limits, false)
    }

    /// Prepare the same persistent payloads and FFN/head batches, retaining the
    /// attention operands instead of one batch per cache position. Every step
    /// records, executes and retires one attention batch at a time. This changes
    /// backend auxiliary/host metadata and recording traffic, not logical payload
    /// totals, source reads or arithmetic. It accepts both legacy plans and plans
    /// from `dense_model_plan_on_demand`, up to the latter's capacity limit of 16384.
    /// Backend-specific attention recording refusals occur at step time and poison
    /// the resident owner; callers must budget transient recording separately.
    pub fn prepare_on_demand_attention(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: DenseModelLimits,
    ) -> Result<ResidentDenseModel> {
        self.prepare_with_attention_recording(device, source, limits, true)
    }

    /// Extra owned model-side heap storage retained by on-demand attention.
    /// Add this to the resident/plan tables and backend recording estimate; no
    /// device access or tensor payload reads occur. See the per-layer cost query.
    pub fn on_demand_attention_host_metadata_bytes(&self) -> Result<u64> {
        self.layers.iter().try_fold(0_u64, |sum, layer| {
            sum.checked_add(layer.attention.on_demand_host_metadata_bytes()?)
                .context("model on-demand attention host metadata overflows")
        })
    }

    fn prepare_with_attention_recording(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: DenseModelLimits,
        on_demand: bool,
    ) -> Result<ResidentDenseModel> {
        ensure!(
            on_demand || !self.on_demand_only,
            "on-demand model plan requires prepare_on_demand_attention"
        );
        ensure!(
            source.checkpoint_id() == self.checkpoint_id,
            "model plan and source belong to different checkpoints"
        );
        for (label, required, limit) in [
            (
                "packed",
                self.memory.packed_weight_bytes,
                limits.packed_weight_bytes,
            ),
            (
                "tensors",
                self.memory.resident_tensor_bytes,
                limits.resident_tensor_bytes,
            ),
            (
                "staging",
                self.memory.largest_staging_bytes,
                limits.largest_staging_bytes,
            ),
        ] {
            ensure!(
                required <= limit,
                "model {label} needs {required} logical bytes, limit is {limit}"
            );
        }
        ensure!(
            device.supports(GgmlType(8)),
            "backend refuses model Q8_0 format"
        );
        let mut layers = Vec::new();
        layers
            .try_reserve_exact(self.layers.len())
            .context("allocating resident layer handles")?;
        let hidden = device.allocate_f32(self.dim())?;
        let logits = device.allocate_f32(self.vocab())?;
        let v = self.vocabulary.memory();
        let vocabulary = self.vocabulary.prepare(
            device,
            source,
            VocabularyLimits {
                packed_weight_bytes: v.packed_weight_bytes,
                resident_tensor_bytes: v.resident_tensor_bytes,
                largest_staging_bytes: v.largest_staging_bytes,
            },
        )?;
        let head = vocabulary.record_logits(device, &hidden.full_view(), &logits.full_view())?;
        for layer in &self.layers {
            let a = if on_demand {
                layer.attention.memory_on_demand(self.capacity)?
            } else {
                layer.attention.memory(self.capacity)?
            };
            let limits = DenseAttentionLimits {
                packed_weight_bytes: a.packed_weight_bytes,
                resident_tensor_bytes: a.resident_tensor_bytes,
                largest_staging_bytes: a.largest_staging_bytes,
            };
            let attention = if on_demand {
                layer
                    .attention
                    .prepare_on_demand(device, source, self.capacity, limits)?
            } else {
                layer
                    .attention
                    .prepare(device, source, self.capacity, limits)?
            };
            let attention = if on_demand {
                attention.record_on_demand(&hidden.full_view())?
            } else {
                attention.record(device, &hidden.full_view())?
            };
            let f = layer.ffn.memory();
            let ffn = layer
                .ffn
                .prepare(
                    device,
                    source,
                    DenseFfnBlockLimits {
                        packed_weight_bytes: f.packed_weight_bytes,
                        resident_tensor_bytes: f.resident_tensor_bytes,
                        largest_staging_bytes: f.largest_staging_bytes,
                    },
                )?
                .record(device, &hidden.full_view())?;
            layers.push(ResidentLayer { attention, ffn });
        }
        Ok(ResidentDenseModel {
            hidden,
            logits,
            vocabulary,
            head,
            layers,
            capacity: self.capacity,
            visible: 0,
            poisoned: Cell::new(false),
        })
    }
}

#[derive(Debug)]
struct ResidentLayer {
    attention: ResidentDenseAttention,
    ffn: RecordedBatch,
}

/// Sequential complete-model owner with private outputs and fixed-capacity KV.
/// A failed execution/read or unwind permanently poisons this owner; create a new
/// preparation to recover. Only successful complete steps publish positions.
#[derive(Debug)]
pub struct ResidentDenseModel {
    hidden: F32Tensor,
    logits: F32Tensor,
    vocabulary: PreparedVocabulary,
    head: RecordedBatch,
    layers: Vec<ResidentLayer>,
    capacity: usize,
    visible: usize,
    poisoned: Cell<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CompletedStage {
    Embedding,
    Attention(usize),
    Ffn(usize),
    Head,
}

impl ResidentDenseModel {
    /// Whether every layer records and retires attention at each step.
    pub fn records_attention_on_demand(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| layer.attention.records_on_demand())
    }

    /// Positions whose complete model execution, including logits, has finished.
    pub fn visible_positions(&self) -> usize {
        self.visible
    }
    /// Fixed maximum number of positions in one request.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    /// Whether an error/unwind made the private state unusable, including for reset.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.get()
    }

    fn validate_state(&self) -> Result<()> {
        ensure!(
            !self.poisoned.get(),
            "resident model is poisoned; create a new preparation"
        );
        if self.visible > self.capacity
            || self.layers.iter().any(|layer| {
                layer.attention.is_poisoned()
                    || layer.attention.visible_positions() != self.visible
                    || layer.attention.capacity() != self.capacity
            })
        {
            self.poisoned.set(true);
            anyhow::bail!("resident model cache states differ; preparation is poisoned");
        }
        Ok(())
    }

    /// Execute one embedding, every layer and the full normalized head in order.
    /// Invalid tokens/full capacity refuse before any recording or submission and
    /// leave a healthy owner reusable. Execution errors and unwinds never publish
    /// a global position, and permanently prohibit reads, further steps and reset.
    pub fn step(&mut self, device: &dyn ExecutionDevice, token: u32) -> Result<()> {
        self.step_with(device, token, |_| Ok(()))
    }

    // Tests inject errors/unwinds after real CPU submissions using the same state
    // transition as production, without weakening the sealed device interface.
    fn step_with(
        &mut self,
        device: &dyn ExecutionDevice,
        token: u32,
        mut completed: impl FnMut(CompletedStage) -> Result<()>,
    ) -> Result<()> {
        self.validate_state()?;
        let token = usize::try_from(token).context("token does not fit usize")?;
        ensure!(
            token < self.logits.len(),
            "token is outside resident vocabulary"
        );
        ensure!(self.visible < self.capacity, "resident model cache is full");
        self.poisoned.set(true);
        {
            let embedding =
                self.vocabulary
                    .record_embedding(device, token, &self.hidden.full_view())?;
            let submission = device.submit(&embedding)?;
            device.wait(&submission)?;
        } // Retire the transient lookup batch/dummy before executing any layer.
        completed(CompletedStage::Embedding)?;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.attention.step(device)?;
            completed(CompletedStage::Attention(index))?;
            let submission = device.submit(&layer.ffn)?;
            device.wait(&submission)?;
            completed(CompletedStage::Ffn(index))?;
        }
        let submission = device.submit(&self.head)?;
        device.wait(&submission)?;
        completed(CompletedStage::Head)?;
        let next = self
            .visible
            .checked_add(1)
            .context("model visible count overflows")?;
        ensure!(
            self.layers
                .iter()
                .all(|layer| !layer.attention.is_poisoned()
                    && layer.attention.visible_positions() == next),
            "model caches did not complete the same position"
        );
        self.visible = next;
        self.poisoned.set(false);
        Ok(())
    }

    /// Read the most recently completed full logits into an exact-length buffer.
    /// Refuses stale output before the first step, after reset, or after poison.
    /// A device read failure/unwind poisons the owner; a shape refusal does not.
    pub fn read_logits(&self, device: &dyn ExecutionDevice, output: &mut [f32]) -> Result<()> {
        self.validate_state()?;
        ensure!(self.visible > 0, "resident model has no completed logits");
        ensure!(
            output.len() == self.logits.len(),
            "logits output length differs from vocabulary"
        );
        self.poisoned.set(true);
        device.read_f32(&self.logits.full_view(), output)?;
        self.poisoned.set(false);
        Ok(())
    }

    /// Start a fresh request with no allocations or device I/O.
    /// All caches are checked before any count changes; newly visible rows are
    /// overwritten by subsequent steps. Previous logits become unavailable.
    pub fn reset(&mut self) -> Result<()> {
        self.validate_state()?;
        self.poisoned.set(true);
        for layer in &mut self.layers {
            layer.attention.reset()?;
        }
        self.visible = 0;
        self.poisoned.set(false);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beholdd_backend::{cpu::Cpu, execution::CpuExecutionDevice};
    use beholdd_format::{Gguf, WeightRange};
    use beholdd_memory::{ResidentSource, ResidentWeight, WeightAccess};
    use std::path::PathBuf;
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    };

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

    // Two distinct complete layers, Q8_0 matrices, F32 norms and a tied nine-row
    // vocabulary exercise production GGUF binding without a large allocation.
    fn fixture(last_gate_kind: u32) -> Fixture {
        fixture_with_context(last_gate_kind, 16)
    }

    fn fixture_with_context(last_gate_kind: u32, context: u32) -> Fixture {
        let mut tensors: Vec<(String, Vec<u64>, u32)> = vec![
            ("token_embd.weight".into(), vec![32, 9], 8),
            ("output_norm.weight".into(), vec![32], 0),
        ];
        for layer in 0..2 {
            for name in ["attn_norm", "attn_q_norm", "attn_k_norm", "ffn_norm"] {
                tensors.push((format!("blk.{layer}.{name}.weight"), vec![32], 0));
            }
            for name in ["attn_q", "attn_k", "attn_v", "attn_output"] {
                tensors.push((format!("blk.{layer}.{name}.weight"), vec![32, 32], 8));
            }
            tensors.extend([
                (
                    format!("blk.{layer}.ffn_gate.weight"),
                    vec![32, 64],
                    if layer == 1 { last_gate_kind } else { 8 },
                ),
                (format!("blk.{layer}.ffn_up.weight"), vec![32, 64], 8),
                (format!("blk.{layer}.ffn_down.weight"), vec![64, 32], 8),
            ]);
        }
        let metadata = [
            ("block_count", 2u32),
            ("embedding_length", 32),
            ("attention.head_count", 1),
            ("attention.head_count_kv", 1),
            ("attention.key_length", 32),
            ("attention.value_length", 32),
            ("feed_forward_length", 64),
            ("context_length", context),
        ];
        let mut header = b"GGUF".to_vec();
        header.extend(3u32.to_le_bytes());
        header.extend((tensors.len() as u64).to_le_bytes());
        header.extend((metadata.len() as u64 + 3).to_le_bytes());
        string(&mut header, "general.architecture");
        header.extend(8u32.to_le_bytes());
        string(&mut header, "qwen3");
        for (key, value) in metadata {
            string(&mut header, &format!("qwen3.{key}"));
            header.extend(4u32.to_le_bytes());
            header.extend(value.to_le_bytes());
        }
        for (key, value) in [
            ("attention.layer_norm_rms_epsilon", 1e-6f32),
            ("rope.freq_base", 10000.0),
        ] {
            string(&mut header, &format!("qwen3.{key}"));
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
            match kind {
                0 | 30 => {
                    for index in 0..elements {
                        let value = 0.5 + ((index + tensor_index) % 5) as f32 * 0.125;
                        if *kind == 0 {
                            payload.extend(value.to_le_bytes());
                        } else {
                            payload.extend(((value.to_bits() >> 16) as u16).to_le_bytes());
                        }
                    }
                }
                8 => {
                    for block in 0..elements / 32 {
                        payload.extend(0x2800u16.to_le_bytes()); // scale = 1/32
                        payload.extend(
                            (0..32).map(|lane| {
                                (((block * 3 + lane + tensor_index) % 7) as i8 - 3) as u8
                            }),
                        );
                    }
                }
                _ => panic!("unsupported test storage"),
            }
            payload.resize(payload.len().next_multiple_of(32), 0);
        }
        header.resize(header.len().next_multiple_of(32), 0);
        header.extend(payload);
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "beholdd-resident-model-{}-{}.gguf",
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
    fn limits(plan: &DenseModelPlan) -> DenseModelLimits {
        let memory = plan.memory();
        DenseModelLimits {
            packed_weight_bytes: memory.packed_weight_bytes,
            resident_tensor_bytes: memory.resident_tensor_bytes,
            largest_staging_bytes: memory.largest_staging_bytes,
        }
    }

    #[test]
    fn aggregate_memory_counts_shared_vocabulary_private_outputs_and_one_weight_staging() {
        let fixture = fixture(8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_model_plan(3, 4).unwrap();
        assert_eq!(
            (plan.dim(), plan.vocab(), plan.layers(), plan.capacity()),
            (32, 9, 2, 3)
        );
        assert_eq!(
            plan.memory(),
            DenseModelMemory {
                packed_weight_bytes: 22066,
                resident_tensor_bytes: 6308,
                largest_staging_bytes: 2176
            }
        );
        let extra_state = plan.on_demand_attention_host_metadata_bytes().unwrap();
        assert_eq!(
            extra_state,
            model
                .dense_attention_plan(0)
                .unwrap()
                .on_demand_host_metadata_bytes()
                .unwrap()
                * 2
        );
        assert!(extra_state > 0);
        assert_eq!(
            model
                .dense_model_plan(16, 4)
                .unwrap()
                .on_demand_attention_host_metadata_bytes()
                .unwrap(),
            extra_state,
            "retained on-demand state is independent of cache capacity",
        );
        for (capacity, expected_tensors, expected_staging) in [
            (8, 4_548_096, 4_456_448),
            (512, 127_378_944, 4_456_448),
            (1024, 252_159_488, 4_456_448),
            (2048, 501_720_576, 4_456_448),
            (8192, 1_999_087_104, 4_456_448),
            (16384, 3_995_575_808, 8_388_608),
        ] {
            let mut full = DenseModelMemory {
                packed_weight_bytes: 165306368,
                resident_tensor_bytes: 8192 + 4096 + 607744,
                largest_staging_bytes: 4456448,
            };
            // Selected 28-layer geometry: attention has 37,888 fixed bytes and
            // 8,704 bytes per cache position. The coefficient pair overtakes the
            // largest packed tile at C16384; none of these payloads are read.
            // Backend upload conversion requires its own additional forecast.
            for _ in 0..28 {
                full.add(
                    6684672,
                    37888 + 8704 * capacity,
                    2228224.max(512 * capacity),
                )
                .unwrap();
                full.add(10027008, 32768, 3342336).unwrap();
            }
            assert_eq!(
                full,
                DenseModelMemory {
                    packed_weight_bytes: 633233408,
                    resident_tensor_bytes: expected_tensors,
                    largest_staging_bytes: expected_staging
                }
            );
        }
        let mut overflow = DenseModelMemory {
            packed_weight_bytes: u64::MAX,
            resident_tensor_bytes: 1,
            largest_staging_bytes: 1,
        };
        let before = overflow;
        assert!(overflow.add(1, 1, 2).is_err());
        assert_eq!(overflow, before);
        overflow.packed_weight_bytes = 0;
        overflow.resident_tensor_bytes = u64::MAX;
        let before = overflow;
        assert!(overflow.add(1, 1, 2).is_err());
        assert_eq!(overflow, before);
    }

    #[test]
    fn complete_cached_execution_matches_immediate_prefixes_and_reset_without_source_reads() {
        let fixture = fixture(8);
        let mut recorded_results: Vec<Vec<Vec<f32>>> = Vec::new();
        for mode in 0..3 {
            let on_demand = mode != 0;
            let model = QwenModel::load(&fixture.checkpoint).unwrap();
            let plan = if mode == 2 {
                model.dense_model_plan_on_demand(3, 4).unwrap()
            } else {
                model.dense_model_plan(3, 4).unwrap()
            };
            assert_eq!(plan.requires_on_demand_attention(), mode == 2);
            let source = CountingSource::new(&fixture.checkpoint);
            let device = CpuExecutionDevice::new();
            let mut resident = if on_demand {
                plan.prepare_on_demand_attention(&device, &source, limits(&plan))
                    .unwrap()
            } else {
                plan.prepare(&device, &source, limits(&plan)).unwrap()
            };
            assert_eq!(resident.records_attention_on_demand(), on_demand);
            assert_eq!(source.reads.lock().unwrap().len(), 26);
            assert_eq!(
                source
                    .reads
                    .lock()
                    .unwrap()
                    .iter()
                    .map(|range| range.nbytes())
                    .sum::<u64>(),
                23218
            );
            let oracle_source = ResidentSource::new(&fixture.checkpoint);
            let cpu = Cpu::new();
            let mut previous_request = Vec::new();
            for (request, tokens) in [[1u32, 2, 8], [4, 5, 0]].iter().enumerate() {
                let mut request_logits = Vec::new();
                for position in 0..3 {
                    let expected = model
                        .forward(&cpu, &oracle_source, &tokens[..=position])
                        .unwrap();
                    resident.step(&device, tokens[position]).unwrap();
                    let mut actual = vec![0.0; 9];
                    resident.read_logits(&device, &mut actual).unwrap();
                    for (&value, &reference) in actual.iter().zip(&expected.logits) {
                        assert!(value.is_finite() && reference.is_finite());
                        assert!(
                            (value - reference).abs() <= 2e-6 + 2e-6 * reference.abs(),
                            "request {request} position {position}: {value} != {reference}"
                        );
                    }
                    if on_demand {
                        assert_eq!(
                            actual
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            recorded_results[request][position]
                                .iter()
                                .map(|value| value.to_bits())
                                .collect::<Vec<_>>(),
                            "on-demand arithmetic must exactly match pre-recorded CPU execution",
                        );
                    }
                    assert_eq!(resident.visible_positions(), position + 1);
                    assert!(!resident.is_poisoned());
                    request_logits.push(actual);
                }
                assert!(resident.step(&device, 0).is_err());
                assert!(!resident.is_poisoned());
                assert_eq!(resident.visible_positions(), 3);
                if request == 1 {
                    assert_ne!(request_logits, previous_request);
                }
                if !on_demand {
                    recorded_results.push(request_logits.clone());
                }
                previous_request = request_logits;
                resident.reset().unwrap();
                assert_eq!(resident.visible_positions(), 0);
                assert!(resident.read_logits(&device, &mut [0.0; 9]).is_err());
                assert!(!resident.is_poisoned());
            }
            assert_eq!(source.reads.lock().unwrap().len(), 26);
            drop(source);
            drop(plan);
            drop(model);
            resident.step(&device, 8).unwrap();
            resident.read_logits(&device, &mut [0.0; 9]).unwrap();
        }
    }

    #[test]
    fn aggregate_limits_and_foreign_source_refuse_before_any_reads() {
        let fixture = fixture(8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_model_plan(3, 4).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        for index in 0..3 {
            let mut bound = limits(&plan);
            match index {
                0 => bound.packed_weight_bytes -= 1,
                1 => bound.resident_tensor_bytes -= 1,
                _ => bound.largest_staging_bytes -= 1,
            }
            assert!(plan.prepare(&device, &source, bound).is_err());
            assert!(plan
                .prepare_on_demand_attention(&device, &source, bound)
                .is_err());
        }
        let foreign_fixture = super::tests::fixture(8);
        let foreign = CountingSource::new(&foreign_fixture.checkpoint);
        assert!(plan.prepare(&device, &foreign, limits(&plan)).is_err());
        assert!(plan
            .prepare_on_demand_attention(&device, &foreign, limits(&plan))
            .is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(foreign.reads.lock().unwrap().is_empty());
    }

    #[test]
    fn metadata_accepts_c64_and_c128_within_context_and_keeps_layer_position_bound() {
        let fixture = fixture(8);
        // Only metadata is changed; this test performs no preparation or payload I/O.
        for capacity in [64, 128] {
            let mut model = QwenModel::load(&fixture.checkpoint).unwrap();
            model.core.cfg.context = capacity;
            let plan = model.dense_model_plan(capacity, 4).unwrap();
            assert_eq!((plan.layers(), plan.capacity()), (2, capacity));
            assert!(model.dense_model_plan(capacity + 1, 4).is_err());
            model.core.cfg.context = 256;
            let overflow_layers = 4096 / capacity + 1;
            while model.core.layers.len() < overflow_layers {
                let mut binding = QwenModel::load(&fixture.checkpoint).unwrap();
                model.core.layers.push(binding.core.layers.pop().unwrap());
            }
            model.core.cfg.n_layers = overflow_layers;
            let error = model.dense_model_plan(capacity, 4).unwrap_err().to_string();
            assert!(error.contains("4096 layer-position batches"));
        }
    }

    #[test]
    fn on_demand_planner_admits_16384_without_expanding_legacy_bounds() {
        let fixture = fixture_with_context(8, 32768);
        let mut model = QwenModel::load(&fixture.checkpoint).unwrap();
        let retained_metadata = model
            .dense_model_plan(128, 4)
            .unwrap()
            .on_demand_attention_host_metadata_bytes()
            .unwrap();
        for capacity in [512, 513, 1024, 2048, 2049, 4096, 8192, 16384] {
            let plan = model.dense_model_plan_on_demand(capacity, 4).unwrap();
            assert!(plan.requires_on_demand_attention());
            assert_eq!((plan.layers(), plan.capacity()), (2, capacity));
            assert_eq!(
                plan.memory(),
                DenseModelMemory {
                    packed_weight_bytes: 22066,
                    resident_tensor_bytes: 4004 + 768 * capacity as u64,
                    largest_staging_bytes: 128 * capacity as u64,
                }
            );
            assert_eq!(
                plan.on_demand_attention_host_metadata_bytes().unwrap(),
                retained_metadata
            );
        }
        for capacity in [0, 16385, usize::MAX] {
            assert!(model.dense_model_plan_on_demand(capacity, 4).is_err());
        }
        for capacity in [257, 512, 2048, 8192, 16384] {
            assert!(model.dense_model_plan(capacity, 4).is_err());
        }
        assert!(!model
            .dense_model_plan(256, 4)
            .unwrap()
            .requires_on_demand_attention());
        for capacity in [2048, 8192, 16384] {
            model.core.cfg.context = capacity - 1;
            assert!(model.dense_model_plan_on_demand(capacity, 4).is_err());
        }
        model.core.cfg.context = MAX_ON_DEMAND_CAPACITY;
        // Seventeen layers exceed the legacy 4096-entry table at capacity 256.
        // On-demand planning retains seventeen layer states, not 4352 batches.
        while model.core.layers.len() < 17 {
            let mut binding = QwenModel::load(&fixture.checkpoint).unwrap();
            model.core.layers.push(binding.core.layers.pop().unwrap());
        }
        model.core.cfg.n_layers = 17;
        assert!(model.dense_model_plan(256, 4).is_err());
        assert_eq!(
            model.dense_model_plan_on_demand(16384, 4).unwrap().layers(),
            17
        );
        model.core.cfg.n_layers = 129;
        assert!(model.dense_model_plan_on_demand(16384, 4).is_err());
    }

    #[test]
    fn on_demand_model_mode_and_aggregate_limits_refuse_before_payload_reads() {
        let fixture = fixture_with_context(8, 16384);
        let foreign_fixture = fixture_with_context(8, 16384);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let foreign = CountingSource::new(&foreign_fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        for capacity in [3, 512, 2048, 8192, 16384] {
            let plan = model.dense_model_plan_on_demand(capacity, 4).unwrap();
            let error = plan.prepare(&device, &source, limits(&plan)).unwrap_err();
            assert!(error
                .to_string()
                .contains("requires prepare_on_demand_attention"));
            assert!(plan
                .prepare_on_demand_attention(&device, &foreign, limits(&plan))
                .is_err());
            for field in 0..3 {
                let mut insufficient = limits(&plan);
                match field {
                    0 => insufficient.packed_weight_bytes -= 1,
                    1 => insufficient.resident_tensor_bytes -= 1,
                    _ => insufficient.largest_staging_bytes -= 1,
                }
                assert!(plan
                    .prepare_on_demand_attention(&device, &source, insufficient)
                    .is_err());
            }
        }
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(foreign.reads.lock().unwrap().is_empty());
    }

    #[test]
    fn complete_on_demand_capacity_512_matches_immediate_final_prefix_and_reset() {
        let fixture = fixture_with_context(8, 512);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_model_plan_on_demand(512, 4).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let mut resident = plan
            .prepare_on_demand_attention(&device, &source, limits(&plan))
            .unwrap();
        let tokens = (0..512).map(|i| (i % 9) as u32).collect::<Vec<_>>();
        let mut first = [0.0; 9];
        for (position, &token) in tokens.iter().enumerate() {
            resident.step(&device, token).unwrap();
            assert_eq!(resident.visible_positions(), position + 1);
            if position == 0 {
                resident.read_logits(&device, &mut first).unwrap();
            }
        }
        let expected = model
            .forward(
                &Cpu::new(),
                &ResidentSource::new(&fixture.checkpoint),
                &tokens,
            )
            .unwrap();
        let mut actual = [0.0; 9];
        resident.read_logits(&device, &mut actual).unwrap();
        for (&value, &reference) in actual.iter().zip(&expected.logits) {
            assert!(value.is_finite() && reference.is_finite());
            assert!((value - reference).abs() <= 2e-6 + 2e-6 * reference.abs());
        }
        assert!(resident.step(&device, 0).is_err());
        assert!(!resident.is_poisoned());
        assert_eq!(resident.visible_positions(), 512);
        resident.reset().unwrap();
        assert_eq!(resident.visible_positions(), 0);
        assert!(resident.read_logits(&device, &mut actual).is_err());
        resident.step(&device, tokens[0]).unwrap();
        resident.read_logits(&device, &mut actual).unwrap();
        assert_eq!(actual.map(f32::to_bits), first.map(f32::to_bits));
        assert_eq!(source.reads.lock().unwrap().len(), 26);
    }

    #[test]
    fn metadata_refuses_late_unsupported_formats_shapes_and_unbounded_recording() {
        let mixed = fixture(30);
        let model = QwenModel::load(&mixed.checkpoint).unwrap();
        assert!(model
            .dense_model_plan(3, 4)
            .unwrap_err()
            .to_string()
            .contains("Q8_0"));
        assert!(model
            .dense_model_plan_on_demand(3, 4)
            .unwrap_err()
            .to_string()
            .contains("Q8_0"));
        let fixture = fixture(8);
        let mut model = QwenModel::load(&fixture.checkpoint).unwrap();
        for capacity in [0, 17, 257, usize::MAX] {
            assert!(model.dense_model_plan(capacity, 4).is_err());
        }
        for rows in [0, 4097, usize::MAX] {
            assert!(model.dense_model_plan(3, rows).is_err());
            assert!(model.dense_model_plan_on_demand(3, rows).is_err());
        }
        model.core.cfg.n_layers = 129;
        assert!(model.dense_model_plan(3, 4).is_err());
        model.core.cfg.n_layers = 2;
        model.core.cfg.expert_ff = 4097;
        assert!(model.dense_model_plan(3, 4).is_err());
        assert!(model.dense_model_plan_on_demand(3, 4).is_err());
        model.core.cfg.expert_ff = 64;
        if let FeedForward::Dense { down, .. } = &mut model.core.layers[1].ffn {
            down.cols -= 1;
        }
        assert!(model.dense_model_plan(3, 4).is_err());
        assert!(model.dense_model_plan_on_demand(3, 4).is_err());
    }

    #[test]
    fn invalid_token_or_output_boundary_keeps_healthy_state_and_transport_read_poisons() {
        let fixture = fixture(8);
        for on_demand in [false, true] {
            let model = QwenModel::load(&fixture.checkpoint).unwrap();
            let plan = model.dense_model_plan(3, 4).unwrap();
            let device = CpuExecutionDevice::new();
            let mut resident = plan
                .prepare_with_attention_recording(
                    &device,
                    &ResidentSource::new(&fixture.checkpoint),
                    limits(&plan),
                    on_demand,
                )
                .unwrap();
            assert!(resident.read_logits(&device, &mut [0.0; 9]).is_err());
            for token in [9, u32::MAX] {
                assert!(resident.step(&device, token).is_err());
            }
            assert_eq!(resident.visible_positions(), 0);
            assert!(!resident.is_poisoned());
            resident.step(&device, 1).unwrap();
            assert!(resident.read_logits(&device, &mut [0.0; 8]).is_err());
            assert!(!resident.is_poisoned());
            assert!(resident
                .read_logits(&CpuExecutionDevice::new(), &mut [0.0; 9])
                .is_err());
            assert!(resident.is_poisoned());
            assert_eq!(resident.visible_positions(), 1);
            assert!(resident.reset().is_err());
            assert!(resident.step(&device, 2).is_err());
            assert!(resident.read_logits(&device, &mut [0.0; 9]).is_err());
        }
    }

    #[test]
    fn late_execution_error_and_post_head_unwind_never_publish_partial_positions() {
        let fixture = fixture(8);
        for on_demand in [false, true] {
            let model = QwenModel::load(&fixture.checkpoint).unwrap();
            let plan = model.dense_model_plan(3, 4).unwrap();
            let device = CpuExecutionDevice::new();
            let source = ResidentSource::new(&fixture.checkpoint);
            for (fail_at, unwind) in [
                (CompletedStage::Ffn(1), false),
                (CompletedStage::Head, false),
                (CompletedStage::Head, true),
            ] {
                let mut resident = if on_demand {
                    plan.prepare_on_demand_attention(&device, &source, limits(&plan))
                        .unwrap()
                } else {
                    plan.prepare(&device, &source, limits(&plan)).unwrap()
                };
                resident.step(&device, 1).unwrap();
                let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    resident.step_with(&device, 2, |stage| {
                        if stage == fail_at {
                            if unwind {
                                panic!("injected error after full head execution");
                            }
                            anyhow::bail!("injected error after completed stage {stage:?}");
                        }
                        Ok(())
                    })
                }));
                if unwind {
                    assert!(outcome.is_err());
                } else {
                    assert!(outcome.unwrap().is_err());
                }
                assert!(resident.is_poisoned());
                assert_eq!(resident.visible_positions(), 1);
                assert!(resident
                    .layers
                    .iter()
                    .all(|layer| layer.attention.visible_positions() == 2));
                assert!(resident.reset().is_err());
                assert!(resident.step(&device, 3).is_err());
                assert!(resident.read_logits(&device, &mut [0.0; 9]).is_err());
            }
        }
    }

    #[test]
    fn on_demand_late_recording_failure_poison_prevents_partial_model_reuse() {
        let fixture = fixture(8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.dense_model_plan(3, 4).unwrap();
        let source = ResidentSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let mut resident = plan
            .prepare_on_demand_attention(&device, &source, limits(&plan))
            .unwrap();
        // Only this private test can replace a layer with foreign prepared
        // operands. Its semantic commands are valid, but device.record refuses
        // their owner after embedding and the earlier layer already executed.
        let foreign = CpuExecutionDevice::new();
        let hidden = foreign.allocate_f32(plan.dim()).unwrap();
        let attention = model.dense_attention_plan(1).unwrap();
        let memory = attention.memory(3).unwrap();
        resident.layers[1].attention = attention
            .prepare(
                &foreign,
                &source,
                3,
                DenseAttentionLimits {
                    packed_weight_bytes: memory.packed_weight_bytes,
                    resident_tensor_bytes: memory.resident_tensor_bytes,
                    largest_staging_bytes: memory.largest_staging_bytes,
                },
            )
            .unwrap()
            .record_on_demand(&hidden.full_view())
            .unwrap();
        assert!(resident.step(&device, 1).is_err());
        assert_eq!(resident.visible_positions(), 0);
        assert_eq!(resident.layers[0].attention.visible_positions(), 1);
        assert_eq!(resident.layers[1].attention.visible_positions(), 0);
        assert!(resident.layers[1].attention.is_poisoned());
        assert!(resident.is_poisoned());
        assert!(resident.reset().is_err());
        assert!(resident.step(&device, 2).is_err());
        assert!(resident.read_logits(&device, &mut [0.0; 9]).is_err());
    }

    #[test]
    fn reset_prevalidates_every_cache_before_mutating_any_position() {
        let fixture = fixture(8);
        for on_demand in [false, true] {
            let model = QwenModel::load(&fixture.checkpoint).unwrap();
            let plan = model.dense_model_plan(3, 4).unwrap();
            let device = CpuExecutionDevice::new();
            let mut resident = plan
                .prepare_with_attention_recording(
                    &device,
                    &ResidentSource::new(&fixture.checkpoint),
                    limits(&plan),
                    on_demand,
                )
                .unwrap();
            resident.step(&device, 1).unwrap();
            // Only this private test can advance a layer independently of the model.
            resident.layers[1].attention.step(&device).unwrap();
            assert!(resident.reset().is_err());
            assert!(resident.is_poisoned());
            assert_eq!(resident.visible_positions(), 1);
            assert_eq!(resident.layers[0].attention.visible_positions(), 1);
            assert_eq!(resident.layers[1].attention.visible_positions(), 2);
        }
    }
}
