//! One tied Qwen3 Q8_0 vocabulary prepared once for embedding lookup and logits.
//!
//! Metadata planning partitions exact row ranges without reading payloads. Each tile
//! is staged independently and shared by both semantic uses; there is no expanded
//! checkpoint cache. Final normalization is a resident F32 tensor operation.

use super::{Bound, CheckpointId, QwenArchitecture, QwenModel};
use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, ExecutionDevice, F32Tensor, F32TensorView, LinearMode, PreparedWeight,
    RecordedBatch,
};
use beholdd_backend::Weight;
use beholdd_format::{GgmlType, WeightRange};
use beholdd_memory::{WeightAccess, WeightSource};

/// Exact logical payloads, excluding caller hidden/logits, backend private scratch,
/// alignment/guards/code, prepared-weight expansion and source-owned caches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocabularyMemory {
    /// Unique packed bytes shared by embedding and output projection.
    pub packed_weight_bytes: u64,
    /// Final F32 norm and one normalized hidden vector.
    pub resident_tensor_bytes: u64,
    /// Largest packed tile or simultaneous raw and decoded final norm payloads.
    pub largest_staging_bytes: u64,
}

/// Independent limits checked before the first source read or device allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VocabularyLimits {
    /// Maximum sum of the unique packed tiles.
    pub packed_weight_bytes: u64,
    /// Maximum norm plus normalized-hidden payloads.
    pub resident_tensor_bytes: u64,
    /// Maximum one-at-a-time source staging payload.
    pub largest_staging_bytes: u64,
}

/// Checked contiguous row range within the original tied vocabulary tensor.
#[derive(Clone, Copy, Debug)]
pub struct VocabularyTile {
    first_token: usize,
    rows: usize,
    range: WeightRange,
}
impl VocabularyTile {
    /// First vocabulary token covered by this tile.
    pub fn first_token(&self) -> usize {
        self.first_token
    }
    /// Number of rows in this tile, at most the plan's tile-row limit.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Checked exact encoded range, retaining the original checkpoint/locator identity.
    pub fn range(&self) -> WeightRange {
        self.range
    }
}

/// Metadata-only Qwen3 tied-embedding/output plan with bounded Q8_0 row tiles.
#[derive(Clone, Debug)]
pub struct VocabularyPlan {
    checkpoint_id: CheckpointId,
    dim: usize,
    vocab: usize,
    tile_rows: usize,
    eps: f32,
    tiles: Vec<VocabularyTile>,
    norm: Bound,
    norm_range: WeightRange,
    memory: VocabularyMemory,
}

impl QwenModel {
    /// Plan a tied Q8_0 embedding/head and F32 final norm without payload reads.
    /// Tile rows must be 1..=4096; hidden width must be a complete Q8_0 block and
    /// at most 4096. At most 4096 tiles are admitted to bound metadata allocation.
    pub fn vocabulary_plan(&self, tile_rows: usize) -> Result<VocabularyPlan> {
        ensure!(
            self.architecture == QwenArchitecture::Qwen3,
            "recorded vocabulary requires exact qwen3 architecture"
        );
        ensure!(
            (1..=4096).contains(&tile_rows),
            "vocabulary tile rows must be in 1..=4096"
        );
        let cfg = &self.core.cfg;
        let embedding = &self.core.embed;
        let output = &self.core.output;
        ensure!(
            embedding.tensor.locator() == output.tensor.locator(),
            "recorded vocabulary requires identical tied embedding/head locators"
        );
        ensure!(
            (1..=4096).contains(&cfg.dim) && cfg.dim.is_multiple_of(32) && cfg.vocab > 0,
            "unsupported vocabulary dimensions"
        );
        ensure!(
            u32::try_from(cfg.vocab - 1).is_ok(),
            "vocabulary token IDs exceed u32"
        );
        ensure!(
            cfg.eps.is_finite() && cfg.eps >= 0.0,
            "vocabulary norm epsilon must be finite and nonnegative"
        );
        ensure!(
            embedding.tensor.kind().0 == 8
                && output.tensor.kind().0 == 8
                && embedding.rows == cfg.vocab
                && output.rows == cfg.vocab
                && embedding.cols == cfg.dim
                && output.cols == cfg.dim,
            "vocabulary requires exact tied Q8_0 matrix shapes"
        );
        let row_bytes = u64::try_from(cfg.dim / 32)?
            .checked_mul(34)
            .context("vocabulary encoded row overflows")?;
        let packed = u64::try_from(cfg.vocab)?
            .checked_mul(row_bytes)
            .context("vocabulary encoded extent overflows")?;
        ensure!(
            embedding.tensor.nbytes() == packed && output.tensor.nbytes() == packed,
            "vocabulary packed span differs"
        );
        let norm = &self.core.out_norm;
        let norm_bytes = u64::try_from(cfg.dim)?
            .checked_mul(4)
            .context("vocabulary norm bytes overflow")?;
        ensure!(
            norm.tensor.kind().0 == 0
                && norm.rows == 1
                && norm.cols == cfg.dim
                && norm.tensor.nbytes() == norm_bytes,
            "vocabulary final norm must be an exact F32 vector"
        );
        let tile_count = cfg.vocab.div_ceil(tile_rows);
        ensure!(tile_count <= 4096, "vocabulary metadata exceeds 4096 tiles");
        let mut tiles = Vec::new();
        tiles
            .try_reserve_exact(tile_count)
            .context("allocating vocabulary tile metadata")?;
        for first_token in (0..cfg.vocab).step_by(tile_rows) {
            let rows = tile_rows.min(cfg.vocab - first_token);
            let first = u64::try_from(first_token)?
                .checked_mul(row_bytes)
                .context("tile offset overflows")?;
            let end = u64::try_from(
                first_token
                    .checked_add(rows)
                    .context("tile end overflows")?,
            )?
            .checked_mul(row_bytes)
            .context("tile byte end overflows")?;
            tiles.push(VocabularyTile {
                first_token,
                rows,
                range: embedding.tensor.locator().range(first..end)?,
            });
        }
        let norm_pair = norm_bytes
            .checked_mul(2)
            .context("vocabulary norm staging overflows")?;
        let largest = tiles
            .iter()
            .map(|tile| tile.range.nbytes())
            .max()
            .context("vocabulary has no tiles")?;
        Ok(VocabularyPlan {
            checkpoint_id: self.core.checkpoint_id,
            dim: cfg.dim,
            vocab: cfg.vocab,
            tile_rows,
            eps: cfg.eps,
            tiles,
            norm: norm.clone(),
            norm_range: norm.tensor.locator().whole_range(),
            memory: VocabularyMemory {
                packed_weight_bytes: packed,
                resident_tensor_bytes: norm_pair,
                largest_staging_bytes: largest.max(norm_pair),
            },
        })
    }
}

impl VocabularyPlan {
    /// Width of hidden vectors and embedding rows, in F32 elements.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Number of vocabulary entries and final logits.
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    /// Maximum rows in each independently prepared packed tile.
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }
    /// Finite, nonnegative final RMSNorm epsilon.
    pub fn eps(&self) -> f32 {
        self.eps
    }
    /// Immutable exact row ranges, in increasing token order.
    pub fn tiles(&self) -> &[VocabularyTile] {
        &self.tiles
    }
    /// Logical requirements before backend expansion/alignment and caller tensors.
    pub fn memory(&self) -> VocabularyMemory {
        self.memory
    }
    /// Read and upload each packed tile once, followed by the final F32 norm.
    /// Source identity and every logical bound/format are checked before any I/O or
    /// allocation. Backend geometry/capacity refusal remains explicit; no fallback.
    pub fn prepare(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        limits: VocabularyLimits,
    ) -> Result<PreparedVocabulary> {
        ensure!(
            source.checkpoint_id() == self.checkpoint_id,
            "vocabulary plan and source belong to different checkpoints"
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
                "vocabulary {label} needs {required} logical bytes, limit is {limit}"
            );
        }
        ensure!(
            device.supports(GgmlType(8)),
            "backend refuses vocabulary Q8_0 format"
        );
        let mut weights = Vec::new();
        weights
            .try_reserve_exact(self.tiles.len())
            .context("allocating prepared vocabulary tile handles")?;
        for tile in &self.tiles {
            let resident = source.load_range(&tile.range, WeightAccess::Cyclic)?;
            ensure!(
                resident.bytes().len() as u64 == tile.range.nbytes(),
                "source returned an incorrect vocabulary tile span"
            );
            let encoded = device.upload_weight(&Weight {
                bytes: resident.bytes(),
                kind: GgmlType(8),
                rows: tile.rows,
                cols: self.dim,
            })?;
            weights.push(device.prepare_weight(&encoded)?);
            // Both source staging and the encoded duplicate handle retire before
            // loading the next tile; the prepared weight owns backend residency.
        }
        let norm = {
            let resident = source.load_range(&self.norm_range, WeightAccess::Cyclic)?;
            ensure!(
                resident.bytes().len() as u64 == self.norm_range.nbytes(),
                "source returned an incorrect vocabulary norm span"
            );
            let mut values = Vec::new();
            values
                .try_reserve_exact(self.norm.cols)
                .context("allocating vocabulary norm staging")?;
            values.extend(
                resident
                    .bytes()
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|bytes| f32::from_le_bytes(*bytes)),
            );
            device.upload_f32(&values)?
        };
        Ok(PreparedVocabulary {
            dim: self.dim,
            vocab: self.vocab,
            tile_rows: self.tile_rows,
            eps: self.eps,
            weights,
            norm,
            normalized: device.allocate_f32(self.dim)?,
        })
    }
}

/// One owned set of packed tiles shared by lookup and the normalized vocabulary head.
/// Recorded batches retain the operands they use after this preparation is dropped.
#[derive(Debug)]
pub struct PreparedVocabulary {
    dim: usize,
    vocab: usize,
    tile_rows: usize,
    eps: f32,
    weights: Vec<PreparedWeight>,
    norm: F32Tensor,
    normalized: F32Tensor,
}
impl PreparedVocabulary {
    /// Record one token lookup; token range/hidden width are checked before building
    /// a command. Only the selected tile and output are retained by the batch.
    pub fn record_embedding(
        &self,
        device: &dyn ExecutionDevice,
        token: usize,
        hidden: &F32TensorView,
    ) -> Result<RecordedBatch> {
        ensure!(
            token < self.vocab,
            "embedding token {token} is outside vocabulary {}",
            self.vocab
        );
        ensure!(hidden.len() == self.dim, "embedding hidden width differs");
        let weight = &self.weights[token / self.tile_rows];
        let mut commands = CommandList::new();
        commands.embedding_row(hidden, weight, token % self.tile_rows)?;
        Ok(device.record(&commands)?)
    }
    /// Record weighted final RMSNorm followed by one overwrite matvec per tile.
    /// Batches share normalized scratch: wait before modifying boundaries or
    /// submitting another batch from this preparation. No concurrency is implied.
    pub fn record_logits(
        &self,
        device: &dyn ExecutionDevice,
        hidden: &F32TensorView,
        logits: &F32TensorView,
    ) -> Result<RecordedBatch> {
        ensure!(
            hidden.len() == self.dim && logits.len() == self.vocab,
            "vocabulary hidden/logit shape differs"
        );
        let mut commands = CommandList::new();
        commands.rmsnorm(
            &self.normalized.full_view(),
            hidden,
            Some(&self.norm.full_view()),
            self.eps,
        )?;
        for (tile, weight) in self.weights.iter().enumerate() {
            let first = tile * self.tile_rows;
            let rows = self.tile_rows.min(self.vocab - first);
            commands.linear(
                &logits.view(first, rows)?,
                &self.normalized.full_view(),
                weight,
                LinearMode::Overwrite,
            )?;
        }
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
            ("token_embd.weight".into(), vec![32, 9], 8),
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
            "beholdd-recorded-vocabulary-{}-{}.gguf",
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

    fn limits(plan: &VocabularyPlan) -> VocabularyLimits {
        let m = plan.memory();
        VocabularyLimits {
            packed_weight_bytes: m.packed_weight_bytes,
            resident_tensor_bytes: m.resident_tensor_bytes,
            largest_staging_bytes: m.largest_staging_bytes,
        }
    }

    #[test]
    fn exact_tied_ranges_partition_rows_once_and_account_staging() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.vocabulary_plan(4).unwrap();
        assert_eq!(plan.dim(), 32);
        assert_eq!(plan.vocab(), 9);
        assert_eq!(plan.tile_rows(), 4);
        assert_eq!(
            plan.tiles()
                .iter()
                .map(|tile| (tile.first_token(), tile.rows()))
                .collect::<Vec<_>>(),
            vec![(0, 4), (4, 4), (8, 1)]
        );
        assert_eq!(
            plan.memory(),
            VocabularyMemory {
                packed_weight_bytes: 306,
                resident_tensor_bytes: 256,
                largest_staging_bytes: 256
            }
        );
        let locator = model.core.embed.tensor.locator();
        assert_eq!(locator, model.core.output.tensor.locator());
        let mut next = locator.start();
        for tile in plan.tiles() {
            assert_eq!(tile.range().locator(), locator);
            assert_eq!(tile.range().start(), next);
            assert_eq!(tile.range().nbytes(), tile.rows() as u64 * 34);
            next = tile.range().end();
        }
        assert_eq!(next, locator.end());
        assert!(model.vocabulary_plan(0).is_err());
        assert!(model.vocabulary_plan(4097).is_err());
    }

    #[test]
    fn shared_tiles_lookup_head_parity_replay_and_batch_retention() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.vocabulary_plan(4).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let prepared = plan.prepare(&device, &source, limits(&plan)).unwrap();
        assert_eq!(source.reads.lock().unwrap().len(), 4);
        let cpu = Cpu::new();
        let embedding = &model.core.embed;
        let packed = ResidentWeight::load(&fixture.checkpoint, embedding.tensor.locator()).unwrap();
        let weight = Weight {
            bytes: packed.bytes(),
            kind: GgmlType(8),
            rows: 9,
            cols: 32,
        };
        let mut all_rows = vec![0.0; 9 * 32];
        cpu.dequantize(&mut all_rows, &weight).unwrap();
        let hidden = device.upload_f32(&[-999.0; 34]).unwrap();
        let logits = device.upload_f32(&[-999.0; 11]).unwrap();
        let hidden_view = hidden.view(1, 32).unwrap();
        let output_view = logits.view(1, 9).unwrap();
        for token in [0, 3, 4, 7, 8] {
            let batch = prepared
                .record_embedding(&device, token, &hidden_view)
                .unwrap();
            assert_eq!(batch.len(), 1);
            let submission = device.submit(&batch).unwrap();
            device.wait(&submission).unwrap();
            let mut actual = vec![0.0; 34];
            device.read_f32(&hidden.full_view(), &mut actual).unwrap();
            assert_eq!(actual[0], -999.0);
            assert_eq!(actual[33], -999.0);
            assert_eq!(
                actual[1..33]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                all_rows[token * 32..(token + 1) * 32]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        let batch = prepared
            .record_logits(&device, &hidden_view, &output_view)
            .unwrap();
        let lookup = prepared.record_embedding(&device, 8, &hidden_view).unwrap();
        assert_eq!(batch.len(), 4);
        drop(prepared);
        let norm = ResidentWeight::load_range(&fixture.checkpoint, &plan.norm_range).unwrap();
        let norm = norm
            .bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        for round in 0..3 {
            let input = (0..32)
                .map(|i| (((i * 7 + round * 3) % 23) as f32 - 11.0) / 16.0)
                .collect::<Vec<_>>();
            let mut normalized = vec![0.0; 32];
            cpu.rmsnorm(&mut normalized, &input, Some(&norm), plan.eps())
                .unwrap();
            let mut expected = vec![0.0; 9];
            cpu.matvec_with_threads(&mut expected, &normalized, &weight, 1)
                .unwrap();
            device.write_f32(&hidden_view, &input).unwrap();
            let submission = device.submit(&batch).unwrap();
            device.wait(&submission).unwrap();
            let mut actual = vec![0.0; 11];
            device.read_f32(&logits.full_view(), &mut actual).unwrap();
            assert_eq!(actual[0], -999.0);
            assert_eq!(actual[10], -999.0);
            assert_eq!(&actual[1..10], expected);
        }
        let submission = device.submit(&lookup).unwrap();
        device.wait(&submission).unwrap();
        let mut actual = vec![0.0; 32];
        device.read_f32(&hidden_view, &mut actual).unwrap();
        assert_eq!(actual, &all_rows[8 * 32..]);
        assert_eq!(source.reads.lock().unwrap().len(), 4);
    }

    #[test]
    fn preparation_limits_source_identity_and_metadata_fail_before_io() {
        let fixture = make_fixture(true, 8);
        let other = make_fixture(true, 8);
        let mut model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.vocabulary_plan(4).unwrap();
        let device = CpuExecutionDevice::new();
        let source = CountingSource::new(&fixture.checkpoint);
        let foreign = CountingSource::new(&other.checkpoint);
        assert!(plan.prepare(&device, &foreign, limits(&plan)).is_err());
        for field in 0..3 {
            let mut bound = limits(&plan);
            match field {
                0 => bound.packed_weight_bytes -= 1,
                1 => bound.resident_tensor_bytes -= 1,
                _ => bound.largest_staging_bytes -= 1,
            };
            assert!(plan.prepare(&device, &source, bound).is_err());
        }
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(foreign.reads.lock().unwrap().is_empty());
        model.core.output = model.core.layers[0].wq.clone();
        assert!(model
            .vocabulary_plan(4)
            .unwrap_err()
            .to_string()
            .contains("identical tied"));
        model.core.output = model.core.embed.clone();
        model.core.cfg.eps = f32::NAN;
        assert!(model.vocabulary_plan(4).is_err());
        model.core.cfg.eps = 1e-6;
        model.core.out_norm = model.core.embed.clone();
        assert!(model.vocabulary_plan(4).is_err());
        let routed = make_fixture(false, 8);
        assert!(QwenModel::load(&routed.checkpoint)
            .unwrap()
            .vocabulary_plan(4)
            .is_err());
    }

    #[test]
    fn invalid_tokens_shapes_and_foreign_owners_cannot_record_or_mutate_outputs() {
        let fixture = make_fixture(true, 8);
        let model = QwenModel::load(&fixture.checkpoint).unwrap();
        let plan = model.vocabulary_plan(4).unwrap();
        let source = CountingSource::new(&fixture.checkpoint);
        let device = CpuExecutionDevice::new();
        let other = CpuExecutionDevice::new();
        let prepared = plan.prepare(&device, &source, limits(&plan)).unwrap();
        let hidden = device.upload_f32(&[7.0; 32]).unwrap();
        let logits = device.upload_f32(&[8.0; 9]).unwrap();
        for token in [9, usize::MAX] {
            assert!(prepared
                .record_embedding(&device, token, &hidden.full_view())
                .is_err());
        }
        assert!(prepared
            .record_embedding(&device, 0, &hidden.view(0, 31).unwrap())
            .is_err());
        assert!(prepared
            .record_embedding(&other, 0, &hidden.full_view())
            .is_err());
        let foreign = other.allocate_f32(32).unwrap();
        assert!(prepared
            .record_embedding(&device, 0, &foreign.full_view())
            .is_err());
        assert!(prepared
            .record_logits(&device, &hidden.full_view(), &logits.view(0, 8).unwrap())
            .is_err());
        assert!(prepared
            .record_logits(&device, &foreign.full_view(), &logits.full_view())
            .is_err());
        let mut actual = [0.0; 32];
        device.read_f32(&hidden.full_view(), &mut actual).unwrap();
        assert_eq!(actual, [7.0; 32]);
        let mut output = [0.0; 9];
        device.read_f32(&logits.full_view(), &mut output).unwrap();
        assert_eq!(output, [8.0; 9]);
        assert_eq!(source.reads.lock().unwrap().len(), 4);
    }
}
