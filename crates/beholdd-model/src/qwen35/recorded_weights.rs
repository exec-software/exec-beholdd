//! Bounded packed-row streaming into an explicitly supplied execution device.
//!
//! Plans retain only a locator and scalar geometry. One exact packed tile is read,
//! uploaded, recorded and fenced at a time; every tile handle is dropped before
//! the next read. Activations remain in caller-owned tensor views. This module
//! neither selects a backend nor falls back to host arithmetic, and does not
//! establish complete-model or serving support.
//!
//! Errors can leave a prefix of the destination changed. The model owner must
//! abandon/reset that incomplete operation; these helpers do not publish sequence
//! positions or pretend that cancellation rolls back an already fenced tile.
//! The caller must serialize tensor/queue use across the complete operation, not
//! merely individual submissions, so its input and accumulated output stay stable.

use std::ops::Range;

use anyhow::{ensure, Context, Result};
use beholdd_backend::execution::{
    CommandList, EncodedWeight, ExecutionDevice, F32TensorView, LinearMode, PreparedWeight,
    RecordedBatch, Submission,
};
use beholdd_backend::Weight;
use beholdd_core::ggml_block_geometry;
use beholdd_format::{GgmlType, GgufTensor, WeightLocator, WeightRange};
use beholdd_memory::{WeightAccess, WeightSource};

use super::weights::ReadLimits;

/// Observable progress, retained by the caller even when an operation fails.
///
/// Read bytes describe returned slices, including an incorrectly sized slice on
/// refusal. Source caches/backing allocations, device allocations and backend
/// scratch are separate costs. No activation readback occurs in this module.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamStats {
    /// Calls attempted on `WeightSource::load_range`, including failed reads.
    pub read_attempts: u64,
    /// Actual bytes in successfully returned source slices.
    pub returned_read_bytes: u64,
    /// Largest returned slice; at most one such slice is held by this helper.
    pub peak_read_bytes: u64,
    /// Packed bytes successfully copied into device-owned weight storage.
    pub uploaded_bytes: u64,
    /// Calls attempted on `ExecutionDevice::submit`, including failures.
    pub submission_attempts: u64,
    /// Tiles whose submit and explicit wait both succeeded. For a chunk this
    /// increments only after every application of that source tile completes.
    pub completed_tiles: u64,
}

/// Additional chunk progress; the existing stream counters continue to describe
/// actual source reads/uploads and submissions, not hypothetical scalar traffic.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChunkStreamStats {
    /// Individually fenced tile applications, including an incomplete chunk tile.
    pub completed_applications: u64,
}

/// Deliberately fixed prefill experiment sizes. This is not arbitrary batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillChunkSize(usize);

impl PrefillChunkSize {
    /// Admit only one, two or four independent token rows.
    pub fn new(tokens: usize) -> Result<Self> {
        ensure!(
            matches!(tokens, 1 | 2 | 4),
            "prefill chunk size must be 1, 2 or 4"
        );
        Ok(Self(tokens))
    }

    /// Number of token rows in the admitted chunk.
    pub fn tokens(self) -> usize {
        self.0
    }
}

fn added(value: u64, increment: usize, label: &str) -> Result<u64> {
    value
        .checked_add(u64::try_from(increment).context("stream counter increment overflows")?)
        .with_context(|| format!("qwen35 streamed {label} counter overflows"))
}

/// Constant-size row plan for an exact Q4_K/Q6_K matrix.
///
/// Native probe limits are enforced without opening a device: 256..=17408
/// columns in whole 256-element blocks, and at most 4096 rows per tile. The
/// decoded-byte limit also bounds a hypothetical F32 tile for CPU oracle runs;
/// native execution does not decode that tile on the host.
#[derive(Debug, Clone)]
pub struct PackedRowPlan {
    locator: WeightLocator,
    kind: GgmlType,
    rows: usize,
    cols: usize,
    row_bytes: usize,
    tile_rows: usize,
    decoded_bytes: usize,
}

impl PackedRowPlan {
    /// Validate geometry and limits using tensor metadata only.
    pub fn new(tensor: &GgufTensor, limits: ReadLimits) -> Result<Self> {
        ensure!(
            limits.encoded_bytes > 0 && limits.decoded_bytes > 0 && limits.tile_rows > 0,
            "qwen35 streamed read limits must be positive"
        );
        ensure!(
            matches!(tensor.kind().0, 12 | 14),
            "qwen35 streamed weights require Q4_K or Q6_K"
        );
        ensure!(
            tensor.dims().len() == 2 && tensor.dims().iter().all(|&dim| dim > 0),
            "qwen35 streamed matrix requires two positive dimensions"
        );
        let cols =
            usize::try_from(tensor.dims()[0]).context("streamed columns exceed host size")?;
        let rows = usize::try_from(tensor.dims()[1]).context("streamed rows exceed host size")?;
        ensure!(
            (256..=17408).contains(&cols) && cols.is_multiple_of(256),
            "qwen35 streamed columns must be whole 256-element blocks in 256..=17408"
        );
        let (block, block_bytes) = ggml_block_geometry(tensor.kind().0)
            .context("unknown streamed weight block geometry")?;
        let row_bytes = (cols / block)
            .checked_mul(block_bytes)
            .context("streamed packed row size overflows")?;
        let total = row_bytes
            .checked_mul(rows)
            .context("streamed matrix size overflows")?;
        ensure!(
            u64::try_from(total)? == tensor.nbytes(),
            "streamed matrix storage length differs"
        );
        let decoded_row = cols
            .checked_mul(4)
            .context("streamed decoded row size overflows")?;
        let tile_rows = limits
            .tile_rows
            .min(4096)
            .min(limits.encoded_bytes / row_bytes)
            .min(limits.decoded_bytes / decoded_row);
        ensure!(
            tile_rows > 0,
            "one streamed matrix row exceeds the read/scratch limits"
        );
        Ok(Self {
            locator: tensor.locator(),
            kind: tensor.kind(),
            rows,
            cols,
            row_bytes,
            tile_rows,
            decoded_bytes: limits.decoded_bytes,
        })
    }

    /// Complete matrix row count; the plan allocates no per-row metadata.
    pub fn rows(&self) -> usize {
        self.rows
    }
    /// Number of input coordinates in each row.
    pub fn cols(&self) -> usize {
        self.cols
    }
    /// Exact packed bytes in one row.
    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }
    /// Maximum rows admitted in one upload/submission.
    pub fn tile_rows(&self) -> usize {
        self.tile_rows
    }
    /// Largest possible returned tile slice for this matrix.
    pub fn peak_read_bytes(&self) -> usize {
        self.row_bytes * self.tile_rows.min(self.rows)
    }

    /// Exact source range for a nonempty contiguous matrix row selection.
    pub fn row_range(&self, rows: Range<usize>) -> Result<WeightRange> {
        ensure!(
            rows.start < rows.end && rows.end <= self.rows,
            "streamed row range is empty or outside the matrix"
        );
        let start = rows
            .start
            .checked_mul(self.row_bytes)
            .context("streamed row offset overflows")?;
        let end = rows
            .end
            .checked_mul(self.row_bytes)
            .context("streamed row end overflows")?;
        self.locator
            .range(u64::try_from(start)?..u64::try_from(end)?)
    }

    fn validate_source(
        &self,
        source: &dyn WeightSource,
        device: &dyn ExecutionDevice,
    ) -> Result<()> {
        ensure!(
            source.checkpoint_id() == self.locator.checkpoint_id(),
            "streamed matrix belongs to a different checkpoint"
        );
        ensure!(
            device.supports(self.kind),
            "{} refuses streamed {} weights",
            device.name(),
            self.kind.name()
        );
        Ok(())
    }

    /// Execute contiguous source rows into an exact-length destination view.
    ///
    /// Input/output aliasing is refused for the whole operation: snapshotting one
    /// tile cannot preserve an input overwritten by an earlier tile. Owner checks
    /// remain the device's responsibility at recording, before submission. Basic
    /// geometry, source identity, limits and aliases refuse before a source read.
    /// Cancellation runs before each read, upload and submit, and after each fence
    /// and retirement. `ScaledAdd` applies the same scale independently to each row.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_rows(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        rows: Range<usize>,
        input: &F32TensorView,
        output: &F32TensorView,
        mode: LinearMode,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        self.row_range(rows.clone())?;
        self.validate_source(source, device)?;
        ensure!(
            input.len() == self.cols && output.len() == rows.end - rows.start,
            "streamed input/output shape differs from the selected matrix rows"
        );
        ensure!(
            !input.overlaps(output),
            "streamed linear input and output must not overlap"
        );
        ensure!(
            output
                .len()
                .checked_mul(4)
                .is_some_and(|n| n <= self.decoded_bytes),
            "streamed result exceeds the decoded output limit"
        );
        if let LinearMode::ScaledAdd { scale } = mode {
            ensure!(
                scale.is_finite(),
                "streamed scaled-add multiplier must be finite"
            );
        }
        let mut executor = DeviceTiles {
            device,
            kind: self.kind,
            cols: self.cols,
            operation: Operation::Linear {
                input,
                output,
                mode,
            },
        };
        self.run_tiles(
            source,
            rows,
            WeightAccess::Cyclic,
            &mut executor,
            stats,
            control,
        )
    }

    /// Reuse one uploaded/prepared tile across B1/B2/B4 scalar matvecs.
    ///
    /// No arithmetic kernel or reduction order changes. All destinations must be
    /// disjoint from one another and from every input, including other tokens'
    /// inputs. Input/input overlap is harmless. Geometry and alias refusals occur
    /// before reads; the backend still checks execution ownership at recording.
    /// There is at most one live recording/submission. Encoded/prepared handles
    /// remain live across applications of one tile and retire before the next read.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_rows_chunk(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        rows: Range<usize>,
        inputs: &[F32TensorView],
        outputs: &[F32TensorView],
        mode: LinearMode,
        stats: &mut StreamStats,
        chunk_stats: &mut ChunkStreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        let size = PrefillChunkSize::new(inputs.len())?;
        ensure!(
            outputs.len() == size.tokens(),
            "streamed chunk input/output counts differ"
        );
        self.row_range(rows.clone())?;
        self.validate_source(source, device)?;
        let output_len = rows.end - rows.start;
        ensure!(
            output_len
                .checked_mul(4)
                .is_some_and(|n| n <= self.decoded_bytes),
            "streamed chunk result exceeds the per-token decoded output limit"
        );
        for input in inputs {
            ensure!(
                input.len() == self.cols,
                "streamed chunk input shape differs"
            );
        }
        for (index, output) in outputs.iter().enumerate() {
            ensure!(
                output.len() == output_len,
                "streamed chunk output shape differs"
            );
            ensure!(
                inputs.iter().all(|input| !input.overlaps(output))
                    && outputs[..index].iter().all(|other| !other.overlaps(output)),
                "streamed chunk destinations overlap an input or another destination"
            );
        }
        if let LinearMode::ScaledAdd { scale } = mode {
            ensure!(
                scale.is_finite(),
                "streamed scaled-add multiplier must be finite"
            );
        }
        let mut executor = ChunkDeviceTiles {
            device,
            kind: self.kind,
            cols: self.cols,
            inputs,
            outputs,
            mode,
        };
        self.run_chunk_tiles(
            source,
            rows,
            size,
            &mut executor,
            stats,
            chunk_stats,
            control,
        )
    }

    /// Read and decode only the selected row, using `row=0` in its uploaded tile.
    /// The caller owns the exact-width resident destination; no host decode occurs.
    pub fn embedding_row(
        &self,
        device: &dyn ExecutionDevice,
        source: &dyn WeightSource,
        row: usize,
        output: &F32TensorView,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        ensure!(
            row < self.rows,
            "streamed embedding token is outside the vocabulary"
        );
        self.validate_source(source, device)?;
        ensure!(
            self.cols <= 5120,
            "streamed embedding width exceeds the native row-decode limit"
        );
        ensure!(
            output.len() == self.cols,
            "streamed embedding output width differs"
        );
        let mut executor = DeviceTiles {
            device,
            kind: self.kind,
            cols: self.cols,
            operation: Operation::Embedding { output },
        };
        self.run_tiles(
            source,
            row..row + 1,
            WeightAccess::Demand,
            &mut executor,
            stats,
            control,
        )
    }

    fn run_tiles(
        &self,
        source: &dyn WeightSource,
        rows: Range<usize>,
        access: WeightAccess,
        executor: &mut impl TileExecutor,
        stats: &mut StreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        for first in (rows.start..rows.end).step_by(self.tile_rows) {
            let count = self.tile_rows.min(rows.end - first);
            let range = self.row_range(first..first + count)?;
            control().context("checking streamed control before read")?;
            stats.read_attempts = added(stats.read_attempts, 1, "read")?;
            let resident = source.load_range(&range, access)?;
            let actual = resident.bytes().len();
            stats.returned_read_bytes = added(stats.returned_read_bytes, actual, "returned bytes")?;
            stats.peak_read_bytes = stats.peak_read_bytes.max(u64::try_from(actual)?);
            ensure!(
                u64::try_from(actual)? == range.nbytes(),
                "source returned an incorrect streamed tile span"
            );
            control().context("checking streamed control before upload")?;
            let uploaded = added(stats.uploaded_bytes, actual, "uploaded bytes")?;
            let encoded = executor.upload(resident.bytes(), count)?;
            stats.uploaded_bytes = uploaded;
            drop(resident);
            let prepared = executor.prepare(&encoded)?;
            let recording = executor.record(&prepared, first - rows.start, count)?;
            control().context("checking streamed control before submit")?;
            let completed = added(stats.completed_tiles, 1, "completed tile")?;
            stats.submission_attempts = added(stats.submission_attempts, 1, "submission")?;
            let submission = executor.submit(&recording)?;
            executor.wait(&submission)?;
            // Waiting does not release ownership: submissions retain batches,
            // batches/commands retain prepared weights, and encoded handles also
            // own the arena range. Retire every one before cancellation/next I/O.
            drop(submission);
            drop(recording);
            drop(prepared);
            drop(encoded);
            stats.completed_tiles = completed;
            control().context("checking streamed control after fence and retirement")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunk_tiles(
        &self,
        source: &dyn WeightSource,
        rows: Range<usize>,
        size: PrefillChunkSize,
        executor: &mut impl TileExecutor,
        stats: &mut StreamStats,
        chunk_stats: &mut ChunkStreamStats,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        for first in (rows.start..rows.end).step_by(self.tile_rows) {
            let count = self.tile_rows.min(rows.end - first);
            let range = self.row_range(first..first + count)?;
            control().context("checking chunk control before read")?;
            stats.read_attempts = added(stats.read_attempts, 1, "read")?;
            let resident = source.load_range(&range, WeightAccess::Cyclic)?;
            let actual = resident.bytes().len();
            stats.returned_read_bytes = added(stats.returned_read_bytes, actual, "returned bytes")?;
            stats.peak_read_bytes = stats.peak_read_bytes.max(u64::try_from(actual)?);
            ensure!(
                u64::try_from(actual)? == range.nbytes(),
                "source returned an incorrect streamed tile span"
            );
            control().context("checking chunk control before upload")?;
            let uploaded = added(stats.uploaded_bytes, actual, "uploaded bytes")?;
            let encoded = executor.upload(resident.bytes(), count)?;
            stats.uploaded_bytes = uploaded;
            drop(resident);
            let prepared = executor.prepare(&encoded)?;
            let completed = added(stats.completed_tiles, 1, "completed tile")?;
            for application in 0..size.tokens() {
                control().context("checking chunk control before application")?;
                let recording = executor.record_application(
                    &prepared,
                    first - rows.start,
                    count,
                    application,
                )?;
                control().context("checking chunk control before submit")?;
                let applied = added(
                    chunk_stats.completed_applications,
                    1,
                    "completed application",
                )?;
                stats.submission_attempts = added(stats.submission_attempts, 1, "submission")?;
                let submission = executor.submit(&recording)?;
                executor.wait(&submission)?;
                drop(submission);
                drop(recording);
                chunk_stats.completed_applications = applied;
                // A completed source tile remains visible even if cancellation
                // fires at the final post-fence boundary.
                if application + 1 == size.tokens() {
                    stats.completed_tiles = completed;
                }
                control()
                    .context("checking chunk control after application fence and retirement")?;
            }
            drop(prepared);
            drop(encoded);
            control().context("checking chunk control after tile retirement")?;
        }
        Ok(())
    }
}

// Small private adapter allows lifetime/failure tests without unsealing the
// execution backend API or introducing a production mock/fallback backend.
trait TileExecutor {
    type Encoded;
    type Prepared;
    type Recording;
    type Submission;
    fn upload(&mut self, bytes: &[u8], rows: usize) -> Result<Self::Encoded>;
    fn prepare(&mut self, encoded: &Self::Encoded) -> Result<Self::Prepared>;
    fn record(
        &mut self,
        prepared: &Self::Prepared,
        offset: usize,
        rows: usize,
    ) -> Result<Self::Recording>;
    fn record_application(
        &mut self,
        prepared: &Self::Prepared,
        offset: usize,
        rows: usize,
        _application: usize,
    ) -> Result<Self::Recording> {
        self.record(prepared, offset, rows)
    }
    fn submit(&mut self, recording: &Self::Recording) -> Result<Self::Submission>;
    fn wait(&mut self, submission: &Self::Submission) -> Result<()>;
}

#[derive(Clone, Copy)]
enum Operation<'a> {
    Linear {
        input: &'a F32TensorView,
        output: &'a F32TensorView,
        mode: LinearMode,
    },
    Embedding {
        output: &'a F32TensorView,
    },
}

struct DeviceTiles<'a> {
    device: &'a dyn ExecutionDevice,
    kind: GgmlType,
    cols: usize,
    operation: Operation<'a>,
}

impl TileExecutor for DeviceTiles<'_> {
    type Encoded = EncodedWeight;
    type Prepared = PreparedWeight;
    type Recording = (CommandList, RecordedBatch);
    type Submission = Submission;
    fn upload(&mut self, bytes: &[u8], rows: usize) -> Result<EncodedWeight> {
        Ok(self.device.upload_weight(&Weight {
            bytes,
            kind: self.kind,
            rows,
            cols: self.cols,
        })?)
    }
    fn prepare(&mut self, encoded: &EncodedWeight) -> Result<PreparedWeight> {
        Ok(self.device.prepare_weight(encoded)?)
    }
    fn record(
        &mut self,
        prepared: &PreparedWeight,
        offset: usize,
        rows: usize,
    ) -> Result<Self::Recording> {
        let mut commands = CommandList::new();
        match self.operation {
            Operation::Linear {
                input,
                output,
                mode,
            } => {
                commands.linear(&output.view(offset, rows)?, input, prepared, mode)?;
            }
            Operation::Embedding { output } => {
                commands.embedding_row(output, prepared, 0)?;
            }
        }
        let batch = self.device.record(&commands)?;
        Ok((commands, batch))
    }
    fn submit(&mut self, recording: &Self::Recording) -> Result<Submission> {
        Ok(self.device.submit(&recording.1)?)
    }
    fn wait(&mut self, submission: &Submission) -> Result<()> {
        Ok(self.device.wait(submission)?)
    }
}

struct ChunkDeviceTiles<'a> {
    device: &'a dyn ExecutionDevice,
    kind: GgmlType,
    cols: usize,
    inputs: &'a [F32TensorView],
    outputs: &'a [F32TensorView],
    mode: LinearMode,
}

impl TileExecutor for ChunkDeviceTiles<'_> {
    type Encoded = EncodedWeight;
    type Prepared = PreparedWeight;
    type Recording = (CommandList, RecordedBatch);
    type Submission = Submission;

    fn upload(&mut self, bytes: &[u8], rows: usize) -> Result<EncodedWeight> {
        Ok(self.device.upload_weight(&Weight {
            bytes,
            kind: self.kind,
            rows,
            cols: self.cols,
        })?)
    }
    fn prepare(&mut self, encoded: &EncodedWeight) -> Result<PreparedWeight> {
        Ok(self.device.prepare_weight(encoded)?)
    }
    fn record(
        &mut self,
        prepared: &PreparedWeight,
        offset: usize,
        rows: usize,
    ) -> Result<Self::Recording> {
        self.record_application(prepared, offset, rows, 0)
    }
    fn record_application(
        &mut self,
        prepared: &PreparedWeight,
        offset: usize,
        rows: usize,
        application: usize,
    ) -> Result<Self::Recording> {
        let mut commands = CommandList::new();
        commands.linear(
            &self.outputs[application].view(offset, rows)?,
            &self.inputs[application],
            prepared,
            self.mode,
        )?;
        let batch = self.device.record(&commands)?;
        Ok((commands, batch))
    }
    fn submit(&mut self, recording: &Self::Recording) -> Result<Submission> {
        Ok(self.device.submit(&recording.1)?)
    }
    fn wait(&mut self, submission: &Submission) -> Result<()> {
        Ok(self.device.wait(submission)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beholdd_backend::execution::CpuExecutionDevice;
    use beholdd_format::{CheckpointId, Gguf};
    use beholdd_memory::ResidentWeight;
    use std::{
        fs,
        path::PathBuf,
        sync::{
            atomic::{AtomicU64, AtomicUsize, Ordering},
            Arc, Mutex,
        },
    };

    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(kind: u32, cols: usize, rows: usize) -> (Temporary, Gguf) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen35-recorded-weights-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let mut bytes = b"GGUF".to_vec();
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.push(b'w');
        bytes.extend(2u32.to_le_bytes());
        bytes.extend((cols as u64).to_le_bytes());
        bytes.extend((rows as u64).to_le_bytes());
        bytes.extend(kind.to_le_bytes());
        bytes.extend(0u64.to_le_bytes());
        bytes.resize((bytes.len() + 31) & !31, 0);
        for value in 1..=rows {
            if kind == 0 {
                bytes.extend(vec![0; cols * 4]);
                continue;
            }
            for _ in 0..cols / 256 {
                let value = value as u8;
                let mut block = vec![0; if kind == 12 { 144 } else { 210 }];
                if kind == 12 {
                    block[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                    block[4..8].fill(1);
                    block[12..16].fill(1);
                    block[16..].fill(value | (value << 4));
                } else {
                    block[..128].fill(value | (value << 4));
                    block[128..192].fill(0xaa);
                    block[192..208].fill(1);
                    block[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
                }
                bytes.extend(block);
            }
        }
        let path = dir.join("test.gguf");
        fs::write(&path, bytes).unwrap();
        (Temporary(dir), Gguf::open(path).unwrap())
    }

    struct Source<'a> {
        checkpoint: &'a Gguf,
        reads: Mutex<Vec<(WeightRange, WeightAccess)>>,
        live: Option<Arc<AtomicUsize>>,
        short: bool,
    }
    impl<'a> Source<'a> {
        fn new(checkpoint: &'a Gguf) -> Self {
            Self {
                checkpoint,
                reads: Mutex::new(Vec::new()),
                live: None,
                short: false,
            }
        }
    }
    impl WeightSource for Source<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.checkpoint.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            if let Some(live) = &self.live {
                assert_eq!(live.load(Ordering::SeqCst), 0);
            }
            self.reads.lock().unwrap().push((*range, access));
            if self.short {
                let short = range.locator().range(0..1)?;
                ResidentWeight::load_range(self.checkpoint, &short)
            } else {
                ResidentWeight::load_range(self.checkpoint, range)
            }
        }
    }

    fn limits(kind: u32) -> ReadLimits {
        ReadLimits {
            encoded_bytes: ggml_block_geometry(kind).unwrap().1 * 2,
            decoded_bytes: 2048,
            tile_rows: 64,
        }
    }

    #[test]
    fn partial_rows_overwrite_scaled_add_and_embedding_keep_surrounding_views() -> Result<()> {
        for kind in [12, 14] {
            let (_temporary, checkpoint) = fixture(kind, 256, 5);
            let source = Source::new(&checkpoint);
            let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(kind))?;
            assert_eq!((plan.rows(), plan.cols(), plan.tile_rows()), (5, 256, 2));
            let device = CpuExecutionDevice::new();
            let input = device.upload_f32(&[1.; 256])?;
            let output = device.upload_f32(&[-7.; 5])?;
            let mut stats = StreamStats::default();
            plan.execute_rows(
                &device,
                &source,
                1..4,
                &input.full_view(),
                &output.view(1, 3)?,
                LinearMode::Overwrite,
                &mut stats,
                &mut || Ok(()),
            )?;
            let mut values = [0.; 5];
            device.read_f32(&output.full_view(), &mut values)?;
            assert_eq!(values, [-7., 512., 768., 1024., -7.]);
            assert_eq!(
                (
                    stats.read_attempts,
                    stats.submission_attempts,
                    stats.completed_tiles
                ),
                (2, 2, 2)
            );
            assert_eq!(stats.returned_read_bytes, (3 * plan.row_bytes()) as u64);
            assert_eq!(stats.uploaded_bytes, stats.returned_read_bytes);
            assert_eq!(stats.peak_read_bytes, plan.peak_read_bytes() as u64);
            let reads = source.reads.lock().unwrap();
            assert_eq!(reads[0], (plan.row_range(1..3)?, WeightAccess::Cyclic));
            assert_eq!(reads[1], (plan.row_range(3..4)?, WeightAccess::Cyclic));
            drop(reads);
            plan.execute_rows(
                &device,
                &source,
                1..4,
                &input.full_view(),
                &output.view(1, 3)?,
                LinearMode::ScaledAdd { scale: -0.5 },
                &mut stats,
                &mut || Ok(()),
            )?;
            device.read_f32(&output.full_view(), &mut values)?;
            assert_eq!(values, [-7., 256., 384., 512., -7.]);
            let embedding = device.upload_f32(&[-7.; 258])?;
            plan.embedding_row(
                &device,
                &source,
                4,
                &embedding.view(1, 256)?,
                &mut stats,
                &mut || Ok(()),
            )?;
            let mut row = vec![0.; 258];
            device.read_f32(&embedding.full_view(), &mut row)?;
            assert_eq!(row[0], -7.);
            assert_eq!(row[257], -7.);
            assert_eq!(&row[1..257], &[5.; 256]);
            assert_eq!(
                source.reads.lock().unwrap().last(),
                Some(&(plan.row_range(4..5)?, WeightAccess::Demand))
            );
        }
        Ok(())
    }

    #[test]
    fn shape_alias_source_and_limit_refusals_precede_reads() -> Result<()> {
        let (_temporary, checkpoint) = fixture(12, 256, 5);
        let source = Source::new(&checkpoint);
        let tensor = checkpoint.get("w").unwrap();
        let plan = PackedRowPlan::new(tensor, limits(12))?;
        for bad in [
            ReadLimits {
                encoded_bytes: 143,
                ..limits(12)
            },
            ReadLimits {
                decoded_bytes: 1023,
                ..limits(12)
            },
            ReadLimits {
                tile_rows: 0,
                ..limits(12)
            },
        ] {
            assert!(PackedRowPlan::new(tensor, bad).is_err());
        }
        let device = CpuExecutionDevice::new();
        let input = device.allocate_f32(256)?;
        let output = device.allocate_f32(5)?;
        let mut stats = StreamStats::default();
        for rows in [0..0, 0..6, Range { start: 4, end: 3 }] {
            assert!(plan
                .execute_rows(
                    &device,
                    &source,
                    rows,
                    &input.full_view(),
                    &output.full_view(),
                    LinearMode::Overwrite,
                    &mut stats,
                    &mut || Ok(())
                )
                .is_err());
        }
        assert!(plan
            .execute_rows(
                &device,
                &source,
                0..5,
                &input.full_view(),
                &input.view(0, 5)?,
                LinearMode::Overwrite,
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert!(plan
            .execute_rows(
                &device,
                &source,
                0..5,
                &input.full_view(),
                &output.full_view(),
                LinearMode::ScaledAdd { scale: f32::NAN },
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert!(plan
            .embedding_row(
                &device,
                &source,
                5,
                &input.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        let (_other, other) = fixture(12, 256, 5);
        let foreign = Source::new(&other);
        assert!(plan
            .embedding_row(
                &device,
                &foreign,
                0,
                &input.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        let (_f32, f32_checkpoint) = fixture(0, 256, 1);
        assert!(PackedRowPlan::new(f32_checkpoint.get("w").unwrap(), limits(12)).is_err());
        let unrestricted = ReadLimits {
            encoded_bytes: usize::MAX,
            decoded_bytes: usize::MAX,
            tile_rows: usize::MAX,
        };
        assert_eq!(PackedRowPlan::new(tensor, unrestricted)?.tile_rows(), 4096);
        let (_wide, wide_checkpoint) = fixture(12, 5376, 1);
        let wide_plan = PackedRowPlan::new(wide_checkpoint.get("w").unwrap(), unrestricted)?;
        let wide_source = Source::new(&wide_checkpoint);
        let wide_output = device.allocate_f32(5376)?;
        assert!(wide_plan
            .embedding_row(
                &device,
                &wide_source,
                0,
                &wide_output.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert!(wide_source.reads.lock().unwrap().is_empty());
        let (_too_wide, too_wide) = fixture(12, 17664, 1);
        assert!(PackedRowPlan::new(too_wide.get("w").unwrap(), unrestricted).is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        assert!(foreign.reads.lock().unwrap().is_empty());
        assert_eq!(stats, StreamStats::default());
        Ok(())
    }

    struct Allocation(Arc<AtomicUsize>);
    impl Drop for Allocation {
        fn drop(&mut self) {
            self.0.fetch_sub(1, Ordering::SeqCst);
        }
    }
    #[derive(Clone)]
    struct Lease(Arc<Allocation>);
    #[derive(Clone)]
    struct Recording {
        _commands: Lease,
        _batch: Lease,
    }
    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Fail {
        None,
        Upload,
        Prepare,
        Record,
        Submit,
        Wait,
    }
    struct Fake {
        live: Arc<AtomicUsize>,
        fail: Fail,
    }
    impl TileExecutor for Fake {
        type Encoded = Lease;
        type Prepared = Lease;
        type Recording = Recording;
        type Submission = Recording;
        fn upload(&mut self, _: &[u8], _: usize) -> Result<Lease> {
            ensure!(self.fail != Fail::Upload, "upload failed");
            assert_eq!(self.live.fetch_add(1, Ordering::SeqCst), 0);
            Ok(Lease(Arc::new(Allocation(self.live.clone()))))
        }
        fn prepare(&mut self, weight: &Lease) -> Result<Lease> {
            ensure!(self.fail != Fail::Prepare, "prepare failed");
            Ok(weight.clone())
        }
        fn record(&mut self, weight: &Lease, _: usize, _: usize) -> Result<Recording> {
            ensure!(self.fail != Fail::Record, "record failed");
            Ok(Recording {
                _commands: weight.clone(),
                _batch: weight.clone(),
            })
        }
        fn submit(&mut self, recording: &Recording) -> Result<Recording> {
            ensure!(self.fail != Fail::Submit, "submit failed");
            Ok(recording.clone())
        }
        fn wait(&mut self, recording: &Recording) -> Result<()> {
            assert!(Arc::strong_count(&recording._batch.0) >= 5);
            ensure!(self.fail != Fail::Wait, "wait failed");
            Ok(())
        }
    }

    #[test]
    fn every_handle_retires_before_the_next_read_and_after_fence_control() -> Result<()> {
        let (_temporary, checkpoint) = fixture(12, 256, 5);
        let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(12))?;
        let live = Arc::new(AtomicUsize::new(0));
        let mut source = Source::new(&checkpoint);
        source.live = Some(live.clone());
        let mut fake = Fake {
            live: live.clone(),
            fail: Fail::None,
        };
        let mut stats = StreamStats::default();
        let mut calls = 0;
        plan.run_tiles(
            &source,
            0..5,
            WeightAccess::Cyclic,
            &mut fake,
            &mut stats,
            &mut || {
                calls += 1;
                if calls % 4 == 0 {
                    assert_eq!(live.load(Ordering::SeqCst), 0);
                }
                Ok(())
            },
        )?;
        assert_eq!(calls, 12);
        assert_eq!(stats.completed_tiles, 3);
        assert_eq!(live.load(Ordering::SeqCst), 0);
        for fail in [
            Fail::Upload,
            Fail::Prepare,
            Fail::Record,
            Fail::Submit,
            Fail::Wait,
        ] {
            let mut fake = Fake {
                live: live.clone(),
                fail,
            };
            let mut stats = StreamStats::default();
            assert!(plan
                .run_tiles(
                    &source,
                    0..5,
                    WeightAccess::Cyclic,
                    &mut fake,
                    &mut stats,
                    &mut || Ok(())
                )
                .is_err());
            assert_eq!(stats.read_attempts, 1);
            assert_eq!(stats.completed_tiles, 0);
            assert_eq!(live.load(Ordering::SeqCst), 0);
        }
        Ok(())
    }

    #[test]
    fn cancellation_progress_and_incorrect_source_span_are_visible() -> Result<()> {
        let (_temporary, checkpoint) = fixture(14, 256, 5);
        let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(14))?;
        let device = CpuExecutionDevice::new();
        let input = device.upload_f32(&[1.; 256])?;
        for stop in 1..=5 {
            let source = Source::new(&checkpoint);
            let output = device.upload_f32(&[-7.; 5])?;
            let mut stats = StreamStats::default();
            let mut calls = 0;
            assert!(plan
                .execute_rows(
                    &device,
                    &source,
                    0..5,
                    &input.full_view(),
                    &output.full_view(),
                    LinearMode::Overwrite,
                    &mut stats,
                    &mut || {
                        calls += 1;
                        ensure!(calls != stop, "cancelled");
                        Ok(())
                    }
                )
                .is_err());
            assert_eq!(stats.read_attempts, u64::from(stop >= 2));
            assert_eq!(stats.uploaded_bytes, if stop >= 3 { 420 } else { 0 });
            assert_eq!(stats.completed_tiles, u64::from(stop >= 4));
            let mut values = [0.; 5];
            device.read_f32(&output.full_view(), &mut values)?;
            assert_eq!(
                values,
                if stop >= 4 {
                    [256., 512., -7., -7., -7.]
                } else {
                    [-7.; 5]
                }
            );
        }
        let mut source = Source::new(&checkpoint);
        source.short = true;
        let mut stats = StreamStats::default();
        assert!(plan
            .embedding_row(
                &device,
                &source,
                2,
                &input.full_view(),
                &mut stats,
                &mut || Ok(())
            )
            .is_err());
        assert_eq!(
            (
                stats.read_attempts,
                stats.returned_read_bytes,
                stats.peak_read_bytes
            ),
            (1, 1, 1)
        );
        assert_eq!(
            (
                stats.uploaded_bytes,
                stats.submission_attempts,
                stats.completed_tiles
            ),
            (0, 0, 0)
        );
        Ok(())
    }

    #[test]
    fn chunk_rows_reuse_each_tile_for_b1_b2_b4_with_exact_scalar_results() -> Result<()> {
        for kind in [12, 14] {
            for count in [1, 2, 4] {
                for mode in [LinearMode::Overwrite, LinearMode::ScaledAdd { scale: 0.5 }] {
                    let (_temporary, checkpoint) = fixture(kind, 256, 5);
                    let source = Source::new(&checkpoint);
                    let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(kind))?;
                    let device = CpuExecutionDevice::new();
                    let mut input_owners = Vec::new();
                    let mut output_owners = Vec::new();
                    for token in 0..count {
                        input_owners.push(device.upload_f32(&vec![token as f32 + 0.5; 256])?);
                        output_owners.push(device.upload_f32(&[-7.; 7])?);
                    }
                    let inputs: Vec<_> = input_owners.iter().map(|t| t.full_view()).collect();
                    let outputs: Vec<_> = output_owners
                        .iter()
                        .map(|t| t.view(1, 5))
                        .collect::<std::result::Result<_, _>>()?;
                    let mut stats = StreamStats::default();
                    let mut chunks = ChunkStreamStats::default();
                    plan.execute_rows_chunk(
                        &device,
                        &source,
                        0..5,
                        &inputs,
                        &outputs,
                        mode,
                        &mut stats,
                        &mut chunks,
                        &mut || Ok(()),
                    )?;
                    for (token, output) in output_owners.iter().enumerate() {
                        let mut actual = [0.; 7];
                        device.read_f32(&output.full_view(), &mut actual)?;
                        assert_eq!((actual[0], actual[6]), (-7., -7.));
                        for row in 0..5 {
                            let dot = 256. * (row + 1) as f32 * (token as f32 + 0.5);
                            let expected = match mode {
                                LinearMode::Overwrite => dot,
                                LinearMode::ScaledAdd { scale } => -7. + scale * dot,
                                _ => panic!("test selected an unsupported linear mode"),
                            };
                            assert_eq!(actual[row + 1], expected);
                        }
                    }
                    assert_eq!(source.reads.lock().unwrap().len(), 3);
                    assert_eq!((stats.read_attempts, stats.completed_tiles), (3, 3));
                    assert_eq!(stats.uploaded_bytes, (5 * plan.row_bytes()) as u64);
                    assert_eq!(stats.returned_read_bytes, stats.uploaded_bytes);
                    assert_eq!(stats.submission_attempts, 3 * count as u64);
                    assert_eq!(chunks.completed_applications, 3 * count as u64);
                }
            }
        }
        Ok(())
    }

    #[test]
    fn chunk_alias_count_and_extent_refusals_precede_source_reads() -> Result<()> {
        let (_temporary, checkpoint) = fixture(12, 256, 5);
        let source = Source::new(&checkpoint);
        let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(12))?;
        let device = CpuExecutionDevice::new();
        let a = device.upload_f32(&[1.; 256])?;
        let b = device.upload_f32(&[2.; 256])?;
        let out = device.allocate_f32(10)?;
        let good_inputs = [a.full_view(), b.full_view()];
        for (inputs, outputs) in [
            (Vec::new(), Vec::new()),
            (vec![a.full_view(); 3], vec![out.view(0, 5)?; 3]),
            (good_inputs.to_vec(), vec![out.view(0, 5)?]),
            (good_inputs.to_vec(), vec![out.view(0, 5)?, out.view(0, 5)?]),
            (good_inputs.to_vec(), vec![b.view(0, 5)?, out.view(0, 5)?]),
            (
                vec![a.view(0, 255)?, b.full_view()],
                vec![out.view(0, 5)?, out.view(5, 5)?],
            ),
        ] {
            let mut stats = StreamStats::default();
            let mut chunks = ChunkStreamStats::default();
            assert!(plan
                .execute_rows_chunk(
                    &device,
                    &source,
                    0..5,
                    &inputs,
                    &outputs,
                    LinearMode::Overwrite,
                    &mut stats,
                    &mut chunks,
                    &mut || panic!("invalid chunk reached control")
                )
                .is_err());
            assert_eq!(stats, StreamStats::default());
            assert_eq!(chunks, ChunkStreamStats::default());
        }
        assert!(source.reads.lock().unwrap().is_empty());
        Ok(())
    }

    struct ChunkFake {
        base: Fake,
        applications: Vec<(usize, usize)>,
        fail_application: Option<usize>,
    }
    impl TileExecutor for ChunkFake {
        type Encoded = Lease;
        type Prepared = Lease;
        type Recording = Recording;
        type Submission = Recording;
        fn upload(&mut self, bytes: &[u8], rows: usize) -> Result<Lease> {
            self.base.upload(bytes, rows)
        }
        fn prepare(&mut self, encoded: &Lease) -> Result<Lease> {
            self.base.prepare(encoded)
        }
        fn record(&mut self, prepared: &Lease, offset: usize, rows: usize) -> Result<Recording> {
            self.record_application(prepared, offset, rows, 0)
        }
        fn record_application(
            &mut self,
            prepared: &Lease,
            offset: usize,
            rows: usize,
            application: usize,
        ) -> Result<Recording> {
            // Only encoded + prepared survive from the previous application.
            assert_eq!(Arc::strong_count(&prepared.0), 2);
            self.applications.push((offset, application));
            ensure!(
                self.fail_application != Some(application),
                "selected application refused"
            );
            self.base.record(prepared, offset, rows)
        }
        fn submit(&mut self, recording: &Recording) -> Result<Recording> {
            self.base.submit(recording)
        }
        fn wait(&mut self, submission: &Recording) -> Result<()> {
            self.base.wait(submission)
        }
    }

    #[test]
    fn chunk_retires_each_recording_but_keeps_one_weight_until_all_applications() -> Result<()> {
        let (_temporary, checkpoint) = fixture(12, 256, 5);
        let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(12))?;
        let live = Arc::new(AtomicUsize::new(0));
        let mut source = Source::new(&checkpoint);
        source.live = Some(live.clone());
        let mut executor = ChunkFake {
            base: Fake {
                live: live.clone(),
                fail: Fail::None,
            },
            applications: Vec::new(),
            fail_application: None,
        };
        let mut stats = StreamStats::default();
        let mut chunks = ChunkStreamStats::default();
        plan.run_chunk_tiles(
            &source,
            0..5,
            PrefillChunkSize::new(4)?,
            &mut executor,
            &mut stats,
            &mut chunks,
            &mut || Ok(()),
        )?;
        assert_eq!(
            executor.applications,
            vec![
                (0, 0),
                (0, 1),
                (0, 2),
                (0, 3),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (4, 0),
                (4, 1),
                (4, 2),
                (4, 3)
            ]
        );
        assert_eq!(
            (
                stats.read_attempts,
                stats.completed_tiles,
                chunks.completed_applications
            ),
            (3, 3, 12)
        );
        assert_eq!(live.load(Ordering::SeqCst), 0);
        executor.fail_application = Some(1);
        let mut stats = StreamStats::default();
        let mut chunks = ChunkStreamStats::default();
        assert!(plan
            .run_chunk_tiles(
                &source,
                0..5,
                PrefillChunkSize::new(2)?,
                &mut executor,
                &mut stats,
                &mut chunks,
                &mut || Ok(())
            )
            .is_err());
        assert_eq!(
            (
                stats.read_attempts,
                stats.completed_tiles,
                chunks.completed_applications
            ),
            (1, 0, 1)
        );
        assert_eq!(live.load(Ordering::SeqCst), 0);
        Ok(())
    }

    #[test]
    fn chunk_cancellation_at_every_boundary_retires_all_handles() -> Result<()> {
        let (_temporary, checkpoint) = fixture(14, 256, 5);
        let plan = PackedRowPlan::new(checkpoint.get("w").unwrap(), limits(14))?;
        let live = Arc::new(AtomicUsize::new(0));
        let mut source = Source::new(&checkpoint);
        source.live = Some(live.clone());
        // Per tile: read/upload + three boundaries per application + retirement.
        for stop in 1..=3 * (3 + 3 * 4) {
            let mut executor = ChunkFake {
                base: Fake {
                    live: live.clone(),
                    fail: Fail::None,
                },
                applications: Vec::new(),
                fail_application: None,
            };
            let mut stats = StreamStats::default();
            let mut chunks = ChunkStreamStats::default();
            let mut calls = 0;
            assert!(plan
                .run_chunk_tiles(
                    &source,
                    0..5,
                    PrefillChunkSize::new(4)?,
                    &mut executor,
                    &mut stats,
                    &mut chunks,
                    &mut || {
                        calls += 1;
                        ensure!(calls != stop, "cancelled");
                        Ok(())
                    }
                )
                .is_err());
            assert_eq!(live.load(Ordering::SeqCst), 0);
            assert!(stats.completed_tiles * 4 <= chunks.completed_applications);
            assert!(chunks.completed_applications <= stats.completed_tiles * 4 + 3);
            assert_eq!(stats.submission_attempts, chunks.completed_applications);
        }
        Ok(())
    }
}
