//! Checkpoint readers.
//!
//! Owns: safetensors and GGUF parsing, tensor discovery, checked byte-range resolution,
//! and physical contiguity discovery.
//!
//! Does NOT own: what the tensors MEAN. A reader that knows the word `attention` has
//! reached into `beholdd-model`.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};

pub mod gguf;
pub mod mxfp4;
pub mod safetensors;

pub use gguf::{GgmlType, Gguf, GgufTensor};
pub use mxfp4::Rows as Mxfp4Rows;
pub use safetensors::{SafeTensors, TensorInfo};

static NEXT_CHECKPOINT_ID: AtomicU64 = AtomicU64::new(1);

/// Identity of one opened checkpoint and every weight locator it issued.
///
/// Byte offsets have meaning only inside the checkpoint that produced them. This ID is
/// intentionally opaque: consumers can compare it, but cannot manufacture an identity
/// for a different file with a coincidentally compatible layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointId(u64);

impl CheckpointId {
    pub(crate) fn fresh() -> Result<Self> {
        NEXT_CHECKPOINT_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map(Self)
            .map_err(|_| anyhow::anyhow!("exhausted in-process checkpoint identities"))
    }
}

/// Opaque location of one complete encoded weight inside an opened checkpoint.
///
/// A locator is format-neutral: GGUF and safetensors issue the same type, while model
/// and memory code need know only the checkpoint, shard, and checked byte extent. Its
/// fields are private so callers cannot forge a stale checkpoint identity or invalid
/// extent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightLocator {
    checkpoint_id: CheckpointId,
    shard: usize,
    start: u64,
    end: u64,
}

impl WeightLocator {
    pub(crate) fn new(
        checkpoint_id: CheckpointId,
        shard: usize,
        offset: u64,
        nbytes: u64,
    ) -> Result<Self> {
        let end = offset.checked_add(nbytes).with_context(|| {
            format!("weight byte extent overflows: shard {shard}, offset {offset}, length {nbytes}")
        })?;
        Ok(Self {
            checkpoint_id,
            shard,
            start: offset,
            end,
        })
    }

    /// Checkpoint that issued this locator.
    pub fn checkpoint_id(self) -> CheckpointId {
        self.checkpoint_id
    }

    /// Index in the checkpoint's ordered shard list.
    pub fn shard(self) -> usize {
        self.shard
    }

    /// Absolute start byte within the shard.
    pub fn start(self) -> u64 {
        self.start
    }

    /// Absolute exclusive end byte within the shard.
    pub fn end(self) -> u64 {
        self.end
    }

    /// Encoded byte length.
    pub fn nbytes(self) -> u64 {
        self.end - self.start
    }

    /// Resolve a weight-relative byte range into a checked absolute range.
    pub fn range(self, relative: Range<u64>) -> Result<WeightRange> {
        relative.end.checked_sub(relative.start).with_context(|| {
            format!(
                "invalid weight-relative byte range {}..{}",
                relative.start, relative.end
            )
        })?;
        if relative.end > self.nbytes() {
            bail!(
                "weight-relative byte range {}..{} is outside the {}-byte weight",
                relative.start,
                relative.end,
                self.nbytes()
            );
        }
        let start = self.start.checked_add(relative.start).with_context(|| {
            format!(
                "weight-relative byte range {}..{} overflows shard offset {}",
                relative.start, relative.end, self.start
            )
        })?;
        let end = self.start.checked_add(relative.end).with_context(|| {
            format!(
                "weight-relative byte range {}..{} overflows shard offset {}",
                relative.start, relative.end, self.start
            )
        })?;
        debug_assert!(end <= self.end);
        Ok(WeightRange {
            locator: self,
            start,
            end,
        })
    }

    /// The checked range covering this entire weight.
    pub fn whole_range(self) -> WeightRange {
        WeightRange {
            locator: self,
            start: self.start,
            end: self.end,
        }
    }
}

/// A checked absolute byte range belonging to one [`WeightLocator`].
///
/// The parent locator remains attached so a resident or prefetched subrange can map
/// back to the complete stored weight without guessing from overlapping intervals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WeightRange {
    locator: WeightLocator,
    start: u64,
    end: u64,
}

impl WeightRange {
    /// Complete weight from which this range was resolved.
    pub fn locator(self) -> WeightLocator {
        self.locator
    }

    /// Checkpoint that issued the parent locator.
    pub fn checkpoint_id(self) -> CheckpointId {
        self.locator.checkpoint_id()
    }

    /// Index in the checkpoint's ordered shard list.
    pub fn shard(self) -> usize {
        self.locator.shard()
    }

    /// Absolute start byte within the shard.
    pub fn start(self) -> u64 {
        self.start
    }

    /// Absolute exclusive end byte within the shard.
    pub fn end(self) -> u64 {
        self.end
    }

    /// Byte length of this range.
    pub fn nbytes(self) -> u64 {
        self.end - self.start
    }
}

/// One checked, physically contiguous byte span inside an opened checkpoint shard.
///
/// Unlike [`WeightRange`], a span may cross weight boundaries. It deliberately does
/// not retain a parent locator, so resident-weight lookup must continue to use
/// `WeightRange`. The private fields prevent callers from manufacturing a foreign,
/// cross-shard, gapped, or overlapping span.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CheckpointSpan {
    checkpoint_id: CheckpointId,
    shard: usize,
    start: u64,
    end: u64,
}

impl CheckpointSpan {
    /// Mint a single-range span from a checked, parent-bounded weight range.
    pub fn from_range(range: WeightRange) -> Self {
        Self {
            checkpoint_id: range.checkpoint_id(),
            shard: range.shard(),
            start: range.start(),
            end: range.end(),
        }
    }

    /// Append one range when it starts exactly at this span's exclusive end.
    ///
    /// Append order is significant. A physically earlier range is rejected rather
    /// than sorted, and an empty range is a no-op only when it lies exactly at the
    /// current end. This prevents an empty range from bridging a physical gap.
    pub fn extend(&mut self, next: WeightRange) -> Result<()> {
        *self = self.merge(Self::from_range(next))?;
        Ok(())
    }

    /// Merge an already checked span that starts exactly at this span's exclusive end.
    ///
    /// Both spans must belong to the same checkpoint and shard. Gaps, overlap, and
    /// reverse ordering are errors rather than implicit reordering.
    pub fn merge(self, next: Self) -> Result<Self> {
        if self.checkpoint_id != next.checkpoint_id {
            bail!(
                "cannot merge checkpoint spans from {:?} and {:?}",
                self.checkpoint_id,
                next.checkpoint_id
            );
        }
        if self.shard != next.shard {
            bail!(
                "cannot merge checkpoint spans from shards {} and {}",
                self.shard,
                next.shard
            );
        }
        if next.start < self.start {
            bail!(
                "checkpoint span {}..{} is reverse ordered after {}..{} in shard {}",
                next.start,
                next.end,
                self.start,
                self.end,
                self.shard
            );
        }
        if next.start < self.end {
            bail!(
                "checkpoint span {}..{} overlaps {}..{} in shard {}",
                next.start,
                next.end,
                self.start,
                self.end,
                self.shard
            );
        }
        if next.start > self.end {
            bail!(
                "checkpoint span {}..{} leaves a gap after {}..{} in shard {}",
                next.start,
                next.end,
                self.start,
                self.end,
                self.shard
            );
        }

        let end = checked_merged_end(self.start, self.nbytes(), next.nbytes())?;
        debug_assert_eq!(end, next.end);
        Ok(Self { end, ..self })
    }

    /// Checkpoint that issued every range in this span.
    pub fn checkpoint_id(self) -> CheckpointId {
        self.checkpoint_id
    }

    /// Index in the checkpoint's ordered shard list.
    pub fn shard(self) -> usize {
        self.shard
    }

    /// Absolute start byte within the shard.
    pub fn start(self) -> u64 {
        self.start
    }

    /// Absolute exclusive end byte within the shard.
    pub fn end(self) -> u64 {
        self.end
    }

    /// Byte length of this physical span.
    pub fn nbytes(self) -> u64 {
        self.end - self.start
    }
}

impl From<WeightRange> for CheckpointSpan {
    fn from(range: WeightRange) -> Self {
        Self::from_range(range)
    }
}

fn checked_merged_end(start: u64, current_nbytes: u64, next_nbytes: u64) -> Result<u64> {
    let merged_nbytes = current_nbytes
        .checked_add(next_nbytes)
        .context("merged checkpoint span byte length overflows u64")?;
    start
        .checked_add(merged_nbytes)
        .context("merged checkpoint span end overflows u64")
}

/// Concurrent-safe positional byte-range reader for an opened checkpoint.
///
/// Implementations perform exact positional reads rather than mutating a shared file
/// cursor. A single reader may therefore serve demand loads and prefetches concurrently.
/// Every read must reject spans issued by another checkpoint, missing shards, extents
/// beyond the recorded shard length, buffer-size mismatches, and short underlying reads.
pub trait RangeReader: Send + Sync {
    /// Identity shared by this reader and its locators.
    fn checkpoint_id(&self) -> CheckpointId;

    /// Every complete weight locator issued by this checkpoint.
    fn locators(&self) -> Box<dyn Iterator<Item = WeightLocator> + '_>;

    /// Number of opened shard files.
    fn shard_count(&self) -> usize;

    /// Recorded length of one shard, or `None` when the index is invalid.
    fn shard_len(&self, shard: usize) -> Option<u64>;

    /// Refuse a foreign, missing-shard or out-of-bounds range without allocating or
    /// reading payload bytes.
    fn validate_range(&self, range: &WeightRange) -> Result<()> {
        validate_range_bounds(self, range)
    }

    /// Refuse a foreign, missing-shard or out-of-bounds physical span without
    /// allocating or reading payload bytes.
    fn validate_span(&self, span: &CheckpointSpan) -> Result<()> {
        validate_span_bounds(self, span)
    }

    /// Read exactly `span.nbytes()` contiguous bytes into an equally sized buffer.
    fn read_span_into(&self, span: &CheckpointSpan, buf: &mut [u8]) -> Result<()>;

    /// Read exactly `range.nbytes()` into an equally sized buffer.
    ///
    /// A single weight range is also a single physical span, so existing range reads
    /// delegate to the one required positional-read primitive.
    fn read_range_into(&self, range: &WeightRange, buf: &mut [u8]) -> Result<()> {
        validate_range_read(self, range, buf)?;
        self.read_span_into(&CheckpointSpan::from_range(*range), buf)
    }
}

fn validate_range_bounds<R: RangeReader + ?Sized>(reader: &R, range: &WeightRange) -> Result<()> {
    if range.checkpoint_id() != reader.checkpoint_id() {
        bail!(
            "weight range belongs to checkpoint {:?}, not source checkpoint {:?}",
            range.checkpoint_id(),
            reader.checkpoint_id()
        );
    }
    let Some(shard_len) = reader.shard_len(range.shard()) else {
        bail!(
            "weight range names shard {}, checkpoint has {} shards",
            range.shard(),
            reader.shard_count()
        );
    };
    if range.end() > shard_len {
        bail!(
            "weight range {}..{} is outside shard {}, which is {shard_len} bytes",
            range.start(),
            range.end(),
            range.shard()
        );
    }
    Ok(())
}

fn validate_span_bounds<R: RangeReader + ?Sized>(reader: &R, span: &CheckpointSpan) -> Result<()> {
    if span.end() < span.start() {
        bail!(
            "checkpoint span has reversed bounds {}..{} in shard {}",
            span.start(),
            span.end(),
            span.shard()
        );
    }
    if span.checkpoint_id() != reader.checkpoint_id() {
        bail!(
            "checkpoint span belongs to checkpoint {:?}, not source checkpoint {:?}",
            span.checkpoint_id(),
            reader.checkpoint_id()
        );
    }
    let Some(shard_len) = reader.shard_len(span.shard()) else {
        bail!(
            "checkpoint span names shard {}, checkpoint has {} shards",
            span.shard(),
            reader.shard_count()
        );
    };
    if span.end() > shard_len {
        bail!(
            "checkpoint span {}..{} is outside shard {}, which is {shard_len} bytes",
            span.start(),
            span.end(),
            span.shard()
        );
    }
    Ok(())
}

pub(crate) fn validate_range_read<R: RangeReader + ?Sized>(
    reader: &R,
    range: &WeightRange,
    buf: &[u8],
) -> Result<()> {
    reader.validate_range(range)?;
    let buf_len = u64::try_from(buf.len()).context("buffer length does not fit in u64")?;
    if buf_len != range.nbytes() {
        bail!(
            "buffer is {buf_len} bytes, but weight range {}..{} in shard {} is {} bytes",
            range.start(),
            range.end(),
            range.shard(),
            range.nbytes()
        );
    }
    Ok(())
}

pub(crate) fn validate_span_read<R: RangeReader + ?Sized>(
    reader: &R,
    span: &CheckpointSpan,
    buf: &[u8],
) -> Result<()> {
    reader.validate_span(span)?;
    let buf_len = u64::try_from(buf.len()).context("buffer length does not fit in u64")?;
    if buf_len != span.nbytes() {
        bail!(
            "buffer is {buf_len} bytes, but checkpoint span {}..{} in shard {} is {} bytes",
            span.start(),
            span.end(),
            span.shard(),
            span.nbytes()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locator_resolves_checked_ranges_and_retains_parent() {
        let locator = WeightLocator::new(CheckpointId::fresh().unwrap(), 3, 100, 20).unwrap();
        let range = locator.range(4..11).unwrap();
        assert_eq!(range.locator(), locator);
        assert_eq!(range.shard(), 3);
        assert_eq!(range.start(), 104);
        assert_eq!(range.end(), 111);
        assert_eq!(range.nbytes(), 7);
        assert_eq!(locator.whole_range().nbytes(), 20);
    }

    #[test]
    fn locator_rejects_reversed_out_of_bounds_and_overflowing_extents() {
        let checkpoint = CheckpointId::fresh().unwrap();
        let locator = WeightLocator::new(checkpoint, 0, 100, 20).unwrap();

        let error = locator
            .range(Range { start: 9, end: 4 })
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid weight-relative byte range 9..4"));

        let error = locator.range(19..21).unwrap_err().to_string();
        assert!(error.contains("outside the 20-byte weight"));

        let error = WeightLocator::new(checkpoint, 0, u64::MAX - 1, 4)
            .unwrap_err()
            .to_string();
        assert!(error.contains("weight byte extent overflows"));
    }

    #[test]
    fn checkpoint_spans_append_only_exactly_adjacent_ranges() {
        let checkpoint = CheckpointId::fresh().unwrap();
        let first_locator = WeightLocator::new(checkpoint, 2, 100, 5).unwrap();
        let first = first_locator.whole_range();
        let second = WeightLocator::new(checkpoint, 2, 105, 7)
            .unwrap()
            .whole_range();

        let mut subranges = CheckpointSpan::from(first_locator.range(0..2).unwrap());
        subranges
            .extend(first_locator.range(2..5).unwrap())
            .unwrap();
        assert_eq!(subranges, CheckpointSpan::from(first));

        let mut span = CheckpointSpan::from_range(first);
        span.extend(second).unwrap();
        assert_eq!(span.checkpoint_id(), checkpoint);
        assert_eq!(span.shard(), 2);
        assert_eq!(span.start(), 100);
        assert_eq!(span.end(), 112);
        assert_eq!(span.nbytes(), 12);

        let third = WeightLocator::new(checkpoint, 2, 112, 3)
            .unwrap()
            .whole_range();
        let merged = span.merge(CheckpointSpan::from_range(third)).unwrap();
        assert_eq!(merged.start(), 100);
        assert_eq!(merged.end(), 115);
        assert_eq!(merged.nbytes(), 15);
    }

    #[test]
    fn checkpoint_spans_reject_gap_overlap_reverse_shard_and_checkpoint_changes() {
        let checkpoint = CheckpointId::fresh().unwrap();
        let other_checkpoint = CheckpointId::fresh().unwrap();
        let base = WeightLocator::new(checkpoint, 0, 100, 10)
            .unwrap()
            .whole_range();

        let error = CheckpointSpan::from(base)
            .merge(CheckpointSpan::from(
                WeightLocator::new(checkpoint, 0, 111, 1)
                    .unwrap()
                    .whole_range(),
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("leaves a gap"), "got: {error}");

        let error = CheckpointSpan::from(base)
            .merge(CheckpointSpan::from(
                WeightLocator::new(checkpoint, 0, 109, 2)
                    .unwrap()
                    .whole_range(),
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("overlaps"), "got: {error}");

        let error = CheckpointSpan::from(base)
            .merge(CheckpointSpan::from(
                WeightLocator::new(checkpoint, 0, 90, 10)
                    .unwrap()
                    .whole_range(),
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("reverse ordered"), "got: {error}");

        let error = CheckpointSpan::from(base)
            .merge(CheckpointSpan::from(
                WeightLocator::new(checkpoint, 1, 110, 1)
                    .unwrap()
                    .whole_range(),
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("shards 0 and 1"), "got: {error}");

        let error = CheckpointSpan::from(base)
            .merge(CheckpointSpan::from(
                WeightLocator::new(other_checkpoint, 0, 110, 1)
                    .unwrap()
                    .whole_range(),
            ))
            .unwrap_err()
            .to_string();
        assert!(error.contains("from"), "got: {error}");
        assert!(error.contains("and"), "got: {error}");
    }

    #[test]
    fn checkpoint_span_empty_ranges_are_boundary_noops_and_never_bridge_gaps() {
        let checkpoint = CheckpointId::fresh().unwrap();
        let empty_start = WeightLocator::new(checkpoint, 0, 100, 0)
            .unwrap()
            .whole_range();
        let value = WeightLocator::new(checkpoint, 0, 100, 4)
            .unwrap()
            .whole_range();
        let empty_end = WeightLocator::new(checkpoint, 0, 104, 0)
            .unwrap()
            .whole_range();

        let mut span = CheckpointSpan::from(empty_start);
        span.extend(value).unwrap();
        span.extend(empty_end).unwrap();
        assert_eq!(span.start(), 100);
        assert_eq!(span.end(), 104);
        assert_eq!(span.nbytes(), 4);

        let gap_empty = WeightLocator::new(checkpoint, 0, 106, 0)
            .unwrap()
            .whole_range();
        let unchanged = span;
        let error = span.extend(gap_empty).unwrap_err().to_string();
        assert!(error.contains("leaves a gap"), "got: {error}");
        assert_eq!(span, unchanged, "failed extension must be transactional");

        let earlier = WeightLocator::new(checkpoint, 0, 96, 4)
            .unwrap()
            .whole_range();
        let error = CheckpointSpan::from(empty_start)
            .merge(CheckpointSpan::from(earlier))
            .unwrap_err()
            .to_string();
        assert!(error.contains("reverse ordered"), "got: {error}");
    }

    #[test]
    fn checkpoint_span_merge_arithmetic_is_checked() {
        let error = checked_merged_end(0, u64::MAX, 1).unwrap_err().to_string();
        assert!(error.contains("byte length overflows"), "got: {error}");

        let error = checked_merged_end(u64::MAX, 1, 0).unwrap_err().to_string();
        assert!(error.contains("end overflows"), "got: {error}");
    }
}
