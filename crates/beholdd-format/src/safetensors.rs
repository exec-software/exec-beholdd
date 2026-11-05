//! Safetensors: header parse, tensor index, byte ranges, contiguity.
//!
//! A safetensors file is a `u64` header length, that many bytes of JSON mapping tensor
//! name to `{dtype, shape, data_offsets}`, then the payload. Offsets in the JSON are
//! relative to the end of the header, so the absolute position of a tensor is
//! `8 + header_len + data_offsets[0]`.
//!
//! Large checkpoints are split across shards with a `model.safetensors.index.json`
//! mapping every tensor name to its shard. We do not require that file: opening a
//! directory reads each shard's own header, which is authoritative and cannot disagree
//! with itself the way a stale index can.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use beholdd_core::DType;

use crate::{validate_span_read, CheckpointId, CheckpointSpan, RangeReader, WeightLocator};

/// One tensor, located.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    locator: WeightLocator,
    dtype: DType,
    /// Logical shape as the checkpoint declares it. For MXFP4 payloads the last
    /// dimension counts BYTES, so the logical width is twice this.
    shape: Vec<u64>,
}

impl TensorInfo {
    /// Index into the checkpoint's ordered shard list.
    pub fn shard(&self) -> usize {
        self.locator.shard()
    }

    /// Storage format.
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Logical shape as non-negative checkpoint dimensions.
    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    /// Absolute byte offset within the shard file.
    pub fn offset(&self) -> u64 {
        self.locator.start()
    }

    /// Exact encoded byte length.
    pub fn nbytes(&self) -> u64 {
        self.locator.nbytes()
    }

    /// The opened checkpoint that issued this tensor locator.
    pub fn checkpoint_id(&self) -> CheckpointId {
        self.locator.checkpoint_id()
    }

    /// Format-neutral, checkpoint-bound location of this tensor's encoded bytes.
    pub fn locator(&self) -> WeightLocator {
        self.locator
    }

    /// Number of stored elements, which for sub-byte formats is not the element count.
    pub fn numel(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |count, &dim| count.checked_mul(dim))
    }

    /// Byte range `[offset, offset + nbytes)`.
    pub fn end(&self) -> u64 {
        self.locator.end()
    }
}

/// An opened checkpoint: one file or a directory of shards.
#[derive(Debug)]
pub struct SafeTensors {
    shards: Vec<PathBuf>,
    files: Vec<File>,
    shard_lens: Vec<u64>,
    tensors: HashMap<String, TensorInfo>,
    checkpoint_id: CheckpointId,
}

impl SafeTensors {
    /// Open a single `.safetensors` file, or a directory containing them.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut shards: Vec<PathBuf> = if path.is_dir() {
            let mut v: Vec<PathBuf> = std::fs::read_dir(path)
                .with_context(|| format!("reading {}", path.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect();
            // Sorted so shard indices are stable across runs; a HashMap iteration order
            // would make `shard: usize` mean something different every time.
            v.sort();
            v
        } else {
            vec![path.to_path_buf()]
        };
        if shards.is_empty() {
            bail!("no .safetensors files in {}", path.display());
        }
        shards.dedup();

        let checkpoint_id = CheckpointId::fresh()?;
        let mut files = Vec::new();
        files
            .try_reserve_exact(shards.len())
            .map_err(|error| anyhow::anyhow!("cannot allocate safetensors file table: {error}"))?;
        let mut shard_lens = Vec::new();
        shard_lens
            .try_reserve_exact(shards.len())
            .map_err(|error| {
                anyhow::anyhow!("cannot allocate safetensors shard-size table: {error}")
            })?;
        let mut tensors: HashMap<String, TensorInfo> = HashMap::new();
        for (idx, sp) in shards.iter().enumerate() {
            let mut f = File::open(sp).with_context(|| format!("opening {}", sp.display()))?;
            let shard_len = f
                .metadata()
                .with_context(|| format!("reading size of {}", sp.display()))?
                .len();
            let (header, base) = read_header(&mut f)
                .with_context(|| format!("parsing header of {}", sp.display()))?;
            if base > shard_len {
                bail!(
                    "safetensors payload starts at {base}, beyond {}-byte shard {}",
                    shard_len,
                    sp.display()
                );
            }
            for (name, v) in header {
                if name == "__metadata__" {
                    continue;
                }
                let info = parse_entry(&name, &v, checkpoint_id, idx, base, shard_len)
                    .with_context(|| format!("tensor {name} in {}", sp.display()))?;
                if let Some(prev) = tensors.insert(name.clone(), info) {
                    bail!(
                        "tensor {name} appears in two shards ({} and {})",
                        shards[prev.shard()].display(),
                        sp.display()
                    );
                }
            }
            validate_shard_layout(&tensors, idx, base, shard_len)
                .with_context(|| format!("validating tensor layout of {}", sp.display()))?;
            shard_lens.push(shard_len);
            files.push(f);
        }
        Ok(SafeTensors {
            shards,
            files,
            shard_lens,
            tensors,
            checkpoint_id,
        })
    }

    /// Identity shared by this opened checkpoint and all tensor locators it issued.
    pub fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint_id
    }

    /// Shard paths in deterministic lexical order.
    pub fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    /// Look up a tensor. Absence is fatal for a caller that needs it: a missing weight
    /// read as zeros runs fine and answers wrongly.
    pub fn get(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Every tensor name.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// How many tensors were indexed.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the checkpoint contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Read a tensor's raw bytes into `buf`, which must be exactly `nbytes` long.
    ///
    /// Positional read rather than seek-then-read, so this is safe to call from several
    /// threads on the same file -- which is the whole point when a streaming engine
    /// wants every miss in flight at once.
    pub fn read_into(&self, info: &TensorInfo, buf: &mut [u8]) -> Result<()> {
        RangeReader::read_range_into(self, &info.locator().whole_range(), buf)
    }

    /// Read a tensor-relative byte slice into an equally sized buffer.
    pub fn read_tensor_slice_into(
        &self,
        info: &TensorInfo,
        range: Range<u64>,
        buf: &mut [u8],
    ) -> Result<()> {
        let range = info.locator().range(range)?;
        RangeReader::read_range_into(self, &range, buf)
    }

    /// Read a tensor into a fresh `Vec`.
    pub fn read(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        let len = usize::try_from(info.nbytes()).context("tensor length does not fit this host")?;
        let mut v = Vec::new();
        v.try_reserve_exact(len)
            .map_err(|error| anyhow::anyhow!("cannot allocate {len}-byte tensor: {error}"))?;
        v.resize(len, 0);
        self.read_into(info, &mut v)?;
        Ok(v)
    }

    /// If these tensors form one contiguous run in one shard, return `(shard, offset,
    /// total_bytes)`.
    ///
    /// This is not bookkeeping, it is the difference between one `pread` and six. On
    /// DeepSeek-V4 an expert's three weight blobs are contiguous and its three scale
    /// blobs are contiguous somewhere else entirely, so a fetch is two reads rather than
    /// six -- **measured per expert rather than assumed**, because a checkpoint revision
    /// that interleaved them differently would otherwise read the wrong bytes at full
    /// speed.
    pub fn contiguous_run(&self, names: &[&str]) -> Option<(usize, u64, u64)> {
        let first = self.get(names.first()?)?;
        let mut end = first.offset();
        let mut total = 0u64;
        for n in names {
            let t = self.get(n)?;
            if t.shard() != first.shard() || t.offset() != end {
                return None;
            }
            end = t.end();
            total = total.checked_add(t.nbytes())?;
        }
        Some((first.shard(), first.offset(), total))
    }
}

fn validate_shard_layout(
    tensors: &HashMap<String, TensorInfo>,
    shard: usize,
    payload_start: u64,
    shard_len: u64,
) -> Result<()> {
    let mut ordered = Vec::new();
    for (name, tensor) in tensors {
        if tensor.shard() == shard {
            ordered
                .try_reserve(1)
                .map_err(|error| anyhow::anyhow!("cannot allocate layout index: {error}"))?;
            ordered.push((name.as_str(), tensor));
        }
    }
    ordered.sort_unstable_by_key(|(name, tensor)| (tensor.offset(), tensor.end(), *name));

    let mut expected = payload_start;
    for (name, tensor) in ordered {
        if tensor.offset() != expected {
            let problem = if tensor.offset() < expected {
                "overlaps an earlier tensor"
            } else {
                "leaves a hole after the earlier tensor"
            };
            bail!(
                "tensor {name} starts at {}, expected {expected}; its range {problem}",
                tensor.offset()
            );
        }
        expected = tensor.end();
    }
    if expected != shard_len {
        bail!(
            "tensor payload ends at {expected}, but shard ends at {shard_len}; trailing payload bytes are not indexed"
        );
    }
    Ok(())
}

impl RangeReader for SafeTensors {
    fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint_id
    }

    fn locators(&self) -> Box<dyn Iterator<Item = WeightLocator> + '_> {
        Box::new(self.tensors.values().map(TensorInfo::locator))
    }

    fn shard_count(&self) -> usize {
        self.files.len()
    }

    fn shard_len(&self, shard: usize) -> Option<u64> {
        self.shard_lens.get(shard).copied()
    }

    fn read_span_into(&self, span: &CheckpointSpan, buf: &mut [u8]) -> Result<()> {
        validate_span_read(self, span, buf)?;
        self.files[span.shard()]
            .read_exact_at(buf, span.start())
            .with_context(|| {
                format!(
                    "reading safetensors weight range or checkpoint span {}..{} from shard {}",
                    span.start(),
                    span.end(),
                    span.shard()
                )
            })?;
        Ok(())
    }
}

/// Parse the `u64` length prefix and the JSON header. Returns the entries and the byte
/// offset the payload starts at.
fn read_header(f: &mut File) -> Result<(Vec<(String, serde_json::Value)>, u64)> {
    let mut len_bytes = [0u8; 8];
    f.read_exact(&mut len_bytes)?;
    let hlen = u64::from_le_bytes(len_bytes);
    // A corrupt or non-safetensors file usually shows up here as an absurd length, and
    // it is far better to say so than to try to allocate it.
    if hlen == 0 || hlen > 512 * 1024 * 1024 {
        bail!("implausible header length {hlen}; is this a safetensors file?");
    }
    let json_len = usize::try_from(hlen).context("safetensors header length does not fit host")?;
    let mut json = Vec::new();
    json.try_reserve_exact(json_len)
        .map_err(|error| anyhow::anyhow!("cannot allocate {hlen}-byte header: {error}"))?;
    json.resize(json_len, 0);
    f.read_exact(&mut json)?;
    f.seek(SeekFrom::Start(0))?;
    let parsed: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&json).context("header is not a JSON object")?;
    Ok((parsed.into_iter().collect(), 8 + hlen))
}

fn parse_entry(
    name: &str,
    v: &serde_json::Value,
    checkpoint_id: CheckpointId,
    shard: usize,
    base: u64,
    shard_len: u64,
) -> Result<TensorInfo> {
    let dt_str = v
        .get("dtype")
        .and_then(|d| d.as_str())
        .context("no dtype")?;
    let dtype = DType::from_safetensors(dt_str)
        // Refusing an unknown dtype is deliberate. Skipping it would leave a tensor
        // silently absent, and an absent weight reads as zeros and still generates text.
        .with_context(|| format!("unsupported dtype {dt_str} on {name}"))?;
    let shape_values = v
        .get("shape")
        .and_then(|s| s.as_array())
        .context("no shape")?;
    let mut shape = Vec::new();
    shape
        .try_reserve_exact(shape_values.len())
        .map_err(|error| anyhow::anyhow!("cannot allocate tensor shape: {error}"))?;
    for (axis, value) in shape_values.iter().enumerate() {
        shape.push(
            value
                .as_u64()
                .with_context(|| format!("shape axis {axis} is not a non-negative integer"))?,
        );
    }
    let offs = v
        .get("data_offsets")
        .and_then(|o| o.as_array())
        .context("no data_offsets")?;
    if offs.len() != 2 {
        bail!("data_offsets must have two entries");
    }
    let start = offs[0].as_u64().context("bad data_offsets[0]")?;
    let end = offs[1].as_u64().context("bad data_offsets[1]")?;
    if end < start {
        bail!("data_offsets are reversed: {start}..{end}");
    }
    let offset = base
        .checked_add(start)
        .context("absolute tensor start offset overflows")?;
    let absolute_end = base
        .checked_add(end)
        .context("absolute tensor end offset overflows")?;
    if absolute_end > shard_len {
        bail!("encoded range {offset}..{absolute_end} is beyond {shard_len}-byte shard");
    }
    let nbytes = end - start;
    let numel = shape.iter().try_fold(1u64, |count, &dim| {
        count
            .checked_mul(dim)
            .with_context(|| format!("element count overflows for shape {shape:?}"))
    })?;
    if let Some(element_bytes) = dtype.size() {
        let expected = numel
            .checked_mul(u64::try_from(element_bytes).expect("dtype size fits u64"))
            .context("encoded tensor byte count overflows")?;
        if nbytes != expected {
            bail!(
                "dtype {dtype:?} and shape {shape:?} require {expected} bytes, data_offsets declare {nbytes}"
            );
        }
    }
    let locator = WeightLocator::new(checkpoint_id, shard, offset, nbytes)?;
    Ok(TensorInfo {
        locator,
        dtype,
        shape,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a tiny valid safetensors file in a temp dir.
    fn write_fixture(dir: &Path, name: &str, entries: &[(&str, &str, Vec<i64>, usize)]) -> PathBuf {
        let mut header = serde_json::Map::new();
        let mut payload: Vec<u8> = Vec::new();
        for (tname, dtype, shape, nbytes) in entries {
            let start = payload.len();
            payload.extend(std::iter::repeat_n(0xABu8, *nbytes));
            header.insert(
                (*tname).to_string(),
                serde_json::json!({
                    "dtype": dtype,
                    "shape": shape,
                    "data_offsets": [start, payload.len()],
                }),
            );
        }
        let json = serde_json::to_vec(&header).unwrap();
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&json).unwrap();
        f.write_all(&payload).unwrap();
        p
    }

    fn write_raw_fixture(
        dir: &Path,
        name: &str,
        header: serde_json::Value,
        payload: &[u8],
    ) -> PathBuf {
        let json = serde_json::to_vec(&header).unwrap();
        let p = dir.join(name);
        let mut f = File::create(&p).unwrap();
        f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&json).unwrap();
        f.write_all(payload).unwrap();
        p
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("beholdd-st-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn parses_header_and_locates_payload() {
        let d = tmpdir("basic");
        write_fixture(
            &d,
            "a.safetensors",
            &[("w", "BF16", vec![2, 4], 16), ("s", "F8_E8M0", vec![2], 2)],
        );
        let st = SafeTensors::open(&d).unwrap();
        assert_eq!(st.len(), 2);

        let w = st.get("w").unwrap();
        assert_eq!(w.dtype, DType::Bf16);
        assert_eq!(w.shape, vec![2, 4]);
        assert_eq!(w.nbytes(), 16);
        assert_eq!(w.numel(), Some(8));

        // The payload must actually be where we said it was.
        let bytes = st.read(w).unwrap();
        assert_eq!(bytes.len(), 16);
        assert!(bytes.iter().all(|&b| b == 0xAB), "read the wrong region");

        // And the second tensor starts exactly where the first ends.
        let s = st.get("s").unwrap();
        assert_eq!(s.offset(), w.end());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn contiguous_run_detects_adjacency_and_rejects_gaps() {
        let d = tmpdir("contig");
        write_fixture(
            &d,
            "a.safetensors",
            &[
                ("w1", "I8", vec![4], 4),
                ("w2", "I8", vec![4], 4),
                ("w3", "I8", vec![4], 4),
            ],
        );
        let st = SafeTensors::open(&d).unwrap();
        let run = st.contiguous_run(&["w1", "w2", "w3"]).unwrap();
        assert_eq!(run.2, 12, "three 4-byte tensors are a 12-byte run");

        // Out of order is NOT a contiguous run, even though the same bytes are covered.
        // A fetch that read [w1..w3] and handed the caller w2's pointer would be
        // catastrophic and silent.
        assert!(st.contiguous_run(&["w1", "w3", "w2"]).is_none());
        assert!(st.contiguous_run(&["w1", "w3"]).is_none());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn coalesced_checkpoint_spans_read_exact_bytes_through_a_dyn_reader() {
        let d = tmpdir("checkpoint-span");
        let p = write_raw_fixture(
            &d,
            "a.safetensors",
            serde_json::json!({
                "a": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "I8", "shape": [3], "data_offsets": [2, 5]}
            }),
            &[1, 2, 3, 4, 5],
        );
        let st = SafeTensors::open(&p).unwrap();
        let a = st.get("a").unwrap();
        let b = st.get("b").unwrap();
        let mut span = CheckpointSpan::from(a.locator().whole_range());
        span.extend(b.locator().whole_range()).unwrap();

        let reader: &dyn RangeReader = &st;
        let mut bytes = [0u8; 5];
        reader.read_span_into(&span, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4, 5]);

        let error = reader
            .read_span_into(&span, &mut [0u8; 4])
            .unwrap_err()
            .to_string();
        assert!(error.contains("buffer is 4 bytes"), "got: {error}");
        assert!(error.contains("checkpoint span"), "got: {error}");

        let empty = CheckpointSpan::from(b.locator().range(b.nbytes()..b.nbytes()).unwrap());
        reader.read_span_into(&empty, &mut []).unwrap();

        File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_len(span.end() - 1)
            .unwrap();
        let error = reader
            .read_span_into(&span, &mut bytes)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("reading safetensors weight range"),
            "got: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn duplicate_tensor_across_shards_is_an_error() {
        let d = tmpdir("dupe");
        write_fixture(&d, "a.safetensors", &[("w", "F32", vec![1], 4)]);
        write_fixture(&d, "b.safetensors", &[("w", "F32", vec![1], 4)]);
        let err = SafeTensors::open(&d).unwrap_err().to_string();
        assert!(err.contains("two shards"), "got: {err}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn unknown_dtype_is_refused_not_skipped() {
        let d = tmpdir("dtype");
        write_fixture(&d, "a.safetensors", &[("w", "F64", vec![1], 8)]);
        // The chain must name the tensor AND the dtype, or the failure is unactionable.
        let full = format!("{:#}", SafeTensors::open(&d).unwrap_err());
        assert!(full.contains("F64"), "got: {full}");
        assert!(full.contains("tensor w"), "got: {full}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn garbage_is_rejected_before_allocating() {
        let d = tmpdir("garbage");
        let p = d.join("bad.safetensors");
        // A huge length prefix is what a non-safetensors file looks like here.
        let mut f = File::create(&p).unwrap();
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        f.write_all(b"not json").unwrap();
        drop(f);
        // `{:#}` prints the whole context chain. Plain Display shows only the OUTERMOST
        // context ("parsing header of ..."), which is the least actionable part -- worth
        // knowing when reading these errors in a log.
        let err = format!("{:#}", SafeTensors::open(&p).unwrap_err());
        assert!(err.contains("implausible"), "got: {err}");
        assert!(err.contains("bad.safetensors"), "must name the file: {err}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn shard_order_is_stable() {
        let d = tmpdir("order");
        write_fixture(&d, "model-00002.safetensors", &[("b", "F32", vec![1], 4)]);
        write_fixture(&d, "model-00001.safetensors", &[("a", "F32", vec![1], 4)]);
        let st = SafeTensors::open(&d).unwrap();
        assert_eq!(
            st.get("a").unwrap().shard(),
            0,
            "shard 0 must be model-00001"
        );
        assert_eq!(st.get("b").unwrap().shard(), 1);
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn checkpoint_bound_ranges_reject_foreign_locators_and_bad_shards() {
        let d = tmpdir("range-identity");
        let p = write_fixture(&d, "a.safetensors", &[("w", "I8", vec![8], 8)]);
        let first = SafeTensors::open(&p).unwrap();
        let second = SafeTensors::open(&p).unwrap();
        let info = first.get("w").unwrap();

        let error = second.read_into(info, &mut [0; 8]).unwrap_err().to_string();
        assert!(error.contains("belongs to checkpoint"), "got: {error}");
        assert!(error.contains("not source checkpoint"), "got: {error}");

        let foreign_span = CheckpointSpan::from(info.locator().range(0..1).unwrap());
        let error = RangeReader::read_span_into(&second, &foreign_span, &mut [0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("checkpoint span belongs"), "got: {error}");

        let bad = WeightLocator::new(first.checkpoint_id(), 9, 0, 1)
            .unwrap()
            .whole_range();
        let error = RangeReader::read_range_into(&first, &bad, &mut [0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("names shard 9"), "got: {error}");

        let overflow = WeightLocator::new(first.checkpoint_id(), 0, u64::MAX - 1, 4)
            .unwrap_err()
            .to_string();
        assert!(overflow.contains("weight byte extent overflows"));
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn tensor_slices_reject_reversed_out_of_bounds_and_wrong_buffer_lengths() {
        let d = tmpdir("range-errors");
        let p = write_fixture(&d, "a.safetensors", &[("w", "I8", vec![8], 8)]);
        let st = SafeTensors::open(&p).unwrap();
        let info = st.get("w").unwrap();

        let error = st
            .read_tensor_slice_into(info, Range { start: 7, end: 3 }, &mut [])
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid weight-relative byte range 7..3"));

        let error = st
            .read_tensor_slice_into(info, 7..9, &mut [0; 2])
            .unwrap_err()
            .to_string();
        assert!(error.contains("outside the 8-byte weight"), "got: {error}");

        let error = st
            .read_tensor_slice_into(info, 2..6, &mut [0; 3])
            .unwrap_err()
            .to_string();
        assert!(error.contains("buffer is 3 bytes"), "got: {error}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn declared_truncation_is_rejected_during_open() {
        let d = tmpdir("truncated-open");
        let p = write_fixture(&d, "a.safetensors", &[("w", "I8", vec![8], 8)]);
        let len = std::fs::metadata(&p).unwrap().len();
        File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_len(len - 1)
            .unwrap();

        let error = format!("{:#}", SafeTensors::open(&p).unwrap_err());
        assert!(error.contains("encoded range"), "got: {error}");
        assert!(error.contains("beyond"), "got: {error}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_file_truncated_after_open_produces_a_short_read_error() {
        let d = tmpdir("truncated-read");
        let p = write_fixture(&d, "a.safetensors", &[("w", "I8", vec![8], 8)]);
        let st = SafeTensors::open(&p).unwrap();
        let info = st.get("w").unwrap();
        File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_len(info.end() - 1)
            .unwrap();

        let error = st.read_into(info, &mut [0; 8]).unwrap_err().to_string();
        assert!(
            error.contains("reading safetensors weight range"),
            "got: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn absolute_offset_overflow_and_invalid_shapes_are_rejected() {
        let d = tmpdir("header-arithmetic");
        let overflow = write_raw_fixture(
            &d,
            "overflow.safetensors",
            serde_json::json!({
                "w": {"dtype": "I8", "shape": [1], "data_offsets": [u64::MAX - 1, u64::MAX]}
            }),
            &[],
        );
        let error = format!("{:#}", SafeTensors::open(&overflow).unwrap_err());
        assert!(
            error.contains("absolute tensor start offset overflows"),
            "got: {error}"
        );

        let negative = write_raw_fixture(
            &d,
            "negative.safetensors",
            serde_json::json!({
                "w": {"dtype": "I8", "shape": [-1], "data_offsets": [0, 1]}
            }),
            &[0],
        );
        let error = format!("{:#}", SafeTensors::open(&negative).unwrap_err());
        assert!(error.contains("shape axis 0"), "got: {error}");
        assert!(error.contains("non-negative integer"), "got: {error}");

        let shape_overflow = write_raw_fixture(
            &d,
            "shape-overflow.safetensors",
            serde_json::json!({
                "w": {"dtype": "I8", "shape": [u64::MAX, 2], "data_offsets": [0, 0]}
            }),
            &[],
        );
        let error = format!("{:#}", SafeTensors::open(&shape_overflow).unwrap_err());
        assert!(error.contains("element count overflows"), "got: {error}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn dtype_shape_and_declared_byte_count_must_agree() {
        let d = tmpdir("byte-count");
        let p = write_raw_fixture(
            &d,
            "bad.safetensors",
            serde_json::json!({
                "w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}
            }),
            &[0; 4],
        );
        let error = format!("{:#}", SafeTensors::open(&p).unwrap_err());
        assert!(error.contains("require 8 bytes"), "got: {error}");
        assert!(error.contains("declare 4"), "got: {error}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn layout_accepts_empty_aliases_and_refuses_overlap_holes_and_trailing_bytes() {
        let d = tmpdir("layout-integrity");
        let valid = write_raw_fixture(
            &d,
            "valid-empty.safetensors",
            serde_json::json!({
                "empty_a": {"dtype": "I8", "shape": [0], "data_offsets": [0, 0]},
                "empty_b": {"dtype": "I8", "shape": [0], "data_offsets": [0, 0]},
                "value": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}
            }),
            &[7],
        );
        let checkpoint = SafeTensors::open(&valid).unwrap();
        assert_eq!(checkpoint.len(), 3);
        assert_eq!(checkpoint.get("empty_a").unwrap().nbytes(), 0);
        assert_eq!(checkpoint.get("empty_b").unwrap().nbytes(), 0);

        let overlap = write_raw_fixture(
            &d,
            "overlap.safetensors",
            serde_json::json!({
                "a": {"dtype": "I8", "shape": [2], "data_offsets": [0, 2]},
                "b": {"dtype": "I8", "shape": [2], "data_offsets": [1, 3]}
            }),
            &[1, 2, 3],
        );
        let error = format!("{:#}", SafeTensors::open(&overlap).unwrap_err());
        assert!(error.contains("overlaps an earlier tensor"), "got: {error}");

        let hole = write_raw_fixture(
            &d,
            "hole.safetensors",
            serde_json::json!({
                "a": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]},
                "b": {"dtype": "I8", "shape": [1], "data_offsets": [2, 3]}
            }),
            &[1, 2, 3],
        );
        let error = format!("{:#}", SafeTensors::open(&hole).unwrap_err());
        assert!(error.contains("leaves a hole"), "got: {error}");

        let trailing = write_raw_fixture(
            &d,
            "trailing.safetensors",
            serde_json::json!({
                "a": {"dtype": "I8", "shape": [1], "data_offsets": [0, 1]}
            }),
            &[1, 2],
        );
        let error = format!("{:#}", SafeTensors::open(&trailing).unwrap_err());
        assert!(
            error.contains("trailing payload bytes are not indexed"),
            "got: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn generic_reader_enumerates_all_locators() {
        let d = tmpdir("locator-enumeration");
        let p = write_fixture(
            &d,
            "a.safetensors",
            &[("a", "I8", vec![4], 4), ("b", "I8", vec![8], 8)],
        );
        let st = SafeTensors::open(&p).unwrap();
        let mut locators: Vec<_> = RangeReader::locators(&st).collect();
        locators.sort_unstable_by_key(|locator| locator.start());
        assert_eq!(
            locators,
            [
                st.get("a").unwrap().locator(),
                st.get("b").unwrap().locator()
            ]
        );
        std::fs::remove_dir_all(&d).ok();
    }
}
