//! GGUF: header, metadata, tensor index.
//!
//! The other half of the world. Safetensors is what a training framework emits; GGUF is
//! what the local-inference ecosystem quantises to, and it is where our Qwen and Laguna
//! checkpoints live.
//!
//! Layout: magic `GGUF`, a `u32` version, `u64` tensor count, `u64` metadata count, then
//! the metadata as typed key/value pairs, then one info record per tensor, then padding
//! to `general.alignment`, then the payload. Tensor offsets in the info records are
//! **relative to the start of the payload**, not to the file.
//!
//! # Two things this reader refuses to guess
//!
//! **Architecture.** `general.architecture` names the model family and every other key
//! is namespaced under it (`laguna.block_count`, `qwen3moe.expert_count`). A reader that
//! hardcodes one prefix silently reports zeros for every other family.
//!
//! **Alignment.** `general.alignment` defaults to 32 but checkpoints may set it. Getting
//! it wrong shifts every tensor by a few bytes, which does not fail — it produces a
//! model that loads and generates nonsense.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use beholdd_core::ggml_block_geometry;

pub use crate::CheckpointId;
use crate::{validate_span_read, CheckpointSpan, RangeReader, WeightLocator};

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Unsigned integer of any width, widened.
    U(u64),
    /// Signed integer of any width, widened.
    I(i64),
    /// Float of any width, widened.
    F(f64),
    /// Boolean.
    Bool(bool),
    /// UTF-8 string.
    Str(String),
    /// Homogeneous array.
    Arr(Vec<Value>),
}

impl Value {
    /// As an integer, if it is one.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U(v) => Some(*v),
            Value::I(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }
    /// As a float, if it is numeric.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F(v) => Some(*v),
            Value::U(v) => Some(*v as f64),
            Value::I(v) => Some(*v as f64),
            _ => None,
        }
    }
    /// As a string slice, if it is one.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
    /// As a slice, if it is an array.
    pub fn as_arr(&self) -> Option<&[Value]> {
        match self {
            Value::Arr(v) => Some(v),
            _ => None,
        }
    }
}

/// A ggml tensor type. Only the numeric code is kept; storage geometry is format
/// vocabulary in `beholdd-core`, while each backend decides which types it can decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlType(pub u32);

impl GgmlType {
    /// The common names, for diagnostics. Unknown codes render as their number.
    pub fn name(self) -> String {
        match self.0 {
            0 => "F32".into(),
            1 => "F16".into(),
            2 => "Q4_0".into(),
            3 => "Q4_1".into(),
            6 => "Q5_0".into(),
            7 => "Q5_1".into(),
            8 => "Q8_0".into(),
            9 => "Q8_1".into(),
            10 => "Q2_K".into(),
            11 => "Q3_K".into(),
            12 => "Q4_K".into(),
            13 => "Q5_K".into(),
            14 => "Q6_K".into(),
            15 => "Q8_K".into(),
            16 => "IQ2_XXS".into(),
            17 => "IQ2_XS".into(),
            18 => "IQ3_XXS".into(),
            19 => "IQ1_S".into(),
            20 => "IQ4_NL".into(),
            21 => "IQ3_S".into(),
            22 => "IQ2_S".into(),
            23 => "IQ4_XS".into(),
            24 => "I8".into(),
            25 => "I16".into(),
            26 => "I32".into(),
            27 => "I64".into(),
            28 => "F64".into(),
            29 => "IQ1_M".into(),
            30 => "BF16".into(),
            34 => "TQ1_0".into(),
            35 => "TQ2_0".into(),
            39 => "MXFP4".into(),
            40 => "NVFP4".into(),
            41 => "Q1_0".into(),
            42 => "F8_E4M3_B128".into(),
            other => format!("type{other}"),
        }
    }
}

/// One tensor, located.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    locator: WeightLocator,
    /// ggml type code.
    kind: GgmlType,
    /// Dimensions, **fastest-varying first** — the opposite of safetensors' convention.
    dims: Vec<u64>,
}

impl GgufTensor {
    /// Index into the checkpoint's ordered shard list.
    pub fn shard(&self) -> usize {
        self.locator.shard()
    }

    /// GGML storage type.
    pub fn kind(&self) -> GgmlType {
        self.kind
    }

    /// Dimensions in fastest-varying-first order.
    pub fn dims(&self) -> &[u64] {
        &self.dims
    }

    /// Absolute byte offset within the shard.
    pub fn offset(&self) -> u64 {
        self.locator.start()
    }

    /// Exact encoded byte length, excluding GGUF alignment padding.
    pub fn nbytes(&self) -> u64 {
        self.locator.nbytes()
    }

    /// Element count, if the dimensions fit in `u64`.
    pub fn numel(&self) -> Option<u64> {
        self.dims
            .iter()
            .try_fold(1u64, |count, &dim| count.checked_mul(dim))
    }

    /// The opened checkpoint that issued this locator.
    pub fn checkpoint_id(&self) -> CheckpointId {
        self.locator.checkpoint_id()
    }

    /// Format-neutral, checkpoint-bound location of this tensor's encoded bytes.
    pub fn locator(&self) -> WeightLocator {
        self.locator
    }
}

/// An opened GGUF checkpoint, single file or split.
#[derive(Debug)]
pub struct Gguf {
    shards: Vec<PathBuf>,
    meta: BTreeMap<String, Value>,
    files: Vec<File>,
    shard_lens: Vec<u64>,
    tensors: BTreeMap<String, GgufTensor>,
    checkpoint_id: CheckpointId,
}

impl Gguf {
    /// Open a `.gguf` file. If it is the first shard of a split, siblings are found and
    /// opened automatically.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let shards = discover_shards(path)?;
        let checkpoint_id = CheckpointId::fresh()?;

        let mut files = Vec::new();
        files
            .try_reserve_exact(shards.len())
            .map_err(|error| anyhow::anyhow!("cannot allocate GGUF file table: {error}"))?;
        let mut shard_lens = Vec::new();
        shard_lens
            .try_reserve_exact(shards.len())
            .map_err(|error| anyhow::anyhow!("cannot allocate GGUF shard-size table: {error}"))?;
        let mut tensors = BTreeMap::new();
        let mut meta = BTreeMap::new();
        let mut split_tensor_count = None;
        let shard_count = u64::try_from(shards.len()).context("shard count does not fit u64")?;

        for (idx, sp) in shards.iter().enumerate() {
            let file = File::open(sp).with_context(|| format!("opening {}", sp.display()))?;
            let Parsed {
                meta: shard_meta,
                tensors: shard_tensors,
            } = parse_header(&file)
                .with_context(|| format!("parsing GGUF header of {}", sp.display()))?;
            validate_split_metadata(
                &shard_meta,
                idx,
                shard_count,
                shards.len() > 1,
                sp,
                &mut split_tensor_count,
            )?;
            if idx == 0 {
                meta = shard_meta;
            }
            for (name, t) in shard_tensors {
                let tensor = GgufTensor {
                    locator: WeightLocator::new(checkpoint_id, idx, t.offset, t.nbytes)
                        .with_context(|| format!("binding GGUF tensor {name}"))?,
                    kind: t.kind,
                    dims: t.dims,
                };
                if tensors.insert(name.clone(), tensor).is_some() {
                    bail!("tensor {name} appears in more than one shard");
                }
            }
            shard_lens.push(
                file.metadata()
                    .with_context(|| format!("reading size of {}", sp.display()))?
                    .len(),
            );
            files.push(file);
        }
        if let Some(expected) = split_tensor_count {
            let actual = u64::try_from(tensors.len()).context("tensor count does not fit u64")?;
            if actual != expected {
                bail!(
                    "split checkpoint declares {expected} tensors across all shards, found {actual}"
                );
            }
        }
        Ok(Gguf {
            shards,
            meta,
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

    /// Shard paths in split order.
    pub fn shards(&self) -> &[PathBuf] {
        &self.shards
    }

    /// Immutable checkpoint metadata from the first shard.
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.meta
    }

    /// Look up a tensor.
    pub fn get(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.get(name)
    }

    /// Every tensor name, sorted.
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

    /// `general.architecture`, which namespaces every other key.
    pub fn arch(&self) -> Option<&str> {
        self.meta.get("general.architecture")?.as_str()
    }

    /// Fetch an architecture-namespaced key: `kv("block_count")` resolves to
    /// `laguna.block_count` or `qwen3moe.block_count` as appropriate.
    ///
    /// This is why the reader refuses to hardcode a prefix — the same logical field
    /// lives under a different key in every family.
    pub fn kv(&self, suffix: &str) -> Option<&Value> {
        let arch = self.arch()?;
        self.meta.get(&format!("{arch}.{suffix}"))
    }

    /// Read an absolute byte range from one shard into `buf`.
    ///
    /// The range is half-open and `buf` must have exactly the same length. Both the
    /// shard index and range are checked before the positional read, so malformed or
    /// stale locations return an error rather than indexing the shard list or reading
    /// beyond the checkpoint. Positional reads are safe to call concurrently.
    pub fn read_range_into(&self, shard: usize, range: Range<u64>, buf: &mut [u8]) -> Result<()> {
        let Some(file) = self.files.get(shard) else {
            bail!(
                "GGUF shard {shard} does not exist (checkpoint has {} shards)",
                self.files.len()
            );
        };
        let shard_len = self.shard_lens[shard];
        let nbytes = range.end.checked_sub(range.start).with_context(|| {
            format!(
                "invalid GGUF byte range {}..{} in shard {shard}",
                range.start, range.end
            )
        })?;
        let buf_len = u64::try_from(buf.len()).context("buffer length does not fit in u64")?;
        if buf_len != nbytes {
            bail!(
                "buffer is {buf_len} bytes, but GGUF range {}..{} in shard {shard} is {nbytes} bytes",
                range.start,
                range.end
            );
        }
        if range.end > shard_len {
            bail!(
                "GGUF range {}..{} is outside shard {shard}, which is {shard_len} bytes",
                range.start,
                range.end
            );
        }
        file.read_exact_at(buf, range.start).with_context(|| {
            format!(
                "reading GGUF range {}..{} from shard {shard}",
                range.start, range.end
            )
        })?;
        Ok(())
    }

    /// Read a byte slice of a tensor into `buf`.
    ///
    /// `range` is relative to the start of `tensor`, not the shard. It must lie wholly
    /// within the tensor and `buf` must have exactly the range's length.
    pub fn read_tensor_slice_into(
        &self,
        tensor: &GgufTensor,
        range: Range<u64>,
        buf: &mut [u8],
    ) -> Result<()> {
        let range = tensor.locator().range(range)?;
        RangeReader::read_range_into(self, &range, buf)
    }

    /// Validate and resolve a tensor-relative byte slice without allocating or reading.
    pub fn resolve_tensor_slice(
        &self,
        tensor: &GgufTensor,
        range: Range<u64>,
    ) -> Result<Range<u64>> {
        let range = tensor.locator().range(range)?;
        if range.checkpoint_id() != self.checkpoint_id {
            bail!(
                "tensor locator belongs to checkpoint {:?}, not source checkpoint {:?}",
                range.checkpoint_id(),
                self.checkpoint_id
            );
        }
        if range.shard() >= self.files.len() {
            bail!(
                "tensor locator names shard {}, checkpoint has {} shards",
                range.shard(),
                self.files.len()
            );
        }
        if range.end() > self.shard_lens[range.shard()] {
            bail!(
                "tensor slice resolves to {}..{}, outside shard {} of {} bytes",
                range.start(),
                range.end(),
                range.shard(),
                self.shard_lens[range.shard()]
            );
        }
        Ok(range.start()..range.end())
    }

    /// Read a tensor's raw bytes. Positional, so it is safe to call concurrently.
    pub fn read_into(&self, t: &GgufTensor, buf: &mut [u8]) -> Result<()> {
        self.read_tensor_slice_into(t, 0..t.nbytes(), buf)
    }
}

impl RangeReader for Gguf {
    fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint_id
    }

    fn locators(&self) -> Box<dyn Iterator<Item = WeightLocator> + '_> {
        Box::new(self.tensors.values().map(GgufTensor::locator))
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
                    "reading GGUF weight range or checkpoint span {}..{} from shard {}",
                    span.start(),
                    span.end(),
                    span.shard()
                )
            })?;
        Ok(())
    }
}

fn validate_split_metadata(
    meta: &BTreeMap<String, Value>,
    shard_index: usize,
    shard_count: u64,
    discovered_split: bool,
    path: &Path,
    expected_tensor_count: &mut Option<u64>,
) -> Result<()> {
    let declared_count = meta.get("split.count").and_then(Value::as_u64);
    if !discovered_split {
        if declared_count.is_some_and(|count| count > 1) {
            bail!(
                "{} declares a {declared_count:?}-shard split but its filename does not have a valid split suffix",
                path.display()
            );
        }
        return Ok(());
    }

    let need = |key: &str| -> Result<u64> {
        meta.get(key)
            .and_then(Value::as_u64)
            .with_context(|| format!("{} is missing integer {key}", path.display()))
    };
    let declared_index = need("split.no")?;
    let expected_index = u64::try_from(shard_index).context("shard index does not fit u64")?;
    if declared_index != expected_index {
        bail!(
            "{} declares split.no={declared_index}, expected {expected_index}",
            path.display()
        );
    }
    let declared_count = need("split.count")?;
    if declared_count != shard_count {
        bail!(
            "{} declares split.count={declared_count}, filename set has {shard_count}",
            path.display()
        );
    }
    let tensor_count = need("split.tensors.count")?;
    match expected_tensor_count {
        Some(expected) if *expected != tensor_count => bail!(
            "{} declares split.tensors.count={tensor_count}, earlier shard declares {expected}",
            path.display()
        ),
        Some(_) => {}
        None => *expected_tensor_count = Some(tensor_count),
    }
    Ok(())
}

/// Find every shard of a possibly-split checkpoint.
///
/// The convention is `name-00001-of-00003.gguf`. Only the FIRST shard carries the
/// metadata, so opening shard 2 directly and finding no architecture is a real failure
/// mode worth naming rather than a puzzle to debug.
fn discover_shards(path: &Path) -> Result<Vec<PathBuf>> {
    let name = path
        .file_name()
        .and_then(|s| s.to_str())
        .context("path has no filename")?;
    let Some(base) = name.strip_suffix(".gguf") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((left, total_text)) = base.rsplit_once("-of-") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let Some((stem, index_text)) = left.rsplit_once('-') else {
        return Ok(vec![path.to_path_buf()]);
    };
    if index_text.is_empty()
        || total_text.is_empty()
        || !index_text.bytes().all(|byte| byte.is_ascii_digit())
        || !total_text.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Ok(vec![path.to_path_buf()]);
    }
    if index_text.len() != total_text.len() {
        bail!("GGUF split index and count widths differ in {name}");
    }
    let index: usize = index_text.parse().context("could not parse shard index")?;
    let total: usize = total_text.parse().context("could not parse shard count")?;
    if total == 0 || total > 1_048_576 {
        bail!("implausible GGUF shard count {total}");
    }
    if index == 0 || index > total {
        bail!("GGUF shard index {index} is outside 1..={total}");
    }
    let dir = path.parent().unwrap_or(Path::new("."));
    let width = index_text.len();
    let mut out = Vec::new();
    out.try_reserve_exact(total).map_err(|error| {
        anyhow::anyhow!("cannot allocate index for {total} GGUF shards: {error}")
    })?;
    for i in 1..=total {
        let p = dir.join(format!("{stem}-{i:0width$}-of-{total:0width$}.gguf"));
        if !p.exists() {
            bail!(
                "split checkpoint is missing shard {i} of {total}: {}",
                p.display()
            );
        }
        out.push(p);
    }
    Ok(out)
}

struct Parsed {
    meta: BTreeMap<String, Value>,
    tensors: Vec<(String, RawGgufTensor)>,
}

struct RawGgufTensor {
    kind: GgmlType,
    dims: Vec<u64>,
    offset: u64,
    nbytes: u64,
}

const MAX_METADATA_ARRAY_ELEMENTS: u64 = 1_048_576;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IntegerType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
}

struct Cursor<R> {
    r: R,
    pos: u64,
}

impl<R: Read> Cursor<R> {
    fn take(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut v = Vec::new();
        v.try_reserve_exact(n)
            .map_err(|error| anyhow::anyhow!("cannot allocate {n}-byte GGUF field: {error}"))?;
        v.resize(n, 0);
        self.r.read_exact(&mut v)?;
        let n = u64::try_from(n).context("read length does not fit u64")?;
        self.pos = self
            .pos
            .checked_add(n)
            .context("GGUF header position overflows")?;
        Ok(v)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u64()?;
        if n > 64 * 1024 * 1024 {
            bail!("implausible string length {n}");
        }
        let n = usize::try_from(n).context("string length does not fit this host")?;
        String::from_utf8(self.take(n)?).context("GGUF string is not valid UTF-8")
    }
    fn value(&mut self, ty: u32) -> Result<Value> {
        Ok(match ty {
            0 => Value::U(self.take(1)?[0] as u64),
            1 => Value::I(self.take(1)?[0] as i8 as i64),
            2 => Value::U(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U(self.u32()? as u64),
            5 => Value::I(self.u32()? as i32 as i64),
            6 => Value::F(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => match self.take(1)?[0] {
                0 => Value::Bool(false),
                1 => Value::Bool(true),
                value => bail!("GGUF boolean must be encoded as 0 or 1, got {value}"),
            },
            8 => Value::Str(self.string()?),
            9 => {
                let et = self.u32()?;
                if et == 9 {
                    bail!("nested GGUF metadata arrays are unsupported");
                }
                let n = self.u64()?;
                if !matches!(et, 0..=8 | 10..=12) {
                    bail!("unknown metadata array element type {et}");
                }
                if n > MAX_METADATA_ARRAY_ELEMENTS {
                    bail!("implausible metadata array length {n}");
                }
                let n =
                    usize::try_from(n).context("metadata array length does not fit this host")?;
                let mut v = Vec::new();
                v.try_reserve_exact(n).map_err(|error| {
                    anyhow::anyhow!("cannot allocate {n}-element metadata array: {error}")
                })?;
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U(self.u64()?),
            11 => Value::I(self.u64()? as i64),
            12 => Value::F(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => bail!("unknown metadata type {other}"),
        })
    }
}

fn parse_header(file: &File) -> Result<Parsed> {
    // Buffered: a header can hold a 150k-entry tokenizer vocabulary, and reading that
    // eight bytes at a time through raw syscalls is the difference between 0.1 s and
    // minutes.
    let mut c = Cursor {
        r: BufReader::with_capacity(1 << 20, file),
        pos: 0,
    };
    let magic = c.take(4)?;
    if &magic != b"GGUF" {
        bail!("not a GGUF file (magic {magic:02x?})");
    }
    let version = c.u32()?;
    if !(2..=3).contains(&version) {
        bail!("unsupported GGUF version {version}");
    }
    let n_tensors = c.u64()?;
    let n_kv = c.u64()?;
    if n_kv > 16 * 1024 * 1024 {
        bail!("implausible metadata count {n_kv}");
    }

    let mut meta = BTreeMap::new();
    let mut integer_types = BTreeMap::new();
    for _ in 0..n_kv {
        let key = c.string()?;
        let ty = c.u32()?;
        let val = c.value(ty).with_context(|| format!("metadata key {key}"))?;
        if meta.insert(key.clone(), val).is_some() {
            bail!("duplicate GGUF metadata key {key}");
        }
        let integer_type = match ty {
            0 => Some(IntegerType::U8),
            1 => Some(IntegerType::I8),
            2 => Some(IntegerType::U16),
            3 => Some(IntegerType::I16),
            4 => Some(IntegerType::U32),
            5 => Some(IntegerType::I32),
            10 => Some(IntegerType::U64),
            11 => Some(IntegerType::I64),
            _ => None,
        };
        if let Some(integer_type) = integer_type {
            integer_types.insert(key, integer_type);
        }
    }

    if n_tensors > 16 * 1024 * 1024 {
        bail!("implausible tensor count {n_tensors}");
    }
    let tensor_count = usize::try_from(n_tensors).context("tensor count does not fit this host")?;
    let mut infos: Vec<(String, GgmlType, Vec<u64>, u64)> = Vec::new();
    infos.try_reserve_exact(tensor_count).map_err(|error| {
        anyhow::anyhow!("cannot allocate index for {n_tensors} tensors: {error}")
    })?;
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()? as usize;
        if !(1..=4).contains(&nd) {
            bail!("tensor {name} has {nd} dimensions");
        }
        let dims: Vec<u64> = (0..nd).map(|_| c.u64()).collect::<Result<_>>()?;
        let mut numel = 1u64;
        for &dim in &dims {
            if dim == 0 {
                bail!("tensor {name} has a zero dimension: {dims:?}");
            }
            numel = numel
                .checked_mul(dim)
                .with_context(|| format!("tensor {name} element count overflows: {dims:?}"))?;
        }
        let kind = GgmlType(c.u32()?);
        let off = c.u64()?;
        infos.push((name, kind, dims, off));
    }

    // Payload starts at the next multiple of general.alignment.
    let align = match meta.get("general.alignment") {
        Some(Value::U(value))
            if integer_types.get("general.alignment") == Some(&IntegerType::U32) =>
        {
            *value
        }
        Some(_) => bail!("general.alignment must be a positive UINT32"),
        None => 32,
    };
    if !align.is_power_of_two() {
        bail!("general.alignment must be a positive power of two, got {align}");
    }
    for (name, _, _, offset) in &infos {
        if !offset.is_multiple_of(align) {
            bail!(
                "tensor {name} payload offset {offset} is not aligned to general.alignment {align}"
            );
        }
    }
    let data_start = if infos.is_empty() {
        c.pos
    } else {
        let rounded = c
            .pos
            .checked_add(align - 1)
            .context("GGUF payload alignment overflows")?;
        (rounded / align)
            .checked_mul(align)
            .context("GGUF payload offset overflows")?
    };

    // GGUF tensors are tightly sequenced in index order, with each encoded payload
    // rounded up to `general.alignment` before the next tensor. Deriving lengths only
    // from offset gaps would let a corrupted type code shrink the apparent encoding and
    // make us decode a plausible-looking prefix of the wrong bytes.
    let file_len = file.metadata()?.len();
    let payload_len = file_len.checked_sub(data_start).with_context(|| {
        format!("GGUF payload begins at {data_start}, beyond the {file_len}-byte file")
    })?;
    let mut nbytes = Vec::new();
    nbytes
        .try_reserve_exact(infos.len())
        .map_err(|error| anyhow::anyhow!("cannot allocate tensor byte spans: {error}"))?;
    nbytes.resize(infos.len(), 0u64);
    let mut expected_offset = 0u64;
    let mut last_encoded_end = 0u64;
    for (i, (name, kind, dims, offset)) in infos.iter().enumerate() {
        if *offset != expected_offset {
            bail!(
                "tensor {name} has payload offset {offset}, expected {expected_offset} from the preceding encoded tensors"
            );
        }
        let (block_elements, block_bytes) = ggml_block_geometry(kind.0)
            .with_context(|| format!("tensor {name} has unknown GGML type {}", kind.0))?;
        let row_elements = dims[0];
        let block_elements =
            u64::try_from(block_elements).context("GGML block element count does not fit u64")?;
        let block_bytes =
            u64::try_from(block_bytes).context("GGML block byte count does not fit u64")?;
        if !row_elements.is_multiple_of(block_elements) {
            bail!(
                "tensor {name} has {row_elements} elements per row, not a multiple of {}'s {block_elements}-element block",
                kind.name()
            );
        }
        let numel = dims.iter().try_fold(1u64, |count, &dim| {
            count
                .checked_mul(dim)
                .with_context(|| format!("tensor {name} element count overflows: {dims:?}"))
        })?;
        let exact = (numel / block_elements)
            .checked_mul(block_bytes)
            .with_context(|| format!("tensor {name} encoded byte count overflows"))?;
        let encoded_end = offset
            .checked_add(exact)
            .with_context(|| format!("tensor {name} encoded byte range overflows"))?;
        if encoded_end > payload_len {
            bail!(
                "tensor {name} encoded range {offset}..{encoded_end} is beyond {payload_len} payload bytes"
            );
        }
        let padded = exact
            .checked_add(align - 1)
            .context("GGUF tensor padding overflows")?
            / align
            * align;
        expected_offset = offset
            .checked_add(padded)
            .with_context(|| format!("tensor {name} padded byte range overflows"))?;
        last_encoded_end = encoded_end;
        nbytes[i] = exact;
    }
    if !infos.is_empty() && !(last_encoded_end..=expected_offset).contains(&payload_len) {
        bail!(
            "GGUF payload is {payload_len} bytes, but encoded tensors end at {last_encoded_end} and permit padding only through {expected_offset}"
        );
    }

    let mut tensors = Vec::new();
    tensors
        .try_reserve_exact(infos.len())
        .map_err(|error| anyhow::anyhow!("cannot allocate parsed tensor index: {error}"))?;
    for (i, (name, kind, dims, off)) in infos.into_iter().enumerate() {
        let offset = data_start
            .checked_add(off)
            .with_context(|| format!("tensor {name} absolute offset overflows"))?;
        tensors.push((
            name,
            RawGgufTensor {
                kind,
                dims,
                offset,
                nbytes: nbytes[i],
            },
        ));
    }
    Ok(Parsed { meta, tensors })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a minimal but genuinely valid GGUF file.
    fn write_fixture(path: &Path, align: Option<u64>) {
        let mut b: Vec<u8> = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend(2u64.to_le_bytes()); // tensors
        let kvs: usize = if align.is_some() { 3 } else { 2 };
        b.extend((kvs as u64).to_le_bytes());

        let kv_str = |b: &mut Vec<u8>, k: &str, v: &str| {
            b.extend((k.len() as u64).to_le_bytes());
            b.extend(k.as_bytes());
            b.extend(8u32.to_le_bytes());
            b.extend((v.len() as u64).to_le_bytes());
            b.extend(v.as_bytes());
        };
        kv_str(&mut b, "general.architecture", "testarch");
        // an architecture-namespaced u32
        let k = "testarch.block_count";
        b.extend((k.len() as u64).to_le_bytes());
        b.extend(k.as_bytes());
        b.extend(4u32.to_le_bytes());
        b.extend(48u32.to_le_bytes());
        if let Some(a) = align {
            let k = "general.alignment";
            b.extend((k.len() as u64).to_le_bytes());
            b.extend(k.as_bytes());
            b.extend(4u32.to_le_bytes());
            b.extend(u32::try_from(a).unwrap().to_le_bytes());
        }

        let tensor_alignment = align.unwrap_or(32);
        let second_offset = align.map_or(64, |value| value.max(64));
        for (name, off) in [("a", 0u64), ("bb", second_offset)] {
            b.extend((name.len() as u64).to_le_bytes());
            b.extend(name.as_bytes());
            b.extend(2u32.to_le_bytes()); // 2 dims
            b.extend(4u64.to_le_bytes());
            b.extend(4u64.to_le_bytes());
            b.extend(0u32.to_le_bytes()); // F32
            b.extend(off.to_le_bytes());
        }

        while !(b.len() as u64).is_multiple_of(tensor_alignment) {
            b.push(0);
        }
        let payload_len = usize::try_from(second_offset + 64).unwrap();
        for value in 0..payload_len {
            b.push(value as u8);
        }
        File::create(path).unwrap().write_all(&b).unwrap();
    }

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("beholdd-gguf-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn parses_metadata_tensors_and_payload() {
        let d = tmpdir("basic");
        let p = d.join("m.gguf");
        write_fixture(&p, None);
        let g = Gguf::open(&p).unwrap();

        assert_eq!(g.arch(), Some("testarch"));
        assert_eq!(g.len(), 2);
        // The namespaced lookup must resolve through the architecture, not a hardcoded
        // prefix -- that is what makes one reader work for every family.
        assert_eq!(g.kv("block_count").unwrap().as_u64(), Some(48));
        assert!(g.kv("nonexistent").is_none());

        let a = g.get("a").unwrap();
        assert_eq!(a.dims, vec![4, 4]);
        assert_eq!(a.numel(), Some(16));
        assert_eq!(a.kind.name(), "F32");
        assert_eq!(
            a.nbytes(),
            64,
            "F32 shape determines the exact encoded size"
        );

        let mut buf = vec![0u8; 64];
        g.read_tensor_slice_into(a, 0..64, &mut buf).unwrap();
        assert_eq!(
            buf,
            (0u8..64).collect::<Vec<_>>(),
            "payload offset is wrong"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    /// Alignment shifts every tensor. Getting it wrong does not fail; it loads and
    /// generates nonsense, which is why it is read rather than assumed.
    #[test]
    fn honours_a_non_default_alignment() {
        let d = tmpdir("align");
        let p = d.join("m.gguf");
        write_fixture(&p, Some(4096));
        let g = Gguf::open(&p).unwrap();
        let a = g.get("a").unwrap();
        assert_eq!(
            a.offset() % 4096,
            0,
            "payload must start on the declared alignment"
        );
        let mut buf = vec![0u8; 64];
        g.read_tensor_slice_into(a, 0..64, &mut buf).unwrap();
        assert_eq!(buf, (0u8..64).collect::<Vec<_>>());
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn positional_ranges_and_tensor_slices_read_exact_bytes() {
        let d = tmpdir("ranges");
        let p = d.join("m.gguf");
        write_fixture(&p, None);
        let g = Gguf::open(&p).unwrap();
        let a = g.get("a").unwrap();

        let mut absolute = [0u8; 5];
        g.read_range_into(a.shard(), a.offset() + 7..a.offset() + 12, &mut absolute)
            .unwrap();
        assert_eq!(absolute, [7, 8, 9, 10, 11]);

        let mut relative = [0u8; 4];
        g.read_tensor_slice_into(a, 10..14, &mut relative).unwrap();
        assert_eq!(relative, [10, 11, 12, 13]);

        let mut empty = [];
        g.read_tensor_slice_into(a, a.nbytes()..a.nbytes(), &mut empty)
            .unwrap();
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn coalesced_checkpoint_spans_read_exact_bytes_through_a_dyn_reader() {
        let d = tmpdir("checkpoint-span");
        let p = d.join("m.gguf");
        write_fixture(&p, None);
        let g = Gguf::open(&p).unwrap();
        let a = g.get("a").unwrap();
        let bb = g.get("bb").unwrap();

        let mut span = CheckpointSpan::from(a.locator().whole_range());
        span.extend(bb.locator().whole_range()).unwrap();
        assert_eq!(span.start(), a.offset());
        assert_eq!(span.end(), bb.offset() + bb.nbytes());
        assert_eq!(span.nbytes(), 128);

        let reader: &dyn RangeReader = &g;
        let mut bytes = [0u8; 128];
        reader.read_span_into(&span, &mut bytes).unwrap();
        assert_eq!(bytes.as_slice(), (0u8..128).collect::<Vec<_>>());

        let error = reader
            .read_span_into(&span, &mut [0u8; 127])
            .unwrap_err()
            .to_string();
        assert!(error.contains("buffer is 127 bytes"), "got: {error}");
        assert!(error.contains("checkpoint span"), "got: {error}");

        let error = reader
            .read_span_into(&span, &mut [0u8; 129])
            .unwrap_err()
            .to_string();
        assert!(error.contains("buffer is 129 bytes"), "got: {error}");
        assert!(error.contains("checkpoint span"), "got: {error}");

        let empty = CheckpointSpan::from(bb.locator().range(bb.nbytes()..bb.nbytes()).unwrap());
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
        assert!(error.contains("reading GGUF weight range"), "got: {error}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn tensor_locators_cannot_be_reused_with_another_open_checkpoint() {
        let d = tmpdir("locator-identity");
        let p = d.join("m.gguf");
        write_fixture(&p, None);
        let first = Gguf::open(&p).unwrap();
        let second = Gguf::open(&p).unwrap();
        let tensor = first.get("a").unwrap();
        let mut byte = [0u8; 1];

        let error = second
            .read_tensor_slice_into(tensor, 0..1, &mut byte)
            .unwrap_err()
            .to_string();
        assert!(error.contains("belongs to checkpoint"), "got: {error}");
        assert!(error.contains("not source checkpoint"), "got: {error}");

        let foreign_span = CheckpointSpan::from(tensor.locator().range(0..1).unwrap());
        let error = RangeReader::read_span_into(&second, &foreign_span, &mut byte)
            .unwrap_err()
            .to_string();
        assert!(error.contains("checkpoint span belongs"), "got: {error}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn range_reads_reject_bad_lengths_bounds_shards_and_overflow() {
        let d = tmpdir("range-errors");
        let p = d.join("m.gguf");
        write_fixture(&p, None);
        let g = Gguf::open(&p).unwrap();
        let a = g.get("a").unwrap();

        let mut short = [0u8; 3];
        let err = g
            .read_tensor_slice_into(a, 0..4, &mut short)
            .unwrap_err()
            .to_string();
        assert!(err.contains("buffer is 3 bytes"), "got: {err}");
        assert!(err.contains("weight range"), "got: {err}");

        let mut eight = [0u8; 8];
        let err = g
            .read_tensor_slice_into(a, a.nbytes() - 4..a.nbytes() + 4, &mut eight)
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside the 64-byte weight"), "got: {err}");

        let mut none = [];
        let err = g
            .read_range_into(7, 0..0, &mut none)
            .unwrap_err()
            .to_string();
        assert!(err.contains("shard 7 does not exist"), "got: {err}");

        let reversed = Range { start: 9, end: 4 };
        let err = g
            .read_range_into(0, reversed, &mut none)
            .unwrap_err()
            .to_string();
        assert!(err.contains("invalid GGUF byte range 9..4"), "got: {err}");

        let file_len = std::fs::metadata(&p).unwrap().len();
        let err = g
            .read_range_into(0, file_len..file_len + 1, &mut [0])
            .unwrap_err()
            .to_string();
        assert!(err.contains("outside shard 0"), "got: {err}");

        let err = WeightLocator::new(g.checkpoint_id(), a.shard(), u64::MAX - 1, 4)
            .unwrap_err()
            .to_string();
        assert!(err.contains("weight byte extent overflows"), "got: {err}");

        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn generic_range_reader_enumerates_and_rejects_bad_shard_and_truncation() {
        let d = tmpdir("generic-reader");
        let p = d.join("m.gguf");
        write_fixture(&p, None);
        let g = Gguf::open(&p).unwrap();

        let mut locators: Vec<_> = RangeReader::locators(&g).collect();
        locators.sort_unstable_by_key(|locator| locator.start());
        assert_eq!(locators.len(), 2);
        assert_eq!(locators[0], g.get("a").unwrap().locator());

        let bad = WeightLocator::new(g.checkpoint_id(), 7, 0, 1)
            .unwrap()
            .whole_range();
        let error = RangeReader::read_range_into(&g, &bad, &mut [0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("names shard 7"), "got: {error}");

        let bad_span = CheckpointSpan::from(bad);
        let error = RangeReader::read_span_into(&g, &bad_span, &mut [0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("span names shard 7"), "got: {error}");

        let tensor = g.get("bb").unwrap();
        let range = tensor.locator().whole_range();
        let shortened = range.end() - 1;
        File::options()
            .write(true)
            .open(&p)
            .unwrap()
            .set_len(shortened)
            .unwrap();
        let mut bytes = vec![0; usize::try_from(range.nbytes()).unwrap()];
        let error = RangeReader::read_range_into(&g, &range, &mut bytes)
            .unwrap_err()
            .to_string();
        assert!(error.contains("reading GGUF weight range"), "got: {error}");

        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn malformed_payload_alignment_is_rejected_not_defaulted() {
        let d = tmpdir("alignment-invalid");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "general.alignment";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(10u32.to_le_bytes()); // u64
        bytes.extend(u64::MAX.to_le_bytes());
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("positive UINT32"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn wrongly_typed_alignment_is_rejected_not_defaulted() {
        let d = tmpdir("alignment-type");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "general.alignment";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(8u32.to_le_bytes()); // string, not an integer
        bytes.extend(2u64.to_le_bytes());
        bytes.extend(b"32");
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("general.alignment must be a positive UINT32"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn nested_metadata_arrays_are_rejected_without_recursion() {
        let d = tmpdir("nested-array");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "nested";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(9u32.to_le_bytes()); // array
        bytes.extend(9u32.to_le_bytes()); // nested array element type
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("nested GGUF metadata arrays are unsupported"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn implausible_split_count_is_rejected_before_allocation() {
        let d = tmpdir("shard-count");
        let p = d.join("m-000000001-of-999999999.gguf");
        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("implausible GGUF shard count"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn duplicate_metadata_keys_are_rejected() {
        let d = tmpdir("duplicate-metadata");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(2u64.to_le_bytes()); // metadata
        for value in [32u32, 64u32] {
            let key = "general.alignment";
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(4u32.to_le_bytes()); // u32
            bytes.extend(value.to_le_bytes());
        }
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("duplicate GGUF metadata key general.alignment"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn empty_array_with_unknown_element_type_is_rejected() {
        let d = tmpdir("unknown-empty-array");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "unknown";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(9u32.to_le_bytes()); // array
        bytes.extend(99u32.to_le_bytes()); // invalid element type
        bytes.extend(0u64.to_le_bytes()); // empty
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("unknown metadata array element type 99"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn noncanonical_boolean_encoding_is_rejected() {
        let d = tmpdir("boolean");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "flag";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(7u32.to_le_bytes()); // bool
        bytes.push(2);
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("boolean must be encoded as 0 or 1"),
            "got: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn alignment_requires_the_specified_uint32_type() {
        let d = tmpdir("alignment-u64");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "general.alignment";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(10u32.to_le_bytes()); // u64, forbidden by GGUF
        bytes.extend(32u64.to_le_bytes());
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("general.alignment must be a positive UINT32"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn tensor_offset_beyond_payload_is_rejected_during_open() {
        let d = tmpdir("offset-beyond-payload");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata
        let name = "bad";
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u32.to_le_bytes()); // F32
        bytes.extend(4096u64.to_le_bytes());
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend(0u32.to_le_bytes());
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("payload offset 4096, expected 0"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn unaligned_tensor_payload_offset_is_rejected() {
        let d = tmpdir("tensor-alignment");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes()); // tensors
        bytes.extend(0u64.to_le_bytes()); // metadata
        let name = "shifted";
        bytes.extend((name.len() as u64).to_le_bytes());
        bytes.extend(name.as_bytes());
        bytes.extend(1u32.to_le_bytes());
        bytes.extend(1u64.to_le_bytes());
        bytes.extend(0u32.to_le_bytes()); // F32
        bytes.extend(1u64.to_le_bytes()); // not aligned to default 32
        while !bytes.len().is_multiple_of(32) {
            bytes.push(0);
        }
        bytes.extend([0u8; 64]);
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let error = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(
            error.contains("payload offset 1 is not aligned"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn rejects_non_gguf() {
        let d = tmpdir("bad");
        let p = d.join("x.gguf");
        File::create(&p)
            .unwrap()
            .write_all(b"NOTGGUF_and_more")
            .unwrap();
        let e = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(e.contains("not a GGUF"), "got: {e}");
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn metadata_only_file_needs_no_trailing_payload_alignment() {
        let d = tmpdir("metadata-only");
        let p = d.join("m.gguf");
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(1u64.to_le_bytes()); // metadata
        let key = "general.architecture";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(8u32.to_le_bytes()); // string
        bytes.extend(4u64.to_le_bytes());
        bytes.extend(b"test");
        File::create(&p).unwrap().write_all(&bytes).unwrap();

        let checkpoint = Gguf::open(&p).unwrap();
        assert_eq!(checkpoint.arch(), Some("test"));
        assert!(checkpoint.is_empty());
        std::fs::remove_dir_all(&d).ok();
    }

    fn write_empty_split_shard(path: &Path, split_no: u32, split_count: u32) {
        let mut bytes = Vec::new();
        bytes.extend(b"GGUF");
        bytes.extend(3u32.to_le_bytes());
        bytes.extend(0u64.to_le_bytes()); // tensors
        bytes.extend(4u64.to_le_bytes()); // metadata

        let push_u32 = |bytes: &mut Vec<u8>, key: &str, value: u32| {
            bytes.extend((key.len() as u64).to_le_bytes());
            bytes.extend(key.as_bytes());
            bytes.extend(4u32.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        };
        let key = "general.architecture";
        bytes.extend((key.len() as u64).to_le_bytes());
        bytes.extend(key.as_bytes());
        bytes.extend(8u32.to_le_bytes());
        bytes.extend(4u64.to_le_bytes());
        bytes.extend(b"test");
        push_u32(&mut bytes, "split.no", split_no);
        push_u32(&mut bytes, "split.count", split_count);
        push_u32(&mut bytes, "split.tensors.count", 0);
        File::create(path).unwrap().write_all(&bytes).unwrap();
    }

    #[test]
    fn split_suffix_and_metadata_are_validated_together() {
        let d = tmpdir("split-metadata");
        let first = d.join("Lord-of-Rings-00001-of-00002.gguf");
        let second = d.join("Lord-of-Rings-00002-of-00002.gguf");
        write_empty_split_shard(&first, 0, 2);
        write_empty_split_shard(&second, 1, 2);

        let checkpoint = Gguf::open(&second).unwrap();
        assert_eq!(checkpoint.shards(), [first.clone(), second.clone()]);
        assert!(checkpoint.is_empty());

        write_empty_split_shard(&second, 0, 2);
        let error = format!("{:#}", Gguf::open(&first).unwrap_err());
        assert!(
            error.contains("declares split.no=0, expected 1"),
            "got: {error}"
        );
        std::fs::remove_dir_all(&d).ok();
    }

    #[test]
    fn a_missing_shard_is_named() {
        let d = tmpdir("split");
        let p = d.join("m-00001-of-00003.gguf");
        write_fixture(&p, None);
        let e = format!("{:#}", Gguf::open(&p).unwrap_err());
        assert!(e.contains("missing shard 2 of 3"), "got: {e}");
        std::fs::remove_dir_all(&d).ok();
    }
}
