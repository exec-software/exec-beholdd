// Distinct C16384 allocation / C2 CPU reference handoff. The frozen supervisor authenticates SHA-256
// identities; this module checks their exact attested bytes, not their hashes.
// No oracle, checkpoint payload source or CPU KV is retained by the reader.

use super::*;
use std::fs::{self, File, Metadata, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::Path;

const MAGIC: &[u8; 16] = b"HAGREF16K2F32\0\0\0";
const CATALOG_MAGIC: &[u8; 16] = b"HAGREFCAT32F32\0\0";
const HEADER_BYTES: usize = 256;
const IDENTITY_BYTES: usize = 128;
const BUFFER_BYTES: usize = 64 * 1024;
// Linux open(2) flags: reject a final symlink and never block on a raced FIFO.
const O_NOFOLLOW: i32 = 0x20000;
const O_NONBLOCK: i32 = 0x800;

/// Checkpoint, source-manifest, binary and contract SHA-256 attestation, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Identity([u8; IDENTITY_BYTES]);

#[derive(Clone, Copy)]
struct Layout {
    capacity: usize,
    vocab: usize,
    positions: &'static [usize],
    cpu_steps: usize,
    tokens: [u32; 2],
    catalog: bool,
}

impl Layout {
    fn production() -> Self {
        Self {
            capacity: CONTEXT,
            vocab: VOCAB,
            positions: &CHECK_POSITIONS,
            cpu_steps: CPU_STEPS,
            tokens: [1, 2],
            catalog: false,
        }
    }

    fn catalog(tokens: [u32; 2]) -> Self {
        Self {
            capacity: CONTEXT,
            vocab: VOCAB,
            positions: &catalog_prefill32::CHECK_POSITIONS,
            cpu_steps: catalog_prefill32::CPU_STEPS,
            tokens,
            catalog: true,
        }
    }

    fn record_bytes(self) -> Result<u32> {
        ensure!(
            self.capacity == CONTEXT
                && (1..=VOCAB).contains(&self.vocab)
                && self.tokens.iter().all(|&token| (token as usize) < VOCAB)
                && if self.catalog {
                    self.positions == catalog_prefill32::CHECK_POSITIONS
                        && self.cpu_steps == catalog_prefill32::CPU_STEPS
                } else {
                    self.positions == CHECK_POSITIONS
                        && self.cpu_steps == CPU_STEPS
                        && self.tokens == [1, 2]
                },
            "invalid reference layout"
        );
        u32::try_from(
            self.vocab
                .checked_mul(4)
                .and_then(|n| n.checked_add(16))
                .context("reference record size overflows")?,
        )
        .context("reference record exceeds u32")
    }

    fn file_bytes(self) -> Result<u64> {
        u64::from(self.record_bytes()?)
            .checked_mul(u64::try_from(self.positions.len())?)
            .and_then(|n| n.checked_add(HEADER_BYTES as u64))
            .context("reference file size overflows")
    }

    fn header(self, identity: &Identity) -> Result<[u8; HEADER_BYTES]> {
        let record_bytes = self.record_bytes()?;
        let mut bytes = [0; HEADER_BYTES];
        bytes[..16].copy_from_slice(if self.catalog { CATALOG_MAGIC } else { MAGIC });
        for (offset, value) in [
            (16, 1),
            (20, HEADER_BYTES as u32),
            (24, u32::try_from(self.capacity)?),
            (28, u32::try_from(self.vocab)?),
            (32, u32::try_from(self.positions.len())?),
            // Allocated native capacity and CPU evaluation count are distinct.
            (36, u32::try_from(self.cpu_steps)?),
            (40, record_bytes),
            (44, 0),
            (48, u32::try_from(self.positions[1])?),
            (52, u32::try_from(self.cpu_steps)?),
        ] {
            bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        }
        bytes[64..192].copy_from_slice(&identity.0);
        Ok(bytes)
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Stamp {
    dev: u64,
    ino: u64,
    len: u64,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl Stamp {
    fn from_metadata(value: Metadata) -> Result<Self> {
        ensure!(value.is_file(), "reference input is not a regular file");
        Ok(Self {
            dev: value.dev(),
            ino: value.ino(),
            len: value.len(),
            modified: (value.mtime(), value.mtime_nsec()),
            changed: (value.ctime(), value.ctime_nsec()),
        })
    }
}

fn canonical_file(path: &Path) -> Result<()> {
    ensure!(
        path.is_absolute() && fs::canonicalize(path)? == path,
        "reference path is not canonical"
    );
    ensure!(
        fs::symlink_metadata(path)?.is_file(),
        "reference path is not a regular file"
    );
    Ok(())
}

fn open_exact(path: &Path, bytes: u64) -> Result<(File, Stamp)> {
    canonical_file(path)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(O_NOFOLLOW | O_NONBLOCK)
        .open(path)?;
    let stamp = Stamp::from_metadata(file.metadata()?)?;
    ensure!(
        stamp.len == bytes,
        "reference file length differs: expected {bytes}, got {}",
        stamp.len
    );
    unchanged(&file, path, &stamp)?;
    Ok((file, stamp))
}

fn unchanged(file: &File, path: &Path, stamp: &Stamp) -> Result<()> {
    canonical_file(path)?;
    ensure!(
        Stamp::from_metadata(file.metadata()?)? == *stamp
            && Stamp::from_metadata(fs::symlink_metadata(path)?)? == *stamp,
        "reference file changed while reading"
    );
    Ok(())
}

fn eof(reader: &mut impl Read) -> Result<()> {
    ensure!(
        reader.read(&mut [0u8; 1])? == 0,
        "reference has trailing bytes"
    );
    Ok(())
}

pub(super) fn read_identity(path: &Path) -> Result<Identity> {
    let (mut file, stamp) = open_exact(path, IDENTITY_BYTES as u64)?;
    let mut bytes = [0; IDENTITY_BYTES];
    file.read_exact(&mut bytes)?;
    eof(&mut file)?;
    unchanged(&file, path, &stamp)?;
    Ok(Identity(bytes))
}

/// Streams selected CPU outputs into an unpublished, exclusively created file.
pub(super) struct ReferenceWriter {
    path: PathBuf,
    partial: PathBuf,
    writer: BufWriter<File>,
    layout: Layout,
    next: usize,
    dev: u64,
    ino: u64,
}

impl ReferenceWriter {
    pub(super) fn create(path: &Path, identity: &Identity) -> Result<Self> {
        Self::create_with_layout(path, identity, Layout::production())
    }

    pub(super) fn create_catalog(
        path: &Path,
        identity: &Identity,
        tokens: [u32; 2],
    ) -> Result<Self> {
        Self::create_with_layout(path, identity, Layout::catalog(tokens))
    }

    fn create_with_layout(path: &Path, identity: &Identity, layout: Layout) -> Result<Self> {
        let header = layout.header(identity)?;
        let parent = path.parent().context("reference output needs a parent")?;
        ensure!(
            path.is_absolute() && fs::canonicalize(parent)? == parent,
            "reference output parent is not canonical"
        );
        ensure!(
            path.file_name().is_some(),
            "reference output needs a filename"
        );
        match fs::symlink_metadata(path) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
            Ok(_) => bail!("reference output already exists"),
        }
        let mut name = path.file_name().unwrap().to_os_string();
        name.push(".partial");
        let partial = parent.join(name);
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(O_NOFOLLOW | O_NONBLOCK)
            .open(&partial)?;
        let metadata = file.metadata()?;
        let mut writer = BufWriter::with_capacity(BUFFER_BYTES, file);
        writer.write_all(&header)?;
        Ok(Self {
            path: path.to_path_buf(),
            partial,
            writer,
            layout,
            next: 0,
            dev: metadata.dev(),
            ino: metadata.ino(),
        })
    }

    pub(super) fn append(&mut self, position: usize, logits: &[f32]) -> Result<()> {
        ensure!(
            self.layout.positions.get(self.next) == Some(&position),
            "reference position/order differs"
        );
        ensure!(
            logits.len() == self.layout.vocab,
            "reference vocabulary length differs"
        );
        // Validation precedes mutation, including finite checks and first-index ties.
        let selected = greedy(logits)?;
        for value in [
            u32::try_from(position)?,
            self.layout.tokens[self.next],
            selected,
            u32::try_from(logits.len())?,
        ] {
            self.writer.write_all(&value.to_le_bytes())?;
        }
        for value in logits {
            self.writer.write_all(&value.to_le_bytes())?;
        }
        self.next += 1;
        Ok(())
    }

    pub(super) fn finish(mut self) -> Result<()> {
        ensure!(
            self.next == self.layout.positions.len(),
            "reference is incomplete"
        );
        self.writer.flush()?;
        self.writer.get_ref().sync_all()?;
        let stamp = Stamp::from_metadata(self.writer.get_ref().metadata()?)?;
        ensure!(
            stamp.dev == self.dev
                && stamp.ino == self.ino
                && stamp.len == self.layout.file_bytes()?,
            "reference output extent/identity differs"
        );
        unchanged(self.writer.get_ref(), &self.partial, &stamp)?;
        // hard_link fails if the destination already exists; rename could overwrite.
        fs::hard_link(&self.partial, &self.path)?;
        let published = Stamp::from_metadata(self.writer.get_ref().metadata()?)?;
        unchanged(self.writer.get_ref(), &self.path, &published)?;
        fs::remove_file(&self.partial)?;
        File::open(self.path.parent().unwrap())?.sync_all()?;
        Ok(())
    }
}

/// Immutable finite references only; no CPU model or KV survives in this owner.
pub(super) struct ReferenceSnapshot {
    positions: &'static [usize],
    logits: Vec<Vec<f32>>,
}

impl ReferenceSnapshot {
    pub(super) fn read(path: &Path, identity: &Identity) -> Result<Self> {
        Self::read_with_layout(path, identity, Layout::production())
    }

    pub(super) fn read_catalog(path: &Path, identity: &Identity, tokens: [u32; 2]) -> Result<Self> {
        Self::read_with_layout(path, identity, Layout::catalog(tokens))
    }

    fn read_with_layout(path: &Path, identity: &Identity, layout: Layout) -> Result<Self> {
        let expected_header = layout.header(identity)?;
        let (file, stamp) = open_exact(path, layout.file_bytes()?)?;
        let mut reader = BufReader::with_capacity(BUFFER_BYTES, file);
        let mut header = [0; HEADER_BYTES];
        reader.read_exact(&mut header)?;
        ensure!(
            header == expected_header,
            "reference header/version/identity/reserved bytes differ"
        );
        let mut logits = Vec::new();
        logits
            .try_reserve_exact(layout.positions.len())
            .context("allocating reference table")?;
        for (index, &position) in layout.positions.iter().enumerate() {
            let mut record = [0; 16];
            reader.read_exact(&mut record)?;
            let words: [u32; 4] = std::array::from_fn(|i| {
                u32::from_le_bytes(record[i * 4..i * 4 + 4].try_into().unwrap())
            });
            ensure!(
                words[0] as usize == position
                    && words[1] == layout.tokens[index]
                    && words[2] < layout.vocab as u32
                    && words[3] as usize == layout.vocab,
                "reference record position/token/greedy/count differs"
            );
            let mut values = Vec::new();
            values
                .try_reserve_exact(layout.vocab)
                .context("allocating reference logits")?;
            for _ in 0..layout.vocab {
                let mut bytes = [0; 4];
                reader.read_exact(&mut bytes)?;
                let value = f32::from_le_bytes(bytes);
                ensure!(value.is_finite(), "reference contains nonfinite logits");
                values.push(value);
            }
            ensure!(
                greedy(&values)? == words[2],
                "reference greedy does not match finite logits"
            );
            logits.push(values);
        }
        eof(&mut reader)?;
        unchanged(reader.get_ref(), path, &stamp)?;
        Ok(Self {
            positions: layout.positions,
            logits,
        })
    }

    pub(super) fn logits(&self, position: usize) -> Result<&[f32]> {
        let index = self
            .positions
            .iter()
            .position(|&p| p == position)
            .context("position has no checked reference")?;
        Ok(&self.logits[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    const TINY: Layout = Layout {
        capacity: CONTEXT,
        vocab: 3,
        positions: &CHECK_POSITIONS,
        cpu_steps: CPU_STEPS,
        tokens: [1, 2],
        catalog: false,
    };
    const VALUES: [[f32; 3]; 2] = [[-0.0, 0.0, -1.0], [2.0, 2.0, 1.0]];
    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            static NEXT: AtomicU64 = AtomicU64::new(0);
            let path = std::env::temp_dir().canonicalize().unwrap().join(format!(
                "beholdd-capacity16384-reference-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn identity() -> Identity {
        Identity(std::array::from_fn(|i| i as u8))
    }
    fn write_tiny(path: &Path) {
        let mut writer = ReferenceWriter::create_with_layout(path, &identity(), TINY).unwrap();
        for (&position, values) in CHECK_POSITIONS.iter().zip(&VALUES) {
            writer.append(position, values).unwrap();
        }
        writer.finish().unwrap();
    }

    #[test]
    fn reference_header_distinguishes_allocation_from_cpu_evaluations() {
        assert_eq!(Layout::production().file_bytes().unwrap(), 1_215_776);
        let header = Layout::production().header(&identity()).unwrap();
        assert_eq!(&header[..16], b"HAGREF16K2F32\0\0\0");
        for (offset, expected) in [
            (16, 1u32),
            (20, 256),
            (24, 16384),
            (28, 151936),
            (32, 2),
            (36, 2),
            (40, 607760),
            (44, 0),
            (48, 1),
            (52, 2),
        ] {
            assert_eq!(&header[offset..offset + 4], &expected.to_le_bytes());
        }
        assert_eq!(&header[64..192], &identity().0);
        assert!(header[56..64].iter().chain(&header[192..]).all(|&b| b == 0));
        assert!(Layout {
            capacity: 2048,
            ..TINY
        }
        .header(&identity())
        .is_err());
        assert!(Layout {
            positions: &[0],
            ..TINY
        }
        .header(&identity())
        .is_err());
        let directory = Directory::new();
        let path = directory.path("references.bin");
        write_tiny(&path);
        let snapshot = ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).unwrap();
        for (&position, expected) in CHECK_POSITIONS.iter().zip(&VALUES) {
            assert_eq!(
                snapshot
                    .logits(position)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.map(f32::to_bits)
            );
        }
        assert!(snapshot.logits(2).is_err());
        assert!(ReferenceSnapshot::read(&path, &identity()).is_err());
        assert!(!directory.path("references.bin.partial").exists());
    }

    #[test]
    fn catalog_layout_has_distinct_magic_positions_and_actual_input_tokens() {
        let layout = Layout {
            vocab: 3,
            ..Layout::catalog([2, 1])
        };
        let header = Layout::catalog([2, 1]).header(&identity()).unwrap();
        assert_eq!(&header[..16], b"HAGREFCAT32F32\0\0");
        assert_eq!(Layout::catalog([2, 1]).file_bytes().unwrap(), 1_215_776);
        for (offset, expected) in [(36, 32u32), (44, 0), (48, 31), (52, 32)] {
            assert_eq!(&header[offset..offset + 4], &expected.to_le_bytes());
        }
        let directory = Directory::new();
        let path = directory.path("catalog.bin");
        let mut writer = ReferenceWriter::create_with_layout(&path, &identity(), layout).unwrap();
        writer.append(0, &VALUES[0]).unwrap();
        assert!(writer.append(1, &VALUES[1]).is_err());
        writer.append(31, &VALUES[1]).unwrap();
        writer.finish().unwrap();
        let snapshot = ReferenceSnapshot::read_with_layout(&path, &identity(), layout).unwrap();
        assert_eq!(snapshot.logits(31).unwrap(), VALUES[1]);
        assert!(snapshot.logits(1).is_err());
        assert!(ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).is_err());
        assert!(ReferenceSnapshot::read_with_layout(
            &path,
            &identity(),
            Layout {
                tokens: [1, 2],
                ..layout
            }
        )
        .is_err());
        let original = fs::read(&path).unwrap();
        assert_eq!(&original[260..264], &2u32.to_le_bytes());
        assert_eq!(&original[288..292], &1u32.to_le_bytes());
        let mut changed = original.clone();
        changed[288..292].copy_from_slice(&32u32.to_le_bytes());
        fs::write(&path, changed).unwrap();
        assert!(ReferenceSnapshot::read_with_layout(&path, &identity(), layout).is_err());
        assert!(Layout {
            cpu_steps: 2,
            ..layout
        }
        .header(&identity())
        .is_err());
        assert!(Layout {
            positions: &[0, 1],
            ..layout
        }
        .header(&identity())
        .is_err());
        assert!(Layout::catalog([VOCAB as u32, 1])
            .header(&identity())
            .is_err());
    }

    #[test]
    fn reference_rejects_foreign_identity_header_corruption_and_bad_numeric_records() {
        let directory = Directory::new();
        let path = directory.path("references.bin");
        write_tiny(&path);
        let original = fs::read(&path).unwrap();
        for offset in [
            0, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64, 191, 192, 255, 256, 260, 264, 268,
            284,
        ] {
            let mut changed = original.clone();
            changed[offset] ^= 1;
            fs::write(&path, changed).unwrap();
            assert!(
                ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).is_err(),
                "offset {offset}"
            );
        }
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut changed = original.clone();
            changed[272..276].copy_from_slice(&invalid.to_le_bytes());
            fs::write(&path, changed).unwrap();
            assert!(ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).is_err());
        }
        for length in [0, 15, 255, 256, 272, original.len() - 1] {
            fs::write(&path, &original[..length]).unwrap();
            assert!(ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).is_err());
        }
        let mut trailing = original.clone();
        trailing.push(0);
        fs::write(&path, trailing).unwrap();
        assert!(ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).is_err());
        fs::write(&path, &original).unwrap();
        assert!(
            ReferenceSnapshot::read_with_layout(&path, &Identity([0; IDENTITY_BYTES]), TINY)
                .is_err()
        );
        let link = directory.path("symlink");
        symlink(&path, &link).unwrap();
        assert!(ReferenceSnapshot::read_with_layout(&link, &identity(), TINY).is_err());
        let mut wrong_tie = original;
        wrong_tie[264..268].copy_from_slice(&1u32.to_le_bytes());
        fs::write(&path, wrong_tie).unwrap();
        assert!(ReferenceSnapshot::read_with_layout(&path, &identity(), TINY).is_err());
    }

    #[test]
    fn publication_is_complete_exclusive_and_validation_precedes_append() {
        let directory = Directory::new();
        let path = directory.path("references.bin");
        let mut writer = ReferenceWriter::create_with_layout(&path, &identity(), TINY).unwrap();
        assert!(writer.append(1, &VALUES[0]).is_err());
        assert!(writer.append(0, &[f32::NAN, 1.0, 2.0]).is_err());
        assert!(writer.append(0, &[1.0]).is_err());
        writer.append(0, &VALUES[0]).unwrap();
        assert!(writer.finish().is_err());
        assert!(!path.exists());
        assert!(ReferenceWriter::create_with_layout(&path, &identity(), TINY).is_err());
        let path = directory.path("raced.bin");
        let mut writer = ReferenceWriter::create_with_layout(&path, &identity(), TINY).unwrap();
        for (&position, values) in CHECK_POSITIONS.iter().zip(&VALUES) {
            writer.append(position, values).unwrap();
        }
        assert!(writer.append(2, &VALUES[0]).is_err());
        fs::write(&path, b"keep").unwrap();
        assert!(writer.finish().is_err());
        assert_eq!(fs::read(&path).unwrap(), b"keep");
        assert!(ReferenceWriter::create(&path, &identity()).is_err());
    }

    #[test]
    fn identity_requires_exact_canonical_regular_file() {
        let directory = Directory::new();
        let path = directory.path("identity");
        fs::write(&path, identity().0).unwrap();
        assert_eq!(read_identity(&path).unwrap(), identity());
        fs::write(&path, [0; IDENTITY_BYTES - 1]).unwrap();
        assert!(read_identity(&path).is_err());
        fs::write(&path, [0; IDENTITY_BYTES + 1]).unwrap();
        assert!(read_identity(&path).is_err());
        assert!(read_identity(&directory.0).is_err());
    }
}
