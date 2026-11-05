//! Bounded exact-weight reads for the CPU `qwen35` reference.
//!
//! Q4_K/Q6_K CPU matvec currently expands its input tile to f32. Bound BOTH the
//! encoded read and this expanded tile before reading, rather than materializing
//! a multi-gigabyte vocabulary projection or an entire layer.

use anyhow::{ensure, Context, Result};
use beholdd_backend::{Device, Weight};
use beholdd_core::ggml_block_geometry;
use beholdd_format::{GgufTensor, WeightLocator};
use beholdd_memory::{WeightAccess, WeightSource};

/// Maximum per-read packed bytes and per-operation decoded scratch/output.
#[derive(Debug, Clone, Copy)]
pub struct ReadLimits {
    /// Encoded bytes held by one range read.
    pub encoded_bytes: usize,
    /// Decoded matvec tile or result allocation bytes (each counted separately).
    pub decoded_bytes: usize,
    /// Maximum rows per matvec; limits cancellation latency as well as memory.
    pub tile_rows: usize,
}

impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            encoded_bytes: 4 * 1024 * 1024,
            decoded_bytes: 8 * 1024 * 1024,
            tile_rows: 64,
        }
    }
}

impl ReadLimits {
    fn validate(self) -> Result<()> {
        ensure!(
            self.encoded_bytes > 0 && self.decoded_bytes > 0 && self.tile_rows > 0,
            "qwen35 read limits must be positive"
        );
        Ok(())
    }
}

fn zeros(len: usize, limit: usize) -> Result<Vec<f32>> {
    let bytes = len
        .checked_mul(4)
        .context("qwen35 decoded byte count overflow")?;
    ensure!(
        bytes <= limit,
        "qwen35 decoded allocation needs {bytes} bytes, limit is {limit}"
    );
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .context("allocating qwen35 decoded storage")?;
    out.resize(len, 0.0);
    Ok(out)
}

fn finite(values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "qwen35 weight operation produced a non-finite value"
    );
    Ok(())
}

struct Matrix {
    cols: usize,
    rows: usize,
    row_bytes: usize,
    rows_per_tile: usize,
    locator: WeightLocator,
}

impl Matrix {
    fn plan(
        tensor: &GgufTensor,
        source: &dyn WeightSource,
        device: &dyn Device,
        limits: ReadLimits,
    ) -> Result<Self> {
        limits.validate()?;
        ensure!(
            source.checkpoint_id() == tensor.checkpoint_id(),
            "qwen35 matrix belongs to a different checkpoint"
        );
        ensure!(
            tensor.dims().len() == 2 && tensor.dims().iter().all(|&d| d > 0),
            "qwen35 matrix must have two positive dimensions"
        );
        ensure!(
            device.supports(tensor.kind()),
            "{} cannot decode {}",
            device.name(),
            tensor.kind().name()
        );
        let cols = usize::try_from(tensor.dims()[0]).context("matrix columns exceed host size")?;
        let rows = usize::try_from(tensor.dims()[1]).context("matrix rows exceed host size")?;
        let (block, bytes) =
            ggml_block_geometry(tensor.kind().0).context("unknown packed matrix format")?;
        ensure!(
            cols.is_multiple_of(block),
            "qwen35 matrix row is not block aligned"
        );
        let row_bytes = (cols / block)
            .checked_mul(bytes)
            .context("matrix row byte count overflow")?;
        let total = row_bytes
            .checked_mul(rows)
            .context("matrix byte count overflow")?;
        ensure!(
            u64::try_from(total)? == tensor.nbytes(),
            "qwen35 matrix storage length mismatch"
        );
        let decoded_row = cols
            .checked_mul(4)
            .context("decoded row byte count overflow")?;
        let rows_per_tile = limits
            .tile_rows
            .min(limits.encoded_bytes / row_bytes)
            .min(limits.decoded_bytes / decoded_row);
        ensure!(
            rows_per_tile > 0,
            "one qwen35 matrix row exceeds the read/scratch limits"
        );
        Ok(Self {
            cols,
            rows,
            row_bytes,
            rows_per_tile,
            locator: tensor.locator(),
        })
    }
}

/// Exact row-tiled matrix-vector product; `control` runs before each bounded read.
pub fn matvec(
    tensor: &GgufTensor,
    source: &dyn WeightSource,
    device: &dyn Device,
    input: &[f32],
    limits: ReadLimits,
    control: &mut dyn FnMut() -> Result<()>,
) -> Result<Vec<f32>> {
    let m = Matrix::plan(tensor, source, device, limits)?;
    ensure!(input.len() == m.cols, "qwen35 matvec input width mismatch");
    finite(input)?;
    let mut out = zeros(m.rows, limits.decoded_bytes)?;
    for first in (0..m.rows).step_by(m.rows_per_tile) {
        control()?;
        let last = first.saturating_add(m.rows_per_tile).min(m.rows);
        let range = m
            .locator
            .range(u64::try_from(first * m.row_bytes)?..u64::try_from(last * m.row_bytes)?)?;
        let encoded = source.load_range(&range, WeightAccess::Cyclic)?;
        device.matvec(
            &mut out[first..last],
            input,
            &Weight {
                bytes: encoded.bytes(),
                kind: tensor.kind(),
                rows: last - first,
                cols: m.cols,
            },
        )?;
        finite(&out[first..last])?;
    }
    Ok(out)
}

/// Decode just the selected vocabulary row, without reading any other row.
pub fn embedding_row(
    tensor: &GgufTensor,
    row: usize,
    source: &dyn WeightSource,
    device: &dyn Device,
    limits: ReadLimits,
) -> Result<Vec<f32>> {
    let m = Matrix::plan(tensor, source, device, limits)?;
    ensure!(row < m.rows, "qwen35 token ID is outside the vocabulary");
    let range = m
        .locator
        .range(u64::try_from(row * m.row_bytes)?..u64::try_from((row + 1) * m.row_bytes)?)?;
    let encoded = source.load_range(&range, WeightAccess::Demand)?;
    let mut out = zeros(m.cols, limits.decoded_bytes)?;
    device.dequantize(
        &mut out,
        &Weight {
            bytes: encoded.bytes(),
            kind: tensor.kind(),
            rows: 1,
            cols: m.cols,
        },
    )?;
    finite(&out)?;
    Ok(out)
}

/// Read a small F32 norm, bias, convolution or recurrent scale tensor.
pub fn f32_tensor(
    tensor: &GgufTensor,
    source: &dyn WeightSource,
    limits: ReadLimits,
) -> Result<Vec<f32>> {
    limits.validate()?;
    ensure!(
        source.checkpoint_id() == tensor.checkpoint_id(),
        "qwen35 tensor belongs to a different checkpoint"
    );
    ensure!(
        tensor.kind().0 == 0,
        "qwen35 scalar/vector tensor must be F32"
    );
    let len = usize::try_from(
        tensor
            .numel()
            .context("F32 tensor element count overflow")?,
    )?;
    ensure!(
        len.checked_mul(4)
            .context("F32 tensor byte count overflow")?
            == usize::try_from(tensor.nbytes())?,
        "qwen35 F32 tensor storage mismatch"
    );
    ensure!(
        tensor.nbytes() <= u64::try_from(limits.encoded_bytes)?,
        "qwen35 F32 tensor exceeds read limit"
    );
    let mut out = zeros(len, limits.decoded_bytes)?;
    let encoded = source.load_range(&tensor.locator().whole_range(), WeightAccess::Cyclic)?;
    ensure!(
        encoded.bytes().len() == len * 4,
        "qwen35 source returned a short F32 tensor"
    );
    for (value, bytes) in out.iter_mut().zip(encoded.bytes().chunks_exact(4)) {
        *value = f32::from_le_bytes(bytes.try_into().expect("four-byte chunk"));
    }
    finite(&out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use beholdd_backend::cpu::Cpu;
    use beholdd_format::{CheckpointId, Gguf, WeightRange};
    use beholdd_memory::ResidentWeight;
    use std::sync::{
        atomic::{AtomicU64, Ordering},
        Mutex,
    };
    use std::{fs, path::PathBuf};

    struct Temporary(PathBuf);
    impl Drop for Temporary {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn fixture(kind: u32, cols: u64, rows: u64, payload: &[u8]) -> (Temporary, Gguf) {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "qwen35-weight-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&dir).unwrap();
        let mut b = b"GGUF".to_vec();
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&1u64.to_le_bytes());
        b.push(b'w');
        b.extend_from_slice(&2u32.to_le_bytes());
        b.extend_from_slice(&cols.to_le_bytes());
        b.extend_from_slice(&rows.to_le_bytes());
        b.extend_from_slice(&kind.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.resize((b.len() + 31) & !31, 0);
        b.extend_from_slice(payload);
        let path = dir.join("test.gguf");
        fs::write(&path, b).unwrap();
        let g = Gguf::open(path).unwrap();
        (Temporary(dir), g)
    }

    struct Spy<'a> {
        g: &'a Gguf,
        reads: Mutex<Vec<(usize, WeightAccess)>>,
    }
    impl WeightSource for Spy<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.g.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            self.reads
                .lock()
                .unwrap()
                .push((range.nbytes() as usize, access));
            ResidentWeight::load_range(self.g, range)
        }
    }

    // Independently encoded constant rows, not an invocation of the production decoder.
    fn constant_rows(kind: u32) -> Vec<u8> {
        let mut out = Vec::new();
        for v in 1u8..=3 {
            if kind == 12 {
                let mut b = vec![0; 144];
                b[..2].copy_from_slice(&0x3c00u16.to_le_bytes());
                b[4..8].fill(1);
                b[12..16].fill(1);
                b[16..].fill(v | (v << 4));
                out.extend(b);
            } else {
                let mut b = vec![0; 210];
                b[..128].fill(v | (v << 4));
                b[128..192].fill(0xaa);
                b[192..208].fill(1);
                b[208..].copy_from_slice(&0x3c00u16.to_le_bytes());
                out.extend(b);
            }
        }
        out
    }

    #[test]
    fn k_quantized_rows_tile_exactly_and_embedding_reads_one_row() -> Result<()> {
        for kind in [12, 14] {
            let (_tmp, g) = fixture(kind, 256, 3, &constant_rows(kind));
            let source = Spy {
                g: &g,
                reads: Mutex::new(Vec::new()),
            };
            let row_bytes = ggml_block_geometry(kind).unwrap().1;
            let limits = ReadLimits {
                encoded_bytes: row_bytes * 2,
                decoded_bytes: 2048,
                tile_rows: 64,
            };
            let tensor = g.get("w").unwrap();
            assert_eq!(
                matvec(tensor, &source, &Cpu, &[1.; 256], limits, &mut || Ok(()))?,
                vec![256., 512., 768.]
            );
            assert_eq!(
                *source.reads.lock().unwrap(),
                vec![
                    (row_bytes * 2, WeightAccess::Cyclic),
                    (row_bytes, WeightAccess::Cyclic)
                ]
            );
            source.reads.lock().unwrap().clear();
            assert_eq!(
                embedding_row(tensor, 1, &source, &Cpu, limits)?,
                vec![2.; 256]
            );
            assert_eq!(
                *source.reads.lock().unwrap(),
                vec![(row_bytes, WeightAccess::Demand)]
            );
        }
        Ok(())
    }

    #[test]
    fn refusal_precedes_read_and_cancel_bounds_io() -> Result<()> {
        let (_tmp, g) = fixture(12, 256, 3, &constant_rows(12));
        let source = Spy {
            g: &g,
            reads: Mutex::new(Vec::new()),
        };
        let tensor = g.get("w").unwrap();
        let limits = ReadLimits {
            encoded_bytes: 144,
            decoded_bytes: 1024,
            tile_rows: 1,
        };
        assert!(matvec(
            tensor,
            &source,
            &Cpu,
            &[1.; 256],
            ReadLimits {
                decoded_bytes: 1023,
                ..limits
            },
            &mut || Ok(())
        )
        .is_err());
        assert!(matvec(tensor, &source, &Cpu, &[1.; 255], limits, &mut || Ok(())).is_err());
        assert!(embedding_row(tensor, 3, &source, &Cpu, limits).is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        let mut calls = 0;
        assert!(matvec(tensor, &source, &Cpu, &[1.; 256], limits, &mut || {
            calls += 1;
            ensure!(calls < 2, "cancelled");
            Ok(())
        })
        .is_err());
        assert_eq!(
            *source.reads.lock().unwrap(),
            vec![(144, WeightAccess::Cyclic)]
        );
        let (_other_tmp, other) = fixture(12, 256, 3, &constant_rows(12));
        let foreign = Spy {
            g: &other,
            reads: Mutex::new(Vec::new()),
        };
        assert!(embedding_row(tensor, 0, &foreign, &Cpu, limits).is_err());
        assert!(foreign.reads.lock().unwrap().is_empty());
        Ok(())
    }

    #[test]
    fn f32_parameters_are_bounded_and_finite() -> Result<()> {
        let bytes: Vec<_> = [1f32, 2., 3., 4.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let (_tmp, g) = fixture(0, 4, 1, &bytes);
        let source = Spy {
            g: &g,
            reads: Mutex::new(Vec::new()),
        };
        let limits = ReadLimits {
            encoded_bytes: 16,
            decoded_bytes: 16,
            tile_rows: 1,
        };
        assert_eq!(
            f32_tensor(g.get("w").unwrap(), &source, limits)?,
            vec![1., 2., 3., 4.]
        );
        source.reads.lock().unwrap().clear();
        assert!(f32_tensor(
            g.get("w").unwrap(),
            &source,
            ReadLimits {
                decoded_bytes: 15,
                ..limits
            }
        )
        .is_err());
        assert!(source.reads.lock().unwrap().is_empty());
        let (_tmp2, bad) = fixture(0, 1, 1, &f32::NAN.to_le_bytes());
        let bad_source = Spy {
            g: &bad,
            reads: Mutex::new(Vec::new()),
        };
        assert!(f32_tensor(bad.get("w").unwrap(), &bad_source, limits).is_err());
        Ok(())
    }
}
