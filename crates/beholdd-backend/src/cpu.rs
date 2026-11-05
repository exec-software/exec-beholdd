//! The CPU device: the correctness reference, and a real target on unified memory.
//!
//! This implementation is never deleted and never allowed to rot. Every other backend is
//! checked against it, which is the discipline that made the previous engine trustworthy
//! — a GPU path with no reference is a GPU path nobody can debug.
//!
//! It is also not a toy. On a unified-memory part the CPU sees the same bandwidth the
//! iGPU does, and a streamed MoE spends most of a token moving expert bytes rather than
//! doing arithmetic on them.
//!
//! # Threading
//!
//! The row-parallel matvec below is the first real use of this crate's own answer to
//! that measurement. A weight's rows are an embarrassingly parallel axis: row `r` writes
//! only `y[r]`, from a self-contained dot product over `x` and row `r` of the
//! dequantised scratch, so splitting the row range across `std::thread::scope` workers
//! changes which thread performs a row's arithmetic and nothing about the arithmetic
//! itself. `worker_count` is the one place that decides whether a call is worth a
//! fork/join at all, using `beholdd_kernel::vecops::Granularity::worth_parallelising` —
//! the mechanism this measurement asked for, wired up for the first time.
//!
//! Still not here: a persistent worker pool (each call spawns and joins fresh
//! `std::thread::scope` workers; see `examples/bench_matvec_threads.rs` for whether that
//! overhead matters at real call volumes), fused packed kernels for the remaining K
//! formats (Q8_0 and BF16 are fused below), and parallelism across the routed-expert loop in
//! `beholdd-model` (that axis needs a fixed-order, thread-completion-independent
//! reduction into a shared accumulator, which row-splitting does not need).
//!
//! The original measurement still holds: on a 16-core SMT2 box, running many small
//! parallel regions on 32 threads instead of 16 cost 9.5x, while one big region (an
//! lm_head over 129,280 rows) was unchanged and a pure-vector control was 22% faster on
//! 32 — the penalty is per fork/join barrier, not per flop. That is why the row-parallel
//! path below only engages above `worth_parallelising`'s threshold, and why the per-head
//! vecops calls elsewhere in this file (rmsnorm, softmax, rope) are untouched.

use beholdd_format::GgmlType;
use beholdd_kernel::{bf16, fp8, kquant, mxfp4, vecops};

use crate::{AttentionConfig, Device, DeviceError, IncrementalAttentionConfig, RopeLayout, Weight};

/// Arithmetic on the host CPU.
#[derive(Debug, Default, Clone, Copy)]
pub struct Cpu;

impl Cpu {
    /// A new CPU device.
    pub fn new() -> Self {
        Cpu
    }

    /// Validate a packed weight's complete row geometry and return its element count.
    ///
    /// This check precedes both fused packed kernels and fallback scratch allocation, so
    /// malformed public weights fail before touching output or reserving memory.
    fn validate_storage(&self, w: &Weight<'_>) -> Result<usize, DeviceError> {
        let n = w.rows.checked_mul(w.cols).ok_or_else(|| {
            DeviceError::Shape(format!(
                "weight element count overflows for {}x{}",
                w.rows, w.cols
            ))
        })?;
        let (block_elements, block_bytes) =
            kquant::block_geometry(w.kind.0).ok_or_else(|| self.unsupported(w))?;
        if !w.cols.is_multiple_of(block_elements) {
            return Err(DeviceError::Shape(format!(
                "{}x{} {} weight has {} elements per row, not a whole number of \
                 {block_elements}-element blocks",
                w.rows,
                w.cols,
                w.cols,
                w.kind.name()
            )));
        }
        let blocks_per_row = w.cols / block_elements;
        let expected_bytes = w
            .rows
            .checked_mul(blocks_per_row)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| {
                DeviceError::Shape(format!(
                    "weight byte count overflows for {}x{} {}",
                    w.rows,
                    w.cols,
                    w.kind.name()
                ))
            })?;
        if w.bytes.len() != expected_bytes {
            return Err(DeviceError::Shape(format!(
                "{}x{} {} weight needs {expected_bytes} bytes, got {}",
                w.rows,
                w.cols,
                w.kind.name(),
                w.bytes.len()
            )));
        }
        Ok(n)
    }

    fn dequant(&self, w: &Weight<'_>, out: &mut [f32]) -> Result<(), DeviceError> {
        let n = self.validate_storage(w)?;
        if out.len() != n {
            return Err(DeviceError::Shape(format!(
                "expected {n} floats, got {}",
                out.len()
            )));
        }
        match w.kind.0 {
            0 => {
                for (o, c) in out.iter_mut().zip(w.bytes.chunks_exact(4)) {
                    *o = f32::from_le_bytes(c.try_into().unwrap());
                }
            }
            1 => {
                for (o, c) in out.iter_mut().zip(w.bytes.chunks_exact(2)) {
                    *o = beholdd_core::f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
                }
            }
            30 => {
                for (o, c) in out.iter_mut().zip(w.bytes.chunks_exact(2)) {
                    *o = beholdd_core::bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap()));
                }
            }
            8 => kquant::q8_0::dequant(w.bytes, out),
            12 => kquant::q4_k::dequant(w.bytes, out),
            13 => kquant::q5_k::dequant(w.bytes, out),
            14 => kquant::q6_k::dequant(w.bytes, out),
            _ => return Err(self.unsupported(w)),
        }
        Ok(())
    }

    fn unsupported(&self, w: &Weight<'_>) -> DeviceError {
        DeviceError::UnsupportedFormat {
            device: self.name(),
            kind: w.kind.name(),
            rows: w.rows,
            cols: w.cols,
        }
    }

    /// The host's logical CPU count as seen by this process (respects cgroup
    /// `AllowedCPUs`/cpuset affinity via `sched_getaffinity`), cached for the process
    /// lifetime -- `available_parallelism` is a syscall, and `run` calls this on the
    /// order of a thousand times per generated token.
    fn available_parallelism() -> usize {
        static AVAILABLE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *AVAILABLE.get_or_init(|| {
            std::thread::available_parallelism()
                .map(std::num::NonZeroUsize::get)
                .unwrap_or(1)
        })
    }

    /// A deliberate ceiling below the host's full logical CPU count.
    ///
    /// `examples/bench_matvec_threads.rs` measured this box's largest real shapes
    /// (wq/wo-scale, 4096x2048) plateauing between 8 and the full 32 logical CPUs --
    /// 17.77ms at 8 threads against 17.81ms at 32 -- so 8 threads already captures
    /// essentially all of the available speedup on the calls that matter most. Below
    /// that plateau, capping here is a deliberate choice, not just an unmeasured
    /// default: this engine runs on a host used for other concurrent work (compiling,
    /// gaming) at the same time, and leaving most cores free for that is worth more than
    /// a speedup this measurement already shows to be marginal past 8.
    const MAX_WORKER_THREADS: usize = 8;

    /// How many worker threads a row-parallel matvec over `rows x cols` should use.
    ///
    /// Returns 1 -- run on the calling thread, spawn nothing -- unless the call clears
    /// [`vecops::Granularity::worth_parallelising`]; a fork/join barrier for a call this
    /// small is exactly the many-small-regions cost this module's own doc comment warns
    /// about. That threshold alone is not enough, though: the same benchmark that
    /// justifies [`Cpu::MAX_WORKER_THREADS`] also measured this model's smallest
    /// above-threshold call (the router, 128 rows) getting *slower* at 32 threads than
    /// at 1 (0.54ms against 0.24ms) while 2-8 threads gave it a real, if modest, gain --
    /// the fork/join barrier cost outrunning the work being split. Otherwise returns
    /// `available`, clamped to [`Cpu::MAX_WORKER_THREADS`] and so that no thread is ever
    /// handed zero rows.
    fn worker_count(rows: usize, cols: usize, available: usize) -> usize {
        let granularity = vecops::Granularity {
            items: rows,
            flops_per_item: cols,
        };
        if !granularity.worth_parallelising() {
            return 1;
        }
        available
            .max(1)
            .min(rows.max(1))
            .min(Self::MAX_WORKER_THREADS)
    }

    /// `y[r] = scale * sum_i tmp[r*cols+i] * x[i]` for every row in `y`; accumulates
    /// (`+=`) rather than overwrites (`=`) when `accumulate` is set.
    ///
    /// The one place this arithmetic is written. The sequential call and every threaded
    /// chunk in [`Cpu::dot_rows`] run this exact function over whatever contiguous row
    /// range they were given.
    fn dot_row_range(
        y: &mut [f32],
        x: &[f32],
        tmp: &[f32],
        cols: usize,
        scale: f32,
        accumulate: bool,
    ) {
        for (r, out) in y.iter_mut().enumerate() {
            let row = &tmp[r * cols..(r + 1) * cols];
            let mut acc = 0.0f32;
            for (i, v) in row.iter().enumerate() {
                acc += v * x[i];
            }
            if accumulate {
                *out += acc * scale;
            } else {
                *out = acc * scale;
            }
        }
    }

    /// Compute every row of `y` from `tmp` and `x`, splitting the row range into at most
    /// `threads` disjoint, contiguous chunks, each run to completion by
    /// [`Cpu::dot_row_range`] on its own OS thread.
    ///
    /// Chunk sizes differ by at most one row (`rows / threads` each, the first
    /// `rows % threads` chunks taking one extra), so a row's absolute index -- and
    /// therefore which slice of `tmp`/`x` it reads and which single element of `y` it
    /// writes -- does not depend on how many chunks the range was split into. A row's
    /// computed value is a pure function of `(tmp[row], x, scale, accumulate,
    /// y[row]-before)`; `threads` does not appear in that function. That is the whole
    /// bit-exactness argument, and it is checked, not just asserted, by the
    /// `row_parallel_matvec_*` tests below and by `examples/bench_matvec_threads.rs`.
    fn dot_rows(
        y: &mut [f32],
        x: &[f32],
        tmp: &[f32],
        cols: usize,
        scale: f32,
        accumulate: bool,
        threads: usize,
    ) {
        let rows = y.len();
        let threads = threads.clamp(1, rows.max(1));
        if threads <= 1 {
            Self::dot_row_range(y, x, tmp, cols, scale, accumulate);
            return;
        }
        let base = rows / threads;
        let extra = rows % threads;
        std::thread::scope(|scope| {
            let (mut y_rest, mut tmp_rest) = (y, tmp);
            // Spawn every chunk but the last; the calling thread runs the last chunk
            // itself instead of spawning then immediately joining it -- one fewer OS
            // thread per call, and it runs the identical `dot_row_range` as every
            // spawned chunk.
            for chunk in 0..threads - 1 {
                let take = base + usize::from(chunk < extra);
                let (y_chunk, y_next) = y_rest.split_at_mut(take);
                let (tmp_chunk, tmp_next) = tmp_rest.split_at(take * cols);
                y_rest = y_next;
                tmp_rest = tmp_next;
                scope.spawn(move || {
                    Self::dot_row_range(y_chunk, x, tmp_chunk, cols, scale, accumulate);
                });
            }
            Self::dot_row_range(y_rest, x, tmp_rest, cols, scale, accumulate);
        });
    }

    /// Fused Q8_0 decode and dot product for a contiguous row range.
    fn q8_0_dot_row_range(
        y: &mut [f32],
        x: &[f32],
        packed: &[u8],
        row_bytes: usize,
        scale: f32,
        accumulate: bool,
    ) {
        for (out, row) in y.iter_mut().zip(packed.chunks_exact(row_bytes)) {
            let acc = kquant::q8_0::dot(row, x);
            if accumulate {
                *out += acc * scale;
            } else {
                *out = acc * scale;
            }
        }
    }

    /// Row-parallel fused Q8_0 matvec. Each worker decodes only the packed rows it owns,
    /// eliminating the full `rows * cols * sizeof(f32)` scratch allocation and making
    /// decode part of the same row-parallel region as the dot product.
    fn q8_0_dot_rows(
        y: &mut [f32],
        x: &[f32],
        packed: &[u8],
        row_bytes: usize,
        scale: f32,
        accumulate: bool,
        threads: usize,
    ) {
        let rows = y.len();
        let threads = threads.clamp(1, rows.max(1));
        if threads <= 1 {
            Self::q8_0_dot_row_range(y, x, packed, row_bytes, scale, accumulate);
            return;
        }

        let base = rows / threads;
        let extra = rows % threads;
        std::thread::scope(|scope| {
            let (mut y_rest, mut packed_rest) = (y, packed);
            for chunk in 0..threads - 1 {
                let take = base + usize::from(chunk < extra);
                let (y_chunk, y_next) = y_rest.split_at_mut(take);
                let (packed_chunk, packed_next) = packed_rest.split_at(take * row_bytes);
                y_rest = y_next;
                packed_rest = packed_next;
                scope.spawn(move || {
                    Self::q8_0_dot_row_range(
                        y_chunk,
                        x,
                        packed_chunk,
                        row_bytes,
                        scale,
                        accumulate,
                    );
                });
            }
            Self::q8_0_dot_row_range(y_rest, x, packed_rest, row_bytes, scale, accumulate);
        });
    }

    /// Fused BF16 decode and dot product for a contiguous row range.
    fn bf16_dot_row_range(
        y: &mut [f32],
        x: &[f32],
        packed: &[u8],
        row_bytes: usize,
        scale: f32,
        accumulate: bool,
    ) {
        for (out, row) in y.iter_mut().zip(packed.chunks_exact(row_bytes)) {
            let acc = bf16::dot_bytes(x, row);
            if accumulate {
                *out += acc * scale;
            } else {
                *out = acc * scale;
            }
        }
    }

    /// Row-parallel fused BF16 matvec over the packed little-endian GGUF bytes.
    fn bf16_dot_rows(
        y: &mut [f32],
        x: &[f32],
        packed: &[u8],
        row_bytes: usize,
        scale: f32,
        accumulate: bool,
        threads: usize,
    ) {
        let rows = y.len();
        let threads = threads.clamp(1, rows.max(1));
        if threads <= 1 {
            Self::bf16_dot_row_range(y, x, packed, row_bytes, scale, accumulate);
            return;
        }

        let base = rows / threads;
        let extra = rows % threads;
        std::thread::scope(|scope| {
            let (mut y_rest, mut packed_rest) = (y, packed);
            for chunk in 0..threads - 1 {
                let take = base + usize::from(chunk < extra);
                let (y_chunk, y_next) = y_rest.split_at_mut(take);
                let (packed_chunk, packed_next) = packed_rest.split_at(take * row_bytes);
                y_rest = y_next;
                packed_rest = packed_next;
                scope.spawn(move || {
                    Self::bf16_dot_row_range(
                        y_chunk,
                        x,
                        packed_chunk,
                        row_bytes,
                        scale,
                        accumulate,
                    );
                });
            }
            Self::bf16_dot_row_range(y_rest, x, packed_rest, row_bytes, scale, accumulate);
        });
    }

    /// The shared body of `matvec` and `matvec_add`.
    ///
    /// `scale` multiplies the result before it lands, which is what an MoE's router
    /// weight is. Folding it here rather than in a separate pass over `y` saves a full
    /// traversal of the output per expert.
    fn run(
        &self,
        y: &mut [f32],
        x: &[f32],
        w: &Weight<'_>,
        scale: f32,
        accumulate: bool,
    ) -> Result<(), DeviceError> {
        let threads = Self::worker_count(w.rows, w.cols, Self::available_parallelism());
        self.run_with_threads(y, x, w, scale, accumulate, threads)
    }

    /// As [`Cpu::run`], but with the row-parallel worker count supplied by the caller
    /// instead of decided by [`Cpu::worker_count`]. [`Cpu::run`] is every real caller's
    /// path (host-decided thread count); [`Cpu::matvec_with_threads`] and
    /// [`Cpu::matvec_add_with_threads`] call this directly so a benchmark or test can
    /// hold thread count constant while shape varies.
    fn run_with_threads(
        &self,
        y: &mut [f32],
        x: &[f32],
        w: &Weight<'_>,
        scale: f32,
        accumulate: bool,
        threads: usize,
    ) -> Result<(), DeviceError> {
        if x.len() != w.cols || y.len() != w.rows {
            return Err(DeviceError::Shape(format!(
                "weight is {}x{}, got x={} y={}",
                w.rows,
                w.cols,
                x.len(),
                y.len()
            )));
        }

        // Validate the packed span before either taking a fused path or reserving the
        // fallback f32 scratch. A malformed public `Weight` must not turn a three-byte
        // input into a multi-gigabyte allocation.
        let weight_len = self.validate_storage(w)?;

        // A rows-by-zero matrix is a valid empty reduction in the established CPU
        // contract. Handle it before packed row chunking: `chunks_exact(0)` panics,
        // whereas the prior dequantize-then-dot path produced scale * 0 for every row.
        if w.cols == 0 {
            let result = 0.0f32 * scale;
            for out in y {
                if accumulate {
                    *out += result;
                } else {
                    *out = result;
                }
            }
            return Ok(());
        }

        // Q8_0 is the selected native-v2 checkpoint's format. Decode it inside each
        // row's dot product: this preserves the reference operation order while avoiding
        // an allocation and two full passes over a 4x larger f32 matrix.
        if w.kind.0 == 8 {
            let blocks_per_row = w.cols / kquant::q8_0::ELEMS;
            let row_bytes = blocks_per_row
                .checked_mul(kquant::q8_0::BYTES)
                .ok_or_else(|| DeviceError::Shape("Q8_0 row byte count overflows".into()))?;
            Self::q8_0_dot_rows(y, x, w.bytes, row_bytes, scale, accumulate, threads);
            return Ok(());
        }

        // BF16 is already the native compute fidelity the caller asked for. Widen each
        // value in-register as its row is consumed instead of materialising an f32 copy
        // of the complete matrix. This removes the expanded matrix's four-byte write
        // and four-byte read, plus four bytes of transient allocation per parameter.
        if w.kind.0 == 30 {
            let row_bytes = w
                .cols
                .checked_mul(2)
                .ok_or_else(|| DeviceError::Shape("BF16 row byte count overflows".into()))?;
            Self::bf16_dot_rows(y, x, w.bytes, row_bytes, scale, accumulate, threads);
            return Ok(());
        }

        // DEQUANTISE-THEN-MULTIPLY for the remaining GGUF block formats.
        //
        // The packed-byte kernels (mxfp4, fp8, bf16) never materialise a float matrix,
        // which is the entire reason a streamed expert stays 12.6 MB instead of 100 MB.
        // They cannot be used here yet because they take the scale grid as a separate
        // argument, which is how SAFETENSORS stores it; GGUF interleaves scales into
        // each block of the row. Wiring them up means either a GGUF-shaped kernel or a
        // repack at load, and choosing between those before there is a forward pass to
        // measure is exactly the guess this project avoids.
        let scratch_bytes = weight_len
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                DeviceError::Shape(format!(
                    "f32 scratch byte count overflows for {}x{}",
                    w.rows, w.cols
                ))
            })?;
        let mut tmp = Vec::new();
        tmp.try_reserve_exact(weight_len)
            .map_err(|_| DeviceError::Allocation {
                device: self.name(),
                bytes: scratch_bytes,
                purpose: "dequantized weight scratch",
            })?;
        tmp.resize(weight_len, 0.0f32);
        self.dequant(w, &mut tmp)?;
        Self::dot_rows(y, x, &tmp, w.cols, scale, accumulate, threads);
        Ok(())
    }
}

impl Device for Cpu {
    fn name(&self) -> &'static str {
        "cpu"
    }

    fn supports(&self, kind: GgmlType) -> bool {
        kquant::block_geometry(kind.0).is_some()
    }

    fn dequantize(&self, out: &mut [f32], weight: &Weight<'_>) -> Result<(), DeviceError> {
        self.dequant(weight, out)
    }

    fn matvec(&self, y: &mut [f32], x: &[f32], w: &Weight<'_>) -> Result<(), DeviceError> {
        self.run(y, x, w, 1.0, false)
    }

    fn matvec_add(
        &self,
        y: &mut [f32],
        x: &[f32],
        w: &Weight<'_>,
        scale: f32,
    ) -> Result<(), DeviceError> {
        self.run(y, x, w, scale, true)
    }

    fn rmsnorm(
        &self,
        y: &mut [f32],
        x: &[f32],
        weight: Option<&[f32]>,
        eps: f32,
    ) -> Result<(), DeviceError> {
        if y.len() != x.len() {
            return Err(DeviceError::Shape(format!(
                "RMSNorm expected y={} to match x={}",
                x.len(),
                y.len()
            )));
        }
        if let Some(weight) = weight {
            if weight.len() != x.len() {
                return Err(DeviceError::Shape(format!(
                    "RMSNorm expected {} weights, got {}",
                    x.len(),
                    weight.len()
                )));
            }
        }
        if x.is_empty() {
            return Err(DeviceError::Shape("RMSNorm input cannot be empty".into()));
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(DeviceError::Shape(format!(
                "RMSNorm epsilon must be finite and non-negative, got {eps}"
            )));
        }
        vecops::rmsnorm(y, x, weight, eps);
        Ok(())
    }

    fn softmax(&self, v: &mut [f32]) -> Result<(), DeviceError> {
        if v.is_empty() {
            return Err(DeviceError::Shape("softmax input cannot be empty".into()));
        }
        if v.iter().any(|value| value.is_nan()) {
            return Err(DeviceError::Shape(
                "softmax input cannot contain NaN".into(),
            ));
        }
        vecops::softmax(v);
        Ok(())
    }

    fn causal_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        config: AttentionConfig,
    ) -> Result<(), DeviceError> {
        let AttentionConfig {
            n_positions,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
        } = config;
        if n_positions == 0 {
            return Err(DeviceError::Shape(
                "causal attention requires at least one position".into(),
            ));
        }
        if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
            return Err(DeviceError::Shape(format!(
                "causal attention dimensions must be non-zero, got heads={n_heads} \
                 kv_heads={n_kv_heads} head_dim={head_dim}"
            )));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(DeviceError::Shape(format!(
                "causal attention needs query heads divisible by kv heads, got \
                 {n_heads} and {n_kv_heads}"
            )));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(DeviceError::Shape(format!(
                "causal attention scale must be positive and finite, got {scale}"
            )));
        }

        let elements = |heads: usize| {
            n_positions
                .checked_mul(heads)
                .and_then(|n| n.checked_mul(head_dim))
                .ok_or_else(|| {
                    DeviceError::Shape(format!(
                        "causal attention shape overflows: positions={n_positions} \
                         heads={heads} head_dim={head_dim}"
                    ))
                })
        };
        let q_len = elements(n_heads)?;
        let kv_len = elements(n_kv_heads)?;
        if out.len() != q_len || q.len() != q_len || k.len() != kv_len || v.len() != kv_len {
            return Err(DeviceError::Shape(format!(
                "causal attention expected out={q_len} q={q_len} k={kv_len} v={kv_len}, \
                 got out={} q={} k={} v={}",
                out.len(),
                q.len(),
                k.len(),
                v.len()
            )));
        }

        let group = n_heads / n_kv_heads;
        let score_bytes = n_positions
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| DeviceError::Shape("causal attention score bytes overflow".into()))?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(n_positions)
            .map_err(|_| DeviceError::Allocation {
                device: self.name(),
                bytes: score_bytes,
                purpose: "causal attention scores",
            })?;
        scores.resize(n_positions, 0.0f32);
        for position in 0..n_positions {
            for head in 0..n_heads {
                let kv_head = head / group;
                let q_start = (position * n_heads + head) * head_dim;
                let query = &q[q_start..q_start + head_dim];

                for (key_position, score) in scores[..=position].iter_mut().enumerate() {
                    let k_start = (key_position * n_kv_heads + kv_head) * head_dim;
                    let key = &k[k_start..k_start + head_dim];
                    let dot: f64 = query
                        .iter()
                        .zip(key)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum();
                    *score = (dot * scale as f64) as f32;
                }
                vecops::softmax(&mut scores[..=position]);

                let out_start = (position * n_heads + head) * head_dim;
                for dimension in 0..head_dim {
                    let mut sum = 0.0f64;
                    for (value_position, probability) in scores[..=position].iter().enumerate() {
                        let v_index =
                            (value_position * n_kv_heads + kv_head) * head_dim + dimension;
                        sum += *probability as f64 * v[v_index] as f64;
                    }
                    out[out_start + dimension] = sum as f32;
                }
            }
        }
        Ok(())
    }

    fn incremental_causal_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        config: IncrementalAttentionConfig,
    ) -> Result<(), DeviceError> {
        let IncrementalAttentionConfig {
            n_query_positions,
            query_start,
            n_kv_positions,
            n_heads,
            n_kv_heads,
            head_dim,
            scale,
        } = config;
        if n_query_positions == 0 {
            return Err(DeviceError::Shape(
                "incremental causal attention requires at least one query position".into(),
            ));
        }
        if n_kv_positions == 0 {
            return Err(DeviceError::Shape(
                "incremental causal attention requires at least one cached position".into(),
            ));
        }
        if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
            return Err(DeviceError::Shape(format!(
                "incremental causal attention dimensions must be non-zero, got heads={n_heads} \
                 kv_heads={n_kv_heads} head_dim={head_dim}"
            )));
        }
        if !n_heads.is_multiple_of(n_kv_heads) {
            return Err(DeviceError::Shape(format!(
                "incremental causal attention needs query heads divisible by kv heads, got \
                 {n_heads} and {n_kv_heads}"
            )));
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(DeviceError::Shape(format!(
                "incremental causal attention scale must be positive and finite, got {scale}"
            )));
        }

        let query_end = query_start.checked_add(n_query_positions).ok_or_else(|| {
            DeviceError::Shape(format!(
                "incremental causal attention query range overflows: start={query_start} \
                 positions={n_query_positions}"
            ))
        })?;
        if query_end > n_kv_positions {
            return Err(DeviceError::Shape(format!(
                "incremental causal attention query range {query_start}..{query_end} exceeds \
                 {n_kv_positions} cached positions"
            )));
        }

        let query_len = n_query_positions
            .checked_mul(n_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(|| {
                DeviceError::Shape(format!(
                    "incremental causal attention query shape overflows: \
                     positions={n_query_positions} heads={n_heads} head_dim={head_dim}"
                ))
            })?;
        let cache_len = n_kv_positions
            .checked_mul(n_kv_heads)
            .and_then(|n| n.checked_mul(head_dim))
            .ok_or_else(|| {
                DeviceError::Shape(format!(
                    "incremental causal attention cache shape overflows: \
                     positions={n_kv_positions} heads={n_kv_heads} head_dim={head_dim}"
                ))
            })?;
        if out.len() != query_len
            || q.len() != query_len
            || k_cache.len() != cache_len
            || v_cache.len() != cache_len
        {
            return Err(DeviceError::Shape(format!(
                "incremental causal attention expected out={query_len} q={query_len} \
                 k_cache={cache_len} v_cache={cache_len}, got out={} q={} k_cache={} v_cache={}",
                out.len(),
                q.len(),
                k_cache.len(),
                v_cache.len()
            )));
        }

        // The last query can see exactly `query_end` keys. Reserve all score storage
        // before touching output, so allocation failure has the same all-or-nothing
        // behavior as shape refusal.
        let score_bytes = query_end
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| {
                DeviceError::Shape("incremental causal attention score byte count overflows".into())
            })?;
        let mut scores = Vec::new();
        scores
            .try_reserve_exact(query_end)
            .map_err(|_| DeviceError::Allocation {
                device: self.name(),
                bytes: score_bytes,
                purpose: "incremental causal attention scores",
            })?;
        scores.resize(query_end, 0.0f32);

        let group = n_heads / n_kv_heads;
        for query_position in 0..n_query_positions {
            let absolute_position = query_start + query_position;
            for head in 0..n_heads {
                let kv_head = head / group;
                let q_start = (query_position * n_heads + head) * head_dim;
                let query = &q[q_start..q_start + head_dim];

                for (key_position, score) in scores[..=absolute_position].iter_mut().enumerate() {
                    let k_start = (key_position * n_kv_heads + kv_head) * head_dim;
                    let key = &k_cache[k_start..k_start + head_dim];
                    let dot: f64 = query
                        .iter()
                        .zip(key)
                        .map(|(a, b)| *a as f64 * *b as f64)
                        .sum();
                    *score = (dot * scale as f64) as f32;
                }
                vecops::softmax(&mut scores[..=absolute_position]);

                let out_start = (query_position * n_heads + head) * head_dim;
                for dimension in 0..head_dim {
                    let mut sum = 0.0f64;
                    for (value_position, probability) in
                        scores[..=absolute_position].iter().enumerate()
                    {
                        let v_index =
                            (value_position * n_kv_heads + kv_head) * head_dim + dimension;
                        sum += *probability as f64 * v_cache[v_index] as f64;
                    }
                    out[out_start + dimension] = sum as f32;
                }
            }
        }
        Ok(())
    }

    fn add_in_place(&self, output: &mut [f32], residual: &[f32]) -> Result<(), DeviceError> {
        if output.len() != residual.len() {
            return Err(DeviceError::Shape(format!(
                "residual add expected output={} to match residual={}",
                output.len(),
                residual.len()
            )));
        }
        for (value, residual) in output.iter_mut().zip(residual) {
            *value += residual;
        }
        Ok(())
    }

    fn swiglu(&self, gate: &mut [f32], up: &[f32], limit: f32) -> Result<(), DeviceError> {
        if gate.len() != up.len() {
            return Err(DeviceError::Shape(format!(
                "SwiGLU expected gate={} to match up={}",
                gate.len(),
                up.len()
            )));
        }
        if !limit.is_finite() || limit < 0.0 {
            return Err(DeviceError::Shape(format!(
                "SwiGLU limit must be finite and non-negative, got {limit}"
            )));
        }
        vecops::swiglu(gate, up, limit);
        Ok(())
    }

    fn rope(
        &self,
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        layout: RopeLayout,
        inverse: bool,
    ) -> Result<(), DeviceError> {
        let expected = cos
            .len()
            .checked_mul(2)
            .ok_or_else(|| DeviceError::Shape("RoPE coordinate count overflows".into()))?;
        if sin.len() != cos.len() || x.len() != expected {
            return Err(DeviceError::Shape(format!(
                "RoPE expected x={expected}, cos={} and sin={}, got x={} cos={} sin={}",
                cos.len(),
                cos.len(),
                x.len(),
                cos.len(),
                sin.len()
            )));
        }
        match layout {
            RopeLayout::Interleaved => vecops::rope_apply(x, cos, sin, inverse),
            RopeLayout::SplitHalf => vecops::rope_apply_split_half(x, cos, sin, inverse),
        }
        Ok(())
    }
}

impl Cpu {
    /// As [`Device::matvec`], but with the row-parallel worker count fixed at exactly
    /// `threads` rather than decided by `Cpu::worker_count`.
    ///
    /// Not part of the `Device` trait -- see this module's "Threading" section. Exists
    /// so a benchmark or test can hold thread count constant while shape varies, or vice
    /// versa. Every real caller uses [`Device::matvec`].
    pub fn matvec_with_threads(
        &self,
        y: &mut [f32],
        x: &[f32],
        w: &Weight<'_>,
        threads: usize,
    ) -> Result<(), DeviceError> {
        self.run_with_threads(y, x, w, 1.0, false, threads)
    }

    /// As [`Device::matvec_add`], but with the row-parallel worker count fixed at
    /// exactly `threads`. See [`Cpu::matvec_with_threads`].
    pub fn matvec_add_with_threads(
        &self,
        y: &mut [f32],
        x: &[f32],
        w: &Weight<'_>,
        scale: f32,
        threads: usize,
    ) -> Result<(), DeviceError> {
        self.run_with_threads(y, x, w, scale, true, threads)
    }
}

/// Keep the direct-kernel paths referenced so they cannot rot unnoticed.
///
/// `mxfp4`, `fp8` and `bf16` matvecs consume packed bytes without materialising a float
/// matrix. GGUF Q8_0 and BF16 now have fused row kernels above; other K formats still
/// take the dequantise-then-dot fallback.
#[allow(dead_code)]
fn direct_kernels_are_live() {
    let _ = mxfp4::matvec;
    let _ = fp8::matvec;
    let _ = bf16::matvec;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bits(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let man = ((bits >> 13) & 0x3FF) as u16;
        sign | ((exp as u16) << 10) | man
    }

    /// One Q8_0 block: scale 0.5, quants 1..32 in the first row.
    fn q8_row(scale: f32, vals: &[i8]) -> Vec<u8> {
        let mut v = Vec::new();
        for chunk in vals.chunks(32) {
            v.extend_from_slice(&f16_bits(scale).to_le_bytes());
            for i in 0..32 {
                v.push(*chunk.get(i).unwrap_or(&0) as u8);
            }
        }
        v
    }

    fn bf16_matrix(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    #[test]
    fn matvec_overwrites_and_matvec_add_accumulates() {
        let cpu = Cpu::new();
        let vals: Vec<i8> = (0..32).map(|i| (i % 5) as i8).collect();
        let bytes = q8_row(0.5, &vals);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 1,
            cols: 32,
        };
        let x = vec![1.0f32; 32];
        let expect: f32 = vals.iter().map(|v| 0.5 * *v as f32).sum();

        let mut y = [99.0f32];
        cpu.matvec(&mut y, &x, &w).unwrap();
        assert_eq!(y[0], expect, "matvec must overwrite");

        let mut y2 = [10.0f32];
        cpu.matvec_add(&mut y2, &x, &w, 2.0).unwrap();
        assert_eq!(
            y2[0],
            10.0 + expect * 2.0,
            "matvec_add must accumulate and scale"
        );
    }

    #[test]
    fn fused_bf16_matvec_matches_its_scalar_definition_and_accumulates() {
        let cpu = Cpu::new();
        let rows = 5;
        let cols = 100;
        let values: Vec<f32> = (0..rows * cols)
            .map(|index| ((index as f32) * 0.031).sin() * 0.5)
            .collect();
        let bytes = bf16_matrix(&values);
        let weight = Weight {
            bytes: &bytes,
            kind: GgmlType(30),
            rows,
            cols,
        };
        let x: Vec<f32> = (0..cols)
            .map(|index| ((index as f32) * 0.019).cos() * 0.25)
            .collect();
        let expected: Vec<f32> = bytes
            .chunks_exact(cols * 2)
            .map(|row| bf16::dot_bytes_scalar(&x, row))
            .collect();

        let mut actual = vec![0.0f32; rows];
        cpu.matvec(&mut actual, &x, &weight).unwrap();
        for (actual, expected) in actual.iter().zip(&expected) {
            let scale = expected.abs().max(1.0);
            assert!((actual - expected).abs() / scale < 1e-5);
        }

        let mut parallel = vec![0.0f32; rows];
        cpu.matvec_with_threads(&mut parallel, &x, &weight, 4)
            .unwrap();
        for (parallel, actual) in parallel.iter().zip(&actual) {
            assert_eq!(parallel.to_bits(), actual.to_bits());
        }

        let before = actual.clone();
        cpu.matvec_add(&mut actual, &x, &weight, 0.25).unwrap();
        for ((actual, before), expected) in actual.iter().zip(&before).zip(&expected) {
            let wanted = before + expected * 0.25;
            let scale = wanted.abs().max(1.0);
            assert!((actual - wanted).abs() / scale < 1e-5);
        }
    }

    #[test]
    fn fused_packed_matvecs_preserve_zero_width_reductions_without_panicking() {
        let cpu = Cpu::new();
        for kind in [GgmlType(8), GgmlType(30)] {
            let weight = Weight {
                bytes: &[],
                kind,
                rows: 3,
                cols: 0,
            };

            let mut overwritten = vec![7.0f32; 3];
            cpu.matvec(&mut overwritten, &[], &weight).unwrap();
            assert_eq!(overwritten, vec![0.0; 3]);

            let mut accumulated = vec![0.25f32; 3];
            cpu.matvec_add(&mut accumulated, &[], &weight, 0.375)
                .unwrap();
            assert_eq!(accumulated, vec![0.25; 3]);
        }
    }

    /// An unsupported format is REFUSED with a message naming the device and the format.
    /// Falling back silently would make a model's numerics depend on where it ran.
    #[test]
    fn an_undecodable_format_is_refused_by_name() {
        let cpu = Cpu::new();
        let bytes = vec![0u8; 256];
        // 23 = IQ4_XS, deliberately not implemented.
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(23),
            rows: 1,
            cols: 256,
        };
        assert!(!cpu.supports(w.kind));
        let mut y = [0.0f32];
        let e = cpu.matvec(&mut y, &vec![0.0; 256], &w).unwrap_err();
        let msg = e.to_string();
        assert!(msg.contains("cpu"), "must name the device: {msg}");
        assert!(msg.contains("IQ4_XS"), "must name the format: {msg}");
    }

    #[test]
    fn a_shape_mismatch_says_what_it_expected() {
        let cpu = Cpu::new();
        let bytes = q8_row(1.0, &[0i8; 32]);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 1,
            cols: 32,
        };
        let mut y = [0.0f32];
        let e = cpu.matvec(&mut y, &[0.0; 8], &w).unwrap_err();
        assert!(e.to_string().contains("1x32"), "got: {e}");
    }

    /// Gate M1.3 in miniature: the same inputs must give bit-identical outputs. An
    /// engine that is not deterministic cannot be bisected.
    #[test]
    fn results_are_bit_identical_across_runs() {
        let cpu = Cpu::new();
        let vals: Vec<i8> = (0..256).map(|i| ((i * 7) % 61) as i8 - 30).collect();
        let bytes = q8_row(0.125, &vals);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 8,
            cols: 32,
        };
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.3).sin()).collect();
        let mut a = vec![0.0f32; 8];
        let mut b = vec![0.0f32; 8];
        cpu.matvec(&mut a, &x, &w).unwrap();
        cpu.matvec(&mut b, &x, &w).unwrap();
        for (p, q) in a.iter().zip(&b) {
            assert_eq!(p.to_bits(), q.to_bits(), "not deterministic");
        }
    }

    #[test]
    fn worker_count_is_pure_and_respects_the_granularity_threshold() {
        // 8x32 = 256: below 1<<16, stays sequential regardless of `available`.
        assert_eq!(Cpu::worker_count(8, 32, 32), 1);
        // 1024x128 = 131,072: above threshold; available <= rows and <= the cap, so
        // returns `available`.
        assert_eq!(Cpu::worker_count(1024, 128, 8), 8);
        // 3x30_000 = 90,000: above threshold, but rows < available -- clamp to rows.
        assert_eq!(Cpu::worker_count(3, 30_000, 8), 3);
        // available=0 (e.g. a failed probe) must not panic and must not return 0.
        assert_eq!(Cpu::worker_count(1024, 128, 0), 1);
        // available=32 exceeds MAX_WORKER_THREADS: measured to regress the router-scale
        // shape (0.54ms at 32 threads vs 0.24ms at 1) while a large shape plateaus well
        // before 32 -- capped at 8 regardless of how many CPUs the host reports.
        assert_eq!(Cpu::worker_count(1024, 128, 32), 8);
        assert_eq!(Cpu::worker_count(151_936, 2048, 32), 8);
    }

    #[test]
    fn dot_rows_partitions_every_row_exactly_once_for_any_thread_count() {
        let (rows, cols) = (37usize, 5usize);
        let tmp: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.013).sin()).collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.7).cos()).collect();

        let mut sequential = vec![0.0f32; rows];
        Cpu::dot_rows(&mut sequential, &x, &tmp, cols, 1.0, false, 1);

        for threads in [2usize, 3, 4, 5, 6, 7, 9, 13, 20, 37, 50, 1000] {
            let mut y = vec![0.0f32; rows];
            Cpu::dot_rows(&mut y, &x, &tmp, cols, 1.0, false, threads);
            for (r, (a, b)) in sequential.iter().zip(&y).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {r} differs at {threads} threads"
                );
            }
        }
    }

    #[test]
    fn matvec_stays_effectively_sequential_below_the_granularity_threshold() {
        let cpu = Cpu::new();
        let vals: Vec<i8> = (0..256).map(|i| ((i * 7) % 61) as i8 - 30).collect();
        let bytes = q8_row(0.125, &vals);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 8,
            cols: 32,
        };
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.3).sin()).collect();

        let mut expect = [0.0f32; 8];
        cpu.matvec(&mut expect, &x, &w).unwrap();

        for threads in [1usize, 8] {
            let mut y = [0.0f32; 8];
            cpu.matvec_with_threads(&mut y, &x, &w, threads).unwrap();
            for (a, b) in expect.iter().zip(&y) {
                assert_eq!(a.to_bits(), b.to_bits(), "threads={threads}");
            }
        }
    }

    #[test]
    fn row_parallel_matvec_is_bit_identical_to_sequential_above_the_granularity_threshold() {
        let cpu = Cpu::new();
        let vals: Vec<i8> = (0..1024 * 128).map(|i| ((i * 7) % 61) as i8 - 30).collect();
        let bytes = q8_row(0.03125, &vals);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 1024,
            cols: 128,
        };
        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.3).sin()).collect();

        let mut sequential = vec![0.0f32; 1024];
        cpu.matvec_with_threads(&mut sequential, &x, &w, 1).unwrap();

        for threads in [2usize, 3, 4, 5, 7, 8, 13, 16, 1024, 4096] {
            let mut y = vec![0.0f32; 1024];
            cpu.matvec_with_threads(&mut y, &x, &w, threads).unwrap();
            for (r, (a, b)) in sequential.iter().zip(&y).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {r} differs at {threads} threads"
                );
            }
        }
    }

    #[test]
    fn fused_q8_matvec_is_bit_identical_to_dequantize_then_dot() {
        let cpu = Cpu::new();
        let (rows, cols) = (37usize, 96usize);
        let mut bytes = Vec::new();
        for row in 0..rows {
            for block in 0..cols / 32 {
                let scale = 0.0078125 * (1 + ((row + block) % 7)) as f32;
                bytes.extend_from_slice(&f16_bits(scale).to_le_bytes());
                for i in 0..32 {
                    bytes.push((((row * 19 + block * 11 + i * 7) % 241) as i16 - 120) as i8 as u8);
                }
            }
        }
        let weight = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows,
            cols,
        };
        let x: Vec<f32> = (0..cols)
            .map(|i| ((i as f32) * 0.137).sin() * ((i % 5 + 1) as f32))
            .collect();

        let mut dequantized = vec![0.0f32; rows * cols];
        cpu.dequantize(&mut dequantized, &weight).unwrap();
        let mut expected = vec![0.0f32; rows];
        Cpu::dot_rows(&mut expected, &x, &dequantized, cols, 1.0, false, 1);

        for threads in [1usize, 2, 3, 4, 8, 37, 1000] {
            let mut actual = vec![0.0f32; rows];
            cpu.matvec_with_threads(&mut actual, &x, &weight, threads)
                .unwrap();
            for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "row {row} differs at {threads} threads"
                );
            }

            let mut actual_add = vec![3.25f32; rows];
            cpu.matvec_add_with_threads(&mut actual_add, &x, &weight, 0.375, threads)
                .unwrap();
            for (row, (actual, expected)) in actual_add.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    (3.25 + expected * 0.375).to_bits(),
                    "add row {row} differs at {threads} threads"
                );
            }
        }
    }

    #[test]
    fn fused_q8_refuses_truncated_storage_without_touching_output() {
        let cpu = Cpu::new();
        let bytes = q8_row(0.5, &[1i8; 32]);
        let weight = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 2,
            cols: 32,
        };
        let mut output = [7.0f32, -3.0];
        let error = cpu
            .matvec(&mut output, &[1.0; 32], &weight)
            .expect_err("one encoded row cannot satisfy a two-row weight");
        assert!(error.to_string().contains("needs 68 bytes"), "got: {error}");
        assert_eq!(output, [7.0, -3.0]);
    }

    #[test]
    fn row_parallel_matvec_add_is_bit_identical_to_sequential_above_the_granularity_threshold() {
        let cpu = Cpu::new();
        let vals: Vec<i8> = (0..1024 * 128)
            .map(|i| ((i * 11) % 61) as i8 - 30)
            .collect();
        let bytes = q8_row(0.03125, &vals);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 1024,
            cols: 128,
        };
        let x: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.3).sin()).collect();
        let seed: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.11).cos()).collect();

        let mut sequential = seed.clone();
        cpu.matvec_add_with_threads(&mut sequential, &x, &w, 0.7, 1)
            .unwrap();

        for threads in [2usize, 3, 4, 5, 7, 8, 13, 16, 1024, 4096] {
            let mut y = seed.clone();
            cpu.matvec_add_with_threads(&mut y, &x, &w, 0.7, threads)
                .unwrap();
            for (r, (a, b)) in sequential.iter().zip(&y).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {r} differs at {threads} threads"
                );
            }
        }
    }

    #[test]
    fn row_parallel_matvec_handles_uneven_row_counts_without_dropping_or_duplicating_rows() {
        let cpu = Cpu::new();
        let vals: Vec<i8> = (0..7 * 32).map(|i| ((i * 7) % 61) as i8 - 30).collect();
        let bytes = q8_row(0.0625, &vals);
        let w = Weight {
            bytes: &bytes,
            kind: GgmlType(8),
            rows: 7,
            cols: 32,
        };
        let x: Vec<f32> = (0..32).map(|i| ((i as f32) * 0.3).sin()).collect();

        let mut sequential = [0.0f32; 7];
        cpu.matvec_with_threads(&mut sequential, &x, &w, 1).unwrap();

        for threads in [2usize, 3, 4, 5, 6, 7, 8, 100] {
            let mut y = [0.0f32; 7];
            cpu.matvec_with_threads(&mut y, &x, &w, threads).unwrap();
            for (r, (a, b)) in sequential.iter().zip(&y).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "row {r} differs at {threads} threads"
                );
            }
        }
    }

    #[test]
    fn supports_reports_the_formats_we_actually_decode() {
        let cpu = Cpu::new();
        for code in [0u32, 1, 8, 12, 13, 14, 30] {
            assert!(cpu.supports(GgmlType(code)), "should support {code}");
        }
        for code in [23u32, 16, 9999] {
            assert!(!cpu.supports(GgmlType(code)), "should refuse {code}");
        }
    }

    #[test]
    fn rope_layout_is_explicitly_routed() {
        let cpu = Cpu::new();
        let (cos, sin) = ([0.0f32, 0.0], [1.0f32, 1.0]);

        let mut interleaved = [1.0f32, 2.0, 3.0, 4.0];
        cpu.rope(&mut interleaved, &cos, &sin, RopeLayout::Interleaved, false)
            .unwrap();
        assert_eq!(interleaved, [-2.0, 1.0, -4.0, 3.0]);

        let mut split_half = [1.0f32, 2.0, 3.0, 4.0];
        cpu.rope(&mut split_half, &cos, &sin, RopeLayout::SplitHalf, false)
            .unwrap();
        assert_eq!(split_half, [-3.0, -4.0, 1.0, 2.0]);

        cpu.rope(&mut split_half, &cos, &sin, RopeLayout::SplitHalf, true)
            .unwrap();
        assert_eq!(split_half, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn causal_attention_matches_a_known_softmax_result() {
        let cpu = Cpu::new();
        let q = [1.0f32, 0.0, 1.0, 0.0];
        let k = [0.0f32, 0.0, 3.0f32.ln(), 0.0];
        let v = [2.0f32, 4.0, 6.0, 8.0];
        let mut out = [0.0f32; 4];

        let config = AttentionConfig {
            n_positions: 2,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 2,
            scale: 1.0,
        };
        cpu.causal_attention(&mut out, &q, &k, &v, config).unwrap();

        // Position zero has one visible value. At position one the scores are
        // [0, ln(3)], hence probabilities [1/4, 3/4].
        let expected = [2.0f32, 4.0, 5.0, 7.0];
        for (actual, expected) in out.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
        }
    }

    #[test]
    fn causal_attention_masks_the_future_and_maps_gqa_groups_contiguously() {
        let cpu = Cpu::new();
        // Four query heads in two contiguous groups. Zero keys make every visible
        // position equiprobable, isolating the causal mask and head mapping.
        let q = [0.0f32; 8];
        let k = [0.0f32; 4];
        // [position][kv_head]: future values are deliberately very different.
        let v = [1.0f32, 10.0, 3.0, 30.0];
        let mut out = [f32::NAN; 8];

        let config = AttentionConfig {
            n_positions: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 1,
            scale: 1.0,
        };
        cpu.causal_attention(&mut out, &q, &k, &v, config).unwrap();

        assert_eq!(
            out,
            [
                1.0, 1.0, 10.0, 10.0, // t=0: the future is excluded
                2.0, 2.0, 20.0, 20.0, // t=1: mean of both visible positions
            ]
        );
    }

    #[test]
    fn causal_attention_refuses_bad_shapes_before_touching_output() {
        let cpu = Cpu::new();
        let mut out = [123.0f32; 4];

        let mut config = AttentionConfig {
            n_positions: 0,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            scale: 1.0,
        };
        let error = cpu
            .causal_attention(&mut out, &[], &[], &[], config)
            .unwrap_err();
        assert!(error.to_string().contains("at least one position"));
        assert_eq!(out, [123.0; 4]);

        config = AttentionConfig {
            n_positions: 1,
            n_heads: 4,
            n_kv_heads: 3,
            head_dim: 1,
            scale: 1.0,
        };
        let error = cpu
            .causal_attention(&mut out, &[0.0; 4], &[0.0; 4], &[0.0; 4], config)
            .unwrap_err();
        assert!(error.to_string().contains("divisible"));
        assert_eq!(out, [123.0; 4]);

        config.n_kv_heads = 2;
        let error = cpu
            .causal_attention(&mut out, &[0.0; 3], &[0.0; 2], &[0.0; 2], config)
            .unwrap_err();
        assert!(error.to_string().contains("expected out=4 q=4 k=2 v=2"));
        assert_eq!(out, [123.0; 4]);

        config = AttentionConfig {
            n_positions: 4,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 1,
            scale: f32::NAN,
        };
        let error = cpu
            .causal_attention(&mut out, &[0.0; 4], &[0.0; 4], &[0.0; 4], config)
            .unwrap_err();
        assert!(error.to_string().contains("positive and finite"));
        assert_eq!(out, [123.0; 4]);
    }

    #[test]
    fn incremental_attention_matches_the_corresponding_full_sequence_chunk() {
        let cpu = Cpu::new();
        let q: Vec<f32> = (0..24)
            .map(|index| ((index as f32 + 1.0) * 0.37).sin())
            .collect();
        let k: Vec<f32> = (0..12)
            .map(|index| ((index as f32 + 2.0) * 0.23).cos())
            .collect();
        let v: Vec<f32> = (0..12).map(|index| index as f32 * 0.5 - 2.0).collect();
        let full_config = AttentionConfig {
            n_positions: 3,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 2,
            scale: 0.5,
        };
        let mut full = vec![0.0f32; q.len()];
        cpu.causal_attention(&mut full, &q, &k, &v, full_config)
            .unwrap();

        let first_new_element = full_config.n_heads * full_config.head_dim;
        let mut incremental = vec![0.0f32; full.len() - first_new_element];
        cpu.incremental_causal_attention(
            &mut incremental,
            &q[first_new_element..],
            &k,
            &v,
            IncrementalAttentionConfig {
                n_query_positions: 2,
                query_start: 1,
                n_kv_positions: 3,
                n_heads: full_config.n_heads,
                n_kv_heads: full_config.n_kv_heads,
                head_dim: full_config.head_dim,
                scale: full_config.scale,
            },
        )
        .unwrap();

        for (actual, expected) in incremental.iter().zip(&full[first_new_element..]) {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "incremental attention must use identical arithmetic"
            );
        }
    }

    #[test]
    fn incremental_attention_uses_absolute_position_and_masks_later_cache_entries() {
        let cpu = Cpu::new();
        // The query is at absolute position one. The third cache entry is present but
        // still in its future and must not affect the result. Visible scores are
        // [0, ln(3)], hence probabilities [1/4, 3/4].
        let q = [1.0f32];
        let k = [0.0f32, 3.0f32.ln(), 100.0];
        let v = [2.0f32, 6.0, 10_000.0];
        let mut out = [f32::NAN];
        cpu.incremental_causal_attention(
            &mut out,
            &q,
            &k,
            &v,
            IncrementalAttentionConfig {
                n_query_positions: 1,
                query_start: 1,
                n_kv_positions: 3,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: 1,
                scale: 1.0,
            },
        )
        .unwrap();
        assert!((out[0] - 5.0).abs() < 1e-6, "got {}", out[0]);
    }

    #[test]
    fn incremental_attention_refuses_invalid_ranges_and_shapes_without_touching_output() {
        let cpu = Cpu::new();
        let mut out = [123.0f32; 2];
        let base = IncrementalAttentionConfig {
            n_query_positions: 1,
            query_start: 2,
            n_kv_positions: 3,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 1,
            scale: 1.0,
        };

        let error = cpu
            .incremental_causal_attention(
                &mut out,
                &[0.0; 2],
                &[0.0; 3],
                &[0.0; 3],
                IncrementalAttentionConfig {
                    query_start: usize::MAX,
                    ..base
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("query range overflows"));
        assert_eq!(out, [123.0; 2]);

        let error = cpu
            .incremental_causal_attention(
                &mut out,
                &[0.0; 2],
                &[0.0; 3],
                &[0.0; 3],
                IncrementalAttentionConfig {
                    query_start: 3,
                    ..base
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("exceeds 3 cached positions"));
        assert_eq!(out, [123.0; 2]);

        let error = cpu
            .incremental_causal_attention(&mut out, &[0.0], &[0.0; 3], &[0.0; 3], base)
            .unwrap_err();
        assert!(error
            .to_string()
            .contains("expected out=2 q=2 k_cache=3 v_cache=3"));
        assert_eq!(out, [123.0; 2]);

        let error = cpu
            .incremental_causal_attention(
                &mut out,
                &[0.0; 2],
                &[0.0; 3],
                &[0.0; 3],
                IncrementalAttentionConfig {
                    scale: f32::NAN,
                    ..base
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("positive and finite"));
        assert_eq!(out, [123.0; 2]);

        let error = cpu
            .incremental_causal_attention(
                &mut out,
                &[0.0; 2],
                &[0.0; 3],
                &[0.0; 3],
                IncrementalAttentionConfig {
                    n_query_positions: 0,
                    ..base
                },
            )
            .unwrap_err();
        assert!(error.to_string().contains("at least one query position"));
        assert_eq!(out, [123.0; 2]);
    }

    #[test]
    fn dequantize_decodes_the_complete_weight_and_requires_exact_output_shape() {
        let cpu = Cpu::new();
        let bytes: Vec<u8> = [1.0f32, -2.5, 3.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let weight = Weight {
            bytes: &bytes,
            kind: GgmlType(0),
            rows: 1,
            cols: 3,
        };

        let mut out = [0.0f32; 3];
        cpu.dequantize(&mut out, &weight).unwrap();
        assert_eq!(out, [1.0, -2.5, 3.25]);

        let mut wrong = [9.0f32; 2];
        let error = cpu.dequantize(&mut wrong, &weight).unwrap_err();
        assert!(error.to_string().contains("expected 3 floats, got 2"));
        assert_eq!(wrong, [9.0; 2], "a refused decode must not touch output");
    }

    #[test]
    fn dequantize_supports_f16_and_refuses_incomplete_storage() {
        let cpu = Cpu::new();
        let bits = [f16_bits(1.0), f16_bits(-2.0)];
        let bytes: Vec<u8> = bits.into_iter().flat_map(u16::to_le_bytes).collect();
        let weight = Weight {
            bytes: &bytes,
            kind: GgmlType(1),
            rows: 1,
            cols: 2,
        };
        let mut out = [0.0f32; 2];
        cpu.dequantize(&mut out, &weight).unwrap();
        assert_eq!(out, [1.0, -2.0]);

        let incomplete = Weight {
            bytes: &bytes[..3],
            ..weight
        };
        let error = cpu.dequantize(&mut out, &incomplete).unwrap_err();
        assert!(error.to_string().contains("needs 4 bytes, got 3"));
    }

    #[test]
    fn shape_sensitive_small_ops_return_errors_instead_of_panicking() {
        let cpu = Cpu::new();

        let mut rms_out = [7.0f32; 2];
        let error = cpu.rmsnorm(&mut rms_out, &[1.0], None, 1e-6).unwrap_err();
        assert!(error.to_string().contains("RMSNorm"));
        assert_eq!(rms_out, [7.0; 2]);

        let mut gate = [3.0f32; 2];
        let error = cpu.swiglu(&mut gate, &[1.0], 0.0).unwrap_err();
        assert!(error.to_string().contains("SwiGLU"));
        assert_eq!(gate, [3.0; 2]);

        let mut residual_out = [3.0f32; 2];
        let error = cpu.add_in_place(&mut residual_out, &[1.0]).unwrap_err();
        assert!(error.to_string().contains("residual add"));
        assert_eq!(residual_out, [3.0; 2]);

        let error = cpu.softmax(&mut []).unwrap_err();
        assert!(error.to_string().contains("softmax input cannot be empty"));
        let mut nan_logits = [f32::NAN, 0.0];
        let error = cpu.softmax(&mut nan_logits).unwrap_err();
        assert!(error.to_string().contains("cannot contain NaN"));
        assert!(nan_logits[0].is_nan());
        assert_eq!(nan_logits[1], 0.0);

        let mut coordinates = [1.0f32; 4];
        let error = cpu
            .rope(
                &mut coordinates,
                &[1.0, 1.0],
                &[0.0],
                RopeLayout::SplitHalf,
                false,
            )
            .unwrap_err();
        assert!(error.to_string().contains("RoPE"));
        assert_eq!(coordinates, [1.0; 4]);
    }
}
