//! bf16 matvec: the embedding and the output head.
//!
//! Not a quantised format and not a small matmul. An lm_head is `[vocab][dim]` — for
//! DeepSeek-V4 that is 129,280 x 4096, i.e. **529 M multiply-accumulates and 1.06 GB of
//! weight reads for a single token**, larger than any individual expert. It is the one
//! place in a decode step where a single parallel region has enough work to amortise
//! anything, which is why it is also the control case in the threading measurements: it
//! ran unchanged at 0.020 s whether given 16 threads or 32, while everything built from
//! small regions got up to 24.8x worse.
//!
//! Weights stay bf16 in memory. Widening to f32 at load would double the bytes this
//! reads, and for a bandwidth-bound matvec bytes are the cost.

use beholdd_core::bf16_to_f32;

/// Scalar dot product between one BF16 row and an `f32` activation.
pub fn dot_scalar(x: &[f32], w: &[u16]) -> f32 {
    assert_eq!(w.len(), x.len());
    w.iter().zip(x).fold(0.0f32, |acc, (&weight, &value)| {
        acc + bf16_to_f32(weight) * value
    })
}

/// Scalar dot product against one little-endian, byte-addressed BF16 row.
///
/// GGUF exposes encoded tensor bytes, and a tensor slice is not required to carry a
/// Rust `u16` alignment guarantee. Decode the scalar reference with `from_le_bytes` so
/// an accelerated caller can safely accept an arbitrarily aligned checkpoint range.
pub fn dot_bytes_scalar(x: &[f32], w: &[u8]) -> f32 {
    assert_eq!(w.len(), x.len() * 2);
    w.chunks_exact(2)
        .zip(x)
        .fold(0.0f32, |acc, (weight, &value)| {
            let bits = u16::from_le_bytes([weight[0], weight[1]]);
            acc + bf16_to_f32(bits) * value
        })
}

/// Dot product between one BF16 row and an `f32` activation, using AVX-512 when present.
pub fn dot(x: &[f32], w: &[u16]) -> f32 {
    assert_eq!(w.len(), x.len());
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: feature and slice lengths checked; the implementation uses
            // unaligned loads and never reads beyond either slice.
            unsafe { return avx512::dot(x, w) }
        }
    }
    dot_scalar(x, w)
}

/// Dot product against one little-endian, byte-addressed BF16 row.
///
/// This is the GGUF-facing kernel. It deliberately accepts bytes instead of casting the
/// checkpoint range to `u16`: the latter would be undefined for an unaligned tensor
/// slice. On little-endian x86 the AVX-512 path widens directly from unaligned packed
/// bytes; every other target uses the portable scalar definition.
pub fn dot_bytes(x: &[f32], w: &[u8]) -> f32 {
    assert_eq!(w.len(), x.len() * 2);
    #[cfg(all(target_arch = "x86_64", target_endian = "little"))]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: feature and byte length checked. `dot_bytes` performs unaligned
            // loads, and every 16-bit pattern is a valid BF16 payload.
            unsafe { return avx512::dot_bytes(x, w) }
        }
    }
    dot_bytes_scalar(x, w)
}

/// The scalar definition. Overwrites `y`.
pub fn matvec_scalar(y: &mut [f32], x: &[f32], w: &[u16], rows: usize, cols: usize) {
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    assert_eq!(w.len(), rows * cols);
    for (r, out) in y.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        *out = dot_scalar(x, row);
    }
}

/// AVX-512 where available, scalar otherwise.
pub fn matvec(y: &mut [f32], x: &[f32], w: &[u16], rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: feature checked; lengths asserted inside.
            unsafe { return avx512::matvec(y, x, w, rows, cols) };
        }
    }
    matvec_scalar(y, x, w, rows, cols);
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    //! bf16 -> f32 is a widen and a 16-bit shift, so the inner loop is three
    //! instructions. Four independent accumulators, because a single FMA chain is
    //! latency-bound rather than throughput-bound at this width.
    //!
    //! f32 accumulation matches the model operator's existing contract. Four independent
    //! 16-lane chains also form a shallower reduction than one serial f32 chain; this is
    //! a throughput choice, not a claim that f32 is more accurate than f64.
    use std::arch::x86_64::*;

    #[inline]
    unsafe fn dot_ptr(x: &[f32], row: *const u16) -> f32 {
        unsafe {
            let cols = x.len();
            let mut a0 = _mm512_setzero_ps();
            let mut a1 = _mm512_setzero_ps();
            let mut a2 = _mm512_setzero_ps();
            let mut a3 = _mm512_setzero_ps();
            let mut i = 0usize;
            while i + 64 <= cols {
                for (k, acc) in [&mut a0, &mut a1, &mut a2, &mut a3].into_iter().enumerate() {
                    let off = i + k * 16;
                    let r16 = _mm256_loadu_si256(row.add(off) as *const __m256i);
                    let w32 = _mm512_slli_epi32(_mm512_cvtepu16_epi32(r16), 16);
                    *acc = _mm512_fmadd_ps(
                        _mm512_castsi512_ps(w32),
                        _mm512_loadu_ps(x.as_ptr().add(off)),
                        *acc,
                    );
                }
                i += 64;
            }
            let mut acc =
                _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)));
            while i < cols {
                // `row` may originate from a byte-addressed GGUF range, so the scalar
                // tail must preserve the vector path's unaligned-load guarantee.
                let bits = row.add(i).read_unaligned();
                acc += f32::from_bits((bits as u32) << 16) * *x.get_unchecked(i);
                i += 1;
            }
            acc
        }
    }

    /// # Safety
    /// Requires `avx512f`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot(x: &[f32], w: &[u16]) -> f32 {
        assert_eq!(w.len(), x.len());
        unsafe { dot_ptr(x, w.as_ptr()) }
    }

    /// # Safety
    /// Requires `avx512f` on a little-endian target. `w` need not be aligned.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn dot_bytes(x: &[f32], w: &[u8]) -> f32 {
        assert_eq!(w.len(), x.len() * 2);
        unsafe { dot_ptr(x, w.as_ptr().cast::<u16>()) }
    }

    /// # Safety
    /// Requires `avx512f`.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn matvec(y: &mut [f32], x: &[f32], w: &[u16], rows: usize, cols: usize) {
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);
        assert_eq!(w.len(), rows * cols);
        unsafe {
            for (r, out) in y.iter_mut().enumerate() {
                let row = w.as_ptr().add(r * cols);
                *out = dot_ptr(x, row);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bf(v: f32) -> u16 {
        (v.to_bits() >> 16) as u16
    }

    #[test]
    fn scalar_matches_hand_computation() {
        let w = vec![bf(1.0), bf(2.0), bf(-0.5), bf(4.0)];
        let x = vec![3.0f32, 5.0, 8.0, 0.25];
        let mut y = [0.0f32];
        matvec_scalar(&mut y, &x, &w, 1, 4);
        // 3 + 10 - 4 + 1 = 10
        assert_eq!(y[0], 10.0);
        assert_eq!(dot_scalar(&x, &w), 10.0);
    }

    #[test]
    fn byte_addressed_dot_is_alignment_independent() {
        let weights = [bf(1.0), bf(2.0), bf(-0.5), bf(4.0)];
        let x = [3.0f32, 5.0, 8.0, 0.25];
        let mut storage = vec![0xabu8];
        for weight in weights {
            storage.extend_from_slice(&weight.to_le_bytes());
        }
        let unaligned = &storage[1..];
        assert_eq!(dot_bytes_scalar(&x, unaligned), 10.0);
        assert_eq!(dot_bytes(&x, unaligned), 10.0);
    }

    /// Gate M0.2 for bf16. Tolerance 1e-5: eight mantissa bits in the weights, and the
    /// paths differ only in summation order over `cols` terms.
    #[test]
    fn vector_agrees_with_the_definition() {
        for (rows, cols) in [(37usize, 4096usize), (8, 64), (5, 100), (1, 6144)] {
            let w: Vec<u16> = (0..rows * cols)
                .map(|i| bf(((i as f32) * 0.013).sin() * 0.4))
                .collect();
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i as f32) * 0.021).cos() * 0.6)
                .collect();
            let mut a = vec![0.0f32; rows];
            let mut b = vec![0.0f32; rows];
            matvec_scalar(&mut a, &x, &w, rows, cols);
            matvec(&mut b, &x, &w, rows, cols);
            let rms = (a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / rows as f64).sqrt();
            let worst = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (*p as f64 - *q as f64).abs())
                .fold(0.0f64, f64::max);
            assert!(rms > 0.0, "{rows}x{cols}: degenerate");
            assert!(
                worst / rms < 1e-5,
                "{rows}x{cols}: {:.2e} relative",
                worst / rms
            );

            for (row, expected) in b.iter().enumerate() {
                let start = row * cols;
                let end = start + cols;
                let got = dot(&x, &w[start..end]);
                assert_eq!(got.to_bits(), expected.to_bits());
            }
        }
    }
}
