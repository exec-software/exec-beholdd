//! Block-scaled FP8-E4M3 matvec: `y[r] += sum_i W[r][i] * x[i]`.
//!
//! This is the *trunk* format — attention projections, routers, shared experts — as
//! shipped by DeepSeek-V4 and GLM-5.2. Weights are e4m3 bytes; scales are one value per
//! `BLOCK x BLOCK` tile of the weight matrix, so tile `(rb, cb)` multiplies rows
//! `rb*BLOCK..` and columns `cb*BLOCK..`.
//!
//! Two things differ from [`crate::mxfp4`] and both matter:
//!
//! **The scale grid is 2-D, not per-row-group.** A tile spans many rows, so the scale
//! cannot be folded into a row's accumulator at the end. It is folded into `x` once per
//! column block instead — one multiply per input element per tile-column rather than one
//! per weight, which is a factor of `BLOCK` fewer.
//!
//! **The scale dtype varies by checkpoint.** GLM ships f32 `weight_scale_inv`; DeepSeek
//! ships E8M0, one byte. Callers widen to f32 at bind time: the grids are a few KB and
//! doing it per call would put an `exp` in the hot loop for nothing.

use beholdd_core::e4m3_decode_fast;

/// Tile size of the scale grid. Both checkpoints we target use 128.
pub const BLOCK: usize = 128;

/// The scalar definition. `y` is ACCUMULATED into, not overwritten: an MoE sums experts.
pub fn matvec_scalar(y: &mut [f32], x: &[f32], w: &[u8], scale: &[f32], rows: usize, cols: usize) {
    assert_eq!(x.len(), cols);
    assert_eq!(y.len(), rows);
    assert_eq!(w.len(), rows * cols);
    let nrb = rows.div_ceil(BLOCK);
    let ncb = cols.div_ceil(BLOCK);
    assert_eq!(
        scale.len(),
        nrb * ncb,
        "scale grid must be ceil(rows/128) x ceil(cols/128)"
    );

    for (r, out) in y.iter_mut().enumerate() {
        let row = &w[r * cols..(r + 1) * cols];
        let rb = r / BLOCK;
        let mut acc = 0.0f32;
        for cb in 0..ncb {
            let s = scale[rb * ncb + cb];
            let lo = cb * BLOCK;
            let hi = (lo + BLOCK).min(cols);
            let mut sub = 0.0f32;
            for i in lo..hi {
                sub += e4m3_decode_fast(row[i]) * x[i];
            }
            acc += sub * s;
        }
        *out += acc;
    }
}

/// AVX-512 where available, scalar otherwise.
pub fn matvec(y: &mut [f32], x: &[f32], w: &[u8], scale: &[f32], rows: usize, cols: usize) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
        {
            // SAFETY: features just checked; lengths asserted inside.
            unsafe { return avx512::matvec(y, x, w, scale, rows, cols) };
        }
    }
    matvec_scalar(y, x, w, scale, rows, cols);
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    //! e4m3 -> f32 without a table, by bit surgery.
    //!
    //! An e4m3 byte laid into an f16's sign+exponent+mantissa positions is the same
    //! number scaled by 2^8, because f16's bias is 15 and e4m3's is 7. So:
    //!
    //! ```text
    //! h = ((b & 0x80) << 8) | ((b & 0x7F) << 7)     then f16 -> f32, times 256
    //! ```
    //!
    //! `vcvtph2ps` converts 16 f16 lanes in one instruction, so the whole decode is
    //! shift, or, convert. A 256-entry lookup table would be the obvious alternative and
    //! is worse: gathers cap at roughly scalar throughput on Zen, so the "fast" table is
    //! slower than the arithmetic it replaces.
    use std::arch::x86_64::*;

    use super::BLOCK;

    /// Decode 32 e4m3 bytes into two f32 vectors. The `* 256` is folded into the
    /// caller's scale, not applied here.
    #[inline]
    #[target_feature(enable = "avx512f,avx512bw")]
    unsafe fn decode32(bytes: __m256i) -> (__m512, __m512) {
        // No `unsafe` block needed: these intrinsics are safe to call once the enabling
        // target features are guaranteed, which `#[target_feature]` does.
        let b16 = _mm512_cvtepu8_epi16(bytes);
        let sign = _mm512_slli_epi16(
            _mm512_and_si512(b16, _mm512_set1_epi16(0x0080u16 as i16)),
            8,
        );
        let rest = _mm512_slli_epi16(_mm512_and_si512(b16, _mm512_set1_epi16(0x007F)), 7);
        let h = _mm512_or_si512(sign, rest);
        (
            _mm512_cvtph_ps(_mm512_extracti64x4_epi64(h, 0)),
            _mm512_cvtph_ps(_mm512_extracti64x4_epi64(h, 1)),
        )
    }

    /// # Safety
    /// Requires `avx512f` and `avx512bw`.
    #[target_feature(enable = "avx512f,avx512bw")]
    pub unsafe fn matvec(
        y: &mut [f32],
        x: &[f32],
        w: &[u8],
        scale: &[f32],
        rows: usize,
        cols: usize,
    ) {
        assert_eq!(x.len(), cols);
        assert_eq!(y.len(), rows);
        assert_eq!(w.len(), rows * cols);
        let nrb = rows.div_ceil(BLOCK);
        let ncb = cols.div_ceil(BLOCK);
        assert_eq!(scale.len(), nrb * ncb);

        unsafe {
            for (r, out) in y.iter_mut().enumerate() {
                let row = w.as_ptr().add(r * cols);
                let rb = r / BLOCK;
                let mut acc = 0.0f32;
                for cb in 0..ncb {
                    // The 256 from the e4m3->f16 relationship rides along with the tile
                    // scale, so the inner loop never multiplies by it.
                    let s = *scale.get_unchecked(rb * ncb + cb) * 256.0;
                    let lo = cb * BLOCK;
                    let hi = (lo + BLOCK).min(cols);

                    let mut v0 = _mm512_setzero_ps();
                    let mut v1 = _mm512_setzero_ps();
                    let mut i = lo;
                    while i + 32 <= hi {
                        let (a, b) = decode32(_mm256_loadu_si256(row.add(i) as *const __m256i));
                        v0 = _mm512_fmadd_ps(a, _mm512_loadu_ps(x.as_ptr().add(i)), v0);
                        v1 = _mm512_fmadd_ps(b, _mm512_loadu_ps(x.as_ptr().add(i + 16)), v1);
                        i += 32;
                    }
                    let mut sub = _mm512_reduce_add_ps(_mm512_add_ps(v0, v1));
                    // Tail: the last column block of a matrix whose width is not a
                    // multiple of 32.
                    while i < hi {
                        sub += super::e4m3_decode_fast(*row.add(i)) * *x.get_unchecked(i) / 256.0;
                        i += 1;
                    }
                    acc += sub * s;
                }
                *out += acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                // Avoid 0x7F/0xFF: those are e4m3 NaN, and the fast decode deliberately
                // returns +-480 there. A checkpoint containing them is broken; a test
                // containing them would only measure that deviation.
                let b = (s >> 33) as u8;
                if b & 0x7F == 0x7F {
                    b ^ 0x01
                } else {
                    b
                }
            })
            .collect()
    }

    #[test]
    fn scalar_matches_hand_computation() {
        // 1x128: one tile. 0x38 is 1.0, 0x40 is 2.0.
        let mut w = vec![0u8; 128];
        w[0] = 0x38;
        w[1] = 0x40;
        let scale = vec![3.0f32];
        let mut x = vec![0.0f32; 128];
        x[0] = 5.0;
        x[1] = 7.0;
        let mut y = [0.0f32];
        matvec_scalar(&mut y, &x, &w, &scale, 1, 128);
        // (1*5 + 2*7) * 3 = 57
        assert_eq!(y[0], 57.0);
    }

    #[test]
    fn accumulates_rather_than_overwrites() {
        let w = vec![0x38u8; 128]; // all 1.0
        let scale = vec![1.0f32];
        let x = vec![1.0f32; 128];
        let mut y = [100.0f32];
        matvec_scalar(&mut y, &x, &w, &scale, 1, 128);
        // An MoE sums experts into one buffer; overwriting would drop every expert but
        // the last, and the output would still be fluent.
        assert_eq!(y[0], 100.0 + 128.0);
    }

    #[test]
    fn each_tile_gets_its_own_scale() {
        // 2 rows x 256 cols = 1 row-block x 2 col-blocks. Different scales per column
        // block must apply to different halves of the input.
        let w = vec![0x38u8; 2 * 256]; // all 1.0
        let scale = vec![1.0f32, 10.0];
        let x = vec![1.0f32; 256];
        let mut y = [0.0f32; 2];
        matvec_scalar(&mut y, &x, &w, &scale, 2, 256);
        // 128*1 + 128*10 = 1408
        assert_eq!(y[0], 1408.0);
        assert_eq!(y[1], 1408.0);
    }

    /// Gate M0.2 for this format.
    ///
    /// Tolerance 1e-5 relative: e4m3 carries three mantissa bits, so a weight is exact
    /// to about one part in sixteen. The paths differ only in summation order, and 1e-5
    /// is far tighter than the data's precision while tolerating reassociation.
    #[test]
    fn vector_agrees_with_the_definition() {
        for (rows, cols) in [(64usize, 512usize), (17, 256), (128, 384), (3, 128)] {
            let w = fill(rows * cols, 0xF8F8);
            let nrb = rows.div_ceil(BLOCK);
            let ncb = cols.div_ceil(BLOCK);
            let scale: Vec<f32> = (0..nrb * ncb)
                .map(|i| 0.01 * (1.0 + (i % 7) as f32))
                .collect();
            let x: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.17).cos() * 0.3).collect();
            let mut a = vec![0.0f32; rows];
            let mut b = vec![0.0f32; rows];
            matvec_scalar(&mut a, &x, &w, &scale, rows, cols);
            matvec(&mut b, &x, &w, &scale, rows, cols);

            let rms = (a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / rows as f64).sqrt();
            let worst = a
                .iter()
                .zip(&b)
                .map(|(p, q)| (*p as f64 - *q as f64).abs())
                .fold(0.0f64, f64::max);
            assert!(rms > 0.0, "{rows}x{cols}: degenerate signal");
            assert!(
                worst / rms < 1e-5,
                "{rows}x{cols}: worst {worst:.3e} vs rms {rms:.3e} = {:.2e} relative",
                worst / rms
            );
        }
    }
}
