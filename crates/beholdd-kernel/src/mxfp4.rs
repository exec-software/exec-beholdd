//! MXFP4 matvec: `y[r] = sum_i W[r][i] * x[i]`, with W read as packed 4-bit codes.
//!
//! This is the innermost loop of the whole engine. On a streamed MoE it is where most of
//! a token goes -- 47% of a DeepSeek-V4 decode step, and essentially all of a Kimi-K3
//! one -- so it gets a scalar definition, a vectorised path, and a test that they agree.
//!
//! # The layout
//!
//! W is `[rows][in/2]` bytes, two E2M1 codes per byte, **low nibble first**. Every group
//! of `group` consecutive values along `in` shares one E8M0 scale byte, so scales are
//! `[rows][in/group]`.
//!
//! # The scale is applied to the group SUM, not to each element
//!
//! It is tempting to fold `2^(sb-127)` into each decoded float by adding to its exponent
//! field, since that is one integer op instead of a multiply. That is wrong, and wrong in
//! a way that produces infinities rather than a small error: E2M1 codes 0 and 8 are zero,
//! whose exponent field is already 0, and adding to it **fabricates a large value out of
//! nothing**. Roughly 12% of a real checkpoint's weights are zero. Scaling the group's
//! finished sum keeps `0 * scale == 0` and costs one multiply per 32 values rather than
//! one per value.

use beholdd_core::{e8m0_scale, E2M1};

/// The scalar definition. Every other path is diffed against this.
///
/// Deliberately unclever: it exists to be obviously correct, not fast.
pub fn matvec_scalar(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    n_in: usize,
    rows: usize,
    group: usize,
) {
    assert_eq!(x.len(), n_in);
    assert_eq!(y.len(), rows);
    assert!(
        n_in.is_multiple_of(group),
        "in must be a whole number of groups"
    );
    assert_eq!(packed.len(), rows * n_in / 2);
    assert_eq!(scales.len(), rows * n_in / group);

    let gcount = n_in / group;
    for (r, out) in y.iter_mut().enumerate() {
        let prow = &packed[r * n_in / 2..(r + 1) * n_in / 2];
        let srow = &scales[r * gcount..(r + 1) * gcount];
        let mut acc = 0.0f32;
        for (g, &s) in srow.iter().enumerate() {
            if s == 255 {
                continue; // NaN scale: the block contributes nothing
            }
            let mut sub = 0.0f32;
            for i in 0..group {
                let idx = g * group + i;
                let byte = prow[idx / 2];
                let code = if idx.is_multiple_of(2) {
                    byte & 0xF
                } else {
                    byte >> 4
                };
                sub += E2M1[code as usize] * x[idx];
            }
            acc += sub * e8m0_scale(s);
        }
        *out = acc;
    }
}

/// The dispatching entry point: AVX-512 where available, scalar otherwise.
pub fn matvec(
    y: &mut [f32],
    x: &[f32],
    packed: &[u8],
    scales: &[u8],
    n_in: usize,
    rows: usize,
    group: usize,
) {
    #[cfg(target_arch = "x86_64")]
    {
        // Runtime detection, not just `target_feature`: a `portable` build must still
        // take the fast path on a machine that has it.
        if group.is_multiple_of(32) && std::arch::is_x86_feature_detected!("avx512f") {
            // SAFETY: avx512f is present (just checked), group is a multiple of 32, and
            // the slice lengths are asserted inside.
            unsafe { return avx512::matvec(y, x, packed, scales, n_in, rows, group) };
        }
    }
    matvec_scalar(y, x, packed, scales, n_in, rows, group);
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    //! Decode in the register file.
    //!
    //! `vpermps` performs a **16-entry f32 table lookup across an entire zmm in one
    //! instruction**, and E2M1 has exactly 16 values -- so the codebook IS a register and
    //! the decode needs no memory at all.
    //!
    //! That matters because of what it replaces. The C engine this descends from expanded
    //! each group into a `float wf[64]` scratch array with a scalar table walk before its
    //! vector unit did anything, and **83% of the kernel's time was that expansion**;
    //! the dot product was 17%. Measured at [3072 x 3584]: 43.1 -> 179.2 GFLOP/s, 4.2x.
    //!
    //! Accumulation is f32, not f64. Weights here carry **two mantissa bits**; an f64
    //! accumulator protects about nineteen binary orders of magnitude that do not exist
    //! in the inputs, at half the lanes. See the tolerance discussion on the tests.
    use std::arch::x86_64::*;

    use beholdd_core::{e8m0_scale, E2M1};

    /// # Safety
    /// Requires `avx512f`. `group` must be a multiple of 32. Slice lengths must match
    /// the shape, which the caller asserts.
    #[target_feature(enable = "avx512f")]
    pub unsafe fn matvec(
        y: &mut [f32],
        x: &[f32],
        packed: &[u8],
        scales: &[u8],
        n_in: usize,
        rows: usize,
        group: usize,
    ) {
        assert_eq!(x.len(), n_in);
        assert_eq!(y.len(), rows);
        assert_eq!(packed.len(), rows * n_in / 2);
        assert_eq!(scales.len(), rows * n_in / group);
        assert!(group.is_multiple_of(32));

        unsafe {
            let lut = _mm512_loadu_ps(E2M1.as_ptr());
            let mask_lo = _mm512_set1_epi32(0xF);
            // Interleave patterns: the low nibble is the EVEN element, so lane 2j comes
            // from the low-nibble vector and lane 2j+1 from the high-nibble one.
            let inter0 = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let inter1 =
                _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
            let gcount = n_in / group;

            for (r, out) in y.iter_mut().enumerate() {
                let prow = packed.as_ptr().add(r * n_in / 2);
                let srow = scales.as_ptr().add(r * gcount);
                let mut acc = 0.0f32;

                for g in 0..gcount {
                    let s = *srow.add(g);
                    if s == 255 {
                        continue;
                    }
                    let xg = x.as_ptr().add(g * group);
                    let pg = prow.add(g * group / 2);

                    let mut v = _mm512_setzero_ps();
                    let mut i = 0usize;
                    while i < group {
                        // 16 bytes -> 32 nibbles -> 32 floats, entirely in registers.
                        let by = _mm_loadu_si128(pg.add(i / 2) as *const __m128i);
                        let w32 = _mm512_cvtepu8_epi32(by);
                        let flo = _mm512_permutexvar_ps(_mm512_and_si512(w32, mask_lo), lut);
                        let fhi = _mm512_permutexvar_ps(_mm512_srli_epi32(w32, 4), lut);
                        let e0 = _mm512_permutex2var_ps(flo, inter0, fhi);
                        let e1 = _mm512_permutex2var_ps(flo, inter1, fhi);
                        v = _mm512_fmadd_ps(e0, _mm512_loadu_ps(xg.add(i)), v);
                        v = _mm512_fmadd_ps(e1, _mm512_loadu_ps(xg.add(i + 16)), v);
                        i += 32;
                    }
                    // The group scale is applied ONCE, to the finished sum. See the
                    // module docs: folding it into each element by exponent-add turns
                    // zero weights into large values.
                    acc += _mm512_reduce_add_ps(v) * e8m0_scale(s);
                }
                *out = acc;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes: a test that fails must fail the same way twice.
    fn fill(n: usize, seed: u64) -> Vec<u8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u8
            })
            .collect()
    }

    fn setup(rows: usize, n_in: usize, group: usize) -> (Vec<u8>, Vec<u8>, Vec<f32>) {
        let packed = fill(rows * n_in / 2, 0xBEE5);
        // Scales in a plausible exponent band. 255 is excluded here and tested separately.
        let scales: Vec<u8> = fill(rows * n_in / group, 0x5CA1)
            .into_iter()
            .map(|b| 118 + (b % 20))
            .collect();
        let x: Vec<f32> = (0..n_in).map(|i| ((i as f32) * 0.37).sin() * 0.5).collect();
        (packed, scales, x)
    }

    /// A hand-computed case, so the scalar definition is anchored to arithmetic rather
    /// than to itself.
    #[test]
    fn scalar_matches_hand_computation() {
        // one row, 32 values, one group. Byte 0x21 -> low nibble 1 (0.5), high 2 (1.0).
        let mut packed = vec![0u8; 16];
        packed[0] = 0x21;
        let scales = vec![128u8]; // 2^1 = 2
        let mut x = vec![0.0f32; 32];
        x[0] = 3.0;
        x[1] = 5.0;
        let mut y = [0.0f32];
        matvec_scalar(&mut y, &x, &packed, &scales, 32, 1, 32);
        // (0.5*3 + 1.0*5) * 2 = 13
        assert_eq!(y[0], 13.0);
    }

    #[test]
    fn a_nan_scale_contributes_nothing() {
        let mut packed = vec![0x77u8; 16]; // all 6.0
        packed[0] = 0x77;
        let scales = vec![255u8];
        let x = vec![1.0f32; 32];
        let mut y = [0.0f32];
        matvec_scalar(&mut y, &x, &packed, &scales, 32, 1, 32);
        assert_eq!(
            y[0], 0.0,
            "a NaN-scaled block must vanish, not poison the row"
        );
        assert!(!y[0].is_nan());
    }

    /// Zero weights must stay zero at any scale. This is the exponent-add trap: folding
    /// the scale into each element's exponent field turns E2M1 code 0 into a large value.
    #[test]
    fn zero_weights_stay_zero_under_a_large_scale() {
        let packed = vec![0x00u8; 16]; // every code 0
        let scales = vec![200u8]; // 2^73, enormous
        let x = vec![1.0f32; 32];
        let mut y = [0.0f32];
        matvec_scalar(&mut y, &x, &packed, &scales, 32, 1, 32);
        assert_eq!(y[0], 0.0);
        let mut y2 = [0.0f32];
        matvec(&mut y2, &x, &packed, &scales, 32, 1, 32);
        assert_eq!(
            y2[0], 0.0,
            "vector path fabricated a value from zero weights"
        );
    }

    /// Gate M0.2: the vectorised path agrees with the definition within a tolerance
    /// **justified by input precision**, not chosen to make the test pass.
    ///
    /// E2M1 carries two mantissa bits, so a weight is exact to at best one part in four.
    /// The two paths differ only in accumulator width and summation order over 3584
    /// terms. 1e-5 relative is roughly 2^-17 -- fifteen binary orders TIGHTER than the
    /// data's own precision, so it detects a real divergence while tolerating reassociation.
    #[test]
    fn vector_agrees_with_the_definition() {
        let (rows, n_in, group) = (64, 3584, 32);
        let (packed, scales, x) = setup(rows, n_in, group);
        let mut a = vec![0.0f32; rows];
        let mut b = vec![0.0f32; rows];
        matvec_scalar(&mut a, &x, &packed, &scales, n_in, rows, group);
        matvec(&mut b, &x, &packed, &scales, n_in, rows, group);

        let rms = (a.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / rows as f64).sqrt();
        let worst = a
            .iter()
            .zip(&b)
            .map(|(p, q)| (*p as f64 - *q as f64).abs())
            .fold(0.0f64, f64::max);
        assert!(rms > 0.0, "test signal is degenerate");
        assert!(
            worst / rms < 1e-5,
            "vector path diverged: worst {worst:.3e} vs signal rms {rms:.3e} = {:.2e} relative",
            worst / rms
        );
    }

    #[test]
    fn shapes_that_are_not_multiples_of_the_vector_width_still_work() {
        // 17 rows exercises the row loop's tail; group 64 exercises two 32-wide
        // iterations per group.
        for (rows, n_in, group) in [(17, 128, 32), (3, 256, 64), (1, 32, 32)] {
            let (packed, scales, x) = setup(rows, n_in, group);
            let mut a = vec![0.0f32; rows];
            let mut b = vec![0.0f32; rows];
            matvec_scalar(&mut a, &x, &packed, &scales, n_in, rows, group);
            matvec(&mut b, &x, &packed, &scales, n_in, rows, group);
            for (p, q) in a.iter().zip(&b) {
                assert!(
                    (p - q).abs() <= p.abs() * 1e-5 + 1e-6,
                    "{rows}x{n_in}/{group}: {p} vs {q}"
                );
            }
        }
    }
}
