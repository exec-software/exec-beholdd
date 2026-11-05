//! ggml block-quantised formats: Q8_0 and Q6_K.
//!
//! These are what the local-inference ecosystem actually ships, and between them they
//! account for **every quantised tensor in Qwen3-Coder-Next** (41.4 GB of Q6_K and
//! 30.4 GB of Q8_0 out of 73.1 GB).
//!
//! Unlike MXFP4 and FP8, these are not simple element formats with a separate scale
//! grid — each is a fixed-size *block* struct with its quants and scales interleaved in
//! a specific byte order. That layout is the whole specification, and there is no way to
//! validate it from the values alone: a wrong bit-shift produces plausible numbers.
//!
//! # How this is verified without an oracle
//!
//! Two structural checks, both against real checkpoints:
//!
//! 1. **Byte accounting.** Each format fixes elements-per-block and bytes-per-block. If
//!    those constants are wrong, `numel / ELEMS * BYTES` will not equal the tensor's
//!    actual byte length — for any tensor, in any file. Getting this to agree across
//!    thousands of tensors of differing shapes is a strong constraint.
//! 2. **Round-trip.** Quantise a known signal, dequantise it, and require the error to
//!    sit inside the format's own grid resolution. A layout error scrambles which scale
//!    applies to which sub-block and blows that bound immediately.

use beholdd_core::f16_to_f32;

/// Elements per K-block. Every `*_K` format uses 256.
pub const QK_K: usize = 256;

/// Q8_0: `f16` scale then 32 `i8` quants. 34 bytes per 32 elements.
pub mod q8_0 {
    use super::*;

    /// Elements per block.
    pub const ELEMS: usize = 32;
    /// Bytes per block: 2 for the scale, 32 for the quants.
    pub const BYTES: usize = 34;

    /// Dequantise `blocks` worth of Q8_0 into `out`.
    pub fn dequant(src: &[u8], out: &mut [f32]) {
        assert_eq!(out.len() % ELEMS, 0, "output must be whole blocks");
        let nb = out.len() / ELEMS;
        assert_eq!(
            src.len(),
            nb * BYTES,
            "input must be {nb} blocks of {BYTES} bytes"
        );
        for b in 0..nb {
            let blk = &src[b * BYTES..(b + 1) * BYTES];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for i in 0..ELEMS {
                out[b * ELEMS + i] = d * (blk[2 + i] as i8) as f32;
            }
        }
    }

    /// Dot one packed Q8_0 row with an `f32` activation without materialising the row.
    ///
    /// The multiply and accumulation order is deliberately identical to calling
    /// [`dequant`] and then walking that output from left to right. This lets a backend
    /// fuse decode with the dot product without changing the CPU reference numerics.
    pub fn dot(src: &[u8], x: &[f32]) -> f32 {
        assert_eq!(x.len() % ELEMS, 0, "input must be whole blocks");
        let nb = x.len() / ELEMS;
        assert_eq!(
            src.len(),
            nb * BYTES,
            "packed row must be {nb} blocks of {BYTES} bytes"
        );

        let mut acc = 0.0f32;
        for (blk, values) in src.chunks_exact(BYTES).zip(x.chunks_exact(ELEMS)) {
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            for (quant, value) in blk[2..].iter().zip(values) {
                let decoded = d * (*quant as i8) as f32;
                acc += decoded * value;
            }
        }
        acc
    }
}

/// Q6_K: six bits per weight, in 256-element super-blocks with 16 sub-block scales.
///
/// ```text
/// ql[128]      low 4 bits of each quant
/// qh[64]       high 2 bits, four quants packed per byte
/// scales[16]   i8 sub-block scales, one per 16 elements
/// d            f16 super-block scale
///              = 210 bytes per 256 elements
/// ```
///
/// The value is `d * scales[sub] * (q - 32)`, where `q` is the 6-bit code assembled from
/// its low nibble and high pair. **The -32 is not cosmetic**: the code is unsigned 0..63
/// and the format centres it, so omitting the bias shifts every weight in the tensor by
/// a constant times its scale — which a model tolerates far better than it should,
/// producing degraded output rather than obvious breakage.
pub mod q6_k {
    use super::*;

    /// Elements per block.
    pub const ELEMS: usize = QK_K;
    /// Bytes per block.
    pub const BYTES: usize = 128 + 64 + 16 + 2;

    /// Dequantise `blocks` worth of Q6_K into `out`.
    ///
    /// Beholdd's output-ordered decoder addresses the GGML Q6_K storage layout
    /// directly. It replaces the earlier transcription of ggml's
    /// `dequantize_row_q6_K`; the checkpoint format remains unchanged.
    ///
    /// Each 128-element half contains four 32-element groups. Groups 0/2 share
    /// low/high nibbles in the first 32 low-plane bytes; groups 1/3 share the next
    /// 32 bytes. One high-plane byte carries the same lane from all four groups,
    /// using successive two-bit fields. Scales follow output order, one per 16 values.
    pub fn dequant(src: &[u8], out: &mut [f32]) {
        assert_eq!(out.len() % ELEMS, 0, "output must be whole blocks");
        let nb = out.len() / ELEMS;
        assert_eq!(
            src.len(),
            nb * BYTES,
            "input must be {nb} blocks of {BYTES} bytes"
        );

        for block in 0..nb {
            let blk = &src[block * BYTES..(block + 1) * BYTES];
            let values = &mut out[block * ELEMS..(block + 1) * ELEMS];
            let d = f16_to_f32(u16::from_le_bytes([blk[208], blk[209]]));
            for half in 0..2 {
                let low = &blk[half * 64..(half + 1) * 64];
                let high = &blk[128 + half * 32..128 + (half + 1) * 32];
                let scales = &blk[192 + half * 8..192 + (half + 1) * 8];
                let output = &mut values[half * 128..(half + 1) * 128];
                decode_group::<0, 0>(&low[..32], high, &scales[..2], d, &mut output[..32]);
                decode_group::<0, 2>(&low[32..], high, &scales[2..4], d, &mut output[32..64]);
                decode_group::<4, 4>(&low[..32], high, &scales[4..6], d, &mut output[64..96]);
                decode_group::<4, 6>(&low[32..], high, &scales[6..], d, &mut output[96..]);
            }
        }
    }

    // Constant plane shifts let the compiler vectorize each contiguous output group.
    #[inline(always)]
    fn decode_group<const LOW_SHIFT: u32, const HIGH_SHIFT: u32>(
        low: &[u8],
        high: &[u8],
        scales: &[u8],
        d: f32,
        output: &mut [f32],
    ) {
        for (scale_group, scale) in scales.iter().enumerate() {
            // Preserve the previous multiplication order, including signed zero.
            let scaled_d = d * f32::from(*scale as i8);
            let start = scale_group * 16;
            let low = &low[start..start + 16];
            let high = &high[start..start + 16];
            let output = &mut output[start..start + 16];
            for ((value, low), high) in output.iter_mut().zip(low).zip(high) {
                let code = ((low >> LOW_SHIFT) & 0x0f) | (((high >> HIGH_SHIFT) & 0x03) << 4);
                *value = scaled_d * (i32::from(code) - 32) as f32;
            }
        }
    }
}

/// The 6-bit packed scale/min pair used by both Q4_K and Q5_K.
///
/// Twelve bytes carry eight 6-bit scales and eight 6-bit mins. The first four of each
/// sit in their own byte's low six bits; the last four are split, taking their low
/// nibble from `q[j+4]` and their top two bits from the high bits of an earlier byte.
///
/// This is the single densest piece of bit-packing in the format, and it is where a
/// transcription error hides best: sub-blocks 0-3 would look perfect while 4-7 came out
/// scaled by an arbitrary factor, which reads as a model that is subtly worse rather
/// than one that is broken.
#[inline]
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        (
            (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
            (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
        )
    }
}

/// Q4_K: four bits per weight, with a per-sub-block scale AND min.
///
/// ```text
/// d       f16 super-block scale          2
/// dmin    f16 super-block min            2
/// scales  6-bit packed scales and mins  12
/// qs      4-bit quants                 128
///                                     = 144 bytes per 256 elements
/// ```
///
/// The value is `d*sc * q - dmin*m`. **This format is asymmetric**, unlike Q6_K: it
/// carries an explicit offset rather than centring codes around a bias. Treating it as
/// symmetric loses the offset entirely, and the result still looks like weights.
pub mod q4_k {
    use super::*;

    /// Elements per block.
    pub const ELEMS: usize = QK_K;
    /// Bytes per block.
    pub const BYTES: usize = 2 + 2 + 12 + 128;

    /// Dequantise `blocks` worth of Q4_K into `out`.
    pub fn dequant(src: &[u8], out: &mut [f32]) {
        assert_eq!(out.len() % ELEMS, 0, "output must be whole blocks");
        let nb = out.len() / ELEMS;
        assert_eq!(
            src.len(),
            nb * BYTES,
            "input must be {nb} blocks of {BYTES} bytes"
        );

        for b in 0..nb {
            let blk = &src[b * BYTES..(b + 1) * BYTES];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let scales = &blk[4..16];
            let qs = &blk[16..144];
            let y = &mut out[b * ELEMS..(b + 1) * ELEMS];

            // Four passes of 64 elements; each pass consumes 32 quant bytes and two
            // scale/min pairs -- low nibbles for the first 32, high nibbles for the next.
            for pass in 0..4 {
                let q = &qs[pass * 32..pass * 32 + 32];
                let (sc1, m1) = scale_min_k4(pass * 2, scales);
                let (sc2, m2) = scale_min_k4(pass * 2 + 1, scales);
                let (d1, off1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, off2) = (d * sc2 as f32, dmin * m2 as f32);
                let base = pass * 64;
                for l in 0..32 {
                    y[base + l] = d1 * (q[l] & 0xF) as f32 - off1;
                    y[base + 32 + l] = d2 * (q[l] >> 4) as f32 - off2;
                }
            }
        }
    }
}

/// Q5_K: Q4_K plus one more bit per weight, carried in a separate plane.
///
/// ```text
/// d       f16 super-block scale          2
/// dmin    f16 super-block min            2
/// scales  6-bit packed scales and mins  12
/// qh      the fifth bit, one plane      32
/// qs      4-bit quants                 128
///                                     = 176 bytes per 256 elements
/// ```
///
/// The fifth bit lives in `qh`, and **which bit of `qh[l]` belongs to which element
/// advances by two per pass** -- pass 0 uses bits 0 and 1, pass 1 bits 2 and 3, and so
/// on. Using the same bit every pass produces a tensor where three quarters of the
/// weights have lost their high bit, halving them.
pub mod q5_k {
    use super::*;

    /// Elements per block.
    pub const ELEMS: usize = QK_K;
    /// Bytes per block.
    pub const BYTES: usize = 2 + 2 + 12 + 32 + 128;

    /// Dequantise `blocks` worth of Q5_K into `out`.
    pub fn dequant(src: &[u8], out: &mut [f32]) {
        assert_eq!(out.len() % ELEMS, 0, "output must be whole blocks");
        let nb = out.len() / ELEMS;
        assert_eq!(
            src.len(),
            nb * BYTES,
            "input must be {nb} blocks of {BYTES} bytes"
        );

        for b in 0..nb {
            let blk = &src[b * BYTES..(b + 1) * BYTES];
            let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
            let scales = &blk[4..16];
            let qh = &blk[16..48];
            let qs = &blk[48..176];
            let y = &mut out[b * ELEMS..(b + 1) * ELEMS];

            for pass in 0..4 {
                let q = &qs[pass * 32..pass * 32 + 32];
                let (sc1, m1) = scale_min_k4(pass * 2, scales);
                let (sc2, m2) = scale_min_k4(pass * 2 + 1, scales);
                let (d1, off1) = (d * sc1 as f32, dmin * m1 as f32);
                let (d2, off2) = (d * sc2 as f32, dmin * m2 as f32);
                // The qh bit pair advances by two each pass.
                let (u1, u2) = (1u8 << (pass * 2), 2u8 << (pass * 2));
                let base = pass * 64;
                for l in 0..32 {
                    let hi1 = if qh[l] & u1 != 0 { 16 } else { 0 };
                    let hi2 = if qh[l] & u2 != 0 { 16 } else { 0 };
                    y[base + l] = d1 * ((q[l] & 0xF) as i32 + hi1) as f32 - off1;
                    y[base + 32 + l] = d2 * ((q[l] >> 4) as i32 + hi2) as f32 - off2;
                }
            }
        }
    }
}

/// Elements and bytes per block for a ggml type code, if we can decode it.
///
/// Returning `None` for an unsupported code is the point: a loader must refuse a tensor
/// it cannot decode at bind time, rather than discovering it mid-forward or, worse,
/// reading it as something else.
pub fn block_geometry(ggml_type: u32) -> Option<(usize, usize)> {
    match ggml_type {
        0 | 1 | 8 | 12 | 13 | 14 | 30 => beholdd_core::ggml_block_geometry(ggml_type),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f16_bits(v: f32) -> u16 {
        // Only used for exact, small values in tests.
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let man = ((bits >> 13) & 0x3FF) as u16;
        sign | ((exp as u16) << 10) | man
    }

    /// Test-only encoder traversing packed planes, independently of the decoder's
    /// output-index address calculations. The inputs remain the numerical oracle.
    fn encode_q6_k(codes: &[u8; QK_K], scales: &[i8; 16], d_bits: u16) -> [u8; q6_k::BYTES] {
        assert!(codes.iter().all(|code| *code < 64));
        let mut packed = [0u8; q6_k::BYTES];
        for half in 0..2 {
            let codes = &codes[half * 128..(half + 1) * 128];
            let low_plane = &mut packed[half * 64..(half + 1) * 64];
            for (byte, (first, second)) in low_plane
                .iter_mut()
                .zip(codes[..64].iter().zip(&codes[64..]))
            {
                *byte = (first & 0x0f) | ((second & 0x0f) << 4);
            }
            let high_plane = &mut packed[128 + half * 32..128 + (half + 1) * 32];
            for group in 0..4 {
                let codes = &codes[group * 32..(group + 1) * 32];
                for (byte, code) in high_plane.iter_mut().zip(codes) {
                    *byte |= (code >> 4) << (2 * group);
                }
            }
        }
        for (byte, scale) in packed[192..208].iter_mut().zip(scales) {
            *byte = *scale as u8;
        }
        packed[208..210].copy_from_slice(&d_bits.to_le_bytes());
        packed
    }

    #[test]
    fn q6_k_decodes_every_code_at_every_position_across_blocks() {
        let scales = [
            -128, -127, -63, -31, -17, -3, -1, 1, 2, 5, 19, 33, 65, 101, 126, 127,
        ];
        let mut packed = Vec::new();
        let mut expected = Vec::new();
        for rotation in 0..64 {
            // Distinguish groups and halves while visiting all 64 codes at each
            // output position. Different block scales also expose block reuse.
            let codes = std::array::from_fn(|index| {
                ((index * 29 + (index / 32) * 7 + (index / 128) * 11 + rotation) % 64) as u8
            });
            let d_bits = [0x3555, 0xbc01, 0x0001, 0x7bff][rotation % 4];
            packed.extend_from_slice(&encode_q6_k(&codes, &scales, d_bits));
            let d = f16_to_f32(d_bits);
            expected.extend(codes.iter().enumerate().map(|(index, code)| {
                d * f32::from(scales[index / 16]) * (i32::from(*code) - 32) as f32
            }));
        }
        let mut actual = vec![0.0; expected.len()];
        q6_k::dequant(&packed, &mut actual);
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "output element {index}"
            );
        }
    }

    #[test]
    fn q6_k_preserves_all_signed_scales_and_signed_zero() {
        let codes = std::array::from_fn(|index| [0, 31, 32, 33, 63][index % 5]);
        for d_bits in [0x3555, 0xb555, 0x0000, 0x8000] {
            let d = f16_to_f32(d_bits);
            for batch in 0..16 {
                let scales = std::array::from_fn(|index| (batch * 16 + index) as u8 as i8);
                let packed = encode_q6_k(&codes, &scales, d_bits);
                let mut actual = [0.0; QK_K];
                q6_k::dequant(&packed, &mut actual);
                for (index, actual) in actual.iter().enumerate() {
                    let expected =
                        d * f32::from(scales[index / 16]) * (i32::from(codes[index]) - 32) as f32;
                    assert_eq!(
                        actual.to_bits(),
                        expected.to_bits(),
                        "d_bits={d_bits:#06x}, scale batch {batch}, element {index}"
                    );
                }
            }
        }
    }

    #[test]
    fn q8_0_dequantises_exactly() {
        let d = 0.5f32;
        let mut src = vec![0u8; q8_0::BYTES];
        src[0..2].copy_from_slice(&f16_bits(d).to_le_bytes());
        for i in 0..32 {
            src[2 + i] = (i as i8 - 16) as u8;
        }
        let mut out = vec![0.0f32; 32];
        q8_0::dequant(&src, &mut out);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(*v, d * (i as i32 - 16) as f32, "element {i}");
        }
    }

    /// The -32 bias, isolated. All-zero codes must dequantise to -32*d*scale, NOT to
    /// zero: an implementation that forgets the bias produces a tensor shifted by a
    /// constant, which degrades a model rather than breaking it visibly.
    #[test]
    fn q6_k_applies_the_centring_bias() {
        let mut src = vec![0u8; q6_k::BYTES];
        src[192..208].fill(1); // every sub-block scale = 1
        src[208..210].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        let mut out = vec![0.0f32; QK_K];
        q6_k::dequant(&src, &mut out);
        assert!(
            out.iter().all(|v| *v == -32.0),
            "all-zero codes must centre to -32, got {}",
            out[0]
        );
    }

    /// The four quants sharing a `qh` byte are 32 elements apart within their half, not
    /// adjacent. This pins the interleave, which is the part a wrong implementation gets
    /// wrong while still producing plausible numbers.
    #[test]
    fn q6_k_interleave_places_quants_where_ggml_does() {
        let mut src = vec![0u8; q6_k::BYTES];
        src[192..208].fill(1);
        src[208..210].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        // Set ONLY element 0's low nibble to 0xF -> code 15, value 15 - 32 = -17.
        src[0] = 0x0F;
        let mut out = vec![0.0f32; QK_K];
        q6_k::dequant(&src, &mut out);
        assert_eq!(out[0], -17.0, "element 0 comes from ql[0] low nibble");
        assert_eq!(out[1], -32.0, "element 1 must be untouched");
        // ql[0]'s HIGH nibble feeds element 64, not element 1.
        let mut src2 = vec![0u8; q6_k::BYTES];
        src2[192..208].fill(1);
        src2[208..210].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        src2[0] = 0xF0;
        let mut out2 = vec![0.0f32; QK_K];
        q6_k::dequant(&src2, &mut out2);
        assert_eq!(out2[64], -17.0, "ql[0] high nibble feeds element 64");
        assert_eq!(out2[0], -32.0, "element 0 must be untouched");
    }

    /// `qh` supplies the top two bits, so a full 6-bit code reaches 63 -> 31 after bias.
    #[test]
    fn q6_k_high_bits_extend_the_range() {
        let mut src = vec![0u8; q6_k::BYTES];
        src[192..208].fill(1);
        src[208..210].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        src[0] = 0x0F; // low nibble of element 0
        src[128] = 0x03; // qh bits 0-1 for element 0
        let mut out = vec![0.0f32; QK_K];
        q6_k::dequant(&src, &mut out);
        assert_eq!(out[0], 31.0, "0xF | (3 << 4) = 63, minus 32 = 31");
    }

    #[test]
    fn sub_block_scales_apply_per_16_elements() {
        let mut src = vec![0u8; q6_k::BYTES];
        // Distinct scales so a mis-mapped sub-block is visible.
        for (i, s) in src[192..208].iter_mut().enumerate() {
            *s = (i + 1) as u8;
        }
        src[208..210].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        let mut out = vec![0.0f32; QK_K];
        q6_k::dequant(&src, &mut out);
        // All codes are 0 -> -32, so each element is -32 * its sub-block scale.
        assert_eq!(out[0], -32.0 * 1.0, "elements 0-15 use scales[0]");
        assert_eq!(out[16], -32.0 * 2.0, "elements 16-31 use scales[1]");
        assert_eq!(out[32], -32.0 * 3.0, "elements 32-47 use scales[2]");
    }

    /// Sub-blocks 0-3 read their 6 bits directly; 4-7 are SPLIT across two bytes. A
    /// transcription error here leaves 0-3 perfect and 4-7 scaled by an arbitrary
    /// factor, which reads as a slightly worse model rather than a broken one.
    #[test]
    fn packed_6bit_scales_split_correctly_past_sub_block_four() {
        let mut q = [0u8; 12];
        // Direct range: low six bits of q[j] are the scale, of q[j+4] the min.
        q[0] = 37;
        q[4] = 21;
        assert_eq!(scale_min_k4(0, &q), (37, 21));
        // Split range: j=4 takes its low nibble from q[8] and its top two bits from
        // q[0]>>6; the min takes q[8]>>4 and q[4]>>6.
        let mut q2 = [0u8; 12];
        q2[8] = 0x9; // low nibble 9 -> scale low bits
        q2[0] = 0b11 << 6; // top two bits -> 3
        let (sc, _) = scale_min_k4(4, &q2);
        assert_eq!(sc, 9 | (3 << 4), "split scale must combine both bytes");
    }

    #[test]
    fn q4_k_carries_an_explicit_offset_not_a_bias() {
        let mut src = vec![0u8; q4_k::BYTES];
        src[0..2].copy_from_slice(&f16_bits(1.0).to_le_bytes()); // d
        src[2..4].copy_from_slice(&f16_bits(1.0).to_le_bytes()); // dmin
        src[4..16].fill(0);
        // scale 0 for sub-block 0, min 5 -> every zero code becomes -5, not 0 and not
        // -8 (which a symmetric reading would give).
        src[4] = 2; // sc[0] = 2
        src[8] = 5; // m[0]  = 5
        let mut out = vec![0.0f32; QK_K];
        q4_k::dequant(&src, &mut out);
        assert_eq!(out[0], -5.0, "q4_k is asymmetric: value = d*sc*q - dmin*m");
        // and a nonzero quant scales by sc
        let mut src2 = src.clone();
        src2[16] = 0x03; // element 0 low nibble = 3
        let mut out2 = vec![0.0f32; QK_K];
        q4_k::dequant(&src2, &mut out2);
        assert_eq!(out2[0], 2.0 * 3.0 - 5.0);
    }

    /// The qh bit pair advances by two per 64-element pass. Using the same bit each pass
    /// would strip the high bit from three quarters of the weights -- halving them,
    /// which still looks like a plausible tensor.
    #[test]
    fn q5_k_high_bit_plane_advances_per_pass() {
        let mut src = vec![0u8; q5_k::BYTES];
        src[0..2].copy_from_slice(&f16_bits(1.0).to_le_bytes());
        src[2..4].copy_from_slice(&f16_bits(0.0).to_le_bytes()); // no offset
        src[4..16].fill(0);
        for j in 0..4 {
            src[4 + j] = 1; // every scale = 1
        }
        // Element 0 of pass 0 takes qh bit 0; element 0 of pass 1 (index 64) takes bit 2.
        src[16] = 0b0000_0001; // qh[0] bit 0 set
        let mut out = vec![0.0f32; QK_K];
        q5_k::dequant(&src, &mut out);
        assert_eq!(out[0], 16.0, "pass 0 element 0 must pick up qh bit 0");
        assert_eq!(out[64], 0.0, "pass 1 must NOT read bit 0");

        let mut src2 = src.clone();
        src2[16] = 0b0000_0100; // bit 2
        let mut out2 = vec![0.0f32; QK_K];
        q5_k::dequant(&src2, &mut out2);
        assert_eq!(out2[0], 0.0, "pass 0 must not read bit 2");
        assert_eq!(out2[64], 16.0, "pass 1 element 0 reads qh bit 2");
    }

    /// Every K-format must consume exactly its declared bytes for a whole number of
    /// blocks. A mismatch here is the same failure the byte-accounting check catches on
    /// real files, caught earlier and cheaper.
    #[test]
    fn k_formats_consume_exactly_their_declared_geometry() {
        for (code, elems, bytes) in [
            (12usize, q4_k::ELEMS, q4_k::BYTES),
            (13, q5_k::ELEMS, q5_k::BYTES),
            (14, q6_k::ELEMS, q6_k::BYTES),
        ] {
            assert_eq!(block_geometry(code as u32), Some((elems, bytes)));
            let src = vec![0u8; bytes * 3];
            let mut out = vec![0.0f32; elems * 3];
            match code {
                12 => q4_k::dequant(&src, &mut out),
                13 => q5_k::dequant(&src, &mut out),
                _ => q6_k::dequant(&src, &mut out),
            }
        }
    }

    #[test]
    fn block_geometry_refuses_what_it_cannot_decode() {
        assert_eq!(block_geometry(8), Some((32, 34)));
        assert_eq!(block_geometry(14), Some((256, 210)));
        assert_eq!(block_geometry(0), Some((1, 4)));
        assert_eq!(block_geometry(12), Some((256, 144)));
        assert_eq!(block_geometry(13), Some((256, 176)));
        // IQ4_XS is real and not implemented: it must say so rather than guess.
        assert_eq!(block_geometry(23), None);
        assert_eq!(block_geometry(9999), None);
    }
}
