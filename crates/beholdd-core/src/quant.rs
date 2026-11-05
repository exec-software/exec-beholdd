//! Storage formats, and exactly what each one means.
//!
//! Every function here is a *definition*, not an optimisation. The fast paths live in
//! `beholdd-kernel` and are diffed against these. That split is deliberate: a vectorised
//! decoder tested only against itself proves nothing, so the reference has to exist
//! somewhere obvious and stay boring.
//!
//! The failure mode these guard against is not a crash. Every mistake below produces a
//! finite, plausible number and a model that is quietly wrong.

/// How a tensor is stored on disk.
///
/// This is *storage*, not compute precision. A `Bf16` tensor and an `F32` tensor holding
/// the same values compute identically; they differ in bytes moved, which for a
/// bandwidth-bound matvec is the thing that matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE binary32.
    F32,
    /// IEEE binary16.
    F16,
    /// bfloat16: the top 16 bits of an f32. Widening is a pure shift.
    Bf16,
    /// Raw bytes, uninterpreted.
    U8,
    /// Signed 8-bit. Used by checkpoints to carry packed 4-bit payloads.
    I8,
    /// Signed 64-bit. Some routing tables ship this way even when the reference
    /// declares them 32-bit -- read at the wrong width and routing silently goes to
    /// the wrong experts.
    I64,
    /// FP8 E4M3 (`e4m3fn`): 1 sign, 4 exponent (bias 7), 3 mantissa. No infinities.
    F8E4M3,
    /// E8M0: a bare biased exponent used as a block scale. Not a float you can add.
    F8E8M0,
    /// MXFP4 payload: two E2M1 codes per byte, low nibble first.
    Mxfp4,
}

impl DType {
    /// Bytes per stored element, or `None` when elements are sub-byte and the answer
    /// depends on count. `Mxfp4` is the sub-byte case: two values share a byte.
    pub fn size(self) -> Option<usize> {
        Some(match self {
            DType::F32 | DType::I64 => {
                if matches!(self, DType::I64) {
                    8
                } else {
                    4
                }
            }
            DType::F16 | DType::Bf16 => 2,
            DType::U8 | DType::I8 | DType::F8E4M3 | DType::F8E8M0 => 1,
            DType::Mxfp4 => return None,
        })
    }

    /// Parse the dtype strings safetensors uses.
    pub fn from_safetensors(s: &str) -> Option<DType> {
        Some(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::Bf16,
            "U8" => DType::U8,
            "I8" => DType::I8,
            "I64" => DType::I64,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E8M0" => DType::F8E8M0,
            _ => return None,
        })
    }
}

/// Fixed GGML storage geometry as `(elements_per_block, bytes_per_block)`.
///
/// This is format vocabulary, independent of whether a particular backend can decode
/// the type. Device capability checks must stay separate so adding a backend never
/// requires changing model semantics.
pub fn ggml_block_geometry(ggml_type: u32) -> Option<(usize, usize)> {
    Some(match ggml_type {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        40 => (64, 36),   // NVFP4
        41 => (128, 18),  // Q1_0
        42 => (128, 129), // F8_E4M3_B128
        _ => return None,
    })
}

#[cfg(test)]
mod ggml_geometry_tests {
    use super::ggml_block_geometry;

    #[test]
    fn every_current_file_type_has_the_authoritative_storage_geometry() {
        let expected = [
            (0, 1, 4),
            (1, 1, 2),
            (2, 32, 18),
            (3, 32, 20),
            (6, 32, 22),
            (7, 32, 24),
            (8, 32, 34),
            (9, 32, 36),
            (10, 256, 84),
            (11, 256, 110),
            (12, 256, 144),
            (13, 256, 176),
            (14, 256, 210),
            (15, 256, 292),
            (16, 256, 66),
            (17, 256, 74),
            (18, 256, 98),
            (19, 256, 50),
            (20, 32, 18),
            (21, 256, 110),
            (22, 256, 82),
            (23, 256, 136),
            (24, 1, 1),
            (25, 1, 2),
            (26, 1, 4),
            (27, 1, 8),
            (28, 1, 8),
            (29, 256, 56),
            (30, 1, 2),
            (34, 256, 54),
            (35, 256, 66),
            (39, 32, 17),
            (40, 64, 36),
            (41, 128, 18),
            (42, 128, 129),
        ];
        for (kind, elements, bytes) in expected {
            assert_eq!(
                ggml_block_geometry(kind),
                Some((elements, bytes)),
                "type {kind}"
            );
        }
        for removed_or_unknown in [4, 5, 31, 32, 33, 36, 37, 38, 43, u32::MAX] {
            assert_eq!(ggml_block_geometry(removed_or_unknown), None);
        }
    }
}

/// The sixteen values of OCP MX E2M1, indexed by code.
///
/// 1 sign, 2 exponent, 1 mantissa. There is **no inf and no NaN** -- every code is a
/// finite number, which is why a decoder can be a plain lookup with no special cases.
/// Note that code 8 is negative zero: it must decode to `-0.0` and not be normalised
/// away, because a decoder that folds it to `+0.0` is no longer a bijection and any
/// round-trip test built on it silently stops testing anything.
pub const E2M1: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Decode one E2M1 code. Only the low 4 bits are read.
#[inline]
pub fn e2m1_decode(code: u8) -> f32 {
    E2M1[(code & 0xF) as usize]
}

/// Decode an E8M0 block scale: `2^(b - 127)`.
///
/// **255 is NaN by spec and maps to zero here.** A block whose scale is NaN contributes
/// nothing; letting it produce an actual NaN would poison an entire row through the
/// accumulator, and clamping it to 1.0 would fabricate weights out of a sentinel.
#[inline]
pub fn e8m0_scale(b: u8) -> f32 {
    match b {
        255 => 0.0,
        // Code 0 is 2^-127, which is SUBNORMAL in f32 (min normal is 2^-126). Building
        // the exponent field would give literal zero and silently delete the block.
        // 2^-127 = 2^22 * 2^-149, i.e. mantissa bit 22 with a zero exponent field.
        0 => f32::from_bits(0x0040_0000),
        // Everything else is a normal f32 and the exponent field IS the code.
        _ => f32::from_bits((b as u32) << 23),
    }
}

/// Decode FP8 E4M3 (`e4m3fn`): 1 sign, 4 exponent (bias 7), 3 mantissa.
///
/// This is the **definition**, NaN cases included. For the branchless form used in hot
/// loops see [`e4m3_decode_fast`], whose deviation is named there.
///
/// `e4m3fn` has no infinities: `0x7F`/`0xFF` are NaN, everything else is finite, and the
/// largest magnitude is 448.
#[inline]
pub fn e4m3_decode(b: u8) -> f32 {
    if b & 0x7F == 0x7F {
        return f32::NAN;
    }
    e4m3_decode_fast(b)
}

/// Branchless E4M3 decode, for kernels.
///
/// The whole conversion is a bit shift, which is the trick worth keeping: an e4m3 value
/// laid into an f16's sign+exponent+mantissa positions is the same number scaled by 2^8,
/// because f16's bias is 15 and e4m3's is 7. So widen, shift, reinterpret, multiply by
/// 256. No table, no branch, and no subnormal special case -- f16 handles e4m3's
/// subnormals natively because its exponent range strictly contains e4m3's.
///
/// **THE DEVIATION.** e4m3's NaN encoding is its all-ones exponent with an all-ones
/// mantissa. In f16 that same bit pattern is an ordinary exponent, so the trick cannot
/// express it: `0x7F` and `0xFF` come out as ±480 rather than NaN. That is deliberate.
/// A per-element NaN check is real cost in the innermost loop of the whole engine, and a
/// weight tensor containing NaN is a broken checkpoint rather than a case to be fast
/// about. Validation goes through [`e4m3_decode`]; arithmetic goes through this.
#[inline]
pub fn e4m3_decode_fast(b: u8) -> f32 {
    let sign = ((b & 0x80) as u16) << 8;
    let rest = ((b & 0x7F) as u16) << 7;
    f16_to_f32(sign | rest) * 256.0
}

/// bf16 to f32: bf16 **is** the top 16 bits of an f32.
///
/// No rounding, no rebias, no table. This is why a bf16 checkpoint costs nothing to use
/// directly instead of inflating it to f32 first -- and inflating it would double the
/// bytes a bandwidth-bound matvec has to move.
#[inline]
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// IEEE binary16 to f32, including subnormals.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = (h as u32 & 0x8000) << 16;
    let exp = (h as u32 >> 10) & 0x1F;
    let man = h as u32 & 0x3FF;
    let bits = match exp {
        0 if man == 0 => sign,
        // Subnormal: the value is `man * 2^-24`. Renormalise by finding the leading 1
        // at bit `p`, which makes the value (1 + f) * 2^(p-24), so the f32 exponent
        // field is p - 24 + 127 and the mantissa is what is left once that bit is
        // removed, shifted up to bit 23.
        0 => {
            let p = 31 - man.leading_zeros();
            sign | ((p + 103) << 23) | ((man - (1 << p)) << (23 - p))
        }
        // inf / NaN
        0x1F => sign | 0x7F80_0000 | (man << 13),
        _ => sign | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Unpack a run of MXFP4 bytes into floats. **Low nibble is the even element.**
///
/// The nibble order is not a convention we get to choose -- it is what the checkpoint
/// contains, and getting it backwards is the worst kind of bug this format can produce:
/// the model still runs and still emits fluent text, because a transposed pair of
/// adjacent weights is not detectable from the output. It is the first thing to suspect
/// if a model is fluent and wrong.
///
/// `out` receives `packed.len() * 2` values.
pub fn mxfp4_unpack(packed: &[u8], out: &mut [f32]) {
    assert_eq!(
        out.len(),
        packed.len() * 2,
        "out must hold two values per byte"
    );
    for (i, &b) in packed.iter().enumerate() {
        out[2 * i] = e2m1_decode(b);
        out[2 * i + 1] = e2m1_decode(b >> 4);
    }
}

/// Round-nearest-even to the E2M1 grid, for activation quantisation.
///
/// Exact midpoints go to the **even code**: 2.5 -> 2.0 (code 4), 0.75 -> 1.0 (code 2).
/// On a grid this coarse that is not a ulp of pedantry -- taking the first minimum
/// instead sends 0.75 to 0.5, a 33% error on a value the format can represent well.
pub fn e2m1_nearest(v: f32) -> f32 {
    let mut best = 0usize;
    let mut bd = f32::INFINITY;
    for (code, &g) in E2M1.iter().enumerate() {
        let d = (v - g).abs();
        if d < bd || (d == bd && code % 2 == 0) {
            bd = d;
            best = code;
        }
    }
    E2M1[best]
}

/// Round-nearest-even to the E4M3 grid: max 448, min normal 2^-6, subnormals on 2^-9.
pub fn e4m3_nearest(v: f32) -> f32 {
    let a = v.abs();
    if a >= 448.0 {
        return 448.0_f32.copysign(v);
    }
    let step = if a < 0.015625 {
        // 2^-6
        0.001953125 // 2^-9, the uniform subnormal grid
    } else {
        // 2^(floor(log2 a) - 3)
        f32::from_bits(((a.to_bits() >> 23) - 3) << 23)
    };
    let q = (a / step).round_ties_even() * step;
    q.min(448.0).copysign(v)
}

/// The power-of-two block scale a quantiser should use: `2^ceil(log2(amax * max_inv))`.
///
/// By bit surgery rather than `log2` and `ceil`, which is both faster and *more correct*
/// at the boundary: the two disagree exactly on powers of two, where floating-point
/// `log2` can land a hair either side and produce a scale one binade off.
///
/// This matters because checkpoints that set a ue8m0 scale format use rounded scales,
/// and quantising with `amax / max` instead puts every value on a shifted grid -- a
/// quiet accuracy loss with nothing to report it.
#[inline]
pub fn round_scale(amax: f32, max_inv: f32) -> f32 {
    let v = amax * max_inv;
    let bits = v.to_bits();
    let e = ((bits >> 23) & 0xFF) as i32 - 127 + i32::from(bits & 0x7F_FFFF != 0);
    let e = e.clamp(-127, 128);
    f32::from_bits(((e + 127) as u32) << 23)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The E2M1 table, written out independently of the constant under test.
    ///
    /// Deliberately a second transcription: a table tested against itself tests nothing.
    /// These are the sixteen values from the OCP microscaling spec.
    #[test]
    fn e2m1_matches_the_spec_grid() {
        let spec: [(u8, f32); 16] = [
            (0x0, 0.0),
            (0x1, 0.5),
            (0x2, 1.0),
            (0x3, 1.5),
            (0x4, 2.0),
            (0x5, 3.0),
            (0x6, 4.0),
            (0x7, 6.0),
            (0x8, -0.0),
            (0x9, -0.5),
            (0xA, -1.0),
            (0xB, -1.5),
            (0xC, -2.0),
            (0xD, -3.0),
            (0xE, -4.0),
            (0xF, -6.0),
        ];
        for (code, want) in spec {
            let got = e2m1_decode(code);
            assert_eq!(got, want, "code {code:#x}");
            // Sign of zero is load-bearing: code 8 is NEGATIVE zero, and == does not
            // distinguish them.
            assert_eq!(
                got.is_sign_negative(),
                want.is_sign_negative(),
                "code {code:#x} sign of zero"
            );
        }
    }

    #[test]
    fn e2m1_reads_only_the_low_nibble() {
        for high in 0..16u8 {
            assert_eq!(e2m1_decode(high << 4 | 0x5), 3.0);
        }
    }

    #[test]
    fn e8m0_is_a_bare_exponent_and_255_is_zero() {
        assert_eq!(e8m0_scale(127), 1.0);
        assert_eq!(e8m0_scale(128), 2.0);
        assert_eq!(e8m0_scale(126), 0.5);
        assert_eq!(e8m0_scale(137), 1024.0);
        assert_eq!(e8m0_scale(117), 1.0 / 1024.0);
        // 255 is NaN by spec. It must become zero, not NaN and not 1.0.
        assert_eq!(e8m0_scale(255), 0.0);
        assert!(!e8m0_scale(255).is_nan());
        // Every other code equals 2^(b-127), checked against an INDEPENDENT f64
        // computation rather than against our own bit trick.
        //
        // Note the low end: code 0 is 2^-127, which f32 can only hold as a SUBNORMAL.
        // An earlier version of this test asserted "mantissa field is zero" as its
        // definition of a power of two, which is false for subnormals -- the test was
        // wrong in the same direction as the code, which is how a bug survives its
        // own test suite.
        for b in 0..255u8 {
            let want = (2.0_f64).powi(b as i32 - 127) as f32;
            let got = e8m0_scale(b);
            assert_eq!(got, want, "code {b}");
            assert!(
                got > 0.0 && got.is_finite(),
                "code {b} must be usable as a scale"
            );
        }
        assert_eq!(e8m0_scale(0), (2.0_f64).powi(-127) as f32);
        assert!(e8m0_scale(0) > 0.0, "code 0 must not collapse to zero");
    }

    /// E4M3 against values computed from the format definition by hand, not by our own
    /// decoder run backwards.
    #[test]
    fn e4m3_matches_hand_computed_values() {
        // (bits, value): sign | exp(4, bias 7) | mantissa(3)
        let cases: [(u8, f32); 10] = [
            (0x00, 0.0),
            (0x38, 1.0), // exp 7 -> 2^0, mantissa 0 -> 1.0
            (0x3C, 1.5), // 1.100 * 2^0
            (0x40, 2.0), // exp 8 -> 2^1
            (0xC0, -2.0),
            (0x48, 4.0),
            (0x08, 0.015625),    // exp 1 -> 2^-6, the smallest normal
            (0x01, 0.001953125), // subnormal: 2^-9
            (0x7E, 448.0),       // largest finite
            (0xFE, -448.0),
        ];
        for (bits, want) in cases {
            assert_eq!(e4m3_decode(bits), want, "bits {bits:#04x}");
        }
        // 0x7F / 0xFF are NaN in e4m3fn.
        assert!(e4m3_decode(0x7F).is_nan());
        assert!(e4m3_decode(0xFF).is_nan());
    }

    /// The fast path is allowed to differ from the definition in EXACTLY two codes, and
    /// this pins which two. If it ever differs anywhere else, the bit trick is broken.
    #[test]
    fn fast_e4m3_differs_from_the_definition_only_on_nan() {
        for b in 0..=255u8 {
            let def = e4m3_decode(b);
            let fast = e4m3_decode_fast(b);
            if b & 0x7F == 0x7F {
                assert!(def.is_nan(), "code {b:#04x} should be NaN by definition");
                assert_eq!(fast.abs(), 480.0, "documented deviation changed");
            } else {
                assert_eq!(def.to_bits(), fast.to_bits(), "code {b:#04x} diverged");
            }
        }
    }

    /// No finite e4m3 code exceeds 448. If one did, a quantiser clamping to 448 would be
    /// silently lossy on values the format can actually hold.
    #[test]
    fn e4m3_finite_range_tops_out_at_448() {
        let max = (0..=255u8)
            .map(e4m3_decode)
            .filter(|v| v.is_finite())
            .fold(0.0f32, |m, v| m.max(v.abs()));
        assert_eq!(max, 448.0);
    }

    #[test]
    fn bf16_is_the_top_half_of_an_f32() {
        for v in [
            1.0f32,
            -1.0,
            0.5,
            core::f32::consts::PI,
            1e-8,
            1e8,
            0.0,
            -0.0,
        ] {
            let truncated = f32::from_bits(v.to_bits() & 0xFFFF_0000);
            assert_eq!(bf16_to_f32((v.to_bits() >> 16) as u16), truncated, "{v}");
        }
        assert!(bf16_to_f32(0x7FC0).is_nan());
        assert_eq!(bf16_to_f32(0x7F80), f32::INFINITY);
    }

    #[test]
    fn f16_handles_normals_subnormals_and_specials() {
        let cases: [(u16, f32); 8] = [
            (0x0000, 0.0),
            (0x3C00, 1.0),
            (0xBC00, -1.0),
            (0x3555, 0.33325195),
            (0x0400, 6.1035156e-5), // smallest normal, 2^-14
            (0x0001, 5.9604645e-8), // smallest subnormal, 2^-24
            (0x03FF, 6.097555e-5),  // largest subnormal, 1023 * 2^-24
            (0x7BFF, 65504.0),      // largest finite
        ];
        for (bits, want) in cases {
            assert_eq!(f16_to_f32(bits), want, "bits {bits:#06x}");
        }
        assert_eq!(f16_to_f32(0x7C00), f32::INFINITY);
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn mxfp4_low_nibble_is_the_even_element() {
        // 0x21 -> low nibble 1 (0.5), high nibble 2 (1.0)
        let mut out = [0.0f32; 4];
        mxfp4_unpack(&[0x21, 0xF7], &mut out);
        assert_eq!(out, [0.5, 1.0, 6.0, -6.0]);
    }

    #[test]
    fn e2m1_nearest_breaks_ties_to_even() {
        assert_eq!(e2m1_nearest(2.5), 2.0); // between codes 4 and 5 -> even code 4
        assert_eq!(e2m1_nearest(0.75), 1.0); // between codes 1 and 2 -> even code 2
        assert_eq!(e2m1_nearest(1.25), 1.0); // between 2 and 3 -> even code 2
        assert_eq!(e2m1_nearest(5.0), 4.0); // between 6 and 7 -> even code 6
                                            // and ordinary rounding still works
        assert_eq!(e2m1_nearest(0.6), 0.5);
        assert_eq!(e2m1_nearest(100.0), 6.0);
        assert_eq!(e2m1_nearest(-100.0), -6.0);
    }

    #[test]
    fn e4m3_nearest_lands_on_representable_values() {
        for v in [0.7f32, 3.3, 100.0, -17.5, 0.02, 1e-4, 500.0, -500.0] {
            let q = e4m3_nearest(v);
            // Whatever it produced must be a value e4m3 can actually store: re-decoding
            // the nearest code has to return it unchanged.
            let round_tripped = (0..=255u8)
                .map(e4m3_decode)
                .filter(|x| x.is_finite())
                .min_by(|a, b| (a - q).abs().partial_cmp(&(b - q).abs()).unwrap())
                .unwrap();
            assert_eq!(q, round_tripped, "{v} -> {q} is not representable");
        }
    }

    #[test]
    fn round_scale_is_a_power_of_two_at_or_above_the_ratio() {
        for (amax, inv) in [
            (24.0f32, 1.0 / 448.0),
            (0.06, 1.0 / 448.0),
            (6.0, 1.0 / 6.0),
        ] {
            let s = round_scale(amax, inv);
            assert_eq!(s.to_bits() & 0x7F_FFFF, 0, "not a power of two");
            assert!(s >= amax * inv, "scale must not clip the block max");
            assert!(s < amax * inv * 2.0, "scale must be tight");
        }
        // Exactly a power of two must NOT round up a binade -- this is the case where
        // log2-then-ceil goes wrong.
        assert_eq!(round_scale(1.0, 1.0), 1.0);
        assert_eq!(round_scale(4.0, 1.0), 4.0);
    }
}
