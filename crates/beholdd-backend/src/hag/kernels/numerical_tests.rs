//! Hardware-free staged arithmetic model, not an emulator or live GPU evidence.
//!
//! The SwiGLU model rounds each arithmetic stage to f32, models SFU denormal
//! flushing, and perturbs normal SFU outputs by -1/0/+1 representable values.
//! RMSNorm uses an ordered F64 reduction and perturbs its F64 reciprocal/rsqrt
//! stages by up to 2^29 binary64 ULP before the final F32 inverse conversion.
//! These catch algorithmic range/clamp regressions independently of the exact-byte ISA tests.
//! It cannot establish hardware rounding, exceptional-value behavior or a global
//! numerical error bound. Qualification of the finite live fixtures is separate.

fn flush_subnormal(value: f32) -> f32 {
    if value.is_subnormal() {
        0.0_f32.copysign(value)
    } else {
        value
    }
}

// Independent finite binary16 value construction for the row shader's F16->F32
// stage. Every finite half and i8 is exact in F32; their product is the single
// rounded operation performed by the shader. This does not emulate GPU hardware.
fn finite_half(bits: u16) -> f32 {
    let exponent = (bits >> 10) & 31;
    let fraction = bits & 1023;
    assert_ne!(exponent, 31);
    let magnitude = if exponent == 0 {
        f64::from(fraction) * 2.0_f64.powi(-24)
    } else {
        f64::from(1024 + fraction) * 2.0_f64.powi(i32::from(exponent) - 25)
    };
    (if bits & 0x8000 != 0 {
        -magnitude
    } else {
        magnitude
    }) as f32
}

fn staged_q8_row(packed: &[u8], columns: usize) -> Vec<f32> {
    assert_eq!(packed.len(), columns / 32 * 34);
    (0..columns)
        .map(|index| {
            let block = index / 32 * 34;
            let scale = finite_half(u16::from_le_bytes([packed[block], packed[block + 1]]));
            let quant = i8::from_le_bytes([packed[block + 2 + index % 32]]);
            scale * f32::from(quant)
        })
        .collect()
}

#[test]
fn q8_row_staged_decode_matches_cpu_for_selected_rows_and_changed_packed_bytes() {
    let scales = [
        0x0000_u16, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x3555, 0x3c00, 0xbc00, 0x7bff,
        0xfbff,
    ];
    for columns in [32, 96, 4096] {
        let row_bytes = columns / 32 * 34;
        for round in 0..3 {
            let mut table = Vec::with_capacity(row_bytes * 3);
            for row in 0..3 {
                for block in 0..columns / 32 {
                    table.extend_from_slice(
                        &scales[(block + row + round) % scales.len()].to_le_bytes(),
                    );
                    table.extend(
                        (0..32).map(|within| {
                            ((block * 32 + within) * 17 + row * 11 + round * 7) as u8
                        }),
                    );
                }
            }
            // The second 32/96-column row starts two bytes into a word. Its
            // exact slice must be consumed without alignment rounding or bleed.
            for row in 0..3 {
                let packed = &table[row * row_bytes..(row + 1) * row_bytes];
                let actual = staged_q8_row(packed, columns);
                let mut expected = vec![0.0; columns];
                beholdd_kernel::kquant::q8_0::dequant(packed, &mut expected);
                assert_eq!(
                    actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "columns={columns} row={row} round={round}"
                );
            }
        }
    }
}

#[test]
fn q8_row_preserves_signed_zero_and_half_subnormal_scales_for_every_i8() {
    for bits in [
        0_u16, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x3c00, 0xbc00, 0x7bff,
    ] {
        let scale = finite_half(bits);
        let mut packed = Vec::with_capacity(8 * 34);
        for block in 0..8 {
            packed.extend_from_slice(&bits.to_le_bytes());
            packed.extend((0..32).map(|i| (block * 32 + i) as u8));
        }
        let actual = staged_q8_row(&packed, 256);
        let mut expected = vec![0.0; 256];
        beholdd_kernel::kquant::q8_0::dequant(&packed, &mut expected);
        for (raw, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits());
            let quant = raw as u8 as i8;
            if scale == 0.0 || quant == 0 {
                assert_eq!(*actual, 0.0);
                assert_eq!(
                    actual.is_sign_negative(),
                    scale.is_sign_negative() ^ (quant < 0)
                );
            } else {
                assert_ne!(*actual, 0.0, "half subnormals must not flush to zero");
                assert_eq!(f64::from(*actual), f64::from(scale) * f64::from(quant));
            }
        }
    }
    assert_eq!(finite_half(1), 2.0_f32.powi(-24));
    // A nonzero finite half times an integer cannot produce an F32 subnormal.
    // The subnormal cases above concern the packed F16 scale, not the output.
    assert!(finite_half(1).is_normal());
}

fn sfu_round(value: f64, ulps: i32) -> f32 {
    let rounded = flush_subnormal(value as f32);
    if rounded.is_normal() {
        let adjusted = f32::from_bits(rounded.to_bits().checked_add_signed(ulps).unwrap());
        flush_subnormal(adjusted)
    } else {
        rounded
    }
}

fn staged_swiglu(gate: f32, up: f32, limit: f32, ulps: [i32; 2], rescale: bool) -> f32 {
    // Match the assembly's ordered select, rather than relying on min/max NaN rules.
    let clamped_gate = if limit > 0.0 && gate.partial_cmp(&limit) != Some(std::cmp::Ordering::Less)
    {
        limit
    } else {
        gate
    };
    let lower_up = if limit > 0.0 && up < -limit {
        -limit
    } else {
        up
    };
    let clamped_up = if limit > 0.0 && lower_up > limit {
        limit
    } else {
        lower_up
    };
    let exponent = clamped_gate * f32::from_bits(0xbfb8aa3b);
    let exponential = sfu_round(f64::from(flush_subnormal(exponent)).exp2(), ulps[0]);
    let denominator = 1.0 + exponential;
    let factor = if rescale && denominator > f32::from_bits(0x6f800000) {
        f32::from_bits(0x2f800000)
    } else {
        1.0
    };
    let reciprocal = sfu_round(
        1.0 / f64::from(flush_subnormal(denominator * factor)),
        ulps[1],
    );
    let quotient = (clamped_gate * reciprocal) * factor;
    quotient * clamped_up
}

fn reference_f64(gate: f32, up: f32, limit: f32) -> f64 {
    let clamped_gate = if limit > 0.0 { gate.min(limit) } else { gate };
    let clamped_up = if limit > 0.0 {
        up.clamp(-limit, limit)
    } else {
        up
    };
    let gate64 = f64::from(clamped_gate);
    (gate64 / (1.0 + (-gate64).exp())) * f64::from(clamped_up)
}

#[test]
fn staged_swiglu_moderate_finite_grid_has_explicit_tolerance() {
    for quarter in -80..=80 {
        let gate = quarter as f32 * 0.25;
        for up in [-20.0, -1.0, -0.0, 0.0, 0.25, 3.0, 20.0] {
            for limit in [0.0, 0.5, 2.0, 10.0, 20.0] {
                let reference = reference_f64(gate, up, limit);
                for exp_error in -1..=1 {
                    for rcp_error in -1..=1 {
                        let actual =
                            f64::from(staged_swiglu(gate, up, limit, [exp_error, rcp_error], true));
                        let tolerance = 2e-5 + 2e-5 * reference.abs();
                        assert!(
                            (actual - reference).abs() <= tolerance,
                            "gate={gate} up={up} limit={limit}: {actual} vs {reference}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn reciprocal_rescaling_preserves_large_up_times_small_negative_silu() {
    for gate in [-87.5_f32, -88.0, -88.5] {
        let up = 1e38_f32;
        let reference = f64::from(gate / (1.0 + (-gate).exp()) * up);
        assert_ne!(reference, 0.0);
        assert_eq!(staged_swiglu(gate, up, 0.0, [0, 0], false), -0.0);
        for exp_error in -1..=1 {
            for rcp_error in -1..=1 {
                let actual = f64::from(staged_swiglu(gate, up, 0.0, [exp_error, rcp_error], true));
                assert!((actual - reference).abs() <= 2e-5 * reference.abs());
            }
        }
    }
}

#[test]
fn staged_overflow_boundary_matches_f32_reference_classification() {
    // Adjacent f32 inputs straddle exp overflow. This is a staged model check,
    // not qualification of the physical SFU's overflow threshold or error bound.
    for magnitude in [
        f32::from_bits(0x42b17217),
        f32::from_bits(0x42b17218),
        f32::from_bits(0x42b17219),
    ] {
        let gate = -magnitude;
        let up = 1e38_f32;
        let expected = gate / (1.0 + magnitude.exp()) * up;
        let actual = staged_swiglu(gate, up, 0.0, [0, 0], true);
        assert_eq!(actual == 0.0, expected == 0.0, "gate={gate}");
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
        if expected != 0.0 {
            assert!((actual - expected).abs() <= 2e-5 * expected.abs());
        }
    }
}

#[test]
fn ordered_clamps_preserve_asymmetry_and_special_value_classification() {
    let below_limit = staged_swiglu(-4.0, 3.0, 2.0, [0, 0], true);
    let symmetric_wrong = staged_swiglu(-2.0, 3.0, 2.0, [0, 0], true);
    assert!((below_limit - reference_f64(-4.0, 3.0, 2.0) as f32).abs() < 2e-5);
    assert!((below_limit - symmetric_wrong).abs() > 0.1);
    assert_eq!(
        staged_swiglu(f32::NAN, 1.0, 2.0, [0, 0], true),
        staged_swiglu(2.0, 1.0, 2.0, [0, 0], true)
    );
    assert!(staged_swiglu(1.0, f32::NAN, 2.0, [0, 0], true).is_nan());
    assert!(staged_swiglu(f32::NAN, 1.0, 0.0, [0, 0], true).is_nan());
    assert!(staged_swiglu(f32::NEG_INFINITY, 1.0, 0.0, [0, 0], true).is_nan());
    assert_eq!(
        staged_swiglu(f32::INFINITY, 1.0, 0.0, [0, 0], true),
        f32::INFINITY
    );
    assert_eq!(
        staged_swiglu(-0.0, 1.0, 0.0, [0, 0], true).to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        staged_swiglu(0.0, -1.0, 0.0, [0, 0], true).to_bits(),
        (-0.0_f32).to_bits()
    );
    for limit in [-1.0, 0.0] {
        assert_eq!(
            staged_swiglu(1.0, 1.0, limit, [0, 0], true),
            staged_swiglu(1.0, 1.0, 0.0, [0, 0], true)
        );
    }
}

#[test]
fn scaled_add_contract_is_not_a_fused_multiply_add() {
    let scale = f32::from_bits(0x3f800001);
    let input = f32::from_bits(0x3f7ffffe);
    let old_output = -1.0_f32;
    let product = scale * input;
    let staged = old_output + product;
    assert_eq!(staged, 0.0);
    assert_ne!(scale.mul_add(input, old_output), staged);
}

// AMD's F64 reciprocal/rsqrt instructions are approximations, despite their
// binary64 input/output format. Use the 2^29-ULP precision envelope documented
// for these instructions in the RDNA ISA family as a stress model, not a claim
// that hardware rounding or exceptional cases have been qualified on GFX1151.
const F64_SFU_ULPS: i64 = 1 << 29;

fn f64_sfu_round(value: f64, ulps: i64) -> f64 {
    assert!((-F64_SFU_ULPS..=F64_SFU_ULPS).contains(&ulps));
    if value.is_normal() {
        assert!(value.is_sign_positive());
        f64::from_bits(value.to_bits().checked_add_signed(ulps).unwrap())
    } else {
        value
    }
}

fn staged_rmsnorm(input: &[f32], weights: Option<&[f32]>, eps: f32, ulps: [i64; 2]) -> Vec<f32> {
    assert!((1..=super::MAX_ELEMENTS).contains(&input.len()));
    assert!(eps.is_finite() && eps >= 0.0);
    if let Some(weights) = weights {
        assert_eq!(weights.len(), input.len());
    }
    let mut sum = 0.0_f64;
    for &value in input {
        let widened = f64::from(value);
        let square = widened * widened;
        sum += square;
    }
    let reciprocal_len = f64_sfu_round(1.0 / input.len() as f64, ulps[0]);
    let mean = sum * reciprocal_len;
    let denominator = mean + f64::from(eps);
    let inverse = f64_sfu_round(1.0 / denominator.sqrt(), ulps[1]) as f32;
    input
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let normalized = value * inverse;
            weights.map_or(normalized, |weights| normalized * weights[index])
        })
        .collect()
}

fn reference_rmsnorm(input: &[f32], weights: Option<&[f32]>, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    beholdd_kernel::vecops::rmsnorm(&mut output, input, weights, eps);
    output
}

#[test]
fn staged_rmsnorm_f64_sfu_envelope_matches_bounded_finite_grid() {
    for len in [1, 2, 31, 32, 63, 64, 65, 127, 128, 1024, 3072, 4096] {
        let input: Vec<_> = (0..len)
            .map(|index| ((index * 17 % 83) as f32 - 41.0) * 0.125)
            .collect();
        let weights: Vec<_> = (0..len)
            .map(|index| ((index * 7 % 37) as f32 - 18.0) * 0.25)
            .collect();
        for eps in [0.0, 1e-6, 1e-5, 0.5] {
            for weights in [None, Some(weights.as_slice())] {
                let expected = reference_rmsnorm(&input, weights, eps);
                for rcp_error in [-F64_SFU_ULPS, 0, F64_SFU_ULPS] {
                    for rsq_error in [-F64_SFU_ULPS, 0, F64_SFU_ULPS] {
                        let actual = staged_rmsnorm(&input, weights, eps, [rcp_error, rsq_error]);
                        for (&actual, &expected) in actual.iter().zip(&expected) {
                            let tolerance = 2e-6 + 2e-6 * f64::from(expected).abs();
                            assert!(actual.is_finite() && expected.is_finite());
                            assert!(
                                (f64::from(actual) - f64::from(expected)).abs() <= tolerance,
                                "len={len}, eps={eps}, SFU=[{rcp_error},{rsq_error}]: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn rmsnorm_f64_reduction_preserves_range_and_small_squared_terms() {
    // All inputs and expected outputs are normal F32. F32 squaring would
    // overflow or underflow, whereas widening before squaring preserves both.
    for magnitude in [1e20_f32, 1e-30_f32] {
        let input = [magnitude, -magnitude];
        let actual = staged_rmsnorm(&input, None, 0.0, [0, 0]);
        assert_eq!(actual, reference_rmsnorm(&input, None, 0.0));
        assert!((actual[0] - 1.0).abs() <= f32::EPSILON);
        assert!((actual[1] + 1.0).abs() <= f32::EPSILON);
    }

    // Sequential F32 accumulation drops every one after 4096^2, while the CPU
    // and shader's ordered F64 accumulation retain all 4095 squared terms.
    let mut input = vec![1.0_f32; 4096];
    input[0] = 4096.0;
    let expected = reference_rmsnorm(&input, None, 0.0);
    let actual = staged_rmsnorm(&input, None, 0.0, [0, 0]);
    assert_eq!(actual, expected);
    let sum_f32 = input
        .iter()
        .fold(0.0_f32, |sum, &value| sum + value * value);
    let wrong = input[0] * (1.0 / (sum_f32 / input.len() as f32).sqrt());
    assert!((wrong - expected[0]).abs() > 0.001);
}

#[test]
fn rmsnorm_zero_epsilon_and_zero_signs_have_explicit_model_behavior() {
    let input = [0.0_f32, -0.0];
    for eps in [1e-6, 0.5] {
        for ulps in [[-F64_SFU_ULPS; 2], [0; 2], [F64_SFU_ULPS; 2]] {
            let actual = staged_rmsnorm(&input, None, eps, ulps);
            assert_eq!(actual[0].to_bits(), 0.0_f32.to_bits());
            assert_eq!(actual[1].to_bits(), (-0.0_f32).to_bits());
            let weighted = staged_rmsnorm(&input, Some(&[-1.0, -1.0]), eps, ulps);
            assert_eq!(weighted[0].to_bits(), (-0.0_f32).to_bits());
            assert_eq!(weighted[1].to_bits(), 0.0_f32.to_bits());
        }
    }
    // eps=0 is admitted. A zero vector produces 0*infinity in the modeled CPU
    // and shader arithmetic; this exceptional case is not a finite live proof.
    assert!(reference_rmsnorm(&input, None, 0.0)
        .iter()
        .all(|value| value.is_nan()));
    assert!(staged_rmsnorm(&input, None, 0.0, [0, 0])
        .iter()
        .all(|value| value.is_nan()));
}

#[test]
fn rmsnorm_weight_stage_uses_current_vectors_and_two_f32_multiplications() {
    let mut input = [0.25_f32, -2.0, 3.0];
    let mut weights = [1.5_f32, -0.75, 0.25];
    let original = staged_rmsnorm(&input, Some(&weights), 1e-5, [0, 0]);
    input[0] = 2.25;
    weights[2] = -1.25;
    let normalized = staged_rmsnorm(&input, None, 1e-5, [0, 0]);
    let actual = staged_rmsnorm(&input, Some(&weights), 1e-5, [0, 0]);
    assert_ne!(actual, original);
    for ((&actual, &normalized), &weight) in actual.iter().zip(&normalized).zip(&weights) {
        assert_eq!(actual.to_bits(), (normalized * weight).to_bits());
    }
    assert_eq!(actual, reference_rmsnorm(&input, Some(&weights), 1e-5));
}

fn staged_rope_pair(a: f32, b: f32, c: f32, sin: f32, inverse: bool) -> [f32; 2] {
    let sign = if inverse { -1.0_f32 } else { 1.0_f32 };
    let s = sign * sin;
    let ac = a * c;
    let bs = b * s;
    let first = ac - bs;
    let as_ = a * s;
    let bc = b * c;
    let second = as_ + bc;
    [first, second]
}

#[test]
fn rope_pair_model_matches_both_cpu_layouts_and_directions() {
    for pairs in [1, 31, 32, 63, 64, 65, 127, 128, 1024, 2048] {
        for round in 0..3 {
            let input: Vec<_> = (0..pairs * 2)
                .map(|i| match i % 11 {
                    0 => -0.0,
                    1 => f32::from_bits(1),
                    _ => (((i * 17 + round * 3) % 47) as f32 - 23.0) * 0.125,
                })
                .collect();
            let cos: Vec<_> = (0..pairs)
                .map(|i| (((i * 7 + round) % 17) as f32 - 8.0) * 0.125)
                .collect();
            let sin: Vec<_> = (0..pairs)
                .map(|i| (((i * 3 + round * 5) % 13) as f32 - 6.0) * 0.125)
                .collect();
            for split_half in [false, true] {
                for inverse in [false, true] {
                    let mut expected = input.clone();
                    if split_half {
                        beholdd_kernel::vecops::rope_apply_split_half(
                            &mut expected,
                            &cos,
                            &sin,
                            inverse,
                        );
                    } else {
                        beholdd_kernel::vecops::rope_apply(&mut expected, &cos, &sin, inverse);
                    }
                    let mut actual = input.clone();
                    for pair in 0..pairs {
                        let [first, second] = if split_half {
                            [pair, pair + pairs]
                        } else {
                            [2 * pair, 2 * pair + 1]
                        };
                        let output = staged_rope_pair(
                            input[first],
                            input[second],
                            cos[pair],
                            sin[pair],
                            inverse,
                        );
                        actual[first] = output[0];
                        actual[second] = output[1];
                    }
                    assert!(actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits()));
                }
            }
        }
    }
}

#[test]
fn rope_pair_products_must_round_before_addition_and_subtraction() {
    let a = f32::from_bits(0x3f800001);
    let c = f32::from_bits(0x3f7ffffe);
    let first = staged_rope_pair(a, 1.0, c, 1.0, false)[0];
    assert_eq!(first, 0.0);
    assert_ne!(a.mul_add(c, -1.0), first);
    let second = staged_rope_pair(a, -1.0, 1.0, c, false)[1];
    assert_eq!(second, 0.0);
    assert_ne!(a.mul_add(c, -1.0), second);
}

fn scaled_attention_scores(scores: &[f64], scale: f32) -> Vec<f32> {
    scores
        .iter()
        .map(|score| (*score * f64::from(scale)) as f32)
        .collect()
}

fn staged_attention_softmax(
    scores: &[f64],
    scale: f32,
    exp_error: i32,
    rcp_error: i32,
) -> Vec<f32> {
    assert!((1..=super::MAX_ATTENTION_POSITIONS).contains(&scores.len()));
    assert!(scale.is_finite() && scale > 0.0);
    let values = scaled_attention_scores(scores, scale);
    let mut maximum = f32::NEG_INFINITY;
    for &value in &values {
        if value > maximum {
            maximum = value;
        }
    }
    if !maximum.is_finite() {
        return vec![0.0; values.len()];
    }
    let mut sum = 0.0_f32;
    for (index, &value) in values.iter().enumerate() {
        sum += attention_exponential(value, maximum, index, exp_error);
    }
    let inverse = sfu_round(1.0 / f64::from(sum), rcp_error);
    let mut result = vec![f32::NAN; values.len()];
    // Every lane has the same ordered maximum and F32 sum. Only the final
    // independent output ownership is striped across this one wave64 group.
    for lane in 0..64.min(values.len()) {
        for index in (lane..values.len()).step_by(64) {
            result[index] =
                attention_exponential(values[index], maximum, index, exp_error) * inverse;
        }
    }
    result
}

fn attention_exponential(value: f32, maximum: f32, index: usize, exp_error: i32) -> f32 {
    let finite = value.is_finite();
    let selected = if finite { value } else { maximum };
    let difference = selected - maximum;
    let exponent = difference * f32::from_bits(0x3fb8aa3b);
    // ±2 means alternating ±1-ULP error, exercising nonuniform SFU error
    // as well as the uniform -1/0/+1 envelope used by the other modes.
    let ulps = if exp_error.abs() == 2 {
        exp_error.signum() * if index.is_multiple_of(2) { 1 } else { -1 }
    } else {
        exp_error
    };
    let exponential = sfu_round(f64::from(flush_subnormal(exponent)).exp2(), ulps);
    if finite {
        exponential
    } else {
        0.0
    }
}

fn reference_attention_softmax(scores: &[f64], scale: f32) -> Vec<f32> {
    let mut values = scaled_attention_scores(scores, scale);
    beholdd_kernel::vecops::softmax(&mut values);
    values
}

// The historical ordered F32 denominator need not normalize a long vector to
// within 2e-5 of one, even in the CPU oracle. Bound that reduction separately,
// then check output mass against E/S: E is the F64 sum of staged exponential
// terms and S is their ordered F32 sum. This still detects missing/duplicated
// output terms without changing either the reduction order or CPU parity bound.
fn long_context_attention_mass_bound(
    scores: &[f64],
    scale: f32,
    exp_error: i32,
    rcp_error: i32,
) -> (f64, f64) {
    let count = scores.len();
    assert!((2049..=super::MAX_ATTENTION_POSITIONS).contains(&count));
    assert!((-1..=1).contains(&rcp_error));
    let values = scaled_attention_scores(scores, scale);
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(maximum.is_finite());
    let mut ordered = 0.0_f32;
    let mut accurate = 0.0_f64;
    for (index, &value) in values.iter().enumerate() {
        let term = attention_exponential(value, maximum, index, exp_error);
        assert!(term.is_finite() && term >= 0.0);
        // SFU outputs are normal or flushed to zero, so the positive ordered
        // additions have neither cancellation nor subnormal rounding errors.
        assert!(term == 0.0 || term.is_normal());
        ordered += term;
        accurate += f64::from(term);
    }
    assert!(ordered.is_normal() && ordered > 0.0);
    assert!(((1.0 / f64::from(ordered)) as f32).is_normal());

    let u32 = f64::from(f32::EPSILON) * 0.5;
    let u64 = f64::EPSILON * 0.5;
    let additions = (count - 1) as f64;
    let gamma32 = additions * u32 / (1.0 - additions * u32);
    let gamma64 = additions * u64 / (1.0 - additions * u64);
    // The first addition to zero is exact. Standard positive-sum gamma bounds
    // cover both reductions; divide by 1-gamma64 because E is rounded too.
    let denominator_bound = (gamma32 + gamma64) * accurate / (1.0 - gamma64);
    assert!(
        (f64::from(ordered) - accurate).abs() <= denominator_bound,
        "count={count}, exp_error={exp_error}: ordered={ordered}, accurate={accurate}, denominator_bound={denominator_bound}"
    );

    let predicted_mass = accurate / f64::from(ordered);
    // One F32 rounding plus each modeled ULP perturbation costs at most
    // (1 + 2*abs(rcp_error))*u32 relatively. One extra u32 conservatively
    // covers the preceding F64 division and rounding of this error bound.
    let reciprocal_error = (2.0 + 2.0 * f64::from(rcp_error.abs())) * u32;
    let product_error = reciprocal_error + u32 + reciprocal_error * u32;
    let output_sum_error = product_error + gamma64 + product_error * gamma64;
    // E/S has the F64 positive-sum error followed by one F64 division.
    let predicted_error = gamma64 + u64 + gamma64 * u64;
    // F32 products can underflow; their absolute error is bounded by half of
    // the smallest subnormal. The final observed mass is summed in F64.
    let underflow = count as f64 * f64::from(f32::from_bits(1)) * 0.5 * (1.0 + gamma64);
    let mass_bound =
        predicted_mass * (output_sum_error + predicted_error) / (1.0 - predicted_error) + underflow;
    (predicted_mass, mass_bound)
}

#[test]
fn attention_softmax_sfu_stress_matches_declared_finite_grid_bound() {
    // Each case is one softmax vector, so the larger context boundaries keep
    // O(C) storage/work without constructing a C-by-C attention matrix.
    for count in [
        1, 2, 31, 32, 63, 64, 65, 127, 128, 255, 256, 257, 341, 342, 511, 512, 513, 1023, 1024,
        1025, 2047, 2048, 8192, 16384,
    ] {
        for shape in 0..4 {
            let scores: Vec<_> = (0..count)
                .map(|i| match shape {
                    0 => 0.0,
                    1 => ((i * 17 % 83) as f64 - 41.0) * 0.5,
                    2 => {
                        if i == count - 1 {
                            32.0
                        } else {
                            -32.0
                        }
                    }
                    _ => f64::from(f32::from_bits(0x3f800000 + (i % 4) as u32)) + 1e-8,
                })
                .collect();
            for scale in [0.0625_f32, 0.088388346, 0.5, 1.0, 2.0] {
                let expected = reference_attention_softmax(&scores, scale);
                let reference_sum: f64 = expected.iter().map(|value| f64::from(*value)).sum();
                for exp_error in -2..=2 {
                    for rcp_error in -1..=1 {
                        let actual = staged_attention_softmax(&scores, scale, exp_error, rcp_error);
                        for (&actual, &expected) in actual.iter().zip(&expected) {
                            assert!(actual.is_finite() && actual >= 0.0);
                            let allowed = 2e-6 + 2e-5 * f64::from(expected).abs();
                            assert!((f64::from(actual) - f64::from(expected)).abs() <= allowed,
                                "count={count}, shape={shape}, scale={scale}, SFU=[{exp_error},{rcp_error}]: {actual} vs {expected}");
                        }
                        let sum: f64 = actual.iter().map(|value| f64::from(*value)).sum();
                        if count <= 2048 {
                            assert!((sum - 1.0).abs() <= 2e-5,
                                "count={count}, shape={shape}, scale={scale}, SFU=[{exp_error},{rcp_error}]: sum={sum}, reference_sum={reference_sum}");
                        } else {
                            let (predicted_mass, mass_bound) = long_context_attention_mass_bound(
                                &scores, scale, exp_error, rcp_error,
                            );
                            assert!((sum - predicted_mass).abs() <= mass_bound,
                                "count={count}, shape={shape}, scale={scale}, SFU=[{exp_error},{rcp_error}]: sum={sum}, reference_sum={reference_sum}, predicted_mass={predicted_mass}, mass_bound={mass_bound}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn attention_softmax_nonfinite_masks_follow_cpu_classification() {
    for scores in [
        vec![f64::NEG_INFINITY; 3],
        vec![f64::NAN; 3],
        vec![0.0, f64::INFINITY, -1.0],
        vec![f64::NAN, f64::NEG_INFINITY, 0.0, 1.0],
        vec![-f64::from(f32::MAX), f64::from(f32::MAX)],
    ] {
        let expected = reference_attention_softmax(&scores, 1.0);
        let actual = staged_attention_softmax(&scores, 1.0, 0, 0);
        for (&actual, &expected) in actual.iter().zip(&expected) {
            assert!(actual.is_finite() && expected.is_finite());
            assert_eq!(actual == 0.0, expected == 0.0);
            assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
            assert!((actual - expected).abs() <= 2e-6 + 2e-5 * expected.abs());
        }
    }
}

#[test]
fn attention_strided_outputs_preserve_nonfinite_and_tail_classification_through_16384() {
    for count in [
        255, 256, 257, 341, 342, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 8191, 8192, 16383,
        16384,
    ] {
        for mode in 0..5 {
            let mut scores: Vec<_> = (0..count).map(|i| (i % 7) as f64 - 3.0).collect();
            match mode {
                0 => scores.fill(f64::NAN),
                1 => scores.fill(f64::NEG_INFINITY),
                2 => scores[count - 1] = f64::INFINITY,
                3 => {
                    for (index, value) in [
                        (0, f64::NAN),
                        (63, f64::NEG_INFINITY),
                        (64, f64::NAN),
                        (count - 1, f64::NEG_INFINITY),
                    ] {
                        scores[index] = value;
                    }
                }
                _ => {
                    scores.fill(f64::NEG_INFINITY);
                    scores[count - 1] = -0.0;
                }
            }
            let actual = staged_attention_softmax(&scores, 1.0, 0, 0);
            let expected = reference_attention_softmax(&scores, 1.0);
            for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
                assert!(actual.is_finite() && expected.is_finite());
                assert_eq!(
                    actual == 0.0,
                    expected == 0.0,
                    "count={count} mode={mode} index={index}"
                );
                assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
                assert!((actual - expected).abs() <= 2e-6 + 2e-5 * expected.abs());
            }
        }
    }
}

#[test]
fn attention_softmax_sum_order_does_not_follow_strided_output_ownership() {
    for count in [
        255, 256, 257, 341, 342, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 8192, 16384,
    ] {
        let mut scores = vec![-20.0; count];
        scores[0] = 0.0;
        let actual = staged_attention_softmax(&scores, 1.0, 0, 0);
        // Each tiny exponential is lost individually after the leading one.
        assert_eq!(actual[0], 1.0);
        scores.reverse();
        let reverse = staged_attention_softmax(&scores, 1.0, 0, 0);
        // Accumulating those terms before the one preserves their total. A
        // tree/striped reduction would silently change this historical order.
        assert!(reverse[count - 1] < 1.0);
        assert!(actual[1..].iter().all(|value| *value > 0.0));
    }
}

#[test]
fn attention_scores_stay_f64_until_after_scaling() {
    let query = [1e20_f32, 1e20_f32];
    let keys = [[1e20_f32, -1e20_f32], [0.0, 1e-20]];
    let scores: Vec<f64> = keys
        .iter()
        .map(|key| {
            query
                .iter()
                .zip(key)
                .map(|(q, k)| f64::from(*q) * f64::from(*k))
                .sum()
        })
        .collect();
    assert_eq!(scores[0], 0.0);
    assert!(scores[1].is_finite() && scores[1] > 0.0);
    let wrong = query[0] * keys[0][0] + query[1] * keys[0][1];
    assert!(wrong.is_nan());
    let scaled = [1e40_f64, 1e40_f64];
    let actual = staged_attention_softmax(&scaled, 1e-20, 0, 0);
    assert_eq!(actual, reference_attention_softmax(&scaled, 1e-20));
    assert_eq!(actual, [0.5, 0.5]);
    let prematurely_narrowed = scaled.map(|value| f64::from(value as f32));
    assert_eq!(
        staged_attention_softmax(&prematurely_narrowed, 1e-20, 0, 0),
        [0.0, 0.0]
    );
}

#[test]
fn attention_value_f64_reduction_preserves_small_weighted_terms() {
    for count in [256, 512, 1024, 2048, 8192, 16384] {
        let probabilities = vec![1.0_f32 / count as f32; count];
        let mut values = vec![1.0_f32; count];
        values[0] = 1e10;
        values[count - 1] = -1e10;
        let mut correct = 0.0_f64;
        let mut wrong = 0.0_f32;
        for (probability, value) in probabilities.iter().zip(values) {
            let product = f64::from(*probability) * f64::from(value);
            correct += product;
            wrong += *probability * value;
        }
        assert_eq!(correct as f32, (count - 2) as f32 / count as f32);
        assert_eq!(wrong, 0.0);
    }
}
