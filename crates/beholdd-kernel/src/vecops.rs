//! The small operators: norms, softmax, activations, rotary embedding.
//!
//! Individually cheap. Collectively they are why a decode step is ~1,300 parallel
//! regions rather than a few dozen, which is the thing that made thread count matter
//! 9.5x more than it should. They are written here as plain loops over slices, with no
//! threading of their own, so that a scheduler above can decide whether a given call is
//! worth a region at all — see `granularity`.
//!
//! Accumulation is f64 where a reduction runs over a whole hidden dimension and f32
//! where it does not. That is not caution, it is the same reasoning applied twice:
//! precision should match what the reduction can actually lose, and a sum of 4096 terms
//! can lose more than a sum of 32.

/// How much work an operator is about to do, so the scheduler can decide whether to
/// parallelise it.
///
/// This exists because of a measurement, not a theory. On a 16-core SMT2 box, running
/// many small parallel regions on 32 threads instead of 16 cost **9.5x** — while one
/// large region (an lm_head over 129,280 rows) was *unchanged*, and a pure-vector
/// control was 22% **faster** on 32. The penalty is per fork/join barrier, not per flop.
/// So the number that decides threading is the work per region, and only the kernel
/// knows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Granularity {
    /// Independent items the work splits into (rows, heads, tokens).
    pub items: usize,
    /// Roughly how many multiply-accumulates each item costs.
    pub flops_per_item: usize,
}

impl Granularity {
    /// Total multiply-accumulates.
    pub fn total(&self) -> usize {
        self.items.saturating_mul(self.flops_per_item)
    }

    /// Whether a fork/join is likely to pay for itself.
    ///
    /// The threshold is deliberately crude and deliberately documented as such: the
    /// measured effect is a factor of ten, so the cost of getting this boundary
    /// approximately right is small compared to the cost of not having it at all. It
    /// should be replaced by a calibrated number once the scheduler can measure its own
    /// barrier cost on the host.
    pub fn worth_parallelising(&self) -> bool {
        self.items > 1 && self.total() >= 1 << 16
    }
}

/// RMS normalisation: `y = x / sqrt(mean(x^2) + eps) * w`, with `w` optional.
///
/// f64 accumulation for the sum of squares. Over 4096 terms in f32 the relative error
/// grows with sqrt(n) and this value divides the entire hidden state, so an error here
/// is not local to one element.
pub fn rmsnorm(y: &mut [f32], x: &[f32], w: Option<&[f32]>, eps: f32) {
    assert_eq!(y.len(), x.len());
    if let Some(w) = w {
        assert_eq!(w.len(), x.len());
    }
    let ss: f64 = x.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let inv = (1.0 / (ss / x.len() as f64 + eps as f64).sqrt()) as f32;
    for (i, out) in y.iter_mut().enumerate() {
        *out = x[i] * inv * w.map_or(1.0, |w| w[i]);
    }
}

/// Softmax in place, numerically stable (subtract the max first).
///
/// `-inf` entries are permitted and produce exactly zero: masked positions must
/// contribute nothing rather than a small something.
pub fn softmax(v: &mut [f32]) {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !m.is_finite() {
        // Every entry is -inf: there is nothing to attend to. Leaving NaN here would
        // propagate through the whole row.
        v.fill(0.0);
        return;
    }
    let mut sum = 0.0f32;
    for e in v.iter_mut() {
        *e = if e.is_finite() { (*e - m).exp() } else { 0.0 };
        sum += *e;
    }
    let inv = 1.0 / sum;
    for e in v.iter_mut() {
        *e *= inv;
    }
}

/// SiLU (swish): `x * sigmoid(x)`.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU: `down_input[i] = silu(gate[i]) * up[i]`, written into `gate`.
///
/// `limit` clamps both operands when non-zero, which some checkpoints require and others
/// do not. **Note that the clamps are asymmetric**: `gate` is clamped only from above
/// while `up` is clamped both ways. That is what the reference does, and making them
/// symmetric "for consistency" changes the activation.
pub fn swiglu(gate: &mut [f32], up: &[f32], limit: f32) {
    assert_eq!(gate.len(), up.len());
    for (g, u) in gate.iter_mut().zip(up) {
        let (mut gv, mut uv) = (*g, *u);
        if limit > 0.0 {
            gv = gv.min(limit);
            uv = uv.clamp(-limit, limit);
        }
        *g = silu(gv) * uv;
    }
}

/// Rotary embedding frequencies, with optional YaRN interpolation.
///
/// YaRN engages only when `orig_seq_len > 0`. Architectures commonly disable it on some
/// layer types and not others — a model can have two rope tables — so this returns the
/// frequencies and lets the caller own which one a layer uses.
pub fn rope_freqs(
    theta: f64,
    dim: usize,
    orig_seq_len: usize,
    factor: f64,
    beta_fast: f64,
    beta_slow: f64,
) -> Vec<f64> {
    let half = dim / 2;
    let mut freqs: Vec<f64> = (0..half)
        .map(|i| 1.0 / theta.powf((2 * i) as f64 / dim as f64))
        .collect();
    if orig_seq_len > 0 {
        let corr = |rot: f64| {
            dim as f64 * (orig_seq_len as f64 / (rot * 2.0 * std::f64::consts::PI)).ln()
                / (2.0 * theta.ln())
        };
        let (mut lo, mut hi) = (corr(beta_fast).floor(), corr(beta_slow).ceil());
        lo = lo.max(0.0);
        hi = hi.min(dim as f64 - 1.0);
        if lo == hi {
            hi += 0.001;
        }
        for (i, f) in freqs.iter_mut().enumerate() {
            let ramp = ((i as f64 - lo) / (hi - lo)).clamp(0.0, 1.0);
            *f = *f / factor * ramp + *f * (1.0 - ramp);
        }
    }
    freqs
}

/// Cos/sin for one position.
pub fn rope_at(freqs: &[f64], pos: usize, cos: &mut [f32], sin: &mut [f32]) {
    assert_eq!(cos.len(), freqs.len());
    assert_eq!(sin.len(), freqs.len());
    for (i, f) in freqs.iter().enumerate() {
        let a = pos as f64 * f;
        cos[i] = a.cos() as f32;
        sin[i] = a.sin() as f32;
    }
}

/// Apply a rotation in place. Pairs are **interleaved**: `(x0,x1)` is one complex
/// number, `(x2,x3)` the next — not split-half.
///
/// `inverse` conjugates (negates the sine). Some architectures de-rotate the attention
/// output on the way back out, which is easy to miss because rope is applied twice on
/// the way in and once, inverted, on the way out.
pub fn rope_apply(x: &mut [f32], cos: &[f32], sin: &[f32], inverse: bool) {
    assert_eq!(x.len(), cos.len() * 2);
    let sgn = if inverse { -1.0 } else { 1.0 };
    for i in 0..cos.len() {
        let (a, b) = (x[2 * i], x[2 * i + 1]);
        let (c, s) = (cos[i], sgn * sin[i]);
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = a * s + b * c;
    }
}

/// Apply a split-half (NEOX) rotary embedding in place.
///
/// The first half of `x` supplies the real components and the second half supplies the
/// imaginary components: `(x[0], x[n])` is one complex number, `(x[1], x[n + 1])`
/// the next. This is the convention used by Qwen-family models and Hugging Face's
/// `rotate_half`; it is deliberately separate from [`rope_apply`], whose pairs are
/// adjacent.
///
/// `inverse` conjugates the rotation by negating the sine.
pub fn rope_apply_split_half(x: &mut [f32], cos: &[f32], sin: &[f32], inverse: bool) {
    assert_eq!(x.len(), cos.len() * 2);
    assert_eq!(sin.len(), cos.len());
    let half = cos.len();
    let sgn = if inverse { -1.0 } else { 1.0 };
    for i in 0..half {
        let (a, b) = (x[i], x[i + half]);
        let (c, s) = (cos[i], sgn * sin[i]);
        x[i] = a * c - b * s;
        x[i + half] = a * s + b * c;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rmsnorm_normalises_to_unit_rms() {
        let x: Vec<f32> = (0..512).map(|i| ((i as f32) * 0.1).sin() * 3.0).collect();
        let mut y = vec![0.0f32; 512];
        rmsnorm(&mut y, &x, None, 1e-6);
        let rms = (y.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 512.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "rms {rms}");
    }

    #[test]
    fn rmsnorm_applies_the_weight() {
        let x = vec![1.0f32; 4];
        let w = vec![2.0f32, 3.0, 4.0, 5.0];
        let mut y = vec![0.0f32; 4];
        rmsnorm(&mut y, &x, Some(&w), 0.0);
        // x is already unit-rms, so y == w
        assert_eq!(y, w);
    }

    #[test]
    fn softmax_sums_to_one_and_masks_are_exactly_zero() {
        let mut v = vec![1.0f32, 2.0, 3.0, f32::NEG_INFINITY];
        softmax(&mut v);
        assert_eq!(
            v[3], 0.0,
            "a masked position must contribute exactly nothing"
        );
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
    }

    #[test]
    fn softmax_survives_an_entirely_masked_row() {
        let mut v = vec![f32::NEG_INFINITY; 4];
        softmax(&mut v);
        // NaN here would propagate through the entire attention output.
        assert!(v.iter().all(|x| *x == 0.0), "got {v:?}");
    }

    #[test]
    fn softmax_is_stable_at_large_magnitudes() {
        let mut v = vec![1000.0f32, 1001.0, 1002.0];
        softmax(&mut v);
        assert!(v.iter().all(|x| x.is_finite()), "overflowed: {v:?}");
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn swiglu_clamps_asymmetrically() {
        // gate is clamped only from ABOVE; up is clamped both ways.
        let mut gate = vec![100.0f32, -100.0];
        let up = vec![100.0f32, -100.0];
        swiglu(&mut gate, &up, 10.0);
        // gate[0] -> 10, up[0] -> 10  => silu(10)*10 ~= 10*10
        assert!((gate[0] - silu(10.0) * 10.0).abs() < 1e-4);
        // gate[1] stays -100 (NOT clamped up to -10), up[1] -> -10
        assert!((gate[1] - silu(-100.0) * -10.0).abs() < 1e-4);
    }

    #[test]
    fn interleaved_rope_round_trips_through_its_inverse() {
        let freqs = rope_freqs(10000.0, 64, 0, 1.0, 32.0, 1.0);
        let (mut cos, mut sin) = (vec![0.0f32; 32], vec![0.0f32; 32]);
        rope_at(&freqs, 733, &mut cos, &mut sin);
        let orig: Vec<f32> = (0..64).map(|i| ((i as f32) * 0.3).sin()).collect();
        let mut x = orig.clone();
        rope_apply(&mut x, &cos, &sin, false);
        assert!(x != orig, "rotation did nothing");
        rope_apply(&mut x, &cos, &sin, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn split_half_rope_matches_neox_rotation() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0];
        rope_apply_split_half(&mut x, &[0.0, 0.0], &[1.0, 1.0], false);
        assert_eq!(x, vec![-3.0, -4.0, 1.0, 2.0]);

        // Adjacent pairs would instead produce [-2, 1, -4, 3]. Keep this assertion
        // explicit: both conventions have the same shape and silently mixing them is
        // a plausible model-level error.
        let mut interleaved = vec![1.0f32, 2.0, 3.0, 4.0];
        rope_apply(&mut interleaved, &[0.0, 0.0], &[1.0, 1.0], false);
        assert_eq!(interleaved, vec![-2.0, 1.0, -4.0, 3.0]);
    }

    #[test]
    fn split_half_rope_round_trips_through_its_inverse() {
        let freqs = rope_freqs(1_000_000.0, 128, 0, 1.0, 32.0, 1.0);
        let (mut cos, mut sin) = (vec![0.0f32; 64], vec![0.0f32; 64]);
        rope_at(&freqs, 1234, &mut cos, &mut sin);
        let orig: Vec<f32> = (0..128).map(|i| ((i as f32) * 0.17).cos()).collect();
        let mut x = orig.clone();
        rope_apply_split_half(&mut x, &cos, &sin, false);
        assert_ne!(x, orig, "rotation did nothing");
        rope_apply_split_half(&mut x, &cos, &sin, true);
        for (a, b) in x.iter().zip(&orig) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn yarn_changes_the_frequencies_it_is_supposed_to() {
        let plain = rope_freqs(10000.0, 64, 0, 16.0, 32.0, 1.0);
        let yarn = rope_freqs(10000.0, 64, 65536, 16.0, 32.0, 1.0);
        // High-frequency dims (early indices) are left alone; low-frequency ones are
        // interpolated. If YaRN touched everything, long-context behaviour would be
        // wrong in a way that only shows up past the original window.
        assert!(
            (plain[0] - yarn[0]).abs() < 1e-12,
            "high freq must be untouched"
        );
        assert!(
            yarn[31] < plain[31] * 0.9,
            "low freq must be interpolated down"
        );
    }

    #[test]
    fn granularity_separates_big_regions_from_small_ones() {
        // An lm_head: one huge region, obviously worth it.
        let head = Granularity {
            items: 129_280,
            flops_per_item: 4096,
        };
        assert!(head.worth_parallelising());
        // A per-head rmsnorm over 512 elements: not worth a barrier.
        let norm = Granularity {
            items: 1,
            flops_per_item: 512,
        };
        assert!(!norm.worth_parallelising());
    }
}
