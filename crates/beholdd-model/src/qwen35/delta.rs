//! Scalar CPU reference for one Qwen3.5 Gated DeltaNet token.
//!
//! The projections and final output matrix multiply belong to the caller. These
//! inputs use the **GGUF** convention: `a` is already `-exp(A_log)`, value heads
//! have already been reordered to tiled order, and `norm` is the stored SSM
//! norm weight (no added one). Value head `h` therefore uses key head `h % Hk`.
//!
//! Reference: llama.cpp b11223, commit 4da6337767f973e2b4d0797e5b323d77d8565e4a,
//! `src/models/qwen35.cpp`, `src/models/models.h`, and
//! `ggml/src/ggml-cpu/ops.cpp` (`ssm_conv` and `gated_delta_net`). This module
//! specifies the per-token equations, not bitwise equivalence to SIMD reductions.

use anyhow::{anyhow, bail, ensure, Result};

/// Fixed dimensions and normalization epsilon for one recurrent layer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeltaShape {
    /// Number of projected query heads and key heads.
    pub key_heads: usize,
    /// Number of value heads; must be a positive multiple of `key_heads`.
    pub value_heads: usize,
    /// Qwen3.5 has equal key and value dimensions within a head.
    pub head_dim: usize,
    /// Number of causal convolution taps, including the current token.
    pub conv_kernel: usize,
    /// Positive finite epsilon for query/key L2 and output RMS normalization.
    pub epsilon: f32,
}

#[derive(Clone, Copy, Debug)]
struct Geometry {
    key_width: usize,
    value_width: usize,
    channels: usize,
    conv_weights: usize,
    conv_state: usize,
    recurrent_state: usize,
}

impl DeltaShape {
    /// Exact retained convolution and recurrent state payload, in bytes.
    ///
    /// This validates the shape and checks all arithmetic without allocating.
    /// Vector metadata, allocator bookkeeping and caller-owned buffers are excluded.
    pub fn state_bytes(self) -> Result<usize> {
        let g = self.geometry()?;
        payload_bytes(&[g.conv_state, g.recurrent_state])
    }

    /// Peak additional payload allocated by one transactional token step, in bytes.
    ///
    /// Includes replacement histories, convolved QKV, and the returned output.
    /// The previous histories remain live until the step commits. Caller-owned
    /// inputs/weights, vector metadata and allocator bookkeeping are excluded.
    /// Shape validation and checked arithmetic finish without allocating.
    pub fn step_scratch_bytes(self) -> Result<usize> {
        let g = self.geometry()?;
        payload_bytes(&[g.conv_state, g.channels, g.recurrent_state, g.value_width])
    }

    fn geometry(self) -> Result<Geometry> {
        ensure!(
            self.key_heads > 0 && self.value_heads > 0 && self.head_dim > 0 && self.conv_kernel > 0,
            "DeltaNet dimensions must be positive"
        );
        ensure!(
            self.value_heads % self.key_heads == 0,
            "DeltaNet value head count must be a multiple of key head count"
        );
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "DeltaNet epsilon must be finite and positive"
        );
        let key_width = product(self.key_heads, self.head_dim)?;
        let value_width = product(self.value_heads, self.head_dim)?;
        let channels = product(key_width, 2)?
            .checked_add(value_width)
            .ok_or_else(|| anyhow!("DeltaNet channel count overflows"))?;
        let geometry = Geometry {
            key_width,
            value_width,
            channels,
            conv_weights: product(channels, self.conv_kernel)?,
            conv_state: product(channels, self.conv_kernel - 1)?,
            recurrent_state: product(value_width, self.head_dim)?,
        };
        for count in [
            geometry.channels,
            geometry.conv_weights,
            geometry.conv_state,
            geometry.recurrent_state,
        ] {
            ensure!(
                product(count, std::mem::size_of::<f32>())? <= isize::MAX as usize,
                "DeltaNet buffer exceeds addressable size"
            );
        }
        Ok(geometry)
    }
}

/// One token's raw linear projections, before convolution or gate activations.
#[derive(Clone, Copy, Debug)]
pub struct DeltaInput<'a> {
    /// Concatenated [all Q heads, all K heads, all V heads], head dimension fastest.
    pub qkv: &'a [f32],
    /// Raw output-gate projection before SiLU, [value head][head dimension].
    pub z: &'a [f32],
    /// One raw decay projection per value head, before adding `dt`.
    pub alpha: &'a [f32],
    /// One raw update projection per value head, before sigmoid.
    pub beta: &'a [f32],
}

/// Small F32 parameters in their stored GGUF representation.
#[derive(Clone, Copy, Debug)]
pub struct DeltaWeights<'a> {
    /// Depthwise causal kernel [channel][oldest tap .. current tap].
    pub conv: &'a [f32],
    /// Stored GGUF `ssm_a`: non-positive `-exp(A_log)`, per value head.
    pub a: &'a [f32],
    /// Bias added to each value head's raw decay projection before softplus.
    pub dt: &'a [f32],
    /// SSM RMSNorm weights shared by value heads; length `head_dim`.
    pub norm: &'a [f32],
}

/// One sequence in one layer. Reuse only for the next token of that same prefix;
/// call `reset` before a new sequence. No process-global state is retained.
#[derive(Debug, PartialEq)]
pub struct DeltaState {
    shape: DeltaShape,
    // Channel-major history, oldest first; stores RAW projected QKV, not SiLU.
    conv: Vec<f32>,
    // [value head][value coordinate][key coordinate], keys contiguous.
    recurrent: Vec<f32>,
    tokens: u64,
}

impl DeltaState {
    /// Allocate zeroed histories only if retained state plus peak step scratch
    /// fits `byte_budget`. Both forecasts and the budget check precede allocation.
    ///
    /// The budget covers the payload described by [`DeltaShape::state_bytes`]
    /// and [`DeltaShape::step_scratch_bytes`], not caller-owned buffers or allocator
    /// bookkeeping. A caller retaining previous returned outputs must count them
    /// separately when enforcing its overall memory limit.
    pub fn new(shape: DeltaShape, byte_budget: usize) -> Result<Self> {
        let g = shape.geometry()?;
        let required = shape
            .state_bytes()?
            .checked_add(shape.step_scratch_bytes()?)
            .ok_or_else(|| anyhow!("DeltaNet state and scratch byte count overflows"))?;
        ensure!(
            required <= byte_budget,
            "DeltaNet state and step scratch need {required} bytes, budget is {byte_budget}"
        );
        Ok(Self {
            shape,
            conv: zeros(g.conv_state)?,
            recurrent: zeros(g.recurrent_state)?,
            tokens: 0,
        })
    }

    /// Return the fixed shape admitted by the constructor.
    pub fn shape(&self) -> DeltaShape {
        self.shape
    }
    /// Number of successfully committed tokens since construction or reset.
    pub fn tokens_seen(&self) -> u64 {
        self.tokens
    }
    /// Raw QKV history, indexed by channel then oldest-to-newest retained tap.
    pub fn convolution_state(&self) -> &[f32] {
        &self.conv
    }
    /// Recurrent matrices, indexed by value head, value coordinate, key coordinate.
    pub fn recurrent_state(&self) -> &[f32] {
        &self.recurrent
    }

    /// Zero both histories and the token count, retaining their allocations.
    pub fn reset(&mut self) {
        self.conv.fill(0.0);
        self.recurrent.fill(0.0);
        self.tokens = 0;
    }

    /// Return normalized/gated value heads, ready for `ssm_out` projection.
    ///
    /// Validation, allocations, and all arithmetic finish before committing either
    /// history. A refused token leaves both histories and `tokens_seen` unchanged.
    pub fn step(&mut self, input: DeltaInput<'_>, weights: DeltaWeights<'_>) -> Result<Vec<f32>> {
        let shape = self.shape;
        let g = shape.geometry()?;
        check_slice("QKV projection", input.qkv, g.channels)?;
        check_slice("Z projection", input.z, g.value_width)?;
        check_slice("alpha projection", input.alpha, shape.value_heads)?;
        check_slice("beta projection", input.beta, shape.value_heads)?;
        check_slice("convolution weights", weights.conv, g.conv_weights)?;
        check_slice("A weights", weights.a, shape.value_heads)?;
        check_slice("dt bias", weights.dt, shape.value_heads)?;
        check_slice("SSM norm weights", weights.norm, shape.head_dim)?;
        ensure!(
            weights.a.iter().all(|a| *a <= 0.0),
            "DeltaNet A must contain stored non-positive -exp(A_log) values"
        );
        let next_tokens = self
            .tokens
            .checked_add(1)
            .ok_or_else(|| anyhow!("DeltaNet token counter overflows"))?;

        let (next_conv, mut qkv) = convolve(shape, g, &self.conv, input.qkv, weights.conv)?;
        for x in &mut qkv {
            *x = finite(silu(*x), "convolution SiLU")?;
        }
        let (q, kv) = qkv.split_at_mut(g.key_width);
        let (k, v) = kv.split_at_mut(g.key_width);
        for head in q
            .chunks_exact_mut(shape.head_dim)
            .chain(k.chunks_exact_mut(shape.head_dim))
        {
            l2_normalize(head, shape.epsilon)?;
        }

        let mut next_recurrent = zeros(g.recurrent_state)?;
        let mut output = zeros(g.value_width)?;
        let head_elements = product(shape.head_dim, shape.head_dim)?;
        for h in 0..shape.value_heads {
            let key_head = h % shape.key_heads;
            let q_head = &q[key_head * shape.head_dim..(key_head + 1) * shape.head_dim];
            let k_head = &k[key_head * shape.head_dim..(key_head + 1) * shape.head_dim];
            let v_head = &v[h * shape.head_dim..(h + 1) * shape.head_dim];
            let z_head = &input.z[h * shape.head_dim..(h + 1) * shape.head_dim];
            let out = &mut output[h * shape.head_dim..(h + 1) * shape.head_dim];
            let previous = &self.recurrent[h * head_elements..(h + 1) * head_elements];
            let next = &mut next_recurrent[h * head_elements..(h + 1) * head_elements];
            let alpha = finite(input.alpha[h] + weights.dt[h], "biased alpha")?;
            let log_decay = finite(weights.a[h] * softplus(alpha), "log decay")?;
            let decay = finite(log_decay.exp(), "decay")?;
            let beta = sigmoid(input.beta[h]);
            recurrent_head(previous, next, q_head, k_head, v_head, decay, beta, out)?;
            rms_gate(out, z_head, weights.norm, shape.epsilon)?;
        }

        self.conv = next_conv;
        self.recurrent = next_recurrent;
        self.tokens = next_tokens;
        Ok(output)
    }
}

fn product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| anyhow!("DeltaNet dimension product overflows"))
}

fn payload_bytes(counts: &[usize]) -> Result<usize> {
    let count = counts.iter().try_fold(0usize, |sum, count| {
        sum.checked_add(*count)
            .ok_or_else(|| anyhow!("DeltaNet payload element count overflows"))
    })?;
    product(count, std::mem::size_of::<f32>())
}

fn zeros(count: usize) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|error| anyhow!("cannot allocate DeltaNet buffer: {error}"))?;
    values.resize(count, 0.0);
    Ok(values)
}

fn check_slice(label: &str, values: &[f32], expected: usize) -> Result<()> {
    ensure!(
        values.len() == expected,
        "DeltaNet {label} has {} values, expected {expected}",
        values.len()
    );
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "DeltaNet {label} contains a non-finite value"
    );
    Ok(())
}

fn finite(value: f32, operation: &str) -> Result<f32> {
    if !value.is_finite() {
        bail!("DeltaNet {operation} produced a non-finite value");
    }
    Ok(value)
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn softplus(value: f32) -> f32 {
    value.max(0.0) + (-value.abs()).exp().ln_1p()
}

fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

fn convolve(
    shape: DeltaShape,
    g: Geometry,
    history: &[f32],
    input: &[f32],
    weights: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let history_len = shape.conv_kernel - 1;
    let mut next = zeros(g.conv_state)?;
    let mut output = zeros(g.channels)?;
    for channel in 0..g.channels {
        let old = &history[channel * history_len..(channel + 1) * history_len];
        let kernel = &weights[channel * shape.conv_kernel..(channel + 1) * shape.conv_kernel];
        // GGML's causal convolution deliberately accumulates in f32, oldest first.
        let mut sum = 0.0f32;
        for (&value, &weight) in old.iter().zip(kernel) {
            sum += value * weight;
        }
        sum += input[channel] * kernel[history_len];
        output[channel] = finite(sum, "causal convolution")?;
        if history_len > 0 {
            let new = &mut next[channel * history_len..(channel + 1) * history_len];
            new[..history_len - 1].copy_from_slice(&old[1..]);
            new[history_len - 1] = input[channel];
        }
    }
    Ok((next, output))
}

fn rms_scale(values: &[f32], epsilon: f32) -> Result<f32> {
    // Like GGML RMSNorm: f32 products with a wider reduction, then f32 mean/scale.
    let mut sum = 0.0f64;
    for &value in values {
        sum += f64::from(finite(value * value, "norm square")?);
    }
    let mean = (sum / values.len() as f64) as f32;
    let denominator = finite(mean + epsilon, "norm denominator")?;
    finite(denominator.sqrt().recip(), "normalization scale")
}

fn l2_normalize(values: &mut [f32], epsilon: f32) -> Result<()> {
    // build_gdn_l2_norm = RMSNorm(x, eps / D) / sqrt(D), equivalent to
    // x / sqrt(sum(x*x) + eps). Preserve the reference's two multiplies.
    let dimension = values.len() as f32;
    let rms = rms_scale(values, epsilon / dimension)?;
    let scale = dimension.sqrt().recip();
    for value in values {
        *value = finite((*value * rms) * scale, "L2 normalization")?;
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> Result<f32> {
    let mut sum = 0.0f64;
    for (&a, &b) in left.iter().zip(right) {
        sum += f64::from(finite(a * b, "state dot product")?);
    }
    finite(sum as f32, "state dot reduction")
}

#[allow(clippy::too_many_arguments)]
fn recurrent_head(
    previous: &[f32],
    next: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
    output: &mut [f32],
) -> Result<()> {
    let dimension = q.len();
    let scale = (dimension as f32).sqrt().recip();
    for (j, row) in next.chunks_exact_mut(dimension).enumerate() {
        let old = &previous[j * dimension..(j + 1) * dimension];
        for (cell, &old_cell) in row.iter_mut().zip(old) {
            *cell = finite(old_cell * decay, "state decay")?;
        }
        // Predict from the DECAYED old state, before applying this token's delta.
        let delta = finite((v[j] - dot(row, k)?) * beta, "value delta")?;
        for (cell, &key) in row.iter_mut().zip(k) {
            *cell = finite(*cell + key * delta, "state update")?;
        }
        output[j] = finite(dot(row, q)? * scale, "state readout")?;
    }
    Ok(())
}

fn rms_gate(output: &mut [f32], z: &[f32], weights: &[f32], epsilon: f32) -> Result<()> {
    let scale = rms_scale(output, epsilon)?;
    for ((value, &gate), &weight) in output.iter_mut().zip(z).zip(weights) {
        *value = finite(((*value * scale) * weight) * silu(gate), "gated RMSNorm")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(actual: f32, expected: f64) {
        assert!(
            (f64::from(actual) - expected).abs() <= 3e-6 * expected.abs().max(1.0),
            "{actual} != {expected}"
        );
    }

    fn shape() -> DeltaShape {
        DeltaShape {
            key_heads: 1,
            value_heads: 1,
            head_dim: 1,
            conv_kernel: 1,
            epsilon: 0.2,
        }
    }

    fn budget(shape: DeltaShape) -> Result<usize> {
        shape
            .state_bytes()?
            .checked_add(shape.step_scratch_bytes()?)
            .ok_or_else(|| anyhow!("test budget overflow"))
    }

    #[test]
    fn forecasts_cover_live_payload_and_refuse_one_byte_below_budget() -> Result<()> {
        // Hk=2, Hv=4, D=3, K=4: 24 QKV channels, 72 history values,
        // 36 recurrent values and 12 output values.
        let shape = DeltaShape {
            key_heads: 2,
            value_heads: 4,
            head_dim: 3,
            conv_kernel: 4,
            epsilon: 1e-6,
        };
        assert_eq!(shape.state_bytes()?, 432);
        assert_eq!(shape.step_scratch_bytes()?, 576);
        let error = DeltaState::new(shape, 1007).unwrap_err().to_string();
        assert!(error.contains("need 1008 bytes, budget is 1007"));
        let state = DeltaState::new(shape, 1008)?;
        assert_eq!(state.convolution_state().len(), 72);
        assert_eq!(state.recurrent_state().len(), 36);

        // A huge, valid shape must also fail the budget gate rather than attempt
        // either allocation. The error identifies that pre-allocation gate.
        let huge = DeltaShape {
            conv_kernel: usize::MAX / 64,
            ..shape
        };
        // Keep all individual geometry products addressable on 32/64-bit hosts.
        let huge = DeltaShape {
            key_heads: 1,
            value_heads: 1,
            head_dim: 1,
            ..huge
        };
        assert!(huge.state_bytes()? > 0);
        assert!(DeltaState::new(huge, 0)
            .unwrap_err()
            .to_string()
            .contains("budget is 0"));
        Ok(())
    }

    #[test]
    fn hand_computed_delta_uses_decayed_state_and_updated_readout() -> Result<()> {
        // Rows are values, columns are keys. M'=0.5M, k=[0,1], beta=0.5.
        // M'k=[1,2], delta=[2,3], so M_new=[[0.5,3],[1.5,5]].
        let mut next = [0.0; 4];
        let mut output = [0.0; 2];
        recurrent_head(
            &[1.0, 2.0, 3.0, 4.0],
            &mut next,
            &[0.6, 0.8],
            &[0.0, 1.0],
            &[5.0, 8.0],
            0.5,
            0.5,
            &mut output,
        )?;
        assert_eq!(next, [0.5, 3.0, 1.5, 5.0]);
        close(output[0], 2.7 / 2.0f64.sqrt());
        close(output[1], 4.9 / 2.0f64.sqrt());
        Ok(())
    }

    #[test]
    fn convolution_keeps_raw_history_and_oldest_to_current_tap_order() -> Result<()> {
        let shape = DeltaShape {
            conv_kernel: 4,
            ..shape()
        };
        let g = shape.geometry()?;
        let mut history = vec![0.0; g.conv_state];
        let weights = [1.0, 2.0, 3.0, 4.0].repeat(3);
        for (input, expected) in [(1.0, 4.0), (10.0, 43.0), (100.0, 432.0), (1000.0, 4321.0)] {
            let (next, output) = convolve(
                shape,
                g,
                &history,
                &[input, 2.0 * input, 3.0 * input],
                &weights,
            )?;
            assert_eq!(output, [expected, 2.0 * expected, 3.0 * expected]);
            history = next;
        }
        assert_eq!(
            history,
            [10.0, 100.0, 1000.0, 20.0, 200.0, 2000.0, 30.0, 300.0, 3000.0]
        );
        Ok(())
    }

    #[test]
    fn normalization_uses_sum_epsilon_and_gates_after_rms() -> Result<()> {
        let mut q = [3.0, 4.0];
        l2_normalize(&mut q, 11.0)?;
        close(q[0], 0.5);
        close(q[1], 2.0 / 3.0);
        let mut output = [3.0, 4.0];
        rms_gate(&mut output, &[0.0, 1.0], &[2.0, 3.0], 3.5)?;
        assert_eq!(output[0], 0.0);
        close(output[1], 3.0 / (1.0 + (-1.0f64).exp()));
        Ok(())
    }

    #[test]
    fn projected_tokens_match_independent_scalar_equations_and_tiled_heads() -> Result<()> {
        let shape = DeltaShape {
            key_heads: 2,
            value_heads: 4,
            ..shape()
        };
        let mut state = DeltaState::new(shape, budget(shape)?)?;
        let conv = [1.0; 8];
        let a = [-0.3, -0.5, -0.7, -0.9];
        let dt = [0.0, 0.1, -0.2, 0.3];
        let alpha = [-1.0, 0.0, 1.0, 2.0];
        let beta = [-2.0, -1.0, 0.0, 1.0];
        let z = [1.0, 2.0, 3.0, 4.0];
        let weights = DeltaWeights {
            conv: &conv,
            a: &a,
            dt: &dt,
            norm: &[1.25],
        };
        let mut reference = [0.0f64; 4];
        // An independent D=1 formula, without any implementation helpers or matrix loops.
        let swish = |x: f64| x / (1.0 + (-x).exp());
        let eps = f64::from(shape.epsilon);
        for t in 1..=3 {
            let qkv = [1.0, 2.0, 0.5, -0.75, 0.3 * t as f32, 0.6, 0.9, 1.2];
            let output = state.step(
                DeltaInput {
                    qkv: &qkv,
                    z: &z,
                    alpha: &alpha,
                    beta: &beta,
                },
                weights,
            )?;
            for h in 0..4 {
                let raw_q = swish(f64::from(qkv[h % 2]));
                let raw_k = swish(f64::from(qkv[2 + h % 2]));
                let q = raw_q / (raw_q * raw_q + eps).sqrt();
                let k = raw_k / (raw_k * raw_k + eps).sqrt();
                let v = swish(f64::from(qkv[4 + h]));
                let rho = (f64::from(a[h])
                    * (1.0 + (f64::from(alpha[h]) + f64::from(dt[h])).exp()).ln())
                .exp();
                let b = 1.0 / (1.0 + (-f64::from(beta[h])).exp());
                let decayed = rho * reference[h];
                reference[h] = decayed + k * b * (v - decayed * k);
                let core = reference[h] * q;
                let expected = core / (core * core + eps).sqrt() * 1.25 * swish(f64::from(z[h]));
                close(state.recurrent_state()[h], reference[h]);
                close(output[h], expected);
            }
        }
        assert_eq!(state.tokens_seen(), 3);
        Ok(())
    }

    #[test]
    fn reset_reuses_allocation_without_carrying_another_sequence() -> Result<()> {
        let shape = DeltaShape {
            conv_kernel: 2,
            ..shape()
        };
        let mut state = DeltaState::new(shape, budget(shape)?)?;
        let input = DeltaInput {
            qkv: &[1.0; 3],
            z: &[1.0],
            alpha: &[0.0],
            beta: &[0.0],
        };
        let weights = DeltaWeights {
            conv: &[1.0; 6],
            a: &[-1.0],
            dt: &[0.0],
            norm: &[1.0],
        };
        let first = state.step(input, weights)?;
        let second = state.step(input, weights)?;
        assert_ne!(first, second);
        let conv_ptr = state.conv.as_ptr();
        let recurrent_ptr = state.recurrent.as_ptr();
        state.reset();
        assert_eq!(state.conv.as_ptr(), conv_ptr);
        assert_eq!(state.recurrent.as_ptr(), recurrent_ptr);
        assert_eq!(state.tokens_seen(), 0);
        assert!(state
            .conv
            .iter()
            .chain(&state.recurrent)
            .all(|value| *value == 0.0));
        assert_eq!(state.step(input, weights)?, first);
        Ok(())
    }

    #[test]
    fn invalid_tokens_and_late_overflow_do_not_commit_either_history() -> Result<()> {
        let shape = DeltaShape {
            conv_kernel: 2,
            ..shape()
        };
        let mut state = DeltaState::new(shape, budget(shape)?)?;
        let input = DeltaInput {
            qkv: &[1.0; 3],
            z: &[1.0],
            alpha: &[0.0],
            beta: &[0.0],
        };
        let weights = DeltaWeights {
            conv: &[1.0; 6],
            a: &[-1.0],
            dt: &[0.0],
            norm: &[1.0],
        };
        state.step(input, weights)?;
        let saved_conv = state.conv.clone();
        let saved_recurrent = state.recurrent.clone();
        for (bad_input, bad_weights) in [
            (
                DeltaInput {
                    qkv: &[f32::NAN, 0.0, 0.0],
                    ..input
                },
                weights,
            ),
            (DeltaInput { z: &[], ..input }, weights),
            (
                input,
                DeltaWeights {
                    a: &[1.0],
                    ..weights
                },
            ),
            (
                DeltaInput {
                    alpha: &[f32::MAX],
                    ..input
                },
                DeltaWeights {
                    dt: &[f32::MAX],
                    ..weights
                },
            ),
            (
                DeltaInput {
                    z: &[f32::MAX],
                    ..input
                },
                DeltaWeights {
                    norm: &[f32::MAX],
                    ..weights
                },
            ),
        ] {
            assert!(state.step(bad_input, bad_weights).is_err());
            assert_eq!(state.tokens_seen(), 1);
            assert_eq!(state.conv, saved_conv);
            assert_eq!(state.recurrent, saved_recurrent);
        }
        // The rejected token has no effect on a later valid continuation.
        assert!(state
            .step(input, weights)?
            .iter()
            .all(|value| value.is_finite()));
        assert_eq!(state.tokens_seen(), 2);
        Ok(())
    }

    #[test]
    fn dimensions_and_extreme_gate_inputs_are_checked_without_overflow() -> Result<()> {
        for bad in [
            DeltaShape {
                key_heads: 0,
                ..shape()
            },
            DeltaShape {
                key_heads: 2,
                value_heads: 3,
                ..shape()
            },
            DeltaShape {
                conv_kernel: 0,
                ..shape()
            },
            DeltaShape {
                head_dim: usize::MAX,
                ..shape()
            },
            DeltaShape {
                epsilon: f32::NAN,
                ..shape()
            },
            DeltaShape {
                epsilon: 0.0,
                ..shape()
            },
        ] {
            assert!(bad.state_bytes().is_err());
            assert!(bad.step_scratch_bytes().is_err());
            assert!(DeltaState::new(bad, usize::MAX).is_err());
        }
        assert_eq!(sigmoid(f32::MAX), 1.0);
        assert_eq!(sigmoid(-f32::MAX), 0.0);
        assert_eq!(softplus(f32::MAX), f32::MAX);
        assert_eq!(softplus(-f32::MAX), 0.0);
        close(softplus(0.0), 2.0f64.ln());
        let mut zero = [0.0; 2];
        l2_normalize(&mut zero, 1e-6)?;
        assert_eq!(zero, [0.0; 2]);
        Ok(())
    }
}
