//! Text-only, correctness-first gated full attention for `qwen35`.
//!
//! The GGUF query projection interleaves `[query, gate]` within each head. Query
//! and key RMS normalization precede partial split-half RoPE; sigmoid gating is
//! applied after attention. Text supplies identical positions to every IMRoPE
//! section, so its rotated prefix is ordinary split-half RoPE. This module does
//! not claim support for image/video position grids.

use anyhow::{ensure, Context, Result};
use beholdd_backend::{Device, IncrementalAttentionConfig, RopeLayout};

/// Shape and explicit cache limit for one full-attention layer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionShape {
    /// Number of query heads.
    pub query_heads: usize,
    /// Number of key/value heads; divides `query_heads`.
    pub kv_heads: usize,
    /// Equal query, key and value head width.
    pub head_dim: usize,
    /// Leading dimensions rotated in each query/key head.
    pub rope_dim: usize,
    /// Rotary frequency base.
    pub rope_base: f64,
    /// RMS normalization epsilon.
    pub epsilon: f32,
    /// Maximum evaluated text positions, fixed before cache allocation.
    pub capacity: usize,
}

impl AttentionShape {
    fn widths(self) -> Result<(usize, usize)> {
        ensure!(
            self.query_heads > 0 && self.kv_heads > 0 && self.head_dim > 0,
            "qwen35 attention dimensions must be positive"
        );
        ensure!(
            self.query_heads.is_multiple_of(self.kv_heads),
            "qwen35 attention head groups do not divide"
        );
        ensure!(
            self.rope_dim > 0 && self.rope_dim <= self.head_dim && self.rope_dim.is_multiple_of(2),
            "qwen35 rotary width must be positive, even and no wider than its head"
        );
        ensure!(
            self.rope_base.is_finite() && self.rope_base > 0.0,
            "qwen35 rotary base must be finite and positive"
        );
        ensure!(
            self.epsilon.is_finite() && self.epsilon > 0.0,
            "qwen35 attention epsilon must be finite and positive"
        );
        ensure!(
            self.capacity > 0,
            "qwen35 attention capacity must be positive"
        );
        Ok((
            self.query_heads
                .checked_mul(self.head_dim)
                .context("query width overflow")?,
            self.kv_heads
                .checked_mul(self.head_dim)
                .context("key/value width overflow")?,
        ))
    }

    /// Exact retained payload bytes, excluding allocator and per-step scratch.
    pub fn cache_bytes(self) -> Result<usize> {
        let (_, kv) = self.widths()?;
        self.capacity
            .checked_mul(kv)
            .and_then(|x| x.checked_mul(8))
            .context("qwen35 attention cache byte count overflow")
    }
}

/// Private per-layer KV state. An unsuccessful step does not advance the cache.
#[derive(Debug, Clone, PartialEq)]
pub struct AttentionState {
    shape: AttentionShape,
    keys: Vec<f32>,
    values: Vec<f32>,
    positions: usize,
}

fn zeroes(len: usize) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .context("allocating qwen35 attention storage")?;
    out.resize(len, 0.0);
    Ok(out)
}

fn finite(values: &[f32], label: &str) -> Result<()> {
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "qwen35 {label} contains a non-finite value"
    );
    Ok(())
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

impl AttentionState {
    /// Reserve a fixed cache only after the caller's byte budget admits it.
    pub fn new(shape: AttentionShape, cache_budget: usize) -> Result<Self> {
        let bytes = shape.cache_bytes()?;
        ensure!(
            bytes <= cache_budget,
            "qwen35 attention cache needs {bytes} bytes, budget is {cache_budget}"
        );
        let (_, kv) = shape.widths()?;
        let len = shape
            .capacity
            .checked_mul(kv)
            .context("cache element count overflow")?;
        Ok(Self {
            shape,
            keys: zeroes(len)?,
            values: zeroes(len)?,
            positions: 0,
        })
    }

    /// Number of successfully committed positions.
    pub fn positions(&self) -> usize {
        self.positions
    }

    /// Clear prior text, retaining the admitted allocation for the next request.
    pub fn reset(&mut self) {
        self.keys.fill(0.0);
        self.values.fill(0.0);
        self.positions = 0;
    }

    /// Evaluate projected Q/G, K and V for one token.
    ///
    /// `query_gate` is `[head][query coordinates, gate coordinates]`; K/V are
    /// `[kv head][coordinate]`. Norm vectors are shared across heads. The returned
    /// gated attention vector precedes the model's output projection/residual.
    pub fn step(
        &mut self,
        device: &dyn Device,
        query_gate: &[f32],
        key: &[f32],
        value: &[f32],
        query_norm: &[f32],
        key_norm: &[f32],
    ) -> Result<Vec<f32>> {
        let s = self.shape;
        let (q_width, kv_width) = s.widths()?;
        ensure!(
            self.positions < s.capacity,
            "qwen35 attention context capacity exhausted"
        );
        ensure!(
            query_gate.len()
                == q_width
                    .checked_mul(2)
                    .context("query/gate width overflow")?
                && key.len() == kv_width
                && value.len() == kv_width
                && query_norm.len() == s.head_dim
                && key_norm.len() == s.head_dim,
            "qwen35 attention projection or normalization shape mismatch"
        );
        for (values, label) in [
            (query_gate, "query/gate"),
            (key, "key"),
            (value, "value"),
            (query_norm, "query norm"),
            (key_norm, "key norm"),
        ] {
            finite(values, label)?;
        }
        let mut q = zeroes(q_width)?;
        let mut k = zeroes(kv_width)?;
        let mut cos = zeroes(s.rope_dim / 2)?;
        let mut sin = zeroes(s.rope_dim / 2)?;
        for i in 0..cos.len() {
            let theta =
                self.positions as f64 * s.rope_base.powf(-2.0 * i as f64 / s.rope_dim as f64);
            cos[i] = theta.cos() as f32;
            sin[i] = theta.sin() as f32;
        }
        for head in 0..s.query_heads {
            let start = head * s.head_dim;
            device.rmsnorm(
                &mut q[start..start + s.head_dim],
                &query_gate[2 * start..2 * start + s.head_dim],
                Some(query_norm),
                s.epsilon,
            )?;
            device.rope(
                &mut q[start..start + s.rope_dim],
                &cos,
                &sin,
                RopeLayout::SplitHalf,
                false,
            )?;
        }
        for head in 0..s.kv_heads {
            let start = head * s.head_dim;
            device.rmsnorm(
                &mut k[start..start + s.head_dim],
                &key[start..start + s.head_dim],
                Some(key_norm),
                s.epsilon,
            )?;
            device.rope(
                &mut k[start..start + s.rope_dim],
                &cos,
                &sin,
                RopeLayout::SplitHalf,
                false,
            )?;
        }
        finite(&q, "normalized query")?;
        finite(&k, "normalized key")?;
        let start = self.positions * kv_width;
        let end = start + kv_width;
        self.keys[start..end].copy_from_slice(&k);
        self.values[start..end].copy_from_slice(value);
        let result = (|| {
            let mut out = zeroes(q_width)?;
            device.incremental_causal_attention(
                &mut out,
                &q,
                &self.keys[..end],
                &self.values[..end],
                IncrementalAttentionConfig {
                    n_query_positions: 1,
                    query_start: self.positions,
                    n_kv_positions: self.positions + 1,
                    n_heads: s.query_heads,
                    n_kv_heads: s.kv_heads,
                    head_dim: s.head_dim,
                    scale: 1.0 / (s.head_dim as f32).sqrt(),
                },
            )?;
            for head in 0..s.query_heads {
                for i in 0..s.head_dim {
                    out[head * s.head_dim + i] *=
                        sigmoid(query_gate[head * 2 * s.head_dim + s.head_dim + i]);
                }
            }
            finite(&out, "gated attention output")?;
            Ok(out)
        })();
        if result.is_ok() {
            self.positions += 1;
        } else {
            self.keys[start..end].fill(0.0);
            self.values[start..end].fill(0.0);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beholdd_backend::cpu::Cpu;
    use beholdd_backend::{AttentionConfig, DeviceError, Weight};
    use beholdd_format::GgmlType;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn shape() -> AttentionShape {
        AttentionShape {
            query_heads: 2,
            kv_heads: 1,
            head_dim: 4,
            rope_dim: 2,
            rope_base: 10_000_000.0,
            epsilon: 1e-6,
            capacity: 2,
        }
    }
    fn close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 2e-5, "{actual:?} != {expected:?}");
        }
    }
    #[test]
    fn interleaved_gates_and_grouped_values_match_two_token_golden() -> Result<()> {
        let mut state = AttentionState::new(shape(), 64)?;
        let qg = [
            1.,
            2.,
            3.,
            4.,
            0.,
            0.,
            0.,
            0.,
            5.,
            6.,
            7.,
            8.,
            3f32.ln(),
            3f32.ln(),
            3f32.ln(),
            3f32.ln(),
        ];
        close(
            &state.step(&Cpu, &qg, &[0.; 4], &[2., 4., 6., 8.], &[1.; 4], &[1.; 4])?,
            &[1., 2., 3., 4., 1.5, 3., 4.5, 6.],
        );
        close(
            &state.step(&Cpu, &qg, &[0.; 4], &[4., 6., 8., 10.], &[1.; 4], &[1.; 4])?,
            &[1.5, 2.5, 3.5, 4.5, 2.25, 3.75, 5.25, 6.75],
        );
        assert_eq!(state.positions(), 2);
        Ok(())
    }
    #[test]
    fn refusals_do_not_advance_or_corrupt_cache_and_reset_replays() -> Result<()> {
        let mut state = AttentionState::new(shape(), 64)?;
        let qg = [0.; 16];
        let first = state.step(&Cpu, &qg, &[0.; 4], &[2.; 4], &[1.; 4], &[1.; 4])?;
        let before = state.clone();
        assert!(state
            .step(&Cpu, &qg, &[0.; 4], &[f32::NAN; 4], &[1.; 4], &[1.; 4])
            .is_err());
        assert_eq!(state, before);
        assert!(state
            .step(&Cpu, &[0.; 15], &[0.; 4], &[2.; 4], &[1.; 4], &[1.; 4])
            .is_err());
        assert_eq!(state, before);
        state.step(&Cpu, &qg, &[0.; 4], &[2.; 4], &[1.; 4], &[1.; 4])?;
        let full = state.clone();
        assert!(state
            .step(&Cpu, &qg, &[0.; 4], &[2.; 4], &[1.; 4], &[1.; 4])
            .is_err());
        assert_eq!(state, full);
        state.reset();
        assert_eq!(
            first,
            state.step(&Cpu, &qg, &[0.; 4], &[2.; 4], &[1.; 4], &[1.; 4])?
        );
        Ok(())
    }
    #[test]
    fn allocation_and_shape_are_checked_before_storage() {
        assert!(AttentionState::new(shape(), 63).is_err());
        for changed in [
            AttentionShape {
                kv_heads: 0,
                ..shape()
            },
            AttentionShape {
                rope_dim: 3,
                ..shape()
            },
            AttentionShape {
                capacity: usize::MAX,
                ..shape()
            },
            AttentionShape {
                epsilon: f32::NAN,
                ..shape()
            },
        ] {
            assert!(AttentionState::new(changed, usize::MAX).is_err());
        }
    }

    fn nonzero_shape() -> AttentionShape {
        AttentionShape {
            query_heads: 4,
            kv_heads: 2,
            head_dim: 6,
            rope_dim: 4,
            rope_base: 100.0,
            epsilon: 0.03125,
            capacity: 3,
        }
    }

    const QUERY_NORM: [f32; 6] = [1.0, 0.5, 1.5, -0.75, 1.25, 0.875];
    const KEY_NORM: [f32; 6] = [0.75, 1.25, -0.5, 1.5, 0.625, 1.0];

    fn nonzero_input(token: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut query_gate = Vec::new();
        for head in 0..4 {
            query_gate.extend(
                (0..6).map(|i| (((token + 1) * (head + 2) * (i + 1) % 13) as f32 - 6.0) / 4.0),
            );
            query_gate.extend((0..6).map(|i| ((token + 2 * head + i) % 7) as f32 / 2.0 - 1.5));
        }
        let key = (0..2)
            .flat_map(|head| {
                (0..6).map(move |i| ((2 * token + 3 * head + 2 * i) % 11) as f32 / 4.0 - 1.25)
            })
            .collect();
        let value = (0..2)
            .flat_map(|head| {
                (0..6).map(move |i| (((token + 1) * (head + 1) * (i + 2) % 17) as f32 - 8.0) / 4.0)
            })
            .collect();
        (query_gate, key, value)
    }

    #[test]
    fn nonzero_three_token_golden_covers_partial_rope_norms_scale_and_kv_groups() -> Result<()> {
        // Independently calculated with scalar Python/math binary64 equations, not
        // Device or this implementation. Normalize all six coordinates; rotate
        // pairs (0,2)/(1,3) by t/t*0.1; head h uses KV head floor(h/2), and
        // softmax(dot(Q,K)/sqrt(6)) precedes the per-coordinate sigmoid gate.
        // Unequal signed norms, distinct KV groups and nonuniform values make
        // omitted/wrong normalization, frequency, suffix rotation and grouping visible.
        let expected: [[f32; 24]; 3] = [
            [
                -0.273638286,
                -0.336176777,
                -0.377540669,
                -0.375000000,
                -0.311229666,
                -0.182764645,
                -0.566311003,
                -0.625000000,
                -0.622459331,
                -0.548293934,
                -0.408787238,
                -0.045606381,
                -0.622459331,
                -0.365529289,
                0.000000000,
                0.091212762,
                0.268941421,
                0.566311003,
                -0.817574476,
                -0.091212762,
                0.000000000,
                0.188770334,
                0.500000000,
                0.933688997,
            ],
            [
                -0.387713923,
                -0.438870055,
                -0.441629633,
                -0.376011549,
                -0.237512816,
                -0.037366191,
                -0.686210359,
                -0.658954793,
                -0.544522723,
                -0.352416947,
                -0.021391610,
                0.052854381,
                -0.508246259,
                -0.035016235,
                0.111199445,
                -0.008973415,
                0.233706928,
                0.635707353,
                -0.128366273,
                -0.014925172,
                0.223757785,
                -0.009294002,
                0.391887944,
                0.934108635,
            ],
            [
                -0.413677656,
                -0.321787606,
                -0.119159640,
                0.190592806,
                -0.534931831,
                -0.036877258,
                -0.520286145,
                -0.185530580,
                0.268400472,
                0.166073062,
                -0.157313162,
                -0.001080708,
                0.507158923,
                -0.214355190,
                0.036342642,
                0.290686742,
                -0.512700859,
                0.177250878,
                0.082201846,
                -0.112848428,
                0.426960392,
                -0.017122845,
                -0.462074375,
                0.425762355,
            ],
        ];
        let s = nonzero_shape();
        let mut state = AttentionState::new(s, s.cache_bytes()?)?;
        for (token, golden) in expected.iter().enumerate() {
            let (qg, k, v) = nonzero_input(token);
            close(
                &state.step(&Cpu, &qg, &k, &v, &QUERY_NORM, &KEY_NORM)?,
                golden,
            );
            assert_eq!(state.positions(), token + 1);
        }
        // The final two coordinates in each head retain normalized, unrotated
        // values; checking stored keys also catches matching Q/K rotary mistakes.
        close(
            &state.keys[24..],
            &[
                0.534488313,
                -0.084487379,
                -0.018417165,
                2.423241459,
                -0.797241005,
                -0.637792804,
                -0.923990788,
                1.847795734,
                0.103188194,
                -1.089654001,
                -0.199310251,
                0.318896402,
            ],
        );
        Ok(())
    }

    #[derive(Clone, Copy, Debug)]
    enum AttentionFault {
        Error,
        Nan,
        Infinity,
    }

    struct FaultyAttention {
        fault: AttentionFault,
        calls: AtomicUsize,
    }

    // Preserve the real CPU transformations, injecting failure only after the
    // attention call has received the newly appended, nonzero KV cache entry.
    impl Device for FaultyAttention {
        fn name(&self) -> &'static str {
            "qwen35-attention-failure-fixture"
        }
        fn supports(&self, kind: GgmlType) -> bool {
            Cpu.supports(kind)
        }
        fn dequantize(&self, out: &mut [f32], weight: &Weight<'_>) -> Result<(), DeviceError> {
            Cpu.dequantize(out, weight)
        }
        fn matvec(&self, y: &mut [f32], x: &[f32], weight: &Weight<'_>) -> Result<(), DeviceError> {
            Cpu.matvec(y, x, weight)
        }
        fn matvec_add(
            &self,
            y: &mut [f32],
            x: &[f32],
            weight: &Weight<'_>,
            scale: f32,
        ) -> Result<(), DeviceError> {
            Cpu.matvec_add(y, x, weight, scale)
        }
        fn rmsnorm(
            &self,
            y: &mut [f32],
            x: &[f32],
            weight: Option<&[f32]>,
            eps: f32,
        ) -> Result<(), DeviceError> {
            Cpu.rmsnorm(y, x, weight, eps)
        }
        fn softmax(&self, values: &mut [f32]) -> Result<(), DeviceError> {
            Cpu.softmax(values)
        }
        fn causal_attention(
            &self,
            out: &mut [f32],
            q: &[f32],
            k: &[f32],
            v: &[f32],
            config: AttentionConfig,
        ) -> Result<(), DeviceError> {
            Cpu.causal_attention(out, q, k, v, config)
        }
        fn incremental_causal_attention(
            &self,
            out: &mut [f32],
            q: &[f32],
            k: &[f32],
            v: &[f32],
            config: IncrementalAttentionConfig,
        ) -> Result<(), DeviceError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            assert_eq!(config.query_start, 1);
            assert_eq!(config.n_kv_positions, 2);
            assert!(k[12..].iter().any(|x| *x != 0.0));
            assert!(v[12..].iter().any(|x| *x != 0.0));
            Cpu.incremental_causal_attention(out, q, k, v, config)?;
            match self.fault {
                AttentionFault::Error => Err(DeviceError::Backend {
                    device: self.name(),
                    message: "injected attention failure".into(),
                }),
                AttentionFault::Nan => {
                    out[0] = f32::NAN;
                    Ok(())
                }
                AttentionFault::Infinity => {
                    out[0] = f32::INFINITY;
                    Ok(())
                }
            }
        }
        fn add_in_place(&self, output: &mut [f32], residual: &[f32]) -> Result<(), DeviceError> {
            Cpu.add_in_place(output, residual)
        }
        fn swiglu(&self, gate: &mut [f32], up: &[f32], limit: f32) -> Result<(), DeviceError> {
            Cpu.swiglu(gate, up, limit)
        }
        fn rope(
            &self,
            x: &mut [f32],
            cos: &[f32],
            sin: &[f32],
            layout: RopeLayout,
            inverse: bool,
        ) -> Result<(), DeviceError> {
            Cpu.rope(x, cos, sin, layout, inverse)
        }
    }

    #[test]
    fn downstream_error_or_nonfinite_output_rolls_back_written_cache_and_retries() -> Result<()> {
        let s = nonzero_shape();
        let mut before = AttentionState::new(s, s.cache_bytes()?)?;
        let (q0, k0, v0) = nonzero_input(0);
        before.step(&Cpu, &q0, &k0, &v0, &QUERY_NORM, &KEY_NORM)?;
        let (q1, k1, v1) = nonzero_input(1);
        let mut expected_state = before.clone();
        let expected = expected_state.step(&Cpu, &q1, &k1, &v1, &QUERY_NORM, &KEY_NORM)?;
        for fault in [
            AttentionFault::Error,
            AttentionFault::Nan,
            AttentionFault::Infinity,
        ] {
            let device = FaultyAttention {
                fault,
                calls: AtomicUsize::new(0),
            };
            let mut state = before.clone();
            let error = state
                .step(&device, &q1, &k1, &v1, &QUERY_NORM, &KEY_NORM)
                .unwrap_err();
            assert_eq!(device.calls.load(Ordering::Relaxed), 1);
            match fault {
                AttentionFault::Error => {
                    assert!(error.to_string().contains("injected attention failure"))
                }
                _ => assert!(error
                    .to_string()
                    .contains("gated attention output contains a non-finite value")),
            }
            assert_eq!(
                state, before,
                "failed {fault:?} must restore all cache bytes and position"
            );
            assert_eq!(
                state.step(&Cpu, &q1, &k1, &v1, &QUERY_NORM, &KEY_NORM)?,
                expected
            );
            assert_eq!(state, expected_state);
        }
        Ok(())
    }

    #[test]
    fn extreme_finite_gates_remain_finite_and_preserve_sigmoid_limits() -> Result<()> {
        let mut state = AttentionState::new(shape(), 64)?;
        let qg = [
            1.,
            2.,
            3.,
            4.,
            f32::MAX,
            -f32::MAX,
            100.,
            -100.,
            5.,
            6.,
            7.,
            8.,
            -f32::MAX,
            f32::MAX,
            -100.,
            100.,
        ];
        let out = state.step(
            &Cpu,
            &qg,
            &[1., -2., 3., -4.],
            &[2., -4., 6., -8.],
            &[1.; 4],
            &[1.; 4],
        )?;
        assert!(out.iter().all(|x| x.is_finite()));
        assert_eq!(
            [out[0], out[1], out[2], out[4], out[5], out[7]],
            [2., 0., 6., 0., -4., -8.]
        );
        assert!(out[3].abs() < 1e-40 && out[6].abs() < 1e-40);
        assert_eq!(state.positions(), 1);
        Ok(())
    }
}
