//! Bounded CPU-only reference execution for the dense `qwen35` hybrid graph.
//!
//! This is an explicit development interface, not a production serving backend.
//! A failed token poisons its sequence until reset, because earlier blocks may
//! already have committed state. No partially evaluated output is published.

use anyhow::{ensure, Context, Result};
use beholdd_backend::{cpu::Cpu, Device};
use beholdd_memory::WeightSource;

use super::attention::{AttentionShape, AttentionState};
use super::config::{Attention, Binding, Config};
use super::delta::{DeltaInput, DeltaShape, DeltaState, DeltaWeights};
use super::weights::{self, ReadLimits};

#[derive(Debug)]
enum State {
    Full(AttentionState),
    Delta(DeltaState),
}

fn delta_shape(c: &Config) -> DeltaShape {
    DeltaShape {
        key_heads: c.ssm_group_count,
        value_heads: c.ssm_time_step_rank,
        head_dim: c.ssm_state_size,
        conv_kernel: c.ssm_conv_kernel,
        epsilon: c.eps,
    }
}

fn attention_shape(c: &Config, capacity: usize) -> AttentionShape {
    AttentionShape {
        query_heads: c.n_heads,
        kv_heads: c.n_kv_heads,
        head_dim: c.key_head_dim,
        rope_dim: c.rope_dim,
        rope_base: c.rope_base,
        epsilon: c.eps,
        capacity,
    }
}

fn checked_sum(values: &[usize]) -> Result<usize> {
    values
        .iter()
        .try_fold(0usize, |n, x| n.checked_add(*x))
        .context("qwen35 reference memory forecast overflow")
}

fn checked_product(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .context("qwen35 reference memory forecast overflow")
}

fn floats(counts: &[usize]) -> Result<usize> {
    checked_product(checked_sum(counts)?, std::mem::size_of::<f32>())
}

// Count the whole block's named vectors even when some stages do not overlap.
// In particular, none of these allocations are bounded by ReadLimits: that
// contract belongs to the weight reader, not to attention or the residual path.
fn common_block_scratch(c: &Config) -> Result<usize> {
    // Caller input, attention norm, normalized input, attended residual,
    // copied residual/output, post norm, down projection, and FFN gate/up.
    floats(&[
        checked_product(7, c.dim)?,
        checked_product(2, c.feed_forward)?,
    ])
}

fn full_scratch(c: &Config, capacity: usize) -> Result<usize> {
    let q = checked_product(c.n_heads, c.key_head_dim)?;
    let kv = checked_product(c.n_kv_heads, c.key_head_dim)?;
    // Projected Q/G, K, V, both head norms, normalized Q/K, RoPE cos/sin,
    // returned attention vector, and Cpu's one shared f32 score vector. Scores
    // grow with context, not decoded_bytes; heads reuse the same score buffer.
    floats(&[
        checked_product(4, q)?,
        checked_product(3, kv)?,
        checked_product(2, c.key_head_dim)?,
        c.rope_dim,
        capacity,
    ])
}

fn delta_scratch(c: &Config) -> Result<usize> {
    let key = checked_product(c.ssm_group_count, c.ssm_state_size)?;
    let qkv = checked_sum(&[checked_product(2, key)?, c.ssm_inner_size])?;
    // QKV/Z, alpha/beta, convolution weights, A/dt, and gated norm. DeltaState
    // separately counts next convolution/recurrent state, convolved QKV and its
    // returned output while the old persistent state is still retained.
    checked_sum(&[
        floats(&[
            qkv,
            c.ssm_inner_size,
            checked_product(4, c.ssm_time_step_rank)?,
            checked_product(qkv, c.ssm_conv_kernel)?,
            c.ssm_state_size,
        ])?,
        delta_shape(c).step_scratch_bytes()?,
    ])
}

/// Accounted CPU payload, separate from checkpoint metadata, allocator/thread
/// overhead and the operating system's reclaimable file cache. Includes the
/// current hidden input/output, but not previous results retained by the caller.
/// A caching/prefetching WeightSource must budget its retained weights separately;
/// this forecast covers one direct range read, not arbitrary source residency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryForecast {
    /// Persistent KV and recurrent/convolution payload.
    pub state_bytes: usize,
    /// Conservative per-token vectors, encoded tile and decoded CPU scratch.
    pub scratch_bytes: usize,
    /// Sum admitted before any state allocation or weight read.
    pub payload_bytes: usize,
}

fn forecast(
    binding: &Binding,
    layers: impl Iterator<Item = usize>,
    capacity: usize,
    limits: ReadLimits,
    include_head: bool,
) -> Result<MemoryForecast> {
    let c = binding.config();
    ensure!(
        capacity > 0 && capacity <= c.context,
        "qwen35 requested capacity is outside checkpoint context"
    );
    ensure!(
        limits.encoded_bytes > 0 && limits.decoded_bytes > 0 && limits.tile_rows > 0,
        "qwen35 reference read limits must be positive"
    );
    let mut state_bytes = 0usize;
    let mut block_scratch = 0;
    let common = common_block_scratch(c)?;
    for layer in layers {
        let block = binding
            .blocks()
            .get(layer)
            .context("qwen35 layer index out of range")?;
        let (bytes, branch_scratch) = match block.attention {
            Attention::Full(_) => (
                attention_shape(c, capacity).cache_bytes()?,
                full_scratch(c, capacity)?,
            ),
            Attention::DeltaNet(_) => {
                let d = delta_shape(c);
                (d.state_bytes()?, delta_scratch(c)?)
            }
        };
        state_bytes = checked_sum(&[state_bytes, bytes])?;
        block_scratch = block_scratch.max(checked_sum(&[common, branch_scratch])?);
    }
    // The final hidden vector, output norm and normalized hidden coexist with
    // the full vocabulary result. This also exceeds embedding-row storage.
    let vectors = if include_head {
        block_scratch.max(floats(&[checked_product(3, c.dim)?, c.vocab])?)
    } else {
        block_scratch
    };
    // A weight operation holds its packed read and Cpu's decoded matrix tile
    // concurrently with the vectors above. Results are already counted by shape.
    let scratch_bytes = checked_sum(&[vectors, limits.encoded_bytes, limits.decoded_bytes])?;
    let payload_bytes = checked_sum(&[state_bytes, scratch_bytes])?;
    Ok(MemoryForecast {
        state_bytes,
        scratch_bytes,
        payload_bytes,
    })
}

fn zeros(len: usize) -> Result<Vec<f32>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len)
        .context("allocating qwen35 reference activation")?;
    out.resize(len, 0.0);
    Ok(out)
}

fn finite(x: &[f32]) -> Result<()> {
    ensure!(
        x.iter().all(|v| v.is_finite()),
        "qwen35 reference activation is non-finite"
    );
    Ok(())
}

/// One checked block, useful for comparing real checkpoint activations with an
/// independent oracle before any complete-model or GPU execution claim.
pub struct BlockReference<'a> {
    binding: &'a Binding,
    layer: usize,
    capacity: usize,
    limits: ReadLimits,
    state: State,
    positions: usize,
    poisoned: bool,
}

impl<'a> BlockReference<'a> {
    /// Forecast a single block's conservative payload without allocating/reading.
    pub fn memory_forecast(
        binding: &Binding,
        layer: usize,
        capacity: usize,
        limits: ReadLimits,
    ) -> Result<MemoryForecast> {
        forecast(binding, std::iter::once(layer), capacity, limits, false)
    }

    /// Admit memory before constructing any per-sequence state.
    pub fn new(
        binding: &'a Binding,
        layer: usize,
        capacity: usize,
        payload_budget: usize,
        limits: ReadLimits,
    ) -> Result<Self> {
        let memory = Self::memory_forecast(binding, layer, capacity, limits)?;
        ensure!(
            memory.payload_bytes <= payload_budget,
            "qwen35 block payload needs {} bytes, budget is {payload_budget}",
            memory.payload_bytes
        );
        let c = binding.config();
        let state = match binding.blocks()[layer].attention {
            Attention::Full(_) => State::Full(AttentionState::new(
                attention_shape(c, capacity),
                memory.state_bytes,
            )?),
            Attention::DeltaNet(_) => {
                let shape = delta_shape(c);
                State::Delta(DeltaState::new(
                    shape,
                    checked_sum(&[shape.state_bytes()?, shape.step_scratch_bytes()?])?,
                )?)
            }
        };
        Ok(Self {
            binding,
            layer,
            capacity,
            limits,
            state,
            positions: 0,
            poisoned: false,
        })
    }

    /// Successfully committed input positions.
    pub fn positions(&self) -> usize {
        self.positions
    }
    /// Whether a failed operation requires an explicit sequence reset.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Clear all CPU state for an unrelated sequence after success or failure.
    pub fn reset(&mut self) {
        match &mut self.state {
            State::Full(s) => s.reset(),
            State::Delta(s) => s.reset(),
        }
        self.positions = 0;
        self.poisoned = false;
    }

    /// Evaluate a hidden input through attention and dense FFN, publishing only a
    /// complete block result. `control` is called before each bounded weight tile.
    pub fn step(
        &mut self,
        source: &dyn WeightSource,
        input: &[f32],
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<f32>> {
        ensure!(
            !self.poisoned,
            "qwen35 reference block is poisoned; reset its sequence"
        );
        ensure!(
            source.checkpoint_id() == self.binding.checkpoint_id(),
            "qwen35 reference source identity mismatch"
        );
        ensure!(
            self.positions < self.capacity,
            "qwen35 reference block context exhausted"
        );
        ensure!(
            input.len() == self.binding.config().dim,
            "qwen35 reference hidden width mismatch"
        );
        finite(input)?;
        let result = self.step_inner(source, input, control);
        if result.is_ok() {
            self.positions += 1;
        } else {
            self.poisoned = true;
        }
        result
    }

    fn step_inner(
        &mut self,
        source: &dyn WeightSource,
        input: &[f32],
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<f32>> {
        control()?;
        let c = self.binding.config();
        let block = &self.binding.blocks()[self.layer];
        let limits = self.limits;
        let norm = weights::f32_tensor(&block.attention_norm, source, limits)?;
        let mut h = zeros(c.dim)?;
        Cpu.rmsnorm(&mut h, input, Some(&norm), c.eps)?;
        finite(&h)?;
        let attended = match (&block.attention, &mut self.state) {
            (Attention::Full(w), State::Full(state)) => {
                let qg = weights::matvec(&w.query_gate, source, &Cpu, &h, limits, control)?;
                let k = weights::matvec(&w.key, source, &Cpu, &h, limits, control)?;
                let v = weights::matvec(&w.value, source, &Cpu, &h, limits, control)?;
                let qn = weights::f32_tensor(&w.query_norm, source, limits)?;
                let kn = weights::f32_tensor(&w.key_norm, source, limits)?;
                let out = state.step(&Cpu, &qg, &k, &v, &qn, &kn)?;
                weights::matvec(&w.output, source, &Cpu, &out, limits, control)?
            }
            (Attention::DeltaNet(w), State::Delta(state)) => {
                let qkv = weights::matvec(&w.qkv, source, &Cpu, &h, limits, control)?;
                let z = weights::matvec(&w.gate, source, &Cpu, &h, limits, control)?;
                let alpha = weights::matvec(&w.alpha, source, &Cpu, &h, limits, control)?;
                let beta = weights::matvec(&w.beta, source, &Cpu, &h, limits, control)?;
                let conv = weights::f32_tensor(&w.conv, source, limits)?;
                let a = weights::f32_tensor(&w.a, source, limits)?;
                let dt = weights::f32_tensor(&w.dt_bias, source, limits)?;
                let norm = weights::f32_tensor(&w.norm, source, limits)?;
                let out = state.step(
                    DeltaInput {
                        qkv: &qkv,
                        z: &z,
                        alpha: &alpha,
                        beta: &beta,
                    },
                    DeltaWeights {
                        conv: &conv,
                        a: &a,
                        dt: &dt,
                        norm: &norm,
                    },
                )?;
                weights::matvec(&w.output, source, &Cpu, &out, limits, control)?
            }
            _ => anyhow::bail!("qwen35 internal block state does not match its binding"),
        };
        let mut residual = zeros(input.len())?;
        residual.copy_from_slice(input);
        Cpu.add_in_place(&mut residual, &attended)?;
        finite(&residual)?;
        control()?;
        let post = weights::f32_tensor(&block.post_attention_norm, source, limits)?;
        Cpu.rmsnorm(&mut h, &residual, Some(&post), c.eps)?;
        finite(&h)?;
        let mut gate = weights::matvec(&block.ffn.gate, source, &Cpu, &h, limits, control)?;
        let up = weights::matvec(&block.ffn.up, source, &Cpu, &h, limits, control)?;
        Cpu.swiglu(&mut gate, &up, 0.0)?;
        finite(&gate)?;
        let down = weights::matvec(&block.ffn.down, source, &Cpu, &gate, limits, control)?;
        Cpu.add_in_place(&mut residual, &down)?;
        finite(&residual)?;
        Ok(residual)
    }
}

/// Explicit CPU model reference. Weight residency stays behind `WeightSource`;
/// the bounded direct-read source is recommended for real-checkpoint probes.
pub struct ModelReference<'a> {
    binding: &'a Binding,
    blocks: Vec<BlockReference<'a>>,
    limits: ReadLimits,
    positions: usize,
    capacity: usize,
    poisoned: bool,
}

impl<'a> ModelReference<'a> {
    /// Forecast aggregate retained state and one block's peak scratch.
    pub fn memory_forecast(
        binding: &Binding,
        capacity: usize,
        limits: ReadLimits,
    ) -> Result<MemoryForecast> {
        forecast(
            binding,
            0..binding.config().n_layers,
            capacity,
            limits,
            true,
        )
    }
    /// Construct only after the complete model's payload fits the explicit budget.
    pub fn new(
        binding: &'a Binding,
        capacity: usize,
        budget: usize,
        limits: ReadLimits,
    ) -> Result<Self> {
        let memory = Self::memory_forecast(binding, capacity, limits)?;
        ensure!(
            memory.payload_bytes <= budget,
            "qwen35 model payload needs {} bytes, budget is {budget}",
            memory.payload_bytes
        );
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(binding.config().n_layers)
            .context("allocating qwen35 reference blocks")?;
        for layer in 0..binding.config().n_layers {
            blocks.push(BlockReference::new(
                binding, layer, capacity, budget, limits,
            )?);
        }
        Ok(Self {
            binding,
            blocks,
            limits,
            positions: 0,
            capacity,
            poisoned: false,
        })
    }
    /// Successfully evaluated input tokens.
    pub fn positions(&self) -> usize {
        self.positions
    }
    /// Whether a partial token failed and requires explicit reset.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
    /// Clear every layer before an unrelated prompt or a retry after failure.
    pub fn reset(&mut self) {
        for b in &mut self.blocks {
            b.reset();
        }
        self.positions = 0;
        self.poisoned = false;
    }
    /// Evaluate one token and return the complete vocabulary logits. No sampling,
    /// chat rendering, fallback, or runtime capability advertisement occurs here.
    pub fn step(
        &mut self,
        source: &dyn WeightSource,
        token: u32,
        control: &mut dyn FnMut() -> Result<()>,
    ) -> Result<Vec<f32>> {
        ensure!(
            !self.poisoned,
            "qwen35 reference model is poisoned; reset its sequence"
        );
        ensure!(
            source.checkpoint_id() == self.binding.checkpoint_id(),
            "qwen35 reference source identity mismatch"
        );
        ensure!(
            self.positions < self.capacity,
            "qwen35 reference model context exhausted"
        );
        ensure!(
            (token as usize) < self.binding.config().vocab,
            "qwen35 token outside vocabulary"
        );
        let result = (|| {
            control()?;
            let mut hidden = weights::embedding_row(
                self.binding.embedding(),
                token as usize,
                source,
                &Cpu,
                self.limits,
            )?;
            for block in &mut self.blocks {
                hidden = block.step(source, &hidden, control)?;
            }
            let norm = weights::f32_tensor(self.binding.output_norm(), source, self.limits)?;
            let mut normalized = zeros(hidden.len())?;
            Cpu.rmsnorm(
                &mut normalized,
                &hidden,
                Some(&norm),
                self.binding.config().eps,
            )?;
            finite(&normalized)?;
            weights::matvec(
                self.binding.output(),
                source,
                &Cpu,
                &normalized,
                self.limits,
                control,
            )
        })();
        if result.is_ok() {
            self.positions += 1;
        } else {
            self.poisoned = true;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::super::config::tests::Fixture;
    use super::*;
    use beholdd_format::{CheckpointId, Gguf, WeightRange};
    use beholdd_memory::{ResidentWeight, WeightAccess};
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

    struct Source<'a> {
        gguf: &'a Gguf,
        reads: AtomicUsize,
        fail_start: AtomicU64,
    }

    impl<'a> Source<'a> {
        fn new(gguf: &'a Gguf) -> Self {
            Self {
                gguf,
                reads: AtomicUsize::new(0),
                fail_start: AtomicU64::new(u64::MAX),
            }
        }
    }

    impl WeightSource for Source<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.gguf.checkpoint_id()
        }
        fn load_range(&self, range: &WeightRange, _access: WeightAccess) -> Result<ResidentWeight> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            ensure!(
                range.start() != self.fail_start.load(Ordering::Relaxed),
                "injected reference read failure"
            );
            ResidentWeight::load_range(self.gguf, range)
        }
    }

    fn limits() -> ReadLimits {
        ReadLimits {
            encoded_bytes: 8192,
            decoded_bytes: 8192,
            tile_rows: 4,
        }
    }

    fn state_positions(block: &BlockReference<'_>) -> usize {
        match &block.state {
            State::Full(state) => state.positions(),
            State::Delta(state) => state.tokens_seen() as usize,
        }
    }

    #[test]
    fn forecasts_count_context_scores_and_shape_buffers_outside_weight_limits() -> Result<()> {
        let (_file, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        // Tiny read limits intentionally cannot execute these matrices. The
        // forecast must still include graph vectors, instead of shrinking them
        // to a multiple of the weight reader's unrelated scratch allowance.
        let tiny = ReadLimits {
            encoded_bytes: 1,
            decoded_bytes: 1,
            tile_rows: 1,
        };
        let full = BlockReference::memory_forecast(&binding, 3, 1, tiny)?;
        let longer = BlockReference::memory_forecast(&binding, 3, 16, tiny)?;
        assert_eq!(longer.state_bytes - full.state_bytes, 15 * 256 * 2 * 4);
        assert_eq!(longer.scratch_bytes - full.scratch_bytes, 15 * 4);
        // Full branch: Q/G + K/V + two norms + normalized Q/K + RoPE +
        // attention output + scores. Common path retains seven hidden vectors
        // and both FFN branches. Packed and decoded weight buffers are extra.
        let full_vectors = 1024 + 256 + 256 + 256 + 256 + 512 + 256 + 64 + 512 + 16;
        assert_eq!(
            longer.scratch_bytes,
            (full_vectors + 7 * 256 + 2 * 256) * 4 + 2
        );
        let delta = BlockReference::memory_forecast(&binding, 0, 16, tiny)?;
        assert_eq!(delta.state_bytes, (512 * 3 + 256 * 128) * 4);
        let delta_vectors = 512 + 256 + 2 + 2 + 2048 + 2 + 2 + 128;
        let staged_delta = 512 * 3 + 512 + 256 * 128 + 256;
        assert_eq!(
            delta.scratch_bytes,
            (delta_vectors + staged_delta + 7 * 256 + 2 * 256) * 4 + 2
        );
        let model = ModelReference::memory_forecast(&binding, 16, tiny)?;
        assert_eq!(
            model.state_bytes,
            3 * delta.state_bytes + longer.state_bytes
        );
        assert_eq!(
            model.scratch_bytes,
            delta.scratch_bytes.max(longer.scratch_bytes)
        );
        Ok(())
    }

    #[test]
    fn admission_refuses_invalid_or_insufficient_budgets_and_accepts_exact_budget() -> Result<()> {
        let (_file, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        for layer in [0, 3] {
            let memory = BlockReference::memory_forecast(&binding, layer, 1, limits())?;
            assert!(
                BlockReference::new(&binding, layer, 1, memory.payload_bytes - 1, limits())
                    .is_err()
            );
            let block = BlockReference::new(&binding, layer, 1, memory.payload_bytes, limits())?;
            assert_eq!(state_positions(&block), 0);
        }
        let memory = ModelReference::memory_forecast(&binding, 1, limits())?;
        assert!(ModelReference::new(&binding, 1, memory.payload_bytes - 1, limits()).is_err());
        assert!(ModelReference::new(&binding, 1, memory.payload_bytes, limits()).is_ok());
        for capacity in [0, 17] {
            assert!(ModelReference::memory_forecast(&binding, capacity, limits()).is_err());
        }
        assert!(BlockReference::memory_forecast(&binding, 4, 1, limits()).is_err());
        for bad in [
            ReadLimits {
                encoded_bytes: 0,
                ..limits()
            },
            ReadLimits {
                decoded_bytes: 0,
                ..limits()
            },
            ReadLimits {
                tile_rows: 0,
                ..limits()
            },
            ReadLimits {
                encoded_bytes: usize::MAX,
                ..limits()
            },
        ] {
            assert!(ModelReference::memory_forecast(&binding, 1, bad).is_err());
        }
        Ok(())
    }

    #[test]
    fn model_head_forecast_includes_whole_vocabulary_with_its_live_hidden_vectors() -> Result<()> {
        let mut fixture = Fixture::hybrid();
        fixture.tensor("token_embd.weight", 12, &[256, 65536]);
        fixture.tensor("output.weight", 14, &[256, 65536]);
        let (_file, gguf) = fixture.open();
        let binding = Binding::load(&gguf)?;
        let memory = ModelReference::memory_forecast(&binding, 1, limits())?;
        assert_eq!(
            memory.scratch_bytes,
            (3 * 256 + 65536) * 4 + limits().encoded_bytes + limits().decoded_bytes
        );
        assert!(
            memory.scratch_bytes
                > BlockReference::memory_forecast(&binding, 0, 1, limits())?.scratch_bytes
        );
        Ok(())
    }

    #[test]
    fn committed_attention_failure_poisons_both_block_kinds_until_reset() -> Result<()> {
        let (_file, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let source = Source::new(&gguf);
        for layer in [0, 3] {
            let memory = BlockReference::memory_forecast(&binding, layer, 1, limits())?;
            let mut block =
                BlockReference::new(&binding, layer, 1, memory.payload_bytes, limits())?;
            let reads = source.reads.load(Ordering::Relaxed);
            assert!(block.step(&source, &[], &mut || Ok(())).is_err());
            assert!(block
                .step(&source, &[f32::NAN; 256], &mut || Ok(()))
                .is_err());
            assert!(!block.is_poisoned());
            assert_eq!(source.reads.load(Ordering::Relaxed), reads);
            // Failing the post-attention norm read happens after state.step has
            // committed, so block-level publication must fail closed.
            source.fail_start.store(
                binding.blocks()[layer]
                    .post_attention_norm
                    .locator()
                    .start(),
                Ordering::Relaxed,
            );
            let error = block.step(&source, &[1.; 256], &mut || Ok(())).unwrap_err();
            assert!(error
                .to_string()
                .contains("injected reference read failure"));
            assert_eq!(state_positions(&block), 1);
            assert_eq!(block.positions(), 0);
            assert!(block.is_poisoned());
            source.fail_start.store(u64::MAX, Ordering::Relaxed);
            let reads = source.reads.load(Ordering::Relaxed);
            assert!(block.step(&source, &[1.; 256], &mut || Ok(())).is_err());
            assert_eq!(source.reads.load(Ordering::Relaxed), reads);
            block.reset();
            assert!(!block.is_poisoned());
            assert_eq!(state_positions(&block), 0);
            assert_eq!(
                block.step(&source, &[1.; 256], &mut || Ok(()))?,
                vec![1.; 256]
            );
            assert_eq!(block.positions(), 1);
            assert!(block.step(&source, &[1.; 256], &mut || Ok(())).is_err());
            assert!(
                !block.is_poisoned(),
                "context refusal must not mutate a valid sequence"
            );
            block.reset();
            assert_eq!(
                block.step(&source, &[1.; 256], &mut || Ok(()))?,
                vec![1.; 256]
            );
        }
        Ok(())
    }

    #[test]
    fn model_partial_token_is_unpublished_and_reset_clears_every_block() -> Result<()> {
        let (_file, gguf) = Fixture::hybrid().open();
        let binding = Binding::load(&gguf)?;
        let source = Source::new(&gguf);
        let memory = ModelReference::memory_forecast(&binding, 1, limits())?;
        let mut model = ModelReference::new(&binding, 1, memory.payload_bytes, limits())?;
        assert!(model.step(&source, 256, &mut || Ok(())).is_err());
        assert_eq!(source.reads.load(Ordering::Relaxed), 0);
        assert!(!model.is_poisoned());
        source.fail_start.store(
            binding.blocks()[1].attention_norm.locator().start(),
            Ordering::Relaxed,
        );
        assert!(model
            .step(&source, 0, &mut || Ok(()))
            .unwrap_err()
            .to_string()
            .contains("injected reference read failure"));
        assert_eq!(model.positions(), 0);
        assert_eq!(model.blocks[0].positions(), 1);
        assert_eq!(model.blocks[1].positions(), 0);
        assert!(model.is_poisoned());
        assert!(model.blocks[1].is_poisoned());
        source.fail_start.store(u64::MAX, Ordering::Relaxed);
        let reads = source.reads.load(Ordering::Relaxed);
        assert!(model.step(&source, 0, &mut || Ok(())).is_err());
        assert_eq!(source.reads.load(Ordering::Relaxed), reads);
        model.reset();
        assert!(!model.is_poisoned());
        assert!(model
            .blocks
            .iter()
            .all(|b| !b.is_poisoned() && b.positions() == 0 && state_positions(b) == 0));
        assert_eq!(model.step(&source, 0, &mut || Ok(()))?, vec![0.; 256]);
        assert_eq!(model.positions(), 1);
        assert!(model.blocks.iter().all(|b| b.positions() == 1));
        // Cancellation before a weight read also has an explicit reset path.
        model.reset();
        let reads = source.reads.load(Ordering::Relaxed);
        assert!(model
            .step(&source, 0, &mut || anyhow::bail!("cancelled"))
            .is_err());
        assert!(model.is_poisoned());
        assert_eq!(source.reads.load(Ordering::Relaxed), reads);
        model.reset();
        assert_eq!(model.step(&source, 0, &mut || Ok(()))?, vec![0.; 256]);
        Ok(())
    }
}
