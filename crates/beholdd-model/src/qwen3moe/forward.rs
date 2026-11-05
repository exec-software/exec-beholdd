//! Correctness-first execution shared by dense Qwen3 and routed Qwen3-MoE blocks.
//!
//! This is the M1 path: a [`WeightSource`] supplies exact encoded tensor slices and every
//! arithmetic operation goes through [`Device`]. Full forwards remain the correctness
//! reference; greedy decoding retains post-RoPE keys and values in explicit per-layer
//! caches while leaving weight residency and scheduling behind the source seam.

use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use beholdd_backend::{AttentionConfig, Device, IncrementalAttentionConfig, RopeLayout, Weight};
use beholdd_core::ggml_block_geometry;
use beholdd_format::GgmlType;
use beholdd_memory::{ResidentWeight, WeightAccess, WeightRequest, WeightSource};

use super::{Bound, FeedForward, Layer, Qwen3Moe, QwenModel};

/// Observable results from one transformer block.
///
/// Routing decisions are included because matching only the final hidden state can hide
/// a router convention error. All position-major fields are flattened: `experts` and
/// `expert_weights` have `n_positions * top_k` entries, and `hidden` has
/// `n_positions * dim` entries.
#[derive(Debug, Clone, PartialEq)]
pub struct BlockOutput {
    /// Number of input positions.
    pub n_positions: usize,
    /// Hidden width.
    pub dim: usize,
    /// Routed experts evaluated per position.
    pub top_k: usize,
    /// Input token IDs, one per position.
    pub tokens: Vec<u32>,
    /// Selected expert indices, position-major.
    pub experts: Vec<usize>,
    /// Renormalised selected-expert probabilities, position-major.
    pub expert_weights: Vec<f32>,
    /// Block output hidden states, position-major.
    pub hidden: Vec<f32>,
}

/// Observable result of a complete model forward pass.
///
/// `hidden` contains the last block's residual stream before the final RMSNorm. `logits`
/// contains only the final position's vocabulary row: earlier rows cannot affect the
/// next-token choice and retaining them would require `positions * vocabulary` memory.
/// Keeping the last row alongside the hidden state makes the model-stack boundary
/// independently testable without imposing a sampling policy on callers.
#[derive(Debug, Clone, PartialEq)]
pub struct ForwardOutput {
    /// Number of input positions.
    pub n_positions: usize,
    /// Hidden width.
    pub dim: usize,
    /// Vocabulary width of each logits row.
    pub vocab: usize,
    /// Input token IDs, one per position.
    pub tokens: Vec<u32>,
    /// Last block's residual stream, position-major.
    pub hidden: Vec<f32>,
    /// Final-normalised language-model logits for the last position.
    pub logits: Vec<f32>,
}

impl ForwardOutput {
    /// Logits for the last input position.
    pub fn last_logits(&self) -> Option<&[f32]> {
        if self.n_positions == 0 || self.vocab == 0 || self.logits.len() != self.vocab {
            return None;
        }
        Some(&self.logits)
    }

    /// Deterministic greedy choice for the last input position.
    ///
    /// Equal logits choose the lower token ID. Non-finite logits are refused rather
    /// than allowing `total_cmp` to turn a corrupted model state into a plausible token.
    pub fn greedy_token(&self) -> Result<u32> {
        let logits = self
            .last_logits()
            .context("forward output has no complete final logits row")?;
        greedy_token(logits)
    }
}

/// Result of correctness-first greedy decoding.
///
/// The prompt is evaluated once and subsequent tokens use per-layer `f32` KV caches.
/// The cache is an execution detail: the observable token and per-step-logit contract
/// remains suitable for deterministic replay and comparison with full-prefix forwards.
#[derive(Debug, Clone, PartialEq)]
pub struct DecodeOutput {
    /// Number of prompt tokens at the start of `tokens`.
    pub prompt_len: usize,
    /// Vocabulary width of each row in `step_logits`.
    pub vocab: usize,
    /// Prompt followed by generated tokens.
    pub tokens: Vec<u32>,
    /// One selected token per decode step.
    pub generated: Vec<u32>,
    /// Logits used for each greedy choice, step-major.
    pub step_logits: Vec<f32>,
}

impl DecodeOutput {
    /// Logits used to choose decode step `step`.
    pub fn logits_for_step(&self, step: usize) -> Option<&[f32]> {
        if step >= self.generated.len() {
            return None;
        }
        let start = step.checked_mul(self.vocab)?;
        let end = start.checked_add(self.vocab)?;
        self.step_logits.get(start..end)
    }
}

/// Why an incremental greedy generation call completed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationFinishReason {
    /// The requested maximum number of content tokens was emitted.
    MaxNewTokens,
    /// The model selected one of the caller-provided stop token IDs.
    StopToken,
}

/// Result of serving-oriented incremental greedy generation.
///
/// `generated` contains exactly the token IDs successfully delivered to the callback.
/// A matched stop token is sampled but neither delivered nor included in `generated`;
/// it is reported separately so callers can account for that sampled token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerateOutput {
    /// Generated content token IDs, in callback order.
    pub generated: Vec<u32>,
    /// Stop token selected by the model, if generation stopped on one.
    pub stop_token: Option<u32>,
    /// The condition that ended generation.
    pub finish_reason: GenerationFinishReason,
}

/// One layer's selected experts at the position that produced one decode step.
///
/// Diagnostic only, from [`Qwen3Moe::greedy_decode_with_trace`] -- a byproduct of the same
/// routing computation `greedy_decode` already performs and discards, not a second
/// execution path. It carries none of the reproducibility weight `DecodeOutput` alone
/// holds for Gate M1.3.
#[derive(Debug, Clone, PartialEq)]
pub struct StepRoute {
    /// Transformer layer index.
    pub layer: usize,
    /// Selected expert indices at this layer, in router order (highest probability first).
    pub experts: Vec<usize>,
    /// The `NEAR_MISS_WINDOW` highest-ranked experts immediately below the selection
    /// cutoff -- ranks `k..k+NEAR_MISS_WINDOW`, not selected this step. Free byproduct of
    /// the router's own full sort: a candidate signal for what the *next* step's routing
    /// might pick, since it costs nothing beyond keeping a slice of a sort already done.
    pub near_miss_experts: Vec<usize>,
}

struct LayerOutput {
    experts: Vec<usize>,
    expert_weights: Vec<f32>,
    near_miss_experts: Vec<usize>,
}

#[derive(Debug, Default)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    n_positions: usize,
}

impl LayerKvCache {
    fn validate_shape(&self, expected_positions: usize, kv_width: usize) -> Result<()> {
        if self.n_positions != expected_positions {
            bail!(
                "layer KV cache has {} positions, expected {expected_positions}",
                self.n_positions
            );
        }
        let expected_len = checked_product(expected_positions, kv_width, "layer KV cache")?;
        if self.keys.len() != expected_len || self.values.len() != expected_len {
            bail!(
                "layer KV cache at {expected_positions} positions needs {expected_len} key and value elements, got {} and {}",
                self.keys.len(),
                self.values.len()
            );
        }
        Ok(())
    }

    fn append(
        &mut self,
        keys: &[f32],
        values: &[f32],
        n_new_positions: usize,
        kv_width: usize,
        context: usize,
    ) -> Result<()> {
        self.validate_shape(self.n_positions, kv_width)?;
        let new_len = checked_product(n_new_positions, kv_width, "new KV cache chunk")?;
        if keys.len() != new_len || values.len() != new_len {
            bail!(
                "new KV cache chunk needs {new_len} key and value elements, got {} and {}",
                keys.len(),
                values.len()
            );
        }
        let end = self
            .n_positions
            .checked_add(n_new_positions)
            .context("KV cache position count overflows")?;
        if end > context {
            bail!("KV cache needs {end} positions, exceeding checkpoint context {context}");
        }
        let grown_len = checked_product(end, kv_width, "grown KV cache")?;
        let appended_len = self
            .keys
            .len()
            .checked_add(new_len)
            .context("grown KV cache element count overflows")?;
        if appended_len != grown_len {
            bail!(
                "grown KV cache shape mismatch: append reaches {appended_len} elements, expected {grown_len}"
            );
        }
        // Reserve both buffers before changing their lengths. If the second reserve
        // fails, the first buffer may retain extra capacity, but its logical contents
        // and every shape invariant remain unchanged, so retrying is safe.
        self.keys.try_reserve_exact(new_len).map_err(|error| {
            anyhow!("cannot reserve {new_len} f32 values for key cache: {error}")
        })?;
        self.values.try_reserve_exact(new_len).map_err(|error| {
            anyhow!("cannot reserve {new_len} f32 values for value cache: {error}")
        })?;
        self.keys.extend_from_slice(keys);
        self.values.extend_from_slice(values);
        self.n_positions = end;
        Ok(())
    }

    fn rollback(&mut self, n_positions: usize, len: usize) {
        self.keys.truncate(len);
        self.values.truncate(len);
        self.n_positions = n_positions;
    }
}

#[derive(Debug)]
struct DecodeCache {
    layers: Vec<LayerKvCache>,
    n_positions: usize,
}

impl DecodeCache {
    fn new(n_layers: usize) -> Result<Self> {
        let mut layers = reserved_vec(n_layers, "decode layer caches")?;
        layers.extend((0..n_layers).map(|_| LayerKvCache::default()));
        Ok(Self {
            layers,
            n_positions: 0,
        })
    }
}

struct ResidentTensor {
    storage: ResidentWeight,
    kind: GgmlType,
    rows: usize,
    cols: usize,
}

impl ResidentTensor {
    fn weight(&self) -> Weight<'_> {
        Weight {
            bytes: self.storage.bytes(),
            kind: self.kind,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Qwen3Moe {
    /// Check every checkpoint tensor against a device before executing any arithmetic.
    ///
    /// `Qwen3Moe::load` binds checkpoint shapes without choosing a backend. This is the
    /// second half of binding: an unsupported stored format is refused up front rather
    /// than halfway through a block after several gigabytes have been read.
    pub fn validate_device(&self, device: &dyn Device) -> Result<()> {
        let check = |bound: &Bound| -> Result<()> {
            if !device.supports(bound.tensor.kind()) {
                bail!(
                    "{} cannot bind {} weight {}x{}",
                    device.name(),
                    bound.tensor.kind().name(),
                    bound.rows,
                    bound.cols
                );
            }
            let expected = encoded_matrix_bytes(bound.tensor.kind(), bound.rows, bound.cols)?;
            if expected != bound.tensor.nbytes() {
                bail!(
                    "{}x{} {} weight needs exactly {expected} encoded bytes, tensor has {}",
                    bound.rows,
                    bound.cols,
                    bound.tensor.kind().name(),
                    bound.tensor.nbytes()
                );
            }
            Ok(())
        };
        let check_support = |bound: &Bound| -> Result<()> {
            if !device.supports(bound.tensor.kind()) {
                bail!(
                    "{} cannot bind {} weight {}x{}",
                    device.name(),
                    bound.tensor.kind().name(),
                    bound.rows,
                    bound.cols
                );
            }
            Ok(())
        };

        check(&self.embed)?;
        check(&self.out_norm)?;
        check(&self.output)?;
        for layer in &self.layers {
            for bound in [
                &layer.attn_norm,
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.q_norm,
                &layer.k_norm,
                &layer.wo,
                &layer.ffn_norm,
            ] {
                check(bound)?;
            }
            match &layer.ffn {
                FeedForward::Dense { gate, up, down } => {
                    for bound in [gate, up, down] {
                        check(bound)?;
                    }
                }
                FeedForward::Routed {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                } => {
                    check(router)?;
                    for bound in [gate_exps, up_exps, down_exps] {
                        check_support(bound)?;
                        let per_expert =
                            encoded_matrix_bytes(bound.tensor.kind(), bound.rows, bound.cols)?;
                        let expected = per_expert
                            .checked_mul(
                                u64::try_from(self.cfg.n_experts)
                                    .context("expert count does not fit u64")?,
                            )
                            .context("aggregate expert encoded byte count overflows")?;
                        if expected != bound.tensor.nbytes() {
                            bail!(
                                "aggregate {}x{}x{} {} expert weight needs exactly {expected} encoded bytes, tensor has {}",
                                bound.rows,
                                bound.cols,
                                self.cfg.n_experts,
                                bound.tensor.kind().name(),
                                bound.tensor.nbytes()
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Run one complete block using weights supplied independently of model semantics.
    ///
    /// This preserves the M1.1 block probe: input activations are real embedding rows,
    /// and the result exposes routing as well as the residual stream.
    pub fn forward_block(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        layer_index: usize,
        tokens: &[u32],
    ) -> Result<BlockOutput> {
        Self::with_prefetch_cleanup(weights, || {
            self.validate_execution(device, weights, tokens, "a block forward")?;
            let mut hidden = self.embed_tokens(device, weights, tokens)?;
            let routes =
                self.forward_layer(device, weights, layer_index, tokens.len(), &mut hidden)?;
            Ok(BlockOutput {
                n_positions: tokens.len(),
                dim: self.cfg.dim,
                top_k: self.cfg.n_experts_used,
                tokens: copy_slice(tokens, "output token IDs")?,
                experts: routes.experts,
                expert_weights: routes.expert_weights,
                hidden,
            })
        })
    }

    /// Run embeddings, every transformer block, final RMSNorm, and the language-model
    /// head for a non-empty token sequence.
    pub fn forward(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        tokens: &[u32],
    ) -> Result<ForwardOutput> {
        Self::with_prefetch_cleanup(weights, || {
            self.validate_execution(device, weights, tokens, "a full forward")?;
            let n_positions = tokens.len();
            let mut hidden = self.embed_tokens(device, weights, tokens)?;
            for layer_index in 0..self.layers.len() {
                self.forward_layer(device, weights, layer_index, n_positions, &mut hidden)
                    .with_context(|| format!("executing transformer layer {layer_index}"))?;
            }
            let logits = self.project_logits(device, weights, n_positions, &hidden)?;
            Ok(ForwardOutput {
                n_positions,
                dim: self.cfg.dim,
                vocab: self.cfg.vocab,
                tokens: copy_slice(tokens, "forward token IDs")?,
                hidden,
                logits,
            })
        })
    }

    /// Greedily append exactly `max_new_tokens` token IDs.
    ///
    /// This is the deterministic correctness path for Gate M1.2. It intentionally has
    /// no EOS policy or tokenizer dependency: callers decide how many IDs to request and
    /// how to turn them into text. Each choice and the exact logits that produced it are
    /// returned for Gate M1.3 reproducibility checks.
    ///
    /// `near_miss_prefetch` gates a speculative extra prefetch phase per layer: the
    /// `NEAR_MISS_WINDOW` highest-ranked experts the router did NOT select, submitted
    /// as a bet on a future step's routing. Off by default so its I/O cost is measured,
    /// not assumed -- see `docs/research/model-streaming.md`.
    pub fn greedy_decode(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        near_miss_prefetch: bool,
    ) -> Result<DecodeOutput> {
        Self::with_prefetch_cleanup(weights, || {
            self.greedy_decode_inner(
                device,
                weights,
                prompt,
                max_new_tokens,
                None,
                None,
                near_miss_prefetch,
            )
        })
    }

    /// As [`Qwen3Moe::greedy_decode`], but also returns every step's per-layer selected
    /// experts at the position that produced it: `route_trace[0]` is the last prompt
    /// position, and `route_trace[i]` for `i > 0` is the position of `generated[i - 1]`.
    pub fn greedy_decode_with_trace(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        near_miss_prefetch: bool,
    ) -> Result<(DecodeOutput, Vec<Vec<StepRoute>>)> {
        Self::with_prefetch_cleanup(weights, || {
            let mut route_trace = Vec::new();
            let output = self.greedy_decode_inner(
                device,
                weights,
                prompt,
                max_new_tokens,
                Some(&mut route_trace),
                None,
                near_miss_prefetch,
            )?;
            Ok((output, route_trace))
        })
    }

    /// As [`Qwen3Moe::greedy_decode`], but also returns each step's wall-clock duration:
    /// `step_durations[0]` covers the prompt prefill (which produces `generated[0]`), and
    /// `step_durations[i]` for `i > 0` covers the cached forward at the position of
    /// `generated[i - 1]` (which produces `generated[i]`) -- the same indexing convention
    /// [`Qwen3Moe::greedy_decode_with_trace`] uses for `route_trace`.
    ///
    /// Diagnostic only: exists to see whether a streaming cache's per-step cost trends
    /// down over a session as it warms, which shapes whether a deliberate cache
    /// pre-warming phase is worth building versus relying on organic per-token fill. See
    /// `docs/research/model-streaming.md`.
    pub fn greedy_decode_with_step_timings(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        near_miss_prefetch: bool,
    ) -> Result<(DecodeOutput, Vec<Duration>)> {
        Self::with_prefetch_cleanup(weights, || {
            let mut step_durations = Vec::new();
            let output = self.greedy_decode_inner(
                device,
                weights,
                prompt,
                max_new_tokens,
                None,
                Some(&mut step_durations),
                near_miss_prefetch,
            )?;
            Ok((output, step_durations))
        })
    }

    /// Incrementally generate greedy token IDs for a serving caller.
    ///
    /// The prompt is evaluated once and subsequent positions use the same per-layer KV
    /// cache as [`Qwen3Moe::greedy_decode`]. Unlike that reproducibility-oriented API,
    /// this method retains no per-step logits history: each selected content token is
    /// delivered immediately to `emit_token`, then only its ID is kept in the returned
    /// [`GenerateOutput`].
    ///
    /// `stop_tokens` should include every token-level terminator the caller recognizes,
    /// including the tokenizer's EOS token. A matched stop token ends generation before
    /// the callback and is reported through [`GenerateOutput::stop_token`]. Callback
    /// errors abort generation and still cancel outstanding speculative weight reads.
    /// `near_miss_prefetch` has the same meaning as on [`Qwen3Moe::greedy_decode`].
    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        near_miss_prefetch: bool,
        mut emit_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenerateOutput> {
        self.generate_greedy_controlled(
            device,
            weights,
            prompt,
            max_new_tokens,
            stop_tokens,
            near_miss_prefetch,
            || Ok(()),
            &mut emit_token,
        )
    }

    /// As [`Qwen3Moe::generate_greedy`], with a separate fallible serving-control poll.
    ///
    /// `control` is invoked before prefill and at every cached-forward transformer-layer
    /// boundary. It can therefore stop a disconnected or timed-out request without
    /// manufacturing a text callback or waiting for the first generated token.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_controlled(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        near_miss_prefetch: bool,
        mut control: impl FnMut() -> Result<()>,
        mut emit_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenerateOutput> {
        Self::with_prefetch_cleanup(weights, || {
            self.generate_greedy_inner(
                device,
                weights,
                prompt,
                max_new_tokens,
                stop_tokens,
                near_miss_prefetch,
                &mut control,
                &mut emit_token,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_greedy_inner(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        near_miss_prefetch: bool,
        control: &mut dyn FnMut() -> Result<()>,
        emit_token: &mut dyn FnMut(u32) -> Result<()>,
    ) -> Result<GenerateOutput> {
        if prompt.is_empty() {
            bail!("greedy generation requires at least one prompt token");
        }
        self.validate_execution(device, weights, prompt, "greedy generation")?;
        let final_len = prompt
            .len()
            .checked_add(max_new_tokens)
            .context("generation token count overflows")?;
        // The final selected token is not evaluated. This is the same context boundary
        // as greedy_decode, applied to the requested maximum even when a stop token may
        // let generation finish earlier.
        let evaluated_len = final_len.saturating_sub(usize::from(max_new_tokens > 0));
        if evaluated_len > self.cfg.context {
            bail!(
                "prompt plus {max_new_tokens} generated tokens needs {evaluated_len} evaluated positions, exceeding checkpoint context {}",
                self.cfg.context
            );
        }

        let mut generated = reserved_vec(max_new_tokens, "generated token IDs")?;
        if max_new_tokens == 0 {
            return Ok(GenerateOutput {
                generated,
                stop_token: None,
                finish_reason: GenerationFinishReason::MaxNewTokens,
            });
        }

        control().context("checking generation control before prefill")?;
        let mut cache = DecodeCache::new(self.layers.len())?;
        let mut output = self
            .forward_cached_chunk(
                device,
                weights,
                prompt,
                &mut cache,
                None,
                near_miss_prefetch,
                Some(control),
            )
            .context("prefilling greedy-generation KV cache")?;
        for step in 0..max_new_tokens {
            let logits = output
                .last_logits()
                .context("cached forward did not produce a final logits row")?;
            let token = greedy_token(logits)
                .with_context(|| format!("selecting greedy token at step {step}"))?;
            if stop_tokens.contains(&token) {
                return Ok(GenerateOutput {
                    generated,
                    stop_token: Some(token),
                    finish_reason: GenerationFinishReason::StopToken,
                });
            }
            emit_token(token).with_context(|| format!("emitting greedy token at step {step}"))?;
            generated.push(token);
            if step + 1 < max_new_tokens {
                output = self
                    .forward_cached_chunk(
                        device,
                        weights,
                        &[token],
                        &mut cache,
                        None,
                        near_miss_prefetch,
                        Some(control),
                    )
                    .with_context(|| {
                        format!("running cached greedy generation step {}", step + 1)
                    })?;
            }
        }

        Ok(GenerateOutput {
            generated,
            stop_token: None,
            finish_reason: GenerationFinishReason::MaxNewTokens,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn greedy_decode_inner(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        mut route_trace: Option<&mut Vec<Vec<StepRoute>>>,
        mut step_durations: Option<&mut Vec<Duration>>,
        near_miss_prefetch: bool,
    ) -> Result<DecodeOutput> {
        if prompt.is_empty() {
            bail!("greedy decode requires at least one prompt token");
        }
        self.validate_execution(device, weights, prompt, "greedy decode")?;
        let final_len = prompt
            .len()
            .checked_add(max_new_tokens)
            .context("decode token count overflows")?;
        // The final generated ID is selected from the last evaluated position but is
        // not itself evaluated. Thus N generated tokens require only N-1 positions
        // beyond the prompt and a context-length prompt may still select one token.
        let evaluated_len = final_len.saturating_sub(usize::from(max_new_tokens > 0));
        if evaluated_len > self.cfg.context {
            bail!(
                "prompt plus {max_new_tokens} generated tokens needs {evaluated_len} evaluated positions, exceeding checkpoint context {}",
                self.cfg.context
            );
        }

        let mut tokens = reserved_vec(final_len, "decoded token IDs")?;
        tokens.extend_from_slice(prompt);
        let mut generated = reserved_vec(max_new_tokens, "generated token IDs")?;
        let logits_len = checked_product(max_new_tokens, self.cfg.vocab, "decode logits")?;
        let mut step_logits = reserved_vec(logits_len, "decode logits")?;

        if max_new_tokens == 0 {
            return Ok(DecodeOutput {
                prompt_len: prompt.len(),
                vocab: self.cfg.vocab,
                tokens,
                generated,
                step_logits,
            });
        }

        let mut cache = DecodeCache::new(self.layers.len())?;
        let mut pending_trace = route_trace.is_some().then(Vec::new);
        let prefill_started = Instant::now();
        let mut output = self
            .forward_cached_chunk(
                device,
                weights,
                prompt,
                &mut cache,
                pending_trace.as_mut(),
                near_miss_prefetch,
                None,
            )
            .context("prefilling greedy-decode KV cache")?;
        if let Some(durations) = step_durations.as_mut() {
            durations.push(prefill_started.elapsed());
        }
        for step in 0..max_new_tokens {
            let logits = output
                .last_logits()
                .context("cached forward did not produce a final logits row")?;
            let token = greedy_token(logits)
                .with_context(|| format!("selecting greedy token at step {step}"))?;
            step_logits.extend_from_slice(logits);
            generated.push(token);
            tokens.push(token);
            if let (Some(sink), Some(routes)) = (route_trace.as_mut(), pending_trace.take()) {
                sink.push(routes);
            }
            if step + 1 < max_new_tokens {
                pending_trace = route_trace.is_some().then(Vec::new);
                let step_started = Instant::now();
                output = self
                    .forward_cached_chunk(
                        device,
                        weights,
                        &[token],
                        &mut cache,
                        pending_trace.as_mut(),
                        near_miss_prefetch,
                        None,
                    )
                    .with_context(|| format!("running cached greedy decode step {}", step + 1))?;
                if let Some(durations) = step_durations.as_mut() {
                    durations.push(step_started.elapsed());
                }
            }
        }

        Ok(DecodeOutput {
            prompt_len: prompt.len(),
            vocab: self.cfg.vocab,
            tokens,
            generated,
            step_logits,
        })
    }

    fn with_prefetch_cleanup<T>(
        weights: &dyn WeightSource,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let result = operation();
        let cleanup = weights
            .cancel_prefetch()
            .context("cancelling speculative weight reads");
        match (result, cleanup) {
            (Ok(value), Ok(())) => Ok(value),
            (Ok(_), Err(error)) => Err(error),
            (Err(error), Ok(())) => Err(error),
            (Err(error), Err(cleanup)) => Err(anyhow!(
                "{error:#}; additionally failed to cancel speculative weight reads: {cleanup:#}"
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_cached_chunk(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        tokens: &[u32],
        cache: &mut DecodeCache,
        mut trace: Option<&mut Vec<StepRoute>>,
        near_miss_prefetch: bool,
        mut control: Option<&mut dyn FnMut() -> Result<()>>,
    ) -> Result<ForwardOutput> {
        self.validate_execution(device, weights, tokens, "a cached forward")?;
        if cache.layers.len() != self.layers.len() {
            bail!(
                "decode cache has {} layers, model has {}",
                cache.layers.len(),
                self.layers.len()
            );
        }
        let position_start = cache.n_positions;
        let position_end = position_start
            .checked_add(tokens.len())
            .context("cached forward position range overflows")?;
        if position_end > self.cfg.context {
            bail!(
                "cached forward needs {position_end} positions, exceeding checkpoint context {}",
                self.cfg.context
            );
        }
        let kv_width = checked_product(
            self.cfg.n_kv_heads,
            self.cfg.head_dim,
            "cached key/value width",
        )?;
        for (layer_index, layer_cache) in cache.layers.iter().enumerate() {
            layer_cache
                .validate_shape(position_start, kv_width)
                .with_context(|| format!("validating transformer layer {layer_index} KV cache"))?;
        }

        let checkpoint_len = checked_product(position_start, kv_width, "KV cache checkpoint")?;
        let top_k = self.cfg.n_experts_used;
        let near_miss_len = NEAR_MISS_WINDOW.min(self.cfg.n_experts.saturating_sub(top_k));
        let result = (|| {
            let n_positions = tokens.len();
            let mut hidden = self.embed_tokens(device, weights, tokens)?;
            for (layer_index, layer_cache) in cache.layers.iter_mut().enumerate() {
                if let Some(control) = control.as_deref_mut() {
                    control().with_context(|| {
                        format!(
                            "checking generation control before cached transformer layer {layer_index}"
                        )
                    })?;
                }
                let routes = self
                    .forward_layer_cached(
                        device,
                        weights,
                        layer_index,
                        n_positions,
                        position_start,
                        &mut hidden,
                        layer_cache,
                        near_miss_prefetch,
                    )
                    .with_context(|| format!("executing cached transformer layer {layer_index}"))?;
                if let Some(trace) = trace.as_mut() {
                    let last = (n_positions - 1) * top_k;
                    let nm_last = (n_positions - 1) * near_miss_len;
                    trace.push(StepRoute {
                        layer: layer_index,
                        experts: routes.experts[last..last + top_k].to_vec(),
                        near_miss_experts: routes.near_miss_experts
                            [nm_last..nm_last + near_miss_len]
                            .to_vec(),
                    });
                }
            }
            if let Some(control) = control.as_deref_mut() {
                control().context("checking generation control before output projection")?;
            }
            let logits = self.project_logits(device, weights, n_positions, &hidden)?;
            Ok(ForwardOutput {
                n_positions,
                dim: self.cfg.dim,
                vocab: self.cfg.vocab,
                tokens: copy_slice(tokens, "cached forward token IDs")?,
                hidden,
                logits,
            })
        })();

        match result {
            Ok(output) => {
                cache.n_positions = position_end;
                Ok(output)
            }
            Err(error) => {
                for layer_cache in &mut cache.layers {
                    layer_cache.rollback(position_start, checkpoint_len);
                }
                cache.n_positions = position_start;
                Err(error)
            }
        }
    }

    fn validate_execution(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        tokens: &[u32],
        operation: &str,
    ) -> Result<()> {
        if weights.checkpoint_id() != self.checkpoint_id {
            bail!(
                "Qwen model was bound to checkpoint {:?}, but weight source is checkpoint {:?}",
                self.checkpoint_id,
                weights.checkpoint_id()
            );
        }
        self.validate_device(device)?;
        if tokens.is_empty() {
            bail!("{operation} requires at least one token");
        }
        if tokens.len() > self.cfg.context {
            bail!(
                "{} positions exceed checkpoint context {}",
                tokens.len(),
                self.cfg.context
            );
        }
        if self.cfg.head_dim == 0 || !self.cfg.head_dim.is_multiple_of(2) {
            bail!("Qwen RoPE requires a positive even head dimension");
        }
        for &token in tokens {
            let token = usize::try_from(token).context("token ID does not fit usize")?;
            if token >= self.cfg.vocab {
                bail!("token {token} is outside vocabulary {}", self.cfg.vocab);
            }
        }
        Ok(())
    }

    fn embed_tokens(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let hidden_len = checked_product(tokens.len(), self.cfg.dim, "embedded hidden state")?;
        let mut hidden = zeroed_f32(hidden_len, "embedded hidden state")?;
        self.prefetch_embeddings(weights, tokens)?;
        for (position, &token) in tokens.iter().enumerate() {
            let token = usize::try_from(token).context("token ID does not fit usize")?;
            let embedding = self.load_row(weights, &self.embed, token, "token embedding")?;
            let out = &mut hidden[position * self.cfg.dim..(position + 1) * self.cfg.dim];
            device
                .dequantize(out, &embedding.weight())
                .with_context(|| format!("decoding embedding row for token {token}"))?;
        }
        Ok(hidden)
    }

    fn forward_layer(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        layer_index: usize,
        n_positions: usize,
        hidden: &mut [f32],
    ) -> Result<LayerOutput> {
        // Near-miss speculative prefetch predicts a FUTURE decode step's routing; a
        // non-cached forward has no future step to predict for, so it stays off here
        // unconditionally rather than taking a parameter nothing would ever set.
        self.forward_layer_at(
            device,
            weights,
            layer_index,
            n_positions,
            0,
            hidden,
            None,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_cached(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        layer_index: usize,
        n_positions: usize,
        position_start: usize,
        hidden: &mut [f32],
        cache: &mut LayerKvCache,
        near_miss_prefetch: bool,
    ) -> Result<LayerOutput> {
        self.forward_layer_at(
            device,
            weights,
            layer_index,
            n_positions,
            position_start,
            hidden,
            Some(cache),
            near_miss_prefetch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn forward_layer_at(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        layer_index: usize,
        n_positions: usize,
        position_start: usize,
        hidden: &mut [f32],
        cache: Option<&mut LayerKvCache>,
        near_miss_prefetch: bool,
    ) -> Result<LayerOutput> {
        let layer = self.layers.get(layer_index).with_context(|| {
            format!(
                "layer {layer_index} is outside {} layers",
                self.layers.len()
            )
        })?;

        let dim = self.cfg.dim;
        let q_width = checked_product(self.cfg.n_heads, self.cfg.head_dim, "query width")?;
        let kv_width = checked_product(self.cfg.n_kv_heads, self.cfg.head_dim, "kv width")?;
        let hidden_len = checked_product(n_positions, dim, "block hidden state")?;
        let position_end = position_start
            .checked_add(n_positions)
            .context("layer position range overflows")?;
        if position_end > self.cfg.context {
            bail!(
                "layer position range {position_start}..{position_end} exceeds checkpoint context {}",
                self.cfg.context
            );
        }
        if cache.is_none() && position_start != 0 {
            bail!("a full-sequence layer forward must start at position zero");
        }
        if let Some(cache) = cache.as_deref() {
            cache
                .validate_shape(position_start, kv_width)
                .with_context(|| format!("validating layer {layer_index} KV cache"))?;
        }
        if hidden.len() != hidden_len {
            bail!(
                "layer {layer_index} hidden state has {} elements, expected {hidden_len}",
                hidden.len()
            );
        }
        let q_len = checked_product(n_positions, q_width, "projected queries")?;
        let kv_len = checked_product(n_positions, kv_width, "projected keys/values")?;
        self.prefetch_layer_weights(weights, layer)?;

        let attn_norm = self.decode_bound(device, weights, &layer.attn_norm, "attention norm")?;
        let q_norm = self.decode_bound(device, weights, &layer.q_norm, "query norm")?;
        let k_norm = self.decode_bound(device, weights, &layer.k_norm, "key norm")?;

        let mut normalised = zeroed_f32(hidden_len, "normalised attention input")?;
        for position in 0..n_positions {
            let range = position * dim..(position + 1) * dim;
            device.rmsnorm(
                &mut normalised[range.clone()],
                &hidden[range],
                Some(&attn_norm),
                self.cfg.eps,
            )?;
        }

        let mut q = zeroed_f32(q_len, "queries")?;
        let mut k = zeroed_f32(kv_len, "keys")?;
        let mut v = zeroed_f32(kv_len, "values")?;
        {
            let wq = self.load_bound(weights, &layer.wq, "query projection")?;
            for position in 0..n_positions {
                let input = &normalised[position * dim..(position + 1) * dim];
                device
                    .matvec(
                        &mut q[position * q_width..(position + 1) * q_width],
                        input,
                        &wq.weight(),
                    )
                    .context("projecting queries")?;
            }
        }
        {
            let wk = self.load_bound(weights, &layer.wk, "key projection")?;
            for position in 0..n_positions {
                let input = &normalised[position * dim..(position + 1) * dim];
                device
                    .matvec(
                        &mut k[position * kv_width..(position + 1) * kv_width],
                        input,
                        &wk.weight(),
                    )
                    .context("projecting keys")?;
            }
        }
        {
            let wv = self.load_bound(weights, &layer.wv, "value projection")?;
            for position in 0..n_positions {
                let input = &normalised[position * dim..(position + 1) * dim];
                device
                    .matvec(
                        &mut v[position * kv_width..(position + 1) * kv_width],
                        input,
                        &wv.weight(),
                    )
                    .context("projecting values")?;
            }
        }
        drop(normalised);

        // QK-norm is per head and happens before RoPE.
        let mut q_normalised = zeroed_f32(q_len, "normalised queries")?;
        let mut k_normalised = zeroed_f32(kv_len, "normalised keys")?;
        for position in 0..n_positions {
            for head in 0..self.cfg.n_heads {
                let start = (position * self.cfg.n_heads + head) * self.cfg.head_dim;
                let range = start..start + self.cfg.head_dim;
                device.rmsnorm(
                    &mut q_normalised[range.clone()],
                    &q[range],
                    Some(&q_norm),
                    self.cfg.eps,
                )?;
            }
            for head in 0..self.cfg.n_kv_heads {
                let start = (position * self.cfg.n_kv_heads + head) * self.cfg.head_dim;
                let range = start..start + self.cfg.head_dim;
                device.rmsnorm(
                    &mut k_normalised[range.clone()],
                    &k[range],
                    Some(&k_norm),
                    self.cfg.eps,
                )?;
            }
        }
        q = q_normalised;
        k = k_normalised;
        drop(q_norm);
        drop(k_norm);
        drop(attn_norm);

        let frequencies = rope_frequencies(self.cfg.rope_base, self.cfg.head_dim)?;
        let mut cos = zeroed_f32(self.cfg.head_dim / 2, "RoPE cosine table")?;
        let mut sin = zeroed_f32(self.cfg.head_dim / 2, "RoPE sine table")?;
        for position in 0..n_positions {
            let absolute_position = position_start
                .checked_add(position)
                .context("RoPE position overflows")?;
            rope_at(&frequencies, absolute_position, &mut cos, &mut sin)?;
            for head in 0..self.cfg.n_heads {
                let start = (position * self.cfg.n_heads + head) * self.cfg.head_dim;
                device.rope(
                    &mut q[start..start + self.cfg.head_dim],
                    &cos,
                    &sin,
                    RopeLayout::SplitHalf,
                    false,
                )?;
            }
            for head in 0..self.cfg.n_kv_heads {
                let start = (position * self.cfg.n_kv_heads + head) * self.cfg.head_dim;
                device.rope(
                    &mut k[start..start + self.cfg.head_dim],
                    &cos,
                    &sin,
                    RopeLayout::SplitHalf,
                    false,
                )?;
            }
        }

        let mut attended = zeroed_f32(q_len, "attention output")?;
        let scale = (self.cfg.head_dim as f32).sqrt().recip();
        if let Some(cache) = cache {
            let prior_positions = cache.n_positions;
            let prior_len = cache.keys.len();
            cache.append(&k, &v, n_positions, kv_width, self.cfg.context)?;
            if let Err(error) = device.incremental_causal_attention(
                &mut attended,
                &q,
                &cache.keys,
                &cache.values,
                IncrementalAttentionConfig {
                    n_query_positions: n_positions,
                    query_start: position_start,
                    n_kv_positions: cache.n_positions,
                    n_heads: self.cfg.n_heads,
                    n_kv_heads: self.cfg.n_kv_heads,
                    head_dim: self.cfg.head_dim,
                    scale,
                },
            ) {
                cache.rollback(prior_positions, prior_len);
                return Err(error).context("incremental causal grouped-query attention");
            }
        } else {
            device
                .causal_attention(
                    &mut attended,
                    &q,
                    &k,
                    &v,
                    AttentionConfig {
                        n_positions,
                        n_heads: self.cfg.n_heads,
                        n_kv_heads: self.cfg.n_kv_heads,
                        head_dim: self.cfg.head_dim,
                        scale,
                    },
                )
                .context("causal grouped-query attention")?;
        }
        drop(q);
        drop(k);
        drop(v);

        let mut projected = zeroed_f32(dim, "attention projection scratch")?;
        {
            let wo = self.load_bound(weights, &layer.wo, "attention output projection")?;
            for position in 0..n_positions {
                device
                    .matvec(
                        &mut projected,
                        &attended[position * q_width..(position + 1) * q_width],
                        &wo.weight(),
                    )
                    .context("projecting attention output")?;
                let row = &mut hidden[position * dim..(position + 1) * dim];
                device
                    .add_in_place(row, &projected)
                    .context("adding attention residual")?;
            }
        }
        drop(attended);
        drop(projected);

        let ffn_norm = self.decode_bound(device, weights, &layer.ffn_norm, "FFN norm")?;
        let mut ffn_input = zeroed_f32(hidden_len, "normalised FFN input")?;
        for position in 0..n_positions {
            let range = position * dim..(position + 1) * dim;
            device.rmsnorm(
                &mut ffn_input[range.clone()],
                &hidden[range],
                Some(&ffn_norm),
                self.cfg.eps,
            )?;
        }
        drop(ffn_norm);

        let (ffn_output, experts, expert_weights, near_miss_experts) = match &layer.ffn {
            FeedForward::Dense { gate, up, down } => {
                let mut output = zeroed_f32(hidden_len, "dense FFN output")?;
                let mut gate_values = zeroed_f32(self.cfg.expert_ff, "dense gate scratch")?;
                let mut up_values = zeroed_f32(self.cfg.expert_ff, "dense up scratch")?;
                // Keep each full matrix resident across the position loop. A streaming
                // source may satisfy `load_range` with real I/O, so loading here rather
                // than per position is part of the execution contract for prefill.
                let gate_weight = self.load_bound(weights, gate, "dense FFN gate")?;
                let up_weight = self.load_bound(weights, up, "dense FFN up")?;
                let down_weight = self.load_bound(weights, down, "dense FFN down")?;
                for position in 0..n_positions {
                    let input = &ffn_input[position * dim..(position + 1) * dim];
                    device
                        .matvec(&mut gate_values, input, &gate_weight.weight())
                        .context("running dense FFN gate")?;
                    device
                        .matvec(&mut up_values, input, &up_weight.weight())
                        .context("running dense FFN up projection")?;
                    device.swiglu(&mut gate_values, &up_values, 0.0)?;
                    device
                        .matvec(
                            &mut output[position * dim..(position + 1) * dim],
                            &gate_values,
                            &down_weight.weight(),
                        )
                        .context("running dense FFN down projection")?;
                }
                (output, Vec::new(), Vec::new(), Vec::new())
            }
            FeedForward::Routed {
                router,
                gate_exps,
                up_exps,
                down_exps,
            } => {
                // Router convention: softmax over all experts, stable top-k, then
                // renormalise only the selected probabilities.
                let route_len = checked_product(n_positions, self.cfg.n_experts, "router output")?;
                let selected_len = checked_product(
                    n_positions,
                    self.cfg.n_experts_used,
                    "selected expert routes",
                )?;
                let near_miss_per_position = NEAR_MISS_WINDOW
                    .min(self.cfg.n_experts.saturating_sub(self.cfg.n_experts_used));
                let near_miss_len = checked_product(
                    n_positions,
                    near_miss_per_position,
                    "near-miss expert routes",
                )?;
                let mut probabilities = zeroed_f32(route_len, "router probabilities")?;
                let mut experts = reserved_vec(selected_len, "selected expert indices")?;
                let mut expert_weights = reserved_vec(selected_len, "selected expert weights")?;
                let mut near_miss_experts =
                    reserved_vec(near_miss_len, "near-miss expert indices")?;
                {
                    let router_weight = self.load_bound(weights, router, "expert router")?;
                    for position in 0..n_positions {
                        let probs = &mut probabilities
                            [position * self.cfg.n_experts..(position + 1) * self.cfg.n_experts];
                        device
                            .matvec(
                                probs,
                                &ffn_input[position * dim..(position + 1) * dim],
                                &router_weight.weight(),
                            )
                            .context("computing expert-router logits")?;
                        device
                            .softmax(probs)
                            .context("normalising expert-router logits")?;
                        let (selected, selected_weights, near_miss) =
                            stable_top_k(probs, self.cfg.n_experts_used)?;
                        experts.extend(selected);
                        expert_weights.extend(selected_weights);
                        near_miss_experts.extend(near_miss);
                    }
                }
                self.prefetch_selected_experts(weights, gate_exps, up_exps, down_exps, &experts)?;
                let is_near_miss_layer = layer_index + NEAR_MISS_LAST_LAYERS
                    >= self.layers.len().max(NEAR_MISS_LAST_LAYERS);
                if near_miss_prefetch && is_near_miss_layer {
                    self.prefetch_near_miss_experts(
                        weights,
                        gate_exps,
                        up_exps,
                        down_exps,
                        &near_miss_experts,
                    )?;
                }

                let mut output = zeroed_f32(hidden_len, "MoE output")?;
                let mut gate = zeroed_f32(self.cfg.expert_ff, "expert gate scratch")?;
                let mut up = zeroed_f32(self.cfg.expert_ff, "expert up scratch")?;
                for position in 0..n_positions {
                    let input = &ffn_input[position * dim..(position + 1) * dim];
                    let output_row = &mut output[position * dim..(position + 1) * dim];
                    for route in 0..self.cfg.n_experts_used {
                        let route_index = position * self.cfg.n_experts_used + route;
                        let expert = experts[route_index];
                        {
                            let weight =
                                self.load_expert(weights, gate_exps, expert, "expert gate")?;
                            device
                                .matvec(&mut gate, input, &weight.weight())
                                .with_context(|| format!("running expert {expert} gate"))?;
                        }
                        {
                            let weight = self.load_expert(weights, up_exps, expert, "expert up")?;
                            device
                                .matvec(&mut up, input, &weight.weight())
                                .with_context(|| {
                                    format!("running expert {expert} up projection")
                                })?;
                        }
                        device.swiglu(&mut gate, &up, 0.0)?;
                        {
                            let weight =
                                self.load_expert(weights, down_exps, expert, "expert down")?;
                            device
                                .matvec_add(
                                    output_row,
                                    &gate,
                                    &weight.weight(),
                                    expert_weights[route_index],
                                )
                                .with_context(|| {
                                    format!("running expert {expert} down projection")
                                })?;
                        }
                    }
                }
                (output, experts, expert_weights, near_miss_experts)
            }
        };

        device
            .add_in_place(hidden, &ffn_output)
            .context("adding feed-forward residual")?;

        Ok(LayerOutput {
            experts,
            expert_weights,
            near_miss_experts,
        })
    }

    fn project_logits(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        n_positions: usize,
        hidden: &[f32],
    ) -> Result<Vec<f32>> {
        let hidden_len = checked_product(n_positions, self.cfg.dim, "final hidden state")?;
        if hidden.len() != hidden_len {
            bail!(
                "final hidden state has {} elements, expected {hidden_len}",
                hidden.len()
            );
        }
        self.prefetch_output(weights)?;
        let out_norm = self.decode_bound(device, weights, &self.out_norm, "output norm")?;
        let position = n_positions
            .checked_sub(1)
            .context("cannot project logits for an empty sequence")?;
        let start = checked_product(position, self.cfg.dim, "last hidden row offset")?;
        let end = start
            .checked_add(self.cfg.dim)
            .context("last hidden row end overflows")?;
        let mut normalised = zeroed_f32(self.cfg.dim, "final normalised hidden state")?;
        device
            .rmsnorm(
                &mut normalised,
                &hidden[start..end],
                Some(&out_norm),
                self.cfg.eps,
            )
            .context("applying final RMSNorm")?;
        drop(out_norm);
        let output = self.load_bound(weights, &self.output, "language-model head")?;
        let mut logits = zeroed_f32(self.cfg.vocab, "model logits")?;
        device
            .matvec(&mut logits, &normalised, &output.weight())
            .context("projecting language-model logits")?;
        Ok(logits)
    }

    fn prefetch_embeddings(&self, source: &dyn WeightSource, tokens: &[u32]) -> Result<()> {
        let mut requests = reserved_vec(tokens.len(), "embedding prefetch requests")?;
        for &token in tokens {
            let row = usize::try_from(token).context("token ID does not fit usize")?;
            requests.push(self.row_request(&self.embed, row, "token embedding")?);
        }
        source
            .prefetch_phase(&requests)
            .context("submitting token-embedding prefetch phase")
    }

    fn prefetch_layer_weights(&self, source: &dyn WeightSource, layer: &Layer) -> Result<()> {
        let common = [
            (&layer.attn_norm, "attention norm"),
            (&layer.q_norm, "query norm"),
            (&layer.k_norm, "key norm"),
            (&layer.wq, "query projection"),
            (&layer.wk, "key projection"),
            (&layer.wv, "value projection"),
            (&layer.wo, "attention output projection"),
            (&layer.ffn_norm, "FFN norm"),
        ];
        let extra = match &layer.ffn {
            FeedForward::Dense { .. } => 3,
            FeedForward::Routed { .. } => 1,
        };
        let mut ordered = reserved_vec(common.len() + extra, "ordered layer weights")?;
        ordered.extend(common);
        match &layer.ffn {
            FeedForward::Dense { gate, up, down } => {
                ordered.extend([
                    (gate, "dense FFN gate"),
                    (up, "dense FFN up"),
                    (down, "dense FFN down"),
                ]);
            }
            FeedForward::Routed { router, .. } => ordered.push((router, "expert router")),
        }
        let mut requests = reserved_vec(ordered.len(), "layer-weight prefetch requests")?;
        for (bound, label) in ordered {
            requests.push(self.bound_request(bound, label)?);
        }
        source
            .prefetch_phase(&requests)
            .context("submitting deterministic layer-weight prefetch phase")
    }

    fn prefetch_selected_experts(
        &self,
        source: &dyn WeightSource,
        gate_exps: &Bound,
        up_exps: &Bound,
        down_exps: &Bound,
        experts: &[usize],
    ) -> Result<()> {
        let capacity = checked_product(experts.len(), 3, "selected-expert prefetch requests")?;
        let mut requests = reserved_vec(capacity, "selected-expert prefetch requests")?;
        for &expert in experts {
            requests.push(self.expert_request(gate_exps, expert, "expert gate")?);
            requests.push(self.expert_request(up_exps, expert, "expert up")?);
            requests.push(self.expert_request(down_exps, expert, "expert down")?);
        }
        source
            .prefetch_phase(&requests)
            .context("submitting selected-expert prefetch phase")
    }

    /// Submit gate/up/down prefetch requests for experts this layer did NOT select this
    /// step, on the bet that a future step's routing does select them.
    ///
    /// Always issued after [`Qwen3Moe::prefetch_selected_experts`], in a separate
    /// [`WeightSource::prefetch_phase`] call: the source admits speculative requests up
    /// to its queue-depth budget in submission order, so requests genuinely needed this
    /// step already claimed their slot before a near-miss candidate can compete for what
    /// remains. A wrong bet costs a wasted prefetch (already tracked in
    /// [`beholdd_memory::CacheStats`]); it can never serve the wrong bytes to the
    /// foreground, because the foreground always names the exact range it needs.
    fn prefetch_near_miss_experts(
        &self,
        source: &dyn WeightSource,
        gate_exps: &Bound,
        up_exps: &Bound,
        down_exps: &Bound,
        near_miss_experts: &[usize],
    ) -> Result<()> {
        let capacity = checked_product(near_miss_experts.len(), 3, "near-miss prefetch requests")?;
        let mut requests = reserved_vec(capacity, "near-miss prefetch requests")?;
        for &expert in near_miss_experts {
            requests.push(self.expert_request(gate_exps, expert, "near-miss expert gate")?);
            requests.push(self.expert_request(up_exps, expert, "near-miss expert up")?);
            requests.push(self.expert_request(down_exps, expert, "near-miss expert down")?);
        }
        source
            .prefetch_phase(&requests)
            .context("submitting near-miss expert prefetch phase")
    }

    fn prefetch_output(&self, source: &dyn WeightSource) -> Result<()> {
        let requests = [
            self.bound_request(&self.out_norm, "output norm")?,
            self.bound_request(&self.output, "language-model head")?,
        ];
        source
            .prefetch_phase(&requests)
            .context("submitting output-weight prefetch phase")
    }

    fn load_bound(
        &self,
        source: &dyn WeightSource,
        bound: &Bound,
        label: &str,
    ) -> Result<ResidentTensor> {
        let request = self.bound_request(bound, label)?;
        let storage = source
            .load_range(&request.range(), request.access())
            .with_context(|| format!("loading {label}"))?;
        Ok(ResidentTensor {
            storage,
            kind: bound.tensor.kind(),
            rows: bound.rows,
            cols: bound.cols,
        })
    }

    fn decode_bound(
        &self,
        device: &dyn Device,
        source: &dyn WeightSource,
        bound: &Bound,
        label: &str,
    ) -> Result<Vec<f32>> {
        let resident = self.load_bound(source, bound, label)?;
        let mut decoded = zeroed_f32(
            checked_product(bound.rows, bound.cols, label)?,
            &format!("decoded {label}"),
        )?;
        device
            .dequantize(&mut decoded, &resident.weight())
            .with_context(|| format!("decoding {label}"))?;
        Ok(decoded)
    }

    fn load_row(
        &self,
        source: &dyn WeightSource,
        bound: &Bound,
        row: usize,
        label: &str,
    ) -> Result<ResidentTensor> {
        let request = self.row_request(bound, row, label)?;
        let storage = source
            .load_range(&request.range(), request.access())
            .with_context(|| format!("loading {label} row {row}"))?;
        Ok(ResidentTensor {
            storage,
            kind: bound.tensor.kind(),
            rows: 1,
            cols: bound.cols,
        })
    }

    fn load_expert(
        &self,
        source: &dyn WeightSource,
        bound: &Bound,
        expert: usize,
        label: &str,
    ) -> Result<ResidentTensor> {
        let request = self.expert_request(bound, expert, label)?;
        let storage = source
            .load_range(&request.range(), request.access())
            .with_context(|| format!("loading {label} {expert}"))?;
        Ok(ResidentTensor {
            storage,
            kind: bound.tensor.kind(),
            rows: bound.rows,
            cols: bound.cols,
        })
    }

    fn bound_request(&self, bound: &Bound, label: &str) -> Result<WeightRequest> {
        let nbytes = encoded_matrix_bytes(bound.tensor.kind(), bound.rows, bound.cols)?;
        let range = bound
            .tensor
            .locator()
            .range(0..nbytes)
            .with_context(|| format!("locating encoded bytes for {label}"))?;
        Ok(WeightRequest::new(range, WeightAccess::Cyclic))
    }

    fn row_request(&self, bound: &Bound, row: usize, label: &str) -> Result<WeightRequest> {
        if row >= bound.rows {
            bail!("{label} row {row} is outside {} rows", bound.rows);
        }
        let row_bytes = encoded_matrix_bytes(bound.tensor.kind(), 1, bound.cols)?;
        let row = u64::try_from(row).with_context(|| format!("{label} row does not fit u64"))?;
        let start = row_bytes
            .checked_mul(row)
            .with_context(|| format!("{label} row offset overflows"))?;
        let end = start
            .checked_add(row_bytes)
            .with_context(|| format!("{label} row end overflows"))?;
        let range = bound
            .tensor
            .locator()
            .range(start..end)
            .with_context(|| format!("locating {label} row {row}"))?;
        Ok(WeightRequest::new(range, WeightAccess::Demand))
    }

    fn expert_request(&self, bound: &Bound, expert: usize, label: &str) -> Result<WeightRequest> {
        if expert >= self.cfg.n_experts {
            bail!(
                "{label} index {expert} is outside {} experts",
                self.cfg.n_experts
            );
        }
        let stride = encoded_matrix_bytes(bound.tensor.kind(), bound.rows, bound.cols)?;
        let expert =
            u64::try_from(expert).with_context(|| format!("{label} index does not fit u64"))?;
        let start = stride
            .checked_mul(expert)
            .with_context(|| format!("{label} {expert} offset overflows"))?;
        let end = start
            .checked_add(stride)
            .with_context(|| format!("{label} {expert} end overflows"))?;
        let range = bound
            .tensor
            .locator()
            .range(start..end)
            .with_context(|| format!("locating {label} {expert}"))?;
        Ok(WeightRequest::new(range, WeightAccess::Demand))
    }
}

impl QwenModel {
    /// Check every bound tensor against a device before executing arithmetic.
    pub fn validate_device(&self, device: &dyn Device) -> Result<()> {
        self.core.validate_device(device)
    }

    /// Run one complete transformer block.
    pub fn forward_block(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        layer_index: usize,
        tokens: &[u32],
    ) -> Result<BlockOutput> {
        self.core
            .forward_block(device, weights, layer_index, tokens)
    }

    /// Run embeddings, every transformer block, final norm, and the language-model head.
    pub fn forward(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        tokens: &[u32],
    ) -> Result<ForwardOutput> {
        self.core.forward(device, weights, tokens)
    }

    /// Greedily append exactly `max_new_tokens` token IDs.
    pub fn greedy_decode(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        near_miss_prefetch: bool,
    ) -> Result<DecodeOutput> {
        self.core
            .greedy_decode(device, weights, prompt, max_new_tokens, near_miss_prefetch)
    }

    /// Greedy decode plus one wall-clock duration per decode step.
    pub fn greedy_decode_with_step_timings(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        near_miss_prefetch: bool,
    ) -> Result<(DecodeOutput, Vec<Duration>)> {
        self.core.greedy_decode_with_step_timings(
            device,
            weights,
            prompt,
            max_new_tokens,
            near_miss_prefetch,
        )
    }

    /// Incrementally generate token IDs while polling a serving-control callback.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_greedy_controlled(
        &self,
        device: &dyn Device,
        weights: &dyn WeightSource,
        prompt: &[u32],
        max_new_tokens: usize,
        stop_tokens: &[u32],
        near_miss_prefetch: bool,
        control: impl FnMut() -> Result<()>,
        emit_token: impl FnMut(u32) -> Result<()>,
    ) -> Result<GenerateOutput> {
        self.core.generate_greedy_controlled(
            device,
            weights,
            prompt,
            max_new_tokens,
            stop_tokens,
            near_miss_prefetch,
            control,
            emit_token,
        )
    }
}

fn checked_product(a: usize, b: usize, label: &str) -> Result<usize> {
    a.checked_mul(b)
        .with_context(|| format!("{label} element count overflows: {a} * {b}"))
}

fn encoded_matrix_bytes(kind: GgmlType, rows: usize, cols: usize) -> Result<u64> {
    let (block_elements, block_bytes) = ggml_block_geometry(kind.0)
        .with_context(|| format!("cannot size unsupported {} weights", kind.name()))?;
    if !cols.is_multiple_of(block_elements) {
        bail!(
            "{rows}x{cols} {} weight has {cols} elements per row, not a whole number of {block_elements}-element blocks",
            kind.name()
        );
    }
    let bytes = rows
        .checked_mul(cols / block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .with_context(|| format!("{} encoded byte count overflows", kind.name()))?;
    u64::try_from(bytes).context("encoded byte count does not fit u64")
}

fn zeroed_f32(len: usize, label: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|error| anyhow!("cannot allocate {len} f32 values for {label}: {error}"))?;
    values.resize(len, 0.0);
    Ok(values)
}

fn reserved_vec<T>(capacity: usize, label: &str) -> Result<Vec<T>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(capacity)
        .map_err(|error| anyhow!("cannot reserve {capacity} entries for {label}: {error}"))?;
    Ok(values)
}

fn copy_slice<T: Copy>(values: &[T], label: &str) -> Result<Vec<T>> {
    let mut copy = reserved_vec(values.len(), label)?;
    copy.extend_from_slice(values);
    Ok(copy)
}

fn greedy_token(logits: &[f32]) -> Result<u32> {
    let (&first, rest) = logits
        .split_first()
        .context("cannot select a token from empty logits")?;
    if !first.is_finite() {
        bail!("logit 0 is not finite: {first}");
    }
    let mut best_index = 0usize;
    let mut best_value = first;
    for (offset, &value) in rest.iter().enumerate() {
        let index = offset + 1;
        if !value.is_finite() {
            bail!("logit {index} is not finite: {value}");
        }
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    u32::try_from(best_index).context("selected token ID does not fit u32")
}

fn rope_frequencies(theta: f64, dim: usize) -> Result<Vec<f64>> {
    let half = dim / 2;
    let mut frequencies = reserved_vec(half, "RoPE frequencies")?;
    for index in 0..half {
        frequencies.push(1.0 / theta.powf((2 * index) as f64 / dim as f64));
    }
    Ok(frequencies)
}

fn rope_at(frequencies: &[f64], position: usize, cos: &mut [f32], sin: &mut [f32]) -> Result<()> {
    if cos.len() != frequencies.len() || sin.len() != frequencies.len() {
        bail!(
            "RoPE table shape mismatch: {} frequencies, {} cosine values, {} sine values",
            frequencies.len(),
            cos.len(),
            sin.len()
        );
    }
    for (index, &frequency) in frequencies.iter().enumerate() {
        let angle = position as f64 * frequency;
        cos[index] = angle.cos() as f32;
        sin[index] = angle.sin() as f32;
    }
    Ok(())
}

/// How many below-cutoff ranks [`stable_top_k`] also reports as `near_miss`. Diagnostic
/// only: this window does not change which experts are selected or how their weights are
/// renormalised, only how much of the router's own already-computed sort is kept instead
/// of discarded at the cutoff.
const NEAR_MISS_WINDOW: usize = 8;

/// Near-miss prefetch is only submitted for the last this-many layers of the model, not
/// every layer. Probe constant, see the comment at its call site in `forward_layer_at`.
const NEAR_MISS_LAST_LAYERS: usize = 8;

/// Selects the top `k` experts by probability, breaking ties by lower index, and
/// renormalises their probabilities to sum to one.
///
/// Also returns up to [`NEAR_MISS_WINDOW`] further experts immediately below the cutoff
/// (ranks `k..k+NEAR_MISS_WINDOW`), unselected this call. This costs only keeping a slice
/// of the sort already required to find the top `k` -- no separate ranking pass.
fn stable_top_k(probabilities: &[f32], k: usize) -> Result<(Vec<usize>, Vec<f32>, Vec<usize>)> {
    if k == 0 || k > probabilities.len() {
        bail!(
            "top-k must be between 1 and {}, got {k}",
            probabilities.len()
        );
    }
    if probabilities
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        bail!("router probabilities must be finite and non-negative");
    }

    let mut indices = reserved_vec(probabilities.len(), "router sort indices")?;
    indices.extend(0..probabilities.len());
    indices.sort_by(|left, right| {
        probabilities[*right]
            .total_cmp(&probabilities[*left])
            .then_with(|| left.cmp(right))
    });
    let near_miss_len = NEAR_MISS_WINDOW.min(indices.len().saturating_sub(k));
    let near_miss = indices[k..k + near_miss_len].to_vec();
    indices.truncate(k);
    let sum: f64 = indices
        .iter()
        .map(|&index| probabilities[index] as f64)
        .sum();
    if !sum.is_finite() || sum <= 0.0 {
        bail!("selected router probabilities do not have a positive finite sum");
    }
    let mut weights = reserved_vec(indices.len(), "renormalised router weights")?;
    weights.extend(
        indices
            .iter()
            .map(|&index| (probabilities[index] as f64 / sum) as f32),
    );
    Ok((indices, weights, near_miss))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    use beholdd_backend::{cpu::Cpu, DeviceError};
    use beholdd_format::{
        CheckpointId, CheckpointSpan, Gguf, RangeReader, WeightLocator, WeightRange,
    };
    use beholdd_memory::{CacheConfig, FullyResidentSource, ResidentSource, StreamingSource};

    use super::*;

    #[derive(Debug, Default)]
    struct OverlapState {
        backend_blocked: bool,
        release_backend: bool,
        future_io_completed: bool,
    }

    #[derive(Debug, Default)]
    struct OverlapGate {
        state: Mutex<OverlapState>,
        changed: Condvar,
    }

    impl OverlapGate {
        fn wait_for(&self, predicate: impl Fn(&OverlapState) -> bool, label: &str) {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut state = self.state.lock().unwrap();
            while !predicate(&state) {
                let remaining = deadline.saturating_duration_since(Instant::now());
                let (next, timeout) = self.changed.wait_timeout(state, remaining).unwrap();
                state = next;
                assert!(
                    !timeout.timed_out() || predicate(&state),
                    "timed out waiting for {label}"
                );
            }
        }

        fn block_backend(&self) {
            let mut state = self.state.lock().unwrap();
            state.backend_blocked = true;
            self.changed.notify_all();
            while !state.release_backend {
                let (next, timeout) = self
                    .changed
                    .wait_timeout(state, Duration::from_secs(10))
                    .unwrap();
                state = next;
                assert!(
                    !timeout.timed_out() || state.release_backend,
                    "timed out waiting to release the blocked backend"
                );
            }
        }

        fn wait_for_backend(&self) {
            self.wait_for(|state| state.backend_blocked, "the backend compute barrier");
        }

        fn finish_future_io(&self) {
            let mut state = self.state.lock().unwrap();
            state.future_io_completed = true;
            self.changed.notify_all();
        }

        fn wait_for_future_io(&self) {
            self.wait_for(
                |state| state.future_io_completed,
                "future weight I/O to complete",
            );
        }

        fn release_backend(&self) {
            let mut state = self.state.lock().unwrap();
            state.release_backend = true;
            self.changed.notify_all();
        }
    }

    struct ReleaseBackendOnDrop(Arc<OverlapGate>);

    impl Drop for ReleaseBackendOnDrop {
        fn drop(&mut self) {
            self.0.release_backend();
        }
    }

    struct CoordinatedReader {
        inner: Arc<Gguf>,
        gate: Arc<OverlapGate>,
        future: CheckpointSpan,
        future_reads: AtomicUsize,
    }

    impl RangeReader for CoordinatedReader {
        fn checkpoint_id(&self) -> CheckpointId {
            self.inner.checkpoint_id()
        }

        fn locators(&self) -> Box<dyn Iterator<Item = WeightLocator> + '_> {
            self.inner.locators()
        }

        fn shard_count(&self) -> usize {
            self.inner.shard_count()
        }

        fn shard_len(&self, shard: usize) -> Option<u64> {
            self.inner.shard_len(shard)
        }

        fn read_span_into(&self, span: &CheckpointSpan, buf: &mut [u8]) -> Result<()> {
            if *span == self.future {
                self.gate.wait_for_backend();
                self.inner.read_span_into(span, buf)?;
                self.future_reads.fetch_add(1, Ordering::SeqCst);
                self.gate.finish_future_io();
                Ok(())
            } else {
                self.inner.read_span_into(span, buf)
            }
        }
    }

    struct BlockingCpu {
        inner: Cpu,
        gate: Arc<OverlapGate>,
        blocked_first_dequantize: AtomicBool,
    }

    impl Device for BlockingCpu {
        fn name(&self) -> &'static str {
            self.inner.name()
        }

        fn supports(&self, kind: GgmlType) -> bool {
            self.inner.supports(kind)
        }

        fn dequantize(&self, out: &mut [f32], weight: &Weight<'_>) -> Result<(), DeviceError> {
            if !self.blocked_first_dequantize.swap(true, Ordering::SeqCst) {
                self.gate.block_backend();
            }
            self.inner.dequantize(out, weight)
        }

        fn matvec(&self, y: &mut [f32], x: &[f32], weight: &Weight<'_>) -> Result<(), DeviceError> {
            self.inner.matvec(y, x, weight)
        }

        fn matvec_add(
            &self,
            y: &mut [f32],
            x: &[f32],
            weight: &Weight<'_>,
            scale: f32,
        ) -> Result<(), DeviceError> {
            self.inner.matvec_add(y, x, weight, scale)
        }

        fn rmsnorm(
            &self,
            y: &mut [f32],
            x: &[f32],
            weight: Option<&[f32]>,
            eps: f32,
        ) -> Result<(), DeviceError> {
            self.inner.rmsnorm(y, x, weight, eps)
        }

        fn softmax(&self, values: &mut [f32]) -> Result<(), DeviceError> {
            self.inner.softmax(values)
        }

        fn causal_attention(
            &self,
            out: &mut [f32],
            q: &[f32],
            k: &[f32],
            v: &[f32],
            config: AttentionConfig,
        ) -> Result<(), DeviceError> {
            self.inner.causal_attention(out, q, k, v, config)
        }

        fn incremental_causal_attention(
            &self,
            out: &mut [f32],
            q: &[f32],
            k_cache: &[f32],
            v_cache: &[f32],
            config: IncrementalAttentionConfig,
        ) -> Result<(), DeviceError> {
            self.inner
                .incremental_causal_attention(out, q, k_cache, v_cache, config)
        }

        fn add_in_place(&self, output: &mut [f32], residual: &[f32]) -> Result<(), DeviceError> {
            self.inner.add_in_place(output, residual)
        }

        fn swiglu(&self, gate: &mut [f32], up: &[f32], limit: f32) -> Result<(), DeviceError> {
            self.inner.swiglu(gate, up, limit)
        }

        fn rope(
            &self,
            values: &mut [f32],
            cos: &[f32],
            sin: &[f32],
            layout: RopeLayout,
            inverse: bool,
        ) -> Result<(), DeviceError> {
            self.inner.rope(values, cos, sin, layout, inverse)
        }
    }

    struct RefusingTensorSource<'a> {
        inner: ResidentSource<'a>,
        refused_offset: u64,
    }

    struct RecordingSource<'a> {
        inner: ResidentSource<'a>,
        phases: Mutex<Vec<Vec<WeightRequest>>>,
        loads: Mutex<Vec<WeightRequest>>,
        cancellations: AtomicUsize,
        refused: Option<WeightRange>,
    }

    impl<'a> RecordingSource<'a> {
        fn new(checkpoint: &'a Gguf, refused: Option<WeightRange>) -> Self {
            Self {
                inner: ResidentSource::new(checkpoint),
                phases: Mutex::new(Vec::new()),
                loads: Mutex::new(Vec::new()),
                cancellations: AtomicUsize::new(0),
                refused,
            }
        }
    }

    impl WeightSource for RecordingSource<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.inner.checkpoint_id()
        }

        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            self.loads
                .lock()
                .map_err(|_| anyhow!("recording-source load lock is poisoned"))?
                .push(WeightRequest::new(*range, access));
            if self.refused == Some(*range) {
                bail!(
                    "injected recorded-range load failure at shard {} offset {}..{}",
                    range.shard(),
                    range.start(),
                    range.end()
                );
            }
            self.inner.load_range(range, access)
        }

        fn prefetch_phase(&self, requests: &[WeightRequest]) -> Result<()> {
            for request in requests {
                if request.range().checkpoint_id() != self.checkpoint_id() {
                    bail!("recorded prefetch request belongs to another checkpoint");
                }
            }
            self.phases
                .lock()
                .map_err(|_| anyhow!("recording-source phase lock is poisoned"))?
                .push(copy_slice(requests, "recorded prefetch phase")?);
            Ok(())
        }

        fn cancel_prefetch(&self) -> Result<()> {
            self.cancellations.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    impl WeightSource for RefusingTensorSource<'_> {
        fn checkpoint_id(&self) -> CheckpointId {
            self.inner.checkpoint_id()
        }

        fn load_range(&self, range: &WeightRange, access: WeightAccess) -> Result<ResidentWeight> {
            if range.locator().start() == self.refused_offset {
                bail!(
                    "injected tensor load failure at offset {}",
                    range.locator().start()
                );
            }
            self.inner.load_range(range, access)
        }
    }

    #[test]
    fn stable_top_k_breaks_ties_by_expert_index_and_renormalises() {
        let (indices, weights, near_miss) = stable_top_k(&[0.1, 0.4, 0.4, 0.1], 2).unwrap();
        assert_eq!(indices, vec![1, 2]);
        assert_eq!(weights, vec![0.5, 0.5]);
        assert_eq!(
            near_miss,
            vec![0, 3],
            "near miss keeps every remaining rank when fewer than the window remain"
        );
    }

    #[test]
    fn stable_top_k_near_miss_window_is_bounded_and_never_overlaps_the_selection() {
        let probabilities: Vec<f32> = (0..40).map(|i| 40 - i).map(|i| i as f32).collect();
        let (selected, _, near_miss) = stable_top_k(&probabilities, 3).unwrap();
        assert_eq!(selected, vec![0, 1, 2]);
        assert_eq!(
            near_miss,
            (3..3 + NEAR_MISS_WINDOW).collect::<Vec<_>>(),
            "capped at NEAR_MISS_WINDOW"
        );
        assert!(near_miss.iter().all(|index| !selected.contains(index)));
    }

    #[test]
    fn greedy_selection_is_stable_and_refuses_non_finite_logits() {
        assert_eq!(greedy_token(&[-1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
        assert!(greedy_token(&[]).is_err());
        assert!(greedy_token(&[0.0, f32::NAN]).is_err());
        assert!(greedy_token(&[f32::INFINITY, 0.0]).is_err());
    }

    #[test]
    fn encoded_q8_rows_exclude_tensor_padding() {
        assert_eq!(encoded_matrix_bytes(GgmlType(8), 1, 2048).unwrap(), 2176);
        assert!(encoded_matrix_bytes(GgmlType(8), 1, 31).is_err());
        assert!(
            encoded_matrix_bytes(GgmlType(8), 32, 1).is_err(),
            "blocks may not straddle GGML rows"
        );
    }

    #[test]
    fn layer_kv_cache_refuses_bad_shapes_and_context_growth_without_mutating_entries() {
        let mut cache = LayerKvCache::default();
        cache.append(&[1.0, 2.0], &[3.0, 4.0], 1, 2, 2).unwrap();
        let before = (cache.keys.clone(), cache.values.clone(), cache.n_positions);

        let shape_error = cache
            .append(&[5.0], &[6.0, 7.0], 1, 2, 2)
            .unwrap_err()
            .to_string();
        assert!(shape_error.contains("needs 2 key and value elements"));
        assert_eq!(
            (cache.keys.clone(), cache.values.clone(), cache.n_positions),
            before
        );

        let context_error = cache
            .append(&[5.0, 6.0, 7.0, 8.0], &[9.0, 10.0, 11.0, 12.0], 2, 2, 2)
            .unwrap_err()
            .to_string();
        assert!(context_error.contains("exceeding checkpoint context"));
        assert_eq!((cache.keys, cache.values, cache.n_positions), before);
    }

    #[test]
    fn synthetic_block_forward_is_nontrivial_and_bit_deterministic() {
        let dir = tempdir("forward");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        let first = model.forward_block(&device, &source, 0, &[0, 1]).unwrap();
        let second = model.forward_block(&device, &source, 0, &[0, 1]).unwrap();

        assert_eq!(first.experts, vec![0, 1]);
        assert_eq!(first.expert_weights, vec![1.0, 1.0]);
        assert!(
            first
                .hidden
                .iter()
                .zip([1.0f32, -0.5, 0.25, 2.0, -0.75, 1.5, 0.5, -1.0])
                .any(|(actual, embedding)| (actual - embedding).abs() > 0.1),
            "attention and MoE branches must change the residual"
        );
        assert_eq!(
            first
                .hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "the complete block must be bit-deterministic"
        );

        // Fixed regression values from this deliberately tiny, hand-authored model.
        // The real-checkpoint f64 oracle remains the independent numerical proof; this
        // fixture ensures ordinary workspace tests exercise every assembly stage.
        let expected = [
            1.566_429_5,
            -0.698_426_66,
            0.364_983_08,
            3.507_802_2,
            -0.960_466_3,
            2.371_297,
            0.779_807_2,
            -1.195_086_2,
        ];
        for (index, (actual, expected)) in first.hidden.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() < 1e-6,
                "hidden[{index}] {actual} != {expected}"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dense_qwen3_dispatches_binds_and_executes_forward_and_decode() {
        let dir = tempdir("dense-forward-decode");
        let path = dir.join("tiny-qwen3.gguf");
        write_tiny_dense_checkpoint(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = QwenModel::load(&checkpoint).unwrap();
        let source = RecordingSource::new(&checkpoint, None);
        let device = Cpu::new();

        assert_eq!(model.architecture(), super::super::QwenArchitecture::Qwen3);
        assert_eq!(model.config().feed_forward_length(), 6);
        model.validate_device(&device).unwrap();

        let forward = model.forward(&device, &source, &[0, 1]).unwrap();
        assert_eq!(forward.hidden.len(), 2 * model.config().dim());
        assert_eq!(forward.logits.len(), model.config().vocab());
        assert!(forward.hidden.iter().all(|value| value.is_finite()));
        let prefill_loads = source.loads.lock().unwrap().clone();
        for (layer_index, layer) in model.core.layers.iter().enumerate() {
            let FeedForward::Dense { gate, up, down } = &layer.ffn else {
                panic!("qwen3 fixture layer {layer_index} must bind dense weights")
            };
            for (bound, label) in [
                (gate, "dense FFN gate"),
                (up, "dense FFN up"),
                (down, "dense FFN down"),
            ] {
                let request = model.core.bound_request(bound, label).unwrap();
                assert_eq!(
                    prefill_loads
                        .iter()
                        .filter(|loaded| **loaded == request)
                        .count(),
                    1,
                    "layer {layer_index} {label} must be loaded once for a multi-token prefill"
                );
            }
        }

        let first = model
            .greedy_decode(&device, &source, &[0], 3, false)
            .unwrap();
        let second = model
            .greedy_decode(&device, &source, &[0], 3, false)
            .unwrap();
        assert_eq!(first, second, "dense cached decode must be deterministic");
        assert_eq!(first.generated.len(), 3);

        let expected_total: u64 = checkpoint
            .names()
            .map(|name| checkpoint.get(name).unwrap().nbytes())
            .sum();
        let memory = model.weight_memory().unwrap();
        assert_eq!(memory.checkpoint_bytes(), expected_total);
        assert_eq!(
            memory.largest_request_bytes(),
            288,
            "the three concurrently retained 96-byte dense matrices dominate every individual fixture weight"
        );
        assert_eq!(memory.largest_matvec_elements(), 24);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dense_qwen3_is_not_accepted_by_the_qwen3moe_binder() {
        let dir = tempdir("dense-strict-dispatch");
        let path = dir.join("tiny-qwen3.gguf");
        write_tiny_dense_checkpoint(&path, false, 1);
        let checkpoint = Gguf::open(&path).unwrap();

        let error = Qwen3Moe::load(&checkpoint).unwrap_err().to_string();
        assert!(error.contains("loader is for qwen3moe"), "got: {error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn routed_qwen3moe_dispatch_matches_the_existing_strict_binder() {
        let dir = tempdir("routed-dispatch");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let routed = Qwen3Moe::load(&checkpoint).unwrap();
        let dispatched = QwenModel::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        assert_eq!(
            dispatched.architecture(),
            super::super::QwenArchitecture::Qwen3Moe
        );
        assert_eq!(
            dispatched.weight_memory().unwrap(),
            routed.weight_memory().unwrap()
        );
        assert_eq!(
            dispatched.forward(&device, &source, &[0, 1]).unwrap(),
            routed.forward(&device, &source, &[0, 1]).unwrap()
        );
        assert_eq!(
            dispatched
                .greedy_decode(&device, &source, &[0, 1], 3, false)
                .unwrap(),
            routed
                .greedy_decode(&device, &source, &[0, 1], 3, false)
                .unwrap()
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dense_qwen3_streamed_decode_matches_resident_and_respects_cache_limits() {
        let dir = tempdir("dense-streamed-decode");
        let path = dir.join("tiny-qwen3.gguf");
        write_tiny_dense_checkpoint(&path, true, 2);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let model = QwenModel::load(&checkpoint).unwrap();
        let resident = FullyResidentSource::load(checkpoint.as_ref()).unwrap();
        let config = CacheConfig::new(128, 256).unwrap();
        let streamed = StreamingSource::new(checkpoint, config).unwrap();
        let device = Cpu::new();

        let expected = model
            .greedy_decode(&device, &resident, &[0, 1], 3, false)
            .unwrap();
        let actual = model
            .greedy_decode(&device, &streamed, &[0, 1], 3, false)
            .unwrap();

        assert_eq!(actual, expected);
        let stats = streamed.stats().unwrap();
        assert!(stats.read_bytes() > 0);
        assert!(stats.pinned_bytes() <= config.pinned_bytes());
        assert!(stats.demand_bytes() <= config.demand_bytes());
        assert!(stats.cached_bytes() <= config.total_bytes());
        assert_eq!(stats.hits() + stats.misses(), stats.requests());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn dispatcher_does_not_alias_later_qwen_families() {
        let dir = tempdir("strict-family-dispatch");
        for architecture in ["qwen3next", "qwen35", "qwen35moe"] {
            let path = dir.join(format!("{architecture}.gguf"));
            write_architecture_only_checkpoint(&path, architecture);
            let checkpoint = Gguf::open(&path).unwrap();
            let error = QwenModel::load(&checkpoint).unwrap_err().to_string();
            assert!(
                error.contains("unsupported GGUF architecture"),
                "{architecture}: {error}"
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[ignore = "requires BEHOLDD_QWEN3_DENSE_MODEL pointing to Qwen3-0.6B-Q8_0.gguf"]
    fn real_dense_qwen3_binds_and_has_a_stable_coherent_decode() -> Result<()> {
        let model_path = std::path::PathBuf::from(
            std::env::var_os("BEHOLDD_QWEN3_DENSE_MODEL")
                .context("set BEHOLDD_QWEN3_DENSE_MODEL to the Qwen3-0.6B-Q8_0.gguf checkpoint")?,
        );
        if !model_path.is_file() {
            bail!(
                "BEHOLDD_QWEN3_DENSE_MODEL is not a checkpoint file: {}",
                model_path.display()
            );
        }
        let checkpoint = Gguf::open(&model_path)?;
        let tokenizer = crate::Qwen2Tokenizer::from_gguf(&checkpoint)?;
        let model = QwenModel::load(&checkpoint)?;
        let device = Cpu::new();
        model.validate_device(&device)?;
        assert_eq!(model.architecture(), super::super::QwenArchitecture::Qwen3);
        assert_eq!(model.config().n_layers(), 28);
        assert_eq!(model.config().dim(), 1024);
        assert_eq!(model.config().feed_forward_length(), 3072);
        assert_eq!(checkpoint.len(), 310);

        let prompt = tokenizer.encode("Hello", false)?;
        assert_eq!(prompt, [9707]);
        let weights = FullyResidentSource::load(&checkpoint)?;
        let first = model.greedy_decode(&device, &weights, &prompt, 8, false)?;
        let second = model.greedy_decode(&device, &weights, &prompt, 8, false)?;
        assert_eq!(first, second, "real dense decode must be stable");
        assert_eq!(
            first.generated,
            [21806, 11, 358, 2776, 14589, 369, 279, 3681]
        );
        assert_eq!(
            tokenizer.decode(&first.generated, false)?,
            " Answer, I'm sorry for the previous"
        );

        // Independent f64 GGUF evaluation gives 8.0050891847 and 7.7874742604
        // for the top two IDs. The direct Q8_0-weight x f32 CPU path agrees closely;
        // llama.cpp's activation-requantized Q8_0 kernel is intentionally recorded in
        // docs as a numerically different reference, not mislabeled as exact parity.
        let logits = first
            .logits_for_step(0)
            .context("missing first-step logits")?;
        assert!((logits[21806] - 8.005_089).abs() < 3e-5);
        assert!((logits[14582] - 7.787_474).abs() < 3e-5);
        assert!((logits[15846] - 7.640_885).abs() < 3e-5);
        assert!(logits[21806] > logits[14582]);
        assert!(((logits[21806] - logits[14582]) - 0.217_609_88).abs() < 3e-5);
        assert_eq!(tokenizer.decode(&[21806], false)?, " Answer");
        assert_eq!(tokenizer.decode(&[14582], false)?, "Question");
        assert_eq!(tokenizer.decode(&[15846], false)?, " Question");
        Ok(())
    }

    #[test]
    fn qwen_prefetch_phases_match_foreground_order_and_only_selected_experts() {
        let dir = tempdir("prefetch-phases");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = RecordingSource::new(&checkpoint, None);
        let output = model.forward_block(&Cpu::new(), &source, 0, &[0]).unwrap();

        assert_eq!(output.experts, vec![0]);
        let phases = source.phases.lock().unwrap().clone();
        assert_eq!(phases.len(), 3);
        assert_eq!(phases[0].len(), 1, "one token embedding row");
        assert_eq!(phases[1].len(), 9, "nine deterministic layer weights");
        assert_eq!(phases[2].len(), 3, "gate/up/down for top-one routing");
        let scheduled = phases.into_iter().flatten().collect::<Vec<_>>();
        let loaded = source.loads.lock().unwrap().clone();
        assert_eq!(scheduled, loaded);

        let layer = &model.layers[0];
        let FeedForward::Routed {
            gate_exps,
            up_exps,
            down_exps,
            ..
        } = &layer.ffn
        else {
            panic!("qwen3moe fixture must bind routed weights")
        };
        for request in [
            model
                .expert_request(gate_exps, 1, "unselected expert gate")
                .unwrap(),
            model
                .expert_request(up_exps, 1, "unselected expert up")
                .unwrap(),
            model
                .expert_request(down_exps, 1, "unselected expert down")
                .unwrap(),
        ] {
            assert!(
                !scheduled.contains(&request),
                "an unselected expert range was submitted"
            );
        }
        assert_eq!(source.cancellations.load(Ordering::Relaxed), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn near_miss_prefetch_is_gated_and_adds_exactly_the_near_miss_experts() {
        let dir = tempdir("near-miss-prefetch");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let device = Cpu::new();
        let expert_count = model.cfg.n_experts;
        let selected_count = model.cfg.n_experts_used;
        let near_miss_count = NEAR_MISS_WINDOW.min(expert_count.saturating_sub(selected_count));
        assert!(
            near_miss_count > 0,
            "fixture must leave a near-miss candidate"
        );

        let off_source = RecordingSource::new(&checkpoint, None);
        let off_output = model
            .greedy_decode(&device, &off_source, &[0], 1, false)
            .unwrap();
        let off_requested: Vec<WeightRequest> = off_source
            .phases
            .lock()
            .unwrap()
            .clone()
            .into_iter()
            .flatten()
            .collect();
        // embedding(1) + layer weights(9) + selected experts(3 each) + output(2:
        // out_norm, LM head) -- greedy_decode runs the full stack, unlike forward_block.
        let base = 1 + 9 + selected_count * 3 + 2;
        assert_eq!(
            off_requested.len(),
            base,
            "no near-miss phase when the flag is off"
        );

        let on_source = RecordingSource::new(&checkpoint, None);
        let on_output = model
            .greedy_decode(&device, &on_source, &[0], 1, true)
            .unwrap();
        let on_requested: Vec<WeightRequest> = on_source
            .phases
            .lock()
            .unwrap()
            .clone()
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(
            on_requested.len(),
            base + near_miss_count * 3,
            "near-miss phase adds exactly gate/up/down per near-miss expert"
        );

        assert_eq!(
            off_output, on_output,
            "a purely speculative extra phase must never change the decode itself"
        );
        for request in &off_requested {
            assert!(
                on_requested.contains(request),
                "turning near-miss prefetch on must not drop a genuinely required request"
            );
        }
        let extra = on_requested.len() - off_requested.len();
        assert_eq!(extra, near_miss_count * 3);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_prefetch_reads_the_next_weight_while_prior_weight_compute_is_blocked() {
        let dir = tempdir("prefetch-compute-overlap");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let model = Arc::new(Qwen3Moe::load(&checkpoint).unwrap());

        let synchronous =
            StreamingSource::new(checkpoint.clone(), CacheConfig::new(0, 0).unwrap()).unwrap();
        let expected = model
            .forward_block(&Cpu::new(), &synchronous, 0, &[0, 1])
            .unwrap();

        let future = model
            .row_request(&model.embed, 1, "future token embedding")
            .unwrap()
            .range();
        let gate = Arc::new(OverlapGate::default());
        let reader = Arc::new(CoordinatedReader {
            inner: checkpoint,
            gate: gate.clone(),
            future: CheckpointSpan::from_range(future),
            future_reads: AtomicUsize::new(0),
        });
        let config = CacheConfig::new(0, 0)
            .unwrap()
            .with_prefetch(32, 2, 1)
            .unwrap();
        let source = Arc::new(StreamingSource::new(reader.clone(), config).unwrap());
        let device = BlockingCpu {
            inner: Cpu::new(),
            gate: gate.clone(),
            blocked_first_dequantize: AtomicBool::new(false),
        };
        let _release = ReleaseBackendOnDrop(gate.clone());
        let running_model = model.clone();
        let running_source = source.clone();
        let run = std::thread::spawn(move || {
            running_model.forward_block(&device, running_source.as_ref(), 0, &[0, 1])
        });

        gate.wait_for_backend();
        gate.wait_for_future_io();
        let overlapped = source.stats().unwrap();
        assert_eq!(
            overlapped.requests(),
            1,
            "the second embedding was physically read before its foreground request"
        );
        assert_eq!(reader.future_reads.load(Ordering::SeqCst), 1);
        assert!(overlapped.prefetch_reserved_bytes_peak() <= config.prefetch_bytes());
        assert!(overlapped.prefetch_outstanding_peak() <= config.prefetch_queue_depth());

        gate.release_backend();
        let actual = run.join().unwrap().unwrap();
        assert_eq!(actual.n_positions, expected.n_positions);
        assert_eq!(actual.dim, expected.dim);
        assert_eq!(actual.top_k, expected.top_k);
        assert_eq!(actual.tokens, expected.tokens);
        assert_eq!(actual.experts, expected.experts);
        assert_eq!(
            actual
                .expert_weights
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .expert_weights
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual
                .hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(reader.future_reads.load(Ordering::SeqCst), 1);
        drop(source);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn qwen_cancels_submitted_prefetch_after_a_foreground_failure() {
        let dir = tempdir("prefetch-failure-cleanup");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let FeedForward::Routed { down_exps, .. } = &model.layers[0].ffn else {
            panic!("qwen3moe fixture must bind routed weights")
        };
        let refused = model
            .expert_request(down_exps, 0, "selected expert down")
            .unwrap()
            .range();
        let source = RecordingSource::new(&checkpoint, Some(refused));

        let error = model
            .forward_block(&Cpu::new(), &source, 0, &[0])
            .unwrap_err();
        assert!(
            format!("{error:#}").contains("injected recorded-range load failure"),
            "got: {error:#}"
        );
        assert_eq!(source.phases.lock().unwrap().len(), 3);
        assert_eq!(source.cancellations.load(Ordering::Relaxed), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn full_forward_runs_every_layer_then_final_norm_and_lm_head() {
        let dir = tempdir("full-forward");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        let first_layer = model.forward_block(&device, &source, 0, &[0, 1]).unwrap();
        let mut independently_stacked = first_layer.hidden.clone();
        model
            .forward_layer(&device, &source, 1, 2, &mut independently_stacked)
            .unwrap();

        let first = model.forward(&device, &source, &[0, 1]).unwrap();
        let second = model.forward(&device, &source, &[0, 1]).unwrap();
        assert_eq!(first.n_positions, 2);
        assert_eq!(first.dim, 4);
        assert_eq!(first.vocab, 2);
        assert_eq!(first.hidden, independently_stacked);
        assert_ne!(first.hidden, first_layer.hidden, "layer 1 must execute");
        assert_eq!(first.logits.len(), 2);
        assert_eq!(
            first
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            second
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            "complete-model logits must be bit deterministic"
        );

        let hidden = &first.hidden[4..8];
        let rms = ((hidden.iter().map(|value| value * value).sum::<f32>() / 4.0) + 1e-6).sqrt();
        assert!((first.logits[0] - hidden[0] / rms).abs() < 1e-6);
        assert!((first.logits[1] - hidden[1] / rms).abs() < 1e-6);
        assert_eq!(first.greedy_token().unwrap(), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cached_chunks_match_full_prefix_hidden_and_logits_bit_for_bit() {
        let dir = tempdir("cached-chunks");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();
        let mut cache = DecodeCache::new(model.layers.len()).unwrap();

        let prefill = model
            .forward_cached_chunk(&device, &source, &[0, 1], &mut cache, None, false, None)
            .unwrap();
        let full_prefill = model.forward(&device, &source, &[0, 1]).unwrap();
        assert_eq!(
            prefill
                .hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            full_prefill
                .hidden
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            prefill
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            full_prefill
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );

        let mut prefix = vec![0, 1];
        for &token in &[0, 1] {
            let cached = model
                .forward_cached_chunk(&device, &source, &[token], &mut cache, None, false, None)
                .unwrap();
            prefix.push(token);
            let recomputed = model.forward(&device, &source, &prefix).unwrap();
            let last_hidden = &recomputed.hidden[recomputed.hidden.len() - model.cfg.dim..];
            assert_eq!(
                cached
                    .hidden
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                last_hidden
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                cached
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                recomputed
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }

        assert_eq!(cache.n_positions, prefix.len());
        let kv_width = model.cfg.n_kv_heads * model.cfg.head_dim;
        for layer in cache.layers {
            assert_eq!(layer.n_positions, prefix.len());
            assert_eq!(layer.keys.len(), prefix.len() * kv_width);
            assert_eq!(layer.values.len(), prefix.len() * kv_width);
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cached_chunk_failure_rolls_back_every_layer_and_can_be_retried() {
        let dir = tempdir("cached-rollback");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let normal_source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        for refused_offset in [
            model.layers[1].attn_norm.tensor.offset(),
            model.out_norm.tensor.offset(),
        ] {
            let refusing_source = RefusingTensorSource {
                inner: normal_source,
                refused_offset,
            };
            let mut cache = DecodeCache::new(model.layers.len()).unwrap();
            let error = model
                .forward_cached_chunk(
                    &device,
                    &refusing_source,
                    &[0, 1],
                    &mut cache,
                    None,
                    false,
                    None,
                )
                .unwrap_err();
            assert!(format!("{error:#}").contains("injected tensor load failure"));
            assert_eq!(cache.n_positions, 0);
            assert!(cache.layers.iter().all(|layer| {
                layer.n_positions == 0 && layer.keys.is_empty() && layer.values.is_empty()
            }));

            let retried = model
                .forward_cached_chunk(
                    &device,
                    &normal_source,
                    &[0, 1],
                    &mut cache,
                    None,
                    false,
                    None,
                )
                .unwrap();
            let recomputed = model.forward(&device, &normal_source, &[0, 1]).unwrap();
            assert_eq!(
                retried
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                recomputed
                    .logits
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
        }
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn cached_greedy_decode_is_bit_identical_to_full_prefix_recomputation() {
        let dir = tempdir("greedy-decode");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        let prompt = [0, 1];
        let decoded = model
            .greedy_decode(&device, &source, &prompt, 3, false)
            .unwrap();
        assert_eq!(decoded.prompt_len, prompt.len());
        assert_eq!(decoded.tokens.len(), 5);
        assert_eq!(&decoded.tokens[..prompt.len()], &prompt);
        assert_eq!(
            &decoded.tokens[prompt.len()..],
            decoded.generated.as_slice()
        );
        assert_eq!(decoded.step_logits.len(), 6);

        for step in 0..decoded.generated.len() {
            let prefix_end = prompt.len() + step;
            let recomputed = model
                .forward(&device, &source, &decoded.tokens[..prefix_end])
                .unwrap();
            let cached = decoded.logits_for_step(step).unwrap();
            assert_eq!(
                cached
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                recomputed
                    .last_logits()
                    .unwrap()
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "cached logits at step {step} differ from a complete prefix forward"
            );
            assert_eq!(decoded.generated[step], recomputed.greedy_token().unwrap());
        }
        assert!(decoded.logits_for_step(3).is_none());

        let empty_decode = model
            .greedy_decode(&device, &source, &prompt, 0, false)
            .unwrap();
        assert_eq!(empty_decode.tokens, prompt);
        assert!(empty_decode.generated.is_empty());
        assert!(empty_decode.step_logits.is_empty());

        let boundary = model
            .greedy_decode(&device, &source, &[0], model.cfg.context, false)
            .unwrap();
        assert_eq!(boundary.tokens.len(), model.cfg.context + 1);

        let error = model
            .greedy_decode(&device, &source, &[0], model.cfg.context + 1, false)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("exceeding checkpoint context"),
            "got: {error}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn incremental_generation_suppresses_a_matched_stop_token() {
        let dir = tempdir("generation-stop");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let device = Cpu::new();
        let reference_source = ResidentSource::new(&checkpoint);
        let stop_token = model
            .greedy_decode(&device, &reference_source, &[0, 1], 1, false)
            .unwrap()
            .generated[0];

        let source = RecordingSource::new(&checkpoint, None);
        let mut emitted = Vec::new();
        let output = model
            .generate_greedy(
                &device,
                &source,
                &[0, 1],
                4,
                &[stop_token],
                false,
                |token| {
                    emitted.push(token);
                    Ok(())
                },
            )
            .unwrap();

        assert!(output.generated.is_empty());
        assert!(emitted.is_empty(), "the stop token must not reach the sink");
        assert_eq!(output.stop_token, Some(stop_token));
        assert_eq!(output.finish_reason, GenerationFinishReason::StopToken);
        assert_eq!(source.cancellations.load(Ordering::Relaxed), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn incremental_generation_emits_in_order_and_matches_greedy_decode() {
        let dir = tempdir("generation-regression");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();
        let prompt = [0, 1];

        let reference = model
            .greedy_decode(&device, &source, &prompt, 4, false)
            .unwrap();
        let mut emitted = Vec::new();
        let output = model
            .generate_greedy(&device, &source, &prompt, 4, &[], false, |token| {
                emitted.push(token);
                Ok(())
            })
            .unwrap();

        assert_eq!(output.generated, reference.generated);
        assert_eq!(emitted, reference.generated, "callback order changed");
        assert_eq!(output.generated.len(), 4, "maximum must be inclusive");
        assert_eq!(output.stop_token, None);
        assert_eq!(output.finish_reason, GenerationFinishReason::MaxNewTokens);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn incremental_generation_callback_failure_aborts_and_cancels_prefetch() {
        let dir = tempdir("generation-callback-failure");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let device = Cpu::new();
        let reference_source = ResidentSource::new(&checkpoint);
        let reference = model
            .greedy_decode(&device, &reference_source, &[0, 1], 3, false)
            .unwrap();
        let source = RecordingSource::new(&checkpoint, None);
        let mut seen = Vec::new();

        let error = model
            .generate_greedy(&device, &source, &[0, 1], 3, &[], true, |token| {
                seen.push(token);
                if seen.len() == 2 {
                    bail!("synthetic sink cancellation");
                }
                Ok(())
            })
            .unwrap_err();

        assert_eq!(seen, reference.generated[..2]);
        let message = format!("{error:#}");
        assert!(message.contains("emitting greedy token at step 1"));
        assert!(message.contains("synthetic sink cancellation"));
        assert_eq!(
            source.cancellations.load(Ordering::Relaxed),
            1,
            "callback failure must cross the prefetch-cleanup boundary"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn incremental_generation_control_cancels_between_prefill_layers_without_text() {
        let dir = tempdir("generation-control-cancellation");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = RecordingSource::new(&checkpoint, None);
        let mut checks = 0;
        let mut emitted = Vec::new();

        let error = model
            .generate_greedy_controlled(
                &Cpu::new(),
                &source,
                &[0, 1],
                3,
                &[],
                false,
                || {
                    checks += 1;
                    if checks == 3 {
                        bail!("synthetic prefill cancellation");
                    }
                    Ok(())
                },
                |token| {
                    emitted.push(token);
                    Ok(())
                },
            )
            .unwrap_err();

        assert_eq!(checks, 3);
        assert!(emitted.is_empty(), "control polls must not emit empty text");
        let message = format!("{error:#}");
        assert!(message.contains("cached transformer layer 1"), "{message}");
        assert!(
            message.contains("synthetic prefill cancellation"),
            "{message}"
        );
        assert_eq!(source.cancellations.load(Ordering::Relaxed), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn incremental_generation_zero_max_emits_nothing() {
        let dir = tempdir("generation-zero-max");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = RecordingSource::new(&checkpoint, None);
        let mut callback_count = 0;

        let output = model
            .generate_greedy(&Cpu::new(), &source, &[0], 0, &[0, 1], false, |_| {
                callback_count += 1;
                Ok(())
            })
            .unwrap();

        assert!(output.generated.is_empty());
        assert_eq!(output.stop_token, None);
        assert_eq!(output.finish_reason, GenerationFinishReason::MaxNewTokens);
        assert_eq!(callback_count, 0);
        assert!(source.loads.lock().unwrap().is_empty());
        assert_eq!(source.cancellations.load(Ordering::Relaxed), 1);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn route_trace_matches_undiscarded_routing_and_leaves_decode_output_unchanged() {
        let dir = tempdir("route-trace");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        let prompt = [0, 1];
        let plain = model
            .greedy_decode(&device, &source, &prompt, 3, false)
            .unwrap();
        let (traced, route_trace) = model
            .greedy_decode_with_trace(&device, &source, &prompt, 3, false)
            .unwrap();

        assert_eq!(
            plain, traced,
            "collecting a trace must not change the decode itself"
        );
        assert_eq!(route_trace.len(), 3, "one entry per decode step");
        let expected_near_miss_len =
            NEAR_MISS_WINDOW.min(model.cfg.n_experts.saturating_sub(model.cfg.n_experts_used));
        for step_routes in &route_trace {
            assert_eq!(step_routes.len(), model.layers.len());
            for (layer_index, route) in step_routes.iter().enumerate() {
                assert_eq!(route.layer, layer_index);
                assert_eq!(route.experts.len(), model.cfg.n_experts_used);
                assert!(route.experts.iter().all(|&e| e < model.cfg.n_experts));
                assert_eq!(route.near_miss_experts.len(), expected_near_miss_len);
                assert!(route
                    .near_miss_experts
                    .iter()
                    .all(|&e| e < model.cfg.n_experts));
                assert!(
                    route
                        .near_miss_experts
                        .iter()
                        .all(|e| !route.experts.contains(e)),
                    "near-miss experts must never overlap the actual selection"
                );
            }
        }

        let empty = model
            .greedy_decode_with_trace(&device, &source, &prompt, 0, false)
            .unwrap();
        assert!(empty.1.is_empty(), "zero new tokens trace nothing");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn step_timings_cover_every_step_and_leave_decode_output_unchanged() {
        let dir = tempdir("step-timings");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let source = ResidentSource::new(&checkpoint);
        let device = Cpu::new();

        let prompt = [0, 1];
        let plain = model
            .greedy_decode(&device, &source, &prompt, 3, false)
            .unwrap();
        let (timed, step_durations) = model
            .greedy_decode_with_step_timings(&device, &source, &prompt, 3, false)
            .unwrap();

        assert_eq!(
            plain, timed,
            "collecting timings must not change the decode itself"
        );
        assert_eq!(step_durations.len(), 3, "one duration per decode step");

        let empty = model
            .greedy_decode_with_step_timings(&device, &source, &prompt, 0, false)
            .unwrap();
        assert!(empty.1.is_empty(), "zero new tokens time nothing");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn streamed_decode_is_bit_identical_to_resident_and_respects_cache_limits() {
        let dir = tempdir("streamed-decode");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Arc::new(Gguf::open(&path).unwrap());
        let model = Qwen3Moe::load(&checkpoint).unwrap();
        let resident = FullyResidentSource::load(checkpoint.as_ref()).unwrap();
        let config = CacheConfig::new(128, 128).unwrap();
        let streamed = StreamingSource::new(checkpoint.clone(), config).unwrap();
        // workers=4 (not 1) so this end-to-end decode also exercises the prefetch pool's
        // concurrency, not just single-worker prefetch: correctness here is bit-identical
        // output, not read timing, so it is not sensitive to worker count.
        let prefetch_config = config.with_prefetch(128, 32, 4).unwrap();
        let prefetched = StreamingSource::new(checkpoint, prefetch_config).unwrap();
        let device = Cpu::new();

        let expected = model
            .greedy_decode(&device, &resident, &[0, 1], 3, false)
            .unwrap();
        let actual = model
            .greedy_decode(&device, &streamed, &[0, 1], 3, false)
            .unwrap();
        // near_miss_prefetch=true here (unlike the two calls above): proves the
        // speculative phase composes with the worker pool over a real streaming source
        // and still lands on the exact same bytes -- a wrong guess must never surface.
        let prefetched_actual = model
            .greedy_decode(&device, &prefetched, &[0, 1], 3, true)
            .unwrap();

        assert_eq!(actual, expected);
        assert_eq!(prefetched_actual.prompt_len, expected.prompt_len);
        assert_eq!(prefetched_actual.vocab, expected.vocab);
        assert_eq!(prefetched_actual.tokens, expected.tokens);
        assert_eq!(prefetched_actual.generated, expected.generated);
        assert_eq!(
            prefetched_actual
                .step_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            expected
                .step_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        let stats = streamed.stats().unwrap();
        assert!(
            stats.hits() > 0,
            "the integration run must exercise cache hits"
        );
        assert!(stats.read_bytes() > stats.cached_bytes());
        assert!(stats.pinned_bytes() <= config.pinned_bytes());
        assert!(stats.demand_bytes() <= config.demand_bytes());
        assert!(stats.cached_bytes() <= config.total_bytes());
        assert_eq!(stats.hits() + stats.misses(), stats.requests());
        let prefetch_stats = prefetched.stats().unwrap();
        assert!(prefetch_stats.prefetch_submitted() > 0);
        assert!(prefetch_stats.prefetch_admitted() > 0);
        assert!(prefetch_stats.prefetch_reserved_bytes_peak() <= prefetch_config.prefetch_bytes());
        assert_eq!(
            prefetch_stats.hits() + prefetch_stats.misses(),
            prefetch_stats.requests()
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_weight_source_from_another_checkpoint_is_refused() {
        let dir = tempdir("source-identity");
        let first_path = dir.join("first.gguf");
        let second_path = dir.join("second.gguf");
        write_tiny_checkpoint(&first_path, true);
        write_tiny_checkpoint(&second_path, true);

        let first = Gguf::open(&first_path).unwrap();
        let second = Gguf::open(&second_path).unwrap();
        let model = Qwen3Moe::load(&first).unwrap();
        let wrong_source = ResidentSource::new(&second);
        let error = model
            .forward_block(&Cpu::new(), &wrong_source, 0, &[0])
            .unwrap_err()
            .to_string();

        assert!(error.contains("was bound to checkpoint"), "got: {error}");
        assert!(
            error.contains("weight source is checkpoint"),
            "got: {error}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn omitted_output_weight_ties_to_token_embeddings() {
        let dir = tempdir("tied-output");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, false);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();

        assert_eq!(model.output.tensor.offset(), model.embed.tensor.offset());
        assert_eq!(model.output.tensor.nbytes(), model.embed.tensor.nbytes());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn weight_memory_counts_the_manifest_but_only_one_expert_slab_per_request() {
        let dir = tempdir("weight-memory");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();
        let expected_total: u64 = checkpoint
            .names()
            .map(|name| checkpoint.get(name).unwrap().nbytes())
            .sum();
        let model = Qwen3Moe::load(&checkpoint).unwrap();

        let memory = model.weight_memory().unwrap();
        assert_eq!(memory.checkpoint_bytes(), expected_total);
        assert_eq!(memory.largest_request_bytes(), 64);
        assert_eq!(memory.largest_matvec_elements(), 16);
        let FeedForward::Routed { gate_exps, .. } = &model.layers[0].ffn else {
            panic!("qwen3moe fixture must bind routed weights")
        };
        assert!(
            gate_exps.tensor.nbytes() > memory.largest_request_bytes(),
            "the aggregate expert tensor must not be treated as one live request"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn decode_buffer_bound_grows_with_context_and_refuses_impossible_runs() {
        let dir = tempdir("decode-memory");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();

        let one = model.greedy_decode_buffer_bytes(1, 1).unwrap();
        let longer = model.greedy_decode_buffer_bytes(4, 4).unwrap();
        assert!(one > 0);
        assert!(longer > one);
        assert!(model.greedy_decode_buffer_bytes(0, 1).is_err());
        assert!(model
            .greedy_decode_buffer_bytes(1, model.cfg.context + 1)
            .is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn generation_buffer_bound_omits_only_retained_logits_history() {
        let dir = tempdir("generation-memory");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint_layers(&path, true, 2);
        let checkpoint = Gguf::open(&path).unwrap();
        let model = Qwen3Moe::load(&checkpoint).unwrap();

        let prompt_tokens = 4;
        let max_new_tokens = 4;
        let decode = model
            .greedy_decode_buffer_bytes(prompt_tokens, max_new_tokens)
            .unwrap();
        let generate = model
            .greedy_generate_buffer_bytes(prompt_tokens, max_new_tokens)
            .unwrap();
        let retained_logits = u64::try_from(max_new_tokens * model.cfg.vocab).unwrap()
            * u64::try_from(std::mem::size_of::<f32>()).unwrap();
        assert_eq!(decode - generate, retained_logits);
        assert!(generate > 0);
        assert!(model.greedy_generate_buffer_bytes(0, 1).is_err());
        assert!(model
            .greedy_generate_buffer_bytes(1, model.cfg.context + 1)
            .is_err());
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn absent_shared_expert_width_binds_only_the_exact_all_routed_manifest() {
        let dir = tempdir("optional-shared-expert");
        let path = dir.join("tiny-qwen3moe.gguf");
        write_tiny_checkpoint(&path, true);
        let checkpoint = Gguf::open(&path).unwrap();

        assert!(checkpoint.kv("expert_shared_feed_forward_length").is_none());
        let config = super::super::Config::from_gguf(&checkpoint).unwrap();
        assert_eq!(config.shared_expert_ff, 0);
        Qwen3Moe::load(&checkpoint).unwrap();
        std::fs::remove_dir_all(dir).ok();
    }

    fn tempdir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("beholdd-qwen3moe-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn identity(scale: f32) -> Vec<f32> {
        let mut matrix = vec![0.0f32; 16];
        for diagonal in 0..4 {
            matrix[diagonal * 4 + diagonal] = scale;
        }
        matrix
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn write_tiny_checkpoint(path: &Path, include_output: bool) {
        write_tiny_checkpoint_layers(path, include_output, 1);
    }

    fn write_tiny_checkpoint_layers(path: &Path, include_output: bool, n_layers: usize) {
        write_tiny_qwen_checkpoint(path, include_output, n_layers, false);
    }

    fn write_tiny_dense_checkpoint(path: &Path, include_output: bool, n_layers: usize) {
        write_tiny_qwen_checkpoint(path, include_output, n_layers, true);
    }

    fn write_tiny_qwen_checkpoint(path: &Path, include_output: bool, n_layers: usize, dense: bool) {
        let mut tensors: Vec<(String, Vec<u64>, Vec<u8>)> = vec![
            (
                "token_embd.weight".into(),
                vec![4, 2],
                f32_bytes(&[1.0, -0.5, 0.25, 2.0, -0.75, 1.5, 0.5, -1.0]),
            ),
            ("output_norm.weight".into(), vec![4], f32_bytes(&[1.0; 4])),
        ];
        if include_output {
            tensors.push((
                "output.weight".into(),
                vec![4, 2],
                f32_bytes(&[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            ));
        }

        for layer in 0..n_layers {
            let prefix = format!("blk.{layer}");
            tensors.extend([
                (
                    format!("{prefix}.attn_norm.weight"),
                    vec![4],
                    f32_bytes(&[1.0; 4]),
                ),
                (
                    format!("{prefix}.attn_q.weight"),
                    vec![4, 4],
                    f32_bytes(&identity(1.0)),
                ),
                (
                    format!("{prefix}.attn_k.weight"),
                    vec![4, 4],
                    f32_bytes(&identity(1.0)),
                ),
                (
                    format!("{prefix}.attn_v.weight"),
                    vec![4, 4],
                    f32_bytes(&identity(0.5)),
                ),
                (
                    format!("{prefix}.attn_q_norm.weight"),
                    vec![4],
                    f32_bytes(&[1.0; 4]),
                ),
                (
                    format!("{prefix}.attn_k_norm.weight"),
                    vec![4],
                    f32_bytes(&[1.0; 4]),
                ),
                (
                    format!("{prefix}.attn_output.weight"),
                    vec![4, 4],
                    f32_bytes(&identity(1.0)),
                ),
                (
                    format!("{prefix}.ffn_norm.weight"),
                    vec![4],
                    f32_bytes(&[1.0; 4]),
                ),
            ]);

            if dense {
                tensors.extend([
                    (
                        format!("{prefix}.ffn_gate.weight"),
                        vec![4, 6],
                        f32_bytes(&[0.25; 24]),
                    ),
                    (
                        format!("{prefix}.ffn_up.weight"),
                        vec![4, 6],
                        f32_bytes(&[0.5; 24]),
                    ),
                    (
                        format!("{prefix}.ffn_down.weight"),
                        vec![6, 4],
                        f32_bytes(&[0.125; 24]),
                    ),
                ]);
            } else {
                tensors.push((
                    format!("{prefix}.ffn_gate_inp.weight"),
                    vec![4, 2],
                    f32_bytes(&[1.0, 0.5, -0.25, 0.1, -1.0, -0.5, 0.25, -0.1]),
                ));
                let mut expert_gate = identity(1.0);
                expert_gate.extend(identity(-1.0));
                let mut expert_up = identity(1.0);
                expert_up.extend(identity(0.75));
                let mut expert_down = identity(0.25);
                expert_down.extend(identity(-0.5));
                tensors.extend([
                    (
                        format!("{prefix}.ffn_gate_exps.weight"),
                        vec![4, 4, 2],
                        f32_bytes(&expert_gate),
                    ),
                    (
                        format!("{prefix}.ffn_up_exps.weight"),
                        vec![4, 4, 2],
                        f32_bytes(&expert_up),
                    ),
                    (
                        format!("{prefix}.ffn_down_exps.weight"),
                        vec![4, 4, 2],
                        f32_bytes(&expert_down),
                    ),
                ]);
            }
        }

        let arch = if dense { "qwen3" } else { "qwen3moe" };
        let mut metadata = vec![
            (format!("{arch}.block_count"), n_layers as u32),
            (format!("{arch}.embedding_length"), 4),
            (format!("{arch}.attention.head_count"), 1),
            (format!("{arch}.attention.head_count_kv"), 1),
            (format!("{arch}.attention.key_length"), 4),
            (format!("{arch}.attention.value_length"), 4),
            (
                format!("{arch}.feed_forward_length"),
                if dense { 6 } else { 4 },
            ),
            (format!("{arch}.context_length"), 16),
        ];
        if !dense {
            metadata.extend([
                (format!("{arch}.expert_count"), 2),
                (format!("{arch}.expert_used_count"), 1),
                (format!("{arch}.expert_feed_forward_length"), 4),
            ]);
        }
        let mut header = Vec::new();
        header.extend(b"GGUF");
        header.extend(3u32.to_le_bytes());
        header.extend((tensors.len() as u64).to_le_bytes());
        header.extend(((metadata.len() + 3) as u64).to_le_bytes());
        push_kv_string(&mut header, "general.architecture", arch);
        for (key, value) in metadata {
            push_string(&mut header, &key);
            header.extend(4u32.to_le_bytes()); // u32
            header.extend(value.to_le_bytes());
        }
        push_string(
            &mut header,
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
        );
        header.extend(6u32.to_le_bytes()); // f32
        header.extend(1e-6f32.to_le_bytes());
        push_string(&mut header, &format!("{arch}.rope.freq_base"));
        header.extend(6u32.to_le_bytes()); // f32
        header.extend(10_000.0f32.to_le_bytes());

        let mut payload = Vec::new();
        let mut offsets = Vec::with_capacity(tensors.len());
        for (_, _, bytes) in &tensors {
            offsets.push(payload.len() as u64);
            payload.extend(bytes);
            while !payload.len().is_multiple_of(32) {
                payload.push(0);
            }
        }
        for ((name, dims, _), offset) in tensors.iter().zip(offsets) {
            push_string(&mut header, name);
            header.extend((dims.len() as u32).to_le_bytes());
            for dim in dims {
                header.extend(dim.to_le_bytes());
            }
            header.extend(0u32.to_le_bytes()); // F32
            header.extend(offset.to_le_bytes());
        }
        while !header.len().is_multiple_of(32) {
            header.push(0);
        }
        header.extend(payload);
        File::create(path).unwrap().write_all(&header).unwrap();
    }

    fn write_architecture_only_checkpoint(path: &Path, architecture: &str) {
        let mut header = Vec::new();
        header.extend(b"GGUF");
        header.extend(3u32.to_le_bytes());
        header.extend(0u64.to_le_bytes());
        header.extend(1u64.to_le_bytes());
        push_kv_string(&mut header, "general.architecture", architecture);
        while !header.len().is_multiple_of(32) {
            header.push(0);
        }
        File::create(path).unwrap().write_all(&header).unwrap();
    }

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend((value.len() as u64).to_le_bytes());
        bytes.extend(value.as_bytes());
    }

    fn push_kv_string(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend(8u32.to_le_bytes());
        push_string(bytes, value);
    }
}
