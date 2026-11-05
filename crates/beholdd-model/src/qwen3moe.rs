//! Strict shared execution core for dense `qwen3` and routed `qwen3moe` GGUF models.
//!
//! [`QwenModel`] dispatches only those two exact architecture tags. Both profiles share
//! GQA, per-head Q/K RMSNorm, split-half RoPE, residuals, final RMSNorm, and the language
//! model head. Dense Qwen3 uses one gated SwiGLU FFN per block; Qwen3-MoE uses full
//! softmax, stable top-k routing over expert SwiGLU FFNs, and selected-probability
//! renormalisation. Later families such as `qwen3next`, `qwen35`, and `qwen35moe` are
//! different architectures and are not aliases here.
//!
//! Qwen3-MoE was chosen as the first concrete architecture because its conventional
//! attention and routing made it a useful seam test. It is also not a toy target:
//! Qwen3-Coder-480B-A35B uses the routed profile, so the work transfers directly from a
//! 30B that fits in memory to a 480B that does not.
//!
//! # Two conventions that bite
//!
//! **GGUF dimensions are fastest-varying first**, the opposite of safetensors. So
//! `attn_q.weight [2048, 4096]` maps 2048 inputs to 4096 outputs — `dims[0]` is the
//! column count and `dims[1]` the row count. Reading them the other way round produces a
//! transposed matrix that is often still the right *shape*, and therefore still runs.
//!
//! **Routed experts are one 3-D tensor, not N tensors.** `ffn_gate_exps.weight
//! [2048, 768, 128]` is 128 contiguous slabs of a 768x2048 matrix. That is a gift for
//! streaming — an expert is one contiguous run at a computable offset, with no index to
//! consult — and it is why the loader records a stride rather than 128 names.
//!
//! # Semantics GGUF does not spell out
//!
//! Both profiles are deliberately narrow and bias-free. Dense blocks require exactly
//! gate/up/down FFN weights. Routed blocks require exactly router and aggregate expert
//! gate/up/down weights and currently reject shared-expert branches. GGUF does not encode
//! every source-config semantic switch, so each binder accepts an architecture-specific
//! metadata whitelist and exact tensor manifest instead of silently ignoring behavior it
//! cannot execute.

use anyhow::{bail, Context, Result};
use beholdd_core::ggml_block_geometry;
use beholdd_format::{CheckpointId, Gguf, GgufTensor};

mod forward;
mod recorded_attention;
mod recorded_ffn;
mod recorded_model;
mod recorded_vocabulary;

pub use forward::{
    BlockOutput, DecodeOutput, ForwardOutput, GenerateOutput, GenerationFinishReason, StepRoute,
};
pub use recorded_attention::{
    DenseAttentionLimits, DenseAttentionMemory, DenseAttentionPlan, PreparedDenseAttention,
    ResidentDenseAttention, MAX_ON_DEMAND_CAPACITY,
};
pub use recorded_ffn::{
    DenseFfnBlockLimits, DenseFfnBlockMemory, DenseFfnBlockPlan, DenseFfnLimits, DenseFfnMemory,
    DenseFfnPlan, PreparedDenseFfn, PreparedDenseFfnBlock,
};
pub use recorded_model::{DenseModelLimits, DenseModelMemory, DenseModelPlan, ResidentDenseModel};

pub use recorded_vocabulary::{
    PreparedVocabulary, VocabularyLimits, VocabularyMemory, VocabularyPlan, VocabularyTile,
};

/// Weight-memory facts needed to plan exact streaming without loading tensor payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightMemory {
    checkpoint_bytes: u64,
    largest_request_bytes: u64,
    largest_matvec_elements: u64,
}

impl WeightMemory {
    /// Sum of every encoded tensor payload in the checkpoint manifest.
    pub const fn checkpoint_bytes(self) -> u64 {
        self.checkpoint_bytes
    }

    /// Largest encoded-weight working set retained at one time by the current execution
    /// path.
    ///
    /// This is one range for ordinary projections and routed-expert execution. Dense
    /// Qwen3 retains gate, up, and down weights across the same position loop, so their
    /// checked sum is one live cohort even though it is obtained as three requests.
    /// The compatibility name predates that dense execution path.
    pub const fn largest_request_bytes(self) -> u64 {
        self.largest_request_bytes
    }

    /// Matrix elements in the largest request by element count.
    ///
    /// The scalar CPU backend currently materialises one `f32` scratch value per
    /// element during a matvec, so callers must include four times this value in its
    /// streaming working set.
    pub const fn largest_matvec_elements(self) -> u64 {
        self.largest_matvec_elements
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FeedForwardProfile {
    Dense,
    Routed,
}

impl FeedForwardProfile {
    const fn architecture(self) -> &'static str {
        match self {
            Self::Dense => "qwen3",
            Self::Routed => "qwen3moe",
        }
    }
}

/// GGUF architectures implemented by the shared Qwen execution core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenArchitecture {
    /// Dense Qwen3 with a gated SwiGLU feed-forward network in every block.
    Qwen3,
    /// Qwen3-MoE with a softmax top-k routed feed-forward network in every block.
    Qwen3Moe,
}

/// Shapes and hyperparameters, read from the checkpoint rather than assumed.
#[derive(Debug)]
pub struct Config {
    /// Transformer blocks.
    pub(crate) n_layers: usize,
    /// Hidden width.
    pub(crate) dim: usize,
    /// Query heads.
    pub(crate) n_heads: usize,
    /// Key/value heads. Fewer than `n_heads` means grouped-query attention.
    pub(crate) n_kv_heads: usize,
    /// Per-head width.
    pub(crate) head_dim: usize,
    /// Per-head value width. The first implementation requires this to equal
    /// [`Config::head_dim`] because the attention output is head-shaped.
    pub(crate) value_head_dim: usize,
    /// Routed experts per layer, or zero for dense Qwen3.
    pub(crate) n_experts: usize,
    /// Experts evaluated per token, or zero for dense Qwen3.
    pub(crate) n_experts_used: usize,
    /// Inner width of a dense FFN or one routed expert.
    pub(crate) expert_ff: usize,
    /// Inner width of a shared expert, or zero when the architecture has none.
    pub(crate) shared_expert_ff: usize,
    /// Vocabulary size.
    pub(crate) vocab: usize,
    /// RMSNorm epsilon.
    pub(crate) eps: f32,
    /// RoPE base frequency.
    pub(crate) rope_base: f64,
    /// Maximum positions the checkpoint claims.
    pub(crate) context: usize,
}

impl Config {
    /// Read from GGUF metadata. Semantic hyperparameters are required rather than
    /// guessed. The one exception is `expert_shared_feed_forward_length`: GGUF writers
    /// may omit it when the architecture has no shared expert, so absence means zero.
    pub fn from_gguf(g: &Gguf) -> Result<Config> {
        Self::from_gguf_profile(g, FeedForwardProfile::Routed)
    }

    fn from_gguf_profile(g: &Gguf, profile: FeedForwardProfile) -> Result<Config> {
        let arch = g.arch().context("checkpoint has no general.architecture")?;
        let expected_arch = profile.architecture();
        if arch != expected_arch {
            bail!("this loader is for {expected_arch}, checkpoint says {arch}");
        }
        validate_arch_metadata(g, arch, profile)?;
        let need = |k: &str| -> Result<u64> {
            g.kv(k)
                .and_then(|v| v.as_u64())
                .with_context(|| format!("missing {arch}.{k}"))
        };
        let need_usize = |k: &str| -> Result<usize> {
            let value = need(k)?;
            usize::try_from(value)
                .with_context(|| format!("{arch}.{k}={value} does not fit this host"))
        };
        let optional_usize = |k: &str| -> Result<usize> {
            let Some(value) = g.kv(k) else {
                return Ok(0);
            };
            let value = value
                .as_u64()
                .with_context(|| format!("{arch}.{k} must be an unsigned integer"))?;
            usize::try_from(value)
                .with_context(|| format!("{arch}.{k}={value} does not fit this host"))
        };
        let embedding = g
            .get("token_embd.weight")
            .context("no token_embd.weight to size the vocabulary from")?;
        if embedding.dims().len() != 2 {
            bail!(
                "token_embd.weight must be 2-D, checkpoint has {:?}",
                embedding.dims()
            );
        }
        let vocab = usize::try_from(embedding.dims()[1])
            .context("token_embd.weight vocabulary does not fit this host")?;
        let n_heads = need_usize("attention.head_count")?;
        let head_dim = need_usize("attention.key_length")?;
        let (n_experts, n_experts_used, expert_ff, shared_expert_ff) = match profile {
            FeedForwardProfile::Dense => (0, 0, need_usize("feed_forward_length")?, 0),
            FeedForwardProfile::Routed => (
                need_usize("expert_count")?,
                need_usize("expert_used_count")?,
                need_usize("expert_feed_forward_length")?,
                optional_usize("expert_shared_feed_forward_length")?,
            ),
        };
        let cfg = Config {
            n_layers: need_usize("block_count")?,
            dim: need_usize("embedding_length")?,
            n_heads,
            n_kv_heads: need_usize("attention.head_count_kv")?,
            head_dim,
            value_head_dim: need_usize("attention.value_length")?,
            n_experts,
            n_experts_used,
            expert_ff,
            shared_expert_ff,
            vocab,
            eps: g
                .kv("attention.layer_norm_rms_epsilon")
                .and_then(|v| v.as_f64())
                .context("missing rms epsilon")? as f32,
            rope_base: g
                .kv("rope.freq_base")
                .and_then(|v| v.as_f64())
                .context("missing rope.freq_base")?,
            context: need_usize("context_length")?,
        };
        if cfg.n_layers == 0
            || cfg.dim == 0
            || cfg.expert_ff == 0
            || cfg.vocab == 0
            || cfg.context == 0
        {
            bail!("model dimensions, vocabulary and context must be non-zero");
        }
        if !cfg.eps.is_finite() || cfg.eps <= 0.0 {
            bail!(
                "RMSNorm epsilon must be positive and finite, got {}",
                cfg.eps
            );
        }
        if !cfg.rope_base.is_finite() || cfg.rope_base <= 0.0 {
            bail!(
                "RoPE base must be positive and finite, got {}",
                cfg.rope_base
            );
        }
        if cfg.n_heads == 0 || cfg.n_kv_heads == 0 {
            bail!("attention head counts must be non-zero");
        }
        if !cfg.n_heads.is_multiple_of(cfg.n_kv_heads) {
            bail!(
                "{} query heads do not group evenly into {} kv heads",
                cfg.n_heads,
                cfg.n_kv_heads
            );
        }
        if cfg.head_dim == 0 || !cfg.head_dim.is_multiple_of(2) {
            bail!("attention head dimension must be positive and even");
        }
        if cfg.value_head_dim != cfg.head_dim {
            bail!(
                "{arch} value head dimension {} differs from key/query dimension {}; this attention path requires them equal",
                cfg.value_head_dim,
                cfg.head_dim
            );
        }
        if profile == FeedForwardProfile::Routed
            && (cfg.n_experts == 0 || cfg.n_experts_used == 0 || cfg.n_experts_used > cfg.n_experts)
        {
            bail!(
                "invalid expert routing: top-{} of {} experts",
                cfg.n_experts_used,
                cfg.n_experts
            );
        }
        if profile == FeedForwardProfile::Routed && cfg.shared_expert_ff != 0 {
            bail!(
                "qwen3moe checkpoint has a shared expert of width {}; shared experts are not implemented",
                cfg.shared_expert_ff
            );
        }
        Ok(cfg)
    }

    /// Query heads sharing one key/value head.
    pub fn group(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    /// Transformer block count.
    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    /// Hidden width.
    pub fn dim(&self) -> usize {
        self.dim
    }
    /// Query-head count.
    pub fn n_heads(&self) -> usize {
        self.n_heads
    }
    /// Key/value-head count.
    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }
    /// Head width.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    /// Routed expert count, or zero for dense Qwen3.
    pub fn n_experts(&self) -> usize {
        self.n_experts
    }
    /// Experts selected per token, or zero for dense Qwen3.
    pub fn n_experts_used(&self) -> usize {
        self.n_experts_used
    }
    /// One expert's inner width, or the dense FFN width for `qwen3`.
    pub fn expert_ff(&self) -> usize {
        self.expert_ff
    }
    /// Dense-FFN or per-expert inner width.
    pub fn feed_forward_length(&self) -> usize {
        self.expert_ff
    }
    /// Vocabulary size.
    pub fn vocab(&self) -> usize {
        self.vocab
    }
    /// RMSNorm epsilon.
    pub fn eps(&self) -> f32 {
        self.eps
    }
    /// Rotary base frequency.
    pub fn rope_base(&self) -> f64 {
        self.rope_base
    }
    /// Maximum declared context.
    pub fn context(&self) -> usize {
        self.context
    }
}

/// Refuse architecture metadata whose semantics this implementation has not bound.
///
/// This is intentionally a whitelist. A future checkpoint that adds RoPE scaling,
/// router scaling, activation clamps, grouped routing or another architecture switch
/// must fail at bind time rather than be run as if that switch were absent.
fn validate_arch_metadata(g: &Gguf, arch: &str, profile: FeedForwardProfile) -> Result<()> {
    let prefix = format!("{arch}.");
    for key in g.metadata().keys().filter(|key| key.starts_with(&prefix)) {
        let suffix = &key[prefix.len()..];
        if !supports_arch_metadata(suffix, profile) {
            bail!("unsupported {arch} architecture metadata: {key}");
        }
    }
    Ok(())
}

fn supports_arch_metadata(suffix: &str, profile: FeedForwardProfile) -> bool {
    let common = matches!(
        suffix,
        "attention.head_count"
            | "attention.head_count_kv"
            | "attention.key_length"
            | "attention.layer_norm_rms_epsilon"
            | "attention.value_length"
            | "block_count"
            | "context_length"
            | "embedding_length"
            | "feed_forward_length"
            | "rope.freq_base"
    );
    common
        || (profile == FeedForwardProfile::Routed
            && matches!(
                suffix,
                "expert_count"
                    | "expert_feed_forward_length"
                    | "expert_shared_feed_forward_length"
                    | "expert_used_count"
            ))
}

/// A bound tensor: where its bytes are and what shape they make.
#[derive(Debug, Clone)]
struct Bound {
    /// Byte range within its shard.
    tensor: GgufTensor,
    /// Output rows.
    rows: usize,
    /// Input columns.
    cols: usize,
}

#[derive(Debug)]
enum FeedForward {
    Dense {
        gate: Bound,
        up: Bound,
        down: Bound,
    },
    Routed {
        router: Bound,
        gate_exps: Bound,
        up_exps: Bound,
        down_exps: Bound,
    },
}

/// One block's weights, located but not loaded.
#[derive(Debug)]
struct Layer {
    /// Pre-attention RMSNorm weight.
    pub attn_norm: Bound,
    /// Query projection.
    pub wq: Bound,
    /// Key projection.
    pub wk: Bound,
    /// Value projection.
    pub wv: Bound,
    /// Per-head query norm. Qwen3 normalises q and k per head before RoPE.
    pub q_norm: Bound,
    /// Per-head key norm.
    pub k_norm: Bound,
    /// Output projection.
    pub wo: Bound,
    /// Pre-FFN RMSNorm weight.
    pub ffn_norm: Bound,
    /// Dense or routed feed-forward weights.
    pub ffn: FeedForward,
}

/// Located Qwen execution core.
///
/// [`Qwen3Moe::load`] remains a strict routed-`qwen3moe` API. [`QwenModel`] also uses
/// this representation internally after its separate dense-`qwen3` profile bind.
#[derive(Debug)]
pub struct Qwen3Moe {
    checkpoint_id: CheckpointId,
    /// Shapes and hyperparameters.
    cfg: Config,
    /// Per-block weights.
    layers: Vec<Layer>,
    /// Token embedding.
    embed: Bound,
    /// Final norm.
    out_norm: Bound,
    /// Output projection to logits.
    output: Bound,
    /// Sum of unique encoded tensor payloads in the GGUF manifest.
    checkpoint_bytes: u64,
}

/// Strict dispatcher for the Qwen architectures beholdd can execute.
///
/// Dispatch is driven only by `general.architecture`. Each branch then validates its
/// own metadata whitelist and exact tensor manifest; related names such as `qwen35`,
/// `qwen35moe`, and `qwen3next` are not aliases for either implemented architecture.
#[derive(Debug)]
pub struct QwenModel {
    architecture: QwenArchitecture,
    core: Qwen3Moe,
}

impl QwenModel {
    /// Bind the exact architecture declared by a GGUF checkpoint.
    pub fn load(gguf: &Gguf) -> Result<Self> {
        let architecture = match gguf
            .arch()
            .context("checkpoint has no general.architecture")?
        {
            "qwen3" => QwenArchitecture::Qwen3,
            "qwen3moe" => QwenArchitecture::Qwen3Moe,
            other => bail!(
                "unsupported GGUF architecture {other:?}; implemented architectures are qwen3 and qwen3moe"
            ),
        };
        let profile = match architecture {
            QwenArchitecture::Qwen3 => FeedForwardProfile::Dense,
            QwenArchitecture::Qwen3Moe => FeedForwardProfile::Routed,
        };
        Ok(Self {
            architecture,
            core: Qwen3Moe::load_profile(gguf, profile)?,
        })
    }

    /// Architecture selected from the checkpoint's exact GGUF tag.
    pub const fn architecture(&self) -> QwenArchitecture {
        self.architecture
    }

    /// Validated, immutable model configuration.
    pub fn config(&self) -> &Config {
        self.core.config()
    }

    /// Encoded checkpoint size, maximum concurrent encoded-weight working set, and
    /// maximum individual matvec geometry.
    pub fn weight_memory(&self) -> Result<WeightMemory> {
        self.core.weight_memory()
    }

    /// Conservative logical bytes used by greedy decode buffers.
    pub fn greedy_decode_buffer_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<u64> {
        self.core
            .greedy_decode_buffer_bytes(prompt_tokens, max_new_tokens)
    }

    /// Conservative logical bytes used by serving-oriented greedy generation.
    pub fn greedy_generate_buffer_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<u64> {
        self.core
            .greedy_generate_buffer_bytes(prompt_tokens, max_new_tokens)
    }

    /// Total routed-expert bytes, or an error when this is a dense model.
    pub fn expert_bytes(&self) -> Result<u64> {
        self.core.expert_bytes()
    }

    /// Byte offset and encoded stride of one routed gate expert.
    pub fn gate_expert_slice(&self, layer: usize, expert: usize) -> Result<(u64, u64)> {
        self.core.gate_expert_slice(layer, expert)
    }
}

impl Qwen3Moe {
    /// Validated, immutable model configuration.
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// Encoded checkpoint size, maximum concurrent encoded-weight working set, and
    /// maximum individual matvec geometry.
    ///
    /// This is metadata-only: it performs no payload read or allocation proportional to
    /// model size. Expert aggregates are counted in `checkpoint_bytes`, while the
    /// routed execution's largest live cohort considers one expert slab because that is
    /// what it materialises at once. Dense execution accounts for all three FFN matrices
    /// retained concurrently across its position loop.
    pub fn weight_memory(&self) -> Result<WeightMemory> {
        let mut largest_request_bytes = 0u64;
        let mut largest_matvec_elements = 0u64;
        let mut consider = |bytes: u64, rows: usize, cols: usize| -> Result<()> {
            let elements = rows
                .checked_mul(cols)
                .context("weight request element count overflows")?;
            let elements =
                u64::try_from(elements).context("weight request elements do not fit u64")?;
            largest_request_bytes = largest_request_bytes.max(bytes);
            largest_matvec_elements = largest_matvec_elements.max(elements);
            Ok(())
        };

        let embedding_row_bytes = self
            .embed
            .tensor
            .nbytes()
            .checked_div(
                u64::try_from(self.embed.rows).context("embedding row count does not fit u64")?,
            )
            .context("embedding tensor has zero rows")?;
        consider(embedding_row_bytes, 1, self.embed.cols)?;
        consider(
            self.out_norm.tensor.nbytes(),
            self.out_norm.rows,
            self.out_norm.cols,
        )?;
        consider(
            self.output.tensor.nbytes(),
            self.output.rows,
            self.output.cols,
        )?;
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
                consider(bound.tensor.nbytes(), bound.rows, bound.cols)?;
            }
            match &layer.ffn {
                FeedForward::Dense { gate, up, down } => {
                    for bound in [gate, up, down] {
                        consider(bound.tensor.nbytes(), bound.rows, bound.cols)?;
                    }
                    let live_dense_bytes = gate
                        .tensor
                        .nbytes()
                        .checked_add(up.tensor.nbytes())
                        .and_then(|bytes| bytes.checked_add(down.tensor.nbytes()))
                        .context("dense FFN live encoded-weight cohort overflows")?;
                    // The cohort contributes encoded residency but no simultaneous
                    // dequantized-matrix scratch: its three matvec calls are sequential.
                    consider(live_dense_bytes, 0, 0)?;
                }
                FeedForward::Routed {
                    router,
                    gate_exps,
                    up_exps,
                    down_exps,
                } => {
                    consider(router.tensor.nbytes(), router.rows, router.cols)?;
                    for bound in [gate_exps, up_exps, down_exps] {
                        let experts = u64::try_from(self.cfg.n_experts)
                            .context("expert count does not fit u64")?;
                        if bound.tensor.nbytes() % experts != 0 {
                            bail!(
                                "aggregate expert tensor has {} bytes, not divisible by {experts} experts",
                                bound.tensor.nbytes()
                            );
                        }
                        let bytes = bound
                            .tensor
                            .nbytes()
                            .checked_div(experts)
                            .context("expert tensor has zero experts")?;
                        consider(bytes, bound.rows, bound.cols)?;
                    }
                }
            }
        }
        Ok(WeightMemory {
            checkpoint_bytes: self.checkpoint_bytes,
            largest_request_bytes,
            largest_matvec_elements,
        })
    }

    /// Conservative logical bytes used by greedy-decode buffers for this model.
    ///
    /// The bound includes output token/logit storage, doubled KV payload for possible
    /// vector growth, every layer scratch vector as if they all coexisted, one prior
    /// forward result during step replacement, and attention-score scratch. It excludes
    /// encoded weights and the backend's dequantized matrix scratch; those come from
    /// [`Self::weight_memory`]. It also excludes allocator/container overhead, tokenizer
    /// and GGUF indexes, thread stacks, and executable mappings, so a fit planner must
    /// add explicit runtime headroom rather than treating this as RSS.
    pub fn greedy_decode_buffer_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<u64> {
        if prompt_tokens == 0 {
            bail!("greedy decode memory requires at least one prompt token");
        }
        let final_tokens = prompt_tokens
            .checked_add(max_new_tokens)
            .context("decode token count overflows")?;
        let evaluated_tokens = final_tokens.saturating_sub(usize::from(max_new_tokens > 0));
        if evaluated_tokens > self.cfg.context {
            bail!(
                "decode memory needs {evaluated_tokens} evaluated positions, exceeding checkpoint context {}",
                self.cfg.context
            );
        }
        let chunk = prompt_tokens.max(1) as u128;
        let dim = self.cfg.dim as u128;
        let q_width = (self.cfg.n_heads as u128)
            .checked_mul(self.cfg.head_dim as u128)
            .context("query width overflows")?;
        let kv_width = (self.cfg.n_kv_heads as u128)
            .checked_mul(self.cfg.head_dim as u128)
            .context("KV width overflows")?;
        let product = |values: &[u128]| -> Result<u128> {
            values.iter().try_fold(1u128, |total, &value| {
                total
                    .checked_mul(value)
                    .context("decode buffer element count overflows")
            })
        };
        let mut bytes = 0u128;
        let mut add = |elements: u128, element_bytes: u128, copies: u128| -> Result<()> {
            let addition = elements
                .checked_mul(element_bytes)
                .and_then(|value| value.checked_mul(copies))
                .context("decode buffer byte count overflows")?;
            bytes = bytes
                .checked_add(addition)
                .context("total decode buffer bytes overflow")?;
            Ok(())
        };

        // Long-lived result and token vectors.
        add(final_tokens as u128, 4, 1)?; // tokens
        add(max_new_tokens as u128, 4, 1)?; // generated
        add(
            (max_new_tokens as u128)
                .checked_mul(self.cfg.vocab as u128)
                .context("decode logit count overflows")?,
            4,
            1,
        )?;

        // K and V for every evaluated position and layer. Count twice again because a
        // growing Vec may briefly need its old and new allocations during reallocation.
        let kv_cache_elements = (evaluated_tokens as u128)
            .checked_mul(self.cfg.n_layers as u128)
            .and_then(|value| value.checked_mul(kv_width))
            .context("KV cache element count overflows")?;
        add(kv_cache_elements, 4, 4)?; // K + V, doubled for growth

        // One layer's deliberately over-counted f32 working vectors.
        add(product(&[chunk, dim])?, 4, 4)?; // hidden, attention input, FFN input, MoE output
        add(product(&[chunk, q_width])?, 4, 3)?; // q, q-normalised, attended
        add(product(&[chunk, kv_width])?, 4, 4)?; // k, v, k-normalised, cache append
        add(dim, 4, 2)?; // attention projection and final-normalised row
        add(product(&[chunk, self.cfg.n_experts as u128])?, 4, 1)?; // router probabilities
        add(product(&[chunk, self.cfg.n_experts_used as u128])?, 4, 2)?; // selected weights plus transient selection weights
        add(self.cfg.expert_ff as u128, 4, 2)?; // gate and up
        add(dim, 4, 3)?; // attention, FFN and output norm vectors
        add(self.cfg.head_dim as u128, 4, 2)?; // query and key norm vectors
        add(self.cfg.head_dim as u128, 4, 1)?; // cosine + sine combined
        add(self.cfg.head_dim as u128 / 2, 8, 1)?; // f64 frequencies
        add(self.cfg.vocab as u128, 4, 1)?; // current logits
        add(evaluated_tokens as u128, 4, 1)?; // attention scores

        // `output = forward_cached_chunk(...)` evaluates the new result before dropping
        // the prior one, so retain one extra hidden/logit result conservatively.
        add(product(&[chunk, dim])?, 4, 1)?;
        add(self.cfg.vocab as u128, 4, 1)?;

        // Selected expert IDs are `usize` rather than f32.
        add(
            product(&[chunk, self.cfg.n_experts_used as u128])?,
            std::mem::size_of::<usize>() as u128,
            2,
        )?; // retained plus transient selection IDs
        add(
            self.cfg.n_experts as u128,
            std::mem::size_of::<usize>() as u128,
            1,
        )?; // stable-top-k sort indices retain expert-count capacity

        u64::try_from(bytes).context("decode buffer byte bound does not fit u64")
    }

    /// Conservative logical bytes used by [`Qwen3Moe::generate_greedy`] buffers.
    ///
    /// This has the same scope and context checks as
    /// [`Qwen3Moe::greedy_decode_buffer_bytes`], except that incremental generation does
    /// not retain `max_new_tokens * vocabulary` logits. The decode bound's full token-ID
    /// allowance remains in this estimate to cover the generated IDs plus transient
    /// [`crate::ForwardOutput`] token vectors conservatively. Callback-owned storage is
    /// outside the model allocation and must be budgeted by its caller.
    pub fn greedy_generate_buffer_bytes(
        &self,
        prompt_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<u64> {
        let decode_bytes = self.greedy_decode_buffer_bytes(prompt_tokens, max_new_tokens)?;
        let retained_logits_bytes = (max_new_tokens as u128)
            .checked_mul(self.cfg.vocab as u128)
            .and_then(|elements| elements.checked_mul(std::mem::size_of::<f32>() as u128))
            .context("generation logit-history byte count overflows")?;
        let retained_logits_bytes = u64::try_from(retained_logits_bytes)
            .context("generation logit-history bytes do not fit u64")?;
        decode_bytes
            .checked_sub(retained_logits_bytes)
            .context("generation buffer bound exceeds decode buffer bound")
    }

    /// Bind every tensor, checking each against the shape the config implies.
    ///
    /// Shape validation is the point of this pass. A tensor that is present but the
    /// wrong shape is the failure mode worth catching here rather than at the first
    /// matvec, and a tensor that is merely *absent* would otherwise read as zeros and
    /// still generate text.
    pub fn load(gguf: &Gguf) -> Result<Qwen3Moe> {
        Self::load_profile(gguf, FeedForwardProfile::Routed)
    }

    fn load_profile(gguf: &Gguf, profile: FeedForwardProfile) -> Result<Qwen3Moe> {
        let cfg = Config::from_gguf_profile(gguf, profile)?;
        let arch = profile.architecture();
        let checkpoint_bytes = gguf.names().try_fold(0u64, |total, name| {
            total
                .checked_add(
                    gguf.get(name)
                        .expect("a name from the GGUF index resolves")
                        .nbytes(),
                )
                .context("total encoded checkpoint bytes overflow")
        })?;
        for name in gguf.names() {
            if !supports_tensor_name(name, cfg.n_layers, profile) {
                bail!("unsupported {arch} tensor: {name}");
            }
        }
        let required_tensors = cfg
            .n_layers
            .checked_mul(match profile {
                FeedForwardProfile::Dense => 11,
                FeedForwardProfile::Routed => 12,
            })
            .and_then(|count| count.checked_add(2))
            .with_context(|| format!("{arch} tensor count overflows"))?;
        let untied_tensors = required_tensors
            .checked_add(1)
            .with_context(|| format!("{arch} tensor count overflows"))?;
        let has_output = gguf.get("output.weight").is_some();
        let expected_tensors = if has_output {
            untied_tensors
        } else {
            required_tensors
        };
        if gguf.len() != expected_tensors {
            bail!(
                "{arch} with {} blocks needs exactly {required_tensors} tensors when output is tied to token embeddings, or {} with output.weight; checkpoint has {}",
                cfg.n_layers,
                untied_tensors,
                gguf.len()
            );
        }
        let d = cfg.dim;
        let q_out = cfg
            .n_heads
            .checked_mul(cfg.head_dim)
            .context("query projection width overflows")?;
        let kv_out = cfg
            .n_kv_heads
            .checked_mul(cfg.head_dim)
            .context("key/value projection width overflows")?;

        // `bind` reads a tensor and asserts its shape. GGUF stores dims fastest-first,
        // so dims[0] is the column count and dims[1] the row count.
        let bind = |name: &str, rows: usize, cols: usize| -> Result<Bound> {
            let t = gguf
                .get(name)
                .with_context(|| format!("missing tensor {name}"))?
                .clone();
            let host_dim = |value: u64| -> Result<usize> {
                usize::try_from(value)
                    .with_context(|| format!("{name} dimension {value} does not fit this host"))
            };
            let (got_cols, got_rows) = match t.dims() {
                [cols] => (host_dim(*cols)?, 1usize),
                [cols, rows] => (host_dim(*cols)?, host_dim(*rows)?),
                _ => bail!(
                    "{name}: expected a 1-D or 2-D tensor, got dims {:?}",
                    t.dims()
                ),
            };
            if got_cols != cols || got_rows != rows {
                bail!(
                    "{name}: expected {rows}x{cols} (rows x cols), checkpoint has dims {:?}",
                    t.dims()
                );
            }
            validate_element_count(name, rows, cols, 1)?;
            Ok(Bound {
                tensor: t,
                rows,
                cols,
            })
        };
        // Experts are a 3-D tensor: [cols, rows, n_experts].
        let bind_exps = |name: &str, rows: usize, cols: usize| -> Result<Bound> {
            let t = gguf
                .get(name)
                .with_context(|| format!("missing tensor {name}"))?
                .clone();
            if t.dims().len() != 3 {
                bail!(
                    "{name}: expected a 3-D expert tensor, got dims {:?}",
                    t.dims()
                );
            }
            let got_cols = usize::try_from(t.dims()[0])
                .with_context(|| format!("{name} column count does not fit this host"))?;
            let got_rows = usize::try_from(t.dims()[1])
                .with_context(|| format!("{name} row count does not fit this host"))?;
            let got_experts = usize::try_from(t.dims()[2])
                .with_context(|| format!("{name} expert count does not fit this host"))?;
            if got_cols != cols || got_rows != rows || got_experts != cfg.n_experts {
                bail!(
                    "{name}: expected [{cols}, {rows}, {}], got {:?}",
                    cfg.n_experts,
                    t.dims()
                );
            }
            validate_element_count(name, rows, cols, cfg.n_experts)?;
            Ok(Bound {
                tensor: t,
                rows,
                cols,
            })
        };

        let mut layers = Vec::new();
        layers.try_reserve_exact(cfg.n_layers).map_err(|error| {
            anyhow::anyhow!(
                "cannot allocate {arch} layer index for {} blocks: {error}",
                cfg.n_layers
            )
        })?;
        for l in 0..cfg.n_layers {
            let p = |s: &str| format!("blk.{l}.{s}");
            let ffn = match profile {
                FeedForwardProfile::Dense => FeedForward::Dense {
                    gate: bind(&p("ffn_gate.weight"), cfg.expert_ff, d)?,
                    up: bind(&p("ffn_up.weight"), cfg.expert_ff, d)?,
                    down: bind(&p("ffn_down.weight"), d, cfg.expert_ff)?,
                },
                FeedForwardProfile::Routed => FeedForward::Routed {
                    router: bind(&p("ffn_gate_inp.weight"), cfg.n_experts, d)?,
                    gate_exps: bind_exps(&p("ffn_gate_exps.weight"), cfg.expert_ff, d)?,
                    up_exps: bind_exps(&p("ffn_up_exps.weight"), cfg.expert_ff, d)?,
                    down_exps: bind_exps(&p("ffn_down_exps.weight"), d, cfg.expert_ff)?,
                },
            };
            layers.push(Layer {
                attn_norm: bind(&p("attn_norm.weight"), 1, d)?,
                wq: bind(&p("attn_q.weight"), q_out, d)?,
                wk: bind(&p("attn_k.weight"), kv_out, d)?,
                wv: bind(&p("attn_v.weight"), kv_out, d)?,
                q_norm: bind(&p("attn_q_norm.weight"), 1, cfg.head_dim)?,
                k_norm: bind(&p("attn_k_norm.weight"), 1, cfg.head_dim)?,
                wo: bind(&p("attn_output.weight"), d, q_out)?,
                ffn_norm: bind(&p("ffn_norm.weight"), 1, d)?,
                ffn,
            });
        }

        let embed = bind("token_embd.weight", cfg.vocab, d)?;
        let output = if has_output {
            bind("output.weight", cfg.vocab, d)?
        } else {
            // Canonical Qwen GGUFs omit `output.weight` when logits share the token
            // embedding matrix. Keep the same immutable locator rather than inventing
            // a duplicate read or rejecting a valid tied checkpoint.
            embed.clone()
        };

        Ok(Qwen3Moe {
            checkpoint_id: gguf.checkpoint_id(),
            embed,
            out_norm: bind("output_norm.weight", 1, d)?,
            output,
            cfg,
            layers,
            checkpoint_bytes,
        })
    }

    /// Byte offset and length of ONE expert inside a 3-D expert tensor.
    ///
    /// Experts are contiguous slabs, so this is arithmetic rather than a lookup — which
    /// is what makes streaming one expert a single positional read with no index to
    /// consult, and no six-way scatter to coalesce.
    fn expert_slice(&self, b: &Bound, expert: usize) -> Result<(u64, u64)> {
        if expert >= self.cfg.n_experts {
            bail!(
                "expert index {expert} is outside checkpoint expert count {}",
                self.cfg.n_experts
            );
        }
        let (elems, bytes) = ggml_block_geometry(b.tensor.kind().0).with_context(|| {
            format!(
                "cannot compute expert stride for {}",
                b.tensor.kind().name()
            )
        })?;
        if !b.cols.is_multiple_of(elems) {
            bail!(
                "expert row width {} is not a whole number of {elems}-element blocks",
                b.cols
            );
        }
        let stride = b
            .rows
            .checked_mul(b.cols / elems)
            .and_then(|blocks| blocks.checked_mul(bytes))
            .context("expert encoded byte stride overflows")?;
        let stride = u64::try_from(stride).context("expert stride does not fit u64")?;
        let expert = u64::try_from(expert).context("expert index does not fit u64")?;
        let relative = stride
            .checked_mul(expert)
            .context("expert byte offset overflows")?;
        let relative_end = relative
            .checked_add(stride)
            .context("expert byte range overflows")?;
        if relative_end > b.tensor.nbytes() {
            bail!(
                "expert byte range {relative}..{relative_end} is outside tensor span {}",
                b.tensor.nbytes()
            );
        }
        let absolute = b
            .tensor
            .offset()
            .checked_add(relative)
            .context("expert absolute byte offset overflows")?;
        absolute
            .checked_add(stride)
            .context("expert absolute byte range overflows")?;
        Ok((absolute, stride))
    }

    /// Absolute offset and encoded stride for a layer's gate expert.
    pub fn gate_expert_slice(&self, layer: usize, expert: usize) -> Result<(u64, u64)> {
        let layer = self
            .layers
            .get(layer)
            .with_context(|| format!("layer {layer} is outside {} layers", self.layers.len()))?;
        match &layer.ffn {
            FeedForward::Routed { gate_exps, .. } => self.expert_slice(gate_exps, expert),
            FeedForward::Dense { .. } => bail!("dense qwen3 has no routed expert slices"),
        }
    }

    /// Total bytes of routed-expert weight in the checkpoint.
    ///
    /// The number that decides whether a model streams or fits, so it is reported rather
    /// than estimated.
    pub fn expert_bytes(&self) -> Result<u64> {
        let mut total = 0u64;
        for layer in &self.layers {
            let FeedForward::Routed {
                gate_exps,
                up_exps,
                down_exps,
                ..
            } = &layer.ffn
            else {
                bail!("dense qwen3 has no routed-expert byte total");
            };
            for tensor in [&gate_exps.tensor, &up_exps.tensor, &down_exps.tensor] {
                total = total
                    .checked_add(tensor.nbytes())
                    .context("total routed-expert bytes overflow")?;
            }
        }
        Ok(total)
    }
}

fn validate_element_count(name: &str, rows: usize, cols: usize, copies: usize) -> Result<()> {
    rows.checked_mul(cols)
        .and_then(|count| count.checked_mul(copies))
        .with_context(|| format!("{name}: element count overflows"))?;
    Ok(())
}

/// Whether a tensor belongs to the selected exact bias-free dense or routed profile.
fn supports_tensor_name(name: &str, n_layers: usize, profile: FeedForwardProfile) -> bool {
    if matches!(
        name,
        "token_embd.weight" | "output_norm.weight" | "output.weight"
    ) {
        return true;
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((layer, suffix)) = rest.split_once('.') else {
        return false;
    };
    let Ok(layer_index) = layer.parse::<usize>() else {
        return false;
    };
    if layer != layer_index.to_string() || layer_index >= n_layers {
        return false;
    }
    let common = matches!(
        suffix,
        "attn_norm.weight"
            | "attn_q.weight"
            | "attn_k.weight"
            | "attn_v.weight"
            | "attn_q_norm.weight"
            | "attn_k_norm.weight"
            | "attn_output.weight"
            | "ffn_norm.weight"
    );
    common
        || match profile {
            FeedForwardProfile::Dense => {
                matches!(
                    suffix,
                    "ffn_gate.weight" | "ffn_up.weight" | "ffn_down.weight"
                )
            }
            FeedForwardProfile::Routed => matches!(
                suffix,
                "ffn_gate_inp.weight"
                    | "ffn_gate_exps.weight"
                    | "ffn_up_exps.weight"
                    | "ffn_down_exps.weight"
            ),
        }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_size_is_heads_over_kv_heads() {
        let cfg = Config {
            n_layers: 48,
            dim: 2048,
            n_heads: 32,
            n_kv_heads: 4,
            head_dim: 128,
            value_head_dim: 128,
            n_experts: 128,
            n_experts_used: 8,
            expert_ff: 768,
            shared_expert_ff: 0,
            vocab: 151936,
            eps: 1e-6,
            rope_base: 1e7,
            context: 262144,
        };
        assert_eq!(cfg.group(), 8);
        // 32 heads over 4 kv heads is 8 queries sharing each key/value pair.
        assert_eq!(cfg.n_heads, cfg.group() * cfg.n_kv_heads);
    }

    #[test]
    fn architecture_metadata_whitelist_excludes_unimplemented_semantics() {
        for unsupported in [
            "rope.scaling.type",
            "rope.scale_linear",
            "expert_weights_scale",
            "expert_weights_norm",
            "expert_gating_func",
            "swiglu_clamp_exp",
            "attention.sliding_window",
        ] {
            assert!(
                !supports_arch_metadata(unsupported, FeedForwardProfile::Routed),
                "{unsupported} must remain an explicit refusal until implemented"
            );
        }
    }

    #[test]
    fn tensor_manifest_rejects_unimplemented_semantic_branches() {
        for supported in [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_q.weight",
            "blk.47.ffn_down_exps.weight",
        ] {
            assert!(
                supports_tensor_name(supported, 48, FeedForwardProfile::Routed),
                "{supported}"
            );
        }
        for unsupported in [
            "blk.0.attn_q.bias",
            "blk.0.attn_output.bias",
            "blk.0.ffn_gate_inp.bias",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_gate_shexp.weight",
            "blk.0.ffn_up_shexp.weight",
            "blk.0.ffn_down_shexp.weight",
            "blk.48.attn_q.weight",
            "blk.00.attn_q.weight",
        ] {
            assert!(
                !supports_tensor_name(unsupported, 48, FeedForwardProfile::Routed),
                "{unsupported} must be refused until its semantics are implemented"
            );
        }
    }

    #[test]
    fn dense_and_routed_manifests_are_disjoint() {
        for dense in [
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
        ] {
            assert!(supports_tensor_name(dense, 1, FeedForwardProfile::Dense));
            assert!(!supports_tensor_name(dense, 1, FeedForwardProfile::Routed));
        }
        for routed in [
            "blk.0.ffn_gate_inp.weight",
            "blk.0.ffn_gate_exps.weight",
            "blk.0.ffn_up_exps.weight",
            "blk.0.ffn_down_exps.weight",
        ] {
            assert!(supports_tensor_name(routed, 1, FeedForwardProfile::Routed));
            assert!(!supports_tensor_name(routed, 1, FeedForwardProfile::Dense));
        }
        assert!(!supports_arch_metadata(
            "expert_count",
            FeedForwardProfile::Dense
        ));
        assert!(supports_arch_metadata(
            "expert_count",
            FeedForwardProfile::Routed
        ));
    }
}
