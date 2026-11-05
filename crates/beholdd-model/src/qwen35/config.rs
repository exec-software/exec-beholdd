//! Header-only binding for dense `qwen35` checkpoints with hybrid gated attention.
//!
//! This describes the OpenMythos/Qwen3.6-27B GGUF graph; it does not establish an
//! executable backend. Full attention and gated DeltaNet have disjoint manifests.
//! Unknown architecture metadata, extra tensors, and unsupported storage fail before
//! any weight payload is read. The existing `qwen3` dispatcher is deliberately separate.

use std::collections::BTreeSet;

use anyhow::{ensure, Context, Result};
use beholdd_core::ggml_block_geometry;
use beholdd_format::gguf::Value;
use beholdd_format::{CheckpointId, Gguf, GgufTensor};

/// Exact GGUF architecture tag; `qwen3`, `qwen35moe`, and `qwen3next` are not aliases.
pub const ARCHITECTURE: &str = "qwen35";

/// Semantic dimensions of the dense hybrid graph, in host element counts.
///
/// These values are metadata, not allocation or execution permission. In particular,
/// the checkpoint context is not a promise that a backend can allocate that cache.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Total number of hybrid blocks.
    pub n_layers: usize,
    /// Residual-stream width.
    pub dim: usize,
    /// Dense SwiGLU intermediate width.
    pub feed_forward: usize,
    /// Full-attention query heads; each has a separate projected output gate.
    pub n_heads: usize,
    /// Full-attention key/value heads.
    pub n_kv_heads: usize,
    /// Full-attention query/key width per head.
    pub key_head_dim: usize,
    /// Full-attention value width per head.
    pub value_head_dim: usize,
    /// Rotated prefix of each full-attention query/key head.
    pub rope_dim: usize,
    /// GGUF multimodal RoPE section lengths, in pairs. Text positions share one axis.
    pub rope_sections: [usize; 4],
    /// Unscaled RoPE base.
    pub rope_base: f64,
    /// RMSNorm epsilon, including per-head and gated DeltaNet normalization.
    pub eps: f32,
    /// Checkpoint-declared maximum positions.
    pub context: usize,
    /// Every Nth block is full attention, using one-based block numbering.
    pub full_attention_interval: usize,
    /// Number of causal convolution taps in each DeltaNet block.
    pub ssm_conv_kernel: usize,
    /// DeltaNet query/key heads (`ssm.group_count`).
    pub ssm_group_count: usize,
    /// Concatenated DeltaNet value-head width (`ssm.inner_size`).
    pub ssm_inner_size: usize,
    /// DeltaNet query/key width per head (`ssm.state_size`).
    pub ssm_state_size: usize,
    /// DeltaNet value heads (`ssm.time_step_rank`), not an independent low-rank factor.
    pub ssm_time_step_rank: usize,
    /// Token rows in both the embedding and the required untied output projection.
    pub vocab: usize,
}

impl Config {
    /// Validate semantic metadata without reading any tensor payload.
    pub fn from_gguf(g: &Gguf) -> Result<Self> {
        ensure!(
            g.arch() == Some(ARCHITECTURE),
            "qwen35 binder requires general.architecture=qwen35"
        );
        for key in g.metadata().keys().filter(|key| key.starts_with("qwen35.")) {
            ensure!(
                supports_metadata(&key[7..]),
                "unsupported qwen35 architecture metadata: {key}"
            );
        }
        if let Some(kind) = g.metadata().get("general.type") {
            ensure!(
                kind.as_str() == Some("model"),
                "qwen35 requires general.type=model when declared"
            );
        }
        let need = |key: &str| -> Result<usize> {
            let value = g
                .kv(key)
                .and_then(Value::as_u64)
                .with_context(|| format!("qwen35.{key} must be an unsigned integer"))?;
            usize::try_from(value).with_context(|| format!("qwen35.{key} does not fit this host"))
        };
        let float = |key: &str| -> Result<f64> {
            g.kv(key)
                .and_then(Value::as_f64)
                .with_context(|| format!("qwen35.{key} must be numeric"))
        };
        let sections = g
            .kv("rope.dimension_sections")
            .and_then(Value::as_arr)
            .context("qwen35.rope.dimension_sections must be a four-element integer array")?;
        ensure!(
            sections.len() == 4,
            "qwen35.rope.dimension_sections must contain four elements"
        );
        let mut rope_sections = [0; 4];
        for (index, value) in sections.iter().enumerate() {
            rope_sections[index] = usize::try_from(value.as_u64().with_context(|| {
                format!("qwen35.rope.dimension_sections[{index}] must be unsigned")
            })?)
            .context("qwen35 RoPE section does not fit this host")?;
        }
        let embedding = g
            .get("token_embd.weight")
            .context("missing tensor token_embd.weight")?;
        ensure!(
            embedding.dims().len() == 2,
            "token_embd.weight must be two-dimensional"
        );
        let config = Self {
            n_layers: need("block_count")?,
            dim: need("embedding_length")?,
            feed_forward: need("feed_forward_length")?,
            n_heads: need("attention.head_count")?,
            n_kv_heads: need("attention.head_count_kv")?,
            key_head_dim: need("attention.key_length")?,
            value_head_dim: need("attention.value_length")?,
            rope_dim: need("rope.dimension_count")?,
            rope_sections,
            rope_base: float("rope.freq_base")?,
            eps: float("attention.layer_norm_rms_epsilon")? as f32,
            context: need("context_length")?,
            full_attention_interval: need("full_attention_interval")?,
            ssm_conv_kernel: need("ssm.conv_kernel")?,
            ssm_group_count: need("ssm.group_count")?,
            ssm_inner_size: need("ssm.inner_size")?,
            ssm_state_size: need("ssm.state_size")?,
            ssm_time_step_rank: need("ssm.time_step_rank")?,
            vocab: usize::try_from(embedding.dims()[1])
                .context("qwen35 vocabulary does not fit this host")?,
        };
        config.validate()?;
        ensure!(
            embedding.dims()[0] == config.dim as u64,
            "token_embd.weight input width differs from qwen35.embedding_length"
        );
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        // Binder bounds prevent hostile metadata from growing unbounded manifest/state
        // arithmetic. They do not replace a backend's separate memory admission.
        ensure!(
            (1..=256).contains(&self.n_layers),
            "qwen35.block_count must be 1..=256"
        );
        ensure!(
            (1..=65536).contains(&self.dim) && (1..=262144).contains(&self.feed_forward),
            "qwen35 hidden or FFN width is outside the binder bounds"
        );
        ensure!(
            self.dim.is_multiple_of(256) && self.feed_forward.is_multiple_of(256),
            "qwen35 Q4_K/Q6_K hidden and FFN widths must contain whole 256-element blocks"
        );
        ensure!(
            (1..=1048576).contains(&self.vocab) && (1..=16777216).contains(&self.context),
            "qwen35 vocabulary or context is outside the binder bounds"
        );
        ensure!(
            (1..=1024).contains(&self.n_heads)
                && (1..=self.n_heads).contains(&self.n_kv_heads)
                && self.n_heads.is_multiple_of(self.n_kv_heads),
            "qwen35 attention heads must group evenly"
        );
        ensure!(
            (1..=1024).contains(&self.key_head_dim) && self.key_head_dim == self.value_head_dim,
            "qwen35 requires equal positive key/value head widths at most 1024"
        );
        ensure!(
            self.rope_dim > 0
                && self.rope_dim <= self.key_head_dim
                && self.rope_dim.is_multiple_of(2),
            "qwen35 rotated head prefix must be positive, even, and within key width"
        );
        let section_sum = self
            .rope_sections
            .iter()
            .try_fold(0usize, |sum, n| sum.checked_add(*n))
            .context("qwen35 RoPE section sum overflows")?;
        ensure!(section_sum.checked_mul(2) == Some(self.rope_dim) && self.rope_sections[3] == 0
            && self.rope_sections[..3].iter().all(|n| *n > 0),
            "qwen35 RoPE sections must cover the rotated pairs with three positive sections and zero fourth section");
        ensure!(
            self.rope_base.is_finite() && self.rope_base > 0.0,
            "qwen35 RoPE base must be positive and finite"
        );
        ensure!(
            self.eps.is_finite() && self.eps > 0.0,
            "qwen35 RMSNorm epsilon must be positive and finite"
        );
        ensure!(self.full_attention_interval >= 2 && self.full_attention_interval <= self.n_layers
            && self.n_layers.is_multiple_of(self.full_attention_interval),
            "qwen35 full_attention_interval must partition complete hybrid groups and be at least two");
        ensure!(
            self.ssm_conv_kernel == 4,
            "qwen35 currently binds exactly four causal convolution taps"
        );
        ensure!(
            (1..=1024).contains(&self.ssm_group_count)
                && (1..=1024).contains(&self.ssm_time_step_rank)
                && self.ssm_time_step_rank.is_multiple_of(self.ssm_group_count),
            "qwen35 DeltaNet value heads must be an integer multiple of query/key heads"
        );
        ensure!(
            (1..=1024).contains(&self.ssm_state_size),
            "qwen35 DeltaNet key head width must be 1..=1024"
        );
        ensure!(
            self.ssm_inner_size
                == checked_product(
                    self.ssm_state_size,
                    self.ssm_time_step_rank,
                    "DeltaNet value width"
                )?,
            "qwen35 DeltaNet value head width must equal its key head width"
        );
        ensure!(
            self.ssm_inner_size.is_multiple_of(256),
            "qwen35 DeltaNet output matrix needs whole 256-element blocks"
        );
        checked_product(self.n_heads, self.key_head_dim, "attention query width")?;
        checked_product(self.n_kv_heads, self.key_head_dim, "attention KV width")?;
        self.delta_qkv_width_checked()?;
        Ok(())
    }

    /// Whether zero-based `index` denotes a full-attention block.
    pub fn is_full_attention(&self, index: usize) -> Result<bool> {
        ensure!(
            index < self.n_layers,
            "qwen35 block index {index} is out of range"
        );
        Ok((index + 1).is_multiple_of(self.full_attention_interval))
    }

    /// Concatenated query/key width for one DeltaNet projection branch.
    pub fn delta_key_width(&self) -> usize {
        self.ssm_group_count * self.ssm_state_size
    }

    /// Query, key, then value concatenated projection width, before causal convolution.
    pub fn delta_qkv_width(&self) -> usize {
        2 * self.delta_key_width() + self.ssm_inner_size
    }

    /// Width of one DeltaNet value head; validated equal to the key head width.
    pub fn delta_value_head_dim(&self) -> usize {
        self.ssm_inner_size / self.ssm_time_step_rank
    }

    fn delta_qkv_width_checked(&self) -> Result<usize> {
        checked_product(
            self.ssm_group_count,
            self.ssm_state_size,
            "DeltaNet key width",
        )?
        .checked_mul(2)
        .and_then(|n| n.checked_add(self.ssm_inner_size))
        .context("qwen35 DeltaNet QKV width overflows")
    }
}

fn supports_metadata(suffix: &str) -> bool {
    matches!(
        suffix,
        "block_count"
            | "context_length"
            | "embedding_length"
            | "feed_forward_length"
            | "attention.head_count"
            | "attention.head_count_kv"
            | "attention.key_length"
            | "attention.value_length"
            | "attention.layer_norm_rms_epsilon"
            | "full_attention_interval"
            | "rope.dimension_count"
            | "rope.dimension_sections"
            | "rope.freq_base"
            | "ssm.conv_kernel"
            | "ssm.group_count"
            | "ssm.inner_size"
            | "ssm.state_size"
            | "ssm.time_step_rank"
    )
}

fn checked_product(a: usize, b: usize, name: &str) -> Result<usize> {
    a.checked_mul(b)
        .with_context(|| format!("qwen35 {name} overflows"))
}

/// Dense SwiGLU matrices; dimensions remain GGUF fastest-varying first.
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// Gate projection from residual width to FFN width.
    pub gate: GgufTensor,
    /// Up projection from residual width to FFN width.
    pub up: GgufTensor,
    /// Down projection from FFN width to residual width.
    pub down: GgufTensor,
}

/// Gated full attention. The doubled Q projection contains both Q and its output gate.
#[derive(Debug, Clone)]
pub struct FullAttention {
    /// Per-head interleaved query and gate projection: `[Q_head, gate_head]` for
    /// each head. Output width is `2 * heads * key_head_dim`, not all Q then all gates.
    pub query_gate: GgufTensor,
    /// Key projection.
    pub key: GgufTensor,
    /// Value projection.
    pub value: GgufTensor,
    /// Query per-head normalization weights.
    pub query_norm: GgufTensor,
    /// Key per-head normalization weights.
    pub key_norm: GgufTensor,
    /// Projection of the gated attended value into the residual stream.
    pub output: GgufTensor,
}

/// Gated DeltaNet projections and recurrent/convolution parameters.
#[derive(Debug, Clone)]
pub struct DeltaNet {
    /// Query, key, value projection before causal convolution.
    pub qkv: GgufTensor,
    /// SiLU output gate, one value per concatenated value-head element.
    pub gate: GgufTensor,
    /// Per-value-head decay input projection.
    pub alpha: GgufTensor,
    /// Per-value-head update input projection.
    pub beta: GgufTensor,
    /// F32 four-tap causal convolution, GGUF dimensions `[tap, channel]`.
    pub conv: GgufTensor,
    /// F32 per-value-head `-exp(A_log)`, already transformed by GGUF conversion.
    /// Stored as `ssm_a` without a suffix; do not apply the conversion a second time.
    pub a: GgufTensor,
    /// F32 per-value-head time-step bias.
    pub dt_bias: GgufTensor,
    /// F32 gated normalization weights, one value-head wide.
    pub norm: GgufTensor,
    /// Projection from concatenated DeltaNet values to the residual stream.
    pub output: GgufTensor,
}

/// The two disjoint attention manifests in a dense `qwen35` model.
#[derive(Debug, Clone)]
pub enum Attention {
    /// Gated grouped-query full attention with partial RoPE.
    Full(FullAttention),
    /// Gated DeltaNet with causal convolution and persistent recurrent state.
    DeltaNet(DeltaNet),
}

/// One checked hybrid block; every field is an immutable checkpoint locator.
#[derive(Debug, Clone)]
pub struct Block {
    /// Pre-attention normalization.
    pub attention_norm: GgufTensor,
    /// Post-attention/pre-FFN normalization.
    pub post_attention_norm: GgufTensor,
    /// Block-specific full attention or DeltaNet parameters.
    pub attention: Attention,
    /// Dense SwiGLU parameters.
    pub ffn: FeedForward,
}

/// Complete validated manifest. Constructing this object performs no weight reads.
#[derive(Debug)]
pub struct Binding {
    config: Config,
    checkpoint_id: CheckpointId,
    checkpoint_bytes: u64,
    embedding: GgufTensor,
    output_norm: GgufTensor,
    output: GgufTensor,
    blocks: Vec<Block>,
}

impl Binding {
    /// Bind the complete untied dense Q4_K/Q6_K + F32 hybrid manifest.
    pub fn load(g: &Gguf) -> Result<Self> {
        let config = Config::from_gguf(g)?;
        let d = config.dim;
        let full_count = config.n_layers / config.full_attention_interval;
        let expected = (config.n_layers - full_count) * 14 + full_count * 11 + 3;
        ensure!(
            g.len() == expected,
            "qwen35 requires exactly {expected} tensors for this hybrid manifest; found {}",
            g.len()
        );
        let mut consumed = BTreeSet::new();
        let mut bind = |name: &str, dims: &[usize], matrix: bool| -> Result<GgufTensor> {
            let tensor = g
                .get(name)
                .with_context(|| format!("missing qwen35 tensor {name}"))?;
            let wanted: Vec<u64> = dims.iter().map(|n| *n as u64).collect();
            ensure!(
                tensor.dims() == wanted,
                "{name}: expected GGUF dimensions {wanted:?}, found {:?}",
                tensor.dims()
            );
            let allowed = if matrix {
                matches!(tensor.kind().0, 12 | 14)
            } else {
                tensor.kind().0 == 0
            };
            ensure!(
                allowed,
                "{name}: expected {}, found {}",
                if matrix {
                    "Q4_K or Q6_K matrix"
                } else {
                    "F32 parameter"
                },
                tensor.kind().name()
            );
            let (block, bytes) = ggml_block_geometry(tensor.kind().0)
                .context("qwen35 tensor has unknown storage geometry")?;
            ensure!(
                dims[0].is_multiple_of(block),
                "{name}: encoded rows must contain whole quantization blocks"
            );
            let elements = dims
                .iter()
                .try_fold(1usize, |total, n| total.checked_mul(*n))
                .with_context(|| format!("{name}: element count overflows"))?;
            let encoded = (elements / block)
                .checked_mul(bytes)
                .context("qwen35 tensor byte count overflows")?;
            ensure!(
                tensor.nbytes() == u64::try_from(encoded)?,
                "{name}: encoded byte count disagrees with its shape"
            );
            ensure!(
                consumed.insert(name.to_owned()),
                "qwen35 tensor {name} was bound twice"
            );
            Ok(tensor.clone())
        };
        let embedding = bind("token_embd.weight", &[d, config.vocab], true)?;
        let output_norm = bind("output_norm.weight", &[d], false)?;
        // This architecture's inspected manifest has a distinct output head. Tied output
        // is a separate semantic variant and must not be invented when a tensor is absent.
        let output = bind("output.weight", &[d, config.vocab], true)?;
        let mut blocks = Vec::new();
        blocks
            .try_reserve_exact(config.n_layers)
            .context("allocating qwen35 block manifest")?;
        let q_width = checked_product(config.n_heads, config.key_head_dim, "query width")?;
        let kv_width = checked_product(config.n_kv_heads, config.key_head_dim, "KV width")?;
        for index in 0..config.n_layers {
            let name = |suffix: &str| format!("blk.{index}.{suffix}");
            let attention_norm = bind(&name("attn_norm.weight"), &[d], false)?;
            let post_attention_norm = bind(&name("post_attention_norm.weight"), &[d], false)?;
            let ffn = FeedForward {
                gate: bind(&name("ffn_gate.weight"), &[d, config.feed_forward], true)?,
                up: bind(&name("ffn_up.weight"), &[d, config.feed_forward], true)?,
                down: bind(&name("ffn_down.weight"), &[config.feed_forward, d], true)?,
            };
            let attention = if config.is_full_attention(index)? {
                Attention::Full(FullAttention {
                    query_gate: bind(
                        &name("attn_q.weight"),
                        &[d, checked_product(q_width, 2, "gated query width")?],
                        true,
                    )?,
                    key: bind(&name("attn_k.weight"), &[d, kv_width], true)?,
                    value: bind(&name("attn_v.weight"), &[d, kv_width], true)?,
                    query_norm: bind(&name("attn_q_norm.weight"), &[config.key_head_dim], false)?,
                    key_norm: bind(&name("attn_k_norm.weight"), &[config.key_head_dim], false)?,
                    output: bind(&name("attn_output.weight"), &[q_width, d], true)?,
                })
            } else {
                Attention::DeltaNet(DeltaNet {
                    qkv: bind(
                        &name("attn_qkv.weight"),
                        &[d, config.delta_qkv_width()],
                        true,
                    )?,
                    gate: bind(&name("attn_gate.weight"), &[d, config.ssm_inner_size], true)?,
                    alpha: bind(
                        &name("ssm_alpha.weight"),
                        &[d, config.ssm_time_step_rank],
                        true,
                    )?,
                    beta: bind(
                        &name("ssm_beta.weight"),
                        &[d, config.ssm_time_step_rank],
                        true,
                    )?,
                    conv: bind(
                        &name("ssm_conv1d.weight"),
                        &[config.ssm_conv_kernel, config.delta_qkv_width()],
                        false,
                    )?,
                    a: bind(&name("ssm_a"), &[config.ssm_time_step_rank], false)?,
                    dt_bias: bind(&name("ssm_dt.bias"), &[config.ssm_time_step_rank], false)?,
                    norm: bind(
                        &name("ssm_norm.weight"),
                        &[config.delta_value_head_dim()],
                        false,
                    )?,
                    output: bind(&name("ssm_out.weight"), &[config.ssm_inner_size, d], true)?,
                })
            };
            blocks.push(Block {
                attention_norm,
                post_attention_norm,
                attention,
                ffn,
            });
        }
        for name in g.names() {
            ensure!(consumed.contains(name), "unsupported qwen35 tensor: {name}");
        }
        let checkpoint_bytes = g
            .names()
            .try_fold(0u64, |total, name| {
                total.checked_add(g.get(name).unwrap().nbytes())
            })
            .context("qwen35 checkpoint payload byte count overflows")?;
        Ok(Self {
            config,
            checkpoint_id: g.checkpoint_id(),
            checkpoint_bytes,
            embedding,
            output_norm,
            output,
            blocks,
        })
    }

    /// Validated architecture metadata.
    pub fn config(&self) -> &Config {
        &self.config
    }
    /// Checkpoint identity shared by every bound tensor.
    pub fn checkpoint_id(&self) -> CheckpointId {
        self.checkpoint_id
    }
    /// Sum of encoded tensor bytes, excluding GGUF header and alignment padding.
    pub fn checkpoint_bytes(&self) -> u64 {
        self.checkpoint_bytes
    }
    /// Token embedding locator.
    pub fn embedding(&self) -> &GgufTensor {
        &self.embedding
    }
    /// Final normalization locator.
    pub fn output_norm(&self) -> &GgufTensor {
        &self.output_norm
    }
    /// Untied language-model output projection locator.
    pub fn output(&self) -> &GgufTensor {
        &self.output
    }
    /// Hybrid blocks in execution order.
    pub fn blocks(&self) -> &[Block] {
        &self.blocks
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
pub(crate) mod tests;
