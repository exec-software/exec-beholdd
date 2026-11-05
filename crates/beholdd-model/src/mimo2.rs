//! Strict binder for `mimo2` checkpoints (MiMo-V2 family).
//!
//! A level-2 architecture of its own rather than a widening of the Qwen core, for the
//! reasons recorded in ADR-0002: values are narrower than keys, key/value head counts vary
//! per layer, a third of each head is rotated with one of two bases depending on the
//! layer's attention type, and routing is sigmoid with a scaling factor rather than
//! softmax top-k. The Qwen core already refuses `attention.sliding_window` and
//! `expert_gating_func` by name; this module is what those refusals were waiting for.
//!
//! This is gate G1 — binding only. Nothing here executes a block; `Config::from_gguf`
//! reads what the checkpoint declares and refuses, by name, whatever it cannot honour yet,
//! so that an unimplemented semantic is an error a person can act on rather than a wrong
//! number that decodes into plausible text.

use anyhow::{bail, Context, Result};
use beholdd_format::gguf::{Gguf, Value};

pub mod block;
pub mod oracle;

/// What binding needs from a checkpoint: its architecture tag, its architecture-scoped
/// metadata, and the shape of the embedding.
///
/// A seam rather than a direct `Gguf` dependency so that the refusals below can be tested
/// against transcribed headers. A `Gguf` is a file; the binder's job is arithmetic about
/// numbers in it, and that job should not require 156 GiB on disk to exercise.
pub trait Header {
    /// `general.architecture`.
    fn arch(&self) -> Option<&str>;
    /// A metadata value by its architecture-scoped suffix, e.g. `attention.head_count`.
    fn kv(&self, suffix: &str) -> Option<&Value>;
    /// Dimensions of `token_embd.weight`, fastest-varying first.
    fn embedding_dims(&self) -> Option<&[u64]>;
    /// Whether the checkpoint carries a tensor by this exact name.
    ///
    /// Still header-only: the tensor table is read with the metadata, and nothing about a
    /// block's payload is touched. It is here because two of MiMo's per-block facts are not
    /// in the metadata at all — whether a block routes and whether it has a sink are things
    /// the checkpoint states only by which tensors it carries.
    fn has_tensor(&self, name: &str) -> bool;
}

impl Header for Gguf {
    fn arch(&self) -> Option<&str> {
        Gguf::arch(self)
    }
    fn kv(&self, suffix: &str) -> Option<&Value> {
        Gguf::kv(self, suffix)
    }
    fn embedding_dims(&self) -> Option<&[u64]> {
        self.get("token_embd.weight").map(|t| t.dims())
    }
    fn has_tensor(&self, name: &str) -> bool {
        self.get(name).is_some()
    }
}

/// How a block attends: over everything, or over a fixed window with a sink bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Attention {
    /// Every position attends to every earlier position.
    Full,
    /// Each position attends to the last `window` positions, plus a learned sink.
    SlidingWindow,
}

/// How a block's feed-forward is built.
///
/// MiMo-V2.6-Flash is not uniformly routed: block 0 is a dense feed-forward of
/// `feed_forward` width and the other 47 are routed over `n_experts`. A loader that assumed
/// every block routes would look for experts block 0 does not have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ffn {
    /// One feed-forward of `Config::feed_forward` width, evaluated for every token.
    Dense,
    /// `n_experts_used` of `n_experts`, chosen per token.
    Routed {
        /// Whether a per-expert bias is added to the scores *before* the top-k choice.
        ///
        /// The bias steers which experts are picked without changing the weight a picked
        /// expert gets, so ignoring it routes to different experts while every shape still
        /// agrees — a wrong answer with nothing to catch it.
        selection_bias: bool,
    },
}

/// One block's shape. MiMo varies these per layer, which is the first reason this cannot be
/// a Qwen `Config` with extra fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Block {
    /// Which attention this block uses.
    pub attention: Attention,
    /// Key/value heads in this block. Query heads are constant across blocks.
    pub n_kv_heads: usize,
    /// How this block's feed-forward is built.
    pub ffn: Ffn,
    /// Whether this block carries a per-head attention sink.
    ///
    /// A learned logit appended to the row before the softmax and dropped after it, so a
    /// position can decline to attend and the remaining weights no longer sum to one. In the
    /// checkpoint we have, the 39 sliding-window blocks carry one and the 9 full blocks do
    /// not, but the operator does not depend on the attention type, so this is read rather
    /// than inferred from it.
    pub sink: bool,
}

/// Shapes and hyperparameters, read from the checkpoint rather than assumed.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Per-block attention shape, in block order. Its length is the block count.
    pub blocks: Vec<Block>,
    /// Hidden width.
    pub dim: usize,
    /// Query heads, constant across blocks.
    pub n_heads: usize,
    /// Per-head key width.
    pub key_head_dim: usize,
    /// Per-head value width. Narrower than the key width in every MiMo V2 checkpoint.
    pub value_head_dim: usize,
    /// Rotated dimensions per head. The rest of the head carries no position.
    pub rope_dim: usize,
    /// RoPE base for full-attention blocks.
    pub rope_base: f64,
    /// RoPE base for sliding-window blocks. MiMo uses a much smaller one.
    pub rope_base_swa: f64,
    /// Attention window for sliding-window blocks.
    pub window: usize,
    /// Scale applied to attention values.
    pub value_scale: f32,
    /// Routed experts per block.
    pub n_experts: usize,
    /// Experts evaluated per token.
    pub n_experts_used: usize,
    /// Inner width of one routed expert.
    pub expert_ff: usize,
    /// Inner width of the dense feed-forward, where a block has one.
    pub feed_forward: usize,
    /// Vocabulary size.
    pub vocab: usize,
    /// RMSNorm epsilon.
    pub eps: f32,
    /// Maximum positions the checkpoint claims.
    pub context: usize,
    /// Whether the chosen experts' weights are renormalised to sum to one.
    ///
    /// **True unless the checkpoint says otherwise, which is a default and is justified.**
    /// `config.json` for every published V2 checkpoint says `"norm_topk_prob": true`, and
    /// llama.cpp's converter does not emit `expert_weights_norm` for this architecture, so
    /// absence carries no information. Reading absence as "off" is what G4 caught: eight
    /// sigmoid scores sum to roughly four rather than one, the feed-forward speaks four times
    /// too loudly, and forty-seven layers of that leaves a model that writes fluent English
    /// and has forgotten that the capital of France is Paris.
    pub norm_topk: bool,
}

/// The architecture tag this module binds.
pub const ARCHITECTURE: &str = "mimo2";

/// `expert_gating_func = 2` is sigmoid. Softmax (1) is the Qwen core's and is not this.
const GATING_SIGMOID: u64 = 2;

impl Config {
    /// Read from GGUF metadata. Every semantic hyperparameter is required: a MiMo
    /// checkpoint that omits one is not a MiMo checkpoint we can execute, and guessing a
    /// default here is how an architecture quietly computes something else.
    pub fn from_header(g: &impl Header) -> Result<Config> {
        let arch = g.arch().context("checkpoint has no general.architecture")?;
        if arch != ARCHITECTURE {
            bail!("this loader is for {ARCHITECTURE}, checkpoint says {arch}");
        }
        refuse_unimplemented(g)?;

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
        let need_f64 = |k: &str| -> Result<f64> {
            g.kv(k)
                .and_then(|v| v.as_f64())
                .with_context(|| format!("missing {arch}.{k}"))
        };

        let n_layers = need_usize("block_count")?;
        let n_heads = need_usize("attention.head_count")?;
        let key_head_dim = need_usize("attention.key_length")?;
        let value_head_dim = need_usize("attention.value_length")?;
        let rope_dim = need_usize("rope.dimension_count")?;
        if rope_dim > key_head_dim {
            bail!("{arch}.rope.dimension_count={rope_dim} exceeds the {key_head_dim}-wide key head");
        }
        if rope_dim % 2 != 0 {
            bail!("{arch}.rope.dimension_count={rope_dim} is odd; rotation takes pairs");
        }

        let kv_heads = per_layer(g, "attention.head_count_kv", n_layers)?;
        let pattern = per_layer(g, "attention.sliding_window_pattern", n_layers)?;
        let blocks = kv_heads
            .iter()
            .zip(pattern.iter())
            .enumerate()
            .map(|(index, (&n_kv_heads, &windowed))| {
                if n_kv_heads == 0 {
                    bail!("{arch} block {index} declares no key/value heads");
                }
                if n_heads % n_kv_heads != 0 {
                    bail!(
                        "{arch} block {index} has {n_heads} query heads over {n_kv_heads} \
                         key/value heads, which do not group evenly"
                    );
                }
                Ok(Block {
                    // The pattern marks sliding-window blocks with 1 and full blocks with 0.
                    attention: if windowed == 1 { Attention::SlidingWindow } else { Attention::Full },
                    n_kv_heads,
                    ffn: ffn_of(g, index)?,
                    sink: g.has_tensor(&format!("blk.{index}.attn_sinks.weight")),
                })
            })
            .collect::<Result<Vec<Block>>>()?;

        let dims = g
            .embedding_dims()
            .context("no token_embd.weight to size the vocabulary from")?;
        if dims.len() != 2 {
            bail!("token_embd.weight must be 2-D, checkpoint has {dims:?}");
        }
        let vocab = usize::try_from(dims[1])
            .context("token_embd.weight vocabulary does not fit this host")?;

        // Honoured when stated, assumed when not, and refused when it disagrees with the
        // only behaviour implemented. See `Config::norm_topk`.
        let norm_topk = match g.kv("expert_weights_norm").and_then(|v| v.as_u64()) {
            Some(0) => bail!(
                "{arch}.expert_weights_norm=0 asks for unnormalised routing weights, which \
                 this loader does not implement"
            ),
            _ => true,
        };

        let gating = need("expert_gating_func")?;
        if gating != GATING_SIGMOID {
            bail!("{arch}.expert_gating_func={gating} is not the sigmoid routing this binds");
        }

        // Group (node-limited) routing restricts the top-k to experts within chosen groups.
        // The checkpoint we have declares 1 group of which 1 is used, which is no restriction
        // at all; anything else changes which experts a token reaches and is not implemented.
        for key in ["expert_group_count", "expert_group_used_count"] {
            match g.kv(key).and_then(|v| v.as_u64()) {
                Some(groups) if groups != 1 => bail!(
                    "{arch}.{key}={groups} is group routing, which this loader does not implement"
                ),
                _ => {}
            }
        }

        Ok(Config {
            blocks,
            dim: need_usize("embedding_length")?,
            n_heads,
            key_head_dim,
            value_head_dim,
            rope_dim,
            rope_base: need_f64("rope.freq_base")?,
            rope_base_swa: need_f64("rope.freq_base_swa")?,
            window: need_usize("attention.sliding_window")?,
            value_scale: need_f64("attention.value_scale")? as f32,
            n_experts: need_usize("expert_count")?,
            n_experts_used: need_usize("expert_used_count")?,
            expert_ff: need_usize("expert_feed_forward_length")?,
            feed_forward: need_usize("feed_forward_length")?,
            vocab,
            eps: need_f64("attention.layer_norm_rms_epsilon")? as f32,
            context: need_usize("context_length")?,
            norm_topk,
        })
    }

    /// Blocks in this checkpoint.
    pub fn n_layers(&self) -> usize {
        self.blocks.len()
    }

    /// Query heads sharing each key/value pair in `layer`.
    pub fn group(&self, layer: usize) -> Result<usize> {
        let block = self
            .blocks
            .get(layer)
            .with_context(|| format!("layer {layer} is past the {} in this model", self.blocks.len()))?;
        Ok(self.n_heads / block.n_kv_heads)
    }

    /// The RoPE base `layer` rotates with. Sliding-window blocks use the shorter one.
    pub fn rope_base_for(&self, layer: usize) -> Result<f64> {
        let block = self
            .blocks
            .get(layer)
            .with_context(|| format!("layer {layer} is past the {} in this model", self.blocks.len()))?;
        Ok(match block.attention {
            Attention::Full => self.rope_base,
            Attention::SlidingWindow => self.rope_base_swa,
        })
    }

    /// Experts evaluated per token as a fraction of those stored — what makes streaming
    /// the routed weights worth the latency.
    pub fn activation(&self) -> f64 {
        if self.n_experts == 0 {
            return 1.0;
        }
        self.n_experts_used as f64 / self.n_experts as f64
    }
}

/// Read a per-layer integer array, accepting a single value as "the same for every block".
///
/// GGUF writers emit these as arrays for hybrid models and as scalars for uniform ones;
/// both are honest descriptions, and a binder that accepted only one shape would refuse
/// checkpoints it can execute.
fn per_layer(g: &impl Header, key: &str, n_layers: usize) -> Result<Vec<usize>> {
    let value = g.kv(key).with_context(|| format!("missing mimo2.{key}"))?;
    let values: Vec<usize> = match value {
        Value::Arr(items) => items
            .iter()
            .map(|item| {
                let raw = item
                    .as_u64()
                    .with_context(|| format!("mimo2.{key} holds a non-integer"))?;
                usize::try_from(raw).with_context(|| format!("mimo2.{key}={raw} does not fit this host"))
            })
            .collect::<Result<Vec<usize>>>()?,
        other => {
            let raw = other
                .as_u64()
                .with_context(|| format!("mimo2.{key} is neither an integer nor an array"))?;
            let one = usize::try_from(raw)
                .with_context(|| format!("mimo2.{key}={raw} does not fit this host"))?;
            vec![one; n_layers]
        }
    };
    if values.len() != n_layers {
        bail!(
            "mimo2.{key} describes {} blocks but the checkpoint has {n_layers}",
            values.len()
        );
    }
    Ok(values)
}

/// Metadata whose meaning this binder does not implement yet.
///
/// Presence is refused rather than ignored: a checkpoint carrying one of these computes
/// something this module would not, and silently binding it is how a model "works" while
/// answering differently. Each entry leaves with its own implementation.
fn refuse_unimplemented(g: &impl Header) -> Result<()> {
    const UNIMPLEMENTED: [&str; 6] = [
        "rope.scaling.type",
        "rope.scaling.factor",
        "expert_shared_feed_forward_length",
        // `expert_weights_scale` changes what a chosen expert contributes. The checkpoint
        // declares none and `config.json` says `routed_scaling_factor: null`, so a checkpoint
        // that declares one computes something this module does not.
        //
        // Its sibling `expert_weights_norm` is NOT here: it is honoured rather than refused,
        // because this architecture normalises and the key's absence says nothing either way.
        "expert_weights_scale",
        "attention.q_lora_rank",
        "attention.kv_lora_rank",
    ];
    for key in UNIMPLEMENTED {
        if g.kv(key).is_some() {
            bail!("mimo2.{key} is present and this loader does not implement it yet");
        }
    }
    Ok(())
}

/// How block `index` feeds forward, read from the tensors it carries.
///
/// Not from metadata, because the metadata does not say: `feed_forward_length` and
/// `expert_count` are both declared for this checkpoint and only one of them applies to any
/// given block. The tensor table is where the checkpoint states which.
fn ffn_of(g: &impl Header, index: usize) -> Result<Ffn> {
    let routed = g.has_tensor(&format!("blk.{index}.ffn_gate_inp.weight"));
    let dense = g.has_tensor(&format!("blk.{index}.ffn_gate.weight"));
    match (routed, dense) {
        (true, false) => Ok(Ffn::Routed {
            selection_bias: g.has_tensor(&format!("blk.{index}.exp_probs_b.bias")),
        }),
        (false, true) => Ok(Ffn::Dense),
        (true, true) => bail!(
            "mimo2 block {index} carries both a router and a dense feed-forward; \
             which one runs is not something this loader should guess"
        ),
        (false, false) => bail!(
            "mimo2 block {index} carries neither ffn_gate_inp.weight nor ffn_gate.weight, \
             so it has no feed-forward this loader can find"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};

    /// A transcribed header. Everything the binder reads, and nothing it does not.
    struct Fixture {
        arch: String,
        meta: BTreeMap<String, Value>,
        embedding: Vec<u64>,
        tensors: BTreeSet<String>,
    }

    impl Header for Fixture {
        fn arch(&self) -> Option<&str> {
            Some(&self.arch)
        }
        fn kv(&self, suffix: &str) -> Option<&Value> {
            self.meta.get(suffix)
        }
        fn embedding_dims(&self) -> Option<&[u64]> {
            Some(&self.embedding)
        }
        fn has_tensor(&self, name: &str) -> bool {
            self.tensors.contains(name)
        }
    }

    /// The real checkpoint's header, transcribed. Every number here was read out of
    /// MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf, so a change that would stop the model
    /// we actually have from binding fails here rather than after a 156 GiB load.
    fn mimo(n_layers: usize) -> Fixture {
        // The real pattern, not a tidy approximation of it: block 0 is full, then every
        // sixth block from block 5. Nine full blocks, thirty-nine windowed.
        let windowed = |i: usize| u64::from(!(i == 0 || (i >= 5 && (i - 5).is_multiple_of(6))));
        let meta = [
            ("block_count", Value::U(n_layers as u64)),
            ("embedding_length", Value::U(4096)),
            ("feed_forward_length", Value::U(16384)),
            ("attention.head_count", Value::U(64)),
            (
                "attention.head_count_kv",
                Value::Arr((0..n_layers).map(|i| Value::U(if windowed(i) == 1 { 8 } else { 4 })).collect()),
            ),
            ("attention.key_length", Value::U(192)),
            ("attention.value_length", Value::U(128)),
            ("attention.sliding_window", Value::U(128)),
            (
                "attention.sliding_window_pattern",
                Value::Arr((0..n_layers).map(|i| Value::U(windowed(i))).collect()),
            ),
            ("attention.value_scale", Value::F(0.707)),
            ("attention.layer_norm_rms_epsilon", Value::F(1e-6)),
            ("rope.dimension_count", Value::U(64)),
            ("rope.freq_base", Value::F(1e7)),
            ("rope.freq_base_swa", Value::F(1e4)),
            ("expert_count", Value::U(256)),
            ("expert_used_count", Value::U(8)),
            ("expert_feed_forward_length", Value::U(2048)),
            ("expert_gating_func", Value::U(2)),
            ("context_length", Value::U(1_048_576)),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        // The tensor table, transcribed from the same file: block 0 is dense and full, the
        // other 47 route with a selection bias, and the 39 windowed blocks carry a sink.
        let mut tensors = BTreeSet::new();
        for i in 0..n_layers {
            for suffix in ["attn_norm.weight", "attn_qkv.weight", "attn_output.weight", "ffn_norm.weight"] {
                tensors.insert(format!("blk.{i}.{suffix}"));
            }
            if i == 0 {
                for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                    tensors.insert(format!("blk.{i}.{suffix}"));
                }
            } else {
                for suffix in [
                    "ffn_gate_inp.weight",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                    "exp_probs_b.bias",
                ] {
                    tensors.insert(format!("blk.{i}.{suffix}"));
                }
            }
            if windowed(i) == 1 {
                tensors.insert(format!("blk.{i}.attn_sinks.weight"));
            }
        }
        Fixture { arch: ARCHITECTURE.into(), meta, embedding: vec![4096, 152_576], tensors }
    }

    fn with(n_layers: usize, key: &str, value: Value) -> Fixture {
        let mut f = mimo(n_layers);
        f.meta.insert(key.into(), value);
        f
    }

    #[test]
    fn the_real_checkpoints_header_binds() {
        let cfg = Config::from_header(&mimo(48)).expect("the checkpoint we have must bind");

        assert_eq!(cfg.n_layers(), 48);
        assert_eq!(cfg.key_head_dim, 192);
        assert_eq!(cfg.value_head_dim, 128, "values are narrower than keys");
        assert_eq!(cfg.rope_dim, 64, "a third of each head is rotated");
        assert_eq!(cfg.vocab, 152_576);
        assert_eq!(cfg.n_experts, 256);
        assert_eq!(cfg.n_experts_used, 8);
        assert!((cfg.activation() - 0.031_25).abs() < 1e-9);
        assert!((cfg.value_scale - 0.707).abs() < 1e-6);
        assert_eq!(cfg.window, 128);

        // Block 0 is full attention with 4 kv heads and a DENSE feed-forward; block 1 is
        // windowed with 8, routes, and carries a sink. Reading block 0 as routed is the
        // mistake this asserts against: it is the only dense block in the model.
        assert_eq!(
            cfg.blocks[0],
            Block { attention: Attention::Full, n_kv_heads: 4, ffn: Ffn::Dense, sink: false }
        );
        assert_eq!(
            cfg.blocks[1],
            Block {
                attention: Attention::SlidingWindow,
                n_kv_heads: 8,
                ffn: Ffn::Routed { selection_bias: true },
                sink: true,
            }
        );
        assert_eq!(
            cfg.blocks.iter().filter(|b| b.sink).count(),
            39,
            "the 39 windowed blocks carry a sink and the 9 full blocks do not"
        );
        assert_eq!(
            cfg.blocks.iter().filter(|b| matches!(b.ffn, Ffn::Routed { .. })).count(),
            47
        );
        assert_eq!(cfg.group(0).unwrap(), 16);
        assert_eq!(cfg.group(1).unwrap(), 8);

        // The two bases are not interchangeable: a windowed block rotating at 1e7 would
        // attend over the wrong positions while looking entirely healthy.
        assert_eq!(cfg.rope_base_for(0).unwrap(), 1e7);
        assert_eq!(cfg.rope_base_for(1).unwrap(), 1e4);
    }

    #[test]
    fn a_uniform_scalar_describes_every_block() {
        let mut f = mimo(4);
        f.meta.insert("attention.head_count_kv".into(), Value::U(8));
        f.meta.insert("attention.sliding_window_pattern".into(), Value::U(1));
        let cfg = Config::from_header(&f).expect("a uniform checkpoint is still a checkpoint");
        assert!(cfg.blocks.iter().all(|b| b.attention == Attention::SlidingWindow));
        assert!(cfg.blocks.iter().all(|b| b.n_kv_heads == 8));
    }

    #[test]
    fn a_per_layer_array_of_the_wrong_length_is_refused() {
        let f = with(48, "attention.head_count_kv", Value::Arr(vec![Value::U(4); 47]));
        let error = Config::from_header(&f).expect_err("47 blocks described for 48 must not bind");
        assert!(format!("{error:#}").contains("describes 47 blocks"), "{error:#}");
    }

    #[test]
    fn softmax_routing_is_not_this_architecture() {
        let f = with(4, "expert_gating_func", Value::U(1));
        let error = Config::from_header(&f).expect_err("softmax gating must not bind as sigmoid");
        assert!(format!("{error:#}").contains("expert_gating_func=1"), "{error:#}");
    }

    #[test]
    fn heads_that_do_not_group_evenly_are_refused() {
        let f = with(4, "attention.head_count_kv", Value::Arr(vec![Value::U(5); 4]));
        let error = Config::from_header(&f).expect_err("64 query heads over 5 kv heads must not bind");
        assert!(format!("{error:#}").contains("do not group evenly"), "{error:#}");
    }

    #[test]
    fn rotating_more_than_the_head_is_refused() {
        let f = with(4, "rope.dimension_count", Value::U(256));
        let error = Config::from_header(&f).expect_err("rotating 256 of a 192-wide head must not bind");
        assert!(format!("{error:#}").contains("exceeds the 192-wide key head"), "{error:#}");
    }

    #[test]
    fn unimplemented_semantics_are_refused_by_name() {
        for key in [
            "rope.scaling.type",
            "expert_shared_feed_forward_length",
            "expert_weights_scale",
            "attention.kv_lora_rank",
        ] {
            let f = with(4, key, Value::U(1));
            let error = Config::from_header(&f)
                .unwrap_err_or_else_panic(key);
            assert!(format!("{error:#}").contains(key), "{error:#}");
        }
    }

    /// The one default this binder allows, and the reason it is allowed.
    #[test]
    fn routing_weights_are_normalised_unless_the_checkpoint_objects() {
        // Absent, as it is in every MiMo GGUF: normalised, because the architecture is.
        assert!(Config::from_header(&mimo(4)).unwrap().norm_topk);
        // Stated: honoured.
        assert!(Config::from_header(&with(4, "expert_weights_norm", Value::U(1))).unwrap().norm_topk);
        // Stated false: refused, because unnormalised routing is not what this computes.
        let error = Config::from_header(&with(4, "expert_weights_norm", Value::U(0)))
            .expect_err("unnormalised routing must not bind as normalised");
        assert!(format!("{error:#}").contains("expert_weights_norm=0"), "{error:#}");
    }

    #[test]
    fn a_missing_hyperparameter_is_an_error_not_a_default() {
        let mut f = mimo(4);
        f.meta.remove("attention.value_scale");
        let error = Config::from_header(&f).expect_err("a missing value scale must not default to 1.0");
        assert!(format!("{error:#}").contains("value_scale"), "{error:#}");
    }

    #[test]
    fn group_routing_is_refused_but_one_group_is_not_group_routing() {
        // 1x1 is what the checkpoint we have declares, and it restricts nothing.
        let mut f = mimo(4);
        f.meta.insert("expert_group_count".into(), Value::U(1));
        f.meta.insert("expert_group_used_count".into(), Value::U(1));
        Config::from_header(&f).expect("one group of which one is used is no restriction");

        let f = with(4, "expert_group_count", Value::U(8));
        let error = Config::from_header(&f).expect_err("8 routing groups must not bind as none");
        assert!(format!("{error:#}").contains("expert_group_count=8"), "{error:#}");
    }

    #[test]
    fn a_block_with_no_feed_forward_is_refused() {
        let mut f = mimo(4);
        f.tensors.remove("blk.2.ffn_gate_inp.weight");
        let error = Config::from_header(&f).expect_err("a block with no feed-forward must not bind");
        assert!(format!("{error:#}").contains("block 2"), "{error:#}");
    }

    #[test]
    fn a_block_that_is_both_dense_and_routed_is_refused() {
        let mut f = mimo(4);
        f.tensors.insert("blk.2.ffn_gate.weight".into());
        let error = Config::from_header(&f).expect_err("a block cannot be both and be executed");
        assert!(format!("{error:#}").contains("both a router and a dense"), "{error:#}");
    }

    #[test]
    fn the_selection_bias_is_read_rather_than_assumed() {
        // Every routed block in the checkpoint we have carries one. A checkpoint without it
        // routes on the bare sigmoid scores, and the difference is which experts run.
        let mut f = mimo(4);
        f.tensors.remove("blk.3.exp_probs_b.bias");
        let cfg = Config::from_header(&f).expect("a router without a bias is still a router");
        assert_eq!(cfg.blocks[2].ffn, Ffn::Routed { selection_bias: true });
        assert_eq!(cfg.blocks[3].ffn, Ffn::Routed { selection_bias: false });
    }

    /// `expect_err` with a message naming the metadata key, so a regression says which
    /// refusal stopped holding rather than only that one did.
    trait UnwrapErr<T> {
        fn unwrap_err_or_else_panic(self, key: &str) -> anyhow::Error;
    }
    impl<T: std::fmt::Debug> UnwrapErr<T> for Result<T> {
        fn unwrap_err_or_else_panic(self, key: &str) -> anyhow::Error {
            match self {
                Ok(bound) => panic!("{key} must be refused, not ignored; it bound as {bound:?}"),
                Err(error) => error,
            }
        }
    }
}
