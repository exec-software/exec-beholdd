//! The `Device` trait and its implementations.
//!
//! Owns: `cpu` (the correctness reference, and a real target on unified memory), the
//! `vulkan` acceleration prototype, and optional `hag` native-hardware preflight.
//! HAG's standalone hardware library owns Linux DRM resources. The separate
//! `hag-native` arithmetic prototype is not hardware-qualified and must not be
//! advertised as a serving backend.
//!
//! # What this trait is for, and what it deliberately is not
//!
//! It exists so a model is written once and runs anywhere. The measure of whether the
//! seam is in the right place is Gate M4.3: **adding Vulkan must change no model code.**
//!
//! So the operations here are named at the level a model actually thinks in — "matvec
//! against a quantised weight", "normalise", "attend" — and not at the level of memory
//! management. A trait full of `alloc`/`copy`/`launch_kernel` would be a device API, and
//! every model written against it would end up containing device logic, which is the
//! same failure with more steps.
//!
//! # A trait with one implementation is a guess
//!
//! This one currently has exactly one, so it is currently a guess. It is written from
//! what the first real model needs (`qwen3moe`: GQA, RMSNorm, RoPE, SwiGLU, a top-k
//! router, and quantised matvecs) rather than from what a GPU offers, and it should be
//! expected to move as native HAG and the Vulkan prototype prove the execution seam.
//!
//! Nothing here is stable until two backends implement it.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod cpu;
pub mod execution;
#[cfg(feature = "hag")]
pub mod hag;
#[cfg(feature = "vulkan")]
pub mod vulkan;

use beholdd_format::GgmlType;

/// A weight as it sits in memory: the raw bytes plus what they mean.
///
/// Weights are **not** dequantised at load. That is the whole economy of this engine: an
/// expert held as MXFP4 is 12.6 MB and as f32 is 100 MB, and since a matvec is
/// bandwidth-bound, reading an eighth of the bytes is faster rather than slower. A
/// `Device` that needs floats must dequantise on its own terms, at its own time.
#[derive(Debug, Clone, Copy)]
pub struct Weight<'a> {
    /// The stored bytes, exactly as read from the checkpoint.
    pub bytes: &'a [u8],
    /// How to interpret them.
    pub kind: GgmlType,
    /// Output length, i.e. the number of rows.
    pub rows: usize,
    /// Input length, i.e. the width of a row.
    pub cols: usize,
}

/// What went wrong, in terms a caller can act on.
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    /// The device has no decoder for this weight format.
    ///
    /// This is a REFUSAL, not a fallback. Silently taking a slower or different path
    /// would make a model's numerics depend on which device it landed on.
    #[error("{device} cannot decode {kind} (weight {rows}x{cols})")]
    UnsupportedFormat {
        /// Device name.
        device: &'static str,
        /// The ggml type name.
        kind: String,
        /// Rows.
        rows: usize,
        /// Columns.
        cols: usize,
    },
    /// Shapes did not agree.
    #[error("shape mismatch: {0}")]
    Shape(String),
    /// Host or device storage could not be reserved.
    #[error("{device} could not allocate {bytes} bytes for {purpose}")]
    Allocation {
        /// Device name.
        device: &'static str,
        /// Requested byte count.
        bytes: usize,
        /// What the storage would have held.
        purpose: &'static str,
    },
    /// A backend API rejected initialization or execution.
    #[error("{device} backend error: {message}")]
    Backend {
        /// Device/backend name.
        device: &'static str,
        /// Actionable backend diagnostic.
        message: String,
    },
}

/// How a head's coordinates are paired for a rotary embedding.
///
/// This is an architecture convention, not an optimisation choice. Both layouts have
/// identical shapes, so selecting the wrong one would otherwise run successfully and
/// produce incorrect attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeLayout {
    /// Adjacent coordinates form pairs: `(x[0], x[1])`, `(x[2], x[3])`, and so on.
    Interleaved,
    /// The two halves form pairs: `(x[0], x[n])`, `(x[1], x[n + 1])`, and so on.
    SplitHalf,
}

/// The logical dimensions and score scaling for one causal-attention call.
///
/// Keeping this metadata together makes the buffer layouts explicit at the backend
/// boundary and leaves room for an accelerated backend to validate or specialise a
/// call before launching its implementation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttentionConfig {
    /// Number of sequence positions present in every input buffer.
    pub n_positions: usize,
    /// Number of query heads, and therefore output heads, per position.
    pub n_heads: usize,
    /// Number of key/value heads per position.
    pub n_kv_heads: usize,
    /// Number of coordinates in each query, key, and value head.
    pub head_dim: usize,
    /// Multiplier applied to query-key dot products before softmax.
    pub scale: f32,
}

/// The logical dimensions and absolute position of an incremental attention call.
///
/// Unlike [`AttentionConfig`], the query buffer contains only a newly projected chunk,
/// while the key and value buffers contain the complete cache visible to the call. This
/// keeps cache ownership in model/runtime code: a backend consumes the cache but does
/// not decide how it is allocated, grown, or persisted between tokens.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IncrementalAttentionConfig {
    /// Number of consecutive positions present in `q` and `out`.
    pub n_query_positions: usize,
    /// Absolute sequence position of the first query in `q`.
    pub query_start: usize,
    /// Number of positions present in each key/value cache buffer.
    pub n_kv_positions: usize,
    /// Number of query heads, and therefore output heads, per query position.
    pub n_heads: usize,
    /// Number of key/value heads per cached position.
    pub n_kv_heads: usize,
    /// Number of coordinates in each query, key, and value head.
    pub head_dim: usize,
    /// Multiplier applied to query-key dot products before softmax.
    pub scale: f32,
}

/// Where arithmetic happens.
///
/// Implementations must be deterministic: the same inputs give bit-identical outputs
/// across runs on the same device. A non-deterministic engine cannot be bisected, which
/// is Gate M1.3.
pub trait Device: Send + Sync {
    /// Name, for diagnostics and for error messages that must say who refused.
    fn name(&self) -> &'static str;

    /// Whether this device can decode a weight format at all.
    ///
    /// Asked at BIND time so an unsupported format is a load failure with a clear
    /// message, rather than a surprise part-way through generating a token.
    fn supports(&self, kind: GgmlType) -> bool;

    /// Decode one complete weight into a contiguous row-major `f32` buffer.
    ///
    /// The output must contain exactly `weight.rows * weight.cols` elements. This is
    /// the architecture-independent path for materialising activations such as an
    /// embedding row or a norm vector without teaching model code about quantisation.
    fn dequantize(&self, out: &mut [f32], weight: &Weight<'_>) -> Result<(), DeviceError>;

    /// `y[r] = sum_i W[r][i] * x[i]`, overwriting `y`.
    fn matvec(&self, y: &mut [f32], x: &[f32], w: &Weight<'_>) -> Result<(), DeviceError>;

    /// As [`Device::matvec`] but accumulating into `y`.
    ///
    /// Separate from `matvec` because an MoE sums experts into one buffer, and a caller
    /// that had to zero-and-add would either allocate per expert or get it wrong.
    fn matvec_add(
        &self,
        y: &mut [f32],
        x: &[f32],
        w: &Weight<'_>,
        scale: f32,
    ) -> Result<(), DeviceError>;

    /// `y = x / sqrt(mean(x^2) + eps) * weight`.
    fn rmsnorm(
        &self,
        y: &mut [f32],
        x: &[f32],
        weight: Option<&[f32]>,
        eps: f32,
    ) -> Result<(), DeviceError>;

    /// Softmax in place. `-inf` entries must produce exactly zero.
    fn softmax(&self, v: &mut [f32]) -> Result<(), DeviceError>;

    /// Causal grouped-query attention over a complete sequence of projected heads.
    ///
    /// `q` and `out` are contiguous `[position][query_head][head_dim]` buffers. `k`
    /// and `v` are contiguous `[position][kv_head][head_dim]` buffers. Query heads are
    /// split into contiguous, equal-sized groups and group `g` uses key/value head `g`.
    /// Position `t` may attend only to positions `0..=t`.
    ///
    /// `scale` is applied to each query-key dot product before softmax. The operation
    /// overwrites `out` and refuses empty or inconsistent shapes without modifying it.
    fn causal_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k: &[f32],
        v: &[f32],
        config: AttentionConfig,
    ) -> Result<(), DeviceError>;

    /// Causal grouped-query attention for newly projected positions against a KV cache.
    ///
    /// `q` and `out` are contiguous
    /// `[query_position][query_head][head_dim]` buffers containing
    /// `config.n_query_positions` positions. `k_cache` and `v_cache` are contiguous
    /// `[cache_position][kv_head][head_dim]` buffers containing
    /// `config.n_kv_positions` positions. Local query position `i` has absolute
    /// sequence position `config.query_start + i` and may attend only to cached
    /// positions through that absolute position, inclusive.
    ///
    /// This operation does not mutate or append to the cache. Cache storage and growth
    /// remain explicit in the caller, so failed attention cannot leave cache state
    /// half-updated. The operation overwrites `out` and refuses empty, overflowing, or
    /// inconsistent shapes without modifying it.
    fn incremental_causal_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        k_cache: &[f32],
        v_cache: &[f32],
        config: IncrementalAttentionConfig,
    ) -> Result<(), DeviceError>;

    /// `output[i] += residual[i]` for a residual connection.
    fn add_in_place(&self, output: &mut [f32], residual: &[f32]) -> Result<(), DeviceError>;

    /// `gate[i] = silu(gate[i]) * up[i]`, with an optional clamp.
    fn swiglu(&self, gate: &mut [f32], up: &[f32], limit: f32) -> Result<(), DeviceError>;

    /// Apply a rotary embedding in place using the architecture's coordinate layout.
    fn rope(
        &self,
        x: &mut [f32],
        cos: &[f32],
        sin: &[f32],
        layout: RopeLayout,
        inverse: bool,
    ) -> Result<(), DeviceError>;
}
