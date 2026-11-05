//! Backend-neutral recorded execution and owned device resources.
//!
//! [`crate::Device`] is the immediate, borrowed-slice interface used by the current
//! model. This module is a deliberately separate staging seam for backends that keep
//! weights and activations resident while recording several semantic operations into
//! one submission. The two interfaces coexist: adding this module does not change the
//! current model or its correctness path.
//!
//! Resources are opaque and carry a private owner identity. A handle created by one
//! [`ExecutionDevice`] instance is refused by every other instance, before a backend
//! can interpret its erased payload. Cloning an execution device preserves its owner
//! identity; constructing a new one creates an isolated resource domain.

use std::any::Any;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use beholdd_format::GgmlType;
use beholdd_kernel::kquant;

use crate::cpu::Cpu;
use crate::{Device, DeviceError, IncrementalAttentionConfig, RopeLayout, Weight};

mod finite;
pub use finite::{FiniteCheck, FiniteCheckSpec};

/// Type-erased storage used only across built-in backend module boundaries.
pub(crate) type ErasedResource = Arc<dyn Any + Send + Sync>;

/// An error raised while creating resources, recording commands, or executing a batch.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExecutionError {
    /// An existing immediate-device operation refused the work.
    #[error(transparent)]
    Device(#[from] DeviceError),
    /// A tensor, weight, batch, or submission came from another device instance.
    #[error("{resource} belongs to another execution device")]
    ForeignResource {
        /// The kind of handle that failed its owner check.
        resource: &'static str,
    },
    /// Tensor or weight dimensions do not satisfy the operation's contract.
    #[error("execution shape mismatch: {0}")]
    Shape(String),
    /// A handle had the right owner but not the backend payload that owner creates.
    ///
    /// This indicates an implementation bug rather than malformed model input.
    #[error("{device} found an incompatible payload in {resource}")]
    IncompatibleResource {
        /// Backend reporting the invariant failure.
        device: &'static str,
        /// The kind of handle whose payload was invalid.
        resource: &'static str,
    },
    /// Backend synchronization state could not be accessed.
    #[error("{device} synchronization failed while accessing {resource}")]
    Synchronization {
        /// Backend reporting the failure.
        device: &'static str,
        /// Queue or resource whose synchronization state failed.
        resource: &'static str,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExecutionOwner(u64);

impl ExecutionOwner {
    pub(crate) fn fresh() -> Self {
        static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);
        let id = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .expect("execution-device owner identity space exhausted");
        Self(id)
    }

    pub(crate) fn require(
        self,
        actual: Self,
        resource: &'static str,
    ) -> Result<(), ExecutionError> {
        if self == actual {
            Ok(())
        } else {
            Err(ExecutionError::ForeignResource { resource })
        }
    }
}

/// Packed, backend-owned weight bytes before backend-specific preparation.
///
/// The encoded bytes are deliberately not exposed. Callers retain shape and format
/// metadata while the backend owns storage placement and representation.
#[derive(Clone)]
pub struct EncodedWeight {
    owner: ExecutionOwner,
    kind: GgmlType,
    rows: usize,
    cols: usize,
    byte_len: usize,
    resource: ErasedResource,
}

impl EncodedWeight {
    pub(crate) fn new(
        owner: ExecutionOwner,
        kind: GgmlType,
        rows: usize,
        cols: usize,
        byte_len: usize,
        resource: ErasedResource,
    ) -> Self {
        Self {
            owner,
            kind,
            rows,
            cols,
            byte_len,
            resource,
        }
    }

    /// Stored GGML type.
    pub fn kind(&self) -> GgmlType {
        self.kind
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of encoded bytes owned by the backend.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.owner
    }

    pub(crate) fn payload<T: Any>(&self) -> Option<&T> {
        self.resource.as_ref().downcast_ref::<T>()
    }
}

impl fmt::Debug for EncodedWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncodedWeight")
            .field("kind", &self.kind)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("byte_len", &self.byte_len)
            .finish_non_exhaustive()
    }
}

/// A weight in the representation used by recorded linear and embedding commands.
///
/// Preparation may be a cheap ownership-preserving transition, as it is for the CPU
/// reference, or may build descriptor and arena metadata for an accelerated backend.
#[derive(Clone)]
pub struct PreparedWeight {
    owner: ExecutionOwner,
    kind: GgmlType,
    rows: usize,
    cols: usize,
    byte_len: usize,
    resource: ErasedResource,
}

impl PreparedWeight {
    pub(crate) fn new(
        owner: ExecutionOwner,
        kind: GgmlType,
        rows: usize,
        cols: usize,
        byte_len: usize,
        resource: ErasedResource,
    ) -> Self {
        Self {
            owner,
            kind,
            rows,
            cols,
            byte_len,
            resource,
        }
    }

    /// Stored GGML type.
    pub fn kind(&self) -> GgmlType {
        self.kind
    }

    /// Number of output rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of encoded weight bytes retained by this prepared representation.
    pub fn byte_len(&self) -> usize {
        self.byte_len
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.owner
    }

    pub(crate) fn payload<T: Any>(&self) -> Option<&T> {
        self.resource.as_ref().downcast_ref::<T>()
    }
}

impl fmt::Debug for PreparedWeight {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedWeight")
            .field("kind", &self.kind)
            .field("rows", &self.rows)
            .field("cols", &self.cols)
            .field("byte_len", &self.byte_len)
            .finish_non_exhaustive()
    }
}

/// An owned one-dimensional `f32` tensor in backend storage.
///
/// Clones alias the same storage; they do not copy its values. Host transfers and
/// recorded commands are the only public ways to mutate the tensor.
#[derive(Clone)]
pub struct F32Tensor {
    owner: ExecutionOwner,
    len: usize,
    resource: ErasedResource,
}

impl F32Tensor {
    pub(crate) fn new(owner: ExecutionOwner, len: usize, resource: ErasedResource) -> Self {
        Self {
            owner,
            len,
            resource,
        }
    }

    /// Number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the tensor contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A view spanning the complete tensor.
    pub fn full_view(&self) -> F32TensorView {
        F32TensorView {
            tensor: self.clone(),
            offset: 0,
            len: self.len,
        }
    }

    /// A checked view spanning `len` elements beginning at `offset`.
    pub fn view(&self, offset: usize, len: usize) -> Result<F32TensorView, ExecutionError> {
        let end = offset.checked_add(len).ok_or_else(|| {
            ExecutionError::Shape(format!(
                "tensor view end overflows: offset={offset} len={len}"
            ))
        })?;
        if end > self.len {
            return Err(ExecutionError::Shape(format!(
                "tensor view {offset}..{end} exceeds tensor length {}",
                self.len
            )));
        }
        Ok(F32TensorView {
            tensor: self.clone(),
            offset,
            len,
        })
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.owner
    }

    pub(crate) fn payload<T: Any>(&self) -> Option<&T> {
        self.resource.as_ref().downcast_ref::<T>()
    }
}

impl fmt::Debug for F32Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("F32Tensor")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// A checked, owning view into an [`F32Tensor`].
///
/// The view keeps its tensor alive and can therefore be stored safely in a recorded
/// batch. Source operands are read when a command executes, not when it is recorded.
#[derive(Clone)]
pub struct F32TensorView {
    tensor: F32Tensor,
    offset: usize,
    len: usize,
}

impl F32TensorView {
    /// Element offset from the start of the underlying tensor.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Number of elements visible through the view.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the view contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether this view and `other` cover at least one common tensor element.
    ///
    /// Views from different tensors or device owners never overlap. This lets callers
    /// reason about aliasing without exposing a backend pointer or allocation identity.
    pub fn overlaps(&self, other: &Self) -> bool {
        if self.is_empty()
            || other.is_empty()
            || self.owner() != other.owner()
            || !Arc::ptr_eq(&self.tensor.resource, &other.tensor.resource)
        {
            return false;
        }
        let self_end = self.offset + self.len;
        let other_end = other.offset + other.len;
        self.offset < other_end && other.offset < self_end
    }

    /// A checked subview relative to this view.
    pub fn view(&self, offset: usize, len: usize) -> Result<Self, ExecutionError> {
        let relative_end = offset.checked_add(len).ok_or_else(|| {
            ExecutionError::Shape(format!(
                "tensor subview end overflows: offset={offset} len={len}"
            ))
        })?;
        if relative_end > self.len {
            return Err(ExecutionError::Shape(format!(
                "tensor subview {offset}..{relative_end} exceeds view length {}",
                self.len
            )));
        }
        let absolute_offset = self.offset.checked_add(offset).ok_or_else(|| {
            ExecutionError::Shape("tensor subview absolute offset overflows".into())
        })?;
        Ok(Self {
            tensor: self.tensor.clone(),
            offset: absolute_offset,
            len,
        })
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.tensor.owner()
    }

    pub(crate) fn tensor(&self) -> &F32Tensor {
        &self.tensor
    }
}

impl fmt::Debug for F32TensorView {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("F32TensorView")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// How a recorded linear operation updates its output.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LinearMode {
    /// Replace the output with `weight * input`.
    Overwrite,
    /// Add `scale * weight * input` to the existing output.
    ScaledAdd {
        /// Multiplier applied before accumulation.
        scale: f32,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum Command {
    EmbeddingRow {
        output: F32TensorView,
        weight: PreparedWeight,
        row: usize,
    },
    Linear {
        output: F32TensorView,
        input: F32TensorView,
        weight: PreparedWeight,
        mode: LinearMode,
    },
    SwiGlu {
        gate: F32TensorView,
        up: F32TensorView,
        limit: f32,
    },
    SigmoidGate {
        output: F32TensorView,
        input: F32TensorView,
        gate: F32TensorView,
    },
    RmsNorm {
        output: F32TensorView,
        input: F32TensorView,
        weight: Option<F32TensorView>,
        eps: f32,
    },
    Rope {
        x: F32TensorView,
        cos: F32TensorView,
        sin: F32TensorView,
        layout: RopeLayout,
        inverse: bool,
    },
    IncrementalCausalAttention {
        output: F32TensorView,
        q: F32TensorView,
        k_cache: F32TensorView,
        v_cache: F32TensorView,
        config: IncrementalAttentionConfig,
    },
}

impl Command {
    fn for_each_owner(
        &self,
        mut visit: impl FnMut(&'static str, ExecutionOwner) -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        match self {
            Self::EmbeddingRow { output, weight, .. } => {
                visit("embedding output tensor", output.owner())?;
                visit("embedding prepared weight", weight.owner)
            }
            Self::Linear {
                output,
                input,
                weight,
                ..
            } => {
                visit("linear output tensor", output.owner())?;
                visit("linear input tensor", input.owner())?;
                visit("linear prepared weight", weight.owner)
            }
            Self::SwiGlu { gate, up, .. } => {
                visit("SwiGLU gate tensor", gate.owner())?;
                visit("SwiGLU up tensor", up.owner())
            }
            Self::SigmoidGate {
                output,
                input,
                gate,
            } => {
                visit("sigmoid output tensor", output.owner())?;
                visit("sigmoid input tensor", input.owner())?;
                visit("sigmoid gate tensor", gate.owner())
            }
            Self::RmsNorm {
                output,
                input,
                weight,
                ..
            } => {
                visit("RMSNorm output tensor", output.owner())?;
                visit("RMSNorm input tensor", input.owner())?;
                if let Some(weight) = weight {
                    visit("RMSNorm weight tensor", weight.owner())?;
                }
                Ok(())
            }
            Self::Rope { x, cos, sin, .. } => {
                visit("RoPE input/output tensor", x.owner())?;
                visit("RoPE cosine tensor", cos.owner())?;
                visit("RoPE sine tensor", sin.owner())
            }
            Self::IncrementalCausalAttention {
                output,
                q,
                k_cache,
                v_cache,
                ..
            } => {
                visit("attention output tensor", output.owner())?;
                visit("attention query tensor", q.owner())?;
                visit("attention key cache", k_cache.owner())?;
                visit("attention value cache", v_cache.owner())
            }
        }
    }

    fn primary_owner(&self) -> ExecutionOwner {
        match self {
            Self::EmbeddingRow { output, .. } => output.owner(),
            Self::Linear { output, .. } => output.owner(),
            Self::SwiGlu { gate, .. } => gate.owner(),
            Self::SigmoidGate { output, .. } => output.owner(),
            Self::RmsNorm { output, .. } => output.owner(),
            Self::Rope { x, .. } => x.owner(),
            Self::IncrementalCausalAttention { output, .. } => output.owner(),
        }
    }
}

/// A backend-neutral sequence of semantic tensor operations.
///
/// Commands execute in insertion order. Each command observes writes made by earlier
/// commands in the same list. Overlapping views are permitted; every command behaves as
/// though all of its input operands were snapshotted before its destination is written.
/// A list may contain resources from only one execution-device owner.
/// A backend may refuse unsupported operations or aliasing at [`ExecutionDevice::record`];
/// building a valid list alone does not establish backend capability.
#[derive(Debug, Clone, Default)]
pub struct CommandList {
    owner: Option<ExecutionOwner>,
    commands: Vec<Command>,
}

impl CommandList {
    /// Create an empty, as-yet-unowned command list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of commands in the list.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    /// Whether the list contains no commands.
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub(crate) fn owner(&self) -> Option<ExecutionOwner> {
        self.owner
    }

    pub(crate) fn commands(&self) -> &[Command] {
        &self.commands
    }

    /// Decode one selected row of an owned prepared weight into `output`.
    ///
    /// The output length must equal the weight's column count, and `row` must be
    /// below its row count. Recording retains the immutable weight and destination;
    /// replay overwrites the destination without rereading a checkpoint. Backends
    /// may refuse unsupported formats at recording. Failed appends preserve the list.
    pub fn embedding_row(
        &mut self,
        output: &F32TensorView,
        weight: &PreparedWeight,
        row: usize,
    ) -> Result<(), ExecutionError> {
        validate_embedding_shape(output.len(), weight, row)?;
        let command = Command::EmbeddingRow {
            output: output.clone(),
            weight: weight.clone(),
            row,
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    /// Append a matrix-vector product with overwrite or scaled-add semantics.
    ///
    /// `input.len()` must equal `weight.cols()` and `output.len()` must equal
    /// `weight.rows()`. A failed append leaves the list unchanged.
    pub fn linear(
        &mut self,
        output: &F32TensorView,
        input: &F32TensorView,
        weight: &PreparedWeight,
        mode: LinearMode,
    ) -> Result<(), ExecutionError> {
        if input.len != weight.cols || output.len != weight.rows {
            return Err(ExecutionError::Shape(format!(
                "linear weight is {}x{}, got input={} output={}",
                weight.rows, weight.cols, input.len, output.len
            )));
        }
        if let LinearMode::ScaledAdd { scale } = mode {
            if !scale.is_finite() {
                return Err(ExecutionError::Shape(format!(
                    "linear scaled-add multiplier must be finite, got {scale}"
                )));
            }
        }

        let command = Command::Linear {
            output: output.clone(),
            input: input.clone(),
            weight: weight.clone(),
            mode,
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    /// Append an in-place SwiGLU operation: `gate = silu(gate) * up`.
    ///
    /// `limit == 0` disables clamping. A positive limit applies the model convention
    /// used by [`Device::swiglu`]. A failed append leaves the list unchanged.
    pub fn swiglu(
        &mut self,
        gate: &F32TensorView,
        up: &F32TensorView,
        limit: f32,
    ) -> Result<(), ExecutionError> {
        if gate.len != up.len {
            return Err(ExecutionError::Shape(format!(
                "SwiGLU expected gate={} to match up={}",
                gate.len, up.len
            )));
        }
        if !limit.is_finite() || limit < 0.0 {
            return Err(ExecutionError::Shape(format!(
                "SwiGLU limit must be finite and non-negative, got {limit}"
            )));
        }

        let command = Command::SwiGlu {
            gate: gate.clone(),
            up: up.clone(),
            limit,
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    /// Append RMS normalization of one nonempty vector into an equally sized output.
    ///
    /// An optional resident f32 weight must have the same length as the input;
    /// `eps` must be finite and non-negative. Input and weight values are read when
    /// the command executes, including writes by earlier commands. Overlapping views
    /// have snapshot semantics, so output may alias either read operand. A backend
    /// may explicitly refuse unsupported aliases at recording. Failed appends leave
    /// both the command list and its owner unchanged.
    pub fn rmsnorm(
        &mut self,
        output: &F32TensorView,
        input: &F32TensorView,
        weight: Option<&F32TensorView>,
        eps: f32,
    ) -> Result<(), ExecutionError> {
        if input.is_empty() || output.len != input.len {
            return Err(ExecutionError::Shape(format!(
                "RMSNorm requires nonempty equal input/output lengths, got input={} output={}",
                input.len, output.len
            )));
        }
        if let Some(weight) = weight {
            if weight.len != input.len {
                return Err(ExecutionError::Shape(format!(
                    "RMSNorm expected {} weights, got {}",
                    input.len, weight.len
                )));
            }
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(ExecutionError::Shape(format!(
                "RMSNorm epsilon must be finite and non-negative, got {eps}"
            )));
        }
        let command = Command::RmsNorm {
            output: output.clone(),
            input: input.clone(),
            weight: weight.cloned(),
            eps,
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    /// Append in-place rotary embedding using resident cosine and sine vectors.
    ///
    /// Cosine and sine lengths must match, and `x.len()` must be exactly twice that
    /// length. Empty views are a valid no-op, matching [`Device::rope`]. Coefficients
    /// are runtime operands, not copied constants: earlier commands may update them.
    /// Every read operand is conceptually snapshotted before `x` changes, including
    /// overlapping coefficient views. A backend may refuse unsupported aliases when
    /// recording. Failed appends leave both the list and its owner unchanged.
    pub fn rope(
        &mut self,
        x: &F32TensorView,
        cos: &F32TensorView,
        sin: &F32TensorView,
        layout: RopeLayout,
        inverse: bool,
    ) -> Result<(), ExecutionError> {
        let expected = cos
            .len
            .checked_mul(2)
            .ok_or_else(|| ExecutionError::Shape("RoPE coordinate count overflows".into()))?;
        if sin.len != cos.len || x.len != expected {
            return Err(ExecutionError::Shape(format!(
                "RoPE expected x={expected} and equal cos/sin lengths, got x={} cos={} sin={}",
                x.len, cos.len, sin.len
            )));
        }
        let command = Command::Rope {
            x: x.clone(),
            cos: cos.clone(),
            sin: sin.clone(),
            layout,
            inverse,
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    /// Append causal grouped-query attention over resident, position-major KV caches.
    ///
    /// Query/output contain `n_query_positions * n_heads * head_dim` values; each
    /// cache contains `n_kv_positions * n_kv_heads * head_dim`. A query at absolute
    /// position `query_start + i` sees only the cache prefix ending at that position.
    /// Query heads map to contiguous groups sharing one KV head. Dimensions are
    /// positive, query heads are divisible by KV heads, and scale is finite/positive.
    ///
    /// Config is copied at append time; query and cache values are read at execution
    /// time, including earlier command writes. Overlapping operands have snapshot
    /// semantics. Cache allocation, append ownership and visible-position commits
    /// remain model/runtime responsibilities. Backends may refuse unsupported shapes
    /// or aliases at recording. Failed appends preserve the list and its owner.
    pub fn incremental_causal_attention(
        &mut self,
        output: &F32TensorView,
        q: &F32TensorView,
        k_cache: &F32TensorView,
        v_cache: &F32TensorView,
        config: IncrementalAttentionConfig,
    ) -> Result<(), ExecutionError> {
        validate_incremental_attention_shape(
            output.len(),
            q.len(),
            k_cache.len(),
            v_cache.len(),
            config,
        )?;
        let command = Command::IncrementalCausalAttention {
            output: output.clone(),
            q: q.clone(),
            k_cache: k_cache.clone(),
            v_cache: v_cache.clone(),
            config,
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    /// Append `output = input * sigmoid(gate)` with stable F32 sigmoid arithmetic.
    ///
    /// All three lengths must agree. CPU execution snapshots both inputs before
    /// writing output, including aliases; native implementations may require
    /// disjoint bounded views. An empty command is a retained no-op. This does not
    /// imply that a particular backend has a qualified implementation.
    pub fn sigmoid_gate(
        &mut self,
        output: &F32TensorView,
        input: &F32TensorView,
        gate: &F32TensorView,
    ) -> Result<(), ExecutionError> {
        if output.len() != input.len() || input.len() != gate.len() {
            return Err(ExecutionError::Shape(
                "sigmoid gate requires equal output, input and gate lengths".into(),
            ));
        }
        let command = Command::SigmoidGate {
            output: output.clone(),
            input: input.clone(),
            gate: gate.clone(),
        };
        self.accept_owners(&command)?;
        self.commands.push(command);
        Ok(())
    }

    fn accept_owners(&mut self, command: &Command) -> Result<(), ExecutionError> {
        let expected = self.owner.unwrap_or_else(|| command.primary_owner());
        command.for_each_owner(|resource, owner| expected.require(owner, resource))?;
        self.owner = Some(expected);
        Ok(())
    }
}

fn validate_embedding_shape(
    output_len: usize,
    weight: &PreparedWeight,
    row: usize,
) -> Result<(), ExecutionError> {
    if output_len != weight.cols() || row >= weight.rows() {
        return Err(ExecutionError::Shape(format!(
            "embedding row {row} of {}x{} requires output length {}, got {output_len}",
            weight.rows(),
            weight.cols(),
            weight.cols()
        )));
    }
    Ok(())
}

/// Common checked shape contract; native backends may impose smaller resource bounds.
pub(crate) fn validate_incremental_attention_shape(
    output_len: usize,
    q_len: usize,
    k_len: usize,
    v_len: usize,
    config: IncrementalAttentionConfig,
) -> Result<(), ExecutionError> {
    let IncrementalAttentionConfig {
        n_query_positions,
        query_start,
        n_kv_positions,
        n_heads,
        n_kv_heads,
        head_dim,
        scale,
    } = config;
    if [
        n_query_positions,
        n_kv_positions,
        n_heads,
        n_kv_heads,
        head_dim,
    ]
    .contains(&0)
    {
        return Err(ExecutionError::Shape(
            "incremental attention position/head dimensions must all be positive".into(),
        ));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(ExecutionError::Shape(
            "incremental attention query heads must be divisible by KV heads".into(),
        ));
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ExecutionError::Shape(
            "incremental attention scale must be positive and finite".into(),
        ));
    }
    let query_end = query_start.checked_add(n_query_positions).ok_or_else(|| {
        ExecutionError::Shape("incremental attention query range overflows".into())
    })?;
    if query_end > n_kv_positions {
        return Err(ExecutionError::Shape(
            "incremental attention query range exceeds the KV cache".into(),
        ));
    }
    let elements = |positions: usize, heads: usize| {
        positions
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(head_dim))
            .filter(|n| n.checked_mul(std::mem::size_of::<f32>()).is_some())
            .ok_or_else(|| {
                ExecutionError::Shape("incremental attention tensor extent overflows".into())
            })
    };
    let expected_q = elements(n_query_positions, n_heads)?;
    let expected_kv = elements(n_kv_positions, n_kv_heads)?;
    if output_len != expected_q
        || q_len != expected_q
        || k_len != expected_kv
        || v_len != expected_kv
    {
        return Err(ExecutionError::Shape(format!(
            "incremental attention expected out={expected_q} q={expected_q} k={expected_kv} v={expected_kv}, got out={output_len} q={q_len} k={k_len} v={v_len}"
        )));
    }
    Ok(())
}

/// A backend-recorded, reusable snapshot of a [`CommandList`].
///
/// Later changes to the source command list do not affect this batch. Resource handles
/// held by the batch keep their storage alive.
#[derive(Clone)]
pub struct RecordedBatch {
    owner: ExecutionOwner,
    command_count: usize,
    resource: ErasedResource,
}

impl RecordedBatch {
    pub(crate) fn new(
        owner: ExecutionOwner,
        command_count: usize,
        resource: ErasedResource,
    ) -> Self {
        Self {
            owner,
            command_count,
            resource,
        }
    }

    /// Number of commands captured in the batch.
    pub fn len(&self) -> usize {
        self.command_count
    }

    /// Whether the batch records no commands.
    pub fn is_empty(&self) -> bool {
        self.command_count == 0
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.owner
    }

    pub(crate) fn payload<T: Any>(&self) -> Option<&T> {
        self.resource.as_ref().downcast_ref::<T>()
    }
}

impl fmt::Debug for RecordedBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RecordedBatch")
            .field("command_count", &self.command_count)
            .finish_non_exhaustive()
    }
}

/// An opaque token representing one submitted batch.
///
/// Call [`ExecutionDevice::wait`] to observe completion or execution errors explicitly.
/// A backend must keep all in-flight resources alive even if every clone of this token
/// is dropped without waiting. The sequential CPU implementation is already complete
/// when [`ExecutionDevice::submit`] returns, but keeps the same synchronization shape
/// as asynchronous backends.
#[must_use = "a submission should be waited to observe completion or execution errors"]
#[derive(Clone)]
pub struct Submission {
    owner: ExecutionOwner,
    resource: ErasedResource,
}

impl Submission {
    pub(crate) fn new(owner: ExecutionOwner, resource: ErasedResource) -> Self {
        Self { owner, resource }
    }

    pub(crate) fn owner(&self) -> ExecutionOwner {
        self.owner
    }

    pub(crate) fn payload<T: Any>(&self) -> Option<&T> {
        self.resource.as_ref().downcast_ref::<T>()
    }
}

impl fmt::Debug for Submission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Submission").finish_non_exhaustive()
    }
}

/// Crate-private marker implemented by built-in execution backends.
pub(crate) mod sealed {
    /// Prevents external implementations from needing access to opaque backend payloads.
    pub trait Sealed {}
}

/// Owned-resource and recorded-execution interface for persistent backends.
///
/// This trait is object-safe. Its methods use concrete opaque handles rather than
/// associated resource types, allowing model/runtime code to select a backend through
/// `dyn ExecutionDevice` while retaining backend-specific storage internally. Backend
/// implementations are sealed inside this crate so the crate-private handle/command IR
/// can evolve without exposing its erased payloads to callers.
///
/// # Backend safety contract
///
/// Although this trait contains no `unsafe` methods, implementations may wrap APIs with
/// asynchronous resource access. A returned [`Submission`] or a device-owned retirement
/// registry must retain its recorded batch, tensor/weight storage, command storage, and
/// synchronization primitive until execution really completes. Dropping an unwaited
/// submission must never release in-flight resources. Host reads and writes must wait
/// for conflicting in-flight use (or refuse it) internally; safe callers are not allowed
/// to create a host/device data race by forgetting an explicit wait.
pub trait ExecutionDevice: sealed::Sealed + Send + Sync {
    /// Backend name used in diagnostics.
    fn name(&self) -> &'static str;

    /// Whether the backend can execute a packed weight format.
    fn supports(&self, kind: GgmlType) -> bool;

    /// Explicit finite-validation capability and bounded payload costs.
    ///
    /// This query performs no allocation or device I/O. Absence means unsupported;
    /// callers must refuse paths requiring validation rather than read activations
    /// back through a generic fallback. Capability is independent of queue health.
    fn finite_check_spec(&self) -> Option<FiniteCheckSpec> {
        None
    }

    /// Prepare a private checker bound to this device's resource owner.
    ///
    /// Preparation validates capability, queue health and required private storage
    /// before I/O. Share one checker across a serialized model sequence. The default
    /// is an explicit refusal and performs no resource access or tensor allocation.
    fn prepare_finite_check(&self) -> Result<FiniteCheck, ExecutionError> {
        Err(finite::unsupported(self.name()))
    }

    /// Validate one nonempty, bounded view without modifying its activations.
    ///
    /// Checker and input ownership, exact storage extent and capability limits must
    /// pass before any validation transfer. Larger tensors require explicit caller
    /// chunks. The caller serializes the complete check and immediate consumption
    /// against every write to the input; success is not a persistent tensor property.
    ///
    /// `control` can cancel at bounded phase boundaries and is never called while a
    /// backend queue/tensor mutex is held. Every callback error aborts acceptance.
    /// A numerical refusal does not poison an otherwise healthy checker. Backend
    /// transport/synchronization failures retain their normal poison semantics.
    /// The default refuses without invoking `control` or accessing either resource.
    fn check_finite(
        &self,
        _check: &mut FiniteCheck,
        _input: &F32TensorView,
        _control: &mut dyn FnMut() -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        Err(finite::unsupported(self.name()))
    }

    /// Copy one complete packed weight into backend-owned encoded storage.
    fn upload_weight(&self, weight: &Weight<'_>) -> Result<EncodedWeight, ExecutionError>;

    /// Prepare encoded weight storage for recorded linear operations.
    fn prepare_weight(&self, weight: &EncodedWeight) -> Result<PreparedWeight, ExecutionError>;

    /// Allocate a zero-initialized one-dimensional `f32` tensor.
    fn allocate_f32(&self, len: usize) -> Result<F32Tensor, ExecutionError>;

    /// Allocate a tensor and initialize it from host values.
    fn upload_f32(&self, values: &[f32]) -> Result<F32Tensor, ExecutionError> {
        let tensor = self.allocate_f32(values.len())?;
        self.write_f32(&tensor.full_view(), values)?;
        Ok(tensor)
    }

    /// Copy host values into a tensor view.
    ///
    /// The implementation synchronizes with earlier conflicting submissions before it
    /// accesses host-visible storage.
    fn write_f32(&self, destination: &F32TensorView, values: &[f32]) -> Result<(), ExecutionError>;

    /// Copy a tensor view into an equally sized host slice.
    ///
    /// The implementation makes earlier writes to the view host-visible before copying.
    fn read_f32(&self, source: &F32TensorView, values: &mut [f32]) -> Result<(), ExecutionError>;

    /// Record an immutable, reusable snapshot of a semantic command list.
    ///
    /// Backends may reject unsupported operators, shapes or aliasing patterns here
    /// without executing commands or changing tensors. Accepted commands must preserve
    /// their declared semantics; for example, a backend without operand snapshots must
    /// reject overlapping linear input/output views, not silently execute them in place.
    fn record(&self, commands: &CommandList) -> Result<RecordedBatch, ExecutionError>;

    /// Submit a recorded batch for execution.
    ///
    /// Commands execute in list order. Concurrent submissions may be serialized in an
    /// unspecified order; callers must express data dependencies explicitly rather than
    /// relying on host thread scheduling. Validation or queue rejection before execution
    /// begins leaves tensor values unchanged. An error discovered after execution begins
    /// may leave destinations partially updated or indeterminate; a backend must still
    /// retain all resources until any launched work has stopped using them.
    fn submit(&self, batch: &RecordedBatch) -> Result<Submission, ExecutionError>;

    /// Wait until a submission is complete and its tensor writes are host-visible.
    ///
    /// Waiting more than once, including through cloned tokens, is valid.
    fn wait(&self, submission: &Submission) -> Result<(), ExecutionError>;
}

#[derive(Debug)]
struct CpuEncodedWeight {
    bytes: Arc<Vec<u8>>,
}

#[derive(Debug)]
struct CpuPreparedWeight {
    bytes: Arc<Vec<u8>>,
    kind: GgmlType,
    rows: usize,
    cols: usize,
}

#[derive(Debug)]
struct CpuTensor {
    values: Mutex<Vec<f32>>,
}

#[derive(Debug, Clone)]
struct CpuRecordedBatch {
    commands: Vec<Command>,
}

#[derive(Debug)]
struct CpuSubmission;

/// Sequential correctness implementation of [`ExecutionDevice`].
///
/// It deliberately delegates arithmetic to the permanent [`Cpu`] reference backend.
/// Tensor values are mutex-protected so handles remain `Send + Sync`, while a shared
/// queue lock prevents commands from different submissions from interleaving. A clone
/// is another handle to the same logical device and can use all existing resources.
#[derive(Debug, Clone)]
pub struct CpuExecutionDevice {
    owner: ExecutionOwner,
    queue: Arc<Mutex<()>>,
    cpu: Cpu,
}

impl CpuExecutionDevice {
    /// Construct a new isolated CPU execution device.
    pub fn new() -> Self {
        Self {
            owner: ExecutionOwner::fresh(),
            queue: Arc::new(Mutex::new(())),
            cpu: Cpu::new(),
        }
    }

    fn require_owner(
        &self,
        actual: ExecutionOwner,
        resource: &'static str,
    ) -> Result<(), ExecutionError> {
        self.owner.require(actual, resource)
    }

    fn queue(&self) -> Result<MutexGuard<'_, ()>, ExecutionError> {
        self.queue
            .lock()
            .map_err(|_| ExecutionError::Synchronization {
                device: self.name(),
                resource: "CPU execution queue",
            })
    }

    fn tensor<'a>(
        &self,
        tensor: &'a F32Tensor,
        resource: &'static str,
    ) -> Result<&'a CpuTensor, ExecutionError> {
        self.require_owner(tensor.owner(), resource)?;
        tensor
            .payload::<CpuTensor>()
            .ok_or(ExecutionError::IncompatibleResource {
                device: self.name(),
                resource,
            })
    }

    fn prepared_weight<'a>(
        &self,
        weight: &'a PreparedWeight,
    ) -> Result<&'a CpuPreparedWeight, ExecutionError> {
        self.require_owner(weight.owner(), "prepared weight")?;
        weight
            .payload::<CpuPreparedWeight>()
            .ok_or(ExecutionError::IncompatibleResource {
                device: self.name(),
                resource: "prepared weight",
            })
    }

    fn lock_values<'a>(
        &self,
        tensor: &'a CpuTensor,
        resource: &'static str,
    ) -> Result<MutexGuard<'a, Vec<f32>>, ExecutionError> {
        tensor
            .values
            .lock()
            .map_err(|_| ExecutionError::Synchronization {
                device: self.name(),
                resource,
            })
    }

    fn snapshot_view(&self, view: &F32TensorView) -> Result<Vec<f32>, ExecutionError> {
        let tensor = self.tensor(view.tensor(), "tensor view")?;
        let values = self.lock_values(tensor, "tensor values")?;
        let end = view.offset.checked_add(view.len).ok_or_else(|| {
            ExecutionError::Shape("tensor view end overflows during execution".into())
        })?;
        let source = values.get(view.offset..end).ok_or_else(|| {
            ExecutionError::Shape(format!(
                "tensor view {}..{end} exceeds tensor length {} during execution",
                view.offset,
                values.len()
            ))
        })?;
        try_copy_f32(source, self.name(), "tensor snapshot")
    }

    fn replace_view(
        &self,
        view: &F32TensorView,
        replacement: &[f32],
    ) -> Result<(), ExecutionError> {
        if replacement.len() != view.len {
            return Err(ExecutionError::Shape(format!(
                "tensor write expected {} values, got {}",
                view.len,
                replacement.len()
            )));
        }
        let tensor = self.tensor(view.tensor(), "tensor view")?;
        let mut values = self.lock_values(tensor, "tensor values")?;
        let end = view.offset.checked_add(view.len).ok_or_else(|| {
            ExecutionError::Shape("tensor view end overflows during execution".into())
        })?;
        let destination = values.get_mut(view.offset..end).ok_or_else(|| {
            ExecutionError::Shape(format!(
                "tensor view {}..{end} exceeds tensor length during execution",
                view.offset
            ))
        })?;
        destination.copy_from_slice(replacement);
        Ok(())
    }

    fn validate_command(&self, command: &Command) -> Result<(), ExecutionError> {
        match command {
            Command::EmbeddingRow {
                output,
                weight,
                row,
            } => {
                validate_embedding_shape(output.len(), weight, *row)?;
                self.embedding_bytes(weight, *row)?;
                let tensor = self.tensor(output.tensor(), "embedding output tensor")?;
                let values = self.lock_values(tensor, "embedding output tensor")?;
                if values.len() != output.tensor().len()
                    || output
                        .offset
                        .checked_add(output.len)
                        .is_none_or(|end| end > values.len())
                {
                    return Err(ExecutionError::Shape(
                        "embedding output metadata differs from its storage".into(),
                    ));
                }
            }
            Command::Linear {
                output,
                input,
                weight,
                ..
            } => {
                self.tensor(output.tensor(), "linear output tensor")?;
                self.tensor(input.tensor(), "linear input tensor")?;
                self.prepared_weight(weight)?;
            }
            Command::SwiGlu { gate, up, .. } => {
                self.tensor(gate.tensor(), "SwiGLU gate tensor")?;
                self.tensor(up.tensor(), "SwiGLU up tensor")?;
            }
            Command::SigmoidGate {
                output,
                input,
                gate,
            } => {
                self.tensor(output.tensor(), "sigmoid output tensor")?;
                self.tensor(input.tensor(), "sigmoid input tensor")?;
                self.tensor(gate.tensor(), "sigmoid gate tensor")?;
            }
            Command::RmsNorm {
                output,
                input,
                weight,
                ..
            } => {
                self.tensor(output.tensor(), "RMSNorm output tensor")?;
                self.tensor(input.tensor(), "RMSNorm input tensor")?;
                if let Some(weight) = weight {
                    self.tensor(weight.tensor(), "RMSNorm weight tensor")?;
                }
            }
            Command::Rope { x, cos, sin, .. } => {
                self.tensor(x.tensor(), "RoPE input/output tensor")?;
                self.tensor(cos.tensor(), "RoPE cosine tensor")?;
                self.tensor(sin.tensor(), "RoPE sine tensor")?;
            }
            Command::IncrementalCausalAttention {
                output,
                q,
                k_cache,
                v_cache,
                config,
            } => {
                validate_incremental_attention_shape(
                    output.len(),
                    q.len(),
                    k_cache.len(),
                    v_cache.len(),
                    *config,
                )?;
                for (view, resource) in [
                    (output, "attention output tensor"),
                    (q, "attention query tensor"),
                    (k_cache, "attention key cache"),
                    (v_cache, "attention value cache"),
                ] {
                    let tensor = self.tensor(view.tensor(), resource)?;
                    let values = self.lock_values(tensor, resource)?;
                    let end = view.offset.checked_add(view.len).ok_or_else(|| {
                        ExecutionError::Shape("attention tensor view extent overflows".into())
                    })?;
                    if values.len() != view.tensor().len() || end > values.len() {
                        return Err(ExecutionError::Shape(
                            "attention tensor view metadata differs from its storage".into(),
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    fn execute_command(&self, command: &Command) -> Result<(), ExecutionError> {
        match command {
            Command::EmbeddingRow {
                output,
                weight,
                row,
            } => {
                let bytes = self.embedding_bytes(weight, *row)?;
                let mut result = try_zeroed_f32(output.len(), self.name(), "embedding row result")?;
                self.cpu.dequantize(
                    &mut result,
                    &Weight {
                        bytes,
                        kind: weight.kind(),
                        rows: 1,
                        cols: weight.cols(),
                    },
                )?;
                self.replace_view(output, &result)
            }
            Command::Linear {
                output,
                input,
                weight,
                mode,
            } => {
                // Snapshot every read operand before locking the destination. Besides
                // avoiding self-deadlock, this defines deterministic overlapping-view
                // behavior shared with future accelerated implementations.
                let input_values = self.snapshot_view(input)?;
                let prepared = self.prepared_weight(weight)?;
                let packed = Weight {
                    bytes: prepared.bytes.as_slice(),
                    kind: weight.kind,
                    rows: weight.rows,
                    cols: weight.cols,
                };
                let mut result = match mode {
                    LinearMode::Overwrite => {
                        try_zeroed_f32(output.len, self.name(), "linear overwrite result")?
                    }
                    LinearMode::ScaledAdd { .. } => self.snapshot_view(output)?,
                };
                match mode {
                    LinearMode::Overwrite => {
                        self.cpu.matvec(&mut result, &input_values, &packed)?;
                    }
                    LinearMode::ScaledAdd { scale } => {
                        self.cpu
                            .matvec_add(&mut result, &input_values, &packed, *scale)?;
                    }
                }
                self.replace_view(output, &result)
            }
            Command::SwiGlu { gate, up, limit } => {
                let mut gate_values = self.snapshot_view(gate)?;
                let up_values = self.snapshot_view(up)?;
                self.cpu.swiglu(&mut gate_values, &up_values, *limit)?;
                self.replace_view(gate, &gate_values)
            }
            Command::SigmoidGate {
                output,
                input,
                gate,
            } => {
                let mut result = self.snapshot_view(input)?;
                let gates = self.snapshot_view(gate)?;
                for (value, gate) in result.iter_mut().zip(gates) {
                    let tail = (-gate.abs()).exp();
                    let sigmoid = if gate >= 0.0 {
                        1.0 / (1.0 + tail)
                    } else {
                        tail / (1.0 + tail)
                    };
                    *value *= sigmoid;
                }
                self.replace_view(output, &result)
            }
            Command::RmsNorm {
                output,
                input,
                weight,
                eps,
            } => {
                let input_values = self.snapshot_view(input)?;
                let weight_values = weight
                    .as_ref()
                    .map(|weight| self.snapshot_view(weight))
                    .transpose()?;
                let mut result = try_zeroed_f32(output.len, self.name(), "RMSNorm result")?;
                self.cpu
                    .rmsnorm(&mut result, &input_values, weight_values.as_deref(), *eps)?;
                self.replace_view(output, &result)
            }
            Command::Rope {
                x,
                cos,
                sin,
                layout,
                inverse,
            } => {
                let mut values = self.snapshot_view(x)?;
                let cos_values = self.snapshot_view(cos)?;
                let sin_values = self.snapshot_view(sin)?;
                self.cpu
                    .rope(&mut values, &cos_values, &sin_values, *layout, *inverse)?;
                self.replace_view(x, &values)
            }
            Command::IncrementalCausalAttention {
                output,
                q,
                k_cache,
                v_cache,
                config,
            } => {
                // Snapshot all reads before any output write, including overlapping
                // cache/output views. The reference owns all attention arithmetic.
                let queries = self.snapshot_view(q)?;
                let keys = self.snapshot_view(k_cache)?;
                let values = self.snapshot_view(v_cache)?;
                let mut result = try_zeroed_f32(output.len(), self.name(), "attention result")?;
                self.cpu.incremental_causal_attention(
                    &mut result,
                    &queries,
                    &keys,
                    &values,
                    *config,
                )?;
                self.replace_view(output, &result)
            }
        }
    }

    // Decode only the selected packed row; never expand the entire vocabulary.
    fn embedding_bytes<'a>(
        &self,
        weight: &'a PreparedWeight,
        row: usize,
    ) -> Result<&'a [u8], ExecutionError> {
        let prepared = self.prepared_weight(weight)?;
        if (
            weight.kind(),
            weight.rows(),
            weight.cols(),
            weight.byte_len(),
        ) != (
            prepared.kind,
            prepared.rows,
            prepared.cols,
            prepared.bytes.len(),
        ) {
            return Err(ExecutionError::IncompatibleResource {
                device: self.name(),
                resource: "embedding prepared weight",
            });
        }
        let packed = Weight {
            bytes: prepared.bytes.as_slice(),
            kind: weight.kind(),
            rows: weight.rows(),
            cols: weight.cols(),
        };
        validate_weight_storage(self.name(), &packed)?;
        validate_embedding_shape(weight.cols(), weight, row)?;
        let (elements, bytes) =
            kquant::block_geometry(weight.kind().0).expect("validated packed geometry");
        let stride = (weight.cols() / elements)
            .checked_mul(bytes)
            .ok_or_else(|| ExecutionError::Shape("embedding row byte count overflows".into()))?;
        let start = row
            .checked_mul(stride)
            .ok_or_else(|| ExecutionError::Shape("embedding row offset overflows".into()))?;
        let end = start
            .checked_add(stride)
            .ok_or_else(|| ExecutionError::Shape("embedding row extent overflows".into()))?;
        prepared.bytes.get(start..end).ok_or_else(|| {
            ExecutionError::Shape("embedding row exceeds owned weight storage".into())
        })
    }
}

impl Default for CpuExecutionDevice {
    fn default() -> Self {
        Self::new()
    }
}

impl sealed::Sealed for CpuExecutionDevice {}

impl ExecutionDevice for CpuExecutionDevice {
    fn name(&self) -> &'static str {
        "cpu-sequential"
    }

    fn supports(&self, kind: GgmlType) -> bool {
        self.cpu.supports(kind)
    }

    fn finite_check_spec(&self) -> Option<FiniteCheckSpec> {
        Some(finite::CPU_FINITE_SPEC)
    }

    fn prepare_finite_check(&self) -> Result<FiniteCheck, ExecutionError> {
        self.prepare_cpu_finite_check()
    }

    fn check_finite(
        &self,
        check: &mut FiniteCheck,
        input: &F32TensorView,
        control: &mut dyn FnMut() -> Result<(), ExecutionError>,
    ) -> Result<(), ExecutionError> {
        self.check_cpu_finite(check, input, control)
    }

    fn upload_weight(&self, weight: &Weight<'_>) -> Result<EncodedWeight, ExecutionError> {
        validate_weight_storage(self.name(), weight)?;
        let bytes = Arc::new(try_copy_bytes(
            weight.bytes,
            self.name(),
            "CPU encoded weight",
        )?);
        let byte_len = bytes.len();
        Ok(EncodedWeight::new(
            self.owner,
            weight.kind,
            weight.rows,
            weight.cols,
            byte_len,
            Arc::new(CpuEncodedWeight { bytes }),
        ))
    }

    fn prepare_weight(&self, weight: &EncodedWeight) -> Result<PreparedWeight, ExecutionError> {
        self.require_owner(weight.owner(), "encoded weight")?;
        let encoded =
            weight
                .payload::<CpuEncodedWeight>()
                .ok_or(ExecutionError::IncompatibleResource {
                    device: self.name(),
                    resource: "encoded weight",
                })?;
        Ok(PreparedWeight::new(
            self.owner,
            weight.kind,
            weight.rows,
            weight.cols,
            weight.byte_len,
            Arc::new(CpuPreparedWeight {
                bytes: encoded.bytes.clone(),
                kind: weight.kind,
                rows: weight.rows,
                cols: weight.cols,
            }),
        ))
    }

    fn allocate_f32(&self, len: usize) -> Result<F32Tensor, ExecutionError> {
        let values = try_zeroed_f32(len, self.name(), "CPU f32 tensor")?;
        Ok(F32Tensor::new(
            self.owner,
            len,
            Arc::new(CpuTensor {
                values: Mutex::new(values),
            }),
        ))
    }

    fn write_f32(&self, destination: &F32TensorView, values: &[f32]) -> Result<(), ExecutionError> {
        self.require_owner(destination.owner(), "host-write tensor view")?;
        if destination.len != values.len() {
            return Err(ExecutionError::Shape(format!(
                "host write expected {} values, got {}",
                destination.len,
                values.len()
            )));
        }
        let _queue = self.queue()?;
        self.replace_view(destination, values)
    }

    fn read_f32(&self, source: &F32TensorView, values: &mut [f32]) -> Result<(), ExecutionError> {
        self.require_owner(source.owner(), "host-read tensor view")?;
        if source.len != values.len() {
            return Err(ExecutionError::Shape(format!(
                "host read expected {} values, got {}",
                source.len,
                values.len()
            )));
        }
        let _queue = self.queue()?;
        let snapshot = self.snapshot_view(source)?;
        values.copy_from_slice(&snapshot);
        Ok(())
    }

    fn record(&self, commands: &CommandList) -> Result<RecordedBatch, ExecutionError> {
        if let Some(owner) = commands.owner() {
            self.require_owner(owner, "command list")?;
        }
        for command in commands.commands() {
            self.validate_command(command)?;
        }
        Ok(RecordedBatch::new(
            self.owner,
            commands.commands().len(),
            Arc::new(CpuRecordedBatch {
                commands: commands.commands().to_vec(),
            }),
        ))
    }

    fn submit(&self, batch: &RecordedBatch) -> Result<Submission, ExecutionError> {
        self.require_owner(batch.owner(), "recorded batch")?;
        let batch =
            batch
                .payload::<CpuRecordedBatch>()
                .ok_or(ExecutionError::IncompatibleResource {
                    device: self.name(),
                    resource: "recorded batch",
                })?;
        let _queue = self.queue()?;
        for command in &batch.commands {
            self.execute_command(command)?;
        }
        Ok(Submission::new(self.owner, Arc::new(CpuSubmission)))
    }

    fn wait(&self, submission: &Submission) -> Result<(), ExecutionError> {
        self.require_owner(submission.owner(), "submission")?;
        submission
            .payload::<CpuSubmission>()
            .ok_or(ExecutionError::IncompatibleResource {
                device: self.name(),
                resource: "submission",
            })?;
        Ok(())
    }
}

fn validate_weight_storage(
    device: &'static str,
    weight: &Weight<'_>,
) -> Result<(), ExecutionError> {
    let (block_elements, block_bytes) =
        kquant::block_geometry(weight.kind.0).ok_or_else(|| DeviceError::UnsupportedFormat {
            device,
            kind: weight.kind.name(),
            rows: weight.rows,
            cols: weight.cols,
        })?;
    if !weight.cols.is_multiple_of(block_elements) {
        return Err(ExecutionError::Shape(format!(
            "{}x{} {} weight has {} elements per row, not a whole number of \
             {block_elements}-element blocks",
            weight.rows,
            weight.cols,
            weight.kind.name(),
            weight.cols
        )));
    }
    let expected_bytes = weight
        .rows
        .checked_mul(weight.cols / block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| {
            ExecutionError::Shape(format!(
                "weight byte count overflows for {}x{} {}",
                weight.rows,
                weight.cols,
                weight.kind.name()
            ))
        })?;
    if weight.bytes.len() != expected_bytes {
        return Err(ExecutionError::Shape(format!(
            "{}x{} {} weight needs {expected_bytes} bytes, got {}",
            weight.rows,
            weight.cols,
            weight.kind.name(),
            weight.bytes.len()
        )));
    }
    Ok(())
}

fn allocation_bytes(len: usize) -> Result<usize, ExecutionError> {
    len.checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| ExecutionError::Shape(format!("f32 element count {len} overflows bytes")))
}

fn try_zeroed_f32(
    len: usize,
    device: &'static str,
    purpose: &'static str,
) -> Result<Vec<f32>, ExecutionError> {
    let bytes = allocation_bytes(len)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| DeviceError::Allocation {
            device,
            bytes,
            purpose,
        })?;
    values.resize(len, 0.0);
    Ok(values)
}

fn try_copy_f32(
    source: &[f32],
    device: &'static str,
    purpose: &'static str,
) -> Result<Vec<f32>, ExecutionError> {
    let bytes = allocation_bytes(source.len())?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(source.len())
        .map_err(|_| DeviceError::Allocation {
            device,
            bytes,
            purpose,
        })?;
    values.extend_from_slice(source);
    Ok(values)
}

fn try_copy_bytes(
    source: &[u8],
    device: &'static str,
    purpose: &'static str,
) -> Result<Vec<u8>, ExecutionError> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(source.len())
        .map_err(|_| DeviceError::Allocation {
            device,
            bytes: source.len(),
            purpose,
        })?;
    bytes.extend_from_slice(source);
    Ok(bytes)
}

#[cfg(test)]
mod normalization_tests;

#[cfg(test)]
mod attention_tests;

#[cfg(test)]
mod embedding_tests;

#[cfg(test)]
mod sigmoid_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    fn prepare_f32(
        device: &dyn ExecutionDevice,
        values: &[f32],
        rows: usize,
        cols: usize,
    ) -> PreparedWeight {
        let bytes = f32_bytes(values);
        let encoded = device
            .upload_weight(&Weight {
                bytes: &bytes,
                kind: GgmlType(0),
                rows,
                cols,
            })
            .unwrap();
        device.prepare_weight(&encoded).unwrap()
    }

    fn read(
        device: &dyn ExecutionDevice,
        view: &F32TensorView,
    ) -> Result<Vec<f32>, ExecutionError> {
        let mut values = vec![0.0; view.len()];
        device.read_f32(view, &mut values)?;
        Ok(values)
    }

    fn submit_and_wait(device: &dyn ExecutionDevice, batch: &RecordedBatch) {
        let submission = device.submit(batch).unwrap();
        device.wait(&submission).unwrap();
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = 1.0e-6 * expected.abs().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual} expected={expected} tolerance={tolerance}"
            );
        }
    }

    #[test]
    fn execution_device_is_object_safe_send_and_sync() {
        fn require_send_sync<T: Send + Sync>() {}
        fn accept_object(_: &dyn ExecutionDevice) {}

        require_send_sync::<CpuExecutionDevice>();
        require_send_sync::<EncodedWeight>();
        require_send_sync::<PreparedWeight>();
        require_send_sync::<F32Tensor>();
        require_send_sync::<RecordedBatch>();
        require_send_sync::<Submission>();
        accept_object(&CpuExecutionDevice::new());
    }

    #[test]
    fn recorded_cpu_pipeline_is_a_sequential_oracle() {
        let device: Box<dyn ExecutionDevice> = Box::new(CpuExecutionDevice::new());
        let gate_weight = prepare_f32(device.as_ref(), &[1.0, 0.0, 0.0, 1.0], 2, 2);
        let up_weight = prepare_f32(device.as_ref(), &[2.0, 0.0, 0.0, 3.0], 2, 2);
        let down_weight = prepare_f32(device.as_ref(), &[1.0, 0.0, 0.0, 1.0], 2, 2);

        // One allocation deliberately hosts every activation. This is the layout an
        // arena-backed GPU implementation can mirror without changing command syntax.
        let activations = device
            .upload_f32(&[1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 10.0, 20.0])
            .unwrap();
        let input = activations.view(0, 2).unwrap();
        let gate = activations.view(2, 2).unwrap();
        let up = activations.view(4, 2).unwrap();
        let output = activations.view(6, 2).unwrap();

        let mut commands = CommandList::new();
        commands
            .linear(&gate, &input, &gate_weight, LinearMode::Overwrite)
            .unwrap();
        commands
            .linear(&up, &input, &up_weight, LinearMode::Overwrite)
            .unwrap();
        commands.swiglu(&gate, &up, 0.0).unwrap();
        commands
            .linear(
                &output,
                &gate,
                &down_weight,
                LinearMode::ScaledAdd { scale: 0.5 },
            )
            .unwrap();

        let batch = device.record(&commands).unwrap();
        assert_eq!(batch.len(), 4);
        let submission = device.submit(&batch).unwrap();
        device.wait(&submission).unwrap();

        let silu = |x: f32| x / (1.0 + (-x).exp());
        let expected = [10.0 + 0.5 * silu(1.0) * 2.0, 20.0 + 0.5 * silu(-2.0) * -6.0];
        assert_close(&read(device.as_ref(), &output).unwrap(), &expected);
    }

    #[test]
    fn resources_are_owner_checked_at_every_boundary() {
        let first = CpuExecutionDevice::new();
        let first_clone = first.clone();
        let second = CpuExecutionDevice::new();
        let first_tensor = first.upload_f32(&[1.0, 2.0]).unwrap();
        let second_tensor = second.upload_f32(&[3.0, 4.0]).unwrap();
        let first_weight = prepare_f32(&first, &[1.0, 0.0, 0.0, 1.0], 2, 2);

        // A clone is the same logical device and may use existing storage.
        assert_eq!(
            read(&first_clone, &first_tensor.full_view()).unwrap(),
            vec![1.0, 2.0]
        );

        let mut mixed = CommandList::new();
        let error = mixed
            .linear(
                &first_tensor.full_view(),
                &second_tensor.full_view(),
                &first_weight,
                LinearMode::Overwrite,
            )
            .unwrap_err();
        assert!(matches!(error, ExecutionError::ForeignResource { .. }));
        assert!(mixed.is_empty());

        let encoded_bytes = f32_bytes(&[1.0]);
        let encoded = first
            .upload_weight(&Weight {
                bytes: &encoded_bytes,
                kind: GgmlType(0),
                rows: 1,
                cols: 1,
            })
            .unwrap();
        assert!(matches!(
            second.prepare_weight(&encoded),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert!(matches!(
            second.write_f32(&first_tensor.full_view(), &[0.0, 0.0]),
            Err(ExecutionError::ForeignResource { .. })
        ));
        assert!(matches!(
            second.read_f32(&first_tensor.full_view(), &mut [0.0, 0.0]),
            Err(ExecutionError::ForeignResource { .. })
        ));

        let mut commands = CommandList::new();
        commands
            .linear(
                &first_tensor.full_view(),
                &first_tensor.full_view(),
                &first_weight,
                LinearMode::Overwrite,
            )
            .unwrap();
        assert!(matches!(
            second.record(&commands),
            Err(ExecutionError::ForeignResource { .. })
        ));
        let batch = first.record(&commands).unwrap();
        assert!(matches!(
            second.submit(&batch),
            Err(ExecutionError::ForeignResource { .. })
        ));
        let submission = first.submit(&batch).unwrap();
        assert!(matches!(
            second.wait(&submission),
            Err(ExecutionError::ForeignResource { .. })
        ));
    }

    #[test]
    fn command_validation_is_eager_and_transactional() {
        let device = CpuExecutionDevice::new();
        let weight = prepare_f32(&device, &[1.0, 0.0, 0.0, 1.0], 2, 2);
        let two = device.allocate_f32(2).unwrap();
        let one = device.allocate_f32(1).unwrap();
        let mut commands = CommandList::new();

        assert!(matches!(
            commands.linear(
                &two.full_view(),
                &one.full_view(),
                &weight,
                LinearMode::Overwrite
            ),
            Err(ExecutionError::Shape(_))
        ));
        assert!(commands.is_empty());
        assert!(matches!(
            commands.linear(
                &two.full_view(),
                &two.full_view(),
                &weight,
                LinearMode::ScaledAdd { scale: f32::NAN }
            ),
            Err(ExecutionError::Shape(_))
        ));
        assert!(commands.is_empty());
        assert!(matches!(
            commands.swiglu(&two.full_view(), &one.full_view(), 0.0),
            Err(ExecutionError::Shape(_))
        ));
        assert!(matches!(
            commands.swiglu(&two.full_view(), &two.full_view(), -1.0),
            Err(ExecutionError::Shape(_))
        ));
        assert!(commands.is_empty());

        commands
            .linear(
                &two.full_view(),
                &two.full_view(),
                &weight,
                LinearMode::Overwrite,
            )
            .unwrap();
        assert_eq!(commands.len(), 1);
    }

    #[test]
    fn tensor_views_and_host_transfers_are_checked() {
        let device = CpuExecutionDevice::new();
        let tensor = device.upload_f32(&[1.0, 2.0, 3.0]).unwrap();
        assert!(tensor.view(3, 0).unwrap().is_empty());
        assert!(matches!(tensor.view(2, 2), Err(ExecutionError::Shape(_))));
        assert!(matches!(
            tensor.view(usize::MAX, 2),
            Err(ExecutionError::Shape(_))
        ));

        let tail = tensor.view(1, 2).unwrap();
        assert_eq!(tail.offset(), 1);
        assert_eq!(tail.view(1, 1).unwrap().offset(), 2);
        assert!(tail.overlaps(&tensor.view(0, 2).unwrap()));
        assert!(!tail.overlaps(&tensor.view(0, 1).unwrap()));
        assert!(!tail.overlaps(&device.allocate_f32(2).unwrap().full_view()));
        assert!(!tail.overlaps(&tensor.view(3, 0).unwrap()));
        assert!(matches!(tail.view(2, 1), Err(ExecutionError::Shape(_))));
        assert!(matches!(
            device.write_f32(&tail, &[9.0]),
            Err(ExecutionError::Shape(_))
        ));
        assert!(matches!(
            device.read_f32(&tail, &mut [0.0]),
            Err(ExecutionError::Shape(_))
        ));
        device.write_f32(&tail, &[8.0, 9.0]).unwrap();
        assert_eq!(
            read(&device, &tensor.full_view()).unwrap(),
            vec![1.0, 8.0, 9.0]
        );
    }

    #[test]
    fn overlapping_views_use_snapshot_semantics() {
        let device = CpuExecutionDevice::new();
        let identity = prepare_f32(&device, &[1.0, 0.0, 0.0, 1.0], 2, 2);
        let tensor = device.upload_f32(&[1.0, 2.0, 99.0]).unwrap();
        let input = tensor.view(0, 2).unwrap();
        let output = tensor.view(1, 2).unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(&output, &input, &identity, LinearMode::Overwrite)
            .unwrap();
        let batch = device.record(&commands).unwrap();
        submit_and_wait(&device, &batch);
        assert_eq!(
            read(&device, &tensor.full_view()).unwrap(),
            vec![1.0, 1.0, 2.0]
        );

        let same = device.upload_f32(&[-2.0, 3.0]).unwrap();
        let mut commands = CommandList::new();
        commands
            .swiglu(&same.full_view(), &same.full_view(), 0.0)
            .unwrap();
        submit_and_wait(&device, &device.record(&commands).unwrap());
        let silu = |x: f32| x / (1.0 + (-x).exp());
        assert_close(
            &read(&device, &same.full_view()).unwrap(),
            &[silu(-2.0) * -2.0, silu(3.0) * 3.0],
        );
    }

    #[test]
    fn recording_is_a_snapshot_and_batches_are_reusable() {
        let device = CpuExecutionDevice::new();
        let identity = prepare_f32(&device, &[1.0], 1, 1);
        let input = device.upload_f32(&[4.0]).unwrap();
        let output = device.upload_f32(&[10.0]).unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &identity,
                LinearMode::Overwrite,
            )
            .unwrap();
        let overwrite_only = device.record(&commands).unwrap();

        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &identity,
                LinearMode::ScaledAdd { scale: 0.5 },
            )
            .unwrap();
        submit_and_wait(&device, &overwrite_only);
        assert_eq!(read(&device, &output.full_view()).unwrap(), vec![4.0]);

        let overwrite_and_add = device.record(&commands).unwrap();
        submit_and_wait(&device, &overwrite_and_add);
        assert_eq!(read(&device, &output.full_view()).unwrap(), vec![6.0]);
        submit_and_wait(&device, &overwrite_and_add);
        assert_eq!(read(&device, &output.full_view()).unwrap(), vec![6.0]);
    }

    #[test]
    fn malformed_and_unsupported_weights_are_refused_on_upload() {
        let device = CpuExecutionDevice::new();
        let short = [0_u8; 3];
        assert!(matches!(
            device.upload_weight(&Weight {
                bytes: &short,
                kind: GgmlType(0),
                rows: 1,
                cols: 1,
            }),
            Err(ExecutionError::Shape(_))
        ));
        assert!(matches!(
            device.upload_weight(&Weight {
                bytes: &[],
                kind: GgmlType(8),
                rows: 1,
                cols: 31,
            }),
            Err(ExecutionError::Shape(_))
        ));
        assert!(matches!(
            device.upload_weight(&Weight {
                bytes: &[],
                kind: GgmlType(999),
                rows: 0,
                cols: 0,
            }),
            Err(ExecutionError::Device(
                DeviceError::UnsupportedFormat { .. }
            ))
        ));
        assert!(matches!(
            device.upload_weight(&Weight {
                bytes: &[],
                kind: GgmlType(0),
                rows: usize::MAX,
                cols: 2,
            }),
            Err(ExecutionError::Shape(_))
        ));
    }

    #[test]
    fn packed_bf16_linear_is_bit_identical_to_the_immediate_cpu_oracle() {
        let device = CpuExecutionDevice::new();
        // Exactly representable BF16 values: [1, 2, -1; 0.5, -2, 4].
        let bits = [0x3f80_u16, 0x4000, 0xbf80, 0x3f00, 0xc000, 0x4080];
        let bytes: Vec<u8> = bits.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        let borrowed = Weight {
            bytes: &bytes,
            kind: GgmlType(30),
            rows: 2,
            cols: 3,
        };
        let encoded = device.upload_weight(&borrowed).unwrap();
        let weight = device.prepare_weight(&encoded).unwrap();
        assert_eq!(weight.kind(), GgmlType(30));
        assert_eq!(weight.byte_len(), bytes.len());

        let input_values = [0.25, -0.5, 2.0];
        let input = device.upload_f32(&input_values).unwrap();
        let output = device.allocate_f32(2).unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &weight,
                LinearMode::Overwrite,
            )
            .unwrap();
        submit_and_wait(&device, &device.record(&commands).unwrap());

        let mut immediate = [0.0; 2];
        Cpu::new()
            .matvec(&mut immediate, &input_values, &borrowed)
            .unwrap();
        assert_eq!(read(&device, &output.full_view()).unwrap(), immediate);
    }

    #[test]
    fn cloned_device_serializes_concurrent_submissions() {
        let device = Arc::new(CpuExecutionDevice::new());
        let identity = prepare_f32(device.as_ref(), &[1.0], 1, 1);
        let input = device.upload_f32(&[1.0]).unwrap();
        let output = device.allocate_f32(1).unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &identity,
                LinearMode::ScaledAdd { scale: 1.0 },
            )
            .unwrap();
        let batch = device.record(&commands).unwrap();

        std::thread::scope(|scope| {
            for _ in 0..4 {
                let device = device.clone();
                let batch = batch.clone();
                scope.spawn(move || {
                    for _ in 0..25 {
                        submit_and_wait(device.as_ref(), &batch);
                    }
                });
            }
        });
        assert_eq!(
            read(device.as_ref(), &output.full_view()).unwrap(),
            vec![100.0]
        );
    }

    #[test]
    fn zero_width_linear_reduction_matches_immediate_cpu_contract() {
        let device = CpuExecutionDevice::new();
        let weight = prepare_f32(&device, &[], 2, 0);
        let input = device.allocate_f32(0).unwrap();
        let output = device.upload_f32(&[7.0, 8.0]).unwrap();
        let mut commands = CommandList::new();
        commands
            .linear(
                &output.full_view(),
                &input.full_view(),
                &weight,
                LinearMode::Overwrite,
            )
            .unwrap();
        submit_and_wait(&device, &device.record(&commands).unwrap());
        assert_eq!(read(&device, &output.full_view()).unwrap(), vec![0.0, 0.0]);
    }

    #[test]
    fn empty_batches_and_allocation_overflow_are_well_defined() {
        let device = CpuExecutionDevice::new();
        let batch = device.record(&CommandList::new()).unwrap();
        assert!(batch.is_empty());
        let submission = device.submit(&batch).unwrap();
        device.wait(&submission).unwrap();

        assert!(matches!(
            device.allocate_f32(usize::MAX),
            Err(ExecutionError::Shape(_))
        ));
    }
}
