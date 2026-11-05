//! Architectures.
//!
//! Model definitions are an extension point independent of storage and serving.
//! Experimental architectures should use the same checked operators and execution
//! interfaces as the built-in models, rather than requiring a separate runtime.
//!
//! Planned extension levels, from configuration to new arithmetic:
//!   1. declarative -- a spec in `.beholdd/archs/*.yaml`, no code compiled
//!   2. composed    -- Rust assembling published operators
//!   3. novel op    -- a new kernel, which then becomes available to levels 1 and 2
//!
//! Dense `qwen3` and routed `qwen3moe` are level-2 architectures sharing one strict
//! attention/tokenizer core and distinct feed-forward binders.
//!
//! Does NOT own: where weights come from.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod artifact;
pub mod execution;
/// MiMo-V2 family (`mimo2`): hybrid sliding-window/full attention with sinks, asymmetric
/// key/value head widths, a split RoPE schedule and sigmoid expert routing. See ADR-0002.
pub mod mimo2;
pub mod qwen35;
pub mod qwen3moe;
pub mod tokenizer;

pub use qwen3moe::{
    BlockOutput, Config, DecodeOutput, ForwardOutput, GenerateOutput, GenerationFinishReason,
    Qwen3Moe, QwenArchitecture, QwenModel, StepRoute, WeightMemory,
};
pub use tokenizer::Qwen2Tokenizer;
