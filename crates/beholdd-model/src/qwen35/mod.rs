//! Native OpenMythos/Qwen3.5 text bring-up, separate from the Qwen3 model core.
//!
//! A strict checkpoint binder, bounded CPU references and explicit resident
//! operator components. These are development gates, not a serving capability:
//! the production dispatcher keeps
//! refusing `qwen35` until complete-model and backend qualification passes.

pub mod attention;
pub mod config;
pub mod delta;
pub mod recorded_attention;
pub mod recorded_delta;
pub mod recorded_ffn;
#[cfg(all(target_os = "linux", feature = "hag-native"))]
pub mod recorded_model;
pub mod recorded_weights;
pub mod reference;
pub mod weights;
