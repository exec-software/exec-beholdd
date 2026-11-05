//! Tensors, dtypes and quantisation formats. The vocabulary every other crate speaks.
//!
//! Owns: shapes, dtype and quant-format definitions (MXFP4/E2M1, FP8-E4M3, E8M0 scales,
//! bf16), and the decode semantics of each.
//!
//! Does NOT own: any I/O, any arithmetic. If it reads a file or multiplies a matrix it is
//! in the wrong crate.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod quant;

pub use quant::{
    bf16_to_f32, e2m1_decode, e2m1_nearest, e4m3_decode, e4m3_decode_fast, e4m3_nearest,
    e8m0_scale, f16_to_f32, ggml_block_geometry, mxfp4_unpack, round_scale, DType, E2M1,
};
