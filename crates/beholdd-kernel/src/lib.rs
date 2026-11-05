//! The arithmetic, per ISA.
//!
//! Owns: matmul, matvec, norms, rope, softmax, activation quantisation -- with a scalar
//! reference that is never deleted, because it is the oracle everything else is diffed
//! against.
//!
//! Two results from the previous engine are baked into the design: decode belongs in the
//! register file (`vpermps` IS a 16-entry float table and E2M1 has exactly 16 values --
//! 4.2x), and a kernel must be able to report the GRANULARITY of the work it is about
//! to do, because region size and not thread count is what cost 9.5x.
//!
//! Does NOT own: threading policy. It reports; the scheduler decides.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod bf16;
pub mod fp8;
pub mod kquant;
pub mod mxfp4;
pub mod vecops;
