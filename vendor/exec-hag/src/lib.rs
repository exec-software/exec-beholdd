// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! HAG's standalone hardware foundation.
//!
//! Model formats, tokenizers, inference, and scheduling belong to callers.
//! Native Linux memory management is opt-in; general native compute is not implemented.
//! The `native-probe` feature exposes one closed experimental shader test;
//! `native-compute` adds an unsafe trusted-shader serial queue prototype. Neither
//! is a validated inference backend or a public arbitrary command-packet API.
//! In particular, successful discovery or allocation does not prove compute readiness.
//! There is no Vulkan dependency, shader compiler, or implicit CPU fallback.

#![forbid(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

pub mod budget;
#[cfg(all(target_os = "linux", feature = "linux-drm"))]
pub mod linux;

use std::fmt;

/// C ABI over the bounded compute queue, for non-Rust callers.
///
/// Gated on `capi` so a Rust-only build exports no C symbols. See
/// [`capi`] for the ownership and panic rules the boundary enforces.
#[cfg(all(feature = "capi", target_os = "linux"))]
pub mod capi;

/// What this build implements, independently of any installed hardware.
///
/// These are library implementation flags, not device capabilities. Discovery must
/// still succeed before allocating memory. They do not define a kernel ABI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImplementationStatus {
    /// The Linux AMDGPU discovery/allocation transport was compiled in.
    pub linux_drm_memory: bool,
    /// A validated general native compute backend is implemented (a proof is not one).
    pub native_compute: bool,
}

/// Describe the compiled implementation without opening a device or loading a driver.
pub const fn implementation_status() -> ImplementationStatus {
    ImplementationStatus {
        linux_drm_memory: cfg!(all(target_os = "linux", feature = "linux-drm")),
        native_compute: false,
    }
}

/// Native compute cannot yet be selected; discovering a GPU is insufficient.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeUnavailable;

impl fmt::Display for ComputeUnavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(
            "HAG general native compute is not implemented: production submission, completion, \
             and validated operator kernels are required; the fixed probe is not an inference backend; \
             no CPU or Vulkan fallback was selected",
        )
    }
}

impl std::error::Error for ComputeUnavailable {}

/// Refuse execution until HAG has a native, validated compute implementation.
///
/// Callers must not turn this error into an implicit fallback or advertise HAG as an
/// available inference backend. This function performs no device I/O.
pub fn require_native_compute() -> Result<(), ComputeUnavailable> {
    Err(ComputeUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discovery_support_does_not_claim_compute_support() {
        let status = implementation_status();
        assert_eq!(
            status.linux_drm_memory,
            cfg!(all(target_os = "linux", feature = "linux-drm"))
        );
        assert!(!status.native_compute);
        let error = require_native_compute().unwrap_err().to_string();
        assert!(error.contains("not implemented"));
        assert!(error.contains("no CPU or Vulkan fallback"));
    }
}
