//! Native HAG readiness, deliberately separate from inference backend selection.
//!
//! HAG owns hardware resources, not Beholdd's model stack. Its initial Linux memory
//! transport alone does not qualify inference. The separate `hag-native` feature
//! exposes a bounded experimental `ExecutionDevice` for resident BF16/Q8_0 linear,
//! SwiGLU and residual operations, not a registered serving backend. `hag` alone enables
//! preflight only. This module never creates a Vulkan context or falls back to CPU.

use crate::DeviceError;

#[cfg(any(test, all(feature = "hag-native", target_os = "linux")))]
mod fixtures;
#[cfg(all(feature = "hag-native", target_os = "linux"))]
mod kernels;
#[cfg(all(feature = "hag-native", target_os = "linux"))]
pub mod linear_probe;
#[cfg(all(feature = "hag-native", target_os = "linux"))]
pub mod native_execution;

/// Kernel-reported identity of a device that HAG can open.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceSummary {
    /// Linux render-node path, for diagnostics only; never a portable device handle.
    pub render_node: String,
    /// Kernel-reported PCI device ID.
    pub device_id: u32,
    /// Kernel-reported AMDGPU family ID, not a guessed ISA target.
    pub family_id: u32,
    /// Kernel-reported active compute-unit count.
    pub compute_units: u32,
}

/// Why HAG cannot be used for inference in this build/on this host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Readiness {
    /// The native Linux memory transport is not available on this target.
    UnsupportedPlatform,
    /// No accessible AMDGPU render node was discovered; see diagnostics.
    NoUsableDevice,
    /// Device/memory support exists, but general native compute is not implemented.
    MemoryOnly,
}

/// Read-only preflight result. This is not a list of available serving backends.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Preflight {
    /// Compiled HAG functionality, distinct from runtime device discovery.
    pub implementation: exec_hag::ImplementationStatus,
    /// Current readiness; no variant authorizes execution yet.
    pub readiness: Readiness,
    /// Devices that successfully answered the native driver queries.
    pub devices: Vec<DeviceSummary>,
    /// Discovery rejections/errors with permission and device details preserved.
    pub diagnostics: Vec<String>,
    /// The explicit native-compute blocker, regardless of discovered hardware.
    pub compute_blocker: String,
}

/// Inspect native memory readiness without allocating buffers or submitting work.
///
/// Only DRM render nodes are opened. No model, shader compiler, Vulkan loader or
/// inference service is involved, and devices are closed before this call returns.
pub fn preflight() -> Preflight {
    #[cfg(target_os = "linux")]
    let (devices, diagnostics) = discover();
    #[cfg(not(target_os = "linux"))]
    let (devices, diagnostics) = (Vec::new(), Vec::new());
    // Keeping construction shared makes platform/device classification independently
    // testable without opening /dev/dri in the ordinary test suite.
    summarize(exec_hag::implementation_status(), devices, diagnostics)
}

#[cfg(target_os = "linux")]
fn discover() -> (Vec<DeviceSummary>, Vec<String>) {
    match exec_hag::linux::AmdGpuDevice::discover_report() {
        Ok(report) => {
            let devices = report
                .devices
                .iter()
                .map(|device| {
                    let info = device.info();
                    DeviceSummary {
                        render_node: info.render_node.display().to_string(),
                        device_id: info.device_id,
                        family_id: info.family_id,
                        compute_units: info.compute_units,
                    }
                })
                .collect();
            let diagnostics = report
                .rejected
                .into_iter()
                .map(|rejected| format!("{}: {}", rejected.path.display(), rejected.error))
                .collect();
            (devices, diagnostics)
        }
        Err(error) => (Vec::new(), vec![error.to_string()]),
    }
}

fn summarize(
    implementation: exec_hag::ImplementationStatus,
    devices: Vec<DeviceSummary>,
    mut diagnostics: Vec<String>,
) -> Preflight {
    let readiness = if !implementation.linux_drm_memory {
        diagnostics
            .push("HAG native memory currently requires Linux with linux-drm enabled".into());
        Readiness::UnsupportedPlatform
    } else if devices.is_empty() {
        diagnostics.push("No usable AMDGPU render nodes were discovered".into());
        Readiness::NoUsableDevice
    } else {
        Readiness::MemoryOnly
    };
    Preflight {
        implementation,
        readiness,
        devices,
        diagnostics,
        compute_blocker: exec_hag::ComputeUnavailable.to_string(),
    }
}

/// Refuse a native execution request instead of hiding CPU or Vulkan fallback.
///
/// The bounded experimental `hag-native` execution adapter is not registered with
/// the serving runtime. General native inference stays unavailable until full
/// model operators, numerical behavior and hardware lifetime are qualified.
pub fn require_compute() -> Result<(), DeviceError> {
    exec_hag::require_native_compute().map_err(|error| DeviceError::Backend {
        device: "hag",
        message: error.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn implementation(linux_drm_memory: bool) -> exec_hag::ImplementationStatus {
        exec_hag::ImplementationStatus {
            linux_drm_memory,
            native_compute: false,
        }
    }

    #[test]
    fn unsupported_targets_refuse_without_discovery() {
        let report = summarize(implementation(false), vec![], vec![]);
        assert_eq!(report.readiness, Readiness::UnsupportedPlatform);
        assert!(!report.implementation.native_compute);
        assert!(report.diagnostics[0].contains("requires Linux"));
    }

    #[test]
    fn no_device_report_preserves_discovery_errors() {
        let report = summarize(
            implementation(true),
            vec![],
            vec!["/dev/dri/renderD128: Permission denied".into()],
        );
        assert_eq!(report.readiness, Readiness::NoUsableDevice);
        assert!(report.diagnostics[0].contains("Permission denied"));
        assert_eq!(report.diagnostics.len(), 2);
    }

    #[test]
    fn discovered_devices_do_not_authorize_compute() {
        let device = DeviceSummary {
            render_node: "/dev/dri/renderD128".into(),
            device_id: 0x1506,
            family_id: 150,
            compute_units: 40,
        };
        let report = summarize(implementation(true), vec![device.clone()], vec![]);
        assert_eq!(report.readiness, Readiness::MemoryOnly);
        assert_eq!(report.devices, [device]);
        assert!(!report.implementation.native_compute);
        assert!(report.compute_blocker.contains("not implemented"));
        let error = require_compute().unwrap_err();
        assert!(matches!(error, DeviceError::Backend { device: "hag", .. }));
        assert!(error.to_string().contains("no CPU or Vulkan fallback"));
    }
}
