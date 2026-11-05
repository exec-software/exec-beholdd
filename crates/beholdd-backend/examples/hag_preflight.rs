//! Read-only HAG discovery. Does not allocate GPU buffers, load models or dispatch.

fn main() -> std::process::ExitCode {
    let report = beholdd_backend::hag::preflight();
    println!("HAG readiness: {:?}", report.readiness);
    println!(
        "Native Linux memory transport compiled: {}",
        report.implementation.linux_drm_memory
    );
    for device in report.devices {
        println!(
            "{}: device=0x{:04x}, family={}, compute units={}",
            device.render_node, device.device_id, device.family_id, device.compute_units
        );
    }
    for diagnostic in report.diagnostics {
        println!("Discovery: {diagnostic}");
    }
    println!("Compute unavailable: {}", report.compute_blocker);
    // A discovered GPU is not a passing compute check.
    std::process::ExitCode::FAILURE
}
