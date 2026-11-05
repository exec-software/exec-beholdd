// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! Opt-in native proof. Ordinary invocation only inspects; no GPU allocation.

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use exec_hag::{budget::MemoryBudget, linux::probe};
    let arguments: Vec<_> = std::env::args().skip(1).collect();
    let (execute, path) = match arguments.as_slice() {
        [path] => (false, path),
        [flag, path] if flag == "--execute" => (true, path),
        _ => return Err("usage: native_probe [--execute] /dev/dri/renderD128".into()),
    };
    let information = probe::inspect(path)?;
    println!(
        "device=0x{:04x} family={} compute_ip={}.{} discovery=0x{:06x} rings=0x{:x} supported={}",
        information.device.device_id,
        information.device.family_id,
        information.ip_major,
        information.ip_minor,
        information.ip_discovery_version,
        information.available_rings,
        information.supported
    );
    if !execute {
        println!("Inspection only: no allocations or command submission. Use --execute only under resource/thermal supervision.");
        return Ok(());
    }
    let budget = MemoryBudget::new(64 * 1024);
    let result = probe::run(path, &budget)?;
    if budget.current_bytes() != 0 {
        return Err("proof cleanup did not return its budget".into());
    }
    println!("PASS native shader: {} output bytes exact; {} guard bytes intact; fence={}; allocation={} bytes; budget_after=0; wall_time_us={}",
        result.verified_bytes, result.guard_bytes, result.sequence, result.allocation_bytes, result.elapsed.as_micros());
    println!("Native submission proof only. No model inference, throughput claim, or serving backend promotion.");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("Native proof is supported only on Linux; no work was submitted.");
    std::process::exit(1);
}
