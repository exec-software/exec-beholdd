// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! Explicit, single-dispatch native compute proof; not an inference backend.
//!
//! This runs only a fixed AMD/IGT GFX11 buffer-fill shader on the initially
//! allowlisted device. No caller-supplied commands, ISA, addresses, or workloads are
//! accepted. All resources belong to a dedicated render-node connection.
//!
//! A finite fence wait is not a guarantee that every kernel ioctl is bounded.
//! A driver/GPU fault can affect other clients of the same GPU. Run only under
//! explicit process/memory/thermal supervision, never as an ordinary unit test.

use super::{
    os_error, probe_packets, submission_uapi as abi, uapi, AmdGpuDevice, Buffer, DeviceInfo, Error,
    Heap,
};
use crate::budget::MemoryBudget;
use std::fmt;
use std::io;
use std::mem::size_of;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::sync::atomic::{fence, Ordering};
use std::time::{Duration, Instant};

const PAGE: u64 = 4096;
const OUTPUT_BYTES: usize = probe_packets::OUTPUT_BYTES;
const GUARDED_BYTES: usize = OUTPUT_BYTES + 2 * PAGE as usize;
const TOTAL_BYTES: u64 = GUARDED_BYTES as u64 + 2 * PAGE;
const MAX_WAIT_NS: u64 = 2_000_000_000;
const CANARY: u8 = 0xa5;

#[cfg(test)]
mod lifecycle_tests;

/// Failure of the closed native proof, including precise kernel/cleanup errors.
#[derive(Debug)]
pub enum ProbeError {
    /// Native transport failure.
    Transport(Error),
    /// The device or geometry has not been qualified for this fixed proof.
    Unsupported(&'static str),
    /// Completion was not established by the finite fence deadline.
    Incomplete,
    /// A reset, device loss, or error was reported after the fence.
    ContextFault(u64),
    /// A reusable queue's data guard or alignment padding was modified.
    GuardCorruption {
        /// Queue-local data slot, numbered zero through two.
        slot: usize,
        /// Physical byte offset within that slot's guarded allocation.
        offset: u64,
        /// Observed byte instead of the initialized 0xa5 guard pattern.
        actual: u8,
    },
    /// Output or guard bytes differ from the exact CPU reference.
    Mismatch {
        /// Byte offset within the guarded allocation.
        offset: usize,
        /// CPU reference value.
        expected: u8,
        /// Observed byte after completion.
        actual: u8,
    },
}

impl fmt::Display for ProbeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transport(error) => error.fmt(f),
            Self::Unsupported(reason) => write!(f, "native proof refused: {reason}"),
            Self::Incomplete => f.write_str(
                "native proof completion unknown; resources quarantined until process exit",
            ),
            Self::ContextFault(flags) => write!(
                f,
                "native proof context fault 0x{flags:x}; resources quarantined"
            ),
            Self::GuardCorruption { slot, offset, actual } => write!(f,
                "compute slot {slot} guard byte {offset}: expected 0xa5, observed 0x{actual:02x}; queue poisoned"),
            Self::Mismatch {
                offset,
                expected,
                actual,
            } => write!(
                f,
                "native proof byte {offset}: expected 0x{expected:02x}, observed 0x{actual:02x}"
            ),
        }
    }
}
impl std::error::Error for ProbeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Transport(error) => Some(error),
            _ => None,
        }
    }
}
impl From<Error> for ProbeError {
    fn from(error: Error) -> Self {
        Self::Transport(error)
    }
}

/// Read-only compute-IP facts, not a declaration of a serving backend.
#[derive(Clone, Debug)]
pub struct ProbeInfo {
    /// Kernel-reported device identity.
    pub device: DeviceInfo,
    /// Kernel-reported compute IP major version.
    pub ip_major: u32,
    /// Kernel-reported compute IP minor version.
    pub ip_minor: u32,
    /// Discovery version: major in bits 23:16, minor 15:8, revision 7:0.
    pub ip_discovery_version: u32,
    /// Kernel-reported available compute rings.
    pub available_rings: u32,
    /// Whether this exact identity passes the deliberately narrow probe gate.
    pub supported: bool,
}

/// A successful exact comparison after completion and successful cleanup.
#[derive(Debug)]
pub struct ProbeResult {
    /// Device/IP that actually executed the shader.
    pub info: ProbeInfo,
    /// Kernel-returned submission sequence number.
    pub sequence: u64,
    /// Bytes written by the shader and matched against the reference.
    pub verified_bytes: usize,
    /// Guard bytes verified unchanged around the output.
    pub guard_bytes: usize,
    /// Total charged GPU allocation bytes (shader, IB, output and guards).
    pub allocation_bytes: u64,
    /// Whole proof wall time, including discovery/allocation/cleanup, not kernel time.
    pub elapsed: Duration,
}

pub(super) fn query_ip(device: &AmdGpuDevice) -> Result<abi::HwIpInfo, Error> {
    let mut result = abi::HwIpInfo::default();
    let mut request = uapi::Info {
        return_pointer: (&mut result as *mut abi::HwIpInfo) as usize as u64,
        return_size: size_of::<abi::HwIpInfo>() as u32,
        query: abi::INFO_HW_IP_INFO,
        query_data: [abi::HW_IP_COMPUTE, 0, 0, 0],
    };
    device
        .inner
        .ioctl(uapi::INFO, &mut request, "query compute IP")?;
    Ok(result)
}

pub(super) fn supports(device: &DeviceInfo, ip: &abi::HwIpInfo) -> bool {
    // The kernel registers gfx_v11_0_ip_block (11.0) for physical GC 11.5.1.
    // Use ip_discovery_version for the real hardware revision, not the software
    // block's compatibility minor, and keep the initial narrow PCI gate.
    device.device_id == 0x1586
        && device.family_id == 150
        && ip.hw_ip_version_major == 11
        && ip.hw_ip_version_minor == 0
        && ip.ip_discovery_version == 0x0b0501
        && ip.available_rings & 1 != 0
}

fn info(device: &AmdGpuDevice, ip: &abi::HwIpInfo) -> ProbeInfo {
    ProbeInfo {
        device: device.info().clone(),
        ip_major: ip.hw_ip_version_major,
        ip_minor: ip.hw_ip_version_minor,
        ip_discovery_version: ip.ip_discovery_version,
        available_rings: ip.available_rings,
        supported: supports(device.info(), ip),
    }
}

/// Inspect the identity gate without allocating GPU buffers or issuing commands.
pub fn inspect(path: impl AsRef<Path>) -> Result<ProbeInfo, ProbeError> {
    let device = AmdGpuDevice::open(path)?;
    Ok(info(&device, &query_ip(&device)?))
}

pub(super) fn once<T>(
    device: &AmdGpuDevice,
    request: libc::c_ulong,
    value: &mut T,
    operation: &'static str,
) -> Result<(), Error> {
    // SAFETY: private callers pair each request with its exact UAPI layout. This
    // path intentionally does not retry an interrupted state-changing operation.
    unsafe {
        device.inner.kernel.ioctl_once(
            device.inner.fd.as_raw_fd(),
            request,
            (value as *mut T).cast(),
        )
    }
    .map_err(|error| os_error(operation, error))
}

#[derive(Clone, Copy)]
pub(super) struct Mapping {
    pub(super) handle: u32,
    pub(super) address: u64,
    pub(super) size: u64,
}

pub(super) struct Resources {
    pub(super) device: AmdGpuDevice,
    pub(super) context: Option<u32>,
    pub(super) list: Option<u32>,
    pub(super) mappings: Vec<Mapping>,
    pub(super) buffers: Vec<Buffer>,
}

pub(super) struct Session {
    resources: Option<Resources>,
    pub(super) uncertain: bool,
}

impl Session {
    pub(super) fn new(device: AmdGpuDevice) -> Self {
        Self {
            resources: Some(Resources {
                device,
                context: None,
                list: None,
                mappings: Vec::new(),
                buffers: Vec::new(),
            }),
            uncertain: false,
        }
    }

    pub(super) fn resources(&self) -> &Resources {
        self.resources.as_ref().expect("live session")
    }
    pub(super) fn resources_mut(&mut self) -> &mut Resources {
        self.resources.as_mut().expect("live session")
    }

    fn quarantine(&mut self) {
        if let Some(resources) = self.resources.take() {
            resources
                .device
                .inner
                .cleanup_failures
                .fetch_add(1, Ordering::Relaxed);
            // Keep the complete ownership chain, including VA/context/list and every
            // GEM buffer/reservation. Killing a process is not cancellation of GPU work.
            std::mem::forget(resources);
        }
    }

    pub(super) fn close(&mut self) -> Result<(), ProbeError> {
        if self.uncertain {
            self.quarantine();
            return Err(ProbeError::Incomplete);
        }
        let Some(resources) = self.resources.as_mut() else {
            return Ok(());
        };
        if let Err(error) = cleanup(resources) {
            self.quarantine();
            return Err(error.into());
        }
        self.resources.take();
        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn cleanup(resources: &mut Resources) -> Result<(), Error> {
    if let Some(list_handle) = resources.list {
        let mut request = abi::BoList {
            input: abi::BoListIn {
                operation: abi::BO_LIST_DESTROY,
                list_handle,
                ..Default::default()
            },
        };
        once(
            &resources.device,
            abi::BO_LIST,
            &mut request,
            "destroy proof BO list",
        )?;
        resources.list = None;
    }
    while let Some(mapping) = resources.mappings.last().copied() {
        let mut request = abi::GemVa {
            handle: mapping.handle,
            operation: abi::VA_UNMAP,
            va_address: mapping.address,
            map_size: mapping.size,
            ..Default::default()
        };
        once(
            &resources.device,
            abi::GEM_VA,
            &mut request,
            "unmap proof GPU VA",
        )?;
        resources.mappings.pop();
    }
    if let Some(ctx_id) = resources.context {
        let mut request = abi::Context {
            input: abi::ContextIn {
                op: abi::CTX_FREE,
                ctx_id,
                ..Default::default()
            },
        };
        once(
            &resources.device,
            abi::CTX,
            &mut request,
            "free proof context",
        )?;
        resources.context = None;
    }
    while let Some(buffer) = resources.buffers.pop() {
        buffer.close()?;
    }
    Ok(())
}

fn checked_align(value: u64, alignment: u64) -> Result<u64, ProbeError> {
    if !alignment.is_power_of_two() {
        return Err(ProbeError::Unsupported("invalid device alignment"));
    }
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .ok_or(ProbeError::Unsupported("GPU VA overflow"))
}

fn plan_addresses(details: &uapi::DeviceInfo, ip: &abi::HwIpInfo) -> Result<[u64; 3], ProbeError> {
    if !details.virtual_address_alignment.is_power_of_two() {
        return Err(ProbeError::Unsupported("invalid queried GPU VA alignment"));
    }
    let alignment = u64::from(details.virtual_address_alignment)
        .max(u64::from(ip.ib_start_alignment))
        .max(PAGE);
    if alignment > 65536
        || !alignment.is_power_of_two()
        || ip.ib_start_alignment == 0
        || !ip.ib_start_alignment.is_power_of_two()
        || ip.ib_size_alignment == 0
        || !ip.ib_size_alignment.is_power_of_two()
        || ip.ib_size_alignment > PAGE as u32
    {
        return Err(ProbeError::Unsupported("unqualified VA/IB alignment"));
    }
    let mut next = checked_align(details.virtual_address_offset.max(PAGE), alignment)?;
    let mut addresses = [0; 3];
    for (index, size) in [PAGE, PAGE, GUARDED_BYTES as u64].into_iter().enumerate() {
        addresses[index] = next;
        let end = next
            .checked_add(size)
            .ok_or(ProbeError::Unsupported("GPU VA overflow"))?;
        if end > details.virtual_address_max || end >= 1 << 48 {
            return Err(ProbeError::Unsupported(
                "proof does not fit queried GPU VA range",
            ));
        }
        next = checked_align(end, alignment)?;
    }
    Ok(addresses)
}

fn deadline_ns(seconds: i64, nanoseconds: i64) -> Result<u64, ProbeError> {
    if seconds < 0 || !(0..1_000_000_000).contains(&nanoseconds) {
        return Err(ProbeError::Unsupported("invalid monotonic clock"));
    }
    let result = (seconds as u64)
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(nanoseconds as u64))
        .and_then(|value| value.checked_add(MAX_WAIT_NS))
        .filter(|value| *value <= i64::MAX as u64)
        .ok_or(ProbeError::Unsupported("finite fence deadline overflow"))?;
    Ok(result)
}

pub(super) fn completion_deadline() -> Result<u64, ProbeError> {
    let mut time: libc::timespec = unsafe { std::mem::zeroed() };
    // SAFETY: valid writable timespec; read-only monotonic clock query.
    if unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut time) } != 0 {
        return Err(os_error("read monotonic clock", io::Error::last_os_error()).into());
    }
    deadline_ns(time.tv_sec, time.tv_nsec)
}

fn verify(bytes: &[u8]) -> Result<(), ProbeError> {
    if bytes.len() != GUARDED_BYTES {
        return Err(ProbeError::Unsupported("invalid proof readback length"));
    }
    for (offset, actual) in bytes.iter().copied().enumerate() {
        let expected = if (PAGE as usize..PAGE as usize + OUTPUT_BYTES).contains(&offset) {
            probe_packets::FILL_BYTE
        } else {
            CANARY
        };
        if actual != expected {
            return Err(ProbeError::Mismatch {
                offset,
                expected,
                actual,
            });
        }
    }
    Ok(())
}

/// Execute one fixed native shader, verify all output and guard bytes, then clean up.
///
/// This is explicitly experimental and never called by normal inference. It uses
/// 32 KiB of charged GTT storage, one normal-priority context, ring zero, and a
/// two-second *completion wait* deadline. Unsupported identities/pages refuse
/// before allocation. Any uncertain submission or teardown retains its resources
/// until process exit. A timeout does not cancel GPU work or prevent a driver reset.
pub fn run(path: impl AsRef<Path>, budget: &MemoryBudget) -> Result<ProbeResult, ProbeError> {
    let started = Instant::now();
    let device = AmdGpuDevice::open(path)?;
    run_device(device, budget, started)
}

fn run_device(
    device: AmdGpuDevice,
    budget: &MemoryBudget,
    started: Instant,
) -> Result<ProbeResult, ProbeError> {
    let ip = query_ip(&device)?;
    let information = info(&device, &ip);
    if !information.supported {
        return Err(ProbeError::Unsupported(
            "requires AMD 0x1586/family150, interface 11.0, physical compute IP 11.5.1, ring zero",
        ));
    }
    // The closed proof's guard geometry is intentionally qualified for 4-KiB pages.
    if unsafe { libc::sysconf(libc::_SC_PAGESIZE) } != PAGE as i64 {
        return Err(ProbeError::Unsupported(
            "proof requires 4096-byte host pages",
        ));
    }
    let details: uapi::DeviceInfo = device.inner.query(uapi::INFO_DEVICE)?;
    let addresses = plan_addresses(&details, &ip)?;
    let shader = probe_packets::shader_bytes();
    let commands =
        probe_packets::build_commands(addresses[0], addresses[2] + PAGE, ip.ib_size_alignment)
            .map_err(ProbeError::Unsupported)?;
    if shader.len() > PAGE as usize || commands.len() > PAGE as usize {
        return Err(ProbeError::Unsupported(
            "proof shader/commands exceed fixed allocation",
        ));
    }
    let mut session = Session::new(device);
    let mut context = abi::Context {
        input: abi::ContextIn {
            op: abi::CTX_ALLOC,
            priority: abi::CTX_PRIORITY_NORMAL,
            ..Default::default()
        },
    };
    once(
        &session.resources().device,
        abi::CTX,
        &mut context,
        "create proof context",
    )?;
    let context_id = unsafe { context.allocated.ctx_id };
    session.resources_mut().context = Some(context_id);
    for (index, size) in [PAGE, PAGE, GUARDED_BYTES as u64].into_iter().enumerate() {
        let mut buffer = session
            .resources()
            .device
            .allocate(size, Heap::Gtt, budget)?;
        match index {
            0 => buffer.write(0, &shader)?,
            1 => buffer.write(0, &commands)?,
            _ => buffer.write(0, &[CANARY; GUARDED_BYTES])?,
        }
        let mapping = Mapping {
            handle: buffer.handle,
            address: addresses[index],
            size,
        };
        session.resources_mut().buffers.push(buffer);
        let mut request = abi::GemVa {
            handle: mapping.handle,
            operation: abi::VA_MAP,
            flags: abi::VM_PAGE_READABLE | abi::VM_PAGE_WRITEABLE | abi::VM_PAGE_EXECUTABLE,
            va_address: mapping.address,
            map_size: size,
            ..Default::default()
        };
        // Conservatively quarantine on interrupted/uncertain VA changes too.
        session.uncertain = true;
        once(
            &session.resources().device,
            abi::GEM_VA,
            &mut request,
            "map proof GPU VA",
        )?;
        session.resources_mut().mappings.push(mapping);
        session.uncertain = false;
    }
    let mut entries: Vec<_> = session
        .resources()
        .buffers
        .iter()
        .map(|buffer| abi::BoListEntry {
            bo_handle: buffer.handle,
            bo_priority: 0,
        })
        .collect();
    let mut list = abi::BoList {
        input: abi::BoListIn {
            operation: abi::BO_LIST_CREATE,
            bo_number: entries.len() as u32,
            bo_info_size: size_of::<abi::BoListEntry>() as u32,
            bo_info_ptr: entries.as_mut_ptr() as usize as u64,
            ..Default::default()
        },
    };
    session.uncertain = true;
    once(
        &session.resources().device,
        abi::BO_LIST,
        &mut list,
        "create proof BO list",
    )?;
    let list_id = unsafe { list.output.list_handle };
    session.resources_mut().list = Some(list_id);
    session.uncertain = false;

    let mut ib = abi::CsChunkIb {
        va_start: addresses[1],
        ib_bytes: commands.len() as u32,
        ip_type: abi::HW_IP_COMPUTE,
        ..Default::default()
    };
    let mut chunk = abi::CsChunk {
        chunk_id: abi::CHUNK_ID_IB,
        length_dw: (size_of::<abi::CsChunkIb>() / 4) as u32,
        chunk_data: (&mut ib as *mut abi::CsChunkIb) as usize as u64,
    };
    let mut chunk_pointers = [(&mut chunk as *mut abi::CsChunk) as usize as u64];
    let mut submit = abi::Cs {
        input: abi::CsIn {
            ctx_id: context_id,
            bo_list_handle: list_id,
            num_chunks: 1,
            chunks: chunk_pointers.as_mut_ptr() as usize as u64,
            ..Default::default()
        },
    };
    let deadline = completion_deadline()?;
    fence(Ordering::SeqCst);
    session.uncertain = true; // Before CS, including panics and EINTR.
    once(
        &session.resources().device,
        abi::CS,
        &mut submit,
        "submit proof once",
    )?;
    let sequence = unsafe { submit.handle };
    if sequence == 0 || sequence == u64::MAX {
        return Err(ProbeError::Unsupported(
            "submission returned a sentinel instead of a fence sequence",
        ));
    }
    let mut wait = abi::WaitCs {
        input: abi::WaitCsIn {
            handle: sequence,
            timeout: deadline,
            ip_type: abi::HW_IP_COMPUTE,
            ctx_id: context_id,
            ..Default::default()
        },
    };
    once(
        &session.resources().device,
        abi::WAIT_CS,
        &mut wait,
        "wait for proof completion",
    )?;
    if unsafe { wait.status } != 0 {
        return Err(ProbeError::Incomplete);
    }
    let mut status = abi::Context {
        input: abi::ContextIn {
            op: abi::CTX_QUERY_STATE2,
            ctx_id: context_id,
            ..Default::default()
        },
    };
    once(
        &session.resources().device,
        abi::CTX,
        &mut status,
        "check proof context health",
    )?;
    let flags = unsafe { status.state.flags };
    if flags != 0 {
        return Err(ProbeError::ContextFault(flags));
    }
    session.uncertain = false;
    fence(Ordering::SeqCst);
    let mut bytes = [0; GUARDED_BYTES];
    session.resources_mut().buffers[2].read(0, &mut bytes)?;
    verify(&bytes)?;
    session.close()?;
    Ok(ProbeResult {
        info: information,
        sequence,
        verified_bytes: OUTPUT_BYTES,
        guard_bytes: 2 * PAGE as usize,
        allocation_bytes: TOTAL_BYTES,
        elapsed: started.elapsed(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_deadline_never_becomes_the_infinite_sentinel() {
        assert_eq!(deadline_ns(10, 123).unwrap(), 12_000_000_123);
        for (seconds, nanos) in [
            (-1, 0),
            (0, -1),
            (0, 1_000_000_000),
            (i64::MAX, 0),
            (i64::MAX / 1_000_000_000, 0),
        ] {
            assert!(deadline_ns(seconds, nanos).is_err());
        }
    }

    #[test]
    fn address_plan_uses_queried_bounds_and_refuses_overflow() {
        let mut details = uapi::DeviceInfo {
            virtual_address_offset: 0x10001,
            virtual_address_max: 0x100000,
            virtual_address_alignment: 4096,
            ..Default::default()
        };
        let ip = abi::HwIpInfo {
            ib_start_alignment: 256,
            ib_size_alignment: 32,
            ..Default::default()
        };
        assert_eq!(
            plan_addresses(&details, &ip).unwrap(),
            [0x11000, 0x12000, 0x13000]
        );
        details.virtual_address_max = 0x13000;
        assert!(plan_addresses(&details, &ip).is_err());
        details.virtual_address_offset = u64::MAX;
        assert!(plan_addresses(&details, &ip).is_err());
        assert!(checked_align(1, 3).is_err());
    }

    #[test]
    fn full_reference_detects_noop_partial_writes_and_guard_corruption() {
        let mut bytes = [CANARY; GUARDED_BYTES];
        assert!(verify(&bytes).is_err());
        bytes[PAGE as usize..PAGE as usize + OUTPUT_BYTES].fill(0x22);
        verify(&bytes).unwrap();
        for offset in [
            0,
            PAGE as usize - 1,
            PAGE as usize,
            PAGE as usize + OUTPUT_BYTES - 1,
            GUARDED_BYTES - 1,
        ] {
            let expected = bytes[offset];
            bytes[offset] ^= 1;
            assert!(
                matches!(verify(&bytes), Err(ProbeError::Mismatch { offset: observed, .. }) if observed == offset)
            );
            bytes[offset] = expected;
        }
    }
}
