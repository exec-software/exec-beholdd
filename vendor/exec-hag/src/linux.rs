// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! Linux AMDGPU discovery and bounded GEM memory ownership.
//!
//! Normal memory APIs do not submit compute commands. The optional `native-probe`
//! module runs one closed experimental shader test; `native-compute` adds a
//! trusted-shader serial queue prototype. Finding a device or passing
//! that proof does not establish that HAG can execute a model on it.

#[cfg(feature = "native-probe")]
pub mod probe;
#[cfg(feature = "native-probe")]
mod probe_packets;
#[cfg(feature = "native-probe")]
mod submission_uapi;
mod uapi;

#[cfg(feature = "native-compute")]
pub mod compute;

use crate::budget::{BudgetError, MemoryBudget, Reservation};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};
use std::os::unix::fs::{FileTypeExt, MetadataExt};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A transport failure; hardware/permission errors are never replaced by CPU fallback.
#[derive(Debug)]
pub enum Error {
    /// An operating-system operation failed.
    Io {
        /// Operation that failed.
        operation: &'static str,
        /// Original OS error, including its errno.
        source: io::Error,
    },
    /// The supplied file is not a DRM render-node character device.
    NotRenderNode,
    /// DRM reports a driver other than AMDGPU.
    WrongDriver,
    /// The requested size is zero or cannot be represented safely.
    InvalidSize,
    /// The requested byte range exceeds the buffer's logical size.
    OutOfBounds,
    /// Host access is unsupported for the selected allocation heap.
    NotHostVisible,
    /// An uncertain mapping cleanup quarantined this buffer; it cannot be reused.
    Quarantined,
    /// The caller's memory budget refused a reservation.
    Budget(BudgetError),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { operation, source } => write!(formatter, "{operation}: {source}"),
            Self::NotRenderNode => formatter.write_str("not a DRM render node"),
            Self::WrongDriver => formatter.write_str("DRM driver is not amdgpu"),
            Self::InvalidSize => {
                formatter.write_str("allocation size is zero or exceeds addressable memory")
            }
            Self::OutOfBounds => formatter.write_str("byte range exceeds buffer size"),
            Self::NotHostVisible => formatter.write_str("VRAM allocation is not host-accessible"),
            Self::Quarantined => {
                formatter.write_str("allocation is quarantined after uncertain cleanup")
            }
            Self::Budget(error) => write!(formatter, "memory budget: {error}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Budget(error) => Some(error),
            _ => None,
        }
    }
}

fn os_error(operation: &'static str, source: io::Error) -> Error {
    Error::Io { operation, source }
}

/// Values reported by AMDGPU, without inferred ISA or matrix capabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Device node opened for this device.
    pub render_node: PathBuf,
    /// PCI device ID reported by AMDGPU.
    pub device_id: u32,
    /// AMDGPU family identifier (not a guessed RDNA generation).
    pub family_id: u32,
    /// Active compute units reported by the kernel.
    pub compute_units: u32,
    /// Kernel-reported wavefront size; zero means unspecified.
    pub wavefront_size: u32,
    /// Maximum engine clock, in kHz.
    pub max_engine_clock_khz: u64,
}

/// A snapshot of one AMDGPU memory heap, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeapInfo {
    /// Physical heap size.
    pub total_bytes: u64,
    /// Kernel-reported usable heap size.
    pub usable_bytes: u64,
    /// Allocation usage across all clients at query time.
    pub used_bytes: u64,
    /// Kernel-reported maximum allocation size.
    pub max_allocation_bytes: u64,
}

impl From<uapi::HeapInfo> for HeapInfo {
    fn from(value: uapi::HeapInfo) -> Self {
        Self {
            total_bytes: value.total_heap_size,
            usable_bytes: value.usable_heap_size,
            used_bytes: value.heap_usage,
            max_allocation_bytes: value.max_allocation,
        }
    }
}

/// A current memory snapshot; UMA heap sizes must not be summed as independent RAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryInfo {
    /// Device-local memory.
    pub vram: HeapInfo,
    /// CPU-visible subset of VRAM, not additional capacity.
    pub cpu_accessible_vram: HeapInfo,
    /// GPU-accessible system memory.
    pub gtt: HeapInfo,
}

/// Allocation domain. No implicit heap or driver fallback is performed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Heap {
    /// Host-accessible GPU-visible system memory.
    Gtt,
    /// Device-local allocation; host mapping is intentionally unavailable.
    Vram,
}

/// A rejected discovery candidate and its original diagnostic.
#[derive(Debug)]
pub struct RejectedDevice {
    /// Candidate render-node path.
    pub path: PathBuf,
    /// Reason this candidate could not be used.
    pub error: Error,
}

/// Deterministic discovery results, including inaccessible and unsupported devices.
pub struct DiscoveryReport {
    /// Successfully opened AMDGPU devices, ordered by render-node number.
    pub devices: Vec<AmdGpuDevice>,
    /// Rejected candidates in the same deterministic order.
    pub rejected: Vec<RejectedDevice>,
}

// Injection stays private: callers cannot substitute mappings or mutate GEM handles.
trait Kernel: Send + Sync {
    /// Issue a state-changing request once; an uncertain submission must not replay.
    #[cfg(feature = "native-probe")]
    unsafe fn ioctl_once(
        &self,
        fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        // Test kernels have no automatic retry; the production kernel overrides this.
        unsafe { self.ioctl(fd, request, argument) }
    }
    /// The pointer must refer to the correctly sized live UAPI request for `request`.
    unsafe fn ioctl(
        &self,
        fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()>;
    /// Return a new, writable, length-byte mapping owned exclusively by the caller.
    unsafe fn map(&self, fd: RawFd, length: usize, offset: libc::off_t) -> io::Result<*mut u8>;
    /// The pointer/length must describe a live mapping returned by this implementation.
    unsafe fn unmap(&self, pointer: *mut u8, length: usize) -> io::Result<()>;
}

struct LinuxKernel;

impl Kernel for LinuxKernel {
    #[cfg(feature = "native-probe")]
    unsafe fn ioctl_once(
        &self,
        fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        // SAFETY: caller supplies a live, correctly sized UAPI request. In particular
        // CS is not retried on EINTR: it may already own resources in the kernel.
        if unsafe { libc::ioctl(fd, request as _, argument) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    unsafe fn ioctl(
        &self,
        fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        loop {
            // SAFETY: the caller supplies a correctly sized UAPI request.
            if unsafe { libc::ioctl(fd, request as _, argument) } == 0 {
                return Ok(());
            }
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
    }

    unsafe fn map(&self, fd: RawFd, length: usize, offset: libc::off_t) -> io::Result<*mut u8> {
        // SAFETY: length/offset were checked and fd owns the allocation.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                offset,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(io::Error::last_os_error());
        }
        Ok(pointer.cast())
    }

    unsafe fn unmap(&self, pointer: *mut u8, length: usize) -> io::Result<()> {
        // SAFETY: caller retains exclusive ownership and passes the original length.
        if unsafe { libc::munmap(pointer.cast(), length) } == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}

struct DeviceInner {
    fd: OwnedFd,
    kernel: Arc<dyn Kernel>,
    cleanup_failures: AtomicU64,
}

impl DeviceInner {
    fn ioctl<T>(
        &self,
        request: libc::c_ulong,
        argument: &mut T,
        operation: &'static str,
    ) -> Result<(), Error> {
        // SAFETY: all private callers pair a request constant with its UAPI structure.
        unsafe {
            self.kernel
                .ioctl(self.fd.as_raw_fd(), request, (argument as *mut T).cast())
        }
        .map_err(|error| os_error(operation, error))
    }

    fn query<T: Default>(&self, selector: u32) -> Result<T, Error> {
        let mut output = T::default();
        let mut request = uapi::Info {
            return_pointer: (&mut output as *mut T) as usize as u64,
            return_size: std::mem::size_of::<T>() as u32,
            query: selector,
            ..Default::default()
        };
        self.ioctl(uapi::INFO, &mut request, "AMDGPU INFO")?;
        Ok(output)
    }
}

/// An owned AMDGPU render-node connection, retained by every allocation.
#[derive(Clone)]
pub struct AmdGpuDevice {
    inner: Arc<DeviceInner>,
    info: DeviceInfo,
}

impl AmdGpuDevice {
    /// Open a render node read/write and verify its actual DRM driver.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(|error| os_error("open render node", error))?;
        let metadata = file
            .metadata()
            .map_err(|error| os_error("stat render node", error))?;
        let minor = libc::minor(metadata.rdev());
        if !metadata.file_type().is_char_device()
            || libc::major(metadata.rdev()) != 226
            || !(128..256).contains(&minor)
        {
            return Err(Error::NotRenderNode);
        }
        Self::from_fd(file.into(), path.to_path_buf(), Arc::new(LinuxKernel))
    }

    fn from_fd(fd: OwnedFd, path: PathBuf, kernel: Arc<dyn Kernel>) -> Result<Self, Error> {
        let inner = Arc::new(DeviceInner {
            fd,
            kernel,
            cleanup_failures: AtomicU64::new(0),
        });
        let mut name = [0u8; 64];
        let mut version = uapi::Version {
            name_len: name.len(),
            name: name.as_mut_ptr().cast(),
            ..Default::default()
        };
        inner.ioctl(uapi::VERSION, &mut version, "DRM VERSION")?;
        if version.name_len != 6 || &name[..6] != b"amdgpu" {
            return Err(Error::WrongDriver);
        }
        let details: uapi::DeviceInfo = inner.query(uapi::INFO_DEVICE)?;
        Ok(Self {
            inner,
            info: DeviceInfo {
                render_node: path,
                device_id: details.device_id,
                family_id: details.family,
                compute_units: details.cu_active_number,
                wavefront_size: details.wave_front_size,
                max_engine_clock_khz: details.max_engine_clock,
            },
        })
    }

    /// Discover AMDGPU render nodes; use `discover_report` for rejected-node diagnostics.
    pub fn discover() -> Result<Vec<Self>, Error> {
        Ok(Self::discover_report()?.devices)
    }

    /// Discover sorted render nodes, retaining permission and unsupported-driver failures.
    /// A machine without `/dev/dri` returns an empty report.
    pub fn discover_report() -> Result<DiscoveryReport, Error> {
        let directory = match fs::read_dir("/dev/dri") {
            Ok(directory) => directory,
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Ok(DiscoveryReport {
                    devices: Vec::new(),
                    rejected: Vec::new(),
                });
            }
            Err(error) => return Err(os_error("list DRM render nodes", error)),
        };
        let mut paths = Vec::new();
        for entry in directory {
            let entry = entry.map_err(|error| os_error("read DRM directory entry", error))?;
            if let Some(index) = render_index(&entry.file_name().to_string_lossy()) {
                paths.push((index, entry.path()));
            }
        }
        paths.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
        Ok(discover_paths(
            paths.into_iter().map(|(_, path)| path),
            Self::open,
        ))
    }

    /// Kernel-reported device information captured when this connection was opened.
    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Number of uncertain cleanups quarantined on this shared connection.
    /// Use `Buffer::close` to obtain the original error; `Drop` cannot return one.
    pub fn cleanup_failure_count(&self) -> u64 {
        self.inner.cleanup_failures.load(Ordering::Relaxed)
    }

    /// Query a fresh memory snapshot. This is not a reservation or allocation guarantee.
    pub fn memory_info(&self) -> Result<MemoryInfo, Error> {
        let details: uapi::MemoryInfo = self.inner.query(uapi::INFO_MEMORY)?;
        Ok(MemoryInfo {
            vram: details.vram.into(),
            cpu_accessible_vram: details.cpu_accessible_vram.into(),
            gtt: details.gtt.into(),
        })
    }

    /// Reserve page-rounded bytes, then allocate in exactly the requested heap.
    /// Allocation failures release the reservation. Buffers keep this device alive.
    pub fn allocate(&self, size: u64, heap: Heap, budget: &MemoryBudget) -> Result<Buffer, Error> {
        // SAFETY: sysconf has no pointer arguments or external mutation.
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if page_size <= 0 {
            return Err(os_error(
                "query page size",
                io::Error::other("invalid page size"),
            ));
        }
        self.allocate_with_page_size(size, heap, budget, page_size as u64)
    }

    fn allocate_with_page_size(
        &self,
        size: u64,
        heap: Heap,
        budget: &MemoryBudget,
        page_size: u64,
    ) -> Result<Buffer, Error> {
        let allocated = round_allocation(size, page_size)?;
        let reservation = budget.reserve(allocated).map_err(Error::Budget)?;
        let mut request = uapi::GemCreate {
            input: uapi::GemCreateIn {
                bo_size: allocated,
                alignment: page_size,
                domains: match heap {
                    Heap::Gtt => uapi::DOMAIN_GTT,
                    Heap::Vram => uapi::DOMAIN_VRAM,
                },
                domain_flags: match heap {
                    Heap::Gtt => uapi::CPU_ACCESS_REQUIRED,
                    Heap::Vram => uapi::NO_CPU_ACCESS,
                },
            },
        };
        self.inner
            .ioctl(uapi::GEM_CREATE, &mut request, "AMDGPU GEM CREATE")?;
        // SAFETY: successful GEM_CREATE initializes the output member.
        let handle = unsafe { request.output.handle };
        Ok(Buffer {
            device: self.inner.clone(),
            handle,
            size,
            allocated,
            heap,
            mapping: None,
            reservation: Some(reservation),
        })
    }
}

fn render_index(name: &str) -> Option<u32> {
    let suffix = name.strip_prefix("renderD")?;
    if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

fn discover_paths(
    paths: impl Iterator<Item = PathBuf>,
    mut open: impl FnMut(PathBuf) -> Result<AmdGpuDevice, Error>,
) -> DiscoveryReport {
    let mut report = DiscoveryReport {
        devices: Vec::new(),
        rejected: Vec::new(),
    };
    for path in paths {
        match open(path.clone()) {
            Ok(device) => report.devices.push(device),
            Err(error) => report.rejected.push(RejectedDevice { path, error }),
        }
    }
    report
}

fn round_allocation(size: u64, page_size: u64) -> Result<u64, Error> {
    if size == 0 || page_size == 0 {
        return Err(Error::InvalidSize);
    }
    let rounded = size.checked_add(page_size - 1).ok_or(Error::InvalidSize)? / page_size;
    let rounded = rounded.checked_mul(page_size).ok_or(Error::InvalidSize)?;
    if rounded > isize::MAX as u64 {
        return Err(Error::InvalidSize);
    }
    Ok(rounded)
}

/// An exclusively owned GEM allocation with bounded, copying host access.
/// No raw handle or borrowed mapping can escape and outlive this allocation.
/// GTT storage is zero-initialized on first host access. On uncertain cleanup,
/// the allocation owner and budget charge are intentionally retained until process
/// exit; quota is never returned while storage might still be live. Prefer `close`
/// when cleanup diagnostics are required; `Drop` records only the failure count.
pub struct Buffer {
    device: Arc<DeviceInner>,
    handle: u32,
    size: u64,
    allocated: u64,
    heap: Heap,
    mapping: Option<NonNull<u8>>,
    reservation: Option<Reservation>,
}

// SAFETY: the private mapping has no escaping references; every host access requires
// exclusive &mut self. No API submits GPU work or exports the allocation to another client.
unsafe impl Send for Buffer {}

impl Buffer {
    /// Requested logical size, excluding inaccessible allocation padding.
    pub fn size_bytes(&self) -> u64 {
        self.size
    }

    /// Page-rounded bytes reserved against the caller's budget.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated
    }

    /// Heap selected at allocation time.
    pub fn heap(&self) -> Heap {
        self.heap
    }

    /// Release this allocation and report OS cleanup failures. A failure quarantines
    /// its device owner and memory charge until process exit, preventing quota reuse.
    pub fn close(mut self) -> Result<(), Error> {
        if self.reservation.is_none() {
            return Err(Error::Quarantined);
        }
        self.cleanup()
    }

    /// Copy bytes into a GTT allocation; bounds failures do not create a mapping.
    pub fn write(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        self.validate_range(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let pointer = self.map()?;
        // SAFETY: checked range fits the live mapping, which cannot alias caller data.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                pointer.as_ptr().add(offset as usize),
                bytes.len(),
            );
        }
        Ok(())
    }

    /// Copy bytes from a GTT allocation without exposing mapping lifetime or aliases.
    pub fn read(&mut self, offset: u64, bytes: &mut [u8]) -> Result<(), Error> {
        self.validate_range(offset, bytes.len())?;
        if bytes.is_empty() {
            return Ok(());
        }
        let pointer = self.map()?;
        // SAFETY: checked range fits the live mapping and destination cannot alias it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                pointer.as_ptr().add(offset as usize),
                bytes.as_mut_ptr(),
                bytes.len(),
            );
        }
        Ok(())
    }

    fn validate_range(&self, offset: u64, length: usize) -> Result<(), Error> {
        if self.reservation.is_none() {
            return Err(Error::Quarantined);
        }
        if self.heap != Heap::Gtt {
            return Err(Error::NotHostVisible);
        }
        let end = offset
            .checked_add(length as u64)
            .ok_or(Error::OutOfBounds)?;
        if end > self.size {
            return Err(Error::OutOfBounds);
        }
        Ok(())
    }

    fn map(&mut self) -> Result<NonNull<u8>, Error> {
        if let Some(pointer) = self.mapping {
            return Ok(pointer);
        }
        let mut request = uapi::GemMmap {
            input: uapi::GemMmapIn {
                handle: self.handle,
                pad: 0,
            },
        };
        self.device
            .ioctl(uapi::GEM_MMAP, &mut request, "AMDGPU GEM MMAP")?;
        // SAFETY: successful GEM_MMAP initializes offset. This is not a host pointer.
        let offset =
            libc::off_t::try_from(unsafe { request.offset }).map_err(|_| Error::InvalidSize)?;
        // SAFETY: allocation owns fd/handle; mapping size and signed offset are checked.
        let pointer = unsafe {
            self.device
                .kernel
                .map(self.device.fd.as_raw_fd(), self.allocated as usize, offset)
        }
        .map_err(|error| os_error("mmap GEM allocation", error))?;
        let Some(pointer) = NonNull::new(pointer) else {
            // mmap may legally choose address zero, but Rust copies cannot use it.
            // Preserve ownership and quota if discarding even that mapping fails.
            if let Err(error) =
                unsafe { self.device.kernel.unmap(pointer, self.allocated as usize) }
            {
                if let Some(reservation) = self.reservation.take() {
                    self.quarantine(reservation);
                }
                return Err(os_error("munmap null GEM allocation", error));
            }
            return Err(os_error(
                "mmap GEM allocation",
                io::Error::other("mmap returned the null address"),
            ));
        };
        // SAFETY: this is an exclusive writable allocation, not yet exposed to users or
        // submitted to a GPU. Do not depend on driver-specific initial memory contents.
        unsafe {
            std::ptr::write_bytes(pointer.as_ptr(), 0, self.allocated as usize);
        }
        self.mapping = Some(pointer);
        Ok(pointer)
    }

    fn cleanup(&mut self) -> Result<(), Error> {
        let Some(reservation) = self.reservation.take() else {
            return Ok(());
        };
        if let Some(pointer) = self.mapping.take() {
            // SAFETY: mapping belongs exclusively to this buffer and length is unchanged.
            if let Err(error) = unsafe {
                self.device
                    .kernel
                    .unmap(pointer.as_ptr(), self.allocated as usize)
            } {
                self.quarantine(reservation);
                return Err(os_error("munmap GEM allocation", error));
            }
        }
        let mut close = uapi::GemClose {
            handle: self.handle,
            pad: 0,
        };
        if let Err(error) = self.device.ioctl(uapi::GEM_CLOSE, &mut close, "GEM CLOSE") {
            self.quarantine(reservation);
            return Err(error);
        }
        drop(reservation);
        Ok(())
    }

    fn quarantine(&self, reservation: Reservation) {
        self.device.cleanup_failures.fetch_add(1, Ordering::Relaxed);
        // Intentional fail-closed quarantine. Keeping this Arc prevents fd reuse and
        // keeps GEM ownership valid; retaining the charge prevents further admission.
        // Neither a failed unmap nor a failed GEM close proves physical reclamation.
        std::mem::forget((self.device.clone(), reservation));
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = self.cleanup();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Mutex;

    #[derive(Default)]
    struct Calls {
        creates: Vec<(u64, u64, u64, u64)>,
        maps: usize,
        mmap_queries: usize,
        releases: Vec<(&'static str, u64)>,
    }

    struct MockKernel {
        calls: Mutex<Calls>,
        budget: MemoryBudget,
        fail_create: AtomicBool,
        fail_map: AtomicBool,
        fail_close: AtomicBool,
        fail_unmap: AtomicBool,
        null_mapping: AtomicBool,
        wrong_driver: bool,
    }

    impl MockKernel {
        fn new(budget: MemoryBudget) -> Self {
            Self {
                calls: Mutex::new(Calls::default()),
                budget,
                fail_create: AtomicBool::new(false),
                fail_map: AtomicBool::new(false),
                fail_close: AtomicBool::new(false),
                fail_unmap: AtomicBool::new(false),
                null_mapping: AtomicBool::new(false),
                wrong_driver: false,
            }
        }
    }

    impl Kernel for MockKernel {
        unsafe fn ioctl(
            &self,
            fd: RawFd,
            request: libc::c_ulong,
            argument: *mut libc::c_void,
        ) -> io::Result<()> {
            // All tests use a private /dev/null fd, never a DRM node.
            assert!(
                unsafe { libc::fcntl(fd, libc::F_GETFD) } >= 0,
                "device fd must remain open"
            );
            match request {
                uapi::VERSION => {
                    let version = unsafe { &mut *argument.cast::<uapi::Version>() };
                    let name: &[u8] = if self.wrong_driver {
                        b"nouveau"
                    } else {
                        b"amdgpu"
                    };
                    assert!(version.name_len >= name.len());
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            name.as_ptr(),
                            version.name.cast(),
                            name.len(),
                        );
                    }
                    version.name_len = name.len();
                }
                uapi::INFO => {
                    let info = unsafe { &mut *argument.cast::<uapi::Info>() };
                    assert_eq!(info.query_data, [0; 4]);
                    match info.query {
                        uapi::INFO_DEVICE => {
                            assert_eq!(
                                info.return_size as usize,
                                std::mem::size_of::<uapi::DeviceInfo>()
                            );
                            let output = unsafe {
                                &mut *(info.return_pointer as usize as *mut uapi::DeviceInfo)
                            };
                            output.device_id = 0x1506;
                            output.family = 150;
                            output.cu_active_number = 40;
                            output.wave_front_size = 0; // Unknown remains unknown.
                            output.max_engine_clock = 3_000_000;
                        }
                        uapi::INFO_MEMORY => {
                            assert_eq!(
                                info.return_size as usize,
                                std::mem::size_of::<uapi::MemoryInfo>()
                            );
                            let output = unsafe {
                                &mut *(info.return_pointer as usize as *mut uapi::MemoryInfo)
                            };
                            output.vram.total_heap_size = 100;
                            output.cpu_accessible_vram.total_heap_size = 50;
                            output.gtt.total_heap_size = 200;
                            output.gtt.usable_heap_size = 190;
                            output.gtt.heap_usage = 20;
                            output.gtt.max_allocation = 170;
                        }
                        selector => panic!("unexpected INFO selector {selector}"),
                    }
                }
                uapi::GEM_CREATE => {
                    let create = unsafe { &mut *argument.cast::<uapi::GemCreate>() };
                    let input = unsafe { create.input };
                    self.calls.lock().unwrap().creates.push((
                        input.bo_size,
                        input.alignment,
                        input.domains,
                        input.domain_flags,
                    ));
                    assert!(
                        self.budget.current_bytes() >= input.bo_size,
                        "reserve before ioctl"
                    );
                    if self.fail_create.load(Ordering::Relaxed) {
                        return Err(io::Error::from_raw_os_error(libc::ENOMEM));
                    }
                    create.output = uapi::GemCreateOut { handle: 7, pad: 0 };
                }
                uapi::GEM_MMAP => {
                    let mapping = unsafe { &mut *argument.cast::<uapi::GemMmap>() };
                    assert_eq!(unsafe { mapping.input.handle }, 7);
                    self.calls.lock().unwrap().mmap_queries += 1;
                    mapping.offset = 0;
                }
                uapi::GEM_CLOSE => {
                    let close = unsafe { &*argument.cast::<uapi::GemClose>() };
                    assert_eq!(close.handle, 7);
                    self.calls
                        .lock()
                        .unwrap()
                        .releases
                        .push(("close", self.budget.current_bytes()));
                    if self.fail_close.load(Ordering::Relaxed) {
                        return Err(io::Error::from_raw_os_error(libc::EIO));
                    }
                }
                _ => panic!("unexpected ioctl {request}"),
            }
            Ok(())
        }

        unsafe fn map(
            &self,
            _fd: RawFd,
            length: usize,
            offset: libc::off_t,
        ) -> io::Result<*mut u8> {
            assert_eq!(offset, 0);
            self.calls.lock().unwrap().maps += 1;
            if self.fail_map.load(Ordering::Relaxed) {
                return Err(io::Error::from_raw_os_error(libc::EACCES));
            }
            if self.null_mapping.load(Ordering::Relaxed) {
                return Ok(std::ptr::null_mut());
            }
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            assert_ne!(pointer, libc::MAP_FAILED);
            // Verify library initialization instead of relying on anonymous-map zeroes.
            unsafe {
                std::ptr::write_bytes(pointer.cast::<u8>(), 0xa5, length);
            }
            Ok(pointer.cast())
        }

        unsafe fn unmap(&self, pointer: *mut u8, length: usize) -> io::Result<()> {
            self.calls
                .lock()
                .unwrap()
                .releases
                .push(("unmap", self.budget.current_bytes()));
            if self.fail_unmap.load(Ordering::Relaxed) {
                return Err(io::Error::from_raw_os_error(libc::EIO));
            }
            if !pointer.is_null() {
                assert_eq!(unsafe { libc::munmap(pointer.cast(), length) }, 0);
            }
            Ok(())
        }
    }

    fn device(kernel: Arc<MockKernel>) -> AmdGpuDevice {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/null")
            .unwrap();
        AmdGpuDevice::from_fd(file.into(), PathBuf::from("/mock/renderD128"), kernel).unwrap()
    }

    #[test]
    fn queries_preserve_kernel_values_and_separate_heap_layouts() {
        let kernel = Arc::new(MockKernel::new(MemoryBudget::new(0)));
        let device = device(kernel);
        assert_eq!(device.info().compute_units, 40);
        assert_eq!(device.info().wavefront_size, 0);
        assert_eq!(device.info().max_engine_clock_khz, 3_000_000);
        let memory = device.memory_info().unwrap();
        assert_eq!(memory.vram.total_bytes, 100);
        assert_eq!(memory.cpu_accessible_vram.total_bytes, 50);
        assert_eq!(
            memory.gtt,
            HeapInfo {
                total_bytes: 200,
                usable_bytes: 190,
                used_bytes: 20,
                max_allocation_bytes: 170
            }
        );
    }

    #[test]
    fn rejects_non_render_node_without_issuing_gpu_operations() {
        assert!(matches!(
            AmdGpuDevice::open("/dev/null"),
            Err(Error::NotRenderNode)
        ));
    }

    #[test]
    fn rejects_other_drm_drivers() {
        let mut kernel = MockKernel::new(MemoryBudget::new(0));
        kernel.wrong_driver = true;
        let file = std::fs::File::open("/dev/null").unwrap();
        assert!(matches!(
            AmdGpuDevice::from_fd(file.into(), PathBuf::new(), Arc::new(kernel)),
            Err(Error::WrongDriver)
        ));
    }

    #[test]
    fn buffer_retains_device_and_releases_mapping_handle_then_budget() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let device = device(kernel.clone());
        let weak_device = Arc::downgrade(&device.inner);
        let mut buffer = device
            .allocate_with_page_size(13, Heap::Gtt, &budget, 4096)
            .unwrap();
        assert_eq!(buffer.size_bytes(), 13);
        assert_eq!(buffer.allocated_bytes(), 4096);
        drop(device);
        assert!(weak_device.upgrade().is_some());
        buffer.write(2, b"hello").unwrap();
        let mut output = [0; 5];
        buffer.read(2, &mut output).unwrap();
        assert_eq!(&output, b"hello");
        assert_eq!(kernel.calls.lock().unwrap().maps, 1);
        drop(buffer);
        assert!(weak_device.upgrade().is_none());
        assert_eq!(budget.current_bytes(), 0);
        let calls = kernel.calls.lock().unwrap();
        assert_eq!(
            calls.creates,
            [(4096, 4096, uapi::DOMAIN_GTT, uapi::CPU_ACCESS_REQUIRED)]
        );
        assert_eq!(calls.releases, [("unmap", 4096), ("close", 4096)]);
    }

    #[test]
    fn bounds_are_logical_checked_and_do_not_map() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let mut buffer = device(kernel.clone())
            .allocate_with_page_size(3, Heap::Gtt, &budget, 4096)
            .unwrap();
        assert!(matches!(buffer.write(3, &[1]), Err(Error::OutOfBounds)));
        assert!(matches!(
            buffer.write(u64::MAX, &[1]),
            Err(Error::OutOfBounds)
        ));
        assert!(matches!(
            buffer.read(2, &mut [0; 2]),
            Err(Error::OutOfBounds)
        ));
        buffer.write(3, &[]).unwrap();
        buffer.read(3, &mut []).unwrap();
        assert_eq!(kernel.calls.lock().unwrap().maps, 0);
    }

    #[test]
    fn exact_vram_domain_has_no_mapping_or_heap_fallback() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let mut buffer = device(kernel.clone())
            .allocate_with_page_size(1, Heap::Vram, &budget, 4096)
            .unwrap();
        assert_eq!(buffer.heap(), Heap::Vram);
        assert!(matches!(buffer.write(0, &[1]), Err(Error::NotHostVisible)));
        assert_eq!(
            kernel.calls.lock().unwrap().creates,
            [(4096, 4096, uapi::DOMAIN_VRAM, uapi::NO_CPU_ACCESS)]
        );
        assert_eq!(kernel.calls.lock().unwrap().maps, 0);
    }

    #[test]
    fn allocation_validation_precedes_budget_and_ioctl() {
        let budget = MemoryBudget::new(4095);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let device = device(kernel.clone());
        for size in [0, u64::MAX, isize::MAX as u64] {
            assert!(matches!(
                device.allocate_with_page_size(size, Heap::Gtt, &budget, 4096),
                Err(Error::InvalidSize)
            ));
        }
        assert!(matches!(
            device.allocate_with_page_size(1, Heap::Gtt, &budget, 4096),
            Err(Error::Budget(_))
        ));
        assert!(kernel.calls.lock().unwrap().creates.is_empty());
        assert_eq!(budget.current_bytes(), 0);
    }

    #[test]
    fn create_failure_rolls_back_reservation_and_preserves_errno() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        kernel.fail_create.store(true, Ordering::Relaxed);
        let result = device(kernel.clone()).allocate_with_page_size(1, Heap::Gtt, &budget, 4096);
        assert!(
            matches!(result, Err(Error::Io { source, .. }) if source.raw_os_error() == Some(libc::ENOMEM))
        );
        assert_eq!(budget.current_bytes(), 0);
        assert!(kernel.calls.lock().unwrap().releases.is_empty());
    }

    #[test]
    fn failed_mapping_can_retry_and_does_not_invalidate_allocation() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let mut buffer = device(kernel.clone())
            .allocate_with_page_size(1, Heap::Gtt, &budget, 4096)
            .unwrap();
        kernel.fail_map.store(true, Ordering::Relaxed);
        assert!(
            matches!(buffer.write(0, &[7]), Err(Error::Io { source, .. }) if source.raw_os_error() == Some(libc::EACCES))
        );
        assert_eq!(budget.current_bytes(), 4096);
        kernel.fail_map.store(false, Ordering::Relaxed);
        buffer.write(0, &[7]).unwrap();
        drop(buffer);
        assert_eq!(budget.current_bytes(), 0);
        assert_eq!(
            kernel.calls.lock().unwrap().releases,
            [("unmap", 4096), ("close", 4096)]
        );
    }

    #[test]
    fn discovery_keeps_rejections_and_continues_to_usable_device() {
        let paths = ["/mock/renderD128", "/mock/renderD129", "/mock/renderD130"].map(PathBuf::from);
        let report = discover_paths(paths.into_iter(), |path| {
            if path.ends_with("renderD128") {
                return Err(os_error(
                    "open render node",
                    io::Error::from_raw_os_error(libc::EACCES),
                ));
            }
            if path.ends_with("renderD129") {
                return Err(Error::WrongDriver);
            }
            Ok(device(Arc::new(MockKernel::new(MemoryBudget::new(0)))))
        });
        assert_eq!(report.devices.len(), 1);
        assert_eq!(report.rejected.len(), 2);
        assert!(
            matches!(&report.rejected[0].error, Error::Io { source, .. } if source.kind() == io::ErrorKind::PermissionDenied)
        );
        assert!(matches!(report.rejected[1].error, Error::WrongDriver));
        for name in ["card0", "renderD", "renderD-1", "renderD128evil"] {
            assert_eq!(render_index(name), None);
        }
        assert_eq!(render_index("renderD129"), Some(129));
    }

    #[test]
    fn newly_mapped_storage_reads_as_zero_without_relying_on_driver_contents() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let mut buffer = device(kernel)
            .allocate_with_page_size(3, Heap::Gtt, &budget, 4096)
            .unwrap();
        let mut output = [0xff; 3];
        buffer.read(0, &mut output).unwrap();
        assert_eq!(output, [0; 3]);
        buffer.close().unwrap();
        assert_eq!(budget.current_bytes(), 0);
    }

    #[test]
    fn close_failure_quarantines_device_and_charge_without_double_close() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let device = device(kernel.clone());
        let weak = Arc::downgrade(&device.inner);
        let mut buffer = device
            .allocate_with_page_size(3, Heap::Gtt, &budget, 4096)
            .unwrap();
        buffer.write(0, &[1]).unwrap();
        kernel.fail_close.store(true, Ordering::Relaxed);
        assert!(
            matches!(buffer.close(), Err(Error::Io { operation: "GEM CLOSE", source }) if source.raw_os_error() == Some(libc::EIO))
        );
        assert_eq!(device.cleanup_failure_count(), 1);
        drop(device);
        assert!(weak.upgrade().is_some(), "quarantine retains the owning fd");
        assert_eq!(budget.current_bytes(), 4096);
        assert!(budget.reserve(1).is_err());
        assert_eq!(
            kernel.calls.lock().unwrap().releases,
            [("unmap", 4096), ("close", 4096)]
        );
    }

    #[test]
    fn failed_unmap_keeps_mapping_owner_and_budget_even_after_drop() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let device = device(kernel.clone());
        let weak = Arc::downgrade(&device.inner);
        let mut buffer = device
            .allocate_with_page_size(3, Heap::Gtt, &budget, 4096)
            .unwrap();
        buffer.write(0, &[1]).unwrap();
        kernel.fail_unmap.store(true, Ordering::Relaxed);
        drop(buffer);
        assert_eq!(device.cleanup_failure_count(), 1);
        drop(device);
        assert!(weak.upgrade().is_some());
        assert_eq!(budget.current_bytes(), 4096);
        assert!(budget.reserve(1).is_err());
        // A failed unmap still owns storage; neither close the GEM handle nor credit quota.
        assert_eq!(kernel.calls.lock().unwrap().releases, [("unmap", 4096)]);
    }

    #[test]
    fn null_mapping_failure_quarantines_and_cannot_be_reused() {
        let budget = MemoryBudget::new(4096);
        let kernel = Arc::new(MockKernel::new(budget.clone()));
        let device = device(kernel.clone());
        let mut buffer = device
            .allocate_with_page_size(1, Heap::Gtt, &budget, 4096)
            .unwrap();
        kernel.null_mapping.store(true, Ordering::Relaxed);
        kernel.fail_unmap.store(true, Ordering::Relaxed);
        assert!(matches!(
            buffer.write(0, &[1]),
            Err(Error::Io {
                operation: "munmap null GEM allocation",
                ..
            })
        ));
        assert!(matches!(buffer.write(0, &[1]), Err(Error::Quarantined)));
        assert_eq!(device.cleanup_failure_count(), 1);
        assert!(matches!(buffer.close(), Err(Error::Quarantined)));
        assert_eq!(budget.current_bytes(), 4096);
        assert_eq!(kernel.calls.lock().unwrap().releases, [("unmap", 4096)]);
    }
}
