// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! Experimental serial GFX1151 queues for trusted, bounded shader programs.
//!
//! This is a hardware transport, not an inference backend. One queue owns one
//! immutable shader allocation, one reusable IB, three guarded data slots and a dedicated
//! context/connection. Host access and synchronous dispatch require exclusive
//! ownership. No resource address, mapping or reference escapes the queue.
//! There is no asynchronous submission or cross-queue resource sharing yet.

use super::probe::{self, Mapping, ProbeError, Session};
use super::{submission_uapi as abi, uapi, AmdGpuDevice, Error, Heap};
use crate::budget::MemoryBudget;
use std::mem::size_of;
use std::path::Path;
use std::sync::atomic::{fence, Ordering};

mod packets;
#[cfg(test)]
mod tests;

const PAGE: u64 = 4096;
const CANARY: u8 = 0xa5;
const MAX_PROGRAMS: usize = 32;
const PROGRAM_ALIGNMENT: usize = 256;

/// Maximum page-rounded storage for an immutable compute library: 32 KiB.
/// Individual programs remain limited to one 4-KiB page, including fetch padding.
pub const MAX_SHADER_BYTES: u64 = 32 * 1024;

/// Maximum logical capacity of each guarded queue data slot: 4 GiB minus 4 KiB.
///
/// Page rounding, guards, shader storage and command storage are additional
/// charges against the caller's shared [`MemoryBudget`]. Each allocation must
/// be admitted before its GEM request; this cap is not a memory reservation or
/// a claim that every workload fitting it has been hardware-qualified.
///
/// The largest page-aligned payload representable by a 32-bit raw descriptor.
/// Guarded allocation sizes and GPU addresses are calculated as 64-bit values.
pub const MAX_SLOT_BYTES: u32 = u32::MAX - (PAGE as u32 - 1);

/// A descriptor-bounded byte region within one queue-owned data slot.
///
/// Regions expose no address or mapped alias. Offsets are relative to the logical
/// payload, excluding guards. The trusted shader supplies alignment requirements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferRegion {
    /// Queue-local data slot, zero through two.
    pub slot: usize,
    /// Byte offset inside the slot's logical payload.
    pub offset: u64,
    /// Nonzero byte length available to this shader descriptor.
    pub length: u32,
}

/// Fixed hardware ABI of this experimental queue's command encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelAbi {
    /// COMPUTE_PGM_RSRC1/2/3, including register allocation and float mode.
    pub program_resources: [u32; 3],
    /// Raw byte-addressed buffer descriptor's fourth word.
    pub raw_descriptor: u32,
    /// Fixed workgroup dimensions (wave64).
    pub threads: [u32; 3],
    /// Workgroups one dispatch under this ABI may launch.
    ///
    /// Unlike the fields above this encodes no hardware register; it bounds
    /// how much work a single submission starts. It lives here because it is
    /// the same kind of fact as the register budget -- a property of the
    /// library whose programs are being dispatched, chosen once and applying
    /// to all of them.
    pub max_groups: u32,
}

/// Shader providers must match this ABI exactly before uploading trusted code.
pub const GFX1151_SERIAL_ABI: KernelAbi = KernelAbi {
    program_resources: [0x408f_0001, 0x9c, 0],
    raw_descriptor: 0x3001_6fac,
    threads: [64, 1, 1],
    max_groups: DEFAULT_DISPATCH_GROUPS,
};

/// Workgroups one dispatch may launch under the qualified ABI.
///
/// Sized for the AI kernels, whose dispatches cover at most 4096 elements.
pub const DEFAULT_DISPATCH_GROUPS: u32 = 64;

/// Highest workgroup count this encoder admits for a derived ABI.
///
/// The cap is a policy bound, not a hardware one: DISPATCH_DIRECT takes a
/// 32-bit count. It exists because a dispatch that launches more work runs
/// for longer, and this queue is synchronous. The real backstop against a
/// runaway kernel is the fence deadline, which reports Incomplete; this
/// simply keeps the launch inside a range that has been reasoned about.
///
/// 65536 groups is 4,194,304 lanes -- a 1920x1080 image in one dispatch, with
/// room to spare. It is a deliberate step rather than an opening of the gate.
pub const MAX_DISPATCH_GROUPS: u32 = 65536;

/// Lowest VGPR allocation this encoder admits, matching [`GFX1151_SERIAL_ABI`].
pub const MIN_KERNEL_VGPRS: u32 = 8;
/// Highest VGPR allocation this encoder admits.
///
/// The ceiling is deliberate rather than hardware-imposed. This queue is
/// serial and synchronous, so a larger allocation costs occupancy that is not
/// being used; the limit exists to keep the register field inside a range that
/// has been reasoned about, not to approach the hardware maximum.
pub const MAX_KERNEL_VGPRS: u32 = 64;

impl KernelAbi {
    /// Derive this ABI with a different per-wave VGPR allocation.
    ///
    /// **Only the granulated VGPR count varies.** Every other bit of
    /// COMPUTE_PGM_RSRC1/2/3 -- float round and denormal modes, IEEE mode,
    /// privilege, workgroup-ID enables, user SGPR count -- is preserved
    /// exactly. These are privileged hardware registers, and a caller must not
    /// be able to set arbitrary bits in them through this interface; a kernel
    /// that wants more registers is a different claim from one that wants a
    /// different float mode or privilege level.
    ///
    /// `vgprs` must be a multiple of four within
    /// [`MIN_KERNEL_VGPRS`]..=[`MAX_KERNEL_VGPRS`], because wave64 allocates
    /// VGPRs in granules of four. Returns `None` otherwise.
    ///
    /// A library's programs share one encoder ABI, so this is chosen per
    /// library. Deriving a 16-VGPR ABI for a new library leaves an existing
    /// library's programs, and their qualification, completely untouched.
    pub const fn with_vgprs(self, vgprs: u32) -> Option<Self> {
        if vgprs % 4 != 0 || vgprs < MIN_KERNEL_VGPRS || vgprs > MAX_KERNEL_VGPRS {
            return None;
        }
        let granulated = vgprs / 4 - 1;
        let mut derived = self;
        derived.program_resources[0] = (self.program_resources[0] & !0x3f) | granulated;
        Some(derived)
    }

    /// The per-wave VGPR allocation this ABI encodes.
    pub const fn vgprs(&self) -> u32 {
        ((self.program_resources[0] & 0x3f) + 1) * 4
    }

    /// Derive this ABI with a different maximum workgroup count.
    ///
    /// A library chooses this because a library knows what its kernels cost
    /// per lane. The qualified default of 64 is right for a matvec over 4096
    /// elements and wrong for an image: at 64 groups a 1920x1080 frame needs
    /// 507 dispatches, and since a dispatch costs about the same whether it
    /// covers 64 pixels or 4000 -- the cost is the submit and the fence, not
    /// the shading -- those 507 round trips are the frame's entire budget.
    ///
    /// Returns `None` outside 1..=[`MAX_DISPATCH_GROUPS`].
    pub const fn with_max_groups(self, groups: u32) -> Option<Self> {
        if groups == 0 || groups > MAX_DISPATCH_GROUPS {
            return None;
        }
        let mut derived = self;
        derived.max_groups = groups;
        Some(derived)
    }

    /// The highest workgroup count a dispatch under this ABI may launch.
    pub const fn max_groups(&self) -> u32 {
        self.max_groups
    }
}

/// An owned reusable queue with a fixed three-buffer, two-scalar shader ABI.
///
/// Shader and allocations stay resident between synchronous submissions. An
/// interrupted/failed submission or a panic permanently poisons this instance;
/// all subsequent access refuses and Drop quarantines the complete ownership
/// chain until process exit. This is not an automatic-recovery mechanism.
pub struct ComputeQueue {
    session: Session,
    sizes: [u32; 3],
    addresses: [u64; 5],
    alignment: u32,
    allocated_bytes: u64,
    context: u32,
    list: u32,
    trace_submissions: bool,
    program_offsets: Vec<u32>,
    /// Encoder ABI for this library's programs. Chosen once at open time;
    /// every program in the library shares it.
    abi: KernelAbi,
}

fn invalid(reason: &'static str) -> ProbeError {
    ProbeError::Unsupported(reason)
}

impl ComputeQueue {
    /// Create resident storage and a normal-priority queue on the allowlisted GPU.
    ///
    /// Each data slot has the given logical capacity (1 byte to [`MAX_SLOT_BYTES`]),
    /// rounded to pages with an additional 4-KiB guard at either end. All storage,
    /// including the 4-KiB shader and IB, is charged to the shared budget.
    ///
    /// # Safety
    /// `kernel` must be independently audited GFX1151 wave64 ISA, including its
    /// required instruction-end padding, fitting one 4-KiB allocation. It must
    /// use at most eight VGPRs and this exact ABI: `s[0:3]`, `s[4:7]`, `s[8:11]` are
    /// byte-addressed buffer descriptors, s12/s13 are scalars, s14 is group X,
    /// and v0 is packed local ID. It must access only descriptor-bounded data,
    /// never modify shader/IB/guards, and terminate for every submitted argument.
    /// This API does not validate arbitrary ISA. Never use untrusted model or
    /// user bytes as shader code. Driver faults can affect other GPU clients.
    pub unsafe fn open(
        path: impl AsRef<Path>,
        budget: &MemoryBudget,
        kernel: &[u8],
        sizes: [u32; 3],
    ) -> Result<Self, ProbeError> {
        let device = AmdGpuDevice::open(path)?;
        // SAFETY: the caller supplies the trusted program under the same contract.
        unsafe { Self::from_device(device, budget, kernel, sizes) }
    }

    unsafe fn from_device(
        device: AmdGpuDevice,
        budget: &MemoryBudget,
        kernel: &[u8],
        sizes: [u32; 3],
    ) -> Result<Self, ProbeError> {
        // SAFETY: same caller-supplied program contract as the multi-entry path.
        unsafe { Self::from_programs(device, budget, &[kernel], sizes) }
    }

    /// Create a queue with a bounded immutable library of one through thirty-two programs.
    ///
    /// Entry points are 256-byte aligned within one page-rounded shader allocation,
    /// up to [`MAX_SHADER_BYTES`]. Each program must fit 4 KiB including its own
    /// instruction-end padding; the whole library includes alignment gaps. Its
    /// actual storage charge is available from [`shader_library_bytes`]. Data slots and resource lifetimes match
    /// [`Self::open`]. Programs can be selected without reuploading shader storage.
    ///
    /// # Safety
    /// Every program must satisfy the trusted ISA, register, padding, termination
    /// and bounded-access requirements of [`Self::open`]. All share exactly
    /// [`GFX1151_SERIAL_ABI`]. No untrusted program bytes may enter this interface.
    pub unsafe fn open_programs(
        path: impl AsRef<Path>,
        budget: &MemoryBudget,
        programs: &[&[u8]],
        sizes: [u32; 3],
    ) -> Result<Self, ProbeError> {
        let device = AmdGpuDevice::open(path)?;
        // SAFETY: delegated unchanged from this method's caller contract.
        unsafe { Self::from_programs(device, budget, programs, sizes) }
    }

    /// Open a library whose programs use a derived encoder ABI.
    ///
    /// `abi` must come from [`KernelAbi::with_vgprs`] and/or
    /// [`KernelAbi::with_max_groups`] applied to [`GFX1151_SERIAL_ABI`]. A library
    /// opened this way is independent of every other library: raising its
    /// register budget or workgroup limit does not change libraries opened
    /// through [`Self::open_programs`].
    ///
    /// # Safety
    /// Same contract as [`Self::open_programs`], except that `kernel` may use
    /// up to `abi.vgprs()` VGPRs and dispatches may use up to `abi.max_groups()`
    /// groups. The caller must independently audit both bounds for its programs.
    pub unsafe fn open_programs_with_abi(
        path: impl AsRef<Path>,
        budget: &MemoryBudget,
        programs: &[&[u8]],
        sizes: [u32; 3],
        abi: KernelAbi,
    ) -> Result<Self, ProbeError> {
        let device = AmdGpuDevice::open(path)?;
        // SAFETY: delegated unchanged from this method's caller contract.
        unsafe { Self::from_programs_with_abi(device, budget, programs, sizes, abi) }
    }

    unsafe fn from_programs(
        device: AmdGpuDevice,
        budget: &MemoryBudget,
        programs: &[&[u8]],
        sizes: [u32; 3],
    ) -> Result<Self, ProbeError> {
        // SAFETY: delegated unchanged; the default ABI is the qualified one.
        unsafe { Self::from_programs_with_abi(device, budget, programs, sizes, GFX1151_SERIAL_ABI) }
    }

    unsafe fn from_programs_with_abi(
        device: AmdGpuDevice,
        budget: &MemoryBudget,
        programs: &[&[u8]],
        sizes: [u32; 3],
        abi: KernelAbi,
    ) -> Result<Self, ProbeError> {
        // Admit only an ABI this encoder reasoned about: the qualified one, or
        // the qualified one with a different VGPR granule. Anything else would
        // let a caller write arbitrary bits into privileged registers.
        if abi.raw_descriptor != GFX1151_SERIAL_ABI.raw_descriptor
            || abi.threads != GFX1151_SERIAL_ABI.threads
            || abi.program_resources[1] != GFX1151_SERIAL_ABI.program_resources[1]
            || abi.program_resources[2] != GFX1151_SERIAL_ABI.program_resources[2]
            || abi.program_resources[0] & !0x3f != GFX1151_SERIAL_ABI.program_resources[0] & !0x3f
            || GFX1151_SERIAL_ABI
                .with_vgprs(abi.vgprs())
                .and_then(|derived| derived.with_max_groups(abi.max_groups()))
                != Some(abi)
        {
            return Err(invalid(
                "kernel ABI must be GFX1151_SERIAL_ABI, optionally with a derived \
                 VGPR count and workgroup limit",
            ));
        }
        let (shader_page, program_offsets) = pack_programs(programs)?;
        if sizes
            .iter()
            .any(|size| *size == 0 || *size > MAX_SLOT_BYTES)
        {
            return Err(invalid(
                "data slot capacity must be 1 byte to 4 GiB minus 4 KiB",
            ));
        }
        let ip = probe::query_ip(&device)?;
        if !probe::supports(device.info(), &ip) {
            return Err(invalid("experimental queue requires allowlisted GC 11.5.1"));
        }
        // SAFETY: read-only host geometry query.
        if unsafe { libc::sysconf(libc::_SC_PAGESIZE) } != PAGE as i64 {
            return Err(invalid("experimental queue requires 4-KiB host pages"));
        }
        let details: uapi::DeviceInfo = device.inner.query(uapi::INFO_DEVICE)?;
        let allocation_sizes = [
            shader_page.len() as u64,
            PAGE,
            guarded_slot_bytes(sizes[0]),
            guarded_slot_bytes(sizes[1]),
            guarded_slot_bytes(sizes[2]),
        ];
        let addresses = plan_addresses(&details, &ip, allocation_sizes)?;
        let mut session = Session::new(device);
        let mut context_request = abi::Context {
            input: abi::ContextIn {
                op: abi::CTX_ALLOC,
                priority: abi::CTX_PRIORITY_NORMAL,
                ..Default::default()
            },
        };
        probe::once(
            &session.resources().device,
            abi::CTX,
            &mut context_request,
            "create compute context",
        )?;
        // SAFETY: successful context allocation initialized the output member.
        let context = unsafe { context_request.allocated.ctx_id };
        session.resources_mut().context = Some(context);
        for (index, size) in allocation_sizes.into_iter().enumerate() {
            let mut buffer = session
                .resources()
                .device
                .allocate(size, Heap::Gtt, budget)?;
            // Map every allocation before GPU use; first-map zeroing must never
            // erase the result of a successful device dispatch. Allocation sizes
            // are page-rounded; one fixed page also bounds initialization staging
            // independently of the requested slot capacities.
            let initial_page = [if index >= 2 { CANARY } else { 0 }; PAGE as usize];
            for offset in (0..size).step_by(PAGE as usize) {
                buffer.write(offset, &initial_page)?;
            }
            if index == 0 {
                buffer.write(0, &shader_page)?;
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
                flags: abi::VM_PAGE_READABLE
                    // The command processor fetches the IB as executable GPU
                    // memory too. R-only causes CPF execute-permission faults.
                    | if index < 2 {
                        abi::VM_PAGE_EXECUTABLE
                    } else {
                        abi::VM_PAGE_WRITEABLE
                    },
                va_address: mapping.address,
                map_size: size,
                ..Default::default()
            };
            session.uncertain = true;
            probe::once(
                &session.resources().device,
                abi::GEM_VA,
                &mut request,
                "map compute VA",
            )?;
            session.resources_mut().mappings.push(mapping);
            session.uncertain = false;
        }
        let entries: Vec<_> = session
            .resources()
            .buffers
            .iter()
            .map(|buffer| abi::BoListEntry {
                bo_handle: buffer.handle,
                bo_priority: 0,
            })
            .collect();
        let mut list_request = abi::BoList {
            input: abi::BoListIn {
                operation: abi::BO_LIST_CREATE,
                bo_number: entries.len() as u32,
                bo_info_size: size_of::<abi::BoListEntry>() as u32,
                bo_info_ptr: entries.as_ptr() as usize as u64,
                ..Default::default()
            },
        };
        session.uncertain = true;
        probe::once(
            &session.resources().device,
            abi::BO_LIST,
            &mut list_request,
            "create compute BO list",
        )?;
        // SAFETY: successful list creation initialized its output member.
        let list = unsafe { list_request.output.list_handle };
        session.resources_mut().list = Some(list);
        session.uncertain = false;
        Ok(Self {
            session,
            sizes,
            addresses,
            alignment: ip.ib_size_alignment,
            allocated_bytes: allocation_sizes.iter().sum(),
            context,
            list,
            trace_submissions: std::env::var("HAG_TRACE_SUBMISSION").as_deref() == Ok("1"),
            program_offsets,
            abi,
        })
    }

    /// Total page-rounded allocation charge, including program, IB and guards.
    pub fn allocation_bytes(&self) -> u64 {
        self.allocated_bytes
    }

    /// Whether uncertainty has permanently disabled this queue instance.
    pub fn is_poisoned(&self) -> bool {
        self.session.uncertain
    }

    /// Whether explicit development tracing will read operand payloads on the host.
    ///
    /// Captured from `HAG_TRACE_SUBMISSION=1` at construction. Resident execution
    /// adapters can refuse this mode so their no-intermediate-readback guarantee
    /// cannot be invalidated by a diagnostic environment setting.
    pub fn payload_tracing_enabled(&self) -> bool {
        self.trace_submissions
    }

    fn ready(&self) -> Result<(), ProbeError> {
        if self.is_poisoned() {
            return Err(Error::Quarantined.into());
        }
        Ok(())
    }

    fn range(&self, slot: usize, offset: u64, length: usize) -> Result<u64, ProbeError> {
        self.ready()?;
        let size = *self
            .sizes
            .get(slot)
            .ok_or_else(|| invalid("unknown data slot"))?;
        payload_range(size, offset, length)
    }

    /// Copy into an idle slot without exposing device addresses or mapped aliases.
    pub fn write(&mut self, slot: usize, offset: u64, bytes: &[u8]) -> Result<(), ProbeError> {
        let start = self.range(slot, offset, bytes.len())?;
        self.session.resources_mut().buffers[slot + 2].write(start, bytes)?;
        Ok(())
    }

    /// Copy from an idle slot; poisoned/pending queues refuse all host access.
    pub fn read(&mut self, slot: usize, offset: u64, bytes: &mut [u8]) -> Result<(), ProbeError> {
        let start = self.range(slot, offset, bytes.len())?;
        self.session.resources_mut().buffers[slot + 2].read(start, bytes)?;
        Ok(())
    }

    /// Submit, synchronously complete and validate one dispatch on resident storage.
    ///
    /// Descriptor lengths must fit each slot, both scalar values are 1..=65536,
    /// and X workgroup count is 1..=64. Shader/IB/data allocations and the BO list
    /// are reused. This is serial, not a concurrent or asynchronous queue API.
    ///
    /// # Safety
    /// The caller must prove that these lengths, scalars and group count satisfy
    /// the immutable kernel's complete shape/access/termination contract. Generic
    /// bounds checks cannot validate model-specific shader indexing. Kernel writes
    /// must finish before program termination and remain within its descriptors.
    pub unsafe fn dispatch(
        &mut self,
        lengths: [u32; 3],
        scalars: [u32; 2],
        groups: u32,
    ) -> Result<u64, ProbeError> {
        let bindings = std::array::from_fn(|slot| BufferRegion {
            slot,
            offset: 0,
            length: lengths[slot],
        });
        // SAFETY: original method's contract applies to program zero and these
        // full-prefix slot descriptors exactly as before.
        unsafe { self.dispatch_regions(0, bindings, scalars, groups) }
    }

    /// Run an immutable program against disjoint regions of resident data slots.
    ///
    /// Program selection and descriptor rebinding do not allocate, upload a shader,
    /// or stage tensor values through the host. Consecutive successful calls retain
    /// device-written intermediates. Each call still uses its own submission/fence;
    /// this is not a multi-dispatch IB or asynchronous batch. Diagnostic payload
    /// tracing, if explicitly enabled, reads operands and is not a residency proof.
    ///
    /// # Safety
    /// The selected program must obey [`Self::open_programs`]'s contract. The caller
    /// must prove descriptor alignment, scalar geometry, indexing and termination
    /// for these exact regions. No descriptor ranges may overlap. Each descriptor
    /// is bounded to its region, not the surrounding arena allocation.
    pub unsafe fn dispatch_regions(
        &mut self,
        program: usize,
        bindings: [BufferRegion; 3],
        scalars: [u32; 2],
        groups: u32,
    ) -> Result<u64, ProbeError> {
        self.ready()?;
        let program_offset = *self
            .program_offsets
            .get(program)
            .ok_or_else(|| invalid("unknown immutable compute program"))?;
        let lengths = bindings.map(|binding| binding.length);
        if lengths.contains(&0)
            || scalars.iter().any(|value| *value == 0 || *value > 65536)
            || groups == 0
            || groups > self.abi.max_groups()
        {
            return Err(invalid("dispatch arguments exceed the bounded queue ABI"));
        }
        let mut data_addresses = [0; 3];
        for (index, binding) in bindings.iter().enumerate() {
            let payload_offset =
                self.range(binding.slot, binding.offset, binding.length as usize)?;
            data_addresses[index] = self.addresses[binding.slot + 2]
                .checked_add(payload_offset)
                .ok_or_else(|| invalid("compute descriptor address overflow"))?;
        }
        let commands = packets::commands(
            self.addresses[0] + u64::from(program_offset),
            data_addresses,
            lengths,
            scalars,
            groups,
            self.alignment,
            &self.abi,
        )
        .map_err(invalid)?;
        self.session.resources_mut().buffers[1].write(0, &commands)?;
        // Explicit development diagnostic only: dump idle storage before CS.
        // Never read mapped buffers after completion becomes uncertain.
        let trace = self.trace_submissions;
        if trace {
            eprintln!("HAG_TRACE program={program} bindings={bindings:?}");
            eprintln!("HAG_TRACE context={} list={} lengths={lengths:?} scalars={scalars:?} groups={groups}", self.context, self.list);
            for (index, mapping) in self.session.resources().mappings.iter().enumerate() {
                eprintln!(
                    "HAG_TRACE mapping={index} handle={} address={:#x} bytes={} permissions={}",
                    mapping.handle,
                    mapping.address,
                    mapping.size,
                    if index < 2 { "RX" } else { "RW" }
                );
            }
            for (index, length) in [
                PAGE as usize,
                commands.len(),
                lengths[0] as usize,
                lengths[1] as usize,
                lengths[2] as usize,
            ]
            .into_iter()
            .enumerate()
            {
                let mut observed = vec![0; length];
                let (buffer_index, offset) = if index < 2 {
                    (index, 0)
                } else {
                    let binding = bindings[index - 2];
                    (binding.slot + 2, PAGE + binding.offset)
                };
                self.session.resources_mut().buffers[buffer_index].read(offset, &mut observed)?;
                for (offset, chunk) in observed.chunks(32).enumerate() {
                    let hex: String = chunk.iter().map(|byte| format!("{byte:02x}")).collect();
                    eprintln!(
                        "HAG_TRACE buffer={index} offset={:#x} hex={hex}",
                        offset * 32
                    );
                }
            }
        }
        let deadline = probe::completion_deadline()?;
        let mut ib = abi::CsChunkIb {
            flags: abi::IB_FLAG_EMIT_MEM_SYNC,
            va_start: self.addresses[1],
            ib_bytes: commands.len() as u32,
            ip_type: abi::HW_IP_COMPUTE,
            ..Default::default()
        };
        let mut chunk = abi::CsChunk {
            chunk_id: abi::CHUNK_ID_IB,
            length_dw: (size_of::<abi::CsChunkIb>() / 4) as u32,
            chunk_data: (&mut ib as *mut abi::CsChunkIb) as usize as u64,
        };
        let pointers = [(&mut chunk as *mut abi::CsChunk) as usize as u64];
        if trace {
            eprintln!("HAG_TRACE submit ip={} instance={} ring={} ib_va={:#x} ib_bytes={} flags={:#x} deadline_ns={deadline}", ib.ip_type, ib.ip_instance, ib.ring, ib.va_start, ib.ib_bytes, ib.flags);
        }
        let mut submit = abi::Cs {
            input: abi::CsIn {
                ctx_id: self.context,
                bo_list_handle: self.list,
                num_chunks: 1,
                chunks: pointers.as_ptr() as usize as u64,
                ..Default::default()
            },
        };
        fence(Ordering::SeqCst);
        // This state survives a caught panic, not merely an error return. Never
        // replay CS, reuse the IB or permit a host transfer after uncertainty.
        self.session.uncertain = true;
        probe::once(
            &self.session.resources().device,
            abi::CS,
            &mut submit,
            "submit compute once",
        )?;
        // SAFETY: successful CS initialized the sequence output.
        let sequence = unsafe { submit.handle };
        if trace {
            eprintln!("HAG_TRACE submitted sequence={sequence}");
        }
        if sequence == 0 || sequence == u64::MAX {
            return Err(invalid("invalid compute fence sequence"));
        }
        let mut wait = abi::WaitCs {
            input: abi::WaitCsIn {
                handle: sequence,
                timeout: deadline,
                ip_type: abi::HW_IP_COMPUTE,
                ctx_id: self.context,
                ..Default::default()
            },
        };
        let waited = probe::once(
            &self.session.resources().device,
            abi::WAIT_CS,
            &mut wait,
            "wait compute fence",
        );
        if trace {
            match &waited {
                // SAFETY: only a successful ioctl initializes the output member.
                Ok(()) => eprintln!("HAG_TRACE wait sequence={sequence} status={}", unsafe {
                    wait.status
                }),
                Err(error) => eprintln!("HAG_TRACE wait sequence={sequence} error={error}"),
            }
        }
        // SAFETY: short-circuiting reads wait.status only after a successful wait.
        if trace && (waited.is_err() || unsafe { wait.status } != 0) {
            // A read-only diagnostic query does not clear poison or allow reuse.
            let mut diagnostic = abi::Context {
                input: abi::ContextIn {
                    op: abi::CTX_QUERY_STATE2,
                    ctx_id: self.context,
                    ..Default::default()
                },
            };
            match probe::once(
                &self.session.resources().device,
                abi::CTX,
                &mut diagnostic,
                "diagnostic compute context",
            ) {
                // SAFETY: successful query initialized the state output.
                Ok(()) => eprintln!("HAG_TRACE context_flags={:#x}", unsafe {
                    diagnostic.state.flags
                }),
                Err(error) => eprintln!("HAG_TRACE context_error={error}"),
            }
        }
        waited?;
        // SAFETY: successful wait initialized its output status.
        if unsafe { wait.status } != 0 {
            return Err(ProbeError::Incomplete);
        }
        let mut status = abi::Context {
            input: abi::ContextIn {
                op: abi::CTX_QUERY_STATE2,
                ctx_id: self.context,
                ..Default::default()
            },
        };
        probe::once(
            &self.session.resources().device,
            abi::CTX,
            &mut status,
            "check compute context",
        )?;
        // SAFETY: successful state query initialized its output flags.
        let flags = unsafe { status.state.flags };
        if trace {
            eprintln!("HAG_TRACE context_flags={flags:#x}");
        }
        if flags != 0 {
            return Err(ProbeError::ContextFault(flags));
        }
        fence(Ordering::SeqCst);
        self.check_guards()?;
        self.session.uncertain = false;
        Ok(sequence)
    }

    fn check_guards(&mut self) -> Result<(), ProbeError> {
        for (slot, size) in self.sizes.into_iter().enumerate() {
            let buffer = &mut self.session.resources_mut().buffers[slot + 2];
            let mut prefix = [0; PAGE as usize];
            buffer.read(0, &mut prefix)?;
            let tail_start = PAGE + u64::from(size);
            let mut suffix = vec![0; (buffer.size_bytes() - tail_start) as usize];
            buffer.read(tail_start, &mut suffix)?;
            let guards = prefix
                .into_iter()
                .enumerate()
                .map(|(index, byte)| (index as u64, byte))
                .chain(
                    suffix
                        .into_iter()
                        .enumerate()
                        .map(|(index, byte)| (tail_start + index as u64, byte)),
                );
            for (offset, actual) in guards {
                if actual != CANARY {
                    return Err(ProbeError::GuardCorruption {
                        slot,
                        offset,
                        actual,
                    });
                }
            }
        }
        Ok(())
    }

    /// Release idle resources; an uncertain queue quarantines them until exit.
    pub fn close(mut self) -> Result<(), ProbeError> {
        self.session.close()
    }
}

/// Check immutable library geometry and report its page-rounded shader charge.
///
/// This performs no device access and does not validate shader instructions or
/// reserve budget. Actual queue construction admits each hardware allocation
/// separately. The trusted-code requirements of [`ComputeQueue::open_programs`]
/// still apply when the library is opened.
pub fn shader_library_bytes(programs: &[&[u8]]) -> Result<u64, ProbeError> {
    program_layout(programs).map(|(bytes, _)| bytes)
}

fn program_layout(programs: &[&[u8]]) -> Result<(u64, Vec<u32>), ProbeError> {
    if programs.is_empty() || programs.len() > MAX_PROGRAMS {
        return Err(invalid(
            "compute library requires one through thirty-two programs",
        ));
    }
    let mut offsets = Vec::with_capacity(programs.len());
    let mut next = 0;
    for program in programs {
        if program.is_empty() || program.len() > PAGE as usize || program.len() % 4 != 0 {
            return Err(invalid("shader must contain 4-byte words and fit one page"));
        }
        let offset = (next + PROGRAM_ALIGNMENT - 1) & !(PROGRAM_ALIGNMENT - 1);
        let end = offset + program.len();
        if end as u64 > MAX_SHADER_BYTES {
            return Err(invalid("aligned compute library exceeds 32 KiB"));
        }
        offsets.push(offset as u32);
        next = end;
    }
    Ok(((next as u64).div_ceil(PAGE) * PAGE, offsets))
}

fn pack_programs(programs: &[&[u8]]) -> Result<(Vec<u8>, Vec<u32>), ProbeError> {
    let (bytes, offsets) = program_layout(programs)?;
    let mut page = vec![0; bytes as usize];
    for (program, &offset) in programs.iter().zip(&offsets) {
        let start = offset as usize;
        page[start..start + program.len()].copy_from_slice(program);
    }
    Ok((page, offsets))
}

fn guarded_slot_bytes(size: u32) -> u64 {
    u64::from(size).div_ceil(PAGE) * PAGE + 2 * PAGE
}

fn payload_range(size: u32, offset: u64, length: usize) -> Result<u64, ProbeError> {
    if offset
        .checked_add(length as u64)
        .filter(|end| *end <= u64::from(size))
        .is_none()
    {
        return Err(Error::OutOfBounds.into());
    }
    Ok(PAGE + offset)
}

fn plan_addresses(
    details: &uapi::DeviceInfo,
    ip: &abi::HwIpInfo,
    sizes: [u64; 5],
) -> Result<[u64; 5], ProbeError> {
    let alignment = u64::from(details.virtual_address_alignment)
        .max(u64::from(ip.ib_start_alignment))
        .max(PAGE);
    if !details.virtual_address_alignment.is_power_of_two()
        || !ip.ib_start_alignment.is_power_of_two()
        || !ip.ib_size_alignment.is_power_of_two()
        || ip.ib_size_alignment < 4
        || ip.ib_size_alignment > PAGE as u32
        || alignment > 65536
        || !alignment.is_power_of_two()
    {
        return Err(invalid("unqualified compute VA or IB alignment"));
    }
    let align = |value: u64| {
        value
            .checked_add(alignment - 1)
            .map(|sum| sum & !(alignment - 1))
            .ok_or_else(|| invalid("compute VA overflow"))
    };
    let mut next = align(details.virtual_address_offset.max(PAGE))?;
    let mut addresses = [0; 5];
    for (index, size) in sizes.into_iter().enumerate() {
        addresses[index] = next;
        let end = next
            .checked_add(size)
            .ok_or_else(|| invalid("compute VA overflow"))?;
        if end > details.virtual_address_max || end >= 1 << 48 {
            return Err(invalid("compute VA outside queried bounds"));
        }
        next = align(end)?;
    }
    Ok(addresses)
}
