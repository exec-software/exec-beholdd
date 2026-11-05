// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

use super::*;
use crate::linux::{
    probe_packets, submission_uapi as abi, uapi, AmdGpuDevice, DeviceInner, Kernel,
};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io;
use std::mem::size_of;
use std::os::fd::{AsRawFd, RawFd};
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, Weak};

const TEST_PAGE: u64 = 4096;
const TEST_SIZES: [u32; 3] = [17, 19, 32];
const TEST_CHARGE: u64 = 11 * TEST_PAGE;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Fault {
    #[default]
    None,
    AllocateThird,
    HostMapThird,
    VaMapThird,
    ListCreate,
    SubmitInterrupted,
    SubmitPanic,
    ZeroSequence,
    LatestSequence,
    WaitBusy,
    WaitInterrupted,
    WaitLost,
    ContextQuery,
    ContextFlags,
    FrontGuard,
    Padding,
    RearGuard,
    ListDestroy,
    VaUnmap,
    ContextFree,
    GemClose,
    HostUnmap,
}

#[derive(Default)]
struct Allocation {
    size: usize,
    pointer: usize,
    address: u64,
    permissions: u32,
}

#[derive(Default)]
struct State {
    fault: Fault,
    allocations: BTreeMap<u32, Allocation>,
    next_handle: u32,
    virtual_address_max: Option<u64>,
    events: Vec<&'static str>,
    once: Vec<libc::c_ulong>,
    ordinary: Vec<libc::c_ulong>,
    submits: u64,
    pending: bool,
    allow_pending_diagnostic_query: bool,
    shader_snapshot: Vec<u8>,
    flexible_bindings: bool,
    shader_entries: Vec<u64>,
    descriptor_lengths: Vec<[u32; 3]>,
    descriptor_locations: Vec<[(u32, usize); 3]>,
    scalar_pairs: Vec<[u32; 2]>,
}

struct MockKernel {
    state: Mutex<State>,
    budget: MemoryBudget,
}

impl MockKernel {
    fn set_fault(&self, fault: Fault) {
        self.state.lock().unwrap().fault = fault;
    }

    fn events(&self) -> Vec<&'static str> {
        self.state.lock().unwrap().events.clone()
    }

    fn failure<T>() -> io::Result<T> {
        Err(io::Error::from_raw_os_error(libc::EIO))
    }

    unsafe fn dispatch_ioctl(
        &self,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        let fault = state.fault;
        match request {
            uapi::VERSION => {
                let value = unsafe { &mut *argument.cast::<uapi::Version>() };
                assert!(value.name_len >= 6);
                unsafe { std::ptr::copy_nonoverlapping(b"amdgpu".as_ptr(), value.name.cast(), 6) };
                value.name_len = 6;
            }
            uapi::INFO => {
                let value = unsafe { &*argument.cast::<uapi::Info>() };
                match value.query {
                    uapi::INFO_DEVICE => {
                        let details = unsafe {
                            &mut *(value.return_pointer as usize as *mut uapi::DeviceInfo)
                        };
                        details.device_id = 0x1586;
                        details.family = 150;
                        details.virtual_address_offset = 0x10000;
                        details.virtual_address_max =
                            state.virtual_address_max.unwrap_or(0x1000000);
                        details.virtual_address_alignment = TEST_PAGE as u32;
                    }
                    abi::INFO_HW_IP_INFO => {
                        assert_eq!(value.query_data, [abi::HW_IP_COMPUTE, 0, 0, 0]);
                        let details =
                            unsafe { &mut *(value.return_pointer as usize as *mut abi::HwIpInfo) };
                        details.hw_ip_version_major = 11;
                        details.hw_ip_version_minor = 0;
                        details.ip_discovery_version = 0x0b0501;
                        details.available_rings = 1;
                        details.ib_start_alignment = 256;
                        details.ib_size_alignment = 32;
                    }
                    selector => panic!("unexpected INFO selector {selector}"),
                }
            }
            abi::CTX => {
                let value = unsafe { &mut *argument.cast::<abi::Context>() };
                let input = unsafe { value.input };
                assert_eq!(input.flags, 0);
                match input.op {
                    abi::CTX_ALLOC => {
                        state.events.push("context-create");
                        assert_eq!(input.priority, abi::CTX_PRIORITY_NORMAL);
                        value.allocated = abi::ContextAllocOut { ctx_id: 3, pad: 0 };
                    }
                    abi::CTX_QUERY_STATE2 => {
                        state.events.push("context-query");
                        assert!(
                            !state.pending || state.allow_pending_diagnostic_query,
                            "check health after completion, not instead of waiting"
                        );
                        if fault == Fault::ContextQuery {
                            return Self::failure();
                        }
                        value.state = abi::ContextStateOut {
                            flags: if fault == Fault::ContextFlags { 1 } else { 0 },
                            ..Default::default()
                        };
                    }
                    abi::CTX_FREE => {
                        state.events.push("context-free");
                        assert!(!state.pending);
                        if fault == Fault::ContextFree {
                            return Self::failure();
                        }
                    }
                    operation => panic!("unexpected context operation {operation}"),
                }
            }
            uapi::GEM_CREATE => {
                state.events.push("allocate");
                let value = unsafe { &mut *argument.cast::<uapi::GemCreate>() };
                let input = unsafe { value.input };
                state.next_handle += 1;
                let handle = state.next_handle;
                if handle == 3 && fault == Fault::AllocateThird {
                    return Self::failure();
                }
                assert_eq!(input.domains, uapi::DOMAIN_GTT);
                assert!(self.budget.current_bytes() >= input.bo_size);
                state.allocations.insert(
                    handle,
                    Allocation {
                        size: input.bo_size as usize,
                        ..Default::default()
                    },
                );
                value.output = uapi::GemCreateOut { handle, pad: 0 };
            }
            uapi::GEM_MMAP => {
                let value = unsafe { &mut *argument.cast::<uapi::GemMmap>() };
                value.offset = u64::from(unsafe { value.input.handle }) * TEST_PAGE;
            }
            abi::GEM_VA => {
                let value = unsafe { &*argument.cast::<abi::GemVa>() };
                assert_eq!(value.va_address % TEST_PAGE, 0);
                assert_eq!(value.offset_in_bo, 0);
                match value.operation {
                    abi::VA_MAP => {
                        state.events.push("va-map");
                        if value.handle == 3 && fault == Fault::VaMapThird {
                            return Self::failure();
                        }
                        let allocation = state.allocations.get_mut(&value.handle).unwrap();
                        assert_eq!(value.map_size as usize, allocation.size);
                        allocation.address = value.va_address;
                        allocation.permissions = value.flags;
                    }
                    abi::VA_UNMAP => {
                        state.events.push("va-unmap");
                        assert!(!state.pending);
                        if fault == Fault::VaUnmap {
                            return Self::failure();
                        }
                        let allocation = state.allocations.get_mut(&value.handle).unwrap();
                        assert_eq!(value.va_address, allocation.address);
                        allocation.address = 0;
                    }
                    operation => panic!("unexpected VA operation {operation}"),
                }
            }
            abi::BO_LIST => {
                let value = unsafe { &mut *argument.cast::<abi::BoList>() };
                let input = unsafe { value.input };
                match input.operation {
                    abi::BO_LIST_CREATE => {
                        state.events.push("list-create");
                        assert_eq!(input.bo_number, 5);
                        assert_eq!(input.bo_info_size as usize, size_of::<abi::BoListEntry>());
                        let entries = unsafe {
                            std::slice::from_raw_parts(
                                input.bo_info_ptr as usize as *const abi::BoListEntry,
                                5,
                            )
                        };
                        assert_eq!(
                            entries
                                .iter()
                                .map(|entry| entry.bo_handle)
                                .collect::<Vec<_>>(),
                            [1, 2, 3, 4, 5]
                        );
                        if fault == Fault::ListCreate {
                            return Self::failure();
                        }
                        value.output = abi::BoListOut {
                            list_handle: 4,
                            pad: 0,
                        };
                    }
                    abi::BO_LIST_DESTROY => {
                        state.events.push("list-destroy");
                        assert!(!state.pending);
                        if fault == Fault::ListDestroy {
                            return Self::failure();
                        }
                    }
                    operation => panic!("unexpected list operation {operation}"),
                }
            }
            abi::CS => {
                state.events.push("submit");
                assert!(!state.pending, "never replay a pending submission");
                state.pending = true;
                state.submits += 1;
                let value = unsafe { &mut *argument.cast::<abi::Cs>() };
                let input = unsafe { value.input };
                assert_eq!(input.ctx_id, 3);
                assert_eq!(input.bo_list_handle, 4);
                assert_eq!(input.num_chunks, 1);
                let chunk_pointer = unsafe { *(input.chunks as usize as *const u64) };
                let chunk = unsafe { &*(chunk_pointer as usize as *const abi::CsChunk) };
                assert_eq!(chunk.chunk_id, abi::CHUNK_ID_IB);
                assert_eq!(chunk.length_dw as usize * 4, size_of::<abi::CsChunkIb>());
                let ib = unsafe { &*(chunk.chunk_data as usize as *const abi::CsChunkIb) };
                assert_eq!(ib.ip_type, abi::HW_IP_COMPUTE);
                assert_eq!(ib.ip_instance, 0);
                assert_eq!(ib.ring, 0);
                assert_ne!(
                    ib.flags & (1 << 6),
                    0,
                    "host-rewritten inputs require pre-IB cache invalidation"
                );
                assert_eq!(ib.va_start, state.allocations[&2].address);
                assert!(ib.ib_bytes > 0 && u64::from(ib.ib_bytes) <= TEST_PAGE);
                assert_eq!(ib.ib_bytes % 32, 0);
                let command_storage = &state.allocations[&2];
                let command_bytes = unsafe {
                    std::slice::from_raw_parts(
                        command_storage.pointer as *const u8,
                        ib.ib_bytes as usize,
                    )
                };
                let command_words: Vec<_> = command_bytes
                    .chunks_exact(4)
                    .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
                    .collect();
                let parameters = command_words
                    .windows(16)
                    .find(|words| words[0] == 0xc00e7602 && words[1] == 0x240)
                    .expect("three SRDs plus two scalars in the trusted command ABI");
                let program = command_words
                    .windows(4)
                    .find(|words| words[0] == 0xc0027602 && words[1] == 0x20c)
                    .expect("shader entry in the trusted command ABI");
                let shader_entry = (u64::from(program[2]) << 8) | (u64::from(program[3]) << 40);
                let shader_base = state.allocations[&1].address;
                assert_eq!(shader_entry % 256, 0);
                let shader_bytes = state.allocations[&1].size as u64;
                assert!((shader_base..shader_base + shader_bytes).contains(&shader_entry));
                if !state.flexible_bindings {
                    assert_eq!(shader_entry, shader_base);
                }
                state.shader_entries.push(shader_entry);
                let mut lengths = [0; 3];
                let mut locations = [(0, 0); 3];
                for slot in 0..3 {
                    let words = &parameters[2 + slot * 4..6 + slot * 4];
                    let address = u64::from(words[0]) | (u64::from(words[1]) << 32);
                    if !state.flexible_bindings {
                        assert_eq!(
                            address,
                            state.allocations[&(slot as u32 + 3)].address + TEST_PAGE
                        );
                        assert!((1..=TEST_SIZES[slot]).contains(&words[2]));
                    }
                    assert_eq!(words[3], 0x30016fac);
                    assert_ne!(words[2], 0);
                    let (&handle, allocation) = state
                        .allocations
                        .iter()
                        .find(|(handle, allocation)| {
                            **handle >= 3
                                && address >= allocation.address + TEST_PAGE
                                && address.checked_add(u64::from(words[2])).is_some_and(|end| {
                                    end <= allocation.address
                                        + TEST_PAGE
                                        + u64::from(TEST_SIZES[**handle as usize - 3])
                                })
                        })
                        .expect("each descriptor stays within a logical data payload");
                    locations[slot] = (handle, (address - allocation.address) as usize);
                    lengths[slot] = words[2];
                }
                state.descriptor_lengths.push(lengths);
                state.descriptor_locations.push(locations);
                state.scalar_pairs.push([parameters[14], parameters[15]]);
                for (index, capacity) in TEST_SIZES.into_iter().enumerate() {
                    let allocation = &state.allocations[&(index as u32 + 3)];
                    assert_ne!(
                        allocation.pointer, 0,
                        "data buffers must be mapped before device access"
                    );
                    let bytes = unsafe {
                        std::slice::from_raw_parts(allocation.pointer as *const u8, allocation.size)
                    };
                    assert!(bytes[..TEST_PAGE as usize].iter().all(|byte| *byte == 0xa5));
                    assert!(bytes[TEST_PAGE as usize + capacity as usize..]
                        .iter()
                        .all(|byte| *byte == 0xa5));
                }
                let shader = &state.allocations[&1];
                let shader_bytes =
                    unsafe { std::slice::from_raw_parts(shader.pointer as *const u8, shader.size) };
                if state.shader_snapshot.is_empty() {
                    state.shader_snapshot = shader_bytes.to_vec();
                } else {
                    assert_eq!(shader_bytes, state.shader_snapshot);
                }
                assert_eq!(
                    self.budget.current_bytes(),
                    state
                        .allocations
                        .values()
                        .map(|allocation| allocation.size as u64)
                        .sum::<u64>()
                );
                if fault == Fault::SubmitPanic {
                    drop(state);
                    panic!("injected submission panic");
                }
                if fault == Fault::SubmitInterrupted {
                    return Err(io::Error::from_raw_os_error(libc::EINTR));
                }
                value.handle = match fault {
                    Fault::ZeroSequence => 0,
                    Fault::LatestSequence => u64::MAX,
                    _ => state.submits + 16,
                };
            }
            abi::WAIT_CS => {
                state.events.push("wait");
                let value = unsafe { &mut *argument.cast::<abi::WaitCs>() };
                let input = unsafe { value.input };
                assert_eq!(input.handle, state.submits + 16);
                assert!(input.timeout > 0 && input.timeout <= i64::MAX as u64);
                match fault {
                    Fault::WaitInterrupted => {
                        return Err(io::Error::from_raw_os_error(libc::EINTR))
                    }
                    Fault::WaitLost => return Err(io::Error::from_raw_os_error(libc::ENODEV)),
                    Fault::WaitBusy => {
                        value.status = 1;
                        return Ok(());
                    }
                    _ => {}
                }
                state.pending = false;
                let locations = *state.descriptor_locations.last().unwrap();
                let (input_handle, input_offset) = locations[1];
                let input_allocation = &state.allocations[&input_handle];
                let simulated_value =
                    unsafe { *((input_allocation.pointer + input_offset) as *const u8) };
                let (output_handle, output_offset) = locations[2];
                let output = &state.allocations[&output_handle];
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(output.pointer as *mut u8, output.size)
                };
                let output_length = state.descriptor_lengths.last().unwrap()[2] as usize;
                bytes[output_offset..output_offset + output_length].fill(simulated_value);
                match fault {
                    Fault::FrontGuard => bytes[0] ^= 1,
                    Fault::Padding => {
                        bytes[TEST_PAGE as usize
                            + TEST_SIZES[output_handle as usize - 3] as usize] ^= 1
                    }
                    Fault::RearGuard => bytes[output.size - 1] ^= 1,
                    _ => {}
                }
                value.status = 0;
            }
            uapi::GEM_CLOSE => {
                state.events.push("gem-close");
                assert!(!state.pending);
                if fault == Fault::GemClose {
                    return Self::failure();
                }
                let value = unsafe { &*argument.cast::<uapi::GemClose>() };
                let allocation = state.allocations.remove(&value.handle).unwrap();
                assert_eq!(allocation.pointer, 0);
                assert_eq!(allocation.address, 0);
            }
            _ => panic!("unexpected ioctl {request:x}"),
        }
        Ok(())
    }
}

impl Kernel for MockKernel {
    unsafe fn ioctl_once(
        &self,
        _fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        self.state.lock().unwrap().once.push(request);
        unsafe { self.dispatch_ioctl(request, argument) }
    }

    unsafe fn ioctl(
        &self,
        _fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        self.state.lock().unwrap().ordinary.push(request);
        unsafe { self.dispatch_ioctl(request, argument) }
    }

    unsafe fn map(&self, _fd: RawFd, length: usize, offset: libc::off_t) -> io::Result<*mut u8> {
        let mut state = self.state.lock().unwrap();
        state.events.push("host-map");
        let handle = (offset as u64 / TEST_PAGE) as u32;
        if handle == 3 && state.fault == Fault::HostMapThird {
            return Self::failure();
        }
        let allocation = state.allocations.get_mut(&handle).unwrap();
        assert_eq!(length, allocation.size);
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
        assert!(!pointer.is_null());
        allocation.pointer = pointer as usize;
        unsafe { std::ptr::write_bytes(pointer.cast::<u8>(), 0x5c, length) };
        Ok(pointer.cast())
    }

    unsafe fn unmap(&self, pointer: *mut u8, length: usize) -> io::Result<()> {
        let mut state = self.state.lock().unwrap();
        state.events.push("host-unmap");
        assert!(!state.pending);
        if state.fault == Fault::HostUnmap {
            return Self::failure();
        }
        let allocation = state
            .allocations
            .values_mut()
            .find(|allocation| allocation.pointer == pointer as usize)
            .unwrap();
        assert_eq!(allocation.size, length);
        assert_eq!(unsafe { libc::munmap(pointer.cast(), length) }, 0);
        allocation.pointer = 0;
        Ok(())
    }
}

struct Fixture {
    device: AmdGpuDevice,
    owner: Weak<DeviceInner>,
    kernel: Arc<MockKernel>,
    budget: MemoryBudget,
}

fn fixture(fault: Fault) -> Fixture {
    fixture_with_budget(fault, 64 * 1024)
}

fn fixture_with_budget(fault: Fault, limit_bytes: u64) -> Fixture {
    let budget = MemoryBudget::new(limit_bytes);
    let kernel = Arc::new(MockKernel {
        state: Mutex::new(State {
            fault,
            ..Default::default()
        }),
        budget: budget.clone(),
    });
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/null")
        .unwrap();
    let device = AmdGpuDevice::from_fd(
        file.into(),
        PathBuf::from("/mock/renderD128"),
        kernel.clone(),
    )
    .unwrap();
    let owner = Arc::downgrade(&device.inner);
    Fixture {
        device,
        owner,
        kernel,
        budget,
    }
}

fn open_queue(device: AmdGpuDevice, budget: &MemoryBudget) -> Result<ComputeQueue, ProbeError> {
    // No GPU observes this shader: the private injected kernel models submission.
    unsafe { ComputeQueue::from_device(device, budget, &probe_packets::shader_bytes(), TEST_SIZES) }
}

fn dispatch(queue: &mut ComputeQueue) -> Result<u64, ProbeError> {
    // Private mock only; no command reaches DRM or physical hardware.
    unsafe { queue.dispatch(TEST_SIZES, [1, 1], 1) }
}

fn region(slot: usize, offset: u64, length: u32) -> BufferRegion {
    BufferRegion {
        slot,
        offset,
        length,
    }
}

fn standard_regions() -> [BufferRegion; 3] {
    [region(0, 0, 17), region(1, 0, 19), region(2, 0, 32)]
}

fn storage_snapshot(kernel: &MockKernel) -> Vec<(u32, Vec<u8>)> {
    let state = kernel.state.lock().unwrap();
    assert!(
        !state.pending,
        "test snapshots must never inspect pending GPU storage"
    );
    state
        .allocations
        .iter()
        .map(|(&handle, allocation)| {
            // Only the injected driver's anonymous host mappings are inspected.
            let bytes = unsafe {
                std::slice::from_raw_parts(allocation.pointer as *const u8, allocation.size)
            };
            (handle, bytes.to_vec())
        })
        .collect()
}

fn assert_quarantined(owner: &Weak<DeviceInner>, budget: &MemoryBudget, charge: u64) {
    let retained = owner.upgrade().expect("quarantine retains owning device");
    assert!(retained.cleanup_failures.load(Ordering::Relaxed) > 0);
    assert_ne!(
        unsafe { libc::fcntl(retained.fd.as_raw_fd(), libc::F_GETFD) },
        -1
    );
    assert_eq!(budget.current_bytes(), charge);
}

fn assert_poisoned(queue: &mut ComputeQueue) {
    assert!(queue.is_poisoned());
    for result in [queue.write(0, 0, &[1]), queue.read(0, 0, &mut [0])] {
        assert!(matches!(
            result,
            Err(ProbeError::Transport(Error::Quarantined))
        ));
    }
    assert!(matches!(
        dispatch(queue),
        Err(ProbeError::Transport(Error::Quarantined))
    ));
}

#[test]
fn shader_and_command_fetch_require_execute_but_data_does_not() {
    let Fixture {
        device,
        kernel,
        budget,
        ..
    } = fixture(Fault::None);
    let queue = open_queue(device, &budget).unwrap();
    {
        let state = kernel.state.lock().unwrap();
        let permissions: Vec<_> = state
            .allocations
            .values()
            .map(|allocation| allocation.permissions)
            .collect();
        // Use explicit UAPI bits: R|X for code AND IB, R|W for all data.
        // In particular, an R-only IB reproduced a CPF fault at its exact VA.
        assert_eq!(permissions, [0x0a, 0x0a, 0x06, 0x06, 0x06]);
        assert!(permissions.iter().all(|flags| flags & 0x0c != 0x0c));
    }
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn programs_are_copied_into_one_immutable_aligned_shader_page() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let mut first = vec![0x11; 4];
    let second = vec![0x22; 260];
    let third = vec![0x33; 4];
    // These mock-only bytes are never executed by physical hardware.
    let mut queue = unsafe {
        ComputeQueue::from_programs(device, &budget, &[&first, &second, &third], TEST_SIZES)
    }
    .unwrap();
    assert_eq!(queue.program_offsets, [0, 256, 768]);
    assert_eq!(queue.allocation_bytes(), TEST_CHARGE);
    let mut expected = vec![0; TEST_PAGE as usize];
    expected[..first.len()].copy_from_slice(&first);
    expected[256..256 + second.len()].copy_from_slice(&second);
    expected[768..768 + third.len()].copy_from_slice(&third);
    first.fill(0x99);
    assert_eq!(storage_snapshot(&kernel)[0], (1, expected.clone()));
    let initialized = kernel.events().len();
    kernel.state.lock().unwrap().flexible_bindings = true;
    for program in [0, 1, 2, 0] {
        unsafe { queue.dispatch_regions(program, standard_regions(), [1, 1], 1) }.unwrap();
        assert_eq!(storage_snapshot(&kernel)[0], (1, expected.clone()));
        assert_eq!(budget.current_bytes(), TEST_CHARGE);
    }
    {
        let state = kernel.state.lock().unwrap();
        assert_eq!(state.shader_entries, [0x10000, 0x10100, 0x10300, 0x10000]);
        assert_eq!(state.allocations.len(), 5);
        assert_eq!(
            &state.events[initialized..],
            ["submit", "wait", "context-query"].repeat(4)
        );
    }
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn eight_programs_may_exactly_fill_the_existing_shader_allocation() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let programs: Vec<_> = (1..=8).map(|value| vec![value; 512]).collect();
    let borrowed: Vec<_> = programs.iter().map(Vec::as_slice).collect();
    let queue =
        unsafe { ComputeQueue::from_programs(device, &budget, &borrowed, TEST_SIZES) }.unwrap();
    assert_eq!(
        queue.program_offsets,
        [0, 512, 1024, 1536, 2048, 2560, 3072, 3584]
    );
    assert_eq!(storage_snapshot(&kernel)[0].1, programs.concat());
    assert_eq!(budget.current_bytes(), TEST_CHARGE);
    assert_eq!(kernel.state.lock().unwrap().allocations.len(), 5);
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn invalid_program_sets_refuse_before_queries_context_or_allocation() {
    for programs in [
        Vec::new(),
        vec![vec![0; 4]; 33],
        vec![Vec::new()],
        vec![vec![0; 3]],
        vec![vec![0; 4100]],
        vec![vec![0; 4096]; 9],
        [vec![vec![0; 4096]; 8], vec![vec![0; 4]]].concat(),
        vec![vec![0; 256], Vec::new()],
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(Fault::None);
        let borrowed: Vec<_> = programs.iter().map(Vec::as_slice).collect();
        assert!(shader_library_bytes(&borrowed).is_err());
        let before = {
            let state = kernel.state.lock().unwrap();
            (state.once.clone(), state.ordinary.clone())
        };
        assert!(
            unsafe { ComputeQueue::from_programs(device, &budget, &borrowed, TEST_SIZES) }.is_err()
        );
        let state = kernel.state.lock().unwrap();
        assert_eq!((state.once.clone(), state.ordinary.clone()), before);
        assert!(state.events.is_empty());
        assert!(state.allocations.is_empty());
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
    }
}

#[test]
fn shader_library_crosses_pages_without_reupload_or_changing_region_ownership() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let programs = [vec![0x11; 3840], vec![0x22; 512], vec![0x33; 260]];
    let borrowed: Vec<_> = programs.iter().map(Vec::as_slice).collect();
    assert_eq!(shader_library_bytes(&borrowed).unwrap(), 2 * TEST_PAGE);
    let mut queue =
        unsafe { ComputeQueue::from_programs(device, &budget, &borrowed, TEST_SIZES) }.unwrap();
    assert_eq!(queue.program_offsets, [0, 3840, 4352]);
    assert_eq!(queue.allocation_bytes(), TEST_CHARGE + TEST_PAGE);
    let mut expected = vec![0; 2 * TEST_PAGE as usize];
    for (program, &offset) in programs.iter().zip(&queue.program_offsets) {
        expected[offset as usize..offset as usize + program.len()].copy_from_slice(program);
    }
    assert_eq!(storage_snapshot(&kernel)[0], (1, expected.clone()));
    kernel.state.lock().unwrap().flexible_bindings = true;
    let initialized = kernel.events().len();
    for index in [1, 2, 0, 2] {
        unsafe { queue.dispatch_regions(index, standard_regions(), [1, 1], 1) }.unwrap();
        assert_eq!(storage_snapshot(&kernel)[0], (1, expected.clone()));
        assert_eq!(budget.current_bytes(), TEST_CHARGE + TEST_PAGE);
    }
    {
        let state = kernel.state.lock().unwrap();
        assert_eq!(state.shader_entries, [0x10f00, 0x11100, 0x10000, 0x11100]);
        assert_eq!(state.allocations.len(), 5);
        assert_eq!(
            state.allocations[&1].permissions,
            abi::VM_PAGE_READABLE | abi::VM_PAGE_EXECUTABLE
        );
        assert_eq!(
            &state.events[initialized..],
            ["submit", "wait", "context-query"].repeat(4)
        );
    }
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn sixteen_programs_can_fill_four_shader_pages_and_charge_the_actual_storage() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let programs: Vec<_> = (1..=16).map(|value| vec![value; 1024]).collect();
    let borrowed: Vec<_> = programs.iter().map(Vec::as_slice).collect();
    assert_eq!(shader_library_bytes(&borrowed).unwrap(), 4 * TEST_PAGE);
    let queue =
        unsafe { ComputeQueue::from_programs(device, &budget, &borrowed, TEST_SIZES) }.unwrap();
    assert_eq!(
        queue.program_offsets,
        (0..16).map(|n| n * 1024).collect::<Vec<_>>()
    );
    assert_eq!(storage_snapshot(&kernel)[0].1, programs.concat());
    assert_eq!(queue.allocation_bytes(), TEST_CHARGE + 3 * TEST_PAGE);
    assert_eq!(budget.current_bytes(), queue.allocation_bytes());
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn thirty_two_programs_fill_eight_pages_and_dispatch_the_last_entry_without_reupload() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture_with_budget(Fault::None, TEST_CHARGE + 7 * TEST_PAGE);
    let programs: Vec<_> = (1..=32).map(|value| vec![value; 1024]).collect();
    let borrowed: Vec<_> = programs.iter().map(Vec::as_slice).collect();
    assert_eq!(shader_library_bytes(&borrowed).unwrap(), MAX_SHADER_BYTES);
    let mut queue =
        unsafe { ComputeQueue::from_programs(device, &budget, &borrowed, TEST_SIZES) }.unwrap();
    assert_eq!(
        queue.program_offsets,
        (0..32).map(|n| n * 1024).collect::<Vec<_>>()
    );
    assert_eq!(queue.allocation_bytes(), TEST_CHARGE + 7 * TEST_PAGE);
    assert_eq!(budget.current_bytes(), queue.allocation_bytes());
    let shader = storage_snapshot(&kernel)[0].clone();
    assert_eq!(shader.1, programs.concat());
    kernel.state.lock().unwrap().flexible_bindings = true;
    let initialized = kernel.events().len();
    for index in [16, 31, 0, 31] {
        unsafe { queue.dispatch_regions(index, standard_regions(), [1, 1], 1) }.unwrap();
        assert_eq!(storage_snapshot(&kernel)[0], shader);
    }
    {
        let state = kernel.state.lock().unwrap();
        assert_eq!(state.shader_entries, [0x14000, 0x17c00, 0x10000, 0x17c00]);
        assert_eq!(
            &state.events[initialized..],
            ["submit", "wait", "context-query"].repeat(4)
        );
    }
    let events = kernel.events();
    assert!(unsafe { queue.dispatch_regions(32, standard_regions(), [1, 1], 1) }.is_err());
    assert_eq!(kernel.events(), events);
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn larger_shader_charge_is_admitted_before_its_gem_allocation() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture_with_budget(Fault::None, 2 * TEST_PAGE - 1);
    let programs = [vec![0; 3840], vec![0; 512]];
    let borrowed: Vec<_> = programs.iter().map(Vec::as_slice).collect();
    assert_eq!(shader_library_bytes(&borrowed).unwrap(), 2 * TEST_PAGE);
    assert!(
        unsafe { ComputeQueue::from_programs(device, &budget, &borrowed, TEST_SIZES) }.is_err()
    );
    assert!(!kernel.events().contains(&"allocate"));
    assert!(kernel.state.lock().unwrap().allocations.is_empty());
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn region_dispatch_remaps_slots_byte_offsets_and_programs_without_reallocation() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let programs = [vec![0x11; 4], vec![0x22; 260], vec![0x33; 4]];
    let mut queue = unsafe {
        ComputeQueue::from_programs(
            device,
            &budget,
            &[&programs[0], &programs[1], &programs[2]],
            TEST_SIZES,
        )
    }
    .unwrap();
    kernel.state.lock().unwrap().flexible_bindings = true;
    let initialized = kernel.events().len();
    let cases = [
        [region(2, 1, 3), region(0, 2, 5), region(1, 3, 7)],
        [region(1, 1, 5), region(2, 3, 7), region(0, 1, 9)],
        [region(0, 1, 3), region(1, 2, 5), region(2, 7, 9)],
    ];
    for (program, bindings) in cases.into_iter().enumerate() {
        let mut expected: Vec<_> = TEST_SIZES
            .into_iter()
            .enumerate()
            .map(|(slot, size)| vec![0x40 + program as u8 * 3 + slot as u8; size as usize])
            .collect();
        for (slot, bytes) in expected.iter().enumerate() {
            queue.write(slot, 0, bytes).unwrap();
        }
        let value = expected[bindings[1].slot][bindings[1].offset as usize];
        let output = &bindings[2];
        expected[output.slot]
            [output.offset as usize..output.offset as usize + output.length as usize]
            .fill(value);
        assert_eq!(
            unsafe { queue.dispatch_regions(program, bindings, [program as u32 + 1, 2], 1) }
                .unwrap(),
            program as u64 + 17
        );
        for (slot, bytes) in expected.iter().enumerate() {
            let mut observed = vec![0; bytes.len()];
            queue.read(slot, 0, &mut observed).unwrap();
            assert_eq!(
                &observed, bytes,
                "only the selected output region may change"
            );
        }
        assert_eq!(budget.current_bytes(), TEST_CHARGE);
        assert!(!queue.is_poisoned());
    }
    {
        let state = kernel.state.lock().unwrap();
        let p = TEST_PAGE as usize;
        assert_eq!(
            state.descriptor_locations,
            [
                [(5, p + 1), (3, p + 2), (4, p + 3)],
                [(4, p + 1), (5, p + 3), (3, p + 1)],
                [(3, p + 1), (4, p + 2), (5, p + 7)],
            ]
        );
        assert_eq!(state.descriptor_lengths, [[3, 5, 7], [5, 7, 9], [3, 5, 9]]);
        assert_eq!(state.scalar_pairs, [[1, 2], [2, 2], [3, 2]]);
        assert_eq!(state.shader_entries, [0x10000, 0x10100, 0x10300]);
        assert_eq!(
            &state.events[initialized..],
            ["submit", "wait", "context-query"].repeat(3)
        );
        assert_eq!(state.allocations.len(), 5);
    }
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn adjacent_disjoint_regions_in_one_slot_are_valid_and_preserve_other_bytes() {
    let Fixture {
        device,
        kernel,
        budget,
        ..
    } = fixture(Fault::None);
    let mut queue = open_queue(device, &budget).unwrap();
    kernel.state.lock().unwrap().flexible_bindings = true;
    let mut expected: Vec<u8> = (0..17).collect();
    queue.write(0, 0, &expected).unwrap();
    unsafe {
        queue.dispatch_regions(
            0,
            [region(0, 1, 3), region(0, 4, 5), region(0, 9, 8)],
            [1, 1],
            1,
        )
    }
    .unwrap();
    expected[9..17].fill(4);
    let mut observed = vec![0; 17];
    queue.read(0, 0, &mut observed).unwrap();
    assert_eq!(observed, expected);
    assert!(!queue.is_poisoned());
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn invalid_program_regions_and_writable_aliases_leave_all_storage_untouched() {
    let Fixture {
        device,
        kernel,
        budget,
        ..
    } = fixture(Fault::None);
    let mut queue = open_queue(device, &budget).unwrap();
    let before = storage_snapshot(&kernel);
    let events = kernel.events();
    for (program, bindings) in [
        (1, standard_regions()),
        (usize::MAX, standard_regions()),
        (0, [region(3, 0, 1), region(1, 0, 1), region(2, 0, 1)]),
        (
            0,
            [region(usize::MAX, 0, 1), region(1, 0, 1), region(2, 0, 1)],
        ),
        (0, [region(0, 0, 0), region(1, 0, 1), region(2, 0, 1)]),
        (0, [region(0, 17, 1), region(1, 0, 1), region(2, 0, 1)]),
        (0, [region(0, 16, 2), region(1, 0, 1), region(2, 0, 1)]),
        (
            0,
            [region(0, u64::MAX, 1), region(1, 0, 1), region(2, 0, 1)],
        ),
        (
            0,
            [region(0, 1, u32::MAX), region(1, 0, 1), region(2, 0, 1)],
        ),
        (0, [region(0, 0, 1), region(1, 19, 1), region(2, 0, 1)]),
        (0, [region(0, 0, 1), region(1, 0, 1), region(2, 32, 1)]),
        (0, [region(0, 0, 1), region(1, 0, 1), region(2, 0, 0)]),
        (0, [region(0, 2, 4), region(1, 0, 4), region(0, 2, 4)]),
        (0, [region(0, 2, 4), region(1, 0, 4), region(0, 5, 4)]),
        (0, [region(0, 0, 4), region(1, 2, 4), region(1, 1, 2)]),
    ] {
        assert!(unsafe { queue.dispatch_regions(program, bindings, [1, 1], 1) }.is_err());
        assert!(!queue.is_poisoned());
        assert_eq!(kernel.events(), events);
        assert_eq!(storage_snapshot(&kernel), before);
        assert_eq!(budget.current_bytes(), TEST_CHARGE);
    }
    dispatch(&mut queue).unwrap();
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
}

#[test]
fn repeated_dispatch_reuses_all_resources_and_observes_rewritten_input() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let mut queue = open_queue(device, &budget).unwrap();
    assert_eq!(queue.allocation_bytes(), TEST_CHARGE);
    let initialized_events = kernel.events();
    for (index, input) in [0x22, 0x73].into_iter().enumerate() {
        queue.write(1, 0, &[input; TEST_SIZES[1] as usize]).unwrap();
        assert_eq!(dispatch(&mut queue).unwrap(), 17 + index as u64);
        assert!(!queue.is_poisoned());
        let mut output = [0; TEST_SIZES[2] as usize];
        queue.read(2, 0, &mut output).unwrap();
        assert_eq!(output, [input; TEST_SIZES[2] as usize]);
        assert_eq!(budget.current_bytes(), TEST_CHARGE);
    }
    let events = kernel.events();
    assert_eq!(
        &events[initialized_events.len()..],
        [
            "submit",
            "wait",
            "context-query",
            "submit",
            "wait",
            "context-query"
        ]
    );
    let state = kernel.state.lock().unwrap();
    assert_eq!(
        state
            .once
            .iter()
            .filter(|request| **request == abi::CS)
            .count(),
        2
    );
    assert!(!state.ordinary.contains(&abi::CS));
    assert_eq!(state.allocations.len(), 5);
    drop(state);
    queue.close().unwrap();
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
    assert!(kernel.state.lock().unwrap().allocations.is_empty());
}

#[test]
fn bad_arguments_do_not_submit_poison_or_mutate_guarded_storage() {
    let Fixture {
        device,
        kernel,
        budget,
        ..
    } = fixture(Fault::None);
    let mut queue = open_queue(device, &budget).unwrap();
    let before = kernel.events();
    for result in [
        queue.write(3, 0, &[1]),
        queue.read(usize::MAX, 0, &mut [0]),
        queue.write(0, u64::MAX, &[1]),
        queue.write(0, TEST_SIZES[0] as u64, &[1]),
    ] {
        assert!(result.is_err());
    }
    for (lengths, scalars, groups) in [
        ([0, 19, 32], [1, 1], 1),
        ([18, 19, 32], [1, 1], 1),
        (TEST_SIZES, [0, 1], 1),
        (TEST_SIZES, [1, 65537], 1),
        (TEST_SIZES, [1, 1], 0),
        (TEST_SIZES, [1, 1], 65),
    ] {
        assert!(unsafe { queue.dispatch(lengths, scalars, groups) }.is_err());
    }
    assert!(!queue.is_poisoned());
    assert_eq!(kernel.events(), before);
    dispatch(&mut queue).unwrap();
    queue.close().unwrap();
}

#[test]
fn consecutive_dispatches_refresh_descriptor_lengths_and_scalar_arguments() {
    let Fixture {
        device,
        kernel,
        budget,
        ..
    } = fixture(Fault::None);
    let mut queue = open_queue(device, &budget).unwrap();
    for (lengths, scalars, input) in [([7, 9, 4], [7, 1], 0x24), ([17, 19, 16], [9, 4], 0x78)] {
        queue.write(1, 0, &[input; TEST_SIZES[1] as usize]).unwrap();
        queue.write(2, 0, &[0xfe; TEST_SIZES[2] as usize]).unwrap();
        unsafe { queue.dispatch(lengths, scalars, 1) }.unwrap();
        let mut output = [0; TEST_SIZES[2] as usize];
        queue.read(2, 0, &mut output).unwrap();
        assert!(output[..lengths[2] as usize]
            .iter()
            .all(|byte| *byte == input));
        assert!(output[lengths[2] as usize..]
            .iter()
            .all(|byte| *byte == 0xfe));
    }
    let state = kernel.state.lock().unwrap();
    assert_eq!(state.descriptor_lengths, [[7, 9, 4], [17, 19, 16]]);
    assert_eq!(state.scalar_pairs, [[7, 1], [9, 4]]);
    drop(state);
    queue.close().unwrap();
}

#[test]
fn large_slot_preserves_page_rounding_guards_bounds_and_owned_lifetime() {
    const LARGE_SIZE: u32 = 2 * 1024 * 1024 + 17;
    const LARGE_ALLOCATION: u64 = 2 * 1024 * 1024 + 3 * TEST_PAGE;
    const CHARGE: u64 = LARGE_ALLOCATION + 8 * TEST_PAGE;
    for explicit_close in [false, true] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture_with_budget(Fault::None, CHARGE);
        // Private mock allocations only; this test does not submit GPU work.
        let mut queue = unsafe {
            ComputeQueue::from_device(
                device,
                &budget,
                &probe_packets::shader_bytes(),
                [LARGE_SIZE, 19, 32],
            )
        }
        .unwrap();
        assert_eq!(queue.allocation_bytes(), CHARGE);
        assert_eq!(budget.current_bytes(), CHARGE);
        assert_eq!(budget.remaining(), 0);
        assert!(owner.upgrade().is_some());
        {
            let state = kernel.state.lock().unwrap();
            for (_, allocation) in state.allocations.range(3..) {
                // Inspect the complete idle mock mapping without copying it:
                // chunked initialization must cover payload, padding and guards.
                let storage = unsafe {
                    std::slice::from_raw_parts(allocation.pointer as *const u8, allocation.size)
                };
                assert!(storage.iter().all(|byte| *byte == CANARY));
            }
        }

        let tail = u64::from(LARGE_SIZE) - 4;
        queue.write(0, tail, &[1, 2, 3, 4]).unwrap();
        let mut bytes = [0; 4];
        queue.read(0, tail, &mut bytes).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        for result in [
            queue.write(0, tail + 1, &[5, 6, 7, 8]),
            queue.read(0, tail + 1, &mut bytes),
        ] {
            assert!(matches!(
                result,
                Err(ProbeError::Transport(Error::OutOfBounds))
            ));
        }
        {
            let state = kernel.state.lock().unwrap();
            assert_eq!(state.allocations.len(), 5);
            let allocation = &state.allocations[&3];
            assert_eq!(allocation.size as u64, LARGE_ALLOCATION);
            // Only the injected driver's idle anonymous mapping is inspected.
            let storage = unsafe {
                std::slice::from_raw_parts(allocation.pointer as *const u8, allocation.size)
            };
            assert!(storage[..TEST_PAGE as usize]
                .iter()
                .all(|byte| *byte == CANARY));
            assert!(storage[TEST_PAGE as usize + LARGE_SIZE as usize..]
                .iter()
                .all(|byte| *byte == CANARY));
            assert_eq!(state.submits, 0);
        }
        if explicit_close {
            queue.close().unwrap();
        } else {
            drop(queue);
        }
        assert_eq!(budget.current_bytes(), 0);
        assert!(kernel.state.lock().unwrap().allocations.is_empty());
        assert!(owner.upgrade().is_none());
    }
}

#[test]
fn insufficient_budget_refuses_large_gem_before_allocation_and_releases_prior_buffers() {
    for (limit, admitted, requested) in [
        (0, 0, TEST_PAGE),
        (64 * 1024, 2, 2 * 1024 * 1024 + 3 * TEST_PAGE),
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture_with_budget(Fault::None, limit);
        // Query/context creation may occur, but each GEM needs its own reservation.
        let result = unsafe {
            ComputeQueue::from_device(
                device,
                &budget,
                &probe_packets::shader_bytes(),
                [2 * 1024 * 1024 + 17, 19, 32],
            )
        };
        assert!(matches!(
            result,
            Err(ProbeError::Transport(Error::Budget(
                crate::budget::BudgetError::LimitExceeded {
                    limit_bytes,
                    current_bytes,
                    requested_bytes,
                }
            ))) if limit_bytes == limit
                && current_bytes == admitted as u64 * TEST_PAGE
                && requested_bytes == requested
        ));
        let state = kernel.state.lock().unwrap();
        assert_eq!(
            state
                .ordinary
                .iter()
                .filter(|request| **request == uapi::GEM_CREATE)
                .count(),
            admitted
        );
        assert!(state.allocations.is_empty());
        assert_eq!(state.submits, 0);
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
    }
}

#[test]
fn large_slot_budget_refuses_before_large_gem_without_large_host_allocation() {
    for capacity in [
        165_306_368,
        633_233_408,
        1 << 31,
        MAX_SLOT_BYTES - 1,
        MAX_SLOT_BYTES,
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture_with_budget(Fault::None, 64 * 1024);
        kernel.state.lock().unwrap().virtual_address_max = Some(1 << 40);
        let result = unsafe {
            ComputeQueue::from_device(
                device,
                &budget,
                &probe_packets::shader_bytes(),
                [capacity, 19, 32],
            )
        };
        let requested = u64::from(capacity).div_ceil(TEST_PAGE) * TEST_PAGE + 2 * TEST_PAGE;
        assert!(matches!(result, Err(ProbeError::Transport(Error::Budget(
            crate::budget::BudgetError::LimitExceeded { limit_bytes: 65536, current_bytes, requested_bytes }
        ))) if current_bytes == 2 * TEST_PAGE && requested_bytes == requested));
        let state = kernel.state.lock().unwrap();
        assert_eq!(
            state
                .ordinary
                .iter()
                .filter(|request| **request == uapi::GEM_CREATE)
                .count(),
            2
        );
        assert!(state.allocations.is_empty());
        assert_eq!(state.submits, 0);
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
    }
}

#[test]
fn maximum_slot_payload_and_guard_ranges_cross_u32_without_wrapping() {
    assert_eq!(MAX_SLOT_BYTES, u32::MAX - 4095);
    assert_eq!(
        guarded_slot_bytes(MAX_SLOT_BYTES),
        (1_u64 << 32) + TEST_PAGE
    );
    assert_eq!(
        guarded_slot_bytes(MAX_SLOT_BYTES - 1),
        guarded_slot_bytes(MAX_SLOT_BYTES)
    );
    assert_eq!(guarded_slot_bytes(MAX_SLOT_BYTES - 4096), 1_u64 << 32);
    assert_eq!(guarded_slot_bytes(1), 3 * TEST_PAGE);

    // Check the actual payload admission arithmetic without backing these
    // descriptor sizes with a host allocation or touching a GPU.
    let size = u64::from(MAX_SLOT_BYTES);
    assert_eq!(
        payload_range(MAX_SLOT_BYTES, 0, MAX_SLOT_BYTES as usize).unwrap(),
        TEST_PAGE
    );
    assert_eq!(
        payload_range(MAX_SLOT_BYTES, size - 4, 4).unwrap(),
        (1_u64 << 32) - 4
    );
    assert_eq!(payload_range(MAX_SLOT_BYTES, size, 0).unwrap(), 1_u64 << 32);
    for (offset, length) in [(size - 3, 4), (size, 1), (size + 1, 0), (u64::MAX, 1)] {
        assert!(payload_range(MAX_SLOT_BYTES, offset, length).is_err());
    }
}

#[test]
fn large_slot_address_plan_counts_guards_without_truncation_or_overlap() {
    let mut details = uapi::DeviceInfo {
        virtual_address_offset: 0xffff_0000,
        virtual_address_max: 1 << 40,
        virtual_address_alignment: 4096,
        ..Default::default()
    };
    let ip = abi::HwIpInfo {
        ib_start_alignment: 256,
        ib_size_alignment: 32,
        ..Default::default()
    };
    let guarded = u64::from(MAX_SLOT_BYTES) + 2 * TEST_PAGE;
    let sizes = [2 * TEST_PAGE, TEST_PAGE, guarded, guarded, guarded];
    let addresses = plan_addresses(&details, &ip, sizes).unwrap();
    assert_eq!(addresses[0], details.virtual_address_offset);
    for index in 1..addresses.len() {
        assert_eq!(addresses[index], addresses[index - 1] + sizes[index - 1]);
        assert_eq!(addresses[index] % TEST_PAGE, 0);
    }
    assert!(addresses[3] > u64::from(u32::MAX));
    let end = addresses[4] + sizes[4];
    details.virtual_address_max = end;
    assert!(plan_addresses(&details, &ip, sizes).is_ok());
    details.virtual_address_max = end - 1;
    assert!(plan_addresses(&details, &ip, sizes).is_err());
    details.virtual_address_max = u64::MAX;
    details.virtual_address_offset = (1 << 48) - TEST_PAGE;
    assert!(plan_addresses(&details, &ip, sizes).is_err());
}

#[test]
fn invalid_constructor_inputs_refuse_before_queries_context_or_allocations() {
    for (code, sizes) in [
        (Vec::new(), TEST_SIZES),
        (vec![0; 3], TEST_SIZES),
        (vec![0; 4100], TEST_SIZES),
        (vec![0; 4], [0, 1, 1]),
        (vec![0; 4], [MAX_SLOT_BYTES + 1, 1, 1]),
        (vec![0; 4], [1, MAX_SLOT_BYTES + 1, 1]),
        (vec![0; 4], [1, 1, MAX_SLOT_BYTES + 1]),
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(Fault::None);
        let before = {
            let state = kernel.state.lock().unwrap();
            (state.once.clone(), state.ordinary.clone())
        };
        assert!(unsafe { ComputeQueue::from_device(device, &budget, &code, sizes) }.is_err());
        let state = kernel.state.lock().unwrap();
        assert_eq!((state.once.clone(), state.ordinary.clone()), before);
        assert!(state.events.is_empty());
        assert!(state.allocations.is_empty());
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
    }
}

#[test]
fn diagnostic_wait_failures_still_poison_and_never_retire_pending_resources() {
    for fault in [Fault::WaitBusy, Fault::WaitInterrupted, Fault::WaitLost] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(Fault::None);
        let mut queue = open_queue(device, &budget).unwrap();
        queue.trace_submissions = true;
        kernel.state.lock().unwrap().allow_pending_diagnostic_query = true;
        kernel.set_fault(fault);
        let before = kernel.events().len();
        assert!(dispatch(&mut queue).is_err());
        let events = kernel.events();
        assert_eq!(&events[before..], ["submit", "wait", "context-query"]);
        assert_poisoned(&mut queue);
        assert!(queue.close().is_err());
        assert_eq!(kernel.events(), events);
        assert_quarantined(&owner, &budget, TEST_CHARGE);
    }
}

#[test]
fn uncertain_dispatch_poison_is_persistent_and_has_no_replay_or_early_cleanup() {
    for fault in [
        Fault::SubmitInterrupted,
        Fault::ZeroSequence,
        Fault::LatestSequence,
        Fault::WaitBusy,
        Fault::WaitInterrupted,
        Fault::WaitLost,
        Fault::ContextQuery,
        Fault::ContextFlags,
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(Fault::None);
        let mut queue = open_queue(device, &budget).unwrap();
        kernel.set_fault(fault);
        assert!(dispatch(&mut queue).is_err(), "{fault:?}");
        let after_failure = kernel.events();
        assert_poisoned(&mut queue);
        assert_eq!(kernel.events(), after_failure);
        assert!(queue.close().is_err());
        assert_eq!(kernel.events(), after_failure);
        assert_quarantined(&owner, &budget, TEST_CHARGE);
    }
}

#[test]
fn corruption_of_front_guard_alignment_padding_or_back_guard_poisons_queue() {
    for fault in [Fault::FrontGuard, Fault::Padding, Fault::RearGuard] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(Fault::None);
        let mut queue = open_queue(device, &budget).unwrap();
        kernel.set_fault(fault);
        let expected_offset = match fault {
            Fault::FrontGuard => 0,
            Fault::Padding => TEST_PAGE + u64::from(TEST_SIZES[2]),
            _ => 3 * TEST_PAGE - 1,
        };
        assert!(matches!(
            dispatch(&mut queue),
            Err(ProbeError::GuardCorruption { slot: 2, offset, actual: 0xa4 }) if offset == expected_offset
        ));
        assert_poisoned(&mut queue);
        let events = kernel.events();
        drop(queue);
        assert_eq!(kernel.events(), events);
        assert_quarantined(&owner, &budget, TEST_CHARGE);
    }
}

#[test]
fn caught_submission_panic_cannot_make_queue_reusable() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let mut queue = open_queue(device, &budget).unwrap();
    kernel.set_fault(Fault::SubmitPanic);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| dispatch(&mut queue))).is_err()
    );
    assert_poisoned(&mut queue);
    let events = kernel.events();
    drop(queue);
    assert_eq!(kernel.events(), events);
    assert_quarantined(&owner, &budget, TEST_CHARGE);
}

#[test]
fn definite_partial_initialization_failure_returns_all_reservations() {
    for fault in [Fault::AllocateThird, Fault::HostMapThird] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(open_queue(device, &budget).is_err());
        assert!(kernel.events().contains(&"allocate"));
        assert!(!kernel.events().contains(&"submit"));
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
    }
}

#[test]
fn uncertain_initialization_retains_every_allocated_resource() {
    for (fault, charge) in [
        (Fault::VaMapThird, 5 * TEST_PAGE),
        (Fault::ListCreate, TEST_CHARGE),
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(open_queue(device, &budget).is_err());
        assert!(!kernel.events().contains(&"submit"));
        assert_quarantined(&owner, &budget, charge);
    }
}

#[test]
fn cleanup_failure_stops_once_and_retains_unreleased_budget() {
    for fault in [
        Fault::ListDestroy,
        Fault::VaUnmap,
        Fault::ContextFree,
        Fault::GemClose,
        Fault::HostUnmap,
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(Fault::None);
        let mut queue = open_queue(device, &budget).unwrap();
        dispatch(&mut queue).unwrap();
        kernel.set_fault(fault);
        assert!(queue.close().is_err());
        let events = kernel.events();
        let expected = match fault {
            Fault::ListDestroy => "list-destroy",
            Fault::VaUnmap => "va-unmap",
            Fault::ContextFree => "context-free",
            Fault::GemClose => "gem-close",
            Fault::HostUnmap => "host-unmap",
            _ => unreachable!(),
        };
        assert_eq!(events.last(), Some(&expected));
        assert_eq!(events.iter().filter(|event| **event == expected).count(), 1);
        assert_quarantined(&owner, &budget, TEST_CHARGE);
    }
}
