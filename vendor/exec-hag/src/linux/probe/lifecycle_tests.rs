// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

use super::*;
use crate::linux::{DeviceInner, Kernel};
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::os::fd::RawFd;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Fault {
    #[default]
    None,
    AllocateSecond,
    MapSecond,
    VaMapSecond,
    ListCreate,
    SubmitInterrupted,
    SubmitPanic,
    ZeroSequence,
    LatestSequence,
    WaitBusy,
    WaitInterrupted,
    WaitDeviceLost,
    ContextState,
    ContextQuery,
    PayloadMismatch,
    GuardCorruption,
    ListDestroy,
    VaUnmap,
    ContextFree,
    GemClose,
    HostUnmap,
}

#[derive(Default)]
struct Allocation {
    size: u64,
    pointer: usize,
    address: u64,
}

#[derive(Default)]
struct Calls {
    events: Vec<&'static str>,
    once: Vec<libc::c_ulong>,
    ordinary: Vec<libc::c_ulong>,
    allocations: BTreeMap<u32, Allocation>,
    next_handle: u32,
    submitted: bool,
    completed: bool,
}

struct MockKernel {
    fault: Fault,
    calls: Mutex<Calls>,
    budget: MemoryBudget,
}

impl MockKernel {
    fn new(fault: Fault, budget: MemoryBudget) -> Self {
        Self {
            fault,
            calls: Mutex::new(Calls::default()),
            budget,
        }
    }

    fn events(&self) -> Vec<&'static str> {
        self.calls.lock().unwrap().events.clone()
    }

    fn fail<T>(&self) -> io::Result<T> {
        Err(io::Error::from_raw_os_error(libc::EIO))
    }

    unsafe fn dispatch(
        &self,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        let mut calls = self.calls.lock().unwrap();
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
                        assert_eq!(value.return_size as usize, size_of::<uapi::DeviceInfo>());
                        let output = unsafe {
                            &mut *(value.return_pointer as usize as *mut uapi::DeviceInfo)
                        };
                        output.device_id = 0x1586;
                        output.family = 150;
                        output.virtual_address_offset = 0x10000;
                        output.virtual_address_max = 0x1000000;
                        output.virtual_address_alignment = PAGE as u32;
                    }
                    abi::INFO_HW_IP_INFO => {
                        assert_eq!(value.query_data, [abi::HW_IP_COMPUTE, 0, 0, 0]);
                        assert_eq!(value.return_size as usize, size_of::<abi::HwIpInfo>());
                        let output =
                            unsafe { &mut *(value.return_pointer as usize as *mut abi::HwIpInfo) };
                        output.hw_ip_version_major = 11;
                        output.hw_ip_version_minor = 0;
                        output.ip_discovery_version = 0x0b0501;
                        output.ib_start_alignment = 256;
                        output.ib_size_alignment = 32;
                        output.available_rings = 1;
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
                        calls.events.push("context-create");
                        assert_eq!(input.ctx_id, 0);
                        assert_eq!(input.priority, abi::CTX_PRIORITY_NORMAL);
                        value.allocated = abi::ContextAllocOut { ctx_id: 3, pad: 0 };
                    }
                    abi::CTX_QUERY_STATE2 => {
                        calls.events.push("context-query");
                        assert!(calls.completed);
                        assert_eq!(input.ctx_id, 3);
                        if self.fault == Fault::ContextQuery {
                            return self.fail();
                        }
                        value.state = abi::ContextStateOut {
                            flags: if self.fault == Fault::ContextState {
                                1
                            } else {
                                0
                            },
                            ..Default::default()
                        };
                    }
                    abi::CTX_FREE => {
                        calls.events.push("context-free");
                        assert_eq!(input.ctx_id, 3);
                        assert!(!calls.submitted || calls.completed);
                        if self.fault == Fault::ContextFree {
                            return self.fail();
                        }
                    }
                    op => panic!("unexpected context operation {op}"),
                }
            }
            uapi::GEM_CREATE => {
                let value = unsafe { &mut *argument.cast::<uapi::GemCreate>() };
                let input = unsafe { value.input };
                calls.events.push("allocate");
                calls.next_handle += 1;
                let handle = calls.next_handle;
                if handle == 2 && self.fault == Fault::AllocateSecond {
                    return self.fail();
                }
                assert_eq!(input.domains, uapi::DOMAIN_GTT);
                assert_eq!(input.alignment, PAGE);
                assert!(self.budget.current_bytes() >= input.bo_size);
                calls.allocations.insert(
                    handle,
                    Allocation {
                        size: input.bo_size,
                        ..Default::default()
                    },
                );
                value.output = uapi::GemCreateOut { handle, pad: 0 };
            }
            uapi::GEM_MMAP => {
                let value = unsafe { &mut *argument.cast::<uapi::GemMmap>() };
                let handle = unsafe { value.input.handle };
                assert!(calls.allocations.contains_key(&handle));
                value.offset = u64::from(handle) * PAGE;
            }
            abi::GEM_VA => {
                let value = unsafe { &*argument.cast::<abi::GemVa>() };
                assert_eq!(value.pad, 0);
                assert_eq!(value.offset_in_bo, 0);
                assert_eq!(value.va_address % PAGE, 0);
                match value.operation {
                    abi::VA_MAP => {
                        calls.events.push("va-map");
                        if value.handle == 2 && self.fault == Fault::VaMapSecond {
                            return self.fail();
                        }
                        let allocation = calls.allocations.get_mut(&value.handle).unwrap();
                        assert_eq!(value.map_size, allocation.size);
                        allocation.address = value.va_address;
                    }
                    abi::VA_UNMAP => {
                        calls.events.push("va-unmap");
                        assert!(!calls.submitted || calls.completed);
                        if self.fault == Fault::VaUnmap {
                            return self.fail();
                        }
                        let allocation = calls.allocations.get_mut(&value.handle).unwrap();
                        assert_eq!(value.va_address, allocation.address);
                        assert_eq!(value.map_size, allocation.size);
                        allocation.address = 0;
                    }
                    op => panic!("unexpected VA operation {op}"),
                }
            }
            abi::BO_LIST => {
                let value = unsafe { &mut *argument.cast::<abi::BoList>() };
                let input = unsafe { value.input };
                match input.operation {
                    abi::BO_LIST_CREATE => {
                        calls.events.push("list-create");
                        assert_eq!(input.list_handle, 0);
                        assert_eq!(input.bo_number, 3);
                        assert_eq!(input.bo_info_size as usize, size_of::<abi::BoListEntry>());
                        let entries = unsafe {
                            std::slice::from_raw_parts(
                                input.bo_info_ptr as usize as *const abi::BoListEntry,
                                3,
                            )
                        };
                        assert_eq!(
                            entries
                                .iter()
                                .map(|entry| entry.bo_handle)
                                .collect::<Vec<_>>(),
                            [1, 2, 3]
                        );
                        if self.fault == Fault::ListCreate {
                            return self.fail();
                        }
                        value.output = abi::BoListOut {
                            list_handle: 4,
                            pad: 0,
                        };
                    }
                    abi::BO_LIST_DESTROY => {
                        calls.events.push("list-destroy");
                        assert_eq!(input.list_handle, 4);
                        assert!(!calls.submitted || calls.completed);
                        if self.fault == Fault::ListDestroy {
                            return self.fail();
                        }
                    }
                    op => panic!("unexpected list operation {op}"),
                }
            }
            abi::CS => {
                calls.events.push("submit");
                assert!(!calls.submitted, "proof must never submit twice");
                calls.submitted = true;
                let value = unsafe { &mut *argument.cast::<abi::Cs>() };
                let input = unsafe { value.input };
                assert_eq!(input.ctx_id, 3);
                assert_eq!(input.bo_list_handle, 4);
                assert_eq!(input.num_chunks, 1);
                assert_eq!(input.flags, 0);
                let chunk_pointer = unsafe { *(input.chunks as usize as *const u64) };
                let chunk = unsafe { &*(chunk_pointer as usize as *const abi::CsChunk) };
                assert_eq!(chunk.chunk_id, abi::CHUNK_ID_IB);
                assert_eq!(chunk.length_dw as usize * 4, size_of::<abi::CsChunkIb>());
                let ib = unsafe { &*(chunk.chunk_data as usize as *const abi::CsChunkIb) };
                assert_eq!(ib.ip_type, abi::HW_IP_COMPUTE);
                assert_eq!(ib.ip_instance, 0);
                assert_eq!(ib.ring, 0);
                assert_eq!(ib.va_start, calls.allocations[&2].address);
                assert!(ib.ib_bytes > 0 && u64::from(ib.ib_bytes) <= PAGE);
                assert_eq!(ib.ib_bytes % 32, 0);
                let output = &calls.allocations[&3];
                assert_eq!(output.size, GUARDED_BYTES as u64);
                assert_ne!(output.pointer, 0, "map and initialize before GPU access");
                let initial = unsafe {
                    std::slice::from_raw_parts(output.pointer as *const u8, output.size as usize)
                };
                assert!(initial.iter().all(|byte| *byte == CANARY));
                assert_eq!(self.budget.current_bytes(), TOTAL_BYTES);
                if self.fault == Fault::SubmitPanic {
                    drop(calls);
                    panic!("injected panic at uncertain submission");
                }
                if self.fault == Fault::SubmitInterrupted {
                    return Err(io::Error::from_raw_os_error(libc::EINTR));
                }
                value.handle = match self.fault {
                    Fault::ZeroSequence => 0,
                    Fault::LatestSequence => u64::MAX,
                    _ => 17,
                };
            }
            abi::WAIT_CS => {
                calls.events.push("wait");
                let value = unsafe { &mut *argument.cast::<abi::WaitCs>() };
                let input = unsafe { value.input };
                assert_eq!(
                    input.handle, 17,
                    "reserved sequence values are not completion fences"
                );
                assert_eq!(input.ctx_id, 3);
                assert_eq!(input.ip_type, abi::HW_IP_COMPUTE);
                assert_eq!(input.ring, 0);
                assert!(input.timeout > 0 && input.timeout <= i64::MAX as u64);
                match self.fault {
                    Fault::WaitInterrupted => {
                        return Err(io::Error::from_raw_os_error(libc::EINTR))
                    }
                    Fault::WaitDeviceLost => {
                        return Err(io::Error::from_raw_os_error(libc::ENODEV))
                    }
                    Fault::WaitBusy => {
                        value.status = 1;
                        return Ok(());
                    }
                    _ => {}
                }
                calls.completed = true;
                let output = &calls.allocations[&3];
                let bytes = unsafe {
                    std::slice::from_raw_parts_mut(output.pointer as *mut u8, output.size as usize)
                };
                bytes[PAGE as usize..PAGE as usize + OUTPUT_BYTES].fill(0x22);
                if self.fault == Fault::PayloadMismatch {
                    bytes[PAGE as usize] = 0;
                }
                if self.fault == Fault::GuardCorruption {
                    bytes[0] = 0;
                }
                value.status = 0;
            }
            uapi::GEM_CLOSE => {
                calls.events.push("gem-close");
                assert!(!calls.submitted || calls.completed);
                if self.fault == Fault::GemClose {
                    return self.fail();
                }
                let value = unsafe { &*argument.cast::<uapi::GemClose>() };
                let allocation = calls.allocations.remove(&value.handle).unwrap();
                assert_eq!(allocation.pointer, 0, "host mapping must close first");
                assert_eq!(allocation.address, 0, "GPU VA must close first");
            }
            _ => panic!("unexpected request {request:x}"),
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
        self.calls.lock().unwrap().once.push(request);
        unsafe { self.dispatch(request, argument) }
    }

    unsafe fn ioctl(
        &self,
        _fd: RawFd,
        request: libc::c_ulong,
        argument: *mut libc::c_void,
    ) -> io::Result<()> {
        self.calls.lock().unwrap().ordinary.push(request);
        unsafe { self.dispatch(request, argument) }
    }

    unsafe fn map(&self, _fd: RawFd, length: usize, offset: libc::off_t) -> io::Result<*mut u8> {
        let mut calls = self.calls.lock().unwrap();
        calls.events.push("host-map");
        let handle = (offset as u64 / PAGE) as u32;
        if handle == 2 && self.fault == Fault::MapSecond {
            return self.fail();
        }
        let allocation = calls.allocations.get_mut(&handle).unwrap();
        assert_eq!(length as u64, allocation.size);
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
        // Dirty host storage catches reliance on anonymous mapping initialization.
        unsafe { std::ptr::write_bytes(pointer.cast::<u8>(), 0x5c, length) };
        Ok(pointer.cast())
    }

    unsafe fn unmap(&self, pointer: *mut u8, length: usize) -> io::Result<()> {
        let mut calls = self.calls.lock().unwrap();
        calls.events.push("host-unmap");
        assert!(!calls.submitted || calls.completed);
        if self.fault == Fault::HostUnmap {
            return self.fail();
        }
        let allocation = calls
            .allocations
            .values_mut()
            .find(|allocation| allocation.pointer == pointer as usize)
            .unwrap();
        assert_eq!(allocation.size as usize, length);
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
    let budget = MemoryBudget::new(TOTAL_BYTES);
    let kernel = Arc::new(MockKernel::new(fault, budget.clone()));
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

fn assert_quarantined(owner: &Weak<DeviceInner>, budget: &MemoryBudget, charged: u64) {
    let retained = owner
        .upgrade()
        .expect("quarantine must retain owning render FD");
    assert!(retained.cleanup_failures.load(Ordering::Relaxed) > 0);
    assert_ne!(
        unsafe { libc::fcntl(retained.fd.as_raw_fd(), libc::F_GETFD) },
        -1
    );
    assert_eq!(budget.current_bytes(), charged);
}

#[test]
fn successful_proof_has_one_submission_exact_output_and_ordered_cleanup() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    let result = run_device(device, &budget, Instant::now()).unwrap();
    assert_eq!(result.sequence, 17);
    assert_eq!(result.verified_bytes, OUTPUT_BYTES);
    assert_eq!(result.guard_bytes, 2 * PAGE as usize);
    assert_eq!(result.allocation_bytes, TOTAL_BYTES);
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
    let calls = kernel.calls.lock().unwrap();
    assert_eq!(
        calls
            .once
            .iter()
            .filter(|request| **request == abi::CS)
            .count(),
        1
    );
    assert!(!calls.ordinary.contains(&abi::CS));
    assert!(!calls.ordinary.contains(&abi::WAIT_CS));
    assert!(calls.allocations.is_empty());
    assert_eq!(
        &calls.events[calls.events.len() - 11..],
        [
            "list-destroy",
            "va-unmap",
            "va-unmap",
            "va-unmap",
            "context-free",
            "host-unmap",
            "gem-close",
            "host-unmap",
            "gem-close",
            "host-unmap",
            "gem-close",
        ]
    );
}

#[test]
fn interrupted_submission_is_not_replayed_and_retains_every_resource() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::SubmitInterrupted);
    let error = run_device(device, &budget, Instant::now()).unwrap_err();
    assert!(
        matches!(error, ProbeError::Transport(Error::Io { source, .. }) if source.raw_os_error() == Some(libc::EINTR))
    );
    let events = kernel.events();
    assert_eq!(events.iter().filter(|event| **event == "submit").count(), 1);
    assert_eq!(events.last(), Some(&"submit"));
    assert_quarantined(&owner, &budget, TOTAL_BYTES);
}

#[test]
fn reserved_sequence_values_never_reach_the_wait_ioctl() {
    for fault in [Fault::ZeroSequence, Fault::LatestSequence] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(run_device(device, &budget, Instant::now()).is_err());
        assert_eq!(kernel.events().last(), Some(&"submit"));
        assert_quarantined(&owner, &budget, TOTAL_BYTES);
    }
}

#[test]
fn pending_interrupted_and_lost_device_waits_do_not_read_or_release_buffers() {
    for fault in [
        Fault::WaitBusy,
        Fault::WaitInterrupted,
        Fault::WaitDeviceLost,
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(run_device(device, &budget, Instant::now()).is_err());
        assert_eq!(kernel.events().last(), Some(&"wait"));
        assert_quarantined(&owner, &budget, TOTAL_BYTES);
    }
}

#[test]
fn context_health_failure_after_fence_is_not_successful_compute() {
    for fault in [Fault::ContextState, Fault::ContextQuery] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        let error = run_device(device, &budget, Instant::now()).unwrap_err();
        if fault == Fault::ContextState {
            assert!(matches!(error, ProbeError::ContextFault(1)));
        }
        assert_eq!(kernel.events().last(), Some(&"context-query"));
        assert_quarantined(&owner, &budget, TOTAL_BYTES);
    }
}

#[test]
fn completed_but_wrong_payload_or_guard_fails_reference_and_cleans_up() {
    for fault in [Fault::PayloadMismatch, Fault::GuardCorruption] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(matches!(
            run_device(device, &budget, Instant::now()),
            Err(ProbeError::Mismatch { .. })
        ));
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
        assert!(kernel.calls.lock().unwrap().allocations.is_empty());
    }
}

#[test]
fn definite_allocation_or_mapping_failure_before_submission_unwinds_partial_resources() {
    for fault in [Fault::AllocateSecond, Fault::MapSecond] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(matches!(
            run_device(device, &budget, Instant::now()),
            Err(ProbeError::Transport(Error::Io { source, .. }))
                if source.raw_os_error() == Some(libc::EIO)
        ));
        let events = kernel.events();
        assert_eq!(
            events.iter().filter(|event| **event == "allocate").count(),
            2
        );
        assert!(!events.contains(&"submit"));
        assert_eq!(budget.current_bytes(), 0);
        assert!(owner.upgrade().is_none());
        assert!(kernel.calls.lock().unwrap().allocations.is_empty());
    }
}

#[test]
fn unsupported_identity_refuses_before_creating_context_or_allocations() {
    let Fixture {
        mut device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::None);
    device.info.device_id = 0xffff;
    assert!(matches!(
        run_device(device, &budget, Instant::now()),
        Err(ProbeError::Unsupported(_))
    ));
    assert!(kernel.events().is_empty());
    assert_eq!(budget.current_bytes(), 0);
    assert!(owner.upgrade().is_none());
}

#[test]
fn uncertain_va_and_list_creation_preserve_all_existing_reservations() {
    for (fault, charged, final_event) in [
        (Fault::VaMapSecond, 2 * PAGE, "va-map"),
        (Fault::ListCreate, TOTAL_BYTES, "list-create"),
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(run_device(device, &budget, Instant::now()).is_err());
        assert_eq!(kernel.events().last(), Some(&final_event));
        assert!(!kernel.events().contains(&"submit"));
        assert_quarantined(&owner, &budget, charged);
    }
}

#[test]
fn cleanup_failures_stop_without_retry_or_releasing_live_storage_quota() {
    for (fault, final_event) in [
        (Fault::ListDestroy, "list-destroy"),
        (Fault::VaUnmap, "va-unmap"),
        (Fault::ContextFree, "context-free"),
        (Fault::GemClose, "gem-close"),
        (Fault::HostUnmap, "host-unmap"),
    ] {
        let Fixture {
            device,
            owner,
            kernel,
            budget,
        } = fixture(fault);
        assert!(run_device(device, &budget, Instant::now()).is_err());
        let events = kernel.events();
        assert_eq!(events.last(), Some(&final_event));
        assert_eq!(
            events.iter().filter(|event| **event == final_event).count(),
            1
        );
        assert_quarantined(&owner, &budget, TOTAL_BYTES);
    }
}

#[test]
fn panic_during_submission_quarantines_without_destructor_gpu_operations() {
    let Fixture {
        device,
        owner,
        kernel,
        budget,
    } = fixture(Fault::SubmitPanic);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_device(device, &budget, Instant::now())
    }));
    assert!(result.is_err());
    assert_eq!(kernel.events().last(), Some(&"submit"));
    assert_quarantined(&owner, &budget, TOTAL_BYTES);
}
