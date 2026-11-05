// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

//! A C ABI over the experimental bounded compute queue.
//!
//! This lets non-Rust applications use HAG through an explicit C ABI. It wraps
//! [`ComputeQueue`](crate::linux::compute::ComputeQueue) and adds nothing: every
//! bound, refusal and poisoning rule of the underlying queue applies unchanged.
//!
//! Three properties this boundary must hold, because a C caller cannot be
//! trusted to hold them:
//!
//! 1. **No unwinding crosses the boundary.** Operations that can panic catch
//!    unwinding panics and convert them to [`HAG_PANIC`](crate::capi::HAG_PANIC).
//!    A surviving handle is poisoned; closing consumes a handle even on failure.
//! 2. **No ownership crosses the boundary.** The caller provides every buffer
//!    and never frees anything HAG allocated; the single opaque handle is
//!    released by [`hag_queue_close`](crate::capi::hag_queue_close) and nothing else.
//! 3. **Errors are codes, never messages the caller must free.**
//!    [`hag_error_text`](crate::capi::hag_error_text) returns a borrowed, static,
//!    NUL-terminated string.
//!
//! # Safety
//!
//! Opening a queue is unsafe for exactly the reason
//! [`ComputeQueue::open`](crate::linux::compute::ComputeQueue::open) is:
//! the program bytes are trusted, audited GFX1151 wave64 ISA. Reaching this
//! interface from another language does not relax that. Never pass untrusted
//! model or user bytes as shader code.
//!
//! Callers must serialize **every call on the same handle**, including queries
//! and close. There may be no overlapping calls or use after close. Distinct
//! handles may be used independently. Every pointer must be aligned for its
//! declared type; every array or byte region must lie within one live allocation
//! and remain valid for the entire call. Readable regions must not be modified
//! during a call, and writable regions must be exclusively accessible and must
//! not overlap other arguments or the opaque handle. Null pointers are accepted
//! only where documented or rejected as invalid arguments.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;

use crate::budget::MemoryBudget;
use crate::linux::compute::{
    BufferRegion, ComputeQueue, KernelAbi, DEFAULT_DISPATCH_GROUPS, GFX1151_SERIAL_ABI,
    MAX_DISPATCH_GROUPS, MAX_KERNEL_VGPRS, MAX_SLOT_BYTES, MIN_KERNEL_VGPRS,
};
use crate::linux::probe::ProbeError;
use crate::linux::Error;

// Mirror the transport's immutable-library limits before forming caller slices
// or allocating pointer metadata. Tests below check these against its public
// hardware-free geometry validator.
const MAX_C_PROGRAMS: u32 = 32;
const MAX_C_PROGRAM_BYTES: u32 = 4096;

/// The call succeeded.
pub const HAG_OK: c_int = 0;
/// A pointer argument was null, or a length was zero where it may not be.
pub const HAG_INVALID_ARGUMENT: c_int = -1;
/// The queue refused the request: out of bounds, wrong geometry, bad ABI.
pub const HAG_REFUSED: c_int = -2;
/// The device rejected the operation, or discovery failed.
pub const HAG_DEVICE: c_int = -3;
/// The queue is poisoned and every further call will refuse.
pub const HAG_POISONED: c_int = -4;
/// A panic was caught at the boundary. The handle is unusable.
pub const HAG_PANIC: c_int = -5;
/// The requested allocation exceeded the budget.
pub const HAG_BUDGET: c_int = -6;
/// A data guard or alignment padding was modified.
///
/// This is distinct from a generic device error on purpose: it means
/// something wrote outside the region it was given, which is precisely the
/// event a bounded-execution story exists to detect. The queue is poisoned.
pub const HAG_GUARD_CORRUPTION: c_int = -7;
/// Completion was not established by the finite fence deadline.
pub const HAG_INCOMPLETE: c_int = -8;

/// An opaque queue handle. Only [`hag_queue_close`] releases it.
#[allow(non_camel_case_types)] // Keep the opaque type's name consistent with the public C ABI.
pub struct hag_queue {
    queue: Option<ComputeQueue>,
    budget: MemoryBudget,
    poisoned: bool,
}

/// A borrowed, static description of a status code.
///
/// Never returns null and never allocates, so a caller has nothing to free.
#[no_mangle]
pub extern "C" fn hag_error_text(code: c_int) -> *const c_char {
    let text: &CStr = match code {
        HAG_OK => c"ok",
        HAG_INVALID_ARGUMENT => c"invalid argument",
        HAG_REFUSED => c"the queue refused this request",
        HAG_DEVICE => c"device error or unsupported hardware",
        HAG_POISONED => c"queue is poisoned and will refuse every further call",
        HAG_PANIC => c"a panic was caught at the C boundary; the handle is unusable",
        HAG_BUDGET => c"allocation exceeded the memory budget",
        HAG_GUARD_CORRUPTION => c"a data guard was modified; the queue is poisoned",
        HAG_INCOMPLETE => c"completion was not established by the fence deadline",
        _ => c"unknown status code",
    };
    text.as_ptr()
}

/// The VGPR allocation range [`hag_queue_open`] admits.
#[no_mangle]
pub extern "C" fn hag_min_kernel_vgprs() -> u32 {
    MIN_KERNEL_VGPRS
}

/// The largest workgroup count `hag_queue_open` will accept.
#[no_mangle]
pub extern "C" fn hag_max_dispatch_groups() -> u32 {
    MAX_DISPATCH_GROUPS
}

/// The workgroup count a queue uses when `max_groups` is zero.
#[no_mangle]
pub extern "C" fn hag_default_dispatch_groups() -> u32 {
    DEFAULT_DISPATCH_GROUPS
}

/// See [`hag_min_kernel_vgprs`].
#[no_mangle]
pub extern "C" fn hag_max_kernel_vgprs() -> u32 {
    MAX_KERNEL_VGPRS
}

fn guard<F: FnOnce() -> c_int>(body: F) -> c_int {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(code) => code,
        Err(_) => HAG_PANIC,
    }
}

fn guard_handle<F: FnOnce(&mut hag_queue) -> c_int>(handle: &mut hag_queue, body: F) -> c_int {
    if handle.poisoned {
        return HAG_POISONED;
    }
    match catch_unwind(AssertUnwindSafe(|| body(handle))) {
        Ok(code) => code,
        Err(_) => {
            // Refuse further operations without guessing whether hardware work
            // is pending. ComputeQueue retains its own retirement/quarantine
            // state, and hag_queue_close still consumes this handle once.
            handle.poisoned = true;
            HAG_PANIC
        }
    }
}

fn checked_program_count(count: u32) -> Result<usize, c_int> {
    if count == 0 {
        return Err(HAG_INVALID_ARGUMENT);
    }
    if count > MAX_C_PROGRAMS {
        return Err(HAG_REFUSED);
    }
    Ok(count as usize)
}

fn checked_program_length(length: u32) -> Result<usize, c_int> {
    if length == 0 {
        return Err(HAG_INVALID_ARGUMENT);
    }
    if length > MAX_C_PROGRAM_BYTES || length % 4 != 0 {
        return Err(HAG_REFUSED);
    }
    // The one-page bound fits usize and isize on every supported target.
    Ok(length as usize)
}

fn checked_transfer_length(length: u64) -> Result<usize, c_int> {
    if length == 0 {
        return Err(HAG_INVALID_ARGUMENT);
    }
    let length = usize::try_from(length).map_err(|_| HAG_REFUSED)?;
    if length > isize::MAX as usize || length as u64 > u64::from(MAX_SLOT_BYTES) {
        return Err(HAG_REFUSED);
    }
    Ok(length)
}

/// Open a bounded compute queue over an immutable program library.
///
/// `programs` is `program_count` pointers with matching `program_sizes`.
/// One through 32 programs are admitted, each containing a nonzero multiple
/// of four bytes no larger than 4096. Oversized counts are refused before any
/// pointer arrays are accessed; invalid program lengths before payload access.
/// `slot_sizes` gives the three data slots' logical capacities in bytes.
/// `vgprs` selects the per-wave register allocation: pass 8 for the qualified
/// ABI, or a larger multiple of four to derive a library-local ABI that leaves
/// every other library untouched. `max_groups` is the largest workgroup count
/// a dispatch on this queue may launch; zero means the qualified default of
/// 64, which suits a kernel covering at most 4096 elements and is far too low
/// for an image, since a dispatch costs about the same whatever it covers.
/// `budget_bytes` of zero means unlimited.
///
/// On success writes an owned handle to `out` and returns [`HAG_OK`]. On failure
/// `*out` is untouched; initialize it to null before calling if desired.
///
/// # Safety
/// The module's pointer, lifetime and serialization requirements apply.
/// `device_path` must be a valid NUL-terminated path. For an admitted count,
/// `programs` and `program_sizes` must contain that many readable entries;
/// `slot_sizes` must contain three. Each admitted program must be a readable
/// region of its stated length containing trusted, audited GFX1151 wave64 ISA
/// conforming to the queue's ABI. `out` must be writable and must not overlap
/// any input. Input storage may be released after this function returns.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_open(
    device_path: *const c_char,
    programs: *const *const u8,
    program_sizes: *const u32,
    program_count: u32,
    slot_sizes: *const u32,
    vgprs: u32,
    max_groups: u32,
    budget_bytes: u64,
    out: *mut *mut hag_queue,
) -> c_int {
    if device_path.is_null()
        || programs.is_null()
        || program_sizes.is_null()
        || slot_sizes.is_null()
        || out.is_null()
    {
        return HAG_INVALID_ARGUMENT;
    }
    let program_count = match checked_program_count(program_count) {
        Ok(count) => count,
        Err(code) => return code,
    };
    guard(|| {
        // SAFETY: the caller guarantees a NUL-terminated path.
        let path = match unsafe { CStr::from_ptr(device_path) }.to_str() {
            Ok(text) => PathBuf::from(text),
            Err(_) => return HAG_INVALID_ARGUMENT,
        };
        let requested_groups = if max_groups == 0 {
            DEFAULT_DISPATCH_GROUPS
        } else {
            max_groups
        };
        let abi: KernelAbi = match GFX1151_SERIAL_ABI
            .with_vgprs(vgprs)
            .and_then(|derived| derived.with_max_groups(requested_groups))
        {
            Some(derived) => derived,
            None => return HAG_INVALID_ARGUMENT,
        };

        // SAFETY: the caller guarantees program_count readable pairs.
        let sizes_slice = unsafe { std::slice::from_raw_parts(program_sizes, program_count) };
        let pointers = unsafe { std::slice::from_raw_parts(programs, program_count) };
        let mut library: Vec<&[u8]> = Vec::with_capacity(program_count);
        for (pointer, size) in pointers.iter().zip(sizes_slice) {
            if pointer.is_null() {
                return HAG_INVALID_ARGUMENT;
            }
            let size = match checked_program_length(*size) {
                Ok(size) => size,
                Err(code) => return code,
            };
            // SAFETY: the caller guarantees each region is readable for its size.
            library.push(unsafe { std::slice::from_raw_parts(*pointer, size) });
        }
        // SAFETY: three readable slot capacities.
        let slots = unsafe { std::slice::from_raw_parts(slot_sizes, 3) };
        let sizes = [slots[0], slots[1], slots[2]];

        let budget = if budget_bytes == 0 {
            MemoryBudget::unlimited()
        } else {
            MemoryBudget::new(budget_bytes)
        };

        // SAFETY: the trusted-program contract is the caller's, as documented.
        let opened =
            unsafe { ComputeQueue::open_programs_with_abi(&path, &budget, &library, sizes, abi) };
        match opened {
            Ok(queue) => {
                let handle = Box::new(hag_queue {
                    queue: Some(queue),
                    budget,
                    poisoned: false,
                });
                // SAFETY: the caller guarantees `out` is writable.
                unsafe { *out = Box::into_raw(handle) };
                HAG_OK
            }
            Err(error) => classify(&error),
        }
    })
}

fn classify(error: &ProbeError) -> c_int {
    // Each failure the caller can act on differently gets its own code. A
    // refusal means "you asked for something out of bounds"; a guard
    // corruption means something wrote where it must not, and is the one a
    // bounded-execution story most needs to surface rather than fold into a
    // generic device error.
    match error {
        ProbeError::Unsupported(_) => HAG_REFUSED,
        ProbeError::GuardCorruption { .. } => HAG_GUARD_CORRUPTION,
        ProbeError::Incomplete => HAG_INCOMPLETE,
        ProbeError::ContextFault(_) | ProbeError::Mismatch { .. } => HAG_DEVICE,
        // A transport failure is not automatically the device's fault. Most
        // of these are the caller asking for something the queue will not do,
        // and reporting them as HAG_DEVICE sends whoever is debugging to the
        // hardware when the fault is in their own arguments. Writing past a
        // slot's end was indistinguishable from a driver failure.
        ProbeError::Transport(inner) => match inner {
            Error::OutOfBounds | Error::NotHostVisible => HAG_REFUSED,
            Error::InvalidSize | Error::NotRenderNode | Error::WrongDriver => HAG_INVALID_ARGUMENT,
            Error::Budget(_) => HAG_BUDGET,
            Error::Quarantined => HAG_POISONED,
            Error::Io { .. } => HAG_DEVICE,
        },
    }
}

unsafe fn borrow<'a>(handle: *mut hag_queue) -> Result<&'a mut hag_queue, c_int> {
    if handle.is_null() {
        return Err(HAG_INVALID_ARGUMENT);
    }
    // SAFETY: the caller guarantees a handle from hag_queue_open that has not
    // been closed, with no overlapping calls on that handle.
    let handle = unsafe { &mut *handle };
    if handle.poisoned || handle.queue.is_none() {
        return Err(HAG_POISONED);
    }
    Ok(handle)
}

/// Copy `length` bytes into a data slot at `offset`.
///
/// Zero length is invalid. Lengths exceeding the slot limit or Rust's host
/// slice limits are refused before accessing `bytes` or the handle.
///
/// # Safety
/// The module's pointer, lifetime and serialization requirements apply.
/// For an admitted length, `handle` must be an open handle and `bytes` must be
/// readable for `length` bytes without overlapping the handle.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_write(
    handle: *mut hag_queue,
    slot: u32,
    offset: u64,
    bytes: *const u8,
    length: u64,
) -> c_int {
    if bytes.is_null() {
        return HAG_INVALID_ARGUMENT;
    }
    let length = match checked_transfer_length(length) {
        Ok(length) => length,
        Err(code) => return code,
    };
    // SAFETY: the caller guarantees the live, exclusively accessed handle.
    let handle = match unsafe { borrow(handle) } {
        Ok(value) => value,
        Err(code) => return code,
    };
    guard_handle(handle, |handle| {
        // SAFETY: the caller guarantees a readable region of `length`.
        let payload = unsafe { std::slice::from_raw_parts(bytes, length) };
        let queue = handle.queue.as_mut().expect("checked by borrow");
        match queue.write(slot as usize, offset, payload) {
            Ok(()) => HAG_OK,
            Err(error) => {
                handle.poisoned = queue.is_poisoned();
                classify(&error)
            }
        }
    })
}

/// Copy `length` bytes out of a data slot at `offset`.
///
/// Zero length is invalid. Lengths exceeding the slot limit or Rust's host
/// slice limits are refused before accessing `bytes` or the handle.
///
/// # Safety
/// The module's pointer, lifetime and serialization requirements apply.
/// For an admitted length, `handle` must be an open handle and `bytes` must be
/// writable for `length` bytes without overlapping the handle.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_read(
    handle: *mut hag_queue,
    slot: u32,
    offset: u64,
    bytes: *mut u8,
    length: u64,
) -> c_int {
    if bytes.is_null() {
        return HAG_INVALID_ARGUMENT;
    }
    let length = match checked_transfer_length(length) {
        Ok(length) => length,
        Err(code) => return code,
    };
    // SAFETY: the caller guarantees the live, exclusively accessed handle.
    let handle = match unsafe { borrow(handle) } {
        Ok(value) => value,
        Err(code) => return code,
    };
    guard_handle(handle, |handle| {
        // SAFETY: the caller guarantees a writable region of `length`.
        let destination = unsafe { std::slice::from_raw_parts_mut(bytes, length) };
        let queue = handle.queue.as_mut().expect("checked by borrow");
        match queue.read(slot as usize, offset, destination) {
            Ok(()) => HAG_OK,
            Err(error) => {
                handle.poisoned = queue.is_poisoned();
                classify(&error)
            }
        }
    })
}

/// Submit one program with explicit per-slot regions and wait for completion.
///
/// `offsets` and `lengths` are three entries each, describing the region of
/// every data slot this dispatch may address. `scalars` is two entries.
///
/// # Safety
/// The module's pointer, lifetime and serialization requirements apply.
/// `handle` must be open; `offsets`, `lengths` and `scalars` readable for
/// three `u64`, three `u32` and two `u32` entries respectively. The selected
/// trusted program must terminate and obey its descriptor bounds, alignment
/// and indexing requirements for every argument it is given.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_dispatch(
    handle: *mut hag_queue,
    program: u32,
    offsets: *const u64,
    lengths: *const u32,
    scalars: *const u32,
    groups: u32,
) -> c_int {
    if offsets.is_null() || lengths.is_null() || scalars.is_null() {
        return HAG_INVALID_ARGUMENT;
    }
    // SAFETY: the caller guarantees the live, exclusively accessed handle.
    let handle = match unsafe { borrow(handle) } {
        Ok(value) => value,
        Err(code) => return code,
    };
    guard_handle(handle, |handle| {
        // SAFETY: the caller guarantees the three/three/two readable entries.
        let offsets = unsafe { std::slice::from_raw_parts(offsets, 3) };
        let lengths = unsafe { std::slice::from_raw_parts(lengths, 3) };
        let scalars = unsafe { std::slice::from_raw_parts(scalars, 2) };
        let bindings: [BufferRegion; 3] = std::array::from_fn(|slot| BufferRegion {
            slot,
            offset: offsets[slot],
            length: lengths[slot],
        });
        let queue = handle.queue.as_mut().expect("checked by borrow");
        // SAFETY: the trusted-program contract is the caller's.
        let result = unsafe {
            queue.dispatch_regions(program as usize, bindings, [scalars[0], scalars[1]], groups)
        };
        match result {
            Ok(_) => HAG_OK,
            Err(error) => {
                handle.poisoned = queue.is_poisoned();
                classify(&error)
            }
        }
    })
}

/// Total page-rounded allocation charge, or zero for a null handle.
///
/// # Safety
/// `handle` must be an open handle or null. The module's pointer, lifetime and
/// serialization requirements apply even to this read-only query.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_allocation_bytes(handle: *const hag_queue) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: the caller guarantees an open handle.
    let handle = unsafe { &*handle };
    handle
        .queue
        .as_ref()
        .map_or(0, |queue| queue.allocation_bytes())
}

/// Nonzero when the queue is poisoned and will refuse every further call.
///
/// # Safety
/// `handle` must be an open handle or null. The module's pointer, lifetime and
/// serialization requirements apply even to this read-only query.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_is_poisoned(handle: *const hag_queue) -> c_int {
    if handle.is_null() {
        return 1;
    }
    // SAFETY: the caller guarantees an open handle.
    let handle = unsafe { &*handle };
    let poisoned = handle.poisoned || handle.queue.as_ref().is_none_or(ComputeQueue::is_poisoned);
    c_int::from(poisoned)
}

/// Remaining budget in bytes, or zero for a null handle.
/// An unlimited budget reports `u64::MAX` minus its live allocation charge.
///
/// # Safety
/// `handle` must be an open handle or null. The module's pointer, lifetime and
/// serialization requirements apply even to this read-only query.
#[no_mangle]
pub unsafe extern "C" fn hag_queue_budget_remaining(handle: *const hag_queue) -> u64 {
    if handle.is_null() {
        return 0;
    }
    // SAFETY: the caller guarantees an open handle.
    unsafe { &*handle }.budget.remaining()
}

/// Close the queue and release the handle. The handle must not be used again.
///
/// Returns [`HAG_OK`] when the queue shut down cleanly. A poisoned queue is
/// still released, and reports its status rather than succeeding silently.
///
/// # Safety
/// The module's pointer, lifetime and serialization requirements apply.
/// `handle` must come from [`hag_queue_open`] and must not be closed twice.
/// It is consumed even if this function returns an error or [`HAG_PANIC`].
#[no_mangle]
pub unsafe extern "C" fn hag_queue_close(handle: *mut hag_queue) -> c_int {
    if handle.is_null() {
        return HAG_INVALID_ARGUMENT;
    }
    guard(|| {
        // SAFETY: the caller guarantees a handle this function has not already
        // consumed.
        let mut owned = unsafe { Box::from_raw(handle) };
        let was_poisoned = owned.poisoned;
        match owned.queue.take() {
            Some(queue) => match queue.close() {
                Ok(()) if !was_poisoned => HAG_OK,
                Ok(()) => HAG_POISONED,
                Err(error) => classify(&error),
            },
            None => HAG_POISONED,
        }
    })
}

/// Null the caller's pointer is not this library's job; a handle is released
/// exactly once by [`hag_queue_close`] and must not be reused afterwards.
const _: () = {
    // Keep the handle pointer-sized and the module honest about ownership.
    assert!(std::mem::size_of::<*mut hag_queue>() == std::mem::size_of::<usize>());
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn error_text_is_static_and_never_null() {
        for code in [
            HAG_OK,
            HAG_INVALID_ARGUMENT,
            HAG_REFUSED,
            HAG_DEVICE,
            HAG_POISONED,
            HAG_PANIC,
            HAG_BUDGET,
            HAG_GUARD_CORRUPTION,
            HAG_INCOMPLETE,
            12345,
        ] {
            let text = hag_error_text(code);
            assert!(!text.is_null());
            // SAFETY: every arm returns a static NUL-terminated literal.
            let rendered = unsafe { CStr::from_ptr(text) };
            assert!(!rendered.to_bytes().is_empty());
        }
    }

    #[test]
    fn null_arguments_are_refused_rather_than_dereferenced() {
        // SAFETY: every call is given the null the function documents as
        // invalid, which is exactly what is under test.
        unsafe {
            let mut out: *mut hag_queue = ptr::null_mut();
            assert_eq!(
                hag_queue_open(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    0,
                    ptr::null(),
                    8,
                    0,
                    0,
                    &mut out
                ),
                HAG_INVALID_ARGUMENT
            );
            assert_eq!(
                hag_queue_write(ptr::null_mut(), 0, 0, ptr::null(), 0),
                HAG_INVALID_ARGUMENT
            );
            assert_eq!(
                hag_queue_read(ptr::null_mut(), 0, 0, ptr::null_mut(), 0),
                HAG_INVALID_ARGUMENT
            );
            assert_eq!(
                hag_queue_dispatch(ptr::null_mut(), 0, ptr::null(), ptr::null(), ptr::null(), 1),
                HAG_INVALID_ARGUMENT
            );
            assert_eq!(hag_queue_close(ptr::null_mut()), HAG_INVALID_ARGUMENT);
            assert_eq!(hag_queue_allocation_bytes(ptr::null()), 0);
            assert_eq!(hag_queue_is_poisoned(ptr::null()), 1);
        }
    }

    #[test]
    fn open_refuses_a_vgpr_count_outside_the_admitted_range() {
        let program = [0u8; 4];
        let pointers = [program.as_ptr()];
        let sizes = [program.len() as u32];
        let slots = [4u32; 3];
        for vgprs in [0u32, 4, 10, 128] {
            let mut out: *mut hag_queue = ptr::null_mut();
            // SAFETY: the bytes are readable; the call is expected to refuse
            // on the VGPR count before touching a device.
            let code = unsafe {
                hag_queue_open(
                    c"/dev/null".as_ptr(),
                    pointers.as_ptr(),
                    sizes.as_ptr(),
                    1,
                    slots.as_ptr(),
                    vgprs,
                    0,
                    0,
                    &mut out,
                )
            };
            assert_eq!(code, HAG_INVALID_ARGUMENT, "vgprs={vgprs}");
            assert!(out.is_null());
        }
    }

    #[test]
    fn caught_panic_poisoning_refuses_the_next_operation() {
        // No device is needed to exercise the C handle's independent panic
        // state. Queue retirement remains the transport's responsibility.
        let mut handle = hag_queue {
            queue: None,
            budget: MemoryBudget::new(0),
            poisoned: false,
        };
        assert_eq!(guard_handle(&mut handle, |_| HAG_REFUSED), HAG_REFUSED);
        assert!(!handle.poisoned, "ordinary refusals do not poison a handle");
        assert_eq!(
            guard_handle(&mut handle, |_| panic!("C boundary panic injection")),
            HAG_PANIC
        );
        assert!(handle.poisoned);
        let mut called = false;
        assert_eq!(
            guard_handle(&mut handle, |_| {
                called = true;
                HAG_OK
            }),
            HAG_POISONED
        );
        assert!(!called, "no work runs after a caught panic");
    }

    #[test]
    fn raw_library_limits_match_the_transport_without_device_access() {
        use crate::linux::compute::shader_library_bytes;

        assert_eq!(checked_program_count(0), Err(HAG_INVALID_ARGUMENT));
        assert_eq!(checked_program_count(1), Ok(1));
        assert_eq!(checked_program_count(MAX_C_PROGRAMS), Ok(32));
        assert_eq!(checked_program_count(MAX_C_PROGRAMS + 1), Err(HAG_REFUSED));
        assert_eq!(checked_program_count(u32::MAX), Err(HAG_REFUSED));
        let word = [0u8; 4];
        let programs = [word.as_slice(); MAX_C_PROGRAMS as usize + 1];
        assert!(shader_library_bytes(&programs[..MAX_C_PROGRAMS as usize]).is_ok());
        assert!(shader_library_bytes(&programs).is_err());

        assert_eq!(checked_program_length(0), Err(HAG_INVALID_ARGUMENT));
        assert_eq!(checked_program_length(3), Err(HAG_REFUSED));
        assert_eq!(checked_program_length(4), Ok(4));
        assert_eq!(checked_program_length(MAX_C_PROGRAM_BYTES), Ok(4096));
        assert_eq!(
            checked_program_length(MAX_C_PROGRAM_BYTES + 4),
            Err(HAG_REFUSED)
        );
        assert_eq!(checked_program_length(u32::MAX), Err(HAG_REFUSED));
        let page_and_word = [0u8; MAX_C_PROGRAM_BYTES as usize + 4];
        assert!(shader_library_bytes(&[&page_and_word[..MAX_C_PROGRAM_BYTES as usize]]).is_ok());
        assert!(shader_library_bytes(&[&page_and_word]).is_err());
    }

    #[test]
    fn open_rejects_oversized_counts_before_arrays_and_preserves_output() {
        let word = [0u8; 4];
        let pointers = [word.as_ptr()];
        let sizes = [4u32];
        let slots = [4u32; 3];
        let sentinel = std::ptr::NonNull::<hag_queue>::dangling().as_ptr();
        for count in [MAX_C_PROGRAMS + 1, u32::MAX] {
            let mut out = sentinel;
            // SAFETY: the documented oversized-count rejection occurs before
            // array or shader access, so these one-entry arrays suffice. The
            // output pointer is writable; its prior value is never dereferenced.
            let code = unsafe {
                hag_queue_open(
                    c"/dev/null".as_ptr(),
                    pointers.as_ptr(),
                    sizes.as_ptr(),
                    count,
                    slots.as_ptr(),
                    8,
                    0,
                    0,
                    &mut out,
                )
            };
            assert_eq!(code, HAG_REFUSED, "count={count}");
            assert_eq!(out, sentinel, "failed open must leave *out untouched");
        }
    }

    #[test]
    fn open_rejects_invalid_program_lengths_before_payload_or_device_access() {
        let word = [0u8; 4];
        let pointers = [word.as_ptr()];
        let slots = [4u32; 3];
        for (size, expected) in [
            (0, HAG_INVALID_ARGUMENT),
            (3, HAG_REFUSED),
            (MAX_C_PROGRAM_BYTES + 4, HAG_REFUSED),
            (u32::MAX, HAG_REFUSED),
        ] {
            let sizes = [size];
            let mut out = ptr::null_mut();
            // SAFETY: invalid program lengths are documented to be refused
            // before accessing the program bytes or opening any device.
            let code = unsafe {
                hag_queue_open(
                    c"/dev/null".as_ptr(),
                    pointers.as_ptr(),
                    sizes.as_ptr(),
                    1,
                    slots.as_ptr(),
                    8,
                    0,
                    0,
                    &mut out,
                )
            };
            assert_eq!(code, expected, "size={size}");
            assert!(out.is_null());
        }
    }

    #[test]
    fn transfer_lengths_obey_slot_usize_and_isize_limits_without_allocating() {
        let maximum = u64::from(MAX_SLOT_BYTES).min(isize::MAX as u64);
        assert_eq!(checked_transfer_length(0), Err(HAG_INVALID_ARGUMENT));
        assert_eq!(checked_transfer_length(1), Ok(1));
        assert_eq!(checked_transfer_length(maximum), Ok(maximum as usize));
        assert_eq!(checked_transfer_length(maximum + 1), Err(HAG_REFUSED));
        assert_eq!(
            checked_transfer_length(u64::from(MAX_SLOT_BYTES) + 1),
            Err(HAG_REFUSED)
        );
        assert_eq!(
            checked_transfer_length(isize::MAX as u64 + 1),
            Err(HAG_REFUSED)
        );
        assert_eq!(checked_transfer_length(u64::MAX), Err(HAG_REFUSED));
        if let Some(above_usize) = (usize::MAX as u64).checked_add(1) {
            assert_eq!(checked_transfer_length(above_usize), Err(HAG_REFUSED));
        }
    }

    #[test]
    fn oversized_transfers_refuse_before_accessing_buffers_or_handle() {
        let mut byte = 0xa5;
        for length in [
            u64::from(MAX_SLOT_BYTES) + 1,
            isize::MAX as u64 + 1,
            u64::MAX,
        ] {
            // SAFETY: oversized lengths are documented to be refused before
            // dereferencing either the handle or buffer. No giant allocation
            // or valid device handle is needed to verify that ordering.
            unsafe {
                assert_eq!(
                    hag_queue_write(ptr::null_mut(), 0, 0, &byte, length),
                    HAG_REFUSED
                );
                assert_eq!(
                    hag_queue_read(ptr::null_mut(), 0, 0, &mut byte, length),
                    HAG_REFUSED
                );
            }
            assert_eq!(byte, 0xa5);
        }
    }

    #[test]
    fn a_callers_mistake_is_not_reported_as_a_device_failure() {
        use crate::linux::Error;
        // Writing past a slot's end is the caller's error. Reported as
        // HAG_DEVICE it sent a real debugging session to the hardware before
        // anyone looked at the arguments.
        assert_eq!(
            classify(&ProbeError::Transport(Error::OutOfBounds)),
            HAG_REFUSED
        );
        assert_eq!(
            classify(&ProbeError::Transport(Error::NotRenderNode)),
            HAG_INVALID_ARGUMENT
        );
        assert_eq!(
            classify(&ProbeError::Transport(Error::WrongDriver)),
            HAG_INVALID_ARGUMENT
        );
        assert_eq!(
            classify(&ProbeError::Transport(Error::Quarantined)),
            HAG_POISONED
        );
        // A genuine OS failure still reads as one.
        assert_eq!(
            classify(&ProbeError::Transport(Error::Io {
                operation: "test",
                source: std::io::Error::other("x"),
            })),
            HAG_DEVICE
        );
        // The codes that carry the bounded-execution story keep their meaning.
        assert_eq!(classify(&ProbeError::Incomplete), HAG_INCOMPLETE);
        assert_eq!(classify(&ProbeError::Unsupported("x")), HAG_REFUSED);
    }

    #[test]
    fn vgpr_bounds_are_exported() {
        assert_eq!(hag_min_kernel_vgprs(), MIN_KERNEL_VGPRS);
        assert_eq!(hag_max_kernel_vgprs(), MAX_KERNEL_VGPRS);
        assert!(hag_min_kernel_vgprs() <= hag_max_kernel_vgprs());
    }
}
