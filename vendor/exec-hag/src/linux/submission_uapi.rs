// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

// Linux AMDGPU submission UAPI; no libdrm linkage. Mirrors amdgpu_drm.h.
// Copyright 2000 Precision Insight, Inc., Cedar Park, Texas.
// Copyright 2000 VA Linux Systems, Inc., Fremont, California.
// Copyright 2002 Tungsten Graphics, Inc., Cedar Park, Texas.
// Copyright 2014 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

use std::mem::size_of;

pub(super) const INFO_HW_IP_INFO: u32 = 0x02;
pub(super) const HW_IP_COMPUTE: u32 = 1;
pub(super) const CTX_ALLOC: u32 = 1;
pub(super) const CTX_FREE: u32 = 2;
pub(super) const CTX_QUERY_STATE2: u32 = 4;
pub(super) const CTX_PRIORITY_NORMAL: i32 = 0;
pub(super) const BO_LIST_CREATE: u32 = 0;
pub(super) const BO_LIST_DESTROY: u32 = 1;
pub(super) const VA_MAP: u32 = 1;
pub(super) const VA_UNMAP: u32 = 2;
pub(super) const VM_PAGE_READABLE: u32 = 1 << 1;
pub(super) const VM_PAGE_WRITEABLE: u32 = 1 << 2;
pub(super) const VM_PAGE_EXECUTABLE: u32 = 1 << 3;
pub(super) const CHUNK_ID_IB: u32 = 1;
#[cfg(feature = "native-compute")]
pub(super) const IB_FLAG_EMIT_MEM_SYNC: u32 = 1 << 6;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct HwIpInfo {
    pub hw_ip_version_major: u32,
    pub hw_ip_version_minor: u32,
    pub capabilities_flags: u64,
    pub ib_start_alignment: u32,
    pub ib_size_alignment: u32,
    pub available_rings: u32,
    pub ip_discovery_version: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ContextIn {
    pub op: u32,
    pub flags: u32,
    pub ctx_id: u32,
    pub priority: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ContextAllocOut {
    pub ctx_id: u32,
    pub pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct ContextStateOut {
    pub flags: u64,
    pub hangs: u32,
    pub reset_status: u32,
}

#[repr(C)]
pub(super) union Context {
    pub input: ContextIn,
    pub allocated: ContextAllocOut,
    pub state: ContextStateOut,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct BoListIn {
    pub operation: u32,
    pub list_handle: u32,
    pub bo_number: u32,
    pub bo_info_size: u32,
    pub bo_info_ptr: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct BoListOut {
    pub list_handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub(super) union BoList {
    pub input: BoListIn,
    pub output: BoListOut,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct BoListEntry {
    pub bo_handle: u32,
    pub bo_priority: u32,
}

// Deliberately use the original 40-byte VA request. Newer kernels add syncobj
// extension fields. DRM dispatch uses the ioctl number, copies the supplied size,
// then zero-fills the remaining kernel structure (drivers/gpu/drm/drm_ioctl.c).
// Thus this request asks for no extension without requiring the newest kernel.
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct GemVa {
    pub handle: u32,
    pub pad: u32,
    pub operation: u32,
    pub flags: u32,
    pub va_address: u64,
    pub offset_in_bo: u64,
    pub map_size: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct CsChunk {
    pub chunk_id: u32,
    pub length_dw: u32,
    pub chunk_data: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct CsChunkIb {
    pub pad: u32,
    pub flags: u32,
    pub va_start: u64,
    pub ib_bytes: u32,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct CsIn {
    pub ctx_id: u32,
    pub bo_list_handle: u32,
    pub num_chunks: u32,
    pub flags: u32,
    // Pointer to an array of u64 pointers to CsChunk, NOT an array of CsChunk.
    pub chunks: u64,
}

#[repr(C)]
pub(super) union Cs {
    pub input: CsIn,
    pub handle: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct WaitCsIn {
    pub handle: u64,
    // Absolute CLOCK_MONOTONIC nanoseconds, not a relative duration.
    pub timeout: u64,
    pub ip_type: u32,
    pub ip_instance: u32,
    pub ring: u32,
    pub ctx_id: u32,
}

#[repr(C)]
pub(super) union WaitCs {
    pub input: WaitCsIn,
    // On success, 0 means complete and 1 means still busy. A fence error is an
    // ioctl error, not successful completion; never consume output in that case.
    pub status: u64,
}

pub(super) const CTX: libc::c_ulong = libc::_IOWR::<Context>(b'd' as u32, 0x42) as _;
pub(super) const BO_LIST: libc::c_ulong = libc::_IOWR::<BoList>(b'd' as u32, 0x43) as _;
pub(super) const CS: libc::c_ulong = libc::_IOWR::<Cs>(b'd' as u32, 0x44) as _;
pub(super) const GEM_VA: libc::c_ulong = libc::_IOW::<GemVa>(b'd' as u32, 0x48) as _;
pub(super) const WAIT_CS: libc::c_ulong = libc::_IOWR::<WaitCs>(b'd' as u32, 0x49) as _;

const _: () = assert!(size_of::<HwIpInfo>() == 32);
const _: () = assert!(size_of::<Context>() == 16);
const _: () = assert!(size_of::<BoList>() == 24);
const _: () = assert!(size_of::<BoListEntry>() == 8);
const _: () = assert!(size_of::<GemVa>() == 40);
const _: () = assert!(size_of::<CsChunk>() == 16);
const _: () = assert!(size_of::<CsChunkIb>() == 32);
const _: () = assert!(size_of::<Cs>() == 24);
const _: () = assert!(size_of::<WaitCs>() == 32);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of};

    #[test]
    fn submission_layouts_preserve_pointer_indirection_and_union_sizes() {
        assert_eq!(offset_of!(CsIn, chunks), 16);
        assert_eq!(size_of::<CsIn>(), 24);
        assert_eq!(offset_of!(CsChunk, chunk_data), 8);
        assert_eq!(offset_of!(CsChunkIb, va_start), 8);
        assert_eq!(offset_of!(CsChunkIb, ring), 28);
        assert_eq!(offset_of!(WaitCsIn, timeout), 8);
        assert_eq!(offset_of!(WaitCsIn, ctx_id), 28);
        assert_eq!(offset_of!(ContextIn, priority), 12);
        assert_eq!(offset_of!(ContextStateOut, reset_status), 12);
        assert_eq!(offset_of!(BoListIn, bo_info_ptr), 16);
        assert_eq!(offset_of!(GemVa, va_address), 16);
        assert_eq!(offset_of!(GemVa, map_size), 32);
        assert_eq!(size_of::<Context>(), size_of::<ContextStateOut>());
        assert_eq!(size_of::<Cs>(), size_of::<CsIn>());
        assert_eq!(size_of::<WaitCs>(), size_of::<WaitCsIn>());
        assert!(align_of::<Cs>() >= align_of::<u64>());
    }

    #[test]
    #[ignore = "requires a C compiler and Linux libdrm development headers; no GPU access"]
    fn submission_matches_installed_linux_headers() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let contracts = [
            (
                "sizeof(struct drm_amdgpu_info_hw_ip)",
                size_of::<HwIpInfo>() as u128,
            ),
            (
                "sizeof(struct drm_amdgpu_ctx_in)",
                size_of::<ContextIn>() as u128,
            ),
            ("sizeof(union drm_amdgpu_ctx)", size_of::<Context>() as u128),
            (
                "sizeof(struct drm_amdgpu_bo_list_in)",
                size_of::<BoListIn>() as u128,
            ),
            (
                "sizeof(struct drm_amdgpu_bo_list_entry)",
                size_of::<BoListEntry>() as u128,
            ),
            (
                "sizeof(union drm_amdgpu_bo_list)",
                size_of::<BoList>() as u128,
            ),
            ("sizeof(struct drm_amdgpu_cs_in)", size_of::<CsIn>() as u128),
            ("sizeof(union drm_amdgpu_cs)", size_of::<Cs>() as u128),
            (
                "sizeof(struct drm_amdgpu_cs_chunk)",
                size_of::<CsChunk>() as u128,
            ),
            (
                "sizeof(struct drm_amdgpu_cs_chunk_ib)",
                size_of::<CsChunkIb>() as u128,
            ),
            (
                "sizeof(struct drm_amdgpu_wait_cs_in)",
                size_of::<WaitCsIn>() as u128,
            ),
            (
                "sizeof(union drm_amdgpu_wait_cs)",
                size_of::<WaitCs>() as u128,
            ),
            (
                "offsetof(struct drm_amdgpu_cs_in, chunks)",
                offset_of!(CsIn, chunks) as u128,
            ),
            (
                "offsetof(struct drm_amdgpu_cs_chunk_ib, ring)",
                offset_of!(CsChunkIb, ring) as u128,
            ),
            (
                "offsetof(struct drm_amdgpu_wait_cs_in, timeout)",
                offset_of!(WaitCsIn, timeout) as u128,
            ),
            (
                "offsetof(struct drm_amdgpu_gem_va, va_address)",
                offset_of!(GemVa, va_address) as u128,
            ),
            (
                "offsetof(struct drm_amdgpu_gem_va, map_size)",
                offset_of!(GemVa, map_size) as u128,
            ),
            ("DRM_IOCTL_AMDGPU_CTX", CTX as u128),
            ("DRM_IOCTL_AMDGPU_BO_LIST", BO_LIST as u128),
            ("DRM_IOCTL_AMDGPU_CS", CS as u128),
            ("DRM_IOCTL_AMDGPU_WAIT_CS", WAIT_CS as u128),
            // Compare the VA ioctl's direction, type, and number independently of
            // newer headers' extension size, then validate our 40-byte encoding.
            (
                "_IOC(_IOC_DIR(DRM_IOCTL_AMDGPU_GEM_VA), _IOC_TYPE(DRM_IOCTL_AMDGPU_GEM_VA), _IOC_NR(DRM_IOCTL_AMDGPU_GEM_VA), 40)",
                GEM_VA as u128,
            ),
            ("AMDGPU_INFO_HW_IP_INFO", INFO_HW_IP_INFO as u128),
            ("AMDGPU_HW_IP_COMPUTE", HW_IP_COMPUTE as u128),
            ("AMDGPU_CTX_OP_ALLOC_CTX", CTX_ALLOC as u128),
            ("AMDGPU_CTX_OP_FREE_CTX", CTX_FREE as u128),
            ("AMDGPU_CTX_OP_QUERY_STATE2", CTX_QUERY_STATE2 as u128),
            ("AMDGPU_CTX_PRIORITY_NORMAL", CTX_PRIORITY_NORMAL as u128),
            ("AMDGPU_BO_LIST_OP_CREATE", BO_LIST_CREATE as u128),
            ("AMDGPU_BO_LIST_OP_DESTROY", BO_LIST_DESTROY as u128),
            ("AMDGPU_VA_OP_MAP", VA_MAP as u128),
            ("AMDGPU_VA_OP_UNMAP", VA_UNMAP as u128),
            ("AMDGPU_VM_PAGE_READABLE", VM_PAGE_READABLE as u128),
            ("AMDGPU_VM_PAGE_WRITEABLE", VM_PAGE_WRITEABLE as u128),
            ("AMDGPU_VM_PAGE_EXECUTABLE", VM_PAGE_EXECUTABLE as u128),
            ("AMDGPU_CHUNK_ID_IB", CHUNK_ID_IB as u128),
            #[cfg(feature = "native-compute")]
            ("AMDGPU_IB_FLAG_EMIT_MEM_SYNC", IB_FLAG_EMIT_MEM_SYNC as u128),
        ];
        let mut source = String::from(
            "#include <stddef.h>\n#include <libdrm/drm.h>\n#include <libdrm/amdgpu_drm.h>\n",
        );
        source.push_str(&format!(
            "_Static_assert(sizeof(struct drm_amdgpu_gem_va) >= {}, \"VA prefix fits\");\n",
            size_of::<GemVa>()
        ));
        for (expression, expected) in contracts {
            source.push_str(&format!(
                "_Static_assert(({expression}) == {expected}ULL, \"{expression}\");\n"
            ));
        }
        let mut compiler = Command::new("cc")
            .args(["-std=c11", "-x", "c", "-fsyntax-only", "-"])
            .stdin(Stdio::piped())
            .spawn()
            .expect("C compiler");
        compiler
            .stdin
            .take()
            .expect("compiler stdin")
            .write_all(source.as_bytes())
            .expect("write C contracts");
        assert!(compiler.wait().expect("compiler result").success());
    }
}
