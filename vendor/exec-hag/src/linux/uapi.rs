// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

// Linux DRM/AMDGPU UAPI. Layouts mirror drm.h and amdgpu_drm.h; no libdrm linkage.
// Upstream AMDGPU UAPI notice:
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
//
// Upstream DRM UAPI notice:
// Copyright 1999 Precision Insight, Inc., Cedar Park, Texas.
// Copyright 2000 VA Linux Systems, Inc., Sunnyvale, California.
// All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice (including the next
// paragraph) shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// VA LINUX SYSTEMS AND/OR ITS SUPPLIERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.

use std::mem::size_of;

pub(super) const INFO_DEVICE: u32 = 0x16;
pub(super) const INFO_MEMORY: u32 = 0x19;
pub(super) const DOMAIN_GTT: u64 = 2;
pub(super) const DOMAIN_VRAM: u64 = 4;
pub(super) const CPU_ACCESS_REQUIRED: u64 = 1;
pub(super) const NO_CPU_ACCESS: u64 = 2;

#[repr(C)]
pub(super) struct Version {
    pub major: libc::c_int,
    pub minor: libc::c_int,
    pub patchlevel: libc::c_int,
    pub name_len: usize,
    pub name: *mut libc::c_char,
    pub date_len: usize,
    pub date: *mut libc::c_char,
    pub desc_len: usize,
    pub desc: *mut libc::c_char,
}

// Raw pointers do not implement Default on the declared Rust 1.85 minimum.
impl Default for Version {
    fn default() -> Self {
        Self {
            major: 0,
            minor: 0,
            patchlevel: 0,
            name_len: 0,
            name: std::ptr::null_mut(),
            date_len: 0,
            date: std::ptr::null_mut(),
            desc_len: 0,
            desc: std::ptr::null_mut(),
        }
    }
}

#[repr(C)]
#[derive(Default)]
pub(super) struct Info {
    pub return_pointer: u64,
    pub return_size: u32,
    pub query: u32,
    pub query_data: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub(super) struct HeapInfo {
    pub total_heap_size: u64,
    pub usable_heap_size: u64,
    pub heap_usage: u64,
    pub max_allocation: u64,
}

#[repr(C)]
#[derive(Default)]
pub(super) struct MemoryInfo {
    pub vram: HeapInfo,
    pub cpu_accessible_vram: HeapInfo,
    pub gtt: HeapInfo,
}

#[repr(C)]
#[derive(Default)]
// Stable prefix of drm_amdgpu_info_device. INFO.return_size limits kernel writes;
// newer fields are deliberately omitted, not assumed to exist on older kernels.
pub(super) struct DeviceInfo {
    pub device_id: u32,
    pub chip_rev: u32,
    pub external_rev: u32,
    pub pci_rev: u32,
    pub family: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub gpu_counter_freq: u32,
    pub max_engine_clock: u64,
    pub max_memory_clock: u64,
    pub cu_active_number: u32,
    pub cu_ao_mask: u32,
    pub cu_bitmap: [[u32; 4]; 4],
    pub enabled_rb_pipes_mask: u32,
    pub num_rb_pipes: u32,
    pub num_hw_gfx_contexts: u32,
    pub pcie_gen: u32,
    pub ids_flags: u64,
    pub virtual_address_offset: u64,
    pub virtual_address_max: u64,
    pub virtual_address_alignment: u32,
    pub pte_fragment_size: u32,
    pub gart_page_size: u32,
    pub ce_ram_size: u32,
    pub vram_type: u32,
    pub vram_bit_width: u32,
    pub vce_harvest_config: u32,
    pub gc_double_offchip_lds_buf: u32,
    pub prim_buf_gpu_addr: u64,
    pub pos_buf_gpu_addr: u64,
    pub cntl_sb_buf_gpu_addr: u64,
    pub param_buf_gpu_addr: u64,
    pub prim_buf_size: u32,
    pub pos_buf_size: u32,
    pub cntl_sb_buf_size: u32,
    pub param_buf_size: u32,
    pub wave_front_size: u32,
    pub num_shader_visible_vgprs: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GemCreateIn {
    pub bo_size: u64,
    pub alignment: u64,
    pub domains: u64,
    pub domain_flags: u64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GemCreateOut {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub(super) union GemCreate {
    pub input: GemCreateIn,
    pub output: GemCreateOut,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(super) struct GemMmapIn {
    pub handle: u32,
    pub pad: u32,
}

#[repr(C)]
pub(super) union GemMmap {
    pub input: GemMmapIn,
    pub offset: u64,
}

#[repr(C)]
pub(super) struct GemClose {
    pub handle: u32,
    pub pad: u32,
}

pub(super) const VERSION: libc::c_ulong = libc::_IOWR::<Version>(b'd' as u32, 0) as _;
pub(super) const INFO: libc::c_ulong = libc::_IOW::<Info>(b'd' as u32, 0x45) as _;
pub(super) const GEM_CREATE: libc::c_ulong = libc::_IOWR::<GemCreate>(b'd' as u32, 0x40) as _;
pub(super) const GEM_MMAP: libc::c_ulong = libc::_IOWR::<GemMmap>(b'd' as u32, 0x41) as _;
pub(super) const GEM_CLOSE: libc::c_ulong = libc::_IOW::<GemClose>(b'd' as u32, 9) as _;

const _: () = assert!(size_of::<Info>() == 32);
const _: () = assert!(size_of::<MemoryInfo>() == 96);
const _: () = assert!(size_of::<GemCreate>() == 32);
const _: () = assert!(size_of::<GemMmap>() == 8);
const _: () = assert!(size_of::<GemClose>() == 8);

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn default_version_has_no_caller_buffers() {
        let version = Version::default();
        assert_eq!(
            (version.major, version.minor, version.patchlevel),
            (0, 0, 0)
        );
        assert_eq!(
            (version.name_len, version.date_len, version.desc_len),
            (0, 0, 0)
        );
        assert!(version.name.is_null());
        assert!(version.date.is_null());
        assert!(version.desc.is_null());
    }

    #[test]
    fn stable_uapi_layouts_and_selectors() {
        assert_eq!(offset_of!(Info, return_pointer), 0);
        assert_eq!(offset_of!(Info, return_size), 8);
        assert_eq!(offset_of!(Info, query), 12);
        assert_eq!(offset_of!(Info, query_data), 16);
        assert_eq!(offset_of!(DeviceInfo, max_engine_clock), 32);
        assert_eq!(offset_of!(DeviceInfo, cu_active_number), 48);
        assert_eq!(offset_of!(DeviceInfo, wave_front_size), 240);
        assert_eq!(offset_of!(MemoryInfo, gtt), 64);
        assert_eq!(INFO_DEVICE, 0x16);
        assert_eq!(INFO_MEMORY, 0x19);
    }

    #[test]
    #[ignore = "requires a C compiler and Linux libdrm development headers; no GPU access"]
    fn matches_installed_linux_headers() {
        use std::io::Write;
        use std::process::{Command, Stdio};
        let contracts = [
            ("sizeof(struct drm_version)", size_of::<Version>() as u128),
            ("sizeof(struct drm_amdgpu_info)", size_of::<Info>() as u128),
            (
                "sizeof(struct drm_amdgpu_memory_info)",
                size_of::<MemoryInfo>() as u128,
            ),
            (
                "offsetof(struct drm_amdgpu_info_device, wave_front_size)",
                offset_of!(DeviceInfo, wave_front_size) as u128,
            ),
            ("DRM_IOCTL_VERSION", VERSION as u128),
            ("DRM_IOCTL_AMDGPU_INFO", INFO as u128),
            ("DRM_IOCTL_AMDGPU_GEM_CREATE", GEM_CREATE as u128),
            ("DRM_IOCTL_AMDGPU_GEM_MMAP", GEM_MMAP as u128),
            ("DRM_IOCTL_GEM_CLOSE", GEM_CLOSE as u128),
            ("AMDGPU_INFO_DEV_INFO", INFO_DEVICE as u128),
            ("AMDGPU_INFO_MEMORY", INFO_MEMORY as u128),
        ];
        let mut source = String::from(
            "#include <stddef.h>\n#include <libdrm/drm.h>\n#include <libdrm/amdgpu_drm.h>\n",
        );
        // INFO is size-versioned: query only the stable prefix we actually consume.
        source.push_str(&format!(
            "_Static_assert(sizeof(struct drm_amdgpu_info_device) >= {}, \"device prefix fits\");\n",
            size_of::<DeviceInfo>()
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
