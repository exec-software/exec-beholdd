// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

// SPDX-License-Identifier: MIT
// Copyright 2014, 2022, 2023, 2024 Advanced Micro Devices, Inc.
//
// Positive compute-register initialization adapted from IGT GPU Tools:
// https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/tree/691cc6fd50d5a3f6f2b5805c9698af1df28f6830
// lib/amdgpu/compute_utils/amd_dispatch_helpers.c
// lib/amdgpu/compute_utils/amd_dispatch.c
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is furnished to do
// so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Narrow GFX1151 PM4 ABI: immutable code, three raw SRDs, two scalar words.
//!
//! Register initialization follows the positive IGT recipe cited in
//! `probe_packets.rs`; no graphics context or privileged commands are emitted.
//! Raw descriptors follow GFX11 FORMAT_32_FLOAT, XYZW, stride zero, OOB_SELECT_RAW.

const PAGE: u64 = 4096;
const LIMIT: u64 = 1 << 48;
use super::KernelAbi;
#[cfg(test)]
use super::GFX1151_SERIAL_ABI;

fn packet(op: u32, count: u32) -> u32 {
    0xc000_0002 | (op << 8) | (count << 16)
}

fn register(words: &mut Vec<u32>, op: u32, offset: u32, values: &[u32]) {
    words.extend([packet(op, values.len() as u32), offset]);
    words.extend_from_slice(values);
}

fn end(address: u64, length: u32) -> Result<u64, &'static str> {
    if address == 0 || length == 0 {
        return Err("compute descriptors require nonzero ranges");
    }
    address
        .checked_add(u64::from(length))
        .filter(|end| *end <= LIMIT)
        .ok_or("compute descriptor exceeds 48-bit VA")
}

pub(super) fn commands(
    shader: u64,
    data: [u64; 3],
    lengths: [u32; 3],
    scalars: [u32; 2],
    groups: u32,
    alignment: u32,
    // The encoder ABI belongs to the library rather than to the build: two
    // libraries in one process may differ in VGPR allocation and must not
    // share a descriptor.
    abi: &KernelAbi,
) -> Result<Vec<u8>, &'static str> {
    if !alignment.is_power_of_two()
        || !(4..=4096).contains(&alignment)
        || groups == 0
        || groups > abi.max_groups()
        || scalars.iter().any(|value| !(1..=65536).contains(value))
    {
        return Err("invalid bounded compute dispatch geometry");
    }
    if shader == 0 || shader % 256 != 0 {
        return Err("compute shader entry must be nonzero and 256-byte aligned");
    }
    // Multiple immutable programs may share this executable page. Keep data
    // descriptors disjoint from the complete page, not just the selected entry.
    let shader_page = shader & !(PAGE - 1);
    let mut ranges = vec![(shader_page, end(shader_page, PAGE as u32)?)];
    for (address, length) in data.into_iter().zip(lengths) {
        ranges.push((address, end(address, length)?));
    }
    for (index, (start, stop)) in ranges.iter().enumerate() {
        if ranges[..index]
            .iter()
            .any(|(other_start, other_stop)| start < other_stop && other_start < stop)
        {
            return Err("overlapping compute descriptor ranges");
        }
    }
    let mut words = Vec::with_capacity(128);
    register(&mut words, 0x76, 0x204, &[0, 0, 0]);
    register(&mut words, 0x76, 0x218, &[0]);
    register(&mut words, 0x76, 0x222, &[0]);
    register(&mut words, 0x76, 0x224, &[0, 0, 0, 0]);
    register(&mut words, 0x76, 0x22a, &[0]);
    register(&mut words, 0x9b, 0x3000_0216, &[u32::MAX; 2]);
    register(&mut words, 0x9b, 0x219, &[u32::MAX; 2]);
    register(
        &mut words,
        0x76,
        0x20c,
        &[(shader >> 8) as u32, (shader >> 40) as u32],
    );
    // Wave64, with the library's VGPR allocation. GFX11 SGPR count is
    // reserved/zero. Preserve f32/f16 denormals, nearest-even arithmetic,
    // IEEE mode and ordered memory operations: only the VGPR granule varies.
    register(&mut words, 0x76, 0x212, &[abi.program_resources[0]]);
    register(&mut words, 0x76, 0x213, &[abi.program_resources[1]]);
    register(&mut words, 0x76, 0x207, &abi.threads);
    register(&mut words, 0x76, 0x228, &[abi.program_resources[2]]);
    let mut parameters = Vec::with_capacity(14);
    for (address, length) in data.into_iter().zip(lengths) {
        parameters.extend([
            address as u32,
            (address >> 32) as u32,
            length,
            abi.raw_descriptor,
        ]);
    }
    parameters.extend(scalars);
    register(&mut words, 0x76, 0x240, &parameters);
    register(&mut words, 0x76, 0x215, &[0]);
    words.extend([packet(0x15, 3), groups, 1, 1, 1]);
    let alignment_words = alignment.max(32) as usize / 4;
    while words.len() % alignment_words != 0 {
        words.push(0xffff_1000);
    }
    if words.len() * 4 > PAGE as usize {
        return Err("compute IB exceeds one page");
    }
    Ok(words.iter().flat_map(|word| word.to_le_bytes()).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::linux::compute::MAX_DISPATCH_GROUPS;

    #[test]
    fn fixed_abi_uses_byte_bounds_and_preserving_float_mode() {
        let bytes = commands(
            0x10000,
            [0x30000, 0x50000, 0x70000],
            [34, 128, 4],
            [32, 1],
            1,
            32,
            &GFX1151_SERIAL_ABI,
        )
        .unwrap();
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert!(words
            .windows(3)
            .any(|words| words == [0xc0017602, 0x212, 0x408f0001]));
        assert!(words
            .windows(3)
            .any(|words| words == [0xc0017602, 0x213, 0x9c]));
        assert!(words
            .windows(4)
            .any(|words| words == [0x30000, 0, 34, 0x30016fac]));
        assert!(words
            .windows(4)
            .any(|words| words == [0x50000, 0, 128, 0x30016fac]));
        assert!(words
            .windows(4)
            .any(|words| words == [0x70000, 0, 4, 0x30016fac]));
        assert!(words
            .windows(5)
            .any(|words| words == [0xc0031502, 1, 1, 1, 1]));
        assert_eq!(bytes.len() % 32, 0);
    }

    #[test]
    fn entries_and_byte_regions_can_share_pages_without_overlapping() {
        let bytes = commands(
            0x10200,
            [0x13002, 0x1d004, 0x1d044],
            [34, 64, 64],
            [16, 16],
            1,
            32,
            &GFX1151_SERIAL_ABI,
        )
        .unwrap();
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert!(words
            .windows(4)
            .any(|words| words == [0xc0027602, 0x20c, 0x102, 0]));
        assert!(words
            .windows(4)
            .any(|words| words == [0x13002, 0, 34, 0x30016fac]));
        assert!(words
            .windows(4)
            .any(|words| words == [0x1d004, 0, 64, 0x30016fac]));
        assert!(words
            .windows(4)
            .any(|words| words == [0x1d044, 0, 64, 0x30016fac]));
        // Protect the entire immutable code page, including other entry points.
        assert!(commands(
            0x10200,
            [0x10004, 0x1d004, 0x1d044],
            [4; 3],
            [1; 2],
            1,
            32,
            &GFX1151_SERIAL_ABI
        )
        .is_err());
        assert!(commands(
            0x10200,
            [0x13002, 0x1d004, 0x1d043],
            [34, 64, 64],
            [1; 2],
            1,
            32,
            &GFX1151_SERIAL_ABI,
        )
        .is_err());
        assert!(commands(
            0x10201,
            [0x13002, 0x1d004, 0x1d044],
            [4; 3],
            [1; 2],
            1,
            32,
            &GFX1151_SERIAL_ABI
        )
        .is_err());
    }

    /// A derived ABI must change the VGPR allocation and nothing else.
    ///
    /// This is the property the whole per-library change rests on: a library
    /// that needs more registers must not disturb the privileged bits of
    /// COMPUTE_PGM_RSRC1/2/3, and must not disturb another library at all.
    #[test]
    fn derived_vgpr_abi_changes_only_the_register_allocation() {
        let base = GFX1151_SERIAL_ABI;
        let wide = base.with_vgprs(16).expect("16 VGPRs is a valid granule");

        assert_eq!(base.vgprs(), 8);
        assert_eq!(wide.vgprs(), 16);
        assert_eq!(wide.program_resources[0], 0x408f_0003);
        // Everything outside the six-bit granule field is preserved.
        assert_eq!(
            wide.program_resources[0] & !0x3f,
            base.program_resources[0] & !0x3f
        );
        assert_eq!(wide.program_resources[1], base.program_resources[1]);
        assert_eq!(wide.program_resources[2], base.program_resources[2]);
        assert_eq!(wide.raw_descriptor, base.raw_descriptor);
        assert_eq!(wide.threads, base.threads);

        // Granules must be multiples of four inside the admitted range.
        assert!(base.with_vgprs(0).is_none());
        assert!(base.with_vgprs(4).is_none());
        assert!(base.with_vgprs(10).is_none());
        assert!(base.with_vgprs(128).is_none());
        for vgprs in [8, 16, 32, 64] {
            let derived = base.with_vgprs(vgprs).expect("valid granule");
            assert_eq!(derived.vgprs(), vgprs);
        }

        // The encoded command stream differs in exactly the RSRC1 word.
        let encode = |abi: &KernelAbi| {
            commands(
                0x10000,
                [0x30000, 0x50000, 0x70000],
                [34, 128, 4],
                [32, 1],
                1,
                32,
                abi,
            )
            .unwrap()
        };
        let narrow_words: Vec<u32> = encode(&base)
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        let wide_words: Vec<u32> = encode(&wide)
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        assert_eq!(narrow_words.len(), wide_words.len());
        let differences: Vec<_> = narrow_words
            .iter()
            .zip(&wide_words)
            .filter(|(a, b)| a != b)
            .collect();
        assert_eq!(
            differences,
            vec![(&0x408f_0001u32, &0x408f_0003u32)],
            "only the RSRC1 register write may differ"
        );
    }

    #[test]
    fn a_derived_abi_raises_the_workgroup_limit_and_nothing_else() {
        let wide = GFX1151_SERIAL_ABI
            .with_max_groups(32_400)
            .expect("1920x1080 worth of workgroups is inside the ceiling");

        // The qualified ABI still refuses what it always refused, so an
        // existing library's dispatches are unaffected by this one existing.
        let args = |abi, groups| {
            commands(
                0x10000,
                [0x30000, 0x50000, 0x70000],
                [4; 3],
                [1; 2],
                groups,
                32,
                abi,
            )
        };
        assert!(args(&GFX1151_SERIAL_ABI, 65).is_err());
        assert!(args(&wide, 65).is_ok());
        assert!(args(&wide, 32_400).is_ok());

        // The ceiling is still a ceiling.
        assert!(args(&wide, 65_537).is_err());
        assert!(GFX1151_SERIAL_ABI.with_max_groups(0).is_none());
        assert!(GFX1151_SERIAL_ABI
            .with_max_groups(MAX_DISPATCH_GROUPS + 1)
            .is_none());

        // Raising the limit must not move a single bit of the command stream
        // for a dispatch both ABIs admit: the workgroup count is an argument,
        // not a register, and nothing else may ride along with it.
        assert_eq!(
            args(&GFX1151_SERIAL_ABI, 64).unwrap(),
            args(&wide, 64).unwrap(),
            "a raised workgroup limit must not change an existing dispatch"
        );
    }

    #[test]
    fn rejects_invalid_ranges_geometry_and_alignment() {
        assert!(commands(
            0x10000,
            [0x10000, 0x50000, 0x70000],
            [4; 3],
            [1; 2],
            1,
            32,
            &GFX1151_SERIAL_ABI
        )
        .is_err());
        for address in [0, 1, 0x10001, 1 << 48, u64::MAX] {
            assert!(commands(
                address,
                [0x30000, 0x50000, 0x70000],
                [4; 3],
                [1; 2],
                1,
                32,
                &GFX1151_SERIAL_ABI
            )
            .is_err());
        }
        for groups in [0, 65, u32::MAX] {
            assert!(commands(
                0x10000,
                [0x30000, 0x50000, 0x70000],
                [4; 3],
                [1; 2],
                groups,
                32,
                &GFX1151_SERIAL_ABI,
            )
            .is_err());
        }
        for alignment in [0, 2, 3, 8192] {
            assert!(commands(
                0x10000,
                [0x30000, 0x50000, 0x70000],
                [4; 3],
                [1; 2],
                1,
                alignment,
                &GFX1151_SERIAL_ABI,
            )
            .is_err());
        }
    }
}
