// Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
// SPDX-License-Identifier: MIT

// SPDX-License-Identifier: MIT
// Copyright 2014, 2022, 2023, 2024 Advanced Micro Devices, Inc.
//
// Shader and positive compute-dispatch recipe adapted from IGT GPU Tools:
// https://gitlab.freedesktop.org/drm/igt-gpu-tools/-/tree/691cc6fd50d5a3f6f2b5805c9698af1df28f6830
// lib/amdgpu/shaders/amd_shaders.c (bufferclear_cs_shader_gfx11)
// lib/amdgpu/compute_utils/amd_dispatch_helpers.c (GFX11 register setup)
// lib/amdgpu/compute_utils/amd_dispatch.c (amdgpu_memset_dispatch_test)
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

//! Closed, hardware-free encoding of one GFX11 compute fill proof.
//!
//! This is not a general command builder: shader, registers, dispatch dimensions,
//! format and fill value are fixed. No user command words or arbitrary shader
//! bytes are accepted. The transport must separately validate the GPU identity,
//! reserve disjoint VAs/BOs, retain them until completion, and check the fence and
//! reset state before reading output. The caller must also exclude the IB and
//! surrounding canary pages from both ranges checked here.

pub(super) const OUTPUT_BYTES: usize = 16 * 1024;
pub(super) const SHADER_ALLOCATION_BYTES: usize = 4096;
pub(super) const MAX_COMMAND_BYTES: usize = 4096;
pub(super) const FILL_BYTE: u8 = 0x22;
const GPU_ADDRESS_LIMIT: u64 = 1 << 48;
const PAGE_BYTES: u64 = 4096;
const SH_REG: u32 = 0x76;
const SH_REG_INDEX: u32 = 0x9b;
const DISPATCH_DIRECT: u32 = 0x15;
const COMPUTE_NOP: u32 = 0xffff_1000;
const CODE_END: u32 = 0xbf9f_0000;

// Exact 15 instruction DWORDs plus all 64 s_code_end padding DWORDs from IGT.
// Offline LLVM 18.1.3 gfx1150 assembly reproduces these bytes, wave64 enabled.
// s[0:3] is a 16-byte-stride SRD, s[4:7] is the fill, s8 is workgroup X;
// packed v0 contains the work-item ID. Each of 16 groups has 64 work-items,
// each storing one 16-byte record: (group_x * 64 + local_x) * 16.
const SHADER_CODE: [u32; 15] = [
    0xb080_2006, // s_version GFX11 | W64
    0xbf84_0003, // s_set_inst_prefetch_distance 3
    0x3600_00ff,
    0x0000_03ff, // v_and_b32 v0, 0x3ff, v0
    0x7e02_0205, // v_mov_b32 v1, s5
    0x7e04_0206, // v_mov_b32 v2, s6
    0x7e06_0207, // v_mov_b32 v3, s7
    0xbf87_0004, // s_delay_alu instid0(VALU_DEP_4)
    0xd646_0004,
    0x0401_0c08, // v_lshl_add_u32 v4, s8, 6, v0
    0x7e00_0204, // v_mov_b32 v0, s4
    0xe01c_0000,
    0x8080_0004, // buffer_store_format_xyzw v[0:3], v4, s[0:3], 0 idxen
    0xbfb6_0003, // s_sendmsg MSG_DEALLOC_VGPRS
    0xbfb0_0000, // s_endpgm
];

pub(super) fn shader_bytes() -> Vec<u8> {
    let mut words = SHADER_CODE.to_vec();
    words.extend([CODE_END; 64]);
    to_bytes(&words)
}

fn checked_end(address: u64, size: usize) -> Result<u64, &'static str> {
    if address == 0 || address % PAGE_BYTES != 0 {
        return Err("probe addresses must be nonzero and 4096-byte aligned");
    }
    let end = address
        .checked_add(size as u64)
        .ok_or("probe address range overflows")?;
    if end > GPU_ADDRESS_LIMIT {
        return Err("probe address range exceeds the supported 48-bit GPU VA");
    }
    Ok(end)
}

fn packet3_compute(op: u32, count: u32) -> u32 {
    0xc000_0002 | (op << 8) | (count << 16)
}

fn set_registers(words: &mut Vec<u32>, op: u32, offset: u32, values: &[u32]) {
    words.push(packet3_compute(op, values.len() as u32));
    words.push(offset);
    words.extend_from_slice(values);
}

fn to_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}

/// Build one fixed compute dispatch; `ib_alignment` is the kernel's IB-size
/// alignment in bytes, not the IB start-address alignment. At least the upstream
/// eight-DWORD alignment is retained; no more than one 4096-byte IB is accepted.
pub(super) fn build_commands(
    shader_addr: u64,
    output_addr: u64,
    ib_alignment: u32,
) -> Result<Vec<u8>, &'static str> {
    let shader_end = checked_end(shader_addr, SHADER_ALLOCATION_BYTES)?;
    let output_end = checked_end(output_addr, OUTPUT_BYTES)?;
    if shader_addr < output_end && output_addr < shader_end {
        return Err("shader and output address ranges overlap");
    }
    if !ib_alignment.is_power_of_two()
        || ib_alignment < 4
        || ib_alignment as usize > MAX_COMMAND_BYTES
    {
        return Err("IB-size alignment must be a power of two from 4 to 4096 bytes");
    }

    let mut words = Vec::with_capacity(96);
    // Compute ring: no GFX CONTEXT_CONTROL, CLEAR_STATE or privileged commands.
    set_registers(&mut words, SH_REG, 0x204, &[0, 0, 0]); // COMPUTE_START_X/Y/Z
    set_registers(&mut words, SH_REG, 0x218, &[0]); // COMPUTE_TMPRING_SIZE
    set_registers(&mut words, SH_REG, 0x222, &[0]);
    set_registers(&mut words, SH_REG, 0x224, &[0, 0, 0, 0]);
    set_registers(&mut words, SH_REG, 0x22a, &[0]); // COMPUTE_SHADER_CHKSUM
    set_registers(&mut words, SH_REG_INDEX, 0x3000_0216, &[u32::MAX; 2]);
    set_registers(&mut words, SH_REG_INDEX, 0x219, &[u32::MAX; 2]);
    set_registers(
        &mut words,
        SH_REG,
        0x20c,
        &[(shader_addr >> 8) as u32, (shader_addr >> 40) as u32],
    );
    // The source starts with 0x600c0041, then clears bit 29 before emission.
    set_registers(&mut words, SH_REG, 0x212, &[0x400c_0041]);
    set_registers(&mut words, SH_REG, 0x213, &[0x90]); // 8 user SGPRs; TGID_X
    set_registers(&mut words, SH_REG, 0x207, &[64]);
    set_registers(&mut words, SH_REG, 0x208, &[1]);
    set_registers(&mut words, SH_REG, 0x209, &[1]);
    set_registers(&mut words, SH_REG, 0x228, &[0x3f0]); // COMPUTE_PGM_RSRC3
    set_registers(
        &mut words,
        SH_REG,
        0x240,
        &[
            output_addr as u32,
            ((output_addr >> 32) as u32) | 0x0010_0000,
            0x400,
            0x1003_dfac,
        ],
    );
    set_registers(&mut words, SH_REG, 0x244, &[0x2222_2222; 4]);
    set_registers(&mut words, SH_REG, 0x215, &[0]); // COMPUTE_RESOURCE_LIMITS
    words.extend([packet3_compute(DISPATCH_DIRECT, 3), 16, 1, 1, 1]);
    let alignment_dw = (ib_alignment.max(32) / 4) as usize;
    while words.len() % alignment_dw != 0 {
        words.push(COMPUTE_NOP);
    }
    debug_assert!(words.len() * 4 <= MAX_COMMAND_BYTES);
    Ok(to_bytes(&words))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn shader_retains_exact_upstream_words_and_padding() {
        let shader = words(&shader_bytes());
        assert_eq!(shader.len(), 79);
        assert_eq!(
            &shader[..15],
            &[
                0xb0802006, 0xbf840003, 0x360000ff, 0x000003ff, 0x7e020205, 0x7e040206, 0x7e060207,
                0xbf870004, 0xd6460004, 0x04010c08, 0x7e000204, 0xe01c0000, 0x80800004, 0xbfb60003,
                0xbfb00000,
            ]
        );
        assert!(shader[15..].iter().all(|word| *word == 0xbf9f0000));
        assert!(shader_bytes().len() < SHADER_ALLOCATION_BYTES);
    }

    #[test]
    fn commands_match_positive_igt_gfx11_recipe() {
        let actual = words(&build_commands(0x1234_5678_9000, 0xabcd_0000_0000, 32).unwrap());
        let expected = [
            0xc0037602, 0x204, 0, 0, 0, 0xc0017602, 0x218, 0, 0xc0017602, 0x222, 0, 0xc0047602,
            0x224, 0, 0, 0, 0, 0xc0017602, 0x22a, 0, 0xc0029b02, 0x30000216, 0xffffffff,
            0xffffffff, 0xc0029b02, 0x219, 0xffffffff, 0xffffffff, 0xc0027602, 0x20c, 0x34567890,
            0x12, 0xc0017602, 0x212, 0x400c0041, 0xc0017602, 0x213, 0x90, 0xc0017602, 0x207, 64,
            0xc0017602, 0x208, 1, 0xc0017602, 0x209, 1, 0xc0017602, 0x228, 0x3f0, 0xc0047602,
            0x240, 0, 0x0010abcd, 0x400, 0x1003dfac, 0xc0047602, 0x244, 0x22222222, 0x22222222,
            0x22222222, 0x22222222, 0xc0017602, 0x215, 0, 0xc0031502, 16, 1, 1, 1, 0xffff1000,
            0xffff1000,
        ];
        assert_eq!(actual, expected);
        assert_eq!(actual.len() * 4, 288);
        assert_eq!(16 * 64 * 16, OUTPUT_BYTES);
        assert_eq!(FILL_BYTE, 0x22);
    }

    #[test]
    fn accepts_supported_ib_size_alignments_without_changing_payload() {
        let base = build_commands(0x10000, 0x20000, 32).unwrap();
        for shift in 2..=12 {
            let alignment = 1 << shift;
            let command = build_commands(0x10000, 0x20000, alignment).unwrap();
            assert_eq!(command.len() % alignment.max(32) as usize, 0);
            assert_eq!(&command[..base.len()], base);
            assert!(words(&command[base.len()..])
                .iter()
                .all(|word| *word == COMPUTE_NOP));
            assert!(command.len() <= MAX_COMMAND_BYTES);
        }
        for alignment in [0, 1, 2, 3, 6, 31, 33, 8192, u32::MAX] {
            assert!(build_commands(0x10000, 0x20000, alignment).is_err());
        }
    }

    #[test]
    fn rejects_null_misaligned_out_of_range_and_overlapping_addresses() {
        for address in [0, 1, 255, 256, 4095, (1 << 48), u64::MAX - 4095] {
            assert!(build_commands(address, 0x20000, 32).is_err());
            assert!(build_commands(0x10000, address, 32).is_err());
        }
        assert!(build_commands(0x10000, 0x10000, 32).is_err());
        assert!(build_commands(0x13000, 0x10000, 32).is_err());
        assert!(build_commands(0x10000, 0xd000, 32).is_err());
        assert!(build_commands(0x10000, GPU_ADDRESS_LIMIT - 4096, 32).is_err());
        assert!(build_commands(0x10000, GPU_ADDRESS_LIMIT - OUTPUT_BYTES as u64, 32).is_ok());
        assert!(build_commands(GPU_ADDRESS_LIMIT - 4096, 0x20000, 32).is_ok());
        assert!(build_commands(0x10000, 0x11000, 32).is_ok());
        assert!(build_commands(0x14000, 0x10000, 32).is_ok());
    }
}
