// SPDX-License-Identifier: Apache-2.0
// UNQUALIFIED SOURCE: GGML Q4_K selected-row decode, 256..5120 columns.
// SRD0 exact (columns/256)*144 bytes; SRD1 exact4-byte unread dummy;
// SRD2 exact columns*4 bytes. Scalars [columns,1]. Up to80 wave64 groups.
// v0 column/subblock, v1 output byte, v2 block base, v3 quant offset/value,
// v4..v7 scale/min packing scratch, then F32 decode temporaries. Eight VGPRs.
// Scales/minima become F32 before multiply; no FMA or integer signed shortcut.
.text
.p2align 8
.globl q4_k_row_gfx1151
.type q4_k_row_gfx1151,@function
q4_k_row_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 79
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 5120
    s_cbranch_scc1 .Lend
    s_and_b32 s15, s12, 255
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s13, 1
    s_cbranch_scc0 .Lend
    s_lshr_b32 s15, s12, 8
    s_mul_i32 s15, s15, 144
    s_cmp_eq_u32 s2, s15
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s6, 4
    s_cbranch_scc0 .Lend
    s_lshl_b32 s15, s12, 2
    s_cmp_eq_u32 s10, s15
    s_cbranch_scc0 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s12, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_lshlrev_b32 v1, 2, v0
    s_nop 7
    v_lshrrev_b32 v2, 8, v0
    s_nop 7
    v_mul_lo_u32 v2, 144, v2
    s_nop 7
    v_and_b32 v3, 31, v0
    s_nop 7
    v_lshrrev_b32 v0, 5, v0
    s_nop 7
    v_and_b32 v0, 7, v0
    s_nop 7
    v_lshrrev_b32 v4, 1, v0
    s_nop 7
    v_lshlrev_b32 v4, 5, v4
    s_nop 7
    v_add_nc_u32 v3, v4, v3
    s_nop 7
    v_add_nc_u32 v3, v2, v3
    s_nop 7
    v_add_nc_u32 v3, 16, v3
    s_nop 7
    // Read header bytes j,4+j,8+j. For j<4, byte j belongs to the half scales;
    // that in-bounds byte is discarded by the select below, never interpreted.
    v_add_nc_u32 v4, v0, v2
    s_nop 7
    buffer_load_ubyte v5, v4, s[0:3], 0 offen
    buffer_load_ubyte v6, v4, s[0:3], 0 offen offset:8
    buffer_load_ubyte v4, v4, s[0:3], 0 offen offset:4
    s_waitcnt vmcnt(0)
    v_lshrrev_b32 v5, 2, v5
    s_nop 7
    v_and_b32 v5, 48, v5
    s_nop 7
    v_and_b32 v7, 15, v6
    s_nop 7
    v_or_b32 v5, v7, v5
    s_nop 7
    v_and_b32 v7, 63, v4
    s_nop 7
    v_cmp_gt_u32_e32 vcc, 4, v0
    s_nop 7
    v_cndmask_b32_e32 v5, v5, v7, vcc
    s_nop 7
    v_lshrrev_b32 v4, 2, v4
    s_nop 7
    v_and_b32 v4, 48, v4
    s_nop 7
    v_lshrrev_b32 v7, 4, v6
    s_nop 7
    v_or_b32 v4, v7, v4
    s_nop 7
    v_and_b32 v6, 63, v6
    s_nop 7
    v_cndmask_b32_e32 v7, v4, v6, vcc
    s_nop 7
    buffer_load_ushort v4, v2, s[0:3], 0 offen
    buffer_load_ushort v6, v2, s[0:3], 0 offen offset:2
    buffer_load_ubyte v3, v3, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_and_b32 v0, 1, v0
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    v_lshrrev_b32 v3, v0, v3
    s_nop 7
    v_and_b32 v3, 15, v3
    s_nop 7
    v_cvt_f32_u32 v3, v3
    s_nop 7
    v_cvt_f32_u32 v5, v5
    s_nop 7
    v_cvt_f32_u32 v7, v7
    s_nop 7
    v_cvt_f32_f16 v4, v4
    s_nop 7
    v_cvt_f32_f16 v6, v6
    s_nop 7
    v_mul_f32 v4, v4, v5
    s_nop 7
    v_mul_f32 v6, v6, v7
    s_nop 7
    v_mul_f32 v3, v4, v3
    s_nop 7
    v_sub_f32 v3, v3, v6
    s_nop 7
    buffer_store_dword v3, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size q4_k_row_gfx1151, .-q4_k_row_gfx1151
.rept 64
    s_code_end
.endr
