// SPDX-License-Identifier: Apache-2.0
// UNQUALIFIED SOURCE: GGML Q6_K selected-row decode, 256..5120 columns.
// SRD0 exact (columns/256)*210 bytes; SRD1 exact4-byte unread dummy;
// SRD2 exact columns*4 bytes. Scalars [columns,1]. Up to80 wave64 groups.
// v0 column, v1 output byte, v2 block base, v3/v4 indexing/high bits,
// v5 low bits/decoded result, v6 half scale, v7 signed scale. Eight VGPRs.
.text
.p2align 8
.globl q6_k_row_gfx1151
.type q6_k_row_gfx1151,@function
q6_k_row_gfx1151:
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
    s_mul_i32 s15, s15, 210
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
    v_mul_lo_u32 v2, 210, v2
    s_nop 7
    v_lshrrev_b32 v3, 4, v0
    s_nop 7
    v_and_b32 v3, 15, v3
    s_nop 7
    v_add_nc_u32 v3, v2, v3
    s_nop 7
    buffer_load_sbyte v7, v3, s[0:3], 0 offen offset:192
    buffer_load_ushort v6, v2, s[0:3], 0 offen offset:208
    s_waitcnt vmcnt(0)
    v_and_b32 v3, 31, v0
    s_nop 7
    v_lshrrev_b32 v4, 5, v0
    s_nop 7
    v_and_b32 v4, 1, v4
    s_nop 7
    v_lshlrev_b32 v4, 5, v4
    s_nop 7
    v_add_nc_u32 v3, v4, v3
    s_nop 7
    v_lshrrev_b32 v4, 7, v0
    s_nop 7
    v_and_b32 v4, 1, v4
    s_nop 7
    v_lshlrev_b32 v4, 6, v4
    s_nop 7
    v_add_nc_u32 v3, v4, v3
    s_nop 7
    v_add_nc_u32 v3, v2, v3
    s_nop 7
    buffer_load_ubyte v5, v3, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_and_b32 v3, 31, v0
    s_nop 7
    v_lshrrev_b32 v4, 7, v0
    s_nop 7
    v_and_b32 v4, 1, v4
    s_nop 7
    v_lshlrev_b32 v4, 5, v4
    s_nop 7
    v_add_nc_u32 v3, v4, v3
    s_nop 7
    v_add_nc_u32 v3, v2, v3
    s_nop 7
    buffer_load_ubyte v4, v3, s[0:3], 0 offen offset:128
    s_waitcnt vmcnt(0)
    v_lshrrev_b32 v3, 6, v0
    s_nop 7
    v_and_b32 v3, 1, v3
    s_nop 7
    v_lshlrev_b32 v3, 2, v3
    s_nop 7
    v_lshrrev_b32 v5, v3, v5
    s_nop 7
    v_and_b32 v5, 15, v5
    s_nop 7
    v_lshrrev_b32 v3, 5, v0
    s_nop 7
    v_and_b32 v3, 3, v3
    s_nop 7
    v_lshlrev_b32 v3, 1, v3
    s_nop 7
    v_lshrrev_b32 v4, v3, v4
    s_nop 7
    v_and_b32 v4, 3, v4
    s_nop 7
    v_lshlrev_b32 v4, 4, v4
    s_nop 7
    v_or_b32 v5, v4, v5
    s_nop 7
    v_add_nc_u32 v5, -32, v5
    s_nop 7
    v_cvt_f32_i32 v5, v5
    s_nop 7
    v_cvt_f32_i32 v7, v7
    s_nop 7
    v_cvt_f32_f16 v6, v6
    s_nop 7
    v_mul_f32 v7, v6, v7
    s_nop 7
    v_mul_f32 v5, v7, v5
    s_nop 7
    buffer_store_dword v5, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size q6_k_row_gfx1151, .-q6_k_row_gfx1151
.rept 64
    s_code_end
.endr
