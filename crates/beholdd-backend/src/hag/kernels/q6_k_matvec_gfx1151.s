// SPDX-License-Identifier: Apache-2.0
// UNQUALIFIED SOURCE: independent GGML Q6_K, 256 values / 210 packed bytes.
// SRD0 exact rows*row_bytes; SRD1 exact columns*4; SRD2 exact rows*4.
// Scalars [columns, rows]; 256..17408 columns, 1..4096 rows, wave64.
// v0 row, v1 packed-row base, v2 output byte, v3 accumulator,
// v4 low/code/product, v5 high/input, v6 block d, v7 signed effective scale.
// Scratch SGPRs s15..s28. Increasing-column F32 decode/product/add, no FMA.
// Half scales are loaded as ushort: 210-byte block/row alignment may be 2 mod4.
.text
.p2align 8
.globl q6_k_matvec_gfx1151
.type q6_k_matvec_gfx1151,@function
q6_k_matvec_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 63
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 17408
    s_cbranch_scc1 .Lend
    s_and_b32 s15, s12, 255
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s13, 4096
    s_cbranch_scc1 .Lend
    s_lshr_b32 s27, s12, 8
    s_mul_i32 s27, s27, 210
    s_mul_i32 s15, s27, s13
    s_cmp_eq_u32 s2, s15
    s_cbranch_scc0 .Lend
    s_lshl_b32 s28, s12, 2
    s_cmp_eq_u32 s6, s28
    s_cbranch_scc0 .Lend
    s_lshl_b32 s15, s13, 2
    s_cmp_eq_u32 s10, s15
    s_cbranch_scc0 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s13, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_mul_lo_u32 v1, s27, v0
    s_nop 7
    v_lshlrev_b32 v2, 2, v0
    s_nop 7
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
.Lblock:
    buffer_load_ushort v6, v1, s[0:3], s15 offen offset:208
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16 v6, v6
    s_nop 7
    s_mov_b32 s17, 0
.Lsubblock:
    // One signed scale per16 consecutive coordinates. Each128-value half has
    // four32-value groups selecting ql low/high nibble and successive qh pairs.
    s_add_u32 s19, s15, s17
    s_add_u32 s19, s19, 192
    buffer_load_sbyte v7, v1, s[0:3], s19 offen
    s_lshr_b32 s20, s17, 3
    s_and_b32 s21, s17, 7
    s_lshr_b32 s21, s21, 1
    s_and_b32 s22, s17, 1
    s_lshl_b32 s22, s22, 4
    s_and_b32 s23, s21, 1
    s_lshl_b32 s23, s23, 5
    s_lshl_b32 s24, s20, 6
    s_add_u32 s23, s23, s24
    s_add_u32 s23, s23, s22
    s_add_u32 s23, s23, s15
    s_lshl_b32 s24, s20, 5
    s_add_u32 s24, s24, s22
    s_add_u32 s24, s24, s15
    s_add_u32 s24, s24, 128
    s_lshr_b32 s25, s21, 1
    s_lshl_b32 s25, s25, 2
    s_lshl_b32 s26, s21, 1
    s_waitcnt vmcnt(0)
    v_cvt_f32_i32 v7, v7
    s_nop 7
    v_mul_f32 v7, v6, v7
    s_nop 7
    s_mov_b32 s18, 0
.Lcolumn:
    buffer_load_ubyte v4, v1, s[0:3], s23 offen
    buffer_load_ubyte v5, v1, s[0:3], s24 offen
    s_waitcnt vmcnt(0)
    v_lshrrev_b32 v4, s25, v4
    s_nop 7
    v_and_b32 v4, 15, v4
    s_nop 7
    v_lshrrev_b32 v5, s26, v5
    s_nop 7
    v_and_b32 v5, 3, v5
    s_nop 7
    v_lshlrev_b32 v5, 4, v5
    s_nop 7
    v_or_b32 v4, v5, v4
    s_nop 7
    v_add_nc_u32 v4, -32, v4
    s_nop 7
    v_cvt_f32_i32 v4, v4
    s_nop 7
    v_mul_f32 v4, v7, v4
    s_nop 7
    buffer_load_dword v5, off, s[4:7], s16
    s_waitcnt vmcnt(0)
    v_mul_f32 v4, v4, v5
    s_nop 7
    v_add_f32 v3, v3, v4
    s_nop 7
    s_add_u32 s16, s16, 4
    s_add_u32 s23, s23, 1
    s_add_u32 s24, s24, 1
    s_add_u32 s18, s18, 1
    s_cmp_lt_u32 s18, 16
    s_cbranch_scc1 .Lcolumn
    s_add_u32 s17, s17, 1
    s_cmp_lt_u32 s17, 16
    s_cbranch_scc1 .Lsubblock
    s_add_u32 s15, s15, 210
    s_cmp_lt_u32 s16, s28
    s_cbranch_scc1 .Lblock
    buffer_store_dword v3, v2, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size q6_k_matvec_gfx1151, .-q6_k_matvec_gfx1151
.rept 64
    s_code_end
.endr
