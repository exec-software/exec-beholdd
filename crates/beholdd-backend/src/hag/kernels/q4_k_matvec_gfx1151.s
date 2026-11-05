// SPDX-License-Identifier: Apache-2.0
// UNQUALIFIED SOURCE: independent GGML Q4_K, 256 values / 144 packed bytes.
// SRD0 exact rows*row_bytes; SRD1 exact columns*4; SRD2 exact rows*4.
// Scalars [columns, rows]; 256..17408 columns, 1..4096 rows, wave64.
// v0 row, v1 packed-row base, v2 output byte, v3 accumulator,
// v4/v5 scratch then quant/input, v6 effective scale, v7 effective minimum.
// Scratch SGPRs s15..s25. Each lane accumulates increasing columns in F32:
// ((f32(d)*scale)*quant - f32(dmin)*minimum)*input, then add; no FMA.
.text
.p2align 8
.globl q4_k_matvec_gfx1151
.type q4_k_matvec_gfx1151,@function
q4_k_matvec_gfx1151:
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
    s_lshr_b32 s25, s12, 8
    s_mul_i32 s25, s25, 144
    s_mul_i32 s15, s25, s13
    s_cmp_eq_u32 s2, s15
    s_cbranch_scc0 .Lend
    s_lshl_b32 s24, s12, 2
    s_cmp_eq_u32 s6, s24
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
    v_mul_lo_u32 v1, s25, v0
    s_nop 7
    v_lshlrev_b32 v2, 2, v0
    s_nop 7
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s17, 0
.Lblock:
    s_mov_b32 s18, 0
.Lsubblock:
    // Subblock j uses low scale/min bits at bytes 4+j / 8+j for j<4.
    // For j>=4, byte 8+j packs low nibbles; bytes j and 4+j supply high bits.
    s_cmp_lt_u32 s18, 4
    s_cbranch_scc0 .Lhigh_scales
    s_add_u32 s21, s15, s18
    s_add_u32 s21, s21, 4
    s_add_u32 s22, s21, 4
    buffer_load_ubyte v5, v1, s[0:3], s21 offen
    buffer_load_ubyte v4, v1, s[0:3], s22 offen
    s_waitcnt vmcnt(0)
    v_and_b32 v5, 63, v5
    s_nop 7
    v_and_b32 v4, 63, v4
    s_nop 7
    s_branch .Lscale_ready
.Lhigh_scales:
    s_add_u32 s21, s15, s18
    s_add_u32 s22, s21, 8
    buffer_load_ubyte v5, v1, s[0:3], s21 offen
    buffer_load_ubyte v4, v1, s[0:3], s22 offen
    s_add_u32 s22, s21, 4
    buffer_load_ubyte v6, v1, s[0:3], s22 offen
    s_waitcnt vmcnt(0)
    v_and_b32 v7, 15, v4
    s_nop 7
    v_lshrrev_b32 v5, 2, v5
    s_nop 7
    v_and_b32 v5, 48, v5
    s_nop 7
    v_or_b32 v5, v7, v5
    s_nop 7
    v_lshrrev_b32 v4, 4, v4
    s_nop 7
    v_lshrrev_b32 v6, 2, v6
    s_nop 7
    v_and_b32 v6, 48, v6
    s_nop 7
    v_or_b32 v4, v6, v4
    s_nop 7
.Lscale_ready:
    buffer_load_ushort v6, v1, s[0:3], s15 offen
    buffer_load_ushort v7, v1, s[0:3], s15 offen offset:2
    s_waitcnt vmcnt(0)
    v_cvt_f32_u32 v5, v5
    s_nop 7
    v_cvt_f32_u32 v4, v4
    s_nop 7
    v_cvt_f32_f16 v6, v6
    s_nop 7
    v_cvt_f32_f16 v7, v7
    s_nop 7
    v_mul_f32 v6, v6, v5
    s_nop 7
    v_mul_f32 v7, v7, v4
    s_nop 7
    s_lshr_b32 s20, s18, 1
    s_lshl_b32 s20, s20, 5
    s_add_u32 s20, s20, s15
    s_add_u32 s20, s20, 16
    s_and_b32 s23, s18, 1
    s_lshl_b32 s23, s23, 2
    s_mov_b32 s19, 0
.Lcolumn:
    buffer_load_ubyte v4, v1, s[0:3], s20 offen
    buffer_load_dword v5, off, s[4:7], s17
    s_waitcnt vmcnt(0)
    v_lshrrev_b32 v4, s23, v4
    s_nop 7
    v_and_b32 v4, 15, v4
    s_nop 7
    v_cvt_f32_u32 v4, v4
    s_nop 7
    v_mul_f32 v4, v6, v4
    s_nop 7
    v_sub_f32 v4, v4, v7
    s_nop 7
    v_mul_f32 v4, v4, v5
    s_nop 7
    v_add_f32 v3, v3, v4
    s_nop 7
    s_add_u32 s17, s17, 4
    s_add_u32 s20, s20, 1
    s_add_u32 s19, s19, 1
    s_cmp_lt_u32 s19, 32
    s_cbranch_scc1 .Lcolumn
    s_add_u32 s18, s18, 1
    s_cmp_lt_u32 s18, 8
    s_cbranch_scc1 .Lsubblock
    s_add_u32 s15, s15, 144
    s_cmp_lt_u32 s17, s24
    s_cbranch_scc1 .Lblock
    buffer_store_dword v3, v2, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size q4_k_matvec_gfx1151, .-q4_k_matvec_gfx1151
.rept 64
    s_code_end
.endr
