// SPDX-License-Identifier: Apache-2.0
// Experimental Delta packet v2, program14; GPU qualification is separate.
// Exact three-SRD/scalar contract: docs/qwen35-delta-packet.md. One lane owns one value row.
// Prediction and readout use F32 products, serial F64 sums, F32 reductions.
// Pass 2 recomputes ACTIVE state * rho, then writes only INACTIVE state.
// Slot12 must have initialized every core word to canonical qNaN. A failed
// lane keeps that marker; host publication waits for all slot15 status words.
// No FMA, LDS, barriers, atomics, cross-lane writes, or CPU fallback.
// v0 address/loaded key or query; v1 immutable row-state byte offset;
// v2:3 sum; v4:5 F64 conversion; v6 state/product; v7 rho then delta.
// Post-admission SGPRs: s15 logD, s16 4D, s17 Hk-1, s18 temporary,
// s19 logD+2, s20 2logD+2, s21 convolved-V base, s23 active-M base,
// s24 inactive-M base, s25 normQ, s26 normK, s27 rho, s28 beta,
// s29 core, s30 loop byte, s31 temporary. s22 is dead after setup.
.macro rec_finite reg
    v_cmp_class_f32_e64 vcc, \reg, 0x1f8
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Ldone
.endm
.text
.p2align 8
.globl delta_recurrence_gfx1151
.type delta_recurrence_gfx1151,@function
delta_recurrence_gfx1151:
    s_version 0x2006
    // Scalar validation precedes every load/store and every derived shift.
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 1
    s_cbranch_scc0 .Lend
    s_sub_u32 s12, s12, 1       // positive transport scalar -> packed geometry
    s_and_b32 s15, s12, 0xffffe000
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend
    s_and_b32 s15, s12, 7
    s_lshl_b32 s16, 1, s15
    s_lshr_b32 s17, s12, 3
    s_and_b32 s17, s17, 7
    s_cmp_gt_u32 s17, 4
    s_cbranch_scc1 .Lend
    s_lshl_b32 s17, 1, s17
    s_lshr_b32 s18, s12, 6
    s_and_b32 s18, s18, 63
    s_add_u32 s18, s18, 1
    s_cmp_gt_u32 s18, 48
    s_cbranch_scc1 .Lend
    s_sub_u32 s24, s17, 1
    s_and_b32 s24, s24, s18
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lend
    s_mul_i32 s19, s17, s16
    s_lshl_b32 s19, s19, 2       // Q bytes
    s_mul_i32 s20, s18, s16
    s_lshl_b32 s20, s20, 2       // V bytes
    s_lshl_b32 s21, s19, 1
    s_add_u32 s21, s21, s20     // C bytes
    s_mul_i32 s23, s20, s16     // R bytes
    s_mul_i32 s22, s21, 3
    s_add_u32 s22, s22, s23
    s_add_u32 s22, s22, s20     // bank B bytes
    s_lshl_b32 s24, s21, 2
    s_lshl_b32 s25, s18, 3
    s_add_u32 s24, s24, s25
    s_lshl_b32 s25, s16, 2
    s_add_u32 s24, s24, s25
    s_add_u32 s24, s24, 4       // finite positive F32 epsilon trailer
    s_cmp_eq_u32 s2, s24        // exact parameter extent
    s_cbranch_scc0 .Lend
    s_add_u32 s24, s21, s20
    s_lshl_b32 s25, s18, 3
    s_add_u32 s24, s24, s25
    s_cmp_eq_u32 s6, s24        // exact projection extent
    s_cbranch_scc0 .Lend
    s_lshl_b32 s24, s22, 1
    s_add_u32 s24, s24, s21
    s_lshl_b32 s25, s19, 1
    s_add_u32 s24, s24, s25
    s_mul_i32 s25, s18, 12
    s_add_u32 s24, s24, s25
    s_add_u32 s24, s24, s20
    s_cmp_eq_u32 s10, s24       // exact frame extent
    s_cbranch_scc0 .Lend
    s_or_b32 s24, s0, s4
    s_or_b32 s24, s24, s8
    s_and_b32 s24, s24, 3
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lend
    s_add_u32 s24, s20, 255
    s_lshr_b32 s24, s24, 8      // ceil(V/64) groups
    s_cmp_ge_u32 s14, s24
    s_cbranch_scc1 .Lend
    s_sub_u32 s31, s2, 4
    buffer_load_dword v6, off, s[0:3], s31
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s13, v6
    s_nop 7
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lend
    s_cmp_ge_u32 s13, 0x7f800000
    s_cbranch_scc1 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s20, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_lshlrev_b32 v1, s15, v0   // n*D*4, immutable
    s_nop 7
    // Derive all frame bases without retaining additional VGPRs.
    s_lshr_b32 s23, s12, 12
    s_mul_i32 s23, s23, s22
    s_sub_u32 s24, s22, s23
    s_mul_i32 s31, s21, 3
    s_add_u32 s23, s23, s31
    s_add_u32 s24, s24, s31
    s_lshl_b32 s25, s22, 1
    s_add_u32 s25, s25, s21
    s_add_u32 s26, s25, s19
    s_add_u32 s27, s26, s19
    s_lshl_b32 s18, s18, 2
    s_add_u32 s28, s27, s18
    s_add_u32 s29, s28, s18
    s_lshl_b32 s21, s22, 1
    s_lshl_b32 s31, s19, 1
    s_add_u32 s21, s21, s31    // convolved V
    s_sub_u32 s17, s17, 1
    s_add_u32 s19, s15, 2
    s_lshl_b32 s20, s15, 1
    s_add_u32 s20, s20, 2
    s_lshl_b32 s16, s16, 2
    // h=(n*D*4)>>(2logD+2); every value head uses key h & (Hk-1).
    v_lshrrev_b32 v0, s20, v1
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    buffer_load_dword v7, v0, s[8:11], s27 offen
    s_waitcnt vmcnt(0)
    rec_finite v7
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    s_mov_b32 s30, 0
.Lpredict:
    s_add_u32 s31, s23, s30
    buffer_load_dword v6, v1, s[8:11], s31 offen
    s_waitcnt vmcnt(0)
    rec_finite v6
    v_mul_f32 v6, v6, v7
    s_nop 7
    rec_finite v6
    v_lshrrev_b32 v0, s20, v1
    s_nop 7
    v_and_b32 v0, s17, v0
    s_nop 7
    v_lshlrev_b32 v0, s19, v0
    s_nop 7
    s_add_u32 s31, s26, s30
    buffer_load_dword v0, v0, s[8:11], s31 offen
    s_waitcnt vmcnt(0)
    rec_finite v0
    v_mul_f32 v6, v6, v0
    s_nop 7
    rec_finite v6
    v_cvt_f64_f32 v[4:5], v6
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    s_add_u32 s30, s30, 4
    s_cmp_lt_u32 s30, s16
    s_cbranch_scc1 .Lpredict
    v_cvt_f32_f64 v2, v[2:3]
    s_nop 7
    rec_finite v2
    v_lshrrev_b32 v0, s15, v1   // n*4
    s_nop 7
    buffer_load_dword v6, v0, s[8:11], s21 offen
    s_waitcnt vmcnt(0)
    rec_finite v6
    v_sub_f32 v6, v6, v2
    s_nop 7
    rec_finite v6
    v_lshrrev_b32 v0, s20, v1
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    buffer_load_dword v7, v0, s[8:11], s28 offen
    s_waitcnt vmcnt(0)
    rec_finite v7
    v_mul_f32 v7, v6, v7        // delta
    s_nop 7
    rec_finite v7
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    s_mov_b32 s30, 0
.Lupdate:
    s_add_u32 s31, s23, s30
    buffer_load_dword v6, v1, s[8:11], s31 offen
    s_waitcnt vmcnt(0)
    // Recompute the exact F32 decayed active state; never read partial next state.
    v_lshrrev_b32 v0, s20, v1
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    buffer_load_dword v0, v0, s[8:11], s27 offen
    s_waitcnt vmcnt(0)
    v_mul_f32 v6, v6, v0
    s_nop 7
    rec_finite v6
    v_lshrrev_b32 v0, s20, v1
    s_nop 7
    v_and_b32 v0, s17, v0
    s_nop 7
    v_lshlrev_b32 v0, s19, v0
    s_nop 7
    s_add_u32 s31, s26, s30
    buffer_load_dword v0, v0, s[8:11], s31 offen
    s_waitcnt vmcnt(0)
    v_mul_f32 v0, v0, v7
    s_nop 7
    rec_finite v0
    v_add_f32 v6, v6, v0
    s_nop 7
    rec_finite v6
    s_add_u32 s31, s24, s30
    buffer_store_dword v6, v1, s[8:11], s31 offen
    // Complete the store before reusing its source VGPR for the query product.
    s_waitcnt_vscnt null, 0
    v_lshrrev_b32 v0, s20, v1
    s_nop 7
    v_and_b32 v0, s17, v0
    s_nop 7
    v_lshlrev_b32 v0, s19, v0
    s_nop 7
    s_add_u32 s31, s25, s30
    buffer_load_dword v0, v0, s[8:11], s31 offen
    s_waitcnt vmcnt(0)
    rec_finite v0
    v_mul_f32 v6, v6, v0
    s_nop 7
    rec_finite v6
    v_cvt_f64_f32 v[4:5], v6
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    s_add_u32 s30, s30, 4
    s_cmp_lt_u32 s30, s16
    s_cbranch_scc1 .Lupdate
    v_cvt_f32_f64 v2, v[2:3]
    s_nop 7
    rec_finite v2
    s_lshr_b32 s18, s16, 2
    v_cvt_f32_u32 v4, s18
    s_nop 7
    v_sqrt_f32 v4, v4
    s_nop 7
    rec_finite v4
    v_rcp_f32 v4, v4
    s_nop 7
    rec_finite v4
    v_mul_f32 v2, v2, v4
    s_nop 7
    rec_finite v2
    v_lshrrev_b32 v0, s15, v1
    s_nop 7
    buffer_store_dword v2, v0, s[8:11], s29 offen
.Ldone:
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size delta_recurrence_gfx1151, .-delta_recurrence_gfx1151
.rept 64
    s_code_end
.endr
