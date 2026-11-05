// SPDX-License-Identifier: Apache-2.0
// Original bounded GFX1151 elementwise proof kernel; no model data embedded.
// SRD0: one finite f32 limit; SRD1: up; SRD2: read-write gate, all disjoint.
// s12: 1..4096 elements; s13: 1; up to 64 wave64 workgroups; eight VGPR ABI.
// Positive limit: gate=min(gate,limit), up=clamp(up,-limit,limit).
// Computes (gate/(1+exp(-gate)))*up approximately, not bitwise CPU equivalence.
// Exp/reciprocal SFUs flush denormals. Denominator scaling avoids flushing a
// small reciprocal before multiplication by gate; see README numerical contract.
.text
.p2align 8
.globl swiglu_gfx1151
.type swiglu_gfx1151,@function
swiglu_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 63
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 4096
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 1
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
    buffer_load_dword v4, off, s[0:3], 0
    buffer_load_dword v3, v1, s[4:7], 0 offen
    buffer_load_dword v2, v1, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    // The parameter is uniform, so either every active lane clamps or none do.
    v_cmp_lt_f32_e32 vcc, 0, v4
    s_nop 7
    s_cbranch_vccz .Lunclamped
    // Ordered gate<limit selects gate, otherwise limit, including gate=NaN.
    v_cmp_lt_f32_e32 vcc, v2, v4
    s_nop 7
    v_cndmask_b32_e32 v2, v4, v2, vcc
    s_nop 7
    // Ordered compares leave up=NaN untouched, unlike a min/max sequence.
    v_xor_b32 v5, 0x80000000, v4
    s_nop 7
    v_cmp_lt_f32_e32 vcc, v3, v5
    s_nop 7
    v_cndmask_b32_e32 v3, v3, v5, vcc
    s_nop 7
    v_cmp_gt_f32_e32 vcc, v3, v4
    s_nop 7
    v_cndmask_b32_e32 v3, v3, v4, vcc
    s_nop 7
.Lunclamped:
    // 0xbfb8aa3b is correctly rounded -log2(e), not the base-e exponent.
    v_mul_f32 v5, 0xbfb8aa3b, v2
    s_nop 7
    v_exp_f32 v6, v5
    s_nop 7
    v_add_f32 v6, 1.0, v6
    s_nop 7
    // For large denominators use (gate*rcp(denominator*2^-32))*2^-32.
    // The scale is 1 otherwise. This keeps the SFU reciprocal normal even near
    // exp overflow; multiplying gate first avoids premature result underflow.
    v_mov_b32 v7, 0x2f800000
    s_nop 7
    v_cmp_lt_f32_e32 vcc, 0x6f800000, v6
    s_nop 7
    v_cndmask_b32_e32 v7, 1.0, v7, vcc
    s_nop 7
    v_mul_f32 v6, v7, v6
    s_nop 7
    v_rcp_f32 v6, v6
    s_nop 7
    v_mul_f32 v2, v2, v6
    s_nop 7
    v_mul_f32 v2, v7, v2
    s_nop 7
    v_mul_f32 v2, v2, v3
    s_nop 7
    buffer_store_dword v2, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size swiglu_gfx1151, .-swiglu_gfx1151
.rept 64
    s_code_end
.endr
