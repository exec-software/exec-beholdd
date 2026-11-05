// SPDX-License-Identifier: Apache-2.0
// Unqualified Qwen35 pointwise gate: output = input * sigmoid(gate).
// SRD0 gate, SRD1 input, SRD2 output; exact count*4 lengths, all disjoint.
// Scalars [1..4096 count, 1], wave64, groups<=64; eight VGPRs, no LDS/FMA.
// Exp/rcp are approximate. A scaled exp tail below -48 prevents premature SFU
// denormal flushing when a small sigmoid multiplies a large finite input.
// The tail first rounds exp2(-abs(gate)*log2(e)+64)*2^-64 to F32, then
// multiplies input. The preserving ABI retains subnormal inputs/results from
// ordinary multiplication; in this range the F32 denominator rounds to1.
.text
.p2align 8
.globl sigmoid_gate_qwen35_gfx1151
.type sigmoid_gate_qwen35_gfx1151,@function
sigmoid_gate_qwen35_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 63
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 4096
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 1
    s_cbranch_scc0 .Lend
    s_lshl_b32 s15, s12, 2
    s_cmp_eq_u32 s2, s15
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s6, s15
    s_cbranch_scc0 .Lend
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
    buffer_load_dword v2, v1, s[4:7], 0 offen
    buffer_load_dword v3, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    v_and_b32 v4, 0x7fffffff, v3
    s_nop 7
    v_mul_f32 v4, 0xbfb8aa3b, v4
    s_nop 7
    v_exp_f32 v5, v4
    s_nop 7
    v_add_f32 v6, 1.0, v5
    s_nop 7
    v_rcp_f32 v6, v6
    s_nop 7
    v_cmp_le_f32_e32 vcc, 0, v3
    s_nop 7
    v_mov_b32 v7, 1.0
    s_nop 7
    v_cndmask_b32_e32 v5, v5, v7, vcc
    s_nop 7
    v_mul_f32 v7, v5, v6
    s_nop 7
    v_mul_f32 v7, v7, v2
    s_nop 7
    v_add_f32 v4, 0x42800000, v4
    s_nop 7
    v_exp_f32 v4, v4
    s_nop 7
    v_mul_f32 v4, 0x1f800000, v4
    s_nop 7
    v_mul_f32 v4, v4, v2
    s_nop 7
    v_cmp_lt_f32_e32 vcc, 0xc2400000, v3
    s_nop 7
    // The comparison above is -48 < gate; choose the ordinary path when true.
    v_cndmask_b32_e32 v7, v4, v7, vcc
    s_nop 7
    buffer_store_dword v7, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size sigmoid_gate_qwen35_gfx1151, .-sigmoid_gate_qwen35_gfx1151
.rept 64
    s_code_end
.endr
