// SPDX-License-Identifier: Apache-2.0
// Resident pointwise weight application. SRD0: f32 norm weights; SRD1: normalized
// f32 input; SRD2: disjoint output. Scalars: [1..4096 elements, 1].
// Weights are loaded when dispatched. Multiply input by weight with one f32 op.
.text
.p2align 8
.globl multiply_gfx1151
.type multiply_gfx1151,@function
multiply_gfx1151:
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
    buffer_load_dword v2, v1, s[0:3], 0 offen
    buffer_load_dword v3, v1, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_mul_f32 v3, v3, v2
    s_nop 7
    buffer_store_dword v3, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size multiply_gfx1151, .-multiply_gfx1151
.rept 64
    s_code_end
.endr
