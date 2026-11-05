// SPDX-License-Identifier: Apache-2.0
// Bounded unweighted RMSNorm proof. SRD0: one finite nonnegative f32 epsilon;
// SRD1: f32 input; SRD2: disjoint f32 output. Scalars: [1..4096 elements, 1].
// Every active lane independently reduces the complete input in increasing order
// using f64 conversion, multiplication and addition. F64 reciprocal/rsqrt are
// approximate; round their final inverse to f32 before multiplying the input.
// v0 index, v1 output byte offset, v[2:3] sum/inverse, v[4:5] f64 scratch,
// v6 f32 load; s15/s16 loop index/byte offset. No LDS, barriers or FMA.
.text
.p2align 8
.globl rmsnorm_gfx1151
.type rmsnorm_gfx1151,@function
rmsnorm_gfx1151:
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
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
.Lsum:
    buffer_load_dword v6, off, s[4:7], s16
    s_waitcnt vmcnt(0)
    v_cvt_f64_f32 v[4:5], v6
    s_nop 7
    v_mul_f64 v[4:5], v[4:5], v[4:5]
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    s_add_u32 s15, s15, 1
    s_add_u32 s16, s16, 4
    s_cmp_lt_u32 s15, s12
    s_cbranch_scc1 .Lsum
    v_cvt_f64_u32 v[4:5], s12
    s_nop 7
    v_rcp_f64 v[4:5], v[4:5]
    s_nop 7
    v_mul_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    buffer_load_dword v6, off, s[0:3], 0
    s_waitcnt vmcnt(0)
    v_cvt_f64_f32 v[4:5], v6
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    v_rsq_f64 v[2:3], v[2:3]
    s_nop 7
    v_cvt_f32_f64 v2, v[2:3]
    s_nop 7
    buffer_load_dword v6, v1, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_mul_f32 v2, v6, v2
    s_nop 7
    buffer_store_dword v2, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size rmsnorm_gfx1151, .-rmsnorm_gfx1151
.rept 64
    s_code_end
.endr
