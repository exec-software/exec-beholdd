// SPDX-License-Identifier: Apache-2.0
// SRD0: one positive finite F32 scale; SRD1: F64 unscaled scores;
// SRD2: F32 probabilities. Scalars [positions=1..16384, 1], exactly one group.
// Each lane repeats serial max and F32 exp-sum passes, then writes lane+64*k.
// The reductions retain increasing-position order; only output ownership changes.
// Ordered max ignores NaNs; a nonfinite maximum writes zero. Nonfinite scores
// are replaced by the finite max before exp and zero-selected afterwards.
// v0 score offset, v1 output offset, v2 max, v3 sum, v[4:5] score/scratch,
// v[6:7] F64 scale. CPU F64 scaling -> F32 precedes all softmax arithmetic.
// s17 selects the all-zero path; s18 is the output-byte bound. Tail masks only
// shrink while advancing, and precede every next output score load/store.
.text
.p2align 8
.globl attention_softmax_gfx1151
.type attention_softmax_gfx1151,@function
attention_softmax_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 0
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 16384
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
    v_lshlrev_b32 v0, 3, v0
    s_nop 7
    buffer_load_dword v6, off, s[0:3], 0
    s_waitcnt vmcnt(0)
    v_cvt_f64_f32 v[6:7], v6
    s_nop 7
    s_mov_b32 s17, 0
    s_lshl_b32 s18, s12, 2
    v_mov_b32 v2, 0xff800000
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
.Lmax:
    buffer_load_dwordx2 v[4:5], off, s[4:7], s16
    s_waitcnt vmcnt(0)
    v_mul_f64 v[4:5], v[4:5], v[6:7]
    s_nop 7
    v_cvt_f32_f64 v4, v[4:5]
    s_nop 7
    v_cmp_gt_f32_e32 vcc, v4, v2
    s_nop 7
    v_cndmask_b32_e32 v2, v2, v4, vcc
    s_nop 7
    s_add_u32 s15, s15, 1
    s_add_u32 s16, s16, 8
    s_cmp_lt_u32 s15, s12
    s_cbranch_scc1 .Lmax
    v_and_b32 v5, 0x7f800000, v2
    s_nop 7
    v_cmp_eq_u32_e32 vcc, 0x7f800000, v5
    s_nop 7
    s_cbranch_vccnz .Lzero
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
.Lsum:
    buffer_load_dwordx2 v[4:5], off, s[4:7], s16
    s_waitcnt vmcnt(0)
    v_mul_f64 v[4:5], v[4:5], v[6:7]
    s_nop 7
    v_cvt_f32_f64 v4, v[4:5]
    s_nop 7
    v_and_b32 v5, 0x7f800000, v4
    s_nop 7
    v_cmp_ne_u32_e32 vcc, 0x7f800000, v5
    s_nop 7
    v_cndmask_b32_e32 v4, v2, v4, vcc
    s_nop 7
    v_sub_f32 v4, v4, v2
    s_nop 7
    v_mul_f32 v4, 0x3fb8aa3b, v4
    s_nop 7
    v_exp_f32 v4, v4
    s_nop 7
    v_cndmask_b32_e32 v4, 0, v4, vcc
    s_nop 7
    v_add_f32 v3, v3, v4
    s_nop 7
    s_add_u32 s15, s15, 1
    s_add_u32 s16, s16, 8
    s_cmp_lt_u32 s15, s12
    s_cbranch_scc1 .Lsum
.Loutput:
    buffer_load_dwordx2 v[4:5], v0, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    v_mul_f64 v[4:5], v[4:5], v[6:7]
    s_nop 7
    v_cvt_f32_f64 v4, v[4:5]
    s_nop 7
    v_and_b32 v5, 0x7f800000, v4
    s_nop 7
    v_cmp_ne_u32_e32 vcc, 0x7f800000, v5
    s_nop 7
    v_cndmask_b32_e32 v4, v2, v4, vcc
    s_nop 7
    v_sub_f32 v4, v4, v2
    s_nop 7
    v_mul_f32 v4, 0x3fb8aa3b, v4
    s_nop 7
    v_exp_f32 v4, v4
    s_nop 7
    v_cndmask_b32_e32 v4, 0, v4, vcc
    s_nop 7
    v_rcp_f32 v5, v3
    s_nop 7
    v_mul_f32 v4, v4, v5
    s_nop 7
    s_branch .Lstore
.Lzero:
    s_mov_b32 s17, 1
    v_mov_b32 v4, 0
    s_nop 7
.Lstore:
    buffer_store_dword v4, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
    v_add_nc_u32 v0, 512, v0
    s_nop 7
    v_add_nc_u32 v1, 256, v1
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s18, v1
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    s_cmp_eq_u32 s17, 1
    s_cbranch_scc1 .Lzero
    s_branch .Loutput
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size attention_softmax_gfx1151, .-attention_softmax_gfx1151
.rept 64
    s_code_end
.endr
