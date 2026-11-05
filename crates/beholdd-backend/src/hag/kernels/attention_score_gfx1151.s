// SPDX-License-Identifier: Apache-2.0
// Serial F64 Q*K scores. SRD0: contiguous Q head; SRD1: strided K prefix;
// SRD2: F64 scores. Scalars: [head_dim=1..256, kv_stride=head_dim..4096].
// The exact SRD2 byte extent encodes positions*8, positions=1..16384.
// Each lane owns one key position; host proves exact disjoint descriptor extents.
// v0 output offset, v1 K base, v[2:3] sum, v[4:5] Q/product, v[6:7] K.
.text
.p2align 8
.globl attention_score_gfx1151
.type attention_score_gfx1151,@function
attention_score_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 255
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 256
    s_cbranch_scc1 .Lend
    s_cmp_lt_u32 s13, s12
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s13, 4096
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s10, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s10, 131072
    s_cbranch_scc1 .Lend
    s_and_b32 s15, s10, 7
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend
    s_lshr_b32 s15, s10, 3
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s15, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_mul_lo_u32 v1, s13, v0
    s_nop 7
    v_lshlrev_b32 v1, 2, v1
    s_nop 7
    v_lshlrev_b32 v0, 3, v0
    s_nop 7
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
.Ldot:
    buffer_load_dword v4, off, s[0:3], s16
    buffer_load_dword v6, v1, s[4:7], s16 offen
    s_waitcnt vmcnt(0)
    v_cvt_f64_f32 v[4:5], v4
    s_nop 7
    v_cvt_f64_f32 v[6:7], v6
    s_nop 7
    v_mul_f64 v[4:5], v[4:5], v[6:7]
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    s_add_u32 s15, s15, 1
    s_add_u32 s16, s16, 4
    s_cmp_lt_u32 s15, s12
    s_cbranch_scc1 .Ldot
    buffer_store_dwordx2 v[2:3], v0, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size attention_score_gfx1151, .-attention_score_gfx1151
.rept 64
    s_code_end
.endr
