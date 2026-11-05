// SPDX-License-Identifier: Apache-2.0
// Bounded in-place RoPE. SRD0: cos[pairs]; SRD1: sin[pairs]; SRD2: x[2*pairs].
// The three exact descriptor extents must be disjoint. Scalars: [pairs, flags],
// pairs=1..2048, flags=1+split_half+2*inverse. Empty operations are host no-ops.
// Each lane owns one complete pair; both x values load before either store.
// v0/v1 byte offsets, v2/v3 original a/b, v4/v5 cos/signed sin, v6/v7 products.
// s15 decoded flags, s16 sign/offset scratch. All arithmetic is separate F32.
.text
.p2align 8
.globl rope_gfx1151
.type rope_gfx1151,@function
rope_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 31
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 2048
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s13, 4
    s_cbranch_scc1 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s12, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    buffer_load_dword v4, v0, s[0:3], 0 offen
    buffer_load_dword v5, v0, s[4:7], 0 offen
    s_waitcnt vmcnt(0)
    s_sub_u32 s15, s13, 1
    s_lshl_b32 s16, s15, 30
    s_and_b32 s16, s16, 0x80000000
    s_or_b32 s16, s16, 0x3f800000
    s_nop 7
    v_mul_f32 v5, s16, v5
    s_nop 7
    s_bitcmp1_b32 s15, 0
    s_cbranch_scc1 .Lsplit
    v_lshlrev_b32 v0, 1, v0
    s_nop 7
    v_add_nc_u32 v1, 4, v0
    s_nop 7
    s_branch .Lload
.Lsplit:
    s_lshl_b32 s16, s12, 2
    s_nop 7
    v_add_nc_u32 v1, s16, v0
    s_nop 7
.Lload:
    buffer_load_dword v2, v0, s[8:11], 0 offen
    buffer_load_dword v3, v1, s[8:11], 0 offen
    s_waitcnt vmcnt(0)
    v_mul_f32 v6, v2, v4
    s_nop 7
    v_mul_f32 v7, v3, v5
    s_nop 7
    v_sub_f32 v6, v6, v7
    s_nop 7
    v_mul_f32 v2, v2, v5
    s_nop 7
    v_mul_f32 v3, v3, v4
    s_nop 7
    v_add_f32 v2, v2, v3
    s_nop 7
    buffer_store_dword v6, v0, s[8:11], 0 offen
    buffer_store_dword v2, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size rope_gfx1151, .-rope_gfx1151
.rept 64
    s_code_end
.endr
