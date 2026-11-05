// SPDX-License-Identifier: Apache-2.0
// Original bounded GFX1151 Q8_0 row decoder; no model weights embedded.
// SRD0 exact packed row, SRD1 retained four-byte dummy, SRD2 exact F32 output.
// Scalars [columns, 1]; each active lane decodes one column. The host selects
// the row and supplies pairwise disjoint descriptor extents. SRD1 is not read.
.text
.p2align 8
.globl q8_0_row_gfx1151
.type q8_0_row_gfx1151,@function
q8_0_row_gfx1151:
    s_version 0x2006
    s_cmp_gt_u32 s14, 63
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 4096
    s_cbranch_scc1 .Lend
    s_and_b32 s15, s12, 31
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend
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
    v_lshrrev_b32 v2, 5, v0
    s_nop 7
    v_mul_lo_u32 v2, 34, v2
    s_nop 7
    v_and_b32 v3, 31, v0
    s_nop 7
    v_add_nc_u32 v3, v2, v3
    s_nop 7
    buffer_load_ushort v4, v2, s[0:3], 0 offen
    buffer_load_sbyte v5, v3, s[0:3], 0 offen offset:2
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16 v4, v4
    v_cvt_f32_i32 v5, v5
    s_nop 7
    v_mul_f32 v5, v4, v5
    s_nop 7
    buffer_store_dword v5, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size q8_0_row_gfx1151, .-q8_0_row_gfx1151
.rept 64
    s_code_end
.endr
