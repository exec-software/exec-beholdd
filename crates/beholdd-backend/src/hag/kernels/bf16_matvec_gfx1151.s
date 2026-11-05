// SPDX-License-Identifier: Apache-2.0
// Original bounded GFX1151 proof kernel; no model weights embedded.
// ABI and required register state are documented in README.md.
// Up to 64 wave64 workgroups: global lane is one row; serial column arithmetic.
// NOPs deliberately favor conservative dependency spacing over performance.
.text
.p2align 8
.globl bf16_matvec_gfx1151
.type bf16_matvec_gfx1151,@function
bf16_matvec_gfx1151:
    s_version 0x2006
    // Defensive geometry checks complement, but do not replace, host validation.
    s_cmp_gt_u32 s14, 63
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 4096
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s13, 4096
    s_cbranch_scc1 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s13, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_mul_lo_u32 v1, s12, v0
    s_nop 7
    v_lshlrev_b32 v1, 1, v1
    s_nop 7
    v_lshlrev_b32 v2, 2, v0
    s_nop 7
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
    s_mov_b32 s17, 0
.Lcolumn:
    s_nop 7
    // Byte offsets: row*cols*2 + column*2, and column*4.
    buffer_load_ushort v4, v1, s[0:3], s16 offen
    buffer_load_dword v5, off, s[4:7], s17
    s_waitcnt vmcnt(0)
    v_lshlrev_b32 v4, 16, v4
    s_nop 7
    v_mul_f32 v4, v4, v5
    s_nop 7
    v_add_f32 v3, v3, v4
    s_nop 7
    s_add_u32 s15, s15, 1
    s_add_u32 s16, s16, 2
    s_add_u32 s17, s17, 4
    s_cmp_lt_u32 s15, s12
    s_cbranch_scc1 .Lcolumn
    buffer_store_dword v3, v2, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size bf16_matvec_gfx1151, .-bf16_matvec_gfx1151
// Preserve safe instruction-fetch padding; not an extra executable work loop.
.rept 64
    s_code_end
.endr
