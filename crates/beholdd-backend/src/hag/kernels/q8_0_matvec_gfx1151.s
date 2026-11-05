// SPDX-License-Identifier: Apache-2.0
// Original bounded GFX1151 proof kernel; no model weights embedded.
// ABI and required register state are documented in README.md.
// Q8_0 block: little-endian f16 scale, then 32 signed i8 weights (34 bytes).
// Up to 64 wave64 workgroups: global lane is one row; decoded = scale*quant, then
// product = decoded*input and accumulator = accumulator+product, without FMA.
.text
.p2align 8
.globl q8_0_matvec_gfx1151
.type q8_0_matvec_gfx1151,@function
q8_0_matvec_gfx1151:
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
    s_lshr_b32 s15, s12, 5
    s_mul_i32 s15, s15, 34
    s_nop 7
    v_mul_lo_u32 v1, s15, v0
    s_nop 7
    v_lshlrev_b32 v2, 2, v0
    s_nop 7
    v_mov_b32 v3, 0
    s_mov_b32 s15, 0
    s_mov_b32 s16, 0
    s_mov_b32 s17, 0
.Lblock:
    s_nop 7
    buffer_load_ushort v6, v1, s[0:3], s15 offen
    s_waitcnt vmcnt(0)
    v_cvt_f32_f16 v6, v6
    s_nop 7
    s_add_u32 s19, s15, 2
    s_mov_b32 s18, 0
.Lcolumn:
    s_nop 7
    buffer_load_sbyte v4, v1, s[0:3], s19 offen
    buffer_load_dword v5, off, s[4:7], s17
    s_waitcnt vmcnt(0)
    v_cvt_f32_i32 v4, v4
    s_nop 7
    v_mul_f32 v4, v6, v4
    s_nop 7
    v_mul_f32 v4, v4, v5
    s_nop 7
    v_add_f32 v3, v3, v4
    s_nop 7
    s_add_u32 s16, s16, 1
    s_add_u32 s17, s17, 4
    s_add_u32 s18, s18, 1
    s_add_u32 s19, s19, 1
    s_cmp_lt_u32 s18, 32
    s_cbranch_scc1 .Lcolumn
    s_add_u32 s15, s15, 34
    s_cmp_lt_u32 s16, s12
    s_cbranch_scc1 .Lblock
    buffer_store_dword v3, v2, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size q8_0_matvec_gfx1151, .-q8_0_matvec_gfx1151
.rept 64
    s_code_end
.endr
