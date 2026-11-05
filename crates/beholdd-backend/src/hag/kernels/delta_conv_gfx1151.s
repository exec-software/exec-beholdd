// SPDX-License-Identifier: Apache-2.0
// Experimental slot12; assembled bytes do not establish GPU qualification.
// Include path must contain this directory. docs/qwen35-delta-packet.md packet v2.
// v0 projection byte offset, v1 history-relative offset, v2 weight-relative
// offset, v3 sample, v4 weight, v5 sum/result, v6/v7 scratch. Eight VGPRs.
// Every invalid lane leaves convolved/core qNaN. Raw history writes target only
// the unpublished inactive bank. All pending markers precede arithmetic masks.
.include "delta_packet_gfx1151.inc"
.text
.p2align 8
.globl delta_conv_gfx1151
.type delta_conv_gfx1151,@function
delta_conv_gfx1151:
    s_version 0x2006
    DELTA_GEOMETRY
    s_add_u32 s29, s22, 63
    s_lshr_b32 s29, s29, 6
    s_cmp_ge_u32 s14, s29
    s_cbranch_scc1 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_lshl_add_u32 v0, s14, 6, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s22, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_lshlrev_b32 v1, 2, v0
    s_nop 7
    v_mov_b32 v5, 0x7fc00000
    // Every active lane owns its convolved marker, even on invalid raw input.
    buffer_store_dword v5, v1, s[8:11], s26 offen
    s_waitcnt_vscnt null, 0
    // The first V lanes own all core markers; restore the complete channel mask.
    v_cmp_gt_u32_e32 vcc, s21, v0
    s_nop 7
    s_and_saveexec_b64 s[30:31], vcc
    s_nop 7
    buffer_store_dword v5, v1, s[8:11], s27 offen
    s_waitcnt_vscnt null, 0
    s_mov_b64 exec, s[30:31]
    s_nop 7
    // The first Hv lanes own the raw-u32 status markers.
    v_cmp_gt_u32_e32 vcc, s17, v0
    s_nop 7
    s_and_saveexec_b64 s[30:31], vcc
    s_nop 7
    v_mov_b32 v5, 0xffffffff
    buffer_store_dword v5, v1, s[8:11], s28 offen
    s_waitcnt_vscnt null, 0
    s_mov_b64 exec, s[30:31]
    s_nop 7
    v_mul_lo_u32 v1, 12, v0
    s_nop 7
    v_lshlrev_b32 v2, 4, v0
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    v_mov_b32 v5, 0

    // Tap0: oldest retained raw value.
    buffer_load_dword v3, v1, s[8:11], s24 offen
    buffer_load_dword v4, v2, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    DELTA_FINITE v3, v6
    DELTA_FINITE v4, v6
    v_mul_f32 v6, v3, v4
    s_nop 7
    DELTA_FINITE v6, v7
    v_add_f32 v5, v5, v6
    s_nop 7
    DELTA_FINITE v5, v7

    // Tap1 also becomes the next bank's oldest retained raw value.
    buffer_load_dword v3, v1, s[8:11], s24 offen offset:4
    buffer_load_dword v4, v2, s[0:3], 0 offen offset:4
    s_waitcnt vmcnt(0)
    DELTA_FINITE v3, v6
    DELTA_FINITE v4, v6
    buffer_store_dword v3, v1, s[8:11], s25 offen
    s_waitcnt_vscnt null, 0
    v_mul_f32 v6, v3, v4
    s_nop 7
    DELTA_FINITE v6, v7
    v_add_f32 v5, v5, v6
    s_nop 7
    DELTA_FINITE v5, v7

    // Tap2 becomes the next bank's middle retained raw value.
    buffer_load_dword v3, v1, s[8:11], s24 offen offset:8
    buffer_load_dword v4, v2, s[0:3], 0 offen offset:8
    s_waitcnt vmcnt(0)
    DELTA_FINITE v3, v6
    DELTA_FINITE v4, v6
    buffer_store_dword v3, v1, s[8:11], s25 offen offset:4
    s_waitcnt_vscnt null, 0
    v_mul_f32 v6, v3, v4
    s_nop 7
    DELTA_FINITE v6, v7
    v_add_f32 v5, v5, v6
    s_nop 7
    DELTA_FINITE v5, v7

    // Tap3 is the current raw projection and next bank's newest retained value.
    buffer_load_dword v3, v0, s[4:7], 0 offen
    buffer_load_dword v4, v2, s[0:3], 0 offen offset:12
    s_waitcnt vmcnt(0)
    DELTA_FINITE v3, v6
    DELTA_FINITE v4, v6
    buffer_store_dword v3, v1, s[8:11], s25 offen offset:8
    s_waitcnt_vscnt null, 0
    v_mul_f32 v6, v3, v4
    s_nop 7
    DELTA_FINITE v6, v7
    v_add_f32 v5, v5, v6
    s_nop 7
    DELTA_FINITE v5, v7

    // Stable SiLU: t=exp(-abs(sum)); sigmoid=(sum>=0 ? 1 : t)/(1+t).
    // Range-limited scaled exp avoids intermediate overflow for finite extremes.
    DELTA_EXP_NEG_ABS v5, v6, v4, v7
    v_add_f32 v7, 1.0, v6
    s_nop 7
    v_rcp_f32 v7, v7
    s_nop 7
    v_cmp_gt_f32_e32 vcc, 0, v5
    s_nop 7
    v_cndmask_b32_e32 v6, 1.0, v6, vcc
    s_nop 7
    v_mul_f32 v6, v6, v7
    s_nop 7
    v_mul_f32 v5, v5, v6
    s_nop 7
    DELTA_FINITE v5, v7
    buffer_store_dword v5, v0, s[8:11], s26 offen
    s_waitcnt_vscnt null, 0
.Lend:
    // A masked arithmetic exit can follow earlier inactive-bank stores.
    s_waitcnt_vscnt null, 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size delta_conv_gfx1151, .-delta_conv_gfx1151
.rept 64
    s_code_end
.endr
