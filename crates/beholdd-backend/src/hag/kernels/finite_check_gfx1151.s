// SPDX-License-Identifier: Apache-2.0
// Explicit slot17 finite-status diagnostic; serving qualification is separate.
// Exactly one wave64: each lane checks lane,lane+64,... below count.
// SRD0 read-only input count*4 bytes; SRD1 private dummy4 bytes (unused);
// SRD2 private64-word U32 status256 bytes. All descriptors disjoint/aligned.
// Scalars[count in1..65536,1]. Finite includes signedzero and subnormals.
// Restore the incoming exec mask after every partial tail before publishing all
// 64 statuses; lanes without input publish SUCCESS0. No activation stores.
// ABI v0..v7, s0..s14; scratch s15..s18, v0..v5. No LDS/atomics/FMA.
.text
.p2align 8
.globl finite_check_gfx1151
.type finite_check_gfx1151,@function
finite_check_gfx1151:
    s_version 0x2006
    s_cmp_eq_u32 s14, 0
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_gt_u32 s12, 65536
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 1
    s_cbranch_scc0 .Lend
    s_lshl_b32 s15, s12, 2
    s_cmp_eq_u32 s2, s15
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s6, 4
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s10, 256
    s_cbranch_scc0 .Lend
    s_or_b32 s15, s0, s4
    s_or_b32 s15, s15, s8
    s_and_b32 s15, s15, 3
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend

    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, 64, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    s_mov_b64 s[16:17], exec
    v_mov_b32 v3, 0
    s_nop 7
    s_mov_b32 s18, 0
.Lscan:
    v_add_nc_u32 v4, s18, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s12, v4
    s_nop 7
    s_and_b64 exec, s[16:17], vcc
    s_nop 7
    s_cbranch_execz .Lrestore
    v_lshlrev_b32 v1, 2, v4
    s_nop 7
    buffer_load_dword v2, v1, s[0:3], 0 offen
    s_waitcnt vmcnt(0)
    // Exponent255 is exactly Inf/NaN for both signs; every other bit pattern
    // is finite, including both signed zeros and every subnormal.
    v_and_b32 v4, 0x7f800000, v2
    s_nop 7
    v_cmp_eq_u32_e32 vcc, 0x7f800000, v4
    s_nop 7
    v_mov_b32 v5, 1
    s_nop 7
    v_cndmask_b32_e32 v5, 0, v5, vcc
    s_nop 7
    v_or_b32 v3, v3, v5
    s_nop 7
.Lrestore:
    s_mov_b64 exec, s[16:17]
    s_nop 7
    s_add_u32 s18, s18, 64
    s_cmp_lt_u32 s18, s12
    s_cbranch_scc1 .Lscan
    // This executes with the complete original64-lane mask even when count<64
    // or count%64!=0. No stale status from a prior check can survive a valid run.
    v_lshlrev_b32 v1, 2, v0
    s_nop 7
    buffer_store_dword v3, v1, s[8:11], 0 offen
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size finite_check_gfx1151, .-finite_check_gfx1151
.rept 64
    s_code_end
.endr
