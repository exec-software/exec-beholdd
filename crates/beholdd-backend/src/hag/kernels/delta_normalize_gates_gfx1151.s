// SPDX-License-Identifier: Apache-2.0
// Experimental slot13; assembled bytes do not establish GPU qualification.
// docs/qwen35-delta-packet.md v2. Exactly2 groups: group0 owns Q/K normalization heads,
// group1 owns rho/beta pairs. Roles are uniform per wave, with no cross-wave
// dependency. Output markers are written before arithmetic can mask a lane.
// Norm: v1=head-relative bytebase, v0 address, v2:3 F64 sum (then scale/dscale),
// v4:5 F64 conversion, v6 input/product, v7 check scratch. Gate: v0=headbytes,
// v1=alpha then rho, v2=beta input, v3=A, v4..7 transient. Eight VGPRs.
.include "delta_packet_gfx1151.inc"
.text
.p2align 8
.globl delta_normalize_gates_gfx1151
.type delta_normalize_gates_gfx1151,@function
delta_normalize_gates_gfx1151:
    s_version 0x2006
    DELTA_GEOMETRY
    s_cmp_gt_u32 s14, 1
    s_cbranch_scc1 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    s_cmp_eq_u32 s14, 1
    s_cbranch_scc1 .Lgates

    // Group0: contiguous Q then K head indices, at most32 lanes.
    s_lshl_b32 s29, s19, 1
    v_cmp_gt_u32_e32 vcc, s29, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    s_add_u32 s29, s15, 2
    v_lshlrev_b32 v1, s29, v0
    s_nop 7
    s_lshl_b32 s31, s22, 2
    s_add_u32 s31, s31, s26
    s_lshl_b32 s20, s18, 2
    // First initialize every normalized coordinate to NaN. No later early exit
    // can leave an older step's normalized values visible to recurrence.
    s_mov_b32 s29, 0
    v_mov_b32 v6, 0x7fc00000
.Lnorm_initialize:
    v_add_nc_u32 v0, s29, v1
    s_nop 7
    buffer_store_dword v6, v0, s[8:11], s31 offen
    s_waitcnt_vscnt null, 0
    s_add_u32 s29, s29, 4
    s_cmp_lt_u32 s29, s20
    s_cbranch_scc1 .Lnorm_initialize
    s_waitcnt_vscnt null, 0
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    s_mov_b32 s29, 0
.Lnorm_sum:
    v_add_nc_u32 v0, s29, v1
    s_nop 7
    buffer_load_dword v6, v0, s[8:11], s26 offen
    s_waitcnt vmcnt(0)
    DELTA_FINITE v6, v7
    v_mul_f32 v6, v6, v6
    s_nop 7
    DELTA_FINITE v6, v7
    v_cvt_f64_f32 v[4:5], v6
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    s_add_u32 s29, s29, 4
    s_cmp_lt_u32 s29, s20
    s_cbranch_scc1 .Lnorm_sum
    // Exact power-of-two division in F64, then F32 mean and epsilon/D.
    s_sub_u32 s30, 127, s15
    s_lshl_b32 s30, s30, 23
    v_cvt_f64_f32 v[4:5], s30
    s_nop 7
    v_mul_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    v_cvt_f32_f64 v2, v[2:3]
    s_nop 7
    v_mov_b32 v6, s13
    s_nop 7
    v_mul_f32 v6, s30, v6
    s_nop 7
    v_add_f32 v2, v2, v6
    s_nop 7
    DELTA_FINITE v2, v7
    v_sqrt_f32 v2, v2
    s_nop 7
    v_rcp_f32 v2, v2
    s_nop 7
    DELTA_FINITE v2, v7
    v_cvt_f32_u32 v3, s18
    s_nop 7
    v_sqrt_f32 v3, v3
    s_nop 7
    v_rcp_f32 v3, v3
    s_nop 7
    DELTA_FINITE v3, v7
    s_mov_b32 s29, 0
.Lnorm_output:
    v_add_nc_u32 v0, s29, v1
    s_nop 7
    buffer_load_dword v6, v0, s[8:11], s26 offen
    s_waitcnt vmcnt(0)
    v_mul_f32 v6, v6, v2
    s_nop 7
    DELTA_FINITE v6, v7
    v_mul_f32 v6, v6, v3
    s_nop 7
    DELTA_FINITE v6, v7
    buffer_store_dword v6, v0, s[8:11], s31 offen
    s_waitcnt_vscnt null, 0
    s_add_u32 s29, s29, 4
    s_cmp_lt_u32 s29, s20
    s_cbranch_scc1 .Lnorm_output
    s_branch .Lend

.Lgates:
    // Group1: one lane/value head, at most48 lanes.
    v_cmp_gt_u32_e32 vcc, s17, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    s_lshl_b32 s15, s17, 2
    s_sub_u32 s31, s27, s15
    s_sub_u32 s30, s31, s15
    v_mov_b32 v7, 0x7fc00000
    buffer_store_dword v7, v0, s[8:11], s30 offen
    buffer_store_dword v7, v0, s[8:11], s31 offen
    s_waitcnt_vscnt null, 0
    s_lshl_b32 s16, s22, 4
    s_add_u32 s18, s16, s15
    s_add_u32 s19, s22, s21
    s_lshl_b32 s19, s19, 2
    s_add_u32 s20, s19, s15
    buffer_load_dword v1, v0, s[4:7], s19 offen
    buffer_load_dword v2, v0, s[4:7], s20 offen
    buffer_load_dword v3, v0, s[0:3], s16 offen
    buffer_load_dword v4, v0, s[0:3], s18 offen
    s_waitcnt vmcnt(0)
    DELTA_FINITE v1, v7
    DELTA_FINITE v2, v7
    DELTA_FINITE v3, v7
    DELTA_FINITE v4, v7
    v_cmp_ge_f32_e32 vcc, 0, v3
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    v_add_f32 v1, v1, v4
    s_nop 7
    DELTA_FINITE v1, v7
    DELTA_EXP_NEG_ABS v1, v6, v5, v4
    // Preserve original rounded t for the small-tail log1p branch. Forming t/2
    // below can lose the least subnormal even though log1p(t) itself rounds to t.
    v_mov_b32 v5, v6
    s_nop 7

    // log1p(t)=2*y*(1+y²/3+...+y^16/17), y=t/(2+t), 0<=y<=1/3.
    // This avoids cancellation in 1+t for small t. Nine positive series terms;
    // truncation before F32 rounding <=2*(1/3)^19/(19*(1-1/9)) <1.1e-10.
    // Coefficients are nearest F32. All Horner multiply/adds remain separate.
    v_add_f32 v7, 2.0, v6
    s_nop 7
    v_rcp_f32 v7, v7
    s_nop 7
    v_mul_f32 v6, v6, v7
    s_nop 7
    v_mul_f32 v7, v6, v6
    s_nop 7
    v_mov_b32 v4, 0x3d70f0f1
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3d888889, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3d9d89d9, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3dba2e8c, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3de38e39, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3e124925, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3e4ccccd, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 0x3eaaaaab, v4
    s_nop 7
    v_mul_f32 v4, v4, v7
    s_nop 7
    v_add_f32 v4, 1.0, v4
    s_nop 7
    v_mul_f32 v4, v4, v6
    s_nop 7
    v_mul_f32 v4, 2.0, v4
    s_nop 7
    // For 0<=t<=2^-24, ln(1+t) rounds to the original F32 t. Preserve this
    // boundary before multiplying by A; no fused or higher-precision fallback.
    v_cmp_ge_f32_e32 vcc, 0x33800000, v5
    s_nop 7
    v_cndmask_b32_e32 v4, v4, v5, vcc
    s_nop 7
    v_max_f32 v1, 0, v1
    s_nop 7
    v_add_f32 v1, v1, v4
    s_nop 7
    DELTA_FINITE v1, v7
    v_mul_f32 v1, v1, v3
    s_nop 7
    DELTA_FINITE v1, v7
    // A<=0 ensures log_decay<=0; exp(-abs(log_decay)) is exp(log_decay).
    DELTA_EXP_NEG_ABS v1, v6, v5, v4
    v_mov_b32 v1, v6
    s_nop 7
    DELTA_FINITE v1, v7

    // Stable beta sigmoid. Preserve rho in v1 until both gates are validated.
    DELTA_EXP_NEG_ABS v2, v6, v5, v4
    v_add_f32 v7, 1.0, v6
    s_nop 7
    v_rcp_f32 v7, v7
    s_nop 7
    v_cmp_gt_f32_e32 vcc, 0, v2
    s_nop 7
    v_cndmask_b32_e32 v6, 1.0, v6, vcc
    s_nop 7
    v_mul_f32 v6, v6, v7
    s_nop 7
    DELTA_FINITE v6, v7
    buffer_store_dword v1, v0, s[8:11], s30 offen
    buffer_store_dword v6, v0, s[8:11], s31 offen
.Lend:
    s_waitcnt_vscnt null, 0
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size delta_normalize_gates_gfx1151, .-delta_normalize_gates_gfx1151
.rept 64
    s_code_end
.endr
