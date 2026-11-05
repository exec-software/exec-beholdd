// SPDX-License-Identifier: Apache-2.0
// Experimental Delta packet v2, program15; GPU qualification is separate.
// One workgroup, one lane per value head. Exact layout: docs/qwen35-delta-packet.md.
// F32 squares -> serial F64 sum -> exact power-of-two divide -> F32 mean,
// epsilon addition, sqrt, reciprocal. Separate F32 norm and stable SiLU products.
// Status is NONFINITE(1) before arithmetic; only full success overwrites SUCCESS(0).
// Failed lanes may leave partial INACTIVE output, which the host never publishes.
// No FMA, LDS, atomics, barriers, or cross-lane writes. v0 scratch address;
// v1 immutable h*D*4; v2:3 reduction (then v2 scale); v4:5 conversion/SFU;
// v6 core/output; v7 gate denominator/reciprocal. Scratch only s15..s31.
// Include path must contain this directory; slots12/13 share this SFU helper.
.include "delta_packet_gfx1151.inc"
.macro final_finite reg
    v_cmp_class_f32_e64 vcc, \reg, 0x1f8
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Ldone
.endm
.text
.p2align 8
.globl delta_final_gfx1151
.type delta_final_gfx1151,@function
delta_final_gfx1151:
    s_version 0x2006
    s_cmp_eq_u32 s14, 0
    s_cbranch_scc0 .Lend
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 1
    s_cbranch_scc0 .Lend
    s_sub_u32 s12, s12, 1       // positive transport scalar -> packed geometry
    s_and_b32 s15, s12, 0xffffe000
    s_cmp_eq_u32 s15, 0
    s_cbranch_scc0 .Lend
    s_and_b32 s15, s12, 7
    s_lshl_b32 s16, 1, s15
    s_lshr_b32 s17, s12, 3
    s_and_b32 s17, s17, 7
    s_cmp_gt_u32 s17, 4
    s_cbranch_scc1 .Lend
    s_lshl_b32 s17, 1, s17
    s_lshr_b32 s18, s12, 6
    s_and_b32 s18, s18, 63
    s_add_u32 s18, s18, 1
    s_cmp_gt_u32 s18, 48
    s_cbranch_scc1 .Lend
    s_sub_u32 s24, s17, 1
    s_and_b32 s24, s24, s18
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lend
    s_mul_i32 s19, s17, s16
    s_lshl_b32 s19, s19, 2       // Q bytes
    s_mul_i32 s20, s18, s16
    s_lshl_b32 s20, s20, 2       // V bytes
    s_lshl_b32 s21, s19, 1
    s_add_u32 s21, s21, s20     // C bytes
    s_mul_i32 s23, s20, s16     // R bytes
    s_mul_i32 s22, s21, 3
    s_add_u32 s22, s22, s23
    s_add_u32 s22, s22, s20     // bank B
    s_lshl_b32 s24, s21, 2
    s_lshl_b32 s25, s18, 3
    s_add_u32 s26, s24, s25     // parameter norm offset
    s_lshl_b32 s25, s16, 2
    s_add_u32 s24, s26, s25
    s_add_u32 s24, s24, 4       // epsilon trailer follows norm
    s_cmp_eq_u32 s2, s24
    s_cbranch_scc0 .Lend
    s_add_u32 s24, s21, s20
    s_lshl_b32 s25, s18, 3
    s_add_u32 s24, s24, s25
    s_cmp_eq_u32 s6, s24
    s_cbranch_scc0 .Lend
    s_lshl_b32 s27, s22, 1
    s_add_u32 s27, s27, s21
    s_lshl_b32 s25, s19, 1
    s_add_u32 s27, s27, s25
    s_lshl_b32 s25, s18, 3
    s_add_u32 s27, s27, s25     // core offset
    s_add_u32 s28, s27, s20     // status offset
    s_lshl_b32 s25, s18, 2
    s_add_u32 s24, s28, s25
    s_cmp_eq_u32 s10, s24
    s_cbranch_scc0 .Lend
    s_or_b32 s24, s0, s4
    s_or_b32 s24, s24, s8
    s_and_b32 s24, s24, 3
    s_cmp_eq_u32 s24, 0
    s_cbranch_scc0 .Lend
    s_sub_u32 s31, s2, 4
    buffer_load_dword v6, off, s[0:3], s31
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s13, v6  // uniform epsilon; sentinel already checked
    s_nop 7
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lend
    s_cmp_ge_u32 s13, 0x7f800000
    s_cbranch_scc1 .Lend
    v_and_b32 v0, 0x3ff, v0
    s_nop 7
    v_cmp_gt_u32_e32 vcc, s18, v0
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
    // Keep an immutable head byte base, shared by core, z, and output indexing.
    s_add_u32 s19, s15, 2
    v_lshlrev_b32 v1, s19, v0
    s_nop 7
    v_lshlrev_b32 v0, 2, v0
    s_nop 7
    v_mov_b32 v6, 1
    s_nop 7
    buffer_store_dword v6, v0, s[8:11], s28 offen
    s_waitcnt_vscnt null, 0
    // Inactive bank output = (1-b)*B + 3*Cbytes + Rbytes.
    s_lshr_b32 s24, s12, 12
    s_xor_b32 s24, s24, 1
    s_mul_i32 s24, s24, s22
    s_mul_i32 s25, s21, 3
    s_add_u32 s24, s24, s25
    s_add_u32 s24, s24, s23
    s_lshl_b32 s16, s16, 2      // D bytes
    // 2^-logD as an exact F64 factor: low word zero, exponent 1023-logD.
    s_sub_u32 s29, 1023, s15
    s_lshl_b32 s29, s29, 20
    v_mov_b32 v2, 0
    v_mov_b32 v3, 0
    s_mov_b32 s30, 0
.Lsum:
    s_add_u32 s31, s27, s30
    buffer_load_dword v6, v1, s[8:11], s31 offen
    s_waitcnt vmcnt(0)
    final_finite v6
    v_mul_f32 v6, v6, v6
    s_nop 7
    final_finite v6
    v_cvt_f64_f32 v[4:5], v6
    s_nop 7
    v_add_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    s_add_u32 s30, s30, 4
    s_cmp_lt_u32 s30, s16
    s_cbranch_scc1 .Lsum
    v_mov_b32 v4, 0
    v_mov_b32 v5, s29
    s_nop 7
    v_mul_f64 v[2:3], v[2:3], v[4:5]
    s_nop 7
    v_cvt_f32_f64 v2, v[2:3]
    s_nop 7
    final_finite v2
    v_add_f32 v2, s13, v2
    s_nop 7
    final_finite v2
    v_sqrt_f32 v2, v2
    s_nop 7
    final_finite v2
    v_rcp_f32 v2, v2
    s_nop 7
    final_finite v2
    s_mov_b32 s30, 0
.Loutput:
    s_add_u32 s31, s27, s30
    buffer_load_dword v6, v1, s[8:11], s31 offen
    s_add_u32 s31, s26, s30
    buffer_load_dword v4, off, s[0:3], s31
    s_waitcnt vmcnt(0)
    final_finite v4
    v_mul_f32 v6, v6, v2
    s_nop 7
    final_finite v6
    v_mul_f32 v6, v6, v4
    s_nop 7
    final_finite v6
    s_add_u32 s31, s21, s30     // projection z offset = C bytes
    buffer_load_dword v4, v1, s[4:7], s31 offen
    s_waitcnt vmcnt(0)
    final_finite v4
    // Stable sigmoid: e=F32(exp(-abs(z))); z>=0 ? 1/(1+e) : e/(1+e).
    // v0 is dead address scratch until final status publication. The shared
    // helper rescales the SFU result before the sigmoid/SiLU products, keeping
    // the CPU's F32-exp boundary: z=-90 has a subnormal e; z=-105 has e=0.
    // Original z remains in v4, including finite extremes beyond the clamp.
    DELTA_EXP_NEG_ABS v4, v5, v0, v7
    final_finite v5
    v_add_f32 v7, 1.0, v5
    s_nop 7
    final_finite v7
    v_rcp_f32 v7, v7
    s_nop 7
    final_finite v7
    v_mul_f32 v5, v5, v7
    s_nop 7
    final_finite v5
    v_cmp_ge_f32_e64 vcc, v4, 0.0
    s_nop 7
    v_cndmask_b32_e32 v5, v5, v7, vcc
    s_nop 7
    v_mul_f32 v4, v4, v5        // SiLU(z)
    s_nop 7
    final_finite v4
    v_mul_f32 v6, v6, v4
    s_nop 7
    final_finite v6
    s_add_u32 s31, s24, s30
    buffer_store_dword v6, v1, s[8:11], s31 offen
    // Complete the store before the next iteration loads another core into v6.
    s_waitcnt_vscnt null, 0
    s_add_u32 s30, s30, 4
    s_cmp_lt_u32 s30, s16
    s_cbranch_scc1 .Loutput
    // All output stores are ordered before the success marker.
    s_waitcnt_vscnt null, 0
    v_lshrrev_b32 v0, s15, v1   // h*4
    s_nop 7
    v_mov_b32 v6, 0
    s_nop 7
    buffer_store_dword v6, v0, s[8:11], s28 offen
.Ldone:
    s_waitcnt_vscnt null, 0
.Lend:
    s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    s_endpgm
.size delta_final_gfx1151, .-delta_final_gfx1151
.rept 64
    s_code_end
.endr
