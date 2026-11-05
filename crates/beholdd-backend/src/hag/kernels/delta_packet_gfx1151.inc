// SPDX-License-Identifier: Apache-2.0
// Shared admission and F32 tail helper for the experimental Delta programs.
// docs/qwen35-delta-packet.md v2. Fixed output SGPR map:
// s15 logD, s16 logHk, s17 Hv, s18 D, s19 Hk, s20 Q, s21 V,
// s22 C, s23 Bbytes, s24 active-bank bytes, s25 inactive-bank bytes,
// s26 conv_qkv bytes=2B, s27 core bytes, s28 status bytes.
// s29..31 scratch. s12 becomes packed geometry; s13 becomes epsilon bits after
// raw scalar sentinel/extent admission. Branch target .Lend is local
// to the standalone entry. Does not touch VGPRs/exec before lane admission.
.macro DELTA_GEOMETRY
    s_cmp_eq_u32 s12, 0
    s_cbranch_scc1 .Lend
    s_cmp_eq_u32 s13, 1
    s_cbranch_scc0 .Lend
    s_sub_u32 s12, s12, 1
    s_and_b32 s29, s12, 0xffffe000
    s_cmp_eq_u32 s29, 0
    s_cbranch_scc0 .Lend
    s_and_b32 s15, s12, 7
    s_lshr_b32 s16, s12, 3
    s_and_b32 s16, s16, 7
    s_cmp_gt_u32 s16, 4
    s_cbranch_scc1 .Lend
    s_lshr_b32 s17, s12, 6
    s_and_b32 s17, s17, 63
    s_add_u32 s17, s17, 1
    s_cmp_gt_u32 s17, 48
    s_cbranch_scc1 .Lend
    s_lshl_b32 s18, 1, s15
    s_lshl_b32 s19, 1, s16
    s_sub_u32 s29, s19, 1
    s_and_b32 s29, s17, s29
    s_cmp_eq_u32 s29, 0
    s_cbranch_scc0 .Lend
    s_lshl_b32 s20, s19, s15
    s_lshl_b32 s21, s17, s15
    s_lshl_b32 s22, s20, 1
    s_add_u32 s22, s22, s21
    s_lshl_b32 s29, s21, s15
    s_mul_i32 s23, s22, 3
    s_add_u32 s23, s23, s29
    s_add_u32 s23, s23, s21
    s_lshl_b32 s23, s23, 2
    s_lshr_b32 s29, s12, 12
    s_and_b32 s29, s29, 1
    s_mul_i32 s24, s23, s29
    s_sub_u32 s25, s23, s24
    s_lshl_b32 s26, s23, 1
    s_add_u32 s27, s20, s17
    s_lshl_b32 s27, s27, 1
    s_add_u32 s27, s27, s22
    s_lshl_b32 s27, s27, 2
    s_add_u32 s27, s27, s26
    s_lshl_b32 s28, s21, 2
    s_add_u32 s28, s28, s27
    // Exact SRD0 length =16C+8Hv+4D+4, including epsilon trailer.
    s_lshl_b32 s29, s22, 4
    s_lshl_b32 s30, s17, 3
    s_add_u32 s29, s29, s30
    s_lshl_b32 s30, s18, 2
    s_add_u32 s29, s29, s30
    s_add_u32 s30, s29, 4
    s_cmp_eq_u32 s2, s30
    s_cbranch_scc0 .Lend
    s_or_b32 s30, s0, s4
    s_or_b32 s30, s30, s8
    s_and_b32 s30, s30, 3
    s_cmp_eq_u32 s30, 0
    s_cbranch_scc0 .Lend
    // Uniform load does not destroy v0's hardware lane ID. Reuse raw s13 after
    // checking it was the required1; subsequent arithmetic sees epsilon bits.
    buffer_load_dword v1, off, s[0:3], s29
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s13, v1
    s_nop 7
    s_cmp_eq_u32 s13, 0
    s_cbranch_scc1 .Lend
    s_cmp_ge_u32 s13, 0x7f800000
    s_cbranch_scc1 .Lend
    // Exact SRD1 length =4(C+V+2Hv).
    s_lshl_b32 s29, s17, 1
    s_add_u32 s29, s29, s21
    s_add_u32 s29, s29, s22
    s_lshl_b32 s29, s29, 2
    s_cmp_eq_u32 s6, s29
    s_cbranch_scc0 .Lend
    // Exact SRD2 length =status_offset+4Hv.
    s_lshl_b32 s29, s17, 2
    s_add_u32 s29, s29, s28
    s_cmp_eq_u32 s10, s29
    s_cbranch_scc0 .Lend
.endm

// Tolerance-based exp(-abs(finite input)). Four distinct registers except value
// may alias result after abs is captured. Clamp128: exp(-128) rounds to F32 zero.
// Split a=n*ln2+r before the SFU: HI=22713/32768 (0x3f317200) makes n*HI exact
// for integer n<=184; LO=0x35bfbe8e is the remaining ln2 rounded to F32. This
// avoids rounding the large a*log2(e) exponent before reduction (notably a=90).
// SFU sees only -r*log2(e), near [-1,0]. Multiply its result by the exact normal
// power 2^(64-n), then FIRST scale by2^-64, rounding t to F32 before sigmoid or
// SiLU products. Do not fuse the original input into that final scaling: the CPU
// reference has the explicit F32-exp boundary (-103 gives bit1, -105 gives0).
// VCC is clobbered. No FMA. FLOAT_MODE0xf0 preserves arithmetic denormals but
// native qualification must still exercise them; this is only source intent.
.macro DELTA_EXP_NEG_ABS value, result, abs_tmp, scale_tmp
    v_and_b32 \abs_tmp, 0x7fffffff, \value
    s_nop 7
    v_min_f32 \abs_tmp, 0x43000000, \abs_tmp
    s_nop 7
    v_mul_f32 \result, 0x3fb8aa3b, \abs_tmp
    s_nop 7
    v_cvt_u32_f32 \result, \result
    s_nop 7
    v_cvt_f32_u32 \result, \result
    s_nop 7
    v_mul_f32 \scale_tmp, 0x3f317200, \result
    s_nop 7
    v_sub_f32 \abs_tmp, \abs_tmp, \scale_tmp
    s_nop 7
    v_mul_f32 \scale_tmp, 0x35bfbe8e, \result
    s_nop 7
    v_sub_f32 \abs_tmp, \abs_tmp, \scale_tmp
    s_nop 7
    v_cvt_u32_f32 \scale_tmp, \result
    s_nop 7
    v_sub_nc_u32 \scale_tmp, 191, \scale_tmp
    s_nop 7
    v_lshlrev_b32 \scale_tmp, 23, \scale_tmp
    s_nop 7
    v_mul_f32 \result, 0xbfb8aa3b, \abs_tmp
    s_nop 7
    v_exp_f32 \result, \result
    s_nop 7
    v_mul_f32 \result, \result, \scale_tmp
    s_nop 7
    v_mul_f32 \result, 0x1f800000, \result
    s_nop 7
.endm

// Shrink exec on nonfinite F32, preserving each stage's preinitialized marker.
// Both registers must differ; scratch does not survive. No class opcode needed.
.macro DELTA_FINITE value, scratch
    v_and_b32 \scratch, 0x7f800000, \value
    s_nop 7
    v_cmp_ne_u32_e32 vcc, 0x7f800000, \scratch
    s_nop 7
    s_and_b64 exec, exec, vcc
    s_nop 7
    s_cbranch_execz .Lend
.endm
