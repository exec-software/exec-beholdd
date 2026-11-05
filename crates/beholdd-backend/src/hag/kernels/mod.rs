//! Bounded native GFX1151 packed matvec and elementwise proof kernels.
//!
//! The checked-in ISA sources and their byte-for-byte LLVM verification are the
//! authority for these original little-endian instruction words. No shader compiler,
//! Vulkan, HIP or ROCm runtime is required when using the embedded bytes.

pub(crate) mod kquant;
pub(crate) mod qwen35_state;

pub(crate) const MAX_ROWS: usize = 4096;
pub(crate) const MAX_COLS: usize = 4096;
pub(crate) const MAX_ELEMENTS: usize = 4096;
pub(crate) const MAX_ROPE_PAIRS: usize = MAX_ELEMENTS / 2;
pub(crate) const MAX_ATTENTION_POSITIONS: usize = 16384;
pub(crate) const RSRC1: u32 = 0x408f_0001;
pub(crate) const RSRC2: u32 = 0x0000_009c;
pub(crate) const RSRC3: u32 = 0;
pub(crate) const RAW_SRD_WORD3: u32 = 0x3001_6fac;
pub(crate) const THREADS: [u32; 3] = [64, 1, 1];
// The historical small linear fixture suite still dispatches a single group.
pub(crate) const GROUPS: [u32; 3] = [1, 1, 1];
const CODE_END: u32 = 0xbf9f_0000;
const CODE_END_WORDS: usize = 64;

/// Exact workgroup count for one bounded row/element dispatch, including its tail.
/// Empty operations are retained no-ops in the adapter and must not be dispatched.
pub(crate) fn dispatch_groups(count: usize) -> Result<u32, &'static str> {
    if !(1..=MAX_ELEMENTS).contains(&count) {
        return Err("native dispatch requires 1..=4096 rows or elements");
    }
    Ok(count.div_ceil(THREADS[0] as usize) as u32)
}

/// Attention score coverage, independent of the 4096-element linear/vector cap.
pub(crate) fn attention_dispatch_groups(positions: usize) -> Result<u32, &'static str> {
    if !(1..=MAX_ATTENTION_POSITIONS).contains(&positions) {
        return Err("native attention dispatch requires 1..=16384 positions");
    }
    Ok(positions.div_ceil(THREADS[0] as usize) as u32)
}

/// Fixed packed BF16 shader and instruction-fetch padding, ready for one upload.
pub(crate) fn bf16_bytes() -> Vec<u8> {
    KernelKind::Bf16.shader_bytes()
}

/// Fixed packed GGML Q8_0 shader and instruction-fetch padding, ready for one upload.
pub(crate) fn q8_0_bytes() -> Vec<u8> {
    KernelKind::Q8_0.shader_bytes()
}

/// Fixed in-place SwiGLU shader: parameter/up/gate SRDs, scalars `[len, 1]`.
/// Requires 1..=4096 elements, disjoint SRDs and a finite clamp limit.
/// The exponential/reciprocal approximation is tolerance-based, not bitwise CPU parity.
pub(crate) fn swiglu_bytes() -> Vec<u8> {
    padded_shader(SWIGLU_CODE)
}

/// Fixed in-place residual shader: parameter/input/output SRDs, scalars `[len, 1]`.
/// Requires 1..=4096 elements, disjoint SRDs and a finite scale; no fused arithmetic.
pub(crate) fn scaled_add_bytes() -> Vec<u8> {
    padded_shader(SCALED_ADD_CODE)
}

/// Unweighted RMSNorm: epsilon/input/output SRDs, scalars `[len, 1]`.
/// Requires 1..=4096 elements, finite nonnegative epsilon and disjoint SRDs.
/// Sum of squares is serial F64; reciprocal/rsqrt are tolerance-based approximations.
pub(crate) fn rmsnorm_bytes() -> Vec<u8> {
    padded_shader(RMSNORM_CODE)
}

/// Separate probe-only full-width normalization, at most 5120 elements/80 groups.
/// Every lane reduces the complete vector; this never normalizes separate tiles.
pub(crate) fn rmsnorm_qwen35_bytes() -> Vec<u8> {
    padded_shader(RMSNORM_QWEN35_CODE)
}

/// Pointwise F32 multiplication: resident weight/input/output SRDs, `[len, 1]`.
/// Requires 1..=4096 elements and three disjoint, exact-sized vector descriptors.
pub(crate) fn multiply_bytes() -> Vec<u8> {
    padded_shader(MULTIPLY_CODE)
}

/// In-place RoPE: cosine/sine/x SRDs, scalars `[pairs, 1 + split_half + 2*inverse]`.
/// Requires 1..=2048 pairs, three disjoint exact extents, and at most 32 groups.
pub(crate) fn rope_bytes() -> Vec<u8> {
    padded_shader(ROPE_CODE)
}

/// F64 unscaled attention scores: Q/K-prefix/F64-scores SRDs, `[head_dim, stride]`.
/// The exact output extent encodes 1..=16384 positions; head_dim is 1..=256.
pub(crate) fn attention_score_bytes() -> Vec<u8> {
    padded_shader(ATTENTION_SCORE_CODE)
}

/// Scaled softmax: F32-scale/F64-scores/F32-probabilities SRDs, `[positions, 1]`.
/// Requires positive finite scale, 1..=16384 positions, disjoint exact extents,
/// and exactly one wave64 workgroup; lane `i` writes positions `i + 64*k`.
pub(crate) fn attention_softmax_bytes() -> Vec<u8> {
    padded_shader(ATTENTION_SOFTMAX_CODE)
}

/// F64 weighted sum: probabilities/V-prefix/F32-output SRDs, `[positions, stride]`.
/// The exact output extent encodes 1..=256 dimensions; positions is 1..=16384.
pub(crate) fn attention_value_bytes() -> Vec<u8> {
    padded_shader(ATTENTION_VALUE_CODE)
}

/// Q8_0 row decode: exact packed row/private four-byte dummy/F32-output SRDs.
/// Scalars `[columns, 1]`, 32..=4096 columns in complete blocks, disjoint extents.
pub(crate) fn q8_0_row_bytes() -> Vec<u8> {
    padded_shader(Q8_0_ROW_CODE)
}

/// Stable adapter program-ID order, shared by queue creation and cost admission.
pub(crate) fn program_bytes() -> [Vec<u8>; 11] {
    [
        bf16_bytes(),
        q8_0_bytes(),
        swiglu_bytes(),
        scaled_add_bytes(),
        rmsnorm_bytes(),
        multiply_bytes(),
        rope_bytes(),
        attention_score_bytes(),
        attention_softmax_bytes(),
        attention_value_bytes(),
        q8_0_row_bytes(),
    ]
}

/// Explicit operator-probe library. The established dense library stays unchanged.
/// IDs 2..9 preserve operator meanings; program 4 has its own wider admission.
pub(crate) const QWEN35_PROGRAM_COUNT: usize = 18;

pub(crate) fn qwen35_program_bytes() -> [Vec<u8>; QWEN35_PROGRAM_COUNT] {
    use kquant::KQuantKind;
    let [convolution, normalize_gates, recurrence, final_gate, sigmoid_gate, finite_check] =
        qwen35_state::program_bytes();
    [
        KQuantKind::Q4K.matvec_bytes(),
        KQuantKind::Q6K.matvec_bytes(),
        swiglu_bytes(),
        scaled_add_bytes(),
        rmsnorm_qwen35_bytes(),
        multiply_bytes(),
        rope_bytes(),
        attention_score_bytes(),
        attention_softmax_bytes(),
        attention_value_bytes(),
        KQuantKind::Q4K.row_decode_bytes(),
        KQuantKind::Q6K.row_decode_bytes(),
        convolution,
        normalize_gates,
        recurrence,
        final_gate,
        sigmoid_gate,
        finite_check,
    ]
}

fn padded_shader(code: &[u32]) -> Vec<u8> {
    code.iter()
        .copied()
        .chain(std::iter::repeat_n(CODE_END, CODE_END_WORDS))
        .flat_map(u32::to_le_bytes)
        .collect()
}

/// The packed matvec layouts supported by the bounded native kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KernelKind {
    Bf16,
    Q8_0,
}

impl KernelKind {
    /// Check bounded geometry before any GPU allocation/upload or submission.
    /// The caller must additionally match exact packed/input/output slice lengths.
    pub(crate) fn packed_bytes(self, rows: usize, cols: usize) -> Result<usize, &'static str> {
        if !(1..=MAX_ROWS).contains(&rows) || !(1..=MAX_COLS).contains(&cols) {
            return Err("native matvec requires 1..=4096 rows and 1..=4096 columns");
        }
        match self {
            Self::Bf16 => Ok(rows * cols * 2),
            Self::Q8_0 if cols.is_multiple_of(32) => Ok(rows * (cols / 32) * 34),
            Self::Q8_0 => Err("Q8_0 columns must contain complete 32-element blocks"),
        }
    }

    /// Immutable shader words followed by 256 bytes of instruction-fetch padding.
    pub(crate) fn shader_bytes(self) -> Vec<u8> {
        padded_shader(self.code())
    }

    fn code(self) -> &'static [u32] {
        match self {
            Self::Bf16 => BF16_CODE,
            Self::Q8_0 => Q8_0_CODE,
        }
    }
}

// Generated from bf16_matvec_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
// Explicit arrays keep ordinary/offline builds independent of assembler tooling.
const BF16_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20034, 0xbf06800c, 0xbfa20032, 0xbf08ff0c, 0x00001000, 0xbfa2002f,
    0xbf06800d, 0xbfa2002d, 0xbf08ff0d, 0x00001000, 0xbfa2002a, 0x360000ff, 0x000003ff, 0xbf800007,
    0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000d, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa5001f,
    0xd72c0001, 0x0002000c, 0xbf800007, 0x30020281, 0xbf800007, 0x30040082, 0xbf800007, 0x7e060280,
    0xbe8f0080, 0xbe900080, 0xbe910080, 0xbf800007, 0xe0480000, 0x10400401, 0xe0500000, 0x11010500,
    0xbf8903f7, 0x30080890, 0xbf800007, 0x10080b04, 0xbf800007, 0x06060903, 0xbf800007, 0x800f810f,
    0x80108210, 0x80118411, 0xbf0a0c0f, 0xbfa2ffef, 0xe0680000, 0x80420302, 0xbc7c0000, 0xbfb60003,
    0xbfb00000,
];

// Generated from q8_0_matvec_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const Q8_0_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20046, 0xbf06800c, 0xbfa20044, 0xbf08ff0c, 0x00001000, 0xbfa20041,
    0x8b0f9f0c, 0xbf06800f, 0xbfa1003e, 0xbf06800d, 0xbfa2003c, 0xbf08ff0d, 0x00001000, 0xbfa20039,
    0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000d, 0xbf800007,
    0x8bfe6a7e, 0xbf800007, 0xbfa5002e, 0x850f850c, 0x960fa20f, 0xbf800007, 0xd72c0001, 0x0002000f,
    0xbf800007, 0x30040082, 0xbf800007, 0x7e060280, 0xbe8f0080, 0xbe900080, 0xbe910080, 0xbf800007,
    0xe0480000, 0x0f400601, 0xbf8903f7, 0x7e0c1706, 0xbf800007, 0x8013820f, 0xbe920080, 0xbf800007,
    0xe0440000, 0x13400401, 0xe0500000, 0x11010500, 0xbf8903f7, 0x7e080b04, 0xbf800007, 0x10080906,
    0xbf800007, 0x10080b04, 0xbf800007, 0x06060903, 0xbf800007, 0x80108110, 0x80118411, 0x80128112,
    0x80138113, 0xbf0aa012, 0xbfa2ffec, 0x800fa20f, 0xbf0a0c10, 0xbfa2ffe1, 0xe0680000, 0x80420302,
    0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from swiglu_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const SWIGLU_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20049, 0xbf06800c, 0xbfa20047, 0xbf08ff0c, 0x00001000, 0xbfa20044,
    0xbf06810d, 0xbfa10042, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007,
    0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50037, 0x30020082, 0xbf800007, 0xe0500000,
    0x80000400, 0xe0500000, 0x80410301, 0xe0500000, 0x80420201, 0xbf8903f7, 0x7c220880, 0xbf800007,
    0xbfa3000f, 0x7c220902, 0xbf800007, 0x02040504, 0xbf800007, 0x3a0a08ff, 0x80000000, 0xbf800007,
    0x7c220b03, 0xbf800007, 0x02060b03, 0xbf800007, 0x7c280903, 0xbf800007, 0x02060903, 0xbf800007,
    0x100a04ff, 0xbfb8aa3b, 0xbf800007, 0x7e0c4b05, 0xbf800007, 0x060c0cf2, 0xbf800007, 0x7e0e02ff,
    0x2f800000, 0xbf800007, 0x7c220cff, 0x6f800000, 0xbf800007, 0x020e0ef2, 0xbf800007, 0x100c0d07,
    0xbf800007, 0x7e0c5506, 0xbf800007, 0x10040d02, 0xbf800007, 0x10040507, 0xbf800007, 0x10040702,
    0xbf800007, 0xe0680000, 0x80420201, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from scaled_add_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const SCALED_ADD_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20022, 0xbf06800c, 0xbfa20020, 0xbf08ff0c, 0x00001000, 0xbfa2001d,
    0xbf06810d, 0xbfa1001b, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007,
    0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50010, 0x30020082, 0xbf800007, 0xe0500000,
    0x80000200, 0xe0500000, 0x80410301, 0xe0500000, 0x80420401, 0xbf8903f7, 0x10060702, 0xbf800007,
    0x06080704, 0xbf800007, 0xe0680000, 0x80420401, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from rmsnorm_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const RMSNORM_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20042, 0xbf06800c, 0xbfa20040, 0xbf08ff0c, 0x00001000, 0xbfa2003d,
    0xbf06810d, 0xbfa1003b, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007,
    0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50030, 0x30020082, 0xbf800007, 0x7e040280,
    0x7e060280, 0xbe8f0080, 0xbe900080, 0xe0500000, 0x10010600, 0xbf8903f7, 0x7e082106, 0xbf800007,
    0xd7280004, 0x00020904, 0xbf800007, 0xd7270002, 0x00020902, 0xbf800007, 0x800f810f, 0x80108410,
    0xbf0a0c0f, 0xbfa2fff1, 0x7e082c0c, 0xbf800007, 0x7e085f04, 0xbf800007, 0xd7280002, 0x00020902,
    0xbf800007, 0xe0500000, 0x80000600, 0xbf8903f7, 0x7e082106, 0xbf800007, 0xd7270002, 0x00020902,
    0xbf800007, 0x7e046302, 0xbf800007, 0x7e041f02, 0xbf800007, 0xe0500000, 0x80410601, 0xbf8903f7,
    0x10040506, 0xbf800007, 0xe0680000, 0x80420201, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from rmsnorm_qwen35_gfx1151.s by LLVM21, gfx1151 wave64.
// This isolated program is assembled and byte-checked, not yet GPU-qualified.
const RMSNORM_QWEN35_CODE: &[u32] = &[
    0xb0802006, 0xbf08ff0e, 0x0000004f, 0xbfa20042, 0xbf06800c, 0xbfa20040, 0xbf08ff0c, 0x00001400,
    0xbfa2003d, 0xbf06810d, 0xbfa1003b, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e,
    0xbf800007, 0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50030, 0x30020082, 0xbf800007,
    0x7e040280, 0x7e060280, 0xbe8f0080, 0xbe900080, 0xe0500000, 0x10010600, 0xbf8903f7, 0x7e082106,
    0xbf800007, 0xd7280004, 0x00020904, 0xbf800007, 0xd7270002, 0x00020902, 0xbf800007, 0x800f810f,
    0x80108410, 0xbf0a0c0f, 0xbfa2fff1, 0x7e082c0c, 0xbf800007, 0x7e085f04, 0xbf800007, 0xd7280002,
    0x00020902, 0xbf800007, 0xe0500000, 0x80000600, 0xbf8903f7, 0x7e082106, 0xbf800007, 0xd7270002,
    0x00020902, 0xbf800007, 0x7e046302, 0xbf800007, 0x7e041f02, 0xbf800007, 0xe0500000, 0x80410601,
    0xbf8903f7, 0x10040506, 0xbf800007, 0xe0680000, 0x80420201, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from multiply_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const MULTIPLY_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa2001e, 0xbf06800c, 0xbfa2001c, 0xbf08ff0c, 0x00001000, 0xbfa20019,
    0xbf06810d, 0xbfa10017, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007,
    0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa5000c, 0x30020082, 0xbf800007, 0xe0500000,
    0x80400201, 0xe0500000, 0x80410301, 0xbf8903f7, 0x10060503, 0xbf800007, 0xe0680000, 0x80420301,
    0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from rope_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const ROPE_CODE: &[u32] = &[
    0xb0802006, 0xbf089f0e, 0xbfa20044, 0xbf06800c, 0xbfa20042, 0xbf08ff0c, 0x00000800, 0xbfa2003f,
    0xbf06800d, 0xbfa2003d, 0xbf08840d, 0xbfa2003b, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000,
    0x04010c0e, 0xbf800007, 0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50030, 0x30000082,
    0xbf800007, 0xe0500000, 0x80400400, 0xe0500000, 0x80410500, 0xbf8903f7, 0x808f810d, 0x84109e0f,
    0x8b10ff10, 0x80000000, 0x8c10f210, 0xbf800007, 0x100a0a10, 0xbf800007, 0xbf0d800f, 0xbfa20005,
    0x30000081, 0xbf800007, 0x4a020084, 0xbf800007, 0xbfa00004, 0x8410820c, 0xbf800007, 0x4a020010,
    0xbf800007, 0xe0500000, 0x80420200, 0xe0500000, 0x80420301, 0xbf8903f7, 0x100c0902, 0xbf800007,
    0x100e0b03, 0xbf800007, 0x080c0f06, 0xbf800007, 0x10040b02, 0xbf800007, 0x10060903, 0xbf800007,
    0x06040702, 0xbf800007, 0xe0680000, 0x80420600, 0xe0680000, 0x80420201, 0xbc7c0000, 0xbfb60003,
    0xbfb00000,
];

// Generated from attention_score_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const ATTENTION_SCORE_CODE: &[u32] = &[
    0xb0802006, 0xbf08ff0e, 0x000000ff, 0xbfa2003f, 0xbf06800c, 0xbfa2003d, 0xbf08ff0c, 0x00000100,
    0xbfa2003a, 0xbf0a0c0d, 0xbfa20038, 0xbf08ff0d, 0x00001000, 0xbfa20035, 0xbf06800a, 0xbfa20033,
    0xbf08ff0a, 0x00020000, 0xbfa20030, 0x8b0f870a, 0xbf06800f, 0xbfa1002d, 0x850f830a, 0x360000ff,
    0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000f, 0xbf800007, 0x8bfe6a7e,
    0xbf800007, 0xbfa50021, 0xd72c0001, 0x0002000d, 0xbf800007, 0x30020282, 0xbf800007, 0x30000083,
    0xbf800007, 0x7e040280, 0x7e060280, 0xbe8f0080, 0xbe900080, 0xe0500000, 0x10000400, 0xe0500000,
    0x10410601, 0xbf8903f7, 0x7e082104, 0xbf800007, 0x7e0c2106, 0xbf800007, 0xd7280004, 0x00020d04,
    0xbf800007, 0xd7270002, 0x00020902, 0xbf800007, 0x800f810f, 0x80108410, 0xbf0a0c0f, 0xbfa2ffed,
    0xe06c0000, 0x80420200, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from attention_softmax_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const ATTENTION_SOFTMAX_CODE: &[u32] = &[
    0xb0802006, 0xbf08800e, 0xbfa2008c, 0xbf06800c, 0xbfa2008a, 0xbf08ff0c, 0x00004000, 0xbfa20087,
    0xbf06810d, 0xbfa10085, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007,
    0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa5007a, 0x30020082, 0xbf800007, 0x30000083,
    0xbf800007, 0xe0500000, 0x80000600, 0xbf8903f7, 0x7e0c2106, 0xbf800007, 0xbe910080, 0x8412820c,
    0x7e0402ff, 0xff800000, 0xbe8f0080, 0xbe900080, 0xe0540000, 0x10010400, 0xbf8903f7, 0xd7280004,
    0x00020d04, 0xbf800007, 0x7e081f04, 0xbf800007, 0x7c280504, 0xbf800007, 0x02040902, 0xbf800007,
    0x800f810f, 0x80108810, 0xbf0a0c0f, 0xbfa2fff0, 0x360a04ff, 0x7f800000, 0xbf800007, 0x7c940aff,
    0x7f800000, 0xbf800007, 0xbfa40040, 0x7e060280, 0xbe8f0080, 0xbe900080, 0xe0540000, 0x10010400,
    0xbf8903f7, 0xd7280004, 0x00020d04, 0xbf800007, 0x7e081f04, 0xbf800007, 0x360a08ff, 0x7f800000,
    0xbf800007, 0x7c9a0aff, 0x7f800000, 0xbf800007, 0x02080902, 0xbf800007, 0x08080504, 0xbf800007,
    0x100808ff, 0x3fb8aa3b, 0xbf800007, 0x7e084b04, 0xbf800007, 0x02080880, 0xbf800007, 0x06060903,
    0xbf800007, 0x800f810f, 0x80108810, 0xbf0a0c0f, 0xbfa2ffe1, 0xe0540000, 0x80410400, 0xbf8903f7,
    0xd7280004, 0x00020d04, 0xbf800007, 0x7e081f04, 0xbf800007, 0x360a08ff, 0x7f800000, 0xbf800007,
    0x7c9a0aff, 0x7f800000, 0xbf800007, 0x02080902, 0xbf800007, 0x08080504, 0xbf800007, 0x100808ff,
    0x3fb8aa3b, 0xbf800007, 0x7e084b04, 0xbf800007, 0x02080880, 0xbf800007, 0x7e0a5503, 0xbf800007,
    0x10080b04, 0xbf800007, 0xbfa00003, 0xbe910081, 0x7e080280, 0xbf800007, 0xe0680000, 0x80420401,
    0xbc7c0000, 0x4a0000ff, 0x00000200, 0xbf800007, 0x4a0202ff, 0x00000100, 0xbf800007, 0x7c980212,
    0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50003, 0xbf068111, 0xbfa2ffed, 0xbfa0ffce, 0xbfb60003,
    0xbfb00000,
];

// Generated from attention_value_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const ATTENTION_VALUE_CODE: &[u32] = &[
    0xb0802006, 0xbf08830e, 0xbfa2003f, 0xbf06800c, 0xbfa2003d, 0xbf08ff0c, 0x00004000, 0xbfa2003a,
    0xbf08ff0d, 0x00001000, 0xbfa20037, 0xbf06800a, 0xbfa20035, 0xbf08ff0a, 0x00000400, 0xbfa20032,
    0x8b0f830a, 0xbf06800f, 0xbfa1002f, 0x850f820a, 0xbf0a0f0d, 0xbfa2002c, 0x360000ff, 0x000003ff,
    0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000f, 0xbf800007, 0x8bfe6a7e, 0xbf800007,
    0xbfa50021, 0x30020082, 0xbf800007, 0x7e040280, 0x7e060280, 0xbe8f0080, 0xbe900080, 0xbe910080,
    0x8412820d, 0xe0500000, 0x10000400, 0xe0500000, 0x11410601, 0xbf8903f7, 0x7e082104, 0xbf800007,
    0x7e0c2106, 0xbf800007, 0xd7280004, 0x00020d04, 0xbf800007, 0xd7270002, 0x00020902, 0xbf800007,
    0x800f810f, 0x80108410, 0x80111211, 0xbf0a0c0f, 0xbfa2ffec, 0x7e041f02, 0xbf800007, 0xe0680000,
    0x80420201, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// Generated from q8_0_row_gfx1151.s by LLVM21 llvm-mc, gfx1151, wave64.
const Q8_0_ROW_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa2002d, 0xbf06800c, 0xbfa2002b, 0xbf08ff0c, 0x00001000, 0xbfa20028,
    0x8b0f9f0c, 0xbf06800f, 0xbfa10025, 0xbf06810d, 0xbfa10023, 0x360000ff, 0x000003ff, 0xbf800007,
    0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000c, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50018,
    0x30020082, 0xbf800007, 0x32040085, 0xbf800007, 0xd72c0002, 0x000204a2, 0xbf800007, 0x3606009f,
    0xbf800007, 0x4a060702, 0xbf800007, 0xe0480000, 0x80400402, 0xe0440002, 0x80400503, 0xbf8903f7,
    0x7e081704, 0x7e0a0b05, 0xbf800007, 0x100a0b04, 0xbf800007, 0xe0680000, 0x80420501, 0xbc7c0000,
    0xbfb60003, 0xbfb00000,
];

#[cfg(test)]
mod numerical_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_geometry_is_bounded_without_expanding_weights() {
        assert_eq!(KernelKind::Bf16.packed_bytes(64, 256), Ok(32768));
        assert_eq!(KernelKind::Q8_0.packed_bytes(64, 256), Ok(17408));
        assert_eq!(KernelKind::Bf16.packed_bytes(1, 1), Ok(2));
        assert_eq!(KernelKind::Q8_0.packed_bytes(1, 32), Ok(34));
        assert_eq!(KernelKind::Bf16.packed_bytes(4096, 4096), Ok(33_554_432));
        assert_eq!(KernelKind::Q8_0.packed_bytes(4096, 4096), Ok(17_825_792));
        for (rows, cols) in [(3072, 1024), (1024, 3072)] {
            assert_eq!(KernelKind::Q8_0.packed_bytes(rows, cols), Ok(3_342_336));
        }
        for kind in [KernelKind::Bf16, KernelKind::Q8_0] {
            for (rows, cols) in [
                (0, 32),
                (1, 0),
                (4097, 32),
                (1, 4097),
                (usize::MAX, usize::MAX),
            ] {
                assert!(kind.packed_bytes(rows, cols).is_err());
            }
        }
        for cols in [1, 31, 33, 255] {
            assert!(KernelKind::Q8_0.packed_bytes(1, cols).is_err());
        }
    }

    // Decode only the audited admission/indexing prologue, directly from embedded
    // words. This is not arithmetic/GPU emulation. Unexpected instructions fail
    // the test rather than silently inheriting the intended source semantics.
    fn active_indices(code: &[u32], scalars: [u32; 2], group: u32) -> Vec<u32> {
        active_indices_with_output(code, scalars, group, 0)
    }

    fn active_indices_with_output(
        code: &[u32],
        scalars: [u32; 2],
        group: u32,
        output_bytes: u32,
    ) -> Vec<u32> {
        let mut scalar = [0; 20];
        scalar[10] = output_bytes;
        scalar[12..14].copy_from_slice(&scalars);
        scalar[14] = group;
        let mut indices: [u32; 64] = std::array::from_fn(|lane| lane as u32);
        let mut active = [true; 64];
        let mut condition = false;
        let mut predicate = [false; 64];
        let mut pc = 0;
        loop {
            let word = code[pc];
            let mut width = 1;
            match word {
                0xb0802006 | 0xbf800007 => {}
                0xbf08bf0e => condition = scalar[14] > 63,
                0xbf089f0e => condition = scalar[14] > 31,
                0xbf08830e => condition = scalar[14] > 3,
                0xbf08870e => condition = scalar[14] > 7,
                0xbf08800e => condition = scalar[14] > 0,
                0xbf08840d => condition = scalar[13] > 4,
                0xbf06800a => condition = scalar[10] == 0,
                0xbf06800c => condition = scalar[12] == 0,
                0xbf06800d => condition = scalar[13] == 0,
                0xbf06800f => condition = scalar[15] == 0,
                0xbf06810d => condition = scalar[13] == 1,
                0xbf08ff0a | 0xbf08ff0c | 0xbf08ff0d | 0xbf08ff0e => {
                    let register = (word & 0xff) as usize;
                    condition = scalar[register] > code[pc + 1];
                    width = 2;
                }
                0x8b0f9f0c => scalar[15] = scalar[12] & 31,
                0x8b0f870a => scalar[15] = scalar[10] & 7,
                0x8b0f830a => scalar[15] = scalar[10] & 3,
                0x850f830a => scalar[15] = scalar[10] >> 3,
                0x850f820a => scalar[15] = scalar[10] >> 2,
                0xbf0a0c0d => condition = scalar[13] < scalar[12],
                0xbf0a0f0d => condition = scalar[13] < scalar[15],
                0x360000ff => {
                    for index in &mut indices {
                        *index &= code[pc + 1];
                    }
                    width = 2;
                }
                0xd6460000 => {
                    assert_eq!(code[pc + 1], 0x04010c0e);
                    for index in &mut indices {
                        *index = (scalar[14] << 6).wrapping_add(*index);
                    }
                    width = 2;
                }
                0x7c98000c | 0x7c98000d | 0x7c98000f => {
                    let count = scalar[(word & 0xff) as usize];
                    predicate = indices.map(|index| count > index);
                }
                0x8bfe6a7e => {
                    for (lane, enabled) in active.iter_mut().enumerate() {
                        *enabled &= predicate[lane];
                    }
                }
                0xbfb60003 => return Vec::new(), // Refused before any memory access.
                0xd72c0001 | 0x850f850c | 0x30020082 | 0x30000082 => {
                    return indices
                        .into_iter()
                        .zip(active)
                        .filter_map(|(index, enabled)| enabled.then_some(index))
                        .collect();
                }
                branch if [0xbfa10000, 0xbfa20000, 0xbfa50000].contains(&(branch & 0xffff0000)) => {
                    let taken = match branch & 0xffff0000 {
                        0xbfa10000 => !condition,
                        0xbfa20000 => condition,
                        _ => !active.iter().any(|enabled| *enabled),
                    };
                    if taken {
                        pc = (pc as isize + 1 + (branch as i16 as isize)) as usize;
                        assert_eq!(code[pc], 0xbfb60003);
                        continue;
                    }
                }
                unexpected => panic!("unknown admission instruction {unexpected:#x} at {pc}"),
            }
            pc += width;
        }
    }

    #[test]
    fn encoded_admission_covers_each_row_once_and_masks_workgroup_tails() {
        for count in [1, 63, 64, 65, 127, 128, 129, 1024, 3072, 4095, 4096] {
            let groups = dispatch_groups(count).unwrap();
            assert!((1..=64).contains(&groups));
            for (code, scalars) in [
                (BF16_CODE, [4096, count as u32]),
                (Q8_0_CODE, [4096, count as u32]),
                (SWIGLU_CODE, [count as u32, 1]),
                (SCALED_ADD_CODE, [count as u32, 1]),
                (RMSNORM_CODE, [count as u32, 1]),
                (MULTIPLY_CODE, [count as u32, 1]),
            ] {
                let rows: Vec<_> = (0..groups)
                    .flat_map(|group| active_indices(code, scalars, group))
                    .collect();
                assert_eq!(rows, (0..count as u32).collect::<Vec<_>>());
                // Excess groups are either all masked or rejected by the group cap.
                for group in [groups, 64, 128, 256, u32::MAX] {
                    assert!(active_indices(code, scalars, group).is_empty());
                }
            }
        }
        for count in [0, 4097, usize::MAX] {
            assert!(dispatch_groups(count).is_err());
        }
    }

    #[test]
    fn encoded_admission_refuses_invalid_shapes_before_memory_access() {
        for code in [BF16_CODE, Q8_0_CODE] {
            for scalars in [[0, 1], [4097, 1], [32, 0], [32, 4097], [u32::MAX, u32::MAX]] {
                assert!(active_indices(code, scalars, 0).is_empty());
            }
        }
        for cols in [1, 31, 33, 4095] {
            assert!(active_indices(Q8_0_CODE, [cols, 1], 0).is_empty());
        }
        for code in [SWIGLU_CODE, SCALED_ADD_CODE, RMSNORM_CODE, MULTIPLY_CODE] {
            for scalars in [[0, 1], [4097, 1], [1, 0], [1, 2], [u32::MAX, 1]] {
                assert!(active_indices(code, scalars, 0).is_empty());
            }
        }
    }

    #[test]
    fn qwen35_norm_covers_its_wide_vector_without_widening_dense_programs() {
        for count in [1u32, 63, 64, 65, 4096, 4097, 5119, 5120] {
            let groups = count.div_ceil(64);
            let indices: Vec<_> = (0..groups)
                .flat_map(|group| active_indices(RMSNORM_QWEN35_CODE, [count, 1], group))
                .collect();
            assert_eq!(indices, (0..count).collect::<Vec<_>>());
            for group in [groups, 80, 256, u32::MAX] {
                assert!(active_indices(RMSNORM_QWEN35_CODE, [count, 1], group).is_empty());
            }
        }
        for args in [[0, 1], [5121, 1], [5120, 0], [5120, 2], [u32::MAX, 1]] {
            assert!(active_indices(RMSNORM_QWEN35_CODE, args, 0).is_empty());
        }
        assert!(active_indices(RMSNORM_CODE, [5120, 1], 0).is_empty());
        let dense = program_bytes();
        let probe = qwen35_program_bytes();
        for index in [2, 3, 5, 6, 7, 8, 9] {
            assert_eq!(dense[index], probe[index]);
        }
        assert_ne!(dense[4], probe[4]);
    }

    #[test]
    fn exact_register_contract_preserves_serial_reference_numerics() {
        assert_eq!(RSRC1 & 0x3f, 1); // eight VGPRs for wave64
        assert_eq!((RSRC1 >> 6) & 0xf, 0); // reserved SGPR allocation bits
        assert_eq!((RSRC1 >> 12) & 0xff, 0xf0); // nearest, preserve denormals
        assert_eq!(RSRC1 & (1 << 21), 0); // no DX10 NaN clamp
        assert_ne!(RSRC1 & (1 << 23), 0); // IEEE mode
        assert_eq!(RSRC1 & (1 << 29), 0); // CU mode, not WGP
        assert_ne!(RSRC1 & (1 << 30), 0); // MEM_ORDERED
        assert_eq!(RSRC2, (14 << 1) | (1 << 7));
        assert_eq!(RSRC3, 0);
        assert_eq!(
            RAW_SRD_WORD3,
            4 | (5 << 3) | (6 << 6) | (7 << 9) | (22 << 12) | (3 << 28)
        );
        assert_eq!(THREADS, [64, 1, 1]);
        assert_eq!(GROUPS, [1, 1, 1]);
    }

    #[test]
    fn shader_payloads_have_bounded_load_store_and_termination_encodings() {
        assert_eq!(bf16_bytes(), KernelKind::Bf16.shader_bytes());
        assert_eq!(q8_0_bytes(), KernelKind::Q8_0.shader_bytes());
        for (kind, expected_code_words) in [(KernelKind::Bf16, 57), (KernelKind::Q8_0, 75)] {
            let code = kind.code();
            assert_eq!(code.len(), expected_code_words);
            assert_eq!(code[0], 0xb0802006); // gfx11 wave64
            assert_eq!(
                &code[code.len() - 3..],
                &[0xbc7c0000, 0xbfb60003, 0xbfb00000]
            );
            assert_eq!(code.iter().filter(|word| **word == 0xe0680000).count(), 1);
            assert!(code.contains(&0xbf8903f7)); // vmcnt(0) before ALU use
            assert!(code.contains(&0x10080b04)); // separate multiply
            assert!(code.contains(&0x06060903)); // separate add
            let bytes = kind.shader_bytes();
            assert_eq!(bytes.len(), (expected_code_words + 64) * 4);
            assert!(bytes.len() < 4096);
            assert!(bytes[code.len() * 4..]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|chunk| *chunk == CODE_END.to_le_bytes()));
        }
        assert!(Q8_0_CODE.contains(&0xe0440000)); // sign-extending i8 load
        assert!(Q8_0_CODE.contains(&0x7e0c1706)); // f16 scale -> f32
        assert!(Q8_0_CODE.contains(&0x7e080b04)); // signed quant -> f32
    }

    #[test]
    fn elementwise_programs_use_bounded_memory_and_stable_library_offsets() {
        for (code, bytes, words, loads) in [
            (SWIGLU_CODE, swiglu_bytes(), 78, 3),
            (SCALED_ADD_CODE, scaled_add_bytes(), 39, 3),
            (RMSNORM_CODE, rmsnorm_bytes(), 71, 3),
            (MULTIPLY_CODE, multiply_bytes(), 35, 2),
        ] {
            assert_eq!(code.len(), words);
            assert_eq!(code[0], 0xb0802006);
            assert_eq!(
                &code[code.len() - 3..],
                &[0xbc7c0000, 0xbfb60003, 0xbfb00000]
            );
            assert!(code.windows(2).any(|words| words == [0xbf08ff0c, 4096]));
            assert!(code.contains(&0xbf06810d)); // s13 == 1
            assert!(code.contains(&0xbf08bf0e)); // workgroup X <= 63
            assert!(code.contains(&0x7c98000c)); // global index < s12
            assert!(code.contains(&0xbf8903f7)); // load completion before ALU
            assert_eq!(
                code.iter().filter(|word| **word == 0xe0500000).count(),
                loads
            );
            assert_eq!(code.iter().filter(|word| **word == 0xe0680000).count(), 1);
            assert!(bytes[words * 4..]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|chunk| *chunk == CODE_END.to_le_bytes()));
        }
        assert!(SCALED_ADD_CODE.contains(&0x10060702)); // scale * input
        assert!(SCALED_ADD_CODE.contains(&0x06080704)); // old output + product
        assert!(SWIGLU_CODE.contains(&0x7e0c4b05)); // exp2
        assert!(SWIGLU_CODE.contains(&0x7e0c5506)); // reciprocal
        assert!(SWIGLU_CODE.contains(&0xbfb8aa3b)); // -log2(e)
        assert!(SWIGLU_CODE.contains(&0x2f800000)); // 2^-32 rescaling
        assert!(SWIGLU_CODE.contains(&0x6f800000)); // denominator > 2^96
        assert!(SWIGLU_CODE.windows(9).any(|words| words
            == [
                0x100c0d07, 0xbf800007, // denominator *= scale
                0x7e0c5506, 0xbf800007, // reciprocal of scaled denominator
                0x10040d02, 0xbf800007, // gate *= reciprocal
                0x10040507, 0xbf800007, // gate *= scale
                0x10040702, // gate *= up
            ]));
        let mut end = 0;
        let mut offsets = Vec::new();
        for bytes in program_bytes() {
            let offset = (end + 255) & !255;
            offsets.push(offset);
            end = offset + bytes.len();
        }
        assert_eq!(
            offsets,
            [0, 512, 1280, 2048, 2560, 3328, 3840, 4608, 5376, 6400, 7168]
        );
        assert_eq!(offsets[5] + multiply_bytes().len(), 3724);
        assert_eq!(offsets[6] + rope_bytes().len(), 4388);
        assert_eq!(offsets[9] + attention_value_bytes().len(), 6928);
        assert_eq!(end, 7624);
        assert_eq!(end.div_ceil(4096) * 4096, 8192);
    }

    // Decode the new row program's integer address body directly from the words.
    // Unexpected instructions fail closed; arithmetic rounding is checked by the
    // separate staged numerical model, not by this address trace.
    fn q8_row_addresses(index: u32) -> [u32; 3] {
        let mut vector = [0; 6];
        vector[0] = index;
        let mut addresses = Vec::new();
        let mut pc = 24;
        loop {
            let mut width = 1;
            match Q8_0_ROW_CODE[pc] {
                0xbf800007 | 0xbf8903f7 | 0x7e081704 | 0x7e0a0b05 | 0x100a0b04 => {}
                0x30020082 => vector[1] = vector[0] << 2,
                0x32040085 => vector[2] = vector[0] >> 5,
                0xd72c0002 => {
                    assert_eq!(Q8_0_ROW_CODE[pc + 1], 0x000204a2);
                    vector[2] *= 34;
                    width = 2;
                }
                0x3606009f => vector[3] = vector[0] & 31,
                0x4a060702 => vector[3] += vector[2],
                0xe0480000 => {
                    assert_eq!(Q8_0_ROW_CODE[pc + 1], 0x80400402);
                    addresses.push(vector[2]);
                    width = 2;
                }
                0xe0440002 => {
                    assert_eq!(Q8_0_ROW_CODE[pc + 1], 0x80400503);
                    addresses.push(vector[3] + 2);
                    width = 2;
                }
                0xe0680000 => {
                    assert_eq!(Q8_0_ROW_CODE[pc + 1], 0x80420501);
                    addresses.push(vector[1]);
                    return addresses.try_into().unwrap();
                }
                unexpected => {
                    panic!("unexpected Q8 row address instruction {unexpected:#x} at {pc}")
                }
            }
            pc += width;
        }
    }

    #[test]
    fn encoded_q8_row_covers_exact_blocks_and_masks_tails_before_memory() {
        for columns in [32, 64, 96, 128, 1024, 4064, 4096] {
            let groups = dispatch_groups(columns).unwrap();
            let packed_bytes = KernelKind::Q8_0.packed_bytes(1, columns).unwrap();
            let indices: Vec<_> = (0..groups)
                .flat_map(|group| active_indices(Q8_0_ROW_CODE, [columns as u32, 1], group))
                .collect();
            assert_eq!(indices, (0..columns as u32).collect::<Vec<_>>());
            for index in indices {
                let [scale, quant, output] = q8_row_addresses(index);
                assert_eq!(scale, index / 32 * 34);
                assert_eq!(quant, scale + 2 + index % 32);
                assert_eq!(output, index * 4);
                assert!(scale as usize + 2 <= packed_bytes);
                assert!((quant as usize) < packed_bytes);
                assert!(output as usize + 4 <= columns * 4);
                // Odd blocks begin at byte 2 modulo 4: never round down to a word.
                assert_eq!(scale % 4, index / 32 % 2 * 2);
            }
            for group in [groups, 64, u32::MAX] {
                assert!(active_indices(Q8_0_ROW_CODE, [columns as u32, 1], group).is_empty());
            }
        }
        for columns in [0, 1, 31, 33, 95, 4095, 4097, u32::MAX] {
            assert!(active_indices(Q8_0_ROW_CODE, [columns, 1], 0).is_empty());
        }
        for flags in [0, 2, u32::MAX] {
            assert!(active_indices(Q8_0_ROW_CODE, [32, flags], 0).is_empty());
        }
    }

    #[test]
    fn q8_row_encodes_exact_widening_multiply_and_no_dummy_load() {
        assert_eq!(Q8_0_ROW_CODE.len(), 50);
        assert_eq!(
            &Q8_0_ROW_CODE[35..],
            &[
                0xe0480000, 0x80400402, // unaligned-to-word F16 scale load
                0xe0440002, 0x80400503, // signed i8, byte offset +2
                0xbf8903f7, // both loads complete
                0x7e081704, 0x7e0a0b05, 0xbf800007, // exact F16/i8 -> F32
                0x100a0b04, 0xbf800007, // scale * signed quant
                0xe0680000, 0x80420501, // only output store
                0xbc7c0000, 0xbfb60003, 0xbfb00000,
            ]
        );
        // The exact suffix above has only two SRD0 loads; SRD1 is retained for
        // the fixed ABI but cannot become a source or expanded-weight buffer.
        let bytes = q8_0_row_bytes();
        assert_eq!(bytes.len(), 456);
        assert!(bytes[Q8_0_ROW_CODE.len() * 4..]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|chunk| *chunk == CODE_END.to_le_bytes()));
    }

    // Decode the RoPE integer address-selection body from embedded words. This
    // traces descriptor-relative addresses; floating arithmetic is tested below.
    fn rope_addresses(pairs: u32, flags: u32, index: u32) -> ([u32; 4], u32) {
        let mut scalar = [0_u32; 20];
        scalar[12] = pairs;
        scalar[13] = flags;
        let mut offsets = [index, 0];
        let mut loads = Vec::new();
        let mut sign = 0;
        let mut condition = false;
        let mut pc = 23;
        loop {
            let mut width = 1;
            match ROPE_CODE[pc] {
                0xbf800007 | 0xbf8903f7 => {}
                0x30000082 => offsets[0] <<= 2,
                0x30000081 => offsets[0] <<= 1,
                0x4a020084 => offsets[1] = 4 + offsets[0],
                0x4a020010 => offsets[1] = scalar[16] + offsets[0],
                0x808f810d => scalar[15] = scalar[13] - 1,
                0x84109e0f => scalar[16] = scalar[15] << 30,
                0x8b10ff10 => {
                    scalar[16] &= ROPE_CODE[pc + 1];
                    width = 2;
                }
                0x8c10f210 => scalar[16] |= 1.0_f32.to_bits(),
                0x8410820c => scalar[16] = scalar[12] << 2,
                0x100a0a10 => sign = scalar[16],
                0xbf0d800f => condition = scalar[15] & 1 != 0,
                0xe0500000 => {
                    let offset = match ROPE_CODE[pc + 1] {
                        0x80400400 | 0x80410500 | 0x80420200 => offsets[0],
                        0x80420301 => offsets[1],
                        unexpected => panic!("unexpected RoPE load {unexpected:#x}"),
                    };
                    loads.push(offset);
                    width = 2;
                }
                branch if [0xbfa00000, 0xbfa20000].contains(&(branch & 0xffff0000)) => {
                    if branch & 0xffff0000 == 0xbfa00000 || condition {
                        pc = (pc as isize + 1 + (branch as i16 as isize)) as usize;
                        continue;
                    }
                }
                0x100c0902 => return (loads.try_into().unwrap(), sign),
                unexpected => panic!("unexpected RoPE address instruction {unexpected:#x} at {pc}"),
            }
            pc += width;
        }
    }

    #[test]
    fn encoded_rope_covers_disjoint_pairs_and_refuses_invalid_flags_or_geometry() {
        assert_eq!(MAX_ROPE_PAIRS, 2048);
        for pairs in [1, 31, 32, 63, 64, 65, 127, 128, 1024, 2047, 2048] {
            let groups = dispatch_groups(pairs).unwrap();
            assert!(groups <= 32);
            for flags in 1..=4 {
                let scalars = [pairs as u32, flags];
                let indices: Vec<_> = (0..groups)
                    .flat_map(|group| active_indices(ROPE_CODE, scalars, group))
                    .collect();
                assert_eq!(indices, (0..pairs as u32).collect::<Vec<_>>());
                let mut coordinates = Vec::new();
                for index in indices {
                    let (addresses, sign) = rope_addresses(pairs as u32, flags, index);
                    assert_eq!(addresses[..2], [index * 4, index * 4]);
                    assert!(addresses[..2]
                        .iter()
                        .all(|offset| *offset + 4 <= pairs as u32 * 4));
                    let expected = if (flags - 1) & 1 == 0 {
                        [index * 8, index * 8 + 4]
                    } else {
                        [index * 4, (index + pairs as u32) * 4]
                    };
                    assert_eq!(addresses[2..], expected);
                    assert!(addresses[2..]
                        .iter()
                        .all(|offset| *offset + 4 <= pairs as u32 * 8));
                    coordinates.extend(addresses[2..].iter().map(|offset| *offset / 4));
                    let expected_sign = if flags <= 2 { 1.0_f32 } else { -1.0_f32 };
                    assert_eq!(sign, expected_sign.to_bits());
                }
                coordinates.sort_unstable();
                assert_eq!(coordinates, (0..pairs as u32 * 2).collect::<Vec<_>>());
                for group in [groups, 32, 63, u32::MAX] {
                    assert!(active_indices(ROPE_CODE, scalars, group).is_empty());
                }
            }
        }
        for pairs in [0, 2049, 4096, u32::MAX] {
            for flags in 1..=4 {
                assert!(active_indices(ROPE_CODE, [pairs, flags], 0).is_empty());
            }
        }
        for flags in [0, 5, u32::MAX] {
            assert!(active_indices(ROPE_CODE, [1, flags], 0).is_empty());
        }
    }

    #[test]
    fn rope_encodes_complete_pair_loads_then_separate_f32_products_and_stores() {
        assert_eq!(ROPE_CODE.len(), 73);
        assert_eq!(
            &ROPE_CODE[49..],
            &[
                0xe0500000, 0x80420200, 0xe0500000, 0x80420301, // both original x values
                0xbf8903f7, // all loads complete
                0x100c0902, 0xbf800007, // a*c
                0x100e0b03, 0xbf800007, // b*s
                0x080c0f06, 0xbf800007, // first = a*c - b*s
                0x10040b02, 0xbf800007, // a*s
                0x10060903, 0xbf800007, // b*c
                0x06040702, 0xbf800007, // second = a*s + b*c
                0xe0680000, 0x80420600, 0xe0680000, 0x80420201, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
            ]
        );
        let bytes = rope_bytes();
        assert_eq!(bytes.len(), 548);
        assert!(bytes[ROPE_CODE.len() * 4..]
            .as_chunks::<4>()
            .0
            .iter()
            .all(|chunk| *chunk == CODE_END.to_le_bytes()));
    }

    // Trace the encoded softmax output loop after its unchanged reductions.
    // Floating instructions are recognized but not emulated: the separate staged
    // model tests their rounding. This checks actual load/store offsets, monotonic
    // EXEC masks, both branch paths and finite termination from embedded words.
    fn softmax_output_addresses(
        positions: u32,
        nonfinite_maximum: bool,
    ) -> (Vec<u32>, Vec<(u32, bool)>) {
        let code = ATTENTION_SOFTMAX_CODE;
        let indices = active_indices_with_output(code, [positions, 1], 0, positions * 4);
        let mut active = std::array::from_fn::<_, 64, _>(|lane| indices.contains(&(lane as u32)));
        let mut score = std::array::from_fn::<_, 64, _>(|lane| lane as u32 * 8);
        let mut output = std::array::from_fn::<_, 64, _>(|lane| lane as u32 * 4);
        let zero_branch = code
            .iter()
            .position(|word| word & 0xffff0000 == 0xbfa40000)
            .unwrap();
        let zero_pc = (zero_branch as isize + 1 + (code[zero_branch] as i16 as isize)) as usize;
        let output_pc = code
            .windows(2)
            .position(|pair| pair == [0xe0540000, 0x80410400])
            .unwrap();
        let mut pc = if nonfinite_maximum {
            zero_pc
        } else {
            output_pc
        };
        let mut zero_path = false;
        let mut stored_zero = false;
        let mut condition = false;
        let mut predicate = [false; 64];
        let mut reads = Vec::new();
        let mut writes = Vec::new();
        // At most 64 decoded instructions per lane-strided output iteration.
        for _ in 0..positions.div_ceil(64) * 64 + 16 {
            let mut width = 1;
            match code[pc] {
                0xbf800007 | 0xbf8903f7 | 0xbc7c0000 => {}
                0xe0540000 => {
                    assert_eq!(code[pc + 1], 0x80410400);
                    for lane in 0..64 {
                        if active[lane] {
                            reads.push(score[lane]);
                        }
                    }
                    stored_zero = false;
                    width = 2;
                }
                0xe0680000 => {
                    assert_eq!(code[pc + 1], 0x80420401);
                    for lane in 0..64 {
                        if active[lane] {
                            writes.push((output[lane], stored_zero));
                        }
                    }
                    width = 2;
                }
                0xd7280004 => {
                    assert_eq!(code[pc + 1], 0x00020d04);
                    width = 2;
                }
                0x360a08ff | 0x7c9a0aff => {
                    assert_eq!(code[pc + 1], 0x7f800000);
                    width = 2;
                }
                0x100808ff => {
                    assert_eq!(code[pc + 1], 0x3fb8aa3b);
                    width = 2;
                }
                0x7e081f04 | 0x02080902 | 0x08080504 | 0x7e084b04 | 0x02080880 | 0x7e0a5503
                | 0x10080b04 => {}
                0xbe910081 => zero_path = true,
                0x7e080280 => stored_zero = true,
                0x4a0000ff | 0x4a0202ff => {
                    let offsets = if code[pc] == 0x4a0000ff {
                        &mut score
                    } else {
                        &mut output
                    };
                    let step = if code[pc] == 0x4a0000ff { 512 } else { 256 };
                    assert_eq!(code[pc + 1], step);
                    for lane in 0..64 {
                        if active[lane] {
                            offsets[lane] += step;
                        }
                    }
                    width = 2;
                }
                0x7c980212 => predicate = output.map(|offset| positions * 4 > offset),
                0x8bfe6a7e => {
                    for lane in 0..64 {
                        active[lane] &= predicate[lane];
                    }
                }
                0xbf068111 => condition = zero_path,
                0xbfb60003 => {
                    assert!(!active.iter().any(|enabled| *enabled));
                    return (reads, writes);
                }
                branch if [0xbfa00000, 0xbfa20000, 0xbfa50000].contains(&(branch & 0xffff0000)) => {
                    let taken = match branch & 0xffff0000 {
                        0xbfa00000 => true,
                        0xbfa20000 => condition,
                        _ => !active.iter().any(|enabled| *enabled),
                    };
                    if taken {
                        pc = (pc as isize + 1 + (branch as i16 as isize)) as usize;
                        continue;
                    }
                }
                unexpected => panic!("unknown softmax output instruction {unexpected:#x} at {pc}"),
            }
            pc += width;
        }
        panic!("softmax output loop did not terminate");
    }

    #[test]
    fn encoded_softmax_single_wave_visits_each_strided_output_once_and_masks_before_loads() {
        for positions in [
            1, 2, 31, 32, 63, 64, 65, 127, 128, 255, 256, 257, 341, 342, 511, 512, 513, 1023, 1024,
            1025, 2047, 2048, 4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384,
        ] {
            let admitted = active_indices_with_output(
                ATTENTION_SOFTMAX_CODE,
                [positions, 1],
                0,
                positions * 4,
            );
            assert_eq!(admitted, (0..positions.min(64)).collect::<Vec<_>>());
            for group in [1, 3, 4, 7, 8, u32::MAX] {
                assert!(active_indices_with_output(
                    ATTENTION_SOFTMAX_CODE,
                    [positions, 1],
                    group,
                    positions * 4
                )
                .is_empty());
            }
            for nonfinite_maximum in [false, true] {
                let (reads, writes) = softmax_output_addresses(positions, nonfinite_maximum);
                assert_eq!(
                    writes,
                    (0..positions)
                        .map(|i| (i * 4, nonfinite_maximum))
                        .collect::<Vec<_>>()
                );
                if nonfinite_maximum {
                    assert!(reads.is_empty());
                } else {
                    assert_eq!(reads, (0..positions).map(|i| i * 8).collect::<Vec<_>>());
                }
                assert!(reads.iter().all(|offset| offset + 8 <= positions * 8));
                assert!(writes.iter().all(|(offset, _)| offset + 4 <= positions * 4));
            }
        }
    }

    #[test]
    fn encoded_attention_admits_only_bounded_exact_output_extents_and_masks_tails() {
        for positions in [
            1, 31, 32, 63, 64, 65, 127, 128, 255, 256, 257, 341, 342, 511, 512, 513, 1023, 1024,
            1025, 2047, 2048, 4095, 4096, 4097, 8191, 8192, 8193, 16383, 16384,
        ] {
            for dim in [1, 64, 65, 128, 256] {
                for stride in [dim, 4096] {
                    for (code, scalars, output_bytes, count) in [
                        (
                            ATTENTION_SCORE_CODE,
                            [dim, stride],
                            positions * 8,
                            positions,
                        ),
                        (ATTENTION_VALUE_CODE, [positions, stride], dim * 4, dim),
                    ] {
                        let groups = if code == ATTENTION_SCORE_CODE {
                            attention_dispatch_groups(count as usize).unwrap()
                        } else {
                            dispatch_groups(count as usize).unwrap()
                        };
                        let limit = if code == ATTENTION_SCORE_CODE { 256 } else { 4 };
                        assert!(groups <= limit);
                        let indices: Vec<_> = (0..groups)
                            .flat_map(|group| {
                                active_indices_with_output(code, scalars, group, output_bytes)
                            })
                            .collect();
                        assert_eq!(indices, (0..count).collect::<Vec<_>>());
                        for group in [groups, limit, u32::MAX] {
                            assert!(
                                active_indices_with_output(code, scalars, group, output_bytes)
                                    .is_empty()
                            );
                        }
                    }
                    // Score and value loop address formulas use the same prefix
                    // extent, bounded by the admitted counts/stride above.
                    let prefix_bytes = ((positions - 1) * stride + dim) * 4;
                    for position in 0..positions {
                        for coordinate in [0, dim - 1] {
                            let address = (position * stride + coordinate) * 4;
                            assert!(address + 4 <= prefix_bytes);
                        }
                    }
                    assert!(prefix_bytes <= 268_420_096);
                }
            }
        }
        assert_eq!(MAX_ATTENTION_POSITIONS, 16384);
        for count in [0, 16385, u32::MAX] {
            assert!(attention_dispatch_groups(count as usize).is_err());
            for code in [ATTENTION_VALUE_CODE, ATTENTION_SOFTMAX_CODE] {
                assert!(active_indices_with_output(code, [count, 1], 0, 8).is_empty());
            }
        }
        for dim in [0, 257, u32::MAX] {
            assert!(active_indices_with_output(ATTENTION_SCORE_CODE, [dim, 4096], 0, 8).is_empty());
        }
        for (code, valid_bytes) in [(ATTENTION_SCORE_CODE, 8), (ATTENTION_VALUE_CODE, 4)] {
            for stride in [0, 4097, u32::MAX] {
                assert!(active_indices_with_output(code, [1, stride], 0, valid_bytes).is_empty());
            }
        }
        assert!(active_indices_with_output(ATTENTION_SCORE_CODE, [2, 1], 0, 8).is_empty());
        assert!(active_indices_with_output(ATTENTION_VALUE_CODE, [1, 1], 0, 8).is_empty());
        for bytes in [0, 1, 4, 7, 9, 131073, 131080, u32::MAX] {
            assert!(active_indices_with_output(ATTENTION_SCORE_CODE, [1, 1], 0, bytes).is_empty());
        }
        for bytes in [0, 1, 3, 5, 1025, 1028, u32::MAX] {
            assert!(
                active_indices_with_output(ATTENTION_VALUE_CODE, [1, 4096], 0, bytes).is_empty()
            );
        }
        for flags in [0, 2, u32::MAX] {
            assert!(
                active_indices_with_output(ATTENTION_SOFTMAX_CODE, [1, flags], 0, 4).is_empty()
            );
        }
    }

    #[test]
    fn attention_encoding_preserves_f64_reductions_and_f32_softmax_stages() {
        let reduction = [
            0x7e082104, 0xbf800007, // F32 first operand -> F64
            0x7e0c2106, 0xbf800007, // F32 second operand -> F64
            0xd7280004, 0x00020d04, 0xbf800007, // F64 product
            0xd7270002, 0x00020902, 0xbf800007, // separate F64 sum
        ];
        for (code, bytes, words) in [
            (ATTENTION_SCORE_CODE, attention_score_bytes(), 69),
            (ATTENTION_SOFTMAX_CODE, attention_softmax_bytes(), 145),
            (ATTENTION_VALUE_CODE, attention_value_bytes(), 68),
        ] {
            assert_eq!(code.len(), words);
            assert_eq!(&code[words - 2..], &[0xbfb60003, 0xbfb00000]);
            assert!(code.contains(&0xbc7c0000));
            assert!(bytes[words * 4..]
                .as_chunks::<4>()
                .0
                .iter()
                .all(|chunk| *chunk == CODE_END.to_le_bytes()));
        }
        for code in [ATTENTION_SCORE_CODE, ATTENTION_VALUE_CODE] {
            assert!(code
                .windows(reduction.len())
                .any(|words| words == reduction));
            assert!(!code.contains(&0x06060903)); // reduction cannot become F32
        }
        assert!(ATTENTION_SCORE_CODE
            .windows(2)
            .any(|words| words == [0xe06c0000, 0x80420200]));
        assert!(!ATTENTION_SCORE_CODE.contains(&0x7e041f02)); // scores stay F64
        assert!(ATTENTION_VALUE_CODE.contains(&0x7e041f02)); // only final output narrows
        assert!(ATTENTION_VALUE_CODE.contains(&0x8412820d)); // stride bytes = s13 << 2
        assert!(ATTENTION_VALUE_CODE.contains(&0x80111211)); // advance V by stride
        let finite_exp = [
            0x02080902, 0xbf800007, // replace nonfinite score with max
            0x08080504, 0xbf800007, // subtract max
            0x100808ff, 0x3fb8aa3b, 0xbf800007, // multiply log2(e)
            0x7e084b04, 0xbf800007, // exp2
            0x02080880, 0xbf800007, // nonfinite probability becomes zero
        ];
        assert_eq!(
            ATTENTION_SOFTMAX_CODE
                .windows(finite_exp.len())
                .filter(|words| *words == finite_exp)
                .count(),
            2
        );
        assert_eq!(
            ATTENTION_SOFTMAX_CODE
                .iter()
                .filter(|word| **word == 0x7e081f04)
                .count(),
            3
        );
        assert!(ATTENTION_SOFTMAX_CODE.contains(&0x06060903)); // ordered F32 exp sum
        assert!(ATTENTION_SOFTMAX_CODE.contains(&0x7e0a5503)); // F32 reciprocal sum
        let branch = ATTENTION_SOFTMAX_CODE
            .iter()
            .position(|word| word & 0xffff0000 == 0xbfa40000)
            .unwrap();
        let target =
            (branch as isize + 1 + (ATTENTION_SOFTMAX_CODE[branch] as i16 as isize)) as usize;
        assert_eq!(
            &ATTENTION_SOFTMAX_CODE[target..target + 4],
            &[0xbe910081, 0x7e080280, 0xbf800007, 0xe0680000]
        );
        // The two reduction loops remain byte-for-byte the historical ordered
        // bodies. The new output loop must never feed back into either reduction.
        let ordered_max = [
            0xe0540000, 0x10010400, 0xbf8903f7, 0xd7280004, 0x00020d04, 0xbf800007, 0x7e081f04,
            0xbf800007, 0x7c280504, 0xbf800007, 0x02040902, 0xbf800007, 0x800f810f, 0x80108810,
            0xbf0a0c0f, 0xbfa2fff0,
        ];
        let ordered_sum = [
            0xe0540000, 0x10010400, 0xbf8903f7, 0xd7280004, 0x00020d04, 0xbf800007, 0x7e081f04,
            0xbf800007, 0x360a08ff, 0x7f800000, 0xbf800007, 0x7c9a0aff, 0x7f800000, 0xbf800007,
            0x02080902, 0xbf800007, 0x08080504, 0xbf800007, 0x100808ff, 0x3fb8aa3b, 0xbf800007,
            0x7e084b04, 0xbf800007, 0x02080880, 0xbf800007, 0x06060903, 0xbf800007, 0x800f810f,
            0x80108810, 0xbf0a0c0f, 0xbfa2ffe1,
        ];
        assert!(ATTENTION_SOFTMAX_CODE
            .windows(ordered_max.len())
            .any(|words| words == ordered_max));
        assert!(ATTENTION_SOFTMAX_CODE
            .windows(ordered_sum.len())
            .any(|words| words == ordered_sum));
        assert!(ATTENTION_SOFTMAX_CODE.contains(&0x8412820c)); // s18 = positions * 4
        assert!(ATTENTION_SOFTMAX_CODE.contains(&0xbe910080)); // initially finite path
    }

    #[test]
    fn rmsnorm_encodes_serial_f64_reduction_and_separate_weight_application() {
        // Ordered input load, exact widening, F64 square then F64 sum. The loop
        // walks the whole input in increasing order independently of output lane.
        assert_eq!(
            &RMSNORM_CODE[27..42],
            &[
                0xe0500000, 0x10010600, // input[s16], no lane offset
                0xbf8903f7, 0x7e082106, 0xbf800007, // load wait, F32 -> F64
                0xd7280004, 0x00020904, 0xbf800007, // F64 square
                0xd7270002, 0x00020902, 0xbf800007, // separate F64 addition
                0x800f810f, 0x80108410, // element += 1, byte offset += 4
                0xbf0a0c0f, 0xbfa2fff1, // loop while element < len, back to load
            ]
        );
        assert!(RMSNORM_CODE.contains(&0x7e082c0c)); // u32 len -> F64
        assert!(RMSNORM_CODE.contains(&0x7e085f04)); // F64 reciprocal len
        assert!(RMSNORM_CODE.contains(&0x7e046302)); // F64 reciprocal sqrt
        assert!(RMSNORM_CODE.contains(&0x7e041f02)); // final inverse -> F32
        assert!(RMSNORM_CODE.contains(&0x10040506)); // input * inverse
        assert!(MULTIPLY_CODE.contains(&0x10060503)); // normalized input * weight
        assert!(!RMSNORM_CODE.contains(&0x06060903)); // not matvec F32 reduction
    }

    /// Explicitly opt-in toolchain check; no GPU, model, driver or network access.
    #[test]
    #[ignore = "requires LLVM llvm-mc, llvm-objcopy and llvm-objdump; never accesses a GPU"]
    fn llvm_sources_match_embedded_words() {
        use std::path::{Path, PathBuf};
        use std::process::Command;
        use std::time::{SystemTime, UNIX_EPOCH};

        struct Temporary(PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        fn run(command: &mut Command) -> std::process::Output {
            let output = command
                .output()
                .expect("could not start LLVM tool; set BEHOLDD_LLVM_BIN");
            assert!(
                output.status.success(),
                "LLVM failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            output
        }
        let tool_dir = std::env::var_os("BEHOLDD_LLVM_BIN").map(PathBuf::from);
        let tool = |name: &str| {
            tool_dir
                .as_ref()
                .map_or_else(|| PathBuf::from(name), |dir| dir.join(name))
        };
        let source_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/hag/kernels");
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let scratch = Temporary(std::env::temp_dir().join(format!(
            "beholdd-hag-kernel-{}-{unique}",
            std::process::id()
        )));
        std::fs::create_dir(&scratch.0).unwrap();
        for (bytes, name) in [
            (bf16_bytes(), "bf16_matvec_gfx1151"),
            (q8_0_bytes(), "q8_0_matvec_gfx1151"),
            (swiglu_bytes(), "swiglu_gfx1151"),
            (scaled_add_bytes(), "scaled_add_gfx1151"),
            (rmsnorm_bytes(), "rmsnorm_gfx1151"),
            (rmsnorm_qwen35_bytes(), "rmsnorm_qwen35_gfx1151"),
            (multiply_bytes(), "multiply_gfx1151"),
            (rope_bytes(), "rope_gfx1151"),
            (attention_score_bytes(), "attention_score_gfx1151"),
            (attention_softmax_bytes(), "attention_softmax_gfx1151"),
            (attention_value_bytes(), "attention_value_gfx1151"),
            (q8_0_row_bytes(), "q8_0_row_gfx1151"),
        ] {
            let object = scratch.0.join(format!("{name}.o"));
            let binary = scratch.0.join(format!("{name}.bin"));
            run(Command::new(tool("llvm-mc"))
                .args([
                    "--triple=amdgcn-amd-amdhsa",
                    "--mcpu=gfx1151",
                    "--mattr=+wavefrontsize64,-wavefrontsize32",
                    "--filetype=obj",
                ])
                .arg(source_dir.join(format!("{name}.s")))
                .arg("-o")
                .arg(&object));
            run(Command::new(tool("llvm-objcopy"))
                .arg("--dump-section")
                .arg(format!(".text={}", binary.display()))
                .arg(&object));
            assert_eq!(
                std::fs::read(binary).unwrap(),
                bytes,
                "assembly drift for {name}"
            );
            let disassembly = run(Command::new(tool("llvm-objdump"))
                .args([
                    "--disassemble",
                    "--mcpu=gfx1151",
                    "--mattr=+wavefrontsize64,-wavefrontsize32",
                ])
                .arg(&object));
            let text = String::from_utf8(disassembly.stdout).unwrap();
            assert!(!text.contains("<unknown>"));
            assert!(text.contains("v_cmp_gt_u32_e32 vcc,"));
            assert!(text.contains("s_waitcnt_vscnt"));
            assert!(!text.contains("v_fma"));
            assert!(!text.contains("v_mac"));
            assert!(text.contains("v_lshl_add_u32 v0, s14, 6, v0"));
            if name == "rope_gfx1151" {
                assert!(text.contains("s_cmp_gt_u32 s14, 31"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x800"));
            } else if name == "attention_score_gfx1151" {
                assert!(text.contains("s_cmp_gt_u32 s14, 0xff"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x100"));
                assert!(text.contains("s_cmp_gt_u32 s10, 0x20000"));
            } else if name == "attention_softmax_gfx1151" {
                assert!(text.contains("s_cmp_gt_u32 s14, 0"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x4000"));
                assert!(text.contains("v_add_nc_u32_e32 v0, 0x200, v0"));
                assert!(text.contains("v_add_nc_u32_e32 v1, 0x100, v1"));
            } else if name == "attention_value_gfx1151" {
                assert!(text.contains("s_cmp_gt_u32 s14, 3"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x4000"));
                assert!(text.contains("s_cmp_gt_u32 s10, 0x400"));
            } else if name == "rmsnorm_qwen35_gfx1151" {
                assert!(text.contains("s_cmp_gt_u32 s14, 0x4f"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x1400"));
            } else {
                assert!(text.contains("s_cmp_gt_u32 s14, 63"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x1000"));
            }
            for token in text.split(|character: char| !character.is_ascii_alphanumeric()) {
                if let Some(register) = token
                    .strip_prefix('v')
                    .and_then(|name| name.parse::<u32>().ok())
                {
                    assert!(register < 8, "VGPR exceeds fixed ABI: {token}");
                }
                if let Some(register) = token
                    .strip_prefix('s')
                    .and_then(|name| name.parse::<u32>().ok())
                {
                    assert!(register < 20, "SGPR exceeds audited scratch: {token}");
                }
            }
            for (prefix, bound) in [("v[", 8), ("s[", 20)] {
                for range in text.split(prefix).skip(1) {
                    let registers = range.split_once(']').unwrap().0;
                    for register in registers.split(':') {
                        assert!(
                            register.parse::<u32>().unwrap() < bound,
                            "register range exceeds fixed ABI: {prefix}{registers}]"
                        );
                    }
                }
            }
            assert!(!text.contains("s[12:"));
            let relocations = run(Command::new(tool("llvm-objdump"))
                .arg("--reloc")
                .arg(&object));
            assert!(!String::from_utf8(relocations.stdout)
                .unwrap()
                .contains("R_AMDGPU_"));
            if matches!(
                name,
                "bf16_matvec_gfx1151"
                    | "q8_0_matvec_gfx1151"
                    | "attention_score_gfx1151"
                    | "attention_value_gfx1151"
            ) {
                assert!(text.contains("s_cmp_gt_u32 s13, 0x1000"));
            } else if name == "rope_gfx1151" {
                assert!(text.contains("s_cmp_gt_u32 s13, 4"));
                assert!(text.contains("s_sub_u32 s15, s13, 1"));
                assert!(text.contains("v_sub_f32_e32 v6, v6, v7"));
            } else {
                assert!(text.contains("s_cmp_eq_u32 s13, 1"));
            }
            if name == "swiglu_gfx1151" {
                assert!(text.contains("v_exp_f32_e32 v6, v5"));
                assert!(text.contains("v_rcp_f32_e32 v6, v6"));
                assert!(text.contains("v_cndmask_b32_e32 v2, v4, v2, vcc"));
            }
            if matches!(name, "rmsnorm_gfx1151" | "rmsnorm_qwen35_gfx1151") {
                for instruction in [
                    "v_cvt_f64_f32",
                    "v_mul_f64",
                    "v_add_f64",
                    "v_rcp_f64",
                    "v_rsq_f64",
                    "v_cvt_f32_f64",
                ] {
                    assert!(text.contains(instruction), "missing {instruction}");
                }
                assert!(!text.contains("v_add_f32"));
            }
            if name == "q8_0_row_gfx1151" {
                assert!(text.contains("s_and_b32 s15, s12, 31"));
                assert!(text.contains("buffer_load_u16 v4, v2, s[0:3], 0 offen"));
                assert!(text.contains("buffer_load_i8 v5, v3, s[0:3], 0 offen offset:2"));
                assert!(text.contains("v_cvt_f32_f16_e32 v4, v4"));
                assert!(text.contains("v_cvt_f32_i32_e32 v5, v5"));
                assert!(text.contains("v_mul_f32_e32 v5, v4, v5"));
                assert!(!text.contains("v_add_f32"));
                assert!(!text.contains("s[4:7]"));
            }
            if name.starts_with("attention_") {
                assert!(text.contains("v_mul_f64"));
                if name == "attention_softmax_gfx1151" {
                    assert!(text.contains("v_exp_f32_e32 v4, v4"));
                    assert!(text.contains("v_rcp_f32_e32 v5, v3"));
                    assert!(text.contains("v_cvt_f32_f64_e32 v4, v[4:5]"));
                } else {
                    assert!(text.contains("v_add_f64"));
                    assert!(!text.contains("v_add_f32"));
                    assert!(text.contains("v_cmp_gt_u32_e32 vcc, s15, v0"));
                }
            }
        }
    }
}
