//! Independent packed K-quant prototype contract and hardware-free arithmetic.
//!
//! The four adjacent ISA sources assemble to the exact embedded words below.
//! They are selected only by the explicit Qwen35 operator-probe profile. Assembly
//! verification establishes source/byte agreement, not GPU numerical or serving
//! qualification; those require separate supervised hardware gates.
//!
//! GGML's Q4_K/Q6_K layouts are specified by `ggml-quants.c` at llama.cpp commit
//! 4da6337767f973e2b4d0797e5b323d77d8565e4a. The scalar model below is derived from
//! that packing specification, independently of the ISA register schedules and
//! Beholdd's production quantization implementation. It specifies F32 decoding,
//! separate F32 multiplication, and increasing-column F32 accumulation, no FMA.

pub(crate) const MAX_KQUANT_ROWS: usize = 4096;
pub(crate) const MAX_KQUANT_COLS: usize = 17408;
pub(crate) const MAX_KQUANT_ROW_COLS: usize = 5120;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum KQuantKind {
    Q4K,
    Q6K,
}

impl KQuantKind {
    pub(crate) const fn block_bytes(self) -> usize {
        match self {
            Self::Q4K => 144,
            Self::Q6K => 210,
        }
    }

    /// Verified assembled matvec words plus 256 bytes of instruction-fetch padding.
    pub(crate) fn matvec_bytes(self) -> Vec<u8> {
        super::padded_shader(match self {
            Self::Q4K => Q4_K_MATVEC_CODE,
            Self::Q6K => Q6_K_MATVEC_CODE,
        })
    }

    /// Verified assembled selected-row decode plus the same fetch padding.
    pub(crate) fn row_decode_bytes(self) -> Vec<u8> {
        super::padded_shader(match self {
            Self::Q4K => Q4_K_ROW_CODE,
            Self::Q6K => Q6_K_ROW_CODE,
        })
    }

    /// Exact packed matrix bytes, with whole 256-coordinate row blocks.
    /// Row and column bounds are checked before multiplication or any allocation.
    pub(crate) fn packed_bytes(self, rows: usize, cols: usize) -> Result<usize, &'static str> {
        if !(1..=MAX_KQUANT_ROWS).contains(&rows)
            || !(256..=MAX_KQUANT_COLS).contains(&cols)
            || !cols.is_multiple_of(256)
        {
            return Err("K-quant matvec needs 1..=4096 rows and 256..=17408 columns in whole 256-element blocks");
        }
        Ok(rows * (cols / 256) * self.block_bytes())
    }

    /// Exact selected-row extent for GPU embedding decode. The matvec's wider
    /// column admission does not authorize more than 5120 row-decode coordinates.
    pub(crate) fn row_bytes(self, cols: usize) -> Result<usize, &'static str> {
        if !(256..=MAX_KQUANT_ROW_COLS).contains(&cols) || !cols.is_multiple_of(256) {
            return Err("K-quant row decode needs 256..=5120 columns in whole 256-element blocks");
        }
        Ok((cols / 256) * self.block_bytes())
    }
}

// LLVM gfx1151 wave64: 150 instruction words + 64 fetch-padding words.
// Padded .text SHA-256: efa1511c5fe4563e6df9077dbc41b8e34d8d7b53a1ef2421242f876992c8167e
const Q4_K_MATVEC_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20091, 0xbf06800c, 0xbfa2008f, 0xbf08ff0c, 0x00004400, 0xbfa2008c,
    0x8b0fff0c, 0x000000ff, 0xbf06800f, 0xbfa10088, 0xbf06800d, 0xbfa20086, 0xbf08ff0d, 0x00001000,
    0xbfa20083, 0x8519880c, 0x9619ff19, 0x00000090, 0x960f0d19, 0xbf060f02, 0xbfa1007d, 0x8418820c,
    0xbf061806, 0xbfa1007a, 0x840f820d, 0xbf060f0a, 0xbfa10077, 0x360000ff, 0x000003ff, 0xbf800007,
    0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000d, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa5006c,
    0xd72c0001, 0x00020019, 0xbf800007, 0x30040082, 0xbf800007, 0x7e060280, 0xbe8f0080, 0xbe910080,
    0xbe920080, 0xbf0a8412, 0xbfa1000d, 0x8015120f, 0x80158415, 0x80168415, 0xe0400000, 0x15400501,
    0xe0400000, 0x16400401, 0xbf8903f7, 0x360a0abf, 0xbf800007, 0x360808bf, 0xbf800007, 0xbfa0001a,
    0x8015120f, 0x80168815, 0xe0400000, 0x15400501, 0xe0400000, 0x16400401, 0x80168415, 0xe0400000,
    0x16400601, 0xbf8903f7, 0x360e088f, 0xbf800007, 0x320a0a82, 0xbf800007, 0x360a0ab0, 0xbf800007,
    0x380a0b07, 0xbf800007, 0x32080884, 0xbf800007, 0x320c0c82, 0xbf800007, 0x360c0cb0, 0xbf800007,
    0x38080906, 0xbf800007, 0xe0480000, 0x0f400601, 0xe0480002, 0x0f400701, 0xbf8903f7, 0x7e0a0d05,
    0xbf800007, 0x7e080d04, 0xbf800007, 0x7e0c1706, 0xbf800007, 0x7e0e1707, 0xbf800007, 0x100c0b06,
    0xbf800007, 0x100e0907, 0xbf800007, 0x85148112, 0x84148514, 0x80140f14, 0x80149014, 0x8b178112,
    0x84178217, 0xbe930080, 0xe0400000, 0x14400401, 0xe0500000, 0x11010500, 0xbf8903f7, 0x32080817,
    0xbf800007, 0x3608088f, 0xbf800007, 0x7e080d04, 0xbf800007, 0x10080906, 0xbf800007, 0x08080f04,
    0xbf800007, 0x10080b04, 0xbf800007, 0x06060903, 0xbf800007, 0x80118411, 0x80148114, 0x80138113,
    0xbf0aa013, 0xbfa2ffe8, 0x80128112, 0xbf0a8812, 0xbfa2ffa4, 0x800fff0f, 0x00000090, 0xbf0a1811,
    0xbfa2ff9f, 0xe0680000, 0x80420302, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// LLVM gfx1151 wave64: 133 instruction words + 64 fetch-padding words.
// Padded .text SHA-256: dae09ee4ae0f6a5e1eb0531f2ba8a0625697f099703c58267736b027508f7739
const Q6_K_MATVEC_CODE: &[u32] = &[
    0xb0802006, 0xbf08bf0e, 0xbfa20080, 0xbf06800c, 0xbfa2007e, 0xbf08ff0c, 0x00004400, 0xbfa2007b,
    0x8b0fff0c, 0x000000ff, 0xbf06800f, 0xbfa10077, 0xbf06800d, 0xbfa20075, 0xbf08ff0d, 0x00001000,
    0xbfa20072, 0x851b880c, 0x961bff1b, 0x000000d2, 0x960f0d1b, 0xbf060f02, 0xbfa1006c, 0x841c820c,
    0xbf061c06, 0xbfa10069, 0x840f820d, 0xbf060f0a, 0xbfa10066, 0x360000ff, 0x000003ff, 0xbf800007,
    0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000d, 0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa5005b,
    0xd72c0001, 0x0002001b, 0xbf800007, 0x30040082, 0xbf800007, 0x7e060280, 0xbe8f0080, 0xbe900080,
    0xe04800d0, 0x0f400601, 0xbf8903f7, 0x7e0c1706, 0xbf800007, 0xbe910080, 0x8013110f, 0x8013ff13,
    0x000000c0, 0xe0440000, 0x13400701, 0x85148311, 0x8b158711, 0x85158115, 0x8b168111, 0x84168416,
    0x8b178115, 0x84178517, 0x84188614, 0x80171817, 0x80171617, 0x80170f17, 0x84188514, 0x80181618,
    0x80180f18, 0x8018ff18, 0x00000080, 0x85198115, 0x84198219, 0x841a8115, 0xbf8903f7, 0x7e0e0b07,
    0xbf800007, 0x100e0f06, 0xbf800007, 0xbe920080, 0xe0400000, 0x17400401, 0xe0400000, 0x18400501,
    0xbf8903f7, 0x32080819, 0xbf800007, 0x3608088f, 0xbf800007, 0x320a0a1a, 0xbf800007, 0x360a0a83,
    0xbf800007, 0x300a0a84, 0xbf800007, 0x38080905, 0xbf800007, 0x4a0808ff, 0xffffffe0, 0xbf800007,
    0x7e080b04, 0xbf800007, 0x10080907, 0xbf800007, 0xe0500000, 0x10010500, 0xbf8903f7, 0x10080b04,
    0xbf800007, 0x06060903, 0xbf800007, 0x80108410, 0x80178117, 0x80188118, 0x80128112, 0xbf0a9012,
    0xbfa2ffdb, 0x80118111, 0xbf0a9011, 0xbfa2ffba, 0x800fff0f, 0x000000d2, 0xbf0a1c10, 0xbfa2ffb0,
    0xe0680000, 0x80420302, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// LLVM gfx1151 wave64: 133 instruction words + 64 fetch-padding words.
// Padded .text SHA-256: 71f5e93ae4a51af200f78f819ece52c94edeb1f33e0e603297eaf711357b1979
const Q4_K_ROW_CODE: &[u32] = &[
    0xb0802006, 0xbf08ff0e, 0x0000004f, 0xbfa2007f, 0xbf06800c, 0xbfa2007d, 0xbf08ff0c, 0x00001400,
    0xbfa2007a, 0x8b0fff0c, 0x000000ff, 0xbf06800f, 0xbfa10076, 0xbf06810d, 0xbfa10074, 0x850f880c,
    0x960fff0f, 0x00000090, 0xbf060f02, 0xbfa1006f, 0xbf068406, 0xbfa1006d, 0x840f820c, 0xbf060f0a,
    0xbfa1006a, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000c,
    0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa5005f, 0x30020082, 0xbf800007, 0x32040088, 0xbf800007,
    0xd72c0002, 0x000204ff, 0x00000090, 0xbf800007, 0x3606009f, 0xbf800007, 0x32000085, 0xbf800007,
    0x36000087, 0xbf800007, 0x32080081, 0xbf800007, 0x30080885, 0xbf800007, 0x4a060704, 0xbf800007,
    0x4a060702, 0xbf800007, 0x4a060690, 0xbf800007, 0x4a080500, 0xbf800007, 0xe0400000, 0x80400504,
    0xe0400008, 0x80400604, 0xe0400004, 0x80400404, 0xbf8903f7, 0x320a0a82, 0xbf800007, 0x360a0ab0,
    0xbf800007, 0x360e0c8f, 0xbf800007, 0x380a0b07, 0xbf800007, 0x360e08bf, 0xbf800007, 0x7c980084,
    0xbf800007, 0x020a0f05, 0xbf800007, 0x32080882, 0xbf800007, 0x360808b0, 0xbf800007, 0x320e0c84,
    0xbf800007, 0x38080907, 0xbf800007, 0x360c0cbf, 0xbf800007, 0x020e0d04, 0xbf800007, 0xe0480000,
    0x80400402, 0xe0480002, 0x80400602, 0xe0400000, 0x80400303, 0xbf8903f7, 0x36000081, 0xbf800007,
    0x30000082, 0xbf800007, 0x32060700, 0xbf800007, 0x3606068f, 0xbf800007, 0x7e060d03, 0xbf800007,
    0x7e0a0d05, 0xbf800007, 0x7e0e0d07, 0xbf800007, 0x7e081704, 0xbf800007, 0x7e0c1706, 0xbf800007,
    0x10080b04, 0xbf800007, 0x100c0f06, 0xbf800007, 0x10060704, 0xbf800007, 0x08060d03, 0xbf800007,
    0xe0680000, 0x80420301, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

// LLVM gfx1151 wave64: 135 instruction words + 64 fetch-padding words.
// Padded .text SHA-256: efc6fbd6b79a88dae1733b583bd11dce7e7cb631d5c8deb1e4e89bf7d4d3e6e1
const Q6_K_ROW_CODE: &[u32] = &[
    0xb0802006, 0xbf08ff0e, 0x0000004f, 0xbfa20081, 0xbf06800c, 0xbfa2007f, 0xbf08ff0c, 0x00001400,
    0xbfa2007c, 0x8b0fff0c, 0x000000ff, 0xbf06800f, 0xbfa10078, 0xbf06810d, 0xbfa10076, 0x850f880c,
    0x960fff0f, 0x000000d2, 0xbf060f02, 0xbfa10071, 0xbf068406, 0xbfa1006f, 0x840f820c, 0xbf060f0a,
    0xbfa1006c, 0x360000ff, 0x000003ff, 0xbf800007, 0xd6460000, 0x04010c0e, 0xbf800007, 0x7c98000c,
    0xbf800007, 0x8bfe6a7e, 0xbf800007, 0xbfa50061, 0x30020082, 0xbf800007, 0x32040088, 0xbf800007,
    0xd72c0002, 0x000204ff, 0x000000d2, 0xbf800007, 0x32060084, 0xbf800007, 0x3606068f, 0xbf800007,
    0x4a060702, 0xbf800007, 0xe04400c0, 0x80400703, 0xe04800d0, 0x80400602, 0xbf8903f7, 0x3606009f,
    0xbf800007, 0x32080085, 0xbf800007, 0x36080881, 0xbf800007, 0x30080885, 0xbf800007, 0x4a060704,
    0xbf800007, 0x32080087, 0xbf800007, 0x36080881, 0xbf800007, 0x30080886, 0xbf800007, 0x4a060704,
    0xbf800007, 0x4a060702, 0xbf800007, 0xe0400000, 0x80400503, 0xbf8903f7, 0x3606009f, 0xbf800007,
    0x32080087, 0xbf800007, 0x36080881, 0xbf800007, 0x30080885, 0xbf800007, 0x4a060704, 0xbf800007,
    0x4a060702, 0xbf800007, 0xe0400080, 0x80400403, 0xbf8903f7, 0x32060086, 0xbf800007, 0x36060681,
    0xbf800007, 0x30060682, 0xbf800007, 0x320a0b03, 0xbf800007, 0x360a0a8f, 0xbf800007, 0x32060085,
    0xbf800007, 0x36060683, 0xbf800007, 0x30060681, 0xbf800007, 0x32080903, 0xbf800007, 0x36080883,
    0xbf800007, 0x30080884, 0xbf800007, 0x380a0b04, 0xbf800007, 0x4a0a0aff, 0xffffffe0, 0xbf800007,
    0x7e0a0b05, 0xbf800007, 0x7e0e0b07, 0xbf800007, 0x7e0c1706, 0xbf800007, 0x100e0f06, 0xbf800007,
    0x100a0b07, 0xbf800007, 0xe0680000, 0x80420501, 0xbc7c0000, 0xbfb60003, 0xbfb00000,
];

#[cfg(test)]
mod tests {
    use super::*;

    const Q4_MATVEC: &str = include_str!("q4_k_matvec_gfx1151.s");
    const Q6_MATVEC: &str = include_str!("q6_k_matvec_gfx1151.s");
    const Q4_ROW: &str = include_str!("q4_k_row_gfx1151.s");
    const Q6_ROW: &str = include_str!("q6_k_row_gfx1151.s");

    // Exact half widening, deliberately independent of crate dequantizers. F32
    // can represent every half exactly, including subnormals and signed zero.
    fn half(bytes: &[u8]) -> f32 {
        let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
        let sign = u32::from(bits & 0x8000) << 16;
        let exponent = u32::from((bits >> 10) & 31);
        let fraction = u32::from(bits & 1023);
        let raw = match exponent {
            0 if fraction == 0 => sign,
            0 => {
                let leading = 31 - fraction.leading_zeros();
                let exponent = leading + 103;
                sign | (exponent << 23) | ((fraction - (1 << leading)) << (23 - leading))
            }
            31 => sign | 0x7f80_0000 | (fraction << 13),
            _ => sign | ((exponent + 112) << 23) | (fraction << 13),
        };
        f32::from_bits(raw)
    }

    fn decode(kind: KQuantKind, raw: &[u8]) -> Result<Vec<f32>, &'static str> {
        if !raw.len().is_multiple_of(kind.block_bytes()) {
            return Err("short quantization block");
        }
        let mut result = Vec::with_capacity(raw.len() / kind.block_bytes() * 256);
        for block in raw.chunks_exact(kind.block_bytes()) {
            for index in 0..256 {
                let value = match kind {
                    KQuantKind::Q4K => {
                        let sub = index / 32;
                        let packed = &block[4..16];
                        let (scale, minimum) = if sub < 4 {
                            (packed[sub] & 63, packed[sub + 4] & 63)
                        } else {
                            (
                                (packed[sub + 4] & 15) | ((packed[sub - 4] >> 6) << 4),
                                (packed[sub + 4] >> 4) | ((packed[sub] >> 6) << 4),
                            )
                        };
                        let code =
                            (block[16 + (index / 64) * 32 + index % 32] >> (4 * (sub % 2))) & 15;
                        let scale = half(&block[0..2]) * f32::from(scale);
                        let minimum = half(&block[2..4]) * f32::from(minimum);
                        scale * f32::from(code) - minimum
                    }
                    KQuantKind::Q6K => {
                        let half_index = index / 128;
                        let within = index % 128;
                        let group = within / 32;
                        let lane = within % 32;
                        let low = (block[half_index * 64 + (group % 2) * 32 + lane]
                            >> (4 * (group / 2)))
                            & 15;
                        let high = (block[128 + half_index * 32 + lane] >> (2 * group)) & 3;
                        let code = i32::from(low | (high << 4)) - 32;
                        let scale = f32::from(block[192 + index / 16] as i8);
                        (half(&block[208..210]) * scale) * code as f32
                    }
                };
                result.push(value);
            }
        }
        Ok(result)
    }

    fn serial_matvec(
        kind: KQuantKind,
        rows: usize,
        cols: usize,
        raw: &[u8],
        input: &[f32],
    ) -> Result<Vec<f32>, &'static str> {
        if raw.len() != kind.packed_bytes(rows, cols)? || input.len() != cols {
            return Err("matrix/input extent mismatch");
        }
        let decoded = decode(kind, raw)?;
        Ok(decoded
            .chunks_exact(cols)
            .map(|row| {
                let mut sum = 0.0_f32;
                for (&weight, &x) in row.iter().zip(input) {
                    let product = weight * x;
                    sum += product;
                }
                sum
            })
            .collect())
    }

    fn q4_block(d: u16, dmin: u16, scales: [u8; 8], minima: [u8; 8], codes: &[u8; 256]) -> Vec<u8> {
        let mut raw = vec![0; 144];
        raw[..2].copy_from_slice(&d.to_le_bytes());
        raw[2..4].copy_from_slice(&dmin.to_le_bytes());
        // Place each semantic six-bit scale and min into its independent field.
        for (i, (&scale, &minimum)) in scales.iter().zip(&minima).enumerate() {
            assert!(scale < 64 && minimum < 64);
            if i < 4 {
                raw[4 + i] |= scale;
                raw[8 + i] |= minimum;
            } else {
                raw[i] |= (scale >> 4) << 6;
                raw[4 + i] |= (minimum >> 4) << 6;
                raw[8 + i] = (scale & 15) | ((minimum & 15) << 4);
            }
        }
        for (i, &code) in codes.iter().enumerate() {
            assert!(code < 16);
            raw[16 + (i / 64) * 32 + i % 32] |= code << (4 * ((i / 32) % 2));
        }
        raw
    }

    fn q6_block(d: u16, scales: [i8; 16], codes: &[u8; 256]) -> Vec<u8> {
        let mut raw = vec![0; 210];
        for (index, &code) in codes.iter().enumerate() {
            assert!(code < 64);
            let group = (index % 128) / 32;
            let lane = index % 32;
            raw[(index / 128) * 64 + (group % 2) * 32 + lane] |= (code & 15) << (4 * (group / 2));
            raw[128 + (index / 128) * 32 + lane] |= (code >> 4) << (2 * group);
        }
        for (dst, value) in raw[192..208].iter_mut().zip(scales) {
            *dst = value as u8;
        }
        raw[208..210].copy_from_slice(&d.to_le_bytes());
        raw
    }

    #[test]
    fn bounds_precede_overflow_and_distinguish_matrix_from_row_decode() {
        for kind in [KQuantKind::Q4K, KQuantKind::Q6K] {
            for (rows, cols) in [
                (0, 256),
                (1, 0),
                (4097, 256),
                (1, 17409),
                (usize::MAX, 256),
                (1, usize::MAX),
                (1, 255),
                (1, 257),
            ] {
                assert!(kind.packed_bytes(rows, cols).is_err());
            }
            assert_eq!(
                kind.packed_bytes(4096, 17408).unwrap(),
                4096 * 68 * kind.block_bytes()
            );
            assert_eq!(kind.row_bytes(5120).unwrap(), 20 * kind.block_bytes());
            for cols in [0, 255, 257, 5121, 5376, 17408, usize::MAX] {
                assert!(kind.row_bytes(cols).is_err());
            }
            assert_eq!(kind.packed_bytes(1, 5376).unwrap(), 21 * kind.block_bytes());
        }
    }

    #[test]
    fn half_widening_preserves_subnormal_finite_extremes_and_signed_zero() {
        for (bits, expected) in [
            (0_u16, 0_u32),
            (0x8000, 0x8000_0000),
            (1, 0x3380_0000),
            (0x8001, 0xb380_0000),
            (0x03ff, 0x387f_c000),
            (0x0400, 0x3880_0000),
            (0x3c00, 0x3f80_0000),
            (0x7bff, 0x477f_e000),
        ] {
            assert_eq!(half(&bits.to_le_bytes()).to_bits(), expected);
        }
    }

    #[test]
    fn q4_six_bit_scales_minima_and_nibble_order_have_independent_values() {
        let scales = [1, 17, 32, 63, 61, 33, 18, 7];
        let minima = [62, 31, 16, 4, 3, 19, 34, 63];
        let codes = std::array::from_fn(|i| ((i * 7 + i / 32) % 16) as u8);
        let raw = q4_block(0x3000, 0x2c00, scales, minima, &codes); // d=1/8,dmin=1/16
        let actual = decode(KQuantKind::Q4K, &raw).unwrap();
        for i in 0..256 {
            let expected = (0.125 * f32::from(scales[i / 32])) * f32::from(codes[i])
                - 0.0625 * f32::from(minima[i / 32]);
            assert_eq!(actual[i].to_bits(), expected.to_bits(), "coordinate {i}");
        }
        let raw = q4_block(1, 1, [1; 8], [2; 8], &[3; 256]);
        assert!(decode(KQuantKind::Q4K, &raw)
            .unwrap()
            .iter()
            .all(|v| v.to_bits() == 0x3380_0000));
    }

    #[test]
    fn q6_signed_scales_all_high_pairs_and_two_byte_block_starts() {
        let scales = [
            -128, -97, -63, -32, -1, 0, 1, 17, 31, 63, 95, 127, -2, 3, -4, 5,
        ];
        let codes = std::array::from_fn(|i| ((i * 13 + i / 16) % 64) as u8);
        let first = q6_block(0x3000, scales, &codes);
        let second = q6_block(1, scales, &codes);
        let raw = [first, second].concat();
        let actual = decode(KQuantKind::Q6K, &raw).unwrap();
        for block in 0..2 {
            let d = if block == 0 {
                0.125
            } else {
                f32::from_bits(0x3380_0000)
            };
            for i in 0..256 {
                let expected = (d * f32::from(scales[i / 16])) * (i32::from(codes[i]) - 32) as f32;
                assert_eq!(actual[block * 256 + i].to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn accumulation_is_increasing_f32_with_distinct_rows_and_blocks() {
        for kind in [KQuantKind::Q4K, KQuantKind::Q6K] {
            let make = |two: bool| match kind {
                KQuantKind::Q4K => {
                    q4_block(0x3c00, 0, [1; 8], [0; 8], &[if two { 2 } else { 1 }; 256])
                }
                KQuantKind::Q6K => q6_block(0x3c00, [1; 16], &[if two { 34 } else { 33 }; 256]),
            };
            let raw = [make(false), make(false), make(true), make(true)].concat();
            let mut input = vec![0.0; 512];
            input[0] = 16777216.0;
            input[1] = 1.0;
            input[2] = -16777216.0;
            input[255] = 0.25;
            input[256] = 0.5;
            let output = serial_matvec(kind, 2, 512, &raw, &input).unwrap();
            // An F64 sum or a different reduction order would retain the lost1.
            assert_eq!(output, [0.75, 1.5]);
            assert!(serial_matvec(kind, 2, 512, &raw[..raw.len() - 1], &input).is_err());
            assert!(serial_matvec(kind, 2, 512, &raw, &input[..511]).is_err());
            assert!(decode(kind, &raw[..raw.len() - 1]).is_err());
        }
    }

    #[test]
    fn sources_keep_eight_vgprs_exact_extents_and_mask_before_loads() {
        for (source, rows, maximum) in [
            (Q4_MATVEC, false, 17408),
            (Q6_MATVEC, false, 17408),
            (Q4_ROW, true, 5120),
            (Q6_ROW, true, 5120),
        ] {
            let instructions: Vec<_> = source
                .lines()
                .map(str::trim)
                .filter(|line| !line.starts_with("//"))
                .collect();
            let code = instructions.join("\n");
            assert!(code.contains("s_version 0x2006"));
            assert!(code.contains(&format!("s_cmp_gt_u32 s12, {maximum}")));
            assert!(code.contains("s_and_b32 s15, s12, 255"));
            for (register, expected) in [("s2", "s15"), ("s10", "s15")] {
                assert!(code.contains(&format!("s_cmp_eq_u32 {register}, {expected}")));
            }
            assert!(code.contains(if rows {
                "s_cmp_eq_u32 s6, 4"
            } else if source == Q4_MATVEC {
                "s_cmp_eq_u32 s6, s24"
            } else {
                "s_cmp_eq_u32 s6, s28"
            }));
            assert!(code.contains(if rows {
                "s_cmp_gt_u32 s14, 79"
            } else {
                "s_cmp_gt_u32 s14, 63"
            }));
            let first_load = code.find("buffer_load_").unwrap();
            assert!(code.find("s_and_b64 exec, exec, vcc").unwrap() < first_load);
            assert!(code.find("s_cmp_eq_u32 s10, s15").unwrap() < first_load);
            assert!(!code.contains("v_fma") && !code.contains("v_fmac") && !code.contains("v_mad"));
            assert!(code.contains("s_waitcnt_vscnt null, 0"));
            assert!(code.contains(".rept 64\ns_code_end\n.endr"));
            for token in code.split(|c: char| !c.is_ascii_alphanumeric()) {
                if let Some(index) = token
                    .strip_prefix('v')
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    assert!(index < 8, "unexpected VGPR {token}");
                }
            }
        }
    }

    #[test]
    fn embedded_programs_have_verified_lengths_and_fetch_padding() {
        for (bytes, expected) in [
            (KQuantKind::Q4K.matvec_bytes(), 856),
            (KQuantKind::Q6K.matvec_bytes(), 788),
            (KQuantKind::Q4K.row_decode_bytes(), 788),
            (KQuantKind::Q6K.row_decode_bytes(), 796),
        ] {
            assert_eq!(bytes.len(), expected);
            assert!(bytes.len() <= 4096 && bytes.len().is_multiple_of(4));
            let words: Vec<_> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(words[0], 0xb080_2006); // GFX11 wave64 s_version.
            assert_eq!(words[words.len() - 65], 0xbfb0_0000); // s_endpgm.
            assert!(words[words.len() - 64..]
                .iter()
                .all(|&word| word == 0xbf9f_0000));
        }
    }

    /// Opt-in reproducibility and ABI inspection; no driver, checkpoint, or GPU.
    #[test]
    #[ignore = "requires LLVM llvm-mc, llvm-objcopy and llvm-objdump; never accesses a GPU"]
    fn llvm_kquant_sources_match_embedded_words() {
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
                .expect("could not start LLVM; set BEHOLDD_LLVM_BIN");
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
            "beholdd-hag-kquant-{}-{unique}",
            std::process::id()
        )));
        std::fs::create_dir(&scratch.0).unwrap();
        for (bytes, name, row, sgpr_bound) in [
            (
                KQuantKind::Q4K.matvec_bytes(),
                "q4_k_matvec_gfx1151",
                false,
                26,
            ),
            (
                KQuantKind::Q6K.matvec_bytes(),
                "q6_k_matvec_gfx1151",
                false,
                29,
            ),
            (
                KQuantKind::Q4K.row_decode_bytes(),
                "q4_k_row_gfx1151",
                true,
                16,
            ),
            (
                KQuantKind::Q6K.row_decode_bytes(),
                "q6_k_row_gfx1151",
                true,
                16,
            ),
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
                std::fs::read(&binary).unwrap(),
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
            assert!(!text.contains("<unknown>"), "unknown instruction in {name}");
            assert!(text.contains("s_version UC_VERSION_GFX11|UC_VERSION_W64_BIT"));
            assert!(text.contains("v_lshl_add_u32 v0, s14, 6, v0"));
            assert!(text.contains("v_cmp_gt_u32_e32 vcc,"));
            assert!(text.contains("s_and_b32 s15, s12, 0xff"));
            assert!(text.contains("s_waitcnt_vscnt"));
            assert!(text.contains("v_cvt_f32_f16"));
            assert!(text.contains("v_mul_f32"));
            for fused in ["v_fma", "v_mac", "v_mad"] {
                assert!(
                    !text.contains(fused),
                    "unexpected fused arithmetic in {name}"
                );
            }
            let first_load = text.find("buffer_load_").unwrap();
            for check in [
                "s_cmp_eq_u32 s2, s15",
                "s_cmp_eq_u32 s10, s15",
                "s_and_b64 exec, exec, vcc",
            ] {
                assert!(
                    text.find(check).unwrap() < first_load,
                    "late/missing guard in {name}: {check}"
                );
            }
            if row {
                assert!(text.contains("s_cmp_gt_u32 s14, 0x4f"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x1400"));
                assert!(text.contains("s_cmp_eq_u32 s13, 1"));
                assert!(text.find("s_cmp_eq_u32 s6, 4").unwrap() < first_load);
                assert!(
                    !text.contains("s[4:7]"),
                    "row decoder must not read dummy descriptor"
                );
                assert!(
                    !text.contains("v_add_f32"),
                    "row decoder must not accumulate"
                );
            } else {
                assert!(text.contains("s_cmp_gt_u32 s14, 63"));
                assert!(text.contains("s_cmp_gt_u32 s12, 0x4400"));
                assert!(text.contains("s_cmp_gt_u32 s13, 0x1000"));
                let check = if name.starts_with("q4") {
                    "s_cmp_eq_u32 s6, s24"
                } else {
                    "s_cmp_eq_u32 s6, s28"
                };
                assert!(text.find(check).unwrap() < first_load);
                assert!(text.contains("v_add_f32_e32 v3, v3, v4"));
            }
            if name.starts_with("q4") {
                assert!(
                    text.contains("v_sub_f32"),
                    "Q4_K must subtract decoded minimum"
                );
            } else {
                assert!(
                    text.contains("buffer_load_i8 v7,"),
                    "Q6_K scales must be signed bytes"
                );
                assert!(text.contains("v_cvt_f32_i32"));
                assert!(
                    text.contains("offset:208"),
                    "Q6_K half scale must use its packed byte offset"
                );
            }
            for token in text.split(|c: char| !c.is_ascii_alphanumeric()) {
                for (prefix, bound) in [('v', 8), ('s', sgpr_bound)] {
                    if let Some(index) = token
                        .strip_prefix(prefix)
                        .and_then(|s| s.parse::<u32>().ok())
                    {
                        assert!(
                            index < bound,
                            "register exceeds audited ABI in {name}: {token}"
                        );
                    }
                }
            }
            for (prefix, bound) in [("v[", 8), ("s[", sgpr_bound)] {
                for range in text.split(prefix).skip(1) {
                    let registers = range.split_once(']').unwrap().0;
                    for register in registers.split(':') {
                        assert!(
                            register.parse::<u32>().unwrap() < bound,
                            "register range exceeds ABI in {name}: {prefix}{registers}]"
                        );
                    }
                }
            }
            for line in text
                .lines()
                .filter(|line| line.trim_start().starts_with("buffer_store_"))
            {
                assert!(
                    line.contains("s[8:11]"),
                    "store outside output descriptor in {name}"
                );
            }
            let relocations = run(Command::new(tool("llvm-objdump"))
                .arg("--reloc")
                .arg(&object));
            assert!(
                !String::from_utf8(relocations.stdout)
                    .unwrap()
                    .contains("R_AMDGPU_"),
                "unexpected relocation in {name}"
            );
        }
    }
}
