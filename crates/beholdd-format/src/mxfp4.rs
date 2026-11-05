//! Reading a ggml `MXFP4` tensor into the layout the kernel's matvec expects.
//!
//! These are two different layouts for the same numbers, and confusing them is the single
//! most dangerous mistake this format admits, because the result is a model that still runs
//! and still writes fluent text.
//!
//! **On disk**, ggml stores 32 values in 17 bytes: one E8M0 scale byte, then 16 bytes of
//! packed codes in which the low nibble of byte `j` is element `j` and the high nibble is
//! element `j + 16`. Split in half, exactly as `Q4_0` is:
//!
//! ```text
//! qs[0] qs[1] ... qs[15]
//!  low   low       low    -> elements  0  1 ... 15
//!  high  high      high   -> elements 16 17 ... 31
//! ```
//!
//! **In memory**, [`beholdd_kernel::mxfp4`] wants the codes interleaved — low nibble is the
//! even element — with the scales lifted out into an array of their own, so a row's codes and
//! a row's scales are each contiguous. That is what lets the matvec walk a row without
//! stepping over a scale byte every 16 bytes.
//!
//! So a reader must **de-interleave**, and [`beholdd_core::mxfp4_unpack`] must not be pointed
//! at raw file bytes: it decodes the in-memory convention, and applying it to the on-disk one
//! permutes the weights within every block. The permutation is invisible in any summary of
//! the tensor — same values, same distribution, same norm — and shows up only as a model that
//! is subtly, confidently wrong.
//!
//! The scale needs no adjustment. ggml folds a factor of two into its code table and takes it
//! back out with a halved scale; [`beholdd_core::E2M1`] holds the true values and
//! [`beholdd_core::e8m0_scale`] the true `2^(b-127)`, and the products agree.

use anyhow::{bail, Result};

/// Values per ggml `MXFP4` block.
pub const BLOCK: usize = 32;
/// Bytes per block: one E8M0 scale, then `BLOCK / 2` packed codes.
pub const BLOCK_BYTES: usize = 1 + BLOCK / 2;

/// One MXFP4 matrix, split the way the kernel reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rows {
    /// `rows × n_in/2` bytes, two codes per byte, low nibble the even element.
    pub packed: Vec<u8>,
    /// `rows × n_in/BLOCK` E8M0 bytes, one per group.
    pub scales: Vec<u8>,
    /// Values sharing one scale. Always [`BLOCK`] for a ggml checkpoint.
    pub group: usize,
}

/// Convert a ggml `MXFP4` tensor's bytes into [`Rows`].
///
/// `n_in` is the fastest-varying dimension — GGUF's `dims[0]` — and must be a whole number of
/// blocks, which it is for every real checkpoint: a row that ended mid-block would put two
/// rows under one scale.
pub fn read(raw: &[u8], n_in: usize, rows: usize) -> Result<Rows> {
    if !n_in.is_multiple_of(BLOCK) {
        bail!("an MXFP4 row of {n_in} is not a whole number of {BLOCK}-value blocks");
    }
    let per_row = n_in / BLOCK;
    let expected = rows * per_row * BLOCK_BYTES;
    if raw.len() != expected {
        bail!(
            "an MXFP4 tensor of {rows}x{n_in} is {expected} bytes, got {}",
            raw.len()
        );
    }

    let mut packed = vec![0u8; rows * n_in / 2];
    let mut scales = vec![0u8; rows * per_row];
    for row in 0..rows {
        for group in 0..per_row {
            let block = &raw[(row * per_row + group) * BLOCK_BYTES..][..BLOCK_BYTES];
            scales[row * per_row + group] = block[0];
            let base = row * n_in + group * BLOCK;
            for (j, &byte) in block[1..].iter().enumerate() {
                // The two halves of the block, put back where they belong.
                place(&mut packed, row, n_in, base + j, byte & 0x0F);
                place(&mut packed, row, n_in, base + j + BLOCK / 2, byte >> 4);
            }
        }
    }
    Ok(Rows { packed, scales, group: BLOCK })
}

impl Rows {
    /// Expand to `f32`, the way a reference path reads them.
    ///
    /// The shipping path does not do this — [`beholdd_kernel::mxfp4::matvec`] consumes the
    /// packed bytes and never materialises the matrix, which is the whole point of a 4-bit
    /// format. This exists so a gate can hand the same weights to an f64 oracle.
    pub fn dequant(&self, n_in: usize) -> Vec<f32> {
        let per_row = n_in / self.group;
        let mut out = vec![0.0f32; self.packed.len() * 2];
        for (i, value) in out.iter_mut().enumerate() {
            let row = i / n_in;
            let within = i - row * n_in;
            let byte = self.packed[row * (n_in / 2) + within / 2];
            let code = if within.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 };
            // The scale multiplies the decoded value here because this is a definition, not
            // the fast path; the kernel scales a group's finished sum instead, for the reason
            // written down in `beholdd_kernel::mxfp4`.
            *value = beholdd_core::E2M1[code as usize]
                * beholdd_core::e8m0_scale(self.scales[row * per_row + within / self.group]);
        }
        out
    }
}

/// Write one 4-bit code at flat element index `at`, low nibble for an even element.
fn place(packed: &mut [u8], row: usize, n_in: usize, at: usize, code: u8) {
    let within = at - row * n_in;
    let byte = row * (n_in / 2) + within / 2;
    if within.is_multiple_of(2) {
        packed[byte] |= code & 0x0F;
    } else {
        packed[byte] |= (code & 0x0F) << 4;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use beholdd_core::{e8m0_scale, E2M1};

    /// Build one ggml block from 32 codes, in ggml's own split-half order.
    fn block(scale: u8, codes: &[u8; BLOCK]) -> Vec<u8> {
        let mut out = vec![scale];
        for j in 0..BLOCK / 2 {
            out.push((codes[j] & 0x0F) | ((codes[j + BLOCK / 2] & 0x0F) << 4));
        }
        out
    }

    /// Decode what the kernel's layout says element `i` of `row` is.
    fn value(rows: &Rows, row: usize, n_in: usize, i: usize) -> f32 {
        let byte = rows.packed[row * (n_in / 2) + i / 2];
        let code = if i.is_multiple_of(2) { byte & 0x0F } else { byte >> 4 };
        E2M1[code as usize] * e8m0_scale(rows.scales[row * (n_in / rows.group) + i / rows.group])
    }

    #[test]
    fn the_two_halves_of_a_block_land_where_ggml_put_them() {
        // Codes 0..32, so every element is distinguishable from every other.
        let codes: [u8; BLOCK] = std::array::from_fn(|i| (i % 16) as u8);
        let raw = block(127, &codes);
        let rows = read(&raw, BLOCK, 1).unwrap();

        for (i, &code) in codes.iter().enumerate() {
            assert_eq!(
                value(&rows, 0, BLOCK, i),
                E2M1[code as usize],
                "element {i} came back as a different code"
            );
        }
    }

    /// The mistake this module exists to prevent, stated as a test: reading the file bytes
    /// with the in-memory convention permutes a block rather than failing.
    #[test]
    fn the_interleaved_reading_of_the_same_bytes_is_a_permutation_not_an_error() {
        let codes: [u8; BLOCK] = std::array::from_fn(|i| (i % 16) as u8);
        let raw = block(127, &codes);
        let correct = read(&raw, BLOCK, 1).unwrap();

        let mut naive = vec![0.0f32; BLOCK];
        beholdd_core::mxfp4_unpack(&raw[1..], &mut naive);
        let right: Vec<f32> = (0..BLOCK).map(|i| value(&correct, 0, BLOCK, i)).collect();

        assert_ne!(naive, right, "if these agreed, the split-half reading would be pointless");
        let mut a = naive.clone();
        let mut b = right.clone();
        a.sort_by(f32::total_cmp);
        b.sort_by(f32::total_cmp);
        assert_eq!(a, b, "and it is a permutation, which is why nothing else catches it");
    }

    #[test]
    fn scales_stay_with_their_own_group() {
        let flat: [u8; BLOCK] = [2u8; BLOCK]; // code 2 is 1.0
        let raw: Vec<u8> = block(128, &flat).into_iter().chain(block(126, &flat)).collect();
        let rows = read(&raw, 2 * BLOCK, 1).unwrap();
        assert_eq!(value(&rows, 0, 2 * BLOCK, 0), 2.0, "first group is 2^1");
        assert_eq!(value(&rows, 0, 2 * BLOCK, BLOCK), 0.5, "second group is 2^-1");
    }

    #[test]
    fn rows_do_not_bleed_into_one_another() {
        let ones: [u8; BLOCK] = [2u8; BLOCK];
        let twos: [u8; BLOCK] = [4u8; BLOCK]; // code 4 is 2.0
        let raw: Vec<u8> = block(127, &ones).into_iter().chain(block(127, &twos)).collect();
        let rows = read(&raw, BLOCK, 2).unwrap();
        assert_eq!(value(&rows, 0, BLOCK, 5), 1.0);
        assert_eq!(value(&rows, 1, BLOCK, 5), 2.0);
    }

    #[test]
    fn dequant_reproduces_ggmls_own_arithmetic() {
        // ggml computes `kvalues_mxfp4[code] * E8M0_TO_FP32_HALF(e)`, where its table holds
        // twice the true E2M1 values. Ours holds the true ones against the unhalved scale,
        // and the products must agree element for element.
        let kvalues: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];
        let codes: [u8; BLOCK] = std::array::from_fn(|i| (i % 16) as u8);
        for scale in [117u8, 126, 127, 128, 137] {
            let rows = read(&block(scale, &codes), BLOCK, 1).unwrap();
            let ours = rows.dequant(BLOCK);
            for (i, &code) in codes.iter().enumerate() {
                let theirs = kvalues[code as usize] as f32 * (e8m0_scale(scale) * 0.5);
                assert_eq!(ours[i], theirs, "element {i} at scale {scale}");
            }
        }
    }

    #[test]
    fn a_row_that_ends_mid_block_is_refused() {
        let error = read(&[0u8; BLOCK_BYTES], 24, 1).expect_err("24 is not a whole block");
        assert!(format!("{error:#}").contains("whole number"), "{error:#}");
    }

    #[test]
    fn the_wrong_number_of_bytes_is_refused() {
        let error = read(&[0u8; 10], BLOCK, 1).expect_err("10 bytes is not one block");
        assert!(format!("{error:#}").contains("17 bytes"), "{error:#}");
    }
}
