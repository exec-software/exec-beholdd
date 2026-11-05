//! Small packed-weight oracles for an explicitly selected native linear proof.
//!
//! Creating or checking these fixtures never opens a device or submits GPU work.
//! Encoded storage has its exact logical length, without shader tail padding.

use beholdd_format::GgmlType;

use crate::{cpu::Cpu, Device, DeviceError, Weight};

const MAX_ROWS: usize = 64;
const MAX_COLS: usize = 256;
const BF16: GgmlType = GgmlType(30);
const Q8_0: GgmlType = GgmlType(8);
const Q8_ELEMENTS: usize = 32;
const Q8_BYTES: usize = 34;
const BF16_BITS: [u16; 11] = [
    0x0000, 0x3f80, 0xbf80, 0x3f00, 0xbe80, 0x4000, 0xc040, 0x3eab, 0xbfa3, 0x3d8f, 0xc10d,
];
// Exact f16 encodings: 0.5, -0.25, 0, 0.75, 0.125, -2. All are finite.
const Q8_SCALE_BITS: [u16; 6] = [0x3800, 0xb400, 0x0000, 0x3a00, 0x3000, 0xc000];

/// One bounded matrix-vector product with exact little-endian checkpoint bytes.
#[derive(Clone, Debug)]
pub(crate) struct LinearFixture {
    pub(crate) name: &'static str,
    pub(crate) kind: GgmlType,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) weight_bytes: Vec<u8>,
    pub(crate) input: Vec<f32>,
}

/// Successful serial-native parity plus the separately bounded CPU comparison.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ParityReport {
    pub(crate) checked_rows: usize,
    pub(crate) max_cpu_absolute_error: f64,
    pub(crate) max_cpu_allowed_error: f64,
}

/// Fixtures cover word/block/lane boundaries, odd rows, signed extremes and scales.
/// Q8_0 columns must be whole blocks; an incomplete 32-element block is invalid,
/// not a zero-padded numerical tail. Odd Q8_0 block counts still have byte tails.
pub(crate) fn linear_fixtures() -> Vec<LinearFixture> {
    let mut fixtures = Vec::new();
    for (name, rows, cols) in [
        ("bf16-1x1", 1, 1),
        ("bf16-3x2", 3, 2),
        ("bf16-5x31", 5, 31),
        ("bf16-7x32", 7, 32),
        ("bf16-9x33", 9, 33),
        ("bf16-11x63", 11, 63),
        ("bf16-13x64", 13, 64),
        ("bf16-15x65", 15, 65),
        ("bf16-31x255", 31, 255),
        ("bf16-64x256", 64, 256),
    ] {
        let mut weight_bytes = Vec::with_capacity(rows * cols * 2);
        for row in 0..rows {
            for col in 0..cols {
                let bits = BF16_BITS[(row * 17 + col * 7 + 1) % BF16_BITS.len()];
                weight_bytes.extend_from_slice(&bits.to_le_bytes());
            }
        }
        fixtures.push(LinearFixture {
            name,
            kind: BF16,
            rows,
            cols,
            weight_bytes,
            input: activation(cols),
        });
    }
    for (name, rows, cols) in [
        ("q8-1x32", 1, 32),
        ("q8-3x64", 3, 64),
        ("q8-5x96", 5, 96),
        ("q8-7x224", 7, 224),
        ("q8-63x256", 63, 256),
        ("q8-64x256", 64, 256),
    ] {
        let mut weight_bytes = Vec::with_capacity(rows * (cols / Q8_ELEMENTS) * Q8_BYTES);
        for row in 0..rows {
            for block in 0..cols / Q8_ELEMENTS {
                let scale = Q8_SCALE_BITS[(row + block * 3) % Q8_SCALE_BITS.len()];
                weight_bytes.extend_from_slice(&scale.to_le_bytes());
                for element in 0..Q8_ELEMENTS {
                    let quant = match element {
                        0 => i8::MIN,
                        1 => i8::MAX,
                        2 => -1,
                        3 => 0,
                        4 => 1,
                        _ => ((row * 19 + block * 11 + element * 7) % 256) as u8 as i8,
                    };
                    weight_bytes.push(quant as u8);
                }
            }
        }
        fixtures.push(LinearFixture {
            name,
            kind: Q8_0,
            rows,
            cols,
            weight_bytes,
            input: activation(cols),
        });
    }
    fixtures
}

fn activation(cols: usize) -> Vec<f32> {
    (0..cols)
        .map(|col| match col {
            0 => 1.0,
            1 => -1.0,
            2 => 0.0,
            // Small dyadic inputs make the first native safety pass exact and avoid
            // underflow/overflow. Broader rounded inputs are covered in CPU tests.
            _ => (((col * 29 + 7) % 97) as f32 - 48.0) / 32.0,
        })
        .collect()
}

impl LinearFixture {
    /// Poison every output row before submission so skipped stores fail validation,
    /// including rows whose correct numerical answer happens to be zero.
    pub(crate) fn unwritten_output(&self) -> Result<Vec<f32>, DeviceError> {
        self.validate()?;
        Ok(vec![f32::from_bits(0x7fc0_4847); self.rows])
    }

    pub(crate) fn weight(&self) -> Weight<'_> {
        Weight {
            bytes: &self.weight_bytes,
            kind: self.kind,
            rows: self.rows,
            cols: self.cols,
        }
    }

    fn validate(&self) -> Result<(), DeviceError> {
        if !(1..=MAX_ROWS).contains(&self.rows) || !(1..=MAX_COLS).contains(&self.cols) {
            return Err(self.error("proof dimensions must be rows 1..=64 and columns 1..=256"));
        }
        let row_bytes = match self.kind {
            BF16 => self.cols * 2,
            Q8_0 if self.cols.is_multiple_of(Q8_ELEMENTS) => self.cols / Q8_ELEMENTS * Q8_BYTES,
            Q8_0 => return Err(self.error("Q8_0 rows require complete 32-element blocks")),
            _ => return Err(self.error("only BF16 and Q8_0 fixtures are supported")),
        };
        if self.weight_bytes.len() != self.rows * row_bytes || self.input.len() != self.cols {
            return Err(
                self.error("packed storage or input length does not match the matrix shape")
            );
        }
        if self.input.iter().any(|value| !value.is_finite()) {
            return Err(self.error("input contains non-finite values"));
        }
        for row in 0..self.rows {
            for col in 0..self.cols {
                if !self.decoded_weight(row, col).is_finite() {
                    return Err(self.error("packed weight contains a non-finite value or scale"));
                }
            }
        }
        Ok(())
    }

    // Called only after geometry and exact byte extents have been validated.
    fn decoded_weight(&self, row: usize, col: usize) -> f32 {
        if self.kind == BF16 {
            let offset = (row * self.cols + col) * 2;
            let bits =
                u16::from_le_bytes([self.weight_bytes[offset], self.weight_bytes[offset + 1]]);
            f32::from_bits(u32::from(bits) << 16)
        } else {
            let block = (row * (self.cols / Q8_ELEMENTS) + col / Q8_ELEMENTS) * Q8_BYTES;
            let scale = beholdd_core::f16_to_f32(u16::from_le_bytes([
                self.weight_bytes[block],
                self.weight_bytes[block + 1],
            ]));
            let quant = self.weight_bytes[block + 2 + col % Q8_ELEMENTS] as i8;
            scale * f32::from(quant)
        }
    }

    /// Serial f32, left-to-right, separate multiply then add; no explicit FMA.
    /// The proof shader must use this same arithmetic contract to claim bitwise parity.
    pub(crate) fn scalar_reference(&self) -> Result<Vec<f32>, DeviceError> {
        self.validate()?;
        let mut output = Vec::with_capacity(self.rows);
        for row in 0..self.rows {
            let mut sum = 0.0f32;
            for col in 0..self.cols {
                let product = self.decoded_weight(row, col) * self.input[col];
                sum += product;
            }
            if !sum.is_finite() {
                return Err(self.error("scalar reference produced a non-finite result"));
            }
            output.push(sum);
        }
        Ok(output)
    }

    pub(crate) fn cpu_reference(&self) -> Result<Vec<f32>, DeviceError> {
        self.validate()?;
        let mut output = vec![f32::NAN; self.rows];
        Cpu::new().matvec(&mut output, &self.input, &self.weight())?;
        if output.iter().any(|value| !value.is_finite()) {
            return Err(self.error("CPU reference produced a non-finite result"));
        }
        Ok(output)
    }

    /// Require serial-native bits exactly, then independently compare the CPU oracle.
    ///
    /// CPU BF16 may use a vector FMA/reduction tree. For each bounded row use
    /// `2 * gamma(2*n) * sum(abs(w*x))`, with f32 unit roundoff `u = 2^-24` and
    /// `gamma(k) = k*u/(1-k*u)`. This covers the sum of two roundoff envelopes for
    /// at most n products and n additions, rather than scaling by a possibly
    /// cancelled output. Fixture decoding is exact and values avoid overflow and
    /// subnormal arithmetic. This is numerical CPU parity, NOT permission for the
    /// native serial kernel to drift; a future fused kernel needs its own contract.
    pub(crate) fn verify_serial_output(&self, actual: &[f32]) -> Result<ParityReport, DeviceError> {
        self.validate()?;
        if actual.len() != self.rows {
            return Err(self.error("native output length does not match the row count"));
        }
        let scalar = self.scalar_reference()?;
        let cpu = self.cpu_reference()?;
        let unit_roundoff = f64::from(f32::EPSILON) / 2.0;
        let ku = (2 * self.cols) as f64 * unit_roundoff;
        let gamma = ku / (1.0 - ku);
        let mut report = ParityReport {
            checked_rows: self.rows,
            max_cpu_absolute_error: 0.0,
            max_cpu_allowed_error: 0.0,
        };
        for row in 0..self.rows {
            if !actual[row].is_finite() || actual[row].to_bits() != scalar[row].to_bits() {
                return Err(self.error(&format!(
                    "row {row} native result {:?} is not bitwise serial-f32 {:?}",
                    actual[row], scalar[row],
                )));
            }
            let absolute_products = (0..self.cols)
                .map(|col| {
                    (f64::from(self.decoded_weight(row, col)) * f64::from(self.input[col])).abs()
                })
                .sum::<f64>();
            let allowed = 2.0 * gamma * absolute_products;
            let error = (f64::from(cpu[row]) - f64::from(scalar[row])).abs();
            if error > allowed {
                return Err(self.error(&format!(
                    "row {row} CPU/scalar error {error:e} exceeds roundoff allowance {allowed:e}",
                )));
            }
            report.max_cpu_absolute_error = report.max_cpu_absolute_error.max(error);
            report.max_cpu_allowed_error = report.max_cpu_allowed_error.max(allowed);
        }
        Ok(report)
    }

    fn error(&self, message: &str) -> DeviceError {
        DeviceError::Shape(format!("HAG fixture {}: {message}", self.name))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_suite_covers_boundaries_without_hidden_padding() {
        let fixtures = linear_fixtures();
        assert_eq!(fixtures.len(), 16);
        for fixture in &fixtures {
            fixture.validate().unwrap();
            assert!(fixture.rows <= MAX_ROWS && fixture.cols <= MAX_COLS);
            let bytes = if fixture.kind == BF16 {
                fixture.rows * fixture.cols * 2
            } else {
                fixture.rows * (fixture.cols / Q8_ELEMENTS) * Q8_BYTES
            };
            assert_eq!(fixture.weight_bytes.len(), bytes, "{}", fixture.name);
        }
        for cols in [1, 2, 31, 32, 33, 63, 64, 65, 255, 256] {
            assert!(fixtures.iter().any(|f| f.kind == BF16 && f.cols == cols));
        }
        for cols in [32, 64, 96, 224, 256] {
            assert!(fixtures.iter().any(|f| f.kind == Q8_0 && f.cols == cols));
        }
        assert!(fixtures.iter().any(|f| f.weight_bytes.len() % 4 == 2));
        assert!(fixtures.iter().any(|f| f.rows == MAX_ROWS));
        assert!(fixtures.iter().any(|f| f.rows > 1 && f.rows % 2 == 1));
    }

    #[test]
    fn q8_layout_has_little_endian_scales_and_signed_extremes() {
        let fixtures = linear_fixtures();
        let mut seen_scales = Vec::new();
        for fixture in fixtures.iter().filter(|f| f.kind == Q8_0) {
            for block in fixture.weight_bytes.chunks_exact(Q8_BYTES) {
                let bits = u16::from_le_bytes([block[0], block[1]]);
                assert!(Q8_SCALE_BITS.contains(&bits));
                seen_scales.push(bits);
                assert_eq!(&block[2..7], &[0x80, 0x7f, 0xff, 0, 1]);
            }
        }
        assert!(Q8_SCALE_BITS.iter().all(|bits| seen_scales.contains(bits)));
        for (bits, expected) in Q8_SCALE_BITS
            .into_iter()
            .zip([0.5, -0.25, 0.0, 0.75, 0.125, -2.0])
        {
            assert_eq!(beholdd_core::f16_to_f32(bits), expected);
        }
    }

    #[test]
    fn references_match_hand_computable_packed_values() {
        let bf16 = LinearFixture {
            name: "known-bf16",
            kind: BF16,
            rows: 1,
            cols: 4,
            weight_bytes: [0x3f80u16, 0x4000, 0xbf00, 0x4080]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
            input: vec![3.0, 5.0, 8.0, 0.25],
        };
        assert_eq!(bf16.scalar_reference().unwrap(), [10.0]);
        bf16.verify_serial_output(&[10.0]).unwrap();
        let mut bytes = vec![0; Q8_BYTES];
        bytes[..2].copy_from_slice(&0x3800u16.to_le_bytes());
        bytes[2] = 0x80;
        bytes[3] = 0x7f;
        let q8 = LinearFixture {
            name: "known-q8",
            kind: Q8_0,
            rows: 1,
            cols: 32,
            weight_bytes: bytes,
            input: vec![1.0; 32],
        };
        assert_eq!(q8.scalar_reference().unwrap(), [-0.5]);
        q8.verify_serial_output(&[-0.5]).unwrap();
    }

    #[test]
    fn complete_suite_has_serial_bits_and_bounded_cpu_reference() {
        for fixture in linear_fixtures() {
            let output = fixture.scalar_reference().unwrap();
            let report = fixture.verify_serial_output(&output).unwrap();
            assert_eq!(report.checked_rows, fixture.rows);
            assert!(report.max_cpu_absolute_error <= report.max_cpu_allowed_error);
            if fixture.kind == Q8_0 {
                assert_eq!(report.max_cpu_absolute_error, 0.0);
            }
        }
    }

    #[test]
    fn independent_decode_matches_existing_scalar_kernel_bits() {
        for fixture in linear_fixtures() {
            let scalar = fixture.scalar_reference().unwrap();
            let row_bytes = fixture.weight_bytes.len() / fixture.rows;
            for (row, bytes) in fixture.weight_bytes.chunks_exact(row_bytes).enumerate() {
                let established = if fixture.kind == BF16 {
                    beholdd_kernel::bf16::dot_bytes_scalar(&fixture.input, bytes)
                } else {
                    beholdd_kernel::kquant::q8_0::dot(bytes, &fixture.input)
                };
                assert_eq!(
                    scalar[row].to_bits(),
                    established.to_bits(),
                    "{} row {row}",
                    fixture.name
                );
            }
        }
    }

    #[test]
    fn every_fixture_rejects_no_op_stale_and_partially_written_outputs() {
        for fixture in linear_fixtures() {
            assert!(fixture
                .verify_serial_output(&fixture.unwritten_output().unwrap())
                .is_err());
            assert!(fixture
                .verify_serial_output(&vec![0.0; fixture.rows])
                .is_err());
            let output = fixture.scalar_reference().unwrap();
            for row in 0..fixture.rows {
                let mut incomplete = output.clone();
                incomplete[row] = f32::NAN;
                assert!(fixture.verify_serial_output(&incomplete).is_err());
            }
            let mut alternate = fixture.clone();
            for value in &mut alternate.input {
                *value *= -0.5;
            }
            assert!(
                alternate.verify_serial_output(&output).is_err(),
                "{}",
                fixture.name
            );
            let refreshed = alternate.scalar_reference().unwrap();
            alternate.verify_serial_output(&refreshed).unwrap();
        }
    }

    #[test]
    fn broader_rounded_inputs_preserve_the_explicit_cpu_roundoff_bound() {
        for mut fixture in linear_fixtures() {
            for (col, value) in fixture.input.iter_mut().enumerate() {
                *value = (((col * 29 + 7) % 97) as f32 - 48.0) / 37.0;
            }
            let serial = fixture.scalar_reference().unwrap();
            fixture.verify_serial_output(&serial).unwrap();
        }
    }

    #[test]
    fn bad_shapes_and_partial_blocks_are_refused() {
        let valid = linear_fixtures().pop().unwrap();
        for (rows, cols) in [
            (0, 32),
            (65, 32),
            (1, 0),
            (1, 257),
            (usize::MAX, 32),
            (1, 33),
        ] {
            let mut fixture = valid.clone();
            fixture.rows = rows;
            fixture.cols = cols;
            assert!(fixture.scalar_reference().is_err());
            assert!(fixture.cpu_reference().is_err());
            assert!(fixture.unwritten_output().is_err());
        }
        let mut truncated = valid.clone();
        truncated.weight_bytes.pop();
        assert!(truncated.scalar_reference().is_err());
        let mut extra = valid.clone();
        extra.weight_bytes.push(0);
        assert!(extra.scalar_reference().is_err());
        let mut bad_input = valid.clone();
        bad_input.input.pop();
        assert!(bad_input.cpu_reference().is_err());
        let mut unsupported = valid;
        unsupported.kind = GgmlType(0);
        assert!(unsupported.scalar_reference().is_err());
    }

    #[test]
    fn non_finite_and_one_bit_output_errors_cannot_pass() {
        let fixture = linear_fixtures().remove(0);
        let expected = fixture.scalar_reference().unwrap();
        assert_eq!(expected, [1.0]);
        for value in [
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::from_bits(1.0f32.to_bits() + 1),
        ] {
            assert!(fixture.verify_serial_output(&[value]).is_err());
        }
        assert!(fixture.verify_serial_output(&[]).is_err());
        let mut bad_input = fixture.clone();
        bad_input.input[0] = f32::NAN;
        assert!(bad_input.scalar_reference().is_err());
        let mut bad_weight = fixture;
        bad_weight
            .weight_bytes
            .copy_from_slice(&0x7f80u16.to_le_bytes());
        assert!(bad_weight.cpu_reference().is_err());
        let mut bad_scale = linear_fixtures().pop().unwrap();
        bad_scale.weight_bytes[..2].copy_from_slice(&0x7c00u16.to_le_bytes());
        assert!(bad_scale.scalar_reference().is_err());
    }
}
