//! Isolated recurrent-state, sigmoid and finite-status programs for the Qwen35 probe library.
//! Source/byte agreement is distinct from native numerical qualification.

include!("qwen35_state_words.rs");

#[cfg(test)]
pub(crate) const SOURCES: [&str; 6] = [
    "delta_conv_gfx1151",
    "delta_normalize_gates_gfx1151",
    "delta_recurrence_gfx1151",
    "delta_final_gfx1151",
    "sigmoid_gate_qwen35_gfx1151",
    "finite_check_gfx1151",
];

pub(crate) fn program_bytes() -> [Vec<u8>; 6] {
    WORDS.map(super::padded_shader)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Host arithmetic model of range reduction only. libm exp2 is not the GPU
    // SFU: these checks cannot substitute for the separate native tail gate.
    fn reduced_exp(value: f32) -> (f32, f64) {
        let a = value.abs().min(128.0);
        let n = (a * f32::from_bits(0x3fb8_aa3b)) as u32;
        let hi = n as f32 * f32::from_bits(0x3f31_7200);
        assert_eq!(hi as f64, n as f64 * (22713.0 / 32768.0));
        let lo = n as f32 * f32::from_bits(0x35bf_be8e);
        let residual = (a - hi) - lo;
        let argument = residual * f32::from_bits(0xbfb8_aa3b);
        let power = f32::from_bits((191 - n) << 23);
        let scaled = argument.exp2() * power;
        let result = scaled * f32::from_bits(0x1f80_0000);
        let error = ((argument as f64 - n as f64) - (-(a as f64) * std::f64::consts::LOG2_E)).abs();
        (result, error)
    }

    #[test]
    fn split_ln2_reduction_bounds_exponent_error_before_the_sfu() {
        let boundaries = [1u32, 64, 129, 184].into_iter().flat_map(|n| {
            let center = (n as f64 * std::f64::consts::LN_2) as f32;
            [center.to_bits() - 1, center.to_bits(), center.to_bits() + 1].map(f32::from_bits)
        });
        for value in (0..=4096).map(|i| i as f32 / 32.0).chain(boundaries) {
            let (actual, error) = reduced_exp(value);
            assert!(error < 1.2e-7, "input {value}: reduction error {error}");
            let expected = (-value).exp();
            assert!(
                actual.to_bits().abs_diff(expected.to_bits()) <= 8,
                "input {value}: host exp2 model {actual:?} / exp {expected:?}"
            );
        }
    }

    #[test]
    fn split_ln2_keeps_the_exp_boundary_before_tail_products() {
        for value in [-90.0f32, -100.0, -103.0, -105.0, -f32::MAX] {
            let (actual, _) = reduced_exp(value);
            let expected = value.exp();
            assert!(actual.to_bits().abs_diff(expected.to_bits()) <= 8);
            let actual_product = value * actual;
            let expected_product = value * expected;
            assert!(
                actual_product
                    .to_bits()
                    .abs_diff(expected_product.to_bits())
                    <= 64
            );
        }
        assert_eq!(reduced_exp(-103.0).0.to_bits(), 1);
        assert_eq!(reduced_exp(-105.0).0.to_bits(), 0);
        assert_eq!(((-105.0f32) * reduced_exp(-105.0).0).to_bits(), 0x8000_0000);
        assert_eq!(reduced_exp(0.0).0.to_bits(), 1.0f32.to_bits());
        assert_eq!(reduced_exp(f32::from_bits(1)).0.to_bits(), 1.0f32.to_bits());
    }

    #[test]
    fn state_library_fits_transport_bounds_with_fixed_register_abi() {
        let library = super::super::qwen35_program_bytes();
        assert_eq!(library.len(), 18);
        for (index, program) in program_bytes().iter().enumerate() {
            assert_eq!(library[index + 12], *program);
            assert!(program.len() <= 4096);
            assert_eq!(program.len() % 4, 0);
            assert_eq!(
                &program[program.len() - 256..],
                [0x00, 0x00, 0x9f, 0xbf].repeat(64)
            );
        }
        let refs: Vec<_> = library.iter().map(Vec::as_slice).collect();
        assert!(exec_hag::linux::compute::shader_library_bytes(&refs).unwrap() <= 32768);
    }

    #[test]
    #[ignore = "requires LLVM tools; no GPU or model access"]
    fn llvm_qwen35_state_sources_match_embedded_words() {
        use std::{path::PathBuf, process::Command};
        let sources = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/hag/kernels");
        let tool_dir = std::env::var_os("BEHOLDD_LLVM_BIN").map(PathBuf::from);
        let tool = |name: &str| {
            tool_dir
                .as_ref()
                .map_or_else(|| PathBuf::from(name), |p| p.join(name))
        };
        struct Temporary(PathBuf);
        impl Drop for Temporary {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let scratch = Temporary(std::env::temp_dir().join(format!(
                "qwen35-state-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            )));
        std::fs::create_dir(&scratch.0).unwrap();
        let run = |command: &mut Command| {
            let output = command
                .output()
                .expect("start LLVM tool; set BEHOLDD_LLVM_BIN");
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            output
        };
        for (name, bytes) in SOURCES.into_iter().zip(program_bytes()) {
            let object = scratch.0.join(format!("{name}.o"));
            let binary = scratch.0.join(format!("{name}.bin"));
            run(Command::new(tool("llvm-mc"))
                .args([
                    "--triple=amdgcn-amd-amdhsa",
                    "--mcpu=gfx1151",
                    "--mattr=+wavefrontsize64,-wavefrontsize32",
                    "--filetype=obj",
                    "-I",
                ])
                .arg(&sources)
                .arg(sources.join(format!("{name}.s")))
                .arg("-o")
                .arg(&object));
            run(Command::new(tool("llvm-objcopy"))
                .arg("--dump-section")
                .arg(format!(".text={}", binary.display()))
                .arg(&object));
            assert_eq!(
                std::fs::read(binary).unwrap(),
                bytes,
                "assembly drift: {name}"
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
            assert!(text.contains("s_waitcnt_vscnt"));
            assert!(!text.contains("v_fma") && !text.contains("v_mac"));
            assert!(!text.contains("flat_") && !text.contains("global_") && !text.contains("ds_"));
            // Tokenize operands to reject ordinary VGPRs above the fixed eight,
            // including endpoints of ranges such as v[6:7].
            for line in text.lines().filter(|l| l.contains('\t')) {
                for token in line.split(|c: char| {
                    !(c.is_ascii_alphanumeric() || c == '[' || c == ']' || c == ':')
                }) {
                    if let Some(register) = token.strip_prefix('v') {
                        let register = register.trim_matches(['[', ']']);
                        if !register.is_empty()
                            && register.chars().all(|c| c.is_ascii_digit() || c == ':')
                        {
                            for number in register.split(':') {
                                assert!(number.parse::<u32>().unwrap() < 8, "{name}: {line}");
                            }
                        }
                    }
                }
            }
            let relocations = run(Command::new(tool("llvm-objdump"))
                .arg("--reloc")
                .arg(&object));
            assert!(!String::from_utf8_lossy(&relocations.stdout).contains("R_AMDGPU"));
        }
    }
}
