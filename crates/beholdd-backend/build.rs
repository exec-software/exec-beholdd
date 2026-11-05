use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=shaders/q8_0_matvec.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_matvec.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_reduce.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_fast.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_fast.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_reduce_fast.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_fast_wg32.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_fast_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_fast_wg32.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_fast_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_subgroup_wg32.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_subgroup_wg64.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_subgroup_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_subgroup_wg32.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_subgroup_wg64.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_subgroup_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_fast_batch_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_fast_batch_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_reduce_fast_batch.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_gate_up_fast_tile4_wg128.comp");
    println!("cargo:rerun-if-changed=shaders/bf16_routed_moe_down_fast_tile4_wg128.comp");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_VULKAN");
    println!("cargo:rerun-if-env-changed=GLSLC");
    if env::var_os("CARGO_FEATURE_VULKAN").is_none() {
        return;
    }

    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("cargo output directory"));
    let compiler = env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());
    for shader in [
        "q8_0_matvec",
        "bf16_matvec",
        "bf16_routed_moe_gate_up",
        "bf16_routed_moe_down",
        "bf16_routed_moe_reduce",
        // The paired-load variants share the guarded descriptor ABI. Host-side
        // eligibility selects them for even model widths and keeps guarded fallback.
        "bf16_routed_moe_gate_up_fast",
        "bf16_routed_moe_down_fast",
        "bf16_routed_moe_reduce_fast",
        // Fixed-tree candidates bracket the qualified WG64 implementation.
        "bf16_routed_moe_gate_up_fast_wg32",
        "bf16_routed_moe_gate_up_fast_wg128",
        "bf16_routed_moe_down_fast_wg32",
        "bf16_routed_moe_down_fast_wg128",
        // Subgroup candidates replace the shared-memory tree with subgroupAdd,
        // one cross-subgroup barrier, and a lane-zero serial fold.
        "bf16_routed_moe_gate_up_subgroup_wg32",
        "bf16_routed_moe_gate_up_subgroup_wg64",
        "bf16_routed_moe_gate_up_subgroup_wg128",
        "bf16_routed_moe_down_subgroup_wg32",
        "bf16_routed_moe_down_subgroup_wg64",
        "bf16_routed_moe_down_subgroup_wg128",
        // Whole-batch Tree128 candidates add only integer item-base addressing to
        // the qualified arithmetic and collapse 3*N launches into three dispatches.
        "bf16_routed_moe_gate_up_fast_batch_wg128",
        "bf16_routed_moe_down_fast_batch_wg128",
        "bf16_routed_moe_reduce_fast_batch",
        // Expert-major Tree128 candidates amortize each weight-pair load across
        // up to four item-route occurrences selected through mapped tile metadata.
        "bf16_routed_moe_gate_up_fast_tile4_wg128",
        "bf16_routed_moe_down_fast_tile4_wg128",
    ] {
        let source = manifest.join("shaders").join(format!("{shader}.comp"));
        let output = out_dir.join(format!("{shader}.spv"));
        let compiled = Command::new(&compiler)
            .arg("--target-env=vulkan1.2")
            .arg("-fshader-stage=compute")
            .arg("-O")
            .arg("-o")
            .arg(&output)
            .arg(&source)
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "the `vulkan` feature needs {:?} to compile {}: {error}",
                    compiler,
                    source.display(),
                )
            });
        if !compiled.status.success() {
            panic!(
                "glslc failed for {}:\n{}",
                source.display(),
                String::from_utf8_lossy(&compiled.stderr)
            );
        }
    }
}
