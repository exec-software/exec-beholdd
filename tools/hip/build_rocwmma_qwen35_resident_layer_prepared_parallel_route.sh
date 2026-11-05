#!/usr/bin/env bash
set -euo pipefail

# Build and statically qualify the combined prepared-WMMA/parallel-router
# experiment. This script compiles binaries but never executes a GPU workload.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-q35-prepared-parallel-route-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_prepared_parallel_route_bench.hip.cpp"
CONTROL="$SOURCE_DIR/rocwmma_qwen35_resident_layer_bench.hip.cpp"
PREPARED_REFERENCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_prepared_bench.hip.cpp"
PARALLEL_REFERENCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_parallel_route_bench.hip.cpp"
PARALLEL_BUILD="$SOURCE_DIR/build_rocwmma_qwen35_resident_layer_parallel_route.sh"

EXPECTED_SOURCE_SHA256="4a9b834332c76accd0d86cd0acace19d448ba1aecdaf8ab2b90310cdad958c60"
EXPECTED_CONTROL_SHA256="dfed8f57bc310e1c8f366a13601ab9dccdd88da82744d5980e8370e858d1f3dd"
EXPECTED_PREPARED_REFERENCE_SHA256="affe38a1c6dcf1d0eadc4e54d47f46dd7c6270d227817da699fd38020f9ac5f4"
EXPECTED_PARALLEL_REFERENCE_SHA256="abd707ec58bcb32a62de5cab063722362197496ba1d0bf2b77acd7bb1817eaaf"
EXPECTED_PARALLEL_BUILD_SHA256="30899048681e2784083256a9c507a8e28221340ee0fa2e77045264cc10954fc3"
EXPECTED_DEVICE_TEXT_SHA256="d902de80d06ab66842c18fd74d03d06ce210d8ac6f44eba298bc9be814b70ef2"

COMPILER="$ROCM_CORE/lib/llvm/bin/clang++"
RESOURCE_ROOT="$ROCM_CORE/lib/llvm/lib/clang/23"
DEVICE_LIBS="$ROCM_CORE/lib/llvm/amdgcn/bitcode"
HIP_RUNTIME="$ROCM_CORE/lib/libamdhip64.so.7"
OBJDUMP="$ROCM_CORE/lib/llvm/bin/llvm-objdump"
OBJCOPY="$ROCM_CORE/lib/llvm/bin/llvm-objcopy"
READOBJ="$ROCM_CORE/lib/llvm/bin/llvm-readobj"

for path in "$SOURCE" "$CONTROL" "$PREPARED_REFERENCE" \
    "$PARALLEL_REFERENCE" "$PARALLEL_BUILD" "$COMPILER" "$OBJDUMP" \
    "$OBJCOPY" "$READOBJ" "$HIP_RUNTIME" \
    "$ROCWMMA_INCLUDE/rocwmma/rocwmma.hpp"; do
    if [[ ! -e "$path" ]]; then
        printf 'missing required path: %s\n' "$path" >&2
        exit 1
    fi
done

check_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local observed
    observed="$(sha256sum "$path" | cut -d' ' -f1)"
    if [[ "$observed" != "$expected" ]]; then
        printf '%s hash mismatch: expected %s, found %s\n' \
            "$label" "$expected" "$observed" >&2
        exit 1
    fi
    printf '%s_sha256=%s (frozen match)\n' "$label" "$observed"
}

check_sha256 source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 frozen_control "$CONTROL" "$EXPECTED_CONTROL_SHA256"
check_sha256 prepared_reference "$PREPARED_REFERENCE" \
    "$EXPECTED_PREPARED_REFERENCE_SHA256"
check_sha256 parallel_reference "$PARALLEL_REFERENCE" \
    "$EXPECTED_PARALLEL_REFERENCE_SHA256"
check_sha256 parallel_build "$PARALLEL_BUILD" \
    "$EXPECTED_PARALLEL_BUILD_SHA256"

# Reuse the already strict parallel-route verifier as the provenance preflight:
# it pins compiler/runtime/HIP headers/device libraries/rocWMMA commit+tree and
# the frozen baseline, and it does not execute its output.
PREFLIGHT_DIR="$(mktemp -d /tmp/beholdd-q35-combined-preflight.XXXXXX)"
cleanup() {
    rm -rf -- "$PREFLIGHT_DIR"
}
trap cleanup EXIT
ROCM_CORE="$ROCM_CORE" ROCWMMA_INCLUDE="$ROCWMMA_INCLUDE" \
    "$PARALLEL_BUILD" "$PREFLIGHT_DIR/parallel-route-preflight"

PRE_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
PRE_CONTROL_SHA256="$(sha256sum "$CONTROL" | cut -d' ' -f1)"
PRE_PREPARED_SHA256="$(sha256sum "$PREPARED_REFERENCE" | cut -d' ' -f1)"
PRE_PARALLEL_SHA256="$(sha256sum "$PARALLEL_REFERENCE" | cut -d' ' -f1)"
HIP_OVERLAY="$PREFLIGHT_DIR/hip-root"
mkdir -p "$HIP_OVERLAY/lib"
ln -s "$HIP_RUNTIME" "$HIP_OVERLAY/lib/libamdhip64.so"

"$COMPILER" \
    -x hip \
    --offload-arch=gfx1151 \
    -nogpuinc \
    --hip-path="$HIP_OVERLAY" \
    --rocm-device-lib-path="$DEVICE_LIBS" \
    -D__HIP_PLATFORM_AMD__=1 \
    -std=c++20 \
    -O3 \
    -Wall \
    -Wextra \
    -Werror \
    -Wno-unused-parameter \
    -Wno-unused-function \
    -Wno-unused-local-typedef \
    -Wno-sign-compare \
    -I"$RESOURCE_ROOT" \
    -I"$RESOURCE_ROOT/include" \
    -I"$ROCM_CORE/include" \
    -I"$ROCWMMA_INCLUDE" \
    "$SOURCE" \
    -Wl,-rpath,"$ROCM_CORE/lib" \
    -o "$OUTPUT"

if [[ "$(sha256sum "$SOURCE" | cut -d' ' -f1)" != "$PRE_SOURCE_SHA256" ||
      "$(sha256sum "$CONTROL" | cut -d' ' -f1)" != "$PRE_CONTROL_SHA256" ||
      "$(sha256sum "$PREPARED_REFERENCE" | cut -d' ' -f1)" != "$PRE_PREPARED_SHA256" ||
      "$(sha256sum "$PARALLEL_REFERENCE" | cut -d' ' -f1)" != "$PRE_PARALLEL_SHA256" ]]; then
    printf 'source or a qualified reference changed during compilation\n' >&2
    exit 1
fi
check_sha256 postcompile_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 postcompile_frozen_control "$CONTROL" "$EXPECTED_CONTROL_SHA256"
check_sha256 postcompile_prepared_reference "$PREPARED_REFERENCE" \
    "$EXPECTED_PREPARED_REFERENCE_SHA256"
check_sha256 postcompile_parallel_reference "$PARALLEL_REFERENCE" \
    "$EXPECTED_PARALLEL_REFERENCE_SHA256"

VERIFY_BINARY="$PREFLIGHT_DIR/verify-binary"
cp -- "$OUTPUT" "$VERIFY_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >/dev/null
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
if [[ ! -s "$DEVICE_OBJECT" ]]; then
    printf 'missing exact gfx1151 device image in %s\n' "$OUTPUT" >&2
    exit 1
fi
"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$PREFLIGHT_DIR/header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' "$PREFLIGHT_DIR/header.txt"; then
    printf 'embedded code object is not exact gfx1151\n' >&2
    exit 1
fi
"$READOBJ" --notes "$DEVICE_OBJECT" >"$PREFLIGHT_DIR/notes.txt"

awk '
    /^[[:space:]]+\.group_segment_fixed_size:/ { group_bytes = $2; next }
    /^[[:space:]]+\.max_flat_workgroup_size:/ { maximum = $2; next }
    /^[[:space:]]+\.name:/ {
        name = $2
        active = name ~ /^q35_/
        if (active) {
            seen[name] = 1
            if (name == "q35_prepared_stable_softmax_topk8_parallel") {
                route_seen = 1
                if (group_bytes != 2056 || maximum != 256) {
                    print "parallel route resource geometry mismatch" > "/dev/stderr"
                    failed = 1
                }
            }
            if (name == "q35_combined_prepared_gate_up_wmma" ||
                name == "q35_combined_prepared_down_wmma") {
                if (group_bytes != 0 || maximum != 32) {
                    print "prepared WMMA resource geometry mismatch: " name > "/dev/stderr"
                    failed = 1
                }
            }
        }
        next
    }
    active && /^[[:space:]]+\.private_segment_fixed_size:/ && $2 != 0 {
        print "private memory in " name ": " $2 > "/dev/stderr"; failed = 1
    }
    active && /^[[:space:]]+\.sgpr_spill_count:/ && $2 != 0 {
        print "SGPR spill in " name > "/dev/stderr"; failed = 1
    }
    active && /^[[:space:]]+\.vgpr_count:/ {
        if (name == "q35_prepared_stable_softmax_topk8_parallel") {
            route_vgpr = $2
            if ($2 > 64) { print "route VGPR cap exceeded" > "/dev/stderr"; failed = 1 }
        }
        if (name == "q35_combined_prepared_gate_up_wmma") {
            gate_vgpr = $2
            if ($2 > 64) { print "prepared gate/up VGPR cap exceeded" > "/dev/stderr"; failed = 1 }
        }
        if (name == "q35_combined_prepared_down_wmma") {
            down_vgpr = $2
            if ($2 > 64) { print "prepared down VGPR cap exceeded" > "/dev/stderr"; failed = 1 }
        }
    }
    active && /^[[:space:]]+\.vgpr_spill_count:/ && $2 != 0 {
        print "VGPR spill in " name > "/dev/stderr"; failed = 1
    }
    active && /^[[:space:]]+\.uses_dynamic_stack:/ && $2 != "false" {
        print "dynamic stack in " name > "/dev/stderr"; failed = 1
    }
    active && /^[[:space:]]+\.wavefront_size:/ && $2 != 32 {
        print "non-wave32 kernel: " name > "/dev/stderr"; failed = 1
    }
    END {
        required[1] = "q35_rmsnorm_f32_to_f32_bf16"
        required[2] = "q35_router_and_shared_gate_f32"
        required[3] = "q35_stable_softmax_topk8_prepared_serial"
        required[4] = "q35_selected_shared_gate_up_wmma"
        required[5] = "q35_silu_multiply_to_bf16"
        required[6] = "q35_selected_shared_down_wmma"
        required[7] = "q35_route_shared_residual_reduce"
        required[8] = "q35_prepared_stable_softmax_topk8_parallel"
        required[9] = "q35_combined_prepare_gate_up_row_major"
        required[10] = "q35_combined_prepare_down_row_major"
        required[11] = "q35_combined_prepared_gate_up_wmma"
        required[12] = "q35_combined_prepared_down_wmma"
        for (idx = 1; idx <= 12; ++idx) {
            if (!seen[required[idx]]) {
                print "missing kernel metadata: " required[idx] > "/dev/stderr"
                failed = 1
            }
        }
        if (!route_seen || route_vgpr == "" || gate_vgpr == "" || down_vgpr == "") {
            print "incomplete combined resource metadata" > "/dev/stderr"
            failed = 1
        }
        print "parallel_route_vgpr=" route_vgpr \
              " prepared_gate_up_vgpr=" gate_vgpr \
              " prepared_down_vgpr=" down_vgpr \
              " route_lds_bytes=2056"
        exit failed
    }
' "$PREFLIGHT_DIR/notes.txt"

"$OBJDUMP" -d --mcpu=gfx1151 "$DEVICE_OBJECT" >"$PREFLIGHT_DIR/isa.txt"
WMMA_COUNT="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$PREFLIGHT_DIR/isa.txt")"
if [[ "$WMMA_COUNT" != 12 ]]; then
    printf 'expected exactly 12 emitted BF16 WMMA opcodes, found %s\n' \
        "$WMMA_COUNT" >&2
    exit 1
fi
if grep -Eq 'scratch_(load|store)|flat_scratch' "$PREFLIGHT_DIR/isa.txt"; then
    printf 'device ISA contains scratch traffic\n' >&2
    exit 1
fi
"$OBJDUMP" -d --mcpu=gfx1151 \
    --disassemble-symbols=q35_prepared_stable_softmax_topk8_parallel \
    "$DEVICE_OBJECT" >"$PREFLIGHT_DIR/parallel-route-isa.txt"
BARRIER_COUNT="$(grep -c 's_barrier' "$PREFLIGHT_DIR/parallel-route-isa.txt")"
if (( BARRIER_COUNT < 1 )); then
    printf 'parallel route contains no workgroup barrier\n' >&2
    exit 1
fi

"$OBJCOPY" --dump-section .text="$PREFLIGHT_DIR/device-text.bin" "$DEVICE_OBJECT"
check_sha256 device_text "$PREFLIGHT_DIR/device-text.bin" \
    "$EXPECTED_DEVICE_TEXT_SHA256"
CODE_OBJECT_SHA256="$(sha256sum "$DEVICE_OBJECT" | cut -d' ' -f1)"

if ! readelf -d "$OUTPUT" | grep -q 'libamdhip64.so.7'; then
    printf 'binary does not depend on pinned libamdhip64.so.7\n' >&2
    exit 1
fi
if ! readelf -d "$OUTPUT" | grep -Fq "$ROCM_CORE/lib"; then
    printf 'binary does not carry the pinned ROCm runpath\n' >&2
    exit 1
fi

printf 'output=%s\n' "$OUTPUT"
printf 'code_object_sha256=%s\n' "$CODE_OBJECT_SHA256"
printf 'gfx1151=true wave32=true private_scratch_bytes=0 '
printf 'native_bf16_wmma_instructions=%s parallel_route_barriers=%s\n' \
    "$WMMA_COUNT" "$BARRIER_COUNT"
printf 'static_qualification=PASS gpu_execution=false\n'
