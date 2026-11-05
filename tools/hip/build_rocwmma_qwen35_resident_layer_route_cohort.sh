#!/usr/bin/env bash
set -euo pipefail

# Build/static-only wrapper for the route-aware independent-request cohort
# benchmark. It materializes an exact main-renamed view of the already
# qualified combined benchmark; neither this script nor either compiled
# executable runs a GPU workload.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-q35-route-cohort-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_route_cohort_bench.hip.cpp"
COMBINED="$SOURCE_DIR/rocwmma_qwen35_resident_layer_prepared_parallel_route_bench.hip.cpp"
COMBINED_BUILD="$SOURCE_DIR/build_rocwmma_qwen35_resident_layer_prepared_parallel_route.sh"

EXPECTED_SOURCE_SHA256="2106f99192c53dba7f1e99330bd244a47f0456fd90464914be7fc3c258f2fb45"
EXPECTED_COMBINED_SHA256="4a9b834332c76accd0d86cd0acace19d448ba1aecdaf8ab2b90310cdad958c60"
EXPECTED_COMBINED_BUILD_SHA256="bed5483771113aceffe04fe8aa4a4b9434e08b36a4e732d1685facb42d5449fb"
EXPECTED_GENERATED_SHA256="3bc820e5574842299c21cdf0fc68349ef32fdfefd46e9ba45ef9bbc30fce61d7"
EXPECTED_DEVICE_TEXT_SHA256="14356d71abf83d52c195ec8af20f9354970ba652eae82a283d9481c6ff8d806b"

COMPILER="$ROCM_CORE/lib/llvm/bin/clang++"
RESOURCE_ROOT="$ROCM_CORE/lib/llvm/lib/clang/23"
DEVICE_LIBS="$ROCM_CORE/lib/llvm/amdgcn/bitcode"
HIP_RUNTIME="$ROCM_CORE/lib/libamdhip64.so.7"
OBJDUMP="$ROCM_CORE/lib/llvm/bin/llvm-objdump"
OBJCOPY="$ROCM_CORE/lib/llvm/bin/llvm-objcopy"
READOBJ="$ROCM_CORE/lib/llvm/bin/llvm-readobj"

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

for path in "$SOURCE" "$COMBINED" "$COMBINED_BUILD" "$COMPILER" \
    "$HIP_RUNTIME" "$OBJDUMP" "$OBJCOPY" "$READOBJ" \
    "$ROCWMMA_INCLUDE/rocwmma/rocwmma.hpp"; do
    if [[ ! -e "$path" ]]; then
        printf 'missing required path: %s\n' "$path" >&2
        exit 1
    fi
done

check_sha256 cohort_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 qualified_combined "$COMBINED" "$EXPECTED_COMBINED_SHA256"
check_sha256 qualified_combined_build "$COMBINED_BUILD" \
    "$EXPECTED_COMBINED_BUILD_SHA256"

GENERATED_BASENAME="q35_combined_components.generated.hpp"
if [[ -e "$SOURCE_DIR/$GENERATED_BASENAME" ||
      -L "$SOURCE_DIR/$GENERATED_BASENAME" ]]; then
    printf 'refusing sibling generated header that could shadow the pinned build input: %s\n' \
        "$SOURCE_DIR/$GENERATED_BASENAME" >&2
    exit 1
fi

BUILD_DIR="$(mktemp -d /tmp/beholdd-q35-route-cohort.XXXXXX)"
INSTALL_TEMP=""
cleanup() {
    if [[ -n "$INSTALL_TEMP" ]]; then
        rm -f -- "$INSTALL_TEMP"
    fi
    rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

# This transitive preflight pins the compiler/runtime/HIP/rocWMMA inputs and
# independently requalifies every inherited combined kernel without executing
# the resulting binary.
ROCM_CORE="$ROCM_CORE" ROCWMMA_INCLUDE="$ROCWMMA_INCLUDE" \
    "$COMBINED_BUILD" "$BUILD_DIR/qualified-combined-preflight"

MAIN_COUNT="$(grep -c '^int main(int argc, char\*\* argv) {$' "$COMBINED")"
if [[ "$MAIN_COUNT" != 1 ]]; then
    printf 'expected exactly one anchored combined main, found %s\n' \
        "$MAIN_COUNT" >&2
    exit 1
fi
GENERATED="$BUILD_DIR/$GENERATED_BASENAME"
sed 's/^int main(int argc, char\*\* argv) {/int q35_combined_qualified_embedded_main(int argc, char** argv) {/' \
    "$COMBINED" >"$GENERATED"
if grep -q '^int main(int argc, char\*\* argv) {$' "$GENERATED" ||
   [[ "$(grep -c '^int q35_combined_qualified_embedded_main(int argc, char\*\* argv) {$' "$GENERATED")" != 1 ]]; then
    printf 'combined main rename was not exact\n' >&2
    exit 1
fi
check_sha256 generated_components "$GENERATED" "$EXPECTED_GENERATED_SHA256"

PRE_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
PRE_COMBINED_SHA256="$(sha256sum "$COMBINED" | cut -d' ' -f1)"
PRE_COMBINED_BUILD_SHA256="$(sha256sum "$COMBINED_BUILD" | cut -d' ' -f1)"
HIP_OVERLAY="$BUILD_DIR/hip-root"
CANDIDATE_BINARY="$BUILD_DIR/qualified-route-cohort"
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
    -I"$BUILD_DIR" \
    -I"$SOURCE_DIR" \
    -I"$RESOURCE_ROOT" \
    -I"$RESOURCE_ROOT/include" \
    -I"$ROCM_CORE/include" \
    -I"$ROCWMMA_INCLUDE" \
    "$SOURCE" \
    -Wl,-rpath,"$ROCM_CORE/lib" \
    -o "$CANDIDATE_BINARY"

if [[ "$(sha256sum "$SOURCE" | cut -d' ' -f1)" != "$PRE_SOURCE_SHA256" ||
      "$(sha256sum "$COMBINED" | cut -d' ' -f1)" != "$PRE_COMBINED_SHA256" ||
      "$(sha256sum "$COMBINED_BUILD" | cut -d' ' -f1)" != "$PRE_COMBINED_BUILD_SHA256" ]]; then
    printf 'cohort source or qualified combined input changed during build\n' >&2
    exit 1
fi
check_sha256 postcompile_cohort_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 postcompile_qualified_combined "$COMBINED" \
    "$EXPECTED_COMBINED_SHA256"
check_sha256 postcompile_qualified_combined_build "$COMBINED_BUILD" \
    "$EXPECTED_COMBINED_BUILD_SHA256"

VERIFY_BINARY="$CANDIDATE_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >"$BUILD_DIR/offloading.txt"
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
if [[ ! -s "$DEVICE_OBJECT" ]]; then
    printf 'missing exact gfx1151 device image\n' >&2
    exit 1
fi
if [[ "$(find "$BUILD_DIR" -maxdepth 1 -type f -name 'qualified-route-cohort.*amdgcn*' | wc -l)" != 1 ]]; then
    printf 'binary must contain exactly one AMDGPU device image\n' >&2
    exit 1
fi
"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$BUILD_DIR/header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' "$BUILD_DIR/header.txt"; then
    printf 'cohort code object is not exact gfx1151\n' >&2
    exit 1
fi
"$READOBJ" --notes "$DEVICE_OBJECT" >"$BUILD_DIR/notes.txt"

# Every Beholdd kernel in this translation unit must remain wave32, spill-free,
# scratch-free, and free of a dynamic stack. The cohort geometry is pinned so
# a launch-bounds or LDS regression cannot hide behind the final text digest.
awk '
    /^[[:space:]]+\.group_segment_fixed_size:/ { group_bytes = $2; next }
    /^[[:space:]]+\.max_flat_workgroup_size:/ { maximum = $2; next }
    /^[[:space:]]+\.name:/ {
        name = $2
        active = name ~ /^q35_/
        if (active) { seen[name]++; kernel_count++ }
        if (name == "q35_cohort_rmsnorm_f32_to_bf16" ||
            name == "q35_cohort_router_and_shared_gate_f32") {
            if (group_bytes != 1024 || maximum != 1024) {
                print "cohort reduction geometry mismatch: " name > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_stable_softmax_topk8_parallel") {
            if (group_bytes != 80 || maximum != 256) {
                print "cohort router geometry mismatch: LDS=" group_bytes \
                      " workgroup=" maximum > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_plan_routes") {
            if (group_bytes != 2052 || maximum != 256) {
                print "cohort planner geometry mismatch" > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_gather_tile_input" ||
            name == "q35_cohort_silu_multiply_to_bf16" ||
            name == "q35_cohort_route_shared_residual_reduce") {
            if (group_bytes != 0 || maximum != 1024) {
                print "cohort element-kernel geometry mismatch: " name > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_prepared_gate_up_wmma" ||
            name == "q35_cohort_prepared_down_wmma") {
            if (group_bytes != 0 || maximum != 32) {
                print "cohort WMMA geometry mismatch: " name > "/dev/stderr"
                failed = 1
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
    active && /^[[:space:]]+\.vgpr_spill_count:/ && $2 != 0 {
        print "VGPR spill in " name > "/dev/stderr"; failed = 1
    }
    active && /^[[:space:]]+\.vgpr_count:/ {
        if (name == "q35_cohort_stable_softmax_topk8_parallel") {
            route_vgpr = $2
            if ($2 > 64) {
                print "cohort router VGPR cap exceeded" > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_plan_routes") {
            planner_vgpr = $2
            if ($2 > 64) {
                print "cohort planner VGPR cap exceeded" > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_prepared_gate_up_wmma") {
            gate_vgpr = $2
            if ($2 > 64) {
                print "cohort gate/up VGPR cap exceeded" > "/dev/stderr"
                failed = 1
            }
        }
        if (name == "q35_cohort_prepared_down_wmma") {
            down_vgpr = $2
            if ($2 > 64) {
                print "cohort down VGPR cap exceeded" > "/dev/stderr"
                failed = 1
            }
        }
    }
    active && /^[[:space:]]+\.uses_dynamic_stack:/ && $2 != "false" {
        print "dynamic stack in " name > "/dev/stderr"; failed = 1
    }
    active && /^[[:space:]]+\.wavefront_size:/ && $2 != 32 {
        print "non-wave32 kernel: " name > "/dev/stderr"; failed = 1
    }
    END {
        required[1] = "q35_cohort_rmsnorm_f32_to_bf16"
        required[2] = "q35_cohort_router_and_shared_gate_f32"
        required[3] = "q35_cohort_stable_softmax_topk8_parallel"
        required[4] = "q35_cohort_plan_routes"
        required[5] = "q35_cohort_gather_tile_input"
        required[6] = "q35_cohort_prepared_gate_up_wmma"
        required[7] = "q35_cohort_silu_multiply_to_bf16"
        required[8] = "q35_cohort_prepared_down_wmma"
        required[9] = "q35_cohort_route_shared_residual_reduce"
        for (idx = 1; idx <= 9; ++idx) {
            if (seen[required[idx]] != 1) {
                print "missing or duplicate cohort kernel metadata: " required[idx] > "/dev/stderr"
                failed = 1
            }
        }
        if (kernel_count != 21) {
            print "expected exactly 21 q35 kernels, found " kernel_count > "/dev/stderr"
            failed = 1
        }
        if (route_vgpr == "" || planner_vgpr == "" ||
            gate_vgpr == "" || down_vgpr == "") {
            print "incomplete cohort resource metadata" > "/dev/stderr"
            failed = 1
        }
        print "qualified_q35_kernel_count=" kernel_count \
              " cohort_route_vgpr=" route_vgpr \
              " cohort_planner_vgpr=" planner_vgpr \
              " cohort_gate_up_vgpr=" gate_vgpr \
              " cohort_down_vgpr=" down_vgpr
        exit failed
    }
' "$BUILD_DIR/notes.txt"

"$OBJDUMP" -d --mcpu=gfx1151 "$DEVICE_OBJECT" >"$BUILD_DIR/isa.txt"
WMMA_COUNT="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$BUILD_DIR/isa.txt" || true)"
if [[ "$WMMA_COUNT" != 16 ]]; then
    printf 'expected exactly 16 native BF16 WMMA opcodes, found %s\n' \
        "$WMMA_COUNT" >&2
    exit 1
fi
if grep -Eq 'scratch_(load|store)|flat_scratch' "$BUILD_DIR/isa.txt"; then
    printf 'cohort device ISA contains scratch traffic\n' >&2
    exit 1
fi
for symbol in q35_cohort_prepared_gate_up_wmma \
    q35_cohort_prepared_down_wmma; do
    symbol_isa="$BUILD_DIR/$symbol.isa"
    "$OBJDUMP" -d --mcpu=gfx1151 --disassemble-symbols="$symbol" \
        "$DEVICE_OBJECT" >"$symbol_isa"
    symbol_wmma="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$symbol_isa" || true)"
    if [[ "$symbol_wmma" != 2 ]]; then
        printf 'expected two native BF16 WMMA opcodes in %s, found %s\n' \
            "$symbol" "$symbol_wmma" >&2
        exit 1
    fi
done
"$OBJDUMP" -d --mcpu=gfx1151 \
    --disassemble-symbols=q35_cohort_stable_softmax_topk8_parallel \
    "$DEVICE_OBJECT" >"$BUILD_DIR/cohort-route-isa.txt"
"$OBJDUMP" -d --mcpu=gfx1151 \
    --disassemble-symbols=q35_cohort_plan_routes \
    "$DEVICE_OBJECT" >"$BUILD_DIR/cohort-planner-isa.txt"
ROUTE_BARRIERS="$(grep -c 's_barrier' "$BUILD_DIR/cohort-route-isa.txt" || true)"
PLANNER_BARRIERS="$(grep -c 's_barrier' "$BUILD_DIR/cohort-planner-isa.txt" || true)"
if (( ROUTE_BARRIERS < 1 || PLANNER_BARRIERS < 1 )); then
    printf 'cohort route/planner synchronization barrier missing\n' >&2
    exit 1
fi
"$OBJCOPY" --dump-section .text="$BUILD_DIR/device-text.bin" "$DEVICE_OBJECT"
check_sha256 device_text "$BUILD_DIR/device-text.bin" \
    "$EXPECTED_DEVICE_TEXT_SHA256"
CODE_OBJECT_SHA256="$(sha256sum "$DEVICE_OBJECT" | cut -d' ' -f1)"

if ! readelf -d "$CANDIDATE_BINARY" | grep -q 'libamdhip64.so.7' ||
   ! readelf -d "$CANDIDATE_BINARY" | grep -Fq "$ROCM_CORE/lib"; then
    printf 'cohort binary lacks pinned HIP runtime/runpath\n' >&2
    exit 1
fi

EXECUTABLE_SHA256="$(sha256sum "$CANDIDATE_BINARY" | cut -d' ' -f1)"
OUTPUT_DIR="$(dirname -- "$OUTPUT")"
OUTPUT_BASE="$(basename -- "$OUTPUT")"
if [[ ! -d "$OUTPUT_DIR" ]]; then
    printf 'output directory does not exist: %s\n' "$OUTPUT_DIR" >&2
    exit 1
fi
INSTALL_TEMP="$(mktemp "$OUTPUT_DIR/.${OUTPUT_BASE}.qualified.XXXXXX")"
cp -- "$CANDIDATE_BINARY" "$INSTALL_TEMP"
chmod --reference="$CANDIDATE_BINARY" "$INSTALL_TEMP"
STAGED_SHA256="$(sha256sum "$INSTALL_TEMP" | cut -d' ' -f1)"
if [[ "$STAGED_SHA256" != "$EXECUTABLE_SHA256" ]]; then
    printf 'qualified executable changed while staging output\n' >&2
    exit 1
fi
mv -f -- "$INSTALL_TEMP" "$OUTPUT"
INSTALL_TEMP=""
INSTALLED_SHA256="$(sha256sum "$OUTPUT" | cut -d' ' -f1)"
if [[ "$INSTALLED_SHA256" != "$EXECUTABLE_SHA256" ]]; then
    printf 'installed executable does not match qualified candidate\n' >&2
    exit 1
fi

printf 'output=%s\n' "$OUTPUT"
printf 'executable_sha256=%s\n' "$EXECUTABLE_SHA256"
printf 'code_object_sha256=%s\n' "$CODE_OBJECT_SHA256"
printf 'gfx1151=true wave32=true private_scratch_bytes=0 '
printf 'native_bf16_wmma_instructions=%s cohort_route_barriers=%s ' \
    "$WMMA_COUNT" "$ROUTE_BARRIERS"
printf 'cohort_planner_barriers=%s\n' "$PLANNER_BARRIERS"
printf 'static_qualification=PASS gpu_execution=false\n'
