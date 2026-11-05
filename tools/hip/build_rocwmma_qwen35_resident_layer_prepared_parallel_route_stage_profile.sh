#!/usr/bin/env bash
set -euo pipefail

# Build/static-only wrapper for the combined stage profiler. It derives a
# temporary main-renamed view of the qualified combined source; neither this
# script nor the profiler executable writes the pack or resident weights.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-q35-prepared-parallel-route-stage-profile}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_prepared_parallel_route_stage_profile.hip.cpp"
COMBINED="$SOURCE_DIR/rocwmma_qwen35_resident_layer_prepared_parallel_route_bench.hip.cpp"
COMBINED_BUILD="$SOURCE_DIR/build_rocwmma_qwen35_resident_layer_prepared_parallel_route.sh"

EXPECTED_SOURCE_SHA256="b1e60c52579da6e172404680a1f80f7f6c7df1fbfd160fd791f233906591fd4c"
EXPECTED_COMBINED_SHA256="4a9b834332c76accd0d86cd0acace19d448ba1aecdaf8ab2b90310cdad958c60"
EXPECTED_COMBINED_BUILD_SHA256="bed5483771113aceffe04fe8aa4a4b9434e08b36a4e732d1685facb42d5449fb"
EXPECTED_GENERATED_SHA256="3bc820e5574842299c21cdf0fc68349ef32fdfefd46e9ba45ef9bbc30fce61d7"
EXPECTED_DEVICE_TEXT_SHA256="d902de80d06ab66842c18fd74d03d06ce210d8ac6f44eba298bc9be814b70ef2"

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

check_sha256 profiler_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 qualified_combined "$COMBINED" "$EXPECTED_COMBINED_SHA256"
check_sha256 qualified_combined_build "$COMBINED_BUILD" \
    "$EXPECTED_COMBINED_BUILD_SHA256"

BUILD_DIR="$(mktemp -d /tmp/beholdd-q35-combined-stage-profile.XXXXXX)"
cleanup() {
    rm -rf -- "$BUILD_DIR"
}
trap cleanup EXIT

# This transitive preflight pins the compiler/runtime/HIP/rocWMMA inputs and
# verifies all qualified combined kernels/resources without executing them.
ROCM_CORE="$ROCM_CORE" ROCWMMA_INCLUDE="$ROCWMMA_INCLUDE" \
    "$COMBINED_BUILD" "$BUILD_DIR/qualified-combined-preflight"

MAIN_COUNT="$(grep -c '^int main(int argc, char\*\* argv) {$' "$COMBINED")"
if [[ "$MAIN_COUNT" != 1 ]]; then
    printf 'expected exactly one anchored combined main, found %s\n' \
        "$MAIN_COUNT" >&2
    exit 1
fi
GENERATED="$BUILD_DIR/q35_combined_components.generated.hpp"
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
HIP_OVERLAY="$BUILD_DIR/hip-root"
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
    -o "$OUTPUT"

if [[ "$(sha256sum "$SOURCE" | cut -d' ' -f1)" != "$PRE_SOURCE_SHA256" ||
      "$(sha256sum "$COMBINED" | cut -d' ' -f1)" != "$PRE_COMBINED_SHA256" ]]; then
    printf 'profiler or qualified combined source changed during build\n' >&2
    exit 1
fi
check_sha256 postcompile_profiler_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 postcompile_qualified_combined "$COMBINED" \
    "$EXPECTED_COMBINED_SHA256"

VERIFY_BINARY="$BUILD_DIR/verify-binary"
cp -- "$OUTPUT" "$VERIFY_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >/dev/null
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
if [[ ! -s "$DEVICE_OBJECT" ]]; then
    printf 'missing exact gfx1151 device image\n' >&2
    exit 1
fi
"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$BUILD_DIR/header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' "$BUILD_DIR/header.txt"; then
    printf 'profiler code object is not exact gfx1151\n' >&2
    exit 1
fi
"$OBJDUMP" -d --mcpu=gfx1151 "$DEVICE_OBJECT" >"$BUILD_DIR/isa.txt"
WMMA_COUNT="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$BUILD_DIR/isa.txt")"
if [[ "$WMMA_COUNT" != 12 ]]; then
    printf 'expected 12 qualified BF16 WMMA opcodes, found %s\n' \
        "$WMMA_COUNT" >&2
    exit 1
fi
if grep -Eq 'scratch_(load|store)|flat_scratch' "$BUILD_DIR/isa.txt"; then
    printf 'profiler device ISA contains scratch traffic\n' >&2
    exit 1
fi
"$OBJCOPY" --dump-section .text="$BUILD_DIR/device-text.bin" "$DEVICE_OBJECT"
check_sha256 device_text "$BUILD_DIR/device-text.bin" \
    "$EXPECTED_DEVICE_TEXT_SHA256"
CODE_OBJECT_SHA256="$(sha256sum "$DEVICE_OBJECT" | cut -d' ' -f1)"

if ! readelf -d "$OUTPUT" | grep -q 'libamdhip64.so.7' ||
   ! readelf -d "$OUTPUT" | grep -Fq "$ROCM_CORE/lib"; then
    printf 'profiler lacks pinned HIP runtime/runpath\n' >&2
    exit 1
fi

printf 'output=%s\n' "$OUTPUT"
printf 'code_object_sha256=%s\n' "$CODE_OBJECT_SHA256"
printf 'device_text_matches_qualified_combined=true gfx1151=true wave32=true '
printf 'private_scratch_bytes=0 native_bf16_wmma_instructions=%s\n' \
    "$WMMA_COUNT"
printf 'static_qualification=PASS gpu_execution=false\n'
