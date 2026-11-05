#!/usr/bin/env bash
set -euo pipefail

# Build the isolated gfx1151 rocWMMA experiment against an explicitly selected
# ROCm SDK. Invoke with bash; this file need not be executable.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-rocwmma-routed-moe-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_routed_moe_bench.hip.cpp"
COMPILER="$ROCM_CORE/lib/llvm/bin/clang++"
RESOURCE_ROOT="$ROCM_CORE/lib/llvm/lib/clang/23"
DEVICE_LIBS="$ROCM_CORE/lib/llvm/amdgcn/bitcode"
OBJDUMP="$ROCM_CORE/lib/llvm/bin/llvm-objdump"
READOBJ="$ROCM_CORE/lib/llvm/bin/llvm-readobj"

for required_path in \
    "$COMPILER" \
    "$OBJDUMP" \
    "$READOBJ" \
    "$ROCM_CORE/include/hip/hip_runtime.h" \
    "$ROCM_CORE/lib/libamdhip64.so.7" \
    "$RESOURCE_ROOT/include/stddef.h" \
    "$DEVICE_LIBS/ocml.bc" \
    "$ROCWMMA_INCLUDE/rocwmma/rocwmma.hpp"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'missing required ROCm path: %s\n' "$required_path" >&2
        exit 1
    fi
done

printf 'compiler:\n'
"$COMPILER" --version | sed -n '1,2p'
ROCWMMA_COMMIT="$(git -C "$ROCWMMA_INCLUDE" rev-parse HEAD 2>/dev/null || printf 'not-a-git-checkout')"
printf 'rocwmma_include=%s\nrocwmma_commit=%s\n' \
    "$ROCWMMA_INCLUDE" "$ROCWMMA_COMMIT"

# The Python ROCm wheel intentionally ships only libamdhip64.so.7, while
# clang's HIP link step asks for libamdhip64.so. Create the missing linker name
# in a disposable HIP root rather than modifying the venv or using Ubuntu's
# system ROCm 5.7 installation.
HIP_OVERLAY="$(mktemp -d /tmp/beholdd-hip-root.XXXXXX)"
cleanup() {
    rm -rf -- "$HIP_OVERLAY"
}
trap cleanup EXIT
mkdir -p "$HIP_OVERLAY/lib"
ln -s "$ROCM_CORE/lib/libamdhip64.so.7" "$HIP_OVERLAY/lib/libamdhip64.so"

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

# Extract a copy inside the disposable directory and prove that the bundled
# device image has the requested machine flag and native BF16 WMMA opcode.
VERIFY_BINARY="$HIP_OVERLAY/verify-binary"
cp -- "$OUTPUT" "$VERIFY_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >/dev/null
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
if [[ ! -s "$DEVICE_OBJECT" ]]; then
    printf 'missing gfx1151 device image in %s\n' "$OUTPUT" >&2
    exit 1
fi
"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' "$HIP_OVERLAY/device-header.txt"; then
    printf 'embedded code object is not tagged for gfx1151\n' >&2
    exit 1
fi
"$OBJDUMP" -d --mcpu=gfx1151 "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-isa.txt"
if ! grep -q 'v_wmma_f32_16x16x16_bf16' "$HIP_OVERLAY/device-isa.txt"; then
    printf 'native gfx1151 BF16 WMMA instruction not found\n' >&2
    exit 1
fi
CODE_OBJECT_SHA256="$(sha256sum "$DEVICE_OBJECT" | cut -d' ' -f1)"

HIP_RESOLUTION="$(LD_LIBRARY_PATH="$ROCM_CORE/lib" ldd "$OUTPUT" | grep 'libamdhip64.so.7' || true)"
if [[ "$HIP_RESOLUTION" != *"$ROCM_CORE/lib/libamdhip64.so.7"* ]]; then
    printf 'binary did not resolve the ROCm 7.13 wheel libamdhip64: %s\n' \
        "$HIP_RESOLUTION" >&2
    exit 1
fi
HIP_SHA256="$(sha256sum "$ROCM_CORE/lib/libamdhip64.so.7" | cut -d' ' -f1)"

printf 'built=%s\n' "$OUTPUT"
printf 'code_object_target=EF_AMDGPU_MACH_AMDGCN_GFX1151\n'
printf 'code_object_sha256=%s\n' "$CODE_OBJECT_SHA256"
printf 'native_instruction=v_wmma_f32_16x16x16_bf16\n'
printf 'libamdhip64_resolution=%s\n' "$HIP_RESOLUTION"
printf 'libamdhip64_sha256=%s\n' "$HIP_SHA256"
