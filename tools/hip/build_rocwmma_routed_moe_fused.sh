#!/usr/bin/env bash
set -euo pipefail

# Build and statically qualify the paired frozen-control/fused-candidate
# gfx1151 experiment. Invoke with bash; this file need not be executable.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-rocwmma-routed-moe-fused-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_routed_moe_fused_bench.hip.cpp"
BASELINE_SOURCE="$SOURCE_DIR/rocwmma_routed_moe_bench.hip.cpp"
EXPECTED_BASELINE_SOURCE_SHA256="8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80"
EXPECTED_ROCWMMA_COMMIT="bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125"
EXPECTED_ROCWMMA_HEADER_TREE_SHA256="951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750"
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
    "$ROCWMMA_INCLUDE/rocwmma/rocwmma.hpp" \
    "$SOURCE" \
    "$BASELINE_SOURCE"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'missing required ROCm path: %s\n' "$required_path" >&2
        exit 1
    fi
done

printf 'compiler:\n'
"$COMPILER" --version | sed -n '1,2p'
ROCWMMA_COMMIT="$(git -C "$ROCWMMA_INCLUDE" rev-parse HEAD 2>/dev/null || printf 'not-a-git-checkout')"
if [[ "$ROCWMMA_COMMIT" != "$EXPECTED_ROCWMMA_COMMIT" ]]; then
    printf 'rocWMMA commit mismatch: expected %s, found %s\n' \
        "$EXPECTED_ROCWMMA_COMMIT" "$ROCWMMA_COMMIT" >&2
    exit 1
fi
ROCWMMA_REPO_ROOT="$(git -C "$ROCWMMA_INCLUDE" rev-parse --show-toplevel)"
ROCWMMA_INCLUDE_RELATIVE="$(realpath --relative-to="$ROCWMMA_REPO_ROOT" "$ROCWMMA_INCLUDE")"
ROCWMMA_HEADER_STATUS="$(git -C "$ROCWMMA_REPO_ROOT" status --porcelain=v1 --untracked-files=all -- "$ROCWMMA_INCLUDE_RELATIVE")"
if [[ -n "$ROCWMMA_HEADER_STATUS" ]]; then
    printf 'rocWMMA header tree has tracked or untracked changes:\n%s\n' \
        "$ROCWMMA_HEADER_STATUS" >&2
    exit 1
fi
ROCWMMA_HEADER_TREE_SHA256="$({
    cd "$ROCWMMA_INCLUDE"
    find rocwmma -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
} | cut -d' ' -f1)"
if [[ "$ROCWMMA_HEADER_TREE_SHA256" != "$EXPECTED_ROCWMMA_HEADER_TREE_SHA256" ]]; then
    printf 'rocWMMA header hash mismatch: expected %s, found %s\n' \
        "$EXPECTED_ROCWMMA_HEADER_TREE_SHA256" \
        "$ROCWMMA_HEADER_TREE_SHA256" >&2
    exit 1
fi
printf 'rocwmma_include=%s\nrocwmma_commit=%s\n' \
    "$ROCWMMA_INCLUDE" "$ROCWMMA_COMMIT"
printf 'rocwmma_header_tree_sha256=%s (clean frozen match)\n' \
    "$ROCWMMA_HEADER_TREE_SHA256"
BASELINE_SOURCE_SHA256="$(sha256sum "$BASELINE_SOURCE" | cut -d' ' -f1)"
if [[ "$BASELINE_SOURCE_SHA256" != "$EXPECTED_BASELINE_SOURCE_SHA256" ]]; then
    printf 'frozen baseline source hash changed: expected %s, found %s\n' \
        "$EXPECTED_BASELINE_SOURCE_SHA256" "$BASELINE_SOURCE_SHA256" >&2
    exit 1
fi
printf 'baseline_source_sha256=%s (frozen match)\n' "$BASELINE_SOURCE_SHA256"
printf 'candidate_source_sha256=%s\n' \
    "$(sha256sum "$SOURCE" | cut -d' ' -f1)"

# The wheel has only the versioned HIP library. Supply the linker's unversioned
# name through a disposable HIP root without modifying the wheel or consulting
# Ubuntu's system ROCm 5.7 installation.
HIP_OVERLAY="$(mktemp -d /tmp/beholdd-hip-fused-root.XXXXXX)"
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

# Extract the device image from a disposable copy and enforce the candidate's
# predeclared resource envelope from AMDGPU metadata.
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

"$READOBJ" --notes "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-notes.txt"
awk '
function finish_kernel() {
    if (name != "fused_gate_up_silu_wmma") {
        return
    }
    found = 1
    missing = (vgpr == "" || private_bytes == "" || lds_bytes == "" ||
               sgpr_spills == "" || vgpr_spills == "" || wavefront == "" ||
               max_workgroup == "" || dynamic_stack == "")
    if (missing) {
        print "incomplete fused-kernel resource metadata" > "/dev/stderr"
        failed = 1
        return
    }
    printf "fused_resource: vgpr=%s private_bytes=%s lds_bytes=%s sgpr_spills=%s vgpr_spills=%s wavefront=%s max_workgroup=%s dynamic_stack=%s\n", \
           vgpr, private_bytes, lds_bytes, sgpr_spills, vgpr_spills, \
           wavefront, max_workgroup, dynamic_stack
    if (vgpr + 0 > 64) {
        print "fused kernel exceeds 64 VGPRs" > "/dev/stderr"
        failed = 1
    }
    if (private_bytes + 0 != 0 || sgpr_spills + 0 != 0 || vgpr_spills + 0 != 0) {
        print "fused kernel spills or allocates private scratch" > "/dev/stderr"
        failed = 1
    }
    if (lds_bytes + 0 != 2048) {
        print "fused kernel must use exactly 2048 bytes of LDS" > "/dev/stderr"
        failed = 1
    }
    if (wavefront + 0 != 32 || max_workgroup + 0 != 32) {
        print "fused kernel must remain wave32 with a 32-thread maximum workgroup" > "/dev/stderr"
        failed = 1
    }
    if (dynamic_stack != "false") {
        print "fused kernel must not use a dynamic stack" > "/dev/stderr"
        failed = 1
    }
}
/^[[:space:]]+- \.args:/ {
    finish_kernel()
    name = ""
    vgpr = private_bytes = lds_bytes = sgpr_spills = vgpr_spills = ""
    wavefront = max_workgroup = dynamic_stack = ""
    next
}
/^[[:space:]]+\.name:/ { name = $2; next }
/^[[:space:]]+\.vgpr_count:/ { vgpr = $2; next }
/^[[:space:]]+\.private_segment_fixed_size:/ { private_bytes = $2; next }
/^[[:space:]]+\.group_segment_fixed_size:/ { lds_bytes = $2; next }
/^[[:space:]]+\.sgpr_spill_count:/ { sgpr_spills = $2; next }
/^[[:space:]]+\.vgpr_spill_count:/ { vgpr_spills = $2; next }
/^[[:space:]]+\.wavefront_size:/ { wavefront = $2; next }
/^[[:space:]]+\.max_flat_workgroup_size:/ { max_workgroup = $2; next }
/^[[:space:]]+\.uses_dynamic_stack:/ { dynamic_stack = $2; next }
END {
    finish_kernel()
    if (!found) {
        print "fused_gate_up_silu_wmma metadata not found" > "/dev/stderr"
        failed = 1
    }
    exit failed
}
' "$HIP_OVERLAY/device-notes.txt" >"$HIP_OVERLAY/resource-report.txt"

"$OBJDUMP" -d --mcpu=gfx1151 \
    --disassemble-symbols=fused_gate_up_silu_wmma \
    "$DEVICE_OBJECT" >"$HIP_OVERLAY/fused-isa.txt"
WMMA_INSTRUCTION_COUNT="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$HIP_OVERLAY/fused-isa.txt" || true)"
if ((WMMA_INSTRUCTION_COUNT != 2)); then
    printf 'fused kernel must contain exactly two native BF16 WMMA operations; found %s\n' \
        "$WMMA_INSTRUCTION_COUNT" >&2
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
printf 'fused_native_wmma_instruction_count=%s\n' "$WMMA_INSTRUCTION_COUNT"
cat "$HIP_OVERLAY/resource-report.txt"
printf 'libamdhip64_resolution=%s\n' "$HIP_RESOLUTION"
printf 'libamdhip64_sha256=%s\n' "$HIP_SHA256"
