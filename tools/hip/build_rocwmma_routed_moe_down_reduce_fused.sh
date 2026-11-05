#!/usr/bin/env bash
set -euo pipefail

# Build and statically qualify the isolated frozen-control/down+route-fusion
# experiment. This script compiles and inspects only; it never runs the binary.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-rocwmma-routed-moe-down-reduce-fused-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_routed_moe_down_reduce_fused_bench.hip.cpp"
BASELINE_SOURCE="$SOURCE_DIR/rocwmma_routed_moe_bench.hip.cpp"
EXPECTED_BASELINE_SOURCE_SHA256="8f43b34f7b8542736bc467913cb2dd87d081fcc4dd24cf9a05a23b5d1c607e80"
EXPECTED_CANDIDATE_SOURCE_SHA256="f37105d5ee4ebc3ffac3022b306c16e8096fdb9bf72009bba80f47b738862417"
EXPECTED_ROCWMMA_COMMIT="bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125"
EXPECTED_ROCWMMA_HEADER_TREE_SHA256="951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750"
EXPECTED_HIP_HEADER_TREE_SHA256="ad0a71b702c2dd044dfd7fb3f2e8e8418b8e367682bbb7188816940cb1741043"
EXPECTED_RESOURCE_HEADER_TREE_SHA256="413604f0742dee20a585685ad7d52f88869f2434c0914244501619fed532fcb8"
EXPECTED_DEVICE_LIB_TREE_SHA256="e8b3b2834ddf0982dd1148f532ee0e6ef956d9759d817f4e9e5bc4f61fba9a40"
EXPECTED_DEVICE_TEXT_SHA256="f514c3a148892e07faa2fde2c6771706f3f0b3d07705c2e0f9dedc9d50e3452d"
EXPECTED_COMPILER_SHA256="1b2f3f4a10e946b3d2ac133c8f0ba4406b20b58679a3cd6d9054465849d18db0"
EXPECTED_HIP_RUNTIME_SHA256="eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05"
EXPECTED_HIP_HEADER_SHA256="0c2270e8f18150798e969daec4838ef84d0bc761394ecc933268e8eda149b37e"
EXPECTED_OCML_SHA256="a9d523fa792412ab9980ee76fb43214dc27983a07f0b23354424e9b04e91682f"
COMPILER="$ROCM_CORE/lib/llvm/bin/clang++"
RESOURCE_ROOT="$ROCM_CORE/lib/llvm/lib/clang/23"
DEVICE_LIBS="$ROCM_CORE/lib/llvm/amdgcn/bitcode"
HIP_HEADER="$ROCM_CORE/include/hip/hip_runtime_api.h"
HIP_RUNTIME="$ROCM_CORE/lib/libamdhip64.so.7"
OCML="$DEVICE_LIBS/ocml.bc"
OBJDUMP="$ROCM_CORE/lib/llvm/bin/llvm-objdump"
OBJCOPY="$ROCM_CORE/lib/llvm/bin/llvm-objcopy"
READOBJ="$ROCM_CORE/lib/llvm/bin/llvm-readobj"

for required_path in \
    "$COMPILER" \
    "$OBJDUMP" \
    "$OBJCOPY" \
    "$READOBJ" \
    "$HIP_HEADER" \
    "$HIP_RUNTIME" \
    "$RESOURCE_ROOT/include/stddef.h" \
    "$OCML" \
    "$ROCWMMA_INCLUDE/rocwmma/rocwmma.hpp" \
    "$SOURCE" \
    "$BASELINE_SOURCE"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'missing required ROCm path: %s\n' "$required_path" >&2
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

check_tree_sha256() {
    local label="$1"
    local path="$2"
    local expected="$3"
    local observed
    observed="$({
        cd "$path"
        find . -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
    } | cut -d' ' -f1)"
    if [[ "$observed" != "$expected" ]]; then
        printf '%s tree hash mismatch: expected %s, found %s\n' \
            "$label" "$expected" "$observed" >&2
        exit 1
    fi
    printf '%s_tree_sha256=%s (frozen match)\n' "$label" "$observed"
}

printf 'compiler:\n'
"$COMPILER" --version | sed -n '1,2p'
check_sha256 compiler "$COMPILER" "$EXPECTED_COMPILER_SHA256"
check_sha256 hip_runtime "$HIP_RUNTIME" "$EXPECTED_HIP_RUNTIME_SHA256"
check_sha256 hip_runtime_api_header "$HIP_HEADER" "$EXPECTED_HIP_HEADER_SHA256"
check_sha256 ocml "$OCML" "$EXPECTED_OCML_SHA256"
check_tree_sha256 hip_headers "$ROCM_CORE/include/hip" \
    "$EXPECTED_HIP_HEADER_TREE_SHA256"
check_tree_sha256 clang_resource_headers "$RESOURCE_ROOT/include" \
    "$EXPECTED_RESOURCE_HEADER_TREE_SHA256"
check_tree_sha256 device_libraries "$DEVICE_LIBS" \
    "$EXPECTED_DEVICE_LIB_TREE_SHA256"

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
check_sha256 baseline_source "$BASELINE_SOURCE" \
    "$EXPECTED_BASELINE_SOURCE_SHA256"
check_sha256 candidate_source "$SOURCE" \
    "$EXPECTED_CANDIDATE_SOURCE_SHA256"
PRECOMPILE_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"

# The wheel has only libamdhip64.so.7. Supply the linker's unversioned name in
# a disposable HIP root; never consult or modify the system ROCm 5.7 install.
HIP_OVERLAY="$(mktemp -d /tmp/beholdd-hip-down-fused-root.XXXXXX)"
cleanup() {
    rm -rf -- "$HIP_OVERLAY"
}
trap cleanup EXIT
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

POSTCOMPILE_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
if [[ "$POSTCOMPILE_SOURCE_SHA256" != "$PRECOMPILE_SOURCE_SHA256" ]]; then
    printf 'candidate source changed during compilation: before %s, after %s\n' \
        "$PRECOMPILE_SOURCE_SHA256" "$POSTCOMPILE_SOURCE_SHA256" >&2
    exit 1
fi
printf 'candidate_source_postcompile_sha256=%s (unchanged)\n' \
    "$POSTCOMPILE_SOURCE_SHA256"

# Recheck every frozen transitive input after compilation. This prevents a
# concurrent baseline, header, compiler, runtime, or device-library change from
# producing evidence that only matched at the beginning of the build.
check_sha256 postcompile_compiler "$COMPILER" "$EXPECTED_COMPILER_SHA256"
check_sha256 postcompile_hip_runtime "$HIP_RUNTIME" \
    "$EXPECTED_HIP_RUNTIME_SHA256"
check_sha256 postcompile_hip_runtime_api_header "$HIP_HEADER" \
    "$EXPECTED_HIP_HEADER_SHA256"
check_sha256 postcompile_ocml "$OCML" "$EXPECTED_OCML_SHA256"
check_sha256 postcompile_baseline_source "$BASELINE_SOURCE" \
    "$EXPECTED_BASELINE_SOURCE_SHA256"
check_tree_sha256 postcompile_hip_headers "$ROCM_CORE/include/hip" \
    "$EXPECTED_HIP_HEADER_TREE_SHA256"
check_tree_sha256 postcompile_clang_resource_headers \
    "$RESOURCE_ROOT/include" "$EXPECTED_RESOURCE_HEADER_TREE_SHA256"
check_tree_sha256 postcompile_device_libraries "$DEVICE_LIBS" \
    "$EXPECTED_DEVICE_LIB_TREE_SHA256"
POSTCOMPILE_ROCWMMA_COMMIT="$(git -C "$ROCWMMA_INCLUDE" rev-parse HEAD 2>/dev/null || printf 'not-a-git-checkout')"
POSTCOMPILE_ROCWMMA_STATUS="$(git -C "$ROCWMMA_REPO_ROOT" status --porcelain=v1 --untracked-files=all -- "$ROCWMMA_INCLUDE_RELATIVE")"
POSTCOMPILE_ROCWMMA_HEADER_TREE_SHA256="$({
    cd "$ROCWMMA_INCLUDE"
    find rocwmma -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
} | cut -d' ' -f1)"
if [[ "$POSTCOMPILE_ROCWMMA_COMMIT" != "$EXPECTED_ROCWMMA_COMMIT" ||
      -n "$POSTCOMPILE_ROCWMMA_STATUS" ||
      "$POSTCOMPILE_ROCWMMA_HEADER_TREE_SHA256" != "$EXPECTED_ROCWMMA_HEADER_TREE_SHA256" ]]; then
    printf 'rocWMMA checkout changed during compilation: commit=%s status=%s headers=%s\n' \
        "$POSTCOMPILE_ROCWMMA_COMMIT" "$POSTCOMPILE_ROCWMMA_STATUS" \
        "$POSTCOMPILE_ROCWMMA_HEADER_TREE_SHA256" >&2
    exit 1
fi
printf 'postcompile_rocwmma_commit=%s (unchanged)\n' \
    "$POSTCOMPILE_ROCWMMA_COMMIT"
printf 'postcompile_rocwmma_header_tree_sha256=%s (clean unchanged)\n' \
    "$POSTCOMPILE_ROCWMMA_HEADER_TREE_SHA256"

VERIFY_BINARY="$HIP_OVERLAY/verify-binary"
cp -- "$OUTPUT" "$VERIFY_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >/dev/null
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
if [[ ! -s "$DEVICE_OBJECT" ]]; then
    printf 'missing gfx1151 device image in %s\n' "$OUTPUT" >&2
    exit 1
fi

"$OBJCOPY" --dump-section .text="$HIP_OVERLAY/device-text.bin" \
    "$DEVICE_OBJECT"
check_sha256 device_text "$HIP_OVERLAY/device-text.bin" \
    "$EXPECTED_DEVICE_TEXT_SHA256"

"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' "$HIP_OVERLAY/device-header.txt"; then
    printf 'embedded code object is not tagged for gfx1151\n' >&2
    exit 1
fi

"$READOBJ" --notes "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-notes.txt"
awk '
function finish_kernel() {
    if (name == "") {
        return
    }
    kernel_count++
    expected = 0
    control_wmma = 0
    candidate = 0
    if (name ~ /f32_to_bf16_padded/) { f32_count++; expected = 1 }
    if (name ~ /gate_up_wmma/) { gate_count++; expected = 1; control_wmma = 1 }
    if (name ~ /silu_multiply_to_bf16/) { silu_count++; expected = 1 }
    if (name ~ /GLOBAL__N_19down_wmma/) { down_count++; expected = 1; control_wmma = 1 }
    if (name ~ /weighted_route_reduce/) { reduce_count++; expected = 1 }
    if (name == "fused_down_route_wmma") {
        fused_count++
        expected = 1
        candidate = 1
    }
    if (!expected) {
        print "unexpected GPU kernel in down-fusion experiment: " name > "/dev/stderr"
        failed = 1
    }
    missing = (vgpr == "" || private_bytes == "" || lds_bytes == "" ||
               sgpr_spills == "" || vgpr_spills == "" || wavefront == "" ||
               max_workgroup == "" || dynamic_stack == "")
    if (missing) {
        print "incomplete resource metadata for " name > "/dev/stderr"
        failed = 1
        return
    }
    printf "kernel_resource: name=%s vgpr=%s private_bytes=%s lds_bytes=%s sgpr_spills=%s vgpr_spills=%s wavefront=%s max_workgroup=%s dynamic_stack=%s\n", \
           name, vgpr, private_bytes, lds_bytes, sgpr_spills, vgpr_spills, \
           wavefront, max_workgroup, dynamic_stack
    if (private_bytes + 0 != 0 || sgpr_spills + 0 != 0 || vgpr_spills + 0 != 0) {
        print "kernel spills or allocates private scratch: " name > "/dev/stderr"
        failed = 1
    }
    if (wavefront + 0 != 32 || dynamic_stack != "false") {
        print "kernel must remain wave32 without a dynamic stack: " name > "/dev/stderr"
        failed = 1
    }
    if (candidate) {
        if (lds_bytes + 0 != 8192 || max_workgroup + 0 != 256 || vgpr + 0 > 64) {
            print "fused kernel violates 8KiB/256-thread/64-VGPR envelope" > "/dev/stderr"
            failed = 1
        }
    } else {
        if (lds_bytes + 0 != 0) {
            print "frozen control unexpectedly uses LDS: " name > "/dev/stderr"
            failed = 1
        }
        if (control_wmma && (max_workgroup + 0 != 32 || vgpr + 0 > 64)) {
            print "frozen WMMA resource envelope changed: " name > "/dev/stderr"
            failed = 1
        }
        if (!control_wmma && max_workgroup + 0 != 1024) {
            print "frozen elementwise workgroup envelope changed: " name > "/dev/stderr"
            failed = 1
        }
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
    if (kernel_count != 6 || f32_count != 1 || gate_count != 1 ||
        silu_count != 1 || down_count != 1 || reduce_count != 1 ||
        fused_count != 1) {
        printf "expected six kernels; total=%d f32=%d gate=%d silu=%d down=%d reduce=%d fused=%d\n", \
               kernel_count, f32_count, gate_count, silu_count, down_count, \
               reduce_count, fused_count > "/dev/stderr"
        failed = 1
    }
    exit failed
}
' "$HIP_OVERLAY/device-notes.txt" >"$HIP_OVERLAY/resource-report.txt"

"$OBJDUMP" -d --mcpu=gfx1151 \
    --disassemble-symbols=fused_down_route_wmma \
    "$DEVICE_OBJECT" >"$HIP_OVERLAY/fused-isa.txt"
FUSED_WMMA_COUNT="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$HIP_OVERLAY/fused-isa.txt" || true)"
FUSED_BARRIER_COUNT="$(grep -Ec '(^|[[:space:]])s_barrier([[:space:]]|$)' "$HIP_OVERLAY/fused-isa.txt" || true)"
FUSED_LDS_STORE_COUNT="$(grep -Ec '(^|[[:space:]])ds_store_' "$HIP_OVERLAY/fused-isa.txt" || true)"
FUSED_LDS_LOAD_COUNT="$(grep -Ec '(^|[[:space:]])ds_load_' "$HIP_OVERLAY/fused-isa.txt" || true)"
if ((FUSED_WMMA_COUNT != 2)); then
    printf 'fused kernel must contain exactly two native BF16 WMMA opcodes; found %s\n' \
        "$FUSED_WMMA_COUNT" >&2
    exit 1
fi
if ((FUSED_BARRIER_COUNT != 1)); then
    printf 'fused kernel must contain exactly one workgroup barrier; found %s\n' \
        "$FUSED_BARRIER_COUNT" >&2
    exit 1
fi
if ((FUSED_LDS_STORE_COUNT == 0 || FUSED_LDS_LOAD_COUNT == 0)); then
    printf 'fused kernel must contain both LDS stores and LDS loads; stores=%s loads=%s\n' \
        "$FUSED_LDS_STORE_COUNT" "$FUSED_LDS_LOAD_COUNT" >&2
    exit 1
fi

HIP_RESOLUTION="$(LD_LIBRARY_PATH="$ROCM_CORE/lib" ldd "$OUTPUT" | grep 'libamdhip64.so.7' || true)"
if [[ "$HIP_RESOLUTION" != *"$HIP_RUNTIME"* ]]; then
    printf 'binary did not resolve the frozen ROCm wheel libamdhip64: %s\n' \
        "$HIP_RESOLUTION" >&2
    exit 1
fi

CODE_OBJECT_SHA256="$(sha256sum "$DEVICE_OBJECT" | cut -d' ' -f1)"
printf 'built=%s\n' "$OUTPUT"
printf 'code_object_target=EF_AMDGPU_MACH_AMDGCN_GFX1151\n'
printf 'code_object_sha256=%s\n' "$CODE_OBJECT_SHA256"
printf 'device_text_sha256=%s (deterministic frozen payload)\n' \
    "$EXPECTED_DEVICE_TEXT_SHA256"
printf 'fused_native_wmma_instruction_count=%s\n' "$FUSED_WMMA_COUNT"
printf 'fused_barrier_count=%s\n' "$FUSED_BARRIER_COUNT"
printf 'fused_lds_instruction_counts=stores:%s,loads:%s\n' \
    "$FUSED_LDS_STORE_COUNT" "$FUSED_LDS_LOAD_COUNT"
printf 'device_kernel_count=6 (five frozen control plus one candidate)\n'
cat "$HIP_OVERLAY/resource-report.txt"
printf 'libamdhip64_resolution=%s\n' "$HIP_RESOLUTION"
