#!/usr/bin/env bash
set -euo pipefail

# Compile and statically qualify the safety-patched batch-1 BF16 vector-dot
# experiment and its included, validated WMMA control. This script never runs the
# resulting GPU binary.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-q35-resident-layer-vector-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_vector_bench.hip.cpp"
CONTROL_SOURCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_bench.hip.cpp"

EXPECTED_SOURCE_SHA256="3fe2d9a89253349e78d452002c9d58b234a9bb60d7d142611077ff78477a1dad"
EXPECTED_CONTROL_SOURCE_SHA256="dfed8f57bc310e1c8f366a13601ab9dccdd88da82744d5980e8370e858d1f3dd"
EXPECTED_ROCWMMA_COMMIT="bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125"
EXPECTED_ROCWMMA_HEADER_TREE_SHA256="951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750"
EXPECTED_HIP_HEADER_TREE_SHA256="ad0a71b702c2dd044dfd7fb3f2e8e8418b8e367682bbb7188816940cb1741043"
EXPECTED_RESOURCE_HEADER_TREE_SHA256="413604f0742dee20a585685ad7d52f88869f2434c0914244501619fed532fcb8"
EXPECTED_DEVICE_LIB_TREE_SHA256="e8b3b2834ddf0982dd1148f532ee0e6ef956d9759d817f4e9e5bc4f61fba9a40"
EXPECTED_COMPILER_SHA256="1b2f3f4a10e946b3d2ac133c8f0ba4406b20b58679a3cd6d9054465849d18db0"
EXPECTED_OBJDUMP_SHA256="138b68a36d21e2b6a2d1a84dbaa23280db62696750b4512ec26961ab871b67bb"
EXPECTED_OBJCOPY_SHA256="be45b02a63ac152a1aed731f3f4ff206801cf7c76e5fe6bef08e271cd8776bdc"
EXPECTED_READOBJ_SHA256="915894309efc62f7a56bfcd1d29231b5ebfa3740adeb53a1164c236d66e80fdd"
EXPECTED_READELF_SHA256="8e008f8871b2f79dd433ece6a4fb714b0f8a5fd9607a6c3030972e28b2b4ce88"
EXPECTED_HIP_RUNTIME_SHA256="eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05"
EXPECTED_HIP_HEADER_SHA256="0c2270e8f18150798e969daec4838ef84d0bc761394ecc933268e8eda149b37e"
EXPECTED_OCML_SHA256="a9d523fa792412ab9980ee76fb43214dc27983a07f0b23354424e9b04e91682f"
EXPECTED_DEVICE_TEXT_SHA256="8522c5fc9e2c5ade38341e40c054774785b9e0837a8cf55578dd66bffbd80c9a"

COMPILER="$ROCM_CORE/lib/llvm/bin/clang++"
RESOURCE_ROOT="$ROCM_CORE/lib/llvm/lib/clang/23"
DEVICE_LIBS="$ROCM_CORE/lib/llvm/amdgcn/bitcode"
HIP_HEADER="$ROCM_CORE/include/hip/hip_runtime_api.h"
HIP_RUNTIME="$ROCM_CORE/lib/libamdhip64.so.7"
OCML="$DEVICE_LIBS/ocml.bc"
OBJDUMP="$ROCM_CORE/lib/llvm/bin/llvm-objdump"
OBJCOPY="$ROCM_CORE/lib/llvm/bin/llvm-objcopy"
READOBJ="$ROCM_CORE/lib/llvm/bin/llvm-readobj"
READELF="$ROCM_CORE/lib/llvm/bin/llvm-readelf"

for required_path in \
    "$COMPILER" \
    "$OBJDUMP" \
    "$OBJCOPY" \
    "$READOBJ" \
    "$READELF" \
    "$HIP_HEADER" \
    "$HIP_RUNTIME" \
    "$RESOURCE_ROOT/include/stddef.h" \
    "$OCML" \
    "$ROCWMMA_INCLUDE/rocwmma/rocwmma.hpp" \
    "$SOURCE" \
    "$CONTROL_SOURCE"; do
    if [[ ! -e "$required_path" ]]; then
        printf 'missing required path: %s\n' "$required_path" >&2
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

"$COMPILER" --version | sed -n '1,2p'
check_sha256 source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 control_source "$CONTROL_SOURCE" \
    "$EXPECTED_CONTROL_SOURCE_SHA256"
check_sha256 compiler "$COMPILER" "$EXPECTED_COMPILER_SHA256"
check_sha256 llvm_objdump "$OBJDUMP" "$EXPECTED_OBJDUMP_SHA256"
check_sha256 llvm_objcopy "$OBJCOPY" "$EXPECTED_OBJCOPY_SHA256"
check_sha256 llvm_readobj "$READOBJ" "$EXPECTED_READOBJ_SHA256"
check_sha256 llvm_readelf "$READELF" "$EXPECTED_READELF_SHA256"
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
ROCWMMA_ROOT="$(git -C "$ROCWMMA_INCLUDE" rev-parse --show-toplevel)"
ROCWMMA_RELATIVE="$(realpath --relative-to="$ROCWMMA_ROOT" "$ROCWMMA_INCLUDE")"
ROCWMMA_STATUS="$(git -C "$ROCWMMA_ROOT" status --porcelain=v1 --untracked-files=all -- "$ROCWMMA_RELATIVE")"
if [[ -n "$ROCWMMA_STATUS" ]]; then
    printf 'rocWMMA header checkout is dirty:\n%s\n' "$ROCWMMA_STATUS" >&2
    exit 1
fi
ROCWMMA_TREE="$({
    cd "$ROCWMMA_INCLUDE"
    find rocwmma -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum
} | cut -d' ' -f1)"
if [[ "$ROCWMMA_TREE" != "$EXPECTED_ROCWMMA_HEADER_TREE_SHA256" ]]; then
    printf 'rocWMMA header hash mismatch: expected %s, found %s\n' \
        "$EXPECTED_ROCWMMA_HEADER_TREE_SHA256" "$ROCWMMA_TREE" >&2
    exit 1
fi
printf 'rocwmma_commit=%s rocwmma_header_tree_sha256=%s (clean frozen match)\n' \
    "$ROCWMMA_COMMIT" "$ROCWMMA_TREE"

PRECOMPILE_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
PRECOMPILE_CONTROL_SHA256="$(sha256sum "$CONTROL_SOURCE" | cut -d' ' -f1)"
HIP_OVERLAY="$(mktemp -d /tmp/beholdd-q35-vector-hip-root.XXXXXX)"
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
POSTCOMPILE_CONTROL_SHA256="$(sha256sum "$CONTROL_SOURCE" | cut -d' ' -f1)"
if [[ "$POSTCOMPILE_SOURCE_SHA256" != "$PRECOMPILE_SOURCE_SHA256" ]]; then
    printf 'candidate source changed during compilation\n' >&2
    exit 1
fi
if [[ "$POSTCOMPILE_CONTROL_SHA256" != "$PRECOMPILE_CONTROL_SHA256" ]]; then
    printf 'control source changed during compilation\n' >&2
    exit 1
fi
check_sha256 postcompile_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 postcompile_control_source "$CONTROL_SOURCE" \
    "$EXPECTED_CONTROL_SOURCE_SHA256"

VERIFY_BINARY="$HIP_OVERLAY/verify-binary"
cp -- "$OUTPUT" "$VERIFY_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >/dev/null
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
shopt -s nullglob
DEVICE_IMAGES=("$VERIFY_BINARY".*.hipv4-amdgcn-amd-amdhsa--*)
shopt -u nullglob
if [[ "${#DEVICE_IMAGES[@]}" -ne 1 || "${DEVICE_IMAGES[0]}" != "$DEVICE_OBJECT" || \
      ! -s "$DEVICE_OBJECT" ]]; then
    printf 'binary must contain exactly one gfx1151 device image\n' >&2
    printf 'device_image=%s\n' "${DEVICE_IMAGES[@]:-<none>}" >&2
    exit 1
fi

"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' \
    "$HIP_OVERLAY/device-header.txt"; then
    printf 'embedded code object is not exact gfx1151\n' >&2
    exit 1
fi
"$READOBJ" --notes "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-notes.txt"

awk '
    BEGIN {
        required["q35_rmsnorm_f32_to_f32_bf16"] = 1
        required["q35_router_and_shared_gate_f32"] = 1
        required["q35_stable_softmax_topk8"] = 1
        required["q35_selected_shared_gate_up_wmma"] = 1
        required["q35_silu_multiply_to_bf16"] = 1
        required["q35_selected_shared_down_wmma"] = 1
        required["q35_route_shared_residual_reduce"] = 1
        required["q35_vector_rmsnorm_f32_to_f32_bf16"] = 1
        required["q35_selected_shared_gate_up_vector_dot"] = 1
        required["q35_vector_silu_multiply_to_bf16"] = 1
        required["q35_selected_shared_down_vector_dot"] = 1
        required["q35_vector_route_shared_residual_reduce"] = 1
        single_wave["q35_selected_shared_gate_up_wmma"] = 1
        single_wave["q35_selected_shared_down_wmma"] = 1
        single_wave["q35_selected_shared_gate_up_vector_dot"] = 1
        single_wave["q35_selected_shared_down_vector_dot"] = 1
    }
    /\.max_flat_workgroup_size:/ { maximum = $2 }
    /\.name:/ {
        name = $2
        active = name in required
        if (active) {
            seen[name] = 1
            if ((name in single_wave) && maximum != 32) {
                print "single-wave kernel maximum workgroup is not 32: " name \
                    " maximum=" maximum > "/dev/stderr"
                failed = 1
            }
        }
    }
    active && /\.private_segment_fixed_size:/ && $2 != 0 {
        print "kernel private segment is nonzero: " name " bytes=" $2 > "/dev/stderr"
        failed = 1
    }
    active && /\.sgpr_spill_count:/ && $2 != 0 {
        print "kernel has SGPR spills: " name " count=" $2 > "/dev/stderr"
        failed = 1
    }
    active && /\.vgpr_spill_count:/ && $2 != 0 {
        print "kernel has VGPR spills: " name " count=" $2 > "/dev/stderr"
        failed = 1
    }
    active && /\.wavefront_size:/ && $2 != 32 {
        print "kernel is not wave32: " name " size=" $2 > "/dev/stderr"
        failed = 1
    }
    END {
        for (name in required) {
            if (!seen[name]) {
                print "missing kernel metadata: " name > "/dev/stderr"
                failed = 1
            }
        }
        exit failed
    }
' "$HIP_OVERLAY/device-notes.txt"

"$OBJDUMP" -d --mcpu=gfx1151 "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-isa.txt"
if grep -Eq 'scratch_(load|store)|flat_scratch' "$HIP_OVERLAY/device-isa.txt"; then
    printf 'device ISA contains scratch traffic\n' >&2
    exit 1
fi

extract_symbol() {
    local symbol="$1"
    local destination="$2"
    if ! awk -v wanted="$symbol" '
        /^[[:xdigit:]]+ <[^>]+>:/ {
            if (active && index($0, "<" wanted ">:") == 0) {
                exit
            }
            if (index($0, "<" wanted ">:") != 0) {
                active = 1
                found = 1
            }
        }
        active { print }
        END { if (!found) exit 1 }
    ' "$HIP_OVERLAY/device-isa.txt" >"$destination"; then
        printf 'missing disassembly for symbol %s\n' "$symbol" >&2
        exit 1
    fi
}

GATE_UP_VECTOR_ISA="$HIP_OVERLAY/gate-up-vector.isa"
DOWN_VECTOR_ISA="$HIP_OVERLAY/down-vector.isa"
GATE_UP_CONTROL_ISA="$HIP_OVERLAY/gate-up-control.isa"
DOWN_CONTROL_ISA="$HIP_OVERLAY/down-control.isa"
extract_symbol q35_selected_shared_gate_up_vector_dot "$GATE_UP_VECTOR_ISA"
extract_symbol q35_selected_shared_down_vector_dot "$DOWN_VECTOR_ISA"
extract_symbol q35_selected_shared_gate_up_wmma "$GATE_UP_CONTROL_ISA"
extract_symbol q35_selected_shared_down_wmma "$DOWN_CONTROL_ISA"

for vector_isa in "$GATE_UP_VECTOR_ISA" "$DOWN_VECTOR_ISA"; do
    if ! grep -q 'v_dot2_f32_bf16' "$vector_isa"; then
        printf 'candidate symbol lacks native BF16 dot2: %s\n' "$vector_isa" >&2
        exit 1
    fi
    if grep -q 'v_wmma_' "$vector_isa"; then
        printf 'candidate symbol unexpectedly contains WMMA: %s\n' "$vector_isa" >&2
        exit 1
    fi
    if grep -Eq 'scratch_(load|store)|flat_scratch' "$vector_isa"; then
        printf 'candidate symbol contains scratch traffic: %s\n' "$vector_isa" >&2
        exit 1
    fi
done

for control_isa in "$GATE_UP_CONTROL_ISA" "$DOWN_CONTROL_ISA"; do
    if ! grep -q 'v_wmma_f32_16x16x16_bf16' "$control_isa"; then
        printf 'included control symbol lacks native BF16 WMMA: %s\n' \
            "$control_isa" >&2
        exit 1
    fi
done

DOT2_COUNT="$(grep -h -c 'v_dot2_f32_bf16' \
    "$GATE_UP_VECTOR_ISA" "$DOWN_VECTOR_ISA" | awk '{ total += $1 } END { print total + 0 }')"
if [[ "$DOT2_COUNT" -ne 2 ]]; then
    printf 'expected exactly two static native BF16 dot2 sites, found %s\n' \
        "$DOT2_COUNT" >&2
    exit 1
fi

"$OBJCOPY" --dump-section .text="$HIP_OVERLAY/device-text.bin" "$DEVICE_OBJECT"
check_sha256 device_text "$HIP_OVERLAY/device-text.bin" \
    "$EXPECTED_DEVICE_TEXT_SHA256"

if ! "$READELF" -d "$OUTPUT" | grep -q 'libamdhip64.so.7'; then
    printf 'binary does not depend on pinned libamdhip64.so.7\n' >&2
    exit 1
fi
if ! "$READELF" -d "$OUTPUT" | grep -Fq "$ROCM_CORE/lib"; then
    printf 'binary does not carry the pinned ROCm runpath\n' >&2
    exit 1
fi

printf 'output=%s\n' "$OUTPUT"
printf 'gfx1151=true wave32=true vector_max_workgroup=32 private_segment_bytes=0 spills=0 scratch_traffic=false\n'
printf 'native_bf16_dot2_sites=%s candidate_wmma_instructions=0 included_wmma_control=true\n' \
    "$DOT2_COUNT"
printf 'static_qualification=PASS gpu_execution=false\n'
