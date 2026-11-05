#!/usr/bin/env bash
set -euo pipefail

# Compile and statically qualify the isolated parallel-route experiment. This
# script never executes the resulting GPU binary.

: "${ROCWMMA_INCLUDE:?Set ROCWMMA_INCLUDE to the directory containing rocwmma/rocwmma.hpp}"

: "${ROCM_CORE:?Set ROCM_CORE to the ROCm SDK core directory documented in tools/hip/README.md}"
OUTPUT="${1:-/tmp/beholdd-q35-parallel-route-bench}"
SOURCE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$SOURCE_DIR/rocwmma_qwen35_resident_layer_parallel_route_bench.hip.cpp"
CONTROL="$SOURCE_DIR/rocwmma_qwen35_resident_layer_bench.hip.cpp"

EXPECTED_SOURCE_SHA256="abd707ec58bcb32a62de5cab063722362197496ba1d0bf2b77acd7bb1817eaaf"
EXPECTED_CONTROL_SHA256="dfed8f57bc310e1c8f366a13601ab9dccdd88da82744d5980e8370e858d1f3dd"
EXPECTED_ROCWMMA_COMMIT="bf1f3c1980e16cd9ceadc96a1d8ab2c72a006125"
EXPECTED_ROCWMMA_HEADER_TREE_SHA256="951fa79924e0a6b281f6701667d6da2733bb906eb8c24e4da60ba5cfb629b750"
EXPECTED_HIP_HEADER_TREE_SHA256="ad0a71b702c2dd044dfd7fb3f2e8e8418b8e367682bbb7188816940cb1741043"
EXPECTED_RESOURCE_HEADER_TREE_SHA256="413604f0742dee20a585685ad7d52f88869f2434c0914244501619fed532fcb8"
EXPECTED_DEVICE_LIB_TREE_SHA256="e8b3b2834ddf0982dd1148f532ee0e6ef956d9759d817f4e9e5bc4f61fba9a40"
EXPECTED_COMPILER_SHA256="1b2f3f4a10e946b3d2ac133c8f0ba4406b20b58679a3cd6d9054465849d18db0"
EXPECTED_HIP_RUNTIME_SHA256="eb25caf1754e031218c5245f2c0aae55e3607a527d9e53d36719b6caba1b4e05"
EXPECTED_HIP_HEADER_SHA256="0c2270e8f18150798e969daec4838ef84d0bc761394ecc933268e8eda149b37e"
EXPECTED_OCML_SHA256="a9d523fa792412ab9980ee76fb43214dc27983a07f0b23354424e9b04e91682f"
EXPECTED_DEVICE_TEXT_SHA256="01eac5f618ec516e260da4110ec793d7c6c1392f71db1118b3ffd966a35ae114"

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
    "$CONTROL"; do
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
check_sha256 frozen_control "$CONTROL" "$EXPECTED_CONTROL_SHA256"
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

PRE_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
PRE_CONTROL_SHA256="$(sha256sum "$CONTROL" | cut -d' ' -f1)"
HIP_OVERLAY="$(mktemp -d /tmp/beholdd-q35-parallel-route-hip-root.XXXXXX)"
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

POST_SOURCE_SHA256="$(sha256sum "$SOURCE" | cut -d' ' -f1)"
POST_CONTROL_SHA256="$(sha256sum "$CONTROL" | cut -d' ' -f1)"
if [[ "$POST_SOURCE_SHA256" != "$PRE_SOURCE_SHA256" ||
      "$POST_CONTROL_SHA256" != "$PRE_CONTROL_SHA256" ]]; then
    printf 'candidate or frozen control changed during compilation\n' >&2
    exit 1
fi
check_sha256 postcompile_source "$SOURCE" "$EXPECTED_SOURCE_SHA256"
check_sha256 postcompile_frozen_control "$CONTROL" "$EXPECTED_CONTROL_SHA256"

VERIFY_BINARY="$HIP_OVERLAY/verify-binary"
cp -- "$OUTPUT" "$VERIFY_BINARY"
"$OBJDUMP" --offloading "$VERIFY_BINARY" >/dev/null
DEVICE_OBJECT="$VERIFY_BINARY.0.hipv4-amdgcn-amd-amdhsa--gfx1151"
if [[ ! -s "$DEVICE_OBJECT" ]]; then
    printf 'missing exact gfx1151 device image in %s\n' "$OUTPUT" >&2
    exit 1
fi

"$READOBJ" --file-headers "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-header.txt"
if ! grep -q 'EF_AMDGPU_MACH_AMDGCN_GFX1151' "$HIP_OVERLAY/device-header.txt"; then
    printf 'embedded code object is not exact gfx1151\n' >&2
    exit 1
fi
"$READOBJ" --notes "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-notes.txt"

awk '
    /^[[:space:]]+\.group_segment_fixed_size:/ { group_bytes = $2; next }
    /^[[:space:]]+\.max_flat_workgroup_size:/ { maximum = $2; next }
    /^[[:space:]]+\.name:/ {
        name = $2
        if (name ~ /^q35_/) {
            seen[name] = 1
            active = 1
            if (name == "q35_stable_softmax_topk8_parallel") {
                candidate_seen = 1
                if (group_bytes != 2056) {
                    print "parallel route LDS mismatch: " group_bytes > "/dev/stderr"
                    failed = 1
                }
                if (maximum != 256) {
                    print "parallel route max workgroup is not 256" > "/dev/stderr"
                    failed = 1
                }
            }
        } else {
            active = 0
        }
        next
    }
    active && /^[[:space:]]+\.private_segment_fixed_size:/ && $2 != 0 {
        print "kernel has private memory: " name " bytes=" $2 > "/dev/stderr"
        failed = 1
    }
    active && /^[[:space:]]+\.sgpr_spill_count:/ && $2 != 0 {
        print "kernel has SGPR spills: " name " count=" $2 > "/dev/stderr"
        failed = 1
    }
    active && /^[[:space:]]+\.vgpr_count:/ {
        if (name == "q35_stable_softmax_topk8_parallel") {
            candidate_vgpr = $2
            if ($2 > 64) {
                print "parallel route exceeds 64 VGPRs: " $2 > "/dev/stderr"
                failed = 1
            }
        }
    }
    active && /^[[:space:]]+\.vgpr_spill_count:/ && $2 != 0 {
        print "kernel has VGPR spills: " name " count=" $2 > "/dev/stderr"
        failed = 1
    }
    active && /^[[:space:]]+\.uses_dynamic_stack:/ && $2 != "false" {
        print "kernel uses dynamic stack: " name > "/dev/stderr"
        failed = 1
    }
    active && /^[[:space:]]+\.wavefront_size:/ && $2 != 32 {
        print "kernel is not wave32: " name > "/dev/stderr"
        failed = 1
    }
    END {
        required[1] = "q35_rmsnorm_f32_to_f32_bf16"
        required[2] = "q35_router_and_shared_gate_f32"
        required[3] = "q35_stable_softmax_topk8_serial"
        required[4] = "q35_stable_softmax_topk8_parallel"
        required[5] = "q35_selected_shared_gate_up_wmma"
        required[6] = "q35_silu_multiply_to_bf16"
        required[7] = "q35_selected_shared_down_wmma"
        required[8] = "q35_route_shared_residual_reduce"
        for (idx = 1; idx <= 8; ++idx) {
            if (!seen[required[idx]]) {
                print "missing kernel metadata: " required[idx] > "/dev/stderr"
                failed = 1
            }
        }
        if (!candidate_seen || candidate_vgpr == "") {
            print "parallel route metadata was incomplete" > "/dev/stderr"
            failed = 1
        }
        print "parallel_route_vgpr=" candidate_vgpr \
              " parallel_route_lds_bytes=2056" \
              " parallel_route_max_workgroup=256"
        exit failed
    }
' "$HIP_OVERLAY/device-notes.txt"

"$OBJDUMP" -d --mcpu=gfx1151 "$DEVICE_OBJECT" >"$HIP_OVERLAY/device-isa.txt"
WMMA_COUNT="$(grep -c 'v_wmma_f32_16x16x16_bf16' "$HIP_OVERLAY/device-isa.txt")"
if [[ "$WMMA_COUNT" != 8 ]]; then
    printf 'expected exactly 8 emitted BF16 WMMA opcodes, found %s\n' \
        "$WMMA_COUNT" >&2
    exit 1
fi
if grep -Eq 'scratch_(load|store)|flat_scratch' "$HIP_OVERLAY/device-isa.txt"; then
    printf 'device ISA contains scratch traffic\n' >&2
    exit 1
fi
"$OBJDUMP" -d --mcpu=gfx1151 \
    --disassemble-symbols=q35_stable_softmax_topk8_parallel \
    "$DEVICE_OBJECT" >"$HIP_OVERLAY/parallel-route-isa.txt"
BARRIER_COUNT="$(grep -c 's_barrier' "$HIP_OVERLAY/parallel-route-isa.txt")"
if (( BARRIER_COUNT < 1 )); then
    printf 'parallel route kernel contains no workgroup barrier\n' >&2
    exit 1
fi

"$OBJCOPY" --dump-section .text="$HIP_OVERLAY/device-text.bin" "$DEVICE_OBJECT"
check_sha256 device_text "$HIP_OVERLAY/device-text.bin" \
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
