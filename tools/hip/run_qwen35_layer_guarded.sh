#!/usr/bin/env bash
set -euo pipefail

# Run a Qwen3.5/3.6 layer benchmark only after binding it to an expected pack
# digest. The digest and file identity are checked again after execution so a
# replaced or modified pack invalidates the result. This wrapper always places
# the GPU process under the repository thermal guard.

if (( $# < 3 )); then
    printf 'usage: %s <benchmark> <layer.pack> <expected-sha256> [benchmark options...]\n' \
        "$0" >&2
    exit 2
fi

benchmark_path="$1"
pack_path="$2"
expected_pack_sha256="$3"
shift 3

if [[ ! -x "$benchmark_path" ]]; then
    printf 'benchmark is not executable: %s\n' "$benchmark_path" >&2
    exit 2
fi
if [[ ! -f "$pack_path" ]]; then
    printf 'layer pack is not a regular file: %s\n' "$pack_path" >&2
    exit 2
fi
if [[ ! "$expected_pack_sha256" =~ ^[0-9a-f]{64}$ ]]; then
    printf 'expected pack SHA-256 must be exactly 64 lowercase hex characters\n' >&2
    exit 2
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
thermal_guard="$script_dir/../thermal_guard.py"
if [[ ! -f "$thermal_guard" ]]; then
    printf 'thermal guard is missing: %s\n' "$thermal_guard" >&2
    exit 2
fi

pack_identity_before="$(stat -Lc '%d:%i:%s:%Y:%Z' -- "$pack_path")"
observed_before="$(sha256sum -- "$pack_path" | cut -d' ' -f1)"
if [[ "$observed_before" != "$expected_pack_sha256" ]]; then
    printf 'layer pack SHA-256 mismatch before execution: expected %s, found %s\n' \
        "$expected_pack_sha256" "$observed_before" >&2
    exit 2
fi
printf 'pack_sha256=%s pack_identity=%s (pre-run match)\n' \
    "$observed_before" "$pack_identity_before"

set +e
python3 "$thermal_guard" \
    --warn-c 75 --critical-c 90 --poll-seconds 0.25 -- \
    env -u LD_LIBRARY_PATH -u LD_PRELOAD -u LD_AUDIT \
    "$benchmark_path" "$@" "$pack_path"
benchmark_status=$?
set -e

pack_identity_after="$(stat -Lc '%d:%i:%s:%Y:%Z' -- "$pack_path")"
observed_after="$(sha256sum -- "$pack_path" | cut -d' ' -f1)"
if [[ "$pack_identity_after" != "$pack_identity_before" ||
      "$observed_after" != "$expected_pack_sha256" ]]; then
    printf 'layer pack changed during execution: before identity=%s sha256=%s; after identity=%s sha256=%s\n' \
        "$pack_identity_before" "$observed_before" \
        "$pack_identity_after" "$observed_after" >&2
    exit 3
fi
printf 'pack_sha256=%s pack_identity=%s (post-run match)\n' \
    "$observed_after" "$pack_identity_after"

exit "$benchmark_status"
