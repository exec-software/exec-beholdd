#!/usr/bin/env python3
"""Validate and summarize paired prefetch-off/prefetch-on decode logs.

The analyzer consumes the stable human-readable telemetry emitted by the `decode`
example.  It does not infer missing measurements or benchmark conditions.  Supplying
logits artifacts enables strict artifact checking: one artifact must accompany every
log, and all artifacts must be byte-identical.

usage: analyze_streaming_benchmark.py \
    --off off-1.log --on on-1.log [--off ... --on ...] \
    [--off-logits off-1.bin --on-logits on-1.bin ...]
"""

import argparse
import hashlib
import json
import math
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path


MIN_RUNS_PER_MODE = 3
READ_CHUNK_BYTES = 1024 * 1024


class BenchmarkError(ValueError):
    """The supplied logs or artifacts cannot prove the benchmark invariants."""


class JsonArgumentParser(argparse.ArgumentParser):
    """Route command-line errors through the analyzer's deterministic JSON failure."""

    def error(self, message):
        raise BenchmarkError(message)


@dataclass(frozen=True)
class DecodeRun:
    """One completely parsed streaming decode report."""

    path: str
    storage: str
    fit_policy: str
    weight_mode: str
    checkpoint_bytes: int
    checkpoint_sha256: str | None
    prompt_ids: tuple
    generated_ids: tuple
    device: str
    metrics: dict


def _read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise BenchmarkError(f"cannot read decode log {path}: {error}") from error


def _one_segment(text, marker, path):
    matches = []
    for line_number, line in enumerate(text.splitlines(), 1):
        offset = line.find(marker)
        if offset >= 0:
            matches.append((line_number, line[offset:].strip()))
    if not matches:
        raise BenchmarkError(f"{path}: missing telemetry line {marker.strip()!r}")
    if len(matches) != 1:
        lines = ", ".join(str(line_number) for line_number, _ in matches)
        raise BenchmarkError(
            f"{path}: telemetry line {marker.strip()!r} appears {len(matches)} times "
            f"(lines {lines})"
        )
    return matches[0][1]


def _match(segment, prefix, pattern, path):
    import re

    match = re.fullmatch(pattern, segment)
    if match is None:
        raise BenchmarkError(f"{path}: malformed {prefix} telemetry: {segment!r}")
    return match.groups()


def _ids(value, label, path):
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise BenchmarkError(f"{path}: malformed {label} token IDs: {value!r}") from error
    if not isinstance(parsed, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in parsed
    ):
        raise BenchmarkError(f"{path}: {label} token IDs must be non-negative integers")
    return tuple(parsed)


def _float(value, label, path):
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise BenchmarkError(f"{path}: {label} must be finite and non-negative")
    return parsed


def parse_decode_log(path):
    """Parse one current `decode` streaming telemetry report."""

    path = str(path)
    text = _read_text(path)
    number = r"\d+(?:\.\d+)?"

    storage = _match(
        _one_segment(text, "  storage:", path),
        "storage",
        r"storage:\s+(.+)",
        path,
    )[0]
    fit_policy = _match(
        _one_segment(text, "  fit policy:", path),
        "fit policy",
        r"fit policy:\s+(\S+)",
        path,
    )[0]
    weight_mode = _match(
        _one_segment(text, "  weight mode:", path),
        "weight mode",
        r"weight mode:\s+(\S+)",
        path,
    )[0]
    checkpoint_bytes = int(
        _match(
            _one_segment(text, "  checkpoint:", path),
            "checkpoint",
            r"checkpoint:\s+(\d+) encoded bytes",
            path,
        )[0]
    )
    checkpoint_sha256_raw = _match(
        _one_segment(text, "  checkpoint sha256:", path),
        "checkpoint sha256",
        r"checkpoint sha256:\s+(not supplied|[0-9a-f]{64})",
        path,
    )[0]
    checkpoint_sha256 = (
        None if checkpoint_sha256_raw == "not supplied" else checkpoint_sha256_raw
    )
    prompt_ids = _ids(
        _match(
            _one_segment(text, "  prompt:", path),
            "prompt",
            r"prompt:\s+(\[.*\])",
            path,
        )[0],
        "prompt",
        path,
    )
    generated_ids = _ids(
        _match(
            _one_segment(text, "  generated:", path),
            "generated",
            r"generated:\s+(\[.*\])",
            path,
        )[0],
        "generated",
        path,
    )
    device = _match(
        _one_segment(text, "  device:", path),
        "device",
        r"device:\s+(.+)",
        path,
    )[0]

    decode_seconds, seconds_per_token = _match(
        _one_segment(text, "  decode:", path),
        "decode",
        rf"decode:\s+({number})s \(({number})s / generated token\)",
        path,
    )
    (
        read_bytes,
        read_bytes_per_token,
        physical_reads,
        read_failures,
        read_seconds,
        read_service_gbps,
        wall_gbps,
        io_wall_seconds,
    ) = _match(
        _one_segment(text, "  weight I/O:", path),
        "weight I/O",
        rf"weight I/O:\s+(\d+) bytes \(({number}) / generated token\) in "
        rf"(\d+) physical reads \((\d+) failures\) over ({number})s "
        rf"\(({number}) GB/s read service, ({number}) GB/s over ({number})s "
        r"continuous decode-to-settlement wall\)",
        path,
    )
    (
        demanded_bytes,
        demanded_bytes_per_token,
        hit_bytes,
        byte_hit_percent,
        read_amplification,
    ) = _match(
        _one_segment(text, "  demand:", path),
        "demand",
        rf"demand:\s+(\d+) bytes \(({number}) / generated token\); (\d+) hit bytes "
        rf"\(({number})%\); ({number})x physical read amplification",
        path,
    )
    request_hits, requests, request_hit_percent, misses, foreground_stall_seconds = _match(
        _one_segment(text, "  requests:", path),
        "requests",
        rf"requests:\s+(\d+) / (\d+) hit \(({number})%\); (\d+) misses; "
        rf"({number})s foreground weight stall",
        path,
    )
    (
        prefetch_submitted,
        prefetch_admitted,
        prefetch_skipped,
        prefetch_useful,
        prefetch_late,
        prefetch_wasted,
        prefetch_cancelled,
    ) = _match(
        _one_segment(text, "  prefetch:", path),
        "prefetch",
        r"prefetch:\s+(\d+) submitted, (\d+) admitted, (\d+) skipped; "
        r"(\d+) useful / (\d+) late / (\d+) wasted / (\d+) cancelled",
        path,
    )
    (
        prefetch_useful_bytes,
        prefetch_useful_bytes_per_token,
        prefetch_wasted_bytes,
        prefetch_read_bytes,
        prefetch_reads,
        prefetch_read_failures,
        prefetch_read_seconds,
        prefetch_hidden_seconds,
    ) = _match(
        _one_segment(text, "  prefetch I/O:", path),
        "prefetch I/O",
        rf"prefetch I/O:\s+(\d+) useful bytes \(({number}) / generated token\), "
        rf"(\d+) wasted bytes; (\d+) bytes in (\d+) reads \((\d+) failures\) "
        rf"over ({number})s, ({number})s hidden",
        path,
    )
    (
        prefetch_attached_peak,
        prefetch_detached_peak,
        prefetch_queued_peak,
        prefetch_reading_peak,
        prefetch_ready_peak,
        prefetch_reserved_bytes_peak,
        prefetch_reserved_bytes_limit,
    ) = _match(
        _one_segment(text, "  prefetch peak:", path),
        "prefetch peak",
        r"prefetch peak:\s+(\d+) attached \+ (\d+) detached, (\d+) queued, "
        r"(\d+) reading, (\d+) ready; (\d+) / (\d+) reserved bytes",
        path,
    )
    prefetch_drain_seconds = _match(
        _one_segment(text, "  prefetch drain:", path),
        "prefetch drain",
        rf"prefetch drain:\s+({number})s after the decode timer",
        path,
    )[0]
    (
        cached_bytes,
        cache_bytes_limit,
        pinned_bytes,
        demand_cache_bytes,
        evictions,
        prefetch_budget_bytes,
        prefetch_queue_depth,
    ) = _match(
        _one_segment(text, "  cache:", path),
        "cache",
        r"cache:\s+(\d+) / (\d+) retained bytes \((\d+) pinned, (\d+) demand, "
        r"(\d+) evictions\); prefetch budget (\d+) bytes / (\d+) attached requests",
        path,
    )
    process_execution, process_final, process_peak = _match(
        _one_segment(text, "  process RSS:", path),
        "process RSS",
        r"process RSS:\s+(\d+) bytes at execution end, (\d+) bytes after "
        r"finalization, (\d+) bytes lifetime peak at final sample",
        path,
    )

    cgroup_segment = _one_segment(text, "  cgroup:", path)
    if "unavailable" in cgroup_segment:
        cgroup_values = (None, None, None)
    elif "membership changed" in cgroup_segment:
        cgroup_values = (None, None, None)
    else:
        cgroup_values = tuple(
            int(value)
            for value in _match(
                cgroup_segment,
                "cgroup",
                r"cgroup:\s+(\d+) bytes at execution end, (\d+) bytes after "
                r"finalization, (\d+) bytes cgroup lifetime peak at final sample",
                path,
            )
        )

    integer_values = {
        "weight_read_bytes": read_bytes,
        "physical_reads": physical_reads,
        "read_failures": read_failures,
        "foreground_demanded_bytes": demanded_bytes,
        "hit_bytes": hit_bytes,
        "request_hits": request_hits,
        "requests": requests,
        "misses": misses,
        "prefetch_submitted": prefetch_submitted,
        "prefetch_admitted": prefetch_admitted,
        "prefetch_skipped": prefetch_skipped,
        "prefetch_useful": prefetch_useful,
        "prefetch_late": prefetch_late,
        "prefetch_wasted": prefetch_wasted,
        "prefetch_cancelled": prefetch_cancelled,
        "prefetch_useful_bytes": prefetch_useful_bytes,
        "prefetch_wasted_bytes": prefetch_wasted_bytes,
        "prefetch_read_bytes": prefetch_read_bytes,
        "prefetch_reads": prefetch_reads,
        "prefetch_read_failures": prefetch_read_failures,
        "prefetch_attached_peak": prefetch_attached_peak,
        "prefetch_detached_peak": prefetch_detached_peak,
        "prefetch_queued_peak": prefetch_queued_peak,
        "prefetch_reading_peak": prefetch_reading_peak,
        "prefetch_ready_peak": prefetch_ready_peak,
        "prefetch_reserved_bytes_peak": prefetch_reserved_bytes_peak,
        "prefetch_reserved_bytes_limit": prefetch_reserved_bytes_limit,
        "cached_bytes": cached_bytes,
        "cache_bytes_limit": cache_bytes_limit,
        "pinned_bytes": pinned_bytes,
        "demand_cache_bytes": demand_cache_bytes,
        "evictions": evictions,
        "prefetch_budget_bytes": prefetch_budget_bytes,
        "prefetch_queue_depth": prefetch_queue_depth,
        "process_execution_rss_bytes": process_execution,
        "process_final_rss_bytes": process_final,
        "process_peak_rss_bytes": process_peak,
    }
    float_values = {
        "decode_seconds": decode_seconds,
        "seconds_per_generated_token": seconds_per_token,
        "weight_read_bytes_per_generated_token": read_bytes_per_token,
        "read_seconds": read_seconds,
        "read_service_gbps": read_service_gbps,
        "io_wall_gbps": wall_gbps,
        "io_wall_seconds": io_wall_seconds,
        "demanded_bytes_per_generated_token": demanded_bytes_per_token,
        "byte_hit_percent": byte_hit_percent,
        "read_amplification": read_amplification,
        "request_hit_percent": request_hit_percent,
        "foreground_stall_seconds": foreground_stall_seconds,
        "prefetch_useful_bytes_per_generated_token": prefetch_useful_bytes_per_token,
        "prefetch_read_seconds": prefetch_read_seconds,
        "prefetch_hidden_seconds": prefetch_hidden_seconds,
        "prefetch_drain_seconds": prefetch_drain_seconds,
    }
    metrics = {key: int(value) for key, value in integer_values.items()}
    metrics.update(
        {key: _float(value, key, path) for key, value in float_values.items()}
    )
    metrics.update(
        {
            "cgroup_execution_bytes": cgroup_values[0],
            "cgroup_final_bytes": cgroup_values[1],
            "cgroup_peak_bytes": cgroup_values[2],
        }
    )
    return DecodeRun(
        path=path,
        storage=storage,
        fit_policy=fit_policy,
        weight_mode=weight_mode,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=checkpoint_sha256,
        prompt_ids=prompt_ids,
        generated_ids=generated_ids,
        device=device,
        metrics=metrics,
    )


def _require_equal(runs, label, getter):
    first = getter(runs[0])
    for run in runs[1:]:
        value = getter(run)
        if value != first:
            raise BenchmarkError(
                f"benchmark {label} differs: {runs[0].path} has {first!r}, "
                f"{run.path} has {value!r}"
            )
    return first


def _require_checksum(runs):
    """Refuse to treat runs as paired unless every one carries an equal checkpoint sha256.

    Encoded byte count alone (`checkpoint_bytes`, checked separately via `_require_equal`)
    does not prove two runs read the same checkpoint content; a missing or mismatched
    checksum must fail loudly rather than be silently trusted.
    """

    missing = [run.path for run in runs if not run.checkpoint_sha256]
    if missing:
        raise BenchmarkError(
            "checkpoint sha256 is required for a paired benchmark but missing from: "
            + ", ".join(missing)
            + " (pass --checkpoint-sha256 to `decode`, computed once outside the timed "
            "path with e.g. `sha256sum <checkpoint>`)"
        )
    return _require_equal(runs, "checkpoint sha256", lambda run: run.checkpoint_sha256)


def _validate_common_run(run):
    metrics = run.metrics
    generated_count = len(run.generated_ids)
    if generated_count == 0:
        raise BenchmarkError(f"{run.path}: benchmark generated zero tokens")
    if run.fit_policy != "exact":
        raise BenchmarkError(f"{run.path}: fit policy is {run.fit_policy!r}, expected 'exact'")
    if run.weight_mode != "streaming":
        raise BenchmarkError(
            f"{run.path}: weight mode is {run.weight_mode!r}, expected 'streaming'"
        )
    if metrics["read_failures"] != 0:
        raise BenchmarkError(
            f"{run.path}: physical weight reads reported {metrics['read_failures']} failures"
        )
    if metrics["prefetch_read_failures"] != 0:
        raise BenchmarkError(
            f"{run.path}: prefetch reads reported "
            f"{metrics['prefetch_read_failures']} failures"
        )
    if metrics["request_hits"] + metrics["misses"] != metrics["requests"]:
        raise BenchmarkError(f"{run.path}: request hits plus misses do not equal requests")
    if metrics["pinned_bytes"] + metrics["demand_cache_bytes"] != metrics["cached_bytes"]:
        raise BenchmarkError(f"{run.path}: pinned plus demand bytes do not equal retained bytes")
    if metrics["cached_bytes"] > metrics["cache_bytes_limit"]:
        raise BenchmarkError(f"{run.path}: retained cache exceeds its declared byte limit")
    if metrics["prefetch_reserved_bytes_peak"] > metrics["prefetch_reserved_bytes_limit"]:
        raise BenchmarkError(f"{run.path}: prefetch reserved-byte peak exceeds its limit")
    if metrics["prefetch_reserved_bytes_limit"] != metrics["prefetch_budget_bytes"]:
        raise BenchmarkError(f"{run.path}: reported prefetch byte limits disagree")
    if metrics["prefetch_detached_peak"] > 1:
        raise BenchmarkError(f"{run.path}: more than one detached prefetch was observed")
    for field in (
        "cgroup_execution_bytes",
        "cgroup_final_bytes",
        "cgroup_peak_bytes",
    ):
        if metrics[field] is None:
            raise BenchmarkError(
                f"{run.path}: cgroup telemetry is unavailable; "
                "a stable cgroup-v2 membership is required"
            )

    expected_per_token = metrics["decode_seconds"] / generated_count
    if not math.isclose(
        metrics["seconds_per_generated_token"],
        expected_per_token,
        rel_tol=0.0,
        abs_tol=0.006,
    ):
        raise BenchmarkError(
            f"{run.path}: decode seconds/token disagrees with total time and token count"
        )


OFF_ZERO_FIELDS = (
    "prefetch_submitted",
    "prefetch_admitted",
    "prefetch_skipped",
    "prefetch_useful",
    "prefetch_late",
    "prefetch_wasted",
    "prefetch_cancelled",
    "prefetch_useful_bytes",
    "prefetch_wasted_bytes",
    "prefetch_read_bytes",
    "prefetch_reads",
    "prefetch_read_failures",
    "prefetch_read_seconds",
    "prefetch_hidden_seconds",
    "prefetch_attached_peak",
    "prefetch_detached_peak",
    "prefetch_queued_peak",
    "prefetch_reading_peak",
    "prefetch_ready_peak",
    "prefetch_reserved_bytes_peak",
    "prefetch_reserved_bytes_limit",
    "prefetch_budget_bytes",
    "prefetch_queue_depth",
)


def _validate_off(run):
    nonzero = {
        field: run.metrics[field]
        for field in OFF_ZERO_FIELDS
        if run.metrics[field] != 0
    }
    if nonzero:
        details = ", ".join(f"{key}={value}" for key, value in sorted(nonzero.items()))
        raise BenchmarkError(f"{run.path}: prefetch-off telemetry is nonzero: {details}")


def _validate_on(run):
    metrics = run.metrics
    if metrics["prefetch_budget_bytes"] <= 0 or metrics["prefetch_queue_depth"] <= 0:
        raise BenchmarkError(f"{run.path}: prefetch-on has a zero byte or queue-depth limit")
    if metrics["prefetch_submitted"] <= 0 or metrics["prefetch_admitted"] <= 0:
        raise BenchmarkError(f"{run.path}: prefetch-on did not submit and admit work")
    if metrics["prefetch_reserved_bytes_peak"] <= 0:
        raise BenchmarkError(f"{run.path}: prefetch-on never reserved bytes")
    if metrics["prefetch_attached_peak"] > metrics["prefetch_queue_depth"]:
        raise BenchmarkError(f"{run.path}: attached prefetch peak exceeds queue depth")
    if metrics["prefetch_queued_peak"] > metrics["prefetch_attached_peak"]:
        raise BenchmarkError(f"{run.path}: queued prefetch peak exceeds attached peak")
    if metrics["prefetch_ready_peak"] > metrics["prefetch_attached_peak"]:
        raise BenchmarkError(f"{run.path}: ready prefetch peak exceeds attached peak")
    if metrics["prefetch_reading_peak"] > (
        metrics["prefetch_attached_peak"] + metrics["prefetch_detached_peak"]
    ):
        raise BenchmarkError(f"{run.path}: reading prefetch peak exceeds tracked work")


def _median_metrics(runs):
    medians = {}
    for key in sorted(runs[0].metrics):
        values = [run.metrics[key] for run in runs]
        present = [value is not None for value in values]
        if any(present) and not all(present):
            raise BenchmarkError(f"metric {key!r} is available in only some logs")
        medians[key] = statistics.median(values) if all(present) else None
    return medians


def _artifact_descriptor(path):
    digest = hashlib.sha256()
    size = 0
    try:
        with Path(path).open("rb") as source:
            while True:
                chunk = source.read(READ_CHUNK_BYTES)
                if not chunk:
                    break
                digest.update(chunk)
                size += len(chunk)
    except OSError as error:
        raise BenchmarkError(f"cannot read logits artifact {path}: {error}") from error
    return size, digest.hexdigest()


def _files_equal(reference, candidate):
    try:
        with Path(reference).open("rb") as left, Path(candidate).open("rb") as right:
            while True:
                left_chunk = left.read(READ_CHUNK_BYTES)
                right_chunk = right.read(READ_CHUNK_BYTES)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    return True
    except OSError as error:
        raise BenchmarkError(
            f"cannot compare logits artifacts {reference} and {candidate}: {error}"
        ) from error


def _verify_logits(off_paths, on_paths, off_count, on_count):
    off_paths = list(off_paths or [])
    on_paths = list(on_paths or [])
    if not off_paths and not on_paths:
        return {"bytes": None, "checked": False, "count": 0, "sha256": None}
    if len(off_paths) != off_count or len(on_paths) != on_count:
        raise BenchmarkError(
            "strict logits checking needs exactly one --off-logits per --off log and "
            "one --on-logits per --on log"
        )

    paths = off_paths + on_paths
    reference = paths[0]
    size, digest = _artifact_descriptor(reference)
    for candidate in paths[1:]:
        candidate_size, _ = _artifact_descriptor(candidate)
        if candidate_size != size or not _files_equal(reference, candidate):
            raise BenchmarkError(
                f"logits artifacts are not byte-identical: {reference} != {candidate}"
            )
    return {"bytes": size, "checked": True, "count": len(paths), "sha256": digest}


def _validate_cgroup_limit(runs, limit):
    if limit is None:
        return None
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise BenchmarkError("cgroup limit must be a positive integer byte count")
    for run in runs:
        for field in (
            "cgroup_execution_bytes",
            "cgroup_final_bytes",
            "cgroup_peak_bytes",
        ):
            value = run.metrics[field]
            if value > limit:
                raise BenchmarkError(
                    f"{run.path}: {field}={value} exceeds declared cgroup limit {limit}"
                )
    return limit


def analyze(
    off_logs,
    on_logs,
    off_logits=None,
    on_logits=None,
    cgroup_limit_bytes=None,
):
    """Analyze paired logs and return the deterministic JSON-ready record."""

    off_logs = list(off_logs or [])
    on_logs = list(on_logs or [])
    if len(off_logs) < MIN_RUNS_PER_MODE or len(on_logs) < MIN_RUNS_PER_MODE:
        raise BenchmarkError(
            f"at least {MIN_RUNS_PER_MODE} --off and {MIN_RUNS_PER_MODE} --on logs are required"
        )
    if len(off_logs) != len(on_logs):
        raise BenchmarkError(
            f"paired benchmark needs equal run counts, got {len(off_logs)} off and "
            f"{len(on_logs)} on"
        )

    off_runs = [parse_decode_log(path) for path in off_logs]
    on_runs = [parse_decode_log(path) for path in on_logs]
    all_runs = off_runs + on_runs
    for run in all_runs:
        _validate_common_run(run)
    for run in off_runs:
        _validate_off(run)
    for run in on_runs:
        _validate_on(run)
    cgroup_limit_bytes = _validate_cgroup_limit(all_runs, cgroup_limit_bytes)

    shared = {
        "checkpoint_bytes": _require_equal(
            all_runs, "checkpoint encoded bytes", lambda run: run.checkpoint_bytes
        ),
        "checkpoint_sha256": _require_checksum(all_runs),
        "cgroup_limit_bytes": cgroup_limit_bytes,
        "device": _require_equal(all_runs, "device", lambda run: run.device),
        "fit_policy": _require_equal(
            all_runs, "fit policy", lambda run: run.fit_policy
        ),
        "foreground_demanded_bytes": _require_equal(
            all_runs,
            "foreground demanded bytes",
            lambda run: run.metrics["foreground_demanded_bytes"],
        ),
        "generated_ids": list(
            _require_equal(all_runs, "generated token IDs", lambda run: run.generated_ids)
        ),
        "prompt_ids": list(
            _require_equal(all_runs, "prompt token IDs", lambda run: run.prompt_ids)
        ),
        "requests": _require_equal(
            all_runs, "foreground request count", lambda run: run.metrics["requests"]
        ),
        "storage": _require_equal(all_runs, "storage class", lambda run: run.storage),
        "weight_mode": _require_equal(
            all_runs, "weight mode", lambda run: run.weight_mode
        ),
    }

    off_medians = _median_metrics(off_runs)
    on_medians = _median_metrics(on_runs)
    off_seconds = off_medians["seconds_per_generated_token"]
    on_seconds = on_medians["seconds_per_generated_token"]
    if not on_seconds < off_seconds:
        raise BenchmarkError(
            "prefetch-on median seconds/generated-token is not lower than prefetch-off: "
            f"{on_seconds} >= {off_seconds}"
        )

    logits = _verify_logits(
        off_logits,
        on_logits,
        len(off_runs),
        len(on_runs),
    )
    reduction = off_seconds - on_seconds
    return {
        "comparison": {
            "off_median_seconds_per_generated_token": off_seconds,
            "on_less_than_off": True,
            "on_median_seconds_per_generated_token": on_seconds,
            "reduction_percent": reduction / off_seconds * 100.0,
            "reduction_seconds_per_generated_token": reduction,
        },
        "logits": logits,
        "off": {
            "logs": off_logs,
            "medians": off_medians,
            "run_count": len(off_runs),
        },
        "on": {
            "logs": on_logs,
            "medians": on_medians,
            "run_count": len(on_runs),
        },
        "pair_count": len(off_runs),
        "shared": shared,
        "status": "pass",
    }


def _parser():
    parser = JsonArgumentParser(description=__doc__)
    parser.add_argument("--off", action="append", default=[], metavar="LOG")
    parser.add_argument("--on", action="append", default=[], metavar="LOG")
    parser.add_argument("--off-logits", action="append", default=[], metavar="ARTIFACT")
    parser.add_argument("--on-logits", action="append", default=[], metavar="ARTIFACT")
    parser.add_argument(
        "--cgroup-limit-bytes",
        type=int,
        metavar="BYTES",
        help="validate every reported cgroup current/peak measurement against BYTES",
    )
    return parser


def main(argv=None):
    try:
        arguments = _parser().parse_args(sys.argv[1:] if argv is None else argv)
        record = analyze(
            arguments.off,
            arguments.on,
            arguments.off_logits,
            arguments.on_logits,
            arguments.cgroup_limit_bytes,
        )
    except (BenchmarkError, OverflowError) as error:
        print(
            json.dumps(
                {"error": str(error), "status": "fail"},
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
