#!/usr/bin/env python3

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    from tools import analyze_streaming_benchmark as analyzer
except ModuleNotFoundError:  # Direct execution puts tools/, not the repo root, on sys.path.
    import analyze_streaming_benchmark as analyzer


def decode_log(
    seconds_per_token,
    *,
    on,
    generated=(25, 26),
    read_failures=0,
    prefetch_read_failures=0,
    reserved_peak=None,
    reserved_limit=None,
    drain_seconds=None,
    cgroup_line=None,
    checkpoint_sha256="a" * 64,
):
    generated = list(generated)
    total_seconds = seconds_per_token * len(generated)
    if on:
        prefetch = (12, 8, 4, 5, 2, 1, 0)
        prefetch_io = (4096, 2048.0, 512, 4608, 6, prefetch_read_failures, 0.8, 0.5)
        peaks = (3, 0, 2, 1, 2, 4096, 8192)
        cache_tail = (8192, 3)
    else:
        prefetch = (0, 0, 0, 0, 0, 0, 0)
        prefetch_io = (0, 0.0, 0, 0, 0, prefetch_read_failures, 0.0, 0.0)
        peaks = (0, 0, 0, 0, 0, 0, 0)
        cache_tail = (0, 0)
    if reserved_peak is not None:
        peaks = peaks[:5] + (reserved_peak, peaks[6])
    if reserved_limit is not None:
        peaks = peaks[:6] + (reserved_limit,)
        cache_tail = (reserved_limit, cache_tail[1])
    if drain_seconds is None:
        drain_seconds = 0.2 if on else 0.0
    if cgroup_line is None:
        cgroup_line = "  cgroup:     300 bytes at execution end, 250 bytes after finalization, 500 bytes cgroup lifetime peak at final sample"
    checkpoint_sha256_line = (
        "  checkpoint sha256: not supplied"
        if checkpoint_sha256 is None
        else f"  checkpoint sha256: {checkpoint_sha256}"
    )

    return "\n".join(
        [
            "/checkpoint/model.gguf",
            "  preflight device: cpu",
            "  storage:         local solid-state",
            "  fit policy:      exact",
            "  weight mode:     streaming",
            "  checkpoint:      123456 encoded bytes",
            checkpoint_sha256_line,
            "  prompt:    [64]",
            f"  generated: {json.dumps(generated)}",
            "  device:     cpu",
            f"  decode:     {total_seconds:.2f}s ({seconds_per_token:.3f}s / generated token)",
            f"  weight I/O: 10000 bytes (5000.0 / generated token) in 10 physical reads ({read_failures} failures) over 1.000s (0.010 GB/s read service, 0.005 GB/s over 2.000s continuous decode-to-settlement wall)",
            "  demand:     12000 bytes (6000.0 / generated token); 2000 hit bytes (16.67%); 0.833x physical read amplification",
            "  requests:   2 / 12 hit (16.67%); 10 misses; 0.750s foreground weight stall",
            "  prefetch:   {} submitted, {} admitted, {} skipped; {} useful / {} late / {} wasted / {} cancelled".format(*prefetch),
            "  prefetch I/O: {} useful bytes ({:.1f} / generated token), {} wasted bytes; {} bytes in {} reads ({} failures) over {:.3f}s, {:.3f}s hidden".format(*prefetch_io),
            "  prefetch peak: {} attached + {} detached, {} queued, {} reading, {} ready; {} / {} reserved bytes".format(*peaks),
            f"  prefetch drain: {drain_seconds:.3f}s after the decode timer",
            "  cache:      2000 / 4096 retained bytes (1000 pinned, 1000 demand, 1 evictions); prefetch budget {} bytes / {} attached requests".format(*cache_tail),
            "  process RSS: 100 bytes at execution end, 80 bytes after finalization, 200 bytes lifetime peak at final sample",
            cgroup_line,
            "",
        ]
    )


class AnalyzeStreamingBenchmarkTests(unittest.TestCase):
    def write_runs(self, directory, off_seconds, on_seconds, **kwargs):
        off_paths = []
        on_paths = []
        for mode, seconds, paths in (
            (False, off_seconds, off_paths),
            (True, on_seconds, on_paths),
        ):
            for index, value in enumerate(seconds):
                path = Path(directory, f"{'on' if mode else 'off'}-{index}.log")
                path.write_text(decode_log(value, on=mode, **kwargs), encoding="utf-8")
                paths.append(str(path))
        return off_paths, on_paths

    def test_success_reports_medians_and_strict_identical_logits(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0, 1.8, 2.2], [1.5, 1.4, 1.6])
            artifacts = []
            for index in range(6):
                path = Path(directory, f"logits-{index}.bin")
                path.write_bytes(b"exact-logits")
                artifacts.append(str(path))

            report = analyzer.analyze(off, on, artifacts[:3], artifacts[3:])

            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["pair_count"], 3)
            self.assertEqual(
                report["comparison"]["off_median_seconds_per_generated_token"], 2.0
            )
            self.assertEqual(
                report["comparison"]["on_median_seconds_per_generated_token"], 1.5
            )
            self.assertTrue(report["comparison"]["on_less_than_off"])
            self.assertEqual(report["logits"]["bytes"], 12)
            self.assertEqual(report["logits"]["count"], 6)
            self.assertEqual(
                report["logits"]["sha256"], hashlib.sha256(b"exact-logits").hexdigest()
            )
            self.assertEqual(report["shared"]["generated_ids"], [25, 26])
            self.assertEqual(report["on"]["medians"]["prefetch_budget_bytes"], 8192)

    def test_off_drain_is_timing_not_prefetch_activity(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0] * 3, [1.0] * 3)
            Path(off[0]).write_text(
                decode_log(2.0, on=False, drain_seconds=0.001), encoding="utf-8"
            )

            report = analyzer.analyze(off, on)

            self.assertEqual(report["status"], "pass")

    def test_requires_cgroup_telemetry_and_checks_declared_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0] * 3, [1.0] * 3)
            Path(off[0]).write_text(
                decode_log(
                    2.0,
                    on=False,
                    cgroup_line="  cgroup:     unavailable (no unified cgroup-v2 membership)",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(analyzer.BenchmarkError, "cgroup telemetry"):
                analyzer.analyze(off, on)

            Path(off[0]).write_text(decode_log(2.0, on=False), encoding="utf-8")
            with self.assertRaisesRegex(analyzer.BenchmarkError, "exceeds declared"):
                analyzer.analyze(off, on, cgroup_limit_bytes=499)

            report = analyzer.analyze(off, on, cgroup_limit_bytes=500)
            self.assertEqual(report["shared"]["cgroup_limit_bytes"], 500)

    def test_checkpoint_sha256_must_be_present_and_equal(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0] * 3, [1.0] * 3)

            Path(off[0]).write_text(
                decode_log(2.0, on=False, checkpoint_sha256=None), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                analyzer.BenchmarkError, "checkpoint sha256 is required"
            ):
                analyzer.analyze(off, on)

            Path(off[0]).write_text(
                decode_log(2.0, on=False, checkpoint_sha256="b" * 64), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                analyzer.BenchmarkError, "checkpoint sha256 differs"
            ):
                analyzer.analyze(off, on)

            Path(off[0]).write_text(decode_log(2.0, on=False), encoding="utf-8")
            report = analyzer.analyze(off, on)
            self.assertEqual(report["shared"]["checkpoint_sha256"], "a" * 64)

    def test_requires_three_equal_sized_groups(self):
        with self.assertRaisesRegex(analyzer.BenchmarkError, "at least 3"):
            analyzer.analyze(["a", "b"], ["c", "d"])
        with self.assertRaisesRegex(analyzer.BenchmarkError, "equal run counts"):
            analyzer.analyze(["a", "b", "c"], ["d", "e", "f", "g"])

    def test_rejects_read_failures_and_nonzero_off_prefetch(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(
                directory,
                [2.0, 2.1, 2.2],
                [1.0, 1.1, 1.2],
                read_failures=1,
            )
            with self.assertRaisesRegex(analyzer.BenchmarkError, "weight reads reported 1"):
                analyzer.analyze(off, on)

            Path(off[0]).write_text(decode_log(2.0, on=True), encoding="utf-8")
            for path in off[1:] + on:
                Path(path).write_text(
                    decode_log(1.0 if path in on else 2.0, on=path in on),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(analyzer.BenchmarkError, "prefetch-off telemetry"):
                analyzer.analyze(off, on)

    def test_rejects_prefetch_read_failure_and_on_bound_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0] * 3, [1.0] * 3)
            Path(on[0]).write_text(
                decode_log(1.0, on=True, prefetch_read_failures=1), encoding="utf-8"
            )
            with self.assertRaisesRegex(analyzer.BenchmarkError, "prefetch reads reported 1"):
                analyzer.analyze(off, on)

            Path(on[0]).write_text(
                decode_log(1.0, on=True, reserved_peak=9000), encoding="utf-8"
            )
            with self.assertRaisesRegex(analyzer.BenchmarkError, "peak exceeds its limit"):
                analyzer.analyze(off, on)

    def test_rejects_non_improvement_and_different_generated_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [1.0, 1.1, 1.2], [1.1, 1.2, 1.3])
            with self.assertRaisesRegex(analyzer.BenchmarkError, "is not lower"):
                analyzer.analyze(off, on)

            Path(on[0]).write_text(
                decode_log(0.8, on=True, generated=(25, 99)), encoding="utf-8"
            )
            with self.assertRaisesRegex(analyzer.BenchmarkError, "generated token IDs differs"):
                analyzer.analyze(off, on)

    def test_logits_are_all_or_nothing_and_compared_by_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0] * 3, [1.0] * 3)
            one = Path(directory, "one.bin")
            two = Path(directory, "two.bin")
            one.write_bytes(b"same")
            two.write_bytes(b"different")

            with self.assertRaisesRegex(analyzer.BenchmarkError, "exactly one"):
                analyzer.analyze(off, on, [str(one)], [])
            with self.assertRaisesRegex(analyzer.BenchmarkError, "not byte-identical"):
                analyzer.analyze(off, on, [str(one)] * 3, [str(one), str(one), str(two)])

    def test_cli_is_deterministic_json_and_failures_are_nonzero_json(self):
        with tempfile.TemporaryDirectory() as directory:
            off, on = self.write_runs(directory, [2.0, 1.8, 2.2], [1.5, 1.4, 1.6])
            script = Path(analyzer.__file__)
            command = [sys.executable, str(script)]
            command.extend(["--cgroup-limit-bytes", "500"])
            for path in off:
                command.extend(["--off", path])
            for path in on:
                command.extend(["--on", path])

            first = subprocess.run(command, check=True, capture_output=True, text=True)
            second = subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first.stderr, "")
            self.assertEqual(first.stdout, second.stdout)
            self.assertEqual(json.loads(first.stdout)["status"], "pass")

            failed = subprocess.run(
                [sys.executable, str(script), "--off", "missing"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(failed.stdout, "")
            self.assertEqual(json.loads(failed.stderr)["status"], "fail")


if __name__ == "__main__":
    unittest.main()
