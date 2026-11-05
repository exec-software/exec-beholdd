"""Fixture-only telemetry tests: no real procfs, sysfs, sleeps or workloads."""
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
SPEC = importlib.util.spec_from_file_location("hag_host_telemetry", Path(__file__).with_name("hag_host_telemetry.py"))
telemetry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(telemetry)


class Clock:
    def __init__(self):
        self.now = 100.0

    def monotonic(self):
        return self.now

    def utc(self):
        return "2026-10-11T00:00:%09.6f+00:00" % (self.now - 100)

    def sleep(self, seconds):
        if not 0 <= seconds <= 0.100001:
            raise AssertionError("unexpected long or negative sleep")
        self.now += seconds


class Source:
    def __init__(self, clock, read_seconds=0, payload=None):
        self.clock, self.read_seconds = clock, read_seconds
        self.payload = payload or {"sensors": {"tctl_c": {"status": "ok", "value": 50.0}}}

    def metadata(self):
        return {"fixture": True}

    def sample(self, now):
        self.clock.now += self.read_seconds
        return self.payload


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.proc, self.sys = self.root / "proc", self.root / "sys"
        self.proc.mkdir()
        self.sys.mkdir()
        self.clock = Clock()

    def write(self, path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def host(self, top=10):
        return telemetry.HostSource(proc_root=self.proc, sys_root=self.sys, top_processes=top,
                                    ticks_per_second=100, page_size=4096)

    def basic_proc(self):
        self.write(self.proc / "stat", "cpu 100 10 20 400 30 4 6 2 10 5\ncpu0 0 0 0 0\n")
        self.write(self.proc / "meminfo", "MemTotal: 1000 kB\nMemAvailable: 600 kB\nSwapTotal: 200 kB\nSwapFree: 150 kB\n")
        self.write(self.proc / "pressure/cpu", "some avg10=1.25 avg60=0.5 avg300=0.0 total=1000\n")
        self.write(self.proc / "pressure/memory", "some avg10=2.0 avg60=1.0 avg300=0.5 total=3000\nfull avg10=0.1 avg60=0.0 avg300=0.0 total=500\n")

    def process(self, pid, ticks, *, started=55, rss=10, comm="private secret ) words"):
        fields = ["0"] * 22
        fields[0], fields[11], fields[12], fields[19], fields[21] = "R", str(ticks), "0", str(started), str(rss)
        return self.write(self.proc / str(pid) / "stat", f"{pid} ({comm}) " + " ".join(fields) + "\n")

    def run_collect(self, *, duration=1, interval=0.1, source=None, **kwargs):
        output = self.root / "telemetry.jsonl"
        summary = telemetry.collect(output, duration, interval, source=source or Source(self.clock),
                                    monotonic=self.clock.monotonic, utc=self.clock.utc, sleep=self.clock.sleep, **kwargs)
        return output, summary, [json.loads(line) for line in output.read_text().splitlines()]

    def test_numeric_sensors_units_and_current_dpm_clock(self):
        hw = self.sys / "class/hwmon/hwmon0"
        self.write(hw / "name", "k10temp\n")
        self.write(hw / "temp1_label", "Tctl\n")
        self.write(hw / "temp1_input", "54375\n")
        base = self.sys / "class/drm/card1/device"
        self.write(base / "vendor", "0x1002\n")
        driver = self.sys / "bus/pci/drivers/amdgpu"
        driver.mkdir(parents=True)
        (base / "driver").symlink_to(driver, target_is_directory=True)
        self.write(base / "gpu_busy_percent", "37\n")
        self.write(base / "pp_dpm_sclk", "0: 500Mhz\n1: 1400Mhz *\n")
        self.write(base / "pp_dpm_mclk", "0: 800 Mhz *\n")
        self.write(base / "hwmon/hwmon1/name", "amdgpu\n")
        self.write(base / "hwmon/hwmon1/power1_average", "17500000\n")
        self.write(base / "hwmon/hwmon1/fan1_input", "2400\n")
        values = self.host().sample(1)["sensors"]
        self.assertEqual({key: row["value"] for key, row in values.items()}, {
            "tctl_c": 54.375, "card1.busy_percent": 37, "card1.core_clock_mhz": 1400,
            "card1.memory_clock_mhz": 800, "card1.power_w": 17.5, "card1.fan_rpm": 2400})
        self.assertEqual((base / "gpu_busy_percent").read_text(), "37\n")

    def test_missing_sensors_and_proc_are_explicit_not_zero(self):
        source = self.host()
        sample = source.sample(1)
        self.assertEqual(sample["sensors"]["tctl_c"], telemetry.unavailable("not_found"))
        self.assertEqual(sample["cpu"]["status"], "unavailable")
        self.assertEqual(sample["memory"]["status"], "unavailable")
        self.assertEqual(sample["pressure"]["cpu"]["status"], "unavailable")
        self.assertIn("DRM discovery unavailable", source.notes)

    def test_tctl_label_does_not_silently_substitute_other_temperature(self):
        hw = self.sys / "class/hwmon/hwmon0"
        self.write(hw / "name", "k10temp")
        self.write(hw / "temp1_label", "Tdie")
        self.write(hw / "temp1_input", "42000")
        self.assertEqual(self.host().sample(1)["sensors"]["tctl_c"]["status"], "unavailable")

    def test_sensor_rejects_malformed_out_of_range_and_ambiguous_clock(self):
        path = self.write(self.root / "sensor", "126000")
        spec = dict(path=path, scale=1000, maximum=125)
        self.assertEqual(telemetry.sensor(spec)["status"], "unavailable")
        path.write_text("NaN")
        self.assertEqual(telemetry.sensor(spec)["status"], "unavailable")
        path.write_text("0: 400Mhz *\n1: 500Mhz *")
        self.assertEqual(telemetry.sensor(dict(spec, kind="dpm"))["status"], "unavailable")

    def test_reader_refuses_fifo_device_symlink_and_oversize_before_open(self):
        fifo = self.root / "fifo"
        os.mkfifo(fifo)
        plain = self.write(self.root / "plain", "x" * 33)
        link = self.root / "link"
        link.symlink_to(plain)
        with patch.object(telemetry.os, "open", side_effect=AssertionError("must not open")):
            for path in (fifo, link, plain):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    telemetry.read_text(path, 32)

    def test_cpu_deltas_exclude_guest_and_separate_iowait(self):
        self.basic_proc()
        source = self.host(top=0)
        self.assertEqual(source.sample(1)["cpu"], telemetry.unavailable("baseline"))
        self.write(self.proc / "stat", "cpu 130 10 30 440 40 4 6 12 10000 5000\n")
        result = source.sample(2)["cpu"]
        self.assertEqual(result["total_ticks"], 100)
        self.assertEqual(result["busy_ticks"], 50)
        self.assertEqual(result["busy_percent"], 50)
        self.assertEqual(result["iowait_percent"], 10)

    def test_cpu_reset_read_failure_and_zero_delta_never_claim_idle(self):
        self.assertEqual(telemetry.cpu_delta([10] * 8, [9] * 8)["reason"], "counter_reset")
        self.assertEqual(telemetry.cpu_delta([10] * 8, [10] * 8)["reason"], "no_counter_progress")
        self.basic_proc()
        source = self.host()
        source.sample(1)
        self.write(self.proc / "stat", "broken")
        self.assertEqual(source.sample(2)["cpu"]["status"], "unavailable")
        self.basic_proc()
        self.assertEqual(source.sample(3)["cpu"]["reason"], "baseline")

    def test_memory_and_psi_units_and_malformed_values(self):
        self.basic_proc()
        sample = self.host(top=0).sample(1)
        self.assertEqual(sample["memory"]["available_bytes"], 600 * 1024)
        self.assertEqual(sample["memory"]["swap_used_bytes"], 50 * 1024)
        self.assertEqual(sample["pressure"]["cpu"]["some"]["avg10"], 1.25)
        self.assertEqual(sample["pressure"]["memory"]["full"]["total"], 500)
        for bad in ("some avg10=nan avg60=0 avg300=0 total=0", "some avg10=0 avg60=0 avg300=0 total=-1",
                    "some avg10=0 avg10=0 avg300=0 total=0", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                telemetry.pressure(bad)
        with self.assertRaises(ValueError):
            telemetry.memory("MemTotal: 1 kB\nMemAvailable: 2 kB\nSwapTotal: 0 kB\nSwapFree: 0 kB")

    def test_numeric_process_activity_omits_names_and_bounds_top(self):
        self.basic_proc()
        self.process(101, 100)
        self.process(102, 200)
        source = self.host(top=1)
        self.assertEqual(source.sample(1)["processes"]["status"], "baseline")
        self.process(101, 150)
        self.process(102, 400)
        value = source.sample(2)["processes"]
        self.assertEqual(value["top"], [{"pid": 102, "start_ticks": 55, "cpu_percent_one_core": 200.0, "rss_bytes": 40960}])
        serialized = json.dumps(value)
        self.assertNotIn("private", serialized)
        self.assertNotIn("comm", serialized)
        self.assertEqual(source.sample(2.5)["processes"]["status"], "not_sampled")

    def test_pid_reuse_exit_and_new_process_do_not_attribute_stale_cpu(self):
        self.basic_proc()
        self.process(101, 100, started=10)
        self.process(102, 100)
        source = self.host()
        source.sample(1)
        self.process(101, 10000, started=20)
        (self.proc / "102/stat").unlink()
        self.process(103, 10000)
        result = source.sample(2)["processes"]
        self.assertEqual(result["top"], [])
        self.assertEqual(result["unreadable_processes"], 1)

    def test_directory_scan_reports_truncation_and_bounds_retained_entries(self):
        for number in range(7):
            self.write(self.root / str(number), "x")
        result, truncated = telemetry.entries(self.root, 3)
        self.assertEqual(len(result), 3)
        self.assertTrue(truncated)

    def test_identity_binds_boot_numeric_process_and_unified_cgroup(self):
        boot = "12345678-1234-1234-1234-123456789abc"
        self.write(self.proc / "sys/kernel/random/boot_id", boot)
        pid = os.getpid()
        process = self.process(pid, 5, started=12345)
        self.write(self.proc / "self/stat", process.read_text())
        self.write(self.proc / "self/cgroup", "0::/user.slice/hag-passive.service\n")
        identity = self.host().metadata()["host_identity"]
        self.assertEqual(identity["collector_pid"], pid)
        self.assertEqual(identity["boot_id"], {"status": "ok", "value": boot})
        self.assertEqual(identity["collector_start_ticks"], {"status": "ok", "value": 12345})
        self.assertEqual(identity["unified_cgroup"], {"status": "ok", "value": "/user.slice/hag-passive.service"})
        self.assertNotIn("private", json.dumps(identity))

    def test_identity_missing_or_malformed_is_explicit(self):
        identity = self.host().identity()
        for key in ("boot_id", "collector_start_ticks", "unified_cgroup"):
            self.assertEqual(identity[key]["status"], "unavailable")
        self.write(self.proc / "sys/kernel/random/boot_id", "not-a-boot-id")
        self.write(self.proc / "self/cgroup", "0::/first\n0::/second")
        identity = self.host().identity()
        self.assertEqual(identity["boot_id"]["status"], "unavailable")
        self.assertEqual(identity["unified_cgroup"]["status"], "unavailable")

    def test_duration_interval_and_process_count_are_finite_and_bounded(self):
        for duration, interval in ((0, .1), (901, 1), (float("nan"), 1), (float("inf"), 1),
                                   (1, .09), (1, 2), (1, float("nan")), (True, .1)):
            with self.subTest(duration=duration, interval=interval), self.assertRaises(ValueError):
                telemetry.validate_options(duration, interval)
        for top in (-1, 21, True):
            with self.subTest(top=top), self.assertRaises(ValueError):
                self.host(top=top)
        telemetry.validate_options(900, .1)

    def test_duration_records_bracketed_samples_and_finite_summary(self):
        output, summary, rows = self.run_collect()
        self.assertEqual(rows[0]["event"], "START")
        samples = rows[1:-1]
        self.assertEqual(len(samples), 10)
        self.assertEqual(summary["reason"], "duration_elapsed")
        self.assertTrue(summary["duration_completed"])
        self.assertGreaterEqual(summary["elapsed_seconds"], 1)
        self.assertLess(summary["elapsed_seconds"], 1.000001)
        self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
        for index, row in enumerate(samples):
            self.assertEqual(row["sample"], index + 1)
            self.assertGreaterEqual(row["sample_end_monotonic_seconds"], row["monotonic_seconds"])
            self.assertIn("+00:00", row["timestamp_utc"])
            self.assertLess(row["elapsed_seconds"], 1)

    def test_slow_read_skips_deadlines_without_catchup_burst(self):
        _, summary, rows = self.run_collect(source=Source(self.clock, read_seconds=.35))
        samples = rows[1:-1]
        self.assertEqual(len(samples), 3)
        self.assertGreaterEqual(summary["skipped_intervals"], 7)
        self.assertTrue(all(row["read_duration_seconds"] >= .34999 for row in samples))
        self.assertTrue(all(b["monotonic_seconds"] - a["monotonic_seconds"] >= .35 for a, b in zip(samples, samples[1:])))

    def test_output_bound_keeps_complete_summary_and_stops_early(self):
        _, summary, rows = self.run_collect(source=Source(self.clock, payload={"fixture_payload": "x" * 2000}), maximum=8192)
        self.assertEqual(summary["reason"], "output_limit")
        self.assertFalse(summary["duration_completed"])
        self.assertEqual(rows[-1]["event"], "STOPPED")
        self.assertLessEqual((self.root / "telemetry.jsonl").stat().st_size, 8192)

    def test_exclusive_output_and_symlink_refusal_preserve_existing_bytes(self):
        output = self.write(self.root / "telemetry.jsonl", "old")
        with self.assertRaises(FileExistsError):
            self.run_collect()
        self.assertEqual(output.read_text(), "old")
        output.unlink()
        target = self.write(self.root / "target", "private")
        output.symlink_to(target)
        with self.assertRaises(FileExistsError):
            self.run_collect()
        self.assertEqual(target.read_text(), "private")

    def test_invalid_options_refuse_before_output_creation(self):
        with self.assertRaises(ValueError):
            self.run_collect(duration=901)
        self.assertFalse((self.root / "telemetry.jsonl").exists())

    def test_interruption_has_summary_and_never_claims_duration_completed(self):
        _, summary, rows = self.run_collect(stop_requested=lambda: self.clock.now >= 100.3)
        self.assertEqual(summary["reason"], "interrupted")
        self.assertFalse(summary["duration_completed"])
        self.assertLess(summary["elapsed_seconds"], 1)
        self.assertEqual(rows[-1], summary)

    def test_read_error_records_type_only_and_closes_output(self):
        source = Source(self.clock)
        source.sample = lambda now: (_ for _ in ()).throw(RuntimeError("private data must not be emitted"))
        output, summary, _ = self.run_collect(source=source)
        self.assertEqual(summary["reason"], "collector_error")
        self.assertEqual(summary["error_type"], "RuntimeError")
        self.assertNotIn("private", output.read_text())


if __name__ == "__main__":
    unittest.main()
