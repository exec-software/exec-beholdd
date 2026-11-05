import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import thermal_guard as guard


class FakeProcess:
    """Stands in for `subprocess.Popen` without spawning anything real."""

    def __init__(self, exit_after=None, exit_code=0, pid=4321, stubborn=False):
        self.pid = pid
        self.stubborn = stubborn
        self.terminated = False
        self.kill_called = False
        self._polls = 0
        self._exit_after = exit_after
        self._exit_code = exit_code
        self._natural_exit_code = None
        self._terminated_code = None
        self._killed_code = None

    def poll(self):
        self._polls += 1
        if self._killed_code is not None:
            return self._killed_code
        if self._terminated_code is not None:
            return self._terminated_code
        if self._natural_exit_code is not None:
            return self._natural_exit_code
        if self._exit_after is not None and self._polls >= self._exit_after:
            self._natural_exit_code = self._exit_code
            return self._natural_exit_code
        return None

    def terminate(self):
        self.terminated = True
        if not self.stubborn:
            self._terminated_code = -15

    def kill(self):
        self.kill_called = True
        self._killed_code = -9

    def signal_group(self, requested):
        if requested == signal.SIGTERM:
            self.terminate()
        elif requested == signal.SIGKILL:
            self.kill()
        else:
            raise AssertionError(f"unexpected signal {requested}")

    def group_alive(self):
        return self._terminated_code is None and self._killed_code is None

    def wait(self, timeout=None):
        if self._killed_code is not None:
            return self._killed_code
        if self._terminated_code is not None:
            return self._terminated_code
        if self._natural_exit_code is not None:
            return self._natural_exit_code
        raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout)


def temps(*values):
    return iter(values).__next__


def after_preflight(callback):
    """Supply a safe preflight sample before the running-only test callback."""
    sampled = False

    def read():
        nonlocal sampled
        if not sampled:
            sampled = True
            return 60.0
        return callback()

    return read


class ThermalGuardTests(unittest.TestCase):
    def assert_preflight_refused(self, read_temp):
        started = []

        def forbidden_start(*args, **kwargs):
            started.append((args, kwargs))
            raise AssertionError("unsafe child must not start")

        report = guard.supervise(["must-not-run"], read_temp=read_temp, popen=forbidden_start)
        self.assertEqual(started, [])
        self.assertFalse(report["ok"])
        self.assertTrue(report["aborted_for_heat"])
        self.assertFalse(report["process_started"])
        self.assertEqual(report["abort_phase"], "preflight")
        self.assertEqual(report["samples"], 1)
        self.assertIsNone(report["pid"])
        self.assertIsNone(report["exit_code"])
        self.assertFalse(report["residual_group_cleaned"])
        return report

    def test_hot_preflight_never_spawns_even_at_the_exact_threshold(self):
        for temperature in [90.0, 95.0, 100.0]:
            with self.subTest(temperature=temperature):
                report = self.assert_preflight_refused(temps(temperature))
                self.assertEqual(report["max_tctl_c"], temperature)
                self.assertIn("critical threshold", report["abort_reason"])

    def test_invalid_preflight_readings_never_spawn_or_enter_json_as_nonfinite(self):
        for value in [float("nan"), float("inf"), -float("inf"), -0.001, 125.001, "60", None, True]:
            with self.subTest(value=value):
                report = self.assert_preflight_refused(temps(value))
                self.assertIsNone(report["max_tctl_c"])
                self.assertIn("invalid", report["abort_reason"])
                json.dumps(report, allow_nan=False)

    def test_unreadable_or_malformed_sensor_files_prevent_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            sensor = Path(directory) / "temp1_input"
            self.assert_preflight_refused(lambda: guard.read_tctl_c(sensor))
            for value in ["", "garbage", "nan", "inf", "60.5", "-1000", "126000"]:
                with self.subTest(value=value):
                    sensor.write_text(value, encoding="utf-8")
                    self.assert_preflight_refused(lambda: guard.read_tctl_c(sensor))

    def test_invalid_running_readings_terminate_the_child_instead_of_silent_safety(self):
        for value in [float("nan"), float("inf"), -1.0, 126.0, "invalid"]:
            with self.subTest(value=value):
                process = FakeProcess()
                report = guard.supervise(
                    ["irrelevant"], read_temp=temps(60.0, value),
                    popen=lambda command, **options: process,
                    signal_group=process.signal_group, group_alive=process.group_alive,
                )
                self.assertTrue(process.terminated)
                self.assertTrue(report["process_started"])
                self.assertEqual(report["abort_phase"], "running")
                self.assertEqual(report["max_tctl_c"], 60.0)
                json.dumps(report, allow_nan=False)

    def test_preflight_happens_before_even_an_immediately_finished_child(self):
        order = []
        process = FakeProcess(exit_after=1)

        def read():
            order.append("read")
            return 60.0

        def start(*args, **kwargs):
            order.append("start")
            return process

        report = guard.supervise(["instant"], read_temp=read, popen=start, group_alive=lambda: False)
        self.assertEqual(order, ["read", "start"])
        self.assertTrue(report["process_started"])
        self.assertEqual(report["samples"], 1)
        self.assertEqual(report["exit_code"], 0)

    def test_cli_preflight_refusal_has_nonzero_exit_and_no_invented_child(self):
        output = io.StringIO()
        with patch.object(guard, "find_k10temp_input", return_value="unused"), patch.object(
            guard, "read_tctl_c", return_value=100.0
        ), patch.object(sys, "stdout", output):
            with self.assertRaises(SystemExit) as result:
                guard.main(["--", "must-not-run"])
        self.assertEqual(result.exception.code, 1)
        report = json.loads(output.getvalue())
        self.assertFalse(report["process_started"])
        self.assertEqual(report["abort_phase"], "preflight")
        self.assertIsNone(report["pid"])

    def test_child_that_finishes_on_its_own_is_never_touched(self):
        process = FakeProcess(exit_after=3, exit_code=0)
        popen_options = []

        def start(command, **options):
            popen_options.append(options)
            return process

        report = guard.supervise(
            ["irrelevant"],
            read_temp=temps(60.0, 60.0, 61.0),
            sleep=lambda seconds: None,
            popen=start,
        )
        self.assertEqual(report["ok"], True)
        self.assertEqual(report["aborted_for_heat"], False)
        self.assertIsNone(report["abort_reason"])
        self.assertEqual(report["exit_code"], 0)
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["max_tctl_c"], 61.0)
        self.assertFalse(process.terminated)
        self.assertFalse(process.kill_called)
        self.assertEqual(popen_options, [{"start_new_session": True}])

    def test_a_nonzero_child_exit_is_reported_without_being_treated_as_a_thermal_event(self):
        process = FakeProcess(exit_after=2, exit_code=7)
        report = guard.supervise(
            ["irrelevant"],
            read_temp=temps(60.0, 60.0),
            sleep=lambda seconds: None,
            popen=lambda command, **options: process,
        )
        self.assertTrue(report["ok"])
        self.assertFalse(report["aborted_for_heat"])
        self.assertEqual(report["exit_code"], 7)

    def test_launcher_exit_cleans_surviving_group_and_preserves_its_status(self):
        process = FakeProcess(exit_after=1, exit_code=7)
        descendants_alive = True
        signals = []

        def signal_group(requested):
            nonlocal descendants_alive
            signals.append(requested)
            if requested == signal.SIGKILL:
                descendants_alive = False

        report = guard.supervise(
            ["launcher", "model"],
            read_temp=temps(60.0),
            sleep=lambda seconds: None,
            popen=lambda command, **options: process,
            signal_group=signal_group,
            group_alive=lambda: descendants_alive,
        )

        self.assertTrue(report["ok"])
        self.assertFalse(report["aborted_for_heat"])
        self.assertTrue(report["residual_group_cleaned"])
        self.assertEqual(report["exit_code"], 7)
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertFalse(descendants_alive)

    def test_reaching_the_critical_threshold_terminates_the_child(self):
        process = FakeProcess(exit_after=None)
        report = guard.supervise(
            ["irrelevant"],
            critical_c=90.0,
            read_temp=temps(70.0, 80.0, 91.0),
            sleep=lambda seconds: None,
            popen=lambda command, **options: process,
            signal_group=process.signal_group,
            group_alive=process.group_alive,
        )
        self.assertFalse(report["ok"])
        self.assertTrue(report["aborted_for_heat"])
        self.assertIn("91.0", report["abort_reason"])
        self.assertIn("90.0", report["abort_reason"])
        self.assertEqual(report["samples"], 3)
        self.assertEqual(report["max_tctl_c"], 91.0)
        self.assertTrue(process.terminated)
        self.assertFalse(process.kill_called)
        self.assertEqual(report["exit_code"], -15)

    def test_a_child_that_ignores_termination_is_killed_after_the_grace_period(self):
        process = FakeProcess(exit_after=None, stubborn=True)
        report = guard.supervise(
            ["irrelevant"],
            critical_c=90.0,
            grace_seconds=0.01,
            read_temp=temps(60.0, 95.0),
            sleep=lambda seconds: None,
            popen=lambda command, **options: process,
            signal_group=process.signal_group,
            group_alive=process.group_alive,
        )
        self.assertTrue(report["aborted_for_heat"])
        self.assertTrue(process.terminated)
        self.assertTrue(process.kill_called)
        self.assertEqual(report["exit_code"], -9)

    def test_an_unreadable_sensor_aborts_rather_than_assuming_a_safe_temperature(self):
        process = FakeProcess(exit_after=None)

        def failing_read():
            raise OSError("sensor vanished")

        report = guard.supervise(
            ["irrelevant"],
            read_temp=after_preflight(failing_read),
            sleep=lambda seconds: None,
            popen=lambda command, **options: process,
            signal_group=process.signal_group,
            group_alive=process.group_alive,
        )
        self.assertTrue(report["aborted_for_heat"])
        self.assertIn("unreadable", report["abort_reason"])
        self.assertTrue(process.terminated)
        self.assertEqual(report["max_tctl_c"], 60.0)

    def test_keyboard_interrupt_terminates_the_process_group_before_propagating(self):
        process = FakeProcess(exit_after=None)

        def interrupt():
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            guard.supervise(
                ["irrelevant"],
                read_temp=after_preflight(interrupt),
                sleep=lambda seconds: None,
                popen=lambda command, **options: process,
                signal_group=process.signal_group,
                group_alive=process.group_alive,
            )

        self.assertTrue(process.terminated)
        self.assertFalse(process.kill_called)

    def test_unexpected_exception_terminates_the_process_group_before_propagating(self):
        process = FakeProcess(exit_after=None)

        def fail_unexpectedly():
            raise RuntimeError("unexpected probe failure")

        with self.assertRaisesRegex(RuntimeError, "unexpected probe failure"):
            guard.supervise(
                ["irrelevant"],
                read_temp=after_preflight(fail_unexpectedly),
                sleep=lambda seconds: None,
                popen=lambda command, **options: process,
                signal_group=process.signal_group,
                group_alive=process.group_alive,
            )

        self.assertTrue(process.terminated)
        self.assertFalse(process.kill_called)

    def test_preflight_log_write_failure_prevents_launch(self):
        process = FakeProcess(exit_after=None)
        with tempfile.TemporaryDirectory() as directory:
            missing_log = Path(directory) / "missing" / "thermal.jsonl"
            with self.assertRaises(FileNotFoundError):
                guard.supervise(
                    ["irrelevant"],
                    read_temp=temps(60.0),
                    sleep=lambda seconds: None,
                    popen=lambda command, **options: process,
                    signal_group=process.signal_group,
                    group_alive=process.group_alive,
                    log_path=missing_log,
                )

        self.assertFalse(process.terminated)
        self.assertFalse(process.kill_called)

    def test_running_log_write_failure_still_terminates_the_process_group(self):
        process = FakeProcess(exit_after=None)
        with tempfile.TemporaryDirectory() as directory:
            log_directory = Path(directory) / "logs"
            log_directory.mkdir()
            log_path = log_directory / "thermal.jsonl"

            def remove_log_directory():
                log_path.unlink()
                log_directory.rmdir()
                return 60.0

            with self.assertRaises(FileNotFoundError):
                guard.supervise(
                    ["irrelevant"],
                    read_temp=after_preflight(remove_log_directory),
                    sleep=lambda seconds: None,
                    popen=lambda command, **options: process,
                    signal_group=process.signal_group,
                    group_alive=process.group_alive,
                    log_path=log_path,
                )
        self.assertTrue(process.terminated)

    def test_thresholds_and_command_are_validated_before_anything_runs(self):
        with self.assertRaises(guard.ThermalGuardError):
            guard.supervise([], read_temp=temps(60.0))
        with self.assertRaises(guard.ThermalGuardError):
            guard.supervise(["x"], warn_c=90.0, critical_c=75.0, read_temp=temps(60.0))
        with self.assertRaises(guard.ThermalGuardError):
            guard.supervise(["x"], warn_c=95.0, critical_c=100.0, read_temp=temps(60.0))
        for bad_poll in [0.0, -1.0, float("nan"), float("inf")]:
            with self.assertRaises(guard.ThermalGuardError):
                guard.supervise(
                    ["x"], poll_seconds=bad_poll, read_temp=temps(60.0)
                )
        for bad_grace in [0.0, -1.0, float("nan"), float("inf")]:
            with self.assertRaises(guard.ThermalGuardError):
                guard.supervise(
                    ["x"], grace_seconds=bad_grace, read_temp=temps(60.0)
                )

    def test_sigterm_is_raised_into_the_fail_closed_cleanup_path(self):
        with self.assertRaises(guard.ThermalGuardTermination) as caught:
            guard.raise_for_termination(signal.SIGTERM, None)
        self.assertEqual(caught.exception.signum, signal.SIGTERM)

    def test_log_path_receives_one_json_line_per_sample(self):
        process = FakeProcess(exit_after=3, exit_code=0)
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "thermal.jsonl"
            guard.supervise(
                ["irrelevant"],
                warn_c=50.0,
                read_temp=temps(60.0, 60.0, 61.0),
                sleep=lambda seconds: None,
                popen=lambda command, **options: process,
                log_path=log_path,
            )
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 3)
            first = json.loads(lines[0])
            self.assertEqual(first["tctl_c"], 60.0)
            self.assertTrue(first["sensor_ok"])
            self.assertTrue(first["warn"])
            self.assertEqual(first["phase"], "preflight")
            self.assertEqual(json.loads(lines[1])["phase"], "running")

    def test_a_surviving_grandchild_is_killed_after_the_direct_child_exits(self):
        process = FakeProcess(exit_after=None)
        descendants_alive = True
        signals = []

        def signal_group(requested):
            nonlocal descendants_alive
            signals.append(requested)
            if requested == signal.SIGTERM:
                process.terminate()
            elif requested == signal.SIGKILL:
                descendants_alive = False

        report = guard.supervise(
            ["runner", "model"],
            critical_c=90.0,
            read_temp=temps(60.0, 95.0),
            sleep=lambda seconds: None,
            popen=lambda command, **options: process,
            signal_group=signal_group,
            group_alive=lambda: descendants_alive,
        )

        self.assertTrue(report["aborted_for_heat"])
        self.assertEqual(signals, [signal.SIGTERM, signal.SIGKILL])
        self.assertFalse(descendants_alive)
        self.assertEqual(report["exit_code"], -15)

    def test_real_descendant_cannot_continue_after_a_thermal_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "ready"
            survived = root / "survived"
            grandchild_code = "\n".join(
                [
                    "import pathlib",
                    "import time",
                    "time.sleep(0.5)",
                    f"pathlib.Path({str(survived)!r}).write_text('survived', encoding='utf-8')",
                    "time.sleep(30)",
                ]
            )
            parent_code = "\n".join(
                [
                    "import os",
                    "import pathlib",
                    "import subprocess",
                    "import sys",
                    "import time",
                    f"child = subprocess.Popen([sys.executable, '-c', {grandchild_code!r}])",
                    f"pathlib.Path({str(ready)!r}).write_text(",
                    "    f'{os.getpid()} {child.pid}', encoding='utf-8'",
                    ")",
                    "time.sleep(30)",
                ]
            )

            def hot_after_tree_starts():
                deadline = time.monotonic() + 5.0
                while not ready.exists():
                    if time.monotonic() >= deadline:
                        raise RuntimeError("supervised process tree did not start")
                    time.sleep(0.01)
                return 95.0

            process_group = None
            try:
                report = guard.supervise(
                    [sys.executable, "-c", parent_code],
                    critical_c=90.0,
                    grace_seconds=0.5,
                    read_temp=after_preflight(hot_after_tree_starts),
                    sleep=lambda seconds: None,
                )
                process_group = report["pid"]
                self.assertTrue(report["aborted_for_heat"])
                time.sleep(0.7)
                self.assertFalse(
                    survived.exists(),
                    "grandchild continued running after the thermal process-group abort",
                )
            finally:
                if process_group is None and ready.exists():
                    process_group = int(ready.read_text(encoding="utf-8").split()[0])
                if process_group is not None:
                    try:
                        os.killpg(process_group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_real_descendant_is_cleaned_when_its_launcher_exits_normally(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = root / "ready"
            survived = root / "survived"
            descendant_code = "\n".join(
                [
                    "import pathlib",
                    "import time",
                    "time.sleep(0.5)",
                    f"pathlib.Path({str(survived)!r}).write_text('survived', encoding='utf-8')",
                    "time.sleep(30)",
                ]
            )
            launcher_code = "\n".join(
                [
                    "import os",
                    "import pathlib",
                    "import subprocess",
                    "import sys",
                    f"child = subprocess.Popen([sys.executable, '-c', {descendant_code!r}])",
                    f"pathlib.Path({str(ready)!r}).write_text(",
                    "    f'{os.getpid()} {child.pid}', encoding='utf-8'",
                    ")",
                ]
            )

            process_group = None
            try:
                report = guard.supervise(
                    [sys.executable, "-c", launcher_code],
                    grace_seconds=0.5,
                    poll_seconds=0.01,
                    read_temp=lambda: 60.0,
                )
                process_group = report["pid"]
                self.assertTrue(ready.exists(), "launcher did not create its descendant")
                self.assertEqual(report["exit_code"], 0)
                self.assertTrue(report["residual_group_cleaned"])
                time.sleep(0.7)
                self.assertFalse(
                    survived.exists(),
                    "descendant continued running after its launcher exited",
                )
            finally:
                if process_group is None and ready.exists():
                    process_group = int(ready.read_text(encoding="utf-8").split()[0])
                if process_group is not None:
                    try:
                        os.killpg(process_group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_split_command_requires_an_explicit_separator(self):
        with self.assertRaises(guard.ThermalGuardError):
            guard.split_command(["--warn-c", "80"])
        own, command = guard.split_command(["--warn-c", "80", "--", "decode", "a.gguf"])
        self.assertEqual(own, ["--warn-c", "80"])
        self.assertEqual(command, ["decode", "a.gguf"])

    def test_parse_arguments_rejects_an_empty_command(self):
        with self.assertRaises(guard.ThermalGuardError):
            guard.parse_arguments(["--warn-c", "80", "--"])

    def test_read_tctl_c_converts_millidegrees(self):
        with tempfile.TemporaryDirectory() as directory:
            sensor = Path(directory) / "temp1_input"
            sensor.write_text("67625\n", encoding="utf-8")
            self.assertEqual(guard.read_tctl_c(sensor), 67.625)

    def test_find_k10temp_input_locates_the_right_hwmon_entry_and_ignores_others(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hwmon0").mkdir()
            (root / "hwmon0" / "name").write_text("acpitz\n", encoding="utf-8")
            (root / "hwmon1").mkdir()
            (root / "hwmon1" / "name").write_text("k10temp\n", encoding="utf-8")
            found = guard.find_k10temp_input(hwmon_root=root)
            self.assertEqual(found, root / "hwmon1" / "temp1_input")

    def test_find_k10temp_input_fails_loudly_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "hwmon0").mkdir()
            (root / "hwmon0" / "name").write_text("acpitz\n", encoding="utf-8")
            with self.assertRaises(guard.ThermalGuardError):
                guard.find_k10temp_input(hwmon_root=root)


if __name__ == "__main__":
    unittest.main()
