"""Hardware-free checks for the delivered watcher; no sudo or real sysfs access."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parent / "hag_crash_watch.py"
sys.dont_write_bytecode = True
spec = importlib.util.spec_from_file_location("crash_watch", SCRIPT)
watcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watcher)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.sysfs = self.root / "sysfs"
        self.sysfs.mkdir()
        (self.sysfs / "disabled").write_text("0\n")
        self.output = watcher.output_directory(os.getgid(), self.root)

    def data(self, value):
        entry = self.sysfs / "devcd0"
        entry.mkdir()
        (entry / "data").write_bytes(value)
        return entry

    def test_complete_capture_hash_permissions_and_source_unchanged(self):
        payload = b"GPU fault evidence\x00\xff"
        entry = self.data(payload)
        before = (entry / "data").stat()
        record = watcher.capture(entry, self.output, os.getgid(), cap=64)
        self.assertTrue(record["complete"])
        self.assertEqual(record["sha256"], hashlib.sha256(payload).hexdigest())
        self.assertEqual(record["bytes"], len(payload))
        self.assertEqual((entry / "data").read_bytes(), payload)
        self.assertEqual((entry / "data").stat().st_mtime_ns, before.st_mtime_ns)
        self.assertEqual((self.sysfs / "disabled").read_text(), "0\n")
        self.assertEqual((self.output / "devcd0.bin").read_bytes(), payload)
        self.assertEqual(json.loads((self.output / "devcd0.json").read_text()), record)
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o750)
        for path in self.output.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o640)
            self.assertEqual(path.stat().st_gid, os.getgid())

    def test_truncation_detects_extra_byte_and_stores_only_cap(self):
        entry = self.data(bytes(range(100)))
        record = watcher.capture(entry, self.output, os.getgid(), cap=16)
        self.assertFalse(record["complete"])
        self.assertTrue(record["truncated"])
        self.assertEqual(record["observed_bytes"], 17)
        self.assertEqual(record["bytes"], 16)
        self.assertEqual((self.output / "devcd0.bin").read_bytes(), bytes(range(16)))
        self.assertEqual((entry / "data").stat().st_size, 100)

    def test_exact_cap_is_complete(self):
        record = watcher.capture(self.data(b"x" * 16), self.output, os.getgid(), cap=16)
        self.assertTrue(record["complete"])
        self.assertFalse(record["truncated"])

    def test_blocked_fifo_times_out_without_writer_or_sysfs(self):
        entry = self.sysfs / "devcd0"
        entry.mkdir()
        os.mkfifo(entry / "data")
        record = watcher.capture(entry, self.output, os.getgid(), cap=16, seconds=0.1)
        self.assertTrue(record["timed_out"])
        self.assertFalse(record["complete"])
        self.assertEqual(record["bytes"], 0)
        self.assertTrue(stat.S_ISFIFO((entry / "data").stat().st_mode))

    def test_missing_data_records_capture_error(self):
        entry = self.sysfs / "devcd0"
        entry.mkdir()
        record = watcher.capture(entry, self.output, os.getgid(), cap=16)
        self.assertFalse(record["complete"])
        self.assertNotEqual(record["returncode"], 0)
        self.assertTrue(record["error"])

    def test_exclusive_output_refuses_existing_file_and_symlink(self):
        entry = self.data(b"new")
        target = self.output / "devcd0.bin"
        target.write_bytes(b"old")
        with self.assertRaises(FileExistsError):
            watcher.capture(entry, self.output, os.getgid())
        self.assertEqual(target.read_bytes(), b"old")
        target.unlink()
        target.symlink_to(entry / "data")
        with self.assertRaises(FileExistsError):
            watcher.capture(entry, self.output, os.getgid())
        self.assertEqual((entry / "data").read_bytes(), b"new")

    def test_startup_root_disabled_and_stale_gates(self):
        with self.assertRaisesRegex(watcher.WatchError, "requires root"):
            watcher.startup_gate(self.sysfs, euid=1000)
        watcher.startup_gate(self.sysfs, euid=0)
        (self.sysfs / "disabled").write_text("1\n")
        with self.assertRaisesRegex(watcher.WatchError, "disabled"):
            watcher.startup_gate(self.sysfs, euid=0)
        self.assertEqual((self.sysfs / "disabled").read_text(), "1\n")
        (self.sysfs / "disabled").write_text("0\n")
        self.data(b"stale")
        with self.assertRaisesRegex(watcher.WatchError, "existing"):
            watcher.startup_gate(self.sysfs, euid=0)

    def test_stop_file_is_only_read_and_armed_event_is_parseable(self):
        stop = self.root / "stop"
        stop.write_text("stop marker")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = watcher.watch(self.output, os.getgid(), 1, stop, sysfs=self.sysfs)
        events = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(result, 0)
        self.assertEqual(events[0]["event"], "ARMED")
        self.assertEqual(events[0]["pid"], os.getpid())
        self.assertTrue(events[0]["timestamp"])
        self.assertEqual(events[1]["event"], "HEARTBEAT")
        self.assertEqual(events[-1]["event"], "STOPPED")
        self.assertEqual(stop.read_text(), "stop marker")

    def test_maximum_four_captures(self):
        for i in range(5):
            entry = self.sysfs / f"devcd{i}"
            entry.mkdir()
            (entry / "data").write_bytes(b"evidence")
        with contextlib.redirect_stdout(io.StringIO()):
            result = watcher.watch(self.output, os.getgid(), 2, sysfs=self.sysfs)
        self.assertEqual(result, 1)
        self.assertEqual(len(list(self.output.glob("*.bin"))), 4)
        self.assertTrue((self.sysfs / "devcd4" / "data").exists())

    def test_cli_rejects_arbitrary_sysfs_and_excessive_duration(self):
        for arguments in (["--sysfs", str(self.sysfs)], ["--seconds", "1501"], ["--seconds", "0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                watcher.main(arguments)

    def test_cli_keeps_default_and_accepts_finite_larger_profile_lease(self):
        for arguments, seconds in (([], 600), (["--seconds", "1500"], 1500)):
            with patch.object(watcher, "startup_gate"), \
                    patch.object(watcher, "sudo_group", return_value=os.getgid()), \
                    patch.object(watcher, "output_directory", return_value=self.output), \
                    patch.object(watcher, "watch", return_value=0) as watch:
                self.assertEqual(watcher.main(arguments), 0)
                watch.assert_called_once_with(self.output, os.getgid(), seconds, None)

    def test_requires_sudo_identity(self):
        self.assertEqual(watcher.sudo_group({"SUDO_UID": "1000", "SUDO_GID": "1000"}), 1000)
        for environment in ({}, {"SUDO_UID": "0", "SUDO_GID": "0"}, {"SUDO_UID": "1000", "SUDO_GID": "-1"}):
            with self.assertRaises(watcher.WatchError):
                watcher.sudo_group(environment)

    def test_pkexec_identity_uses_system_user_group(self):
        account = watcher.pwd.struct_passwd(("tester", "x", 1001, 1002, "", "/unused", "/bin/false"))
        with patch.object(watcher.pwd, "getpwuid", return_value=account) as lookup:
            self.assertEqual(watcher.sudo_group({"PKEXEC_UID": "1001"}), 1002)
            lookup.assert_called_once_with(1001)


if __name__ == "__main__":
    unittest.main()
