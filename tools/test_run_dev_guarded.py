"""Refusal tests; never create memory pressure or launch systemd during unit tests."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_dev_guarded as guard


class DevelopmentGuardTests(unittest.TestCase):
    def test_low_available_memory_never_starts_command(self):
        with patch.object(guard, "available_memory", return_value=guard.MEMORY_MAX), \
             patch.object(guard.subprocess, "run") as run:
            self.assertEqual(guard.main(["cargo", "test"]), 1)
            run.assert_not_called()

    def test_missing_manager_does_not_fall_back(self):
        with patch.object(guard, "available_memory", return_value=32 * 1024**3), \
             patch.object(guard.shutil, "which", return_value=None), \
             patch.object(guard.subprocess, "run") as run:
            self.assertEqual(guard.main(["cargo", "test"]), 1)
            run.assert_not_called()

    def test_killed_command_is_not_restarted_or_reported_successful(self):
        with patch.object(guard, "available_memory", return_value=32 * 1024**3), \
             patch.object(guard.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.object(guard.subprocess, "run") as run:
            run.return_value.returncode = 137
            self.assertEqual(guard.main(["cargo", "test"]), 137)
            self.assertEqual(run.call_count, 1)

    def test_literal_command_arguments_are_not_expanded_by_systemd(self):
        arguments = ["python3", "-c", "print('${HOME} $PATH')"]
        with patch.object(guard, "available_memory", return_value=32 * 1024**3), \
             patch.object(guard.shutil, "which", return_value="/usr/bin/systemd-run"), \
             patch.object(guard.subprocess, "run") as run:
            run.return_value.returncode = 0
            self.assertEqual(guard.main(arguments), 0)
            command = run.call_args.args[0]
            self.assertIn("--expand-environment=no", command)
            self.assertEqual(command[-len(arguments):], arguments)

    def test_missing_or_malformed_available_memory_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "meminfo"
            for contents in ["MemFree: 999999 kB\n", "MemAvailable: -1 kB\n",
                             "MemAvailable: 123 MB\n"]:
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    guard.available_memory(path)
            path.write_text("MemFree: 1 kB\nMemAvailable: 123 kB\n")
            self.assertEqual(guard.available_memory(path), 123 * 1024)


if __name__ == "__main__":
    unittest.main()
