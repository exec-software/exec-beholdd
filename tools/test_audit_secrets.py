#!/usr/bin/env python3
"""Small, offline tests for the audit helper; Gitleaks itself is not required."""

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "audit_secrets", Path(__file__).with_name("audit_secrets.py")
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repository = self.root / "repository"
        self.repository.mkdir()
        self.git("init", "--quiet", "--initial-branch=main", "--template=")
        self.script = self.repository / "tools/audit_secrets.py"
        self.script.parent.mkdir()
        self.script.write_text("# fixture\n")
        self.config = self.repository / ".gitleaks.toml"
        self.config.write_text("[extend]\nuseDefault = true\n")
        self.git("add", ".")
        self.git("commit", "--quiet", "-m", "fixture")
        self.reports = self.root / "reports"
        self.exported = self.root / "exported"
        self.exported.mkdir()

    def git(self, *arguments):
        return subprocess.run(
            [
                "git", "-C", str(self.repository),
                "-c", "user.name=Audit tests",
                "-c", "user.email=audit@example.invalid",
                "-c", "commit.gpgsign=false", *arguments,
            ],
            check=True, capture_output=True,
        ).stdout

    def main(self, scanner=None, *, version=None, report_dir=None):
        output = io.StringIO()
        original_check_output = subprocess.check_output

        def check_output(command, *args, **kwargs):
            if command == ["fixture-scanner", "version"]:
                return version if version is not None else audit.SCANNER_VERSION + "\n"
            return original_check_output(command, *args, **kwargs)

        previous_umask = os.umask(0o077)
        try:
            with (
                patch.object(audit, "__file__", str(self.script)),
                patch.object(sys, "argv", [
                    str(self.script), "--scanner", "fixture-scanner",
                    "--report-dir", str(report_dir or self.reports),
                ]),
                patch.object(audit.subprocess, "check_output", side_effect=check_output),
                patch.object(audit, "run_scan", side_effect=scanner) as scans,
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(output),
            ):
                try:
                    status = audit.main()
                except SystemExit as error:
                    status = error.code
        finally:
            os.umask(previous_umask)
        return status, output.getvalue(), scans

    def test_snapshot_includes_tracked_and_untracked_but_not_untracked_ignored(self):
        (self.repository / ".gitignore").write_text("*.local\n")
        (self.repository / "tracked.local").write_text("tracked\n")
        self.git("add", "--force", "tracked.local")
        (self.repository / "ignored.local").write_text("ignored\n")
        (self.repository / "untracked.txt").write_text("new\n")
        records = audit.snapshot(self.repository, self.exported)
        names = {record["file"] for record in records}
        self.assertIn("tracked.local", names)
        self.assertIn("untracked.txt", names)
        self.assertNotIn("ignored.local", names)
        self.assertEqual((self.exported / "untracked.txt").read_text(), "new\n")
        for record in records:
            self.assertEqual(record["bytes"], (self.exported / record["file"]).stat().st_size)
            self.assertEqual(len(record["sha256"]), 64)

    def test_tracked_deletion_is_explicitly_recorded(self):
        self.script.unlink()
        records = audit.snapshot(self.repository, self.exported)
        self.assertIn({"file": "tools/audit_secrets.py", "state": "deleted-in-worktree"}, records)

    def test_file_and_parent_symlinks_fail_closed(self):
        outside = self.root / "outside.txt"
        outside.write_text("private fixture\n")
        link = self.repository / "link.txt"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            audit.snapshot(self.repository, self.exported)
        link.unlink()
        self.script.unlink()
        self.script.parent.rmdir()
        self.script.parent.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            audit.snapshot(self.repository, self.exported)
        self.assertFalse((self.exported / "link.txt").exists())

    def test_submodule_is_not_silently_omitted(self):
        revision = self.git("rev-parse", "HEAD").decode().strip()
        self.git("update-index", "--add", "--cacheinfo", f"160000,{revision},nested")
        (self.repository / "nested").mkdir()
        with self.assertRaises(ValueError):
            audit.snapshot(self.repository, self.exported)

    def test_unexpected_missing_entry_and_unsafe_paths_fail_closed(self):
        for name in (b"missing.txt", b"../outside.txt", b"/absolute.txt"):
            with self.subTest(name=name), patch.object(audit, "git", side_effect=[name + b"\0", b""]):
                with self.assertRaises(ValueError):
                    audit.snapshot(self.repository, self.exported)

    def test_scanner_commands_cover_full_refs_index_and_disable_inline_exceptions(self):
        self.reports.mkdir()
        for mode in ("dir", "git", "staged"):
            commands = []

            def run(command, **kwargs):
                commands.append(command)
                (self.reports / f"{mode}.json").write_text("[]")
                return subprocess.CompletedProcess(command, 0)

            with patch.object(audit.subprocess, "run", side_effect=run):
                self.assertEqual(audit.run_scan("scanner", mode, self.repository, self.config, self.reports), (0, 0))
            command = commands[0]
            self.assertEqual(command[1], "git" if mode == "staged" else mode)
            for flag in ("--redact=100", "--ignore-gitleaks-allow", "--gitleaks-ignore-path=/dev/null"):
                self.assertIn(flag, command)
            self.assertEqual("--log-opts=--all --full-history" in command, mode == "git")
            self.assertEqual("--staged" in command, mode == "staged")
            self.assertEqual(command[command.index("--config") + 1], str(self.config))

    def test_failed_missing_malformed_and_inconsistent_scanner_reports_fail_closed(self):
        cases = [(2, "[]"), (0, None), (0, "bad json"), (0, "{}"), (0, '[{"RuleID":"fixture"}]')]
        for index, (status, contents) in enumerate(cases):
            reports = self.root / f"failed-{index}"
            reports.mkdir()

            def run(command, **kwargs):
                if contents is not None:
                    (reports / "dir.json").write_text(contents)
                return subprocess.CompletedProcess(command, status)

            with self.subTest(status=status, contents=contents), patch.object(audit.subprocess, "run", side_effect=run):
                with self.assertRaises((RuntimeError, ValueError)):
                    audit.run_scan("scanner", "dir", self.repository, self.config, reports)

    def test_staged_bytes_are_scanned_even_after_worktree_is_cleaned(self):
        marker = "synthetic-audit-marker"
        staged = self.repository / "staged.txt"
        staged.write_text(marker)
        self.git("add", "staged.txt")
        staged.write_text("clean working tree bytes")

        def scan(scanner, mode, target, config, reports):
            if mode == "dir":
                self.assertEqual((target / "staged.txt").read_text(), "clean working tree bytes")
            if mode == "staged":
                self.assertIn(marker.encode(), self.git("diff", "--cached"))
                return 1, 1
            return 0, 0

        status, output, scans = self.main(scan)
        self.assertEqual(status, 1)
        self.assertEqual([call.args[1] for call in scans.call_args_list], ["dir", "git", "staged"])
        summary = json.loads(output)
        self.assertEqual(summary["staged_findings"], 1)
        self.assertEqual(summary["tree_findings"], 0)
        self.assertNotIn(marker, output)
        self.assertEqual(self.reports.stat().st_mode & 0o777, 0o700)
        for path in self.reports.iterdir():
            self.assertTrue(path.is_file(), "temporary source snapshots must be removed")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_version_and_config_are_required_before_scanning(self):
        status, _, scans = self.main(version="0.0.0\n")
        self.assertEqual(status, 2)
        scans.assert_not_called()
        self.assertFalse(self.reports.exists())
        self.config.unlink()
        status, _, scans = self.main()
        self.assertEqual(status, 2)
        scans.assert_not_called()
        self.assertFalse(self.reports.exists())

    def test_existing_or_checkout_report_directory_is_rejected(self):
        for reports in (self.root, self.repository / "reports"):
            status, _, scans = self.main(report_dir=reports)
            self.assertEqual(status, 2)
            scans.assert_not_called()
        self.assertFalse((self.repository / "reports").exists())

    def test_failure_does_not_echo_diagnostic_source_values(self):
        def fail(*args):
            raise RuntimeError("synthetic-private-diagnostic")

        status, output, _ = self.main(fail)
        self.assertEqual(status, 2)
        self.assertNotIn("synthetic-private-diagnostic", output)
        self.assertFalse((self.reports / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
