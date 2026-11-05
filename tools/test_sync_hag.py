#!/usr/bin/env python3
"""Hardware-free tests for the standalone HAG source snapshot contract."""

import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("sync_hag", Path(__file__).with_name("sync-hag.py"))
sync_hag = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sync_hag)


class SyncHagTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "hag"
        self.repository = self.root / "beholdd"
        self.source.mkdir()
        self.repository.mkdir()
        self.destination = self.repository / "vendor" / "exec-hag"
        for checkout in (self.source, self.repository):
            self.run_git(checkout, "init", "--quiet", "--template=")
        for relative in sync_hag.FILES:
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {relative}\n")
        (self.source / "Cargo.toml").write_text(
            '[package]\nname = "exec-hag"\nversion = "0.1.0"\n'
            f'repository = "{sync_hag.SOURCE_REPOSITORY}"\n'
        )
        (self.repository / "Cargo.toml").write_text(
            f'[workspace.package]\nrepository = "{sync_hag.TARGET_REPOSITORY}"\n'
        )
        self.run_git(self.source, "add", ".")
        self.run_git(self.source, "commit", "--quiet", "-m", "fixture")

    @staticmethod
    def run_git(checkout, *arguments):
        return subprocess.run(
            [
                "git", "-C", str(checkout),
                "-c", "user.name=HAG vendoring tests",
                "-c", "user.email=hag-tests@example.invalid",
                "-c", "commit.gpgsign=false",
                *arguments,
            ],
            check=True,
            capture_output=True,
        ).stdout

    def synchronize(self, *, check=False):
        return sync_hag.synchronize(self.source, self.repository, check=check)

    def test_clean_snapshot_exactly_matches_the_allowlist_and_is_idempotent(self):
        manifest = self.synchronize()
        self.assertEqual(manifest["source"]["repository"], sync_hag.SOURCE_REPOSITORY)
        self.assertFalse(manifest["source"]["dirty"])
        self.assertEqual(
            manifest["source"]["revision"],
            self.run_git(self.source, "rev-parse", "HEAD").decode().strip(),
        )
        self.assertEqual(set(manifest["files"]), set(sync_hag.FILES))
        self.assertEqual(
            {path.relative_to(self.destination).as_posix() for path in self.destination.rglob("*") if path.is_file()},
            set(sync_hag.FILES) | {sync_hag.PROVENANCE},
        )
        before = {path: path.stat().st_mtime_ns for path in self.destination.rglob("*")}
        self.assertEqual(self.synchronize(check=True), manifest)
        self.assertEqual(self.synchronize(), manifest)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in self.destination.rglob("*")})
        for relative in sync_hag.FILES:
            self.assertEqual((self.source / relative).read_bytes(), (self.destination / relative).read_bytes())

    def test_dirty_source_is_recorded_and_hashes_change_on_edit(self):
        original = self.synchronize()
        (self.source / "src/budget.rs").write_text("changed source\n")
        changed = self.synchronize()
        self.assertTrue(changed["source"]["dirty"])
        self.assertEqual(changed["source"]["revision"], original["source"]["revision"])
        self.assertNotEqual(changed["files"]["src/budget.rs"], original["files"]["src/budget.rs"])
        self.assertEqual(self.synchronize(check=True), changed)

    def test_snapshot_includes_public_header_and_documentation_dependencies(self):
        required = {
            "CONTRIBUTING.md", "CONTRIBUTORS.md", "SECURITY.md", "VERSION",
            "include/exec_hag.h", "docs/development.md", "docs/publication.md",
            "scripts/check_public_tree.py", "tools/audit_secrets.py", ".gitleaks.toml",
        }
        manifest = self.synchronize()
        self.assertTrue(required.issubset(manifest["files"]))
        for relative in required:
            self.assertEqual(
                (self.source / relative).read_bytes(),
                (self.destination / relative).read_bytes(),
            )

    def test_missing_public_header_prevents_partial_snapshot_update(self):
        self.synchronize()
        original = (self.destination / "README.md").read_bytes()
        (self.source / "README.md").write_text("updated documentation\n")
        (self.source / "include/exec_hag.h").unlink()
        with self.assertRaisesRegex(sync_hag.SyncError, "required file is missing.*exec_hag.h"):
            self.synchronize()
        self.assertEqual((self.destination / "README.md").read_bytes(), original)

    def test_check_missing_snapshot_creates_nothing(self):
        with self.assertRaisesRegex(sync_hag.SyncError, "differs from source"):
            self.synchronize(check=True)
        self.assertFalse((self.repository / "vendor").exists())

    def test_check_detects_source_drift_without_modifying_snapshot(self):
        self.synchronize()
        original = (self.destination / "README.md").read_bytes()
        (self.source / "README.md").write_text("new documentation\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "README.md"):
            self.synchronize(check=True)
        self.assertEqual((self.destination / "README.md").read_bytes(), original)

    def test_check_detects_snapshot_tampering(self):
        self.synchronize()
        (self.destination / "src/lib.rs").write_text("tampered\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "src/lib.rs"):
            self.synchronize(check=True)
        self.assertEqual((self.destination / "src/lib.rs").read_text(), "tampered\n")
        self.synchronize()
        self.synchronize(check=True)

    def test_check_detects_provenance_hash_tampering(self):
        self.synchronize()
        provenance = self.destination / sync_hag.PROVENANCE
        manifest = json.loads(provenance.read_text())
        manifest["files"]["LICENSE"]["sha256"] = "0" * 64
        provenance.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(sync_hag.SyncError, sync_hag.PROVENANCE):
            self.synchronize(check=True)

    def test_missing_source_input_does_not_modify_existing_snapshot(self):
        self.synchronize()
        original = (self.destination / "Cargo.lock").read_bytes()
        (self.source / "Cargo.lock").unlink()
        with self.assertRaisesRegex(sync_hag.SyncError, "required file is missing"):
            self.synchronize()
        self.assertEqual((self.destination / "Cargo.lock").read_bytes(), original)

    def test_new_untracked_rust_file_refuses_instead_of_silently_omitting_it(self):
        (self.source / "src/new_kernel.rs").write_text("// new implementation\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "outside the explicit allowlist.*new_kernel.rs"):
            self.synchronize()
        self.assertFalse(self.destination.exists())

    def test_new_ignored_source_file_is_still_detected(self):
        (self.source / ".gitignore").write_text("src/ignored.rs\n")
        (self.source / "src/ignored.rs").write_text("// implementation\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "outside the explicit allowlist.*ignored.rs"):
            self.synchronize()

    def test_new_build_script_or_source_asset_requires_explicit_contract_update(self):
        (self.source / "build.rs").write_text("fn main() {}\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "outside the explicit allowlist.*build.rs"):
            self.synchronize()
        (self.source / "build.rs").unlink()
        (self.source / "src/data.bin").write_bytes(b"asset")
        with self.assertRaisesRegex(sync_hag.SyncError, "unlisted source inputs"):
            self.synchronize()

    def test_nested_source_package_is_not_a_standalone_checkout(self):
        nested = self.source / "nested"
        nested.mkdir()
        with self.assertRaisesRegex(sync_hag.SyncError, "canonical Git root"):
            sync_hag.synchronize(nested, self.repository)

    def test_wrong_package_or_repository_identity_is_rejected(self):
        manifest_path = self.source / "Cargo.toml"
        original = manifest_path.read_text()
        manifest_path.write_text(original.replace('name = "exec-hag"', 'name = "other"'))
        with self.assertRaisesRegex(sync_hag.SyncError, "package name"):
            self.synchronize()
        manifest_path.write_text(original.replace(sync_hag.SOURCE_REPOSITORY, "https://example.invalid/other"))
        with self.assertRaisesRegex(sync_hag.SyncError, "repository must identify"):
            self.synchronize()
        self.assertFalse(self.destination.exists())

    def test_source_symlink_is_rejected(self):
        source_file = self.source / "src/budget.rs"
        source_file.unlink()
        source_file.symlink_to(self.source / "src/lib.rs")
        with self.assertRaisesRegex(sync_hag.SyncError, "refusing symlink"):
            self.synchronize()
        self.assertFalse(self.destination.exists())

    def test_path_and_workspace_dependencies_cannot_break_the_fresh_checkout(self):
        manifest_path = self.source / "Cargo.toml"
        original = manifest_path.read_text()
        for dependency in (
            '[dependencies]\nother = { path = "../other" }\n',
            '[target.\'cfg(target_os = "linux")\'.dependencies]\nother = { path = "../other" }\n',
            '[dependencies]\nother = { workspace = true }\n',
        ):
            with self.subTest(dependency=dependency):
                manifest_path.write_text(original + dependency)
                with self.assertRaisesRegex(sync_hag.SyncError, "unvendored path or workspace"):
                    self.synchronize()
                self.assertFalse(self.destination.exists())

    def test_custom_library_and_build_inputs_cannot_be_silently_omitted(self):
        manifest_path = self.source / "Cargo.toml"
        original = manifest_path.read_text()
        manifest_path.write_text(original + '[lib]\npath = "different.rs"\n')
        with self.assertRaisesRegex(sync_hag.SyncError, "library path"):
            self.synchronize()
        manifest_path.write_text(original + 'build = "custom-build.txt"\n')
        with self.assertRaisesRegex(sync_hag.SyncError, "build script"):
            self.synchronize()

    def test_non_string_repository_identity_is_a_clean_refusal(self):
        manifest_path = self.source / "Cargo.toml"
        manifest_path.write_text('[package]\nname = "exec-hag"\nrepository = 7\n')
        with self.assertRaisesRegex(sync_hag.SyncError, "repository must identify"):
            self.synchronize()

    def test_destination_and_parent_symlinks_are_rejected(self):
        outside = self.root / "unrelated"
        outside.mkdir()
        vendor = self.repository / "vendor"
        vendor.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(sync_hag.SyncError, "refusing symlink"):
            self.synchronize()
        self.assertEqual(list(outside.iterdir()), [])
        vendor.unlink()
        vendor.mkdir()
        self.destination.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(sync_hag.SyncError, "refusing symlink"):
            self.synchronize()
        self.assertEqual(list(outside.iterdir()), [])

    def test_managed_destination_file_symlink_does_not_overwrite_its_target(self):
        self.synchronize()
        outside = self.root / "unrelated.txt"
        outside.write_text("keep this\n")
        managed = self.destination / "README.md"
        managed.unlink()
        managed.symlink_to(outside)
        with self.assertRaisesRegex(sync_hag.SyncError, "refusing symlink"):
            self.synchronize()
        self.assertEqual(outside.read_text(), "keep this\n")

    def test_unrelated_destination_files_are_preserved_and_prevent_any_overwrite(self):
        self.synchronize()
        original = (self.destination / "README.md").read_bytes()
        (self.destination / "personal-notes.txt").write_text("keep me\n")
        (self.source / "README.md").write_text("new source\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "unrelated entries"):
            self.synchronize()
        self.assertEqual((self.destination / "README.md").read_bytes(), original)
        self.assertEqual((self.destination / "personal-notes.txt").read_text(), "keep me\n")

    def test_nonempty_destination_without_ownership_is_not_overwritten(self):
        self.destination.mkdir(parents=True)
        (self.destination / "README.md").write_text("unowned\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "without ownership provenance"):
            self.synchronize()
        self.assertEqual((self.destination / "README.md").read_text(), "unowned\n")

    def test_invalid_provenance_is_not_claimed_as_owned(self):
        self.synchronize()
        (self.destination / sync_hag.PROVENANCE).write_text("not json\n")
        with self.assertRaisesRegex(sync_hag.SyncError, "invalid destination provenance"):
            self.synchronize()

    def test_concurrently_changed_source_refuses_before_writing(self):
        collect = sync_hag.collect_snapshot
        calls = 0

        def collect_then_mutate(source):
            nonlocal calls
            result = collect(source)
            calls += 1
            if calls == 1:
                (source / "README.md").write_text("changed during collection\n")
            return result

        with patch.object(sync_hag, "collect_snapshot", side_effect=collect_then_mutate):
            with self.assertRaisesRegex(sync_hag.SyncError, "source changed during"):
                self.synchronize()
        self.assertFalse(self.destination.exists())

    def test_target_identity_is_checked_before_any_writes(self):
        (self.repository / "Cargo.toml").write_text('[workspace.package]\nrepository = "other"\n')
        with self.assertRaisesRegex(sync_hag.SyncError, "workspace must identify exec-beholdd"):
            self.synchronize()
        self.assertFalse(self.destination.exists())

    def test_help_and_required_explicit_source(self):
        script = Path(__file__).with_name("sync-hag.py")
        help_result = subprocess.run(["python3", str(script), "--help"], capture_output=True, check=False)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn(b"--source", help_result.stdout)
        missing = subprocess.run(["python3", str(script)], capture_output=True, check=False)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn(b"--source", missing.stderr)


if __name__ == "__main__":
    unittest.main()
