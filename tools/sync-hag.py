#!/usr/bin/env python3
"""Vendor the explicit standalone HAG source contract (Python 3.11 or newer).

The destination is always this Beholdd checkout's vendor/exec-hag directory.
--check performs no writes. The source must be an explicitly supplied HAG Git
checkout, not a nested crate or an unrelated package with the same name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import tomllib


SOURCE_REPOSITORY = "https://github.com/exec-software/exec-hag"
TARGET_REPOSITORY = "https://github.com/exec-software/exec-beholdd"
FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "LICENSE",
    # Upstream's governance documents are deliberately not vendored: they describe
    # how to contribute to HAG, not how this snapshot builds, and a frozen copy of
    # someone else's README only goes stale. Source, headers, licence and notices
    # are what a vendored snapshot owes its readers.
    "VERSION",
    ".gitleaks.toml",
    "thirdparty.md",
    "include/exec_hag.h",
    "docs/development.md",
    "docs/publication.md",
    "docs/native-backend.md",
    "docs/native-probe.md",
    "docs/native-compute.md",
    "docs/exact-inference.md",
    "examples/native_probe.rs",
    "scripts/check_public_tree.py",
    "tools/audit_secrets.py",
    "src/lib.rs",
    "src/capi.rs",
    "src/budget.rs",
    "src/linux.rs",
    "src/linux/uapi.rs",
    "src/linux/submission_uapi.rs",
    "src/linux/probe_packets.rs",
    "src/linux/probe.rs",
    "src/linux/probe/lifecycle_tests.rs",
    "src/linux/compute.rs",
    "src/linux/compute/packets.rs",
    "src/linux/compute/tests.rs",
)
PROVENANCE = "provenance.json"


class SyncError(Exception):
    """The source or destination does not satisfy the vendoring contract."""


def git(checkout: Path, *arguments: str) -> bytes:
    """Run one read-only Git query with a useful, bounded failure message."""
    result = subprocess.run(
        ["git", "-C", str(checkout), *arguments],
        capture_output=True,
        check=False,
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
    )
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace").strip()
        raise SyncError(f"Git query failed for {checkout}: {diagnostic}")
    return result.stdout


def canonical_checkout(path: Path) -> Path:
    """Require a checkout root, rejecting nested crates and ordinary directories."""
    checkout = path.resolve(strict=True)
    if not checkout.is_dir():
        raise SyncError(f"not a checkout directory: {path}")
    top = Path(os.fsdecode(git(checkout, "rev-parse", "--show-toplevel")).strip())
    if top.resolve(strict=True) != checkout:
        raise SyncError(f"source/target must be its canonical Git root: {path}")
    return checkout


def validate_path(root: Path, relative: str, *, missing_ok: bool = False) -> Path:
    """Refuse links and nonregular entries along a known, repository-relative path."""
    path = root
    pieces = Path(relative).parts
    for index, piece in enumerate(pieces):
        path = path / piece
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if missing_ok:
                continue
            raise SyncError(f"required file is missing: {path}") from None
        if stat.S_ISLNK(mode):
            raise SyncError(f"refusing symlink: {path}")
        expected_directory = index < len(pieces) - 1
        if expected_directory and not stat.S_ISDIR(mode):
            raise SyncError(f"expected directory: {path}")
        if not expected_directory and not stat.S_ISREG(mode):
            raise SyncError(f"expected regular file: {path}")
    return path


def read_regular(root: Path, relative: str) -> bytes:
    """Read an allowlisted file after rejecting symlink traversal."""
    return validate_path(root, relative).read_bytes()


def tree_entries(root: Path) -> tuple[set[str], set[str]]:
    """Inspect an owned tree without following symlinks or accepting special files."""
    files = set()
    directories = set()
    for directory, child_directories, child_files in os.walk(root, followlinks=False):
        for name in child_directories + child_files:
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise SyncError(f"refusing symlink: {path}")
            if stat.S_ISDIR(mode):
                directories.add(relative)
            elif stat.S_ISREG(mode):
                files.add(relative)
            else:
                raise SyncError(f"expected regular file or directory: {path}")
    return files, directories


def collect_snapshot(source: Path) -> tuple[dict[str, bytes], dict]:
    """Capture and hash every vendored input, with honest source Git provenance."""
    package_manifest = tomllib.loads(read_regular(source, "Cargo.toml").decode("utf-8"))
    package = package_manifest.get("package", {})
    if not isinstance(package, dict) or package.get("name") != "exec-hag":
        raise SyncError("source package name must be exec-hag")
    source_identity = package.get("repository", "")
    if not isinstance(source_identity, str) or source_identity.removesuffix(".git") != SOURCE_REPOSITORY:
        raise SyncError("source package repository must identify standalone exec-hag")
    library = package_manifest.get("lib", {})
    if not isinstance(library, dict) or library.get("path", "src/lib.rs") != "src/lib.rs":
        raise SyncError("source library path is outside the explicit file contract")
    if package.get("build") not in (None, False):
        raise SyncError("source build script is outside the explicit file contract")
    if any(isinstance(value, dict) and value.get("workspace") for value in package.values()):
        raise SyncError("source package must not inherit workspace metadata")
    validate_dependencies(package_manifest)

    rust_files = {
        os.fsdecode(entry)
        for entry in git(source, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0")
        if entry.endswith(b".rs")
    }
    source_files, _ = tree_entries(source / "src")
    rust_files.update(f"src/{relative}" for relative in source_files if relative.endswith(".rs"))
    if (source / "build.rs").exists():
        rust_files.add("build.rs")
    expected_rust = {relative for relative in FILES if relative.endswith(".rs")}
    unexpected_rust = rust_files - expected_rust
    if unexpected_rust:
        raise SyncError(
            "Rust source is outside the explicit allowlist; update the vendoring contract: "
            + ", ".join(sorted(unexpected_rust))
        )
    unexpected_source = source_files - {
        relative.removeprefix("src/") for relative in FILES if relative.startswith("src/")
    }
    if unexpected_source:
        raise SyncError("unlisted source inputs: " + ", ".join(sorted(unexpected_source)))

    contents = {relative: read_regular(source, relative) for relative in FILES}
    revision = git(source, "rev-parse", "HEAD").decode("ascii").strip()
    dirty = bool(git(source, "status", "--porcelain=v1", "--untracked-files=all"))
    manifest = {
        "schema_version": 1,
        "package": "exec-hag",
        "source": {
            "repository": SOURCE_REPOSITORY,
            "revision": revision,
            "dirty": dirty,
        },
        "files": {
            relative: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for relative, data in contents.items()
        },
    }
    contents[PROVENANCE] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return contents, manifest


def validate_dependencies(table: dict) -> None:
    """Reject local/workspace dependencies that would break a standalone snapshot."""
    for key, value in table.items():
        if key in ("dependencies", "dev-dependencies", "build-dependencies") and isinstance(value, dict):
            for name, dependency in value.items():
                if isinstance(dependency, dict) and ("path" in dependency or dependency.get("workspace")):
                    raise SyncError(f"source dependency {name!r} requires an unvendored path or workspace")
        elif isinstance(value, dict):
            validate_dependencies(value)


def validate_destination(repository: Path, destination: Path) -> None:
    """Refuse unowned trees, extra files, and links before replacing any content."""
    for path in (repository / "vendor", destination):
        if path.is_symlink():
            raise SyncError(f"refusing symlink: {path}")
        if path.exists() and not path.is_dir():
            raise SyncError(f"expected directory: {path}")
    if not destination.exists():
        return
    files, directories = tree_entries(destination)
    allowed_files = set(FILES) | {PROVENANCE}
    allowed_directories = {
        parent.as_posix()
        for relative in allowed_files
        for parent in Path(relative).parents
        if parent != Path(".")
    }
    unexpected = (files - allowed_files) | (directories - allowed_directories)
    if unexpected:
        raise SyncError(
            "refusing to overwrite destination containing unrelated entries: "
            + ", ".join(sorted(unexpected))
        )
    if files:
        if PROVENANCE not in files:
            raise SyncError("refusing existing destination without ownership provenance")
        try:
            previous = json.loads(read_regular(destination, PROVENANCE))
        except (ValueError, UnicodeDecodeError) as error:
            raise SyncError(f"invalid destination provenance: {error}") from error
        if (
            not isinstance(previous, dict)
            or previous.get("schema_version") != 1
            or previous.get("package") != "exec-hag"
            or not isinstance(previous.get("source"), dict)
            or previous["source"].get("repository") != SOURCE_REPOSITORY
        ):
            raise SyncError("refusing destination with unrelated ownership provenance")
    for relative in allowed_files:
        validate_path(destination, relative, missing_ok=True)


def write_atomic(path: Path, data: bytes) -> None:
    """Replace one managed file atomically; publish provenance only after all inputs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".hag-sync-", dir=path.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def synchronize(source: Path, repository: Path, *, check: bool = False) -> dict:
    """Synchronize the fixed vendor directory, or verify it without writing anything."""
    source = canonical_checkout(source)
    repository = canonical_checkout(repository)
    target_manifest = tomllib.loads(read_regular(repository, "Cargo.toml").decode("utf-8"))
    workspace = target_manifest.get("workspace", {})
    target_package = workspace.get("package", {}) if isinstance(workspace, dict) else {}
    target_identity = target_package.get("repository", "") if isinstance(target_package, dict) else ""
    if not isinstance(target_identity, str) or target_identity.removesuffix(".git") != TARGET_REPOSITORY:
        raise SyncError("target workspace must identify exec-beholdd")
    if source == repository or source in repository.parents or repository in source.parents:
        raise SyncError("source and target checkouts must not overlap")
    destination = repository / "vendor" / "exec-hag"
    contents, manifest = collect_snapshot(source)
    validate_destination(repository, destination)
    changed = [
        relative
        for relative, data in contents.items()
        if not (destination / relative).exists()
        or read_regular(destination, relative) != data
    ]
    if check:
        if changed:
            raise SyncError("vendored HAG differs from source: " + ", ".join(changed))
        return manifest
    # Recheck inputs before writing, so concurrent edits cannot produce a snapshot
    # assembled from two observed revisions. A later --check still verifies drift.
    confirmed, _ = collect_snapshot(source)
    if confirmed != contents:
        raise SyncError("source changed during snapshot collection; retry after edits finish")
    for relative in changed:
        write_atomic(destination / relative, contents[relative])
    return manifest


def main(arguments: list[str] | None = None) -> int:
    """Command-line entry point; no default source or arbitrary destination is allowed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="explicit standalone HAG Git checkout root")
    parser.add_argument(
        "--check", action="store_true", help="verify exact source, snapshot and provenance; write nothing"
    )
    options = parser.parse_args(arguments)
    try:
        manifest = synchronize(options.source, Path(__file__).resolve().parents[1], check=options.check)
    except (SyncError, OSError, ValueError, UnicodeDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    action = "verified" if options.check else "synchronized"
    state = "dirty" if manifest["source"]["dirty"] else "clean"
    print(f"{action} {len(FILES)} HAG files at {manifest['source']['revision']} ({state} source)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
