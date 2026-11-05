#!/usr/bin/env python3
"""Scan local Git history and a non-ignored source snapshot; never publish reports."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SCANNER_VERSION = "8.30.1"


def git(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *arguments])


def snapshot(root: Path, destination: Path) -> list[dict]:
    """Copy tracked and non-ignored untracked files, without following links."""
    names = set(git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").split(b"\0"))
    records = []
    for raw_name in sorted(names - {b""}):
        name = os.fsdecode(raw_name)
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe Git file path")
        source = root / relative
        # A tracked deletion is part of the candidate tree, not a scanned file.
        # Record it explicitly; the history scan still covers its earlier bytes.
        if not source.exists() and not source.is_symlink():
            deleted = git(root, "ls-files", "--deleted", "-z", "--", name).split(b"\0")
            if raw_name in deleted:
                records.append({"file": name, "state": "deleted-in-worktree"})
                continue
        # Refuse submodules, symlinks (including a symlinked parent), devices,
        # and unexpected missing entries: none may silently escape the audit.
        if source.is_symlink() or source.resolve() != source or not source.is_file():
            raise ValueError(f"review non-regular or missing source entry: {name!r}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        data = target.read_bytes()
        records.append({"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return records


def run_scan(scanner: str, mode: str, target: Path, config: Path, reports: Path) -> tuple[int, int]:
    report = reports / f"{mode}.json"
    command = [scanner, "git" if mode == "staged" else mode, str(target), "--config", str(config),
               "--redact=100", "--ignore-gitleaks-allow", "--gitleaks-ignore-path=/dev/null",
               "--no-banner", "--no-color", "--max-archive-depth=3", "--timeout=600",
               "--report-format=json", "--report-path", str(report)]
    if mode == "git":
        command += ["--log-opts=--all --full-history"]
    elif mode == "staged":
        command += ["--staged"]
    with (reports / f"{mode}.log").open("x") as output:
        result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, check=False)
    if result.returncode not in (0, 1) or not report.exists():
        raise RuntimeError(f"{mode} scanner failed; inspect private logs")
    findings = json.loads(report.read_text())
    if not isinstance(findings, list) or (result.returncode == 0 and findings):
        raise RuntimeError(f"{mode} scanner result is inconsistent")
    return result.returncode, len(findings)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scanner", default="gitleaks", help=f"path to Gitleaks {SCANNER_VERSION}")
    parser.add_argument("--report-dir", required=True, type=Path, help="new private directory outside the checkout")
    args = parser.parse_args()
    os.umask(0o077)
    root = Path(git(Path(__file__).resolve().parent, "rev-parse", "--show-toplevel").decode().strip()).resolve()
    reports = args.report_dir.resolve()
    if reports.exists() or reports == root or root in reports.parents:
        parser.error("report directory must be new and outside the checkout")
    version = subprocess.check_output([args.scanner, "version"], text=True).strip().removeprefix("v")
    if version != SCANNER_VERSION:
        parser.error(f"expected Gitleaks {SCANNER_VERSION}, got a different version")
    config = root / ".gitleaks.toml"
    if not config.is_file():
        parser.error("repository scanner configuration is missing")
    reports.mkdir(parents=True, mode=0o700)
    try:
        with tempfile.TemporaryDirectory(prefix="source-", dir=reports) as temporary:
            exported = Path(temporary)
            files = snapshot(root, exported)
            (reports / "files.json").write_text(json.dumps(files, indent=2) + "\n")
            (reports / "refs.txt").write_bytes(git(root, "show-ref", "--head"))
            tree_status, tree_count = run_scan(args.scanner, "dir", exported, config, reports)
            history_status, history_count = run_scan(args.scanner, "git", root, config, reports)
            staged_status, staged_count = run_scan(args.scanner, "staged", root, config, reports)
        summary = {"scanner_version": version, "head": git(root, "rev-parse", "HEAD").decode().strip(),
                   "source_files": len(files), "tree_findings": tree_count, "history_findings": history_count,
                   "staged_findings": staged_count,
                   "scope": "local refs, staged changes, tracked/non-ignored working files; archive depth 3",
                   "limits": "not a license, IP, vulnerability, artifact, or remote-only-ref clearance"}
        (reports / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
        return max(tree_status, history_status, staged_status)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        # Detailed scanner logs stay private. Do not echo source values.
        print("Audit did not complete; inspect the private report directory and configuration.")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
