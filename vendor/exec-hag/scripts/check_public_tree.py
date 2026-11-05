#!/usr/bin/env python3
# Copyright (C) 2008 - 2026 Execute Software.  All rights reserved.
# SPDX-License-Identifier: MIT

"""Check publication conventions in tracked and non-ignored new files.

This supplements a secret scanner and review of Git history; it does not replace
either. Diagnostics intentionally omit matching content, which may be sensitive.
"""

from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NOTICE = "Copyright (C) 2008 - 2026 Execute Software.  All rights reserved."
HEADER_EXEMPT = {"Cargo.lock", "VERSION"}
PUBLIC_REPOSITORIES = {"exec-hag", "exec-nil-framework"}
LOCAL_PATH = re.compile(
    r"/(?:home|Users|Incubator|Execute)/[A-Za-z0-9_.-]+"
    r"|[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][A-Za-z0-9_.-]+"
)
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----")
ORG_LINK = re.compile(r"github\.com[:/]exec-software/([A-Za-z0-9_.-]+)", re.I)
CREDENTIAL_NAMES = {"id_rsa", "id_ed25519", ".netrc", ".npmrc"}
CREDENTIAL_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}


def main() -> int:
    paths = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT,
    ).decode("utf-8").split("\0")
    problems = []
    checked = 0
    for name in sorted(set(filter(None, paths))):
        path = ROOT / name
        if path.is_symlink():
            problems.append(f"{name}: review symlink before publication")
            continue
        # A tracked deletion is absent from the intended working tree.
        if not path.exists():
            continue
        if not path.is_file():
            problems.append(f"{name}: unsupported file type")
            continue
        checked += 1
        if (
            path.name in CREDENTIAL_NAMES
            or path.suffix.lower() in CREDENTIAL_SUFFIXES
            or (path.name.startswith(".env") and path.name not in {".env.example", ".env.sample"})
        ):
            problems.append(f"{name}: credential-like filename")
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            problems.append(f"{name}: binary file needs explicit publication review")
            continue
        if name not in HEADER_EXEMPT and NOTICE not in content[:2048]:
            problems.append(f"{name}: missing project copyright header")
        for line_number, line in enumerate(content.splitlines(), 1):
            location = f"{name}:{line_number}"
            if LOCAL_PATH.search(line):
                problems.append(f"{location}: workstation-specific absolute path")
            if PRIVATE_KEY.search(line):
                problems.append(f"{location}: private-key marker")
            for match in ORG_LINK.finditer(line):
                repository = match.group(1).lower().rstrip(".").removesuffix(".git")
                if repository not in PUBLIC_REPOSITORIES:
                    problems.append(f"{location}: organization repository not approved for public references")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    print(f"Public-tree conventions passed for {checked} files (Git history is not checked).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
