#!/usr/bin/env python3
"""Run a development command under fixed memory, CPU, lifetime and thermal limits.

Linux/systemd user manager and the repository's AMD k10temp thermal guard are
required. This bounds this command tree; it cannot prevent unrelated services
from exhausting host memory, and GPU allocations require backend budgets too.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys


MEMORY_MAX = 3 * 1024**3
HOST_RESERVE = 8 * 1024**3
ROOT = Path(__file__).resolve().parent.parent


def available_memory(path=Path("/proc/meminfo")):
    """Require Linux's current available-memory estimate, not free pages alone."""
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0] == "MemAvailable:":
            if len(fields) != 3 or fields[2] != "kB":
                raise ValueError("invalid MemAvailable units in /proc/meminfo")
            value = int(fields[1])
            if value < 0:
                raise ValueError("negative MemAvailable in /proc/meminfo")
            return value * 1024
    raise ValueError("MemAvailable missing from /proc/meminfo")


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    if arguments and arguments[0] == "--":
        arguments.pop(0)
    if not arguments or arguments == ["--help"]:
        print("Usage: python3 tools/run_dev_guarded.py -- COMMAND [ARG ...]")
        print("3 GiB RAM, zero swap, 150% CPU, one Cargo job, one test thread, 5 minutes.")
        return 0 if arguments else 2
    try:
        available = available_memory()
        required = MEMORY_MAX + HOST_RESERVE
        if available < required:
            raise ValueError(
                f"need {required} available bytes (command cap plus host reserve); "
                f"Linux reports {available}; command was not started"
            )
        systemd_run = shutil.which("systemd-run")
        if systemd_run is None:
            raise ValueError("systemd-run is required; refusing an uncontained command")
    except (OSError, ValueError) as error:
        print(f"beholdd development guard: {error}", file=sys.stderr)
        return 1

    print(
        f"Development guard: {available} bytes available; "
        "MemoryMax=3GiB, MemorySwapMax=0, RuntimeMaxSec=300.",
        flush=True,
    )
    command = [
        systemd_run, "--user", f"--unit=beholdd-dev-{os.getpid()}",
        "--wait", "--pipe", "--collect", "--expand-environment=no",
        "-p", "MemoryHigh=2G", "-p", "MemoryMax=3G", "-p", "MemorySwapMax=0",
        "-p", "OOMPolicy=kill", "-p", "Restart=no", "-p", "CPUQuota=150%",
        # Existing row-partition stress tests explicitly request 1,024 workers.
        # Serialize tests while permitting that case inside the fixed memory cap.
        "-p", "TasksMax=1152", "-p", "RuntimeMaxSec=300", "-p", "TimeoutStopSec=5",
        "-p", "KillMode=control-group", "-p", "NoNewPrivileges=yes",
        f"--working-directory={ROOT}",
        "/usr/bin/env", "CARGO_BUILD_JOBS=1", "RUST_TEST_THREADS=1",
        sys.executable, str(ROOT / "tools/thermal_guard.py"), "--", *arguments,
    ]
    # A failed manager/cgroup/thermal check is a failed command. Never retry outside
    # containment, and never restart a child killed by its memory limit.
    try:
        result = subprocess.run(command, check=False)
    except OSError as error:
        print(f"beholdd development guard: {error}", file=sys.stderr)
        return 1
    return result.returncode if result.returncode >= 0 else 128 - result.returncode


if __name__ == "__main__":
    sys.exit(main())
