#!/usr/bin/env python3
"""Bounded, read-only AMDGPU devcoredump capture. Authenticate with sudo or pkexec."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import sys
import tempfile
import time


SYSFS = Path("/sys/class/devcoredump")
MAX_DUMPS = 4
MAX_BYTES = 64 * 1024 * 1024
READ_SECONDS = 15
HEARTBEAT_SECONDS = 10


class WatchError(Exception):
    pass


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(name, *, captured=0, **fields):
    print(json.dumps({"event": name, "timestamp": timestamp(), "pid": os.getpid(),
                      "captured": captured, **fields}), flush=True)


def entries(sysfs=SYSFS):
    # Linux's class entries are symlinks; follow them only for read-only data access.
    return sorted(sysfs.glob("devcd*"))


def startup_gate(sysfs=SYSFS, *, euid=None):
    if (os.geteuid() if euid is None else euid) != 0:
        raise WatchError("live capture requires root; authenticate with sudo or pkexec")
    try:
        disabled = (sysfs / "disabled").read_text().strip()
    except OSError as error:
        raise WatchError(f"cannot read devcoredump availability: {error}") from error
    if disabled != "0":
        raise WatchError("devcoredump is disabled or invalid; no sysfs settings were changed")
    stale = entries(sysfs)
    if stale:
        raise WatchError("existing devcoredumps prevent a fresh session: " + ", ".join(p.name for p in stale))


def sudo_group(environ):
    try:
        if "SUDO_GID" in environ:
            uid = int(environ["SUDO_UID"])
            gid = int(environ["SUDO_GID"])
        else:
            uid = int(environ["PKEXEC_UID"])
            gid = pwd.getpwuid(uid).pw_gid
    except (KeyError, ValueError, OverflowError) as error:
        raise WatchError("invoke with sudo or pkexec from a valid non-root user's account") from error
    if uid <= 0 or gid < 0:
        raise WatchError("the initiating UID must be non-root and its group must be valid")
    return gid


def output_directory(gid, base=Path("/var/tmp")):
    directory = Path(tempfile.mkdtemp(prefix="hag-crash-", dir=base))
    os.chown(directory, -1, gid)
    os.chmod(directory, 0o750)
    return directory


def exclusive_file(directory, name, gid):
    fd = os.open(directory / name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        os.fchown(fd, -1, gid)
        os.fchmod(fd, 0o640)
        return os.fdopen(fd, "w+b")
    except BaseException:
        os.close(fd)
        raise


def bounded_copy(source, destination, *, cap=MAX_BYTES, seconds=READ_SECONDS):
    command = ["/usr/bin/timeout", "--kill-after=1s", f"{seconds}s", "/usr/bin/head", "-c", str(cap + 1), "--", str(source)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=destination, stderr=errors, start_new_session=True)
        forced_timeout = False
        try:
            returncode = process.wait(timeout=seconds + 3)
        except subprocess.TimeoutExpired:
            forced_timeout = True
            os.killpg(process.pid, signal.SIGKILL)
            try:
                returncode = process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                # A kernel-stuck process is not made safe by SIGKILL. Stop capture.
                raise WatchError("capture reader did not retire after timeout and SIGKILL")
        errors.seek(0)
        diagnostic = errors.read(8192).decode("utf-8", errors="replace")
    return returncode, forced_timeout or returncode in (124, 137), diagnostic


def capture(entry, directory, gid, *, cap=MAX_BYTES, seconds=READ_SECONDS):
    if not re.fullmatch(r"devcd[0-9]+", entry.name):
        raise WatchError(f"unexpected devcoredump name: {entry.name}")
    name = entry.name + ".bin"
    result = {"device": entry.name, "source": str(entry / "data"), "file": name, "started_at": timestamp()}
    with exclusive_file(directory, name, gid) as output:
        returncode, timed_out, diagnostic = bounded_copy(entry / "data", output, cap=cap, seconds=seconds)
        output.flush()
        observed = os.fstat(output.fileno()).st_size
        truncated = observed > cap
        if truncated:
            output.truncate(cap)
        output.seek(0)
        digest = hashlib.sha256()
        while block := output.read(1024 * 1024):
            digest.update(block)
        result.update(bytes=min(observed, cap), observed_bytes=observed, sha256=digest.hexdigest(),
                      complete=returncode == 0 and not timed_out and not truncated,
                      truncated=truncated, timed_out=timed_out, returncode=returncode,
                      error=diagnostic, finished_at=timestamp())
    with exclusive_file(directory, entry.name + ".json", gid) as metadata:
        metadata.write((json.dumps(result, indent=2) + "\n").encode())
    return result


def watch(directory, gid, seconds, stop_file=None, *, sysfs=SYSFS):
    started = time.monotonic()
    deadline = started + seconds
    heartbeat = started
    seen = set()
    stop = {"signal": None}
    previous = {}

    def request_stop(signum, _frame):
        stop["signal"] = signum

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.signal(signum, request_stop)
    try:
        event("ARMED", output_dir=str(directory), duration_seconds=seconds,
              max_dumps=MAX_DUMPS, max_bytes_per_dump=MAX_BYTES)
        event("HEARTBEAT", elapsed_seconds=0, remaining_seconds=seconds)
        heartbeat = time.monotonic() + HEARTBEAT_SECONDS
        while time.monotonic() < deadline:
            if stop["signal"] is not None or (stop_file is not None and stop_file.exists()):
                event("STOPPED", signal=stop["signal"], captured=len(seen), output_dir=str(directory), status="stopped")
                return 0
            for entry in entries(sysfs):
                if time.monotonic() >= deadline or stop["signal"] is not None:
                    break
                if entry.name in seen:
                    continue
                seen.add(entry.name)
                result = capture(entry, directory, gid)
                event("CAPTURE", captured=len(seen), **result)
                if not result["complete"]:
                    event("ERROR", captured=len(seen), status="capture_incomplete", error=result["error"] or "capture timed out or exceeded its byte limit")
                    return 1
                if len(seen) >= MAX_DUMPS:
                    event("STOPPED", captured=len(seen), output_dir=str(directory), status="capture_limit")
                    return 1
            now = time.monotonic()
            if now >= heartbeat:
                event("HEARTBEAT", elapsed_seconds=round(now - started, 1),
                      remaining_seconds=max(0, round(deadline - now, 1)), captured=len(seen))
                heartbeat = now + HEARTBEAT_SECONDS
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        event("STOPPED", captured=len(seen), output_dir=str(directory), status="expired")
        return 0
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=600, help="watch duration, 1 through 1500 seconds")
    parser.add_argument("--stop-file", type=Path, help="stop when this file exists; watcher never writes it")
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 1500:
        parser.error("--seconds must be between 1 and 1500")
    try:
        startup_gate()
        gid = sudo_group(os.environ)
        if args.stop_file is not None and args.stop_file.exists():
            raise WatchError("stop file already exists; choose a fresh session stop path")
        directory = output_directory(gid)
        # Catch dumps that appeared while preparing the private output directory.
        startup_gate()
        return watch(directory, gid, args.seconds, args.stop_file)
    except (WatchError, OSError) as error:
        event("ERROR", status="failed", error=str(error))
        return 1


if __name__ == "__main__":
    sys.exit(main())
