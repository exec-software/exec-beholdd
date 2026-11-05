#!/usr/bin/env python3
"""Hard thermal kill-switch for a supervised benchmark command.

This supervisor targets Linux systems exposing a k10temp ``Tctl`` sensor. Its
defaults were used for AMD Ryzen AI MAX+ (Strix Halo) development; they are not a
universal hardware safety specification. Review temperature limits for your device.
The GPU edge reading is not substituted for the requested whole-package sensor.

This tool does not replace firmware or kernel protection. It is scoped
to one supervised process: it validates Tctl before launching, then polls while the
command runs and hard-aborts it the moment Tctl reaches `--critical-c`, rather than
trusting a graduated frequency cap to land in time for a multi-hour unattended run.
A sensor read failure is treated as a
critical breach, not as 0C -- an unreadable sensor must not be read as "safe to proceed."

usage: thermal_guard.py [--warn-c N] [--critical-c N] [--poll-seconds N]
                         [--grace-seconds N] [--log PATH] -- <command> [args...]
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_WARN_C = 75.0
DEFAULT_CRITICAL_C = 90.0
DEFAULT_POLL_SECONDS = 2.0
DEFAULT_GRACE_SECONDS = 5.0
MIN_VALID_TCTL_C = 0.0
MAX_VALID_TCTL_C = 125.0


class ThermalGuardError(ValueError):
    """The supervised run cannot proceed or could not be judged safe."""


class ThermalGuardTermination(BaseException):
    """A termination signal requested fail-closed process-group cleanup."""

    def __init__(self, signum):
        super().__init__(f"received signal {signum}")
        self.signum = signum


def raise_for_termination(signum, _frame):
    """Turn SIGTERM into an exception so ``supervise`` can run its cleanup path."""
    raise ThermalGuardTermination(signum)


def find_k10temp_input(hwmon_root="/sys/class/hwmon"):
    root = Path(hwmon_root)
    for entry in sorted(root.glob("hwmon*")):
        try:
            name = (entry / "name").read_text().strip()
        except OSError:
            continue
        if name == "k10temp":
            return entry / "temp1_input"
    raise ThermalGuardError(f"no k10temp hwmon sensor found under {hwmon_root}")


def read_tctl_c(path):
    raw = Path(path).read_text().strip()
    return validate_tctl_c(int(raw) / 1000.0)


def validate_tctl_c(value: object) -> float:
    """Reject implausible sensor values; the stricter critical limit still applies."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not MIN_VALID_TCTL_C <= value <= MAX_VALID_TCTL_C
    ):
        raise ThermalGuardError(
            f"invalid Tctl reading {value!r}; expected a finite temperature in "
            f"[{MIN_VALID_TCTL_C}, {MAX_VALID_TCTL_C}]C"
        )
    return float(value)


def process_group_alive(process_group):
    """Return whether a supervised POSIX process group still has any members."""
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def terminate_process_group(process, grace_seconds, signal_group, group_alive):
    """Terminate the supervised group and return the direct child's exit code.

    Waiting only for the direct child is unsafe because launchers such as Cargo can
    terminate while a test or model process remains in their process group. Probe the
    group after the leader exits and hard-kill any remaining descendants.
    """

    try:
        signal_group(signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        exit_code = process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        exit_code = None

    if exit_code is None or group_alive():
        try:
            signal_group(signal.SIGKILL)
        except ProcessLookupError:
            pass
        exit_code = process.wait(timeout=grace_seconds)
    return exit_code


def supervise(
    command,
    *,
    tctl_path=None,
    warn_c=DEFAULT_WARN_C,
    critical_c=DEFAULT_CRITICAL_C,
    poll_seconds=DEFAULT_POLL_SECONDS,
    grace_seconds=DEFAULT_GRACE_SECONDS,
    log_path=None,
    read_temp=None,
    sleep=time.sleep,
    clock=time.monotonic,
    popen=subprocess.Popen,
    signal_group=None,
    group_alive=None,
):
    """Validate Tctl before launch, then poll until exit or a critical reading.

    Returns a JSON-serializable report. Never raises for a thermal abort or a normal
    child failure -- both are reported, not thrown. A preflight refusal reports
    ``process_started=False`` with no PID or child exit code. Samples include the
    preflight reading. Raises `ThermalGuardError` only for
    a request that cannot be judged at all (empty command, invalid thresholds).
    """

    if not command:
        raise ThermalGuardError("no command given to supervise")
    if not (0.0 < warn_c < critical_c < 100.0):
        raise ThermalGuardError(
            f"thresholds must satisfy 0 < warn_c < critical_c < 100, got warn_c={warn_c}, "
            f"critical_c={critical_c}"
        )
    if not math.isfinite(poll_seconds) or poll_seconds <= 0.0:
        raise ThermalGuardError(
            f"poll_seconds must be a finite positive number, got {poll_seconds}"
        )
    if not math.isfinite(grace_seconds) or grace_seconds <= 0.0:
        raise ThermalGuardError(
            f"grace_seconds must be a finite positive number, got {grace_seconds}"
        )

    resolved_read_temp = read_temp
    if resolved_read_temp is None:
        sensor_path = tctl_path or find_k10temp_input()

        def resolved_read_temp():
            return read_tctl_c(sensor_path)

    def emit(record):
        if log_path is None:
            return
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    started = clock()
    samples = 0
    max_tctl_c = None
    abort_reason = None
    abort_phase = None
    exit_code = None
    residual_group_cleaned = False

    def sample_temperature(phase):
        nonlocal samples, max_tctl_c
        samples += 1
        elapsed_s = clock() - started
        try:
            tctl_c = validate_tctl_c(resolved_read_temp())
        except (OSError, ValueError, TypeError, OverflowError) as error:
            emit(
                {
                    "sample": samples,
                    "phase": phase,
                    "elapsed_s": elapsed_s,
                    "tctl_c": None,
                    "sensor_ok": False,
                    "error": str(error),
                }
            )
            return f"thermal sensor unreadable or invalid ({error}); refusing unsafe work"

        max_tctl_c = tctl_c if max_tctl_c is None else max(max_tctl_c, tctl_c)
        emit(
            {
                "sample": samples,
                "phase": phase,
                "elapsed_s": elapsed_s,
                "tctl_c": tctl_c,
                "sensor_ok": True,
                "warn": tctl_c >= warn_c,
            }
        )
        if tctl_c >= critical_c:
            return f"Tctl {tctl_c:.1f}C reached the critical threshold {critical_c:.1f}C"
        return None

    # Refuse before Popen, including fast commands that might complete before the
    # first running poll. Logging failure also prevents launching untracked work.
    abort_reason = sample_temperature("preflight")
    if abort_reason is not None:
        return {
            "ok": False,
            "aborted_for_heat": True,
            "abort_reason": abort_reason,
            "abort_phase": "preflight",
            "max_tctl_c": max_tctl_c,
            "samples": samples,
            "exit_code": None,
            "residual_group_cleaned": False,
            "process_started": False,
            "pid": None,
        }

    # A benchmark command can itself be a runner (cargo, a shell harness, timeout) that
    # owns the model process. Give the whole tree a fresh process group so a thermal
    # abort cannot strand a compute-heavy grandchild after killing only its parent.
    process = popen(command, start_new_session=True)
    resolved_signal_group = signal_group or (lambda sig: os.killpg(process.pid, sig))
    resolved_group_alive = group_alive or (lambda: process_group_alive(process.pid))
    cleanup_completed = False
    try:
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                # The direct command may be only a launcher. Because it owns a fresh
                # process group, a surviving member is one of its descendants and must
                # not outlive supervision after the launcher exits. Preserve the
                # launcher's real status while terminating the residual group.
                if resolved_group_alive():
                    launcher_exit_code = exit_code
                    terminate_process_group(
                        process,
                        grace_seconds,
                        resolved_signal_group,
                        resolved_group_alive,
                    )
                    cleanup_completed = True
                    residual_group_cleaned = True
                    exit_code = launcher_exit_code
                break

            abort_reason = sample_temperature("running")
            if abort_reason is not None:
                abort_phase = "running"
                break

            sleep(poll_seconds)

        if abort_reason is not None:
            exit_code = terminate_process_group(
                process,
                grace_seconds,
                resolved_signal_group,
                resolved_group_alive,
            )
            cleanup_completed = True
    except BaseException as error:
        if not cleanup_completed:
            try:
                terminate_process_group(
                    process,
                    grace_seconds,
                    resolved_signal_group,
                    resolved_group_alive,
                )
            except BaseException as cleanup_error:
                error.add_note(
                    "thermal guard also failed while terminating the supervised process group: "
                    f"{cleanup_error}"
                )
        raise

    return {
        "ok": abort_reason is None,
        "aborted_for_heat": abort_reason is not None,
        "abort_reason": abort_reason,
        "abort_phase": abort_phase,
        "max_tctl_c": max_tctl_c,
        "samples": samples,
        "exit_code": exit_code,
        "residual_group_cleaned": residual_group_cleaned,
        "process_started": True,
        "pid": process.pid,
    }


def split_command(argv):
    if "--" not in argv:
        raise ThermalGuardError("missing '--' separator before the supervised command")
    index = argv.index("--")
    return argv[:index], argv[index + 1 :]


def parse_arguments(argv):
    own_args, command = split_command(argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warn-c", type=float, default=DEFAULT_WARN_C)
    parser.add_argument("--critical-c", type=float, default=DEFAULT_CRITICAL_C)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--grace-seconds", type=float, default=DEFAULT_GRACE_SECONDS)
    parser.add_argument("--log", default=None)
    options = parser.parse_args(own_args)
    if not command:
        raise ThermalGuardError("no command given after '--'")
    return options, command


def main(argv=None):
    previous_sigterm = signal.signal(signal.SIGTERM, raise_for_termination)
    try:
        try:
            options, command = parse_arguments(sys.argv[1:] if argv is None else argv)
            report = supervise(
                command,
                warn_c=options.warn_c,
                critical_c=options.critical_c,
                poll_seconds=options.poll_seconds,
                grace_seconds=options.grace_seconds,
                log_path=options.log,
            )
        except ThermalGuardError as error:
            print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
            raise SystemExit(2) from None
        except ThermalGuardTermination as error:
            print(
                json.dumps(
                    {"ok": False, "error": str(error), "signal": error.signum},
                    sort_keys=True,
                )
            )
            raise SystemExit(128 + error.signum) from None
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)

    print(json.dumps(report, sort_keys=True))
    if report["aborted_for_heat"]:
        raise SystemExit(1)
    raise SystemExit(report["exit_code"] if report["exit_code"] is not None else 1)


if __name__ == "__main__":
    main()
