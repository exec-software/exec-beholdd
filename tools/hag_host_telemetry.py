#!/usr/bin/env python3
"""Finite, passive Linux host telemetry. This is not a thermal safety guard.

Example: python3 tools/hag_host_telemetry.py --duration-seconds 60 \
    --interval-seconds 1 --top-processes 10 --output /absolute/new.jsonl

Only bounded regular procfs/sysfs attributes are read. No workload is launched,
device node opened, or driver/power setting changed. Optional process activity
contains numeric identifiers/counters only, never names, arguments or contents.
Missing sensors are data, not evidence that a host is cool or idle. Timestamps
bracket each sample: sensors are sampled sequentially, not atomically. As with
any sysfs reader, an uninterruptible kernel read needs an external supervisor.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

MAX_DURATION = 900.0
MIN_INTERVAL = 0.1
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_LINE_BYTES = 64 * 1024
SUMMARY_RESERVE = 4096
MAX_PROCESSES = 4096
MAX_TOP = 20
MAX_ENTRIES = 256
MAX_GPUS = 8
MAX_SENSORS = 64


def unavailable(reason):
    return {"status": "unavailable", "reason": reason}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_text(path, limit=4096):
    """Never follow a final symlink, block on a FIFO, or open a device node."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
        raise ValueError("not a bounded regular attribute")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode) or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("attribute changed before read")
        raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("attribute exceeded read limit")
    return raw.decode("ascii", errors="strict").strip()


def integer(text):
    if not re.fullmatch(r"-?[0-9]{1,20}", text):
        raise ValueError("malformed integer")
    return int(text)


def entries(path, limit=MAX_ENTRIES):
    """Bound directory scanning and make truncation visible to the caller."""
    found = []
    with os.scandir(path) as scan:
        for entry in scan:
            if len(found) == limit:
                return sorted(found), True
            found.append(entry.name)
    return sorted(found), False


def discover(sys_root):
    specs = {"tctl_c": {"path": None, "unit": "C", "scale": 1000, "maximum": 125}}
    notes = []
    try:
        names, truncated = entries(sys_root / "class/hwmon")
        if truncated:
            notes.append("hwmon discovery truncated")
        for name in names:
            base = sys_root / "class/hwmon" / name
            try:
                if read_text(base / "name") != "k10temp":
                    continue
                labels, labels_truncated = entries(base)
                if labels_truncated:
                    notes.append("k10temp attribute discovery truncated")
                candidates = []
                for label in labels:
                    if re.fullmatch(r"temp[0-9]+_label", label) and read_text(base / label) == "Tctl":
                        candidates.append(base / label.replace("_label", "_input"))
                # k10temp exposes Tctl as temp1 even on kernels without labels.
                if not candidates and not (base / "temp1_label").exists():
                    candidates = [base / "temp1_input"]
                    notes.append("Tctl uses k10temp temp1 without a label")
                if candidates and specs["tctl_c"]["path"] is None:
                    specs["tctl_c"]["path"] = candidates[0]
            except (OSError, ValueError, UnicodeError):
                continue
    except OSError:
        notes.append("hwmon discovery unavailable")
    try:
        cards, truncated = entries(sys_root / "class/drm")
        if truncated:
            notes.append("DRM discovery truncated")
        count = 0
        for card in cards:
            if not re.fullmatch(r"card[0-9]+", card):
                continue
            base = sys_root / "class/drm" / card / "device"
            try:
                if read_text(base / "vendor") != "0x1002":
                    continue
                driver = (base / "driver").resolve(strict=True).name
                if driver != "amdgpu":
                    continue
            except (OSError, ValueError):
                continue
            if count == MAX_GPUS:
                notes.append("GPU discovery truncated")
                break
            count += 1
            definitions = (
                ("busy_percent", base / "gpu_busy_percent", "%", 1, 100, "integer"),
                ("core_clock_mhz", base / "pp_dpm_sclk", "MHz", 1, 100000, "dpm"),
                ("memory_clock_mhz", base / "pp_dpm_mclk", "MHz", 1, 100000, "dpm"),
                ("power_w", None, "W", 1000000, 10000, "integer"),
                ("fan_rpm", None, "RPM", 1, 100000, "integer"),
            )
            for key, path, unit, scale, maximum, kind in definitions:
                specs[card + "." + key] = dict(path=path, unit=unit, scale=scale, maximum=maximum, kind=kind)
            try:
                hwmons, truncated = entries(base / "hwmon")
                if truncated:
                    notes.append(card + " hwmon discovery truncated")
                for hwmon in hwmons:
                    hw = base / "hwmon" / hwmon
                    if read_text(hw / "name") != "amdgpu":
                        continue
                    for field, candidates in (("power_w", ("power1_average", "power1_input")), ("fan_rpm", ("fan1_input",))):
                        specs[card + "." + field]["path"] = next((hw / p for p in candidates if (hw / p).exists()), None)
                    break
            except (OSError, ValueError, UnicodeError):
                notes.append(card + " hwmon unavailable")
        if not count:
            notes.append("amdgpu sensors unavailable")
    except OSError:
        notes.append("DRM discovery unavailable")
    if len(specs) > MAX_SENSORS:
        raise ValueError("sensor inventory exceeds bound")
    return specs, notes


def sensor(spec):
    if spec["path"] is None:
        return unavailable("not_found")
    try:
        text = read_text(spec["path"])
        if spec.get("kind") == "dpm":
            active = [line for line in text.splitlines() if "*" in line]
            match = re.fullmatch(r"\s*[0-9]+:\s*([0-9]+)\s*Mhz\s*\*\s*", active[0], re.IGNORECASE) if len(active) == 1 else None
            if not match:
                raise ValueError("active clock missing or ambiguous")
            value = integer(match[1])
        else:
            value = integer(text)
        value /= spec["scale"]
        if not 0 <= value <= spec["maximum"]:
            raise ValueError("sensor out of range")
        return {"status": "ok", "value": value}
    except FileNotFoundError:
        return unavailable("not_found")
    except PermissionError:
        return unavailable("permission_denied")
    except OSError:
        return unavailable("read_error")
    except (ValueError, UnicodeError):
        return unavailable("malformed_or_out_of_range")


def cpu_ticks(text):
    fields = text.splitlines()[0].split()
    if fields[0] != "cpu" or len(fields) < 9:
        raise ValueError("aggregate CPU counters missing")
    values = [integer(value) for value in fields[1:9]]
    if any(value < 0 for value in values):
        raise ValueError("negative CPU counter")
    return values  # guest/guest_nice are already included in user/nice.


def cpu_delta(previous, current):
    if previous is None:
        return unavailable("baseline")
    delta = [new - old for old, new in zip(previous, current)]
    if any(value < 0 for value in delta):
        return unavailable("counter_reset")
    total = sum(delta)
    if not total:
        return unavailable("no_counter_progress")
    return {"status": "ok", "total_ticks": total, "busy_ticks": total - delta[3] - delta[4],
            "busy_percent": 100 * (total - delta[3] - delta[4]) / total,
            "iowait_percent": 100 * delta[4] / total}


def memory(text):
    wanted = {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}
    values = {}
    for line in text.splitlines():
        fields = line.split()
        key = fields[0].removesuffix(":") if fields else ""
        if key in wanted:
            if key in values or len(fields) != 3 or fields[2] != "kB":
                raise ValueError("malformed memory counters")
            values[key] = integer(fields[1]) * 1024
    if set(values) != wanted or not 0 <= values["MemAvailable"] <= values["MemTotal"] or not 0 <= values["SwapFree"] <= values["SwapTotal"]:
        raise ValueError("invalid memory counters")
    return {"status": "ok", "total_bytes": values["MemTotal"], "available_bytes": values["MemAvailable"],
            "swap_total_bytes": values["SwapTotal"], "swap_used_bytes": values["SwapTotal"] - values["SwapFree"]}


def process_stat(text, pid, page_size):
    # comm may contain spaces and parentheses. It is deliberately never retained.
    prefix, separator, tail = text.rpartition(")")
    fields = tail.split()
    if not separator or not prefix.startswith(str(pid) + " (") or len(fields) < 22:
        raise ValueError("malformed process counters")
    user, system, started, rss = (integer(fields[index]) for index in (11, 12, 19, 21))
    if min(user, system, started, rss) < 0:
        raise ValueError("negative process counter")
    return {"pid": pid, "start_ticks": started, "cpu_ticks": user + system, "rss_bytes": rss * page_size}


def pressure(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] not in ("some", "full") or fields[0] in result:
            raise ValueError("malformed pressure counters")
        values = {}
        for field in fields[1:]:
            key, separator, raw = field.partition("=")
            if not separator or key in values or key not in ("avg10", "avg60", "avg300", "total"):
                raise ValueError("malformed pressure value")
            value = integer(raw) if key == "total" else float(raw)
            if not math.isfinite(value) or value < 0 or (key != "total" and value > 100):
                raise ValueError("pressure value out of range")
            values[key] = value
        if len(values) != 4:
            raise ValueError("missing pressure counters")
        result[fields[0]] = values
    if "some" not in result:
        raise ValueError("missing pressure some row")
    return {"status": "ok", "total_unit": "microseconds", **result}


class HostSource:
    def __init__(self, *, proc_root=Path("/proc"), sys_root=Path("/sys"), top_processes=10,
                 ticks_per_second=None, page_size=None):
        if type(top_processes) is not int or not 0 <= top_processes <= MAX_TOP:
            raise ValueError("top_processes must be in 0..20")
        self.proc_root, self.top = proc_root, top_processes
        self.specs, self.notes = discover(sys_root)
        self.hz = ticks_per_second if ticks_per_second is not None else os.sysconf("SC_CLK_TCK")
        self.page_size = page_size if page_size is not None else os.sysconf("SC_PAGE_SIZE")
        if self.hz <= 0 or self.page_size <= 0:
            raise ValueError("invalid host counter units")
        self.previous_cpu = None
        self.previous_processes = {}
        self.process_time = None

    def processes(self, now):
        if not self.top:
            return {"status": "disabled"}
        if self.process_time is not None and now - self.process_time < 1.0:
            return {"status": "not_sampled", "reason": "one_second_process_interval"}
        names, truncated = entries(self.proc_root, MAX_PROCESSES)
        current, errors = {}, 0
        for name in names:
            if not re.fullmatch(r"[1-9][0-9]{0,9}", name):
                continue
            pid = int(name)
            try:
                row = process_stat(read_text(self.proc_root / name / "stat", 16384), pid, self.page_size)
                current[(pid, row["start_ticks"])] = row
            except (OSError, ValueError, UnicodeError):
                errors += 1
        elapsed = now - self.process_time if self.process_time is not None else None
        ranked = []
        if elapsed is not None and elapsed > 0:
            for identity, row in current.items():
                previous = self.previous_processes.get(identity)
                if previous is None or row["cpu_ticks"] < previous["cpu_ticks"]:
                    continue
                delta = row["cpu_ticks"] - previous["cpu_ticks"]
                if delta:
                    ranked.append({"pid": row["pid"], "start_ticks": row["start_ticks"],
                                   "cpu_percent_one_core": 100 * delta / self.hz / elapsed,
                                   "rss_bytes": row["rss_bytes"]})
        self.previous_processes, self.process_time = current, now
        ranked.sort(key=lambda row: (-row["cpu_percent_one_core"], row["pid"]))
        return {"status": "ok" if elapsed is not None else "baseline", "interval_seconds": elapsed,
                "readable_processes": len(current), "unreadable_processes": errors,
                "scan_truncated": truncated, "top": ranked[:self.top]}

    def sample(self, now):
        result = {"sensors": {key: sensor(spec) for key, spec in self.specs.items()}}
        try:
            current = cpu_ticks(read_text(self.proc_root / "stat", 65536))
            result["cpu"] = cpu_delta(self.previous_cpu, current)
            self.previous_cpu = current
        except (OSError, ValueError, UnicodeError, IndexError):
            self.previous_cpu = None
            result["cpu"] = unavailable("read_or_parse_error")
        try:
            result["memory"] = memory(read_text(self.proc_root / "meminfo", 65536))
        except (OSError, ValueError, UnicodeError):
            result["memory"] = unavailable("read_or_parse_error")
        result["pressure"] = {}
        for resource in ("cpu", "memory"):
            try:
                result["pressure"][resource] = pressure(read_text(self.proc_root / "pressure" / resource))
            except (OSError, ValueError, UnicodeError):
                result["pressure"][resource] = unavailable("read_or_parse_error")
        try:
            result["processes"] = self.processes(now)
        except (OSError, ValueError, UnicodeError):
            self.previous_processes, self.process_time = {}, None
            result["processes"] = unavailable("read_or_parse_error")
        return result

    def metadata(self):
        return {"sensors": {key: {"path": str(spec["path"]) if spec["path"] else None, "unit": spec["unit"]}
                            for key, spec in self.specs.items()}, "discovery_notes": self.notes,
                "host_identity": self.identity(),
                "process_top_limit": self.top, "process_scan_entry_limit": MAX_PROCESSES,
                "process_min_interval_seconds": 1, "process_names_or_arguments_recorded": False,
                "clock_ticks_per_second": self.hz, "page_size_bytes": self.page_size}

    def identity(self):
        result = {"collector_pid": os.getpid()}
        try:
            hostname = os.uname().nodename
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,253}", hostname):
                raise ValueError("invalid hostname")
            result["hostname"] = {"status": "ok", "value": hostname}
        except (OSError, ValueError):
            result["hostname"] = unavailable("read_or_parse_error")
        try:
            boot = read_text(self.proc_root / "sys/kernel/random/boot_id", 4096)
            if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", boot):
                raise ValueError("invalid boot ID")
            result["boot_id"] = {"status": "ok", "value": boot}
        except (OSError, ValueError, UnicodeError):
            result["boot_id"] = unavailable("read_or_parse_error")
        try:
            row = process_stat(read_text(self.proc_root / "self/stat", 16384), os.getpid(), self.page_size)
            result["collector_start_ticks"] = {"status": "ok", "value": row["start_ticks"]}
        except (OSError, ValueError, UnicodeError):
            result["collector_start_ticks"] = unavailable("read_or_parse_error")
        try:
            lines = read_text(self.proc_root / "self/cgroup", 4096).splitlines()
            unified = [line[3:] for line in lines if line.startswith("0::")]
            if len(unified) != 1 or not unified[0].startswith("/") or ".." in unified[0].split("/") or any(ord(c) < 32 for c in unified[0]):
                raise ValueError("invalid unified cgroup membership")
            result["unified_cgroup"] = {"status": "ok", "value": unified[0]}
        except (OSError, ValueError, UnicodeError):
            result["unified_cgroup"] = unavailable("read_or_parse_error")
        return result


class OutputLimit(Exception):
    pass


class JsonLines:
    def __init__(self, path, maximum=MAX_OUTPUT_BYTES):
        if type(maximum) is not int or not SUMMARY_RESERVE * 2 <= maximum <= MAX_OUTPUT_BYTES:
            raise ValueError("invalid output byte bound")
        if not path.is_absolute() or path.parent.resolve(strict=True) != path.parent:
            raise ValueError("output parent must be canonical and absolute")
        self.handle = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb")
        self.maximum, self.written = maximum, 0

    def append(self, record, *, final=False):
        data = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        limit = self.maximum if final else self.maximum - SUMMARY_RESERVE
        if len(data) > MAX_LINE_BYTES or self.written + len(data) > limit:
            raise OutputLimit("telemetry output byte bound reached")
        self.handle.write(data)
        self.handle.flush()
        self.written += len(data)


def validate_options(duration, interval):
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 0 < duration <= MAX_DURATION:
        raise ValueError("duration must be finite and in (0, 900] seconds")
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or not MIN_INTERVAL <= interval <= duration:
        raise ValueError("interval must be finite, >=0.1 seconds and <=duration")


def collect(output, duration, interval, *, source=None, monotonic=time.monotonic, utc=utc_now,
            sleep=time.sleep, stop_requested=lambda: False, maximum=MAX_OUTPUT_BYTES):
    validate_options(duration, interval)
    source = source if source is not None else HostSource()
    writer = JsonLines(output, maximum)
    started = monotonic()
    deadline = started + duration
    samples, skipped, index = 0, 0, 0
    reason, error = "duration_elapsed", None
    try:
        writer.append({"event": "START", "schema": 1, "timestamp_utc": utc(), "monotonic_seconds": started,
                       "pid": os.getpid(), "duration_seconds": duration, "interval_seconds": interval,
                       "maximum_output_bytes": maximum, "passive_only": True, **source.metadata()})
        while index < math.ceil(duration / interval):
            now = monotonic()
            if stop_requested():
                reason = "interrupted"
                break
            if now >= deadline:
                break
            due = started + index * interval
            if now < due:
                sleep(min(due - now, deadline - now, 0.1))
                continue
            wall = utc()
            values = source.sample(now)
            ended = monotonic()
            if ended < now:
                raise ValueError("monotonic clock moved backwards")
            writer.append({"event": "SAMPLE", "sample": samples + 1, "timestamp_utc": wall,
                           "monotonic_seconds": now, "elapsed_seconds": now - started,
                           "sample_end_utc": utc(), "sample_end_monotonic_seconds": ended,
                           "read_duration_seconds": ended - now, "schedule_lateness_seconds": now - due, **values})
            samples += 1
            next_index = max(index + 1, math.floor((ended - started) / interval) + 1)
            skipped += max(0, min(next_index, math.ceil(duration / interval)) - index - 1)
            index = next_index
        # Preserve the requested observation interval even if its final sample is earlier.
        while monotonic() < deadline and reason == "duration_elapsed" and not stop_requested():
            sleep(max(0, min(0.1, deadline - monotonic())))
        if stop_requested() and reason == "duration_elapsed":
            reason = "interrupted"
    except OutputLimit:
        reason = "output_limit"
    except Exception as caught:
        reason, error = "collector_error", type(caught).__name__
    finally:
        summary = {"event": "STOPPED", "timestamp_utc": utc(), "monotonic_seconds": monotonic(),
                   "elapsed_seconds": monotonic() - started, "samples": samples, "skipped_intervals": skipped,
                   "reason": reason, "duration_completed": reason == "duration_elapsed", "error_type": error,
                   "bytes_before_summary": writer.written}
        try:
            writer.append(summary, final=True)
        finally:
            writer.handle.close()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--interval-seconds", type=float, default=1.0)
    parser.add_argument("--top-processes", type=int, default=10, help="0 disables numeric process activity; maximum20")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True

    old = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        validate_options(args.duration_seconds, args.interval_seconds)
        source = HostSource(top_processes=args.top_processes)
        summary = collect(args.output, args.duration_seconds, args.interval_seconds, source=source,
                          stop_requested=lambda: interrupted)
        print(json.dumps(summary, allow_nan=False))
        return 0 if summary["duration_completed"] else 1
    except (OSError, ValueError) as error:
        print("telemetry refused: " + str(error), file=sys.stderr)
        return 2
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    sys.exit(main())
