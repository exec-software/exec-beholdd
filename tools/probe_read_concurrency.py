#!/usr/bin/env python3
"""Measure achieved read throughput on real checkpoint storage at increasing concurrency.

Answers one question before `--prefetch-workers` is set to anything: does this device
actually reward more than one physical read in flight at once, and if so, up to how many?
A prefetch worker pool sized above the plateau this finds spends threads for nothing; sized
below it leaves real throughput unclaimed.

Every level reads fixed-size chunks from randomly chosen, non-overlapping offsets, and no
offset is ever reused across levels -- so a later level cannot look faster only because an
earlier level already pulled its bytes into the page cache. Chunks default to 8 MiB,
matching the shape of a single routed-expert range in a real checkpoint (see
docs/research/model-streaming.md); readers are plain `pread`, matching what
`beholdd-format`'s `RangeReader` implementations actually issue.

usage: probe_read_concurrency.py [--levels 1,2,4,8,16,32] [--chunk-bytes N]
                                  [--chunks-per-level N] [--seed N] <shard> [<shard> ...]
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from random import Random
from time import monotonic


class ProbeError(ValueError):
    """The probe cannot run as configured."""


@dataclass(frozen=True)
class ReadTask:
    file_index: int
    offset: int


def pick_offsets(file_sizes, levels, chunk_bytes, chunks_per_level, seed):
    """Assign each level a list of `chunks_per_level` `ReadTask`s.

    Offsets are drawn without replacement across the ENTIRE run, not just within one
    level: every (file, offset) pair returned anywhere is unique, so no level can benefit
    from bytes an earlier level already faulted into the page cache.
    """
    if chunk_bytes <= 0:
        raise ProbeError("chunk_bytes must be positive")
    if chunks_per_level <= 0:
        raise ProbeError("chunks_per_level must be positive")

    rng = Random(seed)
    available = []
    for file_index, size in enumerate(file_sizes):
        n_chunks = size // chunk_bytes
        available.extend(ReadTask(file_index, i * chunk_bytes) for i in range(n_chunks))

    needed = len(levels) * chunks_per_level
    if needed > len(available):
        raise ProbeError(
            f"need {needed} distinct {chunk_bytes}-byte chunks across all levels "
            f"combined, but the supplied files only offer {len(available)}; supply "
            "larger or more files, or lower --chunks-per-level"
        )

    rng.shuffle(available)
    assignment = {}
    cursor = 0
    for level in levels:
        assignment[level] = available[cursor : cursor + chunks_per_level]
        cursor += chunks_per_level
    return assignment


def _pread(fd, offset, length):
    # os.pread's own argument order is (fd, length, offset); everything in this module
    # uses (fd, offset, length) instead, matching how a checked byte range reads
    # elsewhere in this codebase (offset named before length it applies to).
    return os.pread(fd, length, offset)


def probe_read_concurrency(
    paths,
    levels=(1, 2, 4, 8, 16, 32),
    chunk_bytes=8 * 1024 * 1024,
    chunks_per_level=64,
    seed=0xC0FFEE,
    open_file=lambda p: os.open(p, os.O_RDONLY),
    close_file=os.close,
    read=_pread,
):
    """Run the concurrency sweep and return a JSON-serializable record.

    `read(fd, offset, length)` performs one chunk's positional read and must return the
    bytes read; the default is a real `pread`. Injectable so tests exercise the offset
    accounting and concurrency plumbing without touching a disk.
    """
    if not paths:
        raise ProbeError("at least one checkpoint shard path is required")
    levels = list(levels)
    if not levels or any(level <= 0 for level in levels):
        raise ProbeError("levels must be a non-empty list of positive integers")

    sizes = [os.path.getsize(path) for path in paths]
    assignment = pick_offsets(sizes, levels, chunk_bytes, chunks_per_level, seed)

    fds = [open_file(path) for path in paths]
    try:
        results = []
        for level in levels:
            tasks = assignment[level]

            def do_one(task):
                return len(read(fds[task.file_index], task.offset, chunk_bytes))

            started = monotonic()
            with ThreadPoolExecutor(max_workers=level) as pool:
                total_bytes = sum(pool.map(do_one, tasks))
            elapsed_s = monotonic() - started
            results.append(
                {
                    "concurrency": level,
                    "chunks": len(tasks),
                    "bytes": total_bytes,
                    "elapsed_s": elapsed_s,
                    "gbps": (total_bytes / 1e9) / elapsed_s if elapsed_s > 0 else None,
                }
            )
        return {
            "chunk_bytes": chunk_bytes,
            "chunks_per_level": chunks_per_level,
            "seed": seed,
            "shards": list(paths),
            "levels": results,
        }
    finally:
        for fd in fds:
            close_file(fd)


def positive_int_list(value):
    try:
        levels = [int(part) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "must be a comma-separated list of positive integers"
        ) from error
    if not levels or any(level <= 0 for level in levels):
        raise argparse.ArgumentTypeError("must be a comma-separated list of positive integers")
    return levels


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--levels",
        type=positive_int_list,
        default=[1, 2, 4, 8, 16, 32],
        help="comma-separated concurrency levels to test (default: 1,2,4,8,16,32)",
    )
    parser.add_argument("--chunk-bytes", type=positive_int, default=8 * 1024 * 1024)
    parser.add_argument("--chunks-per-level", type=positive_int, default=64)
    parser.add_argument("--seed", type=int, default=0xC0FFEE)
    parser.add_argument("shards", nargs="+", help="checkpoint shard paths to read from")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        record = probe_read_concurrency(
            options.shards,
            levels=options.levels,
            chunk_bytes=options.chunk_bytes,
            chunks_per_level=options.chunks_per_level,
            seed=options.seed,
        )
    except (ProbeError, OSError) as error:
        raise SystemExit(f"FAIL: {error}") from None
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
