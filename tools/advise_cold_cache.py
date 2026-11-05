#!/usr/bin/env python3
"""Issue and verify file-scoped cold-cache advice for immutable checkpoint shards.

POSIX_FADV_DONTNEED is advisory.  After issuing it, this tool takes a Linux ``mincore``
snapshot and fails when more than the explicitly allowed number of file bytes remain
resident.  The snapshot is evidence of cache state at that instant, not a promise that a
different process will not fault a page afterward.

usage: advise_cold_cache.py [--max-resident-bytes N] <shard> [<shard> ...]
"""

import argparse
import ctypes
import json
import mmap
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path


STATUS = "advisory-applied-mincore-verified"
ADVICE = "POSIX_FADV_DONTNEED"
MINCORE_CHUNK_BYTES = 1024 * 1024 * 1024


class AdvisoryError(ValueError):
    """An input cannot safely participate in the advisory operation."""


@dataclass(frozen=True)
class OpenShard:
    path: Path
    fd: int
    device: int
    inode: int
    size: int
    mtime_ns: int


def require_platform_support():
    if not sys.platform.startswith("linux"):
        raise AdvisoryError("mincore verification requires Linux")
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise AdvisoryError(f"{ADVICE} is unavailable on this platform")


def open_shard(argument):
    path = Path(argument).absolute()
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as error:
        raise AdvisoryError(f"cannot open shard {path} read-only: {error}") from error

    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise AdvisoryError(f"shard is not a regular file: {path}")
        return OpenShard(
            path=path,
            fd=fd,
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
        )
    except BaseException:
        os.close(fd)
        raise


def close_shards(shards):
    for shard in shards:
        os.close(shard.fd)


def validate_unique(shards):
    identities = set()
    for shard in shards:
        identity = (shard.device, shard.inode)
        if identity in identities:
            raise AdvisoryError(
                f"duplicate shard identity {shard.device}:{shard.inode}: {shard.path}"
            )
        identities.add(identity)


def check_stable(shard):
    metadata = os.fstat(shard.fd)
    if (metadata.st_dev, metadata.st_ino) != (shard.device, shard.inode):
        raise AdvisoryError(f"open shard identity changed unexpectedly: {shard.path}")
    if metadata.st_size != shard.size:
        raise AdvisoryError(
            f"shard size changed while advice was issued: {shard.path}: "
            f"{shard.size} -> {metadata.st_size} bytes"
        )
    if metadata.st_mtime_ns != shard.mtime_ns:
        raise AdvisoryError(
            f"shard modification time changed while advice was issued: {shard.path}"
        )

    try:
        path_metadata = os.stat(shard.path, follow_symlinks=False)
    except OSError as error:
        raise AdvisoryError(f"cannot recheck shard path {shard.path}: {error}") from error
    if not stat.S_ISREG(path_metadata.st_mode):
        raise AdvisoryError(f"shard path stopped naming a regular file: {shard.path}")
    if (path_metadata.st_dev, path_metadata.st_ino) != (shard.device, shard.inode):
        raise AdvisoryError(f"shard path was replaced while advice was issued: {shard.path}")
    if path_metadata.st_size != shard.size:
        raise AdvisoryError(
            f"shard path size changed while advice was issued: {shard.path}: "
            f"{shard.size} -> {path_metadata.st_size} bytes"
        )


class MincoreProbe:
    """Page-residency probe with constant-sized mappings and residency vectors."""

    def __init__(self):
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError) as error:
            raise AdvisoryError(f"cannot determine Linux page size: {error}") from error
        if not isinstance(page_size, int) or page_size <= 0:
            raise AdvisoryError(f"invalid Linux page size: {page_size!r}")
        self.page_size = page_size
        self.chunk_bytes = (MINCORE_CHUNK_BYTES // page_size) * page_size
        if self.chunk_bytes == 0:
            raise AdvisoryError("Linux page size exceeds the mincore chunk bound")

        try:
            libc = ctypes.CDLL(None, use_errno=True)
            mmap_function = libc.mmap
            mincore_function = libc.mincore
            munmap_function = libc.munmap
        except (OSError, AttributeError) as error:
            raise AdvisoryError(f"Linux mincore verification is unavailable: {error}") from error

        mmap_function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_longlong,
        ]
        mmap_function.restype = ctypes.c_void_p
        mincore_function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_ubyte),
        ]
        mincore_function.restype = ctypes.c_int
        munmap_function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        munmap_function.restype = ctypes.c_int
        self._mmap = mmap_function
        self._mincore = mincore_function
        self._munmap = munmap_function

    def _probe_chunk(self, shard, offset, length):
        address = self._mmap(
            None,
            length,
            mmap.PROT_READ,
            mmap.MAP_SHARED,
            shard.fd,
            offset,
        )
        if address == ctypes.c_void_p(-1).value:
            error = ctypes.get_errno()
            raise AdvisoryError(
                f"mmap for mincore failed at {shard.path}:{offset}+{length}: "
                f"{os.strerror(error)}"
            )

        page_count = (length + self.page_size - 1) // self.page_size
        vector = (ctypes.c_ubyte * page_count)()
        probe_error = None
        if self._mincore(address, length, vector) != 0:
            error = ctypes.get_errno()
            probe_error = AdvisoryError(
                f"mincore failed at {shard.path}:{offset}+{length}: {os.strerror(error)}"
            )
        unmap_error = None
        if self._munmap(address, length) != 0:
            error = ctypes.get_errno()
            unmap_error = AdvisoryError(
                f"munmap failed after mincore for {shard.path}:{offset}+{length}: "
                f"{os.strerror(error)}"
            )
        if probe_error is not None:
            raise probe_error
        if unmap_error is not None:
            raise unmap_error

        resident_pages = 0
        resident_bytes = 0
        for index, value in enumerate(vector):
            if value & 1:
                resident_pages += 1
                resident_bytes += min(self.page_size, length - index * self.page_size)
        return page_count, resident_pages, resident_bytes

    def __call__(self, shard):
        pages = 0
        resident_pages = 0
        resident_bytes = 0
        offset = 0
        while offset < shard.size:
            length = min(self.chunk_bytes, shard.size - offset)
            chunk_pages, chunk_resident_pages, chunk_resident_bytes = self._probe_chunk(
                shard, offset, length
            )
            pages += chunk_pages
            resident_pages += chunk_resident_pages
            resident_bytes += chunk_resident_bytes
            offset += length
        return {
            "pages": pages,
            "resident_bytes": resident_bytes,
            "resident_pages": resident_pages,
        }


def advise_cold_cache(arguments, max_resident_bytes=0, advise=None, verify=None):
    if not arguments:
        raise AdvisoryError("at least one explicit shard path is required")
    if not isinstance(max_resident_bytes, int) or max_resident_bytes < 0:
        raise AdvisoryError("maximum resident bytes must be a non-negative integer")
    require_platform_support()
    advise = os.posix_fadvise if advise is None else advise
    verify = MincoreProbe() if verify is None else verify

    shards = []
    try:
        # Open and validate the complete set before changing cache policy for any member.
        for argument in arguments:
            shards.append(open_shard(argument))
        validate_unique(shards)

        for shard in shards:
            try:
                advise(shard.fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except OSError as error:
                raise AdvisoryError(
                    f"{ADVICE} failed for {shard.path}: {error}"
                ) from error

        for shard in shards:
            check_stable(shard)

        files = []
        total_pages = 0
        total_resident_pages = 0
        total_resident_bytes = 0
        for shard in shards:
            residency = verify(shard)
            pages = residency["pages"]
            resident_pages = residency["resident_pages"]
            resident_bytes = residency["resident_bytes"]
            if min(pages, resident_pages, resident_bytes) < 0 or resident_pages > pages:
                raise AdvisoryError(f"mincore verifier returned invalid counts for {shard.path}")
            files.append(
                {
                    "bytes": shard.size,
                    "pages": pages,
                    "path": str(shard.path),
                    "resident_bytes": resident_bytes,
                    "resident_pages": resident_pages,
                }
            )
            total_pages += pages
            total_resident_pages += resident_pages
            total_resident_bytes += resident_bytes

        for shard in shards:
            check_stable(shard)
        if total_resident_bytes > max_resident_bytes:
            raise AdvisoryError(
                f"mincore found {total_resident_bytes} resident bytes after {ADVICE}; "
                f"maximum is {max_resident_bytes}"
            )

        return {
            "advice": ADVICE,
            "advice_is_advisory": True,
            "file_count": len(shards),
            "files": files,
            "max_resident_bytes": max_resident_bytes,
            "mincore_chunk_bytes": (
                verify.chunk_bytes if isinstance(verify, MincoreProbe) else None
            ),
            "page_size": verify.page_size if isinstance(verify, MincoreProbe) else None,
            "pages": total_pages,
            "resident_bytes": total_resident_bytes,
            "resident_pages": total_resident_pages,
            "status": STATUS,
            "total_bytes": sum(shard.size for shard in shards),
            "verification": "mincore-snapshot",
            "verified": True,
        }
    finally:
        close_shards(shards)


def non_negative_integer(value):
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-resident-bytes",
        type=non_negative_integer,
        default=0,
        help="fail when the post-advice mincore snapshot exceeds this byte count (default: 0)",
    )
    parser.add_argument("shards", nargs="+", help="explicit immutable regular-file paths")
    return parser.parse_args(argv)


def main(argv=None):
    options = parse_arguments(sys.argv[1:] if argv is None else argv)
    try:
        record = advise_cold_cache(options.shards, options.max_resident_bytes)
    except (AdvisoryError, OverflowError) as error:
        raise SystemExit(f"FAIL: {error}") from None
    print(json.dumps(record, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
