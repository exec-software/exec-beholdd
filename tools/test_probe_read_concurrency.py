#!/usr/bin/env python3

import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_read_concurrency as prc


class PickOffsetsTests(unittest.TestCase):
    def test_offsets_never_repeat_across_levels(self):
        assignment = prc.pick_offsets(
            file_sizes=[64], levels=[1, 2, 4], chunk_bytes=4, chunks_per_level=3, seed=1
        )
        seen = set()
        for level, tasks in assignment.items():
            self.assertEqual(len(tasks), 3)
            for task in tasks:
                key = (task.file_index, task.offset)
                self.assertNotIn(key, seen, f"offset {key} reused across levels")
                seen.add(key)
                self.assertEqual(task.offset % 4, 0)

    def test_is_deterministic_given_a_seed(self):
        first = prc.pick_offsets([64], [1, 2], 4, 2, seed=7)
        second = prc.pick_offsets([64], [1, 2], 4, 2, seed=7)
        self.assertEqual(first, second)

    def test_refuses_when_not_enough_distinct_chunks_exist(self):
        with self.assertRaises(prc.ProbeError):
            prc.pick_offsets([8], [1, 2], chunk_bytes=4, chunks_per_level=2, seed=1)

    def test_refuses_nonpositive_chunk_or_level_size(self):
        with self.assertRaises(prc.ProbeError):
            prc.pick_offsets([64], [1], chunk_bytes=0, chunks_per_level=1, seed=1)
        with self.assertRaises(prc.ProbeError):
            prc.pick_offsets([64], [1], chunk_bytes=4, chunks_per_level=0, seed=1)


class ProbeReadConcurrencyTests(unittest.TestCase):
    def test_reports_correct_bytes_and_never_rereads_an_offset(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [str(Path(directory, f"shard-{i}.bin")) for i in range(2)]
            for path in paths:
                Path(path).write_bytes(bytes(range(32)))

            calls = []
            lock = threading.Lock()

            def counting_read(fd, offset, length):
                with lock:
                    calls.append((fd, offset, length))
                return prc._pread(fd, offset, length)

            record = prc.probe_read_concurrency(
                paths,
                levels=[1, 2, 4],
                chunk_bytes=4,
                chunks_per_level=2,
                seed=3,
                read=counting_read,
            )

            self.assertEqual([level["concurrency"] for level in record["levels"]], [1, 2, 4])
            for level in record["levels"]:
                self.assertEqual(level["bytes"], level["chunks"] * 4)
                self.assertGreaterEqual(level["elapsed_s"], 0.0)

            # 3 levels * 2 chunks/level = 6 reads total, no offset read twice.
            self.assertEqual(len(calls), 6)
            self.assertEqual(len({(c[0], c[1]) for c in calls}), 6)

    def test_closes_every_opened_file_even_when_a_read_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "shard.bin"))
            Path(path).write_bytes(bytes(range(32)))
            opened = []
            closed = []

            def open_file(p):
                fd = prc.os.open(p, prc.os.O_RDONLY)
                opened.append(fd)
                return fd

            def failing_read(fd, offset, length):
                raise OSError("injected failure")

            with self.assertRaises(OSError):
                prc.probe_read_concurrency(
                    [path],
                    levels=[1],
                    chunk_bytes=4,
                    chunks_per_level=1,
                    seed=1,
                    open_file=open_file,
                    close_file=lambda fd: closed.append(fd) or prc.os.close(fd),
                    read=failing_read,
                )
            self.assertEqual(opened, closed)

    def test_refuses_an_empty_shard_list(self):
        with self.assertRaises(prc.ProbeError):
            prc.probe_read_concurrency([], levels=[1])

    def test_refuses_nonpositive_levels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory, "shard.bin"))
            Path(path).write_bytes(bytes(range(32)))
            with self.assertRaises(prc.ProbeError):
                prc.probe_read_concurrency([path], levels=[0])


if __name__ == "__main__":
    unittest.main()
