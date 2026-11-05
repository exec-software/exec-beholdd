#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import advise_cold_cache


class AdviseColdCacheTests(unittest.TestCase):
    @staticmethod
    def empty_residency(shard):
        page_size = os.sysconf("SC_PAGE_SIZE")
        pages = (shard.size + page_size - 1) // page_size
        return {"pages": pages, "resident_pages": 0, "resident_bytes": 0}

    def test_success_records_exact_files_bytes_and_advisory_status(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.gguf")
            second = Path(directory, "second.gguf")
            first.write_bytes(b"abc")
            second.write_bytes(b"defgh")
            calls = []

            def record_call(fd, offset, length, advice):
                calls.append((fd, offset, length, advice))

            record = advise_cold_cache.advise_cold_cache(
                [str(first), str(second)],
                advise=record_call,
                verify=self.empty_residency,
            )

            self.assertEqual(len(calls), 2)
            self.assertTrue(all(call[1:] == (0, 0, os.POSIX_FADV_DONTNEED) for call in calls))
            self.assertEqual(record["advice"], "POSIX_FADV_DONTNEED")
            self.assertTrue(record["advice_is_advisory"])
            self.assertEqual(record["file_count"], 2)
            self.assertEqual(record["total_bytes"], 8)
            self.assertEqual(record["status"], "advisory-applied-mincore-verified")
            self.assertTrue(record["verified"])
            self.assertEqual(record["resident_pages"], 0)
            self.assertEqual(record["resident_bytes"], 0)
            self.assertEqual(record["verification"], "mincore-snapshot")
            self.assertEqual(
                record["files"],
                [
                    {
                        "bytes": 3,
                        "pages": 1,
                        "path": str(first.absolute()),
                        "resident_bytes": 0,
                        "resident_pages": 0,
                    },
                    {
                        "bytes": 5,
                        "pages": 1,
                        "path": str(second.absolute()),
                        "resident_bytes": 0,
                        "resident_pages": 0,
                    },
                ],
            )

    def test_rejects_duplicate_file_identity_before_issuing_advice(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first.gguf")
            alias = Path(directory, "alias.gguf")
            first.write_bytes(b"abc")
            os.link(first, alias)
            calls = []

            with self.assertRaisesRegex(advise_cold_cache.AdvisoryError, "duplicate shard"):
                advise_cold_cache.advise_cold_cache(
                    [str(first), str(alias)],
                    advise=lambda *args: calls.append(args),
                    verify=self.empty_residency,
                )
            self.assertEqual(calls, [])

    def test_rejects_a_non_regular_file_before_issuing_advice(self):
        with tempfile.TemporaryDirectory() as directory:
            regular = Path(directory, "first.gguf")
            regular.write_bytes(b"abc")
            calls = []

            with self.assertRaisesRegex(advise_cold_cache.AdvisoryError, "not a regular file"):
                advise_cold_cache.advise_cold_cache(
                    [str(regular), directory],
                    advise=lambda *args: calls.append(args),
                    verify=self.empty_residency,
                )
            self.assertEqual(calls, [])

    def test_detects_a_size_change_after_advice(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory, "first.gguf")
            shard.write_bytes(b"abc")

            def mutate_after_advice(*_args):
                shard.write_bytes(b"changed")

            with self.assertRaisesRegex(advise_cold_cache.AdvisoryError, "size changed"):
                advise_cold_cache.advise_cold_cache(
                    [str(shard)],
                    advise=mutate_after_advice,
                    verify=self.empty_residency,
                )

    def test_rejects_residency_above_the_explicit_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory, "first.gguf")
            shard.write_bytes(b"abc")

            def resident(_shard):
                return {"pages": 1, "resident_pages": 1, "resident_bytes": 3}

            with self.assertRaisesRegex(advise_cold_cache.AdvisoryError, "maximum is 2"):
                advise_cold_cache.advise_cold_cache(
                    [str(shard)],
                    max_resident_bytes=2,
                    advise=lambda *_args: None,
                    verify=resident,
                )

    def test_mincore_reports_resident_counts_with_bounded_chunks(self):
        with tempfile.TemporaryDirectory() as directory:
            shard_path = Path(directory, "first.gguf")
            page_size = os.sysconf("SC_PAGE_SIZE")
            shard_path.write_bytes(b"x" * (page_size * 2 + 3))
            shard = advise_cold_cache.open_shard(str(shard_path))
            probe = advise_cold_cache.MincoreProbe()
            probe.chunk_bytes = page_size
            try:
                residency = probe(shard)
            finally:
                advise_cold_cache.close_shards([shard])

            self.assertEqual(residency["pages"], 3)
            self.assertGreaterEqual(residency["resident_pages"], 0)
            self.assertLessEqual(residency["resident_pages"], 3)
            self.assertGreaterEqual(residency["resident_bytes"], 0)
            self.assertLessEqual(residency["resident_bytes"], shard.size)
            self.assertLessEqual(probe.chunk_bytes, advise_cold_cache.MINCORE_CHUNK_BYTES)
            self.assertEqual(probe.chunk_bytes % probe.page_size, 0)

    def test_cli_emits_one_compact_json_record(self):
        with tempfile.TemporaryDirectory() as directory:
            shard = Path(directory, "first.gguf")
            shard.write_bytes(b"abc")
            script = Path(advise_cold_cache.__file__)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--max-resident-bytes",
                    str(os.sysconf("SC_PAGE_SIZE")),
                    str(shard),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.stderr, "")
            self.assertEqual(completed.stdout.count("\n"), 1)
            record = json.loads(completed.stdout)
            self.assertEqual(record["total_bytes"], 3)
            self.assertEqual(record["status"], "advisory-applied-mincore-verified")
            self.assertTrue(record["verified"])


if __name__ == "__main__":
    unittest.main()
