#!/usr/bin/env python3
"""Hardware-free checks for explicit HIP SDK configuration."""

import os
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = sorted((ROOT / "tools" / "hip").glob("build_*.sh"))


class HipConfigurationTests(unittest.TestCase):
    def test_wrappers_have_valid_shell_syntax(self):
        self.assertTrue(WRAPPERS)
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper.name):
                result = subprocess.run(
                    ["bash", "-n", str(wrapper)], capture_output=True, text=True, check=False
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_sdk_refuses_before_compilation(self):
        environment = dict(os.environ)
        environment.pop("ROCM_CORE", None)
        environment["ROCWMMA_INCLUDE"] = "/nonexistent/beholdd-rocwmma-test"
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper.name):
                result = subprocess.run(
                    ["bash", str(wrapper)],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Set ROCM_CORE", result.stderr)
                self.assertEqual(result.stdout, "")

    def test_wrappers_do_not_embed_developer_home_paths(self):
        for wrapper in WRAPPERS:
            with self.subTest(wrapper=wrapper.name):
                source = wrapper.read_text()
                self.assertNotIn("/home/", source)
                self.assertNotIn("/Users/", source)


if __name__ == "__main__":
    unittest.main()
