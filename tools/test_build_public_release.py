"""Containment and artifact-boundary regressions; no Docker, model or GPU work."""
import io
import hashlib
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest.mock import patch
import subprocess

import build_public_release as release


class ReleaseBoundaryTests(unittest.TestCase):
    def test_low_memory_refuses_before_launch(self):
        with self.assertRaisesRegex(ValueError, 'no command started'):
            release.available_memory('MemAvailable: 10485760 kB\n')
        self.assertEqual(release.available_memory('MemAvailable: 11534336 kB\n'), 11 * 1024**3)

    def test_missing_available_estimate_fails_closed(self):
        with self.assertRaises(ValueError):
            release.available_memory('MemFree: 99999999 kB\n')

    def test_inside_mode_requires_exact_cgroup_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in [('memory.max', str(release.MEMORY)), ('memory.swap.max', '0'),
                                ('cpu.max', '150000 100000'), ('pids.max', str(release.PIDS))]:
                (root / name).write_text(value)
            release.validate_cgroup(root)
            for name, value in [('memory.max', 'max'), ('memory.swap.max', '4096'),
                                ('cpu.max', 'max 100000'), ('pids.max', 'max')]:
                original = (root / name).read_text()
                (root / name).write_text(value)
                with self.assertRaises(ValueError):
                    release.validate_cgroup(root)
                (root / name).write_text(original)

    def test_source_archive_rejects_links_and_traversal(self):
        for name, kind in [('../escape', tarfile.REGTYPE), ('/escape', tarfile.REGTYPE),
                           ('link', tarfile.SYMTYPE), ('hardlink', tarfile.LNKTYPE)]:
            with self.subTest(name=name), io.BytesIO() as buffer:
                with tarfile.open(fileobj=buffer, mode='w') as archive:
                    member = tarfile.TarInfo(name)
                    member.type = kind
                    archive.addfile(member)
                buffer.seek(0)
                with tarfile.open(fileobj=buffer) as archive, self.assertRaises(ValueError):
                    release.safe_archive_members(archive)

    def test_binary_path_and_abi_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'binary'
            binary.write_bytes(b'\x7fELFsynthetic')
            self.assertEqual(release.verify_binary(binary, 'GLIBC_2.17 GLIBC_2.31'), ['2.17', '2.31'])
            for versions in ['GLIBC_2.35', 'no version information']:
                with self.assertRaises(ValueError):
                    release.verify_binary(binary, versions)
            binary.write_bytes(b'\x7fELF/home/example/source')
            with self.assertRaises(ValueError):
                release.verify_binary(binary, 'GLIBC_2.31')

    def test_timeout_removes_only_its_bounded_container_without_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = release.Builder(Path(temporary))
            builder.logs.mkdir()
            with patch.object(release, 'admission', return_value=11 * 1024**3), \
                    patch.object(release.subprocess, 'run', side_effect=[
                        subprocess.CompletedProcess([], 0),
                        subprocess.TimeoutExpired(['docker', 'start'], release.TIMEOUT),
                        subprocess.CompletedProcess([], 0),
                    ]) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    builder.stage('fixture', ['run', 'fixture-command'])
            self.assertEqual(run.call_count, 3)
            create = run.call_args_list[0].args[0]
            for option, value in [('--memory', str(release.MEMORY)), ('--memory-swap', str(release.MEMORY)),
                                  ('--cpus', '1.5'), ('--pids-limit', str(release.PIDS)),
                                  ('--network', 'none'), ('--cap-drop', 'ALL')]:
                self.assertEqual(create[create.index(option) + 1], value)
            self.assertNotIn('--privileged', create)
            self.assertNotIn('--device', create)
            self.assertIn('TMPDIR=/work/tmp', create)
            for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
                self.assertIn(name + '=1', create)
            self.assertIn('noexec', create[create.index('--tmpfs') + 1])
            self.assertEqual(run.call_args_list[-1].args[0],
                             ['docker', 'rm', '--force', create[create.index('--name') + 1]])
            self.assertEqual(builder.stages, [])

    def test_termination_unwinds_container_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = release.Builder(Path(temporary))
            builder.logs.mkdir()
            with patch.object(release, 'admission', return_value=11 * 1024**3), \
                    patch.object(release.subprocess, 'run', side_effect=[
                        subprocess.CompletedProcess([], 0), SystemExit(143),
                        subprocess.CompletedProcess([], 0),
                    ]) as run, self.assertRaises(SystemExit):
                builder.stage('fixture', ['run', 'fixture-command'])
            self.assertEqual(run.call_args_list[-1].args[0][:3], ['docker', 'rm', '--force'])
            self.assertEqual(run.call_count, 3)

    def test_license_source_cannot_escape_the_build_roots(self):
        metadata = {'packages': [{'id': 'untrusted', 'manifest_path': '/unreviewed/Cargo.toml'}],
                    'resolve': {'root': 'untrusted', 'nodes': [{'id': 'untrusted', 'deps': []}]}}
        with tempfile.TemporaryDirectory() as temporary, self.assertRaisesRegex(ValueError, 'outside'):
            release.license_inventory(metadata, Path(temporary))

    def test_rust_notice_uses_exact_scoped_distribution_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notices = root / 'sysroot/share/doc/rust'
            notices.mkdir(parents=True)
            (notices / 'COPYRIGHT-library.html').write_text('synthetic standard library notice')
            (notices / 'COPYRIGHT.html').write_text('broader synthetic notice')
            result = release.collect_rust_notices(root / 'sysroot', root / 'output', 'synthetic rustc identity')
            self.assertEqual(result['scope'], 'standard-library')
            self.assertEqual([record['file'] for record in result['files']], ['COPYRIGHT-library.html'])
            self.assertEqual(result['files'][0]['sha256'], release.digest(notices / 'COPYRIGHT-library.html'))

    def test_rust_notice_absence_fails_and_broader_fallback_is_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            notices = root / 'sysroot/share/doc/rust'
            notices.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'no standard-library'):
                release.collect_rust_notices(root / 'sysroot', root / 'missing', 'synthetic rustc')
            (notices / 'COPYRIGHT.html').write_text('synthetic full distribution notice')
            result = release.collect_rust_notices(root / 'sysroot', root / 'output', 'synthetic rustc')
            self.assertEqual(result['scope'], 'full-distribution')
            self.assertEqual([record['file'] for record in result['files']], ['COPYRIGHT.html'])

    def test_python_wheel_fetch_is_bound_to_exact_bytes(self):
        good = b'synthetic wheel payload'
        pin = {'filename': 'fixture.whl', 'url': 'https://files.pythonhosted.org/packages/fixture.whl',
               'sha256': hashlib.sha256(good).hexdigest(), 'bytes': len(good)}
        for payload, expected in [(good, True), (good[:-1], False), (good + b'x', False),
                                  (b'x' * len(good), False)]:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / 'wheels'
                with patch.object(release.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
                    if expected:
                        release.download_python_wheel(pin, target)
                        self.assertEqual((target / 'fixture.whl').read_bytes(), good)
                    else:
                        with self.assertRaises(ValueError):
                            release.download_python_wheel(pin, target)
                        self.assertEqual(list(target.iterdir()), [])


if __name__ == '__main__':
    unittest.main()
