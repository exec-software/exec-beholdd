"""Model-free local UI runner tests; all subprocess and system observations are fakes."""
from contextlib import ExitStack
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as common
import ui_client as ui

spec = importlib.util.spec_from_file_location('test_ui_local_runner', Path(__file__).with_name('ui-run-local.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

HOST = {'machine_id_sha256': 'a' * 64, 'boot_id': '11111111-1111-4111-8111-111111111111'}
RUN = '22222222-2222-4222-8222-222222222222'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


class Clock:
    def __init__(self, gap_on=None):
        self.value, self.sleeps, self.gap_on = 100., 0, gap_on

    def monotonic(self):
        return self.value

    def monotonic_ns(self):
        return int(self.value * 10 ** 9)

    def sleep(self, seconds):
        self.sleeps += 1
        self.value += 3. if self.sleeps == self.gap_on else seconds


class UiLocalRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='HAG UI runner ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def prepared_inputs(self):
        metadata = self.root / 'metadata'
        source = {'commit': ui.UI_COMMIT, 'committed_sources': {'source.ts': 'a' * 64}, 'installed_dependencies': {}}
        runtime = {'electron': 'b' * 64}
        identity = {'profile': ui.UI_PROFILE, 'local_host': HOST, 'base_url': 'http://127.0.0.1:33000/v1',
            'source_verification_sha256': ui.canonical_sha(source),
            'electron': {'manifest_sha256': ui.canonical_sha(runtime)}}
        save(self.root / 'expected-manifest.json', {'kind': 'fixture'})
        save(self.root / 'qualification-contract.json', {'profile': ui.UI_PROFILE,
            'stream_client': 'antimatter-ui', 'ui_client_host': 'separate'})
        save(self.root / 'client/identity.json', identity)
        save(metadata / 'identity.json', identity)
        save(metadata / 'source-verification.json', source)
        save(metadata / 'electron-files.json', runtime)
        save(metadata / 'local-inputs.json', {'prepared_root': str(self.root / 'prepared'),
                                             'electron_dist': str(self.root / 'electron')})
        ready = {'schema': 1, 'profile': ui.UI_PROFILE, 'host': '127.0.0.1', 'port': 33001, 'pid': 901,
            'run_id': RUN, 'client_identity_sha256': common.digest(self.root / 'client/identity.json'),
            'boot_id': '33333333-3333-4333-8333-333333333333', 'delivery': 'buffered-sse-after-parity-reset'}
        save(self.root / 'ready.json', ready)
        args = SimpleNamespace(manifest_sha256=common.digest(self.root / 'expected-manifest.json'),
                               metadata=str(metadata), ready=str(self.root / 'ready.json'), target='tester@validation-host')
        return identity, ready, args

    def test_prepared_input_verification_uses_canonical_receipt_hash(self):
        identity, ready, args = self.prepared_inputs()
        metadata = Path(args.metadata)
        self.assertNotEqual(common.digest(metadata / 'source-verification.json'), identity['source_verification_sha256'])
        with patch.object(runner, 'ROOT', self.root), patch.object(common, 'gate_identity') as gate, \
                patch.object(ui, 'client_identity', return_value=identity), patch.object(ui, 'observe_host', return_value=HOST), \
                patch.object(subprocess, 'Popen', side_effect=AssertionError('input verification must not launch anything')):
            result = runner.verify_inputs(args)
        self.assertEqual(result[:2], (identity, ready))
        self.assertEqual(result[-1][-2:], ['127.0.0.1:33000:127.0.0.1:33001', 'tester@validation-host'])
        gate.assert_called_once_with(self.root)

    def test_host_mode_requires_explicit_topology_and_target_matches_it(self):
        identity, original, args = self.prepared_inputs()
        cases = [('same', None, HOST['boot_id'], 33000, True),
                 ('same', 'tester@validation-host', HOST['boot_id'], 33000, False),
                 ('same', None, original['boot_id'], 33000, False),
                 ('same', None, HOST['boot_id'], 33001, False),
                 ('separate', None, original['boot_id'], 33001, False),
                 ('separate', 'tester@validation-host', HOST['boot_id'], 33001, False)]
        with patch.object(runner, 'ROOT', self.root), patch.object(common, 'gate_identity'), \
                patch.object(ui, 'client_identity', return_value=identity), patch.object(ui, 'observe_host', return_value=HOST), \
                patch.object(subprocess, 'Popen', side_effect=AssertionError('verification must not launch anything')):
            for mode, target, boot, port, valid in cases:
                save(self.root / 'qualification-contract.json', {'profile': ui.UI_PROFILE,
                    'stream_client': 'antimatter-ui', 'ui_client_host': mode})
                save(Path(args.ready), {**original, 'boot_id': boot, 'port': port})
                args.target = target
                with self.subTest(mode=mode, target=target, boot=boot, port=port):
                    if valid:
                        self.assertIsNone(runner.verify_inputs(args)[-1])
                    else:
                        with self.assertRaises(ValueError):
                            runner.verify_inputs(args)

    def test_worker_direct_path_launches_only_electron_and_preserves_separate_tunnel(self):
        for mode in ('same', 'separate'):
            with self.subTest(mode=mode):
                directory = self.root / mode
                output, prepared, group = (directory / name for name in ('output', 'prepared', 'cgroup'))
                for path in (output, prepared, group):
                    path.mkdir(parents=True)
                for key, value in ui.UI_LIMITS.items():
                    (group / key).write_text(value)
                (group / 'cgroup.procs').write_text('900 901\n')
                (output / 'start').touch()
                (output / 'finish').touch()
                launch = ['/frozen/electron', '--no-sandbox', '--ozone-platform=headless',
                          str(prepared / 'launcher.mjs'), '--fixture', str(prepared),
                          '--live-base-url=http://127.0.0.1:33000/v1']
                save(prepared / 'prepared-live.json', {'command': launch, 'environment': {}})
                save(prepared / 'report.json', {'process': {'pid': 901}})
                identity = {'profile': ui.UI_PROFILE, 'local_host': HOST, 'base_url': 'http://127.0.0.1:33000/v1',
                            'electron': {'manifest_sha256': 'a' * 64}}
                ready = {'pid': 905, 'port': 33000 if mode == 'same' else 33001, 'run_id': RUN}
                save(directory / 'client/identity.json', identity)
                save(directory / 'ready.json', ready)
                args = SimpleNamespace(output=str(output), ready=str(directory / 'ready.json'), unit='hag-ui-test.service')
                tunnel_args = None if mode == 'same' else runner.tunnel_command(identity, ready, 'tester@validation-host')
                electron = Mock(pid=901, returncode=0)
                polls = iter([None, 0])
                electron.poll.side_effect = lambda: next(polls, 0)
                tunnel = Mock(pid=902)
                tunnel.poll.return_value = None
                clock = Clock()
                def observation(pid, role, observed_group):
                    return {'pid': pid, 'role': role, 'start_time_ticks': pid,
                            'command': launch if role == 'electron' else ['/fake/' + role], 'cgroup': str(observed_group)}
                with ExitStack() as stack:
                    for target, name, options in (
                        (runner, 'ROOT', {'new': directory}),
                        (runner, 'verify_inputs', {'return_value': (identity, ready, prepared, directory / 'dist', {}, tunnel_args)}),
                        (common, 'cgroup_path', {'return_value': group}),
                        (ui, 'observe_process', {'side_effect': observation}),
                        (ui, 'observe_host', {'return_value': HOST}),
                        (ui, 'validate_local_bundle', {}), (ui, 'runtime_manifest', {'return_value': {}}),
                        (runner.os, 'getpid', {'return_value': 900}),
                        (runner.signal, 'signal', {}), (runner.signal, 'setitimer', {}),
                        (runner.time, 'monotonic', {'side_effect': clock.monotonic}),
                        (runner.time, 'monotonic_ns', {'side_effect': clock.monotonic_ns}),
                        (runner.time, 'sleep', {'side_effect': clock.sleep}),
                        (runner, 'stop_child', {})):
                        stack.enter_context(patch.object(target, name, **options))
                    popen = stack.enter_context(patch.object(subprocess, 'Popen',
                        side_effect=[electron] if mode == 'same' else [tunnel, electron]))
                    listener = stack.enter_context(patch.object(runner, 'owns_listener', return_value=True))
                    runner.worker(args)
                self.assertEqual(popen.call_args_list[-1].args[0], launch)
                self.assertEqual(popen.call_count, 1 if mode == 'same' else 2)
                listener.assert_called_once_with(905 if mode == 'same' else 902, 33000)
                receipt = common.read_json(output / 'ui-process.json')
                self.assertEqual([row['role'] for row in receipt['processes']],
                    ['supervisor', 'electron'] if mode == 'same' else ['supervisor', 'tunnel', 'electron'])
                self.assertEqual(receipt['exit_code'], 0)
                self.assertFalse(receipt['timed_out'])
                self.assertEqual((output / 'tunnel.log').exists(), mode == 'separate')

    def test_dedicated_tunnel_has_exact_loopback_mapping_and_no_shared_control_socket(self):
        command = runner.tunnel_command({'base_url': 'http://127.0.0.1:33000/v1'}, {'port': 33001}, 'tester@validation-host')
        self.assertEqual(command, ['/usr/bin/ssh', '-N', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', '-o', 'ControlMaster=no', '-o',
            'ControlPath=none', '-L', '127.0.0.1:33000:127.0.0.1:33001', 'tester@validation-host'])
        for target in ('-oProxyCommand=bad', 'tester@host;command', 'host', 'tester@127.0.0.1/evil'):
            with self.subTest(target=target), self.assertRaises(ValueError):
                runner.tunnel_command({'base_url': 'http://127.0.0.1:33000/v1'}, {'port': 33001}, target)

    def test_process_environment_cannot_inject_unfrozen_runtime_code(self):
        values = {key: 'unsafe' for key in runner.INJECT_ENV}
        with patch.dict(os.environ, {**values, 'PATH': '/usr/bin', 'SSH_AUTH_SOCK': '/agent/socket'}, clear=True):
            clean = runner.clean_environment()
        self.assertTrue(all(key not in clean for key in runner.INJECT_ENV))
        self.assertEqual(clean['SSH_AUTH_SOCK'], '/agent/socket')

    def test_listener_belongs_to_ssh_inode_and_exact_ipv4_loopback_port(self):
        proc = self.root / 'proc'
        fd, tcp = proc / '901/fd', proc / '901/net/tcp'
        fd.mkdir(parents=True)
        tcp.parent.mkdir()
        (fd / '3').symlink_to('socket:[12345]')
        line = '  0: 0100007F:80E8 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 12345\n'
        tcp.write_text('header\n' + line)
        self.assertTrue(runner.owns_listener(901, 33000, proc_root=proc))
        tcp.write_text('header\n' + line.replace('12345', '99999'))
        self.assertFalse(runner.owns_listener(901, 33000, proc_root=proc))
        tcp.write_text('header\n' + line.replace('0100007F', '00000000'))
        self.assertFalse(runner.owns_listener(901, 33000, proc_root=proc))

    def test_retired_child_races_are_skipped_but_live_containment_errors_fail(self):
        proc = self.root / 'proc'
        directory = proc / '901'
        directory.mkdir(parents=True)
        with patch.object(ui, 'observe_process', side_effect=FileNotFoundError()):
            self.assertIsNone(runner.observe_child(901, Path('/sys/fs/cgroup/fixture'), proc))
        with patch.object(ui, 'observe_process', side_effect=ProcessLookupError()):
            self.assertIsNone(runner.observe_child(901, Path('/sys/fs/cgroup/fixture'), proc))
        (directory / 'stat').write_text('901 (retired helper) Z 0 0\n')
        with patch.object(ui, 'observe_process', side_effect=ValueError('invalid UI process command')):
            self.assertIsNone(runner.observe_child(901, Path('/sys/fs/cgroup/fixture'), proc))
        (directory / 'stat').write_text('901 (live helper) S 0 0\n')
        with patch.object(ui, 'observe_process', side_effect=ValueError('UI process escaped local containment')):
            with self.assertRaisesRegex(ValueError, 'escaped'):
                runner.observe_child(901, Path('/sys/fs/cgroup/fixture'), proc)

    def test_atomic_receipt_publish_does_not_overwrite_prior_run(self):
        path = self.root / 'worker-ready.json'
        runner.publish(path, {'pid': 901})
        self.assertEqual(common.read_json(path), {'pid': 901})
        self.assertFalse(path.with_name(path.name + '.writing').exists())
        with self.assertRaises(FileExistsError):
            runner.publish(path, {'pid': 902})
        self.assertEqual(common.read_json(path), {'pid': 901})

    def test_cooldown_restarts_after_heat_and_after_missing_sample_interval(self):
        for kind in ('heat', 'gap'):
            clock = Clock(gap_on=2 if kind == 'gap' else None)
            readings = iter([60] * 5 + [81] + [60] * 100) if kind == 'heat' else iter([60] * 100)
            with self.subTest(kind=kind), patch.object(runner, 'temperature', side_effect=lambda _path: next(readings)), \
                    patch.object(runner.time, 'monotonic', side_effect=clock.monotonic), \
                    patch.object(runner.time, 'monotonic_ns', side_effect=clock.monotonic_ns), \
                    patch.object(runner.time, 'sleep', side_effect=clock.sleep):
                samples = runner.cooldown(Path('/fake/sensor'))
            self.assertGreaterEqual(samples[0]['monotonic_ns'], int((103 if kind == 'heat' else 103.5) * 10 ** 9))
            self.assertGreaterEqual(samples[-1]['monotonic_ns'] - samples[0]['monotonic_ns'], 10 * 10 ** 9)
            self.assertTrue(all(0 < b['monotonic_ns'] - a['monotonic_ns'] <= 10 ** 9 for a, b in zip(samples, samples[1:])))

    def test_sampler_rejects_oom_swap_thermal_and_headroom_limits_before_admission(self):
        normal = {'memory_current_bytes': 1000, 'memory_peak_bytes': 1000, 'swap_current_bytes': 0,
                  'memory_events': {'low': 0, 'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        for failure in ('oom', 'swap', 'heat', 'headroom', 'peak'):
            memory = {**normal, 'memory_events': dict(normal['memory_events'])}
            if failure == 'oom': memory['memory_events']['oom'] = 1
            if failure == 'swap': memory['swap_current_bytes'] = 1
            if failure == 'peak': memory['memory_peak_bytes'] = common.STOP_CGROUP_BYTES
            with self.subTest(failure=failure), patch.object(common, 'memory_snapshot', return_value=memory), \
                    patch.object(common, 'available_memory', return_value=4 * common.GIB - 1 if failure == 'headroom' else 9 * common.GIB), \
                    patch.object(runner, 'temperature', return_value=90 if failure == 'heat' else 60):
                with self.assertRaisesRegex(ValueError, 'cutoff'):
                    runner.sample(Path('/fake/cgroup'), Path('/fake/sensor'))

    def failed_supervisor(self, cleanup_error=None):
        args = SimpleNamespace(metadata=str(self.root / 'metadata'), ready=str(self.root / 'ready.json'),
            output=str(self.root / 'output'), target='tester@validation-host', manifest_sha256='a' * 64)
        child = Mock()
        child.poll.return_value = 1
        child.wait.return_value = 1
        with ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'verify_inputs', return_value=({'local_host': HOST}, {'run_id': RUN})))
            stack.enter_context(patch.object(runner, 'sensor', return_value=Path('/fake/sensor')))
            stack.enter_context(patch.object(runner, 'cooldown', return_value=[]))
            stack.enter_context(patch.object(common, 'available_memory', return_value=9 * common.GIB))
            stack.enter_context(patch.object(subprocess, 'Popen', return_value=child))
            cleanup = stack.enter_context(patch.object(subprocess, 'run', return_value=SimpleNamespace(returncode=0),
                                                        side_effect=cleanup_error))
            with self.assertRaisesRegex(ValueError, 'local UI service failed'):
                runner.supervise(args)
        return cleanup, common.read_json(self.root / 'output/local-failure.json')

    def test_failed_systemd_run_client_still_stops_its_transient_unit(self):
        cleanup, failure = self.failed_supervisor()
        self.assertEqual(cleanup.call_count, 1)
        self.assertEqual(cleanup.call_args.args[0][:3], ['systemctl', '--user', 'stop'])
        self.assertTrue(cleanup.call_args.args[0][3].startswith('hag-ui-'))
        self.assertFalse(failure['backend_qualified'])
        self.assertIn('local UI service failed', failure['reason'])

    def test_cleanup_failure_preserves_original_failure_evidence(self):
        _cleanup, failure = self.failed_supervisor(subprocess.TimeoutExpired(['systemctl'], 8))
        self.assertIn('local UI service failed', failure['reason'])
        self.assertTrue(any('TimeoutExpired' in text for text in failure['cleanup_errors']))


if __name__ == '__main__':
    unittest.main()
