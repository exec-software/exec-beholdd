"""Model-free UI profile, entry point, supervisor and closeout integration."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import gate_common as common
import inputs
import profiles
import ui_client
import ui_history
from hag_qualify import __main__ as cli
from hag_qualify import gate_common as cli_common, inputs as cli_inputs, ui_client as cli_ui_client

PROFILE = profiles.UI_HISTORY_PROFILE


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('ui_wrappers_runner', 'run-gate.py')
child = load('ui_wrappers_child', 'gate-child.py')
closeout = load('ui_wrappers_closeout', 'closeout.py')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


class UiWrappers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = inputs.profile_contract(PROFILE, 'antimatter-ui')
        self.client = {'kind': 'antimatter-ui', 'profile': PROFILE, 'commit': 'fixture',
                       'base_url': 'http://127.0.0.1:48321/v1',
                       'local_host': {'boot_id': '262efb57-e5c3-4393-9144-6bf66a558a48'}}
        save(self.root / 'answer-fixtures.json', profiles.fixture_document(PROFILE))
        save(self.root / 'qualification-contract.json', self.contract)

    def test_ui_contract_preserves_backend_limits_and_uses_current_wire(self):
        old = inputs.profile_contract(profiles.GENERATED_HISTORY_PROFILE, 'antimatter')
        for key in ('capacity', 'hag_charge_bytes', 'hag_budget_bytes', 'cpu_model_budget_bytes',
                    'runtime_combined_forecast_bytes', 'process_memory_max_bytes', 'process_early_stop_bytes',
                    'swap_max_bytes', 'cpu_quota_percent', 'tasks_max', 'no_new_privileges',
                    'host_start_headroom_bytes', 'host_stop_headroom_bytes', 'thermal_cutoff_c',
                    'thermal_poll_seconds', 'thermal_start_max_c', 'thermal_start_stable_seconds',
                    'private_cgroup_admission', 'crash_watcher_seconds', 'stage_deadlines',
                    'stage_request_counts', 'stage_evaluation_ceilings', 'turn_evaluation_ceilings'):
            self.assertEqual(self.contract[key], old[key], key)
        expected = {'qualification_kind': 'ui_history_and_backend_parity', 'stream_client': 'antimatter-ui',
            'client_chat_requests': 0, 'client_stream_requests': 2, 'client_nonstream': False,
            'client_request_timeout_seconds': 240, 'client_process_timeout_seconds': 355,
            'wire_stream': True, 'wire_temperature': 'omitted', 'ui_client_host': 'separate',
            'broker_delivery': 'buffered-sse-after-parity-reset', 'ui_closeout_required': True,
            'first_prompt_tokens': 467, 'second_maximum_prompt_tokens': 491,
            'max_new_tokens': 128, 'turn_evaluation_ceilings': [594, 618]}
        self.assertEqual({key: self.contract[key] for key in expected}, expected)
        self.assertEqual(common.approved_contract(self.root), self.contract)
        self.assertEqual(profiles.profile_modes(PROFILE), ('self-check', 'cpu-reference', 'answers'))
        self.assertFalse(any(key.startswith('timing_') for key in self.contract))
        self.assertNotIn('runtime_phase_diagnostics', self.contract)
        document = profiles.fixture_document(PROFILE)
        self.assertEqual(document['first_fixture']['id'], 'ui_turn_1')
        self.assertEqual(document['second_fixture_id'], 'ui_turn_2')
        self.assertNotIn('temperature', document['first_fixture'])

    def test_old_generated_contract_keeps_nonstream_provider_workload(self):
        old = inputs.profile_contract(profiles.GENERATED_HISTORY_PROFILE, 'antimatter')
        expected = {'profile': profiles.GENERATED_HISTORY_PROFILE, 'qualification_kind': 'backend_parity',
                    'stream_client': 'antimatter', 'client_nonstream': True,
                    'client_chat_requests': 2, 'client_stream_requests': 0,
                    'client_request_timeout_seconds': 240, 'client_process_timeout_seconds': 245,
                    'fixture_ids': ['generated_turn_1', 'generated_turn_2'],
                    'stage_request_counts': {'cpu-reference': 2, 'answers': 2},
                    'stage_evaluation_ceilings': {'cpu-reference': 1212, 'answers': 1212}}
        self.assertEqual({key: old[key] for key in expected}, expected)
        for key in ('wire_stream', 'wire_temperature', 'ui_client_host', 'broker_delivery', 'ui_closeout_required'):
            self.assertNotIn(key, old)
        self.assertEqual(profiles.fixture_document(profiles.GENERATED_HISTORY_PROFILE)['first_fixture']['temperature'], 0)
        save(self.root / 'answer-fixtures.json', profiles.fixture_document(profiles.GENERATED_HISTORY_PROFILE))
        save(self.root / 'qualification-contract.json', old)
        self.assertEqual(common.approved_contract(self.root), old)

    def test_same_host_is_explicit_and_old_profiles_refuse_nondefault_topology(self):
        same = inputs.profile_contract(PROFILE, 'antimatter-ui', ui_client_host='same')
        self.assertEqual(same['ui_client_host'], 'same')
        self.assertIn('same host', same['scope'])
        self.assertIn('4 GiB', same['limitations'])
        self.assertEqual({key: value for key, value in same.items() if key not in ('ui_client_host', 'scope', 'limitations')},
                         {key: value for key, value in self.contract.items() if key not in ('ui_client_host', 'scope', 'limitations')})
        save(self.root / 'qualification-contract.json', same)
        self.assertEqual(common.approved_contract(self.root), same)
        self.assertEqual(closeout.qualification_scope(same, True)['ui_client_host'], 'same')
        self.assertEqual(closeout.qualification_scope(self.contract, True)['ui_client_host'], 'separate')
        for profile in profiles.PROFILE_NAMES:
            if profile in profiles.UI_PROFILES:
                continue
            client = 'antimatter' if profile in profiles.CLIENT_PROFILES else 'python-http'
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, 'UI client host'):
                inputs.profile_contract(profile, client, ui_client_host='same')
            old = inputs.profile_contract(profile, client)
            self.assertNotIn('ui_client_host', old)
            save(self.root / 'answer-fixtures.json', profiles.fixture_document(profile))
            save(self.root / 'qualification-contract.json', {**old, 'ui_client_host': 'same'})
            with self.assertRaisesRegex(ValueError, 'UI client host'):
                common.approved_contract(self.root)

    def test_topology_value_must_be_finite_and_typed(self):
        for value in (None, True, 1, 'remote', 'SAME', ['same']):
            with self.subTest(value=value), self.assertRaises(ValueError):
                inputs.profile_contract(PROFILE, 'antimatter-ui', ui_client_host=value)
            save(self.root / 'qualification-contract.json', {**self.contract, 'ui_client_host': value})
            with self.assertRaises(ValueError):
                common.approved_contract(self.root)
        missing = dict(self.contract)
        del missing['ui_client_host']
        save(self.root / 'qualification-contract.json', missing)
        with self.assertRaises(ValueError):
            common.approved_contract(self.root)

    def test_contract_refuses_weaker_or_cross_profile_admission(self):
        changes = ({'wire_stream': False}, {'wire_temperature': 0}, {'ui_closeout_required': False},
                   {'stream_client': 'antimatter'}, {'ui_client_host': 'remote'},
                   {'client_process_timeout_seconds': 356}, {'client_request_timeout_seconds': 241},
                   {'client_stream_requests': 3}, {'turn_evaluation_ceilings': [594, 619]},
                   {'first_prompt_tokens': 468}, {'second_maximum_prompt_tokens': 492},
                   {'broker_delivery': 'unvalidated-forward'}, {'timing_measured_warm_requests': 1})
        for mutation in changes:
            save(self.root / 'qualification-contract.json', {**self.contract, **mutation})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common.approved_contract(self.root)
        for client_kind in ('python-http', 'antimatter'):
            with self.assertRaises(ValueError):
                inputs.profile_contract(PROFILE, client_kind)
        with self.assertRaises(ValueError):
            inputs.profile_contract(PROFILE, 'antimatter-ui', runtime_phase_diagnostics=True)

    def test_only_three_stages_dispatch_and_replay_before_acceptance(self):
        for mode in ('cpu-reference', 'answers'):
            self.assertEqual(child.probe_command(mode, self.root, PROFILE),
                             ['/usr/bin/python3', str(self.root / 'ui-history-gate.py'), '--stage', mode, '--execute'])
            report = {'client_identity': self.client}
            with patch.object(runner, 'approved_contract', return_value=self.contract), \
                 patch.object(runner, 'client_identity', return_value=self.client), \
                 patch.object(ui_history, 'validate_stage', return_value=report) as replay:
                self.assertEqual(runner.probe_evidence(mode, 'retained raw log', self.root), report)
                replay.assert_called_once_with(self.root, mode, 'retained raw log')
            with patch.object(runner, 'approved_contract', return_value=self.contract), \
                 patch.object(runner, 'client_identity', return_value=self.client), \
                 patch.object(ui_history, 'validate_stage', side_effect=ValueError('bad admission')):
                with self.assertRaisesRegex(ValueError, 'bad admission'):
                    runner.probe_evidence(mode, '', self.root)
        with self.assertRaises(ValueError):
            child.probe_command('timing', self.root, PROFILE)
        with patch.object(runner, 'approved_contract', return_value=self.contract), \
             patch.object(runner, 'client_identity', return_value=self.client):
            with self.assertRaises(ValueError):
                runner.probe_evidence('timing', '', self.root)
            for marker in ('HAG_UI_HISTORY_REPORT {}', 'PASS hag-ui-history'):
                with self.assertRaises(ValueError):
                    runner.probe_evidence('self-check', marker, self.root)
        with patch.object(runner, 'approved_contract', return_value=self.contract), \
             patch.object(runner, 'client_identity', return_value=None):
            with self.assertRaises(ValueError):
                runner.probe_evidence('answers', '', self.root)

    def test_client_identity_dispatch_is_separate_from_legacy_provider(self):
        with patch.object(ui_client, 'client_identity', return_value=self.client) as identify:
            self.assertEqual(common.client_identity(self.root), self.client)
            identify.assert_called_once_with(self.root)

    def test_cli_forwards_new_identity_and_refuses_legacy_flags_or_missing_inspector(self):
        arguments = ['prepare', '--repo', '/repo', '--binary', '/binary', '--model', '/model',
                     '--device', '/dev/dri/renderD999', '--evidence', str(self.root / 'new'),
                     '--profile', PROFILE, '--prompt-inspector', '/inspector', '--ui-client-identity', '/ui-identity.json']
        with patch.object(cli_inputs, 'prepare') as freeze, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(arguments), 0)
        self.assertEqual(freeze.call_args.kwargs, {'profile': PROFILE, 'prompt_inspector': Path('/inspector'),
                                                  'ui_client_identity': Path('/ui-identity.json'),
                                                  'ui_client_host': 'separate'})
        with patch.object(cli_inputs, 'prepare') as freeze, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main(arguments + ['--ui-client-host', 'same']), 0)
        self.assertEqual(freeze.call_args.kwargs['ui_client_host'], 'same')
        self.assertIn('connects directly', output.getvalue())
        self.assertNotIn('SSH', output.getvalue())
        cases = [arguments + ['--antimatter-repo', '/antimatter'],
                 arguments + ['--runtime-phase-diagnostics'],
                 [value for value in arguments if value not in ('--prompt-inspector', '/inspector')]]
        for case in cases:
            with self.subTest(arguments=case), patch.object(cli_inputs, 'prepare') as freeze, \
                 redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(case), 1)
                freeze.assert_not_called()

    def test_cli_refuses_same_host_option_for_every_old_profile_before_preparation(self):
        base = ['prepare', '--repo', '/repo', '--binary', '/binary', '--model', '/model',
                '--device', '/dev/dri/renderD999', '--evidence', str(self.root / 'new')]
        for profile in profiles.PROFILE_NAMES:
            if profile in profiles.UI_PROFILES:
                continue
            with self.subTest(profile=profile), patch.object(cli_inputs, 'prepare') as freeze, \
                 redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(base + ['--profile', profile, '--ui-client-host', 'same']), 1)
                freeze.assert_not_called()

    def test_prepare_identity_exclusivity_is_checked_before_checkpoint_access(self):
        cases = ({'profile': PROFILE, 'prompt_inspector': Path('/missing-inspector')},
                 {'profile': 'list-c128', 'ui_client_identity': Path('/missing-identity')},
                 {'profile': PROFILE, 'ui_client_identity': Path('/missing-identity')},
                 {'profile': 'list-c128', 'ui_client_host': 'same'},
                 {'profile': profiles.GENERATED_HISTORY_PROFILE, 'ui_client_host': 'same'},
                 {'profile': PROFILE, 'ui_client_host': 'remote'})
        for options in cases:
            with self.subTest(options=options), patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                inputs.prepare('/repo', '/binary', '/model', '/dev/dri/renderD999', self.root / 'new', **options)
            checkpoint.assert_not_called()
            self.assertFalse((self.root / 'new').exists())

    def test_prepare_ui_client_command_freezes_only_identity(self):
        arguments = ['prepare-ui-client', '--prepared', '/prepared', '--antimatter-repo', '/antimatter',
                     '--electron-dist', '/electron', '--metadata', str(self.root / 'metadata')]
        with patch.object(cli_ui_client, 'freeze_ui_identity') as freeze, \
             patch.object(cli.subprocess, 'run') as execute, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(arguments), 0)
        freeze.assert_called_once_with(Path('/prepared'), Path('/antimatter'), Path('/electron'), self.root / 'metadata')
        execute.assert_not_called()

    def test_cli_timing_rejected_before_launch_for_ui_profile(self):
        with ExitStack() as stack:
            for name in ('gate_identity', 'dependency_identity', 'provenance_identity', 'client_identity'):
                stack.enter_context(patch.object(cli_common, name, return_value={}))
            stack.enter_context(patch.object(cli_common, 'approved_contract', return_value=self.contract))
            execute = stack.enter_context(patch.object(cli.subprocess, 'run'))
            stack.enter_context(redirect_stderr(io.StringIO()))
            self.assertEqual(cli.main(['run', '--evidence', str(self.root), '--stage', 'timing']), 1)
            execute.assert_not_called()

    def test_closeout_inventory_requires_desktop_and_remote_raw_evidence(self):
        with patch.object(closeout, 'approved_contract', return_value=self.contract):
            names = closeout.closed_evidence_names(self.root)
        for mode in profiles.profile_modes(PROFILE):
            self.assertTrue(set(ui_history.evidence_names(mode)) <= set(names))
        self.assertTrue({'client/identity.json', 'ui-report.json', 'ui-process.json', 'ui-retirement.json'} <= set(names))
        self.assertFalse(any(name.startswith('timing') for name in names))
        self.assertNotIn('client/provider.mjs', names)
        self.assertEqual(len(names), len(set(names)))
        with patch.object(closeout, 'closed_evidence_names', return_value=['ui-report.json']):
            with self.assertRaisesRegex(ValueError, 'ui-report.json'):
                closeout.closed_evidence_hashes(self.root)

    def test_remote_success_cannot_qualify_without_imported_ui_evidence(self):
        # All backend checks pass. Keep the real UI validator through its first
        # local-file read to demonstrate that missing UI evidence blocks closeout.
        save(self.root / 'expected-manifest.json', {'source_files': 2})
        save(self.root / 'client/identity.json', self.client)
        ready = {'client_identity_sha256': common.digest(self.root / 'client/identity.json'),
                 'host': '127.0.0.1', 'port': 48321,
                 'run_id': 'fb6ef24c-a693-4e60-ac8a-d1bb7c809a22',
                 'boot_id': '2fc75136-cc37-4a89-bccc-226cc0047ab2',
                 'delivery': 'buffered-sse-after-parity-reset'}
        save(self.root / 'answers-ui-ready.json', ready)
        save(self.root / 'answers-ui-listener.json', {**ready, 'request_count': 2, 'forwarded_count': 2,
             'response_count': 2, 'rejected_count': 0, 'closed': True})
        for mode in profiles.profile_modes(PROFILE):
            save(self.root / (mode + '-result.json'), {'start': '2026-10-13T12:00:00+00:00',
                                                      'end': '2026-10-13T12:00:01+00:00'})
        (self.root / 'run-gate.py').write_text('# Empty imported runner; backend validation is mocked.\n')
        capture = [{'pid': 123, 'timestamp': '2026-10-13T12:01:00+00:00', 'duration_seconds': 1500},
                   {'pid': 123, 'timestamp': '2026-10-13T12:02:00+00:00'}]
        mocks = {'stopped_capture': capture, 'gate_identity': {}, 'dependency_identity': {},
                 'source_identity': 'a' * 64, 'checkpoint_identity': 'b' * 64,
                 'provenance_identity': {}, 'binary_identity': 'c' * 64, 'client_identity': self.client,
                 'validate_gate': None, 'validate_order': None,
                 'cleanup_evidence': {'all_units_inactive': True, 'all_cgroups_gone': True},
                 'closed_evidence_hashes': {},
                 'kernel_window': {'qualified': True, 'kernel_lines': 1, 'driver_matches': [], 'journal_exit_code': 0}}
        with ExitStack() as stack:
            for name, value in mocks.items():
                stack.enter_context(patch.object(closeout, name, return_value=value))
            stack.enter_context(patch.object(ui_client, 'validate_identity', return_value=self.client))
            stack.enter_context(patch.object(ui_client, 'client_identity', return_value=self.client))
            stack.enter_context(patch.object(ui_history, 'broker_records', return_value=[]))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(closeout.main(self.root), 1)
        result = common.read_json(self.root / 'final-evidence-check.json')
        self.assertTrue(all(result['gates'].values()))
        self.assertFalse(result['qualified'])
        self.assertFalse(result['live_ui_qualified'])
        self.assertIsNone(result['ui_evidence'])
        self.assertTrue(any('ui-report.json' in failure for failure in result['failures']))

    def test_remote_cleanup_checks_broker_and_inspectors_in_remote_namespace(self):
        proc, groups = self.root / 'proc', self.root / 'cgroups'
        proc.mkdir(); groups.mkdir()
        boot = '2fc75136-cc37-4a89-bccc-226cc0047ab2'
        boot_file = proc / 'sys/kernel/random/boot_id'
        boot_file.parent.mkdir(parents=True)
        boot_file.write_text(boot)
        properties = {'ActiveState': 'inactive', 'SubState': 'dead', 'MainPID': '0', 'ControlGroup': ''}
        reports = {mode: {'unit': f'hag-answers-{mode}-20261013-120000-{index:06}',
                         'child': {'pid': 100 + index, 'cgroup': str(groups / mode)},
                         'thermal': {'summary': {'pid': 200 + index}}}
                   for index, mode in enumerate(profiles.profile_modes(PROFILE))}
        reports['answers']['client_identity'] = self.client
        for turn in (1, 2):
            save(self.root / f'cpu-reference-turn-{turn}-measurement.json', {'pid': 401 + turn})
        def processes(_root, mode):
            return [{'pid': 301 if mode == 'cpu-reference' else 302, 'exit_code': 0}]
        def cleanup():
            return closeout.cleanup_evidence(reports, proc, groups, lambda _: properties, self.root, PROFILE)
        with patch.object(ui_history, 'server_processes', side_effect=processes), \
             patch.object(ui_history, 'listener', return_value={'boot_id': boot, 'pid': 501}):
            result = cleanup()
            self.assertEqual(len(result['recorded_pids_absent']), 11)
            self.assertEqual(result['stages']['answers']['broker_pid_absent'], 501)
            self.assertEqual(result['stages']['cpu-reference']['inspector_pids_absent'], [402, 403])
            self.assertNotIn('client_pids_absent', result['stages']['answers'])
            for pid in (301, 302, 402, 403, 501):
                (proc / str(pid)).mkdir()
                with self.assertRaises(ValueError):
                    cleanup()
                (proc / str(pid)).rmdir()


if __name__ == '__main__':
    unittest.main()
