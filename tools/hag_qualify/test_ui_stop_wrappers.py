"""Model-free registration and supervisor dispatch for the finite UI Stop/reuse profile."""
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
import ui_stop
import ui_stop_client
from hag_qualify import __main__ as cli
from hag_qualify import gate_common as cli_common, inputs as cli_inputs

PROFILE = profiles.UI_STOP_PROFILE


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('ui_stop_wrappers_runner', 'run-gate.py')
child = load('ui_stop_wrappers_child', 'gate-child.py')
closeout = load('ui_stop_wrappers_closeout', 'closeout.py')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


class UiStopWrappers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = inputs.profile_contract(PROFILE, 'antimatter-ui')
        self.client = {'profile': PROFILE, 'kind': 'antimatter-ui', 'commit': 'fixture',
                       'base_url': 'http://127.0.0.1:48321/v1',
                       'local_host': {'boot_id': '262efb57-e5c3-4393-9144-6bf66a558a48'}}
        save(self.root / 'answer-fixtures.json', profiles.fixture_document(PROFILE))
        save(self.root / 'qualification-contract.json', self.contract)

    def test_finite_prompts_and_accounting_do_not_inherit_generated_history(self):
        document = profiles.fixture_document(PROFILE)
        self.assertEqual(set(document), {'profile', 'first_fixture', 'reuse_fixture'})
        self.assertEqual(profiles.fixture_profile(document), PROFILE)
        first, reuse = document['first_fixture'], document['reuse_fixture']
        project = profiles.fixture_document(profiles.PROJECT_PROFILE)['fixtures'][0]
        self.assertEqual(first['messages'][0], project['messages'][0])
        self.assertEqual(reuse['messages'], project['messages'])
        self.assertEqual(first['messages'][-1]['content'],
                         'Count upwards from 1 to 100, separating numbers with spaces. /no_think')
        self.assertEqual((first['id'], reuse['id']), ('ui_stop_count', 'ui_stop_reuse'))
        self.assertEqual((first['max_tokens'], reuse['max_tokens']), (64, 16))
        self.assertNotIn('temperature', first)
        self.assertNotIn('temperature', reuse)
        limits = profiles.profile_limits(PROFILE)
        self.assertEqual(limits['turn_prompt_tokens'], [476, 467])
        self.assertEqual(limits['turn_evaluation_ceilings'], [539, 482])
        self.assertEqual(self.contract['stage_request_counts'], {'cpu-reference': 1, 'answers': 2})
        self.assertEqual(self.contract['stage_evaluation_ceilings'], {'cpu-reference': 482, 'answers': 1021})
        self.assertEqual(self.contract['cpu_parity_fixture_ids'], ['ui_stop_reuse'])
        self.assertIs(self.contract['generated_history_required'], False)
        self.assertIs(self.contract['require_eos_stop'], False)
        self.assertIs(self.contract['reuse_require_eos_stop'], True)
        for mutation in ('max_tokens', 'messages'):
            changed = deepcopy(document)
            changed['first_fixture'][mutation] = True
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                profiles.fixture_profile(changed)
        self.assertEqual(common.approved_contract(self.root), self.contract)

    def test_existing_history_profile_keeps_its_exact_workload_and_scope(self):
        old = inputs.profile_contract(profiles.UI_HISTORY_PROFILE, 'antimatter-ui')
        expected = {'qualification_kind': 'ui_history_and_backend_parity',
                    'stage_request_counts': {'cpu-reference': 2, 'answers': 2},
                    'stage_evaluation_ceilings': {'cpu-reference': 1212, 'answers': 1212},
                    'turn_evaluation_ceilings': [594, 618], 'max_new_tokens': 128,
                    'broker_delivery': 'buffered-sse-after-parity-reset',
                    'generated_history_required': True, 'live_cancellation_qualified': False,
                    'history_source': 'same-backend-raw-first-answer'}
        self.assertEqual({key: old[key] for key in expected}, expected)
        for key in ('reuse_prompt_tokens', 'turn_max_new_tokens', 'cpu_parity_fixture_ids',
                    'cancel_after_visible_numeric_content', 'fresh_conversation_reuse_required'):
            self.assertNotIn(key, old)
        document = profiles.fixture_document(profiles.UI_HISTORY_PROFILE)
        self.assertEqual(set(document), {'profile', 'first_fixture', 'followup_user', 'second_fixture_id'})
        self.assertEqual(document['first_fixture']['id'], 'ui_turn_1')
        self.assertEqual(document['second_fixture_id'], 'ui_turn_2')
        self.assertEqual(ui_client.UI_COMMIT, '8a0f8dce59fe7e5d71777fd4f709f5d70f67c15e')

    def test_new_profile_keeps_independent_cpu_ui_memory_and_thermal_limits(self):
        old = inputs.profile_contract(profiles.UI_HISTORY_PROFILE, 'antimatter-ui')
        for key in ('capacity', 'hag_charge_bytes', 'hag_budget_bytes', 'cpu_model_budget_bytes',
                    'runtime_combined_forecast_bytes', 'process_memory_max_bytes', 'process_early_stop_bytes',
                    'swap_max_bytes', 'cpu_quota_percent', 'tasks_max', 'no_new_privileges',
                    'host_start_headroom_bytes', 'host_stop_headroom_bytes', 'thermal_cutoff_c',
                    'thermal_poll_seconds', 'thermal_start_max_c', 'thermal_start_stable_seconds',
                    'private_cgroup_admission', 'crash_watcher_seconds', 'stage_deadlines',
                    'client_request_timeout_seconds', 'client_process_timeout_seconds'):
            self.assertEqual(self.contract[key], old[key], key)
        self.assertEqual(profiles.profile_deadlines(PROFILE)['answers'], (360, 375, 390))
        self.assertEqual(profiles.profile_modes(PROFILE), ('self-check', 'cpu-reference', 'answers'))
        self.assertFalse(any(key.startswith('timing_') for key in self.contract))
        self.assertNotIn('runtime_phase_diagnostics', self.contract)

    def test_weakened_bounds_or_cross_profile_contracts_are_rejected(self):
        changes = ({'first_prompt_tokens': 477}, {'reuse_prompt_tokens': 468},
                   {'turn_max_new_tokens': [64, 17]}, {'turn_max_new_tokens': [True, 16]},
                   {'turn_evaluation_ceilings': [539, 483]},
                   {'stage_request_counts': {'cpu-reference': 2, 'answers': 2}},
                   {'stage_evaluation_ceilings': {'cpu-reference': 482, 'answers': 1022}},
                   {'cpu_parity_fixture_ids': ['ui_stop_count', 'ui_stop_reuse']},
                   {'generated_history_required': True}, {'same_server_reuse_required': False},
                   {'fresh_conversation_reuse_required': False}, {'cancel_after_visible_numeric_content': False},
                   {'live_cancellation_qualified': False}, {'reuse_require_eos_stop': False},
                   {'wire_stream': False}, {'wire_temperature': 0}, {'ui_closeout_required': False},
                   {'broker_delivery': 'buffered-sse-after-parity-reset'}, {'stream_client': 'antimatter'},
                   {'history_source': 'same-backend-raw-first-answer'}, {'timing_measured_warm_requests': 1},
                   {'client_process_timeout_seconds': 356}, {'client_request_timeout_seconds': 241})
        for mutation in changes:
            save(self.root / 'qualification-contract.json', {**self.contract, **mutation})
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                common.approved_contract(self.root)
        for client in ('python-http', 'antimatter'):
            with self.assertRaises(ValueError):
                inputs.profile_contract(PROFILE, client)
        with self.assertRaises(ValueError):
            inputs.profile_contract(PROFILE, 'antimatter-ui', runtime_phase_diagnostics=True)

    def test_same_host_is_explicit_and_only_changes_topology_description(self):
        same = inputs.profile_contract(PROFILE, 'antimatter-ui', ui_client_host='same')
        self.assertEqual({k: v for k, v in same.items() if k not in ('ui_client_host', 'scope', 'limitations')},
                         {k: v for k, v in self.contract.items() if k not in ('ui_client_host', 'scope', 'limitations')})
        self.assertIn('same host', same['scope'])
        self.assertIn('4 GiB', same['limitations'])
        save(self.root / 'qualification-contract.json', same)
        self.assertEqual(common.approved_contract(self.root), same)
        for value in (None, True, 'remote'):
            with self.assertRaises(ValueError):
                inputs.profile_contract(PROFILE, 'antimatter-ui', ui_client_host=value)

    def test_three_stage_dispatch_replays_only_stop_evidence(self):
        for mode in ('cpu-reference', 'answers'):
            self.assertEqual(child.probe_command(mode, self.root, PROFILE),
                             ['/usr/bin/python3', str(self.root / 'ui-stop-gate.py'), '--stage', mode, '--execute'])
            report = {'client_identity': self.client}
            with patch.object(runner, 'approved_contract', return_value=self.contract), \
                 patch.object(runner, 'client_identity', return_value=self.client), \
                 patch.object(ui_stop, 'validate_stage', return_value=report) as replay:
                self.assertEqual(runner.probe_evidence(mode, 'raw retained evidence', self.root), report)
                replay.assert_called_once_with(self.root, mode, 'raw retained evidence')
        with self.assertRaises(ValueError):
            child.probe_command('timing', self.root, PROFILE)
        with patch.object(runner, 'approved_contract', return_value=self.contract), \
             patch.object(runner, 'client_identity', return_value=self.client):
            with self.assertRaises(ValueError):
                runner.probe_evidence('timing', '', self.root)
            for marker in ('HAG_UI_STOP_REPORT {}', 'PASS hag-ui-stop'):
                with self.assertRaises(ValueError):
                    runner.probe_evidence('self-check', marker, self.root)
        with patch.object(ui_client, 'client_identity', return_value=self.client) as identify:
            self.assertEqual(common.client_identity(self.root), self.client)
            identify.assert_called_once_with(self.root)

    def test_prepare_checks_identity_profile_before_checkpoint_or_output_access(self):
        with patch.object(common, 'prompt_inspector_identity', return_value='a' * 64), \
             patch.object(common, 'read_json', return_value={}), \
             patch.object(ui_client, 'validate_identity', return_value={'profile': profiles.UI_HISTORY_PROFILE}), \
             patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaisesRegex(ValueError, 'another qualification profile'):
            inputs.prepare('/repo', '/binary', '/model', '/dev/dri/renderD999', self.root / 'new',
                           profile=PROFILE, prompt_inspector='/inspector', ui_client_identity='/identity')
        checkpoint.assert_not_called()
        self.assertFalse((self.root / 'new').exists())

    def test_cli_forwards_explicit_new_profile_and_rejects_timing(self):
        args = ['prepare', '--repo', '/repo', '--binary', '/binary', '--model', '/model',
                '--device', '/dev/dri/renderD999', '--evidence', str(self.root / 'new'),
                '--profile', PROFILE, '--prompt-inspector', '/inspector', '--ui-client-identity', '/identity']
        with patch.object(cli_inputs, 'prepare') as freeze, redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(args + ['--ui-client-host', 'same']), 0)
        self.assertEqual(freeze.call_args.kwargs, {'profile': PROFILE, 'prompt_inspector': Path('/inspector'),
                                                  'ui_client_identity': Path('/identity'), 'ui_client_host': 'same'})
        for extra in (['--runtime-phase-diagnostics'], ['--antimatter-repo', '/antimatter']):
            with patch.object(cli_inputs, 'prepare') as freeze, redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(args + extra), 1)
            freeze.assert_not_called()
        with ExitStack() as stack:
            for name in ('gate_identity', 'dependency_identity', 'provenance_identity', 'client_identity'):
                stack.enter_context(patch.object(cli_common, name, return_value={}))
            stack.enter_context(patch.object(cli_common, 'approved_contract', return_value=self.contract))
            execute = stack.enter_context(patch.object(cli.subprocess, 'run'))
            stack.enter_context(redirect_stderr(io.StringIO()))
            self.assertEqual(cli.main(['run', '--evidence', str(self.root), '--stage', 'timing']), 1)
            execute.assert_not_called()

    def test_closeout_requires_both_raw_cancel_and_host_qualified_ui_evidence(self):
        with patch.object(closeout, 'approved_contract', return_value=self.contract):
            names = closeout.closed_evidence_names(self.root)
        for mode in profiles.profile_modes(PROFILE):
            self.assertTrue(set(ui_stop.evidence_names(mode)) <= set(names))
        self.assertTrue(set(ui_client.evidence_names()) <= set(names))
        self.assertEqual(len(names), len(set(names)))
        self.assertFalse(any(name.startswith('timing') for name in names))
        for qualified in (False, True):
            scope = closeout.qualification_scope(self.contract, qualified)
            for key in ('live_ui_qualified', 'live_cancellation_qualified', 'same_server_reuse_qualified'):
                self.assertIs(scope[key], qualified)
            for key in ('generated_history_qualified', 'cancellation_answer_parity_qualified',
                        'semantic_quality_qualified', 'performance_qualified', 'timing_executed'):
                self.assertIs(scope[key], False)

    def test_backend_only_success_does_not_qualify_stop_or_reuse_without_ui(self):
        save(self.root / 'expected-manifest.json', {'source_files': 2})
        for mode in profiles.profile_modes(PROFILE):
            save(self.root / (mode + '-result.json'), {'start': '2026-10-13T12:00:00+00:00',
                                                      'end': '2026-10-13T12:00:01+00:00'})
        (self.root / 'run-gate.py').write_text('# Imported runner is mocked for this model-free closeout test.\n')
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
            stack.enter_context(patch.object(ui_stop, 'broker_records', return_value=[]))
            verify = stack.enter_context(patch.object(ui_stop_client, 'validate_ui_evidence',
                                                      side_effect=ValueError('missing host-qualified UI evidence')))
            stack.enter_context(redirect_stdout(io.StringIO()))
            self.assertEqual(closeout.main(self.root), 1)
            verify.assert_called_once_with(self.root, self.client, [])
        result = common.read_json(self.root / 'final-evidence-check.json')
        self.assertTrue(all(result['gates'].values()))
        self.assertFalse(result['qualified'])
        self.assertFalse(result['live_cancellation_qualified'])
        self.assertFalse(result['same_server_reuse_qualified'])
        self.assertIsNone(result['ui_evidence'])


if __name__ == '__main__':
    unittest.main()
