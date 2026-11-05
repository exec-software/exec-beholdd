"""Model-free opt-in, environment isolation and non-qualification regressions."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as common
import inputs
import profiles


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = load('phase_worker', 'answer-gate.py')
child = load('phase_child', 'gate-child.py')
runner = load('phase_runner', 'run-gate.py')
PROFILE = profiles.TEMPLATE_HISTORY_PROFILE


class PhaseDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = inputs.profile_contract(PROFILE, 'antimatter', runtime_phase_diagnostics=True)
        self.policy = self.contract['runtime_phase_diagnostics']

    def test_absent_and_explicit_off_preserve_all_existing_contracts(self):
        for profile in profiles.PROFILE_NAMES:
            client = ('antimatter-ui' if profile in profiles.UI_PROFILES else
                      'antimatter' if profile in (*profiles.GROUNDED_PROFILES, profiles.GENERATED_HISTORY_PROFILE)
                      else 'python-http')
            original = inputs.profile_contract(profile, client)
            self.assertNotIn('runtime_phase_diagnostics', original)
            self.assertEqual(original, inputs.profile_contract(profile, client, runtime_phase_diagnostics=False))
            for mode in profiles.profile_modes(profile):
                self.assertIsNone(common.phase_diagnostics(original, mode))
                common.require_phase_diagnostics({}, original, mode)
                with self.assertRaises(ValueError):
                    common.require_phase_diagnostics({'runtime_phase_diagnostics': None}, original, mode)

    def test_strict_diagnostic_schema_refuses_forged_enablement_and_scope(self):
        values = [None, False, True, {}, [], '1']
        for key, value in [('schema', True), ('schema', 2), ('environment', {'HAG_SERVING_PHASES': 1}),
                           ('environment', {'HAG_SERVING_PHASES': '1', 'HAG_TRACE_SUBMISSION': '1'}),
                           ('stages', ['answers', 'timing']), ('stages', ('answers',)),
                           ('log_prefix', 'HAG_SERVING_AUDIT '), ('qualifying_evidence', 0),
                           ('qualifying_evidence', True), ('unknown', True)]:
            values.append({**self.policy, key: value})
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                common.phase_diagnostics({**self.contract, 'runtime_phase_diagnostics': value})
        for profile in profiles.PROFILE_NAMES:
            if profile != PROFILE:
                with self.subTest(profile=profile), self.assertRaises(ValueError):
                    common.phase_diagnostics({**self.contract, 'profile': profile})

    def test_resolved_receipts_are_strictly_typed_stage_specific_and_not_evidence(self):
        for mode in profiles.profile_modes(PROFILE):
            policy = common.phase_diagnostics(self.contract, mode)
            self.assertIs(policy['enabled'], mode == 'answers')
            self.assertIs(policy['qualifying_evidence'], False)
            common.require_phase_diagnostics({'runtime_phase_diagnostics': policy}, self.contract, mode)
            for key, value in [('schema', True), ('enabled', int(mode == 'answers')),
                               ('qualifying_evidence', 0), ('environment', {} if mode == 'answers' else {'HAG_SERVING_PHASES': '1'})]:
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    common.require_phase_diagnostics({'runtime_phase_diagnostics': {**policy, key: value}}, self.contract, mode)
            with self.assertRaises(ValueError): common.require_phase_diagnostics({}, self.contract, mode)
        with self.assertRaises(ValueError): common.phase_diagnostics(self.contract, 'custom')

    def test_child_never_propagates_ambient_phase_opt_in(self):
        original = {'HAG_SERVING_PHASES': '1', 'KEEP': 'value', 'BEHOLDD_HAG_PRIVATE_CGROUP': '/forged'}
        path = '/user.slice/hag-answers-answers-20261011-000000-000001.service'
        for profile in profiles.PROFILE_NAMES:
            for mode in profiles.profile_modes(profile):
                environment = child.workload_environment(mode, profile, path, original)
                self.assertNotIn('HAG_SERVING_PHASES', environment)
                self.assertEqual(environment['KEEP'], 'value')
        self.assertEqual(original['HAG_SERVING_PHASES'], '1')

    def test_only_frozen_native_answers_gets_opt_in_and_inputs_are_not_mutated(self):
        inherited = {'HAG_SERVING_PHASES': 'wrong', 'KEEP': 'value'}
        saved = deepcopy(self.policy)
        for backend in ('cpu', 'hag'):
            for stage in profiles.profile_modes(PROFILE):
                with self.subTest(backend=backend, stage=stage):
                    default = worker.server_phase_environment(inherited, backend, stage, PROFILE)
                    self.assertNotIn('HAG_SERVING_PHASES', default)
                    actual = worker.server_phase_environment(inherited, backend, stage, PROFILE, self.policy)
                    self.assertEqual(actual.get('HAG_SERVING_PHASES'), '1' if (backend, stage) == ('hag', 'answers') else None)
                    self.assertEqual(actual['KEEP'], 'value')
        self.assertEqual(self.policy, saved)
        self.assertEqual(inherited['HAG_SERVING_PHASES'], 'wrong')

    def test_server_rejects_malformed_policy_before_socket_log_or_process(self):
        server = worker.Server(self.root, 'hag', 'answers', 2048, PROFILE,
                               phase_diagnostics={**self.policy, 'qualifying_evidence': True})
        with patch.object(worker.socket, 'socket') as socket, patch.object(worker.subprocess, 'Popen') as process:
            with self.assertRaises(ValueError): server.__enter__()
        socket.assert_not_called()
        process.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])

    def test_actual_server_launch_gets_only_resolved_native_policy(self):
        attestation = '/user.slice/hag-answers-answers-20261011-000000-000001.service'
        for backend, stage in [('cpu', 'cpu-reference'), ('hag', 'answers'), ('hag', 'timing')]:
            folder = self.root / stage
            folder.mkdir()
            server = worker.Server(folder, backend, stage, 2048, PROFILE, phase_diagnostics=self.policy)
            process = Mock()
            process.poll.return_value = None
            def launched(*args, **kwargs):
                kwargs['stdout'].write(f'beholdd v2 listening on http://127.0.0.1:{server.port}\n')
                kwargs['stdout'].flush()
                return process
            with patch.dict(worker.os.environ, {'HAG_SERVING_PHASES': '1', 'BEHOLDD_HAG_PRIVATE_CGROUP': attestation}), \
                 patch.object(worker, 'verify_containment', return_value=attestation), \
                 patch.object(worker.socket, 'socket') as socket, \
                 patch.object(worker.subprocess, 'Popen', side_effect=launched) as launch, \
                 patch.object(server, 'json', return_value=(200, {'status': 'ok'})):
                socket.return_value.__enter__.return_value.getsockname.return_value = ('127.0.0.1', 12345)
                server.__enter__()
            self.assertEqual(launch.call_args.kwargs['env'].get('HAG_SERVING_PHASES'),
                             '1' if stage == 'answers' else None)
            self.assertEqual(launch.call_args.kwargs['env']['HAG_TRACE_SUBMISSION'], '0')
            server.log.close()

    def test_both_node_launch_paths_strip_phase_env_and_preserve_failed_process_record(self):
        for grounded in (False, True):
            root = self.root / str(grounded)
            root.mkdir()
            server = Mock(root=root, port=12345)
            server.json.return_value = (200, {'data': [{'id': worker.MODEL_ID, 'loaded': False,
                                                       'execution_limits': {'max_evaluated_tokens': 2048}}]})
            process = Mock(pid=12345, returncode=1)
            process.poll.return_value = 1
            profile = PROFILE if grounded else 'backend-parity-c128'
            fixture = profiles.fixture_document(profile)['fixtures'][0]
            with patch.dict(worker.os.environ, {'HAG_SERVING_PHASES': '1'}), \
                 patch.object(worker, 'client_identity', return_value={'kind': 'antimatter'}), \
                 patch.object(worker.subprocess, 'Popen', return_value=process) as launch:
                with self.assertRaisesRegex(ValueError, 'client process failed'):
                    if grounded:
                        worker.grounded_client_record(server, fixture, root/'response.json', 'answers-client', False, profile)
                    else:
                        worker.client_stream_complete(server, fixture, {}, 1, root/'response.json', profile)
            launch.assert_called_once()
            self.assertNotIn('HAG_SERVING_PHASES', launch.call_args.kwargs['env'])
            record = next(root.glob('*-process.json'))
            self.assertEqual(json.loads(record.read_text())['exit_code'], 1)

    def test_phase_only_logs_never_substitute_for_a_qualified_answer_report(self):
        raw = 'HAG_SERVING_PHASE {"phase":"request_complete"}\n'
        with patch.object(runner, 'approved_contract', return_value=self.contract), \
             patch.object(runner, 'client_identity', return_value=None):
            with self.assertRaisesRegex(ValueError, 'need exactly one answer report'):
                runner.probe_evidence('answers', raw, self.root)
        # Unparsed phase lines neither satisfy nor change lifecycle audit checks.
        path = self.root/'server.log'
        event = {'request_sequence': 1, 'evaluated_positions': 497, 'outcome': 'success', 'reset_ok': True, 'poisoned': False}
        path.write_text('HAG_SERVING_PHASE incomplete diagnostic line\nHAG_SERVING_AUDIT '+json.dumps(event)+'\n')
        self.assertEqual(worker.audit_events(path, 2048), [event])

    def test_opt_in_cpu_reference_remains_compatible_with_independent_parity_reader(self):
        document = profiles.fixture_document(PROFILE)
        worker.write_json(self.root/'answer-fixtures.json', document)
        fixture = document['fixtures'][0]
        manifest = {'profile': PROFILE, 'binaries': {'beholdd': 'b'*64}, 'source_manifest_sha256': 's'*64,
                    'model_sha256': 'm'*64, 'gate_files': {'answer-fixtures.json': worker.digest(self.root/'answer-fixtures.json')}}
        worker.write_json(self.root/'expected-manifest.json', manifest)
        identities = {'binary_sha256': 'b'*64, 'source_manifest_sha256': 's'*64, 'checkpoint_sha256': 'm'*64,
                      'fixture_sha256': manifest['gate_files']['answer-fixtures.json'], 'gate_files': manifest['gate_files']}
        answer = worker.generation(200, {'object': 'chat.completion', 'model': worker.MODEL_ID,
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': '<think>\n\n</think>\n\nready.'}}],
            'usage': {'prompt_tokens': 491, 'completion_tokens': 6, 'total_tokens': 497}}, fixture, PROFILE)
        with patch.dict(worker.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(worker, 'verify_containment'), \
             patch.object(worker, 'identity', return_value={**identities, 'runtime_phase_diagnostics': self.policy}), \
             patch.object(worker, 'Server') as server, \
             patch.object(worker, 'complete', return_value=(answer, 1)), \
             patch.object(worker, 'audit_events', return_value=[]), patch('builtins.print'):
            worker.execute('cpu-reference', self.root)
        server.assert_called_once_with(self.root, 'cpu', 'cpu-reference', 2048, PROFILE, phase_diagnostics=self.policy)
        self.assertEqual(worker.read_json(self.root/'cpu-reference.json')['identities'], identities)
        self.assertNotIn('runtime_phase_diagnostics', worker.read_json(self.root/'cpu-reference-report.json'))
        with patch.object(runner, 'client_identity', return_value=None):
            self.assertEqual(runner.reference_payload(self.root, [fixture], PROFILE), [answer])


if __name__ == '__main__':
    unittest.main()
