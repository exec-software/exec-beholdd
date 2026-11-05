"""Model-free lifecycle replay and transport tests; no sockets or CLI are opened."""
import base64
from copy import deepcopy
import http.client
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lifecycle as proof
from profiles import LIFECYCLE_PROFILE

SPEC = importlib.util.spec_from_file_location('lifecycle_gate_tested', Path(__file__).with_name('lifecycle-gate.py'))
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

TEXT = ' Answer'
REFERENCE = {'text': TEXT, 'finish_reason': 'length',
             'usage': {'prompt_tokens': 9, 'completion_tokens': 1, 'total_tokens': 10}}


def completion(text=TEXT):
    return {'object': 'chat.completion', 'model': proof.MODEL_ID,
            'choices': [{'index': 0, 'finish_reason': 'length',
                         'message': {'role': 'assistant', 'content': text}}],
            'usage': deepcopy(REFERENCE['usage'])}


def chunk(delta=None, finish=None, identifier='response-one'):
    return ('data: ' + json.dumps({'id': identifier, 'object': 'chat.completion.chunk',
            'model': proof.MODEL_ID, 'choices': [{'index': 0, 'delta': delta or {},
                                                'finish_reason': finish}]}) + '\n\n').encode()


def stream_bytes(cancel=False):
    prefix = chunk({'role': 'assistant'}) + chunk({'content': TEXT})
    return prefix if cancel else prefix + chunk(finish='length') + b'data: [DONE]\n\n'


def events(cancel_count=10):
    return [{'request_sequence': i + 1, 'evaluated_positions': cancel_count if i == 2 else 9,
             'outcome': 'cancelled' if i == 2 else 'success', 'reset_ok': True, 'poisoned': False}
            for i in range(5)]


def event_bytes(values):
    return b''.join(proof.AUDIT_PREFIX + json.dumps(event).encode() + b'\n' for event in values)


def memory():
    return {'memory_current_bytes': 4096, 'memory_peak_bytes': 8192,
            'swap_current_bytes': 0, 'swap_peak_bytes': 0,
            'memory_events': {'low': 0, 'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0, 'oom_group_kill': 0}}


def record(mode):
    cgroup = ('/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service/app.slice/'
              f'hag-answers-{mode}-20261008-120000-000000.service')
    initial = {'pid': 41002 if mode == 'answers' else 41001, 'start_time_ticks': 12000,
               'boot_id': '12345678-1234-1234-1234-123456789abc', 'cgroup': cgroup,
               'limits': deepcopy(proof.EXPECTED_LIMITS), 'memory': memory(), 'no_new_privileges': True,
               'private_cgroup_attestation': cgroup.removeprefix('/sys/fs/cgroup') if mode == 'answers' else None}
    return {'mode': mode, 'server_pid': initial['pid'], 'server_exit_code': 0,
            'snapshots': [dict(deepcopy(initial), label=label) for label in proof.SNAPSHOTS[mode]],
            'socket_shutdown': {'how': 'SHUT_RDWR', 'succeeded': True} if mode == 'answers' else None}


class ReplayFixture:
    """Tiny complete raw artifact tree; only the outer frozen contract is mocked."""
    def __init__(self, root):
        self.root = root
        self.json('expected-manifest.json', {'binaries': {'beholdd': 'a' * 64},
                  'source_manifest_sha256': 'b' * 64, 'model_sha256': 'c' * 64,
                  'gate_files': {'answer-fixtures.json': 'd' * 64}})
        self.stage('cpu-reference')
        self.json('cpu-reference.json', {'schema': 1, 'profile': LIFECYCLE_PROFILE,
                  'identities': proof.identity(root), 'answer': deepcopy(REFERENCE),
                  'cpu_exit_code': 0, 'final_unloaded': True})
        self.cpu = proof.summarize_stage(root, 'cpu-reference')
        self.json('cpu-reference-report.json', self.cpu)
        self.json('cpu-reference-result.json', {'qualified': True, 'failures': [], 'exit_code': 0, 'probe': self.cpu})
        self.stage('answers')
        self.http('answers-reuse', 'POST', '/v1/chat/completions', proof.request_body(), completion())
        self.http('answers-reload', 'POST', '/v1/chat/completions', proof.request_body(), completion())
        self.http('answers-busy-unload', 'POST', f'/v1/models/{proof.MODEL_ID}/unload', {},
                  {'error': {'code': 'runtime_busy'}}, status=409)
        for name, cancel in (('stream', False), ('cancel', True)):
            self.json('answers-' + name + '-headers.json', {'request': proof.request_body(True, cancel),
                      'status': 200, 'content_type': 'text/event-stream'})
            (root / ('answers-' + name + '.sse')).write_bytes(stream_bytes(cancel))
        (root / 'answers-hag-server.log').write_bytes(event_bytes(events()))

    def json(self, name, value):
        (self.root / name).write_text(json.dumps(value))

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def http(self, stem, method, route, request, response, status=200):
        self.json(stem + '-http.json', {'method': method, 'route': route, 'request': request,
                  'status': status, 'content_type': 'application/json',
                  'body_base64': base64.b64encode(json.dumps(response).encode()).decode(),
                  'truncated': False, 'read_error': None})

    def stage(self, mode):
        native = mode == 'answers'
        self.json('lifecycle-' + mode + '-record.json', record(mode))
        first = record(mode)['snapshots'][0]
        self.json(mode + '-child-result.json', {key: first[key] for key in ('cgroup', 'private_cgroup_attestation')})
        self.http(mode + '-identity', 'GET', '/v1/beholdd/info', None,
                  {'runtime': 'native', 'service': 'beholdd', 'generation': 2,
                   'execution_backends': ['cpu', 'hag'] if native else ['cpu']})
        for name, loaded in (('cold', False), ('unloaded', False),
                             *([('loaded', True), ('reloaded', True), ('final-unloaded', False)] if native else [])):
            self.http(mode + '-' + name + '-model', 'GET', '/v1/models', None,
                      {'data': [{'id': proof.MODEL_ID, 'loaded': loaded,
                                 'execution_limits': {'max_evaluated_tokens': 2048 if native else None}}]})
        self.http(mode + '-hello', 'POST', '/v1/chat/completions', proof.request_body(), completion())
        for stem in ('unload', 'final-unload') if native else ('unload',):
            for index in (1, 2):
                self.http(f'{mode}-{stem}-{index}', 'POST', f'/v1/models/{proof.MODEL_ID}/unload', {},
                          {'id': proof.MODEL_ID, 'loaded': False})
        (self.root / (mode + ('-hag' if native else '-cpu') + '-server.log')).write_bytes(b'CLI ready\n')


class LifecycleParserTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_length_response_keeps_raw_text_without_complete_answer_semantics(self):
        self.assertEqual(proof.generation(200, completion('<think>'))['text'], '<think>')
        self.assertEqual(proof.generation(200, completion()), REFERENCE)

    def test_completion_status_identity_finish_and_typed_counts_are_required(self):
        bad = []
        for key, value in [('prompt_tokens', True), ('completion_tokens', True),
                           ('total_tokens', 2**64), ('prompt_tokens', -1)]:
            item = completion()
            item['usage'][key] = value
            bad.append(item)
        for value in ('stop', None):
            item = completion()
            item['choices'][0]['finish_reason'] = value
            bad.append(item)
        item = completion()
        item['choices'][0]['index'] = False
        bad.append(item)
        item = completion()
        item['model'] = 'other'
        bad.append(item)
        for value in bad:
            with self.subTest(value=value), self.assertRaises(ValueError):
                proof.generation(200, value)
        for status in (True, 400, 500):
            with self.subTest(status=status), self.assertRaises(ValueError):
                proof.generation(status, completion())

    def parse(self, data, cancel=False):
        state = proof.SseResult()
        for line in data.splitlines(keepends=True):
            state.line(line)
        state.finish(REFERENCE, cancel)
        return state

    def test_sse_complete_and_first_content_cancel_are_distinct(self):
        self.assertEqual(self.parse(stream_bytes()).content, TEXT)
        self.assertEqual(self.parse(stream_bytes(True), True).done, 0)
        with self.assertRaises(ValueError):
            self.parse(stream_bytes(), True)
        with self.assertRaises(ValueError):
            self.parse(stream_bytes(True))

    def test_sse_rejects_bad_done_truncation_mixed_id_and_raw_mismatch(self):
        role = chunk({'role': 'assistant'})
        content = chunk({'content': TEXT})
        failures = [b'data: [DONE]\n', role + content + b'data: [DONE]\n',
                    stream_bytes() + b'data: [DONE]\n', stream_bytes().rstrip(b'\n'),
                    role + chunk({'content': TEXT}, identifier='another'),
                    role + chunk({'content': 'different'}) + chunk(finish='length') + b'data: [DONE]\n',
                    role + chunk({'tool_calls': [{}]})]
        for data in failures:
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.parse(data)

    def test_sse_rejects_comments_after_done_and_embedded_line_endings(self):
        with self.assertRaises(ValueError):
            self.parse(stream_bytes() + b': extra after DONE\n')
        for raw in (chunk({'role': 'assistant'}), b': one\rtwo\n', b'data: [DONE]\n\n'):
            # A parser input is exactly one received line. Embedded delimiters
            # must not be erased by rstrip or hidden in comments.
            state = proof.SseResult()
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                state.line(raw)

    def test_sse_rejects_unknown_delta_keys_and_role_after_finish(self):
        role = chunk({'role': 'assistant'})
        for delta in ({'tool_calls': []}, {'reasoning': ''}, {'unknown': None},
                      {'content': TEXT, 'tool_calls': []}):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                self.parse(role + chunk(delta) + chunk(finish='length') + b'data: [DONE]\n')
        with self.assertRaises(ValueError):
            self.parse(role + chunk({'content': TEXT}) + chunk(finish='length')
                       + role + b'data: [DONE]\n')

    def test_audit_polling_skips_partial_line_but_final_rejects_it(self):
        path = self.root / 'server.log'
        raw = event_bytes(events()[:2])
        path.write_bytes(raw + event_bytes(events()[2:3])[:-1])
        self.assertEqual(proof.audit_events(path, final=False), events()[:2])
        with self.assertRaisesRegex(ValueError, 'incomplete final'):
            proof.audit_events(path)

    def test_audit_requires_typed_bounded_healthy_ordered_events(self):
        path = self.root / 'server.log'
        for key, value in [('request_sequence', True), ('evaluated_positions', True),
                           ('evaluated_positions', -1), ('evaluated_positions', 25),
                           ('evaluated_positions', 2**64), ('reset_ok', False),
                           ('reset_ok', 1), ('poisoned', True), ('outcome', 'failed')]:
            bad = events()
            bad[0][key] = value
            path.write_bytes(event_bytes(bad))
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                proof.audit_events(path)

    def test_missing_extra_and_naturally_completed_cancellation_fail(self):
        for bad in (events()[:-1], events() + [dict(events()[-1], request_sequence=6)],
                    [dict(e, outcome='success') for e in events()]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                proof.validate_events(bad)
        self.assertEqual(proof.validate_events(events(9)), 45)
        self.assertEqual(proof.validate_events(events(24)), 60)

    def test_snapshot_rejects_pid_start_boot_cgroup_drift_and_missing_shutdown(self):
        for key, value in [('pid', 999), ('start_time_ticks', 1),
                           ('boot_id', '87654321-1234-1234-1234-123456789abc'),
                           ('cgroup', '/sys/fs/cgroup/other')]:
            bad = record('answers')
            bad['snapshots'][-1][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                proof.validate_record(bad, 'answers')
        for shutdown in (None, {'how': 'SHUT_WR', 'succeeded': True},
                         {'how': 'SHUT_RDWR', 'succeeded': False}):
            bad = record('answers')
            bad['socket_shutdown'] = shutdown
            with self.subTest(shutdown=shutdown), self.assertRaises(ValueError):
                proof.validate_record(bad, 'answers')

    def test_snapshot_typed_memory_bounds_pressure_and_worker_membership(self):
        for key, value in [('memory_current_bytes', True), ('memory_current_bytes', -1),
                           ('memory_current_bytes', 8193), ('memory_peak_bytes', True),
                           ('memory_peak_bytes', 1536 * 1024**2), ('swap_current_bytes', True),
                           ('swap_current_bytes', 1), ('swap_peak_bytes', 1)]:
            bad = record('answers')
            bad['snapshots'][0]['memory'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                proof.validate_record(bad, 'answers')
        for event in ('high', 'max', 'oom', 'oom_kill'):
            for value in (True, 1, -1):
                bad = record('answers')
                bad['snapshots'][0]['memory']['memory_events'][event] = value
                with self.subTest(event=event, value=value), self.assertRaises(ValueError):
                    proof.validate_record(bad, 'answers')
        with self.assertRaises(ValueError):
            proof.validate_record(record('answers'), 'answers', {'cgroup': '/sys/fs/cgroup/wrong',
                                                                 'private_cgroup_attestation': '/wrong'})


class LifecycleReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        policy = patch.object(proof, 'approved_contract', return_value={'profile': LIFECYCLE_PROFILE})
        policy.start()
        self.addCleanup(policy.stop)
        self.fixture = ReplayFixture(self.root)

    def test_complete_raw_cpu_native_replay_and_exact_stage_marker(self):
        self.assertEqual(self.fixture.cpu['evaluated_positions'], 9)
        native = proof.summarize_stage(self.root, 'answers')
        self.assertEqual(native['evaluated_positions'], 46)
        self.assertEqual(native['audit_events'], events())
        self.assertEqual(native['successful_requests'], 4)
        self.assertEqual(native['cancelled_requests'], 1)
        self.fixture.json('answers-report.json', native)
        log = 'HAG_LIFECYCLE_REPORT ' + json.dumps(native) + '\nPASS hag-lifecycle stage=answers qualified=true\n'
        self.assertEqual(proof.validate_stage(self.root, 'answers', log), native)
        with self.assertRaises(ValueError):
            proof.validate_stage(self.root, 'answers', log + log)
        self.assertEqual(proof.server_processes(self.root, 'answers'), [{'pid': 41002, 'exit_code': 0}])

    def test_raw_cpu_http_failure_cannot_be_hidden_by_successful_reports(self):
        raw = self.fixture.read('cpu-reference-hello-http.json')
        raw['status'] = 500
        self.fixture.json('cpu-reference-hello-http.json', raw)
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'cpu-reference')
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_raw_replay_rejects_failure_flags_and_status_boolean(self):
        original = self.fixture.read('answers-hello-http.json')
        for key, value in [('truncated', True), ('read_error', 'IncompleteRead'), ('status', True)]:
            raw = dict(original, **{key: value})
            self.fixture.json('answers-hello-http.json', raw)
            with self.subTest(key=key), self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')
        for key in ('truncated', 'read_error'):
            raw = dict(original)
            del raw[key]
            self.fixture.json('answers-hello-http.json', raw)
            with self.subTest(missing=key), self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_busy_unload_requires_actual_409_and_runtime_busy(self):
        original = self.fixture.read('answers-busy-unload-http.json')
        for status, value in [(200, {'error': {'code': 'runtime_busy'}}),
                              (409, {'error': {'code': 'other'}}),
                              (409, {'error': {'code': 'runtime_busy'}, 'choices': []})]:
            raw = dict(original, status=status, body_base64=base64.b64encode(json.dumps(value).encode()).decode())
            self.fixture.json('answers-busy-unload-http.json', raw)
            with self.subTest(status=status, value=value), self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_replay_detects_reuse_reload_and_reference_identity_changes(self):
        original = self.fixture.read('answers-reuse-http.json')
        bad = dict(original, body_base64=base64.b64encode(json.dumps(completion(' Wrong')).encode()).decode())
        self.fixture.json('answers-reuse-http.json', bad)
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')
        self.fixture.json('answers-reuse-http.json', original)
        reference = self.fixture.read('cpu-reference.json')
        reference['identities']['binary_sha256'] = 'f' * 64
        self.fixture.json('cpu-reference.json', reference)
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_final_log_extra_audit_or_truncation_fails_full_replay(self):
        path = self.root / 'answers-hag-server.log'
        for data in (event_bytes(events())[:-1], event_bytes(events() + [dict(events()[0], request_sequence=6)])):
            path.write_bytes(data)
            with self.subTest(data=data), self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')


class FakeResponse:
    def __init__(self, data, status=200, content_type='application/json'):
        self.body = io.BytesIO(data)
        self.status, self.content_type = status, content_type
        self.closed = False

    def getheader(self, name, default=''):
        return self.content_type if name == 'Content-Type' else default

    def read(self, length):
        return self.body.read(length)

    def readline(self, length):
        return self.body.readline(length)

    def close(self):
        self.closed = True


class LifecycleTransportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.server = gate.Server(self.root, 'hag', 'answers')
        self.connection = Mock()

    def attach(self, response):
        self.server.open = Mock(return_value=(self.connection, response))

    def test_cancel_records_busy_refusal_then_actual_bidirectional_shutdown(self):
        response = FakeResponse(stream_bytes(), content_type='text/event-stream')
        self.attach(response)
        self.server.recorded_json = Mock(return_value=(409, {'error': {'code': 'runtime_busy'}}))
        self.server.stream(REFERENCE, cancel=True)
        self.server.recorded_json.assert_called_once_with('busy-unload', 'POST', f'/v1/models/{proof.MODEL_ID}/unload', {})
        self.connection.sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.assertEqual(self.server.record['socket_shutdown'], {'how': 'SHUT_RDWR', 'succeeded': True})
        self.assertTrue(response.closed)
        self.connection.close.assert_called_once()
        proof.stream_artifact(self.root, 'answers-cancel', REFERENCE, cancel=True)
        self.assertNotIn(b'[DONE]', (self.root / 'answers-cancel.sse').read_bytes())

    def test_cancel_rejects_natural_completion_before_shutdown(self):
        data = chunk({'role': 'assistant'}) + chunk({'content': TEXT}, finish='length') + b'data: [DONE]\n'
        response = FakeResponse(data, content_type='text/event-stream')
        self.attach(response)
        with self.assertRaises(ValueError):
            self.server.stream(REFERENCE, cancel=True)
        self.connection.sock.shutdown.assert_not_called()
        self.assertIsNone(self.server.record['socket_shutdown'])
        self.assertTrue(response.closed)
        self.assertTrue((self.root / 'answers-cancel.sse').read_bytes())

    def test_busy_unload_failure_and_missing_socket_never_claim_shutdown(self):
        for status, value, missing_socket in [(200, {}, False), (409, {'error': {'code': 'runtime_busy'}}, True)]:
            with self.subTest(status=status):
                for path in self.root.iterdir():
                    path.unlink()
                response = FakeResponse(stream_bytes(True), content_type='text/event-stream')
                self.connection = Mock()
                if missing_socket:
                    self.connection.sock = None
                self.attach(response)
                self.server.recorded_json = Mock(return_value=(status, value))
                with self.assertRaises(ValueError):
                    self.server.stream(REFERENCE, cancel=True)
                self.assertIsNone(self.server.record['socket_shutdown'])
                self.assertTrue(response.closed)

    def test_shutdown_oserror_leaves_failure_prefix_without_success_claim(self):
        self.attach(FakeResponse(stream_bytes(True), content_type='text/event-stream'))
        self.server.recorded_json = Mock(return_value=(409, {'error': {'code': 'runtime_busy'}}))
        self.connection.sock.shutdown.side_effect = OSError('simulated shutdown failure')
        with self.assertRaises(OSError):
            self.server.stream(REFERENCE, cancel=True)
        self.assertIsNone(self.server.record['socket_shutdown'])
        self.assertTrue((self.root / 'answers-cancel.sse').read_bytes())

    def test_http_error_and_invalid_json_preserve_raw_response(self):
        for name, data, status in [('http-error', b'{"error":"failed"}', 500), ('invalid-json', b'not json', 200)]:
            response = FakeResponse(data, status=status)
            self.attach(response)
            if status == 500:
                observed, _ = self.server.recorded_json(name, 'GET', '/health')
                self.assertEqual(observed, 500)
            else:
                with self.assertRaises(ValueError):
                    self.server.recorded_json(name, 'GET', '/health')
            saved = proof.read_json(self.root / ('answers-' + name + '-http.json'))
            self.assertEqual(base64.b64decode(saved['body_base64']), data)
            self.assertEqual(saved['status'], status)
            self.assertFalse(saved['truncated'])
            self.assertTrue(response.closed)

    def test_oversized_and_incomplete_http_preserve_bounded_failure_records(self):
        oversized = FakeResponse(b'x' * (proof.BODY_LIMIT + 1))
        self.attach(oversized)
        with self.assertRaises(ValueError):
            self.server.recorded_json('oversize', 'GET', '/health')
        saved = proof.read_json(self.root / 'answers-oversize-http.json', 2 * proof.BODY_LIMIT)
        self.assertTrue(saved['truncated'])
        self.assertLessEqual(len(base64.b64decode(saved['body_base64'])), proof.BODY_LIMIT)
        partial = b'{"partial":'
        broken = FakeResponse(b'')
        broken.read = Mock(side_effect=http.client.IncompleteRead(partial, 20))
        self.attach(broken)
        with self.assertRaises((ValueError, http.client.IncompleteRead)):
            self.server.recorded_json('incomplete', 'GET', '/health')
        saved = proof.read_json(self.root / 'answers-incomplete-http.json')
        self.assertEqual(saved['read_error'], 'IncompleteRead')
        self.assertEqual(base64.b64decode(saved['body_base64']), partial)
        self.assertTrue(broken.closed)


if __name__ == '__main__':
    unittest.main()
