"""Pure raw-evidence rejection tests for Stop, reset, and fresh-chat reuse."""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_stop as proof
import inputs
from profiles import fixture_document
from test_ui_history import captured, completion, sse_bytes, sse_packets

GROUP = '/sys/fs/cgroup/user.slice/ui-stop.service'


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def partial(text='<think>\n\n</think>\n\n1 2'):
    packets = sse_packets(completion(text, prompt=476))[:3]
    return sse_bytes(packets, done=False)


class Evidence:
    def __init__(self, root):
        self.root, self.document = root, fixture_document(proof.PROFILE)
        self.ids = {'dependencies': {'prompt-inspect': 'a' * 64}, 'checkpoint_sha256': 'b' * 64,
                    'client_identity': {'kind': 'antimatter-ui', 'profile': proof.PROFILE, 'commit': 'c' * 40,
                       'base_url': 'http://127.0.0.1:48321/v1',
                       'local_host': {'boot_id': '262efb57-e5c3-4393-9144-6bf66a558a48'}}}
        save(root / 'answer-fixtures.json', self.document)
        save(root / 'qualification-contract.json', inputs.profile_contract(proof.PROFILE, 'antimatter-ui'))
        (root / 'client').mkdir()
        save(root / 'client/identity.json', self.ids['client_identity'])

    def control(self, mode):
        backend = 'cpu' if mode == 'cpu-reference' else 'hag'
        def record(name, value, method='GET', route='/v1/models', request=None):
            save(self.root / f'{mode}-{name}.json', captured(value, request, method, route))
        record('identity', {'service': 'beholdd', 'runtime': 'native', 'generation': 2,
                           'execution_backends': ['cpu'] if backend == 'cpu' else ['cpu', 'hag']}, route='/v1/beholdd/info')
        labels = [('cold', False), ('after-turn-2', True), ('unloaded', False)]
        if mode == 'answers':
            labels.insert(1, ('after-turn-1', True))
        for name, loaded in labels:
            record(name + '-model', {'data': [{'id': proof.MODEL_ID, 'loaded': loaded,
                   'execution_limits': {'max_evaluated_tokens': 2048} if backend == 'hag' else {}}]})
        for number in (1, 2):
            record(f'unload-{number}', {'id': proof.MODEL_ID, 'loaded': False}, 'POST', f'/v1/models/{proof.MODEL_ID}/unload', {})
        save(self.root / f'{mode}-server-process.json', {'pid': 120 if backend == 'cpu' else 121, 'exit_code': 0,
             'idle_sigterm_exit': True, 'start_time_ticks': 77, 'no_new_privileges': True, 'cgroup': GROUP})
        save(self.root / f'{mode}-child-result.json', {'cgroup': GROUP})

    def metadata(self, turn, request):
        stem, prompt = f'cpu-reference-turn-{turn}', proof.prompt_bound(turn)
        path = self.root / (stem + '-request.json')
        save(path, request)
        value = {'execution': 'not_requested', 'tokenization': 'exact_checkpoint_metadata',
                 'checkpoint': '/fixture/model.gguf', 'requests': [{'request': str(path), 'status': 'measured',
                 'message_count': 2, 'tool_count': 0, 'rendered_prompt_bytes': 2750, 'prompt_tokens': prompt,
                 'output_tokens': request['max_tokens'], 'minimum_evaluated_positions': proof.ceiling(turn),
                 'output_budget_source': 'explicit'}]}
        save(self.root / (stem + '-inspector.log'), value)
        (self.root / (stem + '-inspector-stderr.log')).write_bytes(b'')
        save(self.root / (stem + '-measurement.json'), {
            'command': [str(self.root / 'dependencies/prompt-inspect'), '/fixture/model.gguf', str(path)],
            'pid': 200 + turn, 'exit_code': 0, 'timed_out': False, 'elapsed_seconds': .02,
            'inspector_sha256': self.ids['dependencies']['prompt-inspect'], 'request_sha256': proof.digest(path),
            'model_sha256': self.ids['checkpoint_sha256'], 'result': value,
            'stdout_sha256': proof.digest(self.root / (stem + '-inspector.log')),
            'stderr_sha256': proof.digest(self.root / (stem + '-inspector-stderr.log'))})

    def cancellation(self):
        request = proof.request_body(self.document, 1)
        raw = partial()
        response = {'method': 'POST', 'route': '/v1/chat/completions', 'request': request, 'status': 200,
                    'content_type': 'text/event-stream', 'read_error': None, 'body_truncated': False,
                    'raw_body_base64': base64.b64encode(raw).decode()}
        save(self.root / 'answers-turn-1-response.json', response)
        event = {'request_sequence': 1, 'evaluated_positions': 484, 'outcome': 'cancelled', 'reset_ok': True, 'poisoned': False}
        receipt = {'schema': 1, 'clock': 'monotonic_seconds', 'started': 100., 'first_content_forwarded': 102.,
                   'downstream_disconnect': 103., 'upstream_shutdown': 103.01, 'audit_observed': 103.2,
                   'warm_residency_observed': 103.3, 'disconnect_kind': 'eof',
                   'shutdown': {'how': 'SHUT_RDWR', 'succeeded': True},
                   'server_before': {'pid': 121, 'start_time_ticks': 77}, 'server_after': {'pid': 121, 'start_time_ticks': 77},
                   'forwarded_raw_base64': base64.b64encode(raw).decode(), 'forwarded_sha256': hashlib.sha256(raw).hexdigest(),
                   'captured_sha256': hashlib.sha256(raw).hexdigest(), 'audit_event': event}
        save(self.root / 'answers-turn-1-cancellation.json', receipt)
        return receipt

    def listener(self):
        ready = {'schema': 1, 'profile': proof.PROFILE, 'host': '127.0.0.1', 'port': 48321, 'pid': 456,
                 'client_identity_sha256': proof.digest(self.root / 'client/identity.json'),
                 'boot_id': 'b7271430-e184-4633-948e-3e1301e7536d', 'run_id': '27f30da4-674b-42a8-a0ca-549c5a09d364',
                 'delivery': proof.DELIVERY}
        save(self.root / 'answers-ui-ready.json', ready)
        save(self.root / 'answers-ui-listener.json', {**ready, 'request_count': 2, 'forwarded_count': 2,
             'response_count': 1, 'cancelled_count': 1, 'rejected_count': 0, 'closed': True,
             'empty_connection_count': 0, 'empty_connections': []})

    def stage(self, mode):
        self.control(mode)
        cancelled = self.cancellation() if mode == 'answers' else None
        for turn in (1, 2):
            request = proof.request_body(self.document, turn)
            stem = f'{mode}-turn-{turn}'
            save(self.root / (stem + '-request.json'), request)
            if mode == 'cpu-reference':
                self.metadata(turn, request)
            save(self.root / (stem + '-lineage.json'), proof.lineage(self.root, mode, turn))
            if turn == 2:
                response = captured(completion(), request)
                save(self.root / (stem + '-response.json'), response)
            if mode == 'answers':
                raw = json.dumps(request).encode()
                sha = hashlib.sha256(raw).hexdigest()
                save(self.root / (stem + '-admission.json'), {'turn': turn, 'request_raw_base64': base64.b64encode(raw).decode(),
                     'request_raw_sha256': sha, 'forwarded_raw_base64': base64.b64encode(raw).decode(), 'forwarded_raw_sha256': sha,
                     'request_received_monotonic': 99. if turn == 1 else 104.,
                     'response_body_sha256': cancelled['forwarded_sha256'] if turn == 1 else
                         hashlib.sha256(base64.b64decode(response['raw_body_base64'])).hexdigest(),
                     'reset_verified': True, 'parity_verified': turn == 2})
        audit = [] if mode == 'cpu-reference' else [cancelled['audit_event'], {'request_sequence': 2,
                 'evaluated_positions': 473, 'outcome': 'success', 'reset_ok': True, 'poisoned': False}]
        backend = 'cpu' if mode == 'cpu-reference' else 'hag'
        (self.root / f'{mode}-{backend}-server.log').write_text(''.join('HAG_SERVING_AUDIT ' + json.dumps(row) + '\n' for row in audit))
        if mode == 'cpu-reference':
            values, turns, _ = proof.reconstruct(self.root, mode, self.ids)
            save(self.root / 'cpu-reference.json', {'schema': 1, 'profile': proof.PROFILE, 'identities': self.ids,
                 'capacity': 2048, 'answers': values, 'turns': turns, 'cpu_exit_code': 0, 'final_unloaded': True})
        else:
            self.listener()
        report = proof.summarize_stage(self.root, mode)
        save(self.root / (mode + '-report.json'), report)
        save(self.root / (mode + '-result.json'), {'qualified': True, 'failures': [], 'exit_code': 0, 'probe': report})
        text = proof.PREFIX + json.dumps(report) + f'\nPASS hag-ui-stop stage={mode} qualified=true\n'
        (self.root / (mode + '.log')).write_text(text)
        return report, text


class StopProofTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.e = Evidence(self.root)
        for target, name, value in ((proof, 'identity', self.e.ids), (proof, 'model_path', Path('/fixture/model.gguf'))):
            mock = patch.object(target, name, return_value=value)
            mock.start(); self.addCleanup(mock.stop)

    def full(self):
        self.e.stage('cpu-reference')
        return self.e.stage('answers')

    def mutate(self, name, update):
        path = self.root / name
        value = proof.read_json(path)
        update(value)
        save(path, value)

    def test_complete_replay_is_backend_only_until_real_ui_closeout(self):
        report, text = self.full()
        self.assertEqual(proof.validate_stage(self.root, 'answers', text), report)
        self.assertEqual((report['requests'], report['evaluated_positions'], report['maximum_evaluated_positions']), (2, 957, 1021))
        self.assertFalse(report['live_ui_qualified'])
        self.assertFalse(report['live_cancellation_qualified'])
        self.assertTrue(report['backend_cancellation_qualified'])
        self.assertEqual(proof.server_processes(self.root, 'answers'), [{'pid': 121, 'exit_code': 0}])
        self.assertEqual(len(proof.broker_records(self.root)), 2)
        self.assertTrue(set(proof.evidence_names('answers')) <= {p.name for p in self.root.iterdir()})
        cpu = proof.summarize_stage(self.root, 'cpu-reference')
        self.assertEqual(cpu['requests'], 1)
        self.assertEqual([row['executed'] for row in cpu['turns']], [False, True])

    def test_cancelled_stream_requires_content_no_terminal_and_real_socket_shutdown(self):
        self.full()
        original = proof.read_json(self.root / 'answers-turn-1-cancellation.json')
        for update in ({'shutdown': {'how': 'SHUT_RDWR', 'succeeded': False}}, {'disconnect_kind': 'synthetic'},
                       {'first_content_forwarded': 104.}, {'upstream_shutdown': 105.}, {'server_after': {'pid': 122, 'start_time_ticks': 77}},
                       {'audit_event': {**original['audit_event'], 'outcome': 'success'}}):
            save(self.root / 'answers-turn-1-cancellation.json', {**original, **update})
            with self.subTest(update=update), self.assertRaises(ValueError):
                proof.cancellation_receipt(self.root)

    def test_hidden_thinking_or_completed_stream_cannot_qualify_stop(self):
        self.full()
        for raw in (partial('<think>123'), partial('<think>123</think>'),
                    sse_bytes(sse_packets(completion(prompt=476)))):
            response = proof.read_json(self.root / 'answers-turn-1-response.json')
            receipt = proof.read_json(self.root / 'answers-turn-1-cancellation.json')
            encoded, sha = base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()
            save(self.root / 'answers-turn-1-response.json', {**response, 'raw_body_base64': encoded})
            save(self.root / 'answers-turn-1-cancellation.json', {**receipt, 'forwarded_raw_base64': encoded,
                 'forwarded_sha256': sha, 'captured_sha256': sha})
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                proof.cancellation_receipt(self.root)

    def test_reuse_requires_fresh_wire_prompt_and_reset_before_admission(self):
        cpu, _ = self.e.stage('cpu-reference')
        body = proof.request_body(self.e.document, 2)
        self.assertEqual(len(body['messages']), 2)
        mutated = deepcopy(body)
        mutated['messages'].insert(1, {'role': 'assistant', 'content': '1 2'})
        with self.assertRaises(ValueError):
            proof.admit_request(2, mutated, cpu, self.e.document)
        self.e.stage('answers')
        self.mutate('answers-turn-2-admission.json', lambda value: value.update(request_received_monotonic=103.))
        with self.assertRaisesRegex(ValueError, 'before cancellation/reset'):
            proof.summarize_stage(self.root, 'answers')

    def test_listener_replays_bounded_empty_connections_before_admission(self):
        self.full()
        rows = [
            {'sequence': 1, 'reason': 'eof', 'received_bytes': 0, 'after_request_count': 0,
             'observed_monotonic': 98.},
            {'sequence': 2, 'reason': 'reset', 'received_bytes': 0, 'after_request_count': 1,
             'observed_monotonic': 103.5}]
        self.mutate('answers-ui-listener.json', lambda value: value.update(
            empty_connection_count=2, empty_connections=rows))
        self.assertEqual(proof.listener(self.root)['empty_connections'], rows)
        self.assertIn('answers-ui-listener.json', proof.evidence_names('answers'))

    def test_listener_rejects_empty_connection_count_and_byte_tampering(self):
        self.full()
        path = self.root / 'answers-ui-listener.json'
        original = proof.read_json(path)
        row = {'sequence': 1, 'reason': 'eof', 'received_bytes': 0, 'after_request_count': 1,
               'observed_monotonic': 103.5}
        changes = [
            {'empty_connection_count': True, 'empty_connections': [row]},
            {'empty_connection_count': 0, 'empty_connections': [row]},
            {'empty_connection_count': 3, 'empty_connections': [row] * 3},
            *({'empty_connection_count': 1, 'empty_connections': [{**row, **change}]} for change in (
                {'sequence': True}, {'sequence': 2}, {'reason': 'timeout'}, {'received_bytes': 1},
                {'received_bytes': False}, {'after_request_count': True}, {'after_request_count': 2},
                {'observed_monotonic': 0}, {'observed_monotonic': 103.}, {'observed_monotonic': 105.}))]
        for change in changes:
            with self.subTest(change=change):
                save(path, {**original, **change})
                with self.assertRaises(ValueError):
                    proof.listener(self.root)

    def test_listener_rejects_empty_connection_order_tampering(self):
        self.full()
        path = self.root / 'answers-ui-listener.json'
        original = proof.read_json(path)
        row = {'sequence': 1, 'reason': 'eof', 'received_bytes': 0, 'after_request_count': 1,
               'observed_monotonic': 103.6}
        for second in ({**row, 'sequence': 2, 'observed_monotonic': 103.5},
                       {**row, 'sequence': 2, 'after_request_count': 0, 'observed_monotonic': 98.}):
            with self.subTest(second=second):
                save(path, {**original, 'empty_connection_count': 2, 'empty_connections': [row, second]})
                with self.assertRaises(ValueError):
                    proof.listener(self.root)

    def test_native_audit_must_cancel_once_then_succeed_once(self):
        self.full()
        path = self.root / 'answers-hag-server.log'
        original = path.read_text()
        for raw in (original.replace('"cancelled"', '"success"'), original.replace('"reset_ok": true', '"reset_ok": false'),
                    original + original.splitlines()[-1] + '\n'):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_cpu_must_not_execute_first_count_up_request(self):
        self.e.stage('cpu-reference')
        save(self.root / 'cpu-reference-turn-1-response.json', {})
        with self.assertRaisesRegex(ValueError, 'must not execute'):
            proof.summarize_stage(self.root, 'cpu-reference')

    def test_metadata_cannot_widen_fixed_measured_count_prompt(self):
        self.e.stage('cpu-reference')
        name = 'cpu-reference-turn-1-inspector.log'
        self.mutate(name, lambda value: value['requests'][0].update(prompt_tokens=477, minimum_evaluated_positions=540))
        self.mutate('cpu-reference-turn-1-measurement.json', lambda value: value.update(
            result=proof.read_json(self.root / name), stdout_sha256=proof.digest(self.root / name)))
        with self.assertRaisesRegex(ValueError, 'fixed UI Stop bounds'):
            proof.summarize_stage(self.root, 'cpu-reference')

    def test_partial_capture_may_end_midframe_but_delivered_prefix_must_be_exact(self):
        self.full()
        response = proof.read_json(self.root / 'answers-turn-1-response.json')
        receipt = proof.read_json(self.root / 'answers-turn-1-cancellation.json')
        raw = base64.b64decode(response['raw_body_base64']) + b'data: {"i'
        save(self.root / 'answers-turn-1-response.json', {**response, 'raw_body_base64': base64.b64encode(raw).decode()})
        save(self.root / 'answers-turn-1-cancellation.json', {**receipt, 'captured_sha256': hashlib.sha256(raw).hexdigest()})
        self.assertEqual(proof.cancellation_receipt(self.root)['forwarded_sha256'], receipt['forwarded_sha256'])
        self.mutate('answers-turn-1-cancellation.json', lambda value: value.update(forwarded_raw_base64=base64.b64encode(raw).decode(),
                    forwarded_sha256=hashlib.sha256(raw).hexdigest()))
        with self.assertRaisesRegex(ValueError, 'incomplete SSE event'):
            proof.cancellation_receipt(self.root)

    def test_sse_parser_rejects_identity_changes_duplicate_terminal_and_bad_usage(self):
        for mutation in (lambda rows: rows[1].update(id='chatcmpl-beholdd-999'),
                         lambda rows: rows[-1]['usage'].update(total_tokens=1),
                         lambda rows: rows.insert(0, rows[0]), lambda rows: rows.append(rows[-1])):
            rows = sse_packets(completion()); mutation(rows)
            with self.assertRaises(ValueError):
                state = proof.SseState(); state.feed(sse_bytes(rows)); state.complete()


if __name__ == '__main__':
    unittest.main()
