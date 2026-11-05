"""Hardware-free UI admission, exact byte forwarding and backend replay tests."""
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
import ui_history as proof
import inputs
from profiles import fixture_document, GENERATED_HISTORY_PROFILE

TEXT = '<think>\n\n</think>\n\nready.'
GROUP = '/sys/fs/cgroup/user.slice/ui-history.service'


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def completion(text=TEXT, prompt=467, count=6, identity='chatcmpl-beholdd-1'):
    return {'id': identity, 'object': 'chat.completion', 'model': proof.MODEL_ID,
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': count, 'total_tokens': prompt + count}}


def sse_packets(value):
    base = {'id': value['id'], 'object': 'chat.completion.chunk', 'created': 1730000000, 'model': value['model']}
    def packet(delta, finish=None, usage=None):
        result = {**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
        if usage is not None:
            result['usage'] = usage
        return result
    text = value['choices'][0]['message']['content']
    return [packet({'role': 'assistant'}), packet({'content': text[:5]}), packet({'content': text[5:]}),
            packet({}, 'stop', value['usage'])]


def sse_bytes(packets, done=True):
    raw = ''.join('data: ' + json.dumps(packet) + '\n\n' for packet in packets)
    return (raw + ('data: [DONE]\n\n' if done else '')).encode()


def captured(value, request=None, method='POST', route='/v1/chat/completions'):
    streaming = value.get('object') == 'chat.completion'
    raw = sse_bytes(sse_packets(value)) if streaming else json.dumps(value).encode()
    return {'method': method, 'route': route, 'request': request, 'status': 200,
            'content_type': 'text/event-stream' if streaming else 'application/json',
            'read_error': None, 'body_truncated': False,
            'raw_body_base64': base64.b64encode(raw).decode()}


class Evidence:
    """Finite synthetic records; no socket, subprocess or device access."""
    def __init__(self, root):
        self.root = root
        self.document = fixture_document(proof.PROFILE)
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
                           'execution_backends': ['cpu'] if backend == 'cpu' else ['cpu', 'hag']},
               route='/v1/beholdd/info')
        for name, loaded in (('cold', False), ('after-turn-1', True), ('after-turn-2', True), ('unloaded', False)):
            record(name + '-model', {'data': [{'id': proof.MODEL_ID, 'loaded': loaded,
                   'execution_limits': {'max_evaluated_tokens': 2048} if backend == 'hag' else {}}]})
        for number in (1, 2):
            record(f'unload-{number}', {'id': proof.MODEL_ID, 'loaded': False}, 'POST',
                   f'/v1/models/{proof.MODEL_ID}/unload', {})
        save(self.root / f'{mode}-server-process.json', {'pid': 120 if backend == 'cpu' else 121,
             'exit_code': 0, 'idle_sigterm_exit': True, 'start_time_ticks': 77,
             'no_new_privileges': True, 'cgroup': GROUP})
        save(self.root / f'{mode}-child-result.json', {'cgroup': GROUP})

    def metadata(self, turn, request):
        stem = f'cpu-reference-turn-{turn}'
        prompt = 467 if turn == 1 else 491
        path = self.root / (stem + '-request.json')
        save(path, request)
        value = {'execution': 'not_requested', 'tokenization': 'exact_checkpoint_metadata',
                 'checkpoint': '/fixture/model.gguf', 'requests': [
                    {'request': str(path), 'status': 'measured', 'message_count': 2 if turn == 1 else 4,
                     'tool_count': 0, 'rendered_prompt_bytes': 2710 if turn == 1 else 2823,
                     'prompt_tokens': prompt, 'output_tokens': 128,
                     'minimum_evaluated_positions': prompt + 127, 'output_budget_source': 'explicit'}]}
        save(self.root / (stem + '-inspector.log'), value)
        (self.root / (stem + '-inspector-stderr.log')).write_bytes(b'')
        save(self.root / (stem + '-measurement.json'), {
            'command': [str(self.root / 'dependencies/prompt-inspect'), '/fixture/model.gguf', str(path)],
            'pid': 200 + turn, 'exit_code': 0, 'timed_out': False, 'elapsed_seconds': .02,
            'inspector_sha256': self.ids['dependencies']['prompt-inspect'], 'request_sha256': proof.digest(path),
            'model_sha256': self.ids['checkpoint_sha256'], 'result': value,
            'stdout_sha256': proof.digest(self.root / (stem + '-inspector.log')),
            'stderr_sha256': proof.digest(self.root / (stem + '-inspector-stderr.log'))})

    def listener(self):
        ready = {'schema': 1, 'profile': proof.PROFILE, 'host': '127.0.0.1', 'port': 48321, 'pid': 456,
                 'client_identity_sha256': proof.digest(self.root / 'client/identity.json'),
                 'boot_id': 'b7271430-e184-4633-948e-3e1301e7536d',
                 'run_id': '27f30da4-674b-42a8-a0ca-549c5a09d364',
                 'delivery': 'buffered-sse-after-parity-reset'}
        save(self.root / 'answers-ui-ready.json', ready)
        save(self.root / 'answers-ui-listener.json', {**ready, 'request_count': 2, 'forwarded_count': 2,
             'response_count': 2, 'rejected_count': 0, 'closed': True})

    def stage(self, mode):
        self.control(mode)
        answers = []
        for turn in (1, 2):
            prompt = 467 if turn == 1 else 491
            parent = None if turn == 1 else proof.parent_binding(self.root, mode, answers[0])
            parent_text = None if parent is None else parent['text']
            request = proof.request_body(self.document, turn, parent_text)
            stem = f'{mode}-turn-{turn}'
            save(self.root / (stem + '-request.json'), request)
            if mode == 'cpu-reference':
                self.metadata(turn, request)
            lineage = {'turn': turn, 'request_sha256': proof.digest(self.root / (stem + '-request.json')),
                       'history_parent': parent, 'measurement_sha256': proof.digest(self.root / f'cpu-reference-turn-{turn}-measurement.json')}
            if mode == 'answers':
                lineage.update(cpu_reference_sha256=proof.digest(self.root / 'cpu-reference.json'),
                               first_turn_parity_verified=turn == 2)
            save(self.root / (stem + '-lineage.json'), lineage)
            payload = completion(prompt=prompt, identity=f'chatcmpl-beholdd-{1000 + turn if mode == "cpu-reference" else 2000 + turn}')
            raw = captured(payload, request)
            save(self.root / (stem + '-response.json'), raw)
            if mode == 'answers':
                request_raw = json.dumps(request).encode()
                encoded = base64.b64encode(request_raw).decode()
                sha = hashlib.sha256(request_raw).hexdigest()
                save(self.root / (stem + '-admission.json'), {
                    'turn': turn, 'request_raw_base64': encoded, 'request_raw_sha256': sha,
                    'forwarded_raw_base64': encoded, 'forwarded_raw_sha256': sha,
                    'response_body_sha256': hashlib.sha256(base64.b64decode(raw['raw_body_base64'])).hexdigest(),
                    'parity_verified': True, 'reset_verified': True})
            answers.append(proof.generation(200, payload, self.document, turn, prompt, parent_text))
        audit = [] if mode == 'cpu-reference' else [
            {'request_sequence': i, 'evaluated_positions': value['evaluated_positions'],
             'outcome': 'success', 'reset_ok': True, 'poisoned': False} for i, value in enumerate(answers, 1)]
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
        text = proof.PREFIX + json.dumps(report) + f'\nPASS hag-ui-history stage={mode} qualified=true\n'
        (self.root / (mode + '.log')).write_text(text)
        return report, text


class UiHistoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.e = Evidence(self.root)
        identity = patch.object(proof, 'identity', return_value=self.e.ids)
        model = patch.object(proof.history, 'model_path', return_value=Path('/fixture/model.gguf'))
        identity.start(); model.start()
        self.addCleanup(identity.stop); self.addCleanup(model.stop)

    def mutate(self, name, change):
        path = self.root / name
        value = json.loads(path.read_text())
        change(value)
        save(path, value)

    def full(self):
        self.e.stage('cpu-reference')
        return self.e.stage('answers')

    def test_complete_backend_replay_does_not_claim_live_ui(self):
        report, text = self.full()
        self.assertEqual(proof.validate_stage(self.root, 'answers', text), report)
        self.assertEqual(report['evaluated_positions'], 970)
        self.assertFalse(report['live_ui_qualified'])
        self.assertTrue(report['client_evidence_pending'])
        self.assertEqual(report['transport'], 'buffered-sse')
        self.assertEqual(proof.server_processes(self.root, 'answers'), [{'pid': 121, 'exit_code': 0}])
        self.assertEqual(len(proof.broker_records(self.root)), 2)
        self.assertTrue(set(proof.evidence_names('answers')) <= {p.name for p in self.root.iterdir()})

    def test_listener_requires_declared_host_relationship_and_same_host_endpoint(self):
        self.e.listener()
        self.assertEqual(proof.listener(self.root)['port'], 48321)
        save(self.root / 'qualification-contract.json',
             inputs.profile_contract(proof.PROFILE, 'antimatter-ui', ui_client_host='same'))
        with self.assertRaisesRegex(ValueError, 'boot differs'):
            proof.listener(self.root)
        for name in ('answers-ui-ready.json', 'answers-ui-listener.json'):
            self.mutate(name, lambda row: row.update(boot_id=self.e.ids['client_identity']['local_host']['boot_id']))
        self.assertEqual(proof.listener(self.root)['port'], 48321)
        for name in ('answers-ui-ready.json', 'answers-ui-listener.json'):
            self.mutate(name, lambda row: row.update(port=48322))
        with self.assertRaisesRegex(ValueError, 'port differs'):
            proof.listener(self.root)
        save(self.root / 'qualification-contract.json', inputs.profile_contract(proof.PROFILE, 'antimatter-ui'))
        with self.assertRaisesRegex(ValueError, 'boot differs'):
            proof.listener(self.root)

    def test_wire_contract_is_distinct_from_old_nonstream_profile(self):
        request = proof.request_body(self.e.document, 1)
        self.assertIs(request['stream'], True)
        self.assertNotIn('temperature', request)
        with self.assertRaises(ValueError):
            proof.request_body(fixture_document(GENERATED_HISTORY_PROFILE), 1)

    def test_admission_refuses_extra_fields_types_and_changed_prompt(self):
        reference, _ = self.e.stage('cpu-reference')
        original = proof.request_body(self.e.document, 1)
        for change in (lambda v: v.update(temperature=0), lambda v: v.update(stream=1),
                       lambda v: v.update(max_tokens=129), lambda v: v.update(tools=[]),
                       lambda v: v['messages'][1].update(content='other prompt')):
            body = deepcopy(original); change(body)
            with self.assertRaises(ValueError):
                proof.admit_request(1, body, None, reference)
        for turn in (True, 0, 3, '1'):
            with self.assertRaises(ValueError):
                proof.admit_request(turn, original, None, reference)

    def test_admission_requires_complete_actual_first_reply(self):
        reference, _ = self.e.stage('cpu-reference')
        parent = {'text': TEXT, 'text_sha256': proof.text_digest(TEXT), 'response_sha256': 'a' * 64}
        request = proof.request_body(self.e.document, 2, TEXT)
        self.assertEqual(proof.admit_request(2, request, parent, reference), request)
        for mutation in ({'text': 'ready.'}, {'text_sha256': '0' * 64}, {'response_sha256': ''}):
            with self.assertRaises(ValueError):
                proof.admit_request(2, request, {**parent, **mutation}, reference)
        with self.assertRaises(ValueError):
            proof.admit_request(2, request, None, reference)

    def test_admission_does_not_accept_an_old_profile_reference(self):
        reference, _ = self.e.stage('cpu-reference')
        reference['profile'] = GENERATED_HISTORY_PROFILE
        with self.assertRaises(ValueError):
            proof.admit_request(1, proof.request_body(self.e.document, 1), None, reference)

    def test_raw_thinking_whitespace_remains_unmodified(self):
        text = ' \n<think>reason\n</think>\n\nready.\n'
        self.assertEqual(proof.request_body(self.e.document, 2, text)['messages'][2]['content'], text)
        for invalid in ('', '<think>ready.', '<think>x</think><think>y</think>ready'):
            with self.assertRaises(ValueError):
                proof.request_body(self.e.document, 2, invalid)

    def test_usage_eos_and_tool_free_requirements(self):
        for change in (lambda v: v['choices'][0].update(finish_reason='length'),
                       lambda v: v['choices'][0]['message'].update(tool_calls=[]),
                       lambda v: v['usage'].update(prompt_tokens=True),
                       lambda v: v['usage'].update(completion_tokens=128, total_tokens=595)):
            value = completion(); change(value)
            with self.assertRaises(ValueError):
                proof.generation(200, value, self.e.document, 1, 467)

    def test_semantic_quality_is_only_diagnostic(self):
        value = proof.generation(200, completion('WRONG.'), self.e.document, 1, 467)
        self.assertEqual(value['quality'], {'correctness': 'fail', 'format': 'pass'})

    def test_forwarded_bytes_cannot_be_reserialized_even_with_equal_json(self):
        self.full()
        def reformat(record):
            raw = json.dumps(json.loads(base64.b64decode(record['forwarded_raw_base64'])), indent=2).encode()
            record.update(forwarded_raw_base64=base64.b64encode(raw).decode(),
                          forwarded_raw_sha256=hashlib.sha256(raw).hexdigest())
        self.mutate('answers-turn-1-admission.json', reformat)
        with self.assertRaisesRegex(ValueError, 'forwarded request bytes changed'):
            proof.summarize_stage(self.root, 'answers')

    def test_released_response_requires_parity_reset_and_matching_hash(self):
        self.full()
        path = self.root / 'answers-turn-1-admission.json'
        original = proof.read_json(path)
        for mutation in ({'parity_verified': False}, {'reset_verified': False}, {'response_body_sha256': '0' * 64}):
            save(path, {**original, **mutation})
            with self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_missing_extra_or_failed_request_counters_refused(self):
        self.full()
        path = self.root / 'answers-ui-listener.json'
        original = proof.read_json(path)
        for mutation in ({'request_count': 3}, {'forwarded_count': 1}, {'response_count': 1},
                         {'rejected_count': 1}, {'closed': False}, {'request_count': True}):
            save(path, {**original, **mutation})
            with self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_listener_cannot_switch_run_or_client_after_readiness(self):
        self.full()
        path = self.root / 'answers-ui-listener.json'
        original = proof.read_json(path)
        for mutation in ({'run_id': '036e5bfe-54f5-4ffb-9bcf-a8289e8ab8f1'},
                         {'client_identity_sha256': '0' * 64}, {'host': '0.0.0.0'}):
            save(path, {**original, **mutation})
            with self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_second_history_parent_is_native_response_artifact(self):
        self.full()
        self.mutate('answers-turn-2-lineage.json', lambda v: v['history_parent'].update(
            response_sha256=proof.digest(self.root / 'cpu-reference-turn-1-response.json')))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_measurement_is_replayed_and_cannot_widen_second_prompt(self):
        self.e.stage('cpu-reference')
        name = 'cpu-reference-turn-2-inspector.log'
        self.mutate(name, lambda v: v['requests'][0].update(prompt_tokens=492, minimum_evaluated_positions=619))
        self.mutate('cpu-reference-turn-2-measurement.json', lambda v: v.update(
            result=proof.read_json(self.root / name), stdout_sha256=proof.digest(self.root / name)))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'cpu-reference')

    def test_extra_or_unhealthy_native_audit_refused(self):
        self.full()
        path = self.root / 'answers-hag-server.log'
        original = path.read_text()
        for raw in (original + original.splitlines()[0] + '\n', original.replace('"reset_ok": true', '"reset_ok": false')):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')

    def test_failed_cpu_prerequisite_blocks_native_replay(self):
        self.full()
        self.mutate('cpu-reference-result.json', lambda value: value.update(qualified=False))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_duplicate_pass_report_refused(self):
        _, text = self.full()
        with self.assertRaises(ValueError):
            proof.validate_stage(self.root, 'answers', text + text)

    def test_sse_canonical_completion_preserves_chunks_and_allows_keepalive_comments(self):
        record = captured(completion())
        raw = base64.b64decode(record['raw_body_base64'])
        record['raw_body_base64'] = base64.b64encode(b': keep-alive\n\n' + raw.replace(b'\n\n', b'\r\n\r\n')).decode()
        value = proof.http_body(record)
        self.assertEqual(value['choices'][0]['message'], {'role': 'assistant', 'content': TEXT})
        self.assertEqual(value['usage'], completion()['usage'])
        self.assertEqual(value['id'], 'chatcmpl-beholdd-1')

    def test_sse_refuses_changed_identity_extra_fields_tools_and_bad_terminal_usage(self):
        mutations = (
            lambda rows: rows[1].update(id='chatcmpl-beholdd-999'),
            lambda rows: rows[1].update(model='other-model'),
            lambda rows: rows[1].update(created=True),
            lambda rows: rows[1].update(error={'message': 'failure'}),
            lambda rows: rows[1]['choices'][0]['delta'].update(tool_calls=[]),
            lambda rows: rows[1].update(usage=completion()['usage']),
            lambda rows: rows[-1].pop('usage'),
            lambda rows: rows[-1]['usage'].update(completion_tokens=True),
            lambda rows: rows[-1]['usage'].update(total_tokens=999),
            lambda rows: rows[-1]['choices'][0].update(finish_reason='length'),
            lambda rows: rows[-1]['choices'][0].update(delta={'content': 'extra'}),
        )
        for mutate in mutations:
            packets = sse_packets(completion()); mutate(packets)
            record = captured(completion())
            record['raw_body_base64'] = base64.b64encode(sse_bytes(packets)).decode()
            with self.subTest(packets=packets), self.assertRaises(ValueError):
                proof.http_body(record)

    def test_sse_requires_one_role_terminal_and_final_sentinel_with_complete_framing(self):
        packets = sse_packets(completion())
        original = sse_bytes(packets)
        bad = (sse_bytes(packets[1:]), sse_bytes([packets[0], packets[0], *packets[1:]]),
               sse_bytes(packets[:-1]), sse_bytes([*packets, packets[-1]]), sse_bytes(packets, done=False),
               b'data: [DONE]\n\n' + original, original + b'data: [DONE]\n\n',
               original + b': late keep-alive\n\n', original[:-1],
               b'event: error\ndata: {}\n\n', original.replace(b'data:', b'event:', 1),
               original.replace(b'"content":', b'"content":null,"content":', 1))
        for raw in bad:
            record = captured(completion())
            record['raw_body_base64'] = base64.b64encode(raw).decode()
            with self.subTest(raw=raw[:80]), self.assertRaises(ValueError):
                proof.http_body(record)


if __name__ == '__main__':
    unittest.main()
