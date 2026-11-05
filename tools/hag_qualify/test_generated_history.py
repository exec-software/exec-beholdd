"""Hardware-free generated-history parser, ancestry and sequencing regressions."""
import base64
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_history as proof
from profiles import fixture_document

_spec = importlib.util.spec_from_file_location('generated_history_worker', Path(__file__).with_name('generated-history-gate.py'))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

TEXT = '<think>\n\n</think>\n\nready.'
GROUP = '/sys/fs/cgroup/user.slice/generated.service'


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def completion(text=TEXT, prompt=467, count=6):
    return {'object': 'chat.completion', 'model': proof.MODEL_ID,
            'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': count, 'total_tokens': prompt + count}}


def captured(value, request=None, method='POST', route='/v1/chat/completions'):
    return {'method': method, 'route': route, 'request': request, 'status': 200,
            'content_type': 'application/json', 'read_error': None, 'body_truncated': False,
            'raw_body_base64': base64.b64encode(json.dumps(value).encode()).decode()}


class Evidence:
    """Create finite synthetic wire/process evidence, never launch anything."""
    def __init__(self, root):
        self.root = root
        self.document = fixture_document(proof.PROFILE)
        self.ids = {'dependencies': {'prompt-inspect': 'a' * 64}, 'checkpoint_sha256': 'b' * 64,
                    'client_identity': {'kind': 'antimatter', 'commit': 'c' * 40,
                                        'manifest_sha256': 'd' * 64, 'node_sha256': 'e' * 64,
                                        'bundle_sha256': 'f' * 64}}
        save(root / 'answer-fixtures.json', self.document)

    def control(self, mode, backend):
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

    def stage(self, mode):
        backend = 'cpu' if mode == 'cpu-reference' else 'hag'
        self.control(mode, backend)
        answers = []
        for turn in (1, 2):
            prompt = 467 if turn == 1 else 491
            parent = None if turn == 1 else proof.parent_binding(self.root, mode, answers[0])
            parent_text = None if parent is None else parent['text']
            request = proof.request_body(self.document, turn, parent_text)
            stem = f'{mode}-turn-{turn}'
            save(self.root / (stem + '-request.json'), request)
            if backend == 'cpu':
                self.metadata(turn, request)
            lineage = {'turn': turn, 'request_sha256': proof.digest(self.root / (stem + '-request.json')),
                       'history_parent': parent, 'measurement_sha256': proof.digest(self.root / f'cpu-reference-turn-{turn}-measurement.json')}
            if backend == 'hag':
                lineage.update(cpu_reference_sha256=proof.digest(self.root / 'cpu-reference.json'),
                               first_turn_parity_verified=turn == 2)
            save(self.root / (stem + '-lineage.json'), lineage)
            raw = captured(completion(prompt=prompt), request)
            if backend == 'hag':
                pid = 300 + turn
                raw.update(clock='monotonic_elapsed_from_request_start', events=[], elapsed_seconds=1.,
                    client={'kind': 'antimatter', 'pid': pid, 'mode': 'chat', 'attempt_count': 1, 'request_count': 1,
                            'error': None, 'response_complete': True, 'max_evaluated_tokens': 2048,
                            'request': request, 'result': TEXT, 'callbacks': [],
                            'preflight': {'capacity': 2048, 'requested_output_tokens': 2048, 'fetch_attempts': 0,
                                          'track_calls': 0, 'prepare_calls': 0, 'error_code': 'context_length_exceeded'},
                            'hooks': {'track_calls': 1, 'prepare_calls': 1, 'track_end_calls': 1}})
                save(self.root / (stem + '-client-request.json'), {
                    'profile': proof.PROFILE, 'turn': turn, 'prompt_tokens': prompt, 'request': request,
                    'history_parent': parent, 'model_descriptor': {'id': proof.MODEL_ID, 'loaded': turn == 2,
                                                                  'execution_limits': {'max_evaluated_tokens': 2048}}})
                save(self.root / (stem + '-client-process.json'), {'pid': pid, 'exit_code': 0,
                     'timed_out': False, 'identity': self.ids['client_identity']})
                (self.root / (stem + '-client.log')).write_bytes(b'')
            save(self.root / (stem + '-response.json'), raw)
            answers.append(proof.generation(200, completion(prompt=prompt), self.document, turn, prompt, parent_text))
        events = [] if backend == 'cpu' else [
            {'request_sequence': i, 'evaluated_positions': answer['evaluated_positions'],
             'outcome': 'success', 'reset_ok': True, 'poisoned': False} for i, answer in enumerate(answers, 1)]
        (self.root / f'{mode}-{backend}-server.log').write_text(''.join('HAG_SERVING_AUDIT ' + json.dumps(row) + '\n' for row in events))
        if backend == 'cpu':
            values, turns, _ = proof.reconstruct(self.root, mode, self.ids)
            save(self.root / 'cpu-reference.json', {'schema': 1, 'profile': proof.PROFILE, 'identities': self.ids,
                 'capacity': 2048, 'answers': values, 'turns': turns, 'cpu_exit_code': 0, 'final_unloaded': True})
        report = proof.summarize_stage(self.root, mode)
        save(self.root / (mode + '-report.json'), report)
        save(self.root / (mode + '-result.json'), {'qualified': True, 'failures': [], 'exit_code': 0, 'probe': report})
        text = proof.PREFIX + json.dumps(report) + f'\nPASS hag-generated-history stage={mode} qualified=true\n'
        (self.root / (mode + '.log')).write_text(text)
        return report, text


class GeneratedHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.e = Evidence(self.root)
        self.identity = patch.object(proof, 'identity', return_value=self.e.ids)
        self.model = patch.object(proof, 'model_path', return_value=Path('/fixture/model.gguf'))
        self.identity.start(); self.model.start()
        self.addCleanup(self.identity.stop); self.addCleanup(self.model.stop)

    def mutate(self, name, change):
        path = self.root / name
        value = json.loads(path.read_text())
        change(value)
        save(path, value)

    def full(self):
        self.e.stage('cpu-reference')
        return self.e.stage('answers')

    def test_complete_cpu_and_native_raw_replay(self):
        report, text = self.full()
        self.assertEqual(proof.validate_stage(self.root, 'answers', text), report)
        self.assertEqual(report['evaluated_positions'], 970)
        self.assertEqual(report['requests'], 2)
        self.assertEqual(proof.server_processes(self.root, 'answers'), [{'pid': 121, 'exit_code': 0}])
        self.assertTrue(set(proof.evidence_names('answers')) <= {p.name for p in self.root.iterdir()})

    def test_raw_thinking_and_whitespace_are_preserved_in_history(self):
        text = ' \n<think>reason\n</think>\n\nready.\n'
        value = proof.request_body(self.e.document, 2, text)
        self.assertEqual(value['messages'][2], {'role': 'assistant', 'content': text})
        self.assertEqual(len(value['messages']), 4)
        self.assertEqual(value['messages'][:2], self.e.document['first_fixture']['messages'])

    def test_unfinished_or_repeated_thinking_is_not_a_parent(self):
        for text in ('<think>ready.', '<think>x</think><think>y</think>ready', '<THINK>x</THINK>ready', ''):
            with self.subTest(text=text), self.assertRaises(ValueError):
                proof.request_body(self.e.document, 2, text)

    def test_wrong_but_complete_answer_remains_diagnostic(self):
        value = proof.generation(200, completion('WRONG.'), self.e.document, 1, 467)
        self.assertEqual(value['quality'], {'correctness': 'fail', 'format': 'pass'})

    def test_eos_length_bool_or_overflow_usage_refused(self):
        mutations = [lambda v: v['choices'][0].update(finish_reason='length'),
                     lambda v: v['usage'].update(prompt_tokens=True),
                     lambda v: v['usage'].update(completion_tokens=128, total_tokens=595),
                     lambda v: v['usage'].update(total_tokens=999)]
        for change in mutations:
            value = completion(); change(value)
            with self.assertRaises(ValueError):
                proof.generation(200, value, self.e.document, 1, 467)

    def test_metadata_prompt_492_refused_without_widening(self):
        self.e.stage('cpu-reference')
        name = 'cpu-reference-turn-2-inspector.log'
        self.mutate(name, lambda v: v['requests'][0].update(prompt_tokens=492, minimum_evaluated_positions=619))
        self.mutate('cpu-reference-turn-2-measurement.json', lambda v: v.update(
            result=proof.read_json(self.root / name), stdout_sha256=proof.digest(self.root / name)))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'cpu-reference')

    def test_changed_metadata_executable_or_request_hash_refused(self):
        self.e.stage('cpu-reference')
        self.mutate('cpu-reference-turn-1-measurement.json', lambda v: v.update(inspector_sha256='0' * 64))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'cpu-reference')

    def test_second_turn_dropped_or_trimmed_history_refused(self):
        self.full()
        self.mutate('answers-turn-2-client-request.json', lambda v: v['request']['messages'][2].update(content='ready.'))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_parent_response_hash_is_backend_specific(self):
        self.full()
        self.mutate('answers-turn-2-lineage.json', lambda v: v['history_parent'].update(
            response_sha256=proof.digest(self.root / 'cpu-reference-turn-1-response.json')))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_native_wrong_full_raw_answer_never_passes(self):
        self.full()
        self.mutate('answers-turn-1-response.json', lambda v: v['client'].update(result='ready'))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'answers')

    def test_extra_missing_or_unhealthy_native_audit_refused(self):
        self.full()
        path = self.root / 'answers-hag-server.log'
        original = path.read_text()
        for raw in (original.splitlines()[0] + '\n', original + original.splitlines()[0] + '\n',
                    original.replace('"reset_ok": true', '"reset_ok": false')):
            path.write_text(raw)
            with self.assertRaises(ValueError):
                proof.summarize_stage(self.root, 'answers')
        path.write_text(original)

    def test_changed_server_group_or_failed_idle_exit_refused(self):
        self.full()
        self.mutate('answers-server-process.json', lambda v: v.update(cgroup='/sys/fs/cgroup/another'))
        with self.assertRaises(ValueError):
            proof.server_processes(self.root, 'answers')

    def test_duplicate_report_or_pass_refused(self):
        _, text = self.full()
        with self.assertRaises(ValueError):
            proof.validate_stage(self.root, 'answers', text + text)

    def test_failed_cpu_response_cannot_be_reference(self):
        self.e.stage('cpu-reference')
        self.mutate('cpu-reference-turn-1-response.json', lambda v: v.update(read_error='IncompleteRead'))
        with self.assertRaises(ValueError):
            proof.summarize_stage(self.root, 'cpu-reference')

    def test_worker_blocks_turn_two_before_request_write_on_first_mismatch(self):
        report, _ = self.e.stage('cpu-reference')
        # A mocked server/provider is sufficient to protect control flow: no
        # second request file or second client invocation may occur on mismatch.
        class FakeServer:
            def __init__(self, *args): self.exited = False
            def __enter__(self): return self
            def __exit__(self, *_): self.cleanup()
            def identity(self): pass
            def cleanup(self): self.exited = True
            def save_process(self): pass
        wrong = deepcopy(report['answers'][0]); wrong['text'] = 'WRONG'
        with patch.object(worker, 'Server', FakeServer), patch.object(worker.answer, 'verify_containment'), \
             patch.object(proof, 'validate_stage', return_value=report), \
             patch.object(worker, 'native_chat', return_value=wrong) as client, \
             patch.dict(worker.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}):
            with self.assertRaisesRegex(ValueError, 'native full response differs'):
                worker.execute(self.root, 'answers')
        self.assertEqual(client.call_count, 1)
        self.assertFalse((self.root / 'answers-turn-2-request.json').exists())


if __name__ == '__main__':
    unittest.main()
