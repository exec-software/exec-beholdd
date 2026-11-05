"""Hardware-free parser, identity and lifecycle checks; no CLI/checkpoint/GPU work."""
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiles import fixture_document, fixture_profile, profile_spec
from answer_quality import quality_assessment
from inputs import profile_contract

SPEC = importlib.util.spec_from_file_location('answer_gate', Path(__file__).with_name('answer-gate.py'))
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
FIXTURE_DOCUMENT = gate.read_json(Path(__file__).with_name('answer-fixtures.json'))
LIST_TEXT = '1,2,3,4,5,6,7,8'


def fixture(name='list_1_to_8'):
    return next(f for f in FIXTURE_DOCUMENT['fixtures'] if f['id'] == name)


def response(text='<think>\n\n</think>\n\n' + LIST_TEXT, prompt=30, completion=8, finish='stop'):
    return {'object': 'chat.completion', 'model': gate.MODEL_ID,
            'choices': [{'index': 0, 'finish_reason': finish,
                         'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': completion,
                      'total_tokens': prompt + completion}}


def chunk(delta=None, finish=None, identifier='test'):
    return ('data: ' + json.dumps({'id': identifier, 'object': 'chat.completion.chunk',
             'model': gate.MODEL_ID, 'choices': [{'index': 0, 'delta': delta or {},
             'finish_reason': finish}]}) + '\n').encode()


def audit(sequence=1, evaluated=38):
    return {'request_sequence': sequence, 'evaluated_positions': evaluated,
            'outcome': 'success', 'reset_ok': True, 'poisoned': False}


def policy(profile):
    spec = profile_spec(profile)
    return {'qualification_kind': spec['qualification_kind'],
            'require_semantic_match': spec['require_semantic_match'],
            'quality_is_diagnostic': not spec['require_semantic_match']}


def client_stream_record(answer, selected):
    lines = [chunk({'role': 'assistant'}), chunk({'content': answer['text']}),
             chunk(finish='stop'), b'data: [DONE]\n']
    return {'status': 200, 'content_type': 'text/event-stream',
            'clock': 'monotonic_elapsed_from_request_start',
            'events': [{'received_seconds': (i + 1) / 10, 'line': line.decode()}
                       for i, line in enumerate(lines)],
            'client': {'kind': 'antimatter', 'pid': 98765, 'request_count': 1, 'attempt_count': 1,
                       'request': gate.request_body(selected, stream=True),
                       'response_complete': True, 'error': None,
                       'callbacks': [{'id': 'hag-client-list', 'content': answer['text']}],
                       'result': {'content': answer['text'], 'toolCalls': [], 'reasoning': '',
                                  'finishReason': 'stop', 'servedModel': gate.MODEL_ID,
                                  'toolCallsCutOff': False}}}


def snapshot_inputs(root):
    """Tiny snapshot identity fixture: no real binary, source tree, device or model."""
    root.mkdir(exist_ok=True)
    inputs = {'schema': 1, 'repo': str(root / 'source'), 'model': str(root / 'model.gguf'),
              'device': str(root / 'renderD128'), 'binary_input': str(root / 'original-beholdd')}
    gate.write_json(root / 'run-config.json', inputs)
    gate.write_json(root / 'answer-fixtures.json', FIXTURE_DOCUMENT)
    gate.write_json(root / 'qualification-contract.json', profile_contract('list-c128'))
    for backend in ('cpu', 'hag'):
        model = {'id': gate.MODEL_ID, 'path': inputs['model'], 'execution_backend': backend,
                 'weight_mode': 'streaming', 'fit_policy': 'exact',
                 'memory_budget_bytes': (2 if backend == 'hag' else 1) * 1024 ** 3,
                 'runtime_headroom_bytes': 128 * 1024 ** 2}
        if backend == 'hag':
            model['hag'] = {'device': inputs['device'], 'memory_budget_bytes': 768 * 1024 ** 2}
        gate.write_json(root / (backend + '-config.json'), {'models': [model]})
    (root / 'beholdd').write_bytes(b'not an executable')
    (root / 'answer-gate.py').write_bytes(b'not a worker')
    (root / 'answer_quality.py').write_bytes(b'not a quality module')
    (root / 'profiles.py').write_bytes(b'not a profile module')
    (root / 'gate_common.py').write_bytes(b'not a common module')
    gate.write_json(root / 'build-source-sha256.json', {})
    files = ('answer-gate.py', 'answer_quality.py', 'profiles.py', 'gate_common.py', 'qualification-contract.json',
             'answer-fixtures.json', 'run-config.json', 'cpu-config.json', 'hag-config.json')
    manifest = {'profile': 'list-c128', 'binaries': {'beholdd': gate.digest(root / 'beholdd')},
                'source_manifest_sha256': gate.digest(root / 'build-source-sha256.json'),
                'model_sha256': 'c' * 64,
                'gate_files': {name: gate.digest(root / name) for name in files}}
    gate.write_json(root / 'expected-manifest.json', manifest)
    return inputs, manifest


def rehash_gate_file(root, name):
    """Let shape tests reach validation after a deliberately consistent test hash."""
    path = root / 'expected-manifest.json'
    manifest = gate.read_json(path)
    manifest['gate_files'][name] = gate.digest(root / name)
    path.write_text(json.dumps(manifest))


class AnswerGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_real_client_evidence_requires_raw_eos_done_and_matches_callbacks(self):
        selected = fixture_document('backend-parity-c128')['fixtures'][0]
        answer = gate.generation(200, response(), selected, 'backend-parity-c128')
        record = client_stream_record(answer, selected)
        metric = gate.validate_client_stream(record, answer, selected, 2, 'backend-parity-c128')
        self.assertEqual(metric['round'], 2)
        self.assertEqual(metric['done_seconds'], .4)
        variants = []
        no_done = deepcopy(record)
        no_done['events'].pop()
        variants.append(no_done)
        malformed = deepcopy(record)
        malformed['events'][1]['line'] = 'data: {bad}\n'
        variants.append(malformed)
        wrong_raw = deepcopy(record)
        wrong_raw['events'][1]['line'] = chunk({'content': '1,2,3'}).decode()
        variants.append(wrong_raw)
        after_done = deepcopy(record)
        after_done['events'].append({'received_seconds': .5, 'line': chunk({'content': 'extra'}).decode()})
        variants.append(after_done)
        late = deepcopy(record)
        late['events'][-1]['received_seconds'] = 30.01
        variants.append(late)
        for bad in variants:
            with self.subTest(record=bad), self.assertRaises(ValueError):
                gate.validate_client_stream(bad, answer, selected, 1, 'backend-parity-c128')

    def test_real_client_refuses_retries_changed_request_and_forged_success(self):
        selected = fixture_document('backend-parity-c128')['fixtures'][0]
        answer = gate.generation(200, response(), selected, 'backend-parity-c128')
        record = client_stream_record(answer, selected)
        changes = [('request_count', 2), ('attempt_count', 2), ('request_count', True),
                   ('response_complete', False), ('error', 'connection failed'), ('pid', True),
                   ('request', {**record['client']['request'], 'max_tokens': 4096}),
                   ('request', {**record['client']['request'], 'temperature': False}),
                   ('callbacks', [{'id': 'wrong', 'content': answer['text']}]),
                   ('callbacks', [{'id': 'hag-client-list', 'content': 'wrong'}])]
        for key, value in changes:
            bad = deepcopy(record)
            bad['client'][key] = value
            with self.subTest(field=key, value=value), self.assertRaises(ValueError):
                gate.validate_client_stream(bad, answer, selected, 1, 'backend-parity-c128')
        for key, value in [('content', 'wrong'), ('servedModel', 'another-model'),
                           ('finishReason', 'length'), ('toolCalls', [{}]), ('toolCallsCutOff', 0),
                           ('reasoning', 'unexpected')]:
            bad = deepcopy(record)
            bad['client']['result'][key] = value
            with self.subTest(field=key), self.assertRaises(ValueError):
                gate.validate_client_stream(bad, answer, selected, 1, 'backend-parity-c128')

    def test_client_process_records_timeout_and_terminates_without_another_launch(self):
        server = Mock(root=self.root, port=12345)
        process = Mock(pid=321, returncode=-15)
        process.wait.side_effect = [gate.subprocess.TimeoutExpired('node', 35), None]
        process.poll.return_value = -15
        with patch.object(gate, 'client_identity', return_value={'kind': 'antimatter'}), \
             patch.object(gate.subprocess, 'Popen', return_value=process) as launch:
            with self.assertRaisesRegex(ValueError, 'failed or exceeded deadline'):
                gate.client_stream_complete(server, fixture(), {}, 1, self.root / 'stream.json', 'backend-parity-c128')
        launch.assert_called_once()
        process.terminate.assert_called_once()
        self.assertEqual(gate.read_json(self.root / 'timing-round-1-client-process.json'),
                         {'pid': 321, 'exit_code': -15, 'timed_out': True, 'identity': {'kind': 'antimatter'}})

    def test_client_process_binds_the_recorded_node_pid_and_removes_ambient_node_options(self):
        selected = fixture_document('backend-parity-c128')['fixtures'][0]
        answer = gate.generation(200, response(), selected, 'backend-parity-c128')
        record = client_stream_record(answer, selected)
        path = self.root / 'stream.json'
        gate.write_json(path, record)
        # Write as the spawned client, after the exclusive-output preflight.
        raw = path.read_text()
        path.unlink()
        process = Mock(pid=123, returncode=0)
        process.poll.return_value = 0
        def launched(*args, **kwargs):
            path.write_text(raw)
            return process
        with patch.dict(gate.os.environ, {'NODE_OPTIONS': '--import=bad.mjs', 'NODE_PATH': '/mutable'}), \
             patch.object(gate, 'client_identity', return_value={'kind': 'antimatter'}), \
             patch.object(gate.subprocess, 'Popen', side_effect=launched) as launch:
            with self.assertRaisesRegex(ValueError, 'process identity differs'):
                gate.client_stream_complete(Mock(root=self.root, port=12345), selected, answer, 1,
                                            path, 'backend-parity-c128')
        self.assertNotIn('NODE_OPTIONS', launch.call_args.kwargs['env'])
        self.assertNotIn('NODE_PATH', launch.call_args.kwargs['env'])

    def test_antimatter_timing_substitutes_only_three_streams_with_same_four_audits(self):
        profile = 'backend-parity-c128'
        document = fixture_document(profile)
        gate.write_json(self.root / 'answer-fixtures.json', document)
        selected = document['fixtures'][0]
        answer = gate.generation(200, response(), selected, profile)
        events = [audit(index, 38) for index in range(1, 5)]
        client = {'kind': 'antimatter', 'manifest_sha256': 'a' * 64}
        prior = {'reference_sha256': 'reference', 'exact_cpu_parity': True, 'profile': profile}
        with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(gate, 'verify_containment'), \
             patch.object(gate, 'identity', return_value={'client_identity': client}), \
             patch.object(gate, 'Server') as server_type, \
             patch.object(gate, 'complete', return_value=(answer, .1)) as complete, \
             patch.object(gate, 'client_stream_complete', side_effect=[{'round': i} for i in range(1, 4)]) as stream, \
             patch.object(gate, 'stream_complete') as python_stream, \
             patch.object(gate, 'audit_events', return_value=events), \
             patch.object(gate, 'references', return_value=({selected['id']: answer}, 'reference', 'cpu-result')), \
             patch.object(gate, 'qualified_prerequisite', return_value=(None, prior, 'answers-result')), \
             patch('builtins.print'):
            server_type.return_value.__enter__.return_value.event.side_effect = events
            gate.execute('timing', self.root)
        complete.assert_called_once()
        python_stream.assert_not_called()
        self.assertEqual([call.args[3] for call in stream.call_args_list], [1, 2, 3])
        report = gate.read_json(self.root / 'timing-report.json')
        self.assertEqual(report['client_identity'], client)
        self.assertEqual((report['requests'], report['maximum_evaluated_positions']), (4, 512))
        self.assertEqual(report['audit_events'], events)

    def test_fixture_schema_requires_final_approved_prompts_and_budget(self):
        values = gate.fixtures_from(FIXTURE_DOCUMENT)
        self.assertEqual([f['id'] for f in values], ['list_1_to_8'])
        self.assertEqual(FIXTURE_DOCUMENT['timing_fixture_id'], 'list_1_to_8')
        self.assertEqual(values[0]['messages'], [{'role': 'user', 'content':
            'List the integers from 1 to 8, separated by commas. No explanation. /no_think'}])
        self.assertEqual(values[0]['expected_semantic'], {'kind': 'integer_csv', 'values': list(range(1, 9)), 'allow_terminal_period': True})
        self.assertEqual(49 + 80 - 1, gate.CAPACITY)
        for key, bad in [('maximum_prompt_tokens', 50), ('timing_fixture_id', 'other'),
                         ('fixtures', []), ('fixtures', [fixture(), fixture()]), ('fixtures', [None])]:
            with self.assertRaises(ValueError):
                gate.fixtures_from({**FIXTURE_DOCUMENT, key: bad})

    def test_named_profiles_pin_the_default_and_three_fixture_order(self):
        self.assertEqual(FIXTURE_DOCUMENT, fixture_document())
        self.assertEqual(fixture_profile(FIXTURE_DOCUMENT), 'list-c128')
        document = fixture_document('semantic-c128')
        fixtures = gate.fixtures_from(document)
        self.assertEqual(fixture_profile(document), 'semantic-c128')
        self.assertEqual([f['id'] for f in fixtures],
                         ['list_1_to_8', 'arithmetic_7_plus_5', 'recall_mango'])
        self.assertEqual(fixtures[0], fixture())
        self.assertEqual(fixtures[1]['messages'], [{'role': 'user', 'content':
            'What is 7 + 5? Reply with just the number. /no_think'}])
        self.assertEqual(fixtures[2]['messages'], [
            {'role': 'user', 'content': 'The secret word is MANGO.'},
            {'role': 'assistant', 'content': 'Understood.'},
            {'role': 'user', 'content': 'Reply with just the secret word. /no_think'}])
        for mutation in (lambda d: d['fixtures'].reverse(),
                         lambda d: d['fixtures'].pop(),
                         lambda d: d['fixtures'][1]['expected_semantic'].update(value=13),
                         lambda d: d['fixtures'][2]['messages'][2].update(content='What word? /no_think')):
            changed = fixture_document('semantic-c128')
            mutation(changed)
            with self.assertRaises(ValueError):
                gate.fixtures_from(changed)
        parity = fixture_document('backend-parity-c128')
        self.assertEqual(parity, {**document, 'profile': 'backend-parity-c128'})
        self.assertEqual(fixture_profile(parity), 'backend-parity-c128')
        self.assertEqual(gate.fixtures_from(parity), fixtures)

    def test_json_refuses_duplicates_nonfinite_oversize_and_overwrite(self):
        for value in ('NaN', 'Infinity', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                gate.strict_json(value)
        path = self.root / 'value.json'
        gate.write_json(path, {'x': 1})
        with self.assertRaises(FileExistsError):
            gate.write_json(path, {'x': 2})
        self.assertEqual(gate.read_json(path), {'x': 1})
        path.write_bytes(b' ' * (gate.BODY_LIMIT + 1))
        with self.assertRaises(ValueError):
            gate.read_json(path)

    def test_closed_thinking_preserves_entire_text_and_accepts_absent_thinking(self):
        semantic = fixture()['expected_semantic']
        self.assertEqual(gate.closed_answer(' ' + LIST_TEXT + ' \n', semantic), (LIST_TEXT, False))
        self.assertEqual(gate.closed_answer('<think>private\nthought</think>\n' + LIST_TEXT, semantic),
                         (LIST_TEXT, True))
        raw = ' \n<think>\n</think>\n\n' + LIST_TEXT + '\n'
        answer = gate.generation(200, response(raw), fixture())
        self.assertEqual(answer['text'], raw)
        self.assertEqual(answer['final_answer'], LIST_TEXT)

    def test_thinking_never_hides_truncation_nested_stray_or_empty_final(self):
        for text in ('', ' ', '<think>', '<think>' + LIST_TEXT, '<think>' + LIST_TEXT + '</think>',
                     '</think>' + LIST_TEXT, LIST_TEXT + '<think></think>',
                     '<think><think></think>' + LIST_TEXT, '<think></think><think></think>' + LIST_TEXT,
                     '<THINK></THINK>' + LIST_TEXT, '<think x>reason</think>' + LIST_TEXT,
                     '<think></think>' + LIST_TEXT + '</think>', '<think></think>' + LIST_TEXT + '<think'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                gate.closed_answer(text, fixture()['expected_semantic'])

    def test_original_fixture_refuses_changed_prompt_sampling_roles_or_criterion(self):
        for key, value in [('id', 'other'), ('messages', []),
                           ('messages', [{'role': 'assistant', 'content': 'List 1 to 8. /no_think'}]),
                           ('messages', [{'role': 'user', 'content': 'List 1 to 8. /no_think'}]),
                           ('max_tokens', 81), ('max_tokens', 80.0), ('temperature', False),
                           ('expected_semantic', {'kind': 'integer_csv', 'values': list(range(1, 10))})]:
            changed = json.loads(json.dumps(FIXTURE_DOCUMENT))
            changed['fixtures'][0][key] = value
            with self.assertRaises(ValueError):
                gate.fixtures_from(changed)

    def test_unrecognized_or_malformed_semantic_criteria_are_refused(self):
        for semantic in (None, {}, {'kind': 'exact', 'text': LIST_TEXT},
                         {'kind': 'one_of', 'texts': [LIST_TEXT]},
                         {'kind': 'integer_csv', 'values': None},
                         {'kind': 'integer_csv', 'values': [True, 2, 3, 4, 5, 6, 7, 8]},
                         {'kind': 'integer_csv', 'values': list(range(1, 10))}):
            with self.assertRaises(ValueError):
                gate.closed_answer(LIST_TEXT, semantic)

    def test_integer_csv_requires_exact_sequence_with_optional_terminal_period(self):
        semantic = fixture('list_1_to_8')['expected_semantic']
        self.assertEqual(gate.closed_answer('1, 2,3 , 4,\n5,6,7,8', semantic)[0], '1, 2,3 , 4,\n5,6,7,8')
        self.assertEqual(gate.closed_answer(LIST_TEXT + '.', semantic), (LIST_TEXT + '.', False))
        self.assertEqual(gate.generation(200, response(LIST_TEXT + '.'), fixture())['text'], LIST_TEXT + '.')
        for text in ('1 2 3 4 5 6 7 8', '1,2,3,4,5,6,7,8..', '1,2,3,4,5,6,7',
                     '1,2,3,4,5,6,7,8,9', '1,2,3,4,5,7,6,8', '01,2,3,4,5,6,7,8',
                     'Not ' + LIST_TEXT, '**' + LIST_TEXT + '**', LIST_TEXT + '\nExtra',
                     'The integers are ' + LIST_TEXT):
            with self.assertRaises(ValueError):
                gate.closed_answer(text, semantic)

    def test_arithmetic_and_recall_accept_only_preregistered_standalone_answers(self):
        _, arithmetic, recall = fixture_document('semantic-c128')['fixtures']
        for selected, values in ((arithmetic, ['12', '12.']), (recall, ['MANGO', 'mango.', 'MaNgO'])):
            for final in values:
                raw = ' \n<think>\n\n</think>\n\n' + final + ' \n'
                answer = gate.generation(200, response(raw), selected)
                self.assertEqual(answer['text'], raw)
                self.assertEqual(answer['final_answer'], final)
                self.assertEqual(answer['evaluated_positions'], 38)
        for selected, values in (
                (arithmetic, ['13', '012', '12.0', '+12', '12..', '**12**', 'The answer is 12.',
                              '7 + 5 = 12', 'Not 12', '12\nExtra']),
                (recall, ['MANGOS', 'BANANA', 'MANGO..', '**MANGO**', 'The secret word is MANGO.',
                          'Not MANGO', 'MANGO or PEAR', 'MAN GO', 'MANGO\nExtra'])):
            for raw in values:
                with self.subTest(raw=raw), self.assertRaises(ValueError):
                    gate.generation(200, response(raw), selected)
        for semantic in ({'kind': 'integer', 'value': 12.0, 'allow_terminal_period': True},
                         {'kind': 'integer', 'value': 12, 'allow_terminal_period': 1},
                         {'kind': 'word', 'value': 'MANGO', 'case_sensitive': 0, 'allow_terminal_period': True},
                         {'kind': 'word', 'value': 'MANGO', 'case_sensitive': True, 'allow_terminal_period': True}):
            with self.assertRaises(ValueError):
                gate.closed_answer('12', semantic)

    def test_quality_grammar_separates_correctness_from_format_without_prose_inference(self):
        selected = {item['id']: item for item in fixture_document('backend-parity-c128')['fixtures']}
        cases = {
            'list_1_to_8': [(LIST_TEXT, 'pass', 'pass'), ('1, 2,3 ,4,5,6,7,8.', 'pass', 'pass'),
                ('1,2,3', 'fail', 'pass'), ('01,2,3,4,5,6,7,8', 'pass', 'fail'),
                ('00,2,3', 'fail', 'fail'), ('The numbers are ' + LIST_TEXT, 'unassessed', 'fail'),
                ('**' + LIST_TEXT + '**', 'unassessed', 'fail'), (LIST_TEXT + '..', 'unassessed', 'fail')],
            'arithmetic_7_plus_5': [('12.', 'pass', 'pass'), ('13', 'fail', 'pass'),
                ('7 + 5 = 12', 'pass', 'fail'), ('7+5=13.', 'fail', 'fail'),
                ('7\t+\n5 = 12.', 'pass', 'fail'), ('012', 'unassessed', 'fail'),
                ('7+5=012', 'unassessed', 'fail'), ('The answer is 12.', 'unassessed', 'fail'),
                ('6+6=12', 'unassessed', 'fail'), ('**12**', 'unassessed', 'fail'),
                ('Not 12', 'unassessed', 'fail'), ('12.0', 'unassessed', 'fail'),
                ('12..', 'unassessed', 'fail'), ('+12', 'unassessed', 'fail')],
            'recall_mango': [('MaNgO.', 'pass', 'pass'), ('BANANA', 'fail', 'pass'),
                ('MANGOS', 'fail', 'pass'), ('MÁNGO', 'unassessed', 'fail'),
                ('**MANGO**', 'unassessed', 'fail'), ('The word is MANGO.', 'unassessed', 'fail'),
                ('Not MANGO', 'unassessed', 'fail'), ('MANGO or PEAR', 'unassessed', 'fail'),
                ('MANGO..', 'unassessed', 'fail')],
        }
        for name, values in cases.items():
            for final, correctness, format_result in values:
                with self.subTest(fixture=name, final=final):
                    self.assertEqual(quality_assessment(' \n' + final + ' \n', selected[name]),
                                     {'correctness': correctness, 'format': format_result})

    def test_quality_handles_long_digits_without_integer_conversion_and_rejects_bad_criteria(self):
        listing, arithmetic, recall = fixture_document('backend-parity-c128')['fixtures']
        for selected, text, correctness, format_result in (
                (listing, '0' * 5000 + LIST_TEXT, 'pass', 'fail'),
                (listing, '9' * 5000 + ',2,3', 'fail', 'pass'),
                (arithmetic, '9' * 5000, 'fail', 'pass'),
                (arithmetic, '7+5=' + '9' * 5000, 'fail', 'fail'),
                (recall, 'A' * 5000, 'fail', 'pass')):
            self.assertEqual(quality_assessment(text, selected), {'correctness': correctness, 'format': format_result})
        for selected in ({}, {'id': 'unknown'}, {**arithmetic, 'expected_semantic': {}},
                         {**arithmetic, 'expected_semantic': {**arithmetic['expected_semantic'], 'value': 12.0}},
                         {**listing, 'expected_semantic': {**listing['expected_semantic'], 'values': [True, 2, 3, 4, 5, 6, 7, 8]}},
                         {**recall, 'expected_semantic': {**recall['expected_semantic'], 'case_sensitive': 0}}):
            with self.assertRaises(ValueError):
                quality_assessment('12', selected)

    def test_backend_parity_annotations_do_not_gate_wrong_or_unassessed_bounded_answers(self):
        profile = 'backend-parity-c128'
        selected = fixture_document(profile)['fixtures']
        for item, text in zip(selected, ['1,2,3', '7 + 5 = 12', 'The word is MANGO.']):
            raw = ' \n<think>reason</think>\n' + text + '\n'
            answer = gate.generation(200, response(raw), item, profile)
            self.assertEqual(answer['text'], raw)
            self.assertEqual(answer['quality'], quality_assessment(text, item))
            self.assertEqual(gate.validate_answer(answer, item, profile), answer)
            with self.assertRaises(ValueError):
                gate.generation(200, response(raw), item, 'semantic-c128')
        for text in ('13', '9' * 5000, 'arbitrary nonempty answer'):
            self.assertIn('quality', gate.generation(200, response(text), selected[1], profile))

    def test_backend_parity_still_refuses_bad_closure_eos_and_usage(self):
        selected = fixture_document('backend-parity-c128')['fixtures'][1]
        for text in ('', ' ', '<think>unfinished', '<think>only thought</think>',
                     '<think></think><think></think>13', '</think>13', '<THINK></THINK>13',
                     '13<think></think>', '13' + ' ' * 65536):
            with self.subTest(text=text[:64]), self.assertRaises(ValueError):
                gate.generation(200, response(text), selected, 'backend-parity-c128')
        for payload in (response('13', finish='length'), response('13', completion=80),
                        response('13', prompt=50), response('13', completion=0),
                        response('13', completion=True)):
            with self.assertRaises(ValueError):
                gate.generation(200, payload, selected, 'backend-parity-c128')
        answer = gate.generation(200, response('13'), selected, 'backend-parity-c128')
        server = Mock()
        server.event.return_value = audit(evaluated=answer['evaluated_positions'] - 1)
        with self.assertRaises(ValueError):
            gate.event_success(server, 1, answer)

    def test_backend_parity_quality_is_recomputed_and_cannot_cross_strict_schema(self):
        profile = 'backend-parity-c128'
        selected = fixture_document(profile)['fixtures'][1]
        answer = gate.generation(200, response('13'), selected, profile)
        for changes in ({'correctness': 'pass', 'format': 'pass'},
                        {'correctness': 'fail', 'format': 'pass', 'extra': True},
                        {'correctness': False, 'format': 'pass'}, None):
            with self.assertRaises(ValueError):
                gate.validate_answer({**answer, 'quality': changes}, selected, profile)
        with self.assertRaises(ValueError):
            gate.validate_answer(answer, selected, 'semantic-c128')
        strict = gate.generation(200, response('12'), selected, 'semantic-c128')
        with self.assertRaises(ValueError):
            gate.validate_answer(strict, selected, profile)

    def test_backend_parity_sse_accepts_identical_wrong_output_but_rejects_raw_mismatch(self):
        profile = 'backend-parity-c128'
        selected = fixture_document(profile)['fixtures'][0]
        expected = gate.generation(200, response('1,2,3'), selected, profile)
        for text in ('1,2,3', '1, 2, 3'):
            state = gate.SseResult(0)
            state.line(chunk({'role': 'assistant'}), .1)
            state.line(chunk({'content': text}), .2)
            state.line(chunk(finish='stop'), .3)
            state.line(b'data: [DONE]\n', .4)
            if text == expected['text']:
                self.assertEqual(state.finish(expected, selected, 1, profile)['content_chunks'], 1)
            else:
                with self.assertRaises(ValueError):
                    state.finish(expected, selected, 1, profile)

    def test_eos_usage_excludes_stop_token_and_admits_exact_capacity(self):
        answer = gate.generation(200, response(prompt=49, completion=79), fixture())
        self.assertEqual(answer['evaluated_positions'], 128)
        self.assertEqual(answer['usage']['completion_tokens'], 79)
        for prompt, completion in ((50, 8), (0, 8), (49, 80), (30, 0), (30, True), (30, 8.0)):
            with self.subTest(prompt=prompt, completion=completion), self.assertRaises(ValueError):
                gate.generation(200, response(prompt=prompt, completion=completion), fixture())
        for status in (201, 400, 500):
            with self.assertRaises(ValueError):
                gate.generation(status, response(), fixture())
        for finish in ('length', None, 'tool_calls'):
            with self.assertRaises(ValueError):
                gate.generation(200, response(finish=finish), fixture())
        bad = response()
        bad['usage']['total_tokens'] += 1
        with self.assertRaises(ValueError):
            gate.generation(200, bad, fixture())

    def test_native_eos_count_is_prompt_plus_completion_not_length_formula(self):
        answer = gate.generation(200, response(), fixture())
        server = Mock()
        server.event.return_value = audit(evaluated=38)
        self.assertEqual(gate.event_success(server, 1, answer), audit(evaluated=38))
        server.event.return_value = audit(evaluated=37)
        with self.assertRaises(ValueError):
            gate.event_success(server, 1, answer)

    def test_cpu_stage_records_one_request_and_one_capacity_ceiling(self):
        gate.write_json(self.root / 'answer-fixtures.json', FIXTURE_DOCUMENT)
        answer = gate.generation(200, response(), fixture())
        with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
             patch.object(gate, 'Server') as server_type, \
             patch.object(gate, 'complete', return_value=(answer, .1)) as complete, \
             patch.object(gate, 'audit_events', return_value=[]), patch('builtins.print'):
            gate.execute('cpu-reference', self.root)
        server_type.assert_called_once_with(self.root, 'cpu', 'cpu-reference', 128, 'list-c128')
        complete.assert_called_once()
        server = server_type.return_value.__enter__.return_value
        server.unload.assert_called_once()
        server.stop.assert_called_once()
        report = gate.read_json(self.root / 'cpu-reference-report.json')
        self.assertEqual(report['fixture_ids'], ['list_1_to_8'])
        self.assertEqual(report['profile'], 'list-c128')
        self.assertEqual({name: report[name] for name in policy('list-c128')}, policy('list-c128'))
        self.assertEqual(report['requests'], 1)
        self.assertEqual(report['maximum_evaluated_positions'], 128)
        self.assertEqual(report['evaluated_positions'], 38)
        self.assertEqual(report['answers'], [answer])
        self.assertEqual(report['audit_events'], [])

    def test_semantic_cpu_and_native_stages_record_three_ordered_exact_answers(self):
        document = fixture_document('semantic-c128')
        selected = document['fixtures']
        answers = [gate.generation(200, response(text), item)
                   for text, item in zip([LIST_TEXT, '12.', 'mAnGo'], selected)]
        expected = {a['fixture_id']: a for a in answers}
        events = [audit(index, answer['evaluated_positions']) for index, answer in enumerate(answers, 1)]
        for stage in ('cpu-reference', 'answers'):
            root = self.root / stage
            root.mkdir()
            gate.write_json(root / 'answer-fixtures.json', document)
            with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
                 patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
                 patch.object(gate, 'Server') as server_type, \
                 patch.object(gate, 'complete', side_effect=[(answer, .1) for answer in answers]) as complete, \
                 patch.object(gate, 'audit_events', return_value=[] if stage == 'cpu-reference' else events), \
                 patch.object(gate, 'references', return_value=(expected, 'reference', 'cpu-result')) as references, \
                 patch('builtins.print'):
                server = server_type.return_value.__enter__.return_value
                server.event.side_effect = events
                gate.execute(stage, root)
            self.assertEqual(complete.call_count, 3)
            self.assertEqual([call.args[1]['id'] for call in complete.call_args_list], list(expected))
            server.unload.assert_called_once()
            server.stop.assert_called_once()
            report = gate.read_json(root / (stage + '-report.json'))
            self.assertEqual(report['profile'], 'semantic-c128')
            self.assertEqual({name: report[name] for name in policy('semantic-c128')}, policy('semantic-c128'))
            self.assertEqual(report['fixture_ids'], list(expected))
            self.assertEqual(report['requests'], 3)
            self.assertEqual(report['maximum_evaluated_positions'], 384)
            self.assertEqual(report['evaluated_positions'], 114)
            self.assertEqual(report['answers'], answers)
            self.assertEqual(report['audit_events'], [] if stage == 'cpu-reference' else events)
            if stage == 'cpu-reference':
                references.assert_not_called()
                self.assertEqual(gate.read_json(root / 'cpu-reference.json')['profile'], 'semantic-c128')
            else:
                references.assert_called_once_with(root, {}, selected, 'semantic-c128')

    def test_backend_parity_cpu_and_native_stages_allow_identical_quality_failures(self):
        profile = 'backend-parity-c128'
        document = fixture_document(profile)
        selected = document['fixtures']
        answers = [gate.generation(200, response(text), item, profile)
                   for text, item in zip(['1,2,3', '7 + 5 = 12', 'BANANA'], selected)]
        self.assertEqual([a['quality'] for a in answers], [
            {'correctness': 'fail', 'format': 'pass'}, {'correctness': 'pass', 'format': 'fail'},
            {'correctness': 'fail', 'format': 'pass'}])
        expected = {a['fixture_id']: a for a in answers}
        events = [audit(index, answer['evaluated_positions']) for index, answer in enumerate(answers, 1)]
        for stage in ('cpu-reference', 'answers'):
            root = self.root / stage
            root.mkdir()
            gate.write_json(root / 'answer-fixtures.json', document)
            with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
                 patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
                 patch.object(gate, 'Server') as server_type, \
                 patch.object(gate, 'complete', side_effect=[(answer, .1) for answer in answers]) as complete, \
                 patch.object(gate, 'audit_events', return_value=[] if stage == 'cpu-reference' else events), \
                 patch.object(gate, 'references', return_value=(expected, 'reference', 'cpu-result')), \
                 patch('builtins.print'):
                server = server_type.return_value.__enter__.return_value
                server.event.side_effect = events
                gate.execute(stage, root)
            self.assertEqual([call.args[3] for call in complete.call_args_list], [profile] * 3)
            report = gate.read_json(root / (stage + '-report.json'))
            self.assertEqual(report['answers'], answers)
            self.assertEqual(report['profile'], profile)
            self.assertEqual({name: report[name] for name in policy(profile)}, policy(profile))
            self.assertEqual((report['requests'], report['maximum_evaluated_positions']), (3, 384))
            self.assertEqual(report['evaluated_positions'], 114)
            server.unload.assert_called_once()
            server.stop.assert_called_once()

    def test_backend_parity_native_refuses_raw_difference_even_with_identical_quality(self):
        profile = 'backend-parity-c128'
        document = fixture_document(profile)
        gate.write_json(self.root / 'answer-fixtures.json', document)
        selected = document['fixtures'][0]
        cpu = gate.generation(200, response('1,2,3'), selected, profile)
        native = gate.generation(200, response('1, 2, 3'), selected, profile)
        self.assertEqual(cpu['quality'], native['quality'])
        with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
             patch.object(gate, 'Server') as server_type, \
             patch.object(gate, 'complete', return_value=(native, .1)), \
             patch.object(gate, 'references', return_value=({selected['id']: cpu}, 'reference', 'cpu-result')):
            with self.assertRaisesRegex(ValueError, 'differs from CPU'):
                gate.execute('answers', self.root)
        server_type.return_value.__enter__.return_value.event.assert_not_called()
        server_type.return_value.__exit__.assert_called_once()
        self.assertFalse((self.root / 'answers-report.json').exists())

    def test_semantic_profile_timing_keeps_one_list_warmup_and_three_list_samples(self):
        document = fixture_document('semantic-c128')
        gate.write_json(self.root / 'answer-fixtures.json', document)
        selected = document['fixtures']
        answers = [gate.generation(200, response(text), item)
                   for text, item in zip([LIST_TEXT, '12', 'MANGO'], selected)]
        expected = {a['fixture_id']: a for a in answers}
        events = [audit(index, 38) for index in range(1, 5)]
        prior = {'reference_sha256': 'reference', 'exact_cpu_parity': True, 'profile': 'semantic-c128'}
        with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
             patch.object(gate, 'Server') as server_type, \
             patch.object(gate, 'complete', return_value=(answers[0], .1)) as complete, \
             patch.object(gate, 'stream_complete', side_effect=[{'round': i} for i in range(1, 4)]) as stream, \
             patch.object(gate, 'audit_events', return_value=events), \
             patch.object(gate, 'references', return_value=(expected, 'reference', 'cpu-result')), \
             patch.object(gate, 'qualified_prerequisite', return_value=(None, prior, 'answers-result')), \
             patch('builtins.print'):
            server = server_type.return_value.__enter__.return_value
            server.event.side_effect = events
            gate.execute('timing', self.root)
        complete.assert_called_once()
        self.assertEqual(complete.call_args.args[1]['id'], 'list_1_to_8')
        self.assertEqual([call.args[1]['id'] for call in stream.call_args_list], ['list_1_to_8'] * 3)
        self.assertEqual([call.args[3] for call in stream.call_args_list], [1, 2, 3])
        report = gate.read_json(self.root / 'timing-report.json')
        self.assertEqual(report['profile'], 'semantic-c128')
        self.assertEqual(report['fixture_ids'], ['list_1_to_8'])
        self.assertEqual((report['requests'], report['maximum_evaluated_positions']), (4, 512))
        self.assertEqual(report['evaluated_positions'], 152)
        self.assertEqual(report['audit_events'], events)

    def test_backend_parity_timing_remains_four_list_only_requests_with_diagnostic_quality(self):
        profile = 'backend-parity-c128'
        document = fixture_document(profile)
        gate.write_json(self.root / 'answer-fixtures.json', document)
        selected = document['fixtures'][0]
        answer = gate.generation(200, response('1,2,3'), selected, profile)
        events = [audit(index, 38) for index in range(1, 5)]
        prior = {'reference_sha256': 'reference', 'exact_cpu_parity': True, 'profile': profile, **policy(profile)}
        with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
             patch.object(gate, 'Server') as server_type, \
             patch.object(gate, 'complete', return_value=(answer, .1)) as complete, \
             patch.object(gate, 'stream_complete', side_effect=[{'round': i} for i in range(1, 4)]) as stream, \
             patch.object(gate, 'audit_events', return_value=events), \
             patch.object(gate, 'references', return_value=({selected['id']: answer}, 'reference', 'cpu-result')), \
             patch.object(gate, 'qualified_prerequisite', return_value=(None, prior, 'answers-result')), \
             patch('builtins.print'):
            server_type.return_value.__enter__.return_value.event.side_effect = events
            gate.execute('timing', self.root)
        self.assertEqual(complete.call_args.args[3], profile)
        self.assertEqual([call.args[1]['id'] for call in stream.call_args_list], ['list_1_to_8'] * 3)
        self.assertEqual([call.args[5] for call in stream.call_args_list], [profile] * 3)
        report = gate.read_json(self.root / 'timing-report.json')
        self.assertEqual((report['requests'], report['maximum_evaluated_positions']), (4, 512))
        self.assertEqual(report['fixture_ids'], ['list_1_to_8'])
        self.assertEqual(report['warmup']['generation']['quality']['correctness'], 'fail')
        self.assertEqual({name: report[name] for name in policy(profile)}, policy(profile))

    def test_backend_parity_complete_preserves_raw_http_and_attaches_non_gating_quality(self):
        profile = 'backend-parity-c128'
        selected = fixture_document(profile)['fixtures'][1]
        server, connection, incoming = Mock(), Mock(), Mock()
        incoming.status = 200
        incoming.getheader.return_value = 'application/json'
        raw = json.dumps(response('7 + 5 = 12')).encode()
        incoming.read.return_value = raw
        server.open.return_value = connection, incoming
        path = self.root / 'response.json'
        answer, _ = gate.complete(server, selected, path, profile)
        self.assertEqual(answer['quality'], {'correctness': 'pass', 'format': 'fail'})
        self.assertEqual(base64.b64decode(gate.read_json(path)['raw_body_base64']), raw)
        connection.close.assert_called_once()
        incoming.close.assert_called_once()

    def test_answer_artifact_is_revalidated_not_trusted(self):
        answer = gate.generation(200, response(), fixture())
        self.assertEqual(gate.validate_answer(answer, fixture()), answer)
        for key, bad in [('final_answer', LIST_TEXT), ('thinking_present', 1),
                         ('evaluated_positions', 37), ('fixture_id', 'other')]:
            changed = {**answer, key: bad}
            if key == 'final_answer':
                changed['text'] = '<think>' + LIST_TEXT
            with self.assertRaises(ValueError):
                gate.validate_answer(changed, fixture())

    def test_failed_cpu_response_is_saved_before_semantic_check(self):
        server = Mock()
        connection, incoming = Mock(), Mock()
        incoming.status = 200
        incoming.getheader.return_value = 'application/json'
        incoming.read.return_value = json.dumps(response('<think>not closed', finish='length')).encode()
        server.open.return_value = connection, incoming
        path = self.root / 'response.json'
        with self.assertRaises(ValueError):
            gate.complete(server, fixture(), path)
        saved = gate.read_json(path)
        self.assertEqual(saved['response']['choices'][0]['finish_reason'], 'length')
        self.assertEqual(saved['response']['choices'][0]['message']['content'], '<think>not closed')
        self.assertGreater(saved['elapsed_seconds'], 0)

    def test_invalid_json_content_type_and_truncated_body_still_preserve_raw_evidence(self):
        for index, (raw, content_type) in enumerate([(b'{bad', 'application/json'),
                (json.dumps(response()).encode(), 'text/plain'),
                (b'x' * (gate.BODY_LIMIT + 1), 'application/json'), (b'\xff', 'application/json')]):
            server, connection, incoming = Mock(), Mock(), Mock()
            incoming.status = 200
            incoming.getheader.return_value = content_type
            incoming.read.return_value = raw
            server.open.return_value = connection, incoming
            path = self.root / f'bad-{index}.json'
            with self.assertRaises(ValueError):
                gate.complete(server, fixture(), path)
            saved = json.loads(path.read_text())
            self.assertEqual(base64.b64decode(saved['raw_body_base64']), raw[:gate.BODY_LIMIT])
            self.assertEqual(saved['body_truncated'], len(raw) > gate.BODY_LIMIT)
            connection.close.assert_called_once()
            incoming.close.assert_called_once()

    def test_sse_uses_first_nonempty_content_and_api_tokens_not_chunks(self):
        expected = gate.generation(200, response(), fixture())
        state = gate.SseResult(100.0)
        for line, elapsed in [(chunk({'role': 'assistant'}), .1), (chunk({'content': ''}), .2),
                              (chunk({'content': '<think>\n\n</think>\n\n'}), .3),
                              (chunk({'content': LIST_TEXT}), .6), (chunk(finish='stop'), .8),
                              (b'data: [DONE]\n', 1.0)]:
            state.line(line, 100.0 + elapsed)
        metrics = state.finish(expected, fixture(), 0)
        self.assertAlmostEqual(metrics['ttfc_seconds'], .3)
        self.assertAlmostEqual(metrics['last_content_seconds'], .6)
        self.assertEqual(metrics['done_seconds'], 1.0)
        self.assertEqual(metrics['content_chunks'], 2)
        self.assertEqual(metrics['completion_tokens'], 8)
        self.assertEqual(metrics['completion_tokens_per_second'], 8.0)
        self.assertEqual(metrics['usage_source'], 'matched_nonstream_warmup_api')

    def test_sse_refuses_mixed_identity_early_done_error_and_content_after_finish(self):
        for raw in (b'data: [DONE]\n', b'event: error\n', chunk({'content': LIST_TEXT}), chunk(finish='stop')):
            with self.assertRaises(ValueError):
                gate.SseResult(0).line(raw, .1)
        state = gate.SseResult(0)
        state.line(chunk({'role': 'assistant'}), .1)
        with self.assertRaises(ValueError):
            state.line(chunk({'content': LIST_TEXT}, identifier='different'), .2)
        state.line(chunk({'content': LIST_TEXT}), .3)
        state.line(chunk(finish='stop'), .4)
        with self.assertRaises(ValueError):
            state.line(chunk({'content': 'extra'}), .5)
        state.line(b'data: [DONE]\n', .6)
        with self.assertRaises(ValueError):
            state.line(b'data: [DONE]\n', .7)

    def test_sse_refuses_nonmonotonic_time_nonfinite_time_and_wrong_text(self):
        state = gate.SseResult(10)
        for timestamp in (9.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                state.line(b'\n', timestamp)
        state.line(chunk({'role': 'assistant'}), 10.1)
        state.line(chunk({'content': '1,2,3'}), 10.2)
        state.line(chunk(finish='stop'), 10.3)
        state.line(b'data: [DONE]\n', 10.4)
        with self.assertRaises(ValueError):
            state.finish(gate.generation(200, response(LIST_TEXT), fixture()), fixture(), 0)

    def test_failed_sse_preserves_raw_event_arrival_times_and_closes_transport(self):
        server, connection, response_object = Mock(), Mock(), Mock()
        response_object.status = 200
        response_object.getheader.return_value = 'text/event-stream'
        response_object.readline.side_effect = [chunk({'role': 'assistant'}), b'event: error\n']
        server.open.return_value = connection, response_object
        path = self.root / 'sse.json'
        with self.assertRaises(ValueError):
            gate.stream_complete(server, fixture(), gate.generation(200, response(), fixture()), 0, path)
        saved = gate.read_json(path)
        self.assertEqual(len(saved['events']), 2)
        self.assertEqual(saved['events'][1]['line'], 'event: error\n')
        self.assertGreaterEqual(saved['events'][1]['received_seconds'], saved['events'][0]['received_seconds'])
        connection.close.assert_called_once()
        response_object.close.assert_called_once()

    def test_audit_refuses_unhealthy_reordered_or_excess_position_events(self):
        path = self.root / 'server.log'
        for changes in ({'request_sequence': 2}, {'evaluated_positions': 129}, {'outcome': 'failed'},
                        {'reset_ok': False}, {'poisoned': True}):
            path.write_text(gate.AUDIT_PREFIX + json.dumps({**audit(), **changes}) + '\n')
            with self.assertRaises(ValueError):
                gate.audit_events(path)
        path.write_text(gate.AUDIT_PREFIX + json.dumps(audit()) + '\n')
        self.assertEqual(gate.audit_events(path), [audit()])

    def test_snapshot_identity_binds_configs_without_opening_input_binary_or_checkpoint(self):
        inputs, manifest = snapshot_inputs(self.root)
        identities = gate.identity(self.root)
        self.assertEqual(identities['binary_sha256'], manifest['binaries']['beholdd'])
        self.assertEqual(identities['source_manifest_sha256'], manifest['source_manifest_sha256'])
        self.assertEqual(identities['checkpoint_sha256'], manifest['model_sha256'])
        self.assertEqual(identities['gate_files'], manifest['gate_files'])
        self.assertTrue(all(not Path(inputs[name]).exists() for name in ('repo', 'model', 'device', 'binary_input')))

    def test_snapshot_identity_refuses_profile_metadata_not_matching_its_fixtures(self):
        _, manifest = snapshot_inputs(self.root)
        for profile in ('semantic-c128', 'backend-parity-c128', None, True):
            (self.root / 'expected-manifest.json').write_text(json.dumps({**manifest, 'profile': profile}))
            with self.assertRaises(ValueError):
                gate.identity(self.root)

    def test_snapshot_identity_refuses_changed_frozen_worker_config_binary_or_source_manifest(self):
        for index, name in enumerate(('beholdd', 'build-source-sha256.json', 'answer-gate.py', 'profiles.py',
                                      'answer_quality.py', 'answer-fixtures.json', 'run-config.json', 'cpu-config.json', 'hag-config.json')):
            root = self.root / str(index)
            snapshot_inputs(root)
            with (root / name).open('ab') as output:
                output.write(b'\n')
            with self.subTest(name=name), self.assertRaises(ValueError):
                gate.identity(root)

    def test_snapshot_identity_requires_declared_model_device_backend_and_fixed_limits(self):
        changes = [
            ('cpu', lambda m: m.update(path='/different-model.gguf')),
            ('cpu', lambda m: m.update(execution_backend='hag')),
            ('cpu', lambda m: m.update(weight_mode='resident')),
            ('cpu', lambda m: m.update(fit_policy='best_effort')),
            ('cpu', lambda m: m.update(memory_budget_bytes=2 * 1024 ** 3)),
            ('cpu', lambda m: m.update(runtime_headroom_bytes=0)),
            ('cpu', lambda m: m.update(hag={})),
            ('hag', lambda m: m['hag'].update(device='/different-device')),
            ('hag', lambda m: m['hag'].update(memory_budget_bytes=1024 ** 3)),
            ('hag', lambda m: m.update(memory_budget_bytes=float(2 * 1024 ** 3))),
        ]
        for index, (backend, change) in enumerate(changes):
            root = self.root / str(index)
            snapshot_inputs(root)
            name = backend + '-config.json'
            config = gate.read_json(root / name)
            change(config['models'][0])
            (root / name).write_text(json.dumps(config))
            rehash_gate_file(root, name)
            with self.subTest(backend=backend, change=index), self.assertRaises(ValueError):
                gate.identity(root)

    def test_snapshot_identity_requires_canonical_complete_run_configuration(self):
        for index, change in enumerate((lambda c: c.update(schema=True),
                lambda c: c.update(repo='relative'), lambda c: c.update(model='/a/../b'),
                lambda c: c.update(device='/a//b'), lambda c: c.pop('binary_input'),
                lambda c: c.update(extra='unexpected'))):
            root = self.root / str(index)
            inputs, _ = snapshot_inputs(root)
            change(inputs)
            (root / 'run-config.json').write_text(json.dumps(inputs))
            rehash_gate_file(root, 'run-config.json')
            with self.subTest(change=index), self.assertRaises(ValueError):
                gate.identity(root)

    def test_prerequisite_binds_qualified_result_to_identical_report_and_freeze(self):
        identity = {'binary_sha256': 'b', 'source_manifest_sha256': 's', 'checkpoint_sha256': 'c', 'gate_files': {}}
        report = {'reference_sha256': 'r'}
        result = {'mode': 'cpu-reference', 'qualified': True, 'failures': [], 'exit_code': 0,
                  **identity, 'probe': report}
        gate.write_json(self.root / 'cpu-reference-result.json', result)
        gate.write_json(self.root / 'cpu-reference-report.json', report)
        self.assertEqual(gate.qualified_prerequisite(self.root, 'cpu-reference', identity)[1], report)
        for key, bad in [('qualified', False), ('binary_sha256', 'other'), ('source_manifest_sha256', 'other'),
                         ('checkpoint_sha256', 'other'), ('gate_files', {'unexpected': 'x'}), ('probe', {})]:
            (self.root / 'cpu-reference-result.json').write_text(json.dumps({**result, key: bad}))
            with self.assertRaises(ValueError):
                gate.qualified_prerequisite(self.root, 'cpu-reference', identity)

    def test_backend_parity_prerequisite_cannot_cross_profile_or_policy_metadata(self):
        profile = 'backend-parity-c128'
        identities = {'binary_sha256': 'b', 'source_manifest_sha256': 's', 'checkpoint_sha256': 'c', 'gate_files': {}}
        report = {'profile': profile, **policy(profile)}
        result = {'mode': 'cpu-reference', 'profile': profile, 'qualified': True, 'failures': [],
                  'exit_code': 0, **identities, 'probe': report}
        gate.write_json(self.root / 'cpu-reference-report.json', report)
        gate.write_json(self.root / 'cpu-reference-result.json', result)
        self.assertEqual(gate.qualified_prerequisite(self.root, 'cpu-reference', identities, profile)[1], report)
        for changed in ({'profile': 'semantic-c128'}, {'qualification_kind': 'semantic_and_backend_parity'},
                        {'require_semantic_match': True}, {'quality_is_diagnostic': False},
                        {'require_semantic_match': 0}, {'quality_is_diagnostic': 1}):
            bad_report = {**report, **changed}
            (self.root / 'cpu-reference-report.json').write_text(json.dumps(bad_report))
            (self.root / 'cpu-reference-result.json').write_text(json.dumps({**result, 'probe': bad_report}))
            with self.subTest(change=changed), self.assertRaises(ValueError):
                gate.qualified_prerequisite(self.root, 'cpu-reference', identities, profile)
        (self.root / 'cpu-reference-report.json').write_text(json.dumps(report))
        (self.root / 'cpu-reference-result.json').write_text(json.dumps({**result, 'profile': 'semantic-c128'}))
        with self.assertRaises(ValueError):
            gate.qualified_prerequisite(self.root, 'cpu-reference', identities, profile)

    def test_cpu_reference_profile_is_bound_in_both_artifact_and_report(self):
        selected = fixture_document('semantic-c128')['fixtures']
        answers = [gate.generation(200, response(text), item)
                   for text, item in zip([LIST_TEXT, '12', 'MANGO'], selected)]
        identities = {'binary_sha256': 'b', 'source_manifest_sha256': 's', 'checkpoint_sha256': 'c', 'gate_files': {}}
        for index, (artifact_profile, report_profile) in enumerate((('semantic-c128', 'semantic-c128'),
                ('list-c128', 'semantic-c128'), ('semantic-c128', 'list-c128'))):
            root = self.root / str(index)
            root.mkdir()
            artifact = {'schema': 1, 'execution': 'cpu-reference', 'capacity': 128, 'identities': identities,
                        'profile': artifact_profile, 'answers': answers, 'cpu_exit_code': 0, 'final_unloaded': True}
            gate.write_json(root / 'cpu-reference.json', artifact)
            report = {'profile': report_profile, **policy(report_profile), 'reference_sha256': gate.digest(root / 'cpu-reference.json'),
                      'answers': answers}
            gate.write_json(root / 'cpu-reference-report.json', report)
            gate.write_json(root / 'cpu-reference-result.json', {'mode': 'cpu-reference', 'qualified': True,
                            'profile': report_profile, 'failures': [], 'exit_code': 0, **identities, 'probe': report})
            if index == 0:
                self.assertEqual(list(gate.references(root, identities, selected, 'semantic-c128')[0]),
                                 [a['fixture_id'] for a in answers])
            else:
                with self.assertRaises(ValueError):
                    gate.references(root, identities, selected, 'semantic-c128')

    def test_containment_requires_no_swap_capped_cpu_and_no_new_privileges(self):
        proc, cgroups = self.root / 'proc', self.root / 'cgroups'
        (proc / 'self').mkdir(parents=True)
        cg = cgroups / 'test'
        cg.mkdir(parents=True)
        (proc / 'self/cgroup').write_text('0::/test\n')
        (proc / 'self/status').write_text('NoNewPrivs:\t1\n')
        expected = {'memory.max': str(2 * 1024**3), 'memory.swap.max': '0', 'pids.max': '64', 'cpu.max': '75000 100000'}
        for name, value in expected.items():
            (cg / name).write_text(value)
        gate.verify_containment(proc, cgroups)
        for name in expected:
            (cg / name).write_text('max')
            with self.assertRaises(ValueError):
                gate.verify_containment(proc, cgroups)
            (cg / name).write_text(expected[name])

    def test_cli_spawn_uses_explicit_backend_and_only_native_gets_audit(self):
        for backend in ('cpu', 'hag'):
            server = gate.Server(self.root, backend, backend + '-stage')
            process = Mock()
            process.poll.return_value = None
            def spawned(*args, **kwargs):
                line = f'beholdd v2 listening on http://127.0.0.1:{server.port}\n'
                kwargs['stdout'].write(line)
                kwargs['stdout'].flush()
                return process
            with patch.object(gate.subprocess, 'Popen', side_effect=spawned) as launch, \
                 patch.object(server, 'json', return_value=(200, {'status': 'ok'})):
                server.__enter__()
            args, kwargs = launch.call_args
            self.assertEqual(args[0][0], str(self.root / 'beholdd'))
            self.assertIn(str(self.root / (backend + '-config.json')), args[0])
            self.assertEqual(kwargs['env']['HAG_TRACE_SUBMISSION'], '0')
            self.assertEqual(kwargs['env'].get('HAG_SERVING_AUDIT'), '1' if backend == 'hag' else None)
            server.log.close()


class GroundedGateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.profile = gate.GROUNDED_PROFILE
        self.document = fixture_document(self.profile)
        self.fixture = self.document['fixtures'][0]
        self.payload = response('<think>\n\n</think>\n\nready.', prompt=93, completion=5)
        self.answer = gate.generation(200, self.payload, self.fixture, self.profile)

    def chat_record(self):
        return {'status': 200, 'content_type': 'application/json',
                'clock': 'monotonic_elapsed_from_request_start', 'events': [],
                'raw_body_base64': base64.b64encode(json.dumps(self.payload).encode()).decode(),
                'elapsed_seconds': 1.5,
                'client': {'kind': 'antimatter', 'pid': 9876, 'mode': 'chat',
                           'max_evaluated_tokens': 512, 'request_count': 1, 'attempt_count': 1,
                           'request': gate.request_body(self.fixture), 'result': self.answer['text'],
                           'callbacks': [], 'response_complete': True, 'error': None,
                           'preflight': {'capacity': 512, 'requested_output_tokens': 512,
                                         'fetch_attempts': 0, 'track_calls': 0, 'prepare_calls': 0,
                                         'error_code': 'context_length_exceeded'},
                           'hooks': {'track_calls': 1, 'prepare_calls': 1, 'track_end_calls': 1}}}

    def test_exact_prompt_eos_budget_and_diagnostic_quality(self):
        self.assertEqual(self.answer['evaluated_positions'], 98)
        self.assertEqual(self.answer['quality'], {'correctness': 'pass', 'format': 'pass'})
        incorrect = gate.generation(200, response('banana', 93, 5), self.fixture, self.profile)
        self.assertEqual(incorrect['quality'], {'correctness': 'fail', 'format': 'pass'})
        self.assertEqual(gate.generation(200, response('ready', 93, 127), self.fixture,
                                        self.profile)['evaluated_positions'], 220)
        for prompt, completion in ((92, 5), (94, 5), (93, 128), (93, 0), (True, 5)):
            with self.subTest(prompt=prompt, completion=completion), self.assertRaises(ValueError):
                gate.generation(200, response('ready', prompt, completion), self.fixture, self.profile)
        with self.assertRaises(ValueError):
            gate.generation(200, self.payload, fixture(), 'list-c128')

    def test_genuine_chat_result_binds_raw_protocol_usage_and_local_preflight(self):
        record = self.chat_record()
        self.assertEqual(gate.validate_grounded_chat(record, self.fixture, self.profile), (self.answer, 1.5))
        mutations = [lambda r: r['client'].update(result='forged'),
                     lambda r: r['client'].update(max_evaluated_tokens=128),
                     lambda r: r['client']['preflight'].update(fetch_attempts=1),
                     lambda r: r['client']['preflight'].update(track_calls=False),
                     lambda r: r['client']['hooks'].update(prepare_calls=0),
                     lambda r: r['client'].update(attempt_count=2),
                     lambda r: r.update(elapsed_seconds=60.1),
                     lambda r: r.update(raw_body_base64=base64.b64encode(
                         json.dumps(response('ready', 92, 5)).encode()).decode()),
                     lambda r: r['client']['request'].update(max_tokens=129),
                     lambda r: r.update(events=[{'line': 'unexpected'}])]
        for mutation in mutations:
            bad = deepcopy(record)
            mutation(bad)
            with self.subTest(record=bad), self.assertRaises(ValueError):
                gate.validate_grounded_chat(bad, self.fixture, self.profile)

    def test_stream_uses_same_cap_preflight_and_complete_raw_eos(self):
        record = client_stream_record(self.answer, self.fixture)
        record['client'].update({key: value for key, value in self.chat_record()['client'].items()
                                 if key in ('preflight', 'hooks', 'max_evaluated_tokens')})
        record['client']['mode'] = 'stream'
        record['client']['callbacks'][0]['id'] = 'hag-client-grounded'
        self.assertEqual(gate.validate_client_stream(record, self.answer, self.fixture, 1,
                                                    self.profile)['completion_tokens'], 5)
        record['events'].pop()
        with self.assertRaises(ValueError):
            gate.validate_client_stream(record, self.answer, self.fixture, 1, self.profile)

    def test_refusal_retains_raw_400_and_requires_unchanged_residency_and_audit(self):
        for loaded in (False, True):
            root = self.root / str(loaded)
            root.mkdir()
            server, connection, reply = Mock(), Mock(), Mock(status=400)
            reply.getheader.return_value = 'application/json'
            payload = {'error': {'type': 'invalid_request_error', 'param': 'max_tokens',
                                  'code': 'context_length_exceeded', 'message': 'too large'}}
            reply.read.return_value = json.dumps(payload).encode()
            server.open.return_value = connection, reply
            events = [audit(1, 98)] if loaded else []
            with patch.object(gate, 'audit_events', return_value=events):
                summary = gate.capacity_refusal(server, self.fixture, loaded, loaded, len(events),
                                                root / 'refusal.json')
            self.assertEqual(summary['status'], 400)
            self.assertEqual(server.open.call_args.args[2]['max_tokens'], 421)
            self.assertEqual([call.args for call in server.model.call_args_list], [(loaded,), (loaded,)])
            retained = gate.read_json(root / 'refusal.json')
            self.assertEqual(json.loads(base64.b64decode(retained['raw_body_base64'])), payload)
            self.assertTrue(gate.read_json(root / 'refusal-post.json')['unchanged_audit'])
        server, connection, reply = Mock(), Mock(), Mock(status=400)
        reply.getheader.return_value = 'text/event-stream'
        reply.read.return_value = b'{}'
        server.open.return_value = connection, reply
        with patch.object(gate, 'audit_events', return_value=[]), self.assertRaises(ValueError):
            gate.capacity_refusal(server, self.fixture, True, False, 0, self.root / 'bad.json')
        self.assertTrue((self.root / 'bad.json').exists())
        self.assertFalse((self.root / 'bad-post.json').exists())

    def test_native_stages_use_client_chat_and_one_warm_stream_without_extra_generation(self):
        for stage, count in (('answers', 1), ('timing', 2)):
            root = self.root / stage
            root.mkdir()
            gate.write_json(root / 'answer-fixtures.json', self.document)
            events = [audit(index, 98) for index in range(1, count + 1)]
            prior = {'reference_sha256': 'reference', 'exact_cpu_parity': True, 'profile': self.profile}
            with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
                 patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={'client_identity': {}}), \
                 patch.object(gate, 'Server') as server_type, patch.object(gate, 'complete') as python_chat, \
                 patch.object(gate, 'grounded_complete', return_value=(self.answer, 1.5)) as chat, \
                 patch.object(gate, 'grounded_stream_complete', return_value={'round': 1}) as stream, \
                 patch.object(gate, 'capacity_refusal', side_effect=[{'phase': 'cold'}, {'phase': 'warm'}]) as refusal, \
                 patch.object(gate, 'audit_events', return_value=events), \
                 patch.object(gate, 'references', return_value=({self.fixture['id']: self.answer}, 'reference', 'cpu-result')), \
                 patch.object(gate, 'qualified_prerequisite', return_value=(None, prior, 'answers-result')), \
                 patch('builtins.print'):
                server_type.return_value.__enter__.return_value.event.side_effect = events
                gate.execute(stage, root)
            python_chat.assert_not_called()
            chat.assert_called_once()
            self.assertEqual(stream.call_count, count - 1)
            self.assertEqual(refusal.call_count, 2 if stage == 'answers' else 0)
            report = gate.read_json(root / (stage + '-report.json'))
            self.assertEqual((report['capacity'], report['requests'], report['evaluated_positions']),
                             (512, count, 98 * count))
            self.assertEqual(report['maximum_evaluated_positions'], 220 * count)
            self.assertFalse(report['live_cancellation_qualified'])
            self.assertTrue(report['quality_is_diagnostic'])
            self.assertEqual(report['model_free_client_refusals'], count)




class ProjectGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.profile = gate.PROJECT_PROFILE
        self.document = fixture_document(self.profile)
        self.fixture = self.document['fixtures'][0]
        self.payload = response('<think>\n\n</think>\n\nready.', prompt=467, completion=5)
        self.answer = gate.generation(200,self.payload,self.fixture,self.profile)

    def chat_record(self):
        record = GroundedGateTests.chat_record(self)
        record['client']['max_evaluated_tokens'] = 2048
        record['client']['preflight'].update(capacity=2048,requested_output_tokens=1583)
        record['elapsed_seconds'] = 200
        return record

    def test_project_raw_chat_and_stream_use_full_profile_and_240_second_bound(self):
        record = self.chat_record()
        self.assertEqual(gate.validate_grounded_chat(record,self.fixture,self.profile),(self.answer,200))
        stream = client_stream_record(self.answer,self.fixture)
        stream['client'].update({key:record['client'][key] for key in ('preflight','hooks','max_evaluated_tokens')})
        stream['client']['mode'] = 'stream'
        stream['client']['callbacks'][0]['id'] = 'hag-client-grounded'
        for event in stream['events']: event['received_seconds'] += 200
        self.assertGreater(gate.validate_client_stream(stream,self.answer,self.fixture,1,self.profile)['done_seconds'],200)
        stream['events'][-1]['received_seconds'] = 240.01
        with self.assertRaises(ValueError): gate.validate_client_stream(stream,self.answer,self.fixture,1,self.profile)
        for key,value in [('max_evaluated_tokens',512),('attempt_count',2)]:
            changed = deepcopy(record); changed['client'][key] = value
            with self.assertRaises(ValueError): gate.validate_grounded_chat(changed,self.fixture,self.profile)
        changed = deepcopy(record); changed['client']['preflight']['requested_output_tokens'] = 1219
        with self.assertRaises(ValueError): gate.validate_grounded_chat(changed,self.fixture,self.profile)

    def test_project_exact_prompt_eos_budget_and_cpu_request_deadline(self):
        self.assertEqual(self.answer['evaluated_positions'],472)
        self.assertEqual(gate.generation(200,response('ready',467,127),self.fixture,self.profile)['evaluated_positions'],594)
        for prompt,count in ((466,5),(468,5),(467,128)):
            with self.assertRaises(ValueError): gate.generation(200,response('ready',prompt,count),self.fixture,self.profile)
        self.assertEqual(gate.Server(self.root,'cpu','cpu-reference',2048,self.profile).request_seconds,600)
        self.assertEqual(gate.Server(self.root,'cpu','cpu-reference').request_seconds,170)
        self.assertEqual(gate.Server(self.root,'hag','answers',2048,self.profile).request_seconds,170)
        with self.assertRaises(ValueError): gate.validate_grounded_chat(self.chat_record(),self.fixture,gate.GROUNDED_PROFILE)

    def test_server_strips_untrusted_attestation_except_verified_project_native(self):
        path = '/user.slice/hag-answers-answers-20261007-000000-000001.service'
        for backend,profile in [('cpu',self.profile),('hag','list-c128'),('hag',self.profile)]:
            folder = self.root/(backend+profile); folder.mkdir()
            server = gate.Server(folder,backend,'answers',2048,profile)
            process = Mock(); process.poll.return_value = None
            def spawned(*args,**kwargs):
                kwargs['stdout'].write(f'beholdd v2 listening on http://127.0.0.1:{server.port}\n')
                kwargs['stdout'].flush()
                return process
            with patch.dict(gate.os.environ,{'BEHOLDD_HAG_PRIVATE_CGROUP':path}), \
                 patch.object(gate,'verify_containment',return_value=path), \
                 patch.object(gate.subprocess,'Popen',side_effect=spawned) as launch, \
                 patch.object(server,'json',return_value=(200,{'status':'ok'})):
                server.__enter__()
            self.assertEqual(launch.call_args.kwargs['env'].get('BEHOLDD_HAG_PRIVATE_CGROUP'),
                             path if backend == 'hag' and profile == self.profile else None)
            server.log.close()
        server = gate.Server(self.root,'hag','answers',2048,self.profile)
        with patch.dict(gate.os.environ,{'BEHOLDD_HAG_PRIVATE_CGROUP':'/forged'}), \
             patch.object(gate,'verify_containment',return_value=path), \
             patch.object(gate.subprocess,'Popen') as launch, self.assertRaises(ValueError):
            server.__enter__()
        launch.assert_not_called()
        self.assertIsNone(server.log)


class HistoryGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.profile = gate.HISTORY_PROFILE
        self.document = fixture_document(self.profile)
        self.fixture = self.document['fixtures'][0]
        self.payload = response('<think>\n\n</think>\n\nready.', prompt=495, completion=5)
        self.answer = gate.generation(200, self.payload, self.fixture, self.profile)

    def chat_record(self):
        record = GroundedGateTests.chat_record(self)
        record['client']['max_evaluated_tokens'] = 2048
        record['client']['preflight'].update(capacity=2048, requested_output_tokens=1555)
        record['elapsed_seconds'] = 200
        return record

    def test_seeded_history_usage_and_diagnostic_quality_do_not_claim_recall(self):
        messages = self.fixture['messages']
        self.assertEqual([message['role'] for message in messages], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(messages[2]['content'], '<think>\n\n</think>\n\nready.')
        self.assertEqual(gate.request_body(self.fixture)['messages'], messages)
        self.assertEqual(self.answer['evaluated_positions'], 500)
        self.assertEqual(gate.generation(200, response('ready', 495, 127), self.fixture,
                                        self.profile)['evaluated_positions'], 622)
        for prompt, completion in ((467, 5), (494, 5), (496, 5), (495, 128), (495, 0), (True, 5)):
            with self.subTest(prompt=prompt, completion=completion), self.assertRaises(ValueError):
                gate.generation(200, response('ready', prompt, completion), self.fixture, self.profile)
        for text, quality in [('banana', {'correctness': 'fail', 'format': 'pass'}),
                              ('I am ready.', {'correctness': 'unassessed', 'format': 'fail'})]:
            answer = gate.generation(200, response(text, 495, 5), self.fixture, self.profile)
            self.assertEqual(answer['quality'], quality)
            gate.validate_answer(answer, self.fixture, self.profile)
            answer['quality'] = {'correctness': 'pass', 'format': 'pass'}
            with self.assertRaises(ValueError): gate.validate_answer(answer, self.fixture, self.profile)

    def test_history_chat_and_sse_bind_all_messages_and_240_second_deadline(self):
        record = self.chat_record()
        self.assertEqual(gate.validate_grounded_chat(record, self.fixture, self.profile), (self.answer, 200))
        mutations = [lambda r: r['client']['request']['messages'][2].update(content='ready.'),
                     lambda r: r['client']['request']['messages'].pop(2),
                     lambda r: r['client']['preflight'].update(requested_output_tokens=1583),
                     lambda r: r['client']['preflight'].update(requested_output_tokens=1188),
                     lambda r: r['client']['preflight'].update(fetch_attempts=1),
                     lambda r: r.update(elapsed_seconds=240.01)]
        for mutation in mutations:
            changed = deepcopy(record)
            mutation(changed)
            with self.subTest(record=changed), self.assertRaises(ValueError):
                gate.validate_grounded_chat(changed, self.fixture, self.profile)
        stream = client_stream_record(self.answer, self.fixture)
        stream['client'].update({key: record['client'][key]
                                 for key in ('preflight', 'hooks', 'max_evaluated_tokens')})
        stream['client']['mode'] = 'stream'
        stream['client']['callbacks'][0]['id'] = 'hag-client-grounded'
        for event in stream['events']: event['received_seconds'] += 200
        self.assertGreater(gate.validate_client_stream(stream, self.answer, self.fixture, 1,
                                                      self.profile)['done_seconds'], 200)
        stream['events'][-1]['received_seconds'] = 240.01
        with self.assertRaises(ValueError):
            gate.validate_client_stream(stream, self.answer, self.fixture, 1, self.profile)

    def test_history_cpu_request_gets_600_seconds_without_changing_old_deadline(self):
        self.assertEqual(gate.Server(self.root, 'cpu', 'cpu-reference', 2048, self.profile).request_seconds, 600)
        self.assertEqual(gate.Server(self.root, 'cpu', 'cpu-reference').request_seconds, 170)
        for elapsed in (590, 600.01):
            server, connection, reply = Mock(), Mock(), Mock(status=200)
            reply.getheader.return_value = 'application/json'
            reply.read.return_value = json.dumps(self.payload).encode()
            server.open.return_value = connection, reply
            path = self.root / (str(elapsed) + '.json')
            with patch.object(gate.time, 'monotonic', side_effect=[10, 10 + elapsed]):
                if elapsed <= 600:
                    self.assertEqual(gate.complete(server, self.fixture, path, self.profile), (self.answer, elapsed))
                else:
                    with self.assertRaisesRegex(ValueError, 'deadline'):
                        gate.complete(server, self.fixture, path, self.profile)
            self.assertEqual(server.open.call_args.args[2]['messages'], self.fixture['messages'])
            self.assertEqual(gate.read_json(path)['response'], self.payload)
            connection.close.assert_called_once()
            reply.close.assert_called_once()

    def test_actual_history_eos_response_retains_raw_bytes_and_refuses_unclosed_thinking(self):
        payload = response('<think>\n\n**ready**', prompt=495, completion=6, finish='stop')
        raw = json.dumps(payload).encode()
        server, connection, reply = Mock(), Mock(), Mock(status=200)
        reply.getheader.return_value = 'application/json'
        reply.read.return_value = raw
        server.open.return_value = connection, reply
        path = self.root / 'cpu-reference-project_history_ready-response.json'
        with patch.object(gate.time, 'monotonic', side_effect=[0, 195.571]), \
             patch.object(gate, 'quality_assessment') as quality, \
             self.assertRaisesRegex(ValueError, '^thinking tags are stray, nested, repeated or malformed$'):
            gate.complete(server, self.fixture, path, self.profile)
        quality.assert_not_called()
        saved = gate.read_json(path)
        self.assertEqual(base64.b64decode(saved['raw_body_base64']), raw)
        self.assertEqual(saved['response'], payload)
        self.assertEqual(saved['response']['usage'],
                         {'prompt_tokens': 495, 'completion_tokens': 6, 'total_tokens': 501})
        self.assertEqual(saved['status'], 200)
        self.assertEqual(saved['content_type'], 'application/json')
        self.assertEqual(saved['elapsed_seconds'], 195.571)
        self.assertIsNone(saved['parse_error'])
        self.assertIsNone(saved['read_error'])
        self.assertIs(saved['body_truncated'], False)
        connection.close.assert_called_once()
        reply.close.assert_called_once()

    def test_actual_history_cpu_failure_never_publishes_reference_report_or_pass(self):
        gate.write_json(self.root / 'answer-fixtures.json', self.document)
        payload = response('<think>\n\n**ready**', prompt=495, completion=6, finish='stop')
        raw = json.dumps(payload).encode()
        connection, reply = Mock(), Mock(status=200)
        reply.getheader.return_value = 'application/json'
        reply.read.return_value = raw
        with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
             patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={}), \
             patch.object(gate, 'Server') as server_type, \
             patch.object(gate.time, 'monotonic', side_effect=[0, 195.571]), \
             patch.object(gate, 'grounded_complete') as native_chat, \
             patch.object(gate, 'grounded_stream_complete') as native_stream, \
             patch('builtins.print') as printed:
            server = server_type.return_value.__enter__.return_value
            server.open.return_value = connection, reply
            with self.assertRaisesRegex(ValueError, '^thinking tags are stray, nested, repeated or malformed$'):
                gate.execute('cpu-reference', self.root)
        server_type.assert_called_once_with(self.root, 'cpu', 'cpu-reference', 2048, self.profile)
        server_type.return_value.__exit__.assert_called_once()
        self.assertIs(server_type.return_value.__exit__.call_args.args[0], ValueError)
        server.open.assert_called_once()
        native_chat.assert_not_called()
        native_stream.assert_not_called()
        printed.assert_not_called()
        saved = gate.read_json(self.root / 'cpu-reference-project_history_ready-response.json')
        self.assertEqual(base64.b64decode(saved['raw_body_base64']), raw)
        for name in ('cpu-reference.json', 'cpu-reference-report.json', 'answers-report.json', 'timing-report.json'):
            self.assertFalse((self.root / name).exists(), name)
        connection.close.assert_called_once()
        reply.close.assert_called_once()

    def test_history_cold_and_warm_refusals_use_full_prompt_boundary_without_generation(self):
        for loaded in (False, True):
            server, connection, reply = Mock(), Mock(), Mock(status=400)
            reply.getheader.return_value = 'application/json'
            payload = {'error': {'type': 'invalid_request_error', 'param': 'max_tokens',
                                 'code': 'context_length_exceeded', 'message': 'too large'}}
            reply.read.return_value = json.dumps(payload).encode()
            server.open.return_value = connection, reply
            events = [audit(1, 500)] if loaded else []
            path = self.root / (str(loaded) + '-refusal.json')
            with patch.object(gate, 'audit_events', return_value=events):
                summary = gate.capacity_refusal(server, self.fixture, loaded, loaded, len(events), path, self.profile)
            request = server.open.call_args.args[2]
            self.assertEqual(request['max_tokens'], 1555)
            self.assertEqual(request['messages'], self.fixture['messages'])
            self.assertEqual(summary['audit_count'], int(loaded))
            self.assertEqual([call.args for call in server.model.call_args_list], [(loaded,), (loaded,)])
            self.assertTrue(gate.read_json(path.with_name(path.stem + '-post.json'))['unchanged_audit'])

    def test_history_stages_execute_only_one_cpu_one_chat_and_chat_plus_stream(self):
        for stage, count in (('cpu-reference', 1), ('answers', 1), ('timing', 2)):
            root = self.root / stage
            root.mkdir()
            gate.write_json(root / 'answer-fixtures.json', self.document)
            positions = self.answer['evaluated_positions']
            events = [] if stage == 'cpu-reference' else [audit(index, positions) for index in range(1, count + 1)]
            prior = {'reference_sha256': 'reference', 'exact_cpu_parity': True, 'profile': self.profile}
            with patch.dict(gate.os.environ, {'HAG_ANSWERS_GATE': '1', 'HAG_TRACE_SUBMISSION': '0'}), \
                 patch.object(gate, 'verify_containment'), patch.object(gate, 'identity', return_value={'client_identity': {}}), \
                 patch.object(gate, 'Server') as server_type, \
                 patch.object(gate, 'complete', return_value=(self.answer, 590)) as cpu, \
                 patch.object(gate, 'grounded_complete', return_value=(self.answer, 200)) as chat, \
                 patch.object(gate, 'grounded_stream_complete', return_value={'round': 1}) as stream, \
                 patch.object(gate, 'capacity_refusal', side_effect=[{'phase': 'cold'}, {'phase': 'warm'}]) as refusal, \
                 patch.object(gate, 'audit_events', return_value=events), \
                 patch.object(gate, 'references', return_value=({self.fixture['id']: self.answer}, 'reference', 'cpu-result')), \
                 patch.object(gate, 'qualified_prerequisite', return_value=(None, prior, 'answers-result')), \
                 patch('builtins.print'):
                server_type.return_value.__enter__.return_value.event.side_effect = events
                gate.execute(stage, root)
            self.assertEqual(cpu.call_count, int(stage == 'cpu-reference'))
            self.assertEqual(chat.call_count, int(stage != 'cpu-reference'))
            self.assertEqual(stream.call_count, int(stage == 'timing'))
            self.assertEqual(refusal.call_count, 2 if stage == 'answers' else 0)
            for operation in (cpu, chat, stream):
                if operation.called: self.assertEqual(operation.call_args.args[1], self.fixture)
            report = gate.read_json(root / (stage + '-report.json'))
            self.assertEqual((report['capacity'], report['requests'], report['evaluated_positions']), (2048, count, positions * count))
            self.assertEqual(report['maximum_evaluated_positions'], gate.profile_spec(self.profile)['stage_evaluation_ceilings'][stage])
            self.assertEqual(report['qualification_kind'], 'backend_parity')
            self.assertFalse(report['require_semantic_match'])
            self.assertTrue(report['quality_is_diagnostic'])
            if stage != 'cpu-reference': self.assertFalse(report['live_cancellation_qualified'])

    def test_history_attestation_is_only_inherited_by_verified_native_server(self):
        path = '/user.slice/hag-answers-answers-20261008-000000-000001.service'
        for backend in ('cpu', 'hag'):
            folder = self.root / backend
            folder.mkdir()
            server = gate.Server(folder, backend, 'answers', 2048, self.profile)
            process = Mock()
            process.poll.return_value = None
            def spawned(*args, **kwargs):
                kwargs['stdout'].write(f'beholdd v2 listening on http://127.0.0.1:{server.port}\n')
                kwargs['stdout'].flush()
                return process
            with patch.dict(gate.os.environ, {'BEHOLDD_HAG_PRIVATE_CGROUP': path}), \
                 patch.object(gate, 'verify_containment', return_value=path), \
                 patch.object(gate.subprocess, 'Popen', side_effect=spawned) as launch, \
                 patch.object(server, 'json', return_value=(200, {'status': 'ok'})):
                server.__enter__()
            self.assertEqual(launch.call_args.kwargs['env'].get('BEHOLDD_HAG_PRIVATE_CGROUP'),
                             path if backend == 'hag' else None)
            server.log.close()
        server = gate.Server(self.root, 'hag', 'answers', 2048, self.profile)
        with patch.dict(gate.os.environ, {'BEHOLDD_HAG_PRIVATE_CGROUP': '/forged'}), \
             patch.object(gate, 'verify_containment', return_value=path), \
             patch.object(gate.subprocess, 'Popen') as launch, self.assertRaises(ValueError):
            server.__enter__()
        launch.assert_not_called()


class TemplateHistoryGateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.profile = gate.TEMPLATE_HISTORY_PROFILE
        self.document = fixture_document(self.profile)
        self.fixture = self.document['fixtures'][0]
        self.payload = response('<think>\n\n</think>\n\nready.', prompt=491, completion=5)
        self.answer = gate.generation(200, self.payload, self.fixture, self.profile)

    def chat_record(self):
        record = GroundedGateTests.chat_record(self)
        record['client']['max_evaluated_tokens'] = 2048
        record['client']['preflight'].update(capacity=2048, requested_output_tokens=1559)
        record['elapsed_seconds'] = 200
        return record

    def test_corrected_prompt_is_distinct_with_unchanged_raw_history_and_eos_admission(self):
        self.assertEqual(self.fixture['messages'], fixture_document(gate.HISTORY_PROFILE)['fixtures'][0]['messages'])
        self.assertEqual(self.answer['evaluated_positions'], 496)
        self.assertEqual(gate.generation(200, response('ready', 491, 127), self.fixture,
                                        self.profile)['evaluated_positions'], 618)
        for prompt, count in ((495, 5), (490, 5), (492, 5), (491, 128), (491, 0), (True, 5)):
            with self.subTest(prompt=prompt, count=count), self.assertRaises(ValueError):
                gate.generation(200, response('ready', prompt, count), self.fixture, self.profile)
        with self.assertRaises(ValueError):
            gate.generation(200, response('<think>\n\n**ready**', 491, 6), self.fixture, self.profile)
        wrong = gate.generation(200, response('banana', 491, 5), self.fixture, self.profile)
        self.assertEqual(wrong['quality'], {'correctness': 'fail', 'format': 'pass'})
        gate.validate_answer(wrong, self.fixture, self.profile)
        self.assertFalse(gate.profile_spec(self.profile)['require_semantic_match'])

    def test_corrected_chat_stream_and_cpu_deadlines_preserve_exact_provider_body(self):
        record = self.chat_record()
        self.assertEqual(gate.validate_grounded_chat(record, self.fixture, self.profile), (self.answer, 200))
        for mutation in (lambda r: r['client']['request']['messages'][2].update(content='ready.'),
                         lambda r: r['client']['preflight'].update(requested_output_tokens=1555),
                         lambda r: r.update(elapsed_seconds=240.01)):
            changed = deepcopy(record); mutation(changed)
            with self.assertRaises(ValueError): gate.validate_grounded_chat(changed, self.fixture, self.profile)
        stream = client_stream_record(self.answer, self.fixture)
        stream['client'].update({key: record['client'][key] for key in ('preflight', 'hooks', 'max_evaluated_tokens')})
        stream['client']['mode'] = 'stream'
        stream['client']['callbacks'][0]['id'] = 'hag-client-grounded'
        for event in stream['events']: event['received_seconds'] += 200
        self.assertGreater(gate.validate_client_stream(stream, self.answer, self.fixture, 1, self.profile)['done_seconds'], 200)
        stream['events'][-1]['received_seconds'] = 240.01
        with self.assertRaises(ValueError): gate.validate_client_stream(stream, self.answer, self.fixture, 1, self.profile)
        HistoryGateTests.test_history_cpu_request_gets_600_seconds_without_changing_old_deadline(self)

    def test_corrected_cold_warm_refusals_and_request_counts_are_bounded(self):
        for loaded in (False, True):
            server, connection, reply = Mock(), Mock(), Mock(status=400)
            reply.getheader.return_value = 'application/json'
            reply.read.return_value = json.dumps({'error': {'type': 'invalid_request_error', 'param': 'max_tokens',
                'code': 'context_length_exceeded', 'message': 'too large'}}).encode()
            server.open.return_value = connection, reply
            events = [audit(1, 496)] if loaded else []
            path = self.root / (str(loaded) + '-refusal.json')
            with patch.object(gate, 'audit_events', return_value=events):
                gate.capacity_refusal(server, self.fixture, loaded, loaded, len(events), path, self.profile)
            request = server.open.call_args.args[2]
            self.assertEqual(request['max_tokens'], 1559)
            self.assertEqual(request['messages'], self.fixture['messages'])
            self.assertTrue(gate.read_json(path.with_name(path.stem + '-post.json'))['unchanged_audit'])
        HistoryGateTests.test_history_stages_execute_only_one_cpu_one_chat_and_chat_plus_stream(self)


if __name__ == '__main__':
    unittest.main()
