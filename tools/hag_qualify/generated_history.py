"""Pure, bounded reconstruction of a generated two-turn client conversation.

The second wire request is derived from the same backend's complete first raw
answer. No template seed, final-answer trimming, model execution or I/O occurs
on import. Replay binds metadata inspection, responses and process receipts.
"""
import base64
from copy import deepcopy
import hashlib
import importlib.util
import math
from pathlib import Path
import re

try:
    from .gate_common import (approved_contract, dependency_identity, digest, gate_identity,
                              read_json, regular_reader, strict_json, model_path)
    from .profiles import GENERATED_HISTORY_PROFILE, fixture_profile, profile_spec, _typed_equal
except ImportError:
    from gate_common import (approved_contract, dependency_identity, digest, gate_identity,
                             read_json, regular_reader, strict_json, model_path)
    from profiles import GENERATED_HISTORY_PROFILE, fixture_profile, profile_spec, _typed_equal

PROFILE = GENERATED_HISTORY_PROFILE
MODEL_ID = 'hag-context-qwen3-0.6b'
BODY_LIMIT = 1024 * 1024
LOG_LIMIT = 2 * BODY_LIMIT
MODES = ('cpu-reference', 'answers')
PREFIX = 'HAG_GENERATED_HISTORY_REPORT '
_answer = None


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(_typed_equal(actual, expected), message)


def answer_helpers():
    global _answer
    if _answer is None:
        path = Path(__file__).with_name('answer-gate.py')
        spec = importlib.util.spec_from_file_location('generated_answer_helpers', path)
        _answer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_answer)
    return _answer


def bounded_bytes(path, limit=BODY_LIMIT):
    with regular_reader(path, limit) as stream:
        raw = stream.read(limit + 1)
    require(len(raw) <= limit, 'evidence exceeds byte bound')
    return raw


def text_digest(text):
    require(type(text) is str, 'history parent is not text')
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def recipe(root):
    value = read_json(root / 'answer-fixtures.json')
    require(fixture_profile(value) == PROFILE, 'generated-history recipe differs')
    return value


def fixture(document, turn, parent=None):
    require(fixture_profile(document) == PROFILE and type(turn) is int and turn in (1, 2),
            'invalid generated-history turn')
    first = deepcopy(document['first_fixture'])
    if turn == 1:
        require(parent is None, 'first turn has a history parent')
        return first
    answer_helpers().closed_final(parent)
    first['id'] = document['second_fixture_id']
    first['messages'] += [{'role': 'assistant', 'content': parent},
                          {'role': 'user', 'content': document['followup_user']}]
    return first


def request_body(document, turn, parent=None):
    value = fixture(document, turn, parent)
    return {'model': MODEL_ID, 'messages': value['messages'], 'max_tokens': 128,
            'temperature': 0, 'stream': False}


def parent_binding(root, mode, first):
    return {'response_sha256': digest(root / f'{mode}-turn-1-response.json'),
            'text_sha256': text_digest(first['text']), 'text': first['text']}


def generation(status, value, document, turn, prompt_tokens, parent=None):
    require(type(status) is int and status == 200 and type(value) is dict,
            'completion failed')
    require(value.get('object') == 'chat.completion' and value.get('model') == MODEL_ID,
            'completion identity differs')
    choices = value.get('choices')
    require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
            'completion choice count differs')
    choice = choices[0]
    require(type(choice.get('index')) is int and choice['index'] == 0
            and choice.get('finish_reason') == 'stop', 'completion did not reach EOS')
    message = choice.get('message')
    require(type(message) is dict and set(message) == {'role', 'content'}
            and message['role'] == 'assistant', 'completion is not tool-free assistant text')
    final, thinking = answer_helpers().closed_final(message['content'])
    require(type(prompt_tokens) is int and (prompt_tokens == 467 if turn == 1 else 1 <= prompt_tokens <= 491),
            'measured prompt outside fixed turn bounds')
    usage = value.get('usage')
    require(type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
            and all(type(v) is int for v in usage.values()), 'token usage schema differs')
    require(usage['prompt_tokens'] == prompt_tokens and 0 < usage['completion_tokens'] < 128
            and usage['total_tokens'] == prompt_tokens + usage['completion_tokens']
            and usage['total_tokens'] <= (594 if turn == 1 else 618), 'EOS usage exceeds measured bounds')
    bare = final[:-1] if final.endswith('.') else final
    quality = {'correctness': 'unassessed', 'format': 'fail'}
    if re.fullmatch('[A-Za-z]+', bare):
        quality = {'correctness': 'pass' if bare.casefold() == 'ready' else 'fail', 'format': 'pass'}
    return {'fixture_id': fixture(document, turn, parent)['id'], 'text': message['content'],
            'final_answer': final, 'thinking_present': thinking, 'finish_reason': 'stop',
            'usage': usage, 'evaluated_positions': usage['total_tokens'], 'quality': quality}


def http_body(record):
    raw = record.get('raw_body_base64')
    require(type(raw) is str and len(raw) <= 2 * BODY_LIMIT, 'missing or oversized raw response')
    raw = base64.b64decode(raw, validate=True)
    require(len(raw) <= BODY_LIMIT, 'raw response exceeds bound')
    require(record.get('content_type') == 'application/json', 'response content type differs')
    return strict_json(raw)


def cpu_response(root, turn, document, prompt_tokens, parent=None):
    record = read_json(root / f'cpu-reference-turn-{turn}-response.json', 2 * BODY_LIMIT)
    require(record.get('read_error') is None and record.get('body_truncated') is False,
            'CPU response was incomplete')
    equal(record.get('request'), request_body(document, turn, parent), 'CPU request history differs')
    return generation(record.get('status'), http_body(record), document, turn, prompt_tokens, parent)


def measurement(root, turn, request, identities):
    stem = f'cpu-reference-turn-{turn}'
    path = root / (stem + '-request.json')
    equal(read_json(path), request, 'measured request differs from generated history')
    record = read_json(root / (stem + '-measurement.json'))
    stdout = bounded_bytes(root / (stem + '-inspector.log'))
    stderr = bounded_bytes(root / (stem + '-inspector-stderr.log'))
    expected = strict_json(stdout)
    require(type(record.get('pid')) is int and record['pid'] > 0
            and type(record.get('exit_code')) is int and record['exit_code'] == 0
            and record.get('timed_out') is False and stderr == b'', 'metadata inspector did not complete')
    equal(record.get('command'), [str(root / 'dependencies/prompt-inspect'), str(model_path(root)), str(path)],
          'metadata inspector command differs')
    equal(record.get('result'), expected, 'metadata result differs from captured stdout')
    require(record.get('inspector_sha256') == identities['dependencies']['prompt-inspect']
            and record.get('request_sha256') == digest(path)
            and record.get('model_sha256') == identities['checkpoint_sha256']
            and record.get('stdout_sha256') == digest(root / (stem + '-inspector.log'))
            and record.get('stderr_sha256') == digest(root / (stem + '-inspector-stderr.log')),
            'metadata input/output hashes differ')
    require(type(record.get('elapsed_seconds')) in (float, int) and math.isfinite(record['elapsed_seconds'])
            and 0 <= record['elapsed_seconds'] <= 30, 'metadata inspection exceeded deadline')
    require(type(expected) is dict and set(expected) == {'execution', 'tokenization', 'checkpoint', 'requests'}
            and expected['execution'] == 'not_requested' and expected['tokenization'] == 'exact_checkpoint_metadata'
            and expected['checkpoint'] == str(model_path(root)), 'metadata inspector identity differs')
    rows = expected['requests']
    require(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict, 'metadata request count differs')
    row = rows[0]
    require(set(row) == {'request', 'status', 'message_count', 'tool_count', 'rendered_prompt_bytes',
                        'prompt_tokens', 'output_tokens', 'minimum_evaluated_positions', 'output_budget_source'},
            'metadata measurement schema differs')
    prompt = row['prompt_tokens']
    require(type(prompt) is int and (prompt == 467 if turn == 1 else 1 <= prompt <= 491)
            and row['request'] == str(path) and row['status'] == 'measured'
            and type(row['message_count']) is int and row['message_count'] == (2 if turn == 1 else 4)
            and type(row['tool_count']) is int and row['tool_count'] == 0
            and type(row['rendered_prompt_bytes']) is int and 0 < row['rendered_prompt_bytes'] <= 65536
            and type(row['output_tokens']) is int and row['output_tokens'] == 128
            and type(row['minimum_evaluated_positions']) is int
            and row['minimum_evaluated_positions'] == prompt + 127 <= (594 if turn == 1 else 618)
            and row['output_budget_source'] == 'explicit', 'metadata prompt exceeds fixed recipe bounds')
    return prompt


def identity(root):
    contract = approved_contract(root)
    require(contract['profile'] == PROFILE and 'runtime_phase_diagnostics' not in contract,
            'generated history requires its phase-off contract')
    gate_identity(root)
    result = answer_helpers().identity(root)
    result['dependencies'] = dependency_identity(root)
    require('prompt-inspect' in result['dependencies'] and 'client_identity' in result,
            'frozen inspector and real provider required')
    return result


def raw_control(root, mode, name, method, route, request):
    value = read_json(root / f'{mode}-{name}.json', 2 * BODY_LIMIT)
    equal({k: value.get(k) for k in ('method', 'route', 'request', 'status', 'read_error', 'body_truncated')},
          {'method': method, 'route': route, 'request': request, 'status': 200,
           'read_error': None, 'body_truncated': False}, 'server control response differs')
    return http_body(value)


def validate_server(root, mode, child=None):
    record = read_json(root / (mode + '-server-process.json'))
    require(type(record.get('pid')) is int and record['pid'] > 0 and type(record.get('exit_code')) is int
            and record['exit_code'] == 0 and record.get('idle_sigterm_exit') is True,
            'server has no successful idle termination receipt')
    require(type(record.get('start_time_ticks')) is int and record['start_time_ticks'] > 0
            and record.get('no_new_privileges') is True, 'server process identity/NNP differs')
    group = record.get('cgroup')
    require(type(group) is str and Path(group).is_relative_to('/sys/fs/cgroup')
            and '..' not in Path(group).parts, 'server cgroup is invalid')
    if child is not None:
        equal(group, child['cgroup'], 'server escaped the worker cgroup')
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    info = raw_control(root, mode, 'identity', 'GET', '/v1/beholdd/info', None)
    for key, expected in {'service': 'beholdd', 'runtime': 'native', 'generation': 2,
                          'execution_backends': ['cpu'] if backend == 'cpu' else ['cpu', 'hag']}.items():
        equal(info.get(key), expected, 'server identity differs: ' + key)
    for label, loaded in (('cold', False), ('after-turn-1', True), ('after-turn-2', True), ('unloaded', False)):
        value = raw_control(root, mode, label + '-model', 'GET', '/v1/models', None)
        require(type(value.get('data')) is list and len(value['data']) == 1, 'model list differs')
        model = value['data'][0]
        require(model.get('id') == MODEL_ID and model.get('loaded') is loaded, 'server residency differs')
        equal(model.get('execution_limits', {}).get('max_evaluated_tokens'), 2048 if backend == 'hag' else None,
              'server public capacity differs')
    for index in (1, 2):
        equal(raw_control(root, mode, f'unload-{index}', 'POST', f'/v1/models/{MODEL_ID}/unload', {}),
              {'id': MODEL_ID, 'loaded': False}, 'idempotent unload differs')
    return record


def server_processes(root, mode):
    if mode == 'self-check':
        return []
    record = validate_server(root, mode, read_json(root / (mode + '-child-result.json')))
    return [{'pid': record['pid'], 'exit_code': record['exit_code']}]


def client_artifacts():
    return [('answers', f'answers-turn-{turn}', f'answers-turn-{turn}-response.json') for turn in (1, 2)]


def native_response(root, turn, document, prompt, parent, identities):
    stem = f'answers-turn-{turn}'
    record = read_json(root / (stem + '-response.json'), 2 * BODY_LIMIT)
    envelope = read_json(root / (stem + '-client-request.json'))
    body = request_body(document, turn, None if parent is None else parent['text'])
    require(set(envelope) == {'profile', 'turn', 'prompt_tokens', 'request', 'model_descriptor', 'history_parent'},
            'client envelope schema differs')
    equal({k: envelope[k] for k in ('profile', 'turn', 'prompt_tokens', 'request', 'history_parent')},
          {'profile': PROFILE, 'turn': turn, 'prompt_tokens': prompt, 'request': body, 'history_parent': parent},
          'client request is not derived from the same native history')
    descriptor = envelope['model_descriptor']
    require(descriptor.get('id') == MODEL_ID and descriptor.get('loaded') is (turn == 2), 'client discovery residency differs')
    equal(descriptor.get('execution_limits'), {'max_evaluated_tokens': 2048}, 'client discovery capacity differs')
    client = record.get('client')
    require(type(client) is dict and client.get('kind') == 'antimatter' and client.get('mode') == 'chat'
            and type(client.get('request_count')) is int and client['request_count'] == 1
            and type(client.get('attempt_count')) is int and client['attempt_count'] == 1
            and client.get('error') is None and client.get('response_complete') is True,
            'provider did not complete exactly one chat request')
    equal(client.get('max_evaluated_tokens'), 2048, 'provider capacity differs')
    equal(client.get('request'), body, 'provider changed generated-history wire bytes')
    equal(client.get('preflight'), {'capacity': 2048, 'requested_output_tokens': 2048, 'fetch_attempts': 0,
                                  'track_calls': 0, 'prepare_calls': 0, 'error_code': 'context_length_exceeded'},
          'provider local refusal was not before hooks/fetch')
    equal(client.get('hooks'), {'track_calls': 1, 'prepare_calls': 1, 'track_end_calls': 1}, 'provider lifecycle differs')
    equal(record.get('events'), [], 'nonstream provider recorded SSE')
    equal(client.get('callbacks'), [], 'chat emitted stream callbacks')
    require(record.get('clock') == 'monotonic_elapsed_from_request_start'
            and type(record.get('elapsed_seconds')) in (int, float) and math.isfinite(record['elapsed_seconds'])
            and 0 < record['elapsed_seconds'] <= 240, 'provider request clock exceeds bound')
    value = generation(record.get('status'), http_body(record), document, turn, prompt,
                       None if parent is None else parent['text'])
    equal(client.get('result'), value['text'], 'provider return differs from complete raw JSON')
    require(type(client.get('pid')) is int and client['pid'] > 0, 'missing provider process PID')
    equal(read_json(root / (stem + '-client-process.json')),
          {'pid': client['pid'], 'exit_code': 0, 'timed_out': False, 'identity': identities['client_identity']},
          'provider process receipt differs')
    bounded_bytes(root / (stem + '-client.log'), LOG_LIMIT)
    return value


def events(root, mode, answers):
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    raw = bounded_bytes(root / f'{mode}-{backend}-server.log', LOG_LIMIT)
    require(not raw or raw.endswith(b'\n'), 'server log is incomplete')
    require(not any(line.startswith(b'HAG_SERVING_PHASE ') for line in raw.splitlines()), 'phase diagnostics enabled')
    observed = [strict_json(line[len(b'HAG_SERVING_AUDIT '):]) for line in raw.splitlines()
                if line.startswith(b'HAG_SERVING_AUDIT ')]
    expected = [] if mode == 'cpu-reference' else [
        {'request_sequence': turn, 'evaluated_positions': value['evaluated_positions'],
         'outcome': 'success', 'reset_ok': True, 'poisoned': False} for turn, value in enumerate(answers, 1)]
    equal(observed, expected, 'native request count/reset/EOS accounting differs')
    return observed


def evidence_names(mode):
    if mode == 'self-check':
        return []
    require(mode in MODES, 'unknown generated-history stage')
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    names = [f'{mode}-{backend}-server.log', mode + '-server-process.json', mode + '-report.json']
    names += [f'{mode}-{label}.json' for label in ('identity', 'cold-model', 'after-turn-1-model',
                                                  'after-turn-2-model', 'unload-1', 'unload-2', 'unloaded-model')]
    for turn in (1, 2):
        stem = f'{mode}-turn-{turn}'
        names += [stem + '-request.json', stem + '-lineage.json', stem + '-response.json']
        if mode == 'cpu-reference':
            names += [stem + '-measurement.json', stem + '-inspector.log', stem + '-inspector-stderr.log']
        else:
            names += [stem + '-client-request.json', stem + '-client-process.json', stem + '-client.log']
    if mode == 'cpu-reference':
        names.append('cpu-reference.json')
    return names


def reconstruct(root, mode, identities, child=None):
    require(mode in MODES, 'unknown generated-history stage')
    document = recipe(root)
    validate_server(root, mode, child)
    if mode == 'answers':
        cpu = summarize_stage(root, 'cpu-reference')
        result = read_json(root / 'cpu-reference-result.json')
        require(result.get('qualified') is True and result.get('failures') == []
                and type(result.get('exit_code')) is int and result['exit_code'] == 0, 'CPU prerequisite did not qualify')
        equal(result.get('probe'), cpu, 'CPU prerequisite report differs')
    answers, turns = [], []
    for turn in (1, 2):
        parent = None if turn == 1 else parent_binding(root, mode, answers[0])
        request = request_body(document, turn, None if parent is None else parent['text'])
        stem = f'{mode}-turn-{turn}'
        equal(read_json(root / (stem + '-request.json')), request, 'actual wire request differs from generated history')
        prompt = measurement(root, turn, request, identities)
        binding = {'turn': turn, 'request_sha256': digest(root / (stem + '-request.json')),
                   'history_parent': parent, 'measurement_sha256': digest(root / f'cpu-reference-turn-{turn}-measurement.json')}
        if mode == 'answers':
            binding.update(cpu_reference_sha256=digest(root / 'cpu-reference.json'),
                           first_turn_parity_verified=turn == 2)
        equal(read_json(root / (stem + '-lineage.json')), binding, 'turn ancestry/admission differs')
        if mode == 'cpu-reference':
            value = cpu_response(root, turn, document, prompt, None if parent is None else parent['text'])
        else:
            value = native_response(root, turn, document, prompt, parent, identities)
            equal(value, cpu['answers'][turn - 1], 'native answer/text/usage differs from CPU')
        answers.append(value)
        turns.append({**binding, 'response_sha256': digest(root / (stem + '-response.json')), 'prompt_tokens': prompt})
    return answers, turns, events(root, mode, answers)


def summarize_stage(root, mode, *, child=None):
    ids = identity(root)
    answers, turns, audit = reconstruct(root, mode, ids, child)
    reference = {'schema': 1, 'profile': PROFILE, 'identities': ids, 'capacity': 2048,
                 'answers': answers if mode == 'cpu-reference' else read_json(root / 'cpu-reference.json')['answers'],
                 'turns': turns if mode == 'cpu-reference' else read_json(root / 'cpu-reference.json')['turns'],
                 'cpu_exit_code': 0, 'final_unloaded': True}
    equal(read_json(root / 'cpu-reference.json'), reference, 'CPU reference artifact differs')
    evaluated = sum(value['evaluated_positions'] for value in answers)
    require(evaluated <= 1212, 'generated conversation exceeds evaluation bound')
    report = {'stage': mode, 'execution': 'cpu-reference' if mode == 'cpu-reference' else 'native-hag-answers',
              'profile': PROFILE, 'qualification_kind': 'backend_parity', 'require_semantic_match': False,
              'quality_is_diagnostic': True, 'capacity': 2048, 'requests': 2,
              'fixture_ids': [value['fixture_id'] for value in answers], 'answers': answers, 'turns': turns,
              'evaluated_positions': evaluated, 'maximum_evaluated_positions': 1212,
              'all_eos': True, 'closed_answers': True, 'generated_history': True,
              'history_source': 'same-backend-raw-first-answer', 'final_unloaded': True,
              'server_exit_code': 0, 'identities': ids, 'client_identity': ids['client_identity'],
              'reference_sha256': digest(root / 'cpu-reference.json'), 'audit_events': audit}
    if mode == 'cpu-reference':
        report['cpu_exit_code'] = 0
    else:
        report.update(hag_exit_code=0, exact_cpu_parity=True, first_turn_parity_before_second=True,
                      cpu_reference_result_sha256=digest(root / 'cpu-reference-result.json'),
                      client_chat_requests=2, client_stream_requests=0, model_free_client_refusals=2,
                      live_cancellation_qualified=False)
    return report


def validate_stage(root, mode, text):
    report = summarize_stage(root, mode, child=read_json(root / (mode + '-child-result.json')))
    equal(read_json(root / (mode + '-report.json')), report, 'generated-history report differs from raw evidence')
    rows = [strict_json(line[len(PREFIX):]) for line in text.splitlines() if line.startswith(PREFIX)]
    equal(rows, [report], 'expected exactly one matching generated-history report')
    equal(re.findall(r'^PASS hag-generated-history stage=(\S+) qualified=(\S+)$', text, re.M),
          [(mode, 'true')], 'generated-history stage must pass exactly once')
    return report
