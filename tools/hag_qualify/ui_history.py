"""Reconstruct bounded UI-history backend evidence without executing a model.

The desktop client is on another host. These stage receipts qualify only the
remote admission/backend path; the separately imported desktop evidence must
also pass combined closeout before a live UI qualification can be claimed.
"""
import base64
from copy import deepcopy
import hashlib
import re
import uuid

try:
    from . import generated_history as history
    from .gate_common import (approved_contract, dependency_identity, digest, gate_identity,
                              read_json, require_sha, strict_json, model_path)
    from .profiles import UI_HISTORY_PROFILE, fixture_document, fixture_profile
except ImportError:
    import generated_history as history
    from gate_common import (approved_contract, dependency_identity, digest, gate_identity,
                             read_json, require_sha, strict_json, model_path)
    from profiles import UI_HISTORY_PROFILE, fixture_document, fixture_profile

PROFILE = UI_HISTORY_PROFILE
MODEL_ID = history.MODEL_ID
BODY_LIMIT = history.BODY_LIMIT
LOG_LIMIT = history.LOG_LIMIT
MODES = history.MODES
PREFIX = 'HAG_UI_HISTORY_REPORT '
require = history.require
equal = history.equal
bounded_bytes = history.bounded_bytes
text_digest = history.text_digest
answer_helpers = history.answer_helpers
measurement = history.measurement
raw_control = history.raw_control
validate_server = history.validate_server
events = history.events
parent_binding = history.parent_binding


def http_body(record):
    """Validate a complete bounded native SSE reply and project its completion.

    Keep-alive comments are transport metadata. Every data event must belong to
    one role/content/terminal sequence, followed by exactly one final sentinel.
    Raw bytes stay in the evidence and are never replaced by this projection.
    """
    require(type(record) is dict and record.get('read_error') is None
            and record.get('body_truncated') is False, 'SSE response was incomplete')
    content_type = record.get('content_type')
    require(type(content_type) is str and re.fullmatch(r'text/event-stream(?:;\s*charset=utf-8)?',
            content_type, re.IGNORECASE) is not None, 'SSE response content type differs')
    encoded = record.get('raw_body_base64')
    require(type(encoded) is str and len(encoded) <= 2 * BODY_LIMIT, 'missing or oversized raw SSE response')
    raw = base64.b64decode(encoded, validate=True)
    require(0 < len(raw) <= BODY_LIMIT, 'raw SSE response exceeds bound')
    text = raw.decode('utf-8').replace('\r\n', '\n')
    require('\r' not in text and '\x00' not in text and text.endswith('\n\n'),
            'SSE response has malformed or incomplete framing')
    phase, identity, content, usage = 'role', None, [], None
    base_keys = {'id', 'object', 'created', 'model', 'choices'}
    for block in text.split('\n\n'):
        if not block:
            continue
        require(phase != 'done', 'SSE contains data after its completion sentinel')
        lines = [line for line in block.split('\n') if line and not line.startswith(':')]
        if not lines:
            continue
        require(len(lines) == 1 and lines[0].startswith('data:'), 'unsupported SSE event field or multiline data')
        data = lines[0][5:]
        if data.startswith(' '):
            data = data[1:]
        if data == '[DONE]':
            require(phase == 'sentinel', 'SSE sentinel preceded a complete terminal event')
            phase = 'done'
            continue
        value = strict_json(data)
        require(type(value) is dict and set(value) in (base_keys, base_keys | {'usage'}),
                'SSE chunk schema differs')
        require(value['object'] == 'chat.completion.chunk' and value['model'] == MODEL_ID
                and type(value['id']) is str and re.fullmatch(r'chatcmpl-beholdd-[0-9]+', value['id']) is not None
                and type(value['created']) is int and value['created'] >= 0, 'SSE chunk identity differs')
        current = {key: value[key] for key in ('id', 'created', 'model')}
        if identity is None:
            identity = current
        equal(current, identity, 'SSE chunks do not share one response identity')
        choices = value['choices']
        require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict
                and set(choices[0]) == {'index', 'delta', 'finish_reason'}
                and type(choices[0]['index']) is int and choices[0]['index'] == 0,
                'SSE choice schema differs')
        choice = choices[0]
        delta, finish = choice['delta'], choice['finish_reason']
        require(type(delta) is dict, 'SSE delta is not an object')
        if phase == 'role':
            equal(delta, {'role': 'assistant'}, 'SSE must begin with exactly one assistant role')
            require(finish is None and 'usage' not in value, 'SSE role cannot finish or report usage')
            phase = 'content'
        elif phase == 'content' and finish is None:
            require(set(delta) == {'content'} and type(delta['content']) is str and 'usage' not in value,
                    'SSE content event differs or requests tools')
            content.append(delta['content'])
        elif phase == 'content' and finish == 'stop':
            require(bool(content) and delta == {} and 'usage' in value,
                    'SSE terminal event requires content, an empty delta and usage')
            usage = value['usage']
            require(type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
                    and all(type(count) is int and count > 0 for count in usage.values())
                    and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
                    'SSE terminal usage differs')
            phase = 'sentinel'
        else:
            raise ValueError('SSE has an unexpected, repeated or unfinished data event')
    require(phase == 'done' and identity is not None and usage is not None, 'SSE did not complete exactly once')
    return {**identity, 'object': 'chat.completion',
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': ''.join(content)}}], 'usage': usage}


def recipe(root):
    value = read_json(root / 'answer-fixtures.json')
    require(fixture_profile(value) == PROFILE, 'UI-history recipe differs')
    return value


def fixture(document, turn, parent=None):
    require(fixture_profile(document) == PROFILE and type(turn) is int and turn in (1, 2),
            'invalid UI-history turn')
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
    return {'model': MODEL_ID, 'messages': value['messages'], 'max_tokens': 128, 'stream': True}


def admit_request(turn, body, parent, reference, document=None):
    """Refuse every unregistered request before forwarding it to the backend.

    The caller has already reconstructed the CPU reference. Turn two is bound
    to the observed native first reply, whose full text must match that CPU
    reply; the worker additionally verifies the native reset before calling us.
    """
    require(type(turn) is int and turn in (1, 2), 'only two ordered UI requests are admitted')
    document = fixture_document(PROFILE) if document is None else document
    require(type(reference) is dict and reference.get('profile') == PROFILE
            and type(reference.get('capacity')) is int and reference['capacity'] == 2048,
            'UI admission reference profile or capacity differs')
    answers = reference.get('answers')
    require(type(answers) is list and len(answers) == 2 and all(type(v) is dict for v in answers),
            'UI admission requires exactly two CPU answers')
    if turn == 1:
        require(parent is None, 'first UI request has an unexpected parent')
        parent_text = None
    else:
        require(type(parent) is dict and set(parent) == {'response_sha256', 'text_sha256', 'text'},
                'second UI request requires a bound native parent')
        require_sha(parent['response_sha256'])
        equal(parent['text_sha256'], text_digest(parent['text']), 'native parent text hash differs')
        equal(parent['text'], answers[0].get('text'), 'native first reply differs from CPU reference')
        parent_text = parent['text']
    expected = request_body(document, turn, parent_text)
    equal(body, expected, 'UI request differs from the admitted wire recipe')
    return expected


def generation(status, value, document, turn, prompt_tokens, parent=None):
    require(type(status) is int and status == 200 and type(value) is dict, 'completion failed')
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


def identity(root):
    try:
        from .ui_client import client_identity
    except ImportError:
        from ui_client import client_identity
    contract = approved_contract(root)
    require(contract['profile'] == PROFILE and contract.get('stream_client') == 'antimatter-ui'
            and 'runtime_phase_diagnostics' not in contract, 'UI history requires its phase-off contract')
    gate_identity(root)
    result = answer_helpers().identity(root)
    result['dependencies'] = dependency_identity(root)
    client = client_identity(root)
    require('prompt-inspect' in result['dependencies'] and client is not None,
            'frozen inspector and desktop client identity required')
    if 'client_identity' in result:
        equal(result['client_identity'], client, 'desktop client identity dispatch differs')
    result['client_identity'] = client
    return result


def response(root, mode, turn, document, prompt_tokens, parent=None):
    require(mode in MODES, 'unknown UI-history response stage')
    record = read_json(root / f'{mode}-turn-{turn}-response.json', 2 * BODY_LIMIT)
    equal({key: record.get(key) for key in ('method', 'route', 'request', 'status', 'read_error', 'body_truncated')},
          {'method': 'POST', 'route': '/v1/chat/completions', 'request': request_body(document, turn, parent),
           'status': 200, 'read_error': None, 'body_truncated': False}, 'captured UI backend request differs')
    return generation(record['status'], http_body(record), document, turn, prompt_tokens, parent)


def server_processes(root, mode):
    if mode == 'self-check':
        return []
    record = validate_server(root, mode, read_json(root / (mode + '-child-result.json')))
    return [{'pid': record['pid'], 'exit_code': record['exit_code']}]


def listener(root):
    try:
        from .ui_client import client_host
    except ImportError:
        from ui_client import client_host
    approved_contract(root)
    record = read_json(root / 'answers-ui-listener.json')
    expected = {'schema': 1, 'profile': PROFILE, 'host': '127.0.0.1',
                'request_count': 2, 'forwarded_count': 2, 'response_count': 2,
                'rejected_count': 0, 'closed': True,
                'client_identity_sha256': digest(root / 'client/identity.json'),
                'delivery': 'buffered-sse-after-parity-reset'}
    require(type(record) is dict and set(record) == {*expected, 'port', 'pid', 'boot_id', 'run_id'},
            'UI listener receipt schema differs')
    equal({key: record[key] for key in expected}, expected, 'UI listener did not complete exactly two admissions')
    require(type(record['port']) is int and 1024 <= record['port'] <= 65535
            and type(record['pid']) is int and record['pid'] > 0, 'UI listener identity differs')
    for key in ('run_id', 'boot_id'):
        require(type(record[key]) is str and str(uuid.UUID(record[key])) == record[key],
                'UI listener UUID is not canonical: ' + key)
    require(uuid.UUID(record['run_id']).version == 4, 'UI listener run identity is not a fresh UUID4')
    ready_keys = ('schema', 'profile', 'host', 'port', 'pid', 'client_identity_sha256',
                  'boot_id', 'run_id', 'delivery')
    equal(read_json(root / 'answers-ui-ready.json'), {key: record[key] for key in ready_keys},
          'UI listener readiness and final identity differ')
    client_host(root, read_json(root / 'client/identity.json'), record)
    return record


def admission(root, turn, request):
    """Bind the exact incoming bytes, forwarded bytes and released reply."""
    stem = f'answers-turn-{turn}'
    record = read_json(root / (stem + '-admission.json'), 4 * BODY_LIMIT)
    expected_keys = {'turn', 'request_raw_base64', 'request_raw_sha256', 'forwarded_raw_base64',
                     'forwarded_raw_sha256', 'response_body_sha256', 'parity_verified', 'reset_verified'}
    require(type(record) is dict and set(record) == expected_keys, 'UI admission receipt schema differs')
    equal({key: record[key] for key in ('turn', 'parity_verified', 'reset_verified')},
          {'turn': turn, 'parity_verified': True, 'reset_verified': True},
          'UI response was released before parity and reset')
    raw_values = []
    for prefix in ('request', 'forwarded'):
        encoded = record[prefix + '_raw_base64']
        require(type(encoded) is str and len(encoded) <= 2 * BODY_LIMIT, 'UI request capture exceeds bound')
        raw = base64.b64decode(encoded, validate=True)
        require(0 < len(raw) <= BODY_LIMIT, 'UI request bytes exceed bound')
        equal(hashlib.sha256(raw).hexdigest(), record[prefix + '_raw_sha256'], 'UI request byte hash differs')
        equal(strict_json(raw), request, 'UI request raw bytes differ from admitted history')
        raw_values.append(raw)
    equal(raw_values[0], raw_values[1], 'UI forwarded request bytes changed')
    response_record = read_json(root / (stem + '-response.json'), 2 * BODY_LIMIT)
    raw_response = base64.b64decode(response_record['raw_body_base64'], validate=True)
    require(len(raw_response) <= BODY_LIMIT, 'UI response bytes exceed bound')
    equal(hashlib.sha256(raw_response).hexdigest(), record['response_body_sha256'],
          'UI released reply differs from backend capture')
    return record


def broker_records(root):
    """The two raw backend replies consumed by combined desktop closeout."""
    return [{**read_json(root / f'answers-turn-{turn}-response.json', 2 * BODY_LIMIT), 'turn': turn}
            for turn in (1, 2)]


def evidence_names(mode):
    if mode == 'self-check':
        return []
    require(mode in MODES, 'unknown UI-history stage')
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
            names.append(stem + '-admission.json')
    names.append('cpu-reference.json' if mode == 'cpu-reference' else 'answers-ui-listener.json')
    if mode == 'answers':
        names.append('answers-ui-ready.json')
    return names


def reconstruct(root, mode, identities, child=None):
    require(mode in MODES, 'unknown UI-history stage')
    document = recipe(root)
    validate_server(root, mode, child)
    if mode == 'answers':
        cpu = summarize_stage(root, 'cpu-reference')
        result = read_json(root / 'cpu-reference-result.json')
        require(result.get('qualified') is True and result.get('failures') == []
                and type(result.get('exit_code')) is int and result['exit_code'] == 0,
                'CPU prerequisite did not qualify')
        equal(result.get('probe'), cpu, 'CPU prerequisite report differs')
    answers, turns = [], []
    for turn in (1, 2):
        parent = None if turn == 1 else parent_binding(root, mode, answers[0])
        parent_text = None if parent is None else parent['text']
        request = request_body(document, turn, parent_text)
        stem = f'{mode}-turn-{turn}'
        actual = read_json(root / (stem + '-request.json'))
        equal(actual, request, 'actual wire request differs from UI history')
        if mode == 'answers':
            admit_request(turn, actual, parent, cpu, document)
        prompt = measurement(root, turn, request, identities)
        binding = {'turn': turn, 'request_sha256': digest(root / (stem + '-request.json')),
                   'history_parent': parent, 'measurement_sha256': digest(root / f'cpu-reference-turn-{turn}-measurement.json')}
        if mode == 'answers':
            binding.update(cpu_reference_sha256=digest(root / 'cpu-reference.json'),
                           first_turn_parity_verified=turn == 2)
        equal(read_json(root / (stem + '-lineage.json')), binding, 'UI turn ancestry/admission differs')
        value = response(root, mode, turn, document, prompt, parent_text)
        if mode == 'answers':
            equal(value, cpu['answers'][turn - 1], 'native answer/text/usage differs from CPU')
            admission(root, turn, actual)
        answers.append(value)
        turns.append({**binding, 'response_sha256': digest(root / (stem + '-response.json')), 'prompt_tokens': prompt})
    if mode == 'answers':
        listener(root)
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
    require(evaluated <= 1212, 'UI conversation exceeds evaluation bound')
    report = {'stage': mode, 'execution': 'cpu-reference' if mode == 'cpu-reference' else 'native-hag-answers',
              'profile': PROFILE, 'qualification_kind': 'ui_history_and_backend_parity', 'require_semantic_match': False,
              'quality_is_diagnostic': True, 'capacity': 2048, 'requests': 2,
              'fixture_ids': [value['fixture_id'] for value in answers], 'answers': answers, 'turns': turns,
              'evaluated_positions': evaluated, 'maximum_evaluated_positions': 1212,
              'all_eos': True, 'closed_answers': True, 'generated_history': True,
              'history_source': 'same-backend-raw-first-answer', 'final_unloaded': True,
              'server_exit_code': 0, 'identities': ids, 'client_identity': ids['client_identity'],
              'reference_sha256': digest(root / 'cpu-reference.json'), 'audit_events': audit,
              'live_ui_qualified': False, 'client_evidence_pending': True}
    if mode == 'cpu-reference':
        report['cpu_exit_code'] = 0
    else:
        report.update(hag_exit_code=0, exact_cpu_parity=True, first_turn_parity_before_second=True,
                      cpu_reference_result_sha256=digest(root / 'cpu-reference-result.json'),
                      client_chat_requests=0, client_stream_requests=2, transport='buffered-sse',
                      live_cancellation_qualified=False, listener=listener(root))
    return report


def validate_stage(root, mode, text):
    report = summarize_stage(root, mode, child=read_json(root / (mode + '-child-result.json')))
    equal(read_json(root / (mode + '-report.json')), report, 'UI-history report differs from raw evidence')
    rows = [strict_json(line[len(PREFIX):]) for line in text.splitlines() if line.startswith(PREFIX)]
    equal(rows, [report], 'expected exactly one matching UI-history report')
    equal(re.findall(r'^PASS hag-ui-history stage=(\S+) qualified=(\S+)$', text, re.M),
          [(mode, 'true')], 'UI-history stage must pass exactly once')
    return report
