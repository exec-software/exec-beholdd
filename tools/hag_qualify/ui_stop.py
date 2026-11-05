"""Replay a real editor disconnect, healthy native reset, and fresh-chat reuse.

The first stream is deliberately incomplete. Its cancellation comes from the
client's socket, not a synthetic callback. Only the second request has CPU
parity; desktop Stop/New chat evidence is still required by combined closeout.
"""
import base64
from copy import deepcopy
import hashlib
import math
from pathlib import Path
import re
import uuid

try:
    from . import ui_history as history
    from .gate_common import (approved_contract, dependency_identity, digest, gate_identity,
                              read_json, strict_json, model_path)
    from .profiles import UI_STOP_PROFILE, fixture_document, fixture_profile, profile_limits, profile_spec
except ImportError:
    import ui_history as history
    from gate_common import (approved_contract, dependency_identity, digest, gate_identity,
                             read_json, strict_json, model_path)
    from profiles import UI_STOP_PROFILE, fixture_document, fixture_profile, profile_limits, profile_spec

PROFILE = UI_STOP_PROFILE
MODEL_ID, BODY_LIMIT, LOG_LIMIT = history.MODEL_ID, history.BODY_LIMIT, history.LOG_LIMIT
MODES, PREFIX = history.MODES, 'HAG_UI_STOP_REPORT '
require, equal = history.require, history.equal
bounded_bytes, text_digest = history.bounded_bytes, history.text_digest
answer_helpers, raw_control = history.answer_helpers, history.raw_control
DELIVERY = 'live-sse-stop-then-buffered-parity-reset'
MAX_EMPTY_CONNECTIONS = 2


def recipe(root):
    value = read_json(root / 'answer-fixtures.json')
    equal(fixture_profile(value), PROFILE, 'UI Stop recipe differs')
    return value


def fixture(document, turn):
    require(fixture_profile(document) == PROFILE and type(turn) is int and turn in (1, 2),
            'invalid UI Stop turn')
    return deepcopy(document['first_fixture' if turn == 1 else 'reuse_fixture'])


def request_body(document, turn):
    value = fixture(document, turn)
    return {'model': MODEL_ID, 'messages': value['messages'], 'max_tokens': value['max_tokens'], 'stream': True}


def prompt_bound(turn):
    return profile_limits(PROFILE)['turn_prompt_tokens'][turn - 1]


def ceiling(turn):
    return profile_limits(PROFILE)['turn_evaluation_ceilings'][turn - 1]


def admit_request(turn, body, reference, document=None):
    require(type(turn) is int and turn in (1, 2), 'only two ordered UI Stop requests are admitted')
    document = fixture_document(PROFILE) if document is None else document
    require(type(reference) is dict and reference.get('profile') == PROFILE
            and type(reference.get('capacity')) is int and reference['capacity'] == 2048,
            'UI Stop reference profile differs')
    require(type(reference.get('answers')) is list and len(reference['answers']) == 1,
            'UI Stop requires exactly one CPU reuse answer')
    expected = request_body(document, turn)
    equal(body, expected, 'UI Stop request differs from the fixed fresh-chat wire recipe')
    return expected


class SseState:
    """Strict incremental SSE framing, with an explicit incomplete-stream state."""
    def __init__(self):
        self.pending = bytearray()
        self.phase, self.identity, self.content, self.usage, self.finish_reason = 'role', None, [], None, None
        self.bytes = 0

    @property
    def text(self):
        return ''.join(self.content)

    def feed(self, raw):
        require(type(raw) is bytes, 'SSE input must be bytes')
        self.bytes += len(raw)
        require(self.bytes <= BODY_LIMIT, 'SSE exceeds fixed byte bound')
        self.pending.extend(raw)
        while True:
            end = self.pending.find(b'\n\n')
            crlf_end = self.pending.find(b'\r\n\r\n')
            if crlf_end >= 0 and (end < 0 or crlf_end < end):
                end, width = crlf_end, 4
            else:
                width = 2
            if end < 0:
                return
            packet = bytes(self.pending[:end + width])
            del self.pending[:end + width]
            self.block(packet)

    def block(self, raw):
        require(self.phase != 'done', 'SSE contains bytes after completion')
        text = raw.decode('utf-8').replace('\r\n', '\n')
        require('\r' not in text and '\x00' not in text and text.endswith('\n\n'), 'SSE framing differs')
        lines = [line for line in text[:-2].split('\n') if line and not line.startswith(':')]
        if not lines:
            return
        require(len(lines) == 1 and lines[0].startswith('data:'), 'unsupported SSE fields')
        data = lines[0][5:]
        if data.startswith(' '):
            data = data[1:]
        if data == '[DONE]':
            require(self.phase == 'sentinel', 'SSE sentinel without terminal event')
            self.phase = 'done'
            return
        value = strict_json(data)
        base = {'id', 'object', 'created', 'model', 'choices'}
        require(type(value) is dict and set(value) in (base, base | {'usage'}), 'SSE chunk schema differs')
        require(value['object'] == 'chat.completion.chunk' and value['model'] == MODEL_ID
                and type(value['id']) is str and re.fullmatch(r'chatcmpl-beholdd-[0-9]+', value['id'])
                and type(value['created']) is int and value['created'] >= 0, 'SSE identity differs')
        identity = {key: value[key] for key in ('id', 'created', 'model')}
        if self.identity is None:
            self.identity = identity
        equal(identity, self.identity, 'SSE chunks changed response identity')
        choices = value['choices']
        require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict
                and set(choices[0]) == {'index', 'delta', 'finish_reason'}
                and type(choices[0]['index']) is int and choices[0]['index'] == 0, 'SSE choice schema differs')
        delta, finish = choices[0]['delta'], choices[0]['finish_reason']
        require(type(delta) is dict, 'SSE delta differs')
        if self.phase == 'role':
            equal(delta, {'role': 'assistant'}, 'SSE must begin with one assistant role')
            require(finish is None and 'usage' not in value, 'role cannot finish generation')
            self.phase = 'content'
        elif self.phase == 'content' and finish is None:
            require(set(delta) == {'content'} and type(delta['content']) is str and 'usage' not in value,
                    'SSE content chunk differs')
            self.content.append(delta['content'])
        elif self.phase == 'content' and finish in ('stop', 'length'):
            require(bool(self.content) and delta == {} and 'usage' in value, 'terminal event differs')
            usage = value['usage']
            require(type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
                    and all(type(v) is int and v > 0 for v in usage.values())
                    and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
                    'SSE usage differs')
            self.usage, self.finish_reason, self.phase = usage, finish, 'sentinel'
        else:
            raise ValueError('SSE event is repeated or out of order')

    def partial(self):
        require(self.phase == 'content' and bool(self.text) and self.usage is None,
                'cancelled stream must contain content and no terminal event')
        return self.text

    def complete(self):
        require(self.phase == 'done' and not self.pending and self.usage is not None, 'SSE did not complete')
        return {**self.identity, 'object': 'chat.completion',
                'choices': [{'index': 0, 'finish_reason': self.finish_reason,
                             'message': {'role': 'assistant', 'content': self.text}}], 'usage': self.usage}


def raw_response(record):
    require(type(record) is dict and record.get('body_truncated') is False, 'SSE capture exceeded bound')
    require(type(record.get('content_type')) is str and re.fullmatch(
        r'text/event-stream(?:;\s*charset=utf-8)?', record['content_type'], re.I), 'SSE content type differs')
    encoded = record.get('raw_body_base64')
    require(type(encoded) is str and len(encoded) <= 2 * BODY_LIMIT, 'SSE capture exceeds byte bound')
    raw = base64.b64decode(encoded, validate=True)
    require(0 < len(raw) <= BODY_LIMIT, 'SSE response bytes exceed bound')
    return raw


def http_body(record):
    require(record.get('read_error') is None, 'completed SSE response has a read error')
    state = SseState()
    state.feed(raw_response(record))
    return state.complete()


def generation(status, value, document, turn, prompt_tokens):
    require(turn == 2 and type(status) is int and status == 200 and type(value) is dict,
            'only the reuse reply can complete')
    # Reuse exactly the existing ready parser with its independently fixed M16.
    require(value.get('object') == 'chat.completion' and value.get('model') == MODEL_ID, 'reply identity differs')
    choices = value.get('choices')
    require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict, 'reply choice differs')
    choice = choices[0]
    require(type(choice.get('index')) is int and choice['index'] == 0 and choice.get('finish_reason') == 'stop',
            'reuse did not reach EOS')
    message = choice.get('message')
    require(type(message) is dict and set(message) == {'role', 'content'} and message['role'] == 'assistant',
            'reuse reply is not tool-free assistant text')
    final, thinking = answer_helpers().closed_final(message['content'])
    require(type(prompt_tokens) is int and prompt_tokens == prompt_bound(turn), 'reuse prompt bound differs')
    usage = value.get('usage')
    require(type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
            and all(type(v) is int for v in usage.values()) and usage['prompt_tokens'] == prompt_tokens
            and 0 < usage['completion_tokens'] < fixture(document, turn)['max_tokens']
            and usage['total_tokens'] == prompt_tokens + usage['completion_tokens'] <= ceiling(turn),
            'reuse token usage exceeds fixed bound')
    bare = final[:-1] if final.endswith('.') else final
    quality = {'correctness': 'unassessed', 'format': 'fail'}
    if re.fullmatch('[A-Za-z]+', bare):
        quality = {'correctness': 'pass' if bare.casefold() == 'ready' else 'fail', 'format': 'pass'}
    return {'fixture_id': fixture(document, turn)['id'], 'text': message['content'], 'final_answer': final,
            'thinking_present': thinking, 'finish_reason': 'stop', 'usage': usage,
            'evaluated_positions': usage['total_tokens'], 'quality': quality}


def measurement(root, turn, request, identities):
    stem = f'cpu-reference-turn-{turn}'
    path = root / (stem + '-request.json')
    equal(read_json(path), request, 'metadata request differs from UI Stop recipe')
    record = read_json(root / (stem + '-measurement.json'))
    stdout, stderr = bounded_bytes(root / (stem + '-inspector.log')), bounded_bytes(root / (stem + '-inspector-stderr.log'))
    expected = strict_json(stdout)
    require(type(record.get('pid')) is int and record['pid'] > 0 and type(record.get('exit_code')) is int
            and record['exit_code'] == 0 and record.get('timed_out') is False and stderr == b'', 'inspector failed')
    equal(record.get('command'), [str(root / 'dependencies/prompt-inspect'), str(model_path(root)), str(path)],
          'metadata inspector command differs')
    equal(record.get('result'), expected, 'metadata result differs from captured output')
    require(record.get('inspector_sha256') == identities['dependencies']['prompt-inspect']
            and record.get('request_sha256') == digest(path) and record.get('model_sha256') == identities['checkpoint_sha256']
            and record.get('stdout_sha256') == digest(root / (stem + '-inspector.log'))
            and record.get('stderr_sha256') == digest(root / (stem + '-inspector-stderr.log')), 'metadata hash differs')
    elapsed = record.get('elapsed_seconds')
    require(type(elapsed) in (float, int) and math.isfinite(elapsed) and 0 <= elapsed <= 30, 'inspector exceeded deadline')
    require(type(expected) is dict and set(expected) == {'execution', 'tokenization', 'checkpoint', 'requests'}
            and expected['execution'] == 'not_requested' and expected['tokenization'] == 'exact_checkpoint_metadata'
            and expected['checkpoint'] == str(model_path(root)), 'inspector identity differs')
    rows = expected['requests']
    require(type(rows) is list and len(rows) == 1 and type(rows[0]) is dict, 'metadata row count differs')
    row = rows[0]
    equal(set(row), {'request', 'status', 'message_count', 'tool_count', 'rendered_prompt_bytes', 'prompt_tokens',
                     'output_tokens', 'minimum_evaluated_positions', 'output_budget_source'}, 'metadata schema differs')
    prompt = prompt_bound(turn)
    require(type(row['prompt_tokens']) is int and row['prompt_tokens'] == prompt and row['request'] == str(path)
            and row['status'] == 'measured' and type(row['message_count']) is int and row['message_count'] == 2
            and type(row['tool_count']) is int and row['tool_count'] == 0 and type(row['rendered_prompt_bytes']) is int
            and 0 < row['rendered_prompt_bytes'] <= 65536 and type(row['output_tokens']) is int
            and row['output_tokens'] == request['max_tokens'] and type(row['minimum_evaluated_positions']) is int
            and row['minimum_evaluated_positions'] == prompt + request['max_tokens'] - 1 == ceiling(turn)
            and row['output_budget_source'] == 'explicit', 'metadata prompt exceeds fixed UI Stop bounds')
    return prompt


def identity(root):
    try:
        from .ui_client import client_identity
    except ImportError:
        from ui_client import client_identity
    contract = approved_contract(root)
    require(contract['profile'] == PROFILE and contract.get('stream_client') == 'antimatter-ui'
            and 'runtime_phase_diagnostics' not in contract, 'UI Stop requires its phase-off contract')
    gate_identity(root)
    result = answer_helpers().identity(root)
    result['dependencies'] = dependency_identity(root)
    client = client_identity(root)
    require('prompt-inspect' in result['dependencies'] and client is not None, 'frozen inspector and UI required')
    if 'client_identity' in result:
        equal(result['client_identity'], client, 'UI identity dispatch differs')
    result['client_identity'] = client
    return result


def validate_server(root, mode, child=None):
    record = read_json(root / (mode + '-server-process.json'))
    require(type(record.get('pid')) is int and record['pid'] > 0 and type(record.get('exit_code')) is int
            and record['exit_code'] == 0 and record.get('idle_sigterm_exit') is True, 'server did not terminate cleanly')
    require(type(record.get('start_time_ticks')) is int and record['start_time_ticks'] > 0
            and record.get('no_new_privileges') is True, 'server process identity differs')
    group = record.get('cgroup')
    require(type(group) is str and Path(group).is_relative_to('/sys/fs/cgroup') and '..' not in Path(group).parts,
            'server cgroup differs')
    if child is not None:
        equal(group, child['cgroup'], 'server left the worker cgroup')
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    info = raw_control(root, mode, 'identity', 'GET', '/v1/beholdd/info', None)
    for key, value in {'service': 'beholdd', 'runtime': 'native', 'generation': 2,
                       'execution_backends': ['cpu'] if backend == 'cpu' else ['cpu', 'hag']}.items():
        equal(info.get(key), value, 'server identity differs: ' + key)
    labels = [('cold', False), ('after-turn-2', True), ('unloaded', False)]
    if mode == 'answers':
        labels.insert(1, ('after-turn-1', True))
    for label, loaded in labels:
        value = raw_control(root, mode, label + '-model', 'GET', '/v1/models', None)
        require(type(value.get('data')) is list and len(value['data']) == 1, 'model list differs')
        model = value['data'][0]
        require(model.get('id') == MODEL_ID and model.get('loaded') is loaded, 'model residency differs')
        equal(model.get('execution_limits', {}).get('max_evaluated_tokens'), 2048 if backend == 'hag' else None,
              'model capacity differs')
    for index in (1, 2):
        equal(raw_control(root, mode, f'unload-{index}', 'POST', f'/v1/models/{MODEL_ID}/unload', {}),
              {'id': MODEL_ID, 'loaded': False}, 'idempotent unload differs')
    return record


def server_processes(root, mode):
    if mode == 'self-check':
        return []
    record = validate_server(root, mode, read_json(root / (mode + '-child-result.json')))
    return [{'pid': record['pid'], 'exit_code': record['exit_code']}]


def audit_event(value, turn, evaluated=None):
    require(type(value) is dict and set(value) == {'request_sequence', 'evaluated_positions', 'outcome', 'reset_ok', 'poisoned'},
            'native audit schema differs')
    equal({key: value[key] for key in ('request_sequence', 'outcome', 'reset_ok', 'poisoned')},
          {'request_sequence': turn, 'outcome': 'cancelled' if turn == 1 else 'success',
           'reset_ok': True, 'poisoned': False}, 'native cancellation/reset outcome differs')
    positions = value['evaluated_positions']
    require(type(positions) is int and prompt_bound(turn) <= positions <= ceiling(turn), 'native evaluated bound differs')
    if evaluated is not None:
        equal(positions, evaluated, 'native evaluation count differs from reuse usage')
    return value


def visible_count(text):
    """Do not treat digits inside closed or unfinished thinking as visible output."""
    return re.search(r'[0-9]', re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.S)) is not None


def cancellation_receipt(root):
    record = read_json(root / 'answers-turn-1-cancellation.json', 4 * BODY_LIMIT)
    keys = {'schema', 'clock', 'started', 'first_content_forwarded', 'downstream_disconnect', 'upstream_shutdown',
            'audit_observed', 'warm_residency_observed', 'disconnect_kind', 'shutdown', 'server_before', 'server_after',
            'forwarded_raw_base64', 'forwarded_sha256', 'captured_sha256', 'audit_event'}
    require(type(record) is dict and set(record) == keys, 'cancellation receipt schema differs')
    equal({key: record[key] for key in ('schema', 'clock', 'shutdown')},
          {'schema': 1, 'clock': 'monotonic_seconds', 'shutdown': {'how': 'SHUT_RDWR', 'succeeded': True}},
          'actual upstream socket shutdown not proven')
    require(record['disconnect_kind'] in ('eof', 'reset'), 'downstream socket did not disconnect')
    times = [record[key] for key in ('started', 'first_content_forwarded', 'downstream_disconnect', 'upstream_shutdown',
                                    'audit_observed', 'warm_residency_observed')]
    require(all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in times)
            and all(a <= b for a, b in zip(times, times[1:])) and times[-1] - times[0] <= 240,
            'Stop transport/reset clocks are out of order or exceed deadline')
    require(times[3] - times[2] <= 1, 'upstream cancellation was not propagated promptly')
    server = read_json(root / 'answers-server-process.json')
    expected = {key: server[key] for key in ('pid', 'start_time_ticks')}
    equal(record['server_before'], expected, 'cancellation began on another server')
    equal(record['server_after'], expected, 'server restarted after cancellation')
    audit_event(record['audit_event'], 1)
    response = read_json(root / 'answers-turn-1-response.json', 2 * BODY_LIMIT)
    require(response.get('read_error') in (None, 'OSError', 'ConnectionResetError', 'IncompleteRead', 'ValueError'),
            'unexpected upstream read error')
    captured = raw_response(response)
    equal(hashlib.sha256(captured).hexdigest(), record['captured_sha256'], 'cancelled capture hash differs')
    encoded = record['forwarded_raw_base64']
    require(type(encoded) is str and len(encoded) <= 2 * BODY_LIMIT, 'forwarded stream exceeds bound')
    forwarded = base64.b64decode(encoded, validate=True)
    require(bool(forwarded) and captured.startswith(forwarded), 'forwarded SSE is not the captured raw prefix')
    equal(hashlib.sha256(forwarded).hexdigest(), record['forwarded_sha256'], 'forwarded SSE hash differs')
    state, wire = SseState(), SseState()
    state.feed(captured)
    state.partial()
    wire.feed(forwarded)
    text = wire.partial()
    require(not wire.pending, 'forwarded stream contains an incomplete SSE event')
    # Require actual count-up content, not only role, thinking markup, or whitespace.
    require(visible_count(text), 'Stop preceded visible numeric content')
    return record


def listener(root):
    try:
        from .ui_client import client_host
    except ImportError:
        from ui_client import client_host
    approved_contract(root)
    record = read_json(root / 'answers-ui-listener.json')
    expected = {'schema': 1, 'profile': PROFILE, 'host': '127.0.0.1', 'request_count': 2, 'forwarded_count': 2,
                'response_count': 1, 'cancelled_count': 1, 'rejected_count': 0, 'closed': True,
                'client_identity_sha256': digest(root / 'client/identity.json'), 'delivery': DELIVERY}
    require(type(record) is dict and set(record) == {*expected, 'port', 'pid', 'boot_id', 'run_id',
                                                    'empty_connection_count', 'empty_connections'},
            'UI Stop listener schema differs')
    equal({key: record[key] for key in expected}, expected, 'listener did not cancel once and complete once')
    count, empty = record['empty_connection_count'], record['empty_connections']
    require(type(count) is int and 0 <= count <= MAX_EMPTY_CONNECTIONS
            and type(empty) is list and len(empty) == count, 'empty connection budget/count differs')
    previous = 0
    previous_request = 0
    for sequence, row in enumerate(empty, 1):
        require(type(row) is dict and set(row) == {'sequence', 'reason', 'received_bytes',
                'after_request_count', 'observed_monotonic'}, 'empty connection receipt schema differs')
        require(type(row['sequence']) is int and row['sequence'] == sequence
                and row['reason'] in ('eof', 'reset') and type(row['received_bytes']) is int
                and row['received_bytes'] == 0 and type(row['after_request_count']) is int
                and previous_request <= row['after_request_count'] <= 1,
                'nonempty or unbounded connection was ignored')
        observed = row['observed_monotonic']
        require(type(observed) in (int, float) and math.isfinite(observed) and observed > 0
                and observed >= previous, 'empty connection clock differs')
        next_turn = row['after_request_count'] + 1
        next_admission = read_json(root / f'answers-turn-{next_turn}-admission.json')
        require(observed <= next_admission['request_received_monotonic'],
                'empty connection occurred after its next HTTP admission')
        if row['after_request_count'] == 1:
            require(observed >= cancellation_receipt(root)['warm_residency_observed'],
                    'post-Stop empty connection preceded verified reset')
        previous, previous_request = observed, row['after_request_count']
    require(type(record['port']) is int and 1024 <= record['port'] <= 65535
            and type(record['pid']) is int and record['pid'] > 0, 'listener identity differs')
    for key in ('boot_id', 'run_id'):
        require(type(record[key]) is str and str(uuid.UUID(record[key])) == record[key], 'listener UUID differs')
    require(uuid.UUID(record['run_id']).version == 4, 'listener run is not UUID4')
    keys = ('schema', 'profile', 'host', 'port', 'pid', 'boot_id', 'run_id', 'client_identity_sha256', 'delivery')
    equal(read_json(root / 'answers-ui-ready.json'), {key: record[key] for key in keys}, 'listener readiness changed')
    client_host(root, read_json(root / 'client/identity.json'), record)
    return record


def admission(root, turn, request):
    record = read_json(root / f'answers-turn-{turn}-admission.json', 4 * BODY_LIMIT)
    keys = {'turn', 'request_raw_base64', 'request_raw_sha256', 'forwarded_raw_base64', 'forwarded_raw_sha256',
            'request_received_monotonic', 'response_body_sha256', 'reset_verified', 'parity_verified'}
    require(type(record) is dict and set(record) == keys, 'Stop admission schema differs')
    equal({key: record[key] for key in ('turn', 'reset_verified', 'parity_verified')},
          {'turn': turn, 'reset_verified': True, 'parity_verified': turn == 2}, 'Stop admission state differs')
    values = []
    for prefix in ('request', 'forwarded'):
        encoded = record[prefix + '_raw_base64']
        require(type(encoded) is str and len(encoded) <= 2 * BODY_LIMIT, 'request capture exceeds bound')
        raw = base64.b64decode(encoded, validate=True)
        require(0 < len(raw) <= 65536, 'request exceeds bound')
        equal(hashlib.sha256(raw).hexdigest(), record[prefix + '_raw_sha256'], 'request byte hash differs')
        equal(strict_json(raw), request, 'captured request differs')
        values.append(raw)
    equal(values[0], values[1], 'forwarded request bytes changed')
    received = record['request_received_monotonic']
    require(type(received) in (int, float) and math.isfinite(received) and received > 0, 'admission clock differs')
    if turn == 1:
        cancelled = cancellation_receipt(root)
        equal(record['response_body_sha256'], cancelled['forwarded_sha256'], 'cancelled delivery bytes differ')
        require(received <= cancelled['started'], 'stream started before admission')
    else:
        require(received >= cancellation_receipt(root)['warm_residency_observed'], 'reuse admitted before cancellation/reset')
        response = read_json(root / 'answers-turn-2-response.json', 2 * BODY_LIMIT)
        equal(record['response_body_sha256'], hashlib.sha256(raw_response(response)).hexdigest(), 'reuse bytes differ')
    return record


def lineage(root, mode, turn):
    value = {'turn': turn, 'request_sha256': digest(root / f'{mode}-turn-{turn}-request.json'), 'history_parent': None,
             'measurement_sha256': digest(root / f'cpu-reference-turn-{turn}-measurement.json')}
    if mode == 'answers':
        value.update(cpu_reference_sha256=digest(root / 'cpu-reference.json'), fresh_conversation=True,
                     cancellation_sha256=digest(root / 'answers-turn-1-cancellation.json') if turn == 2 else None,
                     cancellation_reset_before_reuse=turn == 2)
    return value


def events(root, mode, reuse):
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    raw = bounded_bytes(root / f'{mode}-{backend}-server.log', LOG_LIMIT)
    require(not raw or raw.endswith(b'\n'), 'server log is incomplete')
    require(not any(line.startswith(b'HAG_SERVING_PHASE ') for line in raw.splitlines()), 'phase diagnostics enabled')
    observed = [strict_json(line[len(b'HAG_SERVING_AUDIT '):]) for line in raw.splitlines()
                if line.startswith(b'HAG_SERVING_AUDIT ')]
    if mode == 'cpu-reference':
        equal(observed, [], 'CPU stage emitted native events')
    else:
        require(len(observed) == 2, 'native request count differs')
        audit_event(observed[0], 1)
        audit_event(observed[1], 2, reuse['evaluated_positions'])
        equal(observed[0], cancellation_receipt(root)['audit_event'], 'cancel reset receipt differs from raw server audit')
    return observed


def reconstruct(root, mode, identities, child=None):
    require(mode in MODES, 'unknown UI Stop stage')
    document = recipe(root)
    validate_server(root, mode, child)
    cpu = None
    if mode == 'answers':
        cpu = summarize_stage(root, 'cpu-reference')
        result = read_json(root / 'cpu-reference-result.json')
        require(result.get('qualified') is True and result.get('failures') == []
                and type(result.get('exit_code')) is int and result['exit_code'] == 0, 'CPU prerequisite failed')
        equal(result.get('probe'), cpu, 'CPU prerequisite report differs')
    turns = []
    for turn in (1, 2):
        request = request_body(document, turn)
        stem = f'{mode}-turn-{turn}'
        equal(read_json(root / (stem + '-request.json')), request, 'wire request differs from fresh conversation recipe')
        prompt = measurement(root, turn, request, identities)
        binding = lineage(root, mode, turn)
        equal(read_json(root / (stem + '-lineage.json')), binding, 'Stop admission lineage differs')
        turns.append({**binding, 'prompt_tokens': prompt, 'executed': mode == 'answers' or turn == 2})
        if mode == 'answers':
            admit_request(turn, request, cpu, document)
            admission(root, turn, request)
        if turn == 1:
            if mode == 'answers':
                first = read_json(root / (stem + '-response.json'), 2 * BODY_LIMIT)
                equal({key: first.get(key) for key in ('method', 'route', 'request', 'status')},
                      {'method': 'POST', 'route': '/v1/chat/completions', 'request': request, 'status': 200},
                      'cancelled backend request differs')
                cancellation_receipt(root)
            else:
                require(not (root / (stem + '-response.json')).exists(), 'CPU must not execute cancelled workload')
            continue
        response = read_json(root / (stem + '-response.json'), 2 * BODY_LIMIT)
        equal({key: response.get(key) for key in ('method', 'route', 'request', 'status')},
              {'method': 'POST', 'route': '/v1/chat/completions', 'request': request, 'status': 200},
              'reuse backend request differs')
        reuse = generation(response['status'], http_body(response), document, turn, prompt)
        turns[-1]['response_sha256'] = digest(root / (stem + '-response.json'))
        if mode == 'answers':
            equal(reuse, cpu['answers'][0], 'reuse full text/usage differs from CPU')
    if mode == 'answers':
        listener(root)
    return [reuse], turns, events(root, mode, reuse)


def summarize_stage(root, mode, *, child=None):
    ids = identity(root)
    answers, turns, audit = reconstruct(root, mode, ids, child)
    reference = {'schema': 1, 'profile': PROFILE, 'identities': ids, 'capacity': 2048,
                 'answers': answers if mode == 'cpu-reference' else read_json(root / 'cpu-reference.json')['answers'],
                 'turns': turns if mode == 'cpu-reference' else read_json(root / 'cpu-reference.json')['turns'],
                 'cpu_exit_code': 0, 'final_unloaded': True}
    equal(read_json(root / 'cpu-reference.json'), reference, 'CPU reference artifact differs')
    evaluated = answers[0]['evaluated_positions'] if mode == 'cpu-reference' else sum(e['evaluated_positions'] for e in audit)
    maximum = profile_spec(PROFILE)['stage_evaluation_ceilings'][mode]
    require(evaluated <= maximum, 'UI Stop stage evaluation ceiling exceeded')
    report = {'stage': mode, 'execution': 'cpu-reference' if mode == 'cpu-reference' else 'native-hag-answers',
              'profile': PROFILE, 'qualification_kind': 'ui_stop_reuse_and_backend_parity', 'require_semantic_match': False,
              'quality_is_diagnostic': True, 'capacity': 2048, 'requests': 1 if mode == 'cpu-reference' else 2,
              'fixture_ids': [fixture(recipe(root), turn)['id'] for turn in ((2,) if mode == 'cpu-reference' else (1, 2))],
              'answers': answers, 'turns': turns, 'evaluated_positions': evaluated,
              'maximum_evaluated_positions': maximum, 'all_eos': mode == 'cpu-reference', 'closed_answers': True,
              'generated_history': False, 'final_unloaded': True, 'server_exit_code': 0,
              'identities': ids, 'client_identity': ids['client_identity'], 'reference_sha256': digest(root / 'cpu-reference.json'),
              'audit_events': audit, 'live_ui_qualified': False, 'client_evidence_pending': True}
    if mode == 'cpu-reference':
        report['cpu_exit_code'] = 0
    else:
        report.update(hag_exit_code=0, reuse_exact_cpu_parity=True, cancellation_reset_before_reuse=True,
                      cpu_reference_result_sha256=digest(root / 'cpu-reference-result.json'),
                      client_chat_requests=0, client_stream_requests=2, transport=DELIVERY,
                      backend_cancellation_qualified=True, live_cancellation_qualified=False, listener=listener(root),
                      cancellation=cancellation_receipt(root))
    return report


def validate_stage(root, mode, text):
    report = summarize_stage(root, mode, child=read_json(root / (mode + '-child-result.json')))
    equal(read_json(root / (mode + '-report.json')), report, 'Stop report differs from raw evidence')
    rows = [strict_json(line[len(PREFIX):]) for line in text.splitlines() if line.startswith(PREFIX)]
    equal(rows, [report], 'expected one matching UI Stop report')
    equal(re.findall(r'^PASS hag-ui-stop stage=(\S+) qualified=(\S+)$', text, re.M), [(mode, 'true')],
          'UI Stop stage must pass exactly once')
    return report


def broker_records(root):
    return [{**read_json(root / f'answers-turn-{turn}-response.json', 2 * BODY_LIMIT), 'turn': turn} for turn in (1, 2)]


def evidence_names(mode):
    if mode == 'self-check':
        return []
    require(mode in MODES, 'unknown UI Stop stage')
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    names = [f'{mode}-{backend}-server.log', mode + '-server-process.json', mode + '-report.json']
    labels = ['identity', 'cold-model', 'after-turn-2-model', 'unload-1', 'unload-2', 'unloaded-model']
    if mode == 'answers':
        labels.append('after-turn-1-model')
    names += [f'{mode}-{label}.json' for label in labels]
    for turn in (1, 2):
        stem = f'{mode}-turn-{turn}'
        names += [stem + '-request.json', stem + '-lineage.json']
        if mode == 'cpu-reference':
            names += [stem + '-measurement.json', stem + '-inspector.log', stem + '-inspector-stderr.log']
        else:
            names += [stem + '-admission.json', stem + '-response.json']
    names += ['cpu-reference-turn-2-response.json', 'cpu-reference.json'] if mode == 'cpu-reference' else [
        'answers-turn-1-cancellation.json', 'answers-ui-ready.json', 'answers-ui-listener.json']
    return names
