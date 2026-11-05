"""Bounded raw-evidence replay for the C2048 TCP lifecycle profile.

Importing this module never starts a process or opens a model/device. Truncated
length responses deliberately do not pass through complete-answer validators.
"""
import base64
import re

from gate_common import (read_json, regular_reader, strict_json, digest, approved_contract,
                         EXPECTED_LIMITS, STOP_CGROUP_BYTES, clean_memory, require_process_headroom,
                         private_cgroup_attestation)
from profiles import LIFECYCLE_PROFILE, profile_spec, _typed_equal

MODEL_ID = 'hag-context-qwen3-0.6b'
BODY_LIMIT = 1024 * 1024
AUDIT_PREFIX = b'HAG_SERVING_AUDIT '
SNAPSHOTS = {
    'cpu-reference': ('cold', 'generated', 'unloaded'),
    'answers': ('cold', 'generated', 'streamed', 'cancelled', 'reused', 'unloaded', 'reloaded', 'final-unloaded'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(_typed_equal(actual, expected), message)


def bounded_bytes(path, limit=BODY_LIMIT):
    with regular_reader(path, limit) as stream:
        data = stream.read(limit + 1)
    require(len(data) <= limit, 'evidence grew beyond its limit')
    return data


def request_body(stream=False, cancel=False):
    return {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': 'Hello'}],
            'temperature': 0, 'max_tokens': 16 if cancel else 1, 'stream': stream}


def generation(status, value):
    require(type(status) is int and status == 200 and type(value) is dict, 'completion failed')
    require(value.get('object') == 'chat.completion' and value.get('model') == MODEL_ID,
            'completion identity differs')
    choices = value.get('choices')
    require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
            'completion choices differ')
    choice = choices[0]
    require(type(choice.get('index')) is int and choice['index'] == 0
            and choice.get('finish_reason') == 'length', 'one-token length finish required')
    message = choice.get('message')
    require(type(message) is dict and message.get('role') == 'assistant'
            and type(message.get('content')) is str and message['content']
            and not message.get('tool_calls'), 'completion text differs')
    usage = {'prompt_tokens': 9, 'completion_tokens': 1, 'total_tokens': 10}
    equal(value.get('usage'), usage, 'completion token accounting differs')
    return {'text': message['content'], 'finish_reason': 'length', 'usage': usage}


class SseResult:
    """Strict finite SSE parser shared by live capture and offline raw replay."""
    def __init__(self):
        self.content = ''
        self.roles = self.finishes = self.done = 0
        self.response_id = None

    def line(self, raw):
        require(len(raw) <= BODY_LIMIT and raw.endswith(b'\n'), 'incomplete or oversized SSE line')
        text = (raw[:-2] if raw.endswith(b'\r\n') else raw[:-1]).decode('utf-8')
        require('\r' not in text and '\n' not in text, 'embedded SSE line terminator')
        if not text:
            return
        require(self.done == 0, 'SSE data after DONE')
        if text.startswith(':'):
            return
        require(text.startswith('data: ') and self.done == 0, 'unexpected SSE field or data after DONE')
        data = text[6:]
        if data == '[DONE]':
            require(self.roles == self.finishes == 1, 'premature DONE')
            self.done += 1
            return
        value = strict_json(data)
        require(type(value) is dict and value.get('object') == 'chat.completion.chunk'
                and value.get('model') == MODEL_ID, 'SSE identity differs')
        identifier = value.get('id')
        require(type(identifier) is str and identifier, 'missing SSE response ID')
        if self.response_id is None:
            self.response_id = identifier
        require(identifier == self.response_id, 'mixed SSE response IDs')
        choices = value.get('choices')
        require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict
                and type(choices[0].get('index')) is int and choices[0]['index'] == 0,
                'SSE choices differ')
        choice = choices[0]
        delta = choice.get('delta')
        require(type(delta) is dict and set(delta) <= {'role', 'content'}, 'invalid SSE delta')
        if 'role' in delta:
            require(delta['role'] == 'assistant' and self.roles == 0 and not self.content,
                    'misplaced SSE role')
            self.roles += 1
        if 'content' in delta:
            require(self.roles == 1 and self.finishes == 0 and type(delta['content']) is str,
                    'misplaced SSE content')
            self.content += delta['content']
        if choice.get('finish_reason') is not None:
            require(self.roles == 1 and choice['finish_reason'] == 'length' and self.finishes == 0,
                    'SSE finish reason differs')
            self.finishes += 1

    def finish(self, reference, cancel=False):
        if cancel:
            require(self.roles == 1 and self.content and self.finishes == self.done == 0,
                    'stream finished before cancellation')
            require(reference['text'].startswith(self.content), 'cancelled first content differs from CPU')
        else:
            require(self.roles == self.finishes == self.done == 1 and self.content == reference['text'],
                    'SSE content or termination differs from CPU')


def audit_events(path, final=True):
    data = bounded_bytes(path, 2 * BODY_LIMIT)
    if final:
        require(not data or data.endswith(b'\n'), 'incomplete final server log')
    events = []
    for raw in data.splitlines(keepends=True):
        if not raw.endswith(b'\n'):
            continue  # Polling may catch a write between payload and newline.
        if not raw.startswith(AUDIT_PREFIX):
            continue
        event = strict_json(raw[len(AUDIT_PREFIX):])
        require(type(event) is dict and set(event) ==
                {'request_sequence', 'evaluated_positions', 'outcome', 'reset_ok', 'poisoned'},
                'audit schema differs')
        require(type(event['request_sequence']) is int and event['request_sequence'] == len(events) + 1,
                'audit sequence differs')
        require(type(event['evaluated_positions']) is int and 0 <= event['evaluated_positions'] <= 24,
                'audit exceeds lifecycle request ceiling')
        require(event['outcome'] in ('success', 'cancelled') and event['reset_ok'] is True
                and event['poisoned'] is False, 'unhealthy audit outcome')
        events.append(event)
    return events


def validate_events(events):
    require(len(events) == 5, 'expected exactly five native requests')
    for index, event in enumerate(events):
        require(event['outcome'] == ('cancelled' if index == 2 else 'success'),
                'cancellation was missing or completed naturally')
        count = event['evaluated_positions']
        require(9 <= count <= 24 if index == 2 else count == 9, 'request evaluation count differs')
    total = sum(event['evaluated_positions'] for event in events)
    require(45 <= total <= 60, 'native lifecycle evaluation ceiling exceeded')
    return total


def http_artifact(root, stem, method, route, body, status=200):
    record = read_json(root / (stem + '-http.json'), 2 * BODY_LIMIT)
    equal({k: record.get(k) for k in ('method', 'route', 'request', 'status')},
          {'method': method, 'route': route, 'request': body, 'status': status}, 'HTTP request/status differs')
    require(type(record.get('content_type')) is str and record['content_type'].startswith('application/json'),
            'HTTP content type differs')
    require(type(record.get('body_base64')) is str, 'missing raw HTTP body')
    require(record.get('truncated') is False and 'read_error' in record and record['read_error'] is None,
            'HTTP body was truncated or failed during capture')
    data = base64.b64decode(record['body_base64'], validate=True)
    require(len(data) <= BODY_LIMIT, 'oversized raw HTTP body')
    return strict_json(data)


def stream_artifact(root, stem, reference, cancel=False):
    header = read_json(root / (stem + '-headers.json'))
    equal({k: header.get(k) for k in ('request', 'status')},
          {'request': request_body(True, cancel), 'status': 200}, 'stream request/status differs')
    require(type(header.get('content_type')) is str and header['content_type'].startswith('text/event-stream'),
            'stream content type differs')
    state = SseResult()
    for line in bounded_bytes(root / (stem + '.sse')).splitlines(keepends=True):
        state.line(line)
    state.finish(reference, cancel)
    return state


def model_artifact(root, stem, loaded, backend):
    value = http_artifact(root, stem, 'GET', '/v1/models', None)
    require(type(value) is dict and type(value.get('data')) is list and len(value['data']) == 1,
            'model list differs')
    model = value['data'][0]
    require(model.get('id') == MODEL_ID and model.get('loaded') is loaded, 'model residency differs')
    limit = model.get('execution_limits', {}).get('max_evaluated_tokens')
    equal(limit, 2048 if backend == 'hag' else None, 'model capacity differs')


def identity(root):
    manifest = read_json(root / 'expected-manifest.json')
    return {'binary_sha256': manifest['binaries']['beholdd'],
            'source_manifest_sha256': manifest['source_manifest_sha256'],
            'checkpoint_sha256': manifest['model_sha256'],
            'fixture_sha256': manifest['gate_files']['answer-fixtures.json'],
            'gate_files': manifest['gate_files']}


def validate_record(record, mode, child=None):
    require(type(record) is dict and type(record.get('server_pid')) is int and record['server_pid'] > 0,
            'invalid server PID')
    equal(record.get('server_exit_code'), 0, 'server did not exit cleanly')
    require(record.get('mode') == mode, 'server record stage differs')
    snapshots = record.get('snapshots')
    require(type(snapshots) is list and [s.get('label') for s in snapshots] == list(SNAPSHOTS[mode]),
            'lifecycle snapshot order differs')
    initial = snapshots[0]
    require(type(initial.get('start_time_ticks')) is int and initial['start_time_ticks'] > 0
            and re.fullmatch(r'[0-9a-f-]{36}', initial.get('boot_id', '')), 'process start identity differs')
    for snapshot in snapshots:
        for key in ('pid', 'start_time_ticks', 'boot_id', 'cgroup'):
            equal(snapshot.get(key), initial[key], 'reload changed server identity: ' + key)
        equal(snapshot['pid'], record['server_pid'], 'snapshot PID differs')
        equal(snapshot.get('limits'), EXPECTED_LIMITS, 'snapshot limits differ')
        require(snapshot.get('no_new_privileges') is True, 'server privileges differ')
        memory = snapshot['memory']
        for key in ('memory_current_bytes', 'memory_peak_bytes', 'swap_current_bytes'):
            require(type(memory.get(key)) is int and memory[key] >= 0, 'invalid memory snapshot counter')
        require(memory['memory_current_bytes'] <= memory['memory_peak_bytes'] < STOP_CGROUP_BYTES,
                'memory snapshot exceeded early-stop threshold')
        events = memory.get('memory_events')
        require(type(events) is dict and all(type(v) is int and v >= 0 for v in events.values())
                and all(events.get(key) == 0 for key in ('high', 'max', 'oom', 'oom_kill')),
                'invalid memory pressure counters')
        require(memory.get('swap_peak_bytes') is None
                or (type(memory['swap_peak_bytes']) is int and memory['swap_peak_bytes'] == 0),
                'invalid swap peak')
        require(clean_memory(memory, memory), 'lifecycle snapshot contains pressure or swap')
        require_process_headroom(memory)
        attested = private_cgroup_attestation(snapshot['cgroup'], snapshot['limits'],
                                              'NoNewPrivs:\t1\n', '0', memory)
        equal(snapshot.get('private_cgroup_attestation'), attested if mode == 'answers' else None,
              'lifecycle private cgroup attestation differs')
        if child is not None:
            equal(snapshot['cgroup'], child['cgroup'], 'server left the dedicated worker cgroup')
            equal(snapshot['private_cgroup_attestation'], child.get('private_cgroup_attestation'),
                  'worker/server private cgroup attestation differs')
    if mode == 'answers':
        equal(record.get('socket_shutdown'), {'how': 'SHUT_RDWR', 'succeeded': True},
              'missing actual bidirectional socket shutdown')
    else:
        equal(record.get('socket_shutdown'), None, 'CPU unexpectedly recorded cancellation')


def server_processes(root, mode):
    if mode == 'self-check':
        return []
    record = read_json(root / ('lifecycle-' + mode + '-record.json'))
    validate_record(record, mode, read_json(root / (mode + '-child-result.json')))
    return [{'pid': record['server_pid'], 'exit_code': record['server_exit_code']}]


def evidence_names(mode):
    if mode == 'self-check':
        return []
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    names = [mode + '-report.json', mode + '-' + backend + '-server.log',
             'lifecycle-' + mode + '-record.json', mode + '-hello-http.json', mode + '-identity-http.json',
             mode + '-cold-model-http.json', mode + '-unload-1-http.json', mode + '-unload-2-http.json',
             mode + '-unloaded-model-http.json']
    if mode == 'cpu-reference':
        names.append('cpu-reference.json')
    else:
        names += ['answers-' + stem for stem in (
            'stream-headers.json', 'stream.sse', 'cancel-headers.json', 'cancel.sse',
            'busy-unload-http.json', 'reuse-http.json', 'reload-http.json',
            'loaded-model-http.json', 'reloaded-model-http.json', 'final-unload-1-http.json',
            'final-unload-2-http.json', 'final-unloaded-model-http.json')]
    return names


def summarize_stage(root, mode, *, child=None):
    contract = approved_contract(root)
    require(contract['profile'] == LIFECYCLE_PROFILE and mode in ('cpu-reference', 'answers'),
            'not a lifecycle execution stage')
    ids = identity(root)
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    record = read_json(root / ('lifecycle-' + mode + '-record.json'))
    validate_record(record, mode, child)
    info = http_artifact(root, mode + '-identity', 'GET', '/v1/beholdd/info', None)
    for key, expected in {'runtime': 'native', 'service': 'beholdd', 'generation': 2,
                          'execution_backends': ['cpu', 'hag'] if backend == 'hag' else ['cpu']}.items():
        equal(info.get(key), expected, 'runtime identity differs: ' + key)
    model_artifact(root, mode + '-cold-model', False, backend)
    first = generation(200, http_artifact(root, mode + '-hello', 'POST', '/v1/chat/completions', request_body()))
    for stem in ('unload', 'final-unload') if mode == 'answers' else ('unload',):
        for number in (1, 2):
            value = http_artifact(root, f'{mode}-{stem}-{number}', 'POST', f'/v1/models/{MODEL_ID}/unload', {})
            equal(value, {'id': MODEL_ID, 'loaded': False}, 'idempotent unload differs')
    model_artifact(root, mode + '-unloaded-model', False, backend)
    artifact = read_json(root / 'cpu-reference.json')
    equal(artifact, {'schema': 1, 'profile': LIFECYCLE_PROFILE, 'identities': ids,
                     'answer': first if mode == 'cpu-reference' else artifact.get('answer'),
                     'cpu_exit_code': 0, 'final_unloaded': True}, 'CPU reference artifact differs')
    events = audit_events(root / (mode + '-' + backend + '-server.log'))
    if mode == 'cpu-reference':
        equal(events, [], 'CPU reference unexpectedly emitted native audit')
        evaluated = 9
    else:
        cpu_report = summarize_stage(root, 'cpu-reference')
        cpu_result = read_json(root / 'cpu-reference-result.json')
        equal(cpu_result.get('probe'), cpu_report, 'CPU prerequisite probe differs')
        require(cpu_result.get('qualified') is True and cpu_result.get('failures') == []
                and type(cpu_result.get('exit_code')) is int and cpu_result['exit_code'] == 0,
                'CPU prerequisite did not qualify')
        reference = artifact['answer']
        equal(first, reference, 'native completion differs from CPU')
        for stem in ('reuse', 'reload'):
            answer = generation(200, http_artifact(root, 'answers-' + stem, 'POST', '/v1/chat/completions', request_body()))
            equal(answer, reference, stem + ' differs from CPU')
        stream_artifact(root, 'answers-stream', reference)
        stream_artifact(root, 'answers-cancel', reference, cancel=True)
        busy = http_artifact(root, 'answers-busy-unload', 'POST', f'/v1/models/{MODEL_ID}/unload', {}, 409)
        require(type(busy) is dict and type(busy.get('error')) is dict
                and busy['error'].get('code') == 'runtime_busy' and 'choices' not in busy,
                'active generation did not refuse unload')
        for stem in ('loaded', 'reloaded'):
            model_artifact(root, 'answers-' + stem + '-model', True, backend)
        model_artifact(root, 'answers-final-unloaded-model', False, backend)
        evaluated = validate_events(events)
    spec = profile_spec(LIFECYCLE_PROFILE)
    report = {'stage': mode, 'execution': 'cpu-reference' if mode == 'cpu-reference' else 'native-hag-answers',
              'profile': LIFECYCLE_PROFILE, 'qualification_kind': spec['qualification_kind'],
              'require_semantic_match': False, 'quality_is_diagnostic': True, 'capacity': 2048,
              'requests': spec['stage_request_counts'][mode], 'evaluated_positions': evaluated,
              'maximum_evaluated_positions': spec['stage_evaluation_ceilings'][mode],
              'final_unloaded': True, 'server_exit_code': 0, 'identities': ids,
              'reference_sha256': digest(root / 'cpu-reference.json'), 'audit_events': events}
    if mode == 'answers':
        report.update(cpu_reference_result_sha256=digest(root / 'cpu-reference-result.json'),
                      nonstream_parity=True, sse_parity=True, busy_unload_refusals=1,
                      socket_disconnect_observed=True, cancellation_reuse=True, unload_reload=True,
                      successful_requests=4, cancelled_requests=1)
    return report


def validate_stage(root, mode, text):
    report = summarize_stage(root, mode, child=read_json(root / (mode + '-child-result.json')))
    equal(read_json(root / (mode + '-report.json')), report, 'lifecycle report differs from raw evidence')
    records = [strict_json(line[len('HAG_LIFECYCLE_REPORT '):]) for line in text.splitlines()
               if line.startswith('HAG_LIFECYCLE_REPORT ')]
    equal(records, [report], 'expected exactly one matching lifecycle report')
    finals = re.findall(r'^PASS hag-lifecycle stage=(\S+) qualified=(\S+)$', text, re.MULTILINE)
    require(finals == [(mode, 'true')], 'lifecycle stage must pass exactly once')
    return report
