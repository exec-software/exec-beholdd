#!/usr/bin/env python3
"""Frozen answer-profile worker and descriptive warm SSE gates.

No model/GPU access on import or --help. CPU references precede native execution.
Every output is exclusive; failed stages are preserved and never retried here.
Execute the copied worker inside a prepared snapshot under its external supervisor.
"""
import argparse
import base64
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from profiles import fixture_profile, profile_spec, profile_limits, GROUNDED_PROFILE, GROUNDED_PROFILES, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, C2048_PROFILES, profile_deadlines
from answer_quality import quality_assessment
from gate_common import client_identity
from gate_common import PHASE_ENVIRONMENT, approved_contract, phase_diagnostics

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'hag-context-qwen3-0.6b'
CAPACITY = 128
BODY_LIMIT = 1024 * 1024
LOG_LIMIT = 2 * 1024 * 1024
REQUEST_SECONDS = 170
CLIENT_SECONDS = 30
CLIENT_PROCESS_SECONDS = 35
AUDIT_PREFIX = 'HAG_SERVING_AUDIT '
STAGES = ('cpu-reference', 'answers', 'timing')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(text):
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(text, parse_constant=invalid, object_pairs_hook=pairs)


def read_json(path):
    with path.open('rb') as source:
        value = source.read(BODY_LIMIT + 1)
    require(len(value) <= BODY_LIMIT, 'oversized JSON evidence')
    return strict_json(value)


def write_json(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            result.update(chunk)
    return result.hexdigest()


def verify_containment(proc=Path('/proc'), cgroup_base=Path('/sys/fs/cgroup')):
    records = [line[3:] for line in (proc / 'self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(records) == 1 and records[0].startswith('/') and '..' not in Path(records[0]).parts,
            'one absolute unified cgroup is required')
    cgroup = cgroup_base / records[0].lstrip('/')
    expected = {'memory.max': str(2 * 1024 ** 3), 'memory.swap.max': '0',
                'pids.max': '64', 'cpu.max': '75000 100000'}
    require({name: (cgroup / name).read_text().strip() for name in expected} == expected,
            'TCP driver is outside the expected resource limits')
    require(re.search(r'^NoNewPrivs:\s+1$', (proc / 'self/status').read_text(), re.MULTILINE),
            'TCP driver must have no-new-privileges')
    return records[0]



def audit_events(path, capacity=CAPACITY):
    with path.open('rb') as source:
        data = source.read(LOG_LIMIT + 1)
    require(len(data) <= LOG_LIMIT, 'oversized CLI log')
    events = [strict_json(line[len(AUDIT_PREFIX):]) for line in data.decode('utf-8').splitlines()
              if line.startswith(AUDIT_PREFIX)]
    for index, event in enumerate(events, 1):
        require(type(event) is dict and type(event.get('request_sequence')) is int
                and event['request_sequence'] == index, 'audit sequence differs')
        require(type(event.get('evaluated_positions')) is int
                and 0 <= event['evaluated_positions'] <= capacity, 'invalid audit position count')
        require(event.get('outcome') in ('success', 'cancelled') and event.get('reset_ok') is True
                and event.get('poisoned') is False, 'failed or unhealthy HAG request audit')
    return events



def server_phase_environment(inherited, backend, stage, profile, diagnostics=None):
    """Ambient opt-in is never authoritative; only frozen native answers opt in."""
    environment = dict(inherited)
    environment.pop(PHASE_ENVIRONMENT, None)
    contract = {'profile': profile}
    if diagnostics is not None:
        contract['runtime_phase_diagnostics'] = diagnostics
    policy = phase_diagnostics(contract, stage)
    if backend == 'hag' and policy is not None and policy['enabled']:
        environment.update(policy['environment'])
    return environment


class Server:
    def __init__(self, root, backend, stage, capacity=CAPACITY, profile='list-c128', *, phase_diagnostics=None):
        self.profile = profile
        self.stage, self.phase_diagnostics = stage, phase_diagnostics
        self.request_seconds = (600
                                if profile in C2048_PROFILES and backend == 'cpu' else REQUEST_SECONDS)
        self.root, self.backend, self.capacity = root, backend, capacity
        self.process = None
        self.log_path = root / (stage + '-' + backend + '-server.log')
        self.log = None
        self.port = None
        self.exited_cleanly = False

    def __enter__(self):
        environment = server_phase_environment(os.environ, self.backend, self.stage,
                                               self.profile, self.phase_diagnostics)
        with socket.socket() as reservation:
            reservation.bind(('127.0.0.1', 0))
            self.port = reservation.getsockname()[1]
        environment['HAG_TRACE_SUBMISSION'] = '0'
        environment.pop('HAG_SERVING_AUDIT', None)
        attestation = environment.pop('BEHOLDD_HAG_PRIVATE_CGROUP', None)
        if self.backend == 'hag' and self.profile in C2048_PROFILES:
            require(attestation == verify_containment(), 'private cgroup attestation differs from current membership')
            environment['BEHOLDD_HAG_PRIVATE_CGROUP'] = attestation
        if self.backend == 'hag':
            environment['HAG_SERVING_AUDIT'] = '1'
        self.log = self.log_path.open('x')
        try:
            self.process = subprocess.Popen([str(self.root / 'beholdd'), 'serve', '--config',
                                            str(self.root / (self.backend + '-config.json')),
                                            '--host', '127.0.0.1', '--port', str(self.port)],
                                           stdout=self.log, stderr=subprocess.STDOUT, env=environment)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                require(self.process.poll() is None, 'CLI exited before readiness; no port retry')
                if f'beholdd v2 listening on http://127.0.0.1:{self.port}' in bounded_text(self.log_path):
                    try:
                        status, value = self.json('GET', '/health')
                        if status == 200 and value == {'status': 'ok'}:
                            return self
                    except OSError:
                        pass
                time.sleep(.05)
            raise ValueError('CLI readiness deadline exceeded')
        except BaseException:
            self.cleanup()
            raise

    def __exit__(self, *_):
        self.cleanup()

    def cleanup(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.log is not None:
            self.log.close()

    def open(self, method, route, body=None, *, raw=None):
        require(self.process.poll() is None, 'CLI process is not running')
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.request_seconds)
        payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        try:
            connection.request(method, route, body=payload,
                               headers={'Content-Type': 'application/json',
                                        'Connection': 'keep-alive' if body and body.get('stream') else 'close'})
            return connection, connection.getresponse()
        except BaseException:
            connection.close()
            raise

    def json(self, method, route, body=None, *, raw=None):
        connection, response = self.open(method, route, body, raw=raw)
        try:
            require(response.getheader('Content-Type', '').startswith('application/json'),
                    'JSON route returned another content type')
            data = response.read(BODY_LIMIT + 1)
            require(len(data) <= BODY_LIMIT, 'oversized HTTP JSON body')
            return response.status, strict_json(data)
        finally:
            response.close()
            connection.close()

    def model(self, loaded):
        status, value = self.json('GET', '/v1/models')
        require(status == 200 and len(value.get('data', [])) == 1, 'model list differs')
        model = value['data'][0]
        require(model.get('id') == MODEL_ID and model.get('loaded') is loaded, 'model residency differs')
        limit = model.get('execution_limits', {}).get('max_evaluated_tokens')
        require((type(limit) is int and limit == self.capacity) if self.backend == 'hag' else limit is None,
                'public execution limit differs')
        return model

    def identity(self):
        status, value = self.json('GET', '/v1/beholdd/info')
        require(status == 200 and value.get('runtime') == 'native'
                and value.get('service') == 'beholdd' and value.get('generation') == 2
                and value.get('execution_backends') == (['cpu', 'hag'] if self.backend == 'hag' else ['cpu']),
                'runtime identity differs')
        self.model(False)

    def event(self, count):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            events = audit_events(self.log_path, self.capacity)
            require(len(events) <= count, 'unexpected additional HAG request')
            if len(events) == count:
                return events[-1]
            require(self.process.poll() is None, 'CLI exited while waiting for HAG outcome')
            time.sleep(.05)
        raise ValueError('HAG request outcome deadline exceeded')

    def unload(self):
        for _ in range(2):
            status, value = self.json('POST', f'/v1/models/{MODEL_ID}/unload', {})
            require(status == 200 and value == {'id': MODEL_ID, 'loaded': False}, 'unload did not succeed')
        self.model(False)

    def stop(self):
        self.model(False)
        self.process.send_signal(signal.SIGTERM)
        require(self.process.wait(timeout=5) == 0, 'idle CLI SIGTERM did not exit successfully')
        self.exited_cleanly = True


def bounded_text(path):
    with path.open('rb') as source:
        data = source.read(LOG_LIMIT + 1)
    require(len(data) <= LOG_LIMIT, 'oversized CLI log')
    return data.decode('utf-8')


def fixtures_from(value):
    # Profile definitions pin every prompt, role, semantic criterion and bound.
    fixture_profile(value)
    return value['fixtures']


def closed_final(text):
    require(type(text) is str and text.strip(), 'empty answer text')
    require(len(text.encode('utf-8')) <= 65536, 'oversized answer text')
    # Only one exact leading thinking pair is accepted. Stray/malformed/nested
    # tags fail; the full unmodified text remains the parity reference.
    markers = re.findall(r'<\s*/?\s*think\b[^>]*>?', text, re.IGNORECASE)
    present = bool(markers)
    final = text
    if present:
        require(markers == ['<think>', '</think>'] and text.lstrip().startswith('<think>'),
                'thinking tags are stray, nested, repeated or malformed')
        opening = text.index('<think>')
        closing = text.index('</think>')
        require(opening < closing, 'thinking tags are reversed')
        final = text[closing + len('</think>'):]
    final = final.strip()
    require(final, 'closed thinking has no final answer')
    return final, present


def closed_answer(text, semantic):
    require(type(semantic) is dict, 'invalid semantic criterion')
    list_criterion = (semantic == {'kind': 'integer_csv', 'values': list(range(1, 9)), 'allow_terminal_period': True}
                      and all(type(value) is int for value in semantic['values']))
    integer_criterion = (semantic == {'kind': 'integer', 'value': 12, 'allow_terminal_period': True}
                         and type(semantic['value']) is int)
    word_criterion = (semantic == {'kind': 'word', 'value': 'MANGO', 'case_sensitive': False,
                                  'allow_terminal_period': True}
                      and semantic['case_sensitive'] is False)
    require((list_criterion or integer_criterion or word_criterion)
            and semantic['allow_terminal_period'] is True, 'invalid semantic criterion')
    final, present = closed_final(text)
    # Ignore one sentence-final period only for semantics; preserve exact output for parity.
    value = final[:-1] if final.endswith('.') else final
    if list_criterion:
        require(re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+)*', value) is not None,
                'final answer must contain only a comma-separated integer list')
        require([piece.strip() for piece in value.split(',')] == [str(v) for v in semantic['values']],
                'final integer sequence differs')
    elif integer_criterion:
        require(value == '12', 'final answer must be the standalone integer 12')
    else:
        require(value.casefold() == 'mango', 'final answer must be the standalone word MANGO')
    return final, present


def request_body(fixture, stream=False):
    return {'model': MODEL_ID, 'messages': fixture['messages'], 'max_tokens': fixture['max_tokens'],
            'temperature': fixture['temperature'], 'stream': stream}


def generation(status, value, fixture, profile='list-c128'):
    spec = profile_spec(profile)
    require(status == 200 and type(value) is dict, 'completion failed')
    require(value.get('object') == 'chat.completion' and value.get('model') == MODEL_ID,
            'completion identity differs')
    choices = value.get('choices')
    require(type(choices) is list and len(choices) == 1 and type(choices[0]) is dict,
            'completion choice count differs')
    choice = choices[0]
    require(type(choice.get('index')) is int and choice['index'] == 0
            and choice.get('finish_reason') == 'stop', 'fixture did not finish at EOS')
    message = choice.get('message')
    require(type(message) is dict and message.get('role') == 'assistant'
            and type(message.get('content')) is str and not message.get('tool_calls'),
            'invalid assistant answer')
    usage = value.get('usage')
    require(type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
            and all(type(v) is int for v in usage.values()), 'invalid token usage schema')
    prompt, completion = usage['prompt_tokens'], usage['completion_tokens']
    limits = profile_limits(profile)
    require(1 <= prompt <= limits['maximum_prompt_tokens']
            and (limits['exact_prompt_tokens'] is None or prompt == limits['exact_prompt_tokens'])
            and 1 <= completion < fixture['max_tokens']
            and usage['total_tokens'] == prompt + completion <= limits['maximum_evaluated_positions_per_request'],
            'EOS usage exceeds preregistered prompt/completion/evaluation bounds')
    if spec['require_semantic_match']:
        final, thinking = closed_answer(message['content'], fixture['expected_semantic'])
    else:
        final, thinking = closed_final(message['content'])
    answer = {'fixture_id': fixture['id'], 'text': message['content'], 'final_answer': final,
            'thinking_present': thinking, 'finish_reason': 'stop', 'usage': usage,
            'evaluated_positions': prompt + completion}
    if not spec['require_semantic_match']:
        answer['quality'] = quality_assessment(final, fixture)
    return answer


def validate_answer(answer, fixture, profile='list-c128'):
    keys = {'fixture_id', 'text', 'final_answer', 'thinking_present', 'finish_reason', 'usage', 'evaluated_positions'}
    if not profile_spec(profile)['require_semantic_match']:
        keys.add('quality')
    require(type(answer) is dict and set(answer) == keys, 'answer artifact schema differs')
    rebuilt = generation(200, {'object': 'chat.completion', 'model': MODEL_ID,
        'choices': [{'index': 0, 'finish_reason': answer['finish_reason'],
                     'message': {'role': 'assistant', 'content': answer['text']}}],
        'usage': answer['usage']}, fixture, profile)
    require(answer == rebuilt and type(answer['thinking_present']) is bool
            and type(answer['evaluated_positions']) is int, 'answer artifact differs from validated response')
    return rebuilt


class SseResult:
    def __init__(self, started):
        self.started = started
        self.content = ''
        self.roles = self.finishes = self.done = self.content_chunks = 0
        self.response_id = None
        self.first_content = self.last_content = self.done_at = None
        self.previous = started

    def line(self, raw, received):
        require(type(received) in (float, int) and math.isfinite(received)
                and received >= self.previous, 'SSE clock moved backwards')
        self.previous = received
        require(len(raw) <= BODY_LIMIT, 'oversized SSE line')
        text = raw.decode('utf-8').rstrip('\r\n')
        if not text or text.startswith(':'):
            return
        require(not text.startswith('event: error'), 'SSE emitted error')
        require(text.startswith('data: ') and self.done == 0, 'unexpected SSE data or data after DONE')
        data = text[6:]
        if data == '[DONE]':
            require(self.roles == self.finishes == 1 and self.content_chunks > 0, 'premature DONE')
            self.done = 1
            self.done_at = received
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
                'SSE choice count differs')
        choice = choices[0]
        delta = choice.get('delta')
        require(type(delta) is dict and set(delta).issubset({'role', 'content'}), 'unexpected SSE delta')
        if 'role' in delta:
            require(delta['role'] == 'assistant' and self.roles == self.finishes == 0 and not self.content,
                    'duplicate or misplaced SSE role')
            self.roles = 1
        if 'content' in delta:
            require(self.roles == 1 and self.finishes == 0 and type(delta['content']) is str,
                    'misplaced SSE content')
            if delta['content']:
                self.content += delta['content']
                require(len(self.content.encode('utf-8')) <= BODY_LIMIT, 'oversized SSE content')
                self.content_chunks += 1
                if self.first_content is None:
                    self.first_content = received
                self.last_content = received
        if choice.get('finish_reason') is not None:
            require(self.roles == 1 and choice['finish_reason'] == 'stop' and self.finishes == 0,
                    'SSE did not finish once at EOS')
            self.finishes = 1

    def finish(self, expected, fixture, round_number, profile='list-c128'):
        require(self.roles == self.finishes == self.done == 1 and self.content == expected['text'],
                'SSE text/termination differs from nonstream reference')
        validate_answer(expected, fixture, profile)
        final, thinking = closed_final(self.content)
        require(final == expected['final_answer'] and thinking == expected['thinking_present'],
                'SSE semantic answer differs')
        elapsed = self.done_at - self.started
        require(0 < elapsed <= (profile_limits(profile)['client_seconds'] if profile in (PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE) else REQUEST_SECONDS) and self.first_content is not None
                and self.started <= self.first_content <= self.last_content <= self.done_at,
                'SSE timing interval is invalid')
        completion = expected['usage']['completion_tokens']
        return {'round': round_number, 'fixture_id': fixture['id'],
                'prompt_tokens': expected['usage']['prompt_tokens'], 'completion_tokens': completion,
                'content_chunks': self.content_chunks, 'ttfc_seconds': self.first_content - self.started,
                'last_content_seconds': self.last_content - self.started, 'done_seconds': elapsed,
                'completion_tokens_per_second': completion / elapsed,
                'usage_source': 'matched_nonstream_warmup_api'}


def complete(server, fixture, path, profile='list-c128'):
    started = time.monotonic()
    connection, response = server.open('POST', '/v1/chat/completions', request_body(fixture))
    read_error = None
    try:
        try:
            raw = response.read(BODY_LIMIT + 1)
        except http.client.IncompleteRead as error:
            raw = error.partial[:BODY_LIMIT + 1]
            read_error = 'incomplete HTTP response body'
        elapsed = time.monotonic() - started
        status = response.status
        content_type = response.getheader('Content-Type', '')
        truncated = len(raw) > BODY_LIMIT
        raw = raw[:BODY_LIMIT]
        decoded, value, parse_error = None, None, None
        try:
            decoded = raw.decode('utf-8')
            value = strict_json(decoded)
        except (UnicodeDecodeError, ValueError) as error:
            parse_error = str(error)[:512]
        # Save exact bounded body bytes and diagnostic JSON before protocol,
        # EOS or semantic validation. Base64 also preserves invalid UTF-8.
        write_json(path, {'status': status, 'content_type': content_type,
                          'raw_body_base64': base64.b64encode(raw).decode('ascii'),
                          'response': value, 'parse_error': parse_error,
                          'body_truncated': truncated, 'read_error': read_error,
                          'elapsed_seconds': elapsed})
        require(read_error is None and not truncated, 'incomplete or oversized HTTP body')
        require(content_type.startswith('application/json'), 'completion content type differs')
        require(parse_error is None, 'completion JSON is invalid')
        require(0 < elapsed <= (600 if profile in (PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE) else REQUEST_SECONDS), 'nonstream request deadline exceeded')
        return generation(status, value, fixture, profile), elapsed
    finally:
        response.close()
        connection.close()


def stream_complete(server, fixture, expected, round_number, path, profile='list-c128'):
    started = time.monotonic()
    connection, response = server.open('POST', '/v1/chat/completions', request_body(fixture, stream=True))
    state, events, total = SseResult(started), [], 0
    try:
        require(response.status == 200 and response.getheader('Content-Type', '').startswith('text/event-stream'),
                'SSE status/content type differs')
        while True:
            remaining = REQUEST_SECONDS - (time.monotonic() - started)
            require(remaining > 0, 'SSE request deadline exceeded')
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            raw = response.readline(BODY_LIMIT + 1)
            received = time.monotonic()
            if not raw:
                break
            total += len(raw)
            require(total <= BODY_LIMIT, 'oversized SSE body')
            events.append({'received_seconds': received - started, 'line': raw.decode('utf-8')})
            state.line(raw, received)
        return state.finish(expected, fixture, round_number, profile)
    finally:
        response.close()
        connection.close()
        write_json(path, {'status': response.status, 'content_type': response.getheader('Content-Type', ''),
                          'clock': 'monotonic_elapsed_from_request_start', 'events': events})


def validate_client_stream(record, expected, fixture, round_number, profile):
    """Reconstruct raw SSE independently of the downstream provider's parser."""
    require(type(record) is dict and record.get('status') == 200
            and type(record.get('content_type')) is str
            and record['content_type'].startswith('text/event-stream')
            and record.get('clock') == 'monotonic_elapsed_from_request_start',
            'client SSE status/content type/clock differs')
    client = record.get('client')
    require(type(client) is dict and client.get('kind') == 'antimatter'
            and type(client.get('pid')) is int and client['pid'] > 0
            and type(client.get('request_count')) is int and client['request_count'] == 1
            and type(client.get('attempt_count')) is int and client['attempt_count'] == 1
            and client.get('response_complete') is True and client.get('error') is None,
            'client request, completion or failure evidence differs')
    request = client.get('request')
    require(type(request) is dict and request == request_body(fixture, stream=True)
            and type(request.get('max_tokens')) is int and type(request.get('temperature')) is int
            and request.get('stream') is True, 'client changed the declared wire request')
    result = client.get('result')
    require(type(result) is dict and result == {
        'content': expected['text'], 'toolCalls': [], 'reasoning': '',
        'finishReason': 'stop', 'servedModel': MODEL_ID, 'toolCallsCutOff': False,
    } and result.get('toolCallsCutOff') is False, 'client parsed result differs from CPU reference')
    callbacks = client.get('callbacks')
    require(type(callbacks) is list and 1 <= len(callbacks) <= 4096,
            'client callback count differs')
    for callback in callbacks:
        require(type(callback) is dict and set(callback) == {'id', 'content'}
                and callback.get('id') == ('hag-client-grounded' if profile in GROUNDED_PROFILES else 'hag-client-list')
                and type(callback.get('content')) is str and callback['content'],
                'client emitted unexpected callback fields or identity')
    require(''.join(callback['content'] for callback in callbacks) == expected['text'],
            'client callback text differs from CPU reference')
    events = record.get('events')
    require(type(events) is list and 1 <= len(events) <= 4096, 'client SSE event count differs')
    state, total = SseResult(0), 0
    for event in events:
        require(type(event) is dict and set(event) == {'received_seconds', 'line'}
                and type(event['line']) is str, 'client raw SSE event schema differs')
        received = event['received_seconds']
        require(type(received) in (int, float) and math.isfinite(received)
                and 0 <= received <= profile_limits(profile)['client_seconds'], 'client request deadline exceeded')
        raw = event['line'].encode('utf-8')
        total += len(raw)
        require(total <= BODY_LIMIT, 'oversized client raw SSE evidence')
        state.line(raw, received)
    if profile in GROUNDED_PROFILES:
        validate_grounded_client(record, fixture, True, profile)
    return state.finish(expected, fixture, round_number, profile)


def client_stream_complete(server, fixture, expected, round_number, path, profile):
    """Run one frozen host-client process in the worker's existing cgroup."""
    root = server.root
    identity = client_identity(root)
    require(identity is not None and identity['kind'] == 'antimatter', 'missing frozen client identity')
    prefix = f'timing-round-{round_number}-client'
    request_path = root / (prefix + '-request.json')
    process_path = root / (prefix + '-process.json')
    log_path = root / (prefix + '.log')
    require(not path.exists() and not process_path.exists(), 'client output already exists')
    write_json(request_path, request_body(fixture, stream=True))
    environment = dict(os.environ)
    for name in ('NODE_OPTIONS', 'NODE_PATH', PHASE_ENVIRONMENT):
        environment.pop(name, None)
    process, timed_out = None, False
    with log_path.open('x') as output:
        try:
            process = subprocess.Popen([
                str(root / 'client/node'), str(root / 'antimatter-client.mjs'),
                '--root', str(root), '--port', str(server.port),
                '--request', str(request_path), '--output', str(path),
            ], cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=CLIENT_PROCESS_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                write_json(process_path, {'pid': process.pid, 'exit_code': process.returncode,
                                         'timed_out': timed_out, 'identity': identity})
    require(process is not None and not timed_out and process.returncode == 0,
            'frozen client process failed or exceeded deadline')
    bounded_text(log_path)
    require(client_identity(root) == identity, 'client identity changed during request')
    record = read_json(path)
    require(record.get('client', {}).get('pid') == process.pid, 'client process identity differs')
    return validate_client_stream(record, expected, fixture, round_number, profile)


def validate_grounded_client(record, fixture, stream, profile=GROUNDED_PROFILE):
    limits = profile_limits(profile)
    """Bind the real provider operation and its zero-I/O capacity preflight."""
    client = record.get('client')
    require(type(client) is dict and client.get('kind') == 'antimatter'
            and type(client.get('pid')) is int and client['pid'] > 0
            and type(client.get('attempt_count')) is int and client['attempt_count'] == 1
            and type(client.get('request_count')) is int and client['request_count'] == 1
            and client.get('error') is None and client.get('response_complete') is True,
            'grounded client did not complete exactly one request')
    require(client.get('mode') == ('stream' if stream else 'chat')
            and type(client.get('max_evaluated_tokens')) is int and client['max_evaluated_tokens'] == limits['capacity'],
            'grounded client capacity or method differs')
    expected_preflight = {'capacity': limits['capacity'], 'requested_output_tokens': limits.get('preflight_requested_output_tokens', 512), 'fetch_attempts': 0,
                          'track_calls': 0, 'prepare_calls': 0, 'error_code': 'context_length_exceeded'}
    preflight = client.get('preflight')
    require(type(preflight) is dict and preflight == expected_preflight
            and all(type(preflight[name]) is int for name in expected_preflight if name != 'error_code'),
            'grounded client did not refuse locally before hooks and fetch')
    hooks = client.get('hooks')
    require(type(hooks) is dict and hooks == {'track_calls': 1, 'prepare_calls': 1, 'track_end_calls': 1}
            and all(type(value) is int for value in hooks.values()), 'grounded provider hook count differs')
    request = client.get('request')
    require(type(request) is dict and request == request_body(fixture, stream)
            and type(request.get('max_tokens')) is int and type(request.get('temperature')) is int
            and request.get('stream') is stream, 'grounded provider changed the declared wire request')


def validate_grounded_chat(record, fixture, profile):
    require(profile in GROUNDED_PROFILES, 'grounded client requires its own profile')
    validate_grounded_client(record, fixture, False, profile)
    require(record.get('status') == 200 and type(record.get('status')) is int
            and type(record.get('content_type')) is str
            and record['content_type'].startswith('application/json')
            and record.get('clock') == 'monotonic_elapsed_from_request_start'
            and record.get('events') == [] and record['client'].get('callbacks') == [],
            'grounded chat response protocol differs')
    raw = record.get('raw_body_base64')
    require(type(raw) is str, 'grounded chat has no retained raw body')
    body = base64.b64decode(raw, validate=True)
    require(len(body) <= BODY_LIMIT, 'oversized grounded chat response')
    answer = generation(200, strict_json(body), fixture, profile)
    require(record['client'].get('result') == answer['text'], 'grounded chat result differs from raw response')
    elapsed = record.get('elapsed_seconds')
    require(type(elapsed) in (float, int) and math.isfinite(elapsed)
            and 0 < elapsed <= profile_limits(profile)['client_seconds'], 'grounded chat deadline exceeded')
    return answer, elapsed


def grounded_client_record(server, fixture, path, prefix, stream, profile):
    """Freeze the checked server descriptor and execute one bounded client process."""
    require(profile in GROUNDED_PROFILES, 'grounded client profile differs')
    root, limits = server.root, profile_limits(profile)
    identity = client_identity(root)
    require(identity is not None and identity['kind'] == 'antimatter', 'missing grounded client identity')
    status, listed = server.json('GET', '/v1/models')
    require(status == 200 and type(listed) is dict and type(listed.get('data')) is list
            and len(listed['data']) == 1, 'grounded model descriptor list differs')
    descriptor = listed['data'][0]
    require(type(descriptor) is dict and descriptor.get('id') == MODEL_ID
            and type(descriptor.get('loaded')) is bool
            and type(descriptor.get('execution_limits')) is dict
            and type(descriptor['execution_limits'].get('max_evaluated_tokens')) is int
            and descriptor['execution_limits']['max_evaluated_tokens'] == limits['capacity']
            and (not stream or descriptor['loaded'] is True), 'grounded server descriptor differs')
    request_path, process_path = root / (prefix + '-request.json'), root / (prefix + '-process.json')
    log_path = root / (prefix + '.log')
    require(not path.exists() and not process_path.exists(), 'grounded client output already exists')
    write_json(request_path, {'profile': profile, 'request': request_body(fixture, stream),
                              'model_descriptor': descriptor})
    environment = dict(os.environ)
    for name in ('NODE_OPTIONS', 'NODE_PATH', PHASE_ENVIRONMENT):
        environment.pop(name, None)
    process, timed_out = None, False
    with log_path.open('x') as output:
        try:
            process = subprocess.Popen([
                str(root / 'client/node'), str(root / 'antimatter-grounded-client.mjs'),
                '--root', str(root), '--port', str(server.port),
                '--request', str(request_path), '--output', str(path),
            ], cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=limits['client_process_seconds'])
            except subprocess.TimeoutExpired:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
        finally:
            if process is not None:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=2)
                write_json(process_path, {'pid': process.pid, 'exit_code': process.returncode,
                                         'timed_out': timed_out, 'identity': identity})
    require(process is not None and not timed_out and process.returncode == 0,
            'frozen grounded client process failed or exceeded deadline')
    bounded_text(log_path)
    require(client_identity(root) == identity, 'grounded client identity changed during request')
    record = read_json(path)
    require(record.get('client', {}).get('pid') == process.pid, 'grounded client process identity differs')
    return record


def grounded_complete(server, fixture, path, profile):
    prefix = path.name.removesuffix('-response.json') + '-client'
    return validate_grounded_chat(grounded_client_record(server, fixture, path, prefix, False, profile),
                                  fixture, profile)


def grounded_stream_complete(server, fixture, expected, round_number, path, profile):
    record = grounded_client_record(server, fixture, path, f'timing-round-{round_number}-client', True, profile)
    return validate_client_stream(record, expected, fixture, round_number, profile)


def capacity_refusal(server, fixture, stream, loaded, audit_count, path, profile=GROUNDED_PROFILE):
    """Exercise exact tokenizer planning before any generation or SSE headers."""
    server.model(loaded)
    before = audit_events(server.log_path, profile_limits(profile)['capacity'])
    require(len(before) == audit_count, 'unexpected audit before capacity refusal')
    request = {**request_body(fixture, stream), 'max_tokens': profile_limits(profile).get('overflow_max_tokens', 421)}
    connection, response = server.open('POST', '/v1/chat/completions', request)
    data, value = b'', None
    try:
        data = response.read(BODY_LIMIT + 1)
        require(len(data) <= BODY_LIMIT, 'oversized capacity-refusal response')
        value = strict_json(data)
    finally:
        response.close()
        connection.close()
        write_json(path, {'request': request, 'status': response.status,
                          'content_type': response.getheader('Content-Type', ''),
                          'raw_body_base64': base64.b64encode(data[:BODY_LIMIT]).decode(), 'response': value,
                          'loaded_before': loaded, 'audit_count_before': audit_count})
    require(response.status == 400 and response.getheader('Content-Type', '').startswith('application/json')
            and type(value) is dict and type(value.get('error')) is dict,
            'capacity refusal was not JSON HTTP400 before stream headers')
    error = value['error']
    require(error.get('type') == 'invalid_request_error' and error.get('param') == 'max_tokens'
            and error.get('code') == 'context_length_exceeded', 'capacity refusal error identity differs')
    server.model(loaded)
    after = audit_events(server.log_path, profile_limits(profile)['capacity'])
    require(after == before, 'capacity refusal generated native work')
    # Separate postconditions keep captured raw response bytes immutable on failure.
    write_json(path.with_name(path.stem + '-post.json'),
               {'loaded_after': loaded, 'audit_count_after': len(after), 'unchanged_audit': True})
    return {'phase': 'warm' if loaded else 'cold', 'stream': stream, 'status': 400,
            'code': 'context_length_exceeded', 'param': 'max_tokens', 'loaded': loaded,
            'audit_count': audit_count}


def identity(root):
    manifest = read_json(root / 'expected-manifest.json')
    binary_sha = manifest['binaries']['beholdd']
    require(digest(root / 'beholdd') == binary_sha, 'frozen CLI identity differs')
    require(digest(root / 'build-source-sha256.json') == manifest['source_manifest_sha256'],
            'frozen source manifest identity differs')
    files = manifest['gate_files']
    for name in ('answer-gate.py', 'answer_quality.py', 'profiles.py', 'gate_common.py',
                 'qualification-contract.json', 'answer-fixtures.json', 'run-config.json', 'cpu-config.json', 'hag-config.json'):
        path = root / name
        require(name in files and path.is_file() and not path.is_symlink() and digest(path) == files[name],
                'frozen runner/config/fixture differs: ' + name)
    require(manifest.get('profile') == fixture_profile(read_json(root / 'answer-fixtures.json')),
            'manifest profile differs from the frozen fixture document')
    contract = approved_contract(root)
    require(contract['profile'] == manifest['profile'], 'contract profile differs from frozen identity')
    inputs = read_json(root / 'run-config.json')
    require(type(inputs) is dict and set(inputs) == {'schema', 'repo', 'model', 'device', 'binary_input'}
            and type(inputs['schema']) is int and inputs['schema'] == 1, 'run configuration schema differs')
    for name in ('repo', 'model', 'device', 'binary_input'):
        value = inputs[name]
        require(type(value) is str and Path(value).is_absolute() and '..' not in Path(value).parts
                and str(Path(value)) == value, 'run configuration path is not canonical: ' + name)
    profile = manifest['profile']
    for backend in ('cpu', 'hag'):
        config = read_json(root / (backend + '-config.json'))
        model = {'id': MODEL_ID, 'path': inputs['model'], 'execution_backend': backend,
                 'weight_mode': 'streaming', 'fit_policy': 'exact',
                 'memory_budget_bytes': (2 if backend == 'hag' else 1) * 1024 ** 3,
                 'runtime_headroom_bytes': 128 * 1024 ** 2}
        if backend == 'hag':
            model['hag'] = {'device': inputs['device'], 'memory_budget_bytes': 768 * 1024 ** 2}
        if profile in C2048_PROFILES:
            if backend == 'cpu':
                model['memory_budget_bytes'] = 1536 * 1024 ** 2
            else:
                model['hag']['memory_budget_bytes'] = 1152 * 1024 ** 2
        require(config == {'models': [model]}, 'explicit backend/path/resource configuration differs: ' + backend)
        actual = config['models'][0]
        require(all(type(actual[name]) is int for name in ('memory_budget_bytes', 'runtime_headroom_bytes'))
                and (backend == 'cpu' or type(actual['hag']['memory_budget_bytes']) is int),
                'resource configuration requires integer byte limits')
    result = {'binary_sha256': binary_sha, 'source_manifest_sha256': manifest['source_manifest_sha256'],
            'checkpoint_sha256': manifest['model_sha256'], 'fixture_sha256': files['answer-fixtures.json'],
            'gate_files': files}
    diagnostics = phase_diagnostics(contract)
    if diagnostics is not None:
        result['runtime_phase_diagnostics'] = diagnostics
    client = client_identity(root)
    if client is not None:
        result['client_identity'] = client
    return result


def qualified_prerequisite(root, stage, identities, profile=None):
    result = read_json(root / (stage + '-result.json'))
    require(result.get('mode') == stage and result.get('qualified') is True and result.get('failures') == []
            and type(result.get('exit_code')) is int and result['exit_code'] == 0 and result.get('binary_sha256') == identities['binary_sha256']
            and result.get('source_manifest_sha256') == identities['source_manifest_sha256']
            and result.get('checkpoint_sha256') == identities['checkpoint_sha256']
            and result.get('gate_files') == identities['gate_files'], 'prerequisite identity/qualification differs: ' + stage)
    report = read_json(root / (stage + '-report.json'))
    require(result.get('probe') == report, 'prerequisite log report differs: ' + stage)
    require(result.get('client_identity') == report.get('client_identity') == identities.get('client_identity'),
            'prerequisite client identity differs: ' + stage)
    if profile is not None:
        spec = profile_spec(profile)
        require(result.get('profile') == report.get('profile') == profile
                and report.get('qualification_kind') == spec['qualification_kind']
                and report.get('require_semantic_match') is spec['require_semantic_match']
                and report.get('quality_is_diagnostic') is (not spec['require_semantic_match']),
                'prerequisite profile/policy differs: ' + stage)
    return result, report, digest(root / (stage + '-result.json'))


def references(root, identities, fixtures, profile):
    _, report, result_sha = qualified_prerequisite(root, 'cpu-reference', identities, profile)
    path = root / 'cpu-reference.json'
    artifact = read_json(path)
    require(type(artifact) is dict and set(artifact) == {'schema', 'execution', 'capacity', 'identities', 'profile',
            'answers', 'cpu_exit_code', 'final_unloaded'}, 'CPU reference artifact schema differs')
    require(type(artifact['schema']) is int and artifact['schema'] == 1 and artifact['execution'] == 'cpu-reference'
            and type(artifact['capacity']) is int and artifact['capacity'] == profile_limits(profile)['capacity'] and artifact['identities'] == identities
            and artifact['profile'] == profile and report.get('profile') == profile
            and type(artifact['cpu_exit_code']) is int and artifact['cpu_exit_code'] == 0
            and artifact['final_unloaded'] is True, 'CPU reference artifact identity/lifecycle differs')
    answers = artifact['answers']
    require(type(answers) is list and len(answers) == len(fixtures), 'CPU reference answer count differs')
    for answer, fixture in zip(answers, fixtures):
        validate_answer(answer, fixture, profile)
    reference_sha = digest(path)
    require(report.get('reference_sha256') == reference_sha and report.get('answers') == answers,
            'CPU reference artifact no longer matches its qualified stage')
    return {a['fixture_id']: a for a in answers}, reference_sha, result_sha


def event_success(server, count, answer):
    event = server.event(count)
    require(event['outcome'] == 'success' and event['evaluated_positions'] == answer['evaluated_positions'],
            'native EOS must evaluate prompt plus all emitted content tokens')
    return event


def execute(stage, root=ROOT):
    require(stage in STAGES, 'unknown answer stage')
    require(os.environ.get('HAG_ANSWERS_GATE') == '1' and os.environ.get('HAG_TRACE_SUBMISSION') == '0',
            'launch only through the frozen answer supervisor')
    verify_containment()
    identities = dict(identity(root))
    diagnostics = identities.pop('runtime_phase_diagnostics', None)
    # The contract is already included in gate_files. Diagnostic launch policy
    # is kept separate from the unchanged CPU reference/parity identity schema.
    server_options = {'phase_diagnostics': diagnostics} if diagnostics is not None else {}
    document = read_json(root / 'answer-fixtures.json')
    profile = fixture_profile(document)
    spec = profile_spec(profile)
    limits = profile_limits(profile)
    fixtures = fixtures_from(document)
    report_path = root / (stage + '-report.json')
    require(not report_path.exists(), 'stage already has an immutable report')
    report = {'stage': stage, 'execution': 'cpu-reference' if stage == 'cpu-reference' else 'native-hag-' + stage,
              'profile': profile, 'capacity': limits['capacity'], 'fixture_ids': [f['id'] for f in fixtures],
              'qualification_kind': spec['qualification_kind'],
              'require_semantic_match': spec['require_semantic_match'],
              'quality_is_diagnostic': not spec['require_semantic_match'],
              'requests': spec['stage_request_counts'][stage],
              'maximum_evaluated_positions': spec['stage_evaluation_ceilings'][stage],
              'all_eos': True, 'closed_answers': True,
              'final_unloaded': True, 'audit_events': []}
    if 'client_identity' in identities:
        report['client_identity'] = identities['client_identity']
    if stage == 'cpu-reference':
        path = root / 'cpu-reference.json'
        require(not path.exists(), 'CPU reference already exists')
        answers = []
        with Server(root, 'cpu', stage, limits['capacity'], profile, **server_options) as server:
            server.identity()
            for fixture in fixtures:
                answer, _ = complete(server, fixture, root / (stage + '-' + fixture['id'] + '-response.json'), profile)
                answers.append(answer)
            server.unload()
            require(audit_events(server.log_path, limits['capacity']) == [], 'CPU reference emitted HAG audit events')
            server.stop()
        artifact = {'schema': 1, 'execution': 'cpu-reference', 'capacity': limits['capacity'], 'identities': identities,
                    'profile': profile,
                    'answers': answers, 'cpu_exit_code': 0, 'final_unloaded': True}
        write_json(path, artifact)
        report.update(cpu_exit_code=0, answers=answers, reference_sha256=digest(path),
                      evaluated_positions=sum(a['evaluated_positions'] for a in answers))
    else:
        expected, reference_sha, cpu_result_sha = references(root, identities, fixtures, profile)
        report.update(reference_sha256=reference_sha, cpu_reference_result_sha256=cpu_result_sha,
                      hag_exit_code=0, exact_cpu_parity=True)
        if stage == 'timing':
            _, prior, answers_sha = qualified_prerequisite(root, 'answers', identities, profile)
            require(prior.get('reference_sha256') == reference_sha and prior.get('exact_cpu_parity') is True
                    and prior.get('profile') == profile,
                    'timing prerequisite does not match CPU reference')
            report['answers_result_sha256'] = answers_sha
        events = []
        with Server(root, 'hag', stage, limits['capacity'], profile, **server_options) as server:
            server.identity()
            if stage == 'answers':
                answers = []
                if profile in GROUNDED_PROFILES:
                    cold = capacity_refusal(server, fixtures[0], False, False, 0, root / 'answers-cold-refusal.json', profile)
                for index, fixture in enumerate(fixtures, 1):
                    complete_request = grounded_complete if profile in GROUNDED_PROFILES else complete
                    answer, _ = complete_request(server, fixture, root / (stage + '-' + fixture['id'] + '-response.json'), profile)
                    require(answer == expected[fixture['id']], 'native answer/text/usage differs from CPU')
                    events.append(event_success(server, index, answer))
                    answers.append(answer)
                report['answers'] = answers
                if profile in GROUNDED_PROFILES:
                    warm = capacity_refusal(server, fixtures[0], True, True, 1, root / 'answers-warm-refusal.json', profile)
                    report.update(capacity_refusals=[cold, warm], client_chat_requests=1, client_stream_requests=0,
                                  model_free_client_refusals=1, live_cancellation_qualified=False)
            else:
                fixture = next(f for f in fixtures if f['id'] == document['timing_fixture_id'])
                complete_request = grounded_complete if profile in GROUNDED_PROFILES else complete
                warmup, elapsed = complete_request(server, fixture, root / 'timing-warmup-response.json', profile)
                require(warmup == expected[fixture['id']], 'warmup answer/text/usage differs from CPU')
                events.append(event_success(server, 1, warmup))
                server.model(True)
                metrics = []
                for index in range(limits['measured_requests']):
                    server.model(True)
                    stream = (grounded_stream_complete if profile in GROUNDED_PROFILES else
                              client_stream_complete if 'client_identity' in identities else stream_complete)
                    metrics.append(stream(server, fixture, warmup, index + 1,
                                          root / f'timing-round-{index + 1}-sse.json', profile))
                    events.append(event_success(server, index + 2, warmup))
                    server.model(True)
                report.update(fixture_ids=[fixture['id']], requests=spec['stage_request_counts'][stage],
                              maximum_evaluated_positions=spec['stage_evaluation_ceilings'][stage],
                              timing_fixture_id=fixture['id'], warmup_requests=1, measured_requests=limits['measured_requests'],
                              descriptive_client_visible_only=True, metrics=metrics,
                              ttfc_definition='first_nonempty_raw_content_including_thinking_markup',
                              rate_definition='api_completion_tokens_over_request_to_done_seconds', cpu_quota_percent=75,
                              warmup={'label': 'first_request_after_lazy_load', 'fixture_id': fixture['id'],
                                      'latency_seconds': elapsed, 'usage_source': 'nonstream_api', 'generation': warmup})
                if profile in GROUNDED_PROFILES:
                    report.update(client_chat_requests=1, client_stream_requests=1, model_free_client_refusals=2,
                                  client_reuse=True, live_cancellation_qualified=False)
            server.unload()
            require(audit_events(server.log_path, limits['capacity']) == events, 'unexpected native generation outside declared workload')
            server.stop()
        report.update(audit_events=events, evaluated_positions=sum(e['evaluated_positions'] for e in events))
    require(report['evaluated_positions'] <= report['maximum_evaluated_positions'], 'stage evaluation ceiling exceeded')
    write_json(report_path, report)
    print('HAG_ANSWERS_REPORT ' + json.dumps(report, allow_nan=False), flush=True)
    print(f'PASS hag-answers stage={stage} qualified=true', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--stage', choices=STAGES, required=True)
    args = parser.parse_args()
    execute(args.stage)
