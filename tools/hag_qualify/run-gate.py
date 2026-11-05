#!/usr/bin/env python3
"""Run one bounded HAG gate, retaining memory, capture and driver evidence."""
import argparse
import base64
from datetime import datetime, timezone
import json
import importlib.util
import math
from pathlib import Path
import re
import subprocess
import sys
import time

from gate_common import (ROOT, MEMORY_LIMIT, model_path, EXPECTED_LIMITS, STOP_HEADROOM, available_memory,
                         binary_identity, source_identity, dependency_identity, checkpoint_identity, clean_memory, memory_snapshot,
                         read_json, require_headroom, write_json, require_process_headroom, gate_identity,
                         strict_json, MODES, GPU_MODES, SERVICE_SECONDS, OUTER_SECONDS,
                         approved_contract, provenance_identity, digest, require_sha, MODEL_SHA256, stage_deadlines, private_cgroup_attestation,
                         phase_diagnostics, require_phase_diagnostics)
from profiles import (fixture_profile, profile_spec, profile_limits, GROUNDED_PROFILE, GROUNDED_PROFILES,
                      PROJECT_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, C2048_PROFILES, profile_deadlines, profile_modes)
from answer_quality import quality_assessment
from gate_common import client_identity


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def bounded_text(path, limit=2 * 1024 * 1024):
    with path.open('rb') as stream:
        value = stream.read(limit + 1)
    if len(value) > limit:
        raise ValueError(f'evidence exceeds text limit: {path}')
    return value.decode('utf-8')


def validate_capture(records, now, status):
    if not records or records[0].get('event') != 'ARMED':
        raise ValueError('crash watcher is not armed')
    pid = records[0].get('pid')
    if type(pid) is not int or pid <= 0:
        raise ValueError('invalid capture PID')
    if any(record.get('event') not in ('ARMED', 'HEARTBEAT') or type(record.get('captured')) is not int or record.get('captured') != 0
           or record.get('pid') != pid for record in records):
        raise ValueError('capture session has a fault, dump, stopped state or inconsistent PID')
    if sum(record['event'] == 'ARMED' for record in records) != 1 or records[-1].get('event') != 'HEARTBEAT':
        raise ValueError('capture needs one ARMED event followed by a fresh heartbeat')
    age = (now - datetime.fromisoformat(records[-1]['timestamp'].replace('Z', '+00:00'))).total_seconds()
    if not 0 <= age <= 15:
        raise ValueError('capture heartbeat is stale or in the future')
    if not re.search(r'^Uid:\s+0\s+0\s+0\s+0$', status, re.MULTILINE) or re.search(r'^State:\s+Z', status, re.MULTILINE):
        raise ValueError('capture PID is not a running root process')
    return {'pid': pid, 'armed_at': records[0]['timestamp']}



def validate_capture_lease(records, now, mode=None, profile=PROJECT_PROFILE):
    armed = records[0]
    if type(armed.get('duration_seconds')) is not int or armed['duration_seconds'] != 1500:
        raise ValueError('project profile requires the preregistered 1500-second watcher lease')
    started = datetime.fromisoformat(armed['timestamp'].replace('Z', '+00:00'))
    elapsed = (now - started).total_seconds()
    if mode is not None and mode not in profile_modes(profile):
        raise ValueError('stage is not admitted by this profile')
    required = 0 if mode is None else profile_deadlines(profile)[mode][2] + 20
    if elapsed < 0 or 1500 - elapsed < required:
        raise ValueError('watcher has insufficient remaining time for this stage and cleanup')


def validate_cooldown(record):
    if (type(record) is not dict or record.get('qualified') is not True
            or record.get('maximum_c') != 60 or record.get('stable_seconds') != 10):
        raise ValueError('project start cooldown did not qualify')
    samples = record.get('samples')
    if type(samples) is not list or not 2 <= len(samples) <= 601:
        raise ValueError('project cooldown sample count differs')
    previous, stable = -1, None
    for sample in samples:
        elapsed = finite_seconds(sample.get('elapsed_seconds'), 'cooldown', maximum=60.5)
        temperature = sample.get('tctl_c')
        if elapsed < previous or (previous >= 0 and elapsed - previous > 1):
            raise ValueError('project cooldown has a sampling gap')
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 125:
            raise ValueError('project cooldown sensor reading differs')
        stable = (elapsed if stable is None else stable) if temperature <= 60 else None
        previous = elapsed
    if stable is None or previous - stable < 10:
        raise ValueError('project cooldown did not hold at most 60C for ten seconds')
    return record


def project_cooldown(mode, root=ROOT):
    """A finite start gate, separate from the unchanged running 90C cutoff."""
    sensors = [label.with_name(label.name.replace('_label', '_input'))
               for label in Path('/sys/class/hwmon').glob('hwmon*/temp*_label')
               if label.read_text().strip() == 'Tctl']
    if len(sensors) != 1:
        raise ValueError('expected exactly one Tctl sensor')
    start, stable, samples = time.monotonic(), None, []
    record = {'maximum_c':60, 'stable_seconds':10, 'qualified':False, 'samples':samples}
    try:
        for _ in range(601):
            elapsed = time.monotonic() - start
            temperature = int(sensors[0].read_text().strip()) / 1000
            if not 0 <= temperature <= 125 or elapsed > 60.5:
                raise ValueError('invalid sensor or expired finite cooldown')
            samples.append({'elapsed_seconds':elapsed, 'tctl_c':temperature})
            stable = (elapsed if stable is None else stable) if temperature <= 60 else None
            if stable is not None and elapsed - stable >= 10:
                record['qualified'] = True
                validate_cooldown(record)
                return record
            if elapsed >= 60:
                break
            time.sleep(.1)
        raise ValueError('finite start cooldown did not reach at most 60C for ten seconds')
    finally:
        write_json(root / (mode + '-start-cooldown.json'), record)


def validate_project_child(child, mode, profile=PROJECT_PROFILE):
    if profile not in C2048_PROFILES or mode not in profile_modes(profile):
        raise ValueError('stage has no C2048 private containment contract')
    expected = dict(zip(('child','service','outer'), profile_deadlines(profile)[mode]))
    require_fields(child, {'stage_deadlines':expected})
    cgroup = Path(child['cgroup'])
    attested = private_cgroup_attestation(cgroup, child['limits'], 'NoNewPrivs:\t1\n', '0', child['memory_before'])
    require_fields(child, {'private_cgroup_attestation':attested if mode in GPU_MODES else None})


def capture_session(root=ROOT, mode=None):
    records = [strict_json(line) for line in bounded_text(root / 'capture.log').splitlines() if line.startswith('{')]
    if not records:
        raise ValueError('empty capture log')
    pid = records[0].get('pid')
    session = validate_capture(records, datetime.now(timezone.utc), Path(f'/proc/{int(pid)}/status').read_text())
    profile = approved_contract(root)['profile']
    if profile in C2048_PROFILES:
        validate_capture_lease(records, datetime.now(timezone.utc), mode, profile)
    if (root / 'capture.stop').exists():
        raise ValueError('capture stop was requested')
    sysfs = Path('/sys/class/devcoredump')
    if list(sysfs.glob('devcd*')) or (sysfs / 'disabled').read_text().strip() != '0':
        raise ValueError('existing devcoredump or disabled capture')
    return session


def require_prior(prior, mode, identity, source_sha, gate_files, session=None, input_provenance=None, profile=None,
                  client=None):
    if (prior.get('mode') != mode or prior.get('qualified') is not True
            or prior.get('failures') != [] or prior.get('binary_sha256') != identity
            or prior.get('source_manifest_sha256') != source_sha
            or prior.get('gate_files') != gate_files or prior.get('checkpoint_sha256') != MODEL_SHA256
            or prior.get('capture_session') != session
            or prior.get('profile') != profile
            or prior.get('input_provenance') != input_provenance):
        raise ValueError('qualified prerequisite identity/order differs: ' + mode)
    if profile is not None:
        spec = profile_spec(profile)
        require_fields(prior, {'qualification_kind': spec['qualification_kind'],
                               'require_semantic_match': spec['require_semantic_match'],
                               'quality_is_diagnostic': not spec['require_semantic_match']})
    require_client_identity(prior, client)


def require_client_identity(record, expected):
    """An absent optional client stays absent; null cannot conceal a changed strategy."""
    if expected is None:
        if 'client_identity' in record:
            raise ValueError('unexpected optional client identity')
    else:
        require_fields(record, {'client_identity': expected})


def require_fields(report, expected):
    if type(report) is not dict:
        raise ValueError('report or nested record must be an object')
    def equal(actual, value):
        if type(actual) is not type(value):
            return False
        if type(value) is list:
            return len(actual) == len(value) and all(equal(a, b) for a, b in zip(actual, value))
        if type(value) is dict:
            return actual.keys() == value.keys() and all(equal(actual[k], v) for k, v in value.items())
        return actual == value
    for key, value in expected.items():
        actual = report.get(key)
        if not equal(actual, value):
            raise ValueError(f'report differs from preregistered contract: {key}')


def bounded_integer(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError('invalid bounded integer: ' + label)


def validate_answer(answer, fixture, profile='list-c128'):
    spec = profile_spec(profile)
    limits = profile_limits(profile)
    keys = {'fixture_id', 'text', 'final_answer', 'thinking_present', 'finish_reason',
            'usage', 'evaluated_positions'}
    if not spec['require_semantic_match']:
        keys.add('quality')
    if type(answer) is not dict or set(answer) != keys:
        raise ValueError('answer schema differs')
    require_fields(answer, {'fixture_id': fixture['id'], 'finish_reason': 'stop'})
    raw = answer.get('text')
    if type(raw) is not str or not raw or len(raw.encode('utf-8')) > 65536:
        raise ValueError('missing or oversized complete answer')
    stripped = raw.strip()
    markers = re.findall(r'<\s*/?\s*think\b[^>]*>?', raw, re.IGNORECASE)
    thinking = bool(markers)
    if thinking:
        if markers != ['<think>', '</think>'] or not stripped.startswith('<think>'):
            raise ValueError('thinking markup is stray, malformed or not closed exactly once')
        if raw.index('<think>') >= raw.index('</think>'):
            raise ValueError('thinking markup is reversed')
        final = raw.split('</think>', 1)[1].strip()
    else:
        final = stripped
    require_fields(answer, {'thinking_present': thinking, 'final_answer': final})
    if not final:
        raise ValueError('missing complete final answer')
    semantic = fixture['expected_semantic']
    admitted = {
        'list_1_to_8': {'kind': 'integer_csv', 'values': list(range(1, 9)), 'allow_terminal_period': True},
        'arithmetic_7_plus_5': {'kind': 'integer', 'value': 12, 'allow_terminal_period': True},
        'recall_mango': {'kind': 'word', 'value': 'MANGO', 'case_sensitive': False, 'allow_terminal_period': True},
    }
    if profile in GROUNDED_PROFILES:
        admitted[profile_spec(profile)['fixture_ids'][0]] = {'kind':'word','value':'READY','case_sensitive':False,'allow_terminal_period':True}
    if fixture['id'] not in admitted:
        raise ValueError('unknown semantic fixture')
    require_fields({'semantic': semantic}, {'semantic': admitted[fixture['id']]})
    if spec['require_semantic_match']:
        # Semantic-only punctuation handling; raw and final text remain exact parity inputs.
        value = final[:-1] if final.endswith('.') else final
        if semantic['kind'] == 'integer_csv':
            if not re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+)*', value):
                raise ValueError('answer is not an integer CSV')
            if [part.strip() for part in value.split(',')] != [str(part) for part in semantic['values']]:
                raise ValueError('integer CSV answer differs')
        elif semantic['kind'] == 'integer':
            if value != str(semantic['value']):
                raise ValueError('answer is not the standalone expected integer')
        elif value.casefold() != semantic['value'].casefold():
            raise ValueError('answer is not the standalone expected word')
    else:
        # Recompute annotations from raw-derived final text; neither a quality
        # failure nor an unassessed answer can masquerade as semantic qualification.
        require_fields(answer, {'quality': quality_assessment(final, fixture)})
    usage = answer.get('usage')
    if type(usage) is not dict or set(usage) != {'prompt_tokens', 'completion_tokens', 'total_tokens'}:
        raise ValueError('answer usage schema differs')
    bounded_integer(usage.get('prompt_tokens'), 1, limits['maximum_prompt_tokens'], 'prompt tokens')
    if limits['exact_prompt_tokens'] is not None:
        require_fields(usage, {'prompt_tokens': limits['exact_prompt_tokens']})
    # Reaching max_tokens ends by length; EOS must precede the frozen output ceiling.
    bounded_integer(usage.get('completion_tokens'), 1, limits['max_new_tokens'] - 1, 'EOS completion tokens')
    evaluated = usage['prompt_tokens'] + usage['completion_tokens']
    require_fields(usage, {'total_tokens': evaluated})
    require_fields(answer, {'evaluated_positions': evaluated})
    bounded_integer(evaluated, 2, limits['maximum_evaluated_positions_per_request'], 'EOS native positions')
    return evaluated


def reference_payload(root, fixtures, profile):
    manifest = read_json(root / 'expected-manifest.json')
    require_fields(manifest, {'profile': profile})
    payload = read_json(root / 'cpu-reference.json')
    if type(payload) is not dict:
        raise ValueError('CPU reference payload must be an object')
    client = client_identity(root)
    identities = {'binary_sha256': manifest['binaries']['beholdd'],
                  'source_manifest_sha256': manifest['source_manifest_sha256'],
                  'checkpoint_sha256': manifest['model_sha256'],
                  'fixture_sha256': manifest['gate_files']['answer-fixtures.json'],
                  'gate_files': manifest['gate_files']}
    if client is not None:
        identities['client_identity'] = client
    require_fields(payload, {'schema': 1, 'profile': profile, 'execution': 'cpu-reference', 'capacity': profile_limits(profile)['capacity'],
                            'cpu_exit_code': 0, 'final_unloaded': True,
                            'identities': identities})
    answers = payload.get('answers')
    if type(answers) is not list or len(answers) != len(fixtures):
        raise ValueError('CPU reference count differs from the frozen profile')
    for answer, fixture in zip(answers, fixtures):
        validate_answer(answer, fixture, profile)
    return answers


def finite_seconds(value, label, *, positive=False, maximum=180):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not (0 < value <= maximum if positive else 0 <= value <= maximum)):
        raise ValueError('invalid bounded timing: ' + label)
    return value


def raw_json_body(record):
    encoded = record.get('raw_body_base64')
    if type(encoded) is not str or len(encoded) > 2 * 1024 * 1024:
        raise ValueError('missing or oversized raw JSON response')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError('invalid raw response base64') from error
    if not raw or len(raw) > 1024 * 1024:
        raise ValueError('raw JSON response exceeds bound')
    return strict_json(raw.decode('utf-8'))


def validate_grounded_client(root, stem, record, request, identity, mode, profile=GROUNDED_PROFILE):
    limits = profile_limits(profile)
    """Bind provider hooks, early refusal, actual wire body and waited child."""
    if identity is None:
        raise ValueError('grounded profile requires the frozen provider client')
    envelope = read_json(root / (stem + '-client-request.json'))
    if type(envelope) is not dict or set(envelope) != {'profile', 'request', 'model_descriptor'}:
        raise ValueError('grounded client envelope differs')
    require_fields(envelope, {'profile': profile, 'request': request})
    descriptor = envelope['model_descriptor']
    require_fields(descriptor, {'id':'hag-context-qwen3-0.6b', 'object':'model', 'owned_by':'beholdd',
                                'execution_limits':{'max_evaluated_tokens':limits['capacity']}, 'loaded':mode == 'stream'})
    client = record.get('client')
    keys = {'kind','pid','request_count','attempt_count','request','result','callbacks',
            'response_complete','error','mode','max_evaluated_tokens','preflight','hooks'}
    if type(client) is not dict or set(client) != keys:
        raise ValueError('grounded client schema differs')
    require_fields(client, {'kind':'antimatter','request_count':1,'attempt_count':1,'request':request,
                            'response_complete':True,'error':None,'mode':mode,'max_evaluated_tokens':limits['capacity'],
                            'preflight':{'capacity':limits['capacity'],'requested_output_tokens':limits.get('preflight_requested_output_tokens',512),'fetch_attempts':0,
                                         'track_calls':0,'prepare_calls':0,'error_code':'context_length_exceeded'},
                            'hooks':{'track_calls':1,'prepare_calls':1,'track_end_calls':1}})
    bounded_integer(client['pid'], 1, 2**31-1, 'grounded client PID')
    process = read_json(root / (stem + '-client-process.json'))
    if type(process) is not dict or set(process) != {'pid','exit_code','timed_out','identity'}:
        raise ValueError('grounded process receipt schema differs')
    require_fields(process, {'pid':client['pid'],'exit_code':0,'timed_out':False,'identity':identity})
    bounded_text(root / (stem + '-client.log'))


def validate_grounded_chat(root, stem, artifact, reference, fixture, identity, profile=GROUNDED_PROFILE):
    record = read_json(root / artifact)
    keys = {'status','content_type','clock','events','client','raw_body_base64','elapsed_seconds'}
    if type(record) is not dict or set(record) != keys:
        raise ValueError('grounded chat raw artifact schema differs')
    require_fields(record, {'status':200,'content_type':'application/json',
                            'clock':'monotonic_elapsed_from_request_start','events':[]})
    elapsed = finite_seconds(record['elapsed_seconds'], 'grounded chat', positive=True, maximum=profile_limits(profile)['client_seconds'])
    if elapsed > profile_limits(profile)['client_seconds']:
        raise ValueError('grounded chat deadline exceeded')
    request = {'model':'hag-context-qwen3-0.6b','messages':fixture['messages'],
               'max_tokens':128,'temperature':0,'stream':False}
    validate_grounded_client(root, stem, record, request, identity, 'chat', profile)
    require_fields(record['client'], {'result':reference['text'], 'callbacks':[]})
    response = raw_json_body(record)
    require_fields(response, {'object':'chat.completion','model':'hag-context-qwen3-0.6b',
                              'usage':reference['usage']})
    choices = response.get('choices')
    if type(choices) is not list or len(choices) != 1:
        raise ValueError('grounded raw completion choice count differs')
    require_fields(choices[0], {'index':0,'finish_reason':'stop'})
    message = choices[0].get('message')
    require_fields(message, {'role':'assistant','content':reference['text']})
    if message.get('tool_calls') or message.get('reasoning_content'):
        raise ValueError('grounded raw completion contains unexpected tools/reasoning')
    return elapsed


def validate_capacity_refusals(root, fixture, profile=GROUNDED_PROFILE):
    summaries = []
    for phase, streaming, loaded, count in [('cold',False,False,0), ('warm',True,True,1)]:
        record = read_json(root / ('answers-' + phase + '-refusal.json'))
        keys = {'request','status','content_type','raw_body_base64','response','loaded_before','audit_count_before'}
        if type(record) is not dict or set(record) != keys:
            raise ValueError('capacity refusal artifact schema differs')
        request = {'model':'hag-context-qwen3-0.6b','messages':fixture['messages'],
                   'max_tokens':profile_limits(profile).get('overflow_max_tokens',421),'temperature':0,'stream':streaming}
        require_fields(record, {'request':request,'status':400,'content_type':'application/json',
                                'loaded_before':loaded,'audit_count_before':count})
        after = read_json(root / ('answers-' + phase + '-refusal-post.json'))
        if type(after) is not dict or set(after) != {'loaded_after','audit_count_after','unchanged_audit'}:
            raise ValueError('capacity refusal postcondition schema differs')
        require_fields(after, {'loaded_after':loaded,'audit_count_after':count,'unchanged_audit':True})
        raw = raw_json_body(record)
        require_fields(record, {'response':raw})
        if type(raw) is not dict or set(raw) != {'error'}:
            raise ValueError('capacity refusal raw body is not a typed error')
        require_fields(raw['error'], {'type':'invalid_request_error','code':'context_length_exceeded','param':'max_tokens'})
        if type(raw['error'].get('message')) is not str or not raw['error']['message']:
            raise ValueError('capacity refusal message absent')
        summaries.append({'phase':phase,'stream':streaming,'status':400,'code':'context_length_exceeded',
                          'param':'max_tokens','loaded':loaded,'audit_count':count})
    return summaries


def validate_antimatter_stream(root, reference, fixture, index, identity, profile='backend-parity-c128'):
    """Corroborate provider output against separately retained strict wire events."""
    stem = 'timing-round-' + str(index)
    record = read_json(root / (stem + '-sse.json'))
    require_fields(record, {'status': 200, 'content_type': 'text/event-stream',
                            'clock': 'monotonic_elapsed_from_request_start'})
    events = record.get('events')
    if type(events) is not list or not 1 <= len(events) <= 4096:
        raise ValueError('missing or oversized client raw SSE events')
    role = finish = done = 0
    content, times = [], []
    previous, total = -1, 0
    response_id = None
    done_at = None
    for event in events:
        if type(event) is not dict or set(event) != {'received_seconds', 'line'}:
            raise ValueError('client raw SSE event schema differs')
        elapsed = finite_seconds(event['received_seconds'], 'client raw SSE arrival', maximum=profile_limits(profile)['client_seconds'])
        line = event['line']
        if elapsed < previous or elapsed > profile_limits(profile)['client_seconds'] or type(line) is not str:
            raise ValueError('client raw SSE timestamps or line differ')
        previous = elapsed
        total += len(line.encode('utf-8'))
        if len(line.encode('utf-8')) > 65536 or total > 1024 * 1024:
            raise ValueError('client raw SSE byte bound exceeded')
        physical = line[:-2] if line.endswith('\r\n') else line[:-1] if line.endswith('\n') else line
        if '\r' in physical or '\n' in physical:
            raise ValueError('client raw SSE entry contains multiple physical lines')
        if not line.strip():
            continue
        if done:
            raise ValueError('client raw SSE framing or post-DONE data differs')
        # Axum emits comments while a long prefill has no content to send.
        # They retain their byte/time bounds but cannot count as model output,
        # satisfy EOS/DONE, hide another physical line or extend a closed stream.
        if line.startswith(':'):
            continue
        if not line.startswith('data:'):
            raise ValueError('client raw SSE framing or post-DONE data differs')
        payload = line[5:].strip()
        if payload == '[DONE]':
            if role != 1 or finish != 1:
                raise ValueError('client raw SSE DONE before role/EOS')
            done, done_at = 1, elapsed
            continue
        value = strict_json(payload)
        require_fields(value, {'object': 'chat.completion.chunk', 'model': 'hag-context-qwen3-0.6b'})
        if type(value.get('id')) is not str or not value['id']:
            raise ValueError('client raw SSE missing response identity')
        response_id = response_id or value['id']
        if value['id'] != response_id:
            raise ValueError('client raw SSE response identity changed')
        choices = value.get('choices')
        if type(choices) is not list or len(choices) != 1:
            raise ValueError('client raw SSE choice count differs')
        choice = choices[0]
        require_fields(choice, {'index': 0})
        delta = choice.get('delta')
        if type(delta) is not dict:
            raise ValueError('client raw SSE delta missing')
        if delta == {'role': 'assistant'}:
            if role or content or finish or choice.get('finish_reason') is not None:
                raise ValueError('client raw SSE role order differs')
            role = 1
        elif set(delta) == {'content'}:
            if role != 1 or finish or choice.get('finish_reason') is not None or type(delta['content']) is not str:
                raise ValueError('client raw SSE content order differs')
            if delta['content']:
                content.append(delta['content'])
                times.append(elapsed)
        elif delta == {} and choice.get('finish_reason') == 'stop':
            if role != 1 or finish or not content:
                raise ValueError('client raw SSE EOS order differs')
            finish = 1
        else:
            raise ValueError('client raw SSE unexpected reasoning/tool/finish payload')
    if (role != 1 or finish != 1 or done != 1 or done_at is None or done_at <= 0
            or ''.join(content) != reference['text']):
        raise ValueError('client raw SSE lacks exact complete CPU reference')
    wire_request = {'model': 'hag-context-qwen3-0.6b', 'messages': fixture['messages'],
                    'max_tokens': fixture['max_tokens'], 'temperature': 0, 'stream': True}
    if profile in GROUNDED_PROFILES:
        validate_grounded_client(root, stem, record, wire_request, identity, 'stream', profile)
    else:
        require_fields({'request': read_json(root / (stem + '-client-request.json'))}, {'request': wire_request})
    client = record.get('client')
    client_keys = {'kind', 'pid', 'request_count', 'attempt_count', 'request', 'result', 'callbacks', 'response_complete', 'error'}
    if profile in GROUNDED_PROFILES:
        client_keys |= {'mode','max_evaluated_tokens','preflight','hooks'}
    if type(client) is not dict or set(client) != client_keys:
        raise ValueError('client result schema differs')
    require_fields(client, {'kind': 'antimatter', 'request_count': 1, 'attempt_count': 1,
                            'request': wire_request, 'response_complete': True, 'error': None})
    bounded_integer(client.get('pid'), 1, 2 ** 31 - 1, 'client PID')
    process = read_json(root / (stem + '-client-process.json'))
    if type(process) is not dict or set(process) != {'pid', 'exit_code', 'timed_out', 'identity'}:
        raise ValueError('client process receipt schema differs')
    require_fields(process, {'pid': client['pid'], 'exit_code': 0, 'timed_out': False, 'identity': identity})
    result = client.get('result')
    if type(result) is not dict or set(result) != {'content', 'reasoning', 'finishReason',
                                                 'servedModel', 'toolCalls', 'toolCallsCutOff'}:
        raise ValueError('provider parsed result schema differs')
    require_fields(result, {'content': reference['text'], 'reasoning': '', 'finishReason': 'stop',
                            'servedModel': 'hag-context-qwen3-0.6b', 'toolCalls': [], 'toolCallsCutOff': False})
    callbacks = client.get('callbacks')
    if type(callbacks) is not list or not 1 <= len(callbacks) <= 4096:
        raise ValueError('missing or oversized provider callbacks')
    delivered = []
    for callback in callbacks:
        require_fields(callback, {'id': 'hag-client-grounded' if profile in GROUNDED_PROFILES else 'hag-client-list'})
        if (set(callback) != {'id', 'content'}
                or type(callback.get('content')) is not str or not callback['content']):
            raise ValueError('provider callback schema differs')
        if callback['content']:
            delivered.append(callback['content'])
    if delivered != content:
        raise ValueError('provider callbacks differ from independent raw SSE chunks')
    # The client library's own timing observations are retained but do not set
    # qualified timing metrics; wire receipt times and matched API usage do.
    return {'round': index, 'fixture_id': fixture['id'],
            'prompt_tokens': reference['usage']['prompt_tokens'],
            'completion_tokens': reference['usage']['completion_tokens'],
            'content_chunks': len(content), 'ttfc_seconds': times[0],
            'last_content_seconds': times[-1], 'done_seconds': done_at,
            'completion_tokens_per_second': reference['usage']['completion_tokens'] / done_at,
            'usage_source': 'matched_nonstream_warmup_api'}


def probe_evidence(mode, text, root=ROOT):
    contract = approved_contract(root)
    if mode not in profile_modes(contract['profile']):
        raise ValueError('stage is not admitted by this profile')
    client = client_identity(root)
    if mode == 'self-check':
        if any(marker in text for marker in ('HAG_ANSWERS_REPORT', 'PASS hag-answers',
                                             'HAG_LIFECYCLE_REPORT', 'PASS hag-lifecycle',
                                             'HAG_GENERATED_HISTORY_REPORT', 'PASS hag-generated-history',
                                             'HAG_UI_HISTORY_REPORT', 'PASS hag-ui-history',
                                             'HAG_UI_STOP_REPORT', 'PASS hag-ui-stop')):
            raise ValueError('self-check unexpectedly contains answer evidence')
        return {'self_check': True, **({'client_identity': client} if client is not None else {})}
    if contract['profile'] == LIFECYCLE_PROFILE:
        if client is not None:
            raise ValueError('lifecycle profile does not use an optional client')
        from lifecycle import validate_stage
        return validate_stage(root, mode, text)
    if contract['profile'] == GENERATED_HISTORY_PROFILE:
        if client is None:
            raise ValueError('generated history requires the frozen Antimatter client')
        from generated_history import validate_stage
        report = validate_stage(root, mode, text)
        require_client_identity(report, client)
        return report
    if contract['profile'] == UI_HISTORY_PROFILE:
        if client is None:
            raise ValueError('UI history requires a frozen separate-host client identity')
        from ui_history import validate_stage
        report = validate_stage(root, mode, text)
        require_client_identity(report, client)
        return report
    if contract['profile'] == UI_STOP_PROFILE:
        if client is None:
            raise ValueError('UI stop/reuse requires a frozen client identity')
        from ui_stop import validate_stage
        report = validate_stage(root, mode, text)
        require_client_identity(report, client)
        return report
    if mode not in ('cpu-reference', 'answers', 'timing'):
        raise ValueError('unknown answer gate mode')
    lines = re.findall(r'^HAG_ANSWERS_REPORT (.+)$', text, re.MULTILINE)
    if len(lines) != 1:
        raise ValueError('need exactly one answer report')
    report = strict_json(lines[0])
    if type(report) is not dict:
        raise ValueError('answer report must be an object')
    require_client_identity(report, client)
    limits = profile_limits(contract['profile'])
    grounded = contract['profile'] in GROUNDED_PROFILES
    if grounded and client is None:
        raise ValueError('grounded profile requires frozen Antimatter identity')
    fixture_file = read_json(root / 'answer-fixtures.json')
    fixtures = fixture_file['fixtures']
    if (fixture_profile(fixture_file) != contract['profile']
            or [fixture.get('id') for fixture in fixtures] != contract['fixture_ids']
            or type(fixture_file.get('maximum_prompt_tokens')) is not int
            or fixture_file['maximum_prompt_tokens'] != limits['maximum_prompt_tokens']):
        raise ValueError('answer fixture scope differs')
    timing_id = fixture_file['timing_fixture_id']
    for fixture in fixtures:
        require_fields(fixture, {'max_tokens': limits['max_new_tokens'], 'temperature': 0})
    references = reference_payload(root, fixtures, contract['profile'])
    execution = 'cpu-reference' if mode == 'cpu-reference' else 'native-hag-' + mode
    require_fields(report, {'stage': mode, 'profile': contract['profile'], 'execution': execution, 'capacity': limits['capacity'],
                           'qualification_kind': contract['qualification_kind'],
                           'require_semantic_match': contract['require_semantic_match'],
                           'quality_is_diagnostic': not contract['require_semantic_match'],
                           'fixture_ids': [timing_id] if mode == 'timing' else contract['fixture_ids'],
                           'requests': contract['stage_request_counts'][mode],
                           'maximum_evaluated_positions': contract['stage_evaluation_ceilings'][mode],
                           'all_eos': True, 'closed_answers': True, 'final_unloaded': True,
                           'reference_sha256': digest(root / 'cpu-reference.json')})
    if mode == 'cpu-reference':
        require_fields(report, {'cpu_exit_code': 0, 'answers': references, 'audit_events': []})
        evaluated = [answer['evaluated_positions'] for answer in references]
    else:
        require_fields(report, {'hag_exit_code': 0, 'exact_cpu_parity': True,
                               'cpu_reference_result_sha256': digest(root / 'cpu-reference-result.json')})
        if mode == 'answers':
            require_fields(report, {'answers': references})
            if grounded:
                require_fields(report, {'client_chat_requests':1,'client_stream_requests':0,
                                       'model_free_client_refusals':1,'live_cancellation_qualified':False,
                                       'capacity_refusals':validate_capacity_refusals(root, fixtures[0], contract['profile'])})
                validate_grounded_chat(root, 'answers-' + fixtures[0]['id'],
                                       'answers-' + fixtures[0]['id'] + '-response.json',
                                       references[0], fixtures[0], client, contract['profile'])
            evaluated = [answer['evaluated_positions'] for answer in references]
        else:
            require_fields(report, {'answers_result_sha256': digest(root / 'answers-result.json'),
                                   'timing_fixture_id': timing_id, 'warmup_requests': 1,
                                   'measured_requests': limits['measured_requests'], 'descriptive_client_visible_only': True,
                                   'ttfc_definition': 'first_nonempty_raw_content_including_thinking_markup',
                                   'rate_definition': 'api_completion_tokens_over_request_to_done_seconds',
                                   'cpu_quota_percent': 75})
            reference = next(answer for answer in references if answer['fixture_id'] == timing_id)
            warmup = report.get('warmup')
            if type(warmup) is not dict:
                raise ValueError('missing lazy-load warmup evidence')
            require_fields(warmup, {'label': 'first_request_after_lazy_load',
                                   'fixture_id': timing_id, 'usage_source': 'nonstream_api',
                                   'generation': reference})
            finite_seconds(warmup.get('latency_seconds'), 'warmup', positive=True, maximum=limits['client_seconds'] if grounded else 180)
            if grounded:
                require_fields(report, {'client_chat_requests':1,'client_stream_requests':1,
                                       'model_free_client_refusals':2,'client_reuse':True,
                                       'live_cancellation_qualified':False})
                latency = validate_grounded_chat(root, 'timing-warmup', 'timing-warmup-response.json',
                                                  reference, fixtures[0], client, contract['profile'])
                require_fields(warmup, {'latency_seconds':latency})
            metrics = report.get('metrics')
            if type(metrics) is not list or len(metrics) != limits['measured_requests']:
                raise ValueError('warm SSE timing record count differs')
            for index, metric in enumerate(metrics, 1):
                if type(metric) is not dict:
                    raise ValueError('timing record must be an object')
                require_fields(metric, {'round': index, 'fixture_id': timing_id,
                                       'prompt_tokens': reference['usage']['prompt_tokens'],
                                       'completion_tokens': reference['usage']['completion_tokens'],
                                       'usage_source': 'matched_nonstream_warmup_api'})
                bounded_integer(metric.get('content_chunks'), 1, 4096, 'SSE content chunks')
                first = finite_seconds(metric.get('ttfc_seconds'), 'first content', maximum=limits['client_seconds'] if grounded else 180)
                last = finite_seconds(metric.get('last_content_seconds'), 'last content', maximum=limits['client_seconds'] if grounded else 180)
                done = finite_seconds(metric.get('done_seconds'), 'done', positive=True, maximum=limits['client_seconds'] if grounded else 180)
                if not first <= last <= done:
                    raise ValueError('SSE timestamps are unordered')
                rate = metric.get('completion_tokens_per_second')
                if (type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
                        or not math.isclose(rate, reference['usage']['completion_tokens'] / done,
                                            rel_tol=1e-9, abs_tol=1e-12)):
                    raise ValueError('client output rate differs from matched API token count / done time')
                if client is not None:
                    fixture = next(item for item in fixtures if item['id'] == timing_id)
                    observed = validate_antimatter_stream(root, reference, fixture, index, client, contract['profile'])
                    require_fields(metric, observed)
            evaluated = [reference['evaluated_positions']] * contract['stage_request_counts']['timing']
        events = report.get('audit_events')
        if type(events) is not list or len(events) != len(evaluated):
            raise ValueError('wrong number of HAG lifecycle audits')
        for index, (event, count) in enumerate(zip(events, evaluated), 1):
            require_fields(event, {'request_sequence': index, 'evaluated_positions': count,
                                  'outcome': 'success', 'reset_ok': True, 'poisoned': False})
        if grounded:
            raw_events = [strict_json(line) for line in re.findall(
                r'^HAG_SERVING_AUDIT (.+)$', bounded_text(root / (mode + '-hag-server.log')), re.MULTILINE)]
            require_fields({'events':raw_events}, {'events':events})
    bounded_integer(sum(evaluated), 1, contract['stage_evaluation_ceilings'][mode], 'stage native positions')
    require_fields(report, {'evaluated_positions': sum(evaluated)})
    finals = re.findall(r'^PASS hag-answers stage=(\S+) qualified=(\S+)$', text, re.MULTILINE)
    if finals != [(mode, 'true')]:
        raise ValueError('answer stage must pass exactly once')
    return report


def thermal_observation(samples_text, log):
    """Retain failure diagnostics without interpreting them as successful evidence."""
    samples = [strict_json(line) for line in samples_text.splitlines() if line.strip()]
    temperatures = [sample.get('tctl_c') for sample in samples if type(sample) is dict]
    temperatures = [value for value in temperatures
                    if type(value) in (int, float) and math.isfinite(value) and value >= 0]
    summaries = []
    for line in log.splitlines():
        if line.startswith('{'):
            try:
                value = strict_json(line)
            except json.JSONDecodeError:
                continue
            if type(value) is dict and 'process_started' in value:
                summaries.append(value)
    return {'max_tctl_c': max(temperatures) if temperatures else None,
            'samples': len(samples), 'summary': summaries[0] if len(summaries) == 1 else None}


def thermal_supervision_evidence(samples_text, log):
    """Validate the guard itself independently of the supervised child's exit."""
    observation = thermal_observation(samples_text, log)
    samples = [strict_json(line) for line in samples_text.splitlines() if line.strip()]
    if not samples or type(samples[0]) is not dict or samples[0].get('phase') != 'preflight':
        raise ValueError('missing thermal preflight evidence')
    previous = -1
    for index, sample in enumerate(samples, 1):
        if type(sample) is not dict:
            raise ValueError('thermal sample is not an object')
        temperature = sample.get('tctl_c')
        elapsed = sample.get('elapsed_s')
        if (sample.get('sensor_ok') is not True or sample.get('sample') != index
                or type(temperature) not in (int, float) or not math.isfinite(temperature)
                or not 0 <= temperature < 90 or type(elapsed) not in (int, float)
                or not math.isfinite(elapsed) or elapsed < previous):
            raise ValueError('invalid, missing or hot thermal sample')
        previous = elapsed
    summary = observation['summary']
    if summary is None:
        raise ValueError('missing or duplicate thermal supervisor summary')
    if (summary.get('ok') is not True or summary.get('aborted_for_heat') is not False
            or summary.get('abort_reason') is not None or summary.get('process_started') is not True
            or summary.get('residual_group_cleaned') is not False
            or type(summary.get('exit_code')) is not int
            or summary.get('samples') != len(samples)
            or summary.get('max_tctl_c') != observation['max_tctl_c']):
        raise ValueError('thermal supervisor did not finish cleanly')
    return observation


def thermal_evidence(samples_text, log):
    observation = thermal_supervision_evidence(samples_text, log)
    if observation['summary']['exit_code'] != 0:
        raise ValueError('child process exited ' + str(observation['summary']['exit_code']))
    return observation


def thermal_result(samples_text, log):
    observation = thermal_observation(samples_text, log)
    try:
        thermal_supervision_evidence(samples_text, log)
    except (ValueError, KeyError, TypeError) as error:
        summary = observation.get('summary') or {}
        reason = summary.get('abort_reason')
        if summary.get('aborted_for_heat') is True:
            return observation, 'thermal guard aborted: ' + str(reason or 'critical temperature reached')
        return observation, 'thermal evidence failed: ' + str(reason or error)
    if observation['summary']['exit_code'] != 0:
        return observation, 'child process exited ' + str(observation['summary']['exit_code'])
    return observation, None


def child_failures(child):
    if type(child) is not dict:
        raise ValueError('child result is not an object')
    failures = []
    if child.get('failure') is not None:
        failures.append('child supervisor stopped: ' + str(child['failure']))
    if type(child.get('exit_code')) is not int or child['exit_code'] != 0:
        failures.append('child process exited ' + str(child.get('exit_code')))
    if (child.get('memory_clean') is not True
            or not clean_memory(child['memory_before'], child['memory_after'])):
        failures.append('cgroup memory counters recorded pressure, OOM or swap')
    return failures



def require_retired_cpu(prior, session, root=ROOT, proc_root=Path('/proc'), unit_reader=None):
    """Admission evidence for the separate CPU process, before any native launch."""
    unit = prior.get('unit')
    if type(unit) is not str or not re.fullmatch(r'hag-answers-cpu-reference-[0-9]{8}-[0-9]{6}-[0-9]{6}', unit):
        raise ValueError('CPU prerequisite unit identity differs')
    if unit_reader is None:
        spec = importlib.util.spec_from_file_location('prerequisite_cleanup', root / 'closeout.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        unit_reader = module.inspect_unit
    properties = unit_reader(unit)
    require_fields(properties, {'ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlGroup':''})
    child = prior['child']
    cgroup = Path(child['cgroup'])
    if (not cgroup.is_relative_to('/sys/fs/cgroup') or cgroup.name != unit + '.service'
            or cgroup.exists() or cgroup.is_symlink()):
        raise ValueError('CPU prerequisite cgroup is unsafe or still present')
    pids = [child['pid'], prior['thermal']['summary']['pid']]
    if prior.get('profile') in (LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, *UI_PROFILES):
        if prior['profile'] == LIFECYCLE_PROFILE:
            from lifecycle import server_processes
        elif prior['profile'] == UI_HISTORY_PROFILE:
            from ui_history import server_processes
        elif prior['profile'] == UI_STOP_PROFILE:
            from ui_stop import server_processes
        else:
            from generated_history import server_processes
        processes = server_processes(root, 'cpu-reference')
        if type(processes) is not list or len(processes) != 1:
            raise ValueError('CPU prerequisite needs exactly one server')
        for process in processes:
            require_fields(process, {'exit_code': 0})
            pids.append(process['pid'])
        if len(set(pids)) != len(pids):
            raise ValueError('CPU prerequisite reused a process identity')
    for pid in pids:
        bounded_integer(pid,1,2**31-1,'CPU prerequisite PID')
        if (proc_root / str(pid)).exists():
            raise ValueError('CPU prerequisite process is still present')
    if datetime.fromisoformat(prior['end']) > datetime.fromisoformat(session['armed_at'].replace('Z','+00:00')):
        raise ValueError('GPU watcher was armed before CPU retirement stage completed')
    return {'unit':unit,'properties':properties,'cgroup':str(cgroup),'pids_absent':pids,
            'cpu_result_sha256':digest(root/'cpu-reference-result.json')}

def unit_command(unit, mode, profile='list-c128'):
    if mode not in profile_modes(profile):
        raise ValueError('unknown gate mode')
    service_seconds = profile_deadlines(profile)[mode][1]
    return ['systemd-run', '--user', '--pipe', '--wait', '--collect', '--expand-environment=no', f'--unit={unit}',
            '--property=MemoryMax=2G', '--property=MemorySwapMax=0', '--property=TasksMax=64',
            '--property=CPUQuota=75%', f'--property=RuntimeMaxSec={service_seconds}s', '--property=TimeoutStopSec=1s',
            '--property=KillMode=control-group', '--property=NoNewPrivileges=yes',
            '--property=OOMPolicy=stop', '--property=Restart=no',
            '/usr/bin/env', 'HAG_TRACE_SUBMISSION=0', '/usr/bin/python3', str(ROOT / 'gate-child.py'), mode]


def stop_unit(unit, output):
    try:
        stopped = subprocess.run(['systemctl', '--user', 'stop', unit], stdout=output,
                                 stderr=subprocess.STDOUT, timeout=5, check=False)
        return stopped.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    mode = parser.parse_args(argv).mode
    contract = approved_contract()
    if mode not in profile_modes(contract['profile']):
        raise ValueError('stage is not admitted by this profile')
    paths = {name: ROOT / (mode + suffix) for name, suffix in
             [('log', '.log'), ('result', '-result.json'), ('headroom', '-headroom.jsonl'),
              ('kernel', '-kernel.log'), ('driver', '-driver.log'), ('child', '-child-result.json'),
              ('ready', '-cgroup.json'), ('thermal', '-thermal.jsonl')]}
    if any(path.exists() for path in paths.values()):
        raise ValueError('gate evidence already exists; no retries or overwrites')
    initial_available = available_memory()
    require_headroom(initial_available)
    dependencies = dependency_identity()
    gate_files = gate_identity()
    input_provenance = provenance_identity()
    client = client_identity(ROOT)
    identity = None if mode == 'self-check' else binary_identity(mode)
    source_sha = None if mode == 'self-check' else source_identity()
    model_sha = None if mode == 'self-check' else checkpoint_identity()
    if contract['profile'] in C2048_PROFILES:
        project_cooldown(mode)
    session = capture_session(mode=mode) if mode in GPU_MODES else None
    if mode in GPU_MODES:
        prior = read_json(ROOT / 'cpu-reference-result.json')
        require_phase_diagnostics(prior, contract, 'cpu-reference')
        require_prior(prior, 'cpu-reference', identity, source_sha, gate_files,
                      input_provenance=input_provenance, profile=contract['profile'], client=client)
        if contract['profile'] in C2048_PROFILES:
            write_json(ROOT / (mode + '-cpu-retirement.json'), require_retired_cpu(prior, session))
        if probe_evidence('cpu-reference', bounded_text(ROOT / 'cpu-reference.log'), ROOT) != prior.get('probe'):
            raise ValueError('CPU reference prerequisite no longer matches its log')
    if mode == 'timing':
        prior = read_json(ROOT / 'answers-result.json')
        require_phase_diagnostics(prior, contract, 'answers')
        require_prior(prior, 'answers', identity, source_sha, gate_files, session, input_provenance, contract['profile'], client)
        if probe_evidence('answers', bounded_text(ROOT / 'answers.log'), ROOT) != prior.get('probe'):
            raise ValueError('answer prerequisite no longer matches its log')
    if mode != 'self-check':
        if not model_path(ROOT).is_file():
            raise ValueError('selected model file is missing')
    start = datetime.now(timezone.utc)
    unit = f'hag-answers-{mode}-{start.strftime("%Y%m%d-%H%M%S-%f")}'
    command = unit_command(unit, mode, contract['profile'])
    failures = []
    minimum = initial_available
    ready = None
    observed_peak = 0
    with paths['log'].open('x') as output, paths['headroom'].open('x') as samples:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + profile_deadlines(contract['profile'])[mode][2]
        next_capture_check = time.monotonic()
        try:
            while process.poll() is None:
                available = available_memory()
                minimum = min(minimum, available)
                sample = {'timestamp': timestamp(), 'available_bytes': available}
                if ready is None and paths['ready'].exists():
                    try:
                        ready = read_json(paths['ready'])
                    except json.JSONDecodeError:
                        pass  # The child may still be writing its small first record.
                if ready is not None:
                    cgroup = Path(ready['cgroup'])
                    if not cgroup.is_relative_to('/sys/fs/cgroup'):
                        raise ValueError('invalid child cgroup path')
                    try:
                        sample['cgroup_memory'] = memory_snapshot(cgroup)
                        observed_peak = max(observed_peak, sample['cgroup_memory']['memory_peak_bytes'])
                        require_process_headroom(sample['cgroup_memory'])
                    except FileNotFoundError:
                        pass  # The child final record retains stats before service cleanup.
                samples.write(json.dumps(sample) + '\n')
                samples.flush()
                if available < STOP_HEADROOM:
                    raise ValueError('host headroom fell below 4 GiB; stopping only this qualification unit')
                if mode in GPU_MODES and time.monotonic() >= next_capture_check:
                    if capture_session() != session:
                        raise ValueError('crash watcher session changed while serving test was running')
                    next_capture_check = time.monotonic() + 1
                if time.monotonic() >= deadline:
                    raise ValueError('outer supervisor deadline reached')
                time.sleep(0.1)
        except BaseException as error:
            failures.append(str(error))
            if not stop_unit(unit, output):
                failures.append('unit stop failed or timed out; manual inspection required')
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()  # Stop this client, never substitute for stopping its unit.
                process.wait(timeout=1)
        returncode = process.returncode
    # Retain a short window for driver diagnostics while crash capture remains armed.
    time.sleep(3)
    end = datetime.now(timezone.utc)
    with paths['kernel'].open('x') as output:
        try:
            journal = subprocess.run(['journalctl', '-k', '--no-pager', '-o', 'short-iso-precise',
                                      '--since', start.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                      '--until', end.strftime('%Y-%m-%d %H:%M:%S UTC')],
                                     stdout=output, stderr=subprocess.STDOUT, timeout=5, check=False)
            journal_code = journal.returncode
        except subprocess.TimeoutExpired:
            journal_code = 124
    relevant = []
    with paths['kernel'].open() as kernel, paths['driver'].open('x') as filtered:
        for line in kernel:
            if re.search(r'amdgpu|\bdrm\b|gpu|ring.*timeout|vm.*fault|reset|oom|out of memory|Killed process', line, re.IGNORECASE):
                filtered.write(line)
                if len(relevant) < 100:
                    relevant.append(line.rstrip())
    evidence = None
    thermal = None
    child = None
    capture_ok = mode not in GPU_MODES
    # Parse thermal evidence independently even when the child or probe failed.
    # A thermal cutoff must retain its cause, never be reported as an OOM by inference.
    log = None
    try:
        log = bounded_text(paths['log'])
        thermal, supervision_failure = thermal_result(bounded_text(paths['thermal'], 16 * 1024 ** 2 if contract['profile'] in C2048_PROFILES else 2 * 1024 ** 2), log)
        if supervision_failure:
            failures.append(supervision_failure)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append('thermal evidence unavailable: ' + str(error))
    try:
        child = read_json(paths['child'])
        require_phase_diagnostics(child, contract, mode)
        require_client_identity(child, client)
        if contract['profile'] in C2048_PROFILES:
            validate_project_child(child, mode, contract['profile'])
        failures.extend(reason for reason in child_failures(child) if reason not in failures)
        minimum = min(minimum, child['minimum_available_bytes'])
        observed_peak = max(observed_peak, child['memory_after']['memory_peak_bytes'])
        if (child.get('binary_sha256') != identity or child.get('limits') != EXPECTED_LIMITS
                or child.get('no_new_privileges') is not True):
            raise ValueError('child artifact or actual limits differ')
        if (child.get('source_manifest_sha256') != source_sha
                or child.get('dependencies') != dependencies or child.get('gate_files') != gate_files
                or child.get('input_provenance') != input_provenance):
            raise ValueError('child source/dependency/provenance identity differs')
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append('child evidence failed: ' + str(error))
    try:
        evidence = probe_evidence(mode, log if log is not None else bounded_text(paths['log']), ROOT)
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append('probe evidence failed: ' + str(error))
    try:
        if mode != 'self-check':
            if binary_identity(mode) != identity or source_identity() != source_sha:
                raise ValueError('binary/source identity changed during execution')
            if checkpoint_identity() != model_sha:
                raise ValueError('checkpoint identity changed during execution')
        if dependency_identity() != dependencies or gate_identity() != gate_files:
            raise ValueError('dependencies or gate inputs changed during execution')
        if provenance_identity() != input_provenance:
            raise ValueError('input provenance changed during execution')
        if client_identity(ROOT) != client:
            raise ValueError('optional client identity changed during execution')
        capture_ok = mode not in GPU_MODES or capture_session() == session
        if not capture_ok:
            raise ValueError('capture session changed after execution')
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(str(error))
    if returncode != 0:
        failures.append(f'service runner exited {returncode}')
    if journal_code != 0 or relevant:
        failures.append('kernel journal unavailable or contains GPU/OOM/fault messages')
    result = {'mode': mode, 'profile': contract['profile'], 'start': start.isoformat(), 'end': end.isoformat(), 'unit': unit,
              'qualification_kind': contract['qualification_kind'],
              'require_semantic_match': contract['require_semantic_match'],
              'quality_is_diagnostic': not contract['require_semantic_match'],
              'exit_code': returncode, 'binary_sha256': identity, 'source_manifest_sha256': source_sha,
              'checkpoint_sha256': model_sha, 'dependencies': dependencies, 'input_provenance': input_provenance, 'gate_files': gate_files, 'capture_session': session,
              'capture_ok': capture_ok, 'journal_exit_code': journal_code, 'driver_messages': relevant,
              'available_before_bytes': initial_available, 'minimum_available_bytes': minimum,
              'cgroup_memory_peak_bytes': observed_peak, 'child': child, 'probe': evidence, 'thermal': thermal,
              'failures': failures, 'qualified': not failures and capture_ok and minimum >= STOP_HEADROOM}
    if contract['profile'] in C2048_PROFILES:
        result['stage_deadlines'] = dict(zip(('child','service','outer'), profile_deadlines(contract['profile'])[mode]))
        result['private_cgroup_attestation'] = child.get('private_cgroup_attestation') if child else None
    if client is not None:
        result['client_identity'] = client
    diagnostics = phase_diagnostics(contract, mode)
    if diagnostics is not None:
        result['runtime_phase_diagnostics'] = diagnostics
    write_json(paths['result'], result)
    print(json.dumps(result, indent=2))
    return 0 if result['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
