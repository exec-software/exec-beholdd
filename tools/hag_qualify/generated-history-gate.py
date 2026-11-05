#!/usr/bin/env python3
"""Execute exactly two dependent requests inside the frozen bounded supervisor."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

import generated_history as proof
from gate_common import PHASE_ENVIRONMENT, write_json

ROOT = Path(__file__).resolve().parent
answer = proof.answer_helpers()


class Server(answer.Server):
    def __init__(self, root, backend, mode):
        super().__init__(root, backend, mode, 2048, proof.PROFILE)
        self.record = {'pid': None, 'exit_code': None, 'idle_sigterm_exit': False}

    def capture_identity(self):
        pid = self.process.pid
        process = Path('/proc') / str(pid)
        membership = answer.verify_containment()
        proof.equal((process / 'cgroup').read_text(), Path('/proc/self/cgroup').read_text(),
                    'server left the worker cgroup')
        status = (process / 'status').read_text()
        ticks = int((process / 'stat').read_text().rsplit(')', 1)[1].split()[19])
        self.record.update(pid=pid, start_time_ticks=ticks,
                           cgroup=str(Path('/sys/fs/cgroup') / membership.lstrip('/')),
                           no_new_privileges=bool(re.search(r'^NoNewPrivs:\s+1$', status, re.M)))

    def recorded_json(self, name, method, route, body=None):
        connection, response = self.open(method, route, body)
        data, error = bytearray(), None
        try:
            try:
                while len(data) <= proof.BODY_LIMIT:
                    part = response.read(min(65536, proof.BODY_LIMIT + 1 - len(data)))
                    if not part:
                        break
                    data.extend(part)
            except (OSError, answer.http.client.HTTPException) as failure:
                data.extend(getattr(failure, 'partial', b'')[:proof.BODY_LIMIT + 1 - len(data)])
                error = type(failure).__name__
            truncated = len(data) > proof.BODY_LIMIT
            write_json(self.root / (name + '.json'), {
                'method': method, 'route': route, 'request': body, 'status': response.status,
                'content_type': response.getheader('Content-Type', ''), 'read_error': error,
                'body_truncated': truncated,
                'raw_body_base64': base64.b64encode(data[:proof.BODY_LIMIT]).decode('ascii')})
            proof.require(error is None and not truncated, 'HTTP body capture failed or exceeded its limit')
            proof.require(response.getheader('Content-Type', '') == 'application/json', 'unexpected HTTP content type')
            return response.status, proof.strict_json(data)
        finally:
            response.close()
            connection.close()

    def identity(self):
        self.capture_identity()
        status, value = self.recorded_json(self.stage + '-identity', 'GET', '/v1/beholdd/info')
        proof.equal(status, 200, 'server identity request failed')
        for key, expected in {'service': 'beholdd', 'runtime': 'native', 'generation': 2,
                              'execution_backends': ['cpu'] if self.backend == 'cpu' else ['cpu', 'hag']}.items():
            proof.equal(value.get(key), expected, 'server identity differs: ' + key)
        self.residency('cold', False)

    def residency(self, label, loaded):
        status, value = self.recorded_json(self.stage + '-' + label + '-model', 'GET', '/v1/models')
        proof.require(status == 200 and type(value.get('data')) is list and len(value['data']) == 1,
                      'model discovery differs')
        model = value['data'][0]
        proof.require(model.get('id') == proof.MODEL_ID and model.get('loaded') is loaded,
                      'model residency differs')
        proof.equal(model.get('execution_limits', {}).get('max_evaluated_tokens'),
                    2048 if self.backend == 'hag' else None, 'model public capacity differs')
        return model

    def unload(self):
        for index in (1, 2):
            status, value = self.recorded_json(f'{self.stage}-unload-{index}', 'POST',
                                               f'/v1/models/{proof.MODEL_ID}/unload', {})
            proof.equal(status, 200, 'unload failed')
            proof.equal(value, {'id': proof.MODEL_ID, 'loaded': False}, 'unload response differs')
        self.residency('unloaded', False)

    def save_process(self):
        if self.process is not None:
            self.record.update(pid=self.process.pid, exit_code=self.process.returncode,
                               idle_sigterm_exit=self.exited_cleanly)
        write_json(self.root / (self.stage + '-server-process.json'), self.record)


def measure(root, turn, identities):
    """Metadata-only inspection in the same CPU leaf, before any request send."""
    stem = f'cpu-reference-turn-{turn}'
    request = root / (stem + '-request.json')
    inspector = root / 'dependencies/prompt-inspect'
    proof.equal(proof.digest(inspector), identities['dependencies']['prompt-inspect'], 'inspector identity changed')
    command = [str(inspector), str(proof.model_path(root)), str(request)]
    environment = dict(os.environ)
    for name in (PHASE_ENVIRONMENT, 'HAG_SERVING_AUDIT', 'BEHOLDD_HAG_PRIVATE_CGROUP',
                 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'NODE_OPTIONS', 'NODE_PATH'):
        environment.pop(name, None)
    process, timed_out, started = None, False, time.monotonic()
    stdout_path, stderr_path = root / (stem + '-inspector.log'), root / (stem + '-inspector-stderr.log')
    # Pipes are drained in fixed chunks. Never allow an unbounded child write
    # to grow a capture file between polling samples.
    with stdout_path.open('xb') as stdout, stderr_path.open('xb') as stderr, selectors.DefaultSelector() as selector:
        try:
            process = subprocess.Popen(command, cwd=root, env=environment, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            counts = {'stdout': 0, 'stderr': 0}
            for stream, output, name in ((process.stdout, stdout, 'stdout'), (process.stderr, stderr, 'stderr')):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, (output, name))
            while selector.get_map() or process.poll() is None:
                if time.monotonic() - started >= 29:
                    timed_out = True
                    break
                for key, _ in selector.select(.02):
                    part = os.read(key.fileobj.fileno(), 65536)
                    if not part:
                        selector.unregister(key.fileobj)
                        continue
                    output, name = key.data
                    room = proof.BODY_LIMIT - counts[name]
                    output.write(part[:room])
                    counts[name] += len(part)
                    proof.require(counts[name] <= proof.BODY_LIMIT, 'inspector output exceeds bounded capture')
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=1)
            if process is not None:
                process.stdout.close()
                process.stderr.close()
    raw = proof.bounded_bytes(stdout_path)
    errors = proof.bounded_bytes(stderr_path)
    result = None
    try:
        result = proof.strict_json(raw)
    finally:
        write_json(root / (stem + '-measurement.json'), {
            'command': command, 'pid': None if process is None else process.pid,
            'exit_code': None if process is None else process.returncode, 'timed_out': timed_out,
            'elapsed_seconds': time.monotonic() - started, 'inspector_sha256': proof.digest(inspector),
            'request_sha256': proof.digest(request), 'model_sha256': identities['checkpoint_sha256'],
            'stdout_sha256': proof.digest(stdout_path), 'stderr_sha256': proof.digest(stderr_path), 'result': result})
    proof.require(process is not None and process.returncode == 0 and not timed_out and errors == b'',
                  'metadata inspection failed or timed out')
    return proof.measurement(root, turn, proof.read_json(request), identities)


def native_chat(server, turn, document, prompt, parent, identities):
    root, stem = server.root, f'answers-turn-{turn}'
    descriptor = proof.read_json(root / ('answers-cold-model.json' if turn == 1 else 'answers-after-turn-1-model.json'))
    descriptor = proof.http_body(descriptor)['data'][0]
    envelope = {'profile': proof.PROFILE, 'turn': turn, 'prompt_tokens': prompt,
                'request': proof.request_body(document, turn, None if parent is None else parent['text']),
                'model_descriptor': descriptor, 'history_parent': parent}
    request_path = root / (stem + '-client-request.json')
    output_path = root / (stem + '-response.json')
    write_json(request_path, envelope)
    environment = dict(os.environ)
    for name in ('NODE_OPTIONS', 'NODE_PATH', PHASE_ENVIRONMENT):
        environment.pop(name, None)
    process, timed_out = None, False
    with (root / (stem + '-client.log')).open('x') as log:
        try:
            process = subprocess.Popen([str(root / 'client/node'), str(root / 'generated-history-client.mjs'),
                                        '--root', str(root), '--port', str(server.port),
                                        '--request', str(request_path), '--output', str(output_path)],
                                       cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            try:
                process.wait(timeout=245)
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
                write_json(root / (stem + '-client-process.json'),
                           {'pid': process.pid, 'exit_code': process.returncode, 'timed_out': timed_out,
                            'identity': identities['client_identity']})
    proof.require(process is not None and process.returncode == 0 and not timed_out,
                  'native provider process failed or exceeded its deadline')
    return proof.native_response(root, turn, document, prompt, parent, identities)


def execute(root, mode):
    proof.require(mode in proof.MODES and os.environ.get('HAG_ANSWERS_GATE') == '1'
                  and os.environ.get('HAG_TRACE_SUBMISSION') == '0', 'launch only through the frozen supervisor')
    answer.verify_containment()
    identities, document = proof.identity(root), proof.recipe(root)
    expected = None
    if mode == 'answers':
        expected = proof.validate_stage(root, 'cpu-reference', answer.bounded_text(root / 'cpu-reference.log'))
        result = proof.read_json(root / 'cpu-reference-result.json')
        proof.equal(result.get('probe'), expected, 'CPU prerequisite changed')
        proof.require(result.get('qualified') is True and result.get('failures') == []
                      and type(result.get('exit_code')) is int and result['exit_code'] == 0, 'CPU prerequisite failed')
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    server = Server(root, backend, mode)
    values = []
    try:
        with server:
            server.identity()
            for turn in (1, 2):
                # This point is reached for turn two only after successful full
                # raw turn-one parity AND its fenced native reset receipt.
                parent = None if turn == 1 else proof.parent_binding(root, mode, values[0])
                request = proof.request_body(document, turn, None if parent is None else parent['text'])
                stem = f'{mode}-turn-{turn}'
                if mode == 'answers':
                    proof.equal(request, proof.read_json(root / f'cpu-reference-turn-{turn}-request.json'),
                                'native-derived second request differs from CPU history')
                write_json(root / (stem + '-request.json'), request)
                prompt = measure(root, turn, identities) if mode == 'cpu-reference' else proof.measurement(root, turn, request, identities)
                lineage = {'turn': turn, 'request_sha256': proof.digest(root / (stem + '-request.json')),
                           'history_parent': parent,
                           'measurement_sha256': proof.digest(root / f'cpu-reference-turn-{turn}-measurement.json')}
                if mode == 'answers':
                    lineage.update(cpu_reference_sha256=proof.digest(root / 'cpu-reference.json'),
                                   first_turn_parity_verified=turn == 2)
                write_json(root / (stem + '-lineage.json'), lineage)
                if mode == 'cpu-reference':
                    status, body = server.recorded_json(stem + '-response', 'POST', '/v1/chat/completions', request)
                    value = proof.generation(status, body, document, turn, prompt, None if parent is None else parent['text'])
                else:
                    value = native_chat(server, turn, document, prompt, parent, identities)
                    proof.equal(value, expected['answers'][turn - 1], 'native full response differs from CPU')
                    event = server.event(turn)
                    proof.equal(event, {'request_sequence': turn, 'evaluated_positions': value['evaluated_positions'],
                                        'outcome': 'success', 'reset_ok': True, 'poisoned': False},
                                'native request was not successfully fenced and reset')
                values.append(value)
                server.residency(f'after-turn-{turn}', True)
            server.unload()
            proof.events(root, mode, values)
            server.stop()
    finally:
        server.cleanup()
        server.save_process()
    proof.equal(proof.identity(root), identities, 'frozen identity changed during stage')
    if mode == 'cpu-reference':
        answers, turns, _ = proof.reconstruct(root, mode, identities)
        write_json(root / 'cpu-reference.json', {'schema': 1, 'profile': proof.PROFILE,
                   'identities': identities, 'capacity': 2048, 'answers': answers, 'turns': turns,
                   'cpu_exit_code': 0, 'final_unloaded': True})
    report = proof.summarize_stage(root, mode)
    write_json(root / (mode + '-report.json'), report)
    print(proof.PREFIX + json.dumps(report, allow_nan=False), flush=True)
    print(f'PASS hag-generated-history stage={mode} qualified=true', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--stage', choices=proof.MODES, required=True)
    args = parser.parse_args()
    execute(ROOT, args.stage)
