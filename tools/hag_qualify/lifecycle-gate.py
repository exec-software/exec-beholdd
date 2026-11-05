#!/usr/bin/env python3
"""Real TCP lifecycle execution, only inside the frozen bounded supervisor."""
import argparse
import base64
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import time

import lifecycle as proof
from gate_common import (approved_contract, gate_identity, memory_snapshot, private_cgroup_attestation,
                         EXPECTED_LIMITS, write_json)
from profiles import LIFECYCLE_PROFILE

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('answer_gate', ROOT / 'answer-gate.py')
answer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(answer)


class Server(answer.Server):
    def __init__(self, root, backend, mode):
        super().__init__(root, backend, mode, 2048, LIFECYCLE_PROFILE)
        self.mode = mode
        self.record = {'mode': mode, 'server_pid': None, 'server_exit_code': None,
                       'snapshots': [], 'socket_shutdown': None}

    def recorded_json(self, stem, method, route, body=None):
        connection, response = self.open(method, route, body)
        try:
            data = bytearray()
            read_error = None
            try:
                while len(data) <= proof.BODY_LIMIT:
                    part = response.read(min(65536, proof.BODY_LIMIT + 1 - len(data)))
                    if not part:
                        break
                    data.extend(part)
            except (OSError, answer.http.client.HTTPException) as error:
                data.extend(getattr(error, 'partial', b'')[:proof.BODY_LIMIT + 1 - len(data)])
                read_error = type(error).__name__
            truncated = len(data) > proof.BODY_LIMIT
            write_json(self.root / (self.mode + '-' + stem + '-http.json'),
                       {'method': method, 'route': route, 'request': body, 'status': response.status,
                        'content_type': response.getheader('Content-Type', ''),
                        'truncated': truncated, 'read_error': read_error,
                        'body_base64': base64.b64encode(data[:proof.BODY_LIMIT]).decode('ascii')})
            proof.require(read_error is None and not truncated, 'HTTP body capture failed or exceeded its limit')
            return response.status, proof.strict_json(data)
        finally:
            response.close()
            connection.close()

    def complete(self, stem):
        status, value = self.recorded_json(stem, 'POST', '/v1/chat/completions', proof.request_body())
        return proof.generation(status, value)

    def residency(self, stem, loaded):
        self.recorded_json(stem + '-model', 'GET', '/v1/models')
        proof.model_artifact(self.root, self.mode + '-' + stem + '-model', loaded, self.backend)

    def snapshot(self, label):
        proof.require(self.process.poll() is None, 'server exited before lifecycle snapshot')
        pid = self.process.pid
        process = Path('/proc') / str(pid)
        cgroup = Path('/sys/fs/cgroup') / answer.verify_containment().lstrip('/')
        proof.equal((process / 'cgroup').read_text(), Path('/proc/self/cgroup').read_text(),
                    'server left the worker cgroup')
        limits = {name: (cgroup / name).read_text().strip() for name in EXPECTED_LIMITS}
        status = (process / 'status').read_text()
        memory = memory_snapshot(cgroup)
        attested = private_cgroup_attestation(cgroup, limits, status, '0', memory)
        if self.backend == 'hag':
            proof.equal(os.environ.get('BEHOLDD_HAG_PRIVATE_CGROUP'), attested,
                        'server private leaf attestation differs')
        stat = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        self.record['server_pid'] = pid
        self.record['snapshots'].append({
            'label': label, 'pid': pid, 'start_time_ticks': int(stat[19]),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'cgroup': str(cgroup), 'limits': limits, 'memory': memory,
            'no_new_privileges': bool(re.search(r'^NoNewPrivs:\s+1$', status, re.MULTILINE)),
            'private_cgroup_attestation': attested if self.backend == 'hag' else None})

    def event(self, count):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            events = proof.audit_events(self.log_path, final=False)
            proof.require(len(events) <= count, 'unexpected native request')
            if len(events) == count:
                event = events[-1]
                expected = 'cancelled' if count == 3 else 'success'
                proof.require(event['outcome'] == expected, 'native request outcome differs')
                proof.require(9 <= event['evaluated_positions'] <= 24 if count == 3
                              else event['evaluated_positions'] == 9, 'native evaluation count differs')
                return event
            proof.require(self.process.poll() is None, 'server exited while awaiting audit')
            time.sleep(.05)
        raise ValueError('native audit deadline exceeded')

    def stream(self, reference, cancel=False):
        stem = self.mode + ('-cancel' if cancel else '-stream')
        body = proof.request_body(True, cancel)
        connection, response = self.open('POST', '/v1/chat/completions', body)
        try:
            write_json(self.root / (stem + '-headers.json'),
                       {'request': body, 'status': response.status,
                        'content_type': response.getheader('Content-Type', '')})
            proof.require(response.status == 200 and response.getheader('Content-Type', '').startswith('text/event-stream'),
                          'SSE status/content type differs')
            state, total = proof.SseResult(), 0
            with (self.root / (stem + '.sse')).open('xb') as output:
                while True:
                    line = response.readline(proof.BODY_LIMIT + 1)
                    if not line:
                        break
                    total += len(line)
                    proof.require(total <= proof.BODY_LIMIT, 'oversized SSE body')
                    output.write(line)
                    output.flush()
                    state.line(line)
                    if cancel and state.content:
                        state.finish(reference, cancel=True)
                        status, value = self.recorded_json('busy-unload', 'POST', f'/v1/models/{proof.MODEL_ID}/unload', {})
                        proof.require(status == 409 and value.get('error', {}).get('code') == 'runtime_busy',
                                      'active generation did not refuse unload')
                        proof.require(connection.sock is not None, 'stream socket already closed')
                        connection.sock.shutdown(socket.SHUT_RDWR)
                        self.record['socket_shutdown'] = {'how': 'SHUT_RDWR', 'succeeded': True}
                        return
            proof.require(not cancel, 'stream ended before cancellable first content')
            state.finish(reference)
        finally:
            response.close()
            connection.close()

    def unload_recorded(self, prefix='unload'):
        for number in (1, 2):
            status, value = self.recorded_json(f'{prefix}-{number}', 'POST', f'/v1/models/{proof.MODEL_ID}/unload', {})
            proof.equal(status, 200, 'unload failed')
            proof.equal(value, {'id': proof.MODEL_ID, 'loaded': False}, 'unload response differs')
        self.residency('unloaded' if prefix == 'unload' else 'final-unloaded', False)


def execute(root, mode):
    proof.require(os.environ.get('HAG_ANSWERS_GATE') == '1' and os.environ.get('HAG_TRACE_SUBMISSION') == '0',
                  'launch through the frozen bounded supervisor')
    answer.verify_containment()
    contract = approved_contract(root)
    proof.require(contract['profile'] == LIFECYCLE_PROFILE, 'lifecycle profile required')
    gate_identity(root)
    ids = answer.identity(root)
    reference = None
    if mode == 'answers':
        cpu = proof.validate_stage(root, 'cpu-reference', answer.bounded_text(root / 'cpu-reference.log'))
        result = proof.read_json(root / 'cpu-reference-result.json')
        proof.equal(result.get('probe'), cpu, 'CPU prerequisite changed')
        proof.require(result.get('qualified') is True and result.get('failures') == [], 'CPU prerequisite failed')
        reference = proof.read_json(root / 'cpu-reference.json')['answer']
    backend = 'cpu' if mode == 'cpu-reference' else 'hag'
    server = Server(root, backend, mode)
    try:
        with server:
            server.identity()
            server.recorded_json('identity', 'GET', '/v1/beholdd/info')
            server.residency('cold', False)
            server.snapshot('cold')
            first = server.complete('hello')
            if mode == 'answers':
                proof.equal(first, reference, 'native first completion differs from CPU')
                server.event(1)
            server.snapshot('generated')
            if mode == 'answers':
                server.residency('loaded', True)
                server.stream(reference)
                server.event(2)
                server.snapshot('streamed')
                server.stream(reference, cancel=True)
                server.event(3)
                server.snapshot('cancelled')
                proof.equal(server.complete('reuse'), reference, 'reuse differs from CPU')
                server.event(4)
                server.snapshot('reused')
            server.unload_recorded()
            server.snapshot('unloaded')
            if mode == 'answers':
                proof.equal(server.complete('reload'), reference, 'reload differs from CPU')
                server.event(5)
                server.residency('reloaded', True)
                server.snapshot('reloaded')
                server.unload_recorded('final-unload')
                server.snapshot('final-unloaded')
            server.stop()
            server.record['server_exit_code'] = server.process.returncode
    finally:
        write_json(root / ('lifecycle-' + mode + '-record.json'), server.record)
    if mode == 'cpu-reference':
        write_json(root / 'cpu-reference.json', {'schema': 1, 'profile': LIFECYCLE_PROFILE, 'identities': ids,
                   'answer': first, 'cpu_exit_code': 0, 'final_unloaded': True})
    report = proof.summarize_stage(root, mode)
    write_json(root / (mode + '-report.json'), report)
    print('HAG_LIFECYCLE_REPORT ' + json.dumps(report, allow_nan=False), flush=True)
    print(f'PASS hag-lifecycle stage={mode} qualified=true', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('cpu-reference', 'answers'))
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    execute(ROOT, args.stage)


if __name__ == '__main__':
    main()
