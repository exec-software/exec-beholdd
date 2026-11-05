#!/usr/bin/env python3
"""Two current-wire CPU requests, or a two-request native UI admission broker.

The broker is loopback-only and releases a buffered SSE reply only after exact
CPU parity and a healthy native reset. It never starts an editor or SSH session.
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import signal
import time
import uuid
from urllib.parse import urlsplit

import ui_history as proof
import ui_client
from gate_common import write_json

ROOT = Path(__file__).resolve().parent
answer = proof.answer_helpers()
_spec = importlib.util.spec_from_file_location('ui_shared_generated_gate', ROOT / 'generated-history-gate.py')
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)


@contextmanager
def request_deadline(seconds):
    """A whole-request clock, including slow headers and partial body reads."""
    previous = signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError('request deadline cannot replace an active alarm')
    def expired(_signum, _frame):
        raise TimeoutError('UI gate request deadline exceeded')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class Server(_shared.Server):
    def __init__(self, root, backend, mode):
        answer.Server.__init__(self, root, backend, mode, 2048, proof.PROFILE)
        self.request_seconds = 600 if backend == 'cpu' else 240
        self.record = {'pid': None, 'exit_code': None, 'idle_sigterm_exit': False}

    def recorded_json(self, name, method, route, body=None, *, raw=None):
        with request_deadline(self.request_seconds):
            connection, response = self.open(method, route, body, raw=raw)
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
                record = {
                    'method': method, 'route': route, 'request': body, 'status': response.status,
                    'content_type': response.getheader('Content-Type', ''), 'read_error': error,
                    'body_truncated': truncated,
                    'raw_body_base64': base64.b64encode(data[:proof.BODY_LIMIT]).decode('ascii')}
                write_json(self.root / (name + '.json'), record)
                proof.require(error is None and not truncated, 'incomplete or oversized response')
                if route == '/v1/chat/completions':
                    return response.status, proof.http_body(record)
                proof.equal(response.getheader('Content-Type', ''), 'application/json', 'control response differs')
                return response.status, proof.strict_json(data)
            finally:
                response.close()
                connection.close()


def lineage(root, mode, turn, parent):
    stem = f'{mode}-turn-{turn}'
    binding = {'turn': turn, 'request_sha256': proof.digest(root / (stem + '-request.json')),
               'history_parent': parent,
               'measurement_sha256': proof.digest(root / f'cpu-reference-turn-{turn}-measurement.json')}
    if mode == 'answers':
        binding.update(cpu_reference_sha256=proof.digest(root / 'cpu-reference.json'),
                       first_turn_parity_verified=turn == 2)
    write_json(root / (stem + '-lineage.json'), binding)


class AdmissionBroker(HTTPServer):
    """Single-threaded, finite listener. Any rejected/failed attempt ends admission."""
    allow_reuse_address = False
    request_queue_size = 1

    def __init__(self, server, document, identities, reference):
        self.backend = server
        self.root = server.root
        self.document, self.identities, self.reference = document, identities, reference
        self.values = []
        self.failure = None
        self.end_by = time.monotonic() + 350
        contract = proof.approved_contract(self.root)
        proof.equal(contract['profile'], proof.PROFILE, 'UI broker profile differs')
        client = identities['client_identity']
        proof.equal(client, proof.read_json(self.root / 'client/identity.json'), 'UI broker client identity differs')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        proof.equal(boot == client['local_host']['boot_id'], contract['ui_client_host'] == 'same',
                    'UI broker boot differs from frozen host mode')
        port = urlsplit(ui_client.endpoint(client['base_url'])).port if contract['ui_client_host'] == 'same' else 0
        proof.require(port == 0 or 1024 <= port <= 65535, 'UI broker endpoint port must be unprivileged')
        super().__init__(('127.0.0.1', port), AdmissionHandler)
        self.timeout = .1
        self.ready = {'schema': 1, 'profile': proof.PROFILE, 'host': '127.0.0.1',
                      'port': self.server_port, 'pid': os.getpid(), 'run_id': str(uuid.uuid4()),
                      'client_identity_sha256': proof.digest(self.root / 'client/identity.json'),
                      'boot_id': boot,
                      'delivery': 'buffered-sse-after-parity-reset'}
        try:
            ui_client.client_host(self.root, client, self.ready)
        except Exception:
            self.server_close()
            raise
        self.record = {**self.ready, 'request_count': 0, 'forwarded_count': 0,
                       'response_count': 0, 'rejected_count': 0, 'closed': False}

    def reject(self, error):
        self.failure = self.failure or str(error)
        self.record['rejected_count'] += 1

    def process(self, raw):
        turn = self.record['request_count']
        proof.require(turn in (1, 2), 'only two UI requests are admitted')
        parent = None if turn == 1 else proof.parent_binding(self.root, 'answers', self.values[0])
        body = proof.strict_json(raw)
        proof.admit_request(turn, body, parent, self.reference, self.document)
        stem = f'answers-turn-{turn}'
        write_json(self.root / (stem + '-request.json'), body)
        prompt = proof.measurement(self.root, turn, body, self.identities)
        lineage(self.root, 'answers', turn, parent)
        self.record['forwarded_count'] += 1
        status, payload = self.backend.recorded_json(stem + '-response', 'POST', '/v1/chat/completions', body, raw=raw)
        value = proof.generation(status, payload, self.document, turn, prompt, None if parent is None else parent['text'])
        proof.equal(value, self.reference['answers'][turn - 1], 'native answer differs from current-wire CPU reference')
        proof.equal(self.backend.event(turn), {'request_sequence': turn, 'evaluated_positions': value['evaluated_positions'],
                    'outcome': 'success', 'reset_ok': True, 'poisoned': False}, 'native reply has no healthy fenced reset')
        self.backend.residency(f'after-turn-{turn}', True)
        response = proof.read_json(self.root / (stem + '-response.json'), 2 * proof.BODY_LIMIT)
        returned = base64.b64decode(response['raw_body_base64'], validate=True)
        encoded = base64.b64encode(raw).decode('ascii')
        digest = hashlib.sha256(raw).hexdigest()
        write_json(self.root / (stem + '-admission.json'), {
            'turn': turn, 'request_raw_base64': encoded, 'request_raw_sha256': digest,
            'forwarded_raw_base64': encoded, 'forwarded_raw_sha256': digest,
            'response_body_sha256': hashlib.sha256(returned).hexdigest(),
            'parity_verified': True, 'reset_verified': True})
        self.values.append(value)
        return returned

    def serve_two(self):
        write_json(self.root / 'answers-ui-ready.json', self.ready)
        try:
            while self.record['response_count'] < 2 and self.failure is None:
                proof.require(time.monotonic() < self.end_by, 'UI broker stage deadline exceeded')
                self.handle_request()
            proof.require(self.failure is None, 'UI admission failed: ' + str(self.failure))
            proof.equal([self.record[key] for key in ('request_count', 'forwarded_count', 'response_count')],
                        [2, 2, 2], 'UI broker request count differs')
            return self.values
        finally:
            self.server_close()
            self.record['closed'] = True
            write_json(self.root / 'answers-ui-listener.json', self.record)


class AdmissionHandler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
        pass

    def setup(self):
        # The outer stage clock still bounds a peer that dribbles request headers.
        self.request.settimeout(5)
        super().setup()

    def handle(self):
        self.handled_attempt = False
        try:
            # One connection owns exactly one complete request. EOF and header
            # timeouts count as failed attempts even when http.server swallows them.
            with request_deadline(5):
                self.handle_one_request()
        finally:
            if not self.handled_attempt:
                self.server.reject('incomplete UI HTTP request')

    def parse_request(self):
        result = super().parse_request()
        if result:
            # Header admission has finished. do_POST sets a separate body-read
            # deadline, then the backend owns its full 240-second request clock.
            signal.setitimer(signal.ITIMER_REAL, 0)
        return result

    def send_error(self, code, message=None, explain=None):
        self.handled_attempt = True
        self.server.reject(f'HTTP {code}: {message}')
        self.close_connection = True
        super().send_error(code, message, explain)

    def do_POST(self):
        self.handled_attempt = True
        self.close_connection = True
        self.server.record['request_count'] += 1
        try:
            proof.equal(self.path, '/v1/chat/completions', 'unexpected UI route')
            proof.require(self.server.failure is None, 'admission already closed')
            proof.require(self.headers.get_all('Transfer-Encoding') is None, 'chunked UI requests are not admitted')
            lengths = self.headers.get_all('Content-Length', [])
            proof.require(len(lengths) == 1 and lengths[0].isdigit(), 'one request length is required')
            length = int(lengths[0])
            proof.require(0 < length <= 65536, 'UI request exceeds fixed byte bound')
            proof.equal(self.headers.get_all('Content-Type'), ['application/json'], 'UI content type differs')
            with request_deadline(5):
                raw = self.rfile.read(length)
            proof.require(len(raw) == length, 'UI request body truncated')
            returned = self.server.process(raw)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(returned)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(returned)
            self.wfile.flush()
            self.server.record['response_count'] += 1
        except Exception as error:
            self.server.reject(error)
            # Do not forward a failed/reset-uncertain completion, or try another inference.
            try:
                self.send_response(502)
                self.send_header('Content-Length', '0')
                self.send_header('Connection', 'close')
                self.end_headers()
            except OSError:
                pass


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
    server = Server(root, 'cpu' if mode == 'cpu-reference' else 'hag', mode)
    values = []
    try:
        with server:
            server.identity()
            if mode == 'cpu-reference':
                for turn in (1, 2):
                    parent = None if turn == 1 else proof.parent_binding(root, mode, values[0])
                    body = proof.request_body(document, turn, None if parent is None else parent['text'])
                    stem = f'{mode}-turn-{turn}'
                    write_json(root / (stem + '-request.json'), body)
                    prompt = _shared.measure(root, turn, identities)
                    lineage(root, mode, turn, parent)
                    status, payload = server.recorded_json(stem + '-response', 'POST', '/v1/chat/completions', body)
                    values.append(proof.generation(status, payload, document, turn, prompt, None if parent is None else parent['text']))
                    server.residency(f'after-turn-{turn}', True)
            else:
                values = AdmissionBroker(server, document, identities, expected).serve_two()
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
    print(f'PASS hag-ui-history stage={mode} qualified=true', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--stage', choices=proof.MODES, required=True)
    args = parser.parse_args()
    execute(ROOT, args.stage)
