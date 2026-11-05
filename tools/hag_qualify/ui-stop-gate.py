#!/usr/bin/env python3
"""Finite live Stop broker: real SSE, real disconnect, fenced reset, fresh reuse."""
import argparse
import base64
import hashlib
from http.server import HTTPServer
import importlib.util
import json
import os
from pathlib import Path
import select
import signal
import socket
import threading
import time
import uuid
from urllib.parse import urlsplit

import ui_stop as proof
import ui_client
from gate_common import write_json

ROOT = Path(__file__).resolve().parent
answer = proof.answer_helpers()
_spec = importlib.util.spec_from_file_location('ui_stop_history_worker', ROOT / 'ui-history-gate.py')
_shared = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared)
request_deadline = _shared.request_deadline
HEADER_SECONDS = 5


class Server(_shared.Server):
    def __init__(self, root, backend, mode):
        answer.Server.__init__(self, root, backend, mode, 2048, proof.PROFILE)
        self.request_seconds = 600 if backend == 'cpu' else 240
        self.record = {'pid': None, 'exit_code': None, 'idle_sigterm_exit': False}

    def process_identity(self):
        proof.require(self.process.poll() is None, 'server exited during Stop/reuse')
        self.capture_identity()
        return {key: self.record[key] for key in ('pid', 'start_time_ticks')}


def measure(root, turn, identities):
    # Reuse the bounded metadata subprocess implementation with this profile's
    # validator. A separate globals dictionary leaves the old profile unchanged.
    import types
    original = _shared._shared.measure
    namespace = {**original.__globals__, 'proof': proof}
    return types.FunctionType(original.__code__, namespace, original.__name__)(root, turn, identities)


class DisconnectWatch:
    """Observe the downstream socket independently of a blocked upstream read.

    The observer does not synthesize Stop or delay generation. EOF/reset alone
    triggers shutdown of the actual backend connection; replay still requires
    visible forwarded content and the native cancelled/reset audit.
    """
    def __init__(self, downstream, upstream):
        self.downstream, self.upstream = downstream, upstream
        self.stop = threading.Event()
        self.disconnected = threading.Event()
        self.observation = None
        self.failure = None
        self.thread = threading.Thread(target=self.run, name='ui-stop-disconnect', daemon=True)

    def run(self):
        try:
            while not self.stop.is_set():
                readable, _, exceptional = select.select([self.downstream], [], [self.downstream], .01)
                if not readable and not exceptional:
                    continue
                kind = None
                try:
                    data = self.downstream.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    if data == b'':
                        kind = 'eof'
                    elif data:
                        raise ValueError('extra downstream bytes after the admitted request')
                except (ConnectionResetError, ConnectionAbortedError):
                    kind = 'reset'
                except BlockingIOError:
                    continue
                if kind is None:
                    continue
                observed = time.monotonic()
                self.upstream.shutdown(socket.SHUT_RDWR)
                self.observation = {'disconnect_kind': kind, 'downstream_disconnect': observed,
                                    'upstream_shutdown': time.monotonic(),
                                    'shutdown': {'how': 'SHUT_RDWR', 'succeeded': True}}
                self.disconnected.set()
                return
        except Exception as error:
            self.failure = str(error)
            try:
                self.upstream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.disconnected.set()

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(1)
        proof.require(not self.thread.is_alive(), 'disconnect observer failed to retire')


class AdmissionBroker(HTTPServer):
    allow_reuse_address = False
    request_queue_size = 1

    def __init__(self, server, document, identities, reference):
        self.backend, self.root = server, server.root
        self.document, self.identities, self.reference = document, identities, reference
        self.failure, self.reuse, self.cancelled = None, None, None
        self.end_by = time.monotonic() + 350
        contract = proof.approved_contract(self.root)
        proof.equal(contract['profile'], proof.PROFILE, 'Stop broker profile differs')
        client = identities['client_identity']
        proof.equal(client, proof.read_json(self.root / 'client/identity.json'), 'Stop client identity differs')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        proof.equal(boot == client['local_host']['boot_id'], contract['ui_client_host'] == 'same',
                    'Stop broker boot differs from frozen host mode')
        port = urlsplit(ui_client.endpoint(client['base_url'])).port if contract['ui_client_host'] == 'same' else 0
        proof.require(port == 0 or 1024 <= port <= 65535, 'broker port must be unprivileged')
        super().__init__(('127.0.0.1', port), AdmissionHandler)
        self.timeout = .1
        self.ready = {'schema': 1, 'profile': proof.PROFILE, 'host': '127.0.0.1', 'port': self.server_port,
                      'pid': os.getpid(), 'run_id': str(uuid.uuid4()), 'boot_id': boot,
                      'client_identity_sha256': proof.digest(self.root / 'client/identity.json'), 'delivery': proof.DELIVERY}
        try:
            ui_client.client_host(self.root, client, self.ready)
        except Exception:
            self.server_close()
            raise
        self.record = {**self.ready, 'request_count': 0, 'forwarded_count': 0, 'response_count': 0,
                       'cancelled_count': 0, 'rejected_count': 0, 'closed': False,
                       'empty_connection_count': 0, 'empty_connections': []}

    def reject(self, error):
        self.failure = self.failure or str(error)
        self.record['rejected_count'] += 1

    def empty_connection(self, reason):
        """Retire bounded zero-byte pool sockets without admitting an HTTP request."""
        proof.require(reason in ('eof', 'reset'), 'unexpected empty connection reason')
        if self.record['empty_connection_count'] >= proof.MAX_EMPTY_CONNECTIONS:
            self.reject('empty UI connection budget exceeded')
            return
        self.record['empty_connection_count'] += 1
        self.record['empty_connections'].append({
            'sequence': self.record['empty_connection_count'], 'reason': reason,
            'received_bytes': 0, 'after_request_count': self.record['request_count'],
            'observed_monotonic': time.monotonic()})

    def admit(self, raw):
        turn = self.record['request_count']
        proof.require(turn in (1, 2), 'only two UI Stop requests are admitted')
        if turn == 2:
            proof.require(self.cancelled is not None and self.record['cancelled_count'] == 1,
                          'reuse arrived before verified cancellation/reset')
        received = time.monotonic()
        body = proof.strict_json(raw)
        proof.admit_request(turn, body, self.reference, self.document)
        stem = f'answers-turn-{turn}'
        write_json(self.root / (stem + '-request.json'), body)
        prompt = proof.measurement(self.root, turn, body, self.identities)
        write_json(self.root / (stem + '-lineage.json'), proof.lineage(self.root, 'answers', turn))
        self.record['forwarded_count'] += 1
        return turn, body, prompt, received

    def save_admission(self, turn, raw, received, returned):
        encoded, sha = base64.b64encode(raw).decode('ascii'), hashlib.sha256(raw).hexdigest()
        write_json(self.root / f'answers-turn-{turn}-admission.json', {
            'turn': turn, 'request_raw_base64': encoded, 'request_raw_sha256': sha,
            'forwarded_raw_base64': encoded, 'forwarded_raw_sha256': sha,
            'request_received_monotonic': received, 'response_body_sha256': hashlib.sha256(returned).hexdigest(),
            'reset_verified': True, 'parity_verified': turn == 2})

    def cancel_stream(self, handler, raw, body, received):
        started, before = time.monotonic(), self.backend.process_identity()
        captured, forwarded, pending = bytearray(), bytearray(), bytearray()
        state, wire = proof.SseState(), proof.SseState()
        error, first_content, watcher = None, None, None
        stream_seconds = self.backend.request_seconds - (time.monotonic() - started)
        proof.require(stream_seconds > 0, 'cancellation/reset exceeded original request deadline')
        with request_deadline(stream_seconds):
            connection, response = self.backend.open('POST', '/v1/chat/completions', body, raw=raw)
            try:
                proof.require(response.status == 200 and response.getheader('Content-Type', '').startswith('text/event-stream'),
                              'first native response is not SSE')
                proof.require(connection.sock is not None, 'upstream socket closed before streaming')
                handler.send_response(200)
                handler.send_header('Content-Type', response.getheader('Content-Type'))
                handler.send_header('Connection', 'close')
                handler.end_headers()
                handler.headers_sent = True
                with DisconnectWatch(handler.connection, connection.sock) as watcher:
                    while True:
                        try:
                            line = response.readline(proof.BODY_LIMIT + 1 - len(captured))
                        except (OSError, answer.http.client.HTTPException, ValueError) as failure:
                            partial = getattr(failure, 'partial', b'')
                            if partial:
                                captured.extend(partial[:proof.BODY_LIMIT + 1 - len(captured)])
                                state.feed(partial)
                            error = type(failure).__name__
                            break
                        if not line:
                            break
                        captured.extend(line)
                        proof.require(len(captured) <= proof.BODY_LIMIT, 'first SSE exceeds bound')
                        state.feed(line)
                        # Each complete backend frame is flushed immediately.
                        # No terminal event may be released as a cancellation.
                        proof.require(state.phase not in ('sentinel', 'done'), 'generation finished before UI Stop')
                        pending.extend(line)
                        if line not in (b'\n', b'\r\n'):
                            continue
                        if watcher.disconnected.is_set():
                            break
                        block = bytes(pending)
                        try:
                            handler.wfile.write(block)
                            handler.wfile.flush()
                        except OSError:
                            # A failed write alone is not a Stop receipt. The
                            # independent reader must observe EOF/reset as well.
                            watcher.disconnected.wait(.2)
                            break
                        forwarded.extend(block)
                        pending.clear()
                        wire.feed(block)
                        if wire.text and first_content is None:
                            first_content = time.monotonic()
                    if not watcher.disconnected.is_set():
                        watcher.disconnected.wait(.2)
                    proof.require(watcher.failure is None and watcher.observation is not None,
                                  'no genuine downstream disconnect: ' + str(watcher.failure))
                    proof.require(first_content is not None, 'client disconnected before streamed content')
                    state.partial()
                    proof.require(proof.visible_count(wire.partial()), 'Stop preceded visible numeric content')
            finally:
                response.close()
                connection.close()
                write_json(self.root / 'answers-turn-1-response.json', {
                    'method': 'POST', 'route': '/v1/chat/completions', 'request': body, 'status': response.status,
                    'content_type': response.getheader('Content-Type', ''), 'read_error': error,
                    'body_truncated': len(captured) > proof.BODY_LIMIT,
                    'raw_body_base64': base64.b64encode(captured[:proof.BODY_LIMIT]).decode('ascii')})
        # Streaming has retired its alarm. Control HTTP owns its own request
        # alarm, so retain one original budget without nesting signal timers.
        event, audit_observed, warm_observed, after = self.finish_cancellation(started, before)
        receipt = {'schema': 1, 'clock': 'monotonic_seconds', 'started': started,
                   'first_content_forwarded': first_content, **watcher.observation,
                   'audit_observed': audit_observed, 'warm_residency_observed': warm_observed,
                   'server_before': before, 'server_after': after,
                   'forwarded_raw_base64': base64.b64encode(forwarded).decode('ascii'),
                   'forwarded_sha256': hashlib.sha256(forwarded).hexdigest(),
                   'captured_sha256': hashlib.sha256(captured).hexdigest(), 'audit_event': event}
        # Proof replay also requires the eventual clean server retirement;
        # validate the live checks now and bind the exact receipt for reuse.
        proof.require(first_content <= receipt['downstream_disconnect'] <= receipt['upstream_shutdown']
                      and receipt['upstream_shutdown'] - receipt['downstream_disconnect'] <= 1,
                      'Stop socket clocks differ')
        write_json(self.root / 'answers-turn-1-cancellation.json', receipt)
        self.save_admission(1, raw, received, bytes(forwarded))
        proof.require(time.monotonic() - started <= self.backend.request_seconds,
                      'cancellation/reset exceeded original request deadline')
        self.cancelled = receipt
        self.record['cancelled_count'] += 1

    def finish_cancellation(self, started, before):
        """Share one remaining budget across audit and independently timed HTTP."""
        budget = self.backend.request_seconds
        def remaining():
            seconds = budget - (time.monotonic() - started)
            proof.require(seconds > 0, 'cancellation/reset exceeded original request deadline')
            return seconds
        with request_deadline(remaining()):
            event = proof.audit_event(self.backend.event(1), 1)
        audit_observed = time.monotonic()
        try:
            self.backend.request_seconds = remaining()
            self.backend.residency('after-turn-1', True)
        finally:
            self.backend.request_seconds = budget
        warm_observed = time.monotonic()
        after = self.backend.process_identity()
        proof.equal(after, before, 'server restarted after Stop')
        remaining()
        return event, audit_observed, warm_observed, after

    def complete_reuse(self, raw, body, prompt, received):
        status, payload = self.backend.recorded_json('answers-turn-2-response', 'POST', '/v1/chat/completions', body, raw=raw)
        value = proof.generation(status, payload, self.document, 2, prompt)
        proof.equal(value, self.reference['answers'][0], 'reuse differs from CPU reference')
        proof.audit_event(self.backend.event(2), 2, value['evaluated_positions'])
        self.backend.residency('after-turn-2', True)
        proof.equal(self.backend.process_identity(), self.cancelled['server_before'], 'reuse ran on another server')
        response = proof.read_json(self.root / 'answers-turn-2-response.json', 2 * proof.BODY_LIMIT)
        returned = proof.raw_response(response)
        self.save_admission(2, raw, received, returned)
        self.reuse = value
        return returned

    def serve_two(self):
        write_json(self.root / 'answers-ui-ready.json', self.ready)
        try:
            while self.record['response_count'] + self.record['cancelled_count'] < 2 and self.failure is None:
                proof.require(time.monotonic() < self.end_by, 'UI Stop stage deadline exceeded')
                self.handle_request()
            proof.require(self.failure is None, 'UI Stop admission failed: ' + str(self.failure))
            proof.equal([self.record[key] for key in ('request_count', 'forwarded_count', 'cancelled_count', 'response_count')],
                        [2, 2, 1, 1], 'UI Stop request count differs')
            return [self.reuse]
        finally:
            self.server_close()
            self.record['closed'] = True
            write_json(self.root / 'answers-ui-listener.json', self.record)


class AdmissionHandler(_shared.AdmissionHandler):
    def setup(self):
        super().setup()
        self.request.settimeout(HEADER_SECONDS)

    def handle(self):
        # Fetch pools may replace an aborted socket then close that replacement
        # before sending any HTTP bytes. Peek without consuming the request so
        # a real partial request still follows the strict HTTP parser path.
        self.handled_attempt = False
        empty = False
        try:
            remaining = self.server.end_by - time.monotonic()
            proof.require(remaining > 0, 'UI Stop stage deadline exceeded')
            with request_deadline(min(HEADER_SECONDS, remaining)):
                reason = 'eof'
                try:
                    first = self.connection.recv(1, socket.MSG_PEEK)
                except (ConnectionResetError, ConnectionAbortedError):
                    first, reason = b'', 'reset'
                if first == b'':
                    empty = True
                    self.close_connection = True
                    self.server.empty_connection(reason)
                    return
                self.handle_one_request()
        finally:
            if not empty and not self.handled_attempt:
                self.server.reject('incomplete UI HTTP request')

    def do_POST(self):
        self.handled_attempt, self.close_connection, self.headers_sent = True, True, False
        self.server.record['request_count'] += 1
        try:
            proof.equal(self.path, '/v1/chat/completions', 'unexpected UI route')
            proof.require(self.server.failure is None, 'admission is already closed')
            proof.require(self.headers.get_all('Transfer-Encoding') is None, 'chunked requests are refused')
            lengths = self.headers.get_all('Content-Length', [])
            proof.require(len(lengths) == 1 and lengths[0].isdigit(), 'one request length required')
            length = int(lengths[0])
            proof.require(0 < length <= 65536, 'request exceeds fixed byte bound')
            proof.equal(self.headers.get_all('Content-Type'), ['application/json'], 'request content type differs')
            with request_deadline(5):
                raw = self.rfile.read(length)
            proof.require(len(raw) == length, 'request body truncated')
            turn, body, prompt, received = self.server.admit(raw)
            if turn == 1:
                self.server.cancel_stream(self, raw, body, received)
                return
            returned = self.server.complete_reuse(raw, body, prompt, received)
            self.send_response(200)
            self.send_header('Content-Type', 'text/event-stream')
            self.send_header('Content-Length', str(len(returned)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.headers_sent = True
            self.wfile.write(returned)
            self.wfile.flush()
            self.server.record['response_count'] += 1
        except Exception as error:
            self.server.reject(error)
            if not self.headers_sent:
                try:
                    self.send_response(502)
                    self.send_header('Content-Length', '0')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                except OSError:
                    pass


def execute(root, mode):
    proof.require(mode in proof.MODES and os.environ.get('HAG_ANSWERS_GATE') == '1'
                  and os.environ.get('HAG_TRACE_SUBMISSION') == '0', 'launch only through frozen supervisor')
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
                    body = proof.request_body(document, turn)
                    stem = f'{mode}-turn-{turn}'
                    write_json(root / (stem + '-request.json'), body)
                    prompt = measure(root, turn, identities)
                    write_json(root / (stem + '-lineage.json'), proof.lineage(root, mode, turn))
                    if turn == 1:
                        continue  # metadata only; no extra count-up reference generation
                    status, payload = server.recorded_json(stem + '-response', 'POST', '/v1/chat/completions', body)
                    values.append(proof.generation(status, payload, document, turn, prompt))
                    server.residency('after-turn-2', True)
            else:
                values = AdmissionBroker(server, document, identities, expected).serve_two()
            server.unload()
            server.stop()
    finally:
        server.cleanup()
        server.save_process()
    proof.equal(proof.identity(root), identities, 'frozen identity changed during stage')
    if mode == 'cpu-reference':
        values, turns, _ = proof.reconstruct(root, mode, identities)
        write_json(root / 'cpu-reference.json', {'schema': 1, 'profile': proof.PROFILE, 'identities': identities,
                   'capacity': 2048, 'answers': values, 'turns': turns, 'cpu_exit_code': 0, 'final_unloaded': True})
    report = proof.summarize_stage(root, mode)
    write_json(root / (mode + '-report.json'), report)
    print(proof.PREFIX + json.dumps(report, allow_nan=False), flush=True)
    print(f'PASS hag-ui-stop stage={mode} qualified=true', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--stage', choices=proof.MODES, required=True)
    args = parser.parse_args()
    execute(ROOT, args.stage)
