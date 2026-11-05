"""Model-free real loopback and socket-disconnect tests for the Stop broker."""
import base64
from contextlib import contextmanager, nullcontext
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_stop as proof
import inputs
from profiles import fixture_document
from test_ui_history import captured, completion
from test_ui_stop import partial

_spec = importlib.util.spec_from_file_location('test_ui_stop_worker', Path(__file__).with_name('ui-stop-gate.py'))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)


class FakeResponse:
    status = 200
    def __init__(self, source):
        self.file = source.makefile('rb')
    def getheader(self, name, default=None):
        return 'text/event-stream' if name == 'Content-Type' else default
    def readline(self, limit):
        return self.file.readline(limit)
    def close(self):
        self.file.close()


class FakeConnection:
    def __init__(self, sock):
        self.sock = sock
    def close(self):
        self.sock.close()


class FakeBackend:
    request_seconds = 2
    def __init__(self, root, fault=None):
        self.root, self.fault = root, fault
        self.calls, self.order = [], []
        self.upstream_closed = threading.Event()
        self.threads = []
        self.returned = None

    def process_identity(self):
        return {'pid': 121, 'start_time_ticks': 77}

    def open(self, method, route, body, *, raw):
        if proof.strict_json(raw) != body:
            raise AssertionError('incoming bytes were rewritten')
        self.calls.append(raw)
        self.order.append('first-forward')
        reader, producer = socket.socketpair()
        reader.settimeout(2)
        producer.settimeout(2)
        def stream():
            try:
                if self.fault == 'natural-eos':
                    producer.sendall(base64.b64decode(captured(completion(prompt=476))['raw_body_base64']))
                elif self.fault == 'no-content':
                    producer.sendall(partial()[:partial().find(b'\n\n') + 2])
                else:
                    producer.sendall(partial())
                # The finite fake producer blocks on its actual socket. This
                # specifically tests disconnect while the broker is waiting
                # for another frame; there is no model or producer slowdown.
                if producer.recv(1) == b'':
                    self.upstream_closed.set()
                    self.order.append('actual-upstream-close')
            except OSError:
                pass
            finally:
                producer.close()
        thread = threading.Thread(target=stream, daemon=True)
        self.threads.append(thread)
        thread.start()
        return FakeConnection(reader), FakeResponse(reader)

    def recorded_json(self, name, method, route, body=None, *, raw=None):
        if not self.upstream_closed.is_set():
            raise AssertionError('reuse preceded actual backend socket close')
        if proof.strict_json(raw) != body:
            raise AssertionError('reuse request was rewritten')
        self.calls.append(raw)
        self.order.append('reuse-forward')
        value = completion('WRONG.' if self.fault == 'reuse-parity' else '<think>\n\n</think>\n\nready.')
        record = captured(value, body)
        worker.write_json(self.root / (name + '.json'), record)
        self.returned = base64.b64decode(record['raw_body_base64'])
        return 200, value

    def event(self, turn):
        if not self.upstream_closed.wait(1):
            raise AssertionError('audit before actual connection shutdown')
        self.order.append(('audit', turn))
        return {'request_sequence': turn, 'evaluated_positions': 484 if turn == 1 else 473,
                'outcome': 'success' if turn == 2 or self.fault in ('success-audit', 'natural-eos') else 'cancelled',
                'reset_ok': self.fault != 'bad-reset', 'poisoned': False}

    def residency(self, label, loaded):
        if loaded is not True:
            raise AssertionError('warm residency required')
        self.order.append(('loaded', label))


class StopBrokerTests(unittest.TestCase):
    @contextmanager
    def running(self, fault=None, *, header_seconds=5):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'client').mkdir()
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            other = '262efb57-e5c3-4393-9144-6bf66a558a48'
            if other == boot:
                other = 'c170621f-9ed5-44cd-9309-f9a3441f3a21'
            client = {'kind': 'antimatter-ui', 'profile': proof.PROFILE, 'base_url': 'http://127.0.0.1:48321/v1', 'local_host': {'boot_id': other}}
            worker.write_json(root / 'client/identity.json', client)
            worker.write_json(root / 'qualification-contract.json', inputs.profile_contract(proof.PROFILE, 'antimatter-ui'))
            document = fixture_document(proof.PROFILE)
            worker.write_json(root / 'answer-fixtures.json', document)
            value = proof.generation(200, completion(), document, 2, 467)
            reference = {'profile': proof.PROFILE, 'capacity': 2048, 'answers': [value]}
            worker.write_json(root / 'cpu-reference.json', reference)
            for turn in (1, 2):
                worker.write_json(root / f'cpu-reference-turn-{turn}-measurement.json', {'synthetic': turn})
            worker.write_json(root / 'answers-server-process.json', {'pid': 121, 'start_time_ticks': 77})
            backend = FakeBackend(root, fault)
            broker = worker.AdmissionBroker(backend, document, {'client_identity': client}, reference)
            broker.end_by = time.monotonic() + 4
            outcome, done = {}, threading.Event()
            def serve():
                try:
                    outcome['values'] = broker.serve_two()
                except Exception as error:
                    outcome['error'] = error
                finally:
                    done.set()
            with patch.object(proof, 'measurement', side_effect=lambda _r, turn, _b, _i: proof.prompt_bound(turn)), \
                 patch.object(worker, 'request_deadline', side_effect=lambda _s: nullcontext()), \
                 patch.object(worker._shared, 'request_deadline', side_effect=lambda _s: nullcontext()), \
                 patch.object(worker._shared.signal, 'setitimer'), \
                 patch.object(worker, 'HEADER_SECONDS', header_seconds):
                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                try:
                    yield root, document, backend, broker, outcome, done
                finally:
                    broker.failure = broker.failure or ('fixture teardown' if not done.is_set() else None)
                    thread.join(5)
                    for producer in backend.threads:
                        producer.join(3)
                        self.assertFalse(producer.is_alive())
                    self.assertFalse(thread.is_alive())

    def cancel(self, broker, document, *, after_content=True):
        raw = json.dumps(proof.request_body(document, 1), separators=(',', ':')).encode()
        client = http.client.HTTPConnection('127.0.0.1', broker.server_port, timeout=2)
        response = None
        try:
            client.request('POST', '/v1/chat/completions', body=raw,
                           headers={'Content-Type': 'application/json', 'Connection': 'keep-alive'})
            response = client.getresponse()
            state = proof.SseState()
            while after_content and '1 2' not in state.text:
                line = response.readline(proof.BODY_LIMIT + 1)
                if not line:
                    break
                state.feed(line)
            # Actual caller-side TCP abort, not an injected broker callback.
            sock = response.fp.raw._sock
            sock.shutdown(socket.SHUT_RDWR)
            return raw, state.text
        finally:
            if response is not None:
                response.close()
            client.close()

    def reuse(self, broker, document):
        raw = json.dumps(proof.request_body(document, 2), indent=2).encode()
        client = http.client.HTTPConnection('127.0.0.1', broker.server_port, timeout=2)
        try:
            client.request('POST', '/v1/chat/completions', body=raw,
                           headers={'Content-Type': 'application/json', 'Connection': 'close'})
            response = client.getresponse()
            return raw, response.status, response.read(proof.BODY_LIMIT)
        finally:
            client.close()

    def empty_connection(self, broker):
        with socket.create_connection(('127.0.0.1', broker.server_port), timeout=2) as client:
            client.shutdown(socket.SHUT_WR)
            # Reading EOF proves the broker retired this connection before the
            # next request, instead of racing a later connection against it.
            self.assertEqual(client.recv(1), b'')

    def test_actual_disconnect_unblocks_upstream_read_then_allows_fresh_parity_reuse(self):
        with self.running() as (root, document, backend, broker, outcome, done):
            first, text = self.cancel(broker, document)
            self.assertIn('1 2', text)
            self.assertTrue(backend.upstream_closed.wait(1))
            # A second connection can arrive while reset is checked; admission
            # itself must occur only after the first handler finishes reset.
            second, status, returned = self.reuse(broker, document)
            self.assertEqual((status, returned), (200, backend.returned))
            self.assertTrue(done.wait(2))
            self.assertNotIn('error', outcome)
            self.assertEqual(backend.calls, [first, second])
            self.assertEqual(backend.order, ['first-forward', 'actual-upstream-close', ('audit', 1),
                 ('loaded', 'after-turn-1'), 'reuse-forward', ('audit', 2), ('loaded', 'after-turn-2')])
            proof.cancellation_receipt(root)
            for turn in (1, 2):
                proof.admission(root, turn, proof.request_body(document, turn))
            listener = proof.listener(root)
            self.assertEqual((listener['cancelled_count'], listener['response_count']), (1, 1))

    def test_empty_eof_after_actual_cancel_allows_exactly_one_fresh_reuse(self):
        with self.running() as (root, document, backend, broker, outcome, done):
            first, _ = self.cancel(broker, document)
            self.empty_connection(broker)
            second, status, returned = self.reuse(broker, document)
            self.assertEqual((status, returned), (200, backend.returned))
            self.assertTrue(done.wait(2))
            self.assertNotIn('error', outcome)
            self.assertEqual(backend.calls, [first, second])
            listener = proof.listener(root)
            self.assertEqual([listener[key] for key in ('request_count', 'forwarded_count',
                'cancelled_count', 'response_count', 'rejected_count', 'empty_connection_count')],
                [2, 2, 1, 1, 0, 1])
            row = listener['empty_connections'][0]
            self.assertEqual({key: row[key] for key in ('sequence', 'reason', 'received_bytes',
                             'after_request_count')},
                             {'sequence': 1, 'reason': 'eof', 'received_bytes': 0, 'after_request_count': 1})
            self.assertEqual(backend.order, ['first-forward', 'actual-upstream-close', ('audit', 1),
                ('loaded', 'after-turn-1'), 'reuse-forward', ('audit', 2), ('loaded', 'after-turn-2')])

    def test_empty_reset_after_actual_cancel_allows_fresh_reuse(self):
        with self.running() as (root, document, backend, broker, outcome, done):
            first, _ = self.cancel(broker, document)
            accepted = threading.Event()
            original_setup = worker.AdmissionHandler.setup
            def setup(handler):
                original_setup(handler)
                accepted.set()
            with patch.object(worker.AdmissionHandler, 'setup', setup):
                with socket.create_connection(('127.0.0.1', broker.server_port), timeout=2) as client:
                    # Ensure accept completed: an RST still in the listen queue
                    # can be discarded by the kernel before a handler sees it.
                    self.assertTrue(accepted.wait(1))
                    client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack('ii', 1, 0))
            # A following EOF socket is processed serially after the reset;
            # observing its close avoids scheduling-dependent counter checks.
            self.empty_connection(broker)
            second, status, returned = self.reuse(broker, document)
            self.assertEqual((status, returned), (200, backend.returned))
            self.assertTrue(done.wait(2))
            self.assertNotIn('error', outcome)
            self.assertEqual(backend.calls, [first, second])
            listener = proof.listener(root)
            self.assertEqual(listener['empty_connection_count'], 2)
            self.assertEqual([row['reason'] for row in listener['empty_connections']], ['reset', 'eof'])
            self.assertEqual(listener['rejected_count'], 0)

    def test_empty_connection_budget_exhaustion_rejects_without_reuse(self):
        with self.running() as (root, document, backend, broker, outcome, done):
            self.cancel(broker, document)
            for _ in range(proof.MAX_EMPTY_CONNECTIONS + 1):
                self.empty_connection(broker)
            self.assertTrue(done.wait(2))
            self.assertIn('empty UI connection budget exceeded', str(outcome.get('error')))
            self.assertEqual(broker.record['empty_connection_count'], proof.MAX_EMPTY_CONNECTIONS)
            self.assertEqual([broker.record[key] for key in ('request_count', 'forwarded_count',
                'cancelled_count', 'response_count', 'rejected_count')], [1, 1, 1, 0, 1])
            self.assertEqual(len(backend.calls), 1)
            self.assertFalse((root / 'answers-turn-2-request.json').exists())

    def test_partial_http_eof_after_cancel_is_rejected(self):
        for raw in (b'P', b'POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 10\r\n'):
            with self.subTest(raw=raw), self.running() as (root, document, backend, broker, outcome, done):
                self.cancel(broker, document)
                with socket.create_connection(('127.0.0.1', broker.server_port), timeout=2) as client:
                    client.sendall(raw)
                    client.shutdown(socket.SHUT_WR)
                    while client.recv(65536):
                        pass
                self.assertTrue(done.wait(2))
                self.assertIn('error', outcome)
                self.assertEqual(broker.record['empty_connection_count'], 0)
                self.assertEqual(broker.record['rejected_count'], 1)
                self.assertEqual(len(backend.calls), 1)
                self.assertFalse((root / 'answers-turn-2-request.json').exists())

    def test_idle_socket_after_cancel_times_out_instead_of_becoming_empty_eof(self):
        for raw in (b'', b'POST /v1/chat/completions HTTP/1.1\r\nX:'):
            with self.subTest(raw=raw), self.running(header_seconds=.1) as (root, document, backend, broker, outcome, done):
                self.cancel(broker, document)
                with socket.create_connection(('127.0.0.1', broker.server_port), timeout=2) as client:
                    if raw:
                        client.sendall(raw)
                    self.assertTrue(done.wait(2))
                self.assertIn('incomplete UI HTTP request', str(outcome.get('error')))
                self.assertEqual(broker.record['empty_connection_count'], 0)
                self.assertEqual(broker.record['rejected_count'], 1)
                self.assertEqual(len(backend.calls), 1)
                self.assertFalse((root / 'answers-turn-2-request.json').exists())

    def test_natural_eos_or_bad_reset_fails_without_reuse(self):
        for fault in ('natural-eos', 'success-audit', 'bad-reset'):
            with self.subTest(fault=fault), self.running(fault) as (root, document, backend, broker, outcome, done):
                if fault == 'natural-eos':
                    self.cancel(broker, document, after_content=False)
                else:
                    self.cancel(broker, document)
                self.assertTrue(done.wait(2))
                self.assertIn('error', outcome)
                self.assertEqual(len(backend.calls), 1)
                self.assertEqual(broker.record['response_count'], 0)
                self.assertFalse((root / 'answers-turn-2-request.json').exists())

    def test_cancellation_before_streamed_content_fails_closed(self):
        with self.running('no-content') as (root, document, backend, broker, outcome, done):
            self.cancel(broker, document, after_content=False)
            self.assertTrue(done.wait(2))
            self.assertIn('error', outcome)
            self.assertEqual(broker.record['cancelled_count'], 0)
            self.assertFalse((root / 'answers-turn-1-cancellation.json').exists())

    def test_reuse_parity_failure_never_releases_second_response(self):
        with self.running('reuse-parity') as (_root, document, backend, broker, outcome, done):
            self.cancel(broker, document)
            _, status, returned = self.reuse(broker, document)
            self.assertEqual((status, returned), (502, b''))
            self.assertTrue(done.wait(2))
            self.assertIn('error', outcome)
            self.assertEqual((broker.record['cancelled_count'], broker.record['response_count']), (1, 0))

    def test_real_stream_deadline_retires_before_real_residency_http_alarm(self):
        # Run cancellation on the main thread with real SIGALRM contexts. The
        # backend's GET uses the production recorded_json alarm; the prior
        # nested stream+residency timer must fail this test.
        class ControlResponse:
            status = 200
            def __init__(self):
                self.raw = json.dumps({'data': [{'id': proof.MODEL_ID, 'loaded': True,
                                    'execution_limits': {'max_evaluated_tokens': 2048}}]}).encode()
            def read(self, limit):
                value, self.raw = self.raw[:limit], self.raw[limit:]
                return value
            def getheader(self, name, default=None):
                return 'application/json' if name == 'Content-Type' else default
            def close(self):
                pass

        class RealControlBackend(FakeBackend):
            recorded_json = worker.Server.recorded_json
            def open(self, method, route, body=None, *, raw=None):
                if method == 'GET':
                    self.control_timeout = self.request_seconds
                    return SimpleNamespace(close=lambda: None), ControlResponse()
                return super().open(method, route, body, raw=raw)
            def residency(self, label, loaded):
                status, value = self.recorded_json('answers-' + label + '-model', 'GET', '/v1/models')
                self.assertion = status == 200 and value['data'][0]['loaded'] is loaded

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = RealControlBackend(root)
            downstream, client = socket.socketpair()
            downstream.settimeout(2); client.settimeout(2)
            writer = downstream.makefile('wb', buffering=0)
            reader = client.makefile('rb')
            client_result = {}
            def click_stop():
                try:
                    state = proof.SseState()
                    while '1 2' not in state.text:
                        state.feed(reader.readline(proof.BODY_LIMIT))
                    client.shutdown(socket.SHUT_RDWR)
                    client_result['stopped'] = True
                except Exception as error:
                    client_result['error'] = error
            peer = threading.Thread(target=click_stop, daemon=True)
            handler = SimpleNamespace(connection=downstream, wfile=writer,
                                      send_response=lambda *_: None, send_header=lambda *_: None,
                                      end_headers=lambda: None, headers_sent=False)
            broker = object.__new__(worker.AdmissionBroker)
            broker.backend, broker.root = backend, root
            broker.record, broker.cancelled = {'cancelled_count': 0}, None
            document = fixture_document(proof.PROFILE)
            body = proof.request_body(document, 1)
            raw = json.dumps(body).encode()
            peer.start()
            try:
                broker.cancel_stream(handler, raw, body, time.monotonic())
                self.assertEqual(broker.record['cancelled_count'], 1)
                self.assertTrue(backend.assertion)
                self.assertGreater(backend.control_timeout, 0)
                self.assertLess(backend.control_timeout, 2)
                self.assertEqual(backend.request_seconds, 2)
                self.assertEqual(worker.signal.getitimer(worker.signal.ITIMER_REAL), (0., 0.))
            finally:
                reader.close(); writer.close(); client.close(); downstream.close()
                peer.join(3)
                for producer in backend.threads:
                    producer.join(3)
                self.assertFalse(peer.is_alive())
            self.assertEqual(client_result, {'stopped': True})

    def test_remaining_audit_deadline_and_residency_timeout_restore(self):
        broker = object.__new__(worker.AdmissionBroker)
        before = {'pid': 121, 'start_time_ticks': 77}
        event = {'request_sequence': 1, 'evaluated_positions': 484, 'outcome': 'cancelled',
                 'reset_ok': True, 'poisoned': False}
        backend = SimpleNamespace(request_seconds=.01, event=lambda _turn: time.sleep(.1),
                                  residency=lambda *_: self.fail('expired audit reached residency'),
                                  process_identity=lambda: before)
        broker.backend = backend
        with self.assertRaises(TimeoutError):
            broker.finish_cancellation(time.monotonic(), before)
        self.assertEqual(backend.request_seconds, .01)
        self.assertEqual(worker.signal.getitimer(worker.signal.ITIMER_REAL), (0., 0.))
        backend.request_seconds = 2
        backend.event = lambda _turn: event
        def fail_residency(*_):
            self.assertLess(backend.request_seconds, 2)
            raise ValueError('fixture control failure')
        backend.residency = fail_residency
        with self.assertRaisesRegex(ValueError, 'fixture control failure'):
            broker.finish_cancellation(time.monotonic(), before)
        self.assertEqual(backend.request_seconds, 2)
        self.assertEqual(worker.signal.getitimer(worker.signal.ITIMER_REAL), (0., 0.))

    def test_unregistered_first_request_does_not_reach_backend(self):
        with self.running() as (_root, document, backend, broker, outcome, done):
            # A fresh reuse request cannot substitute for the count-up request.
            _, status, returned = self.reuse(broker, document)
            self.assertEqual((status, returned), (502, b''))
            self.assertTrue(done.wait(2))
            self.assertIn('error', outcome)
            self.assertEqual(backend.calls, [])


if __name__ == '__main__':
    unittest.main()
