"""Model-free loopback admission tests with a finite fake backend.

Only the admission listener is real. No Beholdd executable, editor, prompt
inspector, GPU device, external host or model is opened by these tests.
"""
import base64
from contextlib import contextmanager, nullcontext
import http.client
import importlib.util
import json
from pathlib import Path
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ui_history as proof
import inputs
from profiles import fixture_document

_spec = importlib.util.spec_from_file_location('test_ui_history_worker', Path(__file__).with_name('ui-history-gate.py'))
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)

TEXT = '<think>\n\n</think>\n\nready.'


def completion(turn, text=TEXT):
    prompt = 467 if turn == 1 else 491
    return {'id': f'chatcmpl-beholdd-{turn}', 'object': 'chat.completion', 'model': proof.MODEL_ID,
            'choices': [{'index': 0, 'finish_reason': 'stop',
                         'message': {'role': 'assistant', 'content': text}}],
            'usage': {'prompt_tokens': prompt, 'completion_tokens': 6, 'total_tokens': prompt + 6}}


class FakeBackend:
    def __init__(self, root, fault=None):
        self.root, self.fault = root, fault
        self.calls, self.order, self.returned = [], [], []

    def recorded_json(self, name, method, route, body=None, *, raw=None):
        turn = len(self.calls) + 1
        if method != 'POST' or route != '/v1/chat/completions' or raw is None or proof.strict_json(raw) != body:
            raise AssertionError('broker did not forward the original JSON request bytes')
        self.calls.append({'turn': turn, 'body': body, 'raw': raw})
        self.order.append(('forward', turn))
        payload = completion(turn, 'WRONG.' if self.fault == 'parity' else TEXT)
        # Preserve actual SSE framing/chunks to expose accidental response rewriting.
        base = {'id': payload['id'], 'object': 'chat.completion.chunk', 'created': 1730000000,
                'model': proof.MODEL_ID}
        def packet(delta, finish=None, usage=None):
            value = {**base, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
            if usage is not None:
                value['usage'] = usage
            return 'data: ' + json.dumps(value, separators=(',', ':')) + '\n\n'
        text = payload['choices'][0]['message']['content']
        returned = (': keep-alive\n\n' + packet({'role': 'assistant'}) + packet({'content': text[:5]})
                    + packet({'content': text[5:]}) + packet({}, 'stop', payload['usage'])
                    + 'data: [DONE]\n\n').encode('utf-8')
        self.returned.append(returned)
        worker.write_json(self.root / (name + '.json'), {
            'method': method, 'route': route, 'request': body, 'status': 200,
            'content_type': 'text/event-stream', 'read_error': None, 'body_truncated': False,
            'raw_body_base64': base64.b64encode(returned).decode('ascii')})
        return 200, payload

    def event(self, turn):
        self.order.append(('reset', turn))
        return {'request_sequence': turn, 'evaluated_positions': 473 if turn == 1 else 497,
                'outcome': 'success', 'reset_ok': self.fault != 'reset', 'poisoned': False}

    def residency(self, label, loaded):
        if loaded is not True:
            raise AssertionError('warm residency should be checked before releasing the response')
        self.order.append(('residency', label))


class UiGateTests(unittest.TestCase):
    @contextmanager
    def running(self, fault=None, *, host='separate'):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'client').mkdir()
            with socket.socket() as available:
                available.bind(('127.0.0.1', 0))
                client_port = available.getsockname()[1]
            boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
            client = {'kind': 'antimatter-ui', 'profile': proof.PROFILE, 'fixture': True,
                      'base_url': f'http://127.0.0.1:{client_port}/v1',
                      'local_host': {'boot_id': boot if host == 'same' else
                          ('262efb57-e5c3-4393-9144-6bf66a558a48' if boot != '262efb57-e5c3-4393-9144-6bf66a558a48'
                           else 'c170621f-9ed5-44cd-9309-f9a3441f3a21')}}
            worker.write_json(root / 'client/identity.json', client)
            worker.write_json(root / 'qualification-contract.json',
                              inputs.profile_contract(proof.PROFILE, 'antimatter-ui', ui_client_host=host))
            document = fixture_document(proof.PROFILE)
            worker.write_json(root / 'answer-fixtures.json', document)
            values = [proof.generation(200, completion(turn), document, turn, 467 if turn == 1 else 491,
                                       None if turn == 1 else TEXT) for turn in (1, 2)]
            reference = {'profile': proof.PROFILE, 'capacity': 2048, 'answers': values}
            worker.write_json(root / 'cpu-reference.json', reference)
            for turn in (1, 2):
                worker.write_json(root / f'cpu-reference-turn-{turn}-measurement.json', {'synthetic': turn})
            backend = FakeBackend(root, fault)
            broker = worker.AdmissionBroker(backend, document, {'client_identity': client}, reference)
            broker.end_by = time.monotonic() + 3
            outcome = {'values': None, 'error': None}
            done = threading.Event()
            def serve():
                try:
                    outcome['values'] = broker.serve_two()
                except Exception as error:
                    outcome['error'] = error
                finally:
                    done.set()
            # CPython installs SIGALRM handlers only in the main thread. Socket
            # protocol tests use an injected no-op alarm in their worker thread;
            # the real whole-request timer is tested separately on the main thread.
            with patch.object(proof, 'measurement', side_effect=lambda _root, turn, _body, _ids: 467 if turn == 1 else 491), \
                 patch.object(worker, 'request_deadline', side_effect=lambda _seconds: nullcontext()), \
                 patch.object(worker.signal, 'setitimer'):
                thread = threading.Thread(target=serve, name='model-free-ui-admission', daemon=True)
                thread.start()
                try:
                    yield root, document, backend, broker, outcome, done
                finally:
                    if not done.is_set():
                        broker.failure = broker.failure or 'test fixture teardown'
                    thread.join(4)
                    self.assertFalse(thread.is_alive(), 'the finite test listener must retire')

    def post(self, broker, raw):
        connection = http.client.HTTPConnection('127.0.0.1', broker.server_port, timeout=2)
        try:
            connection.request('POST', '/v1/chat/completions', body=raw,
                               headers={'Content-Type': 'application/json', 'Connection': 'close'})
            response = connection.getresponse()
            return response.status, response.getheader('Content-Type'), response.read(2 * proof.BODY_LIMIT)
        finally:
            connection.close()

    def raw_http(self, broker, request):
        with socket.create_connection(('127.0.0.1', broker.server_port), timeout=2) as connection:
            connection.sendall(request)
            connection.shutdown(socket.SHUT_WR)
            parts = []
            while True:
                try:
                    part = connection.recv(65536)
                except ConnectionResetError:
                    break
                if not part:
                    break
                parts.append(part)
                self.assertLess(sum(map(len, parts)), 65536)
            return b''.join(parts)

    def assert_closed(self, broker, done, outcome, backend, *, forwarded, replied):
        self.assertTrue(done.wait(2), 'a refused attempt must close admission immediately')
        self.assertIsInstance(outcome['error'], ValueError)
        self.assertEqual(len(backend.calls), forwarded)
        self.assertEqual(broker.record['forwarded_count'], forwarded)
        self.assertEqual(broker.record['response_count'], replied)
        self.assertGreaterEqual(broker.record['rejected_count'], 1)
        self.assertTrue(broker.record['closed'])
        with self.assertRaises(OSError):
            socket.create_connection(('127.0.0.1', broker.server_port), timeout=.2)

    def test_two_actual_loopback_requests_preserve_bytes_and_wait_for_reset(self):
        with self.running() as (root, document, backend, broker, outcome, done):
            first = proof.request_body(document, 1)
            raw_first = json.dumps(first, separators=(',', ':'), ensure_ascii=False).encode()
            status, content_type, returned_first = self.post(broker, raw_first)
            self.assertEqual((status, content_type), (200, 'text/event-stream'))
            self.assertEqual(returned_first, backend.returned[0])
            self.assertEqual(backend.order, [('forward', 1), ('reset', 1), ('residency', 'after-turn-1')])
            first_text = proof.http_body(proof.read_json(root / 'answers-turn-1-response.json'))['choices'][0]['message']['content']
            second = proof.request_body(document, 2, first_text)
            raw_second = json.dumps(second, indent=2).encode() + b'\n'
            status, content_type, returned_second = self.post(broker, raw_second)
            self.assertEqual((status, content_type), (200, 'text/event-stream'))
            self.assertEqual(returned_second, backend.returned[1])
            self.assertTrue(done.wait(2))
            self.assertIsNone(outcome['error'])
            self.assertEqual([call['raw'] for call in backend.calls], [raw_first, raw_second])
            self.assertEqual(backend.calls[1]['body']['messages'][2]['content'], first_text)
            self.assertEqual(len(outcome['values']), 2)
            listener = proof.listener(root)
            self.assertEqual([listener[key] for key in ('request_count', 'forwarded_count', 'response_count')], [2, 2, 2])
            self.assertEqual(listener['rejected_count'], 0)
            for turn, request in ((1, first), (2, second)):
                proof.admission(root, turn, request)
            self.assertEqual(backend.order[-3:], [('forward', 2), ('reset', 2), ('residency', 'after-turn-2')])
            with self.assertRaises(OSError):
                socket.create_connection(('127.0.0.1', broker.server_port), timeout=.2)

    def test_bad_first_prompt_never_reaches_backend(self):
        with self.running() as (_root, document, backend, broker, outcome, done):
            body = proof.request_body(document, 1)
            body['messages'][-1]['content'] = 'unregistered request'
            status, _, returned = self.post(broker, json.dumps(body).encode())
            self.assertEqual((status, returned), (502, b''))
            self.assert_closed(broker, done, outcome, backend, forwarded=0, replied=0)

    def test_same_host_binds_the_frozen_ui_port_and_delivers_two_requests(self):
        with self.running(host='same') as (root, document, backend, broker, outcome, done):
            client = proof.read_json(root / 'client/identity.json')
            self.assertEqual(broker.server_port, int(client['base_url'].split(':')[2].split('/')[0]))
            self.assertEqual(broker.ready['boot_id'], client['local_host']['boot_id'])
            for turn in (1, 2):
                request = proof.request_body(document, turn, None if turn == 1 else TEXT)
                raw = json.dumps(request).encode()
                status, content_type, body = self.post(broker, raw)
                self.assertEqual((status, content_type, body), (200, 'text/event-stream', backend.returned[turn - 1]))
                self.assertEqual(backend.calls[turn - 1]['raw'], raw)
            self.assertTrue(done.wait(2))
            self.assertIsNone(outcome['error'])
            self.assertTrue(proof.listener(root)['closed'])

    def test_false_topology_or_occupied_same_host_port_never_opens_admission(self):
        for host, wrong_boot in (('same', True), ('separate', False), ('same', False)):
            with self.subTest(host=host, wrong_boot=wrong_boot), tempfile.TemporaryDirectory() as directory, socket.socket() as occupied:
                root = Path(directory)
                (root / 'client').mkdir()
                occupied.bind(('127.0.0.1', 0))
                port = occupied.getsockname()[1]
                boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                client = {'kind': 'antimatter-ui', 'profile': proof.PROFILE, 'base_url': f'http://127.0.0.1:{port}/v1',
                          'local_host': {'boot_id': '262efb57-e5c3-4393-9144-6bf66a558a48' if wrong_boot else boot}}
                if wrong_boot and client['local_host']['boot_id'] == boot:
                    client['local_host']['boot_id'] = 'c170621f-9ed5-44cd-9309-f9a3441f3a21'
                worker.write_json(root / 'client/identity.json', client)
                worker.write_json(root / 'qualification-contract.json',
                                  inputs.profile_contract(proof.PROFILE, 'antimatter-ui', ui_client_host=host))
                document = fixture_document(proof.PROFILE)
                worker.write_json(root / 'answer-fixtures.json', document)
                backend = FakeBackend(root)
                with self.assertRaises(ValueError if wrong_boot or host == 'separate' else OSError):
                    worker.AdmissionBroker(backend, document, {'client_identity': client}, {})
                self.assertEqual(backend.calls, [])
                self.assertFalse((root / 'answers-ui-ready.json').exists())

    def test_first_parity_or_reset_failure_never_releases_reply_or_admits_turn_two(self):
        for fault in ('parity', 'reset'):
            with self.subTest(fault=fault), self.running(fault) as (root, document, backend, broker, outcome, done):
                status, _, returned = self.post(broker, json.dumps(proof.request_body(document, 1)).encode())
                self.assertEqual((status, returned), (502, b''))
                self.assert_closed(broker, done, outcome, backend, forwarded=1, replied=0)
                self.assertEqual(broker.values, [])
                self.assertFalse((root / 'answers-turn-1-admission.json').exists())
                self.assertFalse((root / 'answers-turn-2-request.json').exists())

    def test_second_request_cannot_trim_or_substitute_first_reply(self):
        with self.running() as (_root, document, backend, broker, outcome, done):
            status, _, _ = self.post(broker, json.dumps(proof.request_body(document, 1)).encode())
            self.assertEqual(status, 200)
            wrong_second = proof.request_body(document, 2, 'ready.')
            status, _, returned = self.post(broker, json.dumps(wrong_second).encode())
            self.assertEqual((status, returned), (502, b''))
            self.assert_closed(broker, done, outcome, backend, forwarded=1, replied=1)

    def test_truncated_oversized_duplicate_length_and_chunked_requests_fail_closed(self):
        variants = {
            'truncated-body': b'Content-Length: 20\r\nContent-Type: application/json\r\n\r\n{}',
            'oversized-body': b'Content-Length: 65537\r\nContent-Type: application/json\r\n\r\n',
            'duplicate-length': b'Content-Length: 2\r\nContent-Length: 2\r\nContent-Type: application/json\r\n\r\n{}',
            'chunked': b'Transfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n0\r\n\r\n',
            'malformed-json': b'Content-Length: 1\r\nContent-Type: application/json\r\n\r\n{',
            'duplicate-json-key': b'Content-Length: 15\r\nContent-Type: application/json\r\n\r\n{"x":1,"x":2}  ',
        }
        for name, tail in variants.items():
            with self.subTest(case=name), self.running() as (_root, _document, backend, broker, outcome, done):
                response = self.raw_http(broker, b'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n' + tail)
                self.assertIn(b' 502 ', response)
                self.assert_closed(broker, done, outcome, backend, forwarded=0, replied=0)

    def test_malformed_method_or_request_line_is_terminal(self):
        for raw in (b'GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n',
                    b'not valid request line here\r\n\r\n'):
            with self.subTest(raw=raw), self.running() as (_root, _document, backend, broker, outcome, done):
                self.raw_http(broker, raw)
                self.assert_closed(broker, done, outcome, backend, forwarded=0, replied=0)

    def test_truncated_header_or_empty_connection_cannot_be_followed_by_retry(self):
        for raw in (b'', b'POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Len'):
            with self.subTest(raw=raw), self.running() as (_root, _document, backend, broker, outcome, done):
                self.raw_http(broker, raw)
                self.assert_closed(broker, done, outcome, backend, forwarded=0, replied=0)

    def test_whole_request_deadline_interrupts_and_restores_signal_state(self):
        previous = worker.signal.getsignal(worker.signal.SIGALRM)
        with self.assertRaisesRegex(TimeoutError, 'request deadline exceeded'):
            with worker.request_deadline(.01):
                time.sleep(.1)
        self.assertEqual(worker.signal.getitimer(worker.signal.ITIMER_REAL), (0.0, 0.0))
        self.assertIs(worker.signal.getsignal(worker.signal.SIGALRM), previous)


if __name__ == '__main__':
    unittest.main()
