"""Stop/New chat UI evidence joined to real shared process and retirement checks.

The backend cancellation validator has its own raw-socket fixture suite. Only
that boundary is mocked here; SSE parsing, identity, supervision and UI report
validation remain active. No UI, server, model, compiler or SSH process runs.
"""
import base64
from copy import deepcopy
import json
import unittest
from unittest.mock import patch

try:
    from . import gate_common as common, ui_client as ui, ui_stop as backend, ui_stop_client as stop
    from . import test_ui_client as fixtures
except ImportError:
    import gate_common as common
    import ui_client as ui
    import ui_stop as backend
    import ui_stop_client as stop
    import test_ui_client as fixtures

identity, put = fixtures.identity, fixtures.put


def frame(turn, delta, finish=None, usage=None):
    value = {'id': f'chatcmpl-beholdd-{turn}', 'object': 'chat.completion.chunk', 'created': 1,
             'model': backend.MODEL_ID, 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
    if usage is not None:
        value['usage'] = usage
    return ('data: ' + json.dumps(value) + '\n\n').encode()


class UiStopClientTests(fixtures.UiBase):
    rebind = fixtures.UiEvidenceTests.rebind

    def setUp(self):
        super().setUp()
        # The fixture identity is independent of the production pin, including while
        # that pin is deliberately UNPINNED during implementation.
        pin = patch.object(ui, 'UI_STOP_COMMIT', 'a' * 40)
        pin.start()
        self.addCleanup(pin.stop)

    def evidence(self, mode='separate'):
        expected, _old_remote = fixtures.UiEvidenceTests.evidence(self, mode)
        expected.update(profile=ui.UI_STOP_PROFILE, commit=ui.UI_STOP_COMMIT)
        self.seal(expected, mode)
        contract = common.read_json(self.root / 'qualification-contract.json')
        contract['profile'] = ui.UI_STOP_PROFILE
        put(self.root / 'qualification-contract.json', contract)
        digest = common.digest(self.root / 'client/identity.json')
        for name in ('answers-ui-ready.json', 'answers-ui-listener.json'):
            value = common.read_json(self.root / name)
            value.update(profile=ui.UI_STOP_PROFILE, client_identity_sha256=digest,
                         delivery=backend.DELIVERY)
            if name.endswith('listener.json'):
                value.update(response_count=1, cancelled_count=1)
            put(self.root / name, value)

        partial_chunks = ['<think>\n\n</think>\n\n', '1']
        partial_raw = frame(1, {'role': 'assistant'}) + b''.join(frame(1, {'content': part}) for part in partial_chunks)
        forwarded = partial_raw + frame(1, {'content': ' 2'})
        full = '<think>\n\n</think>\n\nready.'
        usage = {'prompt_tokens': 467, 'completion_tokens': 6, 'total_tokens': 473}
        reuse_raw = (frame(2, {'role': 'assistant'}) + frame(2, {'content': full})
                     + frame(2, {}, 'stop', usage) + b'data: [DONE]\n\n')
        report = common.read_json(self.root / 'ui-report.json')
        requests, turns, remote, threads = [], [], [], []
        for index, raw in enumerate((partial_raw, reuse_raw)):
            wire = backend.request_body(backend.fixture_document(backend.PROFILE), index + 1)
            thread_id, answer_id = f'conversation-{index}', f'answer-{index}'
            messages = [{'id': f'user-{index}', **deepcopy(wire['messages'][-1])}]
            request = {'providerId': 'hag-ui-fixture', 'model': backend.MODEL_ID, 'threadId': thread_id,
                       'requestId': answer_id, 'responseMessageId': answer_id,
                       'useTools': True, 'messages': messages}
            chunks = partial_chunks if index == 0 else [full]
            content = ''.join(chunks)
            started, ended = (1000, 1007) if index == 0 else (1010, 1020)
            turn = {'turn': index, 'request': deepcopy(request), 'requestId': answer_id, 'wire': wire,
                    'rawResponse': raw.decode(), 'usage': None if index == 0 else usage,
                    'responseStatus': 200, 'transport': 'sse', 'contentType': 'text/event-stream',
                    'servedModel': None if index == 0 else backend.MODEL_ID, 'finishReason': None if index == 0 else 'stop',
                    'attemptCount': 1, 'providerCalls': 1, 'maxTokens': 64 if index == 0 else 16,
                    'completed': bool(index), 'settled': True, 'outcome': 'cancelled' if index == 0 else 'finished',
                    'timedOut': False, 'error': 'the run was cancelled' if index == 0 else None,
                    'cancelRequests': 1 if index == 0 else 0, 'abortObserved': index == 0,
                    'abortListenerRemoved': True, 'transportClosed': True, 'responseEof': False,
                    'bodyCancelled': bool(index),
                    'resumed': False, 'deadlineMs': 240000, 'transportFragments': 2, 'transportBytes': len(raw),
                    'startedAt': started, 'settledAt': ended, 'contentChunks': deepcopy(chunks),
                    'firstContentAt': 1002 if index == 0 else 1012,
                    'cancelRequestedAt': 1004 if index == 0 else None,
                    'abortObservedAt': 1005 if index == 0 else None,
                    'transportClosedAt': 1006 if index == 0 else 1019,
                    'checkpointed': 1, 'providerResult': None if index == 0 else {'content': full, 'finishReason': 'stop'},
                    'result': None if index == 0 else {'reason': 'finished', 'rounds': 1, 'toolCalls': 0, 'content': full}}
            saved = messages + [{'id': answer_id, 'role': 'assistant',
                'content': content + ('\n\n_Stopped._' if index == 0 else ''),
                'meta': {'failed': False, 'createdAt': started, 'finishedAt': ended,
                         'chunks': len(chunks) + (1 if index == 0 else 0)}}]
            remote_raw = forwarded if index == 0 else raw
            remote.append({'turn': index + 1, 'method': 'POST', 'route': '/v1/chat/completions', 'request': wire,
                           'status': 200, 'content_type': 'text/event-stream', 'read_error': None,
                           'body_truncated': False, 'raw_body_base64': base64.b64encode(remote_raw).decode()})
            requests.append(request)
            turns.append(turn)
            threads.append({'id': thread_id, 'messages': saved})
        report.update(requests=requests, providerPipeline={'transport': 'loopback-http', 'turns': turns},
                      persisted={'activeThreadId': threads[1]['id'], 'threads': threads}, checkpointCount=6,
                      interruption={'kind': 'cancel', 'consoleErrors': [], 'providerErrors': [],
                                    'cancellations': [{'runId': 'answer-0', 'accepted': True}]},
                      stopReuse={'firstThreadId': 'conversation-0', 'firstRequestId': 'answer-0',
                                 'secondThreadId': 'conversation-1',
                                 'stop': {'clickedAt': 1003, 'visibleTextBeforeStop': '1'},
                                 'stoppedCheckpoint': deepcopy(threads[0]['messages'][1]),
                                 'newChatClickedAt': 1008,
                                 'freshCheckpoint': {'id': 'conversation-1', 'messages': []}})
        report['live']['profile'] = ui.UI_STOP_PROFILE
        put(self.root / 'ui-report.json', report)
        process = common.read_json(self.root / 'ui-process.json')
        process.update(identity_sha256=digest, ready_sha256=common.digest(self.root / 'answers-ui-ready.json'))
        next(row for row in process['processes'] if row['role'] == 'electron')['command'].append('--live-stop-reuse')
        put(self.root / 'ui-process.json', process)
        retired = common.read_json(self.root / 'ui-retirement.json')
        retired['identity_sha256'] = digest
        put(self.root / 'ui-retirement.json', retired)
        self.rebind()
        cancellation = {'forwarded_raw_base64': base64.b64encode(forwarded).decode()}
        # The validator's returned digest must bind a real retained file even though
        # validation of this backend-only document is mocked in this suite.
        put(self.root / 'answers-turn-1-cancellation.json', cancellation)
        return expected, remote, cancellation

    def validate(self, expected, remote, cancellation):
        with patch.object(backend, 'cancellation_receipt', return_value=cancellation) as prove:
            result = stop.validate_ui_evidence(self.root, expected, remote)
            prove.assert_called_once_with(self.root)
            return result

    def mutate_report(self, change):
        report = common.read_json(self.root / 'ui-report.json')
        change(report)
        put(self.root / 'ui-report.json', report)
        self.rebind()

    def test_success_joins_actual_stop_new_chat_and_shared_retirement(self):
        expected, remote, cancellation = self.evidence()
        result = self.validate(expected, remote, cancellation)
        self.assertTrue(result['actual_stop_after_visible_content'])
        self.assertTrue(result['fresh_conversation_reuse'])
        self.assertTrue(result['local_retirement_attested'])
        self.assertEqual(result['supervision']['sample_count'], 3)
        self.assertEqual(result['cancellation_receipt_sha256'], common.digest(self.root / 'answers-turn-1-cancellation.json'))
        self.assertFalse(result['cancellation_answer_parity_qualified'])
        self.assertTrue(result['backend_qualification_requires_remote_closeout'])

    def test_same_host_success_keeps_supervision_without_tunnel(self):
        expected, remote, cancellation = self.evidence(mode='same')
        result = self.validate(expected, remote, cancellation)
        self.assertEqual(result['ui_client_host'], 'same')
        process = common.read_json(self.root / 'ui-process.json')
        self.assertEqual([row['role'] for row in process['processes']], ['electron', 'supervisor'])

    def test_cancelled_bytes_must_be_a_prefix_of_forwarded_native_bytes(self):
        expected, remote, cancellation = self.evidence()
        cancellation['forwarded_raw_base64'] = base64.b64encode(b': unrelated\n\n').decode()
        with self.assertRaisesRegex(ValueError, 'prefix'):
            self.validate(expected, remote, cancellation)

    def test_completion_before_stop_cannot_be_relabelled_as_cancellation(self):
        expected, remote, cancellation = self.evidence()
        report = common.read_json(self.root / 'ui-report.json')
        first = report['providerPipeline']['turns'][0]
        raw = (first['rawResponse'].encode() + frame(1, {}, 'stop',
               {'prompt_tokens': 476, 'completion_tokens': 6, 'total_tokens': 482}) + b'data: [DONE]\n\n')
        first.update(rawResponse=raw.decode(), transportBytes=len(raw))
        cancellation['forwarded_raw_base64'] = base64.b64encode(raw).decode()
        put(self.root / 'ui-report.json', report)
        self.rebind()
        with self.assertRaises(ValueError):
            self.validate(expected, remote, cancellation)

    def test_callback_prefix_cannot_end_inside_a_native_content_delta(self):
        expected, remote, cancellation = self.evidence()
        report = common.read_json(self.root / 'ui-report.json')
        first = report['providerPipeline']['turns'][0]
        raw = first['rawResponse'].replace('"content": "1"', '"content": "1 2"').encode()
        first.update(rawResponse=raw.decode(), transportBytes=len(raw))
        cancellation['forwarded_raw_base64'] = base64.b64encode(raw + frame(1, {'content': ' 3'})).decode()
        put(self.root / 'ui-report.json', report)
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'callback|numeric content'):
            self.validate(expected, remote, cancellation)

    def test_fake_abort_or_missing_acknowledgement_is_rejected(self):
        expected, remote, cancellation = self.evidence()
        original = common.read_json(self.root / 'ui-report.json')
        for mutation in ({'abortObserved': False}, {'cancelRequests': 0}, {'responseEof': True},
                         {'completed': True}, {'abortListenerRemoved': False}, {'checkpointed': 0},
                         {'result': {}}, {'usage': {'prompt_tokens': 476}}):
            report = deepcopy(original)
            report['providerPipeline']['turns'][0].update(mutation)
            put(self.root / 'ui-report.json', report)
            self.rebind()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.validate(expected, remote, cancellation)
        report = deepcopy(original)
        report['interruption']['cancellations'][0]['accepted'] = False
        put(self.root / 'ui-report.json', report)
        self.rebind()
        with self.assertRaises(ValueError):
            self.validate(expected, remote, cancellation)

    def test_stop_and_new_chat_clocks_must_follow_transport_events(self):
        expected, remote, cancellation = self.evidence()
        original = common.read_json(self.root / 'ui-report.json')
        for key, value in (('firstContentAt', 1004), ('cancelRequestedAt', 1002),
                           ('abortObservedAt', 1003), ('transportClosedAt', 1004), ('settledAt', 1005)):
            report = deepcopy(original)
            report['providerPipeline']['turns'][0][key] = value
            put(self.root / 'ui-report.json', report)
            self.rebind()
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'clocks'):
                self.validate(expected, remote, cancellation)
        report = deepcopy(original)
        report['stopReuse']['newChatClickedAt'] = 1006
        put(self.root / 'ui-report.json', report)
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'clocks'):
            self.validate(expected, remote, cancellation)

    def test_cancelled_history_cannot_enter_fresh_request_even_with_matching_local_records(self):
        expected, remote, cancellation = self.evidence()
        def change(report):
            report['requests'][1]['messages'].insert(0, deepcopy(report['persisted']['threads'][0]['messages'][1]))
            report['providerPipeline']['turns'][1]['request'] = deepcopy(report['requests'][1])
        self.mutate_report(change)
        with self.assertRaisesRegex(ValueError, 'history'):
            self.validate(expected, remote, cancellation)

    def test_reusing_the_first_thread_cannot_count_as_new_chat(self):
        expected, remote, cancellation = self.evidence()
        def change(report):
            report['requests'][1]['threadId'] = report['requests'][0]['threadId']
            report['providerPipeline']['turns'][1]['request'] = deepcopy(report['requests'][1])
        self.mutate_report(change)
        with self.assertRaisesRegex(ValueError, 'identity'):
            self.validate(expected, remote, cancellation)

    def test_new_chat_checkpoint_and_stopped_checkpoint_are_required(self):
        expected, remote, cancellation = self.evidence()
        original = common.read_json(self.root / 'ui-report.json')
        changes = [('freshCheckpoint', {'id': 'conversation-1', 'messages': [{'content': 'stale'}]}),
                   ('freshCheckpoint', {'id': 'wrong', 'messages': []}),
                   ('stoppedCheckpoint', {'id': 'answer-0', 'content': 'lost partial'}),
                   ('firstRequestId', 'another-run'), ('secondThreadId', 'conversation-0')]
        for key, value in changes:
            report = deepcopy(original)
            report['stopReuse'][key] = value
            put(self.root / 'ui-report.json', report)
            self.rebind()
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(expected, remote, cancellation)

    def test_client_report_cannot_promote_its_backend_qualification_scope(self):
        expected, remote, cancellation = self.evidence()
        self.mutate_report(lambda report: report['live'].update(backendQualified=True))
        with self.assertRaisesRegex(ValueError, 'live report contract'):
            self.validate(expected, remote, cancellation)

    def test_profile_commit_and_command_cannot_be_relabelled(self):
        expected, remote, cancellation = self.evidence()
        for mutation in ({'commit': ui.UI_COMMIT}, {'profile': ui.UI_PROFILE}, {'commit': 'b' * 40}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                ui.validate_identity({**expected, **mutation})
        with self.assertRaisesRegex(ValueError, 'Stop profile'):
            stop.validate_ui_evidence(self.root, identity(), remote)
        process = common.read_json(self.root / 'ui-process.json')
        next(row for row in process['processes'] if row['role'] == 'electron')['command'].remove('--live-stop-reuse')
        put(self.root / 'ui-process.json', process)
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'launch command'):
            self.validate(expected, remote, cancellation)

    def test_shared_supervision_and_retirement_remain_mandatory(self):
        expected, remote, cancellation = self.evidence()
        retired_path = self.root / 'ui-retirement.json'
        retired = common.read_json(retired_path)
        put(retired_path, {**retired, 'cgroup_absent': False})
        with self.assertRaisesRegex(ValueError, 'retirement'):
            self.validate(expected, remote, cancellation)
        put(retired_path, retired)
        summary = common.read_json(self.root / 'ui-supervision.json')
        summary['events_zero'] = False
        put(self.root / 'ui-supervision.json', summary)
        with self.assertRaisesRegex(ValueError, 'supervision'):
            self.validate(expected, remote, cancellation)


if __name__ == '__main__':
    unittest.main()
