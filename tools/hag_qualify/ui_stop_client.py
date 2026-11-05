"""Join a real editor Stop/New chat sequence with native disconnect evidence."""
import base64
from pathlib import Path
import re

try:
    from . import ui_client as ui, ui_stop as backend
except ImportError:
    import ui_client as ui
    import ui_stop as backend

require, integer = ui.require, ui.integer


def visible_text(content):
    return re.sub(r'<think>.*?(?:</think>|$)', '', content, flags=re.S).strip()


def validate_ui_evidence(root, expected_identity, broker_records, local_bundle_root=None):
    require(expected_identity.get('profile') == ui.UI_STOP_PROFILE, 'Stop validator requires Stop profile')
    observed = ui.validate_ui_process_evidence(root, expected_identity, local_bundle_root)
    report = observed['report']
    cancellation = backend.cancellation_receipt(Path(root))
    forwarded = base64.b64decode(cancellation['forwarded_raw_base64'], validate=True)
    pipeline = report.get('providerPipeline', {})
    require(pipeline.get('transport') == 'loopback-http', 'Stop used fixture transport')
    turns, requests = pipeline.get('turns'), report.get('requests')
    require(type(turns) is list and len(turns) == 2 and type(requests) is list and len(requests) == 2
            and type(broker_records) is list and len(broker_records) == 2, 'Stop request count differs')
    stop = report.get('stopReuse')
    require(type(stop) is dict and set(stop) == {'firstThreadId', 'firstRequestId', 'secondThreadId', 'stop',
            'stoppedCheckpoint', 'newChatClickedAt', 'freshCheckpoint'}, 'Stop/New chat evidence schema differs')
    thread_ids, answer_ids, user_ids, saved_expected = [], [], [], []
    for index, (turn, request, remote) in enumerate(zip(turns, requests, broker_records)):
        expected_wire = backend.request_body(backend.fixture_document(backend.PROFILE), index + 1)
        require(remote.get('turn') == index + 1 and remote.get('method') == 'POST'
                and remote.get('route') == '/v1/chat/completions' and remote.get('status') == 200
                and remote.get('request') == expected_wire and turn.get('wire') == expected_wire
                and turn.get('request') == request, 'Stop wire/turn binding differs')
        require(turn.get('turn') == index and turn.get('attemptCount') == 1 and turn.get('providerCalls') == 1
                and turn.get('maxTokens') == (64 if index == 0 else 16) and turn.get('settled') is True
                and turn.get('timedOut') is False and turn.get('transportClosed') is True
                and turn.get('resumed') is False and turn.get('deadlineMs') == 240000
                and turn.get('responseStatus') == 200 and turn.get('transport') == 'sse'
                and turn.get('contentType') == 'text/event-stream' and integer(turn.get('transportFragments'), 1),
                'Stop turn transport/accounting differs')
        started, ended = turn.get('startedAt'), turn.get('settledAt')
        require(integer(started, 1) and integer(ended, started) and ended - started <= 240000,
                'Stop turn exceeded request deadline')
        raw = turn.get('rawResponse')
        require(type(raw) is str and 0 < len(raw.encode()) <= ui.MAX_BODY
                and turn.get('transportBytes') == len(raw.encode()), 'Stop captured transport bytes differ')
        state = backend.SseState()
        state.feed(raw.encode())
        chunks = turn.get('contentChunks')
        require(type(chunks) is list and chunks and all(type(value) is str for value in chunks),
                'Stop content callbacks missing')
        content = ''.join(chunks)
        if index == 0:
            require(forwarded.startswith(raw.encode()), 'UI cancelled prefix differs from forwarded native bytes')
            received = state.partial()
            require(content and received.startswith(content) and chunks == state.content[:len(chunks)]
                    and re.search(r'[0-9]', visible_text(content)),
                    'Stop callbacks lack received visible numeric content')
            require(turn.get('completed') is False and turn.get('outcome') == 'cancelled'
                    and turn.get('cancelRequests') == 1 and turn.get('abortObserved') is True
                    and turn.get('abortListenerRemoved') is True and turn.get('responseEof') is False
                    and turn.get('providerResult') is None and turn.get('result') is None
                    and turn.get('usage') is None and turn.get('finishReason') is None
                    and type(turn.get('error')) is str and turn['error']
                    and integer(turn.get('checkpointed'), 1), 'Stop did not abort an unfinished request')
            click = stop.get('stop')
            require(type(click) is dict and set(click) == {'clickedAt', 'visibleTextBeforeStop'}
                    and type(click['visibleTextBeforeStop']) is str
                    and re.search(r'[0-9]', click['visibleTextBeforeStop'])
                    and ' '.join(click['visibleTextBeforeStop'].split()) in ' '.join(visible_text(content).split()),
                    'Stop lacks visible numeric UI evidence')
            clocks = [started, turn.get('firstContentAt'), click.get('clickedAt'), turn.get('cancelRequestedAt'),
                      turn.get('abortObservedAt'), turn.get('transportClosedAt'), ended,
                      stop.get('newChatClickedAt')]
            require(all(integer(value, 1) for value in clocks) and clocks == sorted(clocks),
                    'Stop/UI abort clocks are out of order')
            persisted_content = content + '\n\n_Stopped._'
        else:
            native_raw = backend.raw_response(remote)
            require(raw.encode() == native_raw, 'reuse UI bytes differ from native response')
            response = state.complete()
            usage = {'prompt_tokens': 467, 'completion_tokens': 6, 'total_tokens': 473}
            require(content == state.text == '<think>\n\n</think>\n\nready.' and response['usage'] == usage
                    and response['choices'][0]['finish_reason'] == 'stop' and turn.get('usage') == usage
                    and turn.get('servedModel') == backend.MODEL_ID and turn.get('finishReason') == 'stop',
                    'reuse completion differs from fixed ready oracle')
            require(turn.get('completed') is True and turn.get('outcome') == 'finished'
                    and turn.get('cancelRequests') == 0 and turn.get('abortObserved') is False
                    and turn.get('error') is None and turn.get('providerResult', {}).get('content') == content
                    and turn.get('providerResult', {}).get('finishReason') == 'stop'
                    and turn.get('result') == {'reason': 'finished', 'rounds': 1, 'toolCalls': 0, 'content': content}
                    and started >= stop['newChatClickedAt'], 'fresh reuse did not finish successfully')
            persisted_content = content
        run_id, thread_id = request.get('requestId'), request.get('threadId')
        require(type(run_id) is str and run_id and request.get('responseMessageId') == run_id
                and turn.get('requestId') == run_id and run_id not in answer_ids
                and type(thread_id) is str and thread_id and thread_id not in thread_ids
                and request.get('providerId') == 'hag-ui-fixture' and request.get('model') == backend.MODEL_ID
                and request.get('maxTokens') is None and request.get('temperature') is None
                and request.get('useTools') is True, 'fresh UI request identity differs')
        messages = request.get('messages')
        require(type(messages) is list and len(messages) == 1 and type(messages[0]) is dict
                and set(messages[0]) == {'id', 'role', 'content'} and type(messages[0]['id']) is str
                and messages[0]['id'] and messages[0]['id'] not in user_ids
                and [{key: message[key] for key in ('role', 'content')} for message in messages] == expected_wire['messages'][1:],
                'cancelled history entered fresh UI request')
        thread_ids.append(thread_id)
        answer_ids.append(run_id)
        user_ids.append(messages[0]['id'])
        saved_expected.append(messages + [{'id': run_id, 'role': 'assistant', 'content': persisted_content}])
    require(report.get('interruption') == {'kind': 'cancel', 'consoleErrors': [], 'providerErrors': [],
            'cancellations': [{'runId': answer_ids[0], 'accepted': True}]}, 'UI cancellation acknowledgement differs')
    require(stop['firstThreadId'] == thread_ids[0] and stop['firstRequestId'] == answer_ids[0]
            and stop['secondThreadId'] == thread_ids[1]
            and type(stop['freshCheckpoint']) is dict and stop['freshCheckpoint'].get('id') == thread_ids[1]
            and stop['freshCheckpoint'].get('messages') == [], 'New chat did not checkpoint an empty conversation')
    threads = report.get('persisted', {}).get('threads')
    require(type(threads) is list and len(threads) == 2 and {row.get('id') for row in threads} == set(thread_ids),
            'Stop persisted thread inventory differs')
    for index, thread_id in enumerate(thread_ids):
        messages = next(row for row in threads if row['id'] == thread_id).get('messages')
        require(type(messages) is list and len(messages) == 2
                and [{key: message.get(key) for key in ('id', 'role', 'content')} for message in messages] == saved_expected[index],
                'Stop durable messages differ from actual callbacks')
        metadata = messages[1].get('meta', {})
        require(metadata.get('failed') is False and integer(metadata.get('createdAt'), 1)
                and integer(metadata.get('finishedAt'), metadata['createdAt'])
                and metadata.get('chunks') == len(turns[index]['contentChunks']) + (1 if index == 0 else 0),
                'Stop completion checkpoint differs')
        if index == 0:
            require(stop['stoppedCheckpoint'] == messages[1], 'stopped checkpoint differs from persisted message')
    return {**observed['summary'], 'actual_stop_after_visible_content': True,
            'fresh_conversation_reuse': True, 'cancellation_receipt_sha256': ui.common.digest(Path(root) / 'answers-turn-1-cancellation.json'),
            'cancellation_answer_parity_qualified': False}
