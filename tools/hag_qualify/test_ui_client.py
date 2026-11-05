"""Model-free UI identity/receipt checks; never launch a compiler, UI, SSH, or model."""
import base64
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import ui_client as ui
    from . import gate_common as common
except ImportError:
    import ui_client as ui
    import gate_common as common


HOST = {'machine_id_sha256': 'a' * 64, 'boot_id': '11111111-1111-4111-8111-111111111111'}
RUN = '22222222-2222-4222-8222-222222222222'
REMOTE_BOOT = '33333333-3333-4333-8333-333333333333'


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def identity():
    return {'schema': 1, 'kind': 'antimatter-ui', 'profile': ui.UI_PROFILE, 'commit': ui.UI_COMMIT,
        'snapshot_id': '44444444-4444-4444-8444-444444444444', 'local_host': copy.deepcopy(HOST),
        'base_url': 'http://127.0.0.1:33000/v1', 'prepared_manifest_sha256': 'b' * 64,
        'artifacts': {name: ui.HTML_SHA if name == 'index.html' else 'c' * 64 for name in ui.ARTIFACTS},
        'source_verification_sha256': 'd' * 64,
        'electron': {'manifest_sha256': 'e' * 64, 'executable_sha256': 'f' * 64, 'files': 2, 'version': '41.1.1'},
        'process_seconds': 355, 'request_seconds': 240, 'maximum_requests': 2,
        'source_bundle_relation': 'esbuild_recorded_inputs_not_independently_rebuilt'}


class UiBase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='HAG UI identity ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def seal(self, value=None, mode='separate'):
        value = identity() if value is None else value
        put(self.root / 'client/identity.json', value)
        put(self.root / 'expected-manifest.json', {'client': {'kind': 'antimatter-ui',
            'identity_sha256': common.digest(self.root / 'client/identity.json')}})
        put(self.root / 'qualification-contract.json', {'profile': ui.UI_PROFILE, 'stream_client': 'antimatter-ui',
                                                       'ui_client_host': mode})
        return value


class UiIdentityTests(UiBase):
    def test_frozen_host_mode_requires_matching_boot_and_direct_port(self):
        expected = self.seal(mode='same')
        ready = {'host': '127.0.0.1', 'port': 33000, 'boot_id': HOST['boot_id']}
        self.assertEqual(ui.client_host(self.root, expected, ready), 'same')
        for changed in ({'port': 33001}, {'boot_id': REMOTE_BOOT}, {'host': '0.0.0.0'}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                ui.client_host(self.root, expected, {**ready, **changed})
        self.seal(mode='separate')
        with self.assertRaisesRegex(ValueError, 'boot'):
            ui.client_host(self.root, expected, ready)
        self.assertEqual(ui.client_host(self.root, expected, {**ready, 'boot_id': REMOTE_BOOT}), 'separate')
        for mode in (None, 'automatic', True):
            self.seal(mode=mode)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                ui.client_identity(self.root)

    def test_only_named_committed_vendor_escape_is_normalized(self):
        name = '../../../../../../../../../workspace/exec-antimatter/vendor/displacer/tokens/index.js'
        self.assertEqual(ui._source_path(name), ('vendor/displacer/tokens/index.js', False))
        for invalid in ('../../elsewhere/vendor/other/index.js', '../../vendor/displacer/../../src/index.js',
                        '../../vendor/displacer', '../../src/preload/index.ts'):
            with self.subTest(name=invalid), self.assertRaises(ValueError):
                ui._source_path(invalid)

    def test_identity_is_profile_specific_and_hash_bound(self):
        expected = self.seal()
        self.assertEqual(ui.client_identity(self.root), expected)
        put(self.root / 'qualification-contract.json', {'profile': 'antimatter-generated-history-c2048', 'stream_client': 'antimatter'})
        with self.assertRaises(ValueError):
            ui.client_identity(self.root)

    def test_identity_mutation_and_unreviewed_fields_fail(self):
        self.seal()
        value = identity()
        value['process_seconds'] = 720
        put(self.root / 'client/identity.json', value)
        with self.assertRaises(ValueError):
            ui.client_identity(self.root)
        for key, changed in [('commit', '0' * 40), ('maximum_requests', True), ('extra', 1),
                             ('base_url', 'http://198.51.100.10:33000/v1')]:
            value = identity()
            value[key] = changed
            with self.subTest(key=key), self.assertRaises(ValueError):
                ui.validate_identity(value)

    def prepared(self):
        prepared, repo, dist = (self.root / name for name in ('prepared', 'repo', 'electron-dist'))
        for path in (prepared, repo, dist):
            path.mkdir()
        sources = {name: ('// ' + name).encode() for name in set(ui.ENTRIES.values()) | ui.SOURCE_EXTRAS}
        sources['package-lock.json'] = json.dumps({'packages': {'node_modules/electron': {'version': '41.1.1'}}}).encode()
        dependencies = repo / 'node_modules/tiny/index.js'
        dependencies.parent.mkdir(parents=True)
        dependencies.write_bytes(b'export const tiny = 1;')
        source_hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sources.items()}
        source_hashes['node_modules/tiny/index.js'] = common.digest(dependencies)
        for name in ui.ARTIFACTS:
            (prepared / name).write_bytes(('bundle ' + name).encode())
        (prepared / 'launcher.mjs').write_bytes(sources['scripts/smoke-hag-conversation.mjs'])
        (prepared / 'replies.json').write_bytes(sources['test/fixtures/hag-generated-history.json'])
        for artifact, entry in ui.ENTRIES.items():
            inputs = {entry: {'bytes': len(sources[entry]), 'imports': [{'path': 'node_modules/tiny/index.js'}]},
                      'node_modules/tiny/index.js': {'bytes': dependencies.stat().st_size, 'imports': []}}
            outputs = {artifact: {'entryPoint': entry, 'imports': [], 'inputs': {name: {} for name in inputs},
                                   'bytes': (prepared / artifact).stat().st_size}}
            if artifact == 'fixture.js':
                outputs['fixture.css'] = {'imports': [], 'inputs': {}, 'bytes': (prepared / 'fixture.css').stat().st_size}
            put(prepared / (artifact + '.meta.json'), {'inputs': inputs, 'outputs': outputs})
        put(prepared / 'source-hashes.json', source_hashes)
        for name in ui.RUNTIME_DIRS:
            (prepared / name).mkdir()
        (dist / 'electron').write_bytes(b'\x7fELF\x02\x01 fixture runtime')
        (dist / 'electron').chmod(0o700)
        (dist / 'version').write_text('41.1.1\n')
        manifest = {'profile': ui.UI_PROFILE, 'backendQualified': False, 'baseUrl': identity()['base_url'],
            'processSeconds': 355, 'requestSeconds': 240, 'maximumRequests': 2,
            'artifacts': {name: common.digest(prepared / name) for name in ui.ARTIFACTS},
            'command': [str(dist / 'electron'), '--no-sandbox', '--ozone-platform=headless',
                        str(prepared / 'launcher.mjs'), '--fixture', str(prepared), '--live-base-url=' + identity()['base_url']],
            'environment': {'ELECTRON_RUN_AS_NODE': '', **{key: str(prepared / value) for key, value in {
                'XDG_CONFIG_HOME': 'config', 'XDG_CACHE_HOME': 'cache', 'XDG_STATE_HOME': 'state',
                'XDG_DATA_HOME': 'data', 'XDG_RUNTIME_DIR': 'runtime', 'TMPDIR': 'tmp'}.items()}}}
        put(prepared / 'prepared-live.json', manifest)
        return prepared, repo, dist, sources

    def freeze(self, inputs):
        prepared, repo, dist, sources = inputs
        with patch.object(ui, '_committed', side_effect=lambda _repo, name: sources[name]), \
                patch.object(ui, 'observe_host', return_value=HOST), \
                patch.object(ui, 'HTML_SHA', common.digest(prepared / 'index.html')):
            return ui.freeze_ui_identity(prepared, repo, dist, self.root / 'metadata')

    def test_freeze_checks_source_dependency_closure_and_distribution_without_execution(self):
        inputs = self.prepared()
        result = self.freeze(inputs)
        self.assertEqual(result['commit'], ui.UI_COMMIT)
        self.assertEqual(result['electron']['files'], 2)
        self.assertEqual(common.read_json(self.root / 'metadata/identity.json'), result)
        self.assertEqual(result['source_verification_sha256'], ui.canonical_sha(
            common.read_json(self.root / 'metadata/source-verification.json')))
        self.assertFalse((inputs[0] / 'user-data').exists())
        with self.assertRaises(ValueError):
            self.freeze(inputs)

    def test_freeze_rejects_changed_dependency_bytes(self):
        inputs = self.prepared()
        (inputs[1] / 'node_modules/tiny/index.js').write_bytes(b'changed after build')
        with self.assertRaisesRegex(ValueError, 'input differs'):
            self.freeze(inputs)

    def test_freeze_rejects_omitted_source_and_unfrozen_artifact(self):
        inputs = self.prepared()
        path = inputs[0] / 'source-hashes.json'
        sources = common.read_json(path)
        sources.pop('node_modules/tiny/index.js')
        put(path, sources)
        with self.assertRaisesRegex(ValueError, 'complete bundle closure'):
            self.freeze(inputs)

    def test_freeze_rejects_used_profile_and_symlink_artifacts(self):
        inputs = self.prepared()
        (inputs[0] / 'report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'used or contains extra'):
            self.freeze(inputs)
        (inputs[0] / 'report.json').unlink()
        artifact = inputs[0] / 'preload.cjs'
        copy_path = inputs[0] / 'preload-original'
        artifact.rename(copy_path)
        artifact.symlink_to(copy_path)
        with self.assertRaises(ValueError):
            self.freeze(inputs)

    def test_process_identity_is_observed_from_local_proc_and_nnp_is_required(self):
        directory = self.root / 'proc/901'
        directory.mkdir(parents=True)
        (directory / 'stat').write_text('901 (electron helper) ' + ' '.join(['S'] + ['0'] * 18 + ['123']) + '\n')
        (directory / 'cgroup').write_text('0::/ui-observed\n')
        (directory / 'status').write_text('Name:\telectron\nNoNewPrivs:\t1\n')
        (directory / 'cmdline').write_bytes(b'/frozen/electron\0--no-sandbox\0')
        (directory / 'environ').write_bytes(b'PATH=/usr/bin\0ELECTRON_RUN_AS_NODE=\0')
        observed = ui.observe_process(901, 'electron', '/sys/fs/cgroup/ui-observed', proc_root=self.root / 'proc')
        self.assertEqual(observed['start_time_ticks'], 123)
        self.assertEqual(observed['command'], ['/frozen/electron', '--no-sandbox'])
        (directory / 'status').write_text('NoNewPrivs:\t0\n')
        with self.assertRaisesRegex(ValueError, 'NoNewPrivs'):
            ui.observe_process(901, 'electron', '/sys/fs/cgroup/ui-observed', proc_root=self.root / 'proc')

    def test_retirement_observer_checks_local_host_and_recorded_pid_absence(self):
        process = {'host': HOST, 'run_id': RUN, 'identity_sha256': 'a' * 64, 'unit': 'hag-ui-observed.service',
                   'cgroup': '/sys/fs/cgroup/hag-ui-does-not-exist-' + RUN, 'processes': [{'pid': 901}]}
        path, output = self.root / 'process.json', self.root / 'retirement.json'
        put(path, process)
        proc = self.root / 'proc'
        (proc / '901').mkdir(parents=True)
        state = {'ActiveState': 'inactive', 'SubState': 'dead', 'MainPID': '0', 'ControlGroup': ''}
        with patch.object(ui, 'observe_host', return_value=HOST):
            with self.assertRaisesRegex(ValueError, 'remains present'):
                ui.observe_retirement(path, output, proc_root=proc, unit_reader=lambda _unit: state)
            (proc / '901').rmdir()
            result = ui.observe_retirement(path, output, proc_root=proc, unit_reader=lambda _unit: state)
        self.assertEqual(result['recorded_pids_absent'], [901])
        self.assertEqual(result['process_receipt_sha256'], common.digest(path))
        with patch.object(ui, 'observe_host', return_value={**HOST, 'boot_id': REMOTE_BOOT}):
            with self.assertRaisesRegex(ValueError, 'different host'):
                ui.observe_retirement(path, self.root / 'wrong-host.json', proc_root=proc, unit_reader=lambda _unit: state)


class UiEvidenceTests(UiBase):
    def evidence(self, mode='separate'):
        expected = self.seal(mode=mode)
        digest = common.digest(self.root / 'client/identity.json')
        ready = {'run_id': RUN, 'client_identity_sha256': digest, 'host': '127.0.0.1',
                 'boot_id': HOST['boot_id'] if mode == 'same' else REMOTE_BOOT,
                 'delivery': 'buffered-sse-after-parity-reset', 'port': 33000 if mode == 'same' else 33001}
        put(self.root / 'answers-ui-ready.json', ready)
        put(self.root / 'answers-ui-listener.json', {**ready, 'request_count': 2, 'forwarded_count': 2,
            'response_count': 2, 'rejected_count': 0, 'closed': True})
        turns, requests, remote, history = [], [], [], []
        for turn in range(2):
            history.append({'id': 'user-' + str(turn), 'role': 'user', 'content': 'ready' if not turn else 'again'})
            request = {'providerId': 'hag-ui-fixture', 'model': 'hag-context-qwen3-0.6b', 'threadId': 'conversation',
                       'requestId': 'answer-' + str(turn), 'responseMessageId': 'answer-' + str(turn),
                       'useTools': True, 'messages': copy.deepcopy(history)}
            wire = {'model': request['model'], 'messages': [{'role': 'system', 'content': 'frozen grounding'}]
                    + [{k: v[k] for k in ('role', 'content')} for v in history], 'stream': True, 'max_tokens': 128}
            usage = {'prompt_tokens': 491 if turn else 467, 'completion_tokens': 6, 'total_tokens': 497 if turn else 473}
            response = {'object': 'chat.completion', 'model': request['model'], 'usage': usage, 'choices': [
                {'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': 'ready.'}}]}
            def frame(delta, finish=None, usage=None):
                value = {'id': 'chatcmpl-beholdd-' + str(turn), 'object': 'chat.completion.chunk', 'created': 1,
                    'model': request['model'], 'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]}
                if usage is not None:
                    value['usage'] = usage
                return 'data: ' + json.dumps(value) + '\n\n'
            raw = (frame({'role': 'assistant'}) + frame({'content': 'ready.'})
                   + frame({}, 'stop', usage) + 'data: [DONE]\n\n').encode()
            remote.append({'turn': turn + 1, 'method': 'POST', 'route': '/v1/chat/completions', 'request': wire,
                'status': 200, 'content_type': 'text/event-stream', 'read_error': None, 'body_truncated': False,
                'raw_body_base64': base64.b64encode(raw).decode()})
            turns.append({'turn': turn, 'request': request, 'wire': wire, 'rawResponse': raw.decode(), 'usage': usage,
                'requestId': request['requestId'], 'responseStatus': 200, 'transport': 'sse', 'contentType': 'text/event-stream',
                'servedModel': request['model'], 'finishReason': 'stop', 'attemptCount': 1, 'providerCalls': 1,
                'maxTokens': 128, 'completed': True, 'settled': True, 'outcome': 'finished', 'timedOut': False,
                'error': None, 'cancelRequests': 0, 'transportClosed': True, 'resumed': bool(turn), 'deadlineMs': 240000,
                'transportFragments': 1, 'transportBytes': len(raw), 'startedAt': 1000 + turn * 20,
                'settledAt': 1010 + turn * 20, 'contentChunks': ['ready.'],
                'providerResult': {'content': 'ready.', 'finishReason': 'stop'},
                'result': {'reason': 'finished', 'rounds': 1, 'toolCalls': 0, 'content': 'ready.'}})
            requests.append(request)
            history.append({'id': request['requestId'], 'role': 'assistant', 'content': 'ready.'})
        saved = copy.deepcopy(history)
        for message in saved[1::2]:
            message['meta'] = {'failed': False, 'createdAt': 1000, 'finishedAt': 1010, 'chunks': 1}
        report = {'passed': True, 'actualApp': True, 'actualPreload': True, 'actualConversationStore': True,
            'rendererNetworkBlocked': True, 'hardwareAcceleration': False, 'networkBlocked': False,
            'modelExecution': 'external-endpoint-unverified', 'reload': None, 'restart': None, 'interruption': None,
            'live': {'profile': ui.UI_PROFILE, 'baseUrl': expected['base_url'], 'backendQualified': False,
                     'processSeconds': 355, 'requestSeconds': 240, 'maximumRequests': 2},
            'process': {'pid': 901, 'instanceId': '55555555-5555-4555-8555-555555555555', 'phase': None},
            'checkpointCount': 4, 'hydrationCount': 2, 'providerPipeline': {'transport': 'loopback-http', 'turns': turns},
            'requests': requests, 'persisted': {'threads': [{'id': 'conversation', 'messages': saved}]}}
        put(self.root / 'ui-report.json', report)
        group = '/sys/fs/cgroup/user.slice/ui-test'
        processes = [{'pid': 901, 'role': 'electron', 'start_time_ticks': 123, 'cgroup': group,
            'command': ['/frozen/electron', '--no-sandbox', '--ozone-platform=headless', '/prepared/launcher.mjs',
                        '--fixture', '/prepared', '--live-base-url=' + expected['base_url']]},
            {'pid': 902, 'role': 'tunnel', 'start_time_ticks': 122, 'cgroup': group,
            'command': ['/usr/bin/ssh', '-N', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', '-o', 'ControlMaster=no',
                '-o', 'ControlPath=none', '-L', '127.0.0.1:33000:127.0.0.1:33001', 'tester@validation-host']}]
        if mode == 'same':
            processes.pop()
        processes.append({'pid': 900, 'role': 'supervisor', 'start_time_ticks': 121, 'cgroup': group,
                          'command': ['/usr/bin/python3', '-B', '/frozen/ui-run-local.py', '--worker']})
        process = {'schema': 1, 'kind': 'antimatter-ui-process', 'host': HOST, 'run_id': RUN, 'identity_sha256': digest,
            'ready_sha256': common.digest(self.root / 'answers-ui-ready.json'),
            'report_sha256': common.digest(self.root / 'ui-report.json'), 'unit': 'hag-ui-test.service',
            'cgroup': group, 'processes': processes, 'started_monotonic_ns': 12000000000,
            'ended_monotonic_ns': 12200000000, 'exit_code': 0, 'timed_out': False,
            'bundle_verified_before': True, 'bundle_verified_after': True,
            'electron_manifest_sha256': expected['electron']['manifest_sha256']}
        put(self.root / 'ui-process.json', process)
        retirement = {'schema': 1, 'kind': 'antimatter-ui-retirement', 'host': HOST, 'run_id': RUN,
            'identity_sha256': digest, 'process_receipt_sha256': common.digest(self.root / 'ui-process.json'),
            'observed_monotonic_ns': 13000000000, 'unit': process['unit'], 'cgroup': group, 'cgroup_absent': True,
            'recorded_pids_absent': sorted(row['pid'] for row in processes), 'state': {'ActiveState': 'inactive', 'SubState': 'dead',
                                                      'MainPID': '0', 'ControlGroup': ''}}
        put(self.root / 'ui-retirement.json', retirement)
        samples = [{'monotonic_ns': 12000000000 + i * 100000000, 'available_bytes': 9 * 1024 ** 3, 'tctl_c': 60,
            'cgroup': {'memory.current': 123456, 'memory.peak': 123456, 'memory.swap.current': 0,
                       'memory.events': dict.fromkeys(['low', 'high', 'max', 'oom', 'oom_kill'], 0)}} for i in range(3)]
        (self.root / 'ui-supervision-samples.jsonl').write_text(''.join(json.dumps(value) + '\n' for value in samples))
        summary = {'schema': 1, 'kind': 'antimatter-ui-supervision', 'host': HOST, 'run_id': RUN,
            'unit': process['unit'], 'limits': ui.UI_LIMITS, 'exit_code': 0, 'abort_reason': None,
            'minimum_available_bytes': 9 * 1024 ** 3, 'initial_available_bytes': 9 * 1024 ** 3,
            'memory_peak_bytes': 123456, 'events_zero': True, 'swap_zero': True, 'peak_tctl_c': 60,
            'thermal_cutoff_c': 90, 'process_receipt_sha256': common.digest(self.root / 'ui-process.json'),
            'samples_sha256': common.digest(self.root / 'ui-supervision-samples.jsonl'),
            'cooldown': [{'monotonic_ns': 1000000000 + i * 500000000, 'tctl_c': 60} for i in range(21)]}
        put(self.root / 'ui-supervision.json', summary)
        return expected, remote

    def rebind(self):
        process = common.read_json(self.root / 'ui-process.json')
        process['report_sha256'] = common.digest(self.root / 'ui-report.json')
        put(self.root / 'ui-process.json', process)
        for name in ('ui-retirement.json', 'ui-supervision.json'):
            value = common.read_json(self.root / name)
            value['process_receipt_sha256'] = common.digest(self.root / 'ui-process.json')
            if name == 'ui-supervision.json':
                value['samples_sha256'] = common.digest(self.root / 'ui-supervision-samples.jsonl')
            put(self.root / name, value)

    def test_two_turn_report_joins_remote_bytes_and_local_retirement(self):
        expected, remote = self.evidence()
        result = ui.validate_ui_evidence(self.root, expected, remote)
        self.assertEqual(result['requests'], 2)
        self.assertTrue(result['local_retirement_attested'])
        self.assertEqual(result['supervision']['sample_count'], 3)
        self.assertEqual(result['ui_client_host'], 'separate')

    def test_same_host_two_turn_report_needs_no_tunnel_but_keeps_supervisor_and_retirement(self):
        expected, remote = self.evidence(mode='same')
        result = ui.validate_ui_evidence(self.root, expected, remote)
        self.assertEqual(result['ui_client_host'], 'same')
        self.assertEqual(result['remote_boot_id'], HOST['boot_id'])
        process = common.read_json(self.root / 'ui-process.json')
        self.assertEqual([row['role'] for row in process['processes']], ['electron', 'supervisor'])
        process['processes'][1]['role'] = 'tunnel'
        put(self.root / 'ui-process.json', process)
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'roles/PIDs'):
            ui.validate_ui_evidence(self.root, expected, remote)

    def test_topology_cannot_be_relabelled_after_a_run(self):
        expected, remote = self.evidence()
        self.seal(mode='same')
        with self.assertRaisesRegex(ValueError, 'boot'):
            ui.validate_ui_evidence(self.root, expected, remote)

    def test_supervision_allows_bounded_cleanup_but_rejects_missing_end_coverage(self):
        expected, remote = self.evidence(mode='same')
        process_path = self.root / 'ui-process.json'
        original_process = common.read_json(process_path)
        sample_path = self.root / 'ui-supervision-samples.jsonl'
        original_samples = [json.loads(line) for line in sample_path.read_text().splitlines()]
        for failure in ('none', 'missing-end', 'long-cleanup'):
            process = copy.deepcopy(original_process)
            samples = copy.deepcopy(original_samples)
            if failure == 'missing-end':
                process['ended_monotonic_ns'] = samples[-1]['monotonic_ns'] + 1000000001
            else:
                count = 20 if failure == 'none' else 51
                for _ in range(count):
                    samples.append({**copy.deepcopy(samples[-1]), 'monotonic_ns': samples[-1]['monotonic_ns'] + 100000000})
            put(process_path, process)
            sample_path.write_text(''.join(json.dumps(value) + '\n' for value in samples))
            self.rebind()
            with self.subTest(case=failure):
                if failure == 'none':
                    self.assertEqual(ui.validate_ui_supervision(self.root, process)['sample_count'], 23)
                else:
                    with self.assertRaisesRegex(ValueError, 'cover the client lifetime'):
                        ui.validate_ui_supervision(self.root, process)

    def test_changed_raw_response_and_visible_history_fail_after_valid_hash_rebinding(self):
        expected, remote = self.evidence()
        original = common.read_json(self.root / 'ui-report.json')
        for field in ('raw', 'history', 'attempts', 'pid', 'fixture', 'qualification'):
            value = copy.deepcopy(original)
            if field == 'raw': value['providerPipeline']['turns'][0]['rawResponse'] += ' '
            if field == 'history': value['persisted']['threads'][0]['messages'][1]['id'] = 'wrong'
            if field == 'attempts': value['providerPipeline']['turns'][0]['attemptCount'] = 2
            if field == 'pid': value['process']['pid'] = 903
            if field == 'fixture': value['providerPipeline']['transport'] = 'in-process-sse-fixture'
            if field == 'qualification': value['live']['backendQualified'] = True
            put(self.root / 'ui-report.json', value)
            self.rebind()
            with self.subTest(field=field), self.assertRaises(ValueError):
                ui.validate_ui_evidence(self.root, expected, remote)

    def test_stale_run_cross_host_retirement_and_extra_requests_fail(self):
        expected, remote = self.evidence()
        original = common.read_json(self.root / 'ui-retirement.json')
        for field in ('run_id', 'host', 'recorded_pids_absent'):
            value = copy.deepcopy(original)
            value[field] = {'run_id': REMOTE_BOOT, 'host': {**HOST, 'boot_id': REMOTE_BOOT},
                            'recorded_pids_absent': [901]}[field]
            put(self.root / 'ui-retirement.json', value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ui.validate_ui_evidence(self.root, expected, remote)
        put(self.root / 'ui-retirement.json', original)
        listener = common.read_json(self.root / 'answers-ui-listener.json')
        listener['request_count'] = 3
        put(self.root / 'answers-ui-listener.json', listener)
        with self.assertRaises(ValueError):
            ui.validate_ui_evidence(self.root, expected, remote)

    def test_oom_sample_and_stale_cooldown_fail_even_with_summary_resealed(self):
        expected, remote = self.evidence()
        path = self.root / 'ui-supervision-samples.jsonl'
        samples = [json.loads(line) for line in path.read_text().splitlines()]
        samples[1]['cgroup']['memory.events']['oom_kill'] = 1
        path.write_text(''.join(json.dumps(value) + '\n' for value in samples))
        self.rebind()
        with self.assertRaisesRegex(ValueError, 'memory events'):
            ui.validate_ui_evidence(self.root, expected, remote)

    def test_wrong_tunnel_mapping_and_too_long_process_fail(self):
        expected, remote = self.evidence()
        path = self.root / 'ui-process.json'
        original = common.read_json(path)
        for field in ('tunnel', 'deadline'):
            value = copy.deepcopy(original)
            if field == 'tunnel': value['processes'][1]['command'][-2] = '0.0.0.0:33000:127.0.0.1:33001'
            else: value['ended_monotonic_ns'] = value['started_monotonic_ns'] + 356 * 10 ** 9
            put(path, value)
            self.rebind()
            with self.subTest(field=field), self.assertRaises(ValueError):
                ui.validate_ui_evidence(self.root, expected, remote)


if __name__ == '__main__':
    unittest.main()
