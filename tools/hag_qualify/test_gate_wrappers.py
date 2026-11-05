"""Hardware-free answer supervisor regressions; no model, GPU, root or systemd execution."""
import copy
import base64
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_common as common
import inputs
import profiles
from answer_quality import quality_assessment


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEST_BINARY_SHA = '1' * 64
TEST_SOURCE_SHA = '2' * 64

runner = load('answers_runner', 'run-gate.py')
child = load('answers_child', 'gate-child.py')
closeout = load('answers_closeout', 'closeout.py')


def thermal_fixture():
    samples = [{'sample': 1, 'phase': 'preflight', 'elapsed_s': 0, 'tctl_c': 50.0, 'sensor_ok': True},
               {'sample': 2, 'phase': 'running', 'elapsed_s': .1, 'tctl_c': 60.0, 'sensor_ok': True}]
    summary = {'ok': True, 'aborted_for_heat': False, 'abort_reason': None, 'process_started': True,
               'exit_code': 0, 'samples': 2, 'max_tctl_c': 60.0, 'residual_group_cleaned': False}
    return samples, summary


def answer(fixture_id, text, prompt=30, completion=20):
    return {'fixture_id': fixture_id, 'text': text, 'final_answer': text,
            'thinking_present': False, 'finish_reason': 'stop',
            'usage': {'prompt_tokens': prompt, 'completion_tokens': completion,
                      'total_tokens': prompt + completion},
            'evaluated_positions': prompt + completion}


def log(mode, report):
    return 'HAG_ANSWERS_REPORT ' + json.dumps(report) + '\nPASS hag-answers stage=' + mode + ' qualified=true\n'


class Gates(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.set_profile('list-c128')
        self.now = datetime.now(timezone.utc)
        self.capture = [{'event': 'ARMED', 'timestamp': (self.now - timedelta(seconds=10)).isoformat(), 'pid': 123, 'captured': 0},
                        {'event': 'HEARTBEAT', 'timestamp': self.now.isoformat(), 'pid': 123, 'captured': 0}]
        self.root_status = 'State:\tS (sleeping)\nUid:\t0\t0\t0\t0\n'

    def set_profile(self, name):
        self.client = None
        self.profile = name
        self.contract = inputs.profile_contract(name, 'antimatter' if name in profiles.GROUNDED_PROFILES else 'python-http')
        fixtures = profiles.fixture_document(name)
        (self.root / 'qualification-contract.json').write_text(json.dumps(self.contract))
        (self.root / 'answer-fixtures.json').write_text(json.dumps(fixtures))
        self.fixtures = fixtures['fixtures']
        answers = {'list_1_to_8': answer('list_1_to_8', '1, 2, 3, 4, 5, 6, 7, 8'),
                   'arithmetic_7_plus_5': answer('arithmetic_7_plus_5', '12', 15, 5),
                   'recall_mango': answer('recall_mango', 'MANGO', 40, 3),
                   'grounded_no_tools_ready': answer('grounded_no_tools_ready', 'Ready.', 93, 5),
                   'project_instructions_ready': answer('project_instructions_ready', 'Ready.', 467, 5),
                   'project_history_ready': answer('project_history_ready', 'Ready.', 495, 5),
                   'project_history_template_ready': answer('project_history_template_ready', 'Ready.', 491, 5)}
        self.references = [answers[fixture['id']] for fixture in self.fixtures]
        if not self.contract['require_semantic_match']:
            for reference, fixture in zip(self.references, self.fixtures):
                reference['quality'] = quality_assessment(reference['final_answer'], fixture)
        manifest = {'profile': name, 'binaries': {'beholdd': TEST_BINARY_SHA}, 'source_manifest_sha256': TEST_SOURCE_SHA,
                    'model_sha256': common.MODEL_SHA256,
                    'gate_files': {'answer-fixtures.json': common.digest(self.root / 'answer-fixtures.json')}}
        self.manifest(manifest)
        identities = {'binary_sha256': TEST_BINARY_SHA, 'source_manifest_sha256': TEST_SOURCE_SHA,
                      'checkpoint_sha256': common.MODEL_SHA256,
                      'fixture_sha256': manifest['gate_files']['answer-fixtures.json'],
                      'gate_files': manifest['gate_files']}
        (self.root / 'cpu-reference.json').write_text(json.dumps({'schema': 1, 'profile': name, 'execution': 'cpu-reference',
                          'capacity': profiles.profile_limits(name)['capacity'], 'identities': identities, 'answers': self.references,
                          'cpu_exit_code': 0, 'final_unloaded': True}))
        (self.root / 'cpu-reference-result.json').write_text(json.dumps({'fixture': 'CPU prior'}))
        (self.root / 'answers-result.json').write_text(json.dumps({'fixture': 'native prior'}))

    def manifest(self, value):
        (self.root / 'expected-manifest.json').write_text(json.dumps(value))

    def set_lifecycle(self):
        """Supervisor-only fixtures; raw protocol replay belongs to test_lifecycle."""
        self.profile = profiles.LIFECYCLE_PROFILE
        self.contract = inputs.profile_contract(self.profile)
        (self.root / 'qualification-contract.json').write_text(json.dumps(self.contract))
        fixture = profiles.fixture_document(self.profile)
        (self.root / 'answer-fixtures.json').write_text(json.dumps(fixture))
        self.manifest({'profile': self.profile, 'binaries': {'beholdd': TEST_BINARY_SHA},
                       'source_manifest_sha256': TEST_SOURCE_SHA, 'model_sha256': common.MODEL_SHA256,
                       'gate_files': {'answer-fixtures.json': common.digest(self.root / 'answer-fixtures.json')}})

    def test_lifecycle_timing_is_rejected_before_any_supervisor_side_effect(self):
        self.set_lifecycle()
        before = {path.name: path.read_bytes() for path in self.root.iterdir()}
        with patch.multiple(runner, ROOT=self.root, approved_contract=lambda: self.contract), \
             patch.object(runner, 'available_memory') as memory, \
             patch.object(runner, 'project_cooldown') as cooldown, \
             patch.object(runner.subprocess, 'Popen') as launch:
            with self.assertRaisesRegex(ValueError, 'stage is not admitted'):
                runner.main(['timing'])
            memory.assert_not_called()
            cooldown.assert_not_called()
            launch.assert_not_called()
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir()})
        with self.assertRaises(ValueError): runner.unit_command('fixture', 'timing', self.profile)
        with self.assertRaises(ValueError): runner.probe_evidence('timing', '', self.root)

    def test_lifecycle_deadlines_preserve_containment_and_reserve_full_watcher_window(self):
        self.set_lifecycle()
        expected = {'self-check': (180,195,210), 'cpu-reference': (720,735,750), 'answers': (600,615,630)}
        self.assertEqual(profiles.profile_modes(self.profile), tuple(expected))
        for mode, deadlines in expected.items():
            self.assertEqual(profiles.profile_deadlines(self.profile)[mode], deadlines)
            command = runner.unit_command('fixture', mode, self.profile)
            for value in ('MemoryMax=2G', 'MemorySwapMax=0', 'TasksMax=64', 'CPUQuota=75%',
                          'NoNewPrivileges=yes', 'OOMPolicy=stop', 'Restart=no',
                          'RuntimeMaxSec=' + str(deadlines[1]) + 's'):
                self.assertIn('--property=' + value, command)
        records = copy.deepcopy(self.capture)
        records[0]['duration_seconds'] = 1500
        runner.validate_capture_lease(records, self.now, 'answers', self.profile)
        # Enough for the project's shorter answers stage, but not lifecycle's630+20 reserve.
        later = self.now + timedelta(seconds=890)
        runner.validate_capture_lease(records, later, 'answers', profiles.PROJECT_PROFILE)
        with self.assertRaisesRegex(ValueError, 'insufficient remaining'):
            runner.validate_capture_lease(records, later, 'answers', self.profile)

    def test_lifecycle_child_deadlines_and_private_attestation_are_bound(self):
        self.set_lifecycle()
        memory = {'memory_current_bytes': 1, 'memory_peak_bytes': 2, 'swap_current_bytes': 0,
                  'swap_peak_bytes': 0, 'memory_events': {'high':0, 'max':0, 'oom':0, 'oom_kill':0}}
        for mode in profiles.profile_modes(self.profile):
            group = Path('/sys/fs/cgroup/user.slice/hag-answers-' + mode + '-20261008-010203-000001.service')
            attestation = '/' + str(group.relative_to('/sys/fs/cgroup')) if mode == 'answers' else None
            record = {'cgroup': str(group), 'limits': common.EXPECTED_LIMITS, 'memory_before': memory,
                      'stage_deadlines': dict(zip(('child','service','outer'), profiles.profile_deadlines(self.profile)[mode])),
                      'private_cgroup_attestation': attestation}
            runner.validate_project_child(record, mode, self.profile)
            wrong = copy.deepcopy(record)
            wrong['stage_deadlines']['child'] -= 1
            with self.assertRaises(ValueError): runner.validate_project_child(wrong, mode, self.profile)
            wrong = {**record, 'private_cgroup_attestation': '/forged'}
            with self.assertRaises(ValueError): runner.validate_project_child(wrong, mode, self.profile)

    def test_lifecycle_probe_delegates_raw_replay_but_never_executes_self_check(self):
        import lifecycle
        self.set_lifecycle()
        for mode in ('cpu-reference', 'answers'):
            report = {'stage': mode, 'profile': self.profile}
            with patch.object(lifecycle, 'validate_stage', return_value=report) as validate:
                self.assertIs(runner.probe_evidence(mode, 'raw lifecycle evidence', self.root), report)
                validate.assert_called_once_with(self.root, mode, 'raw lifecycle evidence')
            with patch.object(lifecycle, 'validate_stage', side_effect=ValueError('raw mismatch')):
                with self.assertRaisesRegex(ValueError, 'raw mismatch'):
                    runner.probe_evidence(mode, 'raw lifecycle evidence', self.root)
        with patch.object(lifecycle, 'validate_stage') as validate:
            self.assertEqual(runner.probe_evidence('self-check', 'true', self.root), {'self_check': True})
            for marker in ('HAG_LIFECYCLE_REPORT {}', 'PASS hag-lifecycle stage=answers qualified=true'):
                with self.assertRaises(ValueError): runner.probe_evidence('self-check', marker, self.root)
            validate.assert_not_called()

    def test_lifecycle_cpu_server_retirement_is_required_before_native(self):
        import lifecycle
        self.set_lifecycle()
        proc = self.root / 'proc'; proc.mkdir()
        unit = 'hag-answers-cpu-reference-20261008-010203-000004'
        prior = {'profile': self.profile, 'unit': unit, 'end': self.now.isoformat(),
                 'child': {'cgroup': '/sys/fs/cgroup/fixture-missing/' + unit + '.service', 'pid': 111},
                 'thermal': {'summary': {'pid': 222}}}
        session = {'armed_at': (self.now + timedelta(seconds=1)).isoformat()}
        properties = {'ActiveState':'inactive', 'SubState':'dead', 'MainPID':'0', 'ControlGroup':''}
        with patch.object(lifecycle, 'server_processes', return_value=[{'pid':333, 'exit_code':0}]) as processes:
            observed = runner.require_retired_cpu(prior, session, self.root, proc, lambda _: properties)
            self.assertEqual(observed['pids_absent'], [111,222,333])
            processes.assert_called_once_with(self.root, 'cpu-reference')
            (proc / '333').mkdir()
            with self.assertRaises(ValueError): runner.require_retired_cpu(prior, session, self.root, proc, lambda _: properties)
            (proc / '333').rmdir()
        for values in ([], [{'pid':333,'exit_code':1}], [{'pid':111,'exit_code':0}],
                       [{'pid':333,'exit_code':0}, {'pid':334,'exit_code':0}]):
            with patch.object(lifecycle, 'server_processes', return_value=values):
                with self.assertRaises(ValueError): runner.require_retired_cpu(prior, session, self.root, proc, lambda _: properties)

    def test_lifecycle_closeout_has_three_stages_raw_artifacts_and_no_timing(self):
        import lifecycle
        self.set_lifecycle()
        names = closeout.closed_evidence_names(self.root)
        for mode in profiles.profile_modes(self.profile):
            self.assertIn(mode + '-result.json', names)
            self.assertIn(mode + '-start-cooldown.json', names)
            for name in lifecycle.evidence_names(mode): self.assertIn(name, names)
        self.assertIn('answers-cpu-retirement.json', names)
        self.assertFalse(any(name.startswith('timing') or name.startswith('client/') for name in names))
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}')
        with patch.object(closeout, 'approved_contract', return_value=self.contract):
            self.assertEqual(set(closeout.closed_evidence_hashes(self.root)), set(names))
            missing = next(name for name in lifecycle.evidence_names('answers') if name not in ('answers-report.json', 'cpu-reference.json'))
            (self.root / missing).unlink()
            with self.assertRaisesRegex(ValueError, 'missing or unsafe'):
                closeout.closed_evidence_hashes(self.root)

    def test_lifecycle_closeout_requires_exact_order_and_scopes_cancellation_to_success(self):
        self.set_lifecycle()
        reports = {mode: {'start': f'2026-10-08T12:0{index * 2}:00+00:00',
                          'end': f'2026-10-08T12:0{index * 2 + 1}:00+00:00'}
                   for index, mode in enumerate(profiles.profile_modes(self.profile))}
        closeout.validate_order(reports, self.profile)
        for changed in ({**reports, 'timing': reports['answers']},
                        {mode: report for mode, report in reports.items() if mode != 'cpu-reference'}):
            with self.assertRaises(ValueError): closeout.validate_order(changed, self.profile)
        changed = copy.deepcopy(reports)
        changed['answers']['start'] = reports['cpu-reference']['start']
        with self.assertRaises(ValueError): closeout.validate_order(changed, self.profile)
        for qualified in (True, False):
            scope = closeout.qualification_scope(self.contract, qualified)
            self.assertIs(scope['live_cancellation_qualified'], qualified)
            self.assertFalse(scope['semantic_quality_qualified'])
            self.assertEqual(scope['qualification_kind'], 'tcp_lifecycle_and_backend_parity')

    def test_lifecycle_cleanup_binds_both_servers_and_three_containment_units(self):
        import lifecycle
        self.set_lifecycle()
        proc, groups = self.root / 'proc', self.root / 'cgroups'
        proc.mkdir(); groups.mkdir()
        properties = {'ActiveState':'inactive', 'SubState':'dead', 'MainPID':'0', 'ControlGroup':''}
        reports = {mode: {'unit': f'hag-answers-{mode}-20261008-120000-{index:06}',
                         'child': {'pid':100+index, 'cgroup':str(groups / mode)},
                         'thermal': {'summary': {'pid':200+index}}}
                   for index, mode in enumerate(profiles.profile_modes(self.profile))}
        def processes(root, mode):
            return [{'pid': 301 if mode == 'cpu-reference' else 302, 'exit_code':0}]
        def cleanup(values=reports):
            return closeout.cleanup_evidence(values, proc, groups, lambda _: properties, self.root, self.profile)
        with patch.object(lifecycle, 'server_processes', side_effect=processes):
            observed = cleanup()
            self.assertEqual(len(observed['recorded_pids_absent']), 8)
            self.assertEqual(observed['stages']['cpu-reference']['server_pids_absent'], [301])
            self.assertEqual(observed['stages']['answers']['server_pids_absent'], [302])
            (proc / '302').mkdir()
            with self.assertRaises(ValueError): cleanup()
            (proc / '302').rmdir()
            with self.assertRaises(ValueError): cleanup({**reports, 'timing': reports['answers']})
        for values in ([{'pid':301,'exit_code':1}], [{'pid':100,'exit_code':0}], [],
                       [{'pid':301,'exit_code':0}, {'pid':302,'exit_code':0}]):
            with patch.object(lifecycle, 'server_processes', return_value=values):
                with self.assertRaises(ValueError): cleanup()

    def report(self, mode):
        timing = mode == 'timing'
        limits = profiles.profile_limits(self.profile)
        timing_id = profiles.fixture_document(self.profile)['timing_fixture_id']
        reference = next(item for item in self.references if item['fixture_id'] == timing_id)
        evaluated = [reference['evaluated_positions']] * self.contract['stage_request_counts']['timing'] if timing else [item['evaluated_positions'] for item in self.references]
        report = {'stage': mode, 'profile': self.profile, 'execution': 'cpu-reference' if mode == 'cpu-reference' else 'native-hag-' + mode,
                  'qualification_kind': self.contract['qualification_kind'],
                  'require_semantic_match': self.contract['require_semantic_match'],
                  'quality_is_diagnostic': not self.contract['require_semantic_match'],
                  'capacity': limits['capacity'], 'fixture_ids': [timing_id] if timing else self.contract['fixture_ids'],
                  'requests': self.contract['stage_request_counts'][mode],
                  'maximum_evaluated_positions': self.contract['stage_evaluation_ceilings'][mode],
                  'evaluated_positions': sum(evaluated), 'all_eos': True, 'closed_answers': True,
                  'final_unloaded': True, 'reference_sha256': common.digest(self.root / 'cpu-reference.json'),
                  'audit_events': []}
        if self.client is not None:
            report['client_identity'] = copy.deepcopy(self.client)
        if mode == 'cpu-reference':
            report.update(cpu_exit_code=0, answers=copy.deepcopy(self.references))
        else:
            report.update(hag_exit_code=0, exact_cpu_parity=True,
                          cpu_reference_result_sha256=common.digest(self.root / 'cpu-reference-result.json'))
            report['audit_events'] = [{'request_sequence': i, 'evaluated_positions': count,
                                      'outcome': 'success', 'reset_ok': True, 'poisoned': False}
                                     for i, count in enumerate(evaluated, 1)]
            if not timing:
                report['answers'] = copy.deepcopy(self.references)
            else:
                report.update(answers_result_sha256=common.digest(self.root / 'answers-result.json'),
                              timing_fixture_id=timing_id, warmup_requests=1, measured_requests=limits['measured_requests'],
                              descriptive_client_visible_only=True,
                              ttfc_definition='first_nonempty_raw_content_including_thinking_markup',
                              rate_definition='api_completion_tokens_over_request_to_done_seconds', cpu_quota_percent=75,
                              warmup={'label': 'first_request_after_lazy_load', 'fixture_id': timing_id,
                                      'latency_seconds': 7., 'usage_source': 'nonstream_api',
                                      'generation': copy.deepcopy(reference)},
                              metrics=[{'round': i, 'fixture_id': timing_id, 'prompt_tokens': reference['usage']['prompt_tokens'],
                                        'completion_tokens': reference['usage']['completion_tokens'], 'content_chunks': 15, 'ttfc_seconds': 1.,
                                        'last_content_seconds': 3., 'done_seconds': 4.,
                                        'completion_tokens_per_second': reference['usage']['completion_tokens'] / 4.,
                                        'usage_source': 'matched_nonstream_warmup_api'} for i in range(1, limits['measured_requests'] + 1)])
        if self.profile in profiles.GROUNDED_PROFILES and mode != 'cpu-reference':
            report.update(client_chat_requests=1, client_stream_requests=int(timing),
                          model_free_client_refusals=2 if timing else 1, live_cancellation_qualified=False)
            if timing:
                report['client_reuse'] = True
                report['metrics'][0]['content_chunks'] = 2
            else:
                report['capacity_refusals'] = [{'phase':phase,'stream':stream,'status':400,
                    'code':'context_length_exceeded','param':'max_tokens','loaded':loaded,'audit_count':count}
                    for phase,stream,loaded,count in [('cold',False,False,0),('warm',True,True,1)]]
        return report

    def set_grounded(self, profile=profiles.GROUNDED_PROFILE):
        self.set_profile(profile)
        limits = profiles.profile_limits(profile)
        self.client = {'kind':'antimatter','commit':'78147b78d8271b993dc3734f469f763df1f0f548',
                       'manifest_sha256':'b'*64,'node_sha256':'c'*64,'bundle_sha256':'d'*64}
        path = self.root / 'cpu-reference.json'
        payload = common.read_json(path)
        payload['identities']['client_identity'] = self.client
        path.write_text(json.dumps(payload))
        mocked = patch.object(runner, 'client_identity', return_value=self.client)
        mocked.start()
        self.addCleanup(mocked.stop)
        reference, fixture = self.references[0], self.fixtures[0]
        for index, (stage, stem, artifact) in enumerate(closeout.client_artifacts(self.profile)):
            stream = artifact.endswith('-sse.json')
            wire = {'model':'hag-context-qwen3-0.6b','messages':fixture['messages'],
                    'max_tokens':128,'temperature':0,'stream':stream}
            client = {'kind':'antimatter','pid':800+index,'request_count':1,'attempt_count':1,
                      'request':wire,'response_complete':True,'error':None,'mode':'stream' if stream else 'chat',
                      'max_evaluated_tokens':limits['capacity'],
                      'preflight':{'capacity':limits['capacity'],'requested_output_tokens':limits.get('preflight_requested_output_tokens',512),'fetch_attempts':0,
                                   'track_calls':0,'prepare_calls':0,'error_code':'context_length_exceeded'},
                      'hooks':{'track_calls':1,'prepare_calls':1,'track_end_calls':1}}
            record = {'status':200,'content_type':'text/event-stream' if stream else 'application/json',
                      'clock':'monotonic_elapsed_from_request_start','events':[],'client':client}
            if stream:
                pieces = [reference['text'][:2],reference['text'][2:]]
                def event(delta, finish, elapsed):
                    value = {'id':'grounded','object':'chat.completion.chunk','model':'hag-context-qwen3-0.6b',
                             'choices':[{'index':0,'delta':delta,'finish_reason':finish}]}
                    return {'received_seconds':elapsed,'line':'data: '+json.dumps(value)+'\n'}
                record['events'] = [event({'role':'assistant'},None,0),event({'content':pieces[0]},None,1.),
                                    event({'content':pieces[1]},None,3.),event({},'stop',3.5),
                                    {'received_seconds':4.,'line':'data: [DONE]\n'}]
                client['result'] = {'content':reference['text'],'reasoning':'','finishReason':'stop',
                                    'servedModel':'hag-context-qwen3-0.6b','toolCalls':[],'toolCallsCutOff':False}
                client['callbacks'] = [{'id':'hag-client-grounded','content':piece} for piece in pieces]
            else:
                response = {'id':'grounded','object':'chat.completion','model':'hag-context-qwen3-0.6b',
                            'choices':[{'index':0,'finish_reason':'stop','message':{'role':'assistant','content':reference['text']}}],
                            'usage':reference['usage']}
                record.update(raw_body_base64=base64.b64encode(json.dumps(response).encode()).decode(),elapsed_seconds=7.)
                client.update(result=reference['text'],callbacks=[])
            envelope = {'profile':self.profile,'request':wire,
                        'model_descriptor':{'id':'hag-context-qwen3-0.6b','object':'model','owned_by':'beholdd',
                                            'loaded':stream,'execution_limits':{'max_evaluated_tokens':limits['capacity']}}}
            for name,value in [(artifact,record),(stem+'-client-request.json',envelope),
                               (stem+'-client-process.json',{'pid':800+index,'exit_code':0,'timed_out':False,'identity':self.client})]:
                (self.root/name).write_text(json.dumps(value))
            (self.root/(stem+'-client.log')).write_text('')
        for phase,stream,loaded,count in [('cold',False,False,0),('warm',True,True,1)]:
            response = {'error':{'type':'invalid_request_error','code':'context_length_exceeded',
                                 'param':'max_tokens','message':'capacity exceeded'}}
            record = {'request':{'model':'hag-context-qwen3-0.6b','messages':fixture['messages'],
                                 'max_tokens':limits.get('overflow_max_tokens',421),'temperature':0,'stream':stream},
                      'status':400,'content_type':'application/json','response':response,
                      'raw_body_base64':base64.b64encode(json.dumps(response).encode()).decode(),
                      'loaded_before':loaded,'audit_count_before':count}
            (self.root/('answers-'+phase+'-refusal.json')).write_text(json.dumps(record))
            (self.root/('answers-'+phase+'-refusal-post.json')).write_text(json.dumps(
                {'loaded_after':loaded,'audit_count_after':count,'unchanged_audit':True}))
        for mode in ('answers','timing'):
            (self.root/(mode+'-hag-server.log')).write_text(''.join(
                'HAG_SERVING_AUDIT '+json.dumps(e)+'\n' for e in self.report(mode)['audit_events']))

    def test_template_history_reports_require_corrected_usage_and_preserve_closed_inventory(self):
        self.set_grounded(profiles.TEMPLATE_HISTORY_PROFILE)
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)
            self.assertEqual(report['requests'], 2 if mode == 'timing' else 1)
            self.assertEqual(report['maximum_evaluated_positions'], 1236 if mode == 'timing' else 618)
            changed = copy.deepcopy(report)
            changed['profile'] = profiles.HISTORY_PROFILE
            with self.assertRaises(ValueError): runner.probe_evidence(mode, log(mode, changed), self.root)
        names = closeout.closed_evidence_names(self.root)
        self.assertIn('cpu-reference-project_history_template_ready-response.json', names)
        self.assertIn('answers-project_history_template_ready-client-request.json', names)
        self.assertNotIn('answers-project_history_ready-client-request.json', names)
        self.assertNotIn('timing-round-2-sse.json', names)
        self.assertFalse(closeout.qualification_scope(self.contract, True)['semantic_quality_qualified'])
        original = copy.deepcopy(self.contract)
        for change in ({'exact_prompt_tokens': 495}, {'overflow_max_tokens': 1555},
                       {'stage_evaluation_ceilings': {'cpu-reference': 622, 'answers': 622, 'timing': 1244}}):
            (self.root / 'qualification-contract.json').write_text(json.dumps({**original, **change}))
            with self.assertRaises(ValueError): common.approved_contract(self.root)

    def test_template_history_supervisor_rejects_prior_count_and_seed_transformations(self):
        self.set_grounded(profiles.TEMPLATE_HISTORY_PROFILE)
        fixture, reference = self.fixtures[0], self.references[0]
        stem = 'answers-project_history_template_ready'
        path = self.root / (stem + '-response.json')
        original = common.read_json(path)
        for change in (lambda r: r['client']['preflight'].update(requested_output_tokens=1555),
                       lambda r: r['client']['request']['messages'][2].update(content='ready.'),
                       lambda r: r.update(elapsed_seconds=240.01)):
            changed = copy.deepcopy(original); change(changed)
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                runner.validate_grounded_chat(self.root, stem, path.name, reference, fixture, self.client, self.profile)
        path.write_text(json.dumps(original))
        for phase in ('cold', 'warm'):
            path = self.root / ('answers-' + phase + '-refusal.json')
            record = common.read_json(path)
            self.assertEqual(record['request']['max_tokens'], 1559)
            record['request']['max_tokens'] = 1555
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): runner.validate_capacity_refusals(self.root, fixture, self.profile)
            record['request']['max_tokens'] = 1559
            path.write_text(json.dumps(record))

    def test_history_reports_preserve_second_turn_scope_and_exact_stage_counts(self):
        self.set_grounded(profiles.HISTORY_PROFILE)
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)
            self.assertEqual(report['requests'], 2 if mode == 'timing' else 1)
            self.assertFalse(report['require_semantic_match'])
            self.assertTrue(report['quality_is_diagnostic'])
        self.assertEqual(self.contract['stage_evaluation_ceilings'],
                         {'cpu-reference': 622, 'answers': 622, 'timing': 1244})
        names = closeout.closed_evidence_names(self.root)
        self.assertIn('cpu-reference-project_history_ready-response.json', names)
        self.assertIn('answers-project_history_ready-client-request.json', names)
        self.assertNotIn('answers-project_instructions_ready-response.json', names)
        self.assertNotIn('timing-round-2-sse.json', names)
        for mode in profiles.profile_modes(self.profile):
            self.assertIn(mode + '-start-cooldown.json', names)
        for mode in ('answers', 'timing'):
            self.assertIn(mode + '-cpu-retirement.json', names)
        scope = closeout.qualification_scope(self.contract, True)
        self.assertFalse(scope['semantic_quality_qualified'])
        self.assertNotIn('live_cancellation_qualified', scope)

    def test_history_supervisor_rejects_seed_rewriting_profile_substitution_and_bad_refusal(self):
        self.set_grounded(profiles.HISTORY_PROFILE)
        fixture, reference = self.fixtures[0], self.references[0]
        self.assertEqual(fixture['messages'][2]['content'], '<think>\n\n</think>\n\nready.')
        stem = 'answers-project_history_ready'
        path = self.root / (stem + '-response.json')
        original = common.read_json(path)
        for mutation in (lambda r: r['client']['request']['messages'][2].update(content='ready.'),
                         lambda r: r['client']['request']['messages'].pop(2),
                         lambda r: r['client']['preflight'].update(requested_output_tokens=1583),
                         lambda r: r['client']['preflight'].update(requested_output_tokens=1188)):
            changed = copy.deepcopy(original)
            mutation(changed)
            path.write_text(json.dumps(changed))
            with self.subTest(record=changed), self.assertRaises(ValueError):
                runner.validate_grounded_chat(self.root, stem, path.name, reference, fixture,
                                              self.client, self.profile)
        path.write_text(json.dumps(original))
        changed = self.report('answers')
        changed['profile'] = profiles.PROJECT_PROFILE
        with self.assertRaises(ValueError): runner.probe_evidence('answers', log('answers', changed), self.root)
        for phase in ('cold', 'warm'):
            path = self.root / ('answers-' + phase + '-refusal.json')
            record = common.read_json(path)
            self.assertEqual(record['request']['max_tokens'], 1555)
            self.assertEqual(record['request']['messages'], fixture['messages'])
            record['request']['max_tokens'] = 1583
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError): runner.validate_capacity_refusals(self.root, fixture, self.profile)
            record['request']['max_tokens'] = 1555
            path.write_text(json.dumps(record))

    def test_history_contract_rejects_previous_prompt_bounds_and_extra_generation(self):
        self.set_grounded(profiles.HISTORY_PROFILE)
        original = copy.deepcopy(self.contract)
        for key, wrong in [('exact_prompt_tokens', 467), ('maximum_prompt_tokens', 467),
                           ('overflow_max_tokens', 1583), ('client_request_timeout_seconds', 241),
                           ('private_cgroup_admission', False), ('crash_watcher_seconds', 600)]:
            (self.root / 'qualification-contract.json').write_text(json.dumps({**original, key: wrong}))
            with self.subTest(key=key), self.assertRaises(ValueError): common.approved_contract(self.root)
        (self.root / 'qualification-contract.json').write_text(json.dumps(original))
        for mutation in (lambda r: r.update(requests=2), lambda r: r.update(live_cancellation_qualified=True),
                         lambda r: r['audit_events'].append({**r['audit_events'][0], 'request_sequence': 2})):
            report = self.report('answers')
            mutation(report)
            with self.subTest(report=report), self.assertRaises(ValueError):
                runner.probe_evidence('answers', log('answers', report), self.root)

    def test_history_inherits_private_containment_cpu_retirement_and_full_watcher_reserve(self):
        self.set_grounded(profiles.HISTORY_PROFILE)
        snapshot = {'memory_current_bytes': 1024, 'memory_peak_bytes': 2048, 'swap_current_bytes': 0,
                    'swap_peak_bytes': 0, 'memory_events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        for mode, deadlines in profiles.profile_deadlines(self.profile).items():
            membership = '/user.slice/hag-answers-' + mode + '-20261008-000000-000001.service'
            child_record = {'stage_deadlines': dict(zip(('child', 'service', 'outer'), deadlines)),
                            'cgroup': '/sys/fs/cgroup' + membership, 'limits': common.EXPECTED_LIMITS,
                            'memory_before': snapshot,
                            'private_cgroup_attestation': membership if mode in common.GPU_MODES else None}
            runner.validate_project_child(child_record, mode, self.profile)
            self.assertIn('--property=RuntimeMaxSec=' + str(deadlines[1]) + 's',
                          runner.unit_command('fixture', mode, self.profile))
            for mutation in (lambda c: c.update(private_cgroup_attestation='/forged'),
                             lambda c: c['stage_deadlines'].update(child=1),
                             lambda c: c['limits'].update(**{'memory.swap.max': '1'})):
                changed = copy.deepcopy(child_record)
                mutation(changed)
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    runner.validate_project_child(changed, mode, self.profile)
        records = copy.deepcopy(self.capture)
        records[0]['duration_seconds'] = 1500
        runner.validate_capture_lease(records, self.now, 'timing', self.profile)
        with self.assertRaises(ValueError):
            runner.validate_capture_lease(records, self.now + timedelta(seconds=850), 'timing', self.profile)
        proc = self.root / 'proc'
        proc.mkdir()
        unit = 'hag-answers-cpu-reference-20261008-000000-000001'
        prior = {'profile': self.profile, 'unit': unit, 'end': self.now.isoformat(),
                 'child': {'cgroup': '/sys/fs/cgroup/fixture-missing/' + unit + '.service', 'pid': 111},
                 'thermal': {'summary': {'pid': 222}}}
        session = {'armed_at': (self.now + timedelta(seconds=1)).isoformat()}
        properties = {'ActiveState': 'inactive', 'SubState': 'dead', 'MainPID': '0', 'ControlGroup': ''}
        observed = runner.require_retired_cpu(prior, session, self.root, proc, lambda _: properties)
        self.assertEqual(observed['pids_absent'], [111, 222])
        (proc / '111').mkdir()
        with self.assertRaises(ValueError):
            runner.require_retired_cpu(prior, session, self.root, proc, lambda _: properties)
        (proc / '111').rmdir()
        with self.assertRaises(ValueError):
            runner.require_retired_cpu(prior, {'armed_at': self.now.isoformat()}, self.root, proc,
                                      lambda _: {**properties, 'ActiveState': 'active'})

    def test_history_closeout_binds_all_three_clients_and_refuses_missing_raw_history(self):
        self.set_grounded(profiles.HISTORY_PROFILE)
        names = closeout.closed_evidence_names(self.root)
        for name in names:
            path = self.root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n')
        hashes = closeout.closed_evidence_hashes(self.root)
        history_request = 'answers-project_history_ready-client-request.json'
        self.assertEqual(hashes[history_request], common.digest(self.root / history_request))
        proc, cgroups = self.root / 'proc', self.root / 'cgroups'
        proc.mkdir()
        cgroups.mkdir()
        reports = {mode: {'unit': f'hag-answers-{mode}-20261008-120000-{i:06}',
                         'child': {'pid': 100 + i, 'cgroup': str(cgroups / mode)},
                         'thermal': {'summary': {'pid': 200 + i}}}
                   for i, mode in enumerate(profiles.profile_modes(self.profile))}
        reports['timing']['client_identity'] = self.client
        properties = {'ActiveState': 'inactive', 'SubState': 'dead', 'MainPID': '0', 'ControlGroup': ''}
        observed = closeout.cleanup_evidence(reports, proc, cgroups, lambda _: properties, self.root, self.profile)
        self.assertEqual(len(observed['recorded_pids_absent']), 11)
        self.assertEqual(observed['stages']['answers']['client_pids_absent'], [800])
        self.assertEqual(observed['stages']['timing']['client_pids_absent'], [801, 802])
        (proc / '802').mkdir()
        with self.assertRaises(ValueError):
            closeout.cleanup_evidence(reports, proc, cgroups, lambda _: properties, self.root, self.profile)
        (self.root / history_request).unlink()
        with self.assertRaises(ValueError): closeout.closed_evidence_hashes(self.root)

    def test_project_full_cpu_chat_stream_scope_and_closed_artifacts(self):
        self.set_grounded(profiles.PROJECT_PROFILE)
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)
        names = closeout.closed_evidence_names(self.root)
        self.assertIn('answers-project_instructions_ready-response.json', names)
        self.assertNotIn('answers-grounded_no_tools_ready-response.json', names)
        self.assertIn('timing-round-1-client-process.json', names)
        self.assertNotIn('timing-round-2-sse.json', names)
        for mode in common.MODES:
            self.assertIn(mode + '-start-cooldown.json', names)
        self.assertEqual(self.contract['stage_evaluation_ceilings'],
                         {'cpu-reference':594, 'answers':594, 'timing':1188})

    def test_project_refuses_changed_budgets_deadlines_cap_or_private_policy(self):
        self.set_grounded(profiles.PROJECT_PROFILE)
        original = copy.deepcopy(self.contract)
        for key, bad in [('capacity',512), ('cpu_model_budget_bytes',1024**3),
                         ('hag_budget_bytes',768*1024**2), ('private_cgroup_admission',False),
                         ('crash_watcher_seconds',600), ('cpu_request_timeout_seconds',240),
                         ('client_request_timeout_seconds',241), ('overflow_max_tokens',1219)]:
            changed = {**original, key:bad}
            (self.root/'qualification-contract.json').write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.approved_contract(self.root)
        for stage in common.MODES:
            changed = copy.deepcopy(original)
            changed['stage_deadlines'][stage]['child'] += 1
            (self.root/'qualification-contract.json').write_text(json.dumps(changed))
            with self.subTest(stage=stage), self.assertRaises(ValueError):
                common.approved_contract(self.root)
        (self.root/'qualification-contract.json').write_text(json.dumps(original))
        self.assertEqual(common.approved_contract(self.root), original)
        for stage, expected in profiles.profile_deadlines(self.profile).items():
            self.assertEqual(common.stage_deadlines(stage, self.root), expected)
            self.assertIn('--property=RuntimeMaxSec=' + str(expected[1]) + 's',
                          runner.unit_command('fixture', stage, self.profile))
        cpu = inputs.model_config('/model','/device','cpu',self.profile)['models'][0]
        hag = inputs.model_config('/model','/device','hag',self.profile)['models'][0]
        self.assertEqual(cpu['memory_budget_bytes'],1536*1024**2)
        self.assertEqual(hag['hag']['memory_budget_bytes'],1152*1024**2)
        self.assertEqual(inputs.model_config('/model','/device','cpu')['models'][0]['memory_budget_bytes'],1024**3)

    def test_project_timing_above_old_deadline_binds_raw_callbacks_and_new_cap(self):
        self.set_grounded(profiles.PROJECT_PROFILE)
        path = self.root/'timing-round-1-sse.json'
        record = common.read_json(path)
        for event in record['events']:
            event['received_seconds'] += 200
        path.write_text(json.dumps(record))
        observed = runner.validate_antimatter_stream(self.root,self.references[0],self.fixtures[0],1,
                                                    self.client,self.profile)
        self.assertEqual(observed['done_seconds'],204)
        for mutation in (lambda x:x['events'][-1].update(received_seconds=240.1),
                         lambda x:x['client'].update(max_evaluated_tokens=512),
                         lambda x:x['client']['preflight'].update(requested_output_tokens=1219),
                         lambda x:x['client']['callbacks'][0].update(content='wrong')):
            changed = copy.deepcopy(record); mutation(changed)
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                runner.validate_antimatter_stream(self.root,self.references[0],self.fixtures[0],1,
                                                  self.client,self.profile)

    def test_project_watcher_lease_and_finite_cool_start(self):
        records = copy.deepcopy(self.capture)
        records[0]['duration_seconds'] = 1500
        runner.validate_capture_lease(records,self.now,'timing')
        for lease in (600,1501,True):
            records[0]['duration_seconds'] = lease
            with self.assertRaises(ValueError): runner.validate_capture_lease(records,self.now,'timing')
        records[0]['duration_seconds'] = 1500
        with self.assertRaises(ValueError):
            runner.validate_capture_lease(records,self.now+timedelta(seconds=900),'timing')
        cool = {'qualified':True,'maximum_c':60,'stable_seconds':10,
                'samples':[{'elapsed_seconds':index/10,'tctl_c':60.0} for index in range(102)]}
        runner.validate_cooldown(cool)
        for mutation in (lambda x:x['samples'][50].update(tctl_c=60.1),
                         lambda x:x['samples'].pop(50),
                         lambda x:x.update(qualified=False)):
            changed = copy.deepcopy(cool); mutation(changed)
            # Missing one .1s sample is allowed; a >1s gap is not.
            if len(changed['samples']) == 101: del changed['samples'][30:45]
            with self.assertRaises(ValueError): runner.validate_cooldown(changed)

    def test_private_attestation_requires_actual_dedicated_clean_containment(self):
        snapshot = {'memory_current_bytes':1024,'memory_peak_bytes':2048,'swap_current_bytes':0,
                    'swap_peak_bytes':0,'memory_events':{'high':0,'max':0,'oom':0,'oom_kill':0}}
        cgroup = Path('/sys/fs/cgroup/user.slice/hag-answers-answers-20261007-123456-123456.service')
        observed = common.private_cgroup_attestation(cgroup,common.EXPECTED_LIMITS,'NoNewPrivs: 1','0',snapshot)
        self.assertEqual(observed,'/user.slice/'+cgroup.name)
        for path, actual, status, values in [(cgroup.parent,common.EXPECTED_LIMITS,'NoNewPrivs: 1',snapshot),
                (cgroup,{**common.EXPECTED_LIMITS,'memory.max':'max'},'NoNewPrivs: 1',snapshot),
                (cgroup,common.EXPECTED_LIMITS,'NoNewPrivs: 0',snapshot),
                (cgroup,common.EXPECTED_LIMITS,'NoNewPrivs: 1',{**snapshot,'swap_current_bytes':1})]:
            with self.assertRaises(ValueError): common.private_cgroup_attestation(path,actual,status,'0',values)

    def test_attestation_environment_is_native_project_only(self):
        path = '/user.slice/hag-answers-answers-20261007-000000-000001.service'
        for profile in profiles.PROFILE_NAMES:
            for mode in profiles.profile_modes(profile):
                env = child.workload_environment(mode,profile,path,{'BEHOLDD_HAG_PRIVATE_CGROUP':'forged'})
                self.assertEqual(env.get('BEHOLDD_HAG_PRIVATE_CGROUP'),
                                 path if profile in profiles.C2048_PROFILES and mode in common.GPU_MODES else None)
        with self.assertRaises(ValueError): child.workload_environment('answers',profiles.PROJECT_PROFILE,None,{})

    def test_project_cpu_retirement_is_required_before_native_capture(self):
        unit = 'hag-answers-cpu-reference-20261007-010203-000004'
        cgroup = '/sys/fs/cgroup/fixture-missing/' + unit + '.service'
        proc = self.root/'proc'; proc.mkdir()
        prior = {'unit':unit,'end':self.now.isoformat(),'child':{'cgroup':cgroup,'pid':111},
                 'thermal':{'summary':{'pid':222}}}
        session = {'armed_at':(self.now+timedelta(seconds=1)).isoformat()}
        properties = {'ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlGroup':''}
        receipt = runner.require_retired_cpu(prior,session,self.root,proc,lambda _:properties)
        self.assertEqual(receipt['pids_absent'],[111,222])
        (proc/'111').mkdir()
        with self.assertRaises(ValueError): runner.require_retired_cpu(prior,session,self.root,proc,lambda _:properties)
        (proc/'111').rmdir()
        with self.assertRaises(ValueError):
            runner.require_retired_cpu(prior,{'armed_at':(self.now-timedelta(seconds=1)).isoformat()},
                                       self.root,proc,lambda _:properties)
        with self.assertRaises(ValueError):
            runner.require_retired_cpu(prior,session,self.root,proc,
                                       lambda _:{**properties,'ActiveState':'active'})

    def test_grounded_complete_cpu_chat_stream_and_refusals_have_exact_scope(self):
        self.set_grounded()
        for mode in ('cpu-reference','answers','timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode,log(mode,report),self.root),report)
        self.assertEqual(self.report('timing')['requests'],2)
        self.assertFalse(closeout.qualification_scope(self.contract,True)['semantic_quality_qualified'])

    def test_grounded_prompt_and_eos_limits_are_exact_without_changing_c128(self):
        self.set_grounded()
        original = self.references[0]
        for prompt, completion in [(92,5),(94,5),(93,128)]:
            value = copy.deepcopy(original)
            value['usage'] = {'prompt_tokens':prompt,'completion_tokens':completion,'total_tokens':prompt+completion}
            value['evaluated_positions'] = prompt+completion
            with self.subTest(prompt=prompt,completion=completion), self.assertRaises(ValueError):
                runner.validate_answer(value,self.fixtures[0],self.profile)
        with self.assertRaises(ValueError):
            runner.validate_answer(original,self.fixtures[0],'backend-parity-c128')

    def test_grounded_raw_chat_cannot_disagree_with_provider_or_cpu(self):
        self.set_grounded()
        path = self.root/'answers-grounded_no_tools_ready-response.json'
        original = common.read_json(path)
        for kind in ('provider','raw','usage','eos','malformed','deadline'):
            value = copy.deepcopy(original)
            body = runner.raw_json_body(value)
            if kind == 'provider': value['client']['result'] = 'Wrong'
            elif kind == 'deadline': value['elapsed_seconds'] = 60.01
            elif kind == 'malformed': value['raw_body_base64'] = '!invalid'
            else:
                if kind == 'raw': body['choices'][0]['message']['content'] = 'Wrong'
                elif kind == 'usage': body['usage']['completion_tokens'] += 1
                else: body['choices'][0]['finish_reason'] = 'length'
                value['raw_body_base64'] = base64.b64encode(json.dumps(body).encode()).decode()
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('answers',log('answers',self.report('answers')),self.root)

    def test_grounded_capacity_and_local_refusal_precede_hooks_or_fetch(self):
        self.set_grounded()
        path = self.root/'timing-round-1-sse.json'
        original = common.read_json(path)
        for kind in ('fetch','prepare','track','end','retry','cap','mode','callback'):
            value = copy.deepcopy(original)
            client = value['client']
            if kind == 'fetch': client['preflight']['fetch_attempts'] = 1
            elif kind == 'prepare': client['preflight']['prepare_calls'] = 1
            elif kind == 'track': client['preflight']['track_calls'] = 1
            elif kind == 'end': client['hooks']['track_end_calls'] = 0
            elif kind == 'retry': client['attempt_count'] = 2
            elif kind == 'cap': client['max_evaluated_tokens'] = 128
            elif kind == 'mode': client['mode'] = 'chat'
            else: client['callbacks'][0]['id'] = 'hag-client-list'
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('timing',log('timing',self.report('timing')),self.root)

    def test_grounded_refusals_bind_raw_typed400_state_and_no_extra_generation(self):
        self.set_grounded()
        path = self.root/'answers-cold-refusal.json'
        original = common.read_json(path)
        for kind in ('status','type','cap','loaded','raw'):
            value = copy.deepcopy(original)
            if kind == 'status': value['status'] = 200
            elif kind == 'type': value['content_type'] = 'text/event-stream'
            elif kind == 'cap': value['request']['max_tokens'] = 420
            elif kind == 'loaded': value['loaded_before'] = True
            else: value['raw_body_base64'] = base64.b64encode(b'{}').decode()
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('answers',log('answers',self.report('answers')),self.root)
        path.write_text(json.dumps(original))
        after = self.root/'answers-warm-refusal-post.json'
        value = common.read_json(after)
        value['audit_count_after'] = 2
        after.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            runner.probe_evidence('answers',log('answers',self.report('answers')),self.root)

    def test_grounded_descriptor_and_wire_body_are_bound(self):
        self.set_grounded()
        path = self.root/'timing-round-1-client-request.json'
        original = common.read_json(path)
        for kind in ('profile','capacity','request'):
            value = copy.deepcopy(original)
            if kind == 'profile': value['profile'] = 'backend-parity-c128'
            elif kind == 'capacity': value['model_descriptor']['execution_limits']['max_evaluated_tokens'] = None
            else: value['request']['max_tokens'] = 4096
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('timing',log('timing',self.report('timing')),self.root)

    def test_grounded_report_cannot_claim_extra_requests_or_live_cancellation(self):
        self.set_grounded()
        for mode, key, value in [('timing','requests',4),('timing','measured_requests',3),
                                 ('answers','client_chat_requests',0),('answers','model_free_client_refusals',0),
                                 ('timing','live_cancellation_qualified',True),('timing','client_reuse',False)]:
            report = self.report(mode)
            report[key] = value
            with self.subTest(mode=mode,key=key), self.assertRaises(ValueError):
                runner.probe_evidence(mode,log(mode,report),self.root)
        path = self.root/'answers-hag-server.log'
        path.write_text(path.read_text() + 'HAG_SERVING_AUDIT ' + json.dumps(
            {**self.report('answers')['audit_events'][0],'request_sequence':2}) + '\n')
        with self.assertRaises(ValueError):
            runner.probe_evidence('answers',log('answers',self.report('answers')),self.root)

    def test_grounded_stream_deadline_raw_protocol_and_waited_exit_are_required(self):
        self.set_grounded()
        path = self.root/'timing-round-1-sse.json'
        original = common.read_json(path)
        for kind in ('malformed','after-done','late','result'):
            value = copy.deepcopy(original)
            if kind == 'malformed': value['events'][1]['line'] = 'data: {broken\n'
            elif kind == 'after-done': value['events'].append({'received_seconds':5.,'line':'data: {}\n'})
            elif kind == 'late': value['events'][-1]['received_seconds'] = 60.01
            else: value['client']['result']['content'] = 'Different'
            path.write_text(json.dumps(value))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('timing',log('timing',self.report('timing')),self.root)
        path.write_text(json.dumps(original))
        process = self.root/'timing-round-1-client-process.json'
        receipt = common.read_json(process)
        receipt['exit_code'] = 1
        process.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):
            runner.probe_evidence('timing',log('timing',self.report('timing')),self.root)

    def test_long_prefill_keepalive_is_not_content_or_a_timing_boundary(self):
        self.set_grounded()
        path = self.root / 'timing-round-1-sse.json'
        record = common.read_json(path)
        # The failed live receipt had a role at 0.061s, this physical comment
        # at 10.036s, first content at 19.567s and EOS/DONE at 20.947s.
        observed = [0.061102809999999994, 19.566607067999996, 20.712566606999996,
                    20.946606727, 20.946606727]
        for event, received in zip(record['events'], observed):
            event['received_seconds'] = received
        record['events'][1:1] = [
            {'received_seconds': 10.036092328999999, 'line': ': keep-alive\n'},
            {'received_seconds': 10.036092328999999, 'line': '\n'},
        ]
        record['events'].append({'received_seconds': observed[-1], 'line': '\n'})
        path.write_text(json.dumps(record))
        metric = runner.validate_antimatter_stream(self.root, self.references[0], self.fixtures[0],
                                                  1, self.client, self.profile)
        self.assertEqual(metric['content_chunks'], 2)
        self.assertEqual(metric['ttfc_seconds'], observed[1])
        self.assertEqual(metric['last_content_seconds'], observed[2])
        self.assertEqual(metric['done_seconds'], observed[-1])
        self.assertEqual(metric['completion_tokens_per_second'], 5 / observed[-1])
        # CRLF comments are also physical SSE comments, within the same bounds.
        record['events'][1]['line'] = ': keep-alive\r\n'
        path.write_text(json.dumps(record))
        self.assertEqual(runner.validate_antimatter_stream(self.root, self.references[0], self.fixtures[0],
                                                         1, self.client, self.profile), metric)

    def test_keepalive_cannot_hide_frames_relax_bounds_or_extend_done(self):
        self.set_grounded()
        path = self.root / 'timing-round-1-sse.json'
        original = common.read_json(path)
        variants = []
        for line in ('event: keep-alive\n', ' keep-alive\n', ' : keep-alive\n',
                     ': keep-alive\ndata: [DONE]\n', ': keep-alive\rdata: [DONE]\n',
                     ': ' + 'x' * 65536 + '\n'):
            value = copy.deepcopy(original)
            value['events'].insert(1, {'received_seconds': .5, 'line': line})
            variants.append(value)
        for received in (-1, float('inf'), 60.01):
            value = copy.deepcopy(original)
            value['events'].insert(1, {'received_seconds': received, 'line': ': keep-alive\n'})
            variants.append(value)
        for line in (': keep-alive\n', 'data: [DONE]\n', 'data: {}\n'):
            value = copy.deepcopy(original)
            value['events'].append({'received_seconds': 5., 'line': line})
            variants.append(value)
        missing_done = copy.deepcopy(original)
        missing_done['events'][-1]['line'] = ': keep-alive\n'
        variants.append(missing_done)
        missing_stop = copy.deepcopy(original)
        missing_stop['events'][-2]['line'] = ': keep-alive\n'
        variants.append(missing_stop)
        excess_events = copy.deepcopy(original)
        excess_events['events'][1:1] = [{'received_seconds': .5, 'line': ':\n'}] * 4096
        variants.append(excess_events)
        for record in variants:
            path.write_text(json.dumps(record))
            with self.subTest(events=record['events'][:3]), self.assertRaises(ValueError):
                runner.validate_antimatter_stream(self.root, self.references[0], self.fixtures[0],
                                                  1, self.client, self.profile)

    def test_grounded_closeout_includes_both_chat_processes_refusals_and_one_stream(self):
        self.set_grounded()
        names = closeout.closed_evidence_names(self.root)
        for phase in ('cold','warm'):
            self.assertIn('answers-'+phase+'-refusal.json',names)
            self.assertIn('answers-'+phase+'-refusal-post.json',names)
        for stage,stem,artifact in closeout.client_artifacts(self.profile):
            self.assertIn(artifact,names)
            for suffix in ('-request.json','-process.json','.log'):
                self.assertIn(stem+'-client'+suffix,names)
        self.assertIn('client/source/src/api/tokenEstimate.ts',names)
        self.assertNotIn('timing-round-2-sse.json',names)
        proc,cgroups = self.root/'proc',self.root/'cgroups'
        proc.mkdir()
        cgroups.mkdir()
        reports = {mode:{'unit':f'hag-answers-{mode}-20261005-120000-{i:06}',
                   'child':{'pid':100+i,'cgroup':str(cgroups/mode)},
                   'thermal':{'summary':{'pid':200+i}}} for i,mode in enumerate(common.MODES)}
        reports['timing']['client_identity'] = self.client
        properties = {'ActiveState':'inactive','SubState':'dead','MainPID':'0','ControlGroup':''}
        observed = closeout.cleanup_evidence(reports,proc,cgroups,lambda _:properties,self.root)
        self.assertEqual(observed['stages']['answers']['client_pids_absent'],[800])
        self.assertEqual(observed['stages']['timing']['client_pids_absent'],[801,802])
        (proc/'800').mkdir()
        with self.assertRaisesRegex(ValueError,'client PID'):
            closeout.cleanup_evidence(reports,proc,cgroups,lambda _:properties,self.root)

    def set_client(self):
        self.set_profile('backend-parity-c128')
        self.client = {'kind': 'antimatter', 'commit': 'a' * 40, 'manifest_sha256': 'b' * 64,
                       'node_sha256': 'c' * 64, 'bundle_sha256': 'd' * 64}
        self.contract['stream_client'] = 'antimatter'
        (self.root / 'qualification-contract.json').write_text(json.dumps(self.contract))
        path = self.root / 'cpu-reference.json'
        payload = common.read_json(path)
        payload['identities']['client_identity'] = self.client
        path.write_text(json.dumps(payload))
        mocked = patch.object(runner, 'client_identity', return_value=self.client)
        mocked.start()
        self.addCleanup(mocked.stop)
        for index in range(1, 4):
            self.client_stream_fixture(index)

    def client_stream_fixture(self, index):
        reference = self.references[0]
        request = {'model': 'hag-context-qwen3-0.6b', 'messages': self.fixtures[0]['messages'],
                   'max_tokens': 80, 'temperature': 0, 'stream': True}
        def event(delta, reason, elapsed):
            value = {'id': 'client-response', 'object': 'chat.completion.chunk',
                     'model': 'hag-context-qwen3-0.6b',
                     'choices': [{'index': 0, 'delta': delta, 'finish_reason': reason}]}
            return {'received_seconds': elapsed, 'line': 'data: ' + json.dumps(value) + '\n'}
        # Fifteen wire chunks agree with the ordinary synthetic timing report.
        content = list(reference['text'][:14]) + [reference['text'][14:]]
        record = {'status': 200, 'content_type': 'text/event-stream',
                  'clock': 'monotonic_elapsed_from_request_start',
                  'events': [event({'role': 'assistant'}, None, 0)]
                            + [event({'content': text}, None, 1 + 2 * pos / 14)
                               for pos, text in enumerate(content)]
                            + [event({}, 'stop', 3.5), {'received_seconds': 4., 'line': 'data: [DONE]\n'}],
                  'client': {'kind': 'antimatter', 'pid': 700 + index, 'request_count': 1, 'attempt_count': 1,
                             'request': request, 'response_complete': True, 'error': None,
                             'result': {'content': reference['text'], 'reasoning': '', 'finishReason': 'stop',
                                        'servedModel': 'hag-context-qwen3-0.6b', 'toolCalls': [], 'toolCallsCutOff': False},
                             'callbacks': [{'id': 'hag-client-list', 'content': text} for text in content]}}
        stem = 'timing-round-' + str(index)
        for name, value in ((stem + '-sse.json', record), (stem + '-client-request.json', request),
                            (stem + '-client-process.json', {'pid': 700 + index, 'exit_code': 0,
                                                           'timed_out': False, 'identity': self.client})):
            (self.root / name).write_text(json.dumps(value))
        (self.root / (stem + '-client.log')).write_text('')
        return record

    def test_optional_client_binds_cpu_native_timing_and_prerequisite_identity(self):
        self.set_client()
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)
            for identity in (None, {**self.client, 'bundle_sha256': 'e' * 64}):
                changed = copy.deepcopy(report)
                if identity is None:
                    changed.pop('client_identity')
                else:
                    changed['client_identity'] = identity
                with self.subTest(mode=mode, identity=identity), self.assertRaises(ValueError):
                    runner.probe_evidence(mode, log(mode, changed), self.root)
        with self.assertRaises(ValueError):
            runner.require_client_identity({}, self.client)
        with self.assertRaises(ValueError):
            runner.require_client_identity({'client_identity': self.client}, None)
        with self.assertRaises(ValueError):
            runner.require_client_identity({'client_identity': None}, None)
        path = self.root / 'cpu-reference.json'
        payload = common.read_json(path)
        payload['identities']['client_identity']['node_sha256'] = 'e' * 64
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)

    def test_optional_client_cannot_hide_parsed_callback_or_raw_stream_mismatch(self):
        self.set_client()
        original = self.client_stream_fixture(1)
        for kind in ('result', 'callback', 'callback-boundaries', 'raw', 'raw-malformed', 'done', 'eos', 'order', 'deadline'):
            record = copy.deepcopy(original)
            if kind == 'result': record['client']['result']['content'] += '!'
            elif kind == 'callback': record['client']['callbacks'][0]['content'] = '!'
            elif kind == 'callback-boundaries':
                record['client']['callbacks'] = [{'id': 'hag-client-list', 'content': self.references[0]['text']}]
            elif kind == 'raw': record['events'][1]['line'] = record['events'][1]['line'].replace('"1"', '"9"')
            elif kind == 'raw-malformed': record['events'][1]['line'] = 'data: {broken\n'
            elif kind == 'done': record['events'].pop()
            elif kind == 'eos': record['events'][-2]['line'] = record['events'][-2]['line'].replace('"stop"', '"length"')
            elif kind == 'order': record['events'][1]['received_seconds'] = 5.
            else: record['events'][-1]['received_seconds'] = 30.01
            (self.root / 'timing-round-1-sse.json').write_text(json.dumps(record))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('timing', log('timing', self.report('timing')), self.root)

    def test_optional_client_refuses_retry_changed_request_missing_error_or_process_failure(self):
        self.set_client()
        original = self.client_stream_fixture(1)
        for kind in ('retry', 'duplicate', 'changed-body', 'missing-error', 'pid', 'process-exit', 'process-timeout', 'process-identity'):
            self.client_stream_fixture(1)
            record = copy.deepcopy(original)
            if kind == 'retry': record['client']['attempt_count'] = 2
            elif kind == 'duplicate': record['client']['request_count'] = 2
            elif kind == 'changed-body': record['client']['request']['max_tokens'] = 4096
            elif kind == 'missing-error': record['client'].pop('error')
            elif kind == 'pid': record['client']['pid'] += 1
            else:
                path = self.root / 'timing-round-1-client-process.json'
                process = common.read_json(path)
                if kind == 'process-exit': process['exit_code'] = 1
                elif kind == 'process-timeout': process['timed_out'] = True
                else: process['identity']['bundle_sha256'] = 'e' * 64
                path.write_text(json.dumps(process))
            (self.root / 'timing-round-1-sse.json').write_text(json.dumps(record))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('timing', log('timing', self.report('timing')), self.root)

    def test_optional_client_timing_comes_from_wire_evidence_and_api_usage(self):
        self.set_client()
        report = self.report('timing')
        report['metrics'][0]['ttfc_seconds'] = .5
        with self.assertRaises(ValueError):
            runner.probe_evidence('timing', log('timing', report), self.root)
        report = self.report('timing')
        report['metrics'][0]['done_seconds'] = 5.
        report['metrics'][0]['completion_tokens_per_second'] = self.references[0]['usage']['completion_tokens'] / 5.
        with self.assertRaises(ValueError):
            runner.probe_evidence('timing', log('timing', report), self.root)

    def test_optional_client_closeout_hashes_all_request_process_and_log_evidence(self):
        self.set_client()
        names = closeout.closed_evidence_names(self.root)
        for index in range(1, 4):
            for suffix in ('-client-request.json', '-client-process.json', '-client.log', '-sse.json'):
                self.assertIn('timing-round-' + str(index) + suffix, names)
        for name in names:
            path = self.root / name
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{}\n')
        hashes = closeout.closed_evidence_hashes(self.root)
        self.assertIn('timing-round-3-client-process.json', hashes)
        (self.root / 'timing-round-2-client.log').unlink()
        with self.assertRaises(ValueError):
            closeout.closed_evidence_hashes(self.root)

    def test_optional_client_closeout_checks_all_three_node_processes(self):
        self.set_client()
        proc, cgroups = self.root / 'proc', self.root / 'cgroups'
        proc.mkdir()
        cgroups.mkdir()
        properties = {'ActiveState': 'inactive', 'SubState': 'dead', 'MainPID': '0', 'ControlGroup': ''}
        reports = {mode: {'unit': f'hag-answers-{mode}-20261004-120000-{index:06}',
                         'child': {'pid': 100 + index, 'cgroup': str(cgroups / mode)},
                         'thermal': {'summary': {'pid': 200 + index}}}
                   for index, mode in enumerate(common.MODES)}
        reports['timing']['client_identity'] = self.client
        observed = closeout.cleanup_evidence(reports, proc, cgroups, lambda unit: properties, self.root)
        self.assertEqual(len(observed['recorded_pids_absent']), 11)
        self.assertEqual(observed['stages']['timing']['client_pids_absent'], [701, 702, 703])
        (proc / '702').mkdir()
        with self.assertRaisesRegex(ValueError, 'client PID'):
            closeout.cleanup_evidence(reports, proc, cgroups, lambda unit: properties, self.root)
        (proc / '702').rmdir()
        path = self.root / 'timing-round-1-client-process.json'
        receipt = common.read_json(path)
        receipt['exit_code'] = 1
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, 'successful bound exit'):
            closeout.cleanup_evidence(reports, proc, cgroups, lambda unit: properties, self.root)

    def test_all_reports_accept_their_exact_complete_reference(self):
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)

    def test_semantic_profile_accepts_three_answers_and_keeps_list_timing(self):
        self.set_profile('semantic-c128')
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)
            self.assertEqual(report['requests'], 4 if mode == 'timing' else 3)
            self.assertEqual(report['maximum_evaluated_positions'], 512 if mode == 'timing' else 384)
            self.assertEqual(report['evaluated_positions'], 200 if mode == 'timing' else 113)
        for key, wrong in [('fixture_ids', self.contract['fixture_ids']),
                           ('timing_fixture_id', 'arithmetic_7_plus_5')]:
            report = {**self.report('timing'), key: wrong}
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.probe_evidence('timing', log('timing', report), self.root)
        report = self.report('timing')
        report['warmup']['generation'] = copy.deepcopy(self.references[1])
        with self.assertRaises(ValueError):
            runner.probe_evidence('timing', log('timing', report), self.root)

    def test_backend_parity_accepts_wrong_answers_only_as_exact_parity_with_diagnostics(self):
        self.set_profile('backend-parity-c128')
        for index, final in ((1, '13'), (2, 'BLUE')):
            self.references[index].update(text=final, final_answer=final,
                                          quality=quality_assessment(final, self.fixtures[index]))
            self.assertEqual(self.references[index]['quality']['correctness'], 'fail')
        path = self.root / 'cpu-reference.json'
        payload = common.read_json(path)
        payload['answers'] = self.references
        path.write_text(json.dumps(payload))
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            self.assertEqual(runner.probe_evidence(mode, log(mode, report), self.root), report)
            self.assertEqual(report['qualification_kind'], 'backend_parity')
            self.assertFalse(report['require_semantic_match'])
            self.assertTrue(report['quality_is_diagnostic'])
            self.assertEqual(report['requests'], 4 if mode == 'timing' else 3)
            self.assertEqual(report['maximum_evaluated_positions'], 512 if mode == 'timing' else 384)

    def test_backend_annotations_are_recomputed_and_cannot_be_forged(self):
        self.set_profile('backend-parity-c128')
        path = self.root / 'cpu-reference.json'
        original = common.read_json(path)
        for changed_quality in ({'correctness': 'fail', 'format': 'pass'},
                                {'correctness': 'pass', 'format': 'fail'},
                                {'correctness': 'pass', 'format': True},
                                {'correctness': 'pass'},
                                {'correctness': 'pass', 'format': 'pass', 'qualified': True}):
            payload = copy.deepcopy(original)
            payload['answers'][1]['quality'] = changed_quality
            path.write_text(json.dumps(payload))
            with self.subTest(quality=changed_quality), self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)
        path.write_text(json.dumps(original))
        reference = copy.deepcopy(self.references[1])
        reference.pop('quality')
        with self.assertRaises(ValueError):
            runner.validate_answer(reference, self.fixtures[1], self.profile)

    def test_backend_parity_still_refuses_different_raw_answers_or_quality(self):
        self.set_profile('backend-parity-c128')
        for kind in ('raw', 'quality'):
            report = self.report('answers')
            observed = report['answers'][1]
            if kind == 'raw':
                observed.update(text='13', final_answer='13',
                                quality=quality_assessment('13', self.fixtures[1]))
                self.assertEqual(runner.validate_answer(observed, self.fixtures[1], self.profile), 20)
            else:
                observed['quality']['correctness'] = 'fail'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                runner.probe_evidence('answers', log('answers', report), self.root)

    def test_backend_profile_cannot_claim_strict_semantic_qualification(self):
        self.set_profile('backend-parity-c128')
        for mode in ('cpu-reference', 'answers', 'timing'):
            for key, wrong in (('qualification_kind', 'semantic_and_backend_parity'),
                               ('require_semantic_match', True), ('quality_is_diagnostic', False),
                               ('profile', 'semantic-c128')):
                report = {**self.report(mode), key: wrong}
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    runner.probe_evidence(mode, log(mode, report), self.root)
        contract_path = self.root / 'qualification-contract.json'
        original = common.read_json(contract_path)
        for change in ({'qualification_kind': 'semantic_and_backend_parity'}, {'require_semantic_match': True},
                       {'require_semantic_match': 0}, {'profile': 'semantic-c128'}):
            contract_path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)

    def test_backend_profile_keeps_eos_closed_thinking_and_nonempty_final_bounds(self):
        self.set_profile('backend-parity-c128')
        valid = self.references[1]
        for change in ({'finish_reason': 'length'}, {'text': ''}, {'evaluated_positions': 128},
                       {'text': '<think>unfinished'},
                       {'text': '<think></think>', 'thinking_present': True, 'final_answer': ''}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.validate_answer({**valid, **change}, self.fixtures[1], self.profile)
        too_long = answer('arithmetic_7_plus_5', '13', 49, 80)
        too_long['quality'] = quality_assessment('13', self.fixtures[1])
        with self.assertRaises(ValueError):
            runner.validate_answer(too_long, self.fixtures[1], self.profile)

    def test_arithmetic_equation_diagnostics_do_not_change_strict_profile_acceptance(self):
        self.set_profile('backend-parity-c128')
        observed = answer('arithmetic_7_plus_5', '7 + 5 = 12')
        observed['quality'] = quality_assessment(observed['final_answer'], self.fixtures[1])
        self.assertEqual(observed['quality'], {'correctness': 'pass', 'format': 'fail'})
        self.assertEqual(runner.validate_answer(observed, self.fixtures[1], self.profile), 50)
        observed.pop('quality')
        for strict in ('list-c128', 'semantic-c128'):
            with self.subTest(profile=strict), self.assertRaises(ValueError):
                runner.validate_answer(observed, self.fixtures[1], strict)
        with self.assertRaises(ValueError):
            runner.validate_answer(observed, self.fixtures[1], 'unknown-profile')

    def test_semantic_reference_count_order_and_profile_are_exact(self):
        self.set_profile('semantic-c128')
        path = self.root / 'cpu-reference.json'
        original = common.read_json(path)
        changes = [{'answers': self.references[:2]}, {'answers': self.references + self.references[:1]},
                   {'answers': self.references[::-1]}, {'answers': [self.references[0]] * 3},
                   {'profile': 'list-c128'}]
        for change in changes:
            path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)
        path.write_text(json.dumps(original))
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = {**self.report(mode), 'profile': 'list-c128'}
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                runner.probe_evidence(mode, log(mode, report), self.root)
        manifest = common.read_json(self.root / 'expected-manifest.json')
        self.manifest({**manifest, 'profile': 'list-c128'})
        with self.assertRaises(ValueError):
            runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)

    def test_semantic_native_stage_requires_all_answers_raw_parity_and_audits(self):
        self.set_profile('semantic-c128')
        for key, value in [('requests', 1), ('maximum_evaluated_positions', 128),
                           ('fixture_ids', self.contract['fixture_ids'][::-1]),
                           ('evaluated_positions', 112)]:
            report = {**self.report('answers'), key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.probe_evidence('answers', log('answers', report), self.root)
        for index in range(3):
            for change in ('period', 'missing_answer', 'missing_audit', 'wrong_count', 'poison'):
                report = self.report('answers')
                if change == 'period':
                    # All optional periods are semantically valid, but native raw parity is exact.
                    report['answers'][index]['text'] += '.'
                    report['answers'][index]['final_answer'] += '.'
                elif change == 'missing_answer':
                    report['answers'].pop(index)
                elif change == 'missing_audit':
                    report['audit_events'].pop(index)
                elif change == 'wrong_count':
                    report['audit_events'][index]['evaluated_positions'] += 1
                else:
                    report['audit_events'][index]['poisoned'] = True
                with self.subTest(index=index, change=change), self.assertRaises(ValueError):
                    runner.probe_evidence('answers', log('answers', report), self.root)
        report = self.report('answers')
        report['audit_events'].append({**report['audit_events'][-1], 'request_sequence': 4})
        with self.assertRaises(ValueError):
            runner.probe_evidence('answers', log('answers', report), self.root)

    def test_arithmetic_and_recall_accept_only_standalone_preregistered_values(self):
        self.set_profile('semantic-c128')
        cases = [('arithmetic_7_plus_5', ('12', '12.'),
                  ('012', '+12', '12.0', 'twelve', 'The answer is 12.', '12..', '**12**', '13')),
                 ('recall_mango', ('MANGO', 'mango', 'MaNgO.'),
                  ('MANGO..', 'The word is MANGO.', 'MANGO is correct', '**MANGO**', '"MANGO"', 'Not MANGO', 'BLUE'))]
        for fixture_id, accepted, refused in cases:
            fixture = next(item for item in self.fixtures if item['id'] == fixture_id)
            for raw in accepted:
                with self.subTest(fixture=fixture_id, raw=raw):
                    self.assertEqual(runner.validate_answer(answer(fixture_id, raw), fixture), 50)
            for raw in refused:
                with self.subTest(fixture=fixture_id, raw=raw), self.assertRaises(ValueError):
                    runner.validate_answer(answer(fixture_id, raw), fixture)
            valid = answer(fixture_id, accepted[0])
            valid.update(text='<think>brief</think>\n' + accepted[0], thinking_present=True)
            self.assertEqual(runner.validate_answer(valid, fixture), 50)
            for change in ({'finish_reason': 'length'}, {'text': '<think>' + accepted[0]},
                           {'text': '<think></think>'}, {'thinking_present': 1}):
                with self.subTest(fixture=fixture_id, change=change), self.assertRaises(ValueError):
                    runner.validate_answer({**valid, **change}, fixture)
            changed = copy.deepcopy(fixture)
            changed['expected_semantic']['allow_terminal_period'] = 1
            with self.assertRaises(ValueError):
                runner.validate_answer(answer(fixture_id, accepted[0]), changed)

    def test_semantic_ceiling_does_not_expand_per_request_context(self):
        self.set_profile('semantic-c128')
        for fixture, reference in zip(self.fixtures, self.references):
            self.assertEqual(runner.validate_answer(answer(fixture['id'], reference['text'], 49, 79), fixture), 128)
            with self.assertRaises(ValueError):
                runner.validate_answer(answer(fixture['id'], reference['text'], 49, 80), fixture)

    def test_frozen_fixture_change_or_profile_switch_is_refused(self):
        self.set_profile('semantic-c128')
        path = self.root / 'answer-fixtures.json'
        original = common.read_json(path)
        for change in ('reorder', 'missing', 'semantic', 'list'):
            modified = copy.deepcopy(original)
            if change == 'reorder':
                modified['fixtures'].reverse()
            elif change == 'missing':
                modified['fixtures'].pop()
            elif change == 'semantic':
                modified['fixtures'][1]['expected_semantic']['value'] = 13
            else:
                modified = profiles.fixture_document('list-c128')
            path.write_text(json.dumps(modified))
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)

    def test_cpu_stage_is_explicit_and_never_selects_gpu_capture(self):
        self.assertNotIn('cpu-reference', common.GPU_MODES)
        command = child.probe_command('cpu-reference', self.root)
        self.assertEqual(command, ['/usr/bin/python3', str(self.root / 'answer-gate.py'),
                                   '--stage', 'cpu-reference', '--execute'])
        self.assertFalse(any('/dev/dri' in item for item in command))
        self.assertEqual(child.probe_command('self-check', profile='list-c128'), ['/usr/bin/true'])
        for mode in ('answers', 'timing'):
            self.assertIn(mode, common.GPU_MODES)
        for mode in ('boundary', 'tcp', 'execute'):
            with self.assertRaises(ValueError):
                child.probe_command(mode, profile='list-c128')

    def test_self_check_cannot_claim_answer_execution(self):
        self.assertEqual(runner.probe_evidence('self-check', 'true', self.root), {'self_check': True})
        with self.assertRaises(ValueError):
            runner.probe_evidence('self-check', log('answers', self.report('answers')), self.root)

    def test_prerequisites_bind_exact_source_checkpoint_and_watcher(self):
        gate_files = {'fixture': 'hash'}
        prior = {'mode': 'cpu-reference', 'qualified': True, 'failures': [],
                 'binary_sha256': 'binary', 'source_manifest_sha256': 'source',
                 'checkpoint_sha256': common.MODEL_SHA256, 'gate_files': gate_files, 'capture_session': None}
        runner.require_prior(prior, 'cpu-reference', 'binary', 'source', gate_files)
        scope = {'qualification_kind': 'semantic_and_backend_parity',
                 'require_semantic_match': True, 'quality_is_diagnostic': False}
        runner.require_prior({**prior, **scope, 'profile': 'semantic-c128'}, 'cpu-reference',
                             'binary', 'source', gate_files, profile='semantic-c128')
        with self.assertRaises(ValueError):
            runner.require_prior({**prior, 'profile': 'list-c128'}, 'cpu-reference',
                                 'binary', 'source', gate_files, profile='semantic-c128')
        backend = {**prior, 'profile': 'backend-parity-c128', 'qualification_kind': 'backend_parity',
                   'require_semantic_match': False, 'quality_is_diagnostic': True}
        runner.require_prior(backend, 'cpu-reference', 'binary', 'source', gate_files, profile='backend-parity-c128')
        for key, value in scope.items():
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.require_prior({**backend, key: value}, 'cpu-reference',
                                     'binary', 'source', gate_files, profile='backend-parity-c128')
        for key, bad in [('mode', 'answers'), ('qualified', False), ('failures', ['fault']),
                         ('binary_sha256', 'different'), ('source_manifest_sha256', 'different'),
                         ('checkpoint_sha256', 'different'), ('gate_files', {}), ('capture_session', {'pid': 1})]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.require_prior({**prior, key: bad}, 'cpu-reference', 'binary', 'source', gate_files)
        observed = {'git_head': 'a' * 40, 'binary_source_relation': 'not_verified_by_runner'}
        runner.require_prior({**prior, 'input_provenance': observed}, 'cpu-reference',
                             'binary', 'source', gate_files, input_provenance=observed)
        with self.assertRaises(ValueError):
            runner.require_prior({**prior, 'input_provenance': observed}, 'cpu-reference',
                                 'binary', 'source', gate_files, input_provenance={})
        session = {'pid': 123, 'armed_at': 'instant'}
        prior.update(mode='answers', capture_session=session)
        runner.require_prior(prior, 'answers', 'binary', 'source', gate_files, session)
        with self.assertRaises(ValueError):
            runner.require_prior(prior, 'answers', 'binary', 'source', gate_files, {**session, 'pid': 124})

    def test_unapproved_or_changed_contract_refuses_reports(self):
        path = self.root / 'qualification-contract.json'
        original = common.read_json(path)
        for change in [{'status': 'pending'}, {'capacity': 256}, {'max_new_tokens': 81},
                       {'maximum_prompt_tokens': 50}, {'cpu_quota_percent': 100}, {'thermal_cutoff_c': 91},
                       {'swap_max_bytes': 1}, {'process_memory_max_bytes': True},
                       {'stage_evaluation_ceilings': {'cpu-reference': 256, 'answers': 128, 'timing': 512}},
                       {'stage_request_counts': {'cpu-reference': 1, 'answers': 2, 'timing': 4}},
                       {'fixture_ids': ['recall', 'list_1_to_8']}]:
            path.write_text(json.dumps({**original, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', self.report('cpu-reference')), self.root)

    def test_complete_answer_requires_eos_and_actual_semantics(self):
        valid = self.references[0]
        for key, bad in [('finish_reason', 'length'), ('text', ''), ('final_answer', 'RED'),
                         ('thinking_present', 0), ('evaluated_positions', 49), ('evaluated_positions', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.validate_answer({**valid, key: bad}, self.fixtures[0])
        with self.assertRaises(ValueError):
            runner.validate_answer(answer('list_1_to_8', '1, 2, 3, 4, 5, 6, 7, 9'), self.fixtures[0])
        with self.assertRaises(ValueError):
            runner.validate_answer(answer('list_1_to_8', '01, 2, 3, 4, 5, 6, 7, 8'), self.fixtures[0])

    def test_thinking_must_close_and_leave_a_complete_correct_answer(self):
        valid = {**self.references[0], 'text': '<think>brief</think>\n1, 2, 3, 4, 5, 6, 7, 8', 'thinking_present': True}
        self.assertEqual(runner.validate_answer(valid, self.fixtures[0]), 50)
        final = self.references[0]['text']
        for raw in ('<think>' + final, '<think></think>', final + '</think>',
                    '<THINK>x</THINK>' + final, '<think bad>x</think>' + final,
                    '<think>x</think><think>y</think>' + final):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                runner.validate_answer({**valid, 'text': raw}, self.fixtures[0])

    def test_integer_list_requires_order_and_no_extra_or_negated_content(self):
        for text in ('1,2,3,4,5,6,7,8', '1, 2, 3, 4, 5, 6, 7, 8', '1,2,3,4,5,6,7,8.'):
            with self.subTest(text=text):
                self.assertEqual(runner.validate_answer(answer('list_1_to_8', text), self.fixtures[0]), 50)
        for text in ('1,2,3,4,5,6,7,8..', '01,2,3,4,5,6,7,8.', '1,2,3,4,5,6,7,9', '1,2,3,4,5,6,7', '1,2,3,4,5,6,8,7',
                     'Not 1,2,3,4,5,6,7,8', '1,2,3,4,5,6,7,8. Extra text.', '**1,2,3,4,5,6,7,8**'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                runner.validate_answer(answer('list_1_to_8', text), self.fixtures[0])
        broader = copy.deepcopy(self.fixtures[0])
        broader['expected_semantic']['values'].append(9)
        with self.assertRaises(ValueError):
            runner.validate_answer(answer('list_1_to_8', '1,2,3,4,5,6,7,8,9'), broader)

    def test_only_one_frozen_list_reference_and_request_are_admitted(self):
        path = self.root / 'cpu-reference.json'
        original = common.read_json(path)
        for answers in ([], original['answers'] * 2):
            path.write_text(json.dumps({**original, 'answers': answers}))
            report = self.report('cpu-reference')
            with self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', report), self.root)
        path.write_text(json.dumps(original))
        for mode in ('cpu-reference', 'answers'):
            for key, value in [('requests', 2), ('maximum_evaluated_positions', 256),
                               ('fixture_ids', ['recall', 'list_1_to_8'])]:
                report = {**self.report(mode), key: value}
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    runner.probe_evidence(mode, log(mode, report), self.root)

    def test_eos_position_accounting_does_not_reuse_length_minus_one(self):
        valid = answer('list_1_to_8', self.references[0]['text'], 49, 79)
        self.assertEqual(runner.validate_answer(valid, self.fixtures[0]), 128)
        with self.assertRaises(ValueError):
            runner.validate_answer({**valid, 'evaluated_positions': 127}, self.fixtures[0])
        for prompt, completion in [(50, 1), (49, 80), (0, 1), (1, 0), (True, 2), (2, 1.0)]:
            with self.subTest(prompt=prompt, completion=completion), self.assertRaises(ValueError):
                runner.validate_answer(answer('list_1_to_8', self.references[0]['text'], prompt, completion), self.fixtures[0])

    def test_reference_payload_and_hash_are_both_bound(self):
        path = self.root / 'cpu-reference.json'
        original = common.read_json(path)
        for key, bad in [('binary_sha256', 'other'), ('source_manifest_sha256', 'other'),
                         ('checkpoint_sha256', 'other'), ('fixture_sha256', 'other'), ('gate_files', {})]:
            changed = copy.deepcopy(original)
            changed['identities'][key] = bad
            path.write_text(json.dumps(changed))
            report = self.report('cpu-reference')
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.probe_evidence('cpu-reference', log('cpu-reference', report), self.root)
        path.write_text(json.dumps(original))
        report = self.report('answers')
        report['reference_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            runner.probe_evidence('answers', log('answers', report), self.root)

    def test_report_counts_booleans_and_artifact_hashes_are_strict(self):
        for mode in ('cpu-reference', 'answers', 'timing'):
            report = self.report(mode)
            for key, value in report.items():
                if type(value) is int:
                    bad = float(value)
                elif type(value) is bool:
                    bad = int(value)
                elif key.endswith('sha256'):
                    bad = '0' * 64
                else:
                    continue
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    runner.probe_evidence(mode, log(mode, {**report, key: bad}), self.root)

    def test_native_reports_require_each_healthy_ordered_audit(self):
        for mode in ('answers', 'timing'):
            for key, bad in [('request_sequence', 2), ('request_sequence', True),
                             ('evaluated_positions', 49), ('outcome', 'cancelled'),
                             ('reset_ok', False), ('poisoned', True)]:
                report = self.report(mode)
                report['audit_events'][0][key] = bad
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    runner.probe_evidence(mode, log(mode, report), self.root)
            report = self.report(mode)
            report['audit_events'].pop()
            with self.assertRaises(ValueError):
                runner.probe_evidence(mode, log(mode, report), self.root)

    def test_timing_uses_matched_api_usage_not_content_chunk_count(self):
        report = self.report('timing')
        report['metrics'][0]['content_chunks'] = 1
        self.assertEqual(runner.probe_evidence('timing', log('timing', report), self.root), report)
        for key, bad in [('completion_tokens_per_second', .25), ('usage_source', 'chunks'),
                         ('completion_tokens', 1), ('prompt_tokens', 31), ('round', 0)]:
            report = self.report('timing')
            report['metrics'][0][key] = bad
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.probe_evidence('timing', log('timing', report), self.root)

    def test_timing_is_finite_ordered_bounded_and_descriptive(self):
        for key in ('ttfc_seconds', 'last_content_seconds', 'done_seconds', 'completion_tokens_per_second'):
            for bad in (float('nan'), float('inf'), -1, True, None):
                report = self.report('timing')
                report['metrics'][0][key] = bad
                with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                    runner.probe_evidence('timing', log('timing', report), self.root)
        for change in [{'ttfc_seconds': 4}, {'last_content_seconds': 5}, {'done_seconds': 0}, {'done_seconds': 181}]:
            report = self.report('timing')
            report['metrics'][0].update(change)
            with self.assertRaises(ValueError):
                runner.probe_evidence('timing', log('timing', report), self.root)
        report = self.report('timing')
        report['warmup']['label'] = 'cold_disk_cache'
        with self.assertRaises(ValueError):
            runner.probe_evidence('timing', log('timing', report), self.root)

    def test_report_and_pass_marker_are_single_and_stage_specific(self):
        for mode in ('cpu-reference', 'answers', 'timing'):
            text = log(mode, self.report(mode))
            for bad in ('', text + text, text.replace('qualified=true', 'qualified=false'),
                        text.replace('stage=' + mode, 'stage=unexpected')):
                with self.assertRaises(ValueError):
                    runner.probe_evidence(mode, bad, self.root)

    def test_gate_inventory_freezes_runner_fixtures_and_closeout_helpers(self):
        hashes = {}
        contents = {}
        for name in common.GATE_FILES:
            if name not in ('qualification-contract.json', 'answer-fixtures.json'):
                (self.root / name).write_text(name)
            contents[name] = (self.root / name).read_bytes()
            hashes[name] = common.digest(self.root / name)
        fixtures = {name: hashes[name] for name in ('answer-fixtures.json', 'cpu-config.json', 'hag-config.json')}
        manifest = {'profile': self.profile, 'gate_files': hashes, 'answer_fixtures': fixtures}
        self.manifest(manifest)
        self.assertEqual(common.gate_identity(self.root), hashes)
        self.assertIn('closeout.py', hashes)
        self.assertIn('run-config.json', hashes)
        self.assertIn('profiles.py', hashes)
        self.assertIn('answer_quality.py', hashes)
        self.assertNotIn('bundle.py', hashes)
        self.assertNotIn('freeze.py', hashes)
        for name in common.GATE_FILES:
            (self.root / name).write_text('changed')
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.gate_identity(self.root)
            (self.root / name).write_bytes(contents[name])
        self.manifest({**manifest, 'profile': 'semantic-c128'})
        with self.assertRaises(ValueError):
            common.gate_identity(self.root)

    def test_actual_two_gib_limits_and_trace_fail_closed(self):
        self.assertEqual(common.EXPECTED_LIMITS['memory.max'], str(2 * common.GIB))
        self.assertEqual(common.EXPECTED_LIMITS['cpu.max'], '75000 100000')
        common.verify_limits(common.EXPECTED_LIMITS, 'NoNewPrivs:\t1\n', '0')
        for quota in ('100000 100000', '200000 100000', 'max 100000'):
            with self.subTest(quota=quota), self.assertRaises(ValueError):
                common.verify_limits({**common.EXPECTED_LIMITS, 'cpu.max': quota}, 'NoNewPrivs:\t1\n', '0')
        for key in common.EXPECTED_LIMITS:
            actual = {**common.EXPECTED_LIMITS, key: 'max'}
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.verify_limits(actual, 'NoNewPrivs:\t1\n', '0')
        for status, trace in [('NoNewPrivs:\t0\n', '0'), ('NoNewPrivs:\t1\n', '1'), ('NoNewPrivs:\t1\n', None)]:
            with self.assertRaises(ValueError):
                common.verify_limits(common.EXPECTED_LIMITS, status, trace)


    def test_systemd_deadline_and_cgroup_properties_are_explicit(self):
        command = runner.unit_command('fixture', 'cpu-reference')
        for value in ['MemoryMax=2G', 'MemorySwapMax=0', 'TasksMax=64', 'CPUQuota=75%',
                      'RuntimeMaxSec=195s', 'TimeoutStopSec=1s', 'KillMode=control-group',
                      'NoNewPrivileges=yes', 'OOMPolicy=stop', 'Restart=no']:
            self.assertIn('--property=' + value, command)
        self.assertIn('HAG_TRACE_SUBMISSION=0', command)
        self.assertEqual(command[-2:], [str(common.ROOT / 'gate-child.py'), 'cpu-reference'])
        with self.assertRaises(ValueError):
            runner.unit_command('fixture', 'execute')


    def test_process_memory_stops_at_1536_mib_before_hard_limit(self):
        self.assertEqual(common.STOP_CGROUP_BYTES, 1536 * 1024 ** 2)
        self.assertLess(common.STOP_CGROUP_BYTES, common.MEMORY_LIMIT)
        common.require_process_headroom({'memory_current_bytes': common.STOP_CGROUP_BYTES - 1})
        for value in [common.STOP_CGROUP_BYTES, common.MEMORY_LIMIT]:
            with self.assertRaises(ValueError):
                common.require_process_headroom({'memory_current_bytes': value})


    def test_host_headroom_and_invalid_meminfo(self):
        path = self.root / 'meminfo'
        path.write_text('MemAvailable: 8388608 kB\n')
        self.assertEqual(common.available_memory(path), 8 * common.GIB)
        common.require_headroom(common.available_memory(path))
        for minimum in [common.START_HEADROOM, common.STOP_HEADROOM]:
            common.require_headroom(minimum, minimum)
            with self.assertRaises(ValueError):
                common.require_headroom(minimum - 1, minimum)
        for text in ['MemAvailable: invalid kB\n', 'MemAvailable: 8 bytes\n', 'MemFree: 9 kB\n']:
            path.write_text(text)
            with self.assertRaises(ValueError):
                common.available_memory(path)


    def source_fixture(self):
        repo = self.root / 'repo'
        repo.mkdir()
        archive = self.root / 'build-source'
        archive.mkdir()
        for base in [repo, archive]:
            (base / 'crates').mkdir()
            (base / 'vendor/exec-hag').mkdir(parents=True)
            (base / 'Cargo.toml').write_bytes(b'fixture source')
            (base / 'Cargo.lock').write_bytes(b'fixture lock')
        sha = common.digest(repo / 'Cargo.toml')
        path = self.root / 'build-source-sha256.json'
        path.write_text(json.dumps({'Cargo.toml': sha, 'Cargo.lock': common.digest(repo / 'Cargo.lock')}))
        self.manifest({'source_files': 2, 'source_manifest_sha256': common.digest(path)})
        return repo, archive, path


    def test_sources_require_current_and_archive_bytes_to_match_frozen_manifest(self):
        repo, archive, path = self.source_fixture()
        self.assertEqual(common.source_identity(self.root, repo), common.digest(path))
        for base in [repo, archive]:
            (base / 'Cargo.toml').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                common.source_identity(self.root, repo)
            (base / 'Cargo.toml').write_bytes(b'fixture source')
            (base / 'Cargo.lock').write_bytes(b'fixture lock')
        path.write_text('{}')
        with self.assertRaises(ValueError):
            common.source_identity(self.root, repo)


    def test_source_manifest_refuses_count_drift_and_path_escape(self):
        repo, _, path = self.source_fixture()
        self.manifest({'source_files': 3, 'source_manifest_sha256': common.digest(path)})
        with self.assertRaises(ValueError):
            common.source_identity(self.root, repo)
        for name in ['../outside', '/absolute', 'a/../Cargo.toml', './Cargo.toml']:
            path.write_text(json.dumps({name: '0' * 64}))
            self.manifest({'source_files': 1, 'source_manifest_sha256': common.digest(path)})
            with self.subTest(name=name), self.assertRaises(ValueError):
                common.source_identity(self.root, repo)


    def test_dependencies_are_both_pinned(self):
        directory = self.root / 'dependencies'
        directory.mkdir()
        hashes = {}
        for name in ['thermal_guard.py', 'hag-crash-watch.py']:
            (directory / name).write_text(name)
            hashes[name] = common.digest(directory / name)
        self.manifest({'dependencies': hashes})
        self.assertEqual(common.dependency_identity(self.root), hashes)
        (directory / 'thermal_guard.py').write_text('changed')
        with self.assertRaises(ValueError):
            common.dependency_identity(self.root)


    def test_checkpoint_has_fixed_identity_without_large_fixture(self):
        path = self.root / 'small-model'
        path.write_bytes(b'fixture')
        self.manifest({'model_sha256': common.MODEL_SHA256, 'model_bytes': common.MODEL_BYTES})
        with self.assertRaises(ValueError):
            common.checkpoint_identity(self.root, path)
        self.manifest({'model_sha256': '0' * 64, 'model_bytes': 7})
        with self.assertRaises(ValueError):
            common.checkpoint_identity(self.root, path)
        # Exercise the successful byte/hash comparison with bounded substitutes.
        with patch.object(common, 'MODEL_BYTES', 7), patch.object(common, 'MODEL_SHA256', common.digest(path)):
            self.manifest({'model_sha256': common.MODEL_SHA256, 'model_bytes': 7})
            self.assertEqual(common.checkpoint_identity(self.root, path), common.MODEL_SHA256)


    def test_evidence_is_exclusive_and_bounded(self):
        path = self.root / 'evidence.json'
        common.write_json(path, {'first': True})
        with self.assertRaises(FileExistsError):
            common.write_json(path, {'second': True})
        self.assertEqual(common.read_json(path), {'first': True})
        with self.assertRaises(ValueError):
            common.read_json(path, limit=1)
        with self.assertRaises(ValueError):
            runner.bounded_text(path, limit=1)


    def test_cgroup_path_refuses_escape_relative_and_v1(self):
        self.assertEqual(common.cgroup_path('0::/user.slice/unit.service\n'), Path('/sys/fs/cgroup/user.slice/unit.service'))
        for record in ['0::/../outside\n', '1:memory:/foo\n', '0::/a\n0::/b\n', '0::relative\n']:
            with self.assertRaises(ValueError):
                common.cgroup_path(record)


    def test_memory_snapshot_and_every_fault_counter(self):
        values = {'memory.current': '100', 'memory.peak': '200', 'memory.swap.current': '0', 'memory.swap.peak': '0',
                  'memory.events': 'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n'}
        for name, value in values.items():
            (self.root / name).write_text(value)
        before = common.memory_snapshot(self.root)
        self.assertTrue(common.clean_memory(before, before))
        for counter in before['memory_events']:
            if counter == 'low':
                continue
            after = {**before, 'memory_events': {**before['memory_events'], counter: 1}}
            self.assertFalse(common.clean_memory(before, after))
        for key, value in [('swap_current_bytes', 1), ('swap_peak_bytes', 1), ('memory_peak_bytes', common.MEMORY_LIMIT + 1)]:
            self.assertFalse(common.clean_memory(before, {**before, key: value}))
        (self.root / 'memory.events').write_text('max 0\n')
        with self.assertRaises(ValueError):
            common.memory_snapshot(self.root)


    def test_capture_requires_fresh_running_root(self):
        self.assertEqual(runner.validate_capture(self.capture, self.now, self.root_status)['pid'], 123)
        for now in [self.now + timedelta(seconds=16), self.now - timedelta(seconds=1)]:
            with self.assertRaises(ValueError):
                runner.validate_capture(self.capture, now, self.root_status)
        for status in [self.root_status.replace('0\t0\t0\t0', '1000\t1000\t1000\t1000'), self.root_status.replace('S (sleeping)', 'Z (zombie)')]:
            with self.assertRaises(ValueError):
                runner.validate_capture(self.capture, self.now, status)
        with self.assertRaises(ValueError):
            runner.validate_capture(self.capture[:1], self.now, self.root_status)


    def test_capture_refuses_dump_error_stop_and_mixed_sessions(self):
        for event in ['CAPTURE', 'ERROR', 'STOPPED', 'ARMED', 'unexpected']:
            with self.assertRaises(ValueError):
                runner.validate_capture([*self.capture, {**self.capture[-1], 'event': event}], self.now, self.root_status)
        for key, value in [('pid', 456), ('pid', True), ('captured', 1), ('captured', False)]:
            with self.assertRaises(ValueError):
                runner.validate_capture([self.capture[0], {**self.capture[-1], key: value}], self.now, self.root_status)


    def test_thermal_evidence_requires_complete_finite_below_90_samples(self):
        samples, summary = thermal_fixture()
        encode = lambda records: '\n'.join(json.dumps(x) for x in records)
        self.assertEqual(runner.thermal_evidence(encode(samples), json.dumps(summary))['max_tctl_c'], 60)
        for bad in [90, float('nan'), float('inf'), None, -1, True]:
            changed = [samples[0], {**samples[1], 'tctl_c': bad}]
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                runner.thermal_evidence(encode(changed), json.dumps(summary))
        with self.assertRaises(ValueError):
            runner.thermal_evidence(encode(samples[1:]), json.dumps(summary))
        with self.assertRaises(ValueError):
            runner.thermal_evidence(encode([samples[0], {**samples[1], 'sensor_ok': False}]), json.dumps(summary))


    def test_thermal_summary_rejects_abort_residual_cleanup_or_missing_result(self):
        samples, summary = thermal_fixture()
        text = '\n'.join(json.dumps(x) for x in samples)
        for key, value in [('ok', False), ('aborted_for_heat', True), ('process_started', False),
                           ('exit_code', 1), ('exit_code', False), ('samples', 1),
                           ('max_tctl_c', 99), ('residual_group_cleaned', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.thermal_evidence(text, json.dumps({**summary, key: value}))
        for log in ['', json.dumps(summary) + '\n' + json.dumps(summary)]:
            with self.assertRaises(ValueError):
                runner.thermal_evidence(text, log)


    def test_new_include_and_nested_lock_cannot_escape_frozen_source_inventory(self):
        repo, _, path = self.source_fixture()
        directory = repo / 'crates/example/include'
        directory.mkdir(parents=True)
        for name in ['context.rs', 'Cargo.lock']:
            target = directory / name
            target.write_text('new compiled input')
            self.assertIn(target, common.compiled_source_paths(repo))
            with self.assertRaises(ValueError):
                common.source_identity(self.root, repo)
            target.unlink()


    def test_provenance_is_current_input_observation_not_inherited_qualification(self):
        observed = {'git_head': 'a' * 40, 'git_dirty': True,
                    'binary_source_relation': 'not_verified_by_runner',
                    'binary_sha256': TEST_BINARY_SHA, 'source_manifest_sha256': TEST_SOURCE_SHA}
        manifest = {'binaries': {'beholdd': TEST_BINARY_SHA}, 'source_manifest_sha256': TEST_SOURCE_SHA,
                    'source_files': 2, 'input_provenance': observed}
        self.manifest(manifest)
        self.assertEqual(common.provenance_identity(self.root), observed)
        self.assertFalse((self.root / 'ancestry').exists())
        for key, bad in [('git_head', 'invalid'), ('git_dirty', 1),
                         ('binary_source_relation', 'qualified'), ('binary_sha256', '3' * 64),
                         ('source_manifest_sha256', '4' * 64)]:
            self.manifest({**manifest, 'input_provenance': {**observed, key: bad}})
            with self.subTest(key=key), self.assertRaises(ValueError):
                common.provenance_identity(self.root)
        self.manifest({**manifest, 'input_provenance': {**observed, 'qualified': True}})
        with self.assertRaises(ValueError):
            common.provenance_identity(self.root)

    def test_closeout_requires_stage_order_but_cpu_precedes_watcher(self):
        reports = {mode: {'start': f'2026-10-04T12:{i * 2:02}:00+00:00',
                          'end': f'2026-10-04T12:{i * 2 + 1:02}:00+00:00'}
                   for i, mode in enumerate(common.MODES)}
        closeout.validate_order(reports)
        for earlier, later in zip(common.MODES, common.MODES[1:]):
            changed = copy.deepcopy(reports)
            changed[later]['start'] = changed[earlier]['start']
            with self.subTest(later=later), self.assertRaises(ValueError):
                closeout.validate_order(changed)

    def test_closeout_cpu_has_full_identity_without_native_watcher(self):
        session = {'pid': 123, 'armed_at': '2026-10-04T13:00:00+00:00'}
        begin = datetime.fromisoformat(session['armed_at'])
        end = begin + timedelta(minutes=10)
        identities = {'profile': self.profile, 'qualification_kind': self.contract['qualification_kind'],
                      'require_semantic_match': self.contract['require_semantic_match'],
                      'gate_files': {}, 'dependencies': {}, 'input_provenance': {}, 'source': 'source',
                      'checkpoint': common.MODEL_SHA256,
                      'binaries': {mode: 'binary' for mode in ('cpu-reference', 'answers', 'timing')}}
        memory = {'memory_current_bytes': 1, 'memory_peak_bytes': 2, 'swap_current_bytes': 0,
                  'swap_peak_bytes': 0, 'memory_events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        stub = Mock()
        stub.bounded_text.return_value = 'fixture'
        stub.probe_evidence.return_value = {'valid': True}
        stub.thermal_evidence.return_value = {'valid': True}
        for mode in ('cpu-reference', 'answers'):
            child_record = {'failure': None, 'exit_code': 0, 'memory_clean': True,
                            'limits': common.EXPECTED_LIMITS, 'no_new_privileges': True,
                            'gate_files': {}, 'dependencies': {}, 'input_provenance': {},
                            'memory_before': memory, 'memory_after': memory,
                            'binary_sha256': 'binary', 'source_manifest_sha256': 'source'}
            (self.root / (mode + '-child-result.json')).write_text(json.dumps(child_record))
            native = mode == 'answers'
            report = {'mode': mode, 'profile': self.profile, 'qualified': True, 'failures': [], 'capture_ok': True,
                      'qualification_kind': self.contract['qualification_kind'],
                      'require_semantic_match': self.contract['require_semantic_match'],
                      'quality_is_diagnostic': not self.contract['require_semantic_match'],
                      'exit_code': 0, 'journal_exit_code': 0, 'driver_messages': [],
                      'gate_files': {}, 'dependencies': {}, 'input_provenance': {},
                      'minimum_available_bytes': common.STOP_HEADROOM,
                      'cgroup_memory_peak_bytes': 2, 'child': child_record,
                      'binary_sha256': 'binary', 'source_manifest_sha256': 'source',
                      'checkpoint_sha256': common.MODEL_SHA256, 'capture_session': session if native else None,
                      'start': '2026-10-04T13:01:00+00:00' if native else '2026-10-04T12:00:00+00:00',
                      'end': '2026-10-04T13:02:00+00:00' if native else '2026-10-04T12:01:00+00:00',
                      'probe': {'valid': True}, 'thermal': {'valid': True}}
            closeout.validate_gate(mode, report, stub, session, begin, end, identities, self.root)
            client = {'kind': 'antimatter', 'commit': 'a' * 40, 'manifest_sha256': 'b' * 64,
                      'node_sha256': 'c' * 64, 'bundle_sha256': 'd' * 64}
            client_child = {**child_record, 'client_identity': client}
            client_report = {**report, 'client_identity': client, 'child': client_child}
            client_identities = {**identities, 'client_identity': client}
            (self.root / (mode + '-child-result.json')).write_text(json.dumps(client_child))
            closeout.validate_gate(mode, client_report, stub, session, begin, end, client_identities, self.root)
            with self.assertRaisesRegex(ValueError, 'client identity'):
                closeout.validate_gate(mode, report, stub, session, begin, end, client_identities, self.root)
            forged_child = {**client_child, 'client_identity': {**client, 'node_sha256': 'e' * 64}}
            (self.root / (mode + '-child-result.json')).write_text(json.dumps(forged_child))
            with self.assertRaisesRegex(ValueError, 'client identity'):
                closeout.validate_gate(mode, {**client_report, 'child': forged_child}, stub,
                                       session, begin, end, client_identities, self.root)
            (self.root / (mode + '-child-result.json')).write_text(json.dumps(child_record))
            with self.assertRaises(ValueError):
                closeout.validate_gate(mode, {**report, 'profile': 'semantic-c128'}, stub,
                                       session, begin, end, identities, self.root)
            for key, wrong in (('qualification_kind', 'backend_parity'), ('require_semantic_match', 1),
                               ('quality_is_diagnostic', True)):
                with self.subTest(key=key), self.assertRaises(ValueError):
                    closeout.validate_gate(mode, {**report, key: wrong}, stub, session, begin, end, identities, self.root)
            bad = {**report, 'capture_session': None if native else session}
            with self.assertRaises(ValueError):
                closeout.validate_gate(mode, bad, stub, session, begin, end, identities, self.root)
            if native:
                with self.assertRaises(ValueError):
                    closeout.validate_gate(mode, {**report, 'start': '2026-10-04T12:00:00+00:00'},
                                           stub, session, begin, end, identities, self.root)

    def test_duplicate_or_nonfinite_json_is_never_accepted(self):
        for value in ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":-Infinity}']:
            with self.assertRaises(ValueError):
                common.strict_json(value)

    def test_closeout_requires_only_list_responses_and_all_timing_artifacts(self):
        names = closeout.closed_evidence_names(self.root)
        for name in ('cpu-reference-list_1_to_8-response.json', 'answers-list_1_to_8-response.json',
                     'timing-warmup-response.json', 'timing-round-1-sse.json',
                     'timing-round-2-sse.json', 'timing-round-3-sse.json'):
            self.assertIn(name, names)
        self.assertFalse(any('recall-response' in name for name in names))
        self.assertIn('cpu-reference.json', names)

    def test_closeout_semantic_profile_hashes_all_six_responses_and_refuses_missing(self):
        self.set_profile('semantic-c128')
        names = closeout.closed_evidence_names(self.root)
        responses = [stage + '-' + fixture['id'] + '-response.json'
                     for stage in ('cpu-reference', 'answers') for fixture in self.fixtures]
        self.assertEqual(len(responses), 6)
        self.assertEqual({name for name in names if name.endswith('-response.json')},
                         set(responses + ['timing-warmup-response.json']))
        self.assertEqual(len(names), len(set(names)))
        for name in names:
            (self.root / name).write_text('{}')
        hashes = closeout.closed_evidence_hashes(self.root)
        self.assertEqual(set(hashes), set(names))
        for name in responses:
            (self.root / name).unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                closeout.closed_evidence_hashes(self.root)
            (self.root / name).write_text('{}')

    def test_closeout_backend_parity_cannot_claim_semantic_quality(self):
        self.set_profile('backend-parity-c128')
        for qualified in (True, False):
            self.assertEqual(closeout.qualification_scope(self.contract, qualified),
                             {'qualification_kind': 'backend_parity', 'require_semantic_match': False,
                              'quality_is_diagnostic': True, 'semantic_quality_qualified': False})
        names = closeout.closed_evidence_names(self.root)
        for stage in ('cpu-reference', 'answers'):
            for fixture in self.fixtures:
                self.assertIn(stage + '-' + fixture['id'] + '-response.json', names)
        for profile in ('list-c128', 'semantic-c128'):
            contract = inputs.profile_contract(profile)
            self.assertTrue(closeout.qualification_scope(contract, True)['semantic_quality_qualified'])
            self.assertFalse(closeout.qualification_scope(contract, False)['semantic_quality_qualified'])

    def test_failed_child_retains_thermal_cause_without_claiming_oom(self):
        samples, summary = thermal_fixture()
        samples[-1]['tctl_c'] = 90.375
        summary.update(ok=False, aborted_for_heat=True, abort_reason='critical_temperature',
                       exit_code=-15, max_tctl_c=90.375)
        data = '\n'.join(json.dumps(sample) for sample in samples)
        observation, failure = runner.thermal_result(data, json.dumps(summary))
        self.assertEqual(observation['max_tctl_c'], 90.375)
        self.assertEqual(observation['summary']['abort_reason'], 'critical_temperature')
        self.assertIn('thermal guard aborted', failure)
        memory = {'memory_current_bytes': 100, 'memory_peak_bytes': 200,
                  'swap_current_bytes': 0, 'swap_peak_bytes': 0,
                  'memory_events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        child_record = {'failure': None, 'exit_code': -15, 'memory_clean': True,
                        'memory_before': memory, 'memory_after': memory,
                        'minimum_available_bytes': common.START_HEADROOM,
                        'binary_sha256': None, 'source_manifest_sha256': None,
                        'limits': common.EXPECTED_LIMITS, 'no_new_privileges': True,
                        'dependencies': {}, 'gate_files': {}, 'input_provenance': {}}
        self.assertEqual(runner.child_failures(child_record), ['child process exited -15'])
        (self.root / 'cpu-reference-result.json').unlink()
        def launch(*args, **kwargs):
            common.write_json(self.root / 'self-check-child-result.json', child_record)
            (self.root / 'self-check-thermal.jsonl').write_text(data)
            kwargs['stdout'].write(json.dumps(summary) + '\n')
            kwargs['stdout'].flush()
            return Mock(returncode=1, poll=lambda: 1)
        with patch.multiple(runner, ROOT=self.root,
                            available_memory=lambda: common.START_HEADROOM,
                            dependency_identity=lambda: {}, gate_identity=lambda: {},
                            provenance_identity=lambda: {}, approved_contract=lambda root=None: self.contract), \
             patch.object(runner.subprocess, 'Popen', side_effect=launch), \
             patch.object(runner.subprocess, 'run', return_value=Mock(returncode=0)), \
             patch.object(runner.time, 'sleep'), patch('builtins.print'):
            self.assertEqual(runner.main(['self-check']), 1)
        result = common.read_json(self.root / 'self-check-result.json')
        self.assertEqual(result['thermal'], observation)
        self.assertEqual(result['cgroup_memory_peak_bytes'], 200)
        self.assertFalse(result['qualified'])
        self.assertTrue(any('thermal guard aborted' in reason for reason in result['failures']))
        self.assertFalse(any('oom' in reason.lower() for reason in result['failures']))

    def test_semantic_child_failure_with_clean_guard_is_not_a_thermal_failure(self):
        samples, summary = thermal_fixture()
        summary['exit_code'] = 1
        data = '\n'.join(json.dumps(sample) for sample in samples)
        observation, failure = runner.thermal_result(data, json.dumps(summary))
        self.assertEqual(failure, 'child process exited 1')
        self.assertTrue(observation['summary']['ok'])
        # Closeout still demands a successful guarded child, even with healthy telemetry.
        with self.assertRaisesRegex(ValueError, 'child process exited 1'):
            runner.thermal_evidence(data, json.dumps(summary))
        memory = {'memory_current_bytes': 100, 'memory_peak_bytes': 200,
                  'swap_current_bytes': 0, 'swap_peak_bytes': 0,
                  'memory_events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        child_record = {'failure': None, 'exit_code': 1, 'memory_clean': True,
                        'memory_before': memory, 'memory_after': memory,
                        'minimum_available_bytes': common.START_HEADROOM,
                        'binary_sha256': TEST_BINARY_SHA, 'source_manifest_sha256': TEST_SOURCE_SHA,
                        'limits': common.EXPECTED_LIMITS, 'no_new_privileges': True,
                        'dependencies': {}, 'gate_files': {}, 'input_provenance': {}}
        (self.root / 'cpu-reference-result.json').unlink()
        def launch(*args, **kwargs):
            common.write_json(self.root / 'cpu-reference-child-result.json', child_record)
            (self.root / 'cpu-reference-thermal.jsonl').write_text(data)
            kwargs['stdout'].write('ValueError: answer is not the standalone expected integer\n')
            kwargs['stdout'].write(json.dumps(summary) + '\n')
            kwargs['stdout'].flush()
            return Mock(returncode=1, poll=lambda: 1)
        with patch.multiple(runner, ROOT=self.root,
                            available_memory=lambda: common.START_HEADROOM,
                            dependency_identity=lambda: {}, gate_identity=lambda: {},
                            provenance_identity=lambda: {}, approved_contract=lambda root=None: self.contract,
                            binary_identity=lambda mode: TEST_BINARY_SHA, source_identity=lambda: TEST_SOURCE_SHA,
                            checkpoint_identity=lambda: common.MODEL_SHA256,
                            model_path=lambda root: self.root / 'answer-fixtures.json'), \
             patch.object(runner.subprocess, 'Popen', side_effect=launch), \
             patch.object(runner.subprocess, 'run', return_value=Mock(returncode=0)), \
             patch.object(runner.time, 'sleep'), patch('builtins.print'):
            self.assertEqual(runner.main(['cpu-reference']), 1)
        result = common.read_json(self.root / 'cpu-reference-result.json')
        self.assertFalse(result['qualified'])
        self.assertEqual(result['thermal'], observation)
        self.assertEqual(result['failures'].count('child process exited 1'), 1)
        self.assertTrue(any('probe evidence failed' in reason for reason in result['failures']))
        self.assertFalse(any(word in reason.lower() for reason in result['failures']
                             for word in ('thermal', 'heat', 'oom')))

    def test_nonzero_child_does_not_hide_invalid_thermal_telemetry(self):
        samples, summary = thermal_fixture()
        summary['exit_code'] = 1
        data = '\n'.join(map(json.dumps, samples))
        for change in ({'ok': False}, {'process_started': False}, {'residual_group_cleaned': True},
                       {'samples': 99}, {'max_tctl_c': 59}, {'exit_code': None}, {'exit_code': True}):
            with self.subTest(change=change):
                _, failure = runner.thermal_result(data, json.dumps({**summary, **change}))
                self.assertIn('thermal evidence failed', failure)
        for malformed in ('', json.dumps(summary) + '\n' + json.dumps(summary)):
            with self.subTest(log=malformed):
                _, failure = runner.thermal_result(data, malformed)
                self.assertIn('thermal evidence failed', failure)
        hot = copy.deepcopy(samples)
        hot[-1]['tctl_c'] = 90
        _, failure = runner.thermal_result('\n'.join(map(json.dumps, hot)), json.dumps(summary))
        self.assertIn('thermal evidence failed', failure)

    def test_nonzero_guarded_exit_cannot_qualify_with_clean_child_and_service_records(self):
        samples, summary = thermal_fixture()
        summary['exit_code'] = 1
        data = '\n'.join(map(json.dumps, samples))
        memory = {'memory_current_bytes': 100, 'memory_peak_bytes': 200,
                  'swap_current_bytes': 0, 'swap_peak_bytes': 0,
                  'memory_events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        child_record = {'failure': None, 'exit_code': 0, 'memory_clean': True,
                        'memory_before': memory, 'memory_after': memory,
                        'minimum_available_bytes': common.START_HEADROOM,
                        'binary_sha256': None, 'source_manifest_sha256': None,
                        'limits': common.EXPECTED_LIMITS, 'no_new_privileges': True,
                        'dependencies': {}, 'gate_files': {}, 'input_provenance': {}}
        def launch(*args, **kwargs):
            common.write_json(self.root / 'self-check-child-result.json', child_record)
            (self.root / 'self-check-thermal.jsonl').write_text(data)
            kwargs['stdout'].write(json.dumps(summary) + '\n')
            kwargs['stdout'].flush()
            return Mock(returncode=0, poll=lambda: 0)
        with patch.multiple(runner, ROOT=self.root,
                            available_memory=lambda: common.START_HEADROOM,
                            dependency_identity=lambda: {}, gate_identity=lambda: {},
                            provenance_identity=lambda: {}, approved_contract=lambda root=None: self.contract), \
             patch.object(runner.subprocess, 'Popen', side_effect=launch), \
             patch.object(runner.subprocess, 'run', return_value=Mock(returncode=0)), \
             patch.object(runner.time, 'sleep'), patch('builtins.print'):
            self.assertEqual(runner.main(['self-check']), 1)
        result = common.read_json(self.root / 'self-check-result.json')
        self.assertFalse(result['qualified'])
        self.assertEqual(result['failures'], ['child process exited 1'])

    def test_memory_failure_is_reported_from_counters_and_missing_thermal_is_not_success(self):
        memory = {'memory_current_bytes': 1, 'memory_peak_bytes': 2,
                  'swap_current_bytes': 0, 'swap_peak_bytes': 0,
                  'memory_events': {'high': 0, 'max': 0, 'oom': 0, 'oom_kill': 0}}
        changed = {**memory, 'memory_events': {**memory['memory_events'], 'oom': 1}}
        failures = runner.child_failures({'failure': None, 'exit_code': 0, 'memory_clean': False,
                                         'memory_before': memory, 'memory_after': changed})
        self.assertEqual(failures, ['cgroup memory counters recorded pressure, OOM or swap'])
        observation, failure = runner.thermal_result('', '')
        self.assertIsNone(observation['summary'])
        self.assertIsNotNone(failure)
        samples, summary = thermal_fixture()
        observation, failure = runner.thermal_result('\n'.join(map(json.dumps, samples)), json.dumps(summary))
        self.assertIsNone(failure)
        self.assertEqual(observation['max_tctl_c'], 60)

    def test_cleanup_requires_inactive_recorded_units_absent_pids_and_gone_cgroups(self):
        proc = self.root / 'proc'
        cgroups = self.root / 'cgroups'
        proc.mkdir()
        cgroups.mkdir()
        properties = {'LoadState': 'not-found', 'ActiveState': 'inactive', 'SubState': 'dead',
                      'MainPID': '0', 'ControlGroup': ''}
        reports = {mode: {'unit': f'hag-answers-{mode}-20261004-120000-{index:06}',
                         'child': {'pid': 100 + index, 'cgroup': str(cgroups / mode)},
                         'thermal': {'summary': {'pid': 200 + index}}}
                   for index, mode in enumerate(common.MODES)}
        read = lambda unit: properties
        clean = closeout.cleanup_evidence(reports, proc, cgroups, read)
        self.assertTrue(clean['all_units_inactive'])
        self.assertEqual(len(clean['recorded_pids_absent']), 8)
        for kind in ('pid', 'cgroup', 'active', 'unit', 'escape'):
            changed = copy.deepcopy(reports)
            reader = read
            path = None
            if kind == 'pid':
                path = proc / '100'
                path.mkdir()
            elif kind == 'cgroup':
                path = cgroups / 'self-check'
                path.mkdir()
            elif kind == 'active':
                reader = lambda unit: {**properties, 'ActiveState': 'active', 'MainPID': '100'}
            elif kind == 'unit':
                changed['self-check']['unit'] = 'unrelated.service'
            else:
                changed['self-check']['child']['cgroup'] = str(cgroups / '../outside')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                closeout.cleanup_evidence(changed, proc, cgroups, reader)
            if path:
                path.rmdir()



if __name__ == '__main__':
    unittest.main()
