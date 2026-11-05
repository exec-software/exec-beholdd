"""Hardware-free preparation and frozen-input regression tests."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    from . import gate_common as common
    from . import inputs
    from .profiles import fixture_document, fixture_profile, profile_spec
    from .antimatter import SUPPORTED_COMMIT, GROUNDED_COMMIT
except ImportError:
    import gate_common as common
    import inputs
    from profiles import fixture_document, fixture_profile, profile_spec
    from antimatter import SUPPORTED_COMMIT, GROUNDED_COMMIT


class PrepareTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='hag preparation with spaces ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / 'source repo'
        self.package = self.root / 'gate package'
        self.repo.mkdir()
        self.package.mkdir()
        self.output = self.root / 'new evidence'
        self.binary = self.root / 'supplied binary'
        self.binary.write_bytes(b'#!/bin/sh\nexit 99\n')
        self.binary.chmod(0o700)
        self.model = self.root / 'small checkpoint.gguf'
        self.model.write_bytes(b'fixture checkpoint bytes\n')
        self.device = Path('/dev/dri/renderD999')
        for name in ('Cargo.toml', 'Cargo.lock', 'crates/backend/src/lib.rs',
                     'vendor/exec-hag/src/lib.rs', 'tools/thermal_guard.py', 'tools/hag_crash_watch.py'):
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture ' + name + '\n')
        for name in common.GATE_FILES:
            if name not in inputs.GENERATED_FILES:
                (self.package / name).write_text('fixture ' + name + '\n')
        self.default_fixture_bytes = Path(inputs.__file__).with_name('answer-fixtures.json').read_bytes()
        (self.package / 'answer-fixtures.json').write_bytes(self.default_fixture_bytes)
        patches = (
            patch.object(inputs, 'PACKAGE_ROOT', self.package),
            patch.object(common, 'MODEL_BYTES', self.model.stat().st_size),
            patch.object(common, 'MODEL_SHA256', hashlib.sha256(self.model.read_bytes()).hexdigest()),
            patch.object(inputs, 'git_observation', return_value={'git_head': 'a' * 40, 'git_dirty': True}),
        )
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def prepare(self, profile='list-c128'):
        return inputs.prepare(self.repo, self.binary, self.model, self.device, self.output, profile=profile)

    def generated_inspector(self):
        path = self.root / 'native prompt inspector'
        header = bytearray(64)
        header[:6] = b'\x7fELF\x02\x01'
        header[18:20] = (62).to_bytes(2, 'little')
        path.write_bytes(header)
        path.chmod(0o700)
        return path

    def generated_prepare(self, inspector):
        with patch.object(inputs, 'prepare_client', return_value={'kind': 'antimatter', 'manifest_sha256': 'c' * 64}):
            return inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                profile='antimatter-generated-history-c2048', prompt_inspector=inspector,
                antimatter_repo=self.repo, antimatter_commit=GROUNDED_COMMIT,
                node=self.binary, esbuild=self.binary)

    def test_generated_preparation_freezes_inspector_without_executing_it(self):
        inspector = self.generated_inspector()
        with patch.object(inputs.subprocess, 'run') as run:
            manifest = self.generated_prepare(inspector)
        run.assert_not_called()
        path = self.output / 'dependencies/prompt-inspect'
        self.assertEqual(path.read_bytes(), inspector.read_bytes())
        self.assertEqual(path.stat().st_mode & 0o777, 0o700)
        self.assertEqual(set(manifest['dependencies']), {'thermal_guard.py', 'hag-crash-watch.py', 'prompt-inspect'})
        self.assertEqual(common.dependency_identity(self.output), manifest['dependencies'])
        self.assertEqual(set(common.run_config(self.output)), {'schema', 'repo', 'model', 'device', 'binary_input'})
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        inspector.unlink()
        self.assertEqual(common.dependency_identity(self.output), manifest['dependencies'])

    def test_inspector_is_required_only_for_generated_profile_before_checkpoint(self):
        for profile, inspector in (('antimatter-generated-history-c2048', None),
                                   ('list-c128', self.binary), ('antimatter-project-c2048', self.binary)):
            with self.subTest(profile=profile), patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                               profile=profile, prompt_inspector=inspector)
            checkpoint.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_inspector_refuses_wrapper_wrong_arch_symlink_and_nonexecutable(self):
        inspector = self.generated_inspector()
        alias = self.root / 'inspector alias'
        alias.symlink_to(inspector)
        for path in (self.binary, alias):
            with self.subTest(path=path), patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                self.generated_prepare(path)
            checkpoint.assert_not_called()
        data = bytearray(inspector.read_bytes())
        data[18:20] = (183).to_bytes(2, 'little')
        inspector.write_bytes(data)
        with self.assertRaises(ValueError):
            self.generated_prepare(inspector)
        data[18:20] = (62).to_bytes(2, 'little')
        inspector.write_bytes(data)
        inspector.chmod(0o600)
        with self.assertRaises(ValueError):
            self.generated_prepare(inspector)
        self.assertFalse(self.output.exists())

    def test_inspector_changes_are_detected_on_each_dependency_identity(self):
        inspector = self.generated_inspector()
        self.generated_prepare(inspector)
        path = self.output / 'dependencies/prompt-inspect'
        original = path.read_bytes()
        for index in (0, 32):
            changed = bytearray(original)
            changed[index] ^= 1
            path.write_bytes(changed)
            with self.subTest(index=index), self.assertRaises(ValueError):
                common.dependency_identity(self.output)
            path.write_bytes(original)
        path.unlink()
        with self.assertRaises(ValueError):
            common.dependency_identity(self.output)

    def test_inspector_inventory_cannot_be_added_to_old_profile_or_dropped_from_generated(self):
        inspector = self.generated_inspector()
        self.generated_prepare(inspector)
        path = self.output / 'expected-manifest.json'
        original = common.read_json(path)
        altered = {**original, 'dependencies': {k: v for k, v in original['dependencies'].items() if k != 'prompt-inspect'}}
        path.write_text(json.dumps(altered))
        with self.assertRaises(ValueError):
            common.dependency_identity(self.output)
        path.write_text(json.dumps({**original, 'profile': 'list-c128'}))
        with self.assertRaises(ValueError):
            common.dependency_identity(self.output)

    def test_portable_prepare_needs_no_ancestor_or_historical_file_count(self):
        with patch.object(inputs.subprocess, 'run') as process:
            manifest = self.prepare()
        process.assert_not_called()
        self.assertEqual(manifest['source_files'], 4)
        self.assertNotIn('ancestry', manifest)
        self.assertFalse((self.output / 'ancestry').exists())
        self.assertFalse((self.output / self.model.name).exists())
        self.assertEqual(common.repo_path(self.output), self.repo)
        self.assertEqual(common.model_path(self.output), self.model)
        self.assertEqual(common.source_identity(self.output), manifest['source_manifest_sha256'])
        self.assertEqual(common.binary_identity('answers', self.output), manifest['binaries']['beholdd'])
        self.assertEqual(common.checkpoint_identity(self.output), common.MODEL_SHA256)
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        self.assertEqual(common.dependency_identity(self.output), manifest['dependencies'])
        provenance = common.provenance_identity(self.output)
        self.assertEqual(provenance['binary_source_relation'], 'not_verified_by_runner')
        self.assertEqual(provenance['git_head'], 'a' * 40)
        self.assertIs(provenance['git_dirty'], True)
        self.assertEqual(set(manifest['gate_files']), set(common.GATE_FILES))
        self.assertEqual(len(manifest['gate_files']), len(common.GATE_FILES))
        self.assertNotIn('client', manifest)
        self.assertEqual(common.approved_contract(self.output)['stream_client'], 'python-http')
        self.assertEqual(manifest['profile'], 'list-c128')
        self.assertEqual((self.output / 'beholdd').read_bytes(), self.binary.read_bytes())
        self.assertEqual((self.output / 'beholdd').stat().st_mode & 0o777, 0o700)

    def test_grounded_profile_freezes_required_client_and_distinct_workload_without_safety_drift(self):
        client = {'kind': 'antimatter', 'manifest_sha256': 'c' * 64}
        with patch.object(inputs, 'prepare_client', return_value=client) as prepare_client:
            manifest = inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                profile='antimatter-grounded-c512', antimatter_repo=self.repo, antimatter_commit=GROUNDED_COMMIT,
                node=self.binary, esbuild=self.binary)
        prepare_client.assert_called_once_with(self.repo, GROUNDED_COMMIT, self.binary, self.binary, self.output)
        self.assertEqual(manifest['client'], client)
        contract = common.approved_contract(self.output)
        self.assertEqual(contract['capacity'], 512)
        self.assertEqual(contract['exact_prompt_tokens'], 93)
        self.assertEqual(contract['max_new_tokens'], 128)
        self.assertEqual(contract['stage_request_counts'], {'cpu-reference': 1, 'answers': 1, 'timing': 2})
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 220, 'answers': 220, 'timing': 440})
        self.assertFalse(contract['require_semantic_match'])
        self.assertFalse(contract['live_cancellation_qualified'])
        prior = inputs.profile_contract('backend-parity-c128', 'antimatter')
        for name in ('process_memory_max_bytes', 'process_early_stop_bytes', 'swap_max_bytes',
                     'cpu_quota_percent', 'tasks_max', 'no_new_privileges', 'host_start_headroom_bytes',
                     'host_stop_headroom_bytes', 'thermal_cutoff_c', 'thermal_poll_seconds',
                     'child_timeout_seconds', 'service_timeout_seconds', 'outer_timeout_seconds'):
            self.assertEqual(contract[name], prior[name])
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        changed = {**contract, 'exact_prompt_tokens': 92}
        (self.output / 'qualification-contract.json').write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            common.approved_contract(self.output)

    def test_grounded_profile_requires_its_client_before_any_payload_preparation(self):
        for options in ({}, {'antimatter_repo': self.repo, 'antimatter_commit': SUPPORTED_COMMIT,
                             'node': self.binary, 'esbuild': self.binary}):
            with patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                               profile='antimatter-grounded-c512', **options)
            checkpoint.assert_not_called()
            self.assertFalse(self.output.exists())
        with self.assertRaises(ValueError):
            inputs.profile_contract('antimatter-grounded-c512')

    def test_history_prepare_binds_seeded_fixture_and_preserves_project_safety_contract(self):
        profile = 'antimatter-history-c2048'
        client = {'kind': 'antimatter', 'manifest_sha256': 'c' * 64}
        with patch.object(inputs, 'prepare_client', return_value=client) as prepare_client:
            manifest = inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                profile=profile, antimatter_repo=self.repo, antimatter_commit=GROUNDED_COMMIT,
                node=self.binary, esbuild=self.binary)
        prepare_client.assert_called_once_with(self.repo, GROUNDED_COMMIT, self.binary, self.binary, self.output)
        self.assertEqual(manifest['client'], client)
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        document = common.read_json(self.output / 'answer-fixtures.json')
        self.assertEqual(document, fixture_document(profile))
        contract = common.approved_contract(self.output)
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 622, 'answers': 622, 'timing': 1244})
        self.assertEqual(contract['stage_request_counts'], {'cpu-reference': 1, 'answers': 1, 'timing': 2})
        self.assertEqual((contract['maximum_prompt_tokens'], contract['exact_prompt_tokens'], contract['overflow_max_tokens']),
                         (495, 495, 1555))
        self.assertFalse(contract['require_semantic_match'])
        self.assertFalse(contract['live_cancellation_qualified'])
        project = inputs.profile_contract('antimatter-project-c2048', 'antimatter')
        changed = {'profile', 'fixture_ids', 'stage_evaluation_ceilings', 'maximum_prompt_tokens',
                   'exact_prompt_tokens', 'overflow_max_tokens', 'scope', 'limitations'}
        self.assertEqual({k: v for k, v in contract.items() if k not in changed},
                         {k: v for k, v in project.items() if k not in changed})
        for backend in ('cpu', 'hag'):
            self.assertEqual(common.read_json(self.output / (backend + '-config.json')),
                             inputs.model_config(self.model, self.device, backend, 'antimatter-project-c2048'))
        self.assertEqual((self.package / 'answer-fixtures.json').read_bytes(), self.default_fixture_bytes)

    def test_history_contract_refuses_old_fixture_bounds_and_client_substitution(self):
        profile = 'antimatter-history-c2048'
        self.output.mkdir()
        original = inputs.profile_contract(profile, 'antimatter')
        common.write_json(self.output / 'answer-fixtures.json', fixture_document(profile))
        path = self.output / 'qualification-contract.json'
        for change in ({'exact_prompt_tokens': 467}, {'maximum_prompt_tokens': 495.0},
                       {'overflow_max_tokens': 1583}, {'stream_client': 'python-http'},
                       {'require_semantic_match': True}, {'client_request_timeout_seconds': 241},
                       {'stage_evaluation_ceilings': {'cpu-reference': 594, 'answers': 594, 'timing': 1188}},
                       {'profile': 'antimatter-project-c2048'}):
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    common.approved_contract(self.output)
        path.write_text(json.dumps(original))
        fixture_path = self.output / 'answer-fixtures.json'
        fixture_path.write_text(json.dumps(fixture_document('antimatter-project-c2048')))
        with self.assertRaises(ValueError):
            common.approved_contract(self.output)

    def test_history_requires_pinned_client_before_checkpoint_or_output(self):
        for options in ({}, {'antimatter_repo': self.repo, 'antimatter_commit': SUPPORTED_COMMIT,
                             'node': self.binary, 'esbuild': self.binary}):
            with patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                               profile='antimatter-history-c2048', **options)
            checkpoint.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_template_history_freezes_corrected_bounds_without_changing_wire_history_or_safety(self):
        profile = 'antimatter-history-template-c2048'
        client = {'kind': 'antimatter', 'manifest_sha256': 'c' * 64}
        with patch.object(inputs, 'prepare_client', return_value=client):
            manifest = inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                profile=profile, antimatter_repo=self.repo, antimatter_commit=GROUNDED_COMMIT,
                node=self.binary, esbuild=self.binary)
        contract = common.approved_contract(self.output)
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        self.assertEqual(contract['fixture_ids'], ['project_history_template_ready'])
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 618, 'answers': 618, 'timing': 1236})
        self.assertEqual((contract['exact_prompt_tokens'], contract['maximum_prompt_tokens'], contract['overflow_max_tokens']),
                         (491, 491, 1559))
        old = inputs.profile_contract('antimatter-history-c2048', 'antimatter')
        changed = {'profile', 'fixture_ids', 'stage_evaluation_ceilings', 'maximum_prompt_tokens',
                   'exact_prompt_tokens', 'overflow_max_tokens', 'scope', 'limitations'}
        self.assertEqual({k: v for k, v in contract.items() if k not in changed},
                         {k: v for k, v in old.items() if k not in changed})
        self.assertEqual(fixture_document(profile)['fixtures'][0]['messages'],
                         fixture_document('antimatter-history-c2048')['fixtures'][0]['messages'])
        self.assertEqual((self.package / 'answer-fixtures.json').read_bytes(), self.default_fixture_bytes)

    def test_phase_diagnostic_opt_in_is_frozen_without_changing_any_workload_bound(self):
        profile = 'antimatter-history-template-c2048'
        with patch.object(inputs, 'prepare_client', return_value={'kind': 'antimatter'}):
            manifest = inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                profile=profile, antimatter_repo=self.repo, antimatter_commit=GROUNDED_COMMIT,
                node=self.binary, esbuild=self.binary, runtime_phase_diagnostics=True)
        contract = common.approved_contract(self.output)
        self.assertEqual({key: value for key, value in contract.items() if key != 'runtime_phase_diagnostics'},
                         inputs.profile_contract(profile, 'antimatter'))
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        for mode in ('self-check', 'cpu-reference', 'answers', 'timing'):
            policy = common.phase_diagnostics(contract, mode)
            self.assertIs(policy['enabled'], mode == 'answers')
            self.assertEqual(policy['environment'], {'HAG_SERVING_PHASES': '1'} if mode == 'answers' else {})
            self.assertIs(policy['qualifying_evidence'], False)
        contract['runtime_phase_diagnostics']['stages'].append('timing')
        (self.output / 'qualification-contract.json').write_text(json.dumps(contract))
        with self.assertRaises(ValueError): common.approved_contract(self.output)
        with self.assertRaises(ValueError): common.gate_identity(self.output)

    def test_invalid_phase_option_refuses_before_checkpoint_io_or_output_creation(self):
        for profile, enabled in [('list-c128', True), ('lifecycle-c2048', True), ('list-c128', 1),
                                 ('list-c128', '1'), ('list-c128', None)]:
            with self.subTest(profile=profile, enabled=enabled), \
                 patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                               profile=profile, runtime_phase_diagnostics=enabled)
            checkpoint.assert_not_called()
            self.assertFalse(self.output.exists())

    def test_template_history_refuses_old_profile_usage_and_unpinned_client_before_io(self):
        profile = 'antimatter-history-template-c2048'
        for options in ({}, {'antimatter_repo': self.repo, 'antimatter_commit': SUPPORTED_COMMIT,
                             'node': self.binary, 'esbuild': self.binary}):
            with patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
                inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                               profile=profile, **options)
            checkpoint.assert_not_called()
            self.assertFalse(self.output.exists())
        self.output.mkdir()
        common.write_json(self.output / 'answer-fixtures.json', fixture_document(profile))
        path = self.output / 'qualification-contract.json'
        original = inputs.profile_contract(profile, 'antimatter')
        for changed in ({'profile': 'antimatter-history-c2048'}, {'exact_prompt_tokens': 495},
                        {'maximum_prompt_tokens': 491.0}, {'overflow_max_tokens': 1555},
                        {'stage_evaluation_ceilings': {'cpu-reference': 622, 'answers': 622, 'timing': 1244}},
                        {'client_request_timeout_seconds': 241}, {'require_semantic_match': True}):
            path.write_text(json.dumps({**original, **changed}))
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                common.approved_contract(self.output)

    def test_generated_configs_preserve_explicit_cpu_native_and_fixed_bounds(self):
        self.prepare()
        cpu = common.read_json(self.output / 'cpu-config.json')['models'][0]
        hag = common.read_json(self.output / 'hag-config.json')['models'][0]
        self.assertEqual(cpu['execution_backend'], 'cpu')
        self.assertNotIn('hag', cpu)
        self.assertEqual(cpu['path'], str(self.model))
        self.assertEqual(cpu['memory_budget_bytes'], 1024 ** 3)
        self.assertEqual(hag['execution_backend'], 'hag')
        self.assertEqual(hag['memory_budget_bytes'], 2 * 1024 ** 3)
        self.assertEqual(hag['hag'], {'device': str(self.device), 'memory_budget_bytes': 768 * 1024 ** 2})
        contract = common.approved_contract(self.output)
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 128, 'answers': 128, 'timing': 512})
        self.assertEqual(contract['stage_request_counts'], {'cpu-reference': 1, 'answers': 1, 'timing': 4})
        self.assertEqual(contract['process_early_stop_bytes'], 1536 * 1024 ** 2)
        self.assertEqual(contract['cpu_quota_percent'], 75)
        self.assertEqual(contract['thermal_poll_seconds'], 0.1)

    def test_default_fixture_is_copied_byte_for_byte(self):
        self.prepare()
        self.assertEqual((self.output / 'answer-fixtures.json').read_bytes(), self.default_fixture_bytes)
        self.assertEqual(fixture_profile(common.read_json(self.output / 'answer-fixtures.json')), 'list-c128')

    def test_lifecycle_prepare_has_three_stages_no_client_and_c2048_containment(self):
        manifest = self.prepare('lifecycle-c2048')
        contract = common.approved_contract(self.output)
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        self.assertNotIn('client', manifest)
        self.assertFalse((self.output / 'client').exists())
        self.assertEqual(common.read_json(self.output / 'answer-fixtures.json'), fixture_document('lifecycle-c2048'))
        self.assertEqual(contract['stage_request_counts'], {'cpu-reference': 1, 'answers': 5})
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 9, 'answers': 60})
        self.assertEqual(contract['qualification_kind'], 'tcp_lifecycle_and_backend_parity')
        for key in ('require_semantic_match', 'require_eos_stop', 'require_complete_final_answer'):
            self.assertIs(contract[key], False)
        self.assertIs(contract['live_cancellation_qualified'], True)
        self.assertEqual(contract['stream_client'], 'python-http')
        self.assertFalse(any(key.startswith(('timing_', 'client_')) for key in contract))
        project = inputs.profile_contract('antimatter-project-c2048', 'antimatter')
        for key in ('capacity', 'hag_charge_bytes', 'hag_budget_bytes', 'cpu_model_budget_bytes',
                    'runtime_combined_forecast_bytes', 'runtime_actual_metadata_forecast_bytes',
                    'private_cgroup_admission', 'crash_watcher_seconds', 'cpu_request_timeout_seconds',
                    'thermal_start_max_c', 'thermal_start_stable_seconds', 'process_memory_max_bytes',
                    'process_early_stop_bytes', 'swap_max_bytes', 'cpu_quota_percent', 'tasks_max',
                    'no_new_privileges', 'host_start_headroom_bytes', 'host_stop_headroom_bytes',
                    'thermal_cutoff_c', 'thermal_poll_seconds'):
            self.assertEqual(contract[key], project[key], key)
        cpu = common.read_json(self.output / 'cpu-config.json')['models'][0]
        hag = common.read_json(self.output / 'hag-config.json')['models'][0]
        self.assertEqual(cpu['memory_budget_bytes'], 1536 * 1024 ** 2)
        self.assertEqual(hag['hag']['memory_budget_bytes'], 1152 * 1024 ** 2)
        self.assertEqual(common.stage_deadlines('answers', self.output), (600, 615, 630))
        with self.assertRaises(ValueError):
            common.stage_deadlines('timing', self.output)

    def test_lifecycle_contract_refuses_scope_deadline_type_and_safety_drift(self):
        self.prepare('lifecycle-c2048')
        path = self.output / 'qualification-contract.json'
        original = common.read_json(path)
        changes = (
            {'profile': 'antimatter-project-c2048'}, {'qualification_kind': 'backend_parity'},
            {'require_eos_stop': True}, {'require_eos_stop': 0},
            {'require_complete_final_answer': True}, {'live_cancellation_qualified': 1},
            {'cancel_max_tokens': 17}, {'maximum_cancel_evaluated_positions': 25},
            {'stage_request_counts': {'cpu-reference': 1, 'answers': 4}},
            {'stage_evaluation_ceilings': {'cpu-reference': 9, 'answers': 60, 'timing': 0}},
            {'stage_deadlines': {**original['stage_deadlines'], 'timing': {'child': 600, 'service': 615, 'outer': 630}}},
            {'stage_deadlines': {**original['stage_deadlines'], 'answers': {'child': 601, 'service': 616, 'outer': 631}}},
            {'child_timeout_seconds': 600}, {'timing_measured_warm_requests': 0},
            {'client_seconds': 240}, {'stream_client': 'antimatter'},
            {'thermal_start_max_c': 61}, {'thermal_cutoff_c': 91}, {'thermal_poll_seconds': 1},
            {'no_new_privileges': False}, {'host_start_headroom_bytes': 4 * 1024 ** 3},
            {'crash_watcher_seconds': 600}, {'private_cgroup_admission': False},
        )
        for change in changes:
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    common.approved_contract(self.output)
        for key in ('require_eos_stop', 'require_complete_final_answer', 'live_cancellation_qualified'):
            changed = dict(original)
            del changed[key]
            path.write_text(json.dumps(changed))
            with self.assertRaises(ValueError):
                common.approved_contract(self.output)

    def test_lifecycle_refuses_optional_client_before_checkpoint_or_evidence_creation(self):
        with patch.object(inputs, 'checkpoint') as checkpoint, self.assertRaises(ValueError):
            inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                           profile='lifecycle-c2048', antimatter_repo=self.repo,
                           antimatter_commit=GROUNDED_COMMIT, node=self.binary, esbuild=self.binary)
        checkpoint.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_semantic_prepare_binds_generated_fixture_profile_and_bounds(self):
        manifest = self.prepare('semantic-c128')
        contract = common.approved_contract(self.output)
        document = common.read_json(self.output / 'answer-fixtures.json')
        self.assertEqual(document, fixture_document('semantic-c128'))
        self.assertEqual(manifest['profile'], contract['profile'])
        self.assertEqual(contract['fixture_ids'], ['list_1_to_8', 'arithmetic_7_plus_5', 'recall_mango'])
        self.assertEqual(contract['stage_request_counts'], {'cpu-reference': 3, 'answers': 3, 'timing': 4})
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 384, 'answers': 384, 'timing': 512})
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        # Selecting another profile never rewrites the package's list fixture.
        self.assertEqual((self.package / 'answer-fixtures.json').read_bytes(), self.default_fixture_bytes)

    def test_backend_parity_prepare_explicitly_binds_nonsemantic_strategy(self):
        manifest = self.prepare('backend-parity-c128')
        contract = common.approved_contract(self.output)
        document = common.read_json(self.output / 'answer-fixtures.json')
        self.assertEqual(document, fixture_document('backend-parity-c128'))
        self.assertEqual(document['profile'], manifest['profile'])
        self.assertEqual(contract['qualification_kind'], 'backend_parity')
        self.assertIs(contract['require_semantic_match'], False)
        self.assertNotIn('semantic_quality_qualified', contract)
        self.assertEqual(contract['stage_request_counts'], {'cpu-reference': 3, 'answers': 3, 'timing': 4})
        self.assertEqual(contract['stage_evaluation_ceilings'], {'cpu-reference': 384, 'answers': 384, 'timing': 512})
        self.assertEqual(common.gate_identity(self.output), manifest['gate_files'])
        self.assertEqual((self.package / 'answer-fixtures.json').read_bytes(), self.default_fixture_bytes)

    def test_contract_strategy_is_required_exactly_typed_and_fixed_by_profile(self):
        for profile in ('list-c128', 'semantic-c128', 'backend-parity-c128'):
            with self.subTest(profile=profile):
                directory = self.root / profile
                directory.mkdir()
                original = inputs.profile_contract(profile)
                common.write_json(directory / 'answer-fixtures.json', fixture_document(profile))
                path = directory / 'qualification-contract.json'
                common.write_json(path, original)
                self.assertEqual(common.approved_contract(directory), original)
                changes = (
                    {'require_semantic_match': not original['require_semantic_match']},
                    {'require_semantic_match': int(original['require_semantic_match'])},
                    {'require_semantic_match': str(original['require_semantic_match'])},
                    {'qualification_kind': ('backend_parity' if original['require_semantic_match']
                                            else 'semantic_and_backend_parity')},
                    {'qualification_kind': None},
                )
                for change in changes:
                    with self.subTest(change=change):
                        path.write_text(json.dumps({**original, **change}))
                        with self.assertRaisesRegex(ValueError, 'preregistered answer contract differs'):
                            common.approved_contract(directory)
                for missing in ('qualification_kind', 'require_semantic_match'):
                    document = dict(original)
                    del document[missing]
                    path.write_text(json.dumps(document))
                    with self.assertRaises(ValueError):
                        common.approved_contract(directory)

    def test_identical_prompts_cannot_cross_bind_backend_and_semantic_profiles(self):
        self.prepare('backend-parity-c128')
        path = self.output / 'answer-fixtures.json'
        path.write_text(json.dumps(fixture_document('semantic-c128')))
        with self.assertRaisesRegex(ValueError, 'fixture document differs'):
            common.approved_contract(self.output)
        path.write_text(json.dumps(fixture_document('backend-parity-c128')))
        contract_path = self.output / 'qualification-contract.json'
        contract_path.write_text(json.dumps(inputs.profile_contract('semantic-c128')))
        with self.assertRaisesRegex(ValueError, 'fixture document differs'):
            common.approved_contract(self.output)

    def test_unknown_profile_refuses_before_reading_checkpoint_or_creating_output(self):
        with patch.object(inputs, 'checkpoint') as checkpoint:
            with self.assertRaisesRegex(ValueError, 'unknown qualification profile'):
                self.prepare('arbitrary-fixtures')
        checkpoint.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_optional_client_arguments_are_all_or_none_and_parity_only_before_io(self):
        from_client = {'antimatter_repo': self.repo, 'antimatter_commit': SUPPORTED_COMMIT,
                       'node': self.binary, 'esbuild': self.binary}
        for missing in from_client:
            with self.subTest(missing=missing), patch.object(inputs, 'checkpoint') as checkpoint:
                arguments = {key: value for key, value in from_client.items() if key != missing}
                with self.assertRaisesRegex(ValueError, 'supplied together'):
                    inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                                   profile='backend-parity-c128', **arguments)
                checkpoint.assert_not_called()
        with patch.object(inputs, 'checkpoint') as checkpoint:
            with self.assertRaisesRegex(ValueError, 'requires an explicitly supported qualification profile'):
                inputs.prepare(self.repo, self.binary, self.model, self.device, self.output, **from_client)
            checkpoint.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_optional_client_freeze_is_recorded_and_never_changes_workload_bounds(self):
        entry = {'kind': 'antimatter', 'manifest_sha256': 'e' * 64}
        commit = SUPPORTED_COMMIT
        with patch.object(inputs, 'prepare_client', return_value=entry) as client:
            manifest = inputs.prepare(self.repo, self.binary, self.model, self.device, self.output,
                                      profile='backend-parity-c128', antimatter_repo=self.repo,
                                      antimatter_commit=commit, node=self.binary, esbuild=self.binary)
        client.assert_called_once_with(self.repo, commit, self.binary, self.binary, self.output)
        self.assertEqual(manifest['client'], entry)
        actual = common.approved_contract(self.output)
        self.assertEqual(actual, inputs.profile_contract('backend-parity-c128', 'antimatter'))
        default = inputs.profile_contract('backend-parity-c128')
        self.assertEqual({**actual, 'stream_client': 'python-http'}, default)

    def test_stream_client_is_explicit_typed_and_profile_scoped(self):
        self.prepare()
        path = self.output / 'qualification-contract.json'
        original = common.read_json(path)
        for value in ('antimatter', 'unknown', None, False):
            path.write_text(json.dumps({**original, 'stream_client': value}))
            with self.subTest(value=value), self.assertRaises(ValueError):
                common.approved_contract(self.output)
        changed = dict(original)
        del changed['stream_client']
        path.write_text(json.dumps(changed))
        with self.assertRaises(ValueError):
            common.approved_contract(self.output)

    def test_contract_cannot_relabel_fixtures_or_change_counts(self):
        self.prepare()
        path = self.output / 'qualification-contract.json'
        original = common.read_json(path)
        cases = (
            {'profile': 'semantic-c128'},
            {'stage_request_counts': {'cpu-reference': 3, 'answers': 3, 'timing': 4}},
            {'stage_evaluation_ceilings': {'cpu-reference': 384, 'answers': 384, 'timing': 512}},
            {'fixture_ids': ['arithmetic_7_plus_5']},
        )
        for change in cases:
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    common.approved_contract(self.output)

    def test_manifest_cannot_relabel_profile_or_omit_frozen_profile_implementation(self):
        manifest = self.prepare()
        path = self.output / 'expected-manifest.json'
        changed = json.loads(json.dumps(manifest))
        changed['profile'] = 'semantic-c128'
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'manifest qualification profile differs'):
            common.gate_identity(self.output)
        changed = json.loads(json.dumps(manifest))
        del changed['gate_files']['profiles.py']
        path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            common.gate_identity(self.output)

    def test_existing_directory_is_never_modified_even_when_empty(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_existing_and_dangling_output_symlinks_are_refused(self):
        for destination in (self.repo, self.root / 'missing'):
            with self.subTest(destination=destination):
                self.output.symlink_to(destination)
                with self.assertRaisesRegex(ValueError, 'already exists'):
                    self.prepare()
                self.assertTrue(self.output.is_symlink())
                self.output.unlink()

    def test_bad_checkpoint_size_or_hash_refuses_before_output(self):
        original = self.model.read_bytes()
        for content in (original + b'x', b'x' * len(original)):
            with self.subTest(content=content):
                self.model.write_bytes(content)
                with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                    self.prepare()
                self.assertFalse(self.output.exists())

    def test_nonexecutable_binary_refuses_before_output(self):
        self.binary.chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'not executable'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_missing_helper_refuses_before_output(self):
        (self.package / 'answer-gate.py').unlink()
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_source_symlink_refuses_before_output(self):
        source = self.repo / 'crates/backend/src/lib.rs'
        source.unlink()
        source.symlink_to(self.repo / 'vendor/exec-hag/src/lib.rs')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_source_copy_and_current_source_changes_are_both_detected(self):
        self.prepare()
        for base in (self.output / 'build-source', self.repo):
            with self.subTest(base=base):
                path = base / 'crates/backend/src/lib.rs'
                original = path.read_bytes()
                path.write_bytes(original + b'changed\n')
                with self.assertRaisesRegex(ValueError, 'frozen/current source differs'):
                    common.source_identity(self.output)
                path.write_bytes(original)

    def test_new_compiled_source_is_not_silently_ignored(self):
        self.prepare()
        (self.repo / 'crates/backend/src/new.rs').write_text('// new\n')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            common.source_identity(self.output)

    def test_compiled_chat_templates_are_frozen_and_tamper_evident(self):
        source = self.repo / 'crates/backend/src/template.jinja'
        source.write_text('exact template bytes')
        manifest = self.prepare()
        frozen = self.output / 'build-source/crates/backend/src/template.jinja'
        self.assertEqual(frozen.read_bytes(), source.read_bytes())
        self.assertEqual(common.source_identity(self.output), manifest['source_manifest_sha256'])
        for path in (source, frozen):
            original = path.read_bytes()
            path.write_bytes(original + b' changed')
            with self.assertRaisesRegex(ValueError, 'frozen/current source differs'):
                common.source_identity(self.output)
            path.write_bytes(original)
        (source.parent / 'extra.jinja').write_text('new template')
        with self.assertRaisesRegex(ValueError, 'inventory differs'):
            common.source_identity(self.output)

    def test_binary_config_dependency_and_checkpoint_tampering_are_detected(self):
        self.prepare()
        cases = (
            (self.output / 'beholdd', lambda: common.binary_identity('answers', self.output)),
            (self.output / 'run-config.json', lambda: common.gate_identity(self.output)),
            (self.output / 'dependencies/thermal_guard.py', lambda: common.dependency_identity(self.output)),
            (self.model, lambda: common.checkpoint_identity(self.output)),
        )
        for path, check in cases:
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b'changed\n')
                with self.assertRaises(ValueError):
                    check()
                path.write_bytes(original)

    def test_provenance_must_not_claim_a_verified_build_association(self):
        manifest = self.prepare()
        manifest['input_provenance']['binary_source_relation'] = 'verified'
        (self.output / 'expected-manifest.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, 'provenance differs'):
            common.provenance_identity(self.output)

    def test_run_config_rejects_relative_paths_and_boolean_schema(self):
        self.prepare()
        path = self.output / 'run-config.json'
        original = common.read_json(path)
        for change in ({'repo': '../source repo'}, {'schema': True}):
            with self.subTest(change=change):
                path.write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    common.run_config(self.output)

    def test_mid_prepare_source_mutation_leaves_unsealed_evidence(self):
        copy = inputs.copy_frozen
        source = self.repo / 'crates/backend/src/lib.rs'

        def mutate_after_copy(*arguments, **keywords):
            copy(*arguments, **keywords)
            if arguments[0] == source:
                source.write_text('// changed after snapshot\n')

        with patch.object(inputs, 'copy_frozen', side_effect=mutate_after_copy):
            with self.assertRaisesRegex(ValueError, 'source changed during preparation'):
                self.prepare()
        self.assertTrue(self.output.exists())
        self.assertFalse((self.output / 'expected-manifest.json').exists())
        self.assertNotEqual((self.output / 'build-source/crates/backend/src/lib.rs').read_bytes(), source.read_bytes())
        with self.assertRaisesRegex(ValueError, 'already exists'):
            self.prepare()

    def test_copy_rejects_changed_bytes_and_size_limit(self):
        destination = self.root / 'copy'
        with self.assertRaisesRegex(ValueError, 'changed while being frozen'):
            inputs.copy_frozen(self.binary, destination, '0' * 64, inputs.MAX_BINARY_BYTES)
        self.assertTrue(destination.exists())
        with self.assertRaisesRegex(ValueError, 'bounded regular'):
            inputs.copy_frozen(self.binary, self.root / 'too large', '0' * 64, 1)
        self.assertFalse((self.root / 'too large').exists())

    def test_compiled_source_snapshot_limits_are_enforced(self):
        with patch.object(common, 'MAX_SOURCE_TOTAL_BYTES', 1):
            with self.assertRaisesRegex(ValueError, 'byte limit'):
                self.prepare()
        self.assertFalse(self.output.exists())


class ProfileTests(unittest.TestCase):
    def test_exact_known_documents_and_independent_returned_values(self):
        for name in ('list-c128', 'semantic-c128', 'backend-parity-c128'):
            with self.subTest(name=name):
                self.assertEqual(fixture_profile(fixture_document(name)), name)
                document = fixture_document(name)
                document['fixtures'][0]['messages'][0]['content'] = 'changed'
                self.assertEqual(fixture_profile(fixture_document(name)), name)
                spec = profile_spec(name)
                spec['fixture_ids'].clear()
                self.assertTrue(profile_spec(name)['fixture_ids'])

    def test_profiles_change_only_declared_workload_not_resource_or_timing_limits(self):
        contracts = [inputs.profile_contract(profile) for profile in
                     ('list-c128', 'semantic-c128', 'backend-parity-c128')]
        for contract in contracts:
            for key in ('profile', 'fixture_ids', 'stage_request_counts', 'stage_evaluation_ceilings',
                        'qualification_kind', 'require_semantic_match', 'scope'):
                del contract[key]
        self.assertEqual(contracts[0], contracts[1])
        self.assertEqual(contracts[1], contracts[2])

    def test_backend_document_is_distinct_but_preserves_all_three_semantic_fixtures(self):
        semantic = fixture_document('semantic-c128')
        parity = fixture_document('backend-parity-c128')
        self.assertNotIn('profile', semantic)
        self.assertNotIn('profile', fixture_document('list-c128'))
        self.assertEqual(parity, {**semantic, 'profile': 'backend-parity-c128'})
        self.assertEqual(fixture_profile(parity), 'backend-parity-c128')
        self.assertEqual(fixture_profile(semantic), 'semantic-c128')
        for name in ('list-c128', 'semantic-c128'):
            self.assertEqual(profile_spec(name)['qualification_kind'], 'semantic_and_backend_parity')
            self.assertIs(profile_spec(name)['require_semantic_match'], True)
        for discriminator in ('semantic-c128', 'list-c128', 'backend_parity', False, None):
            with self.subTest(discriminator=discriminator):
                with self.assertRaises(ValueError):
                    fixture_profile({**semantic, 'profile': discriminator})

    def test_bool_and_float_lookalikes_do_not_match_integer_or_boolean_fields(self):
        mutations = (
            ('temperature', False), ('temperature', 0.0), ('max_tokens', 80.0),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=value):
                document = fixture_document('semantic-c128')
                document['fixtures'][0][field] = value
                with self.assertRaises(ValueError):
                    fixture_profile(document)
        for index, field, value in ((1, 'value', 12.0), (1, 'allow_terminal_period', 1),
                                    (2, 'case_sensitive', 0)):
            with self.subTest(index=index, field=field):
                document = fixture_document('semantic-c128')
                document['fixtures'][index]['expected_semantic'][field] = value
                with self.assertRaises(ValueError):
                    fixture_profile(document)

    def test_wrong_prompt_order_missing_duplicate_extra_or_timing_fixture_refuses(self):
        changed_documents = []
        document = fixture_document('semantic-c128')
        document['fixtures'][1]['messages'][0]['content'] += ' Different task.'
        changed_documents.append(document)
        document = fixture_document('semantic-c128')
        document['fixtures'][1], document['fixtures'][2] = document['fixtures'][2], document['fixtures'][1]
        changed_documents.append(document)
        document = fixture_document('semantic-c128')
        document['fixtures'].pop()
        changed_documents.append(document)
        document = fixture_document('semantic-c128')
        document['fixtures'].append(document['fixtures'][0])
        changed_documents.append(document)
        document = fixture_document('semantic-c128')
        document['fixtures'][0]['extra'] = 'ignored?'
        changed_documents.append(document)
        document = fixture_document('semantic-c128')
        document['timing_fixture_id'] = 'arithmetic_7_plus_5'
        changed_documents.append(document)
        for index, document in enumerate(changed_documents):
            with self.subTest(index=index):
                with self.assertRaisesRegex(ValueError, 'does not match'):
                    fixture_profile(document)

    def test_unknown_profile_names_and_invalid_document_types_refuse(self):
        for name in ('unknown', '', True, None, ['list-c128']):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    profile_spec(name)
                with self.assertRaises(ValueError):
                    fixture_document(name)
        for document in (None, [], True, 'list-c128'):
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    fixture_profile(document)


if __name__ == '__main__':
    unittest.main()
