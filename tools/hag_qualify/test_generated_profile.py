"""Generated-conversation admission only; no model, executable or network work."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

try:
    from . import profiles, inputs, gate_common as common, antimatter
except ImportError:
    import profiles, inputs, gate_common as common, antimatter

PROFILE = profiles.GENERATED_HISTORY_PROFILE


class GeneratedProfileTests(unittest.TestCase):
    def test_recipe_contains_fixed_questions_but_no_seeded_assistant(self):
        document = profiles.fixture_document(PROFILE)
        expected = deepcopy(profiles.fixture_document(profiles.PROJECT_PROFILE)['fixtures'][0])
        expected['id'] = 'generated_turn_1'
        self.assertEqual(document, {'profile': PROFILE, 'first_fixture': expected,
            'followup_user': 'Reply with that same one word again. /no_think',
            'second_fixture_id': 'generated_turn_2'})
        self.assertEqual([m['role'] for m in document['first_fixture']['messages']], ['system', 'user'])
        self.assertNotIn(PROFILE, profiles.GROUNDED_PROFILES)
        self.assertIn(PROFILE, profiles.C2048_PROFILES)
        self.assertIn(PROFILE, profiles.CLIENT_PROFILES)
        self.assertEqual(profiles.fixture_profile(document), PROFILE)

    def test_recipe_rejects_seeded_content_extra_fields_order_and_wrong_types(self):
        original = profiles.fixture_document(PROFILE)
        changes = (
            lambda d: d['first_fixture']['messages'].append({'role': 'assistant', 'content': 'ready.'}),
            lambda d: d['first_fixture']['messages'].reverse(),
            lambda d: d['first_fixture'].update(temperature=False),
            lambda d: d.update(followup_user='Reply again.'),
            lambda d: d.update(timing_fixture_id='generated_turn_2'),
            lambda d: d.update(profile=profiles.TEMPLATE_HISTORY_PROFILE),
        )
        for change in changes:
            document = deepcopy(original)
            change(document)
            with self.subTest(document=document), self.assertRaises(ValueError):
                profiles.fixture_profile(document)
        self.assertEqual(profiles.fixture_document(PROFILE), original)

    def test_exact_two_turn_counts_budgets_and_three_stage_deadlines(self):
        spec = profiles.profile_spec(PROFILE)
        limits = profiles.profile_limits(PROFILE)
        self.assertEqual(spec['fixture_ids'], ['generated_turn_1', 'generated_turn_2'])
        self.assertEqual(spec['stage_request_counts'], {'cpu-reference': 2, 'answers': 2})
        self.assertEqual(spec['stage_evaluation_ceilings'], {'cpu-reference': 1212, 'answers': 1212})
        self.assertEqual(limits['turn_evaluation_ceilings'], [467 + 128 - 1, 491 + 128 - 1])
        self.assertEqual((limits['first_prompt_tokens'], limits['second_maximum_prompt_tokens']), (467, 491))
        self.assertIsNone(limits['exact_prompt_tokens'])
        self.assertEqual(profiles.profile_modes(PROFILE), ('self-check', 'cpu-reference', 'answers'))
        self.assertEqual(profiles.profile_deadlines(PROFILE), {
            'self-check': (180, 195, 210), 'cpu-reference': (720, 735, 750), 'answers': (360, 375, 390)})
        self.assertEqual((limits['client_seconds'], limits['client_process_seconds']), (240, 245))
        self.assertEqual(spec['qualification_kind'], 'backend_parity')
        self.assertFalse(spec['require_semantic_match'])

    def test_contract_has_same_resource_limits_and_no_timing_or_quality_claim(self):
        contract = inputs.profile_contract(PROFILE, 'antimatter')
        previous = inputs.profile_contract(profiles.TEMPLATE_HISTORY_PROFILE, 'antimatter')
        for key in ('capacity', 'hag_charge_bytes', 'hag_budget_bytes', 'cpu_model_budget_bytes',
                    'runtime_combined_forecast_bytes', 'process_memory_max_bytes', 'process_early_stop_bytes',
                    'swap_max_bytes', 'cpu_quota_percent', 'tasks_max', 'no_new_privileges',
                    'host_start_headroom_bytes', 'host_stop_headroom_bytes', 'thermal_cutoff_c',
                    'thermal_poll_seconds', 'thermal_start_max_c', 'thermal_start_stable_seconds',
                    'private_cgroup_admission', 'crash_watcher_seconds'):
            self.assertEqual(contract[key], previous[key], key)
        self.assertEqual((contract['client_chat_requests'], contract['client_stream_requests']), (2, 0))
        self.assertTrue(contract['prompt_inspector_required'])
        self.assertTrue(contract['generated_history_required'])
        self.assertEqual(contract['history_source'], 'same-backend-raw-first-answer')
        self.assertFalse(contract['live_cancellation_qualified'])
        self.assertFalse(any(k.startswith('timing_') for k in contract))
        self.assertNotIn('runtime_phase_diagnostics', contract)
        with self.assertRaises(ValueError):
            inputs.profile_contract(PROFILE, 'antimatter', runtime_phase_diagnostics=True)
        with self.assertRaises(ValueError):
            inputs.profile_contract(PROFILE)

    def test_contract_refuses_per_turn_relaxation_extra_stage_and_false_capability(self):
        contract = inputs.profile_contract(PROFILE, 'antimatter')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / 'answer-fixtures.json').write_text(json.dumps(profiles.fixture_document(PROFILE)))
            path = root / 'qualification-contract.json'
            path.write_text(json.dumps(contract))
            self.assertEqual(common.approved_contract(root), contract)
            changes = ({'first_prompt_tokens': 468}, {'second_maximum_prompt_tokens': 492},
                       {'turn_evaluation_ceilings': [594, 619]}, {'turn_evaluation_ceilings': [True, 618]},
                       {'prompt_inspector_required': False}, {'client_chat_requests': 3},
                       {'client_stream_requests': 1}, {'history_source': 'cpu-seeded'},
                       {'timing_initial_nonstream_requests': 1}, {'stream_client': 'python-http'},
                       {'stage_evaluation_ceilings': {'cpu-reference': 1212, 'answers': 1213}},
                       {'stage_deadlines': {**contract['stage_deadlines'], 'timing': {'child': 600, 'service': 615, 'outer': 630}}})
            for changed in changes:
                path.write_text(json.dumps({**contract, **changed}))
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    common.approved_contract(root)

    def test_generated_client_requires_the_capacity_aware_committed_provider(self):
        self.assertTrue(antimatter.validate_options('/repo', antimatter.GROUNDED_COMMIT, '/node', '/esbuild', PROFILE))
        for values in ((None, None, None, None), ('/repo', antimatter.SUPPORTED_COMMIT, '/node', '/esbuild'),
                       ('/repo', antimatter.GROUNDED_COMMIT, None, '/esbuild')):
            with self.subTest(values=values), self.assertRaises(ValueError):
                antimatter.validate_options(*values, PROFILE)


if __name__ == '__main__':
    unittest.main()
