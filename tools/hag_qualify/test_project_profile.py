"""Exact synthetic project-context profile; no model or client execution."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

try:
    from . import profiles
    from .answer_quality import quality_assessment
except ImportError:
    import profiles
    from answer_quality import quality_assessment


class ProjectProfileTests(unittest.TestCase):
    def test_exact_measured_rendering_and_original_grounding(self):
        document = profiles.fixture_document(profiles.PROJECT_PROFILE)
        fixture = document['fixtures'][0]
        system = fixture['messages'][0]['content']
        self.assertEqual(len(system.encode()), 2593)
        self.assertEqual(hashlib.sha256(system.encode()).hexdigest(),
                         '91e1e04229504e4a2c6d5d58bf1aa9ffd77037b676c63dae31ed5e4bc6e3dfa7')
        content = system.split('<file path="AGENTS.md">\n', 1)[1].split('\n</file>', 1)[0]
        self.assertEqual(content, ('Use the existing project conventions.\n' * 60)[:2048])
        grounded = profiles.fixture_document(profiles.GROUNDED_PROFILE)['fixtures'][0]
        self.assertEqual(system.split('\n\n---\n\n', 1)[1], grounded['messages'][0]['content'])
        self.assertEqual(fixture['messages'][1], grounded['messages'][1])
        self.assertEqual((fixture['max_tokens'], fixture['temperature']), (128, 0))
        self.assertEqual(profiles.fixture_profile(document), profiles.PROJECT_PROFILE)

    def test_server_exact_boundary_differs_from_conservative_client_boundary(self):
        limits = profiles.profile_limits(profiles.PROJECT_PROFILE)
        messages = profiles.fixture_document(profiles.PROJECT_PROFILE)['fixtures'][0]['messages']
        # Shared committed estimator: ceil(characters / 3.2) + four per message.
        estimate = sum((len(m['content']) * 5 + 15) // 16 + 4 for m in messages)
        self.assertEqual(estimate, 831)
        self.assertEqual(limits['conservative_prompt_estimate'], estimate)
        for prompt, fit, overflow in ((467, 'maximum_admitted_output_tokens', 'overflow_max_tokens'),
                                     (831, 'client_maximum_admitted_output_tokens', 'client_overflow_max_tokens')):
            self.assertEqual(prompt + limits[fit] - 1, 2048)
            self.assertEqual(prompt + limits[overflow] - 1, 2049)
        self.assertEqual(limits['preflight_requested_output_tokens'], 1583)
        self.assertEqual(limits['maximum_evaluated_positions_per_request'], 467 + 128 - 1)
        self.assertEqual((limits['client_seconds'], limits['client_process_seconds']), (240, 245))

    def test_backend_only_scope_and_finite_stage_workload(self):
        spec = profiles.profile_spec(profiles.PROJECT_PROFILE)
        self.assertEqual(spec['fixture_ids'], ['project_instructions_ready'])
        self.assertEqual(spec['qualification_kind'], 'backend_parity')
        self.assertIs(spec['require_semantic_match'], False)
        self.assertEqual(spec['stage_request_counts'], {'cpu-reference': 1, 'answers': 1, 'timing': 2})
        self.assertEqual(spec['stage_evaluation_ceilings'], {'cpu-reference': 594, 'answers': 594, 'timing': 1188})
        self.assertEqual(profiles.profile_deadlines(profiles.PROJECT_PROFILE),
                         {'self-check': (180, 195, 210), 'cpu-reference': (720, 735, 750),
                          'answers': (360, 375, 390), 'timing': (600, 615, 630)})

    def test_ready_quality_is_diagnostic_with_anchored_finite_grammar(self):
        fixture = profiles.fixture_document(profiles.PROJECT_PROFILE)['fixtures'][0]
        cases = [('READY', 'pass', 'pass'), (' ready. ', 'pass', 'pass'),
                 ('BUSY', 'fail', 'pass'), ('I am ready.', 'unassessed', 'fail'),
                 ('Not ready.', 'unassessed', 'fail'), ('**READY**', 'unassessed', 'fail')]
        for text, correctness, format_result in cases:
            with self.subTest(text=text):
                self.assertEqual(quality_assessment(text, fixture),
                                 {'correctness': correctness, 'format': format_result})
                # These annotations never change backend-only qualification policy.
                self.assertIs(profiles.profile_spec(profiles.PROJECT_PROFILE)['require_semantic_match'], False)
        invalid = copy.deepcopy(fixture)
        invalid['expected_semantic']['value'] = 'BUSY'
        with self.assertRaises(ValueError):
            quality_assessment('BUSY', invalid)

    def test_fixture_mutations_and_cross_profile_labels_refuse(self):
        base = profiles.fixture_document(profiles.PROJECT_PROFILE)
        changes = [lambda d: d.update(profile=profiles.GROUNDED_PROFILE),
                   lambda d: d.update(maximum_prompt_tokens='467'),
                   lambda d: d['fixtures'][0].update(max_tokens=129),
                   lambda d: d['fixtures'][0].update(temperature=False),
                   lambda d: d['fixtures'][0]['messages'][0].update(content='shortened'),
                   lambda d: d['fixtures'][0]['messages'].reverse()]
        for change in changes:
            document = copy.deepcopy(base)
            change(document)
            with self.assertRaises(ValueError):
                profiles.fixture_profile(document)
        base['fixtures'][0]['messages'][0]['content'] = 'mutated return value'
        self.assertEqual(profiles.fixture_profile(profiles.fixture_document(profiles.PROJECT_PROFILE)),
                         profiles.PROJECT_PROFILE)

    def test_default_document_and_previous_grounded_limits_are_unchanged(self):
        default = json.loads(Path(profiles.__file__).with_name('answer-fixtures.json').read_text())
        self.assertEqual(profiles.fixture_document(), default)
        grounded = profiles.fixture_document(profiles.GROUNDED_PROFILE)['fixtures'][0]
        self.assertEqual(hashlib.sha256(grounded['messages'][0]['content'].encode()).hexdigest(),
                         '127291a4706537b910bee4ff99a0ac4feeee3b58d718484e939afb5b3ec94b39')
        self.assertEqual(profiles.profile_limits(profiles.GROUNDED_PROFILE),
                         {'capacity': 512, 'maximum_prompt_tokens': 93, 'exact_prompt_tokens': 93,
                          'max_new_tokens': 128, 'maximum_evaluated_positions_per_request': 220,
                          'measured_requests': 1, 'client_seconds': 60, 'client_process_seconds': 65})
        for name in (name for name in profiles.PROFILE_NAMES if name not in profiles.C2048_PROFILES):
            self.assertTrue(all(value == (180, 195, 210) for value in profiles.profile_deadlines(name).values()))

    def test_lifecycle_fixture_is_length_limited_and_has_no_answer_or_timing_criterion(self):
        document = profiles.fixture_document(profiles.LIFECYCLE_PROFILE)
        self.assertEqual(document, {'profile': 'lifecycle-c2048', 'maximum_prompt_tokens': 9,
            'fixtures': [{'id': 'hello', 'messages': [{'role': 'user', 'content': 'Hello'}],
                          'max_tokens': 1, 'temperature': 0, 'prompt_tokens': 9}]})
        self.assertEqual(profiles.fixture_profile(document), profiles.LIFECYCLE_PROFILE)
        for change in (lambda d: d.update(profile=profiles.PROJECT_PROFILE),
                       lambda d: d.update(timing_fixture_id='hello'),
                       lambda d: d['fixtures'][0].update(expected_semantic={'kind': 'word', 'value': 'Hello'}),
                       lambda d: d['fixtures'][0].update(max_tokens=True),
                       lambda d: d['fixtures'][0].update(prompt_tokens=9.0),
                       lambda d: d['fixtures'][0]['messages'][0].update(content='Hello!')):
            changed = copy.deepcopy(document)
            change(changed)
            with self.assertRaises(ValueError):
                profiles.fixture_profile(changed)

    def test_lifecycle_bounds_include_four_successes_and_one_cancel_without_timing(self):
        name = profiles.LIFECYCLE_PROFILE
        limits = profiles.profile_limits(name)
        spec = profiles.profile_spec(name)
        success = limits['exact_prompt_tokens'] + limits['max_new_tokens'] - 1
        cancellation = limits['exact_prompt_tokens'] + limits['cancel_max_tokens'] - 1
        self.assertEqual((success, cancellation), (9, 24))
        self.assertEqual(limits['maximum_cancel_evaluated_positions'], cancellation)
        self.assertEqual(spec['stage_request_counts'], {'cpu-reference': 1, 'answers': 5})
        self.assertEqual(spec['stage_evaluation_ceilings'],
                         {'cpu-reference': success, 'answers': 4 * success + cancellation})
        self.assertEqual(spec['fixture_ids'], ['hello'])
        self.assertEqual(spec['qualification_kind'], 'tcp_lifecycle_and_backend_parity')
        self.assertIs(spec['require_semantic_match'], False)
        self.assertEqual(profiles.profile_modes(name), ('self-check', 'cpu-reference', 'answers'))
        self.assertEqual(profiles.profile_deadlines(name),
                         {'self-check': (180, 195, 210), 'cpu-reference': (720, 735, 750),
                          'answers': (600, 615, 630)})
        self.assertNotIn('client_seconds', limits)
        self.assertNotIn('measured_requests', limits)
        for old in (value for value in profiles.PROFILE_NAMES
                    if value not in (name, profiles.GENERATED_HISTORY_PROFILE, *profiles.UI_PROFILES)):
            self.assertEqual(profiles.profile_modes(old), ('self-check', 'cpu-reference', 'answers', 'timing'))
        with self.assertRaises(ValueError):
            profiles.profile_modes('unknown')

    def test_history_preserves_project_bytes_and_raw_seed_in_wire_order(self):
        document = profiles.fixture_document(profiles.HISTORY_PROFILE)
        fixture = document['fixtures'][0]
        project = profiles.fixture_document(profiles.PROJECT_PROFILE)['fixtures'][0]
        self.assertEqual(fixture['id'], 'project_history_ready')
        self.assertEqual(fixture['messages'], project['messages'] + [
            {'role': 'assistant', 'content': '<think>\n\n</think>\n\nready.'},
            {'role': 'user', 'content': 'Reply with that same one word again. /no_think'},
        ])
        self.assertEqual((fixture['max_tokens'], fixture['temperature']), (128, 0))
        self.assertEqual(fixture['expected_semantic'], project['expected_semantic'])
        self.assertEqual(document['timing_fixture_id'], fixture['id'])
        self.assertEqual(profiles.fixture_profile(document), profiles.HISTORY_PROFILE)
        for change in (lambda d: d.update(profile=profiles.PROJECT_PROFILE),
                       lambda d: d['fixtures'][0]['messages'][2].update(content='ready.'),
                       lambda d: d['fixtures'][0]['messages'][2].update(role='user'),
                       lambda d: d['fixtures'][0]['messages'].pop(2),
                       lambda d: d['fixtures'][0].update(temperature=False),
                       lambda d: d.update(maximum_prompt_tokens=495.0)):
            changed = copy.deepcopy(document)
            change(changed)
            with self.assertRaises(ValueError):
                profiles.fixture_profile(changed)

    def test_history_measured_bounds_and_conservative_client_boundary_remain_distinct(self):
        name = profiles.HISTORY_PROFILE
        limits = profiles.profile_limits(name)
        messages = profiles.fixture_document(name)['fixtures'][0]['messages']
        estimate = sum((len(message['content']) * 5 + 15) // 16 + 4 for message in messages)
        self.assertEqual(estimate, limits['conservative_prompt_estimate'])
        self.assertEqual(estimate, 862)
        self.assertEqual(limits['exact_prompt_tokens'], 495)
        self.assertEqual(limits['maximum_evaluated_positions_per_request'], 495 + 128 - 1)
        for prompt, fit, overflow in ((495, 'maximum_admitted_output_tokens', 'overflow_max_tokens'),
                                     (862, 'client_maximum_admitted_output_tokens', 'client_overflow_max_tokens')):
            self.assertEqual(prompt + limits[fit] - 1, 2048)
            self.assertEqual(prompt + limits[overflow] - 1, 2049)
        self.assertEqual(limits['preflight_requested_output_tokens'], 1555)
        self.assertEqual((limits['client_seconds'], limits['client_process_seconds']), (240, 245))
        spec = profiles.profile_spec(name)
        self.assertEqual(spec['stage_request_counts'], {'cpu-reference': 1, 'answers': 1, 'timing': 2})
        self.assertEqual(spec['stage_evaluation_ceilings'], {'cpu-reference': 622, 'answers': 622, 'timing': 1244})
        self.assertEqual(spec['qualification_kind'], 'backend_parity')
        self.assertIs(spec['require_semantic_match'], False)
        self.assertEqual(profiles.profile_deadlines(name), profiles.profile_deadlines(profiles.PROJECT_PROFILE))
        self.assertEqual(profiles.profile_modes(name), profiles.profile_modes(profiles.PROJECT_PROFILE))

    def test_template_history_keeps_wire_messages_but_has_distinct_frozen_identity(self):
        old = profiles.fixture_document(profiles.HISTORY_PROFILE)
        document = profiles.fixture_document(profiles.TEMPLATE_HISTORY_PROFILE)
        fixture = document['fixtures'][0]
        self.assertEqual(fixture['id'], 'project_history_template_ready')
        self.assertEqual(fixture['messages'], old['fixtures'][0]['messages'])
        self.assertEqual(fixture['messages'][2],
                         {'role': 'assistant', 'content': '<think>\n\n</think>\n\nready.'})
        self.assertEqual(fixture['expected_semantic'], old['fixtures'][0]['expected_semantic'])
        self.assertEqual((fixture['max_tokens'], fixture['temperature']), (128, 0))
        self.assertEqual(document['timing_fixture_id'], fixture['id'])
        self.assertEqual(profiles.fixture_profile(document), profiles.TEMPLATE_HISTORY_PROFILE)
        self.assertEqual(profiles.fixture_profile(old), profiles.HISTORY_PROFILE)
        self.assertIn(profiles.TEMPLATE_HISTORY_PROFILE, profiles.GROUNDED_PROFILES)
        self.assertIn(profiles.TEMPLATE_HISTORY_PROFILE, profiles.C2048_PROFILES)
        for change in (lambda d: d.update(profile=profiles.HISTORY_PROFILE),
                       lambda d: d.update(maximum_prompt_tokens=495),
                       lambda d: d.update(maximum_prompt_tokens=491.0),
                       lambda d: d['fixtures'][0].update(id='project_history_ready'),
                       lambda d: d['fixtures'][0]['messages'][2].update(content='ready.'),
                       lambda d: d['fixtures'][0].update(temperature=False)):
            changed = copy.deepcopy(document)
            change(changed)
            with self.assertRaises(ValueError):
                profiles.fixture_profile(changed)
        document['fixtures'][0]['messages'][2]['content'] = 'changed copy'
        self.assertEqual(profiles.fixture_document(profiles.HISTORY_PROFILE), old)
        self.assertEqual(profiles.fixture_document(profiles.TEMPLATE_HISTORY_PROFILE)['fixtures'][0]['messages'],
                         old['fixtures'][0]['messages'])

    def test_template_history_measured_server_boundary_and_unmodified_client_estimate(self):
        name = profiles.TEMPLATE_HISTORY_PROFILE
        limits = profiles.profile_limits(name)
        messages = profiles.fixture_document(name)['fixtures'][0]['messages']
        estimate = sum((len(message['content']) * 5 + 15) // 16 + 4 for message in messages)
        self.assertEqual(estimate, 862)
        self.assertEqual(limits['conservative_prompt_estimate'], estimate)
        self.assertEqual((limits['exact_prompt_tokens'], limits['maximum_prompt_tokens']), (491, 491))
        self.assertEqual(limits['maximum_evaluated_positions_per_request'], 491 + 128 - 1)
        for prompt, fit, overflow in ((491, 'maximum_admitted_output_tokens', 'overflow_max_tokens'),
                                     (862, 'client_maximum_admitted_output_tokens', 'client_overflow_max_tokens')):
            self.assertEqual(prompt + limits[fit] - 1, 2048)
            self.assertEqual(prompt + limits[overflow] - 1, 2049)
        self.assertEqual(limits['preflight_requested_output_tokens'], 1559)
        self.assertEqual((limits['client_seconds'], limits['client_process_seconds']), (240, 245))
        spec = profiles.profile_spec(name)
        self.assertEqual(spec['fixture_ids'], ['project_history_template_ready'])
        self.assertEqual(spec['stage_request_counts'], {'cpu-reference': 1, 'answers': 1, 'timing': 2})
        self.assertEqual(spec['stage_evaluation_ceilings'], {'cpu-reference': 618, 'answers': 618, 'timing': 1236})
        self.assertEqual(spec['qualification_kind'], 'backend_parity')
        self.assertIs(spec['require_semantic_match'], False)
        self.assertEqual(profiles.profile_deadlines(name), profiles.profile_deadlines(profiles.PROJECT_PROFILE))
        self.assertEqual(profiles.profile_modes(name), profiles.profile_modes(profiles.PROJECT_PROFILE))
        old = profiles.profile_limits(profiles.HISTORY_PROFILE)
        self.assertEqual((old['exact_prompt_tokens'], old['maximum_evaluated_positions_per_request'],
                          old['maximum_admitted_output_tokens'], old['overflow_max_tokens']),
                         (495, 622, 1554, 1555))
        self.assertEqual(profiles.profile_spec(profiles.HISTORY_PROFILE)['stage_evaluation_ceilings'],
                         {'cpu-reference': 622, 'answers': 622, 'timing': 1244})


if __name__ == '__main__':
    unittest.main()
