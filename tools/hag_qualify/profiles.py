"""Finite, versioned workloads; selecting a profile cannot loosen safety limits."""
from copy import deepcopy


GROUNDED_PROFILE = 'antimatter-grounded-c512'
PROJECT_PROFILE = 'antimatter-project-c2048'
HISTORY_PROFILE = 'antimatter-history-c2048'
TEMPLATE_HISTORY_PROFILE = 'antimatter-history-template-c2048'
LIFECYCLE_PROFILE = 'lifecycle-c2048'
GENERATED_HISTORY_PROFILE = 'antimatter-generated-history-c2048'
UI_HISTORY_PROFILE = 'antimatter-ui-history-c2048'
UI_STOP_PROFILE = 'antimatter-ui-stop-reuse-c2048'
UI_PROFILES = (UI_HISTORY_PROFILE, UI_STOP_PROFILE)
# Metadata-only checkpoint rendering of the fixed count-up prompt, before any inference.
UI_STOP_FIRST_PROMPT_TOKENS = 476
GROUNDED_PROFILES = (GROUNDED_PROFILE, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE)
C2048_PROFILES = (PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, *UI_PROFILES)
CLIENT_PROFILES = ('backend-parity-c128', *GROUNDED_PROFILES, GENERATED_HISTORY_PROFILE)
PROFILE_NAMES = ('list-c128', 'semantic-c128', 'backend-parity-c128', *GROUNDED_PROFILES,
                 LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, *UI_PROFILES)

_LIST = {
    'id': 'list_1_to_8',
    'messages': [{'role': 'user', 'content':
                  'List the integers from 1 to 8, separated by commas. No explanation. /no_think'}],
    'max_tokens': 80,
    'temperature': 0,
    'expected_semantic': {'kind': 'integer_csv', 'values': [1, 2, 3, 4, 5, 6, 7, 8],
                          'allow_terminal_period': True},
}
_ARITHMETIC = {
    'id': 'arithmetic_7_plus_5',
    'messages': [{'role': 'user', 'content':
                  'What is 7 + 5? Reply with just the number. /no_think'}],
    'max_tokens': 80,
    'temperature': 0,
    'expected_semantic': {'kind': 'integer', 'value': 12, 'allow_terminal_period': True},
}
_RECALL = {
    'id': 'recall_mango',
    'messages': [
        {'role': 'user', 'content': 'The secret word is MANGO.'},
        {'role': 'assistant', 'content': 'Understood.'},
        {'role': 'user', 'content': 'Reply with just the secret word. /no_think'},
    ],
    'max_tokens': 80,
    'temperature': 0,
    'expected_semantic': {'kind': 'word', 'value': 'MANGO', 'case_sensitive': False,
                          'allow_terminal_period': True},
}

_GROUNDED = {
    'id': 'grounded_no_tools_ready',
    'messages': [
        {'role': 'system', 'content':
         "You are the assistant inside Antimatter, a desktop application on the user's own machine.\n\n"
         "You have no tools connected in this conversation, so you cannot read or change files or run\n"
         "commands. Answer from what you are told, and say when something would need a tool you do not\n"
         "have rather than describing what you would have done."},
        {'role': 'user', 'content': 'Reply with one word: ready. /no_think'},
    ],
    'max_tokens': 128,
    'temperature': 0,
    'expected_semantic': {'kind': 'word', 'value': 'READY', 'case_sensitive': False,
                          'allow_terminal_period': True},
}

# Exact renderInstructions()/withSituation() output measured from Antimatter
# 78147b7: one synthetic 2 KiB instruction file, without reading a user's project.
# Keep the truncation before the closing tag: the final line intentionally ends
# in "conventio". The complete system UTF-8 SHA256 is pinned in profile tests.
_PROJECT_INSTRUCTIONS = (
    "The project's own instructions follow, from AGENTS.md. The people who work on it wrote them "
    "for anyone doing so, you included: follow them unless the user tells you otherwise.\n\n"
    '<file path="AGENTS.md">\n'
    + ('Use the existing project conventions.\n' * 60)[:2048]
    + '\n</file>\n\n---\n\n'
    + _GROUNDED['messages'][0]['content']
)
_PROJECT = deepcopy(_GROUNDED)
_PROJECT['id'] = 'project_instructions_ready'
_PROJECT['messages'][0]['content'] = _PROJECT_INSTRUCTIONS

# Seed the previously qualified raw reply, including inline thinking markup.
# This measures a request containing history; it does not run the first turn.
_HISTORY = deepcopy(_PROJECT)
_HISTORY['id'] = 'project_history_ready'
_HISTORY['messages'].extend([
    {'role': 'assistant', 'content': '<think>\n\n</think>\n\nready.'},
    {'role': 'user', 'content': 'Reply with that same one word again. /no_think'},
])

# The wire history is unchanged. This distinct profile records the checkpoint's
# exact no-tools template rendering, without reclassifying the earlier attempt.
_TEMPLATE_HISTORY = deepcopy(_HISTORY)
_TEMPLATE_HISTORY['id'] = 'project_history_template_ready'

_HELLO = {'id': 'hello', 'messages': [{'role': 'user', 'content': 'Hello'}],
          'max_tokens': 1, 'temperature': 0, 'prompt_tokens': 9}


def profile_limits(name):
    """Bounded per-request and client transport parameters, not tokenizer guesses."""
    _require_name(name)
    if name == UI_STOP_PROFILE:
        return {'capacity': 2048, 'maximum_prompt_tokens': max(UI_STOP_FIRST_PROMPT_TOKENS, 467),
                'exact_prompt_tokens': None, 'first_prompt_tokens': UI_STOP_FIRST_PROMPT_TOKENS,
                'reuse_prompt_tokens': 467, 'turn_prompt_tokens': [UI_STOP_FIRST_PROMPT_TOKENS, 467],
                'max_new_tokens': 64, 'turn_max_new_tokens': [64, 16],
                'maximum_evaluated_positions_per_request': max(UI_STOP_FIRST_PROMPT_TOKENS + 63, 482),
                'turn_evaluation_ceilings': [UI_STOP_FIRST_PROMPT_TOKENS + 63, 482],
                'measured_requests': 0, 'client_seconds': 240, 'client_process_seconds': 355}
    if name == UI_HISTORY_PROFILE:
        return {**profile_limits(GENERATED_HISTORY_PROFILE), 'client_process_seconds': 355}
    if name == GENERATED_HISTORY_PROFILE:
        return {'capacity': 2048, 'maximum_prompt_tokens': 491, 'exact_prompt_tokens': None,
                'first_prompt_tokens': 467, 'second_maximum_prompt_tokens': 491,
                'max_new_tokens': 128, 'maximum_evaluated_positions_per_request': 618,
                'turn_evaluation_ceilings': [594, 618], 'measured_requests': 0,
                'client_seconds': 240, 'client_process_seconds': 245}
    if name == LIFECYCLE_PROFILE:
        return {'capacity': 2048, 'maximum_prompt_tokens': 9, 'exact_prompt_tokens': 9,
                'max_new_tokens': 1, 'cancel_max_tokens': 16,
                'maximum_evaluated_positions_per_request': 9,
                'maximum_cancel_evaluated_positions': 24}
    if name == PROJECT_PROFILE:
        return {'capacity': 2048, 'maximum_prompt_tokens': 467, 'exact_prompt_tokens': 467,
                'max_new_tokens': 128, 'maximum_evaluated_positions_per_request': 594,
                'measured_requests': 1, 'client_seconds': 240, 'client_process_seconds': 245,
                'preflight_requested_output_tokens': 1583,
                'maximum_admitted_output_tokens': 1582, 'overflow_max_tokens': 1583,
                'conservative_prompt_estimate': 831,
                'client_maximum_admitted_output_tokens': 1218, 'client_overflow_max_tokens': 1219}
    if name == HISTORY_PROFILE:
        return {'capacity': 2048, 'maximum_prompt_tokens': 495, 'exact_prompt_tokens': 495,
                'max_new_tokens': 128, 'maximum_evaluated_positions_per_request': 622,
                'measured_requests': 1, 'client_seconds': 240, 'client_process_seconds': 245,
                'preflight_requested_output_tokens': 1555,
                'maximum_admitted_output_tokens': 1554, 'overflow_max_tokens': 1555,
                'conservative_prompt_estimate': 862,
                'client_maximum_admitted_output_tokens': 1187, 'client_overflow_max_tokens': 1188}
    if name == TEMPLATE_HISTORY_PROFILE:
        return {'capacity': 2048, 'maximum_prompt_tokens': 491, 'exact_prompt_tokens': 491,
                'max_new_tokens': 128, 'maximum_evaluated_positions_per_request': 618,
                'measured_requests': 1, 'client_seconds': 240, 'client_process_seconds': 245,
                'preflight_requested_output_tokens': 1559,
                'maximum_admitted_output_tokens': 1558, 'overflow_max_tokens': 1559,
                'conservative_prompt_estimate': 862,
                'client_maximum_admitted_output_tokens': 1187, 'client_overflow_max_tokens': 1188}
    if name == GROUNDED_PROFILE:
        return {'capacity': 512, 'maximum_prompt_tokens': 93, 'exact_prompt_tokens': 93,
                'max_new_tokens': 128, 'maximum_evaluated_positions_per_request': 220,
                'measured_requests': 1, 'client_seconds': 60, 'client_process_seconds': 65}
    return {'capacity': 128, 'maximum_prompt_tokens': 49, 'exact_prompt_tokens': None,
            'max_new_tokens': 80, 'maximum_evaluated_positions_per_request': 128,
            'measured_requests': 3, 'client_seconds': 30, 'client_process_seconds': 35}


def profile_deadlines(name):
    """Finite child/service/outer deadlines; independent of per-request limits."""
    _require_name(name)
    if name in (GENERATED_HISTORY_PROFILE, *UI_PROFILES):
        return {'self-check': (180, 195, 210), 'cpu-reference': (720, 735, 750),
                'answers': (360, 375, 390)}
    if name == LIFECYCLE_PROFILE:
        return {'self-check': (180, 195, 210), 'cpu-reference': (720, 735, 750),
                'answers': (600, 615, 630)}
    if name in (PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE):
        return {'self-check': (180, 195, 210), 'cpu-reference': (720, 735, 750),
                'answers': (360, 375, 390), 'timing': (600, 615, 630)}
    return {stage: (180, 195, 210) for stage in ('self-check', 'cpu-reference', 'answers', 'timing')}


def profile_modes(name):
    """Exactly the ordered stages belonging to a profile; no implicit extra work."""
    _require_name(name)
    if name in (LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, *UI_PROFILES):
        return ('self-check', 'cpu-reference', 'answers')
    return ('self-check', 'cpu-reference', 'answers', 'timing')


def _require_name(name):
    if type(name) is not str or name not in PROFILE_NAMES:
        raise ValueError('unknown qualification profile')


def fixture_document(name='list-c128'):
    _require_name(name)
    if name == UI_STOP_PROFILE:
        first, reuse = deepcopy(_PROJECT), deepcopy(_PROJECT)
        first.update(id='ui_stop_count', max_tokens=64,
                     expected_semantic={'kind': 'cancel_after_visible_numeric_content'})
        first['messages'][-1]['content'] = 'Count upwards from 1 to 100, separating numbers with spaces. /no_think'
        reuse.update(id='ui_stop_reuse', max_tokens=16)
        first.pop('temperature')
        reuse.pop('temperature')
        return {'profile': name, 'first_fixture': first, 'reuse_fixture': reuse}
    if name == UI_HISTORY_PROFILE:
        first = deepcopy(_PROJECT)
        first['id'] = 'ui_turn_1'
        first.pop('temperature')
        return {'profile': name, 'first_fixture': first,
                'followup_user': 'Reply with that same one word again. /no_think',
                'second_fixture_id': 'ui_turn_2'}
    if name == GENERATED_HISTORY_PROFILE:
        first = deepcopy(_PROJECT)
        first['id'] = 'generated_turn_1'
        return {'profile': name, 'first_fixture': first,
                'followup_user': 'Reply with that same one word again. /no_think',
                'second_fixture_id': 'generated_turn_2'}
    if name == LIFECYCLE_PROFILE:
        return deepcopy({'profile': name, 'fixtures': [_HELLO], 'maximum_prompt_tokens': 9})
    if name == PROJECT_PROFILE:
        return deepcopy({'profile': name, 'fixtures': [_PROJECT],
                         'timing_fixture_id': _PROJECT['id'], 'maximum_prompt_tokens': 467})
    if name == HISTORY_PROFILE:
        return deepcopy({'profile': name, 'fixtures': [_HISTORY],
                         'timing_fixture_id': _HISTORY['id'], 'maximum_prompt_tokens': 495})
    if name == TEMPLATE_HISTORY_PROFILE:
        return deepcopy({'profile': name, 'fixtures': [_TEMPLATE_HISTORY],
                         'timing_fixture_id': _TEMPLATE_HISTORY['id'], 'maximum_prompt_tokens': 491})
    if name == GROUNDED_PROFILE:
        return deepcopy({'profile': name, 'fixtures': [_GROUNDED],
                         'timing_fixture_id': _GROUNDED['id'], 'maximum_prompt_tokens': 93})
    fixtures = [_LIST] if name == 'list-c128' else [_LIST, _ARITHMETIC, _RECALL]
    document = {'fixtures': fixtures, 'timing_fixture_id': 'list_1_to_8',
                'maximum_prompt_tokens': 49}
    # Existing documents keep their exact shape. The explicit new discriminator
    # prevents identical prompts from silently changing qualification strategy.
    if name == 'backend-parity-c128':
        document['profile'] = name
    return deepcopy(document)


def _typed_equal(actual, expected):
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return (actual.keys() == expected.keys()
                and all(type(key) is str for key in actual)
                and all(type(key) is str and _typed_equal(actual[key], value)
                        for key, value in expected.items()))
    if type(expected) is list:
        return len(actual) == len(expected) and all(_typed_equal(a, b) for a, b in zip(actual, expected))
    return actual == expected


def fixture_profile(document):
    """Identify only an exact known document, including scalar types and order."""
    for name in PROFILE_NAMES:
        if _typed_equal(document, fixture_document(name)):
            return name
    raise ValueError('fixture document does not match a declared qualification profile')


def profile_spec(name):
    _require_name(name)
    document = fixture_document(name)
    if name == UI_STOP_PROFILE:
        limits = profile_limits(name)
        return {'profile': name, 'qualification_kind': 'ui_stop_reuse_and_backend_parity',
                'require_semantic_match': False, 'fixture_ids': ['ui_stop_count', 'ui_stop_reuse'],
                'stage_request_counts': {'cpu-reference': 1, 'answers': 2},
                'stage_evaluation_ceilings': {'cpu-reference': 482,
                                              'answers': sum(limits['turn_evaluation_ceilings'])}}
    if name == UI_HISTORY_PROFILE:
        return {'profile': name, 'qualification_kind': 'ui_history_and_backend_parity', 'require_semantic_match': False,
                'fixture_ids': ['ui_turn_1', 'ui_turn_2'],
                'stage_request_counts': {'cpu-reference': 2, 'answers': 2},
                'stage_evaluation_ceilings': {'cpu-reference': 1212, 'answers': 1212}}
    if name == GENERATED_HISTORY_PROFILE:
        return {'profile': name, 'qualification_kind': 'backend_parity', 'require_semantic_match': False,
                'fixture_ids': ['generated_turn_1', 'generated_turn_2'],
                'stage_request_counts': {'cpu-reference': 2, 'answers': 2},
                'stage_evaluation_ceilings': {'cpu-reference': 1212, 'answers': 1212}}
    count = len(document['fixtures'])
    if name == LIFECYCLE_PROFILE:
        return {'profile': name, 'qualification_kind': 'tcp_lifecycle_and_backend_parity',
                'require_semantic_match': False, 'fixture_ids': ['hello'],
                'stage_request_counts': {'cpu-reference': 1, 'answers': 5},
                'stage_evaluation_ceilings': {'cpu-reference': 9, 'answers': 60}}
    if name in GROUNDED_PROFILES:
        ceiling = profile_limits(name)['maximum_evaluated_positions_per_request']
        return {'profile': name, 'qualification_kind': 'backend_parity', 'require_semantic_match': False,
                'fixture_ids': [document['fixtures'][0]['id']],
                'stage_request_counts': {'cpu-reference': 1, 'answers': 1, 'timing': 2},
                'stage_evaluation_ceilings': {'cpu-reference': ceiling, 'answers': ceiling, 'timing': 2 * ceiling}}
    return {
        'profile': name,
        'qualification_kind': ('backend_parity' if name == 'backend-parity-c128'
                               else 'semantic_and_backend_parity'),
        'require_semantic_match': name != 'backend-parity-c128',
        'fixture_ids': [fixture['id'] for fixture in document['fixtures']],
        'stage_request_counts': {'cpu-reference': count, 'answers': count, 'timing': 4},
        'stage_evaluation_ceilings': {'cpu-reference': 128 * count, 'answers': 128 * count, 'timing': 512},
    }
