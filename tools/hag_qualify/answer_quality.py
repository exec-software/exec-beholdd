"""Finite diagnostic grammars; annotations never replace exact backend parity."""
import re


def quality_assessment(final, fixture):
    """Assess only anchored, recognized forms; do not infer arbitrary prose meaning.

    Numeric strings stay strings, including arbitrarily long recognized values.
    Leading zeroes in CSV retain numeric meaning but fail the requested format.
    No thinking or Markdown is removed here: callers must supply the closed final.
    """
    admitted = {
        'list_1_to_8': {'kind': 'integer_csv', 'values': list(range(1, 9)), 'allow_terminal_period': True},
        'arithmetic_7_plus_5': {'kind': 'integer', 'value': 12, 'allow_terminal_period': True},
        'recall_mango': {'kind': 'word', 'value': 'MANGO', 'case_sensitive': False, 'allow_terminal_period': True},
        'grounded_no_tools_ready': {'kind': 'word', 'value': 'READY', 'case_sensitive': False, 'allow_terminal_period': True},
        'project_instructions_ready': {'kind': 'word', 'value': 'READY', 'case_sensitive': False, 'allow_terminal_period': True},
        'project_history_ready': {'kind': 'word', 'value': 'READY', 'case_sensitive': False, 'allow_terminal_period': True},
        'project_history_template_ready': {'kind': 'word', 'value': 'READY', 'case_sensitive': False, 'allow_terminal_period': True},
    }
    if type(fixture) is not dict or fixture.get('id') not in admitted:
        raise ValueError('unknown quality fixture')
    semantic = fixture.get('expected_semantic')
    expected = admitted[fixture['id']]
    if (type(semantic) is not dict or semantic != expected
            or semantic.get('allow_terminal_period') is not True
            or ('values' in semantic and (type(semantic['values']) is not list
                                         or any(type(value) is not int for value in semantic['values'])))
            or ('value' in semantic and type(semantic['value']) is not type(expected['value']))
            or ('case_sensitive' in semantic and semantic['case_sensitive'] is not False)):
        raise ValueError('invalid quality criterion')
    if type(final) is not str:
        raise ValueError('quality final must be text')
    text = final.strip()
    value = text[:-1] if text.endswith('.') else text
    result = {'correctness': 'unassessed', 'format': 'fail'}
    if semantic['kind'] == 'integer_csv':
        if re.fullmatch(r'[0-9]+(?:\s*,\s*[0-9]+)*', value):
            parts = [part.strip() for part in value.split(',')]
            normalized = [part.lstrip('0') or '0' for part in parts]
            result['correctness'] = 'pass' if normalized == [str(v) for v in semantic['values']] else 'fail'
            result['format'] = 'pass' if parts == normalized else 'fail'
    elif semantic['kind'] == 'integer':
        bare = re.fullmatch(r'(0|[1-9][0-9]*)', value)
        equation = re.fullmatch(r'7\s*\+\s*5\s*=\s*(0|[1-9][0-9]*)', value)
        match = bare or equation
        if match:
            result['correctness'] = 'pass' if match.group(1) == '12' else 'fail'
            result['format'] = 'pass' if bare else 'fail'
    elif re.fullmatch(r'[A-Za-z]+', value):
        result['correctness'] = 'pass' if value.casefold() == semantic['value'].casefold() else 'fail'
        result['format'] = 'pass'
    return result
