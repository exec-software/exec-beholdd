"""Shared identity and containment checks for fresh, bounded answer profiles."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

try:
    from .profiles import fixture_profile, profile_spec, GROUNDED_PROFILE, GROUNDED_PROFILES, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, CLIENT_PROFILES, C2048_PROFILES, profile_limits, profile_deadlines, profile_modes
except ImportError:
    from profiles import fixture_profile, profile_spec, GROUNDED_PROFILE, GROUNDED_PROFILES, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, CLIENT_PROFILES, C2048_PROFILES, profile_limits, profile_deadlines, profile_modes

GIB = 1024 ** 3
MEMORY_LIMIT = 2 * GIB
STOP_CGROUP_BYTES = 1536 * 1024 ** 2
START_HEADROOM = 8 * GIB
STOP_HEADROOM = 4 * GIB
ROOT = Path(__file__).resolve().parent
MODEL_SHA256 = 'e150ed544dfe6016930c026a93913a5e3184181ebfe6ab2223ae01dd0491784c'
MODEL_BYTES = 639447744
EXPECTED_LIMITS = {'memory.max': str(MEMORY_LIMIT), 'memory.swap.max': '0',
                   'pids.max': '64', 'cpu.max': '75000 100000'}
BINARIES = {mode: 'beholdd' for mode in ('cpu-reference', 'answers', 'timing')}
MODES = ('self-check', 'cpu-reference', 'answers', 'timing')
GPU_MODES = ('answers', 'timing')
GATE_FILES = ('gate_common.py', 'profiles.py', 'answer_quality.py', 'antimatter.py', 'antimatter-client.mjs',
              'antimatter-grounded-client.mjs',
              'run-gate.py', 'gate-child.py', 'answer-gate.py',
              'lifecycle-gate.py', 'lifecycle.py', 'test_lifecycle.py',
              'generated-history-gate.py', 'generated_history.py', 'generated-history-client.mjs', 'test_generated_history.py',
              'ui-history-gate.py', 'ui_history.py', 'ui_client.py', 'ui-run-local.py',
              'test_ui_history.py', 'test_ui_client.py', 'test_ui_gate.py', 'test_ui_wrappers.py', 'test_ui_runner.py',
              'ui-stop-gate.py', 'ui_stop.py', 'ui_stop_client.py',
              'test_ui_stop.py', 'test_ui_stop_gate.py', 'test_ui_stop_wrappers.py', 'test_ui_stop_client.py',
              'test_gate_wrappers.py', 'test_answer_gate.py', 'answer-fixtures.json',
              'cpu-config.json', 'hag-config.json', 'qualification-contract.json',
              'closeout.py', 'run-config.json')
CHILD_SECONDS = 180
SERVICE_SECONDS = 195
OUTER_SECONDS = 210
MAX_SOURCE_FILES = 4096
MAX_SOURCE_FILE_BYTES = 16 * 1024 ** 2
MAX_SOURCE_TOTAL_BYTES = 64 * 1024 ** 2
MAX_HASH_BYTES = GIB
PHASE_ENVIRONMENT = 'HAG_SERVING_PHASES'
PHASE_PREFIX = 'HAG_SERVING_PHASE '


def phase_diagnostic_contract(profile, enabled=False):
    """Return the one opt-in policy; omitted diagnostics preserve older snapshots."""
    if type(enabled) is not bool:
        raise ValueError('runtime phase diagnostics requires a boolean option')
    if not enabled:
        return None
    if profile != TEMPLATE_HISTORY_PROFILE:
        raise ValueError('runtime phase diagnostics is restricted to the checkpoint-rendered history profile')
    return {'schema': 1, 'environment': {PHASE_ENVIRONMENT: '1'}, 'stages': ['answers'],
            'log_prefix': PHASE_PREFIX, 'qualifying_evidence': False}


def phase_diagnostics(contract, mode=None):
    """Validate frozen opt-in data and optionally resolve it for one stage."""
    if 'runtime_phase_diagnostics' not in contract:
        return None
    value = contract['runtime_phase_diagnostics']
    expected = phase_diagnostic_contract(contract.get('profile'), True)
    if (type(value) is not dict or set(value) != set(expected)
            or type(value.get('schema')) is not int or value['schema'] != 1
            or type(value.get('environment')) is not dict or value['environment'] != expected['environment']
            or type(value.get('stages')) is not list or value['stages'] != ['answers']
            or type(value.get('log_prefix')) is not str or value['log_prefix'] != PHASE_PREFIX
            or value.get('qualifying_evidence') is not False):
        raise ValueError('preregistered runtime phase diagnostics differs')
    if mode is None:
        return expected
    if mode not in profile_modes(contract['profile']):
        raise ValueError('diagnostic stage is not admitted by this profile')
    enabled = mode == 'answers'
    return {'schema': 1, 'enabled': enabled,
            'environment': {PHASE_ENVIRONMENT: '1'} if enabled else {},
            'log_prefix': PHASE_PREFIX, 'qualifying_evidence': False}


def require_phase_diagnostics(record, contract, mode):
    expected = phase_diagnostics(contract, mode)
    if expected is None:
        if 'runtime_phase_diagnostics' in record:
            raise ValueError('undeclared runtime phase diagnostics receipt')
    else:
        value = record.get('runtime_phase_diagnostics')
        if (type(value) is not dict or value != expected
                or type(value.get('schema')) is not int or type(value.get('enabled')) is not bool
                or value.get('qualifying_evidence') is not False):
            raise ValueError('resolved runtime phase diagnostics receipt differs')


def strict_json(text):
    def invalid(value):
        raise ValueError('nonfinite JSON value: ' + value)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(text, parse_constant=invalid, object_pairs_hook=pairs)


def read_json(path, limit=1024 * 1024):
    with regular_reader(path, limit) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'oversized evidence: {path}')
    return strict_json(data)


def regular_reader(path, limit):
    """Open a bounded regular input without following a final symlink or FIFO."""
    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError(f'input path is not canonical: {path}')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 <= metadata.st_size <= limit:
            raise ValueError(f'input is not a bounded regular file: {path}')
        return os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise


def run_config(root=ROOT):
    value = read_json(root / 'run-config.json')
    if (type(value) is not dict or set(value) != {'schema', 'repo', 'model', 'device', 'binary_input'}
            or type(value.get('schema')) is not int or value['schema'] != 1):
        raise ValueError('run configuration schema differs')
    for name in ('repo', 'model', 'device', 'binary_input'):
        entry = value[name]
        if type(entry) is not str or not entry or '\x00' in entry:
            raise ValueError('run configuration path differs: ' + name)
        path = Path(entry)
        if not path.is_absolute() or str(path) != entry or path.resolve() != path:
            raise ValueError('run configuration path is not canonical: ' + name)
    return value


def repo_path(root=ROOT):
    return Path(run_config(root)['repo'])


def model_path(root=ROOT):
    return Path(run_config(root)['model'])


def write_json(path, value):
    with path.open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def client_identity(root=ROOT):
    """UI artifacts have a distinct contract; legacy provider snapshots stay unchanged."""
    if approved_contract(root)['profile'] in UI_PROFILES:
        try:
            from .ui_client import client_identity as identity
        except ImportError:
            from ui_client import client_identity as identity
    else:
        try:
            from .antimatter import client_identity as identity
        except ImportError:
            from antimatter import client_identity as identity
    return identity(root)


def validate_ui_client_host(profile, value='separate'):
    """Topology is a finite UI-only preparation choice, never inferred at runtime."""
    if type(value) is not str or value not in ('same', 'separate'):
        raise ValueError('UI client host must be same or separate')
    if profile not in UI_PROFILES and value != 'separate':
        raise ValueError('UI client host selection requires the UI conversation profile')
    return value


def approved_contract(root=ROOT):
    contract = read_json(root / 'qualification-contract.json')
    if contract.get('status') != 'approved_before_execution':
        raise ValueError('answer fixtures and bounds must be approved before execution')
    spec = profile_spec(contract.get('profile'))
    ui_client_host = validate_ui_client_host(spec['profile'], contract.get('ui_client_host')
        if spec['profile'] in UI_PROFILES else contract.get('ui_client_host', 'separate'))
    phase_diagnostics(contract)
    if (type(contract.get('stream_client')) is not str
            or contract['stream_client'] not in ('python-http', 'antimatter', 'antimatter-ui')
            or ((contract['stream_client'] == 'antimatter-ui') != (spec['profile'] in UI_PROFILES))
            or (contract['stream_client'] == 'antimatter' and spec['profile'] not in CLIENT_PROFILES)
            or (spec['profile'] in (*GROUNDED_PROFILES, GENERATED_HISTORY_PROFILE) and contract['stream_client'] != 'antimatter')):
        raise ValueError('preregistered stream client differs')
    if fixture_profile(read_json(root / 'answer-fixtures.json')) != spec['profile']:
        raise ValueError('fixture document differs from the declared qualification profile')
    expected = {'qualification_kind': spec['qualification_kind'],
                'require_semantic_match': spec['require_semantic_match'],
                'capacity': 128, 'max_new_tokens': 80, 'maximum_prompt_tokens': 49,
                'hag_charge_bytes': 681910272, 'runtime_combined_forecast_bytes': 1556061376,
                'process_memory_max_bytes': MEMORY_LIMIT, 'process_early_stop_bytes': STOP_CGROUP_BYTES,
                'swap_max_bytes': 0, 'cpu_quota_percent': 75, 'tasks_max': 64,
                'thermal_cutoff_c': 90, 'child_timeout_seconds': CHILD_SECONDS,
                'service_timeout_seconds': SERVICE_SECONDS, 'outer_timeout_seconds': OUTER_SECONDS}
    if spec['profile'] in GROUNDED_PROFILES:
        expected.update(capacity=512, max_new_tokens=128, maximum_prompt_tokens=93,
                        exact_prompt_tokens=93, hag_charge_bytes=760778752,
                        runtime_combined_forecast_bytes=1541609664,
                        runtime_forecast_policy='standalone_retained_metadata_upper_bound',
                        timing_initial_nonstream_requests=1, timing_measured_warm_requests=1,
                        client_request_timeout_seconds=60, client_process_timeout_seconds=65,
                        client_nonstream=True, required_client_max_evaluated_tokens=512,
                        overflow_max_tokens=421, cold_nonstream_refusals=1, warm_stream_refusals=1,
                        model_free_client_refusals_per_process=1, live_cancellation_qualified=False)
    if spec['profile'] in C2048_PROFILES:
        for key in ('child_timeout_seconds', 'service_timeout_seconds', 'outer_timeout_seconds'):
            expected.pop(key)
            if key in contract:
                raise ValueError('C2048 deadlines must be declared per stage')
        expected.update(capacity=2048,
                        hag_charge_bytes=1135140864, hag_budget_bytes=1152 * 1024 ** 2,
                        cpu_model_budget_bytes=1536 * 1024 ** 2,
                        runtime_combined_forecast_bytes=1915984064,
                        runtime_actual_metadata_forecast_bytes=1914935488,
                        private_cgroup_admission=True, crash_watcher_seconds=1500,
                        cpu_request_timeout_seconds=600,
                        thermal_start_max_c=60, thermal_start_stable_seconds=10)
        deadlines = contract.get('stage_deadlines')
        registered = {mode: dict(zip(('child', 'service', 'outer'), values))
                      for mode, values in profile_deadlines(spec['profile']).items()}
        if (type(deadlines) is not dict or set(deadlines) != set(registered)
                or any(type(deadlines[mode]) is not dict or set(deadlines[mode]) != set(values)
                       or any(type(deadlines[mode][key]) is not int or deadlines[mode][key] != value
                              for key, value in values.items())
                       for mode, values in registered.items())):
            raise ValueError('preregistered stage deadlines differ')
    if spec['profile'] == PROJECT_PROFILE:
        expected.update(maximum_prompt_tokens=467, exact_prompt_tokens=467,
                        client_request_timeout_seconds=240, client_process_timeout_seconds=245,
                        required_client_max_evaluated_tokens=2048, overflow_max_tokens=1583)
    if spec['profile'] == HISTORY_PROFILE:
        expected.update(maximum_prompt_tokens=495, exact_prompt_tokens=495,
                        client_request_timeout_seconds=240, client_process_timeout_seconds=245,
                        required_client_max_evaluated_tokens=2048, overflow_max_tokens=1555)
    if spec['profile'] == TEMPLATE_HISTORY_PROFILE:
        expected.update(maximum_prompt_tokens=491, exact_prompt_tokens=491,
                        client_request_timeout_seconds=240, client_process_timeout_seconds=245,
                        required_client_max_evaluated_tokens=2048, overflow_max_tokens=1559)
    if spec['profile'] == LIFECYCLE_PROFILE:
        if any(key.startswith(('timing_', 'client_', 'required_client_', 'model_free_client_'))
               or key == 'no_timing_speed_threshold' for key in contract):
            raise ValueError('lifecycle profile does not declare timing or client workloads')
        expected.update(max_new_tokens=1, maximum_prompt_tokens=9, exact_prompt_tokens=9,
                        cancel_max_tokens=16, maximum_cancel_evaluated_positions=24,
                        require_eos_stop=False, require_complete_final_answer=False,
                        live_cancellation_qualified=True, cpu_before_native=True,
                        exact_cpu_text_usage_finish_parity=True,
                        runtime_forecast_policy='standalone_retained_metadata_upper_bound',
                        no_new_privileges=True, host_start_headroom_bytes=START_HEADROOM,
                        host_stop_headroom_bytes=STOP_HEADROOM, thermal_poll_seconds=0.1)
    if spec['profile'] == GENERATED_HISTORY_PROFILE:
        if any(key.startswith('timing_') or key in ('no_timing_speed_threshold', 'overflow_max_tokens',
               'cold_nonstream_refusals', 'warm_stream_refusals') for key in contract):
            raise ValueError('generated conversation has no timing or extra refusal workload')
        expected.update(max_new_tokens=128, maximum_prompt_tokens=491, exact_prompt_tokens=None,
                        first_prompt_tokens=467, second_maximum_prompt_tokens=491,
                        client_request_timeout_seconds=240, client_process_timeout_seconds=245,
                        client_nonstream=True, required_client_max_evaluated_tokens=2048,
                        client_chat_requests=2, client_stream_requests=0, generated_history_required=True,
                        history_source='same-backend-raw-first-answer', prompt_inspector_required=True,
                        live_cancellation_qualified=False, cpu_before_native=True,
                        exact_cpu_text_usage_finish_parity=True, require_eos_stop=True,
                        require_complete_final_answer=True,
                        runtime_forecast_policy='standalone_retained_metadata_upper_bound',
                        no_new_privileges=True, host_start_headroom_bytes=START_HEADROOM,
                        host_stop_headroom_bytes=STOP_HEADROOM, thermal_poll_seconds=0.1)
        turns = contract.get('turn_evaluation_ceilings')
        if type(turns) is not list or any(type(value) is not int for value in turns) or turns != [594, 618]:
            raise ValueError('generated conversation per-turn bounds differ')
    if spec['profile'] == UI_HISTORY_PROFILE:
        if any(key.startswith('timing_') or key in ('no_timing_speed_threshold', 'overflow_max_tokens',
               'cold_nonstream_refusals', 'warm_stream_refusals') for key in contract):
            raise ValueError('UI conversation has no timing or refusal workload')
        expected.update(max_new_tokens=128, maximum_prompt_tokens=491, exact_prompt_tokens=None,
                        first_prompt_tokens=467, second_maximum_prompt_tokens=491,
                        client_request_timeout_seconds=240, client_process_timeout_seconds=355,
                        client_nonstream=False, required_client_max_evaluated_tokens=2048,
                        client_chat_requests=0, client_stream_requests=2, generated_history_required=True,
                        history_source='same-backend-raw-first-answer', prompt_inspector_required=True,
                        live_cancellation_qualified=False, cpu_before_native=True,
                        exact_cpu_text_usage_finish_parity=True, require_eos_stop=True,
                        require_complete_final_answer=True,
                        runtime_forecast_policy='standalone_retained_metadata_upper_bound',
                        no_new_privileges=True, host_start_headroom_bytes=START_HEADROOM,
                        host_stop_headroom_bytes=STOP_HEADROOM, thermal_poll_seconds=0.1,
                        wire_stream=True, wire_temperature='omitted', ui_client_host=ui_client_host,
                        broker_delivery='buffered-sse-after-parity-reset', ui_closeout_required=True)
        turns = contract.get('turn_evaluation_ceilings')
        if type(turns) is not list or any(type(value) is not int for value in turns) or turns != [594, 618]:
            raise ValueError('UI conversation per-turn bounds differ')
    if spec['profile'] == UI_STOP_PROFILE:
        if any(key.startswith('timing_') or key in ('no_timing_speed_threshold', 'overflow_max_tokens',
               'cold_nonstream_refusals', 'warm_stream_refusals', 'history_source',
               'second_maximum_prompt_tokens') for key in contract):
            raise ValueError('UI stop/reuse has no history, timing or refusal workload')
        limits = profile_limits(UI_STOP_PROFILE)
        expected.update(max_new_tokens=64, maximum_prompt_tokens=limits['maximum_prompt_tokens'],
                        exact_prompt_tokens=None, first_prompt_tokens=limits['first_prompt_tokens'],
                        reuse_prompt_tokens=467, client_request_timeout_seconds=240,
                        client_process_timeout_seconds=355, client_nonstream=False,
                        required_client_max_evaluated_tokens=2048,
                        client_chat_requests=0, client_stream_requests=2, generated_history_required=False,
                        prompt_inspector_required=True, live_cancellation_qualified=True,
                        cpu_before_native=True, exact_cpu_text_usage_finish_parity=True,
                        require_eos_stop=False, require_complete_final_answer=False,
                        reuse_require_eos_stop=True, reuse_require_complete_final_answer=True,
                        fresh_conversation_reuse_required=True, same_server_reuse_required=True,
                        cancel_after_visible_numeric_content=True,
                        runtime_forecast_policy='standalone_retained_metadata_upper_bound',
                        no_new_privileges=True, host_start_headroom_bytes=START_HEADROOM,
                        host_stop_headroom_bytes=STOP_HEADROOM, thermal_poll_seconds=0.1,
                        wire_stream=True, wire_temperature='omitted', ui_client_host=ui_client_host,
                        broker_delivery='live-sse-stop-then-buffered-parity-reset', ui_closeout_required=True)
        for key, values in (('turn_max_new_tokens', [64, 16]),
                            ('turn_evaluation_ceilings', limits['turn_evaluation_ceilings'])):
            actual = contract.get(key)
            if type(actual) is not list or any(type(value) is not int for value in actual) or actual != values:
                raise ValueError('UI stop/reuse per-turn bounds differ: ' + key)
        if contract.get('cpu_parity_fixture_ids') != ['ui_stop_reuse']:
            raise ValueError('UI stop/reuse CPU parity must cover only the completed reuse reply')
    for key, value in expected.items():
        if type(contract.get(key)) is not type(value) or contract[key] != value:
            raise ValueError('preregistered answer contract differs: ' + key)
    ceilings = contract.get('stage_evaluation_ceilings')
    if (type(ceilings) is not dict or set(ceilings) != set(spec['stage_evaluation_ceilings'])
            or any(type(value) is not int for value in ceilings.values())
            or ceilings != spec['stage_evaluation_ceilings']):
        raise ValueError('profile stage evaluation ceilings differ')
    counts = contract.get('stage_request_counts')
    if (type(counts) is not dict or set(counts) != set(spec['stage_request_counts'])
            or any(type(value) is not int for value in counts.values())
            or counts != spec['stage_request_counts']):
        raise ValueError('profile stage request counts differ')
    if contract.get('fixture_ids') != spec['fixture_ids']:
        raise ValueError('profile fixture IDs or order differ')
    return contract


def provenance_identity(root=ROOT):
    manifest = read_json(root / 'expected-manifest.json')
    value = manifest.get('input_provenance')
    if (type(value) is not dict or set(value) != {'git_head', 'git_dirty', 'binary_source_relation',
                                               'binary_sha256', 'source_manifest_sha256'}
            or type(value.get('git_head')) is not str
            or not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', value['git_head'])
            or type(value.get('git_dirty')) is not bool
            or value.get('binary_source_relation') != 'not_verified_by_runner'):
        raise ValueError('observed input provenance differs')
    require_sha(value['binary_sha256'])
    require_sha(value['source_manifest_sha256'])
    if (manifest.get('binaries') != {'beholdd': value['binary_sha256']}
            or manifest.get('source_manifest_sha256') != value['source_manifest_sha256']):
        raise ValueError('input provenance identities differ from manifest')
    return value


def digest(path, limit=MAX_HASH_BYTES):
    value = hashlib.sha256()
    total = 0
    with regular_reader(path, limit) as stream:
        while chunk := stream.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ValueError(f'input exceeded its byte limit: {path}')
            value.update(chunk)
    return value.hexdigest()


def require_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('expected a lowercase SHA-256 digest')
    return value


def binary_identity(mode, root=ROOT):
    if mode not in BINARIES:
        raise ValueError('unknown native gate mode')
    expected = require_sha(read_json(root / 'expected-manifest.json').get('binaries', {}).get(BINARIES[mode]))
    if digest(root / BINARIES[mode]) != expected:
        raise ValueError('frozen gate binary SHA-256 differs from the manifest')
    return expected


def compiled_source_paths(repo=None):
    repo = repo_path() if repo is None else Path(repo)
    paths = [repo / name for name in ('Cargo.toml', 'Cargo.lock')]
    for base in (repo / 'crates', repo / 'vendor/exec-hag'):
        if not base.is_dir() or base.is_symlink():
            raise ValueError('compiled source tree is missing or a symlink')
        for path in base.rglob('*'):
            if path.is_symlink():
                raise ValueError('compiled source tree contains a symlink')
            if path.is_file() and (path.suffix in ('.rs', '.s', '.jinja') or path.name in ('Cargo.toml', 'Cargo.lock')):
                paths.append(path)
                if len(paths) > MAX_SOURCE_FILES:
                    raise ValueError('compiled source inventory exceeds its limit')
    total = 0
    for path in paths:
        with regular_reader(path, MAX_SOURCE_FILE_BYTES) as stream:
            total += os.fstat(stream.fileno()).st_size
        if total > MAX_SOURCE_TOTAL_BYTES:
            raise ValueError('compiled source snapshot exceeds its byte limit')
    return sorted(set(paths))


def source_identity(root=ROOT, repo=None):
    repo = repo_path(root) if repo is None else Path(repo)
    manifest = read_json(root / 'expected-manifest.json')
    source_path = root / 'build-source-sha256.json'
    expected = require_sha(manifest.get('source_manifest_sha256'))
    if digest(source_path) != expected:
        raise ValueError('source manifest SHA-256 differs')
    sources = read_json(source_path)
    if (not isinstance(sources, dict) or not sources
            or type(manifest.get('source_files')) is not int
            or not 1 <= manifest['source_files'] <= MAX_SOURCE_FILES
            or len(sources) != manifest['source_files']):
        raise ValueError('source manifest count or shape differs')
    inventory = {str(path.relative_to(repo)) for path in compiled_source_paths(repo)}
    if set(sources) != inventory:
        raise ValueError('compiled source inventory differs from the frozen manifest')
    for name, sha in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError('source manifest path is not a canonical relative path')
        require_sha(sha)
        for base in (root / 'build-source', repo):
            if digest(base / relative) != sha:
                raise ValueError(f'frozen/current source differs: {name}')
    return expected


def gate_identity(root=ROOT):
    manifest = read_json(root / 'expected-manifest.json')
    expected = manifest.get('gate_files')
    if type(expected) is not dict or set(expected) != set(GATE_FILES):
        raise ValueError('frozen qualification helper/input inventory differs')
    for name, sha in expected.items():
        require_sha(sha)
        path = root / name
        if path.is_symlink() or digest(path) != sha:
            raise ValueError('qualification helper/input differs: ' + name)
    fixtures = {name: expected[name] for name in ('answer-fixtures.json', 'cpu-config.json', 'hag-config.json')}
    if manifest.get('answer_fixtures') != fixtures:
        raise ValueError('answer fixture identity differs')
    if manifest.get('profile') != approved_contract(root)['profile']:
        raise ValueError('manifest qualification profile differs from the contract')
    return expected


def prompt_inspector_identity(path):
    """Bound a native metadata inspector without ever executing it."""
    path = Path(path)
    if not os.access(path, os.X_OK):
        raise ValueError('prompt inspector must be executable')
    with regular_reader(path, 256 * 1024 ** 2) as stream:
        header = stream.read(20)
    if (len(header) != 20 or header[:6] != b'\x7fELF\x02\x01'
            or int.from_bytes(header[18:20], 'little') != 62):
        raise ValueError('prompt inspector must be a native x86-64 ELF executable')
    return digest(path, 256 * 1024 ** 2)


def dependency_identity(root=ROOT):
    manifest = read_json(root / 'expected-manifest.json')
    dependencies = manifest.get('dependencies', {})
    names = ('thermal_guard.py', 'hag-crash-watch.py')
    if manifest.get('profile') in (GENERATED_HISTORY_PROFILE, *UI_PROFILES):
        names += ('prompt-inspect',)
    if type(dependencies) is not dict or set(dependencies) != set(names):
        raise ValueError('qualification dependency inventory differs')
    for name in names:
        expected = require_sha(dependencies.get(name))
        path = root / 'dependencies' / name
        observed = prompt_inspector_identity(path) if name == 'prompt-inspect' else digest(path)
        if observed != expected:
            raise ValueError(f'qualification dependency differs: {name}')
    return {name: dependencies[name] for name in names}


def checkpoint_identity(root=ROOT, model=None):
    model = model_path(root) if model is None else Path(model)
    manifest = read_json(root / 'expected-manifest.json')
    if manifest.get('model_sha256') != MODEL_SHA256 or manifest.get('model_bytes') != MODEL_BYTES:
        raise ValueError('checkpoint manifest is not the qualified Qwen3 checkpoint')
    if model.stat().st_size != MODEL_BYTES or digest(model) != MODEL_SHA256:
        raise ValueError('checkpoint bytes differ from the qualified checkpoint')
    return MODEL_SHA256


def available_memory(path=Path('/proc/meminfo')):
    for line in path.read_text().splitlines():
        if line.startswith('MemAvailable:'):
            fields = line.split()
            if len(fields) != 3 or fields[2] != 'kB' or not fields[1].isdigit():
                break
            return int(fields[1]) * 1024
    raise ValueError('MemAvailable is missing or invalid')


def require_headroom(available, minimum=START_HEADROOM):
    if available < minimum:
        raise ValueError(f'host memory headroom too low: {available} bytes available, {minimum} required')


def cgroup_path(text, base=Path('/sys/fs/cgroup')):
    records = [line[3:] for line in text.splitlines() if line.startswith('0::')]
    if len(records) != 1 or not records[0].startswith('/'):
        raise ValueError('one absolute unified cgroup is required')
    relative = Path(records[0].lstrip('/'))
    if '..' in relative.parts:
        raise ValueError('invalid cgroup path')
    return base / relative


def memory_snapshot(cgroup):
    def number(name):
        value = int((cgroup / name).read_text().strip())
        if value < 0:
            raise ValueError('negative cgroup memory counter')
        return value
    events = dict((key, int(value)) for key, value in
                  (line.split() for line in (cgroup / 'memory.events').read_text().splitlines()))
    if any(events.get(name, -1) < 0 for name in ('high', 'max', 'oom', 'oom_kill')):
        raise ValueError('missing or negative cgroup memory event counter')
    result = {'memory_current_bytes': number('memory.current'), 'memory_peak_bytes': number('memory.peak'),
              'swap_current_bytes': number('memory.swap.current'), 'memory_events': events}
    try:
        result['swap_peak_bytes'] = number('memory.swap.peak')
    except FileNotFoundError:
        result['swap_peak_bytes'] = None
    return result


def verify_limits(actual, status, trace):
    if actual != EXPECTED_LIMITS or not re.search(r'^NoNewPrivs:\s+1$', status, re.MULTILINE):
        raise ValueError(f'actual cgroup/no-new-privileges limits differ: {actual}')
    if trace != '0':
        raise ValueError('HAG_TRACE_SUBMISSION must be exactly 0')


def clean_memory(before, after):
    return (0 <= after['memory_peak_bytes'] <= MEMORY_LIMIT and after['swap_current_bytes'] == 0
            and after.get('swap_peak_bytes') in (None, 0)
            and all(after['memory_events'].get(name, 0) == before['memory_events'].get(name, 0) == 0
                    for name in ('high', 'max', 'oom', 'oom_kill', 'oom_group_kill')))


def require_process_headroom(snapshot):
    if snapshot['memory_current_bytes'] >= STOP_CGROUP_BYTES:
        raise ValueError('process memory reached the 1536-MiB early-stop threshold')


def stage_deadlines(mode, root=ROOT):
    if mode not in MODES:
        raise ValueError('unknown stage deadline')
    profile = approved_contract(root)['profile']
    if mode not in profile_modes(profile):
        raise ValueError('stage is not part of the declared profile')
    return profile_deadlines(profile)[mode]


def private_cgroup_attestation(cgroup, actual, status, trace, snapshot):
    """Only a verified dedicated service may attest its exact unified membership."""
    verify_limits(actual, status, trace)
    require_process_headroom(snapshot)
    if not clean_memory(snapshot, snapshot):
        raise ValueError('private cgroup already has pressure or swap events')
    cgroup = Path(cgroup)
    if (not cgroup.is_relative_to('/sys/fs/cgroup') or '..' in cgroup.parts
            or not re.fullmatch(r'hag-answers-(self-check|cpu-reference|answers|timing)-[0-9-]+\.service', cgroup.name)):
        raise ValueError('private cgroup is not this qualification service')
    return '/' + str(cgroup.relative_to('/sys/fs/cgroup'))
