"""Prepare a fresh answer gate without building or executing the supplied model."""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile

try:
    from . import gate_common as common
    from .profiles import fixture_document, fixture_profile, profile_spec, GROUNDED_PROFILE, GROUNDED_PROFILES, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, CLIENT_PROFILES, C2048_PROFILES, profile_limits, profile_deadlines
    from .antimatter import prepare_client, validate_options
except ImportError:
    import gate_common as common
    from profiles import fixture_document, fixture_profile, profile_spec, GROUNDED_PROFILE, GROUNDED_PROFILES, PROJECT_PROFILE, HISTORY_PROFILE, TEMPLATE_HISTORY_PROFILE, LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, CLIENT_PROFILES, C2048_PROFILES, profile_limits, profile_deadlines
    from antimatter import prepare_client, validate_options


PACKAGE_ROOT = Path(__file__).resolve().parent
GENERATED_FILES = {'cpu-config.json', 'hag-config.json', 'qualification-contract.json', 'run-config.json'}
MAX_HELPER_BYTES = 1024 * 1024
MAX_BINARY_BYTES = 256 * 1024 ** 2
MODEL_ID = 'hag-context-qwen3-0.6b'


def git_observation(repo):
    """Record Git metadata only; neither cleanliness nor HEAD binds an executable."""
    environment = dict(os.environ, GIT_OPTIONAL_LOCKS='0')

    def git(*arguments):
        with tempfile.TemporaryFile() as output:
            result = subprocess.run(['git', '-C', str(repo), *arguments], stdout=output,
                                    stderr=subprocess.DEVNULL, timeout=10, check=False,
                                    env=environment)
            output.seek(0)
            data = output.read(MAX_HELPER_BYTES + 1)
        if result.returncode != 0 or len(data) > MAX_HELPER_BYTES:
            raise ValueError('cannot read bounded repository Git metadata')
        return data.decode('utf-8', errors='strict').strip()

    if Path(git('rev-parse', '--show-toplevel')).resolve() != repo:
        raise ValueError('repository must be the canonical Git root')
    head = git('rev-parse', '--verify', 'HEAD')
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', head):
        raise ValueError('repository has no valid Git HEAD')
    return {'git_head': head, 'git_dirty': bool(git('status', '--porcelain=v1', '--untracked-files=normal'))}


def profile_contract(profile='list-c128', stream_client='python-http', *, runtime_phase_diagnostics=False,
                     ui_client_host='separate'):
    """The fixed, reviewed workload is registered before any stage may execute."""
    spec = profile_spec(profile)
    ui_client_host = common.validate_ui_client_host(profile, ui_client_host)
    diagnostics = common.phase_diagnostic_contract(profile, runtime_phase_diagnostics)
    if (type(stream_client) is not str or stream_client not in ('python-http', 'antimatter', 'antimatter-ui')
            or ((stream_client == 'antimatter-ui') != (profile in UI_PROFILES))
            or (stream_client == 'antimatter' and profile not in CLIENT_PROFILES)
            or (profile in (*GROUNDED_PROFILES, GENERATED_HISTORY_PROFILE) and stream_client != 'antimatter')):
        raise ValueError('unsupported stream client/profile combination')
    contract = {
        **spec,
        'stream_client': stream_client,
        'status': 'approved_before_execution',
        'scope': (('CPU/HAG output parity and EOS for the declared finite fixtures; '
                   'semantic outcomes are recorded without semantic-quality qualification. '
                   'Descriptive integer-list request timing.')
                  if not spec['require_semantic_match'] else
                  ('Complete-answer/EOS CPU/HAG parity for the declared finite fixtures '
                   'and descriptive integer-list request timing.')),
        'capacity': 128, 'max_new_tokens': 80, 'maximum_prompt_tokens': 49,
        'timing_initial_nonstream_requests': 1, 'timing_measured_warm_requests': 3,
        'cpu_before_native': True, 'exact_cpu_text_usage_finish_parity': True,
        'require_eos_stop': True, 'require_complete_final_answer': True,
        'no_timing_speed_threshold': True, 'hag_charge_bytes': 681910272,
        'runtime_combined_forecast_bytes': 1556061376, 'hag_budget_bytes': 805306368,
        'process_memory_max_bytes': common.MEMORY_LIMIT,
        'process_early_stop_bytes': common.STOP_CGROUP_BYTES, 'swap_max_bytes': 0,
        'cpu_quota_percent': 75, 'tasks_max': 64, 'no_new_privileges': True,
        'host_start_headroom_bytes': common.START_HEADROOM,
        'host_stop_headroom_bytes': common.STOP_HEADROOM,
        'thermal_cutoff_c': 90, 'thermal_poll_seconds': 0.1,
        'child_timeout_seconds': common.CHILD_SECONDS,
        'service_timeout_seconds': common.SERVICE_SECONDS,
        'outer_timeout_seconds': common.OUTER_SECONDS,
        'limitations': ('Only the supported Qwen3 Q8_0 checkpoint, C128 and the declared finite fixtures. '
                        'No general conversation, strict-format, concurrency or throughput qualification. '
                        'Binary/source build association is not verified by this runner.'),
    }

    if profile in GROUNDED_PROFILES:
        contract.update(
            scope='One frozen no-tools Antimatter system prompt, CPU/HAG raw answer and EOS parity, '
                  'committed client chat/stream reuse and generation-free capacity refusals. '
                  'Semantic quality and live cancellation are not qualified.',
            capacity=512, max_new_tokens=128, maximum_prompt_tokens=93, exact_prompt_tokens=93,
            timing_measured_warm_requests=1, client_request_timeout_seconds=60,
            client_process_timeout_seconds=65, client_nonstream=True,
            required_client_max_evaluated_tokens=512, overflow_max_tokens=421,
            cold_nonstream_refusals=1, warm_stream_refusals=1,
            model_free_client_refusals_per_process=1, live_cancellation_qualified=False,
            hag_charge_bytes=760778752, runtime_combined_forecast_bytes=1541609664,
            runtime_forecast_policy='standalone_retained_metadata_upper_bound',
            limitations='Only the frozen grounded no-tools prompt and explicit 128-token output allowance. '
                        'No full editor/tools, arbitrary conversations, semantic-quality, live cancellation '
                        'or throughput qualification. Binary/source association is not verified by this runner.')
    if profile in C2048_PROFILES:
        for key in ('child_timeout_seconds', 'service_timeout_seconds', 'outer_timeout_seconds'):
            contract.pop(key)
        contract.update(
            capacity=2048,
            hag_charge_bytes=1135140864, hag_budget_bytes=1152 * 1024 ** 2,
            cpu_model_budget_bytes=1536 * 1024 ** 2,
            runtime_combined_forecast_bytes=1915984064,
            runtime_actual_metadata_forecast_bytes=1914935488,
            private_cgroup_admission=True, crash_watcher_seconds=1500,
            cpu_request_timeout_seconds=600,
            thermal_start_max_c=60, thermal_start_stable_seconds=10,
            stage_deadlines={mode: dict(zip(('child', 'service', 'outer'), values))
                             for mode, values in profile_deadlines(profile).items()})
    if profile == PROJECT_PROFILE:
        contract.update(
            maximum_prompt_tokens=467, exact_prompt_tokens=467,
            client_request_timeout_seconds=240, client_process_timeout_seconds=245,
            required_client_max_evaluated_tokens=2048, overflow_max_tokens=1583,
            scope='Frozen Antimatter no-tools grounding plus synthetic project instructions, '
                  'CPU/HAG raw answer and EOS parity, provider chat/stream reuse and capacity refusals.',
            limitations='Only the declared 467-token no-tools project-instructions prompt and 128-token reply allowance. '
                        'No full editor/tools, arbitrary conversations, semantic-quality, live cancellation or throughput qualification. '
                        'Binary/source association is not verified by this runner.')
    if profile == HISTORY_PROFILE:
        contract.update(
            maximum_prompt_tokens=495, exact_prompt_tokens=495,
            client_request_timeout_seconds=240, client_process_timeout_seconds=245,
            required_client_max_evaluated_tokens=2048, overflow_max_tokens=1555,
            scope='Frozen no-tools project instructions with seeded first-turn history, committed '
                  'Antimatter history transmission and CPU/HAG second-turn raw answer and EOS parity.',
            limitations='Only the declared 495-token seeded-history request and 128-token reply allowance. '
                        'The first turn is fixed input, not a live conversation. No UI-history, full editor/tools, '
                        'semantic-quality, live cancellation, concurrency or throughput qualification. '
                        'Binary/source association is not verified by this runner.')
    if profile == TEMPLATE_HISTORY_PROFILE:
        contract.update(
            maximum_prompt_tokens=491, exact_prompt_tokens=491,
            client_request_timeout_seconds=240, client_process_timeout_seconds=245,
            required_client_max_evaluated_tokens=2048, overflow_max_tokens=1559,
            scope='Frozen no-tools project instructions with seeded first-turn history, checkpoint-selected '
                  'chat rendering, committed Antimatter history transmission and CPU/HAG second-turn raw answer and EOS parity.',
            limitations='Only the declared 491-token checkpoint-rendered seeded-history request and 128-token reply allowance. '
                        'The first turn is fixed input, not a live conversation. No UI-history, full editor/tools, '
                        'semantic-quality, live cancellation, concurrency or throughput qualification. '
                        'Binary/source association is not verified by this runner.')
    if profile == LIFECYCLE_PROFILE:
        for key in ('timing_initial_nonstream_requests', 'timing_measured_warm_requests',
                    'no_timing_speed_threshold'):
            contract.pop(key)
        contract.update(
            max_new_tokens=1, maximum_prompt_tokens=9, exact_prompt_tokens=9,
            cancel_max_tokens=16, maximum_cancel_evaluated_positions=24,
            require_eos_stop=False, require_complete_final_answer=False,
            live_cancellation_qualified=True,
            runtime_forecast_policy='standalone_retained_metadata_upper_bound',
            scope='One fixed Hello prompt with length-limited CPU/HAG parity, real TCP socket '
                  'cancellation, healthy reuse and same-process unload/reload.',
            limitations='Only the declared nine-token prompt, one-token successful replies and '
                        'bounded first-content cancellation. No complete-answer, semantic-quality, '
                        'Antimatter-provider, timing, concurrency, throughput or full-context qualification. '
                        'Binary/source association is not verified by this runner.')
    if profile == GENERATED_HISTORY_PROFILE:
        for key in ('timing_initial_nonstream_requests', 'timing_measured_warm_requests',
                    'no_timing_speed_threshold'):
            contract.pop(key)
        contract.update(
            max_new_tokens=128, maximum_prompt_tokens=491, exact_prompt_tokens=None,
            first_prompt_tokens=467, second_maximum_prompt_tokens=491,
            turn_evaluation_ceilings=[594, 618], client_request_timeout_seconds=240,
            client_process_timeout_seconds=245, client_nonstream=True,
            required_client_max_evaluated_tokens=2048, client_chat_requests=2, client_stream_requests=0,
            generated_history_required=True, history_source='same-backend-raw-first-answer',
            prompt_inspector_required=True, live_cancellation_qualified=False,
            runtime_forecast_policy='standalone_retained_metadata_upper_bound',
            scope='Two generated no-tools turns through committed Antimatter chat, with exact per-turn '
                  'CPU/HAG raw-text, usage and EOS parity. Each backend carries its own raw first answer into turn two.',
            limitations='Only this finite generated conversation and 128-token per-turn reply allowance. '
                        'No SSE, timing, semantic-quality, live cancellation, full editor/tools, concurrency '
                        'or throughput qualification. Binary/source association is not verified by this runner.')
    if profile == UI_HISTORY_PROFILE:
        for key in ('timing_initial_nonstream_requests', 'timing_measured_warm_requests',
                    'no_timing_speed_threshold'):
            contract.pop(key)
        contract.update(
            max_new_tokens=128, maximum_prompt_tokens=491, exact_prompt_tokens=None,
            first_prompt_tokens=467, second_maximum_prompt_tokens=491,
            turn_evaluation_ceilings=[594, 618], client_request_timeout_seconds=240,
            client_process_timeout_seconds=355, client_nonstream=False,
            required_client_max_evaluated_tokens=2048, client_chat_requests=0, client_stream_requests=2,
            generated_history_required=True, history_source='same-backend-raw-first-answer',
            prompt_inspector_required=True, live_cancellation_qualified=False,
            runtime_forecast_policy='standalone_retained_metadata_upper_bound',
            wire_stream=True, wire_temperature='omitted', ui_client_host=ui_client_host,
            broker_delivery='buffered-sse-after-parity-reset', ui_closeout_required=True,
            scope='Two actual editor submissions through a frozen local UI and a remote admission broker, '
                  'with current-wire CPU/native parity and healthy reset before releasing each reply.',
            limitations='Only the declared C2048/M128 two-turn no-tools conversation. Buffered SSE '
                        'delivery, no SSE timing, live cancellation, semantic-quality or full editor services. '
                        'Full qualification requires separate host-qualified UI process retirement evidence. '
                        'Binary/source association is not verified by this runner.')
        if ui_client_host == 'same':
            contract.update(
                scope='Two actual editor submissions through a frozen UI and a direct loopback admission broker '
                      'on the same host, with current-wire CPU/native parity and healthy reset before each reply.',
                limitations='Only the declared C2048/M128 two-turn no-tools conversation. Buffered SSE '
                            'delivery, no SSE timing, live cancellation, semantic-quality or full editor services. '
                            'The backend and UI retain independent process limits on one host; their hard memory '
                            'caps total 4 GiB, with separate HAG device accounting. Full qualification requires '
                            'host-qualified UI process retirement evidence. '
                            'Binary/source association is not verified by this runner.')
    if profile == UI_STOP_PROFILE:
        limits = profile_limits(profile)
        for key in ('timing_initial_nonstream_requests', 'timing_measured_warm_requests',
                    'no_timing_speed_threshold'):
            contract.pop(key)
        contract.update(
            max_new_tokens=64, maximum_prompt_tokens=limits['maximum_prompt_tokens'], exact_prompt_tokens=None,
            first_prompt_tokens=limits['first_prompt_tokens'], reuse_prompt_tokens=467,
            turn_max_new_tokens=[64, 16], turn_evaluation_ceilings=limits['turn_evaluation_ceilings'],
            client_request_timeout_seconds=240, client_process_timeout_seconds=355, client_nonstream=False,
            required_client_max_evaluated_tokens=2048, client_chat_requests=0, client_stream_requests=2,
            generated_history_required=False, prompt_inspector_required=True, live_cancellation_qualified=True,
            require_eos_stop=False, require_complete_final_answer=False,
            reuse_require_eos_stop=True, reuse_require_complete_final_answer=True,
            cpu_parity_fixture_ids=['ui_stop_reuse'], fresh_conversation_reuse_required=True,
            same_server_reuse_required=True, cancel_after_visible_numeric_content=True,
            runtime_forecast_policy='standalone_retained_metadata_upper_bound',
            wire_stream=True, wire_temperature='omitted', ui_client_host=ui_client_host,
            broker_delivery='live-sse-stop-then-buffered-parity-reset', ui_closeout_required=True,
            scope='One real UI Stop after numeric count-up content, followed by a fresh no-tools '
                  'conversation on the same server with exact CPU/native ready-answer, usage and EOS parity.',
            limitations='Only the declared C2048 count-up/M64 cancellation and ready/M16 reuse requests. '
                        'The cancelled answer has no complete-answer or CPU-output parity claim. '
                        'No generated-history, timing, semantic-quality, full editor services, concurrency '
                        'or throughput qualification. Full qualification requires imported host-qualified '
                        'UI process retirement evidence. Binary/source association is not verified by this runner.')
        if ui_client_host == 'same':
            contract['scope'] += ' UI and backend run in independently contained units on the same host.'
            contract['limitations'] += ' Their hard memory caps total 4 GiB, with separate HAG device accounting.'
    if diagnostics is not None:
        contract['runtime_phase_diagnostics'] = diagnostics
    return contract


def model_config(model, device, backend, profile='list-c128'):
    value = {'id': MODEL_ID, 'path': str(model), 'execution_backend': backend,
             'weight_mode': 'streaming', 'fit_policy': 'exact',
             'memory_budget_bytes': common.GIB if backend == 'cpu' else 2 * common.GIB,
             'runtime_headroom_bytes': 128 * 1024 ** 2}
    if backend == 'hag':
        value['hag'] = {'device': str(device), 'memory_budget_bytes': 768 * 1024 ** 2}
    if profile in C2048_PROFILES:
        if backend == 'cpu':
            value['memory_budget_bytes'] = 1536 * 1024 ** 2
        else:
            value['hag']['memory_budget_bytes'] = 1152 * 1024 ** 2
    return {'models': [value]}


def copy_frozen(source, target, expected, limit, *, executable=False):
    """Copy with constant memory, an exclusive destination, and a content bound."""
    total = 0
    observed = hashlib.sha256()
    with common.regular_reader(source, limit) as input_stream, target.open('xb') as output:
        while chunk := input_stream.read(1024 * 1024):
            total += len(chunk)
            if total > limit:
                raise ValueError('input grew beyond its copy limit: ' + str(source))
            observed.update(chunk)
            output.write(chunk)
    target.chmod(0o700 if executable else 0o600)
    if observed.hexdigest() != expected or common.digest(target, limit) != expected:
        raise ValueError('input changed while being frozen: ' + str(source))


def checkpoint(model):
    if model.stat().st_size != common.MODEL_BYTES or common.digest(model, common.MODEL_BYTES) != common.MODEL_SHA256:
        raise ValueError('checkpoint differs from the supported Qwen3 Q8_0 bytes')


def prepare(repo, binary, model, device, evidence, profile='list-c128', *,
            antimatter_repo=None, antimatter_commit=None, node=None, esbuild=None,
            runtime_phase_diagnostics=False, prompt_inspector=None, ui_client_identity=None,
            ui_client_host='separate'):
    """Freeze explicit inputs into a new directory; never run the binary or GPU.

    A failed preparation keeps its partial directory for inspection. The manifest
    is written last, so an incomplete directory cannot be used as a prepared run.
    """
    profile_spec(profile)
    common.validate_ui_client_host(profile, ui_client_host)
    if (profile in (GENERATED_HISTORY_PROFILE, *UI_PROFILES)) != (prompt_inspector is not None):
        raise ValueError('prompt inspector is required only for the generated or UI conversation profiles')
    if (profile in UI_PROFILES) != (ui_client_identity is not None):
        raise ValueError('UI identity is required only for the UI conversation profile')
    inspector = None if prompt_inspector is None else Path(prompt_inspector).expanduser()
    inspector_sha = None if inspector is None else common.prompt_inspector_identity(inspector)
    if profile in UI_PROFILES:
        if any(value is not None for value in (antimatter_repo, antimatter_commit, node, esbuild)):
            raise ValueError('UI identity cannot combine with legacy provider snapshot options')
        has_client = False
        try:
            from .ui_client import validate_identity
        except ImportError:
            from ui_client import validate_identity
        identity = validate_identity(common.read_json(Path(ui_client_identity)))
        if identity['profile'] != profile:
            raise ValueError('UI identity belongs to another qualification profile')
    else:
        has_client = validate_options(antimatter_repo, antimatter_commit, node, esbuild, profile)
    contract = profile_contract(profile, 'antimatter-ui' if profile in UI_PROFILES else
                                'antimatter' if has_client else 'python-http',
                                runtime_phase_diagnostics=runtime_phase_diagnostics,
                                ui_client_host=ui_client_host)
    raw_evidence = Path(evidence).expanduser()
    if raw_evidence.exists() or raw_evidence.is_symlink():
        raise ValueError('evidence output already exists; never overwrite a run')
    root = raw_evidence.parent.resolve(strict=True) / raw_evidence.name
    repo = Path(repo).expanduser().resolve(strict=True)
    binary = Path(binary).expanduser().resolve(strict=True)
    model = Path(model).expanduser().resolve(strict=True)
    device = Path(device).expanduser().resolve(strict=False)
    if not repo.is_dir() or not device.is_absolute():
        raise ValueError('repository and device paths must be canonical')
    if not os.access(binary, os.X_OK):
        raise ValueError('supplied CLI is not executable')
    checkpoint(model)
    observation = git_observation(repo)
    sources = {str(path.relative_to(repo)): common.digest(path, common.MAX_SOURCE_FILE_BYTES)
               for path in common.compiled_source_paths(repo)}
    binary_sha = common.digest(binary, MAX_BINARY_BYTES)
    package = PACKAGE_ROOT.resolve(strict=True)
    generated = GENERATED_FILES | ({'answer-fixtures.json'} if profile != 'list-c128' else set())
    if profile == 'list-c128' and fixture_profile(common.read_json(package / 'answer-fixtures.json')) != profile:
        raise ValueError('default package fixture differs from the list profile')
    scripts = {name: common.digest(package / name, MAX_HELPER_BYTES)
               for name in common.GATE_FILES if name not in generated}
    dependencies = {'thermal_guard.py': repo / 'tools/thermal_guard.py',
                    'hag-crash-watch.py': repo / 'tools/hag_crash_watch.py'}
    if inspector is not None:
        dependencies['prompt-inspect'] = inspector.resolve(strict=True)
    dependency_hashes = {name: common.digest(path, MAX_BINARY_BYTES if name == 'prompt-inspect' else MAX_HELPER_BYTES)
                         for name, path in dependencies.items()}

    root.mkdir(mode=0o700)
    config = {'schema': 1, 'repo': str(repo), 'model': str(model),
              'device': str(device), 'binary_input': str(binary)}
    common.write_json(root / 'run-config.json', config)
    common.write_json(root / 'cpu-config.json', model_config(model, device, 'cpu', profile))
    common.write_json(root / 'hag-config.json', model_config(model, device, 'hag', profile))
    common.write_json(root / 'qualification-contract.json', contract)
    if profile != 'list-c128':
        common.write_json(root / 'answer-fixtures.json', fixture_document(profile))
    for name, expected in scripts.items():
        copy_frozen(package / name, root / name, expected, MAX_HELPER_BYTES)
    (root / 'dependencies').mkdir()
    for name, source in dependencies.items():
        copy_frozen(source, root / 'dependencies' / name, dependency_hashes[name],
                    MAX_BINARY_BYTES if name == 'prompt-inspect' else MAX_HELPER_BYTES,
                    executable=name == 'prompt-inspect')
    copy_frozen(binary, root / 'beholdd', binary_sha, MAX_BINARY_BYTES, executable=True)
    (root / 'build-source').mkdir()
    for name, expected in sources.items():
        target = root / 'build-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        copy_frozen(repo / name, target, expected, common.MAX_SOURCE_FILE_BYTES)
    common.write_json(root / 'build-source-sha256.json', sources)
    source_sha = common.digest(root / 'build-source-sha256.json')
    gate_hashes = {name: common.digest(root / name, MAX_HELPER_BYTES) for name in common.GATE_FILES}
    client = prepare_client(antimatter_repo, antimatter_commit, node, esbuild, root) if has_client else None
    if ui_client_identity is not None:
        (root / 'client').mkdir()
        identity_sha = common.digest(Path(ui_client_identity), MAX_HELPER_BYTES)
        copy_frozen(Path(ui_client_identity), root / 'client/identity.json', identity_sha, MAX_HELPER_BYTES)
        frozen_identity = validate_identity(common.read_json(root / 'client/identity.json'))
        if frozen_identity['profile'] != profile:
            raise ValueError('frozen UI identity belongs to another qualification profile')
        client = {'kind': 'antimatter-ui', 'identity_sha256': identity_sha}

    # Reject changed sources/helpers/executable/checkpoint before sealing inputs.
    current = {str(path.relative_to(repo)): common.digest(path, common.MAX_SOURCE_FILE_BYTES)
               for path in common.compiled_source_paths(repo)}
    if current != sources:
        raise ValueError('compiled source changed during preparation')
    if (common.digest(binary, MAX_BINARY_BYTES) != binary_sha
            or {name: common.digest(package / name, MAX_HELPER_BYTES) for name in scripts} != scripts
            or {name: common.digest(path, MAX_BINARY_BYTES if name == 'prompt-inspect' else MAX_HELPER_BYTES)
                         for name, path in dependencies.items()} != dependency_hashes):
        raise ValueError('binary or helper changed during preparation')
    if inspector is not None and (dependency_hashes['prompt-inspect'] != inspector_sha
            or common.prompt_inspector_identity(root / 'dependencies/prompt-inspect') != inspector_sha):
        raise ValueError('prompt inspector changed during preparation')
    checkpoint(model)
    common.approved_contract(root)
    manifest = {
        'schema': 1, 'profile': profile, 'binaries': {'beholdd': binary_sha},
        'source_manifest_sha256': source_sha, 'source_files': len(sources),
        'model_sha256': common.MODEL_SHA256, 'model_bytes': common.MODEL_BYTES,
        'dependencies': dependency_hashes, 'gate_files': gate_hashes,
        'answer_fixtures': {name: gate_hashes[name] for name in
                            ('answer-fixtures.json', 'cpu-config.json', 'hag-config.json')},
        'input_provenance': {**observation, 'binary_source_relation': 'not_verified_by_runner',
                             'binary_sha256': binary_sha, 'source_manifest_sha256': source_sha},
    }
    if client is not None:
        manifest['client'] = client
    common.write_json(root / 'expected-manifest.json', manifest)
    return manifest
