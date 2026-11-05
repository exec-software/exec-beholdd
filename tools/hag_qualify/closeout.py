#!/usr/bin/env python3
"""Audit a stopped named serving qualification; never executes GPU work."""
from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from gate_common import (ROOT, MODES, GPU_MODES, EXPECTED_LIMITS, STOP_HEADROOM, STOP_CGROUP_BYTES,
                         binary_identity, checkpoint_identity, dependency_identity, digest,
                         gate_identity, read_json, source_identity, strict_json, write_json,
                         clean_memory, provenance_identity, approved_contract, require_phase_diagnostics)
from antimatter import SOURCE_PATHS, GROUNDED_SOURCE_PATHS
from gate_common import client_identity
from profiles import (profile_spec, profile_limits, GROUNDED_PROFILE, GROUNDED_PROFILES, PROJECT_PROFILE,
                      LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, UI_PROFILES, C2048_PROFILES, profile_deadlines, profile_modes)

PATTERN = r'amdgpu|\bdrm\b|gpu|ring.*timeout|vm.*fault|reset|oom|out of memory|Killed process'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(path, limit=16 * 1024 * 1024):
    with path.open('rb') as stream:
        value = stream.read(limit + 1)
    require(len(value) <= limit, 'oversized closeout evidence: ' + str(path))
    return value.decode('utf-8')


def instant(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'evidence timestamp has no timezone')
    return result.astimezone(timezone.utc)


def stopped_capture(root=ROOT):
    records = [strict_json(line) for line in text(root / 'capture.log').splitlines() if line.startswith('{')]
    require(len(records) >= 3 and records[0].get('event') == 'ARMED'
            and records[-1].get('event') == 'STOPPED', 'capture is not fully stopped')
    pid = records[0].get('pid')
    require(type(pid) is int and pid > 0, 'capture PID is invalid')
    require(records[-1].get('status') == 'stopped', 'capture ended without the explicit clean stop')
    require([r.get('event') for r in records].count('ARMED') == 1
            and [r.get('event') for r in records].count('STOPPED') == 1,
            'capture contains repeated start/stop events')
    previous = None
    for record in records:
        require(record.get('event') in ('ARMED', 'HEARTBEAT', 'STOPPED')
                and type(record.get('captured')) is int and record['captured'] == 0
                and type(record.get('pid')) is int and record['pid'] == pid,
                'capture recorded a fault, dump, or mixed session')
        now = instant(record['timestamp'])
        require(previous is None or now >= previous, 'capture timestamps are unordered')
        previous = now
    require(not Path(f'/proc/{pid}').exists(), 'capture process is still present; stop it before closeout')
    directory = Path(records[-1]['output_dir'])
    require(str(directory) == records[0]['output_dir'] and directory.is_absolute()
            and directory.parent == Path('/var/tmp') and directory.name.startswith('hag-crash-')
            and not directory.is_symlink() and directory.is_dir(), 'capture output directory differs or is unsafe')
    require(directory.stat().st_uid == 0 and directory.stat().st_mode & 0o007 == 0,
            'capture output directory is not private and root-owned')
    require(not list(directory.iterdir()), 'capture output directory contains a dump')
    return records


def validate_gate(mode, result, runner, session, start, end, identities, root=ROOT):
    require(mode in profile_modes(identities['profile']), 'stage is not admitted by this profile')
    client = identities.get('client_identity')
    require((client is None and 'client_identity' not in result)
            or (client is not None and result.get('client_identity') == client),
            mode + ' optional client identity differs')
    require(result.get('mode') == mode and result.get('qualified') is True
            and result.get('failures') == [] and result.get('capture_ok') is True,
            mode + ' did not qualify cleanly')
    require(type(result.get('exit_code')) is int and result['exit_code'] == 0
            and type(result.get('journal_exit_code')) is int and result['journal_exit_code'] == 0
            and result.get('driver_messages') == [], mode + ' service/kernel evidence failed')
    require(result.get('gate_files') == identities['gate_files']
            and result.get('dependencies') == identities['dependencies']
            and result.get('profile') == identities['profile']
            and result.get('qualification_kind') == identities['qualification_kind']
            and result.get('require_semantic_match') is identities['require_semantic_match']
            and result.get('quality_is_diagnostic') is (not identities['require_semantic_match'])
            and result.get('input_provenance') == identities['input_provenance'], mode + ' helper/provenance identity differs')
    stage_start, stage_end = instant(result['start']), instant(result['end'])
    require(stage_start <= stage_end, mode + ' timestamps are reversed')
    require(type(result.get('minimum_available_bytes')) is int
            and result['minimum_available_bytes'] >= STOP_HEADROOM
            and type(result.get('cgroup_memory_peak_bytes')) is int
            and 0 <= result['cgroup_memory_peak_bytes'] < STOP_CGROUP_BYTES,
            mode + ' memory supervision limits differ')
    child = read_json(root / (mode + '-child-result.json'))
    contract = approved_contract(root)
    require_phase_diagnostics(result, contract, mode)
    require_phase_diagnostics(child, contract, mode)
    require((client is None and 'client_identity' not in child)
            or (client is not None and child.get('client_identity') == client),
            mode + ' child optional client identity differs')
    require(child == result.get('child') and child.get('failure') is None
            and type(child.get('exit_code')) is int and child['exit_code'] == 0
            and child.get('memory_clean') is True and child.get('limits') == EXPECTED_LIMITS
            and child.get('no_new_privileges') is True
            and child.get('gate_files') == identities['gate_files']
            and child.get('dependencies') == identities['dependencies']
            and child.get('input_provenance') == identities['input_provenance'], mode + ' child evidence differs')
    if identities['profile'] in C2048_PROFILES:
        runner.validate_project_child(child, mode, identities['profile'])
        require(result.get('stage_deadlines') == child['stage_deadlines']
                and result.get('private_cgroup_attestation') == child['private_cgroup_attestation'],
                mode + ' private containment receipt differs')
        runner.validate_cooldown(read_json(root / (mode + '-start-cooldown.json')))
    require(clean_memory(child['memory_before'], child['memory_after']), mode + ' memory events or swap are not clean')
    if mode != 'self-check':
        require(result.get('binary_sha256') == identities['binaries'][mode]
                and result.get('source_manifest_sha256') == identities['source']
                and result.get('checkpoint_sha256') == identities['checkpoint'], mode + ' frozen execution identity differs')
        require(child.get('binary_sha256') == identities['binaries'][mode]
                and child.get('source_manifest_sha256') == identities['source'], mode + ' child artifact identity differs')
    else:
        require(result.get('capture_session') is None and result.get('binary_sha256') is None
                and result.get('source_manifest_sha256') is None
                and result.get('checkpoint_sha256') is None, 'self-check contained native execution identity')
    if mode in GPU_MODES:
        if identities['profile'] in C2048_PROFILES:
            cpu = read_json(root / 'cpu-reference-result.json')
            require(read_json(root / (mode + '-cpu-retirement.json')) == runner.require_retired_cpu(cpu, session, root),
                    mode + ' CPU retirement admission receipt differs')
        require(start <= stage_start <= stage_end <= end,
                mode + ' ran outside the stopped capture interval')
        require(result.get('capture_session') == session, mode + ' used a different capture session')
    else:
        require(result.get('capture_session') is None, mode + ' unexpectedly claims GPU crash capture')
    log = runner.bounded_text(root / (mode + '.log'))
    require(runner.probe_evidence(mode, log, root) == result.get('probe'), mode + ' report differs from its log')
    require(runner.thermal_evidence(runner.bounded_text(root / (mode + '-thermal.jsonl'), 16 * 1024 ** 2 if identities['profile'] in C2048_PROFILES else 2 * 1024 ** 2), log)
            == result.get('thermal'), mode + ' thermal evidence differs')


def validate_order(reports, profile='list-c128'):
    modes = profile_modes(profile)
    require(set(reports) == set(modes), 'stage inventory differs from this profile')
    for earlier, later in zip(modes, modes[1:]):
        require(instant(reports[earlier]['end']) <= instant(reports[later]['start']),
                later + ' started before ' + earlier + ' ended')


def client_artifacts(profile):
    if profile == GENERATED_HISTORY_PROFILE:
        from generated_history import client_artifacts as generated_artifacts
        return generated_artifacts()
    if profile in GROUNDED_PROFILES:
        fixture_id = profile_spec(profile)['fixture_ids'][0]
        return [('answers','answers-' + fixture_id,'answers-' + fixture_id + '-response.json'),
                ('timing','timing-warmup','timing-warmup-response.json'),
                ('timing','timing-round-1','timing-round-1-sse.json')]
    return [('timing','timing-round-' + str(index),'timing-round-' + str(index) + '-sse.json')
            for index in range(1, profile_limits(profile)['measured_requests'] + 1)]


def closed_evidence_names(root=ROOT):
    contract = approved_contract(root)
    modes = profile_modes(contract['profile'])
    limits = profile_limits(contract['profile'])
    names = ['capture.log', 'capture.stop', 'cpu-reference.json', 'final-cleanup-check.json']
    # Bind the full retained responses and receipt timestamps to final closeout,
    # not only the distilled PASS reports used by the outer supervisor.
    if contract['profile'] == LIFECYCLE_PROFILE:
        from lifecycle import evidence_names
        for mode in modes:
            names.extend(evidence_names(mode))
    elif contract['profile'] == GENERATED_HISTORY_PROFILE:
        from generated_history import evidence_names
        for mode in modes:
            names.extend(evidence_names(mode))
    elif contract['profile'] == UI_HISTORY_PROFILE:
        from ui_history import evidence_names
        for mode in modes:
            names.extend(evidence_names(mode))
        from ui_client import evidence_names as ui_client_evidence_names
        names.extend(ui_client_evidence_names())
    elif contract['profile'] == UI_STOP_PROFILE:
        from ui_stop import evidence_names
        for mode in modes:
            names.extend(evidence_names(mode))
        from ui_client import evidence_names as ui_client_evidence_names
        names.extend(ui_client_evidence_names())
    else:
        for stage in ('cpu-reference', 'answers', 'timing'):
            names.append(stage + '-report.json')
            names.append(stage + ('-cpu-server.log' if stage == 'cpu-reference' else '-hag-server.log'))
        for stage in ('cpu-reference', 'answers'):
            names.extend(stage + '-' + fixture_id + '-response.json' for fixture_id in contract['fixture_ids'])
        names.append('timing-warmup-response.json')
        names.extend('timing-round-' + str(index) + '-sse.json' for index in range(1, limits['measured_requests'] + 1))
    if contract.get('stream_client') == 'antimatter':
        names.extend(('client/manifest.json', 'client/build-meta.json', 'client/provider.mjs'))
        paths = GROUNDED_SOURCE_PATHS if contract['profile'] in (*GROUNDED_PROFILES, GENERATED_HISTORY_PROFILE) else SOURCE_PATHS
        names.extend('client/source/' + name for name in paths)
        for _stage, stem, _artifact in client_artifacts(contract['profile']):
            names.extend(stem + '-client' + suffix for suffix in ('-request.json', '-process.json', '.log'))
    if contract['profile'] in GROUNDED_PROFILES:
        names.extend('answers-' + phase + '-refusal' + suffix
                     for phase in ('cold','warm') for suffix in ('.json','-post.json'))
    if contract['profile'] in C2048_PROFILES:
        names.extend(mode + '-start-cooldown.json' for mode in modes)
        names.extend(mode + '-cpu-retirement.json' for mode in modes if mode in GPU_MODES)
    for mode in modes:
        names.extend(mode + suffix for suffix in
                     ('.log', '-result.json', '-child-result.json', '-cgroup.json',
                      '-headroom.jsonl', '-kernel.log', '-driver.log', '-thermal.jsonl'))
    for prefix in ('cpu-final-', 'final-'):
        names.extend(prefix + suffix for suffix in
                     ('kernel-window.log', 'kernel-stderr.log', 'driver-matches.log'))
    return list(dict.fromkeys(names))


def closed_evidence_hashes(root=ROOT):
    hashes = {}
    for name in closed_evidence_names(root):
        path = root / name
        require(path.is_file() and not path.is_symlink(), 'missing or unsafe closed evidence: ' + name)
        require(path.stat().st_size <= 16 * 1024 * 1024, 'oversized closed evidence: ' + name)
        hashes[name] = digest(path)
    return hashes


def kernel_window(root, prefix, start, end):
    require(start <= end, 'kernel interval is reversed')
    paths = [root / (prefix + suffix) for suffix in
             ('kernel-window.log', 'kernel-stderr.log', 'driver-matches.log')]
    with paths[0].open('x') as out, paths[1].open('x') as err:
        result = subprocess.run(['journalctl', '-k', '--no-pager', '-o', 'short-iso-precise',
                                 '--since', start.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                 '--until', (end + timedelta(seconds=1)).strftime('%Y-%m-%d %H:%M:%S UTC')],
                                stdout=out, stderr=err, timeout=10, check=False)
    lines = text(paths[0]).splitlines()
    matches = [line for line in lines if re.search(PATTERN, line, re.IGNORECASE)]
    with paths[2].open('x') as output:
        output.write('\n'.join(matches) + ('\n' if matches else ''))
    return {'start': start.isoformat(), 'end': end.isoformat(), 'kernel_lines': len(lines),
            'driver_matches': matches, 'journal_exit_code': result.returncode,
            'qualified': result.returncode == 0 and not text(paths[1]).strip()
                         and bool(lines) and not matches}


def inspect_unit(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=LoadState,ActiveState,SubState,MainPID,ControlGroup'],
                            capture_output=True, text=True, timeout=5, check=False)
    require(result.returncode == 0 and not result.stderr.strip(), 'unit inspection failed: ' + unit)
    require(len(result.stdout) <= 16384, 'oversized unit inspection')
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def cleanup_evidence(reports, proc_root=Path('/proc'), cgroup_root=Path('/sys/fs/cgroup'), unit_reader=inspect_unit,
                     root=ROOT, profile='list-c128'):
    """Read only the recorded units/PIDs; never stops or clears system state."""
    modes = profile_modes(profile)
    require(set(reports) == set(modes), 'cleanup needs exactly the profile stage records')
    stages, pids = {}, set()
    for mode in modes:
        report = reports[mode]
        unit = report.get('unit')
        require(type(unit) is str
                and re.fullmatch(r'hag-answers-' + re.escape(mode) + r'-[0-9]{8}-[0-9]{6}-[0-9]{6}', unit),
                'cleanup unit identity differs: ' + mode)
        properties = unit_reader(unit)
        require(properties.get('ActiveState') == 'inactive' and properties.get('SubState') == 'dead'
                and properties.get('MainPID') == '0' and properties.get('ControlGroup') == '',
                'qualification unit is not fully retired: ' + unit)
        child = report['child']
        cgroup = Path(child['cgroup'])
        require(cgroup.is_absolute() and cgroup.is_relative_to(cgroup_root)
                and cgroup != cgroup_root and '..' not in cgroup.parts,
                'cleanup cgroup path is unsafe')
        require(not cgroup.exists() and not cgroup.is_symlink(), 'qualification cgroup remains present')
        for pid in (child['pid'], report['thermal']['summary']['pid']):
            require(type(pid) is int and pid > 0, 'cleanup PID is invalid')
            require(not (proc_root / str(pid)).exists(), 'recorded qualification PID is still present')
            pids.add(pid)
        stages[mode] = {'unit': unit, 'properties': properties, 'cgroup_gone': True}
    if profile in (LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, *UI_PROFILES):
        if profile == LIFECYCLE_PROFILE:
            from lifecycle import server_processes
        elif profile == UI_HISTORY_PROFILE:
            from ui_history import server_processes
        elif profile == UI_STOP_PROFILE:
            from ui_stop import server_processes
        else:
            from generated_history import server_processes
        server_pids = []
        for mode in ('cpu-reference', 'answers'):
            processes = server_processes(root, mode)
            require(type(processes) is list and len(processes) == 1,
                    'stage needs exactly one server process: ' + mode)
            for process in processes:
                pid = process.get('pid')
                require(type(pid) is int and pid > 0 and type(process.get('exit_code')) is int
                        and process['exit_code'] == 0 and pid not in pids,
                        'server process receipt differs or reuses a process identity')
                require(not (proc_root / str(pid)).exists(), 'recorded server PID is still present')
                server_pids.append(pid)
                pids.add(pid)
                stages[mode]['server_pids_absent'] = [pid]
        require(len(set(server_pids)) == 2, 'stages share a server PID')
    if profile in UI_PROFILES:
        if profile == UI_STOP_PROFILE:
            from ui_stop import listener
        else:
            from ui_history import listener
        broker = listener(root)
        require(broker['boot_id'] == (proc_root / 'sys/kernel/random/boot_id').read_text().strip(),
                'remote broker boot identity differs')
        remote_pids = [broker['pid']]
        remote_pids.extend(read_json(root / f'cpu-reference-turn-{turn}-measurement.json')['pid'] for turn in (1, 2))
        for pid in remote_pids:
            require(type(pid) is int and pid > 0 and pid not in pids, 'remote UI worker/inspector PID differs')
            require(not (proc_root / str(pid)).exists(), 'remote UI worker/inspector is still present')
            pids.add(pid)
        stages['answers']['broker_pid_absent'] = broker['pid']
        stages['cpu-reference']['inspector_pids_absent'] = remote_pids[1:]
    client_stage = 'answers' if profile == GENERATED_HISTORY_PROFILE else 'timing'
    client = reports.get(client_stage, {}).get('client_identity')
    if profile == GENERATED_HISTORY_PROFILE:
        require(client is not None, 'generated history needs recorded client identity')
    if client is not None:
        client_pids = []
        profile = approved_contract(root)['profile']
        artifacts = client_artifacts(profile)
        for stage, prefix, artifact in artifacts:
            process = read_json(root / (prefix + '-client-process.json'))
            record = read_json(root / artifact)
            pid = process.get('pid')
            require(type(pid) is int and pid > 0 and type(process.get('exit_code')) is int
                    and set(process) == {'pid','exit_code','timed_out','identity'}
                    and process['exit_code'] == 0 and process.get('timed_out') is False
                    and process.get('identity') == client and record.get('client', {}).get('pid') == pid,
                    'client process receipt is not a successful bound exit')
            require(not (proc_root / str(pid)).exists(), 'recorded client PID is still present')
            if profile == GENERATED_HISTORY_PROFILE:
                require(pid not in pids, 'generated client reused a recorded process identity')
            client_pids.append(pid)
            pids.add(pid)
            stages[stage].setdefault('client_pids_absent', []).append(pid)
        require(len(set(client_pids)) == len(artifacts), 'client requests reuse a recorded process identity')
    return {'all_units_inactive': True, 'recorded_pids_absent': sorted(pids),
            'all_cgroups_gone': True, 'stages': stages}


def qualification_scope(contract, qualified):
    """A backend parity result never establishes semantic answer quality."""
    scope = {'qualification_kind': contract['qualification_kind'],
            'require_semantic_match': contract['require_semantic_match'],
            'quality_is_diagnostic': not contract['require_semantic_match'],
            'semantic_quality_qualified': qualified and contract['require_semantic_match']}
    if contract['profile'] == LIFECYCLE_PROFILE:
        scope['live_cancellation_qualified'] = qualified
    if contract['profile'] == GENERATED_HISTORY_PROFILE:
        scope.update(generated_history_qualified=qualified, live_cancellation_qualified=False,
                     timing_executed=False, performance_qualified=False)
    if contract['profile'] == UI_HISTORY_PROFILE:
        scope.update(generated_history_qualified=qualified, live_ui_qualified=qualified,
                     live_cancellation_qualified=False, timing_executed=False,
                     performance_qualified=False, transport='buffered-sse-after-parity-reset',
                     ui_client_host=contract['ui_client_host'])
    if contract['profile'] == UI_STOP_PROFILE:
        scope.update(generated_history_qualified=False, live_ui_qualified=qualified,
                     live_cancellation_qualified=qualified, same_server_reuse_qualified=qualified,
                     cancellation_answer_parity_qualified=False, timing_executed=False,
                     performance_qualified=False, transport=contract['broker_delivery'],
                     ui_client_host=contract['ui_client_host'])
    return scope


def main(root=ROOT):
    outputs = [root / (prefix + suffix) for prefix in ('cpu-final-', 'final-')
               for suffix in ('kernel-window.log', 'kernel-stderr.log', 'driver-matches.log')]
    outputs.extend(root / name for name in ('final-evidence-check.json', 'final-cleanup-check.json'))
    require(not any(path.exists() or path.is_symlink() for path in outputs), 'closeout evidence exists; no overwrites')
    contract = approved_contract(root)
    modes = profile_modes(contract['profile'])
    records = stopped_capture(root)
    if contract['profile'] in C2048_PROFILES:
        require(type(records[0].get('duration_seconds')) is int and records[0]['duration_seconds'] == 1500,
                'project watcher lease differs')
    session = {'pid': records[0]['pid'], 'armed_at': records[0]['timestamp']}
    start, end = instant(records[0]['timestamp']), instant(records[-1]['timestamp'])
    failures = []
    gates = {mode: False for mode in modes}
    identities = None
    reports = {}
    windows = {}
    ui_evidence = None
    try:
        identities = {'gate_files': gate_identity(root), 'dependencies': dependency_identity(root),
                      'profile': contract['profile'],
                      'qualification_kind': contract['qualification_kind'],
                      'require_semantic_match': contract['require_semantic_match'],
                      'source': source_identity(root), 'checkpoint': checkpoint_identity(root),
                      'input_provenance': provenance_identity(root),
                      'binaries': {mode: binary_identity(mode, root) for mode in modes if mode != 'self-check'}}
        client = client_identity(root)
        if client is not None:
            identities['client_identity'] = client
        spec = importlib.util.spec_from_file_location('serving_closeout_runner', root / 'run-gate.py')
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        for mode in modes:
            try:
                reports[mode] = read_json(root / (mode + '-result.json'))
                validate_gate(mode, reports[mode], runner, session, start, end, identities, root)
                gates[mode] = True
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures.append(mode + ': ' + str(error))
        if all(gates.values()):
            validate_order(reports, contract['profile'])
            if contract['profile'] == UI_HISTORY_PROFILE:
                from ui_client import validate_ui_evidence
                from ui_history import broker_records
                ui_evidence = validate_ui_evidence(root, identities['client_identity'], broker_records(root))
            if contract['profile'] == UI_STOP_PROFILE:
                from ui_stop_client import validate_ui_evidence
                from ui_stop import broker_records
                ui_evidence = validate_ui_evidence(root, identities['client_identity'], broker_records(root))
    except (OSError, ValueError, KeyError, TypeError) as error:
        failures.append(str(error))
    # The CPU-only stage intentionally has no root watcher. Audit its complete
    # kernel interval independently instead of treating the later watcher as cover.
    intervals = [('gpu', 'final-', start, end)]
    try:
        intervals.append(('cpu', 'cpu-final-', instant(reports['cpu-reference']['start']),
                          instant(reports['cpu-reference']['end'])))
    except (KeyError, TypeError, ValueError) as error:
        failures.append('CPU reference interval unavailable for independent kernel closeout: ' + str(error))
    for name, prefix, first, last in intervals:
        try:
            windows[name] = kernel_window(root, prefix, first, last)
            if not windows[name]['qualified']:
                failures.append(name + ' complete kernel window unavailable, empty, or contains GPU/OOM/fault matches')
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            failures.append(name + ' kernel window: ' + str(error))
    try:
        cleanup = cleanup_evidence(reports, root=root, profile=contract['profile'])
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as error:
        cleanup = {'all_units_inactive': False, 'all_cgroups_gone': False, 'failure': str(error)}
        failures.append('process/cgroup cleanup: ' + str(error))
    write_json(root / 'final-cleanup-check.json', cleanup)
    evidence_hashes = {}
    try:
        evidence_hashes = closed_evidence_hashes(root)
    except (OSError, ValueError) as error:
        failures.append(str(error))
    manifest = read_json(root / 'expected-manifest.json')
    gpu = windows.get('gpu', {})
    qualified = not failures and all(gates.values())
    audit = {'profile': contract['profile'], **qualification_scope(contract, qualified),
             'armed_at': records[0]['timestamp'], 'stopped_at': records[-1]['timestamp'],
             'capture_pid': session['pid'], 'captured_dumps': 0, 'dump_directory_empty': True,
             'kernel_lines': gpu.get('kernel_lines', 0), 'driver_matches': gpu.get('driver_matches', []),
             'journal_exit_code': gpu.get('journal_exit_code'), 'kernel_windows': windows,
             'gates': gates, 'cleanup': cleanup, 'frozen_sources_verified': manifest['source_files'] if identities else 0,
             'frozen_gates_verified': len(identities['gate_files']) if identities else 0,
             'binaries_and_dependencies_verified': identities is not None,
             'checkpoint_identity_verified': identities is not None,
             'input_provenance_verified': identities is not None,
             'closed_evidence_sha256': evidence_hashes,
             'expected_manifest_sha256': digest(root / 'expected-manifest.json'),
             'closeout_helper_sha256': digest(Path(__file__)),
             'failures': failures, 'qualified': qualified}
    if identities is not None and 'client_identity' in identities:
        audit['client_identity'] = identities['client_identity']
    if contract['profile'] in UI_PROFILES:
        audit['ui_evidence'] = ui_evidence
    write_json(root / 'final-evidence-check.json', audit)
    print(json.dumps(audit, indent=2, allow_nan=False))
    return 0 if audit['qualified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
