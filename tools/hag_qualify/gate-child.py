#!/usr/bin/env python3
"""Verify actual limits before invoking the frozen probe through the thermal guard."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gate_common import (ROOT, EXPECTED_LIMITS, STOP_HEADROOM, available_memory,
                         binary_identity, dependency_identity, source_identity, cgroup_path, memory_snapshot, require_headroom,
                         verify_limits, clean_memory, write_json, require_process_headroom, gate_identity, MODES, CHILD_SECONDS, provenance_identity, approved_contract, stage_deadlines, private_cgroup_attestation,
                         PHASE_ENVIRONMENT, phase_diagnostics)
from profiles import LIFECYCLE_PROFILE, GENERATED_HISTORY_PROFILE, UI_HISTORY_PROFILE, UI_STOP_PROFILE, C2048_PROFILES, profile_modes
from gate_common import client_identity

GUARD = ROOT / 'dependencies/thermal_guard.py'


def probe_command(mode, root=ROOT, profile=None):
    if profile is None:
        profile = approved_contract(root)['profile']
    if mode not in profile_modes(profile):
        raise ValueError('stage is not part of the declared profile')
    if mode == 'self-check':
        return ['/usr/bin/true']
    if mode in ('cpu-reference', 'answers', 'timing'):
        worker = ('lifecycle-gate.py' if profile == LIFECYCLE_PROFILE else
                  'ui-stop-gate.py' if profile == UI_STOP_PROFILE else
                  'ui-history-gate.py' if profile == UI_HISTORY_PROFILE else
                  'generated-history-gate.py' if profile == GENERATED_HISTORY_PROFILE else 'answer-gate.py')
        return ['/usr/bin/python3', str(root / worker), '--stage', mode, '--execute']
    raise ValueError('unknown gate mode')



def workload_environment(mode, profile, attestation, inherited=None):
    if mode not in profile_modes(profile):
        raise ValueError('stage is not part of the declared profile')
    environment = dict(os.environ if inherited is None else inherited)
    environment.pop(PHASE_ENVIRONMENT, None)
    environment.pop('BEHOLDD_HAG_PRIVATE_CGROUP', None)
    if profile in C2048_PROFILES and mode in ('answers', 'timing'):
        if type(attestation) is not str or not attestation.startswith('/'):
            raise ValueError('missing verified private cgroup attestation')
        environment['BEHOLDD_HAG_PRIVATE_CGROUP'] = attestation
    if mode != 'self-check':
        environment['HAG_ANSWERS_GATE'] = '1'
    return environment

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=MODES)
    mode = parser.parse_args(argv).mode
    result_path = ROOT / (mode + '-child-result.json')
    ready_path = ROOT / (mode + '-cgroup.json')
    thermal_path = ROOT / (mode + '-thermal.jsonl')
    if any(path.exists() for path in (result_path, ready_path, thermal_path)):
        raise ValueError('child evidence already exists; no retry/overwrite')
    cgroup = cgroup_path(Path('/proc/self/cgroup').read_text())
    actual = {name: (cgroup / name).read_text().strip() for name in EXPECTED_LIMITS}
    verify_limits(actual, Path('/proc/self/status').read_text(), os.environ.get('HAG_TRACE_SUBMISSION'))
    start_available = available_memory()
    require_headroom(start_available)
    dependencies = dependency_identity()
    gate_files = gate_identity()
    input_provenance = provenance_identity()
    client = client_identity(ROOT)
    contract = approved_contract()
    child_seconds, service_seconds, outer_seconds = stage_deadlines(mode)
    identity = None if mode == 'self-check' else binary_identity(mode)
    source_sha = None if mode == 'self-check' else source_identity()
    before = memory_snapshot(cgroup)
    report = {'mode': mode, 'cgroup': str(cgroup), 'limits': actual, 'no_new_privileges': True,
              'binary_sha256': identity, 'source_manifest_sha256': source_sha,
              'dependencies': dependencies, 'gate_files': gate_files, 'input_provenance': input_provenance, 'available_before_bytes': start_available,
              'memory_before': before, 'pid': os.getpid()}
    diagnostics = phase_diagnostics(contract, mode)
    if diagnostics is not None:
        report['runtime_phase_diagnostics'] = diagnostics
    attestation = None
    if contract['profile'] in C2048_PROFILES:
        attestation = private_cgroup_attestation(cgroup, actual, Path('/proc/self/status').read_text(),
                                                os.environ.get('HAG_TRACE_SUBMISSION'), before)
        report['stage_deadlines'] = {'child': child_seconds, 'service': service_seconds, 'outer': outer_seconds}
        report['private_cgroup_attestation'] = attestation if mode in ('answers', 'timing') else None
    if client is not None:
        report['client_identity'] = client
    write_json(ready_path, report)
    print(json.dumps(report), flush=True)
    command = [sys.executable, str(GUARD), '--warn-c', '75', '--critical-c', '90',
               '--poll-seconds', '0.1', '--grace-seconds', '0.5', '--log', str(thermal_path), '--',
               '/usr/bin/timeout', '--kill-after=1s', f'{child_seconds}s', *probe_command(mode, profile=contract['profile'])]
    environment = workload_environment(mode, contract['profile'], attestation)
    process = subprocess.Popen(command, start_new_session=True, env=environment)
    failure = None
    deadline = time.monotonic() + child_seconds + 8
    minimum = start_available
    try:
        while process.poll() is None:
            available = available_memory()
            minimum = min(minimum, available)
            if available < STOP_HEADROOM:
                failure = 'host_headroom_below_4GiB'
                break
            try:
                require_process_headroom(memory_snapshot(cgroup))
            except ValueError as error:
                failure = str(error)
                break
            if time.monotonic() >= deadline:
                failure = 'thermal_supervisor_timeout'
                break
            time.sleep(0.1)
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        raise
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
        after = memory_snapshot(cgroup)
        try:
            if client_identity(ROOT) != client:
                failure = failure or 'optional_client_identity_changed'
        except (OSError, ValueError, KeyError, TypeError) as error:
            failure = failure or 'optional_client_identity_failed: ' + str(error)
        report.update(exit_code=process.returncode, failure=failure, memory_after=after,
                      minimum_available_bytes=minimum, memory_clean=clean_memory(before, after),
                      finished_at=datetime.now(timezone.utc).isoformat())
        write_json(result_path, report)
        print(json.dumps({'child_result': report}), flush=True)
    return 0 if process.returncode == 0 and failure is None and report['memory_clean'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
