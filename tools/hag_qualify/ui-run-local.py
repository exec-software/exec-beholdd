#!/usr/bin/env python3
"""Run the frozen UI in a local bounded service, with SSH for a separate host.

Execute this copy from a frozen gate mirror, pinned by --manifest-sha256. The
mirror needs the gate files, expected manifest, and client identity, not a model.
Copy the resulting ui-* evidence files back to that remote gate for closeout.
Import/help never opens a socket, launches Electron, or executes a model.
"""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import gate_common as common
import ui_client as ui

ROOT = Path(__file__).resolve().parent
INJECT_ENV = ('NODE_OPTIONS', 'NODE_PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH',
              'ELECTRON_EXTRA_LAUNCH_ARGS', 'ELECTRON_RUN_AS_NODE', 'PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP')


def clean_environment():
    return {key: value for key, value in os.environ.items() if key not in INJECT_ENV}


def publish(path, value):
    """Publish complete JSON exclusively; samplers never observe a partial receipt."""
    temporary = path.with_name(path.name + '.writing')
    common.write_json(temporary, value)
    os.link(temporary, path)
    temporary.unlink()


def tunnel_command(identity, ready, target):
    ui.require(type(target) is str and re.fullmatch(r'[a-z_][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9.-]*', target),
               'invalid SSH target')
    local = identity['base_url'].split(':')[2].split('/')[0]
    ui.require(ui.integer(ready.get('port'), 1024, 65535), 'invalid broker port')
    return ['/usr/bin/ssh', '-N', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5', '-o', 'ControlMaster=no',
            '-o', 'ControlPath=none', '-L', f'127.0.0.1:{local}:127.0.0.1:{ready["port"]}', target]


def owns_listener(pid, port, proc_root=Path('/proc')):
    """Observe a process's listening socket without consuming a broker connection."""
    directory = proc_root / str(pid)
    sockets = set()
    for descriptor in (directory / 'fd').iterdir():
        try:
            link = os.readlink(descriptor)
        except FileNotFoundError:
            continue
        if link.startswith('socket:[') and link.endswith(']'):
            sockets.add(link[8:-1])
    with (directory / 'net/tcp').open() as stream:
        for index, line in enumerate(stream):
            ui.require(index < 8192, 'local socket inventory exceeds bound')
            fields = line.split()
            if len(fields) > 9 and fields[1] == f'0100007F:{port:04X}' and fields[3] == '0A' and fields[9] in sockets:
                return True
    return False


def verify_inputs(args):
    common.require_sha(args.manifest_sha256)
    ui.require(common.digest(ROOT / 'expected-manifest.json') == args.manifest_sha256, 'frozen gate manifest differs')
    common.gate_identity(ROOT)
    identity = ui.client_identity(ROOT)
    ui.require(ui.observe_host() == identity['local_host'], 'frozen UI belongs to another host or boot')
    metadata = Path(args.metadata).resolve(strict=True)
    ui.require(common.read_json(metadata / 'identity.json') == identity, 'local UI metadata differs')
    ui.require(ui.canonical_sha(common.read_json(metadata / 'source-verification.json', ui.MAX_JSON)) == identity['source_verification_sha256'],
               'source verification receipt differs')
    runtime = common.read_json(metadata / 'electron-files.json', ui.MAX_JSON)
    ui.require(ui.canonical_sha(runtime) == identity['electron']['manifest_sha256'], 'runtime manifest differs')
    local = common.read_json(metadata / 'local-inputs.json')
    ui.require(set(local) == {'prepared_root', 'electron_dist'}, 'local inputs schema differs')
    prepared, distribution = Path(local['prepared_root']), Path(local['electron_dist'])
    ready_path = Path(args.ready).resolve(strict=True)
    ready = common.read_json(ready_path)
    ui.require(set(ready) == {'schema', 'profile', 'host', 'port', 'pid', 'run_id', 'client_identity_sha256',
                              'boot_id', 'delivery'} and ready['schema'] == 1
               and ready['profile'] == identity['profile'] and ready['host'] == '127.0.0.1'
               and ui.identifier(ready['run_id']) and uuid.UUID(ready['run_id']).version == 4
               and ui.identifier(ready['boot_id']) and ui.integer(ready['pid'], 1)
               and ready['delivery'] == ui.profile_spec(identity['profile'])['delivery']
               and ready['client_identity_sha256'] == common.digest(ROOT / 'client/identity.json'), 'broker ready identity differs')
    mode = ui.client_host(ROOT, identity, ready)
    if mode == 'same':
        ui.require(args.target is None, '--target is forbidden for same-host UI')
        command = None
    else:
        ui.require(args.target is not None, '--target is required for separate-host UI')
        command = tunnel_command(identity, ready, args.target)
    return identity, ready, prepared, distribution, runtime, command


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=.5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=2)


def observe_child(pid, group, proc_root=Path('/proc')):
    try:
        return ui.observe_process(pid, 'electron-child', group, proc_root=proc_root)
    except (FileNotFoundError, ProcessLookupError):
        return None
    except ValueError:
        # Short-lived children can retire between the procfs reads. An observed
        # live process with bad containment or environment must still fail.
        try:
            state = (proc_root / str(pid) / 'stat').read_text().rpartition(') ')[2].split()[0]
        except (FileNotFoundError, ProcessLookupError):
            return None
        if state == 'Z':
            return None
        raise


def worker(args):
    identity, ready, prepared, distribution, runtime, tunnel_args = verify_inputs(args)
    output = Path(args.output)
    group = common.cgroup_path(Path('/proc/self/cgroup').read_text())
    ui.require({key: (group / key).read_text().strip() for key in ui.UI_LIMITS} == ui.UI_LIMITS,
               'local worker containment differs')
    rows = [ui.observe_process(os.getpid(), 'supervisor', group)]
    publish(output / 'worker-ready.json', {'cgroup': str(group), 'pid': os.getpid()})
    wait_until = time.monotonic() + 5
    while not (output / 'start').exists():
        ui.require(time.monotonic() < wait_until, 'supervisor did not admit UI startup')
        time.sleep(.02)
    started = time.monotonic_ns()
    deadline = time.monotonic() + 355
    tunnel = electron = None
    def expired(signum, frame):
        raise TimeoutError('local UI whole-process deadline expired')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, 355)
    try:
        ui.validate_local_bundle(prepared, identity, fresh=True)
        ui.require(ui.runtime_manifest(distribution) == runtime, 'Electron runtime changed before launch')
        manifest = common.read_json(prepared / 'prepared-live.json', ui.MAX_JSON)
        with ExitStack() as logs:
            port = int(identity['base_url'].split(':')[2].split('/')[0])
            if tunnel_args is not None:
                log = logs.enter_context((output / 'tunnel.log').open('x'))
                tunnel = subprocess.Popen(tunnel_args, stdout=log, stderr=subprocess.STDOUT, env=clean_environment())
                rows.append(ui.observe_process(tunnel.pid, 'tunnel', group))
                tunnel_deadline = time.monotonic() + 8
                while not owns_listener(tunnel.pid, port):
                    ui.require(tunnel.poll() is None and time.monotonic() < tunnel_deadline, 'SSH tunnel did not become ready')
                    time.sleep(.05)
            else:
                ui.require(owns_listener(ready['pid'], port), 'same-host UI broker no longer owns its listener')
            electron_log = logs.enter_context((output / 'electron.log').open('x'))
            electron = subprocess.Popen(manifest['command'], cwd=prepared,
                env={**clean_environment(), **manifest['environment']}, stdout=electron_log, stderr=subprocess.STDOUT)
            rows.append(ui.observe_process(electron.pid, 'electron', group))
            seen = {row['pid'] for row in rows}
            while electron.poll() is None:
                ui.require((tunnel is None or tunnel.poll() is None) and time.monotonic() < deadline,
                           'tunnel exited or UI deadline expired')
                for pid in map(int, (group / 'cgroup.procs').read_text().split()):
                    if pid not in seen:
                        row = observe_child(pid, group)
                        if row is not None:
                            rows.append(row)
                            seen.add(pid)
                ui.require(len(rows) <= 128, 'UI process inventory exceeded bound')
                time.sleep(.1)
            ui.require(electron.returncode == 0, 'Electron failed')
            stop_child(tunnel)
        ui.validate_local_bundle(prepared, identity)
        ui.require(ui.runtime_manifest(distribution) == runtime, 'Electron runtime changed after launch')
        report = common.read_json(prepared / 'report.json', ui.MAX_JSON)
        common.write_json(output / 'ui-report.json', report)
        receipt = {'schema': 1, 'kind': 'antimatter-ui-process', 'host': ui.observe_host(),
            'run_id': ready['run_id'], 'identity_sha256': common.digest(ROOT / 'client/identity.json'),
            'ready_sha256': common.digest(Path(args.ready)), 'report_sha256': common.digest(output / 'ui-report.json', ui.MAX_JSON),
            'unit': args.unit, 'cgroup': str(group), 'processes': rows,
            'started_monotonic_ns': started, 'ended_monotonic_ns': time.monotonic_ns(),
            'exit_code': 0, 'timed_out': False, 'bundle_verified_before': True, 'bundle_verified_after': True,
            'electron_manifest_sha256': identity['electron']['manifest_sha256']}
        ui.require(receipt['ended_monotonic_ns'] - started <= 355 * 10 ** 9, 'UI exceeded process bound')
        publish(output / 'ui-process.json', receipt)
        # Keep the cgroup observable until the outer sampler records final counters.
        wait_until = time.monotonic() + 2
        while not (output / 'finish').exists():
            ui.require(time.monotonic() < wait_until, 'supervisor missed final UI counters')
            time.sleep(.02)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        stop_child(electron)
        stop_child(tunnel)


def sensor():
    sensors = [path.with_name(path.name.replace('_label', '_input'))
               for path in Path('/sys/class/hwmon').glob('hwmon*/temp*_label') if path.read_text().strip() == 'Tctl']
    ui.require(len(sensors) == 1, 'expected one Tctl sensor')
    return sensors[0]


def temperature(path):
    value = int(path.read_text().strip()) / 1000
    ui.require(0 <= value <= 125, 'invalid Tctl reading')
    return value


def cooldown(path):
    samples = []
    end = time.monotonic() + 60
    while time.monotonic() < end:
        item = {'monotonic_ns': time.monotonic_ns(), 'tctl_c': temperature(path)}
        if samples and not 0 < item['monotonic_ns'] - samples[-1]['monotonic_ns'] <= 10 ** 9:
            samples = []
        samples = samples + [item] if item['tctl_c'] <= 80 else []
        if len(samples) >= 21 and samples[-1]['monotonic_ns'] - samples[0]['monotonic_ns'] >= 10 * 10 ** 9:
            return samples
        time.sleep(.5)
    raise ValueError('UI startup did not cool to 80 C for ten seconds')


def sample(group, path):
    memory = common.memory_snapshot(group)
    item = {'monotonic_ns': time.monotonic_ns(), 'available_bytes': common.available_memory(),
            'tctl_c': temperature(path), 'cgroup': {
                'memory.current': memory['memory_current_bytes'], 'memory.peak': memory['memory_peak_bytes'],
                'memory.events': memory['memory_events'], 'memory.swap.current': memory['swap_current_bytes']}}
    ui.require(item['available_bytes'] >= 4 * common.GIB and item['tctl_c'] < 90
               and memory['memory_peak_bytes'] < common.STOP_CGROUP_BYTES
               and memory['swap_current_bytes'] == 0 and not any(memory['memory_events'].values()),
               'local UI memory, headroom, or thermal cutoff reached')
    return item


def supervise(args):
    identity, ready, *_ = verify_inputs(args)
    output = Path(args.output).absolute()
    ui.require(not output.exists() and not output.is_symlink() and output.parent.resolve() == output.parent,
               'UI output must be a fresh canonical directory')
    output.mkdir(mode=0o700)
    path = sensor()
    cool = cooldown(path)
    common.require_headroom(common.available_memory())
    unit = 'hag-ui-' + uuid.uuid4().hex + '.service'
    command = ['systemd-run', '--user', '--wait', '--pipe', '--collect', '--expand-environment=no', '--unit=' + unit,
               '-p', 'CPUQuota=25%', '-p', 'TasksMax=128', '-p', 'MemoryHigh=1536M', '-p', 'MemoryMax=2G',
               '-p', 'MemorySwapMax=0', '-p', 'NoNewPrivileges=yes', '-p', 'OOMPolicy=stop',
               '-p', 'RuntimeMaxSec=375s', '-p', 'TimeoutStopSec=2s', '-p', 'KillMode=control-group', '-p', 'Restart=no',
               '-p', 'UnsetEnvironment=' + ' '.join(INJECT_ENV),
               '/usr/bin/taskset', '-c', '0,1', '/usr/bin/python3', '-B', str(ROOT / 'ui-run-local.py'),
               '--worker', '--unit', unit, '--manifest-sha256', args.manifest_sha256,
               '--metadata', str(Path(args.metadata).resolve()), '--ready', str(Path(args.ready).resolve()),
               '--output', str(output)]
    if args.target is not None:
        command += ['--target', args.target]
    samples, group, reason, child = [], None, None, None
    begin = time.monotonic()
    try:
        with (output / 'supervisor.log').open('x') as log, (output / 'ui-supervision-samples.jsonl').open('x') as evidence:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=clean_environment())
            while child.poll() is None:
                ui.require(time.monotonic() - begin < 390, 'local UI outer deadline expired')
                if group is None:
                    if (output / 'worker-ready.json').exists():
                        row = common.read_json(output / 'worker-ready.json')
                        group = Path(row['cgroup'])
                        ui.observe_process(row['pid'], 'supervisor', group)
                        ui.require({key: (group / key).read_text().strip() for key in ui.UI_LIMITS} == ui.UI_LIMITS,
                                   'local UI limits differ')
                    else:
                        ui.require(time.monotonic() - begin < 4, 'local worker startup receipt absent')
                if group is not None:
                    try:
                        row = sample(group, path)
                    except FileNotFoundError:
                        # A completed service can disappear between cgroup reads.
                        # Continue observing until that happens, including cleanup.
                        ui.require((output / 'finish').exists() and not group.exists(),
                                   'local UI cgroup disappeared before final counters')
                        time.sleep(.1)
                        continue
                    if samples:
                        ui.require(0 < row['monotonic_ns'] - samples[-1]['monotonic_ns'] <= 10 ** 9, 'local UI sampling gap')
                    else:
                        common.require_headroom(row['available_bytes'])
                    samples.append(row)
                    evidence.write(json.dumps(row, allow_nan=False) + '\n')
                    evidence.flush()
                    if not (output / 'start').exists():
                        (output / 'start').touch(exist_ok=False)
                    if (output / 'ui-process.json').exists() and not (output / 'finish').exists():
                        (output / 'finish').touch(exist_ok=False)
                time.sleep(.1)
            ui.require(child.wait(timeout=2) == 0, 'local UI service failed')
        process = common.read_json(output / 'ui-process.json', ui.MAX_JSON)
        ui.require(len(samples) >= 2, 'missing local UI supervision')
        summary = {'schema': 1, 'kind': 'antimatter-ui-supervision', 'host': identity['local_host'],
            'run_id': ready['run_id'], 'unit': unit, 'limits': dict(ui.UI_LIMITS), 'exit_code': 0, 'abort_reason': None,
            'initial_available_bytes': samples[0]['available_bytes'],
            'minimum_available_bytes': min(row['available_bytes'] for row in samples),
            'memory_peak_bytes': max(row['cgroup']['memory.peak'] for row in samples),
            'events_zero': True, 'swap_zero': True, 'peak_tctl_c': max(row['tctl_c'] for row in samples),
            'thermal_cutoff_c': 90, 'process_receipt_sha256': common.digest(output / 'ui-process.json', ui.MAX_JSON),
            'samples_sha256': common.digest(output / 'ui-supervision-samples.jsonl', ui.MAX_JSON), 'cooldown': cool}
        common.write_json(output / 'ui-supervision.json', summary)
        ui.validate_ui_supervision(output, process)
        ui.observe_retirement(output / 'ui-process.json', output / 'ui-retirement.json')
        common.gate_identity(ROOT)
        print(json.dumps({'local_evidence_complete': True, 'backend_qualified': False, 'output': str(output)}))
    except BaseException as error:
        reason = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        cleanup_errors = []
        if child is not None and (reason or child.poll() is None):
            try:
                result = subprocess.run(['systemctl', '--user', 'stop', unit], capture_output=True, timeout=8, check=False)
                if result.returncode:
                    cleanup_errors.append('systemctl stop returned ' + str(result.returncode))
                child.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup_errors.append(type(error).__name__ + ': ' + str(error))
        if reason:
            common.write_json(output / 'local-failure.json', {'reason': reason, 'unit': unit,
                'cleanup_errors': cleanup_errors, 'backend_qualified': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'ready', 'output', 'manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--target', help='user@host; required only for the frozen separate-host mode')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--unit', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        ui.require(type(args.unit) is str and re.fullmatch(r'hag-ui-[a-z0-9-]{1,100}\.service', args.unit), 'invalid UI worker unit')
        worker(args)
    else:
        ui.require(args.unit is None, 'unit is selected by the supervisor')
        supervise(args)


if __name__ == '__main__':
    main()
