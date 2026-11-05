"""Frozen, host-qualified identity and evidence for the opt-in Antimatter UI client.

Historical source pins below are absent from the clean public client history.
These evidence schemas cannot qualify a replacement revision. See the public
source scope in docs/hag-qualification.md before selecting a client workflow.

Importing this module never starts a client, server, model, or SSH connection. The
old provider-only identity deliberately remains in antimatter.py. A local runner
must observe its own processes; remote closeout cannot inspect another host's /proc.
"""
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

try:
    from . import gate_common as common
    from .antimatter import _run
except ImportError:
    import gate_common as common
    from antimatter import _run

UI_PROFILE = 'antimatter-ui-history-c2048'
UI_COMMIT = '8a0f8dce59fe7e5d71777fd4f709f5d70f67c15e'
PROFILE = UI_PROFILE
COMMIT = UI_COMMIT
UI_STOP_PROFILE = 'antimatter-ui-stop-reuse-c2048'
UI_STOP_COMMIT = 'f8f43c91c00e155121ce99493f123fc3711af547'
UI_PROFILES = (UI_PROFILE, UI_STOP_PROFILE)
MAX_JSON = 16 * 1024 ** 2
MAX_ARTIFACT = 32 * 1024 ** 2
MAX_RUNTIME_FILE = 512 * 1024 ** 2
MAX_BODY = 1024 ** 2
ENTRIES = {'preload.cjs': 'src/preload/index.ts',
           'conversations.cjs': 'src/main/conversations.ts',
           'fixture.js': 'test/fixtures/hag-conversation-renderer.tsx',
           'live-host.cjs': 'test/fixtures/hag-conversation-live-host.ts'}
ARTIFACTS = {*ENTRIES, *(name + '.meta.json' for name in ENTRIES), 'fixture.css',
             'index.html', 'launcher.mjs', 'replies.json', 'source-hashes.json'}
RUNTIME_DIRS = {'config', 'cache', 'state', 'data', 'runtime', 'tmp'}
SOURCE_EXTRAS = {'package.json', 'package-lock.json', 'scripts/smoke-hag-conversation.mjs',
                 'test/fixtures/hag-generated-history.json'}
BUILTINS = {'assert', 'assert/strict', 'buffer', 'child_process', 'crypto', 'events', 'fs',
            'fs/promises', 'http', 'https', 'module', 'net', 'os', 'path', 'process',
            'stream', 'stream/promises', 'string_decoder', 'timers', 'timers/promises',
            'tls', 'url', 'util', 'zlib', 'worker_threads'}
HTML_SHA = '175b5804cdcb972ef5e42bccdb34138fab519a9a4ff223902898fc4d3c20ba4d'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def profile_spec(profile):
    require(profile in UI_PROFILES, 'unknown UI profile')
    stop = profile == UI_STOP_PROFILE
    return {'commit': UI_STOP_COMMIT if stop else UI_COMMIT,
            'fixture': 'test/fixtures/hag-stop-reuse.json' if stop else 'test/fixtures/hag-generated-history.json',
            'flags': ['--live-stop-reuse'] if stop else [],
            'delivery': 'live-sse-stop-then-buffered-parity-reset' if stop else 'buffered-sse-after-parity-reset'}


def source_extras(profile):
    return (SOURCE_EXTRAS - {'test/fixtures/hag-generated-history.json'}) | {profile_spec(profile)['fixture']}


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def integer(value, minimum=0, maximum=None):
    return type(value) is int and value >= minimum and (maximum is None or value <= maximum)


def identifier(value):
    try:
        return type(value) is str and str(uuid.UUID(value)) == value
    except (ValueError, TypeError, AttributeError):
        return False


def validate_host(host):
    require(type(host) is dict and set(host) == {'machine_id_sha256', 'boot_id'}, 'UI host schema differs')
    common.require_sha(host['machine_id_sha256'])
    require(identifier(host['boot_id']), 'UI boot identity differs')
    return host


def observe_host(machine_id=Path('/etc/machine-id'), boot_id=Path('/proc/sys/kernel/random/boot_id')):
    # procfs reports size zero; read a fixed bound instead of relying on st_size.
    with Path(machine_id).open('rb') as stream:
        machine = stream.read(129).strip()
    with Path(boot_id).open('r') as stream:
        boot = stream.read(129).strip()
    require(re.fullmatch(b'[0-9a-f]{32}', machine) is not None, 'invalid local machine identity')
    return validate_host({'machine_id_sha256': hashlib.sha256(machine).hexdigest(), 'boot_id': boot})


def endpoint(value):
    match = re.fullmatch(r'http://127\.0\.0\.1:([1-9][0-9]{0,4})/v1', value) if type(value) is str else None
    require(match is not None and 0 < int(match[1]) <= 65535 and int(match[1]) != 80,
            'UI endpoint must be an explicit non-default loopback port')
    return value


def validate_identity(identity):
    require(type(identity) is dict and set(identity) == {
        'schema', 'kind', 'profile', 'commit', 'snapshot_id', 'local_host', 'base_url',
        'prepared_manifest_sha256', 'artifacts', 'source_verification_sha256', 'electron',
        'process_seconds', 'request_seconds', 'maximum_requests', 'source_bundle_relation'},
        'UI identity schema differs')
    require(type(identity['schema']) is int and identity['schema'] == 1
            and identity['kind'] == 'antimatter-ui' and identity['profile'] in UI_PROFILES
            and identity['commit'] == profile_spec(identity['profile'])['commit']
            and re.fullmatch(r'[0-9a-f]{40}', identity['commit']) is not None and identifier(identity['snapshot_id']),
            'UI identity provenance differs')
    validate_host(identity['local_host'])
    endpoint(identity['base_url'])
    for key in ('prepared_manifest_sha256', 'source_verification_sha256'):
        common.require_sha(identity[key])
    artifacts = identity['artifacts']
    require(type(artifacts) is dict and set(artifacts) == ARTIFACTS, 'UI artifact inventory differs')
    for value in artifacts.values():
        common.require_sha(value)
    require(artifacts['index.html'] == HTML_SHA, 'UI page policy differs')
    electron = identity['electron']
    require(type(electron) is dict and set(electron) == {'manifest_sha256', 'executable_sha256', 'files', 'version'}
            and integer(electron['files'], 1, 512)
            and type(electron['version']) is str and re.fullmatch(r'[0-9]+\.[0-9]+\.[0-9]+', electron['version']),
            'Electron identity schema differs')
    common.require_sha(electron['manifest_sha256'])
    common.require_sha(electron['executable_sha256'])
    require(all(type(identity[key]) is int and identity[key] == value for key, value in
                {'process_seconds': 355, 'request_seconds': 240, 'maximum_requests': 2}.items()),
            'UI client limits differ')
    require(identity['source_bundle_relation'] == 'esbuild_recorded_inputs_not_independently_rebuilt',
            'UI importer cannot attest a rebuild it did not perform')
    return identity


def _source_path(name):
    require(type(name) is str and name and len(name) <= 4096 and '\\' not in name and '\x00' not in name,
            'invalid UI source name')
    parts = Path(name).parts
    # Clean archives may symlink node_modules; esbuild then records a relative path
    # back to the installed dependencies. Only its safe node_modules suffix is read.
    if 'node_modules' in parts:
        suffix = parts[parts.index('node_modules'):]
        require(len(suffix) > 1 and '..' not in suffix, 'unsafe UI dependency path')
        return str(Path(*suffix)), True
    # The clean development archive can also symlink this committed vendor tree.
    # Normalize only that named subtree; every resulting file is still compared
    # with a regular blob at UI_COMMIT, never trusted as an installed dependency.
    if 'vendor' in parts and (Path(name).is_absolute() or '..' in parts):
        suffix = parts[parts.index('vendor'):]
        require(parts.count('vendor') == 1 and len(suffix) > 2
                and suffix[:2] == ('vendor', 'displacer') and '..' not in suffix,
                'unsafe UI committed vendor path')
        return str(Path(*suffix)), False
    require(not Path(name).is_absolute() and '..' not in parts and str(Path(name)) == name,
            'UI source escaped the committed tree')
    return name, False


def _external(item):
    name = item.get('path')
    return type(name) is str and (name == 'electron' or name.removeprefix('node:') in BUILTINS)


def _metadata(prepared, profile=UI_PROFILE):
    closure = set()
    for artifact, entry in ENTRIES.items():
        document = common.read_json(prepared / (artifact + '.meta.json'), MAX_JSON)
        require(type(document) is dict and set(document) == {'inputs', 'outputs'}, 'UI esbuild schema differs')
        inputs, outputs = document['inputs'], document['outputs']
        require(type(inputs) is dict and entry in inputs and len(inputs) <= 8192
                and type(outputs) is dict, 'UI bundle input inventory differs')
        closure.update(inputs)
        for name, value in inputs.items():
            _source_path(name)
            require(type(value) is dict and type(value.get('imports')) is list, 'UI input metadata differs')
            for item in value['imports']:
                require(type(item) is dict and (_external(item) if item.get('external') is True
                        else item.get('path') in inputs), 'UI bundle imported unbound code')
        expected = {artifact, 'fixture.css'} if artifact == 'fixture.js' else {artifact}
        require(len(outputs) == len(expected) and {Path(name).name for name in outputs} == expected,
                'UI bundle output inventory differs')
        for name, value in outputs.items():
            require(type(value) is dict and type(value.get('imports')) is list
                    and all(type(item) is dict and item.get('external') is True and _external(item)
                            for item in value['imports'])
                    and set(value.get('inputs', {})) <= set(inputs)
                    and value.get('bytes') == (prepared / Path(name).name).stat().st_size,
                    'UI bundle output metadata differs')
            if Path(name).name == artifact:
                require(value.get('entryPoint') == entry, 'UI entry point differs')
    return closure | source_extras(profile)


def _committed(repo, name, commit=UI_COMMIT):
    args = ['/usr/bin/git', '-C', str(repo)]
    tree = _run(args + ['ls-tree', commit, '--', name], repo, 4096).decode().strip()
    require(re.fullmatch(r'100644 blob [0-9a-f]{40}\t' + re.escape(name), tree) is not None,
            'UI source is not a regular committed blob: ' + name)
    return _run(args + ['show', '--no-ext-diff', commit + ':' + name], repo, MAX_ARTIFACT)


def runtime_manifest(directory):
    directory = Path(directory).absolute()
    require(directory.is_dir() and directory.resolve() == directory, 'Electron distribution is not canonical')
    files, total = {}, 0
    for count, path in enumerate(directory.rglob('*'), 1):
        require(count <= 1024 and len(path.relative_to(directory).parts) <= 8 and not path.is_symlink(),
                'Electron distribution exceeds its safe inventory')
        if path.is_dir():
            continue
        require(len(files) < 512, 'too many Electron distribution files')
        files[str(path.relative_to(directory))] = common.digest(path, MAX_RUNTIME_FILE)
        total += path.stat().st_size
        require(total <= 1024 ** 3, 'Electron distribution exceeds 1 GiB')
    require('electron' in files and 'version' in files and os.access(directory / 'electron', os.X_OK),
            'missing Electron executable/version')
    with common.regular_reader(directory / 'electron', MAX_RUNTIME_FILE) as stream:
        require(stream.read(6) == b'\x7fELF\x02\x01', 'Electron must be a native 64-bit little-endian ELF')
    return files


def validate_local_bundle(prepared_root, identity, *, fresh=False):
    validate_identity(identity)
    prepared = Path(prepared_root).absolute()
    require(prepared.is_dir() and prepared.resolve() == prepared, 'UI bundle directory is not canonical')
    for name, expected in identity['artifacts'].items():
        require(common.digest(prepared / name, MAX_ARTIFACT) == expected, 'UI bundle changed: ' + name)
    require(common.digest(prepared / 'prepared-live.json', MAX_JSON) == identity['prepared_manifest_sha256'],
            'UI prepared manifest changed')
    if fresh:
        require({p.name for p in prepared.iterdir()} == ARTIFACTS | RUNTIME_DIRS | {'prepared-live.json'},
                'prepared UI bundle has been used or contains extra files')
        for name in RUNTIME_DIRS:
            path = prepared / name
            require(path.is_dir() and not path.is_symlink() and not list(path.iterdir()),
                    'prepared UI runtime directory is not fresh')
    return prepared


def freeze_ui_identity(prepared_root, repo, electron_dist, metadata_root):
    """Verify prepared inputs against Git/disk and seal metadata without execution.

    This imports an observed build, not a reproducible rebuild. Node dependency
    bytes are verified and recorded; npm lock integrity is not claimed to verify
    the unpacked installation. No input checkout or runtime is modified.
    """
    prepared, repo, distribution = (Path(p).absolute() for p in (prepared_root, repo, electron_dist))
    require(repo.resolve() == repo and repo.is_dir(), 'UI repository is not canonical')
    manifest = common.read_json(prepared / 'prepared-live.json', MAX_JSON)
    require(type(manifest) is dict and set(manifest) == {'profile', 'backendQualified', 'baseUrl',
        'processSeconds', 'requestSeconds', 'maximumRequests', 'artifacts', 'command', 'environment'},
        'prepared UI manifest schema differs')
    spec = profile_spec(manifest['profile'])
    profile, commit = manifest['profile'], spec['commit']
    committed = lambda name: _committed(repo, name) if profile == UI_PROFILE else _committed(repo, name, commit)
    require(manifest['backendQualified'] is False
            and manifest['processSeconds'] == 355 and manifest['requestSeconds'] == 240
            and manifest['maximumRequests'] == 2, 'prepared UI profile/limits differ')
    require(type(manifest['artifacts']) is dict and set(manifest['artifacts']) == ARTIFACTS,
            'prepared UI artifact inventory differs')
    base = endpoint(manifest['baseUrl'])
    require(manifest['command'] == [str(distribution / 'electron'), '--no-sandbox', '--ozone-platform=headless',
            str(prepared / 'launcher.mjs'), '--fixture', str(prepared), '--live-base-url=' + base] + spec['flags'],
            'prepared UI command differs from explicit inputs')
    environment = {'ELECTRON_RUN_AS_NODE': '', **{key: str(prepared / value) for key, value in {
        'XDG_CONFIG_HOME': 'config', 'XDG_CACHE_HOME': 'cache', 'XDG_STATE_HOME': 'state',
        'XDG_DATA_HOME': 'data', 'XDG_RUNTIME_DIR': 'runtime', 'TMPDIR': 'tmp'}.items()}}
    require(manifest['environment'] == environment, 'prepared UI environment differs')
    sources = common.read_json(prepared / 'source-hashes.json', MAX_JSON)
    require(type(sources) is dict and set(sources) == _metadata(prepared, profile) and len(sources) <= 8192,
            'UI source manifest does not cover the complete bundle closure')
    verified = {'commit': commit, 'committed_sources': {}, 'installed_dependencies': {}}
    for name, expected in sources.items():
        common.require_sha(expected)
        relative, dependency = _source_path(name)
        if dependency:
            path = (repo / relative).resolve(strict=True)
            require(path.is_relative_to((repo / 'node_modules').resolve(strict=True)), 'dependency escaped installed tree')
            observed = common.digest(path, MAX_ARTIFACT)
        else:
            observed = hashlib.sha256(committed(relative)).hexdigest()
        require(observed == expected, 'UI input differs from committed/installed bytes: ' + name)
        group = verified['installed_dependencies' if dependency else 'committed_sources']
        require(relative not in group, 'UI input aliases another source')
        group[relative] = expected
    require(manifest['artifacts'].get('launcher.mjs') == sources['scripts/smoke-hag-conversation.mjs']
            and manifest['artifacts'].get('replies.json') == sources[spec['fixture']],
            'UI launcher or oracle differs from committed source')
    runtime = runtime_manifest(distribution)
    with common.regular_reader(distribution / 'version', 128) as stream:
        version = stream.read(129).decode().strip()
    lock = common.strict_json(committed('package-lock.json'))
    require(lock['packages']['node_modules/electron']['version'] == version, 'Electron differs from the committed lock version')
    identity = validate_identity({'schema': 1, 'kind': 'antimatter-ui', 'profile': profile, 'commit': commit,
        'snapshot_id': str(uuid.uuid4()), 'local_host': observe_host(), 'base_url': base,
        'prepared_manifest_sha256': common.digest(prepared / 'prepared-live.json', MAX_JSON),
        'artifacts': manifest['artifacts'], 'source_verification_sha256': canonical_sha(verified),
        'electron': {'manifest_sha256': canonical_sha(runtime), 'executable_sha256': runtime['electron'],
                     'files': len(runtime), 'version': version},
        'process_seconds': 355, 'request_seconds': 240, 'maximum_requests': 2,
        'source_bundle_relation': 'esbuild_recorded_inputs_not_independently_rebuilt'})
    validate_local_bundle(prepared, identity, fresh=True)
    metadata = Path(metadata_root).absolute()
    require(metadata.parent.resolve() == metadata.parent and not metadata.exists() and not metadata.is_symlink(),
            'UI metadata output exists or is unsafe')
    metadata.mkdir(mode=0o700)
    common.write_json(metadata / 'source-verification.json', verified)
    common.write_json(metadata / 'electron-files.json', runtime)
    common.write_json(metadata / 'local-inputs.json', {'prepared_root': str(prepared), 'electron_dist': str(distribution)})
    # Identity is written last; a failed freeze never leaves an approved identity.
    common.write_json(metadata / 'identity.json', identity)
    return identity


def client_identity(root=common.ROOT):
    root = Path(root)
    entry = common.read_json(root / 'expected-manifest.json').get('client')
    require(type(entry) is dict and set(entry) == {'kind', 'identity_sha256'}
            and entry['kind'] == 'antimatter-ui', 'UI identity reference differs')
    common.require_sha(entry['identity_sha256'])
    require(common.digest(root / 'client/identity.json') == entry['identity_sha256'], 'frozen UI identity differs')
    identity = validate_identity(common.read_json(root / 'client/identity.json'))
    contract = common.read_json(root / 'qualification-contract.json')
    require(contract.get('profile') == identity['profile'] and contract.get('stream_client') == 'antimatter-ui'
            and contract.get('ui_client_host') in ('same', 'separate'),
            'UI identity used outside its separate profile')
    return identity


def client_host(root, identity, ready):
    """Bind direct loopback or dedicated SSH transport to the frozen host mode."""
    contract = common.read_json(Path(root) / 'qualification-contract.json')
    mode = contract.get('ui_client_host')
    require(contract.get('profile') == identity['profile'] and contract.get('stream_client') == 'antimatter-ui'
            and mode in ('same', 'separate'), 'UI client host contract differs')
    require(ready.get('host') == '127.0.0.1' and integer(ready.get('port'), 1024, 65535)
            and identifier(ready.get('boot_id')), 'UI broker host identity differs')
    same_boot = ready['boot_id'] == identity['local_host']['boot_id']
    require(same_boot == (mode == 'same'), 'UI broker boot differs from frozen host mode')
    if mode == 'same':
        require(ready['port'] == int(identity['base_url'].split(':')[2].split('/')[0]),
                'same-host UI broker port differs from frozen endpoint')
    return mode


def observe_process(pid, role, cgroup, *, proc_root=Path('/proc')):
    """Capture actual local process identity before waiting for its exit."""
    require(integer(pid, 1) and role in ('electron', 'tunnel', 'supervisor', 'electron-child'), 'invalid UI process')
    cgroup = Path(cgroup)
    require(cgroup.is_absolute() and cgroup.is_relative_to('/sys/fs/cgroup')
            and cgroup != Path('/sys/fs/cgroup') and '..' not in cgroup.parts, 'invalid UI cgroup')
    directory = Path(proc_root) / str(pid)
    with (directory / 'stat').open() as stream:
        stat = stream.read(8193)
    require(len(stat) <= 8192 and stat.startswith(str(pid) + ' ('), 'invalid local process stat')
    fields = stat.rpartition(') ')[2].split()
    require(len(fields) >= 20 and fields[19].isdigit(), 'missing local process start time')
    with (directory / 'cgroup').open() as stream:
        groups = stream.read(8193)
    expected = '0::/' + str(cgroup.relative_to('/sys/fs/cgroup'))
    require(groups.strip() == expected, 'UI process escaped local containment')
    with (directory / 'status').open() as stream:
        status = stream.read(65537)
    require(len(status) <= 65536 and re.search(r'^NoNewPrivs:\s+1$', status, re.M), 'UI process lacks NoNewPrivs')
    with (directory / 'cmdline').open('rb') as stream:
        command = stream.read(65537)
    require(0 < len(command) <= 65536 and command.endswith(b'\0'), 'invalid UI process command')
    with (directory / 'environ').open('rb') as stream:
        environment = stream.read(65537)
    require(len(environment) <= 65536, 'UI process environment exceeds its bound')
    entries = environment.split(b'\0')
    forbidden = (b'NODE_OPTIONS=', b'NODE_PATH=', b'LD_PRELOAD=', b'LD_LIBRARY_PATH=', b'ELECTRON_EXTRA_LAUNCH_ARGS=')
    require(not any(entry.startswith(forbidden) for entry in entries), 'UI process inherited executable environment overrides')
    require(not any(entry.startswith(b'ELECTRON_RUN_AS_NODE=') and entry != b'ELECTRON_RUN_AS_NODE='
                    for entry in entries), 'UI process is running as Node instead of Electron')
    return {'pid': pid, 'role': role, 'start_time_ticks': int(fields[19]),
            'command': [part.decode() for part in command[:-1].split(b'\0')], 'cgroup': str(cgroup)}


def _unit_state(unit):
    require(type(unit) is str and re.fullmatch(r'hag-ui-[a-z0-9-]{1,100}\.service', unit), 'invalid local UI unit')
    result = subprocess.run(['systemctl', '--user', 'show', unit,
        '--property=LoadState,ActiveState,SubState,MainPID,ControlGroup'], capture_output=True,
        text=True, timeout=5, check=False)
    require(result.returncode == 0 and not result.stderr.strip() and len(result.stdout) <= 16384,
            'cannot observe local UI unit retirement')
    return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)


def observe_retirement(process_path, output, *, proc_root=Path('/proc'), unit_reader=_unit_state):
    """Write an exclusive receipt from actual local observations, never remote PID guesses."""
    process_path = Path(process_path)
    process = common.read_json(process_path, MAX_JSON)
    host = observe_host()
    require(host == process.get('host'), 'UI retirement is running on a different host or boot')
    state = unit_reader(process['unit'])
    require(state.get('ActiveState') == 'inactive' and state.get('SubState') == 'dead'
            and state.get('MainPID') == '0' and state.get('ControlGroup') == '', 'UI unit has not retired')
    group = Path(process['cgroup'])
    require(group.is_absolute() and group.is_relative_to('/sys/fs/cgroup') and '..' not in group.parts
            and group != Path('/sys/fs/cgroup') and not group.exists() and not group.is_symlink(),
            'UI cgroup has not retired')
    pids = [item['pid'] for item in process['processes']]
    require(pids and len(set(pids)) == len(pids) and all(integer(pid, 1) for pid in pids), 'UI PID inventory differs')
    require(all(not (Path(proc_root) / str(pid)).exists() for pid in pids), 'recorded UI process remains present')
    receipt = {'schema': 1, 'kind': 'antimatter-ui-retirement', 'host': host,
        'run_id': process['run_id'], 'identity_sha256': process['identity_sha256'],
        'process_receipt_sha256': common.digest(process_path, MAX_JSON),
        'observed_monotonic_ns': time.monotonic_ns(), 'unit': process['unit'], 'state': state,
        'cgroup': process['cgroup'], 'cgroup_absent': True, 'recorded_pids_absent': sorted(pids)}
    common.write_json(Path(output), receipt)
    return receipt


def _response(record):
    require(record.get('method') == 'POST' and record.get('route') == '/v1/chat/completions'
            and type(record.get('status')) is int and record['status'] == 200
            and record.get('read_error') is None and record.get('body_truncated') is False
            and record.get('content_type') == 'text/event-stream', 'remote UI response is incomplete or differs')
    encoded = record.get('raw_body_base64')
    require(type(encoded) is str and len(encoded) <= 2 * MAX_BODY, 'remote UI response exceeds its bound')
    try:
        raw = base64.b64decode(encoded, validate=True)
        try:
            from .ui_history import http_body
        except ImportError:
            from ui_history import http_body
        value = http_body(record)
    except (ValueError, TypeError) as error:
        raise ValueError('invalid remote UI SSE bytes') from error
    require(len(raw) <= MAX_BODY and type(value) is dict and value.get('object') == 'chat.completion',
            'remote UI response schema differs')
    choices = value.get('choices')
    require(type(choices) is list and len(choices) == 1 and choices[0].get('index') == 0
            and choices[0].get('finish_reason') == 'stop', 'remote UI response did not reach EOS')
    message = choices[0].get('message')
    require(type(message) is dict and set(message) == {'role', 'content'} and message['role'] == 'assistant'
            and type(message['content']) is str and bool(message['content']), 'remote UI assistant differs')
    usage = value.get('usage')
    require(type(usage) is dict and set(usage) == {'prompt_tokens', 'completion_tokens', 'total_tokens'}
            and all(integer(v, 1) for v in usage.values()) and usage['completion_tokens'] < 128
            and usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'],
            'remote UI usage differs')
    return raw, value, message['content'], usage


def evidence_names():
    return ['client/identity.json', 'ui-report.json', 'ui-process.json', 'ui-retirement.json',
            'ui-supervision.json', 'ui-supervision-samples.jsonl']


UI_LIMITS = {'memory.max': '2147483648', 'memory.high': '1610612736', 'memory.swap.max': '0',
             'pids.max': '128', 'cpu.max': '25000 100000'}


def validate_ui_supervision(root, process):
    root = Path(root)
    summary = common.read_json(root / 'ui-supervision.json')
    require(type(summary) is dict and set(summary) == {'schema', 'kind', 'host', 'run_id', 'unit', 'limits',
        'exit_code', 'abort_reason', 'minimum_available_bytes', 'initial_available_bytes', 'memory_peak_bytes',
        'events_zero', 'swap_zero', 'peak_tctl_c', 'thermal_cutoff_c', 'process_receipt_sha256', 'samples_sha256', 'cooldown'},
        'UI supervision schema differs')
    require(summary['schema'] == 1 and summary['kind'] == 'antimatter-ui-supervision'
            and summary['host'] == process['host'] and summary['run_id'] == process['run_id']
            and summary['unit'] == process['unit'] and summary['limits'] == UI_LIMITS
            and type(summary['exit_code']) is int and summary['exit_code'] == 0 and summary['abort_reason'] is None
            and summary['events_zero'] is True and summary['swap_zero'] is True
            and summary['thermal_cutoff_c'] == 90
            and summary['process_receipt_sha256'] == common.digest(root / 'ui-process.json', MAX_JSON)
            and summary['samples_sha256'] == common.digest(root / 'ui-supervision-samples.jsonl', MAX_JSON),
            'UI supervision identity, limits, or completion differs')
    samples = []
    with common.regular_reader(root / 'ui-supervision-samples.jsonl', MAX_JSON) as stream:
        for index, line in enumerate(stream):
            require(index < 5000 and len(line) <= 16384 and line.endswith(b'\n'), 'UI supervision sample bound exceeded')
            sample = common.strict_json(line)
            require(type(sample) is dict and set(sample) == {'monotonic_ns', 'available_bytes', 'tctl_c', 'cgroup'},
                    'UI supervision sample schema differs')
            require(integer(sample['monotonic_ns'], 1) and integer(sample['available_bytes'], 4 * 1024 ** 3)
                    and type(sample['tctl_c']) in (int, float) and math.isfinite(sample['tctl_c'])
                    and 0 <= sample['tctl_c'] < 90, 'UI headroom or temperature exceeded its bound')
            if samples:
                delta = sample['monotonic_ns'] - samples[-1]['monotonic_ns']
                require(0 < delta <= 10 ** 9, 'UI supervision has an unordered or missing sampling interval')
            group = sample['cgroup']
            require(type(group) is dict and set(group) == {'memory.current', 'memory.peak', 'memory.events', 'memory.swap.current'}
                    and integer(group['memory.current'], 0, 1536 * 1024 ** 2 - 1)
                    and integer(group['memory.peak'], group['memory.current'], 1536 * 1024 ** 2 - 1)
                    and type(group['memory.swap.current']) is int and group['memory.swap.current'] == 0,
                    'UI cgroup memory exceeded its bound')
            events = group['memory.events']
            require(type(events) is dict and {'low', 'high', 'max', 'oom', 'oom_kill'} <= set(events)
                    and set(events) <= {'low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill', 'sock_throttled'}
                    and all(type(value) is int and value == 0 for value in events.values()),
                    'UI memory events are not clean')
            samples.append(sample)
    # Sampling continues while systemd retires the cgroup after the client ends.
    # Its cleanup tail does not extend the separate 355-second process deadline.
    require(len(samples) >= 2 and samples[0]['available_bytes'] >= 8 * 1024 ** 3
            and samples[0]['monotonic_ns'] <= process['started_monotonic_ns']
            and process['ended_monotonic_ns'] - 10 ** 9 <= samples[-1]['monotonic_ns']
            <= process['ended_monotonic_ns'] + 5 * 10 ** 9,
            'UI supervision did not cover the client lifetime')
    cooldown = summary['cooldown']
    require(type(cooldown) is list and 21 <= len(cooldown) <= 121, 'UI startup cooldown sample count differs')
    for index, sample in enumerate(cooldown):
        require(type(sample) is dict and set(sample) == {'monotonic_ns', 'tctl_c'}
                and integer(sample['monotonic_ns'], 1) and type(sample['tctl_c']) in (float, int)
                and math.isfinite(sample['tctl_c']) and 0 <= sample['tctl_c'] <= 80,
                'UI startup cooldown exceeded 80 C')
        if index:
            require(0 < sample['monotonic_ns'] - cooldown[index - 1]['monotonic_ns'] <= 10 ** 9,
                    'UI startup cooldown has a missing or unordered interval')
    require(cooldown[-1]['monotonic_ns'] - cooldown[0]['monotonic_ns'] >= 10 * 10 ** 9
            and 0 <= samples[0]['monotonic_ns'] - cooldown[-1]['monotonic_ns'] <= 5 * 10 ** 9,
            'UI startup cooldown is too short or stale')
    expected = {'initial_available_bytes': samples[0]['available_bytes'],
        'minimum_available_bytes': min(sample['available_bytes'] for sample in samples),
        'memory_peak_bytes': max(sample['cgroup']['memory.peak'] for sample in samples),
        'peak_tctl_c': max(sample['tctl_c'] for sample in samples)}
    require(all(summary[key] == value for key, value in expected.items()), 'UI supervision summary differs from raw samples')
    return {'sample_count': len(samples), **expected}


def validate_ui_process_evidence(root, expected_identity, local_bundle_root=None):
    """Join trusted local observation receipts to the remote broker's two replies.

    This checks a transferred local attestation. Only observe_retirement, executed
    on the named local host, can establish local process absence.
    """
    root = Path(root)
    identity = validate_identity(expected_identity)
    require(client_identity(root) == identity, 'UI evidence uses another frozen identity')
    identity_sha = common.digest(root / 'client/identity.json')
    ready = common.read_json(root / 'answers-ui-ready.json')
    listener = common.read_json(root / 'answers-ui-listener.json')
    mode = client_host(root, identity, ready)
    require(client_host(root, identity, listener) == mode, 'UI listener host mode differs')
    for value in (ready, listener):
        require(value.get('client_identity_sha256') == identity_sha and identifier(value.get('run_id'))
                and value.get('delivery') == profile_spec(identity['profile'])['delivery'], 'UI broker binding differs')
    require(ready['run_id'] == listener['run_id'] and ready.get('boot_id') == listener.get('boot_id')
            and identifier(ready.get('boot_id')), 'UI broker host/run identity differs')
    counts = {'request_count': 2, 'forwarded_count': 2, 'response_count': 2, 'rejected_count': 0}
    if identity['profile'] == UI_STOP_PROFILE:
        counts.update(response_count=1, cancelled_count=1)
    require(all(type(listener.get(key)) is int and listener[key] == value for key, value in counts.items())
            and listener.get('closed') is True, 'UI broker admitted extra or incomplete requests')
    report = common.read_json(root / 'ui-report.json', MAX_JSON)
    process = common.read_json(root / 'ui-process.json', MAX_JSON)
    retired = common.read_json(root / 'ui-retirement.json', MAX_JSON)
    require(type(process) is dict and set(process) == {'schema', 'kind', 'host', 'run_id',
        'identity_sha256', 'ready_sha256', 'report_sha256', 'unit', 'cgroup', 'processes',
        'started_monotonic_ns', 'ended_monotonic_ns', 'exit_code', 'timed_out', 'bundle_verified_before',
        'bundle_verified_after', 'electron_manifest_sha256'}, 'UI process receipt schema differs')
    require(type(process['schema']) is int and process['schema'] == 1 and process['kind'] == 'antimatter-ui-process'
            and process['host'] == identity['local_host'] and process['run_id'] == ready['run_id']
            and process['identity_sha256'] == identity_sha
            and process['ready_sha256'] == common.digest(root / 'answers-ui-ready.json')
            and process['report_sha256'] == common.digest(root / 'ui-report.json', MAX_JSON)
            and type(process['exit_code']) is int and process['exit_code'] == 0 and process['timed_out'] is False
            and process['bundle_verified_before'] is True and process['bundle_verified_after'] is True
            and process['electron_manifest_sha256'] == identity['electron']['manifest_sha256'],
            'UI process did not exit successfully with the frozen inputs')
    first, last = process['started_monotonic_ns'], process['ended_monotonic_ns']
    require(integer(first, 1) and integer(last, first) and last - first <= 355 * 10 ** 9,
            'UI process exceeded its monotonic deadline')
    require(type(process['unit']) is str and re.fullmatch(r'hag-ui-[a-z0-9-]{1,100}\.service', process['unit']),
            'UI unit identity differs')
    group = Path(process['cgroup'])
    require(group.is_absolute() and group.is_relative_to('/sys/fs/cgroup') and '..' not in group.parts
            and group != Path('/sys/fs/cgroup'), 'UI process cgroup differs')
    rows = process['processes']
    require(type(rows) is list and 2 <= len(rows) <= 128, 'UI process inventory differs')
    for row in rows:
        require(type(row) is dict and set(row) == {'pid', 'role', 'start_time_ticks', 'command', 'cgroup'}
                and integer(row['pid'], 1) and integer(row['start_time_ticks'], 1)
                and row['role'] in ('electron', 'tunnel', 'supervisor', 'electron-child')
                and row['cgroup'] == process['cgroup'] and type(row['command']) is list
                and row['command'] and all(type(arg) is str and '\x00' not in arg for arg in row['command']),
                'UI process observation differs')
    pids = [row['pid'] for row in rows]
    require(len(set(pids)) == len(pids) and sum(row['role'] == 'electron' for row in rows) == 1
            and sum(row['role'] == 'supervisor' for row in rows) == 1
            and sum(row['role'] == 'tunnel' for row in rows) == (1 if mode == 'separate' else 0),
            'UI process roles/PIDs differ')
    electron = next(row for row in rows if row['role'] == 'electron')
    command = electron['command']
    require(len(command) == 7 + len(profile_spec(identity['profile'])['flags'])
            and command[7:] == profile_spec(identity['profile'])['flags'] and Path(command[0]).name == 'electron'
            and command[1:3] == ['--no-sandbox', '--ozone-platform=headless']
            and command[4] == '--fixture' and Path(command[3]) == Path(command[5]) / 'launcher.mjs'
            and command[6] == '--live-base-url=' + identity['base_url'], 'UI observed launch command differs')
    if mode == 'separate':
        tunnel = next(row for row in rows if row['role'] == 'tunnel')['command']
        port = ready['port']
        local_port = identity['base_url'].split(':')[2].split('/')[0]
        require(len(tunnel) == 18 and Path(tunnel[0]).name == 'ssh'
                and tunnel[1:-1] == ['-N', '-T', '-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes',
                                     '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=5',
                                     '-o', 'ControlMaster=no', '-o', 'ControlPath=none', '-L',
                                     f'127.0.0.1:{local_port}:127.0.0.1:{port}']
                and re.fullmatch(r'[a-z_][a-z0-9_-]*@[A-Za-z0-9][A-Za-z0-9.-]*', tunnel[-1]),
                'UI observed tunnel command differs')
    require(type(retired.get('schema')) is int and retired.get('schema') == 1 and retired.get('kind') == 'antimatter-ui-retirement'
            and retired.get('host') == identity['local_host'] and retired.get('run_id') == ready['run_id']
            and retired.get('identity_sha256') == identity_sha
            and retired.get('process_receipt_sha256') == common.digest(root / 'ui-process.json', MAX_JSON)
            and integer(retired.get('observed_monotonic_ns'), last)
            and retired.get('unit') == process['unit'] and retired.get('cgroup') == process['cgroup']
            and retired.get('cgroup_absent') is True and retired.get('recorded_pids_absent') == sorted(pids),
            'UI local retirement receipt differs')
    state = retired.get('state', {})
    require(state.get('ActiveState') == 'inactive' and state.get('SubState') == 'dead'
            and state.get('MainPID') == '0' and state.get('ControlGroup') == '', 'UI local unit did not retire')
    require(report.get('passed') is True and report.get('actualApp') is True and report.get('actualPreload') is True
            and report.get('actualConversationStore') is True and report.get('rendererNetworkBlocked') is True
            and report.get('hardwareAcceleration') is False and report.get('networkBlocked') is False
            and report.get('modelExecution') == 'external-endpoint-unverified'
            and report.get('reload') is None and report.get('restart') is None,
            'UI report is not a successful live editor run')
    require(report.get('live') == {'profile': identity['profile'], 'baseUrl': identity['base_url'], 'backendQualified': False,
            'processSeconds': 355, 'requestSeconds': 240, 'maximumRequests': 2}, 'UI live report contract differs')
    require(report.get('process', {}).get('pid') == electron['pid']
            and identifier(report.get('process', {}).get('instanceId'))
            and report.get('process', {}).get('phase') is None, 'UI report process binding differs')
    require(integer(report.get('checkpointCount'), 2) and integer(report.get('hydrationCount'), 1),
            'UI report lacks durable renderer checkpoints')
    if local_bundle_root is not None:
        require(observe_host() == identity['local_host'], 'bundle verification requested on another host')
        validate_local_bundle(local_bundle_root, identity)
    supervision = validate_ui_supervision(root, process)
    summary = {'profile': identity['profile'], 'identity_sha256': identity_sha, 'run_id': ready['run_id'],
            'ui_client_host': mode, 'local_host': identity['local_host'], 'remote_boot_id': ready['boot_id'],
            'requests': 2, 'report_sha256': process['report_sha256'],
            'process_receipt_sha256': common.digest(root / 'ui-process.json', MAX_JSON),
            'retirement_sha256': common.digest(root / 'ui-retirement.json', MAX_JSON),
            'local_retirement_attested': True, 'supervision': supervision,
            'backend_qualification_requires_remote_closeout': True}
    return {'summary': summary, 'report': report, 'process': process, 'electron': electron}


def validate_ui_evidence(root, expected_identity, broker_records, local_bundle_root=None):
    observed = validate_ui_process_evidence(root, expected_identity, local_bundle_root)
    require(expected_identity['profile'] == UI_PROFILE, 'history validator requires history profile')
    report = observed['report']
    require(report.get('interruption') is None, 'history UI report contains interruption')
    pipeline = report.get('providerPipeline', {})
    require(pipeline.get('transport') == 'loopback-http', 'UI report used fixture transport')
    turns, requests = pipeline.get('turns'), report.get('requests')
    require(type(turns) is list and len(turns) == 2 and type(requests) is list and len(requests) == 2
            and type(broker_records) is list and len(broker_records) == 2, 'UI request count differs')
    expected_messages, assistant_ids, thread_id = [], [], None
    for index, (turn, request, remote) in enumerate(zip(turns, requests, broker_records)):
        require(remote.get('turn') == index + 1, 'UI remote turn order differs')
        raw, response, content, usage = _response(remote)
        wire = remote['request']
        require(type(wire) is dict and set(wire) == {'model', 'messages', 'max_tokens', 'stream'}
                and wire['model'] == 'hag-context-qwen3-0.6b' and response.get('model') == wire['model']
                and wire['max_tokens'] == 128 and wire['stream'] is True,
                'UI wire profile differs')
        require(usage['prompt_tokens'] == (467 if index == 0 else 491)
                and usage['total_tokens'] <= (594 if index == 0 else 618), 'UI usage exceeds admitted turn')
        require(turn.get('turn') == index and turn.get('wire') == wire and turn.get('request') == request
                and turn.get('rawResponse') == raw.decode('utf-8') and turn.get('usage') == usage
                and turn.get('responseStatus') == 200 and turn.get('transport') == 'sse'
                and turn.get('contentType') == 'text/event-stream' and turn.get('servedModel') == response['model']
                and turn.get('finishReason') == 'stop', 'UI transport differs from remote response bytes')
        require(turn.get('attemptCount') == 1 and turn.get('providerCalls') == 1 and turn.get('maxTokens') == 128
                and turn.get('completed') is True and turn.get('settled') is True and turn.get('outcome') == 'finished'
                and turn.get('timedOut') is False and turn.get('error') is None and turn.get('cancelRequests') == 0
                and turn.get('transportClosed') is True and turn.get('resumed') is bool(index)
                and turn.get('deadlineMs') == 240000 and integer(turn.get('transportFragments'), 1)
                and turn.get('transportBytes') == len(raw), 'UI turn completion/accounting differs')
        started, ended = turn.get('startedAt'), turn.get('settledAt')
        require(integer(started, 1) and integer(ended, started) and ended - started <= 240000,
                'UI turn exceeded its request deadline')
        chunks = turn.get('contentChunks')
        require(type(chunks) is list and chunks and all(type(value) is str for value in chunks)
                and ''.join(chunks) == content and turn.get('providerResult', {}).get('content') == content
                and turn.get('providerResult', {}).get('finishReason') == 'stop'
                and turn.get('result') == {'reason': 'finished', 'rounds': 1, 'toolCalls': 0, 'content': content},
                'UI callbacks/tool loop differ from actual response')
        run_id = request.get('requestId')
        require(type(run_id) is str and run_id and request.get('responseMessageId') == run_id
                and turn.get('requestId') == run_id and run_id not in assistant_ids
                and request.get('providerId') == 'hag-ui-fixture'
                and request.get('model') == wire['model'] and request.get('maxTokens') is None
                and request.get('temperature') is None and request.get('useTools') is True,
                'UI request identity differs')
        messages = request.get('messages')
        require(type(messages) is list and len(messages) == 1 + 2 * index
                and all(type(message) is dict and set(message) == {'id', 'role', 'content'} for message in messages)
                and [{key: message[key] for key in ('role', 'content')} for message in messages] == wire['messages'][1:],
                'UI visible history differs from admitted wire')
        if index:
            require(request.get('threadId') == thread_id and messages[:2] == expected_messages,
                    'UI follow-up did not carry its own raw first answer and IDs')
        else:
            thread_id = request.get('threadId')
            require(type(thread_id) is str and thread_id, 'missing UI thread identity')
        expected_messages = messages + [{'id': run_id, 'role': 'assistant', 'content': content}]
        assistant_ids.append(run_id)
    threads = report.get('persisted', {}).get('threads')
    require(type(threads) is list and len(threads) == 1 and threads[0].get('id') == thread_id,
            'UI persisted thread differs')
    saved = threads[0].get('messages')
    require(type(saved) is list and len(saved) == 4
            and [{key: message.get(key) for key in ('id', 'role', 'content')} for message in saved] == expected_messages,
            'UI durable history differs from actual replies')
    for index, message in enumerate(saved[1::2]):
        metadata = message.get('meta', {})
        require(metadata.get('failed') is False and integer(metadata.get('createdAt'), 1)
                and integer(metadata.get('finishedAt'), metadata['createdAt'])
                and metadata.get('chunks') == len(turns[index]['contentChunks']), 'UI completion checkpoint differs')
    return observed['summary']
