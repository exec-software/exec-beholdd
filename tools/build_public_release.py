#!/usr/bin/env python3
"""Build public CPU/HAG artifacts in bounded, digest-pinned Linux containers.

This is a separate hosted-build supervisor, not a fallback for a locally refused
development run. It requires a fresh external work directory and clean Git source.
Only dependency fetching has network access. No GPU devices or model files enter
the containers; ordinary tests and an empty-registry HTTP smoke run offline.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
MEMORY = 3 * 1024**3
RESERVE = 8 * 1024**3
PIDS = 1152
TIMEOUT = 900
FEATURES = 'beholdd-cli/hag-openmythos,beholdd-memory/test-fault-injection'
HAG_UPSTREAM_SOURCES = ('src/linux/probe_packets.rs', 'src/linux/compute/packets.rs',
                        'src/linux/uapi.rs', 'src/linux/submission_uapi.rs')
REMAP = '\x1f'.join('--remap-path-prefix=' + item for item in (
    '/source=/usr/src/beholdd', '/work=/usr/src/beholdd-build',
    '/usr/local/cargo=/usr/src/cargo', '/usr/local/rustup=/usr/src/rust-toolchain',
))


def digest(path):
    """Return a file's SHA-256 without materializing large artifacts twice."""
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def available_memory(text):
    """Require the same host reserve as the development supervisor."""
    match = re.search(r'^MemAvailable:\s+(\d+) kB$', text, re.M)
    if not match:
        raise ValueError('Linux MemAvailable is unavailable')
    available = int(match[1]) * 1024
    if available < MEMORY + RESERVE:
        raise ValueError(f'need {MEMORY + RESERVE} available bytes; found {available}; no command started')
    return available


def admission():
    return available_memory(Path('/proc/meminfo').read_text())


def validate_cgroup(root=Path('/sys/fs/cgroup')):
    """Refuse an unbounded Docker invocation, including a direct --inside call."""
    quota, period = (root / 'cpu.max').read_text().split()
    if ((root / 'memory.max').read_text().strip() != str(MEMORY)
            or (root / 'memory.swap.max').read_text().strip() != '0'
            or (root / 'pids.max').read_text().strip() != str(PIDS)
            or quota == 'max' or int(quota) * 2 != int(period) * 3):
        raise ValueError('container memory/swap/CPU/PID limits differ from the release contract')


def safe_archive_members(archive):
    """Allow only regular committed files and directories in source snapshots."""
    members = archive.getmembers()
    for member in members:
        path = Path(member.name)
        if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
            raise ValueError('source archive contains an unsafe path or special entry')
    return members


def verify_binary(path, versions):
    if b'/home/' in path.read_bytes() or path.read_bytes()[:4] != b'\x7fELF':
        raise ValueError('release binary is not ELF or contains an unremapped home path')
    required = sorted(set(re.findall(r'\bGLIBC_(\d+\.\d+(?:\.\d+)?)\b', versions)))
    if not required or any(tuple(map(int, version.split('.'))) > (2, 34) for version in required):
        raise ValueError('release binary exceeds the glibc 2.34 ceiling or lacks version evidence')
    return required


def smoke(path):
    """Exercise the actual packaged executable without registering a model."""
    with tempfile.TemporaryDirectory() as temporary:
        config = Path(temporary) / 'empty.yaml'
        config.write_text('models: []\nruntime_headroom_bytes: 16777216\n')
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', 0))
            port = probe.getsockname()[1]
        with (Path(temporary) / 'server.log').open('w+') as log:
            process = subprocess.Popen([str(path), 'serve', '--config', str(config),
                                        '--host', '127.0.0.1', '--port', str(port)],
                                       stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as response:
                            if response.status != 200:
                                raise ValueError('health smoke failed')
                        break
                    except OSError:
                        if process.poll() is not None or time.monotonic() >= deadline:
                            raise ValueError('server did not become healthy')
                        time.sleep(0.05)
                def get(route):
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}{route}', timeout=2) as response:
                        return json.load(response)
                info = get('/v1/beholdd/info')
                if info.get('generation') != 2 or info.get('runtime') != 'native':
                    raise ValueError('unexpected native runtime identity')
                if info.get('execution_backends') != ['cpu'] or get('/v1/models').get('data') != []:
                    raise ValueError('empty registry unexpectedly enabled a model or device backend')
            finally:
                process.terminate()
                try:
                    result = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    raise ValueError('server did not exit after SIGTERM')
            if result != 0:
                raise ValueError('server smoke did not shut down cleanly')


def license_inventory(metadata, output):
    """Retain normal/build dependency notices for the exact CLI feature graph."""
    packages = {package['id']: package for package in metadata['packages']}
    nodes = {node['id']: node for node in metadata['resolve']['nodes']}
    pending = [metadata['resolve']['root']]
    visited = set()
    while pending:
        identifier = pending.pop()
        if identifier in visited:
            continue
        visited.add(identifier)
        pending.extend(dep['pkg'] for dep in nodes[identifier]['deps']
                       if any(kind['kind'] in (None, 'build') for kind in dep['dep_kinds']))
    records = []
    for identifier in sorted(visited):
        package = packages[identifier]
        directory = Path(package['manifest_path']).parent
        if not (directory.is_relative_to('/source') or directory.is_relative_to('/work/cargo/registry/src')):
            raise ValueError('dependency source is outside the reviewed source/registry roots')
        files = sorted(path for path in directory.iterdir()
                       if path.is_file() and re.match(r'^(LICENSE|LICENCE|COPYING|NOTICE|UNLICENSE)', path.name, re.I))
        declared = package.get('license_file')
        if declared:
            declared = Path(declared)
            if not declared.is_absolute():
                declared = directory / declared
            if not declared.resolve().is_relative_to(directory.resolve()):
                raise ValueError('declared dependency license escapes its package')
            if declared not in files:
                files.append(declared)
        if not files and package['source'] is None and directory.is_relative_to('/source/crates'):
            files = [Path('/source/LICENSE')]
        if not files or not package.get('license'):
            raise ValueError(f'missing license text or SPDX declaration for {package["name"]}')
        relative = f'{package["name"]}-{package["version"]}'
        target = output / relative
        target.mkdir(parents=True)
        notices = []
        for source in files:
            if source.is_symlink():
                raise ValueError('license symlinks require explicit review')
            destination = target / source.name
            if destination.exists():
                raise ValueError('duplicate license filename')
            shutil.copyfile(source, destination)
            notices.append({'file': f'{relative}/{source.name}', 'sha256': digest(destination)})
        records.append({'name': package['name'], 'version': package['version'],
                        'license': package['license'], 'features': nodes[identifier]['features'],
                        'source': package['source'] or 'workspace', 'notices': notices})
    (output / 'inventory.json').write_text(json.dumps(records, indent=2) + '\n')


def collect_rust_notices(sysroot, output, rustc_identity):
    """Retain notices actually supplied by the exact pinned Rust distribution."""
    directory = sysroot / 'share/doc/rust'
    scoped = directory / 'COPYRIGHT-library.html'
    if scoped.is_file():
        files = [scoped]
        scope = 'standard-library'
    else:
        files = [path for name in ['COPYRIGHT.html', 'COPYRIGHT']
                 if (path := directory / name).is_file()]
        scope = 'full-distribution'
    if not files:
        raise ValueError('pinned Rust distribution contains no standard-library or distribution copyright notice')
    files += sorted(path for path in directory.glob('LICENSE*') if path.is_file())
    output.mkdir(parents=True)
    records = []
    for source in files:
        if source.is_symlink() or not source.resolve().is_relative_to(sysroot.resolve()):
            raise ValueError('Rust notice source escapes its pinned distribution')
        target = output / source.name
        shutil.copyfile(source, target)
        records.append({'file': source.name, 'source': str(source.relative_to(sysroot)),
                        'sha256': digest(target), 'bytes': target.stat().st_size})
    receipt = {'scope': scope, 'rustc': rustc_identity, 'files': records,
               'origin': 'Unmodified notices from the digest-pinned Rust builder image.'}
    (output / 'provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def collect_hag_upstream(source, output):
    """Keep exact adapted-source notices and their canonical snapshot provenance."""
    provenance = json.loads((source / 'provenance.json').read_text())
    identity = provenance.get('source', {})
    if (identity.get('repository') != 'https://github.com/exec-software/exec-hag'
            or identity.get('dirty') is not False
            or not re.fullmatch(r'[a-f0-9]{40}', identity.get('revision', ''))):
        raise ValueError('HAG upstream notices require clean canonical source provenance')
    files = (*HAG_UPSTREAM_SOURCES, 'thirdparty.md')
    for name in files:
        path = source / name
        record = provenance.get('files', {}).get(name, {})
        if (path.is_symlink() or not path.resolve().is_relative_to(source.resolve())
                or not path.is_file() or digest(path) != record.get('sha256')
                or path.stat().st_size != record.get('bytes')):
            raise ValueError('HAG upstream notice source is absent or differs from its snapshot')
    output.mkdir(parents=True)
    for name in (*files, 'provenance.json'):
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, destination)


def download_python_wheel(pin, directory):
    """Fetch only the pinned wheel, rejecting truncation, excess bytes and hash drift."""
    if (Path(pin['filename']).name != pin['filename']
            or not pin['url'].startswith('https://files.pythonhosted.org/packages/')
            or not re.fullmatch(r'[a-f0-9]{64}', pin['sha256'])
            or not 0 < pin['bytes'] < 100 * 1024**2):
        raise ValueError('invalid Python wheel pin')
    directory.mkdir(parents=True)
    target = directory / pin['filename']
    partial = target.with_suffix(target.suffix + '.part')
    count = 0
    try:
        with urllib.request.urlopen(pin['url'], timeout=30) as response, partial.open('xb') as output:
            for block in iter(lambda: response.read(1024 * 1024), b''):
                count += len(block)
                if count > pin['bytes']:
                    raise ValueError('Python wheel exceeds its pinned size')
                output.write(block)
        if count != pin['bytes'] or digest(partial) != pin['sha256']:
            raise ValueError('Python wheel differs from its pinned size or hash')
        partial.rename(target)
    finally:
        partial.unlink(missing_ok=True)


def inside(arguments):
    validate_cgroup()
    action, *args = arguments
    if action == 'run':
        return subprocess.run(args, check=False).returncode
    if action == 'toolchain':
        commands = {'rustc': ['rustc', '-vV'], 'cargo': ['cargo', '-V'],
                    'gcc': ['gcc', '--version'], 'glibc': ['getconf', 'GNU_LIBC_VERSION'],
                    'python': ['python3', '--version'], 'node': ['node', '--version']}
        facts = {key: subprocess.check_output(command, text=True).strip() for key, command in commands.items()}
        if (facts['glibc'] != 'glibc 2.31' or not facts['rustc'].startswith('rustc 1.98.0 ')
                or facts['python'] != 'Python 3.12.10' or facts['node'] != 'v22.20.0'):
            raise ValueError('builder toolchain differs from its pinned contract')
        sysroot = Path(subprocess.check_output(['rustc', '--print', 'sysroot'], text=True).strip())
        facts['rustRuntimeNotices'] = collect_rust_notices(sysroot, Path('/work/rust-runtime-notices'), facts['rustc'])
        Path('/work/toolchain.json').write_text(json.dumps(facts, indent=2) + '\n')
    elif action in ('fetch-python', 'install-python'):
        pin = json.loads(Path('/source/tools/release/python-wheel.json').read_text())
        wheel = Path('/work/python-wheels') / pin['filename']
        if action == 'fetch-python':
            download_python_wheel(pin, wheel.parent)
        else:
            if wheel.stat().st_size != pin['bytes'] or digest(wheel) != pin['sha256']:
                raise ValueError('downloaded Python wheel no longer matches its pin')
            requirements = Path('/work/python-requirements.txt')
            requirements.write_text(f'{wheel} --hash=sha256:{pin["sha256"]}\n')
            subprocess.run(['python3', '-m', 'pip', 'install', '--no-index', '--no-deps', '--no-cache-dir',
                            '--require-hashes', '--only-binary=:all:', '--target', '/work/python-deps',
                            '--requirement', str(requirements)], check=True)
            version = subprocess.check_output(['python3', '-c', 'import numpy; print(numpy.__version__)'], text=True).strip()
            if version != pin['version']:
                raise ValueError('installed NumPy differs from its pinned version')
            Path('/work/test-dependencies.json').write_text(json.dumps({'numpy': pin}, indent=2) + '\n')
    elif action == 'component':
        component = args[0]
        if component not in ('cpu', 'hag'):
            raise ValueError('unknown release component')
        output = Path('/work/bundle') / component
        output.mkdir(parents=True)
        (output / 'bin').mkdir()
        binary = output / 'bin/beholdd'
        shutil.copyfile('/work/target/release/beholdd', binary)
        binary.chmod(0o755)
        versions = subprocess.check_output(['readelf', '--version-info', '--wide', str(binary)], text=True)
        required = verify_binary(binary, versions)
        smoke(binary)
        command = ['cargo', 'metadata', '--locked', '--offline', '--format-version', '1',
                   '--filter-platform', 'x86_64-unknown-linux-gnu', '--manifest-path',
                   '/source/crates/beholdd-cli/Cargo.toml', '--no-default-features']
        if component == 'hag':
            command += ['--features', 'hag-native']
        metadata = json.loads(subprocess.check_output(command))
        licenses = output / 'licenses'
        licenses.mkdir()
        license_inventory(metadata, licenses)
        shutil.copytree('/work/rust-runtime-notices', licenses / 'rust-standard-library')
        (licenses / 'chat-templates').mkdir()
        shutil.copyfile('/source/crates/beholdd-server/src/templates/README.md', licenses / 'chat-templates/README.md')
        shutil.copyfile('/source/LICENSE', licenses / 'chat-templates/LICENSE-APACHE-2.0')
        if component == 'hag':
            collect_hag_upstream(Path('/source/vendor/exec-hag'), licenses / 'hag-upstream')
        receipt = {'cargoFeatures': ['hag-native'] if component == 'hag' else [],
                   'binary': {'path': f'{component}/bin/beholdd', 'sha256': digest(binary), 'bytes': binary.stat().st_size},
                   'requiredGlibcVersions': required, 'emptyRegistryHttpSmoke': 'passed'}
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    elif action == 'hag-library':
        source = Path('/source/vendor/exec-hag')
        provenance = json.loads((source / 'provenance.json').read_text())
        if provenance['source']['dirty'] or provenance['source']['repository'] != 'https://github.com/exec-software/exec-hag':
            raise ValueError('standalone HAG library needs clean canonical provenance')
        for name, record in provenance['files'].items():
            if digest(source / name) != record['sha256'] or (source / name).stat().st_size != record['bytes']:
                raise ValueError('vendored HAG file differs from its source receipt')
        output = Path('/work/bundle/hag-library')
        (output / 'lib').mkdir(parents=True)
        (output / 'include').mkdir()
        files = {}
        for name in ['libexec_hag.so', 'libexec_hag.a']:
            destination = output / 'lib' / name
            shutil.copyfile(Path('/work/hag-target/release') / name, destination)
            if b'/home/' in destination.read_bytes():
                raise ValueError('HAG library contains an unremapped home path')
            files['lib/' + name] = {'sha256': digest(destination), 'bytes': destination.stat().st_size}
        shutil.copyfile(source / 'include/exec_hag.h', output / 'include/exec_hag.h')
        for name in ['LICENSE', 'Cargo.lock', 'VERSION', 'provenance.json', 'thirdparty.md']:
            shutil.copyfile(source / name, output / name)
        versions = subprocess.check_output(['readelf', '--version-info', '--wide', str(output / 'lib/libexec_hag.so')], text=True)
        required = verify_binary(output / 'lib/libexec_hag.so', versions)
        with tempfile.TemporaryDirectory() as temporary:
            code = Path(temporary) / 'smoke.c'
            code.write_text('#include "exec_hag.h"\n#include <string.h>\nint main(void) {\n'
                            'return strcmp(hag_error_text(HAG_OK), "ok") || !hag_default_dispatch_groups() '
                            '|| !hag_min_kernel_vgprs();\n}\n')
            for suffix in ['so', 'a']:
                executable = Path(temporary) / ('smoke-' + suffix)
                subprocess.run(['gcc', '-Wall', '-Wextra', '-Werror', '-I', str(output / 'include'),
                                str(code), str(output / ('lib/libexec_hag.' + suffix)),
                                '-ldl', '-lpthread', '-lm', '-lrt', '-lutil', '-o', str(executable)], check=True)
                environment = {**os.environ, 'LD_LIBRARY_PATH': str(output / 'lib') + ':/usr/local/lib'}
                subprocess.run([str(executable)], check=True, env=environment, timeout=10)
        metadata = json.loads(subprocess.check_output(['cargo', 'metadata', '--locked', '--offline',
                              '--format-version', '1', '--filter-platform', 'x86_64-unknown-linux-gnu',
                              '--manifest-path', str(source / 'Cargo.toml'), '--features', 'capi']))
        licenses = output / 'licenses'
        licenses.mkdir()
        license_inventory(metadata, licenses)
        shutil.copytree('/work/rust-runtime-notices', licenses / 'rust-standard-library')
        collect_hag_upstream(source, licenses / 'hag-upstream')
        receipt = {'source': provenance['source'], 'version': (source / 'VERSION').read_text().strip(),
                   'cargoFeatures': ['capi'], 'cargoLockSha256': digest(source / 'Cargo.lock'),
                   'files': files, 'headerSha256': digest(output / 'include/exec_hag.h'),
                   'requiredGlibcVersions': required, 'cDynamicAndStaticLinkSmoke': 'passed',
                   'scope': 'No GPU device opened; library calls in smoke query constants only.'}
        (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    else:
        raise ValueError('unknown contained operation')
    return 0


class Builder:
    def __init__(self, work):
        self.work = work
        self.source = work / 'source'
        self.logs = work / 'logs'
        self.image = 'beholdd-public-build:' + uuid.uuid4().hex
        self.stages = []

    def stage(self, name, command, network=False):
        available = admission()
        container = 'beholdd-build-' + uuid.uuid4().hex
        arguments = ['docker', 'create', '--name', container, '--platform', 'linux/amd64',
                     '--memory', str(MEMORY), '--memory-swap', str(MEMORY), '--cpus', '1.5',
                     '--pids-limit', str(PIDS), '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
                     '--read-only', '--tmpfs', '/tmp:rw,nosuid,nodev,noexec,size=268435456',
                     '--network', 'bridge' if network else 'none', '--user', f'{os.getuid()}:{os.getgid()}',
                     '--mount', f'type=bind,source={self.source},target=/source,readonly',
                     '--mount', f'type=bind,source={self.work / "work"},target=/work',
                     '--workdir', '/source']
        environment = {'HOME': '/work/home', 'CARGO_HOME': '/work/cargo', 'CARGO_TARGET_DIR': '/work/target',
                       'CARGO_BUILD_JOBS': '1', 'RUST_TEST_THREADS': '1', 'CARGO_ENCODED_RUSTFLAGS': REMAP,
                       # Rust doctests, C link smokes and Python fixtures execute temporary files.
                       # Keep these inside the bounded writable work mount; /tmp stays noexec.
                       'PYTHONDONTWRITEBYTECODE': '1', 'TMPDIR': '/work/tmp', 'PYTHONPATH': '/work/python-deps',
                       'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                       'BLIS_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
        for key, value in environment.items():
            arguments += ['--env', key + '=' + value]
        arguments += [self.image, 'python3', '/source/tools/build_public_release.py', '--inside', *command]
        print('Starting bounded stage:', name, flush=True)
        log = self.logs / (name + '.log')
        try:
            subprocess.run(arguments, check=True, stdout=subprocess.DEVNULL, timeout=60)
            with log.open('wb') as handle:
                result = subprocess.run(['docker', 'start', '--attach', container], stdout=handle,
                                        stderr=subprocess.STDOUT, timeout=TIMEOUT)
            state = json.loads(subprocess.check_output(['docker', 'inspect', container]))[0]['State']
            if result.returncode or state['OOMKilled'] or state['ExitCode']:
                print(log.read_text(errors='replace')[-12000:], file=sys.stderr)
                raise ValueError(f'{name} failed; no retry or uncontained fallback')
            summaries = [line for line in log.read_text(errors='replace').splitlines()
                         if re.match(r'^(test result:|Ran \d+ tests|# (tests|pass|fail|skipped) )', line)]
            for summary in summaries:
                print(summary, flush=True)
            self.stages.append({'name': name, 'hostAvailableBytes': available, 'network': network,
                                'exitCode': state['ExitCode'], 'oomKilled': state['OOMKilled'],
                                'logSha256': digest(log), 'testSummary': summaries})
        finally:
            subprocess.run(['docker', 'rm', '--force', container], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, timeout=30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path)
    parser.add_argument('--inside', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.inside:
        return inside(args.inside)
    if args.work_dir is None:
        parser.error('--work-dir is required')
    work = args.work_dir.resolve()
    if work.exists() or ROOT == work or ROOT in work.parents or any(char in str(work) for char in ',\n\r'):
        raise ValueError('work directory must be new and outside the source checkout')
    admission()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT):
        raise ValueError('release source must be clean')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tree = subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip()
    version = (ROOT / 'VERSION').read_text().strip()
    if not re.fullmatch(r'v[1-9][0-9]*', version):
        raise ValueError('invalid rolling release version')
    work.mkdir(parents=True)
    builder = Builder(work)
    for directory in [builder.source, builder.logs, work / 'work', work / 'work/home',
                      work / 'work/tmp', work / 'output']:
        directory.mkdir()
    source_tar = subprocess.check_output(['git', 'archive', '--format=tar', 'HEAD'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(source_tar)) as archive:
        archive.extractall(builder.source, members=safe_archive_members(archive))
    dockerfile = builder.source / 'tools/release/Dockerfile'
    images = re.findall(r'^FROM (\S+)', dockerfile.read_text(), re.M)
    if len(images) != 3 or any(not re.fullmatch(r'[a-z0-9:.-]+@sha256:[a-f0-9]{64}', image) for image in images):
        raise ValueError('every builder image must have an immutable digest')
    for image in images:
        admission()
        subprocess.run(['docker', 'pull', '--platform', 'linux/amd64', image], check=True, timeout=600)
    subprocess.run(['docker', 'build', '--network=none', '--pull=false', '--platform=linux/amd64',
                    '--tag', builder.image, '--file', str(dockerfile), str(dockerfile.parent)],
                   check=True, timeout=600)
    builder.stage('toolchain', ['toolchain'])
    builder.stage('fetch-python', ['fetch-python'], network=True)
    builder.stage('install-python', ['install-python'])
    builder.stage('python-tests', ['run', 'python3', '-m', 'unittest', 'discover', '-s', 'tools', '-p', 'test_*.py'])
    scripts = sorted('/source/' + str(path.relative_to(builder.source)) for path in (builder.source / 'tools').glob('test_*.mjs'))
    builder.stage('javascript-tests', ['run', 'node', '--test', '--test-concurrency=1', *scripts])
    builder.stage('fetch', ['run', 'cargo', 'fetch', '--locked', '--target', 'x86_64-unknown-linux-gnu'], network=True)
    builder.stage('fetch-hag-library', ['run', 'cargo', 'fetch', '--locked', '--target', 'x86_64-unknown-linux-gnu',
                                      '--manifest-path', '/source/vendor/exec-hag/Cargo.toml'], network=True)
    builder.stage('rust-tests', ['run', 'cargo', 'test', '--workspace', '--locked', '--offline'])
    builder.stage('feature-tests', ['run', 'cargo', 'test', '--workspace', '--locked', '--offline', '--features', FEATURES])
    for component in ['cpu', 'hag']:
        command = ['run', 'cargo', 'build', '--release', '--locked', '--offline', '-p', 'beholdd-cli']
        if component == 'hag':
            command += ['--features', 'hag-native']
        builder.stage('build-' + component, command)
        builder.stage('verify-' + component, ['component', component])
    hag_arguments = ['--manifest-path', '/source/vendor/exec-hag/Cargo.toml', '--locked', '--offline',
                     '--target-dir', '/work/hag-target']
    builder.stage('hag-library-tests', ['run', 'cargo', 'test', *hag_arguments, '--all-features'])
    builder.stage('build-hag-library', ['run', 'cargo', 'build', *hag_arguments, '--release', '--features', 'capi'])
    builder.stage('verify-hag-library', ['hag-library'])
    bundle = work / 'work/bundle'
    shutil.copytree(builder.source / 'docs', bundle / 'docs')
    for name in ['README.md', 'LICENSE', 'Cargo.lock', 'VERSION', 'config.example.yaml',
                 'config.dev.example.yaml', 'config.hag.example.yaml']:
        shutil.copyfile(builder.source / name, bundle / name)
    records = {str(path.relative_to(builder.source)): {'sha256': digest(path), 'bytes': path.stat().st_size,
               'executable': bool(path.stat().st_mode & 0o111)} for path in sorted(builder.source.rglob('*')) if path.is_file()}
    source_inventory = json.dumps(records, sort_keys=True, separators=(',', ':')).encode()
    (bundle / 'source-files.json').write_bytes(source_inventory + b'\n')
    manifest = {'schemaVersion': 1, 'source': {'repository': 'https://github.com/exec-software/exec-beholdd',
                'commit': head, 'tree': tree, 'dirty': False, 'version': version,
                'cargoLockSha256': digest(builder.source / 'Cargo.lock'),
                'inventorySha256': digest(bundle / 'source-files.json')},
                'toolchain': json.loads((work / 'work/toolchain.json').read_text()),
                'testDependencies': json.loads((work / 'work/test-dependencies.json').read_text()),
                'builderImages': images, 'dockerfileSha256': digest(dockerfile),
                'containerLimits': {'memoryBytes': MEMORY, 'swapBytes': 0, 'cpus': 1.5,
                                    'pids': PIDS, 'timeoutSeconds': TIMEOUT, 'hostReserveBytes': RESERVE,
                                    'blasThreads': 1},
                'rustPathRemapping': REMAP.split('\x1f'), 'stages': builder.stages,
                'testFeatures': FEATURES.split(','),
                'components': {component: json.loads((bundle / component / 'receipt.json').read_text()) for component in ['cpu', 'hag']},
                'hagLibrary': json.loads((bundle / 'hag-library/receipt.json').read_text()),
                'scope': 'Hardware-free tests and empty-registry HTTP smoke; no model/GPU qualification.'}
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    artifacts = {str(path.relative_to(bundle)): {'sha256': digest(path), 'bytes': path.stat().st_size}
                 for path in sorted(bundle.rglob('*')) if path.is_file()}
    (bundle / 'files.json').write_text(json.dumps(artifacts, indent=2) + '\n')
    archive_path = work / 'output' / f'beholdd-{version}-linux-x86_64-build.tar.gz'
    def normalize(member):
        member.uid = member.gid = 0
        member.uname = member.gname = ''
        member.mtime = 0
        member.mode = 0o755 if member.isdir() or member.name.endswith('/bin/beholdd') else 0o644
        return member
    with tarfile.open(archive_path, 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            archive.add(path, arcname=str(path.relative_to(bundle)), recursive=False, filter=normalize)
    (work / 'output/SHA256SUMS').write_text(f'{digest(archive_path)}  {archive_path.name}\n')
    print('Verified build artifact:', archive_path.name)
    return 0


if __name__ == '__main__':
    # Workflow cancellation must unwind stage cleanup and remove its container.
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'public release build: {error}', file=sys.stderr)
        sys.exit(1)
