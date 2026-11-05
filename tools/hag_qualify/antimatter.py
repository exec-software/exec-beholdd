"""Freeze a historical Antimatter client; live identity never reads its checkout.

These private-era source pins are absent from the clean public client history.
They preserve old evidence identity, not a supported public run workflow. A new
public client requires a separate profile and fresh qualification; see
docs/hag-qualification.md#public-source-scope.
"""
import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile

try:
    from . import gate_common as common
except ImportError:
    import gate_common as common


SUPPORTED_COMMIT = '0679c442017083d8dfb9fd2b43c3de1a7eb27e99'
GROUNDED_COMMIT = '78147b78d8271b993dc3734f469f763df1f0f548'
PROFILE_COMMITS = {'backend-parity-c128': SUPPORTED_COMMIT, 'antimatter-grounded-c512': GROUNDED_COMMIT,
                   'antimatter-project-c2048': GROUNDED_COMMIT, 'antimatter-history-c2048': GROUNDED_COMMIT,
                   'antimatter-history-template-c2048': GROUNDED_COMMIT,
                   'antimatter-generated-history-c2048': GROUNDED_COMMIT}
ENTRY_POINT = 'src/api/providers/openai.ts'
# The complete source closure of this explicitly supported commit, including types.
SOURCE_PATHS = (ENTRY_POINT, 'src/api/providers/base.ts', 'src/api/providers/httpBody.ts',
                'src/api/toolArguments.ts', 'src/api/types.ts')
GROUNDED_SOURCE_PATHS = (*SOURCE_PATHS, 'src/api/tokenEstimate.ts')
MAX_SOURCE_BYTES = 1024 * 1024
MAX_TOOL_BYTES = 256 * 1024 ** 2
MAX_BUNDLE_BYTES = 2 * 1024 ** 2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_paths(commit):
    require(type(commit) is str and commit in (SUPPORTED_COMMIT, GROUNDED_COMMIT),
            'unsupported Antimatter commit')
    return GROUNDED_SOURCE_PATHS if commit == GROUNDED_COMMIT else SOURCE_PATHS


def validate_options(repo, commit, node, esbuild, profile):
    values = (repo, commit, node, esbuild)
    require(not any(value is not None for value in values) or all(value is not None for value in values),
            'Antimatter repository, commit, Node and esbuild must be supplied together')
    if profile in ('antimatter-grounded-c512', 'antimatter-project-c2048', 'antimatter-history-c2048',
                   'antimatter-history-template-c2048', 'antimatter-generated-history-c2048'):
        require(repo is not None, 'grounded profile requires the committed Antimatter client')
    if repo is not None:
        require(profile in PROFILE_COMMITS, 'Antimatter client requires an explicitly supported qualification profile')
        require(type(commit) is str and commit == PROFILE_COMMITS[profile], 'unsupported Antimatter commit/profile pair')
    return repo is not None


def _run(arguments, cwd, limit=MAX_SOURCE_BYTES):
    # Known Git/tool commands only; bounded file-backed output avoids capture_output
    # growing Python's heap. Preparation itself is never part of a live model stage.
    environment = {key: value for key, value in os.environ.items()
                   if key not in ('NODE_OPTIONS', 'NODE_PATH', 'LD_PRELOAD', 'LD_LIBRARY_PATH')}
    environment.update(GIT_OPTIONAL_LOCKS='0', GIT_NO_REPLACE_OBJECTS='1',
                       GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL='/dev/null')
    with tempfile.TemporaryFile() as output, tempfile.TemporaryFile() as errors:
        try:
            result = subprocess.run(arguments, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                    stdout=output, stderr=errors, timeout=30, check=False)
        except subprocess.TimeoutExpired as error:
            raise ValueError('client preparation command exceeded 30 seconds') from error
        output.seek(0)
        errors.seek(0)
        data, diagnostic = output.read(limit + 1), errors.read(65537)
    require(len(data) <= limit and len(diagnostic) <= 65536, 'client preparation command output exceeded its bound')
    require(result.returncode == 0, 'client preparation command failed: ' + diagnostic.decode('utf-8', errors='replace'))
    return data


def committed_sources(repo, commit):
    paths = source_paths(commit)
    prefix = ['/usr/bin/git', '-C', str(repo)]
    resolved = _run(prefix + ['rev-parse', '--verify', commit + '^{commit}'], repo, 128).decode().strip()
    require(resolved == commit, 'Antimatter commit did not resolve exactly')
    sources = {}
    for name in paths:
        entry = _run(prefix + ['ls-tree', commit, '--', name], repo, 1024).decode().strip()
        require(re.fullmatch(r'100644 blob [0-9a-f]{40}\t' + re.escape(name), entry) is not None,
                'committed client source is missing or not a regular file: ' + name)
        size_text = _run(prefix + ['cat-file', '-s', commit + ':' + name], repo, 32).decode().strip()
        require(size_text.isascii() and size_text.isdigit() and 0 < int(size_text) <= MAX_SOURCE_BYTES,
                'committed client source exceeds its byte bound: ' + name)
        data = _run(prefix + ['show', '--no-ext-diff', commit + ':' + name], repo, MAX_SOURCE_BYTES)
        require(len(data) == int(size_text), 'committed client source size differs: ' + name)
        sources[name] = data
    return sources


def _copy(source, destination, limit, executable=False):
    expected = common.digest(source, limit)
    with common.regular_reader(source, limit) as input_stream, destination.open('xb') as output:
        if executable:
            require(input_stream.read(4) == b'\x7fELF', 'client tools must be native ELF executables, not unfrozen wrappers')
            input_stream.seek(0)
        total = 0
        while chunk := input_stream.read(1024 * 1024):
            total += len(chunk)
            require(total <= limit, 'client input grew during copying')
            output.write(chunk)
    destination.chmod(0o700 if executable else 0o600)
    require(common.digest(destination, limit) == expected and common.digest(source, limit) == expected,
            'client input changed while freezing')
    return expected


def _version(path, kind, cwd):
    value = _run([str(path), '--version'], cwd, 128).decode('utf-8').strip()
    pattern = r'v([0-9]+)\.[0-9]+\.[0-9]+' if kind == 'node' else r'[0-9]+\.[0-9]+\.[0-9]+'
    match = re.fullmatch(pattern, value)
    require(match is not None and (kind != 'node' or int(match.group(1)) >= 20),
            'unsupported client tool version: ' + kind)
    return value


def validate_metafile(metadata, commit=SUPPORTED_COMMIT):
    require(type(metadata) is dict and set(metadata) == {'inputs', 'outputs'}, 'client build metadata schema differs')
    inputs, outputs = metadata['inputs'], metadata['outputs']
    allowed = {'source/' + name for name in source_paths(commit)}
    require(type(inputs) is dict and 'source/' + ENTRY_POINT in inputs and set(inputs) <= allowed,
            'client build imported a source outside the frozen closure')
    for value in inputs.values():
        require(type(value) is dict and type(value.get('imports')) is list, 'invalid client input metadata')
        require(all(type(item) is dict and item.get('external') is not True and item.get('path') in inputs
                    for item in value['imports']), 'client build imported external code')
    require(type(outputs) is dict and set(outputs) == {'provider.mjs'}, 'client build output inventory differs')
    output = outputs['provider.mjs']
    require(type(output) is dict and output.get('imports') == []
            and output.get('entryPoint') == 'source/' + ENTRY_POINT
            and 'OpenAICompatibleProvider' in output.get('exports', [])
            and set(output.get('inputs', {})) == set(inputs),
            'client bundle is not a self-contained provider')
    return sorted(name.removeprefix('source/') for name in inputs)


def prepare_client(repo, commit, node, esbuild, root):
    """Build only the frozen committed provider; never invoke a model or live client."""
    source_paths(commit)
    repo = Path(repo).expanduser().resolve(strict=True)
    node = Path(node).expanduser().resolve(strict=True)
    esbuild = Path(esbuild).expanduser().resolve(strict=True)
    require(repo.is_dir() and os.access(node, os.X_OK) and os.access(esbuild, os.X_OK),
            'client repository and executable tools must exist')
    sources = committed_sources(repo, commit)
    client = root / 'client'
    client.mkdir(mode=0o700)
    for name, content in sources.items():
        target = client / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(content)
        target.chmod(0o600)
    node_sha = _copy(node, client / 'node', MAX_TOOL_BYTES, executable=True)
    node_version = _version(client / 'node', 'node', client)
    with tempfile.TemporaryDirectory(prefix='hag-client-build-', dir=root) as directory:
        builder = Path(directory) / 'esbuild'
        builder_sha = _copy(esbuild, builder, MAX_TOOL_BYTES, executable=True)
        builder_version = _version(builder, 'esbuild', client)
        _run([str(builder), 'source/' + ENTRY_POINT, '--bundle', '--platform=node', '--format=esm',
              '--packages=external', '--outfile=provider.mjs', '--metafile=build-meta.json',
              '--tsconfig-raw={"compilerOptions":{}}', '--log-level=warning', '--log-limit=10'], client)
    bundle_inputs = validate_metafile(common.read_json(client / 'build-meta.json'), commit)
    require((client / 'provider.mjs').stat().st_size <= MAX_BUNDLE_BYTES, 'client bundle exceeds byte bound')
    manifest = {
        'schema': 1, 'kind': 'antimatter', 'commit': commit, 'entry_point': ENTRY_POINT,
        'sources': {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()},
        'bundle_inputs': bundle_inputs,
        'files': {'node': node_sha, 'provider.mjs': common.digest(client / 'provider.mjs', MAX_BUNDLE_BYTES),
                  'build-meta.json': common.digest(client / 'build-meta.json', MAX_SOURCE_BYTES)},
        'tools': {'node': {'sha256': node_sha, 'version': node_version},
                  'esbuild': {'sha256': builder_sha, 'version': builder_version}},
    }
    common.write_json(client / 'manifest.json', manifest)
    entry = {'kind': 'antimatter', 'manifest_sha256': common.digest(client / 'manifest.json')}
    _frozen_identity(root, entry)
    return entry


def _frozen_identity(root, entry):
    require(type(entry) is dict and set(entry) == {'kind', 'manifest_sha256'} and entry['kind'] == 'antimatter',
            'client manifest reference differs')
    common.require_sha(entry['manifest_sha256'])
    client = root / 'client'
    require(client.is_dir() and not client.is_symlink(), 'missing or unsafe frozen client directory')
    require(common.digest(client / 'manifest.json') == entry['manifest_sha256'], 'frozen client manifest differs')
    manifest = common.read_json(client / 'manifest.json')
    require(type(manifest) is dict and set(manifest) == {'schema', 'kind', 'commit', 'entry_point', 'sources',
                                                       'bundle_inputs', 'files', 'tools'}, 'client manifest schema differs')
    require(type(manifest['schema']) is int and manifest['schema'] == 1 and manifest['kind'] == 'antimatter'
            and type(manifest['commit']) is str and manifest['commit'] in PROFILE_COMMITS.values()
            and manifest['entry_point'] == ENTRY_POINT,
            'frozen client provenance differs')
    paths = source_paths(manifest['commit'])
    sources = manifest['sources']
    require(type(sources) is dict and set(sources) == set(paths), 'frozen client source inventory differs')
    for name, expected in sources.items():
        require(common.digest(client / 'source' / name, MAX_SOURCE_BYTES) == common.require_sha(expected),
                'frozen client source differs: ' + name)
    files = manifest['files']
    require(type(files) is dict and set(files) == {'node', 'provider.mjs', 'build-meta.json'},
            'frozen client artifact inventory differs')
    for name, expected in files.items():
        limit = MAX_TOOL_BYTES if name == 'node' else MAX_BUNDLE_BYTES if name == 'provider.mjs' else MAX_SOURCE_BYTES
        require(common.digest(client / name, limit) == common.require_sha(expected), 'frozen client artifact differs: ' + name)
    require(os.access(client / 'node', os.X_OK), 'frozen Node is not executable')
    tools = manifest['tools']
    require(type(tools) is dict and set(tools) == {'node', 'esbuild'}, 'client tool provenance differs')
    for kind, value in tools.items():
        require(type(value) is dict and set(value) == {'sha256', 'version'}, 'client tool provenance schema differs')
        common.require_sha(value['sha256'])
        version = value['version']
        pattern = r'v([0-9]+)\.[0-9]+\.[0-9]+' if kind == 'node' else r'[0-9]+\.[0-9]+\.[0-9]+'
        match = re.fullmatch(pattern, version) if type(version) is str else None
        require(match is not None and (kind != 'node' or int(match.group(1)) >= 20), 'frozen client tool version differs')
    require(tools['node']['sha256'] == files['node'], 'Node provenance does not match frozen executable')
    require(manifest['bundle_inputs'] == validate_metafile(common.read_json(client / 'build-meta.json'), manifest['commit']),
            'frozen client bundle closure differs')
    expected_files = {'manifest.json', *files, *('source/' + name for name in paths)}
    actual = set()
    for count, path in enumerate(client.rglob('*'), 1):
        require(count <= 64 and len(path.relative_to(client).parts) <= 6, 'frozen client tree exceeds inventory bound')
        require(not path.is_symlink(), 'symlink in frozen client inventory')
        if not path.is_dir():
            actual.add(str(path.relative_to(client)))
    require(actual == expected_files, 'unexpected file in frozen client directory')
    return {'kind': 'antimatter', 'commit': manifest['commit'], 'manifest_sha256': entry['manifest_sha256'],
            'node_sha256': files['node'], 'bundle_sha256': files['provider.mjs']}


def client_identity(root=common.ROOT):
    """Validate frozen inputs only; a live stage never consults the client checkout."""
    contract = common.approved_contract(root)
    manifest = common.read_json(root / 'expected-manifest.json')
    if 'client' not in manifest:
        require(contract['stream_client'] == 'python-http' and not (root / 'client').exists()
                and not (root / 'client').is_symlink(), 'contract requires a missing client snapshot')
        return None
    require(contract['stream_client'] == 'antimatter' and contract['profile'] in PROFILE_COMMITS,
            'client snapshot differs from the qualification contract')
    identity = _frozen_identity(root, manifest['client'])
    require(identity['commit'] == PROFILE_COMMITS[contract['profile']], 'client commit differs from profile')
    return identity
