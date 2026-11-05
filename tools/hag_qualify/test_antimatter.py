"""Small model-free client snapshot tests; no real compiler or executable runs."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

try:
    from . import antimatter as client
    from . import gate_common as common
    from . import inputs
    from .profiles import fixture_document
except ImportError:
    import antimatter as client
    import gate_common as common
    import inputs
    from profiles import fixture_document


class ClientSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='HAG client snapshot ')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.repo = self.root / 'dirty client checkout'
        self.repo.mkdir()
        self.node, self.esbuild = self.root / 'node input', self.root / 'esbuild input'
        for path in (self.node, self.esbuild):
            path.write_bytes(b'\x7fELF fixture ' + path.name.encode())
            path.chmod(0o700)
        self.sources = {name: ('// committed ' + name + '\n').encode() for name in client.SOURCE_PATHS}
        self.bundle = b'export class OpenAICompatibleProvider {}\n'

    def metadata(self):
        paths = ['source/' + name for name in self.sources if not name.endswith('/types.ts')]
        return {'inputs': {name: {'bytes': 12, 'imports': []} for name in paths},
                'outputs': {'provider.mjs': {'bytes': len(self.bundle), 'imports': [],
                    'exports': ['OpenAICompatibleProvider'], 'entryPoint': 'source/' + client.ENTRY_POINT,
                    'inputs': {name: {'bytesInOutput': 1} for name in paths}}}}

    def command(self, arguments, cwd, limit=client.MAX_SOURCE_BYTES):
        if arguments[-1] == '--version':
            return b'v22.15.0\n' if Path(arguments[0]).name == 'node' else b'0.25.12\n'
        self.assertIn('--packages=external', arguments)
        self.assertIn('--tsconfig-raw={"compilerOptions":{}}', arguments)
        self.assertEqual(arguments[1], 'source/' + client.ENTRY_POINT)
        (cwd / 'provider.mjs').write_bytes(self.bundle)
        (cwd / 'build-meta.json').write_text(json.dumps(self.metadata()))
        return b''

    def prepare(self, commit=client.SUPPORTED_COMMIT, profile='backend-parity-c128'):
        with patch.object(client, 'committed_sources', return_value=self.sources) as sources, \
             patch.object(client, '_run', side_effect=self.command):
            entry = client.prepare_client(self.repo, commit, self.node, self.esbuild, self.root)
        sources.assert_called_once_with(self.repo, commit)
        common.write_json(self.root / 'qualification-contract.json',
                          inputs.profile_contract(profile, 'antimatter'))
        common.write_json(self.root / 'answer-fixtures.json', fixture_document(profile))
        common.write_json(self.root / 'expected-manifest.json', {'client': entry})
        return entry

    def reseal(self, manifest):
        path = self.root / 'client/manifest.json'
        path.write_text(json.dumps(manifest))
        (self.root / 'expected-manifest.json').write_text(json.dumps(
            {'client': {'kind': 'antimatter', 'manifest_sha256': common.digest(path)}}))

    def test_freeze_records_committed_closure_tool_hashes_versions_and_self_contained_bundle(self):
        entry = self.prepare()
        identity = client.client_identity(self.root)
        self.assertEqual(identity, {'kind': 'antimatter', 'commit': client.SUPPORTED_COMMIT,
                                   'manifest_sha256': entry['manifest_sha256'],
                                   'node_sha256': common.digest(self.node),
                                   'bundle_sha256': hashlib.sha256(self.bundle).hexdigest()})
        manifest = common.read_json(self.root / 'client/manifest.json')
        self.assertEqual(manifest['tools']['node']['version'], 'v22.15.0')
        self.assertEqual(manifest['tools']['esbuild'], {'sha256': common.digest(self.esbuild), 'version': '0.25.12'})
        self.assertEqual(set(manifest['sources']), set(client.SOURCE_PATHS))
        self.assertNotIn('src/api/types.ts', manifest['bundle_inputs'])
        self.assertEqual((self.root / 'client/node').stat().st_mode & 0o777, 0o700)
        self.assertFalse(list(self.root.glob('hag-client-build-*')))

    def test_grounded_commit_freezes_estimator_and_refuses_cross_profile_substitution(self):
        self.sources['src/api/tokenEstimate.ts'] = b'export const estimate = 1;\n'
        self.prepare(client.GROUNDED_COMMIT, 'antimatter-grounded-c512')
        identity = client.client_identity(self.root)
        self.assertEqual(identity['commit'], client.GROUNDED_COMMIT)
        manifest = common.read_json(self.root / 'client/manifest.json')
        self.assertEqual(set(manifest['sources']), set(client.GROUNDED_SOURCE_PATHS))
        self.assertIn('src/api/tokenEstimate.ts', manifest['bundle_inputs'])
        estimator = self.root / 'client/source/src/api/tokenEstimate.ts'
        estimator.write_bytes(estimator.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            client.client_identity(self.root)
        estimator.write_bytes(self.sources['src/api/tokenEstimate.ts'])
        (self.root / 'qualification-contract.json').write_text(json.dumps(
            inputs.profile_contract('backend-parity-c128', 'antimatter')))
        (self.root / 'answer-fixtures.json').write_text(json.dumps(fixture_document('backend-parity-c128')))
        with self.assertRaisesRegex(ValueError, 'commit.*profile'):
            client.client_identity(self.root)

    def test_project_profile_requires_and_freezes_the_capacity_aware_commit(self):
        profile = 'antimatter-project-c2048'
        for options in ((None, None, None, None),
                        (self.repo, client.SUPPORTED_COMMIT, self.node, self.esbuild)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                client.validate_options(*options, profile)
        self.assertTrue(client.validate_options(self.repo, client.GROUNDED_COMMIT,
                                               self.node, self.esbuild, profile))
        self.sources['src/api/tokenEstimate.ts'] = b'export const estimate = 1;\n'
        self.prepare(client.GROUNDED_COMMIT, profile)
        identity = client.client_identity(self.root)
        self.assertEqual(identity['commit'], client.GROUNDED_COMMIT)
        self.assertEqual(set(common.read_json(self.root / 'client/manifest.json')['sources']),
                         set(client.GROUNDED_SOURCE_PATHS))
        with patch.object(client, '_run', side_effect=AssertionError('no live checkout/tools')):
            self.assertEqual(client.client_identity(self.root), identity)
        estimator = self.root / 'client/source/src/api/tokenEstimate.ts'
        estimator.write_bytes(estimator.read_bytes() + b'changed')
        with self.assertRaises(ValueError):
            client.client_identity(self.root)

    def test_old_commit_cannot_import_new_estimator_or_new_profile(self):
        self.sources['src/api/tokenEstimate.ts'] = b'export const estimate = 1;\n'
        with self.assertRaisesRegex(ValueError, 'outside the frozen closure'):
            client.validate_metafile(self.metadata(), client.SUPPORTED_COMMIT)
        self.assertIn('src/api/tokenEstimate.ts', client.validate_metafile(self.metadata(), client.GROUNDED_COMMIT))
        for profile, commit in (('backend-parity-c128', client.GROUNDED_COMMIT),
                                ('antimatter-grounded-c512', client.SUPPORTED_COMMIT)):
            with self.subTest(profile=profile), self.assertRaises(ValueError):
                client.validate_options(self.repo, commit, self.node, self.esbuild, profile)

    def test_history_requires_capacity_aware_snapshot_and_rechecks_its_frozen_closure(self):
        profile = 'antimatter-history-c2048'
        for options in ((None, None, None, None),
                        (self.repo, client.SUPPORTED_COMMIT, self.node, self.esbuild)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                client.validate_options(*options, profile)
        self.assertTrue(client.validate_options(self.repo, client.GROUNDED_COMMIT,
                                               self.node, self.esbuild, profile))
        self.sources['src/api/tokenEstimate.ts'] = b'export const estimate = 1;\n'
        self.prepare(client.GROUNDED_COMMIT, profile)
        identity = client.client_identity(self.root)
        self.assertEqual(identity['commit'], client.GROUNDED_COMMIT)
        with patch.object(client, '_run', side_effect=AssertionError('live identity must not execute tools')):
            self.assertEqual(client.client_identity(self.root), identity)
        (self.root / 'client/source/src/api/tokenEstimate.ts').write_bytes(b'changed estimator')
        with self.assertRaises(ValueError):
            client.client_identity(self.root)

    def test_template_history_binds_capacity_aware_client_and_distinct_fixture_contract(self):
        profile = 'antimatter-history-template-c2048'
        for options in ((None, None, None, None),
                        (self.repo, client.SUPPORTED_COMMIT, self.node, self.esbuild)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                client.validate_options(*options, profile)
        self.assertTrue(client.validate_options(self.repo, client.GROUNDED_COMMIT,
                                               self.node, self.esbuild, profile))
        self.sources['src/api/tokenEstimate.ts'] = b'export const estimate = 1;\n'
        self.prepare(client.GROUNDED_COMMIT, profile)
        identity = client.client_identity(self.root)
        self.assertEqual(identity['commit'], client.GROUNDED_COMMIT)
        self.assertEqual(set(common.read_json(self.root / 'client/manifest.json')['sources']),
                         set(client.GROUNDED_SOURCE_PATHS))
        with patch.object(client, '_run', side_effect=AssertionError('no live checkout or tools')):
            self.assertEqual(client.client_identity(self.root), identity)
        fixture_path = self.root / 'answer-fixtures.json'
        fixture_path.write_text(json.dumps(fixture_document('antimatter-history-c2048')))
        with self.assertRaises(ValueError):
            client.client_identity(self.root)
        fixture_path.write_text(json.dumps(fixture_document(profile)))
        (self.root / 'client/source/src/api/tokenEstimate.ts').write_bytes(b'changed estimator')
        with self.assertRaises(ValueError):
            client.client_identity(self.root)

    def test_new_git_closure_reads_only_pinned_six_regular_blobs(self):
        self.sources = {name: ('// committed ' + name).encode() for name in client.GROUNDED_SOURCE_PATHS}
        calls = []
        def git(arguments, cwd, limit):
            calls.append(arguments)
            if 'rev-parse' in arguments:
                return client.GROUNDED_COMMIT.encode()
            if 'ls-tree' in arguments:
                return ('100644 blob ' + 'a' * 40 + '\t' + arguments[-1]).encode()
            name = arguments[-1].split(':', 1)[1]
            return str(len(self.sources[name])).encode() if 'cat-file' in arguments else self.sources[name]
        with patch.object(client, '_run', side_effect=git):
            self.assertEqual(client.committed_sources(self.repo, client.GROUNDED_COMMIT), self.sources)
        self.assertEqual(sum('show' in call for call in calls), 6)
        self.assertTrue(all('HEAD' not in argument for call in calls for argument in call))

    def test_live_identity_never_reopens_source_repository_or_runs_tools(self):
        self.prepare()
        identity = client.client_identity(self.root)
        self.repo.rmdir()
        self.node.unlink()
        self.esbuild.unlink()
        with patch.object(client, '_run', side_effect=AssertionError('no live tools')):
            self.assertEqual(client.client_identity(self.root), identity)

    def test_each_frozen_source_runtime_bundle_and_metadata_is_bound(self):
        self.prepare()
        for name in ['node', 'provider.mjs', 'build-meta.json', 'manifest.json',
                     *('source/' + path for path in client.SOURCE_PATHS)]:
            path = self.root / 'client' / name
            original = path.read_bytes()
            path.write_bytes(original + b' changed')
            with self.subTest(name=name), self.assertRaises(ValueError):
                client.client_identity(self.root)
            path.write_bytes(original)

    def test_snapshot_rejects_symlinks_extra_files_and_nonexecutable_node(self):
        self.prepare()
        path = self.root / 'client/source' / client.ENTRY_POINT
        original = path.read_bytes()
        path.unlink()
        path.symlink_to(self.node)
        with self.assertRaises(ValueError):
            client.client_identity(self.root)
        path.unlink()
        path.write_bytes(original)
        extra = self.root / 'client/unfrozen-code.mjs'
        extra.write_text('export {}')
        with self.assertRaisesRegex(ValueError, 'unexpected file'):
            client.client_identity(self.root)
        extra.unlink()
        (self.root / 'client/node').chmod(0o600)
        with self.assertRaisesRegex(ValueError, 'not executable'):
            client.client_identity(self.root)

    def test_manifest_schema_commit_closure_and_tool_provenance_fail_closed(self):
        self.prepare()
        original = common.read_json(self.root / 'client/manifest.json')
        changes = [{'schema': True}, {'commit': 'f' * 40}, {'sources': {}}, {'files': {}},
                   {'bundle_inputs': []}, {'tools': {}}, {'entry_point': '../escape'}]
        for change in changes:
            self.reseal({**original, **change})
            with self.subTest(change=change), self.assertRaises(ValueError):
                client.client_identity(self.root)
        changed = copy.deepcopy(original)
        changed['tools']['node']['sha256'] = 'e' * 64
        self.reseal(changed)
        with self.assertRaisesRegex(ValueError, 'Node provenance'):
            client.client_identity(self.root)

    def test_optional_client_must_match_explicit_contract(self):
        self.prepare()
        contract = self.root / 'qualification-contract.json'
        original = common.read_json(contract)
        contract.write_text(json.dumps({**original, 'stream_client': 'python-http'}))
        with self.assertRaisesRegex(ValueError, 'differs from the qualification contract'):
            client.client_identity(self.root)
        contract.write_text(json.dumps(original))
        (self.root / 'expected-manifest.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'missing client snapshot'):
            client.client_identity(self.root)

    def test_python_default_needs_no_client_but_still_requires_explicit_contract(self):
        common.write_json(self.root / 'qualification-contract.json', inputs.profile_contract())
        common.write_json(self.root / 'answer-fixtures.json', fixture_document())
        common.write_json(self.root / 'expected-manifest.json', {})
        self.assertIsNone(client.client_identity(self.root))
        path = self.root / 'qualification-contract.json'
        value = common.read_json(path)
        del value['stream_client']
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            client.client_identity(self.root)

    def test_metafile_refuses_external_absolute_and_unknown_inputs_or_runtime_imports(self):
        original = self.metadata()
        for change in ('external', 'runtime_import', 'absolute', 'unknown', 'no_entry', 'no_export'):
            value = copy.deepcopy(original)
            if change == 'external':
                value['inputs']['source/' + client.ENTRY_POINT]['imports'] = [{'path': 'fs', 'external': True}]
            elif change == 'runtime_import':
                value['outputs']['provider.mjs']['imports'] = [{'path': 'node:fs', 'external': True}]
            elif change in ('absolute', 'unknown'):
                value['inputs']['/tmp/foreign.ts' if change == 'absolute' else 'source/foreign.ts'] = {'imports': []}
            elif change == 'no_entry':
                del value['inputs']['source/' + client.ENTRY_POINT]
            else:
                value['outputs']['provider.mjs']['exports'] = []
            with self.subTest(change=change), self.assertRaises(ValueError):
                client.validate_metafile(value)

    def test_git_reads_explicit_commit_regular_blobs_and_ignores_dirty_worktree(self):
        for name in client.SOURCE_PATHS:
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('dirty working tree must never be bundled')
        calls = []
        def git(arguments, cwd, limit):
            calls.append(arguments)
            if 'rev-parse' in arguments:
                return client.SUPPORTED_COMMIT.encode()
            if 'ls-tree' in arguments:
                return ('100644 blob ' + 'a' * 40 + '\t' + arguments[-1] + '\n').encode()
            name = arguments[-1].split(':', 1)[1]
            return str(len(self.sources[name])).encode() if 'cat-file' in arguments else self.sources[name]
        with patch.object(client, '_run', side_effect=git):
            self.assertEqual(client.committed_sources(self.repo, client.SUPPORTED_COMMIT), self.sources)
        self.assertTrue(all('HEAD' not in argument for call in calls for argument in call))
        self.assertEqual(sum('show' in call for call in calls), len(client.SOURCE_PATHS))
        for name in client.SOURCE_PATHS:
            self.assertEqual((self.repo / name).read_text(), 'dirty working tree must never be bundled')

    def test_git_refuses_symlink_and_oversized_blob_before_reading_content(self):
        for failure in ('symlink', 'size'):
            def git(arguments, cwd, limit):
                if 'rev-parse' in arguments:
                    return client.SUPPORTED_COMMIT.encode()
                if 'ls-tree' in arguments:
                    mode = '120000' if failure == 'symlink' else '100644'
                    return (mode + ' blob ' + 'a' * 40 + '\t' + arguments[-1]).encode()
                if 'cat-file' in arguments:
                    return str(client.MAX_SOURCE_BYTES + 1).encode()
                raise AssertionError('oversized or symlink source must not be read')
            with self.subTest(failure=failure), patch.object(client, '_run', side_effect=git):
                with self.assertRaises(ValueError):
                    client.committed_sources(self.repo, client.SUPPORTED_COMMIT)

    def test_unfrozen_tool_wrapper_and_repeated_preparation_refuse(self):
        self.node.write_bytes(b'#!/bin/sh\nexec /mutable/node "$@"\n')
        with patch.object(client, 'committed_sources', return_value=self.sources), \
             patch.object(client, '_run') as run:
            with self.assertRaisesRegex(ValueError, 'native ELF'):
                client.prepare_client(self.repo, client.SUPPORTED_COMMIT, self.node, self.esbuild, self.root)
            run.assert_not_called()
        with self.assertRaises(FileExistsError), patch.object(client, 'committed_sources', return_value=self.sources):
            client.prepare_client(self.repo, client.SUPPORTED_COMMIT, self.node, self.esbuild, self.root)

    def test_preparation_commands_timeout_and_nonzero_exit_fail_closed(self):
        with patch.object(client.subprocess, 'run', side_effect=subprocess.TimeoutExpired(['fixture'], 30)):
            with self.assertRaisesRegex(ValueError, 'exceeded 30 seconds'):
                client._run(['fixture'], self.root)
        with patch.object(client.subprocess, 'run', return_value=Mock(returncode=1)):
            with self.assertRaisesRegex(ValueError, 'command failed'):
                client._run(['fixture'], self.root)


if __name__ == '__main__':
    unittest.main()
