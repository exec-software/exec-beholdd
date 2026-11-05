"""Public dispatch checks; no checkpoint, server, device or systemd execution."""

import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from hag_qualify import __main__ as cli
from hag_qualify.antimatter import SUPPORTED_COMMIT, GROUNDED_COMMIT


class CommandLineTests(unittest.TestCase):
    def test_phase_diagnostics_is_an_explicit_prepare_only_flag_for_template_history(self):
        paths = {name: Path('/unused') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        options = {'antimatter_repo': Path('/client'), 'antimatter_commit': GROUNDED_COMMIT,
                   'node': Path('/node'), 'esbuild': Path('/esbuild')}
        arguments = ['prepare', '--profile', 'antimatter-history-template-c2048', '--runtime-phase-diagnostics']
        for name, value in {**paths, **options}.items():
            arguments.extend(['--' + name.replace('_', '-'), str(value)])
        output = io.StringIO()
        with patch('hag_qualify.inputs.prepare') as prepare, patch.object(cli.subprocess, 'run') as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(arguments), 0)
        prepare.assert_called_once_with(*paths.values(), profile='antimatter-history-template-c2048',
                                        **options, runtime_phase_diagnostics=True)
        run.assert_not_called()
        self.assertIn('HAG answers only; raw logs are not qualifying evidence', output.getvalue())
        arguments[arguments.index('antimatter-history-template-c2048')] = 'antimatter-history-c2048'
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments), 1)
        prepare.assert_not_called()

    def test_generated_prepare_requires_and_forwards_explicit_metadata_inspector(self):
        profile = 'antimatter-generated-history-c2048'
        paths = {name: Path('/unused') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        options = {'antimatter_repo': Path('/client'), 'antimatter_commit': GROUNDED_COMMIT,
                   'node': Path('/node'), 'esbuild': Path('/esbuild')}
        arguments = ['prepare', '--profile', profile]
        for name, value in {**paths, **options}.items():
            arguments.extend(['--' + name.replace('_', '-'), str(value)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments), 1)
        prepare.assert_not_called()
        output = io.StringIO()
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(arguments + ['--prompt-inspector', '/metadata inspector']), 0)
        prepare.assert_called_once_with(*paths.values(), profile=profile, **options,
                                        prompt_inspector=Path('/metadata inspector'))
        self.assertIn('two genuine chat requests per backend, no SSE or timing', output.getvalue())
        arguments[arguments.index(profile)] = 'antimatter-project-c2048'
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments + ['--prompt-inspector', '/metadata inspector']), 1)
        prepare.assert_not_called()

    def test_generated_timing_refuses_before_any_frozen_process(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('hag_qualify.gate_common.gate_identity'), \
                 patch('hag_qualify.gate_common.dependency_identity'), \
                 patch('hag_qualify.gate_common.provenance_identity'), \
                 patch('hag_qualify.antimatter.client_identity'), \
                 patch('hag_qualify.gate_common.approved_contract', return_value={'profile': 'antimatter-generated-history-c2048'}), \
                 patch.object(cli.subprocess, 'run') as run:
                with self.assertRaises(ValueError):
                    cli.run_frozen(Path(directory), 'run', 'timing')
            run.assert_not_called()

    def test_help_does_not_launch_anything(self):
        with patch.object(cli.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as exit:
                cli.main(["--help"])
        self.assertEqual(exit.exception.code, 0)
        run.assert_not_called()

    def test_prepare_requires_explicit_inputs(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["prepare", "--evidence", "/tmp/unused"])

    def test_prepare_defaults_to_list_and_forwards_explicit_profiles(self):
        paths = {name: Path('/unused with spaces') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        arguments = ['prepare']
        for name, path in paths.items():
            arguments.extend(['--' + name, str(path)])
        for profile, extra in (('list-c128', []), ('list-c128', ['--profile', 'list-c128']),
                               ('semantic-c128', ['--profile', 'semantic-c128']),
                               ('backend-parity-c128', ['--profile', 'backend-parity-c128'])):
            with self.subTest(profile=profile):
                with patch('hag_qualify.inputs.prepare') as prepare, \
                     patch.object(cli.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(cli.main(arguments + extra), 0)
                prepare.assert_called_once_with(*paths.values(), profile=profile)
                run.assert_not_called()

    def test_unknown_profile_is_rejected_before_preparation(self):
        arguments = ['prepare', '--repo', '/unused', '--binary', '/unused', '--model', '/unused',
                     '--device', '/unused', '--evidence', '/unused', '--profile', 'custom']
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(arguments)
        prepare.assert_not_called()

    def test_optional_committed_client_arguments_forward_as_one_explicit_set(self):
        paths = {name: Path('/unused with spaces') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        options = {'antimatter_repo': Path('/client repo'), 'antimatter_commit': SUPPORTED_COMMIT,
                   'node': Path('/node binary'), 'esbuild': Path('/esbuild binary')}
        arguments = ['prepare', '--profile', 'backend-parity-c128']
        for name, value in {**paths, **options}.items():
            arguments.extend(['--' + name.replace('_', '-'), str(value)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(arguments), 0)
        prepare.assert_called_once_with(*paths.values(), profile='backend-parity-c128', **options)
        for missing in options:
            changed = list(arguments)
            index = changed.index('--' + missing.replace('_', '-'))
            del changed[index:index + 2]
            with self.subTest(missing=missing), patch('hag_qualify.inputs.prepare') as prepare, \
                 contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(changed), 1)
                prepare.assert_not_called()

    def test_only_declared_stages_are_accepted(self):
        with patch.object(cli, "run_frozen") as run, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["run", "--evidence", "/tmp/unused", "--stage", "all"])
        run.assert_not_called()

    def test_grounded_profile_requires_and_forwards_its_pinned_capacity_aware_client(self):
        paths = {name: Path('/unused') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        base = ['prepare', '--profile', 'antimatter-grounded-c512']
        for name, path in paths.items():
            base.extend(['--' + name, str(path)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(base), 1)
            prepare.assert_not_called()
        options = {'antimatter_repo': Path('/client repo'), 'antimatter_commit': GROUNDED_COMMIT,
                   'node': Path('/node'), 'esbuild': Path('/esbuild')}
        arguments = list(base)
        for name, value in options.items():
            arguments.extend(['--' + name.replace('_', '-'), str(value)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main(arguments), 0)
        prepare.assert_called_once_with(*paths.values(), profile='antimatter-grounded-c512', **options)
        arguments[arguments.index(GROUNDED_COMMIT)] = SUPPORTED_COMMIT
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments), 1)
            prepare.assert_not_called()

    def test_project_profile_requires_pinned_client_and_announces_finite_new_scope(self):
        profile = 'antimatter-project-c2048'
        paths = {name:Path('/unused space')/name for name in ('repo','binary','model','device','evidence')}
        arguments = ['prepare','--profile',profile]
        for name,value in paths.items(): arguments.extend(['--'+name,str(value)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments),1)
            prepare.assert_not_called()
        options = {'antimatter_repo':Path('/client'),'antimatter_commit':GROUNDED_COMMIT,
                   'node':Path('/node'),'esbuild':Path('/esbuild')}
        for name,value in options.items(): arguments.extend(['--'+name.replace('_','-'),str(value)])
        output = io.StringIO()
        with patch('hag_qualify.inputs.prepare') as prepare, patch.object(cli.subprocess,'run') as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(arguments),0)
        prepare.assert_called_once_with(*paths.values(),profile=profile,**options)
        run.assert_not_called()
        self.assertIn('1500s watcher only after CPU retirement',output.getvalue())
        self.assertIn('CPU HTTP deadline: 600s',output.getvalue())

    def test_path_with_spaces_is_one_argument_and_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory(prefix="hag CLI ") as directory:
            root = Path(directory)
            with patch("hag_qualify.gate_common.gate_identity"), \
                 patch("hag_qualify.gate_common.dependency_identity"), \
                 patch("hag_qualify.gate_common.provenance_identity"), \
                 patch("hag_qualify.gate_common.approved_contract", return_value={'profile': 'list-c128'}), \
                 patch("hag_qualify.antimatter.client_identity"), \
                 patch.object(cli.subprocess, "run", return_value=Mock(returncode=1)) as run:
                code = cli.run_frozen(root, "run", "answers")
            self.assertEqual(code, 1)
            run.assert_called_once_with(
                [cli.sys.executable, str(root / "run-gate.py"), "answers"], check=False
            )

    def test_history_prepare_requires_pinned_client_and_announces_seeded_scope(self):
        profile = 'antimatter-history-c2048'
        paths = {name: Path('/unused space') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        arguments = ['prepare', '--profile', profile]
        for name, value in paths.items():
            arguments.extend(['--' + name, str(value)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments), 1)
            prepare.assert_not_called()
        options = {'antimatter_repo': Path('/client'), 'antimatter_commit': GROUNDED_COMMIT,
                   'node': Path('/node'), 'esbuild': Path('/esbuild')}
        for name, value in options.items():
            arguments.extend(['--' + name.replace('_', '-'), str(value)])
        output = io.StringIO()
        with patch('hag_qualify.inputs.prepare') as prepare, patch.object(cli.subprocess, 'run') as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(arguments), 0)
        prepare.assert_called_once_with(*paths.values(), profile=profile, **options)
        run.assert_not_called()
        self.assertIn('Seeded first-turn history, not a live two-turn conversation', output.getvalue())
        self.assertIn('495 tokens, reply allowance 128', output.getvalue())
        self.assertIn('CPU HTTP deadline: 600s', output.getvalue())
        self.assertIn('Native provider client deadline: 240s', output.getvalue())
        self.assertIn('1500s watcher only after CPU retirement', output.getvalue())

    def test_lifecycle_prepare_needs_no_client_and_announces_only_three_stages(self):
        paths = {name: Path('/unused space') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        arguments = ['prepare', '--profile', 'lifecycle-c2048']
        for name, value in paths.items():
            arguments.extend(['--' + name, str(value)])
        output = io.StringIO()
        with patch('hag_qualify.inputs.prepare') as prepare, patch.object(cli.subprocess, 'run') as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(arguments), 0)
        prepare.assert_called_once_with(*paths.values(), profile='lifecycle-c2048')
        run.assert_not_called()
        self.assertIn('1500s watcher only after CPU retirement', output.getvalue())
        self.assertIn('CPU HTTP deadline: 600s', output.getvalue())
        self.assertIn('no timing stage', output.getvalue())
        self.assertNotIn('Native provider client deadline', output.getvalue())

    def test_template_history_prepare_announces_corrected_rendering_without_execution(self):
        profile = 'antimatter-history-template-c2048'
        paths = {name: Path('/unused space') / name for name in ('repo', 'binary', 'model', 'device', 'evidence')}
        options = {'antimatter_repo': Path('/client'), 'antimatter_commit': GROUNDED_COMMIT,
                   'node': Path('/node'), 'esbuild': Path('/esbuild')}
        arguments = ['prepare', '--profile', profile]
        for name, value in paths.items(): arguments.extend(['--' + name, str(value)])
        with patch('hag_qualify.inputs.prepare') as prepare, contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(arguments), 1)
            prepare.assert_not_called()
        for name, value in options.items(): arguments.extend(['--' + name.replace('_', '-'), str(value)])
        output = io.StringIO()
        with patch('hag_qualify.inputs.prepare') as prepare, patch.object(cli.subprocess, 'run') as run, \
             contextlib.redirect_stdout(output):
            self.assertEqual(cli.main(arguments), 0)
        prepare.assert_called_once_with(*paths.values(), profile=profile, **options)
        run.assert_not_called()
        self.assertIn('Checkpoint-rendered seeded history, not a live two-turn conversation', output.getvalue())
        self.assertIn('491 tokens, reply allowance 128', output.getvalue())
        self.assertNotIn('495 tokens', output.getvalue())
        self.assertIn('CPU HTTP deadline: 600s', output.getvalue())
        self.assertIn('Native provider client deadline: 240s', output.getvalue())
        self.assertIn('1500s watcher only after CPU retirement', output.getvalue())

    def test_lifecycle_dispatch_refuses_timing_before_subprocess(self):
        with tempfile.TemporaryDirectory(prefix='hag lifecycle CLI ') as directory:
            root = Path(directory)
            with patch('hag_qualify.gate_common.gate_identity'), \
                 patch('hag_qualify.gate_common.dependency_identity'), \
                 patch('hag_qualify.gate_common.provenance_identity'), \
                 patch('hag_qualify.gate_common.approved_contract', return_value={'profile': 'lifecycle-c2048'}), \
                 patch('hag_qualify.antimatter.client_identity'), \
                 patch.object(cli.subprocess, 'run', return_value=Mock(returncode=0)) as run:
                with self.assertRaises(ValueError):
                    cli.run_frozen(root, 'run', 'timing')
                run.assert_not_called()
                self.assertEqual(cli.run_frozen(root, 'run', 'answers'), 0)
                run.assert_called_once_with([cli.sys.executable, str(root / 'run-gate.py'), 'answers'], check=False)

    def test_changed_client_snapshot_refuses_before_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('hag_qualify.gate_common.gate_identity'), \
                 patch('hag_qualify.gate_common.dependency_identity'), \
                 patch('hag_qualify.gate_common.provenance_identity'), \
                 patch('hag_qualify.antimatter.client_identity', side_effect=ValueError('client changed')), \
                 patch.object(cli.subprocess, 'run') as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(['run', '--evidence', directory, '--stage', 'answers']), 1)
            run.assert_not_called()

    def test_changed_snapshot_refuses_before_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("hag_qualify.gate_common.gate_identity", side_effect=ValueError("changed")), \
                 patch.object(cli.subprocess, "run") as run, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(["run", "--evidence", directory, "--stage", "answers"]), 1)
            run.assert_not_called()

    def test_missing_or_symlink_evidence_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            link = root / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaises(ValueError):
                cli.evidence_directory(link)
            with self.assertRaises(FileNotFoundError):
                cli.evidence_directory(root / "missing")


if __name__ == "__main__":
    unittest.main()
