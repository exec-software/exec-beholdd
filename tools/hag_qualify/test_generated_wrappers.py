"""Hardware-free generated-history supervision and retirement regressions."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import generated_history
import inputs
from profiles import GENERATED_HISTORY_PROFILE as PROFILE, profile_modes


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('generated_runner_test', 'run-gate.py')
child = load('generated_child_test', 'gate-child.py')
closeout = load('generated_closeout_test', 'closeout.py')


class GeneratedWrappers(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.contract = inputs.profile_contract(PROFILE, 'antimatter')
        self.client = {'kind': 'antimatter', 'commit': 'fixture'}

    def test_only_three_stages_dispatch_and_replay_before_acceptance(self):
        for mode in ('cpu-reference', 'answers'):
            self.assertEqual(child.probe_command(mode, self.root, PROFILE),
                             ['/usr/bin/python3', str(self.root / 'generated-history-gate.py'),
                              '--stage', mode, '--execute'])
            report = {'client_identity': self.client}
            with patch.object(runner, 'approved_contract', return_value=self.contract), \
                 patch.object(runner, 'client_identity', return_value=self.client), \
                 patch.object(generated_history, 'validate_stage', return_value=report) as replay:
                self.assertEqual(runner.probe_evidence(mode, 'retained raw log', self.root), report)
                replay.assert_called_once_with(self.root, mode, 'retained raw log')
            with patch.object(runner, 'approved_contract', return_value=self.contract), \
                 patch.object(runner, 'client_identity', return_value=self.client), \
                 patch.object(generated_history, 'validate_stage', side_effect=ValueError('bad parent')):
                with self.assertRaisesRegex(ValueError, 'bad parent'):
                    runner.probe_evidence(mode, '', self.root)
        with self.assertRaises(ValueError): child.probe_command('timing', self.root, PROFILE)
        with patch.object(runner, 'approved_contract', return_value=self.contract), \
             patch.object(runner, 'client_identity', return_value=self.client):
            with self.assertRaises(ValueError): runner.probe_evidence('timing', '', self.root)
            for marker in ('HAG_GENERATED_HISTORY_REPORT {}', 'PASS hag-generated-history'):
                with self.assertRaises(ValueError): runner.probe_evidence('self-check', marker, self.root)
        with patch.object(runner, 'approved_contract', return_value=self.contract), \
             patch.object(runner, 'client_identity', return_value=None):
            with self.assertRaises(ValueError): runner.probe_evidence('answers', '', self.root)

    def test_closeout_includes_raw_lineage_and_no_timing(self):
        with patch.object(closeout, 'approved_contract', return_value=self.contract):
            names = closeout.closed_evidence_names(self.root)
        for mode in profile_modes(PROFILE):
            self.assertTrue(set(generated_history.evidence_names(mode)) <= set(names))
        self.assertFalse(any(name.startswith('timing') for name in names))
        self.assertEqual(len(names), len(set(names)))
        for qualified in (True, False):
            scope = closeout.qualification_scope(self.contract, qualified)
            self.assertIs(scope['generated_history_qualified'], qualified)
            for key in ('semantic_quality_qualified', 'live_cancellation_qualified',
                        'timing_executed', 'performance_qualified'):
                self.assertFalse(scope[key])

    def test_cleanup_requires_two_servers_and_both_client_processes_retired(self):
        proc, groups = self.root / 'proc', self.root / 'cgroups'
        proc.mkdir(); groups.mkdir()
        properties = {'ActiveState': 'inactive', 'SubState': 'dead', 'MainPID': '0', 'ControlGroup': ''}
        reports = {mode: {'unit': f'hag-answers-{mode}-20261011-120000-{index:06}',
                         'child': {'pid': 100 + index, 'cgroup': str(groups / mode)},
                         'thermal': {'summary': {'pid': 200 + index}}}
                   for index, mode in enumerate(profile_modes(PROFILE))}
        reports['answers']['client_identity'] = self.client
        artifacts = generated_history.client_artifacts()
        self.assertEqual(len(artifacts), 2)
        for index, (stage, prefix, artifact) in enumerate(artifacts):
            self.assertEqual(stage, 'answers')
            pid = 401 + index
            (self.root / artifact).write_text(json.dumps({'client': {'pid': pid}}))
            (self.root / (prefix + '-client-process.json')).write_text(json.dumps(
                {'pid': pid, 'exit_code': 0, 'timed_out': False, 'identity': self.client}))

        def processes(root, mode):
            return [{'pid': 301 if mode == 'cpu-reference' else 302, 'exit_code': 0}]

        def cleanup():
            return closeout.cleanup_evidence(reports, proc, groups, lambda _: properties, self.root, PROFILE)

        with patch.object(generated_history, 'server_processes', side_effect=processes), \
             patch.object(closeout, 'approved_contract', return_value=self.contract):
            result = cleanup()
            self.assertEqual(len(result['recorded_pids_absent']), 10)
            self.assertEqual(result['stages']['answers']['client_pids_absent'], [401, 402])
            for pid in (301, 302, 401, 402):
                (proc / str(pid)).mkdir()
                with self.assertRaises(ValueError): cleanup()
                (proc / str(pid)).rmdir()
            del reports['answers']['client_identity']
            with self.assertRaises(ValueError): cleanup()


if __name__ == '__main__':
    unittest.main()
