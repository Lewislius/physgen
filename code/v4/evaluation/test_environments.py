"""Environment routing and actual Python 3.10 -> 3.12 feature/gradient round trips."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from physgen_v4.environments import ENVIRONMENTS, python_for, require_environment, subprocess_environment
from physgen_v4.encoders import VideoTeacher
from physgen_v4.remote_teacher import RemoteVideoTeacher


class EnvironmentTests(unittest.TestCase):
    def test_runtime_guard_rejects_foreign_interpreter_and_module_path(self):
        require_environment('runtime')
        with patch.object(sys, 'prefix', '/tmp/unwanted-environment'):
            with self.assertRaisesRegex(RuntimeError, 'runtime requires'):
                require_environment('runtime')
        with patch.object(sys, 'path', [str(ENVIRONMENTS['teacher'] / 'lib/python3.12/site-packages')]):
            with self.assertRaisesRegex(RuntimeError, 'Foreign environment'):
                require_environment('runtime')

    def test_teacher_cannot_load_in_training_interpreter(self):
        with self.assertRaisesRegex(RuntimeError, 'teacher requires'):
            VideoTeacher({}, torch.device('cpu'))

    def test_subprocess_environment_discards_overrides(self):
        with patch.dict(os.environ, dict(PYTHON_EXEC='/unwanted/python', PYTHONPATH='/unwanted/modules',
                                        PYTHONHOME='/unwanted', VJEPA_ENV='/unwanted', VIRTUAL_ENV='/unwanted')):
            for role in ('runtime', 'teacher'):
                result = subprocess_environment(role)
                self.assertEqual(result['PYTHON_EXEC'], python_for(role))
                self.assertEqual(result['CONDA_PREFIX'], str(ENVIRONMENTS[role]))
                for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'VJEPA_ENV'):
                    self.assertNotIn(key, result)

    def test_shell_activation_restores_parent_and_pins_torchrun(self):
        script = '''
set -eo pipefail
V4_ROOT="$1"
source "${V4_ROOT}/tools/runtime_env.sh"
v4_activate runtime
( v4_activate teacher; test "$PYTHON_EXEC" = /home/liuzhirui/miniconda3/envs/vjepa2-312/bin/python )
test "$V4_PYTHON" = /home/liuzhirui/miniconda3/envs/moviestory/bin/python
test "$PYTHON_EXEC" = "$V4_PYTHON"
test -z "${PYTHONPATH:-}${PYTHONHOME:-}${VJEPA_ENV:-}${VIRTUAL_ENV:-}"
case "${PATH}:${LD_LIBRARY_PATH}" in *envs/unwanted/*) exit 1 ;; esac
'''
        environment = os.environ.copy()
        environment.update(PYTHON_EXEC='/unwanted/python', PYTHONPATH='/unwanted/modules',
                           PYTHONHOME='/unwanted', VJEPA_ENV='/unwanted', VIRTUAL_ENV='/unwanted')
        environment['PATH'] = '/home/liuzhirui/miniconda3/envs/unwanted/bin:' + environment['PATH']
        environment['LD_LIBRARY_PATH'] = '/home/liuzhirui/miniconda3/envs/unwanted/lib'
        result = subprocess.run(['bash', '-c', script, 'environment-test', str(ROOT)],
                                env=environment, text=True, capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_and_preparation_reject_wrong_roles(self):
        checks = [('teacher', 'train/train_native_p.py', ['--help'], 'runtime requires'),
                  ('runtime', 'tools/prepare_wisa.py', ['--config', '/unused', '--pass', 'teacher'], 'teacher requires')]
        for encoding_pass in ('index', 'vae', 'text', 'filter', 'finalize', 'audit'):
            checks.append(('teacher', 'tools/prepare_wisa.py',
                           ['--config', '/unused', '--pass', encoding_pass], 'runtime requires'))
        for role, script, args, error in checks:
            with self.subTest(script=script, args=args):
                result = subprocess.run([python_for(role), '-I', '-B', str(ROOT / script), *args],
                                        env=subprocess_environment(role), capture_output=True, text=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)

    def test_all_hardware_launchers_reach_shared_policy(self):
        for hardware in ('4x80g', '4x96g', '4x48g'):
            config = yaml.safe_load((ROOT / f'train/train_wisa_native_p_{hardware}.yaml').read_text())
            entry = Path(config['entrypoint'])
            self.assertTrue(entry.is_file())
            self.assertIn('run_training.sh', entry.read_text())
        for script in [*ROOT.glob('train/*.sh'), *ROOT.glob('tools/*.sh'), *ROOT.glob('inference/*.sh')]:
            subprocess.run(['bash', '-n', str(script)], check=True)


class TeacherBridgeTests(unittest.TestCase):
    def setUp(self):
        original_popen = subprocess.Popen
        def tiny_worker(command, **kwargs):
            command = list(command)
            self.assertEqual(command[0], python_for('teacher'))
            self.assertEqual(command[3], str(ROOT / 'tools/teacher_worker.py'))
            command[3] = str(ROOT / 'evaluation/teacher_test_worker.py')
            return original_popen(command, **kwargs)
        # Only replace the heavy model, retaining the production worker and transport.
        self.patcher = patch('physgen_v4.remote_teacher.subprocess.Popen', side_effect=tiny_worker)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.worker = RemoteVideoTeacher(dict(teacher_code=str(ROOT)), torch.device('cpu'), recompute=True)
        self.addCleanup(self.worker.close)

    @staticmethod
    def reference(video):
        return (torch.sin(video * 0.7) + 0.2 * video.square()).flatten(2).transpose(1, 2)

    def test_features_and_scaled_accumulated_input_gradients(self):
        self.assertEqual(self.worker.runtime['environment'], str(ENVIRONMENTS['teacher']))
        video = torch.linspace(-0.9, 0.9, 120, dtype=torch.float64).reshape(1, 3, 4, 2, 5).requires_grad_()
        reference = video.detach().clone().requires_grad_()
        # Two live graphs, an intervening no-grad validation call, and non-unit upstream gradients.
        first = self.worker(video, 4)
        second = self.worker(video * 0.6, 4)
        with torch.no_grad():
            checked = self.worker(video, 4)
            torch.testing.assert_close(checked, self.reference(video))
            self.assertFalse(checked.requires_grad)
        expected_first, expected_second = self.reference(reference), self.reference(reference * 0.6)
        torch.testing.assert_close(first, expected_first)
        torch.testing.assert_close(second, expected_second)
        (0.13 * first.square().mean() + 0.27 * second.sin().sum()).backward()
        (0.13 * expected_first.square().mean() + 0.27 * expected_second.sin().sum()).backward()
        torch.testing.assert_close(video.grad, reference.grad, rtol=1e-12, atol=1e-12)
        self.assertGreater(video.grad.abs().sum().item(), 0)

    def test_worker_error_is_propagated(self):
        with self.assertRaisesRegex(RuntimeError, 'frame_count must be positive'):
            self.worker(torch.zeros(1, 3, 4, 2, 2), 0)

    def test_worker_exit_is_propagated_and_close_reaps_process(self):
        process = self.worker.process
        process.terminate()
        process.wait(timeout=10)
        with self.assertRaisesRegex(RuntimeError, 'V-JEPA worker failed'):
            self.worker(torch.zeros(1, 3, 4, 2, 2), 4)
        self.worker.close()
        self.assertIsNotNone(process.returncode)


if __name__ == '__main__':
    require_environment('runtime')
    unittest.main(verbosity=2)
