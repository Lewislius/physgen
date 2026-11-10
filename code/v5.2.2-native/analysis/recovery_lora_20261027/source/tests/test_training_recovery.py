"""A retry must restore optimizer/RNG state, never restart from initialization."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_resumable_training import training_command
from tools import run_resumable_training as recovery


class TrainingRecoveryTests(unittest.TestCase):
    def test_device_guard_checks_only_allocated_gpu_and_rejects_known_failure(self):
        import torch
        device = SimpleNamespace(uuid='failed-device', name='test GPU')
        with patch.object(torch.cuda, 'device_count', return_value=1), \
             patch.object(torch.cuda, 'get_device_properties', return_value=device) as properties:
            with self.assertRaisesRegex(RuntimeError, 'recorded SIGKILL'):
                recovery.check_allocated_device(['GPU-failed-device'])
            device.uuid = 'GPU-healthy-device'
            self.assertEqual(recovery.check_allocated_device(['GPU-failed-device'])['uuid'],
                             'GPU-healthy-device')
            self.assertTrue(all(call.args == (0,) for call in properties.call_args_list))
        with patch.object(torch.cuda, 'device_count', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                recovery.check_allocated_device(['GPU-failed-device'])
        with patch.object(torch.cuda, 'device_count', side_effect=AssertionError('unexpected CUDA access')):
            self.assertIsNone(recovery.check_allocated_device([]))

    def test_monitor_does_not_precreate_fresh_training_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'new_run'
            code=(f'from pathlib import Path; p=Path({str(directory)!r}); '
                  'assert not p.exists(); p.mkdir(); (p/"config.yaml").write_text("fresh")')
            command=[sys.executable,'-c',code]
            with patch.object(recovery,'training_command',return_value=(command,directory,False)), \
                 patch.object(sys,'argv',['recovery','--config',str(Path(tmp)/'recipe.yaml')]), \
                 patch.dict(os.environ,DET_EXPERIMENT_ID='123'):
                result=recovery.main()
            self.assertEqual(result,0)
            self.assertTrue((directory/'config.yaml').is_file())
            self.assertEqual(len(list((Path(tmp)/'recovery_logs'/'new_run').glob('*.resources.jsonl'))),1)

    def test_retry_uses_saved_config_and_drops_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'config.yaml'
            config.write_text(json.dumps(dict(run_name='test', paths=dict(runs=str(root/'runs')))))
            first,directory,restoring=training_command(config,'17',initialize_generator=root/'init.pt')
            self.assertFalse(restoring)
            self.assertIn('--initialize-generator',first)
            self.assertNotIn('--resume',first)
            # The same search ID is retained by Determined across restarts.
            directory.mkdir(parents=True)
            (directory/'latest.pt').write_bytes(b'atomic checkpoint fixture')
            (directory/'config.yaml').write_text(config.read_text())
            retry,actual,restoring=training_command(config,'17',initialize_generator=root/'init.pt')
            self.assertTrue(restoring)
            self.assertEqual(actual,directory)
            self.assertNotIn('--initialize-generator',retry)
            self.assertEqual(retry[retry.index('--resume')+1],str(directory))
            self.assertEqual(retry[retry.index('--config')+1],str(directory/'config.yaml'))
            self.assertEqual(retry[retry.index('--checkpoint-every')+1],'4')
            # A separate recovery experiment can keep the original run directory.
            explicit,actual,restoring=training_command(config,'18',resume=directory,
                                                       checkpoint_every=1, trace=True)
            self.assertTrue(restoring)
            self.assertEqual(actual,directory)
            self.assertIn('--trace', explicit)
            self.assertEqual(explicit[explicit.index('--checkpoint-every')+1], '1')

    def test_incomplete_first_save_restarts_fresh_but_explicit_resume_requires_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); config=root/'config.yaml'
            config.write_text(json.dumps(dict(run_name='test', paths=dict(runs=str(root)))))
            directory=root/'test_17'; directory.mkdir()
            (directory/'config.yaml').write_text(config.read_text())
            command,_,restoring=training_command(config,'17',initialize_generator=root/'init.pt')
            self.assertFalse(restoring)
            self.assertIn('--initialize-generator',command)
            with self.assertRaises(FileNotFoundError): training_command(config,'18',resume=directory)
            with self.assertRaises(ValueError): training_command(config,'18',checkpoint_every=0)


if __name__ == '__main__': unittest.main()
