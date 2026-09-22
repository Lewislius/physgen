"""Local logging and W&B contracts. W&B network calls are always mocked here."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from physgen_v4.monitoring import (Logger, TRAIN_LOG_ROOT, checkpoint_source_files,
                                  normalize_logging, read_wandb_key)
from physgen_v4.runtime import read_config
from train.train_native_p import configured_run


class MonitoringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = dict(paths=dict(log_root=str(self.root)), run_name='test', train=dict(steps=10),
                           logging=dict(session_id='session-1',
                                        wandb=dict(enabled=False, project='test-project', entity=None)))
        self.metrics = {'loss/total': torch.tensor(0.5), 'loss/fm': 0.4, 'loss/struct': 0.8,
                        'loss/out': 0, 'learning_rate': 0.0001, 'grad/global_norm_before_clip': 0.9,
                        'A/state_gate_mean': 0.3, 'sigma': 0.25, 'time/step_seconds': 1.5}

    def test_disabled_wandb_never_imports_or_reads_credentials(self):
        with patch('physgen_v4.monitoring.read_wandb_key') as key, \
                patch('physgen_v4.monitoring.importlib.import_module') as importer, \
                redirect_stdout(io.StringIO()) as console:
            with Logger(self.config, 0) as logger:
                logger.start({'python': sys.executable})
                for step in (1, 2):
                    logger.micro(step, 0, 9, self.metrics, {'latent_shape': [1, 48, 2, 4, 4]})
                    logger.step(step, self.metrics)
                logger.step(2, {'validation_loss': 0.2}, validation=True)
            key.assert_not_called()
            importer.assert_not_called()
        rows = [json.loads(line) for line in (logger.root / 'steps.jsonl').read_text().splitlines()]
        self.assertEqual([row['split'] for row in rows], ['train', 'train', 'validation'])
        self.assertEqual(rows[1]['metrics']['A/state_gate_mean'], 0.3)
        latest = json.loads((logger.root / 'latest.json').read_text())
        self.assertEqual(latest['train']['step'], 2)
        self.assertEqual(latest['validation']['step'], 2)
        self.assertIn('[train] step=1/10 loss/total=0.5', console.getvalue())
        self.assertIn('[train] step=2/10', console.getvalue())
        for name in ('steps.txt', 'micro_rank0.jsonl', 'micro_rank0.txt', 'runtime.json', 'config.json'):
            self.assertTrue((logger.root / name).is_file(), name)
        self.assertIn('A/state_gate_mean=0.3', (logger.root / 'steps.txt').read_text())

    def test_nonchief_saves_micro_metrics_without_online_reporting(self):
        self.config['logging']['wandb']['enabled'] = True
        with patch('physgen_v4.monitoring.read_wandb_key') as key, \
                patch('physgen_v4.monitoring.importlib.import_module') as importer, redirect_stdout(io.StringIO()) as console:
            with Logger(self.config, 1) as logger:
                logger.micro(1, 0, 10, self.metrics)
                logger.step(1, self.metrics)
            key.assert_not_called()
            importer.assert_not_called()
        self.assertEqual(console.getvalue(), '')
        self.assertTrue((logger.root / 'micro_rank1.txt').is_file())
        self.assertFalse((logger.root / 'steps.jsonl').exists())

    def test_wandb_key_import_upload_and_same_step_validation(self):
        credentials = self.root / 'credentials.py'
        credentials.write_text('WANDB_API_KEY = "test-only-secret"\n')
        self.config['logging']['wandb']['enabled'] = True
        self.config['logging']['wandb']['api_key'] = 'should-be-redacted'
        sdk = Mock()
        sdk.login.return_value = True
        with patch('physgen_v4.monitoring.CREDENTIALS_FILE', credentials), \
                patch('physgen_v4.monitoring.importlib.import_module', return_value=sdk), redirect_stdout(io.StringIO()):
            with Logger(self.config, 0) as logger:
                logger.step(3, self.metrics)
                logger.step(3, {'validation_loss': 0.2}, validation=True)
        sdk.login.assert_called_once_with(key='test-only-secret', relogin=True)
        arguments = sdk.init.call_args.kwargs
        self.assertFalse(arguments['save_code'])
        self.assertEqual(arguments['config']['logging']['wandb']['api_key'], '[REDACTED]')
        self.assertEqual(arguments['dir'], str(logger.root))
        logs = sdk.init.return_value.log.call_args_list
        self.assertEqual([call.args[0]['global_step'] for call in logs], [3, 3])
        self.assertIn('train/loss/total', logs[0].args[0])
        self.assertIn('validation/validation_loss', logs[1].args[0])
        sdk.init.return_value.finish.assert_called_once_with(exit_code=0)
        for path in logger.root.iterdir():
            if path.is_file():
                self.assertNotIn('test-only-secret', path.read_text())
                self.assertNotIn('should-be-redacted', path.read_text())

    def test_wandb_missing_key_fails_before_login(self):
        credentials = self.root / 'credentials.py'
        credentials.write_text('WANDB_API_KEY = ""\n')
        with patch('physgen_v4.monitoring.CREDENTIALS_FILE', credentials):
            with self.assertRaisesRegex(RuntimeError, 'WANDB_API_KEY is empty'):
                read_wandb_key()

    def test_invalid_credentials_do_not_echo_key_in_traceback(self):
        credentials = self.root / 'credentials.py'
        credentials.write_text('WANDB_API_KEY = "test-secret-with-missing-quote\n')
        with patch('physgen_v4.monitoring.CREDENTIALS_FILE', credentials):
            try:
                read_wandb_key()
            except RuntimeError:
                self.assertNotIn('test-secret-with-missing-quote', traceback.format_exc())
            else:
                self.fail('Malformed credentials must fail without exposing the source line')

    def test_failure_closes_logs_and_finishes_wandb(self):
        self.config['logging']['wandb']['enabled'] = True
        sdk = Mock()
        with patch('physgen_v4.monitoring.read_wandb_key', return_value='test-only'), \
                patch('physgen_v4.monitoring.importlib.import_module', return_value=sdk), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'training failed'):
                with Logger(self.config, 0) as logger:
                    logger.step(1, self.metrics)
                    raise RuntimeError('training failed')
        sdk.init.return_value.finish.assert_called_once_with(exit_code=1)
        self.assertTrue(logger.steps.closed)
        self.assertTrue(logger.micro_json.closed)

    def test_nonfinite_metrics_remain_readable_json(self):
        with redirect_stdout(io.StringIO()), Logger(self.config, 0) as logger:
            logger.step(1, {'loss/total': float('nan'), 'gradient': float('inf')})
        values = json.loads((logger.root / 'steps.jsonl').read_text())['metrics']
        self.assertEqual(values, {'loss/total': 'nan', 'gradient': 'inf'})

    def test_checkpoint_source_excludes_credentials_and_all_log_contents(self):
        train = self.root / 'train'
        logs = train / 'train_log/run/session'
        logs.mkdir(parents=True)
        (train / 'wandb_credentials.py').write_text('WANDB_API_KEY="test-only-secret"')
        (train / 'train_native_p.py').write_text('# source')
        (logs / 'config.yaml').write_text('not: source')
        (logs / 'unexpected.py').write_text('# not source')
        paths = [path.relative_to(self.root) for path in checkpoint_source_files(self.root)]
        self.assertEqual(paths, [Path('train/train_native_p.py')])

    def test_old_checkpoint_uses_new_log_root_and_monitoring_defaults(self):
        old = dict(paths=dict(log_root='/old/logs'), logging=dict(tensorboard=True, determined=True))
        result = normalize_logging(old)
        self.assertEqual(result['paths']['log_root'], str(TRAIN_LOG_ROOT))
        self.assertNotIn('tensorboard', result['logging'])
        self.assertNotIn('determined', result['logging'])
        self.assertFalse(result['logging']['wandb']['enabled'])

    def test_resume_allows_only_monitoring_overrides(self):
        config = read_config(ROOT / 'configs/wisa_native_p.yaml')
        config['logging']['wandb']['enabled'] = True
        (self.root / 'config.yaml').write_text(yaml.safe_dump(config))
        args = argparse.Namespace(resume=str(self.root), wandb=False, steps=999)
        result = configured_run(args)
        self.assertEqual(result['train']['steps'], config['train']['steps'])
        self.assertFalse(result['logging']['wandb']['enabled'])
        self.assertEqual(result['paths']['log_root'], str(TRAIN_LOG_ROOT))

    def test_training_imports_and_resumed_logging_without_removed_sdks(self):
        # A fresh interpreter catches indirect imports, including attempts hidden by fallbacks.
        script = '''
import argparse
import importlib.abc
import json
from pathlib import Path
import sys
import tempfile

blocked = ('determined', 'tensorflow', 'tensorboard', 'torch.utils.tensorboard', 'wandb')
class RejectRemovedSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('Unexpected SDK import: ' + fullname)
sys.meta_path.insert(0, RejectRemovedSDKs())
sys.path.insert(0, sys.argv[1])
import yaml
from physgen_v4.monitoring import Logger
from physgen_v4.runtime import read_config
from train.train_native_p import configured_run

config = read_config(Path(sys.argv[1]) / 'configs/wisa_native_p.yaml')
config['logging'].update(tensorboard=True, determined=True)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'config.yaml').write_text(yaml.safe_dump(config))
    config = configured_run(argparse.Namespace(resume=directory, wandb=False))
    assert set(config['logging']) == {'wandb'}
    config['paths']['log_root'] = directory
    with Logger(config, 0) as logger:
        logger.start({'resume_from': directory})
        logger.micro(1, 0, 9, {'loss/total': 0.5})
        logger.step(1, {'loss/total': 0.5})
        logger.step(1, {'validation_loss': 0.2}, validation=True)
    rows = [json.loads(line) for line in (logger.root / 'steps.jsonl').read_text().splitlines()]
    assert [row['split'] for row in rows] == ['train', 'validation']
    assert not list(root.rglob('*tfevents*'))
assert not any(name in sys.modules for name in blocked)
'''
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', REPORT_DETERMINED='1')
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT)],
                                env=environment, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('[train] step=1/', result.stdout)
        self.assertIn('[validation] step=1/', result.stdout)

    def test_training_failure_releases_process_group(self):
        from train.train_native_p import train
        with patch('train.train_native_p.configured_run', return_value=self.config), \
                patch('train.train_native_p.add_external_paths'), \
                patch('train.train_native_p.init_distributed', return_value=(0, 4, torch.device('cpu'))), \
                patch('train.train_native_p.dist.broadcast_object_list', side_effect=RuntimeError('startup failed')), \
                patch('train.train_native_p.dist.destroy_process_group') as destroy:
            with self.assertRaisesRegex(RuntimeError, 'startup failed'):
                train(argparse.Namespace())
        destroy.assert_called_once_with()

    def test_three_submission_configs_disable_wandb(self):
        for hardware in ('4x80g', '4x96g', '4x48g'):
            config = yaml.safe_load((ROOT / f'train/train_wisa_native_p_{hardware}.yaml').read_text())
            self.assertIn('WANDB_ENABLED=0', config['environment']['environment_variables'])
            self.assertNotIn('profiling', config)
            self.assertFalse(any(value.startswith('REPORT_DETERMINED=')
                                 for value in config['environment']['environment_variables']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
