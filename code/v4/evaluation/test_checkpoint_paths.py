"""CPU checks for repeated launches, resumed output isolation and checkpoint discovery."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from physgen_v4.checkpoint_paths import name_training_run, latest_completed_checkpoint
from physgen_v4.runtime import make_scheduler, read_config, save_checkpoint
from train.train_native_p import configured_run


class CheckpointPathTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='v4-checkpoint-paths-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = 'wisa_native_p_A1_4x96g'

    def completed_run(self, name, completed_time, base=None):
        run = self.root / name
        final = run / 'checkpoint-final'
        final.mkdir(parents=True)
        for file in ('corrector.pt', 'training.pt', 'runtime.json'):
            (final / file).write_text('saved')
        config = dict(run_name=name)
        if base is not None:
            config['run_name_base'] = base
        (final / 'config.yaml').write_text(yaml.safe_dump(config))
        latest = run / 'latest.json'
        latest.write_text(json.dumps(dict(checkpoint=str(final))))
        os.utime(latest, (completed_time, completed_time))
        return final

    def test_repeat_launches_and_resumes_have_distinct_roots_without_stacking_suffixes(self):
        config = dict(run_name=self.base, paths=dict(checkpoint_root=str(self.root)),
                      train=dict(steps=500, save_every=250))
        first = copy.deepcopy(config)
        path1 = name_training_run(first, '20260912T150000Z-11111111')
        second = copy.deepcopy(config)
        path2 = name_training_run(second, '20260912T150000Z-22222222')
        resumed = copy.deepcopy(first)
        path3 = name_training_run(resumed, '20260912T160000Z-33333333', path1 / 'checkpoint-0000250')
        self.assertEqual(len({path1, path2, path3}), 3)
        self.assertEqual(path3.name, self.base + '_20260912T160000Z-33333333')
        self.assertEqual(resumed['source_run_name'], path1.name)
        self.assertEqual(resumed['resume_from'], str(path1 / 'checkpoint-0000250'))
        self.assertEqual(resumed['train'], config['train'])
        # Naming a diagnostic-only run must not create empty checkpoint directories.
        self.assertEqual(list(self.root.iterdir()), [])

    def test_old_and_new_resume_configs_preserve_schedule_with_new_destinations(self):
        for stamped in (False, True):
            with self.subTest(stamped=stamped):
                config = read_config(ROOT / 'configs/wisa_native_p.yaml')
                config['paths']['checkpoint_root'] = str(self.root)
                config['run_name'] = self.base
                config['train'].update(steps=321, learning_rate=.00003)
                if stamped:
                    name_training_run(config, '20260912T150000Z-11111111')
                source = self.root / ('new' if stamped else 'old')
                source.mkdir()
                (source / 'config.yaml').write_text(yaml.safe_dump(config))
                with patch.dict(os.environ, WORLD_SIZE='1'):
                    resumed = configured_run(argparse.Namespace(resume=str(source), wandb=None, steps=999))
                destination = name_training_run(resumed, '20260912T160000Z-22222222', source)
                self.assertEqual(destination.name, self.base + '_20260912T160000Z-22222222')
                self.assertEqual(resumed['train']['steps'], 321)
                self.assertEqual(resumed['train']['learning_rate'], .00003)
                self.assertEqual(resumed['train']['save_every'], 250)

    def test_discovery_ignores_unfinished_runs_and_matches_saved_base(self):
        legacy = self.completed_run(self.base, 100)
        self.assertEqual(latest_completed_checkpoint(self.root, self.base), legacy)
        first = self.completed_run(self.base + '_20260912T150000Z-11111111', 200, self.base)
        newest = self.completed_run(self.base + '_20260912T160000Z-22222222', 300, self.base)
        unfinished = self.completed_run(self.base + '_20260912T170000Z-33333333', 400, self.base)
        (unfinished.parent / 'latest.json').write_text(json.dumps(dict(checkpoint='checkpoint-0000250')))
        self.completed_run(self.base + '_unrelated', 500, self.base + '_unrelated')
        self.assertEqual(latest_completed_checkpoint(self.root, self.base), newest)
        (newest / 'training.pt').unlink()
        self.assertEqual(latest_completed_checkpoint(self.root, self.base), first)

    def test_resolver_cli_honors_config_checkpoint_root(self):
        final = self.completed_run(self.base + '_20260912T150000Z-11111111', 100, self.base)
        config = self.root / 'custom.yaml'
        config.write_text(yaml.safe_dump(dict(paths=dict(checkpoint_root=str(self.root)))))
        result = subprocess.run([sys.executable, '-I', '-B', str(ROOT / 'physgen_v4/checkpoint_paths.py'),
                                 '--config', str(config), '--run-name', self.base],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(final))

    def test_no_completed_run_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, 'INIT_FROM.*CHECKPOINT'):
            latest_completed_checkpoint(self.root, self.base)

    def test_same_checkpoint_save_is_rejected_without_changing_weights_or_metadata(self):
        config = read_config(ROOT / 'configs/wisa_native_p.yaml')
        config['paths'].update(checkpoint_root=str(self.root), cache_root=str(self.root))
        config['run_name'] = self.base
        name_training_run(config, '20260912T150000Z-11111111')
        (self.root / 'manifest.json').write_text('{}')
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
        scheduler = make_scheduler(optimizer, config)
        with patch('physgen_v4.runtime.dist.get_rank', return_value=0), \
                patch('physgen_v4.runtime.dist.get_world_size', return_value=1), \
                patch('physgen_v4.runtime.dist.is_initialized', return_value=False), \
                patch('physgen_v4.runtime.dist.barrier'), \
                patch('physgen_v4.runtime.rng_state', return_value={}), \
                patch('physgen_v4.runtime.ROOT', self.root), \
                patch('physgen_v4.runtime.checkpoint_source_files', return_value=[]):
            destination = save_checkpoint(config, model, optimizer, scheduler, 250, 2000, {})
            original = {file.name: file.read_bytes() for file in destination.iterdir() if file.is_file()}
            original_latest = (destination.parent / 'latest.json').read_bytes()
            with torch.no_grad():
                model.weight.add_(10)
            with self.assertRaisesRegex(RuntimeError, 'never overwritten'):
                save_checkpoint(config, model, optimizer, scheduler, 250, 2000, {})
            self.assertEqual(original, {file.name: file.read_bytes() for file in destination.iterdir() if file.is_file()})
            self.assertEqual(original_latest, (destination.parent / 'latest.json').read_bytes())
            final = save_checkpoint(config, model, optimizer, scheduler, 500, 4000, {}, final=True)
            self.assertEqual(latest_completed_checkpoint(self.root, self.base), final)

    def test_rank_zero_collision_is_reported_to_other_ranks_before_any_write(self):
        config = dict(run_name=self.base, paths=dict(checkpoint_root=str(self.root)))
        def receive_error(payload, src):
            payload[0] = 'existing saves are never overwritten'
        with patch('physgen_v4.runtime.dist.get_rank', return_value=1), \
                patch('physgen_v4.runtime.dist.is_initialized', return_value=True), \
                patch('physgen_v4.runtime.dist.broadcast_object_list', side_effect=receive_error), \
                patch('physgen_v4.runtime.torch.save') as save:
            with self.assertRaisesRegex(RuntimeError, 'never overwritten'):
                save_checkpoint(config, None, None, None, 250, 2000, {})
            save.assert_not_called()
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
