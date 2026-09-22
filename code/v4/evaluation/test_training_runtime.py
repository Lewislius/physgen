"""Real spawn/cache checks plus CPU regressions for hang diagnostics and short checks."""
import argparse
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import multiprocessing
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
from torch.utils.checkpoint import checkpoint
import yaml

from physgen_v4.corrector import CellBlock
from physgen_v4.data import CachedWISA, training_loader
from physgen_v4.diagnostics import RankDiagnostics
from train import train_native_p as training


class ObservedCache(CachedWISA):
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample.update(reader_pid=os.getpid(), start_method=multiprocessing.get_start_method(),
                      cuda_initialized=torch.cuda.is_initialized())
        return sample


class CacheLoaderTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ('vae', 'text', 'teacher'):
            (self.root / directory).mkdir()
        records = [dict(index=i, split='train') for i in range(4)]
        (self.root / 'manifest.json').write_text(json.dumps(dict(records=records)))
        torch.save(torch.zeros(1, 4), self.root / 'null_text.pt')
        for i in range(4):
            name = f'{i:07d}.pt'
            torch.save(dict(latent=torch.full((1, 2, 2, 2, 2), float(i))), self.root / 'vae' / name)
            torch.save(torch.full((i + 1, 4), float(i)), self.root / 'text' / name)
            torch.save(torch.full((1, 8, 4), float(i)), self.root / 'teacher' / name)
        self.cfg = dict(seed=42, workers=2, pin_memory=False, loader_timeout_seconds=30)

    def test_spawn_reads_real_cache_in_order_without_cuda(self):
        dataset = ObservedCache(self.root, 'train')
        samples = list(training_loader(dataset, [3, 0, 2, 1], self.cfg, rank=0))
        self.assertEqual([s['record']['index'] for s in samples], [3, 0, 2, 1])
        self.assertTrue(all(s['reader_pid'] != os.getpid() for s in samples))
        self.assertTrue(all(s['start_method'] == 'spawn' and not s['cuda_initialized'] for s in samples))
        for sample in samples:
            index = sample['record']['index']
            self.assertEqual(sample['text'].shape, (index + 1, 4))
            torch.testing.assert_close(sample['latent'], dataset[index]['latent'])

    def test_workers_zero_reads_in_parent_with_timeout_normalized(self):
        self.cfg['workers'] = 0
        loader = training_loader(ObservedCache(self.root, 'train'), [1, 3], self.cfg, rank=1)
        self.assertEqual(loader.timeout, 0)
        self.assertIsNone(loader.multiprocessing_context)
        self.assertEqual([s['reader_pid'] for s in loader], [os.getpid(), os.getpid()])

    def test_worker_read_error_reaches_parent(self):
        (self.root / 'text/0000000.pt').unlink()
        self.cfg['workers'] = 1
        iterator = iter(training_loader(ObservedCache(self.root, 'train'), [0], self.cfg, rank=0))
        try:
            with self.assertRaises(FileNotFoundError):
                next(iterator)
        finally:
            iterator._shutdown_workers()


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = dict(paths=dict(log_root=str(self.root)), run_name='test',
                           diagnostics=dict(stack_timeout_seconds=0, trace_steps=1))
        env = patch.dict(os.environ, {'V4_DIAGNOSTIC_DIR': str(self.root), 'RANK': '1'})
        env.start()
        self.addCleanup(env.stop)

    def test_failed_stage_is_recorded_before_cleanup_without_cuda_calls(self):
        with redirect_stdout(io.StringIO()), patch('torch.cuda.synchronize', side_effect=AssertionError('GPU call')):
            with self.assertRaisesRegex(RuntimeError, 'stalled loader'):
                with RankDiagnostics(self.config) as trace:
                    with trace.stage('dataloader_iter'):
                        raise RuntimeError('stalled loader')
        records = [json.loads(line) for line in next(self.root.glob('events_rank1_*.jsonl')).read_text().splitlines()]
        self.assertEqual([r['event'] for r in records],
                         ['diagnostics_start', 'stage_begin', 'stage_error', 'diagnostics_end'])
        self.assertEqual(records[2]['stage'], 'dataloader_iter')
        self.assertEqual(records[-1]['status'], 'failed')
        self.assertTrue(trace.events.closed and trace.stacks.closed)

    def test_trace_limit_preserves_startup_and_forced_validation(self):
        with redirect_stdout(io.StringIO()), RankDiagnostics(self.config) as trace:
            with trace.iteration(step=11, offset=0):
                with trace.stage('first_step'):
                    pass
            with trace.iteration(step=12, offset=1):
                with trace.stage('untraced'):
                    pass
                with trace.stage('validation', always=True):
                    pass
        records = [json.loads(line) for line in next(self.root.glob('events_rank1_*.jsonl')).read_text().splitlines()]
        begins = [r for r in records if r['event'] == 'stage_begin']
        self.assertEqual([r['stage'] for r in begins], ['training_step', 'first_step', 'validation'])
        self.assertEqual(begins[-1]['step'], 12)

    def test_real_timer_dumps_python_stack_and_allows_completion(self):
        script = '''import sys, time
sys.path.insert(0, sys.argv[1])
from physgen_v4.diagnostics import RankDiagnostics
config = dict(paths=dict(log_root=sys.argv[2]), run_name='timer', diagnostics=dict(stack_timeout_seconds=.05))
with RankDiagnostics(config) as trace:
    with trace.stage('intentional_wait'):
        time.sleep(.16)
'''
        result = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(ROOT), str(self.root)],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        stacks = next(self.root.glob('stacks_rank1_*.txt')).read_text()
        self.assertIn('Timeout', stacks)
        self.assertIn('intentional_wait', stacks)
        self.assertIn('stage_end', result.stdout)


class FakeCorrector(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.prior_weight = torch.nn.Parameter(torch.ones(()))
        self.gt_weight = torch.nn.Parameter(torch.ones(()))

    def set_stage(self, stage):
        pass

    def parameter_groups(self):
        return dict(initialization=[self.prior_weight, self.gt_weight], shared=[self.weight])

    def gt_condition_parameters(self):
        return [self.gt_weight]

    def initialize(self, first, text, coords, gt_video=None):
        state = first.mean() * self.prior_weight + text.mean()
        return state if gt_video is None else state + gt_video.detach().mean() * self.gt_weight


class FakeWan(torch.nn.Module):
    def __init__(self, wan, corrector, enable_b):
        super().__init__()
        self.corrector = corrector
        self.enable_b = enable_b

    def forward(self, noisy, first, text, sigma, coords, process=None, gt_video=None):
        prior = self.corrector.initialize(first, text, coords, gt_video=gt_video) if process is None else None
        state = prior if process is None else process
        return noisy * self.corrector.weight + state, state, dict(prior=prior)


class CheckpointedCorrector(FakeCorrector):
    """Real attention/checkpoint graph inside the small training-loop fixture."""
    def __init__(self, config):
        super().__init__(config)
        self.prior_cell = CellBlock(64, 4)

    def parameter_groups(self):
        groups = super().parameter_groups()
        groups['initialization'].extend(self.prior_cell.parameters())
        return groups

    def initialize(self, first, text, coords, gt_video=None):
        queries = first.flatten(1).unsqueeze(-1).expand(-1, -1, 64)
        context = text.mean(-1, keepdim=True).expand(-1, -1, 64)
        if torch.is_grad_enabled():
            state = checkpoint(self.prior_cell, queries, context, use_reentrant=False)
        else:
            state = self.prior_cell(queries, context)
        state = state.float().mean() * self.prior_weight
        return state if gt_video is None else state + gt_video.detach().mean() * self.gt_weight


class ShortRunTests(unittest.TestCase):
    def test_legacy_resume_accepts_loader_overrides_without_changing_schedule(self):
        cfg = training.read_config(ROOT / 'configs/wisa_native_p.yaml')
        cfg.pop('diagnostics')
        cfg['train'].pop('loader_timeout_seconds')
        cfg['train'].pop('pin_memory')
        cfg['train'].update(steps=321, learning_rate=.00003)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.yaml').write_text(yaml.safe_dump(cfg))
            with patch.dict(os.environ, WORLD_SIZE='2'):
                config = training.configured_run(argparse.Namespace(resume=directory, wandb=None, steps=999,
                    workers=0, pin_memory=False, check_steps=4, trace_steps=4))
        self.assertEqual(config['train']['steps'], 321)
        self.assertEqual(config['train']['learning_rate'], .00003)
        self.assertEqual(config['train']['accumulation_steps'], 4)
        self.assertEqual(config['train']['workers'], 0)
        self.assertFalse(config['train']['pin_memory'])
        self.assertEqual(config['diagnostics']['check_steps'], 4)

    def test_short_run_executes_backward_and_update_but_skips_validation_and_save(self):
        self.check_training_run(prior_weight=0)

    def test_short_run_keeps_initializer_active_on_warm_steps_with_prior_supervision(self):
        self.check_training_run(prior_weight=.02, warm_every=3)

    def test_bf16_checkpointed_warm_prior_backward_and_optimizer_updates(self):
        self.check_training_run(prior_weight=.02, checkpointed=True, warm_every=3)

    def test_optional_warm_without_prior_supervision_skips_initializer_update(self):
        self.check_training_run(prior_weight=0, warm_every=3)

    def test_no_gt_steps_skip_only_gt_parameters(self):
        self.check_training_run(prior_weight=.02, gt_dropout=1)

    def test_gt_steps_update_gt_parameters(self):
        self.check_training_run(prior_weight=.02, gt_dropout=0)

    def test_bf16_checkpointed_ordinary_steps_without_prior_supervision(self):
        self.check_training_run(prior_weight=0, checkpointed=True)

    def test_formal_a1_runs_all_500_updates_with_validation_and_final_checkpoint(self):
        self.check_training_run(prior_weight=.02, check_steps=0)

    def check_training_run(self, prior_weight, checkpointed=False, check_steps=4, warm_every=0, gt_dropout=.25):
        cfg = training.read_config(ROOT / 'configs/wisa_native_p.yaml')
        cfg['train'].update(accumulation_steps=1, global_batch_size=1, workers=0, pin_memory=False,
                            state_warmup_every=warm_every, gt_condition_dropout=gt_dropout)
        cfg['diagnostics'].update(check_steps=check_steps, trace_steps=1, stack_timeout_seconds=0)
        cfg['loss']['prior_weight'] = prior_weight
        # Fixtures run only on CPU; the production YAML never imports this module.
        view = dict(frames=5, height=32, width=32, source_fps=4., duration=1., teacher_frames=4)
        sample = dict(latent=torch.ones(1, 2, 2, 2, 2), first=torch.ones(1, 2, 1, 2, 2),
                      text=torch.ones(2, 4), times=torch.arange(5).float(), target=torch.ones(1, 8, 4),
                      target_weight=torch.ones(1, 8), video_metadata=view,
                      record=dict(index=0, caption_scope='video'))
        class Dataset:
            records = [dict(view=view)]
            null_text = torch.zeros(2, 4)
            def __len__(self):
                return 1
            def __getitem__(self, index):
                return copy.deepcopy(sample)
        dataset = Dataset()
        fake_model = (CheckpointedCorrector if checkpointed else FakeCorrector)({})
        initializer_spy = patch.object(fake_model, 'initialize', wraps=fake_model.initialize)
        initial_parameters = {name: value.detach().clone() for name, value in fake_model.named_parameters()}
        actual_autocast = torch.autocast
        def cpu_autocast(device_type, *args, **kwargs):
            # Keep the production AMP/cache policy; only substitute the test device.
            return actual_autocast('cpu' if device_type == 'cuda' else device_type, *args, **kwargs)
        def loss(prediction, *args, **kwargs):
            value = prediction.square().mean()
            info = args[5]
            if prior_weight and info.get('prior') is not None:
                value = value + prior_weight * info['prior'].square().mean()
            return value, {'loss/fm': value.detach()}
        def synchronize(groups):
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                                for values in groups.values() for p in values))
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            initialized = stack.enter_context(initializer_spy)
            cfg['paths']['log_root'] = directory
            stack.enter_context(patch.dict(os.environ, V4_DIAGNOSTIC_DIR=directory, RANK='0'))
            stack.enter_context(redirect_stdout(io.StringIO()))
            replacements = dict(configured_run=lambda args: cfg, add_external_paths=lambda config: None,
                init_distributed=lambda: (0, 1, torch.device('cpu')), Corrector=lambda config: fake_model,
                load_wan=lambda *args: None, ProcessWan=FakeWan, CachedWISA=lambda *args, **kwargs: dataset,
                geometry=lambda *args: None, training_loss=loss, synchronize_gradients=synchronize,
                reduce_metrics=lambda values, device: values, memory_metrics=lambda device: {})
            for name, value in replacements.items():
                stack.enter_context(patch.object(training, name, value))
            validate = stack.enter_context(patch.object(training, 'validate', return_value={'validation_loss': 0.}))
            save = stack.enter_context(patch.object(training, 'save_checkpoint'))
            for name in ('synchronize', 'reset_peak_memory_stats'):
                stack.enter_context(patch.object(torch.cuda, name))
            for name in ('memory_allocated', 'memory_reserved', 'max_memory_allocated', 'max_memory_reserved'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            stack.enter_context(patch.object(torch.cuda, 'get_device_properties',
                return_value=argparse.Namespace(name='mock CPU execution', total_memory=0)))
            stack.enter_context(patch.object(torch.cuda.nccl, 'version', return_value=(2, 27, 3)))
            stack.enter_context(patch.object(torch, 'autocast', side_effect=cpu_autocast))
            stack.enter_context(patch.object(training.dist, 'broadcast_object_list'))
            stack.enter_context(patch.object(training.dist, 'destroy_process_group'))
            training.train(argparse.Namespace(resume=None, init_from=None))
            if check_steps:
                validate.assert_not_called()
                save.assert_not_called()
            else:
                # Loss schedules receive zero-based step; logs/saves use step + 1.
                self.assertEqual([call.args[3] for call in validate.call_args_list], [249, 499])
                self.assertEqual([call.args[4] for call in save.call_args_list], [250, 500, 500])
                self.assertTrue(save.call_args.kwargs['final'])
            self.assertNotEqual(fake_model.weight.item(), 1.)
            self.assertEqual(initialized.call_count, check_steps or 500)
            if gt_dropout == 1:
                self.assertEqual(fake_model.gt_weight.item(), 1.)
            if gt_dropout == 0:
                self.assertNotEqual(fake_model.gt_weight.item(), 1.)
            if checkpointed:
                for name in ('prior_cell.self_attention.value.weight', 'prior_cell.mlp.2.weight'):
                    self.assertFalse(torch.equal(dict(fake_model.named_parameters())[name], initial_parameters[name]), name)
            records = [json.loads(line) for line in next(Path(directory).rglob('steps.jsonl')).read_text().splitlines()]
            steps = [record for record in records if record['split'] == 'train']
            self.assertEqual([record['step'] for record in steps], list(range(1, (check_steps or 500) + 1)))
            if not check_steps:
                self.assertEqual([record['step'] for record in records if record['split'] == 'validation'], [250, 500])
            self.assertEqual(cfg['train']['steps'], 500)
            self.assertAlmostEqual(steps[0]['metrics']['learning_rate'], .000001)
            self.assertEqual(steps[0]['metrics']['grad/initialization_active'], float(not warm_every or prior_weight > 0))
            self.assertEqual(steps[1]['metrics']['grad/initialization_active'], 1.)
            for i, record in enumerate(steps):
                self.assertEqual(record['metrics']['prior/gt_conditioned'], float(training.use_gt_condition(cfg, i)))
                self.assertEqual(record['metrics']['compute/wan_forwards'], 1 + int(bool(warm_every and i % warm_every == 0)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
