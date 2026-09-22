"""CPU-only checks for allocation-driven launchers and batch configuration."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from physgen_v4.environments import allocated_gpu_count
from physgen_v4.runtime import configure_batch, read_config, resume_rng
from train import train_native_p as training
from train.train_native_p import configured_run


class AllocationTests(unittest.TestCase):
    def test_cluster_allocation_does_not_import_torch_or_cluster_sdk(self):
        for count in (1, 2, 3, 4, 8, 12):
            with self.subTest(count=count), \
                    patch.dict(os.environ, DET_SLOT_IDS=json.dumps(list(range(count))), CUDA_VISIBLE_DEVICES='0'), \
                    patch.dict(sys.modules, {'torch': None, 'determined': None}):
                self.assertEqual(allocated_gpu_count(), count)

    def test_direct_launch_follows_cuda_visibility(self):
        with patch.dict(os.environ, {}, clear=True), patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(allocated_gpu_count(), 2)

    def test_empty_allocation_does_not_fall_back_to_host_gpus(self):
        with patch.dict(os.environ, DET_SLOT_IDS='[]'), patch('torch.cuda.device_count') as count:
            with self.assertRaisesRegex(RuntimeError, 'No GPUs allocated'):
                allocated_gpu_count()
            count.assert_not_called()

    def test_all_phases_follow_world_size_for_every_pool(self):
        for label in ('4x48g', '4x80g', '4x96g'):
            for world, accumulation in ((1, 8), (2, 4), (3, 3), (4, 2), (8, 1), (12, 1)):
                for phase in ('A1', 'A2', 'A3', 'B1', 'AB'):
                    with self.subTest(label=label, world=world, phase=phase), \
                            patch.dict(os.environ, WORLD_SIZE=str(world)):
                        args = argparse.Namespace(config=str(ROOT / 'configs/wisa_native_p.yaml'),
                            resume=None, phase=phase, hardware=label, run_name=None, init_from=None,
                            steps=None, wandb=None)
                        config = configured_run(args)
                        self.assertEqual(config['corrector']['initialization'], 'image_text_gt')
                        self.assertEqual(config['corrector']['fusion'], 'cross_attention')
                        self.assertEqual(config['corrector']['a_every'], 5)
                        self.assertEqual(config['corrector']['parameter_sharing'], 'per_block')
                        self.assertEqual(config['corrector']['a_blocks'], [5, 10, 15, 20, 25, 30])
                        self.assertEqual(config['train']['steps'], 500 if phase == 'A1' else 250)
                        self.assertEqual(config['world_size'], world)
                        self.assertEqual(config['train']['accumulation_steps'], accumulation)
                        self.assertEqual(config['train']['effective_global_batch_size'], world * accumulation)
                        self.assertEqual(config['run_name'], f'wisa_native_p_{phase}_{label}')
                        self.assertNotIn('hardware', config)

    def test_resume_adjusts_batch_without_overriding_optimizer_schedule(self):
        config = read_config(ROOT / 'configs/wisa_native_p.yaml')
        config.update(world_size=4, hardware_name='4x48g')
        config['train'].update(steps=321, learning_rate=0.00003, accumulation_steps=2)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'config.yaml').write_text(yaml.safe_dump(config))
            with patch.dict(os.environ, WORLD_SIZE='2'):
                resumed = configured_run(argparse.Namespace(resume=directory, wandb=None, steps=999))
        self.assertEqual(resumed['train']['accumulation_steps'], 4)
        self.assertEqual(resumed['train']['effective_global_batch_size'], 8)
        self.assertEqual(resumed['train']['steps'], 321)
        self.assertEqual(resumed['train']['learning_rate'], 0.00003)
        self.assertEqual(resumed['run_name'], config['run_name'])

    def test_old_checkpoint_batch_target_is_preserved(self):
        legacy = dict(train=dict(accumulation_steps=3), hardware_name='4x48g',
                      hardware={'4x48g': dict(gpus=4, accumulation_steps=3)})
        configured = configure_batch(legacy, 2)
        self.assertEqual(configured['train']['global_batch_size'], 12)
        self.assertEqual(configured['train']['accumulation_steps'], 6)

    def test_same_world_size_restores_saved_rng(self):
        with patch('physgen_v4.runtime.torch.load', return_value={'saved': True}) as load, \
                patch('physgen_v4.runtime.restore_rng') as restore, patch('physgen_v4.runtime.seed_all') as seed:
            resume_rng('/checkpoint', 1, 4, 4, 20, 42)
        load.assert_called_once_with(Path('/checkpoint/rng_rank1.pt'), map_location='cpu', weights_only=False)
        restore.assert_called_once_with({'saved': True})
        seed.assert_not_called()

    def test_changed_world_size_reseeds_without_requiring_nonexistent_rank_files(self):
        for world, rank in ((2, 1), (8, 7)):
            with self.subTest(world=world), patch('physgen_v4.runtime.torch.load') as load, \
                    patch('physgen_v4.runtime.seed_all') as seed:
                resume_rng('/checkpoint', rank, world, 4, 20, 42)
            load.assert_not_called()
            seed.assert_called_once_with(42 + 100003 * 20 + rank)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ('train', 'tools', 'configs', 'physgen_v4', 'inference'):
            (self.root / directory).mkdir()
        for relative in ('train/run_training.sh', 'tools/prepare_wisa.sh', 'tools/runtime_env.sh',
                         'physgen_v4/environments.py', 'physgen_v4/__init__.py', 'configs/wisa_native_p.yaml',
                         'physgen_v4/checkpoint_paths.py', 'inference/infer_wisa_native_p.sh',
                         'inference/infer_wisa_native_p_1x48g.sh', 'inference/run_inference.sh'):
            shutil.copy2(ROOT / relative, self.root / relative)
        # Keep the production count reader; replace environment activation and GPU workloads only.
        with (self.root / 'tools/runtime_env.sh').open('a') as handle:
            handle.write('\nv4_activate() { export V4_PYTHON="${V4_ROOT}/record_python" TEST_ROLE="$1"; }\n')
        shim = self.root / 'record_python'
        shim.write_text(f'#!{sys.executable}\n' + '''import json
import os
import subprocess
import sys
args = sys.argv[1:]
if '-c' in args or any(arg.endswith('/checkpoint_paths.py') for arg in args):
    raise SystemExit(subprocess.call([sys.executable, *args]))
with open(os.environ['TEST_CALLS'], 'a') as handle:
    names = ('TMPDIR', 'V4_JOB_TMPDIR', 'V4_DIAGNOSTIC_DIR', 'NCCL_DEBUG', 'TORCH_NCCL_TRACE_BUFFER_SIZE',
             'TORCH_NCCL_DUMP_ON_TIMEOUT', 'TORCH_NCCL_ENABLE_MONITORING', 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC')
    handle.write(json.dumps({'args': args, 'role': os.environ['TEST_ROLE'],
                            'env': {key: os.environ[key] for key in names if key in os.environ}}) + '\\n')
''')
        shim.chmod(0o755)

    def test_only_yaml_slot_change_controls_training_and_all_preparation_passes(self):
        for label in ('4x48g', '4x80g', '4x96g'):
            submission_path = ROOT / f'train/train_wisa_native_p_{label}.yaml'
            original = yaml.safe_load(submission_path.read_text())
            # Pool and slot count are user-editable; the filename is only an entry label.
            pool = original['resources']['resource_pool']
            self.assertTrue(isinstance(pool, str) and pool.strip())
            entry = Path(original['entrypoint'])
            shutil.copy2(entry, self.root / 'train' / entry.name)
            for count in (1, 2, 3, 4, 8):
                for prepare in (False, True):
                    with self.subTest(pool=pool, count=count, prepare=prepare):
                        submission = copy.deepcopy(original)
                        submission['resources']['slots_per_trial'] = count
                        # Simulate only the platform's allocation from the changed YAML.
                        environment = {key: value for key, value in os.environ.items()
                                       if key not in ('RESUME', 'INIT_FROM', 'CONFIG', 'RUN_NAME', 'V4_JOB_TMPDIR',
                                                      'V4_DIAGNOSTIC_DIR') and not key.startswith('TRAIN_')}
                        calls_file = self.root / f'calls-{label}-{count}-{prepare}.jsonl'
                        environment.update(DET_SLOT_IDS=json.dumps(list(range(submission['resources']['slots_per_trial']))),
                                           PREPARE=str(int(prepare)), PHASE='A1', WANDB_ENABLED='0',
                                           V4_LOCAL_TMP_ROOT=str(self.root / 'local-scratch'),
                                           TEST_CALLS=str(calls_file))
                        result = subprocess.run(['bash', str(self.root / 'train' / entry.name)],
                                                env=environment, capture_output=True, text=True, timeout=30)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        calls = [json.loads(line) for line in calls_file.read_text().splitlines()]
                        workers = [call for call in calls if 'torch.distributed.run' in call['args']]
                        self.assertEqual(len(workers), 4 if prepare else 1)
                        for call in workers:
                            self.assertIn(f'--nproc_per_node={count}', call['args'])
                        training = workers[-1]['args']
                        directories = {call['env']['TMPDIR'] for call in calls}
                        self.assertEqual(len(directories), 1)
                        self.assertEqual(Path(directories.pop()).parent, self.root / 'local-scratch')
                        self.assertEqual(training[training.index('--hardware') + 1], label)
                        if prepare:
                            passes = [call['args'][call['args'].index('--pass') + 1]
                                      for call in calls if '--pass' in call['args']]
                            self.assertEqual(passes, ['index', 'vae', 'filter', 'text', 'teacher', 'finalize'])
                            self.assertEqual([call['role'] for call in workers],
                                             ['runtime', 'runtime', 'teacher', 'runtime'])
                        self.assertFalse(any('preflight' in argument for call in calls for argument in call['args']))
                        if not prepare:
                            self.assertEqual(len(calls), 1)

    def test_48g_submission_launches_complete_a1(self):
        self.check_formal_submission('4x48g')

    def test_resume_does_not_implicitly_add_previous_phase_init_from(self):
        for phase in ('A1', 'A2', 'A3', 'B1', 'AB'):
            calls_file = self.root / f'resume-{phase}.jsonl'
            environment = {key: value for key, value in os.environ.items()
                           if key not in ('INIT_FROM', 'CONFIG', 'V4_JOB_TMPDIR', 'V4_DIAGNOSTIC_DIR')}
            environment.update(PHASE=phase, RESUME='/saved/checkpoint-250', PREPARE='0',
                               DET_SLOT_IDS='[0]', TEST_CALLS=str(calls_file),
                               V4_LOCAL_TMP_ROOT=str(self.root / 'local-scratch'))
            result = subprocess.run(['bash', str(self.root / 'train/run_training.sh'), '4x96g'],
                                    env=environment, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            call, = [json.loads(line) for line in calls_file.read_text().splitlines()]
            self.assertIn('--resume', call['args'])
            self.assertNotIn('--init-from', call['args'])

    def test_stage_and_inference_launchers_resolve_timestamped_runs_and_preserve_explicit_paths(self):
        base = 'wisa_native_p_A1_4x96g'
        checkpoint = self.root / 'checkpoints' / (base + '_20260912T150000Z-12345678') / 'checkpoint-final'
        checkpoint.mkdir(parents=True)
        for name in ('corrector.pt', 'training.pt', 'runtime.json'):
            (checkpoint / name).write_text('saved')
        (checkpoint / 'config.yaml').write_text(yaml.safe_dump(dict(run_name_base=base)))
        (checkpoint.parent / 'latest.json').write_text(json.dumps(dict(checkpoint=str(checkpoint))))
        for inference in (False, True):
            for explicit in (False, True):
                with self.subTest(inference=inference, explicit=explicit):
                    calls_file = self.root / f'checkpoint-{inference}-{explicit}.jsonl'
                    environment = {key: value for key, value in os.environ.items()
                                   if key not in ('RESUME', 'INIT_FROM', 'CONFIG', 'CHECKPOINT', 'CHECKPOINT_ROOT',
                                                  'CHECKPOINT_RUN', 'RUN_NAME', 'V4_JOB_TMPDIR', 'V4_DIAGNOSTIC_DIR')}
                    environment.update(PHASE='A2', PREPARE='0', DET_SLOT_IDS='[0]', TEST_CALLS=str(calls_file),
                                       CHECKPOINT_RUN=base, V4_LOCAL_TMP_ROOT=str(self.root / 'local-scratch'))
                    expected = '/explicit/selected/checkpoint-0000250' if explicit else str(checkpoint)
                    if explicit:
                        environment['CHECKPOINT' if inference else 'INIT_FROM'] = expected
                    command = (['bash', str(self.root / 'inference/infer_wisa_native_p.sh')] if inference else
                               ['bash', str(self.root / 'train/run_training.sh'), '4x96g'])
                    result = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    call, = [json.loads(line) for line in calls_file.read_text().splitlines()]
                    option = '--checkpoint' if inference else '--init-from'
                    self.assertEqual(call['args'][call['args'].index(option) + 1], expected)

    def test_80g_submission_launches_complete_a1(self):
        self.check_formal_submission('4x80g')

    def test_96g_submission_launches_complete_single_gpu_a1(self):
        self.check_formal_submission('4x96g')

    def check_formal_submission(self, label):
        submission = yaml.safe_load((ROOT / f'train/train_wisa_native_p_{label}.yaml').read_text())
        count = submission['resources']['slots_per_trial']
        if label == '4x96g':
            self.assertEqual(count, 1)
        variables = dict(item.split('=', 1) for item in submission['environment']['environment_variables'])
        self.assertEqual(variables['PHASE'], 'A1')
        self.assertEqual(variables['TRAIN_CHECK_STEPS'], '0')
        entry = Path(submission['entrypoint'])
        shutil.copy2(entry, self.root / 'train' / entry.name)
        calls_file = self.root / 'formal-a1-calls.jsonl'
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith('TRAIN_') and key not in
                       ('RESUME', 'INIT_FROM', 'CONFIG', 'RUN_NAME', 'V4_JOB_TMPDIR', 'V4_DIAGNOSTIC_DIR')}
        environment.update(DET_SLOT_IDS=json.dumps(list(range(count))), TRAIN_CHECK_STEPS='4', TEST_CALLS=str(calls_file),
                           V4_LOCAL_TMP_ROOT=str(self.root / 'local-scratch'))
        environment.update(variables)
        result = subprocess.run(['bash', str(self.root / 'train' / entry.name)],
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call, = [json.loads(line) for line in calls_file.read_text().splitlines()]
        args = call['args']
        self.assertIn(f'--nproc_per_node={count}', args)
        main_path = str(self.root / 'train/train_native_p.py')
        self.assertIn(main_path, args)
        self.assertFalse(any('evaluation/' in value or 'test_' in value for value in args))
        # Parse exactly what the shell forwarded, using the real training parser/config.
        with patch.object(sys, 'argv', args[args.index(main_path):]), patch.object(training, 'train') as run:
            training.main()
        parsed = run.call_args.args[0]
        with patch.dict(os.environ, WORLD_SIZE=str(count)):
            config = configured_run(parsed)
        self.assertIsNone(parsed.resume)
        self.assertIsNone(parsed.init_from)
        self.assertIsNone(parsed.steps)
        self.assertEqual(config['diagnostics']['check_steps'], 0)
        self.assertEqual((config['phase'], config['stage']), ('A1', 'A'))
        self.assertEqual(config['corrector']['parameter_sharing'], 'per_block')
        self.assertEqual(config['corrector']['a_blocks'], [5, 10, 15, 20, 25, 30])
        self.assertEqual(config['corrector']['iterations'], 1)
        self.assertEqual(config['corrector']['initialization'], 'image_text_gt')
        self.assertEqual(config['train']['gt_condition_dropout'], .25)
        self.assertEqual(config['train']['state_warmup_every'], 0)
        self.assertEqual(config['train']['steps'], 500)
        self.assertEqual(config['train']['sample_limit'], 1000)
        self.assertEqual(config['train']['global_batch_size'], 8)
        self.assertEqual(config['train']['accumulation_steps'], (8 + count - 1) // count)
        self.assertEqual(config['train']['effective_global_batch_size'], count * ((8 + count - 1) // count))
        self.assertEqual(config['validation']['every'], 250)
        self.assertEqual(config['train']['save_every'], 250)
        self.assertGreater(config['loss']['fm_weight'], 0)
        self.assertGreater(config['loss']['struct_weight'], 0)
        self.assertGreater(config['loss']['prior_weight'], 0)
        self.assertEqual(config['loss']['out_weight'], 0)

    def test_diagnostic_overrides_forwarded_without_disabling_heartbeat(self):
        calls_file = self.root / 'diagnostic-calls.jsonl'
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(('TRAIN_', 'TORCH_NCCL_', 'NCCL_'))
                       and key not in ('V4_JOB_TMPDIR', 'V4_DIAGNOSTIC_DIR', 'RESUME', 'INIT_FROM', 'CONFIG', 'RUN_NAME')}
        environment.update(DET_SLOT_IDS='[0,1]', PREPARE='0', PHASE='A1', WANDB_ENABLED='0',
                           TEST_CALLS=str(calls_file), V4_LOCAL_TMP_ROOT=str(self.root / 'local-scratch'),
                           TRAIN_WORKERS='0', TRAIN_CHECK_STEPS='4', TRAIN_PIN_MEMORY='0',
                           TRAIN_TRACE_STEPS='4', TRAIN_TRACE_WAN_BLOCKS='1', TRAIN_NCCL_DEBUG='1',
                           TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC='480', TORCH_NCCL_ENABLE_MONITORING='1')
        result = subprocess.run(['bash', str(self.root / 'train/run_training.sh'), '4x48g'],
                                env=environment, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        call, = [json.loads(line) for line in calls_file.read_text().splitlines()]
        args = call['args']
        self.assertEqual(args[args.index('--workers') + 1], '0')
        self.assertEqual(args[args.index('--check-steps') + 1], '4')
        self.assertIn('--no-pin-memory', args)
        self.assertIn('--trace-wan-blocks', args)
        self.assertNotIn('--steps', args)
        self.assertEqual(call['env']['TORCH_NCCL_TRACE_BUFFER_SIZE'], '20000')
        self.assertEqual(call['env']['TORCH_NCCL_DUMP_ON_TIMEOUT'], '1')
        self.assertEqual(call['env']['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'], '480')
        self.assertEqual(call['env']['TORCH_NCCL_ENABLE_MONITORING'], '1')


if __name__ == '__main__':
    unittest.main(verbosity=2)
