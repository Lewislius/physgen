"""Hardware rejection, exact shared-noise sampling and safe retry classification."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
import torch
from inference import infer_message_demo21 as runner
from inference import submit_no_verifier_benchmarks as submitter
from inference.no_verifier_jobs import JOBS
from physgen_native.config import read_config
from test_no_verifier import no_verifier_model


class NoVerifierInferenceTests(unittest.TestCase):
    def test_h100_is_rejected_before_opening_checkpoint_or_inputs(self):
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.get_device_name', return_value='NVIDIA H100'), \
             patch.object(runner, 'final_checkpoint', side_effect=AssertionError('checkpoint opened')) as load:
            with self.assertRaisesRegex(RuntimeError, 'GPU policy rejected NVIDIA H100'):
                runner.main(['--checkpoint', '/missing.pt', '--mode', 'h_content_message',
                             '--no-verifier-only', '--rtx-pro6000-only'])
            load.assert_not_called()

    def test_no_verifier_launcher_rejects_historical_fixed_judge(self):
        for mode in ('h_content_message', 'q_seed128_message'):
            cfg = read_config(ROOT / 'configs/legacy_fixed_verifier_feedback' /
                              f'native_{mode}_verifier_frozen_simple_wisa3857.yaml')
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'legacy.pt'
                torch.save(dict(config=cfg, revision=cfg['revision'], step=224, model={}, cursor=896), path)
                with self.assertRaisesRegex(ValueError, 'no-verifier checkpoint'):
                    runner.final_checkpoint(path, mode, no_verifier_only=True)

    def test_shared_noise_is_preserved_by_resident_and_staged_no_verifier_sampling(self):
        for query in (False, True):
            model, sample = no_verifier_model(query)
            model.eval().requires_grad_(False)
            model.cfg['inference']['steps'] = 2
            sample['null_text'] = torch.zeros_like(sample['text'])
            shape = list(sample['first'].shape)
            shape[2] = len(sample['times'])
            noise = torch.randn(shape, generator=torch.Generator().manual_seed(12345))
            unchanged = noise.clone()
            with torch.no_grad():
                expected = model.sample(model.conditions(sample), sample['text'], sample['null_text'],
                                        seed=42, initial_noise=noise)
                # A different seed must have no effect when exact noise is supplied.
                actual = runner.sample_staged(model, sample, 987, initial_noise=noise)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(noise, unchanged, rtol=0, atol=0)
            self.assertEqual(model.last_sample_reasoning_round, 2)
            self.assertIsNone(model.last_sample_feedback)
            self.assertIsNone(model.critic)

    def test_noise_geometry_and_subset_are_checked_before_sampling(self):
        cfg = {'data': {'latent_shape': [48, 21, 2, 4]}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'noise.pt'
            noise = torch.randn(1, 48, 21, 2, 4)
            torch.save(noise, path)
            case = dict(id='VP001', initial_noise=str(path))
            torch.testing.assert_close(runner.load_initial_noise(case, cfg), noise)
            torch.save(noise[:, :, :1], path)
            with self.assertRaisesRegex(ValueError, 'incompatible saved'):
                runner.load_initial_noise(case, cfg)
            (root / 'manifest.json').write_text('{}')
            ids = root / 'ids.json'
            ids.write_text(json.dumps(['VP001'] * 40))
            with self.assertRaisesRegex(ValueError, 'fixed VP001-VP040'):
                runner.collect_videophy_cases(root, cfg, ids, 24)

    def test_only_hardware_rejections_are_retried(self):
        for state, accepted, expected in (('ERROR', False, 'waiting_rtx'),
                                           ('ERROR', True, 'failed'),
                                           ('CANCELED', False, 'failed')):
            job = copy.deepcopy(JOBS['h_demo21'])
            job['attempts'] = [dict(experiment_id=1, gpu_policy=dict(accepted=accepted))]
            response = json.dumps([dict(experiment=dict(state='STATE_' + state, trialIds=[2], endTime='done'))])
            with patch.object(submitter, 'det', return_value=response), \
                 patch.object(submitter, 'read_summary', return_value=None):
                submitter.refresh(job)
            self.assertEqual(job['status'], expected)


if __name__ == '__main__':
    unittest.main()
