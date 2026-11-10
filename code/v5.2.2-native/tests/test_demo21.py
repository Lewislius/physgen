"""Input isolation and final-checkpoint safeguards for the actual demo runner."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('message_demo21', ROOT / 'inference/infer_message_demo21.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)
from physgen_native.config import read_config


class Demo21Tests(unittest.TestCase):
    def test_staged_conditioning_preserves_cfg_sampling_and_actual_exit(self):
        sys.path.insert(0, str(ROOT / 'tests'))
        from test_simple_verifier import simple_model
        for query in (False, True):
            model, sample = simple_model(query=query)
            model.eval().requires_grad_(False)
            model.cfg['inference']['steps'] = 2
            sample['null_text'] = torch.zeros_like(sample['text'])
            with torch.no_grad():
                expected = model.sample(model.conditions(sample), sample['text'], sample['null_text'], seed=42)
                expected_analysis = model.last_sample_analysis
                expected_round = model.last_sample_reasoning_round
                actual = demo.sample_staged(model, sample, 42)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(model.last_sample_analysis, expected_analysis)
            self.assertEqual(model.last_sample_reasoning_round, expected_round)
            self.assertEqual(expected_round, 2 if query else 1)

    def test_checkpoint_seed_is_mandatory(self):
        cfg = {'train': {'seed': 42}}
        self.assertEqual(demo.training_seed(cfg, None), 42)
        self.assertEqual(demo.training_seed(cfg, 42), 42)
        with self.assertRaisesRegex(ValueError, 'training seed 42'):
            demo.training_seed(cfg, 43)

    def test_future_and_legacy_condition_tensors_are_never_returned(self):
        cfg = {'data': {'latent_shape': [48, 21, 2, 4]}}
        source = dict(first=torch.zeros(1, 48, 1, 2, 4), text=torch.ones(1, 3, 4096),
                      null_text=torch.ones(1, 1, 4096), times=torch.arange(21).float() / 6,
                      valid=torch.full((2, 4), 1.00000155),
                      latent=torch.full((1, 48, 21, 2, 4), float('nan')),
                      task=torch.full((1, 7, 4096), float('nan')),
                      jepa_times=torch.full((21, 3), float('nan')))
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            torch.save(source, folder / 'conditions.pt')
            actual = demo.load_inputs(folder, cfg, 24.)
            self.assertEqual(set(actual), set(demo.INPUT_KEYS))
            self.assertTrue(all(torch.isfinite(x).all() for x in actual.values()))
            with self.assertRaisesRegex(ValueError, 'timestamp/FPS'):
                demo.load_inputs(folder, cfg, 30.)
            source['first'] = torch.zeros(1, 48, 1, 4, 2)
            torch.save(source, folder / 'conditions.pt')
            with self.assertRaisesRegex(ValueError, 'incompatible first geometry'):
                demo.load_inputs(folder, cfg, 24.)

    def test_incomplete_or_wrong_variant_weights_are_rejected(self):
        cfg = read_config(ROOT / 'configs/native_h_content_message_verifier_frozen_simple_wisa3857.yaml')
        state = dict(revision=cfg['revision'], config=cfg, step=224, cursor=896, model={}, verifier_state={})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'latest.pt'
            torch.save(state, path)
            self.assertEqual(demo.final_checkpoint(path, 'h_content_message')['step'], 224)
            with self.assertRaisesRegex(ValueError, 'requires v522_native_q'):
                demo.final_checkpoint(path, 'q_seed128_message')
            changed = copy.deepcopy(state)
            changed['step'] = 208
            torch.save(changed, path)
            with self.assertRaisesRegex(ValueError, 'final 224-step'):
                demo.final_checkpoint(path, 'h_content_message')

    def test_case_selection_cannot_repeat_or_escape_output_directory(self):
        self.assertEqual(len(demo.select_ids('all')), 21)
        self.assertEqual(demo.select_ids('P20,reference'), ['P20', 'reference'])
        for invalid in ('P01,P01', '../other', '', 'P21'):
            with self.assertRaises(ValueError):
                demo.select_ids(invalid)

    def test_new_two_round_no_verifier_and_legacy_checkpoints_remain_distinct(self):
        for mode in ('h_content_message','q_seed128_message'):
            for legacy in (False,True):
                folder=ROOT/'configs/legacy_fixed_verifier_feedback' if legacy else ROOT/'configs'
                cfg=read_config(folder/f'native_{mode}_verifier_frozen_simple_wisa3857.yaml')
                state=dict(revision=cfg['revision'],config=cfg,step=224,cursor=896,model={})
                if legacy:state['verifier_state']={}
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'latest.pt';torch.save(state,path)
                    result=demo.final_checkpoint(path,mode)
                    self.assertEqual('verifier_state' in result,legacy)
                    self.assertEqual(result['config']['knowledge']['inference_reasoning_round'],
                                     1 if legacy and mode=='h_content_message' else 2)


if __name__ == '__main__':
    unittest.main()
