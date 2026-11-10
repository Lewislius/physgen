"""Qwen-only verifier integration; these CPU checks are not quality evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator import TextKnowledge
from test_native import FakeWan, FakeDecoder, sample, grad_size
from physgen_native.config import read_config, validate
from physgen_native.data import VideoDataset
from physgen_native.system import PhysGenNative


def small_model():
    cfg = read_config(ROOT/'configs/native_qwen_process_wisa3857.yaml')
    cfg['model'].update(width=32, qwen_width=32, attention_width=16,
                        attention_heads=4, interface_rank=4, condition_every=1)
    cfg['knowledge'].update(image_size=[32,64], video_size=[32,64], lora_last_layers=2,
                            latent_reasoning_layers=2, lora_rank=2, lora_alpha=2)
    cfg['stages'] = dict(calibration=1, interface=1, joint=2)
    cfg['wan_lora'] = dict(rank=2, alpha=2)
    cfg['train'].update(accumulation=1, warmup=1, text_dropout=0., requirements_dropout=0.,
                        native_text_only_dropout=0., branch_checkpointing=False)
    cfg['sequence'].update(size=[8,12], scales=[1,2], lags=[1,4,8,16])
    cfg['verifier'].update(size=[8,12], frames=7)
    cfg['completion'].update(steps=3, every=1)
    cfg['inference'].update(steps=6, shift=1.)
    cfg['data'].update(latent_shape=[4,6,4,6], frames=21)
    validate(cfg)
    # An accidental attempt to instantiate JEPA is an immediate test failure.
    with patch('physgen_native.process_encoder.VideoProcessEncoder', side_effect=AssertionError('JEPA loaded')):
        model = PhysGenNative(cfg, FakeWan(cfg), knowledge=TextKnowledge(), decoder=FakeDecoder(), device='cpu')
    return model, sample()


class QwenProcessTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(2610)

    def test_three_stages_keep_four_loss_limit_and_critic_only_calibration(self):
        model, data = small_model()
        self.assertIsNone(model.process_encoder)
        self.assertEqual([len(x) for x in model.cfg['loss'].values()], [1,4,3])
        losses, weights, metrics = model(data, torch.zeros_like(data['text']), 0)
        self.assertEqual(set(losses), {'rank'})
        model.backward_losses(losses, weights)
        self.assertGreater(grad_size(model.critic), 0)
        self.assertEqual(grad_size(model.conditioner)+grad_size(model.knowledge)+grad_size(model.generator), 0)
        self.assertEqual(float(metrics['qwen_verifier_active']), 1)

    def test_full_video_feedback_updates_C_and_knowledge_in_both_generation_phases(self):
        model, data = small_model()
        for step in (1,2):
            model.zero_grad(set_to_none=True)
            losses, weights, metrics = model.feedback_losses(data, step)
            model.backward_losses(losses, weights)
            self.assertGreater(grad_size(model.conditioner), 0)
            self.assertGreater(grad_size(model.knowledge), 0)
            self.assertEqual(grad_size(model.critic)+grad_size(model.generator), 0)
            self.assertTrue(metrics['generated_from_pure_noise'])
            self.assertGreater(float(metrics['C_gradient_rms']), 0)
            self.assertEqual(metrics['full_video_frames'], 21)

    def test_pure_noise_trajectory_does_not_read_reference_future(self):
        model, data = small_model();model.set_phase(1)
        c = model.encode_condition(model.conditions(data)).detach()
        noise = torch.randn_like(data['latent'])
        expected = model.completion(data, c, torch.tensor(1.), noise, differentiable=False)
        altered = dict(data, latent=torch.full_like(data['latent'], float('nan')))
        actual = model.completion(altered, c, torch.tensor(1.), noise, differentiable=False)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_jepa_cache_is_not_read_by_the_qwen_only_dataset(self):
        model, data = small_model()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);(root/'v52/requirements').mkdir(parents=True)
            (root/'v52/requirements/test.json').write_text(json.dumps(dict(input_scope='caption_and_first_image',clauses=[])))
            dataset = VideoDataset.__new__(VideoDataset)
            dataset.cfg=model.cfg;dataset.root=root;dataset.targets=True;dataset.scale=1.
            dataset.rows=[dict(id='test',caption='A process.',pts=list(range(21)),
                geometry=dict(frames=21,h=8,w=12,top=0,left=0,rh=8,rw=12))]
            def load(path):
                if '/base/vae/' in str(path): return dict(latent=data['latent'],first=data['first'])
                if '/base/text/' in str(path): return data['text'][0]
                if '/v52/change/' in str(path): return dict(intensity=data['intensity'][0],valid=data['change_valid'][0])
                raise AssertionError(f'Unexpected cached feature read: {path}')
            with patch('physgen_native.data.load_tensor',side_effect=load): result=dataset[0]
            self.assertNotIn('evidence_activity',result)
            torch.testing.assert_close(result['latent'],data['latent'])

    def test_online_qwen_feedback_preserves_one_sampling_trajectory(self):
        model, data = small_model();model.eval().requires_grad_(False)
        model.cfg['inference']['latent_feedback']=dict(enabled=True,metric='relative_rms',step_size=.1,sigmas=[.8,.5])
        calls=[]
        handle=model.generator.register_forward_pre_hook(lambda module,args:calls.append(float(args[3])))
        result=model.sample(model.conditions(data),data['text'],torch.zeros_like(data['text']),seed=7)
        handle.remove()
        self.assertEqual(len(calls),12)
        self.assertEqual(model.last_sample_feedback['complete_trajectories'],1)
        self.assertTrue(all(x['gradient_rms']>0 for x in model.last_sample_feedback['updates']))
        torch.testing.assert_close(result[:,:,:1],data['first'],atol=0,rtol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__ == '__main__': unittest.main()
