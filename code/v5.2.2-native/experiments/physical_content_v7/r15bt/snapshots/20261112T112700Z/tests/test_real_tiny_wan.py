"""Official tiny WanModel forward/backward and training-objective checks."""
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

from physical_content.common import load_config
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample
from physical_content.train import sample_video


def tiny_wan():
    name = '_r15bt_test_wan'
    package = types.ModuleType(name)
    package.__path__ = []
    attention = types.ModuleType(name + '.attention')

    def cpu_attention(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(),
            v.transpose(1, 2).float()).transpose(1, 2).to(v.dtype)

    attention.flash_attention = cpu_attention
    sys.modules[name] = package
    sys.modules[name + '.attention'] = attention
    spec = importlib.util.spec_from_file_location(
        name + '.model', '/home/liuzhirui/model/Wan2.2/wan/modules/model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    wan = module.WanModel(model_type='ti2v', patch_size=(1, 2, 2),
                          text_len=5, in_dim=2, out_dim=2, dim=16,
                          ffn_dim=32, freq_dim=8, text_dim=16,
                          num_heads=2, num_layers=2)
    with torch.no_grad():
        wan.head.head.weight.normal_(std=.1)
    return wan


def make_model():
    cfg = load_config(ROOT / 'configs/r15bt.yaml')
    cfg['model'].update(width=16, qwen_width=32, q_tokens=8, q_heads=4,
                        grounded_read_heads=2, condition_every=1, lora_rank=2,
                        rounds=2)
    cfg['train']['gradient_checkpointing'] = False
    interface = dict(mean=torch.zeros(32), scale=torch.ones(32),
                     weight=torch.randn(16, 32) * .05, bias=torch.zeros(16),
                     target_scale=torch.tensor(.1))
    return cfg, PhysicalModel(cfg, device='cpu', wan=tiny_wan(),
                              interface=interface)


def evidence(drop=False):
    return dict(
        words=torch.randn(6, 32),
        word_groups=torch.tensor([0, 0, 1, 1, 2, 2]),
        word_positions=torch.arange(6), word_strings=['a'] * 6,
        relations=['a', 'b', 'c'], caption='' if drop else 'x y', sample='s',
        scope='request_only',
        request_words=(torch.empty(0, 32) if drop else torch.randn(5, 32)),
        request_word_strings=([] if drop else ['x'] * 5),
        native_words=torch.randn(6, 16) * .1)


class FakeTeacher:
    def input(self, sample, drop=False):
        return evidence(drop=drop)


class RealTinyWan(unittest.TestCase):
    def test_t2v_forward_backward_and_no_first_frame(self):
        cfg, model = make_model()
        content, groups = model.enhancement(evidence())
        noisy = torch.randn(1, 2, 3, 2, 4)
        times = torch.tensor([[0., 1., 2.]])
        text = torch.randn(1, 4, 16)
        prediction, aux = model.velocity(
            noisy, None, text, torch.tensor(.6), content, times, groups,
            valid=torch.ones(2, 4))
        self.assertEqual(tuple(prediction.shape), (1, 2, 3, 2, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        self.assertEqual(float(aux), 0.0)
        prediction.square().mean().backward()
        self.assertGreater(float(model.fusions['0'].output.weight.grad.norm()), 0.)
        with self.assertRaises(ValueError):
            model.velocity(noisy, noisy[:, :, :1], text, torch.tensor(.6),
                           content, times, groups)

    def test_training_objective_and_lora_phase(self):
        cfg, model = make_model()
        clean = torch.randn(1, 2, 3, 2, 4)
        noise = torch.randn_like(clean)
        text = torch.randn(1, 4, 16)
        times = torch.tensor([[0., 1., 2.]])
        sample = dict(latent=clean, times=times, valid=torch.ones(2, 4))
        model.set_phase(0)
        metrics = backward_sample(model, evidence(), sample, text,
                                  torch.tensor(.6), noise, .03, 1)
        self.assertTrue(all(torch.isfinite(torch.tensor(value))
                            for value in metrics.values()))
        self.assertGreater(
            float(model.enhancement.project.weight.grad.norm()), 0.)
        self.assertGreater(
            float(model.fusions['0'].output.weight.grad.norm()), 0.)
        self.assertEqual(sum(p.grad is not None for name, p in model.wan.named_parameters()
                             if name.endswith(('lora_a', 'lora_b'))), 0)
        model.zero_grad(set_to_none=True)
        self.assertEqual(model.set_phase(128), 'joint')
        backward_sample(model, evidence(), sample, text, torch.tensor(.6),
                        noise, .03, 1)
        self.assertGreater(sum(p.grad is not None for name, p in model.wan.named_parameters()
                               if name.endswith(('lora_a', 'lora_b'))), 0)

    def test_sample_video_uses_null_argument(self):
        cfg, model = make_model()
        cfg['inference']['steps'] = 1
        cfg['data']['latent_shape'] = [1, 2, 3, 2, 4]
        model.cfg['inference']['steps'] = 1
        sample = dict(id='s', caption='x y', text=torch.randn(1, 4, 16),
                      times=torch.tensor([[0., 1., 2.]]))
        null = torch.randn(1, 1, 16)
        value = sample_video(model, FakeTeacher(), sample, null, seed=42)
        self.assertEqual(tuple(value.shape), (1, 2, 3, 2, 4))
        self.assertTrue(torch.isfinite(value).all())


if __name__ == '__main__':
    unittest.main()
