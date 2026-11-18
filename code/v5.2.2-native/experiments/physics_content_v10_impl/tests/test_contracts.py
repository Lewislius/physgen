"""CPU contract checks for the NoChunks-v2 implementation."""
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

from physical_content.common import load_config
from physical_content.mechanisms import Mechanism1, Mechanism2
from physical_content.readers import DirectPhysicalReader, JointPhysicalReader
from physical_content.schedule import lr_multiplier, route_coefficient
from physical_content.supervision import generation_loss


class RMS(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class FakeAttention(nn.Module):
    def __init__(self, width=8, heads=2):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = width, heads, width // heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm_q, self.norm_k = RMS(width), RMS(width)


class FakeBank:
    def __init__(self, attention):
        self._attention = attention

    def attention(self, layer, name):
        return self._attention


def cfg(width=8, ffn_width=16, mechanism=1):
    return dict(
        plan='I1' if mechanism == 1 else 'I2', task='i2v', mechanism=mechanism,
        model=dict(width=width, ffn_width=ffn_width, conditioned_ffn_rank=None
                   if mechanism == 1 else 64, injection_layers=[0]),
        train=dict(peak_lrs={'wan_video_lora': 2e-5, 'enhancement': 5e-5,
                             'physics_read': 2e-5, 'physics_write': 3e-5,
                             'mechanism_adapter': 2e-5},
                   max_updates=400, video_lora_warmup=51, physics_warmup=13,
                   peak_end_update=240, minimum_lr_ratio=0.2),
        loss=dict(route_peak=0.03, route_warmup=13, route_decay_start=240,
                  route_final=0.015))


class Contracts(unittest.TestCase):
    def test_configs_match_v10_design(self):
        import yaml
        source_dir = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v10')
        for design, code in (('T1', 'e'), ('I1', 'f'),
                             ('T2', 'g'), ('I2', 'h')):
            cfg = load_config(ROOT / 'configs' / f'r18{code}.yaml')
            source = yaml.safe_load((source_dir / f'{design}.yaml').read_text())
            self.assertEqual(cfg['plan'], code.upper())
            self.assertEqual(cfg['design_plan'], design)
            self.assertEqual(cfg['task'], source['task'])
            self.assertEqual(cfg['mechanism'], source['mechanism'])
            self.assertEqual(cfg['model']['injection_layers'], source['model']['injection_layers'])
            self.assertEqual(cfg['input'], source['input'])
            self.assertEqual(cfg['inference'], source['inference'])
            self.assertEqual(cfg['train']['max_updates'], 400)
            self.assertEqual(cfg['train']['video_lora_warmup'], 51)
            self.assertEqual(cfg['train']['physics_warmup'], 13)
            self.assertEqual(cfg['train']['peak_end_update'], 240)
            self.assertEqual(cfg['loss']['route_warmup'], 13)
            self.assertEqual(cfg['loss']['route_decay_start'], 240)
            self.assertEqual(cfg['model']['conditioned_ffn_rank'],
                             source['model']['conditioned_ffn_rank'])
            for key in ('physics_slots', 'physics_width', 'q_width',
                        'q_heads', 'q_rounds', 'injection_layers',
                        'extra_video_blocks', 'split_temporal_physics_memory'):
                self.assertEqual(cfg['model'][key], source['model'][key])
            for key in ('model', 'project_checkpoint', 'freeze_text_qkvo',
                        'video_lora_rank', 'video_lora_alpha',
                        'video_lora_from_update'):
                self.assertEqual(cfg['base'][key], source['base'][key])
            for key in ('fm', 'tail', 'tail_fraction', 'route_peak',
                        'route_final', 'local_semantic_enabled', 'st_enabled',
                        'supervise_known_latent', 'unknown_scope'):
                self.assertEqual(cfg['loss'][key], source['loss'][key])
            for key in ('effective_global_batch', 'microbatch_per_device',
                        'gradient_accumulation', 'world_size', 'seed',
                        'minimum_lr_ratio', 'post_training', 'noise_strategy',
                        'sigma_min', 'sigma_max'):
                self.assertEqual(cfg['train'][key], source['train'][key])

    def test_compressed_400_schedule(self):
        c = cfg()
        self.assertAlmostEqual(lr_multiplier('wan_video_lora', 1, c), 1 / 51)
        self.assertAlmostEqual(lr_multiplier('physics_read', 1, c), 1 / 13)
        self.assertEqual(lr_multiplier('wan_video_lora', 51, c), 1.0)
        self.assertEqual(lr_multiplier('physics_read', 13, c), 1.0)
        self.assertEqual(lr_multiplier('physics_read', 240, c), 1.0)
        self.assertAlmostEqual(lr_multiplier('physics_read', 400, c), .2)
        self.assertAlmostEqual(route_coefficient(1, c), .03 / 13)
        self.assertAlmostEqual(route_coefficient(13, c), .03)
        self.assertEqual(route_coefficient(240, c), .03)
        self.assertAlmostEqual(route_coefficient(400, c), .015)

    def test_task_loss_mask(self):
        prediction = torch.zeros(1, 2, 2, 2, 2)
        clean = torch.zeros_like(prediction)
        noise = torch.zeros_like(prediction)
        # First latent frame is wrong; future is exact.
        prediction[:, :, 0] = 1.0
        valid = torch.ones(2, 2)
        i2v, _ = generation_loss(prediction, noise, clean, valid, 'i2v',
                                 tail_weight=0)
        t2v, _ = generation_loss(prediction, noise, clean, valid, 't2v',
                                 tail_weight=0)
        self.assertEqual(float(i2v), 0.0)
        self.assertGreater(float(t2v), 0.0)

    def test_mechanism1_zero_init_and_gradient_path(self):
        attention = FakeAttention()
        mechanism = Mechanism1(FakeBank(attention), 0, cfg(mechanism=1))
        hidden = torch.randn(1, 4, 8)
        qt = torch.randn(1, 2, 4, 4)
        text_keys = torch.randn(1, 2, 3, 4)
        text_values = torch.randn(1, 2, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        route = torch.randn(1, 5, 8)
        execute = torch.randn(1, 5, 8)
        groups = torch.arange(5)
        times = torch.zeros(1, 1)
        output = mechanism(hidden, hidden, qt, text_keys, text_values, mask,
                           route, execute, groups, times, 4, None)
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        with torch.no_grad():
            mechanism.direct_writer.weight.normal_()
            mechanism.semantic_writer.weight.normal_()
        output = mechanism(hidden, hidden, qt, text_keys, text_values, mask,
                           route, execute, groups, times, 4, None)
        self.assertGreater(float(output.abs().sum()), 0.0)

    def test_mechanism2_conditioned_adapter(self):
        attention = FakeAttention()
        mechanism = Mechanism2(FakeBank(attention), 0, cfg(mechanism=2))
        hidden = torch.randn(1, 4, 8)
        text_keys = torch.randn(1, 2, 3, 4)
        text_values = torch.randn(1, 2, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        route = torch.randn(1, 5, 8)
        execute = torch.randn(1, 5, 8)
        groups = torch.arange(5)
        times = torch.zeros(1, 1)
        direct, z_full = mechanism(hidden, hidden, text_keys, text_values, mask,
                                   route, execute, groups, times, 4, None)
        self.assertTrue(torch.equal(direct, torch.zeros_like(direct)))
        self.assertGreater(float(z_full.abs().sum()), 0.0)
        f = torch.randn(1, 4, 16)
        zero = mechanism.conditioned(f, z_full, torch.ones(1, 4, 1))
        self.assertTrue(torch.equal(zero, torch.zeros_like(zero)))
        with torch.no_grad():
            mechanism.conditioned.b.weight.normal_()
        value = mechanism.conditioned(f, z_full, torch.ones(1, 4, 1))
        self.assertGreater(float(value.abs().sum()), 0.0)

    def test_reader_future_only_shapes(self):
        attention = FakeAttention()
        reader = DirectPhysicalReader(attention)
        hidden = torch.randn(1, 8, 8)
        route = torch.randn(1, 5, 8)
        execute = torch.randn(1, 5, 8)
        groups = torch.arange(5)
        times = torch.zeros(1, 2)
        full = reader(hidden, route, execute, groups, times, 4)
        future = reader(hidden, route, execute, groups, times, 4, future_start=4)
        self.assertEqual(tuple(full.shape), (1, 8, 8))
        self.assertEqual(tuple(future.shape), (1, 4, 8))
        joint = JointPhysicalReader(attention)
        text_keys = torch.randn(1, 2, 3, 4)
        text_values = torch.randn(1, 2, 3, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        value = joint(hidden, route, execute, groups, text_keys, text_values,
                      mask, times, 4)
        self.assertEqual(tuple(value.shape), (1, 8, 8))
        self.assertTrue(torch.isfinite(value).all())


if __name__ == '__main__':
    unittest.main()
