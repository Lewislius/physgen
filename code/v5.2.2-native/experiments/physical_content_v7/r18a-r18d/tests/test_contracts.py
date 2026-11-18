"""CPU algebra/ownership checks for R18A-D; not a Wan or video test."""
import copy
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

import torch
from torch import nn
from torch.nn import functional as F

from physical_content.common import (
    EXPECTED_TARGETS, REVISION_1000, SITES, load_config)
from physical_content.content import SemanticPhysicsMemoryEnhancer
from physical_content.experts import GuidedPhysicsExecutor, PhysicalExecutor
from physical_content.lora import FrozenBaseLoRALinear, install_video_lora
from physical_content.readers import PhysicalReader
from physical_content.schedule import lr_multiplier, route_coefficient
from physical_content.supervision import generation_loss


class RMS(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * self.weight


class FakeAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = width, heads, width // heads
        self.q, self.k, self.v, self.o = (nn.Linear(width, width) for _ in range(4))
        self.norm_q, self.norm_k = RMS(width), RMS(width)

    def forward(self, x, context=None, context_lens=None, seq_lens=None,
                grid_sizes=None, freqs=None):
        source = x if context is None else context
        split = lambda value: value.reshape(value.shape[0], -1, self.num_heads, self.head_dim).transpose(1, 2)
        q = split(self.norm_q(self.q(x)))
        k = split(self.norm_k(self.k(source)))
        v = split(self.v(source))
        out = F.scaled_dot_product_attention(q, k, v)
        return self.o(out.transpose(1, 2).flatten(2))


class FakeBlock(nn.Module):
    def __init__(self, width=8, ffn_width=16, heads=2):
        super().__init__()
        self.dim = width
        self.self_attn = FakeAttention(width, heads)
        self.cross_attn = FakeAttention(width, heads)
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm3 = nn.Identity()
        self.ffn = nn.Sequential(nn.Linear(width, ffn_width), nn.GELU(),
                                 nn.Linear(ffn_width, width))
        self.modulation = nn.Parameter(torch.randn(1, 6, width) / width ** .5)

    def forward(self, x, e, seq_lens=None, grid_sizes=None, freqs=None,
                context=None, context_lens=None):
        parts = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        y = self.self_attn(self.norm1(x) * (1 + parts[1].squeeze(2)) + parts[0].squeeze(2))
        x = x + y * parts[2].squeeze(2)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + parts[4].squeeze(2)) + parts[3].squeeze(2))
        return x + y * parts[5].squeeze(2)


class FakeBank:
    def __init__(self, block):
        self.block = block

    def attention(self, layer, name):
        return getattr(self.block, name)

    def ffn(self, layer):
        return self.block.ffn

    def block_norm(self, layer, name):
        return getattr(self.block, name)


def cfg_for(width=8, ffn_width=16, variant='D'):
    return {
        'variant': variant,
        'model': {
            'width': width, 'ffn_width': ffn_width,
            'expert_lora_rank': 2, 'expert_lora_alpha': 2,
            'semantic_reader_lora_rank': 2,
            'memory_enhancer': {'enabled': True, 'source_layer_zero_based': 0,
                                'rounds_shared': 2, 'lora_rank': 2,
                                'route_projection_init': 'identity'},
            'semantic_guide': {'enabled': True, 'maps': ['q', 'k', 'ffn_channels'],
                               'width': width, 'init': 'zeros',
                               'detach_native_text_update': False},
        },
    }


def parts_for(block, length):
    e = torch.randn(1, length, 6, block.dim) * .02
    return tuple(x.squeeze(2) for x in (block.modulation.unsqueeze(0) + e).chunk(6, dim=2))


def fake_wan(blocks):
    return type('FakeWan', (), {'blocks': nn.ModuleList(blocks)})()


class Contracts(unittest.TestCase):
    def test_configs_match_architecture_and_400_schedule(self):
        v9 = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v9/configs')
        for key in 'abcd':
            cfg = load_config(ROOT / 'configs' / f'r18{key}.yaml')
            import yaml
            source = yaml.safe_load((v9 / f'{key}_fresh1000_batch4.yaml').read_text())
            for field in ('variant', 'initialization', 'inference', 'evaluation'):
                self.assertEqual(cfg[field], source[field])
            ours_model = copy.deepcopy(cfg['model'])
            source_model = copy.deepcopy(source['model'])
            ours_model['native_video_lora'].pop('active_updates')
            source_model['native_video_lora'].pop('active_updates')
            self.assertEqual(ours_model, source_model)
            self.assertEqual(cfg['train']['max_updates'], 400)
            self.assertEqual(cfg['model']['native_video_lora']['active_updates'], [1, 400])
            self.assertEqual(cfg['model']['sites_zero_based'], SITES[key.upper()])
            self.assertEqual(set(cfg['model']['native_video_lora']['targets']), EXPECTED_TARGETS)

    def test_schedule_endpoints(self):
        self.assertAlmostEqual(lr_multiplier(1), 1 / 32)
        self.assertEqual(lr_multiplier(32), 1.0)
        self.assertEqual(lr_multiplier(128), 1.0)
        self.assertAlmostEqual(lr_multiplier(320), .2)
        self.assertAlmostEqual(lr_multiplier(400), .2)
        self.assertAlmostEqual(route_coefficient(1), .03 / 32)
        self.assertAlmostEqual(route_coefficient(32), .03)
        self.assertAlmostEqual(route_coefficient(128), .03)
        self.assertAlmostEqual(route_coefficient(129), .03 - .02 / 192)
        self.assertAlmostEqual(route_coefficient(320), .01)
        self.assertEqual(route_coefficient(321), .01)
        self.assertEqual(route_coefficient(400), .01)
        self.assertTrue(all(lr_multiplier(i) >= lr_multiplier(i + 1) for i in range(128, 400)))

    def test_1000_step_config_and_schedule(self):
        import yaml
        v9 = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v9/configs')
        cfg = load_config(ROOT / 'configs/r18d_1000.yaml')
        source = yaml.safe_load((v9 / 'd_fresh1000_batch4.yaml').read_text())
        self.assertEqual(cfg['schema_version'], source['schema_version'])
        self.assertEqual(cfg['train']['max_updates'], 1000)
        self.assertEqual(cfg['revision'], REVISION_1000)
        self.assertEqual(cfg['model']['native_video_lora']['active_updates'], [1, 1000])
        self.assertEqual(cfg['train']['lr_schedule'], source['train']['lr_schedule'])
        self.assertEqual(cfg['loss']['route'], source['loss']['route'])
        self.assertEqual(cfg['train']['keep_updates'], source['train']['keep_updates'])
        self.assertAlmostEqual(lr_multiplier(1, 1000), 1 / 64)
        self.assertEqual(lr_multiplier(64, 1000), 1.0)
        self.assertEqual(lr_multiplier(200, 1000), 1.0)
        self.assertAlmostEqual(lr_multiplier(1000, 1000), .2)
        self.assertAlmostEqual(route_coefficient(32, 1000), .03)
        self.assertEqual(route_coefficient(64, 1000), .03)
        self.assertEqual(route_coefficient(320, 1000), 0.0)
        self.assertEqual(route_coefficient(1000, 1000), 0.0)
        self.assertTrue(all(lr_multiplier(i, 1000) >= lr_multiplier(i + 1, 1000)
                            for i in range(200, 1000)))

    def test_lora_zero_and_gradient_ownership(self):
        base = nn.Linear(8, 8)
        lora = FrozenBaseLoRALinear(base, 2, 2)
        x = torch.randn(3, 8)
        self.assertTrue(torch.equal(lora(x), base(x)))
        target = torch.randn(3, 8)
        lora(x).sub(target).square().mean().backward()
        self.assertIsNotNone(lora.lora_b.grad)
        self.assertGreater(lora.lora_b.grad.abs().sum(), 0)
        self.assertEqual(lora.lora_a.grad.abs().sum(), 0)
        self.assertFalse(base.weight.requires_grad)
        self.assertIsNot(lora.lora_a, base.weight)

    def test_installer_all_180_and_rank_map(self):
        blocks = [FakeBlock() for _ in range(30)]
        wan = fake_wan(blocks)
        cfg = {'variant': 'B', 'model': {'native_video_lora': {
            'enabled': True, 'rank': 32, 'alpha': 32, 'dropout': 0.0,
            'targets': sorted(EXPECTED_TARGETS), 'layers': list(range(30)),
            'active_updates': [1, 400], 'rank64_layers': SITES['B']}}}
        installed = install_video_lora(wan, cfg)
        self.assertEqual(len(installed), 180)
        self.assertFalse(any('cross_attn' in name for name in installed))
        self.assertEqual(wan.blocks[2].self_attn.q.rank, 64)
        self.assertEqual(wan.blocks[1].self_attn.q.rank, 32)

    def test_physical_reader_all_tokens(self):
        attention = FakeAttention(8, 2)
        reader = PhysicalReader(attention, chunk=2)
        hidden = torch.randn(1, 7, 8)
        route = torch.randn(1, 5, 8)
        execute = torch.randn(1, 5, 8)
        groups = torch.tensor([0, 0, 0, 1, 1])
        output = reader(hidden, route, execute, groups)
        self.assertEqual(output.shape, (1, 7, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_c_memory_changes_execution_and_identity_route(self):
        block = FakeBlock()
        memory = SemanticPhysicsMemoryEnhancer(block, cfg_for(variant='C'))
        execution = torch.randn(1, 4, 8)
        context_a = torch.randn(1, 6, 8)
        context_b = torch.randn(1, 6, 8)
        mask = torch.ones(1, 6, dtype=torch.bool)
        raw = dict(routing=torch.randn(1, 4, 8), execution=execution)
        a = memory.enhance(raw, context_a, mask)
        b = memory.enhance(raw, context_a, mask)
        c = memory.enhance(raw, context_b, mask)
        self.assertTrue(torch.allclose(a['execution'], b['execution']))
        self.assertFalse(torch.allclose(a['execution'], c['execution']))
        delta = a['execution'] - raw['execution']
        self.assertTrue(torch.allclose(a['routing'] - raw['routing'], delta, atol=1e-6))


    def test_c_zero_physics_implies_zero_output(self):
        block = FakeBlock()
        executor = PhysicalExecutor(FakeBank(block), 0, cfg_for(variant='C'))
        hidden = torch.randn(1, 4, 8)
        zeros = torch.zeros(1, 3, 8)
        groups = torch.tensor([0, 0, 1])
        out = executor(hidden, zeros, zeros, groups, parts_for(block, 4), 2,
                       torch.tensor([4]), None, None)
        self.assertEqual(float(out.abs().max()), 0.0)

    def test_d_zero_physics_implies_zero_output(self):
        block = FakeBlock()
        executor = GuidedPhysicsExecutor(FakeBank(block), 0, cfg_for(variant='D'))
        hidden = torch.randn(1, 4, 8)
        text = torch.randn(1, 4, 8)
        zeros = torch.zeros(1, 3, 8)
        groups = torch.tensor([0, 0, 1])
        out = executor(hidden, text, zeros, zeros, groups, parts_for(block, 4), 2,
                       torch.tensor([4]), None, None)
        self.assertEqual(float(out.abs().max()), 0.0)

        route = torch.randn(1, 3, 8)
        execute = torch.randn(1, 3, 8)
        optimizer = torch.optim.SGD(executor.parameters(), lr=.05)
        target = torch.randn(1, 4, 8)
        reached = False
        for step in range(3):
            optimizer.zero_grad()
            value = executor(hidden, text, route, execute, groups,
                             parts_for(block, 4), 2, torch.tensor([4]), None, None)
            (value - target).square().mean().backward()
            if step >= 1 and executor.uq.weight.grad is not None:
                reached |= float(executor.uq.weight.grad.abs().sum()) > 0
            optimizer.step()
        self.assertTrue(reached)

    def test_native_segmented_block_equivalence(self):
        block = FakeBlock()
        hidden = torch.randn(1, 4, 8)
        context = torch.randn(1, 6, 8)
        e = torch.randn(1, 4, 6, 8) * .02
        parts = tuple(x.squeeze(2) for x in (block.modulation.unsqueeze(0) + e).chunk(6, dim=2))
        h_sa = hidden + block.self_attn(block.norm1(hidden) * (1 + parts[1]) + parts[0]) * parts[2]
        text = block.cross_attn(block.norm3(h_sa), context, None)
        h_t = h_sa + text
        segmented = h_t + block.ffn(block.norm2(h_t) * (1 + parts[4]) + parts[3]) * parts[5]
        full = block(hidden, e, context=context)
        self.assertTrue(torch.allclose(segmented, full, atol=1e-6, rtol=1e-5))


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TinyHead(nn.Module):
    def forward(self, x, e):
        return x


class TinyWan(nn.Module):
    """Only the small interface needed by PhysicalModel.velocity."""

    def __init__(self):
        super().__init__()
        self.text_len = 16
        self.freqs = None
        self.freq_dim = 8
        self.dim = 8
        self.patch_embedding = nn.Conv3d(4, 8, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.text_embedding = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 8))
        self.time_embedding = nn.Sequential(nn.Linear(8, 8), nn.SiLU(), nn.Linear(8, 8))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))
        self.head = TinyHead()
        self.blocks = nn.ModuleList([FakeBlock() for _ in range(30)])

    def unpatchify(self, x, grid):
        return x.transpose(1, 2).unsqueeze(-1)


class ModelTests(unittest.TestCase):
    def test_every_configured_site_is_called(self):
        from physical_content.common import ROUTE_GROUPS
        from physical_content.model import PhysicalModel
        cfg = {
            'variant': 'C',
            'initialization': {'new_adapter_seed': 42},
            'train': {'gradient_checkpointing': False,
                      'peak_lrs': {role: 1e-5 for role in ROUTE_GROUPS},
                      'grad_clip_per_group': 1.0},
            'paths': {'interface': 'unused'},
            'model': {
                'width': 8, 'ffn_width': 16, 'qwen_width': 4, 'q_tokens': 4,
                'q_heads': 2, 'grounded_heads': 1, 'content_rounds': 2,
                'sites_zero_based': SITES['C'], 'expert_lora_rank': 2,
                'expert_lora_alpha': 2, 'semantic_reader_lora_rank': 2,
                'memory_enhancer': {'source_layer_zero_based': 0, 'rounds_shared': 2,
                                    'lora_rank': 2, 'route_projection_init': 'identity'},
                'semantic_guide': {'enabled': True, 'maps': ['q', 'k', 'ffn_channels'],
                                   'width': 8, 'init': 'zeros',
                                   'detach_native_text_update': False},
                'native_video_lora': {
                    'rank': 32, 'rank64_layers': [],
                    'targets': sorted(EXPECTED_TARGETS), 'layers': list(range(30))},
            },
        }
        interface = dict(mean=torch.zeros(4), scale=torch.tensor(1.),
                         weight=torch.zeros(8, 4), bias=torch.zeros(8),
                         target_scale=torch.tensor(1.))
        model = PhysicalModel(cfg, device='cpu', wan=TinyWan(), interface=interface)
        hits = []
        for key, module in model.executors.items():
            module.register_forward_hook(lambda *args, key=key, **kwargs: hits.append(key))
        evidence = dict(
            words=torch.randn(8, 4), word_groups=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
            request_words=torch.randn(3, 4), native_words=torch.randn(8, 8))
        text = torch.randn(1, 6, 4)
        content, groups = model.prepare_content(evidence, text)
        noisy = torch.randn(1, 4, 2, 4, 4)
        first = torch.randn(1, 4, 1, 4, 4)
        output, _ = model.velocity(noisy, first, text, torch.tensor(.5), content, groups)
        self.assertEqual(output.shape, (1, 8, 8, 1))
        self.assertEqual(hits, [str(site) for site in SITES['C']])
