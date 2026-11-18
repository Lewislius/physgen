"""CPU checks of the actual R9.1 implementation, inference and own-state restore."""
from pathlib import Path
import copy
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE if HERE.name == 'content_value_r9_1' else HERE.parents[1]
sys.path.insert(0, str(HERE))
import bootstrap
import torch
import yaml
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement, Fusion, PhysicalModel
from physical_content.evidence import resample_words
from physical_content.common import load_config
from physical_content.train import save, restore
from physical_scaling.schedule import learning_rates, auxiliary_coefficient
from infer import enhance_pair

PROJECT = bootstrap.BASE / 'experiments/physical_content_v7'
REFERENCE = PROJECT / 'r1_content_reader_r9/snapshots/20261031T150803Z/physical_content/model.py'
spec = importlib.util.spec_from_file_location('physical_content.r9_reference', REFERENCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
torch.set_num_threads(2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = 16, 2, 8
        for name in ['q', 'k', 'v', 'o']:
            setattr(self, name, nn.Linear(16, 16, bias=False))
        self.norm_q, self.norm_k = nn.RMSNorm(16), nn.RMSNorm(16)

    def forward(self, x, context):
        return x


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn, self.cross_attn = Attention(), Attention()
        self.ffn = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16))


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock()])


class ContentValueTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.cfg = dict(model=dict(qwen_width=32, width=16, q_heads=4,
            q_tokens=8, rounds=2, grounded_read_heads=2),
            train=dict(gradient_checkpointing=True))
        self.interface = dict(mean=torch.randn(32), scale=torch.rand(32) + .5,
            weight=torch.randn(16, 32) * .05, bias=torch.randn(16) * .01,
            target_scale=torch.tensor(.1))
        self.evidence = dict(words=torch.randn(6, 32), word_groups=torch.arange(6) // 3,
            request_words=torch.randn(5, 32))

    def test_existing_initializer_rng_and_initial_content(self):
        attention = Attention()
        torch.manual_seed(42)
        old, old_fusion = reference.Enhancement(self.cfg, self.interface), reference.Fusion(attention)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        new, new_fusion = Enhancement(self.cfg, self.interface), Fusion(attention)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(dict(new.named_parameters())) - set(dict(old.named_parameters())),
                         {'shared_offset'})
        for name, parameter in old.named_parameters():
            torch.testing.assert_close(parameter, dict(new.named_parameters())[name], rtol=0, atol=0)
        for name, parameter in old_fusion.state_dict().items():
            torch.testing.assert_close(parameter, new_fusion.state_dict()[name], rtol=0, atol=0)
        torch.testing.assert_close(old(self.evidence)[0], new(self.evidence)[0], rtol=0, atol=0)
        self.assertEqual(new.shared_offset.numel(), self.cfg['model']['width'])

    def test_identical_queries_still_read_different_input_values(self):
        model = Enhancement(self.cfg, self.interface)
        with torch.no_grad():
            model.block.cross.q.weight.zero_()
            model.block.cross.v.weight.copy_(torch.eye(32))
        self.assertIsNone(model.block.cross.v.bias)
        state = torch.ones(1, 8, 32)
        memory, words = torch.randn(1, 11, 32), torch.randn(1, 6, 32)
        groups = torch.arange(8) // 4
        before = model.read_values(state, memory, words, groups, self.evidence['word_groups'])
        shift = torch.linspace(-.3, .4, 32).reshape(1, 1, 32)
        after = model.read_values(state, memory + shift, words + shift, groups, self.evidence['word_groups'])
        torch.testing.assert_close(after - before, shift.expand_as(before), rtol=1e-5, atol=1e-6)

    def test_actual_inference_anchor_control_removes_both_corrections(self):
        enhancement = Enhancement(self.cfg, self.interface)
        with torch.no_grad():
            enhancement.project.weight.normal_(std=.02)
            enhancement.shared_offset.fill_(.3)
        model = nn.Module()
        model.enhancement = enhancement
        negative = dict(self.evidence, words=self.evidence['words'].flip(0),
                        request_words=torch.empty(0, 32))
        before = enhance_pair(model, self.evidence, negative, 'real')
        anchor = enhance_pair(model, self.evidence, negative, 'anchor_only')
        shared = enhance_pair(model, self.evidence, negative, 'shared_only')
        after = enhance_pair(model, self.evidence, negative, 'real')
        for evidence, index in [(self.evidence, 0), (negative, 2)]:
            words = (evidence['words'] - enhancement.anchor_mean) / enhancement.anchor_scale
            values, _, _ = resample_words(words, evidence['word_groups'], 8)
            expected = F.linear(values, enhancement.anchor_weight, enhancement.anchor_bias)
            torch.testing.assert_close(anchor[index], expected, rtol=0, atol=0)
            torch.testing.assert_close(shared[index], expected + enhancement.shared_offset, rtol=0, atol=0)
            torch.testing.assert_close(before[index], after[index], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(after[index]).all())
            self.assertGreater(float((before[index] - anchor[index]).detach().abs().max()), .1)

    def test_original_fusion_generation_gradients_reach_query_selection(self):
        model = Enhancement(self.cfg, self.interface)
        attention = Attention().requires_grad_(False)
        fusion = Fusion(attention)
        optimizer = torch.optim.SGD([*model.parameters(), *fusion.parameters()], lr=.2)
        x, context, target = torch.randn(1, 4, 16), torch.randn(1, 5, 16), torch.randn(1, 2, 16)
        norms = []
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            c, groups = model(self.evidence)
            y, _ = fusion(x, context, c, torch.tensor([[0., 1.]]), (2, 1, 2), attention, groups)
            F.mse_loss(y[:, 2:], target).backward()
            parameters = dict(output=fusion.output.weight, project=model.project.weight,
                              offset=model.shared_offset, seed=model.seed,
                              query=model.block.cross.q.weight, value=model.block.cross.v.weight)
            norms.append({name: float(p.grad.norm()) for name, p in parameters.items()})
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            optimizer.step()
        self.assertGreater(norms[0]['output'], 0.)
        for name in ['project', 'offset']:
            self.assertGreater(norms[1][name], 0.)
        for name in ['seed', 'query', 'value']:
            self.assertGreater(norms[2][name], 0.)

    def test_actual_parameter_ownership_optimizer_and_own_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            torch.save(self.interface, folder / 'interface.pt')
            cfg = copy.deepcopy(self.cfg)
            cfg.update(variant='physical_content', revision='physical_content_v7_content_value_r9_1',
                       paths=dict(interface=str(folder / 'interface.pt')),
                       stages=dict(interface=128, joint=272))
            cfg['model'].update(condition_every=1, lora_rank=2)
            cfg['train'].update(seed=42, keep_steps=[1], peak_lrs=dict(
                enhancement=.001, fusion_read=.001, fusion_write=.001, wan_lora=.001))
            model = PhysicalModel(cfg, device='cpu', wan=TinyWan())
            optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
            self.assertIn('enhancement.shared_offset', model.learned_names)
            owner = next(group for group in optimizer.param_groups if group['name'] == 'enhancement')
            self.assertTrue(any(p is model.enhancement.shared_offset for p in owner['params']))
            model.set_phase(0)
            model.enhancement(self.evidence)[0].square().mean().backward()
            optimizer.step()
            with patch('torch.cuda.get_rng_state_all', return_value=[]):
                save(model, optimizer, cfg, folder, 1, 4)
            restored = PhysicalModel(cfg, device='cpu', wan=TinyWan())
            resumed_optimizer = torch.optim.AdamW(restored.optim_groups(), foreach=False)
            with patch('torch.cuda.set_rng_state_all'):
                self.assertEqual(restore(restored, resumed_optimizer, folder / 'step0001.pt', cfg), (1, 4))
            left, right = dict(model.named_parameters()), dict(restored.named_parameters())
            for name in model.learned_names:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
            for key in ['step', 'exp_avg', 'exp_avg_sq']:
                torch.testing.assert_close(optimizer.state[model.enhancement.shared_offset][key],
                    resumed_optimizer.state[restored.enhancement.shared_offset][key], rtol=0, atol=0)
            bad = copy.deepcopy(cfg)
            bad['revision'] = 'another_revision'
            with self.assertRaises(ValueError):
                restore(restored, resumed_optimizer, folder / 'step0001.pt', bad)

    def test_original_400_recipe_and_declared_evaluation(self):
        original = yaml.safe_load((PROJECT / 'configs/physical_content.yaml').read_text())
        current = load_config(HERE / 'configs/physical_content.yaml')
        self.assertEqual(current['stages'], dict(interface=128, joint=272))
        self.assertEqual(original['loss'], current['loss'])
        self.assertEqual({k: v for k, v in original['train'].items() if k not in ['keep_steps', 'validation_steps']},
                         {k: v for k, v in current['train'].items() if k not in ['keep_steps', 'validation_steps']})
        for step in range(400):
            self.assertEqual(learning_rates(original, step), learning_rates(current, step))
            self.assertEqual(auxiliary_coefficient(original, step), auxiliary_coefficient(current, step))
        self.assertEqual((PROJECT / 'snapshots/20261030T141432Z/physical_content/train.py').read_bytes(),
                         (HERE / 'physical_content/train.py').read_bytes())
        self.assertEqual(current['evaluation_budget']['control_modes'],
                         ['reasoning_swapped', 'empty_reasoning', 'global_train_mean', 'shared_only', 'anchor_only', 'no_c'])
        self.assertEqual(current['evaluation_budget']['extended_evaluation']['cases'], 60)


if __name__ == '__main__':
    unittest.main()
