"""Tiny official-Wan CPU checks for R16Gt: T2V scope and R16G channels."""
import copy
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

from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample

torch.set_num_threads(2)


def tiny_wan():
    name = '_r16gt_test_wan'
    if name + '.model' not in sys.modules:
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
    wan = sys.modules[name + '.model'].WanModel(
        model_type='ti2v', patch_size=(1, 2, 2), text_len=5, in_dim=2, out_dim=2,
        dim=16, ffn_dim=32, freq_dim=8, text_dim=16, num_heads=2, num_layers=2)
    with torch.no_grad():
        wan.head.head.weight.normal_(std=.1)
    return wan


class TinyR16Gt(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.cfg = dict(
            variant='physical_content',
            revision='physical_content_v7_response_fusion_r16gt_t2v',
            task='t2v',
            model=dict(qwen_width=32, width=16, q_heads=4, q_tokens=8, rounds=2,
                       grounded_read_heads=2, condition_every=2, lora_rank=2,
                       physical_window=[5, 3, 3],
                       response_mode='triple_global_pure_local'),
            paths=dict(interface='/nonexistent/interface.pt'),
            stages=dict(interface=128, joint=272),
            loss=dict(generation=dict(tail_weight=.2, tail_fraction=.25)),
            train=dict(seed=42, gradient_checkpointing=False, keep_steps=[129],
                       peak_lrs=dict(enhancement=.001, fusion_read=.001,
                                     fusion_write=.001, wan_lora=.001)))
        torch.manual_seed(42)
        self.wan = tiny_wan()
        with torch.no_grad():
            for block in self.wan.blocks:
                block.cross_attn.o.bias.normal_(std=.2)
        self.evidence = dict(
            words=torch.randn(12, 32),
            word_groups=torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
            word_positions=torch.arange(12), word_strings=['a'] * 12,
            request_words=torch.randn(4, 32), request_word_strings=['x'] * 4,
            relations=['a', 'b', 'c', 'd'], caption='x y', sample='s',
            scope='request_only', native_words=torch.randn(12, 16))
        self.clean = torch.randn(1, 2, 3, 2, 4)
        self.noise = torch.randn_like(self.clean)
        self.text = torch.randn(1, 4, 16)
        self.times = torch.tensor([[0., 1., 2.]])
        self.sigma = torch.tensor(.6)
        self.sample = dict(latent=self.clean, times=self.times,
                           valid=torch.ones(2, 4))

    def model(self):
        interface = dict(mean=torch.zeros(32), scale=torch.ones(32),
                         weight=torch.randn(16, 32) * .01, bias=torch.zeros(16),
                         target_scale=torch.tensor(1.0))
        torch.manual_seed(42)
        return PhysicalModel(self.cfg, device='cpu', wan=copy.deepcopy(self.wan),
                             interface=interface)

    def full(self, model, **options):
        c, groups = model.enhancement(self.evidence)
        noisy = (1 - self.sigma) * self.clean + self.sigma * self.noise
        return model.velocity(noisy, None, self.text, self.sigma, c,
                              self.times, groups, **options)[0]

    @staticmethod
    def activate(model):
        with torch.no_grad():
            model.enhancement.project.weight.normal_(std=.03)
            model.enhancement.execution_project.weight.normal_(std=.03)
            for fusion in model.fusions.values():
                for name in ('output', 'condition_query', 'request_condition_query',
                             'joint_condition_query', 'time', 'direct_output',
                             'request_output', 'joint_output', 'direct_state_value',
                             'request_state_value', 'joint_state_value'):
                    getattr(fusion, name).weight.normal_(std=.04)
            model.value_correction.weight.normal_(std=.04)

    def test_t2v_forward_backward_and_first_frame_rejected(self):
        model = self.model()
        prediction = self.full(model)
        self.assertEqual(tuple(prediction.shape), (1, 2, 3, 2, 4))
        self.assertTrue(torch.isfinite(prediction).all())
        prediction.square().mean().backward()
        self.assertTrue(any(parameter.grad is not None
                            for parameter in model.parameters()
                            if parameter.requires_grad))
        first = torch.zeros(1, 2, 1, 2, 4)
        c, groups = model.enhancement(self.evidence)
        with self.assertRaises(ValueError):
            model.velocity(torch.randn(1, 2, 3, 2, 4), first, self.text,
                           torch.tensor(.5), c, self.times, groups)

    def test_r16g_zero_init_native_start_and_optimizer_ownership(self):
        model = self.model()
        torch.testing.assert_close(self.full(model),
                                   self.full(model, physical_enabled=False),
                                   rtol=1e-5, atol=1e-6)
        fusion = model.fusions['1']
        native = model.wan.blocks[1].cross_attn
        for own, other in (('key', 'k'), ('value', 'v')):
            source = getattr(fusion, own).weight
            target = getattr(native, other).weight
            self.assertTrue(source.requires_grad)
            self.assertFalse(target.requires_grad)
            torch.testing.assert_close(source, target, rtol=0, atol=0)
        for name in ('q', 'k', 'v', 'o'):
            self.assertFalse(getattr(native, name).weight.requires_grad)
        for layer in (fusion.output, fusion.condition_query, fusion.time,
                      fusion.direct_output, fusion.request_output,
                      fusion.joint_output, fusion.direct_state_value,
                      fusion.request_state_value, fusion.joint_state_value,
                      model.value_correction):
            self.assertEqual(float(layer.weight.detach().abs().max()), 0.)
        ids = [id(p) for group in model.optim_groups() for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()
                                    if p.requires_grad})

    def test_all_declared_controls_change_predictions(self):
        model = self.model()
        self.activate(model)
        normal = self.full(model)
        flags = dict(condition_off='condition_enabled',
                     global_off='global_enabled',
                     local_off='local_enabled',
                     global_direct_off='global_direct_enabled',
                     global_request_off='global_request_enabled',
                     global_joint_off='global_joint_enabled',
                     temporal_off='temporal_enabled')
        for mode, flag in flags.items():
            changed = self.full(model, **{flag: False})
            self.assertGreater(float((normal - changed).detach().abs().max()),
                               1e-7, mode)

    def test_physical_write_covers_every_frame(self):
        model = self.model()
        self.activate(model)
        full = self.full(model)
        native = self.full(model, physical_enabled=False)
        per_frame = (full - native).detach().abs().amax(dim=(0, 1, 3, 4))
        self.assertEqual(tuple(per_frame.shape), (3,))
        self.assertTrue(bool((per_frame > 1e-7).all()), per_frame)

    def test_gradient_opening_order(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
        fusion = model.fusions['1']
        grads = []
        for step in (0, 1, 2):
            model.set_phase(step)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text,
                            self.sigma, self.noise, 0., 1)
            get = lambda layer: float(layer.weight.grad.norm())
            grads.append(dict(output=get(fusion.output),
                              direct_output=get(fusion.direct_output),
                              request_output=get(fusion.request_output),
                              joint_output=get(fusion.joint_output),
                              condition_query=get(fusion.condition_query),
                              direct_state_value=get(fusion.direct_state_value),
                              request_state_value=get(fusion.request_state_value),
                              joint_state_value=get(fusion.joint_state_value),
                              video_q=get(fusion.video_q),
                              key=get(fusion.key)))
            optimizer.step()
        for key in ('output', 'direct_output', 'request_output', 'joint_output'):
            self.assertGreater(grads[0][key], 0., key)
        for key in ('condition_query', 'direct_state_value', 'request_state_value',
                    'joint_state_value', 'video_q', 'key'):
            self.assertGreater(grads[1][key], 0., key)

    def test_native_text_and_lora_gradient_ownership(self):
        model = self.model()
        self.activate(model)
        model.set_phase(0)
        optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
        optimizer.zero_grad(set_to_none=True)
        backward_sample(model, self.evidence, self.sample, self.text,
                        self.sigma, self.noise, .03, 1)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model.set_phase(128)
        backward_sample(model, self.evidence, self.sample, self.text,
                        self.sigma, self.noise, .03, 1)
        lora = [p for name, p in model.wan.named_parameters()
                if name.endswith(('lora_a', 'lora_b')) and p.grad is not None]
        self.assertTrue(lora)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in lora), 0.)
        self.assertTrue(all(p.grad is None for name, p in model.wan.named_parameters()
                            if not name.endswith(('lora_a', 'lora_b'))))


if __name__ == '__main__':
    unittest.main()
