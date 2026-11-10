"""Actual small Wan: local operator, two C conditions, gradients and resume."""
from pathlib import Path
import copy, importlib.util, sys, tempfile, types, unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from unittest.mock import patch
from guidance import guided_velocity, LEGACY, execution_options
from physical_content.model import PhysicalModel
from physical_content.local_attention import local_attention
from physical_content.objectives import backward_sample
from physical_content.train import save, restore
torch.set_num_threads(2)


def tiny_wan():
    name = '_r16c_test_wan'
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


class LocalAttentionTest(unittest.TestCase):
    def test_noncyclic_padding_and_temporal_off(self):
        torch.manual_seed(3)
        geometry = (3, 2, 4)
        q = torch.randn(1, 24, 2, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        full = local_attention(q, k, v, geometry, window=(3, 3, 3), shift=1)
        self.assertEqual(full.shape, q.shape)
        self.assertTrue(torch.isfinite(full).all())
        spatial = local_attention(q, k, v, geometry, window=(3, 3, 3), shift=1,
                                  temporal_enabled=False)
        self.assertFalse(torch.allclose(full, spatial))
        # Each frame's value cannot depend on another frame when temporal edges
        # are disabled: repeat one frame's K/V and verify only that frame changes.
        changed_k = k.clone()
        changed_v = v.clone()
        frame = geometry[1] * geometry[2]
        changed_k[:, :frame] += 100
        changed_v[:, :frame] += 100
        altered = local_attention(q, changed_k, changed_v, geometry, window=(3, 3, 3),
                                  shift=1, temporal_enabled=False)
        # Frame0 supplies its own changed K/V, so it must change; every other
        # frame's window is same-frame only and must stay identical.
        self.assertFalse(torch.allclose(altered[:, :frame], spatial[:, :frame]))
        self.assertTrue(torch.allclose(altered[:, frame:], spatial[:, frame:]))


class RequestLocalTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32), scale=torch.rand(32) + .5,
                        weight=torch.randn(16, 32) * .05, bias=torch.randn(16) * .01,
                        target_scale=torch.tensor(.1)), self.path / 'interface.pt')
        self.cfg = dict(
            variant='physical_content', revision='physical_content_v7_native_request_local_r16c',
            model=dict(qwen_width=32, width=16, q_heads=4, q_tokens=8, rounds=2,
                       grounded_read_heads=2, condition_every=2, lora_rank=2,
                       physical_window=[3, 3, 3]),
            paths=dict(interface=str(self.path / 'interface.pt')),
            stages=dict(interface=128, joint=272),
            loss=dict(generation=dict(tail_weight=.2, tail_fraction=.25)),
            train=dict(seed=42, gradient_checkpointing=True, keep_steps=[129],
                       peak_lrs=dict(enhancement=.001, fusion_read=.001,
                                     fusion_write=.001, wan_lora=.001)))
        self.wan = tiny_wan()
        with torch.no_grad():
            for block in self.wan.blocks:
                block.cross_attn.o.bias.normal_(std=.2)
        self.evidence = dict(words=torch.randn(6, 32), word_groups=torch.arange(6) // 3,
                             request_words=torch.randn(5, 32),
                             native_words=torch.randn(6, 16) * .1,
                             memory=torch.randn(1, 17, 32),
                             memory_mask=torch.ones(1, 17, dtype=torch.bool))
        self.clean = torch.randn(1, 2, 3, 2, 4)
        self.noise = torch.randn_like(self.clean)
        self.text = torch.randn(1, 4, 16)
        self.sample = dict(latent=self.clean, first=self.clean[:, :, :1],
                           times=torch.tensor([[0., 1., 2.]]), valid=torch.ones(2, 4))
        self.sigma = torch.tensor(.6)

    def model(self):
        torch.manual_seed(42)
        return PhysicalModel(copy.deepcopy(self.cfg), device='cpu',
                             wan=copy.deepcopy(self.wan))

    def full(self, model, **options):
        c, groups = model.enhancement(self.evidence)
        return model.velocity((1 - self.sigma) * self.clean + self.sigma * self.noise,
                              self.sample['first'], self.text, self.sigma, c,
                              self.sample['times'], groups, **options)[0]

    def activate(self, model):
        with torch.no_grad():
            model.enhancement.project.weight.normal_(std=.03)
            model.enhancement.execution_project.weight.normal_(std=.03)
            for fusion in model.fusions.values():
                for name in ('output', 'condition_query', 'request_condition_query'):
                    getattr(fusion, name).weight.normal_(std=.04)

    def test_native_start_frozen_text_and_optimizer_ownership(self):
        model = self.model()
        torch.testing.assert_close(self.full(model), self.full(model, physical_enabled=False),
                                   rtol=1e-5, atol=1e-6)
        fusion = model.fusions['1']
        native = model.wan.blocks[1].cross_attn
        for own, other in [('key', 'k'), ('value', 'v')]:
            a, b = getattr(fusion, own).weight, getattr(native, other).weight
            self.assertTrue(a.requires_grad)
            self.assertFalse(b.requires_grad)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name in ('q', 'k', 'v', 'o'):
            self.assertFalse(getattr(native, name).weight.requires_grad)
        self.assertFalse(hasattr(fusion, 'query'))
        for layer in (fusion.output, fusion.condition_query, fusion.request_condition_query):
            self.assertEqual(float(layer.weight.abs().max()), 0.)
        ids = [id(p) for group in model.optim_groups() for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})

    def test_both_conditions_change_predictions_independently(self):
        model = self.model()
        self.activate(model)
        normal = self.full(model)
        for options in [dict(physical_enabled=False), dict(condition_enabled=False),
                        dict(request_condition_enabled=False),
                        dict(direct_condition_enabled=False),
                        dict(temporal_enabled=False)]:
            self.assertGreater(float((normal - self.full(model, **options)).detach().abs().max()), 1e-7)
        direct = self.full(model, request_condition_enabled=False)
        request = self.full(model, direct_condition_enabled=False)
        self.assertGreater(float((direct - request).detach().abs().max()), 1e-7)
        # The same C with a different request changes the prediction through
        # the request-bound condition, not through native text (text unchanged).
        original = self.evidence['request_words']
        self.evidence['request_words'] = torch.randn_like(original)
        changed = self.full(model)
        self.evidence['request_words'] = original
        self.assertGreater(float((normal - changed).detach().abs().max()), 1e-7)
        torch.testing.assert_close(normal, self.full(model), rtol=0, atol=0)

    def test_first_frame_is_not_written_and_padding_is_excluded(self):
        model = self.model()
        self.activate(model)
        fusion = model.fusions['1']
        native = model.wan.blocks[1].cross_attn
        c, groups = model.enhancement(self.evidence)
        width = model.cfg['model']['width']
        x = torch.randn(1, 6, width)
        video_state = torch.randn(1, 6, width)
        grid = torch.tensor([[3, 1, 2]])
        rope = native.forward.__func__.__globals__['rope_apply']
        context = torch.randn(1, 5, 16)
        first = fusion(x, video_state, c, self.sample['times'], (3, 1, 2), native,
                       groups, grid, model.wan.freqs, rope, context, 3)
        self.assertEqual(float(first[:, :2].detach().abs().max()), 0.)
        changed = context.clone()
        changed[:, 3:] += 100
        second = fusion(x, video_state, c, self.sample['times'], (3, 1, 2), native,
                        groups, grid, model.wan.freqs, rope, changed, 3)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_generation_and_alignment_gradient_ownership(self):
        model = self.model()
        model.set_phase(0)
        backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                        self.noise, 0., 1)
        self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()), 0.)
        model.zero_grad(set_to_none=True)
        self.activate(model)
        model.set_phase(127)
        backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                        self.noise, 0., 1)
        f = model.fusions['1']
        for p in [model.enhancement.project.weight, model.enhancement.execution_project.weight,
                  *[getattr(f, k).weight for k in
                    ('condition_query', 'request_condition_query', 'video_q', 'video_k',
                     'video_v', 'key', 'value', 'output')]]:
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.norm()), 0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True)
        c, _ = model.enhancement(self.evidence)
        model.enhancement.content_loss(c, self.evidence).backward()
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        model.zero_grad(set_to_none=True)
        self.assertEqual(model.set_phase(128), 'joint')
        backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                        self.noise, .03, 1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n, p in model.wan.named_parameters()
                               if n.endswith(('lora_a', 'lora_b')) and p.grad is not None), 0.)
        self.assertTrue(all(p.grad is None for n, p in model.wan.named_parameters()
                            if not n.endswith(('lora_a', 'lora_b'))))

    def test_zero_init_gradient_opening_order(self):
        # Frozen zero-init stack opens one projection per optimizer step:
        # step0 output, step1 both C-condition projections, step2 C K/V.
        model = self.model()
        optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
        fusion = model.fusions['1']
        names = ('output', 'condition_query', 'request_condition_query')
        seen = []
        for step in (0, 1, 2):
            model.set_phase(step)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                            self.noise, 0., 1)
            seen.append({name: float(getattr(fusion, name).weight.grad.norm())
                         for name in names} |
                        {'key': float(fusion.key.weight.grad.norm()),
                         'execution_C': float(model.enhancement.execution_project.weight.grad.norm())})
            optimizer.step()
        self.assertGreater(seen[0]['output'], 0.)
        self.assertGreater(seen[1]['condition_query'], 0.)
        self.assertGreater(seen[1]['request_condition_query'], 0.)
        self.assertGreater(seen[2]['key'], 0.)
        self.assertGreater(seen[2]['execution_C'], 0.)

    def test_guidance_controls_and_checkpoint_resume(self):
        model = self.model()
        self.activate(model)
        c, groups = model.enhancement(self.evidence)
        noisy = (1 - self.sigma) * self.clean + self.sigma * self.noise
        null = self.text[:, :1] * 0
        for mode in ('real', 'no_c', 'condition_off', 'direct_condition_only',
                     'request_condition_only', 'temporal_off'):
            actual = guided_velocity(model, noisy, self.sample['first'], self.text, null,
                                     self.sigma, c, c, self.sample['times'], groups, groups,
                                     composition=LEGACY, guidance=5., mode=mode)
            options = execution_options(mode)
            pos, _ = model.velocity(noisy, self.sample['first'], self.text, self.sigma,
                                    c, self.sample['times'], groups, **options)
            neg, _ = model.velocity(noisy, self.sample['first'], null, self.sigma,
                                    c, self.sample['times'], groups, **options)
            torch.testing.assert_close(actual, neg + 5. * (pos - neg), rtol=0, atol=0)

        model.set_phase(128)
        optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
        backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                        self.noise, .03, 1)
        optimizer.step()
        with patch('torch.cuda.get_rng_state_all', return_value=[]):
            save(model, optimizer, self.cfg, self.path, 129, 516)
        other = self.model()
        new_optimizer = torch.optim.AdamW(other.optim_groups(), foreach=False)
        with patch('torch.cuda.set_rng_state_all'):
            self.assertEqual(restore(other, new_optimizer, self.path / 'step0129.pt',
                                     self.cfg), (129, 516))
        for options in ({}, {'memory_mode': 'direct'}, {'physical_enabled': False},
                        {'condition_enabled': False}, {'temporal_enabled': False}):
            torch.testing.assert_close(self.full(model, **options),
                                       self.full(other, **options), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
