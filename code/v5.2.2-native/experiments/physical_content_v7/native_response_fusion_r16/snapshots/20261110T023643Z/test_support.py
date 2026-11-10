"""Actual small Wan: local operator, multi-read global writes, gradients and resume."""
from pathlib import Path
import copy, importlib.util, sys, tempfile, types, unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from unittest.mock import patch
from guidance import guided_velocity, LEGACY, CONTROL_MODES
from physical_content.model import PhysicalModel
from physical_content.local_attention import local_attention
from physical_content.objectives import backward_sample
from physical_content.train import save, restore
torch.set_num_threads(2)

MODES = ('triple_global_pure_local', 'triple_global_joint_local', 'triple_global_full_local')


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
        changed_k = k.clone()
        changed_v = v.clone()
        frame = geometry[1] * geometry[2]
        changed_k[:, :frame] += 100
        changed_v[:, :frame] += 100
        altered = local_attention(q, changed_k, changed_v, geometry, window=(3, 3, 3),
                                  shift=1, temporal_enabled=False)
        self.assertFalse(torch.allclose(altered[:, :frame], spatial[:, :frame]))
        self.assertTrue(torch.allclose(altered[:, frame:], spatial[:, frame:]))


class ResponseFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32), scale=torch.rand(32) + .5,
                        weight=torch.randn(16, 32) * .05, bias=torch.randn(16) * .01,
                        target_scale=torch.tensor(.1)), self.path / 'interface.pt')
        self.cfg = dict(
            variant='physical_content', revision='physical_content_v7_native_response_fusion_r16',
            model=dict(qwen_width=32, width=16, q_heads=4, q_tokens=8, rounds=2,
                       grounded_read_heads=2, condition_every=2, lora_rank=2,
                       physical_window=[5, 3, 3], response_mode='triple_global_pure_local'),
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

    def model(self, mode='triple_global_pure_local'):
        cfg = copy.deepcopy(self.cfg)
        cfg['model']['response_mode'] = mode
        torch.manual_seed(42)
        return PhysicalModel(cfg, device='cpu', wan=copy.deepcopy(self.wan))

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
                for name in ('output', 'condition_query', 'request_condition_query',
                             'joint_condition_query', 'time', 'direct_output',
                             'request_output', 'joint_output', 'direct_state_value',
                             'request_state_value', 'joint_state_value',
                             'gate_direct', 'gate_request', 'gate_joint', 'gate_single',
                             'gate_direct_content', 'gate_request_content',
                             'gate_joint_content', 'gate_single_content',
                             'local_value_gate', 'fused_output', 'fused_state_value',
                             'final_local_gate', 'final_global_gate'):
                    getattr(fusion, name).weight.normal_(std=.04)
            model.value_correction.weight.normal_(std=.04)

    def test_all_declared_channels_change_predictions(self):
        cases = {
            'triple_global_pure_local': ('global_off', 'local_off', 'condition_off',
                                         'global_direct_off', 'global_request_off',
                                         'global_joint_off', 'temporal_off'),
            'triple_global_joint_local': ('global_off', 'local_off', 'condition_off',
                                          'global_direct_off', 'global_request_off',
                                          'global_joint_off', 'local_joint_condition_off',
                                          'temporal_off'),
            'triple_global_full_local': ('global_off', 'local_off', 'condition_off',
                                         'global_direct_off', 'global_request_off',
                                         'global_joint_off', 'local_request_condition_off',
                                         'local_joint_condition_off', 'value_correction_off',
                                         'temporal_off'),
            'direct_plus_request_safe': ('global_off', 'local_off', 'condition_off',
                                         'global_direct_off', 'global_request_off',
                                         'temporal_off'),
            'gated_direct_request': ('global_off', 'local_off', 'condition_off',
                                      'global_direct_off', 'global_request_off',
                                      'temporal_off'),
            'gated_triple': ('global_off', 'local_off', 'condition_off',
                              'global_direct_off', 'global_request_off',
                              'global_joint_off', 'temporal_off'),
            'gated_union': ('global_off', 'local_off', 'condition_off',
                             'global_direct_off', 'global_request_off',
                             'global_joint_off', 'global_single_off',
                             'local_value_correction_off', 'temporal_off')}
        flags = {'global_off': 'global_enabled', 'local_off': 'local_enabled',
                 'condition_off': 'condition_enabled',
                 'global_direct_off': 'global_direct_enabled',
                 'global_request_off': 'global_request_enabled',
                 'global_joint_off': 'global_joint_enabled',
                 'global_single_off': 'global_single_enabled',
                 'local_value_correction_off': 'local_value_correction_enabled',
                 'local_request_condition_off': 'local_request_condition_enabled',
                 'local_joint_condition_off': 'local_joint_condition_enabled',
                 'value_correction_off': 'value_correction_enabled',
                 'temporal_off': 'temporal_enabled'}
        for mode, options in cases.items():
            model = self.model(mode)
            self.activate(model)
            normal = self.full(model)
            for option in options:
                changed = self.full(model, **{flags[option]: False})
                self.assertGreater(float((normal - changed).detach().abs().max()), 1e-7,
                                   (mode, option))

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
        zero = (fusion.output, fusion.condition_query, fusion.request_condition_query,
                fusion.joint_condition_query, fusion.time, fusion.direct_output,
                fusion.request_output, fusion.joint_output, fusion.direct_state_value,
                fusion.request_state_value, fusion.joint_state_value, model.value_correction)
        for layer in zero:
            self.assertEqual(float(layer.weight.detach().abs().max()), 0.)
        ids = [id(p) for group in model.optim_groups() for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})

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
                       groups, grid, model.wan.freqs, rope, context, 3,
                       model.value_correction)
        self.assertEqual(float(first[:, :2].detach().abs().max()), 0.)
        changed = context.clone()
        changed[:, 3:] += 100
        second = fusion(x, video_state, c, self.sample['times'], (3, 1, 2), native,
                        groups, grid, model.wan.freqs, rope, changed, 3,
                        model.value_correction)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_generation_and_alignment_gradient_ownership(self):
        model = self.model('triple_global_full_local')
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
                  model.value_correction.weight,
                  *[getattr(f, k).weight for k in
                    ('condition_query', 'request_condition_query', 'joint_condition_query',
                     'video_q', 'video_k', 'video_v', 'key', 'value', 'output',
                     'direct_output', 'request_output', 'joint_output',
                     'direct_state_value', 'request_state_value', 'joint_state_value')]]:
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
        model = self.model()
        optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
        f = model.fusions['1']
        grads = []
        for step in (0, 1, 2):
            model.set_phase(step)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                            self.noise, 0., 1)
            get = lambda layer: float(layer.weight.grad.norm())
            grads.append(dict(output=get(f.output), direct_output=get(f.direct_output),
                              request_output=get(f.request_output),
                              condition_query=get(f.condition_query),
                              direct_state_value=get(f.direct_state_value),
                              request_state_value=get(f.request_state_value),
                              video_q=get(f.video_q), key=get(f.key)))
            optimizer.step()
        for key in ('output', 'direct_output', 'request_output'):
            self.assertGreater(grads[0][key], 0., key)
        for key in ('condition_query', 'direct_state_value', 'request_state_value',
                    'video_q', 'key'):
            self.assertGreater(grads[1][key], 0., key)

    def test_gated_single_write_gradient_opening(self):
        for mode in ('gated_direct_request', 'gated_triple', 'gated_union'):
            model = self.model(mode)
            optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
            f = model.fusions['1']
            model.set_phase(0)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                            self.noise, 0., 1)
            first = dict(output=float(f.output.weight.grad.norm()),
                         fused_output=float(f.fused_output.weight.grad.norm()),
                         final_local=float(f.final_local_gate.weight.grad.norm()))
            optimizer.step()
            model.set_phase(1)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                            self.noise, 0., 1)
            second = dict(fused_state=float(f.fused_state_value.weight.grad.norm()),
                          gate_direct=float(f.gate_direct.weight.grad.norm()),
                          gate_request=float(f.gate_request.weight.grad.norm()),
                          final_local=float(f.final_local_gate.weight.grad.norm()),
                          final_global=float(f.final_global_gate.weight.grad.norm()))
            if mode in ('gated_triple', 'gated_union'):
                second['gate_joint'] = float(f.gate_joint.weight.grad.norm())
            if mode == 'gated_union':
                second['gate_single'] = float(f.gate_single.weight.grad.norm())
                second['gate_single_content'] = float(f.gate_single_content.weight.grad.norm())
            self.assertGreater(first['fused_output'], 0.)
            for key, value in second.items():
                self.assertGreater(value, 0., (mode, key))
            optimizer.step()
            if mode == 'gated_union':
                model.set_phase(2)
                optimizer.zero_grad(set_to_none=True)
                backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                                self.noise, 0., 1)
                self.assertGreater(float(f.local_value_gate.weight.grad.norm()), 0.)

    def test_triple_read_and_joint_read_gradient_opening(self):
        for mode, names in {
                'triple_global_joint_local': ('direct_output', 'joint_output',
                                              'joint_condition_query', 'joint_state_value'),
                'triple_global_full_local': ('request_output', 'joint_output',
                                             'request_condition_query',
                                             'joint_condition_query', 'request_state_value',
                                             'joint_state_value')}.items():
            model = self.model(mode)
            optimizer = torch.optim.AdamW(model.optim_groups(), foreach=False)
            f = model.fusions['1']
            model.set_phase(0)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                            self.noise, 0., 1)
            first = {n: float(getattr(f, n).weight.grad.norm()) for n in names}
            optimizer.step()
            model.set_phase(1)
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model, self.evidence, self.sample, self.text, self.sigma,
                            self.noise, 0., 1)
            second = {n: float(getattr(f, n).weight.grad.norm()) for n in names}
            for name in names:
                self.assertGreater(second[name], 0., (mode, name))
            if mode == 'triple_global_full_local':
                self.assertGreater(float(model.value_correction.weight.grad.norm()), 0.)

    def test_guidance_controls_and_checkpoint_resume(self):
        model = self.model()
        self.activate(model)
        c, groups = model.enhancement(self.evidence)
        noisy = (1 - self.sigma) * self.clean + self.sigma * self.noise
        null = self.text[:, :1] * 0
        for mode in CONTROL_MODES:
            actual = guided_velocity(model, noisy, self.sample['first'], self.text, null,
                                     self.sigma, c, c, self.sample['times'], groups, groups,
                                     composition=LEGACY, guidance=5., mode=mode)
            from guidance import execution_options
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
                        {'condition_enabled': False}, {'global_enabled': False},
                        {'local_enabled': False}, {'global_direct_enabled': False},
                        {'global_request_enabled': False}, {'temporal_enabled': False}):
            torch.testing.assert_close(self.full(model, **options),
                                       self.full(other, **options), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
