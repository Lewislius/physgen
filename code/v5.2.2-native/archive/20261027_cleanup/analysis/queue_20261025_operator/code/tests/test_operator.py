"""Behavioral checks for the new operator, stage ownership and feedback credit.

Small differentiable stand-ins check implementation, not physical video quality.
"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_native import FakeWan, FakeKnowledge, FakeDecoder, sample, grad_size
from physgen_native.config import read_config, validate, phase_at
from physgen_native.conditioning import StateConditionedSite
from physgen_native.system import PhysGenNative
from physgen_native.runtime import parameter_groups, schedule, save_checkpoint, load_checkpoint, restore

torch.set_num_threads(2)


def config():
    cfg = read_config(ROOT/'configs/native_operator_wisa3857.yaml')
    cfg['model'].update(width=32, qwen_width=32, attention_width=16, attention_heads=4,
                        interface_rank=4, condition_every=1)
    cfg['knowledge'].update(image_size=[32,64], video_size=[32,64], lora_last_layers=2,
                            lora_rank=2, lora_alpha=2, reasoning_tokens=4)
    cfg['stages'] = dict(interface=1, calibration=1, joint=2)
    cfg['wan_lora'] = dict(rank=2, alpha=2)
    cfg['train'].update(accumulation=1, warmup=1, text_dropout=0., requirements_dropout=0.)
    cfg['sequence'].update(size=[8,12], scales=[1,2], lags=[1,4,8,16])
    cfg['verifier'].update(size=[8,12], feature_width=3, width=8, lags=[1,2,4,5])
    cfg['completion'].update(steps=3, every=1)
    cfg['inference'].update(steps=3)
    cfg['data']['latent_shape'] = [4,6,4,6]
    return validate(cfg)


class TextKnowledge(FakeKnowledge):
    def encode(self, first_rgb, caption, clauses, **kw):
        kw.pop('duration', None)
        result = super().encode(first_rgb, caption, clauses, **kw)
        sign = 1 if 'body' in caption else -1
        result['contextual'] = result['contextual']+sign*.2*torch.linspace(-1,1,32)
        return result


class Observer(nn.Module):
    def forward(self, video): return video[::4].permute(0,2,3,1)


def model_and_data():
    cfg = config()
    model = PhysGenNative(cfg, FakeWan(cfg), knowledge=TextKnowledge(), decoder=FakeDecoder(),
                         process_encoder=Observer(), device='cpu')
    data = sample()
    data['alternative'] = dict(caption='A different material spreads.', clauses=[])
    return model, data


class OperatorTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(31)

    def test_stage_contract_rejects_a_fifth_loss(self):
        cfg = config()
        self.assertEqual([phase_at(cfg, i) for i in range(4)], ['interface','calibration','joint','joint'])
        cfg['loss']['joint']['intensity'] = .1
        with self.assertRaises(ValueError): validate(cfg)

    def test_operator_acts_on_state_and_preserves_native_path_at_zero_C(self):
        model, data = model_and_data()
        interface = model.generator.interface
        x = torch.randn(1, 6*2*3, 32)
        zero = torch.zeros(1,4,32)
        actual, _ = interface(x, 0, interface.prepare(zero), torch.tensor(.5), data['times'], (6,2,3))
        torch.testing.assert_close(actual, x, atol=0, rtol=0)
        site = StateConditionedSite(32,4)
        message = torch.randn(1,3,32)
        q = torch.randn_like(message)
        a = site(q, message, torch.zeros(1,3,1), torch.tensor(.5))
        b = site(-q, message, torch.zeros(1,3,1), torch.tensor(.5))
        self.assertGreater(float(a.norm()), 0)
        self.assertGreater(float((a-b).norm()), 0)

    def test_text_intervention_preserves_observation_provenance_and_clause_count(self):
        model, data = model_and_data()
        data['clauses'][0]['basis'] = 'image'
        alternative = model._alternative_condition(data)
        self.assertIs(alternative['first'], data['first'])
        self.assertEqual(alternative['clauses'][0], data['clauses'][0])
        self.assertEqual(len(alternative['clauses']), len(data['clauses']))
        self.assertNotEqual(alternative['clauses'][1]['text'], data['clauses'][1]['text'])

    def test_three_stages_have_distinct_parameter_ownership(self):
        model, data = model_and_data()
        null = torch.zeros_like(data['text'])
        for step, expected in ((0, {'fm','process','use','intensity'}), (1, {'rank','matching'}), (2, {'fm','process','use'})):
            model.zero_grad(set_to_none=True)
            losses, weights, _ = model(data, null, step, sigma_override=.65)
            self.assertEqual(set(losses), expected)
            model.backward_losses(losses, weights)
            if step == 1:
                self.assertGreater(grad_size(model.critic), 0)
                self.assertEqual(grad_size(model.generator)+grad_size(model.conditioner)+grad_size(model.knowledge), 0)
            else:
                self.assertGreater(grad_size(model.conditioner), 0)
                self.assertGreater(grad_size(model.knowledge), 0)
                self.assertEqual(grad_size(model.critic), 0)
            if step == 2:
                self.assertEqual(grad_size(model.generator.interface.intensity), 0)
                self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.generator.wan.named_parameters()
                                       if n.endswith(('lora_a','lora_b')) and p.grad is not None), 0)

    def test_condition_use_loss_reaches_actual_C_operator(self):
        model, data = model_and_data()
        losses, _, metrics = model(data, torch.zeros_like(data['text']), 0, sigma_override=.65)
        losses['use'].backward()
        self.assertGreater(float(metrics['condition/output_rms']), 0)
        self.assertGreater(grad_size(model.conditioner), 0)
        self.assertGreater(grad_size(model.generator.interface), 0)
        self.assertGreater(grad_size(model.knowledge), 0)

    def test_whole_completion_derivative_matches_unrolled_autograd(self):
        model, data = model_and_data()
        model.set_phase(2)
        c = model.encode_condition(model.conditions(data)).detach().requires_grad_(True)
        noise = torch.randn_like(data['latent']); sigma = torch.tensor(.7)
        params = [p for p in model.generator.parameters() if p.requires_grad]
        direct = model.completion(data, c, sigma, noise, recompute=False)
        dg = torch.autograd.grad(direct.square().mean(), [c, *params], allow_unused=True)
        actual = model.completion(data, c, sigma, noise)
        ag = torch.autograd.grad(actual.square().mean(), [c, *params], allow_unused=True)
        torch.testing.assert_close(actual, direct, atol=1e-6, rtol=1e-5)
        for a, b in zip(ag, dg):
            if b is None: self.assertIsNone(a)
            else: torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-4)

    def test_verifier_feedback_reaches_C_without_updating_the_judge(self):
        model, data = model_and_data()
        losses, weights, metrics = model.feedback_losses(data, 2)
        self.assertEqual(set(losses), {'verifier'})
        self.assertEqual(metrics['full_video_frames'], 21)
        self.assertFalse(metrics['generated_from_pure_noise'])
        model.backward_losses(losses, weights)
        self.assertGreater(grad_size(model.conditioner), 0)
        self.assertGreater(grad_size(model.knowledge), 0)
        self.assertGreater(grad_size(model.generator), 0)
        self.assertEqual(grad_size(model.critic), 0)

    def test_resume_preserves_three_stage_optimizer_and_future_free_sampling(self):
        model, data = model_and_data()
        cfg = model.cfg
        optimizer = torch.optim.AdamW(parameter_groups(model, cfg))
        scheduler = schedule(optimizer, cfg)
        seen = []
        for step in range(4):
            seen.append({g['name']:g['lr'] for g in optimizer.param_groups})
            optimizer.step(); scheduler.step()
        self.assertEqual(seen[0]['critic'], 0)
        self.assertGreater(seen[1]['critic'], 0)
        self.assertEqual(seen[1]['condition'], 0)
        self.assertGreater(seen[2]['condition'], 0)
        self.assertGreater(seen[2]['wan_lora'], 0)
        model.set_phase(2)
        condition = model.conditions(data)
        self.assertNotIn('latent', condition)
        self.assertNotIn('alternative', condition)
        with torch.no_grad(): before = model.sample(condition, data['text'], torch.zeros_like(data['text']), seed=11)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'state.pt'
            save_checkpoint(model, optimizer, scheduler, cfg, 2, 17, path)
            state = load_checkpoint(path)
            torch.manual_seed(31)
            fresh, _ = model_and_data()
            opt = torch.optim.AdamW(parameter_groups(fresh, cfg)); sched = schedule(opt, cfg)
            self.assertEqual(restore(fresh, opt, sched, state, cfg), (2,17))
            with torch.no_grad(): after = fresh.sample(condition, data['text'], torch.zeros_like(data['text']), seed=11)
        torch.testing.assert_close(before, after, atol=0, rtol=0)


if __name__ == '__main__': unittest.main()
