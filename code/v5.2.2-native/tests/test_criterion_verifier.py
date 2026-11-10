"""Four end-to-end recipes with native tiny Qwen and the native Wan reader."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_message_conditioning import message_model
from test_native import FakeWan, FakeDecoder, grad_size
from test_fullwidth_attention import NativeBlock
from physgen_native.config import read_config, validate, VERIFIER_MESSAGE_VARIANTS, phase_at
from physgen_native.knowledge import QwenKnowledge
from physgen_native.system import PhysGenNative
from physgen_native.lora import verifier_role, role_checkpoint
from physgen_native.criterion_verifier import (CriterionVerifier, LABEL_ORDERS, CRITERIA, canonical_logits,
                                             criterion_prompt)
from physgen_native.runtime import (owner, parameter_groups, schedule, save_checkpoint, load_checkpoint,
                                    restore, initialize_generator)

torch.set_num_threads(2)


def criterion_model(query=False, lora=False, *, observers=True):
    from transformers import Qwen3VLForConditionalGeneration
    original, data = message_model(query)
    cfg = copy.deepcopy(original.cfg)
    mode = cfg['knowledge']['reasoning_mode']
    cfg['revision'] = next(rev for rev, pair in VERIFIER_MESSAGE_VARIANTS.items() if pair == (mode, lora))
    production = read_config(ROOT/'configs'/f'native_{mode}_verifier_{"lora" if lora else "frozen"}_wisa3857.yaml')
    cfg['verifier'] = copy.deepcopy(production['verifier'])
    cfg['verifier'].update(size=[32, 64], frames=3, observation_tokens=8,
                           teacher_min_confidence=.4, teacher_max_order_gap=.2)
    cfg['train']['verifier_lr'] = .001 if lora else 0.
    if lora:
        cfg['verifier']['adapter'].update(rank=2, alpha=2, last_layers=1)
        cfg['stages'] = dict(interface=1, calibration=1, joint=2)
        cfg['loss']['calibration'] = dict(judge=1.)
    validate(cfg)
    knowledge = QwenKnowledge(cfg, 'cpu', backbone=Qwen3VLForConditionalGeneration(original.knowledge.backbone.config),
                              tokenizer=original.knowledge.tokenizer)
    wan = FakeWan(cfg); wan.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
    model = PhysGenNative(cfg, wan, knowledge=knowledge, decoder=FakeDecoder(), device='cpu', training_observers=observers)
    return model, data


def teacher_labels(*args, **kwargs):
    # Synthetic supervised fixture only; production never assumes this label.
    logits = torch.tensor([2., -1., 0.]).view(1, 1, 1, 3).expand(1, 4, 2, 3).clone()
    return logits, dict(observation='A visible source continues.', observation_complete=True, supervision='test fixture')


class CriterionVerifierTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(261030)

    def test_four_configs_enforce_architecture_and_role(self):
        for rev, (mode, lora) in VERIFIER_MESSAGE_VARIANTS.items():
            cfg = read_config(ROOT/'configs'/f'native_{mode}_verifier_{"lora" if lora else "frozen"}_wisa3857.yaml')
            self.assertEqual(cfg['revision'], rev)
            self.assertEqual(cfg['stages'], dict(interface=64, calibration=32, joint=160) if lora else dict(interface=64, joint=160))
            self.assertNotIn('lora_rank', cfg['knowledge'])
            self.assertEqual(cfg['model']['attention_width'], 3072)
            bad = copy.deepcopy(cfg); bad['verifier']['adapter']['enabled'] = not lora
            with self.assertRaises(ValueError): validate(bad)
            bad = copy.deepcopy(cfg); bad['knowledge']['lora_rank'] = 16
            with self.assertRaises(ValueError): validate(bad)
            if mode.startswith('q_'):
                bad = copy.deepcopy(cfg); bad['knowledge']['relation_tokens'] = 64
                with self.assertRaisesRegex(ValueError, '128'): validate(bad)

    def test_canonical_labels_unknown_and_reliability(self):
        cfg = dict(temperature=1., teacher_min_confidence=.5, teacher_max_order_gap=.15)
        critic = CriterionVerifier(cfg)
        semantic = torch.tensor([3., -2., -1.])
        for order in LABEL_ORDERS:
            actual = canonical_logits(semantic[list(order)], order)
            torch.testing.assert_close(actual, semantic)
        def logits(values): return torch.tensor(values).view(1,1,1,3).expand(1,4,2,3).clone()
        good, unknown, bad = (logits(x) for x in ([20.,-20.,-20.], [-20.,-20.,20.], [-20.,20.,-20.]))
        self.assertAlmostEqual(float(critic(good)), 1.)
        self.assertAlmostEqual(float(critic(unknown)), 0.)
        self.assertAlmostEqual(float(critic(bad)), -1.)
        student = logits([.1, .2, .3]).requires_grad_()
        loss, metrics = critic.calibration_loss(student, good)
        loss.backward(); self.assertGreater(float(student.grad.norm()), 0)
        rejected = good.clone(); rejected[..., 1, :] = bad[..., 1, :]
        student.grad = None
        loss, metrics = critic.calibration_loss(student, rejected)
        loss.backward(); self.assertIsNone(student.grad)
        self.assertEqual(int(metrics['teacher_accepted']), 0)
        loss, metrics = critic.calibration_loss(student, good, observation_complete=False)
        self.assertEqual(int(metrics['teacher_accepted']), 0)
        prompt = criterion_prompt('original requirement', CRITERIA[1][1], LABEL_ORDERS[1])
        self.assertIn('A = insufficient', prompt)
        self.assertNotIn('H_content', prompt)

    def test_teacher_observes_without_requirements_and_never_enables_adapter(self):
        model, data = criterion_model(lora=True)
        frames = torch.rand(3, 3, 32, 64)
        captured = []
        original = model.knowledge._inputs
        def inputs(clips, body, *args, **kwargs):
            captured.append(body)
            return original(clips, body, *args, **kwargs)
        with patch.object(model.knowledge, '_inputs', side_effect=inputs), \
             patch.object(model.knowledge, '_reason', return_value=('A source is visible.', True)) as reason, \
             patch.object(model.knowledge, 'criterion_logits', return_value=teacher_labels()[0]) as judge:
            result, trace = model.knowledge.criterion_teacher(frames, torch.arange(3.), 'UNIQUE REQUEST', [])
        self.assertNotIn('UNIQUE REQUEST', captured[0])
        self.assertFalse(reason.call_args.kwargs['adapt'])
        self.assertFalse(judge.call_args.kwargs['adapt'])
        self.assertEqual(judge.call_args.kwargs['observation'], 'A source is visible.')
        self.assertTrue(trace['observation_complete'])

    def test_native_teacher_generation_is_fixed_after_verifier_changes(self):
        model, data = criterion_model(lora=True)
        frames, times = torch.rand(3,3,32,64), torch.arange(3.)
        before, trace = model.knowledge.criterion_teacher(frames, times, data['caption'], [])
        with torch.no_grad():
            for name, parameter in model.knowledge.named_parameters():
                if name.endswith('verifier_lora_b'): parameter.normal_(std=.1)
        after, other = model.knowledge.criterion_teacher(frames, times, data['caption'], [])
        self.assertEqual(trace, other)
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        self.assertFalse(after.requires_grad)

    def test_native_logits_adapter_updates_do_not_change_content_and_keep_rgb_derivative(self):
        model, data = criterion_model(lora=True)
        k = model.knowledge
        image = torch.rand(1,3,32,64)
        before = k.encode(image, data['caption'], [], duration=1.)['memory'].clone()
        frames = torch.rand(3,3,32,64, requires_grad=True); times = torch.arange(3.)
        base = k.criterion_logits(frames, times, data['caption'], [], adapt=False)
        model.set_phase(1)
        active = k.criterion_logits(frames, times, data['caption'], [])
        torch.testing.assert_close(active, base)
        self.assertEqual(active.shape, (1,4,2,3))
        target, _ = teacher_labels()
        loss, _ = model.critic.calibration_loss(active, target)
        loss.backward()
        self.assertGreater(float(frames.grad.norm()), 0)
        params = [p for n,p in model.named_parameters() if owner(n) == 'verifier']
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()) for p in params))
        self.assertTrue(all(p.grad is None for n,p in k.named_parameters() if owner('knowledge.'+n) != 'verifier'))
        with torch.no_grad():
            for p in params:
                if p.grad is not None: p.add_(p.grad, alpha=-.1)
        with verifier_role(True):
            after = k.encode(image, data['caption'], [], duration=1.)['memory']
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        torch.testing.assert_close(base, k.criterion_logits(frames, times, data['caption'], [], adapt=False), rtol=0, atol=0)
        self.assertFalse(torch.equal(active, k.criterion_logits(frames, times, data['caption'], [])))

    def test_four_training_pipelines_and_gradient_ownership(self):
        for query in (False, True):
            for lora in (False, True):
                with self.subTest(query=query, lora=lora):
                    model, data = criterion_model(query, lora)
                    model.train()
                    groups = parameter_groups(model, model.cfg)
                    self.assertEqual(any(g['name'] == 'verifier' for g in groups), lora)
                    self.assertFalse(any(g['name'] in ('knowledge', 'critic') for g in groups))
                    optimizer = torch.optim.AdamW(groups)
                    losses, weights, _ = model(data, torch.zeros_like(data['text']), 0, sigma_override=.2)
                    model.backward_losses(losses, weights)
                    self.assertGreater(grad_size(model.conditioner), 0)
                    self.assertEqual(grad_size(model.knowledge), 0)
                    self.assertEqual(grad_size(model.generator.wan), 0)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if lora:
                        model.set_phase(1)
                        for micro in (0, 1):
                            with patch.object(model.knowledge, 'criterion_teacher', side_effect=teacher_labels):
                                losses, weights, metrics = model.qwen_calibration(data, 1, micro, capture=True)
                            model.backward_losses(losses, weights)
                            self.assertGreater(grad_size(model.knowledge), 0)
                            self.assertEqual(grad_size(model.conditioner), 0)
                            self.assertEqual(grad_size(model.generator.wan), 0)
                            self.assertEqual(metrics['generated_candidate'], (model.cfg['train']['accumulation']+micro) % 2)
                            self.assertEqual(model.last_verifier_comparison['metadata']['accepted'], [[True]*4])
                            optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    step = 2 if lora else 1
                    model.set_phase(step)
                    self.assertFalse(any(p.requires_grad for p in model.knowledge.parameters()))
                    losses, weights, metrics = model.feedback_losses(data, step)
                    model.backward_losses(losses, weights)
                    self.assertGreater(metrics['C_gradient_rms'], 0)
                    self.assertGreater(grad_size(model.conditioner.block), 0)
                    self.assertEqual(grad_size(model.knowledge), 0)
                    self.assertEqual(grad_size(model.generator.wan), 0)
                    self.assertIn('verifier/persistence/unknown', metrics)
                    if query: self.assertGreater(float(model.conditioner.query_seed.grad.norm()), 0)

    def test_checkpointed_verifier_gradient_matches_plain_with_role_changes(self):
        model, data = criterion_model(lora=True)
        model.set_phase(1); k = model.knowledge
        with torch.no_grad():
            for n,p in k.named_parameters():
                if n.endswith('verifier_lora_b'): p.normal_(std=.03)
        frames = torch.rand(3,3,32,64, requires_grad=True); times = torch.arange(3.)
        actual = k.criterion_logits(frames, times, data['caption'], [])
        with verifier_role(False): actual.square().sum().backward()
        grads = {n:p.grad.clone() for n,p in k.named_parameters() if p.grad is not None}
        rgb_grad = frames.grad.clone(); k.zero_grad(set_to_none=True); frames.grad = None
        k.verifier_cfg['gradient_checkpointing'] = False
        plain = k.criterion_logits(frames, times, data['caption'], [])
        plain.square().sum().backward()
        torch.testing.assert_close(actual, plain, rtol=0, atol=0)
        torch.testing.assert_close(rgb_grad, frames.grad, rtol=2e-4, atol=2e-5)
        for n,p in k.named_parameters():
            if n in grads: torch.testing.assert_close(grads[n], p.grad, rtol=2e-4, atol=2e-5)

    def test_resume_inference_and_compatible_message_transfer(self):
        for query in (False, True):
            model, data = criterion_model(query, True)
            optimizer = torch.optim.AdamW(parameter_groups(model, model.cfg)); scheduler = schedule(optimizer, model.cfg)
            # Cross the interface boundary, then save actual LoRA optimizer moments.
            optimizer.step(); scheduler.step(); model.set_phase(1)
            with patch.object(model.knowledge, 'criterion_teacher', side_effect=teacher_labels):
                losses, weights, _ = model.qwen_calibration(data, 1, 0)
            model.backward_losses(losses, weights); optimizer.step(); scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            self.assertGreater(len(optimizer.state), 0)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'lora.pt'
                save_checkpoint(model, optimizer, scheduler, model.cfg, 2, 3, path)
                state = load_checkpoint(path)
                self.assertTrue(any(owner(n) == 'verifier' for n in state['model']))
                clone, _ = criterion_model(query, True, observers=False)
                restore(clone, None, None, state, clone.cfg)
                self.assertEqual(clone.verifier_state, model.verifier_state)
                resumed, _ = criterion_model(query, True)
                opt = torch.optim.AdamW(parameter_groups(resumed, resumed.cfg)); sch = schedule(opt, resumed.cfg)
                self.assertEqual(restore(resumed, opt, sch, state, resumed.cfg), (2, 3))
                self.assertEqual(resumed.verifier_state['accepted'], model.verifier_state['accepted'])
                self.assertEqual(len(opt.state), len(optimizer.state))
                old_state = next(iter(optimizer.state.values()))
                new_state = next(iter(opt.state.values()))
                torch.testing.assert_close(old_state['exp_avg'], new_state['exp_avg'])
                frozen, _ = criterion_model(query, False)
                report = initialize_generator(frozen, frozen.cfg, path)
                self.assertTrue(report['message_migration'])
                for (n,p), (other,q) in zip(model.conditioner.named_parameters(), frozen.conditioner.named_parameters()):
                    self.assertEqual(n, other); torch.testing.assert_close(p, q)
                self.assertFalse(any(owner(n) == 'verifier' for n,p in frozen.named_parameters()))
                # The reverse transfer initializes a NEW verifier rather than importing a judge.
                fp = Path(directory)/'frozen.pt'
                fo = torch.optim.AdamW(parameter_groups(frozen, frozen.cfg)); fs = schedule(fo, frozen.cfg)
                save_checkpoint(frozen, fo, fs, frozen.cfg, 1, 1, fp)
                fresh, _ = criterion_model(query, True)
                initialize_generator(fresh, fresh.cfg, fp)
                self.assertEqual(fresh.verifier_state['accepted'], 0)
                with self.assertRaisesRegex(RuntimeError, 'accepted'): fresh.require_calibrated_verifier()

    def test_sampling_all_variants_and_online_feedback(self):
        for query in (False, True):
            for lora in (False, True):
                model, data = criterion_model(query, lora)
                model.verifier_state['accepted'] = 1 if lora else 0
                model.eval().requires_grad_(False)
                with torch.no_grad():
                    plain = model.sample(model.conditions(data), data['text'], torch.zeros_like(data['text']), seed=17)
                model.cfg['inference']['latent_feedback'].update(enabled=True, sigmas=[.6], step_size=.01)
                with torch.no_grad():
                    updated = model.sample(model.conditions(data), data['text'], torch.zeros_like(data['text']), seed=17)
                self.assertTrue(bool(torch.isfinite(updated).all()))
                torch.testing.assert_close(updated[:, :, :1], data['first'])
                self.assertFalse(torch.equal(plain, updated))
                self.assertEqual(grad_size(model), 0)


if __name__ == '__main__': unittest.main()
