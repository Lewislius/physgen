"""Check the three causal-response design changes without a GPU training sweep."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator import config, TextKnowledge, Observer
from test_native import FakeWan, FakeDecoder, sample, grad_size
from test_reasoning_feedback import native_knowledge
from physgen_native.config import RESPONSE_REVISION, read_config, validate
from physgen_native.system import PhysGenNative
from physgen_native.conditional_verifier import ConditionalProcessVerifier
from physgen_native.latent_reasoning import refine_dependency_latent

torch.set_num_threads(2)


def response_model():
    cfg = config()
    cfg['revision'] = RESPONSE_REVISION
    cfg['knowledge']['latent_reasoning_attention'] = 'joint'
    cfg['verifier']['response_only'] = True
    cfg['completion']['feedback_to'] = 'latent'
    cfg['condition_use']['mode'] = 'positive_only'
    cfg['train'].update(native_text_only_dropout=1., branch_checkpointing=False)
    for phase in ('interface', 'joint'): cfg['loss'][phase].pop('use')
    model = PhysGenNative(validate(cfg), FakeWan(cfg), knowledge=TextKnowledge(),
                         decoder=FakeDecoder(), process_encoder=Observer(), device='cpu')
    return model, sample()


class CausalResponseTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(2026)

    def test_later_working_evidence_can_revise_earlier_C_without_changing_memory(self):
        _, knowledge = native_knowledge()
        language = knowledge.backbone.model.language_model
        memory = torch.randn(1, 8, 32)
        initial = torch.randn(1, 4, 32, requires_grad=True)
        positions = torch.arange(8)[None, None].expand(3, 1, 8)
        mask = torch.ones(1, 8, dtype=torch.long)
        probe = torch.randn(32)
        prefix_outputs = []
        def capture(module, args, output):
            value = output[0] if isinstance(output, tuple) else output
            prefix_outputs.append(value[:, :8].detach().clone())
        hook = language.layers[-1].register_forward_hook(capture)
        gradients = {}
        for attention in ('causal', 'joint'):
            result = refine_dependency_latent(language, memory, initial, positions, mask,
                                              steps=1, depth=2, attention=attention)
            gradients[attention] = torch.autograd.grad((result[:, 0]*probe).sum(), initial)[0][:, -1]
        hook.remove()
        torch.testing.assert_close(gradients['causal'], torch.zeros_like(gradients['causal']), atol=0, rtol=0)
        self.assertGreater(float(gradients['joint'].norm()), 1e-6)
        torch.testing.assert_close(prefix_outputs[0], prefix_outputs[1], atol=0, rtol=0)

    def test_C_keeps_the_request_when_native_text_is_absent_and_only_one_branch_runs(self):
        model, data = response_model()
        null = torch.zeros_like(data['text'])
        calls = []
        handle = model.generator.register_forward_pre_hook(lambda m, a: calls.append(a[2].detach().clone()))
        with patch.object(model.knowledge, 'encode', wraps=model.knowledge.encode) as encode:
            losses, weights, metrics = model(data, null, 0, sigma_override=.65)
            self.assertEqual(encode.call_args.args[1], data['caption'])
        handle.remove()
        self.assertEqual(set(losses), {'fm', 'process', 'intensity'})
        self.assertEqual(metrics['native_text_dropped'], 1)
        self.assertEqual(metrics['C_text_dropped'], 0)
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0], null)
        model.backward_losses(losses, weights)
        self.assertGreater(grad_size(model.conditioner), 0)
        self.assertGreater(grad_size(model.knowledge), 0)

    def test_generator_objective_cannot_improve_by_changing_a_wrong_condition(self):
        model, data = response_model()
        null = torch.zeros_like(data['text'])
        torch.manual_seed(5)
        before, _, _ = model(data, null, 0, sigma_override=.65)
        data['alternative'] = dict(caption='An arbitrary incompatible request.', clauses=[])
        torch.manual_seed(5)
        after, _, _ = model(data, null, 0, sigma_override=.65)
        for name in before: torch.testing.assert_close(before[name], after[name], atol=0, rtol=0)

    def test_verifier_reads_responses_not_constant_appearance_offsets(self):
        critic = ConditionalProcessVerifier(8, 16, 12, (1, 2, 5), response_only=True)
        features = torch.randn(6, 2, 3, 8)
        condition = torch.randn(1, 4, 16)
        times, valid = torch.linspace(0, 3, 6), torch.ones(2, 3)
        original = critic(features, condition, times, valid)
        shifted = critic(features+torch.randn(1, 2, 3, 8), condition, times, valid)
        for lag in original: torch.testing.assert_close(original[lag], shifted[lag], atol=2e-6, rtol=2e-5)
        stationary = torch.randn(1, 2, 3, 8).expand(6, -1, -1, -1).clone().requires_grad_()
        absent = critic(stationary, condition, times, valid)
        for value in absent.values(): torch.testing.assert_close(value, torch.zeros_like(value), atol=0, rtol=0)
        derivative, = torch.autograd.grad(sum(s.sum() for s in absent.values()), stationary)
        self.assertTrue(torch.isfinite(derivative).all())

    def test_late_response_gradient_is_preserved_with_a_frozen_verifier(self):
        critic = ConditionalProcessVerifier(8, 16, 12, (5,), response_only=True)
        features = torch.randn(6, 2, 3, 8, requires_grad=True)
        loss = critic.generation_loss(features, torch.randn(1, 4, 16), torch.linspace(0, 3, 6), torch.ones(2, 3))
        loss.backward()
        self.assertGreater(float(features.grad[-1].norm()), 0)
        self.assertGreater(float(features.grad[0].norm()), 0)
        self.assertTrue(all(p.grad is None for p in critic.parameters()))

    def test_new_recipe_has_three_two_three_losses_and_preserves_latent_feedback(self):
        cfg = read_config(ROOT/'configs/native_causal_response_wisa3857.yaml')
        self.assertEqual([len(x) for x in cfg['loss'].values()], [3, 2, 3])
        self.assertEqual(cfg['completion']['feedback_to'], 'latent')
        self.assertEqual(cfg['knowledge']['latent_reasoning_steps'], 2)
        self.assertEqual(cfg['data']['frames'], 81)


if __name__ == '__main__': unittest.main()
