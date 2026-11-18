"""Small end-to-end checks for changed execution and training ownership."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_wan import FakeWan, W, FF, evidence
from physical_content.common import ROOT, load_config
from physical_content.model import PhysicalModel


def model_for(plan='DI'):
    cfg = load_config(ROOT / 'configs' / f'r19{plan.lower()}.yaml')
    cfg['model'].update(width=W, ffn_width=FF, injection_layers=[0],
                        q_tokens=8, q_heads=4, grounded_heads=2, qwen_width=16)
    cfg['train']['gradient_checkpointing'] = False
    interface = dict(mean=torch.zeros(16), scale=torch.ones(16),
                     weight=torch.randn(W, 16)*.01, bias=torch.zeros(W),
                     target_scale=torch.tensor(1.))
    return PhysicalModel(cfg, device='cpu', wan=FakeWan(), interface=interface)


class ExecutionTest(unittest.TestCase):
    def test_state_conditioning_and_gradient_ownership(self):
        torch.set_num_threads(2)
        for plan in ('DI',):
            torch.manual_seed(42)
            model = model_for(plan)
            branch = model.mechanisms['0']
            self.assertIsNot(branch.physical.query_projection.weight,
                             model.wan.blocks[0].cross_attn.q.weight)
            model.base_bank.assert_frozen()
            self.assertEqual(len(model.optim_groups()) > 0, True)
            text = torch.randn(1, 7, 4096)
            noisy = torch.randn(1, 48, 2, 4, 4)
            first = noisy[:, :, :1].clone()
            args = (noisy, first, text, torch.tensor(.5))
            times = torch.tensor([[0., .5]])

            def predict(mode='normal'):
                c, g = model.prepare_content(evidence_value, text)
                return model.velocity(*args, c, g, times, physical_mode=mode)[0]

            evidence_value = evidence(width=16)
            off = predict('off')
            active = predict()
            torch.testing.assert_close(active, off, atol=0, rtol=0)
            active.square().mean().backward()
            self.assertGreater(float(branch.condition.weight.grad.norm()), 0.)
            self.assertGreater(float(branch.relations.condition_q.weight.grad.norm()), 0.)
            self.assertGreater(float(branch.relations.condition_k.weight.grad.norm()), 0.)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                branch.condition.weight.normal_(0., .05)
                branch.relations.condition_q.weight.normal_(0., .05)
                branch.relations.condition_k.weight.normal_(0., .05)
            changed = predict()
            self.assertGreater(float((changed - off).abs().max()), 0.)
            changed.square().mean().backward()
            for module in (branch.physical.query_projection, branch.physical.key,
                           branch.physical.value, model.enhancement.execution_project,
                           branch.executor_input, branch.executor_output,
                           branch.relations.q, branch.relations.k,
                           branch.relations.v, branch.relations.o):
                self.assertIsNotNone(module.weight.grad)
                self.assertGreater(float(module.weight.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for p in
                                model.wan.blocks[0].cross_attn.parameters()))
            u, z = torch.randn(1, 8, W), torch.randn(1, 8, W)
            mask = torch.ones(1, 8, 1); mask[:, :4] = 0
            delta = branch(u, z, mask)
            torch.testing.assert_close(delta[:, :4], torch.zeros_like(delta[:, :4]), atol=0, rtol=0)
            self.assertGreater(float((branch(u+1, z, mask)-delta).abs().max()), 0.)
            if plan == 'DI':
                model.set_phase(0)
                self.assertFalse(any(p.requires_grad for p in model.wan.parameters()))
                model.set_phase(128)
                self.assertTrue(any(p.requires_grad for p in model.wan.parameters()))
                model.base_bank.assert_frozen()

    def test_relation_uses_video_values_and_zero_condition_has_no_effect(self):
        """Reject an appearance write hidden inside the relation operator."""
        from unittest.mock import patch
        torch.set_num_threads(2)
        torch.manual_seed(7)
        model = model_for()
        branch = model.mechanisms['0'].relations
        with torch.no_grad():
            branch.condition_q.weight.normal_(0., .3)
            branch.condition_k.weight.normal_(0., .3)
        h, c = torch.randn(1, 8, W), torch.randn(1, 8, W)
        mask = torch.ones(1, 8, 1); mask[:, :4] = 0
        parts = tuple(torch.randn(1, 8, W) for _ in range(6))
        args = (mask, parts, torch.tensor([8]), torch.tensor([[2, 2, 2]]), None)
        update = branch(h, c, *args)
        self.assertGreater(float(update[:, 4:].norm()), 0.)
        torch.testing.assert_close(update[:, :4], torch.zeros_like(update[:, :4]), atol=0, rtol=0)
        torch.testing.assert_close(branch(h, torch.zeros_like(c), *args),
                                   torch.zeros_like(update), atol=0, rtol=0)
        # Same value at every video position: changing relations cannot create
        # a new appearance vector from C, even with learned nonzero conditions.
        with patch.object(branch.v, 'forward', side_effect=lambda x: torch.ones_like(x)):
            torch.testing.assert_close(branch(h, c, *args), torch.zeros_like(update),
                                       atol=1e-6, rtol=0)
        self.assertIsNone(branch.v.bias)
        self.assertIsNone(branch.o.bias)

    def test_both_executors_have_independent_functional_contributions(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)
        model = model_for()
        branch = model.mechanisms['0']
        with torch.no_grad():
            for module in (branch.condition, branch.relations.condition_q, branch.relations.condition_k):
                module.weight.normal_(0., .1)
        text=torch.randn(1, 7, 4096); noisy=torch.randn(1, 48, 2, 4, 4)
        c,g=model.prepare_content(evidence(width=16), text)
        args=(noisy,noisy[:,:,:1].clone(),text,torch.tensor(.5),c,g,torch.tensor([[0.,.5]]))
        normal=model.velocity(*args)[0]
        for mode in ('relations_off','ffn_off','off'):
            changed=model.velocity(*args,physical_mode=mode)[0]
            self.assertGreater(float((normal-changed).norm()),0.)

    def test_physics_executes_native_video_features_and_native_text_reads_result(self):
        """Avoid executing a pretrained SA on an unrelated post-text domain."""
        torch.set_num_threads(2)
        torch.manual_seed(23)
        model=model_for();branch=model.mechanisms['0'];block=model.wan.blocks[0]
        with torch.no_grad():
            branch.relations.condition_q.weight.normal_(0., .2)
            branch.relations.condition_k.weight.normal_(0., .2)
        seen={}
        def record(key):
            def hook(module,args):
                seen.setdefault(key,[]).append(args[0].detach().clone())
            return hook
        handles=[block.self_attn.register_forward_pre_hook(record('native_video')),
                 branch.relations.q.register_forward_pre_hook(record('physical_video')),
                 block.cross_attn.register_forward_pre_hook(record('final_text'))]
        text=torch.randn(1,7,4096);noisy=torch.randn(1,48,2,4,4)
        c,g=model.prepare_content(evidence(width=16),text)
        args=(noisy,noisy[:,:,:1].clone(),text,torch.tensor(.5),c,g,torch.tensor([[0.,.5]]))
        try:
            model.velocity(*args)
            model.velocity(*args,physical_mode='relations_off')
        finally:
            for handle in handles:handle.remove()
        torch.testing.assert_close(seen['native_video'][0],seen['physical_video'][0],atol=0,rtol=0)
        self.assertEqual(len(seen['final_text']),2)
        self.assertGreater(float((seen['final_text'][0]-seen['final_text'][1]).norm()),0.)


if __name__ == '__main__':
    unittest.main()
