"""Small end-to-end checks for changed execution and training ownership."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_wan import FakeWan, W, FF, evidence
from physical_content.common import ROOT, load_config
from physical_content.model import PhysicalModel


def model_for(plan):
    cfg = load_config(ROOT / 'configs' / f'r19{plan.lower()}.yaml')
    cfg['model'].update(width=W, ffn_width=FF, injection_layers=[0],
                        q_tokens=8, q_heads=4, grounded_heads=2)
    cfg['train']['gradient_checkpointing'] = False
    interface = dict(mean=torch.zeros(4096), scale=torch.ones(4096),
                     weight=torch.randn(W, 4096)*.01, bias=torch.zeros(W),
                     target_scale=torch.tensor(1.))
    return PhysicalModel(cfg, device='cpu', wan=FakeWan(), interface=interface)


class ExecutionTest(unittest.TestCase):
    def test_state_conditioning_and_gradient_ownership(self):
        torch.set_num_threads(2)
        for plan in ('AT', 'AI'):
            torch.manual_seed(42)
            model = model_for(plan)
            branch = model.mechanisms['0']
            self.assertIsNot(branch.physical.query_projection.weight,
                             model.wan.blocks[0].cross_attn.q.weight)
            model.base_bank.assert_frozen()
            self.assertEqual(len(model.optim_groups()) > 0, True)
            text = torch.randn(1, 7, 4096)
            noisy = torch.randn(1, 48, 2, 4, 4)
            first = noisy[:, :, :1].clone() if plan == 'AI' else None
            args = (noisy, first, text, torch.tensor(.5))
            times = torch.tensor([[0., .5]])

            def predict(mode='normal'):
                c, g = model.prepare_content(evidence_value, text)
                return model.velocity(*args, c, g, times, physical_mode=mode)[0]

            evidence_value = evidence()
            off = predict('off')
            active = predict()
            torch.testing.assert_close(active, off, atol=0, rtol=0)
            active.square().mean().backward()
            self.assertGreater(float(branch.condition.weight.grad.norm()), 0.)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                branch.condition.weight.normal_(0., .05)
            changed = predict()
            self.assertGreater(float((changed - off).abs().max()), 0.)
            changed.square().mean().backward()
            for module in (branch.physical.query_projection, branch.physical.key,
                           branch.physical.value, model.enhancement.execution_project,
                           branch.executor_input, branch.executor_output):
                self.assertIsNotNone(module.weight.grad)
                self.assertGreater(float(module.weight.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for p in
                                model.wan.blocks[0].cross_attn.parameters()))
            u, z = torch.randn(1, 8, W), torch.randn(1, 8, W)
            mask = torch.ones(1, 8, 1); mask[:, :4] = 0
            delta = branch(u, z, mask)
            torch.testing.assert_close(delta[:, :4], torch.zeros_like(delta[:, :4]), atol=0, rtol=0)
            self.assertGreater(float((branch(u+1, z, mask)-delta).abs().max()), 0.)
            if plan == 'AI':
                model.set_phase(0)
                self.assertFalse(any(p.requires_grad for p in model.wan.parameters()))
                model.set_phase(128)
                self.assertTrue(any(p.requires_grad for p in model.wan.parameters()))
                model.base_bank.assert_frozen()


if __name__ == '__main__':
    unittest.main()
