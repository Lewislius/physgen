"""Small end-to-end checks for changed execution and training ownership."""
import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_wan import FakeWan, W, FF, evidence
from physical_content.common import ROOT, load_config
from physical_content.model import PhysicalModel


def model_for(task):
    cfg = load_config(ROOT / 'configs' / 'r19bi.yaml')
    cfg['task'] = task
    cfg['model'].update(width=W, ffn_width=FF, injection_layers=[0],
                        qwen_width=16, q_tokens=8, q_heads=4, grounded_heads=2)
    cfg['train']['gradient_checkpointing'] = False
    interface = dict(mean=torch.zeros(16), scale=torch.ones(16),
                     weight=torch.randn(W, 16)*.01, bias=torch.zeros(W),
                     target_scale=torch.tensor(1.))
    return PhysicalModel(cfg, device='cpu', wan=FakeWan(), interface=interface)


class ExecutionTest(unittest.TestCase):
    def test_state_conditioning_and_gradient_ownership(self):
        torch.set_num_threads(2)
        for task in ('t2v', 'i2v'):
            torch.manual_seed(42)
            model = model_for(task)
            branch = model.mechanisms['0']
            self.assertIsNot(branch.physical.query_projection.weight,
                             model.wan.blocks[0].cross_attn.q.weight)
            model.base_bank.assert_frozen()
            self.assertIsInstance(branch.official_attention.q, torch.nn.Linear)
            self.assertIs(branch.official_attention.q,
                          model.wan.blocks[0].self_attn.q.base)
            self.assertIsNot(branch.video_attention.q.weight,
                             branch.official_attention.q.weight)
            self.assertEqual(len(model.optim_groups()) > 0, True)
            text = torch.randn(1, 7, 4096)
            noisy = torch.randn(1, 48, 2, 4, 4)
            first = noisy[:, :, :1].clone() if task == 'i2v' else None
            args = (noisy, first, text, torch.tensor(.5))
            times = torch.tensor([[0., .5]])

            def predict(mode='normal'):
                c, g = model.prepare_content(evidence_value, text)
                return model.velocity(*args, c, g, times, physical_mode=mode)[0]

            evidence_value = evidence()
            evidence_value['words'] = evidence_value['words'][:, :16]
            evidence_value['request_words'] = evidence_value['request_words'][:, :16]
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
                           branch.executor_input, branch.executor_output,
                           branch.video_attention.q, branch.video_attention.k,
                           branch.video_attention.v, branch.video_attention.o):
                self.assertIsNotNone(module.weight.grad)
                self.assertGreater(float(module.weight.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for p in
                                model.wan.blocks[0].cross_attn.parameters()))
            u, z = torch.randn(1, 8, W), torch.zeros(1, 8, W)
            z[:, -1] = torch.randn(1, W)
            mask = torch.ones(1, 8, 1); mask[:, :4] = 0
            parts = tuple(torch.randn(1, 8, W)*.2 for _ in range(6))
            arguments = (parts, torch.tensor([8]), torch.tensor([[2,2,2]]),
                         None, torch.randn(1, 8, W))
            delta = branch(u, z, mask, *arguments)
            torch.testing.assert_close(delta[:, :4], torch.zeros_like(delta[:, :4]), atol=0, rtol=0)
            # A condition at one video token must affect other valid tokens.
            # The direct-write control cannot do this by itself.
            direct = branch(u, z, mask, *arguments, mode='direct_write_only')
            self.assertEqual(float(direct[:, 4:7].abs().max()), 0.)
            self.assertGreater(float(delta[:, 4:7].abs().max()), 1e-8)
            if task == 'i2v':
                model.set_phase(0)
                self.assertFalse(any(p.requires_grad for p in model.wan.parameters()))
                model.set_phase(128)
                self.assertTrue(any(p.requires_grad for p in model.wan.parameters()))
                model.base_bank.assert_frozen()
            # The real training path checkpoints blocks during joint updates.
            model.zero_grad(set_to_none=True)
            model.checkpointing = True
            predict().square().mean().backward()
            self.assertGreater(float(branch.video_attention.q.weight.grad.norm()), 0.)


if __name__ == '__main__':
    unittest.main()
