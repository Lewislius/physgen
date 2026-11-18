"""Functional checks for an unsubmitted architecture prototype, not quality tests."""
import unittest
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from physical_blocks import InteractionExpert, NativeFeatureGate, SourceContent, SourceGroundedRead


class SourceContentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.model = SourceContent(32, 24, 4, 8, 2)
        self.words = torch.randn(1, 12, 24)
        self.caption = torch.randn(1, 5, 24)
        self.caption[:, -1] = 10000  # Padding must never become a source value.
        self.groups = torch.tensor([0] * 4 + [1] * 4)
        self.word_groups = torch.tensor([0] * 6 + [1] * 6)
        self.args = [torch.randn(1, 8, 32), self.words[:, [0, 1, 2, 3, 6, 7, 8, 9]],
                     torch.randn(1, 12, 32), torch.randn(1, 5, 32), self.words,
                     self.groups, self.word_groups, self.caption,
                     torch.tensor([[True, True, True, True, False]])]

    def test_optimized_queries_cannot_invent_unbounded_source_values(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.03)
        for _ in range(5):
            optimizer.zero_grad()
            content = self.model(*self.args)
            # Deliberately ask for a value far outside the source convex hull.
            loss = (content['execute'] - 100).square().mean() + (content['route'] - 100).square().mean()
            loss.backward()
            optimizer.step()
        content = self.model(*self.args)
        for group in (0, 1):
            source = self.words[:, self.word_groups == group]
            selected = content['execute'][:, self.groups == group]
            self.assertTrue((selected >= source.amin(1, keepdim=True) - 1e-5).all())
            self.assertTrue((selected <= source.amax(1, keepdim=True) + 1e-5).all())
        self.assertTrue((content['caption_weights'][..., -1] == 0).all())
        self.assertTrue((content['route'].abs() < 10).all())

    def test_real_source_content_and_queries_both_affect_result(self):
        with torch.no_grad():
            self.model.source_query.weight.normal_(std=.03)
            self.model.caption_query.weight.normal_(std=.03)
        a = self.model(*self.args)
        changed = list(self.args)
        changed[4] = -self.words
        b = self.model(*changed)
        self.assertGreater((a['execute'] - b['execute']).abs().mean().item(), .1)
        a['execute'].square().mean().backward()
        for name in ('seed', 'reasoning_read.value.weight', 'request_read.value.weight'):
            grad = dict(self.model.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertGreater(grad.abs().sum().item(), 0, name)


class InteractionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        torch.set_num_threads(2)
        self.expert = InteractionExpert(24, 3, 64)
        self.gate = NativeFeatureGate(24, 64)
        self.video = torch.randn(1, 3, 8, 24)
        self.content = torch.randn(1, 8, 24)
        self.times = torch.tensor([[0., .2, .4]])
        self.parts = tuple(torch.randn(1, 3, 24) * .1 for _ in range(6))

    def test_zero_preservation_survives_learning_and_native_time_modulation(self):
        optimizer = torch.optim.Adam(self.expert.parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            out = self.expert(self.video, self.content, self.times, self.parts)
            (out - .4).square().mean().backward()
            optimizer.step()
        for h, c in [(torch.zeros_like(self.video), self.content),
                     (self.video, torch.zeros_like(self.content))]:
            out = self.expert(h, c, self.times, self.parts)
            torch.testing.assert_close(out, torch.zeros_like(out), atol=0, rtol=0)

    def test_visual_and_content_inputs_receive_generation_path_gradients(self):
        with torch.no_grad():
            self.gate.output.weight.normal_(std=.02)
        h = self.video.clone().requires_grad_()
        c = self.content.clone().requires_grad_()
        f = torch.randn(1, 3, 8, 64)
        out = self.gate(f, self.expert(h, c, self.times, self.parts))
        out.square().mean().backward()
        for x in (h, c):
            self.assertGreater(x.grad.abs().sum().item(), 0)
        for name in ('condition.weight', 'relation.v.weight', 'temporal.v.weight', 'ffn.0.weight'):
            self.assertGreater(dict(self.expert.named_parameters())[name].grad.abs().sum().item(), 0, name)

    def test_native_features_are_preserved_at_initialization_and_zero_interaction(self):
        f = torch.randn(1, 3, 8, 64)
        torch.testing.assert_close(self.gate(f, self.video), f, atol=0, rtol=0)
        with torch.no_grad():
            self.gate.output.weight.normal_(std=2)
        torch.testing.assert_close(self.gate(f, torch.zeros_like(self.video)), f, atol=0, rtol=0)
        torch.testing.assert_close(self.gate(torch.zeros_like(f), self.video), torch.zeros_like(f), atol=0, rtol=0)

    def test_temporal_and_relation_exchanges_change_nontrivial_interactions(self):
        full = self.expert(self.video, self.content, self.times)
        no_time = self.expert(self.video, self.content, self.times, temporal=False)
        no_relation = self.expert(self.video, self.content, self.times, relations=False)
        self.assertGreater((full - no_time).abs().mean().item(), 1e-4)
        self.assertGreater((full - no_relation).abs().mean().item(), 1e-4)

    def test_joint_read_preserves_mass_and_depends_on_real_video_values(self):
        reader = SourceGroundedRead(24, 3)
        content = {'route': torch.randn_like(self.content), 'execute': self.content}
        groups = torch.tensor([0, 0, 1, 1, 1, 1, 2, 2])
        message, details = reader(self.video, self.video, content, groups,
                                  self.times, self.expert, return_details=True)
        d, a, p = details['direct_logits'], details['associated_logits'], details['prior']
        explicit = torch.cat((d + p - math.log(2), a + p - math.log(2),
                              torch.zeros_like(d[..., :1])), -1).softmax(-1)
        count = self.content.shape[1]
        torch.testing.assert_close(details['write_weights'],
                                   explicit[..., :count] + explicit[..., count:2*count])
        self.assertTrue((details['write_weights'].sum(-1) < 1).all())
        off = reader(self.video, self.video, content, groups, self.times,
                     self.expert, video_feedback=False)
        torch.testing.assert_close(off, torch.zeros_like(off), atol=0, rtol=0)
        self.assertGreater(message.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
