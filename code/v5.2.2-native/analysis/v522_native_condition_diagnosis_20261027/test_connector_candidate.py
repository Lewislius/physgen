"""Gradient-boundary and masked-token contracts for the isolated prototype."""
from pathlib import Path
import sys
import unittest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from connector_response_candidate import connector_response_loss, reader_response


class Reader(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.num_heads = 16, 2
        self.k = nn.Linear(16, 16)
        self.v = nn.Linear(16, 16)
        self.norm_k = nn.LayerNorm(16)
        self.requires_grad_(False)


class ConnectorCandidateContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.reader = Reader()
        self.query = torch.randn(1, 8, 2, 8, requires_grad=True)
        self.a = torch.randn(1, 5, 16, requires_grad=True)
        self.b = torch.randn(1, 7, 16, requires_grad=True)

    def test_native_response_is_zero_loss_fixed_point(self):
        loss, _ = connector_response_loss(self.reader, self.query, self.a, self.b, self.a, self.b)
        self.assertEqual(float(loss.detach()), 0.)

    def test_collapsed_conditions_receive_gradient_without_teacher_leak(self):
        c = torch.zeros(1, 4, 16, requires_grad=True)
        other = torch.zeros_like(c, requires_grad=True)
        loss, _ = connector_response_loss(self.reader, self.query, c, other, self.a, self.b)
        loss.backward()
        self.assertGreater(float(loss.detach()), 0.)
        self.assertGreater(float(c.grad.norm()), 0.)
        self.assertGreater(float(other.grad.norm()), 0.)
        self.assertIsNone(self.query.grad)
        self.assertIsNone(self.a.grad)
        self.assertIsNone(self.b.grad)
        self.assertTrue(all(p.grad is None for p in self.reader.parameters()))

    def test_masked_tokens_do_not_change_reader_response(self):
        base = reader_response(self.reader, self.query, self.a)
        extended = torch.cat((self.a, torch.randn(1, 3, 16)*100), 1)
        mask = torch.tensor([[True]*5+[False]*3])
        actual = reader_response(self.reader, self.query, extended, mask)
        torch.testing.assert_close(actual, base)


if __name__ == '__main__':
    unittest.main()
