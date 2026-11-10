from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from physgen_native.process_distribution import ordered_process_distribution


class ProcessDistributionTests(unittest.TestCase):
    def test_late_process_and_gradient_are_not_hidden_by_early_change(self):
        torch.manual_seed(8)
        real=torch.randn(21,2,3,12).cumsum(0).requires_grad_(True)
        frozen=real.detach().clone();frozen[13:]=frozen[12:13]
        frozen.requires_grad_(True);valid=torch.ones(2,3)
        exact,_=ordered_process_distribution(real,real,valid)
        loss,metrics=ordered_process_distribution(frozen,real,valid)
        self.assertLess(abs(float(exact.detach())),1e-5)
        self.assertGreater(float(loss.detach()),.005)
        self.assertIn('distribution/lag20',metrics)
        loss.backward()
        self.assertIsNone(real.grad)
        self.assertGreater(float(frozen.grad[13:].abs().sum()),0)

    def test_distribution_allows_spatial_realizations_but_retains_order(self):
        torch.manual_seed(42)
        real=torch.randn(21,2,3,8).cumsum(0);valid=torch.ones(2,3)
        permutation=real.flatten(1,2)[:,[5,1,3,4,0,2]].reshape_as(real)
        same,_=ordered_process_distribution(permutation,real,valid)
        reverse,_=ordered_process_distribution(real.flip(0),real,valid)
        self.assertLess(abs(float(same)),1e-5)
        self.assertGreater(float(reverse),float(same)+.01)

    def test_static_reference_does_not_require_invented_change(self):
        torch.manual_seed(7)
        real=torch.randn(1,2,3,8).expand(21,-1,-1,-1).clone()
        static,_=ordered_process_distribution(real,real,torch.ones(2,3))
        jitter,_=ordered_process_distribution(real+.05*torch.randn_like(real),real,torch.ones(2,3))
        self.assertLess(abs(float(static)),1e-5)
        self.assertGreater(float(jitter),float(static)+.001)


if __name__=='__main__':unittest.main()
