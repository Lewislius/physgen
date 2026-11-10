import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from physgen_native.temporal_critic import TemporalProcessCritic


class TemporalCriticTests(unittest.TestCase):
    def test_generator_gradient_uses_images_only(self):
        torch.manual_seed(42)
        model=TemporalProcessCritic(12,10,16)
        features=torch.randn(21,2,3,12,requires_grad=True)
        condition=torch.randn(1,4,10,requires_grad=True)
        loss=model.generation_loss(features,condition,torch.linspace(0,4,21),torch.ones(4,6))
        loss.backward()
        self.assertGreater(float(features.grad[-4:].abs().sum()),0)
        self.assertIsNone(condition.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_calibration_keeps_temporal_pairs_and_freezes_inputs(self):
        torch.manual_seed(7)
        model=TemporalProcessCritic(12,10,16)
        real=torch.randn(21,2,3,12,requires_grad=True)
        fake=torch.randn_like(real,requires_grad=True)
        condition=torch.randn(1,4,10,requires_grad=True)
        times=torch.linspace(0,4,21);valid=torch.ones(4,6)
        scores=model(real,condition,times,valid)
        self.assertEqual({k:tuple(v.shape) for k,v in scores.items()},
                         {1:(20,4),2:(19,4),4:(17,4),8:(13,4),16:(5,4)})
        loss,_=model.calibration_loss(real,fake,condition,times,valid)
        loss.backward()
        self.assertIsNone(real.grad);self.assertIsNone(fake.grad);self.assertIsNone(condition.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()))


if __name__=='__main__':unittest.main()
