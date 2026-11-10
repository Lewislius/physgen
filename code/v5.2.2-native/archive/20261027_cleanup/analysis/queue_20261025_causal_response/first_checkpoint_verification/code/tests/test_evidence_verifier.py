import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from physgen_native.evidence_verifier import ProcessEvidenceVerifier
from physgen_native.process_distribution import ordered_process_distribution


class EvidenceVerifierTests(unittest.TestCase):
    def test_paired_targets_only_train_verifier_and_comparison_is_symmetric(self):
        torch.manual_seed(42)
        model=ProcessEvidenceVerifier(8,12,16,lags=(1,2,4))
        a=torch.randn(6,2,3,8,requires_grad=True)
        b=torch.randn_like(a,requires_grad=True)
        c=torch.randn(1,3,12,requires_grad=True)
        times=torch.arange(6).float();valid=torch.ones(2,3)
        ta={h:torch.rand(6-h,requires_grad=True) for h in model.lags}
        tb={h:torch.rand(6-h,requires_grad=True) for h in model.lags}
        loss=model.paired_calibration_loss(a,b,ta,tb,c,times,valid)
        reverse=model.paired_calibration_loss(b,a,tb,ta,c,times,valid)
        torch.testing.assert_close(loss,reverse)
        loss.backward()
        self.assertTrue(all(x.grad is None for x in [a,b,c,*ta.values(),*tb.values()]))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None),0)

    def test_target_is_supervision_and_generator_cannot_change_score_weights(self):
        torch.manual_seed(42)
        model=ProcessEvidenceVerifier(8,12,16,lags=(1,2,4))
        reference=torch.randn(6,2,3,8,requires_grad=True)
        candidate=torch.randn_like(reference,requires_grad=True)
        condition=torch.randn(1,3,12,requires_grad=True);times=torch.arange(6).float();valid=torch.ones(2,3)
        _,_,targets=ordered_process_distribution(candidate,reference,valid,lags=model.lags,return_pairs=True)
        calibration,_=model.evidence_calibration_loss(candidate,targets,condition,times,valid)
        calibration.backward()
        self.assertIsNone(candidate.grad);self.assertIsNone(reference.grad);self.assertIsNone(condition.grad)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        model.zero_grad(set_to_none=True)
        model.generation_loss(candidate,condition,times,valid).backward()
        self.assertGreater(float(candidate.grad.norm()),0)
        self.assertIsNone(reference.grad);self.assertIsNone(condition.grad)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
