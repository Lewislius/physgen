"""Check train-only target ownership, time coverage and the actual gradient."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.process_supervision import observation_indices,clean_estimate,active_microstep
from physical_content.process_features import feature_distance


class ProcessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)

    def test_oracle_clean_recovery_preserves_condition(self):
        clean=torch.randn(1,2,5,3,4);noise=torch.randn_like(clean);first=torch.randn_like(clean[:,:,:1])
        prediction=(noise-clean).requires_grad_()
        for sigma in [.2,.5,.8]:
            recovered=clean_estimate((1-sigma)*clean+sigma*noise,sigma,prediction,first)
            torch.testing.assert_close(recovered[:,:,1:],clean[:,:,1:])
            torch.testing.assert_close(recovered[:,:,:1],first)
        recovered.square().sum().backward()
        self.assertEqual(float(prediction.grad[:,:,:1].abs().sum()),0.)
        self.assertGreater(float(prediction.grad[:,:,1:].abs().sum()),0.)

    def test_ordered_state_target_and_mask_have_no_target_gradient(self):
        target=torch.randn(3,2,4,8,requires_grad=True)
        correct=target.detach().clone().requires_grad_()
        zero,_=feature_distance((correct,),(target,))
        self.assertLess(abs(float(zero)),1e-6)
        reversed_states=target.detach().flip(0).clone().requires_grad_()
        valid=torch.ones(2,4);valid[:,-1]=0
        distance,_=feature_distance((reversed_states,),(target,),valid)
        self.assertGreater(float(distance),.1)
        distance.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(float(reversed_states.grad[:,:,-1].abs().sum()),0.)
        self.assertGreater(float(reversed_states.grad[:,:,:-1].abs().sum()),0.)

    def test_resumed_observations_preserve_rng_and_cover_clip(self):
        before=torch.get_rng_state();rows=[observation_indices(i,42) for i in range(200)]
        torch.testing.assert_close(before,torch.get_rng_state())
        self.assertEqual(rows[137],observation_indices(137,42))
        self.assertEqual(set().union(*map(set,rows)),set(range(0,81,3))|{80})
        for row in rows:
            self.assertEqual(row,sorted(set(row)));self.assertEqual(row[-2:],[78,80])
        policy={'sigma_min':.2,'sigma_max':.8}
        for step in range(8):
            self.assertEqual(sum(active_microstep(step,m,.5,4,policy) for m in range(4)),1)
            self.assertFalse(any(active_microstep(step,m,.95,4,policy) for m in range(4)))


if __name__=='__main__':unittest.main()
