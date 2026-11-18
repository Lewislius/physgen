"""Target correctness, masks and regression compatibility of temporal FM errors."""
from pathlib import Path
import importlib.util,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.supervision import generation_loss


class TemporalSupervisionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)
        self.clean=torch.randn(1,3,10,4,6)
        self.noise=torch.randn_like(self.clean)
        self.target=self.noise-self.clean
        self.valid=torch.ones(4,6)

    def test_perfect_changing_and_static_targets_have_zero_all_terms(self):
        for clean in (self.clean,self.clean[:,:,:1].expand_as(self.clean)):
            loss,parts=generation_loss(self.noise-clean,self.noise,clean,self.valid)
            self.assertEqual(float(loss),0.)
            self.assertTrue(all(float(x)==0 for x in parts.values()))

    def test_same_pointwise_error_but_wrong_temporal_structure_is_penalized(self):
        constant=torch.ones_like(self.target)
        alternating=constant.clone();alternating[:,:,::2]*=-1
        a,pa=generation_loss(self.target+constant,self.noise,self.clean,self.valid)
        b,pb=generation_loss(self.target+alternating,self.noise,self.clean,self.valid)
        torch.testing.assert_close(pa['fm'],pb['fm'])
        self.assertLess(float(pa['temporal']),1e-10)
        self.assertGreater(float(pb['temporal']),.1)
        self.assertGreater(float(b-a),.02)

    def test_known_first_frame_and_invalid_pixels_do_not_supply_targets(self):
        self.valid[:,:2]=0
        pred=(self.target+.3*torch.randn_like(self.target)).requires_grad_()
        a,parts=generation_loss(pred,self.noise,self.clean,self.valid)
        changed=pred.detach().clone();changed[:,:,:1]+=100;changed[:,:,:,:,:2]+=100
        b,other=generation_loss(changed,self.noise,self.clean,self.valid)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        a.backward()
        self.assertEqual(float(pred.grad[:,:,:1].abs().max()),0.)
        self.assertEqual(float(pred.grad[:,:,:,:,:2].abs().max()),0.)
        self.assertGreater(float(pred.grad[:,:,1:,: ,2:].abs().sum()),0.)

    def test_zero_temporal_weight_matches_original_objective(self):
        path=bootstrap.BASE/'experiments/physical_content_v7/denoising_matched_fusion_r15d/snapshots/20261106T011032Z/physical_content/supervision.py'
        spec=importlib.util.spec_from_file_location('_original_generation_loss',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        pred=self.target+torch.randn_like(self.target)
        actual,parts=generation_loss(pred,self.noise,self.clean,self.valid,temporal_weight=0)
        expected,previous=old.generation_loss(pred,self.noise,self.clean,self.valid)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        for key in previous:torch.testing.assert_close(parts[key],previous[key],rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
