from pathlib import Path
import sys,unittest,tempfile,json
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.grounded_write import grounded_message
from physical_content.joint_read import read_physical
from physical_content.legacy_joint_read import read_physical as legacy_read
import test_support


class GroundedJointTests(unittest.TestCase):
    def test_joint_reader_recovers_B_operator_when_queries_match(self):
        torch.manual_seed(7)
        q=torch.randn(1,2,7,4);k=torch.randn(1,2,5,4);v=torch.randn_like(k)
        text=torch.randn(1,2,6,4);ref=torch.randn(1,2,1,4);prior=torch.tensor([.5,.5,1.,.5,.5,1.]).log()
        ours=read_physical(q,k,v,text,ref,prior,request_query=q,relation_count=3)
        old=legacy_read(q,k,v,text,ref,prior,relation_count=3)
        torch.testing.assert_close(ours,old,atol=2e-6,rtol=2e-6)

    def test_common_physics_modulates_state_without_free_common_offset(self):
        torch.manual_seed(9)
        m=torch.randn(1,1,8).expand(1,12,8);s=torch.randn_like(m)
        actual=grounded_message(m,s)
        torch.testing.assert_close(actual,(s-s.mean(1,True))*m,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(actual.mean(1),torch.zeros(1,8),atol=1e-6,rtol=0)
        self.assertGreater(float(actual.square().mean()),0.)

    def test_varying_messages_retain_covariance_and_gradients(self):
        torch.manual_seed(10)
        m=torch.randn(1,12,8,requires_grad=True);s=torch.randn_like(m,requires_grad=True)
        actual=grounded_message(m,s)
        expected=((s-s.mean(1,True))*(m-m.mean(1,True))).mean(1)
        torch.testing.assert_close(actual.mean(1),expected,atol=1e-6,rtol=1e-6)
        actual.square().mean().backward()
        self.assertGreater(float(m.grad.norm()),0.);self.assertGreater(float(s.grad.norm()),0.)

    def fixture(self):
        case=test_support.JointMemoryTest();case.setUp();self.addCleanup(case.doCleanups);return case

    def test_full_model_native_start_and_parameter_ownership(self):
        case=self.fixture();model=case.model()
        torch.testing.assert_close(case.full(model),case.full(model,physical_enabled=False),atol=1e-5,rtol=1e-5)
        fusion=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        torch.testing.assert_close(fusion.state_value.weight,torch.eye(16),atol=0,rtol=0)
        for own,old in [('query','q'),('key','k'),('value','v'),('output','o')]:
            self.assertTrue(getattr(fusion,own).weight.requires_grad)
            self.assertFalse(getattr(native,old).weight.requires_grad)
            self.assertNotEqual(getattr(fusion,own).weight.data_ptr(),getattr(native,old).weight.data_ptr())

    def test_full_generation_gradients_reach_physical_reader_and_writer(self):
        case=self.fixture();model=case.model();case.activate(model);model.set_phase(0)
        case.full(model).square().mean().backward()
        for layer in ['query','key','value','output','state_value']:
            grad=getattr(model.fusions['1'],layer).weight.grad
            self.assertIsNotNone(grad);self.assertGreater(float(grad.norm()),0.)
        self.assertGreater(float(model.enhancement.execution_project.weight.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))

    def test_all_declared_execution_controls_are_real_and_finite(self):
        case=self.fixture();model=case.model().eval();case.activate(model)
        normal=case.full(model)
        for options in [dict(memory_mode='direct'),dict(memory_mode='request'),dict(memory_mode='absolute'),
                        dict(shared_physical_query=True),dict(physical_enabled=False)]:
            other=case.full(model,**options)
            self.assertTrue(torch.isfinite(other).all())
            self.assertGreater(float((normal-other).abs().max()),1e-7)

if __name__=='__main__':unittest.main(verbosity=2)
