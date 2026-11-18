from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.joint_read import read_physical
from physical_content.legacy_joint_read import read_physical as legacy
import test_support


class NativeBudgetTests(unittest.TestCase):
    def values(self):
        torch.manual_seed(8)
        q=torch.randn(1,2,7,4);k=torch.randn(1,2,4,4);v=torch.randn_like(k)
        req=torch.randn(1,2,6,4);ref=torch.randn(1,2,1,4);prior=torch.tensor([.5,.5,1.,1.,1.]).log()
        return q,k,v,req,ref,prior

    def test_equal_queries_recover_B_joint_and_direct(self):
        q,k,v,req,ref,prior=self.values()
        for mode in ['joint','direct']:
            actual=read_physical(q,k,v,req,ref,prior,mode,request_query=q,relation_count=3)
            expected=legacy(q,k,v,req,ref,prior,mode,relation_count=3)
            torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)

    def test_physical_query_changes_content_without_changing_total_mass(self):
        q,k,v,req,ref,prior=self.values();v=torch.eye(4)[None,None].expand_as(v)
        a=read_physical(q,k,v,req,ref,prior,request_query=q,relation_count=3)
        b=read_physical(q.flip(-1)*12,k,v,req,ref,prior,request_query=q,relation_count=3)
        torch.testing.assert_close(a.sum(-1),b.sum(-1),atol=3e-6,rtol=3e-6)
        self.assertGreater(float((a-b).abs().max()),.01)
        self.assertTrue((b>=0).all())

    def test_common_physical_values_remain_effective(self):
        q,k,v,req,ref,prior=self.values();v=torch.ones_like(v)
        a=read_physical(q,k,v,req,ref,prior,request_query=q,relation_count=3)
        b=read_physical(-q*8,k,v,req,ref,prior,request_query=q,relation_count=3)
        torch.testing.assert_close(a,b,atol=3e-6,rtol=3e-6)
        self.assertGreater(float(a.mean()),.1)

    def fixture(self):
        c=test_support.JointMemoryTest();c.setUp();self.addCleanup(c.doCleanups);return c

    def test_native_start_and_independent_parameter_ownership(self):
        c=self.fixture();m=c.model();f=m.fusions['1'];native=m.wan.blocks[1].cross_attn
        torch.testing.assert_close(c.full(m),c.full(m,physical_enabled=False),atol=1e-5,rtol=1e-5)
        self.assertEqual(float(f.state_value.weight.abs().max()),0.)
        for own,old in [('query','q'),('key','k'),('value','v'),('output','o')]:
            self.assertTrue(getattr(f,own).weight.requires_grad)
            self.assertFalse(getattr(native,old).weight.requires_grad)
            self.assertNotEqual(getattr(f,own).weight.data_ptr(),getattr(native,old).weight.data_ptr())

    def test_generation_gradients_and_actual_controls(self):
        c=self.fixture();m=c.model();c.activate(m);m.set_phase(0)
        with torch.no_grad():m.fusions['1'].query.weight.add_(torch.randn_like(m.fusions['1'].query.weight)*.05)
        normal=c.full(m);normal.square().mean().backward()
        for layer in ['query','key','value','output','state_value']:
            g=getattr(m.fusions['1'],layer).weight.grad;self.assertIsNotNone(g);self.assertGreater(float(g.norm()),0.)
        self.assertGreater(float(m.enhancement.execution_project.weight.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in m.wan.parameters()))
        for options in [dict(memory_mode='direct'),dict(memory_mode='request'),dict(shared_physical_query=True),
                        dict(native_physical_query=True),dict(physical_enabled=False)]:
            other=c.full(m,**options);self.assertTrue(torch.isfinite(other).all())
            self.assertGreater(float((normal-other).detach().abs().max()),1e-7)


if __name__=='__main__':unittest.main(verbosity=2)
