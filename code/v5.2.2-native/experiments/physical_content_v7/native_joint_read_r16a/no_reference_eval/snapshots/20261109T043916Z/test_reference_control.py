"""Check that the intervention removes just the zero-value attention option."""
from pathlib import Path
import sys, unittest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),'/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7/native_joint_read_r16a/snapshots/20261109T040426Z']
import bootstrap
import torch
from physical_content.joint_read import read_physical
from reference_control import without_reference


class ReferenceControlTest(unittest.TestCase):
    def test_known_reference_mass_and_prior_immutability(self):
        q=torch.zeros(1,1,1,2);k=torch.zeros(1,1,2,2)
        v=torch.eye(2).reshape(1,1,2,2);request_keys=k.clone()
        reference=q.clone();prior=torch.zeros(3);before=prior.clone()
        args=(q,k,v,request_keys,reference,prior)
        options=dict(request_query=q,relation_count=2)
        normal=read_physical(*args,**options)
        changed=without_reference(read_physical)(*args,**options)
        torch.testing.assert_close(normal,torch.full_like(normal,1/3))
        torch.testing.assert_close(changed,torch.full_like(changed,1/2))
        torch.testing.assert_close(prior,before,rtol=0,atol=0)

    def test_removed_reference_cannot_change_result_and_zero_values_stay_zero(self):
        torch.manual_seed(11)
        q=torch.randn(1,2,5,4);k=torch.randn(1,2,3,4);v=torch.randn_like(k)
        keys=torch.randn(1,2,6,4);prior=torch.tensor([.5,.5,1.,1.]).log()
        rq=torch.randn_like(q);reader=without_reference(read_physical)
        first=reader(q,k,v,keys,torch.zeros(1,2,1,4),prior,request_query=rq,relation_count=2)
        other=reader(q,k,v,keys,torch.randn(1,2,1,4)*100,prior,request_query=rq,relation_count=2)
        torch.testing.assert_close(first,other,rtol=0,atol=0)
        zero=reader(q,k,torch.zeros_like(v),keys,torch.zeros(1,2,1,4),prior,request_query=rq,relation_count=2)
        self.assertEqual(float(zero.abs().max()),0.)


if __name__=='__main__': unittest.main(verbosity=2)
