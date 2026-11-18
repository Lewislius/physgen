"""Necessary reader wiring, generation-gradient and anchor-control checks."""
from pathlib import Path
import importlib.util
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement,Fusion
from physical_content.evidence import resample_words

torch.set_num_threads(2)
REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/content_modulation_r6/snapshots/20261031T092555Z/physical_content/model.py'
spec=importlib.util.spec_from_file_location('physical_content.r6_reference',REFERENCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.dim=16;self.num_heads=2;self.head_dim=8
        for name in ['q','k','v','o']:setattr(self,name,nn.Linear(16,16,bias=False))
        self.norm_q=nn.RMSNorm(16);self.norm_k=nn.RMSNorm(16)

    def forward(self,x,context):
        return x


class ContentReaderTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.cfg={'model':{'qwen_width':32,'width':16,'q_heads':4,'q_tokens':8,
            'rounds':2,'grounded_read_heads':2,'content_projection_init_scale':.1},
            'train':{'gradient_checkpointing':True}}
        self.interface={'mean':torch.randn(32),'scale':torch.rand(32)+.5,
            'weight':torch.randn(16,32)*.05,'bias':torch.randn(16)*.01,'target_scale':torch.tensor(.1)}
        self.evidence={'words':torch.randn(6,32),'word_groups':torch.arange(6)//3,
            'request_words':torch.randn(5,32),'memory':torch.randn(1,9,32),
            'memory_mask':torch.ones(1,9,dtype=torch.bool)}

    def test_parameter_initialization_and_rng_are_unchanged(self):
        torch.manual_seed(42);old=reference.Enhancement(self.cfg,self.interface)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42);new=Enhancement(self.cfg,self.interface)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for name,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)

    def test_request_is_read_and_unused_prefix_is_excluded_with_generation_credit(self):
        model=Enhancement(self.cfg,self.interface)
        original,groups=model(self.evidence)
        changed=dict(self.evidence,memory=torch.randn_like(self.evidence['memory'])*20)
        torch.testing.assert_close(original,model(changed)[0],rtol=0,atol=0)
        changed=dict(self.evidence,request_words=self.evidence['request_words'].flip(-1))
        self.assertGreater(float((original-model(changed)[0]).detach().abs().max()),1e-6)
        attention=Attention().requires_grad_(False);fusion=Fusion(attention,.25)
        x=torch.randn(1,4,16);context=torch.randn(1,5,16)
        output,_=fusion(x,context,original,torch.tensor([[0.,1.]]),(2,1,2),attention,groups)
        F.mse_loss(output[:,2:],torch.randn_like(output[:,2:])).backward()
        for parameter in [model.seed,model.block.cross.q.weight,model.block.cross.v.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0)
        self.assertIsNone(model.project.weight.grad)

    def test_image_only_and_fixed_update_and_anchor_controls_remain_well_defined(self):
        model=Enhancement(self.cfg,self.interface).eval()
        negative=dict(self.evidence,request_words=torch.empty(0,32))
        self.assertTrue(torch.isfinite(model(negative)[0]).all())
        update=torch.randn(1,8,32)
        actual=model(self.evidence,update_override=update)[0]
        altered=dict(self.evidence,request_words=torch.randn(5,32))
        torch.testing.assert_close(actual,model(altered,update_override=update)[0],rtol=0,atol=0)
        with torch.no_grad():
            model.project.weight.zero_()
            words=(self.evidence['words']-model.anchor_mean)/model.anchor_scale
            anchor,_,_=resample_words(words,self.evidence['word_groups'],8)
            expected=F.linear(anchor,model.anchor_weight,model.anchor_bias)
            torch.testing.assert_close(model(self.evidence)[0],expected,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
