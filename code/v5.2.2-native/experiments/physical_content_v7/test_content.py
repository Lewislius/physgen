"""Meaningful checks of content ownership, write gradients and masked loss."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement,Fusion
from physical_content.evidence import resample_words
from physical_content.supervision import generation_loss

class NativeAttention(nn.Module):
    def __init__(self,width=24):
        super().__init__();self.dim=width;self.num_heads=3;self.head_dim=8
        self.q=nn.Linear(width,width,bias=False);self.k=nn.Linear(width,width,bias=False)
        self.v=nn.Linear(width,width,bias=False);self.o=nn.Linear(width,width,bias=False)
        self.norm_q=nn.RMSNorm(width);self.norm_k=nn.RMSNorm(width)
    def forward(self,x):return x

class ContentTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(42);torch.set_num_threads(2)
    def enhancement(self):
        interface=dict(mean=torch.randn(32)*.1,scale=torch.tensor(1.2),weight=torch.randn(24,32)/32**.5,
                       bias=torch.randn(24)*.1,target_scale=torch.tensor(.5))
        cfg=dict(model=dict(qwen_width=32,width=24,q_tokens=8,q_heads=4,rounds=2),train=dict(gradient_checkpointing=False))
        return Enhancement(cfg,interface),interface
    def evidence(self):
        return dict(words=torch.randn(6,32),word_groups=torch.tensor([0,0,1,1,2,2]),
                    memory=torch.randn(1,10,32),memory_mask=torch.ones(1,10,dtype=torch.bool),native_words=torch.randn(6,24))
    def test_grounded_path_and_no_target_condition(self):
        module,init=self.enhancement();e=self.evidence();c,groups=module(e)
        source,_,_=resample_words((e['words']-init['mean'])/init['scale'],e['word_groups'],8)
        torch.testing.assert_close(c,F.linear(source,init['weight'],init['bias']))
        e['native_words'].normal_(100,20)
        torch.testing.assert_close(c,module(e)[0])
        changed={**e,'words':torch.randn_like(e['words'])}
        self.assertGreater(float((module(changed)[0]-c).norm()),1.)
        self.assertGreater(float((c-c.mean(1,keepdim=True)).square().mean()),.01)
    def test_content_update_receives_gradient(self):
        module,_=self.enhancement();e=self.evidence();optimizer=torch.optim.SGD(module.parameters(),lr=.01)
        for _ in range(2):
            optimizer.zero_grad();c,_=module(e);module.content_loss(c,e).backward();optimizer.step()
        self.assertGreater(float(module.block.cross.q.weight.grad.norm()),0.)
        self.assertGreater(float(module.seed.grad.norm()),0.)
    def test_native_text_and_first_frame_preserved_then_state_gradient(self):
        attention=NativeAttention();fusion=Fusion(attention)
        x=torch.randn(1,6,24,requires_grad=True);context=torch.randn(1,5,24)
        c=torch.randn(1,8,24,requires_grad=True);groups=torch.arange(8)//2;times=torch.tensor([[0.,1.,2.]])
        initial,_=fusion(x,context,c,times,(3,1,2),attention,groups)
        with torch.no_grad():fusion.output.weight.copy_(torch.eye(24))
        active,_=fusion(x,context,c,times,(3,1,2),attention,groups)
        torch.testing.assert_close(initial[:,:2],active[:,:2])
        self.assertGreater(float((active[:,2:]-initial[:,2:]).norm()),0.)
        active[:,2:].square().mean().backward()
        self.assertGreater(float(c.grad.norm()),0.)
        self.assertGreater(float(fusion.state_value.weight.grad.norm()),0.)
    def test_masked_pixels_cannot_change_objective(self):
        shape=(1,3,9,6,8);noise=torch.randn(shape);clean=torch.randn(shape);pred=noise-clean
        valid=torch.ones(6,8);valid[:,-2:]=0
        zero,_=generation_loss(pred,noise,clean,valid);self.assertEqual(float(zero),0.)
        pred=pred.clone().requires_grad_();pred.data[:,:,:,:,-2:]+=1000
        loss,_=generation_loss(pred,noise,clean,valid);self.assertEqual(float(loss),0.)
        pred.data[:,:,3:5,:,:3]+=2
        loss,parts=generation_loss(pred,noise,clean,valid);loss.backward()
        self.assertGreater(float(parts['tail']),float(parts['fm']))
        self.assertEqual(float(pred.grad[:,:,:,:,-2:].abs().sum()),0.)

if __name__=='__main__':unittest.main()
