"""Check the changed initializer's immediate conditioning derivative and native path."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import Fusion


class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.dim=16;self.num_heads=2;self.head_dim=8
        for name in ['q','k','v','o']:setattr(self,name,nn.Linear(16,16,bias=False))
        self.norm_q=nn.RMSNorm(16);self.norm_k=nn.RMSNorm(16)

    def forward(self,x,context):
        split=lambda v:v.unflatten(-1,(2,8)).transpose(1,2)
        y=F.scaled_dot_product_attention(split(self.norm_q(self.q(x))),
            split(self.norm_k(self.k(context))),split(self.v(context)))
        return self.o(y.transpose(1,2).flatten(2))


class NativeOutputTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.attention=Attention().requires_grad_(False)
        self.fusion=Fusion(self.attention,.25)
        self.x=torch.randn(1,4,16);self.context=torch.randn(1,5,16)
        self.times=torch.tensor([[0.,1.]])
        self.groups=torch.arange(8)//2

    def output(self,c):
        return self.fusion(self.x,self.context,c,self.times,(2,1,2),self.attention,self.groups)[0]

    def test_native_coordinates_and_immediate_content_gradient(self):
        torch.testing.assert_close(self.fusion.output.weight,self.attention.o.weight*.25)
        c=torch.randn(1,8,16,requires_grad=True)
        output=self.output(c)
        gradient,=torch.autograd.grad(output[:,2:].square().sum(),c)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.norm()),0.)
        torch.testing.assert_close(output[:,:2],self.attention(self.x,self.context)[:,:2])

    def test_absent_values_preserve_native_text(self):
        torch.testing.assert_close(self.output(torch.zeros(1,8,16)),self.attention(self.x,self.context))


if __name__=='__main__':unittest.main()
