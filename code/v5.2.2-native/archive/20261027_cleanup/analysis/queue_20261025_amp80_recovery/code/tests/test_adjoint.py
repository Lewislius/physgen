import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn
from physgen_native.adjoint import discrete_adjoint


class AdjointTests(unittest.TestCase):
    def test_recorded_credit_equals_independent_suffix_derivatives(self):
        torch.manual_seed(11)
        x=torch.randn(2,4,dtype=torch.float64)
        c=torch.randn_like(x,requires_grad=True);u=torch.zeros_like(c)
        def step(s,p,n,i):return .9*s+.1*(s+p).tanh()
        recorded={}
        def callback(i,adjoint,credit):recorded[5-i]=credit.clone()
        y=discrete_adjoint(x,c,u,step,5,gradient_steps=5,vjp_callback=callback)
        full=torch.autograd.grad(y.square().sum(),c)[0]
        for k in (1,3,5):
            suffix=discrete_adjoint(x,c,u,step,5,gradient_steps=k)
            expected=torch.autograd.grad(suffix.square().sum(),c)[0]
            torch.testing.assert_close(recorded[k],expected,atol=1e-12,rtol=1e-10)
        torch.testing.assert_close(recorded[5],full)

    def test_connected_suffix_and_parameter_derivatives(self):
        for k in (1,3,7):
            torch.manual_seed(42)
            layer=nn.Linear(5,5).double()
            x=torch.randn(2,5,dtype=torch.float64,requires_grad=True)
            c=torch.randn_like(x,requires_grad=True);u=torch.randn_like(x,requires_grad=True)
            def step(s,c,u,i):return s+.03*(layer(s+c).tanh()*5-layer(s+u).tanh()*4)/(i+1)
            full=x
            for i in range(7):
                if i==7-k and k<7:full=full.detach()
                # Detaching the prefix's state also cuts earlier C/parameter
                # paths. Its values still follow exactly the original sampler.
                full=step(full,c,u,i)
            actual=discrete_adjoint(x,c,u,step,7,gradient_steps=k,parameters=layer.parameters())
            torch.testing.assert_close(actual,full,atol=0,rtol=0)
            inputs=(x,c,u,*layer.parameters())
            expected=torch.autograd.grad(full.square().sum(),inputs,allow_unused=True)
            obtained=torch.autograd.grad(actual.square().sum(),inputs,allow_unused=True)
            for a,b in zip(expected,obtained):
                if a is None:self.assertIsNone(b)
                else:torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)

    def test_inactive_null_and_frozen_generator(self):
        c=torch.randn(2,4,requires_grad=True);u=torch.zeros_like(c);x=torch.randn_like(c)
        def step(s,p,n,i):return .9*s+p.sin()*.1
        y=discrete_adjoint(x,c,u,step,4,gradient_steps=4)
        gradient=torch.autograd.grad(y.sum(),c)[0]
        torch.testing.assert_close(gradient,c.cos()*(1-.9**4))


if __name__=='__main__':unittest.main()
