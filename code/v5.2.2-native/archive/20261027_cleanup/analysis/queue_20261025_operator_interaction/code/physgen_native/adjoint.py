"""Exact first-order reverse differentiation of a discrete sampling trajectory.

Only step-boundary latents are retained. Each native denoising step is recomputed
and its VJP released before the preceding step, so memory does not scale with the
number of retained transformer graphs. This is the derivative of the implemented
discrete sampler, not a continuous-adjoint approximation or a detached-step sum.
"""
import torch
from torch.autograd.function import once_differentiable


def discrete_adjoint(initial,positive,negative,step,steps,*,gradient_steps,parameters=(),vjp_callback=None):
    """Differentiate the last K connected steps, including both CFG branches.

``step(state, positive, negative, index)`` must be deterministic and have no
mutable inference cache. All trainable parameters used by ``step`` are explicit
inputs so their gradients are returned normally to the outer optimizer.
"""
    if not 1<=gradient_steps<=steps:raise ValueError('Invalid connected derivative length')
    parameters=tuple(p for p in parameters if p.requires_grad)
    boundary=steps-gradient_steps

    class Integrate(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x,c,u,*weights):
            states=[]
            for i in range(steps):
                if i>=boundary:states.append(x.detach())
                x=step(x,c,u,i)
            ctx.save_for_backward(c,u,*weights,*states)
            ctx.weight_count=len(weights)
            ctx.device_type=x.device.type
            ctx.amp=torch.is_autocast_enabled(x.device.type)
            ctx.amp_dtype=torch.get_autocast_dtype(x.device.type)
            return x

        @staticmethod
        @once_differentiable
        def backward(ctx,adjoint):
            c,u,*saved=ctx.saved_tensors
            weights=saved[:ctx.weight_count];states=saved[ctx.weight_count:]
            # Separate leaves are essential: this derivative is passed back to
            # the outer Qwen/conditioner graph exactly once after accumulation.
            c=c.detach().requires_grad_(ctx.needs_input_grad[1])
            u=u.detach().requires_grad_(ctx.needs_input_grad[2])
            inputs=[c,u,*weights]
            active=[i for i,x in enumerate(inputs) if x.requires_grad]
            accumulated=[None]*len(inputs)
            for local in reversed(range(gradient_steps)):
                state=states[local].detach().requires_grad_(True)
                with torch.enable_grad(),torch.autocast(ctx.device_type,enabled=ctx.amp,dtype=ctx.amp_dtype):
                    result=step(state,c,u,boundary+local)
                    derivatives=torch.autograd.grad(result,[state]+[inputs[i] for i in active],
                        grad_outputs=adjoint,allow_unused=True)
                adjoint=derivatives[0]
                if adjoint is None:adjoint=torch.zeros_like(state)
                for index,gradient in zip(active,derivatives[1:]):
                    if gradient is not None:
                        accumulated[index]=(gradient if accumulated[index] is None else accumulated[index]+gradient)
                if vjp_callback is not None:
                    # Inspection only: this must not alter the returned VJP or
                    # the state trajectory. Tensors are detached snapshots.
                    with torch.no_grad():
                        vjp_callback(boundary+local,adjoint.detach(),
                                     None if accumulated[0] is None else accumulated[0].detach().clone())
                del result,state,derivatives
            # A truncated prefix intentionally has no derivative to its input.
            return (adjoint if boundary==0 and ctx.needs_input_grad[0] else None,*accumulated)

    return Integrate.apply(initial,positive,negative,*parameters)
