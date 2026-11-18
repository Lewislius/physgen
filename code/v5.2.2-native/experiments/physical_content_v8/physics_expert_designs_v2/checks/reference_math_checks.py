#!/usr/bin/env python3
"""Synthetic CPU checks for the design package, NOT a Wan implementation.

No checkpoint, dataset, evaluator, network, or GPU is accessed. The tiny block
intentionally omits video RoPE/VAE/backends. A pass verifies only the stated
algebra, shapes, initialization, and autograd properties, never PC/SA/quality.

Run: python checks/reference_math_checks.py
Requires: torch, PyYAML (for the design-config consistency check).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
import traceback
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)
D, HEADS, INNER, KNOWN = 48, 6, 8, 4


def heads(x: Tensor, count: int = HEADS) -> Tensor:
    return x.unflatten(-1, (count, x.shape[-1] // count)).transpose(1, 2)


def merge(x: Tensor) -> Tensor:
    return x.transpose(1, 2).flatten(2)


def mask_known(x: Tensor, known: int) -> Tensor:
    if not 0 <= known <= x.shape[1]:
        raise ValueError('invalid known token count')
    return torch.cat((torch.zeros_like(x[:, :known]), x[:, known:]), dim=1)


class TinyAttention(nn.Module):
    def __init__(self, d: int = D):
        super().__init__()
        self.q, self.k, self.v, self.o = [nn.Linear(d, d) for _ in range(4)]
        self.norm_q, self.norm_k = nn.RMSNorm(d, eps=1e-6), nn.RMSNorm(d, eps=1e-6)
        self.calls = 0
        self.last_qkv: tuple[Tensor, Tensor, Tensor] | None = None

    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        self.calls += 1
        context = x if context is None else context
        q, k, v = self.q(x), self.k(context), self.v(context)
        self.last_qkv = (q.detach().clone(), k.detach().clone(), v.detach().clone())
        value = F.scaled_dot_product_attention(
            heads(self.norm_q(q)), heads(self.norm_k(k)), heads(v), dropout_p=0.0,
        )
        return self.o(merge(value))


class TinyNativeBlock(nn.Module):
    """Only a shape/order surrogate for self -> text -> FFN with modulation."""
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(D, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(D, eps=1e-6, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(D, eps=1e-6)
        self.self_attn, self.cross_attn = TinyAttention(), TinyAttention()
        self.ffn = nn.Sequential(nn.Linear(D, 2*D), nn.GELU(approximate='tanh'), nn.Linear(2*D, D))
        self.register_buffer('mod', torch.randn(6, D) * 0.1)
        self.mod[2].add_(0.5)
        self.mod[5].add_(0.5)
        self.block_calls = 0
        self.ffn_calls = 0

    def self_step(self, h: Tensor) -> Tensor:
        u = self.norm1(h) * (1 + self.mod[1]) + self.mod[0]
        return h + self.mod[2] * self.self_attn(u)

    def text_step(self, h: Tensor, text: Tensor) -> Tensor:
        return self.cross_attn(self.norm3(h), text)

    def ffn_input(self, h: Tensor) -> Tensor:
        return self.norm2(h) * (1 + self.mod[4]) + self.mod[3]

    def ffn_eval(self, u: Tensor) -> Tensor:
        self.ffn_calls += 1
        return self.ffn(u)

    def forward(self, h: Tensor, text: Tensor) -> Tensor:
        self.block_calls += 1
        h = self.self_step(h)
        h = h + self.text_step(h, text)
        return h + self.mod[5] * self.ffn_eval(self.ffn_input(h))


class RequestKeyEnhancer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm_c = nn.LayerNorm(D, eps=1e-6, elementwise_affine=False)
        self.norm_t = nn.LayerNorm(D, eps=1e-6, elementwise_affine=False)
        self.q, self.k, self.v, self.o = [nn.Linear(D, D, bias=False) for _ in range(4)]
        for layer in (self.q, self.k, self.v):
            nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(self.o.weight)

    def forward(self, c: Tensor, text: Tensor, text_mask: Tensor) -> Tensor:
        if text_mask.dtype != torch.bool or not bool(text_mask.any(-1).all()):
            raise ValueError('mask must be bool with at least one real token per sample')
        q = heads(self.q(self.norm_c(c)))
        t = self.norm_t(text)
        z = F.scaled_dot_product_attention(
            q, heads(self.k(t)), heads(self.v(t)),
            attn_mask=text_mask[:, None, None, :], dropout_p=0.0,
        )
        return c + self.o(merge(z))


@dataclass
class ReadResult:
    message: Tensor
    write: Tensor


class PhysicsReader(nn.Module):
    """Small direct reader; no native text update and no duplicate module alias."""
    def __init__(self, native: TinyAttention):
        super().__init__()
        self.key, self.value, self.output = [nn.Linear(D, D, bias=False) for _ in range(3)]
        with torch.no_grad():
            self.key.weight.copy_(native.k.weight)
            self.value.weight.copy_(native.v.weight)
        self.key_norm = copy.deepcopy(native.norm_k)
        self.reference = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.zeros_(self.output.weight)

    def forward(self, h: Tensor, cr: Tensor, ce: Tensor, block: TinyNativeBlock,
                groups: Tensor, known: int = KNOWN, chunk: int | None = None) -> ReadResult:
        # The real reader also retains existing video-time query. Omitted here.
        q = heads(block.cross_attn.norm_q(block.cross_attn.q(block.norm3(h[:, known:]))))
        k, v = heads(self.key_norm(self.key(cr))), heads(self.value(ce))
        ref = heads(self.reference.expand(h.shape[0], -1, -1))
        k, v = torch.cat((k, ref), 2), torch.cat((v, torch.zeros_like(ref)), 2)
        counts = torch.bincount(groups, minlength=int(groups.max())+1)
        prior = torch.cat((-counts[groups].float().log(), torch.zeros(1)))
        qchunks = [q] if chunk is None else list(q.split(chunk, dim=2))
        out = []
        for qi in qchunks:
            logits = qi.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
            out.append((logits + prior).softmax(-1).to(v.dtype) @ v)
        future = merge(torch.cat(out, dim=2))
        m = torch.cat((torch.zeros_like(h[:, :known]), future), dim=1)
        return ReadResult(m, mask_known(self.output(m), known))


class SourceExpert(nn.Module):
    def __init__(self, inner: int = INNER):
        super().__init__()
        self.state = nn.Linear(D, inner, bias=False)
        self.source = nn.Linear(D, inner, bias=False)
        self.output = nn.Linear(inner, D, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, u: Tensor, source: Tensor) -> Tensor:
        normalized = F.layer_norm(source, (D,), eps=1e-6)
        return self.output(F.gelu(self.state(u), approximate='tanh') * self.source(normalized))


class MixedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        inner = D // 4  # 4*D*(D/4) = D^2 = two 3*D*(D/6) experts.
        self.state = nn.Linear(D, inner, bias=False)
        self.phys = nn.Linear(D, inner, bias=False)
        self.text = nn.Linear(D, inner, bias=False)
        self.output = nn.Linear(inner, D, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, u: Tensor, m: Tensor, t: Tensor) -> Tensor:
        s = self.phys(F.layer_norm(m, (D,))) + self.text(F.layer_norm(t, (D,)))
        return self.output(F.gelu(self.state(u), approximate='tanh') * s)


def p2(block, reader, h, text, cr, ce, groups):
    r = reader(h, cr, ce, block, groups)
    return block(h + r.write, text)


def post_control(block, reader, h, text, cr, ce, groups):
    h = block.self_step(h)
    t = block.text_step(h, text)
    r = reader(h, cr, ce, block, groups)
    h = h + t + r.write
    return h + block.mod[5] * block.ffn_eval(block.ffn_input(h))


def p4(block, reader, ep, et, h, text, cr, ce, groups):
    hs = block.self_step(h)
    r = reader(hs, cr, ce, block, groups)
    hp = hs + r.write
    t = block.text_step(hp, text)
    hpt = hp + t
    u = block.ffn_input(hpt)
    xp = mask_known(ep(u, r.message), KNOWN)
    xt = et(u, t)
    return hpt + block.mod[5] * (block.ffn_eval(u) + xp + xt)


def setup():
    block = TinyNativeBlock()
    reader = PhysicsReader(block.cross_attn)
    h, text, cr, ce = torch.randn(1, 12, D), torch.randn(1, 7, D), torch.randn(1, 6, D), torch.randn(1, 6, D)
    groups = torch.tensor([0, 0, 1, 1, 1, 2], dtype=torch.long)
    return block, reader, h, text, cr, ce, groups


def activate(linear: nn.Linear):
    with torch.no_grad():
        nn.init.normal_(linear.weight, std=0.02)


def close(a: Tensor, b: Tensor, atol: float = 1e-6):
    torch.testing.assert_close(a, b, atol=atol, rtol=1e-5)


def has_grad(p: nn.Parameter) -> bool:
    return p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0


def test_budget_identity():
    d, f, layers, e = 3072, 14336, 30, 227580928
    old_site = 4*d*d + 256*d + 2*d
    direct_site = old_site-d*d
    lora = lambda r: layers*r*(8*d+2*(d+f))
    text = layers*4*d*16
    assert 6*d*512 == d*d == 4*d*768
    assert e + 10*old_site + lora(32) == 670010368
    assert e + 10*direct_site + lora(64) + text == 638553088
    assert e + 10*old_site + lora(32) + text == 675908608
    assert e + 10*old_site + lora(32) + text + 4*d*d == 713657344


def test_sites():
    assert [i for i in range(30) if (i+1)%3 == 0] == [2,5,8,11,14,17,20,23,26,29]
    assert [i for i in range(30) if (i+1)%5 == 0] == [4,9,14,19,24,29]


def test_p1_initial_identity():
    _, _, _, t, c, _, _ = setup()
    out = RequestKeyEnhancer()(c, t, torch.ones(1, 7, dtype=torch.bool))
    assert torch.equal(out, c)


def test_p1_padding_invariance():
    _, _, _, t, c, _, _ = setup()
    module = RequestKeyEnhancer(); activate(module.o)
    mask = torch.tensor([[True, True, True, True, True, False, False]])
    changed = t.clone(); changed[:, 5:] = torch.randn_like(changed[:, 5:])*100
    close(module(c, t, mask), module(c, changed, mask))


def test_p1_request_response_and_value_identity():
    _, _, _, t, c, ce, _ = setup()
    module = RequestKeyEnhancer(); activate(module.o)
    value_before = ce.clone(); mask = torch.ones(1,7,dtype=torch.bool)
    x = module(c, t, mask); y = module(c, torch.randn_like(t), mask)
    assert not torch.allclose(x, y)
    assert torch.equal(ce, value_before)


def test_p1_staged_gradients():
    _, _, _, t, c, _, _ = setup(); module = RequestKeyEnhancer()
    opt = torch.optim.SGD(module.parameters(), lr=0.05)
    mask = torch.ones(1,7,dtype=torch.bool); target = torch.randn_like(c)
    F.mse_loss(module(c,t,mask), target).backward()
    assert has_grad(module.o.weight)
    assert module.q.weight.grad is not None and not has_grad(module.q.weight)
    opt.step(); opt.zero_grad(set_to_none=True)
    F.mse_loss(module(c,t,mask), target).backward()
    assert all(has_grad(x.weight) for x in [module.q,module.k,module.v,module.o])


def test_reader_zero_execute():
    b,r,h,t,cr,ce,g = setup(); activate(r.output)
    out = r(h,cr,torch.zeros_like(ce),b,g)
    assert torch.count_nonzero(out.message) == 0
    assert torch.count_nonzero(out.write) == 0


def test_query_chunk_equivalence():
    b,r,h,t,cr,ce,g = setup(); activate(r.output)
    whole = r(h,cr,ce,b,g); chunked = r(h,cr,ce,b,g,chunk=3)
    close(whole.message, chunked.message); close(whole.write, chunked.write)


def test_known_slice():
    b,r,h,t,cr,ce,g = setup(); activate(r.output)
    out=r(h,cr,ce,b,g)
    assert torch.count_nonzero(out.message[:,:KNOWN]) == 0
    assert torch.count_nonzero(out.write[:,:KNOWN]) == 0
    expert = SourceExpert(); activate(expert.output)
    xp=mask_known(expert(h,out.message),KNOWN)
    assert torch.count_nonzero(xp[:,:KNOWN]) == 0


def test_p2_zero_matches_native():
    b,r,h,t,cr,ce,g=setup()
    close(p2(b,r,h,t,cr,ce,g), b(h,t))


def test_p2_qkv_respond():
    b,r,h,t,cr,ce,g=setup();activate(r.output)
    p2(b,r,h,t,cr,ce,g); before=b.self_attn.last_qkv
    p2(b,r,h,t,cr,ce+torch.randn_like(ce),g); after=b.self_attn.last_qkv
    assert before is not None and after is not None
    for x,y in zip(before,after):
        assert not torch.allclose(x[:,KNOWN:],y[:,KNOWN:])


def test_p2_native_called_once():
    b,r,h,t,cr,ce,g=setup()
    p2(b,r,h,t,cr,ce,g)
    assert (b.block_calls,b.self_attn.calls,b.cross_attn.calls,b.ffn_calls) == (1,1,1,1)


def test_p2_pre_vs_post_is_real_difference():
    b,r,h,t,cr,ce,g=setup(); activate(r.output)
    before=p2(b,r,h,t,cr,ce,g)
    after=post_control(b,r,h,t,cr,ce,g)
    assert not torch.allclose(before,after)


def test_p2_gradients_through_frozen_block():
    b,r,h,t,cr,ce,g=setup();b.requires_grad_(False)
    ce=ce.requires_grad_();opt=torch.optim.SGD(r.parameters(),lr=0.02)
    clean=torch.randn_like(h); noise=torch.randn_like(h); sigma=0.4
    noisy=(1-sigma)*clean+sigma*noise; target=noise-clean
    # This target matches FM algebra only; tiny tokens are not real VAE latents.
    for step in range(2):
        opt.zero_grad(set_to_none=True);ce.grad=None
        pred=p2(b,r,noisy,t,cr,ce,g)
        F.mse_loss(pred[:,KNOWN:],target[:,KNOWN:]).backward()
        assert has_grad(r.output.weight)
        if step==1:
            assert has_grad(r.value.weight)
            assert has_grad(r.key.weight)
            assert ce.grad is not None and float(ce.grad.abs().sum())>0
        opt.step()
    assert all(p.grad is None for p in b.parameters())


def test_expert_zero_source():
    e=SourceExpert();activate(e.output)
    u=torch.randn(1,12,D)
    assert torch.count_nonzero(e(u,torch.zeros_like(u)))==0


def test_expert_source_response():
    e=SourceExpert();activate(e.output)
    u=torch.randn(1,12,D); s=torch.randn_like(u)
    assert not torch.allclose(e(u,s),e(u,torch.randn_like(s)))


def test_expert_parameter_match():
    e1,e2,mix=SourceExpert(),SourceExpert(),MixedExpert()
    count=lambda m:sum(p.numel() for p in m.parameters())
    assert count(e1)+count(e2)==count(mix)==D*D


def test_expert_staged_gradients():
    e=SourceExpert(); opt=torch.optim.SGD(e.parameters(),lr=0.1)
    u=torch.randn(1,12,D);s=torch.randn_like(u);target=torch.randn_like(u)
    F.mse_loss(e(u,s),target).backward()
    assert has_grad(e.output.weight)
    assert not has_grad(e.source.weight)
    opt.step();opt.zero_grad(set_to_none=True)
    F.mse_loss(e(u,s),target).backward()
    assert all(has_grad(x.weight) for x in [e.state,e.source,e.output])


def test_p4_zero_matches_native():
    b,r,h,t,cr,ce,g=setup(); ep,et=SourceExpert(),SourceExpert()
    close(p4(b,r,ep,et,h,t,cr,ce,g), b(h,t))


def test_p4_called_once_and_order():
    b,r,h,t,cr,ce,g=setup();ep,et=SourceExpert(),SourceExpert();activate(r.output)
    seen=[]
    handle=b.cross_attn.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
    out=p4(b,r,ep,et,h,t,cr,ce,g);handle.remove()
    assert out.shape==h.shape
    assert (b.self_attn.calls,b.cross_attn.calls,b.ffn_calls)==(1,1,1)
    hs=b.self_step(h); rr=r(hs,cr,ce,b,g)
    close(seen[0],b.norm3(hs+rr.write))


def test_save_restore_reference():
    b,r,h,t,cr,ce,g=setup();activate(r.output)
    expected=p2(b,r,h,t,cr,ce,g)
    b2=TinyNativeBlock();b2.load_state_dict(copy.deepcopy(b.state_dict()))
    r2=PhysicsReader(b2.cross_attn);r2.load_state_dict(copy.deepcopy(r.state_dict()))
    close(expected,p2(b2,r2,h,t,cr,ce,g))


def test_sdpa_true_means_allowed():
    q=torch.zeros(1,1,1,4);k=torch.zeros(1,1,2,4)
    v=torch.tensor([[[[1.,2.,3.,4.],[20.,30.,40.,50.]]]])
    result=F.scaled_dot_product_attention(q,k,v,attn_mask=torch.tensor([[[[True,False]]]]),dropout_p=0.0)
    close(result,v[:,:,:1])


def test_configs_consistent():
    paths=sorted((ROOT/'configs').glob('*.yaml'))
    assert len(paths)==12
    for p in paths:
        cfg=yaml.safe_load(p.read_text())
        every=cfg['model']['condition_every']
        assert cfg['model']['sites']==[i for i in range(30) if (i+1)%every==0]
        assert cfg['loss']['extra_teacher_loss'] is False
        assert cfg['loss']['intervention_learning'] is False
        assert cfg['inference']['best_of_n']==1
        assert cfg['evaluation']['actual_results'] is None
        assert cfg['paths']['wan_checkpoint'] is None
        assert cfg['train']['updates']==400
        assert cfg['native_lora']['self_ffn_rank']==cfg['native_lora']['self_ffn_alpha']
        v=cfg['variant']
        if v.startswith('p2'):
            assert not cfg['physics']['state_value'] and cfg['physics']['direct_write']
        if cfg['experts']['enabled']:
            width=cfg['model']['width'];inner=cfg['experts']['inner_width']
            factor=6 if cfg['experts']['type']=='two_source_conditioned' else 4
            assert factor*width*inner==width*width


def main() -> int:
    tests: list[tuple[str,Callable[[],None]]] = sorted((name,fn) for name,fn in globals().items() if name.startswith('test_') and callable(fn))
    results=[]
    for idx,(name,fn) in enumerate(tests):
        torch.manual_seed(20261110+idx)
        try:
            fn();results.append({'name':name,'passed':True})
            print('PASS',name)
        except Exception as exc:
            results.append({'name':name,'passed':False,'error':repr(exc),'traceback':traceback.format_exc()})
            print('FAIL',name,repr(exc))
    report={
        'scope':'Synthetic CPU reference math and autograd checks; NOT real Wan integration, project training, or video-quality evaluation',
        'python':platform.python_version(),'torch':torch.__version__,'device':'cpu',
        'passed':sum(x['passed'] for x in results),'total':len(results),'all_passed':all(x['passed'] for x in results),
        'real_wan_integration_run':False,'project_training_run':False,'pc_sa_joint_measured':False,'checks':results,
    }
    out=Path(__file__).with_name('reference_math_checks_results.json')
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f"{report['passed']}/{report['total']} passed; report: {out}")
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    sys.exit(main())
