from pathlib import Path
import copy
import json
import tempfile
import unittest
from types import SimpleNamespace
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT.parents[1]))
import torch
from torch import nn
from torch.nn import functional as F
from physical_training.common import load_config,is_allowed_gpu,audit
from physical_training.model import Enhancement,Fusion,semantic_ce,PhysicalModel
from physical_training.supervision import (association_target,grouped_probabilities,group_log_prior,
    slot_groups,association_kl,write_support,write_regression,flow_matching,resample_maps)


def config(variant):
    cfg = load_config(ROOT/'configs'/f'{variant}.yaml')
    cfg['model'].update(width=16,qwen_width=24,q_heads=4,q_tokens=128,lora_rank=4)
    return cfg


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim,self.num_heads,self.head_dim = 16,4,4
        self.q,self.k,self.v,self.o = [nn.Linear(16,16) for _ in range(4)]
        self.norm_q,self.norm_k = nn.LayerNorm(16),nn.LayerNorm(16)

    def forward(self,x,context):
        split = lambda v: v.reshape(1,-1,4,4).transpose(1,2)
        q,k,v = split(self.norm_q(self.q(x))),split(self.norm_k(self.k(context))),split(self.v(context))
        return self.o(F.scaled_dot_product_attention(q,k.to(q),v.to(q)).transpose(1,2).flatten(2))


class FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__();self.embed_tokens = nn.Embedding(30,24);self.layer=nn.Linear(24,24)
    def forward(self,inputs_embeds,**kwargs):
        return SimpleNamespace(last_hidden_state=self.layer(inputs_embeds.cumsum(1)))


class FakeTokenizer:
    eos_token='!'
    def apply_chat_template(self,*a,**kw):return 'question'
    def encode(self,text,**kwargs):return [ord(c)%30 for c in text]


class SelfAttention(Attention):
    def forward(self,x,*args):
        return super().forward(x,x)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=SelfAttention();self.cross_attn=Attention()
        self.norm1=nn.LayerNorm(16);self.norm2=nn.LayerNorm(16);self.norm3=nn.LayerNorm(16)
        self.ffn=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))
        self.modulation=nn.Parameter(torch.randn(1,6,16)*.1)
    def forward(self,x,mod,lengths,grid,freqs,context,*args):
        e=(self.modulation.unsqueeze(0)+mod).chunk(6,dim=2)
        y=self.self_attn(self.norm1(x).float()*(1+e[1].squeeze(2))+e[0].squeeze(2))
        x=x+y*e[2].squeeze(2)
        x=x+self.cross_attn(self.norm3(x),context)
        return x+self.ffn(self.norm2(x).float()*(1+e[4].squeeze(2))+e[3].squeeze(2))*e[5].squeeze(2)


class FakeHead(nn.Linear):
    def forward(self,x,e):return super().forward(x)


class SmallWan(nn.Module):
    def __init__(self):
        super().__init__();self.dim=16;self.freq_dim=8;self.text_len=5
        self.freqs=torch.zeros(4)
        self.patch_embedding=nn.Conv3d(2,16,(1,2,2),stride=(1,2,2))
        self.time_embedding=nn.Sequential(nn.Linear(8,16),nn.SiLU(),nn.Linear(16,16))
        self.time_projection=nn.Linear(16,96)
        self.text_embedding=nn.Linear(24,16)
        self.blocks=nn.ModuleList([Block() for _ in range(3)])
        self.head=FakeHead(16,8)
    def unpatchify(self,x,grid):
        t,h,w=grid[0].tolist()
        return [x[0].reshape(t,h,w,1,2,2,2).permute(6,0,3,1,4,2,5).reshape(2,t,h*2,w*2)]


class TestPhysical(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7);torch.set_num_threads(2)

    def test_gpu_policy_rejects_h100_and_other_cards(self):
        self.assertTrue(is_allowed_gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition'))
        self.assertTrue(is_allowed_gpu('RTX 6000 Blackwell'))
        for name in ['NVIDIA H100 80GB HBM3','RTX 6000 Ada Generation','A100','H100 RTX PRO 6000']:
            self.assertFalse(is_allowed_gpu(name))

    def test_three_configs_no_initializer_and_single_gpu(self):
        import yaml
        for variant in ['v5_ce','v6_write','v6_assoc']:
            cfg=load_config(ROOT/'configs'/f'{variant}.yaml')
            self.assertEqual(cfg['verifier']['kind'],'none')
            job=yaml.safe_load((ROOT/'jobs'/f'{variant}_1x96g.yaml').read_text())
            self.assertEqual(job['resources']['slots_per_trial'],1)
            self.assertEqual(job['resources']['resource_pool'],'blk-96g')
            self.assertNotIn('--initialize',job['entrypoint'])
            self.assertFalse(any('PYTHONNOUSERSITE' in x for x in job['environment']['environment_variables']))

    def test_reference_prior_and_group_correspondence(self):
        for n in [1,2,3,4]:
            groups=slot_groups(128,n)
            probabilities=group_log_prior(groups).softmax(-1)[None,None,None]
            grouped=grouped_probabilities(probabilities,groups)
            self.assertTrue(torch.allclose(grouped,torch.full_like(grouped,1/(n+1)),atol=1e-6))
            self.assertEqual(set(groups.tolist()),set(range(n)))

    def test_soft_target_prefers_supported_relation_and_valid_kl(self):
        relation=torch.tensor([[[.8,.1],[.1,.8]]]);bg=torch.tensor([[.1,.1]])
        target=association_target(relation,bg)
        self.assertEqual(target.argmax(-1).tolist(),[[1,2]])
        self.assertTrue(torch.allclose(target.sum(-1),torch.ones(1,2)))
        self.assertLess(float(association_kl(target,target,torch.ones(1,2))),1e-6)
        wrong=target.flip(1).requires_grad_()
        loss=association_kl(wrong,target,torch.ones(1,2));loss.backward()
        self.assertGreater(float(loss.detach()),0);self.assertGreater(float(wrong.grad.abs().sum()),0)

    def test_real_time_interpolation_and_zero_unsupported_write(self):
        maps=torch.tensor([[[[0.]],[[10.]]]])
        out=resample_maps(maps,torch.tensor([0.,10.]),torch.tensor([0.,1.,10.]),1,1)
        self.assertTrue(torch.allclose(out[:,0],torch.tensor([0.,1.,10.])))
        relation=torch.full((1,4,2),.1);bg=torch.full((1,4),.2)
        self.assertEqual(float(write_support(relation,bg).sum()),0)

    def test_first_frame_not_supervised_as_future(self):
        noise=torch.zeros(1,2,3,2,2);clean=torch.zeros_like(noise);pred=torch.zeros_like(noise)
        pred[:,:,0]=999
        self.assertEqual(float(flow_matching(pred,noise,clean,torch.ones(2,2))),0)
        pred[:,:,1:]=2
        self.assertEqual(float(flow_matching(pred,noise,clean,torch.ones(2,2))),4)

    def test_native_text_preserved_at_zero_physical_output(self):
        a=Attention().requires_grad_(False)
        x,txt,c=torch.randn(1,6,16),torch.randn(1,5,16),torch.randn(1,128,16)
        for variant in ['v5_ce','v6_write','v6_assoc']:
            fusion=Fusion(a,variant,rank=4)
            groups=slot_groups(128,3) if variant=='v6_assoc' else None
            with torch.autocast('cpu',dtype=torch.bfloat16):
                ref=a(x,txt)
                result,loss=fusion(x,txt,c,torch.tensor([[0.,.1,.2]]),(3,1,2),a,groups)
            self.assertTrue(torch.allclose(result.float(),ref.float(),atol=1e-5))
            self.assertEqual(float(loss),0)
            self.assertIsNot(fusion.key,a.k)

    def test_assoc_backprop_reaches_c_even_when_write_output_starts_zero(self):
        a=Attention().requires_grad_(False);fusion=Fusion(a,'v6_assoc',rank=4)
        c=torch.randn(1,128,16,requires_grad=True);groups=slot_groups(128,2)
        target={'association':torch.tensor([[[.1,.8,.1],[.1,.1,.8],[.1,.8,.1],[.1,.1,.8]]])}
        with torch.autocast('cpu',dtype=torch.bfloat16):
            _,loss=fusion(torch.randn(1,6,16),torch.randn(1,5,16),c,torch.tensor([[0.,.1,.2]]),(3,1,2),a,groups,target,torch.ones(1,4))
        loss.backward()
        self.assertGreater(float(c.grad.abs().sum()),0)
        self.assertIsNotNone(fusion.key.lora_b.grad)
        self.assertTrue(all(p.grad is None for p in a.parameters()))

    def test_v6_write_has_nonzero_output_gradient_at_initialization(self):
        a=Attention().requires_grad_(False);fusion=Fusion(a,'v6_write',rank=4)
        target={'support':torch.rand(1,4,2),'text_anchors':torch.randn(1,2,16)}
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result,loss=fusion(torch.randn(1,6,16),torch.randn(1,5,16),torch.randn(1,128,16),torch.tensor([[0.,.1,.2]]),(3,1,2),a,None,target,torch.ones(1,4))
        loss.backward()
        self.assertGreater(float(fusion.output.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in a.parameters()))

    def test_q_refinement_uses_two_passes_and_no_future_fields(self):
        for variant in ['v5_ce','v6_assoc']:
            module=Enhancement(config(variant))
            evidence=dict(memory=torch.randn(1,8,24),memory_mask=torch.ones(1,8,dtype=torch.bool),relation_vectors=torch.randn(1,3,24))
            calls=[];handle=module.block.register_forward_hook(lambda *args:calls.append(1))
            c,groups=module(evidence);handle.remove()
            self.assertEqual(c.shape,(1,128,16));self.assertEqual(len(calls),2)
            changed=dict(evidence,latent=torch.randn(1,48,21,30,52),teacher_target=torch.randn(10))
            other,_=module(changed)
            self.assertTrue(torch.equal(c,other))

    def test_ce_reaches_real_c_bridge_not_frozen_language_parameters(self):
        enhance=Enhancement(config('v5_ce'))
        model=SimpleNamespace(enhancement=enhance,bridge=nn.Linear(16,24,bias=False))
        backbone=nn.Module();backbone.model=nn.Module();backbone.model.language_model=FakeLanguage();backbone.lm_head=nn.Linear(24,30)
        backbone.requires_grad_(False)
        knowledge=SimpleNamespace(backbone=backbone,tokenizer=FakeTokenizer())
        evidence=dict(memory=torch.randn(1,8,24),memory_mask=torch.ones(1,8,dtype=torch.bool))
        loss=semantic_ce(model,knowledge,evidence,'response');loss.backward()
        self.assertGreater(float(enhance.seed.grad.abs().sum()),0)
        self.assertGreater(float(model.bridge.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_checkpointed_wan_execution_and_stage_ownership(self):
        for variant in ['v5_ce','v6_write','v6_assoc']:
            cfg=config(variant);model=PhysicalModel(cfg,device='cpu',wan=SmallWan())
            groups=model.optim_groups()
            self.assertEqual(model.set_phase(0),'interface')
            self.assertTrue(all(not p.requires_grad for p in groups[1]['params']))
            self.assertEqual(model.set_phase(64),'joint')
            self.assertTrue(all(p.requires_grad for p in groups[1]['params']))
            evidence=dict(memory=torch.randn(1,8,24),memory_mask=torch.ones(1,8,dtype=torch.bool),relation_vectors=torch.randn(1,2,24))
            with torch.autocast('cpu',dtype=torch.bfloat16):
                c,g=model.enhancement(evidence)
                target=({'support':torch.rand(1,8,2),'text_anchors':torch.randn(1,2,16)} if variant=='v6_write' else
                        {'association':torch.tensor([.1,.8,.1]).expand(1,8,3)} if variant=='v6_assoc' else None)
                result,aux=model.velocity(torch.randn(1,2,3,4,4),torch.randn(1,2,1,4,4),torch.randn(1,5,24),torch.tensor(.5),c,
                                          torch.tensor([[0.,.1,.2]]),g,target,torch.ones(4,4))
                loss=result.square().mean()+aux
            self.assertEqual(result.shape,(1,2,3,4,4));loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in groups[1]['params']))
            self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_teacher_selected_rows_match_native_attention(self):
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention
        from physical_training.teacher import Teacher
        cfg=Qwen3VLTextConfig(hidden_size=16,num_attention_heads=4,num_key_value_heads=2,
                             head_dim=4,num_hidden_layers=2,intermediate_size=32)
        cfg._attn_implementation='eager'
        positions=[[8,9],[10,11]]
        hidden=torch.randn(1,12,16)
        expected=[]
        class NativeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model=nn.Module()
                self.language_model.layers=nn.ModuleList()
                for i in range(2):
                    layer=nn.Module();layer.self_attn=Qwen3VLTextAttention(cfg,i)
                    self.language_model.layers.append(layer)
            def forward(self,**kwargs):
                mask=torch.zeros(12,12).masked_fill(torch.ones(12,12,dtype=torch.bool).triu(1),float('-inf'))[None,None]
                for layer in self.language_model.layers:
                    _,attn=layer.self_attn(hidden_states=hidden,position_embeddings=(torch.ones(1,12,4),torch.zeros(1,12,4)),attention_mask=mask)
                    maps=torch.stack([attn[0,:,p,:8].mean((0,1)) for p in positions])
                    expected.append(maps/maps.sum(-1,keepdim=True))
                return SimpleNamespace(last_hidden_state=hidden)
        model=NativeModel().requires_grad_(False).eval()
        ids=torch.tensor([[7]*8+[1]*4])
        teacher=Teacher.__new__(Teacher)
        teacher.knowledge=SimpleNamespace(backbone=SimpleNamespace(model=model,config=SimpleNamespace(video_token_id=7)),merge=2,temporal=2,
            _inputs=lambda *a,**kw: ({'input_ids':ids,'video_grid_thw':torch.tensor([[2,4,4]])},positions))
        maps,stamps=teacher._attention_maps(torch.zeros(4,3,32,32),torch.tensor([0.,1.,2.,3.]),['contact causes response','support maintains position'])
        self.assertEqual(maps.shape,(2,2,2,2))
        self.assertTrue(torch.allclose(maps.flatten(1),torch.stack(expected).mean(0),atol=1e-6))
        self.assertEqual(stamps.tolist(),[.5,2.5])


if __name__=='__main__':
    unittest.main(verbosity=2)
