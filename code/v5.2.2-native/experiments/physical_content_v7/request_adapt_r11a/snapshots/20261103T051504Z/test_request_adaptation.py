"""Check the real adapter path, frozen physical initializer and ablation semantics."""
from pathlib import Path
import copy
import importlib.util
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import PhysicalModel,NativeTextAdapter,set_text_adapter_enabled
from physical_content.train import save,restore
from physical_content.common import load_config
from physical_scaling.schedule import learning_rates,auxiliary_coefficient
from torch.utils.checkpoint import checkpoint

REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/snapshots/20261101T034557Z'
spec=importlib.util.spec_from_file_location('physical_content.r10_reference',REFERENCE/'physical_content/model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.set_num_threads(2)


class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.dim=16;self.num_heads=2;self.head_dim=8
        for name in ('q','k','v','o'):setattr(self,name,nn.Linear(16,16,bias=False))
        self.norm_q=nn.RMSNorm(16);self.norm_k=nn.RMSNorm(16)

    def forward(self,x,context,context_lens=None):
        split=lambda v:v.unflatten(-1,(2,8)).transpose(1,2)
        q=split(self.norm_q(self.q(x)));k=split(self.norm_k(self.k(context)));v=split(self.v(context))
        return self.o(F.scaled_dot_product_attention(q,k,v).transpose(1,2).flatten(2))


class Block(nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=Attention();self.cross_attn=Attention()
        self.ffn=nn.Sequential(nn.Linear(16,32),nn.GELU(),nn.Linear(32,16))


class Wan(nn.Module):
    def __init__(self):
        super().__init__();self.blocks=nn.ModuleList([Block(),Block()])


class RequestAdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        interface=dict(mean=torch.randn(32),scale=torch.rand(32)+.5,
            weight=torch.randn(16,32)*.05,bias=torch.randn(16)*.01,target_scale=torch.tensor(.1))
        torch.save(interface,self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_request_adapt_r11a',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,
                       condition_every=2,lora_rank=2,native_text_lora_rank=2),
            paths=dict(interface=str(self.path/'interface.pt')),
            stages=dict(interface=128,joint=272),
            train=dict(gradient_checkpointing=True,seed=42,keep_steps=[1],peak_lrs=dict(
                enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)))
        self.wan=Wan()
        self.evidence=dict(words=torch.randn(6,32),word_groups=torch.arange(6)//3,
                           request_words=torch.randn(5,32),native_words=torch.randn(6,16)*.1)
        self.x=torch.randn(1,4,16);self.context=torch.randn(1,5,16)

    def pair(self):
        torch.manual_seed(42)
        reference=old.PhysicalModel(self.cfg,device='cpu',wan=copy.deepcopy(self.wan))
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42)
        new=PhysicalModel(self.cfg,device='cpu',wan=copy.deepcopy(self.wan))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        return reference,new

    def fused(self,model):
        c,g=model.enhancement(self.evidence)
        def forward(x,c):
            return model.fusions['1'](x,self.context,c,torch.tensor([[0.,1.]]),(2,1,2),model.wan.blocks[1].cross_attn,g)[0]
        return checkpoint(forward,self.x,c,use_reentrant=False)

    def test_exact_existing_initializer_and_initial_outputs(self):
        ref,new=self.pair()
        oldparams=dict(ref.named_parameters());newparams=dict(new.named_parameters())
        for name in ref.learned_names:
            torch.testing.assert_close(oldparams[name],newparams[name],rtol=0,atol=0)
        added=set(new.learned_names)-set(ref.learned_names)
        expected={f'wan.blocks.{i}.cross_attn.{name}.lora_{part}' for i in range(2)
                  for name in ('k','v','o') for part in ('a','b')}
        self.assertEqual(added,expected)
        torch.testing.assert_close(self.fused(ref),self.fused(new),rtol=0,atol=0)
        torch.testing.assert_close(ref.wan.blocks[0].cross_attn(self.x,self.context),
                                   new.wan.blocks[0].cross_attn(self.x,self.context),rtol=0,atol=0)

    def test_phase_and_optimizer_ownership(self):
        _,model=self.pair()
        owned={id(p) for g in model.optim_groups() if g['name']=='wan_lora' for p in g['params']}
        for step,joint in [(0,False),(127,False),(128,True),(399,True)]:
            model.set_phase(step)
            for block in model.wan.blocks:
                for name in ('k','v','o'):
                    layer=getattr(block.cross_attn,name)
                    self.assertIsInstance(layer,NativeTextAdapter)
                    for p in (layer.lora_a,layer.lora_b):
                        self.assertEqual(p.requires_grad,joint);self.assertIn(id(p),owned)
                    self.assertFalse(any(p.requires_grad for p in layer.base.parameters()))
                self.assertFalse(block.cross_attn.q.weight.requires_grad)

    def test_actual_fusion_and_native_paths_receive_generation_gradients(self):
        _,model=self.pair();model.set_phase(128)
        optimizer=torch.optim.SGD(model.optim_groups(),lr=.01)
        # Open the pre-existing physical output, so text and physical paths coexist.
        with torch.no_grad():model.fusions['1'].output.weight.normal_(std=.03)
        target=torch.randn_like(self.x)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss=F.mse_loss(self.fused(model),target)
            loss+=F.mse_loss(model.wan.blocks[0].cross_attn(self.x,self.context),target)
            loss.backward()
            for i in range(2):
                for name in ('k','v','o'):
                    p=getattr(model.wan.blocks[i].cross_attn,name).lora_b
                    self.assertGreater(float(p.grad.norm()),0.)
            self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()),0.)
            optimizer.step()

    def test_disable_changes_only_native_text_and_restores_exact_output(self):
        ref,model=self.pair()
        with torch.no_grad():
            for block in model.wan.blocks:
                for name in ('k','v','o'):getattr(block.cross_attn,name).lora_b.normal_(std=.1)
        c=model.enhancement(self.evidence)[0]
        before=self.fused(model).detach()
        set_text_adapter_enabled(model,False)
        torch.testing.assert_close(self.fused(model),self.fused(ref),rtol=0,atol=0)
        torch.testing.assert_close(model.wan.blocks[0].cross_attn(self.x,self.context),
                                   ref.wan.blocks[0].cross_attn(self.x,self.context),rtol=0,atol=0)
        for key,value in model.enhancement(self.evidence)[0].items():
            torch.testing.assert_close(value,c[key],rtol=0,atol=0)
        set_text_adapter_enabled(model,True)
        torch.testing.assert_close(self.fused(model),before,rtol=0,atol=0)
        self.assertGreater(float((before-self.fused(ref)).abs().max()),1e-5)

    def test_checkpoint_and_optimizer_resume_owns_new_adapters(self):
        _,model=self.pair();model.set_phase(128)
        optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        self.fused(model).square().mean().backward();optimizer.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]):save(model,optimizer,self.cfg,self.path,1,4)
        restored=PhysicalModel(self.cfg,device='cpu',wan=copy.deepcopy(self.wan))
        other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with patch('torch.cuda.set_rng_state_all'):
            self.assertEqual(restore(restored,other,self.path/'step0001.pt',self.cfg),(1,4))
        for name in model.learned_names:
            torch.testing.assert_close(dict(model.named_parameters())[name],dict(restored.named_parameters())[name],rtol=0,atol=0)
        p=model.wan.blocks[1].cross_attn.o.lora_b;q=restored.wan.blocks[1].cross_attn.o.lora_b
        torch.testing.assert_close(optimizer.state[p]['exp_avg'],other.state[q]['exp_avg'],rtol=0,atol=0)

    def test_frozen_budget_schedule_and_data_match(self):
        import yaml
        cfg=load_config(HERE/'configs/physical_content.yaml')
        ref=yaml.safe_load((REFERENCE/'configs/physical_content.yaml').read_text())
        for key in ('stages','train','loss','inference'):self.assertEqual(cfg[key],ref[key])
        self.assertEqual(cfg['paths']['cache'],ref['paths']['cache'])
        self.assertEqual(cfg['paths']['content_cache'],ref['paths']['content_cache'])
        self.assertEqual(cfg['paths']['interface'],ref['paths']['interface'])
        self.assertEqual(cfg['evaluation_budget']['primary_cases'],40)
        self.assertEqual(cfg['evaluation_budget']['extended_evaluation']['cases'],60)
        self.assertIn('text_adapter_off',cfg['evaluation_budget']['control_modes'])


if __name__=='__main__':unittest.main()
