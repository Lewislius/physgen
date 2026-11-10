"""Native attention algebra, checkpoint gradients and intentional v12 migration."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_native import FakeAttention,FakeWan,FakeDecoder,sample,grad_size
from test_operator import TextKnowledge
from test_generation_baseline import baseline_model
from physgen_native.attention import RMSNorm
from physgen_native.config import read_config,validate,FULLWIDTH_REVISION
from physgen_native.native_attention import NativeAttentionInterface,NativeAttentionContext
from physgen_native.runtime import owner,initialize_generator,parameter_groups,save_checkpoint,load_checkpoint,restore,schedule
from physgen_native.system import PhysGenNative


def flash_attention(q,k,v,k_lens=None):
    assert k_lens is None
    return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2)


class NativeAttention(nn.Module):
    """CPU counterpart of the native Wan cross-attention projection order."""
    def __init__(self,width=32,heads=4):
        super().__init__();self.dim=width;self.num_heads=heads;self.head_dim=width//heads
        self.q,self.k,self.v,self.o=(nn.Linear(width,width) for _ in range(4))
        self.norm_q,self.norm_k=RMSNorm(width),RMSNorm(width)
    def forward(self,x,context,context_lens):
        reshape=lambda value:value.reshape(x.shape[0],-1,self.num_heads,self.head_dim)
        q=reshape(self.norm_q(self.q(x)));k=reshape(self.norm_k(self.k(context)));v=reshape(self.v(context))
        return self.o(flash_attention(q,k,v,k_lens=context_lens).flatten(2))


class NativeBlock(nn.Module):
    def __init__(self):
        super().__init__();self.self_attn=FakeAttention(32);self.cross_attn=NativeAttention()
        self.norm1=nn.LayerNorm(32);self.norm3=nn.LayerNorm(32);self.norm2=nn.LayerNorm(32)
        self.ffn=nn.Sequential(nn.Linear(32,96),nn.GELU(),nn.Linear(96,32))
    def forward(self,x,e,lengths,grid,freqs,context,context_lens):
        x=x+.1*self.self_attn(self.norm1(x),self.norm1(x))
        x=x+self.cross_attn(self.norm3(x),context,context_lens)
        return x+self.ffn(self.norm2(x))


def fullwidth_model():
    old,_=baseline_model();cfg=copy.deepcopy(old.cfg)
    cfg['revision']=FULLWIDTH_REVISION
    cfg['model'].update(injection='native_cross_attention',attention_width=32)
    del cfg['model']['interface_rank'];validate(cfg)
    wan=FakeWan(cfg);wan.blocks=nn.ModuleList([NativeBlock() for _ in range(3)])
    wan.requires_grad_(False)
    model=PhysGenNative(cfg,wan,knowledge=TextKnowledge(),decoder=FakeDecoder(),device='cpu')
    return model,sample()


class FullwidthAttentionTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(261026)

    def test_native_dimensions_and_no_second_rank_operator(self):
        cfg=read_config(ROOT/'configs/native_fullwidth_attention_wisa3857.yaml')
        self.assertEqual((cfg['model']['attention_width'],cfg['model']['attention_heads']),(3072,24))
        self.assertNotIn('interface_rank',cfg['model'])
        interface=NativeAttentionInterface(cfg,30)
        self.assertEqual(interface.layers,tuple(range(2,30,3)))
        self.assertFalse(any(n.startswith(('attention.','sites.')) for n,_ in interface.named_parameters()))
        self.assertEqual(sum(p.numel() for p in interface.parameters()),596353)
        self.assertEqual([len(cfg['loss'][p]) for p in cfg['stages']],[4,1,4])

    def test_disabled_and_zero_C_preserve_native_output_without_duplicate_bias(self):
        model,_=fullwidth_model();interface=model.generator.interface
        attn=model.generator.wan.blocks[0].cross_attn
        x=torch.randn(1,12,32);text=torch.randn(1,5,32);c=torch.randn(1,3,32)
        original=attn._original_context_forward(x,text,None)
        for strength,tokens in ((0.,c),(1.,torch.zeros_like(c))):
            packet=NativeAttentionContext(text,tokens,interface,(3,2,2),strength,0)
            actual=attn(x,packet,None)
            torch.testing.assert_close(actual,original,atol=0,rtol=0)
        # Calling the original native backbone with normal context remains valid.
        torch.testing.assert_close(attn(x,text,None),original,atol=0,rtol=0)

    def test_single_native_q_projection_fullwidth_message_first_frame_and_soft_intensity(self):
        model,_=fullwidth_model();interface=model.generator.interface
        attn=model.generator.wan.blocks[0].cross_attn
        x=torch.randn(1,12,32);text=torch.randn(1,5,32);c=torch.randn(1,3,32,requires_grad=True)
        original=attn(x,text,None)
        interface.capture=True
        with torch.no_grad():
            interface.intensity[-1].weight.zero_();interface.intensity[-1].bias.fill_(-40)
        packet=NativeAttentionContext(text,c,interface,(3,2,2),1.,0)
        with patch.object(attn.q,'forward',wraps=attn.q.forward) as query:
            actual=attn(x,packet,None)
        self.assertEqual(query.call_count,1)
        torch.testing.assert_close(actual[:,:4],original[:,:4],atol=0,rtol=0)
        self.assertGreater(float((actual[:,4:]-original[:,4:]).detach().square().mean()),1e-5)
        self.assertLess(float(packet.intensity.detach().max()),1e-10)
        actual[:,-1].square().sum().backward()
        self.assertGreater(float(c.grad.abs().sum()),0)
        observation=interface.observations[0]
        self.assertEqual(observation['q_shape'],[1,12,4,8])
        self.assertEqual(observation['c_message_shape'],[1,8,4,8])

    def test_no_forced_low_rank_jacobian_on_C_value_path(self):
        model,_=fullwidth_model();interface=model.generator.interface
        attn=model.generator.wan.blocks[0].cross_attn
        x=torch.randn(1,12,32);text=torch.randn(1,5,32)
        def read(c):
            packet=NativeAttentionContext(text,c[None,None],interface,(3,2,2),1.,0)
            return attn(x,packet,None)[0,-1]
        jacobian=torch.autograd.functional.jacobian(read,torch.randn(32))
        self.assertEqual(int(torch.linalg.matrix_rank(jacobian)),32)

    def test_checkpoint_recomputation_matches_C_and_shared_wan_gradients(self):
        model,data=fullwidth_model();model.train();generator=model.generator
        noisy=torch.randn_like(data['latent']);base=torch.randn(1,5,32)
        results=[]
        for enabled in (False,True):
            model.zero_grad(set_to_none=True);generator.checkpointing=enabled
            c=base.clone().requires_grad_(True)
            result,intensity=generator(noisy,data['first'],data['text'],torch.tensor(.4),c,data['times'],auxiliary=True)
            (result.square().mean()+intensity.mean()).backward()
            gradients={n:p.grad.clone() for n,p in generator.named_parameters() if p.grad is not None}
            results.append((result.detach(),c.grad.clone(),gradients))
        for a,b in zip(results[0][:2],results[1][:2]):torch.testing.assert_close(a,b,atol=0,rtol=0)
        self.assertEqual(set(results[0][2]),set(results[1][2]))
        for n in results[0][2]:torch.testing.assert_close(results[0][2][n],results[1][2][n],atol=0,rtol=0)
        self.assertGreater(float(results[1][1].square().sum()),0)

    def test_generation_and_frozen_verifier_feedback_reach_knowledge(self):
        model,data=fullwidth_model()
        losses,weights,_=model(data,torch.zeros_like(data['text']),0,sigma_override=.2)
        model.backward_losses(losses,weights)
        for module in (model.conditioner,model.generator.interface,model.generator.wan,model.knowledge):
            self.assertGreater(grad_size(module),0)
        model.zero_grad(set_to_none=True)
        losses,weights,metrics=model.feedback_losses(data,2)
        model.backward_losses(losses,weights)
        self.assertGreater(grad_size(model.conditioner)+grad_size(model.knowledge),0)
        self.assertEqual(grad_size(model.critic)+grad_size(model.generator),0)
        self.assertGreater(float(metrics['C_gradient_rms']),0)

    def test_migration_is_explicit_and_rejects_missing_knowledge(self):
        # Use the same native backbone shape for both interface architectures.
        model,_=fullwidth_model();old_cfg=copy.deepcopy(model.cfg)
        old_cfg['revision']='v522_native_generation_baseline_v11'
        old_cfg['model'].update(injection='conditional_operator',attention_width=16,interface_rank=4)
        wan=FakeWan(old_cfg);wan.blocks=nn.ModuleList([NativeBlock() for _ in range(3)]);wan.requires_grad_(False)
        old=PhysGenNative(old_cfg,wan,knowledge=TextKnowledge(),decoder=FakeDecoder(),device='cpu')
        saved={n:p.detach().clone() for n,p in old.named_parameters() if owner(n) is not None}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.pt'
            state=dict(revision=old_cfg['revision'],config=old_cfg,step=16,model=saved)
            torch.save(state,path);report=initialize_generator(model,model.cfg,path)
            self.assertTrue(report['fullwidth_migration'])
            self.assertEqual(report['new_tensors'],['generator.interface.intensity_query'])
            self.assertTrue(report['discarded_tensors'])
            for n,p in model.named_parameters():
                if n in saved and owner(n) not in (None,'critic'):
                    torch.testing.assert_close(p,saved[n],atol=0,rtol=0)
            del saved[next(n for n in saved if n.startswith('knowledge.'))];torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'outside'):initialize_generator(model,model.cfg,path)

    def test_fullwidth_checkpoint_roundtrip_has_no_patched_base_prefixes(self):
        model,_=fullwidth_model();groups=parameter_groups(model,model.cfg)
        optimizer=torch.optim.AdamW(groups);scheduler=schedule(optimizer,model.cfg)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'new.pt';save_checkpoint(model,optimizer,scheduler,model.cfg,1,1,path)
            state=load_checkpoint(path);other,_=fullwidth_model()
            other_opt=torch.optim.AdamW(parameter_groups(other,other.cfg));other_sched=schedule(other_opt,other.cfg)
            self.assertEqual(restore(other,other_opt,other_sched,state,other.cfg),(1,1))
            self.assertIn('generator.wan.blocks.0.cross_attn.q.lora_a',state['model'])
            self.assertFalse(any('_original_context' in n for n in state['model']))


if __name__=='__main__':unittest.main()
