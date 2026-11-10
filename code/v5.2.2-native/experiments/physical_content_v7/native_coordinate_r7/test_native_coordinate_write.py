"""Check initial comparability, generation credit, controls and resume ownership."""
from pathlib import Path
import copy
import importlib.util
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement,PhysicalModel,Fusion
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
        split=lambda v:v.unflatten(-1,(2,8)).transpose(1,2)
        y=F.scaled_dot_product_attention(split(self.norm_q(self.q(x))),
            split(self.norm_k(self.k(context))),split(self.v(context)))
        return self.o(y.transpose(1,2).flatten(2))


class NativeCoordinateWriteTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.cfg={'variant':'physical_content','model':{'qwen_width':32,'width':16,'q_heads':4,
            'q_tokens':8,'rounds':2,'grounded_read_heads':2,'content_projection_init_scale':.1,
            'content_projection_trainable':False,'native_coordinate_update_trainable':True},
            'stages':{'interface':16,'joint':208},'train':{'gradient_checkpointing':True,
                'peak_lrs':dict(enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)}}
        self.interface={'mean':torch.randn(32),'scale':torch.rand(32)+.5,
            'weight':torch.randn(16,32)*.05,'bias':torch.randn(16)*.01,'target_scale':torch.tensor(.1)}
        self.evidence={'words':torch.randn(6,32),'word_groups':torch.arange(6)//3,
            'memory':torch.randn(1,7,32),'memory_mask':torch.ones(1,7,dtype=torch.bool)}

    def holder(self):
        result=PhysicalModel.__new__(PhysicalModel);nn.Module.__init__(result)
        result.cfg=self.cfg;result.wan=nn.Module();result.fusions=nn.ModuleDict()
        result.enhancement=Enhancement(self.cfg,self.interface)
        result.learned_names=tuple(name for name,p in result.named_parameters() if p.requires_grad)
        return result

    def test_initial_output_and_random_stream_match_r6(self):
        torch.manual_seed(42);old=reference.Enhancement(self.cfg,self.interface).eval()
        previous_rng=torch.get_rng_state().clone()
        torch.manual_seed(42);new=Enhancement(self.cfg,self.interface).eval()
        self.assertTrue(torch.equal(torch.get_rng_state(),previous_rng))
        self.assertEqual(set(new.state_dict())-set(old.state_dict()),{'project.native_update'})
        for name,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(new(self.evidence)[0],old(self.evidence)[0],rtol=0,atol=0)

    def test_generation_gradient_reaches_q_and_native_update_and_anchor_control_is_pure(self):
        model=Enhancement(self.cfg,self.interface)
        attention=Attention().requires_grad_(False);fusion=Fusion(attention,.25)
        c,groups=model(self.evidence)
        x=torch.randn(1,4,16);context=torch.randn(1,5,16)
        output,_=fusion(x,context,c,torch.tensor([[0.,1.]]),(2,1,2),attention,groups)
        F.mse_loss(output[:,2:],torch.randn_like(output[:,2:])).backward()
        for parameter in [model.seed,model.project.native_update]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0)
        self.assertIsNone(model.project.weight.grad)
        with torch.no_grad():
            model.project.native_update.fill_(.02)
            model.project.weight.zero_()
            actual,_=model(self.evidence)
            words=(self.evidence['words']-model.anchor_mean)/model.anchor_scale
            anchors,_,_=resample_words(words,self.evidence['word_groups'],8)
            expected=F.linear(anchors,model.anchor_weight,model.anchor_bias)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)

    def test_optimizer_and_checkpoint_keep_reference_map_and_resume_learned_transform(self):
        model=self.holder();optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        fixed=model.enhancement.project.weight.clone()
        def update(current,opt):
            opt.zero_grad(set_to_none=True)
            c,_=current.enhancement(self.evidence);c.square().mean().backward();opt.step()
        model.set_phase(0);update(model,optimizer)
        self.assertGreater(float(model.enhancement.project.native_update.detach().norm()),0)
        torch.testing.assert_close(model.enhancement.project.weight,fixed,rtol=0,atol=0)
        saved=dict(model={n:p.detach().clone() for n,p in model.named_parameters() if n in model.learned_names},
            constants={n:v.clone() for n,v in model.constant_state().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()))
        self.assertIn('enhancement.project.native_update',saved['model'])
        self.assertIn('enhancement.project.weight',saved['constants'])
        restored=self.holder();other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with torch.no_grad():
            named=dict(restored.named_parameters());buffers=dict(restored.named_buffers())
            for name,value in saved['model'].items():named[name].copy_(value)
            for name,value in saved['constants'].items():buffers[name].copy_(value)
        other.load_state_dict(saved['optimizer'])
        model.set_phase(16);restored.set_phase(16)
        update(model,optimizer);update(restored,other)
        torch.testing.assert_close(model.enhancement(self.evidence)[0],restored.enhancement(self.evidence)[0],rtol=0,atol=0)
        torch.testing.assert_close(restored.enhancement.project.weight,fixed,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
