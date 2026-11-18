"""Check the R1-preserving reader transfer, real fusion gradients and schedule."""
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
from physical_content.common import load_config
from physical_scaling.schedule import learning_rates,auxiliary_coefficient
import yaml

torch.set_num_threads(2)
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
REFERENCE=ROOT/'snapshots/20261030T141432Z/physical_content/model.py'
spec=importlib.util.spec_from_file_location('physical_content.r1_reference',REFERENCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


class Attention(nn.Module):
    def __init__(self):
        super().__init__();self.dim=16;self.num_heads=2;self.head_dim=8
        for name in ['q','k','v','o']:setattr(self,name,nn.Linear(16,16,bias=False))
        self.norm_q=nn.RMSNorm(16);self.norm_k=nn.RMSNorm(16)

    def forward(self,x,context):
        return x


class ReaderTransferTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.cfg={'model':{'qwen_width':32,'width':16,'q_heads':4,'q_tokens':8,
            'rounds':2,'grounded_read_heads':2},'train':{'gradient_checkpointing':True}}
        self.interface={'mean':torch.randn(32),'scale':torch.rand(32)+.5,
            'weight':torch.randn(16,32)*.05,'bias':torch.randn(16)*.01,'target_scale':torch.tensor(.1)}
        self.evidence={'words':torch.randn(6,32),'word_groups':torch.arange(6)//3,
            'request_words':torch.randn(5,32),'memory':torch.randn(1,9,32),
            'memory_mask':torch.ones(1,9,dtype=torch.bool)}

    def test_r1_parameter_initialization_rng_and_initial_c_are_preserved(self):
        attention=Attention()
        torch.manual_seed(42)
        old=reference.Enhancement(self.cfg,self.interface);old_fusion=reference.Fusion(attention)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42)
        new=Enhancement(self.cfg,self.interface);new_fusion=Fusion(attention)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for left,right in [(old,new),(old_fusion,new_fusion)]:
            self.assertEqual(set(left.state_dict()),set(right.state_dict()))
            self.assertEqual(set(dict(left.named_parameters())),set(dict(right.named_parameters())))
            for name,value in left.state_dict().items():
                torch.testing.assert_close(value,right.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(new(self.evidence)[0],old(self.evidence)[0],rtol=0,atol=0)
        self.assertTrue(new.project.weight.requires_grad)
        self.assertEqual(float(new_fusion.output.weight.detach().abs().max()),0.)

    def test_reading_request_and_empty_request_with_anchor_control(self):
        model=Enhancement(self.cfg,self.interface)
        with torch.no_grad():model.project.weight.normal_(std=.02)
        original=model(self.evidence)[0]
        prefix=dict(self.evidence,memory=torch.randn_like(self.evidence['memory'])*20)
        torch.testing.assert_close(original,model(prefix)[0],rtol=0,atol=0)
        altered=dict(self.evidence,request_words=self.evidence['request_words'].flip(-1))
        self.assertGreater(float((original-model(altered)[0]).detach().abs().max()),1e-6)
        image_only=dict(self.evidence,request_words=torch.empty(0,32))
        self.assertTrue(torch.isfinite(model(image_only)[0]).all())
        with torch.no_grad():model.project.weight.zero_()
        torch.testing.assert_close(model(self.evidence)[0],model(altered)[0],rtol=0,atol=0)

    def test_zero_initialized_r1_fusion_opens_generation_gradient_path(self):
        model=Enhancement(self.cfg,self.interface)
        attention=Attention().requires_grad_(False);fusion=Fusion(attention)
        optimizer=torch.optim.SGD([*model.parameters(),*fusion.parameters()],lr=.2)
        x=torch.randn(1,4,16);context=torch.randn(1,5,16);target=torch.randn(1,2,16)
        norms=[]
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            content,groups=model(self.evidence)
            output,_=fusion(x,context,content,torch.tensor([[0.,1.]]),(2,1,2),attention,groups)
            F.mse_loss(output[:,2:],target).backward()
            values={name:float(p.grad.norm()) for name,p in
                [('output',fusion.output.weight),('project',model.project.weight),
                 ('seed',model.seed),('query',model.block.cross.q.weight),('value',model.block.cross.v.weight)]}
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            norms.append(values);optimizer.step()
        self.assertGreater(norms[0]['output'],0.)
        self.assertEqual(norms[0]['project'],0.)
        self.assertGreater(norms[1]['project'],0.)
        for name in ['seed','query','value']:self.assertGreater(norms[2][name],0.)

    def test_all_400_update_rates_auxiliary_and_training_recipe_match_r1(self):
        old=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text())
        new=load_config(Path(__file__).parent/'configs/physical_content.yaml')
        self.assertEqual(new['stages'],{'interface':128,'joint':272})
        self.assertEqual(old['loss'],new['loss'])
        self.assertEqual({k:v for k,v in old['train'].items() if k not in ['keep_steps','validation_steps']},
                         {k:v for k,v in new['train'].items() if k not in ['keep_steps','validation_steps']})
        self.assertEqual(old['inference'],new['inference'])
        for step in range(400):
            self.assertEqual(learning_rates(old,step),learning_rates(new,step))
            self.assertEqual(auxiliary_coefficient(old,step),auxiliary_coefficient(new,step))
        self.assertEqual((ROOT/'snapshots/20261030T141432Z/physical_content/train.py').read_bytes(),
                         (Path(__file__).parent/'physical_content/train.py').read_bytes())


if __name__=='__main__':unittest.main()
