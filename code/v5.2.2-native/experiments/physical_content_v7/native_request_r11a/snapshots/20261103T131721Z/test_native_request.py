"""Real Wan phase transition, accumulated updates/resume and R10A equivalence."""
from pathlib import Path
import copy,importlib.util,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample
from physical_content.supervision import generation_loss
from physical_content.common import load_config
from physical_content.train import save,restore
from physical_scaling.schedule import learning_rates,auxiliary_coefficient

HERE=Path(__file__).resolve().parent
REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/snapshots/20261101T034557Z'
spec=importlib.util.spec_from_file_location('physical_content.r10_reference',REFERENCE/'physical_content/model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
torch.set_num_threads(2)


def tiny_wan():
    package=types.ModuleType('_request_test_wan');package.__path__=[]
    attention=types.ModuleType('_request_test_wan.attention')
    def cpu_attention(q,k,v,**kwargs):
        return F.scaled_dot_product_attention(q.transpose(1,2).float(),k.transpose(1,2).float(),
            v.transpose(1,2).float()).transpose(1,2).to(v.dtype)
    attention.flash_attention=cpu_attention
    module=sys.modules.get('_request_test_wan.model')
    if module is None:
        sys.modules['_request_test_wan']=package;sys.modules['_request_test_wan.attention']=attention
        spec=importlib.util.spec_from_file_location('_request_test_wan.model','/home/liuzhirui/model/Wan2.2/wan/modules/model.py')
        module=importlib.util.module_from_spec(spec);sys.modules['_request_test_wan.model']=module;spec.loader.exec_module(module)
    wan=module.WanModel(model_type='ti2v',patch_size=(1,2,2),text_len=5,in_dim=2,out_dim=2,
        dim=16,ffn_dim=32,freq_dim=8,text_dim=16,num_heads=2,num_layers=2)
    with torch.no_grad():wan.head.head.weight.normal_(std=.1)
    return wan


class NativeRequestTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        interface=dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1))
        torch.save(interface,self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_native_request_r11a_staged',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,condition_every=2,lora_rank=2),
            paths=dict(interface=str(self.path/'interface.pt')),stages=dict(interface=128,joint=272),
            loss=dict(generation=dict(tail_weight=.2,tail_fraction=.25)),
            train=dict(gradient_checkpointing=True,seed=42,keep_steps=[129],peak_lrs=dict(
                enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)))
        self.wan=tiny_wan()
        self.evidence=dict(words=torch.randn(6,32),word_groups=torch.arange(6)//3,
            request_words=torch.randn(5,32),native_words=torch.randn(6,16)*.1)
        self.clean=torch.randn(1,2,3,2,4);self.noise=torch.randn_like(self.clean);self.text=torch.randn(1,4,16)
        self.sample=dict(latent=self.clean,first=self.clean[:,:,:1],times=torch.tensor([[0.,1.,2.]]),valid=torch.ones(2,4))
        self.sigma=torch.tensor(.6)

    def model(self,cls=PhysicalModel,variant='physical_content'):
        cfg=copy.deepcopy(self.cfg);cfg['variant']=variant;torch.manual_seed(42)
        return cls(cfg,device='cpu',wan=copy.deepcopy(self.wan))

    def full(self,model,enabled=True):
        c,g=model.enhancement(self.evidence)
        x=(1-self.sigma)*self.clean+self.sigma*self.noise
        kwargs={} if isinstance(model,old.PhysicalModel) else dict(physical_enabled=enabled)
        return model.velocity(x,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def test_r10a_parameter_and_full_function_preserved(self):
        ref=self.model(old.PhysicalModel);new=self.model()
        self.assertEqual(ref.learned_names,new.learned_names)
        for name,value in ref.state_dict().items():torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
        with torch.no_grad():
            for name,p in new.named_parameters():
                if name in new.learned_names:p.add_(torch.randn_like(p)*.005)
            ref.load_state_dict(new.state_dict())
        torch.testing.assert_close(self.full(new),self.full(ref),rtol=0,atol=0)
        for block in new.wan.blocks:
            self.assertFalse(any(p.requires_grad for p in block.cross_attn.parameters()))

    def test_interface_then_joint_gradients_without_unfreezing_foundations(self):
        model=self.model()
        with torch.no_grad():model.fusions['1'].output.weight.normal_(std=.05)
        self.assertEqual(model.set_phase(127),'interface')
        self.full(model).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        self.assertGreater(float(model.enhancement.execution_project.weight.grad.norm()),0.)
        self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()),0.)
        model.zero_grad(set_to_none=True)
        self.assertEqual(model.set_phase(128),'joint')
        self.full(model).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.wan.parameters() if p.grad is not None),0.)
        self.assertGreater(float(model.enhancement.execution_project.weight.grad.norm()),0.)
        self.assertTrue(model.wan.blocks[0].self_attn.q.lora_b.requires_grad)
        for name,p in model.wan.named_parameters():
            if not name.endswith(('lora_a','lora_b')):
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)

    def test_accumulated_optimizer_matches_r10a_across_phase_boundary(self):
        model=self.model();reference=self.model(old.PhysicalModel)
        with torch.no_grad():model.fusions['1'].output.weight.normal_(std=.05)
        reference.load_state_dict(model.state_dict())
        opt=torch.optim.AdamW(model.optim_groups(),foreach=False)
        base=torch.optim.AdamW(reference.optim_groups(),foreach=False)
        physical_before=model.enhancement.execution_project.weight.detach().clone()
        for step in (127,128,129):
            model.set_phase(step);reference.set_phase(step)
            opt.zero_grad(set_to_none=True);base.zero_grad(set_to_none=True)
            for micro in range(2):
                noise=self.noise+micro*.03
                metrics=backward_sample(model,self.evidence,self.sample,self.text,self.sigma,noise,.03,2)
                self.assertIn('generation',metrics)
                self.assertNotIn('native_generation',metrics)
                x=(1-self.sigma)*self.clean+self.sigma*noise
                content,groups=reference.enhancement(self.evidence)
                full,_=reference.velocity(x,self.sample['first'],self.text,self.sigma,content,self.sample['times'],groups)
                full_loss,_=generation_loss(full,noise,self.clean,self.sample['valid'],**self.cfg['loss']['generation'])
                auxiliary=reference.enhancement.content_loss(content,self.evidence)
                ((full_loss+.03*auxiliary)/2).backward()
            ours=dict(model.named_parameters());theirs=dict(reference.named_parameters())
            for name,p in theirs.items():
                if p.grad is not None:torch.testing.assert_close(ours[name].grad,p.grad,rtol=0,atol=0)
            for optimizer in (opt,base):
                for group in optimizer.param_groups:torch.nn.utils.clip_grad_norm_(group['params'],1.)
                optimizer.step()
            for name,p in theirs.items():torch.testing.assert_close(ours[name],p,rtol=0,atol=0)
        self.assertGreater(float((model.enhancement.execution_project.weight-physical_before).detach().abs().max()),0.)

    def test_checkpoint_resume_and_no_c_same_function(self):
        model=self.model();model.set_phase(128);opt=torch.optim.AdamW(model.optim_groups(),foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        opt.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]):save(model,opt,self.cfg,self.path,129,516)
        restored=self.model();other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with patch('torch.cuda.set_rng_state_all'):self.assertEqual(restore(restored,other,self.path/'step0129.pt',self.cfg),(129,516))
        torch.testing.assert_close(self.full(model),self.full(restored),rtol=0,atol=0)
        torch.testing.assert_close(self.full(model,False),self.full(restored,False),rtol=0,atol=0)
        for group,other_group in zip(opt.param_groups,other.param_groups):
            for p,q in zip(group['params'],other_group['params']):
                if p in opt.state:torch.testing.assert_close(opt.state[p]['exp_avg'],other.state[q]['exp_avg'],rtol=0,atol=0)

    def test_anchor_routing_keeps_complete_execution(self):
        model=self.model()
        with torch.no_grad():
            model.enhancement.project.weight.normal_(std=.03)
            model.enhancement.execution_project.weight.normal_(std=.03)
        full,g=model.enhancement(self.evidence)
        changed,cg=model.enhancement(self.evidence,anchor_routing=True)
        anchor,_=model.enhancement(self.evidence,anchor_only=True)
        torch.testing.assert_close(full['execution'],changed['execution'],rtol=0,atol=0)
        torch.testing.assert_close(changed['routing'],anchor['routing'],rtol=0,atol=0)
        self.assertGreater(float((full['routing']-changed['routing']).detach().abs().max()),0.)
        self.assertTrue(torch.equal(g,cg))

    def test_declared_schedule_and_losses_match_r10a(self):
        import yaml
        cfg=load_config(HERE/'configs/physical_content.yaml')
        oldcfg=yaml.safe_load((REFERENCE/'configs/physical_content.yaml').read_text())
        self.assertEqual(cfg['stages'],oldcfg['stages'])
        for step in range(400):
            actual=learning_rates(cfg,step);before=learning_rates(oldcfg,step)
            self.assertEqual(actual,before)
            self.assertEqual(auxiliary_coefficient(cfg,step),auxiliary_coefficient(oldcfg,step))
        self.assertEqual(cfg['loss'],oldcfg['loss'])
        self.assertEqual(cfg['train']['validation_steps'],[])
        self.assertEqual(cfg['data_revision'],'qwen38_wisa_reviewed_v1')
        self.assertIn('anchor_routing',cfg['evaluation_budget']['control_modes'])


if __name__=='__main__':unittest.main()
