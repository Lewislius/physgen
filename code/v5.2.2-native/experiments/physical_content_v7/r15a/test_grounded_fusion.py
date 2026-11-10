"""Actual Wan integration, parameter ownership and declared intervention checks."""
from pathlib import Path
import copy,importlib.util,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample,execution_alignment_weight
from physical_content.train import save,restore
from condition_controls import MeanContent,mean_with_own_prior
torch.set_num_threads(2)


def tiny_wan():
    name='_r15_test_wan'
    package=types.ModuleType(name);package.__path__=[]
    attention=types.ModuleType(name+'.attention')
    def cpu_attention(q,k,v,**kwargs):
        return F.scaled_dot_product_attention(q.transpose(1,2).float(),k.transpose(1,2).float(),
            v.transpose(1,2).float()).transpose(1,2).to(v.dtype)
    attention.flash_attention=cpu_attention
    module=sys.modules.get(name+'.model')
    if module is None:
        sys.modules[name]=package;sys.modules[name+'.attention']=attention
        spec=importlib.util.spec_from_file_location(name+'.model','/home/liuzhirui/model/Wan2.2/wan/modules/model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    wan=module.WanModel(model_type='ti2v',patch_size=(1,2,2),text_len=5,in_dim=2,out_dim=2,
        dim=16,ffn_dim=32,freq_dim=8,text_dim=16,num_heads=2,num_layers=2)
    with torch.no_grad():wan.head.head.weight.normal_(std=.1)
    return wan


class GroundedFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_grounded_fusion_r15a',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,condition_every=2,lora_rank=2),
            paths=dict(interface=str(self.path/'interface.pt')),stages=dict(interface=128,joint=272),
            loss=dict(generation=dict(tail_weight=.2,tail_fraction=.25)),
            train=dict(seed=42,gradient_checkpointing=True,keep_steps=[129],
                peak_lrs=dict(enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)))
        self.wan=tiny_wan()
        self.evidence=dict(words=torch.randn(6,32),word_groups=torch.arange(6)//3,
            request_words=torch.randn(5,32),native_words=torch.randn(6,16)*.1)
        self.clean=torch.randn(1,2,3,2,4);self.noise=torch.randn_like(self.clean);self.text=torch.randn(1,4,16)
        self.sample=dict(latent=self.clean,first=self.clean[:,:,:1],times=torch.tensor([[0.,1.,2.]]),valid=torch.ones(2,4))
        self.sigma=torch.tensor(.6)

    def model(self):
        torch.manual_seed(42)
        return PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))

    def full(self,model,*,execution_residual=True,**kwargs):
        c,g=model.enhancement(self.evidence,execution_residual=execution_residual)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise
        return model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def activate(self,model):
        with torch.no_grad():
            for fusion in model.fusions.values():
                fusion.output.weight.normal_(std=.06)
                fusion.query.weight.add_(torch.randn_like(fusion.query.weight)*.07)
            model.enhancement.project.weight.normal_(std=.02)
            model.enhancement.execution_project.weight.normal_(std=.03)

    def test_independent_physical_projections_preserve_native_start(self):
        model=self.model();fusion=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        for name,reference in [('query','q'),('key','k'),('value','v')]:
            physical=getattr(fusion,name).weight;original=getattr(native,reference).weight
            self.assertNotEqual(physical.data_ptr(),original.data_ptr())
            self.assertTrue(physical.requires_grad);self.assertFalse(original.requires_grad)
            torch.testing.assert_close(physical,original,rtol=0,atol=0)
        self.assertFalse(hasattr(fusion,'state_value'))
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)

    def test_r9_grounding_and_existing_direct_writer_are_recovered_by_controls(self):
        path=bootstrap.BASE/'experiments/physical_content_v7/r1_content_reader_r9/snapshots/20261031T150803Z/physical_content/model.py'
        spec=importlib.util.spec_from_file_location('physical_content.r9_reference',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        model=self.model();self.activate(model)
        reference=old.PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))
        with torch.no_grad():
            for name,value in reference.state_dict().items():
                if '.state_value.' in name:value.zero_()
                else:value.copy_(model.state_dict()[name])
        content,groups=reference.enhancement(self.evidence)
        ours,ours_groups=model.enhancement(self.evidence,execution_residual=False)
        torch.testing.assert_close(content,ours['routing'],rtol=0,atol=0)
        torch.testing.assert_close(ours['routing'],ours['execution'],rtol=0,atol=0)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise
        expected=reference.velocity(noisy,self.sample['first'],self.text,self.sigma,content,self.sample['times'],groups)[0]
        torch.testing.assert_close(expected,self.full(model,execution_residual=False,native_query=True),rtol=1e-5,atol=1e-6)

    def test_declared_controls_change_only_their_components_and_generation(self):
        model=self.model();self.activate(model)
        full,groups=model.enhancement(self.evidence)
        removed,_=model.enhancement(self.evidence,execution_residual=False)
        torch.testing.assert_close(full['routing'],removed['routing'],rtol=0,atol=0)
        self.assertGreater(float((full['execution']-removed['execution']).abs().max()),1e-5)
        normal=self.full(model)
        for flags in [dict(physical_enabled=False),dict(native_query=True),dict(execution_residual=False)]:
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False)
        self.evidence['words']=torch.randn_like(self.evidence['words'])
        self.assertGreater(float((normal-self.full(model)).abs().max()),1e-6)
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_generation_gradients_reach_both_content_and_independent_reader(self):
        model=self.model();model.set_phase(0)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()),0.)
        model.zero_grad(set_to_none=True);self.activate(model)
        self.assertEqual(model.set_phase(127),'interface')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        for p in [model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  model.enhancement.block.cross.v.weight,*[getattr(model.fusions['1'],k).weight for k in ['query','key','value','output']]]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True);self.assertEqual(model.set_phase(128),'joint')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1,execution_alignment=.5)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))
        for layer in model.wan.blocks:self.assertFalse(any(p.requires_grad for p in layer.cross_attn.parameters()))

    def test_execution_alignment_is_released_without_disabling_generation(self):
        weights=[execution_alignment_weight(self.cfg,s) for s in range(400)]
        self.assertEqual(weights[:129],[1.]*129);self.assertEqual(weights[-1],0.)
        self.assertTrue(all(a>=b for a,b in zip(weights,weights[1:])))
        model=self.model();self.activate(model)
        c,_=model.enhancement(self.evidence);loss=model.enhancement.content_losses(c,self.evidence)
        loss['routing'].backward()
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        model.zero_grad(set_to_none=True)
        values=backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1,execution_alignment=0.)
        self.assertEqual(values['weighted_execution_aux'],0.)
        self.assertGreater(float(model.enhancement.execution_project.weight.grad.norm()),0.)

    def test_mean_control_preserves_both_memories_and_own_prior(self):
        model=self.model();self.activate(model);content,g=model.enhancement(self.evidence)
        accumulator=MeanContent();accumulator.add(content,g)
        shifted={k:v+1 for k,v in content.items()};accumulator.add(shifted,g)
        mean,report=accumulator.finish();self.assertEqual(report['count'],2)
        for k in content:torch.testing.assert_close(mean[k],content[k]+.5)
        pos,pg,neg,ng=mean_with_own_prior(dict(positive=mean,negative=mean),g,g,'cpu')
        self.assertIs(pg,g);self.assertIs(ng,g)
        for k in mean:torch.testing.assert_close(pos[k],neg[k],rtol=0,atol=0)

    def test_checkpoint_and_optimizer_ownership(self):
        model=self.model();self.activate(model);model.set_phase(128)
        groups=model.optim_groups();ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        named=dict(model.named_parameters());self.assertEqual(set(ids),{id(named[n]) for n in model.learned_names})
        optimizer=torch.optim.AdamW(groups,foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1);optimizer.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]):save(model,optimizer,self.cfg,self.path,129,516)
        restored=self.model();other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with patch('torch.cuda.set_rng_state_all'):
            self.assertEqual(restore(restored,other,self.path/'step0129.pt',self.cfg),(129,516))
        for flags in ({},{'physical_enabled':False},{'native_query':True},{'execution_residual':False}):
            torch.testing.assert_close(self.full(model,**flags),self.full(restored,**flags),rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
