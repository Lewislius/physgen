"""Actual small Wan: physical-to-native semantic response, native preservation and controls."""
from pathlib import Path
import copy,importlib.util,math,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample
from physical_content.train import save,restore
torch.set_num_threads(2)


def tiny_wan():
    name='_r15e_test_wan'
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


class SemanticResponseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_semantic_response_fusion_r15e',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,condition_every=2,lora_rank=2),
            paths=dict(interface=str(self.path/'interface.pt')),stages=dict(interface=128,joint=272),
            loss=dict(generation=dict(tail_weight=.2,tail_fraction=.25,temporal_weight=.2,temporal_lags=[1])),
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

    def full(self,model,**kwargs):
        c,g=model.enhancement(self.evidence)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise
        return model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def activate(self,model):
        with torch.no_grad():
            for fusion in model.fusions.values():
                fusion.output.weight.normal_(std=.06);fusion.state_value.weight.normal_(std=.05)
            model.enhancement.project.weight.normal_(std=.02)
            model.enhancement.execution_project.weight.normal_(std=.03)

    def reference(self):
        path=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/snapshots/20261101T034557Z/physical_content/model.py'
        spec=importlib.util.spec_from_file_location('physical_content.r10a_reference',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        torch.manual_seed(42)
        return old.PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))


    def test_r10a_enhancement_original_lora_and_rng_are_retained(self):
        model=self.model();rng=torch.get_rng_state();reference=self.reference();old_rng=torch.get_rng_state()
        torch.testing.assert_close(rng,old_rng,rtol=0,atol=0)
        for name,value in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        c,g=model.enhancement(self.evidence);old,og=reference.enhancement(self.evidence)
        for name in c:torch.testing.assert_close(c[name],old[name],rtol=0,atol=0)
        torch.testing.assert_close(g,og)

    def test_native_start_and_independent_physical_projections(self):
        model=self.model();fusion=model.fusions['1'];block=model.wan.blocks[1]
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)
        self.assertFalse(any(p.requires_grad for p in block.cross_attn.parameters()))
        for name,native in [('key','k'),('value','v')]:
            ours=getattr(fusion,name).weight;original=getattr(block.cross_attn,native).weight
            self.assertTrue(ours.requires_grad);self.assertNotEqual(ours.data_ptr(),original.data_ptr())
            torch.testing.assert_close(ours,original,rtol=0,atol=0)
        self.assertTrue(fusion.output.weight.requires_grad)

    def test_semantic_response_off_recovers_R10A_physical_computation(self):
        model=self.model();self.activate(model);reference=self.reference()
        reference.load_state_dict(model.state_dict(),strict=True)
        torch.testing.assert_close(self.full(model,semantic_response=False),self.full(reference),rtol=1e-5,atol=1e-6)

    def test_declared_controls_and_C_change_generator_outputs(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for flags in [dict(physical_enabled=False),dict(semantic_response=False),dict(state_interaction=False)]:
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False)
        self.evidence['words']=torch.randn_like(self.evidence['words'])
        self.assertGreater(float((normal-self.full(model)).abs().max()),1e-6)
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_physical_values_are_written_directly_and_first_frame_is_untouched(self):
        model=self.model();self.activate(model);updates=[]
        handle=model.fusions['1'].register_forward_hook(lambda _,args,out:updates.append(out.detach()))
        self.full(model);self.assertEqual(float(updates[-1][:,:2].abs().max()),0.)
        self.assertGreater(float(updates[-1][:,2:].abs().max()),0.)
        with torch.no_grad():model.fusions['1'].value.weight.zero_()
        self.full(model);self.assertEqual(float(updates[-1].abs().max()),0.)
        handle.remove()

    def test_zero_initialization_opens_generation_gradient_path(self):
        model=self.model();model.set_phase(0)
        optimizer=torch.optim.AdamW(model.optim_groups(),lr=.001,foreach=False)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
            self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()),0.)
            if step>=1:self.assertGreater(float(model.fusions['1'].state_value.weight.grad.norm()),0.)
            if step>=1:self.assertGreater(float(model.enhancement.execution_project.weight.grad.norm()),0.)
            optimizer.step()
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))

    def test_actual_generation_and_alignment_gradient_ownership(self):
        model=self.model();self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        fusion=model.fusions['1']
        expected=[model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  model.enhancement.block.cross.v.weight,*[getattr(fusion,k).weight for k in
                  ('key','value','output','state_value')]]
        for parameter in expected:
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True);c,_=model.enhancement(self.evidence)
        model.enhancement.content_loss(c,self.evidence).backward()
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        self.assertTrue(all(p.grad is None for p in model.fusions.parameters()))
        model.zero_grad(set_to_none=True);self.assertEqual(model.set_phase(128),'joint')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_semantic_response_changes_generation_credit_without_text_parameter_updates(self):
        model=self.model();self.activate(model);model.set_phase(0)
        gradients=[];native_inputs=[]
        handle=model.wan.blocks[1].cross_attn.register_forward_pre_hook(
            lambda module,args:native_inputs.append(args[0].detach().clone()))
        for enabled in (True,False):
            model.zero_grad(set_to_none=True)
            self.full(model,semantic_response=enabled).square().mean().backward()
            gradients.append(model.enhancement.execution_project.weight.grad.clone())
            self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        handle.remove()
        # Checkpoint recomputation may repeat calls. The last forward of each
        # mode is compared separately below without gradient checkpointing.
        model.checkpointing=False;native_inputs=[]
        handle=model.wan.blocks[1].cross_attn.register_forward_pre_hook(
            lambda module,args:native_inputs.append(args[0].detach().clone()))
        self.full(model);self.full(model,semantic_response=False);handle.remove()
        self.assertEqual(len(native_inputs),2)
        self.assertGreater(float((native_inputs[0]-native_inputs[1]).abs().max()),1e-6)
        self.assertGreater(float((gradients[0]-gradients[1]).abs().max()),1e-7)

    def test_checkpoint_optimizer_and_all_controls_roundtrip(self):
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
        for flags in ({},{'physical_enabled':False},{'semantic_response':False},{'state_interaction':False}):
            torch.testing.assert_close(self.full(model,**flags),self.full(restored,**flags),rtol=0,atol=0)



if __name__=='__main__':unittest.main(verbosity=2)
