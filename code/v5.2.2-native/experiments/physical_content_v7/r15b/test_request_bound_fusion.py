"""Actual small Wan: native preservation, request binding, gradients and controls."""
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
    name='_r15b_test_wan'
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


class RequestBoundFusionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_request_bound_fusion_r15b',
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

    def test_exact_r10a_parameters_initializers_and_direct_control(self):
        model=self.model();reference=self.reference()
        self.assertEqual(set(model.learned_names),set(reference.learned_names))
        self.assertEqual(set(model.state_dict()),set(reference.state_dict()))
        for k,v in reference.state_dict().items():torch.testing.assert_close(model.state_dict()[k],v,rtol=0,atol=0)
        self.activate(model);reference.load_state_dict(model.state_dict())
        c,g=reference.enhancement(self.evidence)
        ours,og=model.enhancement(self.evidence)
        for k in c:torch.testing.assert_close(ours[k],c[k],rtol=0,atol=0)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise
        expected=reference.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g)[0]
        torch.testing.assert_close(expected,self.full(model,direct_physical_read=True),rtol=1e-5,atol=1e-6)

    def test_native_start_and_native_text_projections_are_preserved(self):
        model=self.model();f=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        for name,ref in [('key','k'),('value','v')]:
            p=getattr(f,name).weight;original=getattr(native,ref).weight
            self.assertNotEqual(p.data_ptr(),original.data_ptr());self.assertTrue(p.requires_grad)
            torch.testing.assert_close(p,original,rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in native.parameters()))
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)

    def test_binding_and_state_controls_affect_the_actual_generator(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for flags in [dict(physical_enabled=False),dict(direct_physical_read=True),dict(state_value_enabled=False)]:
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False)
        self.evidence['words']=torch.randn_like(self.evidence['words'])
        self.assertGreater(float((normal-self.full(model)).abs().max()),1e-6)
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_padding_and_zero_physical_values_cannot_create_a_request_writer(self):
        model=self.model();self.activate(model);f=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        c,g=model.enhancement(self.evidence);x=torch.randn(1,6,16);context=torch.randn(1,5,16)
        updates=[]
        handle=f.output.register_forward_hook(lambda _,args,out:updates.append(out.detach().clone()))
        kw=dict(context_length=3)
        result,_=f(x,context,c,self.sample['times'],(3,1,2),native,g,**kw)
        changed=context.clone();changed[:,3:]+=100
        f(x,changed,c,self.sample['times'],(3,1,2),native,g,**kw)
        torch.testing.assert_close(updates[0],updates[1],rtol=0,atol=0)
        empty=dict(c,execution=torch.zeros_like(c['execution']))
        f(x,context,empty,self.sample['times'],(3,1,2),native,g,**kw)
        self.assertEqual(float(updates[-1].abs().max()),0.)
        # The known first frame receives only the original native text update.
        split=lambda z:z.reshape(1,-1,2,8)
        native_text=native.o(F.scaled_dot_product_attention(split(native.norm_q(native.q(x))).transpose(1,2),
            split(native.norm_k(native.k(context))).transpose(1,2),split(native.v(context)).transpose(1,2)).transpose(1,2).flatten(2))
        torch.testing.assert_close(result[:,:2],native_text[:,:2],rtol=1e-5,atol=1e-6)
        handle.remove()
        # One-token null T5 remains valid; no special null/request topology.
        f(x,context,c,self.sample['times'],(3,1,2),native,g,context_length=1)

    def test_reference_mass_is_preserved_for_any_prompt_length(self):
        # Unequal C slots: two groups of 3 and 5. Neutral logits must produce
        # exactly the old R:1 physical/reference prior, with no double null gate.
        groups=torch.tensor([0,0,0,1,1,1,1,1]);values=torch.randn(8,4)
        counts=torch.bincount(groups).float();prior=-counts[groups].log()
        old=torch.cat((prior,torch.zeros(1))).softmax(0)[:8]@values
        bound=prior.softmax(0)@values
        for count in [1,2,17,64]:
            p=torch.cat((torch.full((count,),math.log(2)-math.log(count)),torch.zeros(1))).softmax(0)
            actual=p[:count]@bound[None].expand(count,-1)
            torch.testing.assert_close(actual,old,rtol=1e-6,atol=1e-6)

    def test_generation_and_alignment_gradients_have_declared_ownership(self):
        model=self.model();model.set_phase(0)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()),0.)
        model.zero_grad(set_to_none=True);self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        for p in [model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  model.enhancement.block.cross.v.weight,*[getattr(model.fusions['1'],k).weight for k in ['key','value','output','state_value']]]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True);c,_=model.enhancement(self.evidence)
        model.enhancement.content_loss(c,self.evidence).backward()
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        model.zero_grad(set_to_none=True);self.assertEqual(model.set_phase(128),'joint')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

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
        for flags in ({},{'physical_enabled':False},{'direct_physical_read':True},{'state_value_enabled':False}):
            torch.testing.assert_close(self.full(model,**flags),self.full(restored,**flags),rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
