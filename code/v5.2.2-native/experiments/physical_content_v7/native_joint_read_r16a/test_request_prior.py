"""Actual small Wan: residual composition, input use, gradients and resume."""
from pathlib import Path
import copy, importlib.util, sys, tempfile, types, unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from unittest.mock import patch
from guidance import guided_velocity, LEGACY
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample
from physical_content.train import save, restore
from physical_content.joint_read import read_physical
torch.set_num_threads(2)


class ConditionalReadTest(unittest.TestCase):
    def test_uninformative_request_reduces_to_direct_read_with_unequal_groups(self):
        # Three relations, with different slot counts: the null must still have
        # one relation's prior mass. A uniform token prior would violate this.
        torch.manual_seed(19)
        q=torch.randn(1,2,7,4);k=torch.randn(1,2,8,4);v=torch.randn_like(k)
        request_query=torch.randn_like(q);request_keys=torch.zeros(1,2,5,4)
        reference=torch.randn(1,2,1,4)
        prior=torch.tensor([1/2]*2+[1/5]*5+[1.]+[1.]).log()
        args=(q,k,v,request_keys,reference,prior)
        actual=read_physical(*args,request_query=request_query,relation_count=3)
        direct=read_physical(*args,mode='direct',request_query=request_query,relation_count=3)
        torch.testing.assert_close(actual,direct,rtol=1e-6,atol=1e-6)

    def test_native_request_and_physical_query_can_each_select_execution_content(self):
        # Hold direct physical evidence ambiguous, then reverse only native
        # request addressing. Repeat with request addressing ambiguous and only
        # the physical query changed. Both must change the *read C values*.
        keys=torch.tensor([[[[4.,0.],[0.,4.]]]])
        values=torch.tensor([[[[1.,0.],[0.,1.]]]])
        reference=torch.zeros(1,1,1,2);prior=torch.zeros(3)
        neutral=torch.zeros(1,1,1,2)
        first=torch.tensor([[[[4.,0.]]]]);second=first.flip(-1)
        def read(physical,request):
            return read_physical(physical,keys,values,keys,reference,prior,
                                 request_query=request,relation_count=2)[0,0,0]
        for first_read,second_read in [(read(neutral,first),read(neutral,second)),
                                       (read(first,neutral),read(second,neutral))]:
            self.assertGreater(float(first_read[0]),float(first_read[1])+.5)
            self.assertGreater(float(second_read[1]),float(second_read[0])+.5)

    def test_generation_credit_reaches_physical_content_and_request_addressing(self):
        torch.manual_seed(22)
        q=torch.randn(1,2,3,4,requires_grad=True)
        request_query=torch.randn_like(q,requires_grad=True)
        k=torch.randn(1,2,5,4,requires_grad=True)
        v=torch.randn_like(k,requires_grad=True)
        request_keys=torch.randn(1,2,6,4)
        out=read_physical(q,k,v,request_keys,torch.zeros(1,2,1,4),
                          torch.tensor([1/2]*2+[1/3]*3+[1.]).log(),
                          request_query=request_query,relation_count=2)
        (out*torch.randn_like(out)).sum().backward()
        for value in (q,request_query,k,v):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.norm()),1e-5)


def tiny_wan():
    name='_r16a_test_wan'
    if name+'.model' not in sys.modules:
        package=types.ModuleType(name);package.__path__=[]
        attention=types.ModuleType(name+'.attention')
        def cpu_attention(q,k,v,**kwargs):
            return F.scaled_dot_product_attention(q.transpose(1,2).float(),
                k.transpose(1,2).float(),v.transpose(1,2).float()).transpose(1,2).to(v.dtype)
        attention.flash_attention=cpu_attention
        sys.modules[name]=package;sys.modules[name+'.attention']=attention
        spec=importlib.util.spec_from_file_location(name+'.model','/home/liuzhirui/model/Wan2.2/wan/modules/model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    wan=sys.modules[name+'.model'].WanModel(model_type='ti2v',patch_size=(1,2,2),text_len=5,
        in_dim=2,out_dim=2,dim=16,ffn_dim=32,freq_dim=8,text_dim=16,num_heads=2,num_layers=2)
    with torch.no_grad():wan.head.head.weight.normal_(std=.1)
    return wan


class JointMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_native_joint_read_r16a',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,
                condition_every=2,lora_rank=2),paths=dict(interface=str(self.path/'interface.pt')),
            stages=dict(interface=128,joint=272),loss=dict(generation=dict(tail_weight=.2,tail_fraction=.25)),
            train=dict(seed=42,gradient_checkpointing=True,keep_steps=[129],
                peak_lrs=dict(enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)))
        self.wan=tiny_wan()
        # Official pretrained native projections can have nonzero biases;
        # unlike a freshly initialized tiny Wan, this exercises that contract.
        with torch.no_grad():
            for block in self.wan.blocks:block.cross_attn.o.bias.normal_(std=.2)
        self.evidence=dict(words=torch.randn(6,32),word_groups=torch.arange(6)//3,
            request_words=torch.randn(5,32),native_words=torch.randn(6,16)*.1,
            memory=torch.randn(1,17,32),memory_mask=torch.ones(1,17,dtype=torch.bool))
        self.clean=torch.randn(1,2,3,2,4);self.noise=torch.randn_like(self.clean);self.text=torch.randn(1,4,16)
        self.sample=dict(latent=self.clean,first=self.clean[:,:,:1],times=torch.tensor([[0.,1.,2.]]),valid=torch.ones(2,4))
        self.sigma=torch.tensor(.6)

    def model(self):
        torch.manual_seed(42)
        return PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))

    def full(self, model, **options):
        c, groups = model.enhancement(self.evidence)
        return model.velocity((1-self.sigma)*self.clean+self.sigma*self.noise,
            self.sample['first'], self.text, self.sigma, c, self.sample['times'], groups, **options)[0]

    def activate(self, model):
        with torch.no_grad():
            model.enhancement.project.weight.normal_(std=.03)
            model.enhancement.execution_project.weight.normal_(std=.03)
            for fusion in model.fusions.values():
                for name in ('output','state_value'):
                    getattr(fusion,name).weight.normal_(std=.04)
                fusion.query.weight.add_(torch.randn_like(fusion.query.weight)*.03)

    def test_independent_projections_native_start_and_optimizer_ownership(self):
        model = self.model()
        torch.testing.assert_close(self.full(model), self.full(model, physical_enabled=False), rtol=1e-5, atol=1e-6)
        fusion = model.fusions['1']; native = model.wan.blocks[1].cross_attn
        for own, other in [('query','q'),('key','k'),('value','v')]:
            a, b = getattr(fusion,own).weight, getattr(native,other).weight
            self.assertTrue(a.requires_grad); self.assertFalse(b.requires_grad)
            self.assertNotEqual(a.data_ptr(),b.data_ptr())
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertFalse(hasattr(fusion,'semantic_query'))
        ids=[id(p) for group in model.optim_groups() for p in group['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

    def test_memory_query_and_state_controls_reach_actual_predictions(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for options in [dict(memory_mode='direct'),dict(memory_mode='request_prior'),dict(memory_mode='legacy_joint'),dict(shared_physical_query=True),
                        dict(native_physical_query=True),dict(state_value_enabled=False),dict(physical_enabled=False)]:
            self.assertGreater(float((normal-self.full(model,**options)).detach().abs().max()),1e-7)
        no_c=self.full(model,physical_enabled=False)
        self.evidence['words']=torch.randn_like(self.evidence['words'])
        self.assertGreater(float((normal-self.full(model)).detach().abs().max()),1e-7)
        torch.testing.assert_close(no_c,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_padding_first_frame_and_zero_physical_values(self):
        model=self.model();self.activate(model)
        f=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        c,g=model.enhancement(self.evidence);x=torch.randn(1,6,16);context=torch.randn(1,5,16)
        updates=[];handle=f.output.register_forward_hook(lambda _,args,out:updates.append(out.detach().clone()))
        result=f(x,context,c,self.sample['times'],(3,1,2),native,g,context_length=3)
        changed=context.clone();changed[:,3:]+=100
        f(x,changed,c,self.sample['times'],(3,1,2),native,g,context_length=3)
        torch.testing.assert_close(updates[0],updates[1],rtol=0,atol=0)
        f(x,context,dict(c,execution=torch.zeros_like(c['execution'])),self.sample['times'],(3,1,2),native,g,context_length=3)
        self.assertEqual(float(updates[-1].abs().max()),0.)
        split=lambda z:z.reshape(1,-1,2,8)
        expected=native.o(F.scaled_dot_product_attention(split(native.norm_q(native.q(x))).transpose(1,2),
            split(native.norm_k(native.k(context))).transpose(1,2),split(native.v(context)).transpose(1,2)).transpose(1,2).flatten(2))
        torch.testing.assert_close(result[:,:2],expected[:,:2],rtol=1e-5,atol=1e-6)
        handle.remove()

    def test_generation_and_alignment_gradient_ownership(self):
        model=self.model();model.set_phase(0)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        self.assertGreater(float(model.fusions['1'].output.weight.grad.norm()),0.)
        model.zero_grad(set_to_none=True);self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        f=model.fusions['1']
        for p in [model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  *[getattr(f,k).weight for k in ['query','key','value','output','state_value']]]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True);c,_=model.enhancement(self.evidence)
        model.enhancement.content_loss(c,self.evidence).backward()
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        model.zero_grad(set_to_none=True);self.assertEqual(model.set_phase(128),'joint')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_legacy_intervention_matches_frozen_B_joint_operator(self):
        root=bootstrap.BASE/'experiments/physical_content_v7'
        path=root/'request_bound_fusion_r15b/snapshots/20261105T152223Z/physical_content/model.py'
        spec=importlib.util.spec_from_file_location('physical_content.b_reference',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        spec=importlib.util.spec_from_file_location('b_joint_probe',root/'r15i_design/snapshots/20261108T115136Z/joint_read.py')
        probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        ours=self.model();self.activate(ours)
        previous=old.PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))
        previous.load_state_dict({k:ours.state_dict()[k] for k in previous.state_dict()},strict=True)
        c,g=previous.enhancement(self.evidence)
        with patch.object(old.Fusion,'forward',probe.fusion_forward):
            expected,_=previous.velocity((1-self.sigma)*self.clean+self.sigma*self.noise,
                self.sample['first'],self.text,self.sigma,c,self.sample['times'],g)
        torch.testing.assert_close(expected,self.full(ours,native_physical_query=True,memory_mode='legacy_joint'),rtol=1e-5,atol=1e-6)

    def test_guidance_controls_apply_to_both_arms(self):
        model=self.model();self.activate(model);c,g=model.enhancement(self.evidence)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise;null=self.text[:,:1]*0
        for mode in ['joint','direct','request_prior','legacy_joint']:
            for shared in (False,True):
                options=dict(memory_mode=mode,native_physical_query=False,
                                 shared_physical_query=shared,physical_enabled=True)
                actual=guided_velocity(model,noisy,self.sample['first'],self.text,null,self.sigma,c,c,self.sample['times'],g,g,
                                       composition=LEGACY,guidance=5.,**options)
                pos,_=model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g,**options)
                neg,_=model.velocity(noisy,self.sample['first'],null,self.sigma,c,self.sample['times'],g,**options)
                torch.testing.assert_close(actual,neg+5.*(pos-neg),rtol=0,atol=0)

    def test_checkpoint_resume_all_controls(self):
        model=self.model();self.activate(model);model.set_phase(128)
        optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1);optimizer.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]):save(model,optimizer,self.cfg,self.path,129,516)
        other=self.model();new_optimizer=torch.optim.AdamW(other.optim_groups(),foreach=False)
        with patch('torch.cuda.set_rng_state_all'):
            self.assertEqual(restore(other,new_optimizer,self.path/'step0129.pt',self.cfg),(129,516))
        for options in [{},{'memory_mode':'direct'},{'memory_mode':'request_prior'},{'memory_mode':'legacy_joint'},{'shared_physical_query':True},{'native_physical_query':True},{'physical_enabled':False}]:
            torch.testing.assert_close(self.full(model,**options),self.full(other,**options),rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)

