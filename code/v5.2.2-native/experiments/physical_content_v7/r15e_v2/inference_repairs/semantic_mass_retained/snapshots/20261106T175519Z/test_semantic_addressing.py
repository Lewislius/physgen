"""Actual small Wan: semantic addressing, request refinement and generation gradients."""
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


class SemanticAddressingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_semantic_addressing_r15e',
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

    def full(self,model,*,request_refinement=True,text=None,**kwargs):
        c,g=model.enhancement(self.evidence,request_refinement=request_refinement)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise
        return model.velocity(noisy,self.sample['first'],self.text if text is None else text,
                              self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def activate(self,model):
        with torch.no_grad():
            for fusion in model.fusions.values():
                fusion.output.weight.normal_(std=.06);fusion.state_value.weight.normal_(std=.05)
                fusion.query.weight.add_(torch.randn_like(fusion.query.weight)*.03)
            model.enhancement.project.weight.normal_(std=.02)
            model.enhancement.execution_project.weight.normal_(std=.03)

    def reference(self):
        path=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/snapshots/20261101T034557Z/physical_content/model.py'
        spec=importlib.util.spec_from_file_location('physical_content.r10a_reference',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        torch.manual_seed(42)
        return old.PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))

    def test_native_initialization_and_parameter_ownership(self):
        model=self.model();rng=torch.get_rng_state();old=self.reference()
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for name,value in old.state_dict().items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        fusion=model.fusions['1'];attention=model.wan.blocks[1].cross_attn
        self.assertTrue(fusion.query.weight.requires_grad)
        self.assertNotEqual(fusion.query.weight.data_ptr(),attention.q.weight.data_ptr())
        self.assertFalse(any(p.requires_grad for p in attention.parameters()))
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)
        groups=model.optim_groups();ids=[id(p) for g in groups for p in g['params']]
        named=dict(model.named_parameters());self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(named[n]) for n in model.learned_names})

    def test_disabling_new_calculations_recovers_exact_R10A(self):
        model=self.model();self.activate(model);reference=self.reference()
        reference.load_state_dict({k:v for k,v in model.state_dict().items()
            if not (k.startswith('fusions.') and ('.query.' in k or '.query_norm.' in k))},strict=True)
        c,g=reference.enhancement(self.evidence)
        expected,_=reference.velocity((1-self.sigma)*self.clean+self.sigma*self.noise,
            self.sample['first'],self.text,self.sigma,c,self.sample['times'],g)
        actual=self.full(model,request_refinement=False,semantic_routing=False,native_physical_query=True)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)

    def test_neutral_semantics_preserve_unequal_group_prior_and_reference_mass(self):
        from physical_content.model import Fusion
        from physical_scaling.supervision import group_log_prior
        prior=group_log_prior(torch.tensor([0,0,1,1,1]))
        query=torch.randn(1,2,4,8);text_keys=torch.randn(1,2,3,8)
        content_keys=torch.randn(1,2,1,8).expand(1,2,5,8)
        result=Fusion.semantic_log_prior(query,text_keys,content_keys,prior)
        torch.testing.assert_close(result,prior.expand_as(result),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(result[...,:-1].exp().sum(-1),torch.full((1,2,4),2.))
        torch.testing.assert_close(result[...,-1],torch.zeros(1,2,4))

    def test_semantic_addressing_changes_location_without_a_hard_mask(self):
        from physical_content.model import Fusion
        q=torch.tensor([[[[3.,0.],[0.,3.]]]])
        keys=q.clone();prior=torch.zeros(3)
        result=Fusion.semantic_log_prior(q,keys,keys,prior)
        self.assertGreater(float(result[0,0,0,0]-result[0,0,0,1]),3.)
        self.assertGreater(float(result[0,0,1,1]-result[0,0,1,0]),3.)
        torch.testing.assert_close(result[...,:-1].exp().sum(-1),torch.full((1,1,2),2.))
        self.assertTrue(torch.isfinite(result).all())
        direct=torch.tensor([[[[0.,100.,0.],[100.,0.,0.]]]])
        combined=(direct+result).softmax(-1)
        self.assertGreater(float(combined[0,0,0,1]),.99)

    def test_text_order_and_content_order_mapping_are_consistent(self):
        from physical_content.model import Fusion
        from physical_scaling.supervision import group_log_prior
        q=torch.randn(1,2,4,8);k=torch.randn(1,2,3,8);c=torch.randn(1,2,5,8)
        groups=torch.tensor([0,0,1,1,1]);prior=group_log_prior(groups)
        baseline=Fusion.semantic_log_prior(q,k,c,prior)
        torch.testing.assert_close(Fusion.semantic_log_prior(q,k.flip(-2),c,prior),baseline,rtol=1e-5,atol=1e-6)
        order=torch.tensor([4,1,2,0,3]);reordered=Fusion.semantic_log_prior(q,k,c[:,:,order],group_log_prior(groups[order]))
        torch.testing.assert_close(reordered[...,:-1],baseline[...,:-1][...,order],rtol=1e-5,atol=1e-6)

    def test_only_literal_request_is_read_by_contextual_refinement_heads(self):
        model=self.model();block=model.enhancement.block
        state=torch.randn(1,8,32);request=torch.randn(1,5,32);words=torch.randn(1,6,32)
        memory=torch.cat((request,words),1);mask=torch.ones(1,11,dtype=torch.bool)
        groups=torch.arange(8)//4;word_groups=torch.arange(6)//3
        a=block.read_heads(state,memory,mask,words,word_groups,groups,True)
        changed=words+torch.randn_like(words)*3
        b=block.read_heads(state,torch.cat((request,changed),1),mask,changed,word_groups,groups,True)
        torch.testing.assert_close(a[:,2:],b[:,2:],rtol=0,atol=0)
        self.assertGreater(float((a[:,:2]-b[:,:2]).abs().max()),1e-5)

    def test_controls_change_generation_and_off_is_condition_independent(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for flags in [dict(semantic_routing=False),dict(native_physical_query=True),dict(request_refinement=False)]:
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False)
        self.evidence['words']=torch.randn_like(self.evidence['words'])
        self.evidence['request_words']=torch.randn_like(self.evidence['request_words'])
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_null_request_and_native_padding_do_not_create_semantic_conditions(self):
        model=self.model();self.activate(model)
        self.evidence['request_words']=torch.empty(0,32)
        c,_=model.enhancement(self.evidence);old,_=model.enhancement(self.evidence,request_refinement=False)
        for key in c:torch.testing.assert_close(c[key],old[key],rtol=0,atol=0)
        text=self.text[:,:1]
        torch.testing.assert_close(self.full(model,text=text),self.full(model,text=text,semantic_routing=False),rtol=0,atol=0)
        fusion=model.fusions['1'];calls=[]
        original=fusion.semantic_log_prior
        def capture(q,k,c,p):calls.append(k.shape[-2]);return original(q,k,c,p)
        with patch.object(fusion,'semantic_log_prior',side_effect=capture):self.full(model)
        self.assertEqual(set(calls),{4})

    def test_generation_gradients_and_stage_ownership(self):
        model=self.model();self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        fusion=model.fusions['1']
        expected=[model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  model.enhancement.block.cross.q.weight,model.enhancement.block.cross.k.weight,
                  *[getattr(fusion,k).weight for k in ('query','key','value','output','state_value')]]
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

    def test_first_frame_has_no_physical_write(self):
        model=self.model();self.activate(model);fusion=model.fusions['1'];block=model.wan.blocks[1]
        x=torch.randn(1,6,16);context=torch.randn(1,5,16);c,g=model.enhancement(self.evidence)
        update,_=fusion(x,context,c,self.sample['times'],(3,1,2),block.cross_attn,g,text_length=4)
        native=block.cross_attn(x,context,None)
        torch.testing.assert_close(update[:,:2],native[:,:2],rtol=1e-5,atol=1e-6)

    def test_checkpoint_resume_preserves_independent_query_and_next_joint_update(self):
        model=self.model();self.activate(model);model.set_phase(128)
        optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        optimizer.step();optimizer.zero_grad(set_to_none=True)
        save(model,optimizer,self.cfg,self.path,129,516)
        expected=self.full(model).detach().clone()
        resumed=self.model();other=torch.optim.AdamW(resumed.optim_groups(),foreach=False)
        step,cursor=restore(resumed,other,self.path/'latest.pt',self.cfg)
        self.assertEqual((step,cursor),(129,516));resumed.set_phase(step)
        torch.testing.assert_close(self.full(resumed),expected,rtol=0,atol=0)
        for candidate,opt in ((model,optimizer),(resumed,other)):
            backward_sample(candidate,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
            opt.step();opt.zero_grad(set_to_none=True)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name],value,rtol=0,atol=0)


if __name__=='__main__':unittest.main()


