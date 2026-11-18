"""Actual small Wan: residual composition, input use, gradients and resume."""
from pathlib import Path
import copy, importlib.util, sys, tempfile, types, unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_sample
from physical_content.train import save, restore
torch.set_num_threads(2)


def tiny_wan():
    name='_r15g_test_wan'
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


class SourceResidualTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_source_residual_fusion_r15g',
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

    def full(self,model,*,execution_prior_only=False,**kwargs):
        c,g=model.enhancement(self.evidence,execution_prior_only=execution_prior_only)
        return model.velocity((1-self.sigma)*self.clean+self.sigma*self.noise,self.sample['first'],
            self.text,self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def activate(self,model):
        with torch.no_grad():
            model.enhancement.project.weight.normal_(std=.03)
            model.enhancement.execution_project.weight.normal_(std=.03)
            model.enhancement.execution_prior.normal_(std=.01)
            for fusion in model.fusions.values():
                fusion.state_value.weight.normal_(std=.05)
                fusion.query.weight.add_(torch.randn_like(fusion.query.weight)*.03)

    def test_initial_native_identity_and_independent_pretrained_projections(self):
        model=self.model();fusion=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        self.assertIsNone(fusion.output.bias)
        c,_=model.enhancement(self.evidence)
        torch.testing.assert_close(c['execution'],torch.zeros_like(c['execution']),rtol=0,atol=0)
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)
        for name,other in [('query','q'),('key','k'),('value','v'),('output','o')]:
            parameter=getattr(fusion,name).weight;original=getattr(native,other).weight
            self.assertTrue(parameter.requires_grad);self.assertFalse(original.requires_grad)
            self.assertNotEqual(parameter.data_ptr(),original.data_ptr())
            torch.testing.assert_close(parameter,original,rtol=0,atol=0)
        ids=[id(p) for g in model.optim_groups() for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(p) for p in model.parameters() if p.requires_grad})

    def test_first_fm_backward_reaches_execution_without_auxiliary_or_lora(self):
        model=self.model();model.set_phase(0)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        for p in [model.enhancement.execution_project.weight,model.enhancement.execution_prior]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        self.assertEqual(float(model.fusions['1'].output.weight.grad.norm()),0.)

    def test_routing_alignment_does_not_supervise_execution_values(self):
        model=self.model();self.activate(model)
        c,_=model.enhancement(self.evidence);model.enhancement.content_loss(c,self.evidence).backward()
        self.assertGreater(float(model.enhancement.project.weight.grad.norm()),0.)
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        self.assertIsNone(model.enhancement.execution_prior.grad)

    def test_real_source_content_and_request_affect_generation(self):
        model=self.model();self.activate(model)
        normal=self.full(model)
        self.evidence['request_words']+=torch.randn_like(self.evidence['request_words'])*3
        self.assertGreater(float((normal-self.full(model)).abs().max()),1e-6)
        requested=self.full(model)
        self.evidence['words']+=torch.randn_like(self.evidence['words'])*3
        self.assertGreater(float((requested-self.full(model)).abs().max()),1e-6)
        # Direct raw image/template tokens are not a hidden extra path; the
        # contextual word features still retain frozen Qwen image information.
        current=self.full(model);self.evidence['memory'].normal_()
        torch.testing.assert_close(current,self.full(model),rtol=0,atol=0)

    def test_physical_query_reads_true_post_text_residual(self):
        model=self.model();self.activate(model);model.checkpointing=False
        layer=model.wan.blocks[1];norm_inputs=[];norm_outputs=[];texts=[];physical=[]
        hooks=[layer.norm3.register_forward_hook(lambda m,a,o:(norm_inputs.append(a[0]),norm_outputs.append(o)) and None),
            layer.cross_attn.register_forward_hook(lambda m,a,o:texts.append(o)),
            model.fusions['1'].register_forward_pre_hook(lambda m,a:physical.append(a[0]))]
        try:
            self.full(model);self.assertEqual(len(norm_inputs),2)
            torch.testing.assert_close(norm_inputs[1],norm_inputs[0]+texts[0],rtol=0,atol=0)
            torch.testing.assert_close(physical[0],norm_outputs[1],rtol=0,atol=0)
        finally:
            for h in hooks:h.remove()

    def test_controls_and_first_frame_contract(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for flags in [dict(execution_prior_only=True),dict(native_physical_query=True),dict(physical_enabled=False)]:
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False);self.evidence['words'].normal_();self.evidence['request_words'].normal_()
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)
        c,g=model.enhancement(self.evidence)
        delta=model.fusions['1'](torch.randn(1,6,16),c,self.sample['times'],(3,1,2),model.wan.blocks[1].cross_attn,g)
        torch.testing.assert_close(delta[:,:2],torch.zeros_like(delta[:,:2]),rtol=0,atol=0)
        self.assertGreater(float(delta[:,2:].abs().max()),0.)

    def test_generation_gradients_and_two_phase_ownership(self):
        model=self.model();self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        fusion=model.fusions['1']
        expected=[model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  model.enhancement.execution_prior,model.enhancement.block.cross.q.weight,
                  model.enhancement.block.cross.k.weight,
                  *[getattr(fusion,k).weight for k in ('query','key','value','output','state_value')]]
        for parameter in expected:
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True);model.set_phase(128)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_checkpoint_resume_preserves_next_joint_optimizer_update(self):
        model=self.model();self.activate(model);model.set_phase(128)
        optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        optimizer.step();optimizer.zero_grad(set_to_none=True);save(model,optimizer,self.cfg,self.path,129,516)
        expected=self.full(model).detach().clone()
        resumed=self.model();other=torch.optim.AdamW(resumed.optim_groups(),foreach=False)
        self.assertEqual(restore(resumed,other,self.path/'latest.pt',self.cfg),(129,516));resumed.set_phase(129)
        torch.testing.assert_close(self.full(resumed),expected,rtol=0,atol=0)
        for candidate,opt in ((model,optimizer),(resumed,other)):
            backward_sample(candidate,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
            opt.step();opt.zero_grad(set_to_none=True)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name],value,rtol=0,atol=0)


if __name__=='__main__':unittest.main(verbosity=2)
