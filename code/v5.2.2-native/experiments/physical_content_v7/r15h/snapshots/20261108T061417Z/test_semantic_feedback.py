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
    name='_r15h_test_wan'
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


class SemanticFeedbackTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_semantic_feedback_fusion_r15h',
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
                for name in ('output','state_value','semantic_query'):
                    getattr(fusion,name).weight.normal_(std=.04)
                fusion.query.weight.add_(torch.randn_like(fusion.query.weight)*.03)

    def test_initial_native_identity_and_independent_physical_parameters(self):
        model = self.model()
        torch.testing.assert_close(self.full(model), self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)
        fusion=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        for own,other in [('query','q'),('key','k'),('value','v')]:
            a=getattr(fusion,own).weight;b=getattr(native,other).weight
            self.assertTrue(a.requires_grad);self.assertFalse(b.requires_grad)
            self.assertNotEqual(a.data_ptr(),b.data_ptr());torch.testing.assert_close(a,b,rtol=0,atol=0)
        for layer in (fusion.output,fusion.semantic_query):
            self.assertEqual(float(layer.weight.abs().max()),0.)
        groups=[id(p) for group in model.optim_groups() for p in group['params']]
        self.assertEqual(len(groups),len(set(groups)))
        self.assertEqual(set(groups),{id(p) for p in model.parameters() if p.requires_grad})

    def test_zero_response_preserves_direct_physics_not_only_native_start(self):
        model=self.model();self.activate(model)
        for fusion in model.fusions.values():fusion.semantic_query.weight.data.zero_()
        torch.testing.assert_close(self.full(model),self.full(model,semantic_response_enabled=False),rtol=1e-6,atol=1e-7)
        self.assertGreater(float((self.full(model)-self.full(model,physical_enabled=False)).abs().max()),1e-6)

    def test_first_generation_gradient_reaches_both_new_output_routes(self):
        model=self.model();model.set_phase(0)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        for parameter in (model.fusions['1'].output.weight,model.fusions['1'].semantic_query.weight):
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))

    def test_semantic_response_is_exact_native_value_difference_without_bias_duplication(self):
        model=self.model();self.activate(model);fusion=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        x=torch.randn(1,6,16);context=torch.randn(1,5,16);c,g=model.enhancement(self.evidence)
        full=fusion(x,context,c,self.sample['times'],(3,1,2),native,g,direct_write_enabled=False)
        plain=native(x,context,None)
        torch.testing.assert_close(full[:,:2],plain[:,:2],rtol=1e-6,atol=1e-7)
        # A constant native V bank gives exactly the same value for every query.
        # Changing physical C must then produce no semantic-value correction.
        constant=context[:,:1].expand(-1,5,-1).clone()
        full=fusion(x,constant,c,self.sample['times'],(3,1,2),native,g,direct_write_enabled=False)
        torch.testing.assert_close(full,native(x,constant,None),rtol=1e-5,atol=1e-6)
        self.assertGreater(float((fusion(x,context,c,self.sample['times'],(3,1,2),native,g,
            direct_write_enabled=False)-plain).abs().max()),1e-6)

    def test_both_generation_routes_use_actual_C_and_ablation_switches(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for flags in (dict(physical_enabled=False),dict(semantic_response_enabled=False),
                      dict(direct_write_enabled=False),dict(native_physical_query=True)):
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False)
        semantic=self.full(model,direct_write_enabled=False)
        direct=self.full(model,semantic_response_enabled=False)
        self.evidence['words'].normal_()
        self.assertGreater(float((semantic-self.full(model,direct_write_enabled=False)).abs().max()),1e-6)
        self.assertGreater(float((direct-self.full(model,semantic_response_enabled=False)).abs().max()),1e-6)
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_generation_gradient_ownership_and_routing_only_auxiliary(self):
        model=self.model();self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        fusion=model.fusions['1']
        required=[model.enhancement.project.weight,model.enhancement.execution_project.weight,
                  model.enhancement.block.cross.q.weight,
                  *[getattr(fusion,k).weight for k in ('query','key','value','output','state_value','semantic_query')]]
        for parameter in required:
            self.assertIsNotNone(parameter.grad);self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0.)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        model.zero_grad(set_to_none=True);model.set_phase(128)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))
        model.zero_grad(set_to_none=True);c,_=model.enhancement(self.evidence)
        model.enhancement.content_loss(c,self.evidence).backward()
        self.assertIsNone(model.enhancement.execution_project.weight.grad)
        self.assertGreater(float(model.enhancement.project.weight.grad.norm()),0.)

    def test_semantic_only_generation_trains_C_reader(self):
        model=self.model();self.activate(model);model.set_phase(0)
        loss=(self.full(model,direct_write_enabled=False)-(self.noise-self.clean)).square().mean()
        loss.backward()
        for p in (model.enhancement.execution_project.weight,model.fusions['1'].query.weight,
                  model.enhancement.block.cross.q.weight):
            self.assertIsNotNone(p.grad);self.assertGreater(float(p.grad.norm()),0.)

    def test_checkpoint_resume_preserves_next_update(self):
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
            torch.testing.assert_close(value,resumed.state_dict()[name],rtol=0,atol=0)

    def test_both_cfg_arms_receive_identical_controls(self):
        from guidance import guided_velocity,LEGACY
        calls=[]
        class Recorder:
            def velocity(self,*args,**kwargs):
                calls.append(kwargs);return torch.ones(1)*len(calls),None
        actual=guided_velocity(Recorder(),None,None,None,None,None,None,None,None,None,None,
            composition=LEGACY,semantic_response_enabled=False,direct_write_enabled=True)
        self.assertEqual(calls[0],calls[1]);self.assertFalse(calls[0]['semantic_response_enabled'])
        torch.testing.assert_close(actual,torch.tensor([-3.]))


if __name__=='__main__':
    unittest.main(verbosity=2)

