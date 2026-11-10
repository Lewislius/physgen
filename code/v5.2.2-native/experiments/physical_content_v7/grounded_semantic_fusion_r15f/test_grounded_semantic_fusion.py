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
    name='_r15f_test_wan'
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


class GroundedSemanticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_grounded_semantic_fusion_r15f',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,
                condition_every=2,lora_rank=2),paths=dict(interface=str(self.path/'interface.pt')),
            stages=dict(interface=128,joint=272),loss=dict(generation=dict(tail_weight=.2,tail_fraction=.25)),
            train=dict(seed=42,gradient_checkpointing=True,keep_steps=[129],
                peak_lrs=dict(enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)))
        self.wan=tiny_wan()
        self.evidence=dict(words=torch.randn(6,32),word_groups=torch.arange(6)//3,
            request_words=torch.randn(5,32),native_words=torch.randn(6,16)*.1,
            memory=torch.randn(1,17,32),memory_mask=torch.ones(1,17,dtype=torch.bool))
        self.clean=torch.randn(1,2,3,2,4);self.noise=torch.randn_like(self.clean);self.text=torch.randn(1,4,16)
        self.sample=dict(latent=self.clean,first=self.clean[:,:,:1],times=torch.tensor([[0.,1.,2.]]),valid=torch.ones(2,4))
        self.sigma=torch.tensor(.6)

    def model(self):
        torch.manual_seed(42)
        return PhysicalModel(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))

    def full(self,model,*,words_only_memory=False,**kwargs):
        c,g=model.enhancement(self.evidence,words_only_memory=words_only_memory)
        return model.velocity((1-self.sigma)*self.clean+self.sigma*self.noise,self.sample['first'],
            self.text,self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def activate(self,model):
        with torch.no_grad():
            for fusion in model.fusions.values():
                fusion.output.weight.normal_(std=.06);fusion.state_value.weight.normal_(std=.05)
                fusion.query.weight.add_(torch.randn_like(fusion.query.weight)*.03)
            model.enhancement.project.weight.normal_(std=.03)

    def test_zero_initialization_exact_native_and_parameter_ownership(self):
        model=self.model();fusion=model.fusions['1'];native=model.wan.blocks[1].cross_attn
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)
        self.assertTrue(fusion.query.weight.requires_grad)
        self.assertNotEqual(fusion.query.weight.data_ptr(),native.q.weight.data_ptr())
        torch.testing.assert_close(fusion.query.weight,native.q.weight,rtol=0,atol=0)
        self.assertFalse(any(p.requires_grad for p in native.parameters()))
        groups=model.optim_groups();ids=[id(p) for g in groups for p in g['params']]
        params=dict(model.named_parameters());self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(set(ids),{id(params[n]) for n in model.learned_names})

    def test_shared_C_and_full_memory_are_real_generation_inputs(self):
        model=self.model();self.activate(model)
        c,_=model.enhancement(self.evidence)
        self.assertEqual(c['routing'].data_ptr(),c['execution'].data_ptr())
        full=self.full(model);words=self.full(model,words_only_memory=True)
        self.evidence['memory'][:,:4]+=torch.randn(1,4,32)*4
        self.assertGreater(float((full-self.full(model)).abs().max()),1e-6)
        torch.testing.assert_close(words,self.full(model,words_only_memory=True),rtol=0,atol=0)

    def test_grounded_heads_do_not_read_other_groups_or_global_memory(self):
        model=self.model();block=model.enhancement.block
        q=torch.randn(1,8,32);memory=torch.randn(1,13,32);words=torch.randn(1,6,32)
        mask=torch.ones(1,13,dtype=torch.bool);groups=torch.arange(8)//4;wg=torch.arange(6)//3
        a=block.read_heads(q,memory,mask,words,wg,groups)
        b=block.read_heads(q,memory+torch.randn_like(memory)*3,mask,words,wg,groups)
        torch.testing.assert_close(a[:,:2],b[:,:2],rtol=0,atol=0)
        self.assertGreater(float((a[:,2:]-b[:,2:]).abs().max()),1e-5)
        changed=words.clone();changed[:,3:]+=torch.randn_like(changed[:,3:])*3
        b=block.read_heads(q,memory,mask,changed,wg,groups)
        torch.testing.assert_close(a[:,:2,:4],b[:,:2,:4],rtol=0,atol=0)

    def test_native_text_reads_true_updated_residual_not_normalized_sum(self):
        model=self.model();self.activate(model);model.checkpointing=False
        layer=model.wan.blocks[1];norm_inputs=[];norm_outputs=[];text_inputs=[];writes=[]
        hooks=[layer.norm3.register_forward_hook(lambda m,a,o:(norm_inputs.append(a[0]),norm_outputs.append(o)) and None),
            layer.cross_attn.register_forward_pre_hook(lambda m,a:text_inputs.append(a[0])),
            model.fusions['1'].register_forward_hook(lambda m,a,o:writes.append(o))]
        try:
            self.full(model)
            self.assertEqual(len(norm_inputs),2)
            torch.testing.assert_close(norm_inputs[1],norm_inputs[0]+writes[0],rtol=0,atol=0)
            torch.testing.assert_close(text_inputs[0],norm_outputs[1],rtol=0,atol=0)
            self.assertGreater(float((text_inputs[0]-norm_outputs[0]).abs().max()),1e-6)
        finally:
            for h in hooks:h.remove()

    def test_frozen_native_text_feedback_reaches_physical_write(self):
        model=self.model();self.activate(model);model.checkpointing=False
        writes=[];texts=[]
        def capture_write(m,a,o):o.retain_grad();writes.append(o)
        hooks=[model.fusions['1'].register_forward_hook(capture_write),
               model.wan.blocks[1].cross_attn.register_forward_hook(lambda m,a,o:texts.append(o))]
        try:
            self.full(model);texts[0].square().mean().backward()
            self.assertGreater(float(writes[0].grad.norm()),0.)
            self.assertTrue(all(p.grad is None for p in model.wan.blocks[1].cross_attn.parameters()))
        finally:
            for h in hooks:h.remove()

    def test_controls_change_generation_and_disabled_fusion_is_independent(self):
        model=self.model();self.activate(model);normal=self.full(model)
        for flags in [dict(semantic_after_physics=False),dict(native_physical_query=True),dict(words_only_memory=True)]:
            self.assertGreater(float((normal-self.full(model,**flags)).abs().max()),1e-6)
        off=self.full(model,physical_enabled=False)
        self.evidence['memory'].normal_();self.evidence['words'].normal_()
        torch.testing.assert_close(off,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_generation_gradients_and_two_phase_ownership(self):
        model=self.model();self.activate(model);model.set_phase(127)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        fusion=model.fusions['1']
        expected=[model.enhancement.project.weight,model.enhancement.block.cross.q.weight,
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

    def test_first_frame_has_zero_physical_write(self):
        model=self.model();self.activate(model);c,g=model.enhancement(self.evidence)
        delta=model.fusions['1'](torch.randn(1,6,16),c,self.sample['times'],(3,1,2),model.wan.blocks[1].cross_attn,g)
        torch.testing.assert_close(delta[:,:2],torch.zeros_like(delta[:,:2]),rtol=0,atol=0)
        self.assertGreater(float(delta[:,2:].abs().max()),0.)

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
