"""Integration checks using the installed Wan operators, RoPE and gradients."""
from pathlib import Path
import copy,importlib.util,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from physical_content.model import PhysicalModel,Fusion,physical_self_attention
from physical_content.objectives import backward_sample
from physical_content.train import save,restore
torch.set_num_threads(2)
REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/snapshots/20261101T034557Z'
spec=importlib.util.spec_from_file_location('physical_content.r10_reference',REFERENCE/'physical_content/model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('physical_content.r13b_reference',
    REFERENCE.parents[2]/'interaction_query_r13b/snapshots/20261104T214359Z/physical_content/model.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)

def tiny_wan():
    package=types.ModuleType('_reciprocal_test_wan');package.__path__=[]
    attention=types.ModuleType('_reciprocal_test_wan.attention')
    def cpu_attention(q,k,v,**kwargs):
        return F.scaled_dot_product_attention(q.transpose(1,2).float(),k.transpose(1,2).float(),
            v.transpose(1,2).float()).transpose(1,2).to(v.dtype)
    attention.flash_attention=cpu_attention
    module=sys.modules.get('_reciprocal_test_wan.model')
    if module is None:
        sys.modules['_reciprocal_test_wan']=package
        sys.modules['_reciprocal_test_wan.attention']=attention
        spec=importlib.util.spec_from_file_location('_reciprocal_test_wan.model',
            '/home/liuzhirui/model/Wan2.2/wan/modules/model.py')
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    wan=module.WanModel(model_type='ti2v',patch_size=(1,2,2),text_len=5,in_dim=2,out_dim=2,
        dim=16,ffn_dim=32,freq_dim=8,text_dim=16,num_heads=2,num_layers=2)
    with torch.no_grad():wan.head.head.weight.normal_(std=.1)
    return wan


class CoupledAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_coupled_attention_r13c',
            model=dict(qwen_width=32,width=16,q_heads=4,q_tokens=8,rounds=2,grounded_read_heads=2,condition_every=2,lora_rank=2),
            paths=dict(interface=str(self.path/'interface.pt')),stages=dict(interface=32,joint=368),
            loss=dict(generation=dict(tail_weight=.2,tail_fraction=.25)),
            train=dict(seed=42,gradient_checkpointing=True,keep_steps=[33],
                peak_lrs=dict(enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)))
        self.wan=tiny_wan()
        self.evidence=dict(words=torch.randn(6,32),word_groups=torch.arange(6)//3,
            request_words=torch.randn(5,32),native_words=torch.randn(6,16)*.1)
        self.clean=torch.randn(1,2,3,2,4);self.noise=torch.randn_like(self.clean);self.text=torch.randn(1,4,16)
        self.sample=dict(latent=self.clean,first=self.clean[:,:,:1],times=torch.tensor([[0.,1.,2.]]),valid=torch.ones(2,4))
        self.sigma=torch.tensor(.6)

    def model(self,cls=PhysicalModel):
        torch.manual_seed(42)
        return cls(copy.deepcopy(self.cfg),device='cpu',wan=copy.deepcopy(self.wan))

    def full(self,model,**kwargs):
        c,g=model.enhancement(self.evidence)
        noisy=(1-self.sigma)*self.clean+self.sigma*self.noise
        return model.velocity(noisy,self.sample['first'],self.text,self.sigma,c,self.sample['times'],g,**kwargs)[0]

    def activate(self,model):
        with torch.no_grad():
            for fusion in model.fusions.values():
                fusion.query_modulation.weight.normal_(std=.04)
                fusion.key_modulation.weight.normal_(std=.04)
            model.enhancement.project.weight.normal_(std=.01)
            model.enhancement.execution_project.weight.normal_(std=.01)

    def test_same_r10a_parameter_budget_initializer_and_native_start(self):
        model=self.model();reference=self.model(old.PhysicalModel)
        def sizes(m):return {g['name']:sum(p.numel() for p in g['params']) for g in m.optim_groups()}
        self.assertEqual(sizes(model),sizes(reference))
        for name,value in reference.state_dict().items():
            destination=name.replace('.state_value.','.query_modulation.').replace('.output.','.key_modulation.') if name.startswith('fusions.') else name
            torch.testing.assert_close(model.state_dict()[destination],value,rtol=0,atol=0)
        torch.testing.assert_close(self.full(model),self.full(reference),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)

    def test_both_modulations_act_and_query_only_matches_r13b_query_only(self):
        model=self.model();self.activate(model)
        normal=self.full(model);off=self.full(model,physical_enabled=False)
        self.assertGreater(float((normal-off).abs().max()),1e-6)
        torch.testing.assert_close(self.full(model,query_enabled=False,key_enabled=False),off,rtol=1e-5,atol=1e-6)
        for flag in ('query_enabled','key_enabled'):
            self.assertGreater(float((normal-self.full(model,**{flag:False})).abs().max()),1e-6)
        reference=self.model(previous.PhysicalModel)
        with torch.no_grad():
            source=model.state_dict()
            for name,value in reference.state_dict().items():
                destination=name.replace('.interaction.','.query_modulation.').replace('.output.','.key_modulation.') if name.startswith('fusions.') else name
                value.copy_(source[destination])
        torch.testing.assert_close(self.full(model,key_enabled=False),
            self.full(reference,direct_write_enabled=False),rtol=1e-5,atol=1e-6)

    def test_qk_change_but_native_values_and_first_frame_qk_are_preserved(self):
        attention=self.wan.blocks[1].self_attn
        x=torch.randn(1,6,16);gain=torch.randn_like(x)*.4;gain[:,:2]=0
        lengths=torch.tensor([6]);grid=torch.tensor([[3,1,2]])
        native=attention.forward.__func__.__globals__;original=native['flash_attention'];calls=[]
        def capture(**kwargs):
            calls.append({k:kwargs[k].detach().clone() for k in ('q','k','v')})
            return original(**kwargs)
        with patch.dict(native,{'flash_attention':capture}):
            baseline=attention(x,lengths,grid,self.wan.freqs)
            zero=physical_self_attention(attention,x,torch.zeros_like(gain),torch.zeros_like(gain),lengths,grid,self.wan.freqs)
            changed=physical_self_attention(attention,x,gain,gain*.7,lengths,grid,self.wan.freqs)
            query_only=physical_self_attention(attention,x,gain,None,lengths,grid,self.wan.freqs)
            key_only=physical_self_attention(attention,x,None,gain*.7,lengths,grid,self.wan.freqs)
        torch.testing.assert_close(baseline,zero,rtol=0,atol=0)
        for call in calls[1:]:torch.testing.assert_close(calls[0]['v'],call['v'],rtol=0,atol=0)
        for key in ('q','k'):
            torch.testing.assert_close(calls[0][key][:,:2],calls[2][key][:,:2],rtol=0,atol=0)
            self.assertGreater(float((calls[0][key][:,2:]-calls[2][key][:,2:]).abs().max()),1e-6)
        torch.testing.assert_close(calls[0]['k'],calls[3]['k'],rtol=0,atol=0)
        torch.testing.assert_close(calls[0]['q'],calls[4]['q'],rtol=0,atol=0)
        self.assertGreater(float((baseline-changed).abs().max()),1e-6)

    def test_single_physical_read_in_a_selected_block(self):
        model=self.model();self.activate(model);model.checkpointing=False
        fusion=model.fusions['1']
        with patch.object(fusion,'read',wraps=fusion.read) as read:
            self.full(model)
            self.assertEqual(read.call_count,1)

    def test_first_frame_is_not_directly_modulated_and_content_changes_execution(self):
        model=self.model();self.activate(model)
        fusion=model.fusions['1'];attention=model.wan.blocks[1].cross_attn
        content,groups=model.enhancement(self.evidence);state=torch.randn(1,6,16)
        gain,key_gain=fusion(state,content,self.sample['times'],(3,1,2),attention,groups)
        self.assertEqual(int(torch.count_nonzero(gain[:,:2])),0)
        self.assertEqual(int(torch.count_nonzero(key_gain[:,:2])),0)
        self.assertGreater(float(gain[:,2:].norm()),0.)
        self.assertGreater(float(key_gain[:,2:].norm()),0.)
        before=self.full(model);off_before=self.full(model,physical_enabled=False)
        self.evidence['words']=torch.randn_like(self.evidence['words'])
        self.evidence['request_words']=torch.randn_like(self.evidence['request_words'])
        self.assertGreater(float((before-self.full(model)).abs().max()),1e-6)
        torch.testing.assert_close(off_before,self.full(model,physical_enabled=False),rtol=0,atol=0)

    def test_zero_start_receives_real_generation_gradients_without_auxiliary(self):
        model=self.model();model.set_phase(0)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        for p in (model.fusions['1'].query_modulation.weight,model.fusions['1'].key_modulation.weight):
            self.assertIsNotNone(p.grad)
            self.assertGreater(float(p.grad.norm()),0.)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))

    def test_generation_gradient_ownership_in_both_phases(self):
        model=self.model();self.activate(model)
        self.assertEqual(model.set_phase(31),'interface')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,0.,1)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        for p in (model.enhancement.block.cross.v.weight,model.enhancement.execution_project.weight,
                  model.fusions['1'].key.weight,model.fusions['1'].query_modulation.weight,model.fusions['1'].key_modulation.weight):
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.norm()),0.)
        model.zero_grad(set_to_none=True);self.assertEqual(model.set_phase(32),'joint')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters()
            if not n.endswith(('lora_a','lora_b'))))
        for layer in model.wan.blocks:
            self.assertFalse(any(p.requires_grad for p in layer.cross_attn.parameters()))

    def test_checkpoint_preserves_normal_and_control_outputs(self):
        model=self.model();self.activate(model);model.set_phase(32)
        optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1);optimizer.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]):save(model,optimizer,self.cfg,self.path,33,132)
        restored=self.model();other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with patch('torch.cuda.set_rng_state_all'):
            self.assertEqual(restore(restored,other,self.path/'step0033.pt',self.cfg),(33,132))
        for flags in ({},{'physical_enabled':False},{'query_enabled':False},{'key_enabled':False}):
            torch.testing.assert_close(self.full(model,**flags),self.full(restored,**flags),rtol=0,atol=0)

    def test_optimizer_owns_every_trainable_parameter_once(self):
        model=self.model();groups=model.optim_groups();ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        parameters=dict(model.named_parameters())
        self.assertEqual(set(ids),{id(parameters[n]) for n in model.learned_names})


if __name__=='__main__':unittest.main(verbosity=2)

