"""Functional tests with the installed Wan implementation at tiny CPU sizes."""
from pathlib import Path
import copy,importlib.util,sys,tempfile,types,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch.nn import functional as F
from physical_content.model import PhysicalModel,Fusion
from physical_content.objectives import backward_sample
from physical_content.train import save,restore

torch.set_num_threads(2)
REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/execution_value_r10/snapshots/20261101T034557Z'
spec=importlib.util.spec_from_file_location('physical_content.r10_reference',REFERENCE/'physical_content/model.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)


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


class SemanticExecutionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.path=Path(self.temp.name)
        torch.save(dict(mean=torch.randn(32),scale=torch.rand(32)+.5,weight=torch.randn(16,32)*.05,
            bias=torch.randn(16)*.01,target_scale=torch.tensor(.1)),self.path/'interface.pt')
        self.cfg=dict(variant='physical_content',revision='physical_content_v7_semantic_execution_r12b',
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
                fusion.output.weight.normal_(std=.04)
                fusion.state_value.weight.normal_(std=.04)
                fusion.text_value.weight.normal_(std=.04)

    def test_preserved_initializers_and_native_initial_function(self):
        model=self.model();reference=self.model(old.PhysicalModel)
        added=set(model.learned_names)-set(reference.learned_names)
        self.assertEqual(added,{'fusions.1.text_value.weight'})
        self.assertEqual(sum(dict(model.named_parameters())[n].numel() for n in added),16**2)
        self.assertEqual(int(torch.count_nonzero(model.fusions['1'].text_value.weight)),0)
        for name,tensor in reference.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],tensor,rtol=0,atol=0)
        torch.testing.assert_close(self.full(model),self.full(reference),rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(self.full(model),self.full(model,physical_enabled=False),rtol=1e-5,atol=1e-6)
        self.assertFalse(any('request_output' in n or 'request_project' in n for n in model.learned_names))

    def test_local_native_update_modulates_physics_without_independent_write(self):
        attention=self.wan.blocks[0].cross_attn
        fusion=Fusion(attention)
        with torch.no_grad():
            fusion.output.weight.copy_(torch.eye(16));fusion.state_value.weight.copy_(torch.eye(16))
            fusion.text_value.weight.copy_(torch.eye(16))
        state=torch.randn(1,6,16);text=torch.randn_like(state);changed=text.clone()
        changed[:,2]+=torch.randn(16)
        content=dict(routing=torch.randn(1,4,16),execution=torch.randn(1,4,16))
        groups=torch.tensor([0,0,1,1]);times=torch.tensor([[0.,1.,2.]])
        def run(update,enabled=True):
            return fusion(state,content,times,(3,1,2),attention,groups,
                native_update=update,text_value_enabled=enabled)[0]
        original,edited=run(text),run(changed)
        self.assertGreater(float((edited[:,2]-original[:,2]).abs().max()),1e-5)
        torch.testing.assert_close(edited[:,3:],original[:,3:],rtol=0,atol=0)
        torch.testing.assert_close(run(changed,False),run(text,False),rtol=0,atol=0)
        self.assertEqual(int(torch.count_nonzero(original[:,:2])),0)
        content['execution'].zero_()
        self.assertEqual(int(torch.count_nonzero(run(changed))),0)

    def test_learned_common_parameters_recover_r10a_forward(self):
        model=self.model();self.activate(model)
        reference=self.model(old.PhysicalModel)
        with torch.no_grad():
            for name,p in reference.named_parameters():p.copy_(dict(model.named_parameters())[name])
        torch.testing.assert_close(self.full(model,after_text=False,text_value_enabled=False),
            self.full(reference),rtol=2e-5,atol=2e-6)
        self.assertGreater(float((self.full(model)-self.full(model,text_value_enabled=False)).abs().max()),1e-7)

    def test_post_text_addressing_and_native_text_parameter_freezing(self):
        model=self.model();self.activate(model)
        post=self.full(model);pre=self.full(model,after_text=False)
        self.assertGreater(float((post-pre).abs().max()),1e-7)
        for block in model.wan.blocks:
            self.assertFalse(any(p.requires_grad for p in block.cross_attn.parameters()))

    def test_generation_gradients_reach_semantic_physics_then_lora(self):
        model=self.model();self.activate(model)
        self.assertEqual(model.set_phase(127),'interface')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertTrue(all(p.grad is None for p in model.wan.parameters()))
        for p in [model.enhancement.execution_project.weight,model.fusions['1'].key.weight,
                  model.fusions['1'].state_value.weight,model.fusions['1'].text_value.weight,
                  model.fusions['1'].output.weight]:
            self.assertIsNotNone(p.grad);self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.norm()),0.)
        model.zero_grad(set_to_none=True);self.assertEqual(model.set_phase(128),'joint')
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1)
        self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.wan.named_parameters()
            if n.endswith(('lora_a','lora_b')) and p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_checkpoint_restore_preserves_full_and_control_outputs(self):
        model=self.model();self.activate(model);model.set_phase(128)
        optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        backward_sample(model,self.evidence,self.sample,self.text,self.sigma,self.noise,.03,1);optimizer.step()
        with patch('torch.cuda.get_rng_state_all',return_value=[]):save(model,optimizer,self.cfg,self.path,129,516)
        restored=self.model();other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with patch('torch.cuda.set_rng_state_all'):self.assertEqual(restore(restored,other,self.path/'step0129.pt',self.cfg),(129,516))
        for flags in ({},{'physical_enabled':False},{'text_value_enabled':False},{'after_text':False},
                      {'after_text':False,'text_value_enabled':False}):
            torch.testing.assert_close(self.full(model,**flags),self.full(restored,**flags),rtol=0,atol=0)

    def test_optimizer_ownership_is_unique_and_complete(self):
        model=self.model();groups=model.optim_groups();ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        parameters=dict(model.named_parameters())
        self.assertEqual(set(ids),{id(parameters[n]) for n in model.learned_names})


if __name__=='__main__':unittest.main(verbosity=2)
