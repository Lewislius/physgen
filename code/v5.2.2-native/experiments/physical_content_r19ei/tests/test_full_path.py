"""Actual project generation path, with a small Wan-shaped test backbone."""
from pathlib import Path
import sys
import unittest
import tempfile
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tests'))
from fake_wan import FakeWan, W, FF, evidence
from physical_content.common import load_config
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_micro


def model():
    cfg=load_config(ROOT/'configs/r19ei.yaml')
    cfg['model'].update(width=W,ffn_width=FF,native_heads=2,injection_layers=[0,2],
                        q_tokens=8,q_heads=4,qwen_width=16,state_initialization_layer=1)
    cfg['train']['gradient_checkpointing']=True
    wan=FakeWan()
    wan.text_embedding[0]=nn.Linear(16,W)
    wan.time_embedding=nn.Linear(256,W)
    wan.time_projection=nn.Linear(W,6*W)
    interface=dict(mean=torch.zeros(16),scale=torch.ones(()))
    return PhysicalModel(cfg,device='cpu',wan=wan,interface=interface)


def inputs():
    ev=evidence(width=16);ev.update(scope='first_image_and_request_only',native_condition=torch.randn(12,W),visual_tokens=torch.randn(1,9,16))
    sample=dict(latent=torch.randn(1,48,3,4,4),text=torch.randn(1,7,16),
                valid=torch.ones(4,4),times=torch.tensor([[0.,.5,1.]]))
    sample['first']=sample['latent'][:,:,:1].clone()
    return ev,sample


class WholePathTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42);torch.set_num_threads(2)

    def test_real_fm_reaches_all_paths_and_owns_parameters_once(self):
        m=model();ev,sample=inputs();m.set_phase(0)
        c,g=m.prepare_content(ev,sample['text'])
        args=(sample['latent'],sample['first'],sample['text'],torch.tensor(.5),c,g,sample['times'])
        torch.testing.assert_close(m.velocity(*args)[0],m.velocity(*args,physical_mode='off')[0],atol=0,rtol=0)
        opt=torch.optim.AdamW(m.optim_groups(),lr=1e-3)
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            report=backward_micro(m,ev,sample,sample['text'],torch.tensor(.5),
                                  torch.randn_like(sample['latent']),0.,1)
            self.assertTrue(torch.isfinite(torch.tensor(report['generation'])))
            opt.step()
        paths=['enhancement.physical_read.attention.out_proj.weight',
               'enhancement.scene_read.attention.in_proj_weight',
               'enhancement.image_attention.attention.out_proj.weight',
               'enhancement.image_ffn.layers.0.weight',
               'state_expert.reference_attention.q.weight',
               'state_expert.reference_attention.out_proj.weight',
               'enhancement.semantic_attention.attention.out_proj.weight',
               'enhancement.route_delta.weight','enhancement.execute_delta.weight',
               'physical_value.weight','state_expert.condition_attention.v.weight',
               'state_expert.relation_attention.q.weight',
               'state_expert.temporal_attention.q.weight',
               'state_expert.ffn.layers.0.weight','state_expert.noise_modulation.weight',
               'mechanisms.0.video_query.weight','mechanisms.0.content_key.weight',
               'mechanisms.0.feature_output.weight']
        named=dict(m.named_parameters())
        for name in paths:
            self.assertIsNotNone(named[name].grad,name)
            self.assertGreater(named[name].grad.norm().item(),0.,name)
        owned=[id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(owned),len(set(owned)))
        self.assertIs(m.mechanisms['0'].video_value,m.mechanisms['2'].video_value)
        self.assertTrue(all(p.grad is None for b in m.wan.blocks for p in b.cross_attn.parameters()))
        self.assertFalse(any('cross_attn' in n for n in m.learned_names if n.startswith('wan.')))
        m.base_bank.assert_frozen()

    def test_original_text_read_once_and_no_future_input(self):
        m=model();ev,sample=inputs();seen=[]
        with torch.no_grad():m.mechanisms['0'].feature_output.weight.normal_(0.,.01)
        handle=m.wan.blocks[0].cross_attn.q.register_forward_pre_hook(
            lambda module,args:seen.append(args[0].detach().clone()))
        c,g=m.prepare_content(ev,sample['text'])
        args=(sample['latent'],sample['first'],sample['text'],torch.tensor(.5),c,g,sample['times'])
        try:
            m.checkpointing=False
            m.velocity(*args);m.velocity(*args,physical_mode='off')
        finally:handle.remove()
        self.assertEqual(len(seen),2)
        torch.testing.assert_close(seen[0],seen[1],atol=0,rtol=0)
        changed=list(args);changed[1]=None
        with self.assertRaisesRegex(ValueError,'known first latent'):m.velocity(*changed)

    def test_module_interventions_have_actual_effects(self):
        m=model();ev,sample=inputs()
        with torch.no_grad():
            for branch in m.mechanisms.values():branch.feature_output.weight.normal_(0.,.05)
            m.state_expert.reference_attention.out_proj.weight.normal_(0.,.05)
        c,g=m.prepare_content(ev,sample['text'])
        args=(sample['latent'],sample['first'],sample['text'],torch.tensor(.5),c,g,sample['times'])
        normal=m.velocity(*args)[0]
        for mode in ('off','no_video_feedback','no_temporal','scene_memory_off'):
            self.assertGreater((m.velocity(*args,physical_mode=mode)[0]-normal).norm().item(),1e-6)
        nosemantic,_=m.prepare_content(ev,sample['text'],semantic_expert=False)
        self.assertGreater((c['routing']-nosemantic['routing']).norm().item(),1e-6)

    def test_checkpoint_preserves_shared_parameters_and_condition_constants(self):
        from physical_content.checkpointing import save_checkpoint,restore_checkpoint
        torch.manual_seed(77);m=model();opt=torch.optim.AdamW(m.optim_groups())
        ev,sample=inputs()
        backward_micro(m,ev,sample,sample['text'],torch.tensor(.5),
                       torch.randn_like(sample['latent']),0.,1)
        opt.step()
        with tempfile.TemporaryDirectory() as folder:
            path=save_checkpoint(m,opt,m.cfg,folder,1,4)
            torch.manual_seed(77);restored=model()
            optimizer=torch.optim.AdamW(restored.optim_groups())
            update,cursor,_=restore_checkpoint(restored,optimizer,path,restored.cfg)
            self.assertEqual((update,cursor),(1,4))
            actual=dict(restored.named_parameters())
            for name,p in m.named_parameters():
                if name in m.learned_names:torch.testing.assert_close(p,actual[name])
            self.assertIs(restored.physical_value,restored.mechanisms['2'].video_value)
            torch.testing.assert_close(m.condition_mean,restored.condition_mean)

    def test_known_image_receives_no_direct_physical_write(self):
        m=model();ev,sample=inputs();m.checkpointing=False
        with torch.no_grad():
            for branch in m.mechanisms.values():branch.feature_output.weight.normal_(0.,.05)
        before,after=[],[]
        handles=[m.wan.blocks[0].ffn[0].register_forward_hook(
                    lambda module,args,out:before.append(out.detach().clone())),
                 m.wan.blocks[0].ffn[1].register_forward_pre_hook(
                    lambda module,args:after.append(args[0].detach().clone()))]
        try:
            c,g=m.prepare_content(ev,sample['text'])
            m.velocity(sample['latent'],sample['first'],sample['text'],torch.tensor(.5),
                       c,g,sample['times'])
        finally:
            for handle in handles:handle.remove()
        self.assertEqual(len(before),1)
        _,geometry,known=m._geometry(sample['latent'],sample['first'])
        difference=after[0]-before[0]
        self.assertEqual(difference[:,:known].abs().max().item(),0.)
        self.assertGreater(difference[:,known:].norm().item(),0.)


if __name__=='__main__':unittest.main()
