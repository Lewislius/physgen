"""Behavioral checks for T2V native-feature execution and parameter ownership."""
import sys
import unittest
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fake_wan import FakeWan, W, FF, evidence
from physical_content.common import ROOT, load_config
from physical_content.model import PhysicalModel


def small_model():
    cfg=load_config(ROOT/'configs/r19ct.yaml')
    cfg['model'].update(width=W,ffn_width=FF,injection_layers=[0],
                        q_tokens=8,q_heads=4,grounded_heads=2,qwen_width=16)
    cfg['train']['gradient_checkpointing']=False
    interface=dict(mean=torch.zeros(16),scale=torch.ones(16),
                   weight=torch.randn(W,16)*.01,bias=torch.zeros(W),target_scale=torch.tensor(1.))
    return PhysicalModel(cfg,device='cpu',wan=FakeWan(),interface=interface)


def inputs(model):
    ev=evidence(width=16);ev.update(scope='request_only',native_condition=torch.randn(12,W))
    text=torch.randn(1,7,4096);c,g=model.prepare_content(ev,text)
    noisy=torch.randn(1,48,2,4,4)
    return noisy,None,text,torch.tensor(.5),c,g,torch.tensor([[0.,.5]])


class NativeFeatureTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)

    def test_initial_neutrality_and_real_generation_gradients(self):
        m=small_model();m.set_phase(0);args=inputs(m);branch=m.mechanisms['0']
        off=m.velocity(*args,physical_mode='off')[0];normal=m.velocity(*args)[0]
        torch.testing.assert_close(normal,off,atol=0,rtol=0)
        normal.square().mean().backward()
        self.assertGreater(branch.condition.weight.grad.detach().norm().item(),0.)
        m.zero_grad(set_to_none=True)
        with torch.no_grad():branch.condition.weight.normal_(0.,.05)
        # Recreate C to backpropagate through the whole enhancement graph.
        args=inputs(m);m.checkpointing=True
        m.velocity(*args)[0].square().mean().backward()
        for module in [branch.physical.query_projection,branch.physical.key,
                       branch.physical.value,m.enhancement.execution_project,
                       m.enhancement.project]:
            self.assertIsNotNone(module.weight.grad)
            self.assertGreater(module.weight.grad.detach().norm().item(),0.)
        self.assertTrue(all(p.grad is None for p in m.wan.blocks[0].cross_attn.parameters()))
        self.assertEqual({g['role'] for g in m.optim_groups()},
                         {'wan_video_lora','enhancement','physics_read','physics_write'})
        m.base_bank.assert_frozen()

    def test_unchanged_text_read_and_shared_native_output_execution(self):
        m=small_model();block=m.wan.blocks[0];branch=m.mechanisms['0'];seen={}
        with torch.no_grad():branch.condition.weight.normal_(0.,.1)
        def hook(key):
            def save(module,args):seen.setdefault(key,[]).append(args[0].detach().clone())
            return save
        handles=[block.cross_attn.q.register_forward_pre_hook(hook('text')),
                 block.ffn[2].register_forward_pre_hook(hook('native_output')),
                 branch.physical.query_projection.register_forward_pre_hook(hook('physical'))]
        args=inputs(m)
        try:
            normal=m.velocity(*args)[0]
            off=m.velocity(*args,physical_mode='off')[0]
        finally:
            for h in handles:h.remove()
        self.assertEqual(len(seen['text']),2)
        self.assertEqual(len(seen['native_output']),2)
        torch.testing.assert_close(seen['text'][0],seen['text'][1],atol=0,rtol=0)
        torch.testing.assert_close(seen['text'][0],seen['physical'][0],atol=0,rtol=0)
        self.assertGreater((seen['native_output'][0]-seen['native_output'][1]).norm().item(),0.)
        self.assertGreater((normal-off).detach().norm().item(),0.)

    def test_read_strength_and_native_feature_dependence(self):
        branch=small_model().mechanisms['0']
        with torch.no_grad():branch.condition.weight.normal_(0.,.2)
        f=torch.randn(1,8,FF);r=torch.randn(1,8,W);mask=torch.ones(1,8,1)
        y=branch(f,r,mask)
        self.assertGreater(y.detach().norm().item(),0.)
        torch.testing.assert_close(branch(f,r*.1,mask),y*.1,rtol=1e-5,atol=1e-7)
        for output in [branch(f,torch.zeros_like(r),mask),
                       branch(torch.zeros_like(f),r,mask)]:
            torch.testing.assert_close(output,torch.zeros_like(y),atol=0,rtol=0)

    def test_t2v_rejects_first_frame_condition(self):
        m=small_model();args=list(inputs(m));args[1]=args[0][:,:,:1].clone()
        with self.assertRaisesRegex(ValueError,'external first frame'):
            m.velocity(*args)


if __name__=='__main__':unittest.main()
