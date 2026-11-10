"""Check evidence routing and first-update gradients through the complete C/fusion path."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
torch.set_num_threads(2)
from physical_content.model import ContentBlock,Enhancement,Fusion
from test_native_output import Attention


class GroundedReadTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_local_read_keeps_other_relations_separate_and_global_read_available(self):
        block=ContentBlock(32,4,2)
        query=torch.randn(1,8,32);memory=torch.randn(1,7,32)
        words=torch.randn(1,6,32)
        groups=torch.arange(8)//4;word_groups=torch.arange(6)//3
        valid=torch.ones(1,7,dtype=torch.bool)
        read=lambda m,w:block.read_heads(query,m,valid,w,word_groups,groups)
        original=read(memory,words)
        other=words.clone();other[:,3:]=torch.randn_like(other[:,3:])*2
        changed=read(memory,other)
        torch.testing.assert_close(changed[:,:2,:4],original[:,:2,:4])
        torch.testing.assert_close(changed[:,2:],original[:,2:])
        self.assertGreater(float((changed[:,:2,4:]-original[:,:2,4:]).abs().max()),1e-4)
        new_context=read(torch.randn_like(memory),words)
        torch.testing.assert_close(new_context[:,:2],original[:,:2])
        self.assertGreater(float((new_context[:,2:]-original[:,2:]).abs().max()),1e-4)

    def test_generation_gradient_reaches_seed_and_reader_on_first_update(self):
        cfg={'model':{'qwen_width':32,'width':16,'q_heads':4,'q_tokens':8,'rounds':2,
            'grounded_read_heads':2,'content_projection_init_scale':.1},
            'train':{'gradient_checkpointing':True}}
        interface={'mean':torch.zeros(32),'scale':torch.ones(32),'weight':torch.randn(16,32)*.05,
            'bias':torch.zeros(16),'target_scale':torch.tensor(.1)}
        enhancement=Enhancement(cfg,interface)
        attention=Attention().requires_grad_(False);fusion=Fusion(attention,.25)
        evidence={'words':torch.randn(6,32,requires_grad=True),'word_groups':torch.arange(6)//3,
            'memory':torch.randn(1,7,32,requires_grad=True),'memory_mask':torch.ones(1,7,dtype=torch.bool)}
        c,groups=enhancement(evidence)
        self.assertEqual(tuple(c.shape),(1,8,16))
        torch.testing.assert_close(enhancement.project.weight,interface['weight']*.1)
        x=torch.randn(1,4,16);context=torch.randn(1,5,16)
        result,_=fusion(x,context,c,torch.tensor([[0.,1.]]),(2,1,2),attention,groups)
        torch.testing.assert_close(result[:,:2],attention(x,context)[:,:2])
        result[:,2:].square().mean().backward()
        for parameter in [enhancement.seed,enhancement.block.cross.q.weight,
                          enhancement.block.cross.k.weight,enhancement.block.cross.v.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.norm()),0.)
        self.assertIsNone(evidence['words'].grad)
        self.assertIsNone(evidence['memory'].grad)

    def test_local_read_and_anchor_share_input_feature_coordinates(self):
        cfg={'model':{'qwen_width':32,'width':16,'q_heads':4,'q_tokens':8,'rounds':2,
            'grounded_read_heads':2,'content_projection_init_scale':.1},
            'train':{'gradient_checkpointing':False}}
        interface={'mean':torch.randn(32),'scale':torch.rand(32)+.5,'weight':torch.randn(16,32)*.05,
            'bias':torch.randn(16)*.01,'target_scale':torch.tensor(.1)}
        model=Enhancement(cfg,interface).eval()
        evidence={'words':torch.randn(6,32),'word_groups':torch.arange(6)//3,
            'memory':torch.randn(1,7,32),'memory_mask':torch.ones(1,7,dtype=torch.bool)}
        with torch.no_grad():
            before,_=model(evidence)
            # A change of feature origin, represented in both inputs and their
            # frozen training mean, cannot change the local physical content.
            shift=torch.randn(32)*2
            model.anchor_mean.add_(shift)
            evidence['words']=evidence['words']+shift
            after,_=model(evidence)
        torch.testing.assert_close(before,after,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()
