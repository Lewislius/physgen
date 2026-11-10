"""Check the controlled change, generation credit and checkpoint ownership."""
from pathlib import Path
import copy
import importlib.util
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from physical_content.model import Enhancement, PhysicalModel

torch.set_num_threads(2)
REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/grounded_read_r4/snapshots/20261031T064033Z/physical_content/model.py'
spec=importlib.util.spec_from_file_location('physical_content.r4_reference',REFERENCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


class FixedInterfaceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.cfg={'variant':'physical_content','model':{'qwen_width':32,'width':16,'q_heads':4,
            'q_tokens':8,'rounds':2,'grounded_read_heads':2,'content_projection_init_scale':.1},
            'stages':{'interface':16,'joint':48},'train':{'gradient_checkpointing':True,
                'peak_lrs':dict(enhancement=.001,fusion_read=.001,fusion_write=.001,wan_lora=.001)}}
        self.interface={'mean':torch.randn(32),'scale':torch.rand(32)+.5,
            'weight':torch.randn(16,32)*.05,'bias':torch.randn(16)*.01,'target_scale':torch.tensor(.1)}
        self.evidence={'words':torch.randn(6,32),'word_groups':torch.arange(6)//3,
            'memory':torch.randn(1,7,32),'memory_mask':torch.ones(1,7,dtype=torch.bool)}

    def holder(self):
        result=PhysicalModel.__new__(PhysicalModel);nn.Module.__init__(result)
        result.cfg=self.cfg;result.wan=nn.Module();result.fusions=nn.ModuleDict()
        result.enhancement=Enhancement(self.cfg,self.interface)
        result.learned_names=tuple(name for name,p in result.named_parameters() if p.requires_grad)
        return result

    def test_initial_weights_outputs_and_rng_match_r4(self):
        torch.manual_seed(42);old=reference.Enhancement(self.cfg,self.interface)
        old_rng=torch.get_rng_state().clone()
        torch.manual_seed(42);new=Enhancement(self.cfg,self.interface)
        self.assertTrue(torch.equal(torch.get_rng_state(),old_rng))
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for name,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
        torch.testing.assert_close(old(self.evidence)[0],new(self.evidence)[0],rtol=0,atol=0)
        self.assertIn('project.weight',dict(new.named_buffers()))
        self.assertNotIn('project.weight',dict(new.named_parameters()))

    def test_optimizer_and_restore_preserve_fixed_map_and_train_reader(self):
        model=self.holder();optimizer=torch.optim.AdamW(model.optim_groups(),foreach=False)
        fixed=model.enhancement.project.weight.clone();seed=model.enhancement.seed.detach().clone()
        def update(current,opt):
            opt.zero_grad(set_to_none=True)
            c,_=current.enhancement(self.evidence)
            c.square().mean().backward();opt.step()
        update(model,optimizer)
        self.assertGreater(float((model.enhancement.seed-seed).abs().max()),0)
        torch.testing.assert_close(model.enhancement.project.weight,fixed,rtol=0,atol=0)
        saved=dict(model={n:p.detach().clone() for n,p in model.named_parameters() if n in model.learned_names},
            constants={n:v.clone() for n,v in model.constant_state().items()},
            optimizer=copy.deepcopy(optimizer.state_dict()))
        self.assertIn('enhancement.project.weight',saved['constants'])
        self.assertNotIn('enhancement.project.weight',saved['model'])
        restored=self.holder();other=torch.optim.AdamW(restored.optim_groups(),foreach=False)
        with torch.no_grad():
            named=dict(restored.named_parameters());buffers=dict(restored.named_buffers())
            for name,value in saved['model'].items():named[name].copy_(value)
            for name,value in saved['constants'].items():buffers[name].copy_(value)
        other.load_state_dict(saved['optimizer'])
        restored.set_phase(0);restored.set_phase(16)
        update(model,optimizer);update(restored,other)
        torch.testing.assert_close(model.enhancement(self.evidence)[0],restored.enhancement(self.evidence)[0],rtol=0,atol=0)
        torch.testing.assert_close(restored.enhancement.project.weight,fixed,rtol=0,atol=0)


if __name__=='__main__':unittest.main()
