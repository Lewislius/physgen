"""Validate content ownership, generation credit and the raw-update intervention."""
from pathlib import Path
import copy
import importlib.util
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
import torch
from torch import nn
from torch.nn import functional as F
from physical_content.model import Enhancement,PhysicalModel
from physical_content.evidence import resample_words

torch.set_num_threads(2)
REFERENCE=bootstrap.BASE/'experiments/physical_content_v7/fixed_interface_r5/snapshots/20261031T073909Z/physical_content/model.py'
spec=importlib.util.spec_from_file_location('physical_content.r5_reference',REFERENCE)
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


class ContentModulationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(81)
        self.cfg={'variant':'physical_content','model':{'qwen_width':32,'width':16,'q_heads':4,
            'q_tokens':8,'rounds':2,'grounded_read_heads':2,'content_projection_init_scale':.1},
            'stages':{'interface':16,'joint':208},'train':{'gradient_checkpointing':True,
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

    def test_parameter_initialization_and_rng_match_fixed_interface(self):
        torch.manual_seed(42);old=reference.Enhancement(self.cfg,self.interface)
        old_rng=torch.get_rng_state().clone()
        torch.manual_seed(42);new=Enhancement(self.cfg,self.interface)
        self.assertTrue(torch.equal(torch.get_rng_state(),old_rng))
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        for name,value in old.state_dict().items():
            torch.testing.assert_close(value,new.state_dict()[name],rtol=0,atol=0)
        self.assertIn('project.weight',dict(new.named_buffers()))
        self.assertNotIn('project.weight',dict(new.named_parameters()))

    def test_fixed_update_retains_own_content_and_zero_recovers_native_anchor(self):
        model=Enhancement(self.cfg,self.interface).eval()
        words=(self.evidence['words']-model.anchor_mean)/model.anchor_scale
        anchor,groups,_=resample_words(words,self.evidence['word_groups'],8)
        native=F.linear(anchor,model.anchor_weight,model.anchor_bias)
        zero=torch.zeros_like(anchor)
        output,found=model(self.evidence,update_override=zero)
        torch.testing.assert_close(output,native)
        torch.testing.assert_close(found,groups)
        fixed=torch.randn_like(anchor)
        base,_=model(self.evidence,update_override=fixed)
        new_memory={**self.evidence,'memory':torch.randn_like(self.evidence['memory'])}
        unchanged,_=model(new_memory,update_override=fixed)
        torch.testing.assert_close(base,unchanged,rtol=0,atol=0)
        # Only the learned update is averaged. The sample's physical words are
        # still causal inputs to both its anchor and its effective correction.
        changed_words={**self.evidence,'words':self.evidence['words']+torch.randn(6,32)}
        changed,_=model(changed_words,update_override=fixed)
        self.assertGreater(float((base-changed).abs().max()),1e-4)
        learned_before,_=model(self.evidence);learned_after,_=model(new_memory)
        self.assertGreater(float((learned_before-learned_after).abs().max()),1e-5)
        with self.assertRaisesRegex(ValueError,'full Q-token/channel'):
            model(self.evidence,update_override=fixed.mean(1,keepdim=True))

    def test_future_semantic_targets_do_not_enter_generation_condition(self):
        model=Enhancement(self.cfg,self.interface).eval()
        first,_=model(self.evidence)
        changed={**self.evidence,'native_words':torch.randn(6,16)*100,
            'future_video':torch.randn(2,3,4)}
        second,_=model(changed)
        torch.testing.assert_close(first,second,rtol=0,atol=0)

    def test_optimizer_resume_keeps_fixed_coordinates_and_trainable_reader(self):
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
