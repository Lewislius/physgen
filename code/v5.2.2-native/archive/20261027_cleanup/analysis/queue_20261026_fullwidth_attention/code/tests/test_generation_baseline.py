"""Objective and initialization checks, not claims of benchmark improvement."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from test_qwen_process import small_model
from test_native import grad_size
from physgen_native.config import read_config, validate
from physgen_native.sequence import temporal_flow_error
from physgen_native.runtime import initialize_wan_adapter, parameter_groups, schedule


def baseline_model():
    model,data=small_model()
    recipe=read_config(ROOT/'configs/native_generation_baseline_wisa3857.yaml')
    for key in ('revision','loss','flow_temporal'):model.cfg[key]=copy.deepcopy(recipe[key])
    model.cfg['knowledge'].update({k:v for k,v in recipe['knowledge'].items() if k.startswith('video_observation') or k=='video_comparison'})
    model.cfg['stages']=dict(interface=1,calibration=1,joint=2)
    model.cfg['sequence']['max_sigma']=.35
    model.cfg['verifier']['calibrate_current']=True
    model.cfg['train']['wan_lora_in_interface']=True
    validate(model.cfg);model.set_phase(0)
    return model,data


class GenerationBaselineTests(unittest.TestCase):
    def test_temporal_fm_keeps_mean_optimum_for_ambiguous_future_targets(self):
        a=torch.randn(1,3,21,2,4);b=torch.randn_like(a)
        mean=((a+b)/2).requires_grad_(True)
        loss=(temporal_flow_error(mean,a,torch.ones(2,4),[1,2,4,8,16])+
              temporal_flow_error(mean,b,torch.ones(2,4),[1,2,4,8,16]))/2
        gradient,=torch.autograd.grad(loss,mean)
        self.assertLess(float(gradient.abs().max()),1e-7)
        late=a.clone();late[:,:,20]+=1;late.requires_grad_(True)
        value=temporal_flow_error(late,a,torch.ones(2,4),[16])
        grad,=torch.autograd.grad(value,late)
        self.assertGreater(float(grad[:,:,20].abs().sum()),0)
        self.assertEqual(float(grad[:,:,:1].abs().sum()),0)

    def test_high_noise_does_not_decode_and_regress_unique_rgb_future(self):
        model,data=baseline_model();original=model.decoder.forward;lengths=[]
        def decode(z,*args,**kwargs):
            lengths.append(z.shape[2]);return original(z,*args,**kwargs)
        with patch.object(model.decoder,'forward',side_effect=decode):
            losses,weights,metrics=model(data,torch.zeros_like(data['text']),0,sigma_override=.95)
            model.backward_losses(losses,weights)
        self.assertEqual(lengths,[1])
        self.assertEqual(float(losses['process']),0)
        self.assertGreater(float(losses['flow_temporal']),0)
        self.assertEqual(metrics['sequence_active'],0)
        self.assertGreater(grad_size(model.conditioner),0)
        self.assertGreater(grad_size(model.generator.wan),0)

    def test_low_noise_retains_complete_rgb_process_and_four_losses(self):
        model,data=baseline_model()
        losses,_,metrics=model(data,torch.zeros_like(data['text']),0,sigma_override=.2)
        self.assertEqual(set(losses),{'fm','flow_temporal','process','intensity'})
        self.assertGreater(float(losses['process']),0)
        self.assertEqual(metrics['sequence_active'],1)
        self.assertEqual([len(model.cfg['loss'][p]) for p in model.cfg['stages']],[4,1,4])
        self.assertFalse(model.should_feedback(0));model.set_phase(2);self.assertTrue(model.should_feedback(2))

    def test_wan_scheduler_active_from_first_generator_update_not_calibration(self):
        model,data=baseline_model();optimizer=torch.optim.AdamW(parameter_groups(model,model.cfg))
        scheduler=schedule(optimizer,model.cfg)
        rate=lambda:next(g['lr'] for g in optimizer.param_groups if g['name']=='wan_lora')
        self.assertGreater(rate(),0)
        optimizer.step();scheduler.step();self.assertEqual(rate(),0)
        model.zero_grad(set_to_none=True)
        losses,weights,_=model(data,torch.zeros_like(data['text']),1)
        model.backward_losses(losses,weights)
        self.assertGreater(grad_size(model.critic),0)
        self.assertEqual(grad_size(model.generator)+grad_size(model.knowledge)+grad_size(model.conditioner),0)

    def test_existing_wan_adapter_copy_is_exact_and_rejects_partial_weights(self):
        model,_=baseline_model()
        weights={n:torch.randn_like(p) for n,p in model.generator.wan.named_parameters() if n.endswith(('lora_a','lora_b'))}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wan.pt'
            state=dict(config=model.cfg,step=400,wan_lora=weights)
            torch.save(state,path);initialize_wan_adapter(model,model.cfg,path)
            for n,p in model.generator.wan.named_parameters():
                if n in weights:torch.testing.assert_close(p,weights[n],atol=0,rtol=0)
            del weights[next(iter(weights))];torch.save(state,path)
            with self.assertRaises(ValueError):initialize_wan_adapter(model,model.cfg,path)


if __name__=='__main__':unittest.main()
