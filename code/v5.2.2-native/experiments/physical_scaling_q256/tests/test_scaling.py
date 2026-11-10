"""Q256 gradient execution, normalization, schedule, resume and cache ownership."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import unittest
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT.parent/'physical_variants/snapshots/20261029T064043Z'),str(Path(__file__).resolve().parent)]
from physical_scaling.common import load_config,is_allowed_gpu
from physical_scaling.model import Enhancement,Fusion,PhysicalModel
from physical_scaling.supervision import slot_groups,group_log_prior,grouped_probabilities,flow_matching
from physical_scaling.schedule import GROUPS,learning_rates,auxiliary_coefficient
from physical_scaling.teacher import Teacher
from physical_scaling.train import restore
from physgen_native.data import SampleOrder
from fakes import Attention,SmallWan


def config(variant='v6_assoc',small=False):
    cfg=load_config(ROOT/'configs'/f'{variant}_q256_s1000.yaml')
    if small:
        cfg['model'].update(width=16,qwen_width=24,q_heads=4,lora_rank=4)
    return cfg


class TestScaling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17);torch.set_num_threads(2)

    def test_q256_forward_backward_for_both_actual_fusion_variants(self):
        for variant in ('v6_write','v6_assoc'):
            cfg=config(variant,True)
            model=PhysicalModel(cfg,'cpu',wan=SmallWan())
            optim=torch.optim.AdamW(model.optim_groups(),foreach=False)
            self.assertEqual({g['name'] for g in optim.param_groups},set(GROUPS))
            parameters=[p for g in optim.param_groups for p in g['params']]
            self.assertEqual(len(parameters),len({id(p) for p in parameters}))
            self.assertEqual(len(parameters),len(model.learned_names))
            evidence=dict(memory=torch.randn(1,7,24),memory_mask=torch.ones(1,7,dtype=torch.bool),
                          relation_vectors=torch.randn(1,3,24))
            clean=torch.randn(1,2,3,4,4);noise=torch.randn_like(clean);sigma=torch.tensor(.4)
            first=clean[:,:,:1];times=torch.arange(3).float()[None];text=torch.randn(1,4,24)
            valid=torch.ones(4,4)
            if variant=='v6_assoc':target={'association':torch.softmax(torch.randn(1,8,4),-1)}
            else:target={'support':torch.rand(1,8,3),'text_anchors':torch.randn(1,3,16)}
            self.assertEqual(model.set_phase(127),'interface')
            for iteration in range(2):
                optim.zero_grad(set_to_none=True)
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    c,groups=model.enhancement(evidence)
                    self.assertEqual(tuple(c.shape),(1,256,16))
                    pred,aux=model.velocity((1-sigma)*clean+sigma*noise,first,text,sigma,c,times,groups,target,valid)
                    loss=flow_matching(pred,noise,clean,valid)+.02*aux
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertGreater(float(model.fusions['2'].output.weight.grad.norm()),0)
                if iteration or variant=='v6_assoc':
                    self.assertGreater(float(model.enhancement.seed.grad.norm()),0)
                self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if n.endswith(('lora_a','lora_b'))))
                optim.step()
            self.assertEqual(model.set_phase(128),'joint')
            optim.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                c,groups=model.enhancement(evidence)
                pred,aux=model.velocity((1-sigma)*clean+sigma*noise,first,text,sigma,c,times,groups,target,valid)
                loss=flow_matching(pred,noise,clean,valid)+.02*aux
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.wan.named_parameters() if n.endswith('lora_b')))
            self.assertTrue(all(p.grad is None for n,p in model.wan.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_doubling_identical_slots_preserves_both_fusion_outputs(self):
        attention=Attention()
        x=torch.randn(1,12,16);text=torch.randn(1,5,16);c=torch.randn(1,128,16)
        groups=slot_groups(128,3)
        for variant in ('v6_write','v6_assoc'):
            fusion=Fusion(attention,variant,4)
            with torch.no_grad():fusion.output.weight.copy_(torch.eye(16))
            with torch.autocast('cpu',dtype=torch.bfloat16):
                a,_=fusion(x,text,c,torch.arange(3).float()[None],(3,2,2),attention,groups)
                b,_=fusion(x,text,c.repeat_interleave(2,1),torch.arange(3).float()[None],(3,2,2),attention,groups.repeat_interleave(2))
            torch.testing.assert_close(a,b,atol=.02,rtol=.02)
        actual=slot_groups(256,3)
        p=group_log_prior(actual).softmax(-1)[None,None,None]
        torch.testing.assert_close(grouped_probabilities(p,actual),torch.full((1,1,4),.25))

    def test_warmup_hold_decay_and_stage_boundary(self):
        cfg=config()
        rates=[learning_rates(cfg,s) for s in range(1000)]
        peak=cfg['train']['peak_lrs']
        self.assertAlmostEqual(rates[0]['enhancement'],peak['enhancement']/32)
        for step in (31,127,128,599,600):
            for role in GROUPS[:-1]:self.assertAlmostEqual(rates[step][role],peak[role])
        self.assertEqual(rates[127]['wan_lora'],0)
        interface=cfg['stages']['interface']
        warmup=cfg['train']['wan_warmup']
        self.assertAlmostEqual(rates[interface]['wan_lora'],peak['wan_lora']/warmup)
        self.assertAlmostEqual(rates[interface+warmup-1]['wan_lora'],peak['wan_lora'])
        for role in GROUPS:
            self.assertAlmostEqual(rates[-1][role],peak[role]*.2)
            self.assertTrue(all(rates[s+1][role]<=rates[s][role] for s in range(600,999)))
        for variant in ('v6_write','v6_assoc'):
            cfg=config(variant)
            self.assertAlmostEqual(auxiliary_coefficient(cfg,31),cfg['loss']['peak_weight'])
            self.assertAlmostEqual(auxiliary_coefficient(cfg,599),cfg['loss']['peak_weight'])
            self.assertAlmostEqual(auxiliary_coefficient(cfg,999),cfg['loss']['final_weight'])
            with self.assertRaises(ValueError):auxiliary_coefficient(cfg,1000)

    def test_overlay_reads_legacy_without_modifying_it_and_new_writes_stay_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';new=root/'new';(old/'input').mkdir(parents=True)
            previous=old/'input/known.pt';torch.save({'value':torch.tensor([2.])},previous)
            before=previous.stat()
            teacher=Teacher.__new__(Teacher);teacher.root=new;teacher.readonly_root=old
            with teacher.cache('input','known') as (path,value):
                self.assertEqual(float(value['value']),2.)
                self.assertEqual(path,new/'input/known.pt')
            self.assertFalse((old/'input/known.lock').exists())
            self.assertEqual(previous.stat().st_mtime_ns,before.st_mtime_ns)
            with teacher.cache('input','new') as (path,value):
                self.assertIsNone(value);torch.save({'value':1},path)
            self.assertFalse((old/'input/new.pt').exists())

    def test_config_and_resume_reject_old_revision_or_q128(self):
        import yaml
        cfg=config()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'wrong.yaml'
            bad=copy.deepcopy(cfg);bad['model']['q_tokens']=128
            path.write_text(yaml.safe_dump(bad))
            with self.assertRaises(ValueError):load_config(path)
            state=Path(tmp)/'old.pt'
            torch.save({'revision':'physical_variants_20261029_r1','variant':'v6_assoc','config':cfg},state)
            with self.assertRaises(ValueError):restore(None,None,state,cfg)

    def test_sample_cursor_resume_across_epoch_boundary(self):
        import itertools
        all_ids=list(itertools.islice(iter(SampleOrder(3666,42)),4000))
        tail=list(itertools.islice(iter(SampleOrder(3666,42,3658)),342))
        self.assertEqual(tail,all_ids[3658:])
        self.assertEqual(len(set(all_ids[:3666])),3666)
        self.assertEqual(len(set(all_ids)),3666)

    def test_single_card_gpu_policy_and_independent_outputs(self):
        for variant in ('v6_write','v6_assoc'):
            cfg=config(variant)
            self.assertEqual(cfg['stages'],dict(interface=128,joint=872))
            self.assertEqual(cfg['model']['q_tokens'],256)
            self.assertTrue(Path(cfg['paths']['runs']).is_relative_to(ROOT))
            self.assertEqual(cfg['verifier'],{'kind':'none'})
        self.assertTrue(is_allowed_gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition'))
        self.assertFalse(is_allowed_gpu('NVIDIA H100'))


if __name__=='__main__':unittest.main()
