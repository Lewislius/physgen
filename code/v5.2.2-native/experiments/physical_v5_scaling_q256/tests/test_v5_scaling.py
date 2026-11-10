"""V5 Q256 architecture, summed CE/FM gradients, schedule and cache ownership."""
from pathlib import Path
from types import SimpleNamespace
import copy
import sys
import tempfile
import unittest
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
FOUNDATION=ROOT.parent/'physical_variants/snapshots/20261029T064043Z'
sys.path[:0]=[str(ROOT),str(FOUNDATION),str(Path(__file__).resolve().parent)]
from physical_v5_scaling.common import load_config,is_allowed_gpu
from physical_v5_scaling.model import PhysicalModel,semantic_ce
from physical_v5_scaling.schedule import GROUPS,learning_rates,auxiliary_coefficient
from physical_v5_scaling.teacher import Teacher
from physical_v5_scaling.supervision import flow_matching
from fakes import SmallWan,FakeLanguage,FakeTokenizer


def config(small=False):
    cfg=load_config(ROOT/'configs/v5_ce_q256_s1000.yaml')
    if small:cfg['model'].update(width=16,qwen_width=24,q_heads=4,lora_rank=4)
    return cfg


class TestV5Scaling(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_full_kv_ce_and_separate_backwards_match_combined_objective(self):
        cfg=config(True)
        model=PhysicalModel(cfg,'cpu',wan=SmallWan())
        self.assertEqual(tuple(model.enhancement.seed.shape),(1,256,24))
        self.assertIsInstance(model.fusions['2'].key,nn.Linear)
        self.assertIsInstance(model.fusions['2'].value,nn.Linear)
        self.assertTrue(model.fusions['2'].key.weight.requires_grad)
        with torch.no_grad():model.fusions['2'].output.weight.copy_(torch.eye(16))
        groups=model.optim_groups()
        self.assertEqual({g['name'] for g in groups},set(GROUPS))
        ids=[id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(ids),len(model.learned_names))
        self.assertTrue(all(g['params'] for g in groups))
        self.assertEqual(model.set_phase(127),'interface')
        self.assertTrue(all(not p.requires_grad for p in model.wan.parameters()))
        self.assertEqual(model.set_phase(128),'joint')
        other=copy.deepcopy(model)

        backbone=nn.Module();backbone.model=nn.Module()
        backbone.model.language_model=FakeLanguage();backbone.lm_head=nn.Linear(24,30)
        backbone.requires_grad_(False)
        knowledge=SimpleNamespace(backbone=backbone,tokenizer=FakeTokenizer())
        evidence=dict(memory=torch.randn(1,7,24),memory_mask=torch.ones(1,7,dtype=torch.bool))
        clean=torch.randn(1,2,3,4,4);noise=torch.randn_like(clean);sigma=torch.tensor(.4)
        times=torch.arange(3).float()[None];text=torch.randn(1,4,24);valid=torch.ones(4,4)
        def fm(m):
            c,groups=m.enhancement(evidence)
            pred,_=m.velocity((1-sigma)*clean+sigma*noise,clean[:,:,:1],text,sigma,c,times,groups,None,valid)
            return flow_matching(pred,noise,clean,valid)
        weight=auxiliary_coefficient(cfg,128)
        loss=fm(model)+weight*semantic_ce(model,knowledge,evidence,'response')
        self.assertTrue(torch.isfinite(loss));loss.backward()
        fm(other).backward()
        (weight*semantic_ce(other,knowledge,evidence,'response')).backward()
        expected=dict(model.named_parameters());actual=dict(other.named_parameters())
        for name in model.learned_names:
            a,b=expected[name].grad,actual[name].grad
            self.assertEqual(a is None,b is None,name)
            if a is not None:torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4,msg=name)
        for name in ('enhancement.seed','bridge.weight','fusions.2.key.weight','fusions.2.value.weight','fusions.2.output.weight'):
            self.assertGreater(float(expected[name].grad.norm()),0,name)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.wan.named_parameters() if n.endswith('lora_b')))
        self.assertTrue(all(p.grad is None for p in backbone.parameters()))

    def test_schedule_phase_and_floors(self):
        cfg=config();peak=cfg['train']['peak_lrs']
        self.assertEqual(learning_rates(cfg,127)['wan_lora'],0)
        self.assertAlmostEqual(learning_rates(cfg,128)['wan_lora'],2e-5/128)
        self.assertAlmostEqual(learning_rates(cfg,191)['wan_lora'],1e-5)
        self.assertAlmostEqual(learning_rates(cfg,255)['wan_lora'],2e-5)
        for role in GROUPS:
            self.assertAlmostEqual(learning_rates(cfg,599)[role],peak[role])
            self.assertAlmostEqual(learning_rates(cfg,999)[role],peak[role]*.2)
        self.assertAlmostEqual(auxiliary_coefficient(cfg,0),.03/64)
        self.assertAlmostEqual(auxiliary_coefficient(cfg,63),.03)
        self.assertAlmostEqual(auxiliary_coefficient(cfg,599),.03)
        self.assertAlmostEqual(auxiliary_coefficient(cfg,999),.015)
        self.assertTrue(is_allowed_gpu('NVIDIA RTX PRO 6000 Blackwell Server Edition'))
        self.assertFalse(is_allowed_gpu('NVIDIA H100'))

    def test_readonly_semantic_and_shared_input_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);legacy=root/'legacy';v6=root/'v6';local=root/'local'
            (legacy/'semantic').mkdir(parents=True);(v6/'input').mkdir(parents=True)
            semantic=legacy/'semantic/known.pt';torch.save({'answer':'conditional response'},semantic)
            shared=v6/'input/known.pt';torch.save({'memory':torch.ones(1,2,24)},shared)
            before=semantic.stat().st_mtime_ns
            teacher=Teacher.__new__(Teacher);teacher.root=local;teacher.readonly_roots=[legacy,v6]
            self.assertEqual(teacher.semantic_target({'id':'known'}),'conditional response')
            with teacher.cache('input','known') as (path,value):
                self.assertEqual(path,local/'input/known.pt');self.assertEqual(value['memory'].shape,(1,2,24))
            with teacher.cache('semantic','new') as (path,value):
                self.assertIsNone(value);torch.save({'answer':'new'},path)
            self.assertEqual(semantic.stat().st_mtime_ns,before)
            self.assertFalse((legacy/'semantic/known.lock').exists())
            self.assertFalse((legacy/'semantic/new.pt').exists())
            self.assertFalse((v6/'input/known.lock').exists())


if __name__=='__main__':unittest.main()
