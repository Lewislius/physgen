"""The requested ablation has no judge and still trains C / Wan correctly."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
import torch
from torch import nn
import yaml
from test_simple_verifier import simple_model
from test_native import FakeWan, grad_size
from test_fullwidth_attention import NativeBlock
from test_native_geometry import GeometryDecoder
from physgen_native.config import read_config, validate, smoke_config
from physgen_native.knowledge import QwenKnowledge
from physgen_native.system import PhysGenNative
from physgen_native.runtime import parameter_groups, schedule, save_checkpoint, load_checkpoint, restore
from tools.gpu_policy import is_rtx_pro_6000


def no_verifier_model(query):
    old, sample = simple_model(query=query)
    mode = 'q_seed128_message' if query else 'h_content_message'
    production = read_config(ROOT/'configs'/f'native_{mode}_no_verifier_simple_wisa3857.yaml')
    cfg = copy.deepcopy(old.cfg)
    cfg.update(revision=production['revision'], verifier={'kind':'none'}, loss=copy.deepcopy(production['loss']))
    cfg.pop('completion')
    cfg['knowledge'].update(latent_reasoning_steps=2, train_reasoning_rounds=[2], inference_reasoning_round=2)
    for key in ('critic_lr','verifier_lr','verifier_warmup'): cfg['train'].pop(key, None)
    cfg['train'].update(text_dropout=0., native_text_only_dropout=0.)
    validate(cfg)
    knowledge = QwenKnowledge(cfg, 'cpu', backbone=old.knowledge.backbone, tokenizer=old.knowledge.tokenizer)
    wan = FakeWan(cfg); wan.blocks=nn.ModuleList([NativeBlock() for _ in range(3)])
    with patch('physgen_native.system.CriterionVerifier', side_effect=AssertionError('judge constructed')), \
         patch('physgen_native.system.ConditionalProcessVerifier', side_effect=AssertionError('critic constructed')), \
         patch('physgen_native.process_encoder.VideoProcessEncoder', side_effect=AssertionError('JEPA constructed')):
        model=PhysGenNative(cfg,wan,knowledge=knowledge,decoder=GeometryDecoder(),device='cpu')
    return model,sample


class NoVerifierTests(unittest.TestCase):
    def test_both_recipes_remove_feedback_and_preserve_training_budget(self):
        for mode in ('h_content_message','q_seed128_message'):
            stem=f'native_{mode}_no_verifier_simple'
            cfg=read_config(ROOT/'configs'/f'{stem}_wisa3857.yaml')
            alias=read_config(ROOT/'configs'/f'native_{mode}_verifier_frozen_simple_wisa3857.yaml')
            self.assertEqual(alias,cfg)
            self.assertEqual(cfg['stages'],dict(interface=64,joint=160))
            self.assertEqual(cfg['train']['accumulation'],4)
            self.assertEqual(cfg['train']['seed'],42)
            self.assertEqual(cfg['knowledge']['train_reasoning_rounds'],[2])
            self.assertEqual(cfg['knowledge']['inference_reasoning_round'],2)
            self.assertEqual(cfg['verifier'],{'kind':'none'})
            self.assertNotIn('completion',cfg)
            self.assertEqual(cfg['loss'],dict(interface=dict(fm=1.,flow_temporal=.1),joint=dict(fm=1.,flow_temporal=.1)))
            self.assertFalse(any(k.startswith('verifier_') for k in cfg['train']))
            self.assertNotIn('completion',smoke_config(cfg))
            job=yaml.safe_load((ROOT/'train'/f'{stem}_1x96g.yaml').read_text())
            self.assertEqual(job['resources']['resource_pool'],'blk-96g')
            self.assertEqual(job['resources']['slots_per_trial'],1)
            self.assertEqual(job['max_restarts'],0)
            self.assertIn('checkpoints/initialization/generator.pt',job['entrypoint'])
            self.assertNotIn('--resume',job['entrypoint'])
            self.assertEqual(cfg['hardware'],dict(gpu_family='rtx_pro_6000'))
            bad=copy.deepcopy(cfg);bad['loss']['joint']['verifier']=.05
            with self.assertRaises(ValueError):validate(bad)

    def test_real_fm_backprop_has_no_judge_calls_in_either_phase(self):
        for query in (False,True):
            model,sample=no_verifier_model(query)
            self.assertIsNone(model.critic);self.assertIsNone(model.process_encoder)
            self.assertFalse(hasattr(model.knowledge,'answer_ids'))
            self.assertFalse(hasattr(model,'verifier_state'))
            groups=parameter_groups(model,model.cfg)
            self.assertEqual({g['role'] for g in groups},{'condition','wan_lora'})
            optimizer=torch.optim.AdamW(groups);scheduler=schedule(optimizer,model.cfg)
            null=torch.zeros_like(sample['text'])
            with patch.object(model,'completion',side_effect=AssertionError('feedback sampled')), \
                 patch.object(model,'feedback_losses',side_effect=AssertionError('feedback invoked')), \
                 patch.object(model.knowledge,'criterion_logits',side_effect=AssertionError('judge scored')), \
                 patch.object(model.knowledge,'criterion_teacher',side_effect=AssertionError('teacher invoked')):
                for step in (0,model.cfg['stages']['interface']):
                    model.zero_grad(set_to_none=True);model.train();model.set_phase(step)
                    self.assertFalse(model.should_feedback(step))
                    losses,weights,metrics=model(sample,null,step,sigma_override=.65)
                    self.assertEqual(set(losses),{'fm','flow_temporal'})
                    self.assertEqual(model.last_condition_round,2)
                    self.assertFalse(any('verifier' in key for key in metrics))
                    self.assertTrue(all(torch.isfinite(value) for value in losses.values()))
                    model.backward_losses(losses,weights)
                    self.assertGreater(grad_size(model.conditioner),0.)
                    self.assertEqual(grad_size(model.knowledge),0.)
                    if step==0:self.assertEqual(grad_size(model.generator.wan),0.)
                    else:self.assertGreater(grad_size(model.generator.wan),0.)
                    optimizer.step();scheduler.step()
                model.eval()
                with torch.no_grad():
                    losses,_,_=model(sample,null,step,validation=True,sigma_override=.65)
                self.assertEqual(set(losses),{'fm','flow_temporal'})
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'latest.pt'
                save_checkpoint(model,optimizer,scheduler,model.cfg,2,8,path)
                state=load_checkpoint(path)
                self.assertNotIn('verifier_state',state)
                self.assertFalse(any(n.startswith('critic.') or 'verifier_lora' in n for n in state['model']))
                self.assertEqual(restore(model,optimizer,scheduler,state,model.cfg),(2,8))

    def test_saved_model_uses_two_refinement_rounds_for_training_and_sampling(self):
        for query in (False,True):
            model,sample=no_verifier_model(query)
            model.eval().requires_grad_(False)
            with torch.no_grad(), patch.object(model.conditioner.block,'forward',wraps=model.conditioner.block.forward) as block:
                condition=model.conditions(sample)
                model.encode_condition(condition,model.decoder(sample['first'],[0]))
                # Q has one initial grounding read R0 followed by two refinements.
                self.assertEqual(block.call_count,3 if query else 2)
                self.assertEqual(model.last_condition_round,2)
            with torch.no_grad(),patch.object(model.knowledge,'criterion_logits',side_effect=AssertionError('judge during sampling')):
                latent=model.sample(model.conditions(sample),sample['text'],torch.zeros_like(sample['text']),seed=42,steps=2)
                self.assertEqual(model.last_sample_reasoning_round,2)
                self.assertTrue(torch.isfinite(latent).all())

    def test_gpu_policy_accepts_only_requested_family(self):
        for name in ('NVIDIA RTX PRO 6000 Blackwell Server Edition','NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
                     'RTX 6000 Blackwell'):
            self.assertTrue(is_rtx_pro_6000(name),name)
        for name in ('NVIDIA H100 80GB HBM3','NVIDIA H100 NVL','NVIDIA H200','NVIDIA A100-SXM4-80GB',
                     'NVIDIA RTX A6000','NVIDIA RTX 6000 Ada Generation'):
            self.assertFalse(is_rtx_pro_6000(name),name)

    def test_h100_allocation_exits_before_data_or_model_loading(self):
        from train.train import train
        args=SimpleNamespace(config=str(ROOT/'configs/native_h_content_message_no_verifier_simple_wisa3857.yaml'),
                             stop_after=None,check_only=False)
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.get_device_name',return_value='NVIDIA H100 NVL'), \
             patch('train.train.audit',side_effect=AssertionError('data opened')), \
             patch('train.train.load_wan',side_effect=AssertionError('model loaded')):
            with self.assertRaisesRegex(RuntimeError,'GPU policy rejected NVIDIA H100'):
                train(args)


if __name__=='__main__':unittest.main()
