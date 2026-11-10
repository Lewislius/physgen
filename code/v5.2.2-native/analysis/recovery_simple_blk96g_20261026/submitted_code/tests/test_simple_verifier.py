"""Short-budget recipes: native video, full-width messages and real gradients."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_native_geometry import geometry_model, GeometryDecoder
from test_criterion_verifier import teacher_labels
from test_native import FakeWan, grad_size
from test_fullwidth_attention import NativeBlock
from physgen_native.config import SIMPLE_MESSAGE_VARIANTS, read_config, validate
from physgen_native.knowledge import QwenKnowledge
from physgen_native.system import PhysGenNative
from physgen_native.message_conditioning import MessageCondition
from physgen_native.criterion_verifier import CriterionVerifier
from physgen_native.lora import role_checkpoint
from physgen_native.runtime import (parameter_groups, schedule, save_checkpoint, load_checkpoint,
    restore, initialize_generator, clip_owned_gradients, feedback_gradient_metrics, owner)

torch.set_num_threads(2)


def simple_model(query=False, lora=False):
    from transformers import Qwen3VLForConditionalGeneration
    old, data = geometry_model(query, lora)
    cfg = copy.deepcopy(old.cfg)
    mode = cfg['knowledge']['reasoning_mode']
    production = read_config(ROOT/'configs'/f'native_{mode}_verifier_{"lora" if lora else "frozen"}_simple_wisa3857.yaml')
    cfg['revision'] = production['revision']
    cfg['train'] = copy.deepcopy(production['train'])
    cfg['loss'] = copy.deepcopy(production['loss'])
    cfg['sequence'] = dict(size=[32,64], scales=[1])
    cfg['completion'].update(guidance=5., branch_checkpointing=True, steps=2)
    cfg['inference'].update(guidance=5., shift=cfg['completion']['shift'])
    cfg['knowledge'].update(video_observation_mode='video', train_reasoning_rounds=[2 if query else 1],
                            inference_reasoning_round=2 if query else 1)
    cfg['message'].update(projection='output', workspace_width=cfg['model']['qwen_width'])
    cfg['verifier'].update(protocol='visible_process_v2', input_mode='video', frames=8, frame_stride=3)
    validate(cfg)
    knowledge = QwenKnowledge(cfg, 'cpu', backbone=Qwen3VLForConditionalGeneration(old.knowledge.backbone.config),
                              tokenizer=old.knowledge.tokenizer)
    wan = FakeWan(cfg); wan.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
    return PhysGenNative(cfg, wan, knowledge=knowledge, decoder=GeometryDecoder(), device='cpu'), data


class SimpleVerifierTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(261101)

    def test_eval_block_checkpoint_matches_rgb_and_adapter_gradients(self):
        for lora in (False,True):
            model,data=simple_model(lora=lora)
            model.train(); model.set_phase(1)
            k=model.knowledge
            frames=torch.rand(3,3,32,64,requires_grad=True)
            inputs,_=k._inputs([(frames,torch.arange(3.))],'Judge the visible process.',video=True)
            params=[p for n,p in k.named_parameters() if p.requires_grad]
            with torch.no_grad():
                for n,p in k.named_parameters():
                    if n.endswith('verifier_lora_b'): p.normal_(std=.02)
            direct=k._answer_logits(inputs,verifier=lora,checkpointing=False)
            expected=torch.autograd.grad(direct.square().sum(),[frames]+params,retain_graph=True)
            with patch('physgen_native.lora.role_checkpoint',wraps=role_checkpoint) as calls:
                actual=k._answer_logits(inputs,verifier=lora,checkpointing=True)
                gradients=torch.autograd.grad(actual.square().sum(),[frames]+params)
            self.assertGreaterEqual(calls.call_count,len(k.backbone.model.visual.blocks)+len(k.backbone.model.language_model.layers))
            torch.testing.assert_close(actual,direct)
            for got,want in zip(gradients,expected): torch.testing.assert_close(got,want)
            self.assertGreater(float(gradients[0].norm()),0.)
            self.assertTrue(all(not module.training for module in k.backbone.modules()))

    def test_compact_analysis_keeps_budget_limited_text_and_cache_status(self):
        model,data = simple_model()
        k=model.knowledge
        k.cfg.update(reasoning_tokens=4,reasoning_max_tokens=8,analysis_format='compact_four_lines',
                     analysis_on_token_limit='use_partial')
        image=model.decoder(data['first'],[0]); captured=[]
        original=k._inputs
        def inputs(clips,body,*args,**kwargs):
            captured.append(body)
            return original(clips,body,*args,**kwargs)
        with patch.object(k,'_inputs',side_effect=inputs), patch.object(k,'_reason',side_effect=[('Initial evidence',False),('Observed source continues and spreads',False)]) as reason:
            actual=k._multiround_analysis(image,data['caption'],[],adapt=False)
            cached=k._multiround_analysis(image,data['caption'],[],adapt=False)
        self.assertEqual(actual,'Observed source continues and spreads')
        self.assertEqual(cached,actual)
        self.assertEqual(reason.call_count,2)
        self.assertTrue(k.last_analysis_truncated)
        self.assertIn('at most 100 words',captured[0])

    def test_budget_limited_teacher_observation_can_calibrate_without_fake_eos(self):
        model,data=simple_model(lora=True)
        model.knowledge.verifier_cfg.update(observation_format='compact_three_lines',observation_on_token_limit='use_partial')
        frames,times=model.decode_qwen_view(data['latent'],data)
        with patch.object(model.knowledge,'_reason',return_value=('EARLY: A source is visible. MIDDLE: The cloud spreads.',False)), \
             patch.object(model.knowledge,'criterion_logits',return_value=teacher_labels()[0]):
            teacher,trace=model.knowledge.criterion_teacher(frames,times,data['caption'],[])
        self.assertFalse(trace['observation_complete'])
        self.assertTrue(trace['observation_usable'])
        student=torch.zeros_like(teacher,requires_grad=True)
        loss,metrics=model.critic.calibration_loss(student,teacher,observation_complete=False,observation_usable=True)
        loss.backward()
        self.assertEqual(int(metrics['teacher_accepted']),4)
        self.assertGreater(float(student.grad.norm()),0.)

    def test_production_configs_and_dense_frame_coverage(self):
        model, _ = simple_model()
        for revision, (mode, lora) in SIMPLE_MESSAGE_VARIANTS.items():
            stem = f'native_{mode}_verifier_{"lora" if lora else "frozen"}_simple'
            cfg = read_config(ROOT/'configs'/f'{stem}_wisa3857.yaml')
            self.assertEqual(cfg['revision'], revision)
            self.assertEqual(sum(cfg['stages'].values()), 256 if lora else 224)
            self.assertEqual(cfg['train']['accumulation'], 4)
            self.assertNotIn('process', cfg['loss']['joint'])
            self.assertEqual(cfg['message']['workspace_width'], 4096)
            if mode.startswith('q_'): self.assertEqual(cfg['knowledge']['relation_tokens'], 128)
            job = yaml.safe_load((ROOT/'train'/f'{stem}_1x96g.yaml').read_text())
            self.assertEqual(job['resources']['resource_pool'], 'blk-96g')
            self.assertEqual(job['resources']['slots_per_trial'], 1)
            self.assertIn(f'{stem}_wisa3857.yaml', job['entrypoint'])
        ids = model.qwen_frame_indices(81)
        self.assertEqual(ids, list(range(0,81,3))+[80])
        self.assertEqual(len(ids), 28)
        self.assertLessEqual(max(b-a for a,b in zip(ids,ids[1:])), 3)

    def test_projection_is_after_full_width_updates(self):
        model, _ = simple_model()
        for query in (False, True):
            cfg = copy.deepcopy(model.cfg)
            cfg['model']['qwen_width'] = 64
            cfg['message']['workspace_width'] = 64
            cfg['knowledge']['reasoning_mode'] = 'q_seed128_message' if query else 'h_content_message'
            module = MessageCondition(cfg)
            evidence = dict(memory=torch.randn(1,5,64),memory_mask=torch.ones(1,5,dtype=torch.bool),
                            content_positions=torch.tensor([1,2,3]))
            seen = []
            hook = module.block.register_forward_pre_hook(lambda m,a: seen.append((a[0].shape[-1],a[1].shape[-1])))
            result = module(evidence, reasoning_round=1, all_rounds=True)
            hook.remove()
            self.assertTrue(all(pair == (64,64) for pair in seen))
            self.assertEqual(result['readouts'][-1].shape[-1], 64)
            self.assertEqual(result['conditions'][-1].shape[-1], 32)
            (result['conditions'][-1]*torch.randn_like(result['conditions'][-1])).sum().backward()
            self.assertGreater(float(module.block.cross.v.weight.grad[:,32:].norm()), 0.)
            if query: self.assertEqual(module.query_seed.shape, (1,128,64))

    def test_nll_keeps_gradient_for_confident_wrong_predictions(self):
        cfg = dict(temperature=1., protocol='visible_process_v2')
        critic = CriterionVerifier(cfg)
        logits = torch.tensor([-20.,20.,-20.]).view(1,1,1,3).expand(1,4,2,3).clone().requires_grad_()
        loss = critic.generation_loss(logits)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(logits.grad.norm()), .4)
        self.assertLess(float(logits.grad[...,0].sum()), -.99)
        legacy = CriterionVerifier(dict(temperature=1.))
        self.assertAlmostEqual(float(legacy.generation_loss(logits.detach())), 2.)

    def test_four_training_paths_without_rgb_and_with_owned_feedback(self):
        for query in (False, True):
            for lora in (False, True):
                with self.subTest(query=query, lora=lora):
                    model, data = simple_model(query,lora); model.train()
                    null = torch.zeros_like(data['text'])
                    optimizer = torch.optim.AdamW(parameter_groups(model,model.cfg))
                    with patch.object(model.decoder, 'forward', wraps=model.decoder.forward) as decoder:
                        losses,weights,_ = model(data,null,0,sigma_override=.2)
                        model.backward_losses(losses,weights)
                    self.assertEqual(set(losses), {'fm','flow_temporal'})
                    self.assertTrue(all(call.args[0].shape[2] == 1 for call in decoder.call_args_list))
                    self.assertGreater(grad_size(model.conditioner),0.)
                    self.assertEqual(grad_size(model.generator.wan),0.)
                    optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    if lora:
                        model.set_phase(1)
                        with patch.object(model.knowledge,'criterion_teacher',side_effect=teacher_labels):
                            losses,weights,_ = model.qwen_calibration(data,1,1,null_text=null)
                        model.backward_losses(losses,weights)
                        self.assertGreater(grad_size(model.knowledge),0.)
                        self.assertEqual(grad_size(model.conditioner),0.)
                        optimizer.step(); optimizer.zero_grad(set_to_none=True)
                    step = 2 if lora else 1
                    losses,weights,_ = model(data,null,step,sigma_override=.2)
                    model.backward_losses(losses,weights)
                    self.assertGreater(grad_size(model.generator.wan),0.)
                    optimizer.zero_grad(set_to_none=True)
                    losses,weights,metrics = model.feedback_losses(data,step,null_text=null)
                    with feedback_gradient_metrics(optimizer,model.cfg) as gradients:
                        model.backward_losses(losses,weights)
                    self.assertGreater(float(gradients['feedback_C_grad_norm']),0.)
                    self.assertGreater(grad_size(model.conditioner.block),0.)
                    self.assertEqual(grad_size(model.generator.wan),0.)
                    self.assertEqual(grad_size(model.knowledge),0.)
                    self.assertEqual(metrics['completion_guidance'],5.)
                    if query: self.assertGreater(float(model.conditioner.query_seed.grad.norm()),0.)
                    self.assertEqual(model.last_condition_round,2 if query else 1)

    def test_cfg_adjoint_matches_plain_gradients_and_completion(self):
        model,data = simple_model(); model.set_phase(1); model.eval()
        null = torch.zeros_like(data['text']); conditions = model.conditions(data)
        first = model.decoder(data['first'],[0])
        c = model.encode_condition(conditions,first).detach().requires_grad_()
        u = model.encode_condition(dict(conditions,caption='',clauses=[]),first).detach().requires_grad_()
        noise = torch.randn_like(data['latent'])
        policy = dict(model.cfg['inference'],steps=2,gradient_steps=2,discrete_adjoint=True,branch_checkpointing=True)
        actual = model._sample_tokens(conditions,data['text'],null,0,policy,differentiable=True,
            initial_noise=noise,token_override=c,negative_token_override=u,generator_gradients=False)
        gradients = torch.autograd.grad(actual.square().mean(),(c,u))
        plain = model._sample_tokens(conditions,data['text'],null,0,dict(policy,discrete_adjoint=False,branch_checkpointing=False),
            differentiable=True,initial_noise=noise,token_override=c,negative_token_override=u)
        reference = torch.autograd.grad(plain.square().mean(),(c,u))
        torch.testing.assert_close(actual,plain)
        for a,b in zip(gradients,reference):
            torch.testing.assert_close(a,b,rtol=3e-4,atol=3e-5)
            self.assertGreater(float(a.norm()),0.)
        result = model.completion(data,c,data['latent'].new_tensor(1.),noise,differentiable=False,null_text=null)
        torch.testing.assert_close(result,plain.detach())
        altered = dict(data,latent=torch.randn_like(data['latent']))
        other = model.completion(altered,c,data['latent'].new_tensor(1.),noise,differentiable=False,null_text=null)
        torch.testing.assert_close(result,other)
        self.assertFalse(result.requires_grad)

    def test_schedule_resume_and_separate_clipping(self):
        model,data = simple_model(lora=True)
        model.cfg['stages'] = dict(interface=64,calibration=32,joint=160)
        groups = parameter_groups(model,model.cfg)
        optimizer = torch.optim.AdamW(groups); scheduler = schedule(optimizer,model.cfg)
        expected = {0:(1e-4/8,0.),7:(1e-4,0.),63:(5e-5,0.),64:(0.,0.),
                    96:(5e-5,1e-5/8),103:(None,1e-5),255:(1e-5,2e-6)}
        for step in range(256):
            rates = {g['name']:g['lr'] for g in optimizer.param_groups}
            if step in expected:
                a,b = expected[step]
                if a is not None: self.assertAlmostEqual(rates['condition'],a,places=10)
                self.assertAlmostEqual(rates['wan_lora'],b,places=10)
            if step == 95:
                with tempfile.TemporaryDirectory() as directory:
                    path=Path(directory)/'resume.pt'
                    save_checkpoint(model,optimizer,scheduler,model.cfg,step,step*4,path)
                    restored=torch.optim.AdamW(parameter_groups(model,model.cfg))
                    curve=schedule(restored,model.cfg)
                    restore(model,restored,curve,load_checkpoint(path),model.cfg)
                    self.assertEqual(scheduler.get_last_lr(),curve.get_last_lr())
                    restored.step();curve.step()
                    self.assertAlmostEqual(restored.param_groups[0]['lr'],5e-6,places=10)
            optimizer.step();scheduler.step()
        c=next(p for n,p in model.named_parameters() if n.startswith('conditioner.project.'))
        w=next(p for n,p in model.named_parameters() if owner(n)=='wan_lora')
        c.grad=torch.ones_like(c)*1e4;w.grad=torch.ones_like(w)*1e-4
        before=w.grad.clone()
        clip_owned_gradients(optimizer,model.cfg)
        torch.testing.assert_close(w.grad,before)

    def test_legacy_initialization_keeps_projection_and_reinitializes_workspace(self):
        old,_ = geometry_model(query=True)
        model,_ = simple_model(query=True)
        original_block = model.conditioner.block.cross.q.weight.detach().clone()
        groups = parameter_groups(old,old.cfg)
        optimizer = torch.optim.AdamW(groups); curve = schedule(optimizer,old.cfg)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'legacy.pt'
            save_checkpoint(old,optimizer,curve,old.cfg,0,0,path)
            result = initialize_generator(model,model.cfg,path)
        self.assertTrue(result['message_migration'])
        self.assertIn('conditioner.query_seed',result['new_tensors'])
        torch.testing.assert_close(model.conditioner.project.weight,old.conditioner.project.weight)
        torch.testing.assert_close(model.conditioner.block.cross.q.weight,original_block)


if __name__ == '__main__': unittest.main()
