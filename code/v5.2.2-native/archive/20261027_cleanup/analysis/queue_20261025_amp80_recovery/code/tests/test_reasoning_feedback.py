"""Native Qwen recurrence, immutable judging, and single-trajectory C feedback.

These tests prove computational behavior and gradient ownership, not physics.
"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_operator import config, model_and_data, TextKnowledge, Observer
from test_native import FakeWan, FakeDecoder, sample, grad_size
from physgen_native.config import read_config, validate, REASONING_REVISION
from physgen_native.knowledge import QwenKnowledge
from physgen_native.latent_reasoning import refine_dependency_latent
from physgen_native.system import PhysGenNative
from physgen_native.feedback_sampling import feedback_indices, latent_update
from physgen_native.runtime import parameter_groups, schedule, save_checkpoint, load_checkpoint, restore

torch.set_num_threads(2)


def native_knowledge():
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration, AutoTokenizer
    cfg = config()
    cfg['revision'] = REASONING_REVISION
    cfg['completion']['feedback_to'] = 'latent'
    cfg['knowledge'].update(latent_reasoning_steps=2, latent_reasoning_layers=2,
                            reasoning_tokens=0, max_condition_tokens=16)
    native = Qwen3VLConfig(text_config=dict(vocab_size=151936, hidden_size=32, intermediate_size=64,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rope_scaling={'rope_type':'default','mrope_section':[1,1,2],'mrope_interleaved':True}),
        vision_config=dict(depth=3, hidden_size=32, intermediate_size=64, num_heads=4,
            out_hidden_size=32, num_position_embeddings=64, deepstack_visual_indexes=[0,1,2],
            patch_size=16, temporal_patch_size=2, spatial_merge_size=2))
    knowledge = QwenKnowledge(cfg, device='cpu', backbone=Qwen3VLForConditionalGeneration(native),
        tokenizer=AutoTokenizer.from_pretrained(cfg['paths']['qwen_checkpoint'], local_files_only=True)).train()
    for name, p in knowledge.named_parameters(): p.requires_grad_(name.endswith(('lora_a','lora_b')))
    return cfg, knowledge


class ReasoningTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(2026)

    def test_native_checkpointed_rounds_match_gradients_and_leave_judge_fixed(self):
        cfg, knowledge = native_knowledge()
        image = torch.rand(1,3,32,64)*2-1
        clauses = [dict(text='The initial surface contains a local disturbance.',basis='image'),
                   dict(text='The disturbance spreads across the surface.',basis='prompt')]
        states = []
        def record(*args, **kwargs):
            result, trace = refine_dependency_latent(*args, **kwargs, return_states=True)
            for state in trace: state.retain_grad()
            states.extend(trace)
            return result
        fixed_before = knowledge.encode(image,'A surface disturbance spreads.',clauses,adapt=False)['contextual'].detach()
        with patch('physgen_native.knowledge.refine_dependency_latent', side_effect=record):
            result = knowledge.encode(image,'A surface disturbance spreads.',clauses)
        probe = torch.randn_like(result['contextual'])
        (result['contextual']*probe).mean().backward()
        self.assertEqual(result['latent_reasoning_steps'], 2)
        self.assertEqual(len(states), 3)
        for state in states: self.assertGreater(float(state.grad.norm()), 0)
        expected = {n:p.grad.clone() for n,p in knowledge.named_parameters() if p.grad is not None}
        self.assertGreater(sum(float(x.norm()) for x in expected.values()),0)
        # Disable recomputation only; all actual native weights and inputs match.
        knowledge.zero_grad(set_to_none=True)
        knowledge.backbone.gradient_checkpointing_disable()
        plain = knowledge.encode(image,'A surface disturbance spreads.',clauses)
        torch.testing.assert_close(plain['contextual'],result['contextual'],atol=1e-6,rtol=1e-5)
        (plain['contextual']*probe).mean().backward()
        for name,p in knowledge.named_parameters():
            if name in expected: torch.testing.assert_close(p.grad,expected[name],atol=1e-6,rtol=1e-4)
        # The judge condition does not change when learned reasoning changes.
        with torch.no_grad():
            for name,p in knowledge.named_parameters():
                if name.endswith('lora_b'): p.add_(.1*torch.randn_like(p))
        fixed_after = knowledge.encode(image,'A surface disturbance spreads.',clauses,adapt=False)
        self.assertEqual(fixed_after['latent_reasoning_steps'],0)
        torch.testing.assert_close(fixed_before,fixed_after['contextual'],atol=0,rtol=0)

    def test_frozen_verifier_reaches_every_native_reasoning_round(self):
        cfg, knowledge = native_knowledge()
        model = PhysGenNative(cfg,FakeWan(cfg),knowledge=knowledge,decoder=FakeDecoder(),
                             process_encoder=Observer(),device='cpu')
        data = sample()
        traces = []
        def record(*args, **kwargs):
            result, states = refine_dependency_latent(*args, **kwargs, return_states=True)
            for state in states: state.retain_grad()
            traces.append(states)
            return result
        with patch('physgen_native.knowledge.refine_dependency_latent',side_effect=record):
            losses, weights, _ = model.feedback_losses(data,2)
        model.backward_losses(losses,weights)
        self.assertEqual(len(traces),1)  # immutable judge uses no adapted recurrence
        for state in traces[0]: self.assertGreater(float(state.grad.norm()),0)
        self.assertGreater(grad_size(knowledge),0)
        self.assertEqual(grad_size(model.critic),0)
        self.assertEqual(grad_size(model.generator),0)
        self.assertGreater(grad_size(model.conditioner),0)

    def test_latent_only_feedback_preserves_existing_reader_gradients(self):
        model,data=model_and_data()
        model.cfg['revision']=REASONING_REVISION
        model.cfg['completion']['feedback_to']='latent'
        losses,weights,_=model(data,torch.zeros_like(data['text']),2,sigma_override=.65)
        model.backward_losses(losses,weights)
        before={n:p.grad.detach().clone() for n,p in model.generator.named_parameters() if p.grad is not None}
        self.assertTrue(before)
        losses,weights,metrics=model.feedback_losses(data,2)
        model.backward_losses(losses,weights)
        self.assertFalse(metrics['feedback_updates_reader'])
        self.assertGreater(float(metrics['C_gradient_rms']),0)
        for name,p in model.generator.named_parameters():
            if name in before: torch.testing.assert_close(p.grad,before[name],atol=0,rtol=0)
            else:self.assertIsNone(p.grad)

    def test_final_config_is_three_stages_and_has_no_extra_reasoning_loss(self):
        cfg = read_config(ROOT/'configs/native_reasoning_operator_wisa3857.yaml')
        self.assertEqual(cfg['revision'],REASONING_REVISION)
        self.assertEqual([len(x) for x in cfg['loss'].values()],[4,2,4])
        self.assertEqual(cfg['knowledge']['latent_reasoning_steps'],2)
        bad = copy.deepcopy(cfg);bad['revision']='v522_native_conditional_operator_v6'
        with self.assertRaises(ValueError): validate(bad)


class FeedbackSamplingTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(31)

    def test_relative_rms_descent_is_invariant_to_equivalent_latent_replication(self):
        # Repeating a latent leaves this mean-based physical-score surrogate
        # unchanged. It must not shrink the correction by the repeat count.
        initial = torch.tensor([[[1., -2., .5, 3.]]])
        target = initial*.25
        expected = initial+.1*(target-initial)
        for repeats in (1, 128, 3072):
            c = initial.repeat(1, repeats, 1).requires_grad_()
            goal = target.repeat(1, repeats, 1)
            scale = initial.square().mean()
            loss = .5*(c-goal).square().mean()/scale
            gradient, = torch.autograd.grad(loss, c)
            updated = c.detach()+latent_update(gradient, c, .1, 'relative_rms')
            torch.testing.assert_close(updated, expected.repeat(1, repeats, 1))
            self.assertLess(float(.5*(updated-goal).square().mean()/scale), float(loss))
        zero = latent_update(torch.zeros_like(initial), initial, .1, 'relative_rms')
        torch.testing.assert_close(zero, torch.zeros_like(initial), atol=0, rtol=0)

    def test_legacy_euclidean_feedback_policy_is_preserved(self):
        c = torch.randn(1, 7, 11); gradient = torch.randn_like(c)
        expected = -.1*c.square().mean()*gradient
        torch.testing.assert_close(latent_update(gradient, c, .1, 'euclidean'), expected,
                                   atol=0, rtol=0)

    def test_one_trajectory_updates_only_C_and_never_repeats_a_Wan_step(self):
        model,data = model_and_data();model.eval().requires_grad_(False)
        model.cfg['inference'].update(steps=6,shift=1.,latent_feedback=dict(enabled=True,metric='relative_rms',step_size=.1,sigmas=[.8,.5]))
        before = {n:p.detach().clone() for n,p in model.named_parameters()}
        calls = []
        def record(module,args): calls.append((float(args[3]),args[4].detach().clone(),args[0].detach().clone()))
        handle = model.generator.register_forward_pre_hook(record)
        result = model.sample(model.conditions(data),data['text'],torch.zeros_like(data['text']),seed=7)
        handle.remove()
        self.assertEqual(len(calls),12)  # six native steps, one conditional and one CFG call each
        self.assertEqual(model.last_sample_feedback['complete_trajectories'],1)
        self.assertFalse(model.last_sample_feedback['future_reference_used'])
        self.assertEqual([r['step'] for r in model.last_sample_feedback['updates']],[2,3])
        self.assertTrue(all(r['gradient_rms']>0 for r in model.last_sample_feedback['updates']))
        self.assertTrue(all(r['relative_C_update_rms']>0 for r in model.last_sample_feedback['updates']))
        for step in range(6):
            self.assertEqual(calls[2*step][0],calls[2*step+1][0])
            torch.testing.assert_close(calls[2*step][2],calls[2*step+1][2],atol=0,rtol=0)
        self.assertTrue(torch.equal(calls[0][1],calls[4][1]))
        self.assertFalse(torch.equal(calls[4][1],calls[6][1]))
        for name,p in model.named_parameters():
            torch.testing.assert_close(p,before[name],atol=0,rtol=0)
            self.assertIsNone(p.grad)
        torch.testing.assert_close(result[:,:,:1],data['first'],atol=0,rtol=0)

    def test_online_feedback_off_reproduces_native_sampling_exactly(self):
        model,data=model_and_data();model.eval().requires_grad_(False)
        condition=model.conditions(data);null=torch.zeros_like(data['text'])
        expected=model.sample(condition,data['text'],null,seed=9)
        model.cfg['inference']['latent_feedback']=dict(enabled=False,step_size=100,sigmas=[.85])
        actual=model.sample(condition,data['text'],null,seed=9)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        self.assertIsNone(model.last_sample_feedback)

    def test_no_refinement_is_scheduled_after_its_last_possible_effect(self):
        with self.assertRaises(ValueError): feedback_indices(torch.tensor([1.,.8,.3,0.]),[.3])

    def test_lean_inference_loads_same_generator_without_training_observer(self):
        model,data=model_and_data();cfg=model.cfg
        opt=torch.optim.AdamW(parameter_groups(model,cfg));sched=schedule(opt,cfg)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.pt';save_checkpoint(model,opt,sched,cfg,2,17,path)
            state=load_checkpoint(path)
            torch.manual_seed(31)
            lean=PhysGenNative(cfg,FakeWan(cfg),knowledge=TextKnowledge(),decoder=FakeDecoder(),
                               device='cpu',training_observers=False)
            restore(lean,None,None,state,cfg)
        self.assertIsNone(lean.process_encoder)
        self.assertFalse(any(p.requires_grad for p in lean.parameters()))
        condition=model.conditions(data);null=torch.zeros_like(data['text'])
        expected=model.sample(condition,data['text'],null,seed=19)
        actual=lean.sample(condition,data['text'],null,seed=19)
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)


if __name__ == '__main__': unittest.main()
