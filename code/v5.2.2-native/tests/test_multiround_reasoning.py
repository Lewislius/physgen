"""v14 native Qwen recurrence, full-width Wan compatibility and ownership.

Uses the installed Qwen3-VL implementation with tiny random weights on CPU.
These are architectural/gradient checks, not a physical-quality benchmark.
"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_reasoning_feedback import native_knowledge
from test_aligned_attention import aligned_model
from test_native import FakeWan, FakeDecoder, sample
from test_fullwidth_attention import NativeBlock
from physgen_native.config import read_config, validate, MULTIROUND_REVISION, ALIGNED_REVISION, select_reasoning_exit
from physgen_native.knowledge import QwenKnowledge
from physgen_native.lora import knowledge_role, _knowledge
from physgen_native.recurrent_reasoning import capture_prefix_projections, PrefixMemory, read_prefix, pack_conditions
from physgen_native.multiround_knowledge import content_sections
from physgen_native.native_attention import NativeAttentionContext
from physgen_native.runtime import owner, parameter_groups, schedule, save_checkpoint, load_checkpoint, restore, initialize_generator
from physgen_native.system import PhysGenNative

torch.set_num_threads(2)


def multiround_model():
    from transformers import Qwen3VLForConditionalGeneration
    _, seed = native_knowledge()
    previous, _ = aligned_model()
    cfg = copy.deepcopy(previous.cfg)
    cfg['revision'] = MULTIROUND_REVISION
    cfg['knowledge'] = dict(seed.cfg)
    cfg['knowledge'].pop('max_condition_tokens', None)
    cfg['knowledge'].pop('latent_reasoning_layers', None)
    cfg['knowledge'].update(reasoning_mode='recurrent_prefix', relation_tokens=4, reasoning_tokens=0,
        reasoning_max_tokens=64, max_text_tokens=4096, latent_reasoning_steps=2,
        train_reasoning_rounds=[0,1,2], inference_reasoning_round=2, condition_readout='content',
        include_prompt_span=True, latent_reasoning_attention='causal', video_observation_mode='images',
        video_observation_first=True, video_comparison='symmetric_observations')
    validate(cfg)
    knowledge = QwenKnowledge(cfg, device='cpu', backbone=Qwen3VLForConditionalGeneration(seed.backbone.config),
                              tokenizer=seed.tokenizer)
    wan = FakeWan(cfg); wan.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
    model = PhysGenNative(cfg, wan, knowledge=knowledge, decoder=FakeDecoder(), device='cpu')
    return model, sample()


class MultiroundTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(261026)

    def test_active_config_no_pooling_or_rank_operator(self):
        cfg = read_config(ROOT/'configs/native_multiround_reasoning_wisa3857.yaml')
        self.assertEqual(cfg['knowledge']['reasoning_tokens'], 512)
        self.assertEqual(cfg['knowledge']['train_reasoning_rounds'], [0,1,2])
        self.assertEqual(cfg['model']['attention_width'], cfg['model']['width'])
        self.assertNotIn('max_condition_tokens', cfg['knowledge'])
        bad = copy.deepcopy(cfg); bad['knowledge']['max_condition_tokens'] = 128
        with self.assertRaises(ValueError): validate(bad)

    def test_content_is_unpooled_all_rounds_are_available_and_gradients_connected(self):
        model, data = multiround_model(); model.train()
        data['caption'] = 'The source keeps releasing smoke. '*95+'The source does not disappear.'
        result = model.encode_condition_rounds(model.conditions(data), torch.rand(1,3,32,64))
        self.assertEqual(len(result['conditions']), 3)
        self.assertGreater(result['content'].shape[1], 128)
        self.assertTrue(all(c.shape[1] == result['content'].shape[1]+4 for c in result['conditions']))
        self.assertFalse(hasattr(model.conditioner, 'kind'))
        for state in result['relation_states']: state.retain_grad()
        loss = (result['conditions'][2]*torch.randn_like(result['conditions'][2])).sum()
        loss.backward()
        for state in result['relation_states']: self.assertGreater(float(state.grad.norm()), 0)
        self.assertGreater(float(model.knowledge.reasoner.feedback.weight.grad.norm()), 0)
        self.assertGreater(float(model.knowledge.reasoner.query_seed.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for n,p in model.knowledge.backbone.named_parameters() if not n.endswith(('lora_a','lora_b'))))

    def test_cached_multimodal_readout_matches_complete_native_forward(self):
        model, _ = multiround_model(); knowledge = model.knowledge.eval()
        language = knowledge.backbone.model.language_model
        image = torch.rand(1,3,32,64)
        inputs, _ = knowledge._inputs([(image, torch.zeros(1))], 'Smoke remains after the source stops.')
        captured_inputs = {}
        def record(module, args, kwargs): captured_inputs.update(kwargs)
        hook = language.register_forward_pre_hook(record, with_kwargs=True)
        with knowledge_role(True), capture_prefix_projections(language) as captured:
            encoded = knowledge.backbone.model(**inputs, use_cache=False, return_dict=True)
        hook.remove()
        memory = PrefixMemory.from_projections(language, captured, encoded.last_hidden_state,
                                               inputs['position_ids'], inputs['attention_mask'])
        before = [(k.clone(),v.clone()) for k,v in memory.kv]
        state = None
        with knowledge_role(True):
            for _ in range(3):
                working = knowledge.reasoner.input_from(state, 1, encoded.last_hidden_state.dtype)
                actual = read_prefix(language, working, memory)
                full = dict(captured_inputs)
                full.pop('input_ids', None); full.pop('cache_position', None); full.pop('past_key_values', None)
                full['inputs_embeds'] = torch.cat((captured_inputs['inputs_embeds'], working), 1)
                full['attention_mask'] = torch.cat((inputs['attention_mask'], torch.ones(1,4,dtype=torch.long)), -1)
                p = inputs['position_ids']
                extension = p[-3:].amax().item()+1+torch.arange(4)
                extension = extension[None,None].expand(p.shape[0],1,-1).clone()
                if p.shape[0] == 4: extension[0] = torch.arange(p.shape[-1],p.shape[-1]+4)
                full['position_ids'] = torch.cat((p,extension), -1)
                if full.get('visual_pos_masks') is not None:
                    full['visual_pos_masks'] = torch.cat((full['visual_pos_masks'],torch.zeros(1,4,dtype=torch.bool)), -1)
                full['use_cache'] = False
                expected = language(**full).last_hidden_state[:, -4:]
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                state = actual
        for old, new in zip(before,memory.kv):
            for a,b in zip(old,new): torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_checkpoint_recomputation_matches_native_gradients(self):
        model, _ = multiround_model(); k = model.knowledge.train()
        image = torch.rand(1,3,32,64)
        result = k.encode(image, 'A cloud spreads; the source is unknown.', [], reasoning_round=2)
        probe = torch.randn_like(result['contextual'])
        (result['contextual']*probe).sum().backward()
        expected = {n:p.grad.clone() for n,p in k.named_parameters() if p.grad is not None}
        k.zero_grad(set_to_none=True); k.cfg['gradient_checkpointing'] = False
        k.backbone.gradient_checkpointing_disable()
        plain = k.encode(image, 'A cloud spreads; the source is unknown.', [], reasoning_round=2)
        torch.testing.assert_close(plain['contextual'], result['contextual'], rtol=2e-5, atol=2e-6)
        (plain['contextual']*probe).sum().backward()
        for n,p in k.named_parameters():
            if n in expected: torch.testing.assert_close(p.grad,expected[n],rtol=2e-4,atol=2e-5)

    def test_fullwidth_reader_masks_padded_C_and_preserves_gradients(self):
        model, data = multiround_model(); model.eval()
        conditions = model.conditions(data)
        exits = model.encode_condition_rounds(conditions, torch.rand(1,3,32,64))
        c = exits['conditions'][2]
        attention = model.generator.wan.blocks[0].cross_attn
        x, text = torch.randn(1,12,32), torch.randn(1,5,32)
        packet = NativeAttentionContext(text,c,model.generator.interface,(3,2,2),1.,0)
        expected = attention(x,packet,None)
        padding = torch.randn(1,7,32,requires_grad=True)
        padded = torch.cat((c,padding),1)
        mask = torch.arange(padded.shape[1])[None] < c.shape[1]
        packet.condition = padded; packet.condition_mask = mask
        actual = attention(x,packet,None)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=2e-6)
        actual.square().sum().backward()
        self.assertEqual(float(padding.grad.abs().sum()),0)
        self.assertGreater(float(model.knowledge.reasoner.query_seed.grad.norm()),0)
        batch, valid = pack_conditions([c, c[:,:3]])
        self.assertEqual(batch.shape,(2,c.shape[1],32))
        self.assertEqual(valid[1].sum().item(),3)

    def test_analysis_adapter_cache_budget_and_fixed_teacher(self):
        model, _ = multiround_model(); k = model.knowledge
        k.cfg.update(reasoning_tokens=4, reasoning_max_tokens=8)
        image = torch.rand(1,3,32,64)
        calls=[]
        def generate(**kwargs):
            calls.append((_knowledge.get(),torch.is_grad_enabled(),kwargs['max_new_tokens']))
            words = k.tokenizer.encode('A source sustains smoke.',add_special_tokens=False)
            words = words[:kwargs['max_new_tokens']]
            if kwargs['max_new_tokens']==8: words.append(k.tokenizer.eos_token_id)
            return torch.cat((kwargs['input_ids'],torch.tensor([words])),1)
        with patch.object(k.backbone,'generate',side_effect=generate):
            k._multiround_analysis(image,'Smoke',[],adapt=True)
            self.assertEqual(calls,[(True,False,4),(True,False,8)])
            k._multiround_analysis(image,'Smoke',[],adapt=True); self.assertEqual(len(calls),2)
            with torch.no_grad():
                next(p for n,p in k.backbone.named_parameters() if n.endswith('lora_b')).add_(.01)
            k._multiround_analysis(image,'Smoke',[],adapt=True); self.assertEqual(len(calls),4)
            k._multiround_analysis(image,'Smoke',[],adapt=False)
            self.assertEqual(calls[-2:],[(False,False,4),(False,False,8)])
            k.cfg['reasoning_max_tokens']=4
            with self.assertRaisesRegex(ValueError,'without EOS'): k._multiround_analysis(image,'Smoke',[],adapt=True)

    def test_supervised_analysis_and_relation_targets_have_real_gradients(self):
        model, _ = multiround_model(); k=model.knowledge
        image=torch.rand(1,3,32,64)
        loss=k.analysis_sft_loss(image,'Smoke continues.',[], 'A continuing source supplies more smoke.')
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for n,p in k.named_parameters() if n.endswith('lora_b')))
        k.zero_grad(set_to_none=True)
        evidence=k.encode(image,'Smoke continues.',[],reasoning_round=2)
        for state in evidence['relation_states']:state.retain_grad()
        loss=k.relation_sft_loss(evidence['relation_states'][2], 'What supplies new smoke?', 'The continuing source.')
        loss.backward()
        for state in evidence['relation_states']:self.assertGreater(float(state.grad.norm()),0)

    def test_ownership_phase_and_checkpoint_roundtrip(self):
        model,_=multiround_model(); cfg=model.cfg
        groups=parameter_groups(model,cfg); optimizer=torch.optim.AdamW(groups); scheduler=schedule(optimizer,cfg)
        self.assertTrue(all(owner(n)=='reasoning' for n,p in model.named_parameters() if n.startswith('knowledge.reasoner.')))
        self.assertTrue(all(p.requires_grad for p in model.knowledge.reasoner.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.generator.wan.parameters()))
        model.set_phase(cfg['stages']['interface'])
        self.assertFalse(any(p.requires_grad for p in model.knowledge.reasoner.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            save_checkpoint(model,optimizer,scheduler,cfg,3,5,path)
            state=load_checkpoint(path)
            self.assertIn('knowledge.reasoner.query_seed',state['model'])
            other,_=multiround_model()
            self.assertEqual(restore(other,None,None,state,cfg),(3,5))
            for name,p in other.named_parameters():
                if owner(name) is not None:torch.testing.assert_close(p,state['model'][name],rtol=0,atol=0)

    def test_v13_migration_retires_roles_preserves_alignment_and_initializes_workspace(self):
        model,_=multiround_model(); cfg=model.cfg
        saved={n:p.detach().clone() for n,p in model.named_parameters()
               if owner(n) is not None and not n.startswith('knowledge.reasoner.')}
        saved['conditioner.kind.weight']=torch.randn(4,32)
        saved['conditioner.alignment.gain'].fill_(.123)
        old=copy.deepcopy(cfg);old['revision']=ALIGNED_REVISION
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'legacy.pt'
            torch.save(dict(revision=old['revision'],config=old,step=16,model=saved),path)
            before={n:p.clone() for n,p in model.knowledge.reasoner.named_parameters()}
            report=initialize_generator(model,cfg,path)
            self.assertTrue(report['reasoning_migration']);self.assertFalse(report['alignment_migration'])
            self.assertEqual(report['discarded_tensors'],['conditioner.kind.weight'])
            self.assertTrue(all(n.startswith('knowledge.reasoner.') for n in report['new_tensors']))
            torch.testing.assert_close(model.conditioner.alignment.gain,saved['conditioner.alignment.gain'])
            for n,p in model.knowledge.reasoner.named_parameters():torch.testing.assert_close(p,before[n])

    def test_frozen_verifier_completion_reaches_workspace_through_native_Wan(self):
        model,data=multiround_model();model.cfg['knowledge']['train_reasoning_rounds']=[2]
        states=[]
        def record(module,args,result):
            for state in result: state.retain_grad();states.append(state)
        handle=model.knowledge.reasoner.register_forward_hook(record)
        step=sum(model.cfg['stages'][p] for p in ('interface','calibration'))
        losses,weights,_=model.feedback_losses(data,step)
        model.backward_losses(losses,weights);handle.remove()
        self.assertEqual(len(states),3)
        for state in states:self.assertGreater(float(state.grad.norm()),0)
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.generator.parameters()))
        self.assertGreater(float(model.knowledge.reasoner.query_seed.grad.norm()),0)

    def test_exit_selection_zero_round_and_disabled_analysis(self):
        model,_=multiround_model();k=model.knowledge.eval();image=torch.rand(1,3,32,64)
        select_reasoning_exit(model.cfg,without_updates=True,without_analysis=True)
        with patch.object(k,'_multiround_analysis',side_effect=AssertionError('analysis disabled')):
            result=k.encode(image,'Smoke persists.',[])
        self.assertEqual(len(result['relation_states']),1)
        self.assertEqual(result['latent_reasoning_steps'],0)
        with self.assertRaises(ValueError):select_reasoning_exit(model.cfg,reasoning_round=3)
        with self.assertRaises(ValueError):select_reasoning_exit(model.cfg,without_updates=True,reasoning_round=2)

    def test_duplicate_source_records_and_overflow_are_explicit(self):
        clause=dict(text='Smoke remains.',basis='image')
        _,sections=content_sections('Smoke remains.',[clause,clause,dict(clause,basis='prompt')])
        self.assertEqual(len(sections),3)  # request, observed clause, requested clause
        self.assertTrue(sections[1].startswith('[OBSERVED'))
        model,_=multiround_model();model.knowledge.cfg['max_text_tokens']=32
        with self.assertRaisesRegex(ValueError,'budget exceeded'):
            model.knowledge.encode(torch.rand(1,3,32,64),'The smoke persists. '*100,[],reasoning_round=0)

    def test_bfloat16_reasoning_and_native_reader_backward(self):
        model,data=multiround_model();model.cfg['knowledge']['train_reasoning_rounds']=[2]
        with torch.autocast('cpu',dtype=torch.bfloat16):
            c=model.encode_condition(model.conditions(data),torch.rand(1,3,32,64))
            out=model.generator(data['latent'],data['first'],data['text'],torch.tensor(.5),c,data['times'])
            loss=out.float().square().mean()
        self.assertTrue(torch.isfinite(loss));loss.backward()
        gradient=model.knowledge.reasoner.feedback.weight.grad
        self.assertTrue(torch.isfinite(gradient).all());self.assertGreater(float(gradient.norm()),0)

    def test_old_operator_migration_explicitly_retires_bottleneck_and_creates_alignment(self):
        from physgen_native.config import BASELINE_REVISION
        model,_=multiround_model();cfg=model.cfg
        saved={n:p.detach().clone() for n,p in model.named_parameters()
               if owner(n) is not None and not n.startswith(('knowledge.reasoner.','conditioner.alignment.'))}
        saved['conditioner.kind.weight']=torch.randn(4,32)
        saved['generator.interface.sites.0.up.weight']=torch.randn(32,4)
        old=copy.deepcopy(cfg);old['revision']=BASELINE_REVISION
        old['model']['injection']='conditional_operator'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'old.pt'
            torch.save(dict(revision=old['revision'],config=old,step=16,model=saved),path)
            report=initialize_generator(model,cfg,path)
            self.assertTrue(report['alignment_migration']);self.assertTrue(report['reasoning_migration'])
            self.assertIn('generator.interface.sites.0.up.weight',report['discarded_tensors'])
            self.assertIn('conditioner.alignment.gain',report['new_tensors'])


if __name__=='__main__': unittest.main()
