"""Message reasoning through native Qwen, the Wan reader and training checkpoints.

CPU integration checks with tiny random native models; not a video-quality eval.
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
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
from test_multiround_reasoning import multiround_model
from test_native import FakeWan, FakeDecoder, grad_size
from test_fullwidth_attention import NativeBlock
from physgen_native.config import (read_config, validate, select_reasoning_exit,
    H_CONTENT_MESSAGE_REVISION, Q_SEED128_MESSAGE_REVISION)
from physgen_native.knowledge import QwenKnowledge
from physgen_native.message_conditioning import MessageCondition
from physgen_native.native_attention import NativeAttentionContext
from physgen_native.runtime import (parameter_groups, schedule, save_checkpoint, load_checkpoint,
                                    restore, initialize_generator)
from physgen_native.system import PhysGenNative

torch.set_num_threads(2)


def message_model(query=False, *, observers=True):
    from transformers import Qwen3VLForConditionalGeneration
    old, data = multiround_model()
    cfg = copy.deepcopy(old.cfg)
    cfg['revision'] = Q_SEED128_MESSAGE_REVISION if query else H_CONTENT_MESSAGE_REVISION
    k = cfg['knowledge']
    for key in ('lora_rank', 'lora_alpha', 'lora_last_layers', 'relation_tokens'):
        k.pop(key, None)
    k.update(reasoning_mode='q_seed128_message' if query else 'h_content_message',
             freeze_backbone=True, latent_reasoning_attention='bidirectional',
             train_reasoning_rounds=[0, 1, 2] if query else [1, 2], inference_reasoning_round=2)
    if query: k['relation_tokens'] = 128
    cfg['message'] = dict(heads=4, ffn_multiplier=2, init_std=.02)
    cfg['stages'] = dict(interface=1, joint=2)
    cfg['loss'].pop('calibration')
    cfg['verifier'].update(kind='qwen_native', temperature=1.)
    cfg['verifier'].pop('calibrate_current', None)
    cfg['train'].update(qwen_lr=0., critic_lr=0.)
    validate(cfg)
    knowledge = QwenKnowledge(cfg, 'cpu', backbone=Qwen3VLForConditionalGeneration(old.knowledge.backbone.config),
                              tokenizer=old.knowledge.tokenizer)
    wan = FakeWan(cfg); wan.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
    model = PhysGenNative(cfg, wan, knowledge=knowledge, decoder=FakeDecoder(), device='cpu',
                         training_observers=observers)
    return model, data


class MessageConditionTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(261027)

    def test_configs_have_separate_revisions_no_qwen_adapter_or_critic_calibration(self):
        for mode in ('h_content_message', 'q_seed128_message'):
            cfg = read_config(ROOT / 'configs' / f'native_{mode}_wisa3857.yaml')
            self.assertEqual(tuple(cfg['stages']), ('interface', 'joint'))
            self.assertEqual(cfg['verifier']['kind'], 'qwen_native')
            self.assertNotIn('lora_rank', cfg['knowledge'])
            self.assertEqual(cfg['model']['injection'], 'aligned_native_cross_attention')
            bad = copy.deepcopy(cfg); bad['knowledge']['freeze_backbone'] = False
            with self.assertRaises(ValueError): validate(bad)
            select_reasoning_exit(cfg, without_updates=True)
            self.assertEqual(cfg['knowledge']['inference_reasoning_round'], 0)
        cfg = read_config(ROOT / 'configs/native_q_seed128_message_wisa3857.yaml')
        cfg['knowledge']['relation_tokens'] = 32
        with self.assertRaisesRegex(ValueError, '128'): validate(cfg)

    def test_native_evidence_complete_lengths_rounds_and_frozen_encoder(self):
        for query in (False, True):
            with self.subTest(query=query):
                model, data = message_model(query)
                model.train()
                self.assertFalse(model.knowledge.backbone.training)
                self.assertFalse(any('lora_' in n for n, _ in model.knowledge.named_parameters()))
                self.assertEqual(sum(p.numel() for p in model.critic.parameters()), 0)
                image = torch.rand(1, 3, 32, 64, requires_grad=True)
                conditions = model.conditions(data)
                conditions['caption'] = 'Smoke continues from the source and spreads. ' * 40
                with patch.object(model.conditioner.block, 'forward', wraps=model.conditioner.block.forward) as block:
                    result = model.encode_condition_rounds(conditions, image)
                    self.assertEqual(block.call_count, 3 if query else 2)
                self.assertGreater(result['content'].shape[1], 128)
                expected = 128 if query else result['content'].shape[1]
                self.assertEqual([c.shape for c in result['conditions']], [torch.Size([1, expected, 32])] * 3)
                self.assertFalse(torch.equal(result['conditions'][0], result['conditions'][2]))
                probe = torch.randn_like(result['conditions'][2])
                (result['conditions'][2] * probe).sum().backward()
                self.assertIsNone(image.grad)
                self.assertEqual(grad_size(model.knowledge), 0)
                self.assertGreater(grad_size(model.conditioner.project), 0)
                self.assertGreater(grad_size(model.conditioner.block), 0)
                if query: self.assertGreater(float(model.conditioner.query_seed.grad.norm()), 0)

    def test_padding_and_batched_content_positions_do_not_change_valid_outputs(self):
        for query in (False, True):
            model, _ = message_model(query)
            module = model.conditioner.eval()
            memory = torch.randn(2, 11, 32, requires_grad=True)
            memory_mask = torch.arange(11)[None] < torch.tensor([[8], [11]])
            positions = torch.tensor([[1, 3, 6, -1], [0, 2, 5, 9]])
            content_mask = positions >= 0
            evidence = dict(memory=memory, memory_mask=memory_mask,
                            content_positions=positions, content_mask=content_mask)
            batch = module(evidence, reasoning_round=2)
            for i, length in enumerate((8, 11)):
                indices = positions[i, content_mask[i]]
                single = module(dict(memory=memory[i:i+1, :length],
                    memory_mask=torch.ones(1, length, dtype=torch.bool), content_positions=indices), reasoning_round=2)
                count = 128 if query else len(indices)
                torch.testing.assert_close(batch[i:i+1, :count], single, rtol=1e-5, atol=2e-6)
            noisy = memory.detach().clone(); noisy[~memory_mask] = 10000
            repeated = module(dict(evidence, memory=noisy), reasoning_round=2)
            torch.testing.assert_close(batch, repeated, rtol=0, atol=0)
            batch.square().sum().backward()
            self.assertIsNone(memory.grad)
            if not query: self.assertEqual(float(batch[0, 3].detach().abs().sum()), 0)

    def test_checkpointing_preserves_message_values_and_gradients(self):
        model, _ = message_model(True)
        module = model.conditioner
        evidence = dict(memory=torch.randn(1, 9, 32), memory_mask=torch.ones(1, 9, dtype=torch.bool),
                        content_positions=torch.tensor([2, 3, 6]))
        checkpointed = module(evidence, reasoning_round=2)
        probe = torch.randn_like(checkpointed)
        (checkpointed * probe).sum().backward()
        gradients = {n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}
        module.zero_grad(set_to_none=True); module.checkpointing = False
        plain = module(evidence, reasoning_round=2)
        torch.testing.assert_close(plain, checkpointed, rtol=0, atol=0)
        (plain * probe).sum().backward()
        for n, p in module.named_parameters():
            torch.testing.assert_close(p.grad, gradients[n], rtol=2e-5, atol=2e-6)

    def test_native_Wan_reader_uses_each_output_and_masks_padding(self):
        for query in (False, True):
            model, data = message_model(query)
            result = model.encode_condition_rounds(model.conditions(data), torch.rand(1, 3, 32, 64))
            c = result['conditions'][2]
            attention = model.generator.wan.blocks[0].cross_attn
            x, text = torch.randn(1, 12, 32), torch.randn(1, 5, 32)
            packet = NativeAttentionContext(text, c, model.generator.interface, (3, 2, 2), 1., 0)
            expected = attention(x, packet, None)
            padding = torch.randn(1, 5, 32, requires_grad=True)
            packet.condition = torch.cat((c, padding), 1)
            packet.condition_mask = torch.arange(c.shape[1] + 5)[None] < c.shape[1]
            actual = attention(x, packet, None)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
            actual.square().sum().backward()
            self.assertEqual(float(padding.grad.abs().sum()), 0)
            self.assertGreater(grad_size(model.conditioner.block), 0)
            self.assertEqual(grad_size(model.knowledge), 0)

    def test_training_and_native_verifier_feedback_reach_messages_only(self):
        for query in (False, True):
            model, data = message_model(query)
            losses, weights, _ = model(data, torch.zeros_like(data['text']), 0, sigma_override=.2)
            model.backward_losses(losses, weights)
            self.assertGreater(grad_size(model.conditioner.block), 0)
            self.assertEqual(grad_size(model.knowledge), 0)
            self.assertEqual(grad_size(model.generator.wan), 0)
            model.zero_grad(set_to_none=True); model.set_phase(1)
            self.assertTrue(model.should_feedback(1))
            self.assertTrue(any(p.requires_grad for p in model.generator.wan.parameters()))
            self.assertFalse(any(p.requires_grad for p in model.knowledge.parameters()))
            losses, weights, metrics = model.feedback_losses(data, 1)
            model.backward_losses(losses, weights)
            self.assertTrue(torch.isfinite(losses['verifier']))
            self.assertGreater(grad_size(model.conditioner.block), 0)
            self.assertEqual(grad_size(model.knowledge), 0)
            self.assertEqual(grad_size(model.generator.wan), 0)
            self.assertGreater(metrics['C_gradient_rms'], 0)
            if query: self.assertGreater(float(model.conditioner.query_seed.grad.norm()), 0)

    def test_save_resume_generation_only_and_migrate_legacy_weights(self):
        for query in (False, True):
            model, data = message_model(query)
            optimizer = torch.optim.AdamW(parameter_groups(model, model.cfg))
            scheduler = schedule(optimizer, model.cfg)
            self.assertFalse(any(g['name'] in ('knowledge', 'critic') for g in optimizer.param_groups))
            losses, weights, _ = model(data, torch.zeros_like(data['text']), 0, sigma_override=.2)
            model.backward_losses(losses, weights); optimizer.step(); scheduler.step()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'message.pt'
                save_checkpoint(model, optimizer, scheduler, model.cfg, 1, 1, path)
                state = load_checkpoint(path)
                self.assertFalse(any(n.startswith(('knowledge.', 'critic.')) for n in state['model']))
                clone, _ = message_model(query, observers=False)
                restore(clone, None, None, state, clone.cfg)
                model.eval(); clone.eval()
                image = torch.rand(1, 3, 32, 64)
                # Independently constructed tiny base models need identical frozen weights.
                clone.knowledge.load_state_dict(model.knowledge.state_dict())
                torch.testing.assert_close(model.encode_condition(model.conditions(data), image),
                                           clone.encode_condition(clone.conditions(data), image))
                resumed, _ = message_model(query)
                opt = torch.optim.AdamW(parameter_groups(resumed, resumed.cfg)); sch = schedule(opt, resumed.cfg)
                self.assertEqual(restore(resumed, opt, sch, state, resumed.cfg), (1, 1))
                legacy, _ = multiround_model()
                oldopt = torch.optim.AdamW(parameter_groups(legacy, legacy.cfg)); oldsch = schedule(oldopt, legacy.cfg)
                oldpath = Path(folder) / 'old.pt'
                save_checkpoint(legacy, oldopt, oldsch, legacy.cfg, 0, 0, oldpath)
                report = initialize_generator(resumed, resumed.cfg, oldpath)
                self.assertTrue(report['message_migration'])
                self.assertTrue(any(n.startswith('knowledge.') for n in report['discarded_tensors']))
                self.assertTrue(any(n.startswith('conditioner.block.') for n in report['new_tensors']))

    def test_normal_sampling_and_online_C_update_use_new_conditions(self):
        for query in (False, True):
            model, data = message_model(query)
            model.eval().requires_grad_(False)
            conditions = model.conditions(data)
            with torch.no_grad():
                plain = model.sample(conditions, data['text'], torch.zeros_like(data['text']), seed=41)
            self.assertEqual(plain.shape, data['latent'].shape)
            self.assertTrue(torch.isfinite(plain).all())
            torch.testing.assert_close(plain[:, :, :1], data['first'])
            model.cfg['inference']['latent_feedback'].update(enabled=True, sigmas=[.6], step_size=.01)
            with torch.no_grad():
                refined = model.sample(conditions, data['text'], torch.zeros_like(data['text']), seed=41)
            self.assertTrue(torch.isfinite(refined).all())
            self.assertFalse(torch.equal(plain, refined))
            self.assertEqual(grad_size(model), 0)


if __name__ == '__main__': unittest.main()
