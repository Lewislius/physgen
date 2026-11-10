"""v13 alignment, native-query algebra, real optimizer ownership and C feedback."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'tests'))
from test_native import FakeWan, FakeDecoder, sample, grad_size
from test_operator import TextKnowledge
from test_fullwidth_attention import NativeBlock, fullwidth_model
from physgen_native.config import read_config, validate, ALIGNED_REVISION
from physgen_native.context_alignment import FullWidthAlignment, native_text_context, context_statistics
from physgen_native.data import VideoDataset
from physgen_native.native_attention import NativeAttentionContext
from physgen_native.runtime import (owner, parameter_groups, schedule, initialize_generator,
                                     save_checkpoint, load_checkpoint, restore)
from physgen_native.system import PhysGenNative


def aligned_model():
    old, _ = fullwidth_model(); cfg = copy.deepcopy(old.cfg)
    cfg['revision'] = ALIGNED_REVISION
    cfg['model']['injection'] = 'aligned_native_cross_attention'
    cfg['alignment'] = dict(initialization='native_text_context', samples=2, eps=1e-6)
    cfg['train']['wan_lora_in_interface'] = False
    del cfg['loss']['interface']['intensity']
    validate(cfg)
    wan = FakeWan(cfg); wan.blocks = nn.ModuleList([NativeBlock() for _ in range(3)])
    model = PhysGenNative(cfg, wan, knowledge=TextKnowledge(), decoder=FakeDecoder(), device='cpu')
    # Nonzero inherited adapters must remain active in forward and frozen in stage 1.
    with torch.no_grad():
        for name, parameter in model.generator.wan.named_parameters():
            if name.endswith('lora_b'): parameter.normal_(std=.01)
        model.conditioner.alignment.gain.fill_(.1)
    data = sample(); data.pop('intensity'); data.pop('change_valid')
    return model, data


class AlignedAttentionTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(261026)

    def test_active_config_and_parameters_remove_a_and_keep_full_width(self):
        cfg = read_config(ROOT/'configs/native_aligned_attention_wisa3857.yaml')
        self.assertEqual([len(cfg['loss'][p]) for p in cfg['stages']], [3, 1, 4])
        self.assertEqual((cfg['model']['width'], cfg['model']['attention_width'], cfg['model']['attention_heads']),
                         (3072, 3072, 24))
        model, _ = aligned_model()
        self.assertEqual(sum(p.numel() for p in model.generator.interface.parameters()), 0)
        self.assertFalse(any('intensity' in n for n, _ in model.named_parameters()))
        self.assertFalse(any(p.requires_grad for p in model.generator.wan.parameters()))
        self.assertEqual(sum(p.numel() for p in FullWidthAlignment(3072).parameters()), 6144)
        cfg['train']['wan_lora_in_interface'] = True
        with self.assertRaisesRegex(ValueError, 'freezes all Wan'): validate(cfg)

    def test_reader_reuses_exact_native_Q_and_keeps_text_and_first_frame(self):
        model, _ = aligned_model(); interface = model.generator.interface
        attention = model.generator.wan.blocks[0].cross_attn
        x = torch.randn(1, 12, 32); text = torch.randn(1, 5, 32)
        c = torch.randn(1, 4, 32, requires_grad=True)
        native = attention(x, text, None)
        packet = NativeAttentionContext(text, c, interface, (3, 2, 2), 1., 0)
        reshape = lambda value: value.reshape(1, -1, 4, 8).transpose(1, 2)
        q = reshape(attention.norm_q(attention.q(x)))
        k = reshape(attention.norm_k(attention.k(c)))
        v = reshape(attention.v(c)-attention.v.base.bias)
        message = F.scaled_dot_product_attention(q[:, :, 4:], k, v).transpose(1, 2).flatten(2)
        # LoRA in O is linear too; subtract its already-used bias exactly once.
        expected = torch.cat((native[:, :4], native[:, 4:]+attention.o(message)-attention.o.base.bias), 1)
        with patch.object(attention.q, 'forward', wraps=attention.q.forward) as call:
            actual = attention(x, packet, None)
        self.assertEqual(call.call_count, 1)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        torch.testing.assert_close(actual[:, :4], native[:, :4], rtol=0, atol=0)
        off = attention(x, NativeAttentionContext(text, c, interface, (3, 2, 2), 0., 0), None)
        torch.testing.assert_close(off, native, rtol=0, atol=0)
        actual[:, -1].square().mean().backward()
        self.assertGreater(float(c.grad.abs().sum()), 0)

    def test_statistics_use_post_embedding_training_context_including_native_padding(self):
        model, _ = aligned_model(); wan = model.generator.wan
        texts = {'a': torch.randn(3, 12), 'b': torch.randn(11, 12)}
        rows = [dict(id='a', split='train'), dict(id='excluded', split='validation'), dict(id='b', split='train')]
        def load(path): return texts[Path(path).stem]
        before = torch.get_rng_state().clone()
        with patch('physgen_native.data.manifest', return_value=rows), patch('physgen_native.data.load_tensor', side_effect=load):
            statistics = context_statistics(wan, model.cfg)
        full = torch.cat([native_text_context(wan, texts[name][None])[0] for name in ('a', 'b')]).double()
        torch.testing.assert_close(statistics['mean'], full.mean(0).float())
        torch.testing.assert_close(statistics['std'], full.std(0, correction=0).float())
        self.assertEqual(statistics['report']['tokens'], 16)
        self.assertEqual(statistics['report']['unpadded_tokens'], 11)
        self.assertEqual(statistics['report']['train_ids'], ['a', 'b'])
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
        alignment = model.conditioner.alignment
        alignment.initialize(statistics['mean'], statistics['std'])
        torch.testing.assert_close(alignment.gain, statistics['std'])
        torch.testing.assert_close(alignment.bias, statistics['mean'])
        content = torch.randn(1, 4, 32, requires_grad=True)
        output = alignment(content)
        expected = F.layer_norm(content, (32,), eps=1e-6)*statistics['std']+statistics['mean']
        torch.testing.assert_close(output, expected)
        output.square().mean().backward()
        self.assertGreater(float(content.grad.abs().sum()), 0)

    def test_interface_updates_C_and_knowledge_but_not_any_Wan_weight_then_joint_unlocks_LoRA(self):
        model, data = aligned_model(); optimizer = torch.optim.AdamW(parameter_groups(model, model.cfg))
        scheduler = schedule(optimizer, model.cfg)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        losses, weights, _ = model(data, torch.zeros_like(data['text']), 0, sigma_override=.2)
        self.assertEqual(set(losses), {'fm', 'flow_temporal', 'process'})
        model.backward_losses(losses, weights)
        self.assertEqual(grad_size(model.generator.wan), 0)
        for module in (model.conditioner.project, model.conditioner.alignment, model.knowledge):
            self.assertGreater(grad_size(module), 0)
        self.assertEqual(next(g['lr'] for g in optimizer.param_groups if g['name'] == 'wan_lora'), 0.)
        optimizer.step(); scheduler.step()
        for name, parameter in model.generator.wan.named_parameters():
            torch.testing.assert_close(parameter, before['generator.wan.'+name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model.conditioner.alignment.gain, before['conditioner.alignment.gain']))
        self.assertFalse(torch.equal(model.conditioner.project.weight, before['conditioner.project.weight']))
        model.zero_grad(set_to_none=True); model.set_phase(1)
        self.assertTrue(all(owner(n) == 'critic' for n, p in model.named_parameters() if p.requires_grad))
        losses, weights, _ = model(data, torch.zeros_like(data['text']), 1, micro=1)
        model.backward_losses(losses, weights)
        self.assertGreater(grad_size(model.critic), 0)
        self.assertEqual(grad_size(model.conditioner)+grad_size(model.knowledge)+grad_size(model.generator), 0)
        model.zero_grad(set_to_none=True); scheduler.step()
        losses, weights, _ = model(data, torch.zeros_like(data['text']), 2, sigma_override=.2)
        model.backward_losses(losses, weights)
        self.assertGreater(grad_size(model.generator.wan), 0)
        self.assertTrue(all(n.endswith(('lora_a', 'lora_b')) for n, p in model.generator.wan.named_parameters() if p.requires_grad))
        self.assertGreater(next(g['lr'] for g in optimizer.param_groups if g['name'] == 'wan_lora'), 0.)

    def test_verifier_feedback_passes_alignment_and_updates_only_C_knowledge(self):
        model, data = aligned_model()
        losses, weights, metrics = model.feedback_losses(data, 2)
        model.backward_losses(losses, weights)
        for module in (model.conditioner.project, model.conditioner.alignment, model.knowledge):
            self.assertGreater(grad_size(module), 0)
        self.assertEqual(grad_size(model.generator)+grad_size(model.critic), 0)
        self.assertGreater(float(metrics['C_gradient_rms']), 0)

    def test_checkpoint_recomputation_with_frozen_Wan_preserves_C_gradient(self):
        model, data = aligned_model(); generator = model.generator
        base = torch.randn(1, 5, 32); noisy = torch.randn_like(data['latent']); results = []
        for enabled in (False, True):
            generator.checkpointing = enabled
            c = base.clone().requires_grad_(True)
            output = generator(noisy, data['first'], data['text'], torch.tensor(.4), c, data['times'])
            output.square().mean().backward()
            results.append((output.detach(), c.grad.clone()))
        for a, b in zip(*results): torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertGreater(float(results[0][1].abs().sum()), 0)
        self.assertEqual(grad_size(generator), 0)

    def test_dataset_does_not_require_change_scale_or_intensity_cache(self):
        model, data = aligned_model()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'v52/requirements').mkdir(parents=True)
            row = dict(id='a', split='train', caption='A process.', pts=list(range(21)),
                       geometry=dict(frames=21, h=8, w=12, top=0, left=0, rh=8, rw=12))
            (root/'v52/requirements/a.json').write_text(json.dumps(dict(input_scope='caption_and_first_image', clauses=[])))
            cfg = copy.deepcopy(model.cfg); cfg['paths']['cache'] = str(root)
            def load(path):
                if '/base/vae/' in str(path): return dict(latent=data['latent'], first=data['first'])
                if '/base/text/' in str(path): return data['text'][0]
                raise AssertionError(f'Unused target was read: {path}')
            with patch('physgen_native.data.manifest', return_value=[row]), patch('physgen_native.data.load_tensor', side_effect=load):
                result = VideoDataset(cfg)[0]
            self.assertNotIn('intensity', result); self.assertNotIn('change_valid', result)
            self.assertNotIn('evidence_activity', result)

    def test_migration_removes_old_a_and_roundtrip_keeps_learned_affine(self):
        old, _ = fullwidth_model(); model, _ = aligned_model()
        saved = {n: p.detach().clone() for n, p in old.named_parameters() if owner(n) is not None}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'v12.pt'
            torch.save(dict(revision=old.cfg['revision'], config=old.cfg, step=16, model=saved), source)
            report = initialize_generator(model, model.cfg, source)
            self.assertTrue(report['alignment_migration'])
            self.assertEqual(set(report['new_tensors']), {'conditioner.alignment.gain', 'conditioner.alignment.bias'})
            self.assertTrue(all(n.startswith('generator.interface.intensity') for n in report['discarded_tensors']))
            for n, p in model.named_parameters():
                if n in saved and owner(n) not in (None, 'critic'): torch.testing.assert_close(p, saved[n], rtol=0, atol=0)
            with torch.no_grad():
                model.conditioner.alignment.gain.uniform_(.02, .08)
                model.conditioner.alignment.bias.uniform_(-.02, .02)
            optimizer = torch.optim.AdamW(parameter_groups(model, model.cfg)); scheduler = schedule(optimizer, model.cfg)
            path = Path(directory)/'v13.pt'
            save_checkpoint(model, optimizer, scheduler, model.cfg, 1, 1, path)
            state = load_checkpoint(path); other, _ = aligned_model()
            other_optimizer = torch.optim.AdamW(parameter_groups(other, other.cfg)); other_scheduler = schedule(other_optimizer, other.cfg)
            self.assertEqual(restore(other, other_optimizer, other_scheduler, state, other.cfg), (1, 1))
            torch.testing.assert_close(other.conditioner.alignment.gain, model.conditioner.alignment.gain, rtol=0, atol=0)
            torch.testing.assert_close(other.conditioner.alignment.bias, model.conditioner.alignment.bias, rtol=0, atol=0)
            transfer = initialize_generator(other, other.cfg, path)
            self.assertFalse(transfer['alignment_migration']); self.assertFalse(transfer['new_tensors'])


if __name__ == '__main__': unittest.main()
