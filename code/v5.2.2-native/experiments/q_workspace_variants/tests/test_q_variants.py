"""Meaningful CPU checks of initialization, addressing, gradients and resume."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
import torch
from torch.nn import functional as F
import yaml
from fixtures import small_config, model_and_sample
from physgen_native.message_conditioning import MessageCondition, MessageBlock
from physgen_native.q_variants import pool_valid_content, normalized_query_identity
from physgen_native.config import read_config, validate, smoke_config
from physgen_native.runtime import parameter_groups, schedule, save_checkpoint, load_checkpoint, restore

VARIANTS = ('b', 'c', 'd-n', 'd-s', 'd-i')


def evidence(width=32):
    return dict(memory=torch.randn(2, 150, width, requires_grad=True),
                memory_mask=torch.ones(2, 150, dtype=torch.bool),
                content_positions=torch.arange(140),
                content_mask=torch.ones(2, 140, dtype=torch.bool))


class VariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_real_configs_jobs_and_production_parameter_shapes(self):
        revisions = []
        for variant in VARIANTS:
            stem = variant.replace('-', '_')
            cfg = read_config(ROOT/'configs'/f'{stem}.yaml')
            revisions.append(cfg['revision'])
            self.assertEqual(cfg['stages'], {'interface': 64, 'joint': 160})
            self.assertEqual(cfg['train']['accumulation'], 4)
            self.assertEqual(cfg['train']['seed'], 42)
            self.assertEqual(cfg['validation']['sample_seeds'], [42])
            self.assertEqual(cfg['verifier'], {'kind': 'none'})
            self.assertNotIn('completion', cfg)
            self.assertEqual(cfg['loss'], {p: {'fm': 1., 'flow_temporal': .1} for p in ('interface', 'joint')})
            self.assertNotIn('completion', smoke_config(cfg))
            job = yaml.safe_load((ROOT/'train'/f'{stem}_1x{"80" if variant == "b" else "96"}g.yaml').read_text())
            self.assertEqual(job['resources']['resource_pool'], 'amp-80g' if variant == 'b' else 'blk-96g')
            self.assertEqual(job['resources']['slots_per_trial'], 1)
            self.assertIn(str(ROOT/'train'/f'train_{stem}.sh'), job['entrypoint'])
            self.assertNotIn('--resume', job['entrypoint'])
            with torch.device('meta'):
                model = MessageCondition(cfg)
            self.assertEqual(tuple(model.query_seed.shape), (1, 128, 4096))
            self.assertEqual(tuple(model.project.weight.shape), (3072, 4096))
            has_self = variant not in ('c', 'd-n')
            self.assertEqual(hasattr(model.block, 'self_attention'), has_self)
            self.assertEqual(model.block.exchange, has_self)
        self.assertEqual(len(set(revisions)), 5)

    def test_pooling_exact_bins_padding_and_short_inputs(self):
        x = torch.arange(384.).reshape(1, 384, 1).repeat(1, 1, 3)
        actual = pool_valid_content(x, torch.ones(1, 384, dtype=torch.bool))
        torch.testing.assert_close(actual, x.reshape(1, 128, 3, 3).mean(2))
        values = torch.arange(5.).reshape(1, 5, 1)
        torch.testing.assert_close(pool_valid_content(values, torch.ones(1, 5, dtype=torch.bool), 3),
                                   torch.tensor([[[0.], [1.5], [3.5]]]))
        padded = torch.tensor([[[1.], [9999.], [2.], [-9999.]]])
        out = pool_valid_content(padded, torch.tensor([[True, False, True, False]]), 4)
        torch.testing.assert_close(out, torch.tensor([[[1.], [1.], [2.], [2.]]]))
        only = pool_valid_content(torch.tensor([[[7.]]]), torch.tensor([[True]]))
        self.assertEqual(tuple(only.shape), (1, 128, 1))
        self.assertTrue(torch.equal(only, torch.full_like(only, 7)))
        with self.assertRaises(ValueError):
            pool_valid_content(values, torch.zeros(1, 5, dtype=torch.bool))

    def test_all_common_initial_weights_and_rng_match_original_q(self):
        cfg = small_config('b')
        cfg['revision'] = 'v522_native_q_seed128_no_verifier_simple_v30'
        for key in ('workspace_variant', 'query_initialization', 'self_attention', 'cross_query_identity',
                    'self_query_identity', 'content_pooling', 'identity_normalization', 'identity_start_round', 'identity_eps'):
            cfg['message'].pop(key)
        torch.manual_seed(42)
        original = MessageCondition(cfg)
        expected_rng = torch.get_rng_state().clone()
        reference = original.state_dict()
        for variant in VARIANTS:
            torch.manual_seed(42)
            model = MessageCondition(small_config(variant))
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng), variant)
            for name, tensor in model.state_dict().items():
                self.assertTrue(torch.equal(tensor, reference[name]), (variant, name))

    def test_initialization_reads_current_valid_content_only_for_b_c(self):
        data = evidence()
        for variant in VARIANTS:
            cfg = small_config(variant)
            cfg['knowledge']['gradient_checkpointing'] = False
            model = MessageCondition(cfg)
            inputs = []
            handle = model.block.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach().clone()))
            first = model(data)
            changed = dict(data, memory=data['memory'].detach().clone())
            changed['memory'][:, :, 0] += 2.
            model(changed)
            handle.remove()
            self.assertEqual(len(inputs), 6)
            self.assertEqual(tuple(first.shape), (2, 128, 32))
            if variant in ('b', 'c'):
                self.assertFalse(torch.equal(inputs[0], inputs[3]))
            else:
                torch.testing.assert_close(inputs[0], model.query_seed.detach().expand(2, -1, -1))
                torch.testing.assert_close(inputs[0], inputs[3])

    def test_exact_identity_scope_qk_and_content_only_values(self):
        for variant in VARIANTS:
            cfg = small_config(variant)
            cfg['knowledge']['gradient_checkpointing'] = False
            model = MessageCondition(cfg)
            captured = {'state': [], 'cross': [], 'self_q': [], 'self_k': [], 'self_v': [], 'self_h': []}
            def save(key):
                return lambda module, args: captured[key].append(args[0].detach().clone())
            hooks = [model.block.register_forward_pre_hook(save('state')),
                     model.block.cross.q.register_forward_pre_hook(save('cross'))]
            if model.block.exchange:
                hooks += [model.block.self_attention.q.register_forward_pre_hook(save('self_q')),
                          model.block.self_attention.k.register_forward_pre_hook(save('self_k')),
                          model.block.self_attention.v.register_forward_pre_hook(save('self_v')),
                          model.block.exchange_norm.register_forward_hook(
                              lambda module, args, output: captured['self_h'].append(output.detach().clone()))]
            model(evidence())
            for hook in hooks: hook.remove()
            self.assertEqual(len(captured['cross']), 3)
            identity = normalized_query_identity(model.query_seed).detach().expand(2, -1, -1)
            for round_id, state in enumerate(captured['state']):
                want = model.block.query_norm(state)
                if variant.startswith('d-') and round_id > 0: want = want + identity
                torch.testing.assert_close(captured['cross'][round_id], want)
                if model.block.exchange:
                    h = captured['self_h'][round_id]
                    address = h+identity if variant == 'd-i' and round_id > 0 else h
                    torch.testing.assert_close(captured['self_q'][round_id], address)
                    torch.testing.assert_close(captured['self_k'][round_id], address)
                    torch.testing.assert_close(captured['self_v'][round_id], h)

    def test_identity_has_gradient_even_when_content_state_is_independent_of_e(self):
        block = MessageBlock(32, 4, 2, exchange=False)
        state = torch.randn(1, 128, 32, requires_grad=True)
        seed = torch.randn(1, 128, 32, requires_grad=True)
        memory = torch.randn(1, 12, 32)
        result = block(state, memory, torch.ones(1, 12, dtype=torch.bool),
                       torch.ones(1, 128, dtype=torch.bool), normalized_query_identity(seed))
        (result*torch.randn_like(result)).sum().backward()
        self.assertGreater(float(seed.grad.norm()), 0.)

    def test_checkpoint_recomputation_matches_direct_gradients_and_detaches_qwen(self):
        for variant in VARIANTS:
            cfg = small_config(variant)
            checked = MessageCondition(cfg)
            direct_cfg = copy.deepcopy(cfg)
            direct_cfg['knowledge']['gradient_checkpointing'] = False
            direct = MessageCondition(direct_cfg)
            direct.load_state_dict(checked.state_dict())
            data = evidence()
            a, b = checked(data), direct(data)
            torch.testing.assert_close(a, b)
            target = torch.randn_like(a)
            (a*target).mean().backward()
            (b*target).mean().backward()
            self.assertIsNone(data['memory'].grad)
            self.assertGreater(float(checked.query_seed.grad.norm()), 0.)
            for (name, param), (other, expected) in zip(checked.named_parameters(), direct.named_parameters()):
                self.assertEqual(name, other)
                self.assertIsNotNone(param.grad, name)
                torch.testing.assert_close(param.grad, expected.grad, atol=1e-7, rtol=1e-4)

    def test_real_fm_flow_backward_phase_ownership_and_native_sampling(self):
        for variant in VARIANTS:
            model, sample = model_and_sample(variant)
            null = torch.zeros_like(sample['text'])
            self.assertIsNone(model.critic)
            self.assertIsNone(model.process_encoder)
            with patch.object(model, 'feedback_losses', side_effect=AssertionError('verifier invoked')):
                for step in (0, 64):
                    model.zero_grad(set_to_none=True)
                    losses, weights, metrics = model(sample, null, step, sigma_override=.65)
                    self.assertEqual(set(losses), {'fm', 'flow_temporal'})
                    self.assertEqual(model.last_condition_round, 2)
                    model.backward_losses(losses, weights)
                    self.assertGreater(float(model.conditioner.query_seed.grad.norm()), 0.)
                    self.assertTrue(all(p.grad is None for p in model.knowledge.parameters()))
                    wan_grad = sum(float(p.grad.abs().sum()) for p in model.generator.wan.parameters() if p.grad is not None)
                    self.assertEqual(wan_grad == 0., step == 0)
            model.eval()
            with torch.no_grad():
                first = model.sample(model.conditions(sample), sample['text'], null, seed=42, steps=2)
                second = model.sample(model.conditions(sample), sample['text'], null, seed=42, steps=2)
            torch.testing.assert_close(first, second, rtol=0, atol=0)
            torch.testing.assert_close(first[:, :, :1], sample['first'], rtol=0, atol=0)
            self.assertTrue(bool(torch.isfinite(first).all()))

    def test_checkpoint_resume_restores_model_optimizer_and_rng(self):
        for variant in VARIANTS:
            model, sample = model_and_sample(variant)
            optimizer = torch.optim.AdamW(parameter_groups(model, model.cfg), foreach=False)
            scheduler = schedule(optimizer, model.cfg)
            null = torch.zeros_like(sample['text'])
            def update(net, opt, sched, step):
                opt.zero_grad(set_to_none=True)
                losses, weights, _ = net(sample, null, step, sigma_override=.65)
                net.backward_losses(losses, weights)
                opt.step(); sched.step()
            update(model, optimizer, scheduler, 0)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'latest.pt'
                save_checkpoint(model, optimizer, scheduler, model.cfg, 1, 4, path)
                state = load_checkpoint(path)
                update(model, optimizer, scheduler, 1)
                resumed, _ = model_and_sample(variant)
                # Frozen base tensors are normally loaded from the same model files.
                resumed.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})
                opt2 = torch.optim.AdamW(parameter_groups(resumed, resumed.cfg), foreach=False)
                sched2 = schedule(opt2, resumed.cfg)
                self.assertEqual(restore(resumed, opt2, sched2, state, resumed.cfg), (1, 4))
                update(resumed, opt2, sched2, 1)
                for name, expected in model.state_dict().items():
                    torch.testing.assert_close(resumed.state_dict()[name], expected, rtol=0, atol=0)
                self.assertNotIn('verifier_state', state)
                wrong = copy.deepcopy(resumed.cfg)
                wrong['revision'] = read_config(ROOT/'configs'/('c.yaml' if variant != 'c' else 'b.yaml'))['revision']
                with self.assertRaises(ValueError): restore(resumed, opt2, sched2, state, wrong)

    def test_revision_cannot_silently_change_architecture(self):
        cfg = small_config('b')
        cfg['message']['self_attention'] = False
        with self.assertRaises(ValueError): validate(cfg)
        cfg = small_config('d-i')
        cfg['message']['identity_start_round'] = 0
        with self.assertRaises(ValueError): validate(cfg)

    def test_b_accepts_amp80_and_others_reject_h100_before_data_loading(self):
        from train.train import train
        class ReachedAudit(RuntimeError):
            pass
        args = SimpleNamespace(config=str(ROOT/'configs/b.yaml'), stop_after=None,
                               check_only=False, smoke=False, smoke_accumulation=None, checkpoint_every=None)
        props = SimpleNamespace(name='NVIDIA A100-SXM4-80GB', total_memory=80*2**30)
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1), \
             patch('torch.cuda.get_device_properties', return_value=props), \
             patch('train.train.audit', side_effect=ReachedAudit('accepted B device')):
            with self.assertRaises(ReachedAudit): train(args)
        for variant in ('c', 'd_n', 'd_s', 'd_i'):
            args.config = str(ROOT/'configs'/f'{variant}.yaml')
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('torch.cuda.device_count', return_value=1), \
                 patch('torch.cuda.get_device_name', return_value='NVIDIA H100 80GB HBM3'), \
                 patch('train.train.audit', side_effect=AssertionError('data opened before device rejection')):
                with self.assertRaisesRegex(RuntimeError, 'GPU policy rejected'): train(args)

    def test_device_policy_rejects_multiple_gpus_and_small_amp_allocation(self):
        from tools.gpu_policy import require_single_gpu, require_amp_80g
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            with self.assertRaisesRegex(RuntimeError, 'exactly one'): require_single_gpu()
        props = SimpleNamespace(name='NVIDIA A100-SXM4-40GB', total_memory=40*2**30)
        with patch('torch.cuda.get_device_properties', return_value=props):
            with self.assertRaisesRegex(RuntimeError, 'amp-80g'): require_amp_80g()


if __name__ == '__main__':
    unittest.main(verbosity=2)
