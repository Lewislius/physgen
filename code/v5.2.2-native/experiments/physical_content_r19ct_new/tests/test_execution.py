"""Actual project generation path, with a small Wan-shaped test backbone."""
from pathlib import Path
import sys
import unittest
import tempfile
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests'))
from fake_wan import FakeWan, W, FF, evidence
from physical_content.common import load_config
from physical_content.model import PhysicalModel
from physical_content.objectives import backward_micro

SITES = [0, 2]


def model(video_value_init='official_cross_attn_v',
          read_output_init='official_cross_attn_o'):
    cfg = load_config(ROOT / 'configs/r19ctn.yaml')
    cfg['model'].update(
        width=W, ffn_width=FF, native_heads=2, injection_layers=list(SITES),
        physical_initialization_blocks=list(SITES), q_tokens=8, q_heads=4,
        qwen_width=16, state_expert_mode='per_site',
        video_value_init=video_value_init,
        read_output_init=read_output_init, physical_state_width=W)
    cfg['train']['gradient_checkpointing'] = True
    wan = FakeWan()
    wan.text_embedding[0] = nn.Linear(16, W)
    wan.time_embedding = nn.Linear(256, W)
    wan.time_projection = nn.Linear(W, 6 * W)
    interface = dict(mean=torch.zeros(16), scale=torch.ones(()))
    return PhysicalModel(cfg, device='cpu', wan=wan, interface=interface)


def inputs():
    ev = evidence(width=16)
    ev.update(scope='request_only', native_condition=torch.randn(12, W))
    sample = dict(latent=torch.randn(1, 48, 3, 4, 4), text=torch.randn(1, 7, 16),
                  valid=torch.ones(4, 4), times=torch.tensor([[0., .5, 1.]]))
    return ev, sample


class WholePathTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_main_config_uses_prophy_five_layers(self):
        cfg = load_config(ROOT / 'configs/r19ctn.yaml')
        self.assertEqual(cfg['model']['injection_layers'], [0, 7, 14, 21, 28])
        self.assertEqual(cfg['model']['physical_initialization_blocks'],
                         [0, 7, 14, 21, 28])
        self.assertEqual(cfg['model']['state_expert_mode'], 'per_site')
        self.assertEqual(cfg['model']['video_value_init'], 'official_cross_attn_v')
        self.assertEqual(cfg['model']['read_output_init'], 'official_cross_attn_o')

    def test_video_value_and_read_output_must_be_a_coherent_pair(self):
        with self.assertRaises(ValueError):
            model(video_value_init='identity',
                  read_output_init='official_cross_attn_o')

    def test_per_site_initialization_is_same_depth_and_independent(self):
        m = model()
        params_by_site = {}
        for site in SITES:
            mech = m.mechanisms[str(site)]
            expert = m.state_experts[str(site)]
            block = m.wan.blocks[site]
            official_self_q = m.base_bank.linear(site, 'self_attn.q')
            official_ffn0 = m.base_bank.linear(site, 'ffn.0')
            official_ffn2 = m.base_bank.linear(site, 'ffn.2')
            # Read path: Q/K/V/O and RMS norms must come from this exact depth.
            torch.testing.assert_close(mech.video_query.weight, block.cross_attn.q.weight)
            torch.testing.assert_close(mech.content_key.weight, block.cross_attn.k.weight)
            torch.testing.assert_close(mech.video_value.weight, block.cross_attn.v.weight)
            torch.testing.assert_close(mech.read_output.weight, block.cross_attn.o.weight)
            torch.testing.assert_close(mech.query_norm.weight, block.cross_attn.norm_q.weight)
            torch.testing.assert_close(mech.key_norm.weight, block.cross_attn.norm_k.weight)
            # State expert: every copied block component must come from this depth.
            torch.testing.assert_close(expert.condition_attention.q.weight,
                                       block.cross_attn.q.weight)
            torch.testing.assert_close(expert.condition_attention.k.weight,
                                       block.cross_attn.k.weight)
            torch.testing.assert_close(expert.condition_attention.v.weight,
                                       block.cross_attn.v.weight)
            torch.testing.assert_close(expert.condition_attention.out_proj.weight,
                                       block.cross_attn.o.weight)
            torch.testing.assert_close(expert.relation_attention.q.weight,
                                       official_self_q.weight)
            torch.testing.assert_close(expert.temporal_attention.q.weight,
                                       official_self_q.weight)
            torch.testing.assert_close(expert.ffn.layers[0].weight, official_ffn0.weight)
            torch.testing.assert_close(expert.ffn.layers[2].weight, official_ffn2.weight)
            # The write map is per-site and starts as an exact no-op.
            self.assertEqual(float(mech.feature_output.weight.detach().abs().sum()), 0.0)
            params_by_site[site] = (
                {id(p) for p in mech.parameters()} |
                {id(p) for p in expert.parameters()})
        # No trainable physical parameter object may be shared across sites.
        self.assertFalse(params_by_site[0] & params_by_site[2])
        self.assertIsNot(m.mechanisms['0'].video_value, m.mechanisms['2'].video_value)
        self.assertIsNot(m.state_experts['0'], m.state_experts['2'])
        self.assertFalse(torch.equal(m.mechanisms['0'].video_query.weight,
                                     m.mechanisms['2'].video_query.weight))

    def test_identity_video_value_and_read_output_are_per_site(self):
        m = model(video_value_init='identity', read_output_init='identity')
        eye = torch.eye(W)
        for site in SITES:
            mech = m.mechanisms[str(site)]
            torch.testing.assert_close(mech.video_value.weight, eye, atol=0, rtol=0)
            torch.testing.assert_close(mech.read_output.weight, eye, atol=0, rtol=0)
            self.assertEqual(float(mech.video_value.bias.detach().abs().sum()), 0.0)
            self.assertEqual(float(mech.read_output.bias.detach().abs().sum()), 0.0)
        self.assertIsNot(m.mechanisms['0'].video_value,
                         m.mechanisms['2'].video_value)
        self.assertIsNot(m.mechanisms['0'].read_output,
                         m.mechanisms['2'].read_output)

    def test_real_fm_reaches_all_paths_and_owns_parameters_once(self):
        m = model()
        ev, sample = inputs()
        m.set_phase(0)
        c, g = m.prepare_content(ev, sample['text'])
        args = (sample['latent'], None, sample['text'], torch.tensor(.5),
                c, g, sample['times'])
        torch.testing.assert_close(
            m.velocity(*args)[0], m.velocity(*args, physical_mode='off')[0],
            atol=0, rtol=0)
        opt = torch.optim.AdamW(m.optim_groups(), lr=1e-3)
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            report = backward_micro(m, ev, sample, sample['text'], torch.tensor(.5),
                                    torch.randn_like(sample['latent']), 0., 1)
            self.assertTrue(torch.isfinite(torch.tensor(report['generation'])))
            opt.step()
        paths = [
            'enhancement.physical_read.attention.out_proj.weight',
            'enhancement.semantic_attention.attention.out_proj.weight',
            'enhancement.route_delta.weight', 'enhancement.execute_delta.weight',
            'state_experts.0.condition_attention.v.weight',
            'state_experts.0.relation_attention.q.weight',
            'state_experts.0.temporal_attention.q.weight',
            'state_experts.0.ffn.layers.0.weight',
            'state_experts.0.noise_modulation.weight',
            'mechanisms.0.video_query.weight', 'mechanisms.0.content_key.weight',
            'mechanisms.0.video_value.weight', 'mechanisms.0.read_output.weight',
            'mechanisms.0.feature_output.weight']
        named = dict(m.named_parameters())
        for name in paths:
            self.assertIsNotNone(named[name].grad, name)
            self.assertGreater(named[name].grad.norm().item(), 0., name)
        owned = [id(p) for group in opt.param_groups for p in group['params']]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertTrue(all(p.grad is None for b in m.wan.blocks
                            for p in b.cross_attn.parameters()))
        self.assertFalse(any('cross_attn' in n for n in m.learned_names
                             if n.startswith('wan.')))
        m.base_bank.assert_frozen()

    def test_original_text_read_once_and_no_future_input(self):
        m = model()
        ev, sample = inputs()
        seen = []
        with torch.no_grad():
            for branch in m.mechanisms.values():
                branch.feature_output.weight.normal_(0., .01)
        handle = m.wan.blocks[0].cross_attn.q.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone()))
        c, g = m.prepare_content(ev, sample['text'])
        args = (sample['latent'], None, sample['text'], torch.tensor(.5),
                c, g, sample['times'])
        try:
            m.checkpointing = False
            m.velocity(*args)
            m.velocity(*args, physical_mode='off')
        finally:
            handle.remove()
        self.assertEqual(len(seen), 2)
        torch.testing.assert_close(seen[0], seen[1], atol=0, rtol=0)
        changed = list(args)
        changed[1] = sample['latent'][:, :, :1]
        with self.assertRaisesRegex(ValueError, 'external first frame'):
            m.velocity(*changed)

    def test_module_interventions_have_actual_effects(self):
        m = model()
        ev, sample = inputs()
        with torch.no_grad():
            for branch in m.mechanisms.values():
                branch.feature_output.weight.normal_(0., .05)
        c, g = m.prepare_content(ev, sample['text'])
        args = (sample['latent'], None, sample['text'], torch.tensor(.5),
                c, g, sample['times'])
        normal = m.velocity(*args)[0]
        for mode in ('off', 'no_video_feedback', 'no_temporal'):
            self.assertGreater(
                (m.velocity(*args, physical_mode=mode)[0] - normal).norm().item(), 1e-6)
        nosemantic, _ = m.prepare_content(ev, sample['text'], semantic_expert=False)
        self.assertGreater((c['routing'] - nosemantic['routing']).norm().item(), 1e-6)

    def test_checkpoint_preserves_per_site_parameters_and_condition_constants(self):
        from physical_content.checkpointing import save_checkpoint, restore_checkpoint
        torch.manual_seed(77)
        m = model()
        opt = torch.optim.AdamW(m.optim_groups())
        ev, sample = inputs()
        backward_micro(m, ev, sample, sample['text'], torch.tensor(.5),
                       torch.randn_like(sample['latent']), 0., 1)
        opt.step()
        with tempfile.TemporaryDirectory() as folder:
            path = save_checkpoint(m, opt, m.cfg, folder, 1, 4)
            torch.manual_seed(77)
            restored = model()
            optimizer = torch.optim.AdamW(restored.optim_groups())
            update, cursor, _ = restore_checkpoint(restored, optimizer, path, restored.cfg)
            self.assertEqual((update, cursor), (1, 4))
            actual = dict(restored.named_parameters())
            for name, p in m.named_parameters():
                if name in m.learned_names:
                    torch.testing.assert_close(p, actual[name])
            self.assertIsNot(restored.mechanisms['0'].video_value,
                             restored.mechanisms['2'].video_value)
            self.assertIsNot(restored.state_experts['0'], restored.state_experts['2'])
            torch.testing.assert_close(m.condition_mean, restored.condition_mean)


if __name__ == '__main__':
    unittest.main()
