"""CPU integration checks for conditional P0, recurrent insertion and gradient boundaries."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
import yaml

from physgen_v4.backbone import ProcessWan
from physgen_v4.coordinates import Coordinates
from physgen_v4.corrector import Corrector
from physgen_v4.losses import prior_distance, process_losses, training_loss

torch.set_num_threads(2)
CONFIG = yaml.safe_load((ROOT / 'configs/wisa_native_p.yaml').read_text())


class Expand(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, x):
        return x.mean(-1, keepdim=True).expand(*x.shape[:-1], self.width)


class FrozenBlock(nn.Module):
    def forward(self, x, **kwargs):
        return (0.97 * x + 0.01).tanh()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3072, 48 * 4)

    def forward(self, x, condition):
        return self.projection(x)


class TinyWan(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim, self.freq_dim, self.text_len = 3072, 4, 8
        self.patch_embedding = nn.Conv3d(48, 3072, (1, 2, 2), stride=(1, 2, 2))
        self.time_embedding, self.time_projection = Expand(3072), Expand(6 * 3072)
        self.text_embedding = nn.Linear(4096, 3072)
        self.freqs = torch.empty(0)
        self.blocks = nn.ModuleList([FrozenBlock() for _ in range(30)])
        self.head = Head()
        self.requires_grad_(False)

    def unpatchify(self, hidden, grids):
        t, h, w = (int(v) for v in grids[0])
        video = hidden.reshape(1, t, h, w, 1, 2, 2, 48)
        return [video.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(1, 48, t, h * 2, w * 2)[0]]


class PriorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(119)
        cls.config = copy.deepcopy(CONFIG)
        cls.config['corrector'].update(prior_width=64, prior_heads=4, prior_depth=1, checkpoint_cell=False)
        cls.corrector = Corrector(cls.config['corrector'])
        cls.wan = TinyWan()
        cls.coords = Coordinates.build(torch.linspace(0, 2, 9), 64, 64, 4, 1, teacher_size=64)
        cls.first = torch.randn(1, 48, 1, 4, 4)
        cls.text = torch.randn(1, 4, 4096)
        cls.clean = torch.randn(1, 48, 3, 4, 4)
        cls.noise = torch.randn_like(cls.clean)
        cls.target = torch.randn(1, 32, 1664)
        cls.weight = torch.ones(1, 32)

    def setUp(self):
        self.corrector.zero_grad(set_to_none=True)
        self.corrector.set_stage('A')
        self.corrector.config['checkpoint_cell'] = False
        self.corrector.config['block_b'] = 18
        with torch.no_grad():
            for unit in self.corrector.A.values():
                nn.init.normal_(unit.site.writer.weight, std=.001)
            nn.init.normal_(self.corrector.B.writer.weight, std=.001)
            self.corrector.B.b.fill_(.1)
        self.model = ProcessWan(self.wan, self.corrector, True)
        module = SimpleNamespace(sinusoidal_embedding_1d=lambda width, t: t[:, None].expand(-1, width))
        self.module_patch = patch.dict(sys.modules, {'wan.modules.model': module})
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

    def run_model(self, sigma=.4, **kwargs):
        s = torch.tensor(sigma)
        noisy = torch.cat((self.first, ((1 - s) * self.clean + s * self.noise)[:, :, 1:]), 2)
        return self.model(noisy, self.first, self.text, s, self.coords, **kwargs)

    def test_all_six_a_locations_commit_distinct_states_and_replay_exact_suffix(self):
        with torch.no_grad():
            full, final, info = self.run_model(capture='all')
            expected = ['A@5', 'A@10', 'A@15', 'B@18', 'A@20', 'A@25', 'A@30']
            self.assertEqual(list(info['states']), expected)
            self.assertIs(final, info['states']['A@30'])
            for label in expected:
                replay = self.model.suffix(info['snapshots'][label], self.text, torch.tensor(.4),
                                          self.coords, 'on', label, downstream=True)
                torch.testing.assert_close(replay, full, rtol=0, atol=0)
            for before, after in zip(expected, expected[1:]):
                torch.testing.assert_close(info['before'][after], info['states'][before], rtol=0, atol=0)

    def test_a1_uses_every_fifth_block_without_enabling_b(self):
        self.model.enable_b = False
        with torch.no_grad():
            _, _, info = self.run_model()
        self.assertEqual(list(info['states']), [f'A@{i}' for i in range(5, 31, 5)])

    def test_per_call_writer_ablation_preserves_its_state_and_changes_output(self):
        with torch.no_grad():
            full, _, original = self.run_model()
            muted, _, changed = self.run_model(writer_off=('A@10',))
            torch.testing.assert_close(original['states']['A@10'], changed['states']['A@10'], rtol=0, atol=0)
            self.assertGreater(float((full - muted).abs().max()), 0)
            baseline, _, _ = self.run_model(disable=('A', 'B'))
            writers_off, _, _ = self.run_model(writer_off=('A', 'B'))
            torch.testing.assert_close(baseline, writers_off, rtol=0, atol=0)

    def test_overlapping_a_b_locations_replay_in_order(self):
        self.corrector.config['block_b'] = 10
        with torch.no_grad():
            full, _, info = self.run_model(capture='all')
            for label in ('A@10', 'B@10'):
                replay = self.model.suffix(info['snapshots'][label], self.text, torch.tensor(.4),
                                          self.coords, 'on', label)
                torch.testing.assert_close(replay, full, rtol=0, atol=0)

    def test_fm_alone_trains_initializer_through_frozen_wan_when_writer_is_open(self):
        cfg = copy.deepcopy(self.config)
        cfg['loss'].update(struct_weight=0, prior_weight=0)
        prediction, _, info = self.run_model()
        loss, _ = training_loss(prediction, self.noise, self.clean, self.first, self.noise,
            torch.tensor(.4), info, self.target, self.weight, 200, cfg, False)
        loss.backward()
        self.assertGreater(float(self.corrector.prior.initialize.output[-1].weight.grad.norm()), 0)
        for unit in self.corrector.A.values():
            self.assertGreater(float(unit.site.writer.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in self.wan.parameters()))

    def test_later_state_loss_reaches_earlier_independent_writer(self):
        self.model.enable_b = False
        with torch.no_grad():
            for unit in self.corrector.A.values():
                unit.site.writer.weight.zero_()
        _, _, info = self.run_model()
        first_loss = (info['states']['A@5'] - self.target).square().mean()
        first_grad = torch.autograd.grad(first_loss, self.corrector.A["5"].site.writer.weight,
                                         allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(first_grad)
        last_loss = (info['states']['A@30'] - self.target).square().mean()
        last_loss.backward()
        self.assertGreater(float(self.corrector.A["5"].site.writer.weight.grad.norm()), 0)
        # A@30 writes after this state is committed, so its writer is outside this loss.
        self.assertIsNone(self.corrector.A["30"].site.writer.weight.grad)
        self.assertTrue(all(p.grad is None for p in self.wan.parameters()))

    def test_warm_prior_supervision_trains_initializer_despite_detached_carried_state(self):
        with torch.no_grad():
            _, carried, _ = self.run_model(sigma=.9)
        prediction, _, info = self.run_model(process=carried.detach())
        self.assertIsNone(info['prior'])
        info['prior'] = self.corrector.initialize(self.first, self.text, self.coords, gt_video=self.clean)
        loss, metrics = training_loss(prediction, self.noise, self.clean, self.first, self.noise,
            torch.tensor(.4), info, self.target, self.weight, 200, self.config, False)
        loss.backward()
        self.assertEqual(metrics['prior/active'], 1)
        self.assertTrue(all(p.grad is not None for p in self.corrector.parameter_groups()['initialization']))
        self.assertTrue(all(p.grad is not None for ps in self.corrector.parameter_groups().values() for p in ps))

    def test_warm_ab_without_prior_or_b_has_all_six_a_groups_ready_for_sync(self):
        self.corrector.set_stage('AB')
        with torch.no_grad():
            _, carried, _ = self.run_model(sigma=.98)
        prediction, _, info = self.run_model(sigma=.9, process=carried.detach())
        (prediction.square().mean() + sum(p.square().mean() for p in info['states'].values())).backward()
        groups = self.corrector.parameter_groups()
        self.assertTrue(all(p.grad is None for name in ('initialization', 'B') for p in groups[name]))
        for n in (5, 10, 15, 20, 25, 30):
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in groups[f'A@{n}']))

    def test_bf16_warm_checkpoint_trains_prior_and_all_six_independent_a_blocks(self):
        self.model.enable_b = False
        self.corrector.config['checkpoint_cell'] = True
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            with torch.no_grad():
                _, carried, _ = self.run_model(sigma=.9)
            prediction, _, info = self.run_model(process=carried.detach())
            info['prior'] = self.corrector.initialize(self.first, self.text, self.coords, gt_video=self.clean)
            loss, metrics = training_loss(prediction, self.noise, self.clean, self.first, self.noise,
                torch.tensor(.4), info, self.target, self.weight, 200, self.config, False)
        loss.backward()
        self.assertEqual(metrics['prior/active'], 1)
        groups = self.corrector.parameter_groups()
        self.assertEqual(list(groups), ['initialization'] + [f'A@{n}' for n in range(5, 31, 5)])
        for name, parameters in groups.items():
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters), name)
            self.assertTrue(any(bool(p.grad.count_nonzero()) for p in parameters), name)
        self.assertTrue(all(p.grad is None for p in self.wan.parameters()))
        self.assertTrue(all(p.grad is None for p in self.corrector.B_core.parameters()))
        self.assertTrue(all(p.grad is None for p in self.corrector.B.parameters()))

    def test_prior_checkpoint_recomputation_preserves_parameter_gradients(self):
        first = self.corrector.initialize(self.first, self.text, self.coords)
        prior_distance(first, self.target, self.weight, 2).backward()
        reference = self.corrector.prior.initialize.image.weight.grad.clone()
        self.corrector.zero_grad(set_to_none=True)
        self.corrector.config['checkpoint_cell'] = True
        repeated = self.corrector.initialize(self.first, self.text, self.coords)
        prior_distance(repeated, self.target, self.weight, 2).backward()
        torch.testing.assert_close(first, repeated)
        torch.testing.assert_close(reference, self.corrector.prior.initialize.image.weight.grad)

    def test_early_attenuation_changes_writing_but_not_state_update(self):
        with torch.no_grad():
            prior = self.corrector.initialize(self.first, self.text, self.coords)
            hidden = torch.randn(1, 12, 3072)
            output, state, _, metrics = self.corrector('A', hidden, prior, self.text, torch.tensor(1.), self.coords, block=5)
            width = self.corrector.config['early_write_width']
            try:
                self.corrector.config['early_write_width'] = 0
                unrestricted, reference, _, _ = self.corrector('A', hidden, prior, self.text, torch.tensor(1.), self.coords, block=5)
            finally:
                self.corrector.config['early_write_width'] = width
        torch.testing.assert_close(state, reference, rtol=0, atol=0)
        torch.testing.assert_close(output - hidden, .1 * (unrestricted - hidden), rtol=1e-3, atol=3e-7)
        self.assertAlmostEqual(float(metrics['early_write_scale']), .1)

    def test_old_weights_transfer_only_explicitly_replaces_initializer(self):
        prefix = 'prior.initialize.'
        weights = {name: value for name, value in self.corrector.state_dict().items()
                   if not name.startswith((prefix, 'prior.text.'))}
        weights[prefix + 'weight'] = torch.zeros(1664, 97)
        weights[prefix + 'bias'] = torch.zeros(1664)
        original = self.corrector.prior.initialize.image.weight.detach().clone()
        source = dict(self.corrector.config, initialization='latent')
        with self.assertRaises(RuntimeError):
            self.corrector.load_checkpoint_weights(weights, source, allow_new_initializer=False)
        report = self.corrector.load_checkpoint_weights(weights, source, allow_new_initializer=True)
        self.assertIn('new image_text', report['initializer'])
        torch.testing.assert_close(self.corrector.prior.initialize.image.weight, original, rtol=0, atol=0)
        incomplete = dict(weights)
        incomplete.pop('A.10.site.writer.weight')
        with self.assertRaises(RuntimeError):
            self.corrector.load_checkpoint_weights(incomplete, source, allow_new_initializer=True)

    def test_gt_and_no_gt_both_train_p0_through_fm_and_six_independent_correctors(self):
        self.model.enable_b = False
        cfg = copy.deepcopy(self.config)
        cfg['loss'].update(struct_weight=0, prior_weight=0)
        for gt in (self.clean, None):
            self.corrector.zero_grad(set_to_none=True)
            prediction, _, info = self.run_model(gt_video=gt)
            loss, _ = training_loss(prediction, self.noise, self.clean, self.first, self.noise,
                                   torch.tensor(.4), info, self.target, self.weight, 200, cfg, False)
            loss.backward()
            self.assertEqual(info['prior_gt_conditioned'], gt is not None)
            self.assertGreater(float(self.corrector.prior.initialize.image.weight.grad.norm()), 0)
            self.assertGreater(float(self.corrector.prior.text.weight.grad.norm()), 0)
            if gt is None:
                self.assertTrue(all(p.grad is None for p in self.corrector.gt_condition_parameters()))
            else:
                self.assertGreater(float(self.corrector.prior.initialize.video.weight.grad.norm()), 0)
            for n in (5, 10, 15, 20, 25, 30):
                self.assertGreater(float(self.corrector.A[str(n)].site.writer.weight.grad.norm()), 0)
            self.assertTrue(all(p.grad is None for p in self.wan.parameters()))

    def test_inference_carries_p_within_each_cfg_branch_and_resets_for_new_video(self):
        self.model.enable_b = False
        self.corrector.eval()
        identities = {n: [id(p) for p in unit.parameters()] for n, unit in self.corrector.A.items()}
        try:
            with torch.no_grad(), patch.object(self.corrector, 'initialize', wraps=self.corrector.initialize) as init:
                states = dict(cond=None, uncond=None)
                latent = torch.cat((self.first, self.noise[:, :, 1:]), 2)
                first_p = None
                for sigma, next_sigma in ((1., .7), (.7, .3), (.3, 0.)):
                    outputs = {}
                    for branch, text in (('cond', self.text), ('uncond', self.text * 0)):
                        previous = states[branch]
                        outputs[branch], states[branch], info = self.model(latent, self.first, text,
                            torch.tensor(sigma), self.coords, previous)
                        if previous is not None:
                            torch.testing.assert_close(info['initial'], previous, rtol=0, atol=0)
                            self.assertIsNone(info['prior'])
                        elif branch == 'cond':
                            first_p = info['prior'].clone()
                        self.assertTrue(bool(torch.isfinite(states[branch]).all()))
                    self.assertFalse(torch.equal(states['cond'], states['uncond']))
                    latent = latent + (next_sigma - sigma) * (outputs['uncond'] + 5 * (outputs['cond'] - outputs['uncond']))
                    latent = torch.cat((self.first, latent[:, :, 1:]), 2)
                    self.assertTrue(bool(torch.isfinite(latent).all()))
                self.assertEqual(init.call_count, 2)
                _, _, restarted = self.run_model(sigma=1.)
                self.assertEqual(init.call_count, 3)
                torch.testing.assert_close(restarted['prior'], first_p, rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'carrying'):
                    self.run_model(process=states['cond'], gt_video=self.clean)
            for n, unit in self.corrector.A.items():
                self.assertEqual([id(p) for p in unit.parameters()], identities[n])
        finally:
            self.corrector.train()


class PriorLossTests(unittest.TestCase):
    def test_pooling_respects_content_mask_and_preserves_time_order(self):
        target = torch.tensor([[[1.], [3.], [999.], [5.], [7.], [999.]]])
        prior = target.clone().requires_grad_()
        weight = torch.tensor([[1., 1., 0., 1., 1., 0.]])
        self.assertEqual(float(prior_distance(prior, target, weight, 2).detach()), 0)
        reversed_time = prior.reshape(1, 2, 3, 1).flip(1).reshape_as(prior)
        self.assertEqual(float(prior_distance(reversed_time, target, weight, 2).detach()), 16.)
        loss = prior_distance(prior, torch.zeros_like(target), weight, 2)
        loss.backward()
        torch.testing.assert_close(prior.grad[:, [2, 5]], torch.zeros(1, 2, 1))

    def test_prior_loss_remains_active_at_pure_noise_and_does_not_train_teacher(self):
        prior = torch.ones(1, 4, 3, requires_grad=True)
        target = torch.zeros_like(prior, requires_grad=True)
        info = dict(states={'A': prior}, before={'A': prior.detach()}, initial=prior.detach(),
                    prior=prior, prior_time_tokens=2, metrics={})
        z = torch.zeros(1, 3, 2, 2, 2)
        cfg = copy.deepcopy(CONFIG)
        cfg['loss']['fm_weight'] = 0
        loss, metrics = training_loss(z, z, z, z[:, :, :1], z, torch.tensor(1.), info, target,
                                     torch.ones(1, 4), 200, cfg, False)
        torch.testing.assert_close(loss, torch.tensor(.02))
        loss.backward()
        self.assertGreater(float(prior.grad.norm()), 0)
        # Existing state loss also references target, with an exactly zero high-noise coefficient.
        self.assertTrue(target.grad is None or float(target.grad.norm()) == 0)
        self.assertEqual(float(metrics['loss/weighted_struct']), 0)

    def test_repeated_a_losses_do_not_dilute_the_b_family(self):
        a = torch.ones(1, 2, 3)
        b = 3 * a
        states = {f'A@{i}': a for i in range(5, 31, 5)}
        states['B@18'] = b
        loss, _ = process_losses(dict(states=states, before=states, initial=a, metrics={}),
                                 torch.zeros_like(a), torch.ones(1, 2))
        torch.testing.assert_close(loss, torch.tensor(5.))


if __name__ == '__main__':
    unittest.main(verbosity=2)
