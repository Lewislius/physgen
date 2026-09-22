"""Native-width CPU checks of block ownership, updates, migration and checkpoint I/O."""
import copy
import gc
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from physgen_v4.backbone import ProcessWan
from physgen_v4.coordinates import Coordinates
from physgen_v4.corrector import Corrector
from physgen_v4.runtime import group_metrics, make_scheduler, read_config, save_checkpoint

torch.set_num_threads(2)
CONFIG = yaml.safe_load((ROOT / 'configs/wisa_native_p.yaml').read_text())
BLOCKS = [5, 10, 15, 20, 25, 30]


class IndependentCorrectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(512)
        cls.config = copy.deepcopy(CONFIG['corrector'])
        cls.config.update(prior_width=64, prior_heads=4, prior_depth=1, checkpoint_cell=False)
        cls.model = Corrector(cls.config)
        cls.coords = Coordinates.build(torch.linspace(0, 2, 9), 32, 64, 4, 1, teacher_size=64)
        cls.hidden = torch.randn(1, 6, 3072)
        cls.process = torch.randn(1, 32, 1664)
        cls.text = torch.randn(1, 3, 4096)

    @classmethod
    def tearDownClass(cls):
        del cls.model
        gc.collect()

    def setUp(self):
        self.model.zero_grad(set_to_none=True)
        self.model.set_stage('A')
        self.model.config.update(iterations=1, checkpoint_cell=False)

    def call(self, block, sigma=.4):
        return self.model('A', self.hidden, self.process, self.text, torch.tensor(sigma), self.coords, block=block)

    def assert_independent(self, model):
        owners = {'prior': model.prior, **{f'A@{n}': model.A[str(n)] for n in BLOCKS},
                  'B_core': model.B_core, 'B': model.B}
        ids, storage, modules = set(), set(), set()
        for name, module in owners.items():
            with self.subTest(owner=name):
                parameters = list(module.named_parameters(remove_duplicate=False))
                current_ids = {id(p) for _, p in parameters}
                current_storage = {p.untyped_storage().data_ptr() for _, p in parameters}
                current_modules = {id(m) for m in module.modules()}
                self.assertEqual(len(parameters), len(current_ids))
                self.assertEqual(len(parameters), len(current_storage))
                self.assertFalse(ids & current_ids)
                self.assertFalse(storage & current_storage)
                self.assertFalse(modules & current_modules)
                ids.update(current_ids)
                storage.update(current_storage)
                modules.update(current_modules)
        self.assertEqual(len(list(model.named_parameters(remove_duplicate=False))), len(ids))

    def test_every_parameter_and_module_has_one_block_owner(self):
        self.assert_independent(self.model)
        self.assertFalse(hasattr(self.model, 'shared'))
        self.assertFalse(torch.equal(self.model.A['5'].core.text.weight, self.model.A['10'].core.text.weight))

    def test_stage_groups_cover_trainable_parameters_once(self):
        expected = {'A': ['initialization'] + [f'A@{n}' for n in BLOCKS],
                    'B': ['B'], 'AB': ['initialization'] + [f'A@{n}' for n in BLOCKS] + ['B']}
        for stage, names in expected.items():
            self.model.set_stage(stage)
            groups = self.model.parameter_groups()
            self.assertEqual(list(groups), names)
            ids = [id(p) for values in groups.values() for p in values]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), {id(p) for p in self.model.parameters() if p.requires_grad})

    def test_default_native_parameter_budget(self):
        with torch.device('meta'):
            model = Corrector(CONFIG['corrector'])
        expected = {'A': 837957970, 'B': 136224516, 'AB': 974182486}
        for stage, count in expected.items():
            model.set_stage(stage)
            self.assertEqual(sum(p.numel() for ps in model.parameter_groups().values() for p in ps), count)

    def test_invalid_or_missing_block_never_falls_back_to_shared_a(self):
        for block in (None, 6, 0, '5', True):
            with self.subTest(block=block), self.assertRaises(ValueError):
                self.call(block)
        with self.assertRaises(ValueError):
            self.model.modules_for('C', 5)

    def test_layout_validation_rejects_duplicates_and_wrong_wan_depth(self):
        for blocks in ([5, 5], [10, 5], [], [0, 5], [5.0], [5, 15]):
            with self.subTest(blocks=blocks), torch.device('meta'), self.assertRaises(ValueError):
                Corrector(dict(self.config, a_blocks=blocks))
        from types import SimpleNamespace
        with self.assertRaisesRegex(ValueError, 'Wan depth'):
            ProcessWan(SimpleNamespace(blocks=[None] * 29), self.model, False)

    def test_same_block_reuses_its_parameters_across_sigmas(self):
        unit = self.model.A['10']
        identities = [id(p) for p in unit.parameters()]
        with torch.no_grad():
            first = self.call(10, .9)
            second = self.call(10, .2)
            repeated = self.call(10, .9)
        self.assertEqual(identities, [id(p) for p in unit.parameters()])
        torch.testing.assert_close(first[0], repeated[0], rtol=0, atol=0)
        torch.testing.assert_close(first[1], repeated[1], rtol=0, atol=0)
        self.assertGreater(float((first[1] - second[1]).abs().max()), 0)

    def test_a10_backward_and_adam_update_leave_other_blocks_unchanged(self):
        groups = self.model.parameter_groups()
        optimizer = torch.optim.AdamW([dict(name=n, params=ps) for n, ps in groups.items()], lr=1e-3, foreach=False)
        before = {n: p.detach().clone() for n, p in self.model.A['5'].named_parameters()}
        versions = {n: p._version for n, p in self.model.named_parameters() if not n.startswith('A.10.')}
        writer = self.model.A['10'].site.writer.weight
        initial_writer = writer.detach().clone()
        hidden, state, _, _ = self.call(10)
        (hidden.square().mean() + state.square().mean()).backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in groups['A@10']))
        self.assertTrue(all(p.grad is None for n, ps in groups.items() if n != 'A@10' for p in ps))
        self.assertGreater(float(writer.grad.norm()), 0)
        metrics = group_metrics({'A@10': groups['A@10']})
        self.assertGreater(float(metrics['grad/A@10_norm']), 0)
        self.assertTrue(all(bool(torch.isfinite(v)) for v in metrics.values()))
        optimizer.step()
        self.assertGreater(float((writer - initial_writer).detach().abs().max()), 0)
        for n, p in self.model.A['5'].named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        for n, p in self.model.named_parameters():
            if n in versions:
                self.assertEqual(p._version, versions[n], n)
        self.assertEqual({id(p) for p in optimizer.state}, {id(p) for p in groups['A@10']})

    def test_b_training_freezes_all_a_and_prior_but_trains_its_own_core(self):
        self.model.set_stage('B')
        with torch.no_grad():
            self.model.B.b.fill_(.1)
            torch.nn.init.normal_(self.model.B.writer.weight, std=.001)
        h, p, _, _ = self.model('B', self.hidden, self.process, self.text, torch.tensor(.4), self.coords, block=18)
        (h.square().mean() + p.square().mean()).backward()
        self.assertTrue(all(p.grad is None for p in self.model.A.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.prior.parameters()))
        self.assertGreater(float(self.model.B_core.read_attention.attention.value.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is not None for p in self.model.parameter_groups()['B']))

    def test_two_iteration_recomputation_preserves_block_local_gradients(self):
        self.model.config['iterations'] = 2
        with torch.no_grad():
            torch.nn.init.normal_(self.model.A['25'].site.writer.weight, std=.001)
        selected = [self.model.A['25'].core.read_attention.attention.query.weight,
                    self.model.A['25'].core.write_attention.attention.value.weight,
                    self.model.A['25'].site.read.weight]
        h, p, _, _ = self.call(25)
        (h.square().mean() + p.square().mean()).backward()
        reference = [v.grad.clone() for v in selected]
        self.model.zero_grad(set_to_none=True)
        self.model.config['checkpoint_cell'] = True
        rh, rp, _, _ = self.call(25)
        (rh.square().mean() + rp.square().mean()).backward()
        torch.testing.assert_close(h, rh)
        torch.testing.assert_close(p, rp)
        for v, expected in zip(selected, reference):
            torch.testing.assert_close(v.grad, expected)
        self.assertTrue(all(p.grad is None for n, u in self.model.A.items() if n != '25' for p in u.parameters()))

    def test_full_checkpoint_and_optimizer_round_trip_preserves_block_ownership(self):
        config = copy.deepcopy(CONFIG)
        config['corrector'] = copy.deepcopy(self.config)
        config.update(stage='A', run_name='independent_checkpoint_test')
        groups = self.model.parameter_groups()
        optimizer = torch.optim.AdamW([dict(name=n, params=ps) for n, ps in groups.items()], lr=1e-4, foreach=False)
        scheduler = make_scheduler(optimizer, config)
        # Allocate an actual, distinct Adam state for each block without a multi-GB backward graph.
        with torch.no_grad():
            for index, n in enumerate(BLOCKS):
                parameter = self.model.A[str(n)].site.write_gate.bias
                parameter.fill_(index / 10)
                parameter.grad = torch.full_like(parameter, (index + 1) / 10)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        with tempfile.TemporaryDirectory(prefix='v4-independent-checkpoint-') as directory:
            config['paths'].update(checkpoint_root=directory, cache_root=directory)
            (Path(directory) / 'manifest.json').write_text('{}')
            with patch('physgen_v4.runtime.dist.get_rank', return_value=0), \
                    patch('physgen_v4.runtime.dist.get_world_size', return_value=1), \
                    patch('physgen_v4.runtime.dist.barrier'), patch('physgen_v4.runtime.rng_state', return_value={}):
                destination = save_checkpoint(config, self.model, optimizer, scheduler, 1, 8, {})
            saved_config = read_config(destination / 'config.yaml')
            with torch.device('meta'):
                restored = Corrector(saved_config['corrector'])
            restored.to_empty(device='cpu')
            weights = torch.load(destination / 'corrector.pt', map_location='cpu', weights_only=True, mmap=True)
            restored.load_checkpoint_weights(weights, saved_config['corrector'])
            self.assert_independent(restored)
            restored_weights = restored.state_dict()
            for name, value in self.model.state_dict().items():
                torch.testing.assert_close(restored_weights[name], value, rtol=0, atol=0)
            restored.set_stage('A')
            restored_optimizer = torch.optim.AdamW([dict(name=n, params=ps) for n, ps in restored.parameter_groups().items()],
                                                  lr=1e-4, foreach=False)
            restored_scheduler = make_scheduler(restored_optimizer, saved_config)
            training = torch.load(destination / 'training.pt', weights_only=True)
            restored_optimizer.load_state_dict(training['optimizer'])
            restored_scheduler.load_state_dict(training['scheduler'])
            self.assertEqual(training['data_cursor'], 8)
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
            for n in BLOCKS:
                original = self.model.A[str(n)].site.write_gate.bias
                copied = restored.A[str(n)].site.write_gate.bias
                for key in ('step', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(optimizer.state[original][key], restored_optimizer.state[copied][key])
            # Independent inference after strict loading reproduces each trained block's outputs.
            with torch.no_grad():
                for n in BLOCKS:
                    expected = self.call(n)
                    actual = restored('A', self.hidden, self.process, self.text, torch.tensor(.4), self.coords, block=n)
                    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_legacy_shared_weights_split_into_equal_values_with_separate_storage(self):
        source_config = dict(self.config)
        source_config.pop('parameter_sharing')
        source_config.pop('a_blocks')
        legacy = Corrector(source_config)
        self.assertEqual(legacy.parameter_sharing, 'shared')
        weights = legacy.state_dict()
        with self.assertRaisesRegex(RuntimeError, 'INIT_FROM'):
            self.model.load_checkpoint_weights(weights, source_config)
        report = self.model.load_checkpoint_weights(weights, source_config, allow_new_parameter_sharing=True)
        self.assertEqual(report['parameter_sharing'], 'split into independent copies')
        self.assert_independent(self.model)
        for n in BLOCKS:
            for key, value in legacy.A.state_dict().items():
                torch.testing.assert_close(self.model.A[str(n)].site.state_dict()[key], value, rtol=0, atol=0)
            for key, value in legacy.shared.state_dict().items():
                if not key.startswith('initialize.'):
                    torch.testing.assert_close(self.model.A[str(n)].core.state_dict()[key], value, rtol=0, atol=0)
            with torch.no_grad():
                expected = legacy('A', self.hidden, self.process, self.text, torch.tensor(.4), self.coords, block=n)
                actual = self.call(n)
                torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
                torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)

    def test_incomplete_checkpoint_is_rejected_before_any_parameter_changes(self):
        weights = dict(self.model.state_dict())
        weights.pop('A.20.core.text.weight')
        weights['A.5.site.write_gate.bias'] = weights['A.5.site.write_gate.bias'] + 10
        before = self.model.A['5'].site.write_gate.bias.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            self.model.load_checkpoint_weights(weights, self.config)
        torch.testing.assert_close(self.model.A['5'].site.write_gate.bias, before, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'layout mismatch'):
            self.model.load_checkpoint_weights(self.model.state_dict(), dict(self.config, block_b=17))

    def test_split_can_migrate_legacy_initializer_and_fusion_together(self):
        source = dict(self.config, parameter_sharing='shared', initialization='latent', fusion='interpolate')
        legacy = Corrector(source)
        original = self.model.prior.initialize.image.weight.detach().clone()
        attentions = {n: self.model.A[str(n)].core.read_attention.attention.query.weight.detach().clone() for n in BLOCKS}
        report = self.model.load_checkpoint_weights(legacy.state_dict(), source, allow_new_parameter_sharing=True,
                                                    allow_new_initializer=True, allow_new_fusion=True)
        self.assertEqual(report['initializer'], 'new image_text_gt prior')
        self.assertEqual(report['fusion'], 'new cross_attention')
        torch.testing.assert_close(original, self.model.prior.initialize.image.weight, rtol=0, atol=0)
        for n in BLOCKS:
            torch.testing.assert_close(attentions[n], self.model.A[str(n)].core.read_attention.attention.query.weight,
                                       rtol=0, atol=0)
            self.assertEqual(float(self.model.A[str(n)].site.writer.weight.detach().norm()), 0)
        self.assertEqual(float(self.model.B.writer.weight.detach().norm()), 0)
        self.assertEqual(float(self.model.B.b.detach()), 0)
        self.assert_independent(self.model)

    def test_image_text_checkpoint_adds_only_gt_layers_with_explicit_migration(self):
        from physgen_v4.corrector import ConditionPrior
        source = dict(self.config, initialization='image_text')
        legacy_prior = ConditionPrior(source)
        prefix = 'prior.initialize.'
        weights = {name: value for name, value in self.model.state_dict().items() if not name.startswith(prefix)}
        weights.update({prefix + name: value for name, value in legacy_prior.state_dict().items()})
        first = torch.randn(1, 48, 1, 2, 4)
        with torch.no_grad():
            expected = legacy_prior(first, self.model.prior.text(self.text), self.coords, False)
        original_gt = [p.detach().clone() for p in self.model.gt_condition_parameters()]
        original_writer = self.model.A['10'].site.writer.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, 'INIT_FROM'):
            self.model.load_checkpoint_weights(weights, source)
        report = self.model.load_checkpoint_weights(weights, source, allow_new_initializer=True)
        self.assertEqual(report['initializer'], 'added GT video condition')
        with torch.no_grad():
            actual = self.model.initialize(first, self.text, self.coords)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.model.A['10'].site.writer.weight, original_writer, rtol=0, atol=0)
        for value, original in zip(self.model.gt_condition_parameters(), original_gt):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            self.model.load_checkpoint_weights(weights, self.config)


if __name__ == '__main__':
    unittest.main(verbosity=2)
