"""CPU checks of unequal-grid P/H attention, addressing, gradients and checkpoint migration."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml

from physgen_v4.coordinates import Coordinates, resample_tokens, sinusoid
from physgen_v4.corrector import Corrector

torch.set_num_threads(2)
CONFIG = yaml.safe_load((ROOT / 'configs/wisa_native_p.yaml').read_text())['corrector']
NEW_PREFIXES = ('shared.read_attention.', 'shared.write_attention.', 'shared.read_gate.')


class CrossAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(217)
        cfg = copy.deepcopy(CONFIG)
        cfg.update(parameter_sharing="shared", initialization="image_text", prior_width=64,
                   prior_heads=4, prior_depth=1, checkpoint_cell=False)
        cls.model = Corrector(cfg)
        cls.coords = Coordinates.build(torch.linspace(0, 2, 9), 32, 64, 4, 1, teacher_size=64)
        cls.hidden = torch.randn(1, 6, 3072)
        cls.process = torch.randn(1, 32, 1664)
        cls.text = torch.randn(1, 3, 4096)
        cls.sigma = torch.tensor(.4)

    def setUp(self):
        self.model.zero_grad(set_to_none=True)
        self.model.set_stage('A')
        self.model.config.update(iterations=1, checkpoint_cell=False)
        with torch.no_grad():
            torch.nn.init.normal_(self.model.A.writer.weight, std=.001)

    def call(self):
        return self.model('A', self.hidden, self.process, self.text, self.sigma, self.coords, block=5)

    def test_unequal_grid_read_and_write_never_interpolate(self):
        shapes = {}
        def record(name):
            return lambda module, inputs: shapes.update({name: [tuple(x.shape) for x in inputs]})
        read_hook = self.model.shared.read_attention.attention.register_forward_pre_hook(record('read'))
        write_hook = self.model.shared.write_attention.attention.register_forward_pre_hook(record('write'))
        try:
            with torch.no_grad(), patch('physgen_v4.corrector.resample_tokens', side_effect=AssertionError('interpolation')):
                initial = self.model.initialize(torch.randn(1, 48, 1, 2, 4), self.text, self.coords)
                hidden, process, _, _ = self.call()
            self.assertEqual(initial.shape, self.process.shape)
            self.assertEqual(hidden.shape, self.hidden.shape)
            self.assertEqual(process.shape, self.process.shape)
            self.assertEqual(shapes['read'], [(1, 32, 1664), (1, 6, 1664), (1, 6, 1664)])
            self.assertEqual(shapes['write'], [(1, 6, 1664), (1, 16, 1664), (1, 16, 1664)])
        finally:
            read_hook.remove()
            write_hook.remove()

    def test_hidden_and_teacher_positions_share_coordinates_and_mask_only_outer_padding(self):
        self.assertEqual(self.coords.hidden_position.shape, (1, 6, 1664))
        self.assertEqual(self.coords.process_valid_indices.numel(), 16)
        # At t=0, H spans teacher y=.5 and x=.25/.75 in the centered letterbox.
        expected = torch.cat((sinusoid(torch.tensor(0.), 512), sinusoid(torch.tensor(.5 * 24), 576),
                              sinusoid(torch.tensor(.25 * 24), 576)))
        torch.testing.assert_close(self.coords.hidden_position[0, 0], expected)
        rows = (self.coords.process_valid_indices % 16) // 4
        self.assertEqual(set(rows.tolist()), {1, 2})

    def test_masked_p_tokens_are_not_values_in_the_write_attention(self):
        hidden_tokens = self.model.A.read(self.model.A.read_norm(self.hidden))
        valid = self.coords.process_valid_indices
        condition = torch.zeros(1, 1, 1664)
        def write(process):
            return self.model.shared.write_attention(hidden_tokens, process.index_select(1, valid),
                self.coords.hidden_position, self.coords.position.index_select(1, valid), condition)
        with torch.no_grad():
            original = write(self.process)
            changed = self.process.clone()
            mask = torch.ones(32, dtype=torch.bool)
            mask[valid] = False
            changed[:, mask] = 100 * torch.randn_like(changed[:, mask])
            torch.testing.assert_close(write(changed), original, rtol=0, atol=0)

    def test_positions_address_keys_and_attention_can_read_future_tokens(self):
        attention = self.model.shared.read_attention
        query = self.process[:, :1]
        context = torch.randn(1, 4, 1664, requires_grad=True)
        positions = torch.randn_like(context)
        query_position = self.coords.position[:, :1]
        condition = torch.zeros_like(query)
        output = attention(query, context, query_position, positions, condition)
        permutation = torch.tensor([3, 1, 2, 0])
        moved = attention(query, context[:, permutation], query_position, positions[:, permutation], condition)
        torch.testing.assert_close(output, moved, atol=2e-7, rtol=1e-5)
        wrong_time = attention(query, context, query_position, positions[:, permutation], condition)
        self.assertGreater(float((output - wrong_time).detach().abs().max()), 1e-4)
        gradient = torch.autograd.grad(output.square().mean(), context)[0]
        self.assertGreater(float(gradient[:, -1].norm()), 0)

    def test_writer_has_no_hidden_value_residual_bypassing_p(self):
        attention = self.model.shared.write_attention
        # Identical P values must yield identical retrievals regardless of H queries/positions.
        process = torch.ones(1, 4, 1664)
        positions = torch.randn_like(process)
        condition = torch.zeros(1, 1, 1664)
        with torch.no_grad():
            first = attention(torch.randn(1, 6, 1664), process, self.coords.hidden_position, positions, condition)
            second = attention(torch.randn(1, 6, 1664) * 5, process, self.coords.hidden_position, positions, condition)
        torch.testing.assert_close(first, second, atol=2e-7, rtol=1e-5)

    def test_zero_writer_preserves_h_but_read_and_writer_can_start_learning(self):
        with torch.no_grad():
            self.model.A.writer.weight.zero_()
        hidden, process, _, metrics = self.call()
        torch.testing.assert_close(hidden, self.hidden, rtol=0, atol=0)
        (hidden.square().mean() + process.square().mean()).backward()
        self.assertGreater(float(self.model.shared.read_attention.attention.value.weight.grad.norm()), 0)
        self.assertGreater(float(self.model.A.writer.weight.grad.norm()), 0)
        self.assertEqual(float(self.model.shared.write_attention.attention.value.weight.grad.norm()), 0)
        groups = self.model.parameter_groups()
        active = [p for name, ps in groups.items() if name != 'initialization' for p in ps]
        self.assertTrue(all(p.grad is not None for p in active))
        self.assertAlmostEqual(float(metrics['read_gate_mean']), torch.sigmoid(torch.tensor(-2.)).item(), places=5)
        all_parameters = [p for ps in groups.values() for p in ps]
        self.assertEqual(len(all_parameters), len({id(p) for p in all_parameters}))

    def test_two_iterations_requery_updated_p_and_checkpoint_keeps_both_attention_gradients(self):
        self.model.config['iterations'] = 2
        queries = []
        hook = self.model.shared.read_attention.register_forward_pre_hook(
            lambda module, args: queries.append(args[0].detach().clone()))
        try:
            hidden, process, _, _ = self.call()
        finally:
            hook.remove()
        self.assertEqual(len(queries), 2)
        self.assertGreater(float((queries[1] - queries[0]).norm()), 0)
        (hidden.square().mean() + process.square().mean()).backward()
        parameters = [self.model.shared.read_attention.attention.query.weight,
                      self.model.shared.write_attention.attention.query.weight,
                      self.model.shared.write_attention.attention.value.weight,
                      self.model.shared.read_gate[-1].weight]
        gradients = [p.grad.clone() for p in parameters]
        self.assertTrue(all(float(g.norm()) > 0 for g in gradients))
        self.model.zero_grad(set_to_none=True)
        self.model.config['checkpoint_cell'] = True
        repeated_h, repeated_p, _, _ = self.call()
        (repeated_h.square().mean() + repeated_p.square().mean()).backward()
        torch.testing.assert_close(hidden, repeated_h)
        torch.testing.assert_close(process, repeated_p)
        for parameter, expected in zip(parameters, gradients):
            torch.testing.assert_close(parameter.grad, expected)

    def test_legacy_fusion_migration_requires_trainable_new_core_and_restarts_writers(self):
        weights = {name: value for name, value in self.model.state_dict().items()
                   if not name.startswith(NEW_PREFIXES)}
        source = dict(initialization='image_text')  # Missing fusion means old interpolation.
        with self.assertRaisesRegex(RuntimeError, 'A/AB'):
            self.model.load_checkpoint_weights(weights, source)
        incomplete = dict(weights)
        incomplete.pop('A.read.weight')
        with self.assertRaisesRegex(RuntimeError, 'mismatch'):
            self.model.load_checkpoint_weights(incomplete, source, allow_new_fusion=True)
        new_attention = self.model.shared.read_attention.attention.query.weight.detach().clone()
        report = self.model.load_checkpoint_weights(weights, source, allow_new_fusion=True)
        self.assertEqual(report['fusion'], 'new cross_attention')
        self.assertEqual(float(self.model.A.writer.weight.detach().norm()), 0)
        self.assertEqual(float(self.model.B.writer.weight.detach().norm()), 0)
        self.assertEqual(float(self.model.B.b.detach()), 0)
        torch.testing.assert_close(new_attention, self.model.shared.read_attention.attention.query.weight, rtol=0, atol=0)

    def test_old_config_still_loads_strictly_and_matches_original_interpolated_formula(self):
        cfg = dict(self.model.config)
        cfg.pop('fusion')
        cfg.update(iterations=1, checkpoint_cell=False)
        model = Corrector(cfg)
        weights = {name: value for name, value in self.model.state_dict().items()
                   if not name.startswith(NEW_PREFIXES)}
        model.load_state_dict(weights, strict=True)
        self.assertEqual(model.fusion, 'interpolate')
        h, p, c, s = self.hidden, self.process, self.coords, self.sigma
        with torch.no_grad():
            actual_h, actual_p, _, _ = model('A', h, p, self.text, s, c, block=5)
            observation = resample_tokens(model.A.read(model.A.read_norm(h)), c.hidden, c.process)
            condition = model.shared.sigma(sinusoid(s, 1664)).reshape(1, 1, 1664)
            condition = condition + model.shared.site.weight[0] + sinusoid(torch.tensor(5.), 1664).reshape(1, 1, 1664)
            delta, gate = model.shared(p + observation + c.position + condition,
                                       model.shared.text(self.text), False)
            expected_p = p + cfg['state_strength'] * gate * delta
            normalized = model.A.write_norm(expected_p)
            residual = resample_tokens(model.A.write_gate(normalized).sigmoid() * model.A.writer(normalized),
                                       c.process, c.hidden)
            expected_h = h + cfg['write_strength'] * residual
        torch.testing.assert_close(actual_h, expected_h, rtol=0, atol=0)
        torch.testing.assert_close(actual_p, expected_p, rtol=0, atol=0)

    def test_legacy_initializer_and_fusion_can_migrate_together(self):
        weights = {name: value for name, value in self.model.state_dict().items()
                   if not name.startswith(NEW_PREFIXES + ('shared.initialize.',))}
        weights['shared.initialize.weight'] = torch.zeros(1664, 97)
        weights['shared.initialize.bias'] = torch.zeros(1664)
        initialized = self.model.shared.initialize.image.weight.detach().clone()
        report = self.model.load_checkpoint_weights(weights, {}, allow_new_initializer=True, allow_new_fusion=True)
        self.assertEqual(report['initializer'], 'new image_text prior')
        self.assertEqual(report['fusion'], 'new cross_attention')
        torch.testing.assert_close(initialized, self.model.shared.initialize.image.weight, rtol=0, atol=0)
        self.assertEqual(float(self.model.A.writer.weight.detach().norm()), 0)

    def test_same_fusion_phase_transfer_restores_trained_writers_without_reset(self):
        weights = self.model.state_dict()
        trained_writer = self.model.A.writer.weight.detach().clone()
        report = self.model.load_checkpoint_weights(weights, dict(initialization='image_text', fusion='cross_attention'))
        self.assertEqual(report['writers'], 'restored')
        torch.testing.assert_close(trained_writer, self.model.A.writer.weight, rtol=0, atol=0)
        self.assertGreater(float(self.model.A.writer.weight.detach().norm()), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
