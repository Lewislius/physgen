"""Focused restoration checks; no video ablation grid or training sweep."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_v4.corrector import Corrector
from physgen_v4.coordinates import Coordinates, teacher_indices
from physgen_v4.data import CachedWISA
from physgen_v4.losses import training_loss
from physgen_v4.runtime import read_config
from physgen_v4.write_supervision import native_write_loss, joint_gradient_diagnostics, paired_input

V4 = ROOT.parent / 'v4'
CHECKPOINT = V4 / 'checkpoints/wisa_native_p_A1_4x96g_20260912T151800Z-e189c27c/checkpoint-final'


class RestorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        torch.manual_seed(17)
        cls.config = read_config(ROOT / 'configs/stability_v4_fullwidth3.yaml')
        spec = importlib.util.spec_from_file_location('reference_v4', V4 / 'physgen_v4/__init__.py',
                                                     submodule_search_locations=[str(V4 / 'physgen_v4')])
        package = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = package
        spec.loader.exec_module(package)
        from reference_v4.corrector import Corrector as ReferenceCorrector
        source_config = read_config(CHECKPOINT / 'config.yaml')['corrector']
        with torch.device('meta'):
            cls.model = Corrector(dict(cls.config['corrector'], checkpoint_cell=False))
            cls.reference = ReferenceCorrector(dict(source_config, checkpoint_cell=False))
        weights = torch.load(CHECKPOINT / 'corrector.pt', map_location='cpu', mmap=True, weights_only=True)
        cls.reference.load_state_dict(weights, strict=True, assign=True)
        cls.reference.eval().requires_grad_(False)
        # Exact, named source tensors only; no relabelling, padding, or width conversion.
        subset = {name: weights[name] for name in cls.model.state_dict()}
        cls.model.load_state_dict(subset, strict=True, assign=True)
        cls.model.set_stage('A')
        cls.coords = SimpleNamespace(position=torch.randn(1, 4, 1664),
            hidden_position=torch.randn(1, 3, 1664), process_valid_indices=torch.arange(4),
            hidden=(torch.arange(3), torch.arange(1), torch.arange(1)))
        cls.hidden, cls.process = torch.randn(1, 3, 3072), torch.randn(1, 4, 1664)
        cls.text, cls.sigma = torch.randn(1, 2, 4096), torch.tensor(.5)

    def test_fullwidth_count_and_exact_v4_forward(self):
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), 429284425)
        self.assertEqual(tuple(self.model.A), ('5', '15', '25'))
        self.assertFalse(hasattr(self.model, 'B'))
        self.assertEqual(len({unit.site.writer.weight.data_ptr() for unit in self.model.A.values()}), 3)
        for block in (5, 15, 25):
            core = self.model.A[str(block)].core
            self.assertEqual(core.blocks[0].self_attention.query.in_features, 1664)
            self.assertEqual(core.blocks[0].self_attention.heads, 26)
            with torch.no_grad():
                actual = self.model('A', self.hidden, self.process, self.text, self.sigma, self.coords, block)
                expected = self.reference('A', self.hidden, self.process, self.text, self.sigma, self.coords, block)
            for a, b in zip(actual[:3], expected[:3]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for name in actual[3]:
                torch.testing.assert_close(actual[3][name], expected[3][name], rtol=0, atol=0)

    def test_write_is_actual_native_path_and_inputs_are_detached(self):
        reference, student = {}, {}
        tracked_inputs = []
        for block in (5, 15, 25):
            label = f'A@{block}'
            h, p = self.hidden.clone().requires_grad_(), self.process.clone().requires_grad_()
            with torch.no_grad():
                residual, _, _ = self.model.write_residual('A', h, p, self.sigma, self.coords, block)
            after = (h + residual).detach().requires_grad_()
            pair = dict(before=h, process=p, after=after, block=block, family='A')
            reference[label] = pair
            student[label] = pair
            tracked_inputs.extend((h, p, after))
        zero, _ = native_write_loss(self.model, reference, student, self.sigma, self.coords, True)
        self.assertEqual(float(zero.detach()), 0.)
        student = {label: dict(pair, before=pair['before'] + .02) for label, pair in student.items()}
        self.model.zero_grad(set_to_none=True)
        loss, _ = native_write_loss(self.model, reference, student, self.sigma, self.coords, True)
        self.assertGreater(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(all(x.grad is None for x in tracked_inputs))
        self.assertTrue(all(p.grad is None for p in self.model.prior.parameters()))
        for unit in self.model.A.values():
            self.assertGreater(float(unit.site.writer.weight.grad.norm()), 0.)
            self.assertGreater(float(unit.core.write_attention.attention.value.weight.grad.norm()), 0.)
            self.assertTrue(all(p.grad is None for p in unit.core.blocks.parameters()))
        self.model.zero_grad(set_to_none=True)

    def test_jepa32_real_cache_and_time_coordinates(self):
        data = CachedWISA(self.config['paths']['cache_root'], 'train', config=self.config)
        self.assertEqual(len(data), 2279)
        sample = data[0]
        coords = Coordinates.build(sample['times'], 288, 512, 32, 1., teacher_sampling='adjacent_pairs')
        self.assertEqual(tuple(sample['target'].shape), (1, 16 * 576, 1664))
        self.assertEqual(len(coords.process[0]), 16)
        indices = teacher_indices(121, 32, 'adjacent_pairs')
        self.assertEqual(indices.numel(), 32)
        self.assertTrue(torch.equal(indices[1::2] - indices[::2], torch.ones(16, dtype=torch.long)))
        torch.testing.assert_close(coords.process[0], sample['times'][indices].reshape(16, 2).mean(1))
        self.assertEqual(tuple(sample['first'].shape), (1, 48, 1, 18, 32))
        self.assertEqual(self.config['data']['training_mode'], 'i2v')

    def test_original_losses_unchanged_and_write_joint_backward(self):
        from reference_v4.losses import training_loss as original_loss
        prediction = torch.randn(1, 48, 2, 1, 1, requires_grad=True)
        clean, noise = torch.randn_like(prediction), torch.randn_like(prediction)
        first, noisy, sigma = clean[:, :, :1], clean.clone(), torch.tensor(.6)
        prior = torch.randn(1, 4, 1664, requires_grad=True)
        target, weight = torch.randn_like(prior), torch.ones(1, 4)
        states = {f'A@{n}': prior + .01 * n for n in (5, 15, 25)}
        info = dict(metrics={}, initial=prior.detach(), prior=prior, prior_time_tokens=1,
                    states=states, before={name: prior.detach() for name in states})
        write = torch.tensor(2., requires_grad=True)
        base, _ = original_loss(prediction, noise, clean, first, noisy, sigma, info, target,
                                weight, 199, self.config, False)
        total, _, terms = training_loss(prediction, noise, clean, first, noisy, sigma, info, target,
            weight, 199, self.config, False, write_loss=write, return_terms=True)
        torch.testing.assert_close(total, base + self.config['loss']['write_weight'] * write)
        no_prior_config = read_config(ROOT / 'configs/stability_v4_fullwidth3_no_prior.yaml')
        no_prior_total, no_prior_metrics, no_prior_terms = training_loss(prediction, noise, clean, first,
            noisy, sigma, info, target, weight, 199, no_prior_config, False, write_loss=write, return_terms=True)
        torch.testing.assert_close(no_prior_total, total - terms['prior'])
        self.assertEqual(float(no_prior_terms['prior']), 0.)
        self.assertFalse(no_prior_terms['prior'].requires_grad)
        self.assertEqual(no_prior_metrics['prior/active'], 0.)
        no_prior_grad, = torch.autograd.grad(no_prior_total, prior, retain_graph=True)
        self.assertGreater(float(no_prior_grad.norm()), 0.)
        total.backward()
        torch.testing.assert_close(write.grad, torch.tensor(self.config['loss']['write_weight']))
        self.assertGreater(float(prior.grad.norm()), 0.)
        self.assertGreater(float(prediction.grad[:, :, 1:].norm()), 0.)
        self.assertEqual(float(prediction.grad[:, :, :1].norm()), 0.)
        x = torch.nn.Parameter(torch.tensor(1.))
        observed = joint_gradient_diagnostics(dict(fm=x.square(), struct=-x, prior=x * 0, write=x), {'probe': [x]})
        self.assertIsNone(x.grad)
        self.assertEqual(float(observed['loss_grad/probe/fm_struct_cosine']), -1.)
        self.assertEqual(float(observed['loss_grad/probe/fm_write_cosine']), 1.)

    def test_prior_variants_differ_only_in_the_requested_loss(self):
        import copy
        with_prior = copy.deepcopy(self.config)
        without = read_config(ROOT / 'configs/stability_v4_fullwidth3_no_prior.yaml')
        self.assertEqual(without['loss']['prior_weight'], 0.)
        self.assertEqual(with_prior['loss']['prior_weight'], .02)
        for config in (with_prior, without):
            config.pop('run_name')
            config['loss'].pop('prior_weight')
        self.assertEqual(with_prior, without)

    def test_every_write_pair_is_perturbed_and_first_frame_is_fixed(self):
        clean = torch.randn(1, 48, 3, 2, 2)
        first = clean[:, :, :1].clone()
        noisy = torch.randn_like(clean)
        noisy[:, :, :1] = first
        student, active = paired_input(clean, noisy, first, torch.tensor(.5), self.config['write_supervision'])
        self.assertTrue(active)
        torch.testing.assert_close(student[:, :, :1], first, rtol=0, atol=0)
        self.assertGreater(float((student[:, :, 1:] - noisy[:, :, 1:]).norm()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
