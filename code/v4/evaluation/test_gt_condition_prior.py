"""GT/no-GT initialization: real attention, modality boundaries, geometry and scheduling."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import yaml

from physgen_v4.coordinates import Coordinates, encode_position
from physgen_v4.corrector import ConditionPrior
from physgen_v4.losses import prior_distance
from train.train_native_p import runtime_overrides, use_gt_condition
from types import SimpleNamespace

torch.set_num_threads(2)
CONFIG = yaml.safe_load((ROOT / 'configs/wisa_native_p.yaml').read_text())


class GTConditionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9312)
        self.cfg = dict(CONFIG['corrector'], prior_width=64, prior_heads=4, prior_depth=1, prior_video_pool=2)
        self.model = ConditionPrior(self.cfg)
        self.coords = Coordinates.build(torch.linspace(0, 2, 9), 64, 128, 4, 1, teacher_size=64)
        self.first = torch.randn(1, 48, 1, 4, 8)
        self.text = torch.randn(1, 3, 1664)
        self.gt = torch.randn(1, 48, 3, 4, 8)

    def forward(self, gt=None, recompute=False):
        return self.model(self.first, self.text, self.coords, recompute, gt_video=gt)

    def test_future_gt_changes_p0_without_changing_first_or_text(self):
        changed = self.gt.clone()
        changed[:, :, -1] = changed[:, :, -1].roll(1, dims=1)
        with torch.no_grad():
            given = self.forward(self.gt)
            altered = self.forward(changed)
            absent = self.forward()
        self.assertEqual(given.shape, (1, 32, 1664))
        self.assertEqual(given.dtype, torch.float32)
        self.assertGreater(float((given - altered).abs().max()), 1e-6)
        self.assertGreater(float((given - absent).abs().max()), 1e-6)

    def test_no_gt_path_is_identical_at_inference_and_never_uses_video_layers(self):
        with torch.no_grad():
            trained_path = self.forward()
            self.model.eval()
            with patch.object(self.model.video, 'forward', side_effect=AssertionError('GT input at inference')):
                inferred = self.forward()
            torch.testing.assert_close(trained_path, inferred, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'training mode'):
                self.forward(self.gt)

    def test_gt_is_detached_but_gt_encoder_and_shared_prior_learn_from_jepa(self):
        gt = self.gt.clone().requires_grad_()
        target = torch.randn(1, 32, 1664, requires_grad=True)
        predicted = self.forward(gt, recompute=True)
        prior_distance(predicted, target, torch.ones(1, 32), 2).backward()
        self.assertIsNone(gt.grad)
        self.assertIsNone(target.grad)
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
        self.assertGreater(float(self.model.video.weight.grad.norm()), 0)
        self.assertGreater(float(self.model.image.weight.grad.norm()), 0)
        self.assertGreater(float(self.model.text.weight.grad.norm()), 0)

    def test_no_gt_recomputation_preserves_gradients_and_only_gt_weights_are_unused(self):
        reference = None
        for recompute in (False, True):
            self.model.zero_grad(set_to_none=True)
            with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
                result = self.forward(recompute=recompute)
                loss = result.square().mean()
            loss.backward()
            grads = {}
            for name, parameter in self.model.named_parameters():
                if name.startswith(('video.', 'video_norm.', 'video_modality')):
                    self.assertIsNone(parameter.grad, name)
                else:
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)
                    grads[name] = parameter.grad.detach().clone()
            if reference is not None:
                for name, grad in grads.items():
                    torch.testing.assert_close(grad, reference[name], rtol=1e-5, atol=1e-7)
            reference = grads

    def test_pool_retains_all_times_and_letterbox_coordinates(self):
        seen = []
        hook = self.model.video_norm.register_forward_pre_hook(lambda module, args: seen.append(args[0].shape))
        try:
            with torch.no_grad():
                self.forward(self.gt)
        finally:
            hook.remove()
        self.assertEqual(seen, [torch.Size([1, 3 * 2 * 2, 48])])
        # 64x128 content occupies teacher y=[.25,.75]. Mean centers in two bins:
        expected = encode_position(torch.tensor([0., 1., 2.]), torch.tensor([.375, .625]),
                                   torch.tensor([.25, .75]), 1.)
        torch.testing.assert_close(self.coords.pooled_video_position(2, 2), expected)

    def test_invalid_gt_geometry_or_legacy_prior_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.forward(self.gt[:, :, :2])
        legacy = ConditionPrior(dict(self.cfg, initialization='image_text'))
        with self.assertRaisesRegex(ValueError, 'image_text_gt'):
            legacy(self.first, self.text, self.coords, False, gt_video=self.gt)
        for size in (0, -1, 2.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                ConditionPrior(dict(self.cfg, prior_video_pool=size))


class GTPolicyTests(unittest.TestCase):
    def test_validation_never_supplies_gt_to_initializer(self):
        from train import train_native_p as training
        from contextlib import ExitStack
        cfg = copy.deepcopy(CONFIG)
        cfg['validation'].update(sigmas=[.5], samples_per_rank=1)
        sample = dict(latent=torch.ones(1, 2, 2, 2, 2), first=torch.ones(1, 2, 1, 2, 2),
                      text=torch.ones(2, 4), target=torch.ones(1, 4, 3), target_weight=torch.ones(1, 4),
                      record=dict(index=0))
        calls = []
        def model(*args, **kwargs):
            calls.append(kwargs)
            self.assertIsNone(kwargs.get('gt_video'))
            return args[0], None, {}
        actual_autocast = torch.autocast
        with ExitStack() as stack:
            for name, replacement in (('rng_state', lambda: {}), ('restore_rng', lambda state: None),
                    ('move_sample', lambda value, device: value), ('geometry', lambda *args: None),
                    ('reduce_metrics', lambda value, device: value),
                    ('training_loss', lambda *args: (torch.tensor(0.), {'loss/fm': torch.tensor(1.)}))):
                stack.enter_context(patch.object(training, name, replacement))
            stack.enter_context(patch.object(training.dist, 'get_rank', return_value=0))
            stack.enter_context(patch.object(training.dist, 'get_world_size', return_value=1))
            stack.enter_context(patch.object(torch, 'autocast',
                side_effect=lambda device, **kwargs: actual_autocast('cpu', **kwargs)))
            metrics = training.validate(model, [sample], cfg, 249, torch.device('cpu'))
        self.assertEqual(len(calls), 1)
        self.assertEqual(float(metrics['validation_loss']), 1.)

    def test_modality_schedule_is_rank_rng_and_resume_independent(self):
        expected = [use_gt_condition(CONFIG, i) for i in range(500)]
        self.assertTrue(300 < sum(expected) < 430)
        for rank in range(4):
            with patch.dict('os.environ', RANK=str(rank), WORLD_SIZE='4'):
                torch.manual_seed(29 + rank)
                torch.rand(19)
                self.assertEqual([use_gt_condition(CONFIG, i) for i in range(250, 500)], expected[250:])

    def test_explicit_ablation_extremes_and_invalid_dropout(self):
        cfg = copy.deepcopy(CONFIG)
        for dropout, expected in ((0., True), (1., False)):
            cfg['train']['gt_condition_dropout'] = dropout
            self.assertEqual([use_gt_condition(cfg, i) for i in range(10)], [expected] * 10)
        for dropout in (-.1, 1.1, float('nan')):
            cfg['train']['gt_condition_dropout'] = dropout
            with self.assertRaisesRegex(ValueError, 'gt_condition_dropout'):
                runtime_overrides(cfg, SimpleNamespace())


if __name__ == '__main__':
    unittest.main(verbosity=2)
