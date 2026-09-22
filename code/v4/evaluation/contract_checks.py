"""Bounded CPU checks of mathematical invariants; these do not evaluate generated-video quality."""
import copy
import itertools
import json
from pathlib import Path
import sys
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
import numpy as np

from physgen_v4.coordinates import Coordinates, resample_tokens, teacher_indices
from physgen_v4.corrector import Corrector
from physgen_v4.backbone import ProcessWan
from physgen_v4.data import TrainingOrder, frame_window, token_content_weights, teacher_content_weights
from physgen_v4.views import choose_canvas, fit_geometry
from physgen_v4.losses import endpoint, noisy_view, training_loss, weighted_mean
from evaluation.summarize_ratings import wilson

ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/wisa_native_p.yaml").read_text())
torch.set_num_threads(2)


class MathematicalContracts(unittest.TestCase):
    def test_zero_observed_damage_does_not_claim_zero_population_risk(self):
        interval = wilson(0, 50)["confidence_interval_95"]
        self.assertGreater(interval[1], 0.07)
        self.assertLess(interval[1], 0.072)

    def test_content_weights_measure_patch_area_without_counting_black_canvas(self):
        weight = token_content_weights(dict(top=14, left=0, resized_height=36, resized_width=64), 64, 4)
        torch.testing.assert_close(weight.mean(), torch.tensor(36 / 64))
        expected_y = torch.tensor([2 / 16, 1., 1., 2 / 16])
        torch.testing.assert_close(weight.reshape(2, 4, 4)[0, :, 0], expected_y)
        values = torch.tensor([[100., 2., 6.]])
        measured = weighted_mean(values, torch.tensor([[0., 1., 0.5]]))
        torch.testing.assert_close(measured, torch.tensor(5 / 1.5))

    def test_frame_cap_uses_contiguous_original_frames_without_temporal_resampling(self):
        np.testing.assert_array_equal(frame_window(250, 101), np.arange(74, 175))
        indices = frame_window(70, 101)
        self.assertEqual(len(indices), 69)
        self.assertEqual((len(indices) - 1) % 4, 0)
        self.assertTrue(bool((np.diff(indices) == 1).all()))
        np.testing.assert_array_equal(frame_window(250, 81, 25), np.arange(25, 106))

    def test_rectangular_canvas_preserves_aspect_and_teacher_coordinate_mapping(self):
        height, width = choose_canvas(1080, 1920, 512, 147456)
        self.assertEqual((height, width), (288, 512))
        geo = fit_geometry(1080, 1920, height, width, False)
        self.assertEqual(geo["resized_width"] / geo["resized_height"], 1920 / 1080)
        weight = teacher_content_weights(geo, 384, 16)
        torch.testing.assert_close(weight.mean(), torch.tensor(216 / 384))
        coords = Coordinates.build(torch.arange(101) / 25, height, width, 16, 1)
        self.assertEqual(tuple(len(a) for a in coords.hidden), (26, 9, 16))
        self.assertEqual(coords.position.shape, (1, 4608, 1664))
        teacher_y = (torch.arange(24) + 0.5) * 16
        torch.testing.assert_close(coords.process[1], (teacher_y - 84) / 216)
        for original in ((720, 1280), (1920, 1080), (800, 1000), (120, 160)):
            h, w = choose_canvas(*original, 512, 147456)
            self.assertLessEqual(h * w, 147456)
            self.assertLessEqual(max(h, w), 512)
            self.assertEqual((h % 32, w % 32), (0, 0))

    def test_coordinate_sampling_preserves_linear_fields_and_gradient_mass(self):
        source = (torch.tensor([0., 1., 4.]), torch.tensor([0., 0.4, 1.]), torch.tensor([0., 0.3, 1.]))
        target = (torch.tensor([0.3, 2.]), torch.tensor([0.2, 0.7]), torch.tensor([0.1, 0.8]))
        t, y, x = torch.meshgrid(*source, indexing="ij")
        field = (2 * t + 3 * y - 4 * x).reshape(1, -1, 1).requires_grad_()
        sampled = resample_tokens(field, source, target)
        tt, yy, xx = torch.meshgrid(*target, indexing="ij")
        torch.testing.assert_close(sampled, (2 * tt + 3 * yy - 4 * xx).reshape_as(sampled), atol=2e-6, rtol=1e-6)
        sampled.sum().backward()
        torch.testing.assert_close(field.grad.sum(), torch.tensor(float(sampled.numel())))
        self.assertTrue(bool((field.grad >= 0).all()))

    def test_native_grid_and_first_slot_coordinates(self):
        times = torch.linspace(0, 6, 49)
        coords = Coordinates.build(times, 384, 384, 16, 1)
        self.assertEqual(tuple(len(x) for x in coords.process), (8, 24, 24))
        self.assertEqual(tuple(len(x) for x in coords.hidden), (13, 12, 12))
        self.assertEqual(coords.position.shape, (1, 4608, 1664))
        torch.testing.assert_close(coords.latent[0], times[::4])
        torch.testing.assert_close(coords.process[0], times[teacher_indices(49, 16)].reshape(8, 2).mean(1))

    def test_short_clips_use_distinct_teacher_frames_and_invertible_time_coordinates(self):
        for frames in (5, 9, 13, 17, 101):
            indices = teacher_indices(frames, 16)
            self.assertEqual(indices.numel() % 2, 0)
            self.assertLessEqual(indices.numel(), 16)
            self.assertEqual(indices.numel(), indices.unique().numel())
            self.assertEqual((int(indices[0]), int(indices[-1])), (0, frames - 1))
            coords = Coordinates.build(torch.arange(frames) / 25, 64, 64, 16, 1, teacher_size=64)
            self.assertTrue(bool((torch.diff(coords.process[0]) > 0).all()))
            field = torch.ones(1, indices.numel() // 2 * 16, 3, requires_grad=True)
            mapped = resample_tokens(field, coords.process, coords.hidden)
            torch.testing.assert_close(mapped, torch.ones_like(mapped))
            mapped.sum().backward()
            torch.testing.assert_close(field.grad.sum(), torch.tensor(float(mapped.numel())))

    def test_flow_endpoint_is_exact_and_known_frame_receives_no_fm_gradient(self):
        clean = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 2, 2) / 48
        first = clean[:, :, :1] + 0.3
        noise = clean.flip(2)
        sigma = torch.tensor(0.37)
        noisy = noisy_view(clean, first, noise, sigma)
        torch.testing.assert_close(noisy[:, :, :1], first)
        recovered = endpoint(noisy, noise - clean, first, sigma)
        torch.testing.assert_close(recovered[:, :, 1:], clean[:, :, 1:])
        torch.testing.assert_close(recovered[:, :, :1], first)
        prediction = torch.zeros_like(clean, requires_grad=True)
        state = torch.zeros(1, 2, 3, requires_grad=True)
        info = dict(states={"A": state}, before={"A": state.detach()}, initial=state.detach(), metrics={})
        loss, _ = training_loss(prediction, noise, clean, first, noisy, sigma, info, torch.ones_like(state), torch.ones(1, 2),
                                200, CONFIG, False)
        loss.backward()
        torch.testing.assert_close(prediction.grad[:, :, :1], torch.zeros_like(first))
        self.assertGreater(float(prediction.grad[:, :, 1:].abs().sum()), 0)

    def test_noise_taper_survives_state_averaging_and_vanishes_at_pure_noise(self):
        config = copy.deepcopy(CONFIG)
        config["loss"]["fm_weight"] = 0
        state = torch.ones(1, 2, 3, requires_grad=True)
        info = dict(states={"A": state, "B": state}, before={"A": state.detach(), "B": state.detach()},
                    initial=state.detach(), metrics={})
        z = torch.zeros(1, 3, 2, 2, 2)
        losses = []
        for sigma in (0.9, 0.95, 1.0):
            loss, _ = training_loss(z, z, z, z[:, :, :1], z, torch.tensor(sigma), info,
                                    torch.zeros_like(state), torch.ones(1, 2), 200, config, False)
            losses.append(loss)
        torch.testing.assert_close(losses[0], 2 * losses[1])
        torch.testing.assert_close(losses[2], torch.zeros(()))

    def test_resume_order_is_independent_of_prefetch_and_preserves_global_sequence(self):
        size, seed, world, consumed = 37, 23, 4, 28
        for rank in range(world):
            original = list(itertools.islice(TrainingOrder([i % 3 for i in range(size)], seed, rank, world, 0), 25))
            resumed = list(itertools.islice(TrainingOrder([i % 3 for i in range(size)], seed, rank, world, consumed), 18))
            self.assertEqual(original[consumed // world:], resumed)
        interleaved = []
        streams = [iter(TrainingOrder([i % 3 for i in range(size)], seed, rank, world, 0)) for rank in range(world)]
        for index in range(size):
            interleaved.append(next(streams[index % world]))
        self.assertEqual(sorted(interleaved), list(range(size)))


class CorrectorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(71)
        cls.config = copy.deepcopy(CONFIG["corrector"])
        cls.config["checkpoint_cell"] = False
        cls.config["parameter_sharing"] = "shared"  # Legacy architecture regression coverage.
        cls.config["initialization"] = "image_text"
        cls.model = Corrector(cls.config)
        cls.model.set_stage("B")
        cls.coords = Coordinates.build(torch.linspace(0, 2, 9), 64, 64, 4, 1, teacher_size=64)
        cls.hidden = torch.randn(1, 12, 3072)
        cls.process = torch.randn(1, 32, 1664)
        cls.text = torch.randn(1, 3, 4096)
        cls.sigma = torch.tensor(0.4)

    def setUp(self):
        self.model.zero_grad(set_to_none=True)
        self.model.set_stage("B")
        self.model.config["checkpoint_cell"] = False
        with torch.no_grad():
            self.model.B.b.zero_()
            self.model.B.writer.weight.zero_()

    def call_b(self):
        return self.model("B", self.hidden, self.process, self.text, self.sigma, self.coords)

    def test_parameter_budget_matches_native_method(self):
        with torch.device("meta"):
            model = Corrector(self.config)
        expected = {"A": 149992291, "B": 10236418, "AB": 160228709}
        for stage, count in expected.items():
            model.set_stage(stage)
            self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad), count)

    def test_initial_state_uses_image_and_each_cfg_prompt_without_noise(self):
        first = torch.randn(1, 48, 1, 4, 4)
        with torch.no_grad():
            positive = self.model.initialize(first, self.text, self.coords)
            rng = torch.get_rng_state()
            repeated = self.model.initialize(first, self.text, self.coords)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            empty = self.model.initialize(first, torch.zeros(1, 1, 4096), self.coords)
            changed_image = self.model.initialize(first.flip(-1), self.text, self.coords)
        torch.testing.assert_close(positive, repeated, rtol=0, atol=0)
        for changed in (empty, changed_image):
            # Both temporal groups, including the last, read both conditions.
            per_time = (positive - changed).reshape(1, 2, 16, 1664).square().mean((2, 3))
            self.assertTrue(bool((per_time > 1e-8).all()))
        self.assertEqual(positive.shape, (1, 32, 1664))

    def test_initial_state_alone_backpropagates_to_text_and_image(self):
        self.model.set_stage("A")
        text = self.text.detach().clone().requires_grad_()
        first = torch.randn(1, 48, 1, 4, 4, requires_grad=True)
        state = self.model.initialize(first, text, self.coords)
        (state - torch.ones_like(state)).square().mean().backward()
        self.assertGreater(float(self.model.shared.initialize.image.weight.grad.norm()), 0)
        self.assertGreater(float(self.model.shared.text.weight.grad.norm()), 0)
        self.assertGreater(float(text.grad.norm()), 0)
        self.assertGreater(float(first.grad.norm()), 0)
        self.assertTrue(all(p.grad is not None for p in self.model.parameter_groups()["initialization"]))

    def test_legacy_checkpoint_config_keeps_text_independent_initialization(self):
        legacy_config = {k: v for k, v in self.config.items() if k != "initialization"}
        with torch.device("meta"):
            legacy = Corrector(legacy_config)
        legacy.shared.initialize = torch.nn.Linear(97, 1664)
        self.assertEqual(legacy.initialization, "latent")
        self.assertIn("shared.initialize.weight", legacy.state_dict())
        noisy, first = torch.randn(1, 48, 3, 4, 4), torch.randn(1, 48, 1, 4, 4)
        with torch.no_grad():
            positive = legacy.initialize(first, self.text, self.coords, noisy=noisy)
            negative = legacy.initialize(first, torch.zeros_like(self.text), self.coords, noisy=noisy)
        torch.testing.assert_close(positive, negative, rtol=0, atol=0)

    def test_prompt_initialization_preserves_fp32_state_under_autocast(self):
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            state = self.model.initialize(torch.randn(1, 48, 1, 4, 4), self.text.bfloat16(), self.coords)
        self.assertEqual(state.dtype, torch.float32)
        self.assertTrue(bool(torch.isfinite(state).all()))

    def test_wan_entry_keeps_prior_independent_of_noisy_video_and_reuses_state(self):
        wan = SimpleNamespace(patch_embedding=torch.nn.Conv3d(48, 4, 1), freq_dim=4, dim=4,
            time_embedding=torch.nn.Identity(), time_projection=torch.nn.Linear(4, 24),
            text_len=8, text_embedding=torch.nn.Linear(4096, 4), freqs=torch.empty(0), blocks=[],
            head=lambda hidden, condition: hidden, unpatchify=lambda hidden, grid: [hidden])
        model = ProcessWan(wan, self.model, False)
        time_module = SimpleNamespace(sinusoidal_embedding_1d=lambda width, t: t[:, None].expand(-1, width))
        first = torch.randn(1, 48, 1, 4, 4)
        noisy = torch.cat((first, torch.randn(1, 48, 2, 4, 4)), dim=2)
        with torch.no_grad(), patch.dict(sys.modules, {"wan.modules.model": time_module}), \
                patch.object(self.model, "initialize", wraps=self.model.initialize) as initialize:
            _, positive, _ = model(noisy, first, self.text, self.sigma, self.coords)
            self.assertIs(initialize.call_args.args[1], self.text)
            self.assertEqual(initialize.call_args.kwargs, {'gt_video': None})
            _, other_noise, _ = model(noisy + 2, first, self.text, self.sigma, self.coords)
            negative_text = torch.zeros_like(self.text)
            _, negative, _ = model(noisy, first, negative_text, self.sigma, self.coords)
            self.assertIs(initialize.call_args.args[1], negative_text)
            _, carried, info = model(noisy, first, negative_text, self.sigma, self.coords, positive)
            self.assertEqual(initialize.call_count, 3)
        torch.testing.assert_close(positive, other_noise, rtol=0, atol=0)
        self.assertGreater(float((positive - negative).abs().max()), 1e-4)
        self.assertIs(carried, positive)
        self.assertIsNone(info["prior"])

    def test_later_state_loss_trains_prior_without_an_initial_loss(self):
        self.model.set_stage("A")
        state = self.model.initialize(torch.randn(1, 48, 1, 4, 4), self.text, self.coords)
        _, updated, _, _ = self.model("A", self.hidden, state, self.text, self.sigma, self.coords)
        (updated - torch.ones_like(updated)).square().mean().backward()
        self.assertGreater(float(self.model.shared.initialize.output[-1].weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is not None for p in self.model.parameter_groups()["initialization"]))
        self.assertIsNone(self.model.A.writer.weight.grad)

    def test_warm_state_explicitly_excludes_initializer_from_gradient_collectives(self):
        self.model.set_stage("A")
        with torch.no_grad():
            state = self.model.initialize(torch.randn(1, 48, 1, 4, 4), self.text, self.coords)
        hidden, state, _, _ = self.model("A", self.hidden, state, self.text, self.sigma, self.coords)
        (hidden.square().mean() + state.square().mean()).backward()
        groups = self.model.parameter_groups()
        self.assertTrue(all(p.grad is None for p in groups["initialization"]))
        self.assertTrue(all(p.grad is not None for name, parameters in groups.items()
                            if name != "initialization" for p in parameters))

    def test_joint_stage_with_only_a_has_no_b_gradients(self):
        self.model.set_stage("AB")
        state = self.model.initialize(torch.randn(1, 48, 1, 4, 4), self.text, self.coords)
        hidden, state, _, _ = self.model("A", self.hidden, state, self.text, torch.tensor(0.9), self.coords)
        (hidden.square().mean() + state.square().mean()).backward()
        groups = self.model.parameter_groups()
        self.assertTrue(all(p.grad is None for p in groups["B"]))
        self.assertTrue(all(p.grad is not None for name, parameters in groups.items()
                            if name != "B" for p in parameters))

    def test_fp32_persistence_retains_small_b_updates_under_bfloat16_autocast(self):
        with torch.no_grad():
            self.model.B.b.fill_(1e-4)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            state = self.model.initialize(torch.randn(1, 48, 1, 4, 4), self.text, self.coords)
            _, updated, _, _ = self.model("B", self.hidden, state, self.text, self.sigma, self.coords)
        self.assertEqual(state.dtype, torch.float32)
        self.assertEqual(updated.dtype, torch.float32)
        self.assertGreater(float((updated - state).detach().square().mean().sqrt()), 1e-7)

    def test_b_starts_as_identity_but_state_scale_and_writer_can_learn(self):
        hidden, process, _, _ = self.call_b()
        torch.testing.assert_close(hidden, self.hidden, rtol=0, atol=0)
        torch.testing.assert_close(process, self.process, rtol=0, atol=0)
        loss = hidden.square().mean() + process.square().mean()
        loss.backward()
        self.assertGreater(float(self.model.B.b.grad.abs()), 0)
        self.assertGreater(float(self.model.B.writer.weight.grad.norm()), 0)
        torch.testing.assert_close(self.model.B.read.weight.grad, torch.zeros_like(self.model.B.read.weight))

    def test_structure_loss_does_not_train_writer_but_output_uses_committed_state(self):
        with torch.no_grad():
            self.model.B.b.fill_(0.15)
            torch.nn.init.normal_(self.model.B.writer.weight, std=0.001)
        hidden, process, _, _ = self.call_b()
        writer_gradient = torch.autograd.grad(process.square().mean(), self.model.B.writer.weight,
                                              allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(writer_gradient)
        state_gradient = torch.autograd.grad(hidden.square().mean(), process, retain_graph=True)[0]
        self.assertGreater(float(state_gradient.norm()), 0)
        hidden.square().mean().backward()
        self.assertGreater(float(self.model.B.read.weight.grad.norm()), 0)
        self.assertTrue(all(p.grad is None for p in self.model.shared.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.A.parameters()))

    def test_checkpoint_recomputation_preserves_outputs_and_input_adapter_gradients(self):
        with torch.no_grad():
            self.model.B.b.fill_(0.15)
            torch.nn.init.normal_(self.model.B.writer.weight, std=0.001)
        hidden, process, _, _ = self.call_b()
        (hidden.square().mean() + process.square().mean()).backward()
        reference = (hidden.detach(), process.detach(), self.model.B.read.weight.grad.clone(), self.model.B.b.grad.clone())
        self.model.zero_grad(set_to_none=True)
        self.model.config["checkpoint_cell"] = True
        hidden, process, _, _ = self.call_b()
        (hidden.square().mean() + process.square().mean()).backward()
        for measured, expected in zip((hidden, process, self.model.B.read.weight.grad, self.model.B.b.grad), reference):
            torch.testing.assert_close(measured, expected, atol=1e-7, rtol=1e-5)


if __name__ == "__main__":
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    destination = ROOT / "evaluation/results/cpu_contracts.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(tests=result.testsRun, failures=len(result.failures),
        errors=len(result.errors), seconds=time.perf_counter() - started, device="cpu",
        torch_version=torch.__version__, scope="mathematical and gradient contracts; no video-effect evidence"), indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
