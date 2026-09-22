"""CPU correctness checks. Tiny fixture widths are explicit; production stays 3072/1664."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch import nn

from physgen_v43.backbone import ProcessWan
from physgen_v43.corrector import Corrector, ARCHITECTURE
from physgen_v43.coordinates import Coordinates
from physgen_v43.runtime import read_config
from physgen_v43.config import validate_config
from physgen_v43.losses import noisy_view, flow_mse, flow_target, oracle_regression, training_loss
from physgen_v43.oracle import solve_oracle, build_oracle_targets, select_sites

torch.set_num_threads(2)
ROOT = Path(__file__).resolve().parents[1]


def oracle_config(**updates):
    cfg = copy.deepcopy(read_config(ROOT / "configs/flow_oracle.yaml")["oracle"])
    cfg.update(regularization=0., inner_steps=2, relative_step=.1, min_relative_gain=0., min_absolute_gain=1e-9)
    cfg.update(updates)
    return cfg


class FakeBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, hidden, **kwargs):
        return hidden.float() + .2 * torch.tanh(self.linear(hidden).float() + .02 * kwargs["context"].mean(1, keepdim=True))


class FakeHead(nn.Module):
    def __init__(self, width, channels):
        super().__init__()
        self.output = nn.Linear(width, channels * 4)

    def forward(self, hidden, time):
        return self.output(hidden + time * .001)


class FakeWan(nn.Module):
    def __init__(self, width=16, channels=4, text_width=20):
        super().__init__()
        self.dim, self.freq_dim, self.text_len = width, 8, 8
        self.channels = channels
        self.patch_embedding = nn.Conv3d(channels, width, (1, 2, 2), stride=(1, 2, 2))
        self.time_embedding = nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width))
        self.time_projection = nn.Linear(width, width * 6)
        self.text_embedding = nn.Linear(text_width, width)
        self.blocks = nn.ModuleList([FakeBlock(width) for _ in range(4)])
        self.head = FakeHead(width, channels)
        self.register_buffer("freqs", torch.zeros(1))

    def unpatchify(self, hidden, grid):
        t, h, w = grid[0].tolist()
        out = hidden.reshape(1, t, h, w, 2, 2, self.channels)
        return out.permute(0, 6, 1, 2, 4, 3, 5).reshape(1, self.channels, t, h * 2, w * 2)


def fixture(checkpoint=True, channels=4):
    torch.manual_seed(17)
    cfg = read_config(ROOT / "configs/flow_oracle.yaml")
    cfg["corrector"].update(hidden_width=16, jepa_width=12, text_width=20, latent_channels=channels,
        heads=4, depth=2, mlp_ratio=2, mapping_width=20, initializer_width=12,
        initializer_heads=3, initializer_depth=1, a_blocks=[1, 2, 3], checkpoint_cell=checkpoint)
    cfg["data"].update(teacher_size=32, teacher_frames=4)
    cfg["train"].update(accumulation_steps=1)
    cfg["oracle"] = oracle_config(relative_step=.02)
    coords = Coordinates.build(torch.arange(5).float() / 24, 64, 96, 4, 1., 32, "adjacent_pairs")
    corrector = Corrector(cfg["corrector"])
    wan = FakeWan(channels=channels).eval().requires_grad_(False)
    model = ProcessWan(wan, corrector)
    clean, first = torch.randn(1, channels, 2, 4, 6), torch.randn(1, channels, 1, 4, 6)
    noise, text, sigma = torch.randn_like(clean), torch.randn(1, 3, 20), torch.tensor(.5)
    target = torch.randn(1, coords.position.shape[1], 12)
    sample = dict(latent=clean, first=first, text=text[0], target=target,
        target_weight=torch.ones(target.shape[:2]), times=torch.arange(5).float() / 24,
        record=dict(index=0), video_metadata=dict(frames=5, height=64, width=96))
    return cfg, coords, model, sample, noise, text, sigma


class OracleTests(unittest.TestCase):
    def scalar_problem(self, **options):
        before = torch.tensor([[[5.], [10.]]], requires_grad=True)
        current = torch.tensor([[[0.], [-2.]]], requires_grad=True)
        mask = torch.tensor([[[0.], [1.]]])
        target = torch.tensor([5., options.pop("target", 6.)]).reshape(1, 1, 2, 1, 1)
        parameter = nn.Parameter(torch.tensor(1.))
        parameter.grad = torch.tensor(7.)
        def suffix(hidden):
            return (hidden * parameter).reshape(1, 1, 2, 1, 1)
        result = solve_oracle(suffix, before, current, target, mask, oracle_config(**options))
        self.assertIsNone(before.grad)
        self.assertIsNone(current.grad)
        self.assertEqual(parameter.grad.item(), 7.)
        self.assertFalse(result.delta.requires_grad)
        self.assertIsNone(result.delta.grad_fn)
        self.assertFalse(result.scale.requires_grad)
        self.assertEqual(result.delta[0, 0, 0].item(), 0.)
        return result

    def test_existing_delta_is_part_of_complete_target(self):
        result = self.scalar_problem(inner_steps=1)
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.delta[0, 1, 0].item(), -3.)
        self.assertAlmostEqual(result.metrics["initial_fm"], 4.)
        self.assertAlmostEqual(result.metrics["final_fm"], 1.)

    def test_two_steps_accumulate_and_reduce_real_suffix_error(self):
        result = self.scalar_problem(inner_steps=2)
        self.assertEqual(result.metrics["accepted_steps"], 2)
        self.assertAlmostEqual(result.delta[0, 1, 0].item(), -4.)
        self.assertAlmostEqual(result.metrics["final_fm"], 0.)

    def test_backtracking_rejects_overshoot(self):
        result = self.scalar_problem(inner_steps=1, relative_step=1.)
        self.assertTrue(result.accepted)
        self.assertLess(result.metrics["final_relative_step"], 1.)
        self.assertLess(result.metrics["final_fm"], result.metrics["initial_fm"])
        self.assertGreater(result.metrics["suffix_forwards"], 2)

    def test_stationary_point_does_not_manufacture_nonzero_target(self):
        result = self.scalar_problem(target=8.)
        self.assertFalse(result.accepted)
        self.assertEqual(result.metrics["fm_gain"], 0.)

    def test_regularizer_cannot_accept_worse_fm(self):
        result = self.scalar_problem(target=8., regularization=1.)
        self.assertFalse(result.accepted)

    def test_site_rotation_is_explicit_and_deterministic(self):
        labels = ["A@5", "A@15", "A@25"]
        cfg = oracle_config(sites_per_micro=1)
        selected = [select_sites(labels, 0, m, 8, cfg)[0] for m in range(6)]
        self.assertEqual(selected, labels * 2)


class ArchitectureTests(unittest.TestCase):
    def test_production_native_width_contract_and_no_512(self):
        cfg = validate_config(read_config(ROOT / "configs/flow_oracle.yaml"))
        with torch.device("meta"):
            model = Corrector(cfg["corrector"])
        self.assertEqual(model.lift.skip.in_features, 1664)
        self.assertEqual(model.lift.skip.out_features, 3072)
        for site in model.A.values():
            self.assertEqual(site.to_jepa.skip.in_features, 3072)
            self.assertEqual(site.to_jepa.skip.out_features, 1664)
            self.assertEqual(site.writer.weight.shape, (3072, 3072))
            for block in site.blocks:
                for stream in (block.p, block.h):
                    self.assertEqual(stream.qkv.in_features, 3072)
                    self.assertEqual(stream.qkv.out_features, 3 * 3072)
        for module in model.modules():
            if isinstance(module, nn.Linear):
                self.assertNotIn(512, (module.in_features, module.out_features))
        for key in ("hidden_width", "mapping_width", "initializer_width"):
            bad = copy.deepcopy(cfg)
            bad["corrector"][key] = 512
            with self.assertRaises(ValueError):
                validate_config(bad)

    def test_reject_old_losses_and_future_gt(self):
        cfg, coords, model, sample, _, text, _ = fixture()
        for key in ("prior_weight", "write_weight", "out_weight"):
            bad = read_config(ROOT / "configs/flow_oracle.yaml")
            bad["loss"][key] = 0.
            with self.assertRaises(ValueError):
                validate_config(bad)
        with self.assertRaises(ValueError):
            model.corrector.initialize(sample["first"], text, coords, gt_video=sample["latent"])

    def test_zero_initialized_writers_preserve_base(self):
        _, coords, model, sample, noise, text, sigma = fixture()
        noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
        pred, state, info = model(noisy, sample["first"], text, sigma, coords)
        baseline, _, _ = model(noisy, sample["first"], text, sigma, coords, disable=("A",))
        torch.testing.assert_close(pred, baseline, rtol=0, atol=0)
        self.assertEqual(state.shape[-1], 16)
        self.assertTrue(all(s.shape[-1] == 12 for s in info["states"].values()))

    def test_suffix_replay_matches_actual_path_at_all_sites(self):
        _, coords, model, sample, noise, text, sigma = fixture()
        for site in model.corrector.A.values():
            nn.init.normal_(site.writer.weight, std=.03)
        noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
        with torch.no_grad():
            prediction, _, info = model(noisy, sample["first"], text, sigma, coords, capture="all")
            for snapshot in info["snapshots"].values():
                replay = model.suffix(snapshot, snapshot.before + snapshot.delta, text, sigma, coords)
                torch.testing.assert_close(replay, prediction, rtol=1e-6, atol=1e-6)
                self.assertEqual(float((snapshot.delta * (1 - snapshot.write_mask)).abs().sum()), 0.)
        self.assertFalse(torch.equal(info["snapshots"]["A@1"].before, info["snapshots"]["A@2"].before))

    def test_struct_reaches_lift_interaction_and_previous_writer(self):
        cfg, coords, model, sample, noise, text, sigma = fixture()
        for site in model.corrector.A.values():
            nn.init.normal_(site.writer.weight, std=.02)
        pred, _, info = model(noisy_view(sample["latent"], sample["first"], noise, sigma), sample["first"], text, sigma, coords)
        _, _, terms = training_loss(pred, sample["latent"], noise, sample["first"], sigma, info,
            sample["target"], sample["target_weight"], {}, 200, cfg)
        params = [model.corrector.lift.skip.weight,
                  model.corrector.A["1"].blocks[0].p.qkv.weight,
                  model.corrector.A["1"].writer.weight,
                  model.corrector.A["3"].to_jepa.skip.weight]
        gradients = torch.autograd.grad(terms["struct"], params, retain_graph=True)
        self.assertTrue(all(float(g.norm()) > 0 for g in gradients))
        # The final writer has no downstream STRUCT; its supervision is FM + Oracle.
        last, = torch.autograd.grad(terms["struct"], model.corrector.A["3"].writer.weight, allow_unused=True)
        self.assertIsNone(last)

    def test_fm_reaches_lift_and_p_stream_through_h(self):
        _, coords, model, sample, noise, text, sigma = fixture()
        for site in model.corrector.A.values():
            nn.init.normal_(site.writer.weight, std=.02)
        pred, _, _ = model(noisy_view(sample["latent"], sample["first"], noise, sigma), sample["first"], text, sigma, coords)
        fm = flow_mse(pred, flow_target(sample["latent"], noise))
        gradients = torch.autograd.grad(fm, [model.corrector.lift.skip.weight, model.corrector.A["3"].blocks[0].p.qkv.weight])
        self.assertTrue(all(float(g.norm()) > 0 for g in gradients))

    def test_oracle_with_checkpointed_suffix_and_full_joint_training(self):
        from train.train import training_micro
        cfg, coords, model, sample, noise, text, sigma = fixture()
        for site in model.corrector.A.values():
            nn.init.normal_(site.writer.weight, std=.01)
        originals = {n: p.detach().clone() for n, p in model.wan.named_parameters()}
        loss, metrics, terms = training_micro(model, sample, text, noise, sigma, cfg, 200, 0)
        self.assertEqual(set(terms), {"fm", "struct", "oracle"})
        self.assertGreater(metrics["oracle/acceptance"], 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.corrector.parameters()))
        optimizer = torch.optim.AdamW(model.corrector.parameters(), lr=1e-4)
        optimizer.step()
        for name, p in model.wan.named_parameters():
            self.assertIsNone(p.grad)
            torch.testing.assert_close(p, originals[name], rtol=0, atol=0)

    def test_checkpointed_and_uncheckpointed_gradients_agree(self):
        cfg, coords, model, sample, noise, text, sigma = fixture(True)
        _, _, other, _, _, _, _ = fixture(False)
        for site in model.corrector.A.values():
            nn.init.normal_(site.writer.weight, std=.02)
        other.load_state_dict(model.state_dict())
        for net in (model, other):
            pred, _, info = net(noisy_view(sample["latent"], sample["first"], noise, sigma), sample["first"], text, sigma, coords)
            loss, _, _ = training_loss(pred, sample["latent"], noise, sample["first"], sigma, info,
                sample["target"], sample["target_weight"], {}, 200, cfg)
            loss.backward()
        for p, q in zip(model.corrector.parameters(), other.corrector.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-5, atol=1e-6)

    def test_cpu_bf16_autocast_preserves_fp32_residual_path(self):
        _, coords, model, sample, noise, text, sigma = fixture()
        for site in model.corrector.A.values():
            nn.init.normal_(site.writer.weight, std=.001)
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, cache_enabled=False):
            pred, process, info = model(noisy_view(sample["latent"], sample["first"], noise, sigma),
                sample["first"], text, sigma, coords, capture="all")
        self.assertEqual(process.dtype, torch.float32)
        self.assertEqual(pred.dtype, torch.float32)
        self.assertTrue(all(delta.dtype == torch.float32 for delta in info["deltas"].values()))

    def test_inference_has_no_gt_or_oracle_and_preserves_first_frame(self):
        from inference.infer import sample_latents
        _, coords, model, sample, _, text, _ = fixture(channels=48)
        args = SimpleNamespace(mode="i2v", steps=2, shift=1., guidance=2., seed=42, device=0,
            solver="euler", variant="full", writer_off=[], reset_state=True)
        latent, counts = sample_latents(model, sample["first"], text, torch.zeros_like(text), coords, args)
        torch.testing.assert_close(latent[:, :, :1], sample["first"], rtol=0, atol=0)
        self.assertEqual(counts["wan_forwards"], 4)
        self.assertEqual(counts["scheduler_updates"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
