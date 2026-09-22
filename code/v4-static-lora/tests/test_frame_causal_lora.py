"""Numerical frame-mask, native Wan gradient, launch and checkpoint regressions."""
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
import yaml

from static_lora import ROOT
from static_lora.backbone import WanLoRA
from static_lora.checkpoint import inspect_checkpoint as baseline_inspect
from static_lora.config import configuration as baseline_configuration
from static_lora.data import ExposurePlan
from static_lora.lora import assert_lora_only, load_lora_state_dict, lora_named_parameters, lora_state_dict
from static_lora.optim import EMA, optimizer_for, set_schedule
from static_lora.runtime import code_fingerprint as baseline_fingerprint, digest, read_json, save_json
from four_gpu.backbone import ParallelWanLoRA
from four_gpu.runtime import require_four_gpus
from frame_causal_lora.attention import frame_attention, install_frame_attention, validate_attention
from frame_causal_lora import backbone as causal_backbone
from frame_causal_lora.checkpoint import checkpoint_path, inspect_checkpoint, load_training, save_checkpoint
from frame_causal_lora.config import code_fingerprint, configuration, variant_name
from frame_causal_lora.train import implementation as training_implementation, run_name
from frame_causal_lora.infer import implementation as inference_implementation
from frame_causal_lora import infer as causal_inference

sys.path.insert(0, str(ROOT / "tests"))
import test_static_lora as fixtures


def config_path(mode="full", profile="1x96g"):
    return ROOT / f"configs/train_frame_causal_{mode}_{profile}.yaml"


def spec(previous):
    return dict(mode="frame_causal_full" if previous is None else "frame_causal_window", previous_frames=previous)


def sdpa(q, k, v, **kwargs):
    # Never let a token-triangular or token-window shortcut pass these tests.
    assert kwargs == dict(causal=False, window_size=(-1, -1))
    return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2), is_causal=False).transpose(1, 2)


def dense_oracle(q, k, v, spatial, previous):
    frame = torch.arange(q.shape[1], device=q.device) // spatial
    allowed = frame[:, None] >= frame[None, :]
    if previous is not None:
        allowed &= frame[:, None] - frame[None, :] <= previous
    scores = q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / q.shape[-1] ** .5
    weights = scores.masked_fill(~allowed, -torch.inf).softmax(-1)
    return (weights @ v.transpose(1, 2)).transpose(1, 2)


class FrameAttentionTests(unittest.TestCase):
    def test_forward_and_qkv_gradients_match_independent_dense_frame_mask(self):
        torch.manual_seed(54)
        for frames, height, width in ((1, 2, 3), (3, 1, 1), (8, 2, 3)):
            length = frames * height * width
            for previous in (None, 0, 1, 5, 20):
                with self.subTest(grid=(frames, height, width), previous=previous):
                    values = [torch.randn(1, length, 2, 4, dtype=torch.float64, requires_grad=True) for _ in range(3)]
                    reference = [x.detach().clone().requires_grad_() for x in values]
                    actual = frame_attention(*values, torch.tensor([[frames, height, width]]),
                                             torch.tensor([length]), previous, sdpa)
                    expected = dense_oracle(*reference, height * width, previous)
                    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)
                    weight = torch.randn_like(actual)
                    (actual * weight).sum().backward()
                    (expected * weight).sum().backward()
                    for x, y in zip(values, reference):
                        torch.testing.assert_close(x.grad, y.grad, rtol=1e-9, atol=1e-11)

    def test_frame_is_global_and_k5_means_six_complete_frames(self):
        # Uniform attention makes every permitted value contribute equally.
        q = torch.zeros(1, 24, 1, 1, dtype=torch.float64)
        v = torch.arange(24, dtype=torch.float64).reshape(1, 24, 1, 1).requires_grad_()
        grid, lengths = torch.tensor([[8, 1, 3]]), torch.tensor([24])
        out = frame_attention(q, q, v, grid, lengths, 5, sdpa)
        torch.testing.assert_close(out[0, :3, 0, 0], torch.ones(3, dtype=torch.float64))
        # Frame 7 sees frames 2..7: spatial tokens 6..23, including its last token.
        self.assertAlmostEqual(float(out[0, 21].detach()), 14.5)
        out[0, 21].sum().backward()
        self.assertEqual(torch.count_nonzero(v.grad[0, :6]).item(), 0)
        torch.testing.assert_close(v.grad[0, 6:], torch.full_like(v.grad[0, 6:], 1 / 18))
        full = frame_attention(q, q, v.detach(), grid, lengths, None, sdpa)
        self.assertAlmostEqual(float(full[0, 21]), 11.5)

    def test_single_attention_layer_cannot_see_future_or_frames_outside_window(self):
        wan = fixtures.tiny_wan()
        install_frame_attention(wan, spec(5))
        layer = wan.blocks[0].self_attn
        x = torch.randn(1, 9 * 4, 16, requires_grad=True)
        out = layer(x, torch.tensor([36]), torch.tensor([[9, 2, 2]]), wan.freqs)
        out[:, 6 * 4].square().sum().backward()
        # Frame 6: frames 1..6 are allowed, including same-frame token 27.
        self.assertEqual(torch.count_nonzero(x.grad[:, :4]).item(), 0)
        self.assertEqual(torch.count_nonzero(x.grad[:, 7 * 4:]).item(), 0)
        for frame in range(1, 7):
            self.assertGreater(float(x.grad[:, frame * 4:(frame + 1) * 4].abs().sum()), 0.)
        self.assertGreater(float(x.grad[:, 27].abs().sum()), 0.)

    def test_rejects_padding_bad_window_and_temporal_patches(self):
        for attention in (spec(-1), spec(True), spec(1.5), dict(mode="token_causal", previous_frames=5),
                          dict(mode="frame_causal_full", previous_frames=5)):
            with self.assertRaises(ValueError):
                validate_attention(attention)
        q = torch.zeros(1, 12, 2, 4)
        for grid, length in (([[2, 2, 2]], [8]), ([[3, 2, 2]], [11])):
            with self.assertRaisesRegex(ValueError, "exactly equal"):
                frame_attention(q, q, q, torch.tensor(grid), torch.tensor(length), 5, sdpa)
        wan = fixtures.tiny_wan()
        wan.patch_size = (2, 2, 2)
        with self.assertRaisesRegex(ValueError, "one compressed"):
            install_frame_attention(wan, spec(None))

    def test_single_frame_matches_native_and_cross_attention_is_unchanged(self):
        for previous in (None, 5):
            original = WanLoRA(fixtures.tiny_wan(), fixtures.spec(), recompute=False)
            causal = deepcopy(original)
            names = list(dict(causal.named_parameters()))
            cross = [block.cross_attn.forward.__func__ for block in causal.wan.blocks]
            install_frame_attention(causal.wan, spec(previous))
            self.assertEqual(names, list(dict(causal.named_parameters())))
            self.assertEqual(cross, [block.cross_attn.forward.__func__ for block in causal.wan.blocks])
            self.assertEqual(assert_lora_only(original), assert_lora_only(causal))
            sample = fixtures.sample()
            with patch.object(fixtures.backbone, "autocast", side_effect=lambda _: nullcontext()):
                for first in (None, sample["first"]):
                    args = sample["latent"][:, :, :1], first, sample["text"], torch.tensor(.4)
                    torch.testing.assert_close(causal(*args), original(*args), rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "already installed"):
                install_frame_attention(causal.wan, spec(previous))

    def test_full_wan_predictions_cannot_depend_on_future_frames(self):
        for previous in (None, 5):
            model = WanLoRA(fixtures.tiny_wan(), fixtures.spec(), recompute=True)
            install_frame_attention(model.wan, spec(previous))
            source = fixtures.sample()
            x = source["latent"].clone().requires_grad_()
            with patch.object(fixtures.backbone, "autocast", side_effect=lambda _: nullcontext()):
                model(x, None, source["text"], torch.tensor(.4))[:, :, 1].square().sum().backward()
            self.assertEqual(torch.count_nonzero(x.grad[:, :, 2:]).item(), 0)
            self.assertGreater(float(x.grad[:, :, :2].abs().sum()), 0.)

    def test_recompute_and_four_stage_cpu_offload_preserve_lora_gradients(self):
        for previous in (None, 5):
            models = [WanLoRA(fixtures.tiny_wan(), fixtures.spec(), recompute=False),
                      WanLoRA(fixtures.tiny_wan(), fixtures.spec(), recompute=True),
                      ParallelWanLoRA(fixtures.tiny_wan(), fixtures.spec(), ["cpu"] * 4, [1, 1, 0, 0],
                                      recompute=True, activation_offload=True)]
            for name, p in lora_named_parameters(models[0]):
                if name.endswith("lora_B"):
                    torch.nn.init.normal_(p, std=.03)
            for model in models:
                load_lora_state_dict(model, lora_state_dict(models[0]))
                install_frame_attention(model.wan, spec(previous))
            source = fixtures.sample()
            source["latent"] = torch.randn(1, 4, 8, 4, 4)
            for first in (None, source["first"]):
                outputs = []
                with patch.object(fixtures.backbone, "autocast", side_effect=lambda _: nullcontext()):
                    for model in models:
                        model.zero_grad(set_to_none=True)
                        out = model(source["latent"], first, source["text"], torch.tensor(.4))
                        out.square().mean().backward()
                        outputs.append(out.detach())
                for i in (1, 2):
                    torch.testing.assert_close(outputs[i], outputs[0], rtol=0, atol=0)
                    for (name, p), (other, q) in zip(lora_named_parameters(models[0]), lora_named_parameters(models[i])):
                        self.assertEqual(name, other)
                        torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-8)
                self.assertEqual(models[2]._stage_kwargs, {})

    @unittest.skipUnless(os.environ.get("RUN_FRAME_CAUSAL_CUDA_TEST") == "1", "Needs an allocated CUDA GPU")
    def test_native_flash_attention_forward_and_backward_on_cuda(self):
        # Execute the installed Wan CUDA kernel rather than the CPU test substitute.
        path = Path(baseline_configuration()["paths"]["wan_code"]) / "wan/modules/attention.py"
        native = fixtures.import_file("frame_causal_cuda_attention", path)
        torch.manual_seed(25)
        for previous in (None, 5):
            values = [torch.randn(1, 8 * 6, 2, 64, device="cuda", dtype=torch.bfloat16).float().requires_grad_()
                      for _ in range(3)]
            reference = [x.detach().clone().requires_grad_() for x in values]
            actual = frame_attention(*values, torch.tensor([[8, 2, 3]]), torch.tensor([48]), previous,
                                     native.flash_attention)
            expected = dense_oracle(*reference, 6, previous)
            torch.testing.assert_close(actual, expected, rtol=.04, atol=.02)
            weight = torch.randn_like(actual)
            (actual * weight).sum().backward()
            (expected * weight).sum().backward()
            for x, y in zip(values, reference):
                torch.testing.assert_close(x.grad, y.grad, rtol=.06, atol=.04)


class FrameRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def test_recipes_preserve_training_and_roundtrip_and_separate_all_outputs(self):
        base = baseline_configuration()
        paths = set()
        for mode in ("full", "window"):
            for profile in ("1x96g", "4x48g"):
                cfg = configuration(config_path(mode, profile))
                for key in ("train", "lora", "data", "loss", "noise", "inference"):
                    self.assertEqual(cfg[key], base[key])
                self.assertIn(variant_name(cfg["attention"]), cfg["model_version"])
                for key in ("checkpoints", "logs", "outputs"):
                    self.assertNotIn(cfg["paths"][key], paths)
                    paths.add(cfg["paths"][key])
                with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
                    path = Path(tmp) / "config.json"
                    save_json(cfg, path)
                    self.assertEqual(configuration(path), cfg)
                    changed = deepcopy(cfg)
                    changed["attention"]["previous_frames"] = 4
                    save_json(changed, path)
                    with self.assertRaisesRegex(ValueError, "differs"):
                        configuration(path)
        base_hashes, extended = baseline_fingerprint(), code_fingerprint()
        self.assertFalse(any(name.startswith("frame_causal_lora/") for name in base_hashes))
        self.assertEqual({name: extended[name] for name in base_hashes}, base_hashes)

    def test_custom_k_changes_identity_directories_and_name(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            raw = yaml.safe_load(config_path("window").read_text())
            raw["attention"]["previous_frames"] = 3
            path = Path(tmp) / "custom.yaml"
            path.write_text(yaml.safe_dump(raw))
            cfg = configuration(path)
            self.assertIn("window_k3", cfg["model_version"])
            self.assertIn("window_k3", cfg["paths"]["checkpoints"])
            self.assertIn("window_k3", run_name(cfg))

    def test_run_names_always_include_version_profile_utc_timestamp_and_unique_suffix(self):
        for mode in ("full", "window"):
            for profile in ("1x96g", "4x48g"):
                cfg = configuration(config_path(mode, profile))
                prefix = re.escape(variant_name(cfg["attention"]) + "_" + profile)
                self.assertRegex(run_name(cfg), prefix + r"_\d{8}T\d{6}Z_[0-9a-f]{6}$")
                self.assertRegex(run_name(cfg, "trial-a"), prefix + r"_\d{8}T\d{6}Z_[0-9a-f]{6}_trial-a$")
                self.assertNotEqual(run_name(cfg), run_name(cfg))
                with self.assertRaises(ValueError):
                    run_name(cfg, "../escape")

    def test_train_and_inference_loaders_use_saved_policy_without_changing_baseline_globals(self):
        for profile in ("1x96g", "4x48g"):
            cfg = configuration(config_path("window", profile))
            for training in (True, False):
                loader = "four_load_model" if profile == "4x48g" else "single_load_model"
                model = WanLoRA(fixtures.tiny_wan(), fixtures.spec(), recompute=training)
                with patch.object(causal_backbone, loader, return_value=model) as load, patch.object(causal_backbone, "log"):
                    causal_backbone.load_model(cfg, torch.device("cpu"), training=training)
                    load.assert_called_once_with(cfg, torch.device("cpu"), training=training)
                self.assertTrue(all(b.self_attn._frame_attention_spec == cfg["attention"] for b in model.wan.blocks))
            train = training_implementation(config_path("window", profile))
            infer = inference_implementation(config_path("window", profile))
            self.assertIs(train.load_model, causal_backbone.load_model)
            self.assertIs(infer.load_model, causal_backbone.load_model)
            self.assertIs(train.save_checkpoint, save_checkpoint)
            if profile == "4x48g":
                self.assertIs(train.require_single_gpu, require_four_gpus)
                self.assertIs(infer.require_single_gpu, require_four_gpus)
        self.assertIs(fixtures.training.configuration, baseline_configuration)
        self.assertIs(fixtures.training.load_model, fixtures.backbone.load_model)

    def test_inference_sample_and_report_select_profile_from_prepared_cases(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            root = Path(tmp)
            checkpoint = root / "saved"
            checkpoint.mkdir()
            save_json(configuration(config_path("window", "4x48g")), checkpoint / "config.json")
            save_json(dict(checkpoint=str(checkpoint)), root / "cases.json")
            for phase in ("sample", "report"):
                with patch.object(sys, "argv", ["infer.py", "--phase", phase, "--output", str(root)]), \
                        patch.object(causal_inference, "implementation") as implementation:
                    causal_inference.main()
                    implementation.assert_called_once_with(checkpoint / "config.json")
                    implementation.return_value.main.assert_called_once()

    def test_checkpoint_resume_ema_and_rejection_of_other_attention_or_profile(self):
        records = [dict(category="unlabeled") for _ in range(17)]
        for mode in ("full", "window"):
            for profile in ("1x96g", "4x48g"):
                cfg = configuration(config_path(mode, profile))
                model = fixtures.SmallState()
                optimizer, ema = optimizer_for(model, cfg), EMA(model)
                plan = ExposurePlan(records, cfg["train"]["seed"])
                assets = dict(fixture=True)
                identity = dict(code=digest(code_fingerprint()), assets=digest(assets), manifest="fixture")
                for step in range(1, 101):
                    plan.batch(step)
                    set_schedule(optimizer, cfg, step)
                    optimizer.zero_grad(set_to_none=True)
                    sum(p.square().mean() for _, p in lora_named_parameters(model)).backward()
                    optimizer.step()
                    ema.update(model)
                with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
                    saved = save_checkpoint(tmp, 100, model, optimizer, ema, plan, cfg, identity)
                    self.assertEqual(configuration(saved / "config.json"), cfg)
                    self.assertEqual(read_json(saved / "complete.json")["version"], cfg["model_version"])
                    restored = fixtures.SmallState()
                    opt, queue = optimizer_for(restored, cfg), ExposurePlan(records, cfg["train"]["seed"])
                    completed, average, rng = load_training(saved, restored, opt, queue, cfg, identity)
                    self.assertEqual((completed, average.updates), (100, 100))
                    self.assertIn("torch", rng)
                    self.assertEqual(queue.batch(101), plan.batch(101))
                    for name, value in lora_state_dict(model).items():
                        torch.testing.assert_close(value, lora_state_dict(restored)[name], rtol=0, atol=0)
                    with self.assertRaises(ValueError):
                        baseline_inspect(saved, baseline_configuration())
                    for other in (configuration(config_path("window" if mode == "full" else "full", profile)),
                                  configuration(config_path(mode, "4x48g" if profile == "1x96g" else "1x96g"))):
                        with self.assertRaises(ValueError):
                            inspect_checkpoint(saved, other)
                    if mode == "window":
                        changed_k = deepcopy(cfg)
                        changed_k["attention"]["previous_frames"] = 4
                        with self.assertRaisesRegex(ValueError, "configuration differs"):
                            inspect_checkpoint(saved, changed_k)
                    changed_code = dict(identity, code="other-source")
                    with self.assertRaisesRegex(ValueError, "identity differs"):
                        inspect_checkpoint(saved, cfg, changed_code)
                    # Inference needs only EMA and configuration, not optimizer or raw LoRA.
                    (saved / "training.pt").unlink()
                    (saved / "lora.pt").unlink()
                    with patch("frame_causal_lora.checkpoint.lock_assets", return_value=assets):
                        self.assertEqual(checkpoint_path(cfg, saved), saved)

    def test_shells_and_determined_yaml_match_all_four_profiles(self):
        for mode in ("full", "window"):
            for profile in ("1x96g", "4x48g"):
                basename = f"train_frame_causal_{mode}_{profile}"
                document = yaml.safe_load((ROOT / "train" / (basename + ".yaml")).read_text())
                self.assertEqual(document["resources"]["slots_per_trial"], 4 if profile == "4x48g" else 1)
                self.assertEqual(document["entrypoint"], "bash " + str(ROOT / "train" / (basename + ".sh")))
                environment = dict(os.environ, PARSE_ONLY="1", PREPARE="1", RUN_NAME="unit-label",
                                   RESUME="", CONFIG="")
                result = subprocess.run(["bash", str(ROOT / "train" / (basename + ".sh"))],
                                        env=environment, capture_output=True, text=True, check=True)
                lines = result.stdout.splitlines()
                self.assertEqual(len(lines), 3)
                self.assertTrue(all("frame_causal_lora/train.py" in line and str(config_path(mode, profile)) in line
                                    for line in lines))
                self.assertIn("--check-gpu", lines[0])
                self.assertIn("--prepare-only", lines[1])
                self.assertIn("--run-name unit-label", lines[2])
                self.assertNotIn("torchrun", result.stdout)


if __name__ == "__main__":
    unittest.main()
