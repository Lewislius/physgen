"""Four-stage routing checks; optional real four-CUDA gradient test in an allocated job."""
from contextlib import nullcontext
from copy import deepcopy
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch

import test_static_lora as fixtures
from static_lora import ROOT
from static_lora.backbone import WanLoRA
from static_lora.checkpoint import checkpoint_path as single_checkpoint_path, load_training, save_checkpoint
from static_lora.config import configuration as single_config
from static_lora.data import ExposurePlan
from static_lora.lora import lora_named_parameters, lora_state_dict, load_lora_state_dict
from static_lora.optim import EMA, optimizer_for, restore_rng, set_schedule
from static_lora.runtime import code_fingerprint as single_fingerprint, digest, save_json
from four_gpu import backbone as parallel_backbone
from four_gpu.backbone import ParallelWanLoRA, block_layout
from four_gpu.config import code_fingerprint, configuration
from four_gpu.infer import checkpoint_path, implementation as inference_implementation
from four_gpu.runtime import require_four_gpus
from four_gpu.train import implementation as training_implementation


class FourGpuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def test_profile_preserves_bs8_and_all_training_hyperparameters(self):
        base, four = single_config(), configuration()
        for key in ("train", "lora", "data", "loss", "noise", "inference"):
            self.assertEqual(base[key], four[key])
        self.assertEqual(four["train"]["world_size"], 1)
        self.assertEqual(four["train"]["accumulation"], 8)
        self.assertEqual(four["execution"]["devices"], 4)
        for key in ("logs", "checkpoints", "outputs"):
            self.assertNotEqual(four["paths"][key], base["paths"][key])
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            path = Path(tmp) / "config.json"
            save_json(four, path)
            self.assertEqual(configuration(path), four)
        ordinary, extended = single_fingerprint(), code_fingerprint()
        self.assertFalse(any(name.startswith("four_gpu/") for name in ordinary))
        self.assertEqual({name: extended[name] for name in ordinary}, ordinary)
        self.assertIn("four_gpu/backbone.py", extended)

    def test_layout_covers_all_thirty_blocks(self):
        self.assertEqual(block_layout([8, 8, 7, 7], 30), [0] * 8 + [1] * 8 + [2] * 7 + [3] * 7)
        with self.assertRaises(ValueError):
            block_layout([8, 8, 8, 8], 30)

    def test_stage_hooks_cpu_offload_preserve_native_output_and_gradients(self):
        wan = fixtures.tiny_wan()
        standard = WanLoRA(deepcopy(wan), fixtures.spec(), recompute=True)
        parallel = ParallelWanLoRA(deepcopy(wan), fixtures.spec(), ["cpu"] * 4, [1, 1, 0, 0],
                                   recompute=True, activation_offload=True)
        for name, p in lora_named_parameters(standard):
            if name.endswith("lora_B"):
                torch.nn.init.normal_(p, std=.03)
        load_lora_state_dict(parallel, lora_state_dict(standard))
        source = fixtures.sample()
        with patch.object(fixtures.backbone, "autocast", side_effect=lambda device: nullcontext()):
            for mode in ("i2v", "t2v"):
                standard.zero_grad(set_to_none=True)
                parallel.zero_grad(set_to_none=True)
                first = source["first"] if mode == "i2v" else None
                a = standard(source["latent"], first, source["text"], torch.tensor(.4))
                b = parallel(source["latent"], first, source["text"], torch.tensor(.4))
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.square().mean().backward()
                b.square().mean().backward()
                for (name, p), (other, q) in zip(lora_named_parameters(standard), lora_named_parameters(parallel)):
                    self.assertEqual(name, other)
                    torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-8)
                self.assertEqual(parallel._stage_kwargs, {})
        self.assertEqual(set(lora_state_dict(standard)), set(lora_state_dict(parallel)))

    def test_conditioning_cache_never_survives_a_forward(self):
        wan = fixtures.tiny_wan()
        standard = WanLoRA(deepcopy(wan), fixtures.spec(), recompute=False)
        parallel = ParallelWanLoRA(deepcopy(wan), fixtures.spec(), ["cpu"] * 4, [1, 1, 0, 0],
                                   recompute=False, activation_offload=False)
        load_lora_state_dict(parallel, lora_state_dict(standard))
        source = fixtures.sample()
        with torch.no_grad(), patch.object(fixtures.backbone, "autocast", side_effect=lambda device: nullcontext()):
            for sigma, sign in ((.1, 1.), (.9, -1.), (.5, 2.)):
                text = sign * source["text"]
                expected = standard(source["latent"], None, text, torch.tensor(sigma))
                actual = parallel(source["latent"], None, text, torch.tensor(sigma))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(parallel._stage_kwargs, {})

    def test_loader_starts_from_cpu_instead_of_a_full_model_on_gpu0(self):
        placement = {f"cuda:{i}": dict(parameters=2, bytes=8, trainable_parameters=1) for i in range(4)}
        with patch.object(parallel_backbone, "load_wan") as base_loader, \
                patch.object(parallel_backbone, "ParallelWanLoRA") as constructor, \
                patch.object(parallel_backbone, "lora_named_parameters", return_value=[("fixture", torch.ones(2))]), \
                patch.object(parallel_backbone, "log"), patch.object(torch.cuda, "reset_peak_memory_stats"):
            constructor.return_value.placement.return_value = placement
            parallel_backbone.load_model(configuration(), torch.device("cuda", 0))
            self.assertEqual(base_loader.call_args.args[1], torch.device("cpu"))
            self.assertEqual(constructor.call_args.args[2], [torch.device("cuda", i) for i in range(4)])
            self.assertEqual(constructor.call_args.args[3], [8, 8, 7, 7])
            self.assertTrue(constructor.call_args.kwargs["activation_offload"])

    def test_wrappers_do_not_patch_single_gpu_module_globals(self):
        training, inference = training_implementation(), inference_implementation()
        self.assertIs(training.load_model, parallel_backbone.load_model)
        self.assertIs(inference.load_model, parallel_backbone.load_model)
        self.assertIs(fixtures.training.configuration, single_config)
        self.assertIs(fixtures.inference.configuration, single_config)
        self.assertIs(fixtures.training.load_model, fixtures.backbone.load_model)

    def test_four_gpu_checkpoint_config_resume_and_inference_identity(self):
        cfg = configuration()
        records = [dict(category="unlabeled") for _ in range(17)]
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
            with patch("four_gpu.infer.lock_assets", return_value=assets):
                self.assertEqual(checkpoint_path(cfg, saved), saved)
            with self.assertRaisesRegex(ValueError, "Inference source differs"):
                single_checkpoint_path(cfg, saved)
            restored = fixtures.SmallState()
            opt, queue = optimizer_for(restored, cfg), ExposurePlan(records, cfg["train"]["seed"])
            completed, average, rng = load_training(saved, restored, opt, queue, cfg, identity)
            self.assertEqual(completed, 100)
            self.assertEqual(average.updates, 100)
            self.assertEqual(queue.batch(101), plan.batch(101))
            for name, value in lora_state_dict(model).items():
                torch.testing.assert_close(value, lora_state_dict(restored)[name], rtol=0, atol=0)

    def test_wrong_launch_shape_fails_before_loading_model(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            with self.assertRaisesRegex(RuntimeError, "one Python process"):
                require_four_gpus()
        with patch.dict(os.environ, {"WORLD_SIZE": "1", "DET_SLOT_IDS": "[0]"}):
            with self.assertRaisesRegex(RuntimeError, "slots_per_trial must be 4"):
                require_four_gpus()

    @unittest.skipUnless(os.environ.get("RUN_FOUR_GPU_TEST") == "1", "Set RUN_FOUR_GPU_TEST=1 inside a four-GPU allocation")
    def test_actual_four_cuda_devices_preserve_outputs_gradients_and_clipping(self):
        require_four_gpus()
        wan = fixtures.NativeWan(model_type="ti2v", patch_size=(1, 2, 2), text_len=8, in_dim=4, dim=16,
                                 ffn_dim=32, freq_dim=8, text_dim=8, out_dim=4, num_heads=2, num_layers=4)
        torch.nn.init.normal_(wan.head.head.weight, std=.1)
        standard = WanLoRA(deepcopy(wan).to("cuda:0"), fixtures.spec(), recompute=True)
        parallel = ParallelWanLoRA(deepcopy(wan), fixtures.spec(), [f"cuda:{i}" for i in range(4)], [1, 1, 1, 1],
                                   recompute=True, activation_offload=True)
        for name, p in lora_named_parameters(standard):
            if name.endswith("lora_B"):
                torch.nn.init.normal_(p, std=.03)
        load_lora_state_dict(parallel, lora_state_dict(standard))
        source = fixtures.sample()
        x, text, first = (source[key].to("cuda:0") for key in ("latent", "text", "first"))
        a = standard(x, first, text, torch.tensor(.4, device="cuda:0"))
        b = parallel(x, first, text, torch.tensor(.4, device="cuda:0"))
        torch.testing.assert_close(a, b, rtol=1e-3, atol=1e-4)
        a.square().mean().backward()
        b.square().mean().backward()
        p1, p4 = [p for _, p in lora_named_parameters(standard)], [p for _, p in lora_named_parameters(parallel)]
        self.assertEqual({str(p.device) for p in p4}, {f"cuda:{i}" for i in range(4)})
        for p, q in zip(p1, p4):
            torch.testing.assert_close(p.grad.cpu(), q.grad.cpu(), rtol=.02, atol=1e-4)
        torch.testing.assert_close(torch.nn.utils.clip_grad_norm_(p1, 1.), torch.nn.utils.clip_grad_norm_(p4, 1.), rtol=.02, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
