"""CPU checks with the real native Wan class at tiny size and SDPA attention."""
from contextlib import nullcontext
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from static_lora import ROOT, V42_ROOT, TRAINING_RECIPE
from static_lora import backbone
from static_lora.backbone import WanLoRA, enable_checkpointing
from static_lora.checkpoint import inspect_checkpoint, load_training, save_checkpoint
from static_lora.config import configuration
from static_lora.data import Dataset, ExposurePlan
from static_lora.lora import (LoRALinear, assert_lora_only, inject_lora, lora_named_parameters,
                              lora_state_dict, load_lora_state_dict)
from static_lora.optim import EMA, optimizer_for, restore_rng, set_schedule
from static_lora.runtime import code_fingerprint, digest, file_identity, read_json, save_json, save_tensor
from physgen_v42.data import ExposurePlan as ReferencePlan


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


training = import_file("static_lora_test_training", ROOT / "train/train.py")
inference = import_file("static_lora_test_inference", ROOT / "inference/infer.py")


def load_native_class():
    # Avoid wan.__init__ GPU/distributed imports. Execute the actual model.py;
    # only substitute its CUDA-only attention kernel for a tiny CPU SDPA kernel.
    package = ModuleType("static_lora_test_native")
    package.__path__ = []
    attention = ModuleType("static_lora_test_native.attention")
    def sdpa(q, k, v, **kwargs):
        return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                             v.transpose(1, 2)).transpose(1, 2).contiguous()
    attention.flash_attention = sdpa
    sys.modules[package.__name__] = package
    sys.modules[attention.__name__] = attention
    path = Path(configuration()["paths"]["wan_code"]) / "wan/modules/model.py"
    return import_file(package.__name__ + ".model", path).WanModel


NativeWan = load_native_class()


def spec(rank=2):
    value = deepcopy(configuration()["lora"])
    value.update(rank=rank, alpha=rank)
    return value


def tiny_wan():
    torch.manual_seed(33)
    wan = NativeWan(model_type="ti2v", patch_size=(1, 2, 2), text_len=8, in_dim=4, dim=16,
                    ffn_dim=32, freq_dim=8, text_dim=8, out_dim=4, num_heads=2, num_layers=2)
    torch.nn.init.normal_(wan.head.head.weight, std=.1)
    return wan.eval()


def sample():
    torch.manual_seed(44)
    latent = torch.randn(1, 4, 3, 4, 4)
    return dict(latent=latent, first=latent[:, :, :1].clone(), text=torch.randn(1, 4, 8),
                record=dict(id="source0000001", duration=.5))


class SmallState(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = LoRALinear(nn.Linear(3, 3), 2, 2)


class LoRATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (ROOT / "tmp").mkdir(exist_ok=True)

    def test_configuration_matches_formal_data_schedule_and_saved_json(self):
        cfg = configuration()
        reference = read_json(ROOT / "configs/v42_reference.json")
        self.assertEqual(cfg["data"], reference["data"])
        self.assertEqual(cfg["inference"], reference["inference"])
        for key in ("seed", "steps", "world_size", "accumulation", "save_every", "ema_decay", "grad_clip"):
            self.assertEqual(cfg["train"][key], reference["train"][key])
        self.assertEqual(cfg["train"]["peak_lr"], reference["train"]["peak_lr"]["core5"])
        self.assertEqual(cfg["loss"], {"fm": 1.})
        self.assertNotIn("state", cfg)
        self.assertFalse(cfg["noise"]["repair"])
        for key in ("wan_checkpoint", "cache", "annotations", "asset_lock"):
            self.assertEqual(cfg["paths"][key], reference["paths"][key])
        for key in ("checkpoints", "logs", "outputs"):
            self.assertTrue(Path(cfg["paths"][key]).is_relative_to(ROOT))
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            path = Path(tmp) / "config.json"
            save_json(cfg, path)
            self.assertEqual(configuration(path), cfg)

    def test_full_5b_meta_model_has_only_240_lora_targets_and_47185920_trainables(self):
        cfg = configuration()
        original = read_json(Path(cfg["paths"]["wan_checkpoint"]) / "config.json")
        with torch.device("meta"):
            wan = NativeWan(**{k: v for k, v in original.items() if not k.startswith("_")})
            names = inject_lora(wan, cfg["lora"])
        self.assertEqual(len(names), 240)
        self.assertEqual(assert_lora_only(wan), 47_185_920)
        self.assertEqual(len(dict(lora_named_parameters(wan))), 480)
        self.assertTrue(all(not p.requires_grad for name, p in wan.named_parameters() if "lora_" not in name))

    def test_zero_lora_is_exactly_native_wan_in_both_modes(self):
        base = tiny_wan()
        model = WanLoRA(deepcopy(base), spec(), recompute=False)
        source = sample()
        for mode in ("t2v", "i2v"):
            first = None if mode == "t2v" else source["first"]
            noisy = source["latent"].clone()
            expected_x = noisy.clone()
            expected_t = torch.full((1, 12), 300.)
            if first is not None:
                expected_x[:, :, :1] = first
                expected_t[:, :4] = 0.
            expected = base([expected_x[0]], expected_t, [source["text"][0]], seq_len=12)[0][None]
            with patch.object(backbone, "autocast", side_effect=lambda device: nullcontext()):
                actual = model(noisy, first, source["text"], torch.tensor(.3))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_fm_updates_only_lora_and_trains_a_after_b_leaves_zero(self):
        cfg = configuration()
        model = WanLoRA(tiny_wan(), spec(), recompute=False)
        optimizer = optimizer_for(model, cfg)
        frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
        source = sample()
        exposure = dict(low_noise=False)
        with patch.object(backbone, "autocast", side_effect=lambda device: nullcontext()):
            for step in (1, 2):
                optimizer.zero_grad(set_to_none=True)
                set_schedule(optimizer, cfg, step)
                values = training.train_micro(model, source, exposure, cfg)
                self.assertGreater(values["fm"], 0.)
                self.assertAlmostEqual(values["weighted_fm"], values["fm"] / 8, places=6)
                groups = {suffix: [p for name, p in lora_named_parameters(model) if name.endswith(suffix)]
                          for suffix in ("lora_A", "lora_B")}
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in groups["lora_B"]), 0.)
                a_gradient = sum(float(p.grad.abs().sum()) for p in groups["lora_A"])
                self.assertEqual(a_gradient, 0.) if step == 1 else self.assertGreater(a_gradient, 0.)
                optimizer.step()
        for name, p in model.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad)
                torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)

    def test_checkpointed_native_blocks_preserve_output_and_parameter_gradients(self):
        plain = WanLoRA(tiny_wan(), spec(), recompute=False)
        for name, p in lora_named_parameters(plain):
            if name.endswith("lora_B"):
                torch.nn.init.normal_(p, std=.01)
        recompute = deepcopy(plain)
        enable_checkpointing(recompute.wan)
        source = sample()
        outputs = []
        with patch.object(backbone, "autocast", side_effect=lambda device: nullcontext()):
            for model in (plain, recompute):
                out = model(source["latent"], source["first"], source["text"], torch.tensor(.6))
                outputs.append(out.detach())
                out.square().mean().backward()
        torch.testing.assert_close(*outputs, rtol=0, atol=0)
        for (name, p), (other_name, q) in zip(lora_named_parameters(plain), lora_named_parameters(recompute)):
            self.assertEqual(name, other_name)
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-6, atol=1e-8)
        self.assertGreater(float(recompute.wan.blocks[0].self_attn.q.lora_B.grad.abs().sum()), 0.)

    def test_accumulated_fm_gradient_is_the_mean_of_eight_sources(self):
        cfg = configuration()
        model = WanLoRA(tiny_wan(), spec(), recompute=False)
        direct = deepcopy(model)
        source = sample()
        source["first"] = None
        tensors, ranges = [], []
        original_draw = torch.randn_like
        def noise(x):
            value = original_draw(x)
            tensors.append(value.clone())
            return value
        def uniform(tensor, low, high):
            ranges.append((low, high))
            return tensor.fill_(.1 if high == .15 else .7)
        with patch.object(backbone, "autocast", side_effect=lambda device: nullcontext()), \
                patch.object(torch, "randn_like", side_effect=noise), \
                patch.object(torch.Tensor, "uniform_", new=uniform):
            values = [training.train_micro(model, source, dict(low_noise=i == 7), cfg) for i in range(8)]
        losses = []
        with patch.object(backbone, "autocast", side_effect=lambda device: nullcontext()):
            for i, noise_value in enumerate(tensors):
                sigma = torch.tensor(.1 if i == 7 else .7)
                x = (1 - sigma) * source["latent"] + sigma * noise_value
                prediction = direct(x, None, source["text"], sigma)
                losses.append((prediction - (noise_value - source["latent"])).square().mean())
            torch.stack(losses).mean().backward()
        self.assertEqual(ranges, [(.02, .999)] * 7 + [(.05, .15)])
        self.assertTrue(all(not row["repair"] for row in values))
        for (_, p), (_, q) in zip(lora_named_parameters(model), lora_named_parameters(direct)):
            torch.testing.assert_close(p.grad, q.grad, rtol=2e-5, atol=1e-7)

    def test_source_mode_noise_queues_match_formal_v42_for_all_1200_steps(self):
        records = [dict(category="unlabeled") for _ in range(2279)]
        plan, reference = ExposurePlan(records, 20260916), ReferencePlan(records, 20260916)
        for step in range(1, 1201):
            actual, original = plan.batch(step), reference.batch(step)
            self.assertEqual(len(actual), 8)
            self.assertEqual(sum(e["mode"] == "i2v" for e in actual), 6)
            self.assertEqual(sum(e["low_noise"] for e in actual), 1)
            for a, b in zip(actual, original):
                self.assertEqual((a["index"], a["mode"], a["exposure"], a["low_noise"]),
                                 (b["index"], b["mode"], b["exposure"], b["temporal"]))
                self.assertNotIn("repair", a)
                self.assertNotIn("temporal", a)
        plan.validate_state(plan.state_dict(), 1200)
        self.assertEqual(sum(c["i2v"] + c["t2v"] for c in plan.state_dict()["counts"]), 9600)

    def test_dataset_loads_only_vae_and_text_no_teacher_or_anchor(self):
        source = sample()
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            root = Path(tmp)
            save_tensor(dict(latent=source["latent"], first=source["first"]), root / "vae/source0000001.pt")
            save_tensor(source["text"][0], root / "text/source0000001.pt")
            dataset = Dataset.__new__(Dataset)
            dataset.root, dataset.records = root, [source["record"]]
            for mode in ("i2v", "t2v"):
                with patch.object(torch, "load", wraps=torch.load) as read:
                    loaded = dataset.load(0, mode, "cpu")
                self.assertEqual([Path(call.args[0]).parent.name for call in read.call_args_list], ["vae", "text"])
                self.assertEqual(loaded["first"] is not None, mode == "i2v")
                self.assertEqual(set(loaded), {"record", "latent", "text", "first"})

    def test_lora_state_rejects_base_weights_and_changed_rank(self):
        model = SmallState()
        weights = lora_state_dict(model)
        self.assertEqual(set(weights), {"layer.lora_A", "layer.lora_B"})
        load_lora_state_dict(model, weights)
        with self.assertRaisesRegex(ValueError, "target names differ"):
            load_lora_state_dict(model, model.state_dict())
        weights["layer.lora_A"] = torch.zeros(1, 3)
        with self.assertRaisesRegex(ValueError, "Invalid LoRA weight"):
            load_lora_state_dict(model, weights)

    def test_checkpoint_every100_and_exact_optimizer_ema_rng_queue_resume(self):
        cfg = configuration()
        model, records = SmallState(), [dict(category="unlabeled") for _ in range(17)]
        optimizer, ema = optimizer_for(model, cfg), EMA(model)
        plan = ExposurePlan(records, cfg["train"]["seed"])
        identity = dict(code=digest(code_fingerprint()), assets="fixture", manifest="fixture")
        def update(current, opt, average, queue, step):
            exposure = queue.batch(step)
            set_schedule(opt, cfg, step)
            opt.zero_grad(set_to_none=True)
            target = torch.rand(()) + random.random() + np.random.rand()
            sum((p - target).square().mean() for _, p in lora_named_parameters(current)).backward()
            opt.step()
            average.update(current)
            return exposure
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            saved = {}
            for step in range(1, 1201):
                exposure = update(model, optimizer, ema, plan, step)
                if step % 100 == 0:
                    saved[step] = save_checkpoint(tmp, step, model, optimizer, ema, plan, cfg, identity)
                if step == 801:
                    expected = deepcopy((lora_state_dict(model), optimizer.state_dict(), ema.state_dict(), plan.state_dict(), exposure))
            self.assertEqual(len(saved), 12)
            self.assertEqual(read_json(Path(tmp) / "final.json")["step"], 1200)
            self.assertEqual(configuration(saved[1200] / "config.json"), cfg)
            self.assertFalse((saved[1200] / "adapter.pt").exists())
            with patch("static_lora.checkpoint.lock_assets", return_value={}), \
                    patch("static_lora.checkpoint.digest", side_effect=lambda x: "fixture" if x == {} else digest(x)):
                self.assertEqual(inference.checkpoint_path(cfg, saved[1200]), saved[1200])
            restored, queue = SmallState(), ExposurePlan(records, cfg["train"]["seed"])
            opt = optimizer_for(restored, cfg)
            step, average, rng = load_training(saved[800], restored, opt, queue, cfg, identity)
            self.assertEqual(step, 800)
            restore_rng(rng)
            exposure = update(restored, opt, average, queue, 801)
            self.assertEqual(exposure, expected[4])
            self.assertEqual(queue.state_dict(), expected[3])
            for name, value in lora_state_dict(restored).items():
                torch.testing.assert_close(value, expected[0][name], rtol=0, atol=0)
            for name, value in average.values.items():
                torch.testing.assert_close(value, expected[2]["values"][name], rtol=0, atol=0)
            for index, values in opt.state_dict()["state"].items():
                for name, value in values.items():
                    torch.testing.assert_close(value, expected[1]["state"][index][name], rtol=0, atol=0)
            marker = read_json(saved[100] / "complete.json")
            marker["training_recipe"] = "fm_joint_independent_cores_struct_global_share20_v2"
            save_json(marker, saved[100] / "complete.json")
            with self.assertRaisesRegex(ValueError, "static-LoRA checkpoint"):
                inspect_checkpoint(saved[100], cfg, identity)

    def test_cfg_uses_lora_on_both_branches(self):
        model = WanLoRA(tiny_wan(), spec(), recompute=False)
        for name, p in lora_named_parameters(model):
            if name.endswith("lora_B"):
                nn.init.normal_(p, std=.03)
        source = sample()
        negative = -source["text"]
        with patch.object(backbone, "autocast", side_effect=lambda device: nullcontext()):
            cond = model(source["latent"], source["first"], source["text"], torch.tensor(.4))
            uncond = model(source["latent"], source["first"], negative, torch.tensor(.4))
            actual = model.guided(source["latent"], source["first"], source["text"], negative, torch.tensor(.4), 5.)
        torch.testing.assert_close(actual, uncond + 5 * (cond - uncond), rtol=0, atol=0)

    def test_final_inference_cases_exactly_match_formal_v42(self):
        reference = import_file("static_lora_test_reference_inference", V42_ROOT / "inference/infer.py")
        cfg = configuration()
        args = SimpleNamespace(suite="final", portrait=False, duration=5.)
        actual = inference.collect_cases(cfg, args)
        expected = reference.collect_cases(read_json(ROOT / "configs/v42_reference.json"), args)
        # The LoRA plan additionally records the origin.txt content hash.
        self.assertEqual([{k: v for k, v in c.items() if k != "prompt_sha256"} for c in actual], expected)
        self.assertEqual(len(actual), 81)

    def test_bfloat16_base_and_fp32_lora_weights_work_under_autocast(self):
        base = nn.Linear(16, 16).bfloat16()
        layer = LoRALinear(deepcopy(base), 2, 2)
        x = torch.randn(2, 4, 16)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            expected, actual = base(x), layer(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
        self.assertEqual(layer.lora_B.grad.dtype, torch.float32)
        self.assertTrue(torch.isfinite(layer.lora_B.grad).all())
        self.assertGreater(float(layer.lora_B.grad.abs().sum()), 0.)

    def test_inference_preparation_reuses_cached_conditions_without_teacher(self):
        from PIL import Image
        cfg = deepcopy(configuration())
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            root, cache, ckpt = Path(tmp) / "output", Path(tmp) / "cache", Path(tmp) / "checkpoint"
            root.mkdir()
            cfg["paths"]["cache"] = str(cache)
            manifest = dict(negative_prompt="native fixture")
            save_json(manifest, cache / "manifest.json")
            save_tensor(torch.ones(3, 8, dtype=torch.bfloat16), cache / "negative.pt")
            save_tensor(torch.ones(4, 8, dtype=torch.bfloat16) * 2, cache / "text/source0000001.pt")
            first = torch.ones(1, 48, 1, 2, 2)
            save_tensor(dict(first=first), cache / "vae/source0000001.pt")
            names = ("negative.pt", "text/source0000001.pt", "vae/source0000001.pt")
            save_json(dict(files={name: file_identity(cache / name) for name in names}), cache / "ready.json")
            save_json(dict(step=1200, identity=dict(manifest=digest(manifest))), ckpt / "complete.json")
            save_tensor({}, ckpt / "ema.pt")
            picture = Path(tmp) / "input.png"
            Image.new("RGB", (48, 32), color=(40, 90, 120)).save(picture)
            cases = [dict(key="external_i2v", id="new", group="new_prompt", mode="i2v", image=str(picture),
                          prompt="image fixture", duration=5.),
                     dict(key="external_t2v", id="new", group="new_prompt", mode="t2v", prompt="text fixture", duration=5.),
                     dict(key="gt_i2v", id="source0000001", group="train", mode="i2v", prompt="cached fixture", duration=.2,
                          gt_record=dict(geometry=dict(h=32, w=32, frames=5)))]
            encoder = lambda prompts, device: [torch.ones(3, 8, dtype=torch.bfloat16)]
            def encode(vae, pixels):
                return torch.zeros(1, 48, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
            args = SimpleNamespace(checkpoint=str(ckpt), suite="final", portrait=False)
            with patch.object(inference, "checkpoint_path", return_value=ckpt), \
                    patch.object(inference, "collect_cases", return_value=cases), \
                    patch.object(inference, "native_negative", return_value="native fixture"), \
                    patch.object(inference, "load_text", return_value=encoder) as text_loader, \
                    patch.object(inference, "load_vae", return_value=object()) as vae_loader, \
                    patch.object(inference, "encode_vae", side_effect=encode), \
                    patch.object(inference, "log"):
                inference.prepare(cfg, args, root, torch.device("cpu"))
                inference.prepare(cfg, args, root, torch.device("cpu"))
            self.assertEqual(text_loader.call_count, 1)
            self.assertEqual(vae_loader.call_count, 1)
            cached = torch.load(root / "gt_i2v/condition.pt", weights_only=True)
            torch.testing.assert_close(cached["first"], first)
            self.assertEqual(set(cached), {"text", "negative", "first", "geometry"})
            self.assertEqual(len(read_json(root / "conditions_ready.json")["files"]), 3)
            self.assertEqual(list(root.rglob("*anchor*")), [])

    def test_real_unipc_sampling_export_and_per_case_resume_on_cpu(self):
        solver = import_file("static_lora_test_unipc", Path(configuration()["paths"]["wan_code"]) / "wan/utils/fm_solvers_unipc.py")
        wan, utils = ModuleType("wan"), ModuleType("wan.utils")
        wan.__path__, utils.__path__ = [], []
        class Prediction:
            def __init__(self):
                self.calls = 0
            def guided(self, x, first, text, negative, sigma, guidance):
                self.calls += 1
                if first is not None:
                    torch.testing.assert_close(x[:, :, :1], first, rtol=0, atol=0)
                return .05 * x
        class Decoder:
            def frames(self, latent):
                for _ in range(5):
                    yield torch.full((3, 32, 32), .5)
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            root = Path(tmp)
            cases = [dict(key=mode, id=mode, mode=mode, group="new_prompt", duration=.2, prompt="fixture")
                     for mode in ("i2v", "t2v")]
            plan = dict(cases=cases, checkpoint="fixture", checkpoint_step=1200, ema_sha256="fixture",
                        negative="fixture", negative_sha256=digest("fixture"),
                        recipe=dict(steps=50, shift=5., guidance=5., seed=42))
            save_json(plan, root / "cases.json")
            for case in cases:
                save_tensor(dict(text=torch.zeros(2, 8), negative=torch.ones(2, 8),
                                 first=torch.ones(1, 48, 1, 2, 2) if case["mode"] == "i2v" else None,
                                 geometry=dict(h=32, w=32, frames=5)), root / case["key"] / "condition.pt")
            model = Prediction()
            with patch.dict(sys.modules, {"wan": wan, "wan.utils": utils, "wan.utils.fm_solvers_unipc": solver}), \
                    patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "max_memory_allocated", return_value=0), \
                    patch.object(inference, "log"):
                for case in cases:
                    inference.sample_case(model, Decoder(), case, plan, root, torch.device("cpu"))
                    inference.sample_case(model, Decoder(), case, plan, root, torch.device("cpu"))
                inference.report(root)
            self.assertEqual(model.calls, 100)
            self.assertEqual(read_json(root / "report.json")["completed_cases"], 2)
            for case in cases:
                metadata = read_json(root / case["key"] / "metadata.json")
                self.assertEqual(metadata["generation_logical_predictions"], 100)
                self.assertEqual(metadata["frames"], 5)
                self.assertGreater((root / case["key"] / "video.mp4").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
