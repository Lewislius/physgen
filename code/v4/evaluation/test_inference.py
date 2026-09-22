"""CPU checks of real schedulers, sampling state, launchers and saved v4 weights.

Set V4_INFERENCE_CHECKPOINT to additionally exercise trained correctors with a
small frozen Wan substitute. This does not measure real Wan quality or VRAM.
"""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from inference import infer_native_p as inference
from physgen_v4.backbone import ProcessWan
from physgen_v4.coordinates import Coordinates
from physgen_v4.runtime import read_config

torch.set_num_threads(2)
sys.path.insert(0, read_config(ROOT / "configs/wisa_native_p.yaml")["paths"]["wan_code"])


def setUpModule():
    # Wan's package __init__ eagerly initializes CUDA through T5 default arguments.
    # Load the unmodified scheduler files directly for these CPU mathematical checks.
    global solver_modules
    code = Path(read_config(ROOT / "configs/wisa_native_p.yaml")["paths"]["wan_code"])
    modules = {}
    for name in ("fm_solvers", "fm_solvers_unipc"):
        full_name = f"wan.utils.{name}"
        spec = importlib.util.spec_from_file_location(full_name, code / f"wan/utils/{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[full_name] = module
    solver_modules = patch.dict(sys.modules, modules)
    solver_modules.start()


def tearDownModule():
    solver_modules.stop()


class RecordingModel:
    def __init__(self):
        self.calls = []

    def correction_sites(self):
        return [(n, "A", f"A@{n}") for n in (5, 10, 15, 20, 25, 30)]

    def __call__(self, latent, first, text, sigma, coords, process, **kwargs):
        self.calls.append(dict(latent=latent.clone(), text=text, process=process, **kwargs))
        state = text.float().clone() if process is None else process + text
        return torch.ones_like(latent) * text.item(), state, dict(metrics={}, states={})


class SamplingTests(unittest.TestCase):
    def setUp(self):
        self.args = inference.build_parser().parse_args(["--steps", "4"])
        self.first = torch.full((1, 48, 1, 2, 2), .75)
        self.coords = SimpleNamespace(latent=(torch.arange(3),))

    def sample(self, model, args=None, on_step=None):
        return inference.sample_latents(model, self.first, torch.tensor([2.0]), torch.tensor([.5]),
                                        self.coords, args or self.args, on_step)

    def test_real_solvers_cfg_sign_first_frame_and_call_budget(self):
        for solver in ("euler", "unipc", "dpm++"):
            with self.subTest(solver=solver):
                args = copy.copy(self.args)
                args.solver = solver
                sampler, timesteps, sigmas = inference.schedule(solver, args.steps, args.shift, "cpu")
                model, records = RecordingModel(), []
                with patch.object(inference, "schedule", return_value=(sampler, timesteps, sigmas)):
                    result, report = self.sample(model, args, records.append)
                self.assertEqual(len(model.calls), 2 * args.steps)
                self.assertEqual(len(records), args.steps)
                self.assertEqual(report["wan_forwards"], len(model.calls))
                self.assertEqual(report["scheduler_updates"], len(records))
                initial = torch.randn(result.shape, generator=torch.Generator().manual_seed(args.seed))
                expected = initial[:, :, 1:] - sigmas[0] * (.5 + args.guidance * (2 - .5))
                torch.testing.assert_close(result[:, :, 1:], expected, rtol=2e-5, atol=2e-5)
                for call in model.calls:
                    torch.testing.assert_close(call["latent"][:, :, :1], self.first, rtol=0, atol=0)
                torch.testing.assert_close(result[:, :, :1], self.first, rtol=0, atol=0)
                self.assertEqual(result.dtype, torch.float32)

    def test_cfg_states_are_initialized_separately_and_persist_between_steps(self):
        model = RecordingModel()
        self.sample(model)
        for index, call in enumerate(model.calls):
            step, branch = divmod(index, 2)
            if step == 0:
                self.assertIsNone(call["process"])
            else:
                self.assertEqual(call["process"].item(), step * (2 if branch == 0 else .5))
                self.assertEqual(call["process"].dtype, torch.float32)
        self.assertIsNot(model.calls[2]["process"], model.calls[3]["process"])

    def test_reset_state_and_writer_ablation_reach_both_branches(self):
        self.args.reset_state = True
        self.args.writer_off = ["A@5", "A@10"]
        self.args.variant = "A"
        model = RecordingModel()
        self.sample(model)
        for call in model.calls:
            self.assertIsNone(call["process"])
            self.assertEqual(call["disable"], ("B",))
            self.assertEqual(call["writer_off"], ("A@5", "A@10"))
        self.args.writer_off = ["A@6"]
        with self.assertRaisesRegex(ValueError, "Unknown writer"):
            self.sample(RecordingModel())

    def test_sampling_is_reproducible_and_seed_changes_only_future_noise(self):
        first, _ = self.sample(RecordingModel())
        repeated, _ = self.sample(RecordingModel())
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
        self.args.seed += 1
        changed, _ = self.sample(RecordingModel())
        self.assertFalse(torch.equal(first[:, :, 1:], changed[:, :, 1:]))
        torch.testing.assert_close(first[:, :, :1], changed[:, :, :1], rtol=0, atol=0)

    def test_frame_alignment_timing_and_invalid_inputs(self):
        config = {"data": {"max_frames": 101}}
        for frames, expected in ((None, 101), (200, 101), (100, 97), (5, 5)):
            spec = inference.video_spec(config, frames, 25, None)
            self.assertEqual(spec["frames"], expected)
            self.assertEqual(spec["duration"], (expected - 1) / 25)
        self.assertEqual(inference.video_spec(config, 101, 24, 4)["output_fps"], 25)
        for frames, fps, duration in ((1, 24, None), (4, 24, None), (101, 0, None),
                                      (101, float("nan"), None), (101, 24, 0), (101, 24, -1)):
            with self.assertRaises(ValueError):
                inference.video_spec(config, frames, fps, duration)
        for key, value in (("steps", 0), ("shift", 0), ("guidance", float("nan")),
                           ("seed", -1), ("solver", "invalid")):
            args = copy.copy(self.args)
            setattr(args, key, value)
            with self.assertRaises(ValueError):
                inference.validate_sampling(args)


class LauncherTests(unittest.TestCase):
    def test_all_single_gpu_pools_use_fixed_runtime_and_forward_arguments(self):
        for label, pool in (("1x48g", "amp-48g"), ("1x96g", "blk-96g"), ("1x80g", "amp-80g")):
            with self.subTest(label=label):
                config = yaml.safe_load((ROOT / f"inference/infer_wisa_native_p_{label}.yaml").read_text())
                self.assertEqual(config["resources"]["resource_pool"], pool)
                self.assertEqual(config["resources"]["slots_per_trial"], 1)
                entry = shlex.split(config["entrypoint"])
                self.assertTrue(Path(entry[1]).is_file())
                environment = {k: v for k, v in os.environ.items() if k not in
                               ("PYTHONHOME", "DURATION", "WRITER_OFF", "PARSE_ONLY", "CHECK_ONLY")}
                environment.update(dict(item.split("=", 1) for item in config["environment"]["environment_variables"]))
                environment.update(PARSE_ONLY="1", PYTHON_EXEC="/unused/python", WRITER_OFF="A@5 A@10",
                                   DURATION="4", OUTPUT="/tmp/output video.mp4")
                prompt = "A ball hits another ball. Literal $(text) and `text`."
                result = subprocess.run([*entry, "--prompt", prompt, "--seed", "17"], env=environment,
                                        text=True, capture_output=True, timeout=30, check=True)
                command = shlex.split(result.stdout.splitlines()[-1])
                self.assertEqual(command[:3], ["/home/liuzhirui/miniconda3/envs/moviestory/bin/python", "-I", "-B"])
                args = inference.build_parser().parse_args(command[4:])
                self.assertEqual(args.checkpoint, str(inference.DEFAULT_CHECKPOINT))
                self.assertEqual(args.device, 0)
                self.assertEqual(args.prompt, prompt)
                self.assertEqual(args.seed, 17)
                self.assertEqual(args.duration, 4)
                self.assertEqual(args.writer_off, ["A@5", "A@10"])
                self.assertEqual(args.output, "/tmp/output video.mp4")

    def test_shell_syntax_and_compatibility_yaml(self):
        for path in (ROOT / "inference").glob("*.sh"):
            subprocess.run(["bash", "-n", str(path)], check=True)
        self.assertEqual(yaml.safe_load((ROOT / "inference/infer_wisa_native_p.yaml").read_text()),
                         yaml.safe_load((ROOT / "inference/infer_wisa_native_p_1x48g.yaml").read_text()))


@unittest.skipUnless(os.environ.get("V4_INFERENCE_CHECKPOINT"), "Set V4_INFERENCE_CHECKPOINT for trained-weight checks")
class SavedCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config, cls.corrector, cls.report = inference.load_checkpoint(os.environ["V4_INFERENCE_CHECKPOINT"])

    def test_requested_a1_weights_are_restored_exactly(self):
        self.assertEqual(self.report["phase"], "A1")
        self.assertEqual(self.report["initialization"], self.config["corrector"]["initialization"])
        self.assertEqual(self.report["parameter_sharing"], "per_block")
        self.assertEqual(self.report["a_blocks"], [5, 10, 15, 20, 25, 30])
        self.assertEqual(self.report["iterations"], 1)
        self.assertFalse(self.report["enable_b"])
        saved = torch.load(Path(os.environ["V4_INFERENCE_CHECKPOINT"]) / "corrector.pt",
                           map_location="cpu", mmap=True, weights_only=True)
        for block, unit in self.corrector.A.items():
            torch.testing.assert_close(unit.site.writer.weight, saved[f"A.{block}.site.writer.weight"], rtol=0, atol=0)
            self.assertGreater(unit.site.writer.weight.norm().item(), 0)
        self.assertEqual(self.corrector.B.writer.weight.count_nonzero().item(), 0)
        self.assertEqual(self.corrector.B.b.item(), 0)

    def test_mismatched_prior_or_block_layout_is_rejected_without_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "corrector.pt").symlink_to(Path(os.environ["V4_INFERENCE_CHECKPOINT"]) / "corrector.pt")
            wrong_prior = "image_text" if self.report["initialization"] == "image_text_gt" else "image_text_gt"
            for changes in (dict(initialization=wrong_prior),
                            dict(a_every=0, a_blocks=[5, 10, 15, 20, 25, 29])):
                config = copy.deepcopy(self.config)
                config["corrector"].update(changes)
                (root / "config.yaml").write_text(yaml.safe_dump(config))
                with self.assertRaises(RuntimeError):
                    inference.load_checkpoint(root)

    def test_trained_writers_affect_sampling_and_a1_full_equals_a_only(self):
        from evaluation.test_condition_prior import TinyWan
        embedding = SimpleNamespace(sinusoidal_embedding_1d=lambda width, t: t[:, None].expand(-1, width))
        module_patch = patch.dict(sys.modules, {"wan.modules.model": embedding})
        module_patch.start()
        self.addCleanup(module_patch.stop)
        torch.manual_seed(902)
        model = ProcessWan(TinyWan(), self.corrector, self.report["enable_b"]).eval()
        coords = Coordinates.build(torch.linspace(0, .32, 9), 64, 64, 4, 1, teacher_size=64)
        first = torch.randn(1, 48, 1, 4, 4)
        text, negative = torch.randn(1, 3, 4096), torch.randn(1, 2, 4096)
        args = inference.build_parser().parse_args(["--steps", "2"])
        def run():
            return inference.sample_latents(model, first, text, negative, coords, args)
        with patch.object(self.corrector, "initialize", wraps=self.corrector.initialize) as initialize:
            full, report = run()
            self.assertEqual(initialize.call_count, 2)
            self.assertTrue(all(call.kwargs["gt_video"] is None for call in initialize.call_args_list))
        self.assertEqual(report["corrector_calls"], 24)
        self.assertEqual(report["process_dtypes"], dict(cond="torch.float32", uncond="torch.float32"))
        args.variant = "A"
        a_only, _ = run()
        torch.testing.assert_close(full, a_only, rtol=0, atol=0)
        args.variant = "wan"
        baseline, report = run()
        self.assertEqual(report["corrector_calls"], 0)
        self.assertGreater((full - baseline).abs().max().item(), 1e-7)
        args.variant, args.reset_state = "full", True
        with patch.object(self.corrector, "initialize", wraps=self.corrector.initialize) as initialize:
            reset, _ = run()
            self.assertEqual(initialize.call_count, 4)
        self.assertGreater((full - reset).abs().max().item(), 1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
