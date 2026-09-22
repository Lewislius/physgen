"""Inference contracts: saved EMA, demo41 parity, resume and failure reporting."""
from contextlib import ExitStack
from copy import deepcopy
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_static_lora as fixtures
from static_lora import ROOT, TRAINING_RECIPE, VERSION
from static_lora import checkpoint as checkpoints
from static_lora.config import configuration
from static_lora.runtime import digest, file_identity, read_json, save_json, save_tensor

inference = fixtures.inference


class InferenceTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "tmp").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / "tmp")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = configuration()

    def checkpoint(self):
        path = self.root / "step0800"
        save_json(self.cfg, path / "config.json")
        save_tensor({"fixture": torch.ones(1)}, path / "ema.pt")
        complete = dict(step=800, ema_updates=800, version=VERSION, batch_size=8,
                        training_recipe=TRAINING_RECIPE, identity=dict(code="fixture"),
                        files={n: file_identity(path / n) for n in ("config.json", "ema.pt")})
        # These files belong to training resume and are intentionally absent.
        for name in ("lora.pt", "training.pt"):
            complete["files"][name] = dict(bytes=100, sha256="unused", mtime_ns=0)
        save_json(complete, path / "complete.json")
        return path

    def test_inference_needs_only_saved_config_and_ema_training_still_needs_resume_state(self):
        path = self.checkpoint()
        self.assertEqual(checkpoints.inspect_checkpoint(path, self.cfg, inference=True)["step"], 800)
        with self.assertRaises(FileNotFoundError):
            checkpoints.inspect_checkpoint(path, self.cfg)
        with (path / "ema.pt").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "Artifact changed"):
            checkpoints.inspect_checkpoint(path, self.cfg, inference=True)

    def test_inference_rejects_wrong_config_and_unfinished_ema(self):
        path = self.checkpoint()
        cfg = deepcopy(self.cfg)
        cfg["lora"]["rank"] = 16
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            checkpoints.inspect_checkpoint(path, cfg, inference=True)
        status = read_json(path / "complete.json")
        status["ema_updates"] = 799
        save_json(status, path / "complete.json")
        with self.assertRaisesRegex(ValueError, "completed static-LoRA"):
            checkpoints.inspect_checkpoint(path, self.cfg, inference=True)

    def test_source_compatibility_is_exact_and_inference_only(self):
        ledger = read_json(ROOT / "inference/checkpoint_compatibility.json")
        entry = ledger["transitions"][0]
        with patch.object(checkpoints, "checkpoint_code_compatible", return_value=False):
            self.assertTrue(checkpoints.inference_code_compatible(entry["before"], entry["after"]))
            self.assertFalse(checkpoints.inference_code_compatible("unknown", entry["after"]))
            self.assertFalse(checkpoints.inference_code_compatible(entry["before"], "unreviewed-edit"))
        from static_lora.runtime import checkpoint_code_compatible
        self.assertFalse(checkpoint_code_compatible(entry["before"], entry["after"]))

    def test_demo41_matches_v4_prompt_image_mode_and_order(self):
        args = SimpleNamespace(suite="demo", duration=5., portrait=False)
        cases = inference.collect_cases(self.cfg, args)
        reference = read_json(ROOT / "tests/fixtures/demo41.json")["cases"]
        self.assertEqual([(c["id"], c["mode"], c["prompt"], c.get("image")) for c in cases],
                         [(c["id"], c["mode"], c["prompt"], c.get("image")) for c in reference])
        self.assertEqual(len(cases), 41)
        self.assertEqual(sum(c["mode"] == "i2v" for c in cases), 21)
        self.assertTrue(all("gt_record" not in c for c in cases))

    def test_demo_prepare_does_not_read_training_cache_or_load_teacher(self):
        checkpoint = self.checkpoint()
        self.cfg["paths"]["cache"] = str(self.root / "absent_training_cache")
        args = SimpleNamespace(checkpoint=str(checkpoint), suite="demo", duration=5., portrait=False)
        def encode(vae, pixels):
            return torch.zeros(1, 48, 1, pixels.shape[-2] // 16, pixels.shape[-1] // 16)
        with patch.object(inference, "checkpoint_path", return_value=checkpoint), \
                patch.object(inference, "load_text", return_value=lambda prompts, device: [torch.ones(2, 4096)]), \
                patch.object(inference, "load_vae", return_value=object()), \
                patch.object(inference, "encode_vae", side_effect=encode), patch.object(inference, "log"):
            inference.prepare(self.cfg, args, self.root, torch.device("cpu"))
        self.assertEqual(len(read_json(self.root / "conditions_ready.json")["files"]), 41)
        self.assertFalse(Path(self.cfg["paths"]["cache"]).exists())
        self.assertEqual(list(self.root.rglob("*anchor*")), [])

    def test_real_preflight_forbids_all_v42_imports_and_file_access(self):
        checkpoint = ROOT / "checkpoints/lora_20260917T140727Z_dae2a0/step0800"
        result = subprocess.run([sys.executable, "-I", "-B", str(ROOT / "tools/check_standalone.py"),
                                 "--checkpoint", str(checkpoint), "--suite", "demo", "--output", str(self.root)],
                                text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("[standalone] PASS", result.stdout)
        report = read_json(self.root / "preflight.json")
        self.assertEqual((report["checkpoint_step"], report["total_cases"]), (800, 41))
        self.assertEqual(report["weights"]["tensors"], 480)

    def test_yaml_and_direct_shell_select_step0800_saved_config_demo41(self):
        config = yaml.safe_load((ROOT / "inference/infer_1x96g.yaml").read_text())
        self.assertEqual(config["resources"]["slots_per_trial"], 1)
        self.assertEqual(config["resources"]["resource_pool"], "blk-96g")
        env = {k: v for k, v in os.environ.items() if k not in (
            "CONFIG", "CHECKPOINT", "OUTPUT", "SUITE", "CHECK_ONLY", "PROMPT", "IMAGE", "MODE", "PORTRAIT")}
        env["PARSE_ONLY"] = "1"
        command = ["bash", str(ROOT / "inference/infer_1x96g.sh")]
        direct = subprocess.check_output(command, text=True, env=env)
        env.update(dict(item.split("=", 1) for item in config["environment"]["environment_variables"]))
        cluster = subprocess.check_output(command, text=True, env=env)
        self.assertEqual(direct, cluster)
        commands = [shlex.split(line) for line in cluster.splitlines()]
        self.assertEqual([c[c.index("--phase") + 1] for c in commands], ["prepare", "sample", "report"])
        prepare = commands[0]
        self.assertEqual(prepare[prepare.index("--suite") + 1], "demo")
        selected = prepare[prepare.index("--checkpoint") + 1]
        self.assertTrue(selected.endswith("lora_20260917T140727Z_dae2a0/step0800"))
        self.assertEqual(prepare[prepare.index("--config") + 1], selected + "/config.json")
        self.assertTrue(prepare[prepare.index("--output") + 1].endswith("dae2a0_step0800_ema_demo_seed42"))
        env.update(CHECK_ONLY="1", CHECKPOINT=str(self.root / "other run/step0400"))
        checked = shlex.split(subprocess.check_output(command, text=True, env=env))
        self.assertEqual(checked[checked.index("--phase") + 1], "check")
        self.assertEqual(checked[checked.index("--config") + 1], env["CHECKPOINT"] + "/config.json")

    def batch(self):
        path = self.checkpoint()
        cases = [dict(key="P01_" + mode, mode=mode, group="demo", prompt="fixture", duration=5.)
                 for mode in ("i2v", "t2v")]
        plan = dict(checkpoint=str(path), checkpoint_step=800, config=self.cfg, cases=cases,
                    ema_sha256=fixtures.inference.sha256(path / "ema.pt"), recipe=dict(seed=42))
        save_json(plan, self.root / "cases.json")
        save_json(dict(plan_hash=digest(plan), files={}), self.root / "conditions_ready.json")
        return plan

    def test_resuming_completed_batch_skips_model_and_vae_loading(self):
        self.batch()
        with patch.object(inference, "checkpoint_path"), \
                patch.object(inference, "completed_case", return_value=True), \
                patch.object(inference, "load_model") as model, patch.object(inference, "load_vae") as vae:
            inference.sample(self.root, torch.device("cpu"))
        model.assert_not_called()
        vae.assert_not_called()
        summary = read_json(self.root / "summary.json")
        self.assertEqual((summary["status"], summary["completed"], summary["total"]), ("completed", 2, 2))

    def test_batch_failure_keeps_completed_cases_and_reports_failed_case(self):
        self.batch()
        with ExitStack() as stack:
            for name in ("checkpoint_path", "load_model", "load_vae", "load_lora_state_dict", "Decoder"):
                stack.enter_context(patch.object(inference, name))
            stack.enter_context(patch.object(torch.cuda, "reset_peak_memory_stats"))
            stack.enter_context(patch.object(inference, "sample_case", side_effect=[None, RuntimeError("fixture failure")]))
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                inference.sample(self.root, torch.device("cpu"))
        summary = read_json(self.root / "summary.json")
        self.assertEqual((summary["status"], summary["completed"]), ("failed", 1))
        self.assertEqual([c["status"] for c in summary["cases"]], ["completed", "failed"])


if __name__ == "__main__":
    unittest.main()
