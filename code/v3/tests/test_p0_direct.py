from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest
from copy import deepcopy

from PIL import Image


ROOT = Path("/home/liuzhirui/Project/physGen")
DEMO = ROOT / "code/v1/demo"
V3 = ROOT / "code/v3"
SOURCE = ROOT / "prompt/p0_direct_prompts_p01_p20.json"
INFERENCE = V3 / "inference/infer_trace_writer.py"


def load_inference_module():
    spec = importlib.util.spec_from_file_location("p0_direct_inference", INFERENCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class P0DirectDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = json.loads(SOURCE.read_text(encoding="utf-8"))
        cls.by_id = {item["sample_id"]: item for item in cls.source["samples"]}

    def test_all_twenty_samples_are_direct_and_exact(self):
        self.assertEqual(tuple(self.by_id), tuple(f"P{i:02d}" for i in range(1, 21)))
        for sample_id, authored in self.by_id.items():
            sample_dir = DEMO / sample_id
            i0 = json.loads((sample_dir / f"{sample_id}-i0.json").read_text())
            plan = json.loads((sample_dir / f"{sample_id}-V2-plan.json").read_text())
            planimg = json.loads((sample_dir / f"{sample_id}-V2-planimg.json").read_text())
            retired = json.loads(
                (sample_dir / f"{sample_id}-cplus-cminus.json").read_text()
            )

            self.assertEqual(i0["pre_event_image_prompt"], authored["pre_event_image_prompt"])
            self.assertEqual(i0["initial_condition"], authored["initial_condition"])
            self.assertEqual(plan["global_semantic"], authored["global_semantic"])
            self.assertEqual(planimg["global_semantic"], authored["global_semantic"])
            self.assertEqual(plan["stages"], authored["stages"])
            self.assertEqual(planimg["stages"], authored["stages"])
            self.assertEqual(
                plan.get("structured_guidance"), authored.get("structured_guidance")
            )
            self.assertEqual(
                planimg.get("structured_guidance"), authored.get("structured_guidance")
            )
            self.assertEqual(plan["temporal_route"], authored["temporal_route"])
            self.assertEqual(planimg["temporal_route"], authored["temporal_route"])
            self.assertEqual(plan["design_revision"], "p0-stage-direct-json-r2")
            self.assertEqual(planimg["design_revision"], "p0-stage-direct-json-r2")
            self.assertEqual(plan["cfg_negative"], " ")
            self.assertEqual(planimg["cfg_negative"], " ")
            self.assertEqual(plan["generation_basis"]["mode"], "identifier_only")
            self.assertEqual(plan["grounding"]["entity_boxes"], [])
            self.assertEqual(planimg["generation_basis"]["mode"], "visual_grounding")
            self.assertEqual(planimg["grounding"]["entity_boxes"], authored["boxes"])
            self.assertFalse(retired["used_for_generation"])

            stage_ids = [stage["id"] for stage in authored["stages"]]
            stage_text = [stage["positive"] for stage in authored["stages"]]
            self.assertEqual(stage_ids, ["setup", "onset", "evolution", "completion", "terminal"])
            self.assertEqual(len(stage_text), len(set(stage_text)))

            weights = authored["temporal_route"]["weights"]
            self.assertEqual((len(weights), len(weights[0])), (5, 25))
            self.assertEqual(
                sum(sum(value > 0 for value in column) == 2 for column in zip(*weights)),
                8,
            )
            for column in zip(*weights):
                self.assertAlmostEqual(sum(column), 1.0)
                self.assertLessEqual(sum(value > 0 for value in column), 2)

    def test_every_saved_first_frame_is_exactly_1280x704(self):
        for sample_id in self.by_id:
            sample_dir = DEMO / sample_id
            for name in (
                f"{sample_id}-i0.png",
                f"{sample_id}-i0-1280x704.png",
                f"{sample_id}-i0-1280x704.jpg",
            ):
                with Image.open(sample_dir / name) as image:
                    self.assertEqual(image.size, (1280, 704), name)

    def test_unified_loader_returns_json_strings_without_rewriting(self):
        inference = load_inference_module()
        for mode, suffix in (("i2v", "planimg"), ("t2v", "plan")):
            args = inference.build_parser().parse_args(
                ["--mode", mode, "--sample_ids", "P01,P02"]
            )
            loaded = inference.load_inputs(args, ("P01", "P02"))
            for item in loaded:
                plan = json.loads(item["plan_path"].read_text(encoding="utf-8"))
                self.assertEqual(item["prompt"], plan["global_semantic"], suffix)
                self.assertEqual(item["negative_prompt"], plan["cfg_negative"], suffix)
                self.assertEqual(
                    item["stage_prompts"],
                    tuple(stage["positive"] for stage in plan["stages"]),
                    suffix,
                )
                self.assertEqual(item["temporal_route"], plan["temporal_route"])
                self.assertIsNone(item["structured_global_prompt"])
                self.assertEqual(item["structured_stage_prompts"], ())

    def test_all_structured_guidance_is_complete_compact_and_stage_local(self):
        inference = load_inference_module()
        stage_ids = ("setup", "onset", "evolution", "completion", "terminal")
        for sample_id, plan in self.by_id.items():
            structured = plan["structured_guidance"]
            self.assertEqual(
                {item["id"] for item in structured["entities"]},
                {item["id"] for item in plan["entities"]},
                sample_id,
            )
            self.assertEqual(
                tuple(stage["id"] for stage in structured["stages"]),
                stage_ids,
                sample_id,
            )
            global_text, stage_texts = inference.compile_structured_guidance(
                plan, sample_id
            )
            self.assertTrue(global_text, sample_id)
            self.assertEqual(len(stage_texts), 5, sample_id)
            self.assertTrue(all(stage_texts), sample_id)
            self.assertLess(max(len(text.split()) for text in stage_texts), 350, sample_id)
            initial = structured["initial_condition"]
            for fact in initial["facts"]:
                for stage_id, stage_text in zip(stage_ids, stage_texts):
                    self.assertEqual(
                        fact in stage_text,
                        stage_id in initial["active_stages"],
                        f"{sample_id}:{stage_id}",
                    )

        p02_global, p02_stages = inference.compile_structured_guidance(
            self.by_id["P02"], "P02"
        )
        self.assertIn("exactly one glossy red billiard ball", p02_global)
        self.assertIn("unchanged round rigid shape", p02_global)
        self.assertIn("rightward rolling momentum", p02_stages[0])
        self.assertNotIn("rightward rolling momentum", p02_stages[1])
        self.assertIn("edge-to-edge contact", p02_stages[1])
        self.assertIn("exactly the two original separated balls", p02_stages[-1])
        self.assertNotIn("[STAGE", " ".join(p02_stages))

        p08_global, _ = inference.compile_structured_guidance(
            self.by_id["P08"], "P08"
        )
        self.assertIn("at most one connected transparent meltwater puddle", p08_global)

    def test_enhanced_loader_accepts_all_twenty_and_rejects_missing_guidance(self):
        inference = load_inference_module()
        args = inference.build_parser().parse_args(
            [
                "--mode",
                "i2v",
                "--sample_ids",
                "all",
                "--conditioning_variant",
                "strong_stage_json",
            ]
        )
        sample_ids = tuple(f"P{index:02d}" for index in range(1, 21))
        loaded = inference.load_inputs(args, sample_ids)
        self.assertEqual(len(loaded), 20)
        self.assertTrue(all(item["structured_global_prompt"] for item in loaded))
        self.assertTrue(
            all(len(item["structured_stage_prompts"]) == 5 for item in loaded)
        )

        missing = deepcopy(self.by_id["P01"])
        missing.pop("structured_guidance")
        with self.assertRaisesRegex(ValueError, "requires structured_guidance"):
            inference.compile_structured_guidance(missing, "P01")

    def test_rms_matching_exposes_stage_to_global_ratio_directly(self):
        import torch

        inference = load_inference_module()
        candidate = torch.full((1, 4, 3), 0.25)
        reference = torch.full((1, 4, 3), 2.0)
        matched, _ = inference.rms_match(candidate, reference)
        self.assertAlmostEqual(
            float(inference._tensor_rms(matched)),
            float(inference._tensor_rms(reference)),
            places=4,
        )

        applied = 1.5 * matched
        self.assertAlmostEqual(
            inference._rms_ratio(applied, reference, valid_length=4), 1.5, places=4
        )
        capped, scale = inference.rms_cap(applied, reference, ratio=1.0)
        self.assertLess(float(scale), 1.0)
        self.assertAlmostEqual(
            inference._rms_ratio(capped, reference, valid_length=4), 1.0, places=4
        )

    def test_no_active_python_conditioning_modules_remain(self):
        active = sorted(path.name for path in (V3 / "trace_writer_v3").glob("*.py"))
        self.assertEqual(active, [])


if __name__ == "__main__":
    unittest.main()
