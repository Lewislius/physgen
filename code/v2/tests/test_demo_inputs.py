from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

from ace_router.demo_inputs import (
    DemoInputError,
    EXPECTED_DEMO_IDS,
    parse_sample_ids,
    resolve_demo_inputs,
)
from inference.infer_trace_writer import _prepare_samples, _write_static_artifacts


V2_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = V2_ROOT.parent / "v1" / "demo"


class DemoInputTests(unittest.TestCase):
    def test_all_is_exactly_twenty_samples(self) -> None:
        self.assertEqual(parse_sample_ids("all"), EXPECTED_DEMO_IDS)
        self.assertEqual(EXPECTED_DEMO_IDS[0], "P01")
        self.assertEqual(EXPECTED_DEMO_IDS[-1], "P20")

    def test_i2v_selects_planimg_and_high_resolution_image(self) -> None:
        (source,) = resolve_demo_inputs(
            demo_root=DEMO_ROOT,
            plan_root=DEMO_ROOT,
            sample_ids=("P01",),
            mode="i2v",
        )
        self.assertEqual(source.plan_path.name, "P01-V2-planimg.json")
        self.assertEqual(source.image_path.name, "P01-i0-1280x704.png")

    def test_t2v_selects_plan_and_never_resolves_an_image(self) -> None:
        (source,) = resolve_demo_inputs(
            demo_root=DEMO_ROOT,
            plan_root=DEMO_ROOT,
            sample_ids=("P01",),
            mode="t2v",
        )
        self.assertEqual(source.plan_path.name, "P01-V2-plan.json")
        self.assertIsNone(source.image_path)

    def test_missing_inputs_are_reported_together(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "P01").mkdir()
            (root / "P02").mkdir()
            with self.assertRaisesRegex(DemoInputError, "2 requested sample") as raised:
                resolve_demo_inputs(
                    demo_root=root,
                    plan_root=root,
                    sample_ids=("P01", "P02"),
                    mode="t2v",
                )
            self.assertIn("P01-V2-plan.json", str(raised.exception))
            self.assertIn("P02-V2-plan.json", str(raised.exception))

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(DemoInputError, "duplicate"):
            parse_sample_ids("P01,p01")

    def test_actual_m3_input_prepares_compiled_artifacts(self) -> None:
        sources = resolve_demo_inputs(
            demo_root=DEMO_ROOT,
            plan_root=DEMO_ROOT,
            sample_ids=("P01",),
            mode="i2v",
        )
        (sample,) = _prepare_samples(
            sources,
            mode="i2v",
            spatial_mode="spacetime",
            crossfade_tokens=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            _write_static_artifacts(
                artifact_dir,
                sample,
                mode="i2v",
                spatial_mode="spacetime",
            )
            route = json.loads(
                (artifact_dir / "trace.route.compiled.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(route["requested_mode"], "i2v")
            self.assertEqual(route["stage_ids"], [
                "setup", "onset", "evolution", "completion", "terminal"
            ])
            self.assertEqual(len(route["spatial_masks"]), 5)


if __name__ == "__main__":
    unittest.main()
