from __future__ import annotations

import unittest
from pathlib import Path

from ace_router.schema import discover_sample_ids, load_sample


V1_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = V1_ROOT / "demo"


class SchemaTests(unittest.TestCase):
    def test_all_twenty_demo_samples_validate(self) -> None:
        sample_ids = discover_sample_ids(DEMO_ROOT)
        self.assertEqual(sample_ids, [f"P{index:02d}" for index in range(1, 21)])
        for sample_id in sample_ids:
            sample = load_sample(DEMO_ROOT, sample_id, require_image=True)
            self.assertEqual(sample.sample_id, sample_id)
            self.assertTrue(sample.image_path and sample.image_path.is_file())
            self.assertEqual(
                sample.image_path.name, f"{sample_id}-i0-1280x704.png"
            )
            self.assertTrue(sample.semantic_text)
            self.assertNotEqual(sample.positive_text, sample.counterfactual_text)
            self.assertEqual(sample.positive_text, sample.causal_positive)
            self.assertEqual(sample.counterfactual_text, sample.causal_counterfactual)
            self.assertFalse(sample.positive_text.startswith(sample.semantic_text))
            self.assertFalse(sample.counterfactual_text.startswith(sample.semantic_text))

    def test_selection_manifest_is_explicitly_optional_for_current_demo(self) -> None:
        sample = load_sample(DEMO_ROOT, "P01", require_image=True)
        self.assertIsNone(sample.selection_manifest_path)
        self.assertIn("selection_manifest_missing", sample.warnings)
        with self.assertRaises(ValueError):
            load_sample(
                DEMO_ROOT,
                "P01",
                require_image=True,
                require_selection_manifest=True,
            )


if __name__ == "__main__":
    unittest.main()
