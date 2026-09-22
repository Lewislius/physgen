from __future__ import annotations

import unittest
from pathlib import Path

from ace_router.conditioning import compile_conditioning
from ace_router.schema import load_sample


V1_ROOT = Path(__file__).resolve().parents[1]
DEMO_ROOT = V1_ROOT / "demo"


class ConditioningTests(unittest.TestCase):
    def test_compiled_i2v_uses_metadata_but_keeps_causal_pair_relation_only(self) -> None:
        sample = load_sample(DEMO_ROOT, "P02", require_image=True)
        compiled = compile_conditioning(
            sample, mode="compiled", use_initial_image=True
        )
        self.assertEqual(compiled.semantic_variant, "compiled_i2v")
        self.assertIn("Required visible entities", compiled.semantic_text)
        self.assertIn("Required state changes", compiled.semantic_text)
        self.assertIn("Preserve throughout", compiled.semantic_text)
        self.assertNotIn("Required layout", compiled.semantic_text)
        self.assertEqual(compiled.positive_text, sample.causal_positive)
        self.assertEqual(
            compiled.counterfactual_text, sample.causal_counterfactual
        )
        self.assertFalse(
            compiled.positive_text.startswith(sample.original_prompt)
        )
        self.assertFalse(
            compiled.counterfactual_text.startswith(sample.original_prompt)
        )
        self.assertFalse(
            compiled.to_dict()["causal_pair"]["origin_prompt_prefixed"]
        )

    def test_compiled_t2v_adds_initial_scene_and_layout(self) -> None:
        sample = load_sample(
            DEMO_ROOT, "P02", require_image=False, inspect_image=False
        )
        compiled = compile_conditioning(
            sample, mode="compiled", use_initial_image=False
        )
        self.assertEqual(compiled.semantic_variant, "compiled_t2v")
        self.assertIn("Initial scene", compiled.semantic_text)
        self.assertIn("Required layout", compiled.semantic_text)
        self.assertGreater(
            len(compiled.semantic_text),
            len(
                compile_conditioning(
                    sample, mode="compiled", use_initial_image=True
                ).semantic_text
            ),
        )

    def test_original_mode_still_uses_relation_only_causal_contexts(self) -> None:
        sample = load_sample(DEMO_ROOT, "P02", require_image=True)
        compiled = compile_conditioning(
            sample, mode="original", use_initial_image=True
        )
        self.assertEqual(compiled.semantic_text, sample.original_prompt)
        self.assertEqual(compiled.positive_text, sample.causal_positive)
        self.assertEqual(
            compiled.counterfactual_text, sample.causal_counterfactual
        )

    def test_initial_v1_mode_reproduces_prefixed_causal_contexts(self) -> None:
        sample = load_sample(DEMO_ROOT, "P02", require_image=True)
        compiled = compile_conditioning(
            sample, mode="initial_v1", use_initial_image=True
        )
        self.assertEqual(compiled.semantic_text, sample.original_prompt)
        self.assertEqual(
            compiled.positive_text,
            f"{sample.original_prompt} {sample.causal_positive}",
        )
        self.assertEqual(
            compiled.counterfactual_text,
            f"{sample.original_prompt} {sample.causal_counterfactual}",
        )
        self.assertTrue(compiled.causal_origin_prompt_prefixed)
        self.assertTrue(compiled.to_dict()["causal_pair"]["origin_prompt_prefixed"])


if __name__ == "__main__":
    unittest.main()
