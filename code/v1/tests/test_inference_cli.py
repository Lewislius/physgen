from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


V1_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = V1_ROOT / "inference" / "infer_ace_router.py"
SPEC = importlib.util.spec_from_file_location("infer_ace_router", ENTRYPOINT)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class InferenceCliTests(unittest.TestCase):
    def test_method_matrix_matches_requested_conditioning(self) -> None:
        self.assertEqual(CLI._residual_mode("M0"), "none")
        self.assertEqual(CLI._residual_mode("M1"), "none")
        self.assertEqual(CLI._residual_mode("M2"), "positive_only")
        self.assertEqual(
            CLI._residual_mode("M3"), "positive_minus_counterfactual"
        )
        self.assertIn("M1", CLI.I2V_METHODS)
        self.assertIn("M2", CLI.I2V_METHODS)
        self.assertNotIn("M0", CLI.I2V_METHODS)

    def test_m0_and_m4_default_to_fixed_landscape(self) -> None:
        for method in ("M0", "M4"):
            args = CLI._build_parser().parse_args(["--method", method])
            self.assertEqual(CLI._t2v_orientation_policy(args), "landscape")
            self.assertIn(method, CLI.T2V_METHODS)

    def test_generation_defaults_match_high_resolution_pilot(self) -> None:
        args = CLI._build_parser().parse_args(["--method", "M4"])
        self.assertEqual(args.m4_orientation, "landscape")
        self.assertEqual(args.frame_num, 97)
        self.assertEqual(args.max_area, 1280 * 704)
        self.assertFalse(hasattr(args, "negative_prompt"))
        CLI._validate_basic_args(args)

    def test_compiled_m3_and_m4_contexts_do_not_prefix_origin(self) -> None:
        sample = CLI.load_sample(V1_ROOT / "demo", "P02", require_image=True)
        for method, expected_variant in (
            ("M3", "compiled_i2v"),
            ("M4", "compiled_t2v"),
        ):
            args = CLI._build_parser().parse_args(
                ["--method", method, "--conditioning_mode", "compiled"]
            )
            conditioning = CLI._conditioning_for(sample, args)
            self.assertEqual(conditioning.semantic_variant, expected_variant)
            self.assertFalse(
                conditioning.positive_text.startswith(sample.original_prompt)
            )
            self.assertFalse(
                conditioning.counterfactual_text.startswith(sample.original_prompt)
            )

    def test_initial_v1_context_and_cfg_modes_are_explicit(self) -> None:
        sample = CLI.load_sample(V1_ROOT / "demo", "P02", require_image=True)
        args = CLI._build_parser().parse_args(
            [
                "--method",
                "M3",
                "--conditioning_mode",
                "initial_v1",
                "--cfg_negative_mode",
                "wan_default",
            ]
        )
        conditioning = CLI._conditioning_for(sample, args)
        self.assertTrue(conditioning.causal_origin_prompt_prefixed)
        contexts = CLI._method_context_record(
            sample, args, cfg_negative_prompt="wan-default-test"
        )
        self.assertEqual(contexts["cfg_negative_prompt"], "wan-default-test")
        self.assertTrue(contexts["used_by_model"]["cfg_negative"])
        self.assertFalse(contexts["used_by_model"]["cfg_unconditional_empty"])

    def test_m1_and_m2_require_existing_first_frames(self) -> None:
        for method in ("M1", "M2"):
            args = CLI._build_parser().parse_args(
                ["--method", method, "--sample_ids", "P01"]
            )
            samples = CLI._load_samples(args, ["P01"])
            self.assertIsNotNone(samples[0].image_path)

    def test_resolved_configs_make_baseline_and_cplus_usage_explicit(self) -> None:
        sample = CLI.load_sample(V1_ROOT / "demo", "P01", require_image=True)
        for method, enabled, positive, counterfactual in (
            ("M1", False, False, False),
            ("M2", True, True, False),
            ("M3", True, True, True),
        ):
            args = CLI._build_parser().parse_args(["--method", method])
            config = CLI._resolved_config(
                args,
                sample,
                42,
                CLI._resolved_size(sample, args),
                [0.0] * 30,
                [0.0] * 50,
                "test-run",
                {},
            )
            self.assertEqual(config["ace"]["enabled"], enabled)
            self.assertEqual(
                config["conditioning"]["positive_context_injected"], positive
            )
            self.assertEqual(
                config["conditioning"]["counterfactual_context_injected"],
                counterfactual,
            )
            self.assertFalse(config["conditioning"]["negative_prompt_enabled"])
            contexts = CLI._method_context_record(sample, args)
            self.assertEqual(contexts["cfg_negative_prompt"], "")
            self.assertFalse(contexts["used_by_model"]["cfg_negative"])
            self.assertTrue(contexts["used_by_model"]["cfg_unconditional_empty"])

    def test_m4_explicit_match_policy_falls_back_when_image_is_absent(self) -> None:
        sample = CLI.load_sample(
            V1_ROOT / "demo", "P01", require_image=False, inspect_image=False
        )
        args = CLI._build_parser().parse_args(
            ["--method", "M4", "--m4_orientation", "match_image"]
        )
        orientation, reference_size = CLI._sample_orientation(sample, args)
        self.assertEqual(orientation, "landscape")
        self.assertEqual(reference_size, CLI.LANDSCAPE_REFERENCE_SIZE)
        self.assertIn(
            "sizing_image_missing_fallback_landscape",
            CLI._method_warnings(sample, args),
        )

    def test_non_finite_generation_values_are_rejected(self) -> None:
        for argument in ("--lambda0", "--guide_scale", "--shift"):
            args = CLI._build_parser().parse_args(
                ["--method", "M3", argument, "nan"]
            )
            with self.assertRaises(ValueError, msg=argument):
                CLI._validate_basic_args(args)
        for value in ("nan", "0", "-0.1"):
            args = CLI._build_parser().parse_args(
                ["--method", "M3", "--residual_cap_ratio", value]
            )
            with self.assertRaises(ValueError, msg=value):
                CLI._validate_basic_args(args)

    def test_batch_seeds_are_reproducible_and_unique(self) -> None:
        self.assertEqual(CLI._parse_seeds("7,42"), [7, 42])
        with self.assertRaises(ValueError):
            CLI._parse_seeds("-1")
        with self.assertRaises(ValueError):
            CLI._parse_seeds("42,42")


if __name__ == "__main__":
    unittest.main()
