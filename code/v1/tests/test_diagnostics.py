from __future__ import annotations

import unittest

import torch

from ace_router.diagnostics import ResidualDiagnostics


class ResidualDiagnosticsTests(unittest.TestCase):
    def test_capped_candidate_and_applied_values_are_reported(self) -> None:
        diagnostics = ResidualDiagnostics("full")
        semantic = torch.ones(1, 2, 2)
        positive = torch.full((1, 2, 2), 3.0)
        counterfactual = torch.full((1, 2, 2), 2.0)
        candidate = torch.full((1, 2, 2), 0.5)
        applied = torch.full((1, 2, 2), 0.1)
        diagnostics.record(
            residual_mode="positive_minus_counterfactual",
            step_index=0,
            total_steps=50,
            layer_id=14,
            layer_gate=1.0,
            step_gate=1.0,
            lambda0=0.5,
            semantic=semantic,
            positive=positive,
            counterfactual=counterfactual,
            candidate_delta=candidate,
            applied_delta=applied,
            clip_coefficient=torch.full((1, 1, 1), 0.2),
            residual_cap_ratio=0.1,
        )
        record = diagnostics.records[0]
        self.assertAlmostEqual(record["candidate_to_semantic"], 0.5)
        self.assertAlmostEqual(record["applied_to_semantic"], 0.1)
        self.assertAlmostEqual(record["lambda_effective_min"], 0.1)
        self.assertTrue(record["was_clipped"])
        summary = diagnostics.summary()
        self.assertEqual(summary["clipped_record_count"], 1)
        self.assertAlmostEqual(summary["clip_fraction"], 1.0)
        self.assertAlmostEqual(summary["applied_to_semantic_max"], 0.1)


if __name__ == "__main__":
    unittest.main()
