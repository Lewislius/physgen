"""Validate old read endpoints, native preservation, padding and one-null mass."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
BASELINE = HERE.parent / "request_bound_fusion_r15b"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(BASELINE / "snapshots/20261105T152223Z"))
import bootstrap
import torch
from physical_content import model as baseline_model
from joint_read import read_physical, fusion_forward

spec = importlib.util.spec_from_file_location("b_test_fixture", BASELINE / "test_request_bound_fusion.py")
fixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)


class JointReadTest(unittest.TestCase):
    setUp = fixture.RequestBoundFusionTest.setUp
    model = fixture.RequestBoundFusionTest.model
    activate = fixture.RequestBoundFusionTest.activate
    full = fixture.RequestBoundFusionTest.full

    def test_original_bound_and_direct_endpoints(self):
        model = self.model()
        self.activate(model)
        bound = self.full(model)
        direct = self.full(model, direct_physical_read=True)
        with patch.object(baseline_model.Fusion, "forward", fusion_forward):
            for fusion in model.fusions.values():
                fusion.memory_mode = "bound"
            torch.testing.assert_close(bound, self.full(model), rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(direct, self.full(model, direct_physical_read=True), rtol=1e-5, atol=1e-6)
            for fusion in model.fusions.values():
                fusion.memory_mode = "joint"
            combined = self.full(model)
            self.assertGreater(float((combined - bound).abs().max()), 1e-7)
            self.assertGreater(float((combined - direct).abs().max()), 1e-7)

    def test_native_initialization_and_checkpoint_ownership_unchanged(self):
        model = self.model()
        names = set(model.state_dict())
        native = self.full(model, physical_enabled=False)
        with patch.object(baseline_model.Fusion, "forward", fusion_forward):
            torch.testing.assert_close(native, self.full(model), rtol=1e-5, atol=1e-6)
        self.assertEqual(names, set(model.state_dict()))
        self.assertFalse(any(p.requires_grad for p in model.wan.blocks[1].cross_attn.parameters()))

    def test_padding_zero_values_and_first_frame(self):
        with patch.object(baseline_model.Fusion, "forward", fusion_forward):
            fixture.RequestBoundFusionTest.test_padding_and_zero_physical_values_cannot_create_a_request_writer(self)

    def test_neutral_relation_prior_and_duplicate_slot_invariance(self):
        groups = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])
        prior = baseline_model.group_log_prior(groups).float()
        q = torch.zeros(1, 2, 3, 4)
        k = torch.zeros(1, 2, 8, 4)
        v = torch.randn_like(k)
        ref = torch.zeros(1, 2, 1, 4)
        for length in [1, 2, 17, 64]:
            request = torch.zeros(1, 2, length, 4)
            direct = read_physical(q, k, v, request, ref, prior, "direct")
            for mode in ["bound", "joint"]:
                torch.testing.assert_close(direct, read_physical(q, k, v, request, ref, prior, mode), rtol=1e-6, atol=1e-6)
            repeated_prior = baseline_model.group_log_prior(groups.repeat_interleave(2)).float()
            torch.testing.assert_close(direct, read_physical(q, k.repeat_interleave(2, -2), v.repeat_interleave(2, -2), request, ref, repeated_prior), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
