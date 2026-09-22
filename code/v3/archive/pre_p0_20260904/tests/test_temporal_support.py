from __future__ import annotations

import json
from pathlib import Path
import unittest

import torch

from v2.ace_router.trace_schema import TracePlan

from trace_writer_v3.compile import compile_trace_plan_v3
from trace_writer_v3.dynamic_support import build_dynamic_support
from trace_writer_v3.dynamic_support import OperatorSaliencyState


DEMO_ROOT = Path("/home/liuzhirui/Project/physGen/code/v1/demo")


def compiled(sample: str, mode: str):
    suffix = "planimg" if mode == "i2v" else "plan"
    path = DEMO_ROOT / sample / f"{sample}-V2-{suffix}.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return compile_trace_plan_v3(TracePlan.from_dict(raw), raw, mode=mode)


class TemporalSupportTests(unittest.TestCase):
    def test_temporal_and_dynamic_support_contracts(self) -> None:
        value = compiled("P01", "i2v").temporal_weights
        self.assertEqual(tuple(value.shape), (5, 25))
        self.assertTrue(torch.allclose(value.sum(0), torch.ones(25)))
        self.assertTrue(bool(((value > 0).sum(0) <= 2).all().item()))
        for left in range(5):
            for right in range(left + 2, 5):
                self.assertFalse(bool(((value[left] > 0) & (value[right] > 0)).any().item()))

        value, manifest = build_dynamic_support(
            compiled("P01", "i2v"),
            height=22,
            width=40,
            support_mode="planned",
        )
        self.assertEqual(tuple(value.shape), (5, 25, 22, 40))
        self.assertTrue(bool(((value >= 0) & (value <= 1)).all().item()))
        self.assertEqual(manifest["source"], "plan_boxes_and_tracks")

        value, manifest = build_dynamic_support(
            compiled("P02", "t2v"),
            height=22,
            width=40,
            support_mode="planned_saliency",
        )
        self.assertEqual(tuple(value.shape), (5, 25, 22, 40))
        self.assertTrue(manifest["requires_runtime_saliency"])
        self.assertEqual(manifest["source"], "operator_saliency_seed")

        state = OperatorSaliencyState(2, 2, 5, max_area=0.3, ema=0.8)
        indices = torch.arange(20)
        positive = torch.arange(1, 21, dtype=torch.float32).view(1, 20, 1)
        local = state.local_mask("setup", positive, indices, 20)
        self.assertEqual(tuple(local.shape), (1, 20, 1))
        combined = state.combined_support(device="cpu")
        self.assertIsNotNone(combined)
        assert combined is not None
        per_frame = (combined > 0).sum((-1, -2))
        self.assertTrue(bool((per_frame <= 3).all().item()))


if __name__ == "__main__":
    unittest.main()
