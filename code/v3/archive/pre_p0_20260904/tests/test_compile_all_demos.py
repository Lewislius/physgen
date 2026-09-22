from __future__ import annotations

import glob
import json
from pathlib import Path
import unittest

from v2.ace_router.trace_schema import TracePlan

from trace_writer_v3.compile import compile_trace_plan_v3


DEMO_ROOT = Path("/home/liuzhirui/Project/physGen/code/v1/demo")


class CompileAllDemosTests(unittest.TestCase):
    def test_all_twenty_m3_and_m4_plans_compile_strict(self) -> None:
        counts = {"i2v": 0, "t2v": 0}
        for mode, suffix in (("i2v", "planimg"), ("t2v", "plan")):
            files = sorted(glob.glob(str(DEMO_ROOT / "P??" / f"P??-V2-{suffix}.json")))
            self.assertEqual(len(files), 20)
            for filename in files:
                raw = json.loads(Path(filename).read_text(encoding="utf-8"))
                compiled = compile_trace_plan_v3(
                    TracePlan.from_dict(raw),
                    raw,
                    mode=mode,
                    minimal_pair_mode="strict",
                )
                self.assertEqual(len(compiled.contexts.stage_pairs), 5)
                self.assertTrue(
                    all(x.pair_quality == "minimal_verified" for x in compiled.contexts.stage_pairs)
                )
                self.assertTrue(
                    all(item in compiled.contexts.global_anchor for item in compiled.plan.protected_predicates)
                )
                counts[mode] += 1
        self.assertEqual(counts, {"i2v": 20, "t2v": 20})

    def test_entity_lifecycle_and_conservation_contract(self) -> None:
        balloon_path = DEMO_ROOT / "P10" / "P10-V2-planimg.json"
        balloon_raw = json.loads(balloon_path.read_text(encoding="utf-8"))
        balloon = compile_trace_plan_v3(
            TracePlan.from_dict(balloon_raw), balloon_raw, mode="i2v"
        )
        slot = balloon.entity_ledger.slot("balloon")
        self.assertEqual((slot.cardinality.kind, slot.cardinality.exact), ("exact", 1))
        self.assertIn("[CONSERVATION]", balloon.contexts.global_anchor)

        stream_path = DEMO_ROOT / "P13" / "P13-V2-planimg.json"
        stream_raw = json.loads(stream_path.read_text(encoding="utf-8"))
        stream_plan = compile_trace_plan_v3(
            TracePlan.from_dict(stream_raw), stream_raw, mode="i2v"
        )
        stream = {
            item.stage_id: item
            for item in stream_plan.entity_ledger.stage_presence
            if item.entity_id == "stream"
        }
        self.assertEqual(stream["setup"].logical_count.exact, 0)
        self.assertNotEqual(stream["evolution"].logical_count.exact, 0)
        self.assertEqual(stream["terminal"].logical_count.exact, 0)


if __name__ == "__main__":
    unittest.main()
