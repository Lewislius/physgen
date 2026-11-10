import importlib.util
from pathlib import Path
import tempfile
import json
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('paired_baseline',ROOT/'tools/compare_videophy40.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BenchmarkPairingTests(unittest.TestCase):
    def test_identical_scores_have_zero_paired_delta_and_interval(self):
        rows=[dict(id=str(i),sa=(i%5)+1,pc=((i+1)%5)+1) for i in range(40)]
        result=module.paired_summary(rows,rows,repeats=100)
        for m in result.values():
            self.assertEqual(m['delta'],0);self.assertEqual(m['paired_bootstrap95'],[0.,0.]);self.assertEqual(m['ties'],40)

    def test_scores_cannot_pair_different_or_reordered_cases(self):
        rows=[dict(id='a',sa=4,pc=4),dict(id='b',sa=1,pc=1)]
        with self.assertRaises(ValueError):module.paired_summary(rows,rows[::-1],repeats=10)

    def test_joint_percentage_points_are_not_relative_percentage_gain(self):
        baseline=[dict(id=str(i),sa=4,pc=4 if i<9 else 3) for i in range(40)]
        candidate=[dict(r,pc=4 if i<11 else 3) for i,r in enumerate(baseline)]
        result=module.paired_summary(candidate,baseline,repeats=100)
        self.assertEqual(result['joint_pp']['reference'],22.5)
        self.assertEqual(result['joint_pp']['delta'],5.)
        self.assertAlmostEqual(result['pc']['delta'],.05)


if __name__=='__main__':unittest.main()
