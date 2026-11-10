"""Catch wrong checkpoint architectures, changed replay inputs and score arithmetic."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'inference'))
import compare
from jobs import JOBS, Q_REFERENCE


class BenchmarkTests(unittest.TestCase):
    def test_archived_q_replay_reproduces_summary_metrics_and_zero_q_deltas(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = compare.read(Q_REFERENCE / 'manifest.json')
            manifest['workspace_variant'] = 'b'
            (root / 'manifest.json').write_text(json.dumps(manifest))
            for name in ('scores.json', 'summary.json'):
                (root / name).write_text((Q_REFERENCE / name).read_text())
            (root / 'cases').symlink_to(Q_REFERENCE / 'cases', target_is_directory=True)
            result = compare.compare(root, 'b')
            row = result['summaries'][0]
            self.assertEqual((row['sa_total'], row['pc_total'], row['mean_sa_pc'], row['joint_percent']),
                             (121, 138, 3.2375, 22.5))
            self.assertEqual((row['sa_pass_count'], row['pc_pass_count'], row['joint_count']), (11, 16, 9))
            delta = result['candidate_minus_reference'][0]
            self.assertEqual((delta['sa_delta'], delta['pc_delta'], delta['mean_delta'], delta['joint_delta_pp']),
                             (0, 0, 0, 0))
            self.assertEqual((delta['average_wins'], delta['average_ties'], delta['average_losses']), (0, 40, 0))
            self.assertEqual(len(result['summaries']), 10)
            self.assertEqual(len((root / 'scores_by_case_40.csv').read_text().splitlines()), 41)
            self.assertEqual(len((root / 'paired_differences.csv').read_text().splitlines()), 361)

    def test_changed_noise_image_caption_sampler_and_evaluator_are_rejected(self):
        original = compare.read(Q_REFERENCE / 'manifest.json')
        for key, replacement in [('initial_noise', '/tmp/other_noise.pt'),
                                 ('image', '/tmp/other_image.png'), ('caption', 'Changed caption')]:
            candidate = copy.deepcopy(original)
            candidate['cases'][0][key] = replacement
            with self.assertRaises(ValueError):
                compare.validate_manifest(candidate)
        for field in ('sampling', 'evaluation'):
            candidate = copy.deepcopy(original)
            candidate[field]['seed'] = 7
            with self.assertRaises(ValueError):
                compare.validate_manifest(candidate)
        candidate = copy.deepcopy(original)
        candidate['cases'].pop()
        with self.assertRaises(ValueError):
            compare.validate_manifest(candidate)

    def test_missing_noninteger_or_out_of_range_scores_are_rejected(self):
        rows = [dict(sa=3, pc=4) for _ in range(40)]
        for value in (0, 6, 3.5, True):
            changed = copy.deepcopy(rows)
            changed[0]['sa'] = value
            with self.assertRaises(ValueError):
                compare.summarize('test', 'test', changed)
        with self.assertRaises(ValueError):
            compare.summarize('test', 'test', rows[:-1])

    def test_all_jobs_use_one_amp80g_slot_and_their_own_checkpoint(self):
        import yaml
        for variant, job in JOBS.items():
            config = yaml.safe_load(Path(job['yaml']).read_text())
            self.assertEqual(config['resources']['resource_pool'], 'amp-80g')
            self.assertEqual(config['resources']['slots_per_trial'], 1)
            self.assertIn('est3h', config['name'])
            self.assertEqual(config['entrypoint'], f"bash {job['sh']} --local")
            self.assertIn(str(job['training_experiment_id']), job['checkpoint'])
            self.assertNotIn('PYTHONNOUSERSITE', json.dumps(config['environment']))
        self.assertEqual(len({job['revision'] for job in JOBS.values()}), 4)
        self.assertEqual(len({job['output'] for job in JOBS.values()}), 4)


if __name__ == '__main__':
    unittest.main()
