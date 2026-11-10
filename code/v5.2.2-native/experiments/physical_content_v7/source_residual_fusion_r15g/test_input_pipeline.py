"""Exercise real original caches and the partial/full evaluation contract on CPU."""
from pathlib import Path
import contextlib
import copy
import io
import json
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from benchmark_inputs import ROOT, reference, case_input
from condition_controls import InferenceCache
from physical_content.common import load_config
from physical_content.teacher import Teacher
import report_results
import prepare_visual_review
from completed_suite import completion

HERE = Path(__file__).resolve().parent


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.cfg = load_config(HERE / 'configs/physical_content.yaml')
        cls.reader = InferenceCache(cls.cfg['paths']['inference_evidence'], device='cpu')

    def test_guidance_metadata_compatible_with60_merger(self):
        from guidance import description,LEGACY,NATIVE_PLUS_PHYSICS
        sys.path.insert(0,str(ROOT))
        from compare_iteration import effective_guidance
        plan=dict(variant='physical_content',mode='real',guidance=description(LEGACY,5.))
        old=effective_guidance(plan)
        self.assertEqual((old['text_scale'],old['physical_scale']),(5.,5.))
        alternate=effective_guidance(dict(plan,guidance=description(NATIVE_PLUS_PHYSICS,5.)))
        self.assertEqual((alternate['text_scale'],alternate['physical_scale']),(5.,1.))
        self.assertIsNone(effective_guidance(dict(plan,mode='no_c'))['physical_scale'])

    def test_all_original_inputs_and_control_requests(self):
        controls = {v['id'] for v in json.loads((HERE / 'subset16.json').read_text())['cases']}
        count = 0
        for case in reference()['cases']:
            _, item = case_input(case)
            normal = self.reader.input(item, image=item['image'], duration=item['duration'])
            negative = self.reader.input(item, image=item['image'], duration=item['duration'], drop=True)
            self.assertEqual(normal['request_word_strings'], case['caption'].split())
            self.assertEqual(negative['request_words'].shape, (0, 4096))
            count += 2
            if case['id'] in controls:
                for mode in ('empty_reasoning', 'reasoning_swapped'):
                    changed = self.reader.input(item, image=item['image'], duration=item['duration'], mode=mode)
                    self.assertEqual(changed['caption'], normal['caption'])
                    self.assertEqual(changed['own_relation_count'], len(normal['relations']))
                    self.assertEqual(changed['request_word_strings'], normal['request_word_strings'])
                    if mode == 'empty_reasoning':
                        self.assertFalse(any(changed['relations']))
                    else:
                        self.assertNotEqual(changed['donor'], changed['sample'])
                    count += 1
        self.assertEqual(count, 152)

    def test_compaction_preserves_original_tensors_and_catches_wrong_input(self):
        teacher = Teacher(self.cfg, device='cpu')
        for index in (9, 40):
            case = reference()['cases'][index]
            _, item = case_input(case)
            for drop in (False, True):
                arm = 'negative' if drop else 'positive'
                original = (teacher.input(item, drop=drop) if index < 40 else
                            torch.load(ROOT / 'videophy60/inputs' / case['id'] / (arm + '.pt'),
                                       map_location='cpu', weights_only=True))
                compact = self.reader.input(item, image=item['image'], duration=item['duration'], drop=drop)
                for key in ('words', 'word_groups', 'word_positions', 'request_words', 'memory', 'memory_mask'):
                    torch.testing.assert_close(compact[key], original[key], rtol=0, atol=0)
            wrong = dict(item, caption='A deliberately different literal request.')
            with self.assertRaises(ValueError):
                self.reader.input(wrong, image=item['image'], duration=item['duration'])

    def test_reports_partial40_then60_without_changing_historical_scores(self):
        # Historical scores are fixtures for aggregation, never new R15G results.
        previous = ROOT / 'request_memory_r11b/inference_outputs/physical_content_67308/step0400_api_fixed60_controls16_seed42'
        with tempfile.TemporaryDirectory(prefix='r15g_report_contract_') as tmp:
            destination = Path(tmp)
            for folder in ('videophy40', 'controls/request_memory_off/videophy40'):
                (destination / folder).mkdir(parents=True)
                shutil.copyfile(previous / folder / 'scores.json', destination / folder / 'scores.json')
            with patch.object(sys, 'argv', ['report_results.py', '--output', str(destination)]), contextlib.redirect_stdout(io.StringIO()):
                report_results.main()
            result = json.loads((destination / 'comparison_report.json').read_text())
            self.assertEqual(result['pending_main'], ['full60'])
            self.assertEqual(result['main']['original40']['joint_pass_count'], 6)
            self.assertFalse(result['primary_numeric_targets_met'])
            self.assertEqual(result['historical_joint_reference']['required_minimum'], 12)
            self.assertEqual(result['controls']['request_memory_off']['normal_minus_control']['sa']['losses'], 5)
            folder = 'full60/videophy60'
            (destination / folder).mkdir(parents=True)
            shutil.copyfile(previous / folder / 'scores.json', destination / folder / 'scores.json')
            with patch.object(sys, 'argv', ['report_results.py', '--output', str(destination)]), contextlib.redirect_stdout(io.StringIO()):
                report_results.main()
            result = json.loads((destination / 'comparison_report.json').read_text())
            self.assertEqual(result['pending_main'], [])
            self.assertEqual(result['main']['full60']['count'], 60)
            self.assertFalse(result['quality_goal_achieved'])

    def test_completed_suite_reuse_and_partial_scoring_recovery(self):
        # Use historical metadata/video paths as fixtures without reading video contents.
        source=ROOT/'execution_value_r10/snapshots/20261101T034557Z'
        checkpoint=ROOT/'execution_value_r10/runs/physical_content_67003/step0400.pt'
        ids=['VP001','VP002']
        with tempfile.TemporaryDirectory(prefix='r15g_reuse_contract_') as tmp:
            folder=Path(tmp)
            plan=dict(cases=[dict(id=i) for i in ids],checkpoint=str(checkpoint),training_source=str(source),
                      checkpoint_step=400,mode='real',generation_seed=42)
            (folder/'manifest.json').write_text(json.dumps(plan))
            (folder/'summary.json').write_text(json.dumps(dict(status='completed',completed_cases=2)))
            self.assertEqual(completion(folder,checkpoint,source,ids,'real'),(False,False))
            for case in ids:
                video=folder/'cases'/case/'planned.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'fixture')
                video.with_suffix('.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),step=400,
                    mode='real',case=case,seed=42,inference_source=str(source))))
            self.assertEqual(completion(folder,checkpoint,source,ids,'real'),(True,False))
            (folder/'scores.json').write_text(json.dumps(dict(cases=[dict(id=i,sa=3,pc=4) for i in ids])))
            self.assertEqual(completion(folder,checkpoint,source,ids,'real'),(True,True))
            with self.assertRaises(ValueError):completion(folder,checkpoint,source/'different',ids,'real')

    def test_visual_review_uses_explicit_output_and_real_matched_video_metadata(self):
        # Historical R15D is only a temporary I/O fixture, never an R15G result.
        previous=ROOT/'denoising_matched_fusion_r15d/inference_outputs/physical_content_67920/step0400_original_fixed40_controls16_seed42'
        with tempfile.TemporaryDirectory(prefix='r15g_visual_fixture_') as tmp:
            report=Path(tmp)
            with patch.object(sys,'argv',['prepare_visual_review.py','--output',str(previous),'--report',str(report)]),contextlib.redirect_stdout(io.StringIO()):
                prepare_visual_review.main()
            for case in prepare_visual_review.CASES:
                value=json.loads((report/f'{case}_selection.json').read_text())
                self.assertEqual(value['videos']['R15G'],str(previous/'videophy40/cases'/case/'planned.mp4'))
                self.assertIn('R15D',value['videos']);self.assertIn('R1',value['videos'])
                self.assertTrue(Path(value['panels']['R15G']).is_file())
                self.assertEqual(value['metadata']['R15G']['seed'],42)
                self.assertTrue(value['selected_before_R15G_scores'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
