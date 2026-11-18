"""CPU contracts for R16Gt: T2V scope, R16G design, data pairing, evidence."""
import json
import re
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

from physical_content.common import load_config
from physical_content.data import Data
from physical_content.supervision import flow_matching, generation_loss


class Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(ROOT / 'configs/r16gt.yaml')

    def test_task_is_t2v_and_has_no_known_frame(self):
        cfg = self.cfg
        self.assertEqual(cfg['task'], 't2v')
        self.assertFalse(cfg['input']['external_image'])
        self.assertEqual(cfg['input']['known_latent_frames'], 0)
        self.assertEqual(cfg['input']['evidence_scope'], 'request_only')
        self.assertFalse(cfg['input']['future_targets_in_condition'])
        self.assertEqual(cfg['data']['latent_shape'], [1, 48, 21, 30, 52])
        self.assertEqual(cfg['train']['validation_steps'], [])

    def test_r16g_design_is_locked(self):
        model = self.cfg['model']
        self.assertEqual(model['candidate_label'], 'R16Gt')
        self.assertEqual(model['response_mode'], 'triple_global_pure_local')
        self.assertEqual(model['global_execution'],
                         ['direct', 'request_bound', 'joint'])
        self.assertEqual(model['fusion_order'], 'triple_global_pure_r15c_local')
        self.assertEqual(model['local_query_condition'], 'direct_read_only')
        self.assertEqual(model['local_execution_operator'],
                         'r15c_5x3x3_video_state_V')
        self.assertEqual(model['value_correction'], 'none')
        self.assertEqual(model['physical_window'], [5, 3, 3])
        self.assertEqual(model['native_text_adaptation'], 'frozen')
        self.assertEqual(self.cfg['inference']['guidance_composition'],
                         'joint_two_branch_cfg')

    def test_generation_loss_task_boundary(self):
        prediction = torch.zeros(1, 2, 3, 4, 4)
        clean = torch.zeros_like(prediction)
        noise = torch.zeros_like(prediction)
        prediction[:, :, 0] = 1.0
        valid = torch.ones(4, 4)
        i2v, _ = generation_loss(prediction, noise, clean, valid, tail_weight=0,
                                 known_frames=1)
        t2v, _ = generation_loss(prediction, noise, clean, valid, tail_weight=0,
                                 known_frames=0)
        self.assertEqual(float(i2v), 0.0)
        self.assertGreater(float(t2v), 0.0)
        i2v_fm = flow_matching(prediction, noise, clean, valid, known_frames=1)
        t2v_fm = flow_matching(prediction, noise, clean, valid, known_frames=0)
        self.assertEqual(float(i2v_fm), 0.0)
        self.assertGreater(float(t2v_fm), 0.0)

    def test_same_1600_draws_have_matching_evidence(self):
        from physgen_native.data import SampleOrder
        rows = json.loads((Path(self.cfg['paths']['cache'])
                           / 'base/manifest.json').read_text())['records']
        train = [row for row in rows if row['split'] == 'train']
        self.assertEqual(len(train), 3666)
        data = Data(self.cfg)
        self.assertEqual(len(data), 1600)
        order = iter(SampleOrder(len(data), 42, 0))
        draws = [data.rows[next(order)]['id'] for _ in range(1600)]
        self.assertEqual(len(set(draws)), 1600)
        by_id = {row['id']: row for row in data.rows}
        missing = [sample for sample in draws if not data.has_evidence(by_id[sample])]
        self.assertEqual(missing, [])
        # The unique T2V sample set is exactly the 1600 unique samples used by
        # the R16G I2V run, which draws from the original 3666-row WISA order.
        wisa = json.loads((Path('/home/liuzhirui/Project/physGen/data_cache/wisa_4000')
                           / 'base/manifest.json').read_text())['records']
        wisa_train = [row for row in wisa if row['split'] == 'train']
        self.assertEqual(len(wisa_train), 3666)
        wisa_order = iter(SampleOrder(len(wisa_train), 42, 0))
        wisa_draws = {wisa_train[next(wisa_order)]['id'] for _ in range(1600)}
        self.assertEqual(wisa_draws, {row['id'] for row in data.rows})
        input_root = Path(self.cfg['paths']['t2v_input_cache'])
        mismatched = []
        for sample in draws:
            row = by_id[sample]
            value = torch.load(input_root / f'{sample}_text.pt',
                               map_location='cpu', weights_only=True)
            if (value['caption'] != row['caption']
                    or value['request_word_strings'] != row['caption'].split()):
                mismatched.append(sample)
        self.assertEqual(mismatched, [])

    def test_data_ready_checks_caption_source_and_interface(self):
        from physical_content.common import data_ready
        info = data_ready(self.cfg)
        self.assertEqual(info['matched_draw_samples'], 1600)
        self.assertEqual(info['unique_draw_samples'], 1600)
        self.assertEqual(info['evidence_scope'], 'request_only')

    def test_t2v_data_view_has_no_first_frame(self):
        data = Data(self.cfg)
        index = next(i for i, row in enumerate(data.rows) if data.has_evidence(row))
        sample = data.get(index, device='cpu')
        self.assertNotIn('first', sample)
        self.assertNotIn('first_rgb', sample)
        self.assertEqual(tuple(sample['latent'].shape), (1, 48, 21, 30, 52))
        self.assertEqual(tuple(sample['times'].shape), (1, 21))
        self.assertEqual(tuple(sample['valid'].shape), (30, 52))

    def test_prepared_eval_evidence_is_request_only(self):
        root = Path(self.cfg['paths']['inference_evidence'])
        for name in ('videophy40_VP001', 'videophy60_VP060'):
            ready = json.loads((root / name / 'ready.json').read_text())
            self.assertEqual(ready['scope'], 'request_only')
            self.assertEqual(ready['revision'], 'r16gt_request_only_v1')
            self.assertNotIn('image', ready)
            positive = torch.load(root / name / 'positive.pt', weights_only=True)
            negative = torch.load(root / name / 'negative.pt', weights_only=True)
            self.assertEqual(positive['scope'], 'request_only')
            self.assertEqual(negative['scope'], 'request_only')
            self.assertNotIn('image', positive)
            self.assertEqual(negative['request_word_strings'], [])
            self.assertEqual(positive['request_word_strings'],
                             positive['caption'].split())
            native_folder = Path(self.cfg['paths']['eval_native_inputs']) / ready['case']
            native = torch.load(native_folder / 'conditions.pt',
                                weights_only=True)
            self.assertEqual(set(native), {'text', 'null_text', 'times'})
            info = json.loads((native_folder / 'input.json').read_text())
            self.assertEqual(info['prompt'], ready['caption'])
            self.assertFalse(info['image'])
            self.assertFalse(info['first_latent'])

    def test_eval_native_inputs_have_only_text_null_and_times(self):
        root = Path(self.cfg['paths']['eval_native_inputs'])
        folders = sorted(path for path in root.iterdir() if path.is_dir())
        self.assertEqual(len(folders), 60)
        for folder in folders:
            value = torch.load(folder / 'conditions.pt', weights_only=True)
            self.assertEqual(set(value), {'text', 'null_text', 'times'}, folder.name)
            self.assertEqual(tuple(value['times'].shape), (21,))
            self.assertTrue(torch.allclose(value['times'],
                                           torch.arange(21).float() / 6,
                                           atol=1e-5, rtol=0))

    def test_case_input_reads_the_declared_native_root(self):
        from benchmark_inputs import case_input, reference
        case = reference(self.cfg)['cases'][0]
        sample, item = case_input(case, self.cfg['paths']['eval_native_inputs'])
        self.assertEqual(set(sample), {'text', 'null_text', 'times'})
        self.assertEqual(item['case'], case['id'])
        self.assertEqual(item['caption'], case['caption'])
        self.assertNotIn('first', sample)

    def test_infer_case_record_does_not_duplicate_provenance_keys(self):
        from infer import case_record
        provenance = dict(
            checkpoint='checkpoint', step=400, mode='real', case='VP001',
            seed=42, task='t2v', first_frame_used=False,
            inference_revision='revision', inference_source='source',
            inference_evidence='evidence', guidance={'kind': 'cfg'})
        record = case_record(
            provenance, {'id': 'VP001', 'caption': 'caption'},
            {'steps': 50}, {'analysis': 'positive', 'relations': ['r']},
            {'analysis': 'negative'}, 'scope', 1.25)
        self.assertIs(record['first_frame_used'], False)
        self.assertEqual(record['known_latent_frames'], 0)
        self.assertEqual(record['caption'], 'caption')
        self.assertEqual(record['scope'], 'scope')
        self.assertEqual(record['seconds'], 1.25)
        self.assertEqual(record['positive_relations'], ['r'])

    def test_source_has_no_first_frame_use(self):
        forbidden = (re.compile(r"sample\[['\"]first['\"]\]"),
                     re.compile(r"first_rgb"),
                     re.compile(r"x\[\s*:\s*,\s*:\s*,\s*:\s*1\s*\]\s*="))
        allowed = ("del sample['first']", "'first' in sample")
        offenders = []
        for path in ROOT.rglob('*.py'):
            if ('tests' in path.parts or 'snapshots' in path.parts
                    or path.name == 'verify_r18_consistency.py'):
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if any(pattern.search(line) for pattern in forbidden):
                    if any(token in line for token in allowed):
                        continue
                    offenders.append(f'{path.relative_to(ROOT)}:{number}: {line.strip()}')
        self.assertEqual(offenders, [])


if __name__ == '__main__':
    unittest.main()
