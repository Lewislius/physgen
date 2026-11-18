"""Check actual shared60 input identities and variable-count reasoning controls."""
from pathlib import Path
import json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from benchmark_inputs import reference,case_input
from prepare_inputs import donor_map
from report_results import paired


class InferenceContractTest(unittest.TestCase):
    def test_existing_sixty_inputs_keep_native_requests_images_and_saved_noise(self):
        plan=reference();self.assertEqual(len(plan['cases']),60)
        self.assertEqual(len(plan['extra_ids']),20)
        for case in plan['cases']:
            inputs,item=case_input(case)
            self.assertEqual(item['caption'],case['caption'])
            self.assertTrue(Path(item['image']).is_file())
            self.assertTrue(Path(case['initial_noise']).is_file())
            self.assertEqual(set(inputs),{'first','text','null_text','times','valid'})
            self.assertEqual(item['id'],('videophy60_' if case['id'] in plan['extra_ids'] else 'videophy40_')+case['id'])
        fixed=json.loads((Path(__file__).parent/'subset16.json').read_text())['cases']
        self.assertTrue({v['id'] for v in fixed}<={v['id'] for v in plan['cases'] if v['id'] not in plan['extra_ids']})

    def test_donor_count_and_rare_count_training_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);items=[]
            for name,count in [('VP001',4),('VP002',4),('VP003',2)]:
                sample='videophy40_'+name;folder=root/sample;folder.mkdir()
                value=dict(sample=sample,caption=name,relations=['relation']*count,scope='first_image_and_request_only')
                (folder/'positive.json').write_text(json.dumps(value));items.append(dict(case=name,id=sample))
            train=root/'training';train.mkdir()
            (train/'wisa_fixture_text.json').write_text(json.dumps(dict(sample='wisa_fixture',relations=['a','b'],scope='first_image_and_request_only')))
            mapping,values,_=donor_map(items,root,[v['case'] for v in items],train)
            self.assertEqual(mapping,{'VP001':'VP002','VP002':'VP001','VP003':'train:wisa_fixture'})
            for own,donor in mapping.items():self.assertEqual(len(values[own]['relations']),len(values[donor]['relations']))

    def test_report_pairs_original_ids_and_joint_thresholds(self):
        before=dict(cases=[dict(id='a',sa=4,pc=4),dict(id='b',sa=3,pc=5)])
        after=dict(cases=[dict(id='b',sa=4,pc=5),dict(id='a',sa=4,pc=3)])
        result=paired(after,before)
        self.assertEqual(result['joint'],dict(gains=['b'],losses=['a']))
        self.assertEqual(result['pc']['mean_change'],-.5)


if __name__=='__main__':unittest.main()
