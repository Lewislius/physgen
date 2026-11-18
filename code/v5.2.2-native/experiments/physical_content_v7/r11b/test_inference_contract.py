"""Check actual shared60 input identities and variable-count reasoning controls."""
from pathlib import Path
import json,sys,tempfile,unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from benchmark_inputs import reference,case_input
from prepare_inputs import donor_map
from report_results import paired
from request_controls import donor_map as request_donors,load_control,REVISION
from physical_content.common import load_config
from condition_controls import InferenceCache


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

    def test_actual_request_words_are_complete_and_null_arm_has_no_words(self):
        cfg=load_config(Path(__file__).parent/'configs/physical_content.yaml')
        root=Path(cfg['paths']['inference_evidence'])
        inputs=json.loads((root/'inputs.json').read_text());reader=InferenceCache(root,'cpu')
        for item in inputs['cases']:
            own=reader.input(item,image=item['image'],duration=item['duration'])
            null=reader.input(item,image=item['image'],duration=item['duration'],drop=True)
            self.assertEqual(own['request_word_strings'],item['caption'].split())
            self.assertEqual(own['request_words'].shape,(len(item['caption'].split()),4096))
            self.assertEqual(null['request_word_strings'],[])
            self.assertEqual(null['request_words'].shape,(0,4096))
        donors=request_donors(inputs['cases'],inputs['controls'])
        self.assertEqual(len(donors),16)
        self.assertEqual(len({v['case'] for v in donors.values()}),16)
        self.assertTrue(all(k!=v['case'] for k,v in donors.items()))

    def test_swapped_memory_contract_retains_own_image_and_checks_donor_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);own=dict(case='VP001',id='videophy40_VP001',caption='Own request',image='/tmp/own.jpg',duration=5.)
            value=dict(revision=REVISION,sample=own['id'],own_caption=own['caption'],first_image=own['image'],
                duration=5.,donor='videophy40_VP002',donor_caption='Another request',
                request_words=torch.ones(2,4096),request_word_strings=['Another','request'])
            path=root/'VP001.pt';torch.save(value,path)
            actual=load_control(root,own,'cpu');self.assertEqual(actual['first_image'],own['image'])
            for key,invalid in [('first_image','/tmp/donor.jpg'),('request_word_strings',['truncated']),('donor',own['id'])]:
                torch.save(dict(value,**{key:invalid}),path)
                with self.assertRaises(ValueError):load_control(root,own,'cpu')


if __name__=='__main__':unittest.main()
