"""Meaningful cache/template checks; not model-quality evidence."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import encode
import torch
from annotate import input_reasoning, style_profile
from cache_reader import Teacher


class Compatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tokenizer=AutoTokenizer.from_pretrained('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct',local_files_only=True)

    def test_api_reasoning_receives_only_first_image_and_request(self):
        class Client:
            def call(self,prompt,images,**kwargs):
                self.prompt,self.images=prompt,images
                return {'relations':['If released, the ball can fall under gravity.']},{'request_id':'local-test','prompt':prompt}
        client=Client();image=object()
        result=input_reasoning(image,'A ball falls.',3.,client)
        self.assertEqual(client.images,[image])
        self.assertEqual(result['caption'],'A ball falls.')
        self.assertEqual(result['scope'],'first_image_and_request_only')
        self.assertNotIn('independent_visual_notes',client.prompt)

    def test_style_reference_does_not_reintroduce_old_false_event(self):
        old='The video captures a dragon touching a spaceship.\n\nIt explodes.'
        profile=style_profile(old)
        self.assertEqual(profile['paragraph_word_counts'],[8,2])
        self.assertNotIn('dragon',json.dumps(profile))
        self.assertNotIn('explodes',json.dumps(profile))

    def test_actual_local_tokenizer_preserves_request_and_relation_spans(self):
        tokenizer=self.tokenizer
        class Knowledge:
            cfg={}
            def __init__(self):
                self.tokenizer=tokenizer
                self.backbone=SimpleNamespace(model=self.forward)
            def _inputs(self,clips,body,extract_spans):
                text='Initial observation:\n<|vision_start|>'+'<|image_pad|>'*390+'<|vision_end|>\n'+body
                rendered=tokenizer.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True)
                inputs=tokenizer(rendered,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
                offsets=inputs.pop('offset_mapping')[0]
                positions=[]
                for span in extract_spans:
                    start=rendered.index(span);end=start+len(span)
                    positions.append(((offsets[:,1]>start)&(offsets[:,0]<end)).nonzero().flatten().tolist())
                return inputs,positions
            def forward(self,input_ids,attention_mask,**kwargs):
                values=(input_ids.float()%101)[:,:,None]/101+torch.arange(4096)[None,None,:]/4096
                return SimpleNamespace(last_hidden_state=values)
        evidence,words=encode.encode_input(Knowledge(),torch.zeros(1,3,480,832),
            'A ball rises above the racket.',3.333,
            dict(caption='A ball rises above the racket.',scope='first_image_and_request_only',
                 relations=['Upward momentum permits ascent while gravity slows the ball.',
                            'Released powder can spread and disperse through the surrounding air.'],
                 producer='qwen3.8-flash',trace=dict(request_id='local-test',prompt='legal input only')),
            sample='test',revision='test-data')
        self.assertEqual(evidence['memory'].shape[-1],4096)
        self.assertEqual(evidence['request_word_strings'],['A','ball','rises','above','the','racket.'])
        self.assertEqual(words['words'].shape[0],19)
        self.assertLess(max(words['span_errors']),.01)

    def test_reader_rejects_old_caption_or_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'words').mkdir();(root/'native_words').mkdir();(root/'input').mkdir()
            shared=dict(caption='New caption.',relations=['A relation.'],sample='x',revision='new')
            evidence=dict(**shared,scope='first_image_and_request_only',memory=torch.zeros(1,4,4096),request_words=torch.zeros(2,4096))
            words=dict(**shared,words=torch.zeros(2,4096),word_groups=torch.zeros(2,dtype=torch.long),word_positions=torch.arange(2))
            native=dict(**shared,native_words=torch.zeros(2,3072))
            def write():
                torch.save(evidence,root/'input/x_text.pt');torch.save(words,root/'words/x_text.pt');torch.save(native,root/'native_words/x_text.pt')
            write()
            teacher=Teacher.__new__(Teacher);teacher.device='cpu';teacher.cache=root
            teacher.cfg=dict(variant='physical_content',data_revision='new',paths=dict(input_evidence=str(root/'input')))
            self.assertEqual(teacher.input(dict(id='x',caption='New caption.'))['words'].shape,(2,4096))
            with self.assertRaisesRegex(ValueError,'disagree'):teacher.input(dict(id='x',caption='Old caption.'))
            native['relations']=['Old relation.'];write()
            with self.assertRaisesRegex(ValueError,'same text'):teacher.input(dict(id='x',caption='New caption.'))

    def test_partial_relabel_preserves_full_sampling_order_and_old_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';out=root/'new'
            (old/'base/text').mkdir(parents=True)
            rows=[dict(id=name,caption=f'Original {name}.',split='train') for name in ('a','b','c')]
            (old/'base/manifest.json').write_text(json.dumps(dict(records=rows)))
            for name in ('a','b','c'):(old/'base/text'/f'{name}.pt').touch()
            (out/'records/b').mkdir(parents=True)
            (out/'selection.json').write_text(json.dumps(dict(ids=['b'],revision='new',maximum_training_updates=400)))
            (out/'records/b/caption.json').write_text(json.dumps(dict(old_caption='Original b.',caption='Corrected b.')))
            with patch.object(encode,'CACHE',old):selected,_=encode.manifest(out)
            revised=json.loads((out/'encoded/base/manifest.json').read_text())['records']
            self.assertEqual([r['id'] for r in revised],['a','b','c'])
            self.assertEqual([r['caption'] for r in revised],['Original a.','Corrected b.','Original c.'])
            self.assertEqual([r['id'] for r in selected],['b'])
            self.assertTrue((out/'encoded/base/text/a.pt').is_symlink())
            self.assertEqual(json.loads((old/'base/manifest.json').read_text())['records'],rows)

    def test_prepared400_ids_match_the_actual_sampler(self):
        from finish_preparation import check_schedule
        originals=json.loads((encode.CACHE/'base/manifest.json').read_text())['records']
        schedule=encode.ROOT/'r11_r12/data_audit/actual400_inputs.json'
        records=json.loads(schedule.read_text())['records']
        actual=check_schedule(originals,dict(ids=[r['id'] for r in records],schedule=str(schedule)))
        self.assertEqual(len(actual),1600)
        self.assertEqual(sum(r['drop'] for r in actual),154)


if __name__=='__main__':unittest.main()
