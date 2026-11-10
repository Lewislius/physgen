"""Check real local token spans and control boundaries; no generation claims."""
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import encode
import torch
from condition_controls import encode_control, InferenceCache, MeanContent, mean_with_own_prior
from physical_content.evidence import resample_words
from physical_scaling.supervision import group_log_prior


class LocalTemplate:
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer;self.cfg={}
        self.backbone=SimpleNamespace(model=self.forward)

    def _inputs(self,clips,body,extract_spans):
        text='Initial observation:\n<|vision_start|>'+'<|image_pad|>'*390+'<|vision_end|>\n'+body
        text=self.tokenizer.apply_chat_template([dict(role='user',content=text)],tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer(text,add_special_tokens=False,return_offsets_mapping=True,return_tensors='pt')
        offsets=inputs.pop('offset_mapping')[0];positions=[]
        for span in extract_spans:
            start=text.index(span);end=start+len(span)
            positions.append(((offsets[:,1]>start)&(offsets[:,0]<end)).nonzero().flatten().tolist())
        return inputs,positions

    def forward(self,input_ids,attention_mask,**kwargs):
        value=(input_ids.float()%101)[:,:,None]/101+torch.arange(4096)[None,None,:]/4096
        return SimpleNamespace(last_hidden_state=value)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained('/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct',local_files_only=True)
        cls.knowledge=LocalTemplate(tokenizer)
        cls.image=torch.zeros(1,3,480,832)

    def reasoning(self,count=3):
        return dict(sample='own',caption='A ball rises.',scope='first_image_and_request_only',
            relations=['Gravity can slow upward motion.']*count,producer='qwen3.8-flash',
            trace=dict(request_id='offline-template-test',prompt='first image and request'))

    def test_empty_preserves_real_token_spans_and_group_prior(self):
        for count in range(1,5):
            evidence=encode_control(self.knowledge,self.image,'A ball rises.',3.,self.reasoning(count),
                sample='own',mode='empty_reasoning')
            self.assertEqual(evidence['relations'],['']*count)
            self.assertEqual(evidence['analysis'],'')
            self.assertEqual(evidence['request_word_strings'],['A','ball','rises.'])
            words,groups,_=resample_words(evidence['words'],evidence['word_groups'])
            self.assertEqual(tuple(words.shape),(1,256,4096))
            self.assertEqual(int(groups.max())+1,count)
            self.assertAlmostEqual(float(group_log_prior(groups).softmax(-1)[-1]),1/(count+1),places=6)

    def test_swapped_reencodes_own_request_and_rejects_prior_change(self):
        own=self.reasoning();donor=dict(self.reasoning(),sample='other',caption='Unrelated request.',
            relations=['Water can spread over a surface.']*3)
        value=encode_control(self.knowledge,self.image,'A ball rises.',3.,own,
            sample='own',mode='reasoning_swapped',donor=donor)
        self.assertEqual(value['caption'],own['caption'])
        self.assertEqual(value['request_word_strings'],['A','ball','rises.'])
        self.assertEqual(value['relations'],donor['relations'])
        self.assertEqual(value['donor'],'other')
        donor['relations']=donor['relations'][:2]
        with self.assertRaisesRegex(ValueError,'same relation count'):
            encode_control(self.knowledge,self.image,'A ball rises.',3.,own,
                sample='own',mode='reasoning_swapped',donor=donor)

    def test_mean_handles_different_counts_without_changing_test_prior(self):
        one=torch.zeros(256,dtype=torch.long);four=torch.arange(256)*4//256
        accumulator=MeanContent()
        for level,groups in [(2.,one),(4.,four)]:
            accumulator.add(dict(routing=torch.full((1,256,8),level),execution=torch.full((1,256,8),level*2)),groups)
        means,report=accumulator.finish()
        c,g,nc,ng=mean_with_own_prior(dict(positive=means,negative=means),one,four,'cpu')
        self.assertEqual(report['relation_counts'],{1:1,4:1})
        self.assertTrue(torch.equal(c['routing'],torch.full((1,256,8),3.)))
        self.assertAlmostEqual(float(group_log_prior(g).softmax(-1)[-1]),.5)
        self.assertAlmostEqual(float(group_log_prior(ng).softmax(-1)[-1]),.2,places=6)

    def test_explicit_cache_refuses_different_prompt_or_missing_new_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'own';folder.mkdir()
            ready=dict(id='own',caption='A ball rises.',image=str(root/'image.png'),duration=3.,
                scope='first_image_and_request_only',width=4096,native_wan_prompt_unchanged=True)
            (folder/'ready.json').write_text(json.dumps(ready))
            evidence=encode_control(self.knowledge,self.image,'A ball rises.',3.,self.reasoning(),
                sample='own',mode='empty_reasoning')
            evidence.update(first_image=str(root/'image.png'),duration=3.)
            (folder/'controls').mkdir();torch.save(evidence,folder/'controls/empty_reasoning.pt')
            reader=InferenceCache(root,'cpu');sample=dict(id='own',caption='A ball rises.')
            loaded=reader.input(sample,image=root/'image.png',duration=3.,mode='empty_reasoning')
            self.assertEqual(loaded['relations'],['']*3)
            with self.assertRaisesRegex(ValueError,'benchmark'):
                reader.input(dict(id='own',caption='Changed prompt.'),image=root/'image.png',duration=3.)
            with self.assertRaises(FileNotFoundError):
                reader.input(sample,image=root/'image.png',duration=3.,mode='real')
            evidence['first_image']=str(root/'old_image.png')
            torch.save(evidence,folder/'controls/empty_reasoning.pt')
            with self.assertRaisesRegex(ValueError,'different input'):
                reader.input(sample,image=root/'image.png',duration=3.,mode='empty_reasoning')


if __name__=='__main__':unittest.main()
