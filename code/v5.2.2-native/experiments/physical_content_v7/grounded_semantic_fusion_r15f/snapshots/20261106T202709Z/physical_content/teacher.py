"""Cached frozen-Qwen input evidence plus training-only native semantic targets."""
from pathlib import Path
import json
import torch
from .evidence import find_input,recover_request_words

class Teacher:
    def __init__(self,cfg,device='cuda',log=None):
        self.cfg=cfg;self.device=device;self.log=log
        self.cache=Path(cfg['paths']['content_cache'])
        self._tokenizer=None
        records=json.loads((Path(cfg['paths']['cache'])/'base/manifest.json').read_text())['records']
        self.durations={r['id']:float(r['pts'][80]-r['pts'][0]) for r in records}

    def add_request(self,evidence,duration=None):
        if 'request_words' in evidence:return evidence
        if duration is None:
            sample=evidence['sample']
            duration=self.durations.get(sample)
            if duration is None and sample.startswith(('videophy40_','demo21_')):duration=80/24
            if duration is None:raise ValueError('Known presentation interval is required for cached request spans')
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer=AutoTokenizer.from_pretrained(self.cfg['paths']['qwen_checkpoint'],local_files_only=True)
        evidence.update(recover_request_words(evidence,duration,self._tokenizer))
        evidence['reader_scope']='Relation-grounded words plus complete original first-image/request/reasoning Qwen memory; no future video'
        return evidence

    def input(self,sample,drop=False):
        if self.cfg['variant']=='native_lora':
            return dict(memory=sample['first'].new_zeros((1,1,1)))
        name=sample['id']+('_null' if drop else '_text')
        source=find_input(name)
        evidence=torch.load(source,map_location='cpu',weights_only=True)
        if evidence['scope']!='first_image_and_request_only':raise ValueError('C evidence cannot contain future video')
        words=torch.load(self.cache/'words'/f'{name}.pt',map_location='cpu',weights_only=True)
        evidence.update({key:words[key] for key in ['words','word_groups','word_positions']})
        native=self.cache/'native_words'/f'{name}.pt'
        if native.is_file():evidence['native_words']=torch.load(native,map_location='cpu',weights_only=True)['native_words']
        self.add_request(evidence)
        return {key:(value.to(self.device) if torch.is_tensor(value) else value) for key,value in evidence.items()}
