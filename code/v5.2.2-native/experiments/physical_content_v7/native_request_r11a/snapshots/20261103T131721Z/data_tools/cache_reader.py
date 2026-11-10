"""Explicit new-cache reader for the next data revision; no legacy cache fallback."""
from pathlib import Path
import json
import torch
from physical_content.evidence import recover_request_words


class Teacher:
    def __init__(self,cfg,device='cuda',log=None):
        self.cfg=cfg;self.device=device;self.log=log
        self.cache=Path(cfg['paths']['content_cache']);self._tokenizer=None
        root=Path(cfg['paths']['cache'])
        ready=json.loads((root/'readiness.json').read_text())
        if not ready['training_ready'] or ready['revision']!=cfg['data_revision']:
            raise ValueError('Data preparation has not completed for this revision')
        if (sum(cfg['stages'].values())>ready['maximum_training_updates']
                or cfg['train']['seed']!=42 or cfg['train']['accumulation']!=4):
            raise ValueError('This cache covers the matched seed42 accumulation4 400-update schedule only')
        records=json.loads((root/'base/manifest.json').read_text())['records']
        self.durations={r['id']:float(r['pts'][80]-r['pts'][0]) for r in records}

    def add_request(self,evidence,duration=None):
        if 'request_words' in evidence:return evidence
        if duration is None:duration=self.durations[evidence['sample']]
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer=AutoTokenizer.from_pretrained(self.cfg['paths']['qwen_checkpoint'],local_files_only=True)
        evidence.update(recover_request_words(evidence,duration,self._tokenizer))
        return evidence

    def input(self, sample, drop=False):
        if self.cfg['variant']=='native_lora':
            return dict(memory=sample['first'].new_zeros((1,1,1)))
        name=sample['id']+('_null' if drop else '_text')
        root=Path(self.cfg['paths']['input_evidence'])
        evidence=torch.load(root/f'{name}.pt',map_location='cpu',weights_only=True)
        expected='' if drop else sample.get('caption',evidence['caption'])
        if evidence['caption']!=expected or evidence['scope']!='first_image_and_request_only':
            raise ValueError('New caption and locally encoded Qwen input disagree')
        if evidence['sample']!=sample['id'] or evidence['revision']!=self.cfg['data_revision']:
            raise ValueError('Wrong sample or data revision')
        words=torch.load(self.cache/'words'/f'{name}.pt',map_location='cpu',weights_only=True)
        native=torch.load(self.cache/'native_words'/f'{name}.pt',map_location='cpu',weights_only=True)
        for value in (words,native):
            if (value['caption']!=expected or value['relations']!=evidence['relations']
                    or value['revision']!=evidence['revision'] or value['sample']!=sample['id']):
                raise ValueError('New input, physical words and native targets are not from the same text')
        if evidence['memory'].shape[-1]!=4096 or native['native_words'].shape[-1]!=3072:
            raise ValueError('Existing Qwen and Wan widths must be preserved')
        evidence.update({key:words[key] for key in ('words','word_groups','word_positions')})
        evidence['native_words']=native['native_words']
        self.add_request(evidence)
        return {key:(value.to(self.device) if torch.is_tensor(value) else value) for key,value in evidence.items()}
