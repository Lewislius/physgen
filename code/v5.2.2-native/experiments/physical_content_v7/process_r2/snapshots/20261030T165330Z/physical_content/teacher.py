"""Cached frozen-Qwen input evidence plus training-only native semantic targets."""
from pathlib import Path
import torch
from .evidence import find_input

class Teacher:
    def __init__(self,cfg,device='cuda',log=None):
        self.cfg=cfg;self.device=device;self.log=log
        self.cache=Path(cfg['paths']['content_cache'])

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
        return {key:(value.to(self.device) if torch.is_tensor(value) else value) for key,value in evidence.items()}
