"""Frozen input evidence for the two task entries.

I2V uses the existing ``first_image_and_request_only`` cache.  T2V requires the
separate ``request_only`` cache produced by ``prepare_t2v_evidence.py``; the
image-conditioned cache is rejected by scope.
"""
from pathlib import Path
import json

import torch

from .data import find_input
from .evidence import recover_request_words


class Teacher:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.words_cache = Path(cfg['paths']['words_cache'])
        self.native_cache = Path(cfg['paths']['native_words_cache'])
        self.scope = cfg['input']['evidence_scope']
        records = json.loads(
            (Path(cfg['paths']['cache']) / 'base/manifest.json').read_text())['records']
        self.durations = {
            row['id']: float(row['pts'][80] - row['pts'][0]) for row in records}
        self._tokenizer = None

    def _load_tensor(self, root, name):
        path = Path(root) / f'{name}.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        return torch.load(path, map_location='cpu', weights_only=True)

    def add_request(self, evidence):
        if 'request_words' in evidence:
            return evidence
        if 'memory' not in evidence:
            raise ValueError('Request spans must be prepared for text-only evidence')
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg['paths']['qwen_checkpoint'], local_files_only=True)
        duration = self.durations.get(evidence['sample'])
        if duration is None:
            raise ValueError('Known presentation interval is required for request spans')
        evidence.update(recover_request_words(evidence, duration, self._tokenizer))
        return evidence

    def input(self, sample, drop=False):
        name = sample['id'] + ('_null' if drop else '_text')
        evidence = torch.load(find_input(self.cfg, name), map_location='cpu',
                              weights_only=True)
        if evidence.get('scope') != self.scope:
            raise ValueError(
                f'Evidence scope {evidence.get("scope")} is not {self.scope}')
        if evidence.get('sample') != sample['id']:
            raise ValueError('Evidence belongs to a different sample')
        if evidence.get('caption') != ('' if drop else sample['caption']):
            raise ValueError('Evidence caption differs from the training sample')
        words = self._load_tensor(self.words_cache, name)
        evidence.update({key: words[key] for key in
                         ('words', 'word_groups', 'word_positions')})
        native = self._load_tensor(self.native_cache, name)
        evidence['native_words'] = native['native_words']
        self.add_request(evidence)
        if self.cfg['task'] == 't2v':
            forbidden = {'memory', 'relation_vectors', 'image', 'first_image',
                         'first_rgb', 'input_canvas'}
            if forbidden & set(evidence):
                raise ValueError('T2V evidence contains first-image/future tensors')
        return {key: (value.to(self.device) if torch.is_tensor(value) else value)
                for key, value in evidence.items()}
