"""Cached frozen-Qwen evidence and training-only native semantic targets."""
from pathlib import Path
import json

import torch

from .common import FEATURE_CACHE, LEGACY_FEATURE_CACHE
from .evidence import find_input, recover_request_words


class Teacher:
    def __init__(self, cfg, device='cuda'):
        self.cfg = cfg
        self.device = device
        self.cache = FEATURE_CACHE
        self.feature_roots = (FEATURE_CACHE, LEGACY_FEATURE_CACHE)
        records = json.loads((Path(cfg['paths']['cache']) / 'base/manifest.json').read_text())['records']
        self.durations = {row['id']: float(row['pts'][80] - row['pts'][0]) for row in records}
        self._tokenizer = None

    def add_request(self, evidence, duration=None):
        if 'request_words' in evidence:
            return evidence
        if duration is None:
            sample = evidence['sample']
            duration = self.durations.get(sample)
            if duration is None and sample.startswith(('videophy40_', 'demo21_')):
                duration = 80 / 24
            if duration is None:
                raise ValueError('Known presentation interval is required for cached request spans')
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.cfg['paths']['qwen_checkpoint'], local_files_only=True)
        evidence.update(recover_request_words(evidence, duration, self._tokenizer))
        evidence['reader_scope'] = (
            'Contextual literal-request and physical-reasoning words; first-image '
            'context encoded by frozen Qwen, without future video')
        return evidence


    def _find(self, folder, name):
        for root in self.feature_roots:
            path = root / folder / f'{name}.pt'
            if path.is_file():
                return path
        return None

    def _load(self, folder, name):
        path = self._find(folder, name)
        if path is None:
            raise FileNotFoundError(f'No cached {folder} features for {name}')
        return torch.load(path, map_location='cpu', weights_only=True)

    def input(self, sample, drop=False):
        name = sample['id'] + ('_null' if drop else '_text')
        source = find_input(name)
        evidence = torch.load(source, map_location='cpu', weights_only=True)
        if evidence['scope'] != 'first_image_and_request_only':
            raise ValueError('C evidence cannot contain future video')
        words = self._load('words', name)
        evidence.update({key: words[key] for key in ('words', 'word_groups', 'word_positions')})
        native = self._find('native_words', name)
        if native is not None:
            evidence['native_words'] = torch.load(
                native, map_location='cpu', weights_only=True)['native_words']
        self.add_request(evidence)
        return {key: (value.to(self.device) if torch.is_tensor(value) else value)
                for key, value in evidence.items()}
