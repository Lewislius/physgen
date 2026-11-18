"""Request-only frozen-Qwen evidence for the T2V R16Gt model.

No image, future frame, score, or physical label is read.  The source cache is
the audited request-only T2V cache shared with the v10 T2V branch; every record
is checked for scope, sample identity, caption, and literal request words.
"""
from pathlib import Path
import torch

from .common import SOURCE_EVIDENCE_REVISION

SOURCE_REVISION = SOURCE_EVIDENCE_REVISION


class Teacher:
    def __init__(self, cfg, device='cuda', log=None):
        self.cfg = cfg
        self.device = device
        self.log = log
        self.input_root = Path(cfg['paths']['t2v_input_cache'])
        self.native_root = Path(cfg['paths']['t2v_native_words_cache'])
        if not self.input_root.is_dir() or not self.native_root.is_dir():
            raise FileNotFoundError('Request-only T2V evidence cache is missing')

    def input(self, sample, drop=False):
        name = sample['id'] + ('_null' if drop else '_text')
        source = self.input_root / f'{name}.pt'
        evidence = torch.load(source, map_location='cpu', weights_only=True)
        if evidence.get('scope') != 'request_only':
            raise ValueError('R16Gt evidence must be request-only')
        if evidence.get('revision') != SOURCE_REVISION:
            raise ValueError('Unexpected request-only evidence revision')
        if evidence.get('sample') != sample['id']:
            raise ValueError('Evidence belongs to another sample')
        caption = '' if drop else sample['caption']
        if evidence.get('caption') != caption:
            raise ValueError('Evidence caption differs from the training request')
        expected_words = [] if drop else caption.split()
        if evidence.get('request_word_strings') != expected_words:
            raise ValueError('The literal request span is incomplete')
        for key in ('words', 'word_groups', 'word_positions', 'request_words'):
            if key not in evidence:
                raise KeyError(f'Request-only evidence lacks {key}')
        if evidence['words'].shape[-1] != 4096 or evidence['request_words'].shape[-1] != 4096:
            raise ValueError('Retain the frozen local Qwen4096 feature width')
        if len(evidence['words']) != len(evidence['word_groups']):
            raise ValueError('Physical word and group counts differ')
        native_path = self.native_root / f'{name}.pt'
        if not native_path.is_file():
            raise FileNotFoundError(f'No native T5 target for {name}')
        native = torch.load(native_path, map_location='cpu', weights_only=True)
        if native.get('scope') != 'request_only':
            raise ValueError('Native target must be request-only')
        if native.get('sample') != sample['id']:
            raise ValueError('Native target belongs to another sample')
        if list(native.get('word_strings', [])) != list(evidence['word_strings']):
            raise ValueError('Native T5 target word order differs from Qwen evidence')
        evidence['native_words'] = native['native_words']
        return {key: (value.to(self.device) if torch.is_tensor(value) else value)
                for key, value in evidence.items()}
