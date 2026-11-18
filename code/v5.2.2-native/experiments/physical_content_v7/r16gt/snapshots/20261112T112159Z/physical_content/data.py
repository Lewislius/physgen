"""T2V training views over the existing qwen38 T2V cache.

The loader filters the 3666 train rows to the 1600 rows with complete
request-only evidence, then uses ``SampleOrder(seed=42)``, matching the R18E/G
T2V ``TrainingData`` exactly.  The 1600-sample set is the R16G set; the loader
order is the R18 order.  ``data_ready`` verifies evidence captions and native
word order against the qwen38 manifest for every draw.
"""
from pathlib import Path
import json

import torch

from physgen_native.data import valid_grid


class Data:
    def __init__(self, cfg, split='train'):
        self.cfg = cfg
        self.root = Path(cfg['paths']['cache'])
        self.input_root = Path(cfg['paths']['t2v_input_cache'])
        self.words_root = Path(cfg['paths']['t2v_words_cache'])
        self.native_root = Path(cfg['paths']['t2v_native_words_cache'])
        rows = [
            row for row in json.loads(
                (self.root / 'base/manifest.json').read_text())['records']
            if row['split'] == split]
        if split == 'train' and cfg['data'].get('match_r18_t2v_loader'):
            # R18E/G filter to the 1600 rows with complete request-only
            # evidence, then SampleOrder over those 1600 rows.  R16Gt uses the
            # same filtered loader; the sample set is unchanged.
            rows = [row for row in rows if self.has_evidence(row)]
        if not rows:
            raise ValueError(f'No records in split {split}')
        self.rows = rows
        self.by_id = {row['id']: row for row in self.rows}

    def __len__(self):
        return len(self.rows)

    def has_evidence(self, row):
        for suffix in ('_text', '_null'):
            name = f'{row["id"]}{suffix}'
            if not (self.input_root / f'{name}.pt').is_file():
                return False
            if not (self.words_root / f'{name}.pt').is_file():
                return False
            if not (self.native_root / f'{name}.pt').is_file():
                return False
        return True

    def get(self, index, device='cuda', targets=True):
        row = self.rows[index]
        if not self.has_evidence(row):
            raise ValueError(
                f'{row["id"]} is outside the audited T2V evidence set; '
                'the R16G-matched sample order must provide evidence for every draw')
        state = torch.load(self.root / f'base/vae/{row["id"]}.pt',
                           map_location='cpu', weights_only=True)
        latent = state['latent'].float().to(device)
        expected = tuple(self.cfg['data']['latent_shape'])
        if tuple(latent.shape) != expected:
            raise ValueError(f'Unexpected T2V latent shape for {row["id"]}: {latent.shape}')
        text = torch.load(self.root / f'base/text/{row["id"]}.pt',
                          map_location='cpu', weights_only=True).float()[None].to(device)
        times = torch.tensor(
            [row['pts'][i] - row['pts'][0] for i in range(0, 81, 4)],
            device=device).float()[None]
        valid = valid_grid(row, 30, 52).to(device)
        result = dict(id=row['id'], row=row, caption=row['caption'], text=text,
                      times=times, valid=valid)
        if targets:
            result['latent'] = latent
        return result
