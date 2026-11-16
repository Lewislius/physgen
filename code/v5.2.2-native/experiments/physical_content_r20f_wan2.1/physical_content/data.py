"""Training views for the Wan2.1-T2V-1.3B cache.

The Wan2.1 VAE compresses space by 8 and time by 4, so an 81-frame 832x480 clip
is a [1, 16, 21, 60, 104] latent.  The Wan2.2 cache [1, 48, 21, 30, 52] is
rejected explicitly rather than being allowed to fail deep inside patch
embedding.
"""
from pathlib import Path
import json

import torch

from physgen_native.data import valid_grid


def find_input(cfg, name):
    for root in [cfg['paths']['input_cache'], *cfg['paths'].get('input_cache_fallbacks', [])]:
        path = Path(root) / f'{name}.pt'
        if path.is_file():
            return path
    raise FileNotFoundError(f'No {cfg["task"]} evidence for {name}')


class TrainingData:
    def __init__(self, cfg):
        if cfg['task'] != 't2v':
            raise ValueError('R20FT-Wan2.1 is text-to-video only')
        self.cfg = cfg
        self.task = cfg['task']
        self.root = Path(cfg['paths']['cache'])
        rows = json.loads((self.root / 'base/manifest.json').read_text())['records']
        rows = [row for row in rows if row['split'] == 'train']
        self.fixed_sequence = bool(cfg['train'].get('sample_sequence'))
        selected_drops = None
        if self.fixed_sequence:
            specification = json.loads(Path(cfg['train']['sample_sequence']).read_text())
            sequence, selected_drops = specification['ids'], specification['drop']
            by_id = {row['id']: row for row in rows}
            if len(sequence) != cfg['train']['max_updates'] * cfg['train']['gradient_accumulation']:
                raise ValueError('The fixed400 sequence must contain 1600 training samples')
            rows = [by_id[sample] for sample in sequence]
        words = Path(cfg['paths']['words_cache'])
        native = Path(cfg['paths']['native_words_cache'])
        self.latent_shape = tuple(int(value) for value in cfg['data']['latent_shape'])
        if self.latent_shape != (16, 21, 60, 104):
            raise ValueError(f'Unexpected Wan2.1 latent shape: {self.latent_shape}')
        kept = []
        for index, row in enumerate(rows):
            complete = True
            arms = ([selected_drops[index]] if selected_drops is not None else [False, True])
            for drop in arms:
                name = f'{row["id"]}_{"null" if drop else "text"}'
                if not (words / f'{name}.pt').is_file():
                    complete = False
                    break
                if not (native / f'{name}.pt').is_file():
                    complete = False
                    break
                try:
                    find_input(cfg, name)
                except FileNotFoundError:
                    complete = False
                    break
            if complete:
                kept.append(row)
            elif self.fixed_sequence:
                raise ValueError(f'Fixed training sample lacks evidence: {row["id"]}')
        if not kept:
            raise ValueError(f'No train row has complete {self.task} evidence')
        self.rows = kept
        self.by_id = {row['id']: row for row in kept}

    def __len__(self):
        return len(self.rows)

    def get(self, index, device='cuda', targets=True):
        if self.task != 't2v':
            raise ValueError('R20FT-Wan2.1 never builds an image-conditioned view')
        row = self.rows[index]
        state = torch.load(self.root / f'base/vae/{row["id"]}.pt',
                           map_location='cpu', weights_only=True)
        if 'latent' not in state:
            raise KeyError(f'{row["id"]}: Wan2.1 VAE cache has no latent')
        latent = state['latent'].float()
        expected = (1,) + self.latent_shape
        if tuple(latent.shape) != expected:
            raise ValueError(
                f'{row["id"]}: expected Wan2.1 latent {expected}, got {tuple(latent.shape)}')
        text = torch.load(self.root / f'base/text/{row["id"]}.pt',
                          map_location='cpu', weights_only=True).float()[None].to(device)
        if text.shape[1] > 512:
            raise ValueError(f'{row["id"]}: native T5 text exceeds 512 tokens')
        times = torch.tensor(
            [row['pts'][i] - row['pts'][0] for i in range(0, 81, 4)],
            device=device).float()[None]
        if times.shape[1] != self.latent_shape[1]:
            raise ValueError(f'{row["id"]}: time grid does not match 21 latent frames')
        valid = valid_grid(row, self.latent_shape[-2], self.latent_shape[-1]).to(device)
        result = dict(id=row['id'], row=row, caption=row['caption'],
                      text=text, times=times, valid=valid, latent=latent.to(device))
        return result
