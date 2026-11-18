"""Training views for the two task entries, with no cross-task evidence reuse."""
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
                raise ValueError('The fixed400 sequence must contain1600 training samples')
            rows = [by_id[sample] for sample in sequence]
        words = Path(cfg['paths']['words_cache'])
        native = Path(cfg['paths']['native_words_cache'])
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
        row = self.rows[index]
        state = torch.load(self.root / f'base/vae/{row["id"]}.pt',
                           map_location='cpu', weights_only=True)
        text = torch.load(self.root / f'base/text/{row["id"]}.pt',
                          map_location='cpu', weights_only=True).float()[None].to(device)
        times = torch.tensor(
            [row['pts'][i] - row['pts'][0] for i in range(0, 81, 4)],
            device=device).float()[None]
        valid = valid_grid(row, 30, 52).to(device)
        result = dict(id=row['id'], row=row, caption=row['caption'],
                      text=text, times=times, valid=valid)
        if self.task == 'i2v':
            first = state['first'].float().to(device)
            result['first'] = first
            if targets:
                latent = state['latent'].float().to(device)
                result['latent'] = torch.cat((first, latent[:, :, 1:]), 2)
        else:
            # T2V never receives the first image, its RGB, or any condition
            # built from the first image.
            if targets:
                result['latent'] = state['latent'].float().to(device)
        return result
