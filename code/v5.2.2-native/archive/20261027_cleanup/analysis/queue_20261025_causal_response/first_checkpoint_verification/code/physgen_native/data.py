"""Read existing caches; cached video features select training evidence only."""
from pathlib import Path
from collections import Counter
import json
import torch
from torch.utils.data import Dataset


def load_tensor(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def manifest(cfg):
    return json.loads((Path(cfg['paths']['cache'])/'base/manifest.json').read_text())['records']


def audit(cfg):
    root = Path(cfg['paths']['cache'])
    rows = manifest(cfg)
    missing = []
    for row in rows:
        for sub, ext in [('base/vae', '.pt'), ('base/text', '.pt'), ('v52/requirements', '.json'), ('v52/change', '.pt'), ('v52/video', '.pt')]:
            path = root/sub/(row['id']+ext)
            if not path.is_file(): missing.append(str(path))
    for sub in ('base/null_text.pt', 'v52/train_scale.json'):
        if not (root/sub).is_file(): missing.append(str(root/sub))
    models = [(Path(cfg['paths']['wan_checkpoint']), 'diffusion_pytorch_model.safetensors.index.json'),
              (Path(cfg['paths']['qwen_checkpoint']), 'model.safetensors.index.json')]
    for directory, name in models:
        path = directory/name
        if not path.is_file():
            missing.append(str(path)); continue
        index = json.loads(path.read_text())
        for shard in set(index['weight_map'].values()):
            if not (directory/shard).is_file(): missing.append(str(directory/shard))
    for name in ('Wan2.2_VAE.pth', 'models_t5_umt5-xxl-enc-bf16.pth'):
        if not (models[0][0]/name).is_file(): missing.append(str(models[0][0]/name))
    if cfg['verifier'].get('kind') in ('ordered_process','calibrated_process','conditional_process'):
        if not Path(cfg['paths']['video_evidence_checkpoint']).is_file():missing.append(cfg['paths']['video_evidence_checkpoint'])
    if len(rows) != cfg['data']['count']:
        raise ValueError(f'Manifest changed: {len(rows)} != {cfg["data"]["count"]}')
    scale = json.loads((root/'v52/train_scale.json').read_text())
    if scale['source_split'] != 'train' or not 0 < scale['scale'] <= 1:
        raise ValueError('Use the existing train-only visible-change scale')
    return dict(ready=not missing, records=len(rows), splits=dict(Counter(r['split'] for r in rows)),
                missing=missing, intensity_scale=scale['scale'], preprocessing_required=False)


def valid_grid(row, height, width):
    g = row['geometry']
    y = torch.arange(height).float()/height
    x = torch.arange(width).float()/width
    y = ((y+1/height).clamp(max=(g['top']+g['rh'])/g['h'])-y.clamp(min=g['top']/g['h'])).clamp_min(0)*height
    x = ((x+1/width).clamp(max=(g['left']+g['rw'])/g['w'])-x.clamp(min=g['left']/g['w'])).clamp_min(0)*width
    return y[:, None]*x[None]


class VideoDataset(Dataset):
    def __init__(self, cfg, split='train', *, targets=True):
        self.cfg = cfg
        self.targets = targets
        self.root = Path(cfg['paths']['cache'])
        self.rows = [r for r in manifest(cfg) if r['split'] == split]
        # Explicit existing-case selection is for controlled learnability studies.
        # The production recipe leaves this unset and uses the entire split.
        selected = cfg['data'].get(f'{split}_ids')
        if selected is not None:
            available = {r['id']: r for r in self.rows}
            if len(set(selected)) != len(selected) or not set(selected) <= set(available):
                raise ValueError('Diagnostic case selection must be unique and remain within its original split')
            self.rows = [available[name] for name in selected]
        self.scale = json.loads((self.root/'v52/train_scale.json').read_text())['scale']
        if not self.rows: raise ValueError(f'Empty split {split}')

    def __len__(self): return len(self.rows)

    def alternative_condition(self, sample, cursor):
        """Training-only text replacement; keep the caller's own first image.

        Independent deterministic sampling does not disturb the paired FM noise.
        Only existing legal caption/first-image requirements are read, never
        another example's future targets. Matching is not a physical-error label.
        """
        candidates = [i for i, row in enumerate(self.rows)
                      if row['id'] != sample['id'] and row['caption'].strip() != sample['caption'].strip()]
        if not candidates: raise ValueError('Condition-use training needs two distinct training captions')
        rng = torch.Generator().manual_seed(self.cfg['train']['seed']*1000003+cursor)
        row = self.rows[candidates[int(torch.randint(len(candidates), (), generator=rng))]]
        req = json.loads((self.root/'v52/requirements'/(row['id']+'.json')).read_text())
        if req.get('input_scope') != 'caption_and_first_image': raise ValueError('Illegal alternative requirement scope')
        # Image-only clauses describe the other first image, which is not being
        # swapped. Keep only prompt requirements as the text intervention.
        clauses = [c for c in req.get('clauses', []) if c.get('basis', 'prompt') == 'prompt']
        return dict(id=row['id'], caption=row['caption'], clauses=clauses)

    def __getitem__(self, index):
        row = self.rows[index]; name = row['id']+'.pt'
        state = load_tensor(self.root/'base/vae'/name)
        z, first = state['latent'].float(), state['first'].float()
        z = torch.cat((first, z[:, :, 1:]), 2)
        if tuple(z.shape[1:]) != tuple(self.cfg['data']['latent_shape']):
            raise ValueError(f'Unexpected cache geometry for {row["id"]}: {z.shape}')
        req = json.loads((self.root/'v52/requirements'/(row['id']+'.json')).read_text())
        if req.get('input_scope') != 'caption_and_first_image':
            raise ValueError('Requirements must originate only from caption and first image')
        text = load_tensor(self.root/'base/text'/name).float()[None]
        times = torch.tensor([row['pts'][i]-row['pts'][0] for i in range(0, row['geometry']['frames'], 4)]).float()
        result = dict(id=row['id'], first=first, text=text, caption=row['caption'],
                      clauses=req.get('clauses',[]),times=times,valid=valid_grid(row,*z.shape[-2:]))
        if not self.targets: return result
        change = load_tensor(self.root/'v52/change'/name)
        video = load_tensor(self.root/'v52/video'/name)
        # Only a training-view proposal, not a physical speed or inference input.
        j = video['jepa'].float()
        jvalid = video['valid'].float()
        change_energy = ((j[2::2]-j[:-2:2]).square().mean(-1)*jvalid[None]).sum(-1)/jvalid.sum().clamp_min(1)
        return dict(result,latent=z,evidence_activity=change_energy.sqrt(),
                    intensity=(change['intensity'].float()/self.scale).clamp(0, 1)[None],
                    change_valid=change['valid'].float()[None])


def move(value, device):
    if torch.is_tensor(value): return value.to(device, non_blocking=True)
    if isinstance(value, dict): return {k: move(v, device) for k, v in value.items()}
    if isinstance(value, list): return [move(v, device) for v in value]
    return value


class SampleOrder:
    def __init__(self, size, seed, cursor=0): self.size, self.seed, self.cursor = size, seed, cursor
    def __iter__(self):
        index, epoch, order = self.cursor, -1, None
        while True:
            current = index//self.size
            if current != epoch:
                order = torch.randperm(self.size, generator=torch.Generator().manual_seed(self.seed+current)).tolist()
                epoch = current
            yield order[index % self.size]
            index += 1
