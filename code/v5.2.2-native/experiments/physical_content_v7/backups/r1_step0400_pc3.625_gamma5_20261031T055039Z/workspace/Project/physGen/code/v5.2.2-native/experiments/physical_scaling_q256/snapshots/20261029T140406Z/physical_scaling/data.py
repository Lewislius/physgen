from pathlib import Path
import importlib.util
import json
import numpy as np
from PIL import Image
import torch
from physgen_native.data import valid_grid, SampleOrder


class Data:
    def __init__(self, cfg, split='train'):
        self.root = Path(cfg['paths']['cache'])
        self.rows = [r for r in json.loads((self.root/'base/manifest.json').read_text())['records']
                     if r['split'] == split]
        self.by_id = {r['id']: r for r in self.rows}
        if not self.rows:
            raise ValueError(f'No existing cache records in split {split}')

    def __len__(self):
        return len(self.rows)

    def get(self, index, device='cuda', targets=True):
        row = self.rows[index]
        state = torch.load(self.root/f'base/vae/{row["id"]}.pt', weights_only=True, map_location='cpu')
        first = state['first'].float().to(device)
        text = torch.load(self.root/f'base/text/{row["id"]}.pt', weights_only=True, map_location='cpu').float()[None].to(device)
        rgb = np.array(Image.open(self.root/f'base/first/{row["id"]}.png').convert('RGB'), copy=True)
        rgb = torch.from_numpy(rgb).permute(2,0,1)[None].to(device).float()/127.5-1
        times = torch.tensor([row['pts'][i]-row['pts'][0] for i in range(0,81,4)], device=device).float()[None]
        result = dict(id=row['id'], row=row, caption=row['caption'], first=first, text=text,
                      first_rgb=rgb, times=times, valid=valid_grid(row,30,52).to(device))
        if targets:
            latent = state['latent'].float().to(device)
            result['latent'] = torch.cat((first, latent[:,:,1:]),2)
        return result


def read_teacher_video(row, device):
    # Identical decode/resize/crop to the immutable VAE cache, not VAE reconstructions.
    path = Path('/home/liuzhirui/Project/physGen/code/v5.2/physgen_v52/preparation.py')
    spec = importlib.util.spec_from_file_location('_physical_existing_preparation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    indices = sorted(set(range(0,81,3)) | {80})
    frames = module.read_frames(row, indices)
    rgb = torch.from_numpy(np.stack([np.asarray(im) for im in frames])).permute(0,3,1,2).to(device).float()/127.5-1
    times = torch.tensor([row['pts'][i]-row['pts'][0] for i in indices], device=device).float()
    return rgb, times
