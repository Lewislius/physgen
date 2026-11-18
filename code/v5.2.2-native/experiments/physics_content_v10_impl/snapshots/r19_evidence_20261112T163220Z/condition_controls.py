"""Read compact inference evidence for the two task entries."""
from pathlib import Path
import json

import torch

ENCODING = 'NoChunks-v2-20261111'


class InferenceCache:
    def __init__(self, root, task, device='cuda'):
        self.root = Path(root)
        self.task = task
        self.device = device

    def input(self, sample, *, image=None, duration=None, drop=False):
        folder = self.root / sample['id']
        ready = json.loads((folder / 'ready.json').read_text())
        if (ready['id'] != sample['id'] or ready['caption'] != sample['caption']
                or ready['scope'] != (('request_only' if self.task == 't2v'
                                       else 'first_image_and_request_only'))
                or ready['revision'] != ENCODING):
            raise ValueError('Prepared evidence differs from the fixed protocol')
        if self.task == 'i2v':
            if image is None or ready['image'] != str(Path(image).resolve()):
                raise ValueError('I2V evidence was built for a different first image')
            if duration is None or abs(ready['duration'] - duration) > 1e-6:
                raise ValueError('I2V evidence duration differs')
        else:
            if image is not None:
                raise ValueError('T2V must not receive a first image')
        path = folder / ('negative.pt' if drop else 'positive.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        if (value['sample'] != sample['id'] or value['caption'] !=
                ('' if drop else sample['caption'])
                or value['scope'] != ready['scope']
                or value['request_word_strings'] !=
                ([] if drop else sample['caption'].split())):
            raise ValueError('Prepared condition belongs to another input')
        return {key: (item.to(self.device) if torch.is_tensor(item) else item)
                for key, item in value.items()}
