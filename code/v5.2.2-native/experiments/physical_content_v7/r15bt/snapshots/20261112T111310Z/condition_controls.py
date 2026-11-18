"""Read compact request-only inference evidence; never an image or first frame."""
from pathlib import Path
import json

import torch

ENCODING = 'r15bt_request_only_v1'


class InferenceCache:
    def __init__(self, root, device='cuda'):
        self.root = Path(root)
        self.device = device

    def input(self, sample, drop=False):
        folder = self.root / sample['id']
        ready = json.loads((folder / 'ready.json').read_text())
        if (ready['id'] != sample['id'] or ready['case'] != sample['case']
                or ready['caption'] != sample['caption']
                or ready['scope'] != 'request_only'
                or ready['revision'] != ENCODING):
            raise ValueError('Prepared evidence differs from the fixed T2V protocol')
        path = folder / ('negative.pt' if drop else 'positive.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        if (value['sample'] != sample['id']
                or value['caption'] != ('' if drop else sample['caption'])
                or value['scope'] != 'request_only'
                or value['request_word_strings']
                != ([] if drop else sample['caption'].split())):
            raise ValueError('Prepared condition belongs to another input')
        return {key: (item.to(self.device) if torch.is_tensor(item) else item)
                for key, item in value.items()}
