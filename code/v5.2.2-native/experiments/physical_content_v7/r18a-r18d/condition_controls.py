"""Read the compact immutable positive/negative condition arms."""
from pathlib import Path
import json

import torch

ENCODING = 'original_qwen_first_image_request_v1'


class InferenceCache:
    def __init__(self, root, device='cuda'):
        self.root = Path(root)
        self.device = device

    def input(self, sample, *, image, duration, drop=False):
        folder = self.root / sample['id']
        ready = json.loads((folder / 'ready.json').read_text())
        expected_image = str(Path(image).resolve())
        if (ready['id'] != sample['id'] or ready['caption'] != sample['caption']
                or ready['image'] != expected_image or abs(ready['duration'] - duration) > 1e-6
                or ready['scope'] != 'first_image_and_request_only' or ready['width'] != 4096
                or ready['revision'] != ENCODING or not ready['native_wan_prompt_unchanged']):
            raise ValueError('Prepared inputs differ from the original benchmark')
        path = folder / ('negative.pt' if drop else 'positive.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        caption = '' if drop else sample['caption']
        if (value['sample'] != sample['id'] or value['caption'] != caption
                or value['scope'] != 'first_image_and_request_only'
                or value['revision'] != ENCODING or value['first_image'] != expected_image
                or abs(value['duration'] - duration) > 1e-6
                or value['words'].shape[-1] != 4096
                or value['request_words'].shape[-1] != 4096
                or value['request_word_strings'] != caption.split()):
            raise ValueError('Input features or literal request belong to a different input')
        return {key: (value.to(self.device) if torch.is_tensor(value) else value)
                for key, value in value.items()}
