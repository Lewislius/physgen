"""Read compact inference evidence for the two task entries."""
from pathlib import Path
import json

import torch

ENCODING = 'Wan21-R20FT-20261116'


def prepared_i2v_control(root, case_id, mode, *, device='cpu'):
    """Reuse an actual re-encoded empty/wrong reasoning on the OWN image.

    The ordinary negative I2V arm contains nonempty image-only physics. It is
    not an empty-reasoning control.
    """
    if mode not in ('empty_reasoning', 'reasoning_swapped'):
        raise ValueError(mode)
    path = Path(root) / f'videophy40_{case_id}' / 'controls' / f'{mode}.pt'
    value = torch.load(path, map_location='cpu', weights_only=True)
    if (value.get('sample') != f'videophy40_{case_id}'
            or value.get('intervention') != mode
            or value.get('scope') != 'first_image_and_request_only'):
        raise ValueError('Prepared control has a different identity or intervention')
    if mode == 'empty_reasoning' and any(r.strip() for r in value['relations']):
        raise ValueError('Empty reasoning must have no physical sentences')
    value['control_source'] = str(path.resolve())
    return {key: item.to(device) if torch.is_tensor(item) else item
            for key, item in value.items()}


def validate_control_input(value, item):
    if (value['caption'] != item['caption']
            or value['request_word_strings'] != item['caption'].split()
            or Path(value['first_image']).resolve() != Path(item['image']).resolve()
            or abs(value['duration'] - item['duration']) > 1e-6):
        raise ValueError('Reasoning control must preserve own prompt/image/duration')


class InferenceCache:
    def __init__(self, root, task, device='cuda', *, native_width=1536):
        if task != 't2v':
            raise ValueError('R20FT-Wan2.1 inference cache is text-to-video only')
        self.root = Path(root)
        self.task = task
        self.device = device
        self.native_width = int(native_width)

    def input(self, sample, *, image=None, duration=None, drop=False):
        if image is not None:
            raise ValueError('Wan2.1 T2V must not receive a first image')
        folder = self.root / sample['id']
        ready = json.loads((folder / 'ready.json').read_text())
        if (ready['id'] != sample['id'] or ready['caption'] != sample['caption']
                or ready['scope'] != 'request_only'
                or ready['revision'] != ENCODING):
            raise ValueError('Prepared Wan2.1 evidence differs from the fixed protocol')
        path = folder / ('negative.pt' if drop else 'positive.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        if (value['sample'] != sample['id'] or value['caption'] !=
                ('' if drop else sample['caption'])
                or value['scope'] != 'request_only'
                or value['request_word_strings'] !=
                ([] if drop else sample['caption'].split())):
            raise ValueError('Prepared condition belongs to another input')
        native = value.get('native_condition')
        if (not hasattr(native, 'shape') or native.ndim != 2
                or native.shape[1] != self.native_width):
            raise ValueError(
                f'Inference native_condition is not Wan2.1 width {self.native_width}')
        return {key: (item.to(self.device) if torch.is_tensor(item) else item)
                for key, item in value.items()}
