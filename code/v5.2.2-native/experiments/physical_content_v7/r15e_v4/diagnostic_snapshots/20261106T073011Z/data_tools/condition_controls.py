"""Original frozen Qwen inputs and explicit paired C interventions."""
from pathlib import Path
import json
import torch

ENCODING = 'original_qwen_first_image_request_v1'


class InferenceCache:
    """Read compact original inputs prepared without API or model calls."""
    def __init__(self, root, device='cuda'):
        self.root = Path(root)
        self.device = device

    def input(self, sample, *, image, duration, drop=False, mode='real'):
        folder = self.root / sample['id']
        ready = json.loads((folder / 'ready.json').read_text())
        expected_image = str(Path(image).resolve())
        if (ready['id'] != sample['id'] or ready['caption'] != sample['caption']
                or ready['image'] != expected_image or abs(ready['duration'] - duration) > 1e-6
                or ready['scope'] != 'first_image_and_request_only' or ready['width'] != 4096
                or ready['revision'] != ENCODING or not ready['native_wan_prompt_unchanged']):
            raise ValueError('Prepared inputs differ from the original benchmark')
        intervention = not drop and mode in ('reasoning_swapped', 'empty_reasoning')
        path = (folder / 'negative.pt' if drop else
                folder / 'controls' / (mode + '.pt') if intervention else folder / 'positive.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)
        caption = '' if drop else sample['caption']
        if (value['sample'] != sample['id'] or value['caption'] != caption
                or value['scope'] != 'first_image_and_request_only'
                or value['revision'] != ENCODING or value['first_image'] != expected_image
                or abs(value['duration'] - duration) > 1e-6
                or value['words'].shape[-1] != 4096 or value['request_words'].shape[-1] != 4096
                or value['request_word_strings'] != caption.split()):
            raise ValueError('Input features or literal request belong to a different input')
        if intervention and (value['intervention'] != mode
                             or len(value['relations']) != value['own_relation_count']):
            raise ValueError('Reasoning control metadata mismatch')
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in value.items()}


class MeanContent:
    """Average full C tensors without pretending relation groups are constant."""
    def __init__(self):
        self.sums = {}
        self.squared_rms = {}
        self.count = 0
        self.relation_counts = {}

    def add(self, content, groups):
        if set(content) != {'routing', 'execution'}:
            raise ValueError('Mean intervention requires both routing and execution C')
        relations = int(groups.max()) + 1
        for key, tensor in content.items():
            value = tensor.detach().float().cpu()
            if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] != len(groups):
                raise ValueError('Incompatible full-C tensor shape')
            if key in self.sums and self.sums[key].shape != value.shape:
                raise ValueError('C slot/width must match across inputs')
            self.sums[key] = self.sums.get(key, torch.zeros_like(value)) + value
            self.squared_rms[key] = self.squared_rms.get(key, 0.) + float(value.square().mean())
        self.count += 1
        self.relation_counts[relations] = self.relation_counts.get(relations, 0) + 1

    def finish(self):
        if not self.count:
            raise ValueError('No training inputs were accumulated')
        means = {k: v / self.count for k, v in self.sums.items()}
        report = dict(count=self.count, relation_counts=self.relation_counts,
            rms={k: dict(mean_condition=float(v.square().mean().sqrt()),
                per_sample_rms_root_mean_square=(self.squared_rms[k] / self.count) ** .5)
                for k, v in means.items()})
        return means, report


def mean_with_own_prior(saved, positive_groups, negative_groups, device):
    """Fix all C content; retain each arm's original null-reference prior.

    Relation count remains input-specific metadata. Do not describe this as
    removal of *all* sample-specific Qwen information. C amplitude also changes.
    """
    values = []
    for arm, groups in (('positive', positive_groups), ('negative', negative_groups)):
        content = {k: v.to(device) for k, v in saved[arm].items()}
        if any(v.shape[1] != len(groups) for v in content.values()):
            raise ValueError('Fixed mean C and own group prior have different slot counts')
        values.extend((content, groups))
    return tuple(values)
