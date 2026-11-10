"""New-data inference inputs and C interventions, without changing old runners.

Only reasoning text changes in content controls. The own first image, literal
request and relation-count prior remain explicit parts of the comparison.
"""
from pathlib import Path
import json
import torch


def encode_control(knowledge, image, caption, duration, own, *, sample,
                   mode, donor=None):
    from encode import encode_input
    from physical_content.evidence import recover_request_words
    if own['caption'] != caption or own['scope'] != 'first_image_and_request_only':
        raise ValueError('Control must retain the own first-image/request input')
    count = len(own['relations'])
    if not 1 <= count <= 4:
        raise ValueError('Expected one to four own relations')
    revision = 'qwen38_first_image_request_v1'
    if mode == 'reasoning_swapped':
        if donor is None or donor['scope'] != 'first_image_and_request_only':
            raise ValueError('An input-only reasoning donor is required')
        if donor.get('sample') == sample or not donor.get('sample'):
            raise ValueError('Reasoning donor must be a different recorded input')
        if len(donor['relations']) != count:
            raise ValueError('Use a donor with the same relation count; do not change the null-reference prior')
        changed = dict(donor, caption=caption)
        evidence, words = encode_input(knowledge, image, caption, duration,
            changed, sample=sample, revision=revision)
        evidence.update(words)
        evidence['donor'] = donor['sample']
    elif mode == 'empty_reasoning':
        # The retained structural delimiters have contextual states but contain
        # no physical sentences. This is not a no-image/no-Qwen intervention.
        relations = [''] * count
        spans = [f'[RELATION {i+1}] ' for i in range(count)]
        body = (f'Request: {caption or "No text request."}\nPresentation interval: {duration:.3f} seconds.\n'
                'Conditional physical analysis:\n' + '\n'.join(spans))
        inputs, positions = knowledge._inputs([(image, image.new_zeros(1))], body, extract_spans=spans)
        hidden = knowledge.backbone.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
        if hidden.shape[-1] != 4096 or any(not indices for indices in positions):
            raise ValueError('Invalid local Qwen delimiter encoding')
        vectors = torch.stack([hidden[:, indices].mean(1) for indices in positions], 1)
        evidence = dict(memory=hidden.cpu().to(torch.bfloat16),
            memory_mask=inputs['attention_mask'].bool().cpu(),
            relation_vectors=vectors.cpu().to(torch.bfloat16),
            relations=relations, caption=caption, analysis='', analysis_complete=True,
            scope='first_image_and_request_only', sample=sample, revision=revision,
            words=vectors[0].cpu().to(torch.bfloat16),
            word_groups=torch.arange(count), word_positions=torch.zeros(count, dtype=torch.long),
            word_strings=[''] * count, donor=None,
            empty_slot_definition='Contextual state of each retained relation delimiter; no physical sentence')
        evidence.update(recover_request_words(evidence, duration, knowledge.tokenizer))
    else:
        raise ValueError(mode)
    evidence.update(intervention=mode, own_relation_count=count,
        intervention_scope='Own image/request and relation count retained; only positive reasoning sentences changed')
    return evidence


class InferenceCache:
    """Explicit prepared roots only; never falls back to the old Qwen evidence."""
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
                or not ready['native_wan_prompt_unchanged']):
            raise ValueError('Prepared inference inputs do not match the benchmark image/request/duration')
        if drop:
            path = folder / 'negative.pt'
        elif mode in ('reasoning_swapped', 'empty_reasoning'):
            path = folder / 'controls' / f'{mode}.pt'
        else:
            path = folder / 'positive.pt'
        value = torch.load(path, map_location='cpu', weights_only=True)
        if (value['sample'] != sample['id'] or value['caption'] != ('' if drop else sample['caption'])
                or value['scope'] != 'first_image_and_request_only'
                or value['revision'] != 'qwen38_first_image_request_v1'
                or value.get('first_image') != expected_image
                or abs(value.get('duration', -1) - duration) > 1e-6
                or value['memory'].shape[-1] != 4096 or value['words'].shape[-1] != 4096
                or value['request_words'].shape[-1] != 4096):
            raise ValueError('Inference evidence belongs to a different input or encoder')
        if not drop and mode in ('reasoning_swapped', 'empty_reasoning'):
            if value.get('intervention') != mode or len(value['relations']) != value['own_relation_count']:
                raise ValueError('Reasoning intervention metadata mismatch')
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in value.items()}


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
