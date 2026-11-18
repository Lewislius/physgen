"""Read the immutable first-image/request Qwen evidence and lexical word spans."""
import re

import torch
import torch.nn.functional as F

def memory_text(evidence, duration, tokenizer):
    spans = [f'[RELATION {i + 1}] {s}' for i, s in enumerate(evidence['relations'])]
    caption = evidence['caption']
    body = (f'Request: {caption or "No text request."}\n'
            f'Presentation interval: {duration:.3f} seconds.\n'
            'Conditional physical analysis:\n' + '\n'.join(spans))

    def render(count):
        content = ('Initial observation:\n<|vision_start|>' + '<|image_pad|>' * count
                   + '<|vision_end|>\n' + body)
        return tokenizer.apply_chat_template(
            [{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)

    initial = tokenizer(render(1), add_special_tokens=False)['input_ids']
    count = evidence['memory'].shape[1] - len(initial) + 1
    if count < 1:
        raise ValueError('Input template does not match cached evidence')
    text = render(count)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded['input_ids']) != evidence['memory'].shape[1]:
        raise ValueError('Cached memory length differs')
    return text, torch.tensor(encoded['offset_mapping']), spans


def recover_request_words(evidence, duration, tokenizer):
    caption = evidence['caption']
    hidden = evidence['memory'][0]
    if not caption:
        return dict(request_words=hidden.new_empty((0, hidden.shape[-1])),
                    request_word_strings=[])
    text, offsets, _ = memory_text(evidence, duration, tokenizer)
    start = text.index('Request: ') + len('Request: ')
    if text[start:start + len(caption)] != caption:
        raise ValueError('Request span differs from the immutable input caption')
    values, strings = [], []
    for match in re.finditer(r'\S+', caption):
        a, b = start + match.start(), start + match.end()
        index = ((offsets[:, 1] > a) & (offsets[:, 0] < b)).nonzero().flatten()
        if not len(index):
            raise ValueError('Request word has no cached Qwen tokens')
        values.append(hidden.index_select(0, index.to(hidden.device)).float().mean(0))
        strings.append(match.group())
    return dict(request_words=torch.stack(values).to(hidden.dtype),
                request_word_strings=strings)


def recover_visual_tokens(evidence, duration, tokenizer):
    """Recover image-position hidden states, before request/analysis positions.

    This does not run Qwen or encode a future video. Its source is the existing
    image span of the frozen input memory used by the original R1. These are
    language-backbone hidden states, not a separately encoded raw vision tower.
    """
    if evidence.get('scope') != 'first_image_and_request_only':
        raise ValueError('Visual memory must originate from the known first image')
    text, offsets, _ = memory_text(evidence, duration, tokenizer)
    start = text.index('<|vision_start|>') + len('<|vision_start|>')
    end = text.index('<|vision_end|>', start)
    index = ((offsets[:, 0] >= start) & (offsets[:, 1] <= end)
             & (offsets[:, 1] > offsets[:, 0])).nonzero().flatten()
    if not len(index) or any(text[a:b] != '<|image_pad|>' for a, b in offsets[index].tolist()):
        raise ValueError('Recovered visual memory includes non-image tokens')
    if not evidence['memory_mask'][0, index].all():
        raise ValueError('Visual span contains masked tokens')
    return dict(visual_tokens=evidence['memory'][:, index].contiguous(),
                visual_source='frozen_Qwen_first_image_tokens_before_request',
                visual_token_count=len(index))


def recover_words(evidence, duration, tokenizer):
    text, offsets, spans = memory_text(evidence, duration, tokenizer)
    hidden = evidence['memory'][0].float()
    values, groups, strings, positions, span_errors = [], [], [], [], []
    for group, (span, relation) in enumerate(zip(spans, evidence['relations'])):
        start = text.index(span)
        end = start + len(span)
        index = ((offsets[:, 1] > start) & (offsets[:, 0] < end)).nonzero().flatten()
        original = evidence['relation_vectors'][0, group].float()
        relative = float((hidden[index].mean(0) - original).norm() / original.norm().clamp_min(1e-8))
        span_errors.append(relative)
        if relative > .01:
            raise ValueError(f'Cached relation span mismatch: {relative}')
        relation_start = start + len(span) - len(relation)
        for position, match in enumerate(re.finditer(r'\S+', relation)):
            a, b = relation_start + match.start(), relation_start + match.end()
            index = ((offsets[:, 1] > a) & (offsets[:, 0] < b)).nonzero().flatten()
            if not len(index):
                raise ValueError('Physical word has no Qwen tokens')
            values.append(hidden.index_select(0, index.to(hidden.device)).mean(0))
            groups.append(group)
            strings.append(match.group())
            positions.append(position)
    return dict(words=torch.stack(values).to(torch.bfloat16),
                word_groups=torch.tensor(groups),
                word_positions=torch.tensor(positions),
                word_strings=strings,
                relations=evidence['relations'],
                scope=evidence['scope'],
                span_errors=span_errors)


def resample_words(words, word_groups, count=256):
    """Use the same lexical sequence for enhancement anchors and targets."""
    if words.ndim == 2:
        words = words[None]
    groups = torch.arange(count, device=words.device) * int(word_groups.max() + 1) // count
    chunks, positions = [], []
    for group in range(int(word_groups.max()) + 1):
        x = words[:, word_groups == group]
        slots = int((groups == group).sum())
        chunks.append(F.interpolate(
            x.transpose(1, 2).float(), size=slots, mode='linear', align_corners=True).transpose(1, 2))
        positions.append(torch.linspace(0, len(x[0]) - 1, slots, device=words.device))
    return torch.cat(chunks, 1), groups, torch.cat(positions)
