"""Native encodings of input-only physical reasoning, with lexical provenance."""
import html
import re

import ftfy
import torch


def clean(text):
    return re.sub(r'\s+', ' ', html.unescape(html.unescape(
        ftfy.fix_text(text)))).strip()


def check_native(value, evidence, *, sample=None):
    expected = evidence['sample'] if sample is None else sample
    if (value.get('scope') != 'first_image_request_reasoning_only'
            or value.get('sample') != expected
            or value.get('teacher') != 'official frozen T5 then official Wan text_embedding'):
        raise ValueError('Native physics input has different origin or scope')
    native_strings = [clean(s) for s in value['word_strings']]
    input_strings = [clean(s) for s in evidence['word_strings']]
    if native_strings != input_strings:
        raise ValueError('Native physical words do not match this input analysis')
    if tuple(value['native_words'].shape) != (len(input_strings), 3072):
        raise ValueError('Native physical feature shape differs')
    return value['native_words']


class NativeEncoder:
    """Use already installed frozen T5/Wan only for missing input word caches."""

    def __init__(self, cfg, device='cuda'):
        from pathlib import Path
        import json
        from safetensors import safe_open
        from torch import nn
        from .wan_loader import ensure_wan_namespace
        ensure_wan_namespace(cfg)
        from wan.modules.t5 import T5EncoderModel
        root = Path(cfg['paths']['wan_checkpoint'])
        self.device = device
        self.encoder = T5EncoderModel(
            text_len=128, dtype=torch.bfloat16, device=device,
            checkpoint_path=str(root/'models_t5_umt5-xxl-enc-bf16.pth'),
            tokenizer_path=str(root/'google/umt5-xxl'))
        self.native = nn.Sequential(nn.Linear(4096, 3072), nn.GELU(approximate='tanh'),
                                    nn.Linear(3072, 3072))
        index = json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        state = {}
        for key in self.native.state_dict():
            full = 'text_embedding.' + key
            with safe_open(root/index[full], framework='pt', device='cpu') as f:
                state[key] = f.get_tensor(full)
        self.native.load_state_dict(state)
        self.native.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, evidence):
        features, strings = [], []
        groups = evidence['word_groups']
        for group, relation in enumerate(evidence['relations']):
            expected = [s for s, g in zip(evidence['word_strings'], groups.tolist()) if g == group]
            if not relation.strip():
                if expected != ['']:
                    raise ValueError('Empty physical relation must contain its one empty slot')
                features.append(torch.zeros(3072)); strings.append(''); continue
            text = self.encoder.tokenizer._clean(relation)
            encoded = self.encoder.tokenizer.tokenizer(
                text, return_tensors='pt', return_offsets_mapping=True,
                add_special_tokens=True, padding=False, truncation=False)
            # Encode every word; no silent per-relation truncation.
            offsets = encoded.pop('offset_mapping')[0]
            with torch.autocast(self.device, dtype=torch.bfloat16):
                hidden = self.encoder.model(encoded['input_ids'].to(self.device),
                                            encoded['attention_mask'].to(self.device))
                native = self.native(hidden).float()[0]
            matches = list(re.finditer(r'\S+', text))
            if [clean(s) for s in expected] != [m.group() for m in matches]:
                raise ValueError('Native tokenizer changed physical lexical spans')
            for m, word in zip(matches, expected):
                indices = ((offsets[:, 1] > m.start()) & (offsets[:, 0] < m.end())).nonzero().flatten()
                if not len(indices):
                    raise ValueError('A physical word was not encoded')
                features.append(native[indices.to(native.device)].mean(0).cpu())
                strings.append(word)
        return dict(native_words=torch.stack(features).to(torch.bfloat16),
                    word_strings=strings, scope='first_image_request_reasoning_only',
                    sample=evidence['sample'],
                    teacher='official frozen T5 then official Wan text_embedding')
