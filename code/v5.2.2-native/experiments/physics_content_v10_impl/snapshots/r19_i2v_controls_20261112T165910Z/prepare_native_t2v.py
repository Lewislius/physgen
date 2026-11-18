"""Build T5/Wan native route targets for the request-only T2V evidence."""
from pathlib import Path
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch
from torch import nn
from safetensors import safe_open

from physical_content.common import atomic_json, atomic_torch, load_config

WAN = Path('/home/liuzhirui/model/Wan2.2/Wan2.2-TI2V-5B')


def native_embedding(device):
    index = json.loads((WAN / 'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    module = nn.Sequential(nn.Linear(4096, 3072), nn.GELU(approximate='tanh'),
                           nn.Linear(3072, 3072))
    weights = {}
    for key in module.state_dict():
        full = 'text_embedding.' + key
        with safe_open(WAN / index[full], framework='pt', device='cpu') as source:
            weights[key] = source.get_tensor(full)
    module.load_state_dict(weights)
    return module.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    if cfg['task'] != 't2v':
        raise ValueError('This script is only for T2V plans')
    sys.path.insert(0, '/home/liuzhirui/model/Wan2.2')
    from wan.modules.t5 import T5EncoderModel
    encoder = T5EncoderModel(
        text_len=128, dtype=torch.bfloat16, device=args.device,
        checkpoint_path=str(WAN / 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(WAN / 'google/umt5-xxl'))
    native = native_embedding(args.device)
    words_root = Path(cfg['paths']['words_cache'])
    output = Path(cfg['paths']['native_words_cache'])
    output.mkdir(parents=True, exist_ok=True)
    files = sorted(words_root.glob('*.pt'))
    started = time.monotonic()
    for index, path in enumerate(files):
        destination = output / path.name
        if destination.is_file():
            continue
        name = path.stem
        sources = [Path(cfg['paths']['input_cache']) / f'{name}.pt',
                   Path(cfg['paths']['eval_input_cache']) / f'{name}.pt']
        source = next((path for path in sources if path.is_file()), None)
        if source is None:
            raise FileNotFoundError(f'No T2V input evidence for {name}')
        evidence = torch.load(source, map_location='cpu', weights_only=True)
        words = torch.load(path, map_location='cpu', weights_only=True)
        relations = evidence['relations']
        strings = words['word_strings']
        groups = words['word_groups'].tolist()
        features = []
        tokens = []
        for group, relation in enumerate(relations):
            count = sum(1 for item in groups if item == group)
            if not relation:
                if count != 1:
                    raise ValueError('Empty relation must have one delimiter slot')
                features.append(torch.zeros(3072))
                tokens.append('')
                continue
            clean = encoder.tokenizer._clean(relation)
            encoded = encoder.tokenizer.tokenizer(
                clean, return_tensors='pt', return_offsets_mapping=True,
                add_special_tokens=True, padding=False, truncation=True,
                max_length=128)
            offsets = encoded.pop('offset_mapping')[0]
            with torch.autocast(args.device, dtype=torch.bfloat16):
                hidden = encoder.model(encoded['input_ids'].to(args.device),
                                       encoded['attention_mask'].to(args.device))
                target = native(hidden).float()[0]
            matches = list(re.finditer(r'\S+', clean))
            group_strings = [s for s, g in zip(strings, groups) if g == group]
            if len(matches) != len(group_strings):
                raise ValueError('Native tokenization changed a relation word count')
            for match, word in zip(matches, group_strings):
                selected = ((offsets[:, 1] > match.start()) &
                            (offsets[:, 0] < match.end())).nonzero().flatten()
                if not len(selected):
                    raise ValueError('Native physical word has no token')
                features.append(target[selected.to(target.device)].mean(0))
                tokens.append(word)
        atomic_torch(dict(native_words=torch.stack(features).cpu().to(torch.bfloat16),
                          word_strings=tokens, scope='request_only',
                          sample=evidence['sample'],
                          teacher='official frozen T5 then official Wan text_embedding'),
                     destination)
        if index % 50 == 0:
            print(json.dumps(dict(stage='native_t2v', completed=index + 1,
                                  total=len(files), seconds=time.monotonic() - started)),
                  flush=True)
    atomic_json(dict(status='complete', count=len(files),
                     seconds=time.monotonic() - started),
                output.parent / 'native_ready.json')


if __name__ == '__main__':
    main()
