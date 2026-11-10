"""Inspect the v13 initializer using only cached train text and Wan's text MLP.

CPU-only: no full Wan/Qwen loading, video sampling, optimizer or new data.
Training computes the same initialization on its resident native text module.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch import nn
from safetensors import safe_open
from physgen_native.config import read_config, ALIGNED_REVISIONS
from physgen_native.context_alignment import context_statistics
from physgen_native.runtime import atomic_save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT/'configs/native_aligned_attention_wisa3857.yaml'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(); cfg = read_config(args.config)
    if cfg['revision'] not in ALIGNED_REVISIONS: raise ValueError('Use the aligned full-width configuration')
    torch.set_num_threads(4)
    root = Path(cfg['paths']['wan_checkpoint'])
    index = json.loads((root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
    tensors = {}
    for name, shard in index.items():
        if name.startswith('text_embedding.'):
            with safe_open(root/shard, framework='pt', device='cpu') as reader:
                tensors[name.removeprefix('text_embedding.')] = reader.get_tensor(name).to(torch.bfloat16)
    width, incoming = tensors['0.weight'].shape
    with torch.device('meta'):
        embedding = nn.Sequential(nn.Linear(incoming, width), nn.GELU(approximate='tanh'), nn.Linear(width, width))
    embedding.load_state_dict(tensors, assign=True); embedding.requires_grad_(False)
    native_cfg = json.loads((root/'config.json').read_text())
    wan = SimpleNamespace(text_embedding=embedding, text_len=native_cfg['text_len'])
    statistics = context_statistics(wan, cfg)
    statistics['report'].update(wan_checkpoint=str(root), revision=cfg['revision'],
                                formula='C = std_text * LayerNorm(projected_Qwen) + mean_text',
                                interpretation='Initialization scale only; semantic adaptation remains trainable')
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    atomic_save(statistics, out/'native_context_statistics.pt')
    (out/'native_context_statistics.json').write_text(json.dumps(statistics['report'], ensure_ascii=False, indent=2))
    print(json.dumps(statistics['report'], ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
