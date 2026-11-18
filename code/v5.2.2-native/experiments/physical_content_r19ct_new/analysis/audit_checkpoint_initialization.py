"""Audit a trained R19CTN checkpoint against the real Wan2.2 block depths.

For every injection site and every expected copied tensor, find the best-matching
official Wan block.  A correct checkpoint must match the same-numbered block;
any other best match is a layer-mapping failure.

Run after training:

  /home/liuzhirui/miniconda3/envs/moviestory/bin/python \
    analysis/audit_checkpoint_initialization.py \
    --checkpoint runs/r19ctn_prophy5_state_400/step0400.pt
"""
from pathlib import Path
import argparse
import json
import os
import sys

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401
from physical_content.common import load_config


def cosine(left, right):
    left = left.flatten().float()
    right = right.flatten().float()
    left = left / (left.norm() + 1e-12)
    right = right / (right.norm() + 1e-12)
    return float((left * right).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/r19ctn.yaml')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--top', type=int, default=3)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, require_paths=False)
    saved = torch.load(args.checkpoint, map_location='cpu',
                       weights_only=False, mmap=True)
    state = saved['model']
    sites = [int(site) for site in cfg['model']['injection_layers']]
    root = Path(cfg['paths']['wan_checkpoint'])
    index = json.loads((root / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    readers = {}

    def official(key):
        filename = weight_map[key]
        if filename not in readers:
            readers[filename] = safe_open(str(root / filename), framework='pt', device='cpu')
        return readers[filename].get_tensor(key)

    # checkpoint key, official key template, short label
    checks = [
        ('mechanisms.{site}.video_query.weight', 'blocks.{b}.cross_attn.q.weight', 'read.Q'),
        ('mechanisms.{site}.content_key.weight', 'blocks.{b}.cross_attn.k.weight', 'read.K'),
        ('mechanisms.{site}.video_value.weight', 'blocks.{b}.cross_attn.v.weight', 'read.V'),
        ('mechanisms.{site}.read_output.weight', 'blocks.{b}.cross_attn.o.weight', 'read.O'),
        ('state_experts.{site}.condition_attention.q.weight',
         'blocks.{b}.cross_attn.q.weight', 'state.cond.Q'),
        ('state_experts.{site}.relation_attention.q.weight',
         'blocks.{b}.self_attn.q.weight', 'state.relation.Q'),
        ('state_experts.{site}.ffn.layers.0.weight', 'blocks.{b}.ffn.0.weight', 'state.ffn0'),
    ]
    if cfg['model']['video_value_init'] == 'identity':
        # Identity V has no Wan-layer source; only its per-site independence matters.
        checks = [check for check in checks if check[2] != 'read.V']
    report = dict(checkpoint=str(args.checkpoint), sites=sites,
                  video_value_init=cfg['model']['video_value_init'], checks=[])
    failed = []
    for ckpt_template, official_template, label in checks:
        # Read each official tensor once per layer and score all sites.
        scores = {site: [] for site in sites}
        for layer in range(int(cfg['model']['native_layers'])):
            source = official(official_template.format(b=layer)).float()
            for site in sites:
                key = ckpt_template.format(site=site)
                if key not in state:
                    raise KeyError(f'Checkpoint lacks {key}')
                scores[site].append((cosine(state[key], source), layer))
        for site in sites:
            ranked = sorted(scores[site], reverse=True)
            top = ranked[:args.top]
            best_layer = top[0][1]
            record = dict(site=site, key=label, best_layer=best_layer,
                          best_cosine=round(top[0][0], 6),
                          top=[dict(layer=layer, cosine=round(value, 6))
                               for value, layer in top])
            report['checks'].append(record)
            status = 'OK' if best_layer == site else 'FAIL'
            print(f'{status:4s} site={site:2d} {label:16s} best=block{best_layer} '
                  f'cos={top[0][0]:.6f} top={[(layer, round(value, 4)) for value, layer in top]}',
                  flush=True)
            if best_layer != site:
                failed.append(record)
    report['passed'] = not failed
    report['failed'] = failed
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        raise SystemExit(f'{len(failed)} same-depth initialization checks failed')
    print(json.dumps(dict(passed=True, checks=len(report['checks'])),
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
