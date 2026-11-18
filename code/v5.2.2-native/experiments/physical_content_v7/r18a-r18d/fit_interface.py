"""Fit the fresh closed-form Qwen4096 -> native-3072 interface on train only.

This is a no-optimizer calibration of the frozen input features.  It never
loads a project model checkpoint and writes to the configured fresh output path.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

import torch

from physical_scaling.common import atomic_json, atomic_torch
from physical_content.common import FEATURE_CACHE, ROOT, load_config


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    torch.set_num_threads(16)
    begin = time.monotonic()
    inventory = json.loads((FEATURE_CACHE / 'inventory.json').read_text())
    xs, ys, counts = [], [], {}
    for item in inventory:
        if item['split'] != 'train' or item['name'].endswith('_null'):
            continue
        source = torch.load(FEATURE_CACHE / 'words' / f'{item["name"]}.pt',
                            map_location='cpu', weights_only=True)
        target = torch.load(FEATURE_CACHE / 'native_words' / f'{item["name"]}.pt',
                            map_location='cpu', weights_only=True)
        xs.append(source['words'].float())
        ys.append(target['native_words'].float())
        counts[item['sample']] = counts.get(item['sample'], 0) + 1
    if not xs:
        raise RuntimeError('No train feature pairs')
    training_words = sum(len(value) for value in xs)
    x = torch.cat(xs)
    y = torch.cat(ys)
    del xs, ys
    mean = x.mean(0)
    scale = (x - mean).square().mean().sqrt()
    bias = y.mean(0)
    target_scale = (y - bias).square().mean().sqrt()
    x.sub_(mean).div_(scale)
    y.sub_(bias)
    gram = x.T @ x / len(x)
    cross = x.T @ y / len(x)
    ridge = 0.01 * gram.diag().mean()
    gram.diagonal().add_(ridge)
    weight = torch.linalg.solve(gram, cross).T.contiguous()
    train_error = float((x @ weight.T - y).square().mean() / target_scale.square())
    del x, y, gram, cross

    heldout = []
    for item in inventory:
        if item['split'] != 'validation' or item['name'].endswith('_null'):
            continue
        source = torch.load(FEATURE_CACHE / 'words' / f'{item["name"]}.pt',
                            map_location='cpu', weights_only=True)
        target = torch.load(FEATURE_CACHE / 'native_words' / f'{item["name"]}.pt',
                            map_location='cpu', weights_only=True)['native_words'].float()
        predicted = (source['words'].float() - mean) / scale @ weight.T + bias
        error = float((predicted - target).square().mean())
        constant = float((bias - target).square().mean())
        heldout.append(dict(sample=item['sample'], mse=error,
                            constant_mse=constant,
                            prediction_centered_energy=float(
                                (predicted - predicted.mean(0)).square().mean())))
    relative = ((sum(row['mse'] for row in heldout)
                 / max(1e-20, sum(row['constant_mse'] for row in heldout)))
                if heldout else None)
    report = dict(
        source='train-split official Qwen word features -> native Wan text features',
        training_words=training_words,
        training_samples=len(counts),
        train_ids=sorted(counts),
        train_normalized_mse=train_error,
        ridge=float(ridge),
        heldout=heldout,
        heldout_relative_mse=relative,
        seconds=time.monotonic() - begin,
        initialization='fresh closed-form fit; no project model checkpoint',
        test_data_used_for_fit=False,
    )
    if heldout and relative >= 0.95:
        raise RuntimeError('Fresh interface does not materially beat a constant native context')
    output = Path(cfg['paths']['interface'])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output.with_suffix('.json'))
    atomic_torch(dict(
        mean=mean, scale=scale, weight=weight, bias=bias,
        target_scale=target_scale, training_samples=len(counts),
        revision='physics_expert_abcd_fresh400_r1'), output)
    print(json.dumps({key: value for key, value in report.items() if key not in ('heldout', 'train_ids')}))
    print(output)


if __name__ == '__main__':
    main()
