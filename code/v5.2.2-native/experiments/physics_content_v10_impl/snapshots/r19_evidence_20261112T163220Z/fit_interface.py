"""Fit the fresh closed-form Qwen4096 -> native-3072 interface on train only.

I2V and T2V must use separate fits because their Qwen evidence has different
input scope.  No optimizer step is performed and no test row is read.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch

from physical_content.common import REVISION, atomic_json, atomic_torch, load_config


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    torch.set_num_threads(16)
    begin = time.monotonic()
    cache = Path(cfg['paths']['cache'])
    words_root = Path(cfg['paths']['words_cache'])
    native_root = Path(cfg['paths']['native_words_cache'])
    rows = json.loads((cache / 'base/manifest.json').read_text())['records']
    train_ids = [row['id'] for row in rows if row['split'] == 'train']
    xs, ys, counts = [], [], {}
    for sample in train_ids:
        name = f'{sample}_text'
        word_path = words_root / f'{name}.pt'
        native_path = native_root / f'{name}.pt'
        if not word_path.is_file() or not native_path.is_file():
            continue
        xs.append(torch.load(word_path, map_location='cpu', weights_only=True)['words'].float())
        ys.append(torch.load(native_path, map_location='cpu',
                             weights_only=True)['native_words'].float())
        counts[sample] = counts.get(sample, 0) + 1
    if not xs:
        raise RuntimeError(f'No {cfg["task"]} train feature pairs')
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
    for row in rows:
        if row['split'] != 'validation':
            continue
        name = f'{row["id"]}_text'
        word_path = words_root / f'{name}.pt'
        native_path = native_root / f'{name}.pt'
        if not word_path.is_file() or not native_path.is_file():
            continue
        source = torch.load(word_path, map_location='cpu',
                            weights_only=True)['words'].float()
        target = torch.load(native_path, map_location='cpu',
                            weights_only=True)['native_words'].float()
        predicted = (source - mean) / scale @ weight.T + bias
        heldout.append(dict(
            sample=row['id'],
            mse=float((predicted - target).square().mean()),
            constant_mse=float((bias - target).square().mean()),
            prediction_centered_energy=float(
                (predicted - predicted.mean(0)).square().mean())))
    relative = ((sum(row['mse'] for row in heldout)
                 / max(1e-20, sum(row['constant_mse'] for row in heldout)))
                if heldout else None)
    report = dict(
        source='train-split frozen Qwen word features -> native Wan text features',
        plan=cfg['plan'], task=cfg['task'], training_words=training_words,
        training_samples=len(counts), train_ids=sorted(counts),
        train_normalized_mse=train_error, ridge=float(ridge), heldout=heldout,
        heldout_relative_mse=relative, seconds=time.monotonic() - begin,
        initialization='fresh closed-form fit; no project model checkpoint',
        test_data_used_for_fit=False)
    if heldout and relative >= 0.95:
        raise RuntimeError('Fresh interface does not materially beat a constant native context')
    output = Path(cfg['paths']['interface'])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output.with_suffix('.json'))
    atomic_torch(dict(
        mean=mean, scale=scale, weight=weight, bias=bias,
        target_scale=target_scale, training_samples=len(counts),
        revision=REVISION, plan=cfg['plan'], task=cfg['task']), output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('heldout', 'train_ids')}))
    print(output)


if __name__ == '__main__':
    main()
