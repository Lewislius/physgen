"""Fit the R16Gt Qwen4096 -> native-3072 interface on request-only train rows.

I2V and T2V have different Qwen input scopes, so R16Gt does not reuse the I2V
R16G interface.  The fit is closed-form and reads only train-split word targets.
"""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch

from physical_content.common import (DATA_REVISION, REVISION, atomic_json,
                                     atomic_torch, load_config)


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_num_threads(16)
    begin = time.monotonic()
    cache = Path(cfg['paths']['cache'])
    words_root = Path(cfg['paths']['t2v_words_cache'])
    native_root = Path(cfg['paths']['t2v_native_words_cache'])
    rows = json.loads((cache / 'base/manifest.json').read_text())['records']
    train_ids = [row['id'] for row in rows if row['split'] == 'train']
    xs, ys, counts = [], [], {}
    for sample in train_ids:
        name = f'{sample}_text'
        word_path = words_root / f'{name}.pt'
        native_path = native_root / f'{name}.pt'
        if not word_path.is_file() or not native_path.is_file():
            continue
        x = torch.load(word_path, map_location='cpu', weights_only=True)['words'].float()
        y = torch.load(native_path, map_location='cpu',
                       weights_only=True)['native_words'].float()
        if x.shape[-1] != 4096 or y.shape[-1] != 3072 or len(x) != len(y):
            raise ValueError(f'Feature width/word count differs for {sample}')
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f'Nonfinite interface pair for {sample}')
        xs.append(x)
        ys.append(y)
        counts[sample] = counts.get(sample, 0) + 1
    if not xs:
        raise RuntimeError('No request-only train feature pairs')
    training_words = sum(len(value) for value in xs)
    x = torch.cat(xs)
    y = torch.cat(ys)
    del xs, ys
    mean = x.mean(0)
    scale = (x - mean).square().mean().sqrt()
    if not torch.isfinite(scale) or float(scale) <= 0:
        raise ValueError('Degenerate request-only feature scale')
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
    if not torch.isfinite(torch.tensor(train_error)) or train_error >= 1.0:
        raise RuntimeError('Request-only interface does not beat a constant target')
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
            constant_mse=float((bias - target).square().mean())))
    relative = ((sum(row['mse'] for row in heldout)
                 / max(1e-20, sum(row['constant_mse'] for row in heldout)))
                if heldout else None)
    report = dict(
        source='train-split request-only frozen Qwen word features -> native Wan text features',
        plan=cfg['plan'], task=cfg['task'], data_revision=DATA_REVISION,
        training_words=training_words, training_samples=len(counts),
        train_ids=sorted(counts), train_normalized_mse=train_error,
        ridge=float(ridge), heldout=heldout, heldout_relative_mse=relative,
        seconds=time.monotonic() - begin,
        initialization='fresh closed-form fit; no project model checkpoint',
        test_data_used_for_fit=False)
    if heldout and relative >= .95:
        raise RuntimeError('Fresh interface does not materially beat a constant target')
    output = Path(cfg['paths']['interface'])
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(report, output.with_suffix('.json'))
    atomic_torch(dict(
        mean=mean, scale=scale, weight=weight, bias=bias,
        target_scale=target_scale, training_samples=len(counts),
        revision=REVISION, plan=cfg['plan'], task=cfg['task'],
        data_revision=DATA_REVISION), output)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('heldout', 'train_ids')}))
    print(output)


if __name__ == '__main__':
    main()
