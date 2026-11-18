"""Train-only Qwen input moments; DT does not fit a Qwen-to-C value map."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import REVISION, atomic_json, atomic_torch, load_config


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    begin = time.monotonic()
    manifest = json.loads((Path(cfg['paths']['cache']) / 'base/manifest.json').read_text())
    training_ids = {row['id'] for row in manifest['records'] if row['split'] == 'train'}
    selected = set(json.loads(Path(cfg['train']['sample_sequence']).read_text())['ids'])
    if not selected or not selected <= training_ids:
        raise ValueError('Normalization must use exactly the selected training split')
    total = torch.zeros(cfg['model']['qwen_width'], dtype=torch.float64)
    squares = torch.zeros_like(total)
    count = 0
    for sample in sorted(selected):
        path = Path(cfg['paths']['words_cache']) / f'{sample}_text.pt'
        words = torch.load(path, map_location='cpu', weights_only=True)['words'].double()
        if words.ndim != 2 or words.shape[-1] != len(total) or not torch.isfinite(words).all():
            raise ValueError(f'Invalid cached training words: {sample}')
        total += words.sum(0)
        squares += words.square().sum(0)
        count += len(words)
    mean = total / count
    scale = (squares / count - mean.square()).mean().clamp_min(0).sqrt()
    if not torch.isfinite(scale) or scale <= 0:
        raise ValueError('Training word normalization is degenerate')
    output = Path(cfg['paths']['interface'])
    report = dict(revision=REVISION, plan=cfg['plan'], task=cfg['task'],
                  source='selected train-split frozen Qwen reasoning words',
                  training_samples=len(selected), training_words=count,
                  train_ids=sorted(selected), scale=float(scale),
                  seconds=time.monotonic() - begin,
                  learned_value_projection=False, test_data_used=False,
                  project_checkpoint_used=False)
    atomic_torch(dict(mean=mean.float(), scale=scale.float(), revision=REVISION,
                      plan=cfg['plan'], task=cfg['task'], training_samples=len(selected)), output)
    atomic_json(report, output.with_suffix('.json'))
    print(json.dumps({key: value for key, value in report.items() if key != 'train_ids'}))


if __name__ == '__main__':
    main()
