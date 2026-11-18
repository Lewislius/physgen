"""Train-only input normalization; no unused language/video regression fit."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import atomic_json, atomic_torch, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    rows = json.loads((Path(cfg['paths']['cache'])/'base/manifest.json').read_text())['records']
    train = {r['id'] for r in rows if r['split']=='train'}
    ids = sorted(set(json.loads(Path(cfg['train']['sample_sequence']).read_text())['ids']))
    if not set(ids) <= train:
        raise ValueError('Normalization may use only the selected training split')
    count, total, squares, used = 0, None, 0., []
    for sample in ids:
        path = Path(cfg['paths']['words_cache'])/f'{sample}_text.pt'
        if not path.is_file():
            continue
        words = torch.load(path, map_location='cpu', weights_only=True)['words'].double()
        current = words.sum(0)
        total = current if total is None else total + current
        squares += words.square().sum().item()
        count += len(words)
        used.append(sample)
    if not count:
        raise ValueError('No positive training words for normalization')
    mean = total/count
    scale = (torch.tensor(squares/(count*len(mean)), dtype=torch.float64) - mean.square().mean()).sqrt()
    if not scale.isfinite() or scale <= 0:
        raise ValueError('Training normalization is degenerate')
    output = Path(cfg['paths']['interface'])
    atomic_torch(dict(mean=mean.float(), scale=scale.float(), revision=cfg['revision'],
                      task='i2v', plan='EI', training_samples=len(used)), output)
    report = dict(training_samples=len(used), training_words=count, train_ids=used,
                  normalization_only=True, projected_language_to_video=False,
                  benchmark_used=False, visual_features_normalized_by_own_trainable_LN=True)
    atomic_json(report, output.with_suffix('.json'))
    print(json.dumps({k:v for k,v in report.items() if k!='train_ids'}))


if __name__ == '__main__':
    main()
