"""Training-only normalization; no learned value map or model/API forward."""
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
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    torch.set_num_threads(4)
    started = time.monotonic()
    reference = Path(cfg['paths']['normalization_reference'])
    moments = torch.load(reference, map_location='cpu', weights_only=True)
    if moments['mean'].shape != (cfg['model']['qwen_width'],):
        raise ValueError('Qwen normalization width differs')
    manifest = json.loads((Path(cfg['paths']['cache'])/'base/manifest.json').read_text())
    train_ids = {x['id'] for x in manifest['records'] if x['split'] == 'train'}
    seq = json.loads(Path(cfg['train']['sample_sequence']).read_text())
    if not set(seq['ids']) <= train_ids:
        raise ValueError('Normalization may only use selected training inputs')
    square, count, zeros = 0., 0, 0
    for sample, drop in zip(seq['ids'], seq['drop']):
        name = sample+('_null' if drop else '_text')
        path = Path(cfg['paths']['native_words_cache'])/(name+'.pt')
        value = torch.load(path, map_location='cpu', weights_only=True)['native_words'].double()
        if value.ndim != 2 or value.shape[1] != cfg['model']['width'] or not torch.isfinite(value).all():
            raise ValueError(f'Invalid native input encoding: {name}')
        energy = float(value.square().sum())
        square += energy
        count += value.numel()
        zeros += int(energy == 0)
    scale = (square/count)**.5
    if scale <= 0:
        raise ValueError('Degenerate training native-source scale')
    output = Path(cfg['paths']['interface'])
    atomic_torch(dict(mean=moments['mean'].float(), scale=moments['scale'].float(),
                      native_scale=torch.tensor(scale), revision=REVISION, task=cfg['task'],
                      training_samples=len(seq['ids'])), output)
    record = dict(revision=REVISION, task=cfg['task'], samples=len(seq['ids']),
                  native_source_rms=scale, zero_source_arms=zeros,
                  qwen_statistics_reference=str(reference),
                  source='matched training inputs only', benchmark_data_used=False,
                  learned_value_map=False, api_calls=0, model_forward_calls=0,
                  seconds=time.monotonic()-started)
    atomic_json(record, output.with_suffix('.json'))
    print(json.dumps(record))


if __name__ == '__main__':
    main()
