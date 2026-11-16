"""Validate the frozen Wan2.1-compatible input encodings; no new API calls."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
from benchmark_inputs import case_input, reference
from condition_controls import InferenceCache
from physical_content.common import atomic_json, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg['task'] != 't2v':
        raise ValueError('R20FT-Wan2.1 is text-to-video only')
    cache = InferenceCache(cfg['paths']['inference_evidence'], cfg['task'],
                           device='cpu', native_width=cfg['model']['width'])
    arms, words = 0, 0
    for case in reference()['cases']:
        _, item = case_input(case, cfg['task'])
        for drop in (False, True):
            evidence = cache.input(item, image=None, duration=item['duration'], drop=drop)
            if (evidence['native_condition'].shape[0] != evidence['words'].shape[0]
                    or evidence['native_condition'].shape[1] != cfg['model']['width']):
                raise ValueError(f'Physical source-word encodings disagree: {case["id"]}')
            if any(k in evidence for k in ('memory', 'scene_memory', 'first_rgb')):
                raise ValueError('Image memory cannot enter T2V')
            arms += 1
            words += evidence['words'].shape[0]
    result = dict(complete=True, task=cfg['task'], arms=arms, words=words,
                  control_arms=0, api_calls=0, model_forward_calls=0,
                  native_width=cfg['model']['width'],
                  source=cfg['paths']['inference_evidence'])
    atomic_json(result, Path(cfg['paths']['run_dir']).parent.parent/'reports'/f'{cfg["plan"]}_inputs.json')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
