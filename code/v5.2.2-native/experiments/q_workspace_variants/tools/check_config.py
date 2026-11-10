"""Read-only resource and revision check; never allocates a GPU or submits jobs."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from physgen_native.config import read_config, REVISIONS
from physgen_native.data import audit
from physgen_native.q_variants import Q_WORKSPACE_VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--initialize-generator', type=Path, required=True)
    args = parser.parse_args()
    cfg = read_config(args.config)
    if cfg['revision'] not in Q_WORKSPACE_VARIANTS:
        raise ValueError('Select one of the five isolated Q recipes')
    if cfg['stages'] != {'interface': 64, 'joint': 160} or cfg['train']['accumulation'] != 4:
        raise ValueError('Production recipes retain 64+160 steps and accumulation 4')
    if (cfg['model']['qwen_width'], cfg['message']['workspace_width'], cfg['model']['width']) != (4096, 4096, 3072):
        raise ValueError('Production workspace/interface widths are 4096/3072')
    report = audit(cfg)
    if not args.initialize_generator.is_file():
        raise FileNotFoundError(args.initialize_generator)
    state = torch.load(args.initialize_generator, map_location='cpu', weights_only=False, mmap=True)
    if state['revision'] not in REVISIONS or state['revision'] != state['config']['revision']:
        raise ValueError('Initializer revision is not supported')
    for key in ('wan_checkpoint', 'qwen_checkpoint'):
        if Path(state['config']['paths'][key]).resolve() != Path(cfg['paths'][key]).resolve():
            raise ValueError('Initializer uses a different base model')
    if state['config']['wan_lora'] != cfg['wan_lora']:
        raise ValueError('Initializer Wan LoRA dimensions differ')
    report.update(revision=cfg['revision'], variant=cfg['message']['workspace_variant'],
                  stages=cfg['stages'], seed=cfg['train']['seed'], rounds='R0 + R1 + R2',
                  resource_pool=cfg['hardware']['resource_pool'],
                  initializer=str(args.initialize_generator), initializer_revision=state['revision'],
                  run_root=cfg['paths']['runs'], output_root=cfg['paths']['outputs'],
                  resources_modified=False, gpu_allocated=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['ready']:
        raise RuntimeError('Existing resources are incomplete; no downloads or cache regeneration')


if __name__ == '__main__':
    main()
