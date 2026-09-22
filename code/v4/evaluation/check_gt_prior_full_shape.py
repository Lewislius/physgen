"""CPU check with a real cached WISA sample and the default full-size P0 network."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physgen_v4.environments import require_environment
require_environment('runtime')

import torch
from physgen_v4.corrector import ProcessInitializer
from physgen_v4.data import CachedWISA
from physgen_v4.losses import prior_distance
from physgen_v4.runtime import read_config
from train.train_native_p import geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/wisa_native_p.yaml'))
    parser.add_argument('--output', default=str(ROOT / 'evaluation/results/gt_prior_full_shape.json'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(9312)
    cfg = read_config(args.config)
    if cfg['corrector']['initialization'] != 'image_text_gt':
        raise ValueError('This check requires initialization=image_text_gt')
    sample = CachedWISA(cfg['paths']['cache_root'], 'train', limit=1)[0]
    coords = geometry(sample, cfg)
    prior = ProcessInitializer(cfg['corrector']).train()
    report = dict(device='cpu', torch_version=torch.__version__, cache_index=sample['record']['index'],
                  clean_shape=list(sample['latent'].shape), text_shape=list(sample['text'].shape),
                  target_shape=list(sample['target'].shape), prior_width=cfg['corrector']['prior_width'],
                  initializer_parameters=sum(p.numel() for p in prior.parameters()), conditions=[],
                  limits='Real cached inputs and default P0 only; no Wan forward, CUDA training or generation quality test.')
    outputs = []
    for use_gt in (True, False):
        started = time.perf_counter()
        prior.zero_grad(set_to_none=True)
        with torch.autocast('cpu', dtype=torch.bfloat16, cache_enabled=False):
            predicted = prior.initialize(sample['first'], prior.text(sample['text'].unsqueeze(0)), coords,
                                         True, gt_video=sample['latent'] if use_gt else None)
            loss = prior_distance(predicted, sample['target'], sample['target_weight'], len(coords.process[0]))
        if predicted.shape != sample['target'].shape or not bool(torch.isfinite(predicted).all()):
            raise AssertionError('P0 shape/finite check failed')
        loss.backward()
        active, inactive = 0, 0
        for name, parameter in prior.named_parameters():
            optional = name.startswith(('initialize.video.', 'initialize.video_norm.', 'initialize.video_modality'))
            if optional and not use_gt:
                if parameter.grad is not None:
                    raise AssertionError(f'Unused GT layer has a gradient: {name}')
                inactive += 1
            else:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise AssertionError(f'Missing/nonfinite gradient: {name}')
                active += 1
        report['conditions'].append(dict(gt_conditioned=use_gt, prior_loss=float(loss.detach()),
            p0_shape=list(predicted.shape), p0_dtype=str(predicted.dtype), active_gradient_tensors=active,
            inactive_gt_tensors=inactive, image_grad_norm=float(prior.initialize.image.weight.grad.norm()),
            text_grad_norm=float(prior.text.weight.grad.norm()),
            gt_grad_norm=float(prior.initialize.video.weight.grad.norm()) if use_gt else None,
            seconds=time.perf_counter() - started))
        outputs.append(predicted.detach())
        print(json.dumps(report['conditions'][-1]), flush=True)
        del predicted, loss
    report['gt_vs_no_gt_p0_rms'] = float((outputs[0] - outputs[1]).square().mean().sqrt())
    report['passed'] = report['gt_vs_no_gt_p0_rms'] > 0
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    if not report['passed']:
        raise AssertionError('GT context did not affect P0')
    print(str(destination), flush=True)


if __name__ == '__main__':
    main()
