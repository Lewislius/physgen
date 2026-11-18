"""One disposable full-size gradient/memory check before the fresh400 process."""
from pathlib import Path
import argparse
import json
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from physical_content.common import load_config, gpu_guard, atomic_json
from physical_content.model import PhysicalModel
from physical_content.data import TrainingData
from physical_content.teacher import Teacher
from physical_content.objectives import backward_micro


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    gpu_guard(args.output.parent)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.monotonic()
    model = PhysicalModel(cfg)
    model.set_phase(0)
    assert not any(p.requires_grad for p in model.wan.parameters())
    # Exercise the most expensive, joint phase; none of these weights is saved.
    model.set_phase(cfg['train']['interface_updates'])
    opt = torch.optim.AdamW(model.optim_groups(), betas=(.9, .999), eps=1e-8, foreach=False)
    sequence = json.loads(Path(cfg['train']['sample_sequence']).read_text())
    # A nonempty positive condition exercises every reader. Empty inputs are
    # valid training cases but cannot establish key/query gradient coverage.
    sample = TrainingData(cfg).get(sequence['drop'].index(False))
    evidence = Teacher(cfg).input(sample)
    sigma, noise = torch.tensor(.5, device='cuda'), torch.randn_like(sample['latent'])
    required = ['enhancement.seed', 'enhancement.block.cross.q.weight',
                'enhancement.route_update.weight', 'enhancement.execute_update.weight',
                'physical_blocks.0.conditioning.direct.query_base.weight',
                'physical_blocks.0.conditioning.direct.key.weight',
                'physical_blocks.0.conditioning.direct.value.weight',
                'physical_blocks.0.conditioning.direct_output.weight',
                'physical_blocks.0.native.self_attn.q.weight',
                'physical_blocks.4.native.ffn.0.weight', 'physical_blocks.4.output.weight',
                'noise_modulation.project.1.weight']
    if cfg['task']=='i2v':
        required.append('physical_blocks.0.conditioning.associated_output.weight')
    named = dict(model.named_parameters())
    torch.cuda.reset_peak_memory_stats()
    records, reached = [], set()
    for index in range(4):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            # No route auxiliary in this check: measure real generation gradients.
            result = backward_micro(model, evidence, sample, sample['text'], sigma, noise, 0., 1)
        gradients = {name: (float(named[name].grad.norm()) if named[name].grad is not None else 0.)
                     for name in required}
        reached.update(name for name, norm in gradients.items() if norm > 0)
        norms = {}
        for role in cfg['train']['peak_lrs']:
            params = [p for group in opt.param_groups if group['role'] == role
                      for p in group['params'] if p.grad is not None]
            norms[role] = float(torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True))
        opt.step()
        record = dict(disposable_update=index+1, gradients=gradients, roles=norms,
                      generation=result['generation'],
                      peak_gib=torch.cuda.max_memory_allocated()/2**30)
        records.append(record)
        print(json.dumps(record), flush=True)
    if set(required)-reached:
        raise RuntimeError(f'Missing generation gradients: {sorted(set(required)-reached)}')
    model.base_bank.assert_frozen()
    atomic_json(dict(passed=True, records=records, seconds=time.monotonic()-started,
                     parameters=sum(model.trainable_count().values()), checkpoint_saved=False,
                     quality_improvement_proven=False,
                     formal_initialization='separate official-fresh process'), args.output)


if __name__ == '__main__':
    main()
