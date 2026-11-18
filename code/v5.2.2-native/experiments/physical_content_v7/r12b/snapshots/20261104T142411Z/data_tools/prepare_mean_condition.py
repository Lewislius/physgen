"""Train-only mean C from original inputs, preserving per-case group priors."""
from pathlib import Path
import argparse
import json
import os
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import torch
from condition_controls import MeanContent


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=64)
    p.add_argument('--seed', type=int, default=463)
    args = p.parse_args()
    expected = dict(checkpoint=str(args.checkpoint.resolve()), count=args.count,
        selection_seed=args.seed, split='train', kind='fixed_mean_c_own_relation_prior')
    if args.output.exists():
        saved = torch.load(args.output, map_location='cpu', weights_only=True)
        if any(saved.get(k) != v for k, v in expected.items()):
            raise ValueError('Existing mean belongs to another checkpoint/control')
        print(json.dumps(dict(status='reused', **expected))); return
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    if not cfg.get('data_revision'):
        raise ValueError('Use this helper only for a new-data comparison checkpoint')
    sys.path.insert(0, cfg['source_snapshot'])
    import bootstrap
    sys.path.insert(0, str(HERE))
    from physical_scaling.common import gpu_guard, atomic_torch, atomic_json
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    gpu_guard(args.output.parent / 'allocations' / os.getenv('DET_EXPERIMENT_ID', 'local'))
    torch.set_num_threads(4)
    model = Enhancement(cfg).eval().requires_grad_(False)
    values = {k.removeprefix('enhancement.'): v
        for k, v in {**state['model'], **state['constants']}.items() if k.startswith('enhancement.')}
    model.load_state_dict(values, strict=True); model.cuda()
    teacher = Teacher(cfg)
    inventory = json.loads((teacher.cache / 'inventory.json').read_text())
    rows = inventory['records'] if isinstance(inventory, dict) else inventory
    selected = {}; means = {}; statistics = {}
    for arm, suffix in (('positive', '_text'), ('negative', '_null')):
        eligible = [r for r in rows if r['split'] == 'train' and r['name'].endswith(suffix)]
        selected[arm] = random.Random(args.seed).sample(eligible, args.count)
        accumulator = MeanContent()
        for row in selected[arm]:
            evidence = teacher.input(dict(id=row['sample']), drop=arm == 'negative')
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                content, groups = model(evidence)
            accumulator.add(content, groups)
        means[arm], statistics[arm] = accumulator.finish()
    metadata = dict(**expected, checkpoint_step=state['step'], revision=cfg['revision'],
        data_revision=cfg['data_revision'], training_ids={k: [r['sample'] for r in v] for k, v in selected.items()},
        definition='Fixed per-slot/channel training mean of each complete routing/execution C, separately for both CFG arms.',
        prior_policy='Retain each test input arm relation-count groups; only C tensors are fixed.',
        limitation='Input relation count remains. Mean C changes amplitude; this is not a content-only or no-Qwen-information intervention.',
        statistics=statistics)
    atomic_torch(dict(**metadata, **means), args.output)
    atomic_json(metadata, args.output.with_suffix('.json'))
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
