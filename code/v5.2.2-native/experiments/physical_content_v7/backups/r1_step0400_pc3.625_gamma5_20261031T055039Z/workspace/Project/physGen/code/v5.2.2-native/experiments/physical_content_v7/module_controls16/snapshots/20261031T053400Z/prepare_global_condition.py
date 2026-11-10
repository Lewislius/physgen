"""Freeze a checkpoint-specific, sample-independent C from training inputs only."""
from pathlib import Path
import argparse
import json
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--seed', type=int, default=463)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Choose a new output; never replace an existing condition control')
    torch.set_num_threads(16)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    if cfg['variant'] != 'physical_content':
        raise ValueError('A physical-content checkpoint is required')
    sys.path.insert(0, cfg['source_snapshot'])
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    from physical_content.evidence import find_input
    model = Enhancement(cfg).eval().requires_grad_(False)
    values = {name.removeprefix('enhancement.'): tensor
              for name, tensor in {**state['model'], **state['constants']}.items()
              if name.startswith('enhancement.')}
    model.load_state_dict(values, strict=True)
    teacher = Teacher(cfg, device='cpu')
    manifest = Path(cfg['paths']['cache']) / 'base/manifest.json'
    train = [r for r in json.loads(manifest.read_text())['records'] if r['split'] == 'train']
    eligible = {}
    selected = {}
    for arm, suffix in [('positive', '_text'), ('negative', '_null')]:
        eligible[arm] = []
        for row in train:
            name = row['id'] + suffix
            try:
                find_input(name)
                if not (teacher.cache / 'words' / (name + '.pt')).is_file():
                    continue
            except FileNotFoundError:
                continue
            eligible[arm].append(row)
        selected[arm] = random.Random(args.seed).sample(eligible[arm], args.count)
    sums, squared_norms, group_ref = {}, {}, None
    for i in range(args.count):
        for name, drop in [('positive', False), ('negative', True)]:
            row = selected[name][i]
            # Teacher.input reads only the cached first-image/request evidence.
            # No real future frames, latent targets or benchmark data are loaded.
            evidence = teacher.input({'id': row['id']}, drop=drop)
            c, groups = model(evidence)
            if group_ref is None:
                group_ref = groups.clone()
            if not torch.equal(groups, group_ref):
                raise ValueError('Training inputs have incompatible C relation layouts')
            sums[name] = sums.get(name, torch.zeros_like(c)) + c
            squared_norms[name] = squared_norms.get(name, 0.) + float(c.square().mean())
        if (i + 1) % 8 == 0:
            print(json.dumps(dict(completed=i + 1, total=args.count)), flush=True)
    means = {key: value / args.count for key, value in sums.items()}
    metadata = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_step=state['step'],
        revision=cfg['revision'], training_ids={k:[r['id'] for r in v] for k,v in selected.items()}, selection_seed=args.seed,
        source_manifest=str(manifest), split='train', count=args.count,
        total_train_records=len(train), eligible_cached_records={k:len(v) for k,v in eligible.items()},
        selection='Each branch independently samples64 existing training-input caches using the same fixed seed; availability only, no outcome-based selection. Branches need not share sample IDs.',
        definition='Unscaled elementwise mean of C at each slot; both CFG conditions fixed across test samples',
        inference_inputs='No sample-specific Qwen evidence; native Wan first image and T5 request remain',
        rms={key: dict(mean_condition=float(value.square().mean().sqrt()),
                       per_sample_rms_root_mean_square=(squared_norms[key]/args.count)**.5)
             for key, value in means.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(**metadata, **means, groups=group_ref), args.output)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(metadata, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
