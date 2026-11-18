"""Checkpoint-specific fixed C, using training input evidence only."""
from collections import Counter
from pathlib import Path

import torch

from physical_content.common import atomic_torch
from physical_content.data import TrainingData, find_input
from physical_content.teacher import Teacher


@torch.no_grad()
def prepare_train_mean(model, cfg, checkpoint, destination):
    destination = Path(destination)
    provenance = dict(checkpoint=str(Path(checkpoint).resolve()), task=cfg['task'],
                      source='training input evidence only; no benchmark inputs or video targets')
    if destination.is_file():
        saved = torch.load(destination, map_location='cpu', weights_only=True)
        if saved['provenance'] != provenance:
            raise ValueError('Training mean belongs to another checkpoint/task')
        return {key: value.cuda() if torch.is_tensor(value) else value
                for key, value in saved.items()}
    data, teacher = TrainingData(cfg), Teacher(cfg)
    totals, counts, ids, unavailable = {}, Counter(), [], []
    for index, row in enumerate(data.rows):
        # Original I2V preparation encoded only the arm actually used in each
        # training microbatch. Average available positive training evidence;
        # never replace missing positives with image-only/null conditions.
        name = row['id'] + '_text'
        if (not (Path(cfg['paths']['words_cache']) / f'{name}.pt').is_file()
                or not (Path(cfg['paths']['native_words_cache']) / f'{name}.pt').is_file()):
            unavailable.append(row['id'])
            continue
        try:
            find_input(cfg, name)
        except FileNotFoundError:
            unavailable.append(row['id'])
            continue
        evidence = teacher.input(row)
        text = torch.load(Path(cfg['paths']['cache']) / 'base/text' / (row['id'] + '.pt'),
                          map_location='cpu', weights_only=True).float()[None].cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            content, groups = model.prepare_content(evidence, text)
        for key in ('routing', 'execution'):
            value = content[key].float()
            if key not in totals:
                totals[key] = torch.zeros_like(value)
            totals[key].add_(value)
        counts[int(groups.max()) + 1] += 1
        ids.append(row['id'])
        if (index + 1) % 200 == 0:
            print(f'train_mean {index + 1}/{len(data)}', flush=True)
    if not ids:
        raise ValueError('No positive training evidence for mean C')
    result = {key: value.div_(len(ids)).cpu() for key, value in totals.items()}
    slots = result['routing'].shape[1]
    # Position-wise mean remains identical for every benchmark input. Use the
    # modal training group layout, recorded explicitly rather than borrowed
    # from whichever benchmark example happens to be first.
    group_count = counts.most_common(1)[0][0]
    result.update(groups=torch.arange(slots) * group_count // slots,
                  provenance=provenance, sample_ids=ids, count=len(ids),
                  missing_positive_training_ids=unavailable,
                  group_count_histogram={str(k): v for k, v in counts.items()},
                  grouping='fixed modal training layout; position-wise average')
    atomic_torch(result, destination)
    return {key: value.cuda() if torch.is_tensor(value) else value
            for key, value in result.items()}
