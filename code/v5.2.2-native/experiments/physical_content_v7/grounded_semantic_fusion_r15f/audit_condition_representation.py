"""CPU-only inspection of Q updates, native anchors and shared C on the fixed16.

Loads the frozen enhancement checkpoint only; no Wan/Qwen model, generation,
training, API call or download. Representation sensitivity is not quality proof.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F


def cosine_rows(value):
    value = value.detach().float().reshape(-1, value.shape[-1])
    n = len(value)
    if n < 2:
        return None
    unit = F.normalize(value, dim=-1)
    return float((unit.sum(0).square().sum() - unit.square().sum()) / (n * (n - 1)))


def rms(value):
    return float(value.float().square().mean().sqrt())


def describe(value, groups):
    value = value.detach().float()[0]
    relation_means = torch.stack([value[groups == g].mean(0) for g in groups.unique()])
    return dict(token_cosine=cosine_rows(value), relation_mean_cosine=cosine_rows(relation_means),
                rms=rms(value), centered_energy=float((value-value.mean(0)).square().mean()
                / value.square().mean().clamp_min(1e-12)))


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    if state['step'] != 400 or cfg['revision'] != 'physical_content_v7_grounded_semantic_fusion_r15f':
        raise ValueError('Use the frozen R15F own400 checkpoint')
    sys.path.insert(0, cfg['source_snapshot'])
    import bootstrap
    from physical_content.model import Enhancement
    from condition_controls import InferenceCache
    model = Enhancement(cfg).eval().requires_grad_(False)
    values = {k.removeprefix('enhancement.'): v for k,v in
              {**state['model'], **state['constants']}.items() if k.startswith('enhancement.')}
    model.load_state_dict(values, strict=True)
    del values, state
    reader = InferenceCache(cfg['paths']['inference_evidence'], device='cpu')
    ids = [v['id'] for v in json.loads((Path(cfg['source_snapshot'])/'subset16.json').read_text())['cases']]
    captured = {}
    model.project.register_forward_pre_hook(lambda module, inputs: captured.update(q_update=inputs[0].detach()))
    model.project.register_forward_hook(lambda module, inputs, output: captured.update(projected_update=output.detach()))
    report = dict(utc=datetime.now(timezone.utc).isoformat(), checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_step=400, training_source=cfg['source_snapshot'], cases=[],
                  modes=['normal','words_only_memory','reasoning_swapped','empty_reasoning','negative'],
                  scope='CPU float32 enhancement-only diagnostics on the fixed16. No full-Wan execution or quality claim. '
                        'Whole C includes a frozen native word anchor, so its diversity cannot establish learned-update diversity. '
                        'Wrong/empty preserve own image/request; words-only still contains contextualized Qwen words. '
                        'Negative-arm slotwise differences can include different relation-group layouts.',
                  quality_proven=False)
    sample_means = {mode: {field: [] for field in ['native_anchor','q_update','projected_update','shared_C']}
                    for mode in report['modes']}
    start = time.monotonic()
    for index, case in enumerate(ids):
        ready = json.loads((reader.root/f'videophy40_{case}'/'ready.json').read_text())
        sample = dict(id=ready['id'], caption=ready['caption'])
        row = dict(id=case, modes={})
        normal = None
        for mode in report['modes']:
            evidence = reader.input(sample, image=ready['image'], duration=ready['duration'],
                                    drop=mode=='negative', mode=mode)
            anchor, groups = model(evidence, anchor_only=True)
            native = anchor['routing']
            content, actual_groups = model(evidence, words_only_memory=mode=='words_only_memory')
            assert torch.equal(groups, actual_groups)
            assert content['routing'] is content['execution']
            shared = content['routing']
            tensors = dict(native_anchor=native, q_update=captured['q_update'],
                           projected_update=captured['projected_update'], shared_C=shared)
            stats = {key: describe(value, groups) for key,value in tensors.items()}
            stats['projected_update_relative_anchor_rms'] = rms(captured['projected_update']) / max(rms(native),1e-12)
            if normal is None:
                normal = shared.clone()
            stats['shared_C_relative_change_from_normal'] = rms(shared-normal) / max(rms(normal),1e-12)
            row['modes'][mode] = stats
            for field,value in tensors.items():
                sample_means[mode][field].append(value[0].mean(0))
        report['cases'].append(row)
        if (index+1)%4 == 0:
            print(json.dumps(dict(completed_cases=index+1,total=len(ids),seconds=time.monotonic()-start)),flush=True)
    report['summary'] = {}
    for mode in report['modes']:
        entries = [row['modes'][mode] for row in report['cases']]
        aggregate = {}
        for field, means in sample_means[mode].items():
            aggregate[field] = {}
            for metric in ['token_cosine','relation_mean_cosine','rms','centered_energy']:
                observed = [x[field][metric] for x in entries if x[field][metric] is not None]
                aggregate[field][metric] = statistics.mean(observed) if observed else None
            aggregate[field]['different_sample_mean_cosine'] = cosine_rows(torch.stack(means))
        for metric in ['projected_update_relative_anchor_rms','shared_C_relative_change_from_normal']:
            aggregate[metric] = statistics.mean(x[metric] for x in entries)
        report['summary'][mode] = aggregate
    report['elapsed_seconds'] = time.monotonic()-start
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report=str(args.output), summary=report['summary']),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
