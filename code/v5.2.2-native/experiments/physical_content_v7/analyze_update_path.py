"""Locate common Q updates in the actual saved reader, exchange and FFN paths.

Forward observations only; no intervention, fitting or generation-quality claim.
"""
from pathlib import Path
import argparse
import json
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parents[1]


def cosine(value):
    value = F.normalize(value.float().reshape(-1, value.shape[-1]), dim=-1)
    n = len(value)
    return float((value.sum(0).square().sum()-value.square().sum())/(n*(n-1))) if n > 1 else None


def statistics(value, groups):
    value = value.detach().float()[0]
    total = value.square().sum().clamp_min(1e-20)
    grouped = [value[groups == group] for group in groups.unique()]
    return dict(rms=float(value.square().mean().sqrt()), token_cosine=cosine(value),
        centered_energy_fraction=float((value-value.mean(0)).square().sum()/total),
        within_relation_cosine=sum(cosine(v) for v in grouped)/len(grouped),
        between_relation_means_cosine=cosine(torch.stack([v.mean(0) for v in grouped])))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    sys.path[:0] = [cfg['source_snapshot'],
        str(BASE/'experiments/physical_scaling_q256/snapshots/20261029T140406Z'),
        str(BASE/'experiments/physical_variants/snapshots/20261029T064043Z')]
    import bootstrap
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    model = Enhancement(cfg).eval()
    state = {key.removeprefix('enhancement.'): value
        for section in ['model', 'constants'] for key, value in saved[section].items()
        if key.startswith('enhancement.')}
    model.load_state_dict(state)
    teacher = Teacher(cfg, device='cpu')
    active = {}
    records = []
    samples = {}

    def record(name, value):
        active['stages'][name] = statistics(value, active['groups'])
        samples.setdefault((active['round'], name), []).append(value[0].float().mean(0))

    def enter(module, inputs):
        active.update(anchor=inputs[1], groups=inputs[4], stages={})
        active['round'] += 1
        record('input_update', inputs[0]-inputs[1])

    def read(module, inputs, output):
        record('memory_read_write', output)
        heads = getattr(model.block, 'grounded_heads', 0)
        if heads:
            width = heads * model.block.cross.head_dim
            local = F.linear(inputs[0][..., :width], module.weight[:, :width])
            contextual = F.linear(inputs[0][..., width:], module.weight[:, width:])
            record('grounded_read_write', local)
            record('context_read_write', contextual)

    def leave(module, inputs, output):
        record('final_update', output-active['anchor'])
        records.append(dict(case=active['case'], round=active['round'], stages=active['stages']))

    model.block.register_forward_pre_hook(enter)
    model.block.cross.o.register_forward_hook(read)
    model.block.exchange_norm.register_forward_pre_hook(
        lambda module, inputs: record('after_read_update', inputs[0]-active['anchor']))
    model.block.exchange.o.register_forward_hook(
        lambda module, inputs, output: record('exchange_write', output))
    model.block.ffn_norm.register_forward_pre_hook(
        lambda module, inputs: record('after_exchange_update', inputs[0]-active['anchor']))
    model.block.ffn.register_forward_hook(
        lambda module, inputs, output: record('ffn_write', output))
    model.block.update_norm.register_forward_pre_hook(
        lambda module, inputs: record('before_update_norm', inputs[0]))
    model.block.register_forward_hook(leave)
    cases = json.loads((ROOT/'early_execution_r3/subset16.json').read_text())['cases']
    for index, case in enumerate(cases):
        active.update(case=case['id'], round=0)
        evidence = teacher.input(dict(id='videophy40_'+case['id']))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            model(evidence)
        print(json.dumps(dict(case=case['id'], completed=index+1, total=len(cases))), flush=True)
    summary = {}
    for round_id in range(1, model.rounds+1):
        rows = [row for row in records if row['round'] == round_id]
        summary[round_id] = {}
        for name in rows[0]['stages']:
            fields = rows[0]['stages'][name]
            summary[round_id][name] = {field:sum(row['stages'][name][field] for row in rows)/len(rows)
                for field in fields}
            summary[round_id][name]['cross_sample_means_cosine'] = cosine(torch.stack(samples[(round_id, name)]))
    result = dict(checkpoint=str(args.checkpoint), step=saved['step'], revision=cfg['revision'],
        source_snapshot=cfg['source_snapshot'], cases=[case['id'] for case in cases],
        summary=summary, records=records,
        interpretation='Actual saved forward-path measurements. Stage statistics locate shared directions but do not prove causal generation harm, the benefit of removing self-attention, or useful physics. No intervention is performed.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), summary=summary), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
