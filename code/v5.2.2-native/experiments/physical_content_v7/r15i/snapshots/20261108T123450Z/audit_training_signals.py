"""Describe matched training signals; do not infer generation quality from FM.

Reads existing logs only. No training, inference, API requests or GPU work.
"""
import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / 'request_bound_fusion_r15b/runs/physical_content_67829'


def read_rows(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # An actively written final line may be incomplete.
            continue
    return rows


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return dict(n=len(values), mean=mean(values), median=median(values),
                p90=values[math.ceil(.9 * len(values)) - 1],
                maximum=values[-1], minimum=values[0])


def summarize(rows, tail_weight):
    if not rows:
        return None
    components = dict(mean_fm=mean((1-tail_weight)*r['fm'] for r in rows),
                      high_error_tail=mean(tail_weight*r['tail'] for r in rows),
                      routing_alignment=mean(r['weighted_aux'] for r in rows))
    total = sum(components.values())
    return dict(
        microbatches=len(rows),
        unweighted={k: mean(r[k] for r in rows) for k in ('fm', 'tail', 'aux')},
        weighted_component_values=components,
        component_value_fractions={k: v/total for k, v in components.items()},
        auxiliary_over_generation=distribution([
            r['weighted_aux']/max(r['generation'], 1e-20) for r in rows]),
        tail_over_mean_fm=distribution([r['tail']/max(r['fm'], 1e-20) for r in rows]),
        generation_formula_max_error=max(abs(r['generation'] -
            ((1-tail_weight)*r['fm']+tail_weight*r['tail'])) for r in rows),
        auxiliary_formula_max_error=max(abs(r['weighted_aux']-
            r['aux_coefficient']*r['aux']) for r in rows),
        execution_C_cosine=distribution([r['c_pair_cosine'] for r in rows]),
        value_fraction_is_not_gradient_fraction=True,
    )


def gradient_summary(updates):
    result = {}
    for name in ('enhancement', 'fusion_read', 'fusion_write', 'wan_lora'):
        norms = [r['gradients'][name] for r in updates]
        scales = [r['clip_scales'][name] for r in updates]
        result[name] = dict(preclip_norm=distribution(norms),
                            zero_norm_updates=sum(v == 0 for v in norms),
                            clipped_updates=sum(v < .999999 for v in scales),
                            minimum_clip_scale=min(scales) if scales else None)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--through-step', type=int)
    args = parser.parse_args()
    paths = {'H': ROOT/'runs'/args.run_id, 'B': REFERENCE}
    records = {name: read_rows(path/'train.jsonl') for name, path in paths.items()}
    config = {name: json.loads((path/'config.json').read_text()) for name, path in paths.items()}
    updates = {name: [r for r in rows if r.get('kind') == 'train']
               for name, rows in records.items()}
    end = min(max(r['step'] for r in rows) for rows in updates.values())
    if args.through_step:
        end = min(end, args.through_step)
    micros = {name: {(r['step'], r['micro']): r for r in rows
                     if r.get('kind') == 'micro' and r['step'] <= end}
              for name, rows in records.items()}
    keys = sorted(set(micros['H']) & set(micros['B']))
    match_fields = ('phase', 'sample', 'sigma', 'text_dropped', 'aux_coefficient')
    mismatches = [dict(step=k[0], micro=k[1], field=f,
                       H=micros['H'][k].get(f), B=micros['B'][k].get(f))
                  for k in keys for f in match_fields
                  if micros['H'][k].get(f) != micros['B'][k].get(f)]
    h_updates = {r['step']: r for r in updates['H'] if r['step'] <= end}
    b_updates = {r['step']: r for r in updates['B'] if r['step'] <= end}
    schedule_differences = [dict(step=s, field=f)
                           for s in sorted(set(h_updates) & set(b_updates))
                           for f in ('phase', 'cursor', 'learning_rates', 'aux_coefficient')
                           if h_updates[s].get(f) != b_updates[s].get(f)]
    accumulation = config['H']['train']['accumulation']
    complete_match = (not mismatches and not schedule_differences
                      and len(keys) == end*accumulation
                      and set(micros['H']) == set(micros['B']))
    report = dict(
        utc=datetime.now(timezone.utc).isoformat(), through_step=end,
        sources={k: str(v/'train.jsonl') for k, v in paths.items()},
        complete_matched_prefix=complete_match, paired_microbatches=len(keys),
        logged_input_mismatches=mismatches, schedule_differences=schedule_differences,
        noise_evidence=dict(
            seed={k: v['train']['seed'] for k, v in config.items()},
            method='Both trainers seed a separate CUDA noise generator with seed*2000003+cursor; sample, sigma, dropout and cursor are checked from logs.',
            noise_tensor_hash_recorded=False,
            limitation='Logs do not independently prove byte-identical noise tensors across runtimes.'),
        windows={}, sigma_strata=[],
        interpretation=[
            'FM and tail are single-sample single-noise-level denoising losses, not PC/SA/Joint or rollout quality.',
            'Only the same completed sample/sigma/dropout schedule is paired; do not read unrelated successive steps as convergence.',
            'Tail selects high prediction-error regions at four scales, not high motion or specifically physical errors.',
            'Scalar objective shares do not measure per-loss gradient shares or gradient agreement.',
            'Group gradients combine objectives and parameters; nonzero norms show a training signal, not useful conditioning.',
            'Routing alignment has no direct execution-projector term but can affect execution through shared Q/refinement parameters.',
            'Current controls test trained modules/content; they are not separately retrained loss ablations.',
            'Full40, paired16 and actual videos decide benefit; no loss statistic is a quality pass criterion.',
        ])
    interface_end = config['H']['stages']['interface']
    windows = {'interface': (1, min(end, interface_end)),
               'last32_interface': (max(1, interface_end-31), min(end, interface_end))}
    if end > interface_end:
        windows['joint'] = (interface_end+1, end)
    for label, (start, stop) in windows.items():
        if start > stop:
            continue
        selected = [k for k in keys if start <= k[0] <= stop]
        report['windows'][label] = dict(steps=[start, stop], models={})
        for name in paths:
            loss = summarize([micros[name][k] for k in selected],
                             config[name]['loss']['generation']['tail_weight'])
            loss['gradients'] = gradient_summary([
                r for r in updates[name] if start <= r['step'] <= stop])
            report['windows'][label]['models'][name] = loss
        if complete_match:
            report['windows'][label]['paired_H_minus_B'] = {
                f: distribution([micros['H'][k][f]-micros['B'][k][f] for k in selected])
                for f in ('fm', 'tail', 'weighted_aux')}
    if complete_match:
        for phase in ('interface', 'joint'):
            for dropped in (False, True):
                for lo, hi in ((0, .25), (.25, .75), (.75, 1.01)):
                    selected = [k for k in keys if micros['H'][k]['phase'] == phase
                                and micros['H'][k]['text_dropped'] == dropped
                                and lo <= micros['H'][k]['sigma'] < hi]
                    if selected:
                        report['sigma_strata'].append(dict(
                            phase=phase, text_dropped=dropped, sigma=[lo, hi],
                            models={name: summarize([micros[name][k] for k in selected],
                                config[name]['loss']['generation']['tail_weight'])
                                    for name in paths}))
    output = ROOT/'reports'/f'training_signals_step{end:04d}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(report=str(output), through_step=end,
                         paired_microbatches=len(keys), complete_match=complete_match)))


if __name__ == '__main__':
    main()
