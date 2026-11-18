"""Compare existing R1/final-candidate records at a completed common update budget.

This reads logs only. It loads no model and runs no new evaluation.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import json
import statistics

ROOT = Path(__file__).resolve().parent.parent


def records(path, stop):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    micro = [row for row in rows if row['kind'] == 'micro' and row['step'] <= stop]
    updates = [row for row in rows if row['kind'] == 'train' and row['step'] <= stop]
    if len(micro) != stop * 4 or [row['step'] for row in updates] != list(range(1, stop + 1)):
        raise ValueError(f'Incomplete or duplicated update budget: {path}')
    if [(row['step'], row['micro']) for row in micro] != [
            (step, item) for step in range(1, stop + 1) for item in range(1, 5)]:
        raise ValueError(f'Incomplete or duplicated accumulation records: {path}')
    return micro, updates


def average(rows, key):
    return statistics.mean(row[key] for row in rows) if rows else None


def summarize(micro, updates, start, end, arm):
    selected = [row for row in micro if start <= row['step'] <= end
                and (arm == 'all' or row['text_dropped'] == (arm == 'joint_null'))]
    result = dict(count=len(selected), unique_samples=len({row['sample'] for row in selected}),
                  means={key: average(selected, key) for key in [
                      'c_pair_cosine', 'c_centered_energy', 'aux', 'weighted_aux',
                      'fm', 'generation', 'tail']})
    if arm == 'all':
        steps = [row for row in updates if start <= row['step'] <= end]
        result['optimizer_groups'] = {
            group: dict(mean_gradient_norm=statistics.mean(row['gradients'][group] for row in steps),
                        clipped_update_count=sum(row['clip_scales'][group] < 1 for row in steps),
                        mean_clip_scale=statistics.mean(row['clip_scales'][group] for row in steps))
            for group in steps[0]['gradients']}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through', type=int, choices=[250, 400], required=True)
    parser.add_argument('--candidate', choices=['R10-A', 'R10-B'], default='R10-A')
    args = parser.parse_args()
    directory = {'R10-A': 'execution_value_r10',
                 'R10-B': 'grounded_execution_r10b'}[args.candidate]
    metadata = json.loads((ROOT / directory / 'prepared.json').read_text())
    run = metadata['successful_candidate_run']
    sources = dict(R1=ROOT / 'runs/physical_content_66778/train.jsonl',
                   **{args.candidate: ROOT / directory / 'runs' / run / 'train.jsonl'})
    loaded = {name: records(path, args.through) for name, path in sources.items()}
    match_fields = ['step', 'micro', 'sample', 'sigma', 'text_dropped', 'aux_coefficient']
    differences = []
    for before, after in zip(loaded['R1'][0], loaded[args.candidate][0]):
        changed = [key for key in match_fields if before[key] != after[key]]
        if changed:
            differences.append(dict(step=before['step'], micro=before['micro'], fields=changed))
    schedule_differences = [a['step'] for a, b in zip(loaded['R1'][1], loaded[args.candidate][1])
                            if a['learning_rates'] != b['learning_rates']]
    windows = dict(interface=(1, 128), joint=(129, args.through),
                   last50=(args.through - 49, args.through))
    summaries = {}
    flat = []
    for window, (start, end) in windows.items():
        summaries[window] = dict(first_step=start, last_step=end, arms={})
        for arm in ['all', 'both_present', 'joint_null']:
            values = {name: summarize(*items, start, end, arm) for name, items in loaded.items()}
            delta = {key: values[args.candidate]['means'][key] - values['R1']['means'][key]
                     for key in values['R1']['means']}
            summaries[window]['arms'][arm] = dict(**values, **{args.candidate + '_minus_R1': delta})
            for name, value in values.items():
                flat.append(dict(window=window, arm=arm, model=name, first_step=start,
                                 last_step=end, count=value['count'], **value['means']))
    report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), through_step=args.through,
                  candidate=args.candidate, candidate_run=run,
                  sources={key: str(value) for key, value in sources.items()},
                  matched_micro_fields=match_fields, micro_count=args.through * 4,
                  training_input_mismatches=differences, learning_rate_mismatches=schedule_differences,
                  windows=summaries,
                  interpretation=[
                      'C cosine/centered energy describe total execution C for the candidate and shared K/V C for R1, not isolated Q updates; they do not establish quality.',
                      ('Auxiliary measures routing-only C in R10-A versus shared K/V C in R1. Their numerical difference is not a matched execution-target error.'
                       if args.candidate == 'R10-A' else
                       'Auxiliary measures the shared execution core in R10-B versus shared K/V C in R1. The extra B routing correction is excluded from this auxiliary.'),
                      'Auxiliary is native-word interface reconstruction; it is not a physical correctness target.',
                      'FM and tail losses vary with video/noise/sigma and do not establish convergence or generation quality.',
                      'Group gradient norms combine objectives and parameters, and do not isolate useful reasoning credit.',
                      'Matching sample/sigma/dropout/schedule removes those recipe differences; video scores and information controls remain decisive.'
                  ], model_loaded=False, new_evaluation=False)
    output = ROOT / directory / 'reports' / f'training_comparison_through{args.through:04d}'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with output.with_suffix('.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps(dict(report=str(output.with_suffix('.json')),
                          training_input_mismatches=len(differences),
                          learning_rate_mismatches=schedule_differences,
                          last50=summaries['last50']), ensure_ascii=False))


if __name__ == '__main__':
    main()
