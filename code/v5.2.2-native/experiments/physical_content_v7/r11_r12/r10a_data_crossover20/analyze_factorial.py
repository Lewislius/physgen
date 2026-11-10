"""Separate inference evidence, training bundle, and their interaction.

Evaluation-only: consumes the four already declared arms, never alters their
videos or scores. The fixed16 and score-selected diagnostic4 stay separate.
"""
from pathlib import Path
import json
import statistics
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import WORK, OLD_OUTPUT, API_OUTPUT, ARMS, EXTRA, cases


def metrics(record):
    return dict(sa=record['sa'], pc=record['pc'],
                sa_pass=int(record['sa'] >= 4), pc_pass=int(record['pc'] >= 4),
                joint=int(record['sa'] >= 4 and record['pc'] >= 4))


def contrast(rows, coefficients, ids):
    result = {}
    for metric in ('sa', 'pc', 'sa_pass', 'pc_pass', 'joint'):
        differences = {case: sum(weight * metrics(rows[arm][case])[metric]
                                 for arm, weight in coefficients.items()) for case in sorted(ids)}
        values = list(differences.values())
        result[metric] = dict(mean_change=statistics.mean(values),
            positive=sum(value > 0 for value in values), unchanged=sum(value == 0 for value in values),
            negative=sum(value < 0 for value in values), per_case=differences)
    return result


def main():
    _, selected = cases()
    ids = {row['id'] for row in selected}
    sources = {'old_checkpoint_old_reasoning': OLD_OUTPUT / 'scores.json',
               'api_checkpoint_api_reasoning': API_OUTPUT / 'scores.json'}
    sources.update({name: WORK / 'outputs' / name / 'videophy40/scores.json' for name in ARMS})
    rows = {}
    pending = []
    for name, path in sources.items():
        if not path.is_file():
            pending.append(name)
            continue
        scored = json.loads(path.read_text())['cases']
        values = [v for v in scored if v['id'] in ids]
        if len(values) != 20 or {v['id'] for v in values} != ids:
            raise ValueError('Each complete arm must cover the same20 unique cases')
        rows[name] = {v['id']: v for v in values}
    old_old, old_api = 'old_checkpoint_old_reasoning', 'old_checkpoint_api_reasoning'
    api_old, api_api = 'api_checkpoint_old_reasoning', 'api_checkpoint_api_reasoning'
    specifications = {
        'observed_diagonal_change': {api_api: 1, old_old: -1},
        'inference_change_given_old_checkpoint': {old_api: 1, old_old: -1},
        'inference_change_given_api_checkpoint': {api_api: 1, api_old: -1},
        'training_bundle_change_given_old_inference': {api_old: 1, old_old: -1},
        'training_bundle_change_given_api_inference': {api_api: 1, old_api: -1},
        'interaction': {api_api: 1, api_old: -1, old_api: -1, old_old: 1},
    }
    strata = {'original_fixed16': ids - set(EXTRA), 'diagnostic4': set(EXTRA)}
    report = dict(utc=datetime.now(timezone.utc).isoformat(), sources={k: str(v) for k, v in sources.items()},
                  pending=pending, complete=not pending, strata={})
    for label, keep in strata.items():
        comparisons = {name: contrast(rows, weights, keep) for name, weights in specifications.items()
                       if set(weights) <= set(rows)}
        report['strata'][label] = dict(count=len(keep), ids=sorted(keep),
            arm_means={arm: {metric: statistics.mean(metrics(values[case])[metric] for case in keep)
                            for metric in metrics(next(iter(values.values())))} for arm, values in rows.items()},
            contrasts=comparisons)
        if not pending:
            # The diagonal difference has two valid decompositions. Report both
            # conditional effects; never assign the complete difference to one factor.
            for metric in ('sa', 'pc', 'sa_pass', 'pc_pass', 'joint'):
                value = lambda key: comparisons[key][metric]['mean_change']
                observed = value('observed_diagonal_change')
                one = value('training_bundle_change_given_old_inference') + value('inference_change_given_api_checkpoint')
                two = value('inference_change_given_old_checkpoint') + value('training_bundle_change_given_api_inference')
                if abs(observed - one) > 1e-10 or abs(observed - two) > 1e-10:
                    raise ValueError('Factorial decomposition did not preserve paired diagonal differences')
    report['interpretation'] = {
        'training_factor': 'Old versus API400 checkpoint. This jointly changes training captions/native T5, physical evidence and fitted interface; it cannot isolate caption accuracy or any one of these changes.',
        'inference_factor': 'Both positive and negative input-only Qwen reasoning/encoded word features change together. Literal benchmark prompt, native T5, first image, persisted noise and gamma5 remain fixed. This is not a positive-reasoning-only ablation.',
        'interaction': 'Difference between the inference-change effects under the two checkpoints. Nonzero values indicate checkpoint-dependent responses, not proof of a particular calibration mechanism.',
        'direction': 'All differences are API minus original. SA/PC are rating-point changes; pass/Joint are absolute rate changes. Positive is numerically higher, not automatic visual-quality success.',
        'scope': 'Fixed16 is the original diagnostic subset, not the full40 estimate. Extra4 were selected after observing errors and cannot estimate an unbiased population gain. Single seed; never combine per-case best outputs.',
        'action_rule': 'Select a whole training/inference recipe using PC/SA/Joint together, then inspect paired process/visual quality. A same-checkpoint inference gain motivates a full40/60 confirmation; it is not itself that confirmation.'}
    destination = WORK / 'factorial_report.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(complete=report['complete'], pending=pending,
                          strata={k: dict(count=v['count'], arm_means=v['arm_means'],
                                          available_contrasts=list(v['contrasts']))
                                  for k, v in report['strata'].items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
