"""Same-case paired effects, including both means and threshold success."""
from pathlib import Path
import argparse
import json


def rows(path, ids):
    value = json.loads(Path(path).read_text())
    selected = {row['id']: row for row in value['cases'] if row['id'] in ids}
    if set(selected) != set(ids):
        raise ValueError(f'Incomplete paired case set: {path}')
    return selected


def aggregate(data):
    values = list(data.values())
    count = len(values)
    return dict(count=count, sa=sum(r['sa'] for r in values) / count,
                pc=sum(r['pc'] for r in values) / count,
                sa_pass_rate=sum(r['sa'] >= 4 for r in values) / count,
                pc_pass_rate=sum(r['pc'] >= 4 for r in values) / count,
                joint=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in values),
                joint_pass_rate=sum(r['sa'] >= 4 and r['pc'] >= 4 for r in values) / count)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads((args.root / 'plan.json').read_text())
    normal = rows(Path(plan['normal']) / 'scores.json', plan['cases'])
    result = dict(task=plan['task'], checkpoint=plan['checkpoint'], cases=plan['cases'],
                  normal=aggregate(normal), controls={},
                  interpretation='same-checkpoint inference interventions, not independently retrained architectures')
    for path in sorted(args.root.glob('controls/*/videophy40/scores.json')):
        control = rows(path, plan['cases'])
        values = aggregate(control)
        pairs = []
        for case in plan['cases']:
            pairs.append(dict(id=case, normal_sa=normal[case]['sa'], normal_pc=normal[case]['pc'],
                              control_sa=control[case]['sa'], control_pc=control[case]['pc']))
        differences = {metric: dict(
            normal_minus_control=result['normal'][metric] - values[metric],
            normal_wins=sum(normal[c][metric] > control[c][metric] for c in plan['cases']),
            ties=sum(normal[c][metric] == control[c][metric] for c in plan['cases']),
            normal_losses=sum(normal[c][metric] < control[c][metric] for c in plan['cases']))
            for metric in ('sa', 'pc')}
        result['controls'][path.parents[1].name] = dict(**values, differences=differences,
                                                        paired_cases=pairs, source=str(path))
    output = args.root / 'paired_summary.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(dict(normal=result['normal'], controls={k: {a: b for a, b in v.items()
                      if a not in ('paired_cases', 'differences')} for k, v in result['controls'].items()})))


if __name__ == '__main__':
    main()
