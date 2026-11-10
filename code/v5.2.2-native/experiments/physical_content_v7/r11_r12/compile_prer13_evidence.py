"""Recompute Joint bottlenecks and historical R9 controls from saved raw scores."""
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path(__file__).resolve().parent / 'reports'


def read(path):
    return json.loads(Path(path).read_text())


def category(row):
    return ('joint' if row['sa'] >= 4 else 'pc_only') if row['pc'] >= 4 else (
        'sa_only' if row['sa'] >= 4 else 'neither')


def summary(rows):
    groups = {name: [r['id'] for r in rows if category(r) == name]
              for name in ('joint', 'pc_only', 'sa_only', 'neither')}
    return dict(count=len(rows), sa=mean(r['sa'] for r in rows),
                pc=mean(r['pc'] for r in rows),
                counts={k: len(v) for k, v in groups.items()}, cases=groups)


def paired(after, before):
    a = {r['id']: r for r in after}
    b = {r['id']: r for r in before}
    if set(a) != set(b):
        raise ValueError('Paired case IDs differ')
    result = {}
    for key in ('sa', 'pc'):
        differences = [a[i][key] - b[i][key] for i in a]
        result[key] = dict(mean=mean(differences), wins=sum(d > 0 for d in differences),
                           ties=sum(d == 0 for d in differences),
                           losses=sum(d < 0 for d in differences))
    result['joint'] = dict(
        gains=[i for i in a if category(a[i]) == 'joint' and category(b[i]) != 'joint'],
        losses=[i for i in a if category(b[i]) == 'joint' and category(a[i]) != 'joint'])
    result['transitions'] = {
        f'{old}->{new}': [i for i in a if category(b[i]) == old and category(a[i]) == new]
        for old in ('joint', 'pc_only', 'sa_only', 'neither')
        for new in ('joint', 'pc_only', 'sa_only', 'neither')}
    return result


def main():
    evidence = read(REPORT / 'prer13_mechanism_matrix.json')
    sources = {r['model']: Path(r['source']) for r in evidence['rows'] if r['mode'] == 'normal'}
    sources['Wan+LoRA'] = ROOT / 'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40/scores.json'
    sources['R9'] = ROOT / 'r1_content_reader_r9/inference_outputs/physical_content_66968/step0400_full40_controls16_physical5_seed42/videophy40/scores.json'
    sources['R9.1'] = ROOT / 'content_value_r9_1/inference_outputs/physical_content_66970/step0400_full40_controls16_physical5_seed42/videophy40/scores.json'
    r12b = ROOT / 'semantic_execution_r12b/inference_outputs/physical_content_67456/step0400_original_fixed40_controls16_seed42/videophy40/scores.json'
    if r12b.exists():
        sources['R12B'] = r12b
    rows = {name: read(path)['cases'] for name, path in sources.items()}
    expected = {f'VP{i:03d}' for i in range(1, 41)}
    reference = {r['id']: r for r in rows['R1']}
    for name, cases in rows.items():
        if len(cases) != 40 or {r['id'] for r in cases} != expected:
            raise ValueError(f'{name}: incomplete original40')
        for row in cases:
            if row.get('caption') != reference[row['id']].get('caption'):
                raise ValueError(f'{name}: original evaluation caption differs')
    result = dict(
        protocol='Complete historical fixed40 I2V, own400, seed42, gamma5. Categories use SA>=4 and PC>=4 on the same video.',
        limitations='R11B used the API data bundle; versions are not single-factor architecture comparisons. These evaluator categories do not establish actual process or visual quality.',
        models={name: dict(source=str(sources[name]), **summary(cases)) for name, cases in rows.items()},
        paired_to_R1={name: paired(cases, rows['R1']) for name, cases in rows.items() if name != 'R1'},
        paired_to_R10A={name: paired(cases, rows['R10-A']) for name, cases in rows.items() if name != 'R10-A'},
        goal=dict(pc=3.8, pc_preferred=3.85, sa=3.3, sa_preferred=3.35,
                  joint_minimum=12, count=40, actual_video_quality_required=True))
    (REPORT / 'prer13_joint_bottlenecks40.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    ids = {r['id'] for r in read(ROOT / 'semantic_execution_r12b/subset16.json')['cases']}
    additions = dict(protocol=evidence['protocol'], models={})
    for name, folder in [('R9', 'r1_content_reader_r9'), ('R9.1', 'content_value_r9_1')]:
        normal = [r for r in rows[name] if r['id'] in ids]
        controls = {}
        for branch in ('inference_outputs', 'control_dispatch/inference_outputs'):
            for path in (ROOT / folder / branch).glob('*/**/controls/*/videophy40/scores.json'):
                control = read(path)['cases']
                if len(control) != 16 or {r['id'] for r in control} != ids:
                    continue
                mode = path.parents[1].name
                item = dict(source=str(path), summary=summary(control), normal_minus_control=paired(normal, control))
                if mode in controls and controls[mode]['summary'] != item['summary']:
                    raise ValueError(f'{name}/{mode}: differing duplicate results')
                controls[mode] = item
        additions['models'][name] = dict(normal=summary(normal), controls=controls)
    (REPORT / 'prer13_r9_controls16.json').write_text(json.dumps(additions, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({name: {k: v for k, v in value.items() if k not in ('source', 'cases')}
                      for name, value in result['models'].items()}, ensure_ascii=False))
    print(json.dumps({name: {mode: c['summary'] | {'cases': None} for mode, c in value['controls'].items()}
                      for name, value in additions['models'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
