"""Audit completed R-series scores and matched conditions without loading models."""
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parent
HISTORY = ROOT.parent / 'physical_content_v7'


def read(path):
    return json.loads(path.read_text())


def score_rows(path):
    rows = read(path)['cases']
    indexed = {row['id']: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f'Duplicate scored case in {path}')
    return indexed


def summarize(rows):
    values = list(rows.values())
    return dict(count=len(values), sa=mean(row['sa'] for row in values),
                pc=mean(row['pc'] for row in values),
                joint=sum(row['sa'] >= 4 and row['pc'] >= 4 for row in values))


def compare(candidate, reference):
    if candidate.keys() != reference.keys():
        raise ValueError('Paired scores must cover exactly the same cases')
    result = {}
    for metric in ['sa', 'pc']:
        delta = [candidate[key][metric] - reference[key][metric] for key in candidate]
        result[metric] = dict(mean_delta=mean(delta), wins=sum(v > 0 for v in delta),
                              ties=sum(v == 0 for v in delta), losses=sum(v < 0 for v in delta))
    result['joint_delta'] = summarize(candidate)['joint'] - summarize(reference)['joint']
    return result


def training_records(folder):
    micros = {}
    audit = None
    for line in (folder / 'train.jsonl').read_text().splitlines():
        value = json.loads(line)
        if value.get('kind') == 'audit':
            audit = value
        if value.get('kind') == 'micro':
            micros[(value['update'], value['micro'])] = value
    return audit, micros


def main():
    report = dict(models={}, matched_mechanism_comparisons={}, historical_controls={})
    primary = {}
    training = {}
    for letter in 'efgh':
        name = f'R18{letter.upper()}'
        folder = ROOT / 'inference_outputs' / f'r18{letter}_nchunks400' / 'step0400_fixed40_extra20_seed42'
        first = score_rows(folder / 'videophy40/scores.json')
        extra = score_rows(folder / 'videophy60/scores.json')
        if first.keys() & extra.keys() or len(first) != 40 or len(extra) != 20:
            raise ValueError(f'Invalid 40+20 partition for {name}')
        metadata = [read(folder / suite / 'cases' / key / 'planned.json')
                    for suite, rows in [('videophy40', first), ('videophy60', extra)] for key in rows]
        tasks = {value['task'] for value in metadata}
        if len(tasks) != 1:
            raise ValueError(f'Mixed task modes for {name}')
        task = next(iter(tasks))
        if any(value['sampler']['clamp_first'] != (task == 'i2v') for value in metadata):
            raise ValueError(f'Task/masking mismatch for {name}')
        audit, micros = training_records(ROOT / 'runs' / f'r18{letter}_nchunks400')
        training[name] = micros
        primary[name] = dict(main40=first, full60=first | extra)
        report['models'][name] = dict(task=task, main40=summarize(first), full60=summarize(first | extra),
            training_unique_microsteps=len(micros), training_manifest_hash=audit['train_manifest_hash'],
            source_scores=[str(folder / suite / 'scores.json') for suite in ['videophy40', 'videophy60']],
            completed_condition_controls=sorted(str(p) for p in (folder / 'controls').glob('**/scores.json')),
            score_scope_label_matches_task=all(('I2V' not in read(folder / suite / 'scores.json')['summary']['scope'])
                                              for suite in ['videophy40', 'videophy60']) if task == 't2v' else True)
    for candidate, reference in [('R18G', 'R18E'), ('R18H', 'R18F')]:
        left, right = training[candidate], training[reference]
        fields = ['sample', 'sigma', 'condition_dropped']
        matching = left.keys() == right.keys() and all(
            all(left[key][field] == right[key][field] for field in fields) for key in left)
        report['matched_mechanism_comparisons'][f'{candidate}_versus_{reference}'] = dict(
            data_sigma_dropout_match=matching,
            **{suite: compare(primary[candidate][suite], primary[reference][suite])
               for suite in ['main40', 'full60']})
    references = {
        'R9': HISTORY / 'r9/inference_outputs/physical_content_66968/step0400_full40_controls16_physical5_seed42',
        'R10A': HISTORY / 'r10/inference_outputs/physical_content_67003/step0400_full40_controls16_physical5_seed42',
        'R15B': HISTORY / 'r15b/inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42',
    }
    for name, folder in references.items():
        normal = score_rows(folder / 'videophy40/scores.json')
        controls = {}
        for path in sorted((folder / 'controls').glob('*/videophy40/scores.json')):
            rows = score_rows(path)
            matched = {key: normal[key] for key in rows}
            controls[path.parents[1].name] = dict(normal=summarize(matched), control=summarize(rows),
                normal_minus_control=compare(matched, rows), source=str(path))
        report['historical_controls'][name] = controls
    report['limitations'] = [
        'E versus G and F versus H each compare two trained mechanisms, including the FFN adapter change. Their deltas do not isolate one module.',
        'Historical controls diagnose their own trained checkpoints. They do not establish current R18 condition sensitivity.',
        'Physical-process completion and visual stability require video inspection. Score summaries alone do not establish them.',
        'The scorer scope text incorrectly says I2V for R18E/G. Task is verified using all 60 generation records and sampler masks.',
    ]
    destination = ROOT / 'reports/goal_evidence_audit.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(report=str(destination), models=report['models'],
                         comparisons=report['matched_mechanism_comparisons']), ensure_ascii=False))


if __name__ == '__main__':
    main()
