"""Aggregate existing AutoEval outputs and local training evidence; no model execution."""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'archive/20261027_cleanup/analysis/independent_video_evaluation_v2'
COMMON = {'source0000042', 'source0000086'}
RUNS = {
    'direct_fm': 'v522-diagnostic-direct_fm-8cases32updates-seed42_65835',
    'reasoned_fm': 'v522-diagnostic-reasoned_fm-8cases32updates-seed42_65838',
    'reasoned_process': 'v522-diagnostic-reasoned_process-8cases32updates-seed42_65836',
    'H': 'v522-h-content-message-verifier-frozen-simple_66356',
    'Q': 'v522-q-seed128-message-verifier-frozen-simple_66353',
}


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def aggregate(rows):
    return dict(count=len(rows), sa_mean=statistics.mean(r['sa'] for r in rows),
                pc_mean=statistics.mean(r['pc'] for r in rows),
                joint_pass_rate=statistics.mean(r['sa'] >= 4 and r['pc'] >= 4 for r in rows))


rows = []
protocol = None
captions = {}
for label, run in RUNS.items():
    base = HERE / 'evaluation' if label in ('H', 'Q') else OLD
    paths = sorted((base / run).glob('step*_repeat*/scores.json'))
    assert len(paths) == (14 if label in ('H', 'Q') else 4), (label, paths)
    for path in paths:
        step, repeat = [int(x) for x in path.parent.name.replace('step', '').split('_repeat')]
        manifest = read(path.parent / 'manifest.json')
        if protocol is None:
            protocol = manifest['evaluation']
        assert manifest['evaluation'] == protocol, path
        cases = {c['id']: c for c in manifest['cases']}
        scores = read(path)['cases']
        assert len(scores) == len(cases)
        for score in scores:
            case = cases[score['id']]
            assert score['caption'] == case['caption']
            assert score['generation_seed'] == case['generation_seed']
            assert all(isinstance(score[k], int) and 1 <= score[k] <= 5 for k in ('sa', 'pc'))
            expected = captions.setdefault(score['id'], score['caption'])
            assert score['caption'] == expected
            if label in ('H', 'Q'):
                assert (path.parent / 'cases' / score['id'] / 'planned.mp4').resolve() == Path(case['source_video']).resolve()
            rows.append(dict(run=label, run_directory=run, step=step, repeat=repeat,
                             case_id=score['id'], generation_seed=score['generation_seed'],
                             sa=score['sa'], pc=score['pc'],
                             joint_pass=int(score['sa'] >= 4 and score['pc'] >= 4),
                             scores_file=str(path)))

assert len(rows) == 104
with (HERE / 'scores.csv').open('w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

final = []
for label in RUNS:
    step = 224 if label in ('H', 'Q') else 32
    selected = [r for r in rows if r['run'] == label and r['step'] == step and r['case_id'] in COMMON]
    assert {(r['case_id'], r['generation_seed']) for r in selected} == {
        ('source0000042', 42), ('source0000042', 314159),
        ('source0000086', 43), ('source0000086', 314160)}
    final.append(dict(run=label, step=step, **aggregate(selected)))
trajectory = []
for step in (32, 64, 96, 128, 160, 192, 224):
    for label in ('H', 'Q'):
        selected = [r for r in rows if r['run'] == label and r['step'] == step]
        assert len(selected) == 4
        trajectory.append(dict(run=label, step=step, **aggregate(selected)))
summary = dict(
    scope='VideoPhy2 AutoEval applied to held-out WISA videos; not official VideoPhy2 benchmark performance.',
    experiment_id=66423, new_videos_scored=56, new_score_calls=112,
    shared_final_comparison=final, hq_checkpoint_comparison=trajectory,
    old_final_four_case_results=[dict(run=label, **aggregate([
        r for r in rows if r['run'] == label and r['step'] == 32]))
        for label in RUNS if label not in ('H', 'Q')],
    protocol=protocol,
    limits=['Only two shared scenes with two paired seeds; no general superiority or equivalence claim.',
            'Integer AutoEval scores do not capture all visible distortion.',
            'Training verifier is a different model from the independent VideoPhy2 evaluator.'])
write('comparison.json', summary)

audit = read(HERE / 'training_data_audit.json')
current_manifest = ROOT.parent.parent / 'data_cache/wisa_4000/base/manifest.json'
current = read(current_manifest)['records']
older_root = ROOT.parent / 'v5.2-occam'
older_manifest = older_root / 'cache/wisa_832x480_f81/manifest.json'
older = read(older_manifest)['records']
matching = {}
for case in current:
    if case['id'] not in audit['displayed_cases']:
        continue
    matching[case['id']] = dict(
        matching_source_rows=[dict(id=r['id'], split=r['split']) for r in older
                              if r.get('video_name') == case.get('video_name') or r.get('source_id') == case.get('source_id')],
        training_matches_by_existing_metadata={field: [r['id'] for r in older
            if r['split'] == 'train' and case.get(field) and r.get(field) == case[field]]
            for field in ('video', 'video_name', 'source_id', 'caption', 'original_caption')})
old_checkpoint_dir = older_root / 'checkpoints/wan_lora_v5_no_intensity_20261008T155650Z_38/no_intensity'
audit['historical_initialization'] = dict(
    note='The v11 step16 initialization itself inherits earlier training; 896 is only each current H/Q run, not total lifetime training exposure.',
    wan_lora_source=str(old_checkpoint_dir / 'joint/joint_000400.pt'),
    wan_joint_run=read(old_checkpoint_dir / 'joint/run.json'),
    wan_planner_run=read(old_checkpoint_dir / 'planner/run.json'),
    wan_training_manifest=str(older_manifest), older_manifest_splits=dict(Counter(r['split'] for r in older)),
    displayed_case_membership_in_older_manifest=matching,
    condition_source=dict(run='v522-native-causal-response-wisa3857_66180', step=144,
        manifest=str(current_manifest), split='train',
        experiment_config=str(ROOT / 'archive/20261027_cleanup/analysis/queue_20261025_causal_response/experiment.yaml'),
        source_metadata=str(ROOT / 'archive/20261027_cleanup/analysis/queue_20261026_videophy_baseline/job.json'),
        displayed_cases_in_training_pool=[]),
    conclusion='The four displayed sources remain validation in both local historical manifests; no identical source/caption training matches found.')
assert all(not ids for item in matching.values() for ids in item['training_matches_by_existing_metadata'].values())
assert all(r['split'] == 'validation' for item in matching.values() for r in item['matching_source_rows'])
write('training_data_audit.json', audit)

validation = {}
for label in ('H', 'Q'):
    metrics = [json.loads(line) for line in (ROOT / 'checkpoints' / RUNS[label] / 'metrics.jsonl').read_text().splitlines()]
    selected = [r for r in metrics if r['kind'] == 'validation' and r['step'] >= 64]
    points = [dict(step=r['step'], fm={s: r[f'sigma{s}/loss/fm'] for s in ('0.25', '0.65', '0.95')}) for r in selected]
    first = next(r for r in points if r['step'] == 64)
    last = next(r for r in points if r['step'] == 224)
    validation[label] = dict(points=points, step64_to224_relative_change_percent={
        s: 100 * (last['fm'][s] / first['fm'][s] - 1) for s in first['fm']})
write('validation_fm_audit.json', validation)
print(json.dumps(dict(final=final, hq_trajectory=trajectory,
                      historical_validation_confirmed=True,
                      validation_changes={k: v['step64_to224_relative_change_percent'] for k, v in validation.items()}), indent=2))
