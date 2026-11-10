"""Report the original40, fixed added20 and combined60 from completed scores only."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'videophy60'
spec = importlib.util.spec_from_file_location('existing_score_review', ROOT / 'early_execution_r3/review_results.py')
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)
spec = importlib.util.spec_from_file_location('existing_pc_policy', ROOT / 'build_results_tables.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def main():
    models = {
        'R1_400': dict(extension='R1_400', original=ROOT / 'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40'),
        'native400': dict(extension='native400', original=ROOT / 'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40'),
    }
    for label,directory,extension in [('R9','r1_content_reader_r9','R9'),
                                      ('R9.1','content_value_r9_1','R9_1'),
                                      ('R10-A','execution_value_r10','R10'),
                                      ('R10-B','grounded_execution_r10b','R10B')]:
        metadata = ROOT / directory / 'prepared.json'
        if not metadata.exists():continue
        prepared = json.loads(metadata.read_text())
        run = prepared.get('successful_candidate_run')
        if run:
            models[label+'_400'] = dict(extension=extension+'_'+run,
                original=ROOT/directory/'inference_outputs'/run/'step0400_full40_controls16_physical5_seed42/videophy40')
    partition_sizes = dict(original40=40, added20=20, combined60=60)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    ids = [case['id'] for case in protocol['cases']]
    expected = dict(original40=set(ids[:40]), added20=set(ids[40:]), combined60=set(ids))
    suites, paths, rows, case_rows, pending = {}, {}, [], [], []
    for model, config in models.items():
        extension = HERE / 'inference_outputs' / config['extension']
        folders = dict(original40=config['original'],
            added20=extension / 'extra20_gamma5_seed42/videophy60',
            combined60=extension / 'full60_gamma5_seed42/videophy60')
        for partition, folder in folders.items():
            key = model + '_' + partition
            paths[key] = folder
            suites[key] = review.summarize(folder)
            if suites[key]['status'] != 'completed':
                pending.append(key)
                continue
            scores = json.loads((folder / 'scores.json').read_text())['cases']
            if {case['id'] for case in scores} != expected[partition]:
                raise ValueError('Unexpected fixed evaluation cases: ' + key)
            n = partition_sizes[partition]
            if len(scores) != n:
                raise ValueError('Unexpected evaluation count: ' + key)
            counts = dict(sa=sum(case['sa'] >= 4 for case in scores),
                pc=sum(case['pc'] >= 4 for case in scores),
                joint=sum(case['sa'] >= 4 and case['pc'] >= 4 for case in scores))
            rows.append(dict(model=model, partition=partition, count=n,
                sa_mean=suites[key]['sa_mean'], pc_mean=suites[key]['pc_mean'],
                sa_pass_count=counts['sa'], pc_pass_count=counts['pc'], joint_pass_count=counts['joint'],
                sa_pass_rate=counts['sa']/n, pc_pass_rate=counts['pc']/n, joint_pass_rate=counts['joint']/n,
                source_scores=str(folder / 'scores.json')))
            case_rows.extend(dict(model=model, partition=partition, id=case['id'],
                caption=case['caption'], sa=case['sa'], pc=case['pc']) for case in scores)
    pairs, retention = {}, {}
    for partition, n in partition_sizes.items():
        specifications = [('native400', 'R1_400')]
        for model in models:
            if model not in ['native400', 'R1_400']:
                specifications += [('native400', model), ('R1_400', model)]
        for before, after in specifications:
            bk, ck = before + '_' + partition, after + '_' + partition
            if all(suites[key]['status'] == 'completed' for key in [bk, ck]):
                pairs[after + '_vs_' + before + '_' + partition] = review.comparison.compare(paths[bk], paths[ck])
                if before == 'R1_400':
                    retention[ck] = policy.assess_pc_reference(suites[bk], suites[ck], expected_count=n)
    report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), suites=suites,
        completed=rows, pending=pending, paired_comparisons=pairs, pc_reference_preservation=retention,
        protocol=str(HERE / 'protocol.json'), gamma=5, seed=42,
        aggregation='SA/PC are means of integer1-5 scores. Pass means>=4; Joint requires both>=4. Combined60 reuses the completed40 scores and adds20, without re-scoring40.',
        scope='Fixed shared-first-image I2V adaptation, not the official full590-caption T2V benchmark. Designs have used the original40 results; report the added20 separately rather than describing all60 as an independent test set.',
        reference='Compare each candidate with R1/native on the identical partition. The original40 PC3.625 is not a numerical threshold for a different60-case set.',
        latest_target='FinalR10: original40 andfull60 PC>3.7(prefer3.8),SA>3.25,Joint aboveR1(atleast11/40and14/60),with actual visual/process improvement. C diversity is diagnostic; useful shared priors are allowed. Original matchedR1 means and paired outcomes remain reported.',
        new_model_evaluation=False, automatic_adoption=False, automatic_extension=False)
    target = HERE / 'reports'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'generation_review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    for filename, values in [('completed_partition_summary.csv', rows), ('completed_partition_cases.csv', case_rows)]:
        if not values:
            continue
        with (target / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    print(json.dumps(dict(output=str(target / 'generation_review.json'), completed=[row['model'] + '_' + row['partition'] for row in rows], pending=pending)))


if __name__ == '__main__':
    main()
