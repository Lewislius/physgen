"""Merge the fixed 40 + disjoint 20 into the R15Bt T2V 60-case report.

The old videophy60 merge script assumes R15B's guidance schema; this report
only needs the already-validated per-suite scores, so it cannot be blocked by
manifest-format drift.  Every case must have both SA and PC on a 1-5 scale.
"""
from pathlib import Path
import argparse
import json
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_inputs import T2V_SCOPE


def summarize(cases):
    n = len(cases)
    if not n:
        raise ValueError('No scored cases')
    if any(row[key] not in range(1, 6) for row in cases for key in ('sa', 'pc')):
        raise ValueError('Scores must be integer 1..5')
    return dict(count=n,
                sa_mean=statistics.mean(row['sa'] for row in cases),
                pc_mean=statistics.mean(row['pc'] for row in cases),
                sa_pass_count=sum(row['sa'] >= 4 for row in cases),
                sa_pass_rate=sum(row['sa'] >= 4 for row in cases) / n,
                pc_pass_count=sum(row['pc'] >= 4 for row in cases),
                pc_pass_rate=sum(row['pc'] >= 4 for row in cases) / n,
                joint_pass_count=sum(row['sa'] >= 4 and row['pc'] >= 4 for row in cases),
                joint_pass_rate=sum(row['sa'] >= 4 and row['pc'] >= 4 for row in cases) / n,
                scope=T2V_SCOPE, training_verifier_is_independent=True,
                first_frame_used=False, task='t2v')


def read_scores(path, ids):
    value = json.loads(Path(path).read_text())
    cases = value['cases']
    got = {row['id'] for row in cases}
    if len(cases) != len(ids) or got != set(ids):
        raise ValueError(f'{path}: expected exactly {sorted(ids)[:3]}...')
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-extra20', action='store_true')
    parser.add_argument('--exclude-extra20', action='store_true')
    args = parser.parse_args()
    output = args.output
    main40 = read_scores(output / 'videophy40' / 'scores.json',
                         [f'VP{i:03d}' for i in range(1, 41)])
    extra_path = output / 'videophy60' / 'scores.json'
    extra20 = None
    if not args.exclude_extra20 and extra_path.is_file():
        extra20 = read_scores(extra_path, [f'VP{i:03d}' for i in range(41, 61)])
    elif args.require_extra20:
        raise FileNotFoundError(extra_path)
    report = dict(
        candidate='R15Bt', task='t2v', paradigm='R15B enhancement + B400 joint read',
        first_frame_used=False, known_latent_frames=0,
        training_sample_match='Same 1600 draws as R15B: SampleOrder(seed42) over 3666 train rows',
        scoring='VideoPhy2 AutoEval SA/PC 1-5; pass >=4; Joint requires both on the same video',
        scope=T2V_SCOPE, main={'videophy40': summarize(main40)})
    if extra20 is not None:
        combined = main40 + extra20
        if {row['id'] for row in combined} != {f'VP{i:03d}' for i in range(1, 61)}:
            raise ValueError('40 + 20 do not cover the fixed 60 exactly')
        combined.sort(key=lambda row: row['id'])
        destination = output / 'full60' / 'videophy60'
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'scores.json').write_text(json.dumps(
            dict(summary=summarize(combined), cases=combined),
            ensure_ascii=False, indent=2))
        manifest40 = json.loads((output / 'videophy40' / 'manifest.json').read_text())
        manifest20 = json.loads((output / 'videophy60' / 'manifest.json').read_text())
        cases40 = manifest40['cases']
        cases20 = manifest20['cases']
        if ({case['id'] for case in cases40} | {case['id'] for case in cases20}
                != {f'VP{i:03d}' for i in range(1, 61)}):
            raise ValueError('Generation manifests do not cover the fixed 60 exactly')
        manifest = dict(manifest40)
        manifest.update(suite='videophy60', scope=T2V_SCOPE,
                        cases=cases40 + cases20,
                        first_frame_used=False, known_latent_frames=0,
                        task='t2v', merged_from=['videophy40', 'videophy60'])
        (destination / 'manifest.json').write_text(json.dumps(
            manifest, ensure_ascii=False, indent=2))
        report['main']['videophy60_extra20'] = summarize(extra20)
        report['main']['full60'] = summarize(combined)
        report['combined_scores'] = str(destination / 'scores.json')
    else:
        report['pending_main'] = ['videophy60_extra20']
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({key: report[key] for key in ('candidate', 'task', 'main')},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
