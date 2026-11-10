"""Report one read-only probe against fixed B400 endpoints, not as final I scores."""
from pathlib import Path
import argparse
import json
import statistics

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
B = ROOT / 'request_bound_fusion_r15b/inference_outputs/physical_content_67829/step0400_original_fixed40_controls16_seed42'


def summary(cases):
    return dict(count=len(cases), SA=statistics.mean(x['sa'] for x in cases),
                PC=statistics.mean(x['pc'] for x in cases),
                Joint=sum(x['sa'] >= 4 and x['pc'] >= 4 for x in cases))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    actual = json.loads((args.output / 'videophy40/scores.json').read_text())['cases']
    ids = {x['id'] for x in json.loads((Path(__file__).parent / 'subset16.json').read_text())['cases']}
    assert len(actual) == 16 and {x['id'] for x in actual} == ids
    joint = lambda x: x['sa'] >= 4 and x['pc'] >= 4
    result = dict(kind='R15B400_readonly_joint_physical_memory_probe', training_updates=0,
                  probe=summary(actual), references={}, quality_proven=False, automatic_training=False,
                  interpretation='Fixed16 is a diagnostic sample, not full40 acceptance. Review quantitative and actual-video evidence before final I design; no per-case combination.')
    for name, path in [('B_request_bound', B / 'videophy40/scores.json'),
                       ('B_direct', B / 'controls/direct_physical_read/videophy40/scores.json'),
                       ('B_no_c', B / 'controls/no_c/videophy40/scores.json')]:
        before = {x['id']: x for x in json.loads(path.read_text())['cases'] if x['id'] in ids}
        assert set(before) == ids
        for row in actual:
            assert all(row[k] == before[row['id']][k] for k in ['caption', 'generation_seed'])
        changes = {}
        for metric in ['sa', 'pc']:
            values = [x[metric] - before[x['id']][metric] for x in actual]
            changes[metric] = dict(mean_change=statistics.mean(values), wins=sum(v > 0 for v in values),
                                   ties=sum(v == 0 for v in values), losses=sum(v < 0 for v in values))
        changes['joint'] = dict(gains=[x['id'] for x in actual if joint(x) and not joint(before[x['id']])],
                                losses=[x['id'] for x in actual if joint(before[x['id']]) and not joint(x)])
        result['references'][name] = dict(summary=summary(list(before.values())), probe_minus_reference=changes,
                                          source=str(path))
    (args.output / 'paired_report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
