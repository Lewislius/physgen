"""One fixed16 comparison of rebuilt physical inputs with frozen BT400."""
import argparse
import json
import sys
from pathlib import Path

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
BT = BASE/'experiments/physical_content_r20b/snapshots/20261115T052651Z'
CASES = ['VP002','VP004','VP005','VP009','VP010','VP013','VP015','VP016',
         'VP017','VP018','VP024','VP025','VP026','VP027','VP034','VP038']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    ready = json.loads((args.features/'ready.json').read_text())
    if ready['revision'] != 'R20-source-grounded-physical-memory-v5':
        raise ValueError('The comparison requires the revised input cache')
    for case in CASES:
        item = json.loads((args.features/'inference'/('videophy40_'+case)/'ready.json').read_text())
        if item['reasoning_revision'] != ready['revision']:
            raise ValueError('Incomplete or mixed input features')
    sys.path.insert(0, str(BASE/'tools'))
    from gpu_policy import require_rtx_pro_6000
    require_rtx_pro_6000()
    sys.path.insert(0, str(BT))
    from pipeline import evaluator_environment, run_stage
    args.output.mkdir(parents=True, exist_ok=True)

    def stage(name, command, environment=None):
        (args.output/'status.json').write_text(json.dumps(dict(stage=name,
            comparison='frozen BT400; only physical input preparation changed',
            new_training=False, paired_cases=CASES), indent=2)+'\n')
        run_stage(command, args.output/(name+'.log'), environment)

    stage('generation16', [sys.executable, '-I', '-B', '-u',
        str(args.source/'paired_infer.py'), '--reference-checkpoint', str(args.checkpoint),
        '--evidence', str(args.features/'inference'), '--output', str(args.output),
        '--cases', *CASES])
    stage('score16', ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-u',
        str(BASE/'tools/evaluate_videophy2.py'), str(args.output/'videophy40')],
        evaluator_environment())
    (args.output/'status.json').write_text(json.dumps(dict(stage='complete',
        comparison='frozen BT400; only physical input preparation changed',
        new_training=False, paired_cases=CASES, interpretation=
        'Paired input evidence, not a measured result for freshly trained R20DT.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
