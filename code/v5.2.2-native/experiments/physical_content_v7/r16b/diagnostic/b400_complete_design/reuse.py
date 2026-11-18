"""Reuse only the declared completed B400 joint-read16, including raw scores."""
from pathlib import Path
import json
import os

ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
PROBE = ROOT / 'r15i_design/probe_outputs/68309/videophy40'
PROBE_SOURCE = ROOT / 'r15i_design/snapshots/20261108T115136Z'


def verify_existing(previous, expected, output):
    ignored = {'probe_source', 'inference_source'}
    if any(previous.get(k) != v for k, v in expected.items() if k not in ignored):
        raise ValueError(f'Different output provenance: {output}')
    if previous.get('probe_source') != expected['probe_source']:
        if (previous.get('probe_source') != str(PROBE_SOURCE)
                or previous.get('inference_source') != str(PROBE_SOURCE)
                or Path(output).resolve() != (PROBE / 'cases' / previous['case'] / 'planned.mp4').resolve()):
            raise ValueError('Only the original completed16 may be reused')
    elif previous.get('inference_source') != expected['inference_source']:
        raise ValueError('Different generation source')


def reuse_probe16(destination, reference, checkpoint, operator):
    manifest = json.loads((PROBE / 'manifest.json').read_text())
    status = json.loads((PROBE / 'summary.json').read_text())
    rows = json.loads((PROBE / 'scores.json').read_text())['cases']
    expected_ids = {v['id'] for v in json.loads((PROBE_SOURCE / 'subset16.json').read_text())['cases']}
    if (status.get('status') != 'completed' or len(rows) != 16
            or {r['id'] for r in rows} != expected_ids
            or {r['id'] for r in manifest['cases']} != expected_ids):
        raise ValueError('Original probe16 is incomplete')
    if (manifest['checkpoint'] != str(checkpoint) or manifest['checkpoint_step'] != 400
            or manifest['evaluation'] != reference['evaluation']
            or manifest['sampling'] != reference['sampling']
            or Path(operator).read_bytes() != (PROBE_SOURCE / 'joint_read.py').read_bytes()):
        raise ValueError('Operator, weights or evaluation protocol differs')
    references = {c['id']: c for c in reference['cases']}
    for case in manifest['cases']:
        for field in ('caption', 'prepared_inputs', 'initial_noise', 'generation_seed'):
            if case.get(field) != references[case['id']].get(field):
                raise ValueError(f'Changed input: {case["id"]}/{field}')
        target = Path(destination) / 'cases' / case['id']
        target.mkdir(parents=True, exist_ok=True)
        for filename in ('planned.mp4', 'planned.json', 'planned.sa.json', 'planned.pc.json'):
            source = PROBE / 'cases' / case['id'] / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            if not (target / filename).exists():
                os.symlink(source, target / filename)
    return dict(count=16, cases=sorted(expected_ids), source=str(PROBE),
                video_and_raw_scores_reused=True, no_best_of_selection=True,
                original_generation_gpu='RTX PRO6000', additional_gpu='declared allocated Ada48 or RTX6000')
