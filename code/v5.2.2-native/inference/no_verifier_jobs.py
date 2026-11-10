"""The four final v29/v30 jobs share one explicit checkpoint/output mapping."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MASTER = 'http://10.130.130.5:4096'
RECORD = ROOT / 'analysis/no_verifier_inference_20261027/jobs.json'
DOCUMENT = ROOT / 'docs/无Verifier_HQ最终权重推理记录_20261027.md'
DEFAULT_JOBS = ['h_demo21', 'q_demo21', 'h_videophy40']
JOBS = {}
for letter, mode, experiment, run in (
    ('h', 'h_content_message', 66438, 'v522-h-content-message-no-verifier-simple_66438'),
    ('q', 'q_seed128_message', 66443, 'v522-q-seed128-message-no-verifier-simple_66443'),
):
    for suite, count in (('demo21', 21), ('videophy40', 40)):
        key = f'{letter}_{suite}'
        stem = f'infer_{key}_no_verifier_blk96g'
        suffix = 'demo21_final224_round2_seed42' if suite == 'demo21' else 'videophy40_final224_round2_shared_noise'
        JOBS[key] = dict(key=key, variant=letter.upper(), mode=mode, suite=suite, total_cases=count,
                         training_experiment_id=experiment, checkpoint=str(ROOT / 'checkpoints' / run / 'latest.pt'),
                         output=str(ROOT / 'inference_outputs' / run / suffix),
                         sh=str(ROOT / 'inference' / f'{stem}.sh'),
                         yaml=str(ROOT / 'inference' / f'{stem}.yaml'))

