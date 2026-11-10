"""One final checkpoint, one SH/YAML and one complete pipeline per method."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NATIVE_ROOT = ROOT.parents[1]
MASTER = 'http://10.130.130.5:4096'
Q_REFERENCE = NATIVE_ROOT / 'inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/videophy40_final224_round2_shared_noise'
REFERENCE_ANALYSIS = NATIVE_ROOT / 'analysis/q_videophy40_autoeval_20261027'
JOBS = {}
for variant, experiment, trial, revision in (
    ('b', 66523, 67903, 'v522_native_q_content_pool_b_v31'),
    ('c', 66524, 67904, 'v522_native_q_content_pool_no_self_c_v32'),
    ('d-s', 66526, 67906, 'v522_native_q_identity_self_ds_v34'),
    ('d-i', 66527, 67907, 'v522_native_q_identity_qk_self_di_v35'),
):
    stem = variant.replace('-', '_')
    run = f'v522-q-workspace-{variant}_{experiment}'
    JOBS[variant] = dict(
        variant=variant, training_experiment_id=experiment, training_trial_id=trial,
        revision=revision, checkpoint=str(ROOT / 'checkpoints' / run / 'latest.pt'),
        output=str(ROOT / 'inference_outputs' / run / 'final224_demo21_videophy40_amp80g'),
        sh=str(ROOT / 'inference' / f'run_{stem}_amp80g.sh'),
        yaml=str(ROOT / 'inference' / f'{stem}_amp80g.yaml'),
        estimated_hours=3, estimated_range_hours=[2.5, 3.5],
    )
