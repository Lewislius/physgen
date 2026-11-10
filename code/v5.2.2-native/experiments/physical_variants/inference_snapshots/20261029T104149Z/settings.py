"""Pinned final training weights and the established 61-case input protocol."""
from pathlib import Path
import sys

NATIVE_ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = NATIVE_ROOT / 'experiments/physical_variants'
TRAIN_SOURCE = ROOT / 'snapshots/20261029T064043Z'
Q_REFERENCE = NATIVE_ROOT / 'inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/videophy40_final224_round2_shared_noise'
DEMO_REFERENCE = Q_REFERENCE.parent / 'demo21_final224_round2_seed42'
REFERENCE_ANALYSIS = NATIVE_ROOT / 'analysis/q_videophy40_autoeval_20261027'
MASTER = 'http://10.130.130.5:4096'
REVISION = 'physical_variants_20261029_r1'
JOBS = {}
for variant, experiment, run in (
    ('v5_ce', 66643, 'v5_ce_20261029T064043Z-retry1'),
    ('v6_write', 66641, 'v6_write_20261029T064043Z'),
    ('v6_assoc', 66642, 'v6_assoc_20261029T064043Z'),
):
    JOBS[variant] = dict(variant=variant, training_experiment_id=experiment,
        checkpoint=str(ROOT/'runs'/run/'step0224.pt'), training_source=str(TRAIN_SOURCE),
        output=str(ROOT/'inference_outputs'/f'{variant}_{experiment}'/'final224_demo21_videophy40_r1'))


def activate_training():
    # Import precisely the implementation used by all three completed runs.
    sys.path.insert(0, str(TRAIN_SOURCE))
    import physical_training.model as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(TRAIN_SOURCE):
        raise RuntimeError('Inference imported a different training implementation')

