"""Pinned final training weights and the established 61-case input protocol."""
from pathlib import Path
import sys

NATIVE_ROOT = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
ROOT = NATIVE_ROOT / 'experiments/physical_scaling_q256'
TRAIN_SOURCE = ROOT / 'snapshots/20261029T140406Z'
FOUNDATION_SOURCE = NATIVE_ROOT / 'experiments/physical_variants/snapshots/20261029T064043Z'
INPUT_CACHE_READONLY = NATIVE_ROOT / 'experiments/physical_variants/inference_cache/physical_r1_first_image_request'
Q_REFERENCE = NATIVE_ROOT / 'inference_outputs/v522-q-seed128-message-no-verifier-simple_66443/videophy40_final224_round2_shared_noise'
DEMO_REFERENCE = Q_REFERENCE.parent / 'demo21_final224_round2_seed42'
REFERENCE_ANALYSIS = NATIVE_ROOT / 'analysis/q_videophy40_autoeval_20261027'
MASTER = 'http://10.130.130.5:4096'
REVISION = 'physical_scaling_q256_s1000_20261029_r1'
CHECKPOINT_STEP = 1000
Q_TOKENS = 256
JOBS = {}
for variant, experiment, run in (
    ('v6_assoc', 66678, 'v6_assoc_66678'),
):
    JOBS[variant] = dict(variant=variant, training_experiment_id=experiment,
        checkpoint=str(ROOT/'runs'/run/'step1000.pt'), training_source=str(TRAIN_SOURCE),
        output=str(ROOT/'inference_outputs'/f'{variant}_{experiment}'/'final1000_demo21_videophy40_r1'))


def activate_training():
    # Same import order and read-only foundation wrapper as the submitted trainer.
    sys.path.insert(0, str(FOUNDATION_SOURCE))
    sys.path.insert(0, str(TRAIN_SOURCE))
    import physical_scaling.model as implementation
    if not Path(implementation.__file__).resolve().is_relative_to(TRAIN_SOURCE):
        raise RuntimeError('Inference imported a different training implementation')
