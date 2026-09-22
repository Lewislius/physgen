"""External, training-free ACE-Router integration for the local Wan2.2 tree.

The package intentionally does not import Wan or torch at module import time.  This
keeps schema/check-only commands lightweight and lets the entrypoint install the
read-only Wan source directory on ``sys.path`` before GPU modules are imported.
"""

from .schema import AceSample, discover_sample_ids, load_sample
from .schedules import layer_gates_for_preset, step_gate, step_gates

__all__ = [
    "AceSample",
    "discover_sample_ids",
    "layer_gates_for_preset",
    "load_sample",
    "step_gate",
    "step_gates",
]
