"""Expose only the Wan model subpackage when the full package needs optional deps.

The official ``wan/__init__.py`` imports speech-to-video modules (and therefore
librosa) even for TI2V.  The TI2V model files do not need those modules.  A
minimal package namespace lets the verified local Wan source load in the
training environment without changing any Wan weight or operation.
"""
from pathlib import Path
import sys
import types


def ensure_wan_namespace(cfg):
    if 'wan' in sys.modules:
        return
    root = Path(cfg['paths']['wan_code']) / 'wan'
    if not root.is_dir():
        raise FileNotFoundError(root)
    package = types.ModuleType('wan')
    package.__path__ = [str(root)]
    package.__package__ = 'wan'
    sys.modules['wan'] = package
