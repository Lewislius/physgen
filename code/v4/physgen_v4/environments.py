"""The two fixed runtimes used by v4. No interpreter override or fallback."""
import json
import os
from pathlib import Path
import sys


ENV_ROOT = Path('/home/liuzhirui/miniconda3/envs')
ENVIRONMENTS = {'runtime': ENV_ROOT / 'moviestory', 'teacher': ENV_ROOT / 'vjepa2-312'}


def allocated_gpu_count():
    """Read the single-node allocation without importing the cluster SDK."""
    if 'DET_SLOT_IDS' in os.environ:
        slots = json.loads(os.environ['DET_SLOT_IDS'])
        if not isinstance(slots, list):
            raise ValueError('DET_SLOT_IDS must contain the allocated slot list')
        count = len(slots)
    else:
        # Direct execution in an allocated container follows CUDA visibility.
        import torch
        count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError('No GPUs allocated to this training task')
    return count


def python_for(role):
    executable = ENVIRONMENTS[role] / 'bin/python'
    if not executable.is_file():
        raise FileNotFoundError(f'Required {role} interpreter is missing: {executable}; no fallback is allowed')
    return str(executable)


def require_environment(role):
    expected = ENVIRONMENTS[role].resolve()
    if Path(sys.prefix).resolve() != expected or Path(sys.executable).resolve().parent != expected / 'bin':
        raise RuntimeError(f'{role} requires {expected}/bin/python; actual interpreter: {sys.executable}')
    for entry in sys.path:
        path = Path(entry).resolve()
        if ENV_ROOT in path.parents and expected not in path.parents and path != expected:
            raise RuntimeError(f'Foreign environment on sys.path: {path}; clear PYTHONPATH/PYTHONHOME')
    return dict(role=role, environment=str(expected), python=sys.executable)


def subprocess_environment(role):
    """Keep cluster/GPU settings while removing inherited Python environment selection."""
    environment = os.environ.copy()
    for key in list(environment):
        if key.startswith('PYTHON') or key.startswith('CONDA_') or key in ('VIRTUAL_ENV', 'VJEPA_ENV'):
            environment.pop(key)
    prefix = ENVIRONMENTS[role]
    for key in ('PATH', 'LD_LIBRARY_PATH'):
        entries = [entry for entry in environment.get(key, '').split(os.pathsep)
                   if entry and ENV_ROOT not in Path(entry).resolve().parents]
        if key == 'PATH':
            entries.insert(0, str(prefix / 'bin'))
        environment[key] = os.pathsep.join(entries)
    environment.update(CONDA_PREFIX=str(prefix), CONDA_DEFAULT_ENV=prefix.name,
                       PYTHON_EXEC=python_for(role), PYTHONNOUSERSITE='1',
                       PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    return environment
