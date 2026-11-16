"""Install the Wan2.1 source package without executing optional Wan modules.

The shared conda environment also contains an installed ``wan`` package.  The
R20FT candidate must use exactly the Wan2.1 source tree named by the config.  A
minimal package namespace is installed before the first ``import wan`` so that
``wan.modules.model``, ``wan.modules.vae`` and ``wan.modules.t5`` all resolve to
Wan2.1.  Replacing an already-imported foreign ``wan`` package is refused.
"""
from pathlib import Path
import sys
import types

DEFAULT_WAN21_CODE = Path('/home/liuzhirui/model/Wan2.1')


def _resolved_root(path=None):
    root = Path(path or DEFAULT_WAN21_CODE) / 'wan'
    return root.resolve()


def install_wan21_namespace(path=None):
    """Install a ``wan`` namespace pointing at the Wan2.1 source tree.

    This function is idempotent when the installed namespace already points at
    the requested tree.  If another ``wan`` package was imported first, it
    raises instead of silently mixing Wan2.2 operators into the candidate.
    """
    root = _resolved_root(path)
    if not root.is_dir():
        raise FileNotFoundError(f'Wan2.1 source package not found: {root}')
    current = sys.modules.get('wan')
    if current is not None:
        paths = [Path(item).resolve() for item in getattr(current, '__path__', [])]
        if root in paths:
            return current
        raise RuntimeError(
            'A different `wan` package was imported before the Wan2.1 namespace '
            f'was installed: {paths}')
    package = types.ModuleType('wan')
    package.__path__ = [str(root)]
    package.__package__ = 'wan'
    sys.modules['wan'] = package
    return package


def assert_wan21_source(path=None):
    """Fail if the imported Wan operators came from a non-Wan2.1 tree."""
    root = _resolved_root(path)
    import wan.modules.model as model

    module = Path(model.__file__).resolve()
    if root not in module.parents:
        raise RuntimeError(f'Wan model imported from {module}, expected under {root}')
    return module
