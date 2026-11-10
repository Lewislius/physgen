"""Explicit native observation geometry; never silently stretch physics evidence."""


def uses_native_geometry(cfg):
    return cfg.get('geometry', {}).get('mode') == 'native'


def require_native_size(actual, expected):
    if tuple(actual) != tuple(expected):
        raise ValueError(f'Native observation size mismatch: got HxW={tuple(actual)}, expected {tuple(expected)}; '
                         'resizing, stretching and cropping are disabled on this path')


def require_same_aspect(actual, target):
    height, width = actual
    th, tw = target
    if min(height, width, th, tw) <= 0 or height * tw != width * th:
        raise ValueError(f'Aspect ratio would change: HxW={tuple(actual)} -> {tuple(target)}; '
                         'use the native geometry or an exact proportional auxiliary view')
