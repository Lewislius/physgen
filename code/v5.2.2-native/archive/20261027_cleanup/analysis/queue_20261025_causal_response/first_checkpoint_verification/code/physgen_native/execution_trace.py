"""Opt-in first-microbatch locations for failures that leave no Python traceback."""
from contextlib import contextmanager
from contextvars import ContextVar
import json
import time
import torch

_context = ContextVar('execution_trace', default=None)


def _write(stage, event, **extra):
    values = dict(kind='execution', stage=stage, event=event, unix=time.time(), **_context.get())
    if torch.cuda.is_initialized():
        values.update(allocated_gib=torch.cuda.memory_allocated()/2**30,
                      reserved_gib=torch.cuda.memory_reserved()/2**30,
                      peak_gib=torch.cuda.max_memory_allocated()/2**30)
    print(json.dumps(dict(values, **extra)), flush=True)


@contextmanager
def trace_micro(enabled, **context):
    token = _context.set(context if enabled else None)
    try:
        yield
    finally:
        _context.reset(token)


def traced_call(stage, function, *args, **kwargs):
    if _context.get() is None:
        return function(*args, **kwargs)
    _write(stage, 'begin')
    start = time.monotonic()
    result = function(*args, **kwargs)
    # Only the engineering check synchronizes at stage boundaries. A
    # pending CUDA failure is thus attributed to the computation just run.
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
    _write(stage, 'end', seconds=time.monotonic()-start)
    return result
