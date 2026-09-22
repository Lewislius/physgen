"""Per-rank progress and Python stacks, available before CUDA/NCCL initialization.

Stage markers describe CPU calls returning; they do not synchronize CUDA. Explicit
*_sync stages in the caller establish GPU completion. No CUDA API is called here.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import faulthandler
import json
import os
from pathlib import Path
import re
import socket
import sys
import time


class RankDiagnostics:
    def __init__(self, config):
        options = config.get('diagnostics', {})
        self.trace_steps = int(options.get('trace_steps', 1))
        self.trace_blocks = bool(options.get('trace_wan_blocks', False))
        self.stack_timeout = float(options.get('stack_timeout_seconds', 120))
        if self.trace_steps < 0 or self.stack_timeout < 0:
            raise ValueError('diagnostic trace_steps and stack_timeout_seconds must be nonnegative')
        self.enabled = True
        self.context = {}
        self.rank = int(os.environ.get('RANK', '0'))
        self.pid = os.getpid()
        job = os.environ.get('TORCHELASTIC_RUN_ID') or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        job = re.sub(r'[^a-zA-Z0-9_.-]', '_', job)
        self.root = Path(os.environ.get('V43_DIAGNOSTIC_DIR') or
                         str(Path(config['paths']['log_root']) / config['run_name'] / 'diagnostics' / job))
        self.root.mkdir(parents=True, exist_ok=True)
        self.events = (self.root / f'events_rank{self.rank}_pid{self.pid}.jsonl').open('a', buffering=1)
        try:
            self.stacks = (self.root / f'stacks_rank{self.rank}_pid{self.pid}.txt').open('a', buffering=1)
        except BaseException:
            self.events.close()
            raise
        self._depth = 0

    def emit(self, event, **fields):
        value = dict(event=event, rank=self.rank, pid=self.pid,
                     utc=datetime.now(timezone.utc).isoformat(), **self.context)
        value.update(fields)
        line = json.dumps(value, ensure_ascii=False, allow_nan=False)
        # stdout comes first so a blocked shared filesystem still leaves a marker.
        print(line, flush=True)
        self.events.write(line + '\n')

    def _arm(self):
        if self.stack_timeout > 0:
            faulthandler.dump_traceback_later(self.stack_timeout, repeat=True, file=self.stacks)

    def _cancel(self):
        if self.stack_timeout > 0:
            faulthandler.cancel_dump_traceback_later()

    @contextmanager
    def stage(self, name, always=False, **fields):
        if not self.enabled and not always:
            yield
            return
        self._depth += 1
        self._arm()
        started = time.perf_counter()
        try:
            self.emit('stage_begin', stage=name, **fields)
            self.stacks.write(f'[{datetime.now(timezone.utc).isoformat()}] begin {name} {self.context} {fields}\n')
            yield
        except BaseException as error:
            self.emit('stage_error', stage=name, error_type=type(error).__name__, error=str(error), **fields)
            raise
        else:
            self.emit('stage_end', stage=name, seconds=time.perf_counter() - started, **fields)
        finally:
            self._depth -= 1
            if self._depth:
                self._arm()
            else:
                self._cancel()

    @contextmanager
    def iteration(self, step, offset):
        previous_enabled, previous_context = self.enabled, self.context
        self.enabled = offset < self.trace_steps
        self.context = dict(step=step)
        try:
            with self.stage('training_step'):
                yield
        finally:
            self.context = previous_context
            self.enabled = previous_enabled

    def __enter__(self):
        # No torch import or GPU query: this must work before init_process_group.
        if os.environ.get('TRAIN_NCCL_DEBUG') == '1':
            # PyTorch appends the rank. A PID also separates concurrent process groups
            # in different processes; keep an explicitly supplied destination intact.
            os.environ.setdefault('TORCH_NCCL_DEBUG_INFO_TEMP_FILE',
                                  str(self.root / f'nccl_trace_pid{self.pid}_rank'))
        names = ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'TMPDIR',
                 'PYTORCH_CUDA_ALLOC_CONF', 'NCCL_DEBUG', 'NCCL_DEBUG_FILE',
                 'TORCH_NCCL_ENABLE_MONITORING', 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC',
                 'TORCH_NCCL_TRACE_BUFFER_SIZE', 'TORCH_NCCL_DUMP_ON_TIMEOUT',
                 'TORCH_NCCL_TRACE_CPP_STACK', 'TORCH_NCCL_DEBUG_INFO_TEMP_FILE')
        self.emit('diagnostics_start', hostname=socket.gethostname(), python=sys.version.split()[0],
                  directory=str(self.root), trace_steps=self.trace_steps, trace_wan_blocks=self.trace_blocks,
                  stack_timeout_seconds=self.stack_timeout,
                  environment={name: os.environ[name] for name in names if name in os.environ})
        return self

    def __exit__(self, error_type, error, traceback):
        self._cancel()
        try:
            self.emit('diagnostics_end', status='failed' if error_type else 'completed',
                      error_type=error_type.__name__ if error_type else None)
        finally:
            self.stacks.close()
            self.events.close()
