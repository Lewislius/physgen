"""Always-on local metrics, with explicitly enabled W&B reporting."""
from contextlib import ExitStack
import importlib
import json
import math
from pathlib import Path
import runpy
import time


ROOT = Path(__file__).resolve().parents[1]
TRAIN_LOG_ROOT = ROOT / 'train/train_log'
CREDENTIALS_FILE = ROOT / 'train/wandb_credentials.py'


def normalize_logging(config):
    # Old checkpoints also adopt the current output directory and monitoring defaults.
    config['paths']['log_root'] = str(TRAIN_LOG_ROOT)
    logging = {key: value for key, value in config.get('logging', {}).items()
               if key in ('wandb', 'session_id')}
    config['logging'] = logging
    wandb = logging.setdefault('wandb', {})
    wandb.setdefault('enabled', False)
    wandb.setdefault('project', 'physgen-v4')
    wandb.setdefault('entity', None)
    return config


def public_config(value):
    """Do not include credentials in local snapshots, checkpoints, or W&B config."""
    if isinstance(value, dict):
        return {key: ('[REDACTED]' if str(key).lower() in
                      {'api_key', 'wandb_api_key', 'password', 'access_token', 'secret'} else public_config(item))
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_config(item) for item in value]
    return value


def read_wandb_key():
    if not CREDENTIALS_FILE.is_file():
        raise RuntimeError(f'W&B enabled: create {CREDENTIALS_FILE} with WANDB_API_KEY first')
    # Import only when enabled; the key never becomes part of the training config.
    try:
        values = runpy.run_path(str(CREDENTIALS_FILE))
    except Exception:
        # A syntax error must not echo the source line containing the user's key.
        raise RuntimeError(f'Could not load W&B credentials from {CREDENTIALS_FILE}; check the local file') from None
    key = values.get('WANDB_API_KEY', '')
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError(f'W&B enabled but WANDB_API_KEY is empty in {CREDENTIALS_FILE}')
    return key.strip()


def checkpoint_source_files(root):
    root = Path(root)
    for directory in ('physgen_v4', 'train', 'inference', 'tools', 'configs', 'evaluation'):
        for source in (root / directory).rglob('*'):
            relative = source.relative_to(root)
            if relative == Path('train/wandb_credentials.py') or relative.parts[:2] == ('train', 'train_log'):
                continue
            if source.is_file() and source.suffix in ('.py', '.sh', '.yaml', '.md'):
                yield source


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Preserve NaN/Inf diagnostics while producing valid JSON.
    return value


def _json(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, allow_nan=False)


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_json(value) + '\n')
    temporary.replace(path)


class Logger:
    def __init__(self, config, rank):
        self.rank = rank
        self.wandb_run = None
        self._files = ExitStack()
        self._closed = False
        self.latest = {}
        session = config['logging'].get('session_id', 'standalone')
        self.root = Path(config['paths']['log_root']) / config['run_name'] / session
        self.root.mkdir(parents=True, exist_ok=True)
        self.total_steps = config.get('train', {}).get('steps', '?')
        try:
            self.micro_json = self._open(f'micro_rank{rank}.jsonl')
            self.micro_txt = self._open(f'micro_rank{rank}.txt')
            if rank == 0:
                self.steps = self._open('steps.jsonl')
                self.steps_txt = self._open('steps.txt')
                _write_json(self.root / 'config.json', public_config(config))
                options = config['logging'].get('wandb', {})
                if options.get('enabled', False):
                    key = read_wandb_key()
                    wandb = importlib.import_module('wandb')
                    if not wandb.login(key=key, relogin=True):
                        raise RuntimeError('W&B authentication failed')
                    del key
                    self.wandb_run = wandb.init(
                        project=options.get('project', 'physgen-v4'), entity=options.get('entity'),
                        name=config['run_name'], id=session if session != 'standalone' else None,
                        dir=str(self.root), config=public_config(config), mode='online',
                        save_code=False,
                        settings=wandb.Settings(disable_code=True, disable_git=True, console='off'))
                    self.wandb_run.define_metric('global_step')
                    self.wandb_run.define_metric('train/*', step_metric='global_step')
                    self.wandb_run.define_metric('validation/*', step_metric='global_step')
        except BaseException:
            self.close(exit_code=1)
            raise

    def _open(self, name):
        return self._files.enter_context((self.root / name).open('a', buffering=1))

    def start(self, metadata):
        if self.rank == 0:
            value = dict(event='training_start', log_directory=str(self.root), **public_config(metadata))
            _write_json(self.root / 'runtime.json', value)
            print(_json(value), flush=True)

    @staticmethod
    def _metrics(metrics):
        return {name: float(value.detach().item() if hasattr(value, 'detach') else value)
                for name, value in metrics.items()}

    def micro(self, step, micro, index, metrics, variables=None):
        values = self._metrics(metrics)
        record = dict(step=step, micro=micro, index=index, rank=self.rank, time=time.time(),
                      metrics=values, variables=variables or {})
        self.micro_json.write(_json(record) + '\n')
        self.micro_txt.write(f'step={step} micro={micro} index={index} rank={self.rank} ' +
                             ' '.join(f'{key}={value:.8g}' for key, value in values.items()) +
                             ' variables=' + _json(record['variables']) + '\n')

    def step(self, step, metrics, validation=False):
        if self.rank != 0:
            return
        values = self._metrics(metrics)
        split = 'validation' if validation else 'train'
        record = dict(step=step, split=split, time=time.time(), metrics=values)
        self.steps.write(_json(record) + '\n')
        full_text = f'[{split}] step={step}/{self.total_steps} ' + ' '.join(
            f'{key}={value:.8g}' for key, value in values.items())
        self.steps_txt.write(full_text + '\n')
        self.latest[split] = record
        _write_json(self.root / 'latest.json', self.latest)
        # Determined captures stdout without any optional visualization package.
        summary = ('validation_loss',) if validation else (
            'loss/total', 'loss/fm', 'loss/struct', 'loss/prior', 'loss/out', 'loss/weighted_fm',
            'loss/weighted_struct', 'loss/weighted_prior', 'loss/weighted_out', 'learning_rate',
            'grad/global_norm_before_clip', 'sigma', 'output/active', 'prior/gt_conditioned',
            'state/warm_active', 'compute/wan_forwards', 'time/step_seconds')
        print(f'[{split}] step={step}/{self.total_steps} ' + ' '.join(
            f'{key}={values[key]:.8g}' for key in summary if key in values), flush=True)
        if self.wandb_run is not None:
            # A separate x-axis preserves validation logged at the same optimizer step.
            self.wandb_run.log({'global_step': step, **{f'{split}/{key}': value for key, value in values.items()}})

    def close(self, exit_code=0):
        if self._closed:
            return
        self._closed = True
        self._files.close()
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=exit_code)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, *_):
        self.close(exit_code=1 if exception_type is not None else 0)
