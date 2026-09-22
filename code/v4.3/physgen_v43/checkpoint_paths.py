"""Checkpoint naming and completed-run discovery; no model or CUDA imports."""
import argparse
import json
import os
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def name_training_run(config, session_id, resume_from=None):
    """Every invocation, including RESUME, writes into its own timestamped run."""
    previous_name = config['run_name']
    base = config.get('run_name_base', previous_name)
    for label, value in (('run_name', base), ('session_id', session_id)):
        if not isinstance(value, str) or not value.strip() or value in ('.', '..') or Path(value).name != value:
            raise ValueError(f'{label} must be a single nonempty directory name')
    config['run_name_base'] = base
    config['run_name'] = f'{base}_{session_id}'
    config['resume_from'] = str(resume_from) if resume_from else None
    config['source_run_name'] = previous_name if resume_from else None
    return Path(config['paths']['checkpoint_root']) / config['run_name']


def latest_completed_checkpoint(root, run_name_base):
    """Select the most recently completed matching run, including legacy folders.

    A directory or weights file alone is insufficient: latest.json must acknowledge
    the final save, after all ranks finished writing. Never select an in-progress run.
    """
    root = Path(root)
    candidates = []
    for run in root.iterdir() if root.is_dir() else ():
        if run.name != run_name_base and not run.name.startswith(run_name_base + '_'):
            continue
        final = run / 'checkpoint-final'
        files = ('corrector.pt', 'training.pt', 'config.yaml', 'runtime.json')
        if not all((final / name).is_file() and (final / name).stat().st_size > 0 for name in files):
            continue
        try:
            latest = json.loads((run / 'latest.json').read_text())
            if Path(latest['checkpoint']).name != 'checkpoint-final':
                continue
            saved = yaml.safe_load((final / 'config.yaml').read_text())
            if saved.get('run_name_base', saved.get('run_name')) != run_name_base:
                continue
            completed = (run / 'latest.json').stat().st_mtime_ns
        except (OSError, ValueError, TypeError, KeyError, AttributeError, yaml.YAMLError):
            continue
        candidates.append((completed, run.name, final))
    if not candidates:
        raise FileNotFoundError(
            f'No completed checkpoint-final for {run_name_base!r} under {root}. '
            'Set INIT_FROM (training) or CHECKPOINT (inference) to an explicit saved checkpoint directory.')
    return max(candidates)[2]


def main():
    parser = argparse.ArgumentParser(description='Find the newest completed timestamped or legacy training run')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', help='Training YAML containing paths.checkpoint_root')
    source.add_argument('--root', help='Checkpoint root directory')
    parser.add_argument('--run-name', required=True, help='Run name before the automatic timestamp suffix')
    args = parser.parse_args()
    try:
        root = args.root
        if args.config:
            config = yaml.safe_load(Path(args.config).read_text())
            root = os.path.expandvars(config['paths']['checkpoint_root'].replace('${V43_ROOT}', str(ROOT)))
        print(latest_completed_checkpoint(root, args.run_name))
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
