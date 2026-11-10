"""Launch native training with checkpoint-aware Determined restarts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_recorded import run


def training_command(config, experiment_id, *, resume=None, initialize_generator=None,
                     checkpoint_every=4):
    if checkpoint_every < 1:
        raise ValueError('checkpoint_every must be positive')
    config = Path(config).resolve()
    cfg = yaml.safe_load(config.read_text())
    if resume is None:
        if not experiment_id:
            raise ValueError('A Determined experiment ID or explicit resume directory is required')
        runs = Path(cfg['paths']['runs'].replace('${ROOT}', str(ROOT)))
        directory = runs / f'{cfg["run_name"]}_{experiment_id}'
    else:
        directory = Path(resume).resolve()
        if not (directory / 'latest.pt').is_file():
            raise FileNotFoundError(directory / 'latest.pt')
    restoring = (directory / 'latest.pt').is_file()
    if restoring:
        config = directory / 'config.yaml'
        if not config.is_file(): raise FileNotFoundError(config)
    command = ['bash', str(ROOT / 'tools/runtime_python.sh'), str(ROOT / 'train/train.py'),
               '--config', str(config), '--checkpoint-every', str(checkpoint_every)]
    if restoring:
        command += ['--resume', str(directory)]
    elif initialize_generator is not None:
        command += ['--initialize-generator', str(initialize_generator)]
    return command, directory, restoring


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--initialize-generator', type=Path)
    parser.add_argument('--checkpoint-every', type=int, default=4)
    args = parser.parse_args()
    command, directory, restoring = training_command(args.config, os.environ.get('DET_EXPERIMENT_ID'),
        resume=args.resume, initialize_generator=args.initialize_generator,
        checkpoint_every=args.checkpoint_every)
    # An interrupted first checkpoint may leave config.yaml but no latest.pt.
    # The existing trainer then validates that config before a fresh start.
    os.environ['AUTO_RESUME'] = '1'
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    # Do not create the training run before train.py creates config.yaml.
    evidence = directory.parent / 'recovery_logs' / directory.name / f'{stamp}.resources.jsonl'
    print(json.dumps(dict(kind='recovery_launch', restoring=restoring,
        run_directory=str(directory), checkpoint_every=args.checkpoint_every,
        resources=str(evidence), command=shlex.join(command))), flush=True)
    return run(command, evidence)


if __name__ == '__main__':
    sys.exit(main())
