"""Continue the selected variant using its saved config, optimizer and RNG."""
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physgen_native.config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=['b', 'c', 'd-n', 'd-s', 'd-i'])
    parser.add_argument('--run', required=True, type=Path)
    args, rest = parser.parse_known_args()
    run = args.run.resolve()
    cfg = read_config(run/'config.yaml')
    if cfg['message'].get('workspace_variant') != args.variant:
        raise ValueError('The requested launcher differs from the saved variant')
    if not (run/'latest.pt').is_file():
        raise FileNotFoundError(run/'latest.pt')
    if any(x.split('=')[0] in ('--config', '--resume') or x.startswith('--initialize-') for x in rest):
        raise ValueError('Resume restores the saved configuration and weights')
    command = ['bash', str(ROOT/'tools/runtime_python.sh'), str(ROOT/'train/train.py'),
               '--config', str(run/'config.yaml'), '--resume', str(run), *rest]
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
