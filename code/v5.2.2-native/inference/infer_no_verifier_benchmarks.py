"""Run a final no-verifier H/Q checkpoint on demo21 or the fixed VideoPhy2-40."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from inference.no_verifier_jobs import JOBS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True, choices=JOBS)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--cpu-offload', action='store_true')
    args = parser.parse_args(argv)
    job = JOBS[args.job]
    output = Path(job['output'])
    command = ['--checkpoint', job['checkpoint'], '--mode', job['mode'], '--suite', job['suite'],
               '--training-experiment-id', str(job['training_experiment_id']), '--output', str(output),
               '--seed', '42', '--fps', '24', '--steps', '50', '--shift', '5', '--guidance', '5',
               '--no-verifier-only', '--rtx-pro6000-only']
    if (output / 'manifest.json').is_file() and (output / 'checkpoint.pt').is_file():
        command.append('--resume')
    if args.check_only:
        command.append('--check-only')
    if args.cpu_offload:
        command.append('--cpu-offload')
    from inference.infer_message_demo21 import main as generate
    generate(command)


if __name__ == '__main__':
    main()

