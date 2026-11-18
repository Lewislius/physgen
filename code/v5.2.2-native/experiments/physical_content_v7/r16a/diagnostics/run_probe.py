"""Run frozen B and I diagnostics in isolated imports on one allocated GPU."""
from pathlib import Path
import subprocess
import sys

if __name__ == '__main__':
    source = Path(__file__).resolve().parent / 'probe_training_read.py'
    for model in ['B', 'I']:
        subprocess.run([sys.executable, '-I', '-B', '-u', str(source), '--model', model], check=True)
