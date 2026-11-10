"""Replace the process with the project's existing independent evaluator."""
import argparse
import os
from pathlib import Path
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=True)
args = parser.parse_args()
cfg = yaml.safe_load(Path(args.config).read_text())
root = Path(__file__).resolve().parents[1]
python = '/home/liuzhirui/miniconda3/envs/physvid/bin/python'
os.execv(python, [python, '-I', '-B', str(root / 'tools/evaluate_videophy2.py'),
                 str(Path(cfg['output']) / 'original'), str(Path(cfg['output']) / 'physics_text')])
