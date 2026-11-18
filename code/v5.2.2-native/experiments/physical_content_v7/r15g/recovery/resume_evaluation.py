"""Resume only the unfinished evaluation of the preserved R15G400 checkpoint."""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'snapshots/20261107T040316Z'
RUN_ID = 'physical_content_68103'
CHECKPOINT = ROOT / 'runs' / RUN_ID / 'step0400.pt'

if not CHECKPOINT.is_file() or CHECKPOINT.stat().st_size == 0:
    raise FileNotFoundError('The original trained400 checkpoint is required; recovery cannot train a replacement.')
if sys.argv[1:] not in ([], ['--describe']):
    raise ValueError('This recovery has fixed source, checkpoint, cases, and inference settings.')
print(json.dumps(dict(kind='evaluation_only_recovery', checkpoint=str(CHECKPOINT),
    source=str(SOURCE), new_training_steps=0, reuse_completed_videos=True)), flush=True)
sys.path.insert(0, str(SOURCE))
sys.argv = [str(SOURCE / 'run_candidate.py'), '--config',
    str(SOURCE / 'configs/physical_content.yaml'), '--run-id', RUN_ID,
    '--include-extra20', *sys.argv[1:]]
runpy.run_path(str(SOURCE / 'run_candidate.py'), run_name='__main__')
