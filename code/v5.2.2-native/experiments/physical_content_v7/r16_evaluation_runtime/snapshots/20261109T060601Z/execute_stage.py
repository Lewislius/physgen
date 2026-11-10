"""Execute the original evaluation script with explicit runtime overrides."""
from pathlib import Path
import runpy,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from runtime import activate

source,pool,target,*arguments=sys.argv[1:]
if Path(target).name in {'train.py','train_only.py','gpu_preflight.py'}:
    raise ValueError('Evaluation-only entrypoint')
activate(source,pool)
sys.argv=[target,*arguments]
runpy.run_path(target,run_name='__main__')
