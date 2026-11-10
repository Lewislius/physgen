"""Run the frozen no-reference16 intervention on an idle Ada/Amp allocation."""
from pathlib import Path
import argparse,importlib.util,json,sys

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
RUNTIME=ROOT/'r16_evaluation_runtime/snapshots/20261109T060601Z'
CONTROL=ROOT/'native_joint_read_r16a/no_reference_eval/snapshots/20261109T044055Z'
SOURCE=ROOT/'native_joint_read_r16a/snapshots/20261109T040426Z'
sys.path.insert(0,str(RUNTIME))
from runtime import activate,stage_command


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',choices=['ada-48g','amp-48g','amp-80g'],required=True)
    args,original=parser.parse_known_args()
    if '--describe' not in original:
        for name in ['native_joint_read_r16a','native_joint_read_r16a_nozero']:
            root=ROOT/name;prepared=json.loads((root/'prepared.json').read_text())
            status_path=root/'reports/jobs'/str(prepared['experiment'])/'status.json'
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            if not prepared.get('training_complete') or status.get('status')!='declared_evaluation_completed':
                raise RuntimeError('Complete both formal native400 evaluation suites before this auxiliary16')
    activate(SOURCE,args.pool)
    sys.path.insert(0,str(CONTROL))
    spec=importlib.util.spec_from_file_location('frozen_reference_control',CONTROL/'run_control.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    original_stage=module.run_stage
    def run_stage(command,logfile,env=None):
        return original_stage(stage_command(command,SOURCE,args.pool),logfile,env)
    module.run_stage=run_stage
    sys.argv=[str(CONTROL/'run_control.py'),*original]
    module.main()


if __name__=='__main__':main()
