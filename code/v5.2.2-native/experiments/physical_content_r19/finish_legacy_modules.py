"""Stop owned old-model diagnostics after their two useful module comparisons.

This watches local completion files, not the cluster queue. It preserves scored
videos, cancels only the recorded accepted-RTX run, and launches no experiment.
"""
from pathlib import Path
import json
import subprocess
import time

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
E = BASE/'experiments/physics_content_v10_impl'
TASKS = {
    'g': E/'reports/controllers/r19_evidence_r18g_20261112T163220Z.json',
}


def main():
    remaining = dict(TASKS)
    record = BASE/'experiments/physical_content_r19/reports/legacy_modules_completion.json'
    result = {}
    while remaining:
        for task, state_path in list(remaining.items()):
            root = E/f'inference_outputs/r18{task}_r19_controls16_seed42'
            report = root/'paired_summary.json'
            if not report.exists():
                continue
            paired = json.loads(report.read_text())
            controls = paired.get('controls', {})
            if not all(controls.get(mode,{}).get('count') == 16
                       for mode in ['no_c','conditioned_ffn_off']):
                continue
            state = json.loads(state_path.read_text())
            guard = state.get('actual_gpu') or {}
            if not guard.get('accepted'):
                continue
            identity = state['experiment_id']
            response = subprocess.run(
                ['/home/liuzhirui/.local/bin/det','-m','http://10.130.130.5:4096',
                 'experiment','cancel',str(identity)], capture_output=True, text=True)
            result[task] = dict(experiment_id=identity, requested=time.time(),
                                cancel_returncode=response.returncode,
                                preserved_module_scores=str(report),
                                purpose='Prioritize new R19 training; remaining old-model condition matrix deferred')
            record.write_text(json.dumps(result,indent=2))
            if response.returncode == 0:
                remaining.pop(task)
        if remaining:
            time.sleep(60)


if __name__ == '__main__':
    main()
