"""Check the five launchers and CPU behavior without submitting GPU work."""
import ast
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]


def main():
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    rows = json.loads((ROOT/'variants.json').read_text())
    result = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  gpu_training_started=False, cluster_jobs_submitted=False,
                  cpu_test_backbones='Small substitutes; actual conditioner/operator/native attention code.',
                  resource_checks=[])
    for path in ROOT.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
    for path in ROOT.rglob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True, cwd=ROOT)
    result['syntax'] = 'passed'
    for row in rows:
        command = ['bash', str(ROOT/row['launcher']), '--check']
        process = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
        (reports/f'{row["variant"]}_preflight.log').write_text(process.stdout+process.stderr)
        entry = dict(variant=row['variant'], exit_code=process.returncode)
        if process.returncode == 0:
            entry['report'] = json.loads(process.stdout)
        result['resource_checks'].append(entry)
        print(json.dumps(dict(kind='preflight', **entry)), flush=True)
    process = subprocess.run([sys.executable, '-I', '-B', str(ROOT/'tests/test_q_variants.py')],
                             cwd=ROOT, env=env, text=True, capture_output=True)
    log = process.stdout+process.stderr
    (reports/'cpu_tests.log').write_text(log)
    count = re.search(r'Ran (\d+) tests', log)
    result['cpu_tests'] = dict(exit_code=process.returncode, tests=int(count[1]) if count else None,
                               log='reports/cpu_tests.log')
    result['status'] = 'passed' if process.returncode == 0 and all(x['exit_code'] == 0 for x in result['resource_checks']) else 'failed'
    (reports/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(log, flush=True)
    print(json.dumps(dict(status=result['status'], report=str(reports/'verification.json'))), flush=True)
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
