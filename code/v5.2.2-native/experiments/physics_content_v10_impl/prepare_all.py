"""Complete every data/interface/preflight step needed before R18E-H training.

This is intentionally CPU/GPU preprocessing only: no optimizer, no videos and
no VideoPhy2 scoring.  It is safe to rerun; completed files are skipped.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bootstrap  # noqa: F401

from physical_content.common import atomic_json, load_config


def run(name, command, report):
    started = time.monotonic()
    print(json.dumps(dict(stage=name, status='started'), ensure_ascii=False), flush=True)
    result = subprocess.run(command, cwd=ROOT)
    elapsed = time.monotonic() - started
    report[name] = dict(status='ok' if result.returncode == 0 else 'failed',
                         returncode=result.returncode, seconds=elapsed)
    print(json.dumps(dict(stage=name, **report[name]), ensure_ascii=False), flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    started = time.monotonic()
    report = {}
    e = load_config(ROOT / 'configs/r18e.yaml', require_paths=True)
    cache_root = Path(e['paths']['input_cache']).parent
    ready = cache_root / 'ready.json'
    native_ready = Path(e['paths']['native_words_cache']).parent / 'native_ready.json'

    if not ready.is_file():
        run('prepare_t2v_evidence', [
            sys.executable, '-I', '-B', '-u', str(ROOT / 'prepare_t2v_evidence.py'),
            '--config', str(ROOT / 'configs/r18e.yaml'), '--device', 'cuda'], report)
    else:
        report['prepare_t2v_evidence'] = dict(status='already_complete')

    if not native_ready.is_file():
        run('prepare_native_t2v', [
            sys.executable, '-I', '-B', '-u', str(ROOT / 'prepare_native_t2v.py'),
            '--config', str(ROOT / 'configs/r18e.yaml'), '--device', 'cuda'], report)
    else:
        report['prepare_native_t2v'] = dict(status='already_complete')

    for plan in ('e', 'f', 'g', 'h'):
        config = ROOT / f'configs/r18{plan}.yaml'
        cfg = load_config(config, require_paths=True)
        interface = Path(cfg['paths']['interface'])
        if not interface.is_file():
            run(f'fit_interface_r18{plan}', [
                sys.executable, '-I', '-B', '-u', str(ROOT / 'fit_interface.py'),
                '--config', str(config)], report)
        else:
            report[f'fit_interface_r18{plan}'] = dict(status='already_complete')
        run(f'prepare_inputs_r18{plan}', [
            sys.executable, '-I', '-B', '-u', str(ROOT / 'prepare_inputs.py'),
            '--config', str(config)], report)
        run(f'audit_r18{plan}', [
            sys.executable, '-I', '-B', '-u', str(ROOT / 'train.py'),
            '--config', str(config), '--check-only'], report)

    # Persist exact counts for the training jobs to verify.
    checks = {}
    for plan in ('e', 'f', 'g', 'h'):
        cfg = load_config(ROOT / f'configs/r18{plan}.yaml', require_paths=True)
        interface = Path(cfg['paths']['interface'])
        evidence = Path(cfg['paths']['inference_evidence'])
        checks[f'r18{plan}'] = dict(
            interface=interface.is_file(),
            interface_bytes=interface.stat().st_size if interface.is_file() else 0,
            prepared_positive=len(list(evidence.glob('*/positive.pt'))),
            readied=len(list(evidence.glob('*/ready.json'))),
        )
    report['checks'] = checks
    report['status'] = 'complete'
    report['seconds'] = time.monotonic() - started
    atomic_json(report, ROOT / 'reports/preprocess/preflight.json')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
