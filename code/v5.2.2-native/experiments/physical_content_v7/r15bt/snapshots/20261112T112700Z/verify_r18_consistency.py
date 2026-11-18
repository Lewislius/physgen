"""Audit R15Bt's T2V data/eval contract against the R18E/G T2V branch.

This is a read-only consistency check.  It deliberately compares the shared
task/data/evaluation parts, not the R15B physical design or its 128+272 update
schedule, which are intended to differ.
"""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import yaml
import torch

from physical_content.common import (DATA_REVISION, REVISION,
                                     SOURCE_EVIDENCE_REVISION, data_ready,
                                     interface_ready, load_config)

BASE = bootstrap.BASE
V10 = BASE / 'experiments/physics_content_v10_impl'
R15B = Path(__file__).resolve().parent


def r18_paths():
    submission = yaml.safe_load((V10 / 'submission_r18e_ada48g.yaml').read_text())
    env = dict(item.split('=', 1) for item in
               submission['environment']['environment_variables']
               if isinstance(item, str) and '=' in item)
    return dict(train_cache=Path(env['TRAIN_CACHE']),
                run_root=Path(env['RUN_ROOT']))


def main():
    cfg = load_config(R15B / 'configs/r15bt.yaml')
    r18 = r18_paths()
    checks = {}

    checks['task_t2v'] = cfg['task'] == 't2v'
    checks['train_cache_matches_r18'] = (
        Path(cfg['paths']['cache']).resolve() == r18['train_cache'].resolve())
    checks['input_cache_matches_r18'] = (
        Path(cfg['paths']['t2v_input_cache']).resolve()
        == (r18['run_root'] / 't2v_cache/input').resolve())
    checks['words_cache_matches_r18'] = (
        Path(cfg['paths']['t2v_words_cache']).resolve()
        == (r18['run_root'] / 't2v_cache/words').resolve())
    checks['native_cache_matches_r18'] = (
        Path(cfg['paths']['t2v_native_words_cache']).resolve()
        == (r18['run_root'] / 't2v_cache/native_words').resolve())
    checks['eval_cache_matches_r18'] = (
        Path(cfg['paths']['t2v_eval_cache']).resolve()
        == (r18['run_root'] / 't2v_cache/eval_input').resolve())
    checks['eval_protocol_is_videophy60'] = (
        Path(cfg['paths']['eval_protocol']).resolve()
        == (BASE / 'experiments/physical_content_v7/videophy60/protocol.json').resolve())

    protocol = json.loads(Path(cfg['paths']['eval_protocol']).read_text())
    checks['eval_ids_40_plus_20'] = (
        len(protocol['cases']) == 60
        and set(protocol['extra_ids']) == {f'VP{i:03d}' for i in range(41, 61)})
    checks['sampling_matches_r18'] = (
        cfg['inference']['steps'] == 50 and cfg['inference']['shift'] == 5.0
        and cfg['inference']['guidance'] == 5.0
        and cfg['t2v_protocol']['frames'] == 81
        and cfg['t2v_protocol']['fps'] == 24.0
        and cfg['t2v_protocol']['width'] == 832
        and cfg['t2v_protocol']['height'] == 480
        and cfg['t2v_protocol']['cfg'] == 5.0
        and cfg['t2v_protocol']['seed'] == 42)
    checks['input_contract_matches_r18'] = cfg['input'] == dict(
        external_image=False, evidence_scope='request_only',
        known_latent_frames=0, future_targets_in_condition=False,
        chunk_descriptions=False, local_annotation_preprocessing=False)
    checks['train_samples_and_batch'] = (
        cfg['train']['accumulation'] == 4
        and sum(cfg['stages'].values()) == 400)
    checks['data_revision_label'] = cfg['data_revision'] == DATA_REVISION
    checks['source_revision_label'] = (
        SOURCE_EVIDENCE_REVISION == 'NoChunks-v2-20261111')

    counts = {}
    for name in ('input', 'words', 'native_words', 'eval_input'):
        root = Path(cfg['paths']['t2v_input_cache']).parent / name
        counts[name] = len(list(root.glob('*.pt')))
    checks['cache_file_counts'] = (
        counts['input'] == 3200 and counts['words'] == 3320
        and counts['native_words'] == 3320 and counts['eval_input'] == 120)
    ready = json.loads((Path(cfg['paths']['t2v_input_cache']).parent
                        / 'ready.json').read_text())
    checks['evidence_ready'] = (
        ready.get('complete') is True and ready.get('scope') == 'request_only'
        and ready.get('image_given_to_qwen') is False
        and ready.get('revision') == SOURCE_EVIDENCE_REVISION
        and int(ready.get('records', -1)) == 1660)
    native = json.loads((Path(cfg['paths']['t2v_native_words_cache']).parent
                         / 'native_ready.json').read_text())
    checks['native_ready'] = (native.get('status') == 'complete'
                              and int(native.get('count', -1)) == 3320)
    checks['interface_ready'] = interface_ready(cfg)

    try:
        info = data_ready(cfg)
        checks['data_ready'] = (info['matched_draw_samples'] == 1600
                                and info['unique_draw_samples'] == 1600
                                and info['first_frame_used'] is False)
    except Exception as error:  # noqa: BLE001
        checks['data_ready'] = False
        info = dict(error=repr(error))

    # Same filtered 1600-row loader and SampleOrder as R18E/G.
    from physgen_native.data import SampleOrder
    from physical_content.data import Data
    data = Data(cfg)
    checks['filtered_loader_1600'] = len(data) == 1600
    order = iter(SampleOrder(len(data), 42, 0))
    draws_list = [data.rows[next(order)]['id'] for _ in range(1600)]
    draws = set(draws_list)
    cache_ids = {path.stem.removesuffix('_text')
                 for path in Path(cfg['paths']['t2v_input_cache']).glob('*_text.pt')}
    checks['sample_order_matches_cache'] = (
        draws == cache_ids and len(draws_list) == 1600)

    # Every training evidence caption matches the qwen38 training manifest.
    by_id = data.by_id
    mismatched = []
    for sample in draws:
        value = torch.load(Path(cfg['paths']['t2v_input_cache']) / f'{sample}_text.pt',
                           map_location='cpu', weights_only=True)
        if (value.get('caption') != by_id[sample]['caption']
                or value.get('request_word_strings')
                != by_id[sample]['caption'].split()):
            mismatched.append(sample)
    checks['all_1600_evidence_captions_match'] = not mismatched

    # Runtime first-frame ban.
    offenders = []
    for path in R15B.rglob('*.py'):
        if (path.name == 'verify_r18_consistency.py'
                or 'tests' in path.parts or 'snapshots' in path.parts):
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if ("sample['first']" in line or 'first_rgb' in line
                    or 'x[:, :, :1] =' in line.replace('  ', ' ')):
                offenders.append(f'{path.relative_to(R15B)}:{number}')
    checks['no_first_frame_runtime_reference'] = not offenders

    checks['evaluator_code'] = (
        (BASE / 'tools/evaluate_videophy2.py').is_file())

    # The compacted evaluation evidence must be tensor-identical to R18E's
    # prepared inputs; only the local compact revision label may differ.
    r18_prepared = V10 / 'runs/r18e_nchunks400/prepared_inputs'
    evidence_root = Path(cfg['paths']['inference_evidence'])
    evidence_equal = r18_prepared.is_dir()
    if evidence_equal:
        for case in protocol['cases']:
            suite = 'videophy40' if int(case['id'][2:]) <= 40 else 'videophy60'
            name = f'{suite}_{case["id"]}'
            for arm in ('positive', 'negative'):
                ours = torch.load(evidence_root / name / f'{arm}.pt',
                                  map_location='cpu', weights_only=True)
                theirs = torch.load(r18_prepared / name / f'{arm}.pt',
                                    map_location='cpu', weights_only=True)
                for key in ('sample', 'caption', 'scope', 'request_word_strings',
                            'relations', 'analysis'):
                    if ours.get(key) != theirs.get(key):
                        evidence_equal = False
                for key in ('words', 'request_words', 'word_groups',
                            'word_positions'):
                    if not torch.equal(ours[key], theirs[key]):
                        evidence_equal = False
    checks['prepared_eval_evidence_matches_r18e'] = evidence_equal

    # Inference must read only T2V-native tensors, never a first-frame file.
    native_ok = True
    native_root = Path(cfg['paths']['eval_native_inputs'])
    for case in protocol['cases']:
        folder = native_root / case['id']
        value = torch.load(folder / 'conditions.pt', map_location='cpu',
                           weights_only=True)
        info = json.loads((folder / 'input.json').read_text())
        if (set(value) != {'text', 'null_text', 'times'}
                or info.get('image') is not False
                or info.get('first_latent') is not False):
            native_ok = False
    checks['t2v_only_native_inputs'] = native_ok

    checks['noise_and_optimizer_matches_r18_contract'] = (
        cfg['train']['text_dropout'] == 0.1
        and cfg['train']['accumulation'] == 4
        and cfg['train']['seed'] == 42
        and cfg['model']['lora_rank'] == 32)
    checks['revision'] = cfg['revision'] == REVISION
    failed = [name for name, value in checks.items() if not value]
    report = dict(ok=not failed, failed=failed, checks=checks,
                  counts=counts, data_info=info,
                  compared_to=dict(
                      train_cache=str(r18['train_cache']),
                      t2v_cache_root=str(r18['run_root'] / 't2v_cache')))
    print(json.dumps(report, indent=2, default=str))
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
