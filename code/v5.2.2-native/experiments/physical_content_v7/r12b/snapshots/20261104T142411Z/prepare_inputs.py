"""Compact the existing original Qwen inputs; CPU only, no encoding/API call."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from benchmark_inputs import ROOT, reference, case_input
from physical_content.common import load_config, atomic_json, atomic_torch
from physical_content.teacher import Teacher
from physical_content.evidence import find_input

ENCODING = 'original_qwen_first_image_request_v1'
KEYS = ('words', 'word_groups', 'word_positions', 'word_strings',
        'request_words', 'request_word_strings', 'relations', 'caption',
        'analysis', 'analysis_complete', 'sample', 'scope', 'donor')


def compact(evidence, item, source, *, drop=False, mode='real', own_count=None):
    caption = '' if drop else item['caption']
    if (evidence['sample'] != item['id'] or evidence['caption'] != caption
            or evidence['scope'] != 'first_image_and_request_only'):
        raise ValueError('Cached evidence does not belong to this own image/request input')
    if evidence['request_word_strings'] != caption.split():
        raise ValueError('The literal request span is incomplete')
    if evidence['words'].shape[-1] != 4096 or evidence['request_words'].shape[-1] != 4096:
        raise ValueError('Retain the frozen local Qwen4096 feature interface')
    value = {key: evidence[key] for key in KEYS if key in evidence}
    value.update(revision=ENCODING, first_image=item['image'], duration=item['duration'],
                 original_evidence=str(source), intervention=mode)
    if mode in ('reasoning_swapped', 'empty_reasoning'):
        if len(evidence['relations']) != own_count:
            raise ValueError('Reasoning controls must retain the own relation-count prior')
        value['own_relation_count'] = own_count
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    teacher = Teacher(cfg, device='cpu')
    destination = Path(cfg['paths']['inference_evidence'])
    selected = {r['id'] for r in json.loads((Path(cfg['source_snapshot']) / 'subset16.json').read_text())['cases']}
    records = []
    torch.set_num_threads(4)
    for case in reference()['cases']:
        _, item = case_input(case)
        folder = destination / item['id']
        controls = ('reasoning_swapped', 'empty_reasoning') if case['id'] in selected else ()
        expected = dict(id=item['id'], case=case['id'], caption=item['caption'], image=item['image'],
                        duration=item['duration'], scope='first_image_and_request_only', width=4096,
                        revision=ENCODING, native_wan_prompt_unchanged=True, controls=list(controls))
        paths = [folder / 'positive.pt', folder / 'negative.pt']
        paths += [folder / 'controls' / (mode + '.pt') for mode in controls]
        ready = folder / 'ready.json'
        if ready.exists() and all(path.is_file() for path in paths):
            saved = json.loads(ready.read_text())
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError('Prepared input has different provenance')
            records.append(saved)
            continue
        counts = {}
        sources = {}
        for drop in (False, True):
            arm = 'negative' if drop else 'positive'
            if int(case['id'][2:]) <= 40:
                source = find_input(item['id'] + ('_null' if drop else '_text'))
                evidence = teacher.input(item, drop=drop)
            else:
                source = ROOT / 'videophy60/inputs' / case['id'] / (arm + '.pt')
                evidence = torch.load(source, map_location='cpu', weights_only=True)
                if Path(evidence['input_canvas']).resolve() != Path(item['image']):
                    raise ValueError('Extra20 evidence uses a different first image')
                teacher.add_request(evidence, item['duration'])
            value = compact(evidence, item, source, drop=drop)
            atomic_torch(value, folder / (arm + '.pt'))
            counts[arm] = len(value['relations'])
            sources[arm] = str(source)
            del evidence, value
        for mode in controls:
            source = ROOT / 'cache/reasoning_controls' / mode / (case['id'] + '.pt')
            evidence = torch.load(source, map_location='cpu', weights_only=True)
            teacher.add_request(evidence, item['duration'])
            value = compact(evidence, item, source, mode=mode, own_count=counts['positive'])
            atomic_torch(value, folder / 'controls' / (mode + '.pt'))
            sources[mode] = str(source)
            del evidence, value
        record = dict(**expected, relation_counts=counts, sources=sources)
        atomic_json(record, ready)
        records.append(record)
        print(json.dumps(dict(case=case['id'], status='prepared_original_features', complete=len(records))), flush=True)
    report = dict(complete=True, cases=len(records), positive_negative_contexts=2 * len(records),
                  controls=2 * len(selected), revision=ENCODING, api_calls=0,
                  model_forward_calls=0, new_downloads=0, records=records)
    atomic_json(report, destination / 'ready.json')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
