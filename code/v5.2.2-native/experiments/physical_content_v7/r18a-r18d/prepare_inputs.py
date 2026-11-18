"""Compact the existing immutable first-image/request Qwen evidence, CPU only."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

import torch

from benchmark_inputs import ROOT, case_input, reference
from physical_content.common import atomic_json, atomic_torch, load_config
from physical_content.evidence import find_input
from physical_content.teacher import Teacher

ENCODING = 'original_qwen_first_image_request_v1'
KEYS = ('words', 'word_groups', 'word_positions', 'word_strings',
        'request_words', 'request_word_strings', 'relations', 'caption',
        'analysis', 'analysis_complete', 'sample', 'scope', 'donor')


def compact(evidence, item, source, *, drop=False):
    caption = '' if drop else item['caption']
    if (evidence['sample'] != item['id'] or evidence['caption'] != caption
            or evidence['scope'] != 'first_image_and_request_only'):
        raise ValueError('Cached evidence does not belong to this own image/request')
    if evidence['request_word_strings'] != caption.split():
        raise ValueError('Literal request span is incomplete')
    if evidence['words'].shape[-1] != 4096 or evidence['request_words'].shape[-1] != 4096:
        raise ValueError('Retain the frozen Qwen4096 feature interface')
    value = {key: evidence[key] for key in KEYS if key in evidence}
    value.update(revision=ENCODING, first_image=item['image'], duration=item['duration'],
                 original_evidence=str(source), intervention='real')
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    teacher = Teacher(cfg, device='cpu')
    destination = Path(cfg['paths']['inference_evidence'])
    records = []
    torch.set_num_threads(4)
    for case in reference()['cases']:
        _, item = case_input(case)
        folder = destination / item['id']
        expected = dict(id=item['id'], case=case['id'], caption=item['caption'],
                        image=item['image'], duration=item['duration'],
                        scope='first_image_and_request_only', width=4096,
                        revision=ENCODING, native_wan_prompt_unchanged=True)
        ready = folder / 'ready.json'
        paths = [folder / 'positive.pt', folder / 'negative.pt']
        if ready.exists() and all(path.is_file() for path in paths):
            saved = json.loads(ready.read_text())
            if any(saved.get(k) != v for k, v in expected.items()):
                raise ValueError('Prepared input has different provenance')
            records.append(saved)
            continue
        sources, counts = {}, {}
        for drop in (False, True):
            arm = 'negative' if drop else 'positive'
            if int(case['id'][2:]) <= 40:
                source = find_input(item['id'] + ('_null' if drop else '_text'))
                evidence = teacher.input(item, drop=drop)
            else:
                source = ROOT / 'videophy60/inputs' / case['id'] / (arm + '.pt')
                evidence = torch.load(source, map_location='cpu', weights_only=True)
                if Path(evidence['input_canvas']).resolve() != Path(item['image']).resolve():
                    raise ValueError('Extra20 evidence uses a different first image')
                teacher.add_request(evidence, item['duration'])
            value = compact(evidence, item, source, drop=drop)
            atomic_torch(value, folder / (arm + '.pt'))
            sources[arm] = str(source)
            counts[arm] = len(value['relations'])
            del evidence, value
        record = dict(**expected, relation_counts=counts, sources=sources)
        atomic_json(record, ready)
        records.append(record)
        print(json.dumps(dict(case=case['id'], status='prepared',
                              complete=len(records))), flush=True)
    report = dict(complete=True, cases=len(records),
                  positive_negative_contexts=2 * len(records),
                  revision=ENCODING, api_calls=0, model_forward_calls=0,
                  new_downloads=0, records=records)
    atomic_json(report, destination / 'ready.json')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}))


if __name__ == '__main__':
    main()
