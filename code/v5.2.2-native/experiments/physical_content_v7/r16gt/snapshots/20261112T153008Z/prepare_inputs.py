"""Compact request-only VideoPhy2 60 evidence for R16Gt inference.

CPU only; reads the audited request-only Qwen cache and writes positive/
negative conditions plus provenance.  No image, canvas, first latent, future
frame, score, or physical label is read or written.
"""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch

from benchmark_inputs import reference
from condition_controls import ENCODING
from physical_content.common import (SOURCE_EVIDENCE_REVISION, atomic_json,
                                     atomic_torch, load_config)

KEYS = ('words', 'word_groups', 'word_positions', 'word_strings',
        'request_words', 'request_word_strings', 'relations', 'caption',
        'analysis', 'analysis_complete', 'sample', 'scope', 'duration')


def verify(value, *, name, case, drop):
    if value.get('scope') != 'request_only':
        raise ValueError(f'{name}: evidence is not request-only')
    if value.get('revision') != SOURCE_EVIDENCE_REVISION:
        raise ValueError(f'{name}: unexpected evidence revision')
    if value.get('sample') != name:
        raise ValueError(f'{name}: evidence sample identity differs')
    caption = '' if drop else case['caption']
    if value.get('caption') != caption:
        raise ValueError(f'{name}: caption differs')
    if value.get('request_word_strings') != ([] if drop else caption.split()):
        raise ValueError(f'{name}: literal request span is incomplete')
    if not 1 <= len(value['relations']) <= 4:
        raise ValueError(f'{name}: relation count outside 1..4')
    if value['words'].shape[-1] != 4096 or value['request_words'].shape[-1] != 4096:
        raise ValueError(f'{name}: not the frozen Qwen4096 interface')
    if len(value['words']) != len(value['word_groups']):
        raise ValueError(f'{name}: word/group count differs')
    if int(value['word_groups'].max()) >= len(value['relations']):
        raise ValueError(f'{name}: word group exceeds relation count')
    if abs(float(value['duration']) - 80 / 24) > 1e-3:
        raise ValueError(f'{name}: duration differs from the fixed protocol')
    if not all(bool(torch.isfinite(value[key]).all())
               for key in ('words', 'request_words')):
        raise ValueError(f'{name}: nonfinite feature')


def compact(value, *, name, case, drop, source):
    return dict(
        **{key: value[key] for key in KEYS if key in value},
        id=name, case=case['id'], revision=ENCODING,
        native_wan_prompt_unchanged=True, width=4096,
        source=str(source))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    source_root = Path(cfg['paths']['t2v_eval_cache'])
    destination = Path(cfg['paths']['inference_evidence'])
    native_destination = Path(cfg['paths']['eval_native_inputs'])
    protocol = reference(cfg)
    torch.set_num_threads(4)
    records = []
    for case in protocol['cases']:
        suite = 'videophy40' if int(case['id'][2:]) <= 40 else 'videophy60'
        name = f'{suite}_{case["id"]}'
        folder = destination / name
        expected = dict(id=name, case=case['id'], caption=case['caption'],
                        duration=80 / 24, scope='request_only', width=4096,
                        revision=ENCODING, native_wan_prompt_unchanged=True)
        native_folder = native_destination / case['id']
        native_file = native_folder / 'conditions.pt'
        native_ready = native_folder / 'input.json'
        if not (native_file.is_file() and native_ready.is_file()):
            source_conditions = Path(case['prepared_inputs']) / 'conditions.pt'
            stored = torch.load(source_conditions, map_location='cpu',
                                weights_only=True, mmap=True)
            # Copy only the native T5/time tensors; never first/first.png.
            native = {key: stored[key].float()
                      for key in ('text', 'null_text', 'times')}
            text, null, times = native['text'], native['null_text'], native['times']
            if (text.ndim != 3 or text.shape[0] != 1
                    or not 0 < text.shape[1] <= 512 or text.shape[-1] != 4096
                    or null.ndim != 3 or null.shape[0] != 1
                    or null.shape[-1] != 4096
                    or tuple(times.shape) != (21,)):
                raise ValueError(f'{case["id"]}: invalid native T5/time input')
            if not torch.allclose(times, torch.arange(21).float() / 6,
                                  atol=1e-5, rtol=0):
                raise ValueError(f'{case["id"]}: native time grid differs')
            if not all(bool(torch.isfinite(value).all())
                       for value in native.values()):
                raise ValueError(f'{case["id"]}: nonfinite native input')
            atomic_torch(native, native_file)
            atomic_json(dict(id=case['id'], case=case['id'],
                             prompt=case['caption'],
                             source=str(source_conditions),
                             keys=['text', 'null_text', 'times'],
                             image=False, first_latent=False), native_ready)
        paths = [folder / 'positive.pt', folder / 'negative.pt']
        ready = folder / 'ready.json'
        if ready.exists() and all(path.is_file() for path in paths):
            saved = json.loads(ready.read_text())
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError(f'{name}: prepared input has different provenance')
            records.append(saved)
            continue
        counts = {}
        sources = {}
        for drop in (False, True):
            arm = 'negative' if drop else 'positive'
            source = source_root / f'{name}{"_null" if drop else "_text"}.pt'
            if not source.is_file():
                raise FileNotFoundError(source)
            value = torch.load(source, map_location='cpu', weights_only=True)
            verify(value, name=name, case=case, drop=drop)
            atomic_torch(compact(value, name=name, case=case, drop=drop,
                                 source=source), folder / f'{arm}.pt')
            counts[arm] = len(value['relations'])
            sources[arm] = str(source)
            del value
        record = dict(**expected, relation_counts=counts, sources=sources)
        atomic_json(record, ready)
        records.append(record)
        print(json.dumps(dict(case=case['id'], status='prepared_request_only',
                              complete=len(records))), flush=True)
    report = dict(complete=True, cases=len(records), conditions=2 * len(records),
                  revision=ENCODING, source_revision=SOURCE_EVIDENCE_REVISION,
                  scope='request_only', image_given_to_qwen=False,
                  first_frame_written=False, api_calls=0, new_downloads=0)
    atomic_json(report, destination / 'ready.json')
    atomic_json(dict(complete=True, cases=len(records),
                     keys=['text', 'null_text', 'times'],
                     image=False, first_latent=False),
                native_destination / 'ready.json')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
