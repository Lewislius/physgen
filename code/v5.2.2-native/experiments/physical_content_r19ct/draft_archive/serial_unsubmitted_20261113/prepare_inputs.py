"""Compact the fixed 60 evaluation evidence into per-case positive/negative arms."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch

from benchmark_inputs import case_input, reference
from condition_controls import ENCODING
from physical_content.common import atomic_json, atomic_torch, load_config
from physical_content.evidence import recover_request_words, recover_words

KEYS = ('words', 'word_groups', 'word_positions', 'word_strings',
        'request_words', 'request_word_strings', 'relations', 'caption',
        'analysis', 'analysis_complete', 'sample', 'scope', 'donor',
        'first_image')


def _load_arm(source, tokenizer, duration, drop):
    evidence = torch.load(source, map_location='cpu', weights_only=True)
    if 'request_words' not in evidence:
        if 'memory' not in evidence:
            raise ValueError(f'{source} lacks request spans')
        evidence.update(recover_request_words(evidence, duration, tokenizer))
    if 'words' not in evidence:
        if 'memory' not in evidence:
            raise ValueError(f'{source} lacks word features')
        evidence.update(recover_words(evidence, duration, tokenizer))
    value = {key: evidence[key] for key in KEYS if key in evidence}
    value.update(revision=ENCODING, duration=duration,
                 sample=evidence['sample'], caption=('' if drop else evidence['caption']))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    task = cfg['task']
    destination = Path(cfg['paths']['inference_evidence'])
    tokenizer = None
    if task == 'i2v':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cfg['paths']['qwen_checkpoint'], local_files_only=True)
    records = []
    torch.set_num_threads(4)
    for case in reference()['cases']:
        sample, item = case_input(case, task)
        folder = destination / item['id']
        scope = cfg['input']['evidence_scope']
        expected = dict(id=item['id'], case=case['id'], caption=item['caption'],
                        duration=item['duration'], scope=scope, revision=ENCODING)
        if task == 'i2v':
            expected['image'] = item['image']
        ready = folder / 'ready.json'
        paths = [folder / 'positive.pt', folder / 'negative.pt']
        if ready.exists() and all(path.is_file() for path in paths):
            saved = json.loads(ready.read_text())
            if any(saved.get(key) != value for key, value in expected.items()):
                raise ValueError('Prepared input has different provenance')
            records.append(saved)
            continue
        source_root = Path(cfg['paths']['eval_input_cache'])
        sources, counts = {}, {}
        for drop in (False, True):
            arm = 'negative' if drop else 'positive'
            if task == 't2v' or int(case['id'][2:]) <= 40:
                source = source_root / f'{item["id"]}_{"null" if drop else "text"}.pt'
            else:
                source = (Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/'
                               'experiments/physical_content_v7/videophy60/inputs/'
                               f'{case["id"]}/{arm}.pt'))
            value = _load_arm(source, tokenizer, item['duration'], drop)
            if task == 'i2v':
                value['first_image'] = item['image']
            if value['sample'] != item['id'] or value['scope'] != scope:
                raise ValueError('Evaluation evidence has the wrong scope/owner')
            if task == 'i2v':
                if ('first_image' not in value or
                        Path(value['first_image']).resolve() != Path(item['image']).resolve()):
                    raise ValueError('Evaluation evidence uses another first image')
            atomic_torch(value, folder / f'{arm}.pt')
            sources[arm] = str(source)
            counts[arm] = len(value['relations'])
        record = dict(**expected, relation_counts=counts, sources=sources)
        atomic_json(record, folder / 'ready.json')
        records.append(record)
        print(json.dumps(dict(case=case['id'], status='prepared',
                              complete=len(records))), flush=True)
    atomic_json(dict(complete=True, cases=len(records),
                     positive_negative_contexts=2 * len(records),
                     revision=ENCODING, api_calls=0, model_forward_calls=0,
                     records=records), destination / 'ready.json')
    print(json.dumps(dict(complete=True, cases=len(records),
                          revision=ENCODING)))


if __name__ == '__main__':
    main()
