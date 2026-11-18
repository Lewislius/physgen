"""Build the separate request-only Qwen evidence required by T1/T2.

This builder never supplies an image to Qwen and never reads a future video.
It writes compact input/word files under the configured T2V cache roots.  Run
``prepare_native_t2v.py`` afterwards for the T5/Wan native route targets.
"""
from pathlib import Path
import argparse
import json
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap  # noqa: F401

import torch

from benchmark_inputs import reference
from physical_content.common import atomic_json, atomic_torch, load_config

TASK = (
    'The request describes the future video to generate. Give only the physical '
    'dependencies needed to realize that process. Return one to four numbered '
    'sentences, each at most 24 words. Each should connect a relevant cause, '
    'contact, or material condition to an observable response and any essential '
    'property that must persist. Prioritize the requested transition over '
    'incidental details. Do not invent causes, measurements, timing, or unseen '
    'facts. No introduction or law names.'
)


def parse_relations(text):
    values = []
    for line in text.splitlines():
        match = re.match(r'\s*(?:\d+[\).\s]+|[-*]\s+)(.+?)\s*$', line)
        if match:
            values.append(match.group(1).strip())
    if not values:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    values = [value for value in values if value][:4]
    if not values:
        raise ValueError('Qwen returned no physical sentences')
    return values


def word_spans(text, offsets, span, relation, hidden):
    start = text.index(span)
    if relation:
        start = start + len(span) - len(relation)
        end = start + len(relation)
    else:
        end = start + len(span)
    words, positions = [], []
    matches = list(re.finditer(r'\S+', relation)) if relation else []
    if not matches:
        index = ((offsets[:, 1] > start) & (offsets[:, 0] < end)).nonzero().flatten()
        if not len(index):
            raise ValueError('Empty relation slot has no Qwen delimiter token')
        index = index.to(hidden.device)
        return [hidden.index_select(0, index).float().mean(0)], [int(index[0])]
    for match in matches:
        a, b = start + match.start(), start + match.end()
        index = ((offsets[:, 1] > a) & (offsets[:, 0] < b)).nonzero().flatten()
        if not len(index):
            raise ValueError('A physical word has no Qwen token')
        index = index.to(hidden.device)
        words.append(hidden.index_select(0, index).float().mean(0))
        positions.append(int(index[0]))
    return words, positions


@torch.no_grad()
def build_one(model, tokenizer, caption, duration, sample, device, null=False):
    request = 'No text request.' if null else (caption or 'No text request.')
    body = (f'Request: {request}\n'
            f'Presentation interval: {duration:.3f} seconds.\n')
    if null:
        relations = ['', '', '', '']
        analysis = ''
    else:
        prompt = body + TASK
        rendered = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True)
        encoded = tokenizer(rendered, return_tensors='pt', add_special_tokens=False)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded, max_new_tokens=192, do_sample=False, top_k=None,
            top_p=None, temperature=None, use_cache=True,
            pad_token_id=tokenizer.eos_token_id)
        analysis = tokenizer.decode(
            generated[0, encoded['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        relations = parse_relations(analysis)
    spans = [f'[RELATION {i + 1}] {value}' for i, value in enumerate(relations)]
    feature_text = (f'Request: {request}\n'
                    f'Presentation interval: {duration:.3f} seconds.\n'
                    'Conditional physical analysis:\n' + '\n'.join(spans))
    encoded = tokenizer(feature_text, add_special_tokens=False,
                        return_offsets_mapping=True, return_tensors='pt')
    offsets = encoded.pop('offset_mapping')[0]
    ids = encoded['input_ids'].to(device)
    mask = encoded['attention_mask'].to(device)
    hidden = model.model(input_ids=ids, attention_mask=mask,
                         use_cache=False, return_dict=True).last_hidden_state[0]
    words, groups, positions, strings = [], [], [], []
    for group, (span, relation) in enumerate(zip(spans, relations)):
        values, where = word_spans(feature_text, offsets, span, relation, hidden)
        relation_words = relation.split() if relation else ['']
        if len(values) != len(relation_words):
            raise ValueError('Relation word count changed during tokenization')
        for value, position, word in zip(values, where, relation_words):
            words.append(value)
            groups.append(group)
            positions.append(position)
            strings.append(word)
    caption_start = feature_text.index(f'Request: ') + len('Request: ')
    caption_end = caption_start + len(request)
    request_words, request_strings = [], []
    if not null:
        for match in re.finditer(r'\S+', request):
            a, b = caption_start + match.start(), caption_start + match.end()
            index = ((offsets[:, 1] > a) & (offsets[:, 0] < b)).nonzero().flatten()
            if not len(index):
                raise ValueError('A request word has no Qwen token')
            index = index.to(hidden.device)
            request_words.append(hidden.index_select(0, index).float().mean(0))
            request_strings.append(match.group())
    value = dict(
        sample=sample, caption='' if null else caption, relations=relations,
        analysis=analysis, analysis_complete=not null,
        words=torch.stack(words), word_groups=torch.tensor(groups, dtype=torch.long),
        word_positions=torch.tensor(positions, dtype=torch.long), word_strings=strings,
        request_words=(torch.stack(request_words) if request_words else
                       torch.empty(0, hidden.shape[-1])),
        request_word_strings=request_strings, scope='request_only', duration=duration,
        model='Qwen3-VL-8B-Instruct', revision='NoChunks-v2-20261111')
    return value


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--eval-only', action='store_true')
    args = parser.parse_args(argv)
    cfg = load_config(args.config, require_paths=True)
    if cfg['task'] != 't2v':
        raise ValueError('This builder is only for T2V plans')
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    from transformers.utils import logging as transformer_logging
    transformer_logging.disable_progress_bar()
    tokenizer = AutoTokenizer.from_pretrained(cfg['paths']['qwen_checkpoint'],
                                              local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg['paths']['qwen_checkpoint'], dtype=torch.bfloat16,
        local_files_only=True, attn_implementation='sdpa',
        device_map={'': args.device}, low_cpu_mem_usage=True).eval()
    model.requires_grad_(False)
    roots = {
        'input': Path(cfg['paths']['input_cache']),
        'words': Path(cfg['paths']['words_cache']),
        'eval_input': Path(cfg['paths']['eval_input_cache']),
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    rows = []
    if not args.eval_only:
        cache = Path(cfg['paths']['cache'])
        manifest = json.loads((cache / 'base/manifest.json').read_text())['records']
        paired = Path(cfg['paths']['paired_words_cache']) if cfg['paths'].get(
            'paired_words_cache') else None
        rows = [dict(id=row['id'], caption=row['caption'],
                     duration=float(row['pts'][80] - row['pts'][0]))
                for row in manifest if row['split'] == 'train' and row['caption']]
        if paired is not None and paired.is_dir():
            rows = [row for row in rows
                    if (paired / f'{row["id"]}_text.pt').is_file()]
    cases = [dict(id=f'{"videophy40" if int(case["id"][2:]) <= 40 else "videophy60"}_{case["id"]}',
                  caption=case['caption'], duration=80 / 24)
             for case in reference()['cases']]
    work = []
    for row in rows:
        work.append((row, roots['input'], roots['words']))
    for row in cases:
        work.append((row, roots['eval_input'], roots['words']))
    if args.limit:
        work = work[:args.limit]
    started = time.monotonic()
    for index, (row, root, words_root) in enumerate(work):
        for null in (False, True):
            suffix = '_null' if null else '_text'
            destination = root / f'{row["id"]}{suffix}.pt'
            if destination.is_file():
                continue
            value = build_one(model, tokenizer, row['caption'], row['duration'],
                              row['id'], args.device, null=null)
            atomic_torch(value, destination)
            atomic_torch({key: value[key] for key in
                          ('words', 'word_groups', 'word_positions', 'word_strings')},
                         words_root / f'{row["id"]}{suffix}.pt')
        if index % 20 == 0:
            print(json.dumps(dict(stage='t2v_evidence', completed=index + 1,
                                  total=len(work), seconds=time.monotonic() - started)),
                  flush=True)
    atomic_json(dict(complete=not bool(args.limit), records=len(work),
                     revision='NoChunks-v2-20261111', scope='request_only',
                     image_given_to_qwen=False, seconds=time.monotonic() - started),
                roots['input'].parent / ('partial_ready.json' if args.limit else 'ready.json'))
    print(json.dumps(dict(complete=True, records=len(work))))


if __name__ == '__main__':
    main()
