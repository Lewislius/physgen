"""Local input-only physical reasoning; the feature encoder is unchanged.

Produces text records only. Completed reasoning is reviewed before rebuilding
Qwen3-VL-Instruct and native Wan features or using it in a training run.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

BASE = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native')
REVISION = 'R20-entity-grounded-local-thinking-v1'
INSTRUCTIONS = """Prepare concise physical conditioning for the requested video.
The original request is authoritative. Preserve its subjects, participants,
objects, attributes, relationships and visible process. A person or tool causing
an interaction must remain associated with the affected object; do not replace
the requested scene by a close-up of an abstract material or physical property.

Return 1 to 4 concise English sentences, preferably no more than 28 words each.
Connect the named participants and objects to the relevant cause, interaction,
observable response, and properties that should persist. Include only physically
sound dependencies needed for this request. Resolve force and motion directions
carefully. Distinguish elastic response from permanent yielding or breakage.
Cover liquids, gases, light and deformable materials naturally when relevant;
do not assume all phenomena are rigid-body motion or human actions.

This is a bounded video clip and may start or end in the middle of a process.
Do not invent events before or after the clip to complete a story. Do not invent
colors, materials, agents, camera movements, measurements, or event timestamps.
Do not add microscopic explanations that are not necessary for the visible
process. Avoid generic law recitations, repeated rules, and instructions to
freeze the scene. Keep static conditions static only when the request says so.
Check your final sentences for unsupported assumptions and incorrect causality.

After your reasoning, give only a JSON object: {"relations": ["sentence", ...]}.
The sentences must be usable descriptions, without headings or numbering.
"""


def final_relations(text):
    # A Thinking model's unfinished analysis must never become physical C.
    if '</think>' not in text:
        raise ValueError('Thinking did not finish; no final conditioning available')
    final = text.rsplit('</think>', 1)[1]
    for marker in ('<|im_end|>', '<|endoftext|>'):
        final = final.replace(marker, '')
    first, last = final.find('{'), final.rfind('}')
    if first < 0 or last <= first:
        raise ValueError('Missing final JSON')
    value = json.loads(final[first:last + 1])
    rows = value.get('relations')
    if not isinstance(rows, list) or not 1 <= len(rows) <= 4:
        raise ValueError('Expected one to four physical statements')
    if any(not isinstance(row, str) or not row.strip() for row in rows):
        raise ValueError('Empty physical statement')
    rows = [' '.join(row.split()) for row in rows]
    if len(set(rows)) != len(rows):
        raise ValueError('Duplicated physical statements')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', default='/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Thinking')
    parser.add_argument('--max-new-tokens', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    rows = selection['cases']
    if selection['scope'] != 'request_only' or any(not row['caption'] for row in rows):
        raise ValueError('This revision accepts request-only T2V inputs')
    args.output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(BASE/'tools'))
    from gpu_policy import require_rtx_pro_6000
    actual_gpu = require_rtx_pro_6000()
    import torch
    from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
    from transformers.utils import logging
    logging.disable_progress_bar()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={'': 'cuda'},
        attn_implementation='sdpa', local_files_only=True,
        low_cpu_mem_usage=True).eval().requires_grad_(False)
    started = time.monotonic()
    completed, errors = [], []
    pending = []
    for index, row in enumerate(rows):
        destination = args.output/(row['id']+'.json')
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved['caption'] != row['caption'] or saved['revision'] != REVISION:
                raise ValueError('Existing text belongs to a different input or policy')
            completed.append(row['id'])
            continue
        pending.append((index, row))

    def generate(items, budget):
        prompts = [INSTRUCTIONS + '\nOriginal request:\n' + row['caption'] for _, row in items]
        rendered = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True) for prompt in prompts]
        encoded = tokenizer(rendered, add_special_tokens=False, padding=True,
                            return_tensors='pt').to('cuda')
        torch.manual_seed(42 + items[0][0])
        torch.cuda.manual_seed_all(42 + items[0][0])
        with torch.inference_mode():
            result = model.generate(**encoded, max_new_tokens=budget,
                do_sample=True, temperature=1.0, top_p=.95, top_k=20,
                repetition_penalty=1.0, use_cache=True)
        outputs = []
        for ids in result[:, encoded['input_ids'].shape[1]:]:
            end = (ids == tokenizer.eos_token_id).nonzero().flatten()
            if len(end):
                ids = ids[:int(end[0])+1]
            outputs.append((tokenizer.decode(ids, skip_special_tokens=False), len(ids)))
        return prompts, outputs

    for start in range(0, len(pending), args.batch_size):
        part = pending[start:start+args.batch_size]
        began = time.monotonic()
        prompts, outputs = generate(part, args.max_new_tokens)
        for item, prompt, output in zip(part, prompts, outputs):
            index, row = item
            text, token_count = output
            attempt = 1
            try:
                relations = final_relations(text)
            except (ValueError, TypeError):
                _, retry = generate([item], args.max_new_tokens*2)
                text, token_count = retry[0]
                attempt = 2
                try:
                    relations = final_relations(text)
                except (ValueError, TypeError) as exc:
                    errors.append(dict(id=row['id'], error=str(exc)))
                    continue
            record = dict(row, relations=relations,
                analysis='\n'.join(f'{i+1}. {s}' for i, s in enumerate(relations)),
                scope='request_only', revision=REVISION, model=str(args.model),
                prompt=prompt, generated_tokens=token_count, attempt=attempt,
                batch_seconds=time.monotonic()-began, batch_size=len(part),
                analysis_complete=True, final_answer_only=True,
                native_request_unchanged=True,
                word_counts=[len(re.findall(r'\S+', s)) for s in relations])
            (args.output/(row['id']+'.json')).write_text(
                json.dumps(record, ensure_ascii=False, indent=2)+'\n')
            completed.append(row['id'])
        status = dict(revision=REVISION, total=len(rows), completed=len(completed),
            failures=errors, seconds=time.monotonic()-started, gpu=actual_gpu,
            api_calls=0, feature_encoder='/home/liuzhirui/model/Qwen3-VL/Qwen3-VL-8B-Instruct',
            feature_rebuild_required=True, human_review_required=True)
        (args.output/'progress.json').write_text(json.dumps(status, indent=2)+'\n')
        print(json.dumps(dict(completed=len(completed), total=len(rows),
                              failures=len(errors))), flush=True)
    if errors:
        raise RuntimeError('Some inputs lack a completed final reasoning answer; see progress.json')


if __name__ == '__main__':
    main()
