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
REVISION = 'R20-grounded-physical-description-v4'
INSTRUCTIONS = """Write a short, accurate physical guide for generating the
requested video. Preserve the scene, principal participants, affected objects,
and the relationships between them. Explain the visible physical process in
plain language, with correctly directed causes and responses. Do not replace
the scene with an abstract material demonstration.

Use 1 to 4 English sentences, each a separate list item of at most 35 words.
For a detailed request, cover its principal changes across the clip rather
than just its opening event. One sentence is sufficient for a simple event.
Keep each participant connected to what it does or experiences.

Be grounded: retain stated attributes, but do not invent colors, materials,
agents, camera behavior, additional events, or precise material properties.
Do not impose unmentioned restrictions on motion, contact, shape, breakage,
or the aftermath. Do not add generic claims that everything stays intact or
stationary. Include only physically sound information needed to realize the
requested process. Unnecessary microscopic explanations and technical labels
should be omitted. These clips can start or end mid-process; do not invent
preceding or following events. The request remains authoritative.

Return only one JSON object: {"relations": ["sentence", ...]}.
"""


def final_relations(text):
    # A Thinking model's unfinished analysis must never become physical C.
    if '</think>' not in text:
        raise ValueError('Thinking did not finish; no final conditioning available')
    final = text.rsplit('</think>', 1)[1]
    for marker in ('<|im_end|>', '<|endoftext|>'):
        final = final.replace(marker, '')
    first = final.find('{')
    if first < 0:
        raise ValueError('Missing final JSON')
    value, _ = json.JSONDecoder().raw_decode(final[first:])
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
    parser.add_argument('--thinking-tokens', type=int, default=768)
    parser.add_argument('--final-tokens', type=int, default=384)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--causal-lm', action='store_true')
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
    from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3VLForConditionalGeneration
    from transformers.utils import logging
    logging.disable_progress_bar()
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer.padding_side = 'left'
    model_class = AutoModelForCausalLM if args.causal_lm else Qwen3VLForConditionalGeneration
    model = model_class.from_pretrained(
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

    eos = model.generation_config.eos_token_id
    end_ids = list(eos) if isinstance(eos, (tuple,list)) else [eos]
    think_end = tokenizer.convert_tokens_to_ids('</think>')

    def generate(rendered, budget, seed, stop_ids):
        encoded = tokenizer(rendered, add_special_tokens=False, padding=True,
                            return_tensors='pt').to('cuda')
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with torch.inference_mode():
            result = model.generate(**encoded, max_new_tokens=budget,
                do_sample=True, temperature=model.generation_config.temperature,
                top_p=model.generation_config.top_p, top_k=model.generation_config.top_k,
                repetition_penalty=1.0, use_cache=True, eos_token_id=stop_ids)
        outputs = []
        for ids in result[:, encoded['input_ids'].shape[1]:]:
            end = torch.isin(ids, torch.tensor(stop_ids, device=ids.device)).nonzero().flatten()
            if len(end):
                ids = ids[:int(end[0])+1]
            outputs.append((tokenizer.decode(ids, skip_special_tokens=False), len(ids)))
        return outputs

    for start in range(0, len(pending), args.batch_size):
        part = pending[start:start+args.batch_size]
        began = time.monotonic()
        prompts = [INSTRUCTIONS + '\nOriginal request:\n' + row['caption'] for _, row in part]
        rendered = [tokenizer.apply_chat_template(
            [{'role': 'user', 'content': prompt}], tokenize=False,
            add_generation_prompt=True) for prompt in prompts]
        if args.thinking_tokens:
            thoughts = generate(rendered, args.thinking_tokens, 42+part[0][0],
                                end_ids+[think_end])
            final_inputs, endings = [], []
            for prefix, (thought, _) in zip(rendered, thoughts):
                natural = '</think>' in thought
                thought = thought.split('</think>', 1)[0]
                for marker in ('<|im_end|>', '<|endoftext|>'):
                    thought = thought.replace(marker, '')
                final_inputs.append(prefix+thought+'\n</think>\n\n')
                endings.append('natural' if natural else 'budget')
        else:
            thoughts = [('',0)]*len(part)
            final_inputs, endings = rendered, ['not_applicable']*len(part)
        outputs = generate(final_inputs, args.final_tokens, 1042+part[0][0], end_ids)
        for item, prompt, output, thought, ending in zip(part, prompts, outputs, thoughts, endings):
            index, row = item
            text, token_count = output
            attempt = 1
            try:
                relations = final_relations('</think>\n'+text)
            except (ValueError, TypeError) as exc:
                errors.append(dict(id=row['id'], error=str(exc)))
                (args.output/(row['id']+'.failed.txt')).write_text(text)
                continue
            record = dict(row, relations=relations,
                analysis='\n'.join(f'{i+1}. {s}' for i, s in enumerate(relations)),
                scope='request_only', revision=REVISION, model=str(args.model),
                prompt=prompt, generated_tokens=token_count+thought[1], attempt=attempt,
                thinking_tokens=thought[1], final_tokens=token_count,
                thinking_end=ending,
                batch_seconds=time.monotonic()-began, batch_size=len(part),
                analysis_complete=(ending!='budget'), final_answer_complete=True,
                final_answer_only=True,
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
