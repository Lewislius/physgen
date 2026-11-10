"""Inspect the native-value interface on four existing matched250 examples.

Loads only the enhancement module and cached inputs on CPU; no Wan/Qwen
backbone, generation, scoring, training, or downloads.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F


def metrics(value, anchor, target, scale):
    value, anchor = value.float()[0], anchor.float()[0]
    correction = value - anchor
    normed = F.normalize(value, dim=-1)
    n = len(value)
    rms = lambda x: float(x.square().mean().sqrt())
    result = dict(c_rms=rms(value), anchor_rms=rms(anchor),
                  correction_rms=rms(correction),
                  correction_token_mean_rms=rms(correction.mean(0)),
                  centered_correction_rms=rms(correction-correction.mean(0)),
                  correction_to_anchor_rms=rms(correction)/max(rms(anchor), 1e-12),
                  c_token_cosine=float((normed.sum(0).square().sum()-n)/(n*(n-1))))
    if target is not None:
        result['native_word_target_mse'] = float((value-target.float()[0]).square().mean()/scale.square())
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    sys.path.insert(0, str(args.source))
    import bootstrap
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    from physical_content.evidence import resample_words
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = state['config']
    if state['step'] != 250:
        raise ValueError('This comparison uses the existing own250 checkpoints')
    model = Enhancement(cfg).eval().requires_grad_(False)
    values = {k.removeprefix('enhancement.'): v
              for k, v in {**state['model'], **state['constants']}.items()
              if k.startswith('enhancement.')}
    model.load_state_dict(values, strict=True)
    model.checkpointing = False
    teacher = Teacher(cfg, device='cpu')
    rows = []
    for case in ['VP010', 'VP026', 'VP027', 'VP038']:
        conditions = {}
        record = dict(id=case, arms={})
        for arm, drop in [('positive', False), ('negative', True)]:
            evidence = teacher.input(dict(id='videophy40_'+case), drop=drop)
            words = (evidence['words'].float()-model.anchor_mean)/model.anchor_scale
            anchor, groups, _ = resample_words(words, evidence['word_groups'], model.seed.shape[1])
            target = None
            if 'native_words' in evidence:
                target = resample_words(evidence['native_words'].float(), evidence['word_groups'], model.seed.shape[1])[0]
            with torch.autocast('cpu', dtype=torch.bfloat16):
                native = F.linear(anchor, model.anchor_weight, model.anchor_bias)
                content, _ = model(evidence)
            if not isinstance(content, dict):
                content = dict(shared=content)
            conditions[arm] = {k: v.float() for k, v in content.items()}
            record['arms'][arm] = dict(
                available_multimodal_tokens=evidence['memory'].shape[1],
                physical_words=len(evidence['words']),
                explicit_request_words=len(evidence.get('request_words', [])),
                memories={k: metrics(v, native, target, model.anchor_target_scale)
                          for k, v in content.items()})
        record['positive_minus_negative_c_rms'] = {
            k: float((conditions['positive'][k]-conditions['negative'][k]).square().mean().sqrt())
            for k in conditions['positive']}
        rows.append(record)
        print(json.dumps(dict(label=args.label, completed_case=case)), flush=True)
    result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), label=args.label,
                  source=str(args.source), checkpoint=str(args.checkpoint), step=250,
                  selection='Existing VP010/026/027 visual examples plus VP038, a PC gain versus R1; not a population estimate.',
                  scope='CPU enhancement outputs only. Raw C scale does not alone establish actual Wan write strength or causal attribution of quality.',
                  backbones_loaded=False, new_generation_or_scoring=False, records=rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output))), flush=True)


if __name__ == '__main__':
    main()
