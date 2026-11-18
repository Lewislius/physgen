"""Add immutable first-image tokens to the already prepared CI I2V conditions.

No new API call or model encoding. Physical words/native anchors, own-image
re-encoded controls, original request, image and random inputs are preserved.
"""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap
import torch
from transformers import AutoTokenizer
from benchmark_inputs import case_input, reference
from physical_content.common import atomic_json, atomic_torch, load_config
from physical_content.evidence import recover_visual_tokens
from run_candidate import PAIRED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    source = Path(cfg['paths']['prepared_evidence_reference'])
    destination = Path(cfg['paths']['inference_evidence'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['paths']['qwen_checkpoint'], local_files_only=True)
    torch.set_num_threads(4)
    records = []
    for case in reference()['cases']:
        _, item = case_input(case, 'i2v')
        src, dst = source/item['id'], destination/item['id']
        ready = json.loads((src/'ready.json').read_text())
        if ready['image'] != item['image'] or ready['caption'] != item['caption']:
            raise ValueError('The shared I2V image/request changed')
        if (dst/'ready.json').exists():
            old = json.loads((dst/'ready.json').read_text())
            if old.get('scene_encoding') != 'causal_Qwen_image_tokens_v1':
                raise ValueError('Different visual encoding already exists')
            records.append(old)
            continue
        visual = None
        arm_visual_relative_difference = None
        for arm in ('positive', 'negative'):
            evidence = torch.load(src/f'{arm}.pt', map_location='cpu', weights_only=True)
            raw = torch.load(ready['sources'][arm], map_location='cpu', weights_only=True)
            if raw['sample'] != item['id'] or raw['caption'] != evidence['caption']:
                raise ValueError('Visual input owner/request differs')
            recovered = recover_visual_tokens(raw, item['duration'], tokenizer)
            if visual is None:
                visual = recovered
            else:
                if visual['visual_tokens'].shape != recovered['visual_tokens'].shape:
                    raise ValueError('The same first image has different token geometry')
                original = visual['visual_tokens'].float()
                arm_visual_relative_difference = float(
                    (original-recovered['visual_tokens'].float()).norm()/original.norm().clamp_min(1e-8))
            evidence.update(recovered)
            atomic_torch(evidence, dst/f'{arm}.pt')
        if case['id'] in PAIRED:
            for mode in ('empty_reasoning', 'reasoning_swapped'):
                evidence = torch.load(src/'controls'/f'{mode}.pt', map_location='cpu', weights_only=True)
                if evidence['first_image'] != item['image'] or evidence['caption'] != item['caption']:
                    raise ValueError('Control must retain the current image and request')
                # Causal image tokens precede request/reasoning. Changing the
                # latter does not change the meaning/provenance of the former.
                evidence.update(visual)
                atomic_torch(evidence, dst/'controls'/f'{mode}.pt')
        ready.update(scene_encoding='causal_Qwen_image_tokens_v1',
                     visual_token_count=visual['visual_token_count'],
                     cfg_arm_visual_relative_difference=arm_visual_relative_difference)
        atomic_json(ready, dst/'ready.json')
        records.append(ready)
    report = dict(complete=True, cases=len(records), normal_cfg_arms=120,
                  controls=2*len(PAIRED), api_calls=0, model_forward_calls=0,
                  source=str(source), first_image_features_are_causal=True,
                  records=records)
    atomic_json(report, destination/'ready.json')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}), flush=True)


if __name__ == '__main__':
    main()
