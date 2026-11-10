"""Official VideoPhy2 AutoEval on the 600 ALL / 180 HARD prompt list."""
from pathlib import Path
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, aggregate, gpu_guard, physvid_score, plan, save


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--partial', action='store_true')
    args = p.parse_args()
    protocol = plan()
    spec = protocol['model']
    cfg = protocol['evaluation']
    gpu_guard(ROOT / 'jobs' / os.getenv('DET_EXPERIMENT_ID', 'local'))
    sys.path.insert(0, cfg['code'])
    import torch
    from transformers.models.llama.tokenization_llama import LlamaTokenizer
    from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
    from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
    from template import PROMPT_SA, PROMPT_PHYSICS

    torch.set_num_threads(4)
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    checkpoint = cfg['checkpoint']
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint, local_files_only=True)
    processor = MplugOwlProcessor(
        MplugOwlImageProcessor.from_pretrained(checkpoint, local_files_only=True), tokenizer)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        checkpoint, torch_dtype=torch.bfloat16, device_map={'': 'cpu'},
        local_files_only=True).eval().cuda()

    out = ROOT / 'outputs' / spec['name']
    rows = []
    for case in protocol['cases']:
        folder = out / 'cases' / case['id']
        if not (folder / 'video.json').exists():
            if args.partial:
                continue
            raise ValueError('Missing declared video ' + case['id'])
        row = dict(id=case['id'], caption=case['caption'], is_hard=case['is_hard'],
                   source_case=case['source_id'])
        for task in ('sa', 'pc'):
            path = folder / (task + '.json')
            if path.exists():
                record = json.loads(path.read_text())
            else:
                prompt = (PROMPT_SA.format(caption=case['caption'].strip())
                          if task == 'sa' else PROMPT_PHYSICS)
                inputs = processor(text=[prompt], videos=[str(folder / 'video.mp4')],
                                   num_frames=cfg['num_frames'], return_tensors='pt')
                inputs = {k: (v.to(torch.bfloat16) if v.dtype == torch.float32 else v).cuda()
                          for k, v in inputs.items()}
                with torch.inference_mode():
                    result = model.generate(**inputs, **cfg['generation'])
                raw = tokenizer.decode(result[0], skip_special_tokens=True)
                record = dict(task=task, raw=raw, score=physvid_score(raw),
                              case=case['id'], evaluator=cfg)
                save(path, record)
            row[task] = record['score']
        rows.append(row)
        print(json.dumps(row), flush=True)

    all_summary = aggregate(rows)
    hard_rows = [r for r in rows if str(r['is_hard']) == '1']
    hard_summary = aggregate(hard_rows) if hard_rows else None
    result = dict(summary=all_summary, hard=hard_summary,
                  cases=rows, scope=protocol['scope'], model=spec,
                  definitions=dict(sa='SA>=4', pc='PC>=4',
                                   joint='SA>=4 and PC>=4',
                                   threshold=4),
                  prompt_source=protocol['source'])
    save(out / 'scores.json', result)
    with (out / 'scores.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(ROOT / 'summary.json', result)
    print(json.dumps(dict(stage='completed', summary=all_summary, hard=hard_summary)), flush=True)


if __name__ == '__main__':
    main()
