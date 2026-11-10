"""Evaluate existing exported videos without adding or changing training data."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prepare(run, output):
    original = ROOT.parent/'v5.2/inference_outputs/videophy2_lora_run64543_joint400_60_seed42/manifest.json'
    evaluation = json.loads(original.read_text())['evaluation']
    # WISA captions are longer than the historical benchmark prompts. Give the
    # evaluator an output-token budget; never truncate physical conditions to fit
    # a max_length that incorrectly includes the multimodal input embeddings.
    evaluation['generation'] = dict(do_sample=False,max_new_tokens=16)
    evaluation['output_budget_definition'] = '16 newly generated score tokens, identical for every arm'
    groups = {}
    for path in sorted(run.glob('step*.json')):
        record = json.loads(path.read_text())
        video = path.with_suffix('.mp4')
        if not video.is_file(): continue
        groups.setdefault(record['optimizer_step'], []).append((record,video))
    outputs = []
    for step,records in groups.items():
        # Retain seeds as paired repeats within each case, never as new cases.
        by_case = {}
        for record,video in records: by_case.setdefault(record['id'],[]).append((record,video))
        if len(by_case) != 4 or any(len(v) != 2 for v in by_case.values()):
            continue
        for rank in range(2):
            destination = output/run.name/f'step{step:04d}_repeat{rank}'
            cases = []
            for case_id,variants in sorted(by_case.items()):
                record,video = sorted(variants,key=lambda item:item[0]['seed'])[rank]
                folder = destination/'cases'/case_id
                folder.mkdir(parents=True,exist_ok=True)
                target = folder/'planned.mp4'
                if not target.exists(): target.symlink_to(video.resolve())
                elif target.resolve() != video.resolve(): raise ValueError('Different source video')
                cases.append(dict(id=case_id,caption=record['caption'],generation_seed=record['seed'],
                                  source_video=str(video.resolve()),sampler=record['sampler']))
            manifest = dict(cases=cases,evaluation=evaluation,optimizer_step=step,
                            scope='Four held-out WISA cases, two paired generation seeds; not an official VideoPhy2 benchmark score',
                            source_run=str(run.resolve()),training_seed=42)
            file = destination/'manifest.json'
            if file.exists() and json.loads(file.read_text()) != manifest:
                raise ValueError('Existing evaluation protocol differs')
            file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
            outputs.append(str(destination.resolve()))
    return outputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,nargs='+')
    parser.add_argument('--output',type=Path,default=ROOT/'archive/20261027_cleanup/analysis/independent_video_evaluation_v2')
    args = parser.parse_args()
    for run in args.run:
        for output in prepare(run,args.output): print(output)
