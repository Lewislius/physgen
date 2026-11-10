"""Independent scoring manifests for complete, paired video interventions."""
import argparse
import json
from pathlib import Path
from prepare_generated_evaluation import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--field',default='condition_intervention')
    parser.add_argument('--cases',type=int,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    template=ROOT/'analysis/independent_video_evaluation_v2/v522-diagnostic-direct_fm-8cases32updates-seed42_65835/step0000_repeat0/manifest.json'
    evaluation=json.loads(template.read_text())['evaluation'];groups={}
    for file in sorted((args.directory/'videos').glob('*.json')):
        record=json.loads(file.read_text())
        if args.field not in record:continue
        groups.setdefault(record[args.field],{}).setdefault(record['id'],[]).append((record,file.with_suffix('.mp4')))
    for variant,cases in groups.items():
        if len(cases)!=args.cases or any(len(rows)!=args.repeats for rows in cases.values()):
            raise ValueError(f'Incomplete predeclared case/seed coverage for {variant}')
        for repeat in range(args.repeats):
            out=args.directory/'independent_evaluation'/f'{variant}_repeat{repeat}';rows=[]
            for name,records in sorted(cases.items()):
                record,video=sorted(records,key=lambda pair:pair[0]['seed'])[repeat]
                if not video.is_file():raise FileNotFoundError(video)
                folder=out/'cases'/name;folder.mkdir(parents=True,exist_ok=True)
                target=folder/'planned.mp4'
                if not target.exists():target.symlink_to(video.resolve())
                elif target.resolve()!=video.resolve():raise ValueError('Different paired source video')
                rows.append(dict(id=name,caption=record['caption'],generation_seed=record['seed'],
                                 source_video=str(video.resolve()),sampler=record['sampler']))
            manifest=dict(cases=rows,evaluation=evaluation,variant=variant,
                          scope='Matched C intervention on held-out WISA cases; not an official VideoPhy2 score')
            file=out/'manifest.json'
            if file.exists() and json.loads(file.read_text())!=manifest:raise ValueError('Protocol changed')
            file.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(out.resolve())


if __name__=='__main__':main()
