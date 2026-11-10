"""Paired scene-level summaries of complete C-control groups; no best-of selection."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import statistics as st

VARIANTS={'original','gradient_0.1','random_0.1','gradient_0.5','random_0.5','injection_off','other_condition'}


def interval(values):
    rng=random.Random(271828);n=len(values)
    if n<3:return None
    draws=sorted(st.mean(rng.choices(values,k=n)) for _ in range(10000))
    return [draws[249],draws[9749]]


def summarize(directories):
    groups=defaultdict(dict);gradients={};sources=[]
    for directory in directories:
        for path in sorted(Path(directory).glob('metrics_shard*.jsonl')):
            sources.append(str(path))
            for line in path.read_text().splitlines():
                row=json.loads(line)
                if row['kind']=='gradient':gradients[(row['sample'],row['seed'])]=row
                if row['kind']=='full_resample':
                    key=(row['sample'],row['seed']);variant=row['variant']
                    if variant in groups[key]:raise ValueError(f'Duplicate case/seed/variant: {key}, {variant}')
                    groups[key][variant]=row
    complete={k:v for k,v in groups.items() if set(v)==VARIANTS}
    incomplete=[dict(sample=k[0],seed=k[1],present=sorted(v)) for k,v in groups.items() if set(v)!=VARIANTS]
    metrics={};raw=[]
    for variant in sorted(VARIANTS-{'original'}):
        by_case=defaultdict(list);beats_random=[];random_advantage=defaultdict(list)
        for (case,seed),rows in sorted(complete.items()):
            original=rows['original']['objective_loss'];value=rows[variant]['objective_loss']
            gain=100*(1-value/original);by_case[case].append(gain)
            raw.append(dict(sample=case,seed=seed,variant=variant,baseline=original,objective=value,gain_percent=gain))
            if variant.startswith('gradient'):
                random_variant=variant.replace('gradient','random')
                beats_random.append(value<rows[random_variant]['objective_loss'])
                random_advantage[case].append(100*(rows[random_variant]['objective_loss']-value)/original)
        values=[st.mean(v) for v in by_case.values()]
        if values:
            metrics[variant]=dict(mean_scene_relative_gain_percent=st.mean(values),
                scene_bootstrap_95_percent_interval=interval(values),scene_means={k:st.mean(v) for k,v in by_case.items()},
                improved_case_seeds=sum(x>0 for v in by_case.values() for x in v),case_seeds=len(complete),
                beats_equal_norm_random=sum(beats_random) if beats_random else None)
            if random_advantage:
                advantages=[st.mean(v) for v in random_advantage.values()]
                metrics[variant].update(mean_advantage_over_random_percent=st.mean(advantages),
                    advantage_scene_bootstrap_95_percent_interval=interval(advantages))
    return dict(sources=sources,complete_case_seeds=len(complete),scenes=len({k[0] for k in complete}),
        incomplete_groups=incomplete,metrics=metrics,raw=raw,
        gradients=[{k:v for k,v in r.items() if k in ['sample','seed','condition_gradient_rms','recomputed_latent_max_difference']}
                   for key,r in sorted(gradients.items()) if key in complete],
        interpretation='Development oracle using reference futures, not physical-quality proof. Bootstrap resamples scenes, not frames or seeds; very few scenes imply broad uncertainty.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=summarize(args.input)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['complete_case_seeds','scenes','metrics']},indent=2))
