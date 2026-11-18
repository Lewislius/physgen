"""Aggregate the fixed disjoint40+20 using existing scores, without judging again."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bootstrap

from benchmark_inputs import reference
from physical_content.common import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text())
    sources = [Path(record[k]) for k in ['completed_part','additional_part']]
    destination = Path(record['merge_target'])
    plans = [json.loads((p/'manifest.json').read_text()) for p in sources]
    for key in ['checkpoint','checkpoint_step','variant','revision','task','generation_seed','evaluation']:
        if plans[0][key] != plans[1][key]:
            raise ValueError(f'Cannot merge different {key}')
    wanted = {c['id']:c for c in reference()['cases']}
    cases, ratings = {}, {}
    for source, plan in zip(sources,plans):
        summary = json.loads((source/'summary.json').read_text())
        if summary['status'] != 'completed':
            raise ValueError('Cannot aggregate unfinished generation')
        scored = {r['id']:r for r in json.loads((source/'scores.json').read_text())['cases']}
        for case in plan['cases']:
            identity=case['id']
            if identity in cases or case['caption']!=wanted[identity]['caption']:
                raise ValueError('Overlapping or changed comparison case')
            cases[identity]=case
            ratings[identity]=dict(scored[identity])
            target=destination/'cases'/identity;target.mkdir(parents=True,exist_ok=True)
            for name in ['planned.mp4','planned.json','planned.sa.json','planned.pc.json']:
                if not (target/name).exists():
                    os.link(source/'cases'/identity/name,target/name)
    if set(cases)!=set(wanted) or len(cases)!=60:
        raise ValueError('Require all60 fixed cases')
    plan=copy.deepcopy(plans[0]);plan.update(suite='videophy60',cases=[cases[k] for k in wanted],
          generation_parts=[str(p) for p in sources],aggregation='Disjoint40+20 cached scores')
    rows=[ratings[k] for k in wanted]
    threshold=4
    summary=dict(count=60,sa_mean=sum(r['sa'] for r in rows)/60,
                 pc_mean=sum(r['pc'] for r in rows)/60,
                 sa_pass_rate=sum(r['sa']>=threshold for r in rows)/60,
                 pc_pass_rate=sum(r['pc']>=threshold for r in rows)/60,
                 joint_pass_rate=sum(r['sa']>=threshold and r['pc']>=threshold for r in rows)/60,
                 pass_threshold=threshold,scope=plan['scope'],training_verifier_is_independent=True)
    atomic_json(plan,destination/'manifest.json')
    atomic_json(dict(summary=summary,cases=rows),destination/'scores.json')
    atomic_json(dict(status='completed',completed_cases=60),destination/'summary.json')
    atomic_json(dict(record,status='full60_completed',summary=summary),args.record)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
