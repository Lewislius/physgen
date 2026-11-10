"""Combine disjoint, already scored generation parts into the fixed60 comparison.

This creates a separate view of existing completed files, then calls the unchanged
evaluator's aggregation function with all sixty cached scores. No model is loaded.
"""
from pathlib import Path
import argparse
import contextlib
import copy
from datetime import datetime,timezone
import importlib.util
import io
import json
import os
import sys

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
sys.path.insert(0,str(ROOT))
from compare_iteration import effective_guidance


def merge(record_path,if_ready=False):
    record=json.loads(record_path.read_text())
    sources=[Path(record['completed_part']),Path(record['additional_part'])]
    required=['planned.mp4','planned.json','planned.sa.json','planned.pc.json']
    plans=[]
    for source in sources:
        if not (source/'manifest.json').exists() or not (source/'scores.json').exists():
            if if_ready:return dict(status='awaiting_completed_parts',pending=str(source))
            raise FileNotFoundError(f'Complete generation and scores first: {source}')
        plan=json.loads((source/'manifest.json').read_text())
        summary=json.loads((source/'summary.json').read_text())
        if summary['status']!='completed':
            if if_ready:return dict(status='awaiting_completed_parts',pending=str(source))
            raise ValueError('Cannot merge incomplete generation')
        for case in plan['cases']:
            for name in required:
                if not (source/'cases'/case['id']/name).is_file():raise FileNotFoundError(source/'cases'/case['id']/name)
        plans.append(plan)
    first,second=plans
    for name in ['checkpoint','checkpoint_step','variant','revision','training_source','mode','sampling','evaluation']:
        if first[name]!=second[name]:raise ValueError(f'Generation parts differ: {name}')
    fg,sg=effective_guidance(first),effective_guidance(second)
    for name in ['text_scale','physical_scale','physical_branch_active']:
        if fg[name]!=sg[name]:raise ValueError(f'Active guidance differs: {name}')
    reference=json.loads((ROOT/'videophy60/protocol.json').read_text())
    by_id={};origin={}
    for source,plan in zip(sources,plans):
        for case in plan['cases']:
            name=case['id']
            if name in by_id:raise ValueError(f'Overlapping case: {name}')
            by_id[name]=case;origin[name]=source/'cases'/name
    if set(by_id)!={c['id'] for c in reference['cases']} or len(by_id)!=60:
        raise ValueError('Parts must cover the complete predefined sixty inputs exactly')
    for case in reference['cases']:
        for name in ['caption','prepared_inputs','initial_noise','generation_seed']:
            if case.get(name)!=by_id[case['id']].get(name):raise ValueError(f'{case["id"]}: mismatched {name}')
    destination=Path(record['merge_target']);destination.mkdir(parents=True,exist_ok=True)
    manifest=copy.deepcopy(first)
    manifest.update(cases=[by_id[c['id']] for c in reference['cases']],scope=reference['scope'],
        generation_parts=[str(p) for p in sources],aggregation='Sixty distinct cases from disjoint completed parts; no new scoring or regenerated video.')
    manifest['suite']='videophy60'
    manifest.pop('source_scope',None)
    if not fg['physical_branch_active']:
        manifest['inactive_physical_scale_arguments']=[g['requested_physical_scale'] for g in [fg,sg]]
    if (destination/'manifest.json').exists():
        previous=json.loads((destination/'manifest.json').read_text())
        if previous!=manifest:raise ValueError('Destination already represents a different comparison')
    (destination/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for case in manifest['cases']:
        target=destination/'cases'/case['id'];target.mkdir(parents=True,exist_ok=True)
        for name in required:
            if not (target/name).exists():os.link(origin[case['id']]/name,target/name)
    # Every score exists before this call: aggregation never invokes a model.
    module_path=ROOT.parents[1]/'tools/evaluate_videophy2.py'
    spec=importlib.util.spec_from_file_location('cached_videophy_aggregate',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with contextlib.redirect_stdout(io.StringIO()):
        module.evaluate(destination,manifest,manifest['evaluation'],None,None,None,None,None)
    (destination/'summary.json').write_text(json.dumps(dict(status='completed',mode=first['mode'],completed_cases=60))+'\n')
    scores=json.loads((destination/'scores.json').read_text())
    record.update(status='full60_completed',merged_utc=datetime.now(timezone.utc).isoformat(),summary=scores['summary'])
    record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return dict(status='full60_completed',output=str(destination),summary=scores['summary'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record',type=Path)
    parser.add_argument('--if-ready',action='store_true')
    args=parser.parse_args()
    print(json.dumps(merge(args.record,args.if_ready),ensure_ascii=False))
