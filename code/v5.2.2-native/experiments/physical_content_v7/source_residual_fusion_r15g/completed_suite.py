"""Reuse a complete matching suite before loading generation/scoring models."""
import json
from pathlib import Path


def completion(directory,checkpoint,source,ids,mode):
    directory=Path(directory);wanted=set(ids)
    manifest=directory/'manifest.json';summary=directory/'summary.json'
    if not manifest.exists() or not summary.exists():return False,False
    plan=json.loads(manifest.read_text());status=json.loads(summary.read_text())
    if (status.get('status')!='completed' or status.get('completed_cases')!=len(wanted)
        or {c['id'] for c in plan['cases']}!=wanted):return False,False
    required=dict(checkpoint=str(checkpoint),training_source=str(source),checkpoint_step=400,
                  mode=mode,generation_seed=42)
    if any(plan.get(k)!=v for k,v in required.items()):
        raise ValueError(f'Completed suite has different generation provenance: {directory}')
    for case in wanted:
        video=directory/'cases'/case/'planned.mp4';metadata=video.with_suffix('.json')
        if not video.exists() or video.stat().st_size==0 or not metadata.exists():return False,False
        p=json.loads(metadata.read_text())
        expected=dict(checkpoint=str(checkpoint),step=400,mode=mode,case=case,seed=42,inference_source=str(source))
        if any(p.get(k)!=v for k,v in expected.items()):
            raise ValueError(f'Video provenance differs: {video}')
    scores=directory/'scores.json'
    if not scores.exists():return True,False
    rows=json.loads(scores.read_text())['cases']
    scored=(len(rows)==len(wanted) and {c['id'] for c in rows}==wanted
            and all(1<=c['sa']<=5 and 1<=c['pc']<=5 for c in rows))
    return True,scored
