"""Single allocated Ada/Amp evaluation GPU; H100 is never accepted."""
from pathlib import Path
import json,os


def gpu_guard(folder, *, pool):
    import torch
    count=torch.cuda.device_count()
    props=torch.cuda.get_device_properties(0) if count else None
    name=props.name if props else 'NONE';memory=props.total_memory if props else 0
    compact=''.join(c for c in name.upper() if c.isalnum())
    if pool=='amp-80g':
        allowed=any(s in compact for s in ['A100','A800']) and memory>=70*1024**3
    elif pool=='amp-48g':
        allowed=any(s in compact for s in ['A6000','A100']) and memory>=38*1024**3
    elif pool=='ada-48g':
        allowed=any(s in compact for s in ['L40','RTX6000ADA','RTX4090']) and memory>=44*1024**3
    else:allowed=False
    accepted=count==1 and 'H100' not in compact and allowed
    record=dict(actual=name,visible_count=count,memory_bytes=memory,resource_pool=pool,
                accepted=accepted,experiment_id=os.getenv('DET_EXPERIMENT_ID'),scope='evaluation_only')
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    (folder/'gpu.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record),flush=True)
    if not accepted:raise SystemExit(86)
    return record
