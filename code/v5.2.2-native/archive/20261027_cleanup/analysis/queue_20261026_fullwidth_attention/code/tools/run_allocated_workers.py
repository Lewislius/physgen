"""Run independent studies on explicitly allocated GPUs, preserving failures.

The caller reserves all visible slots in Determined. Exclusions are specific
devices with observed failures, never unallocated device access or node changes.
"""
import argparse
import concurrent.futures
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def available_devices(exclude, allocated_slots=None):
    if allocated_slots is not None:
        # CUDA reports only devices admitted by the container and its visibility
        # mask. Never select from a host-wide nvidia-smi listing for a partial
        # allocation. This permits reserving two slots to avoid the observed
        # failure-associated device while running just one resident model.
        import torch
        count = torch.cuda.device_count()
        if count != allocated_slots:
            raise RuntimeError(f'Expected {allocated_slots} allocated CUDA devices, received {count}; refusing ambiguous device ownership')
        # PyTorch exposes UUID objects without NVIDIA's required 'GPU-' prefix.
        # Normalize before exclusion matching and CUDA_VISIBLE_DEVICES use.
        def cuda_uuid(index):
            value = str(torch.cuda.get_device_properties(index).uuid)
            return value if value.startswith('GPU-') else 'GPU-'+value
        devices = [dict(uuid=cuda_uuid(i),
                        name=torch.cuda.get_device_name(i)) for i in range(count)]
        return [device for device in devices if device['uuid'] not in exclude], devices
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],
                          capture_output=True,text=True,check=True)
    devices=[dict(uuid=row[0].strip(),name=row[1].strip()) for row in csv.reader(io.StringIO(result.stdout))]
    return [device for device in devices if device['uuid'] not in exclude],devices


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text())
    output=Path(plan['output']);output.mkdir(parents=True,exist_ok=True)
    devices,visible=available_devices(set(plan.get('exclude_gpu_uuids',[])), plan.get('allocated_slots'))
    workers=plan['workers']
    if len(devices)<len(workers):raise RuntimeError('Insufficient allocated, nonexcluded devices')
    (output/'allocation.json').write_text(json.dumps(dict(visible=visible,selected=devices[:len(workers)],
        exclusions=plan.get('exclude_gpu_uuids',[]),exclusion_reason=plan.get('exclusion_reason','')),indent=2))
    def worker(index,jobs):
        device=devices[index];env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],WORLD_SIZE='1')
        results=[]
        for job in jobs:
            name=job['name'];command=job['command'];record=dict(name=name,device=device,
                started_unix=time.time(),command=command,state='running')
            results.append(record);status=output/f'worker{index}.json'
            status.write_text(json.dumps(results,indent=2))
            monitor=[sys.executable,'-I','-B','-u',str(Path(__file__).with_name('run_recorded.py')),
                     '--output',str(output/f'{name}.resources.jsonl'),'--',*command]
            with (output/f'{name}.stdout.log').open('w') as stream:
                completed=subprocess.run(monitor,env=env,stdout=stream,stderr=subprocess.STDOUT)
            record.update(state='complete' if completed.returncode==0 else 'failed',
                          returncode=completed.returncode,seconds=time.time()-record['started_unix'])
            status.write_text(json.dumps(results,indent=2));print(json.dumps(record),flush=True)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as executor:
        results=list(executor.map(lambda args:worker(*args),enumerate(workers)))
    (output/'results.json').write_text(json.dumps(results,indent=2))
    if any(job['returncode'] for group in results for job in group):raise SystemExit(1)


if __name__=='__main__':main()
