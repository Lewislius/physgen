"""Keep each existing benchmark runtime on the actually allocated device."""
import json,os,subprocess,uuid
import torch

def bind_benchmark_runtimes(folder,environment,interpreters):
    device=torch.cuda.get_device_properties(0)
    raw=str(device.uuid);selected='GPU-'+str(uuid.UUID(raw.removeprefix('GPU-')))
    record=dict(actual_gpu=device.name,allocated_uuid=selected,previous_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),checks=[])
    os.environ['CUDA_VISIBLE_DEVICES']=selected
    for interpreter in interpreters:
        result=subprocess.run([str(interpreter),'-I','-B','-c','import torch,json; torch.cuda.set_device(0); print(json.dumps(dict(device=torch.cuda.get_device_name(0),count=torch.cuda.device_count())))'],env=environment(),capture_output=True,text=True)
        record['checks'].append(dict(interpreter=str(interpreter),returncode=result.returncode,stdout=result.stdout[-1000:],stderr=result.stderr[-2000:]))
        (folder/'allocated_gpu_mapping.json').write_text(json.dumps(record,indent=2)+'\n')
        if result.returncode:raise RuntimeError('Existing benchmark runtime cannot access the allocated GPU')
