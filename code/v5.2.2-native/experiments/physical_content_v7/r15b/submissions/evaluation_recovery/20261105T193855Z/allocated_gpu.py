"""Preserve the actually allocated GPU across the two existing Python runtimes."""
import json
import os
import subprocess
import torch


def bind_allocated_gpu(folder, evaluator_environment):
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, 'uuid', ''))
    if not uuid.startswith('GPU-'):
        raise RuntimeError('Cannot identify the already allocated CUDA device by UUID')
    record = dict(previous_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  allocated_uuid=uuid, actual_gpu=props.name,
                  purpose='Same allocated device in generation and existing scorer runtime; no device reassignment')
    os.environ['CUDA_VISIBLE_DEVICES'] = uuid
    command = ['/home/liuzhirui/miniconda3/envs/physvid/bin/python', '-I', '-B', '-c',
               'import torch,json; torch.cuda.set_device(0); print(json.dumps(dict(device=torch.cuda.get_device_name(0),count=torch.cuda.device_count())))']
    result = subprocess.run(command, env=evaluator_environment(), capture_output=True, text=True)
    record.update(scorer_preflight_returncode=result.returncode,
                  scorer_preflight_stdout=result.stdout[-1000:], scorer_preflight_stderr=result.stderr[-2000:])
    (folder / 'allocated_gpu_mapping.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('Existing scorer cannot access the allocated GPU; inspect allocated_gpu_mapping.json')
    return record
