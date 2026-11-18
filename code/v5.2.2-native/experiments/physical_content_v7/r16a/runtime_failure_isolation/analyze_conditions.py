"""CPU representation check for completed failure examples and their fixed controls."""
from pathlib import Path
import json,sys
import torch
ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
SOURCE=ROOT/'native_joint_read_r16a/snapshots/20261109T040426Z'
sys.path.insert(0,str(SOURCE))
import bootstrap
from physical_content.model import Enhancement
from condition_controls import InferenceCache
from benchmark_inputs import reference,case_input

@torch.no_grad()
def main():
    torch.set_num_threads(4)
    cfg_path=SOURCE/'configs/physical_content.yaml'
    import yaml
    cfg=yaml.safe_load(cfg_path.read_text());model=Enhancement(cfg).eval().requires_grad_(False)
    reader=InferenceCache(cfg['paths']['inference_evidence'],device='cpu')
    wanted={f'VP{i:03d}' for i in range(1,8)}
    cases=[v for v in reference()['cases'] if v['id'] in wanted]
    inputs={}
    for case in cases:
        _,item=case_input(case)
        inputs[case['id']]=[reader.input(item,image=item['image'],duration=item['duration'],drop=drop)
                            for drop in [False,True]]
    result=dict(scope='CPU C representation only; no generated-quality proof or exact CUDA execution trace',
        cases=sorted(wanted),selection='First7 completed R16A videos, selected to investigate VP002/VP004 stripe failures',
        models={})
    checkpoints={
        'R15B400':ROOT/'request_bound_fusion_r15b/runs/physical_content_67829/step0400.pt',
        'R15I400':ROOT/'joint_memory_fusion_r15i/runs/physical_content_68312/step0400.pt',
        'R16A400':ROOT/'native_joint_read_r16a/runs/physical_content_68429/step0400.pt'}
    rms=lambda x:float(x.float().square().mean().sqrt())
    for label,path in checkpoints.items():
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        values={k.removeprefix('enhancement.'):v for k,v in {**saved['model'],**saved['constants']}.items()
                if k.startswith('enhancement.')}
        model.load_state_dict(values,strict=True);del values,saved
        rows={}
        for case in cases:
            contents=[model(evidence)[0] for evidence in inputs[case['id']]]
            row={}
            for kind in ['routing','execution']:
                pos,neg=[c[kind] for c in contents]
                row[kind]=dict(positive_rms=rms(pos),negative_rms=rms(neg),positive_negative_difference_rms=rms(pos-neg))
            rows[case['id']]=row
        result['models'][label]=dict(checkpoint=str(path),rows=rows)
        print(json.dumps(dict(model=label,cases=len(rows))),flush=True)
    result['limitation']='C representation amplitudes are not actual physical write amplitudes; no CFG attribution.'
    output=Path(__file__).resolve().parent/'reports/condition_contrast.json'
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
