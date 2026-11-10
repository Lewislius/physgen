"""Prepare eight preselected real held-out initial conditions, without future inputs."""
from pathlib import Path
import sys,json,os,copy
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
ROOT=bootstrap.BASE/'experiments/physical_content_v7'
import torch,yaml
from physical_scaling.data import Data
from physical_scaling.teacher import Teacher as InputTeacher
from physical_scaling.common import gpu_guard,atomic_torch,atomic_json
from physical_content.evidence import recover_words

CASES=['source0000128','source0000247','source0000278','source0000681',
       'source0000693','source0000846','wisa0028628','wisa0032118']
EXPECTATIONS={
 'source0000128':'Paint descends and deforms continuously under gravity; streaks remain supplied by their own paint.',
 'source0000247':'A connected pour enters the glass; liquid responds with a rising level and transient surface disturbances.',
 'source0000278':'The descending press contacts the car before compressing it; deformation follows the contact and the car remains supported.',
 'source0000681':'Any melting is gradual with water emerging from the solid; persistence is appropriate over short intervals.',
 'source0000693':'Toppling progresses through successive contacts along the domino sequence, preserving untouched pieces until reached.',
 'source0000846':'The flame remains connected to the wick while flickering; shape changes need not freeze or imply motion of the candle.',
 'wisa0028628':'Liquid enters the bottle through its connection to the bowl as the observed demonstration proceeds; container geometry persists.',
 'wisa0032118':'Ripples and bubbles evolve continuously in the water with coherent surface/material appearance.'}

@torch.no_grad()
def main():
    torch.set_num_threads(4);gpu_guard(ROOT/'heldout8/allocations'/os.getenv('DET_EXPERIMENT_ID','local'))
    cfg=yaml.safe_load((ROOT/'configs/physical_content.yaml').read_text());cfg=copy.deepcopy(cfg)
    cfg['paths']['supervision_cache']=str(ROOT/'heldout8/qwen')
    data=Data(cfg,'validation');teacher=InputTeacher(cfg)
    null=torch.load(Path(cfg['paths']['cache'])/'base/null_text.pt',map_location='cpu',weights_only=True).float()[None]
    sys.path.insert(0,str(bootstrap.BASE/'experiments/physical_scaling_q256/inference_snapshots/20261030T034046Z'))
    from protocol import manifest_for
    plan=manifest_for('v6_assoc','videophy40');plan['cases']=[]
    plan.update(suite='heldout8',scope='Eight preselected real validation first frames and existing requests; no benchmark-score claim',
        selection_before_generation=True,source_split='validation',input_keys=['first','text','null_text','times','valid'])
    for name in CASES:
        sample=data.get(data.rows.index(data.by_id[name]),targets=False)
        folder=ROOT/'heldout8/inputs'/name;folder.mkdir(parents=True,exist_ok=True)
        condition={k:sample[k].cpu() for k in ['first','text','valid']}
        condition.update(times=sample['times'][0].cpu(),null_text=null)
        atomic_torch(condition,folder/'conditions.pt')
        duration=float(sample['times'][0,-1])
        for drop in [False,True]:
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):evidence=teacher.input(sample,drop=drop)
            evidence={k:(v.cpu() if torch.is_tensor(v) else v) for k,v in evidence.items()}
            evidence.update(recover_words(evidence,duration,teacher.knowledge.tokenizer))
            atomic_torch(evidence,folder/('negative.pt' if drop else 'positive.pt'))
        canvas=Path(cfg['paths']['cache'])/'base/first'/f'{name}.png'
        plan['cases'].append(dict(id=name,caption=sample['caption'],prepared_inputs=str(folder),canvas=str(canvas),
            frames=81,fps=80/duration,generation_seed=42,expected_process=EXPECTATIONS[name],
            physical_input_image='real validation clip first image',source_split='validation'))
        atomic_json(plan,ROOT/'heldout8/manifest.json');print(json.dumps(dict(case=name,status='prepared')),flush=True)
    atomic_json(dict(status='completed',cases=CASES,scope='Only first frame/request/known duration; no future video'),ROOT/'heldout8/preparation_status.json')

if __name__=='__main__':main()
