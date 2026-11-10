"""Reuse old/API frozen local features; CPU only, no API/model call."""
from pathlib import Path
import json,sys,torch
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,WORK,OLD,API,SHARED,OLD_OUTPUT,API_OUTPUT,cases,setup
setup(OLD)
from physical_content.teacher import Teacher
from physical_content.evidence import find_input
from protocol import load_inputs
from physical_scaling.common import atomic_torch,atomic_json

KEYS=['words','word_groups','word_positions','request_words','request_word_strings',
      'relations','caption','analysis','sample','scope']

def main():
    cfg=json.loads((ROOT/'execution_value_r10/runs/physical_content_67003/config.json').read_text())
    teacher=Teacher(cfg,device='cpu');_,selected=cases();records=[]
    for case in selected:
        inputs,canvas=load_inputs(case);sample=dict(id='videophy40_'+case['id'],caption=case['caption'])
        duration=float(inputs['times'][-1]);destination=WORK/'inputs'/case['id'];destination.mkdir(parents=True,exist_ok=True)
        for previous in [OLD_OUTPUT,API_OUTPUT]:
            meta=json.loads((previous/'cases'/case['id']/'planned.json').read_text())
            if Path(meta['canvas']).resolve()!=canvas.resolve() or meta['caption']!=case['caption']:
                raise ValueError('Paired generation did not use the same image/request')
        for drop in [False,True]:
            arm='negative' if drop else 'positive';old=teacher.input(sample,drop=drop)
            src=SHARED/sample['id']/(arm+'.pt');api=torch.load(src,map_location='cpu',weights_only=True)
            if api.get('first_image')!=str(canvas.resolve()) or abs(api['duration']-duration)>1e-6:
                raise ValueError('API input image/duration mismatch')
            for label,evidence,path in [('old',old,find_input(sample['id']+('_null' if drop else '_text'))),('api',api,src)]:
                if evidence['caption']!=('' if drop else case['caption']) or evidence['sample']!=sample['id'] or evidence['scope']!='first_image_and_request_only':
                    raise ValueError('Input-only own condition required')
                if evidence['request_word_strings']!=('' if drop else case['caption']).split():raise ValueError('Incomplete request words')
                compact={k:evidence[k] for k in KEYS if k in evidence}
                compact.update(first_image=str(canvas.resolve()),duration=duration,source=str(path),encoding=label)
                if compact['words'].shape[-1]!=4096 or compact['request_words'].shape[-1]!=4096:raise ValueError('Encoder width differs')
                atomic_torch(compact,destination/f'{label}_{arm}.pt')
                records.append(dict(case=case['id'],arm=arm,encoding=label,source=str(path),relation_count=len(compact['relations']),words=len(compact['words'])))
    atomic_json(dict(complete=True,cases=20,contexts=80,api_calls=0,model_calls=0,records=records),WORK/'inputs/ready.json')
    print(json.dumps(dict(complete=True,cases=20,contexts=80,api_calls=0,model_calls=0)))

if __name__=='__main__':main()
