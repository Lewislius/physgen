"""Existing API reasoning against DI400; fixed16 and four failures separately."""
import os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PLAN=json.loads((ROOT/'plan.json').read_text())
sys.path.insert(0,PLAN['source'])
import bootstrap
from physical_content.common import gpu_guard,atomic_json
from pipeline import run_stage,evaluator_environment
from summarize_controls import aggregate,rows

def main():
    job=ROOT/'jobs'/os.getenv('DET_EXPERIMENT_ID','local');job.mkdir(parents=True,exist_ok=True)
    gpu_guard(job)
    def stage(name,cmd,env=None):
        atomic_json(dict(status=name),job/'status.json');run_stage(cmd,job/(name+'.log'),env)
    stage('generation',[sys.executable,'-I','-B','-u',str(ROOT/'infer.py'),
                        '--checkpoint',PLAN['checkpoint'],'--physical-evidence-root',PLAN['physical_evidence_root'],
                        '--output',PLAN['output'],'--suite','videophy40','--cases',*PLAN['cases']])
    folder=Path(PLAN['output'])/'videophy40'
    stage('scoring',['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
                     str(bootstrap.BASE/'tools/evaluate_videophy2.py'),str(folder)],evaluator_environment())
    result={'checkpoint':PLAN['checkpoint'],'new_training':False,'intervention':'both physical positive/negative input-reasoning arms use existing API cache; same native prompt, first image and noise','strata':{}}
    for label in ['fixed16','diagnostic4']:
        ids=PLAN[label];old=rows(Path(PLAN['normal'])/'scores.json',ids);new=rows(folder/'scores.json',ids)
        result['strata'][label]={'original':aggregate(old),'api':aggregate(new),'pairs':[dict(id=c,old_sa=old[c]['sa'],old_pc=old[c]['pc'],api_sa=new[c]['sa'],api_pc=new[c]['pc']) for c in ids]}
    atomic_json(result,ROOT/'comparison.json')
    stage('visual_material',[sys.executable,'-I','-B','-u',str(Path(PLAN['source'])/'visual_review.py'),'--root',PLAN['output']])
    atomic_json(dict(status='completed',quality_review_required=True),job/'status.json')

if __name__=='__main__':main()
