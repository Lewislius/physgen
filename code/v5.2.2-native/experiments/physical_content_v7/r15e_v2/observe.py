"""One compact, read-only cluster check; never loops, submits or restarts jobs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,json,subprocess

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',type=int,required=True)
    args=p.parse_args();experiment=args.experiment
    result=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(experiment),'--json'],
                          text=True,capture_output=True,timeout=35)
    result.check_returncode()
    actual=json.loads(result.stdout)[0]['experiment']
    now=datetime.now(timezone.utc)
    run=f'physical_content_{experiment}'
    train=read(ROOT/'runs'/run/'status.json',{})
    job=read(ROOT/'reports/jobs'/str(experiment)/'status.json',{})
    gpu=read(ROOT/'reports/jobs'/str(experiment)/'gpu.json',{})
    output=ROOT/'inference_outputs'/run/'step0400_original_fixed40_controls16_seed42'
    counts={}
    scores={}
    for label,folder in [('40','videophy40'),('extra20','videophy60'),('60','full60/videophy60')]:
        directory=output/folder
        count=len(list((directory/'cases').glob('VP*/planned.json')))
        if count:counts[label]=count
        score=read(directory/'scores.json')
        if score:scores[label]={k:score['summary'].get(k) for k in ['count','sa_mean','pc_mean','joint_pass_rate']}
    controls={}
    control_videos={}
    for directory in sorted((output/'controls').glob('*/videophy40')):
        count=sum(1 for path in (directory/'cases').glob('VP*/planned.json')
                  if path.with_suffix('.mp4').is_file())
        if count:control_videos[directory.parent.name]=count
    for score_file in sorted((output/'controls').glob('*/videophy40/scores.json')):
        s=read(score_file)['summary']
        controls[score_file.parents[1].name]={k:s.get(k) for k in ['count','sa_mean','pc_mean','joint_pass_rate']}
    observation=dict(utc=now.isoformat(),experiment=experiment,state=actual['state'],
        trial_ids=actual.get('trialIds'),end_utc=actual.get('endTime'),gpu=gpu,
        stage=job.get('status'),step=train.get('step'),counts=counts,scores=scores,controls=controls,
        control_videos=control_videos,
        next_check_utc=(now+timedelta(seconds=900)).isoformat(),
        error=train.get('error') or job.get('error'))
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    previous=read(reports/'latest_runtime_observation.json',{})
    (reports/'latest_runtime_observation.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
    changed={k:v for k,v in observation.items() if k not in ['utc','next_check_utc','gpu'] and previous.get(k)!=v}
    if gpu.get('actual')!=previous.get('gpu',{}).get('actual'):
        changed['gpu']={k:gpu.get(k) for k in ['actual','accepted']}
    print(json.dumps(dict(utc=observation['utc'],experiment=experiment,changed=changed or 'none',
                         next_check_utc=observation['next_check_utc']),ensure_ascii=False))


if __name__=='__main__':main()
