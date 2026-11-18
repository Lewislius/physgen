"""One compact, read-only cluster check; never loops, submits or restarts jobs."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
from concurrent.futures import ThreadPoolExecutor
import argparse,json,subprocess

ROOT=Path(__file__).resolve().parent


def read(path,default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def observe_helpers(previous):
    def inspect(kind):
        prepared=read(ROOT/f'{kind}_controls_helper_prepared.json',{})
        experiment=prepared.get('experiment_id')
        if experiment is None:return kind,None
        old=previous.get(kind,{})
        if old.get('experiment')==experiment and old.get('state') in {
                'STATE_COMPLETED','STATE_CANCELED','STATE_ERROR','STATE_DELETED'}:
            return kind,old
        folder=ROOT/'reports/control_helpers'/str(experiment)
        gpu=read(folder/'gpu.json',{})
        status=read(folder/'status.json',{})
        try:
            result=subprocess.run(['det','-m','http://10.130.130.5:4096','experiment','describe',str(experiment),'--json'],
                                  text=True,capture_output=True,timeout=35)
            result.check_returncode();actual=json.loads(result.stdout)[0]['experiment']
        except (subprocess.SubprocessError,ValueError,KeyError,IndexError) as exc:
            return kind,dict(experiment=experiment,state='observation_unavailable',error=str(exc))
        counts={}
        helper_output=Path(prepared.get('output_override', ROOT/'helper_outputs'/f'physical_content_{prepared["parent_experiment"]}'/str(experiment)))
        for mode in prepared['modes']:
            directory=helper_output/'controls'/mode/'videophy40'
            counts[mode]=len(list((directory/'cases').glob('VP*/planned.json')))
        for suite in prepared.get('plan',{}).get('suites',{}):
            directory=ROOT/'helper_outputs'/f'physical_content_{prepared["parent_experiment"]}'/str(experiment)/suite
            counts[suite]=len(list((directory/'cases').glob('VP*/planned.json')))
        return kind,dict(experiment=experiment,state=actual['state'],trial_ids=actual.get('trialIds'),
                         stage=status.get('status'),gpu={k:gpu.get(k) for k in ['actual','accepted']},
                         counts=counts,transferred=status.get('transferred',[]))
    kinds=sorted(path.name.split('_controls_helper_prepared.json')[0]
                 for path in ROOT.glob('*_controls_helper_prepared.json'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        return {kind:value for kind,value in pool.map(inspect,kinds) if value is not None}


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
    for score_file in sorted((output/'controls').glob('*/videophy40/scores.json')):
        s=read(score_file)['summary']
        controls[score_file.parents[1].name]={k:s.get(k) for k in ['count','sa_mean','pc_mean','joint_pass_rate']}
    observation=dict(utc=now.isoformat(),experiment=experiment,state=actual['state'],
        trial_ids=actual.get('trialIds'),end_utc=actual.get('endTime'),gpu=gpu,
        stage=job.get('status'),step=train.get('step'),counts=counts,scores=scores,controls=controls,
        next_check_utc=(now+timedelta(seconds=900)).isoformat(),
        error=train.get('error') or job.get('error'))
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    previous=read(reports/'latest_runtime_observation.json',{})
    observation['helpers']=observe_helpers(previous.get('helpers',{}))
    (reports/'latest_runtime_observation.json').write_text(json.dumps(observation,ensure_ascii=False,indent=2)+'\n')
    changed={k:v for k,v in observation.items() if k not in ['utc','next_check_utc','gpu'] and previous.get(k)!=v}
    if gpu.get('actual')!=previous.get('gpu',{}).get('actual'):
        changed['gpu']={k:gpu.get(k) for k in ['actual','accepted']}
    print(json.dumps(dict(utc=observation['utc'],experiment=experiment,changed=changed or 'none',
                         next_check_utc=observation['next_check_utc']),ensure_ascii=False))


if __name__=='__main__':main()
