"""Summarize newly completed declared stages without re-scoring or submitting jobs."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent.parent
REPORT=ROOT/'reports/stage_observer'


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(value,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def read(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def emit(kind,**fields):
    event=dict(recorded_utc=now(),kind=kind,**fields)
    print(json.dumps(event,ensure_ascii=False),flush=True)
    with (REPORT/'events.jsonl').open('a') as stream:
        stream.write(json.dumps(event,ensure_ascii=False)+'\n')


def review(command,label):
    with (REPORT/f'{label}.log').open('a') as stream:
        result=subprocess.run(['bash',str(BASE/'tools/runtime_python.sh'),*map(str,command)],
            cwd=BASE,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
    if result.returncode:
        raise RuntimeError(f'{label} failed; see {REPORT / (label+".log")}')


def final_revision_stages():
    """Include only actually submitted final work; refresh as submissions appear."""
    stages={};candidates={}
    common=read(ROOT/'videophy60/prepared.json') or {}
    if common.get('successful_experiment_id'):
        stages['videophy60_input20']=(ROOT/'videophy60/preparation_status.json',20,'inputs')
        for name in ['R1_400','native400']:
            for part,count in [('extra20',20),('full60',60)]:
                stages[f'videophy60_{name}_{part}']=(ROOT/'videophy60/inference_outputs'/name/f'{part}_gamma5_seed42/videophy60/scores.json',count,'scores')
        stages['r1_reference250_16']=(ROOT/'training_budget_reference/inference_outputs/r1_step0250_subset16_physical5_seed42/videophy40/scores.json',16,'scores')
    for label,directory,extension in [('r9','r1_content_reader_r9','R9'),
                                      ('r9_1','content_value_r9_1','R9_1'),
                                      ('r10','execution_value_r10','R10'),
                                      ('r10b','grounded_execution_r10b','R10B')]:
        current=read(ROOT/directory/'prepared.json') or {}
        run=current.get('successful_candidate_run')
        if not run:continue
        candidates[label]=(directory,run)
        prefix=label+'_'+run+'_'
        folder=ROOT/directory/'inference_outputs'/run
        final=folder/'step0400_full40_controls16_physical5_seed42'
        stages[prefix+'training400']=(ROOT/directory/'runs'/run/'status.json',400,'train')
        stages[prefix+'intermediate250_16']=(folder/'step0250_subset16_physical5_seed42/videophy40/scores.json',16,'scores')
        stages[prefix+'full40']=(final/'videophy40/scores.json',40,'scores')
        for part,count in [('extra20',20),('full60',60)]:
            stages[prefix+part]=(ROOT/'videophy60/inference_outputs'/(extension+'_'+run)/f'{part}_gamma5_seed42/videophy60/scores.json',count,'scores')
        for mode in current['control_modes']:
            stages[prefix+mode+'16']=(final/'controls'/mode/'videophy40/scores.json',16,'scores')
        # A completed helper frees a GPU before the original owner finishes its
        # score suite. Emit that existing milestone without polling step logs.
        for kind in ['generation_dispatch','control_dispatch']:
            dispatch=read(ROOT/directory/kind/'prepared.json') or {}
            if dispatch.get('training_run')!=run:continue
            workers=([dispatch] if kind=='generation_dispatch'
                     else list(dispatch.get('dispatch_groups',{}).values()))
            for worker in workers:
                experiment=worker.get('experiment_id')
                if not experiment:continue
                name=prefix+kind+'_worker_'+str(experiment)
                stages[name]=(ROOT/directory/kind/'reports/jobs'/str(experiment)/'status.json',None,'dispatch')
    return stages,candidates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--max-hours',type=float,default=6)
    parser.add_argument('--interval-seconds',type=int,default=300)
    args=parser.parse_args()
    config=read(ROOT/'content_modulation_r6/prepared.json')
    candidate=config['successful_candidate_run'];native=config['successful_native128_run']
    output=ROOT/'content_modulation_r6/inference_outputs'
    r6=output/candidate/'step0128_full40_controls16_physical5_seed42'
    r4=ROOT/'grounded_read_r4/inference_outputs/physical_content_66871/step0064_subset16_physical5_seed42/controls'
    stages={
        'r6_training128':(ROOT/'content_modulation_r6/runs'/candidate/'status.json',128,'train'),
        'native_training128':(ROOT/'early_execution_r3/runs'/native/'status.json',128,'train'),
        'native_full40':(output/native/'step0128_full40_text5_seed42/videophy40/scores.json',40,'scores'),
        'r6_full40':(r6/'videophy40/scores.json',40,'scores'),
        'r4_unrelated16':(r4/'reasoning_swapped/videophy40/scores.json',16,'scores'),
        'r4_no_c16':(r4/'no_c/videophy40/scores.json',16,'scores'),
    }
    for mode in config['control_modes']:
        stages['r6_'+mode+'16']=(r6/'controls'/mode/'videophy40/scores.json',16,'scores')
    short_candidates={}
    for label,directory in [('r7','native_coordinate_r7'),('r8','content_reader_r8')]:
        current=read(ROOT/directory/'prepared.json') or {}
        run=current.get('successful_candidate_run')
        if not run:continue
        short_candidates[label]=(directory,run)
        folder=ROOT/directory/'inference_outputs'/run/'step0128_full40_controls16_physical5_seed42'
        # Run-qualified names retain prior completion records across revisions.
        prefix=label+'_'+run+'_'
        stages[prefix+'training128']=(ROOT/directory/'runs'/run/'status.json',128,'train')
        stages[prefix+'full40']=(folder/'videophy40/scores.json',40,'scores')
        for mode in current['control_modes']:
            stages[prefix+mode+'16']=(folder/'controls'/mode/'videophy40/scores.json',16,'scores')
    identity=dict(candidate=candidate,native=native)
    REPORT.mkdir(parents=True,exist_ok=True)
    previous=read(REPORT/'state.json') or {}
    if previous.get('identity') not in (None,identity):
        raise ValueError('Observer belongs to another run pair; use its separate evidence directory')
    completed=previous.get('completed',{})
    atomic_json(dict(started_utc=now(),pid=os.getpid(),identity=identity,additional_candidates=short_candidates,
        policy='One local completion-file pass per configured interval (default five minutes). Remote job state is checked separately through the existing det tool at relevant stages. No submissions, training changes, re-scoring or extra diagnostics.'),REPORT/'observer.json')
    emit('watch_started',identity=identity,pending=sorted(set(stages)-set(completed)))
    deadline=time.monotonic()+args.max_hours*3600
    while time.monotonic()<deadline:
        additional,final_candidates=final_revision_stages()
        stages.update(additional)
        changed=[]
        for name,(path,count,kind) in stages.items():
            if name in completed:continue
            value=read(path)
            if value is None:continue
            if kind=='train':
                if value.get('status')=='failed':
                    emit('attention_required',stage=name,reported_failure=value,
                        policy='Inspect the original job through det; no automatic restart or cancellation')
                    return 2
                ready=value.get('step')==count and value.get('status') in ['paused','complete']
                summary={key:value[key] for key in ['status','step','checkpoint'] if key in value}
            elif kind=='inputs':
                ready=(value.get('status')=='completed' and value.get('total')==count
                       and value.get('completed')==[f'VP{i:03d}' for i in range(41,61)])
                summary={key:value[key] for key in ['status','total','completed'] if key in value}
            elif kind=='dispatch':
                ready=value.get('status') in ['completed','finished','completed_and_published','left_to_original_owner']
                summary={key:value[key] for key in ['status','published','left_to_original_owner'] if key in value}
                if 'modes' in value:
                    summary['modes']={key:item.get('status') for key,item in value['modes'].items()}
            else:
                summary=value.get('summary',{})
                rows=value.get('cases',[])
                ready=(summary.get('count')==count and len(rows)==count
                       and len({row['id'] for row in rows})==count)
            if ready:
                changed.append((name,summary))
        if any(name.startswith('r4_') for name,_ in changed):
            review([ROOT/'grounded_read_r4/review_results.py'],'r4_review')
        if any(name.startswith(('r6_','native_')) and 'training' not in name for name,_ in changed):
            review([ROOT/'content_modulation_r6/review_results.py','--candidate',candidate,'--baseline',native],'r6_review')
        for label,(directory,run) in short_candidates.items():
            if any(name.startswith(label+'_') and 'training' not in name for name,_ in changed):
                review([ROOT/directory/'review_results.py','--candidate',run,'--baseline',native],label+'_review')
        expanded_changed=any(name.startswith('videophy60_') and stages[name][2]=='scores' for name,_ in changed)
        for label,(directory,run) in final_candidates.items():
            if expanded_changed or any((name.startswith(label+'_'+run+'_') or name=='r1_reference250_16') and stages[name][2]=='scores' for name,_ in changed):
                review([ROOT/directory/'review_results.py','--candidate',run],label+'_review')
        if expanded_changed or any(name.startswith(('r9_','r10_','r10b_')) and stages[name][2]=='scores' for name,_ in changed):
            review([ROOT/'videophy60/review_results.py'],'videophy60_review')
        if any(stages[name][2]=='scores' for name,_ in changed):
            review([ROOT/'build_results_tables.py'],'score_tables')
            review([ROOT/'build_video_review.py'],'video_viewer')
        for name,summary in changed:
            completed[name]=dict(completed_utc=now(),path=str(stages[name][0]),summary=summary)
            emit('stage_complete',stage=name,summary=summary)
        atomic_json(dict(recorded_utc=now(),identity=identity,completed=completed,
            pending=sorted(set(stages)-set(completed))),REPORT/'state.json')
        if set(stages)<=set(completed):
            emit('all_declared_stages_complete',note='Completion of these stages does not establish the research goal')
            return 0
        if args.once:return 0
        time.sleep(max(60,args.interval_seconds))
    emit('observer_time_limit',note='Only this observer stops; original training and inference jobs are unchanged')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
