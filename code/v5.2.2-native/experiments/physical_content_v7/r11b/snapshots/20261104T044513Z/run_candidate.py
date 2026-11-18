"""Request-memory R11B: own400, fixed40/60 and seven16 controls."""
from pathlib import Path
import argparse,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import load_config,data_ready,gpu_guard,atomic_json
from benchmark_inputs import reference,PINNED,ROOT
from pipeline import run_stage,evaluator_environment


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-id')
    p.add_argument('--describe',action='store_true')
    args=p.parse_args();cfg=load_config(args.config)
    source=Path(cfg['source_snapshot']);root=Path(cfg['paths']['runs']).parent
    experiment=os.getenv('DET_EXPERIMENT_ID','local');run_id=args.run_id or f'physical_content_{experiment}'
    run=root/'runs'/run_id;checkpoint=run/'step0400.pt'
    output=root/'inference_outputs'/run_id/'step0400_api_fixed60_controls16_seed42'
    budget=cfg['evaluation_budget'];protocol=reference()
    extra=protocol['extra_ids'];main_ids=[v['id'] for v in protocol['cases'] if v['id'] not in extra]
    control_ids=[v['id'] for v in json.loads((source/'subset16.json').read_text())['cases']]
    plan=dict(candidate='R11B-request-memory',revision=cfg['revision'],source=str(source),run=run_id,
        checkpoint=str(checkpoint),data_revision=cfg['data_revision'],steps=400,samples=1600,
        generation_views_per_microbatch=dict(interface=1,joint=1),native_lora_updates=272,
        physical_updates=400,request_memory_updates=400,
        training_stages=cfg['stages'],
        main_cases=main_ids,extra_cases=extra,control_cases=control_ids,modes=budget['control_modes'],
        automatic_next_candidate=False,output=str(output))
    if args.describe:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    data_ready(cfg)
    folder=root/'reports/jobs'/experiment;gpu_guard(folder);atomic_json(plan,folder/'plan.json')
    def stage(name,command,environment=None):
        atomic_json(dict(status=name,run=run_id,checkpoint=str(checkpoint)),folder/'status.json')
        run_stage(command,folder/f'{name}.log',environment) if environment is not None else run_stage(command,folder/f'{name}.log')
    python=[sys.executable,'-I','-B','-u']
    # Finish all legal inference inputs before spending the training budget.
    # These stages reuse complete files on resume and release local Qwen before Wan.
    stage('benchmark_api_reasoning',[*python,source/'prepare_inputs.py','--config',args.config,'--stage','reasoning'])
    stage('benchmark_local_qwen_encoding',[*python,source/'prepare_inputs.py','--config',args.config,'--stage','encode'])
    stage('request_memory_controls16',[*python,source/'request_controls.py','--config',args.config])
    if not checkpoint.exists():
        train=[*python,source/'train.py','--config',args.config,'--run-id',run_id,'--stop-after','400']
        if (run/'latest.pt').exists():train+=['--resume',run/'latest.pt']
        stage('training_to400',train)
    def evaluate(name,suite,ids,mode='real',mean=None):
        command=[*python,source/'infer.py','--checkpoint',checkpoint,'--output',output,
            '--suite',suite,'--modes',mode,'--cases',*ids]
        if mean is not None:command+=['--global-condition',mean]
        stage(name+'_generation',command)
        destination=output/suite if mode=='real' else output/'controls'/mode/suite
        stage(name+'_scoring',['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
            bootstrap.BASE/'tools/evaluate_videophy2.py',destination],evaluator_environment())
        return destination
    part40=evaluate('original40','videophy40',main_ids)
    stage('original40_report',[*python,source/'report_results.py','--output',output])
    part20=evaluate('extra20','videophy60',extra)
    merged=output/'full60/videophy60'
    record=folder/'merge_record.json'
    atomic_json(dict(checkpoint=str(checkpoint),completed_part=str(part40),additional_part=str(part20),
        merge_target=str(merged),new_videos=20,cases_after_merge=60,seed=42,gamma=5),record)
    merge=ROOT/'videophy60/snapshots/20261031T150803Z/merge_completed60.py'
    stage('full60_aggregate',[*python,merge,record])
    stage('full60_report',[*python,source/'report_results.py','--output',output])
    mean=root/'controls'/run_id/'step0400_mean_content.pt'
    stage('train_mean_prepare',[*python,source/'data_tools/prepare_mean_condition.py','--checkpoint',checkpoint,'--output',mean])
    for mode in budget['control_modes']:
        evaluate('control16_'+mode,'videophy40',control_ids,mode,mean if mode=='global_train_mean' else None)
        stage('control16_'+mode+'_report',[*python,source/'report_results.py','--output',output])
    stage('result_report',[*python,source/'report_results.py','--output',output])
    atomic_json(dict(status='declared_evaluation_completed',**plan,
        decision='Compare PC/SA/Joint and actual videos with R10A; do not automatically continue training or select the next architecture.'),folder/'status.json')


if __name__=='__main__':main()
