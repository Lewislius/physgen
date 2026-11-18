"""Own400, original40 and seven paired16 controls; review before any extension."""
from pathlib import Path
import argparse,json,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import bootstrap
from physical_content.common import load_config,data_ready,gpu_guard,atomic_json
from benchmark_inputs import reference,PINNED,ROOT
from pipeline import run_stage,evaluator_environment
from allocated_gpu import bind_allocated_gpu
from completed_suite import completion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--run-id')
    p.add_argument('--describe',action='store_true')
    p.add_argument('--include-extra20',action='store_true',help='Explicitly extend reviewed40 to the same-input60; default is40 plus controls')
    args=p.parse_args();cfg=load_config(args.config)
    source=Path(cfg['source_snapshot']);root=Path(cfg['paths']['runs']).parent
    experiment=os.getenv('DET_EXPERIMENT_ID','local');run_id=args.run_id or f'physical_content_{experiment}'
    run=root/'runs'/run_id;checkpoint=run/'step0400.pt'
    output=root/'inference_outputs'/run_id/'step0400_original_fixed40_controls16_seed42'
    budget=cfg['evaluation_budget'];protocol=reference()
    extra=protocol['extra_ids'];main_ids=[v['id'] for v in protocol['cases'] if v['id'] not in extra]
    control_ids=[v['id'] for v in json.loads((source/'subset16.json').read_text())['cases']]
    plan=dict(candidate='R15E-semantic-addressing',revision=cfg['revision'],source=str(source),run=run_id,
        checkpoint=str(checkpoint),data_revision=cfg['data_revision'],steps=400,samples=1600,
        generation_views_per_microbatch=1,native_lora_updates=cfg['stages']['joint'],physical_updates=400,
        training_stages=cfg['stages'],
        new_api_calls=0,new_model_downloads=0,
        main_cases=main_ids,extra_cases=extra,control_cases=control_ids,modes=budget['control_modes'],
        automatic_next_candidate=False,extra20_enabled=args.include_extra20,
        extra20_policy='Declared same-input60 extension; original40 reused',output=str(output))
    if args.describe:print(json.dumps(plan,ensure_ascii=False,indent=2));return
    data_ready(cfg)
    folder=root/'reports/jobs'/experiment;gpu_guard(folder);atomic_json(plan,folder/'plan.json')
    bind_allocated_gpu(folder,evaluator_environment)
    def stage(name,command,environment=None):
        atomic_json(dict(status=name,run=run_id,checkpoint=str(checkpoint)),folder/'status.json')
        run_stage(command,folder/f'{name}.log',environment) if environment is not None else run_stage(command,folder/f'{name}.log')
    python=[sys.executable,'-I','-B','-u']
    # CPU-only copies from the existing frozen original Qwen feature caches.
    stage('benchmark_existing_features',[*python,source/'prepare_inputs.py','--config',args.config])
    if not checkpoint.exists():
        if not (run/'latest.pt').exists():
            stage('fresh_model_preflight',[*python,source/'gpu_preflight.py','--config',args.config,
                                          '--output',folder/'gpu_preflight'])
        train=[*python,source/'train.py','--config',args.config,'--run-id',run_id,'--stop-after','400']
        if (run/'latest.pt').exists():train+=['--resume',run/'latest.pt']
        stage('training_to400',train)
    def evaluate(name,suite,ids,mode='real',mean=None):
        destination=output/suite if mode=='real' else output/'controls'/mode/suite
        generated,scored=completion(destination,checkpoint,source,ids,mode)
        command=[*python,source/'infer.py','--checkpoint',checkpoint,'--output',output,
            '--suite',suite,'--modes',mode,'--cases',*ids]
        if mean is not None:command+=['--global-condition',mean]
        if not generated:stage(name+'_generation',command)
        if not scored:
            stage(name+'_scoring',['/home/liuzhirui/miniconda3/envs/physvid/bin/python','-I','-B','-u',
                bootstrap.BASE/'tools/evaluate_videophy2.py',destination],evaluator_environment())
        if generated and scored:
            print(json.dumps(dict(status='reused_complete_suite',suite=name,cases=len(ids))),flush=True)
        return destination
    def report(name):
        stage(name,[*python,source/'report_results.py','--output',output])
    part40=evaluate('original40','videophy40',main_ids)
    report('original40_report')
    def visual_review(name):
        stage(name,[*python,source/'data_tools/prepare_visual_review.py','--output',output,
                    '--report',root/'reports'/run_id/'visual_predeclared'])
    visual_review('original40_visual_panels')
    if args.include_extra20:
        part20=evaluate('extra20','videophy60',extra)
        merged=output/'full60/videophy60'
        record=folder/'merge_record.json'
        atomic_json(dict(checkpoint=str(checkpoint),completed_part=str(part40),additional_part=str(part20),
            merge_target=str(merged),new_videos=20,cases_after_merge=60,seed=42,gamma=5),record)
        merge=ROOT/'videophy60/snapshots/20261031T150803Z/merge_completed60.py'
        stage('full60_aggregate',[*python,merge,record])
        report('full60_report')
    mean=root/'controls'/run_id/'step0400_mean_content.pt'
    stage('train_mean_prepare',[*python,source/'data_tools/prepare_mean_condition.py','--checkpoint',checkpoint,'--output',mean])
    for mode in budget['control_modes']:
        evaluate('control16_'+mode,'videophy40',control_ids,mode,mean if mode=='global_train_mean' else None)
        report('control16_'+mode+'_report')
    visual_review('control16_visual_panels')
    atomic_json(dict(status='declared_evaluation_completed',**plan,
        decision='Review R15E scores, controls and real videos before designing R15F; validate promising candidates on the same60/WMB70/PhysicsIQ60 inputs. Maximum R15G; no automatic next training.'),folder/'status.json')


if __name__=='__main__':main()
