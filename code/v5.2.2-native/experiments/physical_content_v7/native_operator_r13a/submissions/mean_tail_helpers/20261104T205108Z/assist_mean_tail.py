"""Generate the last4 already-declared mean-C controls early for primary reuse."""
from pathlib import Path
import argparse,json,os,shutil,sys
import yaml


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--parent-experiment',type=int,required=True)
    p.add_argument('--run-id',required=True)
    p.add_argument('--describe',action='store_true')
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text())
    source=Path(cfg['source_snapshot']);root=Path(cfg['paths']['runs']).parent
    checkpoint=root/'runs'/args.run_id/'step0400.pt'
    mean=root/'controls'/args.run_id/'step0400_mean_content.pt'
    primary=root/'inference_outputs'/args.run_id/'step0400_original_fixed40_controls16_seed42'
    all_ids=[v['id'] for v in json.loads((source/'subset16.json').read_text())['cases']]
    ids=all_ids[12:];mode='global_train_mean'
    assert len(all_ids)==16 and len(ids)==4 and mode in cfg['evaluation_budget']['control_modes']
    plan=dict(parent_experiment=args.parent_experiment,run_id=args.run_id,checkpoint=str(checkpoint),
        source=str(source),global_condition=str(mean),mode=mode,cases=ids,
        full_control_cases=16,helper_videos=4,additional_cases=0,training_updates=0,
        scoring='Primary scores all16 once; helper does not score a partial subset')
    if args.describe:print(json.dumps(plan,indent=2));return
    sys.path.insert(0,str(source))
    import bootstrap
    import benchmark_inputs
    from physical_content.common import gpu_guard,atomic_json
    from pipeline import run_stage
    experiment=os.getenv('DET_EXPERIMENT_ID','local')
    folder=root/'reports/mean_tail_helpers'/experiment;gpu_guard(folder)
    atomic_json(plan,folder/'plan.json')
    status=json.loads((root/'reports/jobs'/str(args.parent_experiment)/'status.json').read_text())
    done_no_c=len(list((primary/'controls/no_c/videophy40/cases').glob('*/planned.json')))
    done_shift=len(list((primary/'controls/shift_off/videophy40/cases').glob('*/planned.json')))
    remaining_before_mean=(16-done_no_c)+(16-done_shift)
    destination=primary/'controls'/mode
    if (not checkpoint.is_file() or not mean.is_file() or destination.exists()
            or status['status']!='control16_no_c_generation' or remaining_before_mean<8):
        atomic_json(dict(status='leave_to_primary',primary_status=status,completed_no_c=done_no_c),folder/'status.json')
        return
    output=root/'helper_outputs'/args.run_id/experiment
    atomic_json(dict(status='mean_tail4_generation',**plan),folder/'status.json')
    run_stage([sys.executable,'-I','-B','-u',source/'infer.py','--checkpoint',checkpoint,
        '--output',output,'--suite','videophy40','--modes',mode,'--cases',*ids,
        '--global-condition',mean],folder/'generation.log')
    generated=output/'controls'/mode
    for case in ids:
        video=generated/'videophy40/cases'/case/'planned.mp4';data=json.loads(video.with_suffix('.json').read_text())
        if (not video.is_file() or data['checkpoint']!=str(checkpoint) or data['case']!=case
                or data['mode']!=mode or data['global_condition']!=str(mean.resolve())
                or data['inference_source']!=str(source)):
            raise ValueError('Partial handoff requires complete matching own400 evidence')
    if destination.exists():
        atomic_json(dict(status='kept_isolated_primary_started',**plan),folder/'status.json');return
    temporary=destination.parent/('.mean_tail_'+experiment)
    shutil.copytree(generated,temporary)
    atomic_json(dict(status='partial_helper_handoff',completed_cases=4,requested_cases=16,
        helper_experiment=experiment),temporary/'videophy40/summary.json')
    try:temporary.rename(destination)
    except OSError:
        if not destination.exists():raise
        shutil.rmtree(temporary)
        atomic_json(dict(status='kept_isolated_primary_started',**plan),folder/'status.json');return
    atomic_json(dict(status='transferred_partial4',primary_result=str(destination),
        evaluation_complete=False,**plan),folder/'status.json')


if __name__=='__main__':main()
