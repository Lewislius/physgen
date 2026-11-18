"""Report means, threshold rates and paired changes without inventing a win."""
from pathlib import Path
import argparse,json,statistics


def paired(a,b):
    before={v['id']:v for v in b['cases']};after={v['id']:v for v in a['cases']}
    if not set(after)<=set(before):raise ValueError('Reference cases differ')
    result={}
    for name in ('sa','pc'):
        diffs=[v[name]-before[k][name] for k,v in after.items()]
        result[name]=dict(mean_change=statistics.mean(diffs),wins=sum(v>0 for v in diffs),
            ties=sum(v==0 for v in diffs),losses=sum(v<0 for v in diffs))
    joint=lambda v:v['sa']>=4 and v['pc']>=4
    result['joint']=dict(gains=[k for k,v in after.items() if joint(v) and not joint(before[k])],
        losses=[k for k,v in after.items() if joint(before[k]) and not joint(v)])
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
    refs=json.loads((root/'reports/r10_final_evidence.json').read_text())['source_paths']
    report=dict(candidate='R11B-request-memory',original_failed_R11A_preserved=True,
        comparison_scope='Full-width independent Qwen request-memory branch, with original R10A physical computation. Same-data R10A400 isolates the trained architectural addition at matched128+272; old-data references do not isolate architecture.',
        main={},controls={},paired={},pending_main=[])
    reference_record=root/'r10a_api400/prepared.json'
    same_data=None
    if reference_record.exists():
        prepared=json.loads(reference_record.read_text())
        experiment=prepared.get('experiment_id')
        if experiment is not None:
            same_data=root/'r10a_api400/inference_outputs'/f'physical_content_{experiment}'/'step0400_api_fixed60_controls16_seed42'
    report['same_data_reference']=dict(
        output=None if same_data is None else str(same_data),available_suites=[],
        scope='Same revised training cache, shared benchmark inputs, initialization draws for original modules,400 real-video updates and128+272 recipe. New request-memory branch differs; no per-case selection.')
    prior_candidate=root/'aligned_request_r11a/inference_outputs/physical_content_67265/step0400_api_fixed60_controls16_seed42'
    for suite,folder in [('original40','videophy40'),('full60','full60/videophy60')]:
        score_path=args.output/folder/'scores.json'
        if not score_path.exists():
            report['pending_main'].append(suite)
            continue
        actual=json.loads(score_path.read_text())
        expected=40 if suite=='original40' else 60
        if len(actual['cases'])!=expected or len({v['id'] for v in actual['cases']})!=expected:
            raise ValueError(f'{suite} score file does not contain the complete fixed suite')
        report['main'][suite]=actual['summary']
        for name in ('R10-A','R1','native400'):
            reference=json.loads(Path(refs[name+'/'+suite]).read_text())
            report['paired'][suite+'/'+name]=paired(actual,reference)
        if same_data is not None and (same_data/folder/'scores.json').exists():
            reference=json.loads((same_data/folder/'scores.json').read_text())
            if {v['id'] for v in reference['cases']}=={v['id'] for v in actual['cases']}:
                report['paired'][suite+'/R10A-API400']=paired(actual,reference)
                report['same_data_reference']['available_suites'].append(suite)
        if (prior_candidate/folder/'scores.json').exists():
            report['paired'][suite+'/R11A-request-execution']=paired(actual,json.loads((prior_candidate/folder/'scores.json').read_text()))
    normal_path=args.output/'videophy40/scores.json'
    normal=json.loads(normal_path.read_text()) if normal_path.exists() else None
    for path in (sorted((args.output/'controls').glob('*/videophy40/scores.json')) if normal else []):
        control=json.loads(path.read_text());ids={v['id'] for v in control['cases']}
        subset=dict(cases=[v for v in normal['cases'] if v['id'] in ids])
        if len(ids)!=16 or len(subset['cases'])!=16:raise ValueError('Expected the complete fixed16 paired control')
        report['controls'][path.parents[1].name]=dict(summary=control['summary'],normal_minus_control=paired(subset,control))
        report['normal_fixed16']=dict(count=16,sa_mean=statistics.mean(v['sa'] for v in subset['cases']),
            pc_mean=statistics.mean(v['pc'] for v in subset['cases']),
            joint_pass_rate=sum(v['sa']>=4 and v['pc']>=4 for v in subset['cases'])/16)
    original40=report['main'].get('original40')
    full60=report['main'].get('full60')
    report['target_checks']={
        'original40_pc_above3_8':None if original40 is None else original40['pc_mean']>3.8,
        'original40_sa_above3_3':None if original40 is None else original40['sa_mean']>3.3,
        'original40_joint_above_R1':None if original40 is None else original40['joint_pass_rate']>10/40,
        'full60_pc_retains_R10A':None if full60 is None else full60['pc_mean']>=3.8333333333333335,
        'full60_sa_above3_3':None if full60 is None else full60['sa_mean']>3.3,
        'full60_joint_above_R1':None if full60 is None else full60['joint_pass_rate']>13/60}
    report['numeric_targets_met']=all(report['target_checks'].values())
    report['numeric_targets_status']='pending' if report['pending_main'] else ('met' if report['numeric_targets_met'] else 'not_met')
    report['preferred_original40_pc_at_least3_85']=None if original40 is None else original40['pc_mean']>=3.85
    report['preferred_original40_sa_at_least3_35']=None if original40 is None else original40['sa_mean']>=3.35
    visual_path=root/'request_memory_r11b/reports'/args.output.parent.name/'visual_predeclared/review.json'
    report['actual_video_review']=json.loads(visual_path.read_text()) if visual_path.exists() else None
    report['quality_goal_achieved']=False
    report['control_interpretation']='Request-off and own-image/donor-request interventions measure the independent semantic branch. No_c disables both extra branches; mean C replaces physical C only, retaining own request memory and group priors. Reasoning controls keep literal request and first image. Inference interventions do not substitute for the separately trained same-data R10A400. Mean C also changes amplitude. Read paired videos with scores.'
    report['remaining_judgment']='Actual generated processes/visual artifacts and module evidence require review even if numeric targets are met.'
    (args.output/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
