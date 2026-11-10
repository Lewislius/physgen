"""Review matched400 VideoPhy2 scores; keep missing results and visual evidence explicit."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import importlib.util
import json

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
spec=importlib.util.spec_from_file_location('existing_review',ROOT/'early_execution_r3/review_results.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)


def summarize(path,expected):
    result=review.summarize(path)
    if result['status']=='completed':
        if result['count']!=expected:raise ValueError(f'Expected{expected}, got{result["count"]}: {path}')
        for metric in ('sa','pc','joint'):
            result[metric+'_passes']=round(result[metric+'_pass_rate']*expected)
    result['expected_count']=expected
    return result


def compare(before,after,subset=None):
    result=review.comparison.compare(before,after,case_ids=subset)
    # Do not use a SA/PC composite to hide a tradeoff.
    result['metrics'].pop('mean',None)
    return result


def targets(candidate,r10a,r1):
    if any(row['status']!='completed' for row in (candidate,r10a,r1)):
        return dict(status='pending_complete_matched_scores')
    checks=dict(pc_mean_retains_r10a=candidate['pc_mean']>=r10a['pc_mean'],
        pc_pass_rate_retains_r10a=candidate['pc_pass_rate']>=r10a['pc_pass_rate'],
        sa_improves_r10a=candidate['sa_mean']>r10a['sa_mean'],
        sa_improves_r1=candidate['sa_mean']>r1['sa_mean'],
        sa_above_3_25=candidate['sa_mean']>3.25,
        joint_exceeds_r1=candidate['joint_passes']>r1['joint_passes'])
    return dict(status='observed_numeric_targets_met' if all(checks.values()) else 'not_met',
        checks=checks,automatic_adoption=False,
        actual_video_quality='Requires review of matched generated processes and visible artifacts')


def build_report(home,run_id,label,controls):
    folder=home/'inference_outputs'/run_id/'step0400_full40_controls16_physical5_seed42'
    extension=ROOT/'videophy60/inference_outputs'/f'{label}_{run_id}'
    paths={'candidate/original40':folder/'videophy40',
        'candidate/extra20':extension/'extra20_gamma5_seed42/videophy60',
        'candidate/full60':extension/'full60_gamma5_seed42/videophy60'}
    sources=json.loads((ROOT/'reports/r10_final_evidence.json').read_text())['source_paths']
    for model in ('R10-A','R1','native400'):
        for scope in ('original40','extra20','full60'):
            paths[f'{model}/{scope}']=Path(sources[f'{model}/{scope}']).parent
    counts={'original40':40,'extra20':20,'full60':60}
    suites={key:summarize(path,counts[key.split('/')[1]]) for key,path in paths.items()}
    paired={}
    for scope in counts:
        after=f'candidate/{scope}'
        for model in ('R10-A','R1','native400'):
            before=f'{model}/{scope}'
            if all(suites[key]['status']=='completed' for key in (before,after)):
                paired[f'{scope}_vs_{model}']=compare(paths[before],paths[after])
    cases=[c['id'] for c in json.loads((home/'subset16.json').read_text())['cases']]
    if len(cases)!=16:raise ValueError('Use the existing fixed16')
    normal16=None
    if suites['candidate/original40']['status']=='completed':
        rows={c['id']:c for c in json.loads((paths['candidate/original40']/'scores.json').read_text())['cases']}
        selected=[rows[key] for key in cases]
        normal16=dict(count=16,sa_mean=sum(c['sa'] for c in selected)/16,
            pc_mean=sum(c['pc'] for c in selected)/16,
            sa_passes=sum(c['sa']>=4 for c in selected),pc_passes=sum(c['pc']>=4 for c in selected),
            joint_passes=sum(c['sa']>=4 and c['pc']>=4 for c in selected))
        for key in ('sa','pc','joint'):normal16[key+'_pass_rate']=normal16[key+'_passes']/16
    control_suites={};control_pairs={}
    for mode in controls:
        path=folder/'controls'/mode/'videophy40'
        control_suites[mode]=summarize(path,16)
        if normal16 is not None and control_suites[mode]['status']=='completed':
            control_pairs[mode]=compare(path,paths['candidate/original40'],cases)
    target_checks={scope:targets(suites[f'candidate/{scope}'],suites[f'R10-A/{scope}'],suites[f'R1/{scope}'])
                   for scope in ('original40','full60')}
    pending=[key for key,row in suites.items() if key.startswith('candidate/') and row['status']!='completed']
    pending += ['control/'+key for key,row in control_suites.items() if row['status']!='completed']
    visual_path=HERE/'reports'/f'{label}_{run_id}'/'visual_predeclared/review.json'
    visual_review=json.loads(visual_path.read_text()) if visual_path.exists() else 'Pending; quantitative scores do not establish visual/process improvement'
    control_visual_path=visual_path.parents[1]/'visual_controls16/review.json'
    control_visual=json.loads(control_visual_path.read_text()) if control_visual_path.exists() else None
    empty_visual_path=visual_path.parents[1]/'visual_controls16/empty_reasoning_review.json'
    empty_visual=json.loads(empty_visual_path.read_text()) if empty_visual_path.exists() else None
    mean_visual_path=visual_path.parents[1]/'visual_controls16/mean_condition_review.json'
    mean_visual=json.loads(mean_visual_path.read_text()) if mean_visual_path.exists() else None
    return dict(recorded_utc=datetime.now(timezone.utc).isoformat(),candidate=label,run_id=run_id,
        checkpoint_step=400,seed=42,gamma=5,suites=suites,paired_comparisons=paired,
        normal16=normal16,control_suites=control_suites,normal_vs_controls=control_pairs,
        numeric_targets=target_checks,pending=pending,all_declared_scores_complete=not pending,
        actual_video_review=visual_review,
        control_video_review=control_visual,
        empty_control_video_review=empty_visual,
        mean_control_video_review=mean_visual,
        goal_complete=False,automatic_adoption=False,
        protocol='Same original40 plus historical20/combined60 I2V, nativeCFG5/physicalgamma5, seed42; not599/PhysVid T2V.',
        uncertainty='Paired bootstrap intervals resample prompts at one seed/evaluator; they do not show cross-seed reproducibility.',
        attribution='Normal-minus-control signs indicate this checkpoint response. Text-adapter/parallel interventions are not independently retrained architectures.',
        training_comparison='Same400 samples/noise/dropout and original physical/backbone rates. Revised R11A semantic LoRA gets400 updates; original backbone LoRA gets272. Do not call all parameter update counts matched.')


def write_report(report,directory):
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'generation_review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=[f"{report['candidate']} / {report['run_id']}：400步匹配评测",'单位：SA/PC为五分均值；通过率为分数>=4，Joint要求两项同时>=4。','']
    for key,row in report['suites'].items():
        if row['status']=='completed':
            lines.append(f"{key}: SA {row['sa_mean']:.4f}, PC {row['pc_mean']:.4f}, SA率 {row['sa_pass_rate']:.4f}, PC率 {row['pc_pass_rate']:.4f}, Joint {row['joint_passes']}/{row['count']} ({row['joint_pass_rate']:.4f})")
        elif key.startswith('candidate/'):lines.append(f"{key}: 待评分，已生成{row['generated']}/{row['expected_count']}")
    lines+=['','固定16：正常结果从原40抽取；对照尚未完成时不估算得分。']
    for mode,row in report['control_suites'].items():
        lines.append(f"{mode}: "+(f"SA {row['sa_mean']:.4f}, PC {row['pc_mean']:.4f}, Joint {row['joint_passes']}/16" if row['status']=='completed' else f"待评分，已生成{row['generated']}/16"))
    visual=report['actual_video_review']
    if isinstance(visual,dict):
        lines+=['','评分前预定案例的实际视频观察：']
        for row in visual['cases']:lines.append(row['case']+'：'+row['r11a']+' '+row['interpretation'])
    control_visual=report.get('control_video_review')
    if control_visual:
        lines+=['','控制组画面复看（评分后按变化选择案例，原始分数不改）：']
        for row in control_visual['cases']:
            lines.append(row['case']+'：'+row['wrong']+' '+row['interpretation'])
        lines.append(control_visual['decision'])
    empty_visual=report.get('empty_control_video_review')
    if empty_visual:
        lines+=['','空reasoning配对画面复看（按最大PC变化及唯一Joint变化选择）：']
        for row in empty_visual['cases']:
            lines.append(row['case']+'：'+row['normal']+' '+row['empty']+' '+row['interpretation'])
        lines.append(empty_visual['decision'])
    mean_visual=report.get('mean_control_video_review')
    if mean_visual:
        lines+=['','均值C配对画面复看（两个新增Joint案例）：']
        for row in mean_visual['cases']:
            lines.append(row['case']+'：'+row['normal']+' '+row['mean']+' '+row['interpretation'])
        lines.append(mean_visual['decision'])
    lines+=['','数值目标：'+json.dumps(report['numeric_targets'],ensure_ascii=False),
        '少数视频复看与数值评分分别保留；脚本不宣布架构有效或目标完成。',
        '同协议比较和逐例得失见generation_review.json。未混入599结果。']
    (directory/'RESULTS.zh.txt').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,default=ROOT/'request_adapt_r11a/prepared.json')
    parser.add_argument('--label',default='R11A')
    args=parser.parse_args();prepared=json.loads(args.prepared.read_text())
    run_id=f"physical_content_{prepared['experiment_id']}"
    report=build_report(args.prepared.parent,run_id,args.label,prepared['control_modes'])
    out=HERE/'reports'/f'{args.label}_{run_id}';write_report(report,out)
    print(json.dumps(dict(report=str(out/'generation_review.json'),pending=report['pending'],
        numeric_targets=report['numeric_targets']),ensure_ascii=False))


if __name__=='__main__':main()
