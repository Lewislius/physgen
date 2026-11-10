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
    report=dict(candidate='R10A-API400',original_failed_R11A_preserved=True,
        comparison_scope='Original R10A architecture/training schedule; updated training captions/features/interface and inference reasoning versus old-data references.',
        main={},controls={},paired={})
    for suite,folder in [('original40','videophy40'),('full60','full60/videophy60')]:
        actual=json.loads((args.output/folder/'scores.json').read_text());report['main'][suite]=actual['summary']
        for name in ('R10-A','R1','native400'):
            reference=json.loads(Path(refs[name+'/'+suite]).read_text())
            report['paired'][suite+'/'+name]=paired(actual,reference)
    normal=json.loads((args.output/'videophy40/scores.json').read_text())
    for path in sorted((args.output/'controls').glob('*/videophy40/scores.json')):
        control=json.loads(path.read_text());ids={v['id'] for v in control['cases']}
        subset=dict(cases=[v for v in normal['cases'] if v['id'] in ids])
        report['controls'][path.parents[1].name]=dict(summary=control['summary'],normal_minus_control=paired(subset,control))
    report['target_checks']={
        'original40_pc_above3_8':report['main']['original40']['pc_mean']>3.8,
        'original40_sa_above3_3':report['main']['original40']['sa_mean']>3.3,
        'original40_joint_above_R1':report['main']['original40']['joint_pass_rate']>10/40,
        'full60_pc_retains_R10A':report['main']['full60']['pc_mean']>=3.8333333333333335,
        'full60_sa_above3_3':report['main']['full60']['sa_mean']>3.3,
        'full60_joint_above_R1':report['main']['full60']['joint_pass_rate']>13/60}
    report['numeric_targets_met']=all(report['target_checks'].values())
    report['preferred_original40_sa_above3_4']=report['main']['original40']['sa_mean']>3.4
    report['quality_goal_achieved']=False
    report['remaining_judgment']='Actual generated processes/visual artifacts and module evidence require review even if numeric targets are met.'
    (args.output/'comparison_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
