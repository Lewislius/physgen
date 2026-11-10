"""Full40/60 and paired16 evidence with the current historical-best Joint target."""
from pathlib import Path
import argparse
import json
import statistics


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



def summarize(cases):
    n = len(cases)
    return dict(count=n, sa_mean=statistics.mean(v['sa'] for v in cases),
                pc_mean=statistics.mean(v['pc'] for v in cases),
                sa_pass_rate=sum(v['sa'] >= 4 for v in cases) / n,
                pc_pass_rate=sum(v['pc'] >= 4 for v in cases) / n,
                joint_pass_count=sum(v['sa'] >= 4 and v['pc'] >= 4 for v in cases),
                joint_pass_rate=sum(v['sa'] >= 4 and v['pc'] >= 4 for v in cases) / n)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
    refs = json.loads((root / 'reports/r10_final_evidence.json').read_text())['source_paths']
    b = root / 'request_memory_r11b/inference_outputs/physical_content_67308/step0400_api_fixed60_controls16_seed42'
    r12a = root / 'reciprocal_fusion_r12a/inference_outputs/physical_content_67346/step0400_original_fixed60_controls16_seed42'
    r12b = root / 'semantic_execution_r12b/inference_outputs/physical_content_67456/step0400_original_fixed40_controls16_seed42'
    r13a = root / 'native_operator_r13a/inference_outputs/physical_content_67598/step0400_original_fixed40_controls16_seed42'
    r13b = root / 'interaction_query_r13b/inference_outputs/physical_content_67610/step0400_original_fixed40_controls16_seed42'
    report = dict(candidate='R13C-coupled-native-video-attention', main={}, paired={}, controls={}, pending_main=[],
        comparison_scope='Original-data R10A400 and R13B400 are empirical references. R13C retains their exact Q/C enhancement, original captions/Qwen/interface,1600-sample order and parameter/group budgets. One physical read modulates native video Q/K before normalization and RoPE; native V/O remain, with no independent C feature addition. Native text QKVO are frozen. Training deliberately changes to32 interface+368 joint (previous128+272); this is a combined architecture/training comparison, not an isolated architectural effect. R11B used revised data.',
        extra20_policy='Declared full60 uses original40 plus disjoint20 with identical protocol; no new training or repeated40 generation.',
        historical_joint_reference=dict(pass_count=11, count=40, required_minimum=12,
            source=str(root / 'r11_r12/reports/historical40_joint_goal_evidence.json')),
        scoring='Original1-to5 means and individual SA>=4,PC>=4 pass rates; Joint requires both on the same video. This fixed I2V protocol is not the official full590-caption T2V benchmark.')
    for suite, folder, count in [('original40', 'videophy40', 40), ('full60', 'full60/videophy60', 60)]:
        path = args.output / folder / 'scores.json'
        if not path.exists():
            report['pending_main'].append(suite)
            continue
        actual = json.loads(path.read_text())
        if len(actual['cases']) != count or {v['id'] for v in actual['cases']} != {f'VP{i:03d}' for i in range(1, count + 1)}:
            raise ValueError('Expected the complete original fixed suite')
        report['main'][suite] = summarize(actual['cases'])
        for name in ('R10-A', 'R1', 'native400'):
            reference = json.loads(Path(refs[name + '/' + suite]).read_text())
            report['paired'][suite + '/' + name] = paired(actual, reference)
        if (b / folder / 'scores.json').exists():
            report['paired'][suite + '/R11B-different-data'] = paired(actual, json.loads((b / folder / 'scores.json').read_text()))
        if (r12a / folder / 'scores.json').exists():
            report['paired'][suite + '/R12A'] = paired(actual, json.loads((r12a / folder / 'scores.json').read_text()))
        if (r12b / folder / 'scores.json').exists():
            report['paired'][suite + '/R12B'] = paired(actual, json.loads((r12b / folder / 'scores.json').read_text()))
        if (r13a / folder / 'scores.json').exists():
            report['paired'][suite + '/R13A'] = paired(actual, json.loads((r13a / folder / 'scores.json').read_text()))
        if (r13b / folder / 'scores.json').exists():
            report['paired'][suite + '/R13B'] = paired(actual, json.loads((r13b / folder / 'scores.json').read_text()))
    normal_path = args.output / 'videophy40/scores.json'
    normal = json.loads(normal_path.read_text()) if normal_path.exists() else None
    control_ids = {v['id'] for v in json.loads((Path(__file__).parent / 'subset16.json').read_text())['cases']}
    if normal:
        subset = dict(cases=[v for v in normal['cases'] if v['id'] in control_ids])
        if len(subset['cases']) != 16:
            raise ValueError('Missing the original fixed16 from main40')
        report['normal_fixed16'] = summarize(subset['cases'])
        for path in sorted((args.output / 'controls').glob('*/videophy40/scores.json')):
            control = json.loads(path.read_text())
            if len(control['cases']) != 16 or {v['id'] for v in control['cases']} != control_ids:
                raise ValueError('Expected the complete fixed16 control')
            report['controls'][path.parents[1].name] = dict(summary=summarize(control['cases']),
                normal_minus_control=paired(subset, control), score_file=str(path))
    main40 = report['main'].get('original40')
    report['primary_numeric_targets'] = {
        'PC_at_least3_8': None if main40 is None else main40['pc_mean'] >= 3.8,
        'SA_at_least3_3': None if main40 is None else main40['sa_mean'] >= 3.3,
        'Joint_strictly_above_historical11of40': None if main40 is None else main40['joint_pass_count'] >= 12}
    report['preferred_targets'] = {
        'PC_at_least3_85': None if main40 is None else main40['pc_mean'] >= 3.85,
        'SA_at_least3_35': None if main40 is None else main40['sa_mean'] >= 3.35}
    report['primary_numeric_targets_met'] = all(report['primary_numeric_targets'].values())
    requirements = json.loads((root / 'r11_r12/goal_requirements.json').read_text())
    report['cross_benchmark_acceptance_required'] = requirements['cross_benchmark_acceptance']
    cross_path = root / 'cross_benchmark_70_60/reports/current_comparison.json'
    cross = json.loads(cross_path.read_text()) if cross_path.exists() else {}
    report['cross_benchmark_numeric_acceptance'] = cross.get('cross_benchmark_numeric_acceptance', {}).get(
        'r13c', dict(wmb70_strictly_above_r1=None,
                     physics_iq_verified_strictly_above_r1=None,
                     both_numeric_requirements_met=False, status='not_yet_evaluated'))
    report['quality_goal_achieved'] = False
    report['remaining_judgment'] = 'Actual processes and visual quality require inspection; all VideoPhy2 numeric targets and strict WMB70/PhysicsIQ improvement over R1 are required together. Complete this candidate and review evidence before the next architecture. Sequential revisions are authorized through R15C, followed by the final report and pause.'
    report['control_interpretation'] = 'No_c bypasses physical attention modulation in both CFG arms. Query_modulation_off retains physical key modulation; key_modulation_off retains physical query modulation. Native values, output projection, RoPE, text path and time modulation remain. Wrong/empty reasoning retain own image/request. Train-mean C retains own relation count and changes amplitude. Inference interventions measure component dependence, not retrained-ablation quality.'
    (args.output / 'comparison_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(candidate=report['candidate'], main=report['main'], controls=list(report['controls']),
                         targets=report['primary_numeric_targets'], pending=report['pending_main']), ensure_ascii=False))


if __name__ == '__main__':
    main()
