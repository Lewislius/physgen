"""Export completed full40 means, >=4 pass rates and auditable case-level scores."""
from pathlib import Path
from datetime import datetime,timezone
import csv
import json

ROOT=Path(__file__).resolve().parent


def assess_pc_reference(reference,candidate,expected_count=40):
    """Describe retention on the specified matched set; never select automatically."""
    keys=['pc_mean','pc_pass_rate','sa_mean','sa_pass_rate','joint_pass_rate']
    result=dict(reference='R1_400',reference_metrics={k:reference[k] for k in keys},
        comparison_case_count=expected_count,sa_gain_can_offset_pc_loss=False,automatic_replacement=False)
    if candidate.get('count',candidate.get('cases'))!=expected_count:
        return dict(**result,status=f'pending_complete{expected_count}',pc_retained_on_this_set=None)
    deltas={k:round(candidate[k]-reference[k],8) for k in keys}
    mean_retained=deltas['pc_mean']>=0
    passes_retained=deltas['pc_pass_rate']>=0
    return dict(**result,
        status='pc_retained_requires_module_and_process_review' if mean_retained and passes_retained else 'pc_not_retained',
        candidate_metrics={k:candidate[k] for k in keys},differences=deltas,
        pc_mean_retained=mean_retained,pc_pass_rate_retained=passes_retained,
        pc_retained_on_this_set=mean_retained and passes_retained,
        interpretation=f'This is the measured replacement reference on the same{expected_count} cases, not by itself an equal-training-budget ablation or proof of retention on other samples/seeds. SA cannot compensate for lost PC. Research continuation requires generation/module evidence; it is not automatic adoption.')


def main():
    sources={
        'native400':ROOT/'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40',
        'R1_400':ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40',
        'R1_1000':ROOT/'inference_outputs/physical_content_66856/step1000_seed42_r1/videophy40',
        'native64':ROOT/'early_execution_r3/inference_outputs/native_lora_66866/step0064_full40_text5_seed42/videophy40',
        'R4_64':ROOT/'grounded_read_r4/inference_outputs/physical_content_66871/step0064_full40_physical5_seed42/videophy40',
        'R1_400_request_anchor':ROOT/'request_anchor_r1/inference_outputs/r1_step0400_full40_gamma5_seed42/controls/request_anchored/videophy40',
    }
    prepared=ROOT/'content_modulation_r6/prepared.json'
    if prepared.exists():
        info=json.loads(prepared.read_text())
        if info.get('successful_candidate_run'):
            sources['R6_128']=ROOT/'content_modulation_r6/inference_outputs'/info['successful_candidate_run']/'step0128_full40_controls16_physical5_seed42/videophy40'
        if info.get('successful_native128_run'):
            sources['native128']=ROOT/'content_modulation_r6/inference_outputs'/info['successful_native128_run']/'step0128_full40_text5_seed42/videophy40'
    for version,directory in [('R7_128','native_coordinate_r7'),('R8_128','content_reader_r8')]:
        prepared=ROOT/directory/'prepared.json'
        if prepared.exists():
            info=json.loads(prepared.read_text())
            if info.get('successful_candidate_run'):
                sources[version]=ROOT/directory/'inference_outputs'/info['successful_candidate_run']/'step0128_full40_controls16_physical5_seed42/videophy40'
    for version,directory in [('R9_400','r1_content_reader_r9'),('R9.1_400','content_value_r9_1'),('R10_400','execution_value_r10'),('R10B_400','grounded_execution_r10b')]:
        prepared=ROOT/directory/'prepared.json'
        if prepared.exists():
            info=json.loads(prepared.read_text())
            if info.get('successful_candidate_run'):
                sources[version]=ROOT/directory/'inference_outputs'/info['successful_candidate_run']/'step0400_full40_controls16_physical5_seed42/videophy40'
    reference=json.loads((sources['R1_400']/'manifest.json').read_text())
    expected={case['id'] for case in reference['cases']}
    rows=[];case_rows=[];pending=[]
    for version,folder in sources.items():
        score=folder/'scores.json'
        if not score.exists():
            pending.append(dict(version=version,path=str(folder),status='awaiting_complete40'))
            continue
        data=json.loads(score.read_text());plan=json.loads((folder/'manifest.json').read_text())
        cases=data['cases']
        if len(cases)!=40 or {case['id'] for case in cases}!=expected:
            raise ValueError(f'{version}: only the complete fixed40 may enter the primary table')
        checkpoint=Path(plan['checkpoint']);config_path=checkpoint.parent/'config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else None
        step=plan['checkpoint_step']
        interface=min(config['stages']['interface'],step) if config else None
        counts={kind:sum(case[kind]>=4 for case in cases) for kind in ['sa','pc']}
        joint=sum(case['sa']>=4 and case['pc']>=4 for case in cases)
        row=dict(version=version,training_updates=step,interface_updates=interface,
            wan_lora_updates=step-interface if interface is not None else None,
            cases=40,seed=42,text_guidance=5,physical_guidance=5 if plan['variant']=='physical_content' else 'inactive',
            sa_mean=sum(case['sa'] for case in cases)/40,
            pc_mean=sum(case['pc'] for case in cases)/40,
            sa_pass_count=counts['sa'],pc_pass_count=counts['pc'],joint_pass_count=joint,
            sa_pass_rate=counts['sa']/40,pc_pass_rate=counts['pc']/40,joint_pass_rate=joint/40,
            source_scores=str(score))
        rows.append(row)
        for case in cases:
            case_rows.append(dict(version=version,id=case['id'],caption=case['caption'],sa=case['sa'],pc=case['pc'],
                sa_pass=int(case['sa']>=4),pc_pass=int(case['pc']>=4),joint=int(case['sa']>=4 and case['pc']>=4)))
    output=ROOT/'reports';output.mkdir(exist_ok=True)
    def write_csv(name,values):
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    write_csv('completed_full40_summary.csv',rows)
    write_csv('completed_full40_cases.csv',case_rows)
    record=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),completed=rows,pending=pending,
        scope='Fixed40 VideoPhy2 captions adapted to common first-image I2V, seed42 and native text/active physical guidance5; not the official full590 T2V multi-seed benchmark.',
        score_definition='SA/PC means use integer1-5 outputs; individual pass rates count>=4; joint requires both>=4. Pass rates are not mean divided by5.',
        comparison='Different budgets and recipes are explicitly listed; only equal-budget matched references isolate the short-recipe comparison. This table alone is not proof of causal module benefit or a successful replacement.')
    reference_row=next(row for row in rows if row['version']=='R1_400')
    record['pc_reference_preservation']={row['version']:assess_pc_reference(reference_row,row)
        for row in rows if row['version'].startswith('R') and row['version']!='R1_400'}
    record['pc_reference_preservation'].update({row['version']:assess_pc_reference(reference_row,row)
        for row in pending if row['version'].startswith('R')})
    (output/'completed_full40_summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(completed=[row['version'] for row in rows],pending=[row['version'] for row in pending],output=str(output/'completed_full40_summary.csv')),ensure_ascii=False))


if __name__=='__main__':main()
