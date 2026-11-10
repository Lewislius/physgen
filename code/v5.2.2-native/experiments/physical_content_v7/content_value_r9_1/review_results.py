"""Review R9.1 against R1 at matched budgets and its declared information controls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import importlib.util
import json

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
HERE=ROOT/'content_value_r9_1'
spec=importlib.util.spec_from_file_location('short_run_review',ROOT/'early_execution_r3/review_results.py')
review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review)
spec=importlib.util.spec_from_file_location('pc_policy',ROOT/'build_results_tables.py')
policy=importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


def strict_pc_requirement(reference,candidate,count):
    if any(row.get('status')!='completed' for row in [reference,candidate]):
        return dict(status='pending',expected_count=count)
    if any(row['count']!=count for row in [reference,candidate]):
        raise ValueError('PC requirement needs matching case sets')
    delta=candidate['pc_mean']-reference['pc_mean']
    return dict(status='observed_strict_pc_gain' if delta>0 and candidate['pc_pass_rate']>=reference['pc_pass_rate'] else 'not_met',
                pc_mean_difference=delta,reference_pc=reference['pc_mean'],candidate_pc=candidate['pc_mean'],
                pc_pass_rate_difference=candidate['pc_pass_rate']-reference['pc_pass_rate'],
                tie_satisfies_goal=False,scope='Observed aggregate only; inspect paired intervals, controls and actual generation before claiming success')


def strict_pc_sa_requirement(reference,candidate,count):
    """Describe the user's two score targets; do not infer visual quality."""
    if any(row.get('status')!='completed' for row in [reference,candidate]):
        return dict(status='pending_matched_scores',expected_count=count)
    if any(row['count']!=count for row in [reference,candidate]):
        raise ValueError('PC/SA targets require the same evaluation scope')
    differences={key:candidate[key]-reference[key]
                 for key in ['pc_mean','sa_mean','pc_pass_rate','sa_pass_rate','joint_pass_rate']}
    both=differences['pc_mean']>0 and differences['sa_mean']>0
    return dict(status='observed_pc_and_sa_mean_gains' if both else 'not_met',
        priority='PC first; SA gains cannot compensate for PC loss',
        differences=differences,
        pc_mean_strictly_improved=differences['pc_mean']>0,
        sa_mean_strictly_improved=differences['sa_mean']>0,
        pc_pass_rate_retained=differences['pc_pass_rate']>=0,
        actual_visual_quality='Requires paired video/process evidence; not inferred from these scores',
        module_effectiveness='Requires declared information controls; not inferred from these scores',
        automatic_success=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',required=True)
    args=parser.parse_args()
    output=HERE/'inference_outputs'/args.candidate
    folder=output/'step0400_full40_controls16_physical5_seed42'
    paths=dict(normal=folder/'videophy40',
        r1_reference=ROOT/'inference_outputs/physical_content_66778/step0400_seed42_r1/videophy40',
        native400=ROOT/'inference_outputs/native_lora_66781/step0400_seed42_r1/videophy40',
        intermediate250=output/'step0250_subset16_physical5_seed42/videophy40',
        r1_250=ROOT/'training_budget_reference/inference_outputs/r1_step0250_subset16_physical5_seed42/videophy40')
    controls=['reasoning_swapped','empty_reasoning','global_train_mean','shared_only','anchor_only','no_c']
    paths.update({mode:folder/'controls'/mode/'videophy40' for mode in controls})
    old_output=ROOT/'r1_content_reader_r9/inference_outputs/physical_content_66968'
    paths.update(r9_reference=old_output/'step0400_full40_controls16_physical5_seed42/videophy40',
                 r9_250=old_output/'step0250_subset16_physical5_seed42/videophy40')
    suites={key:review.summarize(path) for key,path in paths.items()}
    cases=[row['id'] for row in json.loads((HERE/'subset16.json').read_text())['cases']]
    comparisons={}
    specifications=[('normal_vs_'+name,name,'normal',name in controls)
                    for name in ['r1_reference','native400','r9_reference',*controls]]
    specifications += [('r9_1_vs_r1_at250','r1_250','intermediate250',True),
                       ('r9_1_250_to400','intermediate250','normal',True),
                       ('r1_250_to400','r1_250','r1_reference',True),
                       ('r9_1_vs_r9_at250','r9_250','intermediate250',True)]
    for label,before,after,subset in specifications:
        if any(suites[key]['status']!='completed' for key in [before,after]):continue
        for key in [before,after]:
            expected=16 if key in controls or key in ['intermediate250','r1_250','r9_250'] else 40
            if suites[key]['count']!=expected:raise ValueError('Unexpected comparison scope: '+key)
        comparisons[label]=review.comparison.compare(paths[before],paths[after],case_ids=cases if subset else None)
    extension=ROOT/'videophy60/inference_outputs'
    expanded_paths={}
    for model,name in [('normal','R9_1_'+args.candidate),('r1_reference','R1_400'),('native','native400'),('r9_reference','R9_physical_content_66968')]:
        for count,part in [(20,'extra20'),(60,'full60')]:
            expanded_paths[f'{model}_{count}']=extension/name/f'{part}_gamma5_seed42/videophy60'
    expanded_suites={key:review.summarize(path) for key,path in expanded_paths.items()}
    expanded_comparisons={}
    for count in [20,60]:
        for reference in ['r1_reference','native','r9_reference']:
            before,after=f'{reference}_{count}',f'normal_{count}'
            if any(expanded_suites[k]['status']!='completed' for k in [before,after]):continue
            if any(expanded_suites[k]['count']!=count for k in [before,after]):raise ValueError('Wrong expanded comparison count')
            expanded_comparisons[after+'_vs_'+before]=review.comparison.compare(expanded_paths[before],expanded_paths[after])
    retention60=(policy.assess_pc_reference(expanded_suites['r1_reference_60'],expanded_suites['normal_60'],expected_count=60)
        if all(expanded_suites[k]['status']=='completed' for k in ['r1_reference_60','normal_60'])
        else dict(status='pending_matched60'))
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),gamma=5,seed=42,
        candidate=args.candidate,suites=suites,paired_comparisons=comparisons,
        expanded_suites=expanded_suites,expanded_comparisons=expanded_comparisons,
        pc_reference_preservation60=retention60,
        strict_pc_improvement40=strict_pc_requirement(suites['r1_reference'],suites['normal'],40),
        strict_pc_improvement60=strict_pc_requirement(expanded_suites['r1_reference_60'],expanded_suites['normal_60'],60),
        strict_pc_sa_improvement40=strict_pc_sa_requirement(suites['r1_reference'],suites['normal'],40),
        strict_pc_sa_improvement60=strict_pc_sa_requirement(expanded_suites['r1_reference_60'],expanded_suites['normal_60'],60),
        expansion_scope='Original40 remains independent. Added historicalVP041-VP060 are reported as20 and combined60; each uses the matching R1/native reference on that same set, never the40-case3.625 mean as a60-case threshold.',
        pc_reference_preservation=policy.assess_pc_reference(suites['r1_reference'],suites['normal']),
        matching='R9.1 and R1: fresh official initialization, same existing-parameter initializer plus a zero3072 shared offset, input-fitted interface,128 interface+272 joint, optimizer/data/noise/dropout/rates/losses through400. R9.1 retains the R9 reader/workspace and changes final Q-to-C output to Q-selected actual input values plus a learned shared correction. Native400 has400 Wan updates, while each physical candidate has272.',
        changed_readout='C=C_anchor+b+P Attention(Q_refined,K(E),V(E)); actual fitted word values bypass value LayerNorm and final residual/RMSNorm. Q exchange, learned full-width P and original fusion remain.',
        step_comparison='One250-step normal16 per model;400 subsets reuse full40. These compare two exposure/optimization budgets, not a pure causal effect of steps independent of data.',
        information_controls='Unrelated/empty reasoning retain own image/request/T5; global meanC fixes both CFG conditions from training only; shared_only retains anchor+b while removing only the input-value correction; anchor_only removes both learned corrections; no_c removes both fusion outputs.',
        interpretation='Latest user target: both PC and SA must exceed matched R1, alongside actual visual-quality improvement, with PC first. Same40 PC3.625/SA2.950; same60 PC3.666667/SA2.883333. SA cannot offset PC loss. Paired process/appearance stability and useful content controls remain necessary; a different or lower-cosine C alone is not success.',
        automatic_adoption=False,automatic_extension=False,maximum_authorized_revision=10)
    target=HERE/'reports'/args.candidate/'generation_review.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),suites=suites,completed_comparisons=list(comparisons)),ensure_ascii=False))


if __name__=='__main__':main()
