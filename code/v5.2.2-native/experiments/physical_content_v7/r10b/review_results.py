"""Review R10-B against R1/R10-A at matched budgets and its declared controls."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import importlib.util
import json

ROOT=Path('/home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_v7')
HERE=ROOT/'grounded_execution_r10b'
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
        module_effectiveness='Controls explain gains and harms; C diversity is diagnostic, not an independent success criterion',
        automatic_success=False)


def requested_numeric_targets(candidate, count):
    target=dict(pc_mean_strictly_above=3.7,pc_preferred=3.8,
                sa_mean_strictly_above=3.25,
                minimum_joint_passes=11 if count==40 else 14)
    if candidate.get('status')!='completed':
        return dict(status='pending',count=count,targets=target)
    if candidate['count']!=count:raise ValueError('Numeric targets need the matched complete set')
    observed=dict(pc=candidate['pc_mean'],sa=candidate['sa_mean'],
                  joint_passes=round(candidate['joint_pass_rate']*count))
    checks=dict(pc=observed['pc']>3.7,sa=observed['sa']>3.25,
                joint=observed['joint_passes']>=target['minimum_joint_passes'])
    return dict(status='observed_numeric_targets_met' if all(checks.values()) else 'not_met',
                count=count,targets=target,observed=observed,checks=checks,
                preferred_pc_reached=observed['pc']>=3.8,
                visual_quality='Requires actual paired video/process evidence',automatic_success=False)


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
    controls=['reasoning_swapped','empty_reasoning','global_train_mean','core_only','anchor_only','no_c']
    paths.update({mode:folder/'controls'/mode/'videophy40' for mode in controls})
    old_output=ROOT/'content_value_r9_1/inference_outputs/physical_content_66970'
    paths.update(r9_1_reference=old_output/'step0400_full40_controls16_physical5_seed42/videophy40',
                 r9_1_250=old_output/'step0250_subset16_physical5_seed42/videophy40')
    a_output=ROOT/'execution_value_r10/inference_outputs/physical_content_67003'
    paths.update(r10_a_reference=a_output/'step0400_full40_controls16_physical5_seed42/videophy40',
                 r10_a_250=a_output/'step0250_subset16_physical5_seed42/videophy40')
    suites={key:review.summarize(path) for key,path in paths.items()}
    cases=[row['id'] for row in json.loads((HERE/'subset16.json').read_text())['cases']]
    comparisons={}
    specifications=[('normal_vs_'+name,name,'normal',name in controls)
                    for name in ['r1_reference','native400','r9_1_reference','r10_a_reference',*controls]]
    specifications += [('r10_vs_r1_at250','r1_250','intermediate250',True),
                       ('r10_250_to400','intermediate250','normal',True),
                       ('r1_250_to400','r1_250','r1_reference',True),
                       ('r10_vs_r9_1_at250','r9_1_250','intermediate250',True),
                       ('r10_b_vs_a_at250','r10_a_250','intermediate250',True)]
    for label,before,after,subset in specifications:
        if any(suites[key]['status']!='completed' for key in [before,after]):continue
        for key in [before,after]:
            expected=16 if key in controls or key in ['intermediate250','r1_250','r9_1_250','r10_a_250'] else 40
            if suites[key]['count']!=expected:raise ValueError('Unexpected comparison scope: '+key)
        comparisons[label]=review.comparison.compare(paths[before],paths[after],case_ids=cases if subset else None)
    extension=ROOT/'videophy60/inference_outputs'
    expanded_paths={}
    for model,name in [('normal','R10B_'+args.candidate),('r1_reference','R1_400'),('native','native400'),('r9_1_reference','R9_1_physical_content_66970'),('r10_a_reference','R10_physical_content_67003')]:
        for count,part in [(20,'extra20'),(60,'full60')]:
            expanded_paths[f'{model}_{count}']=extension/name/f'{part}_gamma5_seed42/videophy60'
    expanded_suites={key:review.summarize(path) for key,path in expanded_paths.items()}
    expanded_comparisons={}
    for count in [20,60]:
        for reference in ['r1_reference','native','r9_1_reference','r10_a_reference']:
            before,after=f'{reference}_{count}',f'normal_{count}'
            if any(expanded_suites[k]['status']!='completed' for k in [before,after]):continue
            if any(expanded_suites[k]['count']!=count for k in [before,after]):raise ValueError('Wrong expanded comparison count')
            expanded_comparisons[after+'_vs_'+before]=review.comparison.compare(expanded_paths[before],expanded_paths[after])
    retention60=(policy.assess_pc_reference(expanded_suites['r1_reference_60'],expanded_suites['normal_60'],expected_count=60)
        if all(expanded_suites[k]['status']=='completed' for k in ['r1_reference_60','normal_60'])
        else dict(status='pending_matched60'))
    report=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),gamma=5,seed=42,
        candidate=args.candidate,candidate_label='R10-B',suites=suites,paired_comparisons=comparisons,
        expanded_suites=expanded_suites,expanded_comparisons=expanded_comparisons,
        pc_reference_preservation60=retention60,
        strict_pc_improvement40=strict_pc_requirement(suites['r1_reference'],suites['normal'],40),
        strict_pc_improvement60=strict_pc_requirement(expanded_suites['r1_reference_60'],expanded_suites['normal_60'],60),
        strict_pc_sa_improvement40=strict_pc_sa_requirement(suites['r1_reference'],suites['normal'],40),
        strict_pc_sa_improvement60=strict_pc_sa_requirement(expanded_suites['r1_reference_60'],expanded_suites['normal_60'],60),
        requested_targets40=requested_numeric_targets(suites['normal'],40),
        requested_targets60=requested_numeric_targets(expanded_suites['normal_60'],60),
        expansion_scope='Original40 remains independent. Added historicalVP041-VP060 are reported as20 and combined60; each uses the matching R1/native reference on that same set, never the40-case3.625 mean as a60-case threshold.',
        pc_reference_preservation=policy.assess_pc_reference(suites['r1_reference'],suites['normal']),
        matching='Fresh official foundations and input-fitted interface; R1-matched128 interface+272 joint, data/noise/dropout/rates and numerical loss coefficients. Native-word auxiliary targets the shared execution core, as in the R1 shared K/V design. Extra routing correction learns through generation. Parameter initialization/RNG matches R10-A; visual head allocation adds no parameters. Native400 has400 Wan updates, physical candidates272.',
        changed_readout='Q reads16 relation-word,8 first-image visual and8 request/reasoning heads. C_core=A+P_execute U supplies values and the core of keys; C_route=C_core+b+P_route Z adds content-dependent routing. Native-word semantic alignment applies to C_core. Original state-dependent fusion, query exchange and native text remain.',
        step_comparison='One250-step normal16 per model;400 subsets reuse full40. These compare two exposure/optimization budgets, not a pure causal effect of steps independent of data.',
        information_controls='Unrelated/empty reasoning retain own image/request/T5; global meanC fixes both CFG conditions from training only; core_only retains the learned execution core in both keys and values, removing only additional content routing; anchor_only uses only input anchors in both memories; no_c removes both fusion outputs.',
        interpretation='Latest user target: both PC and SA must exceed matched R1, alongside actual visual-quality improvement, with PC first. Same40 PC3.625/SA2.950; same60 PC3.666667/SA2.883333. SA cannot offset PC loss. Paired process/appearance stability remains necessary. Controls explain module contributions; useful shared priors are permitted, C diversity is not an independent target.',
        automatic_adoption=False,automatic_extension=False,maximum_authorized_revision=10)
    target=HERE/'reports'/args.candidate/'generation_review.json';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),suites=suites,completed_comparisons=list(comparisons)),ensure_ascii=False))


if __name__=='__main__':main()
