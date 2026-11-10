"""Matched-input C decomposition and learned value projections on CPU."""
from pathlib import Path
import argparse
import json
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]


def statistics(value):
    value = value.float().reshape(-1, value.shape[-1])
    count = len(value)
    unit = F.normalize(value, dim=-1)
    total = value.square().sum().clamp_min(1e-20)
    return dict(rms=float(value.square().mean().sqrt()),
        pair_cosine=float((unit.sum(0).square().sum()-unit.square().sum())/(count*(count-1))) if count > 1 else None,
        centered_energy_fraction=float((value-value.mean(0)).square().sum()/total))


def rms_ratio(a, b):
    return float(a.float().square().mean().sqrt()/b.float().square().mean().sqrt().clamp_min(1e-20))


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=Path, default=ROOT/'early_execution_r3/subset16.json')
    p.add_argument('--skip-controls', action='store_true', help='Only inspect the normal condition; do not repeat alternate-condition passes')
    args=p.parse_args()
    torch.set_num_threads(8)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    cfg=saved['config']
    sys.path[:0]=[cfg['source_snapshot'],
        str(BASE/'experiments/physical_scaling_q256/snapshots/20261029T140406Z'),
        str(BASE/'experiments/physical_variants/snapshots/20261029T064043Z')]
    import bootstrap
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    from physical_content.evidence import resample_words
    model=Enhancement(cfg).eval()
    parameters={key.removeprefix('enhancement.'):value for key,value in saved['model'].items() if key.startswith('enhancement.')}
    parameters.update({key.removeprefix('enhancement.'):value for key,value in saved['constants'].items() if key.startswith('enhancement.')})
    model.load_state_dict(parameters)
    captured={}
    model.project.register_forward_hook(lambda module,inputs,output:captured.update(
        residual=output.detach().float(),latent_update=inputs[0].detach().float()))
    modulation=hasattr(model,'content_write')
    if modulation:
        model.content_write.register_forward_pre_hook(lambda module,inputs:captured.update(
            raw_query_update=inputs[1].detach().float()))
    update_name='preprojection_content_update' if modulation else 'latent_update'
    teacher=Teacher(cfg,device='cpu')
    case_ids=[case['id'] for case in json.loads(args.cases.read_text())['cases']]
    rows=[];means={key:[] for key in ['C','anchor',update_name,'learned_residual']}
    if modulation:means['raw_query_update']=[]
    modes=[] if args.skip_controls else ['reasoning_swapped','empty_reasoning']
    for case in case_ids:
        evidence=teacher.input(dict(id=f'videophy40_{case}'))
        with torch.autocast('cpu',dtype=torch.bfloat16):
            c,groups=model(evidence)
        words=(evidence['words'].float()-model.anchor_mean)/model.anchor_scale
        anchor,_,_=resample_words(words,evidence['word_groups'],model.seed.shape[1])
        native=F.linear(anchor,model.anchor_weight,model.anchor_bias)
        c=c.float();residual=captured['residual']
        components={'C':c,'anchor':native,update_name:captured['latent_update'],'learned_residual':residual}
        if modulation:components['raw_query_update']=captured['raw_query_update']
        row=dict(id=case,components={key:statistics(value) for key,value in components.items()},
            residual_to_C_rms=rms_ratio(residual,c),residual_to_anchor_rms=rms_ratio(residual,native),
            conditional_sensitivity={},projected_values={})
        if ('content_projection_init_scale' in cfg['model']
                and cfg['model'].get('content_projection_trainable', True)):
            initializer=model.anchor_weight*cfg['model']['content_projection_init_scale']
            initial_part=F.linear(captured['latent_update'],initializer)
            learned_part=F.linear(captured['latent_update'],model.project.weight-initializer)
            row['projection_parts']=dict(initial_operator_on_current_update=statistics(initial_part),
                learned_operator_change_on_current_update=statistics(learned_part),
                reconstruction_error_rms_ratio=rms_ratio(initial_part+learned_part-residual,residual),
                parts_cosine=float(F.cosine_similarity(initial_part,learned_part,dim=-1).mean()))
        if 'native_words' in evidence:
            target,_,_=resample_words(evidence['native_words'].float(),evidence['word_groups'],c.shape[1])
            scale=model.anchor_target_scale.square().clamp_min(1e-8)
            prediction=native+residual
            error=prediction-target;mean_error=error.mean(1,keepdim=True)
            row['semantic_fit']=dict(anchor_normalized_mse=float((native-target).square().mean()/scale),
                corrected_normalized_mse=float(error.square().mean()/scale),
                mean_error_contribution=float(mean_error.square().mean()/scale),
                centered_error_contribution=float((error-mean_error).square().mean()/scale),
                correction_target_cosine=float(F.cosine_similarity(residual,target-native,dim=-1).mean()))
        for key,value in components.items():means[key].append(value[0].mean(0))
        for mode in modes:
            other=torch.load(ROOT/'cache/reasoning_controls'/mode/f'{case}.pt',map_location='cpu',weights_only=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):alternate,_=model(other)
            row['conditional_sensitivity'][mode]=dict(C=statistics(alternate),
                difference_to_real_rms=rms_ratio(alternate-c,c),
                mean_vector_cosine=float(F.cosine_similarity(c.mean(1),alternate.float().mean(1)).mean()))
        for layer in [2,14,29]:
            prefix=f'fusions.{layer}.'
            projected=F.linear(F.linear(c,saved['model'][prefix+'value.weight'].float()),saved['model'][prefix+'output.weight'].float())
            row['projected_values'][str(layer+1)]=statistics(projected)
        rows.append(row)
        print(json.dumps(dict(step=saved['step'],completed=len(rows),total=len(case_ids),case=case)),flush=True)
    fields=['rms','pair_cosine','centered_energy_fraction']
    report=dict(checkpoint=str(args.checkpoint),step=saved['step'],revision=cfg['revision'],source_snapshot=cfg['source_snapshot'],
        cases=case_ids,records=rows,
        mean_components={key:{field:sum(row['components'][key][field] for row in rows)/len(rows) for field in fields} for key in means},
        cross_sample_mean_vectors={key:statistics(torch.stack(values)) for key,values in means.items()},
        mean_projected_values={str(layer):{field:sum(row['projected_values'][str(layer)][field] for row in rows)/len(rows) for field in fields} for layer in [3,15,30]},
        mean_residual_to_C_rms=sum(x['residual_to_C_rms'] for x in rows)/len(rows),
        mean_condition_sensitivity={mode:sum(row['conditional_sensitivity'][mode]['difference_to_real_rms'] for row in rows)/len(rows) for mode in modes},
        interpretation='Representation diagnostics only. O(V(C)) measures learned value directions, not the actual attention-weighted state-dependent write. Content sensitivity and low cosine are not proofs of beneficial physical generation; use paired video scores and processes.')
    if modulation:
        report['update_scope']='raw_query_update is DeltaH before content multiplication; preprojection_content_update is A*DeltaH. A low cosine after multiplication can follow algebraically from different anchors and does not establish sample-specific learned reasoning.'
    if all('semantic_fit' in row for row in rows):
        report['mean_semantic_fit']={key:sum(row['semantic_fit'][key] for row in rows)/len(rows) for key in rows[0]['semantic_fit']}
        report['semantic_fit_scope']='Existing native-word semantic auxiliary, not physical truth. Mean and centered errors exactly decompose its MSE; neither is a new training objective.'
    if all('projection_parts' in row for row in rows):
        report['mean_projection_parts']={part:{field:sum(row['projection_parts'][part][field] for row in rows)/len(rows) for field in fields}
            for part in ['initial_operator_on_current_update','learned_operator_change_on_current_update']}
        report['projection_parts_scope']='Both operators act on the current learned Q update. The initializer term is not the full untrained model; FP32 linear decomposition may differ slightly from the recorded autocast output.'
    elif not cfg['model'].get('content_projection_trainable', True):
        report['projection_parts_scope']='The saved Q-to-C operator is fixed. Its learned operator change is identically zero, so no cosine or diversity statistic is reported for that zero tensor. The learned Q update and its nonzero projected correction remain measured above.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ['records','cases']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
