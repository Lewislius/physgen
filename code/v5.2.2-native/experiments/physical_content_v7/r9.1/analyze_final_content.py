"""One normal-condition decomposition of the saved R9.1 final checkpoint.

Uses the frozen checkpoint implementation and existing fixed16 input caches on
CPU. No Wan/Qwen backbone, new videos, alternate conditions or training changes.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import sys
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    cfg = saved['config']
    if saved['step'] != 400:
        raise ValueError('This bounded diagnostic is for the final400 checkpoint only')
    sys.path[:0] = [cfg['source_snapshot'],
        str(BASE / 'experiments/physical_scaling_q256/snapshots/20261029T140406Z'),
        str(BASE / 'experiments/physical_variants/snapshots/20261029T064043Z')]
    import bootstrap
    from physical_content.model import Enhancement
    from physical_content.teacher import Teacher
    from physical_content.evidence import resample_words
    spec = importlib.util.spec_from_file_location('existing_statistics', ROOT / 'content_reader_r8/analyze_content.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    stats, ratio = helper.statistics, helper.rms_ratio
    model = Enhancement(cfg).eval()
    state = {k.removeprefix('enhancement.'): v for group in ['model', 'constants']
             for k, v in saved[group].items() if k.startswith('enhancement.')}
    model.load_state_dict(state, strict=True)
    captured = {}
    model.block.register_forward_hook(lambda module, inputs, output:
        captured.update(workspace_delta=(output - inputs[1]).float()))
    model.project.register_forward_hook(lambda module, inputs, output:
        captured.update(readout_values=inputs[0].float(), projected_values=output.float()))
    teacher = Teacher(cfg, device='cpu')
    cases = [x['id'] for x in json.loads((ROOT / 'early_execution_r3/subset16.json').read_text())['cases']]
    rows, means = [], {}
    for case in cases:
        evidence = teacher.input({'id': f'videophy40_{case}'})
        with torch.autocast('cpu', dtype=torch.bfloat16):
            c, _ = model(evidence)
            words = (evidence['words'].float() - model.anchor_mean) / model.anchor_scale
            a, _, _ = resample_words(words, evidence['word_groups'], model.seed.shape[1])
            anchor = F.linear(a, model.anchor_weight, model.anchor_bias).float()
        projected = captured['projected_values']
        correction = projected + model.shared_offset.float()
        components = dict(C=c.float(), anchor=anchor,
            workspace_delta=captured['workspace_delta'],
            readout_values=captured['readout_values'], projected_values=projected,
            learned_correction=correction)
        row = dict(id=case, components={k: stats(v) for k, v in components.items()},
            correction_to_C_rms=ratio(correction, c),
            projected_to_C_rms=ratio(projected, c),
            shared_to_projected_rms=ratio(model.shared_offset, projected),
            shared_to_C_rms=ratio(model.shared_offset, c),
            reconstruction_error_rms_ratio=ratio(c.float() - anchor - correction, c))
        if 'native_words' in evidence:
            target, _, _ = resample_words(evidence['native_words'].float(), evidence['word_groups'], c.shape[1])
            scale = model.anchor_target_scale.square().clamp_min(1e-8)
            error = c.float() - target
            mean_error = error.mean(1, keepdim=True)
            row['semantic_fit'] = dict(
                anchor_normalized_mse=float((anchor-target).square().mean()/scale),
                corrected_normalized_mse=float(error.square().mean()/scale),
                mean_error_contribution=float(mean_error.square().mean()/scale),
                centered_error_contribution=float((error-mean_error).square().mean()/scale),
                correction_target_cosine=float(F.cosine_similarity(correction, target-anchor, dim=-1).mean()))
        for key, value in components.items():
            means.setdefault(key, []).append(value.mean(1)[0])
        rows.append(row)
        print(json.dumps(dict(case=case, completed=len(rows), total=len(cases))), flush=True)
    fields = ['rms', 'pair_cosine', 'centered_energy_fraction']
    ratios = ['correction_to_C_rms', 'projected_to_C_rms', 'shared_to_projected_rms',
              'shared_to_C_rms', 'reconstruction_error_rms_ratio']
    report = dict(checkpoint=str(args.checkpoint), source_snapshot=cfg['source_snapshot'],
        step=saved['step'], cases=cases, condition='normal_only', records=rows,
        shared_offset_rms=float(model.shared_offset.float().square().mean().sqrt()),
        mean_components={k: {f: sum(r['components'][k][f] for r in rows)/len(rows)
                             for f in fields} for k in means},
        cross_sample_mean_vectors={k: stats(torch.stack(v)) for k,v in means.items()},
        mean_ratios={k: sum(r[k] for r in rows)/len(rows) for k in ratios},
        scope='Final400 normal fixed16 only; exact frozen source and saved enhancement/constants. Existing local tokenizer/cache only. No alternate model passes, foundation loading, generation, scoring or training.',
        interpretation='Workspace delta precedes actual-input value readout Z. PZ is the projected input-value term; shared_offset+broadcast PZ is the total learned correction. These are representation measurements, not evidence that differences are physically useful. Compare completed paired video controls for utility.')
    if all('semantic_fit' in r for r in rows):
        report['mean_semantic_fit'] = {k: sum(r['semantic_fit'][k] for r in rows)/len(rows)
                                       for k in rows[0]['semantic_fit']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k,v in report.items() if k not in ['records', 'cases']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
