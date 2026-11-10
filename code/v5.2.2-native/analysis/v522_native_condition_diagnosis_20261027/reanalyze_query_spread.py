"""Decompose saved Q conditions; no model loading, generation, or training."""
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(4)


def rms(value):
    return float(value.float().square().mean().sqrt())


def main():
    source = ROOT / 'Q/query_scale'
    files = sorted(source.glob('*_C.pt'))
    data = [torch.load(path, map_location='cpu', weights_only=True) for path in files]
    assert len(data) == 4
    rows = []
    for variant in ('original', 'rms0p2', 'rms1p0'):
        correct = torch.cat([row[f'{variant}/correct'].float() for row in data])
        wrong = torch.cat([row[f'{variant}/wrong'].float() for row in data])
        assert correct.shape == wrong.shape == (4, 128, 3072)
        centered = correct - correct.mean(1, keepdim=True)
        shared_template = centered.mean(0, keepdim=True)
        interaction = centered - shared_template
        energy = centered.square().mean()
        rows.append(dict(
            variant=variant, count=4, condition_rms=rms(correct),
            token_centered_rms=rms(centered), token_centered_fraction=rms(centered)/rms(correct),
            input_shared_slot_template_rms=rms(shared_template),
            input_slot_interaction_rms=rms(interaction),
            shared_slot_template_energy_fraction=float(shared_template.square().mean()/energy),
            input_slot_interaction_energy_fraction=float(interaction.square().mean()/energy),
            caption_change_condition_rms=rms(correct-wrong),
            caption_change_mean_condition_rms=rms(correct.mean(1)-wrong.mean(1))))
    result = dict(
        source_files=[str(path) for path in files],
        scope='Four saved validation cases from the fixed final224 Q checkpoint; not the VideoPhy40 benchmark.',
        decomposition='D[s,i,d] = C[s,i,d] - mean_i(C[s,i,d]); shared_template[i,d] = mean_s(D[s,i,d]); interaction = D - shared_template.',
        interpretation='Shared means common across these four inputs, not proven globally constant or semantically irrelevant. RMS and energy are representation diagnostics, not quality scores.',
        interventions='Inference-only functional query_seed rescaling with all trained weights frozen; no retraining.',
        results=rows)
    (ROOT / 'query_spread_reanalysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
