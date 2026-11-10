"""Inspect the geometry of the existing semantic auxiliary target, without fitting."""
from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from physical_content.evidence import resample_words
from analyze_content import statistics


@torch.no_grad()
def main():
    torch.set_num_threads(8)
    cache = ROOT / 'cache'
    interface = torch.load(cache / 'interface.pt', map_location='cpu', weights_only=True)
    train = [json.loads(line) for line in (HERE / 'runs/physical_content_66870/train.jsonl').read_text().splitlines()]
    recent = [x['sample'] for x in train if x['kind'] == 'micro' and not x['text_dropped']]
    # Fixed latest16 distinct training examples; the fixed16 test inputs are diagnostic only.
    names = list(dict.fromkeys(reversed(recent)))[:16]
    cohorts = {'recent_train16': [x + '_text' for x in names],
        'fixed_test16': ['videophy40_' + x['id'] + '_text' for x in json.loads((HERE/'subset16.json').read_text())['cases']]}
    result = {}
    for cohort, samples in cohorts.items():
        rows = []; mean_errors = []
        for name in samples:
            saved = torch.load(cache/'words'/f'{name}.pt', map_location='cpu', weights_only=True)
            target = torch.load(cache/'native_words'/f'{name}.pt', map_location='cpu', weights_only=True)['native_words'].float()
            words = (saved['words'].float()-interface['mean'])/interface['scale']
            anchor_words = F.linear(words, interface['weight'], interface['bias'])
            anchor, _, _ = resample_words(anchor_words, saved['word_groups'])
            target, _, _ = resample_words(target, saved['word_groups'])
            desired = target-anchor
            mean_errors.append(desired[0].mean(0))
            rows.append(dict(name=name, anchor=statistics(anchor), target=statistics(target),
                raw_word_geometry=statistics(saved['words']),
                layernorm_raw_word_geometry=statistics(F.layer_norm(saved['words'].float(),(saved['words'].shape[-1],))),
                centered_scaled_word_geometry=statistics(words),
                layernorm_centered_scaled_word_geometry=statistics(F.layer_norm(words,(words.shape[-1],))),
                desired_semantic_correction=statistics(desired),
                initial_normalized_mse=float(desired.square().mean()/interface['target_scale'].square().clamp_min(1e-8))))
        result[cohort] = dict(count=len(rows), records=rows,
            mean_desired_correction_token_cosine=sum(x['desired_semantic_correction']['pair_cosine'] for x in rows)/len(rows),
            mean_desired_correction_centered_energy=sum(x['desired_semantic_correction']['centered_energy_fraction'] for x in rows)/len(rows),
            mean_word_token_cosines={key:sum(x[key]['pair_cosine'] for x in rows)/len(rows)
                for key in ['raw_word_geometry','layernorm_raw_word_geometry','centered_scaled_word_geometry','layernorm_centered_scaled_word_geometry']},
            cross_sample_mean_errors=statistics(torch.stack(mean_errors)),
            initial_normalized_mse=sum(x['initial_normalized_mse'] for x in rows)/len(rows))
    report = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), cohorts=result,
        scope='Existing input-derived semantic auxiliary only; no training or model inference. Desired correction is native-word target minus frozen semantic anchor. Its geometry tests whether this auxiliary target itself requires a nearly common correction; it does not attribute the actual generator gradient or establish physical correctness.')
    path=HERE/'reports/semantic_target_geometry.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='records'} for k,v in result.items()},indent=2))


if __name__=='__main__':main()
