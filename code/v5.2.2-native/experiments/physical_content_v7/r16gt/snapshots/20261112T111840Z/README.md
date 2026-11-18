# R16Gt: R16G retrained for text-to-video

This is the text-to-video counterpart of R16G.  R16G's core design is retained
unchanged: three independently normalized global physical reads (direct C,
request-bound C, joint direct+request C), each with its own zero-init full-width
output and multiplicative state gate, plus the R15C 5x3x3 local video-state
operator whose local Q receives only the direct C condition.

## What changed relative to R16G (I2V)

| Item | R16G (I2V) | R16Gt (T2V) |
|---|---|---|
| first latent | 1 known frame | none; all 21 frames noisy |
| patch embedding | first frame + noisy future | full noisy latent |
| time positions | future frames only | all 21 frames |
| physical write | future tokens only | all tokens |
| generation loss | excludes frame 0 | all 21 frames |
| Qwen evidence | first image + request | request only |
| interface | I2V fit | fresh T2V fit |
| sampler | 50 steps, shift5, CFG5 | identical |
| prompts | fixed VideoPhy2 40 + 20 | identical |
| noise | persisted seed42 | identical persisted noise |

Nothing else in the R16G model, objective, optimizer budget or evaluation
protocol is changed.

## Training data pairing

R16G's I2V run draws 1600 micro-batches from the 3666-row WISA train split with
`SampleOrder(seed=42)`.  Those 1600 draws are 1600 unique samples; the CPU
contract test verifies that this set is exactly the T2V evidence-complete 1600
rows.  R16Gt uses the existing R18E/G T2V loader: filter the 3666 rows to those
1600 evidence-complete samples, then `SampleOrder(seed=42)` over the 1600 rows.

- training latents/native T5: `data_revision_qwen38/runs/20261103_dataset/encoded`
- request-only evidence/native targets: `physics_content_v10_impl/runs/t2v_cache`

The sample set matches R16G; the per-step order is the R18E/G T2V order.  The
caption revision is qwen38 reviewed, so R16G-vs-R16Gt is not a pure
task/paradigm difference (the same caveat applies to R15B-vs-R15Bt).

## Commands

```bash
# 1. Fit the request-only Qwen4096 -> Wan3072 interface (CPU, ~3 min)
python -I -B -u fit_interface.py --config configs/r16gt.yaml

# 2. Pack the 60 evaluation conditions (CPU)
python -I -B -u prepare_inputs.py --config configs/r16gt.yaml

# 3. Train 400 updates, generate fixed40, score, generate disjoint20, score
python -I -B -u run_candidate.py --config configs/r16gt.yaml

# Optional: inspect the declared plan without running
python -I -B -u run_candidate.py --config configs/r16gt.yaml --describe

# CPU tests
python -I -B -u -m unittest discover -s tests -v

# Disposable full-size GPU preflight (no formal weights)
python -I -B -u gpu_preflight.py --config configs/r16gt.yaml --output reports/preflight
```

Training and evaluation are guarded to `blk-96g` RTX PRO 6000 / RTX6000
Blackwell.  The guard rejects H100 before loading either foundation.

## Outputs

- `runs/physical_content_<experiment>/step0400.pt`
- `inference_outputs/<run>/step0400_fixed40_extra20_seed42/videophy40/scores.json`
- `.../videophy60/scores.json` (the disjoint 20)
- `.../full60/videophy60/scores.json` (merged 60)
- `.../report.json`

The merged 60 report is produced by this directory's `report_results.py`; it
does not depend on the older `merge_completed60.py` guidance schema.

## Scope limitation

This is the fixed 60 historical VideoPhy2 captions, not the official
590-caption T2V leaderboard.  The judge is the local official VideoPhy2
AutoEval checkpoint; no claim is made that it is the same judge used by ProPhy
or any external report.  No target score is predeclared; I2V results are not
substituted for T2V results.
