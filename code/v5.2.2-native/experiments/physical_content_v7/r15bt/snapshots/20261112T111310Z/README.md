# R15Bt: R15B + B400 joint read, retrained for T2V

This is the text-to-video counterpart of R15B + I.  The R15B enhancement and
the B400 joint direct/request-bound physical read are retained; the task entry
is T2V, so no first image, canvas, first latent, or first-frame-derived
condition exists.

## What changed relative to R15B + I

| Item | R15B + I (I2V) | R15Bt (T2V) |
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

## Training data pairing

R15B's 400-update set is 1600 samples drawn from the 3666 train rows.  R15Bt
uses the **existing processed v10 T2V data as-is**, with the same filtered
1600-row loader and `SampleOrder(seed=42)` as R18E/G:

- training latents/native T5: `data_revision_qwen38/runs/20261103_dataset/encoded`
- request-only evidence/native targets: `physics_content_v10_impl/runs/t2v_cache`

The training loader and the evidence cache cover exactly the same 1600 samples,
and `data_ready()` verifies every evidence caption and request span against the
training manifest.  No new data is prepared.  Because this is the
qwen38 reviewed revision, R15Bt is data-matched to the R18E/G T2V branch, not to
R15B's original WISA captions.  The only change from R15B is the task scope
(no image).

## Inference file separation from I2V R15B+I

- I2V R15B+I stays in `r15b/` (plus `b400_joint_read_eval/` for the joint-read
  inference); its config, prepared inputs, checkpoints and outputs are untouched.
- T2V R15Bt has its own `configs/`, `prepared_inputs/`, `eval_native_inputs/`,
  `runs/`, `inference_outputs/` and snapshot under `r15bt/`.
- T2V inference reads only `r15bt/eval_native_inputs/<case>/conditions.pt`,
  whose keys are exactly `text`, `null_text`, `times`.  It never reads
  `first`, `first.png`, a canvas, or the I2V prepared-input folders.
- The only shared read-only inputs are the base Wan/Qwen models and the
  qwen38 T2V data cache shared with R18E/G; no I2V inference file is written
  or consumed by T2V.

## Commands

```bash
# The existing v10 T2V processed cache is used as-is; no data build.

# 1. Fit the request-only Qwen4096 -> Wan3072 interface (CPU, ~20 s)
python -I -B -u fit_interface.py --config configs/r15bt.yaml

# 2. Pack the 60 evaluation conditions (CPU)
python -I -B -u prepare_inputs.py --config configs/r15bt.yaml

# 3. Train 400 updates, generate fixed40, score, generate disjoint20, score
python -I -B -u run_candidate.py --config configs/r15bt.yaml

# Optional: inspect the declared plan without running
python -I -B -u run_candidate.py --config configs/r15bt.yaml --describe

# Tests
python -I -B -u -m unittest discover -s tests -v

# R18E/G T2V data/eval consistency audit (must print "ok": true)
python -I -B -u verify_r18_consistency.py
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

Data-revision caveat: R15Bt uses the existing qwen38 reviewed T2V data, the
same revision as R18E/G T2V.  R15B was trained on the original WISA captions,
so a direct R15B-vs-R15Bt score comparison is not purely a task/paradigm
difference.

## Scope limitation

This is the fixed 60 historical VideoPhy2 captions, not the official
590-caption T2V leaderboard.  The judge is the local official VideoPhy2
AutoEval checkpoint; no claim is made that it is the same judge used by ProPhy
or any external report.  No target score is predeclared; I2V results are not
substituted for T2V results.
