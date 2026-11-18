# R18E-H: NoChunks-v2 T1/I1/T2/I2 implementation

This directory implements the four `physical_content_v10` designs against the
official Wan2.2-TI2V-5B.  It is a fresh implementation: no R-series checkpoint,
no project LoRA, no native text LoRA, and no local-segment loss.

## Plan mapping

| R18 | v10 design | task | mechanism | evidence scope | first latent |
|---|---|---|---|---|---|
| R18E | T1 | T2V | 1 | `request_only` | none |
| R18F | I1 | I2V | 1 | `first_image_and_request_only` | 1 known frame |
| R18G | T2 | T2V | 2 | `request_only` | none |
| R18H | I2 | I2V | 2 | `first_image_and_request_only` | 1 known frame |

## Compressed 400-step schedule

The v10 documents define a 1000-update plan.  The user requested a **newly
compressed 400-step schedule**, not a step-400 truncation.  Phase boundaries are
scaled by `400/1000 = 0.4` and rounded to integer updates:

| update | video LoRA | physics modules | route auxiliary |
|---|---:|---:|---:|
| 1..13 | warmup to 13/51 | warmup to 1/13 | rises to 0.03 |
| 14..51 | warmup continues | peak | 0.03 |
| 52..240 | peak | peak | 0.03 |
| 241..400 | cosine to 0.2x | cosine to 0.2x | linear to 0.015 |

The peak phase ends at update 240.  `keep_updates` is `[100, 200, 300, 400]`.
The LR floor is 0.2x and the route final value is 0.015, exactly as in the v10
plan after compression.

## Mechanisms

- **Mechanism 1**: direct physical read with a zero reference slot plus a
  request-associated physical value delivered through the native text
  association.  Two independent full-width zero-initialized writers and one
  shared per-layer state interaction.
- **Mechanism 2**: joint direct/request physical read with the B400+I group/request
  prior, one full-width writer, and a rank-64 adapter on the real post-GELU
  native FFN activation.

Both mechanisms run the official self-attention, text attention, and FFN once.
No extra video block, no copied Wan FFN, no temporal segmentation, no L_local,
no ST loss, and no post-training.

## Data and caches

R18F/R18H reuse the existing audited input-only Qwen cache and the existing
832x480/81-frame WISA latent cache.  R18E/R18G require a **new** request-only
Qwen cache; the builder never gives Qwen an image or future video:

```bash
python prepare_t2v_evidence.py --config configs/r18e.yaml --device cuda
python prepare_native_t2v.py  --config configs/r18e.yaml --device cuda
```

The interface is fitted separately for each task/plan:

```bash
python fit_interface.py --config configs/r18f.yaml
python fit_interface.py --config configs/r18h.yaml
```

Evaluation evidence is compacted with:

```bash
python prepare_inputs.py --config configs/r18f.yaml
python prepare_inputs.py --config configs/r18h.yaml
python prepare_inputs.py --config configs/r18e.yaml   # after T2V cache build
python prepare_inputs.py --config configs/r18g.yaml
```

## Training and evaluation

Each plan runs as one job:

```bash
python run_candidate.py --config configs/r18f.yaml
```

The runner fits the interface if missing, prepares fixed-40/extra-20 evidence,
trains exactly 400 optimizer updates with effective batch 4, generates the
fixed 40 and disjoint 20 cases at 832x480, 81 frames, 24 fps, 50 shifted-Euler
steps, shift 5, CFG 5, seed 42, and runs the independent VideoPhy2 scorer.

Submission YAMLs are present under `submission_r18*_400_*.yaml` for `blk-96g`;
the user asked not to submit them yet.

## Verification

- The 8 CPU tests check the v10 YAML contract, the compressed 400-step
  schedule, task-dependent first-frame masking, mechanism zero-init paths, the
  physical reader's future-only shapes, and tiny fake-Wan forward/backward for
  both mechanisms and both task modes.
- The fake-Wan forward/backward checks are also part of the 8 tests; gradients
  reached the expected optimizer groups and all outputs were finite.
- The R18F/R18H interfaces and fixed-60 evidence were materialized with the
  real train cache; the training audit reports 1600 available train records.
- A one-example CPU smoke test of the request-only builder produced four
  physical sentences for the positive arm and four empty slots with no request
  words for the null arm.

These are implementation checks, not evidence of physical-video quality or
benchmark scores.  No formal training job has been submitted.
