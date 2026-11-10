# ProPhy/VideoPhy2 600-ALL / 180-HARD replay

This directory evaluates the **R15B + B400 joint read** checkpoint on the
VideoPhy2 prompt list used by ProPhy:

- 600 unique captions from the official VideoPhy2 test CSV;
- 180 prompts with `is_hard=1`;
- official VideoPhy2 AutoEval checkpoint;
- SA/PC are integer 1–5; pass means `>=4`;
- Joint pass means `SA >= 4 and PC >= 4`.

## Important scope limitation

R15B is an **image-to-video** model, while ProPhy's reported VideoPhy2 numbers
are text-to-video. The 600 prompts are exact, but this replay must use a first
image. It reuses the independently generated shared first images from the
existing `physvid_i2v_599` protocol. Therefore this is an **I2V approximation
of the ProPhy prompt protocol**, not an identical T2V rerun.

The evaluator is the official local VideoPhy2 AutoEval. ProPhy's paper does not
name its judge; no claim is made that the judge is identical.

## Files

- `protocol.json`: 600 cases, 180 hard, mapped to shared I2V source cases
- `generate.py`: R15B checkpoint + B400 joint read, 50 steps, shift5, gamma5
- `evaluate.py`: official AutoEval SA/PC, ALL and HARD summaries
- `run.py`: one diagnostic video, then full generation and scoring
- `joint_read.py`: immutable B400 joint direct/request-bound read

## Expected runtime

Generation is about 100–150 seconds per video on ada-48g; 600 videos plus
AutoEval is expected to take roughly **20–24 hours**. The Determined job name
includes `eta-24h`.
