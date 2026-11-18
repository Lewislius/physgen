# ProPhy/VideoPhy2 600-ALL / 180-HARD: official Wan2.1-T2V-1.3B

This directory evaluates the official **Wan2.1-T2V-1.3B** checkpoint on the
VideoPhy2 prompt list used by ProPhy (arXiv:2512.05564):

- 600 unique captions from the official VideoPhy2 test CSV;
- 180 captions with `is_hard=1`;
- official VideoPhy2 AutoEval SA/PC integer 1–5; pass is `>=4`;
- Joint pass requires both SA>=4 and PC>=4 on the same video.

## Generation scope

Text-to-video only. No first image, no project LoRA/checkpoint, no prompt
extension. The model is the official Wan2.1-T2V-1.3B checkpoint with its own
UMT5 text encoder and Wan2.1 VAE, at 832x480, 81 frames, 16 fps, 50 UniPC steps,
shift 5.0, CFG 5.0, seed 42+index, official Wan2.1 negative prompt.

The ProPhy paper does not report model-specific sampling settings. The values
above follow the established local ProPhy replay protocol and official Wan2.1
T2V defaults; the evaluator is the same local VideoPhy2 AutoEval used by the
other ProPhy replay entries. No claim is made that the paper's judge is
identical.

## Files

- `protocol.json`: 600 cases, 180 hard, Wan2.1 model/sampling/evaluation spec
- `generate.py`: official Wan2.1 T2V generation, resumable per case
- `evaluate.py`: official VideoPhy2 AutoEval SA/PC, ALL + HARD summaries
- `run.py`: one diagnostic video, partial score, then full 600 generation/score
- `common.py`: paths, aggregation, GPU policy (ada-48g; H100 rejected)

Expected runtime is generation-bound (600 videos); the job runs on `ada-48g`.
