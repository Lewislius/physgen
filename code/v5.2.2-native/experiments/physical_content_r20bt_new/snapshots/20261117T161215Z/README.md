# R20BT-new

R20BT corrected to depth-matched, per-site independent physical blocks.

- Each physical block copies `wan.blocks[site]` for `site = 2/8/14/20/26`.
- Each site runs its own block on that site's actual native block input.
- No shared serial physical stream.
- No shared `NoiseConditioning`; `block.modulation + native_mod` drives the
  physical self-attn/FFN and the C reader query/gate.
- Shared C, text context, grid, freqs and lengths (matching R20FT).
- Per-block `times` embedding is retained.
- T2V only.
- R20BT's single conditioned forward, frozen physical caption attention and
  zero-initialised physical output are intentionally preserved.

Read `DESIGN.zh.txt` for the full design and the NoiseConditioning / times
analysis.

## Quick checks

Use the project interpreter:

```bash
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_content_r20bt_new
PY=/home/liuzhirui/miniconda3/envs/moviestory/bin/python
$PY -I -B -u train.py --config configs/r20bt_new.yaml --check-only
$PY -I -B -u tests/test_training_path.py
$PY -I -B -u run_candidate.py --config configs/r20bt_new.yaml --describe
```

The small actual-Wan CPU test writes `reports/training_path_tests.json`.
The full-size disposable gradient/memory check runs automatically in
`run_candidate.py` before the formal 400-update training.

## Submission

`submissions/r20bt_new_blk96g.yaml` is a template. Freeze the workspace to a
snapshot and update the entrypoint paths before submitting; the current file
points at the working directory, not a frozen snapshot.

## Status

- Code and configuration: complete.
- Small actual-Wan CPU checks: passed.
- Formal 400-update training: not submitted.
- Quality/scores: unmeasured.
