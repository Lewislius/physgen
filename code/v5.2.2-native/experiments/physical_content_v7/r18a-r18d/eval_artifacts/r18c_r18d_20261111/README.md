# R18C / R18D generated videos (snapshot 2026-11-11)

This directory contains the generated VideoPhy2 fixed-case videos for R18C and R18D, plus the available R18D scores.

## Snapshot status

| Arm | Suite | Generated in this snapshot | Status |
|---|---|---:|---|
| R18D | fixed 40 (`videophy40`) | 40/40 | generation complete; scoring complete |
| R18D | extra 20 (`videophy60`) | 4/20 | still running when snapshot was taken |
| R18C | fixed 40 (`videophy40`) | 16/40 | still running when snapshot was taken |

- Snapshot time: `2026-11-11T13:06:52.603880+00:00`
- Missing videos are not zero-length placeholders; they simply had not been generated yet.
- R18C outputs currently show systematic high-frequency dot/checkerboard corruption after the first frame. They are included for inspection/reproduction only; R18C has no score yet.

## R18D fixed-40 scores

| Metric | R18D | R15B + B400 joint read | R15B base | R15I |
|---|---:|---:|---:|---:|
| PC mean | **3.375** | 3.775 | 3.700 | 3.475 |
| SA mean | **2.975** | 2.900 | 2.925 | 3.025 |
| Joint pass (PC>=4 and SA>=4) | **9/40** | 10/40 | 10/40 | 10/40 |
| PC pass rate | 0.400 | 0.600 | 0.575 | 0.425 |
| SA pass rate | 0.250 | 0.275 | 0.275 | 0.275 |

Same 40 captions, seed 42 and scorer as the historical anchors.

Key result: R18D's joint-pass set is a strict subset of R15B+B400 (lost `VP008`, gained none). PC is 0.400 below R15B+B400 and 0.325 below the R15B base model.

Known severe visual artifacts from manual review: VP004, VP009, VP023, VP024, VP028. See `visual_review/` for contact sheets.

## Generation protocol

- 400 training updates, final checkpoint `step0400`
- 81 RGB frames, 832x480, 24 fps
- 50 sampling steps, shifted Euler, shift 5.0, CFG 5.0, seed 42
- Joint positive/negative native-T5 + Qwen condition guidance
- Fixed first image and fixed captions from the historical 60-case VideoPhy2 I2V adaptation

## Layout

```text
index.html                    portable gallery (open locally or serve this directory)
data/cases.json               captions, availability, R18D scores
data/snapshot.json            exact snapshot counts/status
scores/r18d_fixed40_scores.json
scores/r18d_fixed40_summary.json
evaluation/videophy40/       per-case PC/SA verdicts, CSV, evaluator log
videos/r18c/VPxxx.mp4
videos/r18d/VPxxx.mp4
videos/r18d_extra20/VPxxx.mp4
visual_review/                R18D contact sheets
```

## Per-case evaluation results

`evaluation/videophy40/` contains the raw VideoPhy2 scorer verdicts for all 40 R18D cases:

- `cases/VPxxx/planned.pc.json` and `planned.sa.json` — original per-case `raw`/`score` files
- `per_case_scores.csv` — flat PC/SA table
- `scores.json` — aggregate summary and per-case scores
- `original40_scoring.log` — evaluator log with the full per-case run

To browse locally:

```bash
python -m http.server 8780 --bind 127.0.0.1 --directory .
# open http://127.0.0.1:8780/
```

Or open `index.html` directly in a browser. Video paths are relative.

## Important scope note

These are the fixed 60 historical VideoPhy2 captions adapted to a common first-image I2V protocol, not the official 590-caption T2V benchmark. Scores are comparable to the historical R15B/R15I fixed-40 anchors only.
