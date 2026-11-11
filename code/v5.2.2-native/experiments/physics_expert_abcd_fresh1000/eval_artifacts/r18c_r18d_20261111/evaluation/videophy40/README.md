# R18D VideoPhy2 fixed-40 evaluation results

Source: Determined job `69233`, checkpoint `step0400`, suite `videophy40`, generation seed 42.

- `scores.json` — aggregate summary plus all 40 per-case PC/SA scores.
- `cases/VPxxx/planned.pc.json` — raw per-case physical-commonsense verdict (`raw`, `score`).
- `cases/VPxxx/planned.sa.json` — raw per-case semantic-alignment verdict (`raw`, `score`).
- `per_case_scores.csv` — flat table for spreadsheets/analysis.
- `original40_scoring.log` — evaluator log with per-case lines and final summary.

Pass convention used in the summary: a case passes a metric at score >= 4; joint pass requires PC >= 4 and SA >= 4.

Summary: PC mean **3.375**, SA mean **2.975**, joint pass **9/40**, PC pass rate 0.400, SA pass rate 0.250.

This is the fixed-60 historical VideoPhy2 captions adapted to common-first-image I2V, not the official 590-caption T2V benchmark.
