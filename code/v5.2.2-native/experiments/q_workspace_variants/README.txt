V5.2.2-native isolated Q workspace variants

Design: DESIGN.html (Chinese; exact formulas, pooling, round semantics and links)
Manifest: variants.json
Training code: train/train.py; physgen_native/message_conditioning.py; physgen_native/q_variants.py

B    configs/b.yaml    train/train_b.sh    train/b_1x80g.yaml    amp-80g, 1 GPU
C    configs/c.yaml    train/train_c.sh    train/c_1x96g.yaml    blk-96g, 1 GPU
D-N  configs/d_n.yaml  train/train_d_n.sh  train/d_n_1x96g.yaml  blk-96g, 1 GPU
D-S  configs/d_s.yaml  train/train_d_s.sh  train/d_s_1x96g.yaml  blk-96g, 1 GPU
D-I  configs/d_i.yaml  train/train_d_i.sh  train/d_i_1x96g.yaml  blk-96g, 1 GPU

Usage (from this directory, or invoke SH by absolute path):
  bash train/train_b.sh --check
  bash train/train_b.sh --submit
  bash train/train_b.sh --local
  bash train/train_b.sh --resume /absolute/path/to/B/run

Replace b with c / d_n / d_s / d_i for the other versions.
--local / --resume require the corresponding already allocated single GPU.
No argument defaults to --check. --check never submits a job.
Every SH supplies the original generator initializer for a fresh run.
Resume instead restores the saved config, model, optimizer and RNG.

All variants: 128 slots, 4096 workspace, final 4096->3072 projector;
R0 + two refinements; seed 42; accumulation 4; 64 interface + 160 joint steps;
FM + 0.1 temporal FM; no verifier and no extra losses.
Persistent identity starts at R1, is normalized without learned affine,
and is added AFTER LayerNorm. D-I adds it to self Q/K only, never directly V.
Validation generation always uses seed 42, with no per-case seed offsets.

Original design files, models, checkpoints and caches remain unchanged.
This directory is an independent source snapshot; use its loader for new revisions.
The runtime reuses the existing moviestory environment via v5.2/tools/runtime_env.sh.
No dependencies or weights are downloaded.

Final224 demo21 + VideoPhy2-40 + independent AutoEval, one amp-80g GPU per method:
  See inference/INFERENCE.txt and reports/inference_submissions_20261028.json.
