v5_ce: final step1000 from training 66706; exact submitted model source snapshots/20261029T142617Z.
Single blk-96g slot; actual RTX6000 check before foundation loading (other GPU exits86).
Sequential demo21 -> fixed VideoPhy2-40 -> independent SA/PC scoring -> comparison with17 existing references.
Q256, two enhancement rounds, seed42, native text, 50 shifted Euler steps, shift5, CFG5, 81 frames at24fps, 832x480.
VideoPhy2 reuses the saved per-case FP32 noise. Training targets and auxiliary losses are absent from inference.
Strictly restore every trained parameter, including native Wan LoRA and all physical fusion weights.
Use run_v5_ce_blk96g.sh --check / --local / --submit. Submit immutable snapshots using det experiment create YAML EMPTY_CONTEXT.
Output: /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/physical_v5_scaling_q256/inference_outputs/v5_ce_66706/final1000_demo21_videophy40_r1
Scoring: videophy40/scores.json, summary.csv, comparison.json, scores_by_case_40.csv.
