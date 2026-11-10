V6 association Q256 final1000 inference, training66678/trial68065

Entry: run_v6_assoc_blk96g.sh --check or --local. YAML uses one blk-96g GPU. Actual RTX PRO6000/RTX6000 Blackwell checked before model loading; H100 exits86.

Pin training snapshot20261029T140406Z and exact final step1000.pt (all464 learned tensors,378252288 parameters). Two Q refinement passes,256 context slots, independent C K/V with LoRA, relation grouping/cardinality correction and zero-value reference key. Preserve native T5 and all Wan LoRA. Frozen Qwen input encoder reads first image/request only; read-only reuse of original inference input evidence, new cache writes local. No training-video target or generation verifier.

Sequential stages:21 demo videos, then40 fixed VideoPhy2 videos, then independent SA+PC evaluation, then aggregate/compare. Same historical first frames/T5/FP32 saved comparison noise, seed42;832x480,81frames,24fps,50 shifted Euler steps,shift5,CFG5. Evaluation uses the unchanged independent AutoEval checkpoint/32frame/seed20261009 protocol.

Artifacts: demo21/summary.json; videophy40/summary.json; videophy40/scores.json (SA/PC means and joint pass rate); videophy40/comparison.json,summary.csv,comparison.txt (SA mean,PC mean,total mean=(sumSA+sumPC)/80,jointSA>=4 and PC>=4, and16 existing references). This is the existing fixed40 I2V comparison protocol, not the full official T2V leaderboard.

Existing model/training files/caches/results are unchanged. Each generated case is committed with provenance metadata; completed cases can resume. Generation process exits before evaluator loading to release GPU memory.
