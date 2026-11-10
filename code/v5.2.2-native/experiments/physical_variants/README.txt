Physical variants, 2026-10-29

User-authorized implementations of all three proposals, kept separate from old
experiments. These are fresh project training runs from the official Qwen/Wan
foundations. No old project initializer, optimizer, LoRA, alignment or verifier.
The existing WISA cache remains read-only (3666 train / 191 validation).

Variants:
  v5_ce: FM + 0.05 C-only language CE. Full independent C K/V/O; frozen Qwen
         decoder retains input gradients through a 3072->4096 bridge. CE is
         computed after freeing the Wan graph and recomputing the same C.
  v6_write: FM + 0.02(1-sigma) SmoothL1(actual deltaH, synthetic D*)/teacher scale.
         D*=support * frozen official text V/O anchors. C K/V use own rank32
         increments; full independent O. This deliberately reproduces the initial
         V6 design requested for training. Its target is synthetic, not established
         correct physical response or amplitude. No positive support -> skip aux.
  v6_assoc: FM + 0.02(1-sigma) KL(teacher association || actual grouped attention).
         Input-side relation sentences initialize groups of the 128 Q slots.
         The same sentences are queried against real training videos by the
         teacher. One extra reference KEY has identically zero VALUE (129 keys,
         still 128 C tokens). Per-slot -log(group_size) makes group priors equal.
         Values/amplitudes are learned by FM; no synthetic deltaH regression.

Shared architecture:
  Frozen Qwen first-image/request/brief evidence H[M,4096].
  Learned Q[128,4096], one cross+self+FFN(4096->8192->4096) block used twice.
  RMSNorm -> projector4096->3072 -> RMSNorm -> actual injected C[128,3072].
  Native T5 text context is retained, with its own unchanged cross-attention.
  Physical fusion in Wan layers 3,6,...,30: current native query reads C through
  independent K/V; full3072 output residual precedes the native FFN. No direct
  physical write to first-frame tokens. O starts at zero; time increment starts
  at zero. Native temporal positional encoding and diffusion time are retained.
  Per-frame PTS and deltaPTS -> sin/cos256 -> 3072 added only to C query.
  PTS is presentation time, not an asserted physical clock.

Two stages: 64 interface steps, 160 joint steps, accumulation4, seed42.
  A: Q/P/fusion/time (+ CE bridge in V5); Wan frozen.
  B: continue those modules; warm up fresh Wan self-attention and FFN rank32 LoRA.
  Native text cross-attention parameters remain frozen. No Qwen LoRA.
  Mixed bf16 compute / FP32 trainable optimizer parameters, per-block gradient
  checkpointing; single visible GPU. No environment installations or upgrades.
  Synchronized 10% request dropout across C and native text, auxiliary skipped
  on dropped requests. Sigma sampled in (0.001,0.999). Two losses only; no
  temporal FM, source/scope/phase heads, trackers, flow or speed labels.

Supervision preparation:
  Frozen input evidence is cached with exact caption and relation text, from
  first image + request only. All variants use the same short relation brief;
  only the revised version uses its group identities.
  V5 targets use WISA phys_law only as a caption-derived candidate; Qwen receives
  first image/request and writes conditional explanation text (128-token cap).
  V6 teacher phrases originate in the real training crop. Revised phrases are
  exactly the input-side ones. Teachers read the actual 28 RGB observations
  (every third frame plus last) at cached832x480 geometry.
  Extract selected phrase->video attention rows from the last four frozen Qwen
  language layers, averaged over heads; never allocate full square attention.
  A generic scene phrase supplies a separate background reference read.
  Map these weak associations with cached PTS/geometry to21x15x26; exclude the
  first390 tokens. Interpolation does not recover unobserved event timing.
  Revised P* = softmax([0, log(a_r+eps)-log(b+eps)]/T), T=1.
  Teacher cache is generated lazily before a sample's training graph, under
  no_grad, and reused across runs. This has real preprocessing cost. Real future
  video/teacher targets never enter the input evidence or sampling functions.

Execution:
  bash ../../tools/runtime_python.sh tests/test_physical_training.py
  bash ../../tools/runtime_python.sh train.py --config configs/v5_ce.yaml --check-only
  det -m http://10.130.130.5:4096 experiment create jobs/v5_ce_1x96g.yaml submission_context
  det -m http://10.130.130.5:4096 experiment create jobs/v6_write_1x96g.yaml submission_context
  det -m http://10.130.130.5:4096 experiment create jobs/v6_assoc_1x96g.yaml submission_context
  Actual submissions use timestamped YAML files referencing immutable source
  snapshots; reports/latest_submission.json records their exact paths and IDs.
  Use direct det CLI commands, as requested; do not run det under the training
  interpreter's Python isolation environment. The established CLI is0.32.1.
  Optional submit.py automates source snapshots and GPU-policy retries; it uses
  the same direct det experiment create command with a sanitized CLI environment.
  Both paths use the established image,
  slots_per_trial1, resource_pool blk-96g. The process checks actual device model
  before loading foundations; H100/other cards exit86. The submission controller
  kills only its rejected experiment and resubmits until RTX PRO6000 / RTX6000
  Blackwell is confirmed. It never alters agents or other users' jobs.

Artifacts:
  reports/latest_submission.json: experiment IDs, exact run paths, confirmed GPUs.
  runs/<id>/train.jsonl: losses, gradients, memory, LR, teacher preparation.
  runs/<id>/latest.pt: trainable model state + optimizer/RNG/cursor for exact resume.
  step0064.pt / step0224.pt: stage/final checkpoints; configs identify variant.
  At steps64 and224: four fixed held-out FM measurements and one matched-seed
  sampled video, using first image/request only. These are not benchmark scores.
  Checks cover code/gradient correctness. They do not establish quality uplift.

Final-checkpoint inference (2026-10-29):
  inference/run_v5_ce_blk96g.sh
  inference/run_v6_write_blk96g.sh
  inference/run_v6_assoc_blk96g.sh
  Each accepts --check, --local or --submit. Submission is a direct
  det -m http://10.130.130.5:4096 experiment create YAML submission_context.
  reports/inference_submissions.json records actual experiments and snapshot
  YAML/SH paths. Each uses its own step0224.pt and the exact training source
  snapshot. Native foundations are reconstructed and ALL saved learned
  parameters are restored strictly, including Wan LoRA. No optimizer is loaded.
  One blk-96g slot; actual RTX PRO6000/RTX6000 Blackwell required before model
  loading. An H100/other allocation exits86 and must be terminated/resubmitted.

  Each job generates demo21 first, then the existing fixed VP001--VP040, then
  independent VideoPhy2 AutoEval scores and comparison tables. Generation and
  evaluation use separate processes on the same allocation to release GPU RAM.
  832x480,81 frames,24fps,50 shifted Euler steps,shift5,CFG5,seed42.
  VideoPhy2 replays the original saved FP32 noise. Native T5 and initial latent
  inputs are reused; raw first-frame canvases feed the frozen Qwen, exactly as
  in this training design. Old generated clauses and future targets are not
  conditioning. There are exactly two enhancement passes, with no extra R0.
  Independent benchmark scoring is not an inference-time generation verifier.

  Outputs: inference_outputs/<variant>_<training_id>/final224_demo21_videophy40_r1/
    demo21/reference/i2v/seed42.mp4 and demo21/P01..P20/i2v/seed42.mp4
    videophy40/cases/VP001..VP040/planned.mp4
    videophy40/scores.json: SA, PC and joint pass rate (both >=4)
    videophy40/comparison.json / summary.csv / comparison.txt
    videophy40/scores_by_case_40.csv / candidate_minus_reference.csv
    pipeline_status.json, generation_status.json, generation.log, evaluation.log
  Comparisons include nine archived methods plus four already-scored workspace
  variants. This is the existing 40-case shared-first-image I2V protocol, not a
  full official VideoPhy2 T2V leaderboard result. No new score exists until the
  generation/scoring pipeline finishes. Partial jobs resume completed videos
  with matching metadata and preserve the originally pinned checkpoint/protocol.
