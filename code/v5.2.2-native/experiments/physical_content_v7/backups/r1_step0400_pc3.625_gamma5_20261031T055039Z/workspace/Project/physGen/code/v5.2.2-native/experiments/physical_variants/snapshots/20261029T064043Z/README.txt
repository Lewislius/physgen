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
  bash ../../tools/runtime_python.sh submit.py
  Optional submit.py --variants v6_assoc (only the named experiment).
  submit.py freezes a source snapshot, uses the established det0.32.1 and image,
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
