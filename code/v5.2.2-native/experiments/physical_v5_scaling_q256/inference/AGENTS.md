# Final Q256 inference for v5_ce

- User authorizes final1000 checkpoint from experiment 66706, demo21 then fixed VideoPhy2-40 then independent scoring.
- Preserve exact training architecture, native text and every learned weight; no generation verifier or training targets.
- One blk-96g slot, RTX PRO6000/RTX6000 Blackwell only; H100 must exit before foundation loading and be resubmitted.
- Preserve all training sources, original inference, caches and snapshots. New results/cache writes stay in the owning scaling directory.
- Published inference snapshots are immutable. Keep the existing environment and use direct det experiment create.
