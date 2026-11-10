V6-WRITE 66674/68061 recovery, 2026-10-30

Original Python process was SIGKILLed (exit137) while preparing the teacher target for wisa0023119 at step888 micro3, after completing step887. Losses/gradients were finite. No CUDA OOM traceback. Original run had no process/cgroup telemetry, so the sender and underlying cause are unconfirmed; host OOM is not established. The exact original frame decoder successfully decoded all28 frames of this sample, peak process RSS641500KiB.

Resume latest.pt: step850, cursor3400, exact same model/config/AdamW states/Python+Torch+CUDA RNG. Repeat851-887 and proceed to1000. Model/data/teacher losses/LR schedules are identical. Memory-map the checkpoint CPU storage, record restore/teacher stages/process memory, save latest every10 updates as a runtime setting (no training config change). Old files remain unchanged. One blk-96g RTX6000 only; H100 rejected before model loading.

Submitted experiment66707/trial68094, actual RTX PRO6000 verified. Final status in reports/recovery_v6_write_20261030.json and runs/v6_write_resume850_66707/status.json.
