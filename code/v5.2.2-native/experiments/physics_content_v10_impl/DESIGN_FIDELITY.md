# R18E-H NoChunks-v2 fidelity audit

Map: R18E=T1, R18F=I1, R18G=T2, R18H=I2.  Source: `experiments/physical_content_v10` (`COMMON_CONTRACT.md`,
`MECHANISM_1.md`, `MECHANISM_2.md`, `PLAN_T1/I1/T2/I2.md`, and the four YAMLs).

## Intentional change requested by the user

The v10 plan is defined over 1000 updates.  The user requested a **newly
compressed 400-step schedule**, not a truncation.  Phase boundaries are scaled
by 400/1000 = 0.4 and rounded to integer updates:

- video-LoRA warmup = 51;
- physics warmup = 13;
- peak phase through update 240;
- cosine LR decay from 241 to 400, reaching 0.2x;
- route warmup 13, hold to 240, linear decay to 0.015 at 400;
- final checkpoint = `step0400.pt`, effective batch = 4, exposures = 1600.

These values are asserted by `physical_content/common.py` and
`physical_content/schedule.py`.

## Task boundary

| requirement | implementation |
|---|---|
| T2V has no first image | `data.TrainingData` does not return `first`; `model._geometry` rejects a non-None first; `infer` never clamps |
| T2V supervises frame 0 | `supervision.generation_loss(task='t2v')` keeps the full temporal axis |
| I2V clamps one latent frame | `infer` writes `first` into latent frame 0 at every step; training passes `first` separately |
| I2V supervises future only | `supervision.generation_loss(task='i2v')` drops frame 0 |
| I2V physical writes future only | readers receive `future_start=known`; mechanism writers scatter zeros into the first frame; the mechanism-2 FFN delta is masked |
| T2V physical writes all positions | readers receive `future_start=None`; FFN mask is all ones |

## Mechanism 1

`physical_content/mechanisms.py::Mechanism1` and
`physical_content/readers.py` implement:

- direct physical read `zd` with group prior and a zero reference key;
- request binding `Kt_valid Kc^T`, `Vp_request = Binding Vc`;
- native text association `Atext`, then `zs = Atext PadZero(Vp_request)`;
- shared per-layer `Wstate`, independent `Wd`/`Ws` writers, all bias-free and
  zero-initialized;
- `Hmix = Hsa + text_update + mask(delta_d + delta_s)`;
- native FFN residual after `Hmix`.

The code does not add a copied video block, a temporal partition, or a second
text attention.

## Mechanism 2

`physical_content/mechanisms.py::Mechanism2` and
`physical_content/readers.py::JointPhysicalReader` implement:

- `Vbound = softmax(Kt_valid Kc^T + b_c) Vc`;
- `Kj = [Kc, Kt_valid, k0]`, `Vj = [Vc, Vbound, 0]`;
- direct/request priors `-log(n_g)-log2`, `log(R/Lvalid)-log2`, reference 0;
- `direct = Wo[z + Wstate(LN0(x)) z]`;
- native FFN `u -> f -> y` with the main LoRA already inside `W1/W2`;
- `delta_ff = Bp[(Ap f) * (Rp LN0(z_full))]`, rank 64, no bias, `Bp=0`;
- `Hout = Hmix + native_ffn_gate * (y + mask*delta_ff)`.

No copied FFN, no physical-language vector through the FFN, and no extra video
attention.

## Shared physical/content contract

- R10A-shaped `Enhancement`: 256 slots, 4096-wide Q, 32 heads (16/16), two
  rounds, 3072-wide `Cr/Ce`, zero-initialized `Pr/Pe/shared b`.
- Full-width reader: Q from frozen native Q plus video-time embedding; K/V
  dense trainable copies of the official K/V; independent trainable key norm;
  group prior; no native text LoRA.
- Main video LoRA: rank32/alpha32 on `self_attn.q/k/v/o` and `ffn.0/ffn.2` on
  all 30 layers, from update 1. Native text Q/K/V/O and norms remain frozen.
- No ST, no local loss, no post-training, no extra training views, no
  best-of-N, no evaluator gradient.

## Training

- `physical_content/training.py`: one loop, micro1 x accum4, fresh official
  init, same-run resume only.
- `physical_content/objectives.py`: one real-video forward/backward and the
  raw-`Cr` route auxiliary only.
- `physical_content/supervision.py`: 0.8 FM + 0.2 four-scale top-25% tail,
  task-parameterized temporal mask.
- Peak LRs: video LoRA 2e-5, enhancement 5e-5, physics read 2e-5, physics write
  3e-5, mechanism-2 adapter 2e-5.

## Verification performed

1. `python -m unittest discover -s tests -p 'test_*.py'` -> **8 tests pass**.
2. A tiny fake-Wan full forward/backward was run for both mechanisms and both
   T2V/I2V task modes; all outputs finite and gradients reached the expected
   optimizer groups.
3. R18F and R18H closed-form interfaces were fitted on the real 1600-sample train
   split; `train.py --check-only` reports 1600 available records for both.
4. The fixed 60 I2V evaluation cases were compacted for R18F and R18H (60/60 each).
5. A checkpoint save/restore round-trip was executed with the fake model; all
   learned parameter names and the update/cursor restored.
6. A one-example CPU smoke test of the request-only Qwen builder produced four
   physical sentences for the positive arm and four empty relation slots with
   no request words for the null arm (no image was supplied to Qwen).

## Not yet executed

- The T2V request-only Qwen cache builder has only been smoke-tested on one
  example, not run for all 1600+60 samples.  It requires Qwen generation; the
  code and paths are present but this is a data-production step, not a code
  claim.
- No real Wan GPU forward/backward has been run for this new package yet.
- No formal `blk-96g` experiment has been submitted, per the user's request.
- No VideoPhy2 score, WMB/Physics-IQ/VBench score, or quality claim is made.
