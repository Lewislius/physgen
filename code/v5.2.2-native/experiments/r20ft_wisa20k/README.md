# R20FT WISA-20k preprocessing

Build the cache used by R20FT Wan2.2-TI2V-5B T2V training:

- **20,000 training clips total**
- **1,600 existing R20FT training clips** reused by symlink
- **18,400 newly processed WISA clips**
- New cache root: `/run/determined/NAS1/public/Dataset/WISA/data/r20ft_wisa20k/`

## What is produced

Only the tensors R20FT actually reads:

```text
r20ft_wisa20k/
  base/
    manifest.json                 # 20,000 records, split=train
    vae/<id>.pt                   # [1,48,21,30,52] Wan2.2 latent
    text/<id>.pt                  # [N,4096] UMT5 caption feature
    null_text.pt                  # symlink to existing cache
  reasoning/
    input/<id>_text.pt            # request-only Qwen evidence + relations
    input/<id>_null.pt            # empty request arm
    words/<id>_text.pt            # [N,4096] Qwen word features
    words/<id>_null.pt
    native_words/<id>_text.pt     # [N,3072] Wan2.2 native words
    native_words/<id>_null.pt
  work/
    candidates.json
    candidates.shard0.json
    candidates.shard1.json
    rows/<id>.json
    annotations/<id>.json
    sample_sequence_20000.json    # ids + drop + sigma
    audit.json
    caption_stats.json
  READY.json
  training_paths.json
```

Deliberately **not** produced: JEPA, first-image VAE, I2V controls,
requirements/tasks/change/video, and any other non-R20FT feature.

## Selection

- Source: `/run/determined/NAS1/public/Dataset/WISA/data/wisa-80k.json`
- 79,480 catalog videos, 412 GB, complete locally.
- Exclude all 3,857 source videos already in the approved cache.
- Prefilter: width >= 832, height >= 480, caption present, estimated frames >= 81.
- Deterministic shuffle with seed `20261117`; 22,400 candidates
  (11,200 per shard, 9,200 per-shard target, 2,000 surplus margin).
- Each clip: one centered contiguous 81-frame window at native fps,
  aspect-preserving resize + center crop to 832x480, no upscaling.

## Qwen parts

The local model is `Qwen3-VL-8B-Thinking`.

- **Caption**: Thinking weights in a no-thinking prefill (empty `<think>` block).
  A pure edit prompt made the model echo the original WISA caption; a
  from-scratch, image-grounded caption prompt is used instead, with the original
  caption only as a style reference. Target length is 1.10x the original, accepted
  ratio 0.88-1.18, with one image-grounded expansion/condensation pass if needed.
- **Reasoning**: full Thinking mode on the corrected caption. The R20FT T2V
  prompt is reused, with an explicit exactly-four-sentences / at-most-24-words
  constraint so the style matches the 1,600 existing clips.
- Null arms are generated without a second reasoning call (empty request and four
  empty relation slots).

Pilot measurements (L40S, 16 frames):

| item | existing 1,600 | new pilot |
|---|---|---|
| caption words median | 145 | ~110-173 (ratio 0.85-1.13) |
| caption ratio median | 1.047 | ~0.95-1.05 |
| relation count | 398/400 are 4 | exactly 4 |
| relation total words | median 58 | 39-79 |
| caption time | — | 10-45 s |
| reasoning time | — | 20-32 s |

## Stages

1. `index` (CPU, 16 workers): decode metadata, 81-frame center window, geometry,
   timestamps, manifest row.
2. `qwen` (one Qwen process per GPU): caption + reasoning + null evidence.
3. `text_native` (one UMT5 process per GPU): `base/text` and 3072-d native words.
4. `vae` (one Wan2.2 VAE process per GPU): 48-channel latent + first frame.
5. `finalize` (CPU): merge 1,600 existing + 18,400 new, manifest, links,
   sequence, audit, READY.

Each row records completed stages in `work/rows/<id>.json`; rerunning a shard
resumes by skipping completed rows/stages.

## Commands

Build candidates (CPU):

```bash
cd /home/liuzhirui/Project/physGen/code/v5.2.2-native/experiments/r20ft_wisa20k
python make_candidates.py --config configs/r20ft_wisa20k.yaml
```

Submit the two 4-GPU shards:

```bash
det -m http://10.130.130.5:4096 experiment create submissions/r20ft_wisa20k_shard0_ada48g.yaml .
det -m http://10.130.130.5:4096 experiment create submissions/r20ft_wisa20k_shard1_amp48g.yaml .
```

After both `work/SHARD_READY{0,1}.json` exist:

```bash
python finalize.py --config configs/r20ft_wisa20k.yaml
```

## Current jobs

| shard | experiment | pool | GPUs | status |
|---|---:|---|---:|---|
| 0 | 70556 | ada-48g | 4 | running index |
| 1 | 70557 | amp-48g | 4 | queued for 4 slots |

The `finalize.py` output is `READY.json` plus `training_paths.json`. A 20k
training config should set:

```yaml
train_cache: /run/determined/NAS1/public/Dataset/WISA/data/r20ft_wisa20k
input_cache: /run/determined/NAS1/public/Dataset/WISA/data/r20ft_wisa20k/reasoning/input
words_cache: /run/determined/NAS1/public/Dataset/WISA/data/r20ft_wisa20k/reasoning/words
native_words_cache: /run/determined/NAS1/public/Dataset/WISA/data/r20ft_wisa20k/reasoning/native_words
sample_sequence: /run/determined/NAS1/public/Dataset/WISA/data/r20ft_wisa20k/work/sample_sequence_20000.json
```

For a 20,000-sample fixed sequence use `max_updates: 5000` and
`gradient_accumulation: 4`.
