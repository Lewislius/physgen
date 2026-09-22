# Wan/ACE-Router V1: M0-M4 inference

All five launchers use `/home/liuzhirui/miniconda3/envs/moviestory` and load the
local Wan2.2-TI2V-5B checkpoint read-only. All generated files, caches, logs,
and temporary data stay under `code/v1`.

## Methods

- `M0`: pure Wan2.2-TI2V-5B T2V with the original short prompt. No image,
  cplus, or cminus tensor enters the model. It defaults to a fixed landscape
  output and does not read an image.
- `M1`: pure Wan2.2-TI2V-5B I2V with the 1280x704 first frame and original
  short prompt. It calls Wan's original `generate` path and does not encode or
  inject cplus/cminus.
- `M2`: Wan2.2-TI2V-5B I2V with the same first frame and a cplus-only
  cross-attention residual. The explicit residual is
  `semantic + scale * (positive - semantic)`; cminus is not encoded or used.
- `M3`: I2V, metadata-compiled semantic prompt, and capped
  `positive - counterfactual` residual. The preferred input is
  `Pxx-i0-1280x704.png`; older `Pxx-i0-wan.png` and `Pxx-i0.png`/JPG files are
  compatibility fallbacks. The exact
  image entering Wan is saved and hashed as `reference_input.png`.
- `M4`: T2V, a fuller metadata-compiled semantic prompt, and a more conservative
  capped `positive - counterfactual` residual. It defaults to fixed landscape
  output; no image tensor enters T5, DiT, VAE, or scheduler.

M1, M2, and M3 use exactly the same first-frame preparation. New M3 defaults
to `lambda0=0.10`, a 2% relative residual cap, `mid_b`, and step gates
`0.5 / 1.0 / 0.3 / 0.1`. New M4 defaults to `lambda0=0.08`, a 1.5% cap,
`mid_b`, and `0.25 / 0.8 / 0.3 / 0.1`. Legacy settings remain available as
CLI overrides.
All methods default to 1280x704 output and 97 generated frames at 24 fps.
No negative prompt is used; CFG's unconditional branch encodes empty text.
For every method, cplus/cminus are now relation-only contexts: the original
prompt is never prefixed to either branch. M3/M4 enable `compiled`
conditioning in their launchers; the original prompt and metadata enhancement
appear only in the semantic branch.

## Local preflight

Run schema, path, upstream-hash, image, and resolved-size checks without
loading the 5B model:

```bash
cd /home/liuzhirui/Project/physGen/code/v1
CHECK_ONLY=1 SAMPLE_IDS=P01 bash inference/infer_ace_router_m0.sh
CHECK_ONLY=1 SAMPLE_IDS=P01 bash inference/infer_ace_router_m1.sh
CHECK_ONLY=1 SAMPLE_IDS=P01 bash inference/infer_ace_router_m2.sh
CHECK_ONLY=1 SAMPLE_IDS=P01 bash inference/infer_ace_router_m3.sh
CHECK_ONLY=1 SAMPLE_IDS=P01 bash inference/infer_ace_router_m4.sh
```

Parse only the launcher arguments:

```bash
PARSE_ONLY=1 bash inference/infer_ace_router_m0.sh
PARSE_ONLY=1 bash inference/infer_ace_router_m1.sh
PARSE_ONLY=1 bash inference/infer_ace_router_m2.sh
PARSE_ONLY=1 bash inference/infer_ace_router_m3.sh
PARSE_ONLY=1 bash inference/infer_ace_router_m4.sh
```

Important overrides are comma-separated `SAMPLE_IDS=P01,P02` and
`SEEDS=7,42`, plus `LAMBDA0`, `RESIDUAL_CAP_RATIO`, `LAYER_PRESET`,
`STEP_PRESET`, `CONDITIONING_MODE`, `MAX_AREA`, and `SAMPLING_STEPS`.
ACE strength and schedule options apply only to M2/M3/M4.
For a formal I2V pilot, set
`REQUIRE_SELECTION_MANIFEST=1` after adding the audited
`Pxx-i0-selection.json` files.

## Determined submission

From `code/v1`, submit an experiment with this directory as the context:

```bash
det experiment create inference/infer_ace_router_m0.yaml .
det experiment create inference/infer_ace_router_m1.yaml .
det experiment create inference/infer_ace_router_m2.yaml .
det experiment create inference/infer_ace_router_m3.yaml .
det experiment create inference/infer_ace_router_m4.yaml .
```

The YAML defaults run all P01-P20 samples with seed 42 on one 48 GB GPU.
For an initial P01 smoke run, change `SAMPLE_IDS=all` to `SAMPLE_IDS=P01` in
the selected YAML, or submit a temporary copy with that environment override.

Outputs are written to:

```text
outputs/ace_router/<run_id>/<sample_id>/<method>/seed_<seed>/
```

Each successful sample contains `video.mp4`, `manifest.json`,
`config.resolved.json`, `contexts.json`, `conditioning.compiled.json`,
`diagnostics.jsonl`, and `reference_image.json`. M1/M2/M3 also contain the exact model input as
`reference_input.png`. `contexts.json.used_by_model` records which text
branches actually entered that method; for M0/M1, cplus/cminus are both
false, and for M2 cminus is false. A failed sample contains `failure.json`
and no final video.

## Initial-V1 HD97 ablation

The two `*_initial_v1_hd97` launchers reproduce the first M3/M4 ACE setup
while changing the generated size and duration to 1280x704 and 97 frames.
They retain `lambda0=0.5`, `mvp_mid16`, the legacy step schedule, origin-
prefixed cplus/cminus contexts, and Wan's built-in CFG negative prompt, with no
relative residual cap. M3 selects `Pxx-i0-1280x704.png`; M4 remains pure T2V.
Their results use the separate `outputs/ace_router_initial_v1_hd97` root.
