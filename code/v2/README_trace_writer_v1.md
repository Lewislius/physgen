# TRACE-Writer v1 implementation (`code/v2`)

This directory contains the training-free implementation of design revision
`fixed5-causal-r3`. It does not modify or train Wan, T5, or an MLLM.

## Fixed stage schedule

The five causal stages are:

```text
SETUP -> ONSET -> EVOLUTION -> COMPLETION -> TERMINAL
```

For Wan HD97 (`F=25`) the default one-token crossfade uses unequal core lengths:

```text
core:       4       3          7             4           3
stage:    SETUP   ONSET     EVOLUTION     COMPLETION   TERMINAL
boundary:       1       1               1            1
```

`crossfade_tokens=2` is also supported globally and uses core lengths
`[3,2,6,3,3]`. Width is a run-level choice, never an MLLM or per-sample output.

## Modules

- `ace_router/trace_schema.py`: typed plan representation.
- `ace_router/trace_validation.py`: diagnostic-only validator. It prints every
  issue and never chooses a fallback mode.
- `ace_router/trace_compile.py`: fixed-five compiler and convex violation weights.
- `ace_router/trace_masks.py`: 1/2-token schedules, spatial masks, f-y-x flattening.
- `ace_router/trace_runtime.py`: encoded context and gate lifecycle.
- `ace_router/trace_writer.py`: query-sliced causal cross-attention and two-level caps.
- `ace_router/trace_pipeline.py`: one-call T5 batching and runtime preparation.
- `ace_router/trace_model_adapter.py`: one-copy frozen-Wan proxy and the exact
  `lambda0 == 0` base-model fast path.
- `ace_router/controllers.py`: read-only audit hook.
- `tools/validate_trace_plan.py`: print plan issues; semantic issues return exit code 0.
- `tools/compile_trace_plan.py`: compile a structurally executable plan; impossible
  structures stop explicitly and never switch to ACE-global.

The reusable one-shot MLLM prompt is:

```text
/home/liuzhirui/Project/physGen/prompt/generate_trace_writer_v1_plan.txt
```

## Quick checks

The commands below use absolute paths and therefore do not depend on the
current working directory:

```bash
/home/liuzhirui/miniconda3/envs/moviestory/bin/python \
  /home/liuzhirui/Project/physGen/code/v2/tools/validate_trace_plan.py \
  /home/liuzhirui/Project/physGen/code/v2/examples/P01-trace-plan.json
/home/liuzhirui/miniconda3/envs/moviestory/bin/python \
  /home/liuzhirui/Project/physGen/code/v2/tools/compile_trace_plan.py \
  /home/liuzhirui/Project/physGen/code/v2/examples/P01-trace-plan.json \
  --output /tmp/P01-trace-route.json --crossfade-tokens 1
cd /home/liuzhirui/Project/physGen/code/v2 && \
  /home/liuzhirui/miniconda3/envs/moviestory/bin/python \
  -m unittest discover \
  -s /home/liuzhirui/Project/physGen/code/v2/tests -v
```

## P01-P20 inference

The two launchers default to `SAMPLE_IDS=all`, which means exactly P01 through
P20. They preflight every requested input before loading the Wan checkpoint.

```bash
# M3: I2V. For each PXX, use PXX-V2-planimg.json and the unique
# PXX-i0-1280x704.jpg/.jpeg/.png first frame.
CHECK_ONLY=1 bash /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m3.sh
bash /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m3.sh

# M4: T2V. For each PXX, use PXX-V2-plan.json. No image is opened or passed
# to T5, DiT, VAE, the scheduler, or the size policy.
CHECK_ONLY=1 bash /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m4.sh
bash /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m4.sh
```

Both launchers read plans directly from
`/home/liuzhirui/Project/physGen/code/v1/demo/PXX` by default. To run a subset
or additional seeds:

```bash
SAMPLE_IDS=P01,P08 SEEDS=42,123 bash /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m3.sh
SAMPLE_IDS=P01,P08 SEEDS=42,123 bash /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m4.sh
```

The matching Determined experiment configurations are submitted with:

```bash
det experiment create \
  /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m3.yaml \
  /home/liuzhirui/Project/physGen/code/v2
det experiment create \
  /home/liuzhirui/Project/physGen/code/v2/inference/infer_trace_writer_m4.yaml \
  /home/liuzhirui/Project/physGen/code/v2
```

Both configurations use the `ALL-IN-ONE / wan-animate` project, one
`amp-48g` slot, the shared `/mount/HOME/liuzhirui` bind mount, and the same
P01-P20 defaults as their shell launchers.

M3 defaults to `SPATIAL_MODE=spacetime`; M4 is pinned to `time`, because its
identifier-only plans contain no image-grounded routing boxes. `LAMBDA0=0`
uses the original Wan generation path and does not encode stage/violation
contexts. Every run records the selected plan/image hashes, validation report,
compiled route, diagnostics, and an upstream source/checkpoint drift snapshot.

## Integration boundary

1. Call the MLLM once outside inference and save its JSON draft. For the demo
   launchers, save `PXX-V2-planimg.json` (M3) and `PXX-V2-plan.json` (M4) under
   the matching `v1/demo/PXX` directory.
2. Load and diagnose the plan. Validation issues are visible but do not select a mode.
3. Compile the plan and call `prepare_trace_contexts`; it sends the global prompt,
   five positives, and all counterfactuals through the existing T5 in one batch.
   If Wan's encoder also requires a device argument, pass a thin closure such as
   `lambda texts: text_encoder(texts, device)`.
4. Build a `RoutingRuntime` using `spatial_mode="time"` or explicit approved masks
   with `spatial_mode="spacetime"`.
5. Wrap the already-loaded frozen Wan model once with `TraceModelProxy`, activate
   blocks 14–23 through `TraceWriterConfig.layer_gates()`, and leave the
   unconditional CFG branch unmodified.

The model integrator must preserve the existing `lambda0 == 0` base-Wan fast path.
This package deliberately does not silently repair plans, invent masks, or switch modes.
