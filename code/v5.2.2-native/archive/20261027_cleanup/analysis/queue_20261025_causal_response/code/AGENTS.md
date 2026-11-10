# V5.2 scope and persistent design constraints

- Model general visible processes: gas expansion/diffusion, waves, liquids, deformation, appearance/disappearance, and rigid motion. Do not narrow this to actions or tracked objects.
- Do not introduce optical-flow, point-tracking, displacement, physical-speed targets, D/S/U classes, fixed-camera filtering, motion transport, or trajectory contracts into preparation, training, routing, or inference.
- Keep a continuous local visible-change intensity and train its effect on generation. RGB structural change is weak supervision, not physical velocity or ground truth physics. VAE/JEPA/history inform the predictor; JEPA feature magnitude alone is not a calibrated motion measure.
- Low intensity must not disable generation or hard-copy the first image into future frames. Preserve soft dynamic/history/native paths and complete signed state-change supervision.
- FM velocity is a derivative along the denoising path, not motion velocity in video time; retain that distinction in code and documentation.
- Future RGB/JEPA/change targets may supervise training and explicit validation oracles, never ordinary inference conditions.
- Verify diffuse/deforming/emerging/transient changes, target leakage, real generation gradients and output sensitivity. Do not claim improved full-model video quality from unit tests or fixed-noise fitting alone.
- Preserve V5.1 sources and existing caches/checkpoints/logs. Changes in target definitions or model parameters require a new checkpoint revision.

These constraints restate the user's requirements and the existing V5.1 design at `../v5.1/markdown/V5.1_因果规划与局部变化方案_20261006.md`, especially the section beginning “位移不是通用证据”.

- This revision uses only the existing wisa_4000 caches; never modify them or sibling versions.
- No independent M organizer, S future predictor, or auxiliary future-state rollout. Native Wan carries the evolving state.
- Supervised video views and generated samples are training computations, not additional datasets.
- Preserve the user's central design: concentrated physical-causal C, a verifier whose feedback reaches the latent, and effective state-dependent injection. Repair invalid supervision and gradient credit; do not remove the verifier or relabel a generic feature loss as a complete physical verifier.
- VideoPhy2 is one external reference, not the definition of success. Establish visible process quality, late-stage consistency, selective condition effects and repeatability across samples and seeds.
- Preserve the user's existing software environment and Determined setup. Use the existing `det 0.32.1` submission tool; a newer-master warning is not a request to upgrade. Do not install, upgrade, downgrade, or reconfigure Determined, conda environments, container images, or cluster services.
- Reuse the previously successful Determined submission environment. Do not put Python isolation variables such as `PYTHONNOUSERSITE` in the experiment-wide environment; the existing training interpreter's `-I -B` flags already isolate the project process after container bootstrap.
- Submit single-slot GPU jobs (`slots_per_trial: 1`). Do not reserve a second GPU to select a preferred device; solve training memory use within the allocated single GPU.
