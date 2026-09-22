"""Offline same-state suffix comparisons, separate from training and production sampling."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v4.environments import require_environment
require_environment('runtime')

import torch
import torch.distributed as dist
import torch.nn.functional as F

from physgen_v4.runtime import read_config, add_external_paths, init_distributed, reduce_metrics
from physgen_v4.corrector import Corrector
from physgen_v4.backbone import load_wan, ProcessWan
from physgen_v4.coordinates import Coordinates
from physgen_v4.data import CachedWISA, move_sample
from physgen_v4.encoders import load_vae, DifferentiableDecoder
from physgen_v4.remote_teacher import RemoteVideoTeacher
from physgen_v4.losses import noisy_view, endpoint, feature_distance


@torch.no_grad()
def main():
    with ExitStack() as cleanup:
        evaluate(cleanup)


def evaluate(cleanup):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samples-per-rank", type=int, default=1)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.1, 0.35, 0.6])
    parser.add_argument("--with-output-loss", action="store_true")
    parser.add_argument("--downstream", choices=("on", "off"), default="on")
    parser.add_argument("--seed", type=int, default=83009)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = read_config(Path(args.checkpoint) / "config.yaml")
    add_external_paths(config)
    rank, world, device = init_distributed()
    corrector = Corrector(config["corrector"]).to(device)
    corrector.load_checkpoint_weights(
        torch.load(Path(args.checkpoint) / "corrector.pt", map_location="cpu", weights_only=True), config["corrector"])
    corrector.eval().requires_grad_(False)
    model = ProcessWan(load_wan(config["paths"]["wan_checkpoint"], device, True, False), corrector,
                       config["stage"] in ("B", "AB"))
    teacher = decoder = None
    if args.with_output_loss:
        teacher = cleanup.enter_context(RemoteVideoTeacher(config["paths"], device, sharded=True))
        decoder = DifferentiableDecoder(load_vae(config["paths"], device), recompute=False)
    dataset = CachedWISA(config["paths"]["cache_root"], "validation")
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    output = (root / f"rank{rank}.jsonl").open("w", buffering=1)
    for local_index in range(args.samples_per_rank):
        sample = move_sample(dataset[rank + local_index * world], device)
        generator = torch.Generator(device=device).manual_seed(args.seed + sample["record"]["index"])
        noise = torch.randn(sample["latent"].shape, generator=generator, device=device)
        coords = Coordinates.build(sample["times"], sample["video_metadata"]["height"], sample["video_metadata"]["width"],
                                   config["data"]["teacher_frames"], config["corrector"]["seconds_scale"], config["data"]["teacher_size"])
        text = sample["text"].unsqueeze(0)
        for sigma_value in args.sigmas:
            output_active = (args.with_output_loss and config["loss"]["out_sigma_min"] <= sigma_value
                             <= config["loss"]["out_sigma_max"])
            sigma = torch.tensor(sigma_value, device=device)
            noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
            sites = [label for _, family, label in model.correction_sites()
                     if family == "A" or sigma_value < config["corrector"]["handoff_sigma"]]
            for site in sites:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    full, _, info = model(noisy, sample["first"], text, sigma, coords, capture=site)
                    snapshot = info["snapshots"][site]
                    metrics = {}
                    for policy in ("on", "writer_off", "off"):
                        prediction = model.suffix(snapshot, text, sigma, coords, policy, site, args.downstream == "on")
                        metrics[f"{policy}/fm"] = F.mse_loss(prediction[:, :, 1:], (noise - sample["latent"])[:, :, 1:])
                        if policy == "on" and args.downstream == "on":
                            metrics["suffix_replay_max_error"] = (prediction - full).abs().max()
                        if output_active:
                            video = decoder(endpoint(noisy, prediction, sample["first"], sigma))
                            metrics[f"{policy}/out"] = feature_distance(teacher(video, config["data"]["teacher_frames"]),
                                                                        sample["target"], sample["target_weight"])
                    for loss_name in ["fm"] + (["out"] if output_active else []):
                        metrics[f"writer_gain/{loss_name}"] = metrics[f"writer_off/{loss_name}"] - metrics[f"on/{loss_name}"]
                        metrics[f"corrector_gain/{loss_name}"] = metrics[f"off/{loss_name}"] - metrics[f"on/{loss_name}"]
                row = dict(index=sample["record"]["index"], sigma=sigma_value, site=site, downstream=args.downstream,
                           metrics={key: float(value) for key, value in metrics.items()})
                output.write(json.dumps(row) + "\n")
                reduced = reduce_metrics(metrics, device)
                if rank == 0:
                    print(json.dumps(dict(local_index=local_index, sigma=sigma_value, site=site, mean_metrics=reduced)), flush=True)
    output.close()
    if rank == 0:
        (root / "protocol.json").write_text(json.dumps(dict(arguments=vars(args), training_config=config,
            evidence_scope="paired legal-noise endpoints; not full-sampling physical correctness"), ensure_ascii=False, indent=2))
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
