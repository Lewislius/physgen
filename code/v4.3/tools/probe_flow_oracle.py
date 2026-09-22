"""Bounded real-Wan verification; reads one cached GT clip, never saves trained weights."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v43.environments import require_environment
require_environment("runtime")

import torch

from physgen_v43.runtime import ROOT, read_config, add_external_paths, seed_all
from physgen_v43.config import validate_config
from physgen_v43.corrector import Corrector
from physgen_v43.backbone import ProcessWan, load_wan
from physgen_v43.data import CachedWISA, move_sample
from physgen_v43.losses import noisy_view, flow_target, flow_mse, training_loss
from physgen_v43.oracle import build_oracle_targets
from train.train import geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/flow_oracle.yaml"))
    parser.add_argument("--checkpoint", help="Optional compatible V4.3 checkpoint; otherwise verifies zero-initialized writers")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation"), default="train")
    parser.add_argument("--sigmas", nargs="+", type=float, default=[.6])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--report", default=str(ROOT / "analysis/checks/real_wan_probe.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("A CUDA GPU is required for the real Wan probe; CPU unit checks do not substitute for it")
    if any(not 0 < s < 1 for s in args.sigmas):
        parser.error("Probe sigmas must lie strictly between zero and one")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    config = validate_config(read_config(args.config))
    add_external_paths(config)
    seed_all(config["train"]["seed"])
    started = time.perf_counter()
    dataset = CachedWISA(config["paths"]["cache_root"], args.split, config=config)
    sample = move_sample(dataset[args.sample_index], device)
    coords = geometry(sample, config)
    corrector = Corrector(config["corrector"]).to(device)
    if args.checkpoint:
        source = Path(args.checkpoint)
        corrector.load_checkpoint_weights(torch.load(source / "corrector.pt", map_location="cpu", weights_only=True),
                                           read_config(source / "config.yaml")["corrector"])
    wan = load_wan(config["paths"]["wan_checkpoint"], device, sharded=False, recompute=True)
    model = ProcessWan(wan, corrector)
    labels = [label for _, _, label in model.correction_sites()]
    text = sample["text"].unsqueeze(0)
    noise = torch.randn_like(sample["latent"], dtype=torch.float32)
    target = flow_target(sample["latent"], noise)
    report = dict(status="running", config=str(Path(args.config).resolve()), checkpoint=args.checkpoint,
                  gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  sample_index=args.sample_index, source_index=sample["record"]["index"],
                  video_metadata=sample["video_metadata"], latent_shape=list(sample["latent"].shape),
                  target_shape=list(sample["target"].shape), parameter_count=sum(p.numel() for p in corrector.parameters()),
                  actual_state_width=config["corrector"]["hidden_width"], teacher_width=config["corrector"]["jepa_width"],
                  optimizer_update=False, cases=[])
    torch.cuda.reset_peak_memory_stats(device)
    for value in args.sigmas:
        sigma = torch.tensor(value, device=device)
        noisy = noisy_view(sample["latent"], sample["first"], noise, sigma)
        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
            with torch.no_grad():
                prediction, process, info = model(noisy, sample["first"], text, sigma, coords, capture=labels)
                current_fm = float(flow_mse(prediction, target))
                shapes = dict(state=list(process.shape), hidden={k: list(s.before.shape) for k, s in info["snapshots"].items()},
                              jepa={k: list(v.shape) for k, v in info["states"].items()})
                replay_errors = {}
                for label, snapshot in info["snapshots"].items():
                    replay = model.suffix(snapshot, snapshot.before + snapshot.delta, text, sigma, coords)
                    replay_errors[label] = float((replay - prediction).abs().max())
                    torch.testing.assert_close(replay, prediction, rtol=1e-4, atol=1e-5)
                    del replay
                if args.checkpoint is None:
                    base, _, _ = model(noisy, sample["first"], text, sigma, coords, disable=("A",))
                    torch.testing.assert_close(base, prediction, rtol=0, atol=0)
                    del base
                snapshots = info["snapshots"]
                del prediction, process, info
            targets, metrics = build_oracle_targets(model, snapshots, labels, text, sigma, coords, target,
                                                    config["oracle"], first_known=True)
            assert all(p.grad is None for p in model.parameters()), "Oracle contaminated model .grad"
            del snapshots
            prediction, process, info = model(noisy, sample["first"], text, sigma, coords)
            loss, loss_metrics, _ = training_loss(prediction, sample["latent"], noise, sample["first"], sigma,
                info, sample["target"], sample["target_weight"], targets, 200, config)
        loss.backward()
        bad = [name for name, p in corrector.named_parameters()
               if p.grad is None or not bool(torch.isfinite(p.grad).all())]
        if bad:
            raise AssertionError(f"Missing/nonfinite corrector gradients: {bad}")
        assert all(p.grad is None and not p.requires_grad for p in wan.parameters())
        case = dict(sigma=value, current_fm=current_fm, suffix_replay_max_errors=replay_errors,
                    shapes=shapes, oracle=metrics,
                    losses={k: float(v) for k, v in loss_metrics.items() if k.startswith("loss/")})
        report["cases"].append(case)
        print(json.dumps(case, ensure_ascii=False), flush=True)
        corrector.zero_grad(set_to_none=True)
        del loss, prediction, process, info, targets
    torch.cuda.synchronize(device)
    report.update(status="runtime_and_gradient_checks_passed", seconds=time.perf_counter() - started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                  oracle_any_improvement=any(c["oracle"]["oracle/acceptance"] > 0 for c in report["cases"]),
                  full_generation_quality_evaluated=False)
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
