import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from physgen_v4.environments import require_environment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pass", dest="encoding_pass",
                        choices=("index", "vae", "text", "teacher", "filter", "finalize", "audit"), required=True)
    parser.add_argument("--overwrite", action="store_true", help="Re-encode this pass even when its cache is valid")
    args = parser.parse_args()
    require_environment('teacher' if args.encoding_pass == 'teacher' else 'runtime')
    from torch.distributed.elastic.multiprocessing.errors import record
    record(prepare)(args)


def prepare(args):
    require_environment('teacher' if args.encoding_pass == 'teacher' else 'runtime')
    import torch
    from physgen_v4.runtime import read_config, add_external_paths
    from physgen_v4.cache import (audit_cache, cache_error, check_manifest_config, exclude_short_views, finalize_cache,
                                  read_view, save_json, save_tensor, validate_value)
    config = read_config(args.config)
    add_external_paths(config)
    from physgen_v4.data import build_manifest, read_video, teacher_content_weights
    if args.encoding_pass == "index":
        manifest = build_manifest(config)
        excluded = exclude_short_views(config["paths"]["cache_root"], manifest)
        print(json.dumps({"selected_samples": len(manifest["records"]),
                          "samples": len(manifest["records"]) - len(excluded), "excluded": excluded,
                          "cache_root": config["paths"]["cache_root"]}), flush=True)
        return
    root = Path(config["paths"]["cache_root"])
    manifest = json.loads((root / "manifest.json").read_text())
    check_manifest_config(manifest, config)
    if args.encoding_pass == "filter":
        excluded = exclude_short_views(root, manifest, require_all_views=True)
        print(json.dumps(dict(event="filter", selected_samples=len(manifest["records"]),
                              samples=len(manifest["records"]) - len(excluded), excluded=excluded)), flush=True)
        return
    if args.encoding_pass in ("finalize", "audit"):
        if args.encoding_pass == "finalize":
            summary = finalize_cache(root, manifest)
        else:
            summary = audit_cache(root, manifest)
            summary["finalized"] = all(record.get("view") == read_view(root, record)
                                       for record in manifest["records"]) if summary["complete"] else False
        print(json.dumps(dict(event=args.encoding_pass, **summary), ensure_ascii=False), flush=True)
        if not summary["complete"] or (args.encoding_pass == "audit" and not summary["finalized"]):
            raise SystemExit("Cache is incomplete or not finalized; rerun with PREPARE=1.")
        return
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    records = [r for r in manifest["records"] if not r.get("excluded_reason")][rank::world]
    cfg = config["data"]
    pending = []
    for sample_record in records:
        error = cache_error(root, args.encoding_pass, sample_record, cfg)
        if args.overwrite or error:
            pending.append(sample_record)
            path = root / args.encoding_pass / f"{sample_record['index']:07d}.pt"
            if error and path.exists():
                print(json.dumps(dict(event="invalid_cache", pass_name=args.encoding_pass,
                                      rank=rank, index=sample_record["index"], error=error)), flush=True)
    needs_null = args.encoding_pass == "text" and rank == 0 and (
        args.overwrite or cache_error(root, "null_text") is not None)
    print(json.dumps(dict(event="cache_plan", pass_name=args.encoding_pass, rank=rank,
                          python=sys.executable, torch=torch.__version__, torch_cuda=torch.version.cuda,
                          assigned=len(records), reused=len(records) - len(pending),
                          pending=len(pending), null_text_pending=needs_null)), flush=True)
    # A fully cached pass does not load its encoder or initialize CUDA in the workers.
    if not pending and not needs_null:
        return
    from physgen_v4.encoders import load_vae, load_text_encoder, VideoTeacher
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    if args.encoding_pass == "vae":
        encoder = load_vae(config["paths"], device)
    elif args.encoding_pass == "text":
        encoder = load_text_encoder(config["paths"], device)
        if needs_null:
            with torch.no_grad():
                null = encoder([""], device)[0].cpu().to(torch.bfloat16)
            validate_value("text", null, None, cfg)
            save_tensor(null, root / "null_text.pt")
    else:
        encoder = VideoTeacher(config["paths"], device)
    for count, record in enumerate(pending, 1):
        metadata = None
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            if args.encoding_pass == "text":
                value = encoder([record["caption"]], device)[0].cpu().to(torch.bfloat16)
            else:
                video, times, metadata = read_video(Path(config["paths"]["wisa_videos"]) / record["video_name"],
                                                   cfg, record["start_frame"], record["requested_frames"])
                if metadata["frames"] < 5:
                    if args.encoding_pass != "vae":
                        raise ValueError("Short clips must be excluded by the filter pass before teacher encoding")
                    save_json(metadata, root / "views" / f"{record['index']:07d}.json")
                    print(json.dumps(dict(event="excluded_short_clip", index=record["index"],
                                          source_frames=metadata["source_frames"], frames=metadata["frames"])), flush=True)
                    continue
                video = video.unsqueeze(0).to(device)
                if args.encoding_pass == "vae":
                    latent = encoder.model.encode(video, encoder.scale).cpu().to(torch.bfloat16)
                    first = encoder.model.encode(video[:, :, :1], encoder.scale).cpu().to(torch.bfloat16)
                    value = dict(latent=latent, first=first, times=times, video_metadata=metadata,
                                 target_weight=teacher_content_weights(metadata["transform"], cfg["teacher_size"], metadata["teacher_frames"]))
                else:
                    if metadata != read_view(root, record):
                        raise ValueError(f"Video view changed for index {record['index']}; choose a new cache_root")
                    value = encoder(video, cfg["teacher_frames"]).cpu().to(torch.bfloat16)
            validate_value(args.encoding_pass, value, metadata, cfg)
            if args.encoding_pass == "vae":
                save_json(metadata, root / "views" / f"{record['index']:07d}.json")
            save_tensor(value, root / args.encoding_pass / f"{record['index']:07d}.pt")
        if count == 1 or count % 25 == 0 or count == len(pending):
            print(json.dumps({"pass": args.encoding_pass, "rank": rank, "encoded": count,
                              "completed": len(records) - len(pending) + count,
                              "reused": len(records) - len(pending),
                              "assigned": len(records), "index": record["index"],
                              "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30}), flush=True)


if __name__ == "__main__":
    main()
