import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from .coordinates import teacher_indices
from .views import choose_canvas, resize_video, fit_geometry
from .cache import preserve_manifest


def letterbox(frames, height, width):
    """THWC uint8 -> CTHW [-1,1], retaining the complete source canvas."""
    video = frames.permute(3, 0, 1, 2).unsqueeze(0).float() / 127.5 - 1
    result, transform = resize_video(video, height, width, upscale=False)
    return result.squeeze(0).contiguous(), transform


def frame_window(source_frames, limit, start_frame=None):
    available = source_frames if start_frame is None else source_frames - start_frame
    frames = 1 + 4 * ((min(available, limit) - 1) // 4)
    start = (source_frames - frames) // 2 if start_frame is None else start_frame
    return np.arange(start, start + frames, dtype=np.int64)


def token_content_weights(transform, size, teacher_frames):
    start = torch.arange(size // 16).float() * 16
    end = start + 16
    top, left = transform["top"], transform["left"]
    bottom, right = top + transform["resized_height"], left + transform["resized_width"]
    y = (end.clamp(max=bottom) - start.clamp(min=top)).clamp(min=0) / 16
    x = (end.clamp(max=right) - start.clamp(min=left)).clamp(min=0) / 16
    return (y[:, None] * x[None, :]).flatten().repeat(teacher_frames // 2).unsqueeze(0)


def teacher_content_weights(transform, teacher_size, teacher_frames):
    geo = fit_geometry(transform["canvas_height"], transform["canvas_width"], teacher_size, teacher_size, True)
    sy = geo["resized_height"] / transform["canvas_height"]
    sx = geo["resized_width"] / transform["canvas_width"]
    content = dict(top=geo["top"] + transform["top"] * sy,
                   left=geo["left"] + transform["left"] * sx,
                   resized_height=transform["resized_height"] * sy,
                   resized_width=transform["resized_width"] * sx)
    return token_content_weights(content, teacher_size, teacher_frames)


def read_video(path, config, start_frame=None, requested_frames=None):
    from decord import VideoReader, cpu
    reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    fps = reader.get_avg_fps()
    limit = config["max_frames"] if requested_frames is None else min(config["max_frames"], requested_frames)
    indices = frame_window(len(reader), limit, start_frame)
    source_times = reader.get_frame_timestamp(indices.tolist())[:, 0].astype(np.float64)
    times = torch.from_numpy(source_times - source_times[0]).float()
    original_shape = reader[int(indices[0])].shape
    height, width = choose_canvas(original_shape[0], original_shape[1], config["max_long_side"], config["max_area"])
    parts = []
    for offset in range(0, len(indices), config["decode_chunk_frames"]):
        raw = torch.from_numpy(reader.get_batch(indices[offset:offset + config["decode_chunk_frames"]]).asnumpy())
        part, transform = letterbox(raw, height, width)
        parts.append(part)
    video = torch.cat(parts, dim=1)
    teacher_view = teacher_indices(len(indices), config["teacher_frames"])
    metadata = dict(source_frames=len(reader), source_fps=fps, frame_indices=indices.tolist(),
                    frames=len(indices), height=height, width=width,
                    start_frame=int(indices[0]), end_frame=int(indices[-1]), duration=float(times[-1]),
                    source_time_origin=float(source_times[0]), actual_times=times.tolist(),
                    teacher_indices=teacher_view.tolist(), teacher_frames=teacher_view.numel(),
                    teacher_source_indices=indices[teacher_view.numpy()].tolist(),
                    full_source_used=bool(len(indices) == len(reader)), transform=transform)
    return video, times, metadata


def build_manifest(config):
    paths, cfg = config["paths"], config["data"]
    entries = json.loads(Path(paths["wisa_index"]).read_text())
    selected = list(range(len(entries)))[:cfg["limit"]] if cfg["limit"] else list(range(len(entries)))
    if cfg["selection_file"]:
        selected = json.loads(Path(cfg["selection_file"]).read_text())
    annotations = {entry["index"]: entry for entry in json.loads(Path(cfg["clip_annotations"]).read_text())} if cfg["clip_annotations"] else {}
    names = list(dict.fromkeys(entries[index]["video_name"] for index in selected))
    random.Random(cfg["split_seed"]).shuffle(names)
    validation = set(names[::cfg["validation_stride"]])
    records = []
    for index in selected:
        annotated = index in annotations
        records.append(dict(index=index, video_name=entries[index]["video_name"],
            caption=annotations[index]["caption"] if annotated else entries[index]["captions"],
            caption_scope="window" if annotated else "video", start_frame=annotations[index]["start_frame"] if annotated else None,
            requested_frames=annotations[index]["frames"] if annotated else None,
            split="validation" if entries[index]["video_name"] in validation else "train"))
    root = Path(paths["cache_root"])
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("vae", "text", "teacher", "views"):
        (root / directory).mkdir(exist_ok=True)
    manifest = dict(data=cfg, paths=paths, records=records,
                    teacher=dict(architecture="vjepa2_1_vitG_384", checkpoint_key="target_encoder",
                                 layer=47, channels=1664, patch_size=16, tubelet_size=2,
                                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return preserve_manifest(root, manifest)


class CachedWISA(Dataset):
    def __init__(self, cache_root, split, limit=0, *, config=None):
        if limit < 0:
            raise ValueError("Dataset sample limit must be nonnegative")
        self.root = Path(cache_root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.native_v42 = bool(config and config['data'].get('cache_format') == 'v42_f121_jepa32')
        if self.native_v42:
            self._initialize_v42(config, split, limit)
            return
        self.records = [record for record in self.manifest["records"]
                        if record["split"] == split and not record.get("excluded_reason")]
        # Select after exclusions and splitting; keep the shared cache manifest intact.
        if limit:
            self.records = self.records[:limit]
        self.null_text = torch.load(self.root / "null_text.pt", map_location="cpu", weights_only=True)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        if self.native_v42:
            return self._load_v42(record)
        filename = f"{record['index']:07d}.pt"
        sample = torch.load(self.root / "vae" / filename, map_location="cpu", weights_only=True)
        sample["text"] = torch.load(self.root / "text" / filename, map_location="cpu", weights_only=True)
        sample["target"] = torch.load(self.root / "teacher" / filename, map_location="cpu", weights_only=True)
        sample["record"] = record
        return sample

    def _initialize_v42(self, config, split, limit):
        """Read existing tensors without importing any former V4.2 model/trainer."""
        expected = config['data']
        if expected['training_mode'] != 'i2v' or expected['teacher_sampling'] != 'adjacent_pairs':
            raise ValueError('This cache bridge requires I2V and the cached adjacent-pair teacher coordinates')
        data = self.manifest['data_config']
        for key, value in dict(frames=expected['max_frames'], teacher_frames=expected['teacher_frames'],
                               teacher_size=expected['teacher_size'], limit=expected['limit'],
                               max_long_side=expected['max_long_side'], max_area=expected['max_area']).items():
            if data[key] != value:
                raise ValueError(f'Cached {key}={data[key]} differs from configured {value}')
        ready = json.loads((self.root / 'ready.json').read_text())
        identity = hashlib.sha256(json.dumps(self.manifest, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if ready['manifest_hash'] != identity:
            raise ValueError('Cache manifest differs from its completion record')
        self.file_identities = ready['files']
        records = self.manifest['records']
        train_sources = {r['source_id'] for r in records if r['split'] == 'train'}
        validation_sources = {r['source_id'] for r in records if r['split'] == 'validation'}
        if train_sources & validation_sources:
            raise ValueError('Train/validation source overlap')
        self.records = [dict(r, view=dict(frames=r['geometry']['frames'],
                        height=r['geometry']['h'], width=r['geometry']['w']))
                        for r in records if r['split'] == split]
        if len(self.records) != self.manifest['source_counts'][split]:
            raise ValueError('Cache split count differs from the manifest')
        if limit:
            self.records = self.records[:limit]
        self.null_text = torch.load(config['paths']['null_text_cache'], map_location='cpu', weights_only=True)
        if self.null_text.ndim != 2 or self.null_text.shape[-1] != 4096:
            raise ValueError('Invalid cached empty UMT5 text')

    def _load_v42(self, record):
        def load(stage):
            relative = f"{stage}/{record['id']}.pt"
            path = self.root / relative
            stat = path.stat()
            expected = self.file_identities[relative]
            if (stat.st_size, stat.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
                raise ValueError(f'Cache tensor changed since completion: {path}')
            return torch.load(path, map_location='cpu', weights_only=True)
        sample = dict(load('vae'))
        sample['text'] = load('text')
        target = load('teacher')
        if target.shape != (1, 16, 576, 1664):
            raise ValueError(f'Unexpected JEPA32 target shape: {tuple(target.shape)}')
        sample['target'] = target.flatten(1, 2)
        geo = record['geometry']
        frames, h, w = geo['frames'], geo['h'], geo['w']
        if sample['latent'].shape != (1, 48, 1 + (frames - 1) // 4, h // 16, w // 16):
            raise ValueError('Cached VAE latent geometry mismatch')
        if sample['first'].shape != (1, 48, 1, h // 16, w // 16):
            raise ValueError('I2V requires the separately encoded cached first image')
        times = torch.tensor(record['pts'], dtype=torch.float64)
        sample['times'] = (times - times[0]).float()
        transform = dict(canvas_height=h, canvas_width=w, top=geo['top'], left=geo['left'],
                         resized_height=geo['rh'], resized_width=geo['rw'])
        sample['target_weight'] = teacher_content_weights(transform, 384, 32)
        sample['video_metadata'] = dict(frames=frames, height=h, width=w, duration=record['duration'],
            source_fps=record['source_fps'], teacher_frames=32, transform=transform,
            teacher_indices=teacher_indices(frames, 32, 'adjacent_pairs').tolist())
        sample['record'] = record
        return sample


class TrainingOrder(Sampler):
    """Deterministic global order; cursor counts consumed samples, not prefetched samples."""
    def __init__(self, keys, seed, rank, world_size, cursor):
        self.keys = keys
        self.size, self.seed, self.rank, self.world_size, self.cursor = len(keys), seed, rank, world_size, cursor

    def __iter__(self):
        position = self.cursor + self.rank
        epoch = -1
        while True:
            next_epoch, offset = divmod(position, self.size)
            if epoch != next_epoch:
                epoch = next_epoch
                generator = torch.Generator().manual_seed(self.seed + epoch)
                groups = {}
                for index in torch.randperm(self.size, generator=generator).tolist():
                    groups.setdefault(self.keys[index], []).append(index)
                grouped = list(groups.values())
                order = [index for group in torch.randperm(len(grouped), generator=generator).tolist() for index in grouped[group]]
            yield order[offset]
            position += self.world_size


def move_sample(sample, device):
    return {**sample, **{key: sample[key].to(device, non_blocking=True)
                         for key in ("latent", "first", "times", "text", "target", "target_weight")}}


def training_loader(dataset, order, config, rank):
    """Workers only read CPU caches; never fork an initialized CUDA/NCCL process."""
    workers = config['workers']
    timeout = config.get('loader_timeout_seconds', 120)
    if workers < 0 or timeout < 0:
        raise ValueError('workers and loader_timeout_seconds must be nonnegative')
    return DataLoader(
        dataset, batch_size=None, sampler=order, num_workers=workers,
        pin_memory=config.get('pin_memory', True),
        generator=torch.Generator().manual_seed(config['seed'] + rank),
        multiprocessing_context='spawn' if workers > 0 else None,
        # PyTorch requires timeout=0 for the single-process iterator.
        timeout=timeout if workers > 0 else 0,
    )
